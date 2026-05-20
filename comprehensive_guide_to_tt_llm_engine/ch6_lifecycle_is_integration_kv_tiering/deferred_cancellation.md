# Deferred Cancellation

[< Multi-Turn Continue](multi_turn_continue.md) | [Context Exhaustion >](context_exhaustion.md)

## Why Deferred?

Cancellation in tt-llm-engine cannot be instantaneous. The system is a three-thread pipeline (`api_thread`, `writer_thread`, `reader_thread`) with tokens in flight through device hardware at any given moment. An immediate cancellation -- one that frees the slot and resets the KV cache the moment the `CANCEL` request arrives -- would corrupt state in at least two ways:

1. **In-flight device operations.** The Writer may have already submitted a prefill or decode batch to the device that includes the cancelled user's slot. Freeing the slot (and its KV cache region) while the device is still reading from it would cause undefined behavior.

2. **Pending output tokens.** The Reader may be about to read results from the device that include tokens for the cancelled user. If the slot is freed and reallocated, those stale tokens could be attributed to the new user.

The solution is a **deferred, four-phase cancellation protocol** that cooperatively drains all in-flight work before performing the actual cleanup.

## The Four Phases

```
Phase 1: API Mark          Phase 2: Writer Skip       Phase 3: Reader Drain      Phase 4: Finalize
--------------------       --------------------       ---------------------      ------------------
Scheduler sets             Writer sees cancel_pending  Reader sees cancel_pending  maybe_finalize_cleanup
cancel_pending bit         -> skips user in next batch -> discards output tokens   checks:
                           -> releases pending_count   -> clears in_flight flags   - cancel_pending set
Stashes cancel_request_id  -> calls maybe_finalize     -> calls maybe_finalize     - in_flight_count == 0
Advances generation                                                                - pending_count == 0
Removes from                                                                       - state != INACTIVE
prefill_queue                                                                      -> CAS state to INACTIVE
                                                                                   -> reset_kv, free slot
                                                                                   -> push cancel ack
```

### Phase 1: API Mark (Scheduler Thread)

When the scheduler processes a `CANCEL` request:

```cpp
case RequestType::CANCEL: {
    uint32_t uid = req.slot_id;
    cancel_request_id[uid] = req.request_id;   // stash BEFORE mark
    cancel_pending.mark(uid);                    // release store
    decode_staging.advance_generation(uid);
    prefill_queue.remove(uid);
    // ... check for immediate cleanup ...
}
```

Actions taken in order:
1. **Stash the request ID.** The `cancel_request_id[uid]` is stored before the cancel bit is set. This ordering, combined with the release-acquire semantics on `cancel_pending`, ensures that any thread that later reads `cancel_pending.is_set(uid)` is guaranteed to see the correct `cancel_request_id`.
2. **Mark the cancel bit.** `cancel_pending.mark(uid)` uses `fetch_or` with release semantics (see Ch2 for `CancelBitmap`). This is the signal that all other threads will check.
3. **Advance the generation counter.** `decode_staging.advance_generation(uid)` increments the generation, invalidating any tokens currently staged for this user.
4. **Remove from prefill queue.** If the user is queued for prefill but hasn't started yet, remove it immediately.

After these actions, the scheduler checks whether immediate cleanup is possible (see the INACTIVE special case below).

### Phase 2: Writer Skip (Writer Thread)

When the Writer constructs its next batch, it checks each candidate user against the cancel bitmap:

**Decode path (popping from DecodeStaging):**

```cpp
DecodeStagingEntry entry;
if (decode_staging.try_pop(entry)) {
    uint32_t uid = entry.slot_id;
    if (cancel_pending.is_set(uid)) {
        decode_staging.release(uid);
        maybe_finalize_cleanup(uid);
        continue;  // Do not inject into pipeline
    }
}
```

Note: `try_pop` takes an output reference parameter and returns `bool` (true if an entry was available). The Writer checks only `cancel_pending.is_set(uid)` -- it does not filter staging entries by generation counter. The generation counter is advanced by Phase 1 and used by the IS for output filtering, but the Writer relies solely on the cancel bit.

**Prefill path:** If the Writer is mid-prefill for a cancelled user, it detects `cancel_pending.is_set(uid)` at the top of the prefill injection loop and stops injecting. The user has already been removed from the PrefillQueue by Phase 1.

### Phase 3: Reader Drain (Reader Thread)

When the Reader processes device output, it checks for cancelled users:

```cpp
auto result = pipeline->read_result();
uint32_t uid = result.slot_id;

if (cancel_pending.is_set(uid)) {
    in_flight_count[uid].fetch_sub(1, std::memory_order_relaxed);
    prefill_in_flight[uid].store(0, std::memory_order_relaxed);
    post_complete_in_flight[uid].store(0, std::memory_order_relaxed);
    maybe_finalize_cleanup(uid);
    continue;  // Do not stage, do not emit output
}
```

The Reader decrements `in_flight_count` (with `relaxed` ordering) for every result it consumes, whether the user is cancelled or not. When the user *is* cancelled, the Reader additionally zeroes the `prefill_in_flight` and `post_complete_in_flight` counters and calls `maybe_finalize_cleanup`.

> **The `ReaderClaim` RAII pattern:** The `ReaderClaim` RAII object (defined in `decode_scheduler.cpp` as a nested struct inside `DecodeScheduler::Impl`) holds a `pending_count` increment for the duration of the Reader's iteration over a given slot's result. If a cancel arrives mid-iteration, the `ReaderClaim` destructor calls `maybe_finalize_cleanup` when it releases. This prevents a race where the Writer drains the FIFO to zero pending while the Reader is still processing the last result.

### Phase 4: Finalize (`maybe_finalize_cleanup`)

This is the convergence point. It is called by the Writer, Reader, and Scheduler -- whichever thread happens to be the last one to release its hold on the user.

```cpp
void maybe_finalize_cleanup(uint32_t uid) {
    if (!cancel_pending.is_set(uid)) return;
    if (user_table.in_flight_count[uid].load(acquire) != 0) return;
    if (decode_staging.pending_count[uid].load(acquire) != 0) return;

    uint32_t request_id = cancel_request_id[uid];

    UserState expected = user_table.state[uid].load(acquire);
    if (expected == UserState::INACTIVE) return;
    if (!user_table.state[uid].compare_exchange_strong(
            expected, UserState::INACTIVE, acq_rel, relaxed))
        return;

    pipeline_->reset_kv(uid);
    spec_state.reset(uid);
    free_ids.free(uid);
    push_cancel_ack(request_id, uid);
}
```

The function performs five checks before cleanup:

| Check | Purpose |
|-------|---------|
| `cancel_pending.is_set(uid)` | Guard: only run for users with active cancel |
| `in_flight_count == 0` | No tokens in the device pipeline |
| `pending_count == 0` | No tokens staged in `DecodeStaging` and no active Reader iterations |
| `state != INACTIVE` | Not already cleaned up by another thread |
| `compare_exchange_strong` succeeds | Atomic ownership claim -- exactly one thread wins |

**Why CAS and not a simple store?** Multiple threads may pass all three counter gates simultaneously. If the Writer releases the last `pending_count` entry at the same instant the Reader releases the last `in_flight_count`, both threads call `maybe_finalize_cleanup` and both pass the counter checks. The CAS ensures exactly one thread transitions the state to `INACTIVE` and performs cleanup.

**Why snapshot `cancel_request_id` before the CAS?** After the CAS succeeds and `free_ids.free(uid)` executes, the slot may be immediately reallocated by another API thread. The new session would overwrite `cancel_request_id[uid]`. The snapshot taken before the CAS captures the correct value.

### Cleanup Actions

Once a thread wins the CAS:

1. **`pipeline_->reset_kv(uid)`** -- Clears the KV cache region for this slot on the device (see Ch5).
2. **`spec_state.reset(uid)`** -- Resets speculative decode state (see Ch4).
3. **`free_ids.free(uid)`** -- Returns the slot to `FreeIdPool`, making it available for new `ALLOCATE` requests.
4. **`push_cancel_ack(request_id, uid)`** -- Sends a `SchedulerResponse` to the `response_queue` confirming cancellation.

## The Three Gates in Detail

### Gate 1: `cancel_pending.is_set(uid)`

The `CancelBitmap` is a multi-word atomic bitmap. `mark()` uses `fetch_or` with release semantics. `is_set()` uses `load` with acquire semantics. This release-acquire pair ensures that any thread reading the bitmap also sees all stores made by the API thread before the mark -- specifically, `cancel_request_id[uid]`.

`clear()` uses `fetch_and` with `memory_order_release` ordering. Although it is only called after the cleanup CAS has been won, the release ensures that the clearing of the cancel bit is visible to other threads observing the slot's state.

### Gate 2: `in_flight_count[uid] == 0`

This counter is incremented by the Writer *before* calling `pipeline_->inject()`:

```cpp
// Writer: before inject
in_flight_count[uid].fetch_add(do_spec ? 2 : 1, std::memory_order_release);
```

It is decremented by the Reader on *every* result for that slot:

```cpp
// Reader: on read_result
in_flight_count[uid].fetch_sub(1, std::memory_order_relaxed);
```

> **Invariant:** `in_flight_count` is incremented (with `release`) *before* `inject()` and decremented (with `relaxed`) *after* `read_result()`. This means the counter is always an upper bound on the true in-flight count. It can momentarily overestimate (between the fetch_add and the actual inject), but it never underestimates. For cancellation safety, overestimation is conservative: it delays cleanup but never causes premature cleanup. The Writer uses `release` on increment so that all preceding stores (batch setup, pipeline state) are visible before the counter increases.

### Gate 3: `pending_count[uid] == 0`

This counter serves a dual purpose:

1. **FIFO entry tracking.** `pending_count` is incremented by `stage()` (called by the Reader for decode loopback, or by the api_thread for disaggregated continue) *before* pushing an entry into the DecodeStaging FIFO. It is decremented by `release()` (called by the Writer after popping from the FIFO, and by the `ReaderClaim` destructor).

2. **Reader iteration tracking.** The `ReaderClaim` RAII object increments `pending_count` at construction (when the Reader begins processing a result for a slot) and decrements it at destruction (when the Reader is done with that result).

The dual purpose closes a subtle race: suppose the Reader has popped a result and is about to stage a new entry, but has not yet called `staging.stage()`. At this instant, `pending_count` would be zero if only FIFO entries were tracked. If the Writer simultaneously drained the FIFO to zero and called `maybe_finalize_cleanup`, the cleanup would proceed while the Reader still holds a reference to the slot. The `ReaderClaim` bump prevents this.

## Cancel on INACTIVE User (Immediate Path)

There is one case where cancellation does not need to be deferred: when the user is in `INACTIVE` state (allocated but not yet submitted) with no in-flight or pending work.

```cpp
UserState st = user_table.state[uid].load(std::memory_order_acquire);
if (st == UserState::INACTIVE && in_flight_count == 0 && pending_count == 0) {
    cancel_pending.clear(uid);
    pipeline_->reset_kv(uid);
    free_ids.free(uid);
    push_cancel_ack(...);
} else {
    maybe_finalize_cleanup(uid);
}
```

This fast path is important for the common pattern of:
1. IS allocates a slot optimistically.
2. Client disconnects before submitting a prompt.
3. IS sends CANCEL to reclaim the slot.

## Interaction with Speculative Decode

When a speculative-decode user is cancelled, there may be both BASE and SPEC tokens in flight. The `in_flight_count` was incremented by 2 for each spec pair (`delta = do_spec ? 2 : 1`), so it will not reach zero until both results have been consumed by the Reader. The Reader's cancel path discards both without attempting acceptance checking or result classification (see Ch4). The `spec_state.reset(uid)` call in cleanup clears any stashed `pending_complete` or `unverified_token` state.

## Timing Diagram

```
Time ------------------------------------------------------------------->

Scheduler:  CANCEL--mark--advance_gen--remove_prefill--maybe_finalize(fail: in_flight>0)
                |
Writer:         |                        +---------------------+
                |                        | skip user           |
                |                        | release pending     |
                |                        | maybe_finalize      |
                |                        | (fail: in_flight>0) |
                |                        +---------------------+
Reader:         |                                               +------------------+
                |                                               | drain output     |
                |                                               | dec in_flight    |
                |                                               | maybe_finalize   |
                |                                               | (SUCCESS: all 0) |
                |                                               | -> reset_kv      |
                |                                               | -> free slot     |
                |                                               | -> cancel_ack    |
                |                                               +------------------+
```

In this example, the Reader is the last thread to release its hold on the user, so it wins the CAS in `maybe_finalize_cleanup` and performs the actual cleanup. In other scenarios, the Writer or even the Scheduler could be the one to finalize. In practice, the Reader usually wins because the last in-flight token exiting the device pipeline is typically the last event to drain.

## Thread-Safety Summary

| Thread | Cancel Action | Calls `maybe_finalize_cleanup`? |
|--------|--------------|--------------------------------|
| Scheduler | Stores `cancel_request_id`, marks bitmap, advances generation, removes from PrefillQueue | Yes (but rarely wins) |
| Writer | Discards stale staging entries, releases `pending_count` | Yes (wins when last pending drains) |
| Reader | Decrements `in_flight_count`, zeroes prefill/post-complete counters | Yes (wins when last in-flight drains) |

## Pipelined Teardown

The deferred protocol is designed to be **non-blocking for the pipeline.** While a cancel is pending, the Writer continues processing other users, the Reader continues processing other users' output, and the Scheduler continues processing requests. No thread ever blocks waiting for cancellation to complete.

In production workloads (e.g., the DeepSeek inference runner), cancellation is *pipelined* with allocation: a batch of CANCELs is issued fire-and-forget, and the next batch's allocator drains cancel-ack responses as part of its allocation loop, hiding the deferred-cleanup latency.

## Correctness Properties

The deferred cancellation protocol guarantees:

1. **No use-after-free.** The KV cache is only reset after all in-flight device operations are drained ($\text{in\_flight\_count} = 0$) and all staged tokens are consumed or discarded ($\text{pending\_count} = 0$).

2. **No stale token emission.** The generation counter is advanced in Phase 1, before any other action. Even if the Reader emits tokens before observing the cancel bit, those tokens carry the old generation and will be discarded by the IS.

3. **Exactly-once cleanup.** The CAS on `user_table.state` ensures that even though `maybe_finalize_cleanup` is called from multiple threads, exactly one thread performs the cleanup sequence.

4. **Exactly-once cancel acknowledgment.** The cancel ack is pushed inside the CAS-protected cleanup path, so it is sent exactly once to the `response_queue`.

5. **No deadlock.** No thread ever blocks waiting for another thread to complete cancellation. All checks are non-blocking loads.

---

*[< Multi-Turn Continue](multi_turn_continue.md) | [Context Exhaustion >](context_exhaustion.md)*
