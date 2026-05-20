# Decode Staging

`DecodeStaging` is the Reader-to-Writer handoff mechanism for decode tokens. When the Reader processes a pipeline result and determines that a token needs to be re-injected (decode loopback, speculative pair, EOS write-back), it stages the token here. The Writer pops entries from the FIFO and injects them into the pipeline. A per-slot reference count (`pending_count`) prevents premature slot cleanup during the gap between staging and injection.

Source: `src/scheduler/decode/decode_staging.hpp`

## DecodeStagingEntry

```cpp
struct DecodeStagingEntry {
    uint32_t slot_id      = INVALID_SLOT;
    uint32_t token_id     = EMPTY_TOKEN;
    uint32_t position     = 0;
    uint32_t spec_token_id = EMPTY_TOKEN;
    uint32_t spec_position = 0;
    uint32_t generation   = 0;
    bool     skip_spec    = false;
};
```

Each entry carries everything the Writer needs to construct one or two `InjectDescriptor`s:

| Field | Purpose |
|---|---|
| `slot_id` | Which user this entry belongs to |
| `token_id` | BASE token ID to inject (the sampled actual token) |
| `position` | Device KV position for the BASE token |
| `spec_token_id` | SPEC (predicted) token ID for speculative decode |
| `spec_position` | Device KV position for the SPEC token |
| `generation` | Generation counter at staging time (stale-entry detection) |
| `skip_spec` | When true, Writer injects only the BASE token even if `spec_decode_enabled`. Used for post-completion EOS write-back. |

## DecodeStaging Structure

```cpp
struct DecodeStaging {
    BoundedQueue<DecodeStagingEntry> fifo;
    std::vector<std::atomic<uint32_t>> generation;
    std::vector<std::atomic<uint32_t>> pending_count;
};
```

Three components:
1. **fifo** -- A `BoundedQueue` (mutex-based ring buffer) with capacity `max_users * 4`. This is the actual FIFO that the Reader pushes to and the Writer pops from.
2. **generation** -- Per-slot generation counter, advanced on CANCEL. Used to detect stale entries.
3. **pending_count** -- Per-slot reference count covering FIFO entries and in-progress Reader iterations.

### Construction

```cpp
explicit DecodeStaging(uint32_t max_users = DEFAULT_MAX_USERS)
    : fifo(static_cast<int>(max_users * 4)),
      generation(max_users),
      pending_count(max_users) {
    for (uint32_t i = 0; i < max_users; i++) {
        generation[i].store(0, std::memory_order_relaxed);
        pending_count[i].store(0, std::memory_order_relaxed);
    }
}
```

The FIFO capacity of `max_users * 4` accommodates the worst case: every user has multiple entries in flight (e.g., spec decode generates two entries per round, plus potential overlap with prior rounds draining).

## stage(): Claiming Before Publishing

```cpp
void stage(uint32_t uid, uint32_t tok, uint32_t pos,
           uint32_t spec_tok, uint32_t spec_pos, bool skip_spec = false) {
    // Claim the slot BEFORE publishing the entry.
    pending_count[uid].fetch_add(1, std::memory_order_acq_rel);
    uint32_t gen = generation[uid].load(std::memory_order_relaxed);
    DecodeStagingEntry entry{
        .slot_id = uid,
        .token_id = tok,
        .position = pos,
        .spec_token_id = spec_tok,
        .spec_position = spec_pos,
        .generation = gen,
        .skip_spec = skip_spec,
    };
    if (!fifo.try_push(entry)) {
        pending_count[uid].fetch_sub(1, std::memory_order_acq_rel);
        std::cerr << "FATAL: decode_staging FIFO full -- loopback token dropped for uid "
                  << uid << std::endl;
    }
}
```

The critical design decision is that `pending_count` is incremented BEFORE the entry is pushed to the FIFO. This prevents a cancel/recycle race:

```
Without pre-increment (BROKEN):
  Reader: push entry to FIFO
  --- gap ---
  API:    cancel_pending.mark(uid)
  API:    check in_flight_count == 0  (true, Reader already decremented)
  API:    check pending_count == 0    (true, entry is in FIFO but not counted!)
  API:    maybe_finalize_cleanup -> frees slot
  Writer: pop entry -> injects token for a freed/reallocated slot  (BUG)

With pre-increment (CORRECT):
  Reader: pending_count++  (now >= 1)
  Reader: push entry to FIFO
  --- gap ---
  API:    cancel_pending.mark(uid)
  API:    check pending_count == 0    (false, >= 1)
  API:    cleanup deferred
  Writer: pop entry -> check cancel_pending -> drop entry -> release()
  Writer: maybe_finalize_cleanup -> now safe to free
```

If the FIFO push fails (capacity exhausted), the pending_count is decremented to maintain the invariant and an error is logged. This should never happen in normal operation given the `max_users * 4` capacity -- FIFO fullness indicates a bug (e.g., the Writer is stuck).

The generation counter is read with relaxed ordering -- it only needs to be a consistent snapshot, not synchronized with anything at stage time. Matching happens when the Writer pops the entry.

## release(): Dropping a Claim

```cpp
void release(uint32_t uid) {
    pending_count[uid].fetch_sub(1, std::memory_order_acq_rel);
}
```

Called in two contexts:

1. **Writer thread**: After popping an entry from the FIFO and either injecting it or discarding it (cancelled slot). The Writer calls `decode_staging.release(uid)` to drop the claim that `stage()` acquired.
2. **ReaderClaim destructor**: After the Reader finishes processing one pipeline result. The `ReaderClaim` RAII guard (see below) increments `pending_count` at the start of each Reader iteration and calls `release()` on destruction, covering the gap between `in_flight_count--` and any subsequent `stage()` call.

The `acq_rel` ordering ensures that all writes performed while holding the claim are visible to any thread that subsequently observes `pending_count == 0`.

## ReaderClaim: RAII Reference Guard

Defined inside `DecodeScheduler::Impl`:

```cpp
struct ReaderClaim {
    Impl& impl;
    uint32_t uid;
    ReaderClaim(Impl& i, uint32_t u) : impl(i), uid(u) {
        impl.decode_staging.pending_count[uid].fetch_add(1, std::memory_order_acq_rel);
    }
    ~ReaderClaim() {
        impl.decode_staging.release(uid);
        impl.maybe_finalize_cleanup(uid);
    }
    ReaderClaim(const ReaderClaim&) = delete;
    ReaderClaim& operator=(const ReaderClaim&) = delete;
};
```

The Reader creates a `ReaderClaim` at the start of processing each pipeline result. This serves two purposes:

1. **Keeps pending_count nonzero** across the entire result-processing iteration, including the gap between `in_flight_count--` and any subsequent `stage()` call. Without this, a concurrent CANCEL could observe both counters at zero during the gap and prematurely finalize the slot.
2. **Calls maybe_finalize_cleanup on destruction**, which handles the case where the Reader's processing was the final outstanding reference for a cancelled slot.

The RAII pattern ensures the claim is always released, even if the Reader takes an early `continue` path (e.g., for cancelled users, prefill results, or post-complete in-flight results).

For the full analysis of how `ReaderClaim` and `pending_count` prevent the cancel/recycle race, see [Lock-Free Design](./lock_free_design.md).

## pending_count as Dual-Purpose Reference Count

`pending_count` serves two roles simultaneously:

### Role 1: FIFO Entry Count

Each `stage()` call increments pending_count. Each Writer pop followed by `release()` decrements it. This counts entries in the FIFO belonging to this slot.

### Role 2: Reader Iteration Guard

The `ReaderClaim` RAII guard increments pending_count at the start of each Reader iteration and decrements it at the end. This covers the window where the Reader has decremented `in_flight_count` but has not yet staged a new entry.

Combined with `in_flight_count`, the two counters form a comprehensive "busy" signal for the slot:

```
slot is safe to free  iff  in_flight_count == 0  AND  pending_count == 0
```

`in_flight_count` tracks tokens inside the pipeline (injected but not yet returned). `pending_count` tracks tokens in the staging FIFO or being processed by the Reader. Together, they cover every possible state where a reference to the slot exists outside of the data structures that `maybe_finalize_cleanup()` can safely reset.

## Generation Counter

```cpp
void advance_generation(uint32_t uid) {
    generation[uid].fetch_add(1, std::memory_order_release);
}
```

The generation counter is advanced by the API handler on CANCEL. Each staged entry captures the generation number at the time of staging. The Writer checks `cancel_pending.is_set(uid)` rather than comparing generations directly -- the `pending_count` mechanism guarantees that any stale entry still in the FIFO belongs to a cancelled slot.

The generation counter is also stamped onto each `OutputMessage` so the inference server can detect and discard late-arriving tokens from a previous session after a cancel-and-reallocate cycle. This is critical: the output queue may still contain tokens from a cancelled session when the slot is reallocated, and the IS needs a way to distinguish old tokens from new ones.

## try_pop()

```cpp
bool try_pop(DecodeStagingEntry& out) {
    return fifo.try_pop(out);
}
```

Delegates directly to the underlying `BoundedQueue`. The Writer calls this in its main loop. On success, the Writer owns the popped entry and is responsible for calling `release(uid)` when done.

## Writer Thread Processing Flow

The Writer's decode staging handling in `writer_loop`:

```
1. try_pop(entry) --> get next entry
2. if cancel_pending.is_set(uid):
       release(uid)
       maybe_finalize_cleanup(uid)
       continue
3. Determine if spec: user_table.spec_decode_enabled[uid] && !entry.skip_spec
4. in_flight_count[uid] += (do_spec ? 2 : 1)  // BEFORE release
5. release(uid)                                 // drop staging claim
6. pipeline_->inject(base token)
7. if do_spec: pipeline_->inject(spec token)
```

Note that `in_flight_count` is bumped **before** `release()`. This ensures the slot's combined busy count (`in_flight_count + pending_count`) never momentarily drops to zero during the handoff from staging claim to in-flight claim. Without this ordering, a cancel arriving between the release and the inject could observe both counts at zero and prematurely finalize.

## Who Calls What

| Operation | Called By | Purpose |
|---|---|---|
| `stage()` | Reader thread (decode loopback, EOS write-back), API thread (disaggregated continue) | Push token for Writer to inject |
| `try_pop()` | Writer thread | Get next decode token to inject |
| `release()` | Writer thread (after pop), Reader thread (ReaderClaim destructor) | Drop a pending_count claim |
| `advance_generation()` | API thread (on CANCEL) | Invalidate stale entries |

## FIFO Capacity Analysis

With capacity `max_users * 4`:
- Regular decode: 1 entry per user per decode round. With 64 users, at most 64 entries.
- Speculative decode: 1 entry per user per spec round (the entry carries both BASE and SPEC tokens). Still 64 entries max.
- The 4x factor provides headroom for bursty staging (e.g., multiple users completing simultaneously and each staging an EOS write-back) and ensures the FIFO never fills under normal operation.

---

**Previous:** [PromptTable and CancelBitmap](./prompt_table_and_cancel_bitmap.md) | **Next:** [PrefillQueue and BoundedQueue](./prefill_queue_and_bounded_queue.md)
