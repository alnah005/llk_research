# Lock-Free Design

The decode scheduler's performance target is microsecond-scale per-tick scheduling. The Writer and Reader threads run tight loops that inject and process pipeline tokens. Any mutex acquisition on these hot paths would introduce unbounded worst-case latency (priority inversion, OS scheduling delays). This section explains how the scheduler achieves mutex-free operation on the hot path, the memory ordering discipline, the dual-purpose reference counting scheme, and the cancel/recycle race prevention mechanism.

## Hot Path: No Mutexes on Writer/Reader

On every iteration, the Writer and Reader threads access `UserTable` atomic fields, `DecodeStaging.pending_count`, `CancelBitmap`, `PromptTable` (Writer only, prefill), and `SpecDecodeState` (Reader only) -- all through per-field atomics or single-writer discipline with no mutex. The full per-structure, per-thread access matrix is in the [Thread Interaction Matrix](./threading_model.md#thread-interaction-matrix).

The only mutex-guarded structures touched on the hot path are `decode_staging.fifo` and `output_queue`, both `BoundedQueue` instances. These are uncontended in the common case: the FIFO is written by the Reader and read by the Writer -- they rarely collide. The output queue is written by the Reader and read by the IS output loop. The critical sections are single-element copies (sub-microsecond). An uncontended `std::mutex` on Linux degrades to a single `futex` fast-path syscall or even a user-space CAS.

The only truly contended mutex is `PrefillQueue.mtx`, and it is only touched by the Writer when the decode staging FIFO is empty (i.e., no decode work is pending). In steady-state decode with active users, the Writer processes decode tokens exclusively and never touches PrefillQueue.

The critical observation is that no mutex is held across a pipeline inject or read_result call. The pipeline I/O dominates latency, and all mutex regions are bounded to a few instructions of array manipulation.

## Memory Ordering Discipline

The scheduler uses five distinct memory ordering patterns:

### Pattern 1: Release-Acquire State Transitions

State transitions that must be visible to other threads before they proceed:

```
API handler:
  user_table.state[uid].store(UserState::PREFILL, memory_order_release);
  // All field writes above this line are visible to any thread
  // that loads state with acquire and sees PREFILL.

Writer:
  UserState st = user_table.state[uid].load(memory_order_acquire);
  // If st == PREFILL, all API handler's field initializations are visible.
```

This pattern applies to `state`, `current_position`, `in_thinking_phase`, and `in_flight_count`.

### Pattern 2: Release-Acquire on CancelBitmap

The CancelBitmap mediates visibility of the cancel request ID:

```
API handler:
  cancel_request_id[uid] = req.request_id;   // plain store
  cancel_pending.mark(uid);                    // fetch_or with release

Cleanup winner (any thread):
  if (cancel_pending.is_set(uid)) {            // load with acquire
      uint32_t rid = cancel_request_id[uid];   // guaranteed visible
  }
```

See [PromptTable and CancelBitmap](./prompt_table_and_cancel_bitmap.md) for full details on the release-acquire pair.

### Pattern 3: CAS for Exclusive Cleanup

`maybe_finalize_cleanup()` uses compare-and-swap on `UserTable.state` to ensure exactly one thread wins the cleanup race. The CAS code and its `acq_rel` ordering rationale are documented in [UserTable -- Atomicity of State Transitions](./user_table.md#atomicity-of-state-transitions). The key point for lock-free design: the `acq_rel` success ordering ensures the winner sees all preceding writes (acquire) and publishes the INACTIVE state before calling `free_ids.free(uid)` (release). The relaxed failure ordering is correct because a failed CAS means another thread already claimed cleanup.

### Pattern 4: Relaxed for Counters with External Ordering

`in_flight_count` is decremented with `relaxed` ordering in the Reader:

```cpp
user_table.in_flight_count[uid].fetch_sub(1, std::memory_order_relaxed);
```

This is safe because the `ReaderClaim`'s `pending_count` (with `acq_rel`) provides the ordering boundary. The decrement to `in_flight_count` does not need to be immediately visible to the API handler -- the `pending_count` still prevents cleanup.

### Pattern 5: Relaxed for Initialization

During ALLOCATE, `reset()` uses relaxed stores because no other thread can access the slot yet (it was just allocated from `FreeIdPool`, and no state transition has made it visible to Writer/Reader):

```cpp
state[uid].store(UserState::INACTIVE, std::memory_order_relaxed);
in_flight_count[uid].store(0, std::memory_order_relaxed);
// ...etc
```

The subsequent release-store to `state` (on SUBMIT) establishes the happens-before boundary that makes all these relaxed stores visible.

### Key Release-Acquire Pairs Summary

| Writer (release) | Reader (acquire) | What it synchronizes |
|---|---|---|
| `in_flight_count.fetch_add(release)` | `in_flight_count.load(acquire)` in cleanup | Writer's inject is visible before cleanup checks in_flight |
| `state.store(release)` | `state.load(acquire)` | State transitions are immediately visible to all threads |
| `cancel_pending.mark(fetch_or, release)` | `cancel_pending.is_set(load, acquire)` | All pre-cancel writes (including cancel_request_id) visible to checker |
| `in_thinking_phase.store(release)` | `in_thinking_phase.load(acquire)` | Thinking phase change visible to Writer's next sampling params |
| `current_position.store(release)` | `current_position.load(acquire/relaxed)` | Position updates visible for multi-turn continuation |
| `pending_count.fetch_add(acq_rel)` | `pending_count.load(acquire)` in cleanup | Staging claim visible before FIFO push; cleanup sees all prior claims |
| `free_ids.free(fetch_or, release)` | `free_ids.allocate(CAS, acq_rel)` | All cleanup writes visible to the next allocator of the same slot |

## pending_count as Dual-Purpose Reference Count

`DecodeStaging::pending_count[uid]` serves two simultaneous roles -- FIFO entry count and Reader iteration guard -- both covered in detail in [Decode Staging](./decode_staging.md#pending_count-as-dual-purpose-reference-count).

### Why Two Reference Counts Instead of One

`in_flight_count` tracks tokens inside the pipeline. `pending_count` tracks tokens/results being processed by the scheduler itself. If these were merged into a single counter, the Reader could not atomically decrement in-flight and re-increment for a staged loopback within the same operation. The gap between the two operations would create a zero-crossing that cleanup code could observe.

## CAS in maybe_finalize_cleanup

`maybe_finalize_cleanup()` is the convergence point for cancellation cleanup. It can be called from any of the three threads:

```cpp
void maybe_finalize_cleanup(uint32_t uid) {
    // Gate 1: cancel must be pending
    if (!cancel_pending.is_set(uid)) return;

    // Gate 2: no in-flight tokens
    if (user_table.in_flight_count[uid].load(std::memory_order_acquire) != 0) return;

    // Gate 3: no staging/reader claims
    if (decode_staging.pending_count[uid].load(std::memory_order_acquire) != 0) return;

    // Snapshot cancel request_id before CAS
    uint32_t request_id = cancel_request_id[uid];

    // Gate 4: CAS to claim cleanup
    UserState expected = user_table.state[uid].load(std::memory_order_acquire);
    if (expected == UserState::INACTIVE) return;
    if (!user_table.state[uid].compare_exchange_strong(
            expected, UserState::INACTIVE,
            std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;  // another thread won
    }

    // Cleanup sequence (exactly one winner executes this)
    pipeline_->reset_kv(uid);
    spec_state.reset(uid);
    free_ids.free(uid);
    push_cancel_ack(request_id, uid);
}
```

### Why CAS Instead of a Separate Cleanup Mutex

A mutex would serialize cleanup for all slots. The CAS operates per-slot and is contention-free in the common case (only one thread reaches this point per cancellation). It also avoids the risk of priority inversion if the mutex were held while calling `pipeline_->reset_kv()`.

### Ordering of Checks

The four gates are ordered from cheapest to most expensive:

1. `cancel_pending.is_set` -- single atomic load, often cached
2. `in_flight_count` -- single atomic load
3. `pending_count` -- single atomic load
4. CAS on `state` -- more expensive, but only reached when gates 1-3 pass

The guard conditions (gates 1-3) are necessary but not sufficient for safety -- they are not atomic as a group. Between checking `in_flight_count == 0` and checking `pending_count == 0`, a new result could arrive and be processed. The CAS is what provides the actual mutual exclusion. The guard conditions serve as a fast-path filter: if any reference count is nonzero, there is definitely work remaining and cleanup should be deferred. Only when all counts appear to be zero does the function attempt the CAS.

### request_id Snapshot

`cancel_request_id[uid]` is read **before** the CAS and **before** `free_ids.free(uid)`. This ordering is critical: after `free_ids.free()`, the slot could be immediately reallocated by the API thread's `allocate()`, and a new CANCEL for the new user could overwrite `cancel_request_id[uid]`. By snapshotting the value early, the cleanup winner uses the correct request_id for the acknowledgment.

### Who Calls maybe_finalize_cleanup?

Three threads can call it for the same slot nearly simultaneously:

- **Writer thread**: after popping a cancelled staging entry and calling `release()`.
- **Reader thread**: via the `ReaderClaim` destructor after processing a cancelled result.
- **API thread**: immediately after marking cancel (for the case where in_flight and pending are already zero).

The CAS ensures exactly one thread wins the cleanup. The loser observes either `INACTIVE` (another thread already cleaned up) or a failed CAS (concurrent modification), and returns without action.

## Cancel/Recycle Race Prevention

The most subtle correctness concern in the scheduler is the cancel/recycle race: the API thread cancels and frees a slot, the slot is immediately reallocated to a new user, and then a stale pipeline result or staging entry for the old user corrupts the new user's state.

### The Complete Cancel Protocol

**Step 1: Mark cancel (API handler)**

```
cancel_request_id[uid] = req.request_id    // stash ID
cancel_pending.mark(uid)                     // release: visible to all threads
decode_staging.advance_generation(uid)       // invalidate stale entries
prefill_queue.remove(uid)                    // remove from prefill scheduling
maybe_finalize_cleanup(uid)                  // attempt immediate cleanup
```

**Step 2: Hot-path threads observe cancel**

Writer: Checks `cancel_pending.is_set(uid)` before injecting. Drops cancelled entries, calls `release()` + `maybe_finalize_cleanup()`.

Reader: Checks `cancel_pending.is_set(uid)` after reading a result. Discards the result, zeroes `prefill_in_flight` and `post_complete_in_flight`, calls `maybe_finalize_cleanup()`.

**Step 3: Drain to zero**

Results for the cancelled user continue to arrive from the pipeline (tokens were already injected). Each result decrements `in_flight_count`. The `ReaderClaim` ensures `pending_count` stays nonzero across the Reader's processing window.

**Step 4: Cleanup gate**

`maybe_finalize_cleanup()` checks three conditions:
1. `cancel_pending.is_set(uid)` -- cancellation was requested
2. `in_flight_count[uid] == 0` -- no tokens in the pipeline
3. `pending_count[uid] == 0` -- no entries in staging FIFO, no Reader iteration in progress

All three must be true simultaneously.

**Step 5: Exclusive cleanup (CAS)**

Exactly one thread wins the CAS on `state`. That thread executes the cleanup sequence: `reset_kv`, `spec_state.reset`, `free_ids.free`, `push_cancel_ack`.

**Step 6: Deferred clear**

`cancel_pending.clear(uid)` is called during the NEXT ALLOCATE for the slot (not during cleanup). This means the cancel bit remains set between cleanup and reallocation, which is safe because the slot is in `FreeIdPool` and no thread accesses it.

### The Busy-Count Lifecycle

The combined `in_flight_count + pending_count` never drops to zero between a result arriving and the next pipeline injection. This table traces one complete decode iteration:

```
Step  | Event                              | in_flight | pending | Safe?
------+------------------------------------+-----------+---------+------
A     | Token in pipeline, no staging      | 1         | 0       | Yes (in-flight)
B     | Reader claims (ReaderClaim ctor)   | 1         | 1       | Yes (reader claims)
C     | Reader decrements in_flight        | 0         | 1       | Yes (claim holds)
D     | Reader stages loopback             | 0         | 2       | Yes (claim + staged)
E     | ReaderClaim destructor             | 0         | 1       | Yes (staged entry)
F     | Writer bumps in_flight before rel  | 1         | 1       | Yes (both nonzero)
G     | Writer releases staging claim      | 1         | 0       | Yes (in-flight)
```

At no point does the combined count reach zero between result arrival (A) and the next pipeline injection (G). A CANCEL arriving at any time would find a nonzero count and defer cleanup.

### Timeline of a Typical Cancel

```
T1: API receives CANCEL
    -> cancel_request_id[uid] = req_id
    -> cancel_pending.mark(uid)            // release
    -> advance_generation(uid)
    -> prefill_queue.remove(uid)
    -> maybe_finalize_cleanup(uid)
       -> in_flight_count > 0? likely yes (tokens in pipeline)
       -> return (cleanup deferred)

T2: Reader receives pipeline result for uid
    -> ReaderClaim(uid)                    // pending_count++
    -> in_flight_count--
    -> cancel_pending.is_set(uid)? yes
       -> zero out prefill_in_flight, post_complete_in_flight
       -> maybe_finalize_cleanup(uid)
          -> in_flight_count == 0? maybe not yet (more in-flight tokens)
          -> return
    -> ~ReaderClaim()                      // pending_count--
       -> maybe_finalize_cleanup(uid)
          -> all counts == 0? check again

... repeat for each remaining in-flight token ...

T_last: Final in-flight token processed
    -> ReaderClaim(uid)                    // pending_count = 1
    -> in_flight_count = 0
    -> ~ReaderClaim()                      // pending_count = 0
       -> maybe_finalize_cleanup(uid)
          -> cancel_pending.is_set? yes
          -> in_flight_count == 0? yes
          -> pending_count == 0? yes
          -> CAS(state, COMPLETE, INACTIVE) succeeds
          -> pipeline_->reset_kv(uid)
          -> free_ids.free(uid)
          -> push_cancel_ack(request_id, uid)
```

### The Staging Gap

The most dangerous moment is the gap between the Reader decrementing `in_flight_count` to zero and calling `stage()` for the next token. Without protection, a concurrent CANCEL could observe both counts at zero and prematurely free the slot. The `ReaderClaim` prevents this by keeping `pending_count` nonzero across the entire Reader iteration, deferring cleanup until the claim is released. This is the same pre-increment pattern used by `stage()` itself -- see [Decode Staging](./decode_staging.md#stage-claiming-before-publishing) for the broken/correct comparison. The `ReaderClaim` adds a second layer: even before `stage()` is called, the Reader's iteration-level claim holds the slot safe.

### The Writer's Contribution

The Writer also participates in the busy-count protocol. It increments `in_flight_count` **before** releasing the staging claim:

```cpp
user_table.in_flight_count[uid].fetch_add(do_spec ? 2 : 1, std::memory_order_release);
decode_staging.release(uid);
```

This ensures the slot's busy count never transiently drops to zero between the staging release and the in-flight increment. The ordering is deliberate: increment first, then release the claim.

## Hardware Deployment Path Mapping

None of the lock-free patterns described here require operating system involvement (no futex, no context switch, no syscall on the hot path). Everything operates in user-space with CPU-level atomic instructions. This maps directly to a hardware co-processor that has no operating system -- the same algorithms run as firmware with register operations replacing atomic instructions. The fixed-size, pre-allocated nature of all data structures means no memory allocator is needed on the hardware target. See the [Hardware Deployment Path Mapping](./index.md#hardware-deployment-path-mapping) table in the chapter overview for the complete software-to-hardware correspondence.

---

**Previous:** [SpecDecodeState](./spec_decode_state.md) | **Next:** [Chapter 3 -- Scheduling Algorithm](../ch3_scheduling_algorithm/index.md)
