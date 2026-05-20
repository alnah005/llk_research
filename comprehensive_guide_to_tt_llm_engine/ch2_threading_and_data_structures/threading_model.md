# Threading Model

The decode scheduler runs exactly three dedicated threads, each with a distinct role and strict access discipline over shared state. This separation is the foundation of the scheduler's ability to sustain microsecond-scale token injection: the Writer and Reader threads never acquire a mutex during steady-state operation, and the API handler thread is isolated from pipeline timing.

## The Three Threads

### Writer Thread (Hot Path)

**Role:** Tight inject loop. Pops decode tokens from `DecodeStaging` (priority 1) or prefill tokens from `PrefillQueue` (priority 2), builds `InjectDescriptor` messages, and calls `pipeline_->inject()`.

**Loop body** (`writer_loop()` in `decode_scheduler.cpp`):

1. Try `decode_staging.try_pop(entry)`. If an entry is available:
   - Check `cancel_pending.is_set(uid)`. If set, release the staging claim and call `maybe_finalize_cleanup()`. Skip.
   - Bump `in_flight_count[uid]` BEFORE releasing the staging claim (to prevent the busy count from momentarily hitting zero).
   - Release the staging claim via `decode_staging.release(uid)`.
   - Inject the BASE token via `pipeline_->inject()`.
   - If spec decode is enabled and `!entry.skip_spec`, also inject the SPEC token (bumping `in_flight_count` by 2 total).
   - Continue to next iteration.
2. Try `prefill_queue.try_front(pfuid)`. If a user is at the front:
   - Check cancellation. If cancelled, pop and finalize.
   - Check chunk remaining. If zero, reload chunk counter and rotate.
   - Check whether prefill is complete (prompt exhausted or max_seq_len reached). If so, pop, mark COMPLETE, and push an OutputMessage.
   - Otherwise, inject the next prefill token via `pipeline_->inject()`, advance prefill_pos, decrement chunk_remaining. If it was the last prompt token, transition state to DECODE and pop from the prefill queue.
3. If neither source has work, the loop spins (no explicit yield or sleep -- the `running` atomic load at the top acts as the spin check).

**Blocking point:** The only place the Writer blocks is inside `pipeline_->inject()`, which blocks when the hardware pipeline input buffer is full. This is the only acceptable stall -- the pipeline is genuinely saturated.

### Reader Thread (Hot Path)

**Role:** Processes pipeline results. Calls `pipeline_->read_result()` (blocking read from pipeline output), then handles decode loopback, completion, speculative decode verification, and cancellation cleanup.

**Loop body** (`reader_loop()` in `decode_scheduler.cpp`):

1. `pipeline_->read_result()` -- blocks until a result is available. Returns `INVALID_SLOT` on shutdown.
2. Acquire a `ReaderClaim` RAII guard (bumps `decode_staging.pending_count[uid]` to keep the slot pinned).
3. Decrement `in_flight_count[uid]` (relaxed ordering -- the ReaderClaim holds the slot safe).
4. Check `cancel_pending.is_set(uid)`. If cancelled, zero out `prefill_in_flight` and `post_complete_in_flight`, call `maybe_finalize_cleanup()`, continue.
5. Check `post_complete_in_flight[uid]`. If positive, this is a result from an EOS write-back inject -- decrement and discard.
6. Check `prefill_in_flight[uid]`. If positive, this is a non-sampled prefill result -- decrement and discard.
7. Process the decode result: emit token to `output_queue`, stage loopback via `decode_staging.stage()`, handle spec decode accept/reject/continue/stale logic.
8. On `ReaderClaim` destruction: release `pending_count`, call `maybe_finalize_cleanup()`.

The Reader exits when `pipeline_->read_result()` returns a sentinel `ResultDescriptor` with `slot_id == INVALID_SLOT`, which the pipeline delivers after `request_stop()` is called.

### API Handler Thread (Dedicated)

**Role:** Drains the `request_queue` (populated by the Inference Server) and processes ALLOCATE, SUBMIT, CONTINUE, and CANCEL requests. Pushes responses to `response_queue`.

**Loop body** (`api_loop()` in `decode_scheduler.cpp`):

```cpp
void api_loop() {
    while (running.load(std::memory_order_acquire)) {
        if (!handle_api_requests()) {
            _mm_pause();     // Intel PAUSE -- yields pipeline cycles
        }
    }
    handle_api_requests();   // Final drain after running goes false
}
```

The API handler spin-waits using `_mm_pause()` (Intel PAUSE instruction) when no requests are available. This yields the processor pipeline for a few cycles without entering a kernel wait, keeping wakeup latency in the sub-microsecond range.

It processes four request types:

| Request Type | Action |
|-------------|--------|
| `ALLOCATE` | Calls `free_ids.allocate()` (CAS), resets slot state, clears cancel bit, pushes `SchedulerResponse` |
| `SUBMIT` | Stores prompt in `PromptTable`, initializes `UserTable` fields, pushes to `PrefillQueue` |
| `CONTINUE` | Local: stores new prompt at `current_position`, sets state=PREFILL, pushes to `PrefillQueue`. Disaggregated: sets state=DECODE, stages migrated token directly |
| `CANCEL` | Stashes `cancel_request_id`, calls `cancel_pending.mark()`, advances generation, removes from `PrefillQueue`, attempts immediate cleanup |

## Why a Dedicated API Thread

The API handler could theoretically piggyback on the Reader thread (drain requests between `read_result()` calls). Three factors prevent this:

1. **Blocking read_result.** The Reader blocks inside `pipeline_->read_result()`. If the pipeline has no results ready (e.g., all users are idle), the Reader cannot process API requests. An ALLOCATE or CANCEL would stall until the next pipeline output.
2. **Latency isolation.** API requests are rare but latency-sensitive (the IS is waiting for an ALLOCATE response). Coupling them to the Reader's pipeline-driven cadence would make API latency unpredictable.
3. **Cancel safety.** Cancel processing involves `prefill_queue.remove()` and `decode_staging.advance_generation()`. These touch the same state the Writer reads. Having them on the Reader thread would add a new concurrent writer to those structures.

## DecodeScheduler::Impl as Central State Holder

All shared state lives inside `DecodeScheduler::Impl`, which is constructed once and destroyed when the scheduler shuts down:

```cpp
struct DecodeScheduler::Impl {
    std::unique_ptr<PipelineInterface> pipeline_;
    SchedulerParams params;

    FreeIdPool        free_ids;
    UserTable         user_table;
    PromptTable       prompt_table;
    CancelBitmap      cancel_pending;
    DecodeStaging     decode_staging;
    PrefillQueue      prefill_queue;
    SpecDecodeState   spec_state;

    std::vector<uint32_t> cancel_request_id;

    BoundedQueue<ISRequest>          request_queue;
    BoundedQueue<SchedulerResponse>  response_queue;
    BoundedQueue<OutputMessage>      output_queue;

    std::thread writer_thread;
    std::thread reader_thread;
    std::thread api_thread;
    std::atomic<bool> running{false};
};
```

The `Impl` struct is hidden behind the PIMPL pattern in `DecodeScheduler`. The public API exposes only queue operations (`push_request()`, `try_pop_output()`, etc.) and diagnostic getters. All three thread functions are methods of `Impl`, capturing `this` to access shared state.

Construction of `Impl` pre-allocates every data structure to its maximum capacity. The `FreeIdPool`, `UserTable`, `PromptTable`, `CancelBitmap`, `DecodeStaging`, and `SpecDecodeState` are all sized to `params.max_users`. Queue capacities are computed as multiples of `max_users`. After construction, no further heap allocation occurs during scheduling.

## Thread Interaction Matrix

This matrix shows which thread reads or writes each shared state field. "W" = writes, "R" = reads, "RW" = reads and writes, "CAS" = compare-and-swap. Memory ordering annotations follow in parentheses where non-relaxed.

| Shared State | API Thread | Writer Thread | Reader Thread |
|---|---|---|---|
| `running` | RW (release store, acquire load) | R (acquire) | R (acquire) |
| `free_ids` (bitmap) | W: allocate (CAS, acq_rel); W: free (fetch_or, release) (sync cancel + cleanup winner) | W: free (fetch_or, release) (cleanup winner, rare) | W: free (fetch_or, release) (cleanup winner) |
| `user_table.state` | W: store (release) | RW: store (release), CAS in cleanup (acq_rel) | RW: store (release), CAS in cleanup (acq_rel) |
| `user_table.in_flight_count` | RW: store 0 on init (relaxed), load (acquire) in cancel/cleanup | W: fetch_add (release) | W: fetch_sub (relaxed) |
| `user_table.current_position` | RW: store on init (relaxed), store on disaggregated continue (release), load on local continue (relaxed) | W: store during prefill (release) | W: store on completion (release) |
| `user_table.prefill_pos` | W: set on SUBMIT/CONTINUE | RW: read and advance | -- |
| `user_table.prefill_start_pos` | W: set on SUBMIT/CONTINUE | R: read | -- |
| `user_table.prefill_chunk_remaining` | W: set on SUBMIT/CONTINUE | RW: read, decrement, reload | -- |
| `user_table.prefill_in_flight` | W: store 0 on init (relaxed) | W: fetch_add (release) | RW: load (acquire), fetch_sub (relaxed), store 0 on cancel |
| `user_table.post_complete_in_flight` | W: store 0 on init (relaxed) | -- | RW: fetch_add (release), load (acquire), fetch_sub (relaxed), store 0 on cancel |
| `user_table.tokens_generated` | W: init to 0 | -- | W: increment, reset to 0 on complete |
| `user_table.max_new_tokens` | W: set | -- | R: read for completion check |
| `user_table.spec_decode_enabled` | W: set | R: read | R: read |
| `user_table.ignore_eos` | W: set | -- | R: read |
| `user_table.temperature` | W: set | R: read (device_sampling_params) | R: read (via device_sampling_params) |
| `user_table.top_p` | W: set | R: read | R: read (check_acceptance) |
| `user_table.top_k` | W: set | R: read | R: read (check_acceptance) |
| `user_table.relaxed_acceptance_threshold` | W: set | -- | R: read (check_acceptance) |
| `user_table.in_thinking_phase` | W: store false (release) | R: load (acquire) | RW: load (acquire, check_acceptance), store (release) |
| `user_table.spec_accepts/rejects` | -- | -- | W: increment |
| `prompt_table` | W: store on SUBMIT/CONTINUE | RW: get_token, get_length (read), clear (write) during prefill | -- |
| `cancel_pending` (bitmap) | W: mark (fetch_or, release); W: clear on ALLOCATE/sync cancel (fetch_and, release) | R: is_set (load, acquire) | R: is_set (load, acquire) |
| `cancel_request_id` | W: plain store (before mark) | R: plain load (after is_set acquire) | R: plain load (after is_set acquire) |
| `decode_staging.fifo` | W: stage/try_push (mutex) (disaggregated continue) | R: try_pop (mutex) | W: stage/try_push (mutex) |
| `decode_staging.pending_count` | W: fetch_add via stage (acq_rel) (disaggregated continue) | W: release via staging release | RW: fetch_add in stage (acq_rel), fetch_sub in release (acq_rel) |
| `decode_staging.generation` | W: fetch_add on CANCEL (release) | R: load (relaxed, via entry) | R: load (relaxed) |
| `prefill_queue` | W: push, remove (mutex) | R: try_front, pop_front, rotate (mutex) | -- |
| `spec_state` (all fields) | W: reset | W: reset (cleanup winner, rare) | RW: full read/write (single-threaded, no atomics) |
| `request_queue` | R: try_pop (mutex) | -- | -- |
| `response_queue` | W: try_push (mutex) | W: push_cancel_ack (cleanup winner) | W: try_push via push_cancel_ack (mutex) |
| `output_queue` | -- | W: try_push on prefill exhaust | W: try_push (mutex) |

Key observations from this matrix:

- The Writer never acquires a per-slot mutex. Its only mutex accesses are `decode_staging.fifo.try_pop`, `prefill_queue`, `output_queue.try_push` (rare, only on prefill-side context exhaustion), and `response_queue.try_push` (rare, only when the Writer wins the cleanup CAS via `push_cancel_ack`).
- The Reader never acquires a per-slot mutex. Its mutex accesses are `decode_staging.fifo.try_push` (via `stage()`), `output_queue.try_push`, and `response_queue.try_push` (via `push_cancel_ack`, rare).
- `cancel_request_id` is written by the API thread before `cancel_pending.mark()` (release), and read by the cleanup winner after `cancel_pending.is_set()` (acquire). The release-acquire pair on CancelBitmap makes this plain-store/plain-load safe.
- `PromptTable` has no concurrent access to the same row: the API stores a row before pushing the user to PrefillQueue, and the Writer only reads rows for users already in the queue.
- `spec_state` is primarily owned by the Reader for read/write during decode. The API thread calls `reset()` during ALLOCATE/SUBMIT/CONTINUE (before active decode). The Writer can also call `reset()` when it wins the cleanup CAS via `maybe_finalize_cleanup()` (rare).

## Start/Stop Lifecycle

### Start

```cpp
void start() {
    if (running.load(std::memory_order_acquire)) return;  // idempotent
    running.store(true, std::memory_order_release);

    int w_cpu = params.writer_cpu;
    int r_cpu = params.reader_cpu;
    int a_cpu = params.api_cpu;
    resolve_cpu_assignments(w_cpu, r_cpu, a_cpu);

    writer_thread = std::thread([this] { writer_loop(); });
    reader_thread = std::thread([this] { reader_loop(); });
    api_thread    = std::thread([this] { api_loop(); });

    pin_to_cpu(writer_thread, w_cpu, "writer_thread");
    pin_to_cpu(reader_thread, r_cpu, "reader_thread");
    pin_to_cpu(api_thread,    a_cpu, "api_thread");
}
```

The `running` flag is set to `true` with release semantics before any thread is spawned, ensuring all threads see `running == true` on their first acquire-load. CPU affinity is resolved and pinned after thread creation (see [CPU Affinity](./cpu_affinity.md)).

### Stop

```cpp
void stop() {
    if (!running.load(std::memory_order_acquire)) return;  // idempotent

    running.store(false, std::memory_order_release);
    pipeline_->request_stop();

    if (api_thread.joinable())    api_thread.join();
    if (writer_thread.joinable()) writer_thread.join();
    if (reader_thread.joinable()) reader_thread.join();

    pipeline_->shutdown();
}
```

The shutdown sequence is:

1. **Set `running` to false** (release store). The API thread and Writer thread check this at the top of their loops and will exit.
2. **Call `pipeline_->request_stop()`.** This causes `read_result()` to return a sentinel `ResultDescriptor` with `slot_id == INVALID_SLOT`, which breaks the Reader out of its blocking loop.
3. **Join all three threads.** The order is API, Writer, Reader -- though any join order would work since all three will eventually terminate after steps 1-2. The convention is to join the API thread first because it may push to `PrefillQueue` or `response_queue`, giving the Writer and Reader a chance to drain those.
4. **Call `pipeline_->shutdown()`.** This performs final device-side cleanup. It is only safe to call after all threads are joined because inject/read_result are no longer in use.

The `Impl` destructor calls `stop()`, ensuring cleanup even if the caller forgets.

## Memory Ordering at the Thread Boundary

The `running` flag uses release/acquire semantics as the outermost synchronization boundary:

- `start()` stores with `memory_order_release` -- all data structure initialization is visible before threads observe `running == true`.
- `stop()` stores with `memory_order_release` -- the false value is visible to all threads.
- Each thread loop loads with `memory_order_acquire` -- seeing `running == false` guarantees all preceding state changes are visible.

Within the running window, each data structure uses its own memory ordering discipline (detailed in subsequent sections and in [Lock-Free Design](./lock_free_design.md)).

---

**Previous:** [Chapter Overview](./index.md) | **Next:** [CPU Affinity](./cpu_affinity.md)
