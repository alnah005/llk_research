# Key Invariants

The tt-llm-engine design is governed by six invariants that are never violated. Every data structure, thread interaction, and scheduling decision in subsequent chapters must satisfy all six simultaneously. They are stated in `docs/scheduler/decode.md` and enforced structurally throughout the implementation.

## The Six Key Invariants

### Invariant 1: PM Never Blocks on IS

> **Invariant:** The Pipeline Manager (Decode Scheduler) never blocks waiting for the Inference Server to produce or consume data.

The PM's Writer and Reader threads run continuously at pipeline-tick frequency. If the `output_queue` is full, PM spins briefly -- it does not call a blocking wait, does not yield the thread to the OS scheduler, and does not drop the message. The spin is bounded because the output queue is sized at `256 * max_users` entries. The `request_queue` is read by the PM's dedicated API handler thread using `try_pop()`, which returns immediately if empty; the API handler spin-waits between drain attempts rather than blocking on a condition variable.

**Why:** PM threads run at microsecond granularity. If any PM thread blocks on IS (which operates at millisecond granularity), the pipeline starves for the duration of the block. A 2ms HTTP handler delay wastes approximately 45 pipeline ticks at a ~44us tick period -- each of which could have injected a token for some user.

**Where enforced:** `decode_scheduler.cpp` -- every `output_queue.try_push()` call is wrapped in a `while (!try_push(msg)) { _mm_pause(); }` spin loop (the `_mm_pause()` intrinsic yields the core pipeline for a few cycles without entering a kernel wait). The API handler uses the same pattern for `request_queue`.

### Invariant 2: IS Never Calls PM Functions Directly

> **Invariant:** All IS-to-PM interaction flows through bounded queues. IS never invokes a PM function, accesses PM internal state, or shares memory with PM outside the three queues.

The `DecodeScheduler` public API exposes exactly three queue operations:

```cpp
bool push_request(const ISRequest& request);        // IS -> PM
bool try_pop_response(SchedulerResponse& response);  // PM -> IS
bool try_pop_output(OutputMessage& output);           // PM -> IS
```

Plus read-only diagnostic accessors (`get_user_state()`, `get_current_position()`, `get_in_flight_count()`, etc.) that perform atomic loads with acquire semantics. These are safe to call from any thread but are not part of the scheduling contract.

**Why:** Direct function calls from IS into PM would create shared-mutex dependencies that risk priority inversion and unbounded blocking. The queue boundary enables IS and PM to run in different processes, on different machines, or eventually across a PCIe/NoC boundary with the PM on a hardware co-processor.

**Where enforced:** The `DecodeScheduler` public API exposes only the three queue methods plus diagnostic getters. The internal `Impl` struct and all data structures are private (PIMPL pattern).

### Invariant 3: Pipeline Never Stalls Due to IS Latency

> **Invariant:** The Writer thread runs independently of IS request handling. The pipeline receives a token injection on every tick that has work available, regardless of IS activity.

The Writer thread's loop has a two-tier priority scheme:

1. **Decode tokens** from the `DecodeStaging` FIFO (populated by the Reader thread).
2. **Prefill tokens** from the `PrefillQueue` (populated by the API handler on SUBMIT/CONTINUE).

Neither source depends on IS input during steady-state operation. Decode tokens are staged by the Reader thread from pipeline results (a closed loop), and prefill tokens are drawn from the pre-stored `PromptTable`. The IS contributes work only at the edges: when submitting a new request or when issuing a cancellation.

The only point where the Writer blocks is `pipeline_->inject()`, which blocks if the hardware pipeline input buffer is full. This is an intentional stall -- the pipeline is genuinely full, there is no other useful work the Writer can do, and the stall resolves as soon as the pipeline advances one stage.

**Why:** Every stalled tick is a wasted pipeline slot for **all** users. If the pipeline has D=64 stages and 64 concurrent users, one stalled tick delays every user's next output token by one tick period. The Writer must inject one token per tick without any dependency on IS timing.

**Where enforced:** `writer_loop()` in `decode_scheduler.cpp`. The loop checks `decode_staging.try_pop()` then `prefill_queue.try_front()`. No other external input is consulted. The `api_loop()` runs on a separate dedicated thread.

### Invariant 4: User IDs Not Recycled Until Fully Drained

> **Invariant:** A user_id is not returned to the `FreeIdPool` until both `in_flight_count` and `pending_count` reach zero for that slot.

This invariant prevents a dangerous race: if a slot were freed while tokens from its previous session were still in the pipeline, those tokens could be delivered to a new session that reuses the same slot, corrupting its output stream.

The cleanup gate is implemented in `maybe_finalize_cleanup()`:

```cpp
void maybe_finalize_cleanup(uint32_t uid) {
    if (!cancel_pending.is_set(uid)) return;
    if (user_table.in_flight_count[uid].load(std::memory_order_acquire) != 0) return;
    if (decode_staging.pending_count[uid].load(std::memory_order_acquire) != 0) return;
    // ... CAS state -> INACTIVE, then cleanup sequence below ...
}
```

Only when all three conditions hold does the cleanup winner (exactly one thread via CAS on user state) execute the four-step cleanup:

1. `pipeline_->reset_kv(uid)` -- free device-side KV cache
2. `spec_state.reset(uid)` -- clear speculative decode tracking
3. `free_ids.free(uid)` -- return slot to the pool
4. `push_cancel_ack(request_id, uid)` -- notify IS that cancellation is complete

Note: `user_table.reset(uid)` and `cancel_pending.clear(uid)` are **not** called during cleanup. Both are performed lazily at the next ALLOCATE for the slot, not during cancel cleanup. This deferred-reset design avoids unnecessary work on the cancel path and ensures that the allocating thread is the one responsible for initializing slot state.

The `in_flight_count` covers tokens currently inside the hardware pipeline. The `pending_count` covers entries in the `DecodeStaging` FIFO plus in-progress reader iterations (via the `ReaderClaim` RAII guard). Both must reach zero before the slot is safe to recycle.

Additionally, a generation counter per slot (`DecodeStaging::generation`) is advanced on CANCEL. The Writer drops stale staging entries by checking `cancel_pending.is_set(uid)` -- the `pending_count` mechanism guarantees that any stale entry still in the FIFO belongs to a slot whose cancel_pending bit is still set. The generation counter is stamped onto each `OutputMessage` so the IS can detect and discard late-arriving output tokens from a previous session after a cancel-and-reallocate cycle.

**Where enforced:** `maybe_finalize_cleanup()` in `decode_scheduler.cpp`, called by the Writer (after dropping a cancelled entry), Reader (via `ReaderClaim` destructor), and API handler (after marking cancel).

### Invariant 5: KV Cache Retained on Completion

> **Invariant:** When generation completes (EOS, max_new_tokens, or context exhaustion), the user transitions to COMPLETE state but the KV cache is NOT freed. `current_position` is preserved for multi-turn continuation.

This is the foundation of multi-turn conversation support. When a generation turn finishes:

1. The user state transitions from DECODE to COMPLETE.
2. The `current_position` is set to one past the final token's KV position (via `set_position_on_complete` in the Reader). This becomes the starting position for the next turn's prefill.
3. The EOS write-back mechanism (`stage_eos_writeback`) injects the completing token back into the pipeline at `actual_token_pos` with `skip_spec=true`, so the terminator is written into KV context for the next turn. The `post_complete_in_flight` counter ensures the Reader discards the resulting loopback BASE result.
4. The `FreeIdPool` is NOT called. The slot remains allocated.

A subsequent CONTINUE request from the IS re-enters PREFILL from `current_position`, reusing all existing KV. Only an explicit CANCEL frees the slot and calls `pipeline_->reset_kv()`.

**Why:** Re-prefilling the entire conversation on every follow-up message would be extremely expensive for multi-turn conversations with long context. Retaining KV trades slot occupancy for prefill savings. The cost is that completed sessions consume a slot and device DRAM until explicitly cancelled. The planned KV tier management (see [Chapter 6](../ch6_lifecycle_is_integration_kv_tiering/kv_cache_tiering.md)) will allow completed sessions to be demoted to cheaper storage tiers under pressure.

**Where enforced:** The Reader's `emit_token` lambda sets `state = COMPLETE`, calls `stage_eos_writeback()`, and calls `set_position_on_complete()`. The `handle_local_continue()` function uses this position as `prefill_start_pos`.

### Invariant 6: PM Does Not Own End-User Waiting Policy

> **Invariant:** If PM cannot allocate a slot or admit work, it returns failure immediately. IS decides the user-facing behavior (wait, retry, timeout, error).

When `FreeIdPool.allocate()` returns `INVALID_SLOT`, the API handler pushes a `SchedulerResponse` with `error_code = 1`. When a tier command returns `NO_SPACE`, PM reports the status. In neither case does PM queue the request, set a timer, or retry autonomously.

IS decides the user-facing behavior:

- Return HTTP 503 (Service Unavailable) with a retry hint
- Enqueue the client on a bounded IS-side wait queue
- Cancel an idle session to free a slot (via the planned KV tiering commands)
- Return an error to the client

**Why:** Wait/retry/timeout logic is business policy -- it depends on SLA tiers, tenant classes, billing, and fairness rules that PM has no knowledge of. Embedding this in PM would couple scheduling to business logic and prevent PM from running on a co-processor with no network stack.

**Where enforced:** `handle_api_requests()` in `decode_scheduler.cpp`, ALLOCATE case: `error_code = (uid == INVALID_SLOT) ? 1 : 0`. No retry loop, no waiting queue.

The timescale argument motivating these invariants is detailed in [Ownership Contract -- The Timescale Argument](./ownership_contract.md#the-timescale-argument).

## Hardware Deployment Path

The decode scheduler is designed for eventual migration to a hardware accelerator's co-processor. Every data structure and algorithm choice is constrained by this goal. All structures are sized at construction time based on `SchedulerParams::max_users` and `SchedulerParams::max_seq_len` -- no dynamic allocation occurs on the hot path.

| Structure | Sizing | Hot-Path Allocation |
|-----------|--------|---------------------|
| `FreeIdPool` | `ceil(max_users / 64)` atomic words | None -- bitmap fits in registers |
| `UserTable` | `max_users` parallel vectors per field (~20 fields) | None -- indexed by slot_id |
| `PromptTable` | `max_users * max_seq_len` token array | None -- fixed 2D array |
| `CancelBitmap` | `ceil(max_users / 64)` atomic words | None -- bitmap fits in registers |
| `DecodeStaging` | `max_users * 4` entry FIFO | None -- bounded ring buffer |
| `PrefillQueue` | deque with mutex (not on hot path) | Rare -- only on SUBMIT |
| `SpecDecodeState` | `max_users` entries per field | None -- parallel arrays |
| `request_queue` | `2 * max_users` entries | None -- bounded ring buffer |
| `response_queue` | `2 * max_users` entries | None -- bounded ring buffer |
| `output_queue` | `256 * max_users` entries | None -- bounded ring buffer |

All enums (`TokenType`, `UserState`, `RequestType`) are `uint8_t`-backed integers; all sampling parameters are fixed-width scalars; sentinel values (`INVALID_SLOT`, `EMPTY_TOKEN`) are compile-time `UINT32_MAX` constants. No string comparison occurs in any scheduling loop. The `UserTable` stores per-user state in parallel arrays indexed by `slot_id`, enabling per-field atomicity without locking and mapping directly to register files or SRAM banks in a hardware implementation.

### Summary: What Maps Where

| Software Structure | Hardware Equivalent |
|-------------------|-------------------|
| `FreeIdPool` bitmap | 64-bit register + CTZ instruction |
| `UserTable` parallel arrays | Register files / SRAM banks |
| `PromptTable` flat token array | DRAM-backed token store |
| `CancelBitmap` | Single register (bitwise ops) |
| `DecodeStaging` FIFO | Hardware FIFO with generation tag |
| `SpecDecodeState` arrays | SRAM arrays |
| `BoundedQueue<T>` (IS queues) | DMA FIFO / hardware mailbox |
| `PipelineInterface::inject()` | H2D DMA write |
| `PipelineInterface::read_result()` | D2H DMA read |

### Migration Path

Separating PM from IS today means PM can be migrated to hardware without rearchitecting the system. Only the queue transport layer changes:

```
Current (software PM):
  IS <-- shared-memory queues --> PM (host threads)

Future (hardware PM):
  IS <-- PCIe DMA / NoC channels --> PM (co-processor firmware)
```

This is not a future aspiration -- it is an active design constraint that governs every code change to the scheduler.

---

**Next:** [Chapter 2 -- Threading and Data Structures](../ch2_threading_and_data_structures/index.md)
