# Decode Loopback

Once a user transitions from PREFILL to DECODE, its token generation enters a closed loop: the Reader receives a pipeline result, stages it in `DecodeStaging`, the Writer pops it and re-injects, the pipeline processes it, and the Reader receives the next result. This cycle repeats for every output token until completion.

## Governing Invariant

**Invariant: every non-completing decode result is re-injected exactly once, and `current_position` is updated only on completion.**

The loopback path ensures that each pipeline result produces exactly one staging entry (via `decode_staging.stage()`), which produces exactly one pipeline injection (via the Writer's decode path). No result is duplicated, no result is lost (absent cancellation). The `current_position` field is deliberately NOT updated during steady-state decode -- it is only set on completion via `set_position_on_complete()`. This design choice is explained in the position tracking section below.

## The Core Decode Cycle

The cycle has four stages. Let $D$ denote the number of pipeline stages (e.g., 64) and $\tau$ denote the pipeline tick period (e.g., 44 $\mu$s).

```
   Reader                     Pipeline                      Writer
     |                           |                            |
     |  <-- read_result() ----   |                            |
     |  (result for uid at       |                            |
     |   position P)             |                            |
     |                           |                            |
     |  --- stage(uid, tok,      |                            |
     |      P, ...) ----------> |  [entry in FIFO] --------> |
     |                           |                            |  try_pop()
     |                           |                            |  in_flight++
     |                           |                            |  release()
     |                           |  <-- inject(tok, P) -----  |
     |                           |                            |
     |         ... D ticks ...   |                            |
     |                           |                            |
     |  <-- read_result() ----   |                            |
     |  (result for uid at       |                            |
     |   position P+1)           |                            |
     |                           |                            |
```

### Stage 1: Reader Receives Result

```cpp
ResultDescriptor result = pipeline_->read_result();
uint32_t uid = result.slot_id;
ReaderClaim scratch(*this, uid);
user_table.in_flight_count[uid].fetch_sub(1, std::memory_order_relaxed);
```

The Reader blocks on `pipeline_->read_result()` until a result is available. Upon receiving a result, it immediately acquires a `ReaderClaim` (bumping `pending_count`) and decrements `in_flight_count`. The `ReaderClaim` RAII guard ensures the slot remains pinned for the duration of this iteration (see [Chapter 2 -- Decode Staging](../ch2_threading_and_data_structures/decode_staging.md#readerclaim-raii-reference-guard)).

### Stage 2: Filter and Discard

Before reaching the decode path, the result passes through a sequence of filters. Understanding this order is important for tracing the lifecycle of a result:

1. **INVALID_SLOT check**: If `result.slot_id == INVALID_SLOT`, the pipeline has shut down. The Reader exits.

2. **ReaderClaim acquisition**: `pending_count[uid]` is incremented. This RAII guard keeps the slot pinned for the duration of this iteration.

3. **in_flight_count decrement**: `in_flight_count[uid]` is decremented (relaxed ordering -- the ReaderClaim provides the safety boundary).

4. **Cancel filter**: If `cancel_pending.is_set(uid)`, the result is discarded. Counters `prefill_in_flight` and `post_complete_in_flight` are zeroed (fast-path cleanup). `maybe_finalize_cleanup()` is called.

5. **Post-complete filter**: If `post_complete_in_flight[uid] > 0` and the result is a BASE token, the result is from an EOS write-back injection (see [`completion_and_eos_writeback.md`](./completion_and_eos_writeback.md)). It is discarded and the counter is decremented.

6. **Prefill-in-flight filter**: If `prefill_in_flight[uid] > 0`, the result is from a non-last prefill injection. It is discarded and the counter is decremented.

Only results that pass all six checks reach the decode processing path. This ensures that stale, prefill, and post-complete results are handled efficiently without polluting the decode path.

### Stage 3: Non-Speculative Decode Path

For users with `spec_decode_enabled == false`, the decode path is straightforward:

```cpp
if (!user_table.spec_decode_enabled[uid]) {
    ctx_pos = result.actual_token_pos;
    if (!emit_token(result.actual_token, ctx_pos)) {
        decode_staging.stage(uid,
            result.actual_token, result.actual_token_pos,
            EMPTY_TOKEN, 0);
    }
    continue;
}
```

The Reader calls `emit_token()` with the actual (sampled) token and its KV position. If the token does not trigger completion (`emit_token` returns false), the Reader stages the token for re-injection. The staging entry carries:

- `token_id = result.actual_token`: the sampled token to inject next.
- `position = result.actual_token_pos`: the KV position where this token should be written.
- `spec_token_id = EMPTY_TOKEN`, `spec_position = 0`: no speculative token (non-spec mode).

If `emit_token` returns true (completion), no staging entry is created. Instead, `emit_token` internally calls `stage_eos_writeback()` and `set_position_on_complete()` (see [`completion_and_eos_writeback.md`](./completion_and_eos_writeback.md)).

### Stage 4: Writer Re-Injection

The Writer pops the staged entry on its next decode-priority tick and injects it into the pipeline (as described in [`writer_priority_and_decode.md`](./writer_priority_and_decode.md)). This completes one iteration of the decode cycle.

## Pipeline Latency and Steady-State Throughput

### Pipeline Timing Diagram

With $D$ pipeline stages and $N$ active decode users, the following timing diagram shows how tokens flow through the system. Each row is one pipeline tick. Each column is one pipeline stage.

Example: $D = 8$, $N = 3$ users ($U_0$, $U_1$, $U_2$), all in steady-state decode.

```
Tick | Writer injects | Pipeline stages (newest -> oldest)   | Reader receives
     |                | S1  S2  S3  S4  S5  S6  S7  S8      |
=====+================+======================================+=================
  1  | U0:T0          | U0  --  --  --  --  --  --  --      | (nothing yet)
  2  | U1:T0          | U1  U0  --  --  --  --  --  --      |
  3  | U2:T0          | U2  U1  U0  --  --  --  --  --      |
  4  | (prefill/idle) | --  U2  U1  U0  --  --  --  --      |
  5  | (prefill/idle) | --  --  U2  U1  U0  --  --  --      |
  6  | (prefill/idle) | --  --  --  U2  U1  U0  --  --      |
  7  | (prefill/idle) | --  --  --  --  U2  U1  U0  --      |
  8  | (prefill/idle) | --  --  --  --  --  U2  U1  U0      |
  9  | U0:T1          | U0  --  --  --  --  --  U2  U1      | U0:T0 result
 10  | U1:T1          | U1  U0  --  --  --  --  --  U2      | U1:T0 result
 11  | U2:T1          | U2  U1  U0  --  --  --  --  --      | U2:T0 result
 12  | (prefill/idle) | --  U2  U1  U0  --  --  --  --      |
 ...                                                         |
 17  | U0:T2          | U0  --  --  --  --  --  U2  U1      | U0:T1 result
 18  | U1:T2          | U1  U0  --  --  --  --  --  U2      | U1:T1 result
 19  | U2:T2          | U2  U1  U0  --  --  --  --  --      | U2:T1 result
```

Legend: `Ux:Ty` = user $x$, decode token $y$. `--` = empty pipeline slot.

### Steady-State Pattern

After the initial fill (ticks 1-8), the system settles into a repeating pattern of period $D$. In each period:

- The Writer injects $N$ decode tokens (one per user) in $N$ consecutive ticks.
- The remaining $D - N$ ticks are available for prefill or idle spin.
- The Reader receives $N$ results, each exactly $D$ ticks after injection.

**Each user receives one output token per $D$ ticks.** This is the fundamental TPOT bound:

$$\text{TPOT} = D \times \tau$$

For a `PipelineSimulatorConfig` with `num_stages=64` and `stage_duration_us=44`:

$$\text{TPOT} = 64 \times 44\,\mu s = 2.816\,\text{ms per token per user}$$

This TPOT is independent of $N$ (the number of active decode users), up to $N = D$. With $N < D$, the pipeline has idle slots that can serve prefill. With $N = D$, every pipeline slot is occupied by a decode token, and prefill is fully starved. When $N > D$, the pipeline is still limited to one result per tick; the excess users add to TPOT without increasing throughput:

$$\text{TPOT}_{N > D} = N \times \tau$$

The aggregate system throughput is always:

$$\text{Throughput}_{\text{system}} = \frac{\min(N, D)}{D \times \tau} \quad \text{tokens/second}$$

### Pipeline Utilization

Prefill tokens fill some of the idle stages when $N < D$, improving pipeline utilization without affecting decode TPOT (because decode has priority).

| Configuration | Pipeline utilization | Prefill bandwidth |
|---|---|---|
| $N < D$ | $N/D$ (partially utilized) | $D - N$ ticks per period available |
| $N = D$ | 100% (fully saturated) | Zero ticks available for prefill |
| $N > D$ | 100% | Zero; TPOT degrades proportionally |

## `current_position` Tracking

**Invariant: `current_position` is only written on completion, not during steady-state decode.**

During the decode loopback cycle, the Reader does NOT update `current_position` on each result. The position of each intermediate token is tracked implicitly through the staging entries -- the `position` field in `DecodeStagingEntry` carries the KV position, and the next position follows naturally from the device's sampling result.

`current_position` is updated only by `set_position_on_complete()`:

```cpp
auto set_position_on_complete = [&]() {
    user_table.current_position[uid].store(
        result.actual_token_pos + 1, std::memory_order_release);
};
```

This sets `current_position` to one past the completing token's position. The value serves as the multi-turn bookmark: when a CONTINUE request arrives, `handle_local_continue()` reads `current_position` and uses it as `prefill_start_pos` for the next turn's prefill.

### Why Not Update on Every Token?

Three reasons:

1. **Unnecessary overhead.** During steady-state decode, no other component reads `current_position`. The API handler only reads it on CONTINUE, which happens after completion. Writing it on every token would be a wasted release-store on every iteration.

2. **Correctness simplification.** If `current_position` were updated on every decode result and a CONTINUE arrived during active decode (which is not currently supported, but might be in the future), the position could be in an intermediate state. Updating only on completion guarantees that `current_position` always reflects a clean boundary.

3. **Hardware mapping.** On a hardware co-processor, a release-store to a shared register on every token would consume a memory fence cycle. Avoiding it keeps the hot path lean.

The position during prefill IS written on every token (`user_table.current_position[pfuid].store(device_pos + 1, ...)` in the Writer's prefill path), because a concurrent CANCEL during prefill may need to observe the current position for cleanup purposes.

## Stale Entry Filtering

**Invariant: stale staging entries from cancelled sessions are never injected.**

Stale entries are filtered by the `cancel_pending` bit (the same check shown in [`writer_priority_and_decode.md` Step 1](./writer_priority_and_decode.md#step-1-cancel-check)), not by generation comparison. The structural guarantee that `cancel_pending.clear(uid)` runs only during ALLOCATE -- which requires `pending_count == 0` -- ensures the bit remains set for the entire duration stale entries could exist in the FIFO. See [Chapter 2 -- Lock-Free Design](../ch2_threading_and_data_structures/lock_free_design.md) for the full cancel/recycle race prevention protocol. The generation counter serves a separate purpose: it tags `OutputMessage` entries so the IS can discard late-arriving tokens after slot reallocation (see [Chapter 2 -- Generation Counter](../ch2_threading_and_data_structures/decode_staging.md#generation-counter)).

## Busy-Count Lifecycle Through One Decode Iteration

The busy-count lifecycle for one decode iteration is traced step-by-step in [Chapter 2 -- Lock-Free Design](../ch2_threading_and_data_structures/lock_free_design.md#the-busy-count-lifecycle). In summary, the combined `in_flight_count + pending_count` never reaches zero between result arrival and the next pipeline injection.

## Decode Loopback Data Flow Summary

```
Pipeline result arrives
    |
    | read_result() -> ResultDescriptor
    v
+---+---------+
| Reader      |
|             |
| [1] ReaderClaim(uid)
|     WRITE: pending_count[uid]++ (acq_rel)
|
| [2] in_flight_count[uid]-- (relaxed)
|
| [3] cancel check?  -> no (normal path)
| [4] post_complete?  -> no
| [5] prefill_in_flight? -> no
|
| [6] emit_token(result.actual_token, result.actual_token_pos)
|     WRITE: output_queue.try_push(OutputMessage)
|     READ:  tokens_generated[uid], max_new_tokens[uid],
|            ignore_eos[uid], params.eos_token
|
| [7] stage(uid, actual_token, actual_token_pos, EMPTY_TOKEN, 0)
|     WRITE: pending_count[uid]++ (acq_rel)
|     WRITE: decode_staging.fifo.try_push(entry)
|
| [8] ~ReaderClaim
|     WRITE: pending_count[uid]-- (acq_rel)
|     CALL:  maybe_finalize_cleanup(uid)
+---+---------+
    |
    v
DecodeStaging FIFO
    |
    | try_pop(entry)
    v
+---+---------+
| Writer      |
|             |
| [1] cancel check? -> no
| [2] in_flight_count[uid]++ (release)
| [3] pending_count[uid]-- (acq_rel) via release()
| [4] device_sampling_params(uid)
| [5] inject(InjectDescriptor{
|       slot_id = uid,
|       token_id = actual_token,
|       position = actual_token_pos,
|       token_type = BASE, ...})
+---+---------+
    |
    v
Pipeline (D ticks)
    |
    v
Next ResultDescriptor
```

---

**Next:** [`completion_and_eos_writeback.md`](./completion_and_eos_writeback.md)
