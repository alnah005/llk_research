# Regular Decode Flow: End-to-End Walkthrough

This section traces a complete non-speculative decode session from ALLOCATE through COMPLETE, integrating all the mechanisms described in previous sections into a single continuous narrative. It then extends the walkthrough to multi-user interleaving, multi-turn CONTINUE (both local and disaggregated), and provides a timing analysis.

All examples use regular (non-speculative) decode.

## Single-User Lifecycle

Consider a single user with a 5-token prompt (`[A, B, C, D, E]`), `max_new_tokens = 3`, and `spec_decode = false`. The pipeline has $D = 8$ stages (simplified for readability). Chunk size is 24 (larger than the prompt, so no rotation occurs).

### Phase 1: ALLOCATE

The IS pushes an `ISRequest` with `type = ALLOCATE` to `request_queue`.

**API thread processing:**
1. `free_ids.allocate()` returns slot `uid = 7`.
2. `user_table.reset(7)` -- all fields zeroed/defaulted (relaxed stores, safe because slot is not yet visible).
3. `cancel_pending.clear(7)` -- clear any stale cancel bit.
4. `spec_state.reset(7)` -- clear speculative decode state.
5. Push `SchedulerResponse{request_id, slot_id=7, error_code=0}` to `response_queue`.

If no slot is available, `free_ids.allocate()` returns `INVALID_SLOT` and the response carries `error_code = 1`. The IS decides how to handle the failure (see [Chapter 1 -- Key Invariants, Invariant 6](../ch1_architecture_and_layering/key_invariants.md#invariant-6-pm-does-not-own-end-user-waiting-policy)).

### Phase 2: SUBMIT

The IS pushes `ISRequest{type=SUBMIT, slot_id=7, tokens=[A,B,C,D,E], gen={max_new_tokens=3}}`.

**API thread processing:**
1. `prompt_table.store(7, [A,B,C,D,E], 5)` -- store prompt starting at index 0.
2. `user_table.state[7] = PREFILL` (release store -- all field writes above are visible to any thread that subsequently loads state with acquire and observes PREFILL).
3. `current_position[7] = 0`, `prefill_pos[7] = 0`, `prefill_start_pos[7] = 0`.
4. `max_new_tokens[7] = 3`, `tokens_generated[7] = 0`.
5. `in_flight_count[7] = 0`, `prefill_chunk_remaining[7] = 24`.
6. `spec_decode_enabled[7] = false`.
7. `prefill_queue.push(7)`.

User 7 is now in the `PrefillQueue`, waiting for the Writer.

### Phases 3-8: Prefill, Decode, Completion, and EOS Drain

The following tables trace the remaining lifecycle. Each mechanism (prefill injection, decode loopback, `emit_token`, `stage_eos_writeback`, etc.) is detailed in its own file; the tables here show only the tick-by-tick counter evolution.

**Phase 3 -- Prefill injection (5 ticks):**

| Tick | Event | in_flight | prefill_in_flight | Notes |
|------|-------|-----------|-------------------|-------|
| 1 | Inject A at pos 0 | 1 | 1 | `is_last=false` |
| 2 | Inject B at pos 1 | 2 | 2 | |
| 3 | Inject C at pos 2 | 3 | 3 | |
| 4 | Inject D at pos 3 | 4 | 4 | |
| 5 | Inject E at pos 4 (LAST) | 5 | 4 | `state[7]=DECODE`, pop from prefill queue |

**Phase 4 -- Prefill result discard (ticks 9-12):**

| Tick | Event | in_flight | prefill_in_flight |
|------|-------|-----------|-------------------|
| 9 | Result for A: discard | 4 | 3 |
| 10 | Result for B: discard | 3 | 2 |
| 11 | Result for C: discard | 2 | 1 |
| 12 | Result for D: discard | 1 | 0 |

**Phase 5 -- First decode result (tick 13):**

| Tick | Event | in_flight | tokens_generated | Output |
|------|-------|-----------|-----------------|--------|
| 13 | Result for E; `emit_token(T1, pos=5)` -> not complete | 0 | 1 | `OutputMessage{T1, complete=false}` |
| | `stage(7, T1, 5, EMPTY, 0)` | | | Loopback entry queued |

**Phase 6 -- Decode loopback (ticks 14-31):**

| Tick | Thread | Event | in_flight | tokens_generated |
|------|--------|-------|-----------|-----------------|
| 14 | Writer | Pop entry, inject T1 at pos 5 | 1 | 1 |
| 22 | Reader | Result for T1; `emit_token(T2, pos=6)` -> not complete; stage T2 | 0 | 2 |
| 23 | Writer | Pop entry, inject T2 at pos 6 | 1 | 2 |
| 31 | Reader | Result for T2; `emit_token(T3, pos=7)` -> **complete** (`3 >= max_new_tokens`) | 0 | 3 |

**Phase 7 -- Completion (tick 31, continued):**

| Action | Effect |
|--------|--------|
| `state[7] = COMPLETE` | Release store |
| `stage_eos_writeback()` | `post_complete_in_flight=1`; staging entry with `skip_spec=true` |
| `set_position_on_complete()` | `current_position[7] = 8` (release store) |
| Publish `OutputMessage{T3, complete=true, tokens_generated=3}` | `tokens_generated` reset to 0 |

**Phase 8 -- EOS write-back drain (ticks 32-40):**

| Tick | Thread | Event | in_flight | post_complete_in_flight |
|------|--------|-------|-----------|------------------------|
| 32 | Writer | Pop EOS entry, inject T3 at pos 7 (BASE only) | 1 | 1 |
| 40 | Reader | Result arrives; `post_complete_in_flight > 0` -> discard | 0 | 0 |

After tick 40, slot 7 is quiescent: `state=COMPLETE`, `in_flight=0`, `current_position=8`, KV retained.

## End-to-End Swim-Lane Diagram

```
         IS                     API Thread              Writer Thread           Reader Thread          Pipeline
          |                         |                        |                       |                     |
   ALLOCATE ----push_request------->|                        |                       |                     |
          |                    free_ids.allocate()           |                       |                     |
          |                    user_table.reset()            |                       |                     |
          |<---response_queue-------+                        |                       |                     |
          |   (slot_id=7)           |                        |                       |                     |
          |                         |                        |                       |                     |
    SUBMIT ----push_request-------->|                        |                       |                     |
          |                    prompt_table.store()          |                       |                     |
          |                    state[7]=PREFILL              |                       |                     |
          |                    prefill_queue.push(7)         |                       |                     |
          |                         |                        |                       |                     |
          |                         |                   try_front(7)                 |                     |
          |                         |                   inject(A,pos=0) ------------>|-------------------->|
          |                         |                   inject(B,pos=1) ------------>|-------------------->|
          |                         |                   ...                          |                     |
          |                         |                   inject(E,pos=4) ------------>|-------------------->|
          |                         |                   state[7]=DECODE              |                     |
          |                         |                        |                       |                     |
          |                         |                        |              prefill results arrive          |
          |                         |                        |              (4x: prefill_in_flight--)       |
          |                         |                        |                       |                     |
          |                         |                        |              1st decode result:              |
          |                         |                        |              emit_token(T1) ->               |
   <-----output_queue.try_pop------+------------------------+-----------  output_queue.push()              |
          |  OutputMessage          |                        |              stage(7, T1, pos)               |
          |  {T1, complete=false}   |                        |                       |                     |
          |                         |                   try_pop(entry)               |                     |
          |                         |                   inject(T1, pos)  ----------->|-------------------->|
          |                         |                        |                       |                     |
          |                     ... decode loopback repeats ...                      |                     |
          |                         |                        |                       |                     |
          |                         |                        |              emit_token(T3: complete) ->     |
   <-----output_queue.try_pop------+------------------------+-----------  output_queue.push()              |
          |  OutputMessage          |                        |              stage_eos_writeback()            |
          |  {T3, complete=true}    |                        |              set_position_on_complete()      |
          |                         |                        |                       |                     |
          |                         |                   EOS write-back inject ------>|-------------------->|
          |                         |                        |              post_complete_in_flight-- (discard)|
          |                         |                        |                       |                     |
          |                         |                   (quiescent, COMPLETE state)  |                     |
```

## Multi-User Interleaving

Consider two users: user A (5-token prompt) and user B (10-token prompt), submitted back-to-back. $D = 8$, `chunk_size = 24`.

```
PrefillQueue initial: [A, B]     DecodeStaging: empty

Tick | Source        | Action                        | Notes
=====+===============+================================+=============================
  1  | PrefillQueue  | Inject A[0] at pos 0           | A at front, chunk_rem=23
  2  | PrefillQueue  | Inject A[1] at pos 1           |
  3  | PrefillQueue  | Inject A[2] at pos 2           |
  4  | PrefillQueue  | Inject A[3] at pos 3           |
  5  | PrefillQueue  | Inject A[4] at pos 4           | Last! A -> DECODE, pop.
     |               |                                | PrefillQueue: [B]
  6  | PrefillQueue  | Inject B[0] at pos 0           | B now at front
  7  | PrefillQueue  | Inject B[1] at pos 1           |
  8  | PrefillQueue  | Inject B[2] at pos 2           |
  9  | PrefillQueue  | Inject B[3] at pos 3           | Reader: A[0] result
     |               |                                | (prefill_in_flight--, discard)
 10  | PrefillQueue  | Inject B[4] at pos 4           |
 11  | PrefillQueue  | Inject B[5] at pos 5           |
 12  | PrefillQueue  | Inject B[6] at pos 6           |
 13  | PrefillQueue  | Inject B[7] at pos 7           | Reader: A's last prefill result
     |               |                                | -> first decode result for A
     |               |                                | emit_token, stage loopback
 14  | DecodeStaging | Re-inject A's decode token     | DECODE PRIORITY over B prefill
 15  | PrefillQueue  | Inject B[8] at pos 8           | Back to B after A served
 16  | PrefillQueue  | Inject B[9] at pos 9           | Last! B -> DECODE, pop.
 ...
```

Key observations:

- User A's decode loopback at tick 14 preempts User B's prefill, demonstrating decode priority.
- User B's short prompt completes within a single chunk and transitions to DECODE without rotation.
- After both enter decode, they interleave in the pipeline $D$ ticks apart.

## Multi-Turn CONTINUE

After a user reaches COMPLETE, the IS can issue a CONTINUE request to start a new turn. There are two paths.

### Local Continue

The IS pushes `ISRequest{type=CONTINUE, slot_id=7, tokens=[F,G], gen={max_new_tokens=5}}`. The API handler executes `handle_local_continue()`:

```
API handler (handle_local_continue):
  start_pos = current_position[7] = 8 (relaxed load, safe due to
    happens-before through output queue)
  prompt_table.store(7, [F,G], 2)     -- new prompt at indices 0,1
  state[7] = PREFILL (release store)
  prefill_pos[7] = 8
  prefill_start_pos[7] = 8
  max_new_tokens[7] = 5
  tokens_generated[7] = 0
  prefill_chunk_remaining[7] = 24
  spec_state.reset(7)
  prefill_queue.push(7)
```

The Writer serves user 7 from the prefill queue:

```
Tick N:   Inject token F at position 8
  prompt_idx = 8 - 8 = 0, is_last = false
  prefill_in_flight++, in_flight_count++
  inject({slot=7, token=F, prefill_token=G, pos=8})

Tick N+1: Inject token G at position 9 (LAST)
  prompt_idx = 9 - 8 = 1, is_last = true (1 == 2-1)
  in_flight_count++ (prefill_in_flight NOT incremented)
  inject({slot=7, token=G, prefill_token=EMPTY, pos=9})
  state[7] = DECODE
  prefill_queue.pop_front()
```

Token F's result will be discarded (`prefill_in_flight`). Token G's result enters the decode path, producing the first sampled token for turn 2. The decode loopback resumes with positions starting at 10, and the KV cache at positions 0-9 (turn 1 context + EOS + turn 2 prompt) is fully reused.

### Disaggregated Continue

For disaggregated decode (where prefill was performed on a separate machine and KV was migrated), `handle_disaggregated_continue()` takes a different path:

```
API handler (handle_disaggregated_continue):
  n = tokens.size() - 1           // number of prefilled positions
  migrated_token = tokens.back()   // last token from remote prefill
  state[uid] = DECODE (release store)   // skip PREFILL entirely
  current_position[uid] = n + 1
  // ... set gen params ...
  decode_staging.stage(uid, migrated_token, n, EMPTY_TOKEN, 0)
```

The user skips PREFILL and enters DECODE directly. The migrated token is staged in `DecodeStaging` as if it were a normal decode loopback entry. The Writer pops it and injects it, starting the decode cycle from position $n$. The KV cache at positions 0 through $n - 1$ was populated by the remote prefill machine and migrated to this device. No prefill queue involvement, no chunk processing.

## Timing Analysis

For a user with prompt length $L$, `chunk_size` $C$, pipeline depth $D$, and decode length $G$ tokens:

| Phase | Duration (ticks) | Formula |
|-------|-----------------|---------|
| ALLOCATE + SUBMIT | ~2-3 | API processing, not tick-bound |
| Prefill injection | $L + \lceil L/C \rceil - 1$ | $C$ tokens per chunk; includes rotation ticks |
| Prefill pipeline drain | $D$ | First result arrives $D$ ticks after first inject |
| First decode token output (TTFT) | $L + D + \lceil L/C \rceil - 1$ | From SUBMIT to first OutputMessage |
| Steady-state decode (per token) | $D$ | One loopback cycle per token |
| Total decode phase | $G \times D$ | $G$ output tokens, each takes $D$ ticks |
| EOS write-back drain | $D$ | Post-completion pipeline drain |
| **Total end-to-end** | $L + (G+1) \times D + \lceil L/C \rceil - 1$ | From SUBMIT to slot idle |

For the single-user example above ($L=5$, $G=3$, $D=8$, $C=24$):

- TTFT: $5 + 8 + \lceil 5/24 \rceil - 1 = 5 + 8 + 0 = 13$ ticks = $13 \times 44\mu s = 572\mu s$
- Total: $5 + 4 \times 8 + 0 = 37$ ticks (tick 1 through tick 40, accounting for the EOS write-back drain)

### Steady-State Multi-User Timing

For a system with $N$ active decode users, no prefill:

| Metric | Formula | Example ($D=64$, $\tau=44\mu s$, $N=32$) |
|--------|---------|---------------------------------------------|
| Steady-state TPOT (per user) | $D \cdot \tau$ | $64 \times 44\mu s = 2.816$ ms |
| System throughput | $\frac{N}{D \cdot \tau}$ | $\frac{32}{2.816\text{ms}} \approx 11{,}364$ tokens/s |
| Maximum prefill-induced decode delay | $\text{chunk\_size} \cdot \tau$ | $24 \times 44\mu s = 1.056$ ms |
| Pipeline utilization (decode only) | $\min(N/D, 1.0)$ | $32/64 = 50\%$ |

The maximum prefill-induced decode delay is the worst case for a single decode token -- it must wait at most `chunk_size` ticks for the Writer to finish a prefill chunk before processing the decode staging entry. In practice, the delay is usually 0 or 1 tick because decode staging is checked on every loop iteration.

When $N > D$, TPOT degrades:

$$\text{TPOT}_{N > D} = N \cdot \tau$$

## End-to-End State Machine Summary

```
INACTIVE --[ALLOCATE]--> INACTIVE (slot allocated, state reset)
         --[SUBMIT]----> PREFILL  (prompt stored, pushed to PrefillQueue)
                         |
         [Writer injects chunk_size tokens per rotation, advancing prefill_pos]
                         |
         --[last token]-> DECODE   (state transition, popped from PrefillQueue)
                         |
         [Reader receives results, emit_token publishes tokens,
          stages loopback entries, Writer re-injects]
                         |
         --[EOS/max/ctx]-> COMPLETE (EOS write-back staged,
                                     current_position set,
                                     OutputMessage published)
                         |
         --[CONTINUE]---> PREFILL  (new prompt from current_position,
                                    KV cache reused)
         --[CONTINUE     DECODE   (disaggregated: skip PREFILL,
            disagg.]-->            migrated token staged directly)
         --[CANCEL]-----> INACTIVE (after drain: reset_kv, free slot)

PREFILL --[ctx_exhaust]-> COMPLETE (no EOS write-back, tokens_generated=0)
```

Every state and transition is traceable to a specific code path in `decode_scheduler.cpp`:

- INACTIVE to PREFILL: `handle_api_requests()`, SUBMIT case (line 648).
- PREFILL to DECODE: `writer_loop()`, `is_last` branch (line 337).
- PREFILL to COMPLETE: `writer_loop()`, context exhaustion branch (line 304).
- DECODE to COMPLETE: `reader_loop()`, `emit_token` lambda (line 479).
- COMPLETE to PREFILL: `handle_local_continue()` (line 837).
- COMPLETE to DECODE: `handle_disaggregated_continue()` (line 813).
- Any to INACTIVE: `maybe_finalize_cleanup()`, CAS (line 732).

---

**Next:** [Chapter 4 -- Speculative Decode](../ch4_speculative_decode/index.md)
