# State Machine Detail

This section presents the complete spec-decode protocol as a formal finite automaton. Every `SpecDecodeState` field assignment is shown for every transition. The `defer_complete` mechanism is analyzed as a sub-protocol embedded within the main automaton.

The `SpecDecodeState` data structure layout and field semantics are introduced in [Chapter 2 -- SpecDecodeState](../ch2_threading_and_data_structures/spec_decode_state.md). This section focuses on the protocol-level transitions: which fields change on which event, in which order, and why.

Source: `src/scheduler/decode/spec_decode_state.hpp` (structure definition), `decode_scheduler.cpp` lines 519-606 (state machine logic in `reader_loop`).

## SpecDecodeState Fields

All fields are plain (non-atomic) types, accessed exclusively by the Reader thread during active decode. The API thread calls `reset()` only at safe transition points (ALLOCATE, SUBMIT, CONTINUE, CANCEL) when no concurrent reader access to the same slot is possible. See [Chapter 2 -- SpecDecodeState](../ch2_threading_and_data_structures/spec_decode_state.md#thread-safety-of-reset) for the thread-safety justification.

| Field | Type | Initial Value | Purpose |
|-------|------|---------------|---------|
| `unverified_token[uid]` | `uint32_t` | `EMPTY_TOKEN` | The predicted token from the previous round, awaiting verification |
| `has_unverified[uid]` | `bool` | `false` | True when `unverified_token` contains a valid prediction |
| `has_verified[uid]` | `bool` | `false` | True after ACCEPT, cleared after CONTINUE/STALE processes the paired result |
| `signal_to_exit[uid]` | `bool` | `false` | Completion detected; deferred until paired result is consumed |
| `pending_complete[uid]` | `OutputMessage` | default | Stashed completion message for deferred publication |
| `has_pending_complete[uid]` | `bool` | `false` | Gate for `flush_pending_complete()` |

## Formal State Space

The protocol's observable state is the tuple $(U, V, S)$ where:
- $U$ = `has_unverified`
- $V$ = `has_verified`
- $S$ = `signal_to_exit`

The reachable states and their names:

| $U$ | $V$ | $S$ | State Name |
|:---:|:---:|:---:|---|
| 0 | 0 | 0 | IDLE (post-reset) |
| 1 | 0 | 0 | SPECULATING |
| 0 | 1 | 0 | ACCEPTED |
| 0 | 1 | 1 | ACCEPTED_EXIT |
| 1 | 0 | 1 | REJECTED_EXIT |

The combination $(U=1, V=1, *)$ is unreachable -- the protocol never sets both flags simultaneously. $(U=0, V=0, S=1)$ is also unreachable -- `signal_to_exit` is only set when transitioning through ACCEPT or REJECT, which always sets $V$ or keeps $U$.

## Complete Transition Diagram

```
                           reset()
                              |
                              v
                    +-------------------+
                    |       IDLE        |
                    | U=0, V=0, S=0    |
                    +--------+----------+
                             |
                        BASE result
                        (INITIAL)
                             |
                             v
                    +-------------------+
         +--------->|   SPECULATING     |<-----------+
         |          | U=1, V=0, S=0    |             |
         |          +----+----+----+----+             |
         |               |    |    |                  |
         |         BASE  |    | BASE                  |
         |        match  |    | mismatch              |
         |               |    |                       |
         |               v    v                       |
         |    +---------+    +----------+             |
         |    | ACCEPTED|    | still    |             |
         |    | U=0,V=1 |    | SPECUL.  |             |
         |    | S=0     |    | U=1,V=0  |             |
         |    +----+----+    | S=0      |             |
         |         |         +----+-----+             |
         |    SPEC |              |                   |
         |    result              | SPEC result       |
         |  (CONTINUE)            | (STALE)           |
         |         |              |                   |
         +---------+              +-------------------+
              (emit bonus,              (discard,
               stage pair)               back to
                                         SPECULATING)

  === COMPLETION PATHS (signal_to_exit = true) ===

         +-------------------+
         |  ACCEPTED_EXIT    |
         |  U=0, V=1, S=1   |
         +--------+----------+
                  |
             SPEC result
          (CONTINUE-suppress)
                  |
                  v
         flush_pending_complete()
         U=0, V=0, S=0  --> COMPLETE

         +-------------------+
         |  REJECTED_EXIT    |
         |  U=1, V=0, S=1   |
         +--------+----------+
                  |
             SPEC result
             (STALE flush)
                  |
                  v
         flush_pending_complete()
         U=0, V=0, S=0  --> COMPLETE
```

## Transition Table

Each row represents one transition. The "Before" columns show the state before the event; the "After" columns show the state after processing.

### Normal Path Transitions

| # | Before $(U,V,S)$ | Event | Case | After $(U,V,S)$ | Actions |
|---|:-:|---|---|:-:|---|
| T1 | (0,0,0) | BASE | INITIAL | (1,0,0) | emit actual; store prediction; stage_pair |
| T2 | (1,0,0) | BASE match | ACCEPT | (0,1,0) | spec_accepts++; emit actual (defer); wait for SPEC |
| T3 | (0,1,0) | SPEC | CONTINUE | (1,0,0) | emit bonus; store prediction; stage_pair |
| T4 | (1,0,0) | BASE mismatch | REJECT | (1,0,0) | spec_rejects++; emit actual (defer); store new prediction; stage_pair |
| T5 | (1,0,0) | SPEC | STALE | (1,0,0) | discard (no-op) |

### Completion Path Transitions

| # | Before $(U,V,S)$ | Event | Case | After $(U,V,S)$ | Actions |
|---|:-:|---|---|:-:|---|
| T6 | (1,0,0) | BASE match + complete | ACCEPT+exit | (0,1,1) | spec_accepts++; emit actual (defer=true, stash msg); signal_to_exit=true |
| T7 | (0,1,1) | SPEC | CONTINUE-suppress | (0,0,0) | suppress bonus; flush pending_complete; tokens_generated=0 |
| T8 | (1,0,0) | BASE mismatch + complete | REJECT+exit | (1,0,1) | spec_rejects++; emit actual (defer=true, stash msg); signal_to_exit=true; NO stage_pair |
| T9 | (1,0,1) | SPEC | STALE flush | (0,0,0) | flush pending_complete; has_unverified=false; tokens_generated=0 |

### Edge Case Transitions

| # | Before $(U,V,S)$ | Event | Case | After $(U,V,S)$ | Actions |
|---|:-:|---|---|:-:|---|
| T10 | (0,0,0) | BASE + complete | INITIAL+complete | (0,0,0) | emit actual; has_unverified stays false (no SPEC to drain) |
| T11 | (0,1,0) | CONTINUE + complete on bonus | CONTINUE+complete | (0,0,0) | emit bonus (completes); has_unverified=false |

T10 covers the rare case where the very first decode token triggers completion (e.g., model immediately produces EOS). No spec pair was injected, so no SPEC result needs draining.

T11 covers the case where the bonus token from CONTINUE triggers completion (e.g., the bonus token is EOS or hits max_new_tokens). Since CONTINUE has already consumed the paired SPEC result, there is nothing left to drain.

## Per-Field Transition Tables

### INITIAL (first result after prefill)

| Field | Before | After (no completion) | After (completion) |
|---|---|---|---|
| `has_unverified` | `false` | `true` | `false` |
| `has_verified` | `false` | `false` | `false` |
| `unverified_token` | `EMPTY_TOKEN` | `result.predicted_token` | unchanged |
| `signal_to_exit` | `false` | `false` | `false` |

Source: lines 530-538. On completion during INITIAL, `has_unverified` is cleared at line 534 (no SPEC in the pipeline, so no deferral needed).

### ACCEPT (BASE matched prediction)

| Field | Before | After (no completion) | After (completion) |
|---|---|---|---|
| `has_unverified` | `true` | `false` | `false` |
| `has_verified` | `false` | `true` | `true` |
| `unverified_token` | prediction P | unchanged | unchanged |
| `signal_to_exit` | `false` | `false` | `true` |
| `has_pending_complete` | `false` | `false` | `true` |
| `pending_complete` | -- | -- | stashed `OutputMessage` |

Source: lines 542-555. `defer_complete=true` in the `emit_token` call (line 552) causes completion messages to be stashed rather than published.

### CONTINUE (SPEC after ACCEPT, normal path)

| Field | Before | After (no completion) | After (completion) |
|---|---|---|---|
| `has_unverified` | `false` | `true` | `false` |
| `has_verified` | `true` | `false` | `false` |
| `unverified_token` | -- | `result.predicted_token` | unchanged |

Source: lines 574-595. On completion, `has_unverified` is cleared at line 591.

### CONTINUE-suppress (SPEC after ACCEPT, exit path)

| Field | Before | After |
|---|---|---|
| `has_unverified` | `false` | `false` (set then cleared) |
| `has_verified` | `true` | `false` |
| `signal_to_exit` | `true` | `false` |
| `has_pending_complete` | `true` | `false` (after `flush_pending_complete()`) |
| `tokens_generated` | N | 0 |

Source: lines 579-588. The bonus token is suppressed (not emitted). The stashed completion message is flushed.

### REJECT (BASE did not match prediction)

| Field | Before | After (no completion) | After (completion) |
|---|---|---|---|
| `has_unverified` | `true` | `true` | `true` |
| `has_verified` | `false` | `false` | `false` |
| `unverified_token` | old prediction P | `result.predicted_token` | `result.predicted_token` |
| `signal_to_exit` | `false` | `false` | `true` |
| `has_pending_complete` | `false` | `false` | `true` |

Source: lines 556-569. On completion, no new pair is staged (line 565: `continue` skips `stage_pair()`).

### STALE (SPEC after REJECT, normal path)

| Field | Before | After |
|---|---|---|
| (no changes to spec state) | | Result discarded entirely |

Source: lines 596-606. The only action is `signal_to_exit=false` (line 604).

### STALE (SPEC after REJECT, exit path)

| Field | Before | After |
|---|---|---|
| `has_unverified` | `true` | `false` |
| `signal_to_exit` | `true` | `false` |
| `has_pending_complete` | `true` | `false` (after `flush_pending_complete()`) |
| `tokens_generated` | N | 0 |

Source: lines 598-604. Identical in effect to CONTINUE-suppress: drain the stale result, flush the deferred completion, reset counters.

## Steady-State Cycles

### Match Cycle (ACCEPT + CONTINUE)

```
            BASE match
(1,0,0) ──────────────> (0,1,0)
  ^    ACCEPT              |
  |                        | SPEC
  |    CONTINUE            |
  +────────────────────────+
       stage_pair
```

Each cycle consumes two pipeline results (one BASE, one SPEC) and produces two output tokens. This is the optimal path.

### Mismatch Cycle (REJECT + STALE)

```
            BASE mismatch
(1,0,0) ──────────────> (1,0,0)
  ^    REJECT              |
  |    stage_pair          | SPEC
  |                        |
  |    STALE (discard)     |
  +────────────────────────+
```

Each cycle consumes two pipeline results but produces only one output token.

### Mixed Cycle Transitions

The protocol freely transitions between match and mismatch cycles. The key invariant:

**After either CONTINUE or STALE, the state is always: `has_unverified=true, has_verified=false, signal_to_exit=false`.**

This is the neutral state from which the next BASE result's acceptance check determines the next cycle. There is no hysteresis -- a REJECT after ten ACCEPTs has the same state-machine behavior as a REJECT after ten REJECTs.

## The `defer_complete` Mechanism

### The Race Without Deferral

```
DANGEROUS TIMELINE (without deferral):

tick N:   Reader: ACCEPT detects completion
                  emit_token pushes OutputMessage{complete=true} to output_queue
                  state[uid] = COMPLETE

tick N+1: IS:     pops OutputMessage, sees completion
                  pushes CONTINUE request

tick N+2: API:    processes CONTINUE
                  spec_state.reset(uid)
                  state[uid] = PREFILL
                  prefill_queue.push(uid)

tick N+3: Writer: starts prefilling the new turn
                  prefill_in_flight[uid]++

tick N+4: Reader: receives STALE SPEC result from the OLD round
                  checks prefill_in_flight > 0 (TRUE -- from the new turn!)
                  prefill_in_flight[uid]-- (UNDERFLOW BUG)
                  discards the result, thinking it is a prefill result
```

The STALE result from the old speculation is misidentified as a prefill result from the new turn. This corrupts the `prefill_in_flight` counter.

### The Fix: Deferred Publication

```
SAFE TIMELINE (with deferral):

tick N:   Reader: ACCEPT detects completion
                  emit_token stashes OutputMessage in pending_complete
                  signal_to_exit = true
                  state[uid] = COMPLETE

tick N+1: Reader: receives STALE SPEC result
                  signal_to_exit is true
                  -> drain: flush_pending_complete()
                     pushes OutputMessage{complete=true} to output_queue
                  -> signal_to_exit = false, tokens_generated = 0

tick N+2: IS:     pops OutputMessage, sees completion
                  pushes CONTINUE request
                  (the stale SPEC has already been consumed)
```

The deferred completion is only published after the stale result has been consumed. By the time the IS sees the completion and sends a CONTINUE, the pipeline is clean for this user.

### Implementation

When `emit_token` is called with `defer_complete=true` and the token triggers completion (lines 484-487):

```cpp
if (complete && defer_complete) {
    ss.pending_complete[uid] = msg;
    ss.has_pending_complete[uid] = true;
    return true;
}
```

The `OutputMessage` is stashed instead of being pushed to `output_queue`. The completion actions (`state=COMPLETE`, `stage_eos_writeback()`, `set_position_on_complete()`) still execute (lines 478-482). Only the publication is deferred.

The `flush_pending_complete` lambda (lines 498-505):

```cpp
auto flush_pending_complete = [&]() {
    if (ss.has_pending_complete[uid]) {
        while (!output_queue.try_push(ss.pending_complete[uid])) {
            _mm_pause();
        }
        ss.has_pending_complete[uid] = false;
    }
};
```

Called from CONTINUE-suppress (line 587) and STALE exit path (line 601). After flushing, `tokens_generated[uid]` is reset to 0 (lines 586, 600), preparing the slot for a potential CONTINUE.

### Which Cases Use `defer_complete=true`?

| Case | `defer_complete` | Reason |
|------|:---:|---|
| INITIAL | `false` | No SPEC result in pipeline (no prior pair injected) |
| ACCEPT | `true` | Paired SPEC result is pending |
| REJECT | `true` | Paired stale SPEC result is pending |
| CONTINUE | `false` | The SPEC result was just consumed; no further result to drain |
| Non-spec decode | `false` | No paired results in non-speculative mode |

The rule: `defer_complete=true` whenever the pipeline contains a SPEC injection whose result has not yet been consumed by the Reader.

## `reset()` and State Machine Reinitialization

`SpecDecodeState::reset(uid)` is called by the API thread during ALLOCATE, SUBMIT, CONTINUE, and cleanup. It returns the slot to the IDLE state $(0, 0, 0)$:

```cpp
void reset(uint32_t uid) {
    unverified_token[uid] = EMPTY_TOKEN;
    has_unverified[uid]   = false;
    has_verified[uid]     = false;
    signal_to_exit[uid]   = false;
    has_pending_complete[uid] = false;
}
```

Note that `pending_complete[uid]` (the `OutputMessage` data) is not cleared -- only its gate `has_pending_complete` is set to false. The stale `OutputMessage` bytes are harmless because `flush_pending_complete` checks the gate before reading the message.

Thread safety of `reset()` is guaranteed by the lifecycle protocol: it runs only when no Reader activity exists for the slot. See [Chapter 2 -- SpecDecodeState, Thread Safety of reset()](../ch2_threading_and_data_structures/spec_decode_state.md#thread-safety-of-reset).

## State Machine Correctness Properties

The automaton satisfies the following invariants:

1. **Mutual exclusion**: `has_unverified` and `has_verified` are never both true simultaneously.
2. **Result draining**: `signal_to_exit` is cleared only after the pending SPEC/STALE result is consumed.
3. **No leaked completions**: Every deferred completion is eventually flushed (either via CONTINUE-suppress or STALE-flush).
4. **Monotonic progression**: The automaton progresses from IDLE through one of two cycles (ACCEPT+CONTINUE or REJECT+STALE) and never regresses to a previous cycle's mid-state.
5. **Clean shutdown**: On completion, the automaton reaches $(0, 0, 0)$ with `has_pending_complete = false` and `tokens_generated = 0`.

---

**Next:** [`throughput_analysis.md`](./throughput_analysis.md)
