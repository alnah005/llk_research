# Result Classification

This section presents the speculative decode result classification through a worked example first, then generalizes to the formal four-case taxonomy. The example traces a complete 10-token speculative generation session tick by tick, showing INITIAL, ACCEPT, CONTINUE, REJECT, and STALE results in a realistic mix.

## Worked Example: 10-Token Speculative Generation

**Setup**: User `uid=3` has `spec_decode_enabled=true`, `max_new_tokens=10`, and has just completed prefill. The pipeline has $D = 8$ stages. The user is outside thinking phase (strict equality acceptance). Token IDs are shown as letters for readability; predictions are shown with a prime mark (e.g., the pipeline predicts `B'` alongside actual token `A`).

### Tick-by-Tick Trace

**Round 1 -- INITIAL** (tick 13, first result after prefill):

```
Result: actual=A (BASE), predicted=B', pos=5, predicted_pos=6
State before: has_unverified=false, has_verified=false
```

This is the INITIAL case (lines 530-538): neither `has_unverified` nor `has_verified` is set. The Reader:

1. Emits token `A` to `output_queue` via `emit_token(A, 6)`. `tokens_generated` = 1.
2. Stores `B'` as `unverified_token[3]`, sets `has_unverified=true`.
3. Calls `stage_pair()`: stages entry with `(token_id=A, pos=5, spec_token_id=B', spec_pos=6)`.

The Writer will inject `A` at position 5 (BASE) and `B'` at position 6 (SPEC).

**Output so far**: `[A]` (1 token)

---

**Round 2 -- ACCEPT + CONTINUE** (ticks 21-22):

Tick 21 -- BASE result arrives:
```
Result: actual=B' (BASE), predicted=C', pos=6, predicted_pos=7
State before: has_unverified=true (B'), has_verified=false
check_acceptance(3, B', result) => result.actual_token == B' => true
```

ACCEPT (lines 543-555): The BASE result's actual token `B'` matches the stored unverified prediction `B'`. The Reader:

1. Increments `spec_accepts[3]`.
2. Sets `has_verified=true`, clears `has_unverified`.
3. Emits token `B'` via `emit_token(B', 7, defer_complete=true)`. `tokens_generated` = 2.
4. Does NOT stage a pair. Waits for the SPEC result.

Tick 22 -- SPEC result arrives:
```
Result: actual=C_bonus (SPEC), predicted=D', pos=7, predicted_pos=8
State before: has_unverified=false, has_verified=true
```

CONTINUE (lines 574-595): The SPEC result after ACCEPT carries a bonus token. The Reader:

1. Clears `has_verified`, sets `has_unverified=true`.
2. Stores `D'` as `unverified_token[3]`.
3. Emits bonus token `C_bonus` via `emit_token(C_bonus, 8)`. `tokens_generated` = 3.
4. Calls `stage_pair()`: stages `(C_bonus, pos=7, D', pos=8)`.

This round produced **2 output tokens** (`B'` and `C_bonus`) from 2 pipeline slots. The match path works as designed.

**Output so far**: `[A, B', C_bonus]` (3 tokens)

---

**Round 3 -- REJECT + STALE** (ticks 30-31):

Tick 30 -- BASE result arrives:
```
Result: actual=X (BASE), predicted=E', pos=8, predicted_pos=9
State before: has_unverified=true (D'), has_verified=false
check_acceptance(3, D', result) => result.actual_token=X != D' => false
```

REJECT (lines 556-569): The actual token `X` does not match the prediction `D'`. The Reader:

1. Increments `spec_rejects[3]`.
2. Keeps `has_unverified=true`, overwrites `unverified_token[3]` with new prediction `E'`.
3. Emits token `X` via `emit_token(X, 9, defer_complete=true)`. `tokens_generated` = 4.
4. Calls `stage_pair()`: stages `(X, pos=8, E', pos=9)`.

Note the KV correction: the BASE token `X` is staged at position 8 -- the *same* position where the now-incorrect SPEC `D'` was written. This overwrites the stale KV entry. The new SPEC `E'` goes at position 9. Because the pipeline processes in order, the SPEC at position 9 will see the corrected KV at position 8. No explicit rollback is needed.

Tick 31 -- SPEC result arrives:
```
Result: actual=<garbage> (SPEC), predicted=<garbage>
State before: has_unverified=true (E'), has_verified=false
```

STALE (lines 596-606): The SPEC result arrived from a pipeline run that used the wrong prediction `D'` at position 8. Its actual token and prediction are both garbage -- computed from incorrect KV state. The Reader discards the result entirely.

This round produced **1 output token** (`X`) from 2 pipeline slots.

**Output so far**: `[A, B', C_bonus, X]` (4 tokens)

---

**Round 4 -- ACCEPT + CONTINUE** (ticks 39-40):

Tick 39 -- BASE result: actual=`E'` matches prediction. ACCEPT. Emit `E'`. `tokens_generated` = 5.

Tick 40 -- SPEC result: CONTINUE. Emit bonus `F_bonus`. `tokens_generated` = 6. Stage new pair.

**Output so far**: `[A, B', C_bonus, X, E', F_bonus]` (6 tokens)

---

**Round 5 -- ACCEPT + CONTINUE** (ticks 48-49):

Tick 48 -- BASE: ACCEPT, emit `G`. `tokens_generated` = 7.

Tick 49 -- SPEC: CONTINUE, emit bonus `H_bonus`. `tokens_generated` = 8. Stage new pair.

**Output so far**: `[A, B', C_bonus, X, E', F_bonus, G, H_bonus]` (8 tokens)

---

**Round 6 -- ACCEPT + CONTINUE with Completion** (ticks 57-58):

Tick 57 -- BASE result: actual matches prediction. ACCEPT. `emit_token(I, ctx_pos, defer_complete=true)` is called. `tokens_generated` = 9, which is less than `max_new_tokens` = 10, so `emit_token` returns false. `has_verified=true`.

Tick 58 -- SPEC result: CONTINUE. `emit_token(J_bonus, ctx_pos)`. `tokens_generated` = 10. $10 \geq 10$, so **complete**. `emit_token` sets `state=COMPLETE`, calls `stage_eos_writeback()`, `set_position_on_complete()`. Returns true. Since `emit_token` returned true for the CONTINUE path (line 590), the Reader sets `has_unverified=false` (no further speculation needed).

**Output**: `[A, B', C_bonus, X, E', F_bonus, G, H_bonus, I, J_bonus]` (10 tokens, complete)

### Example Summary

| Round | BASE Case | SPEC Case | Tokens Out | Slots Used |
|-------|-----------|-----------|------------|------------|
| 1 | INITIAL | -- | 1 | 2 |
| 2 | ACCEPT | CONTINUE | 2 | 2 |
| 3 | REJECT | STALE | 1 | 2 |
| 4 | ACCEPT | CONTINUE | 2 | 2 |
| 5 | ACCEPT | CONTINUE | 2 | 2 |
| 6 | ACCEPT | CONTINUE | 2 | 2 |
| **Total** | | | **10** | **12** |

With 4 ACCEPTs and 1 REJECT out of 5 verified rounds, the accept rate is 80%. The user consumed 12 pipeline slots (6 rounds x 2) to produce 10 tokens. Regular decode would have consumed 10 slots for 10 tokens. The spec decode advantage: 10 tokens in 6 rounds ($6D$ ticks) vs 10 tokens in 10 rounds ($10D$ ticks) -- a 1.67x speedup. This is slightly below the expected $1 + \alpha = 1.8\text{x}$ because the INITIAL round contributes only 1 token; for longer generations, the INITIAL overhead is amortized and the actual speedup converges to $1 + \alpha$.

---

## Formal Four-Case Classification

With the example as a concrete anchor, here is the complete classification. Each result for a speculative user falls into exactly one case based on two discriminators:

1. **`result.actual_token_type`** -- `TokenType::BASE` or `TokenType::SPEC` (set by the pipeline based on how the token was injected).
2. **`SpecDecodeState` flags** -- `has_unverified`, `has_verified`, and `signal_to_exit`.

### State Space

The protocol's observable state is the tuple $(U, V, S)$ where $U$ = `has_unverified`, $V$ = `has_verified`, $S$ = `signal_to_exit`. The reachable states:

| $U$ | $V$ | $S$ | State Name |
|:---:|:---:|:---:|---|
| 0 | 0 | 0 | IDLE (post-reset) |
| 1 | 0 | 0 | SPECULATING |
| 0 | 1 | 0 | ACCEPTED |
| 0 | 1 | 1 | ACCEPTED_EXIT |
| 1 | 0 | 1 | REJECTED_EXIT |

The combination $(U=1, V=1, *)$ is unreachable -- the protocol never sets both flags simultaneously. $(U=0, V=0, S=1)$ is also unreachable -- `signal_to_exit` is only set when transitioning through ACCEPT or REJECT, which always sets $V$ or keeps $U$. If `has_unverified` and `has_verified` are both true, the automaton has entered an invalid state -- a bug. The structure of the transitions makes this impossible.

### Case 1: INITIAL

**Condition**: `!has_unverified[uid] && !has_verified[uid]` (lines 530-538)

**When**: The very first decode result after prefill. No prior speculation exists.

**Actions**:
1. Store prediction: `has_unverified=true`, `unverified_token=result.predicted_token`.
2. Emit actual token via `emit_token(result.actual_token, ctx_pos)`.
3. If complete: clear `has_unverified` (no SPEC in pipeline to drain). Otherwise: `stage_pair()`.

**Transition**: $(0, 0, 0) \xrightarrow{\text{BASE}} (1, 0, 0)$

**Tokens emitted**: 1.

The INITIAL case bootstraps the speculation cycle. After this point, the pipeline has a BASE+SPEC pair in flight, and all subsequent results are classified as ACCEPT, REJECT, CONTINUE, or STALE.

### Case 2: ACCEPT

**Condition**: `result.actual_token_type == BASE && has_unverified[uid] && check_acceptance(uid, unverified_token[uid], result)` (lines 542-555)

**When**: The pipeline's BASE result confirms the stored prediction. `check_acceptance` returns true -- either strict equality (outside thinking phase) or relaxed probability-delta match (inside thinking phase; see [`thinking_phase_and_relaxed_acceptance.md`](./thinking_phase_and_relaxed_acceptance.md)).

**Actions**:
1. `spec_accepts[uid]++` (diagnostic counter).
2. `has_verified=true`, `has_unverified=false`.
3. `emit_token(result.actual_token, ctx_pos, defer_complete=true)`. If this token triggers completion, the output message is stashed in `pending_complete` rather than published immediately. The paired SPEC result must be consumed first (see [`state_machine_detail.md`](./state_machine_detail.md#the-defer_complete-mechanism)).
4. If complete: `signal_to_exit=true`.
5. Do NOT stage a new pair. Wait for the SPEC result.

**Transition**: $(1, 0, 0) \xrightarrow{\text{BASE matches}} (0, 1, 0)$

**Tokens emitted**: 1 (the verified actual token).

**KV state**: Unchanged. The KV at $P_{\text{spec}}$ was valid all along; the ACCEPT simply confirms it. The Reader waits for the SPEC result to arrive, which will carry a new prediction and a bonus output token.

### Case 3: CONTINUE

**Condition**: `result.actual_token_type == SPEC && has_verified[uid]` (lines 574-595)

**When**: After an ACCEPT, the SPEC result arrives carrying the bonus output token and a new prediction.

**Actions** (normal path, `signal_to_exit` is false):
1. `has_verified=false`, `has_unverified=true`.
2. Store new prediction: `unverified_token=result.predicted_token`.
3. `emit_token(result.actual_token, ctx_pos)` -- emits the bonus token.
4. If complete: clear `has_unverified`. Otherwise: `stage_pair()`.

**CONTINUE-suppress** (when `signal_to_exit` is true, lines 579-588):

The ACCEPT that preceded this CONTINUE already detected completion. The bonus token must NOT be emitted. Instead:
1. `signal_to_exit=false`, `has_unverified=false`, `has_verified=false`.
2. `tokens_generated[uid] = 0` (reset for next turn).
3. `flush_pending_complete()` -- pushes the stashed completion `OutputMessage` to `output_queue`.

**Transition**: $(0, 1, 0) \xrightarrow{\text{SPEC}} (1, 0, 0)$

**Tokens emitted**: 1 (bonus) or 0 (suppressed).

ACCEPT + CONTINUE together produce **2 output tokens from 2 pipeline slots** -- the optimal case.

### Case 4: REJECT

**Condition**: `result.actual_token_type == BASE && has_unverified[uid] && !check_acceptance(...)` (lines 556-569)

**When**: The BASE result shows that the model sampled a different token than what was predicted. The KV cache at the SPEC position is stale.

**KV state before**:
```
Position:  ...  P_base  P_spec
KV:             [prev actual]  [WRONG prediction -- stale KV]
                 verified       invalid
```

**Actions**:
1. `spec_rejects[uid]++` (conditioned on `has_unverified` being true).
2. `has_unverified=true` (remains true), `unverified_token=result.predicted_token`.
3. `emit_token(result.actual_token, ctx_pos, defer_complete=true)`.
4. If complete: `signal_to_exit=true`, do NOT stage a new pair. Wait for the STALE SPEC result to drain.
5. Otherwise: `stage_pair()`. The BASE targets `result.actual_token_pos` (= previous SPEC position, overwriting stale KV), and the SPEC targets `result.predicted_token_pos` (next position).

**KV state after correction pair**:
```
Position:  ...  P_base  P_spec     P_spec+1
KV:             [prev]  [CORRECTED] [new prediction]
                         BASE writes  SPEC writes
                         (overwrites stale)
```

**Transition**: $(1, 0, 0) \xrightarrow{\text{BASE mismatches}} (1, 0, 0)$

**Tokens emitted**: 1 (the correct actual token). The STALE result (Case 5) produces no output. Total for REJECT+STALE: 1 token / 2 slots.

### Case 5: STALE

**Condition**: `result.actual_token_type == SPEC && !has_verified[uid]` (lines 596-606)

**When**: After a REJECT, the SPEC result arrives computed from stale KV. It is garbage.

**Actions** (normal path, `signal_to_exit` is false):
1. Discard entirely. No token emitted, no pair staged.

**Actions** (when `signal_to_exit` is true):
1. `has_unverified=false`, `tokens_generated[uid] = 0`.
2. `flush_pending_complete()`.
3. `signal_to_exit=false`.

**Transition**: $(1, 0, 0) \xrightarrow{\text{SPEC (stale)}} (1, 0, 0)$ (no state change; the stale result is absorbed)

**Tokens emitted**: 0.

The STALE case exists solely to drain the pipeline. The REJECT already staged a correction pair, so the next BASE result will verify against the new prediction.

## Complete Case Discrimination Table

The following table shows every reachable combination of input variables and the resulting case classification:

| `has_unverified` | `has_verified` | `actual_token_type` | `check_acceptance` | `signal_to_exit` | Case | Source line |
|:---:|:---:|:---:|:---:|:---:|---|---|
| 0 | 0 | BASE | -- | -- | INITIAL | 530 |
| 1 | 0 | BASE | true | -- | ACCEPT | 543 |
| 1 | 0 | BASE | false | -- | REJECT | 557 |
| 0 | 1 | SPEC | -- | false | CONTINUE | 574 |
| 0 | 1 | SPEC | -- | true | CONTINUE-suppress | 579 |
| 1 | 0 | SPEC | -- | false | STALE (discard) | 597 |
| 1 | 0 | SPEC | -- | true | STALE (flush) | 598 |

The combinations $(U=0, V=0, \text{SPEC})$, $(U=1, V=1, *)$, and $(U=0, V=1, \text{BASE})$ are unreachable by the protocol's transition rules.

## Decision Flow Diagram

```
Result arrives for spec-enabled uid
         |
         v
  has_unverified=false &&
  has_verified=false?
         |
    yes--+--->  INITIAL: emit actual, store prediction,
         |                stage pair (or stop if complete)
    no   |
         v
  actual_token_type == BASE?
         |
    yes--+--->  check_acceptance(uid, unverified, result)?
         |            |
         |       yes--+--->  ACCEPT: emit actual (deferred),
         |            |               has_verified=true,
         |            |               wait for SPEC
         |       no---+--->  REJECT: emit actual (deferred),
         |                           new prediction,
         |                           stage correction pair
    no   |
         v
  actual_token_type == SPEC?
         |
    yes--+--->  has_verified?
         |         |
         |    yes--+--->  CONTINUE: emit bonus, new prediction,
         |         |                 stage pair (or suppress if exit)
         |    no---+--->  STALE: discard (flush if exit)
```

## Per-Round Throughput Summary

Each speculative round consists of one BASE result followed by one SPEC result, consuming 2 pipeline slots over $D$ ticks:

| Path | Tokens Out | Slots Used | Effective Throughput |
|---|---|---|---|
| ACCEPT + CONTINUE (match) | 2 | 2 | $2/D$ ticks = 2x regular |
| REJECT + STALE (mismatch) | 1 | 2 | $1/D$ ticks at 2x cost |
| ACCEPT + CONTINUE-suppress (completion) | 1 | 2 | $1/D$ ticks (final round edge case) |
| INITIAL | 1 | 2 | $1/D$ ticks (startup, amortized) |

The ACCEPT+CONTINUE path is the only path that delivers a throughput improvement. The REJECT+STALE path is strictly worse than regular decode because it consumes two pipeline slots for one output token. This is why the accept rate is the critical metric -- see [`throughput_analysis.md`](./throughput_analysis.md).

---

**Next:** [`pair_injection.md`](./pair_injection.md)
