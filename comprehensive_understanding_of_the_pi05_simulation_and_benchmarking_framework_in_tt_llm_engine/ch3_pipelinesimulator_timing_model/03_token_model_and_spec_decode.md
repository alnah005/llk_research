# 3.3 Token Model and Speculative Decode

## Overview

`PipelineSimulator` includes a synthetic token generation model that produces
deterministic or probabilistic token streams matching the wire-format contract
expected by the `DecodeScheduler`'s speculative-decode verification logic.  This
section covers the accept/reject token arithmetic, the `safeVocabModulus`
wrapping mechanism for long generations, the fixed `tokenId` deterministic mode,
and the relevance of these features to Pi0.5 deployments.

---

## 1. Token Generation Model: makeResult()

All token generation happens in the private `makeResult()` method:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:152-191
// Token model (matches MockPipeline so the spec-decode reader sees
// ACCEPT/REJECT exactly as it would against a real pipeline):
//   accept: actual = token_id + 1, predicted = actual + 1
//   reject: actual = token_id + 3, predicted = actual + 1   (BASE only)
// With acceptRate = 1.0 the simulator always ACCEPTs.
// With a fixed tokenId override, the reject roll is skipped (deterministic
// single-token mode is preserved for non-spec timing benchmarks).
// With safeVocabModulus > 0, all arithmetic wraps inside
// [safeVocabBase, safeVocabBase + safeVocabModulus) so long generations
// don't drift into stop-token ids.
ResultDescriptor makeResult(const InjectDescriptor& desc) {
    ResultDescriptor r{};
    r.slot_id = desc.slot_id;
    if (desc.prefill_token_id != EMPTY_TOKEN) {
        return r;
    }
    bool accept = true;
    if (tokenId == EMPTY_TOKEN && desc.token_type == TokenType::BASE && acceptRate < 1.0f) {
        accept = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < acceptRate;
    }
    r.actual_token_type = desc.token_type;
    r.actual_token_pos = desc.position + 1;
    r.predicted_token_type = TokenType::SPEC;
    r.predicted_token_pos = desc.position + 2;
    if (tokenId != EMPTY_TOKEN) {
        r.actual_token = tokenId;
        r.predicted_token = r.actual_token + 1;
    } else if (safeVocabModulus > 0) {
        uint32_t offset = (desc.token_id - safeVocabBase) % safeVocabModulus;
        uint32_t delta = accept ? 1u : 3u;
        uint32_t actual_off = (offset + delta) % safeVocabModulus;
        r.actual_token    = safeVocabBase + actual_off;
        r.predicted_token = safeVocabBase + ((actual_off + 1u) % safeVocabModulus);
    } else {
        r.actual_token    = accept ? desc.token_id + 1 : desc.token_id + 3;
        r.predicted_token = r.actual_token + 1;
    }
    return r;
}
```

### Prefill Early Return

For non-last prefill tokens (`desc.prefill_token_id != EMPTY_TOKEN`), the
method returns a minimal `ResultDescriptor` with only `slot_id` set (line
165-167).  The reader discards these results via the `prefill_in_flight`
counter (see Section 3.2).

### Position Arithmetic

Every decode result advances positions by fixed offsets:

```cpp
r.actual_token_pos = desc.position + 1;     // line 174
r.predicted_token_pos = desc.position + 2;  // line 176
```

This models a pipeline that writes the "actual" sampled token at `position + 1`
in the KV cache and the speculative prediction at `position + 2`.

---

## 2. Accept and Reject Paths

The simulator produces token values that match the contract expected by
`DecodeScheduler`'s `check_acceptance()` function.  Two paths exist:

### Accept Path

When the BASE token is accepted (either `acceptRate >= 1.0`, or the random roll
succeeds):

$$
\text{actual} = \text{token\_id} + 1
$$

$$
\text{predicted} = \text{actual} + 1 = \text{token\_id} + 2
$$

The scheduler's spec-decode verifier sees that `predicted` from the previous
turn matches `actual` from this turn, confirming the speculation was correct.

### Reject Path

When the BASE token is rejected (`accept == false`):

$$
\text{actual} = \text{token\_id} + 3
$$

$$
\text{predicted} = \text{actual} + 1 = \text{token\_id} + 4
$$

The +3 offset ensures the actual token **does not match** the previous turn's
prediction (which was `token_id + 2`), triggering a spec-decode rejection.  The
verifier then discards the speculative KV-cache entry and re-issues from the
actual token.

### Why +1 and +3 (not +1 and +2)?

The gap between the accept delta (+1) and reject delta (+3) is 2.  This is
deliberate: the predicted token for an accept is `token_id + 2`, so the reject
actual must differ from `token_id + 2`.  Using +3 guarantees a mismatch:

$$
(\text{token\_id} + 3) \ne (\text{token\_id} + 2) \quad \text{for all } \text{token\_id}
$$

A +2 reject delta would collide with the accept prediction, making the verifier
unable to distinguish accept from reject.

---

## 3. Accept Rate Probability

The accept/reject roll is gated by three conditions:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:170
if (tokenId == EMPTY_TOKEN && desc.token_type == TokenType::BASE && acceptRate < 1.0f) {
    accept = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < acceptRate;
}
```

| Condition | Meaning |
|-----------|---------|
| `tokenId == EMPTY_TOKEN` | Not in fixed-token deterministic mode |
| `desc.token_type == TokenType::BASE` | Only BASE tokens can be rejected; SPEC tokens always "accept" (they carry the speculation forward) |
| `acceptRate < 1.0f` | Optimization: skip the PRNG entirely when accept rate is 100% |

When all three conditions hold, a uniform random float in [0, 1) is drawn from
`std::mt19937` (seeded with `seed` at construction).  The token is accepted if
the draw is less than `acceptRate`.

### Typical Values

| Scenario | `acceptRate` | Effect |
|----------|-------------|--------|
| No spec-decode | 1.0 (default) | Always accept; reject path is dead code |
| High-quality speculation | 0.85 - 0.95 | ~85-95% of BASE tokens accept |
| Stress test | 0.5 | 50/50 accept/reject for worst-case verification |
| Always reject | 0.0 | Every BASE token rejects; no spec-decode benefit |

---

## 4. safeVocabModulus: Wrapping Token Arithmetic

### The Drift Problem

Without modular wrapping, the accept path increments token IDs by +1 on every
turn.  After $N$ decode steps:

$$
\text{token\_id}_N = \text{token\_id}_0 + N
$$

For long generations (thousands of tokens), the token ID can drift into ranges
occupied by special tokens (EOS, PAD, etc.) or even approach `EMPTY_TOKEN`
($2^{32} - 1$), causing spurious completion signals in the scheduler.

### The Modular Solution

When `safeVocabModulus > 0`, all token arithmetic wraps inside a ring of size
`safeVocabModulus` offset by `safeVocabBase`:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:180-185
uint32_t offset = (desc.token_id - safeVocabBase) % safeVocabModulus;
uint32_t delta = accept ? 1u : 3u;
uint32_t actual_off = (offset + delta) % safeVocabModulus;
r.actual_token    = safeVocabBase + actual_off;
r.predicted_token = safeVocabBase + ((actual_off + 1u) % safeVocabModulus);
```

The token ID stays within $[\text{safeVocabBase},\; \text{safeVocabBase} + \text{safeVocabModulus})$:

$$
\text{actual} = \text{safeVocabBase} + \bigl((\text{offset} + \delta) \bmod \text{safeVocabModulus}\bigr)
$$

$$
\text{predicted} = \text{safeVocabBase} + \bigl((\text{offset} + \delta + 1) \bmod \text{safeVocabModulus}\bigr)
$$

### Minimum Modulus Constraint

The constructor enforces `safeVocabModulus >= 5` when non-zero:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:72-75
if (safeVocabModulus != 0 && safeVocabModulus < 5) {
    throw std::invalid_argument(
        "PipelineSimulator: safe_vocab_modulus must be 0 or >= 5");
}
```

The comment explains why (lines 69-71):

```
// Modulus must be 0 (disabled) or >= 5. Smaller rings collapse the
// +1/+3 offsets into ambiguous values that fool the spec-decode
// verification check.
```

To see why, consider the five distinct token values that must be distinguishable
in a single verification cycle:

| Offset | Expression | Meaning |
|--------|-----------|---------|
| +0 | $t$ | Input token |
| +1 | $(t + 1) \bmod M$ | Accept actual |
| +2 | $(t + 2) \bmod M$ | Accept predicted |
| +3 | $(t + 3) \bmod M$ | Reject actual |
| +4 | $(t + 4) \bmod M$ | Reject predicted |

For these to remain pairwise distinct for all $t$, we need $M \ge 5$.  With
$M = 4$, the reject predicted $(t + 4) \bmod 4 = t$, colliding with the input
token.  With $M = 3$, $(t + 3) \bmod 3 = t$, making reject actual
indistinguishable from the input -- the verifier could not distinguish a
rejection from a no-op.

### Verification Correctness Under Wrapping

Spec-decode verification in the scheduler compares the previous turn's
`predicted_token` with the current turn's `actual_token`.  Under modular
wrapping:

- **Accept**: previous predicted = $\text{base} + ((o + 2) \bmod M)$, current
  actual = $\text{base} + ((o' + 1) \bmod M)$ where $o' = (o + 1) \bmod M$.
  So current actual = $\text{base} + ((o + 2) \bmod M)$ = previous predicted.
  Match confirmed.

- **Reject**: previous predicted = $\text{base} + ((o + 2) \bmod M)$, current
  actual = $\text{base} + ((o' + 3) \bmod M)$ where $o' = (o + 1) \bmod M$.
  So current actual = $\text{base} + ((o + 4) \bmod M)$.  Since $M \ge 5$,
  $(o + 4) \bmod M \ne (o + 2) \bmod M$, so this never matches. Mismatch
  triggers rejection.

### Worked Example: Wrapping in Practice

Consider `safeVocabBase = 1000`, `safeVocabModulus = 1000`, initial
`token_id = 1000`:

| Step | Input offset | Accept? | Delta | Actual offset | Actual token | Predicted token |
|---|---|---|---|---|---|---|
| 0 | 0 | yes | +1 | 1 | 1001 | 1002 |
| 1 | 1 | yes | +1 | 2 | 1002 | 1003 |
| ... | ... | ... | ... | ... | ... | ... |
| 998 | 998 | yes | +1 | 999 | 1999 | 1000 |
| 999 | 999 | yes | +1 | 0 | 1000 | 1001 |

At step 999, the token wraps back to 1000 -- safely within the `[1000, 2000)`
range.  Without wrapping, step 999 would produce token 1999, and step 1000
would produce 2000, potentially colliding with reserved token ranges.

The test suite exercises this with explicit safe-vocab parameters:

```cpp
// tests/scheduler/decode/test_decode_scheduler.cpp:2035-2043
pl::PipelineSimulatorConfig pipeline_config{
    .num_stages = 64,
    .stage_duration_us = 5000,
    .decode_token_id = EMPTY_TOKEN,  // EMPTY_TOKEN = honor accept_rate
    .accept_rate = 0.5f,             // mix accept/reject branches
    .seed = 42,
    .safe_vocab_base = 1000,
    .safe_vocab_modulus = 1000,      // keep tokens out of EOS/EMPTY range
};
```

---

## 5. Fixed tokenId Override: Deterministic Mode

When `tokenId != EMPTY_TOKEN`, the simulator bypasses the mock token model
entirely and emits the same fixed token on every decode step:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:178-179
if (tokenId != EMPTY_TOKEN) {
    r.actual_token = tokenId;
    r.predicted_token = r.actual_token + 1;
}
```

### Properties of Fixed-Token Mode

1. **No accept/reject roll** -- the condition on line 170 requires
   `tokenId == EMPTY_TOKEN`, so the accept/reject PRNG is never invoked.

2. **Constant output** -- every decode step produces `actual = tokenId`,
   `predicted = tokenId + 1`.  The scheduler sees the same token repeatedly,
   which is useful for non-spec timing benchmarks where token content is
   irrelevant.

3. **No drift** -- since the actual token never changes, there is no risk of
   drifting into stop-token IDs.  `safeVocabModulus` is ignored.

4. **Spec-decode behavior** -- with a fixed token, the predicted value from
   turn $N$ is `tokenId + 1`, but the actual value from turn $N+1$ is
   `tokenId`, not `tokenId + 1`.  This means **spec-decode always rejects**
   in fixed-token mode (unless the scheduler is configured with
   `spec_decode = false`).

### Usage in Tests

Test cases use fixed tokens for deterministic validation:

```cpp
// tests/scheduler/decode/test_decode_scheduler.cpp:1582-1586
pl::PipelineSimulatorConfig pipeline_config{
    .num_stages = 64,
    .stage_duration_us = 10000,
    .decode_token_id = 12345,
};
```

The fixed token `12345` produces a predictable output stream that tests can
assert against without worrying about PRNG state.

---

## 6. Relevance to Pi0.5

### Single Output Token Per Turn

Pi0.5 deployments typically operate with `output_tokens = 1` per scheduler turn
and speculative decode disabled (`spec_decode = false`).  In this mode:

- Only BASE tokens are injected (no SPEC tokens).
- The accept/reject logic is never exercised (no SPEC verification).
- `makeResult()` always takes the accept path, producing
  `actual = token_id + 1`, `predicted = token_id + 2`.
- The scheduler uses the non-spec-decode path in `reader_loop()`:

```cpp
// src/scheduler/decode/decode_scheduler.cpp:549-557
if (!user_table.spec_decode_enabled[uid]) {
    ctx_pos = result.actual_token_pos;
    if (!emit_token(result.actual_token, ctx_pos, result.actual_token_pos, false)) {
        decode_staging.stage(uid,
            result.actual_token, result.actual_token_pos,
            EMPTY_TOKEN, 0);
    }
    continue;
}
```

Note that when `spec_decode_enabled` is false, `decode_staging.stage()` passes
`EMPTY_TOKEN` for the spec token (line 555), and only the BASE inject fires in
`writer_loop()` (the `do_spec` check at line 246 evaluates to false).

### Pi0.5 Pipeline Runner Defaults

The Pi0.5 runner enables `batch_prefill = true` with spec-decode disabled at
the scheduler level (see [Section 3.1](./01_simulator_design.md), Pi0.5
Pipeline Runner Usage for the full config snippet).

- `decode_token_id` defaults to `EMPTY_TOKEN` (mock model active).
- `accept_rate` defaults to 1.0 (always accept, but irrelevant since
  spec-decode is disabled at the scheduler level).
- `safe_vocab_base` and `safe_vocab_modulus` default to 0 (no wrapping).

For Pi0.5 throughput benchmarks, the token content is irrelevant -- what matters
is the timing model.  The token generation just needs to produce non-EOS values
that keep the scheduler's decode loop running for the requested number of
output tokens.

### When Spec Decode Would Matter

The token model becomes relevant beyond Pi0.5's default configuration when:

- Testing the scheduler's spec-decode state machine with `accept_rate < 1.0f`.
- Benchmarking multi-token speculative decoding where BASE and SPEC token pairs
  are injected.
- Stress-testing the STOP/CONTINUE lifecycle with mixed accept/reject sequences
  (as in the `StopMidDecodeSpecModeDiscardsAndAcks` test).

---

## 7. Condition Variable Notification in read_result()

The `read_result()` method uses a two-phase notification pattern that
coordinates the reader and writer threads:

### Phase 1: Wait for Token Availability (emitCv)

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:111-112
emitCv.wait(lock, [this] {
    return !inflight.empty() || stop.load(std::memory_order_acquire);
});
```

The reader blocks on `emitCv` until at least one token is in the FIFO.
`inject()` calls `emitCv.notify_one()` after pushing a token (line 105 for
normal path, line 85 for batch-prefill path).

### Phase 2: Signal Backpressure Release (injectCv)

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:135-136
inflight.pop_front();
injectCv.notify_one();
```

After popping a token, the reader notifies `injectCv` to wake any writer
blocked on backpressure (`inflight.size() >= numStages`).

### Notification Flow Diagram

```
Writer Thread                          Reader Thread
    |                                      |
    |  inject()                            |  read_result()
    |    |                                 |    |
    |    +-- lock(mu)                      |    +-- lock(mu)
    |    +-- wait(injectCv)  <------+      |    +-- wait(emitCv)  <---+
    |    |   [backpressure]         |      |    |   [empty FIFO]      |
    |    +-- push to inflight       |      |    +-- peek exitTime     |
    |    +-- emitCv.notify_one() ---+------+    +-- unlock(mu)        |
    |    +-- unlock(mu)             |      |    +-- busy-wait         |
    |                               |      |    +-- lock(mu)          |
    |                               |      |    +-- pop_front()       |
    |                               +------+--- injectCv.notify_one() |
    |                                      |    +-- return result     |
```

### Producer-Consumer Summary

| Condition Variable | Notified by | Waited on by | Condition |
|---|---|---|---|
| `emitCv` | `inject()` | `read_result()` | `!inflight.empty()` |
| `injectCv` | `read_result()` | `inject()` | `inflight.size() < numStages` |

This dual-CV design avoids spurious wakeups: the injector only wakes when space
is available, and the reader only wakes when data is available.

### Stop Semantics

`request_stop()` notifies **both** condition variables with `notify_all()`:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:142-147
void request_stop() override {
    stop.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(mu);
    emitCv.notify_all();
    injectCv.notify_all();
}
```

Both `inject()` and `read_result()` check `stop` in their wait predicates and
return sentinel values (`INVALID_SLOT` for read_result, early return for
inject).  The memory ordering is:

- **`request_stop()`**: `release` store to `stop`, then `notify_all` under lock.
- **Wait predicates**: `acquire` load from `stop`, ensuring they see the flag
  after the release store.
- **Busy-wait loop**: `acquire` load on each iteration (line 128), allowing
  prompt exit during the spin phase.

This guarantees that once `request_stop()` returns, no thread can remain blocked
indefinitely in either `inject()` or `read_result()`.

---

## 8. Summary of Token Generation Modes

| Mode | tokenId | acceptRate | safeVocabModulus | Behavior |
|---|---|---|---|---|
| Deterministic | Fixed value | Ignored | Ignored | All tokens = `tokenId`, no accept/reject |
| Accept-only (default) | `EMPTY_TOKEN` | 1.0 | 0 | `actual = token_id + 1`, no wrapping |
| Accept-only + wrap | `EMPTY_TOKEN` | 1.0 | >= 5 | `actual = base + (offset + 1) % M`, long-generation safe |
| Mixed accept/reject | `EMPTY_TOKEN` | < 1.0 | 0 | Accept: +1, Reject: +3, unbounded growth |
| Mixed + wrap | `EMPTY_TOKEN` | < 1.0 | >= 5 | Accept: +1, Reject: +3, modular wrap |

Pi0.5 typically operates in the "Accept-only (default)" mode, with
`batch_prefill = true` handling the prefill phase.

---

**Previous:** [Batch Prefill Behavior](02_batch_prefill_behavior.md)

---

**Next:** [Chapter 4 -- Bulk H2D/D2H Socket Protocol](../ch4_bulk_h2d_d2h_socket_protocol/index.md)
