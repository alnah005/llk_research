# Context Exhaustion

[< Deferred Cancellation](deferred_cancellation.md) | [IS Integration Patterns >](is_integration_patterns.md)

## Overview

Every model has a maximum sequence length (`max_seq_len`). When a user's KV cache position reaches this limit, no more tokens can be generated. This is **context exhaustion** -- the model's context window is full.

Context exhaustion is a normal termination condition, not an error. It is signaled to the IS through the `ctx_exhausted` flag on `OutputMessage`, allowing the IS to decide what to do (e.g., inform the user, start a new session with summarization, or end the conversation).

The system handles context exhaustion at two different points, for defense in depth:

| Detection Point | Thread | Phase | Purpose |
|----------------|--------|-------|---------|
| Decode-side (primary) | Reader | Token emission | Normal runtime detection during decode |
| Prefill-side (defensive) | Writer | Pre-check before device submit | Prevents submitting a prefill that would exceed limits |

## Worked Example A: Exhaustion During Decode

### Setup

Continuing the slot-7 example from previous sections:

| Parameter | Value |
|-----------|-------|
| Slot ID | 7 |
| `max_seq_len` | 2048 |
| Position entering turn 4 | 2020 (accumulated from 3 prior turns) |
| Turn 4 prompt | 16 tokens |
| Position after turn 4 prefill | 2036 |
| `max_new_tokens` (turn 4) | 64 |

After prefill, there are $2048 - 2036 = 12$ positions remaining in the KV cache. The decode loop can generate at most 12 tokens before hitting the wall, even though `max_new_tokens` is 64.

### Decode Trace: Approaching the Limit

```
Decode step 1:  position 2036 -> 2037, tokens_generated = 1    ok
Decode step 2:  position 2037 -> 2038, tokens_generated = 2    ok
...
Decode step 11: position 2046 -> 2047, tokens_generated = 11   ok
Decode step 12: position 2047 -> 2048, tokens_generated = 12   LIMIT
```

### The `emit_token` Check

On each decode step, the Reader evaluates the completion condition:

$$\text{is\_complete} = \begin{cases}
\text{true} & \text{if } \texttt{tokens\_generated} \geq \texttt{max\_new\_tokens} \\
\text{true} & \text{if } \texttt{token\_id} = \texttt{EOS} \land \lnot\texttt{ignore\_eos} \\
\text{true} & \text{if } \texttt{ctx\_pos} \geq \texttt{max\_seq\_len}\\
\text{false} & \text{otherwise}
\end{cases}$$

At step 12, $\texttt{ctx\_pos} = 2048 \geq \texttt{max\_seq\_len} = 2048$, so `hit_ctx_limit = true`. Since `tokens_generated = 12 < max_new_tokens = 64`, this is clearly a context exhaustion, not a normal `max_new_tokens` completion.

### OutputMessage for Context Exhaustion

```cpp
OutputMessage{
    .slot_id         = 7,
    .token_id        = <last generated token>,
    .is_complete     = true,
    .ctx_exhausted   = true,      // distinguishes from EOS/max_new
    .tokens_generated = 12,       // only 12 of the requested 64
    .generation      = G
};
```

This is distinct from the other completion conditions:

| Completion Cause | `is_complete` | `ctx_exhausted` | How IS Distinguishes |
|-----------------|---------------|-----------------|---------------------|
| EOS token | `true` | `false` | Model chose to stop |
| `max_new_tokens` reached | `true` | `false` | `tokens_generated == max_new_tokens` |
| Context exhausted | `true` | `true` | `ctx_exhausted` flag |

The distinction matters to the IS because the appropriate response differs:
- **EOS**: Natural completion. The IS can deliver the response and offer CONTINUE.
- **max_new_tokens**: The IS may offer to continue or simply deliver what was generated.
- **Context exhausted**: A `CONTINUE` on this slot will immediately hit the limit again (no room for new tokens). The IS must either start a fresh session or implement context management.

## Worked Example B: Exhaustion During Prefill

### Setup

| Parameter | Value |
|-----------|-------|
| Slot ID | 7 |
| `max_seq_len` | 2048 |
| Position entering turn 5 | 2030 |
| Turn 5 prompt | 32 tokens |

The new prompt (32 tokens) would bring the total to $2030 + 32 = 2062$, which exceeds `max_seq_len = 2048`. Only $2048 - 2030 = 18$ of the 32 tokens can fit.

### Writer-Side Detection

The Writer performs a defensive pre-check before submitting prefill chunks to the device:

```
if (device_pos >= max_seq_len):
    mark user as COMPLETE with ctx_exhausted = true
```

At token index 18 (0-indexed), `device_pos = 2030 + 18 = 2048 >= 2048`, triggering the check. The last 14 prompt tokens ($32 - 18 = 14$) are simply not written.

### State Transition

The state machine takes a shortcut -- this is the only path where PREFILL transitions directly to COMPLETE without passing through DECODE:

```
PREFILL --(ctx_exhausted)--> COMPLETE    (skips DECODE entirely)
```

The IS receives a completion message with `ctx_exhausted = true` and `tokens_generated = 0` (the session never entered decode).

## Responsibility Split

The two detection points cover different phases of the pipeline:

```
                                    max_seq_len boundary
                                           |
  +--------------------------------------- | ----------+
  |              KV Cache                  |           |
  |  [0] [1] [2] ... [pos-1] [pos] ...    | [max-1]   |
  +--------------------------------------- | ----------+
                                           |
       <-- Prefill writes here -->         |  <-- Overflow zone
       Writer checks: will my next         |
       chunk exceed this boundary?         |
                                           |
       <-- Decode reads/writes here ------>|
       Reader checks: did this decode      |
       step reach the boundary?            |
```

| Scenario | Detected By | Phase |
|----------|-------------|-------|
| Decode token pushes position to `max_seq_len` | Reader | During `emit_token` after device output |
| Prefill chunk would push position past `max_seq_len` | Writer | Before submitting chunk to device |
| `CONTINUE` prompt tokens exceed remaining space | Writer | During prefill of continuation tokens |
| Normal decode gradually reaches limit | Reader | Token-by-token during decode |

## Position Arithmetic: Multi-Turn Budget Tracking

At any point, the remaining context budget is:

$$B = \texttt{max\_seq\_len} - \texttt{current\_position}$$

For the slot-7 example across multiple turns:

| Turn | Action | Position After | Budget |
|------|--------|---------------|--------|
| 1 | Prefill 128 + Decode 64 | 192 | 1856 |
| 2 | Prefill 32 + Decode 48 | 272 | 1776 |
| 3 | Prefill 24 + Decode 56 | 352 | 1696 |
| 4 | Prefill 16 + Decode 12 (exhausted!) | 2048 | 0 |

## Interaction with Speculative Decoding

With speculative decoding enabled, the decode pipeline may generate multiple candidate tokens per step. Context exhaustion is checked via the same `ctx_pos >= max_seq_len` condition in `emit_token`, where `ctx_pos` is `result.predicted_token_pos` -- the farthest-ahead position from the speculative result. The check is performed at the token emission level (Reader thread), not at the batch planning level.

This ensures the KV cache is never written beyond `max_seq_len`, even speculatively, because any token whose predicted position reaches the limit triggers the `ctx_exhausted` flag.

## Interaction with Multi-Turn Continue

After a `COMPLETE` with `ctx_exhausted = true`, the IS can still send a `CONTINUE` request. However:

- **Local continue:** The `prefill_pos` will start at the current position (near `max_seq_len`). The new tokens will push past the limit, and the Writer's defensive check will immediately mark COMPLETE again. The IS gains nothing from this.
- **Disaggregated continue:** The position is set based on `req.tokens.size()`. If this exceeds `max_seq_len`, the same problem occurs.

The IS is responsible for tracking context utilization and avoiding futile continuations. The scheduler does not reject a `CONTINUE` on an exhausted context -- it processes it and lets the Writer/Reader detect the overflow. This keeps the scheduler's request-handling path simple and uniform.

## Interaction with Cancellation

Context exhaustion and cancellation are orthogonal. A user can hit context exhaustion while a cancel is pending. In this case:

- The Writer/Reader context check transitions to `COMPLETE`.
- The cancel protocol's `maybe_finalize_cleanup` will eventually transition from `COMPLETE` to `INACTIVE`.
- The cancel ack is sent, not a context-exhaustion output message (the cancel takes priority since it was requested explicitly).

The generation counter ensures that any `ctx_exhausted` output messages from a stale generation are discarded by the IS. See [IS Integration Patterns](is_integration_patterns.md) for the IS-side filtering logic.

## Edge Cases

**Prompt exactly fills context.** If the initial prompt is exactly `max_seq_len` tokens, the Writer detects context exhaustion at the end of prefill (when `device_pos >= max_seq_len`). The state transitions from PREFILL directly to COMPLETE -- it never enters DECODE. The OutputMessage has `tokens_generated = 0` and `ctx_exhausted = true`.

**Prompt exceeds context.** If the IS sends a prompt longer than `max_seq_len`, truncation happens at `prompt_table.store()` time (during SUBMIT processing in the api_thread), which clamps the length: `uint32_t clamped = std::min(len, max_seq_len_)`. The Writer processes whatever was stored.

**Single token remaining.** If `max_seq_len - position = 1`, exactly one decode step can execute. The generated token's OutputMessage has both `is_complete = true` and `ctx_exhausted = true`.

## IS-Side Handling Recommendations

When the IS receives `ctx_exhausted = true`:

1. **Do not send CONTINUE** for this slot without context management.
2. **Option A: Fresh session.** Send `CANCEL` to free the slot, `ALLOCATE` a new one, and `SUBMIT` a summarized version of the conversation.
3. **Option B: Sliding window.** If implemented at the IS level, construct a new prompt from the last $N$ tokens and submit to a fresh slot.
4. **Option C: Inform the user.** Let the client know the context limit was reached.

The planned KV Cache Tiering system (see [KV Cache Tiering](kv_cache_tiering.md)) would provide more sophisticated options, including automatic eviction of older KV entries to make room for new context.

---

*[< Deferred Cancellation](deferred_cancellation.md) | [IS Integration Patterns >](is_integration_patterns.md)*
