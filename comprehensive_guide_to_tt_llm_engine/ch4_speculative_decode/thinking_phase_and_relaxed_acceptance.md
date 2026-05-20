# Thinking Phase and Relaxed Acceptance

Models that support chain-of-thought reasoning emit "thinking" tokens bracketed by special open/close markers (e.g., `<think>` / `</think>`). During the thinking phase, the model's token distribution is higher-entropy -- the next token is less predictable. Strict equality matching for speculative decode would reject most thinking-phase predictions, wasting pipeline bandwidth. The engine addresses this with two linked mechanisms: thinking-phase tracking (which changes device sampling parameters) and relaxed acceptance (which uses a probability-delta check instead of strict equality).

## Thinking-Phase Token Configuration

The `SchedulerParams` structure carries the token IDs that bracket the thinking phase (`decode_types.hpp`, lines 37-41):

```cpp
uint32_t think_open_token_id  = EMPTY_TOKEN;   // e.g., <think> token
uint32_t think_close_token_id = EMPTY_TOKEN;    // e.g., </think> token
```

Default: `EMPTY_TOKEN` (sentinel value `UINT32_MAX`). When set to `EMPTY_TOKEN`, no emitted token ever matches the sentinel, effectively disabling thinking-phase tracking entirely. The engine then operates in strict-equality mode with $k = 1$ sampling for all decode tokens.

These IDs are auto-detected from the tokenizer by the `deepseek_inference_runner` tool: it encodes `<think>` and `</think>` and verifies each encodes to exactly one token ID. If either encodes to multiple tokens, the thinking-phase feature is disabled.

## Per-User `in_thinking_phase` Tracking

Each user has an `atomic<bool>` flag in `UserTable` (`user_table.hpp`, line 42):

```cpp
std::vector<std::atomic<bool>> in_thinking_phase;
```

This is the only spec-decode-related field that crosses the Reader-Writer thread boundary:

- **Reader writes** (release store): toggled in `emit_token` before any staging call (lines 457-461).
- **Writer reads** (acquire load): checked in `device_sampling_params()` to select sampling mode (line 772).

The release-acquire pair ensures the Writer sees the most recent phase transition before constructing the next inject's sampling parameters.

### Toggle Logic

Inside `emit_token` (lines 457-461):

```cpp
if (params.think_open_token_id != EMPTY_TOKEN && tok == params.think_open_token_id) {
    user_table.in_thinking_phase[uid].store(true, std::memory_order_release);
} else if (params.think_close_token_id != EMPTY_TOKEN && tok == params.think_close_token_id) {
    user_table.in_thinking_phase[uid].store(false, std::memory_order_release);
}
```

The toggle happens at the very top of `emit_token`, BEFORE the completion check, BEFORE `stage_eos_writeback`, and BEFORE any follow-on staging. This ordering is an invariant: the next injection must sample with the correct $k$ and $\text{top\_p}$ for the phase the user is in after emitting this token.

### EMPTY_TOKEN Sentinel Disables the Gate

When `think_open_token_id == EMPTY_TOKEN`, the condition `params.think_open_token_id != EMPTY_TOKEN` is false, so the store is never reached. No emitted token can match `EMPTY_TOKEN` (which is `UINT32_MAX`), so the `in_thinking_phase` flag remains at its initial value of `false`. This makes the feature zero-cost when disabled: no branch taken, no atomic store.

### Initialization and Reset

`in_thinking_phase` is initialized to `false` for all slots in the `UserTable` constructor (line 73). It is explicitly set to `false` on:

- `user_table.reset(uid)` (line 94) -- called on ALLOCATE.
- SUBMIT processing (line 664) -- explicit store before entering prefill.
- CONTINUE processing (local: line 848, disaggregated: line 823) -- reset for each new turn.

Every turn starts outside the thinking phase, regardless of where the previous turn ended.

## Device Sampling Parameters

The Writer computes sampling parameters per-user via `device_sampling_params()` (lines 770-778):

```cpp
DeviceSamplingParams device_sampling_params(uint32_t uid) const {
    const float t = user_table.temperature[uid];
    const bool in_thinking = user_table.in_thinking_phase[uid].load(
        std::memory_order_acquire);
    if (!in_thinking) {
        return {t, 1.0f, 1};   // top_p=1.0, top_k=1 --> argmax
    }
    int32_t k = user_table.top_k[uid];
    return {t, user_table.top_p[uid], k};  // user's params
}
```

### Outside Thinking Phase: Argmax Sampling

- `top_k = 1`: the device returns only the single highest-probability token in `p_indices[0]`.
- `top_p = 1.0`: no top-P filtering (moot with $k=1$).
- Temperature is passed through but does not change the argmax result (the highest-logit token is invariant under monotonic temperature scaling).

With $k = 1$, the device skips the top-K sorting kernel beyond the first element. The `p_indices` and `p_scores` arrays in the `ResultDescriptor` contain only one valid entry. This saves device cycles and reduces data traffic.

### Inside Thinking Phase: User-Configured Sampling

- `top_k`: the user's configured value (from `GenerationParams`, stored in `UserTable`).
- `top_p`: the user's configured value.
- Temperature passed through.

The device performs full top-K sorting and top-P rescaling, populating `p_indices[0..k)` and `p_scores[0..k)` with the top-$k$ tokens and their rescaled probabilities. The `p_indices` array in `ResultDescriptor` has a fixed bound of 32 entries (`std::array<uint32_t, 32>`), capping the effective $k$ at 32.

## `check_acceptance()`: The Dual-Mode Verifier

The `check_acceptance` function (lines 782-807) determines whether the previous round's speculation should be accepted:

```cpp
bool check_acceptance(uint32_t uid,
                      uint32_t prev_spec_token_id,
                      const ResultDescriptor& result) const {
    if (!user_table.in_thinking_phase[uid].load(std::memory_order_acquire)) {
        return result.actual_token == prev_spec_token_id;
    }
    // ... relaxed acceptance path ...
}
```

### Strict Mode (Outside Thinking Phase)

```cpp
return result.actual_token == prev_spec_token_id;
```

Simple equality check. The prediction is accepted if and only if it equals the actual (argmax-sampled) token. No probability arrays are consulted.

### Relaxed Mode (Inside Thinking Phase)

```cpp
const float delta = user_table.relaxed_acceptance_threshold[uid];
const int32_t k = device_sampling_params(uid).top_k;

int idx = -1;
for (int i = 0; i < k; ++i) {
    if (result.p_indices[i] == prev_spec_token_id && result.p_scores[i] != 0) {
        idx = i;
        break;
    }
}
if (idx < 0) {
    return false;   // prediction not in top-k or filtered out
}
const float p_max = bf16_to_float(result.p_scores[0]);
const float p_draft = bf16_to_float(result.p_scores[idx]);
return (p_max - p_draft) <= delta;
```

Step by step:

1. **Retrieve the threshold**: `relaxed_acceptance_threshold[uid]` is a per-user `float` set via `GenerationParams` during SUBMIT/CONTINUE. A threshold of 0.0 degenerates to "predicted token must be the argmax."

2. **Scan `p_indices[0..k)`**: search for `prev_spec_token_id` in the top-$k$ candidates returned by the device. Entries beyond $k$ are garbage per the pipeline wire format.

3. **Skip zero-score entries**: `result.p_scores[i] != 0` filters out entries that the device zeroed during top-P filtering. A score of 0 means the token was removed by the top-P cutoff -- it should not be matched even if its index happens to equal the prediction.

4. **Not found**: If the predicted token is not in the top-$k$ (or was filtered out), the prediction is rejected.

5. **Probability delta**: Compare $p_{\max}$ (always at index 0) against $p_{\text{draft}}$ (at index `idx`):

$$\text{accept} = (p_{\max} - p_{\text{draft}}) \leq \delta$$

If the predicted token's probability is within $\delta$ of the maximum, it is accepted.

## Worked Examples: Relaxed Acceptance Check

### Example 1: REJECT (Delta Too Large)

User in thinking phase with `top_k=8`, `relaxed_acceptance_threshold=0.15`, previous speculation was token ID 4200.

The `ResultDescriptor` from the device contains:

```
p_indices: [3001, 4200, 5500, 3002, 0, 0, 0, 0]
p_scores:  [0x3F00, 0x3E80, 0x3D80, 0x3C00, 0x0000, 0x0000, 0x0000, 0x0000]
           (bf16 raw bits)
```

**Step 1: Convert p_scores from bf16 to float.**

| Index | `p_indices` | `p_scores` (bf16 hex) | `p_scores` (float) | Token |
|---|---|---|---|---|
| 0 | 3001 | `0x3F00` | 0.5000 | Most likely |
| 1 | **4200** | `0x3E80` | 0.2500 | **Speculated token** |
| 2 | 5500 | `0x3D80` | 0.0625 | |
| 3 | 3002 | `0x3C00` | 0.0078125 | |
| 4-7 | 0 | `0x0000` | 0.0 | Top-P filtered (skipped) |

**Step 2: Search.** Scan `p_indices[0..8)` for token 4200. Found at `idx=1`. `p_scores[1] = 0x3E80 != 0`, so it is not a top-P-filtered slot.

**Step 3: Compute probability delta.**

$$p_{\max} = \text{bf16\_to\_float}(\texttt{0x3F00}) = 0.5000$$
$$p_{\text{draft}} = \text{bf16\_to\_float}(\texttt{0x3E80}) = 0.2500$$
$$\Delta = 0.5000 - 0.2500 = 0.2500$$

**Step 4: Compare.** $\Delta = 0.2500 > \delta = 0.15$.

**Result: REJECT.** The speculated token is in the top-k but its probability is too far below the most likely token.

### Example 2: ACCEPT (Prediction Is Argmax)

Same setup, but now the scores are:

```
p_indices: [4200, 3001, 5500, 3002, 0, 0, 0, 0]
p_scores:  [0x3F00, 0x3EC0, 0x3D80, 0x3C00, 0x0000, 0x0000, 0x0000, 0x0000]
```

| Index | `p_indices` | `p_scores` (float) |
|---|---|---|
| 0 | **4200** | 0.5000 |
| 1 | 3001 | 0.3750 |

The speculated token 4200 is at index 0 -- it IS the most likely token.

$$p_{\max} = 0.5000, \quad p_{\text{draft}} = 0.5000$$
$$\Delta = 0.0 \leq 0.15$$

**Result: ACCEPT.** The speculated token is the argmax. The delta is zero.

### Example 3: REJECT (Token Not in Top-K)

Same setup, but the speculated token is 9999:

```
p_indices: [3001, 4200, 5500, 3002, 0, 0, 0, 0]
```

Scan: 9999 not found in `p_indices[0..8)`. `idx = -1`.

**Result: REJECT.** The prediction is not in the model's top-k probability region.

## `p_scores`: bf16 Representation

The `p_scores` array in `ResultDescriptor` contains raw bf16 (bfloat16) values packed into `uint16_t` (`pipeline_types.hpp`, line 59):

```cpp
std::array<uint16_t, 32> p_scores{};
```

The conversion to `float` uses a bit-shift (lines 41-46 of `decode_scheduler.cpp`):

```cpp
static inline float bf16_to_float(uint16_t bits) {
    uint32_t f = static_cast<uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}
```

bfloat16 shares the same 8-bit exponent as IEEE 754 float32 but only 7 mantissa bits (vs 23 for float32). The conversion zero-pads the lower 16 mantissa bits -- an exact representation (no rounding error). The `memcpy` through a `float` variable avoids undefined behavior from type-punning (strict aliasing). Compilers optimize this to a single shift instruction.

### bf16 Precision Implications

With only 7 mantissa bits, bf16 has a relative precision of approximately $2^{-7} \approx 0.78\%$. Two probabilities that differ by less than this threshold cannot be distinguished in bf16 representation. For relaxed acceptance with very small thresholds (e.g., $\delta = 0.01$), the bf16 quantization may cause false accepts or rejects at the boundary. In practice, thinking-phase probabilities are typically spread enough that this quantization is not a bottleneck.

### `p_scores` Array Layout

| Index | Content |
|-------|---------|
| 0 | Score of the highest-probability token (always $p_{\max}$) |
| 1 | Score of the second-highest token |
| ... | ... |
| $k-1$ | Score of the $k$-th highest token |
| $k$ through 31 | Garbage (not populated by the device) |

Entries filtered out by top-P rescaling have their score set to 0 by the device. The arrays are sized at 32 entries, which bounds the maximum useful `top_k` to 32.

## `relaxed_acceptance_threshold` Configuration

The threshold $\delta$ is a per-user parameter set via `GenerationParams` (`decode_types.hpp`, line 75):

```cpp
float relaxed_acceptance_threshold = 0.0f;
```

Stored in `UserTable::relaxed_acceptance_threshold[uid]`, set by the API handler during SUBMIT and CONTINUE.

- $\delta = 0.0$: Only exact matches are accepted (equivalent to strict equality, since $p_{\max} - p_{\text{draft}} \leq 0$ only when $p_{\text{draft}} = p_{\max}$).
- $\delta = 1.0$: Any token in the top-K with non-zero score is accepted (maximizes accept rate, degrades quality).
- Practical range: $[0.05, 0.3]$, trading acceptance rate against output quality divergence.

The threshold is only active during thinking phase. Outside thinking phase, `check_acceptance()` returns strict equality regardless of the threshold value.

## Interaction with Speculative Decode Cases

The thinking phase affects only the ACCEPT/REJECT classification at the BASE result (the `check_acceptance` call at lines 543-544). It does not affect:

- **INITIAL**: No prior prediction to verify.
- **CONTINUE**: The bonus token is always emitted (no acceptance check).
- **STALE**: Always discarded (no acceptance check).
- **Pair injection**: Both BASE and SPEC use the same sampling parameters (determined by `device_sampling_params`).

## End-to-End Thinking Phase Flow

```
Token stream: ... normal ... <think> ... thinking tokens ... </think> ... normal ...
```

| Token | `in_thinking_phase` after toggle | Next inject's $k$ | Acceptance mode for next BASE |
|-------|:---:|---|---|
| "The" | false | 1 | strict equality |
| "answer" | false | 1 | strict equality |
| `<think>` | **true** | user's $k$ | relaxed delta |
| "Let" | true | user's $k$ | relaxed delta |
| "me" | true | user's $k$ | relaxed delta |
| "check" | true | user's $k$ | relaxed delta |
| `</think>` | **false** | 1 | strict equality |
| "42" | false | 1 | strict equality |
| EOS | false | 1 | strict equality |

The `<think>` token itself is emitted with the phase toggle happening inside `emit_token` BEFORE the staging call. So the **next** injection (for "Let") uses the thinking-phase sampling parameters. Similarly, `</think>` triggers the toggle, and the injection for "42" reverts to argmax.

## Cross-Thread Synchronization

The thinking-phase feature spans both the Reader and Writer threads:

```
Reader thread (emit_token):
  [1] Toggle in_thinking_phase[uid]  (release store)
  [2] stage_pair() or stage_eos_writeback()  (pushes to FIFO)

Writer thread (decode staging pop):
  [3] decode_staging.try_pop(entry)
  [4] device_sampling_params(uid)  (acquire load of in_thinking_phase[uid])
  [5] pipeline_->inject(...)       (uses k, top_p from step 4)
```

The release-acquire pair on `in_thinking_phase` (steps 1 and 4) ensures the Writer sees the phase toggle before injecting the token that was staged in step 2. The FIFO ordering of the staging queue provides additional structure: the entry staged in step 2 is popped in step 3, and step 4 reads the phase set in step 1. This works because the release store in step 1 happens-before the FIFO push in step 2 (program order), and the FIFO pop in step 3 happens-before the acquire load in step 4 (program order). The FIFO's internal synchronization establishes the cross-thread happens-before edge.

## Edge Cases

### Nested or Missing Close Tags

The phase tracking is a simple toggle: `<think>` sets true, `</think>` sets false. If the model emits `<think>` twice without a closing tag, the second `<think>` is a no-op (already true). If `</think>` is never emitted, the user remains in thinking phase for the rest of the generation. This does not cause correctness issues -- the user simply gets relaxed acceptance for all subsequent tokens. In practice, well-behaved models always emit matching tags.

### Single-Sided Configuration

If only one of `think_open_token_id` / `think_close_token_id` is configured (e.g., `think_open_token_id` is valid but `think_close_token_id` is `EMPTY_TOKEN`), the user can enter thinking phase but never leave it. The `in_thinking_phase` flag remains `true` for the rest of the session. This is a degenerate case -- in practice, both should be configured or neither.

### Top-P Filtering Zeros

The `p_scores[i] != 0` check in the relaxed acceptance scan (line 796) skips entries that the device zeroed out during top-P filtering. Without this check, a speculated token that happened to land in a zeroed slot would be found in `p_indices` but matched against a zero score, producing a misleading probability delta.

### Relaxed Threshold of Zero

With $\delta = 0$, the relaxed check reduces to: "is the speculated token at index 0 in `p_indices`?" This is equivalent to asking whether the speculated token is the argmax -- the same result as strict equality, but with the overhead of scanning `p_indices`. In practice, $\delta = 0$ with thinking-phase tracking enabled is uncommon; if strict acceptance is desired, the operator would simply not enable thinking-phase tokens.

## Configuration Summary

| Parameter | Source | Default | Effect |
|-----------|--------|---------|--------|
| `think_open_token_id` | `SchedulerParams` | `EMPTY_TOKEN` (disabled) | Token ID that enters thinking phase |
| `think_close_token_id` | `SchedulerParams` | `EMPTY_TOKEN` (disabled) | Token ID that exits thinking phase |
| `relaxed_acceptance_threshold` | `GenerationParams` (per-user) | `0.0` | Max probability delta for relaxed acceptance |
| `top_k` | `GenerationParams` (per-user) | `1` | Number of candidates during thinking phase |
| `top_p` | `GenerationParams` (per-user) | `1.0` | Top-P threshold during thinking phase |
| `temperature` | `GenerationParams` (per-user) | `1.0` | Always passed through to device |

---

**Previous:** [`throughput_analysis.md`](./throughput_analysis.md) | **Next:** [Chapter 5 -- Pipeline and Wire Format](../ch5_pipeline_and_wire_format/index.md)
