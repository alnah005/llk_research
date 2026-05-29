# 6.2 Batch Prefill and skip_eos_writeback

## Overview

Pi0.5 sets two flags that alter `DecodeScheduler`'s internal behavior: `batch_prefill = true` (on `PipelineSimulatorConfig`) and `skip_eos_writeback = true` (on `SchedulerParams`). These flags are complementary optimizations for the fresh-context SUBMIT loop described in [Section 6.1](01_scheduler_request_lifecycle.md). `batch_prefill` accelerates prefill; `skip_eos_writeback` eliminates a useless EOS write at generation end. This section explains the mechanics of each flag inside the scheduler, why Pi0.5 needs them, and what would go wrong without them.

## 6.2.1 batch_prefill in PipelineSimulatorConfig

The `batch_prefill` field lives on `PipelineSimulatorConfig`:

```cpp
// include/tt_llm_engine/pipeline/pipeline_types.hpp, lines 82-96
struct PipelineSimulatorConfig {
    uint32_t num_stages = 64;
    uint32_t stage_duration_us = 44;
    uint32_t decode_token_id = EMPTY_TOKEN;
    float accept_rate = 1.0f;
    uint32_t seed = 42;
    uint32_t safe_vocab_base = 0;
    uint32_t safe_vocab_modulus = 0;
    bool batch_prefill = false;  // line 95
};
```

When the `DecodeScheduler` constructs a `PipelineSimulator`, it passes `batch_prefill` directly through:

```cpp
// src/scheduler/decode/decode_scheduler.cpp, lines 99-103
} else if constexpr (std::is_same_v<T, PipelineSimulatorConfig>) {
    return std::make_unique<PipelineSimulator>(
        cfg.num_stages, cfg.stage_duration_us, cfg.decode_token_id,
        cfg.accept_rate, cfg.seed, cfg.safe_vocab_base, cfg.safe_vocab_modulus,
        cfg.batch_prefill);
```

The scheduler itself does not interpret `batch_prefill`; it is purely a pipeline-backend concern. The `PipelineSimulator` uses it to decide whether non-last prefill tokens should traverse the full simulated pipeline or complete instantly (see [Chapter 3](../ch3_pipelinesimulator_timing_model/index.md) for the full mechanics of `batch_prefill` inside `PipelineSimulator`).

Pi0.5 always sets `batch_prefill = true`:

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1208-1213
pl::PipelineSimulatorConfig sim_config{
    .num_stages = total_stages,
    .stage_duration_us = effective_stage_us,
    .accept_rate = accept_rate,
    .seed = seed,
    .batch_prefill = true,   // <---
};
```

### Why Pi0.5 Needs batch_prefill

On real Tenstorrent hardware, a prefill of $N$ tokens is processed in a single forward pass -- only the final token enters the decode pipeline to produce a sampled output. Without `batch_prefill = true`, the `PipelineSimulator` would push every prefill token through the pipeline sequentially, each incurring the full stage latency:

$$T_{\text{prefill, no batch}} = N \times S \times d$$

where $N$ is the prompt length, $S$ is the number of pipeline stages, and $d$ is the per-stage duration. With `batch_prefill = true`:

$$T_{\text{prefill, batch}} \approx S \times d$$

For Pi0.5's default 128-token prompt across a multi-stage pipeline, this is a $128\times$ reduction in simulated prefill time. Without it, the simulated TTFT would be unrealistically high and the benchmark would not reflect real hardware behavior.

The detailed mechanics of how `PipelineSimulator` implements this -- specifically, how the writer thread injects prefill tokens one at a time via `pipeline_->inject()` and the simulator returns `ResultDescriptor` immediately for non-last tokens -- are covered in [Chapter 3, Section 2](../ch3_pipelinesimulator_timing_model/02_batch_prefill_behavior.md).

## 6.2.2 skip_eos_writeback in SchedulerParams

The `skip_eos_writeback` field lives on `SchedulerParams`:

```cpp
// include/tt_llm_engine/scheduler/decode/decode_types.hpp, line 51
bool skip_eos_writeback = false;  // true: no EOS inject on completion
```

Pi0.5 always sets this to `true`:

```cpp
// examples/pi05_pipeline_runner.cpp, line 1217
.skip_eos_writeback = true,
```

### The stage_eos_writeback Lambda

When generation completes (EOS token emitted, max tokens reached, or context exhausted), the reader thread's `emit_token` lambda calls `stage_eos_writeback`:

```cpp
// src/scheduler/decode/decode_scheduler.cpp, lines 447-453
auto stage_eos_writeback = [&]() {
    if (params.skip_eos_writeback) return;   // <--- Pi0.5 exits here
    user_table.post_complete_in_flight[uid].fetch_add(1, std::memory_order_release);
    decode_staging.stage(uid,
        result.actual_token, result.actual_token_pos,
        EMPTY_TOKEN, 0, /*skip_spec=*/true);
};
```

When `skip_eos_writeback` is `false` (the default for LLM serving), the lambda:

1. Increments `post_complete_in_flight` -- a reference count so the reader can discard the loopback result from this extra inject.
2. Stages the completing token at `actual_token_pos` into the decode staging FIFO, which the writer thread will inject into the pipeline as a BASE-only token with `skip_spec=true`.

This writes the EOS (or terminating) token into the KV cache at the final position. The purpose is to set up the KV state correctly for a subsequent `CONTINUE` -- the next turn's prefill starts at `current_position = actual_token_pos + 1`, and the KV at `actual_token_pos` should contain the EOS so the model sees the correct context boundary:

```cpp
// src/scheduler/decode/decode_scheduler.cpp, lines 436-439
auto set_position_on_complete = [&]() {
    user_table.current_position[uid].store(
        result.actual_token_pos + 1, std::memory_order_release);
};
```

### Why Pi0.5 Skips EOS Writeback

When `skip_eos_writeback = true`, the lambda returns immediately, skipping the token injection entirely. Pi0.5 uses this because:

1. **No CONTINUE follows:** Pi0.5 never sends `CONTINUE`. Each turn is a fresh `SUBMIT` that resets position to 0. The next turn does not need the EOS token preserved in KV because it will overwrite the entire KV cache from scratch.

2. **Eliminates an extra pipeline traversal:** Without the skip, the EOS writeback injects a token into `decode_staging`, which the writer thread picks up and pushes through `pipeline_->inject()`. This costs one full pipeline pass ($S \times d$ microseconds) of pure overhead. For a 220-stage Pi0.5 pipeline at 6us/stage, this is $\approx 1.3\text{ms}$ of unnecessary work per turn completion. With 4 users running 3 turns each, that is 12 wasted pipeline traversals.

3. **Avoids `post_complete_in_flight` tracking:** The writeback normally increments `post_complete_in_flight` and expects the reader to discard the loopback result. Skipping avoids this bookkeeping entirely.

4. **Prevents stale EOS in KV:** If the EOS were written back and then a fresh SUBMIT followed, the SUBMIT would reset position to 0 and overwrite KV from the start. But between the EOS writeback completion and the SUBMIT's prefill reaching that position, there is a window where the KV contains a stale EOS that could confuse diagnostics. Skipping eliminates this edge.

### What Would Break Without skip_eos_writeback

If `skip_eos_writeback` were `false` with the Pi0.5 fresh-context pattern:

| Problem | Mechanism |
|---------|-----------|
| **Extra latency** | Each turn completion injects an EOS token that traverses the full pipeline, adding one pipeline-pass of delay before the slot becomes available for re-SUBMIT |
| **Wasted bandwidth** | The EOS inject occupies a pipeline slot for $S$ ticks, displacing useful decode work from other users |
| **Completion-to-resubmit race** | The `post_complete_in_flight` counter must drain to 0 before EVICT can finalize. If a re-SUBMIT arrives while the EOS is still in-flight, the SUBMIT handler resets `in_flight_count` to 0 but the reader thread may later attempt to decrement when the stale EOS result arrives, causing an underflow |

The race is particularly subtle: the SUBMIT handler (`decode_scheduler.cpp:681-701`) unconditionally stores `in_flight_count` to 0, but the EOS writeback result still sits in the pipeline. When the reader processes it, `post_complete_in_flight` (not `in_flight_count`) absorbs the decrement -- but only if the reader sees it before the new SUBMIT's prefill results start arriving. With `skip_eos_writeback = true`, this entire class of timing issues disappears.

The benchmark would still produce correct results without the skip, but latency metrics (especially TTFT for subsequent turns) would be artificially inflated.

## 6.2.3 Interaction: skip_eos + Fresh SUBMIT = Implicit KV Reset

The two flags work together to create a clean contract for each turn transition:

```
Turn N completes:
  1. emit_token() publishes OutputMessage{is_complete=true}
  2. stage_eos_writeback() -> returns immediately (skip_eos)
  3. set_position_on_complete() stores position for potential CONTINUE (unused)
  4. Reader releases ReaderClaim

Turn N+1 begins:
  5. Runner receives is_complete, calls push_request(SUBMIT)
  6. api_loop handles SUBMIT:
     - prompt_table.store(uid, new_tokens)
     - current_position = 0              <-- fresh start
     - prefill_pos = 0, prefill_start_pos = 0
     - tokens_generated = 0, in_flight_count = 0
     - state = PREFILL
     - prefill_queue.push(uid)
  7. Writer drains prefill_queue, injects new prompt tokens
```

Because step 2 is a no-op, there is no pipeline work in progress for this slot when step 6 resets `in_flight_count` to 0. The state machine transitions cleanly from COMPLETE back to PREFILL without any in-flight tokens to drain.

Without `skip_eos`, an additional drain loop would be needed between steps 4 and 5, waiting for `post_complete_in_flight` to reach 0. The skip flag converts a potentially blocking wait into a zero-cost transition.

The KV cache is implicitly reset by the fact that SUBMIT starts prefill at position 0. There is no explicit KV-clear operation -- the old KV entries at positions 0 through $N-1$ are simply overwritten by the new prefill tokens. This is safe because `PipelineSimulator` (and real hardware) treat KV writes as positional overwrites, not appends. The attention mask during decode only considers positions $\le$ `current_position`, so stale entries beyond the new prefill length are never attended to.

## 6.2.4 batch_prefill + skip_eos: The Complete Optimization Picture

Together, these two flags optimize the two ends of each turn:

$$
T_{\text{turn}} = T_{\text{prefill}} + T_{\text{decode}} + T_{\text{teardown}}
$$

- **batch_prefill** collapses $T_{\text{prefill}}$ from $O(L_{\text{prompt}} \times t_{\text{pipeline}})$ to $O(t_{\text{pipeline}})$
- **skip_eos_writeback** collapses $T_{\text{teardown}}$ from $O(t_{\text{pipeline}})$ to $O(1)$

For a 128-token prompt with a 220-stage pipeline at 6us/stage ($\approx 1.32\text{ms}$ pipeline latency):

| Component | Without optimizations | With optimizations |
|-----------|----------------------|-------------------|
| Prefill | $128 \times 1.32\text{ms} \approx 169\text{ms}$ | $\approx 1.32\text{ms}$ |
| Decode | $64 \times 1.32\text{ms} \approx 84.5\text{ms}$ | $\approx 84.5\text{ms}$ (unchanged) |
| Teardown | $\approx 1.32\text{ms}$ | $\approx 0\text{ms}$ |
| **Total** | **$\approx 255\text{ms}$** | **$\approx 86\text{ms}$** |

## 6.2.5 Flag Interaction Matrix

| Scenario | `batch_prefill` | `skip_eos_writeback` | Effect |
|----------|:-:|:-:|--------|
| Pi0.5 (frame-by-frame VLA) | `true` | `true` | Fast prefill, no EOS writeback, SUBMIT resets KV |
| Standard LLM serving | `false` or `true` | `false` | EOS written to KV for CONTINUE continuity |
| Chat with context accumulation | `true` | `false` | Fast prefill, EOS marks turn boundary in KV |

The flags are orthogonal -- either can be set independently. But for the Pi0.5 use case, both are always enabled because the fresh-context pattern makes both optimizations safe and beneficial. Pi0.5 is the only known consumer that sets both flags simultaneously.

---

**Next:** [User Session Management](03_user_session_management.md)
