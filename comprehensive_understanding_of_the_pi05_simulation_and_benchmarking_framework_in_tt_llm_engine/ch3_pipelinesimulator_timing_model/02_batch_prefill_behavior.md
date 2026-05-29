# 3.2 Batch Prefill Behavior

## Overview

The `batchPrefill` flag in `PipelineSimulator` controls whether non-last prefill
tokens are modeled as passing through the full pipeline latency or completing
instantaneously.  When enabled, only the **last** prefill token (the one that
triggers the transition from PREFILL to DECODE) incurs the pipeline's
`totalLatency`; all preceding prefill tokens exit at `clock::now()`.

This behavior models real Tenstorrent hardware, where the device processes an
entire prompt batch in a single forward pass rather than feeding tokens one by
one through the systolic pipeline.

---

## 1. The batchPrefill Flag

### Constructor and Storage

The flag is accepted as the last constructor parameter and stored as a `bool`
member:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:59,68
PipelineSimulator(/* ... */, bool batchPrefill = false)
    : /* ... */
      batchPrefill(batchPrefill) {
```

### Config Mapping

In `PipelineSimulatorConfig`:

```cpp
// include/tt_llm_engine/pipeline/pipeline_types.hpp:95
bool batch_prefill = false;  // true: non-last prefill tokens skip the pipeline
```

Propagated through `DecodeScheduler` via `std::visit` (see [Section 3.1](./01_simulator_design.md), PipelineSimulatorConfig Mapping).

---

## 2. Inject-Path Behavior

The fast path at the top of `inject()` handles batch-prefill tokens:

```cpp
// include/tt_llm_engine/pipeline/pipeline_simulator.hpp:81-87
if (batchPrefill && desc.prefill_token_id != EMPTY_TOKEN) {
    std::lock_guard<std::mutex> lock(mu);
    inflight.push_back({clock::now(), makeResult(desc)});
    emitCv.notify_one();
    return;
}
```

### The EMPTY_TOKEN Sentinel

The `EMPTY_TOKEN` constant is defined as the maximum `uint32_t` value:

```cpp
// include/tt_llm_engine/pipeline/pipeline_types.hpp:21
static constexpr uint32_t EMPTY_TOKEN = std::numeric_limits<uint32_t>::max();
```

This sentinel serves as a "no token" marker throughout the pipeline interface.
In the batch-prefill context, the check `desc.prefill_token_id != EMPTY_TOKEN`
identifies tokens that are non-last prefill tokens -- the writer sets
`prefill_token_id` to the next prompt token for all but the last prefill token,
and to `EMPTY_TOKEN` for the last one.

### How the Writer Distinguishes Prefill Tokens

The `InjectDescriptor` has two token fields:

```cpp
// include/tt_llm_engine/pipeline/pipeline_types.hpp:36-38
uint32_t token_id = EMPTY_TOKEN;
uint32_t prefill_token_id = EMPTY_TOKEN;
```

When the writer thread processes prefill tokens in `writer_loop()`, it sets
`prefill_token_id` to the **next** prompt token (lookahead) -- or to
`EMPTY_TOKEN` for the **last** prefill token:

```cpp
// src/scheduler/decode/decode_scheduler.cpp:336-339
pipeline_->inject(InjectDescriptor{
    .slot_id = pfuid,
    .token_id = prompt_table.get_token(pfuid, prompt_idx),
    .prefill_token_id = is_last ? EMPTY_TOKEN : prompt_table.get_token(pfuid, prompt_idx + 1),
    .position = device_pos,
    .token_type = TokenType::BASE,
    /* ... */
});
```

So the condition `desc.prefill_token_id != EMPTY_TOKEN` is true for every
non-last prefill token and false for the last one.

### What Happens for Each Token Type

| Token | `prefill_token_id` | `batchPrefill` path | Latency |
|-------|-------------------|---------------------|---------|
| Non-last prefill | Next prompt token | Yes (fast path) | `clock::now()` -- immediate |
| Last prefill | `EMPTY_TOKEN` | No (normal path) | `enter + totalLatency` |
| Decode | `EMPTY_TOKEN` | No (normal path) | `enter + totalLatency` |

### Key Differences from Normal Path

The batch-prefill fast path differs from the normal `inject()` path in three
critical ways:

1. **No backpressure wait** -- it does not check `inflight.size() < numStages`.
   Batch-prefill tokens are pushed unconditionally because they exit
   immediately, so they don't accumulate in the FIFO.

2. **No rate limiting** -- it does not update `lastEnter` or compute
   `max(now, lastEnter + stagePeriod)`.  Prefill tokens can be injected at
   unlimited rate.

3. **Immediate exit time** -- `exitTime = clock::now()` means the reader will
   pop these tokens without any busy-wait delay.

---

## 3. Why Pi0.5 Needs Batch Prefill

### Hardware Behavior

On real Tenstorrent silicon, the prompt (prefill) phase operates fundamentally
differently from the decode phase:

- **Prefill**: The entire prompt is loaded into device memory and processed in a
  single batched forward pass.  All KV-cache entries for the prompt are written
  in parallel.  The host sees one latency event (the batch forward pass), not
  one per token.

- **Decode**: Each new token is generated sequentially, passing through the
  systolic pipeline stage by stage.

Without `batchPrefill`, the simulator would model prefill as N sequential
pipeline traversals (one per prompt token), each costing `numStages *
stagePeriod` microseconds.  For a 2048-token prompt with default settings:

$$
\text{simulated prefill} = 2048 \times 64 \times 44 \;\mu\text{s} = 5{,}767{,}168 \;\mu\text{s} \approx 5.8 \;\text{s}
$$

This would grossly overestimate prefill time and produce unrealistic TTFT
numbers.  With `batchPrefill = true`, all 2047 non-last tokens complete
instantly, and only the final token pays the pipeline latency:

$$
\text{simulated prefill} \approx 0 + 64 \times 44 \;\mu\text{s} = 2{,}816 \;\mu\text{s} \approx 2.8 \;\text{ms}
$$

### Pi0.5 vs Generic Runner Contrast

The Pi0.5 runner enables `batch_prefill = true` (see [Section 3.1](./01_simulator_design.md), Pi0.5 Pipeline Runner Usage).

In contrast, the generic `simulated_pipeline_runner.cpp` does **not** set
`batch_prefill`, leaving it at its default `false`:

```cpp
// examples/simulated_pipeline_runner.cpp:254-259
pl::PipelineSimulatorConfig sim_config{
    .num_stages = num_stages,
    .stage_duration_us = stage_duration_us,
    .accept_rate = accept_rate,
    .seed = seed,
};
```

This distinction reflects the architectural difference: Pi0.5 always performs
batch prefill in hardware, while the generic simulation runner models a pipeline
where prefill tokens traverse the pipeline individually.  The generic runner is
useful for testing scenarios where per-token prefill latency matters (e.g.,
measuring pipeline backpressure under sequential prefill load).

---

## 4. Impact on TTFT Measurement

Time to First Token (TTFT) measures the elapsed time from prompt submission to
the first decode token output.  The `batchPrefill` flag directly controls the
simulated TTFT:

### Without batchPrefill (false)

Each prefill token traverses the pipeline independently.  TTFT includes the
cumulative latency of all prefill tokens:

$$
\text{TTFT} \approx N_{\text{prompt}} \times \text{totalLatency} + \text{totalLatency}_{\text{first decode}}
$$

However, throughput pipelining means tokens overlap in the pipeline.  In steady
state (when the pipeline is full), one token exits every `stagePeriod`:

$$
\text{TTFT}_{\text{pipelined}} \approx \text{totalLatency} + (N_{\text{prompt}} - 1) \times \text{stagePeriod} + \text{totalLatency}_{\text{first decode}}
$$

### With batchPrefill (true)

Non-last prefill tokens complete at `clock::now()`, so they are popped by the
reader essentially instantly.  Only the last prefill token traverses the
pipeline, and then the first decode token also traverses the pipeline:

$$
\text{TTFT}_{\text{batch}} \approx \text{totalLatency}_{\text{last prefill}} + \text{totalLatency}_{\text{first decode}}
$$

The last prefill token has `prefill_token_id == EMPTY_TOKEN`, so it falls
through to the normal inject path and incurs the full `totalLatency`.  The
first decode token then also enters the normal path, incurring another
`totalLatency`.  For default Pi0.5 parameters:

$$
\text{TTFT}_{\text{batch}} \approx 2 \times 2{,}816 \;\mu\text{s} = 5{,}632 \;\mu\text{s} \approx 5.6 \;\text{ms}
$$

This closely models the real hardware behavior where prefill is a single batch
operation followed by sequential decode through the pipeline.

---

## 5. Interaction with Backpressure

### Backpressure Bypass for Batch-Prefill Tokens

The normal `inject()` path enforces backpressure by blocking on `injectCv`
when `inflight.size() >= numStages` (line 91 of `pipeline_simulator.hpp`; see
[Section 3.1](./01_simulator_design.md), Invariant 3 and the full `inject()`
listing).

The batch-prefill fast path **skips this check entirely**.  This is safe because
batch-prefill tokens have `exitTime = clock::now()`, so the reader pops them
as fast as they arrive.  In practice, the FIFO briefly grows by one element per
inject but drains immediately in the reader's next iteration.

The design correctly models hardware behavior: the device's prefill batch does
not occupy pipeline slots in the per-token sense, so applying the `numStages`
backpressure limit to prefill tokens would be an incorrect simulation.

### Transient FIFO Growth

If the writer thread injects N batch-prefill tokens faster than the reader can
pop them, the FIFO can temporarily grow to N elements.  This is bounded by the
prompt length (typically 2K-8K tokens) and is acceptable because:

1. Each `InflightToken` is small (~64 bytes: a time_point + ResultDescriptor).
2. The reader's loop is fast for immediate-exit tokens (no busy-wait).
3. Growth is transient -- the FIFO drains completely before decode begins.

### Rate-Limit Isolation

Batch-prefill tokens do not update `lastEnter`:

```cpp
// Fast path (lines 82-87): no lastEnter update
// Normal path (line 103): lastEnter = enter;
```

This means the rate limiter is "reset" when decode begins.  The first decode
token's `enter` time is computed as `max(now, lastEnter + stagePeriod)`, where
`lastEnter` is still the enter time of the last **normal-path** token (the
final prefill token).  Since the batch-prefill tokens don't advance `lastEnter`,
the rate limiter remains anchored to the last prefill token's entry, preserving
natural burst absorption for the decode phase transition.

---

## 6. Config Propagation Through DecodeScheduler

The following diagram traces how `batch_prefill` flows from user-facing
configuration to the simulator:

```
JSON / CLI args
    |
    v
PipelineSimulatorConfig {
    .batch_prefill = true        // pipeline_types.hpp:95
}
    |
    v
DecodeScheduler::Impl constructor
    |  std::visit on PipelineConfig variant
    v
PipelineSimulator(
    /* ... */,
    cfg.batch_prefill            // decode_scheduler.cpp:103
)
    |
    v
PipelineSimulator::batchPrefill  // stored as bool member, line 206
    |
    v
inject() fast path               // checked at line 82 on every inject()
```

The `DecodeScheduler`'s `writer_loop()` is completely unaware of whether batch
prefill is active.  It always sets `prefill_token_id` based on prompt position
(line 339):

```cpp
// src/scheduler/decode/decode_scheduler.cpp:339
.prefill_token_id = is_last ? EMPTY_TOKEN : prompt_table.get_token(pfuid, prompt_idx + 1),
```

The decision to skip pipeline latency for non-last prefill tokens is made
entirely within `PipelineSimulator::inject()`.  This separation of concerns
means the scheduler's prefill chunking, position tracking, and state machine
work identically regardless of the pipeline backend.

For JSON parsing and config schema details, see Chapter 2, Section 2.1.

### Reader-Side Prefill Tracking

On the reader side, the `DecodeScheduler` tracks prefill tokens with a
per-slot counter:

```cpp
// src/scheduler/decode/decode_scheduler.cpp:331-332 (writer side)
if (!is_last) {
    user_table.prefill_in_flight[pfuid].fetch_add(1, std::memory_order_release);
}
```

```cpp
// src/scheduler/decode/decode_scheduler.cpp:424-426 (reader side)
if (user_table.prefill_in_flight[uid].load(std::memory_order_acquire) > 0) {
    user_table.prefill_in_flight[uid].fetch_sub(1, std::memory_order_relaxed);
    continue;
}
```

When `batchPrefill` is active, these non-last prefill results arrive nearly
simultaneously (all have `exitTime = clock::now()`).  The reader drains them
rapidly, decrementing `prefill_in_flight` for each, then processes the last
prefill token's result normally -- transitioning the slot to DECODE and emitting
the first output token.

---

## 7. Prefill Token Result Generation

When `makeResult()` processes a non-last prefill token, it returns a minimal
`ResultDescriptor` containing only `slot_id` -- no token data.  The scheduler
uses these results solely to track prefill progress via the
`prefill_in_flight` counter.  For the full `makeResult()` implementation and
prefill early-return logic, see [Section 3.3](./03_token_model_and_spec_decode.md),
"Prefill Early Return."

---

## 8. Test Coverage

The test suite exercises `batchPrefill` indirectly through the
`StopMidPrefillRewindsToPrefillStart` test, which uses a slow pipeline
(10 ms/stage) to ensure that STOP can arrive while a long prompt is still
in PREFILL:

```cpp
// tests/scheduler/decode/test_decode_scheduler.cpp:1582-1586
pl::PipelineSimulatorConfig pipeline_config{
    .num_stages = 64,
    .stage_duration_us = 10000,
    .decode_token_id = 12345,
};
```

This test deliberately does **not** enable `batch_prefill` (defaults to
`false`), which ensures that prefill tokens take the slow path and the slot
remains in PREFILL long enough for the test to issue a STOP.  If
`batch_prefill` were `true`, prefill would complete nearly instantly,
preventing the test from exercising the mid-prefill stop path.

Tests that model Pi0.5 behavior would set `batch_prefill = true`, causing
prefill to complete nearly instantly and shifting the test focus to
decode-phase timing.

---

**Previous:** [Simulator Design](01_simulator_design.md)

**Next:** [Token Model and Speculative Decode](03_token_model_and_spec_decode.md)
