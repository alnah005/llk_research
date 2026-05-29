# 1.2 Three-Phase Pipeline Model

The Pi0.5 framework models the neural network's three computational phases -- vision, text, and denoise -- as a sequence of timed pipeline stages. This section covers the `PhaseConfig` struct, the default timing for each phase, and how the multi-phase model is flattened into a uniform systolic pipeline for `PipelineSimulator`.

## PhaseConfig Struct

The `PhaseConfig` struct is defined locally in `examples/pi05_pipeline_runner.cpp` (~line 184) inside the anonymous namespace. It is not part of the `tt_llm_engine` library -- it exists only in the Pi0.5 example application:

```cpp
// examples/pi05_pipeline_runner.cpp, ~line 184-197
struct PhaseConfig {
    std::string name;
    std::vector<uint32_t> stage_durations_us;
    uint32_t loop_count = 1;

    uint32_t num_stages() const { return static_cast<uint32_t>(stage_durations_us.size()); }
    uint64_t latency_us() const {
        uint64_t sum = 0;
        for (auto d : stage_durations_us) sum += d;
        return sum;
    }
    uint32_t effective_stages() const { return num_stages() * loop_count; }
    uint64_t effective_latency_us() const { return latency_us() * loop_count; }
};
```

Each field has a precise role:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `std::string` | Phase identifier: `"vision"`, `"text"`, or `"denoise"`. Used for bottleneck analysis (the vision phase is excluded from `max_non_vision_stage_us`). |
| `stage_durations_us` | `std::vector<uint32_t>` | Per-stage latency in microseconds. Each element models one pipeline stage on a device (e.g., one chip in a multi-chip layout). Must have at least one element, each >= 1. |
| `loop_count` | `uint32_t` | How many times this phase repeats per inference. Defaults to 1. Must be >= 1. The denoise phase uses `loop_count=5` for 5 Euler flow-matching iterations. |

### Derived Methods

| Method | Returns | Formula |
|--------|---------|---------|
| `num_stages()` | `uint32_t` | `stage_durations_us.size()` |
| `latency_us()` | `uint64_t` | $\sum_{i} \text{stage\_durations\_us}[i]$ |
| `effective_stages()` | `uint32_t` | $\text{num\_stages()} \times \text{loop\_count}$ |
| `effective_latency_us()` | `uint64_t` | $\text{latency\_us()} \times \text{loop\_count}$ |

## The Three Phases

### Vision Phase (SigLIP Encoder)

Models the SigLIP vision encoder that processes camera frames into visual embeddings.

**Default configuration** (from `pi05_config.json`):

```json
{
  "name": "vision",
  "stages": [1000, 2200, 2200, 2200]
}
```

| Property | Value |
|----------|-------|
| Stages | 4 |
| Durations | 1000, 2200, 2200, 2200 us |
| Loop count | 1 (implicit default) |
| Single-pass latency | $1000 + 2200 + 2200 + 2200 = 7600$ us |
| Effective stages | 4 |
| Effective latency | 7,600 us |

The vision phase has the most heterogeneous stage timing: the first stage (1,000 us) is significantly shorter than the remaining three (2,200 us each). This reflects a real architectural property -- the initial patch embedding projection is cheaper than the subsequent transformer layers in SigLIP.

The vision phase runs **once per inference** -- each camera frame set is encoded exactly once, and the resulting embeddings are consumed by all subsequent phases.

### Text Phase (Gemma Language Backbone)

Models the Gemma language model that fuses visual embeddings with the tokenized instruction.

**Default configuration** (from `pi05_config.json`):

```json
{
  "name": "text",
  "stages": [591, 591, 591, 591, 591, 591, 591, 591, 591, 591,
             591, 591, 591, 591, 591, 591, 591, 591, 591, 591]
}
```

| Property | Value |
|----------|-------|
| Stages | 20 |
| Durations | 591 us each (uniform) |
| Loop count | 1 (implicit default) |
| Single-pass latency | $20 \times 591 = 11{,}820$ us |
| Effective stages | 20 |
| Effective latency | 11,820 us |

The text phase has uniform stage timing across all 20 stages, reflecting a well-balanced partitioning of the Gemma transformer model across 20 pipeline stages. This is the most pipeline-stage-heavy phase, contributing 20 of the 24 non-denoise stages.

### Denoise Phase (Euler Flow-Matching)

Models the iterative denoising loop that converts the language model's output distribution into concrete robot actions.

**Default configuration** (from `pi05_config.json`):

```json
{
  "name": "denoise",
  "stages": [757, 757, 757, 757, 757, 757],
  "loop_count": 5
}
```

| Property | Value |
|----------|-------|
| Stages per iteration | 6 |
| Durations | 757 us each (uniform) |
| Loop count | 5 |
| Single-pass latency | $6 \times 757 = 4{,}542$ us |
| Effective stages | $6 \times 5 = 30$ |
| Effective latency | $4{,}542 \times 5 = 22{,}710$ us |

The `loop_count=5` distinguishes the denoise phase: it models 5 Euler flow-matching denoising iterations, each passing through 6 pipeline stages. This makes denoise the dominant latency contributor despite having fewer physical stages than the text phase.

## Flattening to a Uniform Systolic Pipeline

`PipelineSimulator` models a **uniform systolic pipeline** -- all stages have the same duration. But the Pi0.5 model has heterogeneous stage timings (1,000us, 2,200us, 591us, 757us). The framework bridges this gap by computing a **weighted-average stage duration** that preserves total latency.

### The Math

Given $P$ phases, where phase $p$ has $n_p$ stages, per-stage durations $d_{p,i}$, and loop count $L_p$:

$$\text{total\_stages} = \sum_{p=1}^{P} n_p \cdot L_p$$

$$\text{total\_latency\_us} = \sum_{p=1}^{P} L_p \cdot \sum_{i=1}^{n_p} d_{p,i}$$

$$\text{effective\_stage\_us} = \left\lfloor \frac{\text{total\_latency\_us}}{\text{total\_stages}} \right\rfloor$$

### Implementation

The flattening computation appears in `main()` at approximately lines 801-813:

```cpp
// examples/pi05_pipeline_runner.cpp, ~line 801-813
uint32_t total_stages = 0;
uint64_t total_latency_us = 0;
for (const auto& phase : phases) {
    total_stages += phase.effective_stages();
    total_latency_us += phase.effective_latency_us();
}
if (total_stages < 1) {
    throw std::runtime_error("Total pipeline stages must be >= 1");
}
const uint32_t effective_stage_us = static_cast<uint32_t>(total_latency_us / total_stages);
if (effective_stage_us < 1) {
    throw std::runtime_error("Effective stage duration must be >= 1us");
}
```

### Concrete Example (Default Config)

| Phase | $n_p$ | $L_p$ | Effective stages | Effective latency (us) |
|-------|--------|--------|------------------|------------------------|
| Vision | 4 | 1 | 4 | 7,600 |
| Text | 20 | 1 | 20 | 11,820 |
| Denoise | 6 | 5 | 30 | 22,710 |
| **Total** | | | **54** | **42,130** |

$$\text{effective\_stage\_us} = \left\lfloor \frac{42{,}130}{54} \right\rfloor = 780 \text{ us}$$

The `PipelineSimulator` is then configured with `num_stages=54` and `stage_duration_us=780`:

```cpp
pl::PipelineSimulatorConfig sim_config{
    .num_stages = total_stages,          // 54
    .stage_duration_us = effective_stage_us,  // 780
    .accept_rate = accept_rate,
    .seed = seed,
    .batch_prefill = true,
};
```

### Why Flattening Is Necessary

`PipelineSimulator` (defined in `include/tt_llm_engine/pipeline/pipeline_simulator.hpp`) enforces three invariants:

1. **Latency:** each token exits exactly $\text{num\_stages} \times \text{stage\_duration\_us}$ after entry.
2. **Throughput cap:** at most one token enters per `stage_duration_us` period.
3. **Backpressure:** at most `num_stages` tokens in flight.

These invariants assume a uniform stage period. Modeling heterogeneous stages would require either a per-stage scheduler (which the simulator is not designed to be) or a multi-simulator chain (which would not correctly model inter-phase pipeline filling). The weighted average is a deliberate simplification.

### What Accuracy Is Sacrificed

The weighted-average approach preserves **total pipeline latency**: $54 \times 780 = 42{,}120$ us, which is within 10 us of the true 42,130 us (truncation error from integer division).

However, the flattening sacrifices accuracy in two ways:

1. **Throughput overestimate for bottleneck stages:** In reality, the pipeline's sustained throughput is limited by the slowest stage (2,200 us in the vision phase). The uniform model uses 780 us/stage, which would predict a throughput of $1/780 \approx 1{,}282$ tokens/second -- roughly 2.8x higher than the true bottleneck-limited rate of $1/2{,}200 \approx 454$ tokens/second. However, since Pi0.5 generates very few tokens per inference (typically `output_tokens=1`), this throughput difference is immaterial; what matters is single-request latency, which is preserved exactly.

2. **Stagger timing:** When multiple users are submitted with staggered start times, the uniform stage duration changes when pipeline bubbles collapse and when inter-user interference occurs. With the real heterogeneous pipeline, a user entering during a slow vision stage would experience different interference patterns than the uniform model predicts.

3. **Phase boundary effects:** The simulator has no notion of phase boundaries -- it cannot model a 7,600 us vision phase followed by a 11,820 us text phase followed by a 22,710 us denoise phase. All 54 stages appear as uniform 780 us ticks. In reality, tokens exiting the vision phase immediately enter the text phase with potentially different stage timing characteristics.

For Pi0.5's primary use case (latency measurement of few-token, multi-turn robotics inference), the total latency preservation is the critical property, and the weighted average provides sufficient fidelity.

## Bottleneck Stage Calculation

The framework computes two bottleneck metrics to characterize throughput limits (~lines 816-829):

```cpp
// examples/pi05_pipeline_runner.cpp, ~line 816-829
uint32_t max_stage_duration_us = 0;
for (const auto& phase : phases) {
    for (auto d : phase.stage_durations_us) {
        max_stage_duration_us = std::max(max_stage_duration_us, d);
    }
}
uint32_t max_non_vision_stage_us = 0;
for (const auto& phase : phases) {
    if (phase.name == "vision") continue;
    for (auto d : phase.stage_durations_us) {
        max_non_vision_stage_us = std::max(max_non_vision_stage_us, d);
    }
}
if (max_non_vision_stage_us == 0) max_non_vision_stage_us = max_stage_duration_us;
```

| Metric | Default Value | Meaning |
|--------|:---:|---------|
| `max_stage_duration_us` | 2,200 us | Absolute bottleneck across all phases. Determines peak pipeline throughput: $1/2{,}200 \approx 454$ tokens/s. |
| `max_non_vision_stage_us` | 757 us | Bottleneck excluding the vision phase. Used for corrected TTFT estimation, since vision runs once at the start and does not constrain steady-state decode throughput. |

### Why Exclude Vision?

The vision phase runs once at the start of each inference turn. During the decode loop (where throughput matters most), only the text and denoise phases are active. The `max_non_vision_stage_us` therefore represents the true steady-state throughput bottleneck.

### Corrected TTFT Estimate

The `max_non_vision_stage_us` metric feeds into the **corrected TTFT** calculation at the end of the simulation run (~lines 1432-1434):

```cpp
// examples/pi05_pipeline_runner.cpp, lines 1432-1434
double corrected_ttft_single_ms = total_latency_us / 1000.0;
double corrected_ttft_mean_ms = (total_latency_us
    + (num_users - 1) * max_non_vision_stage_us / 2.0) / 1000.0;
```

For a single user, the corrected TTFT equals the total pipeline latency. For $N$ users submitting concurrently, the mean corrected TTFT adds a contention term:

$$\text{corrected\_ttft\_single\_ms} = \frac{\text{total\_latency\_us}}{1000}$$

$$\text{corrected\_ttft\_mean\_ms} = \frac{\text{total\_latency\_us} + (N - 1) \times \frac{\text{max\_non\_vision\_stage\_us}}{2}}{1000}$$

This estimates the mean TTFT across users by adding half a bottleneck-stage duration per additional user, modeling the expected queuing delay when multiple users' tokens compete for pipeline entry.

## Hybrid Configuration Variants

The phase definitions differ across configuration files, reflecting different hardware layouts:

### Full/Simulation Config (`pi05_config.json` / `pi05_config_sim.json`)

Multi-stage phases modeling a multi-chip pipeline:
- Vision: 4 stages (1000, 2200, 2200, 2200 us)
- Text: 20 stages (591 us each)
- Denoise: 6 stages (757 us each) x 5 loops

### Hybrid Config (`pi05_config_hybrid.json`)

The hybrid config uses single-stage phases (vision: 2200us, text: 591us, denoise: 757us x5), yielding total_stages=7, total_latency=6,576us, effective_stage_us=939us. This reflects the reduced parallelism of a single-chip configuration: fewer stages mean higher effective per-stage duration and lower throughput, but also lower total latency since there is no multi-chip pipeline filling overhead.

---

**Next:** [`03_build_system_and_component_map.md`](./03_build_system_and_component_map.md)
