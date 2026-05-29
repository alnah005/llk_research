# 2.3 Reference Configs Walkthrough

The project ships three reference configuration files that exercise different operating modes of the pipeline runner. This section provides numerical walkthroughs of each, deriving the effective stage counts and weighted-average stage durations that feed into the `PipelineSimulatorConfig` consumed by the timing model (see [Chapter 3](../ch3_pipelinesimulator_timing_model/index.md)).

## 2.3.1 `pi05_config.json` -- Full Socket Mode

**File:** `examples/pi05_config.json`

```json
{
  "num_users": 8,
  "input_tokens": 768,
  "output_tokens": 1,
  "max_turns": 20,
  "socket": {
    "h2d_token_socket_id": "pi05_h2d_token",
    "d2h_token_socket_id": "pi05_d2h_token",
    "h2d_bulk_socket_id": "pi05_h2d_bulk",
    "d2h_bulk_socket_id": "pi05_d2h_bulk",
    "h2d_mode": "DEVICE_PULL",
    "connect_timeout_ms": 30000,
    "pcie_bandwidth_gbps": 16.0,
    "fifo_size": 524288
  },
  "pixel_payload": {
    "width": 224,
    "height": 224,
    "channels": 3,
    "frames": 3,
    "action_history_bytes": 4096
  },
  "phases": [
    {
      "name": "vision",
      "stages": [1000, 2200, 2200, 2200]
    },
    {
      "name": "text",
      "stages": [591, 591, 591, 591, 591, 591, 591, 591, 591, 591,
                 591, 591, 591, 591, 591, 591, 591, 591, 591, 591]
    },
    {
      "name": "denoise",
      "stages": [757, 757, 757, 757, 757, 757],
      "loop_count": 5
    }
  ]
}
```

### Operating Mode

This is the **full socket mode** configuration. The presence of the `"socket"` block sets `use_sockets = true`, and `mode` defaults to `"full"` (no explicit `"mode"` key is present). This means:

- Token path: `SocketPipeline` (real H2D/D2H token sockets)
- Bulk data path: Real H2D/D2H bulk sockets
- All four socket IDs are specified
- Requires `PI05_HAS_SOCKETS` at compile time

### Phase-by-Phase Numerical Walkthrough

#### Vision Phase

| Property | Value |
|----------|-------|
| Stages | `[1000, 2200, 2200, 2200]` |
| `num_stages()` | 4 |
| `loop_count` | 1 (default) |
| `effective_stages()` | $4 \times 1 = 4$ |
| `latency_us()` | $1000 + 2200 + 2200 + 2200 = 7{,}600$ us |
| `effective_latency_us()` | $7{,}600 \times 1 = 7{,}600$ us |

Note: The first vision stage (1,000 us) is roughly half the duration of the other three (2,200 us each). This models a lighter initial preprocessing/embedding step followed by heavier vision transformer encoding stages.

#### Text Phase

| Property | Value |
|----------|-------|
| Stages | `[591, 591, ..., 591]` (20 identical) |
| `num_stages()` | 20 |
| `loop_count` | 1 (default) |
| `effective_stages()` | $20 \times 1 = 20$ |
| `latency_us()` | $591 \times 20 = 11{,}820$ us |
| `effective_latency_us()` | $11{,}820 \times 1 = 11{,}820$ us |

All 20 text stages are identical at 591 us, modeling a uniform transformer decoder.

#### Denoise Phase

| Property | Value |
|----------|-------|
| Stages | `[757, 757, 757, 757, 757, 757]` |
| `num_stages()` | 6 |
| `loop_count` | 5 |
| `effective_stages()` | $6 \times 5 = 30$ |
| `latency_us()` | $757 \times 6 = 4{,}542$ us (per iteration) |
| `effective_latency_us()` | $4{,}542 \times 5 = 22{,}710$ us |

The `loop_count = 5` models the 5 iterative denoising steps of the flow-matching action head in Pi0.5. Each iteration has 6 stages at 757 us each. The `loop_count` multiplier avoids duplicating the stages array -- see the flattening math in [Chapter 1](../ch1_architecture_overview_and_operating_modes/index.md).

### Aggregate Pipeline Metrics

These totals are computed by the flattening logic at lines 801-813:

```cpp
// examples/pi05_pipeline_runner.cpp:801-813
uint32_t total_stages = 0;
uint64_t total_latency_us = 0;
for (const auto& phase : phases) {
    total_stages += phase.effective_stages();
    total_latency_us += phase.effective_latency_us();
}
// ...
const uint32_t effective_stage_us = static_cast<uint32_t>(total_latency_us / total_stages);
```

$$\texttt{total\_stages} = 4 + 20 + 30 = 54$$

$$\texttt{total\_latency\_us} = 7{,}600 + 11{,}820 + 22{,}710 = 42{,}130 \;\mu s$$

$$\texttt{effective\_stage\_us} = \lfloor 42{,}130 \div 54 \rfloor = \lfloor 780.18\ldots \rfloor = 780 \;\mu s$$

The `effective_stage_us` value (780 us) is a **weighted average** across all stages and loop iterations. It is the single stage duration passed to `PipelineSimulatorConfig`, which requires a uniform stage duration. The weighted average preserves total pipeline latency:

$$54 \times 780 = 42{,}120 \;\mu s \approx 42{,}130 \;\mu s$$

The 10 us discrepancy is due to integer truncation in the division. As discussed in [Chapter 1](../ch1_architecture_overview_and_operating_modes/index.md), this is the fundamental flattening tradeoff: the simulator models all stages as equal-duration, losing individual stage timing fidelity but preserving total pipeline latency. The implications for throughput and latency predictions are explored in [Chapter 3](../ch3_pipelinesimulator_timing_model/index.md).

### Bottleneck Stage

The bottleneck analysis (lines 816-829) finds the slowest stage across all phases:

$$\texttt{max\_stage\_duration\_us} = \max(1000, 2200, 591, 757) = 2{,}200 \;\mu s$$

The maximum non-vision stage duration (used for corrected TTFT estimation):

$$\texttt{max\_non\_vision\_stage\_us} = \max(591, 757) = 757 \;\mu s$$

### Pixel Payload Metrics

$$\texttt{pixel\_bytes} = 224 \times 224 \times 3 \times 3 = 451{,}584 \text{ bytes}$$

$$\texttt{total\_bytes} = 451{,}584 + 4{,}096 = 455{,}680 \text{ bytes}$$

$$\texttt{H2D payload} = 8 + 455{,}680 = 455{,}688 \text{ bytes (with BulkH2DHeader)}$$

$$\texttt{padded} = \lceil 455{,}688 / 4{,}096 \rceil \times 4{,}096 = 458{,}752 \text{ bytes (112 pages)}$$

This fits within the 524,288-byte FIFO ($458{,}752 < 524{,}288$).

### Key Design Insight: `output_tokens = 1`

The config sets `output_tokens: 1`, which means each turn produces exactly **one decode token**. This models Pi0.5's single-action-token output architecture: the model processes a full vision+text+denoise pipeline to produce a single action output. The actual continuous action is a 50-dimensional output (50 x 32 bfloat16 = 3,200 bytes, `ACTION_OUTPUT_BYTES` at line 203) delivered via the bulk D2H channel, not through the token pipeline. The token itself serves as a synchronization signal that triggers the bulk D2H read. See Section 2.3.4 for the broader implications.

### Auto-Computed Stagger

When `--stagger-us` is not specified on the command line:

$$\texttt{stagger\_us} = \lfloor \texttt{total\_latency\_us} \div \texttt{num\_users} \rfloor = \lfloor 42{,}130 \div 8 \rfloor = 5{,}266 \;\mu s$$

This spaces the 8 users' initial submissions ~5.3 ms apart, filling the pipeline uniformly over the full 42 ms pipeline latency.

---

## 2.3.2 `pi05_config_hybrid.json` -- Bulk-Only Hybrid Mode

**File:** `examples/pi05_config_hybrid.json`

```json
{
  "num_users": 1,
  "input_tokens": 768,
  "output_tokens": 1,
  "max_turns": 400,
  "socket": {
    "h2d_bulk_socket_id": "pi05_h2d_bulk",
    "d2h_bulk_socket_id": "pi05_d2h_bulk",
    "h2d_mode": "DEVICE_PULL",
    "connect_timeout_ms": 30000,
    "pcie_bandwidth_gbps": 25.0,
    "fifo_size": 524288,
    "mode": "bulk_only"
  },
  "pixel_payload": {
    "width": 224,
    "height": 224,
    "channels": 3,
    "frames": 3,
    "action_history_bytes": 4096
  },
  "phases": [
    {
      "name": "vision",
      "stages": [2200]
    },
    {
      "name": "text",
      "stages": [591]
    },
    {
      "name": "denoise",
      "stages": [757],
      "loop_count": 5
    }
  ]
}
```

### Operating Mode

This is the **bulk-only hybrid mode** configuration. Key differences from the full config:

- `mode` is explicitly `"bulk_only"` -- token path uses `PipelineSimulator` (CPU simulation), only the bulk H2D/D2H channels are real sockets.
- Token socket IDs (`h2d_token_socket_id`, `d2h_token_socket_id`) are **omitted** since they are not needed in bulk-only mode.
- `num_users = 1` -- single user for endurance testing.
- `max_turns = 400` -- long-running stress test (400 inference cycles).
- `pcie_bandwidth_gbps = 25.0` -- documents a different PCIe configuration (Gen4 x8 or Gen5 link), though as noted in [Section 2.1.3](./01_config_schema_reference.md) this value is **never used** in any calculation.
- Each phase has **exactly one stage** -- simplified timing model.

### Differences from Full Config

| Property | Full (`pi05_config.json`) | Hybrid (`pi05_config_hybrid.json`) |
|----------|--------------------------|-------------------------------------|
| `num_users` | 8 | 1 |
| `max_turns` | 20 | 400 |
| `socket.mode` | `"full"` (implicit default) | `"bulk_only"` |
| `pcie_bandwidth_gbps` | 16.0 | 25.0 |
| Token socket IDs | Present | Absent |
| Vision stages | `[1000, 2200, 2200, 2200]` | `[2200]` |
| Text stages | 20 x 591 | `[591]` |
| Denoise stages | 6 x 757 | `[757]` |

### Phase-by-Phase Numerical Walkthrough

#### Vision Phase

| Property | Value |
|----------|-------|
| Stages | `[2200]` (1 stage) |
| `loop_count` | 1 (default) |
| `effective_stages()` | $1 \times 1 = 1$ |
| `effective_latency_us()` | $2{,}200 \;\mu s$ |

#### Text Phase

| Property | Value |
|----------|-------|
| Stages | `[591]` (1 stage) |
| `loop_count` | 1 (default) |
| `effective_stages()` | $1 \times 1 = 1$ |
| `effective_latency_us()` | $591 \;\mu s$ |

#### Denoise Phase

| Property | Value |
|----------|-------|
| Stages | `[757]` (1 stage) |
| `loop_count` | 5 |
| `effective_stages()` | $1 \times 5 = 5$ |
| `effective_latency_us()` | $757 \times 5 = 3{,}785 \;\mu s$ |

### Aggregate Pipeline Metrics

$$\texttt{total\_stages} = 1 + 1 + 5 = 7$$

$$\texttt{total\_latency\_us} = 2{,}200 + 591 + 3{,}785 = 6{,}576 \;\mu s$$

$$\texttt{effective\_stage\_us} = \lfloor 6{,}576 \div 7 \rfloor = \lfloor 939.43\ldots \rfloor = 939 \;\mu s$$

With only 7 effective stages and a single user, the pipeline has minimal parallelism. Each turn completes in approximately $939 \times 7 = 6{,}573$ us (6.6 ms), which aligns closely with the sum of actual stage durations.

### Why Single-Stage Phases?

The hybrid config collapses each phase to a single representative stage. This design choice targets a specific benchmarking scenario: measuring bulk transfer overhead against a simplified pipeline, where the interest is in the H2D/D2H socket performance rather than pipeline scheduling dynamics. The 400-turn endurance run provides statistical depth for H2D/D2H transfer latency measurements.

### Auto-Computed Stagger

$$\texttt{stagger\_us} = \lfloor 6{,}576 \div 1 \rfloor = 6{,}576 \;\mu s$$

With a single user, the stagger value is irrelevant (no other users to stagger against).

---

## 2.3.3 `pi05_config_sim.json` -- Simulation-Only Mode

**File:** `examples/pi05_config_sim.json`

```json
{
  "num_users": 8,
  "input_tokens": 768,
  "output_tokens": 1,
  "max_turns": 20,
  "phases": [
    {
      "name": "vision",
      "stages": [1000, 2200, 2200, 2200]
    },
    {
      "name": "text",
      "stages": [591, 591, 591, 591, 591, 591, 591, 591, 591, 591,
                 591, 591, 591, 591, 591, 591, 591, 591, 591, 591]
    },
    {
      "name": "denoise",
      "stages": [757, 757, 757, 757, 757, 757],
      "loop_count": 5
    }
  ]
}
```

### Operating Mode

This is the **simulation-only** configuration. The defining characteristic is the **absence** of both `"socket"` and `"pixel_payload"` blocks, which means:

- `use_sockets` remains `false`
- The simulation falls through to the `PipelineSimulator` backend (lines 1208-1218)
- No real H2D/D2H transfers occur
- `PixelPayloadConfig` uses all defaults (not relevant since there are no bulk transfers)
- Can be run with the base `pi05_pipeline_runner` binary (no `_device` suffix), which is built without socket dependencies

### What's Missing vs. Full Config

| Feature | `pi05_config.json` | `pi05_config_sim.json` |
|---------|--------------------|-----------------------|
| Socket block | Present | Absent |
| Pixel payload block | Present | Absent |
| `use_sockets` | `true` | `false` |
| Compile requirement | `PI05_HAS_SOCKETS` | None |
| Bulk H2D/D2H transfers | Real socket I/O | Not performed |
| Token pipeline | SocketPipeline | PipelineSimulator |

### Phase Structure: Identical to Full Config

The phases are **exactly the same** as `pi05_config.json`. This is intentional -- the simulation-only config is designed to model the same pipeline timing without requiring socket hardware. All the numerical results computed in Section 2.3.1 apply:

| Metric | Value |
|--------|-------|
| `total_stages` | 54 |
| `total_latency_us` | 42,130 us |
| `effective_stage_us` | 780 us |
| `max_stage_duration_us` | 2,200 us |
| `max_non_vision_stage_us` | 757 us |
| `stagger_us` (auto) | 5,266 us |

### Runtime Behavior

Without the socket block, the binary compiled without `PI05_HAS_SOCKETS` takes the simulation path:

```cpp
// examples/pi05_pipeline_runner.cpp:1208-1218
pl::PipelineSimulatorConfig sim_config{
    .num_stages = total_stages,            // 54
    .stage_duration_us = effective_stage_us, // 780
    .accept_rate = accept_rate,             // 1.0
    .seed = seed,                           // 42
    .batch_prefill = true,
};
pm::DecodeScheduler mgr(sim_config, pm::SchedulerParams{
    .max_users = num_users,                 // 8
    .skip_eos_writeback = true,
});
```

The `PipelineSimulator` creates 54 virtual pipeline stages, each introducing 780 us of simulated latency per token. Combined with the `DecodeScheduler`, this produces realistic multi-user scheduling behavior with TTFT, ITL, and TPOT metrics -- all without touching hardware.

---

## 2.3.4 Summary Comparison and the Single-Action-Token Model

### Comparative Summary

| Property | `pi05_config.json` | `pi05_config_hybrid.json` | `pi05_config_sim.json` |
|----------|-------------------|--------------------------|----------------------|
| **Mode** | Full socket | Bulk-only hybrid | Simulation-only |
| **`use_sockets`** | `true` | `true` | `false` |
| **Socket `mode`** | `"full"` (default) | `"bulk_only"` | N/A |
| **Token path** | SocketPipeline | PipelineSimulator | PipelineSimulator |
| **Bulk transfers** | Real sockets | Real sockets | None |
| **Users** | 8 | 1 | 8 |
| **Turns** | 20 | 400 | 20 |
| **`output_tokens`** | 1 | 1 | 1 |
| **Total stages** | 54 | 7 | 54 |
| **Total latency (us)** | 42,130 | 6,576 | 42,130 |
| **Effective stage (us)** | 780 | 939 | 780 |
| **Bottleneck (us)** | 2,200 | 2,200 | 2,200 |
| **Stagger (us, auto)** | 5,266 | 6,576 | 5,266 |
| **Hardware required** | Yes | Partial | No |

**Common across all three configs:**
- `input_tokens = 768` -- models the tokenized representation of a Pi0.5 robotics prompt
- `output_tokens = 1` -- models the single-action-token output architecture
- Same three-phase pipeline: vision, text, denoise
- Denoise always uses `loop_count = 5`

### The Single-Action-Token Model

All three configs set `output_tokens = 1`. This is the fundamental constraint of Pi0.5's robotics inference architecture: the model consumes a rich multi-frame visual observation (451,584 bytes of pixel data) and produces a single action token via the token pipeline. The actual continuous action output (50 joint positions as bfloat16, totaling 3,200 bytes via `ACTION_OUTPUT_BYTES` at line 203) is delivered through the bulk D2H channel. The token itself serves as a synchronization signal that triggers the bulk D2H read.

This `output_tokens = 1` setting directly impacts the pipeline's throughput characteristics: there is no autoregressive decode loop. Each "generation" is a single forward pass through all 54 effective stages (in the full config), making the TTFT (time to first token) effectively equal to the total pipeline latency. The implications for the `PipelineSimulator` timing model are explored in [Chapter 3](../ch3_pipelinesimulator_timing_model/index.md).

### Testing Progression

The three configs form a progression from pure-software testing (`_sim`) through single-device PCIe characterization (`_hybrid`) to full multi-device socket benchmarking (default config). This progression allows developers to validate scheduling correctness, tune timing parameters, and measure real hardware bandwidth at each level of the stack.

---

**Next:** [Chapter 3 -- PipelineSimulator Timing Model](../ch3_pipelinesimulator_timing_model/index.md)
