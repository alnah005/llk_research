# Chapter 8 -- Part 4: Performance Optimization and Pipeline Escalation

This section covers post-validation performance work: profiling CCL overhead, optimizing the hot path with incremental measurement, and deciding when to escalate from Symbiote TP to Blaze PP. Each section opens with the performance failure that motivates the optimization.

**Prerequisites:** Part 3 of this chapter (model passes TRACED validation on T3K), [Ch3/03](../ch03_symbiote_multi_device/03_ccl_operations.md) (CCL operations and costs), [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md) (PipelineGraph), [Ch7/02](../ch07_integration_and_comparison/02_parallelism_strategies_compared.md) (parallelism strategies compared), [Ch7/03](../ch07_integration_and_comparison/03_integration_pathways.md) (integration pathways).

---

## Step 1: Profile CCL Overhead

### What Goes Wrong If You Skip This

Your model runs on T3K and generates correct output. Decode latency is 150ms/token -- acceptable but not competitive. You spend two weeks optimizing matmul tile sizes, only to discover later that 60% of per-token time is CCL communication, not compute. The matmul optimizations improved latency by 3ms.

### The Correct Approach: CCL-Focused Profiling

Use the timing infrastructure to isolate CCL from compute [Ch2/04, Stage 6](../ch02_symbiote_core/04_end_to_end_model_flow.md):

```python
os.environ["TT_SYMBIOTE_SIGNPOST_MODE"] = "1"  # Tracy correlation

DispatchManager.clear_timings()
model.generate(**inputs, max_new_tokens=64)
ttnn.synchronize_device(mesh_device)
DispatchManager.save_stats_to_file("t3k_timing.csv")

import pandas as pd
df = pd.read_csv("t3k_timing.csv")
ccl_ops = df[df["func_name"].str.contains("reduce_scatter|all_gather|all_to_all")]
compute_ops = df[df["func_name"].str.contains("linear|matmul|rms_norm")]
ccl_total = ccl_ops["duration"].sum()
compute_total = compute_ops["duration"].sum()
total = df["duration"].sum()
print(f"CCL: {ccl_total:.1f}ms ({100*ccl_total/total:.1f}%)")
print(f"Compute: {compute_total:.1f}ms ({100*compute_total/total:.1f}%)")
```

### CCL Cost Model Per Topology

Using the cost model from [Ch3/03](../ch03_symbiote_multi_device/03_ccl_operations.md) for bfloat16 activation `[B=1, S=1, H=4096]`:

| Topology | Devices | Links/Hop | Ring Hops | RS+AG Latency |
|----------|---------|-----------|-----------|---------------|
| N300 (WH) | 2 | 1 | 1 | ~1 us |
| T3K (WH) | 8 | 1 | 7 | ~9 us |
| P150x8 (BH) | 8 | 2 | 7 | ~4.5 us |
| TG horizontal | 4 | 3 | 3 | ~1.6 us |
| TG vertical | 8 | 4 | 7 | ~2.8 us |

For Gemma4 with 6 CCL ops per layer at T3K: `60 * 6 * 9 us ~ 3.2 ms` pure CCL per decode token. At a 50 ms/token decode budget (20 tok/s), CCL is ~6.4% -- manageable. For MoE models with 13 CCL ops per layer: `60 * 13 * 9 us ~ 7 ms`, or ~14%.

> **Warning:** Tracy profiling with signpost mode adds ~5% overhead. Disable `TT_SYMBIOTE_SIGNPOST_MODE` for production timing. Use it only for correlating host-side module calls with device-side kernel execution [Ch2/03](../ch02_symbiote_core/03_run_modes.md).

---

## Step 2: Apply Optimizations

### Optimization 1: Fuse QKV Projections

The highest-impact CCL reduction. Replace three separate Q/K/V linear + `reduce_scatter` chains with a single fused QKV + decomposed all-reduce [gemma4_attention.py, lines 196-221]:

```
Before: 3 matmuls + 6 CCL (3 reduce_scatter + 3 all_gather)
After:  1 matmul + 2 CCL (1 reduce_scatter + 1 all_gather)
Savings: 4 CCL operations per layer (67% attention CCL reduction)
```

See Part 2, Step 2 for the `from_torch()` implementation with `_reverse_permute_weight`.

### Optimization 2: Fuse Gate-Up Projections in MLP

Same principle for MLP gate and up projections [gemma4_mlp.py, lines 36-49]:

```
Before: 2 matmuls + 4 CCL
After:  1 matmul + 2 CCL
Savings: 2 CCL per layer
```

### Optimization 3: Shared Rotary Setup Cache

Without caching, 60 layers x 16-32 MB per cos/sin cache = ~1.1 GB wasted DRAM. The shared cache reduces this to ~64 MB (2 unique configs) [gemma4_attention.py, lines 292-306]:

```python
_shared_rotary_setups = {}  # Class-level cache

def move_weights_to_device_impl(self):
    setup_key = (id(self.device), self.head_dim, rope_theta, partial_rotary_factor)
    if setup_key not in self.__class__._shared_rotary_setups:
        self.__class__._shared_rotary_setups[setup_key] = BailingRotarySetup(...)
    self._rotary_setup = self.__class__._shared_rotary_setups[setup_key]
```

### Optimization 4: Aggressive Tensor Deallocation

Free intermediate tensors immediately after consumption. Critical during trace capture because all allocations are baked into the trace's memory footprint [gemma4_mlp.py, lines 75-87]:

```python
gate_up = self.fused_gate_up_proj(hidden_states)
gate = ttnn.slice(gate_up, ...)
up = ttnn.slice(gate_up, ...)
ttnn.deallocate(gate_up)        # Free fused tensor immediately

gate = ttnn.gelu(gate, ...)
gate_up_mul = ttnn.multiply(gate, up)
ttnn.deallocate(gate)           # Free gate after multiply
ttnn.deallocate(up)             # Free up after multiply

output = self.down_proj(gate_up_mul)
ttnn.deallocate(gate_up_mul)    # Free intermediate
```

> **Warning:** Only deallocate tensors allocated **within** the traced forward. Never deallocate input arguments (trace input buffers) or the residual reference [Ch6/03, Antipattern 3](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

### Optimization 5: Sequence-Length Padding for Trace Reuse

Without padding, each unique `seq_len` creates a new trace key. Power-of-2 padding reduces unique trace keys from O(max_seq_len) to O(log(max_seq_len)) [Ch6/03, Pattern 6](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md):

```python
padded_len = 1 << (seq_len - 1).bit_length()
hidden_states = ttnn.pad(hidden_states, [0, 0, 0, padded_len - seq_len])
# ... run traced forward ...
output = ttnn.slice(output, [..., :seq_len, ...])
```

---

## Step 3: Measure Improvement Incrementally

### What Goes Wrong If You Skip This

You apply all five optimizations simultaneously and measure a 40% latency reduction. But you cannot attribute the improvement to any specific optimization. When a future change regresses performance, you do not know which optimization to check.

### The Correct Approach

Apply each optimization independently and measure:

```python
DispatchManager.clear_timings()
ttnn.synchronize_device(mesh_device)   # Clean start
model.generate(**inputs, max_new_tokens=64)
ttnn.synchronize_device(mesh_device)   # Ensure all ops complete
DispatchManager.save_stats_to_file(f"opt_{optimization_name}.csv")
```

Build a comparison table:

| Optimization | CCL ops/layer | Latency/token | Delta |
|-------------|--------------|---------------|-------|
| Baseline (separate Q/K/V) | 9 | 150 ms | -- |
| + Fused QKV | 5 | 130 ms | -13% |
| + Fused Gate-Up | 3 | 118 ms | -9% |
| + Shared RoPE cache | 3 | 115 ms | -3% (DRAM) |
| + Seq-len padding | 3 | 95 ms | -17% (trace hits) |
| + Aggressive dealloc | 3 | 90 ms | -5% (DRAM) |

> **Warning:** Always call `ttnn.synchronize_device(mesh_device)` before starting the timer and after the timed pass. Without synchronization, the timer captures host-side dispatch time only -- device execution may still be in flight, making TRACED mode appear faster than it actually is [Ch6/03, Pattern 8](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

### Optimization Priority Table

| Priority | Optimization | Impact | Effort |
|----------|-------------|--------|--------|
| 1 | Fused QKV projection | -4 CCL/layer | 1-2 days |
| 2 | On-device residual add (Part 2 Step 5) | -2 host round-trips/layer | Hours |
| 3 | Bypass tensor wrapping (Part 2 Step 6) | Eliminates inter-layer sync | Hours |
| 4 | TracedRun | Eliminates per-op dispatch | 1 day |
| 5 | Fused gate-up projection | -2 CCL/layer | 1 day |
| 6 | Sequence-length padding | O(log N) trace keys | 1 day |
| 7 | Shared rotary cache | -1 GB DRAM | Hours |
| 8 | Aggressive deallocation | Reduces trace DRAM footprint | Hours |

---

## Step 4: Decide Whether to Escalate to Blaze PP

### What Goes Wrong If You Skip This

You have a 70B model on TG with Symbiote TP-32. CCL overhead is 70% of per-token time because 32-way `reduce_scatter` requires 31 hops on a 2D mesh. You keep trying to optimize CCL but the bandwidth ceiling is physical -- each hop is limited by link bandwidth. You needed pipeline parallelism from the start.

### The Escalation Decision Framework

| Condition | Stay with Symbiote TP | Escalate to Blaze PP |
|-----------|----------------------|---------------------|
| Model <= 35B, CCL < 30% | Yes (T3K TP-8) | -- |
| CCL 30-50% after fusion | Optimize further | Consider Pathway C |
| CCL > 50% after all opts | -- | Yes |
| Fits in T3K DRAM (96 GB) | Yes | -- |
| Exceeds T3K DRAM | -- | Yes (or TG TP-32) |
| Batch size = 1 | Yes | -- |
| Continuous batching needed | -- | Yes (`PipelineManager` [Ch5/04]) |
| Model > 100B | -- | PP+TP required |

---

## Step 5: Pathway C -- Sequential Handoff to Blaze

The most practical escalation path [Ch7/03, Pathway C](../ch07_integration_and_comparison/03_integration_pathways.md): use the Symbiote prototype as numerical reference, identify pipeline stage boundaries at decoder-layer seams using the timing CSV, write Blaze stage factories using `PipelineGraph`.

```python
from pipeline_builder.graph import PipelineGraph, Node, Edge

graph = PipelineGraph(
    nodes={
        "embed":     Node(shape=ttnn.MeshShape(4, 2), factory=make_embed_stage),
        "dec_0_14":  Node(shape=ttnn.MeshShape(4, 2), factory=make_decoder_stage(0, 14)),
        "dec_15_29": Node(shape=ttnn.MeshShape(4, 2), factory=make_decoder_stage(15, 29)),
        "lmhead":    Node(shape=ttnn.MeshShape(4, 2), factory=make_lmhead_stage),
    },
    edges=[
        Edge("embed",     "dec_0_14"),
        Edge("dec_0_14",  "dec_15_29"),
        Edge("dec_15_29", "lmhead"),
        Edge("lmhead",    "embed", is_loopback=True),
    ],
)
layout = graph.build_topology(mesh_device)
pipeline = layout.build_pipeline()
pipeline.setup_and_run()
```

> **Warning:** `build_topology()` requires all nodes to have the same submesh shape and the number of nodes must equal the number of submeshes. A 32-chip galaxy with `MeshShape(4,2)` yields 4 submeshes, so the graph must have exactly 4 nodes [Ch5/01, Key Invariants](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md).

> **Warning:** Blaze requires `TT_METAL_SLOW_DISPATCH_MODE=1`, a process-global setting. You cannot mix Symbiote (fast dispatch) and Blaze (slow dispatch) in the same process. Escalation to Blaze is a full rewrite, not a partial migration [Ch7/03, Pathway A Blockers](../ch07_integration_and_comparison/03_integration_pathways.md).

---

## Step 6: The PP+TP Hybrid for Large Models

### Configuration Matrix

For models requiring both pipeline and tensor parallelism [Ch7/02, Hybrid PP+TP](../ch07_integration_and_comparison/02_parallelism_strategies_compared.md):

| Config | Stage Shape | TP Width | PP Depth | Total Chips | Max Model (bf16) |
|--------|------------|----------|----------|-------------|------------------|
| 1G-4x2 | `(4,2)` | 8 | 4 | 32 | ~140B |
| 1G-2x2 | `(2,2)` | 4 | 8 | 32 | ~140B |
| 2G-4x2 | `(4,2)` | 8 | 8 | 64 | ~280B |
| 4G-1x2 | `(1,2)` | 2 | 64 | 128 | ~560B |

Memory per device: `(P * bytes_per_param) / (PP_stages * TP_width)`. For a 70B model at bf16 with 1G-2x2: `70e9 * 2 / (8 * 4) = 4.4 GB` per device for weights. Add KV cache + trace region + activation headroom to verify fit.

Within each submesh, Blaze fuses CCL into the same kernel as compute -- per-block CCL drops to 2 ops versus 6-13 in Symbiote. Cross-submesh communication uses D2D sockets (single ethernet hop) [Ch7/02](../ch07_integration_and_comparison/02_parallelism_strategies_compared.md).

---

## The Complete Scaling Trajectory

```
N150 (1 chip)
  Profile -> identify bottlenecks -> replace modules -> validate (SEL)
    |
N300 (2 chips)
  Add fabric -> add DistributedConfig -> first CCL -> validate (SEL)
    |
T3K (8 chips)
  Full TP -> fuse QKV -> trace mode -> optimize CCL -> measure incrementally
    |
  CCL > 30%?  Model > 70B?  Need batching?
    |              |              |
    NO             YES           YES
    |              |              |
  Ship it.        v              v
              TG (32 chips)     Blaze PP
                2D fabric         PipelineGraph
                dual-axis CCL     stage factories
                                  PipelineManager
    |
  Model > 140B?
    |
    YES -> Multi-Galaxy (64-128 chips)
           Full Blaze PP+TP, 4+ stages per galaxy
```

Symbiote TP on T3K is the fastest path from HuggingFace model to multi-device inference. The optimization phase takes 3-5 days. Escalation to Blaze PP is justified only when the model exceeds T3K bandwidth capacity or requires continuous batching [Ch7/03](../ch07_integration_and_comparison/03_integration_pathways.md).

---

## Production Readiness Checklist

Before deploying your T3K model:

- [ ] NORMAL mode: coherent text for 10 diverse prompts
- [ ] SEL mode: `max_error < 0.5` for all modules (short sequence)
- [ ] DPL mode: error does not diverge over 128 tokens
- [ ] TRACED mode: output matches NORMAL within bfloat16 tolerance
- [ ] KV cache reset verified between independent requests
- [ ] Peak per-device DRAM usage < 10 GB (2 GB headroom)
- [ ] Decode tokens/sec meets latency target
- [ ] 1000 tokens generated without crash
- [ ] Varying sequence lengths handled without trace invalidation

---

| Previous | Up | Next |
|----------|-----|------|
| [Part 3: Test Setup and Validation](03_test_setup_and_validation.md) | [Table of Contents](../README.md) | -- |
