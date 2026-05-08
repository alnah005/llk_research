# Chapter 8 -- Part 1: Model Analysis and Strategy Selection

This chapter walks through scaling a single-device TT-Symbiote model to T3K (8 Wormhole chips) and beyond. The process has four phases: profile and analyze (this file), adapt modules and distribute weights (Part 2), validate on hardware (Part 3), and optimize performance or escalate to pipeline parallelism (Part 4). Every section opens with the failure you will encounter if you skip the step, then shows the correct approach. The worked example is Gemma4-31B; each step generalizes with "For your model" guidance.

**Prerequisites:** [Ch2/04](../ch02_symbiote_core/04_end_to_end_model_flow.md) (end-to-end model flow, `DispatchManager` timing), [Ch3/01](../ch03_symbiote_multi_device/01_distributed_config_and_device_init.md) (`DistributedConfig`), [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md) (weight sharding strategies), [Ch6/03](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md) (patterns and antipatterns).

---

## The Scaling Ladder: Context

Every model port follows a progression of hardware rungs [Ch1/01](../ch01_device_topologies/01_physical_topologies.md):

| Rung | Hardware | Mesh Shape | CCL | Typical Model Size |
|------|----------|-----------|-----|--------------------|
| 1 | N150 / P150 | `(1,1)` | None | < 7B |
| 2 | N300 / P300 | `(1,2)` | Minimal | 7-15B |
| 3 | T3K / P150x8 | `(1,8)` | Full TP | 15-35B |
| 4 | TG / BHGLX | `(8,4)` | 2D TP | 35-100B |
| 5 | Multi-Galaxy | varies | PP + TP | > 100B |

This chapter focuses on Rung 3 (T3K). The same replacement infrastructure, `set_device()` call, and `model.generate()` integration work unchanged from Rung 1 through Rung 4; only the replacement dict contents and device parameters change.

---

## Step 1: Profile the Single-Device Baseline

### What Goes Wrong If You Skip This

You jump straight to multi-device replacement, spend days writing distributed modules, then discover that 80% of wall-clock time is spent in an operation that falls back to CPU. Your CCL optimizations are irrelevant because the bottleneck was never on-device compute. Worse, you have no golden-reference timing to measure improvement against.

### The Correct Approach

Run the model on a single device with `TT_SYMBIOTE_RUN_MODE=NORMAL` first. Use a minimal replacement dict to establish a working baseline [Ch2/04, Stage 6](../ch02_symbiote_core/04_end_to_end_model_flow.md):

```python
os.environ["TT_SYMBIOTE_RUN_MODE"] = "NORMAL"
os.environ["MESH_DEVICE"] = "N150"

nn_to_ttnn = {nn.Linear: TTNNLinearLLama, nn.SiLU: TTNNSilu}
modules = register_module_replacement_dict(model, nn_to_ttnn)
set_device(model, mesh_device)

for m in modules.values():
    m.preprocess_weights()
    m.move_weights_to_device()

# Warm-up pass (primes JIT, avoids 10-100x first-call skew)
model.generate(**inputs, max_new_tokens=1, use_cache=True)
DispatchManager.clear_timings()

# Timed pass
model.generate(**inputs, max_new_tokens=32, use_cache=True)
DispatchManager.save_stats_to_file("baseline_timing.csv")
```

Parse the CSV to find CPU fallbacks and dominant modules:

```python
import pandas as pd
df = pd.read_csv("baseline_timing.csv")
fallbacks = df[df["backend"] == "Torch"]
top_modules = df.groupby("module_name")["duration"].sum().sort_values(ascending=False)
```

> **Warning:** Always run a warm-up pass and call `DispatchManager.clear_timings()` before the timed run. Without this, JIT compilation overhead inflates the first forward pass by 10-100x, making the data useless for identifying steady-state bottlenecks [Ch6/03, Pattern 8](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

> **Warning:** CPU fallbacks that are acceptable on N150 become catastrophic on T3K. Each fallback forces synchronization across all 8 devices, serializing the pipeline. Eliminate all CPU fallbacks in the hot path before scaling to multi-device.

---

## Step 2: Analyze the Architecture for Parallelism

### What Goes Wrong If You Skip This

You apply blanket `nn.Linear: TTNNLinearIColShardedWRowSharded` replacement to every linear layer. The model has a gating linear with `output_size=1` (like Qwen3.5's `shared_expert_gate`). The `reduce_scatter` crashes:

```
RuntimeError: reduce_scatter: output tensor dim 3 size 0 is not valid (must be >= 1)
```

This is Antipattern 2 from [Ch6/03](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md) -- blanket replacement of small gating linears with TP variants.

### The Correct Approach

Inspect the model tree systematically. Gemma4-31B reveals heterogeneous decoder layers [test_gemma4.py, line 8]:

| Property | Sliding Layers (50x) | Global Layers (10x) |
|----------|---------------------|---------------------|
| KV heads | 16 | 4 |
| head_dim | 256 | 512 |
| v_proj | Present | None (K=V sharing) |
| Window | 1024 tokens | Full context |

Discover this by inspecting `config.layer_types`:

```python
text_config = model.config.text_config
for i, lt in enumerate(text_config.layer_types[:text_config.num_hidden_layers]):
    print(f"Layer {i}: {lt}")
```

**Sharding divisibility check** -- run this for every linear before writing any replacement dict:

```python
N_DEVICES = 8
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear):
        if mod.in_features % N_DEVICES != 0 or mod.out_features % N_DEVICES != 0:
            print(f"NOT TP-safe: {name} shape ({mod.in_features}, {mod.out_features})")
```

> **Warning:** The automatic replication fallback in `get_tensor_config_for_tensor` prints a warning but does NOT error when a tensor cannot be sharded [Ch3/01](../ch03_symbiote_multi_device/01_distributed_config_and_device_init.md). If a weight that should be sharded silently falls back to replication, each device stores the full weight (wasting DRAM) and the matmul produces incorrect results. Always verify divisibility before writing the replacement dict.

**For your model:** Run the same tree dump and divisibility check. Record every unique `nn.Module` class, its parameter shapes, and how many instances exist. Check for these special structures:

| Feature | How to Detect | Impact on Strategy |
|---------|--------------|-------------------|
| MoE layers | `hasattr(config, 'num_experts')` | Expert parallelism needed [Ch3/04] |
| Sliding-window attention | `hasattr(config, 'sliding_window')` | Dual KV cache (Gemma4 pattern) |
| Heterogeneous layers | Varying `config.layer_types` | Per-layer strategy (Gemma4, Qwen3.5) |
| Shared expert gate | Small linear with `output_size=1` | Cannot shard; must exclude [Ch6/03, Antipattern 2] |
| Multimodal towers | Vision encoder with incompatible dims | Build exclusion set |
| Tied weights | `lm_head.weight is embed_tokens.weight` | Cannot shard independently |

---

## Step 3: Choose the Parallelism Strategy

### What Goes Wrong If You Skip This

You default to the Ling activation pattern (all col-sharded) for a model with fused QKV projections and per-head operations. The fused output is col-sharded but per-head slicing assumes replicated data. Silent numerical errors produce grammatically correct but semantically garbage output.

### The Strategy Decision Tree

```
Model fits in single-device DRAM (12 GB)?
  YES -> Latency acceptable?
           YES -> STOP. Single device is sufficient.
           NO  -> Enable TracedRun first [Ch2/03]. If still slow, continue.
  NO  -> Continue.

Total weight memory <= 96 GB (T3K aggregate)?
  YES -> hidden_dim % 8 == 0?
           YES -> T3K TP-8 via Symbiote. Choose activation pattern below.
           NO  -> Pad hidden_dim or use smaller mesh (N300, TP=2).
  NO, <= 384 GB -> TG TP-32 via Symbiote. See [Ch7/02].
  NO, > 384 GB  -> Blaze PP required. See Part 4 of this chapter.

Model has MoE layers?
  YES -> Expert parallelism via all_to_all_dispatch/combine [Ch3/04].
         Exclude shared_expert_gate (output=1) from TP.
  NO  -> Dense TP only.

Attention uses fused QKV with per-head ops (norms, partial RoPE)?
  YES -> Pattern B (Gemma4): TTNNLinearIColShardedWAllReduced for QKV.
         Produces replicated output for safe per-head operations.
  NO  -> Pattern A (Ling/Qwen): TTNNLinearIColShardedWRowSharded for all.
```

### The Two Activation Flow Patterns

**Pattern A (Column-Sharded Chain -- Ling/Bailing):** All activations stay col-sharded `[B, S, H/8]` throughout. Every linear uses `TTNNLinearIColShardedWRowSharded`. `TTNNDistributedRMSNorm` produces col-sharded output. Fewer CCL types but every linear adds a `reduce_scatter` [Ch7/02, Pattern 2](../ch07_integration_and_comparison/02_parallelism_strategies_compared.md).

**Pattern B (Alternating Sharded/Replicated -- Gemma4):** Fused projections use `TTNNLinearIColShardedWAllReduced` (replicated output). O-projection and down-projection use `TTNNLinearIReplicatedWColSharded` (col-sharded output). More CCL ops per layer but enables simpler attention because Q/K/V are fully replicated [Ch7/02, Pattern 3](../ch07_integration_and_comparison/02_parallelism_strategies_compared.md).

| Factor | Pattern A (Uniform RP) | Pattern B (Alternating) |
|--------|------------------------|-------------------------|
| CCL ops per block | 7-13 (every linear does RS) | ~6 (fused QKV + col-parallel skips) |
| Memory efficiency | Better (no replicated intermediates) | Slightly worse |
| Implementation | Lower (one linear class) | Higher (multiple classes, fused weights) |
| Best for | MoE models, shared experts | Dense models with fused QKV |

---

## Step 4: Identify Distributed Module Requirements

### What Goes Wrong If You Skip This

You replace only `nn.Linear` layers with distributed variants but leave `RMSNorm` as the default `TTNNRMSNorm`. On T3K, each device holds `[B, S, H/8]`. `TTNNRMSNorm` computes `mean(x^2)` locally on its `H/8` slice, producing an incorrect normalization factor. The model generates coherent-looking tokens initially, then diverges into repetitive garbage. No error is raised.

### The Complete Module Checklist

| Module | Single-Device | Multi-Device (T3K) | Notes |
|--------|--------------|-------------------|-------|
| RMSNorm | `TTNNRMSNorm` | `TTNNDistributedRMSNorm` | 3-phase all_gather for global stats |
| Per-Head Norm | `TTNNLocalRMSNorm` | `TTNNLocalRMSNorm` (same) | Operates on replicated heads |
| Embedding | `TTNNEmbedding` | `TTNNEmbedding` (same) | Auto-shards on dim=1 |
| RoPE | `TTNNRotaryPositionEmbedding` | `TTNNDistributedRotaryPositionEmbedding` | Replicates cos/sin |
| Linear (Q/K/V) | `TTNNLinear` | Strategy-dependent (see Step 3) | `@run_on_devices` gating |
| Decoder Layer | -- | Custom wrapper | On-device residuals, bypass flags |
| Model Wrapper | -- | Custom wrapper | `TTNNLayerStack`, `_decode_cache_position` |
| MoE | N/A | `TTNNQwen3MoE` / `TTNNBailingMoE` | `all_to_all` + `reduce_scatter` |

Source: [Ch3/04, Module Selection Guide](../ch03_symbiote_multi_device/04_distributed_modules.md).

---

## Step 5: Estimate Resource Requirements

### What Goes Wrong If You Skip This

You launch the T3K test and it hangs during weight loading with no error. After 10 minutes, the process is killed by the OOM reaper. The weight shards plus trace buffers plus KV cache exceeded per-device DRAM (12 GB on Wormhole).

### The DRAM Budget

**For your model**, use this memory estimation worksheet:

```
Per-layer weights (bfloat16):
  QKV projections:    3 * H * H * 2 bytes  (fewer if GQA)
  O projection:       H * H * 2 bytes
  MLP gate + up:      2 * H * I * 2 bytes
  MLP down:           I * H * 2 bytes
  Norms:              2 * H * 2 bytes       (negligible)
  Dense layer total:  ~(4*H^2 + 4*H*I) * 2 bytes

Embedding + LM head:  V * H * 2 bytes each

KV cache per token per layer:  2 * N_kv * d_head * 2 bytes
Total KV for S tokens:         L * S * 2 * N_kv * d_head * 2 bytes
```

**Gemma4-31B concrete budget** on T3K [test_gemma4.py, line 54]:

| Component | Memory |
|-----------|--------|
| Weights (bf16) | ~42 GB total / 8 = ~5.3 GB per device |
| Trace region | 200 MB |
| KV cache | Sliding (1024 tok) + global, replicated |
| Activation headroom | ~2 GB for intermediates during trace |
| **Total per device** | **< 10 GB (fits in 12 GB Wormhole DRAM)** |

If the budget exceeds 10 GB (leaving 2 GB headroom), consider `bfloat8_b` weights or reducing KV cache block count.

> **Warning:** `bfloat8_b` variants (`TTNNLinearLLamaIColShardedWRowSharded`) are `@trace_disabled` because `@deallocate_weights_after` creates dynamic allocation incompatible with trace replay [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md). You cannot use both bfloat8 weights and traced execution simultaneously.

---

## Summary: Analysis Outputs

At the end of this phase you should have:

1. **Baseline timing CSV** with per-module wall-clock data and CPU fallback identification.
2. **Module classification table** listing every module type, its parameter shapes, and TP eligibility.
3. **Activation flow pattern** choice (Pattern A or B) based on attention architecture.
4. **Multi-pass replacement plan** with pass ordering and exclusion sets.
5. **DRAM budget estimate** confirming the model fits in T3K per-device memory.

These five artifacts feed directly into Part 2, where you build the replacement dicts and preprocess weights for distribution.

---

| Previous | Up | Next |
|----------|-----|------|
| [Ch7/04: Limitations and Roadmap](../ch07_integration_and_comparison/04_limitations_gaps_and_roadmap.md) | [Table of Contents](../README.md) | [Part 2: Module Adaptation and Weight Distribution](02_module_adaptation_and_weight_distribution.md) |
