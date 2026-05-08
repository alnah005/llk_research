# Chapter 8 -- Part 3: Test Setup and Validation

This section covers the test infrastructure for multi-device execution: environment configuration, fixture setup, a 5-step run-mode progression from CPU through TRACED, debugging decision trees for common failures, and a comprehensive failure mode catalog. Every section starts with the error you will encounter when setup is wrong.

**Prerequisites:** Part 2 of this chapter (module adaptation complete), [Ch2/03](../ch02_symbiote_core/03_run_modes.md) (run modes, TracedRun lifecycle), [Ch6/03, Pattern 8](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md) (mesh-device test parametrization).

---

## Step 1: Environment and Device Configuration

### What Goes Wrong If You Skip This

You set `MESH_DEVICE=T3K` but forget `fabric_config`. The test starts, devices open, weights transfer -- then the first `reduce_scatter` hangs indefinitely:

```
TIMEOUT: ttnn.reduce_scatter did not complete within 300s.
Device synchronization failed -- possible fabric deadlock.
```

Without `fabric_config=ttnn.FabricConfig.FABRIC_1D_RING`, the fabric layer is not initialized and CCL operations cannot route data between devices [Ch6/03, Checklist item 6](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

### The Correct Approach

```python
MESH_DEVICE_MAP = {
    "N150":  (1, 1),  "N300":  (1, 2),  "T3K":   (1, 8),
    "TG":    (8, 4),  "P150":  (1, 1),  "P300":  (1, 2),
    "P150x8": (1, 8), "BHGLX": (8, 4),
}
topo = os.environ.get("MESH_DEVICE", "N150")

@pytest.mark.parametrize("device_params", [{
    "trace_region_size": 200_000_000,   # 200 MB for 30B+ models
    "num_command_queues": 1,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
}], indirect=True)
@pytest.mark.parametrize("mesh_device",
    [MESH_DEVICE_MAP[topo]], indirect=True)
def test_model_t3k(mesh_device):
    ...
```

### Key Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MESH_DEVICE` | `T3K` | Selects mesh shape `(1, 8)` |
| `fabric_config` | `FABRIC_1D_RING` (1D) / `FABRIC_2D` (TG) | Enables fabric routing for CCL |
| `trace_region_size` | 200 MB (60 layers), 300-400 MB (deep MoE) | DRAM reserved for trace buffers |
| `num_command_queues` | `1` | Single CQ for traced execution |
| `TT_SYMBIOTE_DISPATCHER` | `CPU` | Required for TRACED mode |

> **Warning:** The `trace_region_size` is a hard reservation. If too small, `ttnn.begin_trace_capture` crashes with "trace region exhausted." If too large, it steals DRAM from weights and KV caches. For Gemma4-31B, 200 MB is the empirical sweet spot [test_gemma4.py, line 54].

> **Warning:** On TG `(8,4)`, use `FABRIC_2D` instead of `FABRIC_1D_RING`. Using 1D ring on a 2D mesh causes incorrect CCL routing and silent data corruption [Ch1/01](../ch01_device_topologies/01_physical_topologies.md).

---

## Step 2: The 5-Step Run-Mode Progression

Each mode catches a different class of errors. Progress through them in strict order.

```
Step 1: CPU mode -- Does the model load at all?
  export TT_SYMBIOTE_RUN_MODE=CPU
  Verifies: HF model loads, replacement dict is correct, no import errors.
  FAILURE -> Fix class names in replacement dict.

Step 2: NORMAL_WITH_FALLBACK -- Which ops fail on TTNN?
  export TT_SYMBIOTE_RUN_MODE=NORMAL_WITH_FALLBACK
  Verifies: Weight shapes correct, most ops dispatch to TTNN.
  Logs CPU fallbacks to stdout for investigation.
  FAILURE -> Check weight shapes, device init, fabric config.

Step 3: SEL -- Are TTNN results numerically correct?
  export TT_SYMBIOTE_RUN_MODE=SEL
  Runs every op on both TTNN and PyTorch, compares results.
  Catches: shape mismatches, CCL errors, missing replacements.
  FAILURE -> See "Debugging SEL Failures" below.

Step 4: NORMAL -- Does the full model produce correct text?
  export TT_SYMBIOTE_RUN_MODE=NORMAL
  Full TTNN execution, no comparison overhead.
  Captures performance baseline via DispatchManager.
  FAILURE -> Enable DPL_NO_ERROR_PROP for per-op error isolation.

Step 5: TRACED -- Does traced execution produce identical results?
  export TT_SYMBIOTE_RUN_MODE=TRACED
  export TT_SYMBIOTE_DISPATCHER=CPU
  Three-phase lifecycle: warmup -> capture -> replay [Ch2/03].
  FAILURE -> See "Debugging Trace Failures" below.
```

> **Warning:** SEL mode approximately doubles execution time and memory. On T3K with a 31B model, expect 2x wall-clock and ~10 GB per device DRAM usage. If DRAM is tight, use `DPL_NO_ERROR_PROP` instead -- it isolates per-operation errors without doubling memory [Ch2/03](../ch02_symbiote_core/03_run_modes.md).

**Acceptable SEL thresholds for bfloat16** [test_gemma4.py, validation]:
- `max_error < 0.1` for linear layers
- `max_error < 0.5` for attention (accumulated multi-op error)
- `mean_error < 0.01` for all modules

---

## Step 3: The Warmup-Trace-Measure Pattern

Gemma4 uses a three-run pattern [test_gemma4.py, lines 181-191]:

```python
# Run 1: Warmup (2 tokens) -- primes lazy buffers, first-call side effects
outputs = model.generate(**inputs, max_new_tokens=2,
                         past_key_values=kv_cache, use_cache=True)
kv_cache.reset()

# Run 2: Trace capture (4 tokens) -- records device traces for @trace_enabled modules
outputs = model.generate(**inputs, max_new_tokens=4,
                         past_key_values=kv_cache, use_cache=True)
kv_cache.reset()

# Run 3: Measurement -- all traces replay from device memory
DispatchManager.clear_timings()
outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         past_key_values=kv_cache, use_cache=True)
ttnn.synchronize_device(mesh_device)
DispatchManager.save_stats_to_file("traced_timing.csv")
TracedRun.release_all()
```

**Why three runs:** Run 1 allocates persistent buffers (e.g., `_decode_cache_position`) and triggers lazy weight transfers. Run 2 captures the trace (within `TracedRun`, the first call to each `@trace_enabled` module is warmup, the second is capture). Run 3 replays from captured traces with clean timing counters.

> **Warning:** The KV cache must be reset between runs (`kv_cache.reset()`). Without reset, the second run appends to existing cached keys, causing dimension mismatches or stale attention masks [test_gemma4.py, line 185].

> **Warning:** `TracedRun` requires `TT_SYMBIOTE_DISPATCHER=CPU`. Running without it raises an assertion error. The run mode can only be set once -- calling `set_run_mode()` a second time raises an assertion [Ch2/03](../ch02_symbiote_core/03_run_modes.md).

---

## Step 4: Debugging Decision Trees

### Debugging SEL Failures

```
SEL reports numerical divergence at op X. What type is X?
  matmul? --> Check input sharding (shape[-1] should be hidden/8).
              If full-sized, the upstream CCL failed. If correct, check weight_dim.
  CCL op? --> Verify cluster_axis=1 (T3K), num_links=1, topology=Ring.
  Norm?   --> Check all_gather dim and cluster_axis in DistributedRMSNorm.
              Verify norm weight padding used value=1.0, not 0.0.
  Other?  --> Check tensor shapes at distributed/non-distributed boundaries.
```

### Debugging Trace Failures

```
Trace capture fails or replay produces wrong results.
  |
  Crash: "dynamic allocation during trace"?
    YES --> An op inside the traced module allocates a new buffer.
            Common: ttnn.all_reduce (monolithic), ttnn.Topology.Linear,
            @deallocate_weights_after inside a traced module.
            Fix: Decompose all_reduce to RS+AG. Switch to Ring topology.
  |
  Replay produces NaN or garbage?
    YES --> Buffer aliasing issue. Is a KV cache updated inside the trace?
              YES --> Pre-allocate persistent buffer on device.
                      Use ttnn.copy() to update (not tensor creation).
                      See Gemma4 _decode_cache_position [gemma4_text.py, lines 191-217].
              NO  --> Conditional branch inside traced forward?
                        YES --> Traces cannot contain branches. The captured
                                path replays regardless of condition.
  |
  Slightly different (not garbage) results?
    YES --> Ring topology floating-point nondeterminism.
            Expected tolerance: ~1e-3 for bfloat16.
```

### Debugging CCL Hangs

```
Test hangs (no output, no crash).
  fabric_config set in device_params?
    NO  --> Add fabric_config. This is the #1 hang cause.
    YES --> Correct fabric for mesh? (1D: FABRIC_1D_RING, 2D: FABRIC_2D)
            Correct cluster_axis? (T3K: must be 1. Using 0 on (1,8) is a no-op
            that hangs because devices wait for communication that never starts.)
            set_device() called before any CCL? (Triggers DistributedConfig init.)
```

---

## Step 5: Failure Mode Catalog

### Failure: Silent Numerical Divergence

**Symptom:** Model generates plausible but wrong tokens. No error raised.

**Root Cause 1:** `TTNNRMSNorm` used instead of `TTNNDistributedRMSNorm`. Local normalization on `H/8` produces incorrect scaling.

**Root Cause 2:** Missing `@run_on_devices(DeviceArch.T3K)` decorator. Single-device forward runs on multi-device, ignoring CCL [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md).

**Root Cause 3:** `get_tensor_config_for_tensor` silently fell back to replication for a weight that should be sharded. The print warning was missed.

**Fix:** Run SEL mode. Check every norm is the distributed variant. Verify all weight dimensions divisible by 8.

### Failure: Shape Mismatch After CCL

**Symptom:** `RuntimeError: shape mismatch` in `ttnn.linear` or `ttnn.add` following a CCL operation.

**Root Cause:** Output of `reduce_scatter` is `[B, S, H/8]` (col-sharded), but next operation expects `[B, S, H]` (replicated). Wrong linear variant or mismatched RMSNorm variant.

**Fix:** Verify the activation flow pattern end-to-end. Pattern A: all `TTNNLinearIColShardedWRowSharded`. Pattern B: fused projections use `TTNNLinearIColShardedWAllReduced` [Ch7/02](../ch07_integration_and_comparison/02_parallelism_strategies_compared.md).

### Failure: StopIteration During model.generate()

**Symptom:** `StopIteration` exception from HF's `model.device` property.

**Root Cause:** After full module replacement, no `nn.Parameter` objects remain [Ch6/03, Antipattern 1](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

**Fix:** `type(model).device = property(lambda self: torch.device("cpu"))`.

### Failure: Trace Replay Produces Garbage

**Symptom:** NORMAL mode correct; TRACED mode produces random output.

**Root Cause 1:** Dynamic allocation inside traced forward (`ttnn.from_torch`, monolithic `ttnn.all_reduce`, `ttnn.deallocate` of trace inputs) [Ch6/03, Antipattern 3](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

**Root Cause 2:** `_decode_cache_position` buffer not pre-allocated. Trace allocator aliases position address with intermediates, corrupting layers 1-N [gemma4_text.py, lines 170-217].

**Root Cause 3:** `ttnn.Topology.Linear` used in a traced module. Buffer addresses shift between capture and replay [Ch3/03](../ch03_symbiote_multi_device/03_ccl_operations.md).

**Fix:** Audit every `@trace_enabled` module's `forward()`. Move all tensor creation outside. Decompose all-reduce. Use Ring topology.

### Failure: OOM During Weight Loading

**Symptom:** Process killed by OOM reaper during `move_weights_to_device()`.

**Root Cause 1:** Per-layer RoPE allocation. Without shared caching, 60 layers x 16-32 MB = ~1.1 GB wasted [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md).

**Root Cause 2:** `trace_region_size` set too large, starving weight DRAM.

**Fix:** Use shared RoPE caches (class-level `_shared_rotary_setups`). Reduce `trace_region_size`. Switch to `bfloat8_b` if needed (with caveat: disables trace [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md)).

### Failure: CCL Hang (Indefinite)

**Symptom:** Process hangs during `reduce_scatter` or `all_gather`. No error message.

**Root Cause 1:** Missing `fabric_config` in device params.

**Root Cause 2:** Semaphore deadlock from mixed topologies -- one module uses Ring, another uses Linear within the same traced block.

**Fix:** Add `fabric_config=ttnn.FabricConfig.FABRIC_1D_RING` to device params. Use Ring topology uniformly for all CCL in traced code paths [Ch3/03](../ch03_symbiote_multi_device/03_ccl_operations.md).

---

## End-to-End Output Verification

After all validation modes pass, compare against a CPU golden reference:

```python
# CPU-only reference
model_cpu = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
outputs_cpu = model_cpu.generate(**inputs, max_new_tokens=32)
text_cpu = tokenizer.decode(outputs_cpu[0])

# T3K inference
outputs_t3k = model.generate(**inputs, max_new_tokens=32, past_key_values=kv_cache)
text_t3k = tokenizer.decode(outputs_t3k[0])

# First ~20 tokens should match exactly for bfloat16.
# Divergence after that is expected and acceptable.
```

---

| Previous | Up | Next |
|----------|-----|------|
| [Part 2: Module Adaptation and Weight Distribution](02_module_adaptation_and_weight_distribution.md) | [Table of Contents](../README.md) | [Part 4: Performance Optimization and Pipeline Escalation](04_performance_optimization_and_pipeline_escalation.md) |
