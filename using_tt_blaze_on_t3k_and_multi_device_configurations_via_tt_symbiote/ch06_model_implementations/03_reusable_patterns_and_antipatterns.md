# Chapter 6 -- Part 3: Reusable Patterns and Antipatterns

This section distills the cross-cutting patterns observed across all five models (GLM-4.7, Gemma4, Ling, Qwen3.5, DeepSeek V3) into actionable guidance. Each pattern is traced to a specific source location. Each antipattern includes the failure mode and the fix applied in the codebase. The section ends with a decision tree and new-model checklist.

**Prerequisites:** Parts 1 and 2 of this chapter.

---

## Pattern 1: Distributed RMSNorm (3-Phase All-Gather)

**Used by:** Gemma4, Ling, Qwen3.5 (all models with `TTNNDistributedRMSNorm`)

The 3-phase `pre_all_gather → all_gather → post_all_gather` pattern for column-sharded normalization is documented in [Ch3/03 "Distributed RMSNorm CCL Pattern"](../ch03_symbiote_multi_device/03_ccl_operations.md) and the full `TTNNDistributedRMSNorm` module in [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md). Always use `topology=Ring` to avoid dynamic allocations that break trace capture.

---

## Pattern 2: Column-Sharded Linear with Reduce-Scatter

**Used by:** Gemma4, Ling, Qwen3.5 (via `TTNNLinearIColShardedWRowSharded`)

The row-parallel linear pattern (weight sharded dim=-2, `reduce_scatter` to sum partials) and its `all_reduce` variant are documented in [Ch3/02 "Row-Parallel" and "All-Reduce Variant"](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md). The output stays column-sharded, so no all_gather is needed between layers when chaining sharded ops.

---

## Pattern 3: Multi-Pass Module Replacement

**Used by:** Gemma4 (3 passes), Ling (3 passes), Qwen3.5 (1-2 passes)

The replacement engine's recursive DFS walk and pass-ordering rules are covered in [Ch2/01](../ch02_symbiote_core/01_module_replacement_engine.md); the per-model 6-step skeleton is in Part 1, Section 6.1. The key addition for multimodal models is Gemma4's vision-module exclusion set (`test_gemma4.py`, line 123):

```python
exclude_vision = {name for name, _ in model.named_modules()
                  if "vision_tower" in name or "embed_vision" in name}
```

This prevents replacing vision-tower RMSNorm layers that have incompatible dimensions for 8-device sharding.

---

## Pattern 4: On-Device Residual Add (Eliminating Host Round-Trips)

**Used by:** Gemma4 (`TTNNGemma4DecoderLayer`), Ling (`TTNNBailingMoEDecoderLayer`)

The baseline approach (GLM-4.7) returns each layer output to host for `aten::add`. Each round-trip forces a `ttnn.synchronize_device` stall. The decoder-layer replacement pattern keeps residuals on device:

```python
residual = hs                          # stays as ttnn.Tensor on device
hs = self.input_layernorm(hs)          # device op
attn_out = self.attention(hs, ...)     # device ops
hs = ttnn.add(residual, attn_out)      # device op -- NO host trip
```

> **Warning:** Do NOT call `ttnn.deallocate(residual)` inside a `@trace_enabled` forward. During trace replay, the deallocate would free the pre-allocated trace input buffer, causing "Buffer is not allocated" on the next replay. Only deallocate tensors allocated within the traced forward.

---

## Pattern 5: Bypass Tensor Wrapping for Layer Stacks

**Used by:** Gemma4 (`TTNNGemma4TextModel`), Ling (`TTNNBailingMoeV2Model`)

Model-wrapper replacements set `_bypass_tensor_wrapping = True` on decoder layers to avoid wrapping/unwrapping `ttnn.Tensor` objects between consecutive layers (`bailing_moe_v2.py`, lines 45-50; `gemma4_text.py`, lines 46-49):

```python
for layer in model.layers:
    if isinstance(layer, TTNNModule):
        layer._bypass_tensor_wrapping = True
```

Without this, `set_device()` defaults to `_bypass_tensor_wrapping=False`, causing every inter-layer boundary to trigger a `TorchTTNNTensor` unwrap/rewrap that forces spurious host synchronizations. This is safe because no PyTorch ops touch `hidden_states` between layer calls.

---

## Pattern 6: Sequence-Length Padding for Trace Reuse

**Used by:** Ling (`TTNNBailingMoEDecoderLayerPadded`)

TTNN trace capture creates a fixed computation graph bound to specific tensor shapes. The padding wrapper (`decoder_layer.py`, lines 160-260):

1. Rounds `seq_len` up to next power of 2 (`_next_power_of_2`, line 148).
2. Pads `hidden_states`, `attention_mask`, `position_ids`, and `position_embeddings`.
3. Runs the inner decoder layer.
4. Slices output back to original `seq_len`.

This reduces unique trace keys from O(max_seq_len) to O(log(max_seq_len)), dramatically improving trace cache hit rate across conversation turns.

---

## Pattern 7: 3-Pass Centering TopK for BF16 MoE Precision

**Used by:** Ling (`TTNNMoERouterDecode`), Qwen3.5 (`TTNNQwenMoERouterDecode`)

With 256 experts, softmax/sigmoid scores cluster near `1/256 ~ 0.004`. BF16 has only ~3 decimal digits of precision at this magnitude, making naive `ttnn.topk` unreliable. The 3-pass technique (`qwen_moe.py`, lines 106-141):

```
Pass 1: topk(scores_bf16, k+1)       -> rough threshold (k+1-th value)
         scores_centered = scores_f32 - threshold_f32
Pass 2: topk(centered_bf16, k+1)     -> refined threshold (near zero)
         scores_recentered = centered_f32 - refined_f32
Pass 3: topk(recentered_bf16, k)     -> final indices (highest BF16 precision)
```

Intermediate subtractions happen in float32 to preserve precision. The decision boundary is progressively shifted toward zero where BF16 step size is smallest (~0.0001).

---

## Pattern 8: Mesh-Device Test Parametrization

**Used by:** All Symbiote tests (GLM-4.7, Gemma4, Ling, Qwen3.5)

Every test parametrizes `mesh_device` via a topology dictionary gated by the `MESH_DEVICE` env var. This lets the same test run on N150 through BHGLX without code changes:

```python
MESH_DEVICE_MAP = {
    "N150": {"N150": (1, 1)},
    "N300": {"N300": (1, 2)},
    "T3K":  {"T3K":  (1, 8)},
    "TG":   {"TG":   (8, 4)},
}
topo = os.environ.get("MESH_DEVICE", "N150")

@pytest.mark.parametrize("mesh_device", MESH_DEVICE_MAP[topo].values(), indirect=True)
def test_model(mesh_device):
    ...
```

The `indirect=True` delegates to a `mesh_device` fixture (from tt-metal's conftest) that calls `ttnn.open_mesh_device(mesh_shape)`. Device params (trace_region_size, fabric_config, num_command_queues) are set in a companion `device_params` fixture.

> **Warning:** The warmup-then-measure pattern is essential — run at least one full forward pass before timing. Call `ttnn.synchronize_device(mesh_device)` before starting the timer and after the timed pass to ensure accurate measurement.

---

## Antipattern 1: HF `model.device` After Full Replacement

**Symptom:** `StopIteration` during `model.generate()`.

**Cause:** After replacing all `nn.Module` children with TTNN modules, HF's `model.device` property (calls `next(self.parameters())`) finds no `nn.Parameter` objects.

**Fix:** Patch the class property (Gemma4: line 179; Ling: line 121):
```python
type(model).device = property(lambda self: torch.device("cpu"))
```

**Why this remains an antipattern:** The patch modifies the class (not the instance), affecting all instances and silently lying about data location. The correct fix is either a sentinel `nn.Parameter` or an override at the `TTNNModule` base class level.

---

## Antipattern 2: Replacing Small Gating Linears with TP Variants

**Symptom:** Crash in `reduce_scatter` with a 1-element output dimension.

**Cause:** Blanket `nn.Linear` replacement catches `shared_expert_gate` (output_size=1) which cannot be column-sharded across 8 devices.

**Fix:** Use targeted class-level replacement instead of blanket `nn.Linear` replacement. Leave generic `nn.Linear` commented out for models with small gating layers (`test_qwen3_5_35b_a3b.py`, lines 207-209).

---

## Antipattern 3: Dynamic Allocation Inside Traced Forward

**Symptom:** "Buffer is not allocated" or address aliasing on trace replay.

**Cause:** `ttnn.from_torch()`, `ttnn.all_reduce()` (internal intermediate), or `ttnn.deallocate()` of trace input buffers executed during capture.

**Fix:**
- Move `ttnn.from_torch` calls outside the traced forward (e.g., to the model wrapper's `call` method).
- Decompose `all_reduce` into `reduce_scatter + all_gather` (both have deterministic memory footprints).
- Never deallocate the `residual` tensor (the trace's pre-allocated input buffer).

---

## Antipattern 4: Vision Tower Replacement on Multi-Device

**Symptom:** Shape mismatch during `preprocess_weights()`.

**Cause:** Gemma4's `Gemma4RMSNorm` is shared between language model and vision tower. Vision norms have incompatible shapes for 8-device sharding.

**Fix:** Use `exclude_replacement` with a programmatic set of module names (`test_gemma4.py`, lines 123-131):
```python
exclude_vision = {name for name, _ in model.named_modules()
                  if "vision_tower" in name or "embed_vision" in name}
```

---

## Antipattern 5: Test Utility Dependencies in Production Code

**Symptom:** Fragile import dependency between production and test code.

**Cause:** `deepseek_decoder.py` (line 28) imports `create_decoder_block_tensors` from a test file. The code itself acknowledges this: `# TODO: This shouldn't live in the test file`.

**Fix:** Refactor shared utilities into a proper production module.

---

## Antipattern 6: Environment Variable Debug Overrides

**Symptom:** Combinatorial explosion of untested execution modes.

**Cause:** Qwen3.5 has 4 env vars (`TT_QWEN_CPU_LINEAR_ATTN`, `TT_QWEN_CPU_EXPERTS`, `TT_SYMBIOTE_RUN_MODE`, `TTNN_LINEAR_ATTN_PROJECTIONS`) creating 2^4 = 16 configurations. The cache compatibility check (lines 240-245) already shows the complexity.

**Fix:** Replace env var flags with explicit configuration objects that validate compatible option combinations.

---

## Antipattern 7: Hardcoded Hardware Constants

**Symptom:** Silent failures or wrong results on different hardware generations.

**Cause:** `MOE_SENDER_CORE = ttnn.CoreCoord(12, 9)` (`deepseek_decoder.py`, line 36) and `PIPELINE_CORE_COORD = ttnn.CoreCoord(11, 0)` assume specific grid sizes.

**Fix:** Query grid sizes from the device at runtime. Gemma4 does this correctly: `grid = self.device.compute_with_storage_grid_size()` (`gemma4_attention.py`, line 249).

---

## Quick Reference: Pattern Decision Tree

```
New model to port?
  |
  +-- How many module types to replace?
  |     1-2 types   -> Single-pass (GLM pattern)
  |     3+ types    -> Multi-pass ordered (Ling/Gemma4 pattern)
  |
  +-- Multi-device CCL needed?
  |     No  -> Omit fabric_config (GLM pattern)
  |     Yes -> FABRIC_1D_RING + Ring topology for all CCL in traced code
  |
  +-- MoE or dense?
  |     Dense -> TTNNLinearIColShardedWRowSharded for TP
  |     MoE   -> Custom decoder layer, check small gating layers
  |
  +-- Paged attention needed?
  |     No  -> use_cache=True with HF default
  |     Yes -> TTNNPagedAttentionKVCache + reset between runs
  |
  +-- Pipeline (Blaze) or module replacement (Symbiote)?
        Symbiote -> HF generate loop, module-by-module replacement
        Blaze    -> PipelineConfiguration, factory-driven stages, persistent mode
```

## Checklist for Adding a New Model

1. Identify HF module classes to replace (inspect `model.model.layers[0].__class__` etc.)
2. Create `nn_to_ttnn` dicts ordered by specificity (high-level first)
3. Use `exclude_replacement` for multimodal models with shared norm classes
4. Add `MESH_DEVICE_MAP` with all topologies (N150 through BHGLX)
5. Set `trace_region_size` (50 MB for small models, 200 MB for 30B+ models)
6. Add `fabric_config=ttnn.FabricConfig.FABRIC_1D_RING` if multi-device CCL is used
7. Set `_bypass_tensor_wrapping = True` on decoder layers in model wrapper
8. Patch `model.device` if all parameters are replaced
9. Add warmup + reset + timed run for paged KV cache models
10. Call `TracedRun.release_all()` at the end

---

## Summary: TP vs PP for a New Model

| Factor | Symbiote (Tensor Parallel) | Blaze (Pipeline Parallel) |
|--------|---------------------------|--------------------------|
| **Model integration** | Drop-in HF replacement | Custom stage factories + fused kernels |
| **Device utilization** | All devices run every layer | Each submesh runs assigned layers |
| **CCL per layer** | ~9-13 collectives (norms + linears + MoE) | 0 cross-stage CCL; 2 intra-submesh |
| **Latency** | Higher per-layer CCL overhead | Higher pipeline fill/drain overhead |
| **Memory** | Each device holds 1/N of every layer | Each submesh holds full layers for its stages |
| **Dispatch** | Fast dispatch (default) | Slow dispatch required |
| **Best for** | Models up to ~35B on T3K/TG | Models > 100B requiring multi-pod distribution |

---

| Previous | Up | Next |
|----------|-----|------|
| [Part 2: Blaze DeepSeek V3 Pipeline](02_blaze_deepseek_v3_pipeline.md) | [Table of Contents](../README.md) | [Ch7: Next Steps](../ch7/) |
