# Migration Guide

This final file provides practical migration paths for three developer audiences: Blaze developers adopting TensorAdapter, PyTorch developers new to Tenstorrent hardware, and TT-Symbiote users moving to fused-kernel execution. Each path addresses different starting knowledge, different goals, and different pitfalls. The guide concludes with a consolidated pitfall reference that catalogs the most common mistakes and their remedies.

**What you will learn:**

- A decision flowchart for choosing the right API entry point
- Migration path for existing Blaze developers: incremental adoption via the hybrid model
- Migration path for PyTorch developers: from zero to hardware execution in three steps
- Migration path for Symbiote users: from TTNN-level ops to fused LLK kernels
- Ten common pitfalls with symptoms, root causes, and remedies
- A quick-reference API cheat sheet

---

## Decision Flowchart: Which API Should I Use?

Before beginning migration, determine which API entry point is right for your situation:

```text
Start: I have a PyTorch model to run on Tenstorrent hardware.
  |
  +---> Is the model a standard architecture (LLaMA, Mistral, DeepSeek)?
  |       |
  |       +-- Yes --> Is there already a hand-tuned Blaze implementation?
  |       |            |
  |       |            +-- Yes --> Use the existing implementation
  |       |            |           (nothing to migrate)
  |       |            |
  |       |            +-- No  --> Use blaze.from_pytorch() for the full model
  |       |                        Override attention if KV cache is paged
  |       |
  |       +-- No  --> Does the model use only standard ops?
  |                    (matmul, layernorm, silu, softmax, add, mul)
  |                    |
  |                    +-- Yes --> Use blaze.from_pytorch()
  |                    |
  |                    +-- No  --> Does it use custom ops not in Blaze?
  |                                |
  |                                +-- Yes --> Use Symbiote (TTNNModule)
  |                                |           for unsupported ops
  |                                |           + BlazeModule for supported ops
  |                                |
  |                                +-- No  --> Use blaze.from_pytorch()
  |                                            with format_hints for unusual dtypes
  |
  +---> Am I optimizing an existing Blaze implementation?
          |
          +-- Yes --> Keep manual code for critical ops
          |           Wrap non-critical ops with blaze.from_pytorch()
          |
          +-- No  --> Start with blaze.from_pytorch()
                      Profile -> override -> manual (as needed)
```

---

## Supported Ops

Before migrating, verify that your model's operations are supported by the Blaze op library:

| PyTorch Op | Blaze Mapping | Status |
|-----------|--------------|--------|
| `torch.nn.Linear` | Matmul | Fully supported |
| `torch.nn.LayerNorm` | BroadcastRMSNorm | Supported (maps to RMSNorm variant) |
| `torch.nn.functional.silu` | Silu / ClampedSiLU | Fully supported |
| `torch.nn.functional.gelu` | Decomposed (mul + erf + add) | Supported via decomposition |
| `torch.nn.functional.relu` | EltwiseUnary | Fully supported |
| `torch.nn.functional.softmax` | SDPA (internal) | Supported within SDPA |
| `torch.nn.functional.scaled_dot_product_attention` | DsaSparseAttention | Supported (dense mode) |
| `torch.add`, `torch.mul`, `torch.sub` | EltwiseAdd/Mul/Sub | Fully supported |
| `torch.matmul` | Matmul | Fully supported |
| `torch.cat`, `torch.stack` | Data movement ops | Partial (may require fallback) |
| `torch.reshape`, `torch.view` | Reshape/Transpose | Supported when tile-aligned |
| Custom Python functions | N/A | Requires Level 2 manual op registration |

Ops not in this table fall back to Symbiote's TTNNModule dispatch or require Level 2 manual implementation.

---

## Path 1: Existing Blaze Developers

### Starting Point

You write `FusedProgram` code directly. You call `emit()`, compute CT args, size CBs, manage reconfig boundaries, and debug with `l1_profile.py`. You have deep knowledge of the hardware but spend too much time on plumbing.

### Goal

Reduce boilerplate for standard ops (matmul, MLP, normalization) while retaining full control for custom ops (attention pipelines, MoE routing, custom NOC patterns).

### Migration Strategy: Incremental Adoption via Hybrid Model

**Phase 1: Replace CT arg computation with derive\_user\_args()**

The lowest-risk change. Keep your existing graph construction and CB allocation. Replace only the manual `user_args` computation:

```python
# BEFORE -- manual user_args
user_args = {
    "gate_mm": {"k_num_tiles": 128, "out_w_per_core": 25},
    "down_mm": {"k_num_tiles": 344, "out_w_per_core": 9},
}

# AFTER -- auto-derived user_args
# Attach ShapeDescriptor to ExternalTensors
ext_act = ExternalTensor.from_torch("activation", act_tensor, op_hint="matmul.in0")
ext_gate_wt = ExternalTensor.from_ttnn("gate_weights", gate_tensor)
ext_down_wt = ExternalTensor.from_ttnn("down_weights", down_tensor)

# Build graph with shape metadata
with blaze.fuse(shape_engine=ShapeEngine()) as ctx:
    # ... same graph construction as before ...
    pass

auto_args = derive_user_args(ctx.graph)
# auto_args == {"gate_mm": {"k_num_tiles": 128, ...}, ...}
```

**Risk:** Near-zero. The graph and CB allocation are unchanged. Only the CT arg values change, and the `TensorAdapterValidator` confirms they match your manual values.

**Phase 2: Replace weight preprocessing with adapter.preprocess\_weights()**

The second-lowest-risk change. Replace your manual `ttnn.from_torch()` calls for weight tensors:

```python
# BEFORE -- manual weight preprocessing
weight_ttnn = ttnn.from_torch(
    linear.weight.T,
    dtype=ttnn.bfloat8_b,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        shard_spec=ttnn.ShardSpec(core_range_set, [K, N // num_cores], ...),
    ),
    tile=ttnn.Tile([32, 32]),
)

# AFTER -- automatic weight preprocessing
adapter = TensorAdapter(device=device, precision="performance")
shapes = adapter.analyze_shapes(model, sample_input)
weights = adapter.preprocess_weights(model, shapes)
# weights["gate_proj.weight"] is a ttnn.Tensor, pre-transposed, format-converted,
# padded, sharded, ready for device placement
```

**Risk:** Low. The resulting `ttnn.Tensor` objects are functionally equivalent. Verify by comparing `buffer_address()` and shard shapes.

**Phase 3: Replace standard op graphs with blaze.from\_pytorch()**

For standard patterns (MLP, normalization), replace the entire graph construction with `from_pytorch()`. Keep manual construction for custom ops:

```python
# BEFORE -- manual gated MLP graph
graph, user_args = build_gated_mlp_graph(config)
program = compiler.compile(graph=graph, tensors=tensors,
                            output_tensor=output, user_args=user_args)

# AFTER -- automatic gated MLP
blaze_mlp = blaze.from_pytorch(
    model.mlp, sample_input=torch.randn(1, 4096),
    device=mesh_device, precision="performance",
)
```

**Risk:** Medium. The compiled program should be functionally equivalent, but the fusion pattern and CB allocation may differ slightly. Verify with golden tests.

**Phase 4: Adopt the hybrid model for production**

The final architecture uses `blaze.from_pytorch()` for standard ops and raw `FusedProgram` for custom ops, connected via `ExternalTensor`:

```python
class ProductionDecoder:
    def __init__(self, model, device):
        self.norm = blaze.from_pytorch(model.norm, ...)
        self.mlp = blaze.from_pytorch(model.mlp, ...)
        self.attention = build_custom_attention(model.attention, device)

    def forward(self, hidden, kv_cache):
        normed = self.norm(hidden)
        attn_out = self.attention.run(normed, kv_cache)
        mlp_out = self.mlp(normed)
        return hidden + attn_out + mlp_out
```

### What Changes and What Does Not

| Component | Changes? | Details |
|-----------|----------|---------|
| CBHandle | No | Dataclass unchanged, same fields |
| FusedProgram | No | Same API: `cb_from_tensor()`, `cb_scratch()`, `build()` |
| CBEngine | No | Same assignment algorithm |
| SemEngine | No | Same semaphore allocation |
| CTArgEngine | No | Same resolution logic (receives auto-derived `user_args`) |
| BlazeOp hierarchy | No | Same `emit()` and `compose()` signatures |
| BlazeCompiler | No | Same `compile()` API |
| Your custom ops | No | Continue using raw `emit()` / `compose()` |
| Weight preprocessing | Yes (optional) | Can use `adapter.preprocess_weights()` instead of manual |
| CT arg computation | Yes (optional) | Can use `derive_user_args()` instead of manual |
| Standard op graphs | Yes (optional) | Can use `blaze.from_pytorch()` instead of manual graph |

---

## Path 2: PyTorch Developers New to Tenstorrent

### Starting Point

You write standard PyTorch code. You know `nn.Module`, `forward()`, `torch.Tensor`, and PyTorch's implicit contracts (shape inference, broadcasting, type promotion, memory management). You have zero knowledge of tiles, CBs, L1, or NOC.

### Goal

Run your PyTorch model on Tenstorrent hardware with minimal code changes and no hardware knowledge.

### Migration Strategy: Three Steps

**Step 1: Install and verify**

```python
import blaze
import torch.nn as nn

# Verify the device is accessible
device = blaze.open_device()
print(f"Device: {device}, cores: {device.num_cores}")
```

**Step 2: Convert your model**

```python
# Your existing PyTorch model (unchanged)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4096, 11008, bias=False)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(11008, 4096, bias=False)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))

model = MyModel()

# One-line conversion
blaze_model = blaze.from_pytorch(
    model,
    sample_input=torch.randn(1, 4096),
    device=device,
)

# Use it exactly like the original
output = blaze_model(torch.randn(1, 4096))
```

**Step 3: Verify correctness**

```python
# Compare against PyTorch CPU
with torch.no_grad():
    cpu_output = model(torch.randn(1, 4096))
    hw_output = blaze_model(torch.randn(1, 4096))

# Check numerical agreement
print(f"Max diff: {(cpu_output - hw_output).abs().max():.6f}")
# Expected: < 1e-4 for balanced profile, < 5e-3 for performance
```

### What You Need to Know (and What You Do Not)

| Topic | Need to Know? | Why |
|-------|---------------|-----|
| Tile shapes (32x32) | No | TensorAdapter selects them |
| Padding strategies | No | Per-op registry handles them |
| Data formats (bfloat8\_b) | Optional | Only if you choose `precision="performance"` |
| CB sizing | No | Automatic from shapes |
| L1 memory budget (~1.5 MB total, ~1.3 MB available) | No | Validated at compile time |
| Core grid | No | Auto-selected |
| CT args | No | Derived from shapes |
| Fusion | No | Automatic pattern detection |
| `FusedProgram` API | No | Abstracted away |
| Precision profiles | Recommended | Choose between speed and accuracy |

### Precision Profile Selection Guide

```text
Start here: precision="balanced" (default)
  |
  +-> Output matches PyTorch reference?
  |     |
  |     +-- Yes --> Is speed acceptable?
  |     |            |
  |     |            +-- Yes --> Done.
  |     |            +-- No  --> Switch to precision="performance"
  |     |                        (bfloat8_b, 1,088 B/tile; LoFi)
  |     |                        Re-check accuracy.
  |     |
  |     +-- No  --> Switch to precision="accuracy"
  |                  (float32 accumulation, HiFi4, fp32_dest_acc_en)
  |                  Re-check speed.
  |
  +-> Still inaccurate with "accuracy"?
        |
        +-- Check if the op is supported (see table above)
        +-- Check if weights loaded correctly
        +-- File a bug report with input/output comparison
```

### When You Will Need to Learn More

| Scenario | What to Learn | Guide Section |
|----------|--------------|---------------|
| Output quality is wrong | Precision profiles (Ch6) | [Ch6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md) |
| Model is too slow | L1 profiling, overrides (Ch10) | [File 05](./05_performance_implications_and_tradeoffs.md), [File 06](./06_escape_hatches_for_power_users.md) |
| Some ops are unsupported | Op mapping, fallback (Ch9) | [Ch9, File 04](../ch09_op_fusion/04_fusion_detection_and_limitations.md) |
| Need KV cache | ExternalTensor, hybrid model (Ch10) | [File 06](./06_escape_hatches_for_power_users.md) |
| Compile error: L1 exceeded | L1 budget, core grid (Ch7) | [Ch7, File 01](../ch07_cb_sizing/01_l1_memory_model.md) |

---

## Path 3: TT-Symbiote Users

### Starting Point

You use TT-Symbiote's `TTNNModule` for model deployment. Your models run through Symbiote's `__torch_dispatch__` interception, which replaces PyTorch modules with TTNN ops. Each op runs independently via TTNN, with intermediate data passing through DRAM.

### Goal

Move performance-critical subgraphs from TTNN-level execution (DRAM round-trips between ops) to fused LLK kernel execution (intermediates in L1). This is the transition from "works" to "works fast."

### Migration Strategy: Targeted Replacement

**Phase 1: Identify the bottleneck**

Profile your Symbiote model to find the hottest subgraph:

```python
# Existing Symbiote model
model = TTNNModule.from_torch(pytorch_model, device=device)
output = model(input_tensor)

# Profile with ttnn.profiler
profiler_output = ttnn.profiler.profile(model, input_tensor)
# Identify: MLP block takes 65% of latency
```

**Phase 2: Replace the bottleneck with BlazeModule**

```python
# Replace only the MLP block
blaze_mlp = blaze.from_pytorch(
    pytorch_model.mlp,
    sample_input=torch.randn(1, 4096),
    device=device,
    precision="performance",
)

# Hybrid model: Symbiote for most ops, BlazeModule for MLP
class HybridModel:
    def __init__(self, symbiote_model, blaze_mlp):
        self.symbiote = symbiote_model
        self.blaze_mlp = blaze_mlp

    def forward(self, x):
        # Symbiote handles attention, normalization, etc.
        normed = self.symbiote.norm(x)
        attn_out = self.symbiote.attention(normed)

        # BlazeModule handles MLP (fused, L1-resident intermediates)
        mlp_out = self.blaze_mlp(normed)

        return x + attn_out + mlp_out
```

**Phase 3: Expand to more subgraphs**

As you gain confidence, replace more Symbiote ops with BlazeModule:

```python
blaze_norm = blaze.from_pytorch(pytorch_model.norm, ...)
blaze_mlp = blaze.from_pytorch(pytorch_model.mlp, ...)
# Keep Symbiote for attention (KV cache management)
```

### Key Differences: TTNNModule vs. BlazeModule

| Dimension | TTNNModule | BlazeModule |
|-----------|-----------|-------------|
| Intermediate storage | DRAM (each op round-trips) | L1 (fused ops, no DRAM) |
| Op granularity | Per-op (matmul, add, silu separately) | Fused (matmul+silu+mul as one kernel) |
| Performance ceiling | Lower (DRAM bandwidth limited) | Higher (L1 bandwidth, compute limited) |
| Ease of use | Very high (drop-in replacement) | High (3-line API, similar ease) |
| Op coverage | Broad (all TTNN ops) | Narrower (ops with Blaze kernels) |
| Escape hatches | `exclude_replacement` set | Three-level override system |
| Weight format control | Limited (TTNN decides) | Full (profiles + format\_hints) |
| L1 visibility | None | `l1_profile.py` |

### The Tensor Bridge: Symbiote to Blaze

The critical integration point is passing tensors between Symbiote's `TorchTTNNTensor` and BlazeModule's input/output. The conversion path is:

```text
Symbiote output (TorchTTNNTensor)
  .ttnn_tensor -> ttnn.Tensor (on device, DRAM)
  ttnn.to_torch() -> torch.Tensor (on host)
  -> BlazeModule.forward(torch.Tensor)
  -> BlazeModule prepares activation (tilize, pad, place on device)
```

This path involves a host round-trip. For production, the `ExternalTensor` mechanism can avoid this:

```python
# PROPOSED -- direct device-to-device tensor handoff
symbiote_output = symbiote_model.norm(input_tensor)
# symbiote_output is a TorchTTNNTensor with .ttnn_tensor on device

# Create ExternalTensor from the device tensor
ext = ExternalTensor.from_ttnn("normed", symbiote_output.ttnn_tensor)

# Pass directly to BlazeModule (no host round-trip)
blaze_mlp_output = blaze_mlp.forward_from_device(ext)
```

This `forward_from_device()` method bypasses the host-side tensor preparation, accepting a device tensor directly. The `ShapeDescriptor` is inferred from the TTNN tensor's metadata.

---

## Consolidated Pitfall Reference

### Pitfall 1: Forgetting to Transpose nn.Linear Weights

**Symptom:** Matmul produces wrong results; output shape is correct but values are garbage.

**Root cause:** `nn.Linear` stores weights as `[out_features, in_features]`. The matmul kernel expects `[K, N]` where K = in\_features. If you pass the weight without transposing, the kernel multiplies by the wrong matrix.

**Remedy:** `blaze.from_pytorch()` handles this automatically. If using Level 2, always transpose: `linear.weight.T.contiguous()`.

### Pitfall 2: Using ZERO Padding for Softmax

**Symptom:** Attention weights are wrong; padded positions have non-zero probability.

**Root cause:** ZERO padding in the attention score matrix means padded positions have score 0.0. After `exp(0.0) = 1.0`, these positions receive non-zero attention weight, diluting attention to real positions.

**Remedy:** Use NEG\_INF padding for softmax inputs. `blaze.from_pytorch()` handles this automatically via the padding registry (P6: Per-Operation Padding Semantics). If using Level 2, explicitly set `padding_fill = float('-inf')` for attention score CBs.

### Pitfall 3: L1 Budget Exceeded with No Error

**Symptom:** Model produces wrong results or device hangs. No error message.

**Root cause:** Total CB allocations exceed available L1 (~1.3 MB available per core, out of ~1.5 MB total L1 SRAM). The current Blaze infrastructure has no compile-time L1 bounds checking. Overcommit manifests as silent data corruption or hardware hang.

**Remedy:** Use `blaze.from_pytorch()`, which validates L1 budget at compile time (P3: Validate at Each Lowering Step, Not Just at the End). If using Level 2, enable `BLAZE_L1_PROFILE=1` and check `print_cb_stats()` output. Sum all CB `total_size_B` values and verify the total is under the available L1 budget.

### Pitfall 4: fp32\_dest\_acc\_en Without dst\_full\_sync\_en

**Symptom:** Matmul throughput drops by 50% under the accuracy profile.

**Root cause:** Setting `fp32_dest_acc_en=True` without `dst_full_sync_en=True` drops `max_dest` from 8 to 4 tiles, halving the subblock size and doubling inner-loop iterations. This is the "dest register pressure footgun" described in [File 04](./04_worked_example_attention.md).

**Remedy:** Always set both flags together. `blaze.from_pytorch()` with `precision="accuracy"` handles this automatically. The `compute_subblock_w()` function in `utils.py` encodes the hardware constraint.

### Pitfall 5: Non-Aligned Shape with Unexpected Padding

**Symptom:** Output tensor has unexpected values at the boundaries.

**Root cause:** A tensor with shape `[1, 37]` is padded to `[32, 64]`. The output extraction step must slice back to `[1, 37]`. If the slice indices are wrong, padding values leak into the output.

**Remedy:** `blaze.from_pytorch()` handles output unpadding automatically using the `ShapeDescriptor.logical_shape`. If using Level 2, always track the logical shape and slice the output: `output[:logical_H, :logical_W]`.

### Pitfall 6: CB ID Exhaustion in Large Fused Ops

**Symptom:** `RuntimeError: CB ID exceeds maximum (64)` during compilation.

**Root cause:** The fused op uses more than 64 unique CB IDs. The 64-slot hardware limit is a hard constraint on Blackhole silicon.

**Remedy:** Insert `reconfig()` boundaries to enable CB ID reuse across phases. `blaze.from_pytorch()` inserts reconfig boundaries automatically. If using Level 2, call `f.reconfig()` between phases and use `CircularBufferIdManager` for ID management. See [File 03](./03_worked_example_gated_mlp.md) for a worked example of `compact_cb_ids()` exploiting non-overlapping lifetimes.

### Pitfall 7: Stale KV Cache After Model Recompilation

**Symptom:** Attention output is garbage after recompiling the model.

**Root cause:** The KV cache `ExternalTensor` holds a reference to a device tensor that was allocated during the previous compilation. After recompilation, the device memory layout may change, invalidating the old tensor.

**Remedy:** Always reinitialize `ExternalTensor` objects after `from_pytorch()`. Do not reuse device tensors across compilations.

### Pitfall 8: Format Mismatch at Fused Op Boundary

**Symptom:** Data corruption at the boundary between two fused ops.

**Root cause:** Phase 0 writes data in `bfloat8_b` format (1,088 B/tile). Phase 1 reads the same CB ID in `bfloat16` format (2,048 B/tile) after reconfig. The bytes are reinterpreted incorrectly.

**Remedy:** Ensure format consistency across reconfig boundaries. `blaze.from_pytorch()` handles this automatically via format negotiation (Chapter 6). If using Level 2, verify that producer and consumer CBs at reconfig boundaries use the same `data_format`.

### Pitfall 9: Using bfloat4\_b with Non-32x32 Tiles

**Symptom:** `ValueError: bfloat4_b requires 32x32 tiles, got (1, 32)`.

**Root cause:** Block floating-point formats (`bfloat8_b` at 1,088 B/tile, `bfloat4_b` at 576 B/tile) require 32x32 tile geometry because the shared exponent is computed over a 16-element face row within the 32x32 block. Non-32x32 tiles break the BFP encoding.

**Remedy:** Use `bfloat16` for ops that require non-32x32 tiles (e.g., RMSNorm with 1x32 tiles). `blaze.from_pytorch()` validates this at compile time.

### Pitfall 10: Assuming from\_pytorch() Handles Dynamic Shapes

**Symptom:** Model works for the sample input shape but fails for a different shape.

**Root cause:** `blaze.from_pytorch()` traces the module with a specific `sample_input` and compiles a program for that exact shape. Different input shapes require different tile grids, CB sizes, CT args, and potentially different fusion decisions.

**Remedy:** For variable-length inputs (e.g., different sequence lengths), use bucketed compilation:

```python
# Compile for multiple sequence lengths
blaze_models = {}
for seq_len in [128, 256, 512, 1024, 2048]:
    blaze_models[seq_len] = blaze.from_pytorch(
        model,
        sample_input=torch.randn(1, seq_len, 4096),
        device=device,
    )

# At inference time, select the closest bucket
def inference(x):
    seq_len = x.shape[1]
    bucket = min(k for k in blaze_models if k >= seq_len)
    return blaze_models[bucket](x)
```

---

## Quick Reference: API Cheat Sheet

```python
# === Level 0: Automatic ===
blaze_model = blaze.from_pytorch(model, sample_input, device)
output = blaze_model(input_tensor)

# === Level 1a: Precision profile ===
blaze_model = blaze.from_pytorch(model, sample_input, device,
                                  precision="performance")

# === Level 1b: Per-op overrides ===
blaze_model = blaze.from_pytorch(model, sample_input, device,
    precision="performance",
    overrides={
        "sdpa": {"fp32_dest_acc_en": True, "math_fidelity": "HiFi4"},
        "down_proj.matmul": {"subblock_k": 64},
    })

# === Level 1c: Format hints ===
blaze_model = blaze.from_pytorch(model, sample_input, device,
    precision="performance",
    format_hints={
        "q_proj.weight": ttnn.bfloat16,     # 2,048 B/tile
        "gate_proj.weight": ttnn.bfloat4_b,  # 576 B/tile
    })

# === Level 1d: Grid override ===
blaze_model = blaze.from_pytorch(model, sample_input, device,
    grid=custom_core_grid)

# === Level 1e: Scoped format override ===
with blaze.format_hint(dtype=ttnn.bfloat8_b):
    output = blaze_model(input_tensor)

# === Level 2: Manual bypass ===
f = FusedProgram(kernel=path, device=device, ...)
handle = f.cb_from_tensor(tensor)
out = CustomOp.emit(f, handle, prefix="custom")
result = f.build()

# === Hybrid: Mix automated and manual ===
blaze_mlp = blaze.from_pytorch(model.mlp, ...)
attn_program = build_custom_attention(...)
output = hidden + attn_program.run(normed) + blaze_mlp(normed)

# === ExternalTensor: Bridge between worlds ===
ext = ExternalTensor.from_torch("name", tensor, op_hint="matmul.in0")
ext = ExternalTensor.from_ttnn("name", ttnn_tensor)
ext = ExternalTensor.from_cb_handle("name", cb_handle)

# === Profiling ===
os.environ["BLAZE_L1_PROFILE"] = "1"
blaze_model.l1_profile()
```

---

## Conclusion

This guide has traced the complete path from PyTorch's implicit tensor contracts to Tenstorrent's explicit CB management, through ten chapters covering tile decomposition, padding, data formats, CB sizing, broadcasting, op fusion, and the end-to-end developer API. The central thesis is that a shape-aware adapter layer -- the TensorAdapter -- can bridge this gap while preserving both the developer experience (3-line API, zero hardware knowledge required) and the performance ceiling (byte-identical compiled kernels, zero runtime overhead).

The seven design principles that emerged from surveying six frameworks (TT-Symbiote, TVM, Triton, XLA, MLIR, torch.compile) govern every design decision:

| # | Principle | How This Chapter Applies It |
|---|-----------|---------------------------|
| P1 | Separate Computation Intent from Execution Strategy | `blaze.from_pytorch()` separates the module (intent) from the compiled program (strategy) |
| P2 | Carry Metadata from Both Worlds Simultaneously | `ShapeDescriptor` carries logical shape + padded shape + tile grid + format + padding through all 6 stages |
| P3 | Validate at Each Lowering Step, Not Just at the End | L1 budget checked at compile time, not at runtime crash |
| P4 | Provide Defaults with Per-Decision Overrides | Three override levels, each orthogonal: changing one decision does not require changing others |
| P5 | Operate on Graphs, Not Individual Tensors | Fusion analyzer operates on the full graph; format negotiation and L1 budgeting are graph-level passes |
| P6 | Per-Operation Padding Semantics | ZERO for matmul, NEG\_INF for softmax, automatically selected from registry |
| P7 | Transparent Interception That Preserves Identity | `BlazeModule` extends `nn.Module`; the developer calls `model(x)` unchanged |

The escape hatch architecture ensures that automation serves developers at every level: PyTorch newcomers who want zero-friction hardware access, experienced developers who want to tune specific decisions after profiling, and expert kernel authors who need full manual control for custom ops. The principle is simple: **automate the common case, provide overrides for every decision, and never force the developer to accept a suboptimal default without an escape route.**

---

**End of guide.** Return to [Guide Index](../index.md)
