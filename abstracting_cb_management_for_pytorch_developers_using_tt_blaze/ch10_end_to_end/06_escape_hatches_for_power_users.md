# Escape Hatches for Power Users

This is the deepest treatment in the chapter. Every automatic decision TensorAdapter makes has an explicit override mechanism. The escape hatch architecture is designed around the principle that **overriding one decision must not require overriding all others** (P4: Provide Defaults with Per-Decision Overrides). This file defines the three override levels, demonstrates each with worked examples including a progressive profiling workflow, documents the hybrid model for mixing automated and manual ops, explains the `ExternalTensor` bridge for cross-framework integration, and maps out the expert developer workflow.

**What you will learn:**

- Three override levels with concrete API examples: Level 0 (automatic), Level 1 (configurable), Level 2 (manual bypass)
- A worked example of progressive override: starting at Level 0, profiling, overriding to Level 1, profiling again, and dropping to Level 2 for a critical op
- The hybrid model: mixing TensorAdapter-managed ops and manually-configured ops within a single FusedProgram
- ExternalTensor: the bridge between Symbiote's TTNNModule, TensorAdapter's BlazeModule, and raw Blaze ops
- The expert workflow: profile, identify, override, verify

---

## The Three Override Levels

The three levels correspond to the developer journey established in [Chapter 1, File 04](../ch01_the_gap/04_abstraction_goals_and_scope.md):

```text
Level 0: AUTOMATIC
  Developer provides: module + sample_input + precision profile
  System decides:     Everything (tile, padding, format, CB sizing, CT args,
                      blocking, grid, fusion, reconfig boundaries)
  Override scope:     None -- full trust in automation
  Target audience:    PyTorch developers new to Tenstorrent

Level 1: CONFIGURABLE
  Developer provides: Level 0 inputs + targeted overrides for specific decisions
  System decides:     Everything NOT explicitly overridden
  Override scope:     Per-op settings, format hints, grid, fusion on/off
  Target audience:    Developers tuning performance after profiling

Level 2: MANUAL BYPASS
  Developer provides: Raw FusedProgram API calls for critical ops
  System decides:     Nothing for bypassed ops (full manual control)
  Override scope:     Complete -- developer takes full responsibility
  Target audience:    Expert Blaze developers, kernel authors
```

### Level 0: Automatic (No Overrides)

```python
# Level 0 -- zero configuration
blaze_model = blaze.from_pytorch(
    model,
    sample_input=torch.randn(1, 4096),
    device=mesh_device,
)
output = blaze_model(input_tensor)
```

All decisions use defaults:
- Precision: "balanced" (bfloat16, HiFi4)
- Format: uniform bfloat16 for all tensors (2,048 B/tile)
- Grid: auto-selected from device and weight shape
- Fusion: enabled (analyzer detects patterns)
- Blocking: heuristic (`subblock_k = max(1, Kt // 4)`)
- Padding: ZERO for all ops (NEG\_INF auto-selected for softmax)

This is the XLA-style experience: provide tensors and operations, get results. The system makes every decision. Performance is typically 85-100% of hand-tuned for standard shapes.

### Level 1: Configurable (Targeted Overrides)

Level 1 provides five override mechanisms, each targeting a different decision category:

#### 1a. Precision Profile Override

```python
# Override the global precision profile
blaze_model = blaze.from_pytorch(
    model, sample_input, device,
    precision="performance",  # bfloat8_b weights (1,088 B/tile), LoFi, math_approx
)
```

This single string changes the weight format, math fidelity, math approximation mode, and accumulation precision for every op in the model. It is the broadest override.

**Tile size impact of profile selection:**

| Profile | Weight Format | Tile Size | Relative to bfloat16 |
|---------|-------------|-----------|---------------------|
| balanced | bfloat16 | 2,048 B | 1.0x (baseline) |
| performance | bfloat8\_b | 1,088 B | 0.53x |
| (via format\_hint) | bfloat4\_b | 576 B | 0.28x |

The bfloat4\_b format (576 B/tile) is not directly selectable through a precision profile but can be applied to specific weights via `format_hints` or the `format_hint()` context manager. It provides the most aggressive L1 savings but requires careful validation of quantization error.

#### 1b. Per-Op Setting Override

```python
# Override specific settings for specific ops
blaze_model = blaze.from_pytorch(
    model, sample_input, device,
    precision="performance",
    overrides={
        "sdpa": {
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "math_approx_mode": False,
        },
        "lm_head": {
            "fp32_dest_acc_en": True,
        },
        "gate_mm": {
            "subblock_k": 64,
            "num_weights_buffers": 2,
        },
    },
)
```

Per-op overrides are **orthogonal**: overriding `subblock_k` for `gate_mm` does not change the format, fidelity, or any other setting. The system still auto-derives all non-overridden values from shapes and the profile. This is P4 in action: each override targets exactly one decision without side effects.

#### 1c. Format Hint Override

```python
# Override the format for specific weight tensors
blaze_model = blaze.from_pytorch(
    model, sample_input, device,
    precision="performance",
    format_hints={
        "q_proj.weight": ttnn.bfloat16,   # keep Q projection at bfloat16 (2,048 B/tile)
        "k_proj.weight": ttnn.bfloat16,   # keep K projection at bfloat16
        "gate_proj.weight": ttnn.bfloat4_b, # aggressive compression (576 B/tile)
        # all other weights use bfloat8_b from the performance profile (1,088 B/tile)
    },
)
```

Format hints override the profile's weight format for specific parameters. They do not affect the math fidelity, accumulation precision, or any other setting. The `gate_proj.weight` entry above demonstrates bfloat4\_b usage: at 576 B/tile, the gate projection weight CB is 53% smaller than the bfloat8\_b default, freeing L1 for other CBs. However, bfloat4\_b requires 32x32 tile geometry and carries higher quantization error, so it should only be applied after validating output quality.

#### 1d. Grid Override

```python
# Override the core grid
blaze_model = blaze.from_pytorch(
    model, sample_input, device,
    grid=ttnn.CoreRangeSet([ttnn.CoreRange((0, 0), (7, 3))]),  # 4x8 grid
)
```

#### 1e. format_hint() Context Manager

```python
# Override format within a specific scope
with blaze.format_hint(dtype=ttnn.bfloat4_b):
    # All CBs allocated within this scope use bfloat4_b (576 B/tile)
    gate_out = blaze_gate_proj(activation)
# CBs outside the scope use the profile's format
```

The `format_hint()` context manager uses `contextvars.ContextVar` for proper nesting support (see [Chapter 6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md)).

### Level 2: Manual Bypass

Level 2 bypasses TensorAdapter entirely for specific ops, using the raw `FusedProgram` API:

```python
# Level 2 -- raw FusedProgram for a critical op
adapter = TensorAdapter(device=mesh_device, precision="performance")

f = FusedProgram(
    kernel=custom_attention_kernel,
    device=mesh_device,
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
)

# Manual CB allocation -- developer controls everything
q_cb = f.cb_from_tensor(q_tensor)
k_cb = f.cb_from_tensor(k_tensor, is_direct_address=True)
v_cb = f.cb_from_tensor(v_tensor, is_direct_address=True)

# Manual CT args -- developer computes from shapes
f.ncrisc_ct_args(
    prefix="attn",
    k_num_tiles=seq_len // 32,
    num_heads=32,
    head_dim_tiles=128 // 32,
)

# Manual output wiring
attn_out = CustomAttention.emit(f, q_cb, k_cb, v_cb, prefix="attn")
f.wire_output(attn_out, output_tensor)

result = f.build()
```

Level 2 is the existing Blaze API, unchanged. TensorAdapter does not interfere with manual FusedProgram construction. The developer has full control over every CB, every CT arg, and every reconfig boundary.

---

## Progressive Override: A Worked Profiling Example

This section traces a developer through the complete workflow: start at Level 0, identify a performance gap, apply Level 1 overrides, profile again, and ultimately drop to Level 2 for a specific op.

### Step 0: Level 0 Baseline

```python
# Start with the simplest possible configuration
model = LLaMADecoder(config)
blaze_model = blaze.from_pytorch(
    model,
    sample_input=torch.randn(1, 1, 4096),
    device=mesh_device,
)

# Profile
import time
start = time.perf_counter()
for _ in range(1000):
    output = blaze_model(input_tensor)
elapsed = (time.perf_counter() - start) / 1000
print(f"Level 0 latency: {elapsed * 1e6:.1f} us/token")
```

```text
Level 0 latency: 52.3 us/token
Target: 45.0 us/token
Gap: 16.2%
```

### Step 1: Enable L1 Profiling

```python
import os
os.environ["BLAZE_L1_PROFILE"] = "1"
output = blaze_model(input_tensor)
```

```text
Fusion Report:
  gated_mlp: FUSED (SwigluOp, 11 nodes) -- 12.1 us
  attention: SEQUENTIAL (4 ops) -- 35.2 us  <-- bottleneck
  rmsnorm:   FUSED (2 nodes) -- 5.0 us

CB Stats (attention):
  cb[0] weight Q: 262 KB (bfloat16, balanced profile, 2,048 B/tile)
  cb[1] weight K: 262 KB
  cb[2] weight V: 262 KB
  Total attention weight CBs: 786 KB (60% of ~1.3 MB available L1!)
```

The profiling reveals: (1) attention is sequential (not fused), consuming 67% of total latency; (2) attention weight CBs consume 60% of the ~1.3 MB available L1 per core (out of ~1.5 MB total) because the balanced profile uses bfloat16 for all weights.

### Step 2: Level 1 Override -- Switch to Performance Profile

```python
blaze_model = blaze.from_pytorch(
    model,
    sample_input=torch.randn(1, 1, 4096),
    device=mesh_device,
    precision="performance",  # bfloat8_b weights (1,088 B/tile), LoFi
)
```

```text
Level 1a latency: 44.1 us/token  (improved from 52.3)
Target: 45.0 us/token
Already within target!

But checking quality:
  Perplexity (balanced): 5.12
  Perplexity (performance): 5.19  (+1.4%)
  Acceptable? Depends on application.
```

The performance profile reduces weight CB sizes by 47% (bfloat8\_b at 1,088 B/tile vs. bfloat16 at 2,048 B/tile), freeing L1 and enabling faster streaming. But there is a quality trade-off.

### Step 3: Level 1 Override -- Mixed Precision

```python
# Keep performance for MLP, use balanced for attention Q/K
blaze_model = blaze.from_pytorch(
    model,
    sample_input=torch.randn(1, 1, 4096),
    device=mesh_device,
    precision="performance",
    format_hints={
        "attention.q_proj.weight": ttnn.bfloat16,   # 2,048 B/tile for quality
        "attention.k_proj.weight": ttnn.bfloat16,   # 2,048 B/tile for quality
        # V and MLP weights stay bfloat8_b (1,088 B/tile for performance)
    },
    overrides={
        "sdpa": {
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
        },
    },
)
```

```text
Level 1b latency: 46.8 us/token
Perplexity: 5.14 (+0.4% vs. balanced)
Trade-off: 0.4% quality loss for 10.5% latency improvement
```

### Step 4: Level 2 Bypass -- Custom Attention

The attention block is still sequential. To fuse it, the developer drops to Level 2 for the attention while keeping Level 0/1 for everything else:

```python
# Level 2: Custom fused attention
adapter = TensorAdapter(device=mesh_device, precision="performance")

# Automated: MLP via from_pytorch()
blaze_mlp = blaze.from_pytorch(
    model.mlp, sample_input=torch.randn(1, 4096),
    device=mesh_device, precision="performance",
)

# Automated: RMSNorm via from_pytorch()
blaze_norm = blaze.from_pytorch(
    model.norm, sample_input=torch.randn(1, 4096),
    device=mesh_device, precision="balanced",
)

# Manual: Attention via existing DsaPipeline (Level 2)
attn_config = DsaPipelineConfig(
    hidden_dim=4096, num_heads=32, head_dim=128,
    seq_len=128, math_fidelity="HiFi2",
)
attn_program = build_dsa_pipeline(attn_config, mesh_device)

# Hybrid execution loop
def decode_step(hidden, kv_cache):
    normed = blaze_norm(hidden)
    attn_out = attn_program.run(normed, kv_cache)    # Level 2: manual
    mlp_out = blaze_mlp(normed)                       # Level 0: automatic
    return hidden + attn_out + mlp_out
```

```text
Level 2 latency: 38.5 us/token  (26% improvement over Level 0)
Perplexity: 5.13 (+0.2% vs. balanced)
```

### Summary of the Progressive Override

| Step | Level | Latency (us) | Quality | Change |
|------|-------|-------------|---------|--------|
| 0 | Level 0 (balanced) | 52.3 | 5.12 ppl | Baseline |
| 1 | Level 1a (performance) | 44.1 | 5.19 ppl | Profile switch |
| 2 | Level 1b (mixed) | 46.8 | 5.14 ppl | Format hints for Q/K |
| 3 | Level 2 (hybrid) | 38.5 | 5.13 ppl | Custom attention kernel |

The developer invested approximately 30 minutes of profiling and override work to reduce latency by 26% while maintaining quality within 0.2% of the balanced baseline. At each step, they only changed what the profiling data indicated was suboptimal. This is the P4 principle (Provide Defaults with Per-Decision Overrides) in its most concrete form: the developer escalates from Level 0 to Level 2 incrementally, overriding only the decisions that the profiler identifies as suboptimal.

---

## The Hybrid Model

The hybrid model is the recommended architecture for production deployments. It mixes TensorAdapter-managed ops (Level 0/1) with manually-configured ops (Level 2) within the same inference pipeline.

### Architecture

```text
Input tensor
     |
     v
[BlazeModule: RMSNorm]        <-- Level 0 (automatic)
     |
     +--> [Manual: DsaPipeline]  <-- Level 2 (manual attention)
     |        |
     |        v
     |    attention_output
     |
     +--> [BlazeModule: GatedMLP] <-- Level 0 (automatic)
              |
              v
         mlp_output
              |
              v
     residual_add(input + attn_out + mlp_out)
              |
              v
         Output tensor
```

### Key Design Decisions

**Why not automate everything?** Because some ops (attention with KV cache, custom NOC patterns, MoE routing) require hardware-specific optimizations that cannot be derived from a PyTorch module trace alone. The hybrid model acknowledges this reality: automate what can be automated, allow manual control where automation falls short. As documented in [File 04](./04_worked_example_attention.md), the four current gaps (paged KV cache, GQA, sparse attention, RoPE) all reside in the attention block.

**How do tensors flow between automated and manual ops?** Through device tensors. A `BlazeModule.forward()` call produces a `torch.Tensor` (after unpadding and format conversion). The manual op receives this tensor and converts it to a device tensor. The overhead of this host-device round-trip can be avoided by using `ExternalTensor` (see below).

**How do shapes track across the boundary?** The `ShapeDescriptor` from the automated op's output provides the shape metadata that the manual op needs. For example, the RMSNorm output's `ShapeDescriptor` provides the activation shape that the attention pipeline needs for its CB sizing. This is P2 (Carry Metadata from Both Worlds Simultaneously) enabling seamless handoff: the dual-world metadata (logical shape from PyTorch, padded shape from hardware) travels across the Level 0/Level 2 boundary.

---

## ExternalTensor: The Cross-Framework Bridge

`ExternalTensor` bridges three worlds: Symbiote's TTNNModule (TTNN device tensors), TensorAdapter's BlazeModule (Blaze CBHandles), and raw Blaze ops (FusedProgram). It carries shape metadata alongside the tensor data, enabling seamless handoff between automated and manual paths.

### Three Factory Methods

```python
# PROPOSED (illustrative, not prescriptive) -- ExternalTensor factories

# From PyTorch (entering the system)
ext_act = ExternalTensor.from_torch(
    "activation",
    torch.randn(1, 4096),
    op_hint="matmul.in0",
)
# ext_act.shape = ShapeDescriptor(logical=(1, 4096), padded=(32, 4096), ...)

# From TTNN (from Symbiote's preprocessing)
ext_weight = ExternalTensor.from_ttnn(
    "weights",
    ttnn_weight_tensor,  # already on device, sharded, format-converted
)
# ext_weight.shape = ShapeDescriptor(from_ttnn: page_size, tile_shape, etc.)

# From CBHandle (from a prior manual emit() call)
ext_intermediate = ExternalTensor.from_cb_handle(
    "intermediate",
    manual_output_handle,
)
# ext_intermediate.shape = ShapeDescriptor(best-effort from CBHandle metadata)
```

### ExternalTensor in the Hybrid Model

The critical use case is passing tensors between automated and manual paths without host-device round-trips:

```python
# PROPOSED (illustrative, not prescriptive) -- ExternalTensor in hybrid model

# Phase 1: Automated RMSNorm produces a device tensor
normed_tensor = blaze_norm(hidden)

# Phase 2: Create ExternalTensor for manual attention
ext_normed = ExternalTensor.from_torch(
    "normed_activation", normed_tensor, op_hint="matmul.in0",
)

# Phase 3: Manual attention uses the ExternalTensor
with blaze.fuse() as ctx:
    attn_out = blaze.custom_attention(
        {"activation": ext_normed, "kv_cache": ext_kv_cache},
        grid=attention_grid,
    )

# Phase 4: Automated MLP uses the same activation
mlp_out = blaze_mlp(normed_tensor)
```

The `ExternalTensor` carries the `ShapeDescriptor` from Phase 1 into Phase 2, so the manual attention path does not need to recompute shape metadata. This is the principle of P2 (Carry Metadata from Both Worlds Simultaneously) in action.

### ExternalTensor for KV Cache

The KV cache is the most important use case for `ExternalTensor`. It is a device-resident tensor that persists across inference steps and is updated incrementally:

```python
# PROPOSED (illustrative, not prescriptive) -- KV cache management

# Initialize KV cache at model creation
kv_cache = ExternalTensor(
    "kv_cache",
    tensor=ttnn.zeros([num_layers, 2, max_seq_len, num_heads, head_dim],
                       dtype=ttnn.bfloat16, device=mesh_device),
    shape=ShapeDescriptor(
        logical_shape=(num_layers, 2, 0, num_heads, head_dim),
        padded_shape=(num_layers, 2, max_seq_len, num_heads, head_dim),
        # ... tile and format metadata
    ),
)

# Each decode step: update cache position
def decode_step(token_embedding, step_idx):
    # Manual attention reads from and writes to the KV cache
    attn_out = attn_pipeline.run(
        activation=token_embedding,
        kv_cache=kv_cache.tensor,
        cache_position=step_idx,
    )
    # KV cache is updated in-place on device (no host round-trip)
    kv_cache.shape = kv_cache.shape.with_logical_seq_len(step_idx + 1)
    return attn_out
```

The `ExternalTensor` wrapping the KV cache provides:
1. The backing `ttnn.Tensor` (on device, updated in place)
2. The `ShapeDescriptor` (tracking the logical sequence length)
3. A name for debugging and graph visualization

---

## The Expert Workflow

For developers who regularly work at Level 1 and Level 2, the following workflow maximizes productivity:

### Phase 1: Prototype at Level 0

```python
# Fast iteration -- 3 lines, no hardware knowledge needed
blaze_model = blaze.from_pytorch(model, sample_input, device)
output = blaze_model(input_tensor)
assert torch.allclose(output, reference, atol=1e-4)
```

**Goal:** Verify functional correctness. Performance is secondary.
**Time:** Minutes.

### Phase 2: Profile and Identify Bottlenecks

```python
# Enable profiling
os.environ["BLAZE_L1_PROFILE"] = "1"
output = blaze_model(input_tensor)

# Inspect fusion report
for entry in blaze_model.fusion_report.fusions:
    print(f"Fused: {entry.pattern_name}, CB IDs: {entry.cb_count}, "
          f"L1: {entry.l1_bytes:,} B")

for entry in blaze_model.fusion_report.fallbacks:
    print(f"Sequential: {entry.reason}, ops: {entry.op_names}")
```

**Goal:** Identify which ops are fused, which are sequential, and where L1 is consumed.
**Time:** 10-15 minutes.

### Phase 3: Apply Level 1 Overrides

Based on profiling data, apply targeted overrides:

```python
# Common override patterns:

# Pattern 1: Switch to performance profile for throughput
blaze_model = blaze.from_pytorch(model, sample_input, device,
                                  precision="performance")

# Pattern 2: Keep quality for specific ops
blaze_model = blaze.from_pytorch(model, sample_input, device,
    precision="performance",
    overrides={"sdpa": {"fp32_dest_acc_en": True, "math_fidelity": "HiFi4"}},
    format_hints={"lm_head.weight": ttnn.bfloat16},
)

# Pattern 3: Override blocking for specific matmuls
blaze_model = blaze.from_pytorch(model, sample_input, device,
    precision="performance",
    overrides={
        "gate_mm": {"subblock_k": 64},
        "down_mm": {"subblock_k": 64, "num_weights_buffers": 2},
    },
)
```

**Goal:** Close the performance gap for specific ops while maintaining correctness.
**Time:** 30-60 minutes with profiling iterations.

### Phase 4: Drop to Level 2 for Critical Ops

When Level 1 overrides are insufficient (e.g., the op needs a custom kernel, custom NOC pattern, or dynamic tensor management), the developer bypasses TensorAdapter for that specific op:

```python
# Level 2: Custom op within a hybrid model
adapter = TensorAdapter(device=mesh_device, precision="performance")

# Build a custom FusedProgram for the critical path
f = FusedProgram(kernel=custom_kernel, device=mesh_device)

# Manual CB allocation with shape guidance from TensorAdapter
act_desc = ShapeDescriptor.from_torch(activation, op_hint="custom.in0")
act_cb = f.cb_from_tensor(
    activation_ttnn,
    # Use ShapeDescriptor for guidance, but override specifics
)

# Manual emit with custom parameters
custom_out = CustomOp.emit(
    f, act_cb,
    prefix="custom",
    custom_param_1=value_from_profiling,
    custom_param_2=another_value,
)

f.wire_output(custom_out, output_tensor)
result = f.build()
```

**Goal:** Maximum performance for the critical op, at the cost of manual control.
**Time:** Hours to days (kernel development).

### Phase 5: Production Integration

The final production model uses a mix of levels:

```python
# Production model with mixed override levels
class ProductionLLaMA:
    def __init__(self, config, device):
        # Level 0: Standard ops via TensorAdapter
        self.norm = blaze.from_pytorch(
            model.norm, sample_input, device, precision="balanced")
        self.mlp = blaze.from_pytorch(
            model.mlp, sample_input, device, precision="performance")

        # Level 1: Tuned ops via TensorAdapter with overrides
        self.lm_head = blaze.from_pytorch(
            model.lm_head, sample_input, device,
            precision="performance",
            format_hints={"weight": ttnn.bfloat16},
            overrides={"matmul": {"fp32_dest_acc_en": True}},
        )

        # Level 2: Custom attention via manual Blaze
        self.attention = build_dsa_pipeline(config, device)
        self.kv_cache = ExternalTensor("kv_cache", tensor=...)

    def decode(self, token_embedding):
        normed = self.norm(token_embedding)
        attn_out = self.attention.run(normed, self.kv_cache.tensor)
        mlp_out = self.mlp(normed)
        hidden = token_embedding + attn_out + mlp_out
        return self.lm_head(hidden)
```

---

## Override Interaction Rules

When multiple override mechanisms are active simultaneously, they follow a strict priority order:

```text
Priority (highest to lowest):
  1. Required constraints (softmax safety: HiFi4 + fp32 + exact math)
     -- NEVER overridden by any mechanism
  2. Level 2 bypass (raw FusedProgram)
     -- Developer takes full control, no automation
  3. Per-op overrides (the "overrides" dict)
     -- Surgical settings for specific ops
  4. format_hint() context manager
     -- Scoped format override
  5. Format hints (the "format_hints" dict)
     -- Per-parameter format override
  6. Precision profile
     -- Global defaults for all ops
  7. System defaults (bfloat16, HiFi4, ZERO padding)
```

### Conflict Resolution Example

```python
blaze_model = blaze.from_pytorch(
    model, sample_input, device,
    precision="performance",                # Level 6: LoFi for all matmuls
    format_hints={
        "q_proj.weight": ttnn.bfloat16,     # Level 5: bfloat16 for Q weights
    },
    overrides={
        "sdpa": {
            "math_fidelity": "HiFi4",       # Level 3: HiFi4 for SDPA
            "fp32_dest_acc_en": True,
        },
    },
)
```

Resolution for each op:

| Op | Setting Source | Value | Reason |
|----|--------------|-------|--------|
| `gate_mm.math_fidelity` | Profile (Level 6) | LoFi | No override |
| `gate_mm.weight_format` | Profile (Level 6) | bfloat8\_b (1,088 B/tile) | No override |
| `q_proj.weight_format` | Format hint (Level 5) | bfloat16 (2,048 B/tile) | Overrides profile |
| `q_proj.math_fidelity` | Profile (Level 6) | LoFi | No override for fidelity |
| `sdpa.math_fidelity` | Per-op override (Level 3) | HiFi4 | Overrides profile |
| `sdpa.fp32_dest_acc_en` | Required constraint (Level 1) | True | Would be True anyway |

Note that the `sdpa` override for `math_fidelity="HiFi4"` happens to match the required constraint. If the developer had specified `"LoFi"` instead, the required constraint would override it to `"HiFi4"` silently. A warning is emitted when a developer override is overridden by a required constraint.

---

## The adapt() Escape Hatch

For developers working at the Composition API level (Level 2), the `adapt()` method provides a middle ground: automatic shape analysis with manual CB construction. This is detailed in [Chapter 3](../ch03_tensor_adapter/03_the_adapt_method.md); here we summarize its role in the override architecture.

```python
# PROPOSED (illustrative, not prescriptive) -- adapt() with raw_cb_params

# Level 0 (automatic): provide tensor + op_hint
typed_handle = adapter.adapt(tensor, op_hint="matmul.in0")
# System decides: tile shape, padding, format, page count

# Level 1 (configurable): override specific parameters
typed_handle = adapter.adapt(
    tensor,
    op_hint="matmul.in0",
    tile_shape=(16, 32),     # Override tile geometry
    num_pages=64,            # Override page count
    # System still derives: padding, format, page size
)

# Level 2 (bypass): provide raw CB parameters
typed_handle = adapter.adapt(
    tensor,
    raw_cb_params={
        "num_pages": 64,
        "page_size": 1088,       # bfloat8_b: 1,088 B/tile
        "data_format": ttnn.bfloat8_b,
        "tile": ttnn.Tile([32, 32]),
        "core_ranges": custom_core_range,
    },
    # System derives: nothing (developer controls everything)
)
```

At Level 1, overriding `tile_shape` causes the system to recompute `padded_shape`, `tile_grid_shape`, `page_size`, and `total_tiles` from the new tile geometry. The developer only changes one parameter; all dependent values are recalculated automatically. This is the "orthogonal override" principle in action.

At Level 2, the `raw_cb_params` dict is passed directly to `FusedProgram.cb_scratch()` or `FusedProgram.cb_from_tensor()`. TensorAdapter performs no inference. The `TypedHandle` returned still carries a `ShapeDescriptor` (constructed from the raw params), so downstream ops can use the metadata if needed.

---

## When Each Level Is Appropriate

| Situation | Recommended Level | Rationale |
|-----------|------------------|-----------|
| New model, first deployment | Level 0 | Get it working, measure baseline |
| Standard LLM, throughput tuning | Level 1 (profile) | Profile switch is highest-leverage |
| Specific op quality concern | Level 1 (format\_hints) | Target the specific weight |
| Blocking factor tuning | Level 1 (overrides) | Profile-guided micro-optimization |
| Custom attention kernel | Level 2 | Kernel-level control needed |
| KV cache management | Level 2 + ExternalTensor | Dynamic tensor management |
| Custom NOC pattern | Level 2 | Hardware-specific optimization |
| MoE expert routing | Level 2 | Dynamic control flow |
| Debugging numerical issues | Level 1 (accuracy profile) | fp32 accumulation isolates errors |
| Performance regression investigation | Level 1 (l1\_profile) | Profile to identify the cause |
| Aggressive L1 savings needed | Level 1 (bfloat4\_b format\_hints) | 576 B/tile for maximum compression |

---

## Key Takeaways

- Three override levels provide progressive disclosure: Level 0 (automatic, 3 lines), Level 1 (configurable, targeted overrides), Level 2 (manual bypass, full FusedProgram control). Each level addresses a different developer audience and use case.
- **Overriding one decision does not require overriding all others.** At Level 1, the developer changes specific parameters (`subblock_k`, `weight_format`, `math_fidelity`) while the system auto-derives all dependent values from the override. This orthogonal override principle (P4: Provide Defaults with Per-Decision Overrides) is the core design constraint.
- The **progressive profiling workflow** (Level 0 baseline, profile, Level 1 override, profile again, Level 2 for critical ops) enables developers to close the performance gap iteratively. The worked example demonstrates a 26% latency improvement through progressive override.
- The **hybrid model** mixes TensorAdapter-managed ops (MLP, normalization) with manually-configured ops (attention with KV cache) in the same inference pipeline. This is the recommended architecture for production LLM serving.
- **ExternalTensor** bridges automated and manual paths, carrying `ShapeDescriptor` across the boundary. It is essential for KV cache integration, cross-framework tensor handoff (Symbiote to Blaze), and incremental migration.
- The **override priority order** is: required constraints > Level 2 bypass > per-op overrides > format\_hint() > format\_hints > precision profile > system defaults. Conflicts are resolved silently with warnings when developer overrides are superseded by required constraints.
- The `adapt()` method provides the same three-level override within the Composition API: automatic (`op_hint` only), configurable (specific parameter overrides), and bypass (`raw_cb_params` dict). See [Chapter 3](../ch03_tensor_adapter/03_the_adapt_method.md) for the full treatment.
- All three BFP tile sizes are available as override targets: bfloat16 (2,048 B/tile), bfloat8\_b (1,088 B/tile), and bfloat4\_b (576 B/tile). The choice cascades through CB sizing, L1 budgeting, and streaming bandwidth as described in the profile cascade formula ([File 05](./05_performance_implications_and_tradeoffs.md)).

---

**Next:** [`07_migration_guide.md`](./07_migration_guide.md)
