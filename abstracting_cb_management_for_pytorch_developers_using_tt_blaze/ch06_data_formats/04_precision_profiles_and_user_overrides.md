# Precision Profiles and User Overrides: Controlling Format from the Developer's Perspective

The format negotiation protocol (File 02) and the conversion infrastructure (File 03) provide the mechanics of format selection. But the developer still needs a way to express intent: "I want maximum throughput and can tolerate some precision loss" versus "I need bit-accurate accumulation for this specific layer." This section defines three built-in precision profiles that configure the entire format and math-fidelity stack, explains how profiles interact with per-op overrides without conflicting, documents the `format_hint()` escape hatch for forcing a specific format at a graph boundary, and connects these mechanisms to the existing `math_fidelity` and `math_approx_mode` fields on `BlazeOp` and `FusedProgram`. The walkthroughs trace format selection through a DeepSeek V3 MoE layer under each profile and show how a single per-op override can target the softmax within a LLaMA attention block without affecting the surrounding matmuls.

> *Design Principle P4 (Provide Defaults with Per-Decision Overrides):* The developer should be able to set a global precision profile ("I want performance") and override specific ops ("except this softmax needs HiFi4"). Overriding one decision must not require overriding all others.

> *Design Principle P1 (Separate Computation Intent from Execution Strategy):* The developer expresses intent ("I want this model to run fast" or "I need numerical accuracy") through a profile name. TensorAdapter translates that intent into concrete format, fidelity, and accumulation settings for each op in the graph. The developer never writes `ttnn.MathFidelity.LoFi` unless they explicitly want to override the profile.

**What you will learn:**

- Three built-in precision profiles -- "performance", "balanced", "accuracy" -- and the exact format, math_fidelity, and fp32_dest_acc_en settings each configures
- How profile settings map to existing `BlazeOp.math_fidelity`, `BlazeOp.math_approx_mode`, `FusedProgram.math_fidelity`, and `FusedProgram.fp32_dest_acc_en` fields
- The interaction model: profile sets defaults, per-op overrides take precedence, required constraints (e.g., SDPA requires HiFi4) are never violated
- The `format_hint()` context manager: an escape hatch for forcing a specific format at a graph boundary
- How the compiler's `_get_compute_config()` method merges op-level preferences with user-supplied overrides, including the HiFi2 validation gap
- Walkthrough: DeepSeek V3 MoE under each profile, quantifying L1 savings and precision trade-offs
- Walkthrough: LLaMA attention with a per-op override targeting softmax precision
- Dest register capacity under `fp32_dest_acc_en` and the `dst_full_sync_en` coupling

---

## The Three Precision Profiles

### Overview

| Profile | Weight Format | Activation Format | Accumulation | Math Fidelity | Math Approx | Primary Use Case |
|---------|--------------|-------------------|-------------|---------------|-------------|-----------------|
| **performance** | bfloat8_b | bfloat16 | bfloat16 dest | LoFi | True | Maximum throughput, large models |
| **balanced** | bfloat16 | bfloat16 | bfloat16 dest | HiFi4 | False | Default: good accuracy + good speed |
| **accuracy** | bfloat16 | bfloat16 | float32 dest | HiFi4 | False | Debugging, quality-critical layers |

# EXISTING

The existing codebase defines math fidelity per op and per FusedProgram:

```python
# EXISTING -- from blaze/blaze_op.py, line 199
class BlazeOp:
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
```

```python
# EXISTING -- from blaze/fused_program.py, line 399
class FusedProgram:
    def __init__(
        self,
        # ...
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        math_approx_mode: bool = False,
        fp32_dest_acc_en: bool = False,
    ):
```

These fields are set once at construction time and apply uniformly. There is no profile concept -- the developer must manually configure each setting.

# PROPOSED

```python
# PROPOSED -- Precision profile definitions
from dataclasses import dataclass
from enum import Enum

class PrecisionLevel(str, Enum):
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    ACCURACY = "accuracy"

@dataclass(frozen=True)
class PrecisionProfile:
    """Complete precision configuration derived from a single intent."""
    name: str
    weight_format: ttnn.DataType
    activation_format: ttnn.DataType
    accumulation_format: ttnn.DataType  # dest register format
    math_fidelity: ttnn.MathFidelity
    math_approx_mode: bool
    fp32_dest_acc_en: bool
    dst_full_sync_en: bool              # coupled with fp32_dest_acc_en
    softmax_override: bool = True       # always enforce HiFi4 + fp32 for softmax

    @property
    def weight_page_size_32x32(self) -> int:
        """Page size for a 32x32 tile in the weight format."""
        return _TILE_SIZES[self.weight_format]

    @property
    def activation_page_size_32x32(self) -> int:
        """Page size for a 32x32 tile in the activation format."""
        return _TILE_SIZES[self.activation_format]


_TILE_SIZES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}


PROFILES: dict[str, PrecisionProfile] = {
    "performance": PrecisionProfile(
        name="performance",
        weight_format=ttnn.bfloat8_b,
        activation_format=ttnn.bfloat16,
        accumulation_format=ttnn.bfloat16,   # no fp32 accumulation
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    ),
    "balanced": PrecisionProfile(
        name="balanced",
        weight_format=ttnn.bfloat16,
        activation_format=ttnn.bfloat16,
        accumulation_format=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        dst_full_sync_en=False,
    ),
    "accuracy": PrecisionProfile(
        name="accuracy",
        weight_format=ttnn.bfloat16,
        activation_format=ttnn.bfloat16,
        accumulation_format=ttnn.float32,    # fp32 dest accumulation
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=True,               # always coupled with fp32_dest_acc_en
    ),
}
```

---

## Profile Details

### Performance Profile

**Intent:** Maximum throughput. Tolerate precision loss in weight representation and math approximations. Use for large production models where latency matters more than the last fraction of perplexity.

| Setting | Value | Effect |
|---------|-------|--------|
| `weight_format` | `bfloat8_b` | 1,088 bytes/tile -- 1.9x compression vs bfloat16 |
| `activation_format` | `bfloat16` | Standard compute precision |
| `accumulation_format` | `bfloat16` | No fp32 accumulation -- faster FPU throughput |
| `math_fidelity` | `LoFi` | Reduced-precision multiply-accumulate |
| `math_approx_mode` | `True` | Use approximations for exp(), rsqrt(), etc. |
| `fp32_dest_acc_en` | `False` | Dest register stays at bfloat16 width |
| `dst_full_sync_en` | `False` | Standard dest synchronization |

**LoFi math fidelity** uses truncated multipliers that compute fewer mantissa bits per cycle. This increases effective throughput by ~25% for matmul-heavy workloads at the cost of ~1-2 ulp error per multiply-accumulate. For a matmul accumulating 128 K-tiles, the cumulative error is bounded by the number of accumulations times the per-step error -- typically under 1% relative error for well-conditioned matrices.

**math_approx_mode** replaces transcendental functions (exp, log, rsqrt) with polynomial approximations that execute in fewer SFPU cycles. For softmax's `exp()`, the approximation is accurate to ~6 bits of mantissa (vs. 7 for bfloat16 exact). This is why the softmax safety override (below) always forces exact math regardless of profile.

### Balanced Profile

**Intent:** Good accuracy with good performance. The default for most workloads. Uses `bfloat16` for all tensors (weights and activations alike) with full-precision math, matching the PyTorch developer's expectation of uniform precision across the model.

| Setting | Value | Effect |
|---------|-------|--------|
| `weight_format` | `bfloat16` | 2,048 bytes/tile -- full bfloat16 precision, no block-level quantization |
| `activation_format` | `bfloat16` | Standard compute precision |
| `accumulation_format` | `bfloat16` | No fp32 accumulation |
| `math_fidelity` | `HiFi4` | Full-precision multiply-accumulate |
| `math_approx_mode` | `False` | Exact transcendental functions |
| `fp32_dest_acc_en` | `False` | bfloat16 dest register width |
| `dst_full_sync_en` | `False` | Standard dest synchronization |

This profile matches the default settings in the existing codebase (`BlazeOp.math_fidelity = "HiFi4"`, `math_approx_mode = False`) and uses `bfloat16` uniformly. It is the closest analog to PyTorch's default `torch.bfloat16` execution: every tensor has the same per-element precision with no shared-exponent quantization artifacts. For large models, this profile may exceed per-core L1 capacity for weight CBs, requiring either more cores or streaming (see the DeepSeek walkthrough below).

### Accuracy Profile

**Intent:** Maximum numerical fidelity. Use for debugging numerical discrepancies, validating against PyTorch reference, or quality-critical layers (softmax in attention, final classification head).

| Setting | Value | Effect |
|---------|-------|--------|
| `weight_format` | `bfloat16` | Full bfloat16 weights -- no block-level precision loss |
| `activation_format` | `bfloat16` | Standard compute precision |
| `accumulation_format` | `float32` | fp32 dest accumulation -- prevents catastrophic cancellation |
| `math_fidelity` | `HiFi4` | Full-precision multiply-accumulate |
| `math_approx_mode` | `False` | Exact transcendental functions |
| `fp32_dest_acc_en` | `True` | 32-bit dest register -- affects dest capacity |
| `dst_full_sync_en` | `True` | Full-dest sync (coupled with fp32_dest_acc_en) |

> **Warning:** Enabling `fp32_dest_acc_en=True` without `dst_full_sync_en=True` is rarely correct. The standard pattern (and the one profiles enforce) is to set both flags together, yielding `max_dest=8` instead of the punitive `max_dest=4`. The compiler already follows this convention at lines 798-799 of `compiler.py`: `fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=fp32_dest_acc_en`. TensorAdapter profiles preserve this coupling.

---

## Dest Register Capacity Under fp32_dest_acc_en

### The Hardware Constraint

The Tensix core's dest register file has a fixed physical size. When `fp32_dest_acc_en=True`, each tile occupies twice as many register bits, so the number of tiles that fit in dest depends on the sync mode. The `compute_subblock_w()` function in `utils.py` encodes this:

# EXISTING

```python
# EXISTING -- from blaze/utils.py, lines 80-113
def compute_subblock_w(
    num_tiles: int,
    *,
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool | None = None,
) -> int:
    """Return the largest subblock_w that divides num_tiles and fits in dest."""
    if dst_full_sync_en is None:
        dst_full_sync_en = fp32_dest_acc_en
    if dst_full_sync_en and fp32_dest_acc_en:
        max_dest = 8
    elif dst_full_sync_en:
        max_dest = 16
    elif fp32_dest_acc_en:
        max_dest = 4     # <-- punitive: fp32 without sync
    else:
        max_dest = 8     # <-- standard: performance/balanced profile
    subblock_w = min(max_dest, num_tiles)
    while subblock_w > 1 and num_tiles % subblock_w != 0:
        subblock_w -= 1
    return subblock_w
```

### Dest Capacity Table

| Mode | fp32_dest_acc_en | dst_full_sync_en | max_dest tiles | Profile |
|------|-----------------|-----------------|----------------|---------|
| Standard | `False` | `False` | 8 | performance, balanced |
| Full sync only | `False` | `True` | 16 | (unusual) |
| FP32 acc only | `True` | `False` | **4** | (footgun -- avoid) |
| FP32 acc + full sync | `True` | `True` | 8 | accuracy |

When `fp32_dest_acc_en=True` without full sync, `max_dest` drops to 4. For a matmul with 8 output tiles per row, `compute_subblock_w(8, fp32_dest_acc_en=True)` returns 4 instead of 8, doubling inner-loop iterations. The accuracy profile avoids this by coupling `dst_full_sync_en=True`, maintaining `max_dest=8`. The throughput cost under the accuracy profile comes from 32-bit dest register writes, not from reduced subblock size.

---

## The Fidelity Flow Chain: From BlazeOp to Hardware

### How BlazeOp Declares Fidelity

# EXISTING

Every BlazeOp subclass sets its preferred math fidelity as a class attribute. The `register()` method copies these into an `OpSpec`:

```python
# EXISTING -- from blaze/blaze_op.py, line 199
class BlazeOp:
    math_fidelity: str = "HiFi4"        # Default: maximum precision
    math_approx_mode: bool = False       # Default: exact transcendentals
```

```python
# EXISTING -- from blaze/blaze_op.py (register method, lines 283-298)
register_op(
    OpSpec(
        name=cls.name,
        preferred_math_fidelity=cls.math_fidelity,
        math_approx_mode=cls.math_approx_mode,
    )
)
```

### Op-Level Fidelity Choices in the Existing Codebase

The codebase reveals a clear pattern: throughput-dominant ops use `LoFi` + `math_approx_mode=True`, attention ops use `HiFi2`, and numerically sensitive ops use `HiFi4`:

| Op | math_fidelity | math_approx_mode | Rationale |
|----|--------------|-------------------|-----------|
| `DramStreamingMatmul` | `LoFi` | `True` | DRAM-bound matmul; mantissa bits rarely matter |
| `DownProj` | `LoFi` | (default) | Weight projection: same reasoning |
| `DenseSwiGLU` | `LoFi` | `True` | SiLU activation: hardware approx is accurate enough |
| `ClampedSiLU` | `LoFi` | `True` | SiLU with clamping: approx + clamp = sufficient |
| `BroadcastRMSNorm` | `LoFi` | (default) | Normalization: accumulation precision matters more than FPU mantissa |
| `AllReduceMoE` | `LoFi` | `True` | MoE routing: throughput-critical |
| `DsaSparseAttention` | `HiFi2` | (default) | Attention: QK dot product needs more mantissa precision than LoFi |
| `DsaPipeline` | `HiFi2` | (default) | Attention pipeline: same reasoning |
| `GatherReduce` | `HiFi4` | (default) | Reduction: error compounds, needs full precision |
| `BlazeOp` (base) | `HiFi4` | `False` | Safety default: new ops get max precision until proven safe for lower |

### How Fidelity Flows to FusedProgram

The compiler's `_get_compute_config()` method reads the fidelity from the first node's `OpSpec` and constructs a `ComputeConfigDescriptor`:

```python
# EXISTING -- from blaze/compiler.py, lines 777-804
def _get_compute_config(self, graph, merged_overrides=None):
    fidelity_map = {"HiFi4": ttnn.MathFidelity.HiFi4, "LoFi": ttnn.MathFidelity.LoFi}
    fp32_dest_acc_en = False
    if merged_overrides:
        for node in graph.nodes:
            ov = merged_overrides.get(node.id, {})
            if "fp32_dest_acc_en" in ov:
                fp32_dest_acc_en = bool(ov["fp32_dest_acc_en"])
                break
    node = graph.nodes[0]
    spec = node.spec
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity_map.get(
            spec.preferred_math_fidelity, ttnn.MathFidelity.HiFi4
        ),
        math_approx_mode=spec.math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc_en,
        dst_full_sync_en=fp32_dest_acc_en,
    )
    return cfg
```

The complete chain: **BlazeOp.math_fidelity** (class attribute, string) -> **OpSpec.preferred_math_fidelity** (string) -> **_get_compute_config()** -> **ComputeConfigDescriptor.math_fidelity** (ttnn enum) -> **FusedProgram.__init__()** -> hardware configuration.

### The HiFi2 Validation Gap

> **Warning (P3 -- Validate at Each Lowering Step):** The `fidelity_map` in `_get_compute_config()` maps only `"HiFi4"` and `"LoFi"` to ttnn enums. The string `"HiFi2"` -- used by `DsaSparseAttention` and `DsaPipeline` -- falls through to the default `HiFi4`. This means ops that explicitly declare `HiFi2` silently receive `HiFi4`, which is more precise but slower. TensorAdapter's profile system should (a) extend the fidelity_map to include `"HiFi2": ttnn.MathFidelity.HiFi2` and (b) emit a diagnostic warning when a fidelity string falls through to the default.

---

## Softmax Safety Override

### The Problem

Softmax computes `exp(x_i - max(x))` and normalizes. The `exp()` function has a steep gradient, so even small precision errors in the input or accumulation can produce meaningfully wrong attention weights. Under the performance profile, LoFi + math_approx_mode would corrupt these weights.

### The Rule

Regardless of which profile is active, softmax (SDPA) always receives:
- `math_fidelity = HiFi4`
- `fp32_dest_acc_en = True`
- `math_approx_mode = False`

This is a **hard safety constraint**, not a preference. It cannot be relaxed by a profile, a per-op override, or a `format_hint()`. The only way to bypass it is to drop to the raw `FusedProgram` API.

```python
# PROPOSED -- softmax safety enforcement
REQUIRED_CONSTRAINTS = {
    "sdpa":     {"math_fidelity": ttnn.MathFidelity.HiFi4,
                 "math_approx_mode": False, "fp32_dest_acc_en": True},
    "softmax":  {"math_fidelity": ttnn.MathFidelity.HiFi4,
                 "math_approx_mode": False, "fp32_dest_acc_en": True},
    "rmsnorm":  {},  # Gamma must remain bfloat16 (BFP corrupts small values)
}

def apply_profile_to_op(op_name, profile, overrides=None):
    """Compute effective settings: profile -> override -> required constraint."""
    settings = {
        "weight_format": profile.weight_format,
        "activation_format": profile.activation_format,
        "math_fidelity": profile.math_fidelity,
        "math_approx_mode": profile.math_approx_mode,
        "fp32_dest_acc_en": profile.fp32_dest_acc_en,
        "dst_full_sync_en": profile.dst_full_sync_en,
    }
    if overrides:
        settings.update(overrides)
    # Required constraints always win -- fidelity is a minimum, not exact match
    for key, value in REQUIRED_CONSTRAINTS.get(op_name, {}).items():
        if key == "math_fidelity":
            if _fidelity_rank(settings[key]) < _fidelity_rank(value):
                settings[key] = value
        else:
            settings[key] = value
    return settings
```

### Override Hierarchy Summary

```text
Priority (highest to lowest):
  1. Op-level REQUIRED constraints (never overridden) -- e.g., softmax safety
  2. Per-op developer overrides
  3. Precision profile defaults
  4. System defaults (bfloat16, HiFi4)
```

Overrides are surgical: setting `fp32_dest_acc_en=True` on a specific matmul does not change the weight format or math fidelity. This follows P4 -- overriding one decision must not require overriding all others.

---

## The format_hint() Escape Hatch

For cases where the profile system and per-op overrides are not sufficient, `format_hint()` provides a context manager that forces a specific format at a graph boundary:

# PROPOSED

```python
# PROPOSED -- format_hint context manager using contextvars
import contextvars

@dataclass(frozen=True)
class FormatHintState:
    dtype: Optional[ttnn.DataType] = None
    math_fidelity: Optional[str] = None
    fp32_dest_acc_en: Optional[bool] = None

# ContextVar (not threading.local) -- supports both threaded and async contexts
_FORMAT_HINT_VAR: contextvars.ContextVar[FormatHintState] = contextvars.ContextVar(
    "format_hint", default=FormatHintState()
)

@contextlib.contextmanager
def format_hint(dtype=None, math_fidelity=None, fp32_dest_acc_en=None):
    """Force format/fidelity for all CBs allocated within this scope.

    Does NOT override required constraints (SDPA still gets HiFi4 + fp32).
    Uses ContextVar.set()/reset() for proper nesting support.
    """
    token = _FORMAT_HINT_VAR.set(FormatHintState(dtype, math_fidelity, fp32_dest_acc_en))
    try:
        yield
    finally:
        _FORMAT_HINT_VAR.reset(token)
```

### How format_hint() Integrates with Negotiation

When the negotiation algorithm (File 02) selects a format for an edge, it checks for an active hint after checking required constraints but before consulting profile preferences. The priority order within `select_format_for_edge()` is: (1) required constraint from `FormatPreference.required`, (2) active `format_hint()`, (3) profile-adjusted preferred set, (4) system fallback. If the hint forces a format that a downstream consumer cannot accept, negotiation inserts a conversion op at the boundary rather than overriding the hint.

### Relationship to cb_from_view's data_format Override

The existing `cb_from_view()` method in `fused_program.py` (line 903) already accepts a `data_format` keyword argument that overrides `tensor.dtype`. This is the low-level mechanism that `format_hint()` ultimately invokes. When TensorAdapter compiles a graph region inside a `format_hint(dtype=ttnn.bfloat8_b)` block, it passes `data_format=ttnn.bfloat8_b` to `cb_from_view()` for every tensor in that region.

### When to Use format_hint()

| Scenario | format_hint() Appropriate? | Better Alternative |
|----------|--------------------------|-------------------|
| "I want bfloat4_b for this one weight tensor" | Yes | Per-op override if available |
| "I want float32 everywhere for debugging" | No | Use "accuracy" profile |
| "I want different formats for gate vs. up projection" | Yes (two scopes) | Per-op override |
| "I want to bypass format negotiation entirely" | No | Drop to raw FusedProgram API |

---

## math_fidelity and math_approx_mode: The Existing Control Points

### math_fidelity

The `math_fidelity` field controls the precision of the FPU's multiply-accumulate operation:

| Fidelity | Mantissa Bits Computed | Throughput Relative to LoFi | Use Case |
|----------|----------------------|---------------------------|----------|
| `LoFi` | ~4 | 1.0x (baseline) | Maximum throughput |
| `HiFi2` | ~8 | ~0.85x | Good accuracy/speed balance |
| `HiFi3` | ~10 | ~0.75x | Near-full precision |
| `HiFi4` | Full (7 for bf16) | ~0.70x | Full precision (default) |

### math_approx_mode

When `math_approx_mode=True`, transcendental SFPU operations use polynomial approximations:

| Function | Exact Mode | Approx Mode | Error Bound |
|----------|-----------|-------------|-------------|
| `exp(x)` | LUT + polynomial (7-bit) | Degree-3 polynomial (~5-bit) | ~2 ulp for bfloat16 |
| `rsqrt(x)` | Newton-Raphson (2 iterations) | Newton-Raphson (1 iteration) | ~3 ulp |
| `sigmoid(x)` | `1 / (1 + exp(-x))` exact | `exp(-relu(-x))` fast path | ~4 ulp |

For softmax, the `exp()` error directly affects attention weight distribution. With 128 sequence positions, a ~2 ulp error per position accumulates to at most ~0.1% total weight redistribution -- usually acceptable for inference but not for numerical validation. This is why the softmax safety override enforces `math_approx_mode=False`.

### The Pattern: LoFi + Approx = Maximum Throughput

Every op in the existing codebase that sets `math_approx_mode=True` also sets `math_fidelity="LoFi"`. This is not a coincidence -- if precision loss from hardware approximations is acceptable, you also do not need full FPU mantissa precision. The combination `LoFi + math_approx_mode=True` is the "maximum throughput" setting, and it is the implicit default for the performance profile. Conversely, no op sets `math_approx_mode=True` with `math_fidelity="HiFi4"` -- that combination would be contradictory.

### fp32_dest_acc_en

When `fp32_dest_acc_en=True`, the FPU's dest register operates at 32-bit width instead of 16-bit. This prevents catastrophic cancellation during long accumulations: with bfloat16 accumulation, `1000.0 + 0.001 = 1000.0` (0.001 is below the ULP at that magnitude); with float32 accumulation, the result is `1000.001` (23-bit mantissa preserves it). The trade-off is dest register capacity -- see the dest capacity table above.

---

## Walkthrough: DeepSeek V3 MoE Under Each Profile

DeepSeek V3 MoE: 256 experts, each with [7168, 2048] gate/up weights and [2048, 7168] down weights. 14 cores per expert.

### Performance Profile

```text
Weights: bfloat8_b (1,088 bytes/tile)
  Gate weight shard per core: (7168/32) * (2048/32) / 14 = 1024 tiles
  Per-core weight CB: 1024 * 1088 = 1,114,112 bytes (1.06 MB)

Activations: bfloat16 (2048 bytes/tile)
  Activation CB: 2 tiles (double-buffered streaming) = 4,096 bytes

Output: bfloat16 (2048 bytes/tile)
  Output CB: (2048/32)/14 ~ 5 tiles = 10,240 bytes

Total per core: ~1,092 KB out of ~1,500 KB L1 = 73% utilization

Math: LoFi + math_approx
  Throughput: ~1.3x vs balanced (LoFi + approx speedup)
  Precision: ~1-2% relative error on expert outputs
  Effect on model quality: <0.1% perplexity increase (typical for well-trained models)
```

### Balanced Profile

```text
Weights: bfloat16 (2,048 bytes/tile)
  Per-core weight CB: 1024 * 2048 = 2,097,152 bytes (2.0 MB)
  PROBLEM: Exceeds 1.5 MB L1 budget!

  Resolution: Must use streaming (reduce num_pages) or re-shard to more cores
  With 28 cores: per-core weight CB = 512 * 2048 = 1,048,576 bytes (1.0 MB) -- fits
  Total per core (28 cores): ~1,024 KB + 4 KB + 5 KB = ~1,033 KB = 69% utilization

Math: HiFi4, no approx
  Throughput: ~0.85x vs performance (same math, but more cores = more all-gather overhead)
  Precision: ~0.01% relative error on expert outputs (bfloat16 rounding only)
  Effect on model quality: negligible
```

> **Note:** The balanced profile uses the same `bfloat16` weight format as accuracy, so both share the same L1 budget pressure for large weight matrices. The distinction is accumulation precision: balanced uses bfloat16 dest (full dest capacity), while accuracy uses float32 dest (halved capacity). For models with smaller per-expert weights, balanced fits on 14 cores without re-sharding.

### Accuracy Profile

```text
Weights: bfloat16 (2048 bytes/tile)
  Per-core weight CB: 1024 * 2048 = 2,097,152 bytes (2.0 MB)
  PROBLEM: Exceeds 1.5 MB L1 budget!

  Resolution: Must use streaming (reduce num_pages) or re-shard to more cores
  With 28 cores: per-core weight CB = 512 * 2048 = 1,048,576 bytes (1.0 MB) -- fits

Math: HiFi4 + fp32_dest_acc_en + dst_full_sync_en
  max_dest: 8 tiles (fp32 + sync)
  subblock_w: compute_subblock_w(8, fp32_dest_acc_en=True, dst_full_sync_en=True) = 8
  Throughput: ~0.7x vs balanced (fp32 dest writes, not subblock reduction)
  Precision: near-float32 accumulation accuracy
```

> **Warning:** The accuracy profile may not fit in L1 for large models without adjusting the core grid or streaming strategy. TensorAdapter must validate L1 budget after applying the profile and fall back to a less aggressive weight format if the budget is exceeded. This is the profile-to-L1-budget feedback loop described in Chapter 7.

---

## Walkthrough: LLaMA Attention with Per-Op Softmax Override

The developer wants the "performance" profile for the entire model but needs the attention softmax to use exact math for quality:

```python
# PROPOSED -- per-op override usage
blaze_model = blaze.from_pytorch(
    llama_model,
    sample_input=torch.randn(1, 4096),
    device=device,
    precision="performance",  # Global: LoFi, bfloat8_b weights, math_approx
    overrides={
        "sdpa": {
            "math_fidelity": ttnn.MathFidelity.HiFi4,
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
        },
    },
)
```

### Resulting Format Selection

```text
Q/K/V projections (matmul):
  Profile: performance
  No override -> use profile settings
  Weights: bfloat8_b, Math: LoFi, Approx: True
  Activation: bfloat16, Output: bfloat16

QK^T scores (matmul):
  Profile: performance
  No override -> use profile settings
  Both inputs: bfloat16, Math: LoFi, Approx: True

Softmax (SDPA):
  Profile: performance BUT override + required constraint
  Override: HiFi4, exact math, fp32_dest_acc_en
  Required constraint: HiFi4 (from REQUIRED_CONSTRAINTS)
  Resolution: override matches required -> use override
  Input: bfloat16, Accumulation: float32, Output: bfloat16

Attention-value multiply (matmul):
  Profile: performance
  No override -> use profile settings
  Both inputs: bfloat16, Math: LoFi

Output projection (matmul):
  Profile: performance
  No override -> use profile settings
  Weights: bfloat8_b, Math: LoFi, Approx: True
```

### The Math-Fidelity Transition at the Softmax Boundary

The QK^T matmul produces bfloat16 output with LoFi math. The softmax needs bfloat16 input with HiFi4 math and float32 accumulation. The **format** does not change (both bfloat16), but the **math configuration** changes. This is not a format conversion -- it is a math-fidelity transition that happens at the FusedProgram level through a `reconfig()` phase boundary.

In practice, `math_fidelity` is set per FusedProgram, not per op phase. The SDPA kernel handles this internally (it configures HiFi4 for the exp() operation regardless of the program-level setting). For multi-op fused programs, per-phase fidelity requires the TRISC kernel to read the fidelity setting as a CT arg and configure the FPU accordingly.

### The DeepSeek V3 lm_head Pattern

# EXISTING

The DeepSeek V3 B1 model demonstrates real-world per-op override through the `lm_head_fp32_dest_acc_en` flag in `cli.py` line 212. The language model head's softmax is sensitive to accumulation error in the final logit computation. This existing pattern flows through `user_args` into `merged_overrides`, where `_get_compute_config()` applies it to the `ComputeConfigDescriptor`. TensorAdapter's profile + override system generalizes this: instead of a dedicated boolean flag for one specific op, the override dictionary adjusts any compute setting for any op.

---

## Quantization Error Estimation

### Detecting Swamping Before It Causes Problems

When using the performance profile with BFP weight formats (bfloat8_b), the shared-exponent quantization can cause swamping (File 01) in layers with high dynamic range. TensorAdapter provides a debug-mode estimator:

# PROPOSED

```python
# PROPOSED -- quantization error estimator (debug mode)
def estimate_quantization_error(
    tensor: torch.Tensor,
    target_format: str,
    block_size: int = 16,
) -> dict:
    """Estimate quantization error from converting to a BFP format.

    Simulates the shared exponent quantization per 16-element block.
    For each block: computes block_max, checks if max/min ratio exceeds
    2^(mantissa_bits - 1) (the swamping threshold), and measures the
    rounding error from quantizing to mantissa_bits precision.

    Debug mode only -- too expensive for production.

    Returns:
        {"max_error": float, "mean_error": float, "blocks_with_swamping": int}
    """
    mantissa_bits = 7 if target_format == "bfloat8_b" else 3
    SWAMPING_THRESHOLD = 2 ** (mantissa_bits - 1)
    # ... iterate over 16-element blocks, simulate quantization,
    # count blocks where block_max / block_min > SWAMPING_THRESHOLD
```

This estimator connects back to File 01's discussion of the swamping effect: within any 16-element face row that shares an exponent, if the ratio of the largest to smallest non-zero value exceeds `2^(mantissa_bits - 1)`, the smaller values lose all mantissa precision. For `bfloat8_b` (7 mantissa bits), the swamping threshold is 64x; for `bfloat4_b` (3 mantissa bits), it is only 4x. A high `blocks_with_swamping` count suggests the developer should use a higher-precision format for that weight tensor.

---

## Profile Affects Format Negotiation Preferences

The precision profile modifies the FORMAT_PREFERENCES registry (from File 02) by adjusting the "preferred" sets. For weight ports (`in1` on matmul), the profile's `weight_format` becomes the new preferred format; for activation/output ports, `activation_format` becomes preferred. The `required` and `incompatible` sets are never modified by the profile -- they represent hard constraints.

```python
# PROPOSED -- profile adjusts format preferences (abbreviated)
def adjust_preferences_for_profile(base_prefs, profile):
    """For each port, move the profile's format into 'preferred'
    and demote the original preferred set to 'tolerable'.
    required and incompatible sets are preserved unchanged."""
```

---

## API Summary

```python
# PROPOSED -- the three levels of control

# Level 1: Profile (broadest)
blaze_model = blaze.from_pytorch(model, sample_input, device,
                                  precision="performance")

# Level 2: Per-op overrides (targeted)
blaze_model = blaze.from_pytorch(model, sample_input, device,
    precision="performance",
    overrides={
        "sdpa": {"fp32_dest_acc_en": True, "math_fidelity": ttnn.MathFidelity.HiFi4},
        "lm_head": {"fp32_dest_acc_en": True},
    },
)

# Level 3: format_hint() escape hatch (most granular)
with blaze.format_hint(dtype=ttnn.bfloat8_b):
    gate_out = blaze_gate_proj(activation)
```

---

## Interaction with Other Subsystems

### Profiles and Format Negotiation (File 02)

The profile adjusts the "preferred" sets in the FORMAT_PREFERENCES registry. The negotiation algorithm sees different preferred formats under different profiles, producing different edge assignments. Required constraints and incompatible sets are never modified by the profile.

### Profiles and CB Sizing (Chapter 7)

The profile's weight format directly determines page_size, which determines CB memory. The L1 budget validator (Chapter 7) must check feasibility after profile application. If the accuracy profile's bfloat16 weights exceed L1, the validator suggests falling back to "balanced" for that specific weight.

### Profiles and Padding (Chapter 5)

The padding fill value must be representable in the profile's weight format. Under the "performance" profile with bfloat8_b weights, NEG_INF padding uses the format's maximum negative value (~-3.4e38 for bfloat8_b). The padding system validates this for softmax paths (Chapter 5, File 01).

### Profiles and Op Fusion (Chapter 9)

Within a fused op, all micro-ops share the FusedProgram's `math_fidelity`. Per-op overrides require either: (1) phase boundaries (reconfig) that allow different math configs per phase, or (2) kernel-internal fidelity switching (as SDPA already implements).

---

## Key Takeaways

- Three precision profiles -- **performance** (bfloat8_b weights, LoFi, math_approx), **balanced** (bfloat16 weights, HiFi4), **accuracy** (bfloat16 weights, HiFi4, fp32 accumulation + dst_full_sync_en) -- configure the entire format and math stack with a single setting. The developer writes `precision="performance"` instead of manually setting format, fidelity, and accumulation mode for each op.
- The **softmax safety override** ensures SDPA always receives HiFi4 + fp32_dest_acc_en + exact math, regardless of the active profile. This is a hard constraint that no profile, override, or format_hint() can relax.
- The **override hierarchy** is: (1) required constraints (softmax safety, never violated), (2) per-op developer overrides (surgical, one setting at a time), (3) profile defaults, (4) system defaults (bfloat16, HiFi4). Overriding one setting does not require overriding all others (P4).
- `format_hint()` uses `contextvars.ContextVar` (not `threading.local`) to support both threaded and async contexts. It is the most granular override mechanism, scoped to a `with` block, and maps to `cb_from_view()`'s existing `data_format` keyword at the infrastructure level.
- The **fidelity flow chain** -- `BlazeOp.math_fidelity` -> `OpSpec.preferred_math_fidelity` -> `_get_compute_config()` -> `ComputeConfigDescriptor` -> `FusedProgram.__init__()` -- is the existing path that profiles configure. The `fidelity_map` has a validation gap: `"HiFi2"` falls through to `HiFi4` silently (P3 warning).
- `fp32_dest_acc_en` and `dst_full_sync_en` are coupled: the accuracy profile sets both to `True`, maintaining `max_dest=8`. Setting `fp32_dest_acc_en=True` alone drops `max_dest` to 4 -- a footgun that profiles prevent. The `compute_subblock_w()` function in `utils.py` encodes this hardware constraint.
- For DeepSeek V3 MoE, the performance profile enables 0.56 MB per-core weight CBs (40% L1), the balanced profile uses 1.06 MB (72%), and the accuracy profile requires 2.0 MB (exceeds L1, requires re-sharding). The `estimate_quantization_error()` function helps identify layers where BFP compression causes unacceptable swamping.
- Every mechanism a profile needs already exists in TT-Blaze: `math_fidelity` and `math_approx_mode` on `BlazeOp`, `fp32_dest_acc_en` in the compiler's `merged_overrides`, `data_format` override on `cb_from_view()`. Profiles are a naming and bundling layer (P1), not new infrastructure.

## Source Files

- `blaze/blaze_op.py` -- `BlazeOp.math_fidelity` (line 199, per-op default: `"HiFi4"`), `BlazeOp.math_approx_mode` (line 200, per-op default: `False`), `register()` method propagating these to `OpSpec` (lines 283-298)
- `blaze/graph.py` -- `OpSpec.preferred_math_fidelity` (line 55), `OpSpec.math_approx_mode` (line 56)
- `blaze/compiler.py` -- `_get_compute_config()` merging op preferences with user overrides (lines 777-804), `fidelity_map` with HiFi2 validation gap (line 785), `dst_full_sync_en=fp32_dest_acc_en` coupling (line 799), `FusedProgram` construction (lines 939-942)
- `blaze/fused_program.py` -- `FusedProgram.__init__(math_fidelity, math_approx_mode, fp32_dest_acc_en)` (lines 394-413), `cb_from_view()` accepting `data_format` override (line 903)
- `blaze/utils.py` -- `compute_subblock_w(fp32_dest_acc_en, dst_full_sync_en)` dest register capacity calculation (lines 80-113)
- `blaze/models/deepseek_v3_b1/cli.py` -- `lm_head_fp32_dest_acc_en: bool = True` as real-world per-op override example (line 212)
- `blaze/ops/clamped_silu/op.py` -- Example of `LoFi + math_approx_mode=True` pattern (lines 27-28)
- `blaze/ops/dsa_attention/op.py` -- Example of `HiFi2` fidelity declaration (line 31)
- `blaze/ops/gather_reduce/op.py` -- Example of explicit `HiFi4` for numerically sensitive reduction (line 27)

---

← Previous: [Chapter 5 — Padding](../ch05_padding/) | Next: [Chapter 7 — Automatic CB Sizing and L1 Memory Budgeting](../ch07_cb_sizing/) →

< Previous: [File 03 -- Format Conversion and CB Reconfig](./03_format_conversion_and_cb_reconfig.md) | Next: [Chapter 7 -- Automatic CB Sizing and L1 Memory Budgeting](../ch07_cb_sizing/) >
