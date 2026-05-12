# Triton and XLA: Tile Programming and HLO Lowering as Design Patterns

**What you will learn:**

- How Triton's tile programming model with explicit masks solves the padding problem declaratively
- How Triton 2.0+ block pointers (`tl.make_block_ptr`) bundle shape, stride, and tile into a single object
- How XLA's HLO lowering pipeline handles tile alignment on TPUs with automatic padding
- The contrast between Triton's "developer controls tiles" and XLA's "compiler controls tiles" approaches
- The 64-CB constraint as a fusion limiter and the CB elision optimization concept
- Four design patterns (DP9-DP12) and a tiered API proposal for TensorAdapter

---

## Triton: Developer-Controlled Tile Programming

### The Triton Model

Triton (OpenAI Triton) occupies a unique position in the tensor-to-hardware landscape. It is a language and compiler for writing GPU kernels at the **tile level** -- the developer thinks in terms of tile-sized blocks of data, and Triton handles the mapping to GPU threads, shared memory, and global memory access patterns.

```python
# EXISTING -- Triton kernel for matmul (simplified)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Tile-level load with mask for boundary handling
    mask_m = offs_m < M
    mask_k = offs_k < K
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    # Accumulate over K dimension in tiles
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        # ... load tiles of A and B with masks, accumulate into acc
        pass

    # Store with boundary mask
    mask_out = mask_m[:, None] & (offs_n[None, :] < N)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)
```

### Design Pattern 9 (DP9): Padding via Masks, Not Data Copies

#### EXISTING (Triton)

Triton's most relevant design pattern for TensorAdapter is **padding via masks rather than data copies**. When a tensor dimension is not a multiple of the block size, the `tl.load()` call takes a `mask` parameter and an `other` value:

```python
# Triton -- mask-based boundary handling
mask = offs < actual_dimension
tile = tl.load(ptr + offs, mask=mask, other=0.0)
# Elements where mask is False get `other` (0.0), no memory is accessed
```

Properties of this approach:

1. **No extra memory**: The tensor stays at its original size. No padded copy is allocated.
2. **Op-specific fill values**: `other=0.0` for matmul (additively neutral), `other=-float('inf')` for softmax (ignored by max), `other=1.0` for multiply (multiplicatively neutral). The fill value is co-located with the load.
3. **Compile-time masks**: When `BLOCK_M` and `M` are both constexpr, the compiler can eliminate the masking entirely for interior tiles.
4. **Per-load granularity**: Different loads in the same kernel can use different masks and fill values.

### Mapping to Tenstorrent Padding

On Tenstorrent hardware, CBs expect tile-aligned data -- the unpacker hardware reads exactly `page_size` bytes per tile. There is no hardware mask mechanism analogous to Triton's `tl.load(mask=...)`. Current Blaze handles non-aligned dimensions by:

- Using separate `PaddedRMSNorm` ops that allocate extra scratch CBs for padded data
- Requiring the developer to compute `padded_bytes`, `valid_rows`, `pad_rows`, and wire them as CT args
- Paying an L1 cost for the padded copy

TensorAdapter cannot adopt Triton's "no-copy mask" approach directly because the hardware does not support it. However, TensorAdapter can adopt the **conceptual model**: padding is a per-load decision (not a global tensor property) with an op-specific fill value:

```python
# PROPOSED -- TensorAdapter padding inspired by Triton (illustrative)
class PaddingPolicy:
    @staticmethod
    def for_op(op_type: str) -> float:
        """Return the correct padding fill value for a given op type."""
        PADDING_VALUES = {
            "matmul": 0.0,       # additively neutral
            "softmax": -1e38,    # ignored by max
            "rmsnorm": 0.0,      # zero contribution to variance
            "multiply": 1.0,     # multiplicatively neutral
            "add": 0.0,          # additively neutral
        }
        return PADDING_VALUES.get(op_type, 0.0)
```

The lesson from Triton: **the fill value is a property of the operation, not the tensor**. A tensor might need zero-padding for matmul and negative-infinity padding for softmax. TensorAdapter should maintain a per-op padding registry rather than a per-tensor padding attribute. The canonical padding value table is in [Chapter 5: Padding](../ch05_padding/index.md).

---

## Design Pattern 10 (DP10): Block Size as a Compile-Time Constant

### EXISTING (Triton)

Triton's block sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) are declared with `tl.constexpr`, making them compile-time constants:

```python
# Triton -- block sizes as constexpr with auto-tuning
@triton.autotune(configs=[
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}),
], key=["M", "N"])
@triton.jit
def matmul_kernel(BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compiler generates different code for BLOCK_M=32 vs BLOCK_M=64
    pass
```

### Mapping to Blaze CT Args

Tenstorrent's compile-time arguments (CT args) serve the same purpose as Triton's constexpr block sizes:

| Triton | Blaze |
|--------|-------|
| `BLOCK_M: tl.constexpr` | `f.trisc_ct_args([("prefix.k_num_tiles", k_num_tiles)])` |
| Auto-tuner searches block sizes | Developer manually computes tile counts |
| Compiler generates per-config variants | Single compilation with developer-chosen values |

TensorAdapter should fill the "auto-tuner" role: given a tensor shape and op type, automatically compute the tile count parameters that Blaze's CT arg system expects.

---

## Triton Block Pointers: Higher-Level Memory Abstraction

Triton 2.0+ introduced `tl.make_block_ptr` that bundles shape, strides, offsets, and tile dimensions into a single object -- analogous to how `CBHandle` bundles `cb_id`, `page_size`, `num_pages`, and `core_ranges`. Loads become `tl.load(block_ptr, boundary_check=(0, 1))` with automatic boundary handling, and iteration is `tl.advance(block_ptr, (0, BLOCK_K))`.

The critical difference: Triton's block pointer is derived from tensor shape and tile size automatically, while Blaze's `CBHandle` requires manual specification of every field. TensorAdapter should close this gap -- deriving CBHandle fields from tensor metadata with the same automation that Triton applies to block pointers.

---

## XLA: Compiler-Controlled Tile Alignment for TPUs

### The XLA/HLO Model

XLA (Accelerated Linear Algebra) takes the opposite approach from Triton: **the compiler controls all tiling and padding decisions**. The developer writes operations on arbitrary-shaped tensors, and XLA's lowering pipeline inserts padding, selects tile sizes, and assigns operations to TPU matrix units automatically.

### Design Pattern 11 (DP11): Automatic Padding at the Compiler Level

#### EXISTING (XLA)

TPUs have matrix units (MXUs) that operate on fixed-size tiles -- typically 128x128 for bf16. When a tensor dimension is not a multiple of 128, XLA automatically:

1. **Pads the tensor** to the next tile-aligned boundary
2. **Selects the padding value** based on the operation
3. **Strips the padding** from the output before returning to the user

```
# XLA HLO -- automatic padding example
Input: f32[7, 37]
    |
    v
XLA pad: f32[7, 37] -> f32[128, 128]   (zero-padded)
    |
    v
XLA matmul: f32[128, 128] x f32[128, 128] -> f32[128, 128]
    |
    v
XLA slice: f32[128, 128] -> f32[7, output_dim]  (padding stripped)
```

XLA's approach is closest to what TensorAdapter should provide for Level 1 developers (zero-configuration):

| XLA Behavior | TensorAdapter Equivalent |
|-------------|--------------------------|
| Auto-pad to TPU tile boundary (128x128) | Auto-pad to Tenstorrent tile boundary (32x32 or variant) |
| Op-specific padding value | `PaddingPolicy.for_op(op_type)` registry |
| Pad fusion with preceding op | Elide separate padding when the preceding CB can be oversized |
| Pad stripping on output | `ShapeDescriptor.logical_shape` tracks the unpadded shape for output slicing |

### BroadcastInDim: Explicit Shape Alignment

XLA's `BroadcastInDim` HLO op makes implicit broadcasting explicit:

```
# EXISTING -- XLA BroadcastInDim semantics
%bias = f32[768] parameter(2)
%broadcast = f32[1024,768] broadcast(%bias), dimensions={1}
# dimensions={1} means: bias dimension 0 maps to output dimension 1
```

This explicitness is valuable for TensorAdapter because CB allocation requires knowing the exact shape at every point. A broadcast input needs fewer CB pages (only 1 tile's worth for a 1D bias) than a fully materialized intermediate.

### Design Pattern 12 (DP12): Shape Tracking Through the Compilation Pipeline

#### EXISTING (XLA)

XLA's HLO maintains shapes at every node in the graph. When padding is inserted, the HLO shape updates to reflect the padded dimensions. A subsequent slice instruction restores the logical shape. At no point is the shape ambiguous.

This is directly relevant to TensorAdapter's `ShapeDescriptor` design. The descriptor should maintain three shapes at every point in the pipeline:

```python
# PROPOSED -- ShapeDescriptor inspired by XLA shape tracking (illustrative)
@dataclass
class ShapeDescriptor:
    logical_shape: tuple[int, ...]    # Original PyTorch shape (7, 37)
    padded_shape: tuple[int, ...]     # After padding to tile boundary (32, 64)
    tile_shape: tuple[int, int]       # Tile dimensions (32, 32)
    # Derived:
    tile_grid: tuple[int, int]        # Number of tiles per dimension (1, 2)
    padding_fill: float               # Op-specific fill value (0.0 for matmul)
```

The triple-shape invariant (logical, padded, tiled) must propagate through op chains: when a matmul output feeds into a softmax, the softmax's input ShapeDescriptor must carry the matmul's padding information so that the softmax knows which positions to mask.

---

## The 64-CB Constraint as a Fusion Limiter

Neither XLA nor Triton has an analog to Tenstorrent's 64-CB slot limit. Consider a gated MLP (common in transformer architectures):

```
  Input -> GateProj -> SiLU -> Mul -> DownProj -> Output
        -> UpProj ----^
```

This requires CB allocations for: input activation, gate weight, gate output, up weight, up output, SiLU output, mul output, down weight, down output, and per-op internal scratch buffers. With double-buffering, this easily reaches 15-20 CB allocations for a single MLP block. When fused with attention (Q/K/V projections, RoPE, softmax, O projection), the total can approach or exceed 64.

XLA bounds fusion by TPU VMEM capacity; Blaze bounds fusion by the 64-CB slot count. TensorAdapter must validate CB counts during planning and automatically trigger multi-phase reconfiguration when they exceed 64.

---

## CB Elision: An Optimization Analog to XLA's Algebraic Simplification

XLA performs algebraic simplifications that reduce intermediate buffer counts (e.g., folding `transpose(transpose(x))` to `x`). An analogous optimization for CB management is **CB elision**: if an intermediate CB connects a producer to a single consumer on the same core range with compatible data formats, the intermediate can potentially be eliminated. Blaze's `cb_alias` mechanism partially implements this:

```python
# EXISTING -- Blaze cb_alias (from FusedProgram)
handle = f.cb_alias(source_handle, new_core_ranges=different_cores)
# Reuses source's cb_id but with different core_ranges
# handle._node_id = source._node_id (inherits graph identity)
```

TensorAdapter should generalize CB elision as an automatic optimization pass. See [Chapter 4: CB Optimization Passes](../ch04_cb_optimization/index.md).

---

## Triton vs. XLA: Two Poles of Control

| Dimension | Triton | XLA | TensorAdapter (Proposed) |
|-----------|--------|-----|--------------------------|
| Who controls tile size | Developer (constexpr) | Compiler | System default, developer override |
| Who handles padding | Developer (masks) | Compiler (auto-pad) | System (PaddingPolicy), developer override |
| Who selects fill value | Developer (per-load `other=`) | Compiler (per-op rules) | System (per-op registry), developer override |
| Abstraction level | Tile-level programming | Tensor-level programming | Tensor-level intent, tile-level execution |
| Escape hatch | Full control (it is the kernel) | None | Progressive: auto -> hints -> manual FusedProgram |

TensorAdapter occupies the middle ground: it provides XLA-like automation (auto-pad, auto-tile, auto-format) with Triton-like transparency (every decision is visible and overridable). This matches the three-level developer journey defined in [Chapter 1: Abstraction Goals](../ch01_the_gap/04_abstraction_goals_and_scope.md).

### The Tiered API Proposal

Combining lessons from both frameworks, TensorAdapter should offer three tiers of control:

| Tier | Style | Developer Experience | Analog |
|------|-------|---------------------|--------|
| **Level 0: Automatic** | XLA-style | Provide tensor + op hint; all CB decisions automatic | `TensorAdapter.adapt(tensor, "matmul.in0")` |
| **Level 1: Configurable** | Triton-style | Override specific parameters (tile shape, num_pages) | `TensorAdapter.adapt(tensor, "matmul.in0", tile_shape=(16, 32))` |
| **Level 2: Manual** | Blaze-style | Full control via raw `FusedProgram.cb_scratch()` | Existing Blaze API, unchanged |

---

## Key Takeaways

- **DP9: Triton's mask-based padding** cannot be replicated on Tenstorrent hardware (no hardware mask), but the conceptual model -- op-specific fill values co-located with the load -- should inform TensorAdapter's `PaddingPolicy` design.
- **DP10: Triton's constexpr block sizes** are the exact analog of Blaze CT args. TensorAdapter automates the computation of these values from shape analysis.
- **DP11: XLA's automatic padding** is the right UX model for Level 0 developers. TensorAdapter should auto-pad to tile boundaries with op-specific fill values, then strip padding from outputs.
- **DP12: XLA's shape tracking** provides the template for `ShapeDescriptor`: maintain logical, padded, and tiled shapes at every point in the pipeline.
- **CB elision** via `cb_alias` is an optimization analog to XLA's algebraic simplification -- TensorAdapter should generalize this as an automatic pass.
- **TensorAdapter sits between Triton (full developer control) and XLA (full compiler control)**, providing automation with escape hatches at every level via a tiered API.

## Source Files

- Triton documentation: `triton-lang.org/main/getting-started/tutorials/`
- Triton source: `triton/language/core.py` -- `tl.load`, `tl.store`, `tl.make_block_ptr`
- Triton source: `triton/runtime/autotuner.py` -- `@autotune` decorator
- XLA HLO documentation: `openxla.org/xla/operation_semantics`
- XLA source: `xla/service/buffer_assignment.cc` -- buffer lifetime analysis
- XLA source: `xla/service/gpu/gpu_fusible.cc` -- fusion heuristics
- Blaze parallel: `blaze/ops/padded_rmsnorm/op.py` -- current manual padding approach
- Blaze parallel: `blaze/ct_args.py` -- CTArgEngine (analogous to Triton's constexpr wiring)
- Blaze parallel: `blaze/fused_program.py` -- `cb_alias` mechanism

---

**Next:** [04 -- MLIR and torch.compile](./04_mlir_and_torch_compile.md)
