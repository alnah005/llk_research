# Broadcast Shape Tracking and Validation

After a broadcast has been detected and the correct primitive selected ([Section 02](./02_mapping_to_blaze_broadcast_primitives.md)), two critical tasks remain. First, the output shape must be tracked correctly -- the ShapeDescriptor and TypedHandle for the broadcast result must reflect the expanded shape, not the original shape of either input. Second, the broadcast decision must be validated against the tile model, CB sizing constraints, and downstream op expectations. This section covers both: how the expanded ShapeDescriptor is constructed, how TypedHandle carries dual shape information (physical CB layout vs. logical broadcast shape), how broadcast-aware CB sizing produces concrete L1 savings (with a detailed feasibility analysis for DeepSeek V3 MoE), and how the validation pipeline rejects broadcasts that cannot be represented.

> *Design Principle P3 (Validate at Each Lowering Step, Not Just at the End):* Broadcasting is a particularly dangerous place for deferred validation. An incorrect broadcast shape that propagates through the graph will produce wrong CB sizes, wrong CT args, and eventually wrong numerical results -- with no indication of where the error originated. Every broadcast decision must be validated at the point of insertion.

> *Design Principle P2 (Carry Metadata from Both Worlds Simultaneously):* After broadcasting, the TypedHandle must carry both the logical broadcast-expanded shape (for downstream shape inference) and the physical CB layout (which may be smaller than the logical shape because the compute kernel handles replication). These two views must coexist without confusion.

**What you will learn:**

- How ShapeDescriptor is updated after broadcast: the `broadcast_with()` factory method
- The dual shape in TypedHandle: `physical_shape` (actual CB layout) vs. `logical_shape` (broadcast-expanded)
- How broadcasting affects CB sizing: why the broadcast source needs fewer pages and the output needs full pages
- The optimization where inner-dimension broadcast requires no extra CB (tile re-read)
- Detailed L1 budget impact analysis: DeepSeek V3 MoE with broadcast-naive (2,706 KB, fails) vs. broadcast-aware (1,224 KB, passes)
- Format interaction: how bfloat8_b (1,088 bytes/tile) and bfloat4_b (576 bytes/tile) affect broadcast source savings
- How TensorAdapter validates that a broadcast shape is representable in the tile model
- Integration with CT args: how the resolver emits `is_broadcast`, `bcast_type`, and related args

---

## 1. ShapeDescriptor After Broadcasting

### 1.1 The broadcast_with() Factory Method

When a binary operation involves broadcasting, the output ShapeDescriptor must reflect the expanded shape -- the element-wise maximum of the two input shapes, following PyTorch's broadcasting rules ([Section 01](./01_pytorch_broadcasting_rules.md)). The frozen nature of ShapeDescriptor ([Chapter 3](../ch03_tensor_adapter/)) means we must create a new instance rather than mutating the existing one.

```python
# PROPOSED -- ShapeDescriptor.broadcast_with() factory method
@dataclass(frozen=True)
class ShapeDescriptor:
    # ... (fields from Chapter 3) ...

    def broadcast_with(self, other: "ShapeDescriptor") -> "ShapeDescriptor":
        """Create a new ShapeDescriptor for the broadcast output.

        Steps:
        1. Compute output logical shape via broadcast_shapes()
        2. Select tile shape (prefer larger operand, or use matching)
        3. Pad last-2 dims to tile boundaries -> padded_shape
        4. Derive tile_grid_shape from padded_shape / tile_shape
        5. Negotiate data_format (Chapter 6 format negotiation)
        6. Compute page_size, total_tiles
        7. Return frozen ShapeDescriptor with source="broadcast"

        Raises ValueError if not broadcast-compatible or if the output
        cannot be represented in the tile model.
        """
        output_logical = broadcast_shapes(self.logical_shape, other.logical_shape)
        output_tile = (self.tile_shape if self.tile_shape == other.tile_shape
                       else (self.tile_shape if self.total_tiles >= other.total_tiles
                             else other.tile_shape))
        tile_h, tile_w = output_tile
        padded = output_logical[:-2] + (
            _ceil_to(output_logical[-2], tile_h),
            _ceil_to(output_logical[-1], tile_w),
        )
        grid = output_logical[:-2] + (padded[-2] // tile_h, padded[-1] // tile_w)
        fmt = _negotiate_broadcast_format(self.data_format, other.data_format)
        total = 1
        for d in grid:
            total *= d
        return ShapeDescriptor(
            logical_shape=output_logical, tile_shape=output_tile,
            padded_shape=padded, tile_grid_shape=grid, total_tiles=total,
            data_format=fmt, page_size=_compute_page_size(fmt, output_tile),
            padding_strategy=self.padding_strategy,
            padding_fill=self.padding_fill,
            source="broadcast",
            op_hint=f"broadcast({self.op_hint}, {other.op_hint})",
        )
```

### 1.2 Worked Example: Bias Addition Shape Tracking

```text
Input shapes:
  activation: ShapeDescriptor(
    logical_shape=(128, 4096),
    padded_shape=(128, 4096),
    tile_grid_shape=(4, 128),
    total_tiles=512,
    data_format="bfloat16",
    page_size=2048
  )
  bias: ShapeDescriptor(
    logical_shape=(1, 4096),
    padded_shape=(32, 4096),   # padded to tile boundary
    tile_grid_shape=(1, 128),
    total_tiles=128,
    data_format="bfloat16",
    page_size=2048
  )

broadcast_shapes((128, 4096), (1, 4096)) = (128, 4096)

Output ShapeDescriptor:
  logical_shape=(128, 4096),
  padded_shape=(128, 4096),
  tile_grid_shape=(4, 128),
  total_tiles=512,
  data_format="bfloat16",
  page_size=2048,
  source="broadcast"
```

The output shape matches the activation shape (the larger operand), which is exactly what PyTorch's broadcasting produces. This is essential: if a downstream matmul reads `tile_grid[-2] = 4` for its RT blocking, it gets the correct value. If the descriptor incorrectly carried the bias's grid `(1, 128)`, the matmul would compute `RT = 1` instead of `RT = 4`.

---

## 2. Dual Shape in TypedHandle

### 2.1 The Problem: Physical vs. Logical Shape

After broadcasting, the output of a binary operation has a logical shape of `[128, 4096]` (the broadcast-expanded shape). But the physical CB layout differs:

1. **The broadcast source CB** holds only the unexpanded data (e.g., `[1, 4096]` = 128 tiles)
2. **The output CB** holds the full expanded result (e.g., `[128, 4096]` = 512 tiles)
3. **During computation**, the compute kernel produces the expanded output by re-reading source tiles

TypedHandle ([Chapter 3](../ch03_tensor_adapter/)) must carry both views:

```python
# PROPOSED -- TypedHandle with broadcast-aware dual shape
@dataclass
class TypedHandle:
    """Composite return type: CBHandle + ShapeDescriptor.

    After broadcasting, carries both:
    - cb_handle: physical CB, with page counts matching actual L1 allocation
    - shape: logical ShapeDescriptor reflecting broadcast-expanded shape

    The physical CB may have fewer pages than shape.total_tiles
    when the broadcast source is not physically replicated. Downstream
    ops must use shape.total_tiles for CT args like num_tiles, not
    cb_handle.num_pages.
    """

    cb_handle: "CBHandle"
    shape: "ShapeDescriptor"

    @property
    def physical_tiles(self) -> int:
        """Actual number of tiles in the CB (may be < logical tiles)."""
        return self.cb_handle.num_pages

    @property
    def logical_tiles(self) -> int:
        """Number of tiles the downstream op should process."""
        return self.shape.total_tiles

    @property
    def is_broadcast_result(self) -> bool:
        """True if this handle resulted from a broadcast operation."""
        return self.shape.source == "broadcast"

    @property
    def broadcast_ratio(self) -> int:
        """How many times each physical tile is logically used.

        1 = no broadcast, >1 = broadcast source tiles are reused.
        """
        if self.physical_tiles == 0:
            return 1
        return max(1, self.logical_tiles // self.physical_tiles)
```

### 2.2 Why Both Are Needed

The dual shape is essential for correctness:

- **CB allocation** uses `physical_tiles` -- the CB must be large enough to hold the actual data in L1, not the logically expanded version
- **CT arg derivation** uses `logical_tiles` -- the compute kernel's `num_tiles` CT arg must reflect how many output tiles to produce
- **Downstream shape propagation** uses the logical ShapeDescriptor -- the next op in the graph sees the broadcast-expanded shape
- **L1 budget validation** uses `physical_tiles * page_size` -- the L1 budget ([Chapter 7](../ch07_cb_sizing/)) must account for actual bytes, not logical bytes

### 2.3 Example: TypedHandle After Scalar Broadcast

```text
Operation: expert_activation * gate_weight
  expert_activation: [1, 2048], 64 tiles, page_size=2048
  gate_weight: scalar (1 tile)

After broadcast detection (BCAST_SCALAR):
  Output shape: [1, 2048]
  Physical output CB: 64 pages (full expanded output, 131,072 bytes)
  Gate weight CB: 1 page (2,048 bytes)

TypedHandle for gate_weight:
  cb_handle.num_pages = 1     (physical: 1 tile in L1)
  shape.total_tiles = 64      (logical: broadcast to 64 tiles)
  broadcast_ratio = 64

TypedHandle for output:
  cb_handle.num_pages = 64    (physical: 64 tiles in L1)
  shape.total_tiles = 64      (logical: 64 tiles)
  broadcast_ratio = 1
```

---

## 3. Broadcasting and CB Sizing

Broadcasting has direct implications for CB sizing ([Chapter 7](../ch07_cb_sizing/)). The broadcast source needs fewer pages, the output needs full pages, and blocking strategies must account for the asymmetry.

### 3.1 Broadcast Source CB Sizing Rules

| Broadcast Pattern | Source CB Pages | Formula |
|-------------------|----------------|---------|
| `bcast_rows` | `C_tiles` | One row of tiles (replicated across rows) |
| `bcast_cols` | `R_tiles` | One column of tiles (replicated across columns) |
| `bcast_scalar` | 1 | A single tile |
| `none` | `R_tiles * C_tiles` | Full tile grid |

```python
# PROPOSED -- Broadcast-aware CB page count computation
def broadcast_source_pages(
    pattern: "BroadcastPattern",
    source_shape: "ShapeDescriptor",
) -> int:
    """Compute the CB page count for the broadcast source operand."""
    R, C = source_shape.tile_grid_shape[-2], source_shape.tile_grid_shape[-1]

    if pattern == BroadcastPattern.BCAST_SCALAR:
        return 1
    elif pattern == BroadcastPattern.BCAST_ROWS:
        return C  # One tile row, C columns
    elif pattern == BroadcastPattern.BCAST_COLS:
        return R  # R rows, one tile column
    elif pattern == BroadcastPattern.NONE:
        return R * C  # Full grid
    else:
        raise ValueError(f"Cannot compute source pages for pattern {pattern}")
```

### 3.2 L1 Impact: Simple Example

```text
Example: Bias addition [128, 4096] + [1, 4096], bfloat16, 32x32 tiles

Without broadcast awareness:
  bias CB: 512 tiles * 2048 bytes = 1,048,576 bytes (1 MB!)
  activation CB: 512 tiles * 2048 bytes = 1,048,576 bytes
  output CB: 512 tiles * 2048 bytes = 1,048,576 bytes
  Total: 3 MB -- EXCEEDS L1 BUDGET!

With broadcast awareness:
  bias CB: 128 tiles * 2048 bytes = 262,144 bytes (256 KB)
  activation CB: 512 tiles * 2048 bytes = 1,048,576 bytes
  output CB: 512 tiles * 2048 bytes = 1,048,576 bytes
  Total: 2.3 MB -- still exceeds L1, but blocking helps

With broadcast awareness + blocking (block_R=1):
  bias CB: 128 tiles * 2048 bytes = 262,144 bytes (full row, reused)
  activation CB: 128 tiles * 2048 bytes = 262,144 bytes (one tile row)
  output CB: 128 tiles * 2048 bytes = 262,144 bytes (one tile row)
  Total: 786,432 bytes (~770 KB) -- fits in L1!
```

The key insight: **the broadcast source CB is not blocked** because all its tiles are needed every iteration. Only the non-broadcast inputs and the output are blocked across the broadcast dimension.

### 3.3 Inner-Dimension Broadcast: No Extra CB

When the broadcast is along the tile's inner dimension (within a single tile), no extra CB is needed at all:

```text
Example: [32, 32] * [1, 32] with a single 32x32 tile

The [1, 32] operand occupies the top row of a single tile.
The compute kernel's bcast_rows mode reads that top row for all 32 rows.
Both operands fit in a single tile page each.

CB layout:
  activation CB: 1 page (2048 bytes)
  bias CB: 1 page (2048 bytes) -- same size as activation!
  output CB: 1 page (2048 bytes)

The broadcast is entirely within the FPU pipeline.
No extra data movement, no extra L1, no extra CB.
```

This optimization applies whenever the broadcast dimension is smaller than the tile dimension. For 32x32 tiles, a row broadcast is free when the target has at most 32 logical rows:

| Source Logical | Target Logical | Tile Shape | Source Padded | Target Padded | Optimization? |
|---------------|---------------|------------|--------------|---------------|--------------|
| [1, D] | [1, D] | (32, 32) | [32, D] | [32, D] | Yes -- shapes match |
| [1, D] | [7, D] | (32, 32) | [32, D] | [32, D] | Yes -- both pad to 1 tile row |
| [1, D] | [31, D] | (32, 32) | [32, D] | [32, D] | Yes -- still 1 tile row |
| [1, D] | [33, D] | (32, 32) | [32, D] | [64, D] | **No** -- target needs 2 tile rows |
| [1, D] | [128, D] | (32, 32) | [32, D] | [128, D] | **No** -- target needs 4 tile rows |

---

## 4. L1 Budget Impact: DeepSeek V3 MoE Feasibility Analysis

This is the critical analysis demonstrating that broadcast-aware CB sizing is not an optimization -- it is a feasibility requirement. Without it, the DeepSeek V3 MoE layer does not fit in L1.

### 4.1 Setup

```text
DeepSeek V3 MoE layer on 7 compute cores:
  hidden_dim = 7,168
  expert_intermediate = 2,048 per core
  8 active routed experts + 1 shared expert
  Activation format: bfloat16 (page_size = 2,048 B per 32x32 tile)
  Weight format: bfloat8_b (page_size = 1,088 B per 32x32 tile)
  Tile shape: (32, 32)
  Per-core L1 available: ~1,291 KB (from Chapter 7 File 01)
```

### 4.2 Broadcast-Naive L1 Budget (Per Core)

```text
CB                                    Pages    Page Size   L1 (KB)
-------------------------------------------------------------------
RMSNorm input                           7      2,048       14
RMSNorm gamma                           7      2,048       14
RMSNorm output                          7      2,048       14
Mcast dest (activation)                 7      2,048       14
Gate matmul act CB                    224      2,048      448
Gate matmul weight CB (streaming)      96      1,088      102
Gate matmul output CB                  64      2,048      128
Up matmul (similar to gate)           ---      ---        228
EltwiseMul output                      64      2,048      128
Gate score CB (naive, per expert x8)  512      2,048    1,024  (**)
Shared expert gate score (naive)      224      2,048      448  (**)
Down matmul weight CB                  96      1,088      102
Down matmul output                      7      2,048       14
Residual add CBs                       14      2,048       28
-------------------------------------------------------------------
Total:                                                  ~2,706 KB

Budget: 1,291 KB
OVER BUDGET BY: 1,415 KB (210% of available)

(**) Broadcast-naive: gate score allocated with full tiles matching
     the expert output shape, not the single-tile scalar source.
```

### 4.3 Broadcast-Aware L1 Budget (Per Core)

```text
CB                                    Pages    Page Size   L1 (KB)
-------------------------------------------------------------------
RMSNorm input                           7      2,048       14
RMSNorm gamma                           7      2,048       14
RMSNorm output                          7      2,048       14
Mcast dest (activation)                 7      2,048       14
Gate matmul act CB                    224      2,048      448
Gate matmul weight CB (streaming)      96      1,088      102
Gate matmul output CB                  64      2,048      128
Up matmul (similar to gate)           ---      ---        228
EltwiseMul output                      64      2,048      128
Gate score CB (aware, scalar x8)        8      2,048       16
Shared expert gate score (aware)        1      2,048        2
Down matmul weight CB                  96      1,088      102
Down matmul output                      7      2,048       14
Residual add CBs                       14      2,048       28
-------------------------------------------------------------------
Total:                                                  ~1,224 KB

Budget: 1,291 KB
WITHIN BUDGET with 67 KB headroom (5.2%)
```

### 4.4 Side-by-Side Comparison

```text
                           Naive      Aware     Savings
                           (KB)       (KB)      (KB)
---------------------------------------------------------
Gate scores (8 experts)   1,024        16      1,008
Shared gate score           448         2        446
All other CBs             1,234     1,234          0
---------------------------------------------------------
Total                     2,706     1,252      1,454
Budget                    1,291     1,291
Status                    FAILS     PASSES
Headroom                 -1,415       +39
```

The broadcast-aware sizing converts an infeasible allocation (210% of budget) into a feasible one (97% of budget) with total savings of 1,454 KB -- nearly the entire L1 capacity. The savings come entirely from two optimizations:

1. **Scalar gate score broadcast:** 1 page instead of 64 pages per expert (8 experts)
2. **Scalar shared expert gate score:** 1 page instead of 224 pages

> *Design Principle P4 (Provide Defaults with Per-Decision Overrides):* The broadcast-aware CB sizing is the default behavior. A developer who knows that a particular broadcast requires full materialization (e.g., for a non-standard kernel) can override by passing `num_pages` explicitly to `cb_scratch()`.

---

## 5. Format Interaction Under Broadcasting

Broadcasting interacts with data format because the broadcast source's page_size depends on its format. The tile size constants for a 32x32 tile are:

| Format | Bytes per Tile (32x32) | 224-Tile Row | Savings vs bf16 |
|--------|----------------------|-------------|-----------------|
| float32 | 4,096 B | 917,504 B | -458,752 B (more) |
| bfloat16 | 2,048 B | 458,752 B | baseline |
| bfloat8_b | 1,088 B | 243,712 B | 215,040 B (47% less) |
| bfloat4_b | 576 B | 129,024 B | 329,728 B (72% less) |

For a row-broadcast source with 224 tiles, choosing bfloat8_b instead of bfloat16 saves ~210 KB -- a meaningful fraction of the L1 budget. The format negotiation system ([Chapter 6](../ch06_data_formats/)) should consider broadcast source status when selecting formats: broadcast sources benefit disproportionately from compressed formats because the per-tile savings are multiplied by the source's tile count.

The precision profiles from Chapter 6 interact with broadcast as follows:

- **Performance profile** (bfloat8_b/LoFi): Broadcast source at 1,088 B/tile. Best L1 savings.
- **Balanced profile** (bfloat16/HiFi4): Broadcast source at 2,048 B/tile. Standard tradeoff.
- **Accuracy profile** (float32 accumulation): Broadcast source remains in its native format; only the accumulator uses float32.

---

## 6. Validation Pipeline

### 6.1 What Must Be Validated

After broadcasting, several invariants must hold. Violating any of them produces silent numerical errors or hardware crashes.

```text
Broadcast Validation Checklist:

[V1] Shape compatibility: input shapes must be PyTorch broadcast-compatible
[V2] Tile representability: broadcast output shape must decompose into valid tiles
[V3] CB sizing: broadcast source + output must fit in L1 budget
[V4] Pattern coverage: detected pattern must map to an available hardware primitive
[V5] Output shape propagation: downstream ops must receive correct expanded shape
[V6] CT arg consistency: broadcast flags must match the kernel's expectations
```

### 6.2 Three Error Categories

**Category 1: Shape Incompatibility (PyTorch-Level)**

Shapes that PyTorch itself would reject. The detection algorithm catches these with original logical shapes in the error message:

```python
# PROPOSED -- shape incompatibility error
class BroadcastShapeError(ValueError):
    def __init__(self, left_shape, right_shape, dim, left_dim, right_dim):
        super().__init__(
            f"Cannot broadcast shapes {left_shape} and {right_shape}: "
            f"dimension {dim} has sizes {left_dim} and {right_dim} "
            f"(neither is 1 and they are not equal).\n"
            f"Hint: reshape one operand so dimension {dim} is either "
            f"1 or matches the other operand."
        )
```

**Category 2: Tile-Level Unrepresentability**

Some broadcasts are valid at the logical level but cannot be efficiently mapped to a single tile-level primitive:

1. **Bidirectional broadcast** (both operands have size-1 dims on different tile dims): requires decomposition into two kernel invocations
2. **Non-tile-aligned broadcast dimension**: padding fill values must correctly represent the broadcast intent (zero for additive, one for multiplicative per P6)
3. **Extremely large broadcast ratio** (target/source > 1024x): suggests physical Mcast may be more efficient than per-tile replication

**Category 3: L1 Budget Violation**

Even with broadcast-aware sizing, the reduced source CB plus the full target CB may exceed the L1 budget. The validation produces broadcast-specific diagnostics:

- The source CB allocation (broadcast-aware pages vs. naive pages, showing savings already applied)
- The target and output CB allocations
- Three actionable suggestions: stream the target operand via blocking, switch the broadcast source to a compressed format (bfloat8_b at 1,088 B/tile), or increase the core count to reduce per-core tile counts

### 6.3 Validation Code Structure

```python
# PROPOSED -- broadcast validation
def validate_broadcast(
    plan: "BroadcastPlan",
    l1_budget_remaining: int,
    source_desc: "ShapeDescriptor",
    target_desc: "ShapeDescriptor",
) -> list[str]:
    """Validate a broadcast plan against all constraints.

    Returns an empty list on success, or a list of error strings.
    Per P3, called immediately after broadcast detection.
    """
    errors = []

    # V2: Tile representability
    output = plan.output_shape
    if len(output) < 2:
        errors.append(f"Broadcast output rank {len(output)} < 2 (requires tile dims)")
    if any(d <= 0 for d in output):
        errors.append(f"Non-positive dimension in broadcast output: {output}")

    # V3: L1 budget
    source_bytes = plan.source_cb_pages * source_desc.page_size
    target_bytes = target_desc.total_tiles * target_desc.page_size
    output_bytes = plan.output_tiles * source_desc.page_size
    total = source_bytes + target_bytes + output_bytes
    if total > l1_budget_remaining:
        errors.append(
            f"Broadcast L1 requirement {total:,} B exceeds budget "
            f"{l1_budget_remaining:,} B. Source CB: {source_bytes:,} B "
            f"({plan.source_cb_pages} pages), target: {target_bytes:,} B, "
            f"output: {output_bytes:,} B. Consider blocking or bfloat8_b."
        )

    # V6: CT arg consistency
    if plan.pattern == BroadcastPattern.BCAST_SCALAR:
        if plan.source_cb_pages != 1:
            errors.append(
                f"Scalar broadcast source must have 1 page, got {plan.source_cb_pages}"
            )

    return errors
```

---

## 7. Integration with CT Arg Resolution

### 7.1 Broadcast CT Arg Flow

```text
Broadcast Detection
    |
    v
BroadcastPattern + source operand
    |
    v
select_broadcast_primitive() -> {is_broadcast, bcast_type, ...}
    |
    v
Op.emit() receives broadcast overrides in kwargs
    |
    v
FusedProgram.unified_ct_args() / trisc_ct_args() records them
    |
    v
CTArgEngine.generate() resolves via "derived" source
    |
    v
Compute kernel reads via get_named_compile_time_arg_val()
```

### 7.2 Per-Core Flags for Mcast Integration

When physical broadcast (Mcast) is involved, per-core CT flags distinguish sender and receiver:

```python
# EXISTING -- from blaze/ops/mcast/op.py (lines 82-87)
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
    f.flag(f"{prefix}.pop_src", f.all_cores),
    f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
])
```

On the sender core, `is_sender=1` and `is_receiver=0`; on all other cores, the inverse. The compute kernel uses these flags to branch between NOC multicast initiation and multicast data consumption.

### 7.3 The Resolver's Role

The CTArgEngine (`blaze/ct_args.py`, lines 276-313) resolves broadcast-related args through the `"derived"` source path. The broadcast detection computes the values; the engine packages them into per-RISC tuples:

```python
# EXISTING -- from blaze/ct_args.py (lines 298-313)
if spec.source_key in user:
    val = user[spec.source_key]
    if isinstance(val, float):
        from .utils import _float_to_bits
        return _float_to_bits(val), True
    return int(val), True
return 0, False
```

---

## 8. Shape Propagation Through Broadcast Op Chains

### 8.1 The Propagation Rule

After broadcasting, the output ShapeDescriptor is the broadcast-expanded shape. All downstream ops see this expanded shape, regardless of the internal broadcast mechanism:

```text
Chain: RMSNorm -> EltwiseMul (with gate scalar) -> Matmul

ShapeDescriptor propagation:
  RMSNorm input: (1, 7168)
  RMSNorm output: (1, 7168)     -- no broadcast in decode
  EltwiseMul input: (1, 7168) * scalar
  EltwiseMul output: (1, 7168)  -- scalar broadcast, output shape unchanged
  Matmul input: (1, 7168)       -- downstream sees (1, 7168), not "scalar"
```

In the prefill case, shapes propagate differently:

```text
Chain: RMSNorm -> BiasAdd (with broadcast) -> Matmul

ShapeDescriptor propagation:
  RMSNorm input: (128, 4096)
  RMSNorm output: (128, 4096)
  BiasAdd inputs: (128, 4096) + (1, 4096)
  BiasAdd output: (128, 4096)   -- broadcast expanded, matches activation
  Matmul input: (128, 4096)     -- downstream sees full shape
```

### 8.2 Multi-Op Pipeline: DeepSeek V3 MoE Expert

```text
DeepSeek V3 MoE Expert Pipeline (single token decode)
======================================================

1. Input: hidden_state [1, 7168]
   ShapeDescriptor: logical=(1, 7168), tile_grid=(1, 224)

2. Expert gate score: gate_scores[expert_idx]
   ShapeDescriptor: logical=(1, 1), tile_grid=(1, 1)

3. Expert up projection: up_proj(hidden_state)
   output: [1, 2048]
   ShapeDescriptor: logical=(1, 2048), tile_grid=(1, 64)

4-6. Gate projection, SiLU, gated multiply: all [1, 2048]
     No broadcast in any of these steps.

7. Down projection: down_proj(gated_output)
   output: [1, 7168]
   ShapeDescriptor: logical=(1, 7168), tile_grid=(1, 224)

8. Gate weight application: down_output * gate_score
   tile_grid: (1, 224) vs (1, 1)
   Pattern: BCAST_SCALAR (scalar optimization: source has all-1 tile dims)
   Gate CB: 1 page (2,048 B). Output CB: 224 pages (458,752 B).
   Output ShapeDescriptor: logical=(1, 7168), tile_grid=(1, 224)

9. Expert output accumulation: sum += weighted_output
   Both [1, 7168] -- no broadcast.
```

The broadcast at step 8 is the only broadcast in the entire expert pipeline during decode. The ShapeTracker propagates `[1, 7168]` to the accumulation step without modification.

---

## 9. Optimization: CB Reuse for Broadcast Sources

### 9.1 Shared Broadcast Source

When the same broadcast source is used by multiple downstream ops (e.g., a gamma vector reused for pre- and post-norm), the CB can be shared:

```text
gamma: [7168]  used by both pre_rmsnorm and post_rmsnorm

Without sharing:
  CB 5: gamma for pre_rmsnorm  (224 pages * 2048 B = 458,752 B)
  CB 9: gamma for post_rmsnorm (224 pages * 2048 B = 458,752 B)
  Total: 917,504 B for gamma alone

With sharing (tensor-backed CB):
  CB 5: gamma (shared by both ops) (224 pages * 2048 B = 458,752 B)
  Total: 458,752 B -- 50% savings
```

The CBEngine supports this through `shared_tensor_cb` (lines 226-262 of `cb_engine.py`):

```python
# EXISTING -- from blaze/cb_engine.py (lines 226-262)
tensor_name = port_to_tensor.get((node.id, port_spec.name))
if tensor_name and tensor_name in shared_tensor_cb:
    cb_id = shared_tensor_cb[tensor_name]
else:
    cb_id = _alloc(dtype, tile_shape)
    if tensor_name:
        shared_tensor_cb[tensor_name] = cb_id
```

### 9.2 Temporal Reuse of Broadcast CBs

The CB compaction system ([Chapter 7](../ch07_cb_sizing/)) can reuse broadcast source CB slots after their last consumer via interval graph coloring (`CBEngine.compact_cb_ids()`, lines 422-437 of `cb_engine.py`). This does not reduce L1 usage but reduces CB slot count, staying under the 64-slot limit.

### 9.3 When Broadcast CBs Cannot Be Reused

Mcast destination CBs marked as `balanced=False` are excluded from temporal reuse:

```python
# EXISTING -- from blaze/fused_program.py (lines 513-517)
# Scratch CBs whose push/pop is NOT balanced per core (e.g. mcast
# dst pushed on every receiver but popped only by the consumer
# subset). Excluded from temporal reuse — fifo state would leak
# onto cores that pushed but never popped.
self._unbalanced_cbs: set[int] = set()
```

---

## 10. End-to-End Walkthrough: LLaMA-7B Attention Mask Broadcast

This walkthrough exercises multi-dimensional broadcast, shape tracking, and CB sizing.

### Step 1: PyTorch Code

```python
# Developer writes:
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # [1, 32, 128, 128]
causal_mask = torch.triu(torch.ones(128, 128), diagonal=1) * -1e9  # [128, 128]
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 128]
masked_scores = scores + causal_mask  # broadcast mask across heads
```

### Step 2: Broadcast Detection

```text
scores:      tile_grid = (1, 32, 4, 4)
causal_mask: tile_grid = (1, 1, 4, 4)

Dimension analysis:
  dim 0: 1 vs 1 -> equal
  dim 1: 32 vs 1 -> mask broadcasts on dim 1
  dim 2: 4 vs 4 -> equal
  dim 3: 4 vs 4 -> equal

Broadcast dim 1 is a batch dim (not last-2), not a tile dim.
Result: BATCH, source="right"
```

### Step 3: Implementation

The mask tiles `(4, 4)` = 16 tiles are loaded once. For each of the 32 heads, the same 16 mask tiles are re-read. No tile-level hardware broadcast is needed.

### Step 4: CB Sizing

```text
Mask CB: 16 tiles * 2048 B = 32 KB (NOT blocked -- all tiles needed per head)
Scores CB (per head): 16 tiles * 2048 B = 32 KB
Output CB (per head): 16 tiles * 2048 B = 32 KB
Total per head iteration: 96 KB -- easily fits in L1
```

### Step 5: Output Shape

```text
Output ShapeDescriptor:
  logical_shape=(1, 32, 128, 128)
  tile_grid_shape=(1, 32, 4, 4)
  total_tiles=512  (32 heads * 16 tiles per head)
  source="broadcast"
```

The downstream softmax receives this `(1, 32, 128, 128)` shape and correctly processes all 32 heads.

---

## 11. The Broadcast Abstraction Stack

The complete broadcast abstraction, from developer code to hardware execution:

```text
Layer 5: PyTorch Developer Code
  a + b  (shapes may not match)

Layer 4: Broadcast Detection (Section 02)
  detect_broadcast(shape_a, shape_b) -> BroadcastPattern

Layer 3: Primitive Selection (Section 02)
  select_broadcast_primitive(pattern) -> CT arg overrides
  needs_physical_broadcast(source_cores, compute_cores) -> bool

Layer 2: Graph Transformation (Section 02)
  insert_broadcasts(graph) -> modified graph with Mcast nodes

Layer 1: CB Sizing + Shape Tracking (this section)
  broadcast_source_pages(pattern, shape) -> page count
  TypedHandle carries physical + logical shape
  ShapeTracker propagates expanded shapes

Layer 0: Hardware Execution
  Compute kernel: mul_tiles_bcast<BroadcastType::ROW/COL/SCALAR>
  Data movement: Mcast.emit() -> NOC multicast
  Cross-device: CclBroadcast.emit() -> Ethernet fabric
```

Each layer hides the complexity below it. The developer sees Layer 5. TensorAdapter and the shape engine operate at Layers 4-3. The graph engines and CB system operate at Layers 2-1. The hardware executes Layer 0. This clean separation follows P1 (Separate Computation Intent from Execution Strategy): the developer writes `a + b`, and the stack handles everything else.

---

## 12. Remaining Challenges

Four areas remain as future work:

1. **Dynamic shapes:** Broadcast detection currently operates on concrete integers at trace time; supporting variable sequence lengths would require conditional broadcast patterns.
2. **Multi-level compound broadcast:** `BroadcastRMSNormMcast` demonstrates chaining CCL Broadcast + Mcast for a specific case, but generalizing to arbitrary compound cross-device/cross-core broadcasts needs additional graph transformation logic.
3. **Broadcast fusion:** Consecutive broadcast operations (e.g., `(x + bias) * scale`) could be fused into a single kernel to avoid materializing intermediates.
4. **Non-rectangular broadcasts:** Sliding window attention masks and other irregular replication patterns may require extending the 2D tile grid broadcast model.

---

## Source Files

- `blaze/ops/mcast/op.py` -- Mcast.emit() (lines 26-126): physical broadcast with `balanced=False` and sender/receiver CB sizing
- `blaze/ops/broadcast_rmsnorm/op.py` -- BroadcastRMSNorm.emit() (lines 58-165): combined physical + logical broadcast, gamma CB sizing
- `blaze/cb_engine.py` -- `_union_grids()` (lines 90-119): core range merging; `CBAssignment` page count fields; `shared_tensor_cb` (lines 226-262); `compact_cb_ids()` (lines 422-437)
- `blaze/ct_args.py` -- CTArgSpec/CTArgEngine (lines 276-313): `is_broadcast` and `bcast_type` resolution
- `blaze/fused_program.py` -- `cb_scratch()` with broadcast-reduced `num_pages`; `_unbalanced_cbs` (lines 513-517)
- `blaze/l1_profile.py` -- Post-hoc verification of broadcast-aware L1 utilization

---

*Navigation:*

- Previous in chapter: [02 -- Mapping to Blaze Broadcast Primitives](./02_mapping_to_blaze_broadcast_primitives.md)
- First in chapter: [01 -- PyTorch Broadcasting Rules](./01_pytorch_broadcasting_rules.md)
- Previous chapter: [Chapter 7 -- CB Sizing and L1 Budget Management](../ch07_cb_sizing/01_l1_memory_model.md)
- Next chapter: [Chapter 9 -- End-to-End Op Fusion Walkthrough](../ch09_fusion_walkthrough/)
