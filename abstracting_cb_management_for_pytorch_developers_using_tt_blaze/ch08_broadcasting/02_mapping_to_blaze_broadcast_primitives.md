# Mapping PyTorch Broadcasting to TT-Blaze Broadcast Primitives

The previous section established that PyTorch broadcasting creates five patterns (row, column, scalar, none, and multi-dimensional) and that the developer expects all of them to work transparently. This section maps those patterns to the concrete TT-Blaze primitives that implement them: the three logical broadcast modes (`bcast_rows`, `bcast_cols`, `bcast_scalar`) at the compute kernel level, and the physical multicast mechanism (`Mcast`) for cross-core data replication. It develops the detection algorithm that compares ShapeDescriptors to select the correct primitive, provides the complete 27-entry mapping taxonomy covering every broadcast pattern class, addresses compound broadcasts that require decomposition, and traces the byte-level path of a physical Mcast operation from Python through the NOC hardware.

> *Design Principle P2 (Carry Metadata from Both Worlds Simultaneously):* The broadcast mapping algorithm depends on ShapeDescriptor carrying both the logical shape (which dimension is size 1?) and the tile grid shape (how many tiles does that translate to?). Without both, the algorithm cannot distinguish between "this dimension is size 1 because it was always size 1" and "this dimension became size 1 after padding absorbed it."

> *Design Principle P5 (Operate on Graphs, Not Individual Tensors):* Broadcast detection and insertion must happen at the graph level, not per-tensor. A broadcast decision affects CB allocation, CT arg emission, and downstream shape propagation -- all of which are graph-level concerns managed by BlazeGraph and its engine pipeline.

**What you will learn:**

- The three logical broadcast primitives in TT-Blaze's compute kernels: `bcast_rows`, `bcast_cols`, `bcast_scalar`, including the LLK intrinsics they map to
- The physical multicast primitive (`Mcast.emit()` in `blaze/ops/mcast/op.py`) and a byte-level NOC trace of how it transfers data from sender to receivers
- The complete 27-entry mapping taxonomy covering all broadcast pattern categories (no-broadcast, scalar, row, column, mixed, batch-only, and unsupported)
- The broadcast detection algorithm: given two ShapeDescriptors, classify the broadcast pattern and select the primitive
- How compound broadcasts are decomposed into sequences of single-dimension broadcasts
- The distinction between logical broadcast (shape expansion) and physical broadcast (data movement via Mcast/Gather)
- When cross-core Mcast is needed versus when within-kernel tile re-reading suffices
- How `_union_grids()` in `cb_engine.py` handles grid merging for cross-core broadcasts

---

## 1. Logical Broadcast Primitives in the Compute Kernel

At the lowest level, Tenstorrent's compute LLK (Low-Level Kernel) library provides three broadcast modes for binary tile operations. These are not TT-Blaze abstractions -- they are hardware-level instructions that the FPU executes differently depending on the broadcast mode.

### 1.1 bcast_rows: Replicate Across Rows

The `bcast_rows` mode tells the FPU that one operand has a single row (within a tile) that should be applied to every row of the other operand. In tile coordinates, the broadcast source tile's row data is replicated across all 32 rows of each output tile.

```text
bcast_rows: source [1, W_tiles] + target [H_tiles, W_tiles] -> [H_tiles, W_tiles]

Source CB:     Target CB:        Output CB:
+--+--+       +--+--+           +------+------+
|s0|s1|       |t00|t01|         |s0+t00|s1+t01|  row 0
+--+--+       +--+--+           +------+------+
              |t10|t11|         |s0+t10|s1+t11|  row 1 (s0,s1 reused)
              +--+--+           +------+------+
              |t20|t21|         |s0+t20|s1+t21|  row 2 (s0,s1 reused)
              +--+--+           +------+------+

Source CB pages: W_tiles (one row of tiles)
Target CB pages: H_tiles * W_tiles (full grid)
```

```cpp
// EXISTING -- from blaze/ops/rope/kernels/op.hpp (lines 154-157)
// RoPE uses bcast_rows to apply cos/sin across head tiles
mul_bcast_rows_init_short(compute_cta::rotated_in_interm, compute_cta::cos_sin);
for (uint32_t j = 0; j < Wt; j++) {
    mul_tiles_bcast<BroadcastType::ROW>(
        compute_cta::rotated_in_interm, compute_cta::cos_sin, j, Wt + j, j);
}
```

The LLK `mul_tiles_bcast<BroadcastType::ROW>` instruction takes two CB tile indices and an output position. For each of the 32 rows in the output tile, it reads the same row from the broadcast source tile and the corresponding row from the non-broadcast tile, multiplies them, and writes the result.

**When to use:** The broadcast source tensor has `dim[-2] == 1` (a single tile row) and the other tensor has `dim[-2] > 1` (multiple tile rows).

**CB implication:** The broadcast source CB only needs `C_tiles` pages (one tile per column position), while the non-broadcast CB needs `R_tiles * C_tiles` pages for full data.

### 1.2 bcast_cols: Replicate Across Columns

The `bcast_cols` mode tells the FPU that one operand has a single column that should be applied to every column of the other operand.

```text
bcast_cols: source [H_tiles, 1] + target [H_tiles, W_tiles] -> [H_tiles, W_tiles]

Source CB:    Target CB:          Output CB:
+--+         +--+--+--+         +------+------+------+
|s0|         |t00|t01|t02|      |s0+t00|s0+t01|s0+t02| (s0 reused across cols)
+--+         +--+--+--+         +------+------+------+
|s1|         |t10|t11|t12|      |s1+t10|s1+t11|s1+t12| (s1 reused across cols)
+--+         +--+--+--+         +------+------+------+

Source CB pages: H_tiles (one column of tiles)
Target CB pages: H_tiles * W_tiles (full grid)
```

**When to use:** The broadcast source tensor has `dim[-1] == 1` (a single tile column) and the other tensor has `dim[-1] > 1`. Examples include softmax denominator application and gate weight multiplication.

**CB implication:** The broadcast source CB only needs `R_tiles` pages (one tile per row position).

### 1.3 bcast_scalar: Replicate a Single Value

The `bcast_scalar` mode tells the FPU that one operand is a single value that should be applied to every element of the other operand.

```cpp
// EXISTING -- from blaze/ops/eltwise_mul/kernels/op.hpp (lines 139-145)
// EltwiseMul uses bcast_scalar for gate weight multiplication
deepseek_mul_tiles_bcast_scalar_init_short(compute_cta::in0, compute_cta::scalar_local);

tile_regs_acquire();
for (uint32_t i = 0; i < subblock_w; i++) {
    deepseek_mul_tiles_bcast_scalar<compute_cta::fp32_dest_acc_en>(
        compute_cta::in0, compute_cta::scalar_local, base + i, 0, i);
}
```

The scalar value is loaded into a dedicated CB (`scalar_local`, a single-page scratch CB), and the FPU reads that same value for every output element. This is the most memory-efficient broadcast mode -- the broadcast source needs exactly 1 tile page regardless of output size.

```cpp
// EXISTING -- from blaze/ops/rmsnorm/kernels/op.hpp (lines 140-141)
// RMSNorm uses bcast_scalar to multiply the rsqrt(variance) across all tiles
rmsnorm_mul_bcast_scalar_reuse_tiles_init<num_tiles>(compute_cta::input);
rmsnorm_mul_bcast_scalar_reuse_tiles<num_tiles, true>(compute_cta::input, 0, 0, 0);
```

**When to use:** The broadcast source tensor has all dimensions equal to 1 (or is a scalar). Both `dim[-2] == 1` AND `dim[-1] == 1`.

**CB implication:** The broadcast source CB needs exactly 1 page. This is the minimum possible allocation.

### 1.4 Summary of LLK Broadcast Modes

| Mode | BroadcastType Enum | Source Shape Constraint | Source CB Pages | FPU Behavior |
|------|-------------------|------------------------|-----------------|--------------|
| `bcast_rows` | `BroadcastType::ROW` | `dim[-2] == 1` | `C_tiles` | Same row data for all output rows |
| `bcast_cols` | `BroadcastType::COL` | `dim[-1] == 1` | `R_tiles` | Same column data for all output columns |
| `bcast_scalar` | `BroadcastType::SCALAR` | All dims == 1 | 1 | Same value for all output elements |

---

## 2. The Complete Mapping Taxonomy

The following table covers every possible broadcast pattern class for two operands. The classification is based on the number and position of size-1 dimensions relative to the tile dimensions (last two) and batch dimensions (all others). This taxonomy uses the following notation:

- `H` = tile row dimension (second-to-last), `W` = tile column dimension (last)
- `B*` = zero or more batch dimensions, `1` = size 1 (broadcast source), `N` = size > 1

### Category A: No Broadcast (Shapes Match)

| # | Left Shape | Right Shape | Output Shape | Primitive | Notes |
|---|-----------|-------------|-------------|-----------|-------|
| A1 | `[H, W]` | `[H, W]` | `[H, W]` | None | Direct element-wise |
| A2 | `[B, H, W]` | `[B, H, W]` | `[B, H, W]` | None | Batched element-wise |
| A3 | `[B1, B2, H, W]` | `[B1, B2, H, W]` | `[B1, B2, H, W]` | None | Multi-batch element-wise |

### Category B: Scalar Broadcast

| # | Left Shape | Right Shape | Output Shape | Primitive | Source CB Pages |
|---|-----------|-------------|-------------|-----------|----------------|
| B1 | `[1]` or `[]` | `[H, W]` | `[H, W]` | `bcast_scalar` | 1 |
| B2 | `[1, 1]` | `[H, W]` | `[H, W]` | `bcast_scalar` | 1 |
| B3 | `[1]` or `[]` | `[B, H, W]` | `[B, H, W]` | `bcast_scalar` | 1 |
| B4 | `[1, 1, 1]` | `[B, H, W]` | `[B, H, W]` | `bcast_scalar` | 1 |

**Detection rule:** After right-alignment, all dimensions of the broadcast source are 1.

### Category C: Row Broadcast (bcast_rows)

| # | Left Shape | Right Shape | Output Shape | Primitive | Source CB Pages |
|---|-----------|-------------|-------------|-----------|----------------|
| C1 | `[1, W]` | `[H, W]` | `[H, W]` | `bcast_rows` | W_t |
| C2 | `[W]` | `[H, W]` | `[H, W]` | `bcast_rows` | W_t (after dim prepend) |
| C3 | `[1, W]` | `[B, H, W]` | `[B, H, W]` | `bcast_rows` + batch loop | W_t |
| C4 | `[1, 1, W]` | `[B, H, W]` | `[B, H, W]` | `bcast_rows` + batch loop | W_t |
| C5 | `[B, 1, W]` | `[B, H, W]` | `[B, H, W]` | `bcast_rows` per batch | B * W_t |

### Category D: Column Broadcast (bcast_cols)

| # | Left Shape | Right Shape | Output Shape | Primitive | Source CB Pages |
|---|-----------|-------------|-------------|-----------|----------------|
| D1 | `[H, 1]` | `[H, W]` | `[H, W]` | `bcast_cols` | H_t |
| D2 | `[H, 1]` | `[B, H, W]` | `[B, H, W]` | `bcast_cols` + batch loop | H_t |
| D3 | `[B, H, 1]` | `[B, H, W]` | `[B, H, W]` | `bcast_cols` per batch | B * H_t |

### Category E: Mixed Broadcast (Row + Column)

| # | Left Shape | Right Shape | Output Shape | Primitive | Decomposition |
|---|-----------|-------------|-------------|-----------|---------------|
| E1 | `[1, 1]` | `[H, W]` | `[H, W]` | `bcast_scalar` | Single-step (special case) |
| E2 | `[1, W]` | `[H, 1]` | `[H, W]` | **Decompose** | Two-step: expand both, then element-wise |
| E3 | `[H, 1]` | `[1, W]` | `[H, W]` | **Decompose** | Two-step: expand both, then element-wise |

**Pattern E2/E3 decomposition:** Neither operand can serve as the "full" operand in a single broadcast primitive. The BroadcastResolver must materialize one operand to full size via `bcast_cols` or `bcast_rows`, then perform the element-wise operation with a single-primitive broadcast on the other operand. This requires an intermediate scratch CB.

### Category F: Batch-Only Broadcast

| # | Left Shape | Right Shape | Output Shape | Primitive | Notes |
|---|-----------|-------------|-------------|-----------|-------|
| F1 | `[1, H, W]` | `[B, H, W]` | `[B, H, W]` | Mcast or loop | Batch dim broadcast; tile dims match |
| F2 | `[1, 1, H, W]` | `[B1, B2, H, W]` | `[B1, B2, H, W]` | Mcast or loop | Two batch dims broadcast |
| F3 | `[B1, 1, H, W]` | `[B1, B2, H, W]` | `[B1, B2, H, W]` | Mcast or loop | One batch dim broadcast |

**Detection rule:** The last two dimensions match exactly; one or more leading dimensions of the source are 1. These broadcasts affect core-mapping and iteration strategy rather than tile-level compute.

### Category G: Unsupported Broadcasts

| # | Left Shape | Right Shape | Reason |
|---|-----------|-------------|--------|
| G1 | `[3, 4]` | `[2, 4]` | Dim 0: 3 vs 2, neither is 1 |
| G2 | `[3, 5]` | `[3, 4]` | Dim 1: 5 vs 4, neither is 1 |
| G3 | `[3]` | `[4]` | Dim 0: 3 vs 4, neither is 1 |
| G4 | `[2, 3, 4]` | `[5, 3, 4]` | Dim 0: 2 vs 5, neither is 1 |
| G5 | `[1, W, 1]` | `[B, 1, W]` | Inner dims mismatch (requires reshape) |
| G6 | `[H, W]` | `[W, H]` | Transposed, not broadcastable |

**Legend:** `W_t` = ceil(W / tile_w), `H_t` = ceil(H / tile_h). "Left" as broadcast source means the left operand has the size-1 dimensions. The table is symmetric -- if right is the source, swap roles.

---

## 3. The Broadcast Detection Algorithm

Given a binary operation (add, mul, etc.) with left operand `A` and right operand `B`, each carrying a ShapeDescriptor ([Chapter 3](../ch03_tensor_adapter/)), the detection algorithm classifies the broadcast pattern and selects the appropriate primitive.

### 3.1 The Classification Algorithm

```python
# PROPOSED -- Broadcast pattern detection
from enum import Enum

class BroadcastPattern(Enum):
    """Classification of broadcast patterns for TT-Blaze primitives."""
    NONE = "none"              # No broadcast needed
    BCAST_ROWS = "bcast_rows"  # Replicate across rows (dim[-2] == 1)
    BCAST_COLS = "bcast_cols"  # Replicate across columns (dim[-1] == 1)
    BCAST_SCALAR = "bcast_scalar"  # Replicate scalar to all positions
    BATCH = "batch"            # Batch-only broadcast (tile dims match)
    COMPOUND = "compound"      # Multiple dimensions, needs decomposition


def detect_broadcast(
    shape_a: "ShapeDescriptor",
    shape_b: "ShapeDescriptor",
) -> tuple[BroadcastPattern, str]:
    """Detect the broadcast pattern between two ShapeDescriptors.

    Returns:
        (pattern, which_operand) where which_operand is "left", "right", or "none"
        indicating which operand is the broadcast source.

    Implements P3 (Validate at Each Lowering Step): rejects incompatible
    shapes before any CB allocation occurs.
    """
    # Use tile grid shapes for hardware-level comparison
    grid_a = shape_a.tile_grid_shape  # (R_a, C_a)
    grid_b = shape_b.tile_grid_shape  # (R_b, C_b)

    # Step 1: Right-align and pad with 1s (same as PyTorch)
    max_rank = max(len(grid_a), len(grid_b))
    padded_a = (1,) * (max_rank - len(grid_a)) + grid_a
    padded_b = (1,) * (max_rank - len(grid_b)) + grid_b

    # Step 2: Identify broadcast dimensions
    left_bcast_dims = []   # Dims where A has size 1, B has size > 1
    right_bcast_dims = []  # Dims where B has size 1, A has size > 1

    for i, (a, b) in enumerate(zip(padded_a, padded_b)):
        if a == 1 and b > 1:
            left_bcast_dims.append(i)
        elif b == 1 and a > 1:
            right_bcast_dims.append(i)
        elif a != b:
            raise ValueError(
                f"Shapes not broadcast-compatible at tile grid dim {i}: "
                f"{grid_a} vs {grid_b}"
            )

    # Step 3: Classify the pattern
    if not left_bcast_dims and not right_bcast_dims:
        return BroadcastPattern.NONE, "none"

    if left_bcast_dims and right_bcast_dims:
        return BroadcastPattern.COMPOUND, "both"

    # Determine which operand broadcasts
    bcast_dims = left_bcast_dims if left_bcast_dims else right_bcast_dims
    source = "left" if left_bcast_dims else "right"

    # Step 4: Separate tile dims (last 2) from batch dims
    last_dim = max_rank - 1
    second_last_dim = max_rank - 2
    tile_bcast = [d for d in bcast_dims if d >= second_last_dim]
    batch_bcast = [d for d in bcast_dims if d < second_last_dim]

    # Step 5: Scalar optimization -- if source has size 1 in BOTH tile dims,
    # use bcast_scalar regardless of which specific dim triggered the broadcast
    src_shape = padded_a if source == "left" else padded_b
    if max_rank >= 2 and src_shape[second_last_dim] == 1 and src_shape[last_dim] == 1:
        return BroadcastPattern.BCAST_SCALAR, source

    # Step 6: Map tile-dimension broadcasts to primitives
    is_row_bcast = second_last_dim in tile_bcast
    is_col_bcast = last_dim in tile_bcast

    if is_row_bcast and is_col_bcast:
        return BroadcastPattern.BCAST_SCALAR, source
    elif is_row_bcast:
        return BroadcastPattern.BCAST_ROWS, source
    elif is_col_bcast:
        return BroadcastPattern.BCAST_COLS, source
    elif not tile_bcast and batch_bcast:
        return BroadcastPattern.BATCH, source
    else:
        return BroadcastPattern.COMPOUND, source
```

### 3.2 Decision Tree

```text
detect_broadcast(A, B):
  |
  +-- All dims equal? --> NONE
  |
  +-- Any dim incompatible (a != b, a != 1, b != 1)? --> ERROR
  |
  +-- Both operands have size-1 dims on different dims? --> COMPOUND
  |
  +-- Source has all-1 tile dims? --> BCAST_SCALAR
  |
  +-- Which tile dim(s) have size 1?
      |
      +-- dim[-2]=1 AND dim[-1]=1? --> BCAST_SCALAR
      +-- dim[-2]=1 only?          --> BCAST_ROWS
      +-- dim[-1]=1 only?          --> BCAST_COLS
      +-- only batch dims?         --> BATCH (Mcast or loop)
```

### 3.3 Worked Examples

| Operation | Grid A | Grid B | Bcast Dims | Pattern | Source |
|-----------|--------|--------|------------|---------|--------|
| `[128,4096]+[1,4096]` | `(4,128)` | `(1,128)` | dim 0: 4 vs 1 | `BCAST_ROWS` | right |
| `[128,2048]*[128,1]` | `(4,64)` | `(4,1)` | dim 1: 64 vs 1 | `BCAST_COLS` | right |
| `[128,4096]*[1,1]` | `(4,128)` | `(1,1)` | dims 0,1 | `BCAST_SCALAR` | right |
| `[1,7168]*scalar` | `(1,224)` | `(1,1)` | dim 1: 224 vs 1 | `BCAST_SCALAR` | right (scalar opt) |
| `[1,32,128,128]+[1,1,128,128]` | inner `(4,4)` | inner `(4,4)` | dim 1 (batch) | `BATCH` | right |

Note that the scalar optimization (Step 5) catches the case where `[1,7168]*scalar` has source `(1,1)` -- both tile dims are 1, so it becomes `BCAST_SCALAR` with 1 CB page, rather than `BCAST_COLS` with `R_tiles` pages.

---

## 4. Mapping Patterns to Hardware Primitives

### 4.1 Within-Core Broadcast Mapping

When both operands are available on the same core, the broadcast is handled within the compute kernel:

```python
# PROPOSED -- Broadcast primitive selection
def select_broadcast_primitive(
    pattern: BroadcastPattern,
    op_type: str,  # "add", "mul", etc.
) -> dict:
    """Select the compute kernel configuration for a broadcast pattern.

    Returns a dict of CT arg overrides to apply to the op emission.
    """
    if pattern == BroadcastPattern.NONE:
        return {"is_broadcast": 0, "bcast_type": 0}

    if pattern == BroadcastPattern.BCAST_ROWS:
        return {
            "is_broadcast": 1,
            "bcast_type": 0,  # BroadcastType::ROW
            # Source CB: only needs C_tiles pages
            # Compute: uses mul_tiles_bcast<BroadcastType::ROW>
        }

    if pattern == BroadcastPattern.BCAST_COLS:
        return {
            "is_broadcast": 1,
            "bcast_type": 1,  # BroadcastType::COL
            # Source CB: only needs R_tiles pages
            # Compute: uses mul_tiles_bcast<BroadcastType::COL>
        }

    if pattern == BroadcastPattern.BCAST_SCALAR:
        return {
            "is_broadcast": 1,
            "bcast_type": 2,  # BroadcastType::SCALAR
            # Source CB: needs exactly 1 page
            # Compute: uses mul_tiles_bcast_scalar
        }

    if pattern == BroadcastPattern.BATCH:
        return {
            "is_broadcast": 0,  # No tile-level broadcast
            # Batch broadcast handled by iteration + Mcast
        }

    if pattern == BroadcastPattern.COMPOUND:
        raise ValueError(
            "Compound broadcast requires decomposition into "
            "multiple ops. Use decompose_compound_broadcast()."
        )
```

### 4.2 Cross-Core Broadcast: Mcast Insertion

When the broadcast source lives on one core but computation is distributed, an Mcast must be inserted before the compute kernel:

```text
Before (developer intent):
  [add]
  inputs: A (on all cores), B (on core 0)

After (with Mcast insertion):
  [mcast]
  input: B (on core 0)
  output: B' (on all cores)

  [add]
  inputs: A (on all cores), B' (on all cores)
```

### 4.3 Decision Matrix: Logical vs. Physical Broadcast

| Scenario | Logical Broadcast | Physical Broadcast | Example |
|----------|------------------|-------------------|---------|
| Same shape, same cores | None | None | Residual add |
| Size-1 dim, data on all cores | bcast_rows/cols/scalar | None | Gamma in RMSNorm (already distributed) |
| Size-1 dim, data on 1 core | bcast_rows/cols/scalar | Mcast | Bias addition (bias on core 0) |
| Size-1 dim, data on 1 device | bcast_rows/cols/scalar | CCL Broadcast + Mcast | Activation in multi-device decode |
| Multi-dim broadcast | Compound | Mcast + kernel | Attention mask broadcast |

---

## 5. Physical Broadcast: The Mcast Mechanism

Logical broadcast (the three modes above) handles shape expansion within a single core's compute pipeline. Physical broadcast handles data movement: getting data from where it is to where it needs to be across multiple cores.

### 5.1 The Mcast Op Implementation

The `Mcast` class in `blaze/ops/mcast/op.py` is a MicroOp that performs NOC multicast from a sender core to all cores in the grid:

```python
# EXISTING -- from blaze/ops/mcast/op.py (lines 26-125)
@staticmethod
def emit(
    f: FusedProgram,
    src,
    *,
    prefix: str,
    sender_sem: int | None = None,
    dst_num_pages: int | None = None,
    dst_tile_info: TileInfo | None = None,
) -> CBHandle:
    """Multicast src to all cores. Returns handle to dst CB.

    src can be a ttnn.Tensor (allocates a new input CB) or a CBHandle
    (uses the existing CB, e.g. for mcast from a TRISC-filled scratch buffer).
    """
    if isinstance(src, CBHandle):
        src_handle = src
    else:
        src_handle = f.cb_from_tensor(src)

    num_pages = src_handle.num_pages
    tile_size = src_handle.page_size
    data_size_bytes = num_pages * tile_size
```

The destination CB is allocated as a scratch buffer on **all cores** (`f.all_cores`). The `balanced=False` flag is critical for correctness: in a multicast, every receiver core gets the data pushed into its CB, but only the cores that actually consume the data will pop from it. Cores that receive but do not consume have unbalanced push/pop counts, making temporal reuse unsafe ([Chapter 7](../ch07_cb_sizing/), CB compaction).

```python
    # EXISTING -- (continued, lines 63-75)
    dst = f.cb_scratch(
        name=BlazeOp.cb_name(prefix, "dst"),
        num_pages=_dst_num_pages,
        core_ranges=f.all_cores,
        data_format=_dst_data_format,
        tile=_dst_tile_desc,
        page_size=_dst_page_size,
        balanced=False,
    )
```

Per-core CT args differentiate sender and receiver roles:

```python
    # EXISTING -- (continued, lines 82-87)
    f.per_core_unified_ct_args([
        f.flag(f"{prefix}.is_sender", f.sender_grid),
        f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
        f.flag(f"{prefix}.pop_src", f.all_cores),
        f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
    ])
```

BRISC CT args configure the NOC multicast transaction:

```python
    # EXISTING -- (continued, lines 102-123)
    f.brisc_ct_args(
        [
            (f"{prefix}.dest_noc_start_x", f.noc_start.x),
            (f"{prefix}.dest_noc_start_y", f.noc_start.y),
            (f"{prefix}.dest_noc_end_x", f.noc_end.x),
            (f"{prefix}.dest_noc_end_y", f.noc_end.y),
            (f"{prefix}.num_cores", f.num_mcast_cores),
            (f"{prefix}.sender_semaphore", sender_sem),
            (f"{prefix}.receiver_semaphore", receiver_sem),
            (f"{prefix}.data_size_bytes", data_size_bytes),
            (f"{prefix}.src", src_handle),
            (f"{prefix}.src_num_pages", num_pages),
            (f"{prefix}.dst", dst),
            (f"{prefix}.is_part_of_receiver_grid",
             int(f.mcast_range.contains(f.sender))),
            (f"{prefix}.linked", 1),
            (f"{prefix}.posted", 1),
            (f"{prefix}.loopback", 0),
        ]
    )
```

### 5.2 Byte-Level NOC Trace: DeepSeek V3 RMSNorm Activation Mcast

Let us trace a concrete Mcast operation at byte granularity. After the pre-attention RMSNorm completes on the sender core, its output must be multicast to all 7 compute cores for the attention projections.

```text
Scenario: DeepSeek V3 single-token decode
  hidden_dim = 7,168
  Tile geometry: 1x32 (row tiles for single-token)
  num_tiles = 7,168 / 32 = 224  (using 1x32 tiles)
  Data format: bfloat16
  page_size = 1 * 32 * 2 = 64 bytes per tile (1x32)
  Total payload = 224 tiles * 64 bytes = 14,336 bytes

  Core grid: 8 cores total (1 sender + 7 compute)
  Sender: core (0, 0)
  Receivers: cores (1,0) through (7,0)
```

**Phase 1: CB Allocation**

```text
Source CB (sender core only):
  cb_id: assigned by CBEngine (e.g., CB 5)
  num_pages: 224
  page_size: 64 bytes
  core_ranges: {(0,0)}  -- sender only
  L1 footprint on sender: 224 * 64 = 14,336 bytes

Destination CB (ALL cores):
  cb_id: assigned by CBEngine (e.g., CB 6)
  num_pages: 224
  page_size: 64 bytes
  core_ranges: {(0,0), (1,0), ..., (7,0)}  -- all 8 cores
  L1 footprint per core: 14,336 bytes
  balanced: False  (receivers push but may not all pop)
```

**Phase 2: NOC Multicast DMA Transaction**

```text
Time T0: BRISC on sender core issues NOC multicast write command
  +-----------------------------------------------------------+
  | NOC Multicast Write Descriptor:                            |
  |   src_addr:    0x1A000 (sender L1, CB 5 base)             |
  |   dst_addr:    0x1C000 (receiver L1, CB 6 base)           |
  |   mcast_rect:  x=[1..7], y=[0..0]                         |
  |   num_bytes:   14,336                                      |
  |   linked:      1 (chained with semaphore increment)        |
  |   posted:      1 (sender does not wait for completion)     |
  +-----------------------------------------------------------+

Time T1: NOC fabric begins multicast
  The NOC router on the sender core splits the write into 7 copies,
  one for each destination core. The hardware handles this as a
  single multicast transaction -- NOT 7 individual unicast writes.

  +------+    NOC multicast     +------+ +------+ +------+
  |Sender| ------------------> |Core 1| |Core 2| |Core 3| ...
  |(0,0) |    14,336 bytes      |(1,0) | |(2,0) | |(3,0) |
  +------+    to each core      +------+ +------+ +------+

  Each core receives the data at the SAME L1 offset (CB 6 base).
  The NOC hardware guarantees all cores see identical data.

Time T2: NOC multicast completes
  - Each receiver's CB 6 now contains 224 tiles (14,336 bytes)
  - Receiver semaphores on each destination core are incremented
    by the linked NOC transaction
  - cb_wait_front(CB 6, 224) unblocks on all receiver cores

Multicast vs unicast performance:
  Unicast (7 individual writes): ~49 us (7 * ~7 us per hop)
  Multicast (1 command):         ~7-10 us (single traversal + fan-out)
  Speedup: ~5-7x for 7 cores
```

The `linked=1` flag chains a semaphore increment after the data write. The `posted=1` flag means the sender does not wait for acknowledgment, enabling pipelining.

### 5.3 The Three Levels of Physical Broadcast

```text
Level 1: Within-Core Broadcast (no data movement)
  Mechanism: Compute kernel re-reads tiles from the same CB position
  Cost: Zero data movement; compute ALU does the replication
  When: Broadcast source and destination are on the same core
  Example: bcast_scalar in EltwiseMul

Level 2: Cross-Core Broadcast (Mcast)
  Mechanism: NOC multicast from sender core to all cores
  Cost: NOC bandwidth (one write per receiver core)
  When: Data lives on one core but computation is distributed
  Example: Bias broadcast before distributed matmul

Level 3: Cross-Device Broadcast (CCL Broadcast)
  Mechanism: Ethernet fabric unicasts from sender device to mesh
  Cost: Fabric bandwidth (hop-by-hop forwarding)
  When: Data lives on one device but computation spans mesh
  Example: BroadcastRMSNorm in DeepSeek V3 multi-device decode
```

The `BroadcastRMSNormMcast` op (`blaze/ops/broadcast_rmsnorm_mcast/op.py`) chains all three levels:

```python
# EXISTING -- from blaze/ops/broadcast_rmsnorm_mcast/op.py (lines 102-118)
# Level 3: CclBroadcast sends activation from sender device to all devices
rmsnormed = BroadcastRMSNorm.emit(
    f, activation, intermediate, rmsnorm_gamma,
    prefix=prefix,
    sender_coord=sender_coord,
    secondary_cluster_axis=secondary_cluster_axis,
    num_links=num_links, is_torus=is_torus,
    dst_cb=f.cb_output(output, tile=itile_k, page_size=ips_k) if output is not None else None,
)

# Level 2: Mcast normalized result from sender core to all compute cores
if output is not None:
    return None

mcast_dst_pages = K // tile_1x32.tile_shape[1]  # 7168 // 32 = 224
return Mcast.emit(
    f, rmsnormed, prefix=BlazeOp.child_prefix(prefix, "act_mcast"),
    dst_num_pages=mcast_dst_pages, dst_tile_info=ti_1x32,
)
```

Note the `dst_tile_info=ti_1x32`: the Mcast destination uses 1x32 row tiles (matching the downstream matmul's expected input format), even though the RMSNorm output may use a different tile geometry. This tile format conversion during Mcast is a common Blaze pattern.

---

## 6. Cross-Core Broadcast and Grid Management

### 6.1 FusedProgram Grid Context

The `FusedProgram` constructor (lines 424-446 of `fused_program.py`) precomputes grid values that Mcast uses:

```python
# EXISTING -- from blaze/fused_program.py (lines 424-446)
self.sender = ttnn.CoreCoord(*self.grid.sender_core)
self.sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(self.sender, self.sender)])
self.mcast_range = ttnn.CoreRange(
    ttnn.CoreCoord(0, 0),
    ttnn.CoreCoord(self.grid.grid_cols - 1, self.grid.grid_rows - 1),
)
self.all_cores = ttnn.CoreRangeSet([self.mcast_range])
self.num_mcast_cores = self.grid.total_cores

self.mcast_receiver_grid = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(c, r), ttnn.CoreCoord(c, r))
        for r in range(self.grid.grid_rows)
        for c in range(self.grid.grid_cols)
        if (c, r) != self.grid.sender_core
    ]
)
```

### 6.2 Grid Union in CBEngine

The `_union_grids()` function in `cb_engine.py` handles grid merging when broadcast CBs span multiple core sets:

```python
# EXISTING -- from blaze/cb_engine.py (lines 90-119)
def _union_grids(grids: list[Any]) -> Any:
    """Compute the union of grid specifiers."""
    non_none = [g for g in grids if g is not None]
    if not non_none:
        return None
    if len(non_none) == 1:
        return non_none[0]
    if all(g == non_none[0] for g in non_none):
        return non_none[0]
    first = non_none[0]
    if hasattr(first, "__or__"):
        result = first
        for g in non_none[1:]:
            result = result | g
        return result
```

Called by `CBEngine.assign()` (lines 333-337) when computing core ranges for inter-op edges:

```python
# EXISTING -- from blaze/cb_engine.py (lines 333-337)
all_grids = [node.grid] + [e.consumer.grid for e in port_edges]
core_ranges = _union_grids(all_grids)
```

For Mcast edges, the producer grid is the sender core and the consumer grid is all cores, so the union is all cores -- exactly what the Mcast destination CB needs.

---

## 7. Decomposing Compound Broadcasts

When broadcasting occurs along multiple dimensions simultaneously, no single LLK primitive handles it. The system must decompose the compound broadcast.

### 7.1 Batch-Dimension Broadcasts (Common Case)

In many LLM cases, compound broadcasts simplify at the tile level:

```text
Attention mask: [1, 1, 128, 128] * [1, 32, 128, 128]

At the tile level (inner 2D):
  mask tiles: (4, 4)
  scores tiles: (4, 4)
  -> No tile-level broadcast needed!

The "broadcast" is entirely in the batch/head dimensions,
which are handled by iteration over batches/heads.
Each batch/head iteration reads the same mask tiles.
```

This is a critical optimization: **if the broadcast dimensions are all in the batch/head dimensions (outer to the tile grid), the compound broadcast reduces to "re-read the same tile data in the outer loop" and requires no special hardware primitive.**

### 7.2 True Mixed Broadcasts (Rare Case)

When both operands broadcast on different tile dimensions (Category E2/E3), decomposition into two kernel invocations is required:

```text
Mixed broadcast: [1, W] op [H, 1] -> [H, W]

Decomposition:
  Step 1: Materialize [1, W] -> [H, W] via bcast_rows into temp_cb
  Step 2: Perform element-wise: temp_cb op [H, 1] via bcast_cols -> output

Or equivalently:
  Step 1: Materialize [H, 1] -> [H, W] via bcast_cols into temp_cb
  Step 2: Perform element-wise: [1, W] op temp_cb via bcast_rows -> output
```

Mixed broadcasts (Category E) are rare in production LLM workloads. The most common patterns (bias addition, gamma scaling, attention masking, expert scaling) all involve a single broadcast operand.

---

## 8. Integration with the CT Arg System

When a broadcast pattern is detected and a primitive is selected, the decision must be communicated to the compute kernel via compile-time arguments.

### 8.1 Broadcast-Related CT Args

```python
# PROPOSED -- CT arg schema for broadcast-aware binary ops
broadcast_ct_args = [
    CompileTimeArg(
        name="is_broadcast",
        type=Type.BOOL,
        kind=Derived(),
        riscs=Risc.TRISC,
    ),
    CompileTimeArg(
        name="bcast_type",
        type=Type.UINT32,
        kind=Derived(),
        riscs=Risc.TRISC,
    ),
    CompileTimeArg(
        name="bcast_source_pages",
        type=Type.UINT32,
        kind=Derived(),
        riscs=Risc.NCRISC | Risc.TRISC,
    ),
]
```

### 8.2 Example: EltwiseMul with Scalar Broadcast

When `EltwiseMul.emit()` handles a gating operation with a scalar gate weight:

```text
1. Detection: BCAST_SCALAR, source="right" (gate weight is the scalar)
2. CT args emitted:
   enable_scalar = True
   scalar CB = cb_from_tensor(gate_weight)  -> 1 page
3. Compute kernel path (eltwise_mul/kernels/op.hpp, line 41):
   deepseek_mul_tiles_bcast_scalar_init_short(in0, scalar_local)
   deepseek_mul_tiles_bcast_scalar<fp32_dest_acc_en>(in0, scalar_local, ...)
```

The `enable_scalar` CT arg controls whether the kernel executes `deepseek_mul_tiles_bcast_scalar` or the standard element-wise multiply. The `scalar_local` CB (a 1-page scratch buffer, lines 39, 59) holds the single scalar value broadcast to all output tiles.

### 8.3 CT Arg Resolution

The CTArgEngine (from `blaze/ct_args.py`, lines 276-313) resolves broadcast-related args through the `"derived"` source path:

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

Broadcast values (`is_broadcast=1`, `bcast_type=2`) are integers that pass through this path directly.

---

## 9. The End-to-End Broadcast Pipeline

Tracing the complete pipeline for `activation + bias` where `activation: [128, 4096]` and `bias: [1, 4096]`, showing how each chapter's component participates:

```text
STEP 1: PyTorch interception (Ch3 -- TensorAdapter / P7)
  Developer writes: output = activation + bias
  __torch_dispatch__ intercepts the add operation

STEP 2: ShapeDescriptor creation (Ch3 -- ShapeDescriptor / P2)
  act_desc = ShapeDescriptor.from_torch(activation)
    logical_shape = (128, 4096), tile_grid = (4, 128), page_size = 2048 (bf16)
  bias_desc = ShapeDescriptor.from_torch(bias)
    logical_shape = (1, 4096), padded_shape = (32, 4096), tile_grid = (1, 128)

STEP 3: Tile decomposition (Ch4)
  Both descriptors have tile_grid computed.
  act: (4, 128) -- 4 tile rows, 128 tile cols
  bias: (1, 128) -- 1 tile row, 128 tile cols

STEP 4: Padding analysis (Ch5 -- P6)
  Bias padded from [1, 4096] to [32, 4096] with zeros (additive identity).
  Activation already tile-aligned: [128, 4096].

STEP 5: Format negotiation (Ch6)
  Both operands are bfloat16. page_size = 2,048 bytes.

STEP 6: Broadcast detection (Ch8 -- THIS CHAPTER)
  detect_broadcast(act_desc, bias_desc):
    tile grids: (4, 128) vs (1, 128)
    dim 0: 4 vs 1 -> right broadcasts
    dim 1: 128 == 128 -> match
    Pattern: BCAST_ROWS, source="right"

STEP 7: CB allocation (Ch7 -- broadcast-aware sizing)
  bias CB:       128 pages * 2,048 B = 262,144 B (256 KB) -- 1 tile row
  activation CB: 512 pages * 2,048 B = 1,048,576 B (1 MB)
  output CB:     512 pages * 2,048 B = 1,048,576 B (1 MB)
  (With blocking at block_R=1: bias 256 KB + act 256 KB + out 256 KB = 768 KB)

STEP 8: CT arg emission (Ch3 -- CTArgEngine)
  user_args = {
      "is_broadcast": 1, "bcast_type": 0,  # 0 = ROWS
      "num_tiles_per_row": 128, "num_broadcast_rows": 4,
  }

STEP 9: Kernel execution
  for row in range(4):
      for col in range(128):
          act_tile = cb_wait_front(in0_cb)
          bias_tile = read_direct(in1_cb, col)  # re-read each iteration
          result = tile_add(act_tile, bias_tile)
          cb_push_back(out_cb, result)
          cb_pop_front(in0_cb)

STEP 10: Output TypedHandle (Ch3 -- TypedHandle)
  Returns TypedHandle(
    cb_handle = out_cb (512 pages),
    shape = ShapeDescriptor(logical=(128, 4096), tile_grid=(4, 128))
  )
  Downstream matmul sees tile_grid[-1]=128 -> k_num_tiles=128 (correct)
```

> *Design Principle P3 (Validate at Each Lowering Step):* Each step validates before proceeding. Step 2 validates shapes are non-zero. Step 4 validates padding fill is the operation's identity. Step 6 validates broadcast compatibility. Step 7 validates L1 budget. A failure at any step produces a diagnostic error.

---

## 10. When Broadcasting Requires No Extra Op

An important optimization: many broadcasts in LLM workloads do not require any extra operations at all.

### 10.1 Tile Re-Reading

When the broadcast source has fewer tile rows than the output but the data is already in the core's L1, the compute kernel simply reads the same tile multiple times:

```text
Activation: 4 tile rows, 128 tile cols (128 rows x 4096 cols)
Bias:       1 tile row,  128 tile cols (1 row x 4096 cols)

Compute kernel outer loop:
  for row_tile in range(4):
    for col_tile in range(128):
      output[row_tile][col_tile] = activation[row_tile][col_tile] + bias[0][col_tile]
                                                                         ^ always row 0
The bias CB has 128 pages. The kernel reads each page 4 times.
No extra op, no Mcast, no data replication.
```

### 10.2 Source Reference: BroadcastRMSNorm Golden

The golden reference in `broadcast_rmsnorm/op.py` (lines 167-171) shows the pure PyTorch broadcast behavior that the hardware execution must match:

```python
# EXISTING -- from blaze/ops/broadcast_rmsnorm/op.py (lines 167-171)
@staticmethod
def golden(input_tensor: torch.Tensor, gamma_tensor: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """PyTorch reference: RMSNorm(input, gamma)."""
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + epsilon)
    return normalized * gamma_tensor
```

In this golden function, `gamma_tensor` with shape `[D]` is multiplied against `normalized` with shape `[B, S, D]`. PyTorch broadcasts gamma automatically. The TT-Blaze implementation must produce the same result using the hardware primitives described in this section.

---

## Source Files

- `blaze/ops/mcast/op.py` -- `Mcast.emit()` (lines 28-126): complete multicast implementation with BRISC/NCRISC CT args for NOC configuration
- `blaze/ops/gather/op.py` -- `Gather.emit()` (lines 38-180): inverse of Mcast, collects data from scattered cores
- `blaze/ops/broadcast_rmsnorm/op.py` -- `BroadcastRMSNorm.emit()` (lines 58-165): three-level broadcast cascade (CCL + kernel gamma)
- `blaze/ops/broadcast_rmsnorm_mcast/op.py` -- `BroadcastRMSNormMcast.emit()` (lines 59-118): CCL + RMSNorm + Mcast pipeline
- `blaze/ops/ccl_broadcast/op.py` -- `CclBroadcast.emit()` (lines 143-251): cross-device fabric broadcast with routing
- `blaze/cb_engine.py` -- `_union_grids()` (lines 90-119): core range union for fan-out edges
- `blaze/ops/eltwise_mul/op.py` -- `EltwiseMul.emit()` (lines 48-172): element-wise multiply with scalar broadcast
- `blaze/ops/rope/kernels/op.hpp` -- (lines 154-157): `mul_tiles_bcast<BroadcastType::ROW>` for RoPE
- `blaze/ops/rmsnorm/kernels/op.hpp` -- (lines 140-141): `rmsnorm_mul_bcast_scalar_reuse_tiles`
- `blaze/ops/eltwise_mul/kernels/op.hpp` -- (lines 139-145): `deepseek_mul_tiles_bcast_scalar`
- `blaze/ct_args.py` -- CTArgEngine (lines 276-313): compile-time arg resolution for broadcast flags

---

*Navigation:*

- Previous in chapter: [01 -- PyTorch Broadcasting Rules](./01_pytorch_broadcasting_rules.md)
- Next in chapter: [03 -- Broadcast Shape Tracking and Validation](./03_broadcast_shape_tracking_and_validation.md)
- Previous chapter: [Chapter 7 -- CB Sizing and L1 Budget Management](../ch07_cb_sizing/01_l1_memory_model.md)
- Next chapter: [Chapter 9 -- End-to-End Op Fusion Walkthrough](../ch09_fusion_walkthrough/)
