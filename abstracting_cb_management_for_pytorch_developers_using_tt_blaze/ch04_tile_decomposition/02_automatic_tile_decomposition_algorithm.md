# Automatic Tile Decomposition Algorithm

Given an arbitrary PyTorch tensor shape `[A, B, C, D]`, a data type, and an operation context, the tile decomposition algorithm produces a complete ShapeDescriptor -- the three shape layers from File 01, a tile geometry, page size, and tile counts. This section defines the algorithm step by step, with validation and error handling at every stage.

> *Design Principle P1 (Separate Computation Intent from Execution Strategy):* The developer provides a shape, a dtype, and an op hint -- the "what." The tile decomposition algorithm applies tile selection, padding arithmetic, and page size computation -- the "how." These two concerns never interleave: `tile_decompose()` is a pure function with no side effects and no dependence on global state.

**What you will learn:**

- The complete five-step tile decomposition algorithm with inputs, outputs, validation, and error conditions at each step
- Step 1: How to identify tiled dimensions vs. batch dimensions (the last-two-dims convention)
- Step 2: How to select tile shape based on operation type, mirroring `interpret_tile()` in `blaze/utils.py`
- Step 3: How to compute padded dimensions with correct ceiling arithmetic and overflow protection
- Step 4: How to compute `num_pages` and `page_size` from padded dimensions and data format, including correct handling of block floating-point formats (bfloat8_b = 1,088 bytes, not 1,024)
- Step 5: How higher-rank tensors map batch dimensions to loop bounds or core-mapping dimensions
- Edge cases: scalar tensors, 1D tensors, tensors smaller than a single tile, non-standard tile shapes, and dimension overflow
- The relationship between this algorithm and the ShapeEngine layer of TensorAdapter

---

## Algorithm Overview

The tile decomposition algorithm is a pure function: given a tensor shape, dtype, and op_hint, it produces a ShapeDescriptor. No hardware state, no side effects, no global configuration.

> *Design Principle P3 (Validate at Each Lowering Step):* Each of the five steps validates its output before passing to the next. A zero-dimension caught in Step 1 never reaches Step 4's page-size computation; a tile-shape incompatibility caught in Step 2 never reaches Step 3's padding arithmetic.

```
Input:
  shape:    tuple[int, ...]     e.g., [2, 8, 4096, 11008]
  dtype:    str                 e.g., "bfloat16"
  op_hint:  str                 e.g., "matmul.in0"

Output:
  ShapeDescriptor with all fields populated and validated

Pipeline:
  Step 1: Normalize shape (identify batch dims vs. tiled dims)
  Step 2: Select tile shape from op_hint and tensor dimensions
  Step 3: Compute padded dimensions (ceiling alignment)
  Step 4: Compute num_pages, page_size, total_tiles
  Step 5: Handle higher-rank tensors (batch dims -> loop bounds)
  Return: ShapeDescriptor (frozen, validated)
```

---

## Step 1: Identify Tiled Dimensions vs. Batch Dimensions

### The Convention

Tenstorrent hardware processes data in 2D tiles. The **last two dimensions** of a tensor are the tile candidates (the "spatial" dimensions that map to tile rows and columns). All preceding dimensions are **batch dimensions** that become outer loop bounds or core-distribution indices.

```
Shape:  [A, B, C, D]
         ^  ^  ^  ^
         |  |  |  +-- tiled dimension (tile width / columns)
         |  |  +----- tiled dimension (tile height / rows)
         |  +-------- batch dimension
         +----------- batch dimension
```

```python
# PROPOSED -- dimension classification
def _classify_dimensions(shape: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Split a shape into batch dimensions and spatial (tiled) dimensions.

    Returns:
        (batch_dims, (height, width))

    Raises:
        ValueError: if shape has fewer than 2 dimensions after normalization.
    """
    if len(shape) < 2:
        raise ValueError(
            f"Shape {shape} has fewer than 2 dimensions. "
            f"Normalize with _validate_logical_shape() first."
        )
    batch = shape[:-2]
    spatial = (shape[-2], shape[-1])
    return batch, spatial
```

### Normalization

Before dimension classification, the logical shape is normalized:

| Input Shape | Normalized Shape | Rule |
|-------------|-----------------|------|
| `()` (scalar) | `(1, 1)` | Scalar promoted to single-element 2D |
| `(N,)` (1D) | `(1, N)` | 1D treated as single row |
| `(H, W)` (2D) | `(H, W)` | No change |
| `(B, H, W)` (3D) | `(B, H, W)` | B is batch, H and W are tiled |
| `(B1, B2, H, W)` (4D) | `(B1, B2, H, W)` | B1 and B2 are batch |

#### Error Conditions in Step 1

| Condition | Error | Impact If Undetected |
|-----------|-------|---------------------|
| Empty shape tuple | `ValueError("Empty shape")` | Division by zero in tile computation |
| Shape with zero dimension | `ValueError("Zero-size dim at index {i}")` | Zero-tile grid produces empty CB |
| Shape with negative dimension | `ValueError("Negative dim at index {i}")` | Negative tile counts wrap to huge unsigned values on hardware |
| Extremely high rank (>8) | Warning (not error) | Works correctly but may indicate a bug |

---

## Step 2: Select Tile Shape Based on Op Type

### The Tile Shape Registry

Different operations require different tile geometries because their compute kernels are compiled for specific tile layouts. The selection is driven by the `op_hint` parameter:

```python
# PROPOSED -- tile shape registry
TILE_REGISTRY: dict[str, tuple[int, int]] = {
    # Matmul family: 32x32 standard tiles
    "matmul":               (32, 32),
    "matmul_cb":            (32, 32),
    "kn_sliced_matmul":     (32, 32),
    "dram_streaming_matmul": (32, 32),

    # Normalization family: row tiles for vector processing
    "rmsnorm":              (1, 32),
    "padded_rmsnorm":       (1, 32),
    "layernorm":            (1, 32),
    "broadcast_rmsnorm":    (1, 32),

    # Elementwise family: inherit from input (default 32x32)
    "add":                  (32, 32),
    "mul":                  (32, 32),
    "silu":                 (32, 32),
    "relu":                 (32, 32),
    "gelu":                 (32, 32),
    "eltwise_mul":          (32, 32),

    # Attention family
    "sdpa":                 (32, 32),
    "softmax":              (32, 32),

    # Data movement: inherit from input
    "mcast":                (32, 32),
    "gather":               (32, 32),

    # Reduction family
    "reduce":               (32, 32),
    "gated_reduce":         (32, 32),
    "gated_local_reduce":   (32, 32),
}

DEFAULT_TILE_SHAPE = (32, 32)
```

### The Selection Algorithm

> *Design Principle P4 (Provide Defaults with Per-Decision Overrides):* The registry provides sensible defaults for every known op. A developer can override the tile shape on a per-tensor basis (Level 1 override), which takes priority over the registry.

```python
# PROPOSED -- tile shape selection with validation
VALID_TILE_SHAPES = {(32, 32), (16, 32), (1, 32), (4, 32), (8, 32)}

def _select_tile_shape(
    op_hint: str,
    spatial_dims: tuple[int, int],
    override: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Select tile shape for a given operation and tensor dimensions.

    Priority:
    1. Developer override (Level 1)
    2. Op-specific registry lookup
    3. Default (32, 32)

    Validates that the selected tile shape is in the hardware-supported set.
    """
    # Level 1 override
    if override is not None:
        _validate_tile_shape(override)
        return override

    # Parse op_hint to extract op_type
    op_type = op_hint.split(".")[0] if "." in op_hint else op_hint

    # Registry lookup
    tile = TILE_REGISTRY.get(op_type, DEFAULT_TILE_SHAPE)

    # Special case: interpret_tile logic for normalization ops
    if op_type in ("rmsnorm", "padded_rmsnorm", "layernorm", "broadcast_rmsnorm"):
        tile = _select_norm_tile_with_limit(spatial_dims[-1])

    _validate_tile_shape(tile)
    return tile


def _validate_tile_shape(tile: tuple[int, int]) -> None:
    """Validate that a tile shape is hardware-supported."""
    th, tw = tile
    if th <= 0 or tw <= 0:
        raise ValueError(f"Tile dimensions must be positive, got ({th}, {tw})")
    if tw != 32:
        raise ValueError(
            f"Tile width must be 32 (hardware constraint), got {tw}. "
            f"Tenstorrent datapaths operate on 32-element-wide rows."
        )
    if tile not in VALID_TILE_SHAPES:
        raise ValueError(
            f"Tile shape {tile} not in hardware-supported set {VALID_TILE_SHAPES}."
        )
```

### Normalization Tile Selection: Mirroring interpret_tile()

For normalization operations, the tile shape depends on the width dimension. The logic mirrors `interpret_tile()` from `blaze/utils.py`. Data is stored in L1 as contiguous 1x32 rows (the **storage tile**), but the compute LLK reinterprets them as either (32, 32) or (16, 32) **compute tiles** for the SFPU reduction operations (see the storage-tile vs. compute-tile distinction in File 01).

```python
# PROPOSED -- norm tile selection with hardware dest register limit
def _select_norm_tile_with_limit(width: int) -> tuple[int, int]:
    """Select norm tile shape with hardware dest register limit.

    The SFPU mul_reduce_scalar_tile instruction can accumulate at most
    8 tiles in the destination register. If the half-tile count exceeds
    this limit, fall back to full tiles.

    Mirrors interpret_tile() from blaze/utils.py.
    """
    MAX_TILES = 8
    FULL_W = 32
    FULL_H = 32
    HALF_H = 16

    full_sz = FULL_H * FULL_W   # 1024
    half_sz = HALF_H * FULL_W   # 512

    num_cols = width // FULL_W
    if num_cols == 0:
        return (1, 32)  # Width less than 32: pad to one row tile

    # Already aligned to full or half tile boundary?
    if width % full_sz == 0:
        return (FULL_H, FULL_W)
    if width % half_sz == 0:
        num_half = width // half_sz
        if num_half <= MAX_TILES:
            return (HALF_H, FULL_W)
        else:
            return (FULL_H, FULL_W)

    # Not aligned: need padding
    padded_half = round_up(width, half_sz)
    if padded_half // half_sz <= MAX_TILES:
        return (HALF_H, FULL_W)

    return (FULL_H, FULL_W)
```

#### Walkthrough: Norm Tile Selection for LLaMA-7B (width=4096) and DeepSeek V3 (width=7168)

```
LLaMA-7B: width = 4096
  4096 % 1024 == 0 -> use full (32, 32)

DeepSeek V3: width = 7168
  7168 % 1024 == 0 -> use full (32, 32)
  (7168 / 1024 = 7 tiles, within MAX_TILES limit)
```

Both model dimensions happen to be multiples of 1024, selecting full (32, 32) tiles. Dimensions like 768 (GPT-2 small) would select (16, 32) tiles: `768 % 1024 != 0`, `768 % 512 == 0`, `768 / 512 = 1.5` -- actually `768 / 512 = 1` remainder `256`, so `768 % 512 != 0` either. Checking: `round_up(768, 512) = 1024`, `1024 / 512 = 2 <= 8`, so (16, 32).

> **Warning:** The `interpret_tile()` function in `blaze/utils.py` also enforces the `MAX_TILES = 8` constraint for the `mul_reduce_scalar_tile` destination register limit. If `width // (tile_h * tile_w) > 8` with half-tiles, the function falls back to full tiles. TensorAdapter must replicate this constraint.

#### Error Conditions in Step 2

| Condition | Error | Recovery |
|-----------|-------|----------|
| Unknown op_hint | `ValueError("Unknown op_hint: '{op_type}'")` | Use default tile or provide override |
| Unsupported tile shape override | `ValueError("Tile shape {tile} not in supported set")` | Choose from VALID_TILE_SHAPES |
| Width < 32 for norm op | Returns `(1, 32)` with padding to 32 elements | Valid but wasteful |
| Half-tile count exceeds MAX_TILES | Automatic fallback to full tiles | Transparent; logged |

---

## Step 3: Compute Padded Dimensions

### The Ceiling Alignment Algorithm

Each of the last two dimensions is rounded up to the nearest multiple of its corresponding tile dimension:

```python
# PROPOSED -- padded dimension computation with overflow protection
def _compute_padded_dims(
    spatial: tuple[int, int],
    tile_shape: tuple[int, int],
) -> tuple[int, int]:
    """Compute padded spatial dimensions.

    padded_h = ceil(h / tile_h) * tile_h
    padded_w = ceil(w / tile_w) * tile_w
    """
    h, w = spatial
    th, tw = tile_shape

    padded_h = round_up(h, th)
    padded_w = round_up(w, tw)

    # Overflow check: hardware addresses are 32-bit
    MAX_DIM = 2**31 - 1
    if padded_h > MAX_DIM or padded_w > MAX_DIM:
        raise ValueError(
            f"Padded dimension overflow: ({padded_h}, {padded_w}) exceeds "
            f"hardware address space."
        )

    assert padded_h >= h and padded_w >= w, "Padding must not shrink dimensions"
    assert padded_h % th == 0 and padded_w % tw == 0, "Alignment invariant"

    return (padded_h, padded_w)
```

### Full Padded Shape Construction

```python
# PROPOSED -- full padded shape from logical shape
def _build_padded_shape(
    logical: tuple[int, ...],
    tile_shape: tuple[int, int],
) -> tuple[int, ...]:
    """Build complete padded shape: batch dims unchanged, spatial dims padded."""
    batch = logical[:-2]
    spatial = (logical[-2], logical[-1])
    padded_spatial = _compute_padded_dims(spatial, tile_shape)
    return batch + padded_spatial
```

#### Edge Case: Already-Aligned Dimensions

When a dimension is already a multiple of the tile dimension, `round_up` returns the value unchanged:

```
round_up(4096, 32) = 4096    # LLaMA-7B hidden dim: no change
round_up(11008, 32) = 11008  # LLaMA-7B intermediate: 11008 / 32 = 344 exactly
round_up(7168, 32) = 7168    # DeepSeek V3 hidden dim: 7168 / 32 = 224 exactly
round_up(2048, 32) = 2048    # DeepSeek V3 MoE expert width: 2048 / 32 = 64 exactly
```

The algorithm correctly produces `padded == logical` in these cases, and the ShapeDescriptor sets `padding_strategy = "NONE"`.

#### Error Conditions in Step 3

| Condition | Error | Impact |
|-----------|-------|--------|
| Overflow in `round_up` | `ValueError("Padded dimension overflow")` | Would produce invalid L1 addresses |
| Batch dim changed | `RuntimeError("BUG: Batch dim changed")` | Indicates a logic error in padding code |

---

## Step 4: Compute num_pages and page_size

### Page Size Computation

Page size depends on tile shape and data format. For unpacked formats, it is a simple product. For block floating-point formats, it includes shared exponent overhead.

```python
# PROPOSED -- page size computation with format dispatch
def _compute_page_size(
    tile_shape: tuple[int, int],
    data_format: str,
) -> int:
    """Compute the byte size of one tile page.

    For unpacked formats (bfloat16, float32, uint types):
        page_size = tile_h * tile_w * DTYPE_BYTES[format]

    For block floating-point formats (bfloat8_b, bfloat4_b):
        page_size includes shared exponent overhead per face.
        Must use tile.get_tile_size(data_format) for accuracy.
    """
    th, tw = tile_shape

    # Known BFP page sizes for 32x32 tiles (from blaze/ops/sdpa/op.py)
    BFP_TILE_SIZES = {
        "bfloat8_b": 1088,   # NOT 1024; includes exponent bytes
        "bfloat4_b": 576,    # NOT 512; includes exponent bytes
    }

    if data_format in BFP_TILE_SIZES:
        if tile_shape == (32, 32):
            return BFP_TILE_SIZES[data_format]
        else:
            raise ValueError(
                f"BFP format '{data_format}' is only supported with (32, 32) tiles. "
                f"Got tile_shape={tile_shape}."
            )

    # Unpacked formats
    DTYPE_BYTES = {
        "bfloat16": 2, "float16": 2, "float32": 4,
        "uint8": 1, "uint16": 2, "uint32": 4,
    }

    if data_format not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{data_format}'.")

    page_size = th * tw * DTYPE_BYTES[data_format]
    if page_size <= 0:
        raise ValueError(f"Invalid page_size {page_size}")

    return page_size
```

> **Warning:** A common mistake is computing bfloat8_b page size as `32 * 32 * 1 = 1024 bytes`. The correct value is **1,088 bytes** because each 16x16 face includes an extra exponent byte per row (16 exponent bytes per face, 4 faces per tile = 64 extra bytes). Similarly, bfloat4_b is **576 bytes**, not 512. Using the wrong page_size causes L1 address misalignment for every subsequent CB, corrupting all downstream data.

### num_pages Computation

The number of pages (tiles) a CB holds depends on the operation and the role of the tensor:

```python
# PROPOSED -- num_pages computation by operation role
def _compute_num_pages(
    tile_grid: tuple[int, ...],
    op_hint: str,
    total_tiles: int,
) -> int:
    """Compute the number of pages a CB should hold for a given role.

    This is operation-dependent:
    - Matmul in0 (activation): K tiles (contraction dimension)
    - Matmul in1 (weight): total weight tiles (direct address, all at once)
    - Matmul out: out_w_per_core tiles
    - Elementwise input: 1 page (streaming, single-buffered)
    - RMSNorm input: all tiles (accumulation across width)
    - Reduction: all tiles (accumulation)

    For double-buffering, multiply by 2 (handled by CB sizing in Chapter 7).
    """
    op_type, port = _parse_op_hint(op_hint)

    if op_type in ("matmul", "matmul_cb", "kn_sliced_matmul"):
        if port == "in0":
            return tile_grid[-1]            # K dimension
        elif port == "in1":
            return total_tiles              # Weight: all pages
        elif port == "out":
            return tile_grid[-1]            # Updated by CT arg derivation
        else:
            return total_tiles

    elif op_type in ("rmsnorm", "padded_rmsnorm", "layernorm"):
        return tile_grid[-1]                # Width accumulation

    elif op_type in ("reduce", "gated_reduce", "gated_local_reduce"):
        return total_tiles

    elif op_type in ("add", "mul", "silu", "relu", "gelu", "eltwise_mul"):
        return 1                            # Streaming: one tile at a time

    elif op_type in ("mcast", "gather"):
        return total_tiles

    elif op_type in ("sdpa", "softmax"):
        return tile_grid[-1]                # K tiles as default

    else:
        warnings.warn(f"Unknown op_type '{op_type}'. Defaulting to total_tiles.")
        return total_tiles
```

> **Warning:** The `num_pages` computation above provides initial upper-bound estimates. The actual CB sizing is refined by the L1 budget validator in Chapter 7, which may reduce `num_pages` to fit within the ~1.5 MB L1 constraint.

#### Error Conditions in Step 4

| Condition | Error | Impact |
|-----------|-------|--------|
| Unknown data format | `ValueError("Unsupported dtype")` | Cannot compute page_size |
| BFP format with non-32x32 tile | `ValueError("BFP only supports 32x32")` | Hardware constraint |
| Page size is zero | `ValueError("Invalid page_size")` | CB allocation with zero size crashes Metal |
| num_pages is zero | Cannot happen if tile_grid has positive dimensions | Would produce empty CB |

---

## Step 5: Handle Higher-Rank Tensors

### Batch Dimensions as Loop Bounds

For tensors with rank > 2, the batch dimensions represent independent work items that can be distributed across cores or iterated over:

```python
# PROPOSED -- batch dimension handling
def _compute_batch_strategy(
    batch_dims: tuple[int, ...],
    tile_grid_spatial: tuple[int, int],
    available_cores: int,
) -> dict:
    """Determine how batch dimensions map to hardware execution.

    Two strategies:
    1. Loop: batch dims become outer loop iterations on each core.
    2. Distribute: batch dims are flattened and distributed across cores.
    """
    total_batch = 1
    for d in batch_dims:
        total_batch *= d

    if total_batch == 1:
        return {"strategy": "spatial_only", "loop_count": 1}
    elif total_batch <= available_cores:
        return {"strategy": "batch_distribute", "cores_per_batch": available_cores // total_batch}
    else:
        return {"strategy": "batch_loop", "loop_count": total_batch}
```

### How Batch Affects the Tile Grid

The tile grid shape includes batch dimensions:

```
Shape:      [2, 8, 4096, 4096]
Tile:       (32, 32)
Tile grid:  [2, 8, 128, 128]
             ^  ^   ^    ^
             |  |   |    +-- spatial: 4096 / 32 = 128 column tiles
             |  |   +------- spatial: 4096 / 32 = 128 row tiles
             |  +----------- batch: 8 (loop or core-distributed)
             +-------------- batch: 2 (loop or core-distributed)

Total tiles: 2 * 8 * 128 * 128 = 262,144
```

### The Impact on CB Allocation

Batch dimensions do **not** increase CB `num_pages` directly. A CB holds tiles for a single batch element. The batch loop iterates over elements, reusing the same CB for each:

```
For a [2, 8, 128, 128] tile grid:
  CB num_pages = f(spatial_grid)    e.g., 128 tiles for K dimension
  Outer loop: for batch_idx in range(2 * 8 = 16):
      process(batch[batch_idx], cb)  # same CB, different data
```

> **Warning:** If batch dimensions are incorrectly included in `num_pages`, the CB will request `16 * 128 = 2048` pages instead of 128 pages. At 2,048 bytes per page, this is 4 MB instead of 256 KB -- exceeding L1. The batch-spatial separation is a critical correctness requirement.

---

## The Complete Algorithm

Putting all five steps together:

```python
# PROPOSED -- complete tile decomposition algorithm
def tile_decompose(
    shape: tuple[int, ...],
    dtype: str = "bfloat16",
    op_hint: str = "",
    *,
    tile_shape_override: tuple[int, int] | None = None,
) -> "ShapeDescriptor":
    """Decompose a PyTorch tensor shape into a tile-aligned ShapeDescriptor.

    This is the core algorithm of ShapeEngine.full_analysis().

    Args:
        shape: PyTorch tensor shape (any rank)
        dtype: Data format string (e.g., "bfloat16", "bfloat8_b")
        op_hint: Operation context (e.g., "matmul.in0")
        tile_shape_override: Override tile shape (Level 1)

    Returns:
        ShapeDescriptor with all fields populated and validated
    """
    # Step 1: Normalize and validate
    logical = _normalize_shape(shape)
    batch, spatial = _classify_dimensions(logical)

    # Step 2: Select tile shape
    tile = _select_tile_shape(op_hint, spatial, override=tile_shape_override)

    # Step 3: Compute padded dimensions
    padded = _build_padded_shape(logical, tile)

    # Step 4: Compute tile grid, page size, num_pages
    tile_grid = _compute_tile_grid(padded, tile)
    total_tiles = _total_tiles(tile_grid)
    page_size = _compute_page_size(tile, dtype)
    num_pages = _compute_num_pages(tile_grid, op_hint, total_tiles)

    # Step 5: Determine padding strategy (deferred to Chapter 5)
    needs_padding = logical != padded
    padding_strategy = "ZERO" if needs_padding else "NONE"

    # Construct and validate ShapeDescriptor
    return ShapeDescriptor(
        logical_shape=logical,
        tile_shape=tile,
        padded_shape=padded,
        tile_grid_shape=tile_grid,
        total_tiles=total_tiles,
        data_format=dtype,
        page_size=page_size,
        padding_strategy=padding_strategy,
        padding_fill=0.0,
        source="tile_decompose",
        op_hint=op_hint,
    )
```

---

## Edge Cases Catalog

### Scalar Tensors

```python
shape = ()
# Step 1: Promoted to (1, 1)
# Step 2: Tile (32, 32) for matmul, (1, 32) for norm
# Step 3: Padded to (32, 32) or (1, 32)
# Step 4: 1 tile, page_size depends on format
```

> **Warning:** Scalar tensors in LLM inference are rare but appear in scaling factors and epsilon values. A scalar promoted to `(1, 1)` and padded to `(32, 32)` wastes 1,023 of 1,024 tile positions. Consider passing scalars as compile-time constants (CT args) rather than as tiles.

### 1D Tensors

```python
shape = (4096,)
# Step 1: Promoted to (1, 4096)
# Step 2: (1, 32) for norm ops, (32, 32) for others
# Step 3: (1, 4096) for (1, 32) tiles; (32, 4096) for (32, 32) tiles
# Step 4: 128 tiles either way; page_size differs
```

### Tensors Smaller Than One Tile

```python
shape = (3, 5)
tile = (32, 32)
# Step 3: padded = (32, 32)
# Step 4: 1 tile, 2048 bytes
# 15 of 1024 elements are real data (1.5% utilization)
```

TensorAdapter should emit a warning for utilization below a threshold:

```python
# PROPOSED -- utilization warning
def _check_tile_utilization(logical: tuple[int, ...], padded: tuple[int, ...]):
    """Warn when tile utilization is very low."""
    logical_elems = 1
    padded_elems = 1
    for d in logical:
        logical_elems *= d
    for d in padded:
        padded_elems *= d
    utilization = logical_elems / padded_elems
    if utilization < 0.1:
        warnings.warn(
            f"Very low tile utilization: {utilization:.1%}. "
            f"Shape {logical} padded to {padded}."
        )
```

### Non-Standard Tile Shapes (16x32 for RMSNorm)

```python
shape = (1, 768)
op_hint = "rmsnorm.input"
# Step 2: interpret_tile(768): 768/32=24 tiles, 24%32!=0 -> use_half -> (16, 32)
# Step 3: padded = (16, 768)  -- height padded from 1 to 16
# Step 4: grid = (1, 24), 24 tiles of (16, 32) = 512 elements each
# page_size = 16 * 32 * 2 = 1024 bytes
# Total: 24 * 1024 = 24,576 bytes (24 KB)
```

Compare with (32, 32) tiles: padded would be (32, 768), grid (1, 24), page_size 2,048, total 49,152 bytes. The (16, 32) tile saves 50% of L1 memory.

### Integer Overflow in Total Tiles

```python
shape = (1000, 1000, 4096, 4096)
tile = (32, 32)
grid = (1000, 1000, 128, 128)
total_tiles = 1000 * 1000 * 128 * 128 = 16,384,000,000
```

This exceeds `2^31 - 1`. While Python handles arbitrary-precision integers, the hardware address space does not:

```python
# PROPOSED -- total tiles overflow check
MAX_TOTAL_TILES = 2**31 - 1

def _validate_total_tiles(total: int):
    if total > MAX_TOTAL_TILES:
        raise ValueError(
            f"Total tile count {total:,} exceeds hardware limit {MAX_TOTAL_TILES:,}. "
            f"Distribute across more cores or reduce batch dimensions."
        )
```

---

## Relationship to ShapeEngine

The tile decomposition algorithm is implemented as `ShapeEngine.full_analysis()` in TensorAdapter's Layer 2 (as defined in [Chapter 3, File 01](../ch03_tensor_adapter/01_architecture_overview.md)):

```python
# PROPOSED -- ShapeEngine.full_analysis() calls tile_decompose
class ShapeEngine:
    def full_analysis(
        self,
        tensor_or_shape,
        op_hint: str,
        *,
        tile_shape: tuple[int, int] | None = None,
        data_format: str | None = None,
    ) -> ShapeDescriptor:
        """Complete shape analysis: normalize, tile, pad, compute.

        This is the entry point that orchestrates all five steps.
        """
        if isinstance(tensor_or_shape, torch.Tensor):
            shape = tuple(tensor_or_shape.shape)
            dtype = data_format or _torch_dtype_to_str(tensor_or_shape.dtype)
        elif isinstance(tensor_or_shape, tuple):
            shape = tensor_or_shape
            dtype = data_format or "bfloat16"
        else:
            raise TypeError(f"Expected torch.Tensor or tuple, got {type(tensor_or_shape)}")

        return tile_decompose(
            shape=shape,
            dtype=dtype,
            op_hint=op_hint,
            tile_shape_override=tile_shape,
        )
```

---

## Validation Summary

Every step in the algorithm includes validation. The complete set of error conditions:

| Step | Condition | Error Type | Severity |
|------|-----------|-----------|----------|
| 1 | Scalar tensor | Warning + promote | Low |
| 1 | 1D tensor | Warning + promote | Low |
| 1 | Zero-size dimension | ValueError | High (crash prevention) |
| 1 | Negative dimension | ValueError | High (crash prevention) |
| 2 | Unknown op_hint | ValueError | Medium (recoverable) |
| 2 | Invalid tile shape | ValueError | High (hardware constraint) |
| 2 | BFP with non-32x32 tile | ValueError | High (hardware constraint) |
| 3 | Overflow in round_up | ValueError | High (address space limit) |
| 3 | Batch dimension changed | RuntimeError | Critical (algorithm bug) |
| 4 | Unknown data format | ValueError | High |
| 4 | Page size zero | ValueError | High (empty CB) |
| 4 | Total tiles overflow | ValueError | High (address space limit) |
| 5 | Low tile utilization | Warning | Low (performance only) |

---

## Key Takeaways

- The tile decomposition algorithm is a five-step pipeline: normalize shape, select tile geometry, compute padded dimensions, derive page counts and sizes, and handle batch dimensions. Each step validates its output before passing to the next.
- Tile shape selection is operation-dependent: 32x32 for matmul and most compute ops, 16x32 or 1x32 for normalization ops. The selection mirrors `interpret_tile()` in `blaze/utils.py` and respects the `MAX_TILES = 8` dest register constraint.
- Page size computation must dispatch on data format: simple multiplication for unpacked formats, lookup table for BFP formats. Using the wrong formula for bfloat8_b produces a 64-byte-per-tile error that accumulates and corrupts L1 layout.
- Batch dimensions (all dims except the last two) are never padded and never increase CB `num_pages`. They become outer loop bounds or core-distribution indices. Including batch dims in `num_pages` is a critical correctness bug.
- The algorithm handles all edge cases -- scalars, 1D tensors, sub-tile shapes, extremely large shapes, integer overflow -- with explicit validation and clear error messages, replacing the current silent-corruption failure mode.

## Source Files

- `blaze/cb_engine.py` -- `_tile_page_size()`, `DTYPE_BYTES`, `DEFAULT_TILE_SHAPE`, `DEFAULT_DTYPE`
- `blaze/utils.py` -- `interpret_tile()`, `interpret_tile_padded()`, `round_up()`, `UNPACKED_DTYPE_BYTES`
- `blaze/ops/sdpa/op.py` -- `_TILE_SIZES` (BFP page size constants: bfloat8_b=1088, bfloat4_b=576)
- `blaze/graph.py` -- `TensorPort` (tile_shape field used by CBEngine)
- `blaze/cb_handle.py` -- `CBHandle` (num_pages, page_size target of this algorithm)

---

← Previous: [Chapter 3: The TensorAdapter Architecture](../ch03_tensor_adapter/) | Next: [Chapter 5: Transparent Padding Management](../ch05_padding/) →
