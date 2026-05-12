# The Three Shape Layers: Logical, Padded, and Tile Grid

Every PyTorch tensor that enters the Tenstorrent execution model must be described at three levels of abstraction simultaneously: the logical shape the developer wrote, the padded shape that satisfies tile-boundary alignment, and the tile grid shape that tells the hardware how many tiles to process. Losing any one of these representations creates bugs -- silent numerical errors, incorrect output slicing, or misconfigured circular buffers. This section formally defines all three layers, traces worked examples through the decomposition using dimensions from both LLaMA-7B and DeepSeek V3, explains why simultaneous tracking is essential, and connects each layer to the existing TT-Blaze code that computes it today.

> *Design Principle P2 (Carry Metadata from Both Worlds Simultaneously):* The three shape layers are the concrete realization of P2 from Chapter 2. `ShapeDescriptor` carries PyTorch-world information (logical shape) alongside hardware-world information (padded shape, tile grid, page size) in a single immutable structure, ensuring neither world's data is lost during lowering.

**What you will learn:**

- Formal definitions of logical_shape, padded_shape, and tile_grid_shape, including the mathematical relationships and invariants that bind them
- The distinction between **storage tiles** (how data is laid out in L1, e.g., 1x32 rows) and **compute tiles** (how the FPU reinterprets data, e.g., 16x32 or 32x32 faces)
- Worked examples tracing `[1, 37]`, `[1, 768]`, `[1, 4096]` (LLaMA-7B), `[1, 7168]` (DeepSeek V3), and `[2, 8, 4096, 4096]` through all three layers
- Why all three must be tracked simultaneously: logical for the user API and output slicing, padded for hardware correctness, tiled for CB allocation and CT arg derivation
- How TT-Blaze currently computes these values: `_tile_page_size()` in `cb_engine.py`, `DEFAULT_TILE_SHAPE`, `interpret_tile()`, and `interpret_tile_padded()` in `utils.py`
- How ShapeDescriptor (from Chapter 3) encapsulates all three layers into a single frozen dataclass

---

## Formal Definitions

### Layer 1: Logical Shape

The **logical shape** is the tensor shape as specified by the PyTorch developer. It has no tile-alignment constraints, no padding, and no hardware awareness. It is the shape the developer sees when they call `tensor.shape`.

```
logical_shape: tuple[int, ...]
```

Examples from real LLM workloads:

| Tensor | Logical Shape | Context |
|--------|--------------|---------|
| LLaMA-7B activation (single token) | `[1, 4096]` | Hidden dim |
| LLaMA-7B gate/up weight | `[4096, 11008]` | Hidden-to-intermediate |
| DeepSeek V3 activation (single token) | `[1, 7168]` | Hidden dim |
| DeepSeek V3 MoE expert weight | `[7168, 2048]` | Hidden-to-expert-intermediate |
| RMSNorm gamma | `[4096]` | 1D, promoted to `[1, 4096]` |
| Non-aligned vector | `[1, 37]` | Pathological case |
| Attention mask | `[1, 1, 128, 128]` | Causal mask for 128-token sequence |

The logical shape is the **ground truth** for the developer's intent. When the final output returns to PyTorch, the result must match the logical shape exactly -- any padding introduced for hardware execution must be stripped before the developer sees the result.

> **Warning:** Logical shape is the only shape layer that the developer should ever need to reason about. If the abstraction forces the developer to think about padded or tiled shapes, it has failed its design goal (P1: Separate Computation Intent from Execution Strategy).

#### Validation Rules for Logical Shape

| Condition | Error | Recovery Strategy |
|-----------|-------|-------------------|
| Any dimension is zero | `ValueError("Zero-size dimension at index {i}")` | Fix the input tensor |
| Any dimension is negative | `ValueError("Negative dimension at index {i}")` | Fix the input tensor |
| Rank is 0 (scalar) | Promote to `[1, 1]` with warning | Automatic promotion |
| Rank is 1 | Promote to `[1, N]` with warning | Automatic promotion treats 1D as a single row |

```python
# PROPOSED -- logical shape validation
def _validate_logical_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Validate and normalize a logical shape.

    Raises ValueError for invalid shapes. Promotes scalars to [1,1]
    and 1D tensors to [1, N] with a logged warning.
    """
    if len(shape) == 0:
        warnings.warn("Scalar tensor promoted to [1, 1] for tile decomposition")
        return (1, 1)
    if len(shape) == 1:
        warnings.warn(f"1D tensor [{shape[0]}] promoted to [1, {shape[0]}]")
        return (1, shape[0])
    for i, dim in enumerate(shape):
        if dim < 0:
            raise ValueError(f"Negative dimension {dim} at index {i}")
        if dim == 0:
            raise ValueError(
                f"Zero-size dimension at index {i}. "
                f"Cannot decompose empty tensor into tiles."
            )
    return shape
```

---

### Layer 2: Padded Shape

The **padded shape** is the logical shape after rounding the last two dimensions up to the nearest tile boundary. Batch dimensions (all dimensions except the last two) are never padded -- they become loop bounds or core-mapping dimensions.

The padding formula for each of the last two dimensions:

$$\text{padded\_dim} = \lceil \text{logical\_dim} / \text{tile\_dim} \rceil \times \text{tile\_dim}$$

In Python:

```python
# EXISTING -- round_up utility (from blaze/utils.py)
def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
```

Applied to the last two dimensions:

```python
# PROPOSED -- padding computation
def _pad_to_tiles(
    logical: tuple[int, ...], tile_shape: tuple[int, int]
) -> tuple[int, ...]:
    """Pad the last two dimensions of logical_shape to tile boundaries.

    Batch dimensions (all but the last two) are preserved unchanged.
    """
    th, tw = tile_shape
    batch = logical[:-2]
    h, w = logical[-2], logical[-1]
    padded_h = round_up(h, th)
    padded_w = round_up(w, tw)
    return batch + (padded_h, padded_w)
```

#### Invariants

The padded shape must satisfy these invariants at all times:

1. **Same rank**: `len(padded_shape) == len(logical_shape)`
2. **Non-shrinking**: `padded_shape[i] >= logical_shape[i]` for all `i`
3. **Batch preservation**: `padded_shape[i] == logical_shape[i]` for `i < len(shape) - 2`
4. **Tile alignment**: `padded_shape[-2] % tile_h == 0` and `padded_shape[-1] % tile_w == 0`

> **Warning:** Violating invariant 3 (padding a batch dimension) would silently multiply the amount of data the hardware processes. A `[2, 8, 33, 64]` tensor with 32x32 tiles should become `[2, 8, 64, 64]`, NOT `[2, 32, 64, 64]`. The batch dimensions 2 and 8 are loop iteration counts, not spatial dimensions to tile over.

---

### Layer 3: Tile Grid Shape

The **tile grid shape** expresses the number of tiles along each dimension. For the last two dimensions, this is derived from the padded shape divided by the tile dimensions. For batch dimensions, the values pass through unchanged.

```
tile_grid_shape: tuple[int, ...]
```

The formula:

```python
# PROPOSED -- tile grid computation
def _compute_tile_grid(
    padded: tuple[int, ...], tile_shape: tuple[int, int]
) -> tuple[int, ...]:
    """Compute the number of tiles per dimension."""
    th, tw = tile_shape
    batch = padded[:-2]
    grid_h = padded[-2] // th
    grid_w = padded[-1] // tw
    return batch + (grid_h, grid_w)
```

The **total tile count** is the product of all dimensions in the tile grid shape:

$$\text{total\_tiles} = \prod_{i} \text{tile\_grid\_shape}[i]$$

| Tile Grid Dimension | Hardware Meaning | CB Allocation Impact |
|--------------------|-----------------|---------------------|
| `grid[-2]` (row tiles) | Number of row-tile iterations | Affects `RT_DIM` blocking for matmul |
| `grid[-1]` (col tiles) | Number of column-tile iterations | Affects `CT_DIM` blocking and `out_w_per_core` |
| `grid[:-2]` (batch dims) | Outer loop bounds or multi-core distribution | Determines core grid size or loop count |

---

## Storage Tiles vs. Compute Tiles

A critical subtlety -- drawn from V2's analysis -- is that the "tile shape" has two meanings depending on context:

| Concept | Definition | Example |
|---------|-----------|---------|
| **Storage tile** | How data is physically laid out in L1 | `(1, 32)` -- contiguous 32-element rows |
| **Compute tile** | How the FPU reinterprets that data for SFPU/FPU operations | `(16, 32)` or `(32, 32)` -- face-ordered for reduction |

The `interpret_tile()` function in `blaze/utils.py` selects the **compute tile**. The data in L1 is stored as contiguous 32-element rows (storage tile = 1x32), but the compute kernel reinterprets them as 16x32 or 32x32 tiles for face-order operations. ShapeDescriptor's `tile_shape` field tracks the **compute tile** because it determines `page_size` and the tile grid. The storage layout is implicit -- data always enters L1 as contiguous elements along the innermost dimension.

> **Warning:** Confusing storage tiles with compute tiles leads to incorrect page_size calculations. A `(1, 32)` storage row occupies 64 bytes (bfloat16), but the compute tile `(16, 32)` used by `interpret_tile()` has page_size = 1,024 bytes. ShapeDescriptor must use the compute tile geometry for `page_size` and `tile_grid_shape` derivations.

---

## Worked Examples

### Example 1: `[1, 37]` with 32x32 Tiles (Non-Aligned)

```
Logical shape:   [1, 37]
Tile shape:      (32, 32)

Step 1 - Pad height: ceil(1 / 32) * 32 = 32
Step 2 - Pad width:  ceil(37 / 32) * 32 = 64

Padded shape:    [32, 64]
Tile grid shape: [1, 2]
Total tiles:     2

Page size (bfloat16): 32 * 32 * 2 = 2,048 bytes
CB memory for all tiles: 2 * 2,048 = 4,096 bytes

Padding analysis:
  Real data:     1 * 37 = 37 elements
  Padded total:  32 * 64 = 2,048 elements
  Padding ratio: 98.2%
```

> **Warning:** A 98.2% padding ratio means the hardware processes 55x more data than necessary. For production workloads, shapes this misaligned should trigger a diagnostic warning. In practice, LLM workloads use dimensions like 4096 and 7168 that are already multiples of 32, making extreme padding ratios uncommon.

### Example 2: `[1, 768]` with 1x32 Row Tiles (RMSNorm)

```
Logical shape:   [1, 768]
Tile shape:      (1, 32)

Step 1 - Pad height: ceil(1 / 1) * 1 = 1   (no padding needed)
Step 2 - Pad width:  768 / 32 = 24 exactly  (already aligned)

Padded shape:    [1, 768]
Tile grid shape: [1, 24]
Total tiles:     24

Page size (bfloat16): 1 * 32 * 2 = 64 bytes
CB memory: 24 * 64 = 1,536 bytes
Padding ratio: 0%
```

Compare with 32x32 tiles: padded `[32, 768]`, grid `[1, 24]`, page_size 2,048, total 49,152 bytes. Row tiles reduce L1 usage by 32x for 1D data.

### Example 3: `[1, 4096]` with 32x32 Tiles (LLaMA-7B Activation)

```
Logical shape:   [1, 4096]
Tile shape:      (32, 32)

Padded shape:    [32, 4096]     (height padded from 1 to 32)
Tile grid shape: [1, 128]
Total tiles:     128

Page size (bfloat16): 2,048 bytes
CB memory: 128 * 2,048 = 262,144 bytes (256 KB)
L1 fraction: 256 KB / ~1,536 KB = 16.7% of per-core L1
```

This is the activation tensor for LLaMA-7B's linear layers. The `k_num_tiles` CT arg for matmul equals `tile_grid_shape[-1] = 128`.

### Example 4: `[1, 7168]` with 32x32 Tiles (DeepSeek V3 Activation)

DeepSeek V3 uses a 7168-dimensional hidden space (7168 = 224 * 32).

```
Logical shape:   [1, 7168]
Tile shape:      (32, 32)

Padded shape:    [32, 7168]     (already width-aligned)
Tile grid shape: [1, 224]
Total tiles:     224

Page size (bfloat16): 2,048 bytes
CB memory: 224 * 2,048 = 458,752 bytes (448 KB)
L1 fraction: 448 KB / ~1,536 KB = 29.2% of per-core L1
```

At 448 KB, this activation alone consumes nearly 30% of per-core L1. DeepSeek V3's larger hidden dimension makes L1 budget management more critical than for LLaMA-7B.

### Example 5: `[7168, 2048]` with 32x32 Tiles, bfloat8_b (DeepSeek V3 MoE Expert Weight)

```
Logical shape:   [7168, 2048]
Tile shape:      (32, 32)
Data format:     bfloat8_b

Padded shape:    [7168, 2048]   (both aligned)
Tile grid shape: [224, 64]
Total tiles:     14,336

Page size (bfloat8_b): 1,088 bytes (NOT 1,024)
Total: 14,336 * 1,088 = 15,597,568 bytes (~14.9 MB)
```

This weight tensor must be sharded across cores. With 16 cores, each core holds `14,336 / 16 = 896` tiles consuming `896 * 1,088 = 975 KB` per core -- a feasible L1 footprint.

### Example 6: `[2, 8, 4096, 4096]` with 32x32 Tiles (Batched)

```
Logical shape:   [2, 8, 4096, 4096]
Tile shape:      (32, 32)

Batch dims pass through: [2, 8, ...]
Padded shape:    [2, 8, 4096, 4096]    (already aligned)
Tile grid shape: [2, 8, 128, 128]
Total tiles:     2 * 8 * 128 * 128 = 262,144
```

The batch dimensions (2 and 8) become outer loop bounds. In a multi-core execution, these may be distributed across cores.

---

## Why All Three Layers Must Be Tracked Simultaneously

Each shape layer serves a distinct consumer, and losing any one creates a specific class of bug:

| Lost Layer | Failure Mode | Symptom |
|-----------|-------------|---------|
| **Logical** | Cannot unpad output | Developer sees `[32, 4096]` instead of `[1, 4096]`; downstream PyTorch ops receive wrong shape |
| **Padded** | Cannot compute correct CB size | CB too small (data overflows) or too large (wastes L1); kernel reads incorrect tile boundaries |
| **Tile grid** | Cannot derive CT args | `k_num_tiles` is wrong -> matmul accumulates incorrect number of tiles -> silent numerical corruption |

> **Warning:** All three failure modes produce **silent** errors in the current Blaze infrastructure. There are no runtime assertions that catch a wrong `k_num_tiles` or a missing output unpad. The three-level tracking is a correctness requirement, not a convenience feature. P3 (Validate at Each Lowering Step) demands catching these errors early via ShapeDescriptor's `__post_init__` checks.

### Logical Shape: Required for Output Correctness

```python
# PROPOSED -- output extraction uses logical_shape
def extract_output(result_tensor, output_descriptor: ShapeDescriptor):
    """Slice a padded output tensor back to the developer's logical shape."""
    logical = output_descriptor.logical_shape
    slices = tuple(slice(0, dim) for dim in logical)
    return result_tensor[slices]
```

### Padded Shape: Required for Hardware Correctness

The hardware datapaths operate on complete tiles. If the padded shape is incorrect, the Tensix core reads or writes out of bounds in L1, producing silent data corruption.

### Tile Grid Shape: Required for CB Allocation and CT Args

| Parameter | Derived From | Formula |
|-----------|-------------|---------|
| `num_pages` (input CB) | `tile_grid_shape[-1]` | K tiles for matmul in0 |
| `k_num_tiles` (CT arg) | `tile_grid_shape[-1]` of activation | Contraction dimension tiles |
| `out_w_per_core` (CT arg) | `total_tiles(weight) / k_num_tiles` | Output columns per core |
| `num_tiles` (reduction) | `total_tiles` | Total tiles to accumulate |
| CB total size | `num_pages * page_size` | Must fit in ~1.5 MB L1 |

---

## How TT-Blaze Currently Computes These Layers

Today, the three shape layers are computed in scattered locations without a unified data structure.

### _tile_page_size() in cb_engine.py

```python
# EXISTING -- from blaze/cb_engine.py
DEFAULT_TILE_SHAPE = (32, 32)
DEFAULT_DTYPE = "bfloat16"

def _tile_page_size(dtype: Optional[str], tile_shape: Optional[tuple[int, int]]) -> int:
    dt = dtype or DEFAULT_DTYPE
    ts = tile_shape or DEFAULT_TILE_SHAPE
    if dt not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dt}'.")
    return DTYPE_BYTES[dt] * ts[0] * ts[1]
```

This function handles only unpacked formats (bfloat16, float32, uint types). For block floating-point formats (bfloat8_b, bfloat4_b), the actual page size includes shared exponent overhead and must be obtained via `tile.get_tile_size(data_format)`.

> **Warning:** Using `_tile_page_size()` for bfloat8_b produces the wrong result. A 32x32 bfloat8_b tile is approximately 1,088 bytes, not `1 * 32 * 32 = 1,024 bytes`. The 64-byte discrepancy (shared exponent bytes) accumulates over many tiles and causes L1 address misalignment.

### The Complete Page Size Reference Table

For every combination of tile shape and data format used in TT-Blaze:

| Data Format | (32, 32) Tile | (16, 32) Tile | (1, 32) Tile |
|-------------|--------------|--------------|-------------|
| **float32** | 4,096 B | 2,048 B | 128 B |
| **bfloat16** | 2,048 B | 1,024 B | 64 B |
| **bfloat8_b** | 1,088 B | N/A* | N/A* |
| **bfloat4_b** | 576 B | N/A* | N/A* |
| **uint8** | 1,024 B | 512 B | 32 B |
| **uint16** | 2,048 B | 1,024 B | 64 B |
| **uint32** | 4,096 B | 2,048 B | 128 B |

\* Block floating-point formats are currently defined only for `(32, 32)` tiles.

**bfloat8_b page_size breakdown for a (32, 32) tile:**
```
32 * 32 = 1,024 elements
Each element: 8-bit mantissa = 1 byte -> 1,024 bytes for mantissa data
Shared exponent overhead: 4 faces * 16 bytes per face = 64 bytes
Total: 1,024 + 64 = 1,088 bytes
```

**bfloat4_b page_size breakdown for a (32, 32) tile:**
```
32 * 32 = 1,024 elements
Each element: 4-bit mantissa = 0.5 byte -> 512 bytes for mantissa data
Shared exponent overhead: 4 faces * 16 bytes per face = 64 bytes
Total: 512 + 64 = 576 bytes
```

### TensorPort.tile_shape in graph.py

```python
# EXISTING -- from blaze/graph.py
@dataclass
class TensorPort:
    name: str
    dtype: Optional[str] = None
    tile_shape: Optional[tuple[int, int]] = None
```

TensorPort carries the tile shape per graph edge, but it does not carry logical shape, padded shape, or tile grid counts. The gap between what TensorPort provides and what the system needs is precisely what ShapeDescriptor fills.

### The Scattered State of Shape Knowledge

| Shape Information | Where It Lives Today | Problem |
|-------------------|---------------------|---------|
| Logical shape | `torch.Tensor.shape` or developer's head | Lost after tilization |
| Padded dimensions | Computed inline in each op's `emit()` | Not propagated to downstream ops |
| Tile shape | `TensorPort.tile_shape` or `CBHandle.tile_desc` | Does not include logical or padded |
| Tile count | Computed as `num_pages` or from tensor metadata | Meaning overloaded |
| Page size | `CBHandle.page_size` or `_tile_page_size()` | No connection to logical shape |

ShapeDescriptor consolidates all of this into a single frozen dataclass with validated invariants. The decomposition algorithm defined in [File 02](./02_automatic_tile_decomposition_algorithm.md) populates ShapeDescriptor from a PyTorch tensor and an op hint.

---

## Summary Table: Three Layers for Common Shapes

| Logical Shape | Tile Shape | Padded Shape | Tile Grid | Total Tiles | Page Size (bf16) | CB Total |
|--------------|-----------|-------------|-----------|-------------|-----------------|---------|
| `[1, 37]` | (32, 32) | `[32, 64]` | `[1, 2]` | 2 | 2,048 B | 4,096 B |
| `[1, 37]` | (1, 32) | `[1, 64]` | `[1, 2]` | 2 | 64 B | 128 B |
| `[1, 768]` | (1, 32) | `[1, 768]` | `[1, 24]` | 24 | 64 B | 1,536 B |
| `[1, 4096]` | (32, 32) | `[32, 4096]` | `[1, 128]` | 128 | 2,048 B | 256 KB |
| `[1, 7168]` | (32, 32) | `[32, 7168]` | `[1, 224]` | 224 | 2,048 B | 448 KB |
| `[7168, 2048]` | (32, 32) | `[7168, 2048]` | `[224, 64]` | 14,336 | 1,088 B* | 14.9 MB |
| `[4096, 11008]` | (32, 32) | `[4096, 11008]` | `[128, 344]` | 44,032 | 1,088 B* | 45.7 MB |

\* bfloat8_b format. Note: "CB Total" shows the theoretical total if all tiles were buffered simultaneously. In practice, CB `num_pages` is much smaller (1-4 for streaming, K tiles for matmul accumulation) and the actual L1 footprint is a fraction of this total. CB sizing strategy is the subject of Chapter 7.

---

## Key Takeaways

- The three shape layers -- logical (developer's shape), padded (tile-aligned), and tile grid (tile counts) -- are the fundamental vocabulary of tile decomposition. They are connected by the tile shape through a deterministic derivation chain: `logical -> pad -> grid -> total_tiles`.
- The **storage tile vs. compute tile** distinction matters for normalization ops: data is stored as contiguous 1x32 rows in L1 but reinterpreted as 16x32 or 32x32 tiles by `interpret_tile()` for the compute kernel. ShapeDescriptor tracks the compute tile because it determines page_size and grid shape.
- All three layers must be tracked simultaneously because each serves a different consumer: logical for output slicing, padded for hardware tile processing, tile grid for CB allocation and CT arg computation.
- ShapeDescriptor (from Chapter 3) encapsulates all three layers plus tile shape, page size, and padding metadata in a single frozen dataclass with validated invariants. It replaces the scattered tracking across `TensorPort`, `CBHandle`, `_tile_page_size()`, and inline calculations.
- Block floating-point formats require special page_size computation: bfloat8_b = 1,088 bytes/tile (not 1,024) and bfloat4_b = 576 bytes/tile (not 512). The 64-byte shared exponent overhead per tile accumulates and causes L1 misalignment if ignored.
- Real model dimensions from both LLaMA-7B (4096, 11008) and DeepSeek V3 (7168, 2048) demonstrate that production shapes are typically tile-aligned, but DeepSeek V3's larger hidden dimension consumes nearly 30% of per-core L1 for a single activation -- making automated L1 budget management essential.

## Source Files

- `blaze/cb_engine.py` -- `DEFAULT_TILE_SHAPE = (32, 32)`, `DEFAULT_DTYPE = "bfloat16"`, `DTYPE_BYTES`, `_tile_page_size()`
- `blaze/graph.py` -- `TensorPort` (carries `dtype` and `tile_shape`, but not logical/padded/grid shapes)
- `blaze/utils.py` -- `round_up()`, `interpret_tile()`, `interpret_tile_padded()`, `UNPACKED_DTYPE_BYTES`
- `blaze/cb_handle.py` -- `CBHandle` (carries `num_pages`, `page_size`, `tile_desc` but not logical shape)
- `blaze/ops/sdpa/op.py` -- `_TILE_SIZES` mapping (format to per-tile byte sizes for BFP formats)

---

← Previous: [Chapter 3: The TensorAdapter Architecture](../ch03_tensor_adapter/) | Next: [Chapter 5: Transparent Padding Management](../ch05_padding/) →
