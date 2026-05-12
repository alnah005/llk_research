# Automatic Page Count and Page Size Determination

Every circular buffer has two sizing parameters that together determine its L1 footprint: `page_size` (bytes per tile) and `num_pages` (how many tiles the buffer holds simultaneously). Page size is determined by tile geometry and data format -- it is fundamentally a property of the data. Page count, by contrast, is a scheduling decision: how much data to buffer in L1 at once, trading off memory pressure against data movement latency. This section designs the algorithm for automatically computing both values, replacing the manual calculations that currently appear in every op's `emit()` method with a systematic approach that considers the op's computational pattern, the data format ([Chapter 6](../ch06_data_formats/)), the tile decomposition ([Chapter 4](../ch04_tile_decomposition/)), and the L1 budget (File 01 of this chapter).

> *Design Principle P3 (Validate at Each Lowering Step, Not Just at the End):* Every CB sizing decision must be validated against the L1 budget immediately. Developers should never need to manually compute `num_pages = k_num_tiles` for a matmul or `num_pages = compute_n` for an RMSNorm — these values are derivable from tensor shapes, op semantics, and available L1, and each derivation is checked before proceeding to the next.

**What you will learn:**

- The page size computation pipeline: from `DTYPE_BYTES` to `_tile_page_size()` to `tile.get_tile_size()`, and when each path is correct
- How `num_pages` is determined for different op categories: streaming (1-2 pages), matmul (K-dimension tiles), reduction (accumulation tiles), and DRAM streaming (subblock_k x buffer_count)
- The five-step automatic sizing algorithm: minimum computation, L1 budget check, reduction, multi-pass fallback, and double-buffering optimization
- Concrete worked examples for DeepSeek V3 matmul, LLaMA RMSNorm, and SDPA attention patterns
- How the `compute_subblock_w()` function from `utils.py` constrains output tile counts based on dest register capacity
- The proposed `AutoSizer` class that integrates with `FusedProgram` to replace manual sizing

---

## 1. Page Size Determination

Page size is the simpler of the two parameters -- it depends only on tile geometry and data format. However, there are three different computation paths in the current codebase, and choosing the wrong one produces incorrect values for block-floating-point formats.

### Path 1: Simple Multiplication (DTYPE_BYTES)

```python
# EXISTING -- from blaze/cb_engine.py (lines 73-79)
def _tile_page_size(dtype: Optional[str], tile_shape: Optional[tuple[int, int]]) -> int:
    """Compute page size in bytes from dtype and tile shape."""
    dt = dtype or DEFAULT_DTYPE
    ts = tile_shape or DEFAULT_TILE_SHAPE
    if dt not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dt}'. Supported: {list(DTYPE_BYTES.keys())}")
    return DTYPE_BYTES[dt] * ts[0] * ts[1]
```

This path uses `DTYPE_BYTES` to look up per-element byte widths:

```python
# EXISTING -- from blaze/cb_engine.py (lines 20-27)
DTYPE_BYTES: dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
}
```

For standard formats, this produces correct results:

```text
page_size("bfloat16", (32, 32)) = 2 * 32 * 32 = 2,048 bytes
page_size("float32",  (32, 32)) = 4 * 32 * 32 = 4,096 bytes
page_size("bfloat16", (16, 32)) = 2 * 16 * 32 = 1,024 bytes
page_size("bfloat16", (1,  32)) = 2 * 1  * 32 =    64 bytes
```

> **Warning:** This path does **not** support `bfloat8_b` or `bfloat4_b`. These formats are absent from `DTYPE_BYTES` and the function raises `ValueError`. As noted in [Chapter 6, File 01](../ch06_data_formats/01_data_format_landscape.md), BFP formats have a shared-exponent overhead that makes the per-tile size non-uniform across element widths.

### Path 2: Hardware Tile Size (tile.get_tile_size)

```python
# EXISTING -- from blaze/fused_program.py (lines 41-49)
@staticmethod
def from_tensor(tensor) -> "TileInfo":
    tile = tensor.get_tile()
    data_format = tensor.dtype
    return TileInfo(
        tile=tile,
        data_format=data_format,
        size=tile.get_tile_size(data_format),
        desc=ttnn.TileDescriptor(tile),
    )
```

`tile.get_tile_size(data_format)` is the authoritative computation -- it handles all formats including BFP:

```text
tile.get_tile_size(bfloat16, (32, 32)) = 2,048 bytes
tile.get_tile_size(bfloat8_b, (32, 32)) = 1,088 bytes
tile.get_tile_size(bfloat4_b, (32, 32)) =   576 bytes
tile.get_tile_size(float32,   (32, 32)) = 4,096 bytes
```

These values match the `_TILE_SIZES` constant in SDPA:

```python
# EXISTING -- from blaze/ops/sdpa/op.py (lines 38-43)
_TILE_SIZES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
    ttnn.float32: 4096,
}
```

### Path 3: Unpacked Element Size (UNPACKED_DTYPE_BYTES)

```python
# EXISTING -- from blaze/utils.py (lines 65-70)
UNPACKED_DTYPE_BYTES: dict = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.int32: 4,
}
```

This dictionary represents the **compute-side** element size after BFP formats are unpacked to bfloat16 in the SFPU registers. It is used for buffer sizing in the compute pipeline but **not** for L1 page size. BFP formats are absent because they unpack to bfloat16 before compute.

### The Automatic Page Size Rule

The proposed automatic system should always use `tile.get_tile_size(data_format)` as the authoritative path. When a `ttnn.Tile` object is not available (e.g., during compile-time analysis before tensor creation), the system falls back to the `_TILE_SIZES` lookup table:

```python
# PROPOSED -- authoritative page size computation
def compute_page_size(
    tile_shape: tuple[int, int],
    data_format: ttnn.DataType,
) -> int:
    """Compute page size from tile shape and data format.

    Uses tile.get_tile_size() for all formats including BFP.
    Falls back to TILE_SIZE_TABLE for compile-time analysis.
    """
    TILE_SIZE_TABLE = {
        ttnn.bfloat16: lambda h, w: h * w * 2,
        ttnn.float32:  lambda h, w: h * w * 4,
        ttnn.bfloat8_b: lambda h, w: 1088 if (h, w) == (32, 32) else h * w,
        ttnn.bfloat4_b: lambda h, w: 576  if (h, w) == (32, 32) else h * w,
        ttnn.uint8:    lambda h, w: h * w * 1,
        ttnn.uint16:   lambda h, w: h * w * 2,
        ttnn.uint32:   lambda h, w: h * w * 4,
    }
    try:
        tile = ttnn.Tile(list(tile_shape))
        return tile.get_tile_size(data_format)
    except Exception:
        compute = TILE_SIZE_TABLE.get(data_format)
        if compute is None:
            raise ValueError(f"No page size rule for format {data_format}")
        return compute(*tile_shape)
```

---

## 2. Page Count by Op Category

While page size is a property of data format and tile geometry, `num_pages` depends on the op's computational pattern. Different categories of ops have fundamentally different buffering requirements.

### Category A: Streaming Ops (1-2 Pages)

Ops that process tiles one at a time need only 1 page (or 2 for double-buffering). Examples:

- **EltwiseMul:** reads one tile from each input, produces one output tile
- **Copy/TileCopy:** moves one tile from source to destination
- **Mcast destination:** receives tiles one at a time from the sender

For these ops, `num_pages = 1` (single-buffered) or `num_pages = 2` (double-buffered to hide NOC latency).

```python
# PROPOSED -- streaming op page count
def streaming_num_pages(double_buffer: bool = True) -> int:
    """Page count for streaming ops: 1 or 2."""
    return 2 if double_buffer else 1
```

### Category B: Matmul Input (K-Dimension Tiles)

For matrix multiply, the activation input CB must hold all tiles along the K (accumulation) dimension, because the matmul kernel loops over K tiles in the inner loop:

```python
# EXISTING -- from blaze/ops/matmul/op.py (line 48)
k_num_tiles = in0_handle.num_pages
```

The matmul computes `output[i, j] = sum_k(in0[i, k] * in1[k, j])`. Each output tile requires reading **all** K tiles of the activation. If `K = 4096` and `tile_w = 32`:

```text
k_num_tiles = K / tile_w = 4096 / 32 = 128 tiles
```

The activation CB needs `num_pages = 128` to hold the entire K row. At bfloat16:

```text
activation_cb = 128 * 2,048 = 262,144 bytes (256 KB)
```

This is 20% of the L1 budget -- significant, but it is the minimum required for correctness. The matmul kernel reads each activation tile once and accumulates partial products in the dest registers.

### Category C: Matmul Output (N-Dimension Tiles Per Core)

The matmul output CB holds one complete output row per core:

```python
# EXISTING -- from blaze/ops/matmul/op.py (lines 58-63)
if k_num_tiles <= 0 or in1_cb.num_pages % k_num_tiles:
    raise ValueError(
        f"Matmul weight pages ({in1_cb.num_pages}) must be a "
        f"positive multiple of K tiles ({k_num_tiles})."
    )
out_w_per_core = in1_cb.num_pages // k_num_tiles
```

For a DeepSeek V3 MoE expert with `N_total = 18,432` sharded across 7 cores:

```text
N_per_core = 18432 / 7 = 2632 -> rounded to 2624 (82 tiles)
k_num_tiles = 128
weight tiles per core = 82 * 128 = 10,496 tiles (direct-address in DRAM)
out_w_per_core = 10,496 / 128 = 82 tiles
output_cb = 82 * 2,048 = 167,936 bytes (164 KB)
```

### Category D: Reduction/Accumulation (All Tiles)

Reduction ops like RMSNorm accumulate across all tiles of a dimension. The accumulation CB holds partial results:

```python
# EXISTING -- from blaze/ops/rmsnorm/op.py (lines 121-123)
compute_n = num_tiles if num_tiles is not None else inp.num_pages
n = inp.num_pages if not isinstance(input_handle, CBHandle) else (num_tiles if num_tiles is not None else inp.num_pages)
```

For RMSNorm with hidden_dim=7,168 and 32x32 tiles:

```text
num_tiles = 7168 / 1024 = 7  (using FULL tile geometry)
rmsnorm_input_cb = 7 * 2,048 = 14,336 bytes (14 KB)
rmsnorm_gamma_cb = 7 * 2,048 = 14,336 bytes (14 KB)
rmsnorm_out_cb   = 7 * 2,048 = 14,336 bytes (14 KB)
```

For wider hidden dimensions like LLaMA 70B (hidden_dim=8,192):

```text
num_tiles = 8192 / 1024 = 8
rmsnorm total = 3 * 8 * 2,048 = 49,152 bytes (48 KB)
```

### Category E: DRAM Streaming (Subblock x Buffer Count)

DRAMStreamingMatmul reads weight tiles from DRAM in subblocks, triple-buffered to overlap NOC reads with compute:

```python
# EXISTING -- from blaze/ops/dram_streaming_matmul/op.py (lines 128-132)
if subblock_k is None:
    subblock_k = max(1, Kt // 4)
while Kt % subblock_k != 0 and subblock_k > 1:
    subblock_k -= 1

# ...
num_weights_buffers = 3
weights_cb = f.cb_scratch(
    name=...,
    num_pages=subblock_k * num_weights_buffers,
    ...
)
```

The `subblock_k = max(1, Kt // 4)` heuristic chooses a subblock that is roughly 1/4 of the K dimension. For `Kt = 128`:

```text
subblock_k = max(1, 128 // 4) = 32
weights_cb = 32 * 3 * 1,088 = 104,448 bytes (bfloat8_b)
                  or
weights_cb = 32 * 3 * 2,048 = 196,608 bytes (bfloat16)
```

This shows how data format ([Chapter 6](../ch06_data_formats/)) directly affects L1 consumption. The bfloat8_b format saves nearly 50% of the weight buffer space.

---

## 3. The Dest Register Constraint

Before computing `num_pages` for output CBs, the system must account for the dest register capacity. The `compute_subblock_w()` function in `utils.py` encodes this hardware constraint:

```python
# EXISTING -- from blaze/utils.py (lines 80-113)
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
        max_dest = 4
    else:
        max_dest = 8
    subblock_w = min(max_dest, num_tiles)
    while subblock_w > 1 and num_tiles % subblock_w != 0:
        subblock_w -= 1
    return subblock_w
```

The dest register file holds a fixed number of tiles. The `max_dest` value depends on two configuration flags:

| `dst_full_sync_en` | `fp32_dest_acc_en` | `max_dest` tiles | Dest bytes (bfloat16) |
|--------------------|--------------------|------------------|-----------------------|
| True | True | 8 | 16,384 |
| True | False | 16 | 32,768 |
| False | True | 4 | 8,192 |
| False | False | 8 | 16,384 |

`subblock_w` is the number of output tiles processed in one compute invocation. It must:
1. Divide `num_tiles` evenly (no partial subblocks)
2. Fit within `max_dest` tiles

This constraint determines how many output tiles are produced per compute cycle, which in turn affects how many pages the output CB needs to hold simultaneously.

### Interaction with Precision Profiles (Chapter 6)

The three precision profiles from [Chapter 6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md) affect dest capacity:

- **Performance profile** (`fp32_dest_acc_en=False`, `dst_full_sync_en=False`): `max_dest=8` — standard configuration
- **Balanced profile** (`fp32_dest_acc_en=False`, `dst_full_sync_en=False`): `max_dest=8` — same standard configuration as performance (both use bfloat16 accumulation)
- **Accuracy profile** (`fp32_dest_acc_en=True`, `dst_full_sync_en=True`): `max_dest=8` — fp32 tiles are twice as wide, but full sync recovers capacity

The automatic sizing system should query the active precision profile to determine `max_dest`, then pass it to `compute_subblock_w()`.

---

## 4. The Five-Step Automatic Sizing Algorithm

Given a fused op's complete list of CBs, the automatic sizing algorithm determines `num_pages` for each CB to maximize compute throughput while fitting within the L1 budget.

### Step 1: Compute Minimum num_pages Per CB

Each CB has a minimum page count determined by its role:

```python
# PROPOSED -- minimum page count rules
def minimum_num_pages(cb_role: CBRole, op_params: OpParams) -> int:
    """Compute the minimum num_pages for a CB based on its role.

    CBRole is an enum: STREAMING, ACCUMULATION, WEIGHT_BUFFER,
    OUTPUT, REDUCTION, DOUBLE_BUFFER.
    """
    match cb_role:
        case CBRole.STREAMING:
            # Streaming ops: 1 page minimum, 2 for double-buffering
            return 1

        case CBRole.ACCUMULATION:
            # Matmul activation: must hold all K tiles for correctness
            return op_params.k_num_tiles

        case CBRole.WEIGHT_BUFFER:
            # DRAM streaming: subblock_k * buffer_count
            return op_params.subblock_k * op_params.buffer_count

        case CBRole.OUTPUT:
            # Matmul output: N tiles per core
            return op_params.out_tiles_per_core

        case CBRole.REDUCTION:
            # RMSNorm, softmax: all tiles for accumulation
            return op_params.reduction_tiles

        case CBRole.DOUBLE_BUFFER:
            # Double-buffered version of another role
            return 2 * minimum_num_pages(cb_role.inner, op_params)
```

### Step 2: Compute Total L1 at Minimum Sizes

Sum the L1 consumption of all CBs at their minimum page counts:

```python
# PROPOSED -- total L1 computation
total_minimum = sum(
    cb.minimum_num_pages * cb.page_size
    for cb in all_cbs
)
```

### Step 3: Compare Against Budget

```python
# PROPOSED -- budget check
budget = l1_budget.available_bytes

if total_minimum <= budget:
    # All CBs fit at minimum sizes. Proceed to optimization.
    pass
elif total_minimum <= budget * OVERCOMMIT_WARNING_THRESHOLD:
    # Tight but possible. Warn the developer.
    warnings.warn(
        f"L1 utilization at {total_minimum/budget:.0%} with minimum sizing. "
        f"No room for double-buffering optimizations."
    )
else:
    # Minimum sizes exceed budget. Must reduce.
    pass
```

### Step 4: Reduce If Necessary

When minimum sizes exceed the budget, the algorithm has two reduction strategies:

**Strategy A: Reduce accumulation CB pages.** For matmul accumulation CBs, the "minimum" of `k_num_tiles` is a correctness requirement only if the entire K dimension is accumulated in a single pass. If the matmul kernel supports multi-pass accumulation (accumulating partial results across passes), the page count can be reduced to `k_num_tiles / num_passes`:

```python
# PROPOSED -- multi-pass reduction
def reduce_accumulation_pages(
    cb: CBDescriptor,
    target_pages: int,
) -> tuple[int, int]:
    """Reduce accumulation CB pages by introducing multi-pass tiling.

    Returns (new_num_pages, num_passes).
    """
    min_pages = cb.minimum_num_pages
    if target_pages >= min_pages:
        return min_pages, 1

    # Find the largest divisor of min_pages that is <= target_pages
    for pages in range(target_pages, 0, -1):
        if min_pages % pages == 0:
            return pages, min_pages // pages
    return 1, min_pages
```

**Strategy B: Switch to DRAM streaming.** When L1 is too constrained even for multi-pass tiling, convert the matmul from in-L1 accumulation to DRAM streaming. This replaces the large activation CB with a small streaming buffer:

```text
Before (in-L1):    activation_cb = 128 tiles * 2,048 B = 262,144 B (256 KB)
After (streaming):  activation_cb = 3 * 32 tiles * 2,048 B = 196,608 B (192 KB)
```

The streaming approach trades L1 for NOC bandwidth -- each activation tile is read from DRAM rather than from a local L1 buffer. This reduces L1 consumption but increases NOC traffic.

### Step 5: Optimize with Double-Buffering

When the L1 budget has headroom after minimum allocation, the algorithm allocates additional pages for double-buffering. Double-buffering allows NOC reads to overlap with compute: while the TRISC processes tiles from buffer A, the NCRISC prefetches the next batch into buffer B.

```python
# PROPOSED -- double-buffering optimization
def apply_double_buffering(
    cbs: list[CBDescriptor],
    budget_remaining: int,
) -> list[CBDescriptor]:
    """Double-buffer CBs where L1 budget permits, prioritizing
    high-traffic CBs (weight buffers > accumulation > streaming).
    """
    # Sort by impact: double-buffering a high-traffic CB hides more latency
    priority_order = sorted(cbs, key=lambda cb: cb.traffic_volume, reverse=True)

    for cb in priority_order:
        doubling_cost = cb.num_pages * cb.page_size  # cost to double
        if doubling_cost <= budget_remaining:
            cb.num_pages *= 2
            budget_remaining -= doubling_cost

    return cbs
```

The priority order reflects the performance impact of double-buffering:
1. **Weight buffers** (highest impact -- DRAM reads are the primary latency source)
2. **Accumulation inputs** (medium impact -- NOC reads from sharded activations)
3. **Streaming intermediates** (lowest impact -- typically intra-core transfers)

The double-buffering threshold for KT_DIM follows the general rule: `double_buffered_pages = 2 * KT_DIM`, where `KT_DIM` is the number of tiles in the K blocking dimension. This ensures one K-block can be consumed while the next is prefetched.

---

## 5. Worked Examples

### Example 1: DeepSeek V3 Matmul (Direct-Address Weights)

```text
Parameters:
  M = 1 (decode mode, single token)
  K = 4,096 (activation dimension for attention Q/K/V projection)
  N = 1,536 per core (7 cores, N_total = 10,752)
  activation format: bfloat16, tile (1, 32)
  weight format: bfloat8_b, tile (32, 32) -- DIRECT_ADDRESS in DRAM
  precision: performance (fp32_dest_acc_en=False)

Page sizes:
  activation page: 1 * 32 * 2 = 64 bytes (1x32 tile, bfloat16)
  output page: 1 * 32 * 2 = 64 bytes (1x32 tile, bfloat16)

Minimum num_pages:
  activation_cb: k_num_tiles = 4096 / 32 = 128 pages
  output_cb:     out_w_per_core = N_per_core / tile_w = 1536 / 32 = 48 pages

L1 consumption:
  activation_cb: 128 * 64 = 8,192 bytes (8 KB)
  output_cb:     48 * 64 = 3,072 bytes (3 KB)
  Total: 11,264 bytes (11 KB)

Budget check: 11 KB << 1,291 KB available. Abundant headroom.

Double-buffering:
  activation_cb: 256 * 64 = 16,384 bytes (16 KB) -- doubles to hide NOC reads
  Total with double-buffering: 19 KB (1.5% utilization)
```

Note how the 1x32 tile geometry (decode mode with a single-token activation) makes this matmul extremely L1-efficient. The weight data stays in DRAM and is accessed via direct-address NOC reads -- it consumes zero L1 pages.

### Example 2: LLaMA 70B RMSNorm

```text
Parameters:
  hidden_dim = 8,192
  format: bfloat16
  tile: (32, 32) via interpret_tile(8192)

Page size: 32 * 32 * 2 = 2,048 bytes

interpret_tile(8192):
  8192 // 32 = 256 rows
  256 % 32 = 0 -> use FULL tile (32, 32)
  num_tiles = 8192 / 1024 = 8

Minimum num_pages:
  input_cb:  8 pages (all tiles for reduction)
  gamma_cb:  8 pages (all tiles for element-wise multiply)
  output_cb: 8 pages (accumulation result)

L1 consumption:
  Total: 3 * 8 * 2,048 = 49,152 bytes (48 KB)

Budget check: 48 KB << 1,291 KB. Comfortable.
```

### Example 3: DeepSeek V3 DRAM Streaming Matmul (Expert Down Projection)

```text
Parameters:
  K = 18,432 (intermediate dimension, divided by core count)
  N_per_core = 1 output tile column (routed expert)
  weight format: bfloat8_b
  activation format: bfloat16, tile (1, 32)
  Kt = K / tile_w = 18432 / 32 = 576 activation tiles

Subblock derivation:
  subblock_k = max(1, 576 // 4) = 144
  Check: 576 % 144 = 0. Valid.

  But 144 * 3 * 1,088 = 470,016 bytes (~459 KB) -- 36% of L1!
  Too aggressive. Try smaller:
  subblock_k = max(1, 576 // 8) = 72
  72 * 3 * 1,088 = 235,008 bytes (~230 KB) -- 18% of L1.

The automatic system would start at subblock_k = 576 // 4 = 144 and
progressively reduce until the total fits the remaining budget:

  subblock_k candidates: 144, 72, 48, 36, 24, 18, 16, 12, 9, 8, 6, 4, 3, 2, 1
  (all divisors of 576 that are <= 144)

  Selection: the largest subblock_k where
  subblock_k * 3 * weight_tile_size <= remaining_l1_budget
```

### Example 4: SDPA Attention (Multiple CB Categories)

```text
Parameters (DeepSeek V3 MLA, decode mode):
  num_q_groups = 4
  dhead = 128 (head dimension)
  seq_len = 2,048 (KV cache length for chunked attention)
  Q format: bfloat16, K/V format: bfloat8_b

SDPA CB requirements:
  Q CB:       4 tiles (128 / 32 tiles per head)   * 2,048 = 8,192 B
  K CB:       varies by chunk_size                 * 1,088 = varies
  V CB:       same as K CB                         * 1,088 = varies
  QK CB:      softmax intermediate, chunk_size tiles * 2,048 = varies
  Scale CB:   1 tile (scalar broadcast)            * 2,048 = 2,048 B
  Output CB:  4 tiles (same as Q)                  * 2,048 = 8,192 B
  Partial CB: reduction intermediate               * 4,096 = varies (float32)

The chunk_size determines how many KV tiles are processed per chunk:
  chunk_size = min(64, seq_len_tiles)  -- hardware tree reduction limit
  K_cb = chunk_size * 1,088

For seq_len = 2,048 with 32x32 tiles: seq_tiles = 64
  K_cb = 64 * 1,088 = 69,632 bytes (68 KB)
  V_cb = 64 * 1,088 = 69,632 bytes (68 KB)
  QK_cb = 64 * 2,048 = 131,072 bytes (128 KB)

Total SDPA per core: ~282 KB (22% of L1)
```

---

## 6. The Proposed AutoSizer

Integrating all the above into a single system:

```python
# PROPOSED -- automatic CB sizing system
@dataclass
class CBSizingRequest:
    """A request for CB allocation with sizing constraints."""
    name: str
    role: CBRole
    tile_shape: tuple[int, int]
    data_format: ttnn.DataType
    page_size: int                # computed from tile_shape + data_format
    minimum_pages: int            # from op semantics
    preferred_pages: int          # with double-buffering
    priority: int                 # for budget allocation (0 = highest)
    reducible: bool               # can num_pages be reduced via multi-pass?

    @property
    def minimum_bytes(self) -> int:
        return self.minimum_pages * self.page_size

    @property
    def preferred_bytes(self) -> int:
        return self.preferred_pages * self.page_size


class AutoSizer:
    """Determines num_pages for all CBs in a fused op, respecting L1 budget."""

    def __init__(self, l1_budget: L1Budget):
        self._budget = l1_budget
        self._requests: list[CBSizingRequest] = []

    def request(self, req: CBSizingRequest) -> None:
        """Register a CB sizing request."""
        self._requests.append(req)

    def solve(self) -> dict[str, int]:
        """Solve for num_pages per CB.

        Algorithm:
        1. Start with minimum_pages for all CBs
        2. Check total against budget
        3. If over budget, reduce reducible CBs
        4. If still over, raise L1BudgetExceeded
        5. If under budget, upgrade to preferred_pages by priority
        """
        available = self._budget.available_bytes

        # Step 1: minimum allocation
        allocation = {req.name: req.minimum_pages for req in self._requests}
        total = sum(req.minimum_bytes for req in self._requests)

        # Step 2: budget check
        if total > available:
            # Step 3: reduce
            reducible = [r for r in self._requests if r.reducible]
            reducible.sort(key=lambda r: r.minimum_bytes, reverse=True)

            for req in reducible:
                excess = total - available
                if excess <= 0:
                    break
                # Reduce this CB's pages
                current_pages = allocation[req.name]
                target_bytes = req.minimum_bytes - excess
                target_pages = max(1, target_bytes // req.page_size)
                # Find nearest valid divisor
                for p in range(target_pages, 0, -1):
                    if current_pages % p == 0:
                        savings = (current_pages - p) * req.page_size
                        allocation[req.name] = p
                        total -= savings
                        break

        # Step 4: still over budget?
        if total > available:
            raise L1BudgetExceeded(
                f"Cannot fit minimum CB allocations in L1. "
                f"Need {total:,} bytes, have {available:,} bytes.\n"
                f"Allocations:\n" +
                "\n".join(f"  {r.name}: {allocation[r.name]} pages x "
                         f"{r.page_size} B = {allocation[r.name] * r.page_size:,} B"
                         for r in self._requests)
            )

        # Step 5: upgrade to preferred where budget permits
        remaining = available - total
        upgradeable = sorted(self._requests, key=lambda r: r.priority)
        for req in upgradeable:
            upgrade_cost = req.preferred_bytes - allocation[req.name] * req.page_size
            if upgrade_cost > 0 and upgrade_cost <= remaining:
                allocation[req.name] = req.preferred_pages
                remaining -= upgrade_cost

        return allocation
```

### Integration with FusedProgram

The `AutoSizer` would be instantiated by `FusedProgram.__init__()` and consulted by each `cb_scratch()` call:

```python
# PROPOSED -- FusedProgram integration
class FusedProgram:
    def __init__(self, kernel, device, *, auto_size: bool = False, **kwargs):
        # ... existing init ...
        if auto_size:
            self._auto_sizer = AutoSizer(L1Budget.for_device(device))
        else:
            self._auto_sizer = None

    def cb_scratch(self, name, *, num_pages, core_ranges, data_format,
                   tile, page_size, **kwargs):
        if self._auto_sizer is not None:
            # Validate against budget before allocating
            self._auto_sizer._budget.allocate(name, num_pages, page_size)
        # ... existing allocation logic ...
```

This preserves backward compatibility (P4 -- defaults with overrides): existing ops work unchanged with `auto_size=False`. New ops can opt in to automatic sizing by passing `auto_size=True`, and the system validates every allocation against the L1 budget.

---

## 7. Page Count and the ShapeDescriptor Pipeline

The automatic page count computation connects to the shape pipeline from [Chapter 3](../ch03_tensor_adapter/) and [Chapter 4](../ch04_tile_decomposition/):

```text
ShapeDescriptor Pipeline for CB Sizing:

PyTorch tensor [1, 4096]
        |
        v (Chapter 4: tile decomposition)
tile_grid_shape = (1, 128)   # 128 tiles of 32 elements
padded_shape = (32, 4096)    # padded to tile boundary
        |
        v (Chapter 6: format selection)
data_format = bfloat16
page_size = 2,048 bytes
        |
        v (This chapter: page count)
op_role = ACCUMULATION (matmul input)
minimum_pages = 128
        |
        v (This chapter: L1 budget)
budget_check: 128 * 2,048 = 262,144 B < 1,322,240 B available
final_pages = 128 (or 256 with double-buffering)
```

The `TypedHandle` returned by `TensorAdapter.adapt()` ([Chapter 3, File 02](../ch03_tensor_adapter/)) would carry the computed `num_pages`:

```python
# PROPOSED -- TypedHandle with auto-sized page count
@dataclass(eq=False)
class TypedHandle:
    cb_handle: CBHandle
    shape_desc: ShapeDescriptor
    sizing_info: CBSizingInfo  # records how num_pages was determined

@dataclass
class CBSizingInfo:
    role: CBRole
    minimum_pages: int
    allocated_pages: int
    double_buffered: bool
    l1_bytes: int
    budget_utilization: float
```

This provides full transparency into the sizing decision: a developer inspecting a `TypedHandle` can see exactly why `num_pages` was set to a particular value, how much L1 it consumes, and whether double-buffering is active.

---

## Key Takeaways

- **Page size** is determined by tile geometry and data format via `tile.get_tile_size(data_format)`. The simple `DTYPE_BYTES * h * w` formula is incorrect for BFP formats (`bfloat8_b` = 1,088 bytes/tile, `bfloat4_b` = 576 bytes/tile) and the system must always use the authoritative path.
- **Page count** depends on op semantics: streaming ops need 1-2 pages, matmul accumulation needs all K tiles, DRAM streaming needs `subblock_k * buffer_count`, and reduction needs all tiles along the reduced dimension. These rules are currently hard-coded in each op's `emit()` method.
- The **dest register capacity** (4-16 tiles depending on `fp32_dest_acc_en` and `dst_full_sync_en`) constrains `subblock_w`, which affects output CB sizing. The `compute_subblock_w()` function in `utils.py` encodes this hardware constraint.
- The **five-step sizing algorithm** (minimum computation, budget check, reduction, multi-pass fallback, double-buffering) systematizes the manual decisions developers currently make by trial and error.
- Double-buffering (allocating `2x` pages, specifically `2 * KT_DIM` for accumulation CBs) is the primary optimization for hiding NOC latency behind compute. The automatic system applies it in priority order (weight buffers first, streaming intermediates last) when the L1 budget permits.
- The `AutoSizer` integrates with `FusedProgram` as a compile-time budget tracker that validates every allocation and provides actionable error messages (P6) when the budget is exceeded.

## Source Files

- `blaze/cb_engine.py` -- `DTYPE_BYTES` (lines 20-27), `_tile_page_size()` (lines 73-79), `CBAssignment` (lines 122-138)
- `blaze/fused_program.py` -- `TileInfo.from_tensor()` (lines 41-49, authoritative page size), `FusedProgram.cb_scratch()` (lines 1540-1593), `FusedProgram.cb_from_tensor()` (lines 1050-1065)
- `blaze/utils.py` -- `compute_subblock_w()` (lines 80-113, dest register capacity), `interpret_tile()` (lines 121-133, tile geometry selection), `UNPACKED_DTYPE_BYTES` (lines 65-70)
- `blaze/ops/matmul/op.py` -- `k_num_tiles` derivation (line 48), `out_w_per_core` derivation (line 63), output CB sizing (lines 67-74)
- `blaze/ops/rmsnorm/op.py` -- `compute_n` derivation (line 121), output CB sizing (lines 132-139)
- `blaze/ops/dram_streaming_matmul/op.py` -- `subblock_k` heuristic (lines 128-132), triple-buffered weights CB (lines 161-172), `_get_max_page_size_and_num_pages()` (lines 16-31)
- `blaze/ops/sdpa/op.py` -- `_TILE_SIZES` (lines 38-43), `_ELEMENT_SIZES` (lines 45-50)

---

< Previous: [File 01 -- L1 Memory Model](./01_l1_memory_model.md) | Next: [File 03 -- Blocking Strategy Selection](./03_blocking_strategy_selection.md) >

← [Chapter 6 -- Data Format Selection and Automatic Negotiation](../ch06_data_formats/) | [Chapter 8 -- Broadcasting and Multi-Dimensional Shape Alignment](../ch08_broadcasting/) →
