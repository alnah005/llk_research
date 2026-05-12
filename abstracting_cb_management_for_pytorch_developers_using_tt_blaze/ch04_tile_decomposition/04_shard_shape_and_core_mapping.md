# Shard Shape and Core Mapping

The previous three files defined the tile decomposition algorithm and shape propagation from a single-core perspective. But Tenstorrent hardware distributes work across a grid of Tensix cores, and each core processes only a **shard** of the full tensor. This section defines how the global tile grid is partitioned into per-core shards, how `OverlappedView`, `GridConfig`, and `pick_matmul_cores()` from the existing Blaze codebase inform this mapping, and how TensorAdapter automates the computation of `shard_shape`, per-core `num_pages`, and `page_size` from a logical shape and a core grid.

> *Design Principle P1 (Separate Computation Intent from Execution Strategy):* The developer declares tensor shapes; the shard computation algorithm determines per-core partitioning, CB sizing, and CT arg derivation automatically. The six manual steps (choose cores, compute shard, configure shard spec, allocate CB, extract core_ranges, compute CT args) are replaced by a single `compute_shard()` call.

**What you will learn:**

- How the global tile grid from File 01 is partitioned across Tensix cores, and why per-core shard shape determines CB allocation
- The FIFO vs. DIRECT_ADDRESS CB mode distinction and its impact on L1 budgeting
- The relationship between `OverlappedView.shard_shape`, `OverlappedView.core_ranges`, and CB `num_pages` in `blaze/fused_program.py`
- How `GridConfig` and `pick_matmul_cores()` select core grids for different op types
- The prime-number factorization problem: why `344 = 2^3 x 43` limits core count options for LLaMA-7B, and why DeepSeek V3's `64 = 2^6` is more amenable
- The `compute_subblock_w()` function: how prime values of `out_w_per_core` force subblock_w=1 and reduce compute efficiency
- The DRAM streaming fallback: when weight shards exceed L1, stream K-slices from DRAM
- Edge cases: uneven tile distribution, single-core execution, sender-only vs. compute cores

---

## From Global Tile Grid to Per-Core Shard

### The Problem

Files 01-02 compute a **global** tile grid shape from a logical tensor shape. For example, a weight matrix `[4096, 11008]` with (32, 32) tiles in bfloat8_b produces a tile grid `[128, 344]` with 44,032 total tiles at 1,088 bytes per tile -- a total of ~45.7 MB. No single Tensix core's ~1.5 MB L1 can hold this. The tensor must be **sharded** across multiple cores, with each core holding a subset of the tiles.

The shard shape determines the CB allocation on each core:

```
Global tile grid:  [128, 344]       (44,032 tiles total)
Core grid:         8 cores (width sharding)
Per-core shard:    [128, 43] tiles  (344 / 8 = 43 tiles per core)
Per-core bytes:    128 * 43 * 1,088 = 5,988,352 bytes (~5.7 MB)
```

> **Warning:** 5.7 MB per core exceeds the ~1.5 MB L1 limit. Whether this is a problem depends on the CB mode (see next section). For DIRECT_ADDRESS CBs backed by pre-placed tensors in L1, the shard must physically fit. For DRAM-streaming matmul, the CB uses a small FIFO and streams tiles from DRAM, so the full shard need not be L1-resident simultaneously.

### FIFO vs. DIRECT_ADDRESS CB Modes

This distinction is critical for understanding shard-to-L1 mapping:

| CB Mode | num_pages Meaning | L1 Impact | When Used |
|---------|-------------------|-----------|-----------|
| **FIFO** | How many tiles the ring buffer holds simultaneously | New L1 allocation: `num_pages * page_size` | Streaming activations, matmul outputs, elementwise I/O |
| **DIRECT_ADDRESS** | Total tiles in the pre-sharded tensor | No new allocation: CB points to existing L1 buffer | Weight tensors already placed in L1 by the host |

For a weight CB in DIRECT_ADDRESS mode, `num_pages = 5,504` does not mean the CB allocates 5,504 pages of new L1 -- it means the CB can address 5,504 pages in an existing L1 buffer. The L1 occupancy is determined by the backing tensor placement, not the CB allocation.

For a FIFO CB (e.g., activation input to matmul), `num_pages` determines the ring buffer size. With `num_pages = 128` and `page_size = 2,048`, the FIFO allocates `128 * 2,048 = 262,144 bytes = 256 KB` of new L1.

> **Warning:** Conflating FIFO and DIRECT_ADDRESS semantics causes either L1 overflow (treating all CBs as FIFO when some are DIRECT_ADDRESS) or stalls (treating a streaming CB as DIRECT_ADDRESS when it needs a real ring buffer).

### Sharding Strategies

| Strategy | Sharding Axis | Use Case | Per-Core Shard |
|----------|--------------|----------|---------------|
| **Column sharding** | Last dim (width) | Weight matrices distributed by output columns | `[all_rows, cols_per_core]` |
| **Row sharding** | Second-to-last dim (height) | Activation batches distributed by rows | `[rows_per_core, all_cols]` |
| **Block sharding** | Both dimensions | Large matrices on 2D core grids | `[rows_per_core, cols_per_core]` |

For the gated MLP example from File 03:

```
Gate/up weights [4096, 11008]:
  Column-sharded across 8 cores -> shard [4096, 1376] (11008/8 = 1376)
  In tiles: [128, 43] per core

Activation [1, 4096]:
  Multicast to all cores (no sharding) -> shard [1, 4096] (full tensor)
  In tiles: [1, 128] per core

Down weights [11008, 4096]:
  Column-sharded across 8 cores -> shard [11008, 512] (4096/8 = 512)
  In tiles: [344, 16] per core
```

---

## OverlappedView in blaze/fused_program.py

### What OverlappedView Represents

When `FusedProgram.cb_from_tensor()` is called with a pre-sharded tensor, the method creates an `OverlappedView` that records the per-core shard geometry:

```python
# EXISTING -- OverlappedView concept (from blaze/fused_program.py)
@dataclass
class OverlappedView:
    """A view into a shared L1 tensor buffer.

    Multiple CBHandles can reference the same physical L1 allocation
    via different OverlappedViews with different byte_offsets.
    """
    shard_shape: tuple[int, int]     # Per-core dimensions in ELEMENTS (not tiles)
    core_ranges: ttnn.CoreRangeSet   # Which cores hold shards
    backing_tensor: ttnn.Tensor      # The on-device tensor
    data_format: object              # ttnn.DataType
    tile: object                     # Tile geometry
```

The `shard_shape` in `OverlappedView` is in **elements** (not tiles). For a weight shard of `[4096, 1376]` elements with (32, 32) tiles:

```
shard_shape (elements):  (4096, 1376)
shard_shape (tiles):     (4096/32, 1376/32) = (128, 43)
num_pages (for CB):      128 * 43 = 5,504 tiles
page_size:               1,088 bytes (bfloat8_b)
```

> **Warning:** `OverlappedView.shard_shape` is in elements, not tiles. A shard_shape of `(4096, 1376)` with (32, 32) tiles means 128 x 43 = 5,504 tiles. If you mistakenly treat shard_shape as tile counts and compute `num_pages = 4096 * 1376 = 5,636,096`, the CB allocation requests 11 GB of L1 per core. The element-to-tile conversion must always be applied.

### How cb_from_tensor() Creates DIRECT_ADDRESS CBHandles

```python
# EXISTING -- simplified logic from FusedProgram.cb_from_tensor()
def cb_from_tensor(self, tensor_or_view):
    if isinstance(tensor_or_view, OverlappedView):
        view = tensor_or_view
        shard_h, shard_w = view.shard_shape
        tile_h, tile_w = view.tile.tile_shape
        num_pages = (shard_h // tile_h) * (shard_w // tile_w)

        return CBHandle(
            num_pages=num_pages,
            page_size=view.tile.get_tile_size(view.data_format),
            core_ranges=view.core_ranges,
            access_mode=CBAccessMode.DIRECT_ADDRESS,
            backing_tensor=view.backing_tensor,
        )
```

The critical chain: **shard_shape (elements) -> divide by tile dims -> num_pages for CBHandle**.

---

## GridConfig and pick_matmul_cores() in blaze/utils.py

### GridConfig: The Core Grid Planner

`GridConfig` organizes available Tensix cores into functional groups:

```python
# EXISTING -- GridConfig usage (from blaze/_gated_mlp.py)
gc = grid_config or GridConfig.default()
sender = gc.sender_core
matmul_cores = gc.get_matmul_cores()
a_cores, b_cores = gc.build_ab_grids()
```

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_all_cores()` | `CoreRangeSet` | All available Tensix cores |
| `get_matmul_cores()` | `CoreRangeSet` | Subset assigned to matmul compute |
| `build_ab_grids()` | `(CoreRangeSet, CoreRangeSet)` | Disjoint subsets for dual-matmul (gate + up) |
| `sender_core` | `CoreCoord` | Single core for multicast sends |

### How Core Count Determines Shard Shape

The number of matmul cores determines how the weight tensor is sharded:

```
Gate/up weights [4096, 11008]:
  a_cores = 4 cores -> gate shard [4096, 2752] (11008/4 = 2752)
  b_cores = 4 cores -> up shard   [4096, 2752] (11008/4 = 2752)
  Total: 8 cores for dual matmul
```

### pick_matmul_cores(): Automatic Core Selection

`pick_matmul_cores()` selects the core grid based on the tensor shapes and available hardware:

```python
# EXISTING -- pick_matmul_cores() concept (from blaze/utils.py)
def pick_matmul_cores(
    M: int, K: int, N: int,
    device_grid: tuple[int, int],
) -> CoreRangeSet:
    """Select an optimal core grid for matmul [M, K] x [K, N].

    Balances parallelism, L1 budget, and NOC traffic.
    """
    ...
```

| Factor | Impact on Core Count |
|--------|---------------------|
| Output width N | More cores for larger N |
| L1 budget | Fewer cores for large K |
| Minimum shard width | At least 1 tile per core |
| Device grid size | Upper bound on core count |

---

## The Prime-Number Factorization Problem

### Why 344 = 2^3 x 43 Creates Difficulties

LLaMA-7B's intermediate dimension is 11008, producing `11008 / 32 = 344` column tiles. The factorization of 344 is:

```
344 = 2^3 * 43

Divisors: 1, 2, 4, 8, 43, 86, 172, 344
```

This means the only valid core counts that produce even column sharding are: 1, 2, 4, 8, 43, 86, 172, or 344. Core counts like 16, 32, or 64 (the grid-friendly numbers on an 8x8 device) do **not** divide 344 evenly:

```
344 % 16 = 8   -> uneven
344 % 32 = 24  -> uneven
344 % 64 = 24  -> uneven
```

With 8 cores (the largest power-of-2 divisor), each core gets `344 / 8 = 43` tiles. But 43 is prime, which has consequences for `compute_subblock_w()` (see below).

### Why 64 = 2^6 Is More Amenable (DeepSeek V3)

DeepSeek V3's MoE expert width is 2048, producing `2048 / 32 = 64` column tiles:

```
64 = 2^6

Divisors: 1, 2, 4, 8, 16, 32, 64
```

Every power of 2 up to 64 divides evenly. This gives the shard planner maximum flexibility:

```
With 16 cores: N_per_core = 4 tiles
  shard_tiles = 224 * 4 = 896
  shard_bytes = 896 * 1,088 = 975 KB  -> fits L1

With 32 cores: N_per_core = 2 tiles
  shard_bytes = 224 * 2 * 1,088 = 487 KB  -> fits with headroom

With 64 cores: N_per_core = 1 tile
  shard_bytes = 224 * 1 * 1,088 = 244 KB  -> fits easily
```

> **Warning:** When designing model architectures for Tenstorrent hardware, intermediate dimensions that are highly composite (powers of 2 or products of small primes) produce significantly better sharding characteristics. Dimensions with large prime factors (like LLaMA's factor of 43) constrain core-count options and may force DRAM streaming.

---

## The compute_subblock_w() Function

The `compute_subblock_w()` function determines the matmul subblock width -- the innermost tile loop dimension that controls how many output tiles are accumulated in dest registers before writeback:

```python
# EXISTING -- from blaze/utils.py
def compute_subblock_w(out_w_per_core: int, max_subblock_w: int = 8) -> int:
    """Select the largest subblock width that divides out_w_per_core.

    Larger subblocks increase compute efficiency but must divide evenly.
    """
    for sw in range(min(out_w_per_core, max_subblock_w), 0, -1):
        if out_w_per_core % sw == 0:
            return sw
    return 1
```

### Walkthrough: LLaMA-7B with 8 Cores (out_w_per_core = 43)

```
compute_subblock_w(43):
  Try 8: 43 % 8 = 3 -> skip
  Try 7: 43 % 7 = 1 -> skip
  Try 6: 43 % 6 = 1 -> skip
  Try 5: 43 % 5 = 3 -> skip
  Try 4: 43 % 4 = 3 -> skip
  Try 3: 43 % 3 = 1 -> skip
  Try 2: 43 % 2 = 1 -> skip
  Try 1: 43 % 1 = 0 -> return 1

subblock_w = 1 (43 is prime -- worst case for subblocking)
```

A `subblock_w` of 1 means each output tile is computed and written individually, losing the efficiency benefit of accumulating multiple output tiles in dest registers.

### Walkthrough: DeepSeek V3 with 16 Cores (out_w_per_core = 4)

```
compute_subblock_w(4):
  Try 4: 4 % 4 = 0 -> return 4

subblock_w = 4 (all 4 output tiles accumulated before writeback)
```

### Walkthrough: LLaMA-7B with 4 Cores (alternative)

```
out_w_per_core = 344 / 4 = 86
compute_subblock_w(86):
  86 = 2 * 43
  Try 8: 86 % 8 = 6 -> skip
  Try 7: 86 % 7 = 2 -> skip
  ...
  Try 2: 86 % 2 = 0 -> return 2

subblock_w = 2 (better than 1, but still limited by factor 43)
```

> **Warning:** When TensorAdapter detects `subblock_w = 1`, it should emit a diagnostic suggesting alternative core counts. Using 4 cores instead of 8 for the LLaMA gate matmul doubles `out_w_per_core` from 43 to 86, improving `subblock_w` from 1 to 2. The tradeoff is less parallelism (4 cores vs. 8) but better per-core compute efficiency.

---

## DRAM Streaming Fallback

### When Weight Shards Exceed L1

For the LLaMA-7B gate weight `[4096, 11008]` in bfloat8_b, even with 8 cores the per-core shard is ~5.7 MB -- far exceeding L1. The DRAM streaming matmul addresses this by streaming K-slices from DRAM rather than holding all weight tiles in L1 simultaneously.

```
Standard matmul: weight CB holds ALL shard tiles in L1
  num_pages = K_tiles * N_per_core = 128 * 43 = 5,504
  L1 = 5,504 * 1,088 = 5.7 MB  -> DOES NOT FIT

DRAM streaming matmul: weight CB holds K_BLOCK tiles at a time
  K_BLOCK = 4 (process 4 K-tiles per iteration)
  num_pages = K_BLOCK * N_per_core = 4 * 43 = 172
  L1 = 172 * 1,088 = 187 KB  -> FITS

  Outer loop: for k_iter in range(128 / 4 = 32):
    Stream 172 weight tiles from DRAM into FIFO CB
    Accumulate partial matmul result
    Repeat until all K tiles processed
```

### L1 Budget with DRAM Streaming

```
Per-core CB budget (LLaMA-7B gate matmul, 8 cores, DRAM streaming, K_BLOCK=4):
  CB in0 (activation, FIFO):    128 * 2,048 = 256 KB
  CB in1 (weight, FIFO/stream): 172 * 1,088 = 183 KB
  CB out (output, FIFO):         43 * 2,048 =  86 KB
  Total:                                       525 KB
  Available L1:                              ~1,300 KB (after kernel overhead)
  Utilization:                                   40%
```

> **Warning:** The `k_num_tiles` CT arg in a DRAM streaming context represents the **per-iteration** K block, not the total K dimension. The total K dimension becomes the outer loop bound. TensorAdapter must distinguish between "fit-in-L1" sharding (where k_num_tiles is the full K) and "DRAM streaming" strategies (where k_num_tiles is K_BLOCK).

---

## The Shard Computation Algorithm

### Overview

Given a `ShapeDescriptor` (from Files 01-02) and a core grid specification (from `GridConfig`), the shard computation algorithm produces per-core parameters:

```
Input:
  descriptor:  ShapeDescriptor (from tile_decompose())
  core_grid:   CoreGridSpec (num_cores, sharding_dim, core_ranges)
  op_hint:     str (e.g., "matmul.in1")

Output:
  ShardDescriptor:
    shard_shape:        tuple[int, ...]  (per-core element dimensions)
    shard_tile_grid:    tuple[int, ...]  (per-core tile dimensions)
    shard_num_pages:    int              (tiles buffered per core)
    page_size:          int              (bytes per tile)
    shard_l1_bytes:     int              (total L1 footprint per core)
    core_ranges:        CoreRangeSet     (cores for this shard pattern)
    is_even:            bool             (all cores have same shard size)
```

> *Design Principle P3 (Validate at Each Lowering Step):* The algorithm validates at each step: strategy selection checks op compatibility, shard shape computation checks even divisibility, tile grid computation checks alignment, and L1 budget checks catch overflow before CB allocation.

### Step 1: Determine Sharding Strategy

```python
# PROPOSED -- sharding strategy selection
def _determine_sharding_strategy(descriptor, op_hint, num_cores):
    op_type = op_hint.split(".")[0] if "." in op_hint else op_hint
    port = op_hint.split(".")[1] if "." in op_hint else ""

    if num_cores == 1:
        return {"strategy": "none", "sharding_dim": -1}

    if op_type in ("matmul", "matmul_cb", "kn_sliced_matmul", "dram_streaming_matmul"):
        if port in ("in1", "weight"):
            return {"strategy": "column", "sharding_dim": -1}
        elif port in ("in0", "activation"):
            return {"strategy": "broadcast", "sharding_dim": -1}
        elif port in ("out", "output"):
            return {"strategy": "column", "sharding_dim": -1}

    elif op_type in ("rmsnorm", "padded_rmsnorm", "layernorm"):
        return {"strategy": "none", "sharding_dim": -1}

    elif op_type == "mcast":
        return {"strategy": "broadcast", "sharding_dim": -1}

    return {"strategy": "column", "sharding_dim": -1}
```

### Step 2: Compute Per-Core Shard Shape

```python
# PROPOSED -- shard shape computation
def _compute_shard_shape(descriptor, sharding_dim, num_cores, strategy):
    if strategy in ("none", "broadcast"):
        return descriptor.padded_shape, True  # Full tensor on each core

    tile_grid = descriptor.tile_grid_shape
    tiles_in_dim = tile_grid[sharding_dim]

    if num_cores > tiles_in_dim:
        raise ValueError(
            f"Cannot shard {tiles_in_dim} tiles across {num_cores} cores."
        )

    tiles_per_core = tiles_in_dim // num_cores
    remainder = tiles_in_dim % num_cores
    is_even = (remainder == 0)

    # Convert tile count to element count
    tile = descriptor.tile_shape
    tile_dim = tile[1] if sharding_dim == len(tile_grid) - 1 else tile[0]
    shard_elements = tiles_per_core * tile_dim

    shard = list(descriptor.padded_shape)
    shard[sharding_dim] = shard_elements
    return tuple(shard), is_even
```

### Step 3: Compute Per-Core Tile Grid and num_pages

```python
# PROPOSED -- per-core num_pages by role
def _compute_shard_num_pages(shard_tile_grid, total_shard_tiles, op_hint, global_desc):
    op_type = op_hint.split(".")[0] if "." in op_hint else op_hint
    port = op_hint.split(".")[1] if "." in op_hint else ""

    if op_type in ("matmul", "matmul_cb", "kn_sliced_matmul"):
        if port in ("in1", "weight"):
            return total_shard_tiles       # DIRECT_ADDRESS: all tiles addressable
        elif port in ("in0", "activation"):
            return global_desc.tile_grid_shape[-1]  # Global K tiles
        elif port in ("out", "output"):
            return shard_tile_grid[-1]     # out_w_per_core

    elif op_type in ("rmsnorm", "padded_rmsnorm", "layernorm"):
        return shard_tile_grid[-1]         # Width accumulation

    elif op_type in ("add", "mul", "silu", "relu", "gelu", "eltwise_mul"):
        return 1                           # Streaming

    return total_shard_tiles               # Default: all tiles
```

### Step 4: Validate L1 Budget

```python
# PROPOSED -- L1 budget validation
L1_BUDGET_BYTES = 1_500_000
L1_OVERHEAD_BYTES = 200_000

def _validate_shard_l1(shard_num_pages, page_size, op_hint):
    shard_bytes = shard_num_pages * page_size
    available = L1_BUDGET_BYTES - L1_OVERHEAD_BYTES

    if shard_bytes > available:
        raise L1BudgetError(
            f"Shard CB for '{op_hint}' requires {shard_bytes:,} bytes "
            f"but only ~{available:,} available per core. "
            f"Options: increase core count, reduce num_pages for streaming, "
            f"or use a lower-precision data format."
        )
```

> **Warning:** This is a per-CB check, not a per-core check. A core typically hosts 3-6 CBs. The total of all CBs must fit within L1. The full multi-CB budget validation is the subject of Chapter 7.

### Step 5: Assemble ShardDescriptor

```python
# PROPOSED -- ShardDescriptor dataclass
@dataclass(frozen=True)
class ShardDescriptor:
    shard_shape: tuple[int, ...]
    shard_tile_grid: tuple[int, ...]
    shard_num_pages: int
    page_size: int
    shard_l1_bytes: int
    core_ranges: object           # ttnn.CoreRangeSet
    num_cores: int
    is_even: bool
    sharding_strategy: str
    global_descriptor: ShapeDescriptor
    op_hint: str


def compute_shard(descriptor, num_cores, core_ranges, op_hint):
    """Compute per-core shard parameters from a global ShapeDescriptor."""
    strategy_info = _determine_sharding_strategy(descriptor, op_hint, num_cores)
    shard_shape, is_even = _compute_shard_shape(
        descriptor, strategy_info["sharding_dim"], num_cores, strategy_info["strategy"]
    )
    shard_tile_grid, total_shard_tiles = _compute_shard_tile_grid(
        shard_shape, descriptor.tile_shape
    )
    shard_num_pages = _compute_shard_num_pages(
        shard_tile_grid, total_shard_tiles, op_hint, descriptor
    )
    page_size = descriptor.page_size
    _validate_shard_l1(shard_num_pages, page_size, op_hint)

    return ShardDescriptor(
        shard_shape=shard_shape,
        shard_tile_grid=shard_tile_grid,
        shard_num_pages=shard_num_pages,
        page_size=page_size,
        shard_l1_bytes=shard_num_pages * page_size,
        core_ranges=core_ranges,
        num_cores=num_cores,
        is_even=is_even,
        sharding_strategy=strategy_info["strategy"],
        global_descriptor=descriptor,
        op_hint=op_hint,
    )
```

---

## Worked Example: LLaMA-7B Gated MLP Shard Computation

Tracing the full shard computation using an 8-core grid with `build_ab_grids()` splitting into 4 `a_cores` and 4 `b_cores`.

### Gate Weights: `[4096, 11008]` on `a_cores` (4 cores)

```
Global ShapeDescriptor:
  logical_shape  = (4096, 11008)
  tile_shape     = (32, 32)
  tile_grid      = (128, 344)
  total_tiles    = 44,032
  page_size      = 1,088 bytes (bfloat8_b)

Shard computation:
  Strategy:       column (weight: in1 port)
  Tiles in width: 344
  Tiles per core: 344 / 4 = 86
  Even?           Yes

Per-core results:
  shard_shape      = (4096, 2752)  (86 * 32 = 2,752 elements)
  shard_tile_grid  = (128, 86)
  shard_num_pages  = 11,008        (DIRECT_ADDRESS: all tiles)
  shard_l1_bytes   = 11,008 * 1,088 = ~11.4 MB
```

This exceeds L1 -- the weight uses DIRECT_ADDRESS backed by a pre-placed tensor, or the op falls back to DRAM streaming.

### Activation: `[1, 4096]` broadcast to all 8 cores

```
Strategy: broadcast (no sharding)
Per-core: shard_shape = (32, 4096), shard_tile_grid = (1, 128)
num_pages = 128 (K tiles), page_size = 2,048
shard_l1_bytes = 256 KB  -- fits comfortably
```

### Summary Table

| Tensor | Shape | Cores | Strategy | Shard Tiles | num_pages | L1 per core |
|--------|-------|-------|----------|------------|-----------|-------------|
| Activation | `[1, 4096]` | 8 (all) | broadcast | 128 | 128 | 256 KB |
| Gate wt | `[4096, 11008]` | 4 (a) | column | 11,008 | 11,008* | ~11 MB* |
| Up wt | `[4096, 11008]` | 4 (b) | column | 11,008 | 11,008* | ~11 MB* |
| Gate out | `[1, 11008]` | 4 (a) | column | 86 | 86 | 172 KB |
| Up out | `[1, 11008]` | 4 (b) | column | 86 | 86 | 172 KB |
| Down wt | `[11008, 4096]` | 8 (all) | column | 5,504 | 5,504* | ~5.7 MB* |
| Down out | `[1, 4096]` | 8 (all) | column | 16 | 16 | 32 KB |

*DIRECT_ADDRESS: backed by pre-placed tensor or DRAM-streamed; L1 column shows total addressable, not simultaneous buffering.

---

## Worked Example: DeepSeek V3 MoE Expert Shard Planning

```
Weight: expert_gate [7168, 2048], bfloat8_b
Device: 8x8 grid (64 available cores)

Tile decomposition:
  tile_grid = (224, 64), page_size = 1,088

Shard planning (N_tiles = 64, all power-of-2 divisors):
   8 cores: N_per_core = 8  -> 224*8*1088 = 1.9 MB -> exceeds L1
  16 cores: N_per_core = 4  -> 224*4*1088 = 975 KB -> FITS
  32 cores: N_per_core = 2  -> 224*2*1088 = 487 KB -> fits with headroom

Selected: 16 cores (4x4 or 2x8 grid)
  shard_tile_shape = (224, 4)
  shard_bytes = 975 KB
  out_w_per_core = 4
  compute_subblock_w(4) = 4  (excellent efficiency)

Remaining L1: ~1300 - 975 = ~325 KB
  Activation CB: 224 * 2,048 = 448 KB at bfloat16
  Problem: activation alone exceeds remaining budget!
  Resolution: use bfloat8_b for activation too, or fewer weight tiles via DRAM streaming
```

This example shows that even DeepSeek V3's cleaner dimensions can hit L1 limits when K is large (224 tiles).

---

## Deriving CT Args from ShardDescriptor

Several CT args depend on per-core shard dimensions rather than global dimensions:

> *Design Principle P2 (Carry Metadata from Both Worlds):* `k_num_tiles` is a **global** property (same across all cores), but `out_w_per_core` is a **shard** property (varies with core count). The convention: any CT arg depending on the distribution dimension uses ShardDescriptor; any depending on non-distributed dimensions uses ShapeDescriptor.

```python
# PROPOSED -- CT arg derivation from ShardDescriptor
def derive_shard_ct_args(op_type, shard_descs, global_descs):
    args = {}
    if op_type in ("matmul", "matmul_cb", "kn_sliced_matmul"):
        in0_global = global_descs.get("in0")
        in1_shard = shard_descs.get("in1")
        if in0_global and in1_shard:
            k_num_tiles = in0_global.tile_grid_shape[-1]
            args["k_num_tiles"] = k_num_tiles
            args["out_w_per_core"] = in1_shard.shard_tile_grid[-1]
            args["subblock_w"] = compute_subblock_w(args["out_w_per_core"])
    return args
```

---

## Edge Cases

### Uneven Tile Distribution

When the tile count does not divide evenly:

```
Global tile grid: [128, 100] (100 column tiles)
Core count: 8
Tiles per core: 100 / 8 = 12 remainder 4

First 4 cores: 12 tiles each
Last 4 cores: 13 tiles each
```

> **Warning:** Uneven sharding complicates CT arg derivation. If `out_w_per_core` differs across cores, it cannot be a single compile-time constant. Most existing Blaze ops assume even sharding. TensorAdapter should prefer core counts that divide evenly and warn when uneven sharding is unavoidable.

### Single-Core Execution

When `num_cores == 1`, shard == global. This arises for the sender core in multicast, the gated_reduce core, and small tensors. Single-core execution must still pass through `compute_shard()` to ensure consistent validation.

### Sender Core vs. Compute Cores

The sender core has different CB requirements than matmul cores:

| Aspect | Sender Core | Matmul Core |
|--------|-------------|-------------|
| Activation CB | Full tensor (source for mcast) | Full tensor (received via mcast) |
| Weight CB | None | Column shard (DIRECT_ADDRESS) |
| Output CB | Gathered results (all columns) | Column shard (matmul output) |

---

## Integration: adapt_with_sharding()

The complete flow from tensor to per-core CB:

```python
# PROPOSED -- complete adaptation pipeline
def adapt_with_sharding(adapter, tensor, op_hint, grid_config, fused_program):
    """tensor -> global shape -> shard -> CB"""
    # Step 1: Global shape analysis (File 02)
    global_desc = adapter.shape_engine.full_analysis(tensor, op_hint=op_hint)

    # Step 2: Determine core grid
    core_ranges = grid_config.get_matmul_cores()
    num_cores = _count_cores(core_ranges)

    # Step 3: Compute shard (this file)
    shard_desc = compute_shard(global_desc, num_cores, core_ranges, op_hint)

    # Step 4: Allocate CB via CBBackend
    handle = adapter.cb_backend.allocate(fused_program, shard_desc, tensor)

    # Step 5: Record in ShapeTracker
    adapter.shape_tracker.record(handle, global_desc)
    return TypedHandle(handle=handle, shape=global_desc)
```

This replaces the six manual steps (choose cores, compute shard, configure shard spec, allocate CB, extract core_ranges, compute CT args) with a single function call.

---

## Validation Summary

| Step | Condition | Error Type | Severity |
|------|-----------|-----------|----------|
| Strategy | Unknown op_type | Warning + default | Low |
| Shard shape | More cores than tiles | `ValueError` | High |
| Shard shape | Uneven distribution | Warning | Medium |
| Tile grid | Shard not tile-aligned | `ValueError` | High |
| L1 budget | Shard exceeds L1 | `L1BudgetError` | High |
| L1 budget | Shard uses >80% L1 | Warning | Medium |
| CT args | k_num_tiles == 0 | `ValueError` | High |
| CT args | out_w_per_core inconsistent | `ValueError` | High |
| Subblock | subblock_w == 1 | Diagnostic | Low (perf) |

---

## Key Takeaways

- The global tile grid from File 01 represents the tensor's total tile footprint. Per-core **shard shape** is the global tile grid partitioned across cores, and it determines the CB `num_pages` on each core. Global shapes inform shape propagation (File 03); shard shapes inform CB allocation and CT arg derivation.
- The **FIFO vs. DIRECT_ADDRESS** CB mode distinction is critical: FIFO CBs allocate new L1 ring buffers, while DIRECT_ADDRESS CBs point to pre-existing L1 buffers. The shard computation must account for which mode is in use; treating all CBs as FIFO overestimates L1 usage, while treating all as DIRECT_ADDRESS underestimates it.
- `GridConfig` and `pick_matmul_cores()` determine the core grid. The core count directly determines the shard width: `tiles_per_core = total_tiles / num_cores`. Dimensions with large prime factors (LLaMA's 43 in `344 = 2^3 * 43`) severely constrain options, while power-of-2 dimensions (DeepSeek V3's `64 = 2^6`) provide maximum flexibility.
- The `compute_subblock_w()` function derives the matmul subblock width from `out_w_per_core`. Prime values of `out_w_per_core` force `subblock_w = 1`, reducing compute efficiency. TensorAdapter should emit a diagnostic when this occurs and suggest alternative core counts.
- When weight shards exceed L1, the **DRAM streaming fallback** streams K-slices from DRAM through a small FIFO CB. The `k_num_tiles` CT arg then represents the per-iteration K block, not the total K dimension. TensorAdapter must distinguish between L1-resident and DRAM-streaming strategies.
- Two critical distinctions prevent common bugs: (1) `shard_num_pages` (tiles buffered simultaneously) vs. `total_shard_tiles` (tiles addressable per core); (2) global CT args (`k_num_tiles`) vs. shard CT args (`out_w_per_core`).

## Source Files

- `blaze/fused_program.py` -- `OverlappedView` (shard_shape, core_ranges), `FusedProgram.cb_from_tensor()` (shard-aware CB creation)
- `blaze/utils.py` -- `GridConfig`, `pick_matmul_cores()`, `compute_subblock_w()` (core grid and subblock computation)
- `blaze/_gated_mlp.py` -- `build_gated_mlp_graph()` (production 11-node graph using GridConfig)
- `blaze/cb_handle.py` -- `CBHandle.core_ranges`, `CBAccessMode.DIRECT_ADDRESS`
- `blaze/ops/matmul/op.py` -- `Matmul.emit()` (manual `out_w_per_core = in1_cb.num_pages // k_num_tiles`)

---

← Previous: [Chapter 3: The TensorAdapter Architecture](../ch03_tensor_adapter/) | Next: [Chapter 5: Transparent Padding Management](../ch05_padding/) →
