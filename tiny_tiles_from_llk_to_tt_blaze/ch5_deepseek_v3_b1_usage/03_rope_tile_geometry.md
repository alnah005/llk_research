# 5.3 RoPE Tile Geometry — Deriving Tiny Tiles from the Shard Spec

RoPE is the upstream producer that feeds Q into Flash MLA. Its tile geometry is not a free parameter chosen by the op author — it is _derived_ at host time from the input tensor's `ShardSpec`, then propagated into every CB descriptor and every compile-time arg the kernel sees. This section walks through that derivation in `RopeOp.emit()`, shows how the test harness sets up a tensor whose shard shape encodes the desired tiny-tile geometry, and explains why RoPE's tile height must agree exactly with Flash MLA's `Q_TILE_HEIGHT`.

**Prerequisites:** Chapter 3 (host-side tile descriptors and CB plumbing), Chapter 4 (compute kernel CT-args for tiny tiles), Section 5.1 (Flash MLA Q tile geometry), Section 5.2 (Flash MLA CB chain).

---

## 5.3.1 Shard shape is the source of truth

For width-sharded activations in DeepSeek V3 B1, the producer's `ShardSpec` encodes two things at once:

1. The **number of heads per core** (shard rows), which becomes the tiny-tile height.
2. The **head-dimension slice per core** (shard columns), which becomes the tile-column count `Wt`.

`RopeOp.emit()` reads both directly off the input tensor — there is no separate `tile_height` parameter. The op trusts the producer's shard spec and uses it to construct every CB descriptor downstream.

```python
# models/demos/deepseek_v3_b1/micro_ops/rope/op.py:104-127
# Get tensor properties
data_format = input_tensor.dtype

# Get shard spec from input tensor
shard_spec = input_tensor.memory_config().shard_spec
shard_shape = shard_spec.shape

# Get core grid from shard spec
core_grid = shard_spec.grid

# Calculate dimensions in tiles
# With tiny tiles: shard_shape[0] = n_heads, shard_shape[1] = head_dim
head_dim_per_core_t = shard_shape[1] // ttnn.TILE_SIZE  # head_dim in tiles (Wt) (64 // 32 = 2)

# Calculate tile sizes
# For tiny tiles, the tile height matches the shard height (num_q_heads_per_core)
# This is derived from the input tensor's shard shape
num_q_heads_per_core = shard_shape[0]  # Get from input tensor's shard height
tile = ttnn.Tile((num_q_heads_per_core, ttnn.TILE_SIZE))
tile_size = tile.get_tile_size(data_format)

# Number of tiles for intermediate buffers
num_interm_tiles = head_dim_per_core_t  # Intermediate buffers sized for one head row
```

A few things to notice:

- `ttnn.TILE_SIZE` here is `32`, the tile **width** in elements — not a byte count. Both `head_dim_per_core_t` (tile-column count along width) and `ttnn.Tile((H, 32))` (tile shape) use it for their natural meaning.
- The tile height `num_q_heads_per_core` is unrestricted by the host code — it can be 1, 2, 8, 16, anything the SFPU and unpacker support (see Chapter 2 for the per-arch list of legal heights). Whatever the producer sharded as, RoPE will faithfully match.
- `tile_size = tile.get_tile_size(data_format)` is the **byte size** of one tiny tile in the chosen dtype. This is what gets fed into every `CBFormatDescriptor` and into the NCRISC's `cos_sin_page_size` CT-arg so DRAM reads land on tile-sized boundaries.

The intermediate buffer for the rotated-input scratchpad (CB 24) is sized at exactly `num_interm_tiles = head_dim_per_core_t` tiles — one tile per column of the head dimension, holding a single row of `num_q_heads_per_core` heads. That sizing is deliberate: the kernel processes one Ht-row × Wt-column tile at a time, and only needs one row of scratch active at any moment.

## 5.3.2 What the kernel sees

Once the host code has derived the tile geometry, it pushes it into the kernel through two channels: the CB format descriptor (so the kernel's `cb_reserve_back` / `cb_push_back` operate on the right page size) and per-core compile-time args.

```python
# models/demos/deepseek_v3_b1/micro_ops/rope/op.py:184-198
ncrisc_named_compile_time_args = [
    ("in_cb", input_cb),
    ("cos_tensor_address", cos_tensor.buffer_address()),
    ("sin_tensor_address", sin_tensor.buffer_address()),
    ("cos_sin_cb", cos_sin_cb),
    ("trans_mat_cb", trans_mat_cb),
    ("cos_sin_page_size", tile_size),
    ("Wt", head_dim_per_core_t),
    ("Ht", 1),
    ("total_Wt", total_Wt),
]

# Per-core start_tile_offset: each core reads its width slice from DRAM
all_cores = ttnn.corerange_to_cores(core_grid)
start_tile_offset_core_values = [(core, idx * head_dim_per_core_t) for idx, core in enumerate(all_cores)]
```

The kernel only knows `(Wt, Ht=1)` — it does not separately know the tile height. It does not need to. The tile height is baked into `cos_sin_page_size = tile_size`, which already accounts for height × width × element_bytes. The TRISC compute kernel, similarly, receives no explicit `tile_h` arg; it operates on whatever tile the CB descriptor declared, and the LLK call chain (`unpack_tile`, `mul_tiles`, `pack_tile`) handles the partial-face count internally based on the configured tile descriptor (see Chapter 4, "Tile descriptor flow into LLKs").

`Ht=1` here is the **height in tiles**: with `num_q_heads_per_core` heads and a tile height equal to `num_q_heads_per_core`, the whole shard is exactly one tile tall. The number of heads only re-enters the kernel's view through the per-tile element count, which the LLKs read from the configured tile descriptor at unpack time.

The per-core `start_tile_offset` distributes `total_Wt = head_dim_per_core_t * num_cores` tile-columns evenly across the grid: core `idx` starts at tile offset `idx * head_dim_per_core_t` in the DRAM cos/sin cache. Because each core's slice is `Wt` tile-columns wide, all cores read disjoint, non-overlapping page ranges.

## 5.3.3 Test harness: how the geometry gets set up

The test constructs an input tensor whose shard shape already encodes the desired tiny-tile geometry, then passes the matching `ttnn.Tile` to `from_torch`. This is the same pattern used by Flash MLA (`test_flash_mla.py:91`) and is the canonical way to set tile geometry on a sharded tensor at creation.

```python
# models/demos/deepseek_v3_b1/tests/unit_tests/test_rope.py:71-95
# Create WIDTH_SHARDED memory config for input
# Use tiny tiles: tile height = num_heads (no padding to 32)
tiny_tile = ttnn.Tile((num_heads, ttnn.TILE_SIZE))
# Create core grid from grid_size parameter
start_x, start_y = 0, 0
end_x = start_x + grid_size[0] - 1
end_y = start_y + grid_size[1] - 1
core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))})

input_shard_spec = ttnn.ShardSpec(
    core_grid,
    (num_heads, head_dim // (core_grid.num_cores())),
    ttnn.ShardOrientation.ROW_MAJOR,
)
input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

# Create TTNN input tensor with WIDTH_SHARDED memory and tiny tile
tt_x = ttnn.from_torch(
    x_ttnn,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=input_mem_config,
    tile=tiny_tile,
)
```

Note that **two things must agree** at tensor creation:

- The shard shape `(num_heads, head_dim // num_cores)` declares the per-core slice in elements.
- The tile shape `(num_heads, 32)` passed via `tile=tiny_tile` declares the on-device tile geometry.

If these disagree (e.g. `tile=(8, 32)` but `shard_shape=(2, ...)`), `ttnn.from_torch` will either fail validation or — worse — silently produce a tensor whose tile metadata lies about its real layout. The op's later check `num_q_heads_per_core = shard_shape[0]` would then read `2` while every consumer expects `8`, and CB page sizes would diverge.

## 5.3.4 Tested geometries

`test_rope.py` parametrizes `num_heads` and `grid_size` and exercises several legal combinations. The table below maps the test inputs to the derived tile geometry and the resulting `num_interm_tiles`:

| num_heads | head_dim | num_cores | shard_shape (H, W) | tile (H, W) | Wt = `head_dim_per_core_t` | num_interm_tiles |
|-----------|----------|-----------|--------------------|-------------|---------------------------|------------------|
| 8         | 64       | 2         | (8, 32)            | (8, 32)     | 1                         | 1                |
| 2         | 64       | 1         | (2, 64)            | (2, 32)     | 2                         | 2                |
| 1         | 64       | 1         | (1, 64)            | (1, 32)     | 2                         | 2                |
| 8         | 64       | 1         | (8, 64)            | (8, 32)     | 2                         | 2                |

The first row is the Flash MLA production geometry: `num_q_heads_per_core = 8`, exactly matching `Q_TILE_HEIGHT = 8` in `flash_mla/op.py:414`. The other rows exercise the host code's flexibility — the same `RopeOp.emit()` produces correct CB descriptors for tile heights of 1, 2, and 8 with no source change, because every dimension is derived from `shard_shape`.

## 5.3.5 Why RoPE's tile height must equal Flash MLA's `Q_TILE_HEIGHT`

RoPE writes its output Q tiles into a CB that Flash MLA reads as `cb_q_in`. The handoff is a producer-consumer CB chain, not a copy — the same L1 pages that RoPE pushes are what MLA's unpacker pulls. For that to work correctly, three things have to line up:

1. **Page size.** RoPE's output CB is declared with `page_size = tile.get_tile_size(data_format)` where `tile = (num_q_heads_per_core, 32)`. Flash MLA's `cb_q_in` is declared with `page_size = q_tile_size` where `q_tile_size = q_tiny_tile.get_tile_size(q_df)` and `q_tiny_tile = (Q_TILE_HEIGHT, TILE_WIDTH) = (8, 32)` (see `flash_mla/op.py:538, 552`). If `num_q_heads_per_core != Q_TILE_HEIGHT`, the two byte counts disagree — RoPE pushes pages of one size while MLA pulls pages of another. `cb_wait_front` returns at the wrong byte offset, and subsequent unpack instructions read garbage.

2. **Tile descriptor.** The `TileDescriptor` configured into the unpacker controls how many faces are unpacked per tile (Chapter 4). If RoPE's pushed pages carry an `(8, 32)` payload but MLA's unpacker is configured for `(8, 32)`, all is well. If MLA were configured for `(32, 32)` instead, it would unpack four 16×16 faces per tile and trample three tiles' worth of downstream pages.

3. **Tile count along the head dimension.** MLA expects `q_tiles = PNHt * DHt` pages in `cb_q_in` (`flash_mla/op.py:520`), where `PNHt = num_q_heads_per_core / Q_TILE_HEIGHT`. With `num_q_heads_per_core = 8` and `Q_TILE_HEIGHT = 8`, `PNHt = 1` and `q_tiles = DHt`. RoPE writes exactly `head_dim_per_core_t` tiles per row × `Ht=1` rows, so the counts match. Change RoPE's tile height without changing MLA's, and the page-count expectation also drifts.

In short: the contract between RoPE and Flash MLA is `(tile_height, tile_width, dtype, tile-count-per-shard)`. RoPE derives this from its input shard spec; Flash MLA derives the consumer side from `input_tensor_q.memory_config().shard_spec.shape[0]` (line 423). The only reason the contract holds is that the **same tensor** is being passed — RoPE writes into it, MLA reads from it, and both read its shard spec to compute compatible tile dimensions. Break that invariant (e.g. allocate two separate tensors with different shard specs and connect them through a CB chain), and the system silently corrupts data or hangs at the first `cb_wait_front`.

## 5.3.6 Summary

`RopeOp.emit()` is a minimal, mechanical translation from shard spec to CB descriptors:

- `shard_shape[0] -> tile height` (heads packed into one tile).
- `shard_shape[1] -> tile width` in elements; `// 32` gives the tile-column count `Wt`.
- `tile = ttnn.Tile((shard_shape[0], 32))` is constructed once and threaded through every CB and into the NCRISC's `cos_sin_page_size` CT-arg.
- Intermediate scratch (CB 24) is sized at `Wt` tiles — one row of head-dim tiles.

No parameter on the op itself controls the tile geometry. The op is _shard-driven_. Get the producer's shard spec right and the consumer side falls out automatically; get it wrong and the failure mode is a corrupt-CB hang at the first downstream `cb_wait_front`. This is the same discipline that Sections 5.4 (`create_q_heads`) and 5.5 (`kv_cache_branch`) follow, and it is what makes the fused attention pipeline composable across five tiny-tile-aware ops without per-op tile-geometry plumbing.
