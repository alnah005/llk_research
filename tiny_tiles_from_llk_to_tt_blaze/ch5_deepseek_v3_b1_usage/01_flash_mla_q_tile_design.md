# Flash MLA: The 8x32 Q Tile Design

Flash MLA decode is the canonical tiny-tile consumer in DeepSeek V3 B1. The query side of the attention matmul carries exactly 8 logical rows per core — one per attention head assigned to that core — and the kernel uses an 8x32 tile geometry to match that activation height precisely. The key/value side keeps the standard 32x32 geometry because KV cache lines and matmul accumulators are already 32-row aligned. This file explains where the "8" comes from, how it propagates through the `PNHt` / `Q_TILE_HEIGHT` / `DHt` parameter set, and what the resulting mixed-geometry matmul looks like.

**Prerequisites:** Chapter 2 (tile geometry, faces, BFP8 layout) and Chapter 3 (CB descriptors and tile size computation). Familiarity with multi-head latent attention helps but is not required.

## Why 8 rows per Q tile?

DeepSeek V3 uses 128 query heads. In the B1 deployment with TP=2, each device sees 64 heads. Flash MLA decode shards those heads across an 8-core column of the Tensix grid, giving each core exactly `n_heads_per_core = 8` heads to process. Because Flash MLA is the *decode* path, the time dimension is 1 token — so the per-position activation is a single row per head, and the per-core Q activation is an `[8, head_dim]` slab.

That slab maps naturally onto a single tile high by `head_dim / 32` tiles wide. For DeepSeek's MLA head dimension `head_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576`, that is one tile high by 18 tiles wide. With a standard 32x32 tile, the 8 real rows would sit in the top quarter and 24 rows of zero padding would ride along through every CB, every matmul, and every SFPU pass. With an 8x32 tile, the geometry matches the data exactly and the L1 footprint of `cb_q_in` drops by 4x (see Chapter 5, Section "L1 Budget Comparison" for the full per-CB accounting).

The kernel does not hard-code "8" — it reads the height from the input tensor's tile object:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:414
Q_TILE_HEIGHT = q_tile.tile_shape[0]
TILE_WIDTH = 32
```

And it cross-checks against the actual shard shape:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:423
num_q_heads_per_core = input_tensor_q.memory_config().shard_spec.shape[0]
```

The two must agree — `Q_TILE_HEIGHT` must equal `num_q_heads_per_core`, or the per-tile row count would not line up with the per-core head count. The relationship is captured one line later:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:459
PNHt = num_q_heads_per_core / Q_TILE_HEIGHT
```

For Flash MLA `PNHt = 1`: each core's Q activation is exactly one tile tall. That `1` flows into every CB sizing calculation downstream.

## The parameter set

The full set of tile-count parameters that the Flash MLA op emits to its compute kernels:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `Q_TILE_HEIGHT` | 8 | Q tile row count (= heads per core) |
| `K_TILE_HEIGHT` | 32 | K/V tile row count (standard) |
| `TILE_WIDTH` | 32 | All tiles, column count |
| `PNHt` | 1 | Q tiles in the height dimension (= heads_per_core / Q_TILE_HEIGHT) |
| `DHt` | 18 | head_dim tiles wide (576 / 32) |
| `vDHt` | 16 | value head_dim tiles wide (512 / 32) |
| `Sk_chunk_t` | 4 | K tiles per attention chunk along the sequence axis |

The width and value-dim parameters fall out arithmetically:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:457
DHt = DH // TILE_WIDTH        # 576 // 32 = 18
vDHt = head_dim_v // TILE_WIDTH  # 512 // 32 = 16
```

`PNHt` and `DHt` together give the Q tile-count along each axis:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:520
q_tiles = PNHt * DHt                  # 1 * 18 = 18 tiles in cb_q_in
k_tiles = Sk_chunk_t * DHt * 2        # 4 * 18 * 2 = 144 (double-buffered)
out0_t = PNHt * vDHt                  # 1 * 16 = 16 tiles in cb_out_o
```

The 8x32 Q geometry never escapes its lane: it lives in `cb_q_in`, in the per-position output statistics CBs (`cb_out_ms`, `cb_interm_ms`, etc.), and in the final `cb_out_o`. The K/V side stays at 32x32 throughout, so `k_tiles` and the score-accumulation CBs use the standard tile size.

## Constructing the tile objects

The op emits two distinct `ttnn.Tile` objects — one tiny, one full — and tags every CB with the appropriate one:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:536-545
Q_TILE_HEIGHT = q_tile.tile_shape[0]
K_TILE_HEIGHT = 32
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))   # (8, 32)
k_tile_obj  = ttnn.Tile((K_TILE_HEIGHT, TILE_WIDTH))   # (32, 32)
full_tile   = k_tile_obj
q_tile_size = q_tiny_tile.get_tile_size(q_df)
```

`get_tile_size(q_df)` is the critical bridge between the logical tile shape and the byte count the CB allocator needs. For an 8x32 BFP8 tile that returns 9 bytes/row × 32 rows + per-row exponent metadata, well under the 1024 + 64 = 1088 bytes of a 32x32 BFP8 tile. Chapter 3, Section "Tile Sizes Across Data Formats" walks through the arithmetic; here it suffices to note that the op never computes the size manually — it asks the tile object.

The same tile object is then passed to the CB format descriptor. For Q:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py (paraphrased from 548, 744)
cb_q_in_descriptor = ttnn.CBFormatDescriptor(
    cb_q_in, q_df, q_tile_size, q_tile_descriptor=q_tiny_tile
)
```

The producer (`cb_descriptor_from_sharded_tensor` at line 713) takes the shortcut of reading the tile geometry straight off the sharded input tensor — Q tensor already carries `tile=(8, 32)` from the calling layer, so the descriptor inherits it automatically. The Q tiny-tile geometry is therefore set in exactly one place upstream (the host-side tensor creation) and propagates through tile object → CB descriptor → compute kernel CT args without manual restatement.

## The mixed-geometry matmul

The Flash MLA inner loop performs `scores = Q · Kᵀ` once per K chunk. Q is tiny, K is full, and the result lands in DST registers with the Q tile geometry preserved:

```text
Q       : [PNHt,        DHt] = [1, 18] tiles  (each tile 8x32)
K_chunk : [Sk_chunk_t,  DHt] = [4, 18] tiles  (each tile 32x32)
Scores  : [PNHt, Sk_chunk_t] = [1,  4] tiles  (each tile 8x32, in DST)
```

The matmul itself is invoked as `sdpa_custom_mm_block(Q, K_chunk, transpose_k=true)`. From the LLK's point of view this is a perfectly ordinary matmul along the inner `DHt` axis — the partial products accumulate across 18 inner steps, each one pulling one Q tile (8x32) and one K tile (32x32, transposed at the matmul-init level), producing one 8x32 output face per `Sk_chunk_t` column. The hardware does not need a separate "tiny matmul" path; the unpacker just emits 8 real rows out of the 32-row Q tile (the other 24 rows do not exist in L1 to begin with — see Chapter 4 for the unpack-side details), and the math engine accumulates into an 8-row DST slice.

The output geometry [PNHt, Sk_chunk_t] = [1, 4] then feeds the online softmax. Because the softmax statistics (`m`, `l`) are per-row, and there are 8 rows per Q tile, those statistics are themselves 8x32 (one row per head, padded out to 32 columns by the SFPU's column-wise layout). This is why `cb_out_ms`, `cb_interm_ms`, and friends all use the tiny tile too — the per-position dimension survives intact from Q through the whole attention pipeline. Chapter 5, Section "L1 Budget Comparison" lays out the full CB-by-CB accounting.

## Why K and V stay at 32x32

A natural question: if Q is 8 rows tall to match heads, why doesn't K shrink to match something on the KV side? Two reasons.

First, K and V are read from the KV cache, which is laid out in standard 32x32 tiles per the cache writer's contract. Cache lines are 32-row aligned in DRAM. Reshaping every read into 8x32 would require an extra repack on the producer side with no offsetting saving on the consumer side — the cache rows are real data, not padding.

Second, the K tile height is what feeds the matmul inner-dimension accumulator. The matmul achieves peak throughput when both operands fill 32-row faces (full `MATH_FIDELITY` lanes engaged each cycle). Shrinking K would idle the math engine. The 4x L1 savings target on the Q side is paid for by Q being padding-saturated (24 padding rows out of 32); K has no such padding to recover, so the standard geometry wins.

A useful mental model: tiny tiles win where the activation dimension is *intrinsically narrow*. Q is narrow because the model only assigns 8 heads to this core. K is wide because the chunked sequence dimension is fully populated.

## Tracing the design end-to-end

The geometry is created once on the host, stored on the input tensor, and read back by the op:

```python
# tests/.../test_flash_mla.py:91
tiny_tile = ttnn.Tile((num_q_heads_per_core, 32))   # (8, 32)
input_q   = ttnn.from_torch(torch_q, tile=tiny_tile, ...)
```

The op extracts it from the tensor, validates it against the shard shape, derives `PNHt`, and emits it as a tile object into every Q-side CB descriptor. The compute kernel sees `Q_TILE_HEIGHT = 8` and `PNHt = 1` as CT args and lays out its inner loops accordingly. Subsequent CBs that carry per-position state (softmax statistics, intermediate outputs) inherit the same 8x32 geometry; CBs that carry K/V or score-vs-K state stay at 32x32. The boundary is sharp and lives in the parameter set — no global "tiny mode" flag.

The next file, `02_flash_mla_mixed_geometry_cb_chain.md`, walks the full CB chain and tabulates the per-CB tile geometry, page count, and L1 budget, showing where the 4x saving compounds and where it gets clawed back by buffers that must stay 32-row tall.
