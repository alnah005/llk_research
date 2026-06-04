# 1.3 Why Tiny Tiles Exist

Tiny tiles are not a feature added for completeness or symmetry; they are a direct response to a specific shape problem that arises in batch-1 decode. When per-core activation row counts are naturally small — eight attention heads, one sequence position, a handful of sampling outputs — rounding up to a 32x32 canonical tile burns the majority of dest, L1, and matmul cycles on padding zeros. This section quantifies that waste using Flash MLA's Q tile as the running example, then explains why tiny tiles, once chosen, force their geometry through every downstream CB and consumer kernel in the program.

**Prerequisites:** Section 1.1 (canonical 32x32 tile structure, four 16x16 faces), Section 1.2 (legal tiny-tile geometries: `num_faces ∈ {1, 2, 4}`, `face_r_dim ∈ {1, 2, 4, 8, 16}`, `face_c_dim = 16`, as defined by `validate_tensor_shape_tile_dependent_ops_()` in `tt_metal/tt_metal/tt-llk/common/tensor_shape.h:87-94`).

---

## 1. The Problem: Batch-1 Decode Has Tall-Thin Activation Shapes

Decode-mode inference processes one new token at a time. The sequence dimension collapses to 1, and the attention pattern changes from "many queries against many keys" (prefill, batched) to "one query position per request, against the full KV cache" (decode, latency-bound). When this workload is sharded across a Tensix grid, each core ends up owning a vanishingly thin slice of the activation tensor along the row dimension.

The Flash MLA decode op in DeepSeek V3 B1 is the canonical example. Looking at the op:

```python
# tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:421-434
# Q tensor shape: [batch=1, num_pos=1, num_q_heads=64, head_dim+pe_dim=576]
# Sharded across a 8x1 grid → num_q_heads_per_core = 64 / 8 = 8
# Q_TILE_HEIGHT = q_tile.tile_shape[0] = 8  (line 414)
```

Each core's Q shard is naturally 8 rows tall — one position, eight heads per core, 576 columns wide (later split into 512-wide KV-PE and 64-wide RoPE pieces). The "8" is not a tuning knob the kernel author picked; it is forced by the geometry of the model (64 query heads) divided by the grid width assigned (8 cores). Increasing the grid width would shrink it further; on a wider deployment, you would see 4 heads per core, or 2.

Now consider what happens if the kernel uses a canonical 32x32 tile. The Q shard is 8 rows; rows 8–31 of each tile must be zero-padded. Concretely:

- **Dest waste**: the 32x32 tile occupies one dest slot; 24 of its 32 rows are zeros. 24/32 = **75% of the dest rows are doing nothing useful**.
- **L1 waste per Q tile**: a BF16 32x32 tile is 2,048 bytes; the meaningful payload is 8x32 BF16 = 512 bytes. 1,536 bytes per tile per CB page is padding.
- **Matmul cycle waste**: the matmul FPU walks all four faces of the LHS tile. With 8 rows, only the top half of two faces is real; the bottom 8 rows of those faces and *all* of the lower two faces are multiplying zeros into the result.

The 75% waste compounds across every CB that carries Q-derived data: the input Q buffer, the attention mask, the softmax statistics (max and sum), the per-step partial output, and the running accumulator. In Flash MLA this is at least five CBs, all naturally sized to match Q.

---

## 2. The Hardware Solution: Sub-Tile Geometries

Tensix supports sub-32-row tiles natively. The unpacker, packer, and math FPU/SFPU can be programmed with `face_r_dim` and `num_faces` settings that describe a tile with fewer than 32 rows or fewer than 4 faces. The validator in `tt_metal/tt_metal/tt-llk/common/tensor_shape.h:87-94` defines the legal space:

```cpp
// tt_metal/tt_metal/tt-llk/common/tensor_shape.h:87-94
bool validate_tensor_shape_tile_dependent_ops_(const TensorShape &tensor_shape) {
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) &&
           (face_c_dim == 16);
}
```

For Flash MLA's 8x32 Q tile, this resolves to `face_r_dim = 8`, `num_faces_r_dim = 1`, `num_faces_c_dim = 2` — one row of two 8x16 faces. The total face count is 2, satisfying the `{1, 2, 4}` constraint. The `face_c_dim = 16` constraint is non-negotiable: there is no escape hatch for narrow-column tiles. The column dimension can only be reduced by dropping faces (32→16), not by shrinking face width.

The one exception worth flagging: a 16x16 (single-face) by 16x16 (single-face) matmul is explicitly unsupported by the unpacker, asserted at `tt_metal/tt_metal/tt-llk/llk_lib/llk_unpack_AB_matmul.h:203`:

```cpp
// tt_metal/tt_metal/tt-llk/llk_lib/llk_unpack_AB_matmul.h:203
LLK_ASSERT(!(unpA_num_faces == 1 && unpB_num_faces == 1), "16x16 by 16x16 matmul is not supported");
```

The hardware has every other tiny-tile combination, but the one-face-times-one-face matmul corner case is excluded — useful to know when designing the inner-product step of a tiny-tile chain.

---

## 3. Production Evidence: Flash MLA's 8x32 Q Pipeline

Flash MLA constructs the Q-side CB chain around the 8x32 geometry. From `tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:538-555`:

```python
# tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:538-555
q_tiny_tile  = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))     # (8, 32)
k_tile_obj   = ttnn.Tile((K_TILE_HEIGHT, TILE_WIDTH))     # (32, 32)
stats_tile   = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))     # (8, 32)

q_tile_size      = q_tiny_tile.get_tile_size(q_df)        # 512 B (BF16, 8x32)
k_tile_size      = k_tile_obj.get_tile_size(k_df)         # 1,088 B (BFP8, 32x32)
mask_tile_size   = q_tile_size                            # 512 B
stats_tile_size  = stats_tile.get_tile_size(stats_df)
```

The K and V tiles stay 32x32 — they are the "wide" side of the operation, with the KV cache contributing many rows per query position. But every CB that mirrors Q geometry shrinks to 8x32, which is one face's worth of rows split across two faces of width.

The L1 footprint comparison is what makes tiny tiles a hard requirement, not an optimization. Pulling the Q-side CB sizing from `/home/tt-admin/aperezvicente/llk_research/comprehensive_understanding_of_deepseek_v3_b1/ch05_multi_head_latent_attention_deep_dive/03_flash_mla_decode.md:245-254`:

| CB name         | tile geometry | pages | L1 size      | role                          |
|-----------------|---------------|-------|--------------|-------------------------------|
| cb_q_in         | 8x32          | 18    | 9,216 B      | input Q shard                 |
| cb_mask         | 8x32          | 1     | 512 B        | attention mask                |
| cb_ms_in        | 8x32          | 3     | 1,536 B      | softmax max/sum stats         |
| cb_out_o        | 8x32          | 48    | 24,576 B     | per-step partial output       |
| cb_interm_out   | 8x32          | 16    | 8,192 B      | running accumulator           |
| **Q-side total** |              |       | **~44 KB**   |                               |

If those same CBs were sized for 32x32 tiles — same page counts, 4x larger pages — the Q-side L1 budget would be:

| CB name         | tile geometry | pages | L1 size      |
|-----------------|---------------|-------|--------------|
| cb_q_in         | 32x32         | 18    | 36,864 B     |
| cb_mask         | 32x32         | 1     | 2,048 B      |
| cb_ms_in        | 32x32         | 3     | 6,144 B      |
| cb_out_o        | 32x32         | 48    | 98,304 B     |
| cb_interm_out   | 32x32         | 16    | 32,768 B     |
| **Q-side total** |              |       | **~176 KB**  |

The difference is 132 KB per core. On a Blackhole P150 with roughly 1.3 MB of L1 budget per core (after subtracting reserved regions), that's about 10% of the entire L1 budget reclaimed on the Q side alone — not counting the K-side savings if the same logic propagated to other narrow tensors, and not counting the matmul-cycle savings.

The dest-register story is more nuanced. As covered in Chapter 2.3, dest slots are tile-granular: an 8x32 tile occupies the same dest slot as a 32x32 tile. Tiny tiles **do not reduce dest pressure** the way they reduce L1 pressure. The motivation is bandwidth and L1 capacity, not dest fanout. This matters when reasoning about which optimizations tiny tiles enable: they let you fit more pages per CB and reduce noc traffic per tile, but they do not let you keep more tiles live in dest simultaneously.

A second production example is **RoPE** (rotary position embedding). As discussed in Chapter 5.3, the RoPE op derives its output tile shape from the input shard. If `shard_shape[0] = num_q_heads_per_core = 8`, RoPE's output tile is 8x32. This is forced consistency, not choice: if RoPE writes 32x32 tiles into a CB that Flash MLA reads as 8x32, the page-size mismatch causes `cb_wait_front` to hang indefinitely — the consumer is waiting for 512-byte pages, but the producer is delivering 2,048-byte pages, and the front pointer never advances by what the consumer expects.

Two other places tiny tiles show up in production:

- **Sampling / argmax narrow outputs.** A top-1 sampling kernel emits one row per request. Padding to 32 rows wastes 31/32 = 97% of the output tile.
- **Gated local-reduce intermediates** (Chapter 5.5). The `local_reduce` and `reduce_to_one_b1` ops have intermediate buffers sized to match the activation shape they operate on. With an 8x32 activation, the reduction-gate intermediate must also be 8x32; sizing it to 32x32 silently breaks the chain.

---

## 4. The Propagation Burden

Choosing a tiny tile is not a local change. Tile geometry must propagate consistently through:

- **Every CB descriptor** in the program (page size = tile size × elements; one mismatch hangs the kernel).
- **Every kernel's compile-time args**: `TILE_HEIGHT`, `FACE_R_DIM`, `NUM_FACES` are baked into the unpacker, packer, and math configuration. A producer with `FACE_R_DIM=8` writing to a consumer with `FACE_R_DIM=16` will read garbage from L1.
- **SFPU iteration counts**. SFPU ops process a tile face-by-face; the iteration count N in calls like `llk_math_eltwise_unary_sfpu_sigmoid<approx_mode, false, N>()` must equal the number of faces in the tile. For an 8x32 tile with 2 faces, `N = 2`; for a 32x32 tile, `N = 4`. Getting this wrong does not assert — it just processes the wrong number of faces and produces silently wrong results.
- **Every consumer op's input/output tile geometry**. If Flash MLA produces 8x32 partials and feeds them into a downstream all-gather that expects 32x32, the all-gather will treat one of the producer's tiles as 1/4 of a tile and stride incorrectly.

The most common tiny-tile bug is the page-size mismatch on `cb_wait_front`. The producer pushes a tile to a CB sized for 8x32 pages (512 B). The consumer was templated against 32x32 (2,048 B) and waits for one page. The CB has enough bytes for a 32x32 page after the producer writes four 8x32 tiles, but the consumer's view of the layout — where row 0 of the tile lives in L1, where row 16 starts — is wrong because the unpacker is configured for 8-row faces. The kernel either hangs (waiting for more pages than the producer will ever send), or it doesn't hang and silently produces corrupted attention scores.

This is the central reason tiny tiles deserve a guide of their own. The hardware capability is straightforward; the discipline of threading one consistent geometry through twenty CB descriptors, eight kernel CT-arg blocks, and four downstream ops is where every real-world tiny-tile project burns its time. Subsequent chapters work through that thread one layer at a time, starting from the LLK primitives in Chapter 2 and building up to the op-level orchestration that Flash MLA represents.
