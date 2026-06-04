# 5.2 Flash MLA Decode — The Mixed-Geometry CB Chain

Flash MLA decode is the most demanding tiny-tile consumer in DeepSeek V3 B1. Its compute graph stitches together 8x32 Q tiles, 32x32 BFP8 K/V tiles, and a tree-reduction pipeline whose output and intermediate buffers all use 8x32 tiles — ten circular buffers in total, two pairs of which alias the same L1 region. This section walks the CB table tile-by-tile, exposes the CB-aliasing pattern that compresses the output side of the pipeline, and quantifies the L1 budget against a hypothetical "everything is 32x32" baseline.

**Prerequisites:** Chapter 3 (tile descriptors and CB format descriptors), Chapter 4 (production op-emit anatomy), Section 5.1 (overview of where tiny tiles appear in DeepSeek V3 B1).

---

## 5.2.1 The Ten CBs

The op-emit code for Flash MLA decode declares CB IDs 0–9 explicitly so that the same integer names can be threaded into the BRISC, NCRISC, and TRISC compile-time arg packs. The IDs are not opaque slots — they are how reader, writer, and compute agree on the per-tile geometry of each buffer.

```python
# tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:557-569
# =========================================================================
# CB IDs - used by both CB descriptors and kernel compile-time args
# =========================================================================
cb_q_in = 0  # Q  input
cb_k_in = 1  # K/V cache input
cb_mask = 2  # Mask input
cb_ms_in = 3  # m/s stats input (from sender in tree reduction)
cb_out_in = 4  # output input for tree reduction
cb_out_o = 5  # output O from compute
cb_out_ms = 6  # output m/s stats from compute
cb_interm_out = 7  # intermediate output for tree reduction
cb_interm_ms = 8  # intermediate m/s stats for tree reduction
cb_out_final = 9  # final sharded output
```

Of the ten CBs, **nine use 8x32 tiny tiles** and only one — `cb_k_in` — uses standard 32x32 tiles (in BFP8 format). That ratio is the point: once the Q activation is tiny, every downstream buffer that touches Q-shaped data (mask, stats, partial O, tree-reduction inputs, final output) collapses to 8x32 as well.

Tile dimensions and the per-CB tile size are computed up front from the input tensors' tile descriptors:

```python
# tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:537-555
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))   # (8, 32)
full_tile   = ttnn.Tile((TILE_HEIGHT, TILE_WIDTH))     # (32, 32)

# All intermediate/stats tiles use tiny tile dimensions (same as Q)
im_tile     = q_tiny_tile
stats_tile  = q_tiny_tile
k_tile_obj  = full_tile

q_tile_descriptor     = ttnn.TileDescriptor(q_tiny_tile)
stats_tile_descriptor = ttnn.TileDescriptor(stats_tile)

q_tile_size    = q_tiny_tile.get_tile_size(q_df)
k_tile_size    = k_tile_obj.get_tile_size(k_df)
mask_tile_size = q_tile_size
stats_tile_size = stats_tile.get_tile_size(stats_df)
```

`get_tile_size(dtype)` is the bridge between the logical tile descriptor and the byte-level page size of the CB. For BF16 it returns 512 B for an 8x32 tile and 2,048 B for a 32x32 tile; for BFP8 a 32x32 tile is 1,088 B (1 KB data + 64 B exponents).

---

## 5.2.2 CB Table

The full layout, per Flash MLA worker core, with the values that op.py computes at descriptor-creation time:

| CB             | ID | Tile geometry         | Pages (tiles)                                                                 | L1 size   | Role                                                  |
|----------------|----|-----------------------|-------------------------------------------------------------------------------|-----------|-------------------------------------------------------|
| `cb_q_in`      | 0  | 8x32 (tiny)           | `PNHt * DHt = 1 * 18 = 18`                                                    | 9,216 B   | Q input (sharded from `CreateQHeads`)                 |
| `cb_k_in`      | 1  | 32x32 BFP8            | `Sk_chunk_t * DHt * 2 = 4 * 18 * 2 = 144`                                     | 156,672 B | K/V chunks, double-buffered                           |
| `cb_mask`      | 2  | 8x32 (tiny)           | 1                                                                             | 512 B     | Causal mask for partial chunks                        |
| `cb_ms_in`     | 3  | 8x32 (tiny)           | `PNHt * NUM_TREE_REDUCTION_STEPS = 1 * 3 = 3`                                 | 1,536 B   | Tree-reduction received m/s stats                     |
| `cb_out_in`    | 4  | 8x32 (tiny)           | `vDHt * NUM_TREE_REDUCTION_STEPS = 16 * 3 = 48`                               | 24,576 B  | Tree-reduction received O                             |
| `cb_out_o`     | 5  | 8x32 (tiny)           | `PNHt * vDHt = 16`                                                            | 8,192 B   | Compute output O                                      |
| `cb_out_ms`    | 6  | 8x32 (tiny)           | `PNHt = 1`                                                                    | 512 B     | Compute output m/s stats                              |
| `cb_interm_out`| 7  | 8x32 (tiny)           | `vDHt = 16` (aliased with `cb_out_o`)                                         | 0 B added | Intermediate O for tree reduction                     |
| `cb_interm_ms` | 8  | 8x32 (tiny)           | `PNHt = 1` (aliased with `cb_out_ms`)                                         | 0 B added | Intermediate m/s for tree reduction                   |
| `cb_out_final` | 9  | 8x32 (tiny)           | per output sharded spec                                                       | 8,192 B   | Final sharded output                                  |

Numerical conventions: `PNHt = num_q_heads_per_core / Q_TILE_HEIGHT = 8 / 8 = 1`; `DHt = 18` head-dim tiles; `vDHt = 16` value-head-dim tiles; `Sk_chunk_t = 4`. The 8x32 tile size in BF16 is 512 B, so every "tiny" entry above is `pages * 512 B`.

The single 32x32 BFP8 K tile is **1,088 B**, so `cb_k_in` is `144 * 1088 = 156,672 B` — 75% of the per-core CB budget. Everything else is tiny.

---

## 5.2.3 CB Descriptor Construction

Two patterns appear in the op-emit:

1. **Sharded tensors** delegate the descriptor to `ttnn.cb_descriptor_from_sharded_tensor`, which reads the tile geometry off the tensor itself:

   ```python
   # tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:712-715, 786-788
   # cb_q_in: Q input (tiny tile)
   q_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_q_in, input_tensor_q)
   q_input_cb_descriptor.core_ranges = core_grid
   cb_descriptors.append(q_input_cb_descriptor)
   ...
   # cb_out_final: final sharded output
   cb_out_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_out_final, output_tensor)
   cb_descriptors.append(cb_out_descriptor)
   ```

   For `cb_q_in` this works because `input_tensor_q` was created with `ttnn.from_torch(..., tile=tiny_tile, ...)` upstream in `CreateQHeads`; the tile geometry flows through the tensor object, not as a separate kernel arg.

2. **Internal CBs** (K input, mask, tree-reduction buffers, output O and stats) are built explicitly with a tile-size byte count and a `TileDescriptor`:

   ```python
   # tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:737-758
   if grid.NUM_TREE_REDUCTION_STEPS > 0:
       # cb_out_in: output input (tiny tile)
       cb_descriptors.append(
           ttnn.CBDescriptor(
               total_size=intermed_output_tiles * stats_tile_size,
               core_ranges=core_grid,
               format_descriptors=[
                   ttnn.CBFormatDescriptor(cb_out_in, stats_df, stats_tile_size, stats_tile_descriptor)
               ],
           )
       )
       # cb_ms_in: m/s stats input (m and s are packed into single tile)
       cb_descriptors.append(
           ttnn.CBDescriptor(
               total_size=intermed_ms_tiles * stats_tile_size,
               core_ranges=core_grid,
               format_descriptors=[
                   ttnn.CBFormatDescriptor(cb_ms_in, stats_df, stats_tile_size, stats_tile_descriptor)
               ],
           )
       )
   ```

   Note that `stats_tile_descriptor` was built from the 8x32 tile object; this is what tells the LLK pack/unpack pipeline to address only two 16-row faces' worth of memory per tile rather than the standard four.

---

## 5.2.4 Tree-Reduction Buffer Sizing

The tree-reduction step count drives two of the larger tiny-tile CBs:

```python
# tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:571-577
# Intermediate output tiles for tree reduction
# With tree reduction, senders can complete their steps out of order (e.g., S5 may send
# in step 3 before S3 sends in step 2). To prevent data corruption, each tree reduction
# step uses a separate buffer slot. This requires num_tree_reduction_steps * per_step_tiles.
# Each transfer contains: output tiles (out0_t) + m/s stats (PNHt, packed into single tile)
intermed_output_tiles = out0_t * grid.NUM_TREE_REDUCTION_STEPS   # 16 * 3 = 48
intermed_ms_tiles     = PNHt   * grid.NUM_TREE_REDUCTION_STEPS   # 1  * 3 =  3
```

`NUM_TREE_REDUCTION_STEPS = 3` for the 8-batch decode topology, so `cb_out_in` holds 48 partial-O tiles (3 steps × 16 tiles per step) and `cb_ms_in` holds 3 stats tiles (one per step). At 32x32 these would consume `48 * 2048 = 96 KB` and `3 * 2048 = 6 KB`; at 8x32 they collapse to `48 * 512 = 24 KB` and `3 * 512 = 1.5 KB` — a flat 4x savings on both buffers.

The comment in the source is the load-bearing constraint: senders complete out-of-order, so the receiver cannot reuse a single slot per step. The tiny-tile geometry is what makes the "one slot per step" memory budget tractable.

---

## 5.2.5 CB Aliasing: Two Descriptors, One Buffer

The output-side CBs use a less obvious trick: `cb_out_o` and `cb_interm_out` point at the **same physical L1 region** via two `CBFormatDescriptor`s on a single `CBDescriptor`. The same pattern is repeated for `cb_out_ms` and `cb_interm_ms`:

```python
# tt-metal/models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:762-784
# cb_out_o/cb_interm_out: output O (tiny tile)
cb_descriptors.append(
    ttnn.CBDescriptor(
        total_size=out0_t * stats_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(cb_out_o,     stats_df, stats_tile_size, stats_tile_descriptor),
            ttnn.CBFormatDescriptor(cb_interm_out, stats_df, stats_tile_size, stats_tile_descriptor),
        ],
    )
)

# cb_out_ms/cb_interm_ms: output m/s stats (tiny tile, shared for both m and s)
cb_descriptors.append(
    ttnn.CBDescriptor(
        total_size=statistics_tiles * stats_tile_size,
        core_ranges=core_grid,
        format_descriptors=[
            ttnn.CBFormatDescriptor(cb_out_ms,     stats_df, stats_tile_size, stats_tile_descriptor),
            ttnn.CBFormatDescriptor(cb_interm_ms, stats_df, stats_tile_size, stats_tile_descriptor),
        ],
    )
)
```

Both descriptors in each aliased pair are constructed with the **same** `stats_tile_descriptor` (the 8x32 tile object), the **same** `stats_df` data format, and the **same** `stats_tile_size`. That identical-geometry constraint is what makes the aliasing safe: when the compute kernel emits a tile through `cb_out_o`, the LLK pack pipeline writes 8 rows × 32 columns laid out in two 16-row faces; when the tree-reduction code later reads from `cb_interm_out`, the LLK unpack pipeline reads the same two faces from the same L1 addresses.

If the two descriptors disagreed on tile geometry — say, `cb_out_o` was 8x32 but `cb_interm_out` was 32x32 — the writer and reader would step through L1 at different strides and the buffer would corrupt silently. Aliasing only works because Flash MLA decided up front that everything Q-shaped is 8x32; that decision is made once at `q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))` and propagates to every downstream descriptor.

The L1 savings from aliasing are exactly the size of the aliased CBs: `cb_interm_out` (16 tiles × 512 B = 8,192 B) and `cb_interm_ms` (1 tile × 512 B = 512 B) cost **zero additional L1** because they share storage with their `cb_out_*` counterparts.

---

## 5.2.6 L1 Budget: Tiny Q vs. Padded Q

Summing the per-core CBs as Flash MLA actually allocates them:

| Category                              | Size        |
|---------------------------------------|------------:|
| Q shard (`cb_q_in`)                   | 9,216 B     |
| Double-buffered K (`cb_k_in`)         | 156,672 B   |
| Mask (`cb_mask`)                      | 512 B       |
| Tree-reduction receive (`cb_out_in` + `cb_ms_in`) | 26,112 B |
| Output O + stats (`cb_out_o` + `cb_out_ms`, alias absorbs interm) | 8,704 B |
| Final output (`cb_out_final`)         | 8,192 B     |
| **Total**                             | **209,408 B (~204.5 KB)**  |

Now the counterfactual: assume Q had been padded to 32x32 instead. Every CB tagged "8x32 (tiny)" in the table above grows by 4x in tile size; aliasing still works (both descriptors still see matched geometry), but the absolute bytes inflate. CB-by-CB:

| CB             | Tiny (8x32) | Padded (32x32) | Delta      |
|----------------|------------:|---------------:|-----------:|
| `cb_q_in`      | 9,216 B     | 36,864 B       | +27,648 B  |
| `cb_k_in`      | 156,672 B   | 156,672 B      | 0          |
| `cb_mask`      | 512 B       | 2,048 B        | +1,536 B   |
| `cb_ms_in`     | 1,536 B     | 6,144 B        | +4,608 B   |
| `cb_out_in`    | 24,576 B    | 98,304 B       | +73,728 B  |
| `cb_out_o`     | 8,192 B     | 32,768 B       | +24,576 B  |
| `cb_out_ms`    | 512 B       | 2,048 B        | +1,536 B   |
| `cb_out_final` | 8,192 B     | 32,768 B       | +24,576 B  |
| **Total**      | **209,408 B** | **367,616 B**   | **+158,208 B (~155 KB)**  |

The padded total — roughly 360 KB — is well above the per-core L1 budget that Flash MLA has to share with the rest of the layer's working set. (The chapter overview cited "~280 KB" as a rough ceiling for "what would actually fit if you really tried"; the difference between 280 KB and 367 KB is the difference between making aggressive cuts elsewhere — smaller K double-buffer, fewer tree-reduction slots — and not running at all.)

The savings concentrate in three places:
- **Tree-reduction input (`cb_out_in`)** — 73 KB recovered. Largest single win because it scales with `vDHt * NUM_TREE_REDUCTION_STEPS = 48` tiles.
- **Output accumulators (`cb_out_o` + `cb_out_final`)** — 49 KB combined. These are the buffers the aliasing optimization is designed to compress further; padding would defeat it.
- **Q shard itself (`cb_q_in`)** — 27 KB. Modest in absolute terms but the *origin* of the cascade: if Q is padded, every Q-shaped buffer downstream has to pad too.

The mask and stats CBs save only a few KB each, but they confirm the pattern: every CB that participates in the Q chain shrinks 4x, and the only buffer immune to the savings is `cb_k_in`, whose geometry is dictated by the KV-cache tensor (32x32 BFP8) and never touches the Q activation shape.

---

## 5.2.7 What Propagates From the Tile Descriptor

Tracing one tile object — `q_tiny_tile = ttnn.Tile((8, 32))` — through the op-emit:

1. **Sharded tensor creation** (upstream in `CreateQHeads`, see Section 5.4): `ttnn.from_torch(..., tile=tiny_tile, ...)` attaches the tile geometry to the tensor.
2. **CB descriptor for `cb_q_in`** (op.py:713): `cb_descriptor_from_sharded_tensor` reads the tile geometry off the tensor and produces a `CBFormatDescriptor` with `stats_tile_size = 512 B`.
3. **CB descriptors for the entire Q chain** (op.py:744, 755, 768, 769, 780, 781): all use `stats_tile_descriptor`, which was built from `q_tiny_tile`.
4. **Compile-time args to the compute kernel** (op.py:683–705): `cb_q_in`, `cb_interm_out`, `cb_out_o`, `cb_out_ms`, `cb_interm_ms`, `cb_out_final` are passed as integer CB IDs; the *geometry* is not in the args — it is implicit in the CB's format descriptor.
5. **LLK pack/unpack** (compute kernel, see Section 5.6): the pack/unpack pipelines retrieve face count and stride from the CB's tile descriptor and address only the two 16-row faces present in the tile.

The single 8x32 declaration is the only place tile geometry lives; everything else is derivation. That property is what makes the CB-aliasing in §5.2.5 trustworthy — the two aliased descriptors *cannot* disagree on geometry because they were both initialized from the same `stats_tile_descriptor` object.

---

## 5.2.8 Takeaways for the Next Sections

- **Mixed geometry is fine.** `cb_k_in` at 32x32 BFP8 sits alongside nine 8x32 tiny-tile CBs without any geometry-conversion code; the matmul kernel that multiplies them is the topic of Section 5.6.
- **Aliasing depends on a single-source-of-truth tile descriptor.** Section 5.4 (`CreateQHeads`) and Section 5.5 (RoPE) follow the same pattern: derive a `tiny_tile` from the per-core shard shape, then thread it through every downstream CB descriptor.
- **The L1 savings are real and quantifiable.** The 158 KB recovered here is what lets `cb_k_in` keep its double-buffer at the size Flash MLA actually needs (4 chunks × 18 head-dim tiles × 2). Chapter 6 picks up this trade-off and walks through the choice of `Sk_chunk_t` against the L1 ceiling.
