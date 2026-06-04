# Tile Shape Propagation Through a Program

Tile geometry is not a property of a single op — it is a property of the dataflow graph. A producer op writes its output with some `ttnn.Tile(H, W)`, and every downstream consumer that reads that data must agree on the same `(H, W)` (and on the derived face shape, `tile_hw`, and L1 byte size) unless an explicit re-tiling step intervenes. When this agreement holds, kernels page through L1 correctly; when it breaks, `cb_wait_front` either hangs (consumer waits for more bytes than the producer pushed) or silently corrupts (consumer reads past the page boundary into the next tile).

**Prerequisites:** Chapter 4, Section "The `ttnn.Tile` Constructor and `get_tile_size()`" (file 1); Chapter 4, Section "Building CB Descriptors from Sharded Tensors" (file 2). Familiarity with the BFP8 exponent overhead computation and the `TileDescriptor { height, width, transpose }` wrapper is assumed.

## The propagation rule

Stated precisely:

> For any CB *c* connecting a producer kernel *P* to a consumer kernel *C*, the `TileDescriptor` stored in *c*'s `CBFormatDescriptor` is observed by both sides. The page size that *P* uses for `cb_push_back` and the page size that *C* uses for `cb_wait_front` are both derived from this single descriptor. Therefore the tile geometry that *P* emits and the tile geometry that *C* expects must be the same object — or two `TileDescriptor`s that compare equal field-by-field.

There is no per-side override. There is no implicit truncation or padding at the CB boundary. If *P* writes 8x32 BFP8 tiles (272 bytes each, see Chapter 4 Section 1) and *C* expects 32x32 BFP8 tiles (1088 bytes each), `cb_wait_front(1)` on *C* will block until 1088 bytes are available, which never happens for a one-tile push from *P*. The kernel hangs.

## How CBHandle chains enforce the rule implicitly

The cleanest case is when a consumer's input CB is constructed by `cb_descriptor_from_sharded_tensor()` against the same tensor that the producer's output CB was constructed against. The host-side flow is:

```python
# host op-emit (producer)
output_cb = ttnn.cb_descriptor_from_sharded_tensor(cb_out, output_tensor)

# host op-emit (consumer, later in the same program)
input_cb  = ttnn.cb_descriptor_from_sharded_tensor(cb_in,  output_tensor)
```

Both calls reach into `output_tensor.tensor_spec().tile()` (see `tensor_utils.cpp:19–43`) and pull out the same `Tile` object. Both then build a `TileDescriptor(tile)` whose `height`, `width`, and `transpose` are identical. Both ask `output_tensor.buffer()->aligned_page_size()` for the page size, which is itself a function of the tile and the data format. The two `CBFormatDescriptor`s are not literally the same C++ object, but they compare equal field-by-field, and the page-size arithmetic in the kernels lines up.

This is the chain in action in the RoPE → Flash MLA path. RoPE's op-emit declares its output tile from the shard:

```python
# tt-blaze/.../deepseek_v3_b1/micro_ops/rope/op.py:121-122
num_q_heads_per_core = shard_shape[0]
tile = ttnn.Tile((num_q_heads_per_core, ttnn.TILE_SIZE))
```

For an 8-head-per-core shard this yields an 8x32 tile, attached to RoPE's output tensor's spec. Flash MLA then consumes that tensor:

```python
# tt-blaze/.../deepseek_v3_b1/micro_ops/flash_mla/op.py:713
ttnn.cb_descriptor_from_sharded_tensor(cb_q_in, input_tensor_q)
```

`input_tensor_q` *is* RoPE's output tensor. Flash MLA's `cb_q_in` therefore carries an 8x32 `TileDescriptor` without anyone in Flash MLA's op-emit having to spell that out. Q stays 8x32 all the way through Flash MLA's compute kernel (the kernel preserves Q's row count), so the implicit propagation suffices for the entire Q half of the dataflow.

The K side flows the same way: K's input tensor has a 32x32 `Tile` in its spec, `cb_descriptor_from_sharded_tensor(cb_k_in, input_tensor_k)` lifts that into the CB, and the matmul kernel sees a 32x32 LHS face count without any manual reconciliation.

## Where the chain breaks down: hand-constructed CB descriptors

The implicit chain only protects CBs whose `CBFormatDescriptor` came out of `cb_descriptor_from_sharded_tensor()`. The moment an op needs a CB that is *not* backed by a sharded tensor — most commonly an intermediate scratch CB internal to one fused op — the author has to construct a `CBFormatDescriptor` by hand and pass an explicit `TileDescriptor`. From this point on, nothing in the host API checks that the hand-built descriptor matches the producer's geometry. It is a free-form declaration.

Flash MLA's intermediate stats CBs are the canonical example. The op needs a small CB to carry per-row softmax statistics (running max, running sum) between the inner matmul/softmax loop iterations. There is no host-side tensor backing this CB; it lives entirely in L1 between two compute kernels. The op-emit declares the tile shape inline:

```python
# tt-blaze/.../deepseek_v3_b1/micro_ops/flash_mla/op.py:538
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))   # (8, 32)
...
# op.py:552
q_tile_size = q_tiny_tile.get_tile_size(q_df)
...
# stats CBs (intermediate output + interm scratch), op.py:737-757
stats_tile_descriptor = ttnn.TileDescriptor(stats_tile)   # wraps 8x32
ttnn.CBDescriptor(
    total_size=out0_t * stats_tile_size,
    core_ranges=core_grid,
    format_descriptors=[
        ttnn.CBFormatDescriptor(cb_out_o,      stats_df, stats_tile_size, stats_tile_descriptor),
        ttnn.CBFormatDescriptor(cb_interm_out, stats_df, stats_tile_size, stats_tile_descriptor),
    ],
)
```

(See `op.py:763–771` for the exact aliasing block.) Two things are happening:

1. The 8x32 geometry is asserted twice — once for `cb_out_o`, once for `cb_interm_out` — by reusing the same `stats_tile_descriptor` Python object. If someone refactored one line to use a different descriptor, the aliasing would be silently wrong: the two CBs would still share the same L1 region (via `CBDescriptor`'s `total_size` + `core_ranges`), but the kernel would page-step at two different strides depending on which CB index it referenced.
2. There is no upstream tensor to derive the 8x32 from. The author had to *know* — from looking at the Q path and the kernel's row-stationary plan — that the stats tiles must be 8 rows tall to match Q's 8-row tiles. Nothing in the host API will catch a typo here.

This is the failure mode to watch for. If a kernel author writes 32x32 instead of 8x32 in a hand-constructed `TileDescriptor`, three things change at once: `tile_hw` quadruples, `face_shape` flips from (8, 16) to (16, 16), and `partial_face` flips from 1 to 0. The BFP8 exponent overhead doubles (from 16 bytes to 64 bytes), so the page size changes from 272 to 1088. The producer kernel pushes 272-byte pages; the consumer waits for 1088-byte pages; the program hangs on the first `cb_wait_front`.

## Decision tree: which CB descriptors must be re-derived when a producer's tile changes?

When you change a producer op's output tile geometry — say, you switch RoPE from 8x32 output to 4x32 output by changing the shard's `num_q_heads_per_core` from 8 to 4 — you need to walk the downstream graph and update every CB descriptor that does not inherit its `TileDescriptor` automatically. The walk has three cases:

**Case 1: Consumer CB is built via `cb_descriptor_from_sharded_tensor()` against the producer's output tensor.** Nothing to do. The new tile is in the tensor's spec; the CB picks it up. Example: Flash MLA's `cb_q_in` automatically becomes 4x32 when RoPE's output becomes 4x32.

**Case 2: Consumer constructs a separate CB (not from a tensor), but the CB's role is to carry data that is element-wise aligned with the producer's output rows.** Update the hand-constructed `TileDescriptor` to match the new producer geometry. Example: Flash MLA's `cb_out_o` / `cb_interm_out` stats CBs use the same row count as Q. If Q drops to 4 rows, the stats tiles must drop to 4 rows too, otherwise the per-row stats won't line up with the per-row Q reductions in the compute kernel.

**Case 3: Consumer explicitly re-tiles.** This is rare in tt-blaze and does not appear in Flash MLA; in the whole Flash MLA op the data flow is "Q stays 8x32, K stays 32x32, O is emitted as 8x32" with no intermediate re-tiling. When it does occur (e.g. a transpose-and-retile op), the consumer constructs a fresh `Tile` and a fresh `TileDescriptor`, builds its own output CB from those, and the propagation walk restarts from that consumer as a new producer.

The Flash MLA op-emit is a useful reference for what "no re-tiling" looks like in practice: every CB in the op either comes from `cb_descriptor_from_sharded_tensor()` (case 1) or is hand-built with `stats_tile_descriptor` (case 2). There is no case-3 transition.

## The aliasing pattern and why it is safe

A subtlety in the Flash MLA snippet above is that `cb_out_o` and `cb_interm_out` share the same L1 region — one `CBDescriptor` with `total_size = out0_t * stats_tile_size` carries two `CBFormatDescriptor`s. Aliasing CBs is a memory optimization: the compute kernel uses `cb_interm_out` during the inner Flash attention loop to scratch running stats, and once the loop completes the same bytes are reinterpreted as `cb_out_o` for the writer to drain to DRAM.

This is only safe because the two format descriptors carry the same `TileDescriptor`. Both views agree on `tile_hw = 256` (8 × 32), the same `num_faces = 2`, the same `partial_face = 1`, and therefore the same page size and the same stride for `cb_push_back` / `cb_pop_front`. If the two aliased format descriptors had different tile geometries, the kernel would step through the same L1 bytes at two different strides depending on which CB index it referenced, which is a memory corruption hazard with no warning.

The general rule: aliased CBs (multiple `CBFormatDescriptor`s in one `CBDescriptor`) must share a `TileDescriptor`. The Flash MLA op-emit enforces this by reusing the same Python `stats_tile_descriptor` variable across both format descriptors; static checkers can audit this by confirming object identity rather than just field equality.

## Debugging checklist

When a tiny-tile program hangs or produces garbage and you suspect a tile-geometry mismatch, the propagation-aware checklist is:

1. **Producer output tile.** Find the producer op's output tensor and read its `tensor_spec().tile()`. This is the ground truth — the geometry every consumer must agree with.
2. **Tensor-backed consumer CBs.** For each consumer CB built via `cb_descriptor_from_sharded_tensor()`, confirm the call argument is the producer's output tensor (not a stale copy with a different `Tile`). If the host has multiple tensors with similar names, this is a frequent typo source.
3. **Hand-constructed consumer CBs.** For each consumer CB built via `ttnn.CBFormatDescriptor(..., tile=tile_descriptor)`, confirm `tile_descriptor.height` and `tile_descriptor.width` equal the producer's output `Tile.height` / `Tile.width`. Pay attention to derived quantities the kernel will compute on its own (`face_shape`, `num_faces`, `partial_face`) — if the kernel hardcodes a face shape that disagrees with the descriptor, the kernel-side computation will diverge from the host-side page size.
4. **Aliased CBs.** For each `CBDescriptor` carrying multiple `CBFormatDescriptor`s, confirm all format descriptors share the same `TileDescriptor` (ideally the same Python object, not just equal fields).
5. **Page-size cross-check.** For each CB, recompute `tile.get_tile_size(dtype)` by hand using the formula from Chapter 4 Section 1 and compare to what the kernel believes its page size to be (typically wired through a CT arg; see Chapter 4 Section 4). A page-size mismatch between host and kernel is the dominant root cause of `cb_wait_front` hangs in tiny-tile programs.

If steps 1–5 all check out, the bug is not a tile-shape propagation issue and you should look elsewhere — at face traversal order in the kernel, at the BFP8 exponent placement, or at the matmul output configuration (see Chapter 5).