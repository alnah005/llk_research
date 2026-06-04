# 5.5 Local Reduce and Gated Paths

`LocalReduce` and `GatedReduce` are the simplest tiny-tile consumers in DeepSeek V3 B1: they sum a stack of input tiles in a single CB to one output tile, with an optional SiLU. They illustrate the most important property of well-written tiny-tile micro-ops — the kernel body is fully *tile-agnostic*. The whole 8x32-vs-32x32 distinction lives in the CB descriptor and never appears in the compute code. The same `add_tiles` / `silu_tile` pipeline drives both shapes. This file walks the kernels, the fused-op wrappers, and the dest-occupancy argument for why these reductions specifically want tiny tiles when fed by gate-path producers.

**Prerequisites:** Chapter 2 (LLK init/compute/pack patterns and `acc_to_dest` semantics), Chapter 3 (tile descriptors and CB format descriptors), Chapter 5 Sections 1–4 (Flash MLA tiny-tile geometry, Q-tile sharding, and the tree-reduction CB chain).

## 5.5.1 `LocalReduce`: a tile-agnostic accumulator

`LocalReduce` is documented as "Element-wise sum reduction of N tiles with optional SiLU" — see `models/demos/deepseek_v3_b1/unified_kernels/local_reduce.hpp:21-34`. It is used by MoE shared-expert and dense-MLP paths after an inner-dim matmul shard, where each contributing core has produced its partial and the partials need to be summed into a single output tile on the consumer core.

The compile-time interface is two scalars:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/local_reduce.hpp:38-42
template <uint32_t NumTiles, bool ApplySilu>
struct ComputeCTArgs {
    static constexpr uint32_t num_tiles = NumTiles;
    static constexpr bool apply_silu = ApplySilu;
};
```

There is no `tile_r_dim_` here, no face count, no per-tile geometry. The kernel only needs to know *how many tiles to consume* and *whether to fuse SiLU*. The tile geometry rides in the CB descriptor that the host attaches to `in_cb` and `out_cb`.

The compute body is correspondingly compact:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/local_reduce.hpp:69-110
// Initialize operations before waiting for data
reconfig_data_format<false, true>(args.in_cb, args.in_cb);
pack_reconfig_data_format<true>(args.out_cb);
add_tiles_init(args.in_cb, args.in_cb, true /* acc_to_dest */);
if constexpr (apply_silu) {
    silu_tile_init();
}

// Wait for all input tiles
cb_wait_front(args.in_cb, num_tiles);

// Reserve output
cb_reserve_back(args.out_cb, 1);

// Acquire dest register
tile_regs_acquire();

// Sum all tiles using acc_to_dest mode
// acc_to_dest=true means: DST[0] = A + B + DST[0]
// DST accumulator starts at zero for each tile position
for (uint32_t i = 0; i < num_tiles; i += 2) {
    add_tiles(args.in_cb, args.in_cb, i, i + 1, 0);
}

// Optionally apply SiLU activation
if constexpr (apply_silu) {
    silu_tile(0);
}

// Commit and wait for compute
tile_regs_commit();
tile_regs_wait();

// Pack result
pack_tile(0, args.out_cb);

// Release dest register
tile_regs_release();

// Pop inputs and push output
cb_pop_front(args.in_cb, num_tiles);
cb_push_back(args.out_cb, 1);
```

A few points worth pulling out:

- **Pairwise `acc_to_dest`.** The third template argument to `add_tiles_init(...)` is `true`, meaning each `add_tiles` accumulates into `DST[0]` instead of overwriting it (`local_reduce.hpp:72`). With `DST[0]` zero-initialized inside the acquire/release scope, the loop `add_tiles(in, in, i, i+1, 0)` for `i = 0, 2, 4, ...` performs `DST[0] += in[i] + in[i+1]`, accumulating all `num_tiles` partials into a single dest slot. Because of this, `num_tiles` must be even — the same constraint shows up in the host wrapper's `assert group1_num_tiles % 2 == 0` (see `fused_ops/gated_local_reduce/op.py:85-87`).
- **One dest slot for any tile geometry.** A 32x32 tile occupies 4 faces of `DST`. An 8x32 tile occupies 1 face. The kernel does not care: `add_tiles` and `silu_tile` walk whatever the CB descriptor says is in the page. The same `pack_tile(0, args.out_cb)` writes the correct row count back out.
- **No iteration-count gymnastics.** Unlike the SFPU sigmoid path in the unified matmul that needs `<approx, false, 2>` to fix the iteration count to "two 16-row faces" for a 1x32 or 8x32 tile (see `unified_kernels/matmul.hpp:166-178` and Section 5.4), `silu_tile` here is invoked with no template-level face hint. The SFPU runs across the whole dest slot; for an 8x32 tile that is one face's worth of work, for a 32x32 it is four faces — both are correct because the `pack_tile` step truncates to the declared tile height.

This is the cleanest statement in the whole codebase of "tiny tiles are a CB-level concern, not a compute-level one." If you have an LLK reduction kernel that you would like to make tiny-tile friendly, this is the pattern.

## 5.5.2 When the consumer is the tile-geometry source

Because the kernel does not name the tile geometry, the question of "do we run with 8x32 or 32x32 tiles?" is answered upstream by whoever defined the input CB. The unit test `tests/unit_tests/test_local_reduce.py` exercises this directly with parametrized tile shapes:

```python
# tests/unit_tests/test_local_reduce.py:27-41
@pytest.mark.parametrize(
    "tile_h, tile_w, num_tiles, apply_silu",
    [
        # Without SiLU
        (32, 32, 2, False),   # Standard tiles, 2 inputs
        (32, 32, 4, False),   # Standard tiles, 4 inputs
        (32, 32, 8, False),   # Standard tiles, 8 inputs
        (16, 16, 4, False),   # Small tiles, 4 inputs
        (1, 32, 6, False),    # Tiny tiles, 6 inputs
        # With SiLU
        (32, 32, 2, True),    # Standard tiles, with SiLU
        (32, 32, 4, True),    # Standard tiles, 4 inputs, with SiLU
        (16, 16, 4, True),    # Small tiles, with SiLU
        (1, 32, 6, True),     # Tiny tiles, with SiLU
    ],
)
def test_local_reduce(device, tile_h, tile_w, num_tiles, apply_silu):
    tile = ttnn.Tile([tile_h, tile_w])
    ...
    ttnn_input = ttnn.from_torch(
        torch_input_stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )
```

The same kernel binary handles `(32, 32, 8)`, `(16, 16, 4)`, and `(1, 32, 6)` — only the `ttnn.Tile([tile_h, tile_w])` passed to `ttnn.from_torch` differs. The CB allocated for that tensor inherits its `tile` field, and the kernel reads/writes pages of that size.

This is also where tiny tiles earn their L1 keep on the reduce path. Compare two configurations that produce the same logical output (one 32-row activation row):

| Tile geom | tiles needed for 8-row activation | bytes per CB slot (bf16) | wasted |
|---|---|---|---|
| 32x32 padded | 1 tile, padded with 24 rows of zeros | 2048 B | 75% |
| 8x32 tiny | 1 tile, full | 512 B | 0% |

A reduction over `N` such activations multiplies the savings by `N`: `LocalReduce` with `num_tiles=6` and 32x32 padded tiles holds 6 * 2048 = 12 KB on the input CB; with 8x32 tiles it holds 6 * 512 = 3 KB. The dest budget is identical (one dest slot either way) but L1 footprint is 4x smaller and `pack_tile` writes 4x less data back to L1.

There is also a face-view optimization layered on top (`test_local_reduce_face_view`, `tests/unit_tests/test_local_reduce.py:117-238`): when the input is `N*8` tiles of `[1, 32]` (i.e. `N * 256` elements grouped into N faces), the op repackages the CB as `N` tiles of `[16, 16]`, cutting `add_tiles` calls from `N*4` to `N/2`. The kernel is unchanged — the host just rewrites the CB tile descriptor. This is the same principle: geometry is metadata, the compute is invariant.

## 5.5.3 `GatedReduce`: two reductions and a multiply

`GatedReduce` is the building block for the gated-MLP shape `SiLU(reduce(group1)) * reduce(group2)`. From `unified_kernels/gated_reduce.hpp:22-37`:

```cpp
// GatedReduce micro-op: SiLU(sum(group1)) * sum(group2)
// Produces k_num_tiles output tiles (one per K iteration).
// Each iteration consumes tiles_per_k tiles from each group CB.
```

Compile-time args:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/gated_reduce.hpp:42-46
template <uint32_t TilesPerK, uint32_t KNumTiles>
struct ComputeCTArgs {
    static constexpr uint32_t tiles_per_k = TilesPerK;
    static constexpr uint32_t k_num_tiles = KNumTiles;
};
```

The body is a `k_num_tiles`-loop that, per K position, does (1) reduce + SiLU on `group1`, (2) reduce on `group2`, (3) multiply them — using a 2-tile intermediate CB that is reused every iteration (`gated_reduce.hpp:83-134`). The shape echoes `LocalReduce`: pairwise `add_tiles(..., true /* acc_to_dest */)`, then `silu_tile(0)` only on group1, then `mul_tiles` between the two reduced intermediates.

The intermediate CB layout is the key tiny-tile signal:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/gated_reduce.hpp:32-36
//   group1_cb:   Gate partials (tiles_per_k tiles consumed per iteration)
//   group2_cb:   Up partials (tiles_per_k tiles consumed per iteration)
//   intermed_cb: Intermediate buffer (2 tiles, reused each iteration)
//   out_cb:      Output (1 tile produced per iteration)
```

The host-side wrapper allocates that intermediate CB with the exact tile descriptor of the *output* tensor:

```python
# models/demos/deepseek_v3_b1/fused_ops/gated_local_reduce/op.py:101-123
output_tile = output_tensor.tile
tile_h, tile_w = output_tile.tile_shape
data_format = output_tensor.dtype
tile_size = output_tile.get_tile_size(data_format)
tile_descriptor = ttnn.TileDescriptor(tile_h, tile_w, False)

# CB descriptors for inputs (backed by tensors)
in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in0_cb, input_tensor_group1)
in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, input_tensor_group2)

# Intermediate CB: 2 tiles (group1 result + group2 result)
intermed_format = ttnn.CBFormatDescriptor(
    buffer_index=intermed_cb,
    data_format=data_format,
    page_size=tile_size,
    tile=tile_descriptor,
)
intermed_cb_descriptor = ttnn.CBDescriptor(
    total_size=2 * tile_size,  # 2 tiles
    core_ranges=all_cores,
    format_descriptors=[intermed_format],
)
```

So if the gate path produces 1x32 (or 8x32) tiles upstream, the intermediate CB is sized at `2 * tile_1x32_size = 2 * 64 B` (bfp8) or `2 * 128 B` (bf16), not the 4 KB / 8 KB of a 32x32 standard tile. That is the load-bearing point for this section: the gate path's narrow activation is preserved end-to-end through the reduce, the multiply, and into the consumer. If you had to pad to 32x32 here, the intermediate buffer would be 32x larger than necessary and the output written back to the consumer would also be padded — and then the consumer would also have to be padded, propagating waste forward.

Two host-level invariants enforced at the call site (`fused_ops/gated_local_reduce/op.py:84-99`):

- Both groups must have even tile counts of at least 2 (mirrors the pairwise `add_tiles` loop, which steps by 2 and assumes `tiles_per_k >= 2 && tiles_per_k % 2 == 0` — see the `static_assert` at `gated_reduce.hpp:74`).
- All three CB data formats must match. `binary_op_init_common` is called once outside the loop, and the kernel only reconfigures the data format between `group1_cb` and itself (`gated_reduce.hpp:78-80`) — there is no per-iteration format reconfigure for `group2_cb` or `intermed_cb`, so they must already match.

## 5.5.4 `down_proj`: where mixed tile geometry actually lives

`down_proj` (the fused `Mcast1 + Mcast2 + Matmul + ResidualAdd + Gather`) is the consumer that pins the output of `GatedReduce` to a specific tiny-tile shape. From `fused_ops/down_proj/op.py:163-167`:

```python
# Tile definitions
TILE_1x32 = ttnn.Tile((1, 32))
tile_1x32_size = TILE_1x32.get_tile_size(data_format)
```

The whole op operates on 1x32 tiles end-to-end:

- The matmul output CB on the 112 matmul cores uses `TILE_1x32` (`down_proj/op.py:382-392`).
- The mcast-2 destination (residual add input) uses `TILE_1x32` (`down_proj/op.py:402-413`).
- The residual-add output CB uses `TILE_1x32` (`down_proj/op.py:415-426`).
- The gather destination on the (12, 9) sender uses `TILE_1x32` via the tensor-backed CB descriptor (`down_proj/op.py:394-395`).

What *is* mixed-geometry in `down_proj` is the matmul itself: `matmul_in0` (the mcast destination CB) uses the *input tensor's* tile shape via `input_tile.get_tile_size(data_format)` (`down_proj/op.py:199-206, 365-376`). The matmul weights (CB 2) come from the weights tensor's tile geometry (standard 32x32, `down_proj/op.py:378-379`). The matmul output is a 1x32 tile per `out_w` column.

In short, the in0 is a tiny tile (matching whatever the activation pipeline is producing — 1x32 for the standalone `down_proj`, or face-tiles when the fused `gated_local_reduce_down_proj` form is used), in1 is a standard 32x32 weight tile, and the output is 1x32. This is the same `tile_r_dim_`-aware DRAM-streaming matmul pattern that Section 5.4 covered — `down_proj` is the host-side fused op that wires it up to a network of 112 cores plus a 130-core mcast.

The fused `GatedLocalReduceDownProj` op (`fused_ops/gated_local_reduce_down_proj/op.py`) chains gate-and-up source cores → input-gather → `GatedReduce` on the (12, 9) sender → `DownProj` mcast/matmul/gather, all in one program. The CB layout (13 CBs total, `fused_ops/gated_local_reduce_down_proj/op.py:19-33`) preserves the tiny-tile shape from the source cores all the way through the reduce into the matmul's in0 — the mcast source CB (CB 3) uses the reduce-output tile size:

```python
# fused_ops/gated_local_reduce_down_proj/op.py:556-567
mcast_src_format = ttnn.CBFormatDescriptor(
    buffer_index=ctx.mcast_src_cb,
    data_format=ctx.data_format,
    page_size=ctx.reduce_tile_size,
    tile=tile_desc,
)
mcast_src_cb_descriptor = ttnn.CBDescriptor(
    total_size=ctx.mcast_src_num_pages * ctx.reduce_tile_size,
    core_ranges=ctx.mcast_gather_core_grid,
    format_descriptors=[mcast_src_format],
)
```

Here `reduce_tile_size` and `tile_desc` come from either the face-view path (16x16 face tile) or the input tile (1x32), and `mcast_src_num_pages` is `1` in face-view mode versus `k_num_tiles` otherwise (`fused_ops/gated_local_reduce_down_proj/op.py:249-263`). The mcast destination CB on the 130-core grid, by contrast, always uses the *input* tile geometry — it is what the matmul in0 consumes (`fused_ops/gated_local_reduce_down_proj/op.py:569-582`). So the program juggles three coexisting tile shapes (face tile for the reduce intermediate, input tile for the activation, 1x32 for the matmul output) without ever quoting them inside a compute kernel.

## 5.5.5 The dest-occupancy story

A useful mental model for tying this section back to the rest of Chapter 5: think about `DST` register slots, not L1 bytes.

For a standard 32x32 tile, one dest slot holds 4 faces of 16x16. For an 8x32 tiny tile, one dest slot holds 1 face. `add_tiles` and `silu_tile` operate on a dest slot — they walk all 4 faces on a 32x32 tile and only 1 face on an 8x32 tile.

In a multi-tile reduction:

| N partials | 8-row activation, 8x32 tiny tile | 8-row activation, 32x32 padded |
|---|---|---|
| add_tiles calls | N/2 | N/2 |
| Per-call face traversal | 1 face | 4 faces (3 are zero-pad) |
| Useful work per dest slot | 100% | 25% |
| `silu_tile` SFPU iterations | 1 face | 4 faces (3 are wasted) |
| L1 page footprint | 1x | 4x |

The reduce kernel itself does the same number of `add_tiles` calls in both modes, but on tiny tiles each call only touches the rows that hold real data. The savings show up at three levels: (1) L1 footprint of the input CB, (2) SFPU iteration count on the fused activation, and (3) bytes written by `pack_tile` to the output CB. The same argument applies to the multiply step inside `GatedReduce`. Tying this back to Section 5.1 / 5.2: this is exactly the reason the Flash MLA decode pipeline uses an 8x32 Q tile, and the reason `down_proj` continues that tile shape downstream — tiny tiles propagate forward through every consumer that doesn't insist on the standard 32x32 face count.

## 5.5.6 What to remember

- The `LocalReduce` and `GatedReduce` compute kernels have *no* tile-shape compile-time parameter. Geometry comes from the CB descriptor; the kernel handles any shape that `add_tiles` / `silu_tile` / `mul_tiles` accept.
- `acc_to_dest` and the pairwise `add_tiles(..., i, i+1, 0)` loop require `num_tiles % 2 == 0`. Host wrappers assert this; the gated kernel has a `static_assert`.
- Gate-path narrow outputs (1x32 / 8x32) drop through the reduce into the matmul's in0 with no padding, keeping CB pages 4x smaller than the 32x32 fallback and avoiding wasted SFPU work on the SiLU.
- `down_proj` is the host-side wiring that locks the post-reduce pipeline to tiny tiles (`TILE_1x32 = ttnn.Tile((1, 32))`, used for mcast-2, residual-add, and gather CBs). The matmul itself is mixed-geometry: tiny in0, standard in1, tiny out.
- `GatedLocalReduceDownProj` is the maximal fused form — input-gather + gated reduce + down-proj — and demonstrates the discipline of carrying tiny-tile geometry across a 130-core mcast program by parametrizing every CB by `input_tile`, `reduce_tile_size`, or `TILE_1x32` rather than hard-coding sizes.

For the Flash MLA tiny-tile story that produces these narrow activations upstream, see Chapter 5, Section 1 (Q-tile geometry) and Section 2 (Flash MLA CB chain). For the matmul kernel that consumes them in `down_proj`, see Chapter 5, Section 4 (DRAM-streaming matmul with `tile_r_dim_`). For the broader trade-off analysis of "when to pick tiny tiles vs. when to pad," see Chapter 6.
