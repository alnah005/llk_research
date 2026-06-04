# Math Engine with Tiny Tiles

The math engine receives data from SrcA/SrcB registers (populated by the unpacker) and emits results into the destination register. For matmul specifically, the LLK lays out a deterministic MOP (Math OPeration) program whose shape — number of inner-loop MVMUL instructions, address-modifier strides, and fidelity-phase replay count — is computed at init time from the tile geometry passed by the kernel. Tiny tiles change two of those three knobs (the inner-loop count and the address strides) but leave a third (fidelity-phase count) untouched. They also leave the destination-register slot accounting completely unchanged, which is the single most common source of misunderstanding when planning a new tiny-tile op.

**Prerequisites.** Chapter 1 (tiny-tile geometry, face count, validator constraints) and Chapter 2 Section 1 (unpacker programming and partial_face semantics).

## `_llk_math_matmul_init_<>()` and its parameter surface

The math-side matmul initializer lives at `tt_llk_blackhole/llk_lib/llk_math_matmul.h:596–621`:

```cpp
// tt_llk_blackhole/llk_lib/llk_math_matmul.h:596
template <int MATH_FIDELITY = 0, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_init_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const std::uint32_t partial_face   = 0,
    const std::uint32_t transpose      = 0,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1)
{
    LLK_ASSERT(
        !(in0_tile_r_dim == FACE_R_DIM && in0_tile_c_dim == FACE_C_DIM &&
          in1_tile_r_dim == FACE_R_DIM && in1_tile_c_dim == FACE_C_DIM),
        "16x16 by 16x16 matmul is not supported");
    ...
}
```

Two template parameters and eight runtime parameters fully determine the MOP. The template parameters select which fidelity-phase replay schedule and which throttle pattern get compiled in; the runtime parameters describe the tile geometry and the outer reuse loop bounds (`ct_dim`, `rt_dim`).

The assertion at line 607 is load-bearing: there is no MOP program for 16x16-by-16x16 matmul. A face-pair-by-face-pair matmul of two single-face tiles would degenerate into a single MVMUL with no inner loop and no reuse; the LLK declines to generate that. In practice this is a non-issue because the validator at `tt_metal/third_party/tt_llk/common/tensor_shape.h:86–99` requires `face_c_dim == 16`, and the most narrow legal tile is still 1x32 (`num_faces=2, face_r_dim=1`) — which gives two faces and therefore exercises the standard 32x32 or 16x32 MOP paths.

## Geometry flags drive the MOP inner-loop count

The MOP is built by `matmul_configure_mop<MathFidelity>()` (`llk_math_matmul.h:280–376`). It computes three booleans that classify the geometry, then selects `replay_buf_len`:

```cpp
// tt_llk_blackhole/llk_lib/llk_math_matmul.h:301-307
const bool is_in0_16x32 = (in0_tile_r_dim <= FACE_R_DIM) && (in0_tile_c_dim > FACE_C_DIM);
const bool is_in0_32x16 = (in0_tile_r_dim >  FACE_R_DIM) && (in0_tile_c_dim <= FACE_C_DIM);
const bool is_in1_16x32 = (in1_tile_r_dim <= FACE_R_DIM) && (in1_tile_c_dim > FACE_C_DIM);
const bool is_in1_32x16 = (in1_tile_r_dim >  FACE_R_DIM) && (in1_tile_c_dim <= FACE_C_DIM);

const std::uint32_t replay_buf_len =
    (is_in0_16x32 && is_in1_32x16) ? 4 :
    ((is_in0_16x32 || is_in1_32x16 || is_in0_32x16 || is_in1_16x32) ?
        (partial_face ? 4 : 8) : 16);
```

Reading this as a decision table:

| Geometry (in0 × in1)                       | partial_face | `replay_buf_len` | MVMUL inner-loop count |
|--------------------------------------------|:------------:|:----------------:|:----------------------:|
| 32x32 × 32x32 (standard)                   |       —      |        16        |           16           |
| 16x32 × 32x16                              |       —      |         4        |            4           |
| 16x32 × 32x32 (or other single-axis short) |    false     |         8        |            8           |
| same as above                              |    true      |         4        |            4           |
| 32x32 × 32x16                              |    false     |         8        |            8           |

The inner-loop count is the number of MVMUL instructions executed per outer-loop iteration. A standard 32x32-by-32x32 matmul fires 16 MVMULs (4 faces of in0 × 4 faces of in1, with one MVMUL per face-row group). A 16x32-by-32x32 matmul fires 8 — half the work — because only two faces of in0 contribute. A 16x32-by-32x16 case collapses further to 4. Note that tiny tiles in the "tall-thin" sense (e.g. 8x32, 4x32, 1x32) still have `face_c_dim==16`, so they take the same `is_in0_16x32` branch as a 16x32 tile — the sub-16 row count is handled by `partial_face` and by the unpacker's face-row stride, not by a separate MOP variant.

## `reuse_a` vs `reuse_b` under tiny tiles

`matmul_configure_mop` and `matmul_configure_addrmod` both decide which input to reuse:

```cpp
// tt_llk_blackhole/llk_lib/llk_math_matmul.h:298-299
const bool reuse_a = (ct_dim >= rt_dim);
const std::uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
```

The principle is "reuse the input that gets multiplied against more partners." If the output block is wider than tall (`ct_dim >= rt_dim`), each row of A is multiplied against many columns of B, so A is held in SrcA across the inner loop and B is restreamed. The outer reuse-loop trip count `t_dim` becomes the *other* dimension — the one we sweep across.

For tiny tiles this is more than a micro-optimization. Consider a Flash MLA decode step where Q is laid out as 8x32 tiles and K is laid out as 32x32 tiles. The Q·K^T pass produces an output block whose `ct_dim` (sequence-length axis, after tiling) is typically much larger than `rt_dim` (1, because Q is a single 8-row strip). The `reuse_a = true` branch is taken: the single Q tiny tile sits in SrcA, and K tiles stream through SrcB. If instead the caller were to swap the operand order and present Q as the *right-hand* matrix, `reuse_a` would still be `true` for the same shape — but now the wrong tensor is pinned, and every outer-loop iteration would refetch Q from L1. The choice of operand ordering interacts with `ct_dim`/`rt_dim` to determine which input the math engine keeps resident, and a tiny tile on the wrong side amplifies the cost of getting that wrong because the savings (avoided MVMULs in the inner loop) come from the *short* axis, while the reuse decision determines the long-axis traffic pattern.

The address-modifier setup (`matmul_configure_addrmod`, `llk_math_matmul.h:23–277`) propagates this choice into the SrcA/SrcB/Dest increment fields. For example, in the `is_in0_16x32` case:

```cpp
// tt_llk_blackhole/llk_lib/llk_math_matmul.h:48-53 (paraphrased)
addr_mod_t{
    .srca = {.incr = 0, .clr = 0, .cr = 0},  // hold A
    .srcb = {.incr = 8, .clr = 0, .cr = 0},  // stride one face-row group
    .dest = {.incr = 8, .clr = 0, .cr = 0},
    ...
}.set(ADDR_MOD_0);
```

SrcA `incr = 0` is the literal mechanism by which "A is reused": the address pointer does not advance between MVMULs in the inner loop.

## Fidelity-phase scaling: tiny tiles do not save fidelity passes

A common — and incorrect — intuition is that a HiFi4 8x32 matmul should run "fewer fidelity passes" because the result is smaller. It does not. The fidelity phase count is set by `MATH_FIDELITY`, not by tile geometry. From `_llk_math_matmul_()` at `llk_math_matmul.h:629–676`:

```cpp
// tt_llk_blackhole/llk_lib/llk_math_matmul.h:634 (paraphrased)
constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
...
for (std::uint32_t t = 0; t < t_dim; ++t) {
    if constexpr (high_fidelity) {
        // run the MOP once per fidelity increment
        for (int phase = 0; phase < num_fidelity_phases; ++phase) {
            ckernel_template::run(...);
        }
    } else {
        ckernel_template::run(...);
    }
}
```

The MOP is constructed once by `matmul_configure_mop`. At runtime, `_llk_math_matmul_()` replays it once per fidelity increment per `t_dim` iteration. HiFi4 replays 4 times, HiFi2 replays 2, LoFi replays 1. Each replay is a full execution of `replay_buf_len` MVMULs — the inner loop count is *not* divided by the fidelity factor.

Concrete cycle accounting for the 8x32-by-32x32 case (`is_in0_16x32 = true`, `partial_face = true` after Chapter 2 Section 1's unpacker setup):
- `replay_buf_len = 4` (geometry is 16x32 × 32x32 with partial_face).
- LoFi: 1 fidelity phase × 4 MVMULs = 4 MVMULs per `t_dim` step.
- HiFi4: 4 fidelity phases × 4 MVMULs = 16 MVMULs per `t_dim` step.

Compared to a standard 32x32-by-32x32 matmul:
- LoFi 32x32: 1 × 16 = 16 MVMULs.
- HiFi4 32x32: 4 × 16 = 64 MVMULs.

So at LoFi the tiny tile is 4× cheaper per `t_dim` step (4 vs 16 MVMULs), and at HiFi4 it is still 4× cheaper (16 vs 64). The fidelity multiplier is the same on both sides of the ratio — tiny tiles save cycles in the *inner loop*, not in the *fidelity replay*. If you are budgeting for a HiFi4 attention kernel, do not assume the small Q-tile gives you "extra fidelity headroom"; you still pay the 4× fidelity factor.

## Throttle is only enabled for full 32x32

`_llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>()` carries `THROTTLE_LEVEL` as a template parameter, but the test harness configures it only for full tiles. `test_math_matmul.py` sweeps `TINY_TILES_MATMUL_COMBINATIONS` (lines 67–93) with `throttle=0` exclusively (line 89), and an assertion inside the throttle-enabled MVMUL emission paths (`llk_math_matmul.h:534–535`) rejects non-32x32 geometry. The reason is structural: throttle works by injecting NOP-equivalent spacing between MVMUL instructions via `run_throttled_sequence<N>()` (file:422–500), and the spacing constants are tuned to the 16-MVMUL standard inner loop. Reapplying them to a 4-MVMUL inner loop would push the average issue rate below the level where throttle is meaningful, so it is gated off.

The practical consequence: if you are porting a high-throttle full-tile op to tiny tiles, you lose the throttle knob. Power/thermal budgeting needs to be redone with throttle disabled.

## Destination register occupancy: the misconception

The dest register on Blackhole is partitioned into tile-sized slots. The number of slots available depends on `DstSync` (half vs. full) and `AccumMode`, and is computed at `ckernel.h:844–849`:

```cpp
// tt_llk_blackhole/common/inc/ckernel.h:844 (paraphrased)
template <DstSync sync, AccumMode accum, DstTileShape shape>
constexpr std::uint32_t get_dest_max_tiles() {
    return DEST_REGISTER_SIZE<sync, accum> >> DstTileSizeLog2[shape];
}
```

`DstTileShape` is an enum with **exactly three values** (`ckernel_defs.h:60–65`):

| `DstTileShape` | `DstTileSizeLog2` | Bytes per slot |
|----------------|:-----------------:|:--------------:|
| `Tile32x32`    |        10         |      1024      |
| `Tile32x16`    |         9         |       512      |
| `Tile16x16`    |         8         |       256      |

There is no `Tile8x32`, no `Tile4x32`, no `Tile1x32`. An 8x32 tiny tile is allocated against the `Tile32x32` slot — the same slot a full 32x32 tile would occupy. The dest hardware does not track "this slot is only 8 rows tall, I can pack three more 8-row tiles in here." Slots are tile-granular, not row-granular.

This means tiny tiles **do not give you more parallel dest occupancy**. If `get_dest_max_tiles<SyncHalf, AccumMode::Disable, Tile32x32>()` returns 8 slots, you have 8 slots whether you fill them with 32x32 tiles or 8x32 tiny tiles. The unused 24 rows in each slot are wasted from a parallelism perspective. You will recover the L1 budget from the smaller tile size when the packer writes back (Section 3 in this chapter), but you do not get to fit more tiles in dest concurrently.

This is the single most common porting bug when adapting a fused op to tiny tiles: the engineer assumes "8x32 is a quarter the size, so I can stage 4× as many in dest" and writes a tile-loop that exhausts dest after the first iteration. The dest budget must be computed from the `DstTileShape` enum's three values, not from the actual byte size of the tiny tile.

## Putting it together: the math-side cost model

For a tiny-tile matmul step the math-engine cost is approximately:

> cycles ≈ `t_dim` × `num_fidelity_phases` × `replay_buf_len` × cycles_per_MVMUL

with `replay_buf_len` drawn from the geometry table above and `num_fidelity_phases ∈ {1, 2, 4}` for LoFi/HiFi2/HiFi4. Tiny tiles cut `replay_buf_len` (16 → 8 → 4), preserve `num_fidelity_phases`, preserve `t_dim`, and do not change cycles_per_MVMUL. They do not change dest-slot count.

The unpacker (Section 1) and packer (Section 3) bear the other half of the geometry adaptation. Section 4 covers the SFPU face-iteration rule, which is the math-adjacent place where the per-face cost model differs from the matmul story above.
