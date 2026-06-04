# Unpacker with Tiny Tiles

The unpacker learns about tile geometry through six template parameters on `_llk_unpack_AB_matmul_init_<>()`: `face_r_dim_A`, `face_r_dim_B`, `num_faces_A`, `num_faces_B`, `partial_face_A`, and `partial_face_B`. These parameters drive three downstream effects: the replay buffer (MOP) length, the address-counter `x_end` value, and the activation of partial-face mode in the unpacker hardware. When a face has fewer than 16 rows — i.e. a tiny tile — the inner-loop count shrinks, the address strides change, and the unpacker walks faces one at a time instead of streaming a full 16-row block.

**Prerequisites:** Chapter 1, Sections "Geometry Vocabulary" and "Legal Tile Geometries". See also [[introduction_to_tt_llk]] for the baseline unpacker MOP model.

---

## 1. The Init Signature

`_llk_unpack_AB_matmul_init_<>()` is defined in `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:188-241`. The full template signature:

```cpp
// tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:188
template <
    const std::uint32_t transpose       = 0,
    const std::uint32_t ct_dim          = 1,
    const std::uint32_t rt_dim          = 1,
    const std::uint32_t kt_dim          = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4,
    const bool unpA_partial_face        = false,
    const bool unpB_partial_face        = false>
__attribute__((always_inline)) inline void _llk_unpack_AB_matmul_init_(...)
```

Naming subtlety worth pinning down up front: in this LLK, `unpA` corresponds to *in0* (which lands in `SrcB`) and `unpB` corresponds to *in1* (which lands in `SrcA`). This is the LHS / RHS swap that matmul does relative to the natural "A * B" reading. The test harness (see Section 4) makes the mapping explicit by passing `in1` parameters to `unpB_*` slots.

The six geometry parameters mean:

| Parameter | Meaning | Legal values |
|---|---|---|
| `unpA_face_r_dim`, `unpB_face_r_dim` | Rows per face for inputs A/B | 1, 2, 4, 8, 16 |
| `unpA_num_faces`, `unpB_num_faces` | Faces per tile for inputs A/B | 1, 2, 4 |
| `unpA_partial_face`, `unpB_partial_face` | Enable per-face address walking | bool |

The legal values for `num_faces` are enforced at file:200-201 by `static_assert`. The legal `face_r_dim` set comes from `tensor_shape.h:86-99`'s `validate_tensor_shape_tile_dependent_ops_()` validator. `face_c_dim` is fixed at 16 — the unpacker has no template knob for narrowing column dimension here, because all tiny-tile geometry in production is row-truncated (8x32, 4x32, etc.), not column-truncated.

The constants `FACE_R_DIM` (16) and `FACE_C_DIM` (16) come from `tt_llk_blackhole/common/inc/ckernel_defs.h:86-95`.

---

## 2. MOP Length: 12 vs. 18

The MOP (replay buffer) is configured in `_llk_unpack_AB_matmul_mop_config_()`. The replay buffer length is selected by which operand is being reused and whether that operand uses partial-face mode:

```cpp
// tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:28-29
const bool reuse_a = ct_dim >= rt_dim;
const std::uint32_t replay_buf_prog_len =
    (reuse_a && unpA_partial_face) ? 18
  : ((!reuse_a && unpB_partial_face) ? 18 : 12);
```

The matmul iterates over a `(rt_dim, ct_dim)` output block. `reuse_a` picks which operand sits stationary across the inner loop: when `ct_dim >= rt_dim`, A is reused along the column dimension; otherwise B is reused along the row dimension. The MOP is the inner-loop body, so the partial-face flag of the *reused* operand is the one that drives MOP length.

- **12 instructions** — standard path. One face-row block is unpacked per MOP iteration; the address counter strides over a complete 16-row face in one shot.
- **18 instructions** — partial-face path. The extra six instructions perform per-face address reconfiguration: after each `face_r_dim`-row chunk, the inner X counter must be reset and the face stride applied manually, because the hardware's auto-increment is sized for full-face blocks.

For the non-reused operand, `partial_face` still affects the address-counter setup (Section 3), but not the MOP length — the non-reused operand re-issues its init MOP per K-step rather than running inside the inner replay.

---

## 3. Address-Counter Setup: `x_end`

The X counter is what walks the unpacker through a tile's data in L1. Its end value is computed differently depending on `partial_face`:

```cpp
// tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:215-235 (paraphrased)
if (unpA_partial_face) {
    config_unpacker_x_end<p_setadc::UNP_A>(unpA_face_r_dim);
} else {
    constexpr std::uint32_t unpA_x_end =
        unpA_num_faces * unpA_face_r_dim * FACE_C_DIM - 1;
    // ... program x_end = unpA_x_end ...
}
// (analogous block for UNP_B)
```

Two regimes:

1. **`partial_face == false`** — `x_end = num_faces * face_r_dim * FACE_C_DIM - 1`. This is the total cell count of the tile minus one. The unpacker streams the entire tile under one auto-incrementing X counter. For a standard 32x32 (4 faces, face_r_dim=16, face_c_dim=16) this is `4 * 16 * 16 - 1 = 1023`.

2. **`partial_face == true`** — `x_end = face_r_dim` (programmed by `config_unpacker_x_end`). The X counter only covers one face's row dimension at a time; the MOP epilogue handles the per-face stride to step from one face to the next. This is what tiny-tile unpacks need: when `face_r_dim < 16`, the hardware's face-stride logic assumes 16-row faces, so we deliberately constrain the counter to the smaller granule and walk faces explicitly.

The practical rule for a kernel author: if any face dimension is non-default (face_r_dim < 16, or num_faces ∈ {1, 2}), the matching `partial_face_*` flag must be set to true. The Flash MLA decode op does this for the Q operand (the 8x32 tiny tile) while leaving K's flag false.

---

## 4. Haloize and Transpose Interaction

Haloize mode is the within-face transpose. It's set unconditionally in init based on the `transpose` template parameter:

```cpp
// tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:208
cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose);
```

The comment at file:204-206 notes that on [WH] (and [BH], which inherits this LLK), the unpacker performs both *transpose-of-faces* (re-ordering F0/F1/F2/F3) and *transpose-within-face* (the 16x16 transpose inside one face). The within-face transpose is what `THCON_SEC0_REG2_Haloize_mode_RMW` toggles. There is no special tiny-tile interaction here — haloize operates on whatever the face geometry is, so a transposed 8x32 unpack runs the within-face transpose on the 8x16 sub-faces. The user-visible constraint, validated upstream in `tensor_shape.h`, is that transpose with tiny tiles is supported only for the same legal `face_r_dim` set (1, 2, 4, 8, 16).

---

## 5. Source-Register Population

A 32x32 tile populates `SrcA` (or `SrcB`) as four 16x16 faces tiled in F0/F1 over F2/F3 layout. An 8x32 tile (num_faces=2, face_r_dim=8) populates only the top 8 rows of F0 and F1 — F2/F3 are not written. The hardware doesn't zero-fill the unused rows; subsequent consumers (math engine, SFPU) must restrict their iteration counts to match. This is why the math `_llk_math_matmul_init_<>()` call also takes `num_faces` and `face_r_dim` — Chapter 2, Section "Math Engine with Tiny Tiles" covers the math side.

Critically, the *Dest* slot allocation is per-tile, not per-row: an 8x32 tiny tile occupies a full Tile32x32 slot in the destination register file (see `DstTileShape` enum at `ckernel_defs.h:60-65`, which only enumerates Tile32x32, Tile32x16, Tile16x16). Tiny tiles waste destination capacity. This trade-off is discussed in Chapter 2's destination-register section.

---

## 6. Worked Example: Unpacking a 2x32 Tile into SrcB

Suppose in0 is a `2x32` tile: `num_faces_A = 2`, `face_r_dim_A = 2`, `partial_face_A = true`. The init call is:

```cpp
// num_faces=2 (F0 and F1 side-by-side), face_r_dim=2, partial_face=true
_llk_unpack_AB_matmul_init_<
    /* transpose       */ 0,
    /* ct_dim          */ 1,
    /* rt_dim          */ 1,
    /* kt_dim          */ 1,
    /* unpA_face_r_dim */ 2,
    /* unpB_face_r_dim */ 16,
    /* unpA_num_faces  */ 2,
    /* unpB_num_faces  */ 4,
    /* unpA_partial_face */ true,
    /* unpB_partial_face */ false>();
```

What happens inside:

1. `reuse_a = (ct_dim >= rt_dim) = true`. Combined with `unpA_partial_face = true`, the MOP gets the 18-instruction replay buffer (file:29).
2. UNP_A's `x_end` is set via `config_unpacker_x_end<p_setadc::UNP_A>(2)`. Only 2 rows of the X counter are valid at a time.
3. UNP_B's `x_end` is `4 * 16 * 16 - 1 = 1023` (full 32x32 tile, standard path).
4. Haloize is off (`transpose=0`).
5. On unpack, F0 of the tiny tile is loaded into rows 0-1 of `SrcB`'s F0 region; F1 of the tiny tile is loaded into rows 0-1 of `SrcB`'s F1 region. Rows 2-15 of F0 and F1, and all of F2/F3, are untouched by this unpacker call.

The match-up between this `SrcB` layout and what the math engine consumes is what Section 2 of this chapter ("Math Engine with Tiny Tiles") covers — the math side must issue a correspondingly truncated MVMUL sequence so it doesn't read stale rows.

---

## 7. Test Harness Reference

The unit test for this path lives at `tt_llk_blackhole/tests/sources/unpack_matmul_test.cpp:40-50`:

```cpp
// tests/sources/unpack_matmul_test.cpp:40-50 (paraphrased)
_llk_unpack_AB_matmul_init_<
    transpose, ct_dim, rt_dim, kt_dim,
    params.in1_tile_r_dim < FACE_R_DIM ? params.in1_tile_r_dim : FACE_R_DIM,
    params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM,
    params.num_faces_B,
    params.num_faces_A,
    params.PARTIAL_FACE_B,
    params.PARTIAL_FACE_A>();
```

Two patterns worth absorbing:

- **Clamp to `FACE_R_DIM`**: when the input tile has 16 or more rows per face, the template gets `FACE_R_DIM` (the default). The non-default value is only passed for genuine tiny tiles. This keeps the standard-path template instantiations from proliferating.
- **A/B swap**: `params.in1_*` feeds `unpB_*` (the rightmost positional in the natural reading) but the *first* explicit template argument after the defaults. The harness consistently uses `in0` ↔ `unpA` ↔ `SrcB`, `in1` ↔ `unpB` ↔ `SrcA`.

`TestConfig` in `tests/python/test_math_matmul.py` lines 186-200 generates the parameter pack via helper templates `TILE_COUNT`, `NUM_FACES`, `UNPACK_TRANS_FACES`, `UNPACK_TRANS_WITHIN_FACE`, `PARTIAL_FACE`. The tiny-tile combinations are swept at lines 67-93 (`TINY_TILES_MATMUL_COMBINATIONS`), restricted to `throttle = 0` — the throttle path is not exercised with tiny tiles. See Chapter 2's destination-register / throttle section for why.

---

## 8. Summary

| Concern | Standard tile (32x32) | Tiny tile (face_r_dim < 16) |
|---|---|---|
| `face_r_dim` | 16 (default) | 1, 2, 4, or 8 |
| `num_faces` | 4 (default) | 1, 2, or 4 |
| `partial_face` | false | **must be true** |
| MOP length (reused operand) | 12 | 18 |
| `x_end` | `num_faces * face_r_dim * 16 - 1` | `face_r_dim` |
| SrcA/SrcB occupancy | All 4 faces, all rows | Top `face_r_dim` rows of `num_faces` faces |

The next file in this chapter walks the math engine's mirror-image of these parameters: how `_llk_math_matmul_init_<>()` consumes `in0_tile_r_dim` and `in1_tile_r_dim`, why the 16x16-by-16x16 case is forbidden, and how the MVMUL inner-loop count is derived.
