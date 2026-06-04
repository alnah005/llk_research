# 8.03 Test Infrastructure Gaps and Recommendations

Chapter 3 identified four coverage gaps in standalone tiny-tile tests: SFPU unary at 8x32 / 16x32, reduce at tiny geometry, tilize/untilize at tiny geometry, and Quasar tiny matmul. This section proposes concrete test sources to close those gaps, and a PCC golden helper that handles the tiny-tile output layout that today's `MatmulGolden` cannot. Each proposal is grounded in the existing test scaffolding (`sweep_tiny_tiles_matmul()`, `generate_matmul_tiny_tiles_combinations()`, `eltwise_unary_sfpu_test.cpp`) so the work is incremental rather than greenfield.

**Prerequisites:** Chapter 3 (test infrastructure, especially Section 3.04 on standalone-op coverage), Chapter 4 (TT-Metal/TT-Blaze API surface), Chapter 5 (production usage patterns), and Section 8.01 (API surface audit) for the SFPU iteration-count rule.

## Gap 1: Standalone SFPU at 8x32 / 16x32

Today the only path that exercises SFPU on a tiny destination tile is the fused matmul-with-activation kernel — for example Flash MLA's `unified_kernels/matmul.hpp:173–176`, which instantiates `_llk_math_eltwise_unary_sfpu_*<approx_mode, false, 2>(...)` because the Q tile is 8x32 (2 faces). The `<approx_mode, false, N>` template parameter is the per-tile face iteration count and must equal `num_faces` of the destination geometry (see Section 8.01). When `N` is wrong, the SFPU either skips rows (PCC drop) or walks past the tile (silent corruption) — neither symptom is caught by a fused matmul golden because the activation is convolved with the matmul reduction.

The existing standalone test, `tests/sources/eltwise_unary_sfpu_test.cpp`, hardcodes the 32x32 geometry:

```cpp
// tests/sources/eltwise_unary_sfpu_test.cpp:57
const int iterations = 32;
```

and at the math phase (lines 88–94) calls:

```cpp
// tests/sources/eltwise_unary_sfpu_test.cpp:88-94 (paraphrased)
_llk_math_eltwise_unary_sfpu_start_<DST_SYNC>(block_tile);
test_utils::call_sfpu_operation<APPROX_MODE, /* ... */, iterations, /* ... */>();
```

There is no parametrization over `face_r_dim` or `num_faces`, so 8x32 and 16x32 paths are effectively untested in isolation.

### Proposed: `tests/sources/eltwise_tiny_tile_test.cpp`

A standalone SFPU test that mirrors `eltwise_unary_sfpu_test.cpp` but parametrizes the geometry. Suggested axes:

| Axis | Values | Notes |
| --- | --- | --- |
| Tile geometry (RxC) | 1x32, 2x32, 4x32, 8x32, 16x32 | Covers `face_r_dim ∈ {1, 2, 4, 8, 16}` with `num_faces ∈ {1, 2}` |
| SFPU op | sigmoid, silu, exp | Same op coverage as existing fused-matmul activations |
| Data format | Float16_b, Float16, Float32 | Match `eltwise_unary_sfpu_test.cpp` format axis |
| Approximation mode | true, false | Both branches of `_llk_math_eltwise_unary_sfpu_*<approx_mode, ...>` |

The SFPU iteration count `N` must be derived from the geometry, not hardcoded. Until a constexpr helper exists (see Section 8.01), the test can compute it from the same `TensorShape` it uses to size the CB:

```cpp
// tests/sources/eltwise_tiny_tile_test.cpp (proposed)
//
// Tile geometry comes from CT args; iteration count is derived.
constexpr uint32_t NUM_FACES   = get_compile_time_arg_val(0);
constexpr uint32_t FACE_R_DIM  = get_compile_time_arg_val(1);
constexpr uint32_t SFPU_ITERS  = NUM_FACES;  // SFPU walks face-by-face

_llk_math_eltwise_unary_sfpu_start_<DST_SYNC>(block_tile);
test_utils::call_sfpu_operation<APPROX_MODE, /* ... */, SFPU_ITERS, /* ... */>();
```

The legal `(num_faces, face_r_dim)` pairs are bounded by `validate_tensor_shape_tile_dependent_ops_` (`tensor_shape.h:87–94`): `num_faces ∈ {1, 2, 4}`, `face_r_dim ∈ {1, 2, 4, 8, 16}`, `face_c_dim == 16`. The test sweep should refuse illegal combinations at parametrization time so failure modes surface as test errors rather than kernel hangs.

## Gap 2: Reduce at Tiny Geometry

Reduce is structurally the most interesting gap because it is the only op where source and destination geometries genuinely differ — a row-reduce of a 32x32 tile produces a 1x32 result (in TT-Blaze terms, `face_r_dim=1`, `num_faces=1`). The standalone reduce test today (`tests/sources/reduce_test.cpp`) only exercises the 32x32 → 1x32 / 1x1 collapse using a full-tile source; the inverse problem — reducing a *tiny* source tile — is untested.

This matters because reduce's LLK path uses both unpack and math face loops, and `face_r_dim` flows into both. A row-reduce of an 8x32 tile must walk 8 rows per face (not 16), and the destination layout is a single-face 1x32 tile — `face_r_dim=1, num_faces=1`. If either side gets the geometry wrong, the symptom is a silent off-by-half-tile shift in the accumulator.

### Proposed: `tests/sources/reduce_tiny_tile_test.cpp`

| Axis | Values | Notes |
| --- | --- | --- |
| Source tile height | 1, 2, 4, 8, 16 | Width fixed at 32 |
| Reduce op | row, col, scalar | Each produces a different destination geometry |
| Reduce kind | sum, max | Two representative paths |
| Data format | Float16_b, Float16, Float32 | Match `reduce_test.cpp` |

Row reduce of an Rx32 source yields a 1x32 destination (`face_r_dim=1, num_faces=1`); col reduce of an Rx32 source yields a Rx1 destination (`face_c_dim=1`, which is **not** in the legal set — `face_c_dim` is fixed at 16, see Section 8.04 for why). Practically this means col-reduce of a tiny tile pads to `face_c_dim=16` with zeros / `-inf` depending on the reduce kind, and the test must encode that expectation in its golden.

Because `MatmulGolden` already assumes a 32x32 destination layout, a reduce golden cannot reuse it directly — see the PCC helper proposal below.

## Gap 3: Quasar Tiny Matmul

The Wormhole and Blackhole matmul paths have rich tiny-tile coverage through `test_math_matmul.py` and `test_unpack_matmul.py`, both of which consume `generate_matmul_tiny_tiles_combinations()` (`matmul_sweep.py:195–218`):

```python
# matmul_sweep.py:195-218 (paraphrased)
def generate_matmul_tiny_tiles_combinations():
    tile_in0_rows = [1, 2, 4, 8, 16]
    tile_in1_rows = 32
    # ... yields ((in0_row, in0_col), (in1_row, in1_col)) pairs
```

and the higher-level `sweep_tiny_tiles_matmul()` (`matmul_sweep.py:468–578`) wraps those combinations with format, `dest_acc`, and stochastic-rounding axes. Quasar's standalone matmul test, `tests/sources/quasar/matmul_quasar_test.cpp`, does *not* use this infrastructure; it tests only the 32x32 geometry.

This is a meaningful gap because Quasar's unpacker is TDMA-descriptor driven (see Chapter 6, Section 6.04 for the TDMA-vs-tensix-unpack contrast). A tiny-tile descriptor on Quasar has to encode the same `(num_faces, face_r_dim)` pair *and* the TDMA stride/offset table, and the two encodings live in different code paths. Until tiny matmuls are tested on Quasar in isolation, any divergence is masked by the fact that no production op exercises that combination yet.

### Proposed: `tests/sources/quasar/matmul_quasar_tiny_test.cpp`

Copy `matmul_quasar_test.cpp` as the base, then:

1. Add a CT arg layer that takes `(in0_tile_rows, in0_tile_cols, in1_tile_rows, in1_tile_cols)`.
2. Drive the sweep from `generate_matmul_tiny_tiles_combinations()` in the Python harness (no need to rewrite the combination generator).
3. Validate against `MatmulGolden` extended for tiny output geometry (see below).
4. Gate the test on `arch == quasar`, matching the existing Quasar-only directory convention.

The minimum useful coverage is the same in0 row sweep `{1, 2, 4, 8, 16}` that already exists for Wormhole/Blackhole. Beyond parity, the test should specifically exercise Quasar's TDMA-descriptor edge cases at `face_r_dim < 16`, where the descriptor's row-stride field is the most likely failure point.

## PCC Golden Helper for Tiny Tiles

`MatmulGolden` (`matmul_sweep.py`) computes its reference output and then tilizes it for comparison against the device readback. The tilization path assumes a 32x32 destination tile — concretely, it hardcodes `face_r_dim=16, num_faces=4` when it lays out the result faces. For a tiny output (e.g. an 8x32 matmul C tile produced by the Flash MLA Q path), this tilization is wrong: faces are laid out in the wrong order, and the unused half of the tile is filled with non-zero garbage that fails PCC even when the device output is bit-correct.

The same problem applies to the proposed reduce test — a row-reduce produces a 1x32 destination that `MatmulGolden` simply cannot represent — and to any standalone tiny-SFPU test that wants to compare against torch in tilized space rather than untilized.

### Proposed: `TinyTileGolden` helper class

A small Python helper that takes an explicit output `(face_r_dim, num_faces, face_c_dim, num_faces_per_column)` and tilizes a torch reference to match. Sketch of the surface:

```python
# tests/python_tests/helpers/tiny_tile_golden.py (proposed)

class TinyTileGolden:
    """PCC reference for tiny-tile ops. Unlike MatmulGolden, the destination
    tile layout is an explicit parameter, not assumed 32x32."""

    def __init__(self, output_tile_shape: ttnn.Tile, data_format):
        self.tile = output_tile_shape           # e.g. ttnn.Tile((8, 32))
        self.data_format = data_format

    def matmul(self, a_torch, b_torch, *, in0_tile, in1_tile):
        ref = a_torch @ b_torch
        return tilize_block(
            ref,
            face_r_dim=self.tile.face_r_dim,
            num_faces=self.tile.num_faces,
        )

    def sfpu(self, a_torch, op):
        ref = op(a_torch)
        return tilize_block(
            ref,
            face_r_dim=self.tile.face_r_dim,
            num_faces=self.tile.num_faces,
        )

    def reduce(self, a_torch, *, kind, axis):
        ref = REDUCE_OPS[kind](a_torch, axis=axis)
        # destination face_c_dim stays 16 even when axis-1 reduce
        # collapses the data dim — pad with reduce identity.
        return tilize_block(
            ref,
            face_r_dim=self.tile.face_r_dim,
            num_faces=self.tile.num_faces,
            pad_value=REDUCE_IDENTITY[kind],
        )
```

The key design choice is that `tilize_block` becomes geometry-parametric. `MatmulGolden` can be refactored to delegate to `TinyTileGolden` once the helper exists; in the interim the helper lives alongside it and the new tests above consume it directly. The reduce method also encodes the `face_c_dim=16` padding rule (Section 8.04 covers why `face_c_dim` is fixed) so col-reduce goldens match the device's zero/`-inf` padding behavior.

## Summary

These four additions — `eltwise_tiny_tile_test.cpp`, `reduce_tiny_tile_test.cpp`, `quasar/matmul_quasar_tiny_test.cpp`, and the `TinyTileGolden` helper — would close the standalone-op coverage gaps identified in Chapter 3 and give the next round of tiny-tile work (whatever shape it takes) a regression net to land against. The matmul side already has the sweep infrastructure (`generate_matmul_tiny_tiles_combinations`, `sweep_tiny_tiles_matmul`) to make the SFPU, reduce, and Quasar tests cheap to write; the missing piece is a golden helper that doesn't assume 32x32. Building the helper first unblocks all three test sources in parallel.
