# 3.2 `generate_matmul_tiny_tiles_combinations()` — Sweep Generator Walkthrough

The Python driver that feeds tiny-tile shapes into the LLK matmul test harness lives in a single helper: `generate_matmul_tiny_tiles_combinations()` in `tests/python_tests/helpers/matmul_sweep.py`. This file dissects that helper line by line, contrasts it with its sibling `generate_matmul_dimension_combinations()` (the standard 32x32 sweep), and explains why the current matrix restricts tiny-tile variation to input 0. The rationale is not arbitrary: it follows from how the unpacker stride is parameterized and how the math-engine selects between `reuse_a` and `reuse_b` inner-loop strategies.

**Prerequisites:** Chapter 1 (tiny-tile fundamentals and face layout), Chapter 2 (matmul fidelity / face geometry), and Section 3.1 (overview of the LLK matmul test drivers). Readers should already understand the meaning of `in0_tile_r_dim`, `num_faces`, and the difference between a "partial face" and a full 16x16 face.

---

## 1. Overview of `generate_matmul_tiny_tiles_combinations()`

The helper produces a flat list of `(in0_dims, in1_dims)` dimension pairs that will be expanded by the surrounding `sweep_tiny_tiles_matmul()` driver into full pytest parametrizations. Its job is purely combinatorial: enumerate the cartesian product of the sub-32-row in0 variants against the in1 column counts allowed by `max_tiles`.

```python
# tests/python_tests/helpers/matmul_sweep.py:195-218
def generate_matmul_tiny_tiles_combinations(max_tiles: int) -> List[tuple]:
    valid_combinations = []
    tile_in0_rows = [1, 2, 4, 8, 16]
    tile_in0_columns = 32
    tile_in1_rows = 32
    tile_in1_columns = list(range(32, (max_tiles + 1) * 32, 32))
    return [
        ((tile_in0_row, tile_in0_columns), (tile_in1_rows, tile_in1_column))
        for tile_in0_row in tile_in0_rows
        for tile_in1_column in tile_in1_columns
    ]
```

A few observations land immediately from the signature:

- The function takes a single integer `max_tiles` and returns a list of two-tuples. Each tuple is `((in0_h, in0_w), (in1_h, in1_w))`, expressed in **element** counts, not tile counts. So `((8, 32), (32, 64))` means in0 is one 8x32 tile and in1 is a 1x2 tile-grid of standard 32x32 tiles.
- The `valid_combinations` local is allocated on line 196 but never appended to — the function returns the list comprehension directly. This is dead code left over from an earlier iteration; it is harmless but worth noting if you grep for it.
- `tile_in0_rows` is hard-coded to `[1, 2, 4, 8, 16]`. These are exactly the row counts the LLK harness supports for in0 today (powers of two from 1 up to a half-face). 32 is omitted because that case is already covered by `generate_matmul_dimension_combinations()`.
- `tile_in0_columns` is fixed at 32. Tiny-tile variation never narrows in0's column dimension.
- `tile_in1_columns` sweeps `[32, 64, 96, ..., max_tiles*32]` in steps of 32. For the default `max_tiles=8`, this gives `[32, 64, 96, 128, 160, 192, 224, 256]` — 8 values.
- `tile_in1_rows` is fixed at 32.

The output cardinality is therefore `len(tile_in0_rows) * len(tile_in1_columns)` = `5 * max_tiles`. For `max_tiles=8` you get 40 dimension pairs. (The chapter overview rounds this to "~35"; the exact number depends on `max_tiles`.)

## 2. Dimension Sweep Table and Rules

The helper enumerates a 2D grid: rows of the grid are in0 row variants, columns are in1 column counts. The K dimension (in0's columns = in1's rows) is always 32 — there is no K sweep at all in the tiny-tile matrix.

| in0 shape | in1 shape (max_tiles=8 examples) | K (always) | Output shape |
|-----------|------------------------------------|------------|--------------|
| 1x32 | 32x32, 32x64, ..., 32x256 | 32 | 1xN |
| 2x32 | 32x32, 32x64, ..., 32x256 | 32 | 2xN |
| 4x32 | 32x32, 32x64, ..., 32x256 | 32 | 4xN |
| 8x32 | 32x32, 32x64, ..., 32x256 | 32 | 8xN |
| 16x32 | 32x32, 32x64, ..., 32x256 | 32 | 16xN |

The 16x32 row deserves a footnote: it sits at the boundary between "tiny" and "full-face" because a 16-row tile fills exactly one face's worth of rows. As we will see in Section 5, `partial_face_math` flips to `False` for this case, so 16x32 exercises a slightly different code path than 1/2/4/8.

## 3. Comparison to `generate_matmul_dimension_combinations()`

The standard helper sits 30 lines above the tiny variant in the same file and looks like:

```python
# tests/python_tests/helpers/matmul_sweep.py:165-192
def generate_matmul_dimension_combinations(
    max_tiles: int, kt_dims: Iterable[int] = range(1, 5)
) -> List[tuple]:
    return [
        ([mt_dim * TILE_DIM, kt_dim * TILE_DIM], [kt_dim * TILE_DIM, nt_dim * TILE_DIM])
        for mt_dim in range(1, max_tiles + 1)
        for nt_dim in range(1, max_tiles // mt_dim + 1)
        for kt_dim in kt_dims
    ]
```

Three structural differences set the two helpers apart:

1. **Granularity of M and N.** The standard helper multiplies its loop indices by `TILE_DIM` (32), so every shape is a multiple of 32. The tiny helper draws in0 rows from `[1, 2, 4, 8, 16]`, which are *element* counts not multiples of any tile dimension.
2. **K-dimension sweep.** Standard sweeps `kt_dim ∈ {1, 2, 3, 4}` — four K values per (M, N) pair. The tiny helper fixes K=32 (one tile) and does not sweep it at all.
3. **M-times-N envelope.** Standard caps the product `mt_dim * nt_dim` at `max_tiles` (note the `max_tiles // mt_dim` in the inner range). The tiny helper has no such cap because in0's M is held to a single tile-row and only in1's N is varied.

The standard helper produces a triangular sweep (e.g. for `max_tiles=8, kt_dims=1..4`: `8+4+2+2+1+1+1+1` = 20 (M,N) pairs × 4 K values = 80 shapes). The tiny helper produces a rectangular sweep (5 × 8 = 40 shapes for `max_tiles=8`). Both are flat lists of dimension tuples and are concatenated into a single pytest parametrization downstream.

## 4. Composition in the Python Drivers

Both helpers are wrapped by higher-level sweep functions (`sweep_matmul` and `sweep_tiny_tiles_matmul`) that join the dimension lists with the format / dest-acc / rounding / sync axes. The composition pattern is identical across `test_math_matmul.py` and `test_unpack_matmul.py`:

```python
# tests/python_tests/llk/test_math_matmul.py:60-72
MATMUL_COMBINATIONS = sweep_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
    math_matmul=True,
)
TINY_TILES_MATMUL_COMBINATIONS = sweep_tiny_tiles_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
    math_matmul=True,
)
```

The two lists are then chained (test_math_matmul.py:76–93). Standard matmuls are tested across throttle levels 1–5; tiny-tile matmuls are restricted to throttle level 0 (no throttling sweep). The chained list is passed as the sole argument to `pytest.parametrize`, so pytest sees a single test function `test_math_matmul()` taking parameters from both pools indiscriminately. The driver does not branch on "is this a tiny tile?" at runtime — the FaceLayoutConfig differences carry through naturally because they were baked in at sweep-construction time.

`test_matmul.py` and `test_matmul_custom.py`, by contrast, call only `generate_matmul_dimension_combinations()`. They do not run any tiny-tile cases. This is the coverage gap discussed in Section 3.3.

## 5. Why In1 Is Restricted to 32x32

The most surprising thing about `generate_matmul_tiny_tiles_combinations()` is what it does *not* sweep: tiny in1. There is no `[1, 2, 4, 8, 16]` axis on in1's row dimension. The function explicitly fixes `tile_in1_rows = 32` and `tile_in0_columns = 32`. Three forces push this restriction; understanding them is essential for anyone who later wants to extend the matrix.

### 5.1 Unpacker stride is parameterized by in1 width

The unpacker walks the K-dimension by stepping in1's row pointer by `ct_dim * TILE_SIZE_UNPACK` bytes per K iteration. This stride calculation assumes in1's tile width is exactly 32. If in1 had only 8 columns (a hypothetical 32x8 tile), the unpacker's `Y` increment would no longer match the host-side stimulus's row layout, and the in-flight FaceLayoutConfig values for `unpack_transpose_*` axes would need new code paths. The matmul_sweep.py file flags this as a known limitation:

```python
# tests/python_tests/helpers/matmul_sweep.py:300
# TODO: These can be removed when tiny tiles are supported for both in0 and in1
```

The `config_params` dict (lines 301–336) shows two `num_faces` configurations commented out:

```python
# tests/python_tests/helpers/matmul_sweep.py:301-336 (abbreviated)
config_params = {
    # num_faces=1: COMMENTED OUT — blocked on in1 tiny-tile support
    # num_faces=2: COMMENTED OUT — blocked on in1 tiny-tile support
    "num_faces=4": { ... },  # enabled
}
```

Only the `num_faces=4` row is active. Until in1 can be narrowed, the harness cannot exercise the `num_faces=1` or `num_faces=2` output paths in the tiny-tile matmul matrix at all.

### 5.2 Math-engine inner-loop factorization (`reuse_a` vs. `reuse_b`)

The math kernel's MVMUL inner loop is structured around one of two reuse strategies: hold SrcA constant and iterate SrcB (reuse_a), or hold SrcB constant and iterate SrcA (reuse_b). The choice is made at init time by `_llk_unpack_AB_matmul_init_<>()` (see `llk_unpack_AB_matmul.h` lines 28–29 — the reuse mode is encoded in a template parameter that propagates through to MVMUL operand selection).

When in0 (= SrcB on Wormhole, SrcA on Blackhole — see Chapter 2 for the swap) becomes tiny, the inner loop's *count* shrinks but the *structure* stays the same: MVMUL still iterates faces in the same order, just over fewer rows. When in1 becomes tiny, the inner loop's *structure* changes — the dot-product accumulation order is no longer K-aligned in the same way, and `partial_face_math` would need to be reinterpreted on the opposite operand. None of the existing test drivers wires this through.

### 5.3 Golden-reference tilization gaps

`MatmulGolden.tilize_block()` (golden_generators.py:1103–1108) computes the expected output tile layout from the matmul result:

```python
# tests/python_tests/helpers/golden_generators.py:1103-1108 (approx)
if tilize:
    res = tilize_block(
        res,
        dimensions=(input_A_dimensions[0], input_B_dimensions[1]),
        stimuli_format=data_format,
    )
```

Note the `dimensions` argument: it takes A's row count and B's column count — the output shape. The tilizer is called with the *output* tile shape, not in1's shape. `tilize_block()` itself (tilize_untilize.py:22–114) does support a `tile_dimensions` override for sub-32-row tiles, so output-side partiality is fine. But there is no inverse path that lets the golden generator construct a *narrow-width* in1 reference: the input tilization assumes 32-column inputs throughout. Extending this would require new code in `tilize_block()` to handle width-partial tiles symmetrically with the existing height-partial support.

The combined effect is that even if the kernel supported tiny in1, the test infrastructure could not generate a correct golden — so the sweep generator simply refuses to enumerate those cases.

## 6. Face Layout Configuration for Tiny Tiles

`sweep_tiny_tiles_matmul()` materializes a `FaceLayoutConfig` for every dimension pair before it goes into the pytest list. The relevant block is:

```python
# tests/python_tests/helpers/matmul_sweep.py:506-529 (abbreviated)
tile_dims = generate_tile_dims(
    ([32, 32], input1_dims), in0_tile_r_dim=input0_dims[0]
)
output_num_faces = calculate_matmul_output_faces(
    num_faces_in0=2,
    num_faces_in1=4,
    is_in0_horizontal=True,
)
face = FaceLayoutConfig(
    num_faces_in0=2,
    num_faces_in1=4,
    num_faces=output_num_faces,  # 2
    unpack_transpose_faces=Transpose.No,
    unpack_transpose_within_face=Transpose.No,
    partial_face_in0=True,
    partial_face_in1=False,
    partial_face_math=input0_dims[0] < 16,
    partial_face_pack=True,
)
```

Two things to call out. First, line 509 passes `input0_dims[0]` (the tiny row count) as the explicit `in0_tile_r_dim` to `generate_tile_dims()`, which is what eventually flows into the kernel's `_llk_unpack_A_init_<>()` and `_llk_math_matmul_init_<>()` calls as their `in0_tile_r_dim` parameter. Second, the only field that depends on the dimension pair is `partial_face_math`. All the other fields are constants for the tiny-tile sweep.

The full per-shape face layout grid:

| in0 shape | `num_faces_in0` | `partial_face_in0` | `partial_face_math` | `partial_face_pack` | Output `num_faces` |
|-----------|----------------|--------------------|---------------------|---------------------|--------------------|
| 1x32 | 2 | True | True | True | 2 |
| 2x32 | 2 | True | True | True | 2 |
| 4x32 | 2 | True | True | True | 2 |
| 8x32 | 2 | True | True | True | 2 |
| 16x32 | 2 | True | **False** | True | 2 |

The interesting row is 16x32. Its in0 still uses a 2-face horizontal layout (because it occupies one full face's worth of rows, but only one row of the 4-face grid), so `partial_face_in0=True`. But `partial_face_math=False` because the math engine treats a 16-row tile as a full face — MVMUL's inner loop iterates the full 16 rows without truncation. The pack side stays partial because the *output* shape inherits in0's row count (16, not 32), so the pack engine still needs the partial-face write pattern.

## 7. Worked Example: `((8, 32), (32, 64))`

Take the pair `((8, 32), (32, 64))` — an 8x32 tiny in0 multiplied against a 32x64 standard in1. The sweep generator emits this as one of the 40 entries for `max_tiles=8`.

Downstream:

1. `generate_tile_dims(([32, 32], [32, 64]), in0_tile_r_dim=8)` produces a `tile_dims` struct where in0's logical shape is `[8, 32]` and in1's logical shape is `[32, 64]` (two 32x32 tiles wide).
2. `FaceLayoutConfig` is built with `num_faces_in0=2`, `num_faces_in1=4`, `num_faces=2`, `partial_face_in0=True`, `partial_face_in1=False`, `partial_face_math=True` (because 8 < 16), `partial_face_pack=True`.
3. The C++ test harness (math_matmul_test.cpp:80–88) receives `params.in0_tile_r_dim=8` and calls `_llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>()` with `PARTIAL_FACE_MATH=true`. The MVMUL inner loop is reduced from 16 to 8 iterations per face.
4. The unpacker (math_matmul_test.cpp:33–49) computes `effective_face_r_dim = min(8, FACE_R_DIM) = 8` and uses that to compute SrcB's face stride.
5. The pack engine writes 8 rows of output per tile (output shape `[8, 64]` = 8 rows × 2 column-tiles).
6. The golden generator computes `result = A @ B` where A is `[8, 32]` and B is `[32, 64]`, then calls `tilize_block(result, dimensions=(8, 64), ...)`. `tilize_block` (tilize_untilize.py:94–114) detects that `tile_rows=8 != DEFAULT_TILE_R_DIM=32` and routes to the partial-face tilization path with `face_r_dim=8` and `tile_dimensions=[8, 32]`.
7. The expected tilized output size (tilize_untilize.py:117–120) is `total_tiles * num_faces * elements_per_face` = `2 * 2 * (8 * 16)` = `512 elements` (vs. `2 * 4 * 256` = `2048` for a standard `[32, 64]` output).

The end-to-end pipeline shows why each piece of metadata matters: a single integer (`in0_tile_r_dim=8`) propagates through the FaceLayoutConfig, the kernel init template parameters, the unpacker stride calculation, the MVMUL inner-loop count, the pack output footprint, and the golden tilization. The sweep generator's job is just to feed that integer the right values; the surrounding infrastructure handles the rest.

---

The next file (Section 3.3) catalogues the standalone coverage gaps — eltwise, reduce, tilize/untilize, and Quasar — where no equivalent of `generate_matmul_tiny_tiles_combinations()` exists today.
