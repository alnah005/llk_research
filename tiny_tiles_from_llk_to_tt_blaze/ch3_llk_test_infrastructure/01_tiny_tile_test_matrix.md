# Tiny-Tile Test Matrix Across LLK Matmul Drivers

This file catalogues the matmul drivers in `tt-llk/tests/` and identifies which of them actually exercise tiny tiles, which ones only sweep standard 32x32 tiles, and where Quasar stands. The goal is to give the reader a precise map of "what ground is already covered" before Chapters 4–5 walk into production usage on top of the same harness.

**Prerequisites:** Chapter 1 (what is a tiny tile, face geometry, partial-face semantics) and Chapter 2 (LLK data path: unpack / math / pack for tiny tiles). Familiarity with the matmul test layout from [[minimum_requirements_to_run_a_matmul_on_p150_using_pure_llk]] is helpful but not required.

## Scope

The LLK matmul test suite consists of five principal Python drivers, each paired with one or more C++ kernel sources under `tests/sources/`. Only three of the five drivers currently chain `sweep_tiny_tiles_matmul()` into their parametrization; the remaining two restrict themselves to standard 32x32 sweeps. A separate Quasar driver exists and is explicitly hard-coded to full faces. The table below summarizes the matrix; the rest of this file walks each row.

| Driver (Python) | Kernel source (C++) | Tiny tiles? | Geometry swept (in0 / in1) | Throttle | Stoch. rounding modes | Face modes | LOOP_FACTOR |
|---|---|---|---|---|---|---|---|
| `test_math_matmul.py` | `math_matmul_test.cpp` | Yes | in0_r ∈ {1,2,4,8,16}, in1=32x32 | 0 (tiny) / 1–5 (std) | `No` | 4 (in1), 2 (in0) | n/a |
| `test_unpack_matmul.py` | `unpack_matmul_test.cpp` | Yes | in0_r ∈ {1,2,4,8,16}, in1=32x32 | 0 | `No`, `Fpu`, `Pack`, `All` | 1, 2, 4 (std) + tiny-tiles fixed at 2x4 | n/a |
| `perf_math_matmul.py` | `math_matmul_perf.cpp` | Yes (only) | in0_r ∈ {1,2,4,8,16}, in1=32x32 | 0 | `No` | 4 (in1), 2 (in0) | 1024 |
| `test_matmul.py` | `matmul_test.cpp` | **No** | 32x32 only (canonical baseline) | n/a | n/a | default | n/a |
| `test_matmul_custom.py` | `matmul_custom_test.cpp` | **No** | 32x32 only (no-MOP experimental) | n/a | n/a | default | n/a |
| `quasar/matmul_quasar_test.cpp` | (same file) | **No** | 32x32 only (hard-coded face dims) | n/a | n/a | 4 (hard-coded) | n/a |

The five-row tiny-tile axis `{1, 2, 4, 8, 16}` is produced by `generate_matmul_tiny_tiles_combinations(max_tiles)` in `helpers/matmul_sweep.py` (lines 195–218) and is the same shape every driver consumes; the differences are entirely in what is *combined with* it.

## The Three Drivers That Exercise Tiny Tiles

### `test_math_matmul.py` and `math_matmul_test.cpp`

`test_math_matmul.py` is the principal correctness driver for the math engine's matmul path: it sweeps math fidelities (LoFi, HiFi2/3/4), DstSync modes (Half, Full), DestAccumulation (Yes/No), and throttle levels (1–5 for standard tiles), and now overlays the `sweep_tiny_tiles_matmul()` axis on top of it.

The crucial bit of plumbing is at the top of the test:

```python
# tt-llk/tests/python_tests/test_math_matmul.py:60-73
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

The two sweep results are then chained — but with different throttle policies:

```python
# tt-llk/tests/python_tests/test_math_matmul.py:76-92
ALL_TEST_PARAMS = list(
    chain(
        # Regular matmul with all throttle levels
        (
            (fidelity, combinations, throttle)
            for fidelity, combinations, throttle in product(
                MATH_FIDELITIES, MATMUL_COMBINATIONS, [1, 2, 3, 4, 5]
            )
        ),
        # Tiny tiles matmul with throttle level 1 only
        (
            (fidelity, combinations, 0)
            for fidelity, combinations in product(
                MATH_FIDELITIES, TINY_TILES_MATMUL_COMBINATIONS
            )
        ),
    )
)
```

Two observations worth flagging:

1. The comment in the source says "throttle level 1 only" but the actual constant passed in is `0`. Throttle level 0 means "do not inject extra NOPs into the math inner loop." The decision to forgo a throttle sweep on tiny tiles is pragmatic: throttle's role is to space out FPU issue so the unpacker doesn't outrun the math engine; for tiny tiles the inner-loop count is already reduced (4, 8, or 16 MVMULs per face instead of the full 16x16 strided sequence), so throttle interaction is proportionally smaller. The harness opts to keep the sweep at one throttle setting and spend the test budget on the dimension axis instead.
2. The full Cartesian product is built once at module import and parametrized via `pytest.mark.parametrize`. That is what makes the tiny-tile variants visible in the nightly regression queue alongside the standard 32x32 matmuls.

On the kernel side, the math thread reads the per-variant tile dimensions and passes them straight into the LLK initializer:

```cpp
// tt-llk/tests/sources/math_matmul_test.cpp:80-88
_llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
    params.in0_tile_r_dim,
    params.in0_tile_c_dim,
    params.in1_tile_r_dim,
    params.in1_tile_c_dim,
    params.PARTIAL_FACE_MATH,
    params.UNPACK_TRANSPOSE_FACES,
    params.CT_DIM,
    params.RT_DIM);
```

The `PARTIAL_FACE_MATH` flag is what closes the loop between the Python sweep and the LLK inner-loop count: `sweep_tiny_tiles_matmul()` sets `partial_face_math = input0_dims[0] < 16`, meaning that for tiny in0 of 1, 2, 4, or 8 rows the math kernel will shrink its MVMUL count to match the active row band; for in0 of 16 rows it still uses two faces but with the full 16-row MVMUL sequence per face.

The unpacker setup (lines 28–49 of `math_matmul_test.cpp`) clamps the face-r-dim parameter with `params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM`. This is the standard tiny-tile guard you will see in every kernel source under `tests/sources/`: the face dimension passed to the unpacker is the *smaller* of the tile row dimension and 16. For an 8x32 tiny in0 the unpacker is told face_r_dim=8 and num_faces=2 (i.e. one band of two 8x16 faces side by side); for the standard 32x32 in0 the unpacker is told face_r_dim=16 and num_faces=2.

Coverage delivered: math-engine behavior for all four fidelities, both DstSync modes, both DestAcc settings, and all five tiny-row variants — with the additional axis of unpack-transpose for in1.

### `test_unpack_matmul.py` and `unpack_matmul_test.cpp`

`test_unpack_matmul.py` targets the unpacker rather than the math engine and is the only driver that sweeps stochastic rounding modes:

```python
# tt-llk/tests/python_tests/test_unpack_matmul.py:50-67
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
STOCHASTIC_ROUNDING_MODES = [
    StochasticRounding.No,
    StochasticRounding.Fpu,
    StochasticRounding.Pack,
    StochasticRounding.All,
]

FACE_MODES = [1, 2, 4]
TRANSPOSE_MODES = [Transpose.No, Transpose.Yes]
DEST_SYNC_MODES = [DestSync.Half]

MATMUL_COMBINATIONS = sweep_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
)
```

The standard `sweep_matmul()` axis here varies face mode in `{1, 2, 4}` — that is the unpacker's main test surface (1-face, 2-face, and full 4-face tiles). The tiny-tile axis is overlaid on top:

```python
# tt-llk/tests/python_tests/test_unpack_matmul.py:69-86
TINY_TILES_MATMUL_COMBINATIONS = sweep_tiny_tiles_matmul(
    MATMUL_FORMATS,
    DEST_ACC_MODES,
    STOCHASTIC_ROUNDING_MODES,
    DEST_SYNC_MODES,
)

@pytest.mark.nightly
@parametrize(
    math_fidelity=[...],
    matmul_config=MATMUL_COMBINATIONS + TINY_TILES_MATMUL_COMBINATIONS,
)
def test_unpack_matmul(math_fidelity, matmul_config):
    ...
```

Two things to note. First, `sweep_tiny_tiles_matmul()` does *not* take a face-mode argument: tiny-tile variants are fixed at `num_faces_in0 = 2` and `num_faces_in1 = 4` (see `matmul_sweep.py:514–529`). The face-mode sweep is therefore standard-tile-only; tiny tiles ride on top of that with a single face layout. Second, every tiny-tile variant is exercised against all four stochastic rounding modes — `No`, `Fpu`, `Pack`, and `All` — without any per-tile PCC tolerance adjustment. That means the harness is implicitly asserting that tiny-tile matmul outputs do not need looser tolerances under stochastic rounding than standard tiles do.

The kernel side adds one extra step compared to `math_matmul_test.cpp`: the unpacker explicitly configures stochastic rounding before initializing the matmul, which is precisely what `test_unpack_matmul.py` is built to exercise.

```cpp
// tt-llk/tests/sources/unpack_matmul_test.cpp:39-50
_llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>();
_llk_unpack_AB_matmul_init_<>(
    params.UNPACK_TRANSPOSE_FACES,
    params.CT_DIM,
    params.RT_DIM,
    params.KT_DIM,
    params.in1_tile_r_dim < FACE_R_DIM ? params.in1_tile_r_dim : FACE_R_DIM,
    params.in0_tile_r_dim < FACE_R_DIM ? params.in0_tile_r_dim : FACE_R_DIM,
    params.num_faces_B,     // in1
    params.num_faces_A,     // in0
    params.PARTIAL_FACE_B,  // in1
    params.PARTIAL_FACE_A); // in0
```

Coverage delivered: unpacker behavior for all stoch rounding combinations × all four fidelities × tiny-tile heights. This is the driver that catches unpack-side regressions when LLK's face-stride computation or partial-face zeroing is changed.

### `perf_math_matmul.py` and `math_matmul_perf.cpp`

`perf_math_matmul.py` is the performance-only driver. Its structure mirrors `test_math_matmul.py` but with one decisive twist: the standard 32x32 sweep is *commented out*, leaving only tiny tiles in the perf regression:

```python
# tt-llk/tests/python_tests/perf_math_matmul.py:68-85
ALL_TEST_PARAMS = list(
    chain(
        # Regular matmul combinations with all throttle levels
        # ( Commented to reduce number of tests since CI fails with no free space left on device
        #     (fidelity, combinations, throttle)
        #     for fidelity, combinations, throttle in product(
        #         MATH_FIDELITIES, MATMUL_COMBINATIONS, [1, 2, 3, 4, 5]
        #     )
        # ),
        # Tiny tiles matmul combinations with throttle level 1 only
        (
            (fidelity, combinations, 0)
            for fidelity, combinations in product(
                MATH_FIDELITIES, TINY_TILES_MATMUL_COMBINATIONS
            )
        ),
    )
)
```

This is worth dwelling on. The harness was forced to drop standard-tile perf coverage due to CI disk pressure (every perf variant emits its own measurement artifact) and chose to keep the tiny-tile axis. The implied prioritization is significant: tiny-tile perf data is judged more valuable to track over time than yet another standard-tile baseline.

The perf driver runs five isolation modes per variant — `L1_TO_L1`, `UNPACK_ISOLATE`, `MATH_ISOLATE`, `PACK_ISOLATE`, `L1_CONGESTION` — and inflates `LOOP_FACTOR` to 1024 (versus 16 in the correctness drivers) so that single-MVMUL latency dominates the measurement window and one-time setup overhead is amortized:

```python
# tt-llk/tests/python_tests/perf_math_matmul.py:113-161
run_types = [
    PerfRunType.L1_TO_L1,
    PerfRunType.UNPACK_ISOLATE,
    PerfRunType.MATH_ISOLATE,
    PerfRunType.PACK_ISOLATE,
    PerfRunType.L1_CONGESTION,
]
...
runtimes=[
    ...
    LOOP_FACTOR(1024),
],
```

The expected qualitative finding is that tiny-tile MVMUL cycle counts do *not* scale linearly with the row reduction: the inner loop count drops from 16 to e.g. 8 for an 8-row tile, but pipeline fill, semaphore round-trips, and unpacker stride setup remain. This driver is what makes that trend visible in the regression record.

## The Two Drivers That Do NOT Exercise Tiny Tiles

### `test_matmul.py` — canonical baseline

`test_matmul.py` is the canonical, simplest end-to-end matmul correctness test. It does not chain in a tiny-tiles sweep:

```python
# tt-llk/tests/python_tests/test_matmul.py:57-65
MATMUL_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32, DataFormat.Bfp8_b]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
ALL_MATMUL_COMBINATIONS = generate_format_aware_matmul_combinations(
    MATMUL_FORMATS, DEST_ACC_MODES
)
```

The combinations generator calls `generate_matmul_dimension_combinations(max_tiles)` only — there is no `generate_matmul_tiny_tiles_combinations` step. The rationale is that this driver is the regression *baseline*: a clean 32x32 matmul that anchors the rest of the suite. Adding tiny tiles here would change what "the canonical matmul test" means; tiny tiles get their own dedicated entry points (the three drivers above) so failures can be localized cleanly.

`test_matmul.py` is also the test that exercises `boot_mode` (`BootMode.DEFAULT` plumbed through the test signature). Mixing tiny-tile sweep into the boot-mode matrix would inflate the cross-product without buying additional coverage of tiny-tile-specific code paths.

### `test_matmul_custom.py` — no-MOP experimental

`test_matmul_custom.py` (added in 2026) has a structure that is essentially identical to `test_matmul.py` and uses the same `generate_format_aware_matmul_combinations()` helper. The kernel source it points at is `sources/matmul_custom_test.cpp`, which is a no-MOP (no Macro-OP) experimental matmul: an alternate inner-loop unrolling strategy meant to be benchmarked against the production path.

Like the canonical baseline, this driver only covers 32x32. Tiny tiles are deferred until the no-MOP variant has matured: the priority on the experimental path is to validate the unrolling correctness on the most common case (32x32 LoFi/HiFi) before adding the dimension axis. Once that variant stabilizes, layering `sweep_tiny_tiles_matmul()` on top of it is a low-cost addition; the harness is structured to make it a one-line change in the parametrize call.

## Quasar Tiny-Tile Status

The Quasar architecture has its own matmul test under `tests/sources/quasar/matmul_quasar_test.cpp`. The status is explicit and unambiguous: tiny tiles are not supported. The buffer-descriptor setup hard-codes face dimensions to 16 and num_faces to 4, with comments stating exactly that:

```cpp
// tt-llk/tests/sources/quasar/matmul_quasar_test.cpp:43-45
tdma_desc_src_a.buf_desc.f.x_dim = FACE_C_DIM;  // Default face dimension is 16, tiny tiles not supported for quasar
tdma_desc_src_a.buf_desc.f.y_dim = FACE_R_DIM;  // Default face dimension is 16, tiny tiles not supported for quasar
tdma_desc_src_a.buf_desc.f.z_dim = num_faces_A; // Number of faces = 4, tiny tiles not supported for quasar
```

The same hard-coded values appear on the `src_b` descriptor (lines 54–56). Unlike WH/BH, where the unpacker face dimensions are runtime parameters from the test harness, the Quasar test wires constants directly into the TDMA buffer descriptor table. There is no Python driver in `tests/python_tests/` that produces tiny-tile variants for the Quasar kernel either.

[Q] The implication for Chapter 5 (tt-blaze / DeepSeek production usage) is that any code path that needs to run on Quasar must currently use 32x32 tiles. Lifting that restriction will require (a) parameterizing the TDMA descriptor fields, (b) adding a `sweep_tiny_tiles_matmul()` equivalent for the Quasar test, and (c) extending `_llk_unpack_matmul_init_` semantics to cover sub-16 face heights in the Quasar LLK headers. See Chapter 7 (cross-cutting concerns) for the broader discussion.

## Summary

Of the six matmul test entry points in `tt-llk/tests/` (five Python drivers plus the Quasar C++ test), three exercise tiny tiles today: `test_math_matmul.py`, `test_unpack_matmul.py`, and `perf_math_matmul.py`. Two are deliberately 32x32-only (`test_matmul.py` as the canonical baseline, `test_matmul_custom.py` as the no-MOP experimental path), and Quasar is hard-coded out. The tiny-tile axis is always restricted to in0_r ∈ {1, 2, 4, 8, 16} with in1 fixed at 32x32 and the in0 face layout fixed at 2 faces (in0) × 4 faces (in1) — see Chapter 3, Section "Why in1 is restricted to 32x32" for the design rationale, and Chapter 3, Section "Standalone Coverage Gaps" for the eltwise/SFPU/reduce/tilize categories where no tiny-tile sweep currently exists.
