# Coverage Gaps in tt-llk Tiny-Tile Testing

Tiny-tile testing in tt-llk is, today, a matmul-only story. The three drivers that exercise sub-32-row tiles (`test_math_matmul.py`, `test_unpack_matmul.py`, and `perf_math_matmul.py`) all share the same `sweep_tiny_tiles_matmul()` helper and the same in0-only restriction; no other op family has an equivalent. This section enumerates what is missing, why each gap exists, and concretely how to close it with new drivers that follow the conventions already established by the matmul tests.

**Prerequisites:**
- Chapter 2, all sections (face-count rules, SFPU iteration count rule, partial-face semantics).
- Chapter 3, Sections 1–3 (the matmul-test inventory, the `sweep_tiny_tiles_matmul()` helper, and the `MatmulGolden` tilize path).
- Familiarity with the LLK test harness layout from [[minimum_requirements_to_run_a_matmul_on_p150_using_pure_llk]].

---

## 1. Overview of Coverage Gaps

Surveying the LLK test directory, the operations that currently have tiny-tile parametrization are:

| Test driver | Op family | Tiny-tile coverage |
| --- | --- | --- |
| `tests/test_math_matmul.py` + `sources/math_matmul_test.cpp` | Matmul | Yes (in0 ∈ {1, 2, 4, 8, 16}x32) |
| `tests/test_unpack_matmul.py` + `sources/unpack_matmul_test.cpp` | Matmul (unpack focus, SR modes) | Yes |
| `tests/perf_math_matmul.py` + `sources/math_matmul_perf.cpp` | Matmul (perf) | Yes (perf-only; standard matmul perf is disk-budgeted out) |
| `tests/test_matmul.py`, `tests/test_matmul_custom.py` | Matmul | No — only `generate_matmul_dimension_combinations()` (32x32) |
| `tests/test_matmul_pack_untilize.py` | Pack-untilize fused with matmul | No tiny variant |
| `tests/test_matmul_unpack_tilize.py` | Unpack-tilize fused with matmul | No tiny variant |
| Any standalone SFPU / reduce / transpose / tilize / untilize test | Non-matmul | No tiny variant |
| `tests/quasar/` matmul test (`sources/matmul_quasar_test.cpp`) | [Q] Matmul | No — face dims are hard-coded |

This produces five concrete gap categories:

1. **Standalone eltwise SFPU at tiny geometry** — only exercised via matmul-fused SFPU.
2. **Standalone reduce at tiny geometry** — no driver.
3. **Standalone transpose at tiny geometry** — only via the matmul `unpack_transpose_faces` path.
4. **Standalone tilize / untilize at tiny geometry** — driver exists for 32x32 only.
5. **[Q] Quasar tiny-tile matmul** — hard-coded `FACE_R_DIM`, no parametrization.

The remainder of this section walks each one and proposes a closing test in the style of the existing matmul drivers.

---

## 2. Eltwise / SFPU Operations at Tiny Geometry

### 2.1 Current coverage (matmul-fused only)

The only place an SFPU runs against a tiny operand in LLK regression today is the matmul-fused SFPU test (`matmul_and_unary_sfpu_test.cpp`, referenced by the file inventory in [[minimum_requirements_to_run_a_matmul_on_p150_using_pure_llk]]). That test composes `_llk_math_matmul_<>()` with `_llk_math_eltwise_unary_sfpu_<>()` on the matmul output tile in Dst. But the SFPU phase there only ever sees a 32x32 output tile because the matmul-fused tests use the same restriction as `sweep_tiny_tiles_matmul()`: in0 is variable, in1 is 32x32, and the output tile inherits in0's row dimension only on the math side — by the time the output reaches Dst for SFPU, it is the full-tile shape produced by pack.

This means the SFPU iteration-count rule documented in Chapter 2, Section "SFPU iteration count rule" is **never exercised at N=2 against a true tiny destination**. An SFPU iterating over a tile that lives in Dst at sub-16-row geometry has no test today; the regression suite has no signal on whether `_llk_math_eltwise_unary_sfpu_<>(0, VectorMode::R)` correctly handles a `face_r_dim < 16` destination layout.

### 2.2 Why standalone tests are needed

The SFPU iteration count is `num_faces`. For a tiny tile of any of {1, 2, 4, 8, 16} rows × 32 columns, the face layout is 2 horizontal faces (f0, f1), so `N=2`. The rule is the same as for 16x32 — but for 1, 2, 4, and 8 rows, the face is **partial** (`partial_face = (row_dim < 16)`), which means the SFPU must operate over a Dst region whose `face_r_dim` is less than `MAX_FACE_R_DIM`. A bug in the SFPU phase that depends on `face_r_dim == 16` would be silent against the current regression. A standalone test pins this contract down.

### 2.3 Proposed test structure

```
tests/test_eltwise_tiny_tile.py
sources/eltwise_tiny_tile_test.cpp
```

Parameter space, modelled on `test_unpack_matmul.py`:

- `tile_rows ∈ {1, 2, 4, 8, 16}` (5 variants; widths fixed at 32).
- `sfpu_op ∈ {sigmoid, silu, exp, gelu, reciprocal}` (5 ops; pick a representative SFPU surface).
- `data_format ∈ {Float16_b, Float32}`.
- `math_fidelity ∈ {LoFi, HiFi4}` (only matters for fidelity-masked SFPU paths).

Driver pattern (modeled on `test_unpack_matmul.py:69–85`):

```python
# tests/test_eltwise_tiny_tile.py
@pytest.mark.parametrize(
    "in0_dims, sfpu_op, format",
    sweep_eltwise_tiny_tiles(SFPU_OPS, FORMATS),
)
def test_eltwise_tiny_tile(in0_dims, sfpu_op, format, ...):
    ...
```

The C++ kernel mirrors the unpack/math/pack phase split in `math_matmul_test.cpp`, but the math phase calls SFPU rather than MVMUL:

```cpp
// sources/eltwise_tiny_tile_test.cpp (proposed)
_llk_math_eltwise_unary_sfpu_init_<SFPU_OP>();
_llk_math_eltwise_unary_sfpu_<SFPU_OP>(/*dst_index=*/0, VectorMode::R);
```

Golden: a plain element-wise reference (e.g., `torch.sigmoid(input_tensor)`); no `MatmulGolden` involvement, but the tilize path from `tilize_untilize.py:22–114` is still used to produce the expected packed output (`tile_dimensions=[tile_rows, 32]`, `face_r_dim=tile_rows if tile_rows<16 else 16`).

LLK functions validated: `_llk_math_eltwise_unary_sfpu_init_<>()`, `_llk_math_eltwise_unary_sfpu_<SFPU_OP>()`, and the partial-face handling inside the SFPU init for `face_r_dim != 16`.

---

## 3. Reduce Operations at Tiny Geometry

A reduce across rows of an 8x32 input produces a 1x32 output: a true tiny tile on both sides. There is no driver in tt-llk that exercises this today. The closest exercise is the matmul reduction along K, but that uses full 32x32 partial faces accumulated into Dst, not a single-pass reduce LLK call.

Proposed structure:

```
tests/test_reduce_tiny_tile.py
sources/reduce_tiny_tile_test.cpp
```

Parameter space:

- `tile_rows ∈ {8, 16}` (the two interesting tiny rows for reduce — 1, 2, 4 are degenerate but could be added).
- `reduce_op ∈ {Sum, Max}` (and optionally `Avg` if exposed).
- `reduce_dim ∈ {ReduceDim::REDUCE_ROW, ReduceDim::REDUCE_COL}` — row reduction is the interesting case because it collapses the partial-face row dimension to 1.

LLK functions validated: `_llk_math_reduce_init_<>()` and `_llk_math_reduce_<>()` with `face_r_dim < 16`. The golden is `torch.sum(input, dim=-2)` (for row-reduce) or `torch.max(input, dim=-2)[0]`.

The SFPU iteration count rule applies here too: input has 2 horizontal faces, so reduce iterates twice per tile.

---

## 4. Transpose Operations at Tiny Geometry

The matmul tests already exercise a transposed unpack via `unpack_transpose_faces` (test_math_matmul.py, test_unpack_matmul.py), but **only as part of matmul**. A standalone transpose op — i.e., a kernel that calls `_llk_math_transpose_xy_<>()` directly — has no tiny-tile driver.

The interesting tiny case for transpose is `8x32 → 32x8`: a transpose of a partial-face tile produces a tile whose **columns** are partial. This swaps which axis triggers the face-fullness check, and the current tests do not cover it.

Proposed structure:

```
tests/test_transpose_tiny_tile.py
sources/transpose_tiny_tile_test.cpp
```

Parameter space:

- `tile_rows ∈ {1, 2, 4, 8, 16}`.
- `data_format ∈ {Float16_b, Float32}`.

Golden: `torch.transpose(input, -2, -1)` followed by `tilize_block(..., dimensions=(32, tile_rows), tile_dimensions=[32, tile_rows])`. Note that this is the first place in LLK testing where `tile_dimensions[1] < 32` (i.e., **tiny in the column axis** rather than the row axis), which exercises a code path `tilize_untilize.py:60` and `tilize_untilize.py:94–114` do not currently see in regression.

---

## 5. Tilize / Untilize at Tiny Geometry

`test_matmul_pack_untilize.py` and `test_matmul_unpack_tilize.py` validate the pack-untilize and unpack-tilize fused paths at 32x32 only. The standalone tilize/untilize operations also need tiny-tile coverage because the pack untilize / unpack tilize machinery in `_llk_pack_untilize_<>()` and `_llk_unpack_tilize_<>()` reads the per-tile geometry from runtime parameters.

The matmul golden already relies on `tilize_block(..., tile_dimensions=[tile_rows, 32])` (golden_generators.py:1103–1108, tilize_untilize.py:60). The C++ side of that path — i.e., the on-device pack-untilize when the tile is tiny — is exercised indirectly by `sweep_tiny_tiles_matmul()` because `partial_face_pack = True` (matmul_sweep.py:514–529). But the **unpack-tilize** direction at tiny geometry is not exercised: tiny tiles enter the matmul tests already tilized in L1 (the host produces the tilized stream).

Proposed structure:

```
tests/test_tilize_untilize_tiny_tile.py
sources/tilize_untilize_tiny_tile_test.cpp
```

Parameter space:

- `tile_rows ∈ {1, 2, 4, 8, 16}`, columns 32.
- `direction ∈ {tilize, untilize}`.
- `num_faces ∈ {2, 4}` — to cover both the partial-face (2-face) and full (4-face) layouts.

LLK functions validated: `_llk_unpack_tilize_init_<>()`, `_llk_unpack_tilize_<>()`, `_llk_pack_untilize_init_<>()`, `_llk_pack_untilize_<>()` with `face_r_dim < 16`.

---

## 6. [Q] Quasar Tiny-Tile Matmul

The Quasar matmul test (`tests/quasar/` driver + `sources/matmul_quasar_test.cpp`) explicitly bails on tiny tiles. The relevant declaration:

```cpp
// sources/matmul_quasar_test.cpp:43-45
tdma_desc_src_a.buf_desc.f.x_dim = FACE_C_DIM;  // Default face dimension is 16, tiny tiles not supported for quasar
tdma_desc_src_a.buf_desc.f.y_dim = FACE_R_DIM;  // Default face dimension is 16, tiny tiles not supported for quasar
tdma_desc_src_a.buf_desc.f.z_dim = num_faces_A; // Number of faces = 4, tiny tiles not supported for quasar
```

And the corresponding mirror for `src_b`. These TDMA buffer descriptor fields drive the Quasar DMA engine's face-step traversal; they are static compile-time parameters in the current test (no `RUNTIME_PARAMETERS` block analogous to the Blackhole/Wormhole tests), so any tiny-tile variant requires:

1. Promoting `x_dim` / `y_dim` / `z_dim` to runtime parameters (or to template parameters expanded by the Python driver).
2. Wiring `partial_face_math` and `partial_face_pack` through the Quasar math/pack init helpers.
3. Updating the TDMA descriptor builder to compute strides from the runtime `face_r_dim`.

This is a non-trivial restructuring — bigger than adding any of the other four gap tests above — and is best filed as its own follow-up rather than as part of a single tiny-tile coverage PR.

Proposed structure when picked up:

```
tests/quasar/test_matmul_quasar_tiny.py
sources/quasar/matmul_quasar_tiny_test.cpp
```

Parameter space, modeled on `sweep_tiny_tiles_matmul()`:

- `in0_tile_r_dim ∈ {1, 2, 4, 8, 16}`, columns 32.
- in1 fixed at 32x32 (mirrors the existing Blackhole restriction; see matmul_sweep.py:300 — "TODO: These can be removed when tiny tiles are supported for both in0 and in1").
- `math_fidelity ∈ {LoFi, HiFi4}`.

LLK functions validated: the Quasar equivalents of `_llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>()` with partial-face TDMA descriptors.

---

## 7. Recommended Test Sources and Naming Conventions

A consolidated plan for closing the gaps, following the existing tt-llk naming pattern (`test_<op>.py` driver + `sources/<op>_test.cpp` kernel; the Python driver parametrizes and dispatches, the C++ kernel implements one phase split per build target):

| Gap | Python driver | C++ kernel | Parameter space (rough) | Key LLK calls validated |
| --- | --- | --- | --- | --- |
| Eltwise SFPU | `tests/test_eltwise_tiny_tile.py` | `sources/eltwise_tiny_tile_test.cpp` | 5 rows × 5 ops × 2 fmts × 2 fidelities ≈ 100 | `_llk_math_eltwise_unary_sfpu_<>()` |
| Reduce | `tests/test_reduce_tiny_tile.py` | `sources/reduce_tiny_tile_test.cpp` | 2 rows × 2 ops × 2 dims ≈ 8 | `_llk_math_reduce_<>()` |
| Transpose | `tests/test_transpose_tiny_tile.py` | `sources/transpose_tiny_tile_test.cpp` | 5 rows × 2 fmts ≈ 10 | `_llk_math_transpose_xy_<>()` |
| Tilize / untilize | `tests/test_tilize_untilize_tiny_tile.py` | `sources/tilize_untilize_tiny_tile_test.cpp` | 5 rows × 2 dirs × 2 face-counts ≈ 20 | `_llk_unpack_tilize_<>()`, `_llk_pack_untilize_<>()` |
| [Q] Quasar matmul | `tests/quasar/test_matmul_quasar_tiny.py` | `sources/quasar/matmul_quasar_tiny_test.cpp` | 5 in0-rows × 2 fidelities ≈ 10 (+ a TDMA-desc refactor) | Quasar `_llk_math_matmul_<>()` |

### 7.1 Driver pattern to copy

The combine-standard-and-tiny-into-a-single-parametrize-list pattern from `test_unpack_matmul.py:69–74` is the recommended shape:

```python
# tests/test_unpack_matmul.py (reference pattern, lines 69-85)
@pytest.mark.parametrize(
    "math_fidelity, in0_dims, in1_dims, stochastic_rounding, ...",
    sweep_matmul(..., math_matmul=False) + sweep_tiny_tiles_matmul(..., math_matmul=False),
)
def test_unpack_matmul(...):
    ...
```

Each proposed driver should expose a `sweep_<op>(...)` helper (standard geometry) and a `sweep_tiny_tiles_<op>(...)` helper (tiny geometry), so that future tests can pick either or both. This mirrors the matmul split in `matmul_sweep.py` (generate_matmul_dimension_combinations vs. generate_matmul_tiny_tiles_combinations) and keeps the per-op sweep file as the single source of truth for what "tiny" means for that op.

### 7.2 Golden and tilization

For every gap test except Quasar, the host-side reference output must be tilized via `tilize_block(input_tensor, dimensions, stimuli_format, tile_dimensions=[tile_rows, tile_cols], face_r_dim=tile_rows if tile_rows<16 else 16)` (tilize_untilize.py:22–29, 94–114). Reusing this helper guarantees the expected output bytes match the on-device layout for partial-face tiles, exactly as `MatmulGolden.tilize_block()` already does for matmul (golden_generators.py:1103–1108).

### 7.3 Performance follow-on

Once the functional tests above land, the perf-test pattern from `perf_math_matmul.py:113–123` (`LOOP_FACTOR=1024`, 5 isolation modes) is the recommended next step for each op family. The expectation — same as for matmul today — is that tiny-tile cycle counts do not scale linearly with row reduction, because the math-engine inner-loop structure is preserved.

---

## Summary

Tiny-tile coverage in tt-llk is currently load-bearing on matmul: three drivers, one helper, in0-only. Five distinct gaps (SFPU, reduce, transpose, tilize/untilize, Quasar) sit immediately downstream of the rules and constraints documented in Chapter 2 and the helper machinery documented in Chapter 3, Sections 1–3. Closing the first four is a matter of mechanical replication of the matmul driver pattern; closing the Quasar gap requires a TDMA-descriptor refactor and is appropriately a separate workstream. See Chapter 5 for how production tt-metal usage currently masks these gaps (and where, conversely, real models would exercise them).
