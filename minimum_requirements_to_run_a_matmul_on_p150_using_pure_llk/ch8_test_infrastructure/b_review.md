# Chapter 8 -- Agent B Factual Review (Pass 1)

## Verdict: CONDITIONAL APPROVAL

Chapter 8 is remarkably thorough and accurate. All file names and paths are correct, all line counts match exactly, the experimental header inventory is complete (13/13), the stochastic rounding attribution is correct, the isolation mode table is correct, and the LLK API call descriptions match the source code. Two factual errors require correction; the remaining findings are minor imprecisions.

---

### Finding 1: Incorrect L1 stimuli base address for coverage builds
**File:** `02_python_test_harness.md`, Section 8.2.3
**Error:** The text states: "The base address depends on the build mode: `0x21000` for standard builds, `0x6E000` for coverage builds." The actual stimuli base address for coverage builds is `0x70000` (`STIMULI_L1_ADDRESS_DEBUG`), not `0x6E000`. The `0x6E000` value is the *runtime parameter* address for coverage builds (`RUNTIME_ADDRESS_COVERAGE` in `test_config.py`), which is a different address space.
**Evidence:**
- `tests/python_tests/helpers/stimuli_config.py` line 41: `STIMULI_L1_ADDRESS_PERF = 0x21000`
- `tests/python_tests/helpers/stimuli_config.py` line 42: `STIMULI_L1_ADDRESS_DEBUG = 0x70000`
- `tests/python_tests/helpers/test_config.py` line 182: `RUNTIME_ADDRESS_NON_COVERAGE: ClassVar[int] = 0x20000`
- `tests/python_tests/helpers/test_config.py` line 183: `RUNTIME_ADDRESS_COVERAGE: ClassVar[int] = 0x6E000`
**Fix:** Change `0x6E000` to `0x70000` in Section 8.2.3 when referring to the stimuli base address. Optionally add a note distinguishing the stimuli base address (`0x21000` / `0x70000`) from the runtime parameter address (`0x20000` / `0x6E000`).

---

### Finding 2: Section 8.2.15 conflates runtime parameter and stimuli addresses
**File:** `02_python_test_harness.md`, Section 8.2.15
**Error:** The text states: "Coverage builds use a different runtime parameter address (`0x6E000` vs. `0x20000`) to avoid conflicting with the coverage data region. The `StimuliConfig` class also adjusts its L1 buffer allocation to account for the reduced available memory." This is technically correct for the runtime parameter addresses but the `0x20000` vs `0x6E000` values belong to `TestConfig`, not `StimuliConfig`. Section 8.2.3 (Finding 1) incorrectly attributes `0x6E000` to the stimuli base, so these two sections are internally inconsistent.
**Fix:** Ensure consistent terminology: runtime parameter addresses are `0x20000` (standard) and `0x6E000` (coverage) from `TestConfig`; stimuli base addresses are `0x21000` (standard) and `0x70000` (coverage) from `StimuliConfig`.

---

### Finding 3: Master table row #2 throttle range is incomplete
**File:** `01_matmul_test_survey.md`, Section 8.1.2, row #2
**Error:** The "Special Features" column for `test_math_matmul.py` states "Throttle 1--5" but the actual test also uses throttle level 0 for tiny tile configurations. See `test_math_matmul.py` lines 86-92: regular matmul uses `[1, 2, 3, 4, 5]` and tiny tiles use throttle `0`.
**Fix:** Change "Throttle 1--5" to "Throttle 0--5 (0 for tiny tiles, 1--5 for standard)" or add a footnote. Note: Section 8.2.9 already correctly states "Throttle level 0 means no throttling (tiny tiles only run at level 0)" so this is only a table incompleteness, not a conceptual error.

---

### Finding 4: Stochastic rounding attribution is CORRECT (positive confirmation)
**File:** `01_matmul_test_survey.md`, Section 8.1.2
**Verification:** The chapter correctly attributes `_llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>()` to `unpack_matmul_test.cpp` (line 39) and correctly notes that `math_matmul_test.cpp` does NOT contain this call. The note in Section 8.1.2 explicitly addresses this common confusion: "The C++ kernel `math_matmul_test.cpp` does not call `_llk_unpack_configure_stoch_rnd_`. Actual stochastic rounding sweeps are exclusive to `test_unpack_matmul.py`." This is verified as factually correct.

---

### Finding 5: All file names, paths, and line counts are correct (positive confirmation)
**File:** All three files
**Verification results:**
- `matmul_test.cpp`: claimed 119 lines, actual 119 lines
- `math_matmul_test.cpp`: claimed 155 lines, actual 155 lines
- `unpack_matmul_test.cpp`: claimed 151 lines, actual 151 lines
- `matmul_custom_test.cpp`: claimed 123 lines, actual 123 lines
- `matmul_pack_untilize_test.cpp`: claimed 90 lines, actual 90 lines
- `matmul_unpack_tilize_test.cpp`: claimed 197 lines, actual 197 lines
- `matmul_and_unary_sfpu_test.cpp`: claimed 160 lines, actual 160 lines
- `matmul_perf.cpp`: claimed 250 lines, actual 250 lines
- `math_matmul_perf.cpp`: claimed 276 lines, actual 276 lines
- `quasar/matmul_quasar_test.cpp`: claimed 134 lines, actual 134 lines
- `llk_math_matmul.h`: claimed 676 lines, actual 676 lines
- Experimental headers: claimed 13, actual 13, all names match exactly
- All Python test files exist at claimed paths
- All helper modules exist at claimed paths

---

### Finding 6: All LLK API call attributions verified correct
**File:** `01_matmul_test_survey.md`, Section 8.1.2 master table
**Verification:** Each row's "Key LLK API Differences" column was cross-referenced against the actual C++ source. All API calls are correctly attributed:
- Row 1 (`matmul_test.cpp`): `_llk_math_matmul_init_<MATH_FIDELITY>` confirmed (line 72, single template param)
- Row 2 (`math_matmul_test.cpp`): `THROTTLE_LEVEL` template on init/exec confirmed (lines 81, 98)
- Row 3 (`unpack_matmul_test.cpp`): `_llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>()` confirmed (line 39)
- Row 4 (`matmul_custom_test.cpp`): `_llk_math_matmul_init_no_mop_` / `_llk_math_matmul_no_mop_` confirmed (lines 81, 88)
- Row 5 (`matmul_pack_untilize_test.cpp`): `_llk_pack_untilize_init_` / `_llk_pack_untilize_` confirmed (lines 79, 86); `_llk_math_reconfig_remap_` on BH confirmed (line 56)
- Row 6 (`matmul_unpack_tilize_test.cpp`): `_llk_unpack_tilize_init_` / `_llk_unpack_tilize_` confirmed (lines 57-63); `_llk_math_eltwise_unary_datacopy_` confirmed (lines 120-126); semaphore ops confirmed (lines 65-67, 178)
- Row 7 (`matmul_and_unary_sfpu_test.cpp`): `_llk_math_eltwise_unary_sfpu_init_` / `_llk_math_eltwise_unary_sfpu_start_` confirmed (lines 103-104)
- Row 10 (`matmul_quasar_test.cpp`): `_llk_unpack_matmul_init_` / `_llk_unpack_matmul_` confirmed (lines 65, 70); `_llk_math_matmul_block_` confirmed (line 96); `_llk_pack_matmul_init_` / `_llk_pack_matmul_` confirmed (lines 128, 130)

---

### Finding 7: Format lists verified correct for all tests
**File:** `01_matmul_test_survey.md`, Section 8.1.2 master table
**Verification:**
- `test_matmul.py`: Float16_b, Float16, Float32, Bfp8_b -- confirmed (lines 59-60)
- `test_math_matmul.py`: Float16_b, Float16, Float32 -- confirmed (lines 44-49)
- `test_unpack_matmul.py`: Float16_b, Float16, Float32 -- confirmed (lines 43-49)
- `test_matmul_custom.py`: Float16_b, Float16, Float32, Bfp8_b -- confirmed (lines 62-65)
- `test_matmul_pack_untilize.py`: Float16_b, Float16, Bfp8_b (input only), Float32 -- confirmed; Bfp8_b output skipped confirmed (lines 41-42)
- `test_zzz_matmul_and_unary_sfpu.py`: Float16, Float16_b, Float32, Bfp8_b -- confirmed (lines 46-49)
- `quasar/test_matmul_quasar.py`: Float16, Float16_b -- confirmed (lines 68-69)

---

### Finding 8: Performance test isolation modes verified correct
**File:** `01_matmul_test_survey.md`, Section 8.1.8
**Verification:** The five isolation modes (L1_TO_L1, UNPACK_ISOLATE, MATH_ISOLATE, PACK_ISOLATE, L1_CONGESTION) and their per-TRISC behavior (real/mock/returns early) were verified against `matmul_perf.cpp`. All entries in the table match the `if constexpr (PERF_RUN_TYPE == ...)` branches in the source code. LOOP_FACTOR values (16 for perf_matmul, 1024 for perf_math_matmul) confirmed.

---

### Finding 9: Quasar test description verified correct
**File:** `01_matmul_test_survey.md`, Section 8.1.7
**Verification:** TDMA descriptor usage, `set_up_dest_dvalid_per_thread`, `_llk_math_set_dvalid_`, BootMode.TRISC, ImpliedMathFormat, and DstSync Half/Full all confirmed against source code.

---

### Finding 10: conftest.py infrastructure description verified correct
**File:** `02_python_test_harness.md`, Section 8.2.16
**Verification:** `pytest_configure`, `pytest_sessionstart` (GO_BUSY), `pytest_sessionfinish` (GO_IDLE), `workers_tensix_coordinates` fixture with `divmod(int(worker_id[2:]), 8)`, `pytest_runtest_makereport` with `LLKAssertException` and `TimeoutError` handling, all CLI options (--coverage, --compile-producer, --compile-consumer, --detailed-artefacts, --no-debug-symbols, --speed-of-light, --logging-level, --run-simulator, --reset-simulator-per-test) all confirmed in `conftest.py`.

---

### Finding 11: Experimental header directory path is correct
**File:** `01_matmul_test_survey.md`, Section 8.1.11
**Verification:** The chapter references `tt_llk_blackhole/llk_lib/experimental/` which is the correct path. All 13 headers listed in the table exist at this path with the exact names shown.

---

### Finding 12: Fuser config YAML verified correct
**File:** `01_matmul_test_survey.md`, Section 8.1.12
**Verification:** The YAML content shown in the chapter matches the actual file at `tests/python_tests/fuser_config/fpu_matmul.yaml` exactly -- operations structure, dimensions (256x256), formats (Float16_b), fidelity (HiFi2), and block_size ([64, 128]) all match.

---

### Finding 13: Skip conditions verified correct
**File:** `01_matmul_test_survey.md`, Section 8.1.2
**Verification:**
- `test_matmul_custom.py` skips on Wormhole (`get_chip_architecture() == ChipArchitecture.WORMHOLE`) -- confirmed (line 89-90)
- `test_zzz_matmul_and_unary_sfpu.py` skipped on both Blackhole and Wormhole -- confirmed via `@skip_for_blackhole` and `@skip_for_wormhole` decorators (lines 40-41)

---

### Finding 14: Alternatives analysis (Section 8.3) is technically sound
**File:** `03_alternatives_to_llk.md`
**Verification:** The eight alternatives analyzed are comprehensive and the conclusions are architecturally valid. The key claims -- that MVMUL is the sole matmul instruction, that it requires unpacker/FPU/packer configuration, that SFPU cannot perform matmul (unary-only on dest register), and that software RISC-V matmul would be ~1000x slower -- are all consistent with the hardware architecture documented in earlier chapters and the source code structure. The minimum API surface of 15 LLK calls (3 unpack + 6 math + 6 pack) matches the actual calls in `matmul_test.cpp`.

---

## Summary

**Accuracy rate:** Very high. Out of dozens of verifiable technical claims, only 2 factual errors were found (both involving L1 address values in Section 8.2), plus 1 minor table incompleteness. All file names (20+), all line counts (11 files), all API call attributions (10 test variants), all format lists (7 tests), all experimental header names (13), all conftest features (10+ CLI options), all isolation mode behaviors (5 modes x 3 TRISCs), and all skip conditions are correct.

**Required fixes (2):**
1. Section 8.2.3: Change coverage stimuli base address from `0x6E000` to `0x70000`
2. Section 8.2.15: Ensure address references are consistent with Section 8.2.3

**Recommended fix (1):**
3. Master table row #2: Expand throttle range from "1--5" to "0--5" or add a footnote about tiny tiles using throttle 0

The stochastic rounding attribution -- flagged as a common error across generator versions -- is handled correctly throughout the chapter, with an explicit note in Section 8.1.2 that preempts confusion.

---

## Fixes Applied (2026-05-02)

- **Finding 1 fixed:** Changed coverage stimuli base address from `0x6E000` to `0x70000` in Section 8.2.3. Added note distinguishing stimuli base (`0x21000`/`0x70000`) from runtime parameter address (`0x20000`/`0x6E000`).
- **Finding 2 fixed:** Section 8.2.15 now consistently references both address pairs.

**APPROVED**
