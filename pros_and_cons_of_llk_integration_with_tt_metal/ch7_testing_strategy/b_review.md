# Chapter 7 -- Agent B (Critic) Review

**Reviewer:** Agent B (Pass 1)
**Scope:** Flag only issues where a reader would get a wrong answer, implement something incorrectly, or be materially misled. Max 5 items.

---

## Issues Found: 2

### Issue 1: Wrong macro name in metal_integration_testing.md

**File:** `metal_integration_testing.md`, line 70
**Claim:** "The `SFPU_OP` macro expands to calls like `exp_tile`, `gelu_tile`, or `recip_tile`"
**Actual:** The macro used in `eltwise_sfpi.cpp` (line 28) is `SFPI_OP_AND_PACK`, not `SFPU_OP`. The comment block in the source file (lines 23-26) describes the concept but the actual C preprocessor token is `SFPI_OP_AND_PACK`. A reader searching the codebase for `SFPU_OP` would find nothing; they need to search for `SFPI_OP_AND_PACK`.
**Impact:** Someone trying to locate or modify the macro injection mechanism would fail to find it.
**Source:** `/localdev/salnahari/testing_dir/tt-metal/tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpi.cpp`

### Issue 2: fake_kernels_target scope omits kernels_ng/ directories

**File:** `metal_integration_testing.md`, lines 82-83
**Claim:** The CMakeLists.txt "globs all kernel `.cpp` files across `tt_metal/kernels/`, `tt_metal/impl/dispatch/kernels/`, `tt_metal/fabric/impl/kernels/`, and `ttnn/cpp/ttnn/operations/*/kernels/`"
**Actual:** The CMakeLists.txt (line 31) filters for both `*/kernels/` and `*/kernels_ng/` path patterns. The `kernels_ng/` directories contain real kernel files (e.g., binary_ng dataflow and compute kernels under `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/`). Omitting `kernels_ng/` from the description understates the build scope.
**Impact:** A reader trying to replicate or extend the fake_kernels_target build, or trying to understand which kernel files are compile-checked on submodule update, would miss an entire class of kernel directories.
**Source:** `/localdev/salnahari/testing_dir/tt-metal/tt_metal/jit_build/fake_kernels_target/CMakeLists.txt` (line 31: `if(file MATCHES ".*/kernels/|.*/kernels_ng/")`)

---

## Verified Claims (no issues found)

- 158 SFPU header files in `wormhole_b0/metal/llk_api/llk_sfpu/` -- confirmed (exact count: 158)
- "over 40 Python modules" in helpers/ -- confirmed (43 files including `__init__.py`)
- All referenced file paths (test sources, linker scripts, helper modules, CMakeLists, fake_jit_prelude.h, test kernels) -- confirmed to exist
- Code snippets for `eltwise_binary_test.cpp` and `eltwise_copy_block.cpp` -- faithful paraphrases of actual source
- `fake_jit_prelude.h` hardcoded values (MATH_FIDELITY=255, format arrays all 255) -- confirmed
- `llk_math_eltwise_binary` wrapper function name in `llk_math_binary_api.h` -- confirmed
- LLK test structure with `#ifdef LLK_TRISC_UNPACK` guards and `run_kernel()` entry point -- confirmed
- Performance test files (`perf_*.py`) and C++ perf sources (`*_perf.cpp`) -- confirmed to exist
