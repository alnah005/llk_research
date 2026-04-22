# Agent B Review -- Chapter 6: Template Design Trade-offs

## Issue 1 (Incorrect numerical answer) -- `EltwiseBinaryType` count and combinatorial total

**Files:** `compilation_impact.md` lines 30, 37; `index.md` line 26

The table in `compilation_impact.md` lists 3 values for `EltwiseBinaryType` (`ELWADD`, `ELWSUB`, `ELWMUL`) and calls the resulting product of 576 the "theoretical maximum." However, the actual enum in `tt_llk_blackhole/llk_lib/llk_defs.h:51-58` defines **5** values: `ELWMUL`, `ELWDIV`, `ELWADD`, `ELWSUB`, `ELWLESS`. If the intent is "theoretical maximum," the count should be 5, yielding 5 x 4 x 2 x 2 x 4 x 3 = **960**, not 576. The table header says "Typical values," which partially hedges this, but then the text calls 576 the "theoretical maximum" -- those two framings contradict each other. Either label 576 as "practical/typical maximum" or use the full enum count and report 960.

**Source:** `/localdev/salnahari/testing_dir/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h` lines 51-58.

## Issue 2 (Materially misleading) -- `DstSync` enum value names are wrong

**File:** `compilation_impact.md` line 32

The table lists `DstSync` values as `Half` and `Full`. The actual enum values are `SyncHalf` and `SyncFull` (see `tt_llk_blackhole/llk_lib/llk_defs.h:67-71`). A reader trying to use these names in code would get a compile error.

**Source:** `/localdev/salnahari/testing_dir/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h` lines 67-71.

## Issue 3 (Could mislead implementation) -- `if constexpr` count of "over 40" understates Blackhole, overstates for the file cited

**Files:** `index.md` line 34; `binary_size_and_alternatives.md` line 10

Both files claim "over 40 `if constexpr` branch points" in `llk_math_eltwise_binary.h`. A count across all three arch variants in tt-llk shows 26 occurrences in the Blackhole version of this file, 29 in the Wormhole B0 version, and 9 in Quasar -- for a total of 64 across all three files. However, since the text refers to "a single file," the relevant count for Blackhole is **26**, not "over 40." The claim is numerically wrong; either the count methodology should be clarified or the number corrected.

**Source:** `grep -c "if constexpr"` on `/localdev/salnahari/testing_dir/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h` returns 26.

---

**Items reviewed and found correct:**
- Template parameter list and signature for `_llk_math_eltwise_binary_` (lines 571-578 of Blackhole file) -- matches source exactly.
- Metal wrapper template parameters and `DST_SYNC_MODE` injection -- matches source.
- Compiler flags (`-O3`, `-flto=auto`, `-Os` for non-compute) -- confirmed in `build.cpp` lines 147, 314-315, 319-320.
- JIT cache hash and "JIT build cache hit" log message -- confirmed at `build.cpp` lines 431-432 and 566.
- `ccache` integration via `TT_METAL_CCACHE_KERNEL_SUPPORT` -- confirmed at `build.cpp` lines 105-107.
- `constexpr` hardware configuration values (lines 25-26 of Blackhole file) -- match source exactly.
- `static_assert` for `binary_reuse_dest` -- confirmed at line 393.
- Code snippets for `eltwise_binary_func` and dispatch pattern -- match source.
