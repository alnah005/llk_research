# Compression Analysis: Build System Integration — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~607 lines
- Estimated post-compression line count: ~534 lines
- Estimated reduction: ~12%

## CRUCIAL Suggestions
None.

## MINOR Suggestions

1. **Collapse three near-identical HAL code blocks into one representative block plus a diff table.**
   `jit_compilation_pipeline.md` lines 42-125 show the Wormhole, Blackhole, and Quasar HAL include-push sequences. The three blocks share identical structure; only the architecture path segment differs (`wormhole_b0`, `blackhole`, `quasar`). Show the Wormhole block as the canonical example and summarize the other two with a short table of differences (the Blackhole firmware-src placement difference and the Quasar extra root include are already called out in prose and can remain as notes). This would save roughly 30 lines.

2. **Remove the second TRISC prolog example.**
   `per_kernel_code_generation.md` lines 39-52 show two generated prolog examples (`chlkc_unpack.cpp` and `chlkc_math.cpp`). They differ only in the `#define` name, which is already captured in the table at lines 117-122. One example is sufficient. Saves ~8 lines.

3. **Deduplicate the LLK include-path enumeration that appears in three files.**
   The same `tt_llk_${ARCH}/common/inc` and `tt_llk_${ARCH}/llk_lib` paths are listed in:
   - `submodule_and_cmake.md` lines 86-91 (firmware table)
   - `jit_compilation_pipeline.md` lines 140-147 (common-includes table)
   - `per_kernel_code_generation.md` lines 259-261 (fake-kernels target)

   The `jit_compilation_pipeline.md` table and the `per_kernel_code_generation.md` snippet could cross-reference `submodule_and_cmake.md` instead of re-enumerating the paths. Saves ~15 lines.

4. **Merge the UCK_CHLKC guard explanation with the chlkc_descriptors.h section.**
   `per_kernel_code_generation.md` lines 144-185 ("HAL Common Defines: Origin of UCK_CHLKC_* Guards") explains that UCK_CHLKC_* guards control which LLK code paths are compiled. Lines 224-228 in the chlkc_descriptors.h section restate this ("guarded by..."). Folding the guard-origin explanation into a short preamble paragraph before the descriptor header section would eliminate the restatement. Saves ~10 lines.

5. **Trim signposting sentences.**
   Several sentences serve only as forward/backward pointers that duplicate the section structure:
   - `jit_compilation_pipeline.md` line 32: "The architecture-specific LLK paths ... are NOT set here -- they come from the HAL, as described next."
   - `per_kernel_code_generation.md` line 185: "The TRISC_* guards control code paths within the user's kernel source, while the UCK_CHLKC_* guards control code paths within the LLK library and generated descriptor headers." (restates what lines 146 and 53 already say)
   These can be shortened to half their length without losing meaning. Saves ~5 lines.

## Load-Bearing Evidence

- **`index.md`** line 9: `"selecting the correct architecture-specific subdirectory (tt_llk_wormhole_b0/, tt_llk_blackhole/, or tt_llk_quasar/)"` -- this enumeration is repeated in the directory tree at `submodule_and_cmake.md` lines 18-29 and in the firmware table at lines 86-91.
- **`submodule_and_cmake.md`** lines 86-91: The three-row `ARCH_ALIAS` to include-path table enumerates the same paths that `jit_compilation_pipeline.md` lines 140-147 re-enumerates in a different table.
- **`jit_compilation_pipeline.md`** lines 42-62 vs 67-96 vs 102-125: Three architecture HAL blocks with identical structure differing only by path name.
- **`per_kernel_code_generation.md`** lines 39-52: Two TRISC prolog examples (`chlkc_unpack.cpp` and `chlkc_math.cpp`) that differ by a single `#define` token, already captured in the table at lines 117-122.

## VERDICT
- Crucial updates: no
