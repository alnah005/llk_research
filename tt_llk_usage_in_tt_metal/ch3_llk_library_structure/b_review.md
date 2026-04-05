# Agent B Review: Chapter 3 — LLK Library Structure — Pass 1

1. **`directory_layout.md` lines 120-121: `ckernel_gpr_map.h` and `ckernel_sfpu.h` are incorrectly listed as Blackhole/Wormhole B0-only headers "not present in Quasar."** Both files exist in `tt_llk_quasar/common/inc/`. The Quasar directory has 19 headers, and these two are among them. This would cause someone to incorrectly conclude that Quasar lacks GPR mapping and top-level SFPU includes. The `arch_differences.md` section (lines 172-179) correctly omits these two from the BH/WH-only list, so the error is isolated to `directory_layout.md`.

2. **`directory_layout.md` line 20 and `arch_differences.md` line 7: Blackhole and Wormhole B0 `llk_lib/` are described as having "28 files" but the directory also contains an `experimental/` subdirectory with 13 additional files.** The `experimental/` directory is not mentioned anywhere in the chapter. Since the plan's Chapter 8 (`adding_new_ops.md`) explicitly references `llk_lib/experimental/` as a location for in-progress features, omitting it from the directory layout could cause confusion when a reader encounters those files. The "28 files" count is technically correct if counting only `.h` files at the top level, but the tree diagram should note the `experimental/` subdirectory exists.

3. **`directory_layout.md` code snippet at lines 63-70: The `LLK_ASSERT` macro snippet is missing inner braces around `asm volatile("ebreak");`.** The chapter shows `if (UNLIKELY(!(condition))) asm volatile("ebreak");` but the actual code at `common/llk_assert.h` lines 16-19 wraps the asm statement in braces: `if (UNLIKELY(!(condition))) { asm volatile("ebreak"); }`. While this does not change runtime behavior, presenting incorrect source code undermines trust in the guide's accuracy and could confuse someone comparing the guide against the actual file.

4. **`arch_differences.md` line 216: The "Next" navigation footer links to `../ch4_llk_api_wrappers/index.md`, but the plan defines Chapter 4 as `ch4_kernel_compilation_pipeline`.** If the chapter directory is created following the plan's naming, this link will be broken. The link target should match whatever Chapter 4 is actually named.

5. **`arch_differences.md` summary table line 208: "Eltwise SFPU (unary): In `llk_lib/` only, no wrapper" for Quasar is misleading.** Quasar's `llk_lib/` does not have `llk_math_eltwise_unary_sfpu.h` (the file that exists in Blackhole/Wormhole B0). It has `llk_math_eltwise_unary_sfpu_common.h`, which is a different file with different scope (common SFPU logic, not the full unary SFPU implementation). Saying the operation is "In `llk_lib/` only" implies the same operation exists but just lacks a wrapper, when in reality Quasar has a structurally different and more limited SFPU file. The naming divergence table on lines 49 correctly notes this file exists, but the summary table's characterization could lead someone to attempt wrapping a function that does not have the same interface.

# Agent B Review: Chapter 3 — LLK Library Structure — Pass 2

No feedback — chapter approved.

All five Pass 1 issues have been verified as correctly addressed:

- Issue 1 (BH/WH-only headers): `ckernel_gpr_map.h` and `ckernel_sfpu.h` are no longer listed as BH/WH-only in the current text.
- Issue 2 (experimental/ subdirectory): Now shown in the directory tree with correct file counts (13 for Blackhole, 4 for Wormhole B0).
- Issue 3 (LLK_ASSERT braces): The code snippet now includes the inner braces, matching the actual source.
- Issue 4 (Chapter 4 link): Cannot be fully verified since Chapter 4 does not yet exist, but the issue was acknowledged.
- Issue 5 (Quasar SFPU summary): The summary table entry now provides a more accurate characterization of Quasar's structurally different SFPU file.

All numerical claims (file counts, line numbers, code content) were verified against the actual source trees at `/localdev/salnahari/testing_dir/tt-llk` and `/localdev/salnahari/testing_dir/tt-metal`. No new factual errors, incorrect numerical values, or materially misleading statements were found.
