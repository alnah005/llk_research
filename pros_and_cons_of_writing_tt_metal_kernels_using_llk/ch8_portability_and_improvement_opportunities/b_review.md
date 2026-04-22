# Agent B Review: Chapter 8 — Portability and Improvement Opportunities — Pass 1

1. **Wrong count for `sfpu_split_includes.h` conditional blocks (improvement_opportunities.md, proposal 5).** The chapter states "47 conditional include blocks" in `sfpu_split_includes.h`. Grepping for `#if SFPU_OP_` in `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` yields **46**, not 47.

2. **Fabricated enum type `DstTileFaceLayout` in the define schema example (improvement_opportunities.md, proposal 3).** The YAML example lists `DST_ACCUM_MODE` with values `[DstTileFaceLayout::RowMajor, DstTileFaceLayout::ColMajor]`. The type `DstTileFaceLayout` does not exist anywhere in tt-metal or tt-llk. In practice, `DST_ACCUM_MODE` is a boolean-like value (e.g., `#define DST_ACCUM_MODE is_fp32_dest_acc_en`). Illustrative examples should not invent types that contradict the actual codebase — readers may take the schema as authoritative.

3. **Misleading "exception-safety" claim for RAII proposal (improvement_opportunities.md, proposal 1).** The text states the RAII guards "provide exception-safety guarantees even in the presence of early returns." Kernel firmware is compiled with `-fno-exceptions` (`tt_metal/hw/CMakeLists.txt:308`). The early-return benefit is real, but claiming "exception-safety guarantees" for code that cannot throw exceptions is misleading and may confuse readers into thinking exceptions are part of the kernel execution model.

4. **`#ifdef ARCH_QUASAR` / `#else` column in the quantitative table attributes 3 to `cb_api.h` alone (architecture_divergence.md).** Actual `#ifdef ARCH_QUASAR` occurrences in `cb_api.h` number **1** (line 8). The remaining 2 occurrences are in `common.h`, which is not listed in the table at all. The table total of 3 is numerically correct but the per-file attribution is wrong, and the omission of `common.h` from the table means the table does not actually cover all files with `ARCH_QUASAR` guards in the compute API directory.

5. **No issues found with the remaining factual claims.** The 45 `#ifndef ARCH_QUASAR` guard count (verified: 45 across 9 files), the 31 TODO stub count (verified: 31 across 7 files), Quasar's `TRISC_COUNT = 4`, 4 MB L1 (`L1_SIZE_IN_BYTES = 0x00400000`), 8 DM cores, and the `matmul_block_math_dynamic_throttle` Blackhole-only guard all check out against source.

# Agent B Review: Chapter 8 — Pass 2

No feedback — chapter approved.

All four fixes from Pass 1 have been verified against source:

1. **SFPU count 47->46**: Corrected. `improvement_opportunities.md` proposal 5 now says "46 conditional include blocks", matching the 46 `#if SFPU_OP_` directives in `sfpu_split_includes.h`.
2. **DstTileFaceLayout removed**: Corrected. The YAML example in proposal 3 now shows `DST_ACCUM_MODE` with `type: bool` and `values: [0, 1]`, which accurately reflects the codebase.
3. **exception-safety -> scope-safety**: Corrected. Proposal 1 now says "provide automatic cleanup on scope exit even in the presence of early returns" with no mention of exceptions.
4. **cb_api.h count corrected**: Corrected. The table in `architecture_divergence.md` now shows `cb_api.h` with 1 `#ifdef ARCH_QUASAR` branch and `common.h` with 2, both matching source. The total of 3 remains correct.
