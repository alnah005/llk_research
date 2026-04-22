# Chapter 5 -- Agent B Review (Pass 1)

## Issue 1 — Quasar `common/inc/` top-level file count is wrong

**File:** `duplication_analysis.md`, line 32 (table row for quasar)
**Claim:** quasar has 16 top-level files in `common/inc/`
**Actual:** 19 top-level files (verified with `find ... -maxdepth 1 -type f`)

The total of 34 files across all subdirectories (top-level + sfpu + internal) is correct, but the component breakdown is not: 19 + 14 + 1 = 34, not 16 + 13 + 1 = 30. A reader using the sub-counts to reason about quasar's coverage relative to wormhole/blackhole (18 top-level files) would incorrectly conclude quasar has 2 fewer top-level headers when it actually has 1 more.

**Severity:** Numerically wrong. Would mislead analysis of per-subdirectory coverage.

## Issue 2 — Quasar `common/inc/sfpu/` file count is wrong

**File:** `duplication_analysis.md`, line 32 (same table row); also `maintenance_burden.md`, line 93
**Claim:** quasar has 13 sfpu files / "13 SFPU operation headers"
**Actual:** 14 sfpu files (verified with `find ... -type f` under `sfpu/`)

The 14th file is `ckernel_sfpu_typecast_int32_fp32.h`. The line count of 658 for quasar sfpu is correct.

**Severity:** Minor numerical error but directly quoted as a comparison point ("13 sfpu files vs. 52").

---

**Items checked and confirmed correct (no issue):**
- All `llk_lib/` file counts and line counts (wormhole 28/7646, blackhole 28/6615, quasar 23/4210)
- All `common/inc/` line counts (top-level and sfpu) for all three architectures
- All `assembly.yaml` line counts
- Grand totals per architecture (27542, 28975, 17528, combined 74045)
- Experimental file/line counts
- Repository-root `common/` has exactly 2 files
- Quasar-only file list (10 files) and wormhole-only file list (15 files) -- all names verified
- Metal `llk_api/` line counts (12310, 12284, 593) and top-level file counts (21, 21, 7)
- HAL per-arch file counts (7 each) and line counts (921, 1012, 1161)
- `llk_math_matmul.h` line counts (874 wormhole, 356 quasar)
- Wormhole/blackhole file list identity (28 files, same names)
