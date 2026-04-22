# Chapter 5 Compression Analysis

**Agent:** C (Compressor) -- Pass 1
**VERDICT:** Crucial updates: no

---

## Load-Bearing Evidence

- **index.md**: The "Key Statistics" table (lines 11-21) and all four "Key Findings" (lines 27-43) restate data and conclusions that are elaborated in the sub-files. The table duplicates every major number from `duplication_analysis.md`, and each finding is a condensed version of a full section in one of the sub-files. The phrase "exactly 2 files (`llk_assert.h` and `tensor_shape.h`)" appears on line 23 and is repeated verbatim in both sub-files.
- **duplication_analysis.md**: The `instructions/assembly.yaml` table (lines 39-43) is restated in `maintenance_burden.md` (lines 107-111) and summarized in `index.md` (line 21). The full 28-file listing (lines 68-84) occupies 17 lines to make the single point that the file lists are identical; the preceding sentence already states this.
- **maintenance_burden.md**: The "Quasar's Structural Divergence Prevents Mechanical Porting" section (lines 66-101, 36 lines) recaps quasar's different naming conventions, file decomposition, hardware abstractions, and SFPU coverage -- all of which are already covered in `duplication_analysis.md` lines 89-128. The "Instructions Directory" section (lines 103-121, 19 lines) restates the instructions table and commentary from `duplication_analysis.md` lines 38-46.

---

## Cross-File Redundancy

| Duplicated content | Locations | Estimated removable lines |
|---|---|---|
| "exactly 2 files (llk_assert.h, tensor_shape.h)" fact | index.md:23, duplication_analysis.md:57-59, maintenance_burden.md:6-9 | ~6 (keep one, reference from others) |
| instructions/assembly.yaml table + commentary | index.md:21, duplication_analysis.md:38-46, maintenance_burden.md:103-121 | ~25 (keep in duplication_analysis.md, cut from maintenance_burden.md) |
| Quasar structural divergence explanation (naming, broadcast files, SFPU coverage) | duplication_analysis.md:89-128, maintenance_burden.md:66-101 | ~30 (keep detailed version in duplication_analysis.md, reduce maintenance_burden.md to a forward reference) |
| Key Statistics table vs. sub-file tables | index.md:11-21 vs. duplication_analysis.md:14-55 | ~10 (index.md table can be trimmed to 3-4 headline numbers with a "see duplication_analysis.md" pointer) |
| Metal llk_api wrapper line counts | index.md:38, duplication_analysis.md:137-139 | ~2 |

---

## Within-File Bloat

| Item | File | Lines | Issue |
|---|---|---|---|
| Full 28-file listing | duplication_analysis.md:68-84 | 17 | The preceding sentence already states the lists are identical. A collapsed listing or "28 identical filenames" suffices; the raw listing adds no analytical value. |
| Metal llk_api file listing (19 filenames) | duplication_analysis.md:148-159 | 12 | Same pattern: the text already says "21 files, same API surface." The listing is reference material, not analysis. |
| Quasar divergence section in maintenance_burden.md | maintenance_burden.md:66-101 | 36 | Near-complete restatement of duplication_analysis.md:89-128. Should be replaced with a one-sentence reference and focus only on the maintenance *consequence* (cannot automate porting). |
| Instructions section in maintenance_burden.md | maintenance_burden.md:103-121 | 19 | Restates table and analysis from duplication_analysis.md. The only new content is "architecturally justified" (one sentence). |

---

## MINOR Suggestions

1. **Consolidate the "2 shared files" fact.** State it once in `index.md` and reference it from sub-files. Current state: three independent statements of the same fact with nearly identical wording. (~6 lines removable)

2. **Replace the full 28-file listing in duplication_analysis.md with a summary line.** E.g., "All 28 non-experimental filenames are identical across wormhole_b0 and blackhole (see repository for full listing)." If the listing must stay, put it in a collapsed `<details>` block. (~15 lines recoverable)

3. **Merge or cross-reference the quasar divergence discussion.** `maintenance_burden.md` lines 66-101 should reference `duplication_analysis.md` for the structural facts and add only the maintenance-specific conclusion ("this cannot be automated with simple file-level tooling"). (~30 lines removable)

4. **Remove the instructions/assembly.yaml table from maintenance_burden.md.** It is already in `duplication_analysis.md` and `index.md`. The one new sentence ("architecturally justified") can be a single line with a cross-reference. (~17 lines removable)

5. **Trim the index.md statistics table.** Replace with 2-3 headline numbers (total LLK lines: 74,045; shared files: 2; Metal wrapper lines: ~25,187) and a pointer to `duplication_analysis.md` for the full breakdown. (~6 lines removable)

---

## Summary

| Metric | Value |
|---|---|
| Total lines across 3 files | ~367 |
| Estimated removable lines (redundancy) | ~75-80 |
| Estimated removable lines (bloat/verbose listings) | ~30-35 |
| **Total compressible** | **~105-115 lines (~30%)** |
| Factual issues found | 0 (out of scope) |
| Crucial updates needed | No |
