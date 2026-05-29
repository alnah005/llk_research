# Compression Analysis -- Chapter 7: Metrics, Reporting, and Benchmark Orchestration

**Files reviewed:** index.md, 01_timing_metrics.md, 02_bulk_transfer_metrics.md, 03_output_format.md, 04_run_hybrid_orchestration.md
**Total lines:** 1,024

---

## Verdict

No CRUCIAL bloat found. The chapter is well-structured with each file covering a distinct topic. The duplications that exist are cross-reference restatements and repeated code snippets, none of which individually constitute structural bloat that would mislead or confuse a reader.

---

## Load-Bearing Evidence (No CRUCIAL Issues)

The chapter's four content files divide cleanly by concern: timing metrics (01), bulk transfer metrics (02), output formatting (03), and orchestration script (04). While there are cross-file restatements (detailed below as MINOR findings), none rise to the level of entire restated sections, duplicated tables with identical data, or paragraphs that repeat prior content verbatim. The closest candidate -- the `mode_label` code snippet appearing in both 02 and 03 -- is a single line repeated in service of two different explanatory contexts (bulk metric labeling vs. output format header).

---

## MINOR Findings

### MINOR-1: `mode_label` code snippet duplicated verbatim across files

**Files:** 02_bulk_transfer_metrics.md (line 183), 03_output_format.md (line 119)

The identical code snippet and surrounding explanation appear in both files:

```cpp
const char* mode_label = bulk_only ? "Hybrid" : "Socket";
```

In 02, it appears in section 7.2.6 to explain the output label column of the socket-vs-hybrid comparison table. In 03, it appears in section 7.3.2 to explain the mode label logic for the output header. Both files include the same line reference (pi05_pipeline_runner.cpp:1176). A cross-reference from one to the other would eliminate the repetition.

**Estimated savings:** ~8 lines (code block + surrounding prose in whichever file defers).

---

### MINOR-2: TTFT "includes queue wait time" stated twice in the same section

**File:** 01_timing_metrics.md, section 7.1.1

The fact that TTFT includes DecodeScheduler queue wait time is stated twice within the same section:

1. Line 46 (under "What TTFT includes"): "Queue wait time inside `DecodeScheduler`"
2. Line 52 (under "`turn_start` baseline"): "Set at `Clock::now()` immediately after the `SUBMIT` request is pushed, meaning TTFT includes any queue wait time inside the `DecodeScheduler`."

The second bullet restates the same fact with slightly different framing. Merging these into a single explanation under "What TTFT includes" would be cleaner.

**Estimated savings:** ~2 lines.

---

### MINOR-3: "Aggregation: Mean, median, P99 via `compute_stats()`" repeated identically three times

**File:** 01_timing_metrics.md, lines 54, 79, 106

The sentence "**Aggregation:** Mean, median, P99 via `compute_stats(X_samples)`." appears at the end of sections 7.1.1 (TTFT), 7.1.2 (ITL), and 7.1.3 (TPOT) with only the vector name changing. Since section 7.1.7 provides the full `compute_stats()` definition and the opening paragraph (line 11) already states "Every metric is accumulated into `std::vector<double>` sample vectors ... then summarized via `compute_stats()` after the benchmark completes," the per-section repetitions are redundant.

**Estimated savings:** ~3 lines (remove from individual sections; the intro + 7.1.7 cover it).

---

### MINOR-4: Stale process / IOMMU cleanup rationale restated multiple times in 04

**File:** 04_run_hybrid_orchestration.md

The rationale for killing stale processes and the role of the 2-second sleep is explained in three overlapping locations within the same file:

1. **Section 7.4.2** (lines 62-85): Full prose explanation of pkill, sleep 2, and /dev/shm cleanup with "Why" blocks.
2. **Section 7.4.3** (lines 131-136): The end-to-end diagram repeats "Kill stale processes," "Sleep 2s (IOMMU cleanup)," "Remove stale /dev/shm descriptors" as diagram annotations.
3. **Section 7.4.8** (lines 309-315): The summary table restates the same information a third time as a table row for each mechanism.

The diagram annotations (location 2) are acceptable as a visual summary, but the summary table (location 3) largely restates what is already fully explained in 7.4.2 without adding new information. Either the summary table or the detailed prose could be trimmed.

**Estimated savings:** ~12 lines (remove the summary table, since the walkthrough already covers each mechanism in detail).

---

### MINOR-5: D2H payload size stated three times in 02

**File:** 02_bulk_transfer_metrics.md

The D2H payload size of 3,208 bytes is presented three times:

1. Line 73: "one page containing a `BulkD2HHeader` (8 bytes) followed by `ACTION_OUTPUT_BYTES` (3,200 bytes)"
2. Lines 104-109: Dedicated table breaking down components to 3,208 bytes total.
3. Line 145: "With the constant D2H payload of 3,208 bytes:" in the bandwidth computation section.

The table (location 2) is the canonical definition. Locations 1 and 3 restate it. Location 1 is reasonable as an introduction; location 3 could simply reference "the D2H payload" without re-specifying the size.

**Estimated savings:** ~2 lines.

---

## Summary

| ID | Severity | File(s) | Description | Est. Savings |
|----|----------|---------|-------------|-------------|
| MINOR-1 | MINOR | 02, 03 | `mode_label` code snippet duplicated across files | ~8 lines |
| MINOR-2 | MINOR | 01 | TTFT queue-wait-time fact stated twice in same section | ~2 lines |
| MINOR-3 | MINOR | 01 | "Aggregation via compute_stats()" boilerplate repeated 3x | ~3 lines |
| MINOR-4 | MINOR | 04 | Stale-process/IOMMU cleanup rationale triple-stated | ~12 lines |
| MINOR-5 | MINOR | 02 | D2H payload size (3,208B) stated 3 times | ~2 lines |
| -- | -- | -- | **Total estimated savings** | **~27 lines (~2.6%)** |

No action required. All findings are minor restatements within a well-organized chapter.
