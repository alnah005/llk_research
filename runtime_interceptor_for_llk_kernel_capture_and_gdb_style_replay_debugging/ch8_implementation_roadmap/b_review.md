# Chapter 8 B-Level Quality Review

**Reviewer:** B-Level Structural Quality Reviewer
**Files Reviewed:**
- `ch8_implementation_roadmap/01_gap_analysis.md` (653 lines)
- `ch8_implementation_roadmap/02_phased_implementation_plan.md` (465 lines)
- `ch8_implementation_roadmap/03_open_questions_and_validation_experiments.md` (594 lines)
- `ch8_implementation_roadmap/index.md` (68 lines)

**Comparison chapters:** Ch5 index, Ch6 index, Ch7 index

---

## Checklist

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | PROPOSED prefix on non-existing designs | PASS (partial) | `01_gap_analysis.md` uses PROPOSED consistently (22 occurrences). `02_phased_implementation_plan.md` and `03_open_questions_and_validation_experiments.md` have **zero** PROPOSED prefixes. The phased plan describes proposed tasks/deliverables (e.g., `tt-kernel-capture` CLI, `.ttksnap` schema, GDB RSP server) without the PROPOSED prefix. |
| 2 | Cross-references cite specific chapter/file/section numbers | PASS | All three files include header cross-reference blocks citing Ch 1-7 with file and section granularity. Gap analysis references like "Ch 6, File 1, Section 4.3" and "Ch 6, File 3, Section 3" are specific. |
| 3 | Commit 621b949 referenced | PASS | Each of the three main files references `621b949` exactly once (in the header). `index.md` does not reference the commit but indexes rarely do. |
| 4 | Key Takeaways at end of each file | PASS | All three files have a "## Key Takeaways" section at the end (lines 623, 441, 564 respectively). |
| 5 | Line counts within 400-700 range | PASS | `01_gap_analysis.md`: 653 (in range). `02_phased_implementation_plan.md`: 465 (in range). `03_open_questions_and_validation_experiments.md`: 594 (in range). |
| 6 | Effort estimates match spec | PASS (with caveat) | Overview table shows P0: 2-3, P1: 4-6, P2: 4-6, P3: 6-8, P4: 4-6, total 20-29 pw. All match spec. However, Phase 4 task-level effort sums to 6.5-8 pw (see inconsistency I-3 below). |
| 7 | Consistency with Ch5-Ch7 | PASS (with caveats) | Architecture claims (WH lacks RISC_DBG_CNTL, BH has raw registers, Quasar has rvdbg_cmd) are consistent. Schema references (.ttksnap, TKSN identifier, 93% coverage) are consistent with Ch6. See inconsistencies I-4, I-5 for minor discrepancies. |
| 8 | Source file paths cited and consistent across files | FAIL (minor) | Three path inconsistencies identified (see I-4, I-5, I-6 below). |
| 9 | Gap-to-phase traceability | PASS | Every gap in the priority summary table (Section 6 of `01_gap_analysis.md`) includes a "Phase" column. Every phase in the implementation plan (Section 1 table of `02_phased_implementation_plan.md`) includes a "Primary Gaps Addressed" column. Bidirectional traceability is complete. |
| 10 | Experiment-to-question mapping | PASS | Section 6 of `03_open_questions_and_validation_experiments.md` provides a complete question-to-experiment mapping table. All 13 OQs are mapped. All 7 VEs are linked to at least one OQ. |

---

## Inconsistencies

### I-1: Index.md Claims "12 open questions" and "12 validation experiments" -- Actual Counts Differ (MEDIUM)

**Location:** `index.md`, line 17

**Issue:** The index states "12 open questions in 3 tiers, 12 validation experiments." The actual file contains **13 open questions** (OQ-1 through OQ-13) and **7 validation experiments** (VE-1 through VE-7).

**Fix:** Change line 17 to: `13 open questions in 3 tiers, 7 validation experiments with procedures, decision trees, experiment priority matrix, scheduling aligned to phases`

### I-2: Index.md Key Finding 1 Misidentifies a CRITICAL Gap (MEDIUM)

**Location:** `index.md`, line 25

**Issue:** Key Finding 1 lists 4 CRITICAL gaps as:
- (a) No L1 snapshot capability
- (b) No GDB stub for Tensix RISC-V
- **(c) No kernel invocation capture**
- (d) Quasar debug register host accessibility unvalidated

The actual 4 CRITICAL gaps in `01_gap_analysis.md` are:
- GAP-SC-1: No L1 Memory Snapshot
- **GAP-SC-2: Compiled ELF Binaries Not Captured by LightMetal**
- GAP-DI-1: No GDB Stub for Tensix RISC-V Cores
- GAP-DI-2: Debug Register Host Accessibility Unvalidated

Item (c) should be "No ELF binary capture" (GAP-SC-2), not "No kernel invocation capture."

**Fix:** Change "(c) No kernel invocation capture" to "(c) Compiled ELF binaries not captured (GAP-SC-2)"

### I-3: Phase 4 Task-Level Effort Sum Exceeds Stated Phase Total (MEDIUM)

**Location:** `02_phased_implementation_plan.md`, lines 257-265

**Issue:** Phase 4 individual task efforts sum to 6.5-8 pw:
- P4-T1: 1 pw
- P4-T2: 2 pw
- P4-T3: 2-3 pw
- P4-T4: 1 pw
- P4-T5: 0.5-1 pw

Total: 6.5-8 pw. But the phase total is stated as "4-6 pw" (line 265).

The text on line 265 notes "within spec range when deferring features to keep scope bounded" and lines 268-270 describe VE-1-dependent scope adjustments (e.g., VE-1 negative cuts P4-T3 to ~1.5 pw, yielding 3.5-4 pw). However, the task table as written does not reflect these deferrals. The reader encounters 6.5-8 pw of tasks followed by a 4-6 pw total, which is confusing.

**Fix:** Either (a) add a "Deferred" column to the task table indicating which tasks are conditionally excluded, or (b) split tasks into "Core" (must-do) and "Conditional" (VE-1 dependent) groups with separate subtotals that demonstrate how 4-6 pw is achieved, or (c) add explicit parenthetical notes on deferral-eligible tasks.

### I-4: `trisc.cpp` Path Inconsistency Across Files (LOW)

**Location:** Multiple files

**Issue:** `trisc.cpp` is referenced with two different paths:
- `tt-llk/tests/helpers/src/trisc.cpp` in `01_gap_analysis.md` (line 331), `02_phased_implementation_plan.md` (line 204), `03_open_questions_and_validation_experiments.md` (lines 103, 321)
- `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp` in `index.md` (line 51) and Ch7 `index.md` (line 50)

These likely resolve to the same file via a symlink/submodule, but the inconsistency is jarring. Ch5's index uses `tt-llk/tests/helpers/src/trisc.cpp` (the shorter form).

**Fix:** Standardize on one path form across all files. Since Ch5 uses the shorter form and it appears more frequently, prefer `tt-llk/tests/helpers/src/trisc.cpp` and update the Ch8 `index.md` accordingly.

### I-5: `noc_debugging.hpp` Path Inconsistency (LOW)

**Location:** `ch8_implementation_roadmap/index.md` line 52 vs Ch5 index

**Issue:** Two different paths for `noc_debugging.hpp`:
- Ch5 index: `tt_metal/hw/inc/hostdev/noc_debugging.hpp`
- Ch7 index and Ch8 index: `tt_metal/impl/debug/noc_debugging.hpp`

Ch8's `01_gap_analysis.md` references `noc_debugging.hpp` without a full path (lines 168, 177). The correct path should be verified and standardized.

**Fix:** Verify which path is correct and update all references.

### I-6: BH `tensix.h` Line Numbers: 162-163 vs 162-165 (LOW)

**Location:** Multiple files

**Issue:** BH RISC_DBG_CNTL register line references differ:
- `01_gap_analysis.md` and `03_open_questions_and_validation_experiments.md`: `tensix.h:162-163`
- `index.md` and Ch7 `index.md`: `tensix.h:162-165`

**Fix:** Verify the actual line range and standardize.

### I-7: Index.md Key Finding 4 Uses "E2, E3, E6" Naming Instead of "VE-2, VE-3, VE-6" (LOW)

**Location:** `index.md`, line 37

**Issue:** The index refers to experiments as "E2, E3, E6" but the actual file uses "VE-1" through "VE-7" naming convention.

**Fix:** Change "E2, E3, E6" to "VE-2, VE-3, VE-6" for consistency with the file content.

### I-8: Index Key Finding 4 Claim vs File Content (LOW)

**Location:** `index.md`, line 37

**Issue:** Claims "Three experiments (E2, E3, E6) should be completed before Phase 1." Looking at the experiment schedule in `03_open_questions_and_validation_experiments.md` Section 7:
- VE-6 (capture overhead) is scheduled during Phase 0 -- this aligns with "before Phase 1."
- VE-2 (L1 read consistency) is scheduled during Phase 2 -- NOT before Phase 1.
- VE-3 (host model fidelity) is scheduled during Phase 3 -- NOT before Phase 1.

The claim appears incorrect. Only VE-6 is scheduled before Phase 1.

**Fix:** Correct the index Key Finding 4 to accurately reflect the experiment schedule, or if the intent is different from the schedule, clarify.

### I-9: `02_phased_implementation_plan.md` Missing PROPOSED Prefix (LOW)

**Location:** Throughout `02_phased_implementation_plan.md`

**Issue:** The phased plan describes numerous non-existing designs (e.g., `kernel_snapshot.fbs` FlatBuffer schema, `tt-kernel-capture` CLI, GDB RSP Server, `CAPTURE_BREAKPOINT()` macro, `CaptureFilter`, `libttreplay`, VS Code DAP adapter) without using the PROPOSED prefix. The convention established in Ch5 and Ch6 index files requires PROPOSED for designs not yet implemented. `03_open_questions_and_validation_experiments.md` similarly lacks PROPOSED prefixes.

**Fix:** While the implementation plan is inherently about proposed work (the entire chapter is a roadmap), consider adding a blanket disclaimer at the top of each file (similar to Ch6 index notation section) stating that all designs described are PROPOSED, or add PROPOSED prefix to the first mention of each non-existing component.

---

## Recommended Fixes (Priority Order)

1. **I-1 (MEDIUM):** Fix open question and experiment counts in `index.md` (13 OQs, 7 VEs).
2. **I-2 (MEDIUM):** Fix CRITICAL gap (c) label in `index.md` to match GAP-SC-2.
3. **I-3 (MEDIUM):** Clarify Phase 4 effort discrepancy with core/conditional task split or explicit deferral notes.
4. **I-8 (LOW):** Fix or clarify experiment schedule claim in index Key Finding 4.
5. **I-7 (LOW):** Standardize experiment naming to VE-N in index.
6. **I-4 (LOW):** Standardize `trisc.cpp` path across all Ch8 files.
7. **I-5 (LOW):** Verify and standardize `noc_debugging.hpp` path.
8. **I-6 (LOW):** Verify and standardize BH `tensix.h` line range.
9. **I-9 (LOW):** Add PROPOSED notation to `02_phased_implementation_plan.md` and `03_open_questions_and_validation_experiments.md`.

---

## Strengths

- **Gap-to-phase traceability is excellent.** Every gap has a phase assignment, every phase lists addressed gaps, and the priority summary table (Section 6) provides a single-view mapping. The critical path analysis (Section 10 of gap analysis, Section 9 of implementation plan) is well-articulated.

- **Developer pain scoring adds unique value.** The composite pain score (P1 x 0.35 + P2 x 0.35 + P3 x 0.30) and three workflow scenario analyses (Sections 10-11 of gap analysis) provide concrete justification for prioritization decisions.

- **Risk registers are thorough and actionable.** Each phase has specific risks with probability/impact/severity and concrete mitigations. The CRITICAL risks (R2-1, R2-2 on L1 readback, R4-1 on debug registers) correctly align with the validation experiments.

- **Decision trees (Section 8 of open questions) are a standout feature.** They make explicit how experiment outcomes change the plan, with effort adjustments for each branch. No other chapter provides this level of adaptive planning.

- **Architecture-specific gap analysis (Section 8 of gap analysis) is comprehensive.** The WH/BH/Quasar comparison table correctly captures the different debug register capabilities and their implications.

- **Cross-chapter references are specific and verifiable.** References like "Ch 6, File 1, Section 4.3" and "Ch 5 Key Finding 3" enable direct navigation.

- **Effort budget reconciliation (Section 12 of implementation plan) explicitly maps prior chapter estimates to plan phases**, demonstrating internal consistency.

---

## Overall Assessment

**Grade: B+ (Good with minor fixes needed)**

The three Ch8 files are structurally sound, well-organized, and internally consistent on major claims (22 gaps, 4 CRITICAL, 20-29 pw total, three-tier debug strategy, architecture-specific paths). Gap-to-phase traceability and experiment-to-question mapping -- the two most important structural properties for an implementation roadmap -- are both complete and bidirectional.

The primary issues are:
1. Three factual errors in the `index.md` (wrong OQ/VE counts, wrong CRITICAL gap label, wrong experiment naming and scheduling claims) -- these are straightforward to fix.
2. Phase 4 task effort sum (6.5-8 pw) does not reconcile with the stated phase total (4-6 pw) without reading the scope adjustment note. This needs clearer presentation.
3. Minor path and line-number inconsistencies across files.

None of the issues affect the substantive content or recommendations of the chapter. All are fixable with targeted edits.
