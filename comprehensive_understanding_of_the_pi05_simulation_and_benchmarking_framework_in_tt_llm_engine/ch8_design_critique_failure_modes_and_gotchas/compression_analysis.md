# Compression Analysis -- Chapter 8: Design Critique, Failure Modes, and Gotchas

**Total lines (5 files):** 1153

---

## CRUCIAL -- Systematic Cross-Section Duplication

The chapter is organized as four sections: design critique (8.1), failure modes (8.2), production kernel differences (8.3), and operational gotchas (8.4). Sections 8.2 and 8.4 cover **seven overlapping topics** where 8.4 restates the failure mode from 8.2 with near-identical content -- same root causes, same code snippets, same `run_hybrid.sh` excerpts. The duplication is structural: the chapter chose a "catalog by perspective" layout (theory vs. practice) but the perspectives do not carry enough distinct information to justify separate treatments.

### Duplicate pair 1: `/dev/shm` stale descriptors

- **8.2.2** (lines 39-64, 26 lines): trigger, symptom, root cause, `run_hybrid.sh` cleanup script, `_exit(EXIT_SUCCESS)` explanation
- **8.4.1** (lines 9-33, 25 lines): identical symptom, identical root cause, **verbatim same** `run_hybrid.sh` code block, same `_exit(EXIT_SUCCESS)` explanation

The `run_hybrid.sh` cleanup snippet appears identically in both:
```bash
rm -f /dev/shm/tt_d2h_* /dev/shm/tt_h2d_* /dev/shm/tt_socket_manifest_* /dev/shm/pi05_*
```

### Duplicate pair 2: `TT_VISIBLE_DEVICES` crash

- **8.2.5** (lines 131-180, 50 lines): trigger, symptom, code, `run_hybrid.sh` default of `1`, parent/child asymmetry
- **8.4.2** (lines 36-69, 34 lines): same symptom, same root cause, **same** `run_hybrid.sh` code block, same parent-defaults-to-1/child-defaults-to-0 asymmetry explanation

### Duplicate pair 3: PCIe alignment violations

- **8.2.4** (lines 96-128, 33 lines): `BULK_PAGE_SIZE` mismatch, host and kernel code, three failure scenarios
- **8.4.4** (lines 100-139, 40 lines): same alignment requirement, **same** host-side and kernel-side code snippets, same divergence warning, adds the three-file sync list

### Duplicate pair 4: Zombie processes / process cleanup

- **8.2.1** (lines 7-35, 29 lines): `PR_SET_PDEATHSIG`, `run_hybrid.sh` pkill + sleep
- **8.4.5** (lines 142-168, 27 lines): **verbatim same** `run_hybrid.sh` code block, same SIGKILL rationale, same sleep-2-seconds explanation

The `run_hybrid.sh` process cleanup snippet appears identically in both:
```bash
pkill -9 -f pi05_device_launcher 2>/dev/null || true
pkill -9 -f pi05_pipeline_runner_device 2>/dev/null || true
sleep 2
```

### Duplicate pair 5: FIFO sizing / deadlock

- **8.2.7** (lines 225-249, 25 lines): payload-exceeds-FIFO, deadlock scenario, sizing math
- **8.4.7** (lines 210-230, 21 lines): same deadlock scenario, same sizing formula `BULK_PAGE_SIZE * num_users`, same L1 memory budget concern

### Duplicate pair 6: D2H slot_id mismatch + action_history corruption

- **8.1.6** (lines 219-252, 34 lines): mismatch logged as WARNING, cascading corruption chain ("user A receives user B's action output"), no recovery mechanism
- **8.4.9** (lines 271-317, 47 lines): re-explains the same corruption chain with the same user-A/user-B framing, re-derives the same propagation sequence (turn k -> turn k+1), same conclusion about per-user buffers

### Duplicate pair 7: `output_tokens=1` and D2H ordering

- **8.1.6** (lines 240-243): explains that `output_tokens=1` prevents slot_id mismatches, and `output_tokens > 1` causes ordering divergence
- **8.4.6** (lines 171-207, 37 lines): re-explains `output_tokens=1` semantics, restates that `output_tokens > 1` makes D2H mismatch "more likely" -- same point as 8.1.6

### Estimated savings

The seven duplicate pairs total approximately 230 lines of restated content in 8.4. Consolidating each topic into a single treatment (keeping the richer version, adding a one-line cross-reference from the thinner section) would remove ~180-200 lines (~16-17% of the chapter).

**Recommendation:** Merge 8.2 and 8.4 into a single "Failure Modes and Operational Gotchas" section organized by topic rather than by perspective. Each topic gets one entry covering theory, symptoms, root cause, and operational fix. Alternatively, keep 8.4 as a quick-reference checklist with one-line symptom/fix pairs that cross-reference 8.2 for details, eliminating the restated root-cause explanations and duplicated code blocks.

---

## MINOR -- Verbose prose in Mitigation subsections (01_design_critique.md)

Each of the six design critiques in 8.1 follows the Observation/Concern/Mitigation template. The Mitigation subsections frequently re-summarize the Observation and Concern before stating the mitigation, adding ~2-4 sentences of recap per entry. Examples:

- 8.1.1 Mitigation (line 40): "The flattening is **intentional**. The `PipelineSimulator` backend does not support heterogeneous stage durations. The total-latency-preserving average ensures that **throughput** ... and **total pipeline latency** ... are correct." -- the first two sentences restate what the Observation already established.
- 8.1.4 Mitigation (line 168): "The single-channel design matches the hardware constraint: there is **one** H2D socket and **one** D2H socket between host and device for bulk data." -- restates the Observation's opening sentence nearly verbatim.
- 8.1.5 Mitigation (line 214): "`prctl(PR_SET_PDEATHSIG, SIGTERM)` (line 640) is the critical safety net" -- the Observation already quoted this exact line with the same explanation.

These recaps add ~30 lines across the six entries. Trimming each Mitigation to start directly with the justification or defense (skipping the restatement of what was already covered in Observation/Concern) would tighten the section without losing information.

---

## MINOR -- Repeated `run_hybrid.sh` code blocks across files

The `run_hybrid.sh` cleanup script appears **four times** across the chapter (twice for the `/dev/shm` cleanup line, twice for the pkill+sleep block). Each appearance is a full fenced code block with the same line-reference comment. A single "Reference: `run_hybrid.sh` cleanup sequence" block in one canonical location, with cross-references elsewhere, would eliminate three redundant code fences.

---

## Change Log

**2026-05-28 -- Fixes applied (Agent A):**

1. **B1 (factual error):** Fixed H2D payload in 01_design_critique.md Section 8.1.4 from "456,704 bytes" to "458,752 bytes" (112 * 4096 = 458,752). Updated downstream calculations (total MB, per-transfer time).
2. **B2 (misleading claim):** Added clarifying note to 01_design_critique.md Section 8.1.5 that `mgr->start()` is called after `DeviceLauncher::start()` returns, so no scheduler threads exist at fork time; the fork-after-multithreading concern is theoretical.
3. **B3 (_exit() cross-references):** Added cross-references between the three `_exit()` discussion locations: 8.2.2 now links to 8.2.6 and 8.4.1; 8.2.6 now links back to 8.2.2; 8.4.1 now links to both 8.2.2 and 8.2.6.
4. **C1 (systematic duplication):** Converted six overlapping entries in 04_operational_gotchas.md (8.4.1, 8.4.2, 8.4.4, 8.4.5, 8.4.7, 8.4.9) from full restated treatments into brief quick-reference items (symptom + fix + cross-reference to the canonical analysis in 8.1/8.2). Removed duplicated code blocks and detailed prose. Trimmed 8.4.6 D2H ordering duplication. Updated section intro to describe the new quick-reference format. Unique gotchas (8.4.3, 8.4.6, 8.4.8) retained full treatment.
