# Agent B Review — Pass 1

**Date:** 2026-11-06
**Reviewer:** Agent B (Critic) — cold pass, scope: factual correctness, critical coherence, critical structural gaps only.
**Method:** plan.md §Ch4 spec read as structural authority; all six chapter files read; every doubtful number re-verified against the read-only regen tree (notes/glm-decode-roofline.md, notes/glm-decode-ops.md, notes/glm-bfp8.md, notes/noc-ids.md, notes/glm-batch.md, notes/glm-glx-layouts.md, notes/glm-glx-mtp2.md, notes/glm-glx-mtp3.md, notes/post-rc/glm-rc1.md, src/brisc.cpp, src/fw.h via note cross-checks) and the orchestrator registry.

**Verdict: 3 items.** Structure, navigation footers, planned-file inventory (index + 01–05), ownership fences, the moves-table F/H semantics and hot3-meas blanks, the Phase-1 hotE/hotM column caveat, the NoC-ID note-vs-source flag, and the batch/MTP registry numbers all verified correct against the corpus.

---

**Item 1 — Arithmetic error: hot-expert delta at the end of the ledger**
- **File:** `04-optimization-ledger.md`, ~line 112 ("Where it landed" close-out).
- **Error:** "~+0.55 ms at the end ($6.05 - 5.59$)" — the chapter's own equation gives $6.05 - 5.59 = 0.46$ ms, not 0.55. A reader recomputing the narrowed delta gets a number that contradicts the text.
- **Fix:** Change "~+0.55 ms" to "~+0.46 ms" (keeping "$6.05 - 5.59$"). The narrowing story (+0.63 → +0.46) survives unchanged.

**Item 2 — Misattributed weight file: the +370 MB / 2134 MB per-chip budget belongs to `hotE-meas`, not `hot3-meas`**
- **File:** `01-roofline-and-byte-budget.md`, ~line 14 (prose after the per-chip byte table).
- **Error:** "The hottest weight file (`hot3-meas` …) adds +370 MB per token per chip on top of plan0h's 1763.8, i.e. 2134 MB/chip [MEAS, notes/glm-decode-roofline.md (lines 48–49)]." The cited line 49 attributes the +370 to **`hotE-meas`** (70% of routed reads at BFP8), and the table row directly above correctly says `hotE-meas`. `hotE-meas` and `hot3-meas` are distinct builds (the chapter index's own table-reading key lists them as separate weight files; `notes/glm-batch.md`'s format table gives hot3's routed bytes as 932 MB/chip vs plan0h's 595 → hot3-meas adds ~+337 MB, i.e. ~2.10 GB/chip per README "Roofline", not 2134 MB). As written, a porter sizing the production file's DRAM budget inherits hotE's numbers, and the guide contradicts its own F/H/weight-file column discipline.
- **Fix:** In that sentence, replace "`hot3-meas`" with "`hotE-meas`" (matching the table row and the citation). If the production file's own budget is wanted, add it as a separate tagged claim: hot3-meas ≈ 1763.8 + (932 − 595) ≈ 2101 MB ≈ 2.10 GB/chip [MEAS, notes/glm-batch.md format table; README.md "Roofline"], leaving the 4.27/4.77 ms floors explicitly `hotE`'s.

**Item 3 — "Gap to max(floors)" column is not recomputable from the columns shown (mHC sites row)**
- **File:** `01-roofline-and-byte-budget.md`, "Where the 9.3 ms goes" table, ~lines 72–84 (mHC sites row ~line 76).
- **Error:** The chapter's table omits the note's "Sync floor (rounds × 2 us)" column but keeps "Gap to max(floors)", and the parenthetical says only "the fetch, chip-barrier and mesh-sync columns" were dropped. For mHC sites the gap (1.61) is measured − **max(DRAM floor 0.030, sync floor 0.180)** = 1.785 − 0.180 = 1.61; a reader recomputing from the shown columns gets 1.785 − 0.030 = 1.76 and must conclude the 1.61 cell (and the ranked list "mHC sites 1.61" that the ledger and later files lean on) is wrong. All other rows happen to be DRAM-floor-dominated, so the omission is invisible except here — exactly where it matters, since mHC sites is ranked the largest gap.
- **Fix:** Either restore the sync-floor column (0.168 / 0.068 / 0.180 / 0.044 / 0 / 0.006 / 0.004, total 0.470, per notes/glm-decode-roofline.md line 90 header row) or amend the parenthetical to: "The table also omits the note's Sync-floor (rounds × 2 us) column (MoE 0.168, KDA 0.068, mHC sites 0.180, MLA 0.044, dense 0.006, head 0.004); 'Gap to max(floors)' is measured − max(DRAM floor, sync floor) — for mHC sites 1.785 − 0.180 = 1.61."

---

**Not flagged (checked and cleared, for the record):** moves-table F/H labels and hot3-meas blanks match roofline lines 299–305 exactly, including the em-dash in plan0h F on the last row; Phase-1 hotE chain ends 8980 with the hotM caveat verbatim from decode-ops lines 20–21; Phase-4 compositions (6149−171=5978, 6774−176=6598) verified; per-column moves compositions (5924/5939, 6607/6627) verified; NoC-ID table matches src/brisc.cpp:354–373 (three chunk IDs per buffer, 4–12, ID 12 prefill role) with the note-vs-source supersession correctly flagged; strip-byte formulas match brisc.cpp:1499–1502 and glm-bfp8.md; byte-table row sum 1763.8 verified; all floor divisions (3.53/3.95/4.27/4.77/4.10/4.59 ms and t/s) verified; batch union table, ~3.8×, 38.7→53.1 = +37%, acceptance 0.892/0.876/0.802–0.84, and the mtp2/mtp3 results cells verified against the notes; the mtp2 "tokens per user step" column is copied verbatim from the note's own table (it describes the MTP shape only — worth a clarifying word in a later pass, but it is the corpus's own presentation, not a chapter error).

---

## Change Log — Agent A (B pass-1 feedback)

**Date:** 2026-11-06. All three pass-1 items applied; each anchor re-verified in the read-only regen tree (`/home/alex/mpi-shfs/tenstorrent/regen-glm5.3-flash`) before editing. Only `01-roofline-and-byte-budget.md` and `04-optimization-ledger.md` changed; footers, H1s, and registry numbers untouched.

- **Item 1 (04-optimization-ledger.md, line 110):** Fixed — "~+0.55 ms at the end" → "~+0.46 ms at the end" (keeping $6.05 - 5.59$; 6.05 − 5.59 = 0.46). Narrowing story +0.63 → +0.46 unchanged.
- **Item 2 (01-roofline-and-byte-budget.md, line 32):** Fixed — re-attributed the +370 MB / 2134 MB/chip budget to `hotE-meas` (per note lines 48–49, verified: line 49 says "hotE-meas adds (70% of routed reads at BFP8) | +370"), and added `hot3-meas`'s own separately tagged budget: 1763.8 + (932 − 595) ≈ 2101 MB ≈ 2.10 GB/chip [MEAS, notes/glm-batch.md (lines 30–31); README.md "Roofline"], both anchors verified (glm-batch lines 30–31 give routed 595/932; README "Roofline" states 2.10 GB/chip for hot3-meas), with the 4.27/4.77 ms floors left explicitly `hotE-meas`'s.
- **Item 3 (01-roofline-and-byte-budget.md, lines 72–84):** Fixed by restoring the note's Sync floor (rounds × 2 us) column verbatim (MoE 0.168, KDA 0.068, mHC sites 0.180, MLA 0.044, expand 0, dense 0.006, head 0.004, total 0.470 — verified against the note's table rows at lines 92–99) and amending the parenthetical to state the gap semantics: "Gap to max(floors)" is measured − max(DRAM floor, sync floor), for mHC sites 1.785 − 0.180 = 1.61. All seven gap cells now recompute from the shown columns.

---

# Agent B Review — Pass 2

**Date:** 2026-11-06
**Reviewer:** Agent B (Critic) — cold pass 2, scope: verify the three pass-1 fixes and check the edits introduced no new correctness issues.
**Method:** both changed files (`01-roofline-and-byte-budget.md`, `04-optimization-ledger.md`) read in full; every fixed claim re-verified against the read-only regen tree (`notes/glm-decode-roofline.md` lines 10–21, 24, 48–49, 90–99, 210–213; `notes/glm-batch.md` lines 11, 30–31; `README.md` §"Roofline" line 180) and the orchestrator registry (`/tmp/research_regen_decode_engine_methodology/briefing.md` lines 99, 115).

**Verdict — all three fixes verified correct; no new issues introduced.** Per-item verification:

- **Item 1 (04, "Where it landed" close-out):** now reads "~+0.46 ms at the end ($6.05 - 5.59$)"; 6.05 − 5.59 = 0.46 ✓, consistent with the surrounding "+0.63 ms when the base was 9.3 ms" (roofline lines 210–211) and the [INFER, lines 212–213] attribution ("items 1, 7 and 8 also remove" — matches the note's own wording).
- **Item 2 (01, per-chip budget prose):** "+370 MB / 2134 MB/chip" now attributed to `hotE-meas`, matching the table row above it and `notes/glm-decode-roofline.md` line 49 verbatim ("hotE-meas adds (70% of routed reads at BFP8) | +370") ✓. The new `hot3-meas` sentence recomputes: 1763.8 + (932 − 595) = 2100.8 ≈ 2101 MB ≈ 2.10 GB/chip, with both anchors correct (`notes/glm-batch.md` lines 30–31 give plan0h 595 / hot3 932 routed MB; `README.md` "Roofline" line 180 states "hot3-meas decode reads 2.10 GB per chip per token") ✓. The file-head "1.76–2.13 GB" range and its "1763.8/2134" gloss remain the note's own plan0h/hotE framing and are undisturbed; the floors table row (hotE-meas 2134 → 4.27/4.77 ms) is consistent with the sentence's "floors below are hotE-meas's".
- **Item 3 (01, "Where the 9.3 ms goes" table):** the restored Sync floor column matches the note's original table (roofline lines 92–99) cell for cell — 0.168 / 0.068 / 0.180 / 0.044 / 0 / 0.006 / 0.004, total 0.470 ✓ — and the DRAM-floor column (1.503/2.243, 1.312, 0.030, 0.388, 0, 0.120, 0.169; total 3.524/4.264) matches too. All seven gap cells now recompute from the shown columns under the stated semantics (measured − max(DRAM floor, sync floor)): MoE 2.559−1.503=1.06, KDA 2.099−1.312=0.79, mHC 1.785−0.180=1.61, MLA 1.673−0.388=1.29, expand 0.503−0=0.50, dense 0.261−0.120=0.14, head 0.245−0.169=0.08 ✓. The parenthetical's "the only row where the sync floor, not the DRAM floor, is the binding one" is true of the note's data (mHC sites 0.180 > 0.030; every other row has DRAM floor ≥ sync floor).

**New-issue scan around the edits (cleared):** the "restored verbatim" parenthetical is accurate; the ranked-gap list ("mHC sites 1.61 …") now follows from the shown columns; the registry numbers in 04 (165.3 t/s = 6.05 ms, 178.9 = 5.59; 9295/9925 → 8312/9021) still match `briefing.md` lines 99 and 115; the floors, sync-chain, diagnosis and window sections of 01 and all ledger tables of 04 are byte-identical to the pass-1-reviewed text apart from the three fixes.

**No feedback — chapter approved.**
