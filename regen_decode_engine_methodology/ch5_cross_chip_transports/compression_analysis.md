# Compression Analysis: Cross-Chip Transports — Pass 1

## Summary
- Total files analyzed: 5 (index.md, 01-pcie-peer-bar-dma.md, 02-erisc-message-service.md, 03-rounds-and-transport-selection.md, 04-latency-compendium.md) + spot-checks of Ch1/Ch2 for cross-chapter duplication
- Estimated current line count: ~513 lines (58 + 122 + 109 + 111 + 113)
- Estimated post-compression line count: ~435 lines
- Estimated reduction: ~15%

Cross-chapter note: Ch1 (census table) and Ch2 `03-lead-helper-teams.md` (relay boundary) are correctly pointed at, not restated — no cross-chapter CRUCIAL items. All CRUCIAL items are intra-chapter: the mechanism files re-carry material that the chapter's own ownership fence assigns to `04-latency-compendium.md` (or to each other).

## CRUCIAL Suggestions

### [01-pcie-peer-bar-dma.md] ~lines 87-118 (measured-latencies table + structural-facts bullets + in-situ paragraph)
**Issue:** The entire "Measured latencies (a10u28, a LoudBox host)" section duplicates `04-latency-compendium.md` row-for-row: the 13-row table repeats compendium rows 1-3, 5-7, 9-10, 12-17, 21 (same values, same citations); the "Two structural facts" bullets restate the numbers behind compendium rows 8-11; the closing in-situ paragraph restates compendium rows 16-17 nearly verbatim ("506/550 us on the critical path of an 8.8-9.9 ms token; the mean wait over chips is 680 us"). 01 even admits the duplication in text: "the compendium (04) repeats them with full host context" — a direct violation of the index's fence ("ALL transport latency and transport-bandwidth numbers live here … the single table. Other chapters cite, never restate"). The `GLM_GATHER`-vs-cmd-146 comparison table in "Why device-originated beats host-relayed" (~lines 69-73) likewise duplicates compendium rows 21-22.
**Suggestion:** Keep in 01 only the 2-3 anchor rows the argument actually leans on (peer write 0.50/0.55 us; `glm_round` 2.15-2.93 us; the 1.46→1.02 spread arithmetic, which is analysis, not registry) and replace both full tables with pointers to the corresponding compendium rows. Cut the in-situ paragraph to one sentence citing 04 rows 16-17. Keep the "three reasons the device path wins" prose — the reasoning is 01's; the number registry is 04's.

### [01-pcie-peer-bar-dma.md ~lines 55-58] + [02-erisc-message-service.md ~lines 73-76]
**Issue:** The `glm_round_with` runtime-counter mechanism (read count at `TT_GLM_ROUND_COUNT_ADDR`, parity from the count, count stamped into word 1 of the seq line, stale-line stop `0xBADA0000 | chip`) is explained in full in 01's "Device side" section, then re-explained in 02's "Restart-correct counters — PCIe side" paragraph — including the identical MEAS quote "a chip cannot then start round N + 2, which reuses N's parity, before every peer has sent N + 1, and a peer sends N + 1 only after it has delivered N" appearing verbatim in both files.
**Suggestion:** 01 owns the runtime walk (it is the round-send-path home). Reduce 02's PCIe-side paragraph to its restart-unique content — the host clears slots *and count together* at image start, and why that joint clearing is safe — with a one-clause pointer to 01 for the runtime mechanism and the safety quote kept in only one file.

### [03-rounds-and-transport-selection.md] ~lines 33-39 ("Measured round cost, per host" + "What the transport choice is worth")
**Issue:** This section restates compendium rows 14, 16-18, 20, 37-43 and 50-52 value-by-value (2.84/2.88/2.91/3.07/3.50/3.53 lean-tree series, the 2.84-4.34 in-decode buckets vs LoudBox 2.20-3.51, box C 2.30-3.10, c01u20 3.4-4.5, the 99.4/142.3 and 165.0/161-162 t/s verdicts, the +224/+238/+480 decomposition). The caveat "the sub-4.6 KB savings did not by themselves move decode t/s, because the round waits on the slow chips absorb them" is stated here AND again in 04's reading note D. 03's own scope admits it: "every number here also appears with full host context in 04-latency-compendium.md."
**Suggestion:** Keep the *argument* in 03 (same code, opposite verdicts per host; the token-time decomposition) but cite compendium row numbers instead of restating each value series; drop the duplicate caveat sentence in favor of "see 04 note D." Target: this section shrinks from ~7 dense paragraph-lines to ~3.

### [02-erisc-message-service.md ~lines 14-15 ("The link" bullet)] + [04-latency-compendium.md ~lines 78-80 + rows 23-26, reading note A]
**Issue:** The 0.67-vs-0.39 us link-latency correction story is narrated twice in full: 02 tells it in the substrate section (early report, later trace, "it was a round trip", porter warning) and 04 tells it again in rows 23-26 plus reading note A — both versions end with the near-identical porter warning "a porter re-measuring must state which figure they mean."
**Suggestion:** Reading note A (next to the rows it explains) is the natural single home. Reduce 02's link bullet to the load-bearing number (0.39 us traced one-way) plus one pointer clause to 04 note A; delete the duplicate warning sentence from 02.

### [03-rounds-and-transport-selection.md] ~lines 44-52 vs ~lines 91-99 ("The decision rule" table vs "The rule, stated once" table)
**Issue:** Two tables state the selection rule with heavy overlap: the far-chip `far_pcie` <1 KB exception appears in both (with the same 0.4-0.5 us measurement), the decode-round PCIe-vs-Ethernet verdict appears in both (with the same bench numbers), and the prefill ≥34 KB Ethernet choice in the second table is a restatement of the economic-spine paragraph above it. "Stated once" is the third statement of material already given twice in the same file.
**Suggestion:** Merge: keep the "decision rule" ownership table (who decides, mechanism) and reduce "The rule, stated once" to only the rows not derivable from it (chip barriers stay on-chip; courier exchange stays PCIe). Drop the repeated far_pcie and round-latency values from the second table (they are one pointer away in the first table and in 04).

## MINOR Suggestions

### [index.md] ~lines 42-49 ("Files in this chapter" table)
**Issue:** The "What it gives you" column restates each file's own opening-paragraph content (e.g. 01's table description repeats 01's intro sentence nearly clause-for-clause).
**Suggestion:** Compress each cell to a 5-8 word label ("PCIe peer-BAR DMA: windows, setup, relay, LoudBox numbers"); the reading-order line plus the files' own intros carry the rest.

### [index.md ~line 52] + [01 ~lines 114-116] + [04 ~lines 8-14]
**Issue:** The a10u28 host description ("8 × p150b, each on its own Gen5 x16 root port; chips 0-3 on NUMA 0 and 4-7 on NUMA 1; 2 × EPYC 9124") appears three times near-verbatim (index host caution, 01 "Host and socket dependence", 04 host key), and the Galaxy x1 description twice (index, 04). Index says this is "stated once and kept throughout."
**Suggestion:** 04's host key is the canonical registry; 01's restatement can become "a10u28's topology is 04's host key" with only the NUMA-split latency consequence (0.50 vs 0.55 us) kept locally.

### [02-erisc-message-service.md] ~lines 99-106 (reliability-model contrast table)
**Issue:** The table's cells re-quote citations and phrasing already given verbatim earlier in the same file (credits/NIU counts, defaced slots) and in 01 (read-back fence, `0xBADA0000`). The table is the only side-by-side contrast and should stay, but the duplicate quote-and-citation payload inflates it.
**Suggestion:** Compress cells to short phrases without repeating full quotes/citations already given in 02's body; one pointer per row suffices.

### [02-erisc-message-service.md] ~lines 107-108 (final "design notes" paragraph)
**Issue:** The relay cost (0.30-0.35/0.43 us) and ~34 GB/s loop cap duplicate 04 rows 32 and 54; only the round decomposition ("2.34 + 0.45 contention + 0.13 per KB") and the rejected ERISC-direct delivery are unique to this paragraph.
**Suggestion:** Keep the two unique items, point to 04 rows 32/54 for the rest.

### [03-rounds-and-transport-selection.md] ~lines 82-88 (topology fabric-reality bullets)
**Issue:** The bullets quote measured values that are also compendium rows (cables +0.05 us = row 26; one-link-per-pair penalty = row 36; uniform-fabric pings = rows 46-47; 4th-hop +0.65 = row 61). The *argument* (why Galaxy runs the service on cube groups, not bus groups) is unique; the raw numbers are not.
**Suggestion:** Keep the argument, replace inlined values with row pointers where identical.

### [03-rounds-and-transport-selection.md ~line 102] + [04-latency-compendium.md ~lines 106-109]
**Issue:** Both file endings enumerate the same re-measure checklist items ("PCIe per socket topology, ERISC per fabric, plus the DRAM window rate, the t/s ladder, and smoke-gate results") and the same Ch8 pointer that the index fence also states.
**Suggestion:** Keep the enumerated list once in 04 (the checklist's reference table); 03's ending becomes a single pointer sentence.

### [01-pcie-peer-bar-dma.md] ~lines 112-113 (end of latency section)
**Issue:** The two-sentence "old step mesh sync measured 4.51 us bench / 4.4-5.4 in situ before it was deleted" restates what 03's machinery section explains in full (with the md5s and savings ledger) and what 04 row 15 + note C register — and 01 itself ends the sentence pointing to 03 as the owner.
**Suggestion:** Cut to "the deleted step mesh sync is 03's story (04 row 15)."

### [02-erisc-message-service.md ~line 69] + [03 ~line 28]
**Issue:** The quote "exactly where a PCIe round's write lands" is used in both files to make the same point (Ethernet delivery reuses the PCIe round's arrival machinery), and 02 also repeats it in its reliability table row.
**Suggestion:** Keep the quote at 02's round-integration point; 03's step-4 of the machinery list can say "delivery and release are identical either way (02's round-integration point)".

### [all files, scope paragraphs]
**Issue:** Each file's **Scope** paragraph re-lists the ownership division (ERISC is 02's; selection is 03's; image budget is Ch7's; mechanics are 01/02's) that the index fence states once — three to four repeated pointer-sentences per file.
**Suggestion:** Trim each scope paragraph to the file's own claim plus primary evidence; the fence in the index already assigns every boundary.

## Load-Bearing Evidence
(Verdict is 'yes', but included so Agent A does not over-cut while applying the CRUCIAL items.)
- `index.md` ~line 22: "The largest round a GLM step can ask for is 4608 B per chip … that fact drives most of the selection rule" — load-bearing: this single sentence is the bridge from the per-kind round table to the selection rule; it is stated once here, correctly not restated in 03.
- `01-pcie-peer-bar-dma.md` ~lines 30-38 (host-side steps 1-3 + window table row for window 19): the iATU/BAR mapping procedure and the relaxed-window bandwidth rationale ("a few GB/s against ~36 GB/s inbound") exist only here; 04 has none of the setup mechanics. Do not cut on latency-table grounds.
- `01-pcie-peer-bar-dma.md` ~lines 46-48: the gather ordering evidence ("0 in 390k pieces" vs "30-107 reorderings") — unique measured argument for seq-after-payload, not in 04; keep.
- `02-erisc-message-service.md` ~lines 73-79 (ERISC-side resume-from-published-counts + the two flanks): the restart-correctness content unique to 02 (nonce per load, resume across image starts, defaced slots) is the file's core and appears nowhere else; only its PCIe-side paragraph is the duplicate (CRUCIAL #2).
- `03-rounds-and-transport-selection.md` ~lines 20-29 (round phase table + "sends cost 75 cycles, limit downstream of the NIU"): the phase decomposition exists only here (04 has no per-phase rows); keep.
- `03-rounds-and-transport-selection.md` ~lines 60-69 (economic spine, the two fits and both crossovers with arithmetic): the *derivation* is unique to 03 even though the fit constants appear in 04 rows 58-60; keep the arithmetic, cite rows for constants.
- `04-latency-compendium.md` entire table + reading notes A-E: the registry itself is the fence's single source of truth; every CRUCIAL item above *adds* load to it rather than removing from it. Note E and the host key are its irreducible core.

## VERDICT
- Crucial updates: yes

## Change Log — Pass 1 (2026-11-06, Agent A)

All five CRUCIAL items applied; numbers re-verified in the read-only regen tree where touched (`notes/glm-ccl.md` lines 36/38-39/43, `notes/glm-eth.md` lines 408-412, `notes/glm-glx-ethdec.md` lines 12-13/58/71).

1. **[01] measured-latencies table + gather table + in-situ paragraph → anchors + pointers.** The 13-row latency table is now a 3-row anchor table (peer write 0.50/0.55 = row 5; `glm_round` 2.15/2.26/2.93 = row 14; all-to-all 1.46 = row 9), each row carrying a clickable pointer into `04-latency-compendium.md`; the "compendium repeats them" admission is gone. The two structural-facts bullets and the $1.46-1.02$ spread arithmetic stay (analysis, per the suggestion). The B=32 `GLM_GATHER`-vs-cmd-146 table is replaced by a pointer to row 22 (the FACT replacement claim and its citation stay; the three-reasons prose is untouched). The in-situ paragraph is now one sentence pointing to rows 15-17 and to 03 for the sync-removal story — this also applies the MINOR-9 step-mesh-sync cut.
2. **[01 + 02] `glm_round_with` counter mechanism de-duplicated.** 01's "Device side" walk keeps the full runtime mechanism and the MEAS safety quote ("cannot start round N + 2 ..."), which now appears exactly once in the chapter. 02's "PCIe side — counters restart WITH the image" paragraph is reduced to its restart-unique content — the host zeroes RX slots *and count together* at image start, and why that joint clearing is safe — with pointer clauses to 01's walk for the runtime mechanism and the safety argument.
3. **[03] "Measured round cost, per host" + t/s verdict → row-cited argument.** The section no longer restates rows 14, 16-18, 20, 37-43, 50-52 value-by-value; it cites those rows by number, keeps the reading (rows 41-42 as the only like-for-like in-situ comparison, 04 note B's no-conflation rule, 04 note D for the slow-chips caveat — the duplicated caveat sentence is gone), and keeps the argument's own anchors that are not in 04: the md5 `383e072f` bit-exactness and the 6972-vs-6029 us absolute token times feeding row 52's decomposition. Section is now 3 dense paragraph-lines (target met).
4. **[02 + 04] 0.67-vs-0.39 correction story → 04 note A only.** 02's link bullet keeps the load-bearing traced 0.39 us one-way with its citation and one pointer clause to 04 reading note A + rows 23-26; the early-report narration, the in-board/cable split, and the duplicated porter warning are deleted from 02 (the warning now lives only in 04 note A). 02's scope sentence ("appear here with citations and again in 04") updated to match.
5. **[03] "The decision rule" + "The rule, stated once" merged.** The ownership table is untouched. "The rule, stated once" is replaced by "Two rulings the ownership table does not derive" — chip barriers stay on-chip (cost now a row-3 pointer), courier exchange stays PCIe — with the repeated far_pcie values, round-latency values, and the prefill choice row dropped (each is one pointer away in the ownership table, the economic-spine paragraph, and 04). The closing paragraph is a single pointer to Ch8 `04` + the compendium (applies MINOR 7; the bench-template enumeration now lives only in 04's ending).

MINOR items applied (at discretion, all row-pointer reductions verified against 04's table): #2 partially (01's host-topology restatement → "04's host key", keeping the NUMA-split consequence and lead-placement quote; index and 04 host keys untouched); #4 (02's design-notes paragraph keeps the round decomposition and rejected ERISC-direct delivery, points to rows 32/54); #5 (03's fabric-reality bullets keep the argument, inlined values → row 26/36/46-47/61 pointers); #7 (03's ending); #8 (03's step-4 now points to 02's round-integration point; the quote stays at 02); #9 (folded into CRUCIAL 1's in-situ cut). MINORs #1 (index table cells), #3 (02's reliability table), and #11 (scope paragraphs) skipped as flagged-keep/risky; one accuracy tweak made instead: index's 01 cell and 01's intro now say "latency anchors" rather than implying 01 carries the latency registry.

Non-breakage checks: all five files keep correct navigation footers (04 still ends pointing to `../ch6_host_driver_weights_oracle/index.md`); every surviving claim keeps exactly one evidence tag; no citation was orphaned (claims and their tags were removed together); all new cross-references are clickable relative links; plan coverage maintained — every number the plan's Ch5 bullets name (0.50/0.55, 1.02, 1.74, 2.84-3.5) is still stated in the chapter, now in 04's single table per the index fence, with 01/03 citing by row. Line counts after pass 1: index 58, 01: 106, 02: 109, 03: 102, 04: 113 (04 unchanged — the registry only gained load).

## Change Log — Pass 2 (2026-11-06, Agent A)

Item 1 from Agent B pass 2 applied — the one change it names, and nothing else:

1. **[01-pcie-peer-bar-dma.md] "Why device-originated beats host-relayed", second bullet — inverted verdict fixed.** "The one restructure that *did* pay was measured: spreading round traffic over the receiving chip's leads was tried and dropped" → "The one restructure that was tried did **not** pay: spreading round traffic over the receiving chip's leads was measured and dropped". The bullet now leads with the negative verdict and matches both its own continuation ("The root-relay shape is the measured winner") and the cited source (read-only verification against `regen-glm5.3-flash/notes/glm-ccl.md` lines 175-207: the section is titled "Tried: spreading round traffic over the receiving chip's leads (negative)" and ends "Dropped."). Both evidence tags on the bullet ([MEAS] and [INFER], notes/glm-ccl.md lines 175-207) are untouched. The same inverted claim does not occur anywhere else in the chapter (checked all chapter files; the only other hit is Agent B's own review file).

# Compression Analysis: Cross-Chip Transports — Pass 2

(Re-check pass, 2026-11-06, Agent C. All five content files re-read line by line; every Pass-1 CRUCIAL item re-verified against the change logs. Scope strictly redundancy/bloat — factual correctness left to Agent B.)

## Summary
- Total files analyzed: 5 (index.md, 01-pcie-peer-bar-dma.md, 02-erisc-message-service.md, 03-rounds-and-transport-selection.md, 04-latency-compendium.md)
- Estimated current line count: ~488 lines (58 + 106 + 109 + 102 + 113, wc-l verified)
- Estimated post-compression line count: ~478 lines
- Estimated reduction: ~2% (Pass 1 already took the bulk: 513 → 488 actual; Pass-2 findings are pointer swaps, one duplicate table row, and intra-file repetition, not table eliminations)

## CRUCIAL Suggestions
None. All five Pass-1 CRUCIAL items are resolved (see Pass-1 Resolution below), and the fresh sweep found no new table-vs-table or mechanism-vs-mechanism duplication of Pass-1 scale. The remaining findings are MINOR.

## MINOR Suggestions

### [04-latency-compendium.md] rows 27 and 53 are the same measurement, listed twice
**Issue:** Row 27 ("ERISC stream, one link, 4 KB+ transfers", 49.15 GB/s) and row 53 ("Bandwidth, link — ERISC stream, one link", 49.15 GB/s) carry identical values and the identical citation [MEAS, notes/glm-ccl.md (line 127)]. In the file whose stated purpose is "Every measured cross-chip transport number in the corpus, in ONE table", a self-duplicated row is a fence self-violation — the one registry that must not restate itself.
**Suggestion:** Merge into one row (the Bandwidth section is its natural home); if the row-numbering is load-bearing (other files cite by row number), renumber or leave a tombstone cross-reference ("row 27 = row 53").

### [03-rounds-and-transport-selection.md] economic-spine section still restates compendium constants without row pointers
**Issue:** The two fits (5 us + 0.33/KB vs 15 us + 0.037/KB) and the ~34 KB / ~7 KB crossover restate rows 58-59; the bandwidth parenthetical restates rows 27/53 (49 GB/s raw), 54 (~34 GB/s), 55 (195 GB/s with feeders), and 21 (21-23 GB/s); the x1 crossover quote restates row 60 verbatim; and the closing "On the LoudBox, the same ranking read at round level" paragraph restates row 33's (2.77-3.31 us) and row 20's (2.30-3.10 us) endpoints. Pass-1's load-bearing note said keep the derivation but "cite rows for constants" — the arithmetic stayed, the row cites never arrived.
**Suggestion:** Keep the fits, the crossover algebra, and both verdict quotes (they are 03's analysis); add "rows 58-59 / 60" pointer clauses; compress the LoudBox-round-level paragraph to its unique content (the 2.73-2.91 3-cube series, which is not a compendium row, and the "0.2-0.5 us: one hop of x16 against three hops" verdict) with rows 33/20 cited by number.

### [02] + [03] "a round may mix transports" stated three times
**Issue:** The bolded claim appears at 03 line 17 (machinery step 3), 03 line 50 ("which is exactly why ... a round may mix transports", re-quoting the same citation block already cited at line 17), and 02 line 69 (round-integration bullet).
**Suggestion:** Keep the full statement at 03's machinery step 3 (it is the machinery's own point); 02's bullet keeps its unique mask detail ("none / the antipode / all of them") and drops the bolded slogan in favor of a "(03's mix rule)" clause; 03's line-50 restatement drops the clause, keeping the bit-exact quote.

### [03] machinery step 1 restates the counter mechanism 01 now owns, without a pointer
**Issue:** Pass-1 item 2 made 01 the single home of the `glm_round_with` runtime-counter walk, and 02 now correctly points there — but 03's machinery step 1 ("The root reads the round count at `TT_GLM_ROUND_COUNT_ADDR`, bumps it, and picks that round's RX parity") re-describes the same mechanism with no pointer to 01. A residual of the CRUCIAL-2 class, one file over.
**Suggestion:** Keep 03's unique addition (counts agree across chips → no mesh sync between steps) and add one clause: "the full counter walk and safety argument are 01's."

### [index] + [01] + [04] a10u28 topology string still appears three times
**Issue:** The topology ("8 × p150b, one Gen5 x16 root port each, chips 0-3/4-7 across two EPYC 9124 sockets") is near-verbatim in the index host caution (line 52), 01's host/socket section (line 100), and 04's host key (line 11) — while the index says "stated once and kept throughout". Pass-1 MINOR #2 was applied only partially: 01 line 100 now *points at* 04's host key but restates the full topology first anyway.
**Suggestion:** 04's host key is the registry; 01's clause shrinks to "a10u28's topology is 04's host key" keeping only the NUMA-consequence sentence (0.50/0.55, 0.96/1.06); the index host caution keeps the never-average warning but can hand the fabric facts to 04's key.

### [01] "host work is load-time, not per-token" stated three times in one file
**Issue:** Intro (line 3, "The host's entire job is load-time plumbing"), "What device-originated means" (line 9, "The host's entire role is load-time"), end of host-side steps (line 32, "Nothing in steps 1-3 costs a token anything"), and the third why-device-wins bullet (line 77) re-opens the same point. Four statements, one fact.
**Suggestion:** Keep line 9 (mechanism section) and line 32 (closes the setup steps); trim the intro clause and the line-77 bullet to not re-argue it.

### [index] + [01] + [03] + [04] the port re-measure pointer appears in all four files
**Issue:** "A port inherits none of these rows / nothing transfers; the re-measure checklist is Ch8 `04`'s" appears at index (fence), 01 (line 100 end), 03 (line 92), and 04 (line 107, the full owner's statement with the enumerated checklist). 04's is the load-bearing one; the others are one-sentence restatements.
**Suggestion:** Keep the full statement in 04 and the one-line fence in index; 01/03 collapse to bare "(re-measure: Ch8 `04`)" clauses. Low severity — navigational repetition, but four copies is three more than the fence needs.

### Still-open Pass-1 MINORs, deliberately skipped by the writer (recorded, not re-litigated)
Pass-1 MINORs #1 (index files-table cells restating each file's intro), #3 (02's reliability-table cells re-quoting payload already in 02/01 bodies), and #11 (each file's Scope paragraph re-listing the index's ownership fence) were skipped as flagged-keep/risky and remain present. Agent C notes them for honesty but does not re-flag them: the writer's keep rationale (navigational table, contrast-table readability, per-file self-containment) is defensible, and none has grown worse.

## Load-Bearing Evidence
- `index.md` ~line 22: "The largest round a GLM step can ask for is 4608 B per chip … that fact drives most of the selection rule" — the single bridge from the per-kind round table to the selection rule; stated exactly once, and correctly NOT restated in 03. Cannot be cut or the round table loses its punchline.
- `01-pcie-peer-bar-dma.md` ~line 52 (relay section): the gather ordering evidence — "0 in 390k pieces" against "30-107 reorderings per run", "so a seq written after the pieces is enough. A seq carried inside the payload is not" — a unique measured argument with no compendium row; 04 registers values, not this design proof. Keep verbatim.
- `02-erisc-message-service.md` ~line 75: "resumes each ERISC's ring 0 from the ERISC's published sent count, because every round of the previous image finished on every chip" — the restart-unique core of the ERISC counter discipline; the whole nonce/resume/deface taxonomy hangs off this sentence and it appears nowhere else in the guide. Not cuttable.
- `03-rounds-and-transport-selection.md` ~lines 60-62: the crossover algebra "$5 + 0.33k = 15 + 0.037k \Rightarrow k \approx 34$ KB" — the derivation is unique to 03 even though the constants mirror rows 58-59; cutting it would leave the selection rule's economics as unexplained registry rows (this is why MINOR 2 above asks for row cites, not cuts).
- `04-latency-compendium.md` ~line 86 (reading note A): "A porter re-measuring must state which figure they mean." — now the single home of the 0.67-vs-0.39 correction story (02 correctly points here since Pass 1); both the note and the dual rows 23-25 stay because both readings were used in real decisions, per the note's own justification.

## VERDICT
- Crucial updates: no

## Pass-1 Resolution
1. [01 latency/gather/in-situ duplication] **resolved** — 01's latency section is now a 3-row anchor table with clickable row-5/14/9 pointers; the B=32 gather table is a one-sentence row-22 pointer; the in-situ paragraph is one sentence citing rows 15-17 and 03. No 13-row table, no "compendium repeats them" admission.
2. [01+02 `glm_round_with` counter mechanism] **resolved** — the safety quote "a chip cannot then start round N + 2 …" appears exactly once in the chapter (01's device-side walk, grep-verified); 02's PCIe-side paragraph is reduced to the joint-clearing argument with pointer clauses to 01. (Residual one-file-over overlap with 03 noted as Pass-2 MINOR #4.)
3. [03 per-host costs + t/s verdict restating rows] **resolved** — the section cites rows 14, 16-18, 20, 37-43, 50-52 and notes B/D by number; no value series restated; the kept 6972-vs-6029 us and md5 `383e072f` anchors are not compendium rows (change-log rationale verified). (Economic-spine constants still lack row cites — Pass-2 MINOR #2.)
4. [02+04 0.67-vs-0.39 story narrated twice] **resolved** — 02's link bullet keeps only the traced 0.39 us with citation plus a pointer to 04 note A/rows 23-26; the narration and porter warning live only in 04 note A (grep-verified: "porter re-measuring" occurs only in 04).
5. [03 two overlapping decision-rule tables] **resolved** — "The rule, stated once" is gone; replaced by "Two rulings the ownership table does not derive" (two bullets: chip barriers on-chip with a row-3 cost pointer; courier exchange stays PCIe). The far_pcie and round-latency values now appear only in the ownership table.


