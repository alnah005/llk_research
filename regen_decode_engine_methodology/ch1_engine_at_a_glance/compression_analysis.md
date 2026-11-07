# Compression Analysis: Chapter 1 — The Engine at a Glance — Pass 1

## Summary
- Total files analyzed: 6
- Estimated current line count: ~507 lines
- Estimated post-compression line count: ~430 lines
- Estimated reduction: ~15%

## CRUCIAL Suggestions

### [01-mental-model.md + 05-measured-envelope-and-evidence.md] ~lines 46-49 / 12-30
**Issue:** The "Why it pays — the headline numbers" section of `01-mental-model.md` restates, with full citations, four envelope rows that `05-measured-envelope-and-evidence.md` declares as its sole registry: 165.3 t/s = 6.05 ms, 178.9 t/s = 5.59 ms, repeat stability 165.29/165.30, and the 48-token → 262,014-token context decay. Both files even carry the identical citations (`notes/post-rc/glm-rc1.md lines 136-139`, `README.md "Performance"`). This is a restated table — the exact class of duplication the chapter's own ownership rules prohibit ("the owning chapter carries the full exposition").
**Suggestion:** Cut the measured-number detail from `01`'s headline section down to one sentence — "the engine serves GLM-5.3-Flash at 165.3 t/s (6.05 ms/token) near a hard DRAM floor" — with a single pointer to `05`'s envelope rows 1-4. Delete the repeat-stability and decay numbers from `01` entirely; they are rows 3 and 4 of the envelope table and nowhere else needed in this file.

### [01-mental-model.md + 02-anatomy-of-a-token.md] ~lines 55-58 / 8-9
**Issue:** The host-edge cost triplet (0.05 ms embedding lookup, 0.09 ms input streams, 0.05 ms post and argmax read) is stated twice, in `01`'s "What the host still does" and again in `02`'s walk step 0, with the same citation (`notes/glm-decode-roofline.md "Headline"`). Both passages also both note that the embedding row is read on the host and cost no device DRAM.
**Suggestion:** Keep the numbers once — in `02`'s step 0, where they sit on the token walk they describe — and reduce `01`'s section to two sentences (host owns the embedding lookup and the final token read; the boundary is Ch2 `05`'s) with a link, dropping the ms triplet and its citation from `01`.

### [01-mental-model.md] ~lines 22-37
**Issue:** The five "inversion" bullets and the side-by-side contrast table encode the same contrast twice. Bullet 2 (nothing launched per step) = table row "Per-step host cost"; bullet 3 (raw MMIO, no llk/CBs, TRISCs idle) = table rows "Compute interface" and "TRISCs"; bullet 4 (device-initiated peer-BAR DMA + ERISC) = table row "Cross-chip I/O"; bullet 5 (record list is the trace) = table row "The 'trace'"; bullet 1 (records as data) = table rows "Resident program" and "Dispatch-overhead mitigation". Roughly 16 lines of prose restate what the 10-line table already shows.
**Suggestion:** Keep the table as the contrast artifact and trim the bullets to the anchors and citations only (the bullets carry the `src/` line facts the table lacks — `TtGlmStepRecord` 256 B, `TT_COMMAND_GLM_STEPS` = 124, `brisc.cpp` anchors). Each bullet should be one line of fact + citation, not a paragraph that pre-narrates the table row.

### [02-anatomy-of-a-token.md + 05-measured-envelope-and-evidence.md] ~lines 55-57 / 20-21
**Issue:** The streaming-window rate (447 GB/s = 89% of 500 GB/s peak) and the DRAM-busy plan requirement (79% `plan0h` / 95% `hotE`) appear with full citations in BOTH `02`'s "Where the 6.05 ms goes" section and `05`'s envelope rows 5-6. The rate also appears a third time, again with citation, in `01` principle 3. Three restatements of registry rows 5-6 inside the same chapter.
**Suggestion:** `05` is the registry owner. In `02`, collapse the two sentences to "inside streaming windows the engine sustains 447 GB/s per chip; the plan's DRAM-busy constraint and the window-rate exposition are Ch4 `01`'s" (envelope pointer, no re-citation). In `01` principle 3, drop the parenthetical citation and rate; the principle needs only "bandwidth-bound → streaming discipline," with the number left to `05`/Ch4.

### [03-workload-context.md + 05-measured-envelope-and-evidence.md] ~lines 43-45 / 22, 50-58
**Issue:** The full requant-class policy sentence ("routed experts BFP4 with hot experts at BFP8, router and indexer compressor gate BF16 strips, every other matrix BFP8, small parameters at source precision") is written out three times in the chapter: `03`'s "Strip packing" section, `05`'s envelope row 11, and `05`'s worked example 1 (which quotes the `requant.cpp` comment verbatim). Three full restatements of the same policy in two files.
**Suggestion:** Keep the row-11 statement (registry) and the worked-example quotation (it is the pedagogical payload of example 1). In `03`'s "Strip packing," reduce to "which tensor group gets which class is requant policy (envelope row 11, owned by Ch6 `02`)" — the per-class detail adds nothing `05` row 11 doesn't already carry, and `03`'s own scope note promises it "restates neither."

### [02-anatomy-of-a-token.md + 03-workload-context.md] ~lines 37-39 / 92
**Issue:** The index-list selection arithmetic (top-512 pooled keys → 2048 positions plus a tail of up to 3 → at most 2051 selected positions) is explained twice, once in `02`'s "Two step-list variants" and again in `03`'s MLA+DSA decoder-ring entry, both citing the same inventory line.
**Suggestion:** Keep the full explanation in `02` (it motivates the two-variant split); in `03`'s decoder ring, reduce to "the DSA indexer — compressor gate, pooled keys; selection capped at 2051 positions (see `02`'s list variants)".

## MINOR Suggestions

### [index.md + 01-mental-model.md] ~lines 3-5 / 1-3
**Issue:** The index's opening paragraph and `01`'s opening paragraph state the same inversion in nearly the same words ("the host compiles model, weights, and topology into flat step-record lists… one persistent BRISC interpreter per Tensix core… no per-step dispatch and no host in the token loop").
**Suggestion:** Let the index paragraph be the one-sentence hook and start `01` directly at "In the stack the audience knows…" — or vice versa. One of the two narrations can go.

### [05-measured-envelope-and-evidence.md + 04-source-map.md] ~lines 58-63 / 36-62
**Issue:** Worked example 2 (anchor precision) restates two anchors that `04`'s verified-anchor table already carries: `glm_step_list` at `brisc.cpp:21153` with its doc comment at `21151-21152`, and `NOC1_BASE`/`NOC0_BASE` at `brisc.cpp:694` / `fw.h:16`. The example needs the *pattern*, not the full re-listing.
**Suggestion:** Trim example 2 to one anchor (`glm_step_list`) plus a one-clause "same rule at file grain: NOC1 vs NOC0," pointing at `04`'s anchor table instead of restating both pairs.

### [03-workload-context.md] ~lines 20 and 96
**Issue:** The clamped-SwiGLU bounds are stated twice within the same file — "(gate <= 10, -10 <= up <= 10)" in the layer-map table and "gate clamped at 10, up clamped to [-10, 10], then silu(gate) × up" in the decoder ring, both citing the same inventory line.
**Suggestion:** Keep the clamp once (the decoder-ring entry, where the SwiGLU semantics live) and drop the parenthetical from the table row.

### [04-source-map.md] ~lines 90-97
**Issue:** The "what you can safely never read" list overlaps the corpus map above it: the `qwen-*` notes exclusion appears in both (corpus table: "not primary evidence for this guide except where a GLM note cites them"; never-read list: "Read one only when a GLM note cites it"), and the host API-surface files are excluded both in the `src/` map paragraph and again in the list.
**Suggestion:** Merge: drop the `qwen-*` and API-surface items from the never-read list and mark those rows in the earlier tables with "(skip)" — the list then carries only its unique items (bench/proxy paths, `pending/`, prefill design, vision).

### [index.md] ~lines 3-5 and 72-74
**Issue:** The "Evidence conventions" closing paragraph summarizes the four tag types that `05` defines in full; the summary adds only the tag names, which the glossary-style value is low since the reader must open `05` for the rules anyway.
**Suggestion:** Reduce to one sentence: "Every claim carries exactly one evidence tag (`[MEAS, …]`, `[FACT, …]`, `[EST, …]`, `[INFER, …]`) — rules and worked examples in `05`."

### [01-mental-model.md] ~line 36
**Issue:** Principle 3 ends with the full citation parenthetical "(static_assert-pinned instruction counts [FACT, src/brisc.cpp:65-69]; full argument in Ch3 `05`)" — hedging/justification language ("viable only because of…") that restates principle 5's determinism point two items later.
**Suggestion:** Trim to "…replace llk, CBs, and TRISC kernels (Ch3 `05`)." The determinism justification is principle 5's job.

### [04-source-map.md] ~lines 99-100
**Issue:** The "Navigating the big files" paragraph repeats the anchor-verification caveat already stated in the scope note (line 7: "every anchor in the anchor table was re-verified line-by-line") and in `05`'s worked example 2 ("any file that claims its anchors were verified must have verified every one of them") — the same caveat three times in one chapter.
**Suggestion:** State the re-verification claim once (the `04` scope note is the natural home) and drop the mid-paragraph repetition in "Navigating the big files."

### [02-anatomy-of-a-token.md] ~lines 68-78
**Issue:** The "Where the chapters land on this walk" table substantially duplicates the index's forward map (both assign Ch2-Ch8 to the same subsystems) plus the scope pointers already embedded in each walk step ("Ch2 `04`'s subject", "Ch5's subject", etc.).
**Suggestion:** Keep the walk-stage table (its stage-indexed framing is genuinely different from the index's question-indexed map) but strip the in-step "ChX's subject" pointers from steps 1-4, which the table then makes redundant.

## Load-Bearing Evidence
- `index.md` line ~4: "the host compiles model, weights, and topology into flat step-record lists once, and one persistent BRISC interpreter per Tensix core executes those records as data" — load-bearing because it is the chapter's single-sentence thesis that orients every later chapter; the index must stand alone for readers who jump straight to Ch2-Ch8.
- `01-mental-model.md` line ~24: "one command can cover a whole multi-position row because `glm_steps` iterates its `tokens` field on the device [FACT, src/brisc.cpp:21182-21190]" — load-bearing because the no-per-step-dispatch claim (the inversion's core) rests on this specific device-side loop fact; cutting it would leave the headline claim uncited.
- `02-anatomy-of-a-token.md` lines ~40-47: the sync-chain per-variant table (561/693 steps, 194/216 mesh syncs, 429/473 PCIe rounds) — load-bearing because it is the only place both list variants' counts are stated side by side, and its preamble's warning (a count quoted for one variant alone would be wrong) prevents a real misreading; no other chapter file carries these numbers.
- `03-workload-context.md` lines ~50-62: the per-class byte-budget table (KDA 4.972 GB, routed 4.756 GB, … device total 13.61 GB) — load-bearing because it is the substrate for the entire DRAM-bound argument and Ch4's motivation; it appears nowhere else and each row is the denominator of a later roofline claim.
- `04-source-map.md` lines ~9-30: the source-file map with verified `wc -l` counts and responsibility columns — load-bearing because "locate by anchor, never by scrolling" is the chapter's operational contract for a 23,665-line `brisc.cpp`; removing any row would orphan that file's citations.
- `05-measured-envelope-and-evidence.md` lines ~48-57: worked example 1's WRONG/right citation pair for the hot-expert claim — load-bearing because it is the canonical enforcement illustration of the one-tag-per-claim rule; the rule text above it is inert without the concrete failure shown.

## VERDICT
- Crucial updates: yes

## Change Log — Agent A, Pass 1

CRUCIAL items (all 6 applied):
1. **01 headline numbers vs 05 rows 1-4** — applied: `01`'s "Why it pays" cut to the 165.3 t/s (6.05 ms/token) sentence plus an envelope-rows-1-4 pointer; repeat-stability and decay numbers deleted from `01` (05 rows 3-4 own them with citations).
2. **host-edge cost triplet in 01 vs 02 step 0** — applied: `01`'s "What the host still does" reduced to two sentences (host owns embedding lookup + final token read; boundary is Ch2 `05`'s) linking `02`'s step 0; ms triplet and its citation dropped from `01` (02 keeps them).
3. **01 inversion bullets pre-narrating the contrast table** — applied: bullets trimmed to fact + citation + glossary links only (record 256 B, `TT_COMMAND_GLM_STEPS` = 124, `glm_steps` device-side `tokens` loop, `brisc.cpp` anchors kept; the load-bearing glm_steps/tokens citation retained verbatim).
4. **447 GB/s / DRAM-busy triple restatement (02, 05 rows 5-6, 01 principle 3)** — applied: `02`'s rate bullet collapsed to the bare number + Ch4 `01` pointer + envelope-rows-5-6 pointer, no re-citation; `01` principle 3 drops both the 2.10 GB and 447 GB/s numbers and citations, pointing at envelope rows 5-7.
5. **requant-class policy stated three times (03, 05 row 11, 05 example 1)** — applied: `03`'s "Strip packing" reduced to "which tensor group gets which class is requant policy — envelope row 11, owned by Ch6 `02`"; 05 row 11 and worked example 1 kept.
6. **2051 selection arithmetic in 02 vs 03 decoder ring** — applied: `03`'s MLA+DSA entry reduced to "compressor gate, pooled keys; selection capped at 2051 selected positions" with a pointer to `02`'s list variants; full arithmetic and citation stay in `02`.

MINOR items:
1. **index vs 01 opening duplication** — applied: `01`'s opening paragraph trimmed to a one-sentence abstract deferring the thesis to the index (the load-bearing hook stays in `index.md`).
2. **05 worked example 2 restating 04's anchors** — applied: trimmed to the `glm_step_list` anchor plus a one-clause file-grain pointer (NOC1 vs NOC0) into `04`'s anchor table; the addresses' citations survive in 05 rows 15-16.
3. **SwiGLU clamp stated twice in 03** — applied: parenthetical dropped from the layer-map table row; decoder-ring entry keeps the semantics and the citation.
4. **04 never-read list overlapping the corpus map** — applied: `qwen-*` and API-surface bullets removed from the list; those rows now carry explicit "(skip)"/"**skip**" markers in the `src/` map paragraph and the notes-corpus table; vision bullet kept as a unique item minus the duplicated `gguf.cpp`.
5. **index "Evidence conventions" summary** — applied: reduced to a single sentence naming the four tags, deferring rules/envelope/examples to `05`.
6. **01 principle 2 determinism hedging** — applied: "viable only because..." parenthetical and the `[FACT, src/brisc.cpp:65-69]` citation removed (the static_assert anchors remain in `04`'s verified anchor table and src map); principle 5 owns the determinism argument.
7. **04 "Navigating the big files" repeating the anchor-verification caveat** — applied: mid-paragraph repetition dropped; the scope note retains the single verification claim.
8. **02 in-step "ChX's subject" pointers vs the walk-stage table** — applied: stripped from steps 1, 2 (Ch3 clause only; intra-chapter 03 pointer kept), and 4; step 0's pointers left per the suggestion's explicit steps-1-4 scope, and the walk-stage table kept intact.

---

# Compression Analysis: Chapter 1 — The Engine at a Glance — Pass 2
Date: 2026-11-06

Re-check pass: only the 6 CRUCIAL items from pass 1 were re-examined, plus a scan for new redundancy introduced by the fixes.

## Summary
- Files re-read: 6 (index.md, 01, 02, 03, 04, 05) — content files untouched by this pass.
- Actual line counts (`wc -l`): index.md 75, 01-mental-model.md 57, 02-anatomy-of-a-token.md 86, 03-workload-context.md 102, 04-source-map.md 105, 05-measured-envelope-and-evidence.md 77 — chapter total 502 content lines (analysis files excluded).
- Pass 1 estimated ~507 lines → ~430 post-compression; actual is 502, so Agent A's cuts were mostly pointer-substitutions rather than net deletions — but the six restatement classes are genuinely gone (verified item-by-item below), which was the actual goal.
- All 6 CRUCIAL items resolved; no fix introduced NEW significant redundancy. Three small pointer-duplication side effects noted under MINOR.

## CRUCIAL
none of the 6 items recurred — each verified as follows:

1. **01 headline numbers vs 05 rows 1-4** — RESOLVED. `01`'s "Why it pays" is now two sentences: the 165.3 t/s (6.05 ms/token) headline plus pointers to envelope rows 1-4 and Ch4's roofline. Repeat-stability (165.29/165.30) and decay (48 → 262,014 tokens) numbers no longer appear in `01`; they live only in `05` rows 3-4 with citations.
2. **Host-edge cost triplet in 01 vs 02 step 0** — RESOLVED. `01`'s "What the host still does" carries no ms numbers and no roofline citation; the 0.05/0.09/0.05 triplet with `[MEAS, notes/glm-decode-roofline.md "Headline"]` appears once, in `02`'s step 0.
3. **01 inversion bullets pre-narrating the contrast table** — RESOLVED. The five bullets are now one-fact-plus-citation lines (256-byte record + constants.h anchor, `TT_COMMAND_GLM_STEPS` = 124, `glm_steps` device `tokens` loop, brisc/erisc pointers); the contrast table is the only place the conventional-vs-regen narration is spelled out. The retained `glm_steps`/`tokens` citation was explicitly load-bearing (pass-1 evidence section) and correctly kept.
4. **447 GB/s / DRAM-busy triple restatement** — RESOLVED. `02`'s rate bullet keeps only the bare 447 GB/s number with Ch4 `01` and envelope-rows-5-6 pointers, no citation; `01` principle 3 carries no rate, no byte count, no citation — just "envelope rows 5-7" and the Ch4 pointer. The number-with-citation now appears exactly once in the chapter (`05` row 5).
5. **Requant-class policy stated three times** — RESOLVED. `03`'s "Strip packing" now ends with the one-line "which tensor group gets which class is requant policy — envelope row 11, owned by Ch6 `02`"; the full policy sentence appears twice, both in `05` (row 11, the registry, and the worked-example-1 quotation, the pedagogical payload) — exactly the two keepers pass 1 designated.
6. **2051 selection arithmetic in 02 vs 03 decoder ring** — RESOLVED. `03`'s MLA+DSA entry is now "compressor gate, pooled keys; selection capped at 2051 selected positions (see `02`'s list variants)" — no top-512/2048/tail arithmetic and no inventory citation; the full arithmetic with citation lives once, in `02`'s "Two step-list variants".

## MINOR Suggestions

1. **[01-mental-model.md + 02-anatomy-of-a-token.md] new pointer duplication from fix 2** — the clause "the exact boundary is Ch2 `05`'s elimination argument" now appears in BOTH `01`'s "What the host still does" AND `02`'s step 0 ("the exact boundary is Ch2 `05`'s elimination argument; the builder that produced the lists is Ch6 `04`'s"). `02` step 0 is where the reader walks the edge, so it should keep the boundary pointer; `01`'s sentence can drop its copy ("the exact boundary is Ch2 `05`'s") since it already links `02`'s step 0, which carries it.
2. **[02-anatomy-of-a-token.md] Ch4-ownership pointer pileup from fix 4** — Ch4 ownership of the roofline decomposition is now asserted three times inside `02`: the scope note ("the full exposition ... is owned by Ch4"), the section heading ("Where the 6.05 ms goes — preview, owned by Ch4"), and the rate bullet ("the plan's DRAM-busy constraint and the full window-rate exposition are Ch4 `01`'s"). Two of the three suffice; the heading's "preview, owned by Ch4" and the scope note say the same thing.
3. **[01-mental-model.md] "6.05 ms" twice in two consecutive sentences** — "Why it pays" states 6.05 ms/token in the headline sentence and again in "the roofline decomposition of the 6.05 ms is Ch4's". Trivial, but the second occurrence could be "the roofline decomposition is Ch4's".
4. **[02-anatomy-of-a-token.md] residual row-7 restatement (pre-existing, not a recurrence)** — `02`'s bytes bullet still carries "2.10 GB per chip per token — 16.8 GB across the box ... [MEAS, README.md "Roofline"]", which is envelope row 7 verbatim with its citation. Pass 1's item 4 covered rows 5-6 only, so this is not a recurrence; but the same registry logic suggests the same bare-number-plus-pointer treatment (`05` row 7 owns the citation) would close it. Low priority — the number is on the walk it budgets.

## Load-Bearing Evidence

- `index.md` line 4: "the host compiles model, weights, and topology into flat step-record lists once, and one persistent BRISC interpreter per Tensix core executes those records as data, token after token, with no per-step dispatch and no host in the token loop" — cannot be cut: it is the chapter's single-sentence thesis; every later chapter's opening premise and the whole inversion narrative hang off it, and it is the only place the full inversion is stated as one citable sentence for readers who jump straight to Ch2-Ch8.
- `01-mental-model.md` line 16: "one command can cover a whole multi-position row because `glm_steps` iterates its `tokens` field on the device [FACT, src/brisc.cpp:21182-21190]" — cannot be cut: it is the sole citation under the no-per-step-dispatch headline claim; removing it leaves the inversion's core assertion uncited anywhere in the chapter (the contrast table carries no source anchors).
- `02-anatomy-of-a-token.md` lines 40-46: the sync-chain per-variant table (561/693 steps, 194/216 mesh syncs, 429/473 PCIe rounds) with its per-variant warning preamble — cannot be cut: the only side-by-side statement of both list variants' counts in the guide's opening chapter; without it a count quoted for one variant would silently misdescribe the other, and no later Ch1 file restates them.
- `03-workload-context.md` lines 50-62: the per-class byte-budget table (KDA 4.972 GB, routed 4.756 GB, ..., device total 13.61 GB) — cannot be cut: it is the denominator of the entire DRAM-bound argument; Ch4's roofline motivation, the 3.53/4.27 ms floors, and the 65%-non-routed inversion all derive from these rows, and they appear nowhere else.
- `04-source-map.md` lines 9-31: the source-file map with verified `wc -l` counts and per-file responsibilities — cannot be cut: it operationalizes "locate by anchor, never by scrolling" for a 23,665-line `brisc.cpp`; deleting any row orphans every citation this chapter and later chapters make into that file.
- `05-measured-envelope-and-evidence.md` lines 50-53: worked example 1's WRONG/right citation pair for the hot-expert claim — cannot be cut: the one-tag-per-claim rule and the rows 10/11 split are inert without this concrete failure shown; it is the canonical enforcement illustration the whole guide's citation discipline leans on.

## VERDICT
- Crucial updates: no
