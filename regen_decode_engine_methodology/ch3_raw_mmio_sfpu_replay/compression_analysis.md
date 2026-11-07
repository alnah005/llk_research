# Compression Analysis: Chapter 3 — Raw-MMIO Compute and SFPU Replay — Pass 1

## Summary
- Total files analyzed: 7
- Estimated current line count: ~515 lines
- Estimated post-compression line count: ~490 lines
- Estimated reduction: ~5%

The chapter is already dense, evidence-tagged, and fenced; most lines carry citations or anchored quotes. The compressible material is concentrated in three patterns: (a) a verbatim restatement of Ch1's window-terminology table in the chapter index, which the guide's own rules declare out of bounds; (b) the SFPU composition/boundary-fact material re-derived in both `03` and `05`; (c) the SwiGLU-inline fact and the replay issue-economics argument each stated 3–4 times across `03` and `04`.

## CRUCIAL Suggestions

### [index.md] ~lines 19–29
**Issue:** The "Window terminology, pinned" table restates Ch1's glossary verbatim. Three of its four rows — SFPU replay window, DRAM prefetch window, PCIe BAR window — are word-for-word copies of the corresponding rows in [Ch1's glossary window table](`../ch1_engine_at_a_glance/index.md`), including identical "What it is" text. Ch1 declares terms are "defined once here, bolded and linked on first use per chapter"; the briefing explicitly constrains that Ch1/Ch2 material must not be restated. Only the fourth row (the record `window` flag of `TtGlmStepRecord`, deferred to Ch2) is new relative to Ch1's table.
**Suggestion:** Replace the 3 duplicated rows with one sentence: "The guide's three pinned window senses (SFPU replay, DRAM prefetch, PCIe BAR) are defined once in [Ch1's glossary](../ch1_engine_at_a_glance/index.md); this chapter owns the first." Keep only the record-`window`-flag row, since it is the one sense Ch1's table does not cover.

### [05-correctness-discipline.md] ~lines 10–19 (Layer-1 pin table) vs [03-sfpu-replay-windows.md]
**Issue:** Duplicate explanation across files. 05's "Fragment range pins" and "Window count pins" rows restate, with the same anchors, the compiler-guarantee bullets 03 already establishes in "The compiler: what it guarantees" (template < 8192, AddrMod < 4, offset 0..8191; 1..32 replay bodies; `<name>_instruction_count`). Worse, 05's "Composition pins" cell re-derives 03's boundary facts nearly verbatim ("`sfpu_glm_moe_combine_body` sits outside this block and `sfpu_glm_hc_expand_body` carries its own local pin") — a fact 03 spends a full paragraph establishing at lines ~47 and owns by fence.
**Suggestion:** Compress 05's first three table rows to citations: "Range pins, count pins, and composition pins — mechanisms and boundary facts established in [`03`](./03-sfpu-replay-windows.md) — enforced here as build failures." Keep only what is new to 05: the causal claim paragraph below the table (which is 05's own argument) and the layout-pins row (05's only unique row).

### [03-sfpu-replay-windows.md] ~line 51 and [04-op-bodies.md] ~lines 34, 78
**Issue:** The "GLM SwiGLU runs inline, no REPLAY, ±10 clamps" fact is stated four times: twice inside 03's borrower-discipline paragraph (line 51 states it, then immediately re-derives it with "So replay windows serve the row-iterating recurrences, while per-block activations like GLM SwiGLU stay inline"), once in 04's shared conventions (line 34, with the clamp constants and restore), and again in 04's remaining-families SwiGLU line (line 78, same clamps, same "no replay"). 03 line 83 then states the ±10 override a third time in the exp-family section.
**Suggestion:** 04 owns the op-level statement — keep line 78 (the op catalog entry) and fold line 34's clamp detail into it (or vice versa; one of the two in 04 must go). In 03, keep the borrower-paragraph contrast but delete the trailing "So replay windows serve..." sentence (it repeats the sentence directly above it) and drop the GLM-SwiGLU clamp override sentence at line 83 in favor of "GLM decode's SwiGLU installs its own ±10 clamps ([`04`](./04-op-bodies.md)'s catalog)."

### [04-op-bodies.md] ~line 73
**Issue:** Repeated explanation. The replay issue-economics argument ("one REPLAY push replaces several instruction pushes per row; row-iterating recurrences are the shape that rewards it") is fully stated in 03's intro (line 3, as the *reason replay exists*) and then re-derived as a full paragraph in 04's KDA example ("Why this op above all rewards replay: the recurrence is 128 (or 64) repetitions... exactly the shape where one REPLAY push replaces several instruction pushes per row") with overlapping INFER anchors. 02's "densest sustained port-traffic pattern" paragraph (line 46) adds the third telling of the same kda_rows immediate-stream story.
**Suggestion:** Keep 03's one-sentence economics statement (it is the mechanism owner) and 04's record-level walk. Cut 04's paragraph to one sentence: "This is the op 03's replay economics was built for: 128 (or 64) iterations of one count-pinned body with per-row scalar operands [INFER, src/brisc.cpp:5768–5800]." Keep 02's port-traffic view (it is the port-traffic owner) but trim 04's predict bullet (line 69) to "operands via `kda_rows`' runtime pushes ([`02`](./02-instruction-port.md)'s densest pattern), one REPLAY per row."

## MINOR Suggestions

### [03-sfpu-replay-windows.md] ~lines 38–45
**Issue:** The 6-line `static_assert` code block restates in full what the surrounding prose then re-describes sentence by sentence ("plus one further static_assert whose four && conditions cap...").
**Suggestion:** Quote one representative assert + a 5-line ellipsis comment; the prose already carries the enumeration. Saves ~4 lines without losing the pin idiom.

### [01-mmio-register-layer.md] ~line 45
**Issue:** The closing sentence of the cfg-word section ("This is exactly the register set a ckernel's `init` functions program in a conventional stack — here it is engine code...") restates the file intro's framing sentence ("doing, by hand, the work llk/ckernel init code does in a conventional stack") — same fact, same file, ~40 lines apart.
**Suggestion:** Keep the intro framing; end the line-45 sentence at the citation, dropping the re-comparison.

### [index.md] ~lines 3 and 30–40
**Issue:** The chapter opening paragraph and the "Files in this chapter" table describe the same content twice within the same file — e.g. 05's probes/smoke-gate are enumerated in the opening paragraph ("instruction-count static_asserts, startup probes like the SrcA bank test, the smoke gate") and again in its table row ("count pins, deterministic sizing, startup probes, the smoke gate, and the oracle's role").
**Suggestion:** Trim the opening paragraph to name the mechanism chain once and let the table carry the per-file descriptions.

### [index.md] ~lines 41–46
**Issue:** The "Evidence conventions" section re-explains Ch1's tag rules ("exactly one tag per claim, no hybrids, comparative/superlative readings never tagged [FACT]") that Ch1's own evidence-conventions section states once. Only the chapter-specific [MEAS] inventory (smoke yield, probe back-story, diagnostic-cost figure) is new.
**Suggestion:** Keep one pointer sentence to Ch1's discipline plus the three-item [MEAS] inventory; drop the re-stated rules.

### [06-what-no-llk-means.md] ~lines 11 and 34
**Issue:** The gives-up "CB flow control" bullet and the table's "Data movement & flow control" row carry the same fact with the same anchors ("plain L1 stores + unpack instructions; readers own all synchronization; no CBs", `src/brisc.cpp:218-227, 920-922`). The "readers own all synchronization" phrase itself now appears five times across 02 (twice), 04, 06, and the chapter index fence.
**Suggestion:** As the summary file, 06 is allowed a compact restatement — but one of the two CB statements inside 06 can go: keep the table row (Ch8 consumes it) and compress the bullet to its gives-up point ("no CBs anywhere; the producer/consumer safety a CB constructs must be re-created by op-body waiting discipline"). In 02, the phrase at line 52 is the mechanism owner and stays; the chapter-index fence occurrence is a one-liner and fine.

### [06-what-no-llk-means.md] ~lines 47–48
**Issue:** Cost-ledger item 1 ("Every guarantee is homemade. The net of 05... replaces guarantees a library would give") restates the file's own opening framing ("what regen gives up... the price of each line") and 05's thesis sentence.
**Suggestion:** Cut the first clause of item 1; the citation to 05 already carries the argument.

## Load-Bearing Evidence
- `05-correctness-discipline.md` lines ~43–46: the verbatim 28-name `TT_LAB_CHECK` smoke invocation — load-bearing because the text itself declares "the list is not glob-compressible, since a compressed form would silently change the check set the driver matches by literal name"; it must survive compression untouched.
- `05-correctness-discipline.md` line ~26: "wrong at every position, usually finite, +Inf at position 19" — load-bearing because it is the [MEAS] evidence for the entire probe family's existence; deleting it would leave the SrcA bank test unmotivated.
- `03-sfpu-replay-windows.md` line ~47: "the two layers pin different things (per body vs per shared buffer)" — load-bearing because this distinction is exactly what 05's causal argument depends on and the one piece of the composition-pin material 05 must cite rather than re-derive.
- `01-mmio_engine_at_a_glance` glossary table (Ch1 `index.md` lines ~50–56) — load-bearing because it is the guide's single declared definition site for the three window senses; compression must point there, not delete the pointer.
- `02-instruction-port.md` line ~28: the GPR63 handshake quotes ("GPR63 is reserved for completion", "FENCE alone does not drain a BRISC store") — load-bearing because they are the operational contract every op body's `wait_tensix` depends on.

## VERDICT
- Crucial updates: yes

## Change Log — Agent A, Pass 1

CRUCIAL:
1. [index.md] Replaced the window-terminology table's three Ch1-duplicated rows (SFPU replay / DRAM prefetch / PCIe BAR) with one pointer sentence to Ch1's glossary; kept only the record-`window`-flag row (the sense Ch1's table does not cover).
2. [05] Compressed the Layer-1 pin table's first three rows (range, count, composition) to one citation sentence pointing to 03 as the mechanism/boundary-fact owner; kept 05's causal claim paragraph and its unique layout-pins content (now prose, evidence tag retained).
3. [03 + 04] GLM-SwiGLU-inline fact now single-owned by 04's catalog entry: 04's shared-conventions Live-LReg bullet lost its SwiGLU example and the clamp detail was folded into the SwiGLU catalog line (with the `0x41200000`/`0xC1200000` constants); deleted 03's trailing "So replay windows serve..." re-derivation sentence; converted 03's exp-family ±10-override sentence to a pointer to 04's catalog.
4. [03 + 04] Replay issue-economics argument single-owned by 03's intro: 04's KDA "Why this op above all rewards replay" paragraph cut to one sentence citing 03; 04's predict bullet trimmed to "operands via `kda_rows`' runtime pushes ([`02`'s] densest pattern), one REPLAY per row"; 02's port-traffic paragraph untouched (port-traffic owner).

MINOR:
1. [03] Applied — `static_assert` block truncated to one representative assert + 5-line ellipsis comment (prose already enumerates the six bodies and the four-condition chunk assert).
2. [01] Applied — dropped the ckernel-`init` re-comparison from the cfg-word section's closing sentence; the intro framing remains the owner.
3. [index.md] Applied — trimmed the opening paragraph's parenthetical enumeration of the disciplines (static_asserts / SrcA bank test / smoke gate); the files table carries the per-file descriptions.
4. [index.md] Applied — Evidence-conventions section compressed to a pointer to Ch1's tag rules plus the three-item [MEAS] inventory; dropped the re-stated one-tag/no-hybrid/superlative rules.
5. [06] Applied — gives-up "CB flow control" bullet compressed to its gives-up point ("no CBs anywhere; producer/consumer safety must be re-created by op-body waiting discipline"), pointing at the table's data-movement row (kept for Ch8); table row and its FACT anchors untouched.
6. [06] Applied — cost-ledger item 1's redundant first clause ("Every guarantee is homemade") replaced with a catalog-growth heading; the 05 citation carries the argument.

Preserved as flagged: 05's verbatim 28-name `TT_LAB_CHECK` invocation (byte-identical), the SrcA "wrong at every position... +Inf at position 19" evidence, 03's per-body vs per-shared-buffer pin distinction, 02's GPR63 handshake quotes, 06's engine-vs-llk table, all footers, one H1 per file. Chapter total: 515 → 501 lines.

# Compression Analysis: Chapter 3 — Raw-MMIO Compute and SFPU Replay — Pass 2

Date: 2026-11-06. Re-check pass: Agent C re-read all 7 chapter files and verified only the 4 pass-1 CRUCIAL items. Content files were not modified.

## Summary

- Real line counts (wc -l): `index.md` 44, `01-mmio-register-layer.md` 74, `02-instruction-port.md` 81, `03-sfpu-replay-windows.md` 91, `04-op-bodies.md` 88, `05-correctness-discipline.md` 67, `06-what-no-llk-means.md` 56 — **chapter total 501 lines** (excluding this analysis file), matching the change log's claimed 515 → 501 reduction.
- All 4 pass-1 CRUCIAL items are genuinely resolved; the fixes match the suggested shapes closely and introduce no new significant redundancy.
- Net further compression available: minor only, a few words to a couple of lines.

## CRUCIAL — none recurred

Re-verification of each pass-1 CRUCIAL item:

1. **[index.md] window-terminology table — RESOLVED.** The three Ch1-duplicated rows are gone. The section now opens with the pointer sentence ("The guide's three pinned window senses (SFPU replay, DRAM prefetch, PCIe BAR) are defined once in Ch1's glossary; this chapter owns the first") and keeps only the record-`window`-flag row, correctly labeled as "the one sense Ch1's table does not cover." No new duplication introduced.
2. **[05] Layer-1 pin table vs 03 — RESOLVED.** The range/count/composition rows are compressed to a single citation sentence: mechanisms and boundary facts "established once in [`03-sfpu-replay-windows.md`](./03-sfpu-replay-windows.md), and are enforced here as build failures." The layout pins are kept as 05's own fourth family, and 05's causal claim paragraph (count change ⇒ composition-assert build failure) survives intact — it is 05's own argument, not a re-derivation of 03's. The composition-boundary re-derivation ("`sfpu_glm_moe_combine_body` sits outside this block...") is gone from 05.
3. **[03 + 04] GLM-SwiGLU-inline fact — RESOLVED.** Single ownership is established as suggested: 04's SwiGLU catalog entry (`105`) owns the op-level statement with the `0x41200000`/`0xC1200000` constants folded in; 04's shared-conventions Live-LReg bullet no longer carries the SwiGLU example; 03's trailing "So replay windows serve..." re-derivation sentence is deleted; 03's exp-family ±10 sentence is now a pointer to "04's SwiGLU catalog entry owns the constants." The fact survives exactly twice, as pass 1 prescribed: 03's borrower-paragraph contrast ("GLM decode does **not** borrow for SwiGLU") and 04's catalog owner.
4. **[03 + 04] replay issue-economics argument — RESOLVED.** 03's intro keeps the single economics statement ("Why replay exists is issue economics: one REPLAY push re-issues a whole recorded body per loop iteration... the row-iterating recurrences are exactly the shape that rewards this"). 04's KDA paragraph is cut to the one-sentence form ("This is the op 03's replay economics was built for: 128 (or 64) iterations of one count-pinned body with per-row scalar operands") and 04's predict bullet is trimmed to the pointer ("via `kda_rows`' runtime pushes ([`02`]'s densest pattern), one REPLAY per row"). 02's port-traffic paragraph is untouched, as prescribed.

New-redundancy check on the edited seams: the pointer/citation sentences added by the fixes are one-liners that name their owner, not re-statements; nothing new rises above minor.

## MINOR Suggestions

1. **[04-op-bodies.md] ~line 70–71 (same worked example).** The predict bullet ("one REPLAY per row") and the trailing economics sentence four lines below ("...one count-pinned body with per-row scalar operands") both connect KDA to replay within one worked example; the trailing sentence could drop "with per-row scalar operands" — the bullet above already carries it. Saves a few words; the fix's prescribed shape is otherwise correct as-is.
2. **[03-sfpu-replay-windows.md] census table "Qwen-image borrowers" row vs the borrower-discipline paragraph.** The table cell ("recorded into slot 0 'borrows the whole replay buffer' — after which the resident replays are re-recorded") restates the borrower paragraph's re-record mechanism in the same file. The table cell could stop at "exp(−|x|) and sigmoid/SiLU gate passes recorded into slot 0" and leave the re-record discipline to the paragraph that owns it. Saves ~1 line.
3. **[06-what-no-llk-means.md] three no-CB statements in one file.** After the pass-1 fix the no-CB fact still appears three times inside 06: the gives-up bullet ("No CBs anywhere"), the gains bullet ("No CB sync races — because no CBs"), and the table's data-movement row (kept for Ch8). The gains bullet's unique content is the locality claim ("its replacement hazard... is local to the code that owns the data"); its opening clause could compress to "No CB sync races (the gives-up bullet's flip side): the replacement hazard..." — one phrasing change, no line saved, but removes the third verbatim "no CBs" in a 56-line file.

## Load-Bearing Evidence

- `index.md` — the record-`window`-flag table row: the one "window" sense Ch1's glossary does not cover; deleting it would orphan the read-ahead-restart sense for every reader who hits `window` in a `TtGlmStepRecord` and looks here first.
- `01-mmio-register-layer.md` — the `hw_init` AddrMod sentence ("the fragments compile against a *fixed* AddrMod table, which is part of why window instruction counts are stable" [FACT, src/brisc.cpp:1201-1220]): the causal bridge that 03's census stability claim and 05's count-pin argument both rest on; it must stay in the file that owns per-op cfg sequencing.
- `02-instruction-port.md` — the GPR63 handshake quotes ("GPR63 is reserved for completion", "FENCE alone does not drain a BRISC store" [FACT, src/brisc.cpp:319, 323]): the operational contract every op body's `wait_tensix` depends on; carried from pass 1 and still intact.
- `03-sfpu-replay-windows.md` — the per-body vs per-shared-buffer pin distinction ("the two layers pin different things" [INFER, src/brisc.cpp:62-73]): exactly what 05's compressed citation must point at rather than re-derive; deleting it would strand 05's Layer-1 sentence. Also intact: the boundary-facts paragraph (`moe_combine` outside the 62-73 asserts, `hc_expand`'s local pin).
- `04-op-bodies.md` — the command-id map with `src/constants.h` anchors (KDA 64-79 / MLA 80-99 / mHC-MoE 100-119 / integration 120-127): the chapter's only frozen id index; 05's smoke-check-to-family mapping and every cross-file "command N" reference resolve through it.
- `05-correctness-discipline.md` — the verbatim 28-name `TT_LAB_CHECK` invocation (byte-identical, preserved as flagged): the text itself declares the list not glob-compressible; any alteration silently changes the check set. Also intact: the SrcA "wrong at every position, usually finite, +Inf at position 19" [MEAS] evidence.
- `06-what-no-llk-means.md` — the engine-vs-llk table plus its honesty note (llk column is audience-baseline [INFER], "no measured engine-vs-llk comparison", the 50 µs diagnostic-cost datum): the declared Ch8 input consumed verbatim; the honesty note is what keeps the table from overclaiming.

## VERDICT

- Crucial updates: no
