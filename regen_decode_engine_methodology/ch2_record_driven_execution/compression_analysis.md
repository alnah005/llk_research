# Compression Analysis: Chapter 2 — The Record-Driven Execution Model — Pass 1

## Summary
- Total files analyzed: 6
- Estimated current line count: ~530 lines
- Estimated post-compression line count: ~475 lines
- Estimated reduction: ~10%

## CRUCIAL Suggestions

### [01-step-records.md + 05-eliminating-the-runtime.md] ~lines 90 (01, final section) / ~10-15 (05, Claim (a))
**Issue:** The barrier-skip proof — `glm_command_meets` / `glm_skip_barrier`, the identical mechanism description, the -64/-71 and -33 us/token measurements, and the aphorism "A boundary you can delete with a data edit was never a boundary" — is stated in full twice: once at the end of 01's "Why records-as-data beats code-as-program" and again, nearly verbatim, in 05's Claim (a). 01 even flags the duplication ("The same numbers return as claim (a)'s evidence in 05") yet restates both the mechanism and the numbers anyway. Ch1 also already covers records-as-data as a design principle.
**Suggestion:** Make 05 Claim (a) the single owner of the measured barrier-skip proof (it is the evidence for the elimination thesis). Reduce 01's closing paragraph to one sentence plus a pointer: "the builder tunes synchronization through record data alone — see 05 Claim (a) for the mechanism and its measured -64/-71 / -33 us/token wins." Saves ~6-8 lines and removes a true cross-file duplicate.

### [02-interpreter-loop.md] ~lines 94-105 ("Contrast with a launch-per-kernel model")
**Issue:** A 7-row launch-per-kernel vs interpreter contrast table that overlaps ~5 rows with 05's "Cost-structure contrast" table (residency, per-step cost, host role, trace, failure attribution) — and Ch1 `01-mental-model.md` already owns a side-by-side contrast table covering the same rows. This is the third statement of the same comparison within the guide and the second within this chapter. 02 even ends by deferring the elimination argument to 05 while simultaneously restating its inputs (0.34 us, ~0.45 ms, 561 steps).
**Suggestion:** Delete 02's table and its closing paragraph entirely; replace with one sentence: "the per-step cost terms this loop produces (0.34 us fetch/copy, ~0.45 ms bookkeeping per token) feed 05's elimination argument, which contrasts them with a dispatch-per-step engine (and see Ch1 01's side-by-side)." Saves ~14 lines.

### [05-eliminating-the-runtime.md] ~lines 18-29 (Claim (b) prose) and ~lines 68-79 (Cost-structure table)
**Issue:** Within 05 itself the dispatch-per-step contrast is made twice: a prose paragraph ("Contrast with the dispatch-per-step engine the audience knows...") and, ~50 lines later, a full 7-row table with the same content (host round trip, launch, allocator, 561-or-693-times, 0.34 us copy). The 0.34 us / ~0.45 ms pair also appears a third time here after 02's table and closing paragraph.
**Suggestion:** Keep the table (the structured home of the contrast), compress the Claim (b) prose paragraph to its one unique sentence ("Dispatch did not become free — it became device-side bookkeeping") and drop the repeated cost enumeration from the prose. Saves ~4-5 lines and removes the intra-file double contrast.

### [02-interpreter-loop.md] ~lines 43-49 (per-step chain, item 6) vs [03-lead-helper-teams.md] ~lines 59-65 ("Step inputs")
**Issue:** The step-input multicast mechanism is explained in full twice with overlapping anchors (`src/brisc.cpp:21077-21103`): 02's chain item 6 describes the chip-root multicast with the sequence flag "cast behind the input on the same NoC and VC", and 03's "Step inputs" section explains the identical mechanism at owner-level detail. 02's scope note defers team mechanics to 03, so the full mechanism in 02 contradicts its own fence.
**Suggestion:** In 02 item 6, keep one clause ("optionally broadcast the step input to every helper — mechanism is 03's") and drop the NoC/VC ordering detail, which 03 owns. Saves ~2-3 lines and one duplicated explanation.

### [02-interpreter-loop.md] ~lines 22-27 (after the 4-step command loop) vs [03-lead-helper-teams.md] ~lines 72-80 ("The sentinel")
**Issue:** Sentinel content is duplicated despite 02 explicitly deferring "the sentinel's full stop-path discipline" to 03: 02 states the shard-0 `TT_SENTINEL_TOKEN` (0x544F4B4E) completion marker, the host poll, "the startup path itself writes none", and the load-time zeroing [FACT, src/tt_backend.cpp:5043-5046]; 03's sentinel section repeats the token value, the shard-0 completion fact, and the identical zeroing fact under the identical anchor.
**Suggestion:** 02 needs only the completion marker (its loop's close) and a pointer; move the zeroing-at-load fact and the "no startup write" observation to appear once, in 03. Saves ~3-4 lines of restated facts and one duplicated citation.

### [04-read-ahead-rings.md] ~lines 26-27 (Geometry) vs [01-step-records.md] ~lines 72-78 ("One flat list per core")
**Issue:** The list-replication mechanism ("the host mirrored the identical list into every DRAM channel of every chip at placement time, per-chip words adjusted", anchor `src/tt_backend.cpp:24368-24382`) is explained in both 01 and 04; 05 Claim (c) then cites the same anchor a third time. 01's "One flat list per core, replicated per channel" section is the declared owner of replication; 04 re-derives it to motivate ring locality.
**Suggestion:** 04's Geometry paragraph should carry only its ring-locality conclusion and point to 01 for the replication contract ("each lead reads its own channel's copy of the replicated list — 01's placement contract"). 05 Claim (c) needs only the once-per-configuration fact, not the mechanism. Saves ~4 lines.

### [03-lead-helper-teams.md] ~lines 68-69 ("Chip-wide jobs") vs [01-step-records.md] ~lines 62-66 ("Sync classes")
**Issue:** 03 re-quotes the identical `constants.h:2412` sentence ("the op returns with the chip's leads met, as glm_chip_job leaves them") and re-explains why `TT_GLM_SYNC_MET` exists — the exact explanation 01's "Sync classes" section already gives as the format's owner. The two-value sync enum is restated four times across the chapter (01, 02 chain item 5, 03, 05 Claim (a)); 01 and 05's single restatement as evidence is defensible, but 03's full re-quote is not.
**Suggestion:** In 03, state the causal link in one clause ("chip-wide jobs are why a record can declare `TT_GLM_SYNC_MET` — 01's sync-class section") without re-quoting the constants.h sentence. Saves ~2 lines and one duplicate quotation.

### [index.md] ~lines 36-40 ("Evidence conventions")
**Issue:** The evidence-tag discipline (one tag per claim, the four tag types, worked examples in Ch1 05) is restated from Ch1's index "Evidence conventions" section, which already establishes it as a guide-wide rule. Only the Ch2-specific caveat (notes-silent on team mechanics → strict [INFER]) is new here.
**Suggestion:** Compress to one sentence: "Evidence tags per [Ch1 05]; where the corpus is silent — notably team mechanics — this chapter tags strictly [INFER, src/...] and never presents code reading as measurement." Saves ~3 lines.

## MINOR Suggestions

### [04-read-ahead-rings.md] ~lines 3-8 (Scope) and ~lines 74-76 ("The same family, one pointer")
**Issue:** The Ch4 fence list (strip claims-ahead, two-strips-in-flight, NoC-ID budget, window rates) is enumerated three times in this one file: the opening paragraph, the Scope paragraph, and the closing family section ("Strip claims-ahead, two-strips-in-flight, and window rates — ... belong to Ch4, not here"). The index's pinned-boundary section also enumerates it.
**Suggestion:** Keep the enumeration once (Scope), reduce the opening paragraph's restatement to "the strip-level generalization is Ch4's, deferred by name throughout," and drop the closing repetition — the family-section sentence can end at the ERISC/trace-ring disambiguation.

### [04-read-ahead-rings.md] ~lines 74-76 ("The same family, one pointer")
**Issue:** The section re-explains the `TtTeamJob.next` hook and the record's `prefetch` descriptor mechanism that 03 ("The job object") and 02 (chain item 8) already explain, before doing its actual synthesis work.
**Suggestion:** Lead with the synthesis ("three prefetch mechanisms share the claim-ahead/in-flight/horizon-drain shape") and cite 03/02 for the members rather than restating their mechanics.

### [02-interpreter-loop.md] ~lines 3 (abstract) and ~38 (per-step chain close)
**Issue:** The "no queue, no callback, no completion interrupt, no scheduler, no return to any loader" enumeration appears at the chain close and is pre-echoed by the abstract's "nothing launched and no host in between"; similar no-X enumerations ("no debugger, no capture pass", "no init pass, no 'forward phase' marker") recur across 02 and 05. Two occurrences of the full four-way negation in one file is one more than needed.
**Suggestion:** Keep the fullest enumeration once (the chain close, where it is earned); in the abstract keep only "one resident program per core."

### [index.md] ~lines 32-35 ("Files in this chapter" table)
**Issue:** Each row's description (40-60 words) substantially restates the linked file's opening abstract sentence, so a reader gets the same summary twice within one click.
**Suggestion:** Trim table rows to ~15 words each; the files' own abstracts carry the detail.

### [01-step-records.md] ~lines 58-60 (command-id space, dense paragraph)
**Issue:** The paragraph is a single 200+-word run-on enumerating every id block with embedded parentheticals, one of which ("can take its token from the chips' own argmax candidates") duplicates a fact 05 Claim (c) states at owner level.
**Suggestion:** Break into a short list keyed by block range and drop the parenthetical semantics that 05 and Ch1 own.

### [02-interpreter-loop.md / 03-lead-helper-teams.md / 05-eliminating-the-runtime.md] ~contrast-table disclaimers
**Issue:** The disclaimer "the conventional/left column is the assumed tt-metal background, not a claim about regen" appears in 02, 03 (header phrasing), and 05, and already in Ch1 01's table footnote — repeated boilerplate for a rule the guide can state once.
**Suggestion:** State the assumption once in the chapter index (or rely on Ch1 01's footnote) and drop the per-table sentences; keep only 05's, where the contrast is load-bearing.

### [05-eliminating-the-runtime.md] ~line 15 (Claim (a), between-tokens bullet)
**Issue:** Restates the `for (uint32_t t = 0; t < tokens; ++t) glm_step_list(...)` code line and its no-host-in-between explanation verbatim from 02's "GLM_STEPS" section (and Ch1 anatomy stage 6 quotes it a third time guide-wide).
**Suggestion:** Replace the inline code with "(the device-side token loop, 02's `glm_steps`)" — the code block has one owner.

### [05-eliminating-the-runtime.md] ~line 20 (Claim (b)) and index.md claim 2
**Issue:** The 0.34 us figure is quoted in index claim 2, 02's table, 02's closing prose, 04's "Does fetch ever gate compute?", and 05's table. As a headline number some repetition is legitimate, but 02's closing paragraph re-derives the ~0.45 ms sum that 05 Claim (b)'s table already prices.
**Suggestion:** Keep 02's table row and the index headline; cut the closing-paragraph re-derivation (it duplicates 05) and keep only the pointer sentence that already ends it.

## Load-Bearing Evidence
- `index.md` line ~12: "A launch, post, or host round-trip inside one record's execution" — load-bearing because the falsifier column is the chapter's unique testability contract; no other file states what would refute claim 2, and compression must not touch it.
- `01-step-records.md` line ~32: "Records do not carry pointers; they carry *word indices*" — load-bearing because the arg-layer inversion is the format's central novelty, stated nowhere else in the guide at this precision.
- `02-interpreter-loop.md` lines ~40-49: the nine-step `glm_step` chain — load-bearing because it is the chapter's mechanism core; every other file references it by name, so it must stay complete and ordered.
- `03-lead-helper-teams.md` line ~66: "A decode prefetch job counts as done once this helper holds its args" — load-bearing because this early-arrive refinement is the single quoted fact carrying the prefetch-overlap design and appears nowhere else.
- `04-read-ahead-rings.md` line ~51: "Before posting a burst, land or move to L1 every small read the steps up to its consumer need" — load-bearing because it is the measured learned rule that justifies the window restart's existence; the burst-protection rationale is unique to this file.
- `05-eliminating-the-runtime.md` line ~44: "`TT_COMMAND_GLM_HOST_SYNC` step — 'where the host joins a post'" — load-bearing because the GLM_HOST_SYNC exception is the precise boundary of Claim (c) and the only mid-post host mechanism in the chapter; it must survive any compression of the claim's surroundings.

## VERDICT
- Crucial updates: yes

## Change Log — Agent A, Pass 1

CRUCIAL items:
1. **01/05 barrier-skip proof → single owner 05(a):** 01's closing reduced to one sentence + pointer to 05 Claim (a); mechanism, the −64/−71/−33 measurements, and the aphorism now live only in 05.
2. **02 contrast table deleted:** 02's "Contrast with a launch-per-kernel model" table and closing paragraph removed; replaced by a one-sentence pointer to 05's elimination argument and Ch1 01's side-by-side table (verified `ch1_engine_at_a_glance/01-mental-model.md` exists for the link).
3. **05 Claim (b) prose compressed:** dropped the prose contrast paragraph and the 561/693-times re-enumeration; kept only the unique "Dispatch did not become free — it became device-side bookkeeping" sentence; table kept as the contrast's structured home.
4. **02 chain item 6 → clause + pointer:** step-input multicast item now one clause deferring to 03; NoC/VC ordering detail and the `21077-21103` anchor left to 03's "Step inputs" section.
5. **Sentinel facts → single owner 03:** 02 keeps only the shard-0 completion marker (step 4), the host poll, and a pointer enumerating what 03 owns; load-time zeroing (`5043-5046`) and no-startup-write now appear only in 03 (both already existed there).
6. **List replication → single owner 01:** 04's Geometry now states only its ring-locality conclusion and points to 01's replication contract; 05 Claim (c) keeps the once-per-configuration fact with a pointer; the third citation of `24368-24382` removed.
7. **03 constants.h re-quote dropped:** chip-wide-job paragraph keeps the causal link as one clause pointing to 01's sync-class section; the `constants.h:2412` quotation and its tag removed (01 owns them).
8. **index evidence conventions → one sentence:** compressed to a pointer to Ch1 05 plus only the Ch2-specific strict-[INFER]-on-silent-corpus caveat; the measured/inferred split clause dropped (Ch1 05's discipline covers it).

MINOR items:
1. **04 triple fence enumeration — applied:** abstract trimmed to "is Chapter 4's, deferred by name throughout"; Scope keeps the one enumeration; the closing family-section repetition dropped. No line savings (single-line paragraphs), but the triple statement is gone.
2. **04 family section mechanics — applied:** section now leads with the shared shape and cites 03/02 for the members; kept the unique "neither the next step's args nor the list's result overwrite" quote with its `2429-2432` anchor since it appears nowhere else.
3. **02 no-X enumeration — applied:** abstract keeps "one resident program per core"; the full four-way negation stays only at the chain close where it is earned.
4. **index files-table trim — applied:** rows cut to ~10 words each; the files' own abstracts carry the detail.
5. **01 command-id paragraph → list — applied:** broken into a 3-bullet list keyed by id range; the BATCH_EMBED argmax parenthetical dropped (05 Claim (c) owns it with the `3469-3478` anchor).
6. **Contrast-table disclaimers — partially applied:** 02's disclaimer deleted along with its table; 05's kept (load-bearing there); 03's "(assumed background)" kept because it is a two-word table header, not a restated sentence, and Ch1 01's footnote already states the rule.
7. **05 between-tokens code quote — applied:** inline `for (...) glm_step_list(...)` replaced by a pointer to 02's `glm_steps` section; anchor `21189-21190` survives in 02 (owner) and 05's cost table.
8. **0.34 us / ~0.45 ms re-derivation — applied (subsumed):** 02's closing re-derivation deleted with CRUCIAL 2 and 05's prose re-derivation with CRUCIAL 3; the headline 0.34 us keeps its MEAS citation in 05's table and 04, and the index claim-2 row.

Note on line counts: this guide writes one paragraph per physical line, so prose/word compressions do not reduce `wc -l`; line savings come only from structural deletions (02's table, 05's paragraph). Chapter total went ~532 → 525 lines; all eight CRUCIAL deduplications are fully applied.

# Compression Analysis: Chapter 2 — The Record-Driven Execution Model — Pass 2

*Re-check pass, 2026-11-06. Verifies only the 8 CRUCIAL items from pass 1 against the post-fix files; pass-1 findings and the Agent A change log above are preserved unchanged.*

## Summary

- Files re-read: 6 (index, 01–05), plus the pass-1 analysis + change log.
- Real line counts (`wc -l`, content files only): index.md 42, 01-step-records.md 100, 02-interpreter-loop.md 99, 03-lead-helper-teams.md 113, 04-read-ahead-rings.md 83, 05-eliminating-the-runtime.md 88 — chapter total **525 lines**, matching the change log's claim (~532 → 525).
- All 8 CRUCIAL items verified resolved; no fix introduced new significant redundancy.

## CRUCIAL

None recurred. Per-item verification:

1. **01/05 barrier-skip proof → single owner 05(a): RESOLVED.** 01's closing is now one sentence + pointer ("tunes synchronization through record data alone ... see 05 Claim (a) for the mechanism (`glm_command_meets` / `glm_skip_barrier`) and its measured −64/−71 and −33 us/token wins") — exactly the pointer form pass 1 prescribed. The mechanism description, the aphorism, and the proof live only in 05 Claim (a). The one-sentence pointer naming the mechanism and headline numbers is the sanctioned form, not a restatement.
2. **02 contrast table → pointer: RESOLVED.** 02's "Contrast with a launch-per-kernel model" table and closing re-derivation are gone; "Where the per-step cost points" is a single sentence handing 0.34 us / ~0.45 ms to 05's elimination argument and naming Ch1 01 as the side-by-side owner. No third contrast remains in the chapter.
3. **05 Claim (b) intra-file double contrast: RESOLVED.** The prose contrast paragraph and the 561/693-times re-enumeration are gone; Claim (b) keeps the table (the structured home) and its one unique sentence ("Dispatch did not become free — it became *device-side bookkeeping*"). The 0.34 us / ~0.45 ms pair now appears once in 05 (table) plus the pointer sentence in 02 — no re-derivation.
4. **02 chain item 6 multicast duplication: RESOLVED.** Item 6 is now one clause + pointer to 03; the NoC/VC ordering detail and the `brisc.cpp:21077-21103` anchor appear only in 03's "Step inputs" section (grep-verified: the anchor survives in exactly one content file). 02 retains only a call-site anchor (`21133-21136`), which is the chain's own citation, not the mechanism's.
5. **Sentinel facts → single owner 03: RESOLVED.** 02 keeps the shard-0 completion marker at its loop's close, the host poll, and a pointer enumerating what 03 owns ("the helper-to-lead post, load-time zeroing, and the absence of any startup write"); the `tt_backend.cpp:5043-5046` zeroing fact and the no-startup-write observation now appear only in 03. The token value in 02's loop close is the marker pass 1 explicitly allowed to stay.
6. **List replication → single owner 01: RESOLVED.** `tt_backend.cpp:24368-24382` is cited in exactly one content file (01's "One flat list per core, replicated per channel"); 04's Geometry states only its ring-locality conclusion and points to "01's replication contract"; 05 Claim (c) keeps the once-per-configuration fact with a pointer to 01's contract. Grep-verified.
7. **03 constants.h re-quote dropped: RESOLVED.** 03's chip-wide-jobs paragraph states the causal link as one clause ("That is precisely why a record can declare `TT_GLM_SYNC_MET` ... (01's sync-class section)") without re-quoting `constants.h:2412`. The two-value enum now appears at owner level in 01, as chain step 5's validation in 02, and once as evidence in 05(a) — the distribution pass 1 deemed defensible.
8. **index evidence conventions → one sentence: RESOLVED.** The section is now a single sentence: pointer to Ch1 05 plus only the Ch2-specific strict-[INFER]-on-silent-corpus caveat. The tag-type enumeration and worked-example pointers are gone.

New-redundancy check on the fixes themselves: the pointer sentences in 01/02/03/04/05 name what their target owns without restating it; no fix duplicated previously-unique content, and no previously-unique line was lost (all five pass-1 load-bearing lines verified present).

## MINOR Suggestions

1. **[01 / 03 / 04] `constants.h:2385` "from its own DRAM channel (the lead's index)" is quoted verbatim in three files:** 01 line ~79 (owner, "One flat list per core"), 03 line ~9 (Roles), and 04 line ~5 ("Why a ring", with the extended `2385-2386` anchor). 04's re-quote motivates the ring; a bare "[FACT, src/constants.h:2385] — 01's replication sentence" would serve. Small, but it is now the most-repeated quotation in the chapter.
2. **[02 item 5 vs 01 Sync classes]** Chain item 5 restates the two-value enum and the `0xBAD90000 | index` stop code that 01's "Sync classes" owns (same anchor `21130-21131` in both). As a chain step it must exist, but it could be one clause + pointer ("Validate sync against 01's two classes; anything else stops the lead") instead of re-enumerating both values.
3. **[03 vs 05] `brisc.cpp:3337-3348` "one posted word per helper" appears in two contrast tables:** 03's "Why a fixed team replaces per-op grids" row and 05's cost-structure "Grid formation" cell carry the same fact and anchor. 05's cell could cite 03's table ("load-time teams; 03's table prices the post") since 03 is the job protocol's owner.

## Load-Bearing Evidence

- `index.md` line ~16: "A launch, post, or host round-trip inside one record's execution" — the claim-2 falsifier; the chapter's unique testability contract, stated nowhere else, and the reason "no per-step dispatch" is checkable rather than rhetorical. Cannot be cut.
- `01-step-records.md` line ~32: "Records do not carry pointers; they carry *word indices*" — the arg-layer inversion is the format's central novelty; no other file (or chapter) states it at this precision, and every downstream addressing claim depends on it.
- `02-interpreter-loop.md` line ~50: "there is no queue, no callback, no completion interrupt, no scheduler, and no return to any loader between steps" — the earned close of the nine-step `glm_step` chain, the mechanism core every other file references by name; the chain must stay complete and ordered, and this sentence is its structural summary.
- `03-lead-helper-teams.md` line ~53: "A decode prefetch job counts as done once this helper holds its args" — the early-arrive refinement is the single quoted fact carrying the prefetch-overlap design; remove it and the resident-prefetch claims in 02 item 8 and 04's family section lose their justification.
- `04-read-ahead-rings.md` line ~52: "Before posting a burst, land or move to L1 every small read the steps up to its consumer need" — the measured learned rule that alone justifies the window restart's existence (the +23 us counterfactual hangs off it); unique to this file.
- `05-eliminating-the-runtime.md` line ~44: "the list carries a `TT_COMMAND_GLM_HOST_SYNC` step — 'where the host joins a post'" — the precise boundary of Claim (c) and the chapter's only mid-post host mechanism; without it "no host in the token loop" would be stated too strongly and Ch8's invariant list would lose its one sanctioned exception.

## VERDICT

- Crucial updates: no
