# Compression Analysis: Chapter 1 — Architecture and Mental Model — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1004 lines
- Estimated post-compression line count: ~830 lines
- Estimated reduction: ~17%

## CRUCIAL Suggestions

### [01_tensix_core_model.md] ~lines 278-313 (Section 1.1.5)
**Issue:** The Mcast sender/receiver code block (lines 288-313) is a near-verbatim duplicate of code already shown earlier in the same file. The NCRISC receiver snippet (lines 302-313) repeats lines 91-99 word-for-word, and the BRISC sender snippet (lines 290-299) substantially overlaps with lines 129-137. Section 1.1.5 repackages the same two code blocks from sections 1.1.2, adding only the framing sentence about semaphore signals.
**Suggestion:** In section 1.1.5, replace the full code blocks with back-references: "The sender/receiver pattern was shown in full in Section 1.1.2. The semaphore-specific addition is the `mcast_detail::send_with_state` call on the sender side, which multicasts the semaphore signal to all receivers." Show only the 2-line semaphore send call if needed, not the entire 25-line block. This saves ~25 lines and eliminates the jarring deja-vu of reading the same Mcast code a second time.

### [02_blaze_compilation_pipeline.md] ~lines 129-203 (Section 1.2.2)
**Issue:** Section 1.2.2 ("How `emit()` drives the engine pipeline") restates material already covered in Section 1.2.1 ("Phase 2: Engine pipeline"). The Mcast `emit()` code example (lines 134-175) demonstrates calls like `f.cb_from_tensor()`, `f.semaphore()`, `f.ncrisc_ct_args()`, and `f.brisc_ct_args()` -- all of which were already introduced and described in the Phase 2 subsection (lines 104-106). The summary table at lines 179-188 restates what each `emit()` call does, duplicating the explanation at lines 104-106 almost point-for-point. Then lines 191-203 show `CTArgEngine._resolve_value()`, which was already covered conceptually at lines 98-103.
**Suggestion:** Merge section 1.2.2 into the Phase 2 subsection of 1.2.1. Keep the Mcast `emit()` code example (it is useful as a concrete walkthrough), but remove the summary table at lines 179-188, as it merely re-explains the `FusedProgram` methods already introduced at lines 104-106. Remove the `_resolve_value()` snippet (lines 191-203), which adds little beyond the `CTArgEngine.generate_tuples()` call already shown. This saves ~30 lines.

### [03_micro_op_vs_fused_op.md] ~lines 93-114 (Section 1.3.1, "Auto-derivation from the kernel header")
**Issue:** The auto-derivation subsection (lines 98-114) repeats information already stated in the same file's earlier prose. Line 99 says MicroOps "auto-derives from the kernel header" and then enumerates the six things `_auto_derive_from_kernel_hpp()` does. But lines 108-114 immediately restate the conclusion ("This means that for most MicroOps, the Python class only needs to declare: name, port descriptors, emit(), compose(). Everything else is derived..."), which is the same claim already made at lines 17-22 of the same section (the "two artifacts" framing) and again in key takeaway #1 (line 477). Three statements of the same point in one file.
**Suggestion:** Keep the bulleted list of what `_auto_derive_from_kernel_hpp()` does (lines 100-107) -- it is specific and useful. Remove the restatement paragraph at lines 108-114. The reader has already absorbed the "two artifacts" framing and does not need a third restatement. Saves ~7 lines.

### [01_tensix_core_model.md + 02_blaze_compilation_pipeline.md + 03_micro_op_vs_fused_op.md] Cross-file: `ct_types.h` type alias explanation
**Issue:** The explanation that `CB`, `Semaphore`, `PerCore`, and `Flag` are type aliases for `uint32_t`/`bool` with zero runtime cost, used by the Python parser to classify CT args, appears three times:
  - `01_tensix_core_model.md` lines 265-275 (section 1.1.4): "These type aliases carry no runtime cost -- they are all uint32_t or bool at the C++ level. Their purpose is entirely for the Python parser..."
  - `02_blaze_compilation_pipeline.md` lines 250-255 (section 1.2.4): "Because all CT args are constexpr, the compiler can... Inline CB IDs, semaphore addresses, and tile counts as immediate constants."
  - `03_micro_op_vs_fused_op.md` lines 465-472 (section 1.3.7): "The typed aliases from blaze/kernels/ct_types.h (CB, Semaphore, PerCore, Flag) serve as documentation in the C++ header. They are all uint32_t or bool at the C++ level, but they tell the Python parser..."
  File 03 quotes the `ct_types.h` snippet conceptually even though file 01 already showed it verbatim.
**Suggestion:** Give the full explanation once -- in `01_tensix_core_model.md` where it naturally belongs (hardware model). In `02_blaze_compilation_pipeline.md` and `03_micro_op_vs_fused_op.md`, replace with a single-sentence cross-reference: "The typed aliases (`CB`, `Semaphore`, `PerCore`, `Flag`) and their role in Python parsing were described in Section 1.1.4." Saves ~15 lines across the two downstream files.

### [02_blaze_compilation_pipeline.md] ~lines 446-472 (Section 1.2.8)
**Issue:** Section 1.2.8 ("Putting it all together: the full flow") is a 6-step summary that restates the entire chapter's content in compressed form. But the chapter already has a "Key takeaways" section at lines 490-503, which provides a different compressed summary. The reader encounters two consecutive summaries, each covering the same material from slightly different angles. The 6-step flow chart (lines 449-472) overlaps heavily with the three-phase lifecycle diagram at lines 18-28.
**Suggestion:** Remove section 1.2.8 entirely. The three-phase lifecycle diagram at the top of the file (lines 18-28) plus the key takeaways at the end are sufficient. Alternatively, if the step-by-step numbered flow is preferred, remove the ASCII diagram at lines 18-28 and let section 1.2.8 serve as the single overview. Either way, eliminate one of the two summaries. Saves ~25 lines.

### [03_micro_op_vs_fused_op.md] ~lines 325-393 (Section 1.3.5, "The dual API surface")
**Issue:** The dual API surface section repeats the `_OpHandle` class definition that was already shown in `02_blaze_compilation_pipeline.md` lines 49-57 (section 1.2.1, Phase 1). Lines 329-339 of file 03 show a slightly abbreviated version of the same class. Additionally, the graph API usage example (lines 346-350) closely parallels the example at `02_blaze_compilation_pipeline.md` lines 38-44. The composition API example (lines 358-363) parallels `02_blaze_compilation_pipeline.md` lines 63-68. The "When to use which" table at lines 388-393 duplicates the table at `02_blaze_compilation_pipeline.md` lines 362-368.
**Suggestion:** In file 03, section 1.3.5, remove the `_OpHandle` code listing (already shown in file 02) and the duplicated usage examples. Replace with a single cross-reference sentence and focus on what is new in this section: the `compose()` bridge (lines 369-384), which is the only content in 1.3.5 not already covered in file 02. Move the "When to use which" table to file 02 only (it is about compilation modes, not about op types). Saves ~30 lines.

## MINOR Suggestions

### [index.md] ~lines 21-27 ("Reading order")
**Issue:** The "Reading order" section paraphrases the table at lines 15-19 with only marginal additional guidance. Both say "read 01 first, then 02, then 03" with slight elaboration.
**Suggestion:** Either remove the "Reading order" section (the table's sequential numbering implies reading order) or keep it and remove the table, using the reading-order prose as the single navigation element.

### [01_tensix_core_model.md] ~lines 452-464 ("Key takeaways")
**Issue:** Several takeaway bullets restate section headers almost verbatim. E.g., takeaway #5 ("Empty phases overlap for free") restates the section 1.1.8 title "Why empty processors enable overlap" with minimal added information beyond what the section already conveyed.
**Suggestion:** Tighten each takeaway to one sentence maximum. Remove explanatory subordinate clauses that restate section content. The takeaways should be a quick-reference checklist, not a mini-reread.

### [01_tensix_core_model.md] ~lines 15-16 (Section 1.1.1, opening paragraph)
**Issue:** "Every Tensix core on a Tenstorrent Blackhole chip contains **five RISC-V processor cores**. These five cores are organized into **three logical processors**..." -- the second sentence restates "five cores" from the first. Minor but easily tightened.
**Suggestion:** Combine into one sentence: "Every Tensix core on a Tenstorrent Blackhole chip contains five RISC-V processor cores, organized into three logical processors with distinct roles in the data-movement and compute pipeline."

### [01_tensix_core_model.md] ~lines 61-63 (Section 1.1.2, opening)
**Issue:** "Understanding which processor owns which responsibility is the single most important mental model for writing TT-Blaze ops." -- hedging/emphasis language. The reader is already reading this section; they do not need to be told it is "the single most important" thing.
**Suggestion:** Remove the sentence or change to a neutral lead-in: "Each processor has a distinct responsibility."

### [02_blaze_compilation_pipeline.md] ~lines 357-358
**Issue:** "Both modes ultimately produce the same artifact: a ProgramDescriptor with three kernel binaries, CB descriptors, and an ordered tensor list for ttnn.generic_op()." This same point is made at lines 383-384 of the same section: "Both APIs ultimately produce the same compiled kernel and program descriptor." (The latter is in file 03, but the same idea recurs within file 02 as well at line 358.)
**Suggestion:** State the convergence point once, at the end of the two-mode comparison. Remove the earlier occurrence.

### [03_micro_op_vs_fused_op.md] ~lines 396-441 (Section 1.3.6)
**Issue:** The "When to write a MicroOp vs. a FusedOp" section, while useful, has some verbose bullet points. E.g., "A MicroOp requires: 1. A Python class... 2. A C++ kernel header... 3. Registration of the Op struct..." (lines 405-408) restates what was already defined at lines 16-21 of the same file.
**Suggestion:** Replace the "A MicroOp requires" list with: "See Section 1.3.1 for the full MicroOp file layout." Similarly for the "A FusedOp requires" list -- reference Section 1.3.2.

### [03_micro_op_vs_fused_op.md] ~lines 475-488 ("Key takeaways")
**Issue:** Takeaway #4 ("Two APIs, one implementation") and takeaway #5 ("Start with FusedOp, drop to MicroOp when needed") are wordy restatements of section content. Takeaway #3 on the `name` field repeats the five-role enumeration already given at lines 307-315.
**Suggestion:** Trim each takeaway to one sentence. Remove the parenthetical lists that re-enumerate roles or methods.

## VERDICT
- Crucial updates: yes

## Change Log (Applied 2026-05-13)

All 6 CRUCIAL suggestions were applied:

1. **[01_tensix_core_model.md] Section 1.1.5 Mcast code dedup:** Replaced the near-verbatim 25-line sender/receiver code blocks with a back-reference to Section 1.1.2 and a 2-line semaphore send snippet.

2. **[02_blaze_compilation_pipeline.md] Section 1.2.2 emit() walkthrough merge:** Merged the Mcast `emit()` walkthrough into the Phase 2 subsection of Section 1.2.1. Removed the redundant summary table and the `_resolve_value()` snippet. Deleted the standalone Section 1.2.2 heading and renumbered subsequent sections (1.2.3-1.2.7 became 1.2.2-1.2.6).

3. **[03_micro_op_vs_fused_op.md] Section 1.3.1 triple restatement:** Removed the restatement paragraph (lines 108-114) about what MicroOps need to declare, keeping only the bulleted list of what `_auto_derive_from_kernel_hpp()` does.

4. **Cross-file ct_types.h dedup:** Kept the full explanation in `01_tensix_core_model.md` Section 1.1.4. In `03_micro_op_vs_fused_op.md` Section 1.3.7, replaced the 8-line type alias explanation with a single-sentence cross-reference to Section 1.1.4.

5. **[02_blaze_compilation_pipeline.md] Section 1.2.8 double summary:** Removed Section 1.2.8 ("Putting it all together") entirely. The three-phase lifecycle diagram at lines 18-28 and the key takeaways are sufficient.

6. **[03_micro_op_vs_fused_op.md] Section 1.3.5 dual API dedup:** Removed the `_OpHandle` code listing, duplicated graph/composition API usage examples, and "When to use which" table (all already present in file 02). Replaced with a cross-reference to Sections 1.2.1 and 1.2.5. Retained the `compose()` bridge content, which is unique to this section.

---

# Compression Analysis: Chapter 1 — Architecture and Mental Model — Pass 2

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1362 lines
- Estimated post-compression line count: ~1290 lines
- Estimated reduction: ~5%

## CRUCIAL Suggestions

### [01_tensix_core_model.md + 02_blaze_compilation_pipeline.md] Cross-file: duplicate `kernel_main()` generated code blocks
**Issue:** Two nearly identical auto-generated `kernel_main()` code blocks appear in consecutive files. File 01 (lines 309-331, Section 1.1.6) shows a 2-phase kernel (Mcast + Matmul). File 02 (lines 353-391, Section 1.2.6) shows a 3-phase kernel (ActMcast + Matmul + Rmsnorm) that is a strict superset of the file 01 version -- same structure, same `#if defined(COMPILE_FOR_TRISC)` guard, same `init()`/`operator()()`/`teardown()` lifecycle pattern, same `DeviceZoneScopedN` tracing calls. The file 02 version additionally includes `#include` directives and `using` type aliases, making it strictly more informative. Showing the same pattern twice across two files adds ~20 lines of near-duplicate code.
**Suggestion:** In file 01 (Section 1.1.6), replace the full 22-line `kernel_main()` listing with a condensed 6-8 line sketch that shows only the parallel-compilation concept (the `#if defined` structure and the sequential phase calls), then add a forward reference: "A complete auto-generated kernel is shown in Section 1.2.6." Keep the full, annotated version in file 02 where it belongs (kernel codegen context). Saves ~15 lines.

### [01_tensix_core_model.md + 02_blaze_compilation_pipeline.md] Cross-file: `init_is_empty` / `teardown_is_empty` explanation stated twice
**Issue:** The explanation that `init_is_empty` and `teardown_is_empty` are auto-detected by `blaze/cpp_parser.py` and cause codegen to skip empty lifecycle calls appears in two places:
  - File 01, line 430: "The codegen engine is aware of this: when `init_is_empty` or `teardown_is_empty` is detected by the C++ parser during op registration (see `blaze/blaze_op.py`, `_auto_derive_from_kernel_hpp()`), the generated kernel skips the corresponding `init()` / `teardown()` calls entirely, avoiding even the function-call overhead."
  - File 02, lines 404-408: Shows the `PhaseInfo` dataclass with `init_is_empty` and `teardown_is_empty` fields, then explains: "The `init_is_empty` and `teardown_is_empty` flags are auto-detected by the C++ parser (`blaze/cpp_parser.py`), which inspects the op header's `init()` and `teardown()` method bodies. If they contain only whitespace or comments, the codegen skips emitting those calls -- eliminating the function call overhead entirely."
  These two passages convey identical information (auto-detection by cpp_parser, codegen skips empty calls). The file 02 version is more detailed and contextually appropriate (it sits alongside the `PhaseInfo` dataclass definition).
**Suggestion:** In file 01 (line 430), replace the full sentence with a brief forward reference: "The codegen engine detects empty `init()` / `teardown()` bodies and skips them entirely (see Section 1.2.6, `PhaseInfo`)." Keep the full explanation in file 02. Saves ~3 lines of text but, more importantly, eliminates a conceptual duplicate that may confuse readers who encounter both.

## MINOR Suggestions

### [02_blaze_compilation_pipeline.md + 03_micro_op_vs_fused_op.md] ~lines 330 and 339
**Issue:** The statement "Both modes/APIs ultimately produce the same artifact/compiled kernel and program descriptor" appears in both files. File 02 line 330 says it with detail ("a `ProgramDescriptor` with three kernel binaries, CB descriptors, and an ordered tensor list for `ttnn.generic_op()`"). File 03 line 339 says it more briefly ("the same compiled kernel and program descriptor"). The file 02 version is the canonical one; file 03's restatement adds no new information.
**Suggestion:** Remove line 339 from file 03. The reader has already been told this in file 02, and the cross-reference at file 03 line 319 already points the reader to Sections 1.2.1 and 1.2.5.

### [03_micro_op_vs_fused_op.md] ~lines 349-352 and 361-363
**Issue:** The "A MicroOp requires:" and "A FusedOp requires only:" lists in Section 1.3.6 restate the file layout already given in Sections 1.3.1 (lines 17-21) and 1.3.2 (lines 111-114). This was noted as a MINOR item in Pass 1 and was not applied.
**Suggestion:** Replace each list with a back-reference: "See Section 1.3.1 for the MicroOp file layout" and "See Section 1.3.2 for the FusedOp file layout." Saves ~6 lines.

### [01_tensix_core_model.md] ~lines 305 and 334
**Issue:** The "compiled three times" fact is stated twice within Section 1.1.6. Line 305: "it is compiled three times -- once for each processor -- with different preprocessor defines active." Line 334: "This single source file is compiled three times by the JIT -- once with `COMPILE_FOR_NCRISC` defined (for DM0), once with `COMPILE_FOR_BRISC` defined (for DM1), and once with `COMPILE_FOR_TRISC` defined (for the three TRISC sub-cores)." Then takeaway #4 (line 440) states it a third time.
**Suggestion:** Remove the brief version at line 305 or the detailed version at line 334. One statement plus the takeaway is sufficient. Keeping line 334 (more detailed) and cutting line 305's second clause saves ~10 words.

### [index.md] ~lines 21-27 ("Reading order")
**Issue:** Carried over from Pass 1 (not applied). The "Reading order" section paraphrases the table at lines 15-19. Both say "read 01 first, then 02, then 03."
**Suggestion:** Remove the "Reading order" section. The table's sequential numbering implies reading order, and the brief elaboration does not add enough to justify the ~7 extra lines.

### [01_tensix_core_model.md] ~line 62
**Issue:** Carried over from Pass 1 (not applied). "Understanding which processor owns which responsibility is the single most important mental model for writing TT-Blaze ops." -- hedging/emphasis language.
**Suggestion:** Replace with a neutral lead-in or remove the sentence entirely.

## Load-Bearing Evidence
- `01_tensix_core_model.md` line ~16: "Every Tensix core on a Tenstorrent Blackhole chip contains **five RISC-V processor cores**." -- load-bearing because it establishes the hardware fact that anchors the entire chapter; no shorter phrasing is possible without losing precision about the Blackhole chip target.
- `02_blaze_compilation_pipeline.md` line ~345: "When a fused op does not provide a handwritten kernel source, TT-Blaze's kernel codegen system (`blaze/kernel_codegen.py`) generates the C++ kernel automatically from the `BlazeGraph` (or shadow graph)." -- load-bearing because it is the only place the distinction between handwritten and auto-generated kernels is introduced, and it names the exact source file.
- `03_micro_op_vs_fused_op.md` line ~379: "Some ops start as FusedOps and later become MicroOps when the fused composition proves too slow." -- load-bearing because this "gray zone" migration path is unique guidance not stated elsewhere; removing it would leave a conceptual gap.
- `01_tensix_core_model.md` lines ~224-234 (CB operations table): The four-row `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` table -- load-bearing because it is the only definition of the complete synchronization API, and every subsequent section depends on readers understanding these four calls.

## VERDICT
- Crucial updates: yes

## Change Log (Applied 2026-05-13, Pass 2)

Both CRUCIAL suggestions from Pass 2 were applied:

1. **[01_tensix_core_model.md] Section 1.1.6: Cross-file duplicate kernel_main() code block.** Replaced the 22-line auto-generated kernel listing with a 6-8 line structural sketch (showing the init/call/teardown pattern) and added a forward reference: "See Section 1.2.6 for the full codegen output including PhaseInfo, DPRINT markers, and role guards." Saved ~15 lines.

2. **[01_tensix_core_model.md] Section 1.1.8: Cross-file init_is_empty/teardown_is_empty duplicate.** Replaced the full explanation of how cpp_parser auto-detects empty init/teardown bodies with a one-sentence forward reference: "The code generator detects empty `init()` and `teardown()` bodies and omits them (see Section 1.2.6)." The detailed explanation is retained in file 02 Section 1.2.6 where it belongs alongside the `PhaseInfo` dataclass.

---

# Compression Analysis: Chapter 1 — Architecture and Mental Model — Pass 3

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1349 lines
- Estimated post-compression line count: ~1310 lines
- Estimated reduction: ~3%

## CRUCIAL Suggestions
None.

All 8 CRUCIAL items from Passes 1 and 2 have been applied. A systematic re-check of the six previously identified redundancy categories confirms each fix is in place:

1. **Mcast code in Section 1.1.5**: Now shows only the 2-line `send_with_state` snippet with a back-reference to Section 1.1.2. No duplicate code blocks remain.
2. **emit() walkthrough merge (file 02)**: Section 1.2.1 Phase 2 now contains the Mcast `emit()` walkthrough inline. No standalone Section 1.2.2 restating the same material.
3. **Auto-derivation triple restatement (file 03)**: Section 1.3.1 retains the bulleted list of `_auto_derive_from_kernel_hpp()` steps without the redundant concluding paragraph.
4. **Cross-file ct_types.h dedup**: Full explanation lives in file 01 Section 1.1.4. File 03 Section 1.3.7 (line ~410) uses a single-sentence cross-reference.
5. **kernel_main() code block dedup**: File 01 Section 1.1.6 shows a condensed sketch with a forward reference to Section 1.2.6. File 02 Section 1.2.6 retains the full annotated version.
6. **init_is_empty/teardown_is_empty dedup**: File 01 Section 1.1.8 (line ~417) uses a one-sentence forward reference. File 02 Section 1.2.6 retains the full explanation alongside the `PhaseInfo` dataclass.

No new cross-file or within-file CRUCIAL redundancy was identified.

## MINOR Suggestions

### [01_tensix_core_model.md] ~lines 305 and 320-321
**Issue:** The "compiled three times" fact is stated twice within Section 1.1.6. Line 305: "it is compiled three times -- once for each processor -- with different preprocessor defines active." Lines 320-321: "This single source file is compiled three times by the JIT -- once with `COMPILE_FOR_NCRISC` defined (for DM0), once with `COMPILE_FOR_BRISC` defined (for DM1), and once with `COMPILE_FOR_TRISC` defined (for the three TRISC sub-cores)." Then takeaway #4 (line ~440) states it a third time. (Carried over from Pass 2.)
**Suggestion:** Remove the brief version at line 305 (second clause of that sentence). The detailed version at lines 320-321 is more informative and should be the single in-section statement. The takeaway is acceptable as a summary.

### [03_micro_op_vs_fused_op.md] ~lines 339
**Issue:** "Both APIs ultimately produce the same compiled kernel and program descriptor." This echoes file 02 line 330: "Both modes ultimately produce the same artifact: a ProgramDescriptor with three kernel binaries, CB descriptors, and an ordered tensor list for ttnn.generic_op()." (Carried over from Pass 2.)
**Suggestion:** Remove line 339 from file 03. The reader has already been told this in file 02, and file 03 line 319 already cross-references Sections 1.2.1 and 1.2.5.

### [03_micro_op_vs_fused_op.md] ~lines 349-352 and 361-363
**Issue:** The "A MicroOp requires:" and "A FusedOp requires only:" lists in Section 1.3.6 restate the file layout already given in Sections 1.3.1 and 1.3.2. (Carried over from Pass 1 and Pass 2.)
**Suggestion:** Replace each list with a back-reference: "See Section 1.3.1 for the MicroOp file layout" and "See Section 1.3.2 for the FusedOp file layout." Saves ~6 lines.

### [index.md] ~lines 21-27 ("Reading order")
**Issue:** The "Reading order" section paraphrases the table at lines 15-19. Both say "read 01 first, then 02, then 03" with slight elaboration. (Carried over from Pass 1 and Pass 2.)
**Suggestion:** Remove the "Reading order" section. The table's sequential numbering implies reading order.

### [01_tensix_core_model.md] ~line 62
**Issue:** "Understanding which processor owns which responsibility is the single most important mental model for writing TT-Blaze ops." -- emphasis/hedging language. (Carried over from Pass 1 and Pass 2.)
**Suggestion:** Replace with a neutral lead-in or remove the sentence entirely.

## Load-Bearing Evidence
- `01_tensix_core_model.md` line ~16: "Every Tensix core on a Tenstorrent Blackhole chip contains **five RISC-V processor cores**." -- load-bearing because it establishes the foundational hardware fact (chip target, core count, core type) that the entire chapter builds on; cannot be shortened without losing precision.
- `02_blaze_compilation_pipeline.md` line ~345: "When a fused op does not provide a handwritten kernel source, TT-Blaze's kernel codegen system (`blaze/kernel_codegen.py`) generates the C++ kernel automatically from the `BlazeGraph` (or shadow graph)." -- load-bearing because it is the sole introduction of the handwritten vs. auto-generated kernel distinction, naming the exact source file responsible.
- `03_micro_op_vs_fused_op.md` line ~379: "Some ops start as FusedOps and later become MicroOps when the fused composition proves too slow." -- load-bearing because this migration-path guidance is unique to this section and cannot be inferred from any other passage in the chapter.
- `01_tensix_core_model.md` lines ~226-233 (CB operations table): The four-row table defining `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` -- load-bearing because it is the only complete definition of the synchronization API; every subsequent code example and explanation depends on it.

## VERDICT
- Crucial updates: no
