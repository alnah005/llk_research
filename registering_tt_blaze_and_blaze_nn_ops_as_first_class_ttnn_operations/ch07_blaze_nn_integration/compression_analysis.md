# Compression Analysis: Connecting blaze-nn to Registered Operations — Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~1560 lines
- Estimated post-compression line count: ~1150 lines
- Estimated reduction: ~26%

## CRUCIAL Suggestions

### [02_rewired_dispatch_and_module_integration.md] ~lines 287-375
**Issue:** The scenario walkthrough in Section 7.2.5 (Calls 1-8) repeats the entire dispatch trace from Section 7.1.6 in file 01, line by line, with `[CHANGED]` annotations. The rmsnorm trace (lines 293-322) is 30 lines for a single call that could be shown once in a condensed diff format. Calls 3-4, 5-6, and 8 are described identically to their 7.1.6 counterparts ("same dispatch path as Call 2"), which was already said in 01. The full 90-line walkthrough duplicates structure that the reader has already internalized from Section 7.1.6.
**Suggestion:** Replace the 8-call walkthrough with one fully annotated call (Call 1 or Call 2), then a compact summary noting "all eight calls follow the same pattern; the only change is at Layers 6-9 where `generic_op` is replaced by `ttnn.blaze.{op}`." This eliminates ~50 lines without losing any information the reader cannot derive from the single example.

### [02_rewired_dispatch_and_module_integration.md] ~lines 378-441
**Issue:** Section 7.2.6 (Before/After Comparison) restates three observability artifacts -- Tracy timeline, graph trace, and program cache -- that are already shown in Section 7.1.7 (01, lines 396-427) and again in the Section 7.2.11 layer map (02, lines 770-787) and Section 7.2.12 summary matrix (02, lines 790-830). The "BEFORE" halves of the Tracy timeline, graph trace, and program cache blocks are near-verbatim copies of the content in 01 lines 400-425. The "AFTER" halves are then restated in the layer map and summary matrix.
**Suggestion:** Merge the before/after comparison into a single condensed table or inline it into Section 7.2.11 (the layer map), which already provides a before/after column. Remove the standalone code blocks for the "BEFORE" state since they duplicate Section 7.1.7. The profiler aggregation table (lines 429-438) is the only net-new content and should be kept. This saves ~35 lines.

### [02_rewired_dispatch_and_module_integration.md] ~lines 770-840
**Issue:** Section 7.2.11 (Dispatch Layer Map) and Section 7.2.12 (Comprehensive Summary Matrix) overlap heavily. The layer map table (lines 774-784) restates the same Layers 1-9 already shown in 01 Section 7.1.10 (lines 483-494) with a new "After" column. Then the summary matrix (lines 796-830) restates the same dimensions again in three separate tables: "Dispatch Mechanics," "Observable System Behaviors," and "Unchanged Components." The "Observable System Behaviors" table (lines 806-812) is a subset of the layer map. The "Unchanged Components" table (lines 815-829) restates information already given in Section 7.2.7 (Module integration), the index.md Key Takeaways, and the chapter's own Key Takeaways section.
**Suggestion:** Keep Section 7.2.11 (the layer map with before/after columns) as the single consolidated comparison artifact. Fold the "Dispatch Mechanics" rows into prose or a short paragraph. Remove the "Observable System Behaviors" table entirely (it is a strict subset of 7.2.11). Condense "Unchanged Components" to a single sentence referencing Section 7.2.7. This saves ~40 lines.

### [02_rewired_dispatch_and_module_integration.md] ~lines 952-968 and [index.md] ~lines 20-31
**Issue:** The Key Takeaways at the end of 02 (8 bullet points, lines 952-968) largely restate the Key Takeaways in index.md (5 bullet points, lines 20-31). Specific duplications: (a) "Module integration is transparent/zero-cost" appears in both, (b) "fallback strategy is layered" appears in both with nearly identical wording, (c) "two dispatch modes, one registration benefit" / "graph mode and direct mode both benefit" is the same point, (d) the `_REGISTRY` as the natural place for `ttnn_callable` is restated. The index.md also duplicates takeaway content from the body of 02 Sections 7.2.2, 7.2.3, 7.2.4, and 7.2.7.
**Suggestion:** Remove the Key Takeaways section from index.md entirely (the chapter index should not pre-summarize the conclusions) OR remove the Key Takeaways section from 02 and let index.md serve as the sole summary. Keeping both is pure duplication. Estimated saving: ~20 lines from whichever is removed.

### [01_current_blaze_nn_dispatch.md] ~lines 304-393
**Issue:** The scenario walkthrough in Section 7.1.6 devotes 90 lines to tracing 8 calls, but Calls 3-4, 5-6, and 8 add no new information beyond Call 1 and Call 2. Calls 3-4 (lines 358-361) say "identical dispatch path to Call 2" and Calls 5-6 (lines 363-373) show a nearly identical trace to Call 1 with a different op name. Call 8 (lines 390-392) says "same dispatch path as Calls 2-4." The reader learns the dispatch pattern from Call 1 (rmsnorm) and Call 2 (linear/matmul); the remaining calls demonstrate only that the pattern repeats.
**Suggestion:** After Call 2, replace Calls 3-8 with a single paragraph: "The remaining six calls (K/V projections, two RoPE applications, SDPA, output projection) follow the same dispatch path, differing only in the Blaze op name and grid configuration. All terminate at `ttnn.generic_op`." This saves ~30 lines.

### [02_rewired_dispatch_and_module_integration.md] ~lines 686-766
**Issue:** Section 7.2.10 (End-to-End Flow Diagram) is a 78-line ASCII diagram that recapitulates the entire dispatch pipeline already explained in Sections 7.2.1-7.2.5 and traced in the scenario walkthrough. Every box in the diagram corresponds to a section that already provided both prose explanation and code/pseudocode. The diagram adds visual structure but duplicates all content.
**Suggestion:** Condense the diagram to approximately 20 lines by removing the inline annotations (e.g., `[blaze-nn functional API]`, `[lazily resolved, cached]`) and collapsing the adapter/launch internals into a single box. The detailed adapter internals (lines 738-761) were already covered in Chapters 4-5 and do not need re-expansion here.

## MINOR Suggestions

### [01_current_blaze_nn_dispatch.md] ~lines 53-56
**Issue:** The sentence "blaze-nn maintains a central registry that maps functional API names to Blaze op specifications. The registry is populated at import time and serves as the single source of truth for all dispatch decisions." restates what was already said in the index.md line 22 ("The `_REGISTRY` is the natural place to store `ttnn_callable` references. Each `OpMapping` already maps a functional name to a Blaze op name...") and in 01 line 3 ("the `_REGISTRY`, `OpMapping`...").
**Suggestion:** Trim to a single sentence introducing the code block.

### [02_rewired_dispatch_and_module_integration.md] ~lines 119-132
**Issue:** The "Design Decisions" subsection uses verbose explanatory paragraphs for decisions that the code comments already make clear. For example, lines 120-122 explain lazy resolution in 3 sentences, but the `ttnn_callable` property code (lines 39-52) already has a docstring explaining the same thing. Similarly, lines 124-126 explain three-state caching, which is already documented in the code comment on lines 33-35.
**Suggestion:** Reduce each design decision paragraph to one sentence, since the code and its comments already convey the reasoning. This saves ~15 lines.

### [02_rewired_dispatch_and_module_integration.md] ~lines 186-209
**Issue:** Section 7.2.3 (Fallback Strategy) describes three fallback levels in prose (lines 188-198), then immediately presents a "Fallback Decision Table" (lines 202-209) that restates the same three levels plus two additional edge cases. The prose and table say the same thing in two formats.
**Suggestion:** Keep only the table (it is more scannable) and remove the three "Level" prose subsections. Add the two additional edge cases (import failure, env var override) as brief inline notes if needed. Saves ~12 lines.

### [01_current_blaze_nn_dispatch.md] ~lines 196-199
**Issue:** "During graph tracing, `_dispatch()` receives `TensorProxy` objects instead of real `ttnn.Tensor` objects. The tracing context uses the proxy metadata (shape, dtype) to construct graph nodes with correct edge types, then returns a new `TensorProxy` representing the output. No hardware execution occurs." This repeats what the class docstring on lines 187-191 already says and what Section 7.1.5 (lines 268-270) says again for graph mode.
**Suggestion:** Remove this paragraph; the docstring and the graph-mode walkthrough cover it.

### [02_rewired_dispatch_and_module_integration.md] ~lines 462-488
**Issue:** Section 7.2.7 repeats "No changes" for Parameter, state_dict, and Module in four separate subsections with code snippets showing UNCHANGED code. The point "module integration is transparent" is made in index.md line 28, in 01 line 222-223, and again here.
**Suggestion:** Collapse the three "unchanged" subsections into a single sentence: "Parameter, state_dict(), load_state_dict(), and Module.forward() are unchanged -- parameter ownership is orthogonal to dispatch and the adapter operates on MeshProgramDescriptor objects produced after parameter resolution." Remove the UNCHANGED code snippets.

### [02_rewired_dispatch_and_module_integration.md] ~lines 573-586
**Issue:** The "Tensor Compatibility" subsection (lines 573-586) includes a 7-row table where 5 rows say "Compatible" with minimal additional information. The two rows with meaningful content (program cache isolation, graph trace unification) are already covered in Section 7.2.12.
**Suggestion:** Replace the table with a two-sentence paragraph noting that tensors are type-compatible and that cache/trace behaviors are unified, referencing the summary matrix.

### [01_current_blaze_nn_dispatch.md] ~lines 429-431
**Issue:** Hedging language: "Performance analysis becomes a forensic exercise rather than a diagnostic one." This is rhetorical flair that adds no technical content.
**Suggestion:** Remove or replace with a direct statement like "Performance analysis requires manual timestamp correlation."

### [02_rewired_dispatch_and_module_integration.md] ~lines 1-3
**Issue:** The opening sentence recaps Section 7.1 in detail: "Section 7.1 traced a transformer attention block through blaze-nn's current dispatch stack and showed that op identity is lost at Layer 6 when `_run_program()` unconditionally calls `ttnn.generic_op()`." This is a useful cross-reference but is overly specific -- the reader just finished 7.1.
**Suggestion:** Shorten to "Section 7.1 showed that op identity is lost at Layer 6. This section rewires the dispatch path..."

### [02_rewired_dispatch_and_module_integration.md] ~lines 155-181
**Issue:** Section 7.2.2 (Dual-Resolution Architecture) has a paragraph explaining "Why two resolution points?" (lines 162-163) that restates what was just said in the two "Resolution Point" paragraphs above it. The ASCII diagram (lines 165-180) then restates it a third time.
**Suggestion:** Remove the "Why two resolution points?" paragraph and let the ASCII diagram serve as the explanation. Saves ~5 lines.

### [02_rewired_dispatch_and_module_integration.md] ~lines 831-841
**Issue:** The "Per-File Change Summary" table (lines 831-841) restates the "Source File Index" table in index.md (lines 65-79). Both list the same files with the same modification status.
**Suggestion:** Remove the per-file table from 02 and reference the index.md table instead.

## Load-Bearing Evidence
(Not applicable -- verdict is "Crucial updates: yes".)

## VERDICT
- Crucial updates: yes

---

## Change Log

### Pass 1 Compression Applied (2026-05-14)

All 6 CRUCIAL suggestions applied:

1. **CRUCIAL 1 [02, Section 7.2.5]:** Replaced 8-call walkthrough with one fully annotated call (`F.linear`) plus a compact summary paragraph stating all eight calls follow the same pattern. Eliminated ~50 lines.

2. **CRUCIAL 2 [02, Section 7.2.6]:** Removed standalone BEFORE/AFTER code blocks for Tracy timeline, graph trace, and program cache (duplicated Section 7.1.7 and Section 7.2.11). Retained the per-op profiler aggregation table (net-new content). Section retitled to "Observability Impact: Per-Op Profiler Aggregation" with cross-references to 7.2.11 and 7.1.7. Saved ~35 lines.

3. **CRUCIAL 3 [02, Section 7.2.12]:** Removed "Observable System Behaviors" table (strict subset of layer map in 7.2.11). Folded "Dispatch Mechanics" rows into prose. Condensed "Unchanged Components" to a single sentence referencing Section 7.2.7. Consolidated per-file table. Section retitled to "Change Summary". Saved ~40 lines.

4. **CRUCIAL 4 [02, Key Takeaways]:** Removed the Key Takeaways section from 02 entirely. index.md serves as the sole summary. Saved ~20 lines.

5. **CRUCIAL 5 [01, Section 7.1.6]:** Replaced Calls 3-8 with a single paragraph summarizing that the remaining six calls follow the same dispatch path, differing only in Blaze op name and grid configuration, all terminating at `ttnn.generic_op`. Saved ~30 lines.

6. **CRUCIAL 6 [02, Section 7.2.10]:** Condensed the 78-line ASCII flow diagram to ~20 lines by removing inline annotations and collapsing adapter/launch internals into a single box with a cross-reference to Chapters 4-5. Saved ~55 lines.

Navigation footers preserved on all three files. Section numbering unchanged (no renumbering needed).

---

# Compression Analysis: Connecting blaze-nn to Registered Operations — Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~1320 lines
- Estimated post-compression line count: N/A
- Estimated reduction: N/A

## Pass 1 CRUCIAL Verification

1. **8-call scenario walkthrough in 02 Section 7.2.5 condensed to 1 call + summary:** VERIFIED. Section 7.2.5 (lines 287-324) shows only one fully annotated call (`F.linear(h, self.wq)`) followed by a single-paragraph summary at line 324 stating all eight calls follow the same pattern with `generic_op` replaced by `ttnn.blaze.{op}` at Layers 6-9.

2. **Before/after observability blocks in 02 Section 7.2.6 -- BEFORE blocks removed, profiler aggregation table kept:** VERIFIED. Section 7.2.6 (lines 329-345) is retitled "Observability Impact: Per-Op Profiler Aggregation." No BEFORE code blocks remain. Only the per-op profiler aggregation table and cross-references to Sections 7.2.11 and 7.1.7 are present.

3. **Layer map + summary matrix overlap in 02 Sections 7.2.11-7.2.12 merged:** VERIFIED. Section 7.2.11 (lines 621-637) retains the consolidated before/after layer map table. Section 7.2.12 (lines 639-654) is retitled "Change Summary" with a brief prose paragraph, a per-file change summary table, and a single-sentence reference to Section 7.2.7. The "Observable System Behaviors" table and "Comprehensive Summary Matrix" are gone.

4. **Key Takeaways removed from 02 (index.md is sole summary):** VERIFIED. File 02 ends at line 767 with a navigation footer. No Key Takeaways section exists in 02. Key Takeaways appear only in index.md (lines 20-31).

5. **Calls 3-8 in 01 Section 7.1.6 replaced with summary paragraph:** VERIFIED. After Call 2 (ending at line 356), lines 358-360 contain a single summary paragraph: "The remaining six calls -- K/V projections, two RoPE applications, SDPA, output projection -- follow the same dispatch path demonstrated in Calls 1 and 2..."

6. **ASCII flow diagram condensed from ~78 to ~20 lines:** VERIFIED. Section 7.2.10 (lines 592-615) contains the end-to-end flow diagram spanning approximately 19 lines. Inline annotations are removed and adapter internals are collapsed into a single line with a cross-reference to Chapters 4-5.

## CRUCIAL Suggestions
None.

## MINOR Suggestions

1. **[02_rewired_dispatch_and_module_integration.md] lines 119-132 (Design Decisions subsection):** The five "Design Decisions" paragraphs under Section 7.2.1 (lazy resolution, three-state caching, `__ttnn_operation__` guard, `try/except`, `BLAZE_NN_FORCE_GENERIC`, per-op granularity) restate what the code and its inline comments already convey. For example, the lazy resolution paragraph (lines 120-121) explains the same logic documented in the `ttnn_callable` property docstring (lines 40-51), and the three-state caching paragraph (lines 122-123) restates the comments on lines 33-35. These could be condensed to one-sentence bullets each, saving approximately 15 lines.

2. **[02_rewired_dispatch_and_module_integration.md] lines 186-209 (Fallback Strategy):** Section 7.2.3 describes three fallback levels in prose (Level 1, Level 2, Level 3 subsections at lines 188-198), then immediately presents a "Fallback Decision Table" (lines 202-209) that restates the same three levels plus two additional edge cases. The prose and table are redundant. Keeping only the table (which is more scannable) and removing the three "Level" prose subsections would save approximately 12 lines.

3. **[01_current_blaze_nn_dispatch.md] lines 196-199 (TensorProxy explanation paragraph):** The paragraph "During graph tracing, `_dispatch()` receives `TensorProxy` objects instead of real `ttnn.Tensor` objects..." restates the `TensorProxy` class docstring on lines 187-191 and is repeated again in the graph-mode discussion in Section 7.1.5 (lines 268-270). Removing this paragraph would eliminate a triple-statement of the same concept.

## Load-Bearing Evidence

- **index.md:** The Key Takeaways section (lines 20-31) is the sole consolidated summary for the chapter, covering the five architectural conclusions (single dispatch choke point, `_REGISTRY` as natural storage, dual dispatch mode benefit, transparent module integration, layered fallback). This is the only location where these conclusions appear after the removal of 02's Key Takeaways, making every bullet load-bearing.

- **01_current_blaze_nn_dispatch.md:** The dispatch layer map in Section 7.1.10 (lines 448-463) is the definitive "before" baseline for op-identity visibility across Layers 1-9, directly referenced by the "after" layer map in 02 Section 7.2.11. The layer map table at lines 452-461 establishes that identity is lost at Layers 6-9, which is the foundational claim that motivates the entire chapter. Removing or altering this table would break the before/after comparison.

- **02_rewired_dispatch_and_module_integration.md:** The `_resolve_ttnn_callable()` function (lines 85-116) is the authoritative specification for how the `OpMapping.ttnn_callable` property resolves a registered TTNN callable, including the `BLAZE_NN_FORCE_GENERIC` env var check, the `_get_ttnn_blaze_module()` gate, and the `__ttnn_operation__` verification guard. This function encodes the three-level fallback logic in executable form and is not duplicated elsewhere.

## VERDICT
- Crucial updates: no
