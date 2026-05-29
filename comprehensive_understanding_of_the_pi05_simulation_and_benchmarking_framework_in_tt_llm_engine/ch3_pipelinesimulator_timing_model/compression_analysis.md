# Compression Analysis: Chapter 3 -- PipelineSimulator Timing Model

## Summary

Chapter 3 spans four files (~950 lines of content excluding navigation). The material is technically precise and well-structured, but contains significant cross-file code duplication -- the same source snippets are reproduced verbatim across multiple sections. One section (`01_simulator_design.md` section 5) also triple-states the busy-wait rationale within a single page. Estimated compressible content: ~15-18% of total line count.

---

## CRUCIAL Suggestions

### C1. Pi0.5 simulator config snippet repeated 3 times verbatim

The `pi05_pipeline_runner.cpp:881-887` code block (the `PipelineSimulatorConfig` with `.batch_prefill = true`) appears identically in:
- `01_simulator_design.md` lines 483-489
- `02_batch_prefill_behavior.md` lines 173-179
- `03_token_model_and_spec_decode.md` lines 380-386

**Recommendation:** Show the full snippet once in `01_simulator_design.md` (its natural home as the config mapping section). In the other two files, replace with a one-line forward reference: "The Pi0.5 runner enables `batch_prefill = true` (see Section 3.1, Pi0.5 Pipeline Runner Usage)."

Saves ~30 lines.

### C2. DecodeScheduler constructor mapping code duplicated across two files

The `decode_scheduler.cpp:99-103` `std::make_unique<PipelineSimulator>(...)` block appears identically in:
- `01_simulator_design.md` lines 457-463
- `02_batch_prefill_behavior.md` lines 43-47

**Recommendation:** Keep in `01_simulator_design.md` (section 7, its dedicated context). In `02_batch_prefill_behavior.md`, replace with a cross-reference: "Propagated through `DecodeScheduler` via `std::visit` (see Section 3.1, PipelineSimulatorConfig Mapping)."

Saves ~8 lines.

### C3. Backpressure `injectCv.wait()` code block appears 3 times

The `injectCv.wait(lock, ...)` snippet appears at:
- `01_simulator_design.md` line 196 (standalone excerpt)
- `01_simulator_design.md` line 226 (inside full `inject()` listing)
- `02_batch_prefill_behavior.md` line 258 (re-quoted to explain bypass)

**Recommendation:** The standalone excerpt in section 2 of `01_simulator_design.md` is redundant with the full `inject()` listing in section 3 (which is only ~20 lines later). Remove the standalone excerpt and reference the full listing. In `02_batch_prefill_behavior.md`, use a prose description with a line-number reference instead of re-quoting.

Saves ~12 lines.

### C4. `makeResult()` prefill early return explained in two files

`02_batch_prefill_behavior.md` section 7 (lines 377-394) explains the prefill early return from `makeResult()` with a code block and prose. `03_token_model_and_spec_decode.md` section 1 "Prefill Early Return" (lines 63-66) re-explains the same behavior.

**Recommendation:** Keep the explanation in `03_token_model_and_spec_decode.md` (where the full `makeResult()` is analyzed). In `02_batch_prefill_behavior.md`, shorten section 7 to a brief note with a forward reference to Section 3.3.

Saves ~15 lines.

---

## MINOR Suggestions

### M1. Triple-stated busy-wait rationale in `01_simulator_design.md`

Section 5 ("Why No Tick Thread") conveys the same busy-wait advantage three ways:
1. The quoted source comment (lines 330-339)
2. "Trade-off: Busy-wait vs sleep_until" prose (lines 369-384)
3. Post-table bullet list (lines 394-401), which restates: "burns zero CPU when idle," "sub-microsecond timing accuracy," and "5-50 us wake-up jitter" -- all present in the preceding prose.

**Recommendation:** Delete the post-table bullet list (lines 394-401). The table row "Busy-wait (current)" plus the preceding prose already cover these points.

Saves ~8 lines.

### M2. Test snippet `test_decode_scheduler.cpp:1582-1586` appears in two files

The same `pipeline_config` with `decode_token_id = 12345` appears in:
- `02_batch_prefill_behavior.md` lines 407-412
- `03_token_model_and_spec_decode.md` lines 332-336

Each file uses it to make a different point (mid-prefill stop vs. deterministic mode), so duplication is more defensible here. Still, one could be replaced with a cross-reference.

### M3. `02_batch_prefill_behavior.md` section 5 re-explains backpressure mechanism

Lines 252-260 re-quote and re-explain the backpressure wait predicate before explaining why batch-prefill bypasses it. Since the mechanism was fully covered in `01_simulator_design.md` section 2 (Invariant 3), a one-sentence recap with cross-reference would suffice.

### M4. Hedging note about stale comment in `01_simulator_design.md`

Lines 340-341 contain a "Note" about the class comment referencing `sleep_until` while the implementation uses busy-wait. This is useful context but could be shortened to a single parenthetical rather than a bold-formatted multi-line note.

---

## Load-Bearing Evidence (do NOT compress)

- **`index.md`**: Section descriptions with cross-references -- already minimal, serves as chapter TOC.
- **`01_simulator_design.md`**: The three-invariant definitions (section 2, lines 127-204), the full `inject()` listing with step-by-step (section 3), the full `read_result()` listing with step-by-step (section 4), the FIFO ordering guarantee proof (lines 255-259), the single-reader contract explanation (lines 313-322), the comparative analysis table (lines 388-393), and the `PipelineSimulatorConfig` field mapping table (lines 465-474).
- **`02_batch_prefill_behavior.md`**: The token-type behavior table (lines 112-116), the hardware behavior explanation (section 3, lines 139-149), the TTFT formulas for both modes (section 4), the rate-limit isolation analysis (lines 285-296), the config propagation diagram (lines 305-327), and the reader-side prefill tracking with `prefill_in_flight` atomics (lines 346-368).
- **`03_token_model_and_spec_decode.md`**: The full `makeResult()` listing (lines 18-58), the accept/reject arithmetic with the "+1 vs +3" rationale (section 2), the minimum modulus >= 5 formal proof (lines 226-241), the verification correctness proof under wrapping (lines 246-258), the worked wrapping example table (lines 265-275), the fixed-token mode properties (section 5), and the summary table of all token generation modes (lines 500-507).

---

## VERDICT

**COMPRESS -- ~65-75 lines recoverable** primarily from cross-file code snippet duplication (C1-C4) and one instance of intra-file triple-statement (M1). The technical content is high-quality and well-organized; the bloat is almost entirely repeated code blocks that should be shown once and cross-referenced thereafter.

---

## Change Log

### 2026-05-28 -- Applied all 4 CRUCIAL compression suggestions

- **C1 (Pi0.5 config snippet x3):** Kept full `PipelineSimulatorConfig` snippet in `01_simulator_design.md`. Replaced with one-line cross-references in `02_batch_prefill_behavior.md` (section 3) and `03_token_model_and_spec_decode.md` (section 6).
- **C2 (DecodeScheduler constructor mapping x2):** Kept `std::make_unique<PipelineSimulator>(...)` block in `01_simulator_design.md`. Replaced with cross-reference in `02_batch_prefill_behavior.md` (section 1).
- **C3 (Backpressure injectCv.wait() x3):** Removed standalone code excerpt from `01_simulator_design.md` section 2 (full `inject()` listing in section 3 already shows it). Replaced re-quoted block in `02_batch_prefill_behavior.md` section 5 with prose description and line-number reference.
- **C4 (makeResult() prefill early return x2):** Kept explanation in `03_token_model_and_spec_decode.md` (section 1, "Prefill Early Return"). Shortened `02_batch_prefill_behavior.md` section 7 to a brief note with forward reference to Section 3.3.

---

# Compression Analysis: Chapter 3 -- Pass 2

## Summary

All four CRUCIAL items from pass 1 (C1-C4) have been resolved. The Pi0.5 config snippet, DecodeScheduler constructor mapping, backpressure code excerpt, and makeResult() prefill early return are each shown once in their canonical location with cross-references elsewhere. No new CRUCIAL issues found. One MINOR suggestion remains.

## CRUCIAL

Crucial updates: no

## Load-Bearing Evidence

- **`index.md`** (36 lines): Minimal chapter TOC with section descriptions. No duplication, no compression targets. Verified: three section entries with accurate cross-links to 01, 02, 03.
- **`01_simulator_design.md`** (493 lines): C1 resolved -- full Pi0.5 config snippet retained here (lines 477-484) as the canonical location. C2 resolved -- `std::make_unique<PipelineSimulator>` block retained here (line 454) as the canonical location. C3 resolved -- standalone `injectCv.wait()` excerpt removed from section 2; Invariant 3 (lines 189-199) now uses prose with a forward reference to the full `inject()` listing in section 3 (line 221). Three-invariant definitions, full `inject()` and `read_result()` listings, FIFO ordering proof, single-reader contract, comparative analysis table, and config mapping table all intact.
- **`02_batch_prefill_behavior.md`** (390 lines): C1 resolved -- Pi0.5 config replaced with one-line cross-reference at line 161. C2 resolved -- DecodeScheduler mapping replaced with cross-reference at line 40. C3 resolved -- backpressure section 5 (lines 232-237) uses prose with line-number reference and cross-link instead of re-quoting code. C4 resolved -- section 7 (lines 349-356) is now a brief note with forward reference to Section 3.3. Token-type behavior table, hardware behavior explanation, TTFT formulas, rate-limit isolation analysis, config propagation diagram, and prefill_in_flight atomics all intact.
- **`03_token_model_and_spec_decode.md`** (508 lines): C1 resolved -- Pi0.5 config replaced with cross-reference at lines 376-378. Full `makeResult()` listing, accept/reject arithmetic with +1/+3 rationale, minimum modulus >= 5 formal proof, verification correctness proof under wrapping, worked wrapping example, fixed-token mode properties, and summary table of all token generation modes all intact.

## MINOR

### M5. `prefill_token_id = is_last ? EMPTY_TOKEN : ...` line appears twice within `02_batch_prefill_behavior.md`

The single line `.prefill_token_id = is_last ? EMPTY_TOKEN : prompt_table.get_token(pfuid, prompt_idx + 1)` appears at line 92 (inside the full `InjectDescriptor` construction block in section 2) and again at line 311 (standalone single-line excerpt in the config propagation section 6). Both serve distinct analytical purposes -- the first explains how the writer distinguishes prefill tokens; the second highlights separation of concerns between scheduler and pipeline. The duplication is a single line and defensible given the different contexts, but the line 311 instance could be replaced with a prose reference to the code block at line 89-97 for a marginal improvement.

## VERDICT

**PASS -- no CRUCIAL items remain.** All four pass 1 CRUCIAL compressions (C1-C4) were applied correctly: duplicated code snippets were consolidated to single canonical locations with cross-references. Chapter 3 content is clean and well-organized. One minor single-line duplication (M5) is the only remaining compression target, and it is defensible in context.
