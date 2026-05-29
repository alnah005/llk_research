# Compression Analysis: Chapter 2 — Pass 1

## Summary
- Total files analyzed: 4
- Estimated current line count: ~1,089 lines
- Estimated post-compression line count: ~930 lines
- Estimated reduction: ~15%

## CRUCIAL Suggestions

None.

## MINOR Suggestions

1. **Duplicate pixel payload arithmetic (3 renditions).** The pixel_bytes / total_bytes / padded-payload calculation is worked out fully in `01_config_schema_reference.md` lines 140-150 (derived methods), again in lines 252-262 (FIFO size check), and a third time in `03_reference_configs_walkthrough.md` lines 140-149 (pixel payload metrics for the full config). The first occurrence in 01 should be the canonical derivation; the FIFO-check section can reference it ("using the values derived in Section 2.1.4"); and the walkthrough in 03 can replace the four-line re-derivation with a one-line forward reference plus only the final result. Saves ~15 lines.

2. **Duplicate phase-validation enumeration across files.** The five parse-time validation rules (name required, stages non-empty, stage duration >= 1us, loop_count >= 1, at least one phase) are listed with error messages in `01_config_schema_reference.md` Section 2.1.2 (lines 49-58) and the Phase Validation table (lines 213-220), then repeated verbatim in `02_custom_json_parser.md` Section 2.2.7 error table (lines 370-376). One canonical table is sufficient; the other should cross-reference it. Saves ~10 lines.

3. **Single-action-token explanation restated.** The `output_tokens = 1` / single-action-token explanation (50-joint bfloat16, 3,200 bytes, synchronization signal) appears in `03_reference_configs_walkthrough.md` Section 2.3.1 (lines 152-153) and is then restated almost identically in Section 2.3.4 (lines 397-401). The second occurrence should be a one-sentence summary with a back-reference to Section 2.3.1. Saves ~5 lines.

4. **`pcie_bandwidth_gbps` informational-only warning restated.** The warning that this field is never used in any calculation appears as a block-quote WARNING in `01_config_schema_reference.md` line 94 and is re-explained in `03_reference_configs_walkthrough.md` line 217. The walkthrough should cite the schema reference rather than re-explain. Saves ~2 lines.

5. **`skip_json_value` call-site blocks are repetitive.** Section 2.2.3 of `02_custom_json_parser.md` (lines 194-230) shows four near-identical code-plus-prose blocks for the four call sites (phase, socket, pixel_payload, top-level). Each block is a 3-line code snippet and 1-line description following the same pattern. These could be consolidated into a bulleted list with line references, dropping the repetitive code fences. Saves ~15 lines.

6. **`parse_number` / `parse_float_val` asymmetry stated three times.** The note that `parse_number()` cannot handle negatives while `parse_float_val()` can is mentioned at `02_custom_json_parser.md` line 88, line 109, and line 133. Keep the first occurrence (line 109, under Key Design Decisions) and drop the other two. Saves ~4 lines.

7. **Sim config re-lists numbers already declared identical.** `03_reference_configs_walkthrough.md` Section 2.3.3 states "All the numerical results computed in Section 2.3.1 apply" (line 336) then immediately re-lists them in a table (lines 338-346). The table is redundant given the explicit forward reference. Replace with "See Section 2.3.1 for the complete derivation." Saves ~8 lines.

8. **Stagger auto-computation explained twice.** The formula and prose for auto-computed stagger appears in `01_config_schema_reference.md` lines 198-206 and again in `03_reference_configs_walkthrough.md` lines 157-161. The walkthrough should reference the schema section for the formula and show only the substituted result. Saves ~3 lines.

## Load-Bearing Evidence

- `01_config_schema_reference.md`: Pixel payload arithmetic at lines 140-150 is duplicated in both the same file's FIFO section (lines 252-262) and in `03_reference_configs_walkthrough.md` lines 140-149.
- `02_custom_json_parser.md`: The `parse_number`/`parse_float_val` asymmetry note appears at lines 88, 109, and 133 -- three separate statements of the same observation.
- `03_reference_configs_walkthrough.md`: The single-action-token explanation at lines 152-153 is restated nearly verbatim at lines 397-401 in the same file.
- `index.md`: No bloat detected; concise and well-structured chapter overview.

## VERDICT
- Crucial updates: no
