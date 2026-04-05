# Cross-Chapter Compression Analysis — Pass 1

## Summary
- Total files analyzed: 17 (9 index files + 8 content files spot-checked)
- Estimated cross-chapter redundancy: ~120 lines
- Estimated reduction: ~2%

## CRUCIAL Suggestions

None.

## MINOR Suggestions

1. **`addr_mod_t` struct and usage pattern duplicated between Ch4 and Ch8.**
   - `ch4_unpack_math_pack_pipeline/math_operations.md` (lines 146-164) defines the `addr_mod_t` struct with code example and field descriptions.
   - `ch8_kernel_development_patterns/api_conventions.md` (lines 216-244) repeats the struct definition, field descriptions, and a near-identical usage example.
   - **Suggestion:** Ch4 should be the authoritative definition (it covers the math pipeline where `addr_mod_t` is used). Ch8 should reference Ch4 with a brief summary and a link, removing the duplicated struct definition and field table (~20 lines saved).

2. **`ckernel_template` MOP structure duplicated between Ch4 and Ch8.**
   - `ch4_unpack_math_pack_pipeline/synchronization.md` (lines 208-257) documents the MOP loop structure, constructor API, last-iteration replacement, and usage pattern.
   - `ch8_kernel_development_patterns/api_conventions.md` (lines 148-186) repeats the loop structure diagram, constructor, `set_*` methods, and `program()`/`run()` pattern.
   - **Suggestion:** Keep the full MOP reference in Ch4 (synchronization). Ch8 should provide a concise recap with a cross-reference link, removing the duplicated loop diagram and constructor walkthrough (~25 lines saved).

3. **`ckernel_unpack_template` duplicated between Ch4 and Ch8.**
   - `ch4_unpack_math_pack_pipeline/synchronization.md` (lines 259-276) describes the zmask-based unpack MOP.
   - `ch8_kernel_development_patterns/api_conventions.md` (lines 189-211) repeats the loop structure and factory methods.
   - **Suggestion:** Keep the full description in Ch4. Ch8 should link to Ch4 with a one-sentence summary (~15 lines saved).

4. **SFPU load/store architecture described in both Ch2 and Ch5.**
   - `ch2_tensix_core_architecture/data_flow.md` (lines 47-61) describes the SFPU's load/compute/store model and the two paths to get data into Dest.
   - `ch5_sfpu_operations/sfpu_architecture.md` (lines 13-25) describes the same three-step model and the same two paths (FPU datacopy and direct unpack).
   - **Suggestion:** Ch2 provides architectural context (data flow overview) and Ch5 provides programming detail. The Ch2 version is already brief. No action strictly needed, but Ch2 could add a forward-reference sentence ("See Ch5 for the full SFPU programming model") to signal the deeper treatment. (~0 lines saved, clarity improvement only.)

5. **`InstrModLoadStore` enum and table duplicated between Ch3 and Ch5.**
   - `ch3_data_organization/data_formats.md` (lines 77-95) shows the full `InstrModLoadStore` enum code with comments.
   - `ch5_sfpu_operations/sfpu_architecture.md` (lines 31-44) shows a table with the same 12 values, numeric codes, and data format mappings.
   - **Suggestion:** The enum definition belongs in Ch3 (data formats). Ch5 should reference Ch3's definition and only show the subset relevant to SFPU usage, or link to the Ch3 table (~12 lines saved).

6. **Key Terms glossaries in chapter index files repeat definitions across chapters.**
   - "FPU" is defined in ch1/index.md, ch4/index.md.
   - "SFPU" is defined in ch1/index.md, ch4/index.md.
   - "Source A / Source B" is defined in ch2/index.md, ch4/index.md.
   - "Destination Register" is defined in ch2/index.md, ch4/index.md.
   - "Tile" is defined in ch1/index.md, ch3/index.md.
   - **Suggestion:** This is a deliberate pedagogical pattern (each chapter glossary defines terms introduced in that chapter's context). The definitions are slightly different in each chapter, reflecting the chapter's focus. This is acceptable redundancy for a tutorial guide. No action needed.

## Load-Bearing Evidence

1. **`addr_mod_t` duplication (Ch4 vs Ch8):**
   - Ch4 `math_operations.md` lines 150-154: `addr_mod_t { .srca = {.incr = MAX_FPU_ROWS}, .srcb = {.incr = MAX_FPU_ROWS}, .dest = {.incr = MAX_FPU_ROWS}, }.set(ADDR_MOD_0);`
   - Ch8 `api_conventions.md` lines 237-240: `addr_mod_t { .srca = {.incr = MAX_FPU_ROWS}, .srcb = {.incr = MAX_FPU_ROWS}, .dest = {.incr = MAX_FPU_ROWS}, }.set(ADDR_MOD_0);`
   - These are verbatim identical code blocks.

2. **MOP loop structure diagram (Ch4 vs Ch8):**
   - Ch4 `synchronization.md` lines 213-222: `LOOP_OUTER: <m_outer_loop_len> / m_start_op0 / LOOP_INNER: <m_inner_loop_len> / m_loop_op0 / m_loop_op1 / END_LOOP_INNER / m_end_op0 / m_end_op1 / END_LOOP_OUTER`
   - Ch8 `api_conventions.md` lines 153-164: identical structure diagram with identical field names.

3. **`InstrModLoadStore` values (Ch3 vs Ch5):**
   - Ch3 `data_formats.md` lines 82-94: enum code listing `DEFAULT = 0, FP16A = 1, FP16B = 2, FP32 = 3, INT32 = 4, INT8 = 5, LO16 = 6, HI16 = 7, INT32_2S_COMP = 12, INT8_2S_COMP = 13, LO16_ONLY = 14, HI16_ONLY = 15`
   - Ch5 `sfpu_architecture.md` lines 32-43: table with same 12 values, same numeric codes, same data format mappings.

4. **`ckernel_unpack_template` loop structure (Ch4 vs Ch8):**
   - Ch4 `synchronization.md` lines 263-274: `LOOP: / if (!zmask[iteration]): UNPACR_A0 / UNPACR_A1 (if halo) / ...`
   - Ch8 `api_conventions.md` lines 195-200: `LOOP: / if (zmask[iteration]): UNPACR_A0, A1, A2, A3 (halo mode) / UNPACR_B / else: SKIP_A / SKIP_B`
   - Near-identical structure with minor phrasing differences.

## VERDICT
- Crucial updates: no
