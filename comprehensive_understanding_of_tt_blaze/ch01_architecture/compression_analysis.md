# Compression Analysis: Chapter 1 -- Architecture and Position in the Stack -- Pass 1

## Summary
- Total files analyzed: 3
- Estimated current line count: ~691 lines
- Estimated post-compression line count: ~570 lines
- Estimated reduction: ~17%

## CRUCIAL Suggestions

### [01_stack_position.md + 02_repository_map.md] ~lines 39-52 / 127-139: Duplicate kernel_includes / LLK extension description
**Issue:** The `kernel_includes/` directory, custom SFPU headers, and vendored LLK layout are described in detail twice. In `01_stack_position.md` lines 39-52, the section "Relationship to TT-LLK" enumerates the `kernel_includes/` tree, the custom SFPU headers (`ckernel_sfpu_sdpa_reduce_row.h`, etc.), the `sfpu/` directory files (`clamped_silu_sfpu.hpp`, `zero_pad_sfpu.hpp`), the `tt_llk/` and `tt_metal/` namespace layout, and the Blackhole vs. Wormhole B0 variants. Then in `02_repository_map.md` lines 127-139, the "Kernel Headers" table restates the same content: `ops.hpp`, `kernel_op_api.hpp`, `ct_types.h`, `kernel_includes/tt_llk/`, `kernel_includes/tt_metal/`, and `sfpu/` with the same two files.
**Suggestion:** Keep the detailed enumeration in one place only. `01_stack_position.md` should describe LLK's role and that Blaze extends it via `kernel_includes/` (conceptual), then defer the directory-by-directory breakdown to `02_repository_map.md`. Remove the five-bullet header list and the `sfpu/` paragraph from `01_stack_position.md` lines 41-52, replacing with a single sentence pointing to the repository map. Eliminates ~12 lines of duplicated content.

### [01_stack_position.md + 02_repository_map.md] ~lines 54-95 / 130-135: Duplicate description of ops.hpp, kernel_op_api.hpp, and ct_types.h
**Issue:** `01_stack_position.md` lines 54-95 provide prose descriptions plus code snippets for three kernel headers: `ops.hpp` (aggregator include), `kernel_op_api.hpp` (RISC detection, `SelectByRISCV`), and `ct_types.h` (typed aliases). `02_repository_map.md` lines 130-135 repeats the same three files with purpose descriptions in a table. The reader encounters the same files, the same role summaries, and overlapping detail in two places.
**Suggestion:** Move the detailed prose and code examples for these three headers to `02_repository_map.md` where the "Kernel Headers" table lives. In `01_stack_position.md`, keep only a brief mention that Blaze adds custom kernel headers (covered in Section 02). Removes ~30 lines of duplication from `01_stack_position.md`.

### [01_stack_position.md + 02_repository_map.md] ~lines 63-84 / 17: Duplicate tt-metal submodule description
**Issue:** `01_stack_position.md` lines 63-84 contain a full section "The tt-metal Submodule" describing the git submodule pin, C++20 kernel compilation, named CT/RT arg infrastructure, and the `kernel_op_api.hpp` code pattern. `02_repository_map.md` line 17 notes `tt-metal/` as "Git submodule -- pinned tt-metal commit on main." The `kernel_op_api.hpp` snippet in `01_stack_position.md` (lines 71-84) duplicates content that belongs in the kernel headers discussion already present in `02_repository_map.md`.
**Suggestion:** The submodule feature list (C++20, named CT/RT args, ProgramDescriptor API) is load-bearing in `01_stack_position.md`. But trim the `kernel_op_api.hpp` and `SelectByRISCV` code snippet (lines 71-84) from `01_stack_position.md` since that file is already covered in the kernel headers table in `02_repository_map.md`. Saves ~14 lines.

### [02_repository_map.md + 03_two_api_design.md] ~lines 199-207 / 166-184: Duplicate BlazeProgram / FusedProgram method listings
**Issue:** `02_repository_map.md` lines 199-207 list BlazeProgram's capabilities: CB allocation methods (`cb_from_tensor`, `cb_scratch`, `cb_alias`, `cb_output`), semaphore allocation, named CT/RT args, per-core CT args, validation, build, and run. Then `03_two_api_design.md` lines 166-184 presents a table of `FusedProgram` methods repeating the same method names with nearly identical purpose descriptions. Since `FusedProgram` wraps `BlazeProgram`, these are the same API surface described twice.
**Suggestion:** In `02_repository_map.md`, reduce the `BlazeProgram` description to a brief summary stating it is the low-level program builder and note that the full method listing appears in Section 03. Keep the comprehensive table in `03_two_api_design.md` only. Removes ~8 lines from `02_repository_map.md`.

### [02_repository_map.md + 03_two_api_design.md] ~lines 77 / 188-195: Duplicate CBHandle description
**Issue:** `CBHandle` is described in `02_repository_map.md` line 77 with field list (cb_id, num_pages, page_size, data_format, tile_desc, backing tensor, access mode) and `int()` coercion note. Then `03_two_api_design.md` lines 188-195 provides a more detailed enumeration of the same fields. Both mention `int()` coercion.
**Suggestion:** Keep the full description in `03_two_api_design.md`. In `02_repository_map.md`, reduce to: "`CBHandle` dataclass -- typed reference to a circular buffer; see Section 03 for full field list." Eliminates the duplicate field enumeration.

## MINOR Suggestions

### [01_stack_position.md] ~lines 18, 33: Restated ProgramDescriptor convergence point
**Issue:** Line 18 says "the same primitive objects that a hand-rolled TT-Metal op would construct" and line 33 says "Both systems ultimately produce ProgramDescriptor objects that TT-Metal dispatches identically." These are the same point stated twice within 15 lines.
**Suggestion:** Remove one instance. The line 33 version is better placed (end of comparison section). Cut the phrase from line 18.

### [01_stack_position.md] ~lines 31
**Issue:** "Blaze takes the opposite approach: the developer hand-writes C++ kernel structs (micro-ops) that run on individual RISC processors, then uses Blaze's Python API to compose those ops into fused pipelines" is a near-verbatim restatement of line 14 in the stack table and the opening paragraph on line 5.
**Suggestion:** Replace with: "Blaze inverts this: the developer writes the kernels and Blaze handles composition."

### [01_stack_position.md] ~lines 129-133: Four pillars partially redundant
**Issue:** Pillars 1 ("Kernel code is real C++") and 2 ("Composition is Python") restate what lines 5, 14, and 31 already established. Pillars 3 ("No hidden compiler passes") and 4 ("Rapid iteration") are new information.
**Suggestion:** Compress pillars 1 and 2 into a single sentence referencing the earlier description. Keep pillars 3 and 4 at current length.

### [02_repository_map.md] ~lines 180-223: Key Infrastructure Modules duplicate earlier tables
**Issue:** The prose summaries for DeviceContext (line 180 vs. table line 82), GridConfig/RoleEngine (line 185 vs. table line 81), Barrier (line 217 vs. table line 88), and CCL (line 221 vs. table line 89) repeat purpose descriptions already given in the "Engines and Resource Allocation" and "Infrastructure and Utilities" tables earlier in the same file.
**Suggestion:** Merge the prose details into the tables (expand "Purpose" column) rather than having both a table entry and a separate prose section. Eliminates ~15 lines of structural duplication within the same file.

### [03_two_api_design.md] ~lines 143-152: Numbered steps restate preceding code
**Issue:** The six numbered steps under "How emit() Works Internally" are a prose restatement of the code snippet at lines 120-128. The target audience (C++ kernel developers) can read the code directly.
**Suggestion:** Remove the numbered list or reduce to a 2-sentence summary. The code snippet is self-explanatory.

### [03_two_api_design.md] ~lines 226-234: Prose restates ASCII diagram
**Issue:** The paragraphs explaining the graph API path and composition API path restate the ASCII convergence diagram at lines 209-224 in prose form.
**Suggestion:** Reduce to 1-2 sentences per path noting only what the diagram cannot show (e.g., the shadow graph's dual purpose). Cut ~8 lines.

### [03_two_api_design.md] ~lines 273-291: "When to Use Each API" restates convergence table
**Issue:** The five bullets under each API and the closing paragraph ("In practice, the two APIs are complementary") largely restate the convergence summary table at lines 262-271.
**Suggestion:** Shorten to 2-3 bullets per API and cut the closing paragraph. Saves ~8 lines.

### [02_repository_map.md] ~lines 226-255: Auto-discovery prose narrates code step-by-step
**Issue:** Lines 228-231 describe in prose what the code at lines 236-241 shows, then lines 254-255 restate what the `_OpHandle` code at lines 246-253 demonstrates.
**Suggestion:** Keep the code snippets and the final dispatch explanation ("blaze.matmul(...) dispatches to the graph API... blaze.matmul.emit(f, ...) dispatches to the composition API"). Trim the intervening narration. Saves ~5 lines.

## Load-Bearing Evidence
- `01_stack_position.md` line ~7: "The full stack, from bottom to top:" -- load-bearing because the six-layer stack table (lines 9-16) is the canonical reference for Blaze's position; not replicated elsewhere.
- `01_stack_position.md` line ~64: "Blaze pins TT-Metal as a git submodule under tt-metal/" -- load-bearing because this is the only place enumerating the four TT-Metal features Blaze depends on (C++20, named CT args, named RT args, ProgramDescriptor API).
- `01_stack_position.md` line ~101: "the SharedExpert FusedOp in Blaze (67 lines... replaces 979 lines)" -- load-bearing because this concrete quantification is the primary evidence for Blaze's value proposition.
- `02_repository_map.md` line ~3: "Top-Level Directory Structure" -- load-bearing because this is the only navigational map of the entire repository.
- `02_repository_map.md` line ~57: "Core Modules" table -- load-bearing because this is the only comprehensive enumeration of every Python module in `blaze/` with its purpose.
- `03_two_api_design.md` line ~9: "Entry Point: blaze.fuse()" -- load-bearing because this is the canonical graph API entry point description with the only usage example.
- `03_two_api_design.md` line ~209: ASCII convergence diagram -- load-bearing because this is the only visual showing how both API paths converge to program descriptor output.
- `03_two_api_design.md` line ~262: "Convergence Summary" table -- load-bearing because this is the only side-by-side comparison of the two APIs across all relevant dimensions.

## VERDICT
- Crucial updates: yes

## Change Log

### 2026-05-07 -- All 5 CRUCIAL compression items applied (Agent A)

1. **kernel_includes/LLK duplication (CRUCIAL 1):** In `01_stack_position.md`, replaced the five-bullet custom header list, the `tt_llk/`/`tt_metal/` namespace paragraph, and the `sfpu/` two-bullet paragraph (former lines ~41-52) with a single paragraph that mentions the extensions conceptually and defers to `02_repository_map.md` Section "Kernel Headers" for the full breakdown. Removed ~12 lines.

2. **ops.hpp/kernel_op_api.hpp/ct_types.h duplication (CRUCIAL 2):** Removed the `ops.hpp` code snippet, `kernel_op_api.hpp` prose, and `ct_types.h` prose+code from `01_stack_position.md` (former lines ~54-95). Added a brief mention deferring to `02_repository_map.md`. Moved the detailed prose and code examples (`ops.hpp` usage snippet, `SelectByRISCV` template alias snippet, `ct_types.h` type aliases snippet) into `02_repository_map.md` below the Kernel Headers table. Also enriched the table entries for `kernel_includes/tt_llk/`, `kernel_includes/tt_metal/`, and `sfpu/` with the specific file examples and descriptions that were removed from `01_stack_position.md`. Net removal of ~30 lines from `01_stack_position.md`.

3. **kernel_op_api.hpp/SelectByRISCV snippet duplication (CRUCIAL 3):** Trimmed the `kernel_op_api.hpp` code snippet and `ct_types.h` code snippet from the "tt-metal Submodule" section of `01_stack_position.md` (former lines ~71-84 + ~86-95). Replaced with a two-sentence summary referencing `SelectByRISCV` and deferring to `02_repository_map.md`. Preserved the load-bearing submodule feature list (C++20, named CT/RT args, ProgramDescriptor API). Removed ~14 lines.

4. **BlazeProgram/FusedProgram method duplication (CRUCIAL 4):** In `02_repository_map.md`, replaced the 8-bullet BlazeProgram method listing (former lines ~199-207) with a two-sentence summary noting it is the low-level program builder and deferring to Section 03 "Key `FusedProgram` Methods" for the full method listing. The comprehensive table in `03_two_api_design.md` is now the single source of truth. Removed ~8 lines.

5. **CBHandle field duplication (CRUCIAL 5):** In `02_repository_map.md`, replaced the CBHandle table entry (former line ~77) which listed all fields (cb_id, num_pages, page_size, data_format, tile_desc, backing tensor, access mode) and the `int()` coercion note with a brief entry deferring to Section 03 for the full field list. The comprehensive description in `03_two_api_design.md` lines 188-195 is now the single source of truth.

**Files modified:** `01_stack_position.md`, `02_repository_map.md`. `03_two_api_design.md` was not modified (it is the target for deferred content). Navigation footers preserved in all files.

---

# Compression Analysis: Chapter 1 -- Architecture and Position in the Stack -- Pass 2

## Summary
- Total files analyzed: 3
- Estimated current line count: ~668 lines
- Estimated post-compression line count: ~630 lines
- Estimated reduction: ~6%

## CRUCIAL Suggestions
None -- all previous CRUCIAL items resolved.

Verification of each item:

1. **kernel_includes/LLK extension duplication (CRUCIAL 1):** RESOLVED. `01_stack_position.md` line 39 now contains a single paragraph mentioning `kernel_includes/` conceptually with a cross-reference to `02_repository_map.md` Section "Kernel Headers." The former five-bullet header list and `sfpu/` paragraph are gone. The detailed enumeration lives only in `02_repository_map.md` lines 127-139.

2. **ops.hpp/kernel_op_api.hpp/ct_types.h duplication (CRUCIAL 2):** RESOLVED. `01_stack_position.md` line 39 briefly names the three headers inline and defers to `02_repository_map.md`. Code snippets and prose descriptions for all three live only in `02_repository_map.md` lines 130-169. No duplication remains.

3. **SelectByRISCV snippet duplication (CRUCIAL 3):** RESOLVED. `01_stack_position.md` line 50 mentions `SelectByRISCV` by name without a code snippet, deferring to `02_repository_map.md`. The `SelectByRISCV` template alias code snippet appears only in `02_repository_map.md` lines 151-160. The `ct_types.h` code snippet likewise appears only in `02_repository_map.md` lines 162-169.

4. **BlazeProgram/FusedProgram method listing duplication (CRUCIAL 4):** RESOLVED. `02_repository_map.md` lines 227-228 now contain a two-sentence summary deferring to Section 03 "Key `FusedProgram` Methods." The comprehensive method table lives only in `03_two_api_design.md` lines 166-184.

5. **CBHandle field enumeration duplication (CRUCIAL 5):** RESOLVED. `02_repository_map.md` line 77 now reads: "`CBHandle` dataclass -- typed reference to a circular buffer. See [Section 03](./03_two_api_design.md) for the full field list and usage patterns." The full field list is only in `03_two_api_design.md` lines 188-195.

No new CRUCIAL-level redundancy was introduced by the fixes. The cross-references added by the fixes are clean, appropriately scoped, and do not themselves introduce duplication.

## MINOR Suggestions

### [01_stack_position.md] ~lines 18, 33: Restated ProgramDescriptor convergence point (carried from Pass 1)
**Issue:** Line 18 says "the same primitive objects that a hand-rolled TT-Metal op would construct" and line 33 says "Both systems ultimately produce `ProgramDescriptor` objects that TT-Metal dispatches identically." These make the same point twice within 15 lines.
**Suggestion:** Remove the redundant clause from line 18 or line 33. The line 33 version is better placed at the end of the Forge comparison section.

### [01_stack_position.md] ~lines 31: Restated Blaze description (carried from Pass 1)
**Issue:** "Blaze takes the opposite approach: the developer hand-writes C++ kernel structs (micro-ops) that run on individual RISC processors, then uses Blaze's Python API to compose those ops into fused pipelines" restates lines 5 and 14.
**Suggestion:** Compress to: "Blaze inverts this: the developer writes the kernels and Blaze handles composition."

### [02_repository_map.md] ~lines 209-240: Key Infrastructure Modules duplicate earlier tables (carried from Pass 1)
**Issue:** Prose summaries for DeviceContext (~line 212), GridConfig/RoleEngine (~line 215), Barrier (~line 239), and CCL (~line 243) repeat purpose descriptions already given in the "Engines and Resource Allocation" and "Infrastructure and Utilities" tables at lines 72-96.
**Suggestion:** Merge the distinguishing details from the prose into the tables and eliminate the separate prose sections. Saves ~15 lines.

### [03_two_api_design.md] ~lines 143-152: Numbered steps restate preceding code (carried from Pass 1)
**Issue:** The six numbered steps under "How emit() Works Internally" narrate the code snippet at lines 120-128 in prose form.
**Suggestion:** Reduce to a 2-sentence summary highlighting only what the code does not make obvious (e.g., shadow graph recording).

## Load-Bearing Evidence
- `01_stack_position.md` line ~39: "Blaze brings its own LLK extensions via the `kernel_includes/` directory tree under `blaze/kernels/`." -- load-bearing because this is now the sole conceptual anchor for the LLK extension story in the stack-position file, with all detail properly deferred to 02.
- `02_repository_map.md` line ~77: "`CBHandle` dataclass -- typed reference to a circular buffer. See [Section 03](./03_two_api_design.md) for the full field list and usage patterns." -- load-bearing because this cross-reference is the single navigational link connecting the repository map to the definitive CBHandle description; removing it would leave readers without a path to the field details.
- `03_two_api_design.md` line ~188: "`CBHandle` -- the currency of the composition API." -- load-bearing because this is now the single source of truth for the CBHandle field enumeration after deduplication; all other files defer to this location.

## VERDICT
- Crucial updates: no
