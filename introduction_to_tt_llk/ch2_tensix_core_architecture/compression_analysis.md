# Compression Analysis: Tensix Core Architecture -- Pass 1

## Summary
- Total files analyzed: 5
- Estimated current line count: ~423 lines
- Estimated post-compression line count: ~340 lines
- Estimated reduction: ~20%

## CRUCIAL Suggestions
### [core_components.md] ~lines 54-63 / [riscv_processors.md] ~lines 4-11
**Issue:** The five-RISC-V-processor table is duplicated across these two files with near-identical columns (name, full name, role). `core_components.md` has a 5-row table; `riscv_processors.md` has the same 5-row table with an added "Thread ID" and "LLK Relevant?" column.
**Suggestion:** Remove the table from `core_components.md` and replace it with a single sentence referencing `riscv_processors.md` for the full breakdown. Keep only the canonical table in `riscv_processors.md`.

### [riscv_processors.md] ~lines 58-62 / [tensix_engine.md] ~lines 74-85
**Issue:** The "two instruction sets" distinction (RISC-V ISA vs. Tensix ISA) is explained in both files. `riscv_processors.md` has a prose paragraph; `tensix_engine.md` has a 4-column comparison table plus a prose paragraph. The content is largely redundant.
**Suggestion:** Keep the full explanation (table + prose) in `tensix_engine.md` only. In `riscv_processors.md`, reduce to a one-sentence summary with a cross-reference: "The TRISC processors execute RISC-V code that injects Tensix ISA instructions into the engine (see [Tensix Engine -- Tensix ISA vs. RISC-V ISA](./tensix_engine.md#tensix-isa-vs-risc-v-isa))."

### [data_flow.md] ~lines 99-138
**Issue:** The "Complete Data Flow Summary" section (lines 99-138) contains a second ASCII diagram of the full pipeline that repeats the same topology shown in the first diagram (lines 9-36). The three numbered paths (Path 1, 2, 3) at lines 128-136 add value, but the second diagram is visually redundant with the first.
**Suggestion:** Remove the second ASCII diagram (lines 103-126). Keep the "Path 1 / Path 2 / Path 3" text summary, which is compact and adds clarity without repeating the visual.

## MINOR Suggestions
### [core_components.md] ~lines 71-81 / [data_flow.md] ~lines 39-43
**Issue:** The five-step LLK pipeline walkthrough in `core_components.md` ("How LLKs Use These Components") restates the same unpack-math-pack sequence described in `data_flow.md` pipeline stages. Both describe TRISC0 unpacking, TRISC1 computing, TRISC2 packing.
**Suggestion:** Shorten the `core_components.md` section to two sentences and a forward-reference to `data_flow.md` for the full pipeline description.

### [core_components.md] ~line 67 / [tensix_engine.md] ~lines 1-3
**Issue:** The introductory description of the Tensix Engine is stated almost identically in both places: "The Tensix Engine is the hardware accelerator at the heart of the Tensix Core" appears verbatim in both files.
**Suggestion:** In `core_components.md` line 67, reduce to: "The Tensix Engine is the Tensix Core's hardware accelerator, described in detail in [tensix_engine.md](./tensix_engine.md)." Remove the second sentence about "multi-threaded coprocessor optimized for matrix computations."

### [Multiple files] repeated phrase "hardware gaskets for data format conversion"
**Issue:** The phrase "hardware gaskets" performing "data format conversion" appears 6 times across `riscv_processors.md` (lines 29, 51), `tensix_engine.md` (lines 57, 62), and `data_flow.md` (lines 41, 43). The concept is restated each time it is mentioned.
**Suggestion:** Define "hardware gaskets" once in `core_components.md` or `data_flow.md` and use a short-hand ("with format conversion") in subsequent mentions.

### [core_components.md] ~line 49 / [riscv_processors.md] ~line 15
**Issue:** Both files state that NoC communication is "outside the scope of compute kernels" / "outside the scope of LLK development." The same boundary is drawn twice.
**Suggestion:** Keep the statement in `core_components.md` (where NoC is introduced). In `riscv_processors.md`, remove or reduce to a parenthetical: "(not LLK-relevant)".

### [index.md] ~line 5
**Issue:** The overview sentence is 63 words and packs four subordinate clauses into a single sentence, making it harder to parse.
**Suggestion:** Break into two sentences: one on what a Tensix Core is, one on what the chapter covers.

## Load-Bearing Evidence
- `core_components.md` line ~57: "| **BRISC** | Board RISC-V | NOC communication and board setup |" -- load-bearing because this is one of two near-identical RISC-V processor tables; removing the duplicate from this file saves ~8 lines without losing any information that is not already in `riscv_processors.md`.
- `riscv_processors.md` line ~60: "An important distinction to keep in mind: the TRISC processors execute RISC-V instructions (standard 32-bit RISC-V ISA) as their native instruction set." -- load-bearing because this opens the duplicated two-ISA explanation; the same concept plus a comparison table exists in `tensix_engine.md` lines 74-85.
- `data_flow.md` line ~99: "Putting it all together, here is every data path through the Tensix Engine:" -- load-bearing because it introduces the second (redundant) ASCII diagram; the first diagram at line 9 already shows the same topology.
- `tensix_engine.md` line ~1: "The Tensix Engine is the hardware accelerator at the heart of each Tensix Core." -- load-bearing because it duplicates the opening line of `core_components.md` line 67 nearly verbatim.
- `index.md` line ~5: "This chapter explains the hardware architecture that LLKs program." -- load-bearing because the 63-word sentence that follows is the most verbose single sentence in the chapter and could be split for clarity.

## VERDICT
- Crucial updates: yes

## Change Log -- Compression Applied (2026-04-05)

### CRUCIAL fixes applied:
1. **core_components.md**: Removed the duplicated 5-row RISC-V processor table (lines 55-61) and replaced it with a single sentence referencing `riscv_processors.md` for the full breakdown.
2. **riscv_processors.md**: Replaced the "Two Instruction Sets" section (lines 58-62, two full paragraphs) with a one-sentence summary cross-referencing `tensix_engine.md#tensix-isa-vs-risc-v-isa`.
3. **data_flow.md**: Removed the second redundant ASCII diagram (lines 103-126) from the "Complete Data Flow Summary" section. Kept the Path 1/2/3 text summary.

### MINOR fixes applied:
4. **core_components.md**: Shortened the Tensix Engine intro (section 4) from two sentences to one, removing the duplicate description already present in `tensix_engine.md`.
5. **core_components.md**: Condensed the "How LLKs Use These Components" five-step list into a two-sentence summary with a forward-reference to `data_flow.md`.

# Compression Analysis: Tensix Core Architecture — Pass 2

## Summary
- Total files analyzed: 5
- Estimated current line count: ~381 lines
- Estimated post-compression line count: ~365 lines
- Estimated reduction: ~4%

## CRUCIAL Suggestions
None. All three Pass 1 CRUCIAL items have been verified as fixed:
1. The duplicated RISC-V processor table has been removed from `core_components.md` (line 53 now contains a prose sentence with cross-reference instead of a table).
2. The duplicated two-ISA explanation in `riscv_processors.md` has been replaced with a single-sentence cross-reference to `tensix_engine.md` (lines 58-60).
3. The redundant second ASCII pipeline diagram has been removed from `data_flow.md`; only the Path 1/2/3 text summary remains (lines 99-113).

## MINOR Suggestions
### [riscv_processors.md] ~lines 27-56
**Issue:** The three TRISC subsections each repeat the phrase pattern "Key LLK headers controlled by TRISCN include:" followed by a bullet list. The intro sentence is boilerplate repeated three times with only the processor name changed.
**Suggestion:** Replace the three repeated intro sentences with a single shared sentence before the subsections: "Each TRISC controls specific LLK header files:" Then begin each subsection's bullet list directly.

### [data_flow.md] ~lines 40-43
**Issue:** The three pipeline stage descriptions (lines 40-43) each begin with a bold label and restate the TRISC responsible, which is already shown in the ASCII diagram directly above. The phrase "Hardware gaskets convert the data format during transfer" at line 41 restates a concept also found in `riscv_processors.md` line 29 and `tensix_engine.md` lines 57, 62.
**Suggestion:** Drop the parenthetical TRISC labels from each stage description (the diagram already maps them). Shorten the gasket mention to "with format conversion" since the concept is defined in `tensix_engine.md`.

### [tensix_engine.md] ~lines 46-47
**Issue:** The sentence "This means that TRISC0's unpack instructions are processed sequentially, TRISC1's math instructions are processed sequentially, and TRISC2's pack instructions are processed sequentially" is a three-part restatement of the preceding sentence ("The frontends operate in-order within each thread").
**Suggestion:** Remove the three-part expansion; the preceding sentence already conveys in-order dispatch clearly.

### [index.md] ~lines 29-37
**Issue:** The Key Terms table defines "Source A / Source B" and "Destination Register (Dest)" with descriptions that are nearly identical to the register file tables in `data_flow.md` lines 70-97.
**Suggestion:** Shorten the index definitions to one-clause summaries (e.g., "Double-buffered input operand registers for the FPU") since the full specifications are in `data_flow.md`.

## Load-Bearing Evidence
- `core_components.md` line ~53: "The Tensix Core contains five 32-bit, in-order, single-issue RISC-V cores: BRISC, NCRISC, TRISC0, TRISC1, and TRISC2." — load-bearing because it confirms the duplicated table has been replaced with prose plus a cross-reference.
- `riscv_processors.md` line ~60: "The TRISC processors execute RISC-V code that injects Tensix ISA instructions into the engine (see [Tensix Engine -- Tensix ISA vs. RISC-V ISA](./tensix_engine.md#tensix-isa-vs-risc-v-isa))." — load-bearing because it confirms the two-ISA duplication has been collapsed to a single cross-reference sentence.
- `data_flow.md` line ~102: "**Path 1 -- FPU operations:**" — load-bearing because it confirms the second ASCII diagram was removed and only the compact Path 1/2/3 text summary remains.
- `tensix_engine.md` line ~46: "The frontends operate in-order within each thread -- instructions from a single thread are decoded and dispatched in the order they were issued by the corresponding TRISC." — load-bearing because the sentence immediately following it is a verbose three-part restatement that could be cut.
- `index.md` line ~36: "Double-buffered source operand registers that hold unpacked tile data for the FPU." — load-bearing because this definition is a condensed repeat of the register table in `data_flow.md` and is a candidate for further shortening but not removal.

## VERDICT
- Crucial updates: no
