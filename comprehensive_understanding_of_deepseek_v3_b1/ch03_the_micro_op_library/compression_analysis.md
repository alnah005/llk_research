# Compression Analysis — Chapter 3: The Micro-Op Library

## Summary

| File | Lines | Estimated Compressible Lines | Notes |
|---|---|---|---|
| `01_compute_micro_ops.md` | 816 | ~105 | Repeated structural boilerplate across 8 op sections; summary table restates content |
| `02_data_movement_micro_ops.md` | 581 | ~65 | Deep dive on dram_streaming_matmul is well-structured but has verbose preambles; some restated cross-chapter content |
| `03_communication_and_infrastructure_micro_ops.md` | 837 | ~130 | Most compressible file: two summary tables restate each other; cross-cutting patterns section restates Chapter 2 material; MoE gate explanation overlaps with Chapter 1 |
| **Total** | **2234** | **~300** | |

**Overall compression potential: ~13%.**

## Load-Bearing Evidence

- `01_compute_micro_ops.md` line ~90: "The kernel uses `custom_mm_block`, an LLK that programs the Tensix MOP (Micro-Operation Processor) for the matmul inner loop" -- load-bearing because this is the first and only place that defines what MOP is and how `custom_mm_block` drives it, with the accompanying code and template parameter table being the primary reference for the matmul inner loop.

- `02_data_movement_micro_ops.md` line ~386: "The pre-shuffle is the critical insight. In the original row-major layout, tiles for different N columns of the same K row are contiguous, which is wrong for the streaming pattern." -- load-bearing because this explains the non-obvious column-major tile reorder that makes DRAM streaming feasible, with the before/after diagram being the only place this layout transformation is documented.

- `03_communication_and_infrastructure_micro_ops.md` line ~396: "Workers are classified as 'Type A' (forward-first) or 'Type B' (backward-first) based on `(row + worker_idx) % 2`." -- load-bearing because this is the sole explanation of the Type A/B alternation pattern that achieves balanced fabric link utilization in the SDPA cross-device reduction, and the formula is not documented elsewhere.

## CRUCIAL Suggestions

### `03_communication_and_infrastructure_micro_ops.md` ~lines 773-798 (Communication Complexity Ladder)

**Issue:** The "Communication Complexity Ladder" table at the end of the file restates information already presented in the "Multi-Device Equivalence Table" (lines 773-784) and in each op's individual section. The ladder adds columns for "Devices", "Topology", "Synchronization", and "Key Pattern", but every value in those columns was already stated in the corresponding op section (e.g., ccl_all_reduce section already states "2-device linear mesh", "2 global semaphores", "neighbor exchange + local sum"). The equivalence table likewise restates each op's purpose in a less precise way (e.g., "ccl_broadcast" mapped to "mcast" with "Both replicate data from one source to many destinations" -- already stated in the Purpose line of each section).

**Suggestion:** Merge the two tables into a single summary table at the end of the file. Remove the "Multi-Device Equivalence Table" entirely (5 lines of content that adds little beyond what the reader already absorbed from the per-op sections). For the Communication Complexity Ladder, keep only columns that are not already stated in each op's section header -- specifically, remove "Key Pattern" (verbatim from each section) and "Topology" (stated in each section's diagrams). The merged table should be ~10 lines instead of ~25.

### `03_communication_and_infrastructure_micro_ops.md` ~lines 652-698 (deepseek_moe_gate hierarchical selection)

**Issue:** The "Why Hierarchical?" subsection (lines 683-689) explains three reasons for the hierarchical top-8 selection: compute efficiency, group diversity, and bias separation. The compute efficiency point ("Sorting 16 elements per group ... is much cheaper than sorting 256 elements") restates what the algorithm description in lines 659-681 already implies by showing the two-level structure. The bias separation point ("The selection uses biased scores ... but the final weights use unbiased scores") is a near-verbatim restatement of what lines 668 and 679-680 already say ("Take the top-2 bias scores and their corresponding **original** (pre-bias) scores" and "Map back to original scores (not bias scores)"). Furthermore, the entire hierarchical selection algorithm description overlaps significantly with Chapter 1, Section 1.1.2 (lines 78-91 of `01_deepseek_v3_architecture_on_tenstorrent.md`), which already covers sigmoid scoring, top-8 selection, and score normalization with the same mathematical notation.

**Suggestion:** (a) Remove the "Why Hierarchical?" subsection entirely (7 lines). The compute efficiency is self-evident from the algorithm structure, and the bias separation is already stated twice in the algorithm steps. (b) In the algorithm description itself, add a one-line cross-reference to Chapter 1 Section 1.1.2 ("See Section 1.1.2 for the model-level view of this gating algorithm") rather than re-deriving the sigmoid + bias + normalization from scratch. The hardware-specific details (single 16x16 face, register-level operations, custom LLK functions) are unique and should remain.

### `03_communication_and_infrastructure_micro_ops.md` ~lines 800-829 (Cross-Cutting Patterns section)

**Issue:** The "Cross-Cutting Patterns" section presents three patterns: MeshProgramDescriptor, Fabric Connection Setup, and Socket-Based Persistent Kernels. The MeshProgramDescriptor code example (lines 806-816) is nearly identical to the code example in Chapter 1, Section 2.1 (`02_tt_blaze_execution_pattern.md`, lines 190-198 of that file), which shows the same `MeshProgramDescriptor` loop pattern with the same variable names. The Fabric Connection Setup pattern (lines 818-829) shows a `setup_routing_plane_connection` call that is already described in the ccl_all_reduce section (line 108: "Each device uses `setup_routing_plane_connection` three times") and ccl_broadcast section. The Socket-Based Persistent Kernels pattern (lines 831-833) restates what the d2d_exchange section already says ("The `d2d_exchange` kernel runs in an infinite loop, polling the termination semaphore").

**Suggestion:** Remove this entire "Cross-Cutting Patterns" section (30 lines). Replace with a single sentence: "All multi-device ops follow the `MeshProgramDescriptor` and fabric connection patterns described in Section 2.1; d2d_exchange, host_io, and pipeline_block are persistent kernels terminated via global semaphore (see their individual sections above)."

## MINOR Suggestions

### `01_compute_micro_ops.md` ~lines 78, 416, 506, 608 ("Runtime Args: None")

**Issue:** Four ops (matmul, kn_sliced_matmul, local_reduce, eltwise_add) include a "### Runtime Args" heading followed by a single line "None" or "None -- all CB indices and dimensional parameters are compile-time args." This adds 8-12 lines across the file (heading + blank line + one-liner + blank line, four times) that convey a single bit of information.

**Suggestion:** Add a convention note at the top of the file: "Unless a Runtime Args section is present, assume all parameters are compile-time constants." Then remove the four empty Runtime Args sections. Saves ~10 lines with zero information loss.

### `01_compute_micro_ops.md` ~lines 800-813 (Summary Table)

**Issue:** The Summary Table at the end restates the Purpose and Key Mechanism of all 8 ops that were already introduced in their individual sections. The "TRISC Complexity" column (High/Medium/Low/None) is new information, but the "Computation" and "Key Mechanism" columns are condensed restatements.

**Suggestion:** Keep the table but remove the "Computation" column (it exactly matches each op's "### Computation" section heading content). The "Key Mechanism" and "TRISC Complexity" columns add navigational value and should remain.

### `02_data_movement_micro_ops.md` ~lines 1-4 (preamble)

**Issue:** The opening paragraph ("Data movement micro-ops shuttle tiles between cores, between L1 and DRAM, and between logical tensor shapes and physical memory layouts. Their TRISC is either a no-op or performs only tilize/untilize -- no FPU arithmetic. What makes them architecturally interesting is the NOC programming...") contains hedging language ("What makes them architecturally interesting") and could be tightened.

**Suggestion:** Replace with: "Data movement micro-ops transfer tiles between cores, L1, and DRAM, and convert between logical tensor shapes and physical memory layouts. TRISC is either idle or performs tilize/untilize only. The NOC programming -- multicast addressing, semaphore choreography, dual-NOC routing, and DRAM bank assignment -- is the primary complexity."

### `02_data_movement_micro_ops.md` ~lines 568-578 (Summary Table)

**Issue:** Same pattern as the compute file: the summary table restates each op's direction, core count, and key mechanism. All values are already in the per-op sections.

**Suggestion:** Keep the table for navigation but tighten the "Key Mechanism" column to a keyword or two rather than a phrase (e.g., "dual-NOC, per-core index, separate semaphores" -> "dual-NOC routing"). Saves a few words per row.

### `01_compute_micro_ops.md` ~line 5 (stability note)

**Issue:** The boxed stability note about "Composed Into" sections being subject to change is useful but verbose: "The CB index assignments listed in each op's 'Composed Into' section reflect the codebase at the time of writing and are subject to change when fused ops are refactored. Treat them as navigational aids, not as stable API contracts."

**Suggestion:** Shorten to: "CB index assignments in 'Composed Into' sections are navigational aids, not stable API contracts."

### `01_compute_micro_ops.md` ~line 787 ("This is the simplest micro-op in the library.")

**Issue:** The sentence "This is the simplest micro-op in the library." in the tilize_8x32 section is an editorial judgment that adds no technical content.

**Suggestion:** Remove the sentence.

### `03_communication_and_infrastructure_micro_ops.md` ~lines 760-761 (deepseek_moe_gate Compute Config)

**Issue:** The Compute Config says: "Math fidelity: `HiFi4` (makes no practical difference since the operation is dominated by comparisons and reductions rather than multiply-accumulate)". The parenthetical is an aside that explains why a configuration choice does not matter -- low-value editorial.

**Suggestion:** Shorten to: "Math fidelity: `HiFi4` (operation is comparison-dominated; fidelity is irrelevant)."

### `03_communication_and_infrastructure_micro_ops.md` ~lines 597-600 (pipeline_block Computation section)

**Issue:** The Computation section for pipeline_block says "Not applicable -- `PipelineBlock` is an infrastructure orchestrator, not a compute op." This is self-evident given the section title and purpose.

**Suggestion:** Remove the "### Computation" heading and its one-liner entirely for pipeline_block, or replace with a single word: "None."

### `01_compute_micro_ops.md` recurring pattern: "BRISC | No-op."

**Issue:** In the RISC Roles table for matmul (line 85), rope (line 339), kn_sliced_matmul (line 423), local_reduce (line 512), the BRISC row says "No-op." identically each time. This is correct but repetitive given the introductory paragraph already says "NCRISC and BRISC handle sharded-buffer signaling or remain no-ops."

**Suggestion:** This is a minor structural repetition inherent to the per-op template format. No change needed -- each op must stand alone as a reference. Noting for completeness only.

## VERDICT

Crucial updates: yes

## Pass 2

### Summary

| File | Lines | Change from Pass 1 |
|---|---|---|
| `01_compute_micro_ops.md` | 816 | 0 (unchanged) |
| `02_data_movement_micro_ops.md` | 581 | 0 (unchanged) |
| `03_communication_and_infrastructure_micro_ops.md` | 784 | -53 (down from 837) |
| **Total** | **2181** | **-53** |

**CRUCIAL resolution status from Pass 1:**
1. Two summary tables (Multi-Device Equivalence + Communication Complexity Ladder) merged into one: **RESOLVED** -- Lines 767-778 now contain a single "Communication Summary" table with columns Micro-Op, Devices, Topology, Synchronization, and Single-Device Equivalent. The old separate "Multi-Device Equivalence Table" and "Communication Complexity Ladder" are both gone. The merged table retains all non-redundant columns from both predecessors.
2. "Why Hierarchical?" subsection removed, cross-reference to Ch1 added: **RESOLVED** -- No "Why Hierarchical?" subsection exists anywhere in the file. Line 659 contains the cross-reference: `See [Section 1.1.2](...) for the model-level view of this gating algorithm.` The hardware-specific algorithm steps (Level 1/2/3, normalization) remain intact, which is correct since those detail the on-chip implementation.
3. Cross-Cutting Patterns section replaced with a single sentence: **RESOLVED** -- Line 780 contains a single sentence covering MeshProgramDescriptor, fabric connections (with a link to Section 2.1), and persistent kernel termination. The old 30-line "Cross-Cutting Patterns" section with its redundant code example and per-pattern descriptions is entirely removed.

### Load-Bearing Evidence

- `03_communication_and_infrastructure_micro_ops.md` line 780: "All multi-device ops follow the `MeshProgramDescriptor` and fabric connection patterns described in [Section 2.1](../ch02_the_unified_kernel_system/01_unified_kernel_descriptor.md); `d2d_exchange`, `host_io`, and `pipeline_block` are persistent kernels terminated via global semaphore (see their individual sections above)." -- This single sentence is the replacement for the old Cross-Cutting Patterns section; it preserves the three key facts (MeshProgramDescriptor pattern, fabric connections, persistent kernel termination) while eliminating the redundant code example and verbose per-pattern descriptions.
- `03_communication_and_infrastructure_micro_ops.md` line 659: "See [Section 1.1.2](../ch01_architecture_overview_and_model_mapping/01_deepseek_v3_architecture_on_tenstorrent.md) for the model-level view of this gating algorithm." -- Confirms the cross-reference was added as specified, and the surrounding algorithm description (lines 660-682) retains only hardware-level detail (single 16x16 face, per-group sort, register-level operations).
- `03_communication_and_infrastructure_micro_ops.md` lines 767-778: The merged Communication Summary table contains 8 rows (one per op) with 5 columns. No information was lost from the old two-table layout; the "Key Pattern" column (which was flagged as redundant with per-op sections) has been correctly omitted.

### CRUCIAL Suggestions

None. The three Pass 1 CRUCIAL items were all resolved correctly. No new CRUCIAL issues were introduced by the fixes. The 53-line reduction is consistent with the estimated ~130 compressible lines for this file (the remaining ~77 lines of potential compression correspond to the MINOR suggestions from Pass 1, which were not required to be addressed).

### MINOR Suggestions

1. **`03_communication_and_infrastructure_micro_ops.md` line 754** -- The Compute Config parenthetical for deepseek_moe_gate still reads: `(makes no practical difference since the operation is dominated by comparisons and reductions rather than multiply-accumulate)`. This was flagged in Pass 1 as MINOR. Suggested shortening: `(comparison-dominated; fidelity irrelevant)`.

2. **`03_communication_and_infrastructure_micro_ops.md` lines 597-598** -- The pipeline_block Computation section still contains `Not applicable -- 'PipelineBlock' is an infrastructure orchestrator, not a compute op.` This was flagged in Pass 1 as MINOR. Could be shortened to `None.` or removed entirely.

3. **`01_compute_micro_ops.md` line 787** -- The editorial sentence `This is the simplest micro-op in the library.` in the tilize_8x32 section remains. Flagged in Pass 1 as MINOR.

## VERDICT

Crucial updates: no
