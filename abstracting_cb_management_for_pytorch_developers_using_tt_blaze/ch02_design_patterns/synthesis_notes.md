# Chapter 2 Synthesis Notes

This document records the provenance of each section in the synthesized Chapter 2 files, documenting which candidate version contributed what material and what changes were made relative to the source.

---

## Structural Decisions

**Base structure:** V5 was used as the structural base for all files, providing:
- DP1-DP16 sequential numbering (4 DPs per file)
- The adopt/adapt/reject decision matrix format
- The 12-decision coverage matrix format
- The "What you will learn" bullet format for each file
- The Key Takeaways / Source Files / Navigation footer pattern

**DP3 reclassification:** All five evaluations flagged V5's classification of DP3 (`__torch_dispatch__`) as "Reject." The synthesized version changes this to **"Adapt"** based on evaluator consensus that `__torch_dispatch__` serves as a valuable secondary entry point for eager-mode integration, even though TensorAdapter's primary path is graph/compilation-time analysis.

**Padding table consolidation:** Evaluators flagged repetition of the full padding value table across multiple files. The synthesis places a brief illustrative table in File 03 (Triton/XLA section, where per-op padding is first discussed) and a complete table in File 05 (principles), with forward references to the canonical table location in Chapter 5.

**PROPOSED code disclaimers:** Per evaluator feedback, all PROPOSED code blocks include "(illustrative, not prescriptive)" to avoid pre-constraining Chapter 3's concrete design.

---

## File 01: TT-Symbiote Module Replacement (DP1-DP4)

| Section | Primary Source | Additional Sources | Changes |
|---------|---------------|-------------------|---------|
| Module replacement algorithm | V1 (detailed 4-step algorithm) | V5 (structure) | Combined V1's procedural detail with V5's DP numbering |
| `TorchTTNNTensor` lifecycle | V2 (4-state table, dual-backing detail) | V4 (architectural invariants reference) | Used V2's four-state table verbatim; added V2's `compose_transforms` pipeline |
| `__torch_dispatch__` dispatch | V2 (can_dispatch_to_ttnn routing diagram) | V5 (DP numbering) | Changed from V5's "Reject" to "Adapt" per evaluator consensus |
| `_bypass_tensor_wrapping` | V2 (standard vs bypass path comparison) | -- | Included V2's optimization detail as a subsection |
| Four-phase lifecycle | V5 (DP4 structure) | V2 (preprocessing flag warning) | Added V2's warning about `_preprocessed_weight` flag race condition |
| "What Symbiote Does Not Solve" | V3 (5 specific CB-control-gap limitations) | V5 (table format) | Used V3's enumerated list of 5 limitations with V5's table |
| Pluggable dispatcher system | V2 (4 dispatcher types table) | -- | Included V2's dispatcher table and the CPU-default warning |
| Source Files section | V5 (format) | V2 (file paths) | Combined |

---

## File 02: TVM Relay/TIR Scheduling (DP5-DP8)

| Section | Primary Source | Additional Sources | Changes |
|---------|---------------|-------------------|---------|
| Two-Level IR overview | V5 (structure, DP numbering) | V1 (table format) | Used V5's DP5-DP8 numbering with V1's comparison table |
| Schedule primitives example | V4 (TVM code) | V5 (mapping table) | Combined V4's code with V5's Blaze mapping table |
| Buffer allocation with scopes | V4 (address space table) | V5 (DP6 structure) | Used V4's explicit MLIR-to-TT mapping |
| StorageRewrite parallel | V1 (mathematical equivalence argument) | V3 (actual `compact_cb_ids` code) | Used V3's actual Blaze code with V1's side-by-side comparison table |
| Predicated access warning | V2 (unique contribution) | -- | Included V2's warning that Tenstorrent tiles do not support per-element predicate masks |
| L1 budget calculation | V1 (detailed numerical example) | V5 (DP7 structure) | Used V1's bfloat16/bfloat8_b matmul example |
| Validation pipeline (DP8) | V5 (pipeline diagram) | V1 (contrast with runtime-only detection) | Combined V5's proposed pipeline with V1's observation about current lack of compile-time validation |
| TVM limitations table | V3 (7-row comparison table) | V5 (format) | Used V3's detailed comparison |

---

## File 03: Triton and XLA Patterns (DP9-DP12)

| Section | Primary Source | Additional Sources | Changes |
|---------|---------------|-------------------|---------|
| Triton model overview | V5 (simplified matmul kernel) | V4 (block pointer coverage) | Combined |
| DP9: Mask-based padding | V5 (structure) | V2 (properties list) | Used V5's four properties with V2's fill value examples |
| PaddingPolicy example | V5 (brief illustrative code) | -- | Kept brief; added forward reference to Ch5 canonical table to reduce repetition |
| DP10: Constexpr block sizes | V5 (Triton-to-Blaze mapping table) | V4 (autotune example) | Combined |
| Triton block pointers | V4 (unique contribution) | -- | Included V4's `tl.make_block_ptr` subsection with the CBHandle parallel |
| DP11: XLA automatic padding | V5 (HLO example) | V1 (XLA Behavior / TensorAdapter Equivalent table) | Combined |
| BroadcastInDim | V4 (unique contribution) | -- | Included as a subsection showing explicit shape alignment |
| DP12: Shape tracking | V5 (ShapeDescriptor triple-shape invariant) | V2 (propagation requirement) | Combined |
| 64-CB fusion limiter | V3 (gated MLP CB count breakdown) | V5 (brief mention) | Used V3's detailed analysis showing 15-20 CBs for one MLP block |
| CB elision | V3 (cb_alias mechanism) | V5 (brief mention) | Used V3's `cb_alias` code reference and optimization concept |
| Two Poles of Control table | V5 (unique contribution) | V4 (tiered API) | Combined V5's 5-dimension comparison with V4's tiered API proposal |
| Tiered API proposal | V4 (3-tier table: XLA-style, Triton-style, Blaze-style) | V5 (Level numbering) | Used V4's table format with V5's level numbering |

---

## File 04: MLIR and torch.compile (DP13-DP16)

| Section | Primary Source | Additional Sources | Changes |
|---------|---------------|-------------------|---------|
| MLIR dialect stack | V5 (progressive lowering diagram) | V4 (detailed linalg/memref code), V1 (metadata preservation principles) | Combined V5's structure with V4's concrete MLIR code examples |
| linalg dialect example | V4 (MLIR code) | V5 (Blaze MicroOp parallel) | Combined |
| Tiling transform | V4 (scf.for tiled matmul code) | V1 (tile-pad-promote 3-step pattern) | Used V4's code with V1's conceptual framing |
| Bufferization section | V4 (memref code with address space table) | V2 (bufferization decision mapping table) | Combined V4's code with V2's decision parallel |
| memref vs CB comparison table | V3 (6-row property comparison) | V4 (address space table) | Used V3's detailed comparison including flow control gap analysis |
| Hypothetical CB dialect | V3 (unique contribution: `cb.allocate`, `cb.push_back`, `cb.wait_front`, `cb.reconfigure`) | -- | Included V3's MLIR-style notation with `!cb.handle` types; added numbered list explaining what each operation captures |
| DP13: Dialect-like layers | V5 (3-layer architecture) | V4 (partial lowering advantage) | Combined V5's Tensor/Tile/CB layers with V4's partial lowering subsection |
| torch.compile pipeline | V5 (pipeline diagram) | V4 (FX graph code example) | Combined |
| DP14: FX graph capture | V5 (DP structure) | V4 (FX-to-BlazeGraph comparison table), V3 (from_fx_graph illustrative code) | Combined V5's structure with V4's table and V3's proposed code |
| Fusion boundary cost analysis | V3 (unique contribution: ~4us per tensor, ~40% overhead, 3-tier cost table) | V5 (boundary mention) | Used V3's FusionRegion concept and cost numbers with V5's classification of boundary types |
| DP15: Pluggable backend | V5 (DP structure) | V4 (Inductor tile heuristic code) | Combined; included V4's Inductor heuristic code and mapping table |
| Inductor memory planning | V3 (simplified greedy algorithm code) | V1 (parallel to compact_cb_ids) | Combined |
| TT-Forge section | V5 (TT-Forge pipeline) | V3 (where Forge stops / TensorAdapter begins comparison), V4 (warning about non-competing systems) | Combined V5's pipeline with V3's side-by-side and V4's warning blockquote |
| DP16: Compiler + manual | V5 (unique contribution) | V4 (analyze_graph illustrative code) | Combined V5's conceptual framing with V4's proposed code |
| 3-way comparison table | V5 (torch.compile / TT-Forge / TensorAdapter) | V4 (added CB-level control row) | Extended V5's table with escape hatch detail |

---

## File 05: Synthesized Design Principles

| Section | Primary Source | Additional Sources | Changes |
|---------|---------------|-------------------|---------|
| 7 principles (P1-P7) | V5 (principle numbering and names) | V4 (principle phrasing), V2 (per-framework evidence tables), V1 (per-principle evidence tables) | Merged V5's directive names with V2/V1 evidence tables; kept V4's tiered API framing for P4 |
| P1: Separation | V5 | V4 (evidence table) | Added V4's evidence table |
| P2: Dual-world metadata | V5 | V2 (per-framework logical/physical table), V1 (ShapeDescriptor code) | Combined V2's evidence with V1's code example |
| P3: Per-step validation | V5 | V2 (per-framework lowering levels table) | Added V2's evidence table |
| P4: Defaults with overrides | V5 | V4 (tiered API table: XLA/Triton/Blaze-style) | Used V4's 3-tier table as the core framing |
| P5: Graph-level analysis | V5 | V3 (format negotiation detail) | Added V3's cross-op decision examples |
| P6: Per-op padding | V5 (fill value table) | V1 (PaddingRegistry code), V2 (per-framework evidence table) | Combined; forward-referenced Ch5 for canonical table |
| P7: Transparent interception | V5 | V1 (adoption barrier argument) | Added V1's argument about developer adoption cost |
| Decision matrix | V5 (adopt/adapt/reject format) | Evaluator consensus | Changed DP3 from Reject to Adapt; updated summary counts |
| 12-decision coverage matrix | V5 (unique contribution) | V1 (broader column set) | Used V5's matrix structure with V1's column breadth |
| 6 architectural invariants | V4 (unique contribution) | V3 (64-CB analysis) | Used V4's invariant set; added warning about runtime-only detection |
| 5 novel capabilities | V3 (unique contribution) | V5 (gap analysis) | Used V3's enumerated list verbatim |
| Gap summary ASCII diagram | V3 (unique contribution) | -- | Used V3's diagram as-is |
| Comprehensive comparison table | V3 (8-column format) | V4 (dimension names), V1 (developer effort rows) | Synthesized from all versions; added TensorAdapter column |
| Architectural blueprint | V5 (unique contribution) | V3 (pipeline diagram) | Used V5's principle-annotated pipeline |
| Principle-to-chapter mapping | V5 (table format) | V4 (component names) | Combined |

---

## Cross-File Consistency Checks

1. **DP numbering**: Files 01-04 each contain exactly 4 DPs (DP1-4, DP5-8, DP9-12, DP13-16). File 05's decision matrix references all 16.
2. **Forward references**: All `[Chapter N: ...]` references use the `../chNN_name/index.md` format consistently.
3. **Navigation footers**: Each file ends with `**Next:** [NN -- Title](./NN_filename.md)` linking to the next file, with File 05 linking to Chapter 3.
4. **Code annotations**: All existing code uses `# EXISTING` comments; all proposed code uses `# PROPOSED` with "(illustrative, not prescriptive)" disclaimers.
5. **Warning blockquotes**: Each file contains at least one `> **Warning:**` blockquote for correctness-critical notes.
6. **Source Files**: Each file ends with a Source Files section listing framework-specific and Blaze-parallel file references.
