# Chapter 5 Synthesis Notes

## Base Version

**V3** was selected as the base for all four files. V3 earned the strongest evaluation for source-code grounding (explicit connection to existing Blaze internals like `padded_rmsnorm/op.py` CT args, SDPA `identity_scalar_packed`/`zero_scalar_packed` constants), consistency with prior chapters (P1-P7 principle references, ShapeDescriptor fields from Ch03, tile decomposition terminology from Ch04), and a well-balanced depth-to-length ratio. V3's unique contributions retained in the final synthesis include the compatibility matrix, `output_padding_propagation` on OpSpec, and `encode_fill_value()` validated against SDPA constants.

## Cherry-Picks by Version

### From V1

- **Numerical corruption examples with quantified error rates** (File 01): V1 provided the most compelling "why this matters" argument through concrete error numbers -- 45.6% softmax probability leak, 84.4% mean reduction error, 31.5% RMS normalization error. These were integrated into File 01's wrong-padding consequence sections to make the mathematical argument visceral rather than abstract.

### From V2

- **INHERIT policy** (File 02): V2 introduced the 6th PaddingPolicy variant for data-movement ops (multicast, gather, copy, pad, nop) that simply pass through whatever padding their input carries. This was added to the PaddingPolicy enum, PaddingSpec dataclass, compatibility matrix, and registry table throughout File 02. The INHERIT policy fills a genuine gap: without it, every data-movement op would need to re-declare its upstream producer's padding strategy.
- **Wildcard `"*"` fallback for registry entries** (File 02): V2's pattern for ops with many identically-padded ports (e.g., `"*": PaddingSpec(PaddingPolicy.ZERO)` instead of listing each port) was integrated into the registry lookup logic and explicitly shown in the `_resolve_padding_spec()` function.
- **NCRISC execution model for PadOp** (File 03): V2 provided the clearest explanation of why the PadOp kernel runs on NCRISC (data movement pipeline) rather than SFPU (compute pipeline), emphasizing zero SFPU cycles and the BRISC-initiated, NCRISC-executed fill pattern.

### From V4

- **Three-level validation** (File 02): V4's validation framework -- registration-time (type checks, fill value compatibility), pass-time (edge-by-edge compatibility), and golden-reference (numerical correctness against PyTorch) -- was integrated into File 02's validation section. This adds rigor that V3 lacked.
- **Stale padding detection** (File 04): V4's debug-mode validator that checks padding positions still contain expected fill values was added to File 04. This catches subtle bugs where intermediate kernels write to padding positions, corrupting downstream computations.
- **PaddingPassStats and PaddingPassResult** (File 03): V4's diagnostic dataclasses that report how many PadOps and RePadOps were inserted, how many were skipped by optimization, and the overall padding ratio were integrated into File 03's return types.

### From V5

- **Consumer-determines-padding principle** (File 01): V5 articulated this principle most clearly. Rather than burying it in the registry design, V5 elevated it to a first-class concept with a directional diagram showing the fill-value decision flow. This framing was integrated into File 01's opening P6 callout and the "consumer determines padding" subsection.
- **Cost model table for PadOp and RePad** (File 03): V5 provided a concise 7-scenario optimization table showing when PadOps can be skipped, which was integrated as a replacement for V3's more verbose per-scenario narrative.
- **STRICT_PADDING_REGISTRY mode for CI** (File 02): V5's CI integration concept -- a mode where any unregistered op causes a hard failure rather than a warning -- was added to File 02. This ensures the registry stays complete as new ops are added.

## Deviations from Plan Spec

1. **INHERIT as 6th PaddingPolicy**: The plan spec (line 216) defines five policies: ZERO, NEG_INF, MASK, IDENTITY, NONE. The synthesis adds INHERIT as a 6th policy. This deviation is deliberate: data-movement ops (multicast, gather, copy) are pervasive in the graph, and without INHERIT the registry would need to duplicate padding declarations that should simply pass through from the producer.

2. **File 03 length reduction**: V3's File 03 was 1,030 lines, well above the 550-700 target. The synthesis trimmed it to 620 lines by consolidating redundant walkthroughs, replacing verbose per-scenario narratives with a summary table, and removing repeated code patterns where a single example sufficed for the pattern.

3. **PaddingChainTracker scope**: The plan spec mentions "track the padding metadata through the op chain" in the File 04 description (line 234). The synthesis introduces PaddingChainTracker in File 03 (where padding decisions are made) and references it in File 04 (where it informs unpadding). This split across files was a judgment call -- the tracker is conceptually part of propagation (File 03) but operationally consumed by unpadding (File 04).

4. **Host-side unpadding as strong default**: The plan spec says "Performance consideration: unpadding is a host-side operation that should not add device overhead" (line 235). The synthesis goes further by providing a concrete `should_unpad_on_device()` heuristic with thresholds (>10 MB and >50% padding ratio), making the host-side preference actionable rather than advisory.

5. **Three-level validation not in plan**: The plan spec does not mention validation levels for the registry. The synthesis adds this from V4 because correctness guarantees require more than just a lookup table -- they require validation at registration, insertion, and testing boundaries.

## Cross-Chapter Dependencies

### From Chapter 2 (Design Patterns)

- **P1 (Separate Computation Intent from Execution Strategy)**: Referenced in Files 01 and 04. The core argument: padding is an execution strategy that must be invisible to the PyTorch developer.
- **P2 (Dual-World Metadata)**: Referenced in File 01. ShapeDescriptor carries both logical and padded shape, serving as the bridge between PyTorch's view and the device's tiled reality.
- **P3 (Validate at Each Lowering Step)**: Referenced in File 03. The padding pass is one such lowering step, with pre/post validation.
- **P4 (Provide Defaults with Per-Decision Overrides)**: Referenced in File 02. The registry provides defaults; the developer can override per-tensor or per-op.
- **P6 (Per-Operation Padding Semantics)**: The primary design principle for Chapter 5. Referenced in Files 01, 02, and 03.
- **P7 (Transparent Interception)**: Referenced in File 04. Unpadding makes the padding system invisible at the output boundary.

### From Chapter 3 (Tensor Adapter)

- **ShapeDescriptor**: Files 01-04 use ShapeDescriptor's fields (logical_shape, padded_shape, tile_shape, data_format, page_size, padding_strategy, padding_fill) as the metadata carrier. Chapter 5 adds padding_strategy and padding_fill to the existing dataclass definition from Ch03.
- **ShapeTracker**: File 04 references ShapeTracker for intermediate result lookup during debugging.
- **TypedHandle (CBHandle + ShapeDescriptor)**: Files 03-04 reference TypedHandle as the wrapper that carries ShapeDescriptor through the graph.

### From Chapter 4 (Tile Decomposition)

- **Three shape layers**: File 01 uses the logical_shape -> padded_shape -> tile_grid_shape progression defined in Ch04 File 01.
- **bfloat8_b page_size = 1,088 bytes**: Used consistently (1,024 data + 64 header) wherever page size calculations appear.
- **Shape inference rules**: File 03's padding pass runs after Ch04's shape inference. The matmul, elementwise, and reduction shape rules from Ch04 File 03 determine the padded_shape that the padding pass then annotates with fill values.

### Forward References to Later Chapters

- **Chapter 6 (Data Format Selection)**: File 01 references format-specific fill value encoding (bfloat16 bit patterns, packed scalar constants). Chapter 6 will define the complete data format system.
- **Chapter 7 (CB Allocation)**: File 03's PadOp consumes a CB, and the padding pass must coordinate with CB allocation. The files reference "before CB allocation" as the pass ordering constraint.
- **Chapter 8 (Compilation Pipeline)**: File 04 references `BlazeCompiler.compile()` as the point where output ShapeDescriptors are finalized.

## Line Counts

| File | Lines | Target | Status |
|------|-------|--------|--------|
| 01_padding_strategy_taxonomy.md | 572 | 550-700 | Within range |
| 02_op_padding_registry.md | 725 | 550-700 | Slightly over (registry table is dense) |
| 03_automatic_padding_insertion_and_propagation.md | 620 | 550-700 | Within range |
| 04_output_unpadding.md | 603 | 550-700 | Within range |
| **Total** | **2,520** | **2,200-2,800** | **Within range** |

File 02 exceeds the per-file target by 25 lines. The excess comes from the registry table (every Blaze op with per-port padding specs) and the compatibility matrix, both of which are reference material that would lose value if truncated.

## Formatting Compliance

All four content files include:
- "What you will learn" section after the opening paragraph
- `# EXISTING` / `# PROPOSED` markers on all code blocks
- Key Takeaways section (6 bullet points each)
- Source Files section referencing actual Blaze source paths
- Warning blockquotes (2-4 per file) for correctness pitfalls
- Navigation footer with Previous/Next links

## Known Limitations

1. **No runtime benchmarks**: All performance numbers (PCIe transfer times, kernel execution estimates) are analytical calculations, not measured benchmarks. The P150 hardware was not available for validation.

2. **Registry completeness**: The PADDING_REGISTRY covers all ops mentioned in the Blaze source code examined during research. New ops added after the research window would need registry entries added. The STRICT_PADDING_REGISTRY mode (from V5) addresses this by failing CI when an unregistered op is encountered.

3. **Reshape/transpose padding propagation**: File 03 includes a section on reshape/transpose handling that was identified as a gap in all five candidate versions. The synthesis provides the conceptual framework (reshape as logical reinterpretation, transpose as stride manipulation) but the implementation is less detailed than the matmul/elementwise paths because no existing Blaze reshape op was available for source-code grounding.

4. **Multi-device padding**: The files focus on single-device padding. Multi-device scenarios (T3K, Galaxy) where tensors are sharded across devices may require padding consistency across shards. This is flagged in File 04's device-side unpadding section but not fully designed.

5. **Dynamic shapes**: All examples use static shapes known at compile time. Dynamic sequence lengths (common in inference serving) would require the padding pass to emit shape-dependent PadOps, which is not addressed.
