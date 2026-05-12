# Chapter 4 Synthesis Notes

## Base Version

**V4** was selected as the structural base via Borda count ranking (21 points, highest across 5 evaluators). V4 scored 9.30/10 overall, with 10/10 on coverage of the plan spec's four files.

## Trimming Strategy

V4's original total was ~3,691 lines across 4 files. The synthesis target was 2,400-2,800 total lines (550-700 per content file), representing a 10-15% trim. Trimming focused on:

1. **Removing redundant code comments** that restated the logic already explained in prose
2. **Consolidating error-condition tables** where V4 had both inline and end-of-section tables covering the same conditions
3. **Shortening validation code blocks** to show the essential logic without full docstrings when the surrounding prose already explains the intent
4. **Removing the full _build_padded_shape() post-condition check** (the assertion block) since it duplicates the validation in _compute_padded_dims()
5. **Compressing edge-case catalog entries** where V4 spent 10+ lines on cases that need only 3-4 lines

## Cherry-Picks Integrated

### From V2: Storage-Tile vs. Compute-Tile Naming (File 01)

V2's dedicated section explaining that data is stored in L1 as contiguous 1x32 rows (the "storage tile") but the FPU reinterprets them as either 16x32 or 32x32 tiles (the "compute tile") was added to File 01. This distinction was flagged as a gap by 4 of 5 evaluators. The section is also referenced in File 02 when explaining normalization tile selection.

### From V3: Dual-Model Examples (All Files)

V3's DeepSeek V3 examples (hidden dim 7168, MoE expert width 2048) were integrated throughout, alongside the existing LLaMA-7B examples. Specific additions:
- **File 01:** Worked examples for `[1, 7168]` and `[7168, 2048]` shapes
- **File 02:** Norm tile walkthrough for width=7168; already-aligned dimension examples for 7168 and 2048
- **File 03:** Complete DeepSeek V3 MoE expert matmul propagation walkthrough (5 steps: gate matmul -> SiLU -> up matmul -> eltwise multiply -> down projection)
- **File 04:** DeepSeek V3 shard planning walkthrough showing how 64 = 2^6 provides grid-friendly divisors

### From V3: Prime-Number Factorization Problem (File 04)

V3's analysis of `344 = 2^3 * 43` as a sharding bottleneck for LLaMA-7B was integrated into File 04 as a dedicated section ("The Prime-Number Factorization Problem"). This contrasts with DeepSeek V3's `64 = 2^6` to illustrate how model dimension choices affect hardware mapping.

### From V5: Design Principle Callouts (All Files)

V5's pattern of referencing P1-P7 design principles as blockquotes was adopted. Each file includes 2-3 principle callouts as margin references:
- **File 01:** P2 (Carry Metadata from Both Worlds) -- for ShapeDescriptor design; P1, P3 also referenced
- **File 02:** P1 (Separate Intent from Strategy) -- tile_decompose() as pure function; P3 (Validate at Each Step) -- pipeline validation; P4 (Defaults with Overrides) -- tile registry + override
- **File 03:** P1 (Separate Intent from Strategy) -- developer chains ops, protocol fills hardware details; P3 (Validate at Each Step) -- CT arg cross-validation
- **File 04:** P1 (Separate Intent from Strategy) -- single compute_shard() replaces manual steps; P2 (Carry Both Worlds) -- global vs. shard CT arg distinction; P3 (Validate at Each Step) -- per-step validation in shard algorithm

### From V1: BFP Page-Size Breakdown Table (File 01)

V1's detailed byte breakdown for block floating-point formats was added to File 01:
- bfloat8_b: 1,024 mantissa bytes + 64 exponent bytes = 1,088 bytes total
- bfloat4_b: 512 mantissa bytes + 64 exponent bytes = 576 bytes total

The table covers all format x tile combinations and includes the exponent-overhead explanation that prevents the common 1,024-byte mistake.

### From V1: FIFO vs. DIRECT_ADDRESS Distinction (File 04)

V1's dedicated table and explanation of FIFO (ring buffer, new L1 allocation) vs. DIRECT_ADDRESS (pointer to existing L1 buffer) CB modes was integrated into File 04. This is a critical distinction that V4 mentioned but did not give dedicated coverage to.

### From V3: Broadcast Shape Inference Rule (File 03)

V3's `infer_broadcast()` function, which handles logical dimension expansion (size-1 dims expanding to match a target shape), was added to File 03. This was flagged as a gap in V4, which covered mcast/gather as data movement but did not have a separate broadcast inference rule for logical shape expansion.

### From V5: compute_subblock_w() Analysis (File 04)

V5's walkthrough showing that `out_w_per_core = 43` (prime) forces `subblock_w = 1` was integrated into File 04 with three walkthroughs:
- LLaMA-7B with 8 cores: out_w_per_core=43 -> subblock_w=1 (worst case)
- DeepSeek V3 with 16 cores: out_w_per_core=4 -> subblock_w=4 (excellent)
- LLaMA-7B with 4 cores: out_w_per_core=86 -> subblock_w=2 (improvement)

### DRAM Streaming Fallback (File 04)

Added dedicated section on DRAM streaming as a fallback when weight shards exceed L1. This was mentioned in V3 and V1 but not given structured coverage in V4. Includes:
- When it is needed (LLaMA gate weight ~5.7 MB per core with 8 cores)
- How K_BLOCK reduces CB size (172 tiles * 1,088 = 187 KB)
- Complete L1 budget breakdown with DRAM streaming
- Warning about k_num_tiles semantics changing in streaming context

## Consistency Checks

### bfloat8_b = 1,088 Bytes

Verified across all four files that bfloat8_b page size is consistently stated as 1,088 bytes (not 1,024). Appears in:
- File 01: BFP page-size reference table
- File 02: BFP_TILE_SIZES dictionary in _compute_page_size(), Warning blockquote
- File 04: All shard byte calculations use 1,088

### Cross-Chapter References

- File 01 references Chapter 3 File 02 (ShapeDescriptor definition) and Chapter 2 File 05 (design principles P1-P3)
- File 02 references File 01 (three shape layers, storage vs. compute tile distinction) and Chapter 3 File 01 (ShapeEngine architecture)
- File 03 references Chapter 3 File 02 (ShapeTracker), File 02 (norm tile selection function)
- File 04 references Files 01-03 (tile decomposition, shape propagation), Chapter 7 (L1 budget validator)
- Navigation footer on all files: Chapter 3 (previous) and Chapter 5 (next)

### Formatting

All four content files include:
- "What you will learn" section at the top
- `# EXISTING` and `# PROPOSED` markers in code blocks
- Key Takeaways section at the bottom
- Source Files section
- Warning blockquotes where appropriate
- Navigation footer: `← Previous: [Chapter 3] | Next: [Chapter 5] →`

## Known Tradeoffs

1. **File 04 length**: Even after trimming, File 04 is the longest because it must cover FIFO/DIRECT_ADDRESS, GridConfig, shard algorithm, prime factorization, compute_subblock_w(), DRAM streaming, and two full worked examples. Splitting into two files was considered but rejected to match the plan spec's four-file structure.

2. **DeepSeek V3 attention walkthrough omitted**: V3 included a DeepSeek V3 Q/K/V attention walkthrough in File 03. This was omitted from the final synthesis because (a) the MoE expert walkthrough already provides DeepSeek V3 coverage in File 03, (b) the attention walkthrough adds ~50 lines, and (c) SDPA shape propagation is architecturally simpler than the gated MLP chain (output matches Q shape). The attention pattern is better suited to Chapter 8 (attention-specific ops).

3. **V4's 8-rule class structure preserved**: V4 defines 8 separate ShapeInferenceRule classes (Matmul, ElementwiseUnary, ElementwiseBinary, Reduction, GatedReduce, Norm, DataMovement, SDPA). V3 used a flatter function-based approach. The class-based approach was preserved because it aligns with the SHAPE_INFERENCE_REGISTRY pattern and makes extension easier.

4. **Broadcast inference added as standalone function, not registered class**: The broadcast rule from V3 was added as `infer_broadcast()` rather than as a registered `BroadcastShapeRule` class, because broadcast is typically implicit in binary elementwise ops (handled by _broadcast_shapes inside ElementwiseBinaryShapeRule) rather than a standalone op. This matches the existing Blaze architecture where mcast is data movement, not a logical broadcast.
