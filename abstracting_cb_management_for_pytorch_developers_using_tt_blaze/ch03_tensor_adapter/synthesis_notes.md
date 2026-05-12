# Chapter 3 Synthesis Notes

## Base Version

**V5** was selected as the structural base per Borda count (23 points, highest among all five versions). V5 provided the strongest architectural framing ("Why Three Layers?"), the most complete ShapeDescriptor design (frozen, three factories, with_*() mutation methods), the best backward-compatibility story (from_cb_handle()), and the clearest hybrid compilation path example.

## Cherry-Picks Applied

### From V1 (Borda: 14)

1. **CBParams intermediate type** (File 01): Added as the boundary type between ShapeEngine output and CBBackend CB allocation. This provides a clean separation between shape-world and CB-world concerns. Source: V1 File 01, lines 596-644.

2. **Complete data flow trace** (File 01): The end-to-end numerical walkthrough from `BlazeModule.from_torch(nn.Linear(4096, 11008))` through all pipeline stages. Adapted to use correct bfloat8_b page size of 1,088 bytes. Source: V1 File 01, lines 650-708.

3. **TypedHandle __getattr__ delegation** (File 02): V1's transparent proxy pattern where TypedHandle delegates attribute access to the underlying CBHandle, enabling drop-in compatibility with existing emit() code. Source: V1 File 02/File 03, TypedHandle sections.

### From V3 (Borda: 15)

4. **11-node gated MLP shape propagation walkthrough** (File 02): All 11 nodes traced with full ShapeDescriptor at each step, demonstrating that the shape tracking system works for production-scale fused ops. Source: V3 File 02, lines 358-403.

5. **Edge cases section** (File 02): Scalar tensors, 1D tensors, tensors smaller than one tile (with "99.3% padding" warning), batch dimensions, and non-aligned shapes ([1, 37] for RMSNorm). Source: V3 File 02, lines 566-610.

6. **[A]/[B] annotation pattern** (File 03): Per-line labels distinguishing adapter-generated code [A] from existing Blaze code [B]. Applied throughout File 03's code examples. Source: V3 File 03, lines 406-428.

7. **MatmulAdapter delegation example** (File 03): Concrete class showing the adapter calling Matmul.emit() directly with shape tracking before and after. Source: V3 File 03, lines 436-459.

8. **"Why Not Add adapt() to BlazeOp" rationale** (File 03): Design decision explaining why shape awareness belongs on the adapter, not on the op base class. Source: V3 File 03.

### From V4 (Borda: 15)

9. **ShapeDescriptor __post_init__ validation** (File 02): Construction-time validation of rank consistency, padded >= logical, tile alignment, and total_tiles consistency. Source: V4 File 02, lines 95-141.

10. **Error pipeline examples** (File 01): Table showing what happens when invalid inputs hit each adapt() pipeline stage (scalar tensor, zero-size dim, unknown op_hint, format-tile mismatch, L1 overflow). Source: V4 File 01, lines 568-594.

11. **Four-level override hierarchy** (File 03): user_args > adapt() overrides > adapter defaults > Blaze defaults, with explanation. Source: V4 File 03, lines 410-442.

12. **Graceful degradation section with test code** (File 04): Backward-compatibility tests showing that unextended FusionResult/ExternalTensor still work, and that compile() falls back to manual user_args. Source: V4 File 04, lines 582-621.

13. **Conflict detection in _build_user_overrides** (File 04): Warning when shape-derived values conflict with user-provided values. Source: V4 File 04.

### From V2 (Borda: 8)

14. **ExternalTensor Option A/B discussion** (File 04): Rationale for choosing a dedicated `shape` field over the existing `metadata` dict. Incorporated as a design decision note. Source: V2 File 04.

15. **Compile-time only / zero runtime overhead statement** (File 01): Explicit statement that TensorAdapter is compile-time-only and the CompiledProgram is byte-identical to hand-written code. Source: V2 File 01.

16. **Two-path compilation comparison table** (File 04): Compact table comparing Graph API vs. Composition API on entry point, shape carrier, CT arg derivation, and best-for scenarios. Source: V2 File 04.

## Known Issues Fixed

### 1. BFP page_size (CRITICAL)

**Problem:** V2 and V3 computed bfloat8_b page size as `32 * 32 * 1 = 1024`, which is incorrect. The actual page size is ~1,088 bytes due to shared exponent overhead in block floating-point format.

**Fix:** All references to page_size now note the BFP exception. The worked example in File 01 uses 1,088 bytes for bfloat8_b weights. File 01's allocate_cb() section includes a Warning block explaining that `tile.get_tile_size(data_format)` must be used for packed formats. The ShapeDescriptor invariant comment no longer claims the simple multiplication formula applies to all formats.

### 2. FusedProgram Classification (plan spec discrepancy)

**Problem:** Plan spec line 136 labels FusedProgram as "EXTEND" but describes it as "gains no new methods." V5 and most evaluators classified it as WRAP.

**Fix:** File 03 classifies FusedProgram as WRAP with an explicit Note acknowledging the plan spec discrepancy and explaining that WRAP matches the described behavior (no new methods, adapter calls existing API).

### 3. Mutable ShapeDescriptor (V2)

**Problem:** V2 used a non-frozen ShapeDescriptor, which is a correctness risk for fan-out propagation.

**Fix:** The synthesis uses V5's `frozen=True` approach throughout, with `with_*()` mutation methods that return new instances. V2's mutable approach was not incorporated.

### 4. Missing Worked Numerical Example (V5 weakness)

**Problem:** V5 lacked concrete end-to-end numerical traces, which all evaluators flagged.

**Fix:** File 01 now includes a complete worked example tracing [1, 4096] x [4096, 11008] matmul through all six pipeline steps with specific values (padded_shape, tile_grid, page_size, num_pages, CT args). File 02 includes V3's 11-node gated MLP propagation walkthrough.

### 5. from_cb_handle() Shape Reconstruction Warning

**Problem:** from_cb_handle() assumes single-tile width for shape reconstruction, which may not match the original layout for multi-column tile grids.

**Fix:** File 02's from_cb_handle() Warning block is strengthened to note that the reconstruction formula is ambiguous without additional metadata, and explicitly calls out the single-tile-width assumption.

## Formatting Conventions

- **"What you will learn"** bullet list at the top of each file
- **EXISTING / PROPOSED** code annotations on all code blocks, with "(illustrative, not prescriptive)" on all PROPOSED blocks
- **Key Takeaways** section at the end of each file
- **Source Files** section listing relevant Blaze source files
- **Warning blocks** for correctness-critical notes (BFP page sizes, CT arg exactness, from_cb_handle limitations)
- **Navigation footer** with Previous/Next links
- **[A]/[B] annotations** in File 03 code examples

## Line Counts

| File | Lines | Target Range |
|------|-------|-------------|
| 01_architecture_overview.md | ~620 | 500-700 |
| 02_shape_metadata_system.md | ~680 | 500-700 |
| 03_wrapping_vs_extending_vs_replacing.md | ~520 | 500-700 |
| 04_interaction_with_graph_and_compilation_paths.md | ~560 | 500-700 |
| **Total** | **~2,380** | **2,000-2,800** |

## What Was NOT Included (and Why)

1. **V1's per-op BlazeModule subclasses** (BlazeLinear, BlazeRMSNorm): Replaced by V5's single BlazeModule with from_torch() factory, which is more composable and avoids class explosion.

2. **V2's extra ShapeDescriptor fields** (core_ranges, shard_shape, device_tensor): These conflate shape metadata with placement metadata, violating the three-layer separation.

3. **V4's per-component error tables**: The comprehensive error tables (5+ tables in File 03 alone) would interrupt the narrative flow. The error pipeline examples table in File 01 provides representative coverage.

4. **V3's page_size and num_pages_hint on ShapeDescriptor**: These are derived quantities that belong in CBSizer/CBBackend, not in the shape metadata itself. Keeping them off ShapeDescriptor avoids dual sources of truth.

5. **V1's ExternalTensor.metadata dict approach for BlazeGraph**: Less type-safe than the dedicated Edge.shape_descriptor field adopted from V5/V4.

6. **V3's hardcoded per-op derive_user_args()**: The per-op-name branching is preserved in the synthesis (since a registry-based alternative would require more infrastructure), but the limitation is acknowledged and forward-referenced to future work.
