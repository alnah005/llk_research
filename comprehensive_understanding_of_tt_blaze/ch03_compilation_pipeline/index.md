# Chapter 3 -- The Compilation Pipeline

This chapter traces the complete path from a declarative `BlazeGraph` to an executable program descriptor ready for dispatch on Blackhole silicon. It covers graph construction via the fusion context, the three-engine compilation pass (CBEngine, SemEngine, CTArgEngine), kernel codegen, and the final `BlazeCompiler` assembly that produces mesh-aware program descriptors.

## Key Takeaways

- The `FusionContext` context manager (`blaze.fuse()`) records op calls and data dependencies, then builds a validated `BlazeGraph` DAG on exit. The graph preserves both **insertion order** (used by kernel codegen for phase sequencing) and supports **topological order** (Kahn's algorithm, used by all engines for dependency-respecting analysis).
- The `compile_engines()` orchestrator runs three engines in strict sequence -- CBEngine, SemEngine, CTArgEngine -- each reading the graph and upstream results. The unified `EngineResult` holds CB assignments, semaphore assignments, and per-RISC CT arg tuples.
- CBEngine walks topological order to assign sequential CB IDs across four categories (`ext_in`, `internal`, `intermed`, `ext_out`), handles fan-out via grid union, tracks lifetimes, and offers interval-coloring compaction for the 64-CB hardware limit.
- SemEngine derives its sync points from `Sem`-kind CT arg schema entries, groups mcast senders sharing persistent NOC command state, and assigns sequential global semaphore indices. These logical indices are later materialized as `ttnn.GlobalSemaphore` objects with L1 addresses by the compiler.
- CTArgEngine replaces 130--400+ hand-tuned CT arg tuples with auto-generated, auto-prefixed, collision-checked arguments. Each arg's value is derived from its declared source (CB ID, semaphore address, user kwarg, grid context, or per-core descriptor). Shared-prefix dedup prevents false collisions for multi-branch ops.
- Kernel codegen generates a complete C++ `.cpp` from the graph's node sequence and registered `PhaseInfo` metadata. The Python graph **is** the kernel spec -- the developer writes only the C++ `Op` struct and the Python `BlazeOp` subclass; the kernel wiring file is generated automatically. Codegen handles mcast leader/follower persistent-state sharing, empty-body `init`/`teardown` elimination, noop kernel stubs, and DPRINT debug injection.
- `BlazeCompiler` decomposes mesh tensors per-device, runs the engine path (six steps) or the fused op path (via `FusedProgram`), creates global semaphores, builds the `UnifiedKernelDescriptor`, assembles `ProgramDescriptor` objects, and wraps them in a `MeshCompiledProgram`. Execution is a single call to `ttnn.generic_op()`.

## Reading Order

1. [**BlazeGraph and Fusion Context**](./01_blaze_graph_and_fusion_context.md) -- Graph IR data structures (`BlazeGraph`, `OpNode`, `Edge`), the `FusionContext` context manager, `ExternalTensor`/`FusionResult` handles, topological ordering, graph validation, and the `compile_engines()` orchestrator.
2. [**CB Engine**](./02_cb_engine.md) -- Topological walk, four CB categories, sequential ID assignment, fan-out and shared tensor optimization, `CBAssignment` lifetime tracking, `compact_cb_ids()` interval coloring, and `to_cb_id_manager()` interop.
3. [**Sem Engine**](./03_sem_engine.md) -- Sync point identification from CT arg schemas, protocol source_keys, conditional dual-NOC, mcast sender grouping, and `SemAssignment` global semaphore indices.
4. [**CT Arg Engine**](./04_ct_arg_engine.md) -- `CTArgSchema`/`CTArgSpec` declarations, auto-prefixing, shared-prefix dedup, value resolution from CB/sem/user/grid sources, collision detection, and RISC grouping for `UnifiedKernelDescriptor`.
5. [**Kernel Codegen**](./05_kernel_codegen.md) -- `PhaseInfo`/`PhaseDecl` structures, type alias generation, mcast grouping, `kernel_main()` emission, empty-body optimization, noop stubs, DPRINT support, and hash-based file caching.
6. [**BlazeCompiler**](./06_blaze_compiler.md) -- Mesh decomposition, engine path (six steps) vs. fused op path, `MeshCompiledProgram`/`CompiledProgram`, IO tensor ordering, compute configuration, `ttnn.generic_op()` dispatch, `BLAZE_EXPORT` auto-export, `BLAZE_L1_PROFILE` diagnostics, and per-device overrides.

## Files in This Chapter

| File | Description |
|------|-------------|
| [`01_blaze_graph_and_fusion_context.md`](./01_blaze_graph_and_fusion_context.md) | BlazeGraph IR, FusionContext, ExternalTensor/FusionResult, topo sort, validation, compile_engines() |
| [`02_cb_engine.md`](./02_cb_engine.md) | CB auto-assignment: four categories, fan-out, shared tensor, compaction, lifetime tracking |
| [`03_sem_engine.md`](./03_sem_engine.md) | Semaphore engine: sync points, protocol source_keys, mcast sender grouping |
| [`04_ct_arg_engine.md`](./04_ct_arg_engine.md) | CT arg generation: schema, auto-prefix, dedup, collision detection, RISC grouping |
| [`05_kernel_codegen.md`](./05_kernel_codegen.md) | C++ kernel generation: PhaseInfo, type aliases, mcast groups, DPRINT, hash caching |
| [`06_blaze_compiler.md`](./06_blaze_compiler.md) | BlazeCompiler: mesh decomp, engine/fused paths, ProgramDescriptor assembly, dispatch |

## Cross-References

- [Chapter 1 -- Architecture](../ch01_architecture/index.md): Stack position, the two-API design (graph API + composition API) that feeds into this pipeline.
- [Chapter 2 -- BlazeOp Hierarchy](../ch02_blazeop_hierarchy/index.md): `OpSpec` declarations, C++ parser that auto-derives `CTArgSchema` and `PhaseInfo`, auto-discovery and registration that populates the engine registries.

---
[Chapter 2 -- BlazeOp Hierarchy](../ch02_blazeop_hierarchy/index.md) | **Index** | [01 -- BlazeGraph and Fusion Context](./01_blaze_graph_and_fusion_context.md) ->
