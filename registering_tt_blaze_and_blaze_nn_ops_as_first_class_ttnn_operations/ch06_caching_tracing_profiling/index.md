# Chapter 6: Program Caching, Graph Tracing, and Profiling Integration

Chapters 4 and 5 established that `BlazeDeviceOperationAdapter<OpTag>` wraps any Blaze op's `MeshProgramDescriptor` in a `DeviceOperationConcept`-conforming C++ type, injecting per-op identity at a cost of roughly five lines per op. But stating that registration "unlocks" TTNN infrastructure is only half the story. This chapter examines *exactly* how three cross-cutting TTNN subsystems -- the program cache, the graph tracer, and the Tracy profiler -- behave before and after registration, what adapter-level work is needed to make each feature function correctly, and where the adapter's design deliberately preserves `generic_op` behavior rather than rewriting it.

The organizing principle is **systematic before/after comparison**. For every subsystem, we trace the concrete behavior when a Blaze Matmul (MicroOp) and a Blaze GatedReduce (FusedOp) are dispatched via `generic_op` versus via `BlazeDeviceOperationAdapter<MatmulTag>` and `BlazeDeviceOperationAdapter<GatedReduceTag>`. Side-by-side tables, diff-style formatting, quantitative cost models, and ASCII diagrams make each difference visible at the source-code level.

---

## Chapter Contents

| # | File | Topic |
|---|------|-------|
| 1 | [Program Cache Integration](./01_program_cache_integration.md) | Hash computation before and after registration, cache key design (generic adapter key vs individual wrapper key), the three-tier hash cascade with quantitative benchmarks, `override_runtime_arguments` and `shared_variables_t` design sketch, cache invalidation comparison, performance cost models, and "What Changes / What Stays the Same" summary. |
| 2 | [Graph Tracing and Inspector](./02_graph_tracing_and_inspector.md) | `GraphTracker::track_function_start/end` mechanics before and after, `NO_DISPATCH` mode handling, `compute_output_specs` for graph capture, Inspector integration via `get_operation_name<>()`, graph-level optimization implications, concrete graph JSON fragments for Matmul and GatedReduce, and limitations vs native TTNN ops. |
| 3 | [Tracy Profiling](./03_tracy_profiling.md) | `TracyOpMeshWorkload` macro behavior before and after, flamegraph comparison, `op_profiler.hpp` and reflectable attributes, custom profiling metadata for high-value Tier 3 ops, performance model hooks, profiler lookup tables, environment variables and compilation guards, and end-to-end profiler output comparison. |

---

## Key Takeaways

1. **All three infrastructure systems derive identity from `type_hash<device_operation_t>` and `get_type_name<device_operation_t>()`.** Registration changes the type from `GenericOpDeviceOperation` to `BlazeDeviceOperationAdapter<OpTag>`, propagating through caching, tracing, and profiling simultaneously.

2. **The program cache gains per-op namespace isolation without changing the cache data structure.** Disjoint type seeds eliminate cross-op hash collision risk and enable $O(1)$ hashing for Tier 2 ops.

3. **Graph tracing gains semantic node names, enabling graph-level analysis.** Named nodes replace the flat `"GenericOpDeviceOperation"` sequence, supporting pattern detection, fusion analysis, and memory scheduling.

4. **Tracy profiling gains per-op flamegraph zones and reflectable attribute metadata.** Post-processing scripts can produce per-op-type time breakdowns and invocation counts.

5. **The adapter requires zero new infrastructure code.** Every mechanism described in this chapter -- `compute_program_hash`, `GraphTracker::track_function_start`, `TracyOpMeshWorkload`, Inspector annotations -- already exists in TTNN ([Section 1.3](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md)). The adapter's contribution is satisfying the type-level contract that activates these mechanisms per op.

---

## Prerequisites

| Prerequisite | Why |
|---|---|
| [Chapter 1, Section 1.3: Program Cache, Graph Tracing, and Profiling](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md) | Defines the infrastructure mechanisms (`ProgramCache`, `GraphTracker`, `TracyOpMeshWorkload`, Inspector) that this chapter shows in action for Blaze ops. |
| [Chapter 2: How generic_op Works](../ch02_generic_op_dispatch/index.md) | The "before" baseline: `GenericOpDeviceOperation`'s hash computation, graph trace behavior, and profiler output. |
| [Chapter 4, Section 4.2: BlazeDeviceOperationAdapter](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md) | The adapter template whose `compute_program_hash`, `compute_output_specs`, and `BlazeAdapterAttributes` are exercised throughout this chapter. |
| [Chapter 5: Registering MicroOps and FusedOps](../ch05_op_registration/index.md) | Concrete Matmul (MicroOp) and GatedReduce (FusedOp) registration walkthroughs whose cache/trace/profiler behavior is examined here. |

---

## Cross-References

- **[Section 1.3](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md)** introduced the infrastructure mechanisms at the contract level. This chapter shows how each mechanism behaves with the adapter in practice.
- **[Section 2.3](../ch02_generic_op_dispatch/03_divergence_analysis.md)** cataloged 15 divergences between `generic_op` and native TTNN ops. This chapter resolves divergences related to profiling opacity, graph trace collapse, and cache namespace pollution.
- **[Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)** defined the three-tier hash cascade. Section 6.1 exercises all three tiers with concrete ops and performance data.
- **[Section 5.1](../ch05_op_registration/01_microop_registration_walkthrough.md)** and **[Section 5.2](../ch05_op_registration/02_fusedop_registration_walkthrough.md)** registered Matmul and GatedReduce. This chapter picks up where those walkthroughs ended -- at the point where the adapter interacts with infrastructure.
- **[Chapter 7](../ch07_blaze_nn_integration/index.md)** covers blaze-nn integration, which inherits all infrastructure benefits described here.
- **[Chapter 8](../ch08_build_migration_alternatives/index.md)** discusses future directions including dynamic registration and performance model hooks previewed in Section 6.3.

---

## Addresses Research Questions

| Question | Coverage |
|----------|---------|
| **Q8:** How does registration affect program cache behavior for Blaze ops? | Section 6.1: hash computation comparison, cache key isolation, `override_runtime_arguments` opportunity, invalidation semantics, performance benchmarks. |
| **Q9:** What changes in graph tracing and the Inspector after registration? | Section 6.2: `track_function_start/end` with named types, `NO_DISPATCH` capture, `compute_output_specs` for shape propagation, Inspector annotations, graph-level optimization implications. |
| **Q10:** How does Tracy profiling output change with per-op registration? | Section 6.3: flamegraph zones, reflectable attribute serialization, custom metadata for Tier 3 ops, performance model integration points. |

---

## Source File Index

### TTNN Infrastructure (Existing)

| File | Role |
|------|------|
| `tt_metal/api/tt-metalium/program_cache.hpp` | `ProgramCache`, `CachedProgramFactory`, `CachedMeshWorkload` |
| `ttnn/api/ttnn/device_operation.hpp` | `compute_program_hash<>`, `launch<>`, `enqueue_mesh_workload<>`, `emit_mesh_workload_annotation<>`, cache hit/miss paths |
| `ttnn/api/ttnn/device_operation_detail.hpp` | `emit_mesh_workload_annotation_impl`, non-template helpers |
| `tt_metal/api/tt-metalium/graph_tracking.hpp` | `GraphTracker`, `IGraphProcessor`, `IGraphHooks` |
| `ttnn/api/ttnn/graph/graph_processor.hpp` | `GraphProcessor`, `ProcessorHooks`, `ScopedGraphCapture` |
| `ttnn/api/tools/profiler/op_profiler.hpp` | `TracyOpMeshWorkload`, `op_meta_data_serialized_json`, `RuntimeIDToOpName`, `ProgramHashToOpName` |
| `ttnn/api/ttnn/mesh_device_operation_utils.hpp` | `set_runtime_id`, `track_workload` |
| `tt_metal/api/tt-metalium/experimental/inspector.hpp` | `EmitMeshWorkloadAnnotation`, `EmitMeshWorkloadRuntimeId` |

### Adapter Files (Proposed, from Chapter 4)

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/blaze/blaze_device_operation_adapter.hpp` | `BlazeDeviceOperationAdapter<OpTag>`, `BlazeAdapterAttributes`, `BlazeAdapterMeshProgramFactory` |
| `ttnn/cpp/ttnn/operations/blaze/blaze_op_list.hpp` | X-macro list of registered ops |
| `ttnn/cpp/ttnn/operations/blaze/blaze_op_registrations.hpp` | Tag structs + `register_operation<>` calls |

### Blaze Files (Existing)

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp` | `GenericOpDeviceOperation` -- the "before" baseline |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp` | `compute_program_hash`, `compute_output_specs` for `generic_op` |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp` | `GenericMeshProgramFactory::override_runtime_arguments` |

---

| [Chapter 5: Registering MicroOps and FusedOps](../ch05_op_registration/index.md) | **Chapter 6** | [Chapter 7: blaze-nn Integration](../ch07_blaze_nn_integration/index.md) |
|:---|:---:|---:|
