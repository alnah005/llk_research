# Chapter 1: The TTNN Op Registration Contract

TTNN exposes hardware operations to Python through a layered contract that starts with a C++20 concept (`DeviceOperationConcept`), passes through a compile-time registration template (`register_operation`), and terminates in a nanobind class visible from `import ttnn`. Every native TTNN op -- from `ttnn.add` to `ttnn.matmul` -- follows this same pipeline. Understanding the contract is a prerequisite for promoting any external kernel (including those originating in TT-Blaze or blaze-nn) to a first-class TTNN citizen.

## Chapter Files

| # | File | Topic |
|---|------|-------|
| 1 | [DeviceOperationConcept](./01_device_operation_concept.md) | The `DeviceOperationConcept` -- five type aliases, required/optional static methods, program factory concepts, `CachedProgram` types, `AllFactoriesValid`, concept hierarchy diagram, and a concrete example (`BinaryDeviceOperation`). |
| 2 | [Registration, Dispatch, and Bindings](./02_registration_dispatch_and_bindings.md) | The `register_operation<>` template, the `registered_operation_t` callable, `MeshDeviceOperationAdapter`, `device_operation::launch<>`, `dispatch_to_mesh_workload_factory`, and `bind_registered_operation()` for nanobind. |
| 3 | [Program Cache, Graph Tracing, and Profiling](./03_program_cache_graph_tracing_and_profiling.md) | The program cache (hash computation, hit/miss paths, `CachedProgramFactory`), `GraphTracker` for graph capture, Tracy/`TracyOpMeshWorkload` profiling, Inspector annotations, and why `generic_op` loses identity in these systems. |

## Guiding Principles

The registration contract serves three purposes:

1. **Static safety** -- C++20 concepts reject malformed operations at compile time, before any silicon is touched.
2. **Uniform dispatch** -- A single `launch<>` template handles output allocation, mesh adaptation, caching, graph tracking, and enqueue for every conforming operation.
3. **Python transparency** -- `bind_registered_operation` generates `__call__` overloads, doc-strings, and attribute accessors automatically, so every registered op is callable as `ttnn.op_name(...)` with zero hand-written Python.

## Prerequisites

| Prerequisite | Why |
|---|---|
| C++20 concepts and `std::variant` | The entire op contract is enforced at compile time through concepts and variant visitors. |
| `tt-metalium` Program / MeshWorkload model | Operations ultimately produce `Program` objects that are enqueued onto device command queues. |
| Basic familiarity with nanobind | The Python surface is generated through nanobind class/method bindings. |

## Key Source Files

All paths are relative to the tt-metal repository root:

| File | Role |
|------|------|
| [`ttnn/api/ttnn/operation_concepts.hpp`](./01_device_operation_concept.md) | C++20 concepts defining the op contract |
| [`ttnn/api/ttnn/device_operation.hpp`](./03_program_cache_graph_tracing_and_profiling.md) | `launch<>` dispatch, cache logic, mesh workload creation |
| [`ttnn/api/ttnn/device_operation_detail.hpp`](./03_program_cache_graph_tracing_and_profiling.md) | Non-template helpers factored out to reduce instantiation cost |
| [`ttnn/api/ttnn/mesh_device_operation_adapter.hpp`](./02_registration_dispatch_and_bindings.md) | `MeshDeviceOperationAdapter<>` that wraps single-device ops for mesh dispatch |
| [`ttnn/api/ttnn/decorators.hpp`](./02_registration_dispatch_and_bindings.md) | `register_operation<>` and `registered_operation_t` |
| [`ttnn/cpp/ttnn-nanobind/decorators.hpp`](./02_registration_dispatch_and_bindings.md) | `bind_registered_operation()` for Python exposure |
| [`ttnn/api/tools/profiler/op_profiler.hpp`](./03_program_cache_graph_tracing_and_profiling.md) | Tracy profiling macros (`TracyOpMeshWorkload`, etc.) |
| [`tt_metal/api/tt-metalium/graph_tracking.hpp`](./03_program_cache_graph_tracing_and_profiling.md) | `GraphTracker` singleton, `IGraphProcessor`, `IGraphHooks` |
| [`ttnn/api/ttnn/graph/graph_processor.hpp`](./03_program_cache_graph_tracing_and_profiling.md) | `GraphProcessor`, `ProcessorHooks`, `ScopedGraphCapture` |
| [`tt_metal/api/tt-metalium/program_cache.hpp`](./01_device_operation_concept.md) | `ProgramCache`, `CachedProgram`, `CachedMeshWorkload`, `CachedProgramFactory` |
| [`ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.hpp`](./01_device_operation_concept.md) | Concrete example: `BinaryDeviceOperation` |
| [`ttnn/cpp/ttnn/operations/generic/generic_op.hpp`](./03_program_cache_graph_tracing_and_profiling.md) | `GenericOp` -- the identity-erased escape hatch |

---

*Next: [DeviceOperationConcept](./01_device_operation_concept.md)*
