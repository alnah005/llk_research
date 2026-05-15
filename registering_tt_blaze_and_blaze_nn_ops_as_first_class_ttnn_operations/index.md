# Registering TT-Blaze and blaze-nn Ops as First-Class TTNN Operations

A comprehensive technical guide for Tenstorrent engineers investigating what it takes to promote Blaze and blaze-nn operations from `ttnn.generic_op()` dispatch to fully registered TTNN operations with per-op profiling, program cache isolation, graph tracing, and Python-callable bindings. The guide covers the TTNN registration contract, the current generic_op dispatch path, the structural gap between Blaze and TTNN compilation models, a parameterized adapter pattern that bridges them, concrete registration walkthroughs, infrastructure integration (caching, tracing, profiling), blaze-nn reconnection, and the build system / migration plan / open questions needed to ship.

---

## How to Use This Guide

| If you want to... | Start here | Then read |
|---|---|---|
| Understand the TTNN registration contract | [Ch 1 -- DeviceOperationConcept](ch01_ttnn_registration_contract/01_device_operation_concept.md) | [Ch 1 -- Registration & Dispatch](ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md) |
| See how Blaze currently dispatches via generic_op | [Ch 2 -- generic_op Internals](ch02_generic_op_dispatch/01_generic_op_internals.md) | [Ch 2 -- Blaze Compilation Pipeline](ch02_generic_op_dispatch/02_blaze_compilation_pipeline.md) |
| Understand why a direct port is impractical | [Ch 3 -- Blaze vs TTNN Lifecycles](ch03_compilation_model_gap/01_blaze_vs_ttnn_lifecycles.md) | [Ch 3 -- Translation Challenges](ch03_compilation_model_gap/02_translation_challenges.md) |
| See the proposed adapter design | [Ch 4 -- Design Goals](ch04_adapter_pattern/01_design_goals_and_tradeoffs.md) | [Ch 4 -- BlazeDeviceOperationAdapter](ch04_adapter_pattern/02_blaze_device_operation_adapter.md) |
| Walk through registering a specific op | [Ch 5 -- MicroOp Walkthrough](ch05_op_registration/01_microop_registration_walkthrough.md) | [Ch 5 -- FusedOp Walkthrough](ch05_op_registration/02_fusedop_registration_walkthrough.md) |
| Understand profiling / caching / tracing impact | [Ch 6 -- Program Cache](ch06_caching_tracing_profiling/01_program_cache_integration.md) | [Ch 6 -- Tracy Profiling](ch06_caching_tracing_profiling/03_tracy_profiling.md) |
| Connect blaze-nn to registered ops | [Ch 7 -- Current Dispatch](ch07_blaze_nn_integration/01_current_blaze_nn_dispatch.md) | [Ch 7 -- Rewired Dispatch](ch07_blaze_nn_integration/02_rewired_dispatch_and_module_integration.md) |
| Plan the migration and evaluate build options | [Ch 8 -- Build System](ch08_build_migration_alternatives/01_build_system_integration.md) | [Ch 8 -- Migration Plan](ch08_build_migration_alternatives/02_migration_plan.md) |
| Evaluate alternatives to the adapter approach | [Ch 8 -- Precedents & Alternatives](ch08_build_migration_alternatives/03_existing_patterns_and_precedents.md) | [Ch 8 -- Open Questions](ch08_build_migration_alternatives/04_open_questions.md) |

---

## Chapter Index

| # | Chapter | Description | Key Concepts |
|---|---------|-------------|--------------|
| 1 | [The TTNN Op Registration Contract](ch01_ttnn_registration_contract/index.md) | How native TTNN operations are defined, registered, dispatched, and exposed to Python. | `DeviceOperationConcept`, `register_operation<>`, `device_operation::launch<>`, `bind_registered_operation()` |
| 2 | [How generic_op Works](ch02_generic_op_dispatch/index.md) | The complete path from Blaze `emit()` through `BlazeCompiler` to `ttnn.generic_op()`. | `GenericOpDeviceOperation`, `GenericMeshProgramFactory`, `MeshProgramDescriptor`, `custom_program_hash` |
| 3 | [The Compilation Model Gap](ch03_compilation_model_gap/index.md) | Why Blaze ops cannot be trivially registered -- seven structural mismatches between Blaze's Python compilation and TTNN's C++ registration. | Language boundary, stateful vs stateless, output tensor management, multi-phase programs, the "100+ ops problem" |
| 4 | [The Adapter Pattern](ch04_adapter_pattern/index.md) | A parameterized C++ adapter (`BlazeDeviceOperationAdapter<OpTag>`) that wraps Blaze's `MeshProgramDescriptor` into `DeviceOperationConcept` without per-op factory code. | `BlazeDeviceOperationAdapter<OpTag>`, `OpTag` struct, `GenericMeshProgramFactory` reuse, `ttnn::blaze::*` namespace |
| 5 | [Registering MicroOps and FusedOps](ch05_op_registration/index.md) | End-to-end walkthroughs for both op categories plus batch registration strategies for scaling to 112+ ops. | Matmul MicroOp walkthrough, GatedReduce FusedOp walkthrough, X-macro registration, codegen |
| 6 | [Program Caching, Graph Tracing, and Profiling](ch06_caching_tracing_profiling/index.md) | How first-class registration unlocks TTNN's cache, graph capture, and profiling infrastructure for Blaze ops. | Three-tier hash cascade, `type_hash<Adapter<Tag>>`, `GraphTracker` named nodes, `TracyOpMeshWorkload` per-op zones |
| 7 | [blaze-nn Integration](ch07_blaze_nn_integration/index.md) | Rewiring blaze-nn's functional API to dispatch through registered `ttnn.blaze.*` callables. | `OpMapping.ttnn_callable`, dual-resolution architecture, `BLAZE_NN_FORCE_GENERIC` escape hatch |
| 8 | [Build System, Migration Path, and Open Questions](ch08_build_migration_alternatives/index.md) | Build integration options, a five-phase migration plan (~2,000 LoC / 7--10 weeks), ecosystem precedents, and seven unresolved questions. | Option A/B/D build models, Phase 0--4 migration, CCL/moreh/experimental precedents, priority matrix |

---

## Quick Reference

| Concept | What It Does | Where to Learn More |
|---------|-------------|---------------------|
| `DeviceOperationConcept` | C++20 concept defining the 5 type aliases and 3 required methods for a TTNN operation | [Section 1.1](ch01_ttnn_registration_contract/01_device_operation_concept.md) |
| `register_operation<"name", Struct>()` | Compile-time template that wraps an operation struct in a `registered_operation_t` callable | [Section 1.2](ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md) |
| `ttnn.generic_op()` | The registered TTNN operation that accepts raw `MeshProgramDescriptor` objects -- current dispatch target for all Blaze ops | [Section 2.1](ch02_generic_op_dispatch/01_generic_op_internals.md) |
| `BlazeDeviceOperationAdapter<OpTag>` | Proposed parameterized adapter wrapping Blaze descriptors to satisfy `DeviceOperationConcept` | [Section 4.2](ch04_adapter_pattern/02_blaze_device_operation_adapter.md) |
| `OpTag` struct | Minimal stateless C++ struct providing type identity and name (e.g., `MatmulTag`) | [Section 4.2](ch04_adapter_pattern/02_blaze_device_operation_adapter.md) |
| Three-tier hash cascade | Tier 1: descriptor walk, Tier 2: `custom_program_hash`, Tier 3: `OpTag` hook | [Section 6.1](ch06_caching_tracing_profiling/01_program_cache_integration.md) |
| `_resolve_dispatch()` | Python method in `MeshCompiledProgram` that decides registered vs `generic_op` path | [Section 7.2](ch07_blaze_nn_integration/02_rewired_dispatch_and_module_integration.md) |
| Phase 0 (`op_name` field) | Zero-adapter quick win: add `op_name` to `MeshProgramDescriptor` for immediate profiler visibility (~35 LoC) | [Section 8.2](ch08_build_migration_alternatives/02_migration_plan.md) |

---

## Prerequisites

| Prerequisite | Why |
|---|---|
| Python fluency and basic C++ template literacy | The guide covers Python op authoring, C++20 concepts, `std::variant`, and `constexpr` templates. |
| TT-Metal fundamentals (`Program`, `ProgramDescriptor`, `MeshProgramDescriptor`, circular buffers, kernels) | The adapter wraps these primitives; understanding their structure is essential. |
| Experience with at least one Blaze op (`BlazeOp`, `MicroOp`, `FusedOp`, `emit()`, `compose()`) | The walkthroughs assume familiarity with Blaze's op authoring model. |
| Basic TTNN tensor usage (`ttnn.generic_op()`, device placement) | The dispatch path being modified starts at `ttnn.generic_op()`. |
| Basic nanobind awareness | Python-C++ bindings are generated through nanobind. |

---

## Source Code Location

The primary source files referenced throughout this guide span two repositories:

**TT-Metal / TTNN:**
- `ttnn/api/ttnn/operation_concepts.hpp` -- `DeviceOperationConcept` definition
- `ttnn/api/ttnn/decorators.hpp` -- `register_operation<>` template
- `ttnn/cpp/ttnn/operations/generic/` -- `GenericOpDeviceOperation`, `GenericMeshProgramFactory`
- `tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp` -- `MeshProgramDescriptor`
- `ttnn/cpp/ttnn-nanobind/decorators.hpp` -- `bind_registered_operation()`

**TT-Blaze / blaze-nn:**
- `blaze/compiler.py` -- `BlazeCompiler`, `_run_program()`, `_resolve_dispatch()`
- `blaze/ops/*/op.py` -- MicroOp definitions
- `blaze_nn/_registry.py` -- `OpMapping`, `_REGISTRY`
- `blaze_nn/functional.py` -- `F.linear()`, `F.rmsnorm()`, etc.

---

## Research Questions Addressed

| # | Question | Primary Chapter |
|---|---------|:---:|
| Q1 | What is the TTNN registration contract? | [1](ch01_ttnn_registration_contract/index.md) |
| Q2 | How does `generic_op` currently work? | [2](ch02_generic_op_dispatch/index.md) |
| Q3 | What does the `generic_op` path bypass? | [2](ch02_generic_op_dispatch/index.md) |
| Q4 | What is the structural gap between Blaze and TTNN? | [3](ch03_compilation_model_gap/index.md) |
| Q5 | What adapter strategies exist? | [4](ch04_adapter_pattern/index.md) |
| Q6 | How would per-op and batch registration work? | [5](ch05_op_registration/index.md) |
| Q7 | What unique challenges do FusedOps present? | [5](ch05_op_registration/index.md) |
| Q8 | How does registration change program caching? | [6](ch06_caching_tracing_profiling/index.md) |
| Q9 | How do registered ops appear in graph capture? | [6](ch06_caching_tracing_profiling/index.md) |
| Q10 | How do registered ops appear in Tracy profiling? | [6](ch06_caching_tracing_profiling/index.md) |
| Q11 | How would blaze-nn connect to registered ops? | [7](ch07_blaze_nn_integration/index.md) |
| Q12 | What build system changes are needed? | [8](ch08_build_migration_alternatives/index.md) |
| Q13 | What is a practical phased migration plan? | [8](ch08_build_migration_alternatives/index.md) |
| Q14 | What alternative approaches exist? | [8](ch08_build_migration_alternatives/index.md) |
