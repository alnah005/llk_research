# Chapter 4: The Adapter Pattern -- Bridging Blaze Programs to DeviceOperationConcept

[Chapter 3](../ch03_compilation_model_gap/index.md) established that seven translation challenges and three structural incompatibilities prevent naive registration of Blaze ops as TTNN device operations. The language boundary, stateful compilation, and decision-timing inversion together rule out any approach that requires re-executing Blaze logic inside C++ factories. But Blaze already produces valid programs that run correctly on silicon -- the only deficiency is *how those programs reach the hardware*, not *what they do once they get there*. The adapter pattern exploits this observation: it wraps Blaze's pre-built `MeshProgramDescriptor` in a `DeviceOperationConcept`-conforming template that injects per-op identity, enables semantic caching and profiling, and integrates with TTNN's validation and graph-tracing infrastructure -- all without writing a single line of new program-factory code.

The central contribution is `BlazeDeviceOperationAdapter<OpTag>`, a C++ class template parameterized by a lightweight tag struct that carries op identity. Each instantiation -- `BlazeDeviceOperationAdapter<MatmulTag>`, `BlazeDeviceOperationAdapter<RMSNormTag>`, etc. -- is a distinct C++ type that satisfies `DeviceOperationConcept`, produces a unique `type_hash`, and appears as a named operation in the program cache, graph tracer, and Tracy profiler. The adapter reuses `GenericMeshProgramFactory` through a thin factory wrapper (`BlazeAdapterMeshProgramFactory`) that bridges the type-level mismatch between the adapter's `operation_attributes_t` and the factory's expected `MeshProgramDescriptor`. Per-op cost is reduced from the 250--850 lines of dedicated C++ estimated in [Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md) to approximately 5 lines (one tag struct + one `register_operation<>` call).

---

## Chapter Contents

| # | File | Topic |
|---|------|-------|
| 1 | [Design Goals and Tradeoffs](01_design_goals_and_tradeoffs.md) | Generic adapter vs per-op wrapping: six-dimension tradeoff analysis, three-tier hybrid strategy (generic / enhanced hash / per-op wrapper), decision matrix, hash cost quantification, phased effort estimates, backward compatibility with hash disjointness proof. |
| 2 | [BlazeDeviceOperationAdapter](02_blaze_device_operation_adapter.md) | Complete `BlazeDeviceOperationAdapter<OpTag>` C++ template: OpTag struct with SFINAE-detected hooks, shared external types, all required and optional methods, three-tier `compute_program_hash` cascade, `BlazeAdapterMeshProgramFactory` wrapper, registration syntax with X-macro convenience, concept satisfaction proof, end-to-end data flow diagram, challenge resolution table, and scope boundaries. |
| 3 | [Python Dispatch and Registration](03_python_dispatch_and_registration.md) | Modified `_run_program()` with compile-time dispatch resolution, `ttnn.blaze` namespace via nanobind, three auto-generation approaches (codegen / X-macro / hybrid with CI validation), `.pyi` type stubs, nanobind binding details, backward compatibility specification, concrete test functions, blaze-nn integration, and end-to-end registration walkthrough with before/after Tracy output. |

---

## Key Takeaways

1. **The adapter is a type-identity wrapper, not a program builder.** It satisfies `DeviceOperationConcept` structurally by wrapping Blaze's pre-built `MeshProgramDescriptor` in typed attributes and delegating program construction to the existing `GenericMeshProgramFactory` via a thin factory wrapper. Zero new factory logic is written.

2. **Per-op cost drops from ~500 lines to ~5 lines.** Each Blaze op requires only a tag struct (`struct MatmulTag { static constexpr const char* name = "blaze::matmul"; }`) and a `register_operation<>` call. The adapter template handles everything else.

3. **A three-tier hybrid strategy is recommended.** Tier 1 (generic adapter) covers ~95% of ops. Tier 2 (enhanced hash via `custom_program_hash` in Python, zero C++ changes per op) addresses hash-sensitive ops. Tier 3 (per-op wrapper) is reserved for the 3--5 ops where custom validation, semantic profiling attributes, or performance models justify the investment.

4. **All seven translation challenges are addressed** without moving any program-construction logic from Python to C++. The adapter accepts pre-built descriptors (Challenges 1, 2, 7), preserves pre-allocated output tensors (Challenge 3), registers at FusedOp granularity (Challenge 4), passes through dynamic kernel sources (Challenge 5), and delegates per-core specialization to `ProgramBuilder` (Challenge 6).

5. **Backward compatibility is inherent in the design.** `ttnn.generic_op()` is unchanged, `MeshProgramDescriptor` is unchanged, and hash disjointness is guaranteed by `type_hash` differentiation. Migration is gradual and reversible.

---

## Prerequisites

| Prerequisite | Why |
|---|---|
| [Chapter 1: The TTNN Op Registration Contract](../ch01_ttnn_registration_contract/index.md) | The adapter must satisfy `DeviceOperationConcept` ([Section 1.1](../ch01_ttnn_registration_contract/01_device_operation_concept.md)), use `register_operation<>` ([Section 1.2](../ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md)), and integrate with the program cache, graph tracing, and profiling ([Section 1.3](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md)). |
| [Chapter 2: How generic_op Works](../ch02_generic_op_dispatch/index.md) | The adapter reuses `GenericMeshProgramFactory` ([Section 2.1](../ch02_generic_op_dispatch/01_generic_op_internals.md)) and replaces the `generic_op` dispatch path. |
| [Chapter 3: The Compilation Model Gap](../ch03_compilation_model_gap/index.md) | The adapter design is driven by the seven translation challenges ([Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)) and must respect the lifecycle incompatibilities ([Section 3.1](../ch03_compilation_model_gap/01_blaze_vs_ttnn_lifecycles.md)). |

---

## Cross-References

- **[Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)** enumerated seven translation challenges and the "100+ ops problem." Section 4.2 maps each challenge to a specific adapter design decision.
- **[Section 2.1](../ch02_generic_op_dispatch/01_generic_op_internals.md)** documented `GenericOpDeviceOperation` and `GenericMeshProgramFactory`, which the adapter reuses.
- **[Section 1.1](../ch01_ttnn_registration_contract/01_device_operation_concept.md)** defined the `DeviceOperationConcept` that the adapter must satisfy. Section 4.2 maps every concept requirement to the adapter's implementation.
- **[Chapter 5](../ch05_op_registration/index.md)** instantiates the adapter for concrete MicroOp and FusedOp walkthroughs, demonstrating batch registration at scale.
- **[Chapter 6](../ch06_caching_tracing_profiling/index.md)** details how the adapter's per-op identity integrates with program caching, graph tracing, and Tracy profiling.
- **[Chapter 7](../ch07_blaze_nn_integration/index.md)** covers blaze-nn integration, which benefits automatically from the adapter's named dispatch path.
- **[Chapter 8](../ch08_build_migration_alternatives/index.md)** discusses build system integration, dynamic registration as future work, and long-term migration alternatives.

---

## Source File Index

### Proposed New Files

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/blaze/blaze_device_operation_adapter.hpp` | `BlazeDeviceOperationAdapter<OpTag>` template + `BlazeAdapterAttributes` + `BlazeAdapterTensorArgs` + `BlazeAdapterMeshProgramFactory` |
| `ttnn/cpp/ttnn/operations/blaze/blaze_op_list.hpp` | X-macro list of all registered Blaze ops (`BLAZE_OP_LIST(X)`) |
| `ttnn/cpp/ttnn/operations/blaze/blaze_op_registrations.hpp` | Tag struct definitions + `register_operation<>` calls (generated from X-macro) |
| `ttnn/cpp/ttnn/operations/blaze/blaze_ops_nanobind.cpp` | Nanobind bindings for `ttnn._ttnn.blaze` submodule |
| `ttnn/blaze/__init__.pyi` | Type stubs for IDE autocompletion |

### Existing Files (Reused by Adapter)

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.hpp` | `GenericMeshProgramFactory` -- the adapter's sole factory (via wrapper) |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp` | `GenericOpDeviceOperation` -- reference for validation logic and type structures |
| `ttnn/api/ttnn/decorators.hpp` | `register_operation<>` template |
| `ttnn/cpp/ttnn-nanobind/decorators.hpp` | `bind_registered_operation()` for Python exposure |
| `ttnn/api/ttnn/operation_concepts.hpp` | `DeviceOperationConcept` and related concept definitions |

---

## Addresses Research Questions

| Question | Coverage |
|----------|---------|
| **Q5:** What adapter strategies can bridge Blaze programs to DeviceOperationConcept? | Section 4.1 (design goals and tradeoff analysis), Section 4.2 (the adapter template) |
| **Q6:** How would per-op and batch registration work in practice? | Section 4.2 (registration syntax, X-macro), Section 4.3 (Python dispatch, auto-generation, end-to-end walkthrough) |

---

| [Chapter 3: The Compilation Model Gap](../ch03_compilation_model_gap/index.md) | **Chapter 4** | [Chapter 5: Registering MicroOps and FusedOps](../ch05_op_registration/index.md) |
|:---|:---:|---:|
