# Chapter 5: Registering MicroOps and FusedOps

[Chapter 4](../ch04_adapter_pattern/index.md) designed `BlazeDeviceOperationAdapter<OpTag>` as a parameterized template that wraps any Blaze op's `MeshProgramDescriptor` in a `DeviceOperationConcept`-conforming struct. The design was abstract: a template definition, shared types, a factory wrapper, and a hash cascade. This chapter makes it concrete. We instantiate the adapter for a real MicroOp (Matmul), a real FusedOp (GatedReduce), and then scale the pattern to all 112+ Blaze ops through batch registration. Each walkthrough starts from the Blaze Python op definition and traces every layer -- tag struct, adapter instantiation, `register_operation<>` call, nanobind binding, Python dispatch routing, program cache behavior, and profiler output -- showing the exact code and data at each transition point.

The central insight is that the adapter's generality does not sacrifice concreteness: every MicroOp and FusedOp follows the same five-step registration path, and the only per-op artifact is a single line in the X-macro list.

---

## Chapter Contents

| # | File | Topic |
|---|------|-------|
| 1 | [MicroOp Registration Walkthrough](01_microop_registration_walkthrough.md) | MicroOp characteristics, `cpp_parser.py` auto-derivation, end-to-end Matmul registration from `blaze/ops/matmul/op.py` through every C++ and Python layer to profiler output, program cache isolation verification, edge cases (multi-device, source ops, existing custom hashes). |
| 2 | [FusedOp Registration Walkthrough](02_fusedop_registration_walkthrough.md) | FusedOp characteristics, why multi-phase composition is transparent to the adapter, why FusedOps cannot be decomposed into per-phase TTNN ops, `FusedOpConfig` integration, hash strategy for graph-topology-dependent programs, `_graph_branch_keys` for dynamic composition, `override_runtime_arguments` correctness for multi-phase programs, end-to-end GatedReduce walkthrough. |
| 3 | [Batch Registration and Scaling](03_batch_registration_and_scaling.md) | Three approaches for registering all 112+ ops (variadic template, Python codegen, macro table), build system integration, template instantiation cost mitigation, runtime discovery via `ttnn.blaze.list_ops()`, and the open set problem for dynamically-added ops. |

---

## Registration Checklist

The following table summarizes every artifact required to register a single Blaze op, with columns indicating where MicroOps and FusedOps differ.

| Step | Artifact | MicroOp | FusedOp | Where Defined |
|:----:|----------|:-------:|:-------:|---------------|
| 1 | Op name string (e.g., `"matmul"`) | `MicroOp.name` | `FusedOp.name` | `blaze/ops/{op}/op.py` |
| 2 | `OpTag` struct with `static constexpr const char* name` | Identical | Identical | `blaze_op_list.hpp` (X-macro) |
| 3 | `BlazeDeviceOperationAdapter<OpTag>` instantiation | Identical | Identical | Template auto-instantiation |
| 4 | `register_operation<"ttnn::blaze::{name}", Adapter<Tag>>()` | Identical | Identical | `blaze_op_registrations.hpp` |
| 5 | Nanobind `bind_blaze_op(mod, op, doc)` | Identical | Identical | `blaze_ops_nanobind.cpp` |
| 6 | `_run_program()` dispatch resolution | Identical | Identical | `compiler.py` |
| 7 | `MeshProgramDescriptor` structure | Single `ProgramDescriptor` per device | Single multi-phase `ProgramDescriptor` per device | Blaze compilation pipeline |
| 8 | `compute_program_hash` behavior | Hashes single-kernel descriptor | Hashes multi-phase descriptor (graph topology + user_args + shapes) | Adapter template |
| 9 | Program cache key | `type_hash<Adapter<MatmulTag>>` + descriptor hash | `type_hash<Adapter<GatedReduceTag>>` + descriptor hash | Adapter template |
| 10 | Validation hook (optional) | Port count check | Composition graph consistency | `OpTag::validate()` (Tier 3 only) |

**Key observation:** Steps 2--6 are structurally identical for both op categories. The divergence is confined to steps 7--10, which depend on the *content* of the `MeshProgramDescriptor` and the *semantics* of the op, not on the registration machinery.

---

## Op Registration Decision Tree

Before diving into walkthroughs, use this decision tree to determine the right registration tier for any Blaze op:

```text
                        Is this a Blaze op?
                              |
                     +--------+--------+
                     |                 |
                    Yes               No --> Use standard TTNN registration
                     |                       (Chapter 1)
                     v
              Single kernel / single emit()?
                     |
              +------+------+
              |             |
             Yes           No
              |             |
              v             v
          MicroOp       FusedOp
              |             |
              v             v
         Does it need    Does it have dynamic
         custom C++      graph structure
         validation?     (conditional compose)?
              |             |
         +----+----+   +---+---+
         |         |   |       |
        Yes       No  Yes     No
         |         |   |       |
         v         v   v       v
       Tier 3    Tier 1  Tier 2    Tier 1
       (per-op   (generic  with     (generic
       wrapper)  adapter)  topology  adapter,
                          -aware    Tier 2
                          hash +    optional
                          _graph_   for perf)
                          branch_
                          keys
```

Most ops (~95%) use Tier 1 (generic adapter). Ops with expensive descriptors or high dispatch frequency benefit from Tier 2 (Python-computed `custom_program_hash`, zero C++ changes). Only 3--5 ops justify Tier 3 (per-op C++ wrapper with semantic validation).

---

## Prerequisites

| Prerequisite | Why |
|---|---|
| [Chapter 1: The TTNN Op Registration Contract](../ch01_ttnn_registration_contract/index.md) | The walkthroughs reference `DeviceOperationConcept` ([Section 1.1](../ch01_ttnn_registration_contract/01_device_operation_concept.md)), `register_operation<>` ([Section 1.2](../ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md)), and program cache/profiler mechanics ([Section 1.3](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md)). |
| [Chapter 2: How generic_op Works](../ch02_generic_op_dispatch/index.md) | The "before" state in all comparisons is the `generic_op` dispatch path ([Section 2.1](../ch02_generic_op_dispatch/01_generic_op_internals.md)). |
| [Chapter 3: The Compilation Model Gap](../ch03_compilation_model_gap/index.md) | The walkthroughs show how each translation challenge ([Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)) is resolved concretely. |
| [Chapter 4: The Adapter Pattern](../ch04_adapter_pattern/index.md) | The adapter template ([Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)), Python dispatch ([Section 4.3](../ch04_adapter_pattern/03_python_dispatch_and_registration.md)), and X-macro design ([Section 4.2.6](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)) are instantiated here. |

---

## Cross-References

- **[Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)** defined `BlazeDeviceOperationAdapter<OpTag>` abstractly. Sections 5.1 and 5.2 instantiate it for Matmul and GatedReduce, respectively.
- **[Section 4.3](../ch04_adapter_pattern/03_python_dispatch_and_registration.md)** designed the `_run_program()` dispatch modification and X-macro auto-generation. Section 5.3 shows the build system integration end-to-end.
- **[Section 2.2](../ch02_generic_op_dispatch/02_blaze_compilation_pipeline.md)** documented the Blaze compilation pipeline. Sections 5.1 and 5.2 trace a MicroOp and FusedOp through that pipeline and into the adapter.
- **[Chapter 6](../ch06_caching_tracing_profiling/index.md)** details program cache, graph tracing, and profiling integration at depth; this chapter previews the observable effects.
- **[Chapter 8](../ch08_build_migration_alternatives/index.md)** covers build system alternatives and migration strategy; Section 5.3 addresses the build integration needed for batch registration.

---

## Source File Index

### Blaze Op Sources (Existing)

| File | Role |
|------|------|
| `blaze/ops/matmul/op.py` | Matmul MicroOp definition (walkthrough subject, Section 5.1) |
| `blaze/ops/matmul/kernels/op.hpp` | Matmul kernel header (auto-derivation source) |
| `blaze/cpp_parser.py` | C++ header parser for auto-derivation |
| `blaze/blaze_op.py` | `BlazeOp`, `MicroOp`, `FusedOp` base classes |
| `blaze/compiler.py` | `BlazeCompiler`, `MeshCompiledProgram` |

### Proposed New/Modified Files

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/blaze/blaze_device_operation_adapter.hpp` | Adapter template (from [Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)) |
| `ttnn/cpp/ttnn/operations/blaze/blaze_op_list.hpp` | X-macro list (single source of truth for all ops) |
| `ttnn/cpp/ttnn/operations/blaze/blaze_op_registrations.hpp` | Generated tag structs + `register_operation<>` calls |
| `ttnn/cpp/ttnn/operations/blaze/blaze_ops_nanobind.cpp` | Nanobind bindings for `ttnn._ttnn.blaze` |
| `blaze/compiler.py` (modified) | `_resolve_dispatch()` for named routing |

---

## Addresses Research Questions

| Question | Coverage |
|----------|---------|
| **Q5:** How is the adapter instantiated for concrete MicroOps and FusedOps? | Sections 5.1 (Matmul MicroOp) and 5.2 (GatedReduce FusedOp) provide end-to-end walkthroughs |
| **Q7:** What are the specific challenges of registering FusedOps? | Section 5.2 covers multi-phase descriptors, graph-topology hashing, dynamic composition edge cases, `FusedOpConfig` integration, `_graph_branch_keys`, and `override_runtime_arguments` correctness |

---

| [Chapter 4: The Adapter Pattern](../ch04_adapter_pattern/index.md) | **Chapter 5** | [Chapter 6: Program Caching, Graph Tracing, and Profiling Integration](../ch06_caching_tracing_profiling/index.md) |
|:---|:---:|---:|
