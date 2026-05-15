# Chapter 2: How `generic_op` Works as Blaze's Dispatch Mechanism

> **Scope.** This chapter traces the full path from a Blaze `emit()` call through `BlazeCompiler` to `ttnn.generic_op()`, dissects the `GenericOp` / `GenericOpDeviceOperation` internals in TT-Metal, and identifies every point at which that path diverges from the contract followed by native TTNN operations (documented in [Chapter 1](../ch01_ttnn_registration_contract/index.md)).

---

## Summary

`ttnn.generic_op` is a *meta-operation* registered inside TT-Metal that accepts a fully pre-built `ProgramDescriptor` (or `MeshProgramDescriptor`) and executes it verbatim on hardware. Unlike every native TTNN op -- which owns output-spec inference, output-tensor allocation, program-factory selection, and program caching internally -- `generic_op` delegates **all** of those responsibilities to the caller. TT-Blaze fills that role: its `BlazeCompiler` assembles descriptors from a dataflow graph of composable micro-ops, then hands the finished artifact to `generic_op` for launch.

This design gives Blaze total control over kernel fusion, CB layout, and multi-device orchestration. But it also means Blaze programs bypass every TTNN infrastructure feature that lives inside the `DeviceOperation` template: automatic output-shape inference, validator hooks, the semantic program-cache hash, Tracy profiling annotations, and the graph-trace capture system. These divergences are not bugs -- they are the engineering consequences of Blaze's design decision to treat the hardware as a programmable substrate rather than an opaque library. Understanding them precisely is the first step toward promoting Blaze and blaze-nn ops to first-class TTNN citizens.

---

## Chapter Contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_generic_op_internals.md`](01_generic_op_internals.md) | `GenericOp` and `GenericOpDeviceOperation` internals -- struct layout, type aliases, program hashing, mesh workload factory, `ProgramDescriptor` / `MeshProgramDescriptor` data model |
| 2 | [`02_blaze_compilation_pipeline.md`](02_blaze_compilation_pipeline.md) | End-to-end Blaze compilation pipeline: `emit()` $\to$ `FusedProgram` $\to$ `BlazeCompiler.compile()` $\to$ `MeshCompiledProgram.run()` $\to$ `ttnn.generic_op()`. The `BlazeOp` hierarchy, engine stack, kernel codegen, `ProgramDescriptor` assembly, metadata discard analysis |
| 3 | [`03_divergence_analysis.md`](03_divergence_analysis.md) | Systematic divergence analysis: 15 dimensions of the TTNN op contract compared, with severity and priority classifications, cache pollution quantification, root cause analysis, and resolution paths |

---

## Key Takeaways

1. **`generic_op` is intentionally minimal.** It performs no validation beyond "at least two tensors" and "no duplicate mesh-coordinate ranges." All semantic correctness is the caller's job.

2. **The operation's semantics are inverted relative to native ops.** A native op like `ttnn.matmul` receives high-level parameters (shapes, fidelity, data types) and internally builds its `Program`. `generic_op` receives a fully-built `ProgramDescriptor` -- kernels, CBs, semaphores, runtime args already assembled -- and acts as a pass-through executor.

3. **Blaze owns the full program lifecycle.** `BlazeCompiler` runs CB, semaphore, and CT-arg engines; builds `KernelDescriptor` / `CBDescriptor` / `SemaphoreDescriptor` structs; and assembles them into a `ProgramDescriptor` per mesh coordinate. The result is a `MeshProgramDescriptor` containing one `ProgramDescriptor` per device.

4. **The dispatch bottleneck is the `ProgramDescriptor` boundary.** Because `generic_op` receives a fully opaque descriptor, TTNN cannot introspect it for caching, profiling, or autotuning. Any future promotion of Blaze ops to first-class TTNN ops must replace this opaque boundary with the structured `DeviceOperation` interface (see [Chapter 3](../ch03_compilation_model_gap/index.md)).

5. **15 concrete divergences** separate `generic_op` from native TTNN ops, classified into P0/P1/P2 priorities. The highest-impact gaps are profiling opacity, absent semantic validation, and no output-spec inference. Each maps to a specific change required for first-class registration.

---

## Cross-References

- **[Chapter 1](../ch01_ttnn_registration_contract/index.md)** (TTNN Registration Contract) defines the `DeviceOperationConcept` protocol that `GenericOpDeviceOperation` implements. The present chapter shows *how* `generic_op` implements that protocol and where it deviates.
- **[Chapter 3](../ch03_compilation_model_gap/index.md)** (Compilation Model Gap) will build on the divergence table here to propose concrete registration strategies.

---

## Source File Index

### TT-Metal (`generic_op` implementation)

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/generic/generic_op.hpp` | `GenericOp` struct with `invoke()` overloads |
| `ttnn/cpp/ttnn/operations/generic/generic_op.cpp` | SPMD $\to$ mesh-program adapter logic |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp` | `GenericOpDeviceOperation` -- types, factory variant, validation, hash |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp` | Implementation: `compute_output_specs`, `create_output_tensors`, `compute_program_hash` |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation_types.hpp` | Type aliases: `operation_attributes_t = MeshProgramDescriptor` |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.hpp` | `GenericMeshProgramFactory` declaration |
| `ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp` | `create_mesh_workload`, `override_runtime_arguments` |
| `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp` | Python bindings via nanobind |
| `tt_metal/api/tt-metalium/program_descriptors.hpp` | `ProgramDescriptor`, `KernelDescriptor`, `CBDescriptor`, `SemaphoreDescriptor` |
| `tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp` | `MeshProgramDescriptor` |

### TT-Blaze (compilation pipeline)

| File | Role |
|------|------|
| `blaze/blaze_op.py` | `BlazeOp` / `FusedOp` / `MicroOp` base classes, `emit()` / `compose()` contracts |
| `blaze/compiler.py` | `BlazeCompiler.compile()` -- graph-to-mesh-descriptor pipeline |
| `blaze/fused_program.py` | `FusedProgram` -- composition context, `build()`, `run()` |
| `blaze/program.py` | `BlazeProgram` -- CB/semaphore/CT-arg builder, `build()` $\to$ `ProgramDescriptor` |
| `blaze/graph.py` | `BlazeGraph`, `OpNode`, `Edge`, `OpSpec` -- dataflow IR |
| `blaze/cb_engine.py` | `CBEngine` -- assigns CB IDs and sizes from graph |
| `blaze/sem_engine.py` | `SemEngine` -- assigns global semaphore indices |
| `blaze/ct_args.py` | `CTArgEngine` -- resolves compile-time args per RISC |
| `blaze/kernel_codegen.py` | Generates C++ kernel source from `BlazeGraph` |
| `blaze/unified_kernel_descriptor.py` | `UnifiedKernelDescriptor` -- assembles `KernelDescriptor` triplets |

---

## Prerequisites

| Prerequisite | Why |
|---|---|
| [Chapter 1: The TTNN Op Registration Contract](../ch01_ttnn_registration_contract/index.md) | `generic_op` is defined in terms of the `DeviceOperationConcept`, `register_operation<>`, and `launch<>` machinery described there. |
| Basic Python familiarity | The Blaze compilation pipeline is implemented entirely in Python. |
| Understanding of RISC-V processor roles | Blaze generates per-RISC kernel descriptors (NCRISC/reader, BRISC/writer, TRISC/compute). |

---

| [Chapter 1: TTNN Registration Contract](../ch01_ttnn_registration_contract/index.md) | **Chapter 2** | [Chapter 3: Compilation Model Gap](../ch03_compilation_model_gap/index.md) |
|:---|:---:|---:|
