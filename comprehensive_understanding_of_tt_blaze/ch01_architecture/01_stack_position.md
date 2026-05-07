# 01 -- Stack Position

## Where Blaze Sits

TT-Blaze occupies a specific niche in the Tenstorrent software stack. It is a **kernel composition and dispatch layer** that sits above TT-Metal's dispatch API (program descriptors, CB descriptors, kernel descriptors) but below model-level application code. Its purpose is to let developers write handwritten C++ kernel ops and compose them into fused multi-op pipelines entirely in Python, without manually wiring circular buffers, semaphores, compile-time arguments, or core grids.

The full stack, from bottom to top:

| Layer | Role |
|---|---|
| **Hardware** (Tensix cores, NOC, L1, DRAM) | Physical execution substrate -- Blackhole is the primary target, Wormhole B0 is secondary. |
| **TT-LLK** | Low-Level Kernel library. SFPU intrinsics, tile math, data-movement primitives that run on individual RISC processors inside each Tensix core. |
| **TT-Metal** | Hardware abstraction and dispatch. Provides `ProgramDescriptor`, circular-buffer management, NOC primitives, compile-time/runtime arg infrastructure, and the `generic_op` dispatch path. |
| **TT-Blaze** | Kernel composition layer. Lets developers hand-author C++ micro-ops, compose them into fused pipelines in Python, and dispatch them through TT-Metal. |
| **TT-Forge** | Graph compiler. Lowers an ML framework graph (PyTorch, ONNX) into optimized TT-Metal programs via automatic tiling, placement, and scheduling. |
| **Model code** (Python) | Defines the network graph -- attention, MLP, MoE routing, pipeline orchestration. |

Blaze does not replace TT-Metal; it automates the mechanical boilerplate that TT-Metal requires. A `FusedProgram` in Blaze ultimately produces a `ttnn.ProgramDescriptor` with kernel descriptors, CB descriptors, and semaphore descriptors -- the same primitive objects that a hand-rolled TT-Metal op would construct. The difference is that Blaze derives these programmatically from a declarative graph of ops rather than requiring manual bookkeeping.

## Blaze vs. TT-Forge

TT-Forge and Blaze are **peers targeting different workflows**:

| | TT-Forge | TT-Blaze |
|---|---|---|
| **Input** | PyTorch / ONNX model graph | Handwritten C++ kernel ops |
| **Core mechanism** | Graph compiler lowers high-level ops to generated kernels | Composition framework that fuses developer-authored kernel ops |
| **Target user** | ML engineers who want automatic optimization | C++ kernel developers who want explicit control |
| **Who writes the kernel** | The compiler | The developer |

TT-Forge takes a framework-level graph, applies graph transformations and scheduling, and emits optimized kernel code -- the developer never writes C++ kernel code. Blaze takes the opposite approach: the developer **hand-writes** C++ kernel structs (micro-ops) that run on individual RISC processors, then uses Blaze's Python API to **compose** those ops into fused pipelines. Blaze handles the wiring; the developer retains full control over what runs on the hardware.

In production inference scenarios (e.g., DeepSeek V3 decode), the performance-critical path demands hand-tuned kernels with precise control over L1 memory layout, NOC communication patterns, and RISC processor scheduling. Blaze provides the composition framework for these kernels. Both systems ultimately produce `ProgramDescriptor` objects that TT-Metal dispatches identically.

## Relationship to TT-LLK

TT-LLK (Low-Level Kernel library) provides the SFPU and math intrinsics that Tensix compute cores use for tile-level operations: matrix multiplies, elementwise functions, unpacking, and packing. Blaze kernel code calls LLK primitives extensively.

Blaze brings its own LLK extensions via the `kernel_includes/` directory tree under `blaze/kernels/`. This tree mirrors TT-Metal's LLK header layout and adds custom SFPU functions that do not exist in upstream TT-LLK. The kernel compilation include path places `kernel_includes/` before Metal's own headers, so Blaze's versions shadow the upstream ones where needed. Blaze also adds custom kernel headers (`ops.hpp`, `kernel_op_api.hpp`, `ct_types.h`) and standalone SFPU extensions. See [`02_repository_map.md`](./02_repository_map.md) Section "Kernel Headers" for the full directory-by-directory breakdown.

## The `tt-metal` Submodule

Blaze pins TT-Metal as a git submodule under `tt-metal/`. The `.gitmodules` file points to `git@github.com:tenstorrent/tt-metal.git`. The submodule is pinned to a specific commit on TT-Metal's `main` branch (the `.gitmodules` file has no `branch` field; the reference to a `blaze-metal` branch in the README is a historical artifact -- no such branch exists on the remote). The features listed below are available at the pinned commit:

- **C++20 kernel compilation** -- enables `constexpr if`, structured bindings, and other C++20 features in Tensix kernel code, which Blaze ops rely on extensively for compile-time dispatch across RISC processors.
- **Named compile-time argument infrastructure** -- instead of positional `get_compile_time_arg_val(index)`, kernels declare named CT args via C++ structs and access them as typed fields. `BlazeProgram`'s `.unified_ct_args()`, `.ncrisc_ct_args()`, etc., emit these named arguments, and the `UnifiedKernelDescriptor` passes them through to TT-Metal's `KernelDescriptor`.
- **Named runtime argument infrastructure** -- runtime args use the same namespace-qualified naming (`"prefix.field"`) with `constexpr` accessors generated in a header, accessed on-device via `rt::get<rt::ns::field>()`.
- **`ProgramDescriptor` / `CBDescriptor` API** -- the descriptor-based program assembly that `BlazeProgram.build()` targets.

The C++20 features above enable Blaze's "one kernel, all RISCs" compilation model, where a single unified kernel source file dispatches to the correct op struct per-RISC at compile time via `SelectByRISCV` (defined in `kernel_op_api.hpp`). See [`02_repository_map.md`](./02_repository_map.md) Section "Kernel Headers" for the code patterns and type definitions used by these headers.

## Why Blaze Exists

The project README states the core philosophy directly: **"C++ kernel developers are first-class citizens -- you write real tt-metal kernel code, Blaze makes it composable."** The motivation is a concrete productivity problem: writing fused ops on TT-Metal requires extensive boilerplate for CB allocation, semaphore management, compile-time argument wiring, and core grid configuration. This boilerplate is error-prone and grows multiplicatively with fusion depth.

The README quantifies the claim with a specific example: the `SharedExpert` FusedOp in Blaze (67 lines of `emit()` composition logic; 115 lines including the full `op.py`) replaces 979 lines of manual wiring in the equivalent TT-Metal implementation -- approximately a **$14\times$ code reduction**. SharedExpert chains four building blocks (KNMatmul → GatedReduce → Mcast → DownProj), each encapsulating CB allocation, CT arg registration, per-core flags, and semaphore setup. The composition is plain Python -- no graph compiler, no IR transforms, no code generation required for the wiring itself. For the full SharedExpert walkthrough, see [Ch 2 Section 06 -- Writing a FusedOp](../ch02_blazeop_hierarchy/06_writing_a_fused_op.md).

The reduction comes from three concrete sources:

1. **Automatic CB/semaphore allocation.** `BlazeProgram` allocates CB IDs sequentially, constructs `CBDescriptor` objects from tensors, and manages semaphore slots per-core. Manual Metal code must track all of this by hand.
2. **Composable `emit()` pattern.** Each `MicroOp.emit()` adds its CB args, CT args, and role flags to a `FusedProgram`. The output `CBHandle` of one `emit()` feeds directly as input to the next -- no manual wiring between op boundaries.
3. **Named CT/RT arg infrastructure.** Instead of managing hundreds of positional arg indices, Blaze uses scoped names (`prefix.field`). The `CTArgEngine` validates that every arg the kernel reads is supplied and every supplied arg is read.

The design philosophy has four pillars:

1. **Kernel code is real C++**, not a DSL or IR. The `op.hpp` header for each MicroOp is a genuine C++ struct with `init()`, `operator()()`, and `teardown()` lifecycle methods that call LLK intrinsics directly.
2. **Composition is Python.** The `emit()` method on each op allocates CBs, sets CT args, and wires outputs to inputs -- all in Python, all explicit.
3. **No hidden compiler passes.** Blaze's "compilation" is CB engine assignment, semaphore engine assignment, and CT arg generation -- mechanical transformations on an explicit graph, not optimization passes that restructure the computation.
4. **Rapid iteration.** Write a new op, compose it with existing ops, test on silicon. The build cycle is the C++ kernel compile (seconds) plus Python-level composition (milliseconds). The op library (over 100 registered ops spanning matmul, attention, MoE gating, CCL, RMSNorm, RoPE, and data movement) means new models should mostly assemble existing building blocks.

---

**Next:** [`02_repository_map.md`](./02_repository_map.md)
