# Chapter 2: The Unified Kernel System

## Overview

Chapter 1 introduced the five-step execution pattern that every DeepSeek V3 B1 operation follows: allocate tensors, describe circular buffers, describe kernels, build a `ProgramDescriptor`, and dispatch with `generic_op`. This chapter opens the hood on the *kernel system itself* -- the infrastructure that makes the five-step pattern scale from a three-CB matmul to a sixty-CB fused MoE pipeline without any structural change to the programming model.

The key insight is **fractal self-similarity**: the same `UnifiedKernelDescriptor` dataclass, the same `struct Op` pattern in C++, and the same `CircularBufferIdManager` allocation scheme compose at every level of complexity. What changes between a simple matmul and the fused MoE op is not the *kind* of infrastructure, but the *amount* -- more CBs, more kernel groups, more role descriptors. The infrastructure itself is invariant.

We develop this insight through a four-level **complexity ladder**:

| Level | Example | CBs | Kernel Groups | Descriptor Path |
|-------|---------|-----|---------------|-----------------|
| 1 | Matmul | 3 | 1 | Simple |
| 2 | Mcast / Gather | 2-3 | 2-3 | Split (sender/receiver) |
| 3 | SharedExpertOp | 15 | 8 | Split (multi-role) |
| 4 | MoeOp | 40+ | 10+ | Split + CB reconfig |

At each level, the chapter traces the exact Python code that builds the descriptors and the exact C++ kernel code that consumes them, showing how the same abstractions stretch without breaking.

## Sections

1. **[The Unified Kernel Descriptor](./01_unified_kernel_descriptor.md)**
   The Python-side infrastructure: `UnifiedKernelDescriptor`, RISC roles, the simple vs. split code paths, `UnifiedCompileTimeCoreDescriptor`, `PerCoreCompileTimeDescriptor`, `PerCoreRuntimeArgsDescriptor`, `KernelGroup`, and `UnifiedKernelResult`. Includes an end-to-end Gather trace from Python to NOC write.

2. **[The Kernel Op API and Unified Headers](./02_kernel_op_api_and_unified_headers.md)**
   The C++ side: `kernel_op_api.hpp` RISC detection and `SelectByRISCV`, the canonical `struct Op` pattern with `constexpr` dead-code elimination, `kernel_utils.hpp` grid utilities and synchronization primitives, the 27-header micro-op catalog organized into five structural categories, and a DRAMStreamingMatmul deep dive.

3. **[Circular Buffer Management](./03_circular_buffer_management.md)**
   `CircularBufferIdManager` and its `Context` scoping, three `CBDescriptor` construction patterns, the `build_dummy_cb_descriptors` / `build_cb_reconfig_tensor` pipeline with its 264-word per-core memory layout, the four-RISC synchronization barrier, and a complete MoeOp reconfig flow with a worked CB trace through hex addresses.

## Key Source Files

| Source File | Role |
|---|---|
| `unified_kernel_descriptor.py` (461 lines) | Python-side descriptor generation |
| `unified_kernels/kernel_op_api.hpp` (38 lines) | RISC detection, `SelectByRISCV` |
| `unified_kernels/kernel_utils.hpp` (197 lines) | Grid math, sync, CB reconfig |
| `unified_kernels/*.hpp` (25 op headers) | Reusable micro-op library |
| `circular_buffer_utils.py` (224 lines) | CB ID management, reconfig tensor |

---

*Previous: [Chapter 1 -- The Five-Step Execution Pattern](../ch1_final/index.md)* | *Next: [Chapter 3 -- The Micro-Op Library](../ch03_the_micro_op_library/index.md)*
