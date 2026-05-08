# Chapter 4 -- TT-Blaze Kernel Composition Framework

This chapter dissects the TT-Blaze kernel composition framework: the system that defines, composes, and compiles fused GPU-style kernels for Tenstorrent hardware. Where Chapter 2 covered TT-Symbiote's transparent op-by-op dispatch and Chapter 3 addressed multi-device weight sharding and CCL operations at the ttnn API level, this chapter descends to the kernel-authoring layer -- the BlazeOp class hierarchy, the FusedProgram composition context, the compilation pipeline that produces executable ProgramDescriptors, the CCL fabric integration at kernel level, and the catalog of 114 ops (54 MicroOps, 49 FusedOps, plus utility modules) that implement DeepSeek V3, GLM-5.1, and other model architectures.

## Contents

1. [**`01_blaze_op_architecture.md`**](./01_blaze_op_architecture.md) -- Blaze Op Architecture and Compilation Pipeline
   - The three-tier op hierarchy: `BlazeOp` (base), `FusedOp` (composition of MicroOps), `MicroOp` (backed by C++ kernel header). Port descriptors (`Input`, `Output`, `Internal`). Compile-time arguments (`ct_args`) with `Type.UINT32`, `Type.BOOL`; RISC processor targeting via the `Risc` Flag enum. The op lifecycle: class definition, `register()`, `compose()`/`emit()`. `FusedProgram`: multi-kernel programs with CB management (including `OverlappedView` and `CBAccessMode`), semaphore allocation, role engine, grid configuration, `MultiOutput`. `DeviceContext`: per-device compilation context. The BlazeGraph IR. The full compilation pipeline: op definition, graph construction, engine passes, program generation, and device execution via `ttnn.generic_op()`.

2. [**`02_ccl_in_blaze.md`**](./02_ccl_in_blaze.md) -- CCL Operations at the Blaze Kernel Level
   - `setup_fabric()` in `blaze/ccl.py` and its role in establishing inter-device fabric connections. Fabric semaphore allocation via `FusedProgram.semaphore()`. The `ttnn.compute_fabric_connection_rt_args()` API. Fabric kernel defines. How CCL operations (AllReduce, CCLBroadcast, ReduceToOne, Scatter) are implemented at kernel level in Blaze. The RISC processor model for CCL data movement. The `injected_fabric_barrier_builder.py` system for automatic inter-CCL synchronization, including barrier lifecycle in decode loops.

3. [**`03_ops_catalog_overview.md`**](./03_ops_catalog_overview.md) -- Ops Catalog and Composition Patterns
   - Survey of the `blaze/ops/` directory (114 ops: 54 MicroOps, 49 FusedOps, utility modules). DeepSeek V3 ops (MLA attention pipeline, MoE with routed/shared experts, gate routing). GLM-5.1 ops (fused projections, sparse attention, large MoE variants). Five named FusedOp composition patterns with worked examples including a full SharedExpert walkthrough. The `OverlappedView` system for weight packing. How ops handle multi-device via `mesh_coord`/`mesh_shape` injection. The `weight_provider` system for synthetic, state-dict, and Blitz-cache weight sourcing.

## Prerequisites

- Chapter 1 (Device Topologies) for mesh shapes, fabric configuration, and ethernet topology.
- Chapter 2 (Symbiote Core) for understanding how model inference dispatches to TTNN operations.
- Chapter 3 (Symbiote Multi-Device) for the higher-level CCL and weight sharding APIs that Blaze kernels ultimately implement.
- Familiarity with C++ kernel programming concepts (circular buffers, NOC, RISC processors) is helpful but not required -- this chapter explains the Python-side abstractions.

## Reading Guide

Developers building custom fused kernels should read File 01 end-to-end for the class hierarchy, composition API, and compilation pipeline. Those integrating CCL operations into fused programs should additionally read File 02. File 03 serves as a reference catalog -- skim it for the op closest to your use case, then study its source for implementation details. The MoE and MLA fused ops in File 03 are the most complex composition examples and are worth studying even if you are not working on those models.

---

**Next:** [`01_blaze_op_architecture.md`](./01_blaze_op_architecture.md)
