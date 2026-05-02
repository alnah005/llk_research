# Chapter 1: The Current CCL Programming Model -- What We Are Proposing to Eliminate

This chapter establishes the baseline by documenting how explicit Collective Communication Library (CCL) kernels work today in TT-Metal. It catalogs every operation type, dissects the device operation lifecycle using all-gather as a canonical example, explains the command-stream programming model that drives device-side kernel execution, and maps the host-side orchestration that binds topology discovery, communication partner selection, and op fusion together. Together, these four sections constitute the "problem statement" that motivates the entire guide: the complexity, rigidity, and programmer-visible surface area that a Global SRAM Pool abstraction would aim to eliminate.

## Files

| # | File | Description |
|---|------|-------------|
| 1 | [`01_collective_operations_catalog.md`](./01_collective_operations_catalog.md) | Catalogs every CCL operation currently implemented -- all-gather, reduce-scatter, all-reduce, all-broadcast, broadcast, reduce-to-root, all-to-all-dispatch, all-to-all-combine, and mesh-partition -- with API signatures, supported topologies, memory configurations, data types, and experimental variants. |
| 2 | [`02_device_operation_anatomy.md`](./02_device_operation_anatomy.md) | End-to-end walkthrough of a single CCL operation (all-gather as canonical example), tracing from `ExecuteAllGather::invoke` through device operation attributes, the mesh workload factory, per-device program creation, global semaphore allocation, worker core selection, and runtime argument override. |
| 3 | [`03_command_stream_architecture.md`](./03_command_stream_architecture.md) | The CCL command-stream abstraction: opcodes (`CclCommandCode`, `CclCommandArgCode`), command argument variants, the lowering pipeline from host-level commands to device kernel arguments, host-side builders, slice descriptors, sharding address generation, and a worked example showing concrete command encoding for a ring all-gather. |
| 4 | [`04_host_orchestration_and_topology.md`](./04_host_orchestration_and_topology.md) | How CCL operations discover communication partners (`SenderReceiverConfig`, `RingTopology`, `LineTopology`), boundary modes and topology degradation logic, CCL op fusion for overlapping communication with computation, and the transition from the legacy ERISC async datamover path to the newer fabric-based CCL path. |

## Reading Guide

Readers already familiar with TT-Metal's per-chip memory architecture (L1 SRAM, NOC addressing, circular buffers) can read this chapter directly. Readers who are new to the hardware should consider reading Chapter 2 first for foundational context, then returning here. This chapter is intentionally positioned first because it frames the central question: what exactly does "eliminating explicit CCL kernels" mean in practice?

Each file builds on the previous: the catalog (file 01) establishes what operations exist, the anatomy (file 02) shows how one operation works end-to-end, the command stream (file 03) explains the data-movement instruction set, and the orchestration (file 04) covers how the host coordinates everything.

## Key Source Directories

- `ttnn/cpp/ttnn/operations/ccl/` -- Top-level CCL operation definitions
- `ttnn/cpp/ttnn/operations/ccl/common/` -- Command stream infrastructure and shared utilities
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/` -- Canonical device operation example
- `ttnn/cpp/ttnn/operations/experimental/ccl/` -- Experimental and async CCL variants
