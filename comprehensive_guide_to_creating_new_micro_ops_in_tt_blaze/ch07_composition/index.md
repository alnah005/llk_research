# Chapter 7: Composing Micro-Ops into Fused Operations

## Overview

Previous chapters covered the anatomy of individual micro-ops -- the atomic building blocks of TT-Blaze. A single micro-op handles one hardware concern: a multicast, a matrix multiply, a gather, a residual addition. Real model layers, however, chain dozens of these concerns into a single fused kernel that runs without returning to the host. This chapter explains the machinery that makes that composition work.

The central abstraction is `FusedOp`, a subclass of `BlazeOp` defined in `blaze/blaze_op.py`. Where a `MicroOp` wraps a C++ kernel header and exposes a single `emit()` function, a `FusedOp` stitches multiple `emit()` calls together inside a `compose()` classmethod. The compiler calls `compose()`, which in turn calls each child micro-op's `emit()`, threading `CBHandle` objects from one stage's output to the next stage's input. The result is a single kernel binary whose RISC processors execute all stages in sequence, with data flowing through on-chip circular buffers rather than DRAM round-trips.

This chapter is organized into three sections:

1. **FusedOp Class Structure** (Section 01) -- The `FusedOp` base class, the contract between `compose()` and `emit()`, the role of the `kernel` attribute, directory layout conventions, the `FusedOpConfig` registration mechanism, and the Graph API path through `graph_call()`.

2. **Prefix Namespaces and Shared Resources** (Section 02) -- How the `child_prefix()` method with its `__` delimiter prevents compile-time argument collisions, how `cb_name()` with its `___` delimiter prevents circular buffer collisions, how semaphores are scoped, how scratch CBs flow between stages and are shared across disjoint core grids, how `ABGrid` partitions the compute grid, how `reconfig()` enables temporal CB compaction, and how sub-FusedOps embed within parent FusedOps.

3. **Real-World Example: SharedExpert** (Section 03) -- An annotated line-by-line walkthrough of the `SharedExpert` fused op, which chains `KNMatmul`, `GatedReduce`, `Mcast`, and `DownProj` into a complete gated MLP pipeline. This section traces the `CBHandle` chain from activation input through gate/up matmul, gated SiLU reduction, multicast fan-out, down-projection matmul, bias addition, and final gather output. It also examines how `DownProj` is itself a sub-FusedOp embedded within `SharedExpert`, and provides a complete resource inventory, debugging guidance, and a checklist for creating new FusedOps.

## Prerequisites

Readers should be comfortable with:

- **Chapter 2**: The `BlazeOp` class hierarchy (`BlazeOp` -> `MicroOp` / `FusedOp`), `Input`/`Output` port descriptors, and the `emit()` contract.
- **Chapter 3-4**: The `emit()` function contract, `CBHandle`, and `FusedProgram` -- how `FusedProgram` manages CBs, compile-time arguments, semaphores, and the shadow graph.
- **Chapter 5**: Compile-time argument registration (`unified_ct_args`, `per_core_unified_ct_args`, `flag()`) and writing a single micro-op's `emit()` method.
- **Chapter 6**: Circular buffer allocation (`cb_scratch`, `cb_from_tensor`) and how per-core flags (`f.flag()`) control which RISC processors run which phases on which cores.

## Key Source Files

| File | Role |
|------|------|
| `blaze/blaze_op.py` | `BlazeOp`, `FusedOp`, `MicroOp` base classes; `FusedOpConfig`; `child_prefix()` and `cb_name()` |
| `blaze/fused_program.py` | `FusedProgram` context; `CBHandle`; `ABGrid`; `TileInfo`; `cb_scratch()`; `semaphore()` |
| `blaze/cb_handle.py` | `CBHandle` dataclass; `CBAccessMode`; `require_fifo_handle()` |
| `blaze/compiler.py` | `BlazeCompiler`; `_compile_fused_op()` (line 905); `config.compose_fn()` invocation (line 961) |
| `blaze/context.py` | `FusionContext`; `_op_call()`; `ExternalTensor`; `blaze.fuse()` context manager |
| `blaze/ops/shared_expert/op.py` | `SharedExpert` fused op |
| `blaze/ops/down_proj/op.py` | `DownProj` sub-fused op |
| `blaze/ops/kn_sliced_matmul/op.py` | `KNMatmul` micro-op |
| `blaze/ops/gated_reduce/op.py` | `GatedReduce` micro-op |
| `blaze/ops/mcast/op.py` | `Mcast` micro-op |
| `blaze/ops/gather/op.py` | `Gather` micro-op |
| `blaze/ops/residual_add/op.py` | `ResidualAdd` micro-op |
| `blaze/ops/dense_mlp/op.py` | `DenseMLP` higher-level fused op embedding `SharedExpert` |

## Terminology

- **FusedOp**: A `BlazeOp` subclass whose `compose()` chains multiple micro-op `emit()` calls into a single kernel. Does not have its own C++ Op struct. The kernel is either auto-generated from the shadow graph recorded during `compose()` or provided as a handwritten source file via the `kernel` attribute.

- **Sub-FusedOp**: A `FusedOp` embedded inside another `FusedOp`. `DownProj` is a sub-FusedOp within `SharedExpert`. Sub-FusedOps are just FusedOps whose `emit()` is called from another FusedOp's `emit()`, receiving a scoped prefix.

- **CBHandle chain**: The sequence of `CBHandle` objects returned by successive `emit()` calls within a `compose()`. Each handle carries `cb_id`, `num_pages`, `page_size`, `core_ranges`, `data_format`, and `tile_desc` metadata that the next stage consumes.

- **Prefix namespace**: A hierarchy-delimited string (e.g., `"shared_expert__gu"`) that scopes all compile-time argument names, CB names, and semaphore names within a composed pipeline. The `__` delimiter is used by `child_prefix()` and the `___` delimiter by `cb_name()`.

- **Shadow graph**: An internal `BlazeGraph` maintained by `FusedProgram` that records every `f.output()` call. Used for kernel auto-generation, visualization, and debugging. Each call to `f.output()` creates a node; CBHandle connections create edges.

## How to Read This Chapter

Section 01 establishes the structural rules: what makes a FusedOp different from a MicroOp, how `compose()` bridges the compiler to `emit()`, and what happens when a FusedOp has its own `kernel` attribute versus relying on codegen.

Section 02 dives into the naming and resource-sharing mechanics that make multi-stage composition possible. Without proper prefix namespacing, two Mcast stages in the same pipeline would collide on CT-arg names. Without shared scratch CBs, data cannot flow between stages. This section explains how the framework solves both.

Section 03 puts everything together with a real production op. SharedExpert is complex enough to illustrate all the patterns -- hierarchical prefix nesting, ABGrid core partitioning, sub-FusedOp embedding, cross-stage CB handoff, and multi-semaphore coordination -- yet small enough to trace in its entirety.

By the end of this chapter, you will know how to design, implement, and debug a multi-stage fused operation from scratch.

## Chapter Map

```
Chapter 7
  |
  +-- 01_fused_op_structure.md
  |     FusedOp class, compose() vs emit(), kernel attribute,
  |     directory layout, FusedOpConfig registration, Graph API path
  |
  +-- 02_prefix_namespaces_and_shared_resources.md
  |     Unique prefixes, child_prefix with __ delimiter,
  |     cb_name with ___ delimiter, shared semaphores,
  |     shared scratch CBs, ABGrid partitioning, reconfig() boundaries,
  |     sub-FusedOp embedding, TileInfo propagation
  |
  +-- 03_real_world_example.md
        SharedExpert annotated walkthrough with data flow,
        sub-FusedOp DownProj, CBHandle chain, resource inventory,
        debugging guidance, new FusedOp checklist
```
