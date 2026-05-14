# Chapter 1: Architecture and Mental Model

## What you will learn

After completing this chapter you will be able to:

- Describe the Tensix core's internal structure: its five RISC-V processors, three logical roles (DM0, DM1, TRISC), and their division of labor.
- Explain how L1 SRAM, circular buffers, and semaphores provide the intra-core and inter-core synchronization primitives that micro-ops depend on.
- Trace TT-Blaze's compilation pipeline from Python graph construction through engine processing to JIT compilation and device dispatch.
- Distinguish between MicroOps and FusedOps, understand their class hierarchy, and know when to use each abstraction.
- Read any TT-Blaze kernel header (`op.hpp`) or Python op file (`op.py`) and immediately identify the per-processor responsibilities, CB wiring, and CT arg flow.

## Chapter contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_tensix_core_model.md`](./01_tensix_core_model.md) | The Tensix core: processors, L1 SRAM, circular buffers, semaphores, and parallel execution |
| 2 | [`02_blaze_compilation_pipeline.md`](./02_blaze_compilation_pipeline.md) | From `blaze.fuse()` to device dispatch: graph construction, engine pipeline, JIT, and the two compilation modes |
| 3 | [`03_micro_op_vs_fused_op.md`](./03_micro_op_vs_fused_op.md) | MicroOp vs. FusedOp: the class hierarchy, auto-discovery, registration, and the dual API surface |

## Reading order

1. **Start with the hardware model.** Read [`01_tensix_core_model.md`](./01_tensix_core_model.md) to build the mental model of the Tensix core, its three processor groups, and the CB/semaphore synchronization primitives. Every design decision in TT-Blaze traces back to this hardware.

2. **Understand the compilation pipeline.** Read [`02_blaze_compilation_pipeline.md`](./02_blaze_compilation_pipeline.md) to see how Python-level `emit()` calls and `blaze.fuse()` graph construction flow through the engine pipeline into JIT-compiled, per-processor kernel binaries dispatched to hardware.

3. **Learn the op abstractions.** Read [`03_micro_op_vs_fused_op.md`](./03_micro_op_vs_fused_op.md) to understand the MicroOp and FusedOp class hierarchy, auto-registration, and the dual API surface (`blaze.<name>(...)` for graphs, `blaze.<name>.emit(f, ...)` for composition). This sets the stage for Chapter 2, where you will write your first MicroOp from scratch.

## Prerequisites

- Familiarity with Python class hierarchies and descriptors
- Basic understanding of GPU/accelerator programming concepts (tiles, shards, data movement)
- Access to the `tt-blaze` repository (all file paths in this chapter are relative to the repository root)

---

**Next:** [`01_tensix_core_model.md`](./01_tensix_core_model.md)
