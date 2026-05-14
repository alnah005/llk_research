# Chapter 3: The `emit()` Method -- Bridging Python to C++

## Overview

`emit()` is the single most important method in the TT-Blaze framework. It is the boundary where a Python-level op definition becomes a concrete hardware program: circular buffers are allocated, compile-time arguments are registered for each RISC processor, and an output handle is created so the next op in the chain can consume this op's results without any coupling to its internals.

Every MicroOp must implement `emit()` as a `@staticmethod` that takes a `FusedProgram` as its first parameter. The method follows a strict three-step structure that this chapter dissects in detail.

## Sections

### 1. The Three-Step `emit()` Structure (01_three_step_structure.md)

The canonical pattern every `emit()` follows: allocate resources, declare CT args per RISC, and return an output handle via `f.output()`. Covers the `@staticmethod` convention, the `prefix` and `cores` parameters, how CT arg tuples map to generated C++ header fields, CBHandle chaining between ops, and a complete annotated walkthrough of `RMSNorm.emit()`.

**Read this first** -- it provides the mental model for everything else in the chapter.

### 2. CB Allocation Methods (02_cb_allocation.md)

Deep dive into every CB allocation method on FusedProgram: `cb_from_tensor()`, `cb_scratch()`, `cb_alias()`, `cb_from_view()`, `cb_from_shared_l1_views()`, and `cb_output()`. Covers scratch naming conventions, page size derivation, the `balanced` flag for temporal reuse, CB ID limits (64 on Blackhole), and disjoint-core CB sharing.

### 3. Role Flags and CT Arg Registration (03_role_flags_and_ct_args.md)

How ops declare which cores are active, which are senders vs. receivers, and what compile-time arguments each RISC processor receives. Covers `f.flag()`, per-RISC CT arg methods, `per_core_unified_ct_args()` and per-RISC per-core variants, the CTArgEngine's auto-prefixing and type validation, `CBHandle.__int__()` coercion, `_float_to_bits()`, semaphore allocation, runtime args (including the C++ `rt_args::get<>()` access pattern), and the CT vs. RT arg recompilation tradeoff.

### 4. CBHandle Data Flow (04_cbhandle_data_flow.md)

The `CBHandle` dataclass as the universal currency of inter-op data flow. Covers every field, FIFO vs. direct-address access modes, `require_fifo_handle()`, handle chaining across ops, zero-coupling between ops, `MultiOutput` for multi-output ops, and metadata propagation through the shadow graph.

## Reading Order

1. **01_three_step_structure.md** -- Understand the pattern before the details.
2. **02_cb_allocation.md** -- Step 1 in depth: how resources get allocated.
3. **03_role_flags_and_ct_args.md** -- Step 2 in depth: telling each RISC what to do.
4. **04_cbhandle_data_flow.md** -- Step 3 in depth: how ops connect without coupling.

## Prerequisites

- **Chapter 1:** MicroOp class structure and the BlazeOp base
- **Chapter 2:** `compose()` and how the compiler invokes it
- Familiarity with Tenstorrent's RISC processors (NCRISC, BRISC, TRISC)

## Key Source Files

| File | Role |
|------|------|
| `blaze/fused_program.py` | FusedProgram class with all CB allocation and CT arg methods |
| `blaze/cb_handle.py` | CBHandle dataclass and CBAccessMode enum |
| `blaze/cb_engine.py` | CB ID assignment engine, MAX_CB_ID constant |
| `blaze/ct_args.py` | CTArgEngine, CTArgSchema, CTArgSpec |
| `blaze/blaze_op.py` | BlazeOp base class, MicroOp, CB_NAME_DELIMITER |
| `blaze/ops/rmsnorm/op.py` | RMSNorm emit() -- primary walkthrough |
| `blaze/ops/rmsnorm/kernels/op.hpp` | RMSNorm C++ kernel -- `if constexpr` pattern |
| `blaze/ops/copy/op.py` | Copy emit() -- data-movement example |
| `blaze/ops/mcast/op.py` | Mcast emit() -- inter-core example |
| `blaze/ops/rt_args_demo/kernels/` | Runtime args demo kernels |
