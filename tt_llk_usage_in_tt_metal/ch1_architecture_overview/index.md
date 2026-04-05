# Chapter 1: Architecture Overview

This chapter provides a bird's-eye view of how TT-Metal consumes the TT-LLK (Low-Level Kernel) library, where each codebase's responsibilities begin and end, and the fundamental hardware concepts that shape the entire software stack.

## What You Will Learn

1. **Codebase Layout** -- How the TT-LLK repository is organized per architecture (Blackhole, Wormhole B0, Quasar) and which directories in TT-Metal bridge the gap between high-level operations and low-level kernel primitives.
2. **Key Concepts** -- The Tensix core architecture with its three TRISC processors, the tile-based computation model, and the macro system (`MATH()`, `PACK()`, `UNPACK()`) that allows a single kernel source file to be compiled into three separate binaries.

## Reading Order

| File | Topic |
|------|-------|
| [`codebase_layout.md`](./codebase_layout.md) | Directory structures, layering diagram, submodule relationship |
| [`key_concepts.md`](./key_concepts.md) | Tensix core architecture, tile-based data flow, TRISC macros |

## Relationship Summary

TT-Metal is the **consumer**. TT-LLK is the **library**. TT-Metal vendors TT-LLK as a Git submodule at `tt_metal/third_party/tt_llk/` and wraps its internal functions with higher-level API headers that live in the `tt_metal/hw/ckernels/` tree. User-visible compute kernel APIs (e.g., `matmul_tiles`, `pack_tile`, `cb_wait_front`) are defined in `tt_metal/hw/inc/api/compute/` and delegate to these wrappers, which in turn call into the architecture-specific LLK implementations.

---

**Next:** [`codebase_layout.md`](./codebase_layout.md)
