# Research Topics

This file tracks research topics that the Architect needs to investigate for making informed decisions.

---

## Format

```
## [Topic Name]
**Date:** YYYY-MM-DD
**Status:** Pending | In Progress | Completed
**Why Needed:** [Reason this research is necessary]
**Questions:**
- Question 1
- Question 2

**Findings:**
[Results of research go here]

---
```

## Topics

---

## Introduction to TT-LLK (Tenstorrent Low-Level Kernel Library)
**Date:** 2026-04-05
**Status:** Completed
**Guide:** introduction_to_tt_llk/
**Why Needed:** Developers and op writers need a comprehensive introduction to the TT-LLK library — the foundational software layer for programming Tenstorrent Tensix cores. The library is header-only C++17 with hardware-specific implementations across three chip architectures (Blackhole, Wormhole B0, Quasar), and understanding its structure, APIs, and programming model is essential for writing efficient kernels. The source code lives at /localdev/salnahari/testing_dir/tt-llk.
**Questions:**
- What is TT-LLK and where does it sit in the Tenstorrent software stack (relative to TT-Metal and TT-Forge)?
- What is the Tensix Core architecture and how do its 5 RISC-V cores (B, T0, T1, T2, NC) and 3-way threaded coprocessor model work?
- How is data organized (tiles, data formats, L1 memory) and how do unpack/math/pack operations form the core compute pipeline?
- What are the key LLK APIs — eltwise operations, matrix multiply, reduce, transpose — and how are they structured as C++ templates?
- How do the three hardware targets (Blackhole, Wormhole B0, Quasar) differ in their LLK implementations?
- What is the SFPU (Scalar Floating Point Unit) and what operations does it support?
- How does the test infrastructure work and how do developers run and write tests?
- What conventions, data formats (FP32, FP16, BFP8, Int8, etc.), and synchronization mechanisms (DstSync, semaphores) must kernel writers understand?

**Findings:**
[Results of research go here]

---

## TT-LLK Usage in TT-Metal
**Date:** 2026-04-05
**Status:** Completed
**Guide:** tt_llk_usage_in_tt_metal/
**Why Needed:** Understanding how TT-Metal consumes and integrates the TT-LLK library is critical for developers working across the stack. TT-Metal is the higher-level runtime/compiler layer, and TT-LLK provides the low-level kernel primitives it builds on. Mapping the integration points, call patterns, and build-system linkage helps developers understand how to write kernels that work correctly within TT-Metal, and how changes in LLK propagate to the Metal layer. The source code for TT-LLK lives at /localdev/salnahari/testing_dir/tt-llk and TT-Metal at /localdev/salnahari/testing_dir/tt-metal.
**Questions:**
- How does TT-Metal reference and include TT-LLK? What is the build-system integration (CMake, submodule, header paths)?
- Which TT-Metal components directly call LLK APIs (e.g., kernel dispatch, op implementations, compute kernels)?
- How does TT-Metal's kernel compilation pipeline use LLK headers — are they compiled per-kernel, per-op, or globally?
- What is the mapping between TT-Metal ops/operations (e.g., matmul, eltwise, reduce) and the underlying LLK API calls?
- How does TT-Metal configure LLK for different hardware targets (Blackhole, Wormhole B0, Quasar)?
- What runtime parameters does TT-Metal pass to LLK kernels, and how are they plumbed through?
- How do TT-Metal's compute kernel source files (the .cpp files that run on Tensix) use LLK includes and macros?
- Are there abstraction layers or wrappers in TT-Metal between the user-facing op API and raw LLK calls?

**Findings:**
[Results of research go here]

---

