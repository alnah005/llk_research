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

---

## Pros and Cons of LLK Integration with TT-Metal
**Date:** 2026-04-22
**Status:** Completed
**Guide:** pros_and_cons_of_llk_integration_with_tt_metal/
**Why Needed:** TT-Metal consumes TT-LLK as its low-level kernel layer, but the integration design has significant architectural implications. Understanding the strengths and weaknesses of the current integration model — from build system coupling, to API abstraction boundaries, to hardware target configuration, to developer ergonomics — is essential for informing future architectural decisions about how these two layers should evolve together. The source code for TT-LLK lives at /localdev/salnahari/testing_dir/tt-llk and TT-Metal at /localdev/salnahari/testing_dir/tt-metal.
**Questions:**
- What are the advantages of the current submodule-based integration model (header inclusion, build-time compilation)?
- What are the disadvantages or pain points (tight coupling, version synchronization, build complexity)?
- How well does the current LLK API abstraction boundary serve TT-Metal op writers? Are there leaky abstractions?
- What are the pros and cons of LLK being header-only vs. a compiled library from TT-Metal's perspective?
- How does the per-architecture code duplication in LLK (wormhole_b0/, blackhole/, quasar/) affect TT-Metal's maintenance burden?
- What are the trade-offs of the current template-heavy API design for kernel compilation time and binary size?
- How well does the current testing strategy work across the LLK/TT-Metal boundary? Are there gaps?
- What alternative integration patterns could address identified weaknesses, and what would their trade-offs be?

---

## Pros and Cons of Writing TT-Metal Kernels Using LLK
**Date:** 2026-04-22
**Status:** Completed
**Guide:** pros_and_cons_of_writing_tt_metal_kernels_using_llk/
**Why Needed:** Kernel authors working in TT-Metal must navigate a multi-layered stack — from host-side op factories through the Compute Kernel API down to raw LLK primitives — to implement new compute kernels. Understanding the strengths and pain points of this current kernel-authoring workflow is essential for improving developer productivity, reducing bugs, and informing API design decisions. This analysis should cover the full authoring experience: the macro system (MATH/PACK/UNPACK), the init/tile pattern, template-heavy APIs, defines-as-configuration, compile-time vs runtime args, SFPU extension patterns, debugging tools, and the per-architecture considerations. The source code for TT-LLK lives at /localdev/salnahari/testing_dir/tt-llk and TT-Metal at /localdev/salnahari/testing_dir/tt-metal.
**Questions:**
- What are the advantages of the current TRISC macro system (MATH/PACK/UNPACK) for writing compute kernels? What are its pitfalls?
- How well does the Compute Kernel API abstract away LLK complexity? Where does the abstraction leak?
- What are the pros and cons of the defines-as-configuration pattern (host passes preprocessor defines to parameterize kernels)?
- How ergonomic is the init/acquire/compute/commit/wait/pack/release tile processing pattern? What are common mistakes?
- What are the trade-offs of the template-heavy API design for kernel authors (compile times, error messages, flexibility)?
- How effective are the current debugging tools (LLK_ASSERT, fake_kernels_target, generated file inspection) for kernel development?
- What is the experience of adding a new SFPU operation end-to-end? What friction points exist?
- How does per-architecture divergence (especially Quasar) affect kernel portability and developer burden?
- What improvements to the kernel-authoring workflow would have the highest impact on developer productivity?

**Findings:**
[Results of research go here]

---

## Minimum Requirements to Run a MatMul on P150 Using Pure LLK
**Date:** 2026-05-02
**Status:** Completed
**Guide:** minimum_requirements_to_run_a_matmul_on_p150_using_pure_llk/
**Why Needed:** Understanding the bare-minimum code, configuration, and infrastructure needed to execute a matrix multiplication on a Tenstorrent P150 chip using only the LLK (Low-Level Kernel) library — without TT-Metal's abstractions — is essential for developers who want to work at the lowest software layer. This research should produce a concrete, end-to-end understanding of what is required: from host-side setup and memory allocation, through kernel compilation and dispatch, to the actual LLK API calls on the Tensix cores. The source code for TT-LLK lives at /localdev/salnahari/testing_dir/tt-llk.
**Questions:**
- What is the P150 chip and which architecture variant (Blackhole, Wormhole B0, Quasar) does it correspond to in the LLK codebase?
- What host-side setup is required to initialize the device and allocate memory before running a kernel on P150?
- What is the minimal set of LLK API calls needed for a matrix multiplication (init, unpack, math, pack sequences)?
- What tile format, data format, and memory layout must be configured for the matmul operands and output?
- How are the three TRISC threads (unpack, math, pack) coordinated for a matmul operation using pure LLK?
- What kernel source files need to be written, and how are they compiled and loaded onto the Tensix cores without TT-Metal?
- What preprocessor defines, compile-time parameters, and configuration macros are required for a matmul kernel?
- Are there existing LLK test cases or examples for standalone matmul that can serve as a reference?
- What are the minimum dependencies (libraries, toolchain, firmware) needed to build and run a pure LLK matmul on P150?
- What synchronization, semaphore, and data movement considerations are unique to running without TT-Metal's runtime?
- What better alternatives exist to run a matmul on P150 that do not use LLK at any level of the stack (e.g., non-Tensix execution paths, host-side computation, alternative firmware, or frameworks that bypass LLK entirely)?

---

## Eliminating Explicit CCL Kernels via Global SRAM Pooling and Runtime-Managed Packet Hopping
**Date:** 2026-05-02
**Status:** Completed
**Guide:** eliminating_explicit_ccl_kernels_via_global_sram_pooling_and_runtime_managed_packet_hopping/
**Why Needed:** Today, multi-device communication on Tenstorrent systems requires explicit CCL (Collective Communication Library) kernels — hand-written or generated kernels that manage data movement between chips. This research investigates whether it is architecturally feasible to eliminate the need for these explicit kernels entirely by treating all SRAM across chips as a single global memory pool, with the runtime and/or hardware transparently handling packet hopping between devices. Understanding the feasibility, trade-offs, and requirements of this model is critical for simplifying multi-chip programming and scaling Tenstorrent systems. The source code for TT-Metal lives at /localdev/salnahari/testing_dir/tt-metal.
**Questions:**
- How do explicit CCL kernels work today in TT-Metal? What collective operations (all-gather, reduce-scatter, all-reduce, etc.) do they implement, and what is the programming model?
- What is the current SRAM (L1) architecture per chip, and how is inter-chip communication handled at the hardware level (Ethernet links, NOC, routing)?
- Is the hardware capable of transparently routing packets across chip boundaries, or does it require explicit software-managed data movement?
- What would a "global SRAM pool" abstraction look like — how would addresses be mapped across chips, and how would coherence or consistency be managed?
- What are the latency and bandwidth implications of packet hopping across chips compared to explicit CCL kernel orchestration?
- What runtime infrastructure would be needed to manage transparent cross-chip data movement (address translation, routing tables, flow control, deadlock avoidance)?
- Are there existing hardware features (e.g., Ethernet core firmware, NOC multicast/unicast capabilities) that could serve as building blocks for runtime-managed packet hopping?
- What are the trade-offs of implicit runtime-managed communication vs. explicit CCL kernels in terms of performance, predictability, debuggability, and resource utilization?
- How do other multi-chip architectures (e.g., Cerebras wafer-scale, GPU NVLink with NCCL, Graphcore IPU exchange) handle this problem, and what lessons apply?
- Under what workload patterns (model parallelism, data parallelism, pipeline parallelism) would a global SRAM pool model be most and least effective?
- What are the minimum hardware and firmware changes (if any) required to support this model on current Tenstorrent chips (Wormhole, Blackhole)?

---

