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

## Runtime Interceptor for LLK Kernel Capture and GDB-Style Replay Debugging
**Date:** 2026-05-04
**Status:** Completed
**Guide:** runtime_interceptor_for_llk_kernel_capture_and_gdb_style_replay_debugging/
**Why Needed:** Debugging kernels on Tenstorrent hardware is difficult because the execution happens on Tensix RISC-V cores with limited visibility. CUDA developers can attach cuda-gdb to individual GPU threads, set breakpoints, inspect registers, and single-step through kernel code. Tenstorrent has no equivalent workflow today. This research investigates the feasibility of building a runtime interceptor layer in TT-Metal that captures everything needed to reproduce an LLK kernel invocation — dispatch commands, arguments, memory state, configuration defines, tile data — and replay it in a GDB-style debugger (potentially on a RISC-V emulator or the actual hardware via JTAG/debug firmware). Understanding what cuda-gdb actually does under the hood, what TT-Metal's current debug infrastructure provides, and what gaps exist is essential for designing such a system. The source code for TT-LLK lives at /localdev/salnahari/testing_dir/tt-llk and TT-Metal at /localdev/salnahari/testing_dir/tt-metal.
**Questions:**
- How does cuda-gdb work under the hood? What runtime hooks, driver interfaces, and hardware debug features does it rely on to attach to individual GPU threads?
- What does TT-Metal's current kernel dispatch path look like end-to-end — from host op invocation through kernel compilation, argument passing, memory setup, and launch on Tensix cores?
- What is the complete set of state that would need to be captured to fully reproduce an LLK kernel invocation (compiled binary, CB/L1 memory contents, runtime args, preprocessor defines, tile data, semaphore state, NOC configuration)?
- What existing debug infrastructure does TT-Metal provide today (watcher, DPRINT, fake_kernels_target, LLK_ASSERT, device logs) and what are their limitations?
- Do the Tensix RISC-V cores support hardware debug features (JTAG, breakpoints, watchpoints, single-step) that could enable GDB-style attach?
- Could a RISC-V emulator (e.g., Spike, QEMU) run captured LLK kernels with GDB attached, and what hardware-specific behavior would be lost in emulation?
- Where in the TT-Metal runtime would an interceptor hook need to be placed to capture all relevant state without missing implicit configuration?
- What serialization format and storage strategy would be appropriate for kernel capture snapshots (size estimates, versioning, reproducibility guarantees)?
- How would the replay debugger handle multi-core coordination (the 5 RISC-V cores per Tensix, inter-core semaphores, NOC transactions)?
- What are the performance and storage overhead implications of always-on capture vs. selective capture triggered by breakpoint or failure?
- Are there existing tools or prototypes in the TT-Metal or TT-LLK codebase that partially address kernel capture or replay?
- What lessons from other hardware debuggers (cuda-gdb, rocgdb, Intel GT debugger, Arm DS) apply to designing this for Tenstorrent?

---

## Comprehensive Understanding of TT-Blaze (Tenstorrent Blaze Kernel Composition Framework)
**Date:** 2026-05-07
**Status:** Completed
**Guide:** comprehensive_understanding_of_tt_blaze/
**Why Needed:** TT-Blaze is Tenstorrent's lightweight kernel composition layer built on top of TT-Metal, designed for low-latency inference on Blackhole silicon. It enables C++ kernel developers to write real TT-Metal kernel code and compose individual ops (MicroOps) into fused pipelines (FusedOps) in Python. The framework includes a full compilation pipeline (CB/Sem/CT-arg engines, JIT per-RISC compilation, kernel codegen), a multi-host pipeline orchestration system (pipeline_builder), a production-grade C++20 token scheduling library (pipeline_manager), a KV cache migration library for disaggregated inference, and an interactive dataflow visualizer. Understanding the entire system — from the op abstraction and graph IR through the compilation pipeline, the ~70+ op library (data movement, compute, attention, MoE), pipeline orchestration, token scheduling, and KV cache migration — is essential for developers working on or extending the Blaze inference stack. The source code lives at /localdev/salnahari/testing_dir/tt-blaze.
**Questions:**
- What is TT-Blaze's overall architecture and where does it sit in the Tenstorrent software stack relative to TT-Metal, TT-LLK, and TT-Forge?
- How does the two-API design work — the graph API (`blaze.fuse()` context manager with `BlazeGraph` DAG) versus the composition API (`emit()` inside FusedOps) — and how do they interact?
- What is the BlazeOp class hierarchy (BlazeOp → MicroOp/FusedOp), and how does the C++ kernel header parsing (`cpp_parser.py`) eliminate redundant Python declarations by treating `op.hpp` as the single source of truth?
- How does the compilation pipeline work end-to-end — from `BlazeGraph` through CBEngine, SemEngine, CTArgEngine, kernel codegen, `UnifiedKernelDescriptor`, to `ProgramDescriptor` and dispatch via `ttnn.generic_op()`?
- How does data flow between ops via the CBHandle chain (circular buffer IDs, page counts, data formats, core ranges), and how does the CBEngine auto-assign buffer IDs?
- How does the per-RISC compilation model work — JIT compiling for NCRISC, BRISC, and TRISC with `COMPILE_FOR_*` guards and baked-in `static constexpr` CT args?
- What is the op library structure and what categories of ops exist — data movement (mcast, gather, scatter, copy), compute (matmul variants, rmsnorm, rope, activations), attention (SDPA, Flash MLA, GQA, DSA), and MoE (gates, routed experts, shared experts)?
- How do FusedOps compose MicroOps — what is the FusedProgram context, how do CB aliases/scratch/output work, and what is the OverlappedView pattern for zero-copy L1 weight sharing?
- How does the kernel codegen (`kernel_codegen.py`) auto-generate C++ kernel source from a BlazeGraph using PhaseInfo metadata?
- What is the pipeline_builder system — how does PipelineGraph describe multi-host inference pipelines, how does SubmeshPartition carve MeshDevices into stage-sized submeshes, and how does topology-driven pipeline ordering work on Galaxy pods?
- What is the pipeline_manager C++20 token scheduling library — how does PipelineManager handle writer/reader threads, token injection, result collection, and what are the internal data structures (user_table, prefill_queue, decode_staging, spec_decode_state)?
- How does the KV cache migration library work for disaggregated inference — what are the migration_layer, device I/O, DCN transport, and MPI-based cross-host migration components?
- How does the interactive visualizer work — what does the JSON schema look like, and how does it render the dataflow DAG, CT/RT args, circular buffers, semaphores, spatial core grids, and L1 memory layouts?
- What is the weight provider abstraction (SyntheticWeightProvider, StateDictWeightProvider, BlitzCacheWeightProvider) and how does it support different weight sourcing strategies?
- How does the test infrastructure work — what are the test categories (infra, micro-ops, fused-ops, backed, generality, migration, pipeline_builder) and how do silicon tests run?
- What model-level integrations exist (DeepSeek V3, GLM-5.1) and how are they structured as multi-stage distributed pipelines?

---

