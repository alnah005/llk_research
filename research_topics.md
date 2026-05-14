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

## Comprehensive Understanding of DeepSeek V3 B1 (Batch-1 Decode Implementation on TT-Blaze)
**Date:** 2026-05-08
**Status:** Completed
**Guide:** comprehensive_understanding_of_deepseek_v3_b1/
**Why Needed:** DeepSeek V3 B1 is a production-grade, batch-1 decode implementation of the DeepSeek V3 671B-parameter Mixture-of-Experts model built on the TT-Blaze kernel composition framework for Tenstorrent hardware. It features a complete custom op library (~20 micro-ops and ~10 fused ops with hand-written C++ kernels), a unified kernel descriptor system, multi-host pipeline parallelism across Galaxy pods (single-pod, 2-pod, superpod configurations), Flash MLA attention, DeepSeek-specific MoE gating with routed and shared experts, DRAM-streaming matmul for large weight matrices, a Blitz weight caching system, and a demo runtime with CLI. Understanding this implementation end-to-end — from the model architecture mapping and weight preparation, through the op library design (micro-ops, fused ops, unified kernels), the multi-device scaleout strategy, and the inference runtime — is essential for developers working on large-model deployment on Tenstorrent systems and for understanding how TT-Blaze is used to build real production models. The source code lives at /localdev/salnahari/testing_dir/tt-metal/models/demos/deepseek_v3_b1.
**Questions:**
- What is the overall architecture of the DeepSeek V3 B1 implementation — how does model.py define the transformer layers, attention, MoE, and the full forward pass, and how does it map the 671B-parameter DeepSeek V3 architecture to TT-Blaze ops?
- How does the micro-op library work — what does each micro-op (matmul, rope, rmsnorm, sdpa, sdpa_tail, sdpa_reduce_to_all, flash_mla, gather, mcast, kv_cache_update, create_q_heads, deepseek_moe_gate, kn_sliced_matmul, dram_streaming_matmul, eltwise_add, local_reduce, reduce_to_one_b1, tilize_8x32, sampling/argmax, host_io, d2d_exchange, ccl_all_reduce, ccl_broadcast, pipeline_block) do, and how are they structured as TT-Blaze MicroOps with paired Python op.py and C++ kernel files?
- How does the fused-op library work — what does each fused op (moe, moe_routed_expert, shared_expert, pre_sdpa, post_sdpa, broadcast_rms, lm_head_sampling, kv_cache_branch, gated_local_reduce, gated_local_reduce_down_proj, down_proj) compose from micro-ops, and what fusion strategies do they implement for performance?
- How does the unified kernel system work — what is the relationship between unified_kernel_descriptor.py, the unified_kernels/*.hpp headers, and the per-op kernel .cpp files, and how does the kernel_op_api.hpp tie them together?
- How does the attention mechanism work — how do pre_sdpa, sdpa, flash_mla, sdpa_tail, sdpa_reduce_to_all, post_sdpa, and create_q_heads compose to implement DeepSeek V3's Multi-head Latent Attention (MLA) with compressed KV cache?
- How does the MoE (Mixture of Experts) subsystem work — how do deepseek_moe_gate, moe, moe_routed_expert, shared_expert, gated_local_reduce, and down_proj implement DeepSeek V3's 256-expert routing with top-K selection, expert parallelism, and shared expert fusion?
- How does multi-device scaleout work — what do the scaleout_configs (single_pod, 2_pod, superpod YAML and textproto files) define, how is the model partitioned across pipeline stages, and how do d2d_exchange, ccl_all_reduce, ccl_broadcast, and reduce_to_one_b1 handle inter-device communication?
- How does weight preparation work — what does prepare_weights.py do to transform HuggingFace DeepSeek V3 checkpoints into the device-ready format, and how does blitz_decode_weights.py implement the Blitz weight caching system for fast model loading?
- How does the circular buffer management work — what does circular_buffer_utils.py provide, and how are circular buffers allocated and shared across ops in fused programs?
- How does the demo/inference runtime work — what do demo/runner.py, demo/runtime.py, and demo/cli.py implement for end-to-end token generation, and how do they handle continuous decode with the pipeline?
- How does DRAM-streaming matmul work — what problem does dram_streaming_matmul solve (large weight matrices that don't fit in L1), and how does it stream weight tiles from DRAM during computation?
- What is the data flow through a single decode step — from host input through embedding, through N transformer layers (attention + MoE/MLP), through the LM head, to argmax/sampling output — and how does each stage map to specific ops?

---

## Using TT-Blaze on T3K and Multi-Device Configurations via TT-Symbiote
**Date:** 2026-05-08
**Status:** Completed
**Guide:** using_tt_blaze_on_t3k_and_multi_device_configurations_via_tt_symbiote/
**Why Needed:** TT-Symbiote is a PyTorch-to-TTNN acceleration framework that transparently replaces PyTorch modules with TTNN-optimized equivalents, and it already has multi-device support (DistributedConfig, ShardTensor2dMesh, CCL integration, device arch enums for T3K/TG/P150x4/P150x8/BHGLX). TT-Blaze is Tenstorrent's kernel composition framework with a sophisticated multi-device pipeline system (PipelineGraph, SubmeshPartition, topology-based auto-discovery, pipeline_manager token scheduling). Understanding how TT-Blaze's pipeline capabilities can be used on T3K and other multi-device configurations — and specifically how TT-Symbiote's transparent module replacement approach could leverage or integrate with TT-Blaze's pipeline infrastructure — is essential for enabling high-performance distributed inference of large models through the Symbiote pathway. The source code for TT-Symbiote lives at /localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote, TT-Blaze at /localdev/salnahari/testing_dir/tt-blaze, and TT-Metal at /localdev/salnahari/testing_dir/tt-metal.
**Questions:**
- How does TT-Symbiote's current multi-device support work — what do DistributedConfig, DistributedTensorConfig, ShardTensor2dMesh, and ConcatMesh2dToTensor do, and how does the DeviceInit/set_device flow handle mesh devices with get_num_devices() > 1?
- How does TT-Symbiote currently handle tensor parallelism — how are weights sharded across devices via mesh_mapper, how are activations distributed, and how does the CCL manager (TT_CCL) perform all-reduce/all-gather operations?
- What does TT-Blaze's PipelineGraph/PipelineLayout system provide for multi-device orchestration — how do Nodes, Edges, SubmeshPartition, and topology-based auto-discovery (build_topology) work together to create multi-stage pipelines across physical chips?
- How does TT-Blaze's pipeline_manager C++20 token scheduling library handle multi-device inference — what are the writer/reader thread model, token injection, result collection, and how does it manage continuous batching across pipeline stages?
- What are the concrete steps needed to run a model on T3K (8 Wormhole chips) using TT-Symbiote — device initialization, mesh shape configuration (1,8), fabric config, weight sharding strategy, and CCL operations?
- How does TT-Symbiote's module replacement approach interact with TT-Blaze's FusedOp/MicroOp composition — could Symbiote-replaced modules delegate to Blaze ops for better performance, and what are the integration boundaries?
- What are the different multi-device parallelism strategies available through each framework — data parallelism (batch sharding), tensor parallelism (weight/activation sharding), pipeline parallelism (stage-based), and how do they compare in terms of performance and complexity?
- How do existing multi-device model implementations (GLM-4.7, Qwen3-Coder-Next in Symbiote; DeepSeek V3, GLM-5.1 in Blaze) structure their distributed execution, and what patterns can be reused?
- What are the device-specific considerations for T3K vs TG (Galaxy) vs P150x4/P150x8/BHGLX — topology differences, bandwidth constraints, fabric configuration, and how does each framework adapt to these?
- What are the current limitations and gaps — which models are too large for single-device but not yet supported on multi-device, what operations don't parallelize well, and what infrastructure is missing for full pipeline parallelism through Symbiote?
- How does the MESH_DEVICE environment variable drive the entire multi-device configuration — how do conftest.py fixtures create mesh devices, how does run_on_devices decorator restrict execution, and how do tests parametrize across device topologies?
- What is the recommended path for taking a single-device TT-Symbiote model and scaling it to T3K — step-by-step guide covering model analysis, parallelism strategy selection, module adaptation, weight distribution, CCL operation insertion, and validation?

---

## Abstracting CB Management for PyTorch Developers Using TT-Blaze
**Date:** 2026-05-11
**Status:** Completed
**Guide:** abstracting_cb_management_for_pytorch_developers_using_tt_blaze/
**Why Needed:** PyTorch developers working with TT-Blaze currently need to understand circular buffer (CB) constraints, 32x32 tile padding, face-order memory layout, data format conversions, L1 memory budgeting, and hardware address transforms to use Blaze ops effectively. This creates a significant barrier to entry. An abstraction layer that maps standard PyTorch tensor semantics — arbitrary shapes, familiar data types, automatic broadcasting, shape inference — to TT-Blaze's tile/CB model would dramatically lower this barrier and enable PyTorch developers to compose Blaze ops without manual CB plumbing. This research should explore what such an abstraction layer looks like architecturally, how it handles shape propagation and padding through op chains, how it integrates with TT-Blaze's existing CBHandle/FusedProgram/CBEngine systems, and what the performance trade-offs are versus hand-tuned CB management. The source code for TT-Blaze lives at /localdev/salnahari/testing_dir/tt-blaze, TT-Metal at /localdev/salnahari/testing_dir/tt-metal, and TT-Symbiote at /localdev/salnahari/testing_dir/tt-metal/models/experimental/tt_symbiote.
**Questions:**
- How can a PyTorch tensor of arbitrary shape [A, B, C, D] be automatically decomposed into tiles and mapped to CBs without the developer specifying tile geometry, face layout, or padding?
- What abstraction layer design patterns (from frameworks like TVM, Triton, XLA, MLIR, or TT-Symbiote's module replacement) can inform a "TensorAdapter" that sits between PyTorch and TT-Blaze's CBHandle/FusedProgram API?
- How should shape metadata (logical shape vs padded shape vs tile decomposition) propagate through chains of ops so downstream ops automatically receive correct shape information?
- How can padding be managed transparently — including zero-padding for matmul, -inf padding for softmax, and mask-based padding for reductions — without requiring the developer to specify padding strategy per op?
- How should the abstraction handle data format selection and conversion (Float32, BFloat16, Bfp8_b, etc.) between ops that may prefer different formats, including automatic format negotiation between producers and consumers?
- How can the abstraction automatically determine optimal CB sizing (num_pages, page_size) and blocking strategies (RT_DIM, CT_DIM, KT_DIM for matmul) based on tensor shapes and available L1 memory?
- How should the abstraction handle broadcasting semantics — mapping PyTorch-style broadcasting to TT-Blaze's bcast_rows/bcast_cols/bcast_scalar patterns?
- What is the integration surface between this abstraction and TT-Blaze's existing CBHandle, FusedProgram, CBEngine, SemEngine, and CTArgEngine systems — should it wrap, extend, or replace them?
- How can the abstraction support op fusion (combining multiple PyTorch ops into a single FusedOp) while maintaining correct shape and padding tracking across the fused boundary?
- What are the performance implications of the abstraction layer — where does automation sacrifice performance compared to hand-tuned CB management, and how can escape hatches be provided for power users?
- How do existing higher-level frameworks (TT-Symbiote's transparent module replacement, TT-Forge's graph compiler, PyTorch's FX/torch.compile, JAX's XLA lowering) solve similar tensor-to-hardware mapping problems, and what lessons apply?
- What does the end-to-end developer API look like — from defining a PyTorch module to having it run on Tenstorrent hardware via TT-Blaze with zero manual CB, padding, or tile management?

---

## Comprehensive Understanding of pi0.5 and Implementation on Tenstorrent via TT-Blaze and blaze-nn
**Date:** 2026-05-13
**Status:** Completed
**Guide:** comprehensive_understanding_of_pi05_and_implementation_on_tenstorrent_via_tt_blaze_and_blaze_nn/
**Why Needed:** pi0.5 is Physical Intelligence's state-of-the-art vision-language-action (VLA) model for robotic control, built on a dual-expert PaliGemma/Gemma architecture with flow-matching action generation. Understanding this model end-to-end — from its SigLIP vision encoder and dual-expert Gemma transformer (PaliGemma 2B + Action Expert 300M with shared attention and adaRMSNorm), through its flow-matching inference loop with KV-cached prefix and iterative Euler denoising — and mapping it to Tenstorrent hardware via TT-Blaze's kernel composition framework and blaze-nn's PyTorch-compatible Module interface is essential for enabling robotics inference on Tenstorrent silicon. This requires understanding which existing TT-Blaze ops can be reused, what new ops must be written, how the dual-expert shared-attention pattern maps to Blaze's FusedOp model, how flow-matching's iterative denoising loop interacts with Blaze's pipeline system, and how blaze-nn's tracing API can express the full pi0.5 forward pass. The pi0.5 source code lives at https://github.com/Physical-Intelligence/openpi (cloned to /tmp/openpi), TT-Blaze at /localdev/salnahari/testing_dir/tt-blaze, and blaze-nn at /localdev/salnahari/testing_dir/blaze-nn.
**Questions:**
- What is the pi0.5 model architecture end-to-end — how do the SigLIP vision encoder (So400m/14, 224x224 input), PaliGemma backbone (Gemma 2B: width=2048, depth=18, 8 heads, 1 KV head, head_dim=256, mlp_dim=16384), and Action Expert (Gemma 300M: width=1024, depth=18, 8 heads, 1 KV head, head_dim=256, mlp_dim=4096) compose into a dual-expert transformer with shared self-attention?
- How does the dual-expert shared attention work — how are Q/K/V projections computed separately per expert but concatenated for joint attention, how does the attention mask enforce prefix-LM semantics (image+language tokens attend bidirectionally, action tokens attend causally), and how are output projections split back per expert?
- How does adaRMSNorm (adaptive RMS normalization) work in pi0.5 — how does the flow-matching timestep condition the Action Expert's layer norms via a learned MLP that produces scale, shift, and gate modulations, and how does the gated residual pattern (x + y * gate) differ from standard residual connections?
- How does the flow-matching inference loop work — what is the Euler denoising process (x_t = t*noise + (1-t)*actions, v_t = noise - actions, dt = -1/num_steps), how does KV caching separate the prefix (computed once) from the suffix (recomputed each step), and what are the 10-step denoising dynamics?
- What is the complete data flow through pi0.5 inference — from input images (3 cameras: base, left_wrist, right_wrist at 224x224) and tokenized language prompt through SigLIP embedding, PaliGemma language embedding, prefix KV caching, action token embedding with sinusoidal timestep encoding, time MLP (SiLU activations), iterative denoising with suffix-only forward passes, to final action_out_proj producing action sequences (batch x 50 x 32)?
- Which existing TT-Blaze ops (from the ~100+ op library) can be directly reused for pi0.5 — matmul for linear projections, rmsnorm for layer normalization, rope for rotary position embeddings, sdpa/flash_mla for attention, embedding for token embedding, residual_add for skip connections, swiglu/clamped_silu for activations — and what is the gap analysis?
- What new TT-Blaze ops or FusedOps must be written for pi0.5 — adaRMSNorm (timestep-conditioned normalization with scale/shift/gate), SigLIP vision encoder (ViT with patch embedding, sincos 2D positional embedding, GELU MLP blocks), gated residual connections (x + y * gate), sinusoidal timestep embedding, and dual-expert attention with split Q/K/V and merged attention?
- How does the dual-expert shared-attention pattern map to TT-Blaze's FusedOp composition model — can two separate FusedOps (PaliGemma layer + Action Expert layer) share a single SDPA computation, or must a new DualExpertBlock FusedOp be written that handles Q/K/V concatenation, joint attention, and output splitting within a single fused program?
- How does pi0.5's flow-matching iterative denoising loop interact with TT-Blaze's pipeline system — can the prefix computation be a separate pipeline stage from the denoising loop, how does KV cache persistence across denoising steps map to Blaze's CB/L1 memory model, and how does the variable-iteration denoising loop (10 steps) fit within Blaze's pipeline_manager?
- How can blaze-nn's PyTorch-compatible Module/functional API express the full pi0.5 forward pass — what does the module hierarchy look like (Pi05Module containing SigLIPEncoder, GemmaEmbedder, DualExpertTransformer, TimeMLP, ActionProjection), how do blaze_nn.functional ops map to the required computations, and what new functional ops must be added?
- What are the weight loading and format conversion requirements — how do the pretrained pi0.5 weights (from HuggingFace/safetensors format with PaliGemma + Action Expert parameters) map to TT-Blaze's weight provider abstraction (SyntheticWeightProvider, StateDictWeightProvider, BlitzCacheWeightProvider), and what transformations are needed?
- What are the performance and memory considerations for pi0.5 on Tenstorrent — how does the model's ~2.3B total parameters (2B PaliGemma + 300M Action Expert) fit in device memory, what is the L1/DRAM strategy for weights and KV cache, how does the 10-step iterative inference affect latency compared to single-pass models, and what is the target latency for real-time robotic control (typically <100ms per action)?
- How do existing TT-Blaze model implementations (DeepSeek V3 B1, GLM-5.1) inform the pi0.5 implementation — what architectural patterns (unified kernel descriptors, fused ops, pipeline stages, weight preparation, KV cache management) transfer directly, and what is fundamentally different about pi0.5's dual-expert + flow-matching architecture?

---

## Comprehensive Guide to Creating New Micro-Ops in TT-Blaze
**Date:** 2026-05-13
**Status:** Completed
**Guide:** comprehensive_guide_to_creating_new_micro_ops_in_tt_blaze/
**Why Needed:** Developers working with TT-Blaze need a complete, practical guide to creating new micro-ops from scratch. This includes the full lifecycle: designing the Python class and C++ kernel, understanding how compile-time args flow from Python to the generated header, allocating circular buffers and semaphores, handling device-specific grid configurations (harvested Blackhole P150 with 11x10 grids, unharvested P300 with 13x10 grids, multi-device Galaxy/T3K mesh configurations), composing micro-ops into fused ops, writing tests, debugging hangs and correctness issues, and avoiding common pitfalls. The source code lives at /localdev/salnahari/testing_dir/tt-blaze.
**Questions:**
- What is the complete anatomy of a TT-Blaze micro-op — what are the required files (op.py, kernels/op.hpp, __init__.py), how does auto-discovery and registration work (register_all() in ops/__init__.py → BlazeOp.register() → OpSpec + CTArgSchema + PhaseInfo), and what is the relationship between the Python class (MicroOp subclass) and the C++ Op struct?
- How do you design the C++ kernel header (op.hpp) — what are the CT arg struct conventions (CoreCTArgs for all RISCs, ReaderCTArgs for NCRISC/DM0, WriterCTArgs for BRISC/DM1, ComputeCTArgs for TRISC), what do the typed field annotations mean (CB, Semaphore, PerCore, Flag vs plain uint32_t/bool), and how does the cpp_parser.py auto-extract metadata from the header?
- How do the three Tensix processors (NCRISC, BRISC, TRISC) divide work within a micro-op — when does NCRISC handle init() for sharded buffer setup vs data fetching, when does BRISC handle NOC writes/multicasts/sends, how does TRISC handle unpack→math→pack, and how do #if defined(COMPILE_FOR_*) guards and if constexpr(is_active) compile out inactive code paths?
- How does emit() bridge Python to C++ — what is the three-step structure (allocate resources → declare CT args → return CBHandle), what are the CB allocation methods (cb_from_tensor, cb_scratch, cb_alias, cb_output), how do prefixes become ct_args:: namespaces, and how does CBHandle chain data flow between ops?
- How do you handle device-specific grid configurations — how does GridConfig.from_device() detect harvested vs unharvested Blackhole devices (11x10 vs 12x10 vs 13x10), how do DRAM worker positions vary, how does build_matmul_core_grid() exclude DRAM workers and phantom columns, and what is the sender_core convention?
- How do you target multi-device configurations (P150 single-chip, P300 single-chip, Galaxy/T3K multi-chip mesh) — how does MeshDevice and submesh partitioning work, how do CCL ops (CclBroadcast, AllReduce) handle inter-device communication via fabric NOC, and what changes in emit() for mesh-aware compilation (DeviceContext per device)?
- How do you write and run tests for micro-ops — what are the two testing patterns (FusedProgram.emit() composition vs blaze.fuse()+BlazeCompiler graph compilation), how do pytest fixtures (device, mesh_device) work via tt-metal's conftest, what does @pytest.mark.requires_grid_size do, and how do you compute golden references and verify with comp_pcc?
- How does the kernel codegen pipeline work — how does kernel_codegen.py generate the unified .cpp from a BlazeGraph, how does PhaseInfo map Python ops to C++ Op struct instantiations, how does named_args_generated.h get produced with ct_args:: and rt_args:: namespaces, and when should you handwrite a kernel .cpp vs use auto-codegen?
- What are the common causes of hangs in micro-ops and how do you debug them — what happens when CB waits deadlock (cb_wait_front without matching cb_push_back, mismatched num_pages), how do semaphore protocol mismatches cause hangs, what does TT_METAL_DPRINT_CORES show, and how does BLAZE_DEBUG_KERNELS=1 inject phase markers?
- What are the debugging tools and techniques — how to use BLAZE_DEBUG_KERNELS (with RISC filtering: brisc, ncrisc, trisc, trisc0/1/2), BLAZE_L1_PROFILE for CB address/type/dtype reporting, inspecting named_args_generated.h in ~/.cache/tt-metal-cache, reading program.generated_kernel, and using the visualizer/viz_export for graph inspection?
- What are the common gotchas and pitfalls when writing micro-ops — CT arg name mismatches between Python emit() and C++ header (compile error), forgetting is_active flag (all cores execute), wrong RISC struct placement (ReaderCTArgs vs ComputeCTArgs), CB page count mismatches (weight pages not multiple of K tiles), forgetting cb_push_back after writing tiles, forgetting cb_pop_front causing CB overflow, and incorrect tile geometry (row tiles vs square tiles)?
- How do FusedOps compose multiple micro-ops — what is the FusedOp base class contract (compose() + optional emit()), how do prefix namespaces prevent CT arg collisions, how do shared resources (semaphores, CBs) flow between phases, and how does the phase ordering in Python emit() calls map to kernel_main() execution order?
- What are the performance considerations and L1 memory management strategies — how does cb_scratch vs cb_from_tensor affect L1 usage, how does CB reuse (disjoint-core sharing, format-key sharing) reduce L1 pressure, how does direct-address mode (OverlappedView, cb_from_shared_l1_views) enable weight packing, and what are the L1 budget constraints on different devices?
- How do inter-core ops (mcast, gather, scatter, CCL) differ from compute-only ops — what is the sender/receiver role pattern, how do NOC coordinates translate from logical to physical (worker_core_from_logical_core), what semaphore protocols exist (sender_semaphore, receiver_semaphore, noc0/noc1), and how do you avoid NOC congestion?
- What is the complete end-to-end workflow for adding a production micro-op — from initial design (choosing which processors, CT args, CB layout), through implementation (Python class, C++ kernel, emit()), testing (standalone micro-op test, integration in a FusedOp), debugging (DPRINT, L1 profiling, watcher), to deployment (compose into a model pipeline, handle harvested devices, run on Galaxy)?

---
