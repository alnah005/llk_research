# Comprehensive Understanding of DeepSeek V3 B1

A deep technical guide to the batch-1 decode implementation of DeepSeek V3 (671B parameters, 256-expert MoE) on Tenstorrent hardware using the TT-Blaze kernel composition framework.

**Source code:** `tt-metal/models/demos/deepseek_v3_b1/`

---

## Chapters

### [Chapter 1: Architecture Overview and Model Mapping](./ch01_architecture_overview_and_model_mapping/index.md)

How the 671B-parameter DeepSeek V3 architecture maps to TT-Blaze ops and Tenstorrent hardware. The three-tier op hierarchy, the `DeepSeekV3` class interface, the TT-Blaze execution pattern (`Matmul.op()` walkthrough), layer taxonomy (3 dense + 58 MoE), and a high-level decode step overview.

**3 files, ~1,269 lines**

### [Chapter 2: The Unified Kernel System](./ch02_the_unified_kernel_system/index.md)

The infrastructure that lets a single C++ source file compile for all three RISC processors on each Tensix core. `UnifiedKernelDescriptor`, `kernel_op_api.hpp`, the simple/split compilation paths, per-core runtime args, and the circular buffer management system (`CircularBufferIdManager`, CB reconfig tensors).

**3 files, ~1,774 lines**

### [Chapter 3: The Micro-Op Library](./ch03_the_micro_op_library/index.md)

A detailed reference for every micro-op organized by category: compute (matmul, rmsnorm, rope, local_reduce, eltwise_add, argmax), data movement (gather, mcast, create_q_heads, kv_cache_update, dram_streaming_matmul), and communication/infrastructure (ccl_all_reduce, ccl_broadcast, reduce_to_one_b1, sdpa_reduce_to_all, d2d_exchange, host_io, pipeline_block, deepseek_moe_gate).

**3 files, ~2,181 lines**

### [Chapter 4: The Fused-Op Library](./ch04_the_fused_op_library/index.md)

How micro-ops compose into fused operations that execute as single kernel dispatches. Fusion principles (CB reconfig, face-view, OverlappedTensor), attention fused ops (broadcast_rms, pre_sdpa, kv_cache_branch, post_sdpa), MoE fused ops (moe, moe_routed_expert, shared_expert, gated_local_reduce, down_proj), and output fused ops (lm_head_sampling).

**4 files, ~2,024 lines**

### [Chapter 5: Multi-head Latent Attention Deep Dive](./ch05_multi_head_latent_attention_deep_dive/index.md)

End-to-end MLA trace: query projection and weight layout (q_a/q_b with column shuffling), the compressed KV cache (576-element latent vector, BFP8 DRAM layout), Flash MLA decode (chunked online softmax, S-block parallelism, tree reduction), and post-SDPA output projection (kv_b2 + o_proj + CCL all-reduce).

**4 files, ~2,321 lines**

### [Chapter 6: Mixture-of-Experts Deep Dive](./ch06_mixture_of_experts_deep_dive/index.md)

End-to-end MoE trace: gating and routing (hierarchical top-8 selection with bias correction), routed expert computation (DRAM streaming gate/up/down projections, ReduceToOne across 8 devices), shared expert and combination (KN-sliced matmul, gated local reduce, 17-semaphore coordination), and a DRAM streaming matmul deep dive (column-major reordering, triple-buffering, indexed expert access).

**4 files, ~2,192 lines**

### [Chapter 7: Weight Preparation and Blitz Caching](./ch07_weight_preparation_and_blitz_caching/index.md)

The offline pipeline from HuggingFace checkpoint to device-ready tensors. Weight preparation (key mapping, transpose, kv_b split, TP slicing, fusion groups), Blitz decode weight overlapping (OverlappedTensor, 4 overlap specs, BFP8 encoding), and cache serialization (generate_cache.py, .tensorbin format, special tensors).

**3 files, ~1,442 lines**

### [Chapter 8: Multi-Device Scaleout, Demo Runtime, and Full Decode Trace](./ch08_multi_device_scaleout_demo_runtime_and_full_decode_trace/index.md)

The capstone chapter. Pipeline parallelism and configuration (63 stages, snake-order traversal, Galaxy pod topology), multi-device communication (D2D sockets, TP=2/EP=8, CCL collectives), the demo inference runtime (CLI, weight loading, generation loop, termination protocol), and a fully annotated decode step data flow with tensor shapes at every boundary, pipeline bottleneck analysis, and a hardware-software co-design summary.

**4 files, ~2,706 lines**

---

## Reading Orders

### Linear (recommended for first read)

Chapter 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

Builds understanding bottom-up: architecture overview, then kernel infrastructure, then op catalogs, then domain deep dives, then weight preparation, then multi-device integration.

### Attention-first

Chapter 1 → 2 → 3 (compute + data movement only) → 5 → 4.2 (attention fused ops)

For readers focused on the MLA attention mechanism.

### MoE-first

Chapter 1 → 2 → 3 → 6 → 4.3 (MoE fused ops) → 7

For readers focused on the Mixture-of-Experts subsystem and weight handling.

### Deployment-focused

Chapter 1 → 7 → 8 → 2

For readers deploying or configuring DeepSeek V3 on Galaxy pods.

---

## Guide Statistics

| Chapter | Files | Lines |
|---------|-------|-------|
| Ch1: Architecture Overview | 3 | 1,269 |
| Ch2: Unified Kernel System | 3 | 1,774 |
| Ch3: Micro-Op Library | 3 | 2,181 |
| Ch4: Fused-Op Library | 4 | 2,024 |
| Ch5: MLA Deep Dive | 4 | 2,321 |
| Ch6: MoE Deep Dive | 4 | 2,192 |
| Ch7: Weight Preparation | 3 | 1,442 |
| Ch8: Multi-Device Scaleout | 4 | 2,706 |
| **Total** | **28** | **~15,909** |
