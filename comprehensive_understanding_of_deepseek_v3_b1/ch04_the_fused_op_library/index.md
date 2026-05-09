# Chapter 4 -- The Fused-Op Library

Chapter 3 cataloged the micro-ops -- single-program building blocks that each perform one well-defined operation. This chapter shows how DeepSeek V3 B1 composes those micro-ops into **fused operations**: multi-stage pipelines that execute as a single kernel dispatch, eliminating all inter-op launch overhead and enabling aggressive L1 buffer sharing across stages.

Every fused op in this chapter follows the same construction pattern: a Python `op()` or `get_program_context()` staticmethod allocates circular buffers, assigns per-RISC compile-time arguments for every stage, builds `UnifiedCompileTimeCoreDescriptor` role maps, and invokes `ttnn.generic_op`. The kernel source is a single `.cpp` file that conditionally executes stages based on core role flags (e.g., `is_matmul_core`, `is_mcast_sender_core`, `is_qrope_core`).

Source root: `models/demos/deepseek_v3_b1/fused_ops/`

---

## Contents

### [4.1 Fusion Principles and Patterns](./01_fusion_principles_and_patterns.md)

| Topic | Key Concepts |
|-------|-------------|
| Why fuse | Eliminating dispatch overhead; enabling L1 overlap; producer-consumer CB pipelining |
| CB reconfig | `CircularBufferIdManager`, `Context`, `build_cb_reconfig_tensor`, dummy CBs |
| Face-view optimization | `can_use_face_view`, $16 \times 16$ face tiles for gather-reduce |
| Compile-time roles | `UnifiedCompileTimeCoreDescriptor` and per-core differentiation |
| `OverlappedTensor` | Fused device buffer with sub-tensor CB descriptors (`byte_offset`, `total_size`) |
| Construction pattern | `_setup_dimensions` $\to$ `_build_compile_time_args` $\to$ `_build_cb_descriptors` $\to$ `_build_core_descriptors` $\to$ `UnifiedKernelDescriptor` $\to$ `ProgramDescriptor` $\to$ `generic_op` |

### [4.2 Attention Fused Ops](./02_attention_fused_ops.md)

| Section | Fused Op | Stages | Cores |
|---------|----------|--------|-------|
| 4.2.1 | `broadcast_rms` | CCL broadcast + RMSNorm | 1 core |
| 4.2.2 | `pre_sdpa` (deep dive) | CCL + RMS + Mcast + Matmul1 + GatherReduce + RMS2 + Mcast2 + Matmul2 + Matmul3/RoPE + CreateQHeads + KV branch + FlashMLA | 96 Q-path + 18 KV + 64 MLA |
| 4.2.3 | `kv_cache_branch` | DKV Matmul + Gather + RMSNorm + RoPE + KV cache update | 18 cores |
| 4.2.4 | `post_sdpa` | Matmul4 + Gather2 + Mcast3 + Matmul5 + Gather3 + CCL AllReduce (+ SDPA ReduceToAll) | 13$\times$10 grid; 14 CBs |

### [4.3 MoE Fused Ops](./03_moe_fused_ops.md)

| Section | Fused Op | Stages | Cores |
|---------|----------|--------|-------|
| 4.3.1 | `moe` | Full MoE orchestrator (`MoeContext`, 17 semaphores) | Full device grid |
| 4.3.2 | `moe_routed_expert` (deep dive) | Input Mcast + Gate MM + Gather + Gate + Index/Scale Mcast + gate/up/down proj + Mul + Add + ReduceToOne | 14 stages |
| 4.3.3 | `shared_expert` | Activation Mcast + Gate/Up Matmul + Gather + GatedReduce + Mcast + Down Proj + ResidualAdd + Gather | 128 compute cores |
| 4.3.4 | `gated_local_reduce` | SiLU(reduce(G1)) $\times$ reduce(G2) | 1 core; 4 CBs |
| 4.3.5 | `gated_local_reduce_down_proj` | Input Gather + GatedReduce + Mcast + Matmul + ResidualAdd + Gather | 130-core grid; 13 CBs |
| 4.3.6 | `down_proj` | Mcast + Matmul + ResidualAdd + Gather | 112 matmul cores; 8 CBs |

### [4.4 Output Fused Ops](./04_output_fused_ops.md)

| Section | Fused Op | Stages | Cores |
|---------|----------|--------|-------|
| 4.4.1 | `lm_head_sampling` | CCL Broadcast + RMSNorm + Mcast + Matmul + Fused Argmax + Mesh Reduce | ~101 cores |
| 4.4.2 | Argmax / sampling details | Multi-core argmax with mesh-wide allreduce and socket output | Per-device final core |

---

**Previous:** [Chapter 3 -- The Micro-Op Library](../ch03_the_micro_op_library/index.md)
**Next:** [Chapter 5 -- Multi-head Latent Attention Deep Dive](../ch05_multi_head_latent_attention_deep_dive/index.md)
