# Chapter 3 -- The Micro-Op Library

This chapter catalogs every micro-op used in the DeepSeek V3 B1 inference implementation on Tenstorrent hardware. Each micro-op is a single-program building block that performs one well-defined function -- a matrix multiplication, a normalization, a data gather, a cross-device reduction. Fused ops (Chapter 4) compose these micro-ops into multi-stage pipelines by including their kernel headers into unified kernel source files.

All micro-ops follow the unified kernel pattern described in Chapter 2: a Python `op()` staticmethod builds circular buffers and a `UnifiedKernelDescriptor`, then calls `ttnn.generic_op`. The differentiator for this chapter is the internal structure of each op -- which CB indices carry which tensors, what each RISC processor does, and how the op maps mathematical intent onto hardware resources.

Source root: `models/demos/deepseek_v3_b1/micro_ops/`

---

## Contents

### [3.1 Compute Micro-Ops](01_compute_micro_ops.md)

| Section | Op | Purpose |
|---------|-----|---------|
| 3.1.1 | `matmul` | Single-tile-height matrix multiply with optional fused activation |
| 3.1.2 | `rmsnorm` | Root-mean-square normalization with tile-size autodetection |
| 3.1.3 | `rope` | Rotary position embedding (Meta-style rotate_half via matmul) |
| 3.1.4 | `kn_sliced_matmul` | K-sliced parallel matmul with per-core K-offset |
| 3.1.5 | `local_reduce` | Pairwise tile reduction with optional SiLU |
| 3.1.6 | `eltwise_add` | Indexed element-wise add via CB read pointer manipulation |
| 3.1.7 | `sampling` | Multi-core argmax with mesh-wide reduction |
| 3.1.8 | `tilize_8x32` | Row-major to tiled format conversion |

### [3.2 Data Movement Micro-Ops](02_data_movement_micro_ops.md)

| Section | Op | Purpose |
|---------|-----|---------|
| 3.2.1 | `gather` (+ `gather_reduce` variant) | Many-to-one data collection with dual-NOC routing |
| 3.2.2 | `mcast` | One-to-many hardware multicast |
| 3.2.3 | `create_q_heads` | Three-phase Q head assembly from QNOPE + QROPE |
| 3.2.4 | `kv_cache_update` | DRAM KV cache read-modify-write with BFP8 conversion |
| 3.2.5 | `dram_streaming_matmul` | DRAM-streamed matmul with triple buffering (deep dive) |

### [3.3 Communication and Infrastructure Micro-Ops](03_communication_and_infrastructure_micro_ops.md)

| Section | Op | Purpose |
|---------|-----|---------|
| 3.3.1 | `ccl_all_reduce` | 2-device neighbor-exchange all-reduce |
| 3.3.2 | `ccl_broadcast` | Single-sender broadcast with dual-axis support |
| 3.3.3 | `reduce_to_one_b1` | 3-level tree reduction across 4x2 mesh |
| 3.3.4 | `sdpa_reduce_to_all` | Cross-device SDPA partial result reduction |
| 3.3.5 | `d2d_exchange` | Bidirectional D2D socket relay |
| 3.3.6 | `host_io` | PCIe H2D/D2H with optional fused embedding |
| 3.3.7 | `pipeline_block` | Multi-host pipeline stage orchestration |
| 3.3.8 | `deepseek_moe_gate` | Hierarchical top-8 expert selection (deep dive) |

---

| Navigation | Link |
|------------|------|
| Previous | [Chapter 2 -- The Unified Kernel Framework](../ch2/index.md) |
| Next | [Chapter 4 -- Fused Ops](../ch4/index.md) |
