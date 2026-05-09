# Chapter 6 -- Mixture-of-Experts Deep Dive

Chapter 5 traced Multi-head Latent Attention end-to-end. This chapter does the same for the Mixture-of-Experts subsystem: the second major compute block in each MoE transformer layer (layers 3-60). DeepSeek V3 uses 256 routed experts with top-8 selection, a bias-corrected hierarchical gating mechanism, and a shared expert that runs concurrently on a separate set of cores. All expert projections stream weights from DRAM because a single expert's gate/up/down matrices exceed L1 capacity.

The narrative follows a single token through four phases:

1. **Gating and routing** -- how the hidden state becomes 8 expert indices and normalized scores through sigmoid activation, bias correction, group ranking, and top-8 selection on a single 16x16 tile.
2. **Routed expert computation** -- how 8 selected experts execute gate/up/down projections via DRAM streaming matmul with indexed expert access, then reduce across all 8 devices via a 3-level tree.
3. **Shared expert and combination** -- how the shared expert runs concurrently on 128 cores using KN-sliced matmul, gated local reduce, and down projection, with 17 semaphores coordinating the two pipelines.
4. **DRAM streaming matmul deep dive** -- the micro-op that makes MoE possible: column-major tile reordering, triple-buffered DRAM reads, page size optimization, and indexed expert access.

Source root: `models/demos/deepseek_v3_b1/`

---

## Contents

### [6.1 MoE Gating and Routing](./01_moe_gating_and_routing.md)

| Topic | Key Concepts |
|-------|-------------|
| Gate matmul | $[1, 7168] \times [7168, 256]$ on 8 cores; sigmoid activation |
| Hierarchical selection | Sort per 16-expert group, rank groups by top-2 sum, flatten top-4 groups, final top-8 |
| Bias correction | `e_score_correction_bias` added post-sigmoid before selection |
| Score normalization | Configurable 2.5x scaling factor applied to final top-8 scores |
| Dense vs MoE | Layers 0-2 dense (`FIRST_K_DENSE_REPLACE=3`), layers 3-60 MoE |

### [6.2 Routed Expert Computation](./02_routed_expert_computation.md)

| Topic | Key Concepts |
|-------|-------------|
| Expert pipeline | Input mcast -> gate_proj (SiLU) -> up_proj -> fused mul -> down_proj -> eltwise add |
| Expert parallelism | Each DRAM core processes one expert independently via indexed DRAM streaming |
| Weight layout | 256 experts × {gate, up, down}_proj as individual DRAM-sharded tensors |
| ReduceToOne | 3-level tree: LEAF → ROOT3 → ROOT2 → ROOT1 across 4x2 mesh |
| `_MoeRoutedExpertContext` | 50+ field dataclass: CB indices, semaphore addresses, core grids |

### [6.3 Shared Expert and Combination](./03_shared_expert_and_combination.md)

| Topic | Key Concepts |
|-------|-------------|
| Shared expert fusion | Activation mcast + KN-sliced matmul (64 gate + 64 up cores) + gather + gated reduce + down proj |
| `gated_local_reduce` | SiLU(reduce(gate)) × reduce(up) three-phase pattern; face-view optimization |
| Combination | Element-wise addition of routed and shared expert down_proj outputs |
| Semaphore coordination | 17 `MoeSem` constants synchronizing mcast, gather, reduce, fabric operations |
| CB reconfig | Runtime reconfiguration between routed and shared expert phases |

### [6.4 DRAM Streaming Matmul Deep Dive](./04_dram_streaming_matmul_deep_dive.md)

| Topic | Key Concepts |
|-------|-------------|
| Problem | Expert weight matrices (7168×2048) exceed L1; must stream from DRAM |
| Tile reordering | Column-major reorder for K-tile contiguity in physical memory |
| Triple-buffering | 3 CB slots for DRAM read pipelining with NOC page size optimization |
| `subblock_k` | K subblock size for partial accumulation when full K exceeds triple-buffer |
| Optional features | Fused SiLU, indexed expert access, element-wise mul with gate scores |
| Bank/VC assignment | Per-core compile-time args for DRAM bank and virtual channel conflict avoidance |

---

**Previous:** [Chapter 5 -- Multi-head Latent Attention Deep Dive](../ch05_multi_head_latent_attention_deep_dive/index.md)
**Next:** [Chapter 7 -- Weight Preparation and Blitz Caching](../ch07_weight_preparation_and_blitz_caching/index.md)
