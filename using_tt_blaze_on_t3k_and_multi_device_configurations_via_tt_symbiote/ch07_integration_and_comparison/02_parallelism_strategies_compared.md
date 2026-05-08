# 02 -- Parallelism Strategies Compared

This file compares every parallelism strategy available in Symbiote and Blaze -- including strategies that are absent -- and grounds each comparison in per-model CCL cost evidence from the five documented implementations. The dispatch model, memory model, and multi-device model from [File 01](./01_framework_comparison.md) determine which strategies each framework can express, and the hardware constraints from the same file set the efficiency bounds.

**Prerequisites:** [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md) (weight sharding), [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md) (PipelineGraph), [Ch5/05](../ch05_blaze_pipeline_system/05_hardware_configurations.md) (hardware configs), [Ch6/01-02](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) (model implementations).

---

## Strategy Inventory

### Strategies That Exist Today

| Strategy | Framework | Implementation | Hardware |
|----------|-----------|---------------|----------|
| Tensor Parallelism (TP) | Symbiote | Linear module hierarchy (`TTNNLinearIColShardedWRowSharded`, etc.) [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md) | T3K (1,8), TG (8,4) |
| Tensor Parallelism (TP) | Blaze | Within-submesh CCL: `AllReduce`, `ReduceToOne` [Ch4/02](../ch04_blaze_kernel_composition/02_ccl_in_blaze.md) | Per-stage submesh (4x2, 2x2, 1x2) |
| Pipeline Parallelism (PP) | Blaze | `PipelineGraph` + D2D sockets + `StageKind` [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md) | 4-64 stages across 1-4 galaxies |
| Expert Parallelism (EP) | Symbiote | `TTNNQwen3MoE` + `all_to_all_dispatch/combine` [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md) | T3K: 256 experts / 8 devices |
| Expert Parallelism (EP) | Blaze | Experts local to submesh; `DramStreamingMatmul` [Ch4/03](../ch04_blaze_kernel_composition/03_ops_catalog_overview.md) | Per-stage submesh |
| Hybrid PP+TP | Blaze | PP across submeshes + TP within [Ch5/05](../ch05_blaze_pipeline_system/05_hardware_configurations.md) | All multi-galaxy configs |

### Strategies That Do Not Exist

| Missing Strategy | Why It Matters | Why It Is Missing |
|-----------------|---------------|-------------------|
| **Data Parallelism (DP)** | Throughput scaling via batch replication | Symbiote's `ShardTensor2dMesh` shards batch dim but is not independent-replica DP. Blaze's pipeline is hardcoded to `batch_size=1` [Ch6/02, Section 6.13](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md). |
| **PP in Symbiote** | Scale beyond single-mesh DRAM | `set_device()` binds all modules to one mesh. No submesh partitioning exists [Ch2/01](../ch02_symbiote_core/01_module_replacement_engine.md). |
| **Sequence Parallelism (SP)** | Long-context models exceeding single-device KV cache | Neither framework shards along the sequence dimension. Symbiote defaults to `(batch=0, hidden=-1)` [Ch3/01](../ch03_symbiote_multi_device/01_distributed_config_and_device_init.md). |

---

## Tensor Parallelism: Symbiote vs Blaze

Symbiote implements TP through a hierarchy of distributed linear module classes [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md):

| Class | Weight Shard | Input State | Output State | CCL |
|-------|-------------|-------------|--------------|-----|
| `TTNNLinearIReplicatedWColSharded` | dim=-1 | Replicated | Col-sharded | None |
| `TTNNLinearIColShardedWRowSharded` | dim=-2 | Col-sharded | Col-sharded | `reduce_scatter` |
| `TTNNLinearIColShardedWAllReduced` | dim=-2 | Col-sharded | Replicated | `reduce_scatter` + `all_gather` |

Each CCL operation is a **separate dispatch**. Blaze fuses CCL into the same kernel as the surrounding compute -- `AllReduce.emit()` adds its CBs and CT args to the same `FusedProgram`, so the entire computation executes as one dispatch with zero inter-op overhead.

```
Symbiote per-layer dispatch:              Blaze per-layer dispatch:
  dispatch(ttnn.linear)                     dispatch(fused_decoder_block)
  dispatch(ttnn.reduce_scatter)               [internally contains:
  dispatch(ttnn.linear)                         matmul, mcast, all_reduce,
  dispatch(ttnn.reduce_scatter)                 rmsnorm, sdpa, residual_add,
  dispatch(ttnn.linear)                         moe_gate, expert_compute,
  dispatch(ttnn.all_gather)                     reduce_to_one, ...]
  ...  (~9-20 dispatches)                   (1 dispatch)
```

---

## CCL Cost Breakdown Per Model

### GLM-4.7 (Symbiote, No TP)

Zero CCL. Every device runs independently with a full copy of each weight (`TTNNLinearLLama` uses `bfloat8_b` but no sharding). Attention runs on CPU [Ch6/01, Section 6.2](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md).

### Gemma4 (Symbiote, TP-8 on T3K)

Per decoder layer [Ch6/01, Section 6.3](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md):

| Module | CCL Operations | Topology |
|--------|---------------|----------|
| Pre-attention RMSNorm | 1x `all_gather` (stats) | Ring |
| Fused QKV projection | 1x `reduce_scatter` + 1x `all_gather` | Ring |
| Post-attention RMSNorm | 1x `all_gather` (stats) | Ring |
| Pre-feedforward RMSNorm | 1x `all_gather` (stats) | Ring |
| Fused gate-up projection | 1x `reduce_scatter` + 1x `all_gather` | Ring |
| Post-feedforward RMSNorm | 1x `all_gather` (stats) | Ring |
| **Total per block** | **~9 CCL operations** | |

With 60 layers: **~540 CCL operations per forward pass.** Fused QKV saves 4 CCL ops per layer compared to separate Q/K/V projections [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md).

### Ling MoE (Symbiote, TP+EP on T3K)

Per MoE decoder layer [Ch6/01, Section 6.4](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md):

| Module | CCL Operations | Topology |
|--------|---------------|----------|
| Input layernorm | 1x `all_gather` | Ring |
| Q/K/V projections | 3x `reduce_scatter` | Ring |
| O projection | 1x `reduce_scatter` | Ring |
| Post-attn layernorm | 1x `all_gather` | Ring |
| MoE input revert TP | 1x `all_gather` | Linear |
| Expert routing | 1x `all_to_all_dispatch` | Custom |
| Expert combine | 1x `all_to_all_combine` | Custom |
| MoE output | 1x `reduce_scatter` | Ring |
| Shared experts (gate+up+down) | 3x `reduce_scatter` | Ring |
| **Total per block** | **~13 CCL operations** | |

The `all_to_all_dispatch/combine` pair is asymmetric and data-dependent, making it the costliest CCL pattern.

### Qwen3.5 (Symbiote, TP+EP on T3K)

Similar to Ling: 256 experts, top-8 routing, fused W1/W3 `sparse_matmul`. DeltaNet layers (no KV cache, no SDPA) alternate with full GQA layers. MoE follows the same `all_to_all` pattern. Total: **~10-13 CCL operations per MoE layer** [Ch6/01, Section 6.5](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md).

### DeepSeek V3 (Blaze, PP+TP on 4-Galaxy)

Per decoder layer (intra-submesh only) [Ch6/02, Section 6.8](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md):

| Module | CCL Operations | Type |
|--------|---------------|------|
| Attention AllReduce | 1x `AllReduce` (ring sum, `cluster_axis=0`) | Fabric |
| Output ReduceToOne | 1x `ReduceToOne` (tree reduction, `cluster_axis=1`) | Fabric |
| **Total per block** | **2 intra-submesh CCL operations** | |

Cross-submesh communication: **0 CCL operations.** Data moves via D2D socket (14KB page, single ethernet hop per stage boundary). Blaze's PP strategy eliminates most cross-device CCL by confining each layer within a submesh.

---

## Activation Flow Pattern Taxonomy

The five models exhibit three distinct activation sharding patterns:

### Pattern 1: Replicated (GLM-4.7)

All activations replicated on every device. No CCL, no TP benefit. Each linear executes on full hidden dimension.

### Pattern 2: Column-Sharded Chain (Ling, Qwen3.5)

All activations stay column-sharded throughout. Every linear uses row-parallel (`reduce_scatter` output stays sharded) [Ch3/02, "Ling/Bailing Activation Pattern"](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md):

```
  hidden_states: COL-SHARDED [B, S, H/8]
       |
  DistributedRMSNorm --> COL-SHARDED (all_gather stats only)
       |
  Row-parallel Q/K/V --> COL-SHARDED (reduce_scatter)
       |
  Attention --> COL-SHARDED
       |
  Row-parallel O --> COL-SHARDED (reduce_scatter)
       |
  Residual add --> COL-SHARDED
```

### Pattern 3: Alternating Sharded/Replicated (Gemma4)

Activations toggle between column-sharded and replicated at each projection boundary [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md):

```
  hidden_states: COL-SHARDED [B, S, H/8]
       |
  Fused QKV (AllReduced) --> REPLICATED [B, S, QKV_dim]
       |
  Attention (SDPA on replicated data) --> REPLICATED
       |
  O-proj (ColSharded) --> COL-SHARDED [B, S, H/8]
       |
  Fused gate-up (AllReduced) --> REPLICATED [B, S, 2*I]
       |
  Down (ColSharded) --> COL-SHARDED [B, S, H/8]
```

Pattern 3 costs more CCL than Pattern 2 (extra `all_gather` after every fused projection) but enables simpler attention because Q/K/V are fully replicated.

---

## Pipeline Parallelism (Blaze Only)

PP is Blaze's primary multi-device strategy. The three-layer architecture [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md): (1) Logical graph (`PipelineGraph` with `Node`/`Edge`), (2) Physical partition (`SubmeshPartition` via `build_topology()`), (3) Runtime execution (`StageKind` subclasses creating `PipelineBlock` with D2D sockets).

DeepSeek V3's 64-stage SP4 configuration distributes 61 transformer layers plus bookend stages across 128 chips. The loopback edge from the last stage to the first enables continuous token-by-token decode [Ch5/01](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md).

**Pipeline bubble analysis:** For a 64-stage pipeline (single-token decode), the first 63 tokens must fill the pipeline before steady-state throughput is reached. At 128-token generation, fill overhead is ~49%; at 1024 tokens, it drops to ~6%. Blaze mitigates this with persistent-mode kernels and continuous batching via `PipelineManager` [Ch5/04](../ch05_blaze_pipeline_system/04_pipeline_manager_cpp.md).

> **Warning:** Blaze's prefill path processes tokens one at a time. For a 4096-token prompt on 64 stages, this means 4096 full pipeline traversals -- extremely slow compared to Symbiote's batched `model.generate()` which processes the full prompt in one forward pass [Ch6/02, Section 6.13](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md).

---

## Hybrid PP+TP Configuration Matrix (Blaze)

| Config | Stage Shape | TP Width | PP Depth | Total Chips |
|--------|------------|----------|----------|-------------|
| 1G-4x2 | 4x2 | 8 | 4 | 32 |
| 1G-2x2 | 2x2 | 4 | 8 | 32 |
| 1G-1x2 | 1x2 | 2 | 16 | 32 |
| 4G-4x2 | 4x2 | 8 | 16 | 128 |
| 4G-1x2 | 1x2 | 2 | 64 | 128 |

The 4G-1x2 configuration (64 stages, TP=2) is the production target for DeepSeek V3 [Ch5/05](../ch05_blaze_pipeline_system/05_hardware_configurations.md).

---

## Tradeoffs Summary

| Factor | TP (Symbiote) | PP (Blaze) | Hybrid PP+TP (Blaze) |
|--------|--------------|-----------|---------------------|
| **Latency/token** | Low compute + CCL overhead | Pipeline fill/drain | Moderate |
| **Throughput** | Limited by CCL bandwidth | High (continuous batching) | Highest |
| **Memory/device** | 1/N of every layer | Full layers for assigned stages | 1/TP of assigned layers |
| **CCL ops/layer** | 6-13 [Ch6/01](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) | 0 cross-stage; 2 intra [Ch6/02](../ch06_model_implementations/02_blaze_deepseek_v3_pipeline.md) | 2 intra per layer |
| **Max model size** | ~35B on T3K (96 GB) | 671B+ on 4 galaxies | Same as PP |
| **Effort** | Hours (module replacement) | Weeks (kernel engineering) | Weeks |
| **Batch support** | 1 (no multi-batch) | 64 concurrent users | 64 concurrent users |
| **Prefill** | HF chunked (fast) | Token-by-token (slow) | Same as PP |

---

## The Developer's Decision Flow

```
Model size?
  |
  <= 8B --> Single device (N150/P150) or DP
  |
  8B-35B --> TP on T3K (Symbiote)
  |          - Fastest time-to-working-model
  |          - HuggingFace model.generate() works
  |          - TracedRun for production decode speed
  |
  35B-100B --> TP on TG (Symbiote) or PP on galaxy (Blaze)
  |            - Symbiote: simpler but 32-way CCL is bandwidth-limited
  |            - Blaze: better throughput but requires kernel development
  |
  > 100B --> PP+TP on multi-galaxy (Blaze only)
             - Pipeline across galaxies, TP within submesh
             - PipelineManager for continuous batching
```

The 35B-100B range is the critical gap. Models here are too large for Symbiote's TP-only approach on T3K but too small to justify weeks of Blaze kernel development. See [File 04](./04_limitations_gaps_and_roadmap.md) for infrastructure changes that close this gap.

---

| Previous | Up | Next |
|----------|-----|------|
| [01_framework_comparison.md](./01_framework_comparison.md) | [Table of Contents](../README.md) | [03_integration_pathways.md](./03_integration_pathways.md) |
