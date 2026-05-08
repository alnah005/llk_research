# Chapter 3 -- TT-Symbiote Multi-Device Support

This chapter covers how TT-Symbiote extends its single-device architecture (Chapter 2) to tensor-parallel execution across mesh devices. It details the `DistributedConfig` system that auto-activates on multi-chip hardware, the weight sharding strategies that partition linear layers across devices, the CCL operations that synchronize partial results, and the distributed module implementations (RMSNorm, RoPE, Embedding, Gemma4 attention/MLP, MoE) that compose these primitives into working multi-device inference. Prerequisites: Chapter 1 (mesh shapes, fabric config, topology) and Chapter 2 (TTNNModule lifecycle, TorchTTNNTensor dispatch, traced execution).

## Contents

1. [**`01_distributed_config_and_device_init.md`**](./01_distributed_config_and_device_init.md) -- Distributed Configuration and Device Initialization
   - `DistributedConfig` dataclass, `DistributedTensorConfig`, default sharding with `ShardTensor2dMesh(0, -1)`, logical shape computation, automatic replication fallback via `get_tensor_config_for_tensor()`, `DeviceInit` singleton, `CCLManagerConfig`, the `set_device()` recursive walk, and `get_default_distributed_tensor_config()` bridging dispatch to distributed state.

2. [**`02_weight_sharding_strategies.md`**](./02_weight_sharding_strategies.md) -- Weight Sharding Strategies
   - Column-parallel (`TTNNLinearIReplicatedWColSharded`), row-parallel (`TTNNLinearIColShardedWRowSharded`), all-reduce variant (`TTNNLinearIColShardedWAllReduced`), mesh mappers via `shard_tensor_to_mesh_mapper`, bfloat8 LLaMA variants, and the replicated-input-to-sharded-activation flow pattern.

3. [**`03_ccl_operations.md`**](./03_ccl_operations.md) -- CCL Operations
   - `TT_CCL` class initialization and semaphore management, `get_num_links()` per-topology lookup, `ttnn.reduce_scatter`, `ttnn.all_gather`, `tt_all_reduce()` topology adaptation, trace-compatible all-reduce decomposition, `cluster_axis` semantics, double-buffered semaphore cycling, Ring vs Linear topology selection, and CCL operation costs on T3K.

4. [**`04_distributed_modules.md`**](./04_distributed_modules.md) -- Distributed Module Implementations
   - `TTNNDistributedRMSNorm` (three-phase normalization with all-gather), `TTNNDistributedRotaryPositionEmbedding` and `BailingRotarySetup`, `TTNNEmbedding` with hidden-dim sharding, Gemma4 fused QKV attention (weight permutation, K=V sharing, partial RoPE, shared rotary setup), Gemma4 gated MLP, `TTNNQwen3MoE` expert parallelism with 3-pass centering topk, and KV cache replication.

## Reading Guide

Developers who only need tensor-parallel linear layers should read files 01 and 02. Those building custom distributed modules should additionally read file 03 for CCL operation details. File 04 serves as a reference catalog of existing distributed module implementations that can be used directly or adapted for new model architectures. Readers on the "just make it work on T3K" path can skim File 1 for setup, then jump to File 2 for the weight sharding recipes they will apply in Chapter 8.

---

**Next:** [`01_distributed_config_and_device_init.md`](./01_distributed_config_and_device_init.md)
