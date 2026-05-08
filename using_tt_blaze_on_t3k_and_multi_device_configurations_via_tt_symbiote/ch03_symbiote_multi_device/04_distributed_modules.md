# Distributed Module Implementations

This file covers the concrete `TTNNModule` subclasses that compose the weight sharding strategies (File 2) and CCL operations (File 3) into complete distributed layers: normalization that gathers statistics across devices, rotary position embeddings that work with sharded heads, embeddings that shard the vocabulary table, Gemma4 attention and MLP modules that use fused QKV with all-reduce, Qwen3 MoE modules that distribute experts across devices, and KV cache replication strategies. Each module builds on the `DistributedConfig` system (File 1) and the linear layer hierarchy (File 2). Prerequisites: all three preceding files in this chapter, plus [Chapter 2, File 1](../ch02_symbiote_core/01_module_replacement_engine.md) (TTNNModule lifecycle, `@run_on_devices`).

---

## TTNNDistributedRMSNorm

`TTNNDistributedRMSNorm` [normalization.py:TTNNDistributedRMSNorm] implements RMSNorm across devices where each device holds only a shard of the hidden dimension. The challenge: RMSNorm computes `output = (x / rms(x)) * weight` where `rms(x) = sqrt(mean(x^2))`. When hidden states are column-sharded, no single device has the full `x` needed for `mean(x^2)`.

### Three-Phase Forward

```
Phase 1: rms_norm_pre_all_gather(inp)
  Each device computes partial sum-of-squares from its shard.
  Output: tt_stats [B, S, 32] per device (compact statistics).

Phase 2: all_gather(tt_stats, dim=-1, topology=Ring)
  Statistics gathered so each device has all 8 devices' stats.
  Output: tt_stats [B, S, 32*8] on all devices.

Phase 3: rms_norm_post_all_gather(inp, tt_stats, weight)
  Each device normalizes its local shard using global statistics.
  Output: normalized [B, S, H/8] per device (same sharding as input).
```

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, inp):
    original_shape = inp.shape
    if len(original_shape) == 3:
        inp = ttnn.unsqueeze(inp, 1)
    # Phase 1: local statistics
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, dtype=ttnn.bfloat16)
    # Phase 2: gather stats across devices (Ring for trace safety)
    tt_stats = ttnn.all_gather(
        tt_stats, dim=-1, num_links=1, topology=ttnn.Topology.Ring
    )
    # Phase 3: apply normalization with gathered stats
    eps = getattr(self.torch_layer, "variance_epsilon",
                  getattr(self.torch_layer, "eps", 1e-6))
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats, epsilon=eps, weight=self.weight_distributed
    )
    tt_stats.deallocate(True)
    return tt_out
```

### Weight Distribution

The norm weight is reshaped to `[1, 1, padded_dim//32, 32]` and sharded along dim 2 across mesh columns using `ShardTensor2dMesh(dims=(None, 2))` [normalization.py:TTNNDistributedRMSNorm.move_weights_to_device_impl]:

```python
def move_weights_to_device_impl(self):
    dim = self.torch_layer.weight.shape[0]
    padded_dim = ((dim + 31) // 32) * 32  # Pad to TILE boundary
    weight = self.torch_layer.weight
    if padded_dim != dim:
        weight = torch.nn.functional.pad(weight, (0, padded_dim - dim), value=1.0)
    self.weight_distributed = ttnn.as_tensor(
        weight.unsqueeze(0).view(1, 1, padded_dim // 32, 32).to(torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            self.device, dims=(None, 2), mesh_shape=list(self.device.shape)
        ),
    )
    self.weight_distributed = ttnn.to_device(self.weight_distributed, self.device)
```

> **Warning:** The padding uses `value=1.0`, not `0.0`. RMSNorm weight is multiplicative, so padding with 1.0 means padded elements act as identity (no scaling), while 0.0 would zero out those dimensions entirely.

### TTNNLocalRMSNorm: Per-Device Variant

`TTNNLocalRMSNorm` [normalization.py:TTNNLocalRMSNorm] performs normalization locally on each device without any CCL. It is used for per-head norms (Q-norm, K-norm, V-norm) in Gemma4 attention, where each device already holds complete head data after the fused QKV all-reduce. It handles Gemma4's `Gemma4RMSNorm(with_scale=False)` variant by creating a dummy all-ones weight when no learnable scale exists.

---

## TTNNDistributedRotaryPositionEmbedding

`TTNNDistributedRotaryPositionEmbedding` [rope.py:TTNNDistributedRotaryPositionEmbedding] applies RoPE using `ttnn.experimental.rotary_embedding_llama`, optimized for multi-device scenarios where Q/K heads are sharded across devices.

### Transformation Matrix Replication

The key device-side resource is a `[1, 1, head_dim, head_dim]` transformation matrix that implements `[x1, x2] -> [-x2, x1]`:

```python
def move_weights_to_device_impl(self):
    for is_decode in [True, False]:
        trans_mat = torch.zeros(1, 1, dhead, dhead)
        trans_mat[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
        trans_mat[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1

        mesh_mapper = None
        if isinstance(self.device, ttnn._ttnn.multi_device.MeshDevice):
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.device)

        self._trans_mat_cache[is_decode] = ttnn.from_torch(
            trans_mat, device=self.device, layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper,
        )
```

The transformation matrix is **replicated** across all devices because every device needs the same rotation pattern regardless of which head shard it holds. Since Q/K are head-sharded and cos/sin are replicated, each device independently applies RoPE to its local heads with no CCL needed.

### BailingRotarySetup: Pre-Computed Cos/Sin Cache

`BailingRotarySetup` [rope.py:BailingRotarySetup] pre-computes cos/sin caches and transformation matrices on device at initialization time, enabling trace-safe RoPE. It is used by the Gemma4 attention module and the Ling/Bailing models:

```python
class BailingRotarySetup:
    def __init__(self, device, head_dim, max_seq_len, rope_theta,
                 partial_rotary_factor=1.0, use_head_dim_for_freq=False):
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None
        # TILE_LAYOUT cos/sin for prefill (sliced by seq_len)
        self.cos_cache = ttnn.from_torch(cos_cache, device=device,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, ...)
        self.sin_cache = ttnn.from_torch(sin_cache, device=device,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper, ...)
        # ROW_MAJOR cos/sin for decode embedding lookup
        self.cos_cache_row_major = ttnn.from_torch(...)
        self.sin_cache_row_major = ttnn.from_torch(...)
        # Transformation matrices (replicated)
        self.trans_mat_decode = ttnn.from_torch(...)
        self.trans_mat_prefill = ttnn.from_torch(...)
```

All caches are replicated to every device. The `use_head_dim_for_freq` parameter handles a difference between RoPE conventions: standard models (Phi/Ling) compute inv_freq over `rotary_dim` only, while Gemma3/4 computes over full `head_dim` with only the first `rotary_dim` frequencies used. When `partial_rotary_factor < 1.0`, the cos/sin cache is identity-padded (`cos=1, sin=0`) for non-rotary dimensions.

---

## TTNNEmbedding

`TTNNEmbedding` [embedding.py:TTNNEmbedding] shards the embedding weight table along the hidden dimension (dim=1) across mesh columns:

```python
def move_weights_to_device_impl(self):
    mesh_mapper = ttnn.ShardTensor2dMesh(
        self.device, dims=(None, 1), mesh_shape=list(self.device.shape)
    )
    self.tt_weight = ttnn.to_device(
        ttnn.from_torch(
            self.torch_layer.weight.data, dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper,
        ),
        self.device,
    )
```

Each device holds `[vocab_size, hidden/N]`. Input indices are replicated (same tokens on every device), and `ttnn.embedding` looks up each device's shard, producing `[batch, seq, hidden/N]` per device. No CCL is needed because downstream column-parallel layers expect this sharded layout.

`TTNNBailingPaddedEmbedding` wraps `TTNNEmbedding` with power-of-2 sequence padding for optimal TTNN kernel performance, gated to T3K via `@run_on_devices(DeviceArch.T3K)`.

---

## Gemma4 Attention: Fused QKV with Distributed Projections

`TTNNGemma4Attention` [gemma4_attention.py:TTNNGemma4Attention] demonstrates the most sophisticated distributed attention pattern in TT-Symbiote, combining fused QKV projection, weight permutation for Meta-format RoPE, K=V sharing in global layers, partial RoPE for large head dimensions, per-head normalization, and shared rotary setup caching.

### Fused QKV Projection

Instead of three separate projections each requiring their own CCL, Gemma4 fuses Q, K, V weights into a single matmul [gemma4_attention.py:TTNNGemma4Attention.from_torch]:

```python
# Sliding layers: Q + K + V fused
fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
# Global layers (K=V sharing): only Q + K
fused_weight = torch.cat([q_weight, k_weight], dim=0)

self.qkv_proj = TTNNLinearIColShardedWAllReduced.from_torch(fused_linear)
```

The fused approach uses `TTNNLinearIColShardedWAllReduced`: a single matmul followed by `reduce_scatter` + `all_gather` produces a replicated output. This replaces 3 matmuls + 6 CCL ops with 1 matmul + 2 CCL ops (see [File 2](./02_weight_sharding_strategies.md) for the all-reduce variant details).

### Weight Permutation for Meta Interleaved Layout

The `rotary_embedding_llama` kernel expects interleaved cos/sin pairs (Meta format). Q/K weights must be permuted from HuggingFace's split-half format [gemma4_attention.py:_reverse_permute_weight]:

```python
def _reverse_permute_weight(tensor, n_heads, head_dim):
    dim1, dim2 = tensor.shape
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2) \
                 .transpose(1, 2).reshape(dim1, dim2)
```

Per-head norm weights must be permuted identically to maintain alignment [gemma4_attention.py:_reverse_permute_norm_weight]:

```python
def _reverse_permute_norm_weight(weight, head_dim):
    return weight.view(2, head_dim // 2).T.reshape(-1)
```

### K=V Sharing in Global Layers

Gemma4 global attention layers have `v_proj = None`. The K projection output is cloned for the V path, then de-interleaved to undo the Meta-format permutation that was applied to K but is not appropriate for V [gemma4_attention.py:_deinterleave_heads]:

```python
def _deinterleave_heads(tensor, head_dim):
    shape = list(tensor.shape)
    new_shape = shape[:-1] + [head_dim // 2, 2]
    tensor = ttnn.reshape(tensor, new_shape)
    perm = list(range(len(new_shape) - 2)) + [len(new_shape) - 1, len(new_shape) - 2]
    tensor = ttnn.permute(tensor, perm)
    tensor = ttnn.reshape(tensor, shape)
    return tensor
```

### Partial RoPE for Large Head Dimensions

The `rotary_embedding_llama` kernel has a maximum dimension of 256. Gemma4's global layers have `head_dim=512` but `rotary_dim=128`. The `_apply_rotary_embedding_llama` method handles three paths [gemma4_attention.py:TTNNGemma4Attention._apply_rotary_embedding_llama]:

1. **head_dim <= 256** (sliding layers): Direct single kernel call.
2. **rotary_dim <= 256 < head_dim** (global layers): Slice out rotary dims, apply one RoPE call, concatenate with pass-through dims.
3. **rotary_dim > 256**: Chunked fallback -- split into 256-dim chunks.

### Per-Head Normalization

Gemma4 uses per-head RMSNorm (Q-norm, K-norm, V-norm) instead of `1/sqrt(d)` scaling. These use `TTNNLocalRMSNorm` (no CCL needed):

```python
def _apply_per_head_norm(self, states, norm_module, batch_size, seq_length,
                         num_heads, head_dim):
    orig_shape = states.shape
    states = ttnn.reshape(states, (batch_size * seq_length * num_heads, head_dim))
    states = norm_module(states)  # TTNNLocalRMSNorm -- purely local
    states = ttnn.reshape(states, orig_shape)
    return states
```

### Shared Rotary Setup Cache

To avoid OOM from per-layer cos/sin allocation (~16-32MB per instance times 60 layers = ~1.1GB), `BailingRotarySetup` instances are shared across layers with the same configuration via a class-level cache:

```python
# Class-level in TTNNGemma4Attention:
_shared_rotary_setups = {}

# In move_weights_to_device_impl:
setup_key = (id(self.device), self.head_dim, rope_theta, partial_rotary_factor)
if setup_key not in TTNNGemma4Attention._shared_rotary_setups:
    TTNNGemma4Attention._shared_rotary_setups[setup_key] = BailingRotarySetup(...)
self._rotary_setup = TTNNGemma4Attention._shared_rotary_setups[setup_key]
```

Only 2 unique configs exist for Gemma4 (sliding: head_dim=256, rope_theta=10000; global: head_dim=512, rope_theta=1000000), reducing memory from ~1.1GB to ~64MB.

### O-Projection

The output projection uses `TTNNLinearIReplicatedWColSharded` -- input is replicated (from attention output after all-reduce), weight is column-sharded, producing column-sharded output for the residual stream:

```python
LinearClsOut = TTNNLinearIReplicatedWColSharded if distributed else TTNNLinear
self.o_proj = LinearClsOut.from_torch(hf_attn.o_proj)
```

---

## Gemma4 MLP: Fused Gate-Up Projection

`TTNNGemma4TextMLP` [gemma4_mlp.py:TTNNGemma4TextMLP] applies the same fusion strategy as attention:

```python
gate_weight = torch_mlp.gate_proj.weight.data.clone()   # [21504, 5376]
up_weight = torch_mlp.up_proj.weight.data.clone()        # [21504, 5376]
fused_weight = torch.cat([gate_weight, up_weight], dim=0) # [43008, 5376]
tt_module.fused_gate_up_proj = TTNNLinearIColShardedWAllReduced.from_torch(fused_linear)
tt_module.down_proj = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.down_proj)
```

The forward pass:

```
Input (col-sharded): [B, S, H/8]
  |
  fused_gate_up_proj (TTNNLinearIColShardedWAllReduced)
  matmul + reduce_scatter + all_gather
  |
Gate+Up (replicated): [B, S, 2*I]
  |
  slice -> gate [B,S,I], up [B,S,I]
  gelu(gate) * up
  |
Intermediate (replicated): [B, S, I]
  |
  down_proj (TTNNLinearIReplicatedWColSharded)
  matmul only (no CCL)
  |
Output (col-sharded): [B, S, H/8]
```

Note the aggressive `ttnn.deallocate()` calls in the forward to free intermediate tensors (gate_up, gate, up, gate_up_mul) as soon as they are no longer needed.

---

## TTNNQwen3MoE: Expert Parallelism

`TTNNQwen3MoE` [qwen_moe.py:TTNNQwen3MoE] implements Mixture-of-Experts for Qwen3.5-35B-A3B (256 experts, top-8 routing). With 8 devices on T3K, each device holds 32 experts.

### Forward Flow

```
 1. all_gather(x, dim=-1)            -- reconstruct full hidden for routing
 2. Router: softmax -> 3-pass centering topk -> normalize -> scale
 3. all_to_all_dispatch               -- route tokens to expert-owning devices
 4. sparse_matmul (fused w1/w3)       -- gate+up projection on local experts
 5. silu(gate) * up                   -- activation
 6. sparse_matmul (w2)                -- down projection
 7. all_to_all_combine                -- route expert outputs back
 8. Apply routing weights + sum over experts
 9. reduce_scatter(dim=3)             -- re-shard output for TP
10. Add shared_expert output (optionally gated by shared_expert_gate)
```

### Expert Weight Sharding

Expert weights are sharded across devices using `ShardTensorToMesh(device, dim=1)` [qwen_moe.py:TTNNQwenExperts.preprocess_weights_impl]. The fused w1/w3 weight eliminates duplicate memory bandwidth:

```python
torch_w1_w3_fused = torch.cat([torch_w1_4d, torch_w3_4d], dim=-1)  # [1, 256, H, 2*I]
self.tt_w1_w3_proj = ttnn.from_torch(
    torch_w1_w3_fused, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1),
)
```

Each device gets experts `[32*i : 32*(i+1)]`. The `ttnn.sparse_matmul` operation with a sparsity tensor ensures only activated experts are computed, avoiding wasted compute on the 24+ inactive experts per device.

### All-to-All Dispatch and Combine

Unlike symmetric `all_gather`/`reduce_scatter`, MoE requires asymmetric communication where different tokens go to different devices based on routing decisions:

```python
all_to_all_dispatch_output, metadata = ttnn.all_to_all_dispatch(
    x_rm, topk_experts_indices_rm, self.expert_mapping_tensors,
    cluster_axis=1, memory_config=decode_memory_config,
)
# ... expert computation with sparse_matmul ...
combined_output = ttnn.all_to_all_combine(
    expert_output, metadata, self.expert_mapping_tensors,
    cluster_axis=1, memory_config=decode_memory_config,
)
```

The `expert_mapping_tensors` is a replicated identity matrix mapping expert indices to device indices.

### Router: 3-Pass Centering TopK

`TTNNQwenMoERouterDecode` [qwen_moe.py:TTNNQwenMoERouterDecode.forward] uses a 3-pass centering technique to work around BF16 precision limits when selecting top-8 from 256 experts. With softmax scores clustered near ~1/256 = 0.0039, BF16 has only ~3 decimal digits of precision at that range:

```
Pass 1: rough BF16 topk(k+1) -> coarse threshold
Pass 2: center by coarse threshold -> refined BF16 topk(k+1) -> fine threshold
Pass 3: center by fine threshold -> final BF16 topk(k)
```

Each centering step moves the decision boundary closer to zero where BF16 has maximum resolution (~0.0001).

### Output Tensor Config Override

`TTNNQwen3MoE` overrides `set_output_tensors_config_impl()` to declare that its output is column-sharded (after `reduce_scatter`), ensuring downstream modules receive the correct `DistributedTensorConfig` [qwen_moe.py:TTNNQwen3MoE.set_output_tensors_config_impl]:

```python
def set_output_tensors_config_impl(self, output_tensors):
    def set_col_sharded_config(e):
        if isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None:
            config = DistributedTensorConfig(
                mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=-1),
                mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1),
                logical_shape_fn=logical_shape_for_col_sharded,
            )
            e.set_distributed_tensor_config(config)
        return e
    return tree_map(set_col_sharded_config, output_tensors)
```

This override is necessary because the default `DistributedTensorConfig` (set during `set_device()`) assumes 2D batch+channel sharding, which would incorrectly interpret the 1D column-sharded MoE output.

---

## KV Cache on Multi-Device

### Pattern 1: Replicated KV Cache (Gemma4)

Gemma4 uses `TTNNGemma4PagedAttentionKVCache` [gemma4_attention.py:TTNNGemma4PagedAttentionKVCache], which routes operations to two sub-caches based on layer type:

```
Sliding layers: 16 KV heads, head_dim=256, sliding window
Global layers:  4 KV heads,  head_dim=512, full context
```

```python
class TTNNGemma4PagedAttentionKVCache(Cache):
    def __init__(self, text_config, global_layer_indices, device=None):
        self._routing = {}
        for layer_idx in range(text_config.num_hidden_layers):
            if layer_idx in self.global_layer_indices:
                self._routing[layer_idx] = ("global", global_idx)
            else:
                self._routing[layer_idx] = ("sliding", sliding_idx)

        self.sliding_cache = TTNNPagedAttentionKVCache(
            num_layers=num_sliding, num_kv_heads=16, head_dim=256,
            config=PagedAttentionConfig(block_size=64, max_num_blocks=32),
        )
        self.global_cache = TTNNPagedAttentionKVCache(
            num_layers=num_global, num_kv_heads=4, head_dim=512,
            config=PagedAttentionConfig(block_size=64, max_num_blocks=64),
        )
```

Both sub-caches are replicated to all devices via `ReplicateTensorToMesh` when `to_device()` is called. This works because the fused QKV all-reduce produces replicated Q/K/V -- attention SDPA operates on replicated tensors, and cache updates write the same K/V to all devices.

For multi-device decode, the attention module pre-allocates a replicated position buffer for trace stability:

```python
if self.device.get_num_devices() > 1:
    self._decode_cur_pos = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32), device=self.device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
    )
```

The current position is copied into this pre-allocated buffer via `ttnn.copy()` at each decode step, ensuring the tensor's device address remains stable across trace replay iterations.

### Pattern 2: Gathered KV (Ling/Bailing)

Some attention modules store KV sharded and use `_maybe_all_gather()` to reconstruct full KV before SDPA:

```python
def _maybe_all_gather(self, tensor):
    if not self._is_distributed:
        return tensor
    return ttnn.all_gather(
        tensor, dim=-1, num_links=1, cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )
```

> **Warning:** `_maybe_all_gather()` checks `self._is_distributed` (which queries `self.device_state.ccl_manager`). If `set_device()` was not called or the device state is not properly initialized, this check returns `False` and skips the all-gather, leading to silent dimension mismatches in the SDPA computation.

---

## Module Selection Guide

| Component | Single-Device Module | Multi-Device Module | Gating |
|-----------|---------------------|---------------------|--------|
| Layer Norm | `TTNNRMSNorm` | `TTNNDistributedRMSNorm` | `@run_on_devices(T3K)` |
| Per-Head Norm | `TTNNLocalRMSNorm` | `TTNNLocalRMSNorm` (same) | None |
| RoPE | `TTNNRotaryPositionEmbedding` | `TTNNDistributedRotaryPositionEmbedding` | None |
| Embedding | `TTNNEmbedding` | `TTNNEmbedding` (shards on dim 1) | None |
| QKV Projection | `TTNNLinear` | `TTNNLinearIColShardedWAllReduced` (fused) | `@run_on_devices(T3K)` |
| O/Down Projection | `TTNNLinear` | `TTNNLinearIReplicatedWColSharded` | `@run_on_devices(T3K)` |
| MoE | N/A | `TTNNQwen3MoE` / `TTNNMoE` | `@run_on_devices(T3K)` |

> **Warning:** All distributed forward methods are gated with `@run_on_devices(DeviceArch.T3K)`. If you add a new distributed module, you must decorate the T3K forward path. Without the decorator, the base class single-device forward executes even on multi-device, silently producing incorrect results from the non-distributed code path.

### CCL Operations Per Transformer Block

For a standard Gemma4 transformer block (attention + MLP + 2x RMSNorm):

| Module | CCL Operations |
|--------|---------------|
| Pre-attention RMSNorm | 1x `all_gather` (stats) |
| Fused QKV projection | 1x `reduce_scatter` + 1x `all_gather` |
| O-projection | None (column-parallel) |
| Pre-MLP RMSNorm | 1x `all_gather` (stats) |
| Fused gate-up projection | 1x `reduce_scatter` + 1x `all_gather` |
| Down projection | None (column-parallel) |
| **Total per block** | **6 CCL operations** |

For MoE blocks, add `all_gather` (input), `all_to_all_dispatch`, `all_to_all_combine`, and `reduce_scatter` (output) for the expert routing path.

---

**Previous:** [`03_ccl_operations.md`](./03_ccl_operations.md) | **Up:** [index.md](./index.md)
