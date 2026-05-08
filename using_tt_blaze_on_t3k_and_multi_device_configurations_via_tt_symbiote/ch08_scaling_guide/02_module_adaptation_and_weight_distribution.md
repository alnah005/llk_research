# Chapter 8 -- Part 2: Module Adaptation and Weight Distribution

This section covers the implementation phase: building the TP replacement map for T3K, implementing weight fusion in `from_torch()`, preprocessing and distributing weights across 8 devices, handling non-shardable tensors, writing the custom decoder layer and model wrapper, and wiring up `@run_on_devices` gating. Each section starts with the failure you will hit if the step is done incorrectly. The worked example is Gemma4-31B; each step generalizes.

**Prerequisites:** Part 1 of this chapter (analysis outputs), [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md) (linear module hierarchy), [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md) (distributed modules), [Ch6/01](../ch06_model_implementations/01_symbiote_tensor_parallel_models.md) (model implementations).

---

## Step 1: Build the Ordered Multi-Pass Replacement Map

### What Goes Wrong If You Skip This (or Get It Wrong)

You write a single-pass replacement dict that maps `nn.Linear` to `TTNNLinearIColShardedWRowSharded`. Two things break: (1) the decoder layer's `RMSNorm` children have been walked by the DFS before you can replace them -- norm layers stay as `TTNNRMSNorm` (no distributed stats), producing subtly wrong output; (2) the model wrapper is never replaced, so `_bypass_tensor_wrapping` is never set, and every inter-layer boundary forces a host synchronization adding ~50ms per layer per forward [Ch6/03, Pattern 5](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

### The Correct Approach

Follow the 3-pass pattern established by Gemma4 and Ling [test_gemma4.py, lines 126-144]:

```python
# Pass 1: Structural modules (highest specificity first)
pass_1_dict = {
    Gemma4TextDecoderLayer: TTNNGemma4DecoderLayer,
    Gemma4RMSNorm:          TTNNDistributedRMSNorm,
    Gemma4ScaledWordEmbed:  TTNNGemma4ScaledEmbedding,
}
exclude_vision = {name for name, _ in model.named_modules()
                  if "vision_tower" in name or "embed_vision" in name}
modules_1 = register_module_replacement_dict(
    model, pass_1_dict, exclude_replacement=exclude_vision)

# Pass 2: Remaining linears (blanket, catches lm_head and stragglers)
pass_2_dict = {nn.Linear: TTNNLinearIColShardedWRowSharded}
modules_2 = register_module_replacement_dict(model, pass_2_dict)

# Pass 3: Model wrapper (outermost, enables TTNNLayerStack + bypass)
pass_3_dict = {Gemma4TextModel: TTNNGemma4TextModel}
modules_3 = register_module_replacement_dict(model, pass_3_dict)
```

**For MoE models with small gating linears** (output_size < 8), exclude them from Pass 2:

```python
small_gates = {name for name, mod in model.named_modules()
               if isinstance(mod, nn.Linear) and mod.out_features < 8}
modules_2 = register_module_replacement_dict(
    model, pass_2_dict, exclude_replacement=small_gates)
```

> **Warning:** `register_module_replacement_dict` performs a single DFS walk per call. It will NOT re-visit already-replaced subtrees. Always replace higher-level modules first, then inner generic modules, then the outermost model wrapper [Ch6/03, Pattern 3](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

---

## Step 2: Implement Weight Fusion in from_torch()

### What Goes Wrong If You Skip This

Your model uses separate Q, K, V projections with individual `reduce_scatter` operations (6 CCL ops for attention projections). The equivalent fused QKV approach needs only 1 matmul + 2 CCL ops. At 60 layers, unfused wastes ~1.1ms per token on redundant CCL.

### Fused QKV Projection (Gemma4 Pattern)

The attention module concatenates Q, K, and V weight matrices along the output dimension [gemma4_attention.py, lines 196-211]:

```python
@classmethod
def from_torch(cls, hf_attn, distributed=True):
    new_attn = cls()
    # Permute Q/K weights from HF split-half to Meta interleaved layout
    q_weight = _reverse_permute_weight(
        hf_attn.q_proj.weight.data.clone(),
        new_attn.num_attention_heads, new_attn.head_dim)
    k_weight = _reverse_permute_weight(
        hf_attn.k_proj.weight.data.clone(),
        new_attn.num_key_value_heads, new_attn.head_dim)

    # Fuse: concatenate along output dimension
    if hf_attn.v_proj is not None:
        v_weight = hf_attn.v_proj.weight.data.clone()
        fused_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    else:
        fused_weight = torch.cat([q_weight, k_weight], dim=0)  # K=V sharing

    fused_linear = nn.Linear(hidden_size, fused_weight.shape[0], bias=False)
    fused_linear.weight.data = fused_weight
    new_attn.qkv_proj = TTNNLinearIColShardedWAllReduced.from_torch(fused_linear)
    new_attn._q_size = num_attention_heads * head_dim
    new_attn._kv_size = num_key_value_heads * head_dim
    return new_attn
```

In `forward()`, the fused output is sliced back [gemma4_attention.py, lines 458-471]:

```python
qkv_states = self.qkv_proj(hidden_states)
query_states = ttnn.slice(qkv_states, [0, 0, 0], [B, S, q_size])
key_states = ttnn.slice(qkv_states, [0, 0, q_size], [B, S, q_size + kv_size])
if self._has_v_proj:
    value_states = ttnn.slice(qkv_states, [0, 0, q_size + kv_size],
                              [B, S, q_size + 2 * kv_size])
else:
    value_states = ttnn.clone(key_states)  # K=V sharing for global layers
ttnn.deallocate(qkv_states)
```

### Weight Permutation for RoPE Compatibility

HF stores RoPE-compatible weights in split-half format. TTNN's `rotary_embedding_llama` expects interleaved [gemma4_attention.py, line 47]:

```python
def _reverse_permute_weight(tensor, n_heads, head_dim):
    dim1, dim2 = tensor.shape
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2) \
                 .transpose(1, 2).reshape(dim1, dim2)
```

> **Warning:** Forgetting to permute the norm weights while permuting Q/K weights causes a subtle numerical error: per-head norm applies the wrong scale to each dimension. The model produces garbage output but does not crash. Always permute both Q/K weights AND their corresponding per-head norms [gemma4_attention.py, line 57].

### Fused Gate+Up Projection (Same Pattern)

```python
gate_weight = torch_mlp.gate_proj.weight.data.clone()   # [21504, 5376]
up_weight = torch_mlp.up_proj.weight.data.clone()        # [21504, 5376]
fused_weight = torch.cat([gate_weight, up_weight], dim=0) # [43008, 5376]
fused_linear = nn.Linear(5376, 43008, bias=False)
fused_linear.weight.data = fused_weight
tt_module.fused_gate_up_proj = TTNNLinearIColShardedWAllReduced.from_torch(fused_linear)
```

Source: [gemma4_mlp.py, lines 36-49].

---

## Step 3: Weight Preprocessing and Distribution

### What Goes Wrong If You Skip This

You call `set_device(model, mesh_device)` followed by `move_weights_to_device()`, and the first forward pass crashes:

```
AssertionError: Weight tensor shape [4096, 4096] dim -2 (4096) is not divisible
by number of devices (8) for shard_tensor_to_mesh_mapper
```

The weight was never preprocessed. `preprocess_weights_impl()` must be called first to transpose, tile, and prepare for sharding [Ch2/04, Stage 4](../ch02_symbiote_core/04_end_to_end_model_flow.md).

### The Weight Pipeline

```python
all_modules = {**modules_1, **modules_2, **modules_3}
set_device(model, mesh_device)

for name, mod in tqdm(all_modules.items(), desc="Preprocessing"):
    mod.preprocess_weights()

for name, mod in tqdm(all_modules.items(), desc="Moving to device"):
    mod.move_weights_to_device()
```

Each distributed linear handles sharding in `move_weights_to_device_impl()` [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md):

- **Column-parallel** (`TTNNLinearIReplicatedWColSharded`): shards on `dim=-1`. `[4096, 4096]` becomes `[4096, 512]` per device.
- **Row-parallel** (`TTNNLinearIColShardedWRowSharded`): shards on `dim=-2`. `[4096, 4096]` becomes `[512, 4096]` per device.
- **All-reduce variant** (`TTNNLinearIColShardedWAllReduced`): same as row-parallel, but output goes through `reduce_scatter` + `all_gather` to produce replicated result.

---

## Step 4: Handle Non-Shardable Tensors

### Replicated Tensors

Some tensors must be replicated to all devices rather than sharded:

1. **RoPE cos/sin caches** -- identical rotation patterns. Use `ReplicateTensorToMesh` [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md).
2. **Causal masks** -- replicated via `ttnn.ReplicateTensorToMesh` [gemma4_text.py, lines 149-159].
3. **KV cache buffers** -- Gemma4 replicates KV caches because fused QKV all-reduce produces replicated K/V [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md).
4. **Per-head norm weights** -- after all-reduce produces replicated Q/K/V, per-head norms operate locally using `TTNNLocalRMSNorm` (no CCL).
5. **Small scalar tensors** (rank < 2) -- automatically replicated by `get_tensor_config_for_tensor` [Ch3/01](../ch03_symbiote_multi_device/01_distributed_config_and_device_init.md).

> **Warning:** The `DistributedRMSNorm` weight uses special sharding: reshaped to `[1, 1, padded_dim//32, 32]` and sharded via `ShardTensor2dMesh(dims=(None, 2))`. The padding uses `value=1.0`, not `0.0`, because RMSNorm weight is multiplicative -- padding with 0 would zero out those dimensions [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md).

---

## Step 5: Write the Custom Decoder Layer

### What Goes Wrong If You Skip This

Without a custom decoder layer, residual adds execute on CPU. Every layer boundary forces `ttnn.synchronize_device`, then `aten::add` on host, then transfer back. For a 60-layer model at decode, this adds ~3 seconds per forward pass (50ms per layer x 60 layers).

### On-Device Residuals

```python
class TTNNMyDecoderLayer(TTNNModule):
    @classmethod
    def from_torch(cls, torch_layer, **kwargs):
        tt_module = cls()
        tt_module.torch_layer = torch_layer
        tt_module.input_layernorm = torch_layer.input_layernorm
        tt_module.self_attn = torch_layer.self_attn
        tt_module.post_attention_layernorm = torch_layer.post_attention_layernorm
        tt_module.mlp = torch_layer.mlp
        return tt_module

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states, **kwargs):
        residual = hidden_states                          # stays on device
        hs = self.input_layernorm.forward(hidden_states)  # TTNNDistributedRMSNorm
        attn_out = self.self_attn.forward(hs, **kwargs)   # device ops + CCL
        hs = ttnn.add(residual, attn_out)                 # ON DEVICE

        residual = hs
        hs = self.post_attention_layernorm.forward(hs)
        mlp_out = self.mlp.forward(hs, **kwargs)
        hs = ttnn.add(residual, mlp_out)                  # ON DEVICE
        return hs
```

> **Warning:** Do NOT call `ttnn.deallocate(residual)` inside a `@trace_enabled` forward. During trace replay, this frees the pre-allocated trace input buffer, causing "Buffer is not allocated" on the next replay. Only deallocate tensors allocated within the traced forward [Ch6/03, Antipattern 3](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

---

## Step 6: Write the Model Wrapper

### What Goes Wrong If You Skip This

Without a model wrapper: (1) each decoder layer is traced individually -- 60 traces, 60 input copies, 60 `execute_trace()` calls, with host dispatch dominating [Ch2/03](../ch02_symbiote_core/03_run_modes.md); (2) `model.device` crashes via `StopIteration` because `next(self.parameters())` fails after full module replacement [Ch6/03, Antipattern 1](../ch06_model_implementations/03_reusable_patterns_and_antipatterns.md).

### The Correct Approach

```python
@trace_enabled
class TTNNMyTextModel(TTNNModule):
    @classmethod
    def from_torch(cls, torch_model, **kwargs):
        tt_module = cls()
        tt_module.torch_layer = torch_model
        tt_module.layer_stack = TTNNLayerStack(torch_model.layers)
        for layer in torch_model.layers:
            if isinstance(layer, TTNNModule):
                layer._bypass_tensor_wrapping = True
        return tt_module

    def forward(self, hidden_states, **kwargs):
        return self.layer_stack.forward(hidden_states, **kwargs)

# After all replacements, patch model.device
type(model).device = property(lambda self: torch.device("cpu"))
```

The `TTNNLayerStack` wraps all decoder layers into a single traceable unit, reducing host dispatch from O(N_layers) to O(1) [Ch2/03](../ch02_symbiote_core/03_run_modes.md).

---

## Step 7: Handle Output Tensor Config Overrides

MoE modules that perform `reduce_scatter` produce col-sharded output, but the default `DistributedTensorConfig` assumes 2D batch+channel sharding. Without an override, downstream `ttnn.to_torch()` concatenates along wrong dimensions, producing garbled activations. Override `set_output_tensors_config_impl()` as Qwen3 MoE does [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md):

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

---

## Complete Replacement Map Reference (T3K TP=8)

| Original Module | Distributed Replacement | Weight Shard | CCL | Pattern |
|-----------------|------------------------|-------------|-----|---------|
| Fused QKV | `TTNNLinearIColShardedWAllReduced` | dim=-2 | RS + AG | B |
| O projection | `TTNNLinearIReplicatedWColSharded` | dim=-1 | None | B |
| Fused gate+up | `TTNNLinearIColShardedWAllReduced` | dim=-2 | RS + AG | B |
| Down projection | `TTNNLinearIReplicatedWColSharded` | dim=-1 | None | B |
| All linears (Ling) | `TTNNLinearIColShardedWRowSharded` | dim=-2 | RS | A |
| RMSNorm | `TTNNDistributedRMSNorm` | dim=2 (2D mesh) | AG (stats) | Both |
| Embedding | `TTNNEmbedding` | dim=1 (hidden) | None | Both |
| Small linear (out<8) | `TTNNLinear` (replicated) | None | None | Both |

Source: [Ch3/02](../ch03_symbiote_multi_device/02_weight_sharding_strategies.md), [Ch3/04](../ch03_symbiote_multi_device/04_distributed_modules.md).

---

## Putting It All Together

The complete sequence: (1) build exclusion set, (2) Pass 1 structural, (3) Pass 2 blanket linears, (4) Pass 3 model wrapper, (5) patch `model.device`, (6) `set_device()`, (7) preprocess weights, (8) move weights to device. Each step is idempotent. Use `tqdm` loops for visibility on large models where weight transfer takes minutes.

---

| Previous | Up | Next |
|----------|-----|------|
| [Part 1: Model Analysis and Strategy Selection](01_model_analysis_and_strategy_selection.md) | [Table of Contents](../README.md) | [Part 3: Test Setup and Validation](03_test_setup_and_validation.md) |
