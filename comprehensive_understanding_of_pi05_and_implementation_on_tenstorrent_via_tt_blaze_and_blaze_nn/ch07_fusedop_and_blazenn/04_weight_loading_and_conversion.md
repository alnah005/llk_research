# 04 -- Weight Loading and Conversion: HF Safetensors to TT-Blaze

## Context

Pi0.5 weights are distributed as HuggingFace safetensors checkpoints. Loading them
into TT-Blaze requires:

1. **Name mapping**: HF key paths follow a different convention than blaze-nn
   parameter names
2. **Transposes**: HF stores weights in `[out_features, in_features]` while Blaze
   matmul ops expect `[in_features, out_features]` (pre-transposed)
3. **GatedFFN handling**: The gating einsum weight `[2, width, mlp_dim]` packs gate
   and up projections into a single tensor
4. **Quantization**: Some checkpoints use FP8 or mixed precision

The existing `WeightProvider` pattern in `blaze/weight_provider.py` provides the
abstraction framework. Pi0.5 needs a new provider that understands the PaliGemma +
action expert naming scheme.

## Key Takeaways

1. The naming convention in `openpi/models/gemma.py` uses `_name(name, i)` -- the
   first expert (VLM, index 0) has no suffix, the second (action, index 1) gets
   a `_1` suffix. This directly determines the HF state dict key structure.

2. All linear weights must be transposed (`.T.contiguous()`) when loading from HF
   into Blaze. This is the same transform used by the DeepSeek weight provider.

3. The gated FFN `gating_einsum` parameter `[2, width, mlp_dim]` stores two
   matrices stacked along dim 0: `gating_einsum[0]` = gate projection,
   `gating_einsum[1]` = up projection. Blaze's `kn_sliced_matmul` consumes this
   directly.

4. adaRMSNorm modulation weights (`Dense(width*3)`) are stored as a standard
   linear layer in HF format. The weight shape is `[width*3, width]` and needs
   transposing to `[width, width*3]`.

5. The `WeightProvider` ABC pattern ensures that inference code is agnostic to
   weight source -- synthetic, HF safetensors, or Blitz cache all produce the
   same tensor dict format.

## HF State Dict Key Structure

The Pi0.5 checkpoint follows PaliGemma's naming convention. The `_name(name, i)`
function in `gemma.py` determines suffixes:

```python
def _name(name, i):
    if i == 0:
        return name       # VLM expert: no suffix
    return f"{name}_{i}"  # Action expert: "_1" suffix
```

### Complete key mapping table

```
HF State Dict Key                                          | blaze-nn Parameter Path
------------------------------------------------------------|----------------------------------------------
PaliGemma.llm.params.embedder.input_embedding              | backbone.embedder.input_embedding
                                                            |
PaliGemma.llm.params.layers.attn.q_einsum.w                | backbone.layers[i].attn.q_einsum
PaliGemma.llm.params.layers.attn.kv_einsum.w               | backbone.layers[i].attn.kv_einsum
PaliGemma.llm.params.layers.attn.attn_vec_einsum.w         | backbone.layers[i].attn.attn_vec_einsum
PaliGemma.llm.params.layers.attn.q_einsum_1.w              | backbone.layers[i].attn.q_einsum_1
PaliGemma.llm.params.layers.attn.kv_einsum_1.w             | backbone.layers[i].attn.kv_einsum_1
PaliGemma.llm.params.layers.attn.attn_vec_einsum_1.w       | backbone.layers[i].attn.attn_vec_einsum_1
                                                            |
PaliGemma.llm.params.layers.pre_attention_norm.scale        | backbone.layers[i].pre_attention_norm.scale
PaliGemma.llm.params.layers.pre_attention_norm_1.__call__.Dense_0.kernel | backbone.layers[i].pre_attention_norm_1.w_mod
                                                            |
PaliGemma.llm.params.layers.mlp.gating_einsum              | backbone.layers[i].mlp.gating_einsum
PaliGemma.llm.params.layers.mlp.linear                     | backbone.layers[i].mlp.linear
PaliGemma.llm.params.layers.mlp_1.gating_einsum            | backbone.layers[i].mlp_1.gating_einsum
PaliGemma.llm.params.layers.mlp_1.linear                   | backbone.layers[i].mlp_1.linear
                                                            |
PaliGemma.llm.params.layers.pre_ffw_norm.scale              | backbone.layers[i].pre_ffw_norm.scale
PaliGemma.llm.params.layers.pre_ffw_norm_1.__call__.Dense_0.kernel | backbone.layers[i].pre_ffw_norm_1.w_mod
                                                            |
PaliGemma.llm.params.final_norm.scale                       | backbone.final_norm.scale
PaliGemma.llm.params.final_norm_1.__call__.Dense_0.kernel   | backbone.final_norm_1.w_mod
                                                            |
action_in_proj.kernel                                       | action_in_proj.weight
time_mlp_in.kernel                                          | time_mlp_in.weight
time_mlp_out.kernel                                         | time_mlp_out.weight
action_out_proj.kernel                                      | action_out_proj.weight
```

Note: The HF key structure reflects Flax's naming conventions (`.params.`, `.__call__.`,
`.Dense_0.kernel`). The exact key paths depend on the checkpoint format -- some
checkpoints flatten the layer scan dimension, others use a leading axis. The `layers`
dimension is typically a leading batch dimension in the scanned parameter tensors
(shape `[18, ...]` for 18 layers).

## Transpose Rules

All linear/einsum weights in HF safetensors are stored in `[out, in]` order (PyTorch
convention). Blaze matmul ops expect `[in, out]` (pre-transposed for the hardware's
systolic array flow). The conversion rule is:

```
Transform per weight type:

Weight Type          | HF Shape              | Blaze Shape           | Transform
---------------------|----------------------|----------------------|------------------
Linear (2D)          | [out, in]            | [in, out]            | .T.contiguous()
Q einsum (3D)        | [num_heads, width, head_dim] | same          | None (einsum)
KV einsum (4D)       | [2, kv_heads, width, head_dim] | same       | None (einsum)
O einsum (3D)        | [num_heads, head_dim, width] | same          | None (einsum)
Gating einsum (3D)   | [2, width, mlp_dim]  | same                 | None (stacked)
FFN down proj (2D)   | [mlp_dim, width]     | [width, mlp_dim].T   | .T.contiguous()
adaRMS modulation    | [width*3, width]     | [width, width*3]     | .T.contiguous()
RMSNorm scale (1D)   | [width]              | [1, width]           | .unsqueeze(0)
```

The einsum weights (Q, KV, O) are stored as multi-dimensional tensors that the
einsum operations consume directly. They do not need transposing.

## GatedFFN Weight Handling

The gated FFN in Gemma uses a packed gating einsum:

```python
# From gemma.py FeedForward:
w_gating = self.param("gating_einsum", ..., (2, self.features, self.hidden_dim))
ff_gate = jnp.dot(x, w_gating[0])     # gate projection
gate_value = nn.gelu(ff_gate)
ff1 = jnp.dot(x, w_gating[1])          # up projection
activations = gate_value * ff1
```

The weight tensor has shape `[2, width, mlp_dim]`:
- `gating_einsum[0]`: gate weights `[width, mlp_dim]`
- `gating_einsum[1]`: up weights `[width, mlp_dim]`

In Blaze, this maps to `F.sliced_matmul` which accepts the full `[2, width, mlp_dim]`
tensor and a `branch` parameter:

```python
gate = F.sliced_matmul(x, self.gating_einsum, branch="gate")  # uses [0]
up   = F.sliced_matmul(x, self.gating_einsum, branch="up")    # uses [1]
```

The `kn_sliced_matmul` op in Blaze handles the slicing internally. No weight
splitting is needed during loading.

Dimension summary for both experts:
```
VLM FFN:     gating_einsum [2, 2048, 16384]  ->  linear [16384, 2048]
Action FFN:  gating_einsum [2, 1024, 4096]   ->  linear [4096, 1024]
```

## WeightProvider Pattern for Pi0.5

The existing `WeightProvider` ABC in `blaze/weight_provider.py` has two methods:
`get_mla_torch_weights()` and `get_moe_torch_weights()`. These are specific to
DeepSeek V3. For Pi0.5, we extend the pattern:

```python
class Pi05WeightProvider(WeightProvider):
    """Weight provider for Pi0.5 dual-expert transformer.

    Loads weights from HuggingFace safetensors and transforms them
    into the format expected by blaze-nn's Pi05Module.
    """

    def __init__(self, model_path: Path):
        self._model_path = Path(model_path)
        self._state_dict = None

    def _ensure_state_dict(self):
        if self._state_dict is not None:
            return self._state_dict
        from safetensors.torch import load_file
        sd = {}
        for p in sorted(self._model_path.glob("*.safetensors")):
            sd.update(load_file(p, device="cpu"))
        self._state_dict = sd
        return sd

    def get_block_weights(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Return all weights for a single transformer block."""
        sd = self._ensure_state_dict()
        prefix = f"PaliGemma.llm.params.layers"
        result = {}

        # VLM pre-attention norm
        result["pre_attention_norm.scale"] = (
            sd[f"{prefix}.pre_attention_norm.scale"][layer_idx]
            .to(torch.bfloat16).unsqueeze(0)
        )

        # Action pre-attention adaRMSNorm modulation
        result["pre_attention_norm_1.w_mod"] = (
            sd[f"{prefix}.pre_attention_norm_1.__call__.Dense_0.kernel"][layer_idx]
            .to(torch.bfloat16).T.contiguous()
        )

        # QKV projections (no transpose needed for einsum weights)
        for suffix in ["q_einsum", "kv_einsum", "attn_vec_einsum"]:
            for expert_suffix in ["", "_1"]:
                key = f"{prefix}.attn.{suffix}{expert_suffix}.w"
                if key in sd:
                    result[f"attn.{suffix}{expert_suffix}"] = (
                        sd[key][layer_idx].to(torch.bfloat16)
                    )

        # FFN weights
        for expert_suffix in ["", "_1"]:
            mlp_prefix = f"{prefix}.mlp{expert_suffix}"
            result[f"mlp{expert_suffix}.gating_einsum"] = (
                sd[f"{mlp_prefix}.gating_einsum"][layer_idx].to(torch.bfloat16)
            )
            result[f"mlp{expert_suffix}.linear"] = (
                sd[f"{mlp_prefix}.linear"][layer_idx].to(torch.bfloat16)
            )

        return result

    def get_top_level_weights(self) -> dict[str, torch.Tensor]:
        """Return non-layer weights (embedder, action projections, time MLP)."""
        sd = self._ensure_state_dict()
        return {
            "embedder.input_embedding": sd["PaliGemma.llm.params.embedder.input_embedding"].to(torch.bfloat16),
            "action_in_proj.weight": sd["action_in_proj.kernel"].to(torch.bfloat16).T.contiguous(),
            "time_mlp_in.weight": sd["time_mlp_in.kernel"].to(torch.bfloat16).T.contiguous(),
            "time_mlp_out.weight": sd["time_mlp_out.kernel"].to(torch.bfloat16).T.contiguous(),
            "action_out_proj.weight": sd["action_out_proj.kernel"].to(torch.bfloat16).T.contiguous(),
        }
```

### Loading into blaze-nn Module

```python
# Usage:
provider = Pi05WeightProvider("/path/to/pi05-checkpoint/")

model = Pi05Module(config)

# Load top-level weights
top_weights = provider.get_top_level_weights()
for key, tensor in top_weights.items():
    parts = key.split(".")
    # Navigate module hierarchy and set parameter
    ...

# Load per-layer weights
for layer_idx in range(18):
    block_weights = provider.get_block_weights(layer_idx)
    model.backbone.layers[layer_idx].load_state_dict(block_weights)

# Move to device
model.to(device)
```

## Quantization Considerations

Pi0.5 checkpoints may use mixed precision:

| Component | Training Precision | Inference Precision | Notes |
|-----------|-------------------|---------------------|-------|
| Embedder | float32 | bfloat16 | Large vocab table, cast on load |
| Q/K/V projections | bfloat16 | bfloat16 | No conversion needed |
| adaRMS modulation | bfloat16 | bfloat16 | Small matrix per layer |
| FFN gating einsum | bfloat16 | bfloat16 | Largest per-layer weights |
| FFN down proj | bfloat16 | bfloat16 | |
| RMSNorm scale | float32 | bfloat16 | Cast on load |
| SiGLIP | float32 | bfloat16 | Image encoder |
| Action projections | float32 | bfloat16 | Small matrices |
| Time MLP | float32 | bfloat16 | Small matrices |

The `.to(torch.bfloat16)` call in the weight provider handles all casting. For future
INT8/FP8 quantization on Wormhole:
- Weight-only quantization: store weights in INT8, dequantize in the matmul kernel
- Activation quantization: requires calibration data and per-tensor scale factors
- The `WeightProvider` pattern supports this by returning pre-quantized tensors with
  associated scale factors as additional dict entries

## Weight Size Summary

```
Per-layer weight budget (1 of 18 blocks):

Component                    | Shape              | Size (BF16)
-----------------------------|--------------------|-----------
VLM pre-attn norm scale     | [2048]             |     4 KB
Action pre-attn adaRMS w_mod| [1024, 3072]       | 6,144 KB
VLM Q einsum                | [8, 2048, 256]     | 8,192 KB
VLM KV einsum               | [2, 1, 2048, 256]  | 2,048 KB
VLM O einsum                | [8, 256, 2048]     | 8,192 KB
Action Q einsum              | [8, 1024, 256]     | 4,096 KB
Action KV einsum             | [2, 1, 1024, 256]  | 1,024 KB
Action O einsum              | [8, 256, 1024]     | 4,096 KB
VLM pre-FFN norm scale      | [2048]             |     4 KB
Action pre-FFN adaRMS w_mod | [1024, 3072]       | 6,144 KB
VLM gating einsum            | [2, 2048, 16384]   | 131,072 KB
VLM FFN down                | [16384, 2048]      | 65,536 KB
Action gating einsum         | [2, 1024, 4096]    | 16,384 KB
Action FFN down              | [4096, 1024]       |  8,192 KB
-----------------------------|--------------------|-----------
Per-layer total              |                    | ~261 MB

Total model (18 layers + embedder + projections):
  18 * 261 MB + ~1 GB (embedder) + ~1 MB (projections) ~= 5.7 GB
```

## ASCII Diagram: Weight Loading Pipeline

```
  HF Safetensors (.safetensors files)
       |
       v
  [safetensors.torch.load_file()]
       |
       v
  Raw state_dict (torch.Tensors, may be FP32/FP8/BF16)
       |
       v
  [Pi05WeightProvider]
       |
       +-- .to(torch.bfloat16)     (precision cast)
       +-- .T.contiguous()          (transpose for linear weights)
       +-- .unsqueeze(0)            (reshape for norm scales)
       +-- layer indexing [i]       (extract from scanned dim)
       |
       v
  Transformed weight dict (all BF16, correct shapes)
       |
       v
  [model.load_state_dict()]
       |
       v
  blaze-nn Parameters (torch.Tensors in Module._parameters)
       |
       v
  [model.to(device)]
       |
       v
  [to_device_tensor()] -- ttnn.from_torch() + ttnn.to_device()
       |
       v
  Device tensors (ttnn.Tensor on Wormhole DRAM)
       |
       v
  Ready for inference
```

## Source References

- `blaze/weight_provider.py:85-100` -- WeightProvider ABC with `get_mla_torch_weights`, `get_moe_torch_weights`
- `blaze/weight_provider.py:114-197` -- SyntheticWeightProvider implementation
- `blaze/weight_provider.py:203-279` -- StateDictWeightProvider (HF safetensors)
- `blaze/weight_provider.py:391-479` -- `_state_dict_to_mla_tensors` (transpose/deinterleave examples)
- `blaze/weight_provider.py:482-627` -- `_state_dict_to_moe_tensors` (expert weight handling)
- `blaze/weight_provider.py:634-658` -- `get_weight_provider()` factory
- `blaze_nn/_conversion.py:12-30` -- `to_device_tensor()` (torch -> ttnn)
- `blaze_nn/module.py:187-222` -- `state_dict()` / `load_state_dict()`
- `blaze_nn/module.py:226-252` -- `to(device)` with parameter conversion
- `openpi/models/gemma.py:443-449` -- `_name(name, i)` naming convention
- `openpi/models/gemma.py:253-280` -- FeedForward with packed gating_einsum
- `openpi/models/model.py:243-247` -- `load_pytorch()` with safetensors
