# 5.4 PyTorch Implementation Differences

## Context

The PyTorch implementation (`pi0_pytorch.py` and `gemma_pytorch.py`) is a port of the JAX
reference (`pi0.py` and `gemma.py`) that uses HuggingFace Transformers as its backbone. For
Tenstorrent engineers starting from the PyTorch code, this file catalogs every meaningful
difference between the two implementations so you know exactly what is a framework artifact
versus a real behavioral change.

---

## Difference 1: HuggingFace Model Wrappers

The JAX implementation uses custom Gemma modules built with Flax Linen. The PyTorch
implementation wraps HuggingFace's `PaliGemmaForConditionalGeneration` and
`GemmaForCausalLM`.

```
  JAX Architecture:                    PyTorch Architecture:
  ==================                   ====================

  Pi0                                  PI0Pytorch
    |                                    |
    +-- PaliGemma (nnx.Dict)             +-- paligemma_with_expert
    |     +-- llm (custom Module)        |     +-- paligemma (HF PaliGemma)
    |     +-- img (SigLIP Module)        |     |     +-- vision_tower (HF SigLIP)
    |                                    |     |     +-- language_model (HF Gemma)
    +-- action_in_proj                   |     +-- gemma_expert (HF GemmaForCausalLM)
    +-- time_mlp_in                      |
    +-- time_mlp_out                     +-- action_in_proj
    +-- action_out_proj                  +-- time_mlp_in
                                         +-- time_mlp_out
                                         +-- action_out_proj
```

The HuggingFace wrapper introduces an extra level of indirection. The custom JAX Module
handles both experts in a single scan loop. The PyTorch version routes to either
`paligemma.language_model` or `gemma_expert.model` depending on which inputs are non-None.

Source: `gemma_pytorch.py`, lines 11-58 (construction); lines 90-280 (forward routing).

### Three Forward Paths

The PyTorch `PaliGemmaWithExpertModel.forward()` has three distinct code paths:

```python
# gemma_pytorch.py, lines 101-113
if inputs_embeds[1] is None:
    # Path A: prefix-only (PaliGemma language model)
    prefix_output = self.paligemma.language_model.forward(...)

elif inputs_embeds[0] is None:
    # Path B: suffix-only (action expert)
    suffix_output = self.gemma_expert.model.forward(...)

else:
    # Path C: both experts (training only)
    # Manual layer-by-layer loop with fused attention
```

Path C (training) is the most complex: it manually iterates through layers, fusing Q/K/V
across both experts, computing joint attention, then splitting the output back to
per-expert representations. This mirrors the JAX scan-based approach but uses explicit
Python loops.

Source: `gemma_pytorch.py`, lines 125-278.

---

## Difference 2: 4D Float Attention Mask

The JAX implementation uses boolean attention masks:

```python
# gemma.py, line 401
mask = jnp.asarray(mask)[:, None, :, :]   # [b, 1, T, S] bool
```

Then applies the mask inside attention:

```python
# gemma.py, lines 225-226
big_neg = -2.3819763e38
masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
```

The PyTorch implementation converts to float masks upfront:

```python
# pi0_pytorch.py, lines 157-160
def _prepare_attention_masks_4d(self, att_2d_masks):
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)
```

This produces a 4D float tensor where:
- Allowed positions have value **0.0** (added to logits, no effect)
- Masked positions have value **-2.3819763e38** (added to logits, effectively -inf)

The magic number `-2.3819763e38` matches Gemma's original implementation and is close to
but not equal to the minimum float32 value. Both JAX and PyTorch use the same constant.

Source: `gemma.py`, line 225; `pi0_pytorch.py`, line 160.

### Mask Shape Comparison

```
  JAX:     bool  [b, 1, T, S]   -- expanded to [b, K, G, T, S] inside attention
  PyTorch: float [b, 1, T, S]   -- 0.0 or -2.38e38, added directly to logits
```

The float mask approach is standard for HuggingFace Transformers and avoids a `where`
operation inside the attention kernel. For TT-Metal, either approach is viable, but the
float-additive mask may be simpler to fuse with the QK matmul.

---

## Difference 3: Float64 Sinusoidal Embedding

The time embedding computation uses different precision:

**JAX** (`pi0.py`, lines 48-63):
```python
# Computes in default precision (float32)
fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
period = min_period * (max_period / min_period) ** fraction
sinusoid_input = jnp.einsum("i,j->ij", pos, 1.0 / period * 2 * jnp.pi,
                            precision=jax.lax.Precision.HIGHEST)
```

**PyTorch** (`pi0_pytorch.py`, lines 26-42):
```python
# Explicitly uses float64 on GPU, float32 on CPU
dtype = get_safe_dtype(torch.float64, device.type)
fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
period = min_period * (max_period / min_period) ** fraction
scaling_factor = 1.0 / period * 2 * math.pi
sin_input = scaling_factor[None, :] * time[:, None]
```

The `get_safe_dtype` function (`pi0_pytorch.py`, lines 14-22):
```python
def get_safe_dtype(target_dtype, device_type):
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype
```

On GPU, the sinusoidal computation uses float64 for extra precision. On CPU, it stays in
float64 as well (since the fallback only triggers for bfloat16). The JAX version uses
`Precision.HIGHEST` for the einsum but still operates in float32.

**Impact**: The period values span from 4e-3 to 4.0, and the highest-frequency component
has `1/4e-3 * 2*pi = 1570.8`. At float32 precision, the sinusoidal values are accurate
to about 6 decimal places, which is sufficient. The float64 usage in PyTorch is a
conservative choice. For TT-Metal, float32 should be fine.

---

## Difference 4: torch.compile Integration

The PyTorch implementation optionally wraps `sample_actions` with `torch.compile`:

```python
# pi0_pytorch.py, lines 112-113
if config.pytorch_compile_mode is not None:
    self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)
```

Default compile mode is `"max-autotune"` (`pi0_config.py`, line 35).

Available modes from `pi0_config.py`, lines 43-48:
```python
assert self.pytorch_compile_mode in [
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
```

Additionally, the constructor sets matmul precision:

```python
# pi0_pytorch.py, line 111
torch.set_float32_matmul_precision("high")
```

This enables TensorFloat-32 (TF32) on NVIDIA GPUs, using 19-bit precision for float32
matmuls instead of full 23-bit mantissa.

**TT-Metal relevance**: Neither `torch.compile` nor `set_float32_matmul_precision` applies
to Tenstorrent hardware. These are GPU-specific optimizations. However, the precision
choice signals that the model tolerates reduced matmul precision, which is useful
information for choosing TT-Metal data formats.

---

## Difference 5: Attention Implementation Selection

The PyTorch code explicitly forces `"eager"` attention during inference:

```python
# pi0_pytorch.py, line 392
self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

# pi0_pytorch.py, line 448
self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
```

HuggingFace Transformers supports multiple attention backends:
- `"eager"`: standard PyTorch attention (explicit Q@K, softmax, @V)
- `"sdpa"`: torch.nn.functional.scaled_dot_product_attention (may use Flash Attention)
- `"flash_attention_2"`: FlashAttention-2

The `"eager"` mode is chosen for compatibility -- the KV cache pass-through with mixed
experts does not work with Flash Attention's fused kernel. For TT-Metal, you will
implement attention natively so this setting is irrelevant.

---

## Difference 6: Weight Format and Naming

The JAX model uses Flax Linen conventions for weight naming. The PyTorch model uses
HuggingFace conventions. This matters when loading pretrained weights.

### Expert Naming Convention

JAX (`gemma.py`, lines 443-450):
```python
def _name(name, i):
    # Expert 0: "attn", "mlp", etc.     (no suffix)
    # Expert 1: "attn_1", "mlp_1", etc. (suffix "_1")
    if i == 0:
        return name
    return f"{name}_{i}"
```

PyTorch (`gemma_pytorch.py`, lines 56-58):
```python
self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
self.gemma_expert.model.embed_tokens = None  # Not needed, actions are continuous
```

The PyTorch version uses completely separate model instances rather than shared layers with
suffixed names. Weight conversion between formats must account for this structural
difference.

### Precision Handling

The PyTorch model has explicit dtype management:

```python
# gemma_pytorch.py, lines 62-82
def to_bfloat16_for_selected_params(self, precision):
    if precision == "bfloat16":
        self.to(dtype=torch.bfloat16)
    # Keep certain params in float32:
    params_to_keep_float32 = [
        "vision_tower.vision_model.embeddings.patch_embedding.weight",
        "vision_tower.vision_model.embeddings.patch_embedding.bias",
        "vision_tower.vision_model.embeddings.position_embedding.weight",
        "input_layernorm",
        "post_attention_layernorm",
        "model.norm",
    ]
```

Vision embedding and normalization parameters are kept in float32 even when the rest of
the model is in bfloat16. This is important for numerical stability in SigLIP's patch
embedding and the RMSNorm/adaRMSNorm layers.

---

## Difference 7: Activation Function Naming

Both implementations use the same activation functions but with different names:

| Operation | JAX (pi0.py) | PyTorch (pi0_pytorch.py) |
|-----------|-------------|--------------------------|
| Time MLP activation | `nnx.swish` (line 165-167) | `F.silu` (line 283-294) |
| FFN gating | `nn.gelu` (gemma.py line 268) | `gelu_pytorch_tanh` (config) |

`swish` and `silu` are the same function: `x * sigmoid(x)`. The naming differs by
framework convention.

The FFN activation is more subtle: JAX uses `nn.gelu` (exact GeLU), while PyTorch uses
`gelu_pytorch_tanh` (the tanh approximation of GeLU). The tanh approximation is:
```
  gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

This is a minor numerical difference that should not affect model quality.

---

## Difference 8: Loop Structure (JAX vs Python)

The JAX implementation uses `jax.lax.while_loop` for the denoising loop:

```python
# pi0.py, line 278
x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
```

This is a JAX primitive that gets compiled into XLA. The loop body and condition are traced
once and executed as a single fused operation. This has implications:
- The loop count must be determined at runtime (it cannot be unrolled at trace time)
- All tensors in the carry must have static shapes
- Side effects inside the loop are not supported

The PyTorch version uses a standard Python `while` loop:

```python
# pi0_pytorch.py, lines 407-419
while time >= -dt / 2:
    v_t = self.denoise_step(...)
    x_t = x_t + dt * v_t
    time += dt
```

This is a normal Python loop that executes 10 iterations eagerly (or gets compiled by
`torch.compile` if enabled). The loop overhead is minimal since each iteration involves a
full transformer forward pass.

**TT-Metal implication**: The Python loop is the natural starting point. Each iteration
dispatches a sequence of kernel calls. There is no need for a compiled loop primitive.

---

## Difference 9: Denoise Step Extraction

The PyTorch implementation extracts the denoising step into a separate method:

```python
# pi0_pytorch.py, lines 422-462
def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
    """Apply one denoising step of the noise `x_t` at a given timestep."""
    ...
```

The JAX version keeps everything inline within the `step` closure:

```python
# pi0.py, lines 239-271
def step(carry):
    x_t, time = carry
    ...
    return x_t + dt * v_t, time + dt
```

This is a code organization difference with no functional impact. However, the PyTorch
`denoise_step` method provides a cleaner API for testing and profiling individual steps.

---

## Difference 10: Gradient Checkpointing (Training Only)

The PyTorch implementation includes extensive gradient checkpointing support for training:

```python
# pi0_pytorch.py, lines 127-155
def gradient_checkpointing_enable(self):
    self.gradient_checkpointing_enabled = True
    self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
    self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
    self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
```

The JAX version uses `nn.remat` (rematerialization) at the block level:

```python
# gemma.py, lines 359-364
block_cls = nn.remat(
    Block,
    prevent_cse=False,
    static_argnums=(5,),
    policy=jax.checkpoint_policies.nothing_saveable,
)
```

These are equivalent concepts (recompute activations during backward pass to save memory)
but with different APIs. Neither is relevant to inference on TT hardware.

---

## Difference 11: SigLIP Integration

The JAX version uses a custom SigLIP implementation:

```python
# pi0.py, lines 81-90
img = nnx_bridge.ToNNX(_siglip.Module(
    num_classes=paligemma_config.width,
    variant="So400m/14",
    pool_type="none",
    scan=True,
    dtype_mm=config.dtype,
))
```

The PyTorch version uses HuggingFace's built-in SigLIP through PaliGemma:

```python
# gemma_pytorch.py, line 84
def embed_image(self, image: torch.Tensor):
    return self.paligemma.model.get_image_features(image)
```

Both produce the same 256 tokens per image with embedding dimension matching PaliGemma's
width (2048). The PyTorch version benefits from HuggingFace's tested implementation but
adds the dependency on the `transformers` library.

Note: The PyTorch code requires a patched version of HuggingFace Transformers:

```python
# pi0_pytorch.py, lines 118-125
msg = "transformers_replace is not installed correctly. Please install it with ..."
try:
    from transformers.models.siglip import check
    if not check.check_whether_transformers_replace_is_installed_correctly():
        raise ValueError(msg)
except ImportError:
    raise ValueError(msg) from None
```

---

## Summary Table

```
  +--------------------------------+---------------------------+---------------------------+
  | Aspect                         | JAX (pi0.py/gemma.py)     | PyTorch (pi0_pytorch.py)  |
  +--------------------------------+---------------------------+---------------------------+
  | Model backbone                 | Custom Flax Linen         | HuggingFace Transformers  |
  | Attention mask format          | bool, jnp.where inside    | float, 0/-2.38e38 added   |
  | Sinusoidal embedding precision | float32 + HIGHEST einsum  | float64 on GPU            |
  | Compilation                    | XLA (always)              | torch.compile (optional)  |
  | Matmul precision               | bfloat16 default          | TF32 (set_float32_matmul) |
  | Attention backend              | Custom GQA                | HF eager (forced)         |
  | Expert structure               | Shared scan, suffix names | Separate model instances  |
  | Norm precision                 | Cast to float32 in norm   | Keep norm params float32  |
  | Activation: time MLP           | swish                     | silu (identical)          |
  | Activation: FFN gate           | gelu (exact)              | gelu_pytorch_tanh (approx)|
  | Denoising loop                 | jax.lax.while_loop        | Python while loop         |
  | Gradient checkpointing         | nn.remat (block-level)    | torch.utils.checkpoint    |
  | SigLIP                         | Custom implementation     | HF PaliGemma built-in     |
  | KV cache format                | (K_all, V_all) stacked    | List[(K,V)] per layer     |
  +--------------------------------+---------------------------+---------------------------+
```

---

## Key Takeaways

1. **The PyTorch code wraps HuggingFace Transformers**, adding a layer of indirection that
   a TT-Metal port can skip entirely. The core math is identical -- only the plumbing
   differs.

2. **Attention masks are float-valued in PyTorch (0.0 / -2.38e38)** versus boolean in JAX.
   The float approach is additive and may be simpler to fuse with QK matmul on TT hardware.

3. **Sinusoidal embeddings use float64 in PyTorch** for extra precision, but float32 is
   sufficient given the period range [4e-3, 4.0]. TT-Metal can safely use float32 or
   bfloat16.

4. **torch.compile and TF32 precision are GPU-specific optimizations** that do not apply
   to Tenstorrent hardware but signal that reduced-precision matmuls are acceptable.

5. **Normalization and embedding parameters are kept in float32 even in bf16 mode**. This
   precision split must be preserved in any TT-Metal implementation to maintain numerical
   stability.
