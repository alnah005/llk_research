# 3.4 Feed-Forward Network and Gated Residual Connections

## Context

Each transformer layer in pi0.5 follows the same pattern: normalization, shared
attention, gated residual, normalization, per-expert feed-forward, gated
residual. The FFN uses GeGLU (GELU-gated linear unit), not SwiGLU -- a detail
that matters for hardware implementation because the activation function choice
affects numerical behavior and kernel selection. The gated residual connection,
introduced for the pi0.5 variant's adaptive RMSNorm (adaRMS), modulates the
residual addition with a learned gate derived from the diffusion timestep. This
section traces the complete block data-flow and the 18-layer scan structure.

---

## GeGLU Feed-Forward Network

The FFN is defined in the `FeedForward` class (JAX) and `GemmaMLP` (PyTorch).
Both use GELU, not SwiGLU (which would use SiLU/Swish):

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 253-280

```python
class FeedForward(nn.Module):
    features: int       # input/output width (2048 or 1024)
    hidden_dim: int     # intermediate dimension (16384 or 4096)

    @nn.compact
    def __call__(self, x):
        w_gating = self.param(
            "gating_einsum", ...,
            (2, self.features, self.hidden_dim),  # two projections packed together
        )
        ff_gate = jnp.dot(x, w_gating[0])     # gate projection
        gate_value = nn.gelu(ff_gate)          # GELU activation (NOT SiLU/Swish)

        ff1 = jnp.dot(x, w_gating[1])         # up projection
        activations = gate_value * ff1         # element-wise gating

        w_linear = self.param("linear", ...,
            (self.hidden_dim, self.features),   # down projection
        )
        outputs = jnp.dot(activations, w_linear)
        return outputs
```

**File:** `/tmp/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`, lines 113-126

```python
class GemmaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # "gelu_pytorch_tanh"

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

The PyTorch config specifies `hidden_activation="gelu_pytorch_tanh"`:

**File:** `/tmp/openpi/src/openpi/models_pytorch/gemma_pytorch.py`, line 32

```python
vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
```

The GeGLU data-flow:

```
Input x [B, T, D]
     |
     +-- gate_proj [D, MLP_DIM] --> GELU --> gate [B, T, MLP_DIM]
     |                                         |
     +-- up_proj [D, MLP_DIM] -------> up [B, T, MLP_DIM]
                                               |
                                        gate * up  (element-wise)
                                               |
                                   down_proj [MLP_DIM, D]
                                               |
                                   output [B, T, D]

Expert 0: D=2048, MLP_DIM=16384   (gating_einsum: [2, 2048, 16384])
Expert 1: D=1024, MLP_DIM=4096    (gating_einsum: [2, 1024, 4096])
```

The JAX implementation packs both projections (gate and up) into a single
`gating_einsum` parameter of shape `[2, features, hidden_dim]`, where index 0
is the gate projection and index 1 is the up projection. The PyTorch version
uses separate `gate_proj` and `up_proj` Linear layers.

---

## GeGLU vs SwiGLU Distinction

This is an important implementation detail:

| Property    | GeGLU (used in pi0.5/Gemma) | SwiGLU (used in Llama) |
|-------------|----------------------------|----------------------|
| Activation  | GELU                       | SiLU (Swish)         |
| Formula     | `GELU(Wx) * Vx`           | `SiLU(Wx) * Vx`     |
| JAX call    | `nn.gelu(ff_gate)`         | `nn.silu(ff_gate)`   |
| PyTorch     | `gelu_pytorch_tanh`        | `silu`               |
| Derivative  | Smooth, Gaussian-based     | Smooth, sigmoid-based|

The GELU variant used here is the tanh approximation:
`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

For Tenstorrent hardware implementation, this means GELU (not SiLU) LLK kernels
are needed in the FFN path.

---

## The `_gated_residual` Function

The gated residual connection is central to the pi0.5 architecture. It appears
twice per block -- once after attention and once after FFN:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 453-459

```python
def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y       # standard residual (when no adaRMS)
    return x + y * gate    # gated residual (when adaRMS provides a gate)
```

**File:** `/tmp/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`, lines 209-227

```python
def _gated_residual(x, y, gate):
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate
```

The `gate` parameter comes from the adaptive RMSNorm. When `adarms_cond` is
`None` (expert 0, which never uses adaRMS), the gate is `None` and the function
reduces to a standard residual `x + y`. When `adarms_cond` is provided (expert
1 with timestep conditioning), the gate is a tensor of the same shape as `y`,
element-wise modulating how much of the sub-layer output is added to the
residual.

---

## Adaptive RMSNorm (adaRMS)

The normalization layer produces scale, shift, and gate when conditioned on the
diffusion timestep:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 113-131

```python
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = x * jnp.reciprocal(jnp.sqrt(var + 1e-06))

        if cond is None:
            # regular RMSNorm: scale = learned parameter
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None  # gate = None

        # adaptive RMSNorm: project cond to 3x width -> split into scale, shift, gate
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate
```

For expert 0 (PaliGemma), `cond=None` always, so regular RMSNorm is used and
gate is `None`. For expert 1 (action expert), `cond` is the processed timestep
embedding, and the norm produces a `gate` tensor that flows into
`_gated_residual`.

The modulation projection is initialized to zeros, meaning at the start of
training `scale=0`, `shift=0`, `gate=0`, and the adaRMS layer acts exactly like
a standard RMSNorm with no residual contribution from the sub-layer (since
`gate * y = 0`). This is a common initialization strategy for conditional
normalization layers -- it ensures the model starts from a well-behaved state.

---

## Complete Block Data-Flow

Each of the 18 layers follows this exact sequence:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 293-333 (`Block.__call__`)

```
Input: xs = [expert_0_hidden, expert_1_hidden]   (or None for inactive expert)

    Step 1: Pre-attention normalization (per-expert)
    ================================================
    for each expert i:
        normed_i, gate_i = RMSNorm(name=f"pre_attention_norm{'_'+i if i>0}")(xs[i], adarms_cond[i])

    Step 2: Shared attention (joint across experts)
    ================================================
    post_attn, kv_cache = Attention(configs)(normed_list, positions, attn_mask, kv_cache)

    Step 3: First gated residual (per-expert)
    ================================================
    for each expert i:
        xs[i] = _gated_residual(xs[i], post_attn[i], gate_i)

    Step 4: Pre-FFN normalization (per-expert)
    ================================================
    for each expert i:
        normed_i, gate_i = RMSNorm(name=f"pre_ffw_norm{'_'+i if i>0}")(xs[i], adarms_cond[i])

    Step 5: Per-expert feed-forward network
    ================================================
    for each expert i:
        ffn_out_i = FeedForward(features=config_i.width, hidden_dim=config_i.mlp_dim,
                                name=f"mlp{'_'+i if i>0}")(normed_i)

    Step 6: Second gated residual (per-expert)
    ================================================
    for each expert i:
        xs[i] = _gated_residual(xs[i], ffn_out_i, gate_i)

Output: xs = [expert_0_hidden, expert_1_hidden], kv_cache
```

---

## Block Data-Flow (ASCII Diagram)

```
  Expert 0 [B, Tp, 2048]              Expert 1 [B, Ts, 1024]
       |                                    |
  pre_attention_norm                   pre_attention_norm_1
  (RMSNorm, cond=None)                (adaRMS, cond=timestep)
  gate0 = None                         gate1 = [B, 1, 1024]
       |                                    |
       +------->  Attention  <--------------+
       |         (shared, joint)            |
       |    concat Q/K/V along seq dim      |
       |    RoPE + GQA + softmax            |
       |    split output by seq dim         |
       |                                    |
  post_attn0 [B, Tp, 2048]           post_attn1 [B, Ts, 1024]
       |                                    |
  _gated_residual                      _gated_residual
  x + y  (gate=None)                   x + y * gate1
       |                                    |
  pre_ffw_norm                         pre_ffw_norm_1
  (RMSNorm, cond=None)                (adaRMS, cond=timestep)
  gate0 = None                         gate1 = [B, 1, 1024]
       |                                    |
  FeedForward                          FeedForward
  mlp [2,2048,16384]                   mlp_1 [2,1024,4096]
  + linear [16384,2048]                + linear [4096,1024]
  GeGLU activation                     GeGLU activation
       |                                    |
  ffn_out0 [B, Tp, 2048]             ffn_out1 [B, Ts, 1024]
       |                                    |
  _gated_residual                      _gated_residual
  x + y  (gate=None)                   x + y * gate1
       |                                    |
  Expert 0 [B, Tp, 2048]              Expert 1 [B, Ts, 1024]
```

---

## 18-Layer Scan and Final Norms

The 18 identical blocks are wrapped in Flax's `nn.scan`, which stacks the
parameters along a leading axis and executes the blocks as a vectorized loop:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 359-381

```python
block_cls = nn.remat(Block, ...)  # gradient checkpointing per layer
self.layers = nn.scan(
    block_cls,
    variable_axes={"params": 0},     # params stacked along axis 0
    split_rngs={"params": True, "dropout": True},
    in_axes=(
        0,           # kv_cache: per-layer
        nn.broadcast, # positions: shared
        nn.broadcast, # mask: shared
        nn.broadcast, # adarms_cond: shared
        nn.broadcast, # deterministic: shared
    ),
    length=self.configs[0].depth,  # 18
)(configs=self.configs, ...)
```

After the 18 scanned layers, per-expert final norms are applied:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 382, 409-411

```python
self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

# In __call__:
return [
    f(e, a)[0] if e is not None else e
    for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
], kv_cache
```

Expert 0 gets `final_norm` (regular RMSNorm), expert 1 gets `final_norm_1`
(adaRMS in pi0.5 mode). The gate output from the final norm is discarded
(index `[0]`), since there is no subsequent residual connection.

In the PyTorch implementation, the same final norm appears in:

**File:** `/tmp/openpi/src/openpi/models_pytorch/gemma_pytorch.py`, lines 261-274

```python
def compute_final_norms(inputs_embeds, adarms_cond):
    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
        outputs_embeds.append(out_emb)
    return outputs_embeds
```

---

## End-to-End Module Flow

```
Input tokens / action embeddings
         |
    Embedder (expert 0 only, vocab=257152, dim=2048)
    action_in_proj (expert 1 only, action_dim -> 1024)
         |
    +----+----+
    |         |
  Expert 0  Expert 1
  [B,Tp,2048] [B,Ts,1024]
    |         |
    +----+----+
         |
    +---------+
    | Layer 0 |  <-- Block: norm -> shared attn -> gated res -> norm -> FFN -> gated res
    +---------+
    | Layer 1 |
    +---------+
    |  ...    |
    +---------+
    | Layer 17|
    +---------+
         |
    +----+----+
    |         |
  final_norm  final_norm_1
  (RMSNorm)   (adaRMS)
    |         |
  Expert 0  Expert 1
  output    output
    |         |
  (unused)  action_out_proj -> v_t (velocity prediction)
```

---

## PyTorch Gradient Checkpointing

The PyTorch implementation wraps the entire per-layer computation (including
the dual-expert attention loop) in `torch.utils.checkpoint.checkpoint`:

**File:** `/tmp/openpi/src/openpi/models_pytorch/gemma_pytorch.py`, lines 240-255

```python
for layer_idx in range(num_layers):
    if use_gradient_checkpointing:
        inputs_embeds = torch.utils.checkpoint.checkpoint(
            compute_layer_complete,
            layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    else:
        inputs_embeds = compute_layer_complete(
            layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
        )
```

This mirrors the JAX `nn.remat(Block, ...)` wrapping in the JAX version and is
essential for fitting the dual-expert model in GPU memory during training.

---

## Key Takeaways

- The FFN uses GeGLU (GELU gating), not SwiGLU. The activation is
  `gelu_pytorch_tanh` in PyTorch and `nn.gelu` in JAX. Both the gate and up
  projections are packed into a single `gating_einsum` parameter in JAX.

- `_gated_residual(x, y, gate)` is the core residual function. When `gate` is
  `None` (expert 0, no adaRMS), it computes `x + y`. When `gate` is a tensor
  (expert 1, adaRMS active), it computes `x + y * gate`.

- Adaptive RMSNorm produces a 3-way split: `(scale, shift, gate)` from the
  timestep conditioning. The projection is initialized to zeros, so training
  starts from standard RMSNorm behavior.

- Each block applies two RMSNorms (pre-attention and pre-FFN) and two gated
  residuals. The attention is shared; everything else is per-expert.

- The 18 layers are scanned (JAX) or iterated with gradient checkpointing
  (PyTorch). After the final layer, per-expert final norms produce the output
  representations.
