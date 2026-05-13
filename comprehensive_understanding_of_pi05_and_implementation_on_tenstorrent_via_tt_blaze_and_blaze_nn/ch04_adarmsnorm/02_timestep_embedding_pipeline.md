# 02 -- Timestep Embedding Pipeline

## Context

The diffusion denoising process in pi0.5 operates over a scalar timestep `t`
that indicates how much noise remains in the action trajectory. This scalar must
be converted into a rich vector representation before it can condition the
transformer's normalization layers. The pipeline has two stages: a sinusoidal
positional encoding that maps the scalar to a 1024-dimensional vector, followed
by a two-layer MLP with SiLU activations that refines this representation into
the final conditioning signal `adarms_cond`. The resulting `[B, 1024]` vector is
then broadcast identically across all action tokens and all 18 transformer
layers -- it is not per-token and not per-layer. This section traces the full
pipeline, compares it to the pi0 (non-pi0.5) variant, and examines the broadcast
semantics that make this design both simple and hardware-friendly.

---

## 1. Sinusoidal Positional Encoding

The timestep is first encoded via `posemb_sincos` at `pi0.py:48-63`:

```
def posemb_sincos(
    pos: at.Real[at.Array, " b"],
    embedding_dim: int,
    min_period: float,
    max_period: float,
) -> at.Float[at.Array, "b {embedding_dim}"]:
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
```

Called at `pi0.py:161`:

```
time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
```

Where `self.action_in_proj.out_features` equals `action_expert_config.width = 1024`.

### Frequency computation

The function generates 512 frequency bands (`embedding_dim // 2 = 512`):

```
fraction = linspace(0.0, 1.0, 512)         # 512 evenly spaced values in [0, 1]
period   = 0.004 * (4.0 / 0.004) ^ fraction # geometric spacing from 4e-3 to 4.0
freq     = (1 / period) * 2 * pi            # angular frequencies
```

The period range `[4e-3, 4.0]` is chosen for sensitivity in the denoising
timestep range `[0.001, 1.0]` (see `pi0.py:197`: `time = beta(1.5, 1) * 0.999 + 0.001`).

```
  Period spectrum (log scale):
  
  dim index:  0         128       256       384       511
              |----------|---------|---------|---------|
  period:   4e-3                                     4.0
              |                                       |
         high frequency                        low frequency
         (resolves fine                    (resolves coarse
          timestep diffs)                   timestep diffs)
```

The output concatenates `[sin(input), cos(input)]` to produce the final
`[B, 1024]` embedding.

### Contrast with standard transformer positional encoding

Standard transformer positional encodings use `min_period = 1` and
`max_period = 10000`. The pi0.5 timestep encoding uses a much narrower range
(`4e-3` to `4.0`) because it encodes a continuous scalar in `[0, 1]` rather than
integer token positions in `[0, seq_len]`.

## 2. Time MLP (pi0.5 path)

After sinusoidal encoding, the embedding passes through a two-layer MLP.
Source: `pi0.py:163-169`.

```
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
```

The layers are defined at `pi0.py:94-95`:

```
self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
```

The full pipeline:

```
  timestep [B]               (scalar per batch element)
       |
  posemb_sincos(t, 1024, 4e-3, 4.0)
       |
  time_emb [B, 1024]         (sinusoidal encoding)
       |
  time_mlp_in: Linear(1024, 1024)
       |
  SiLU (aka swish)
       |
  [B, 1024]
       |
  time_mlp_out: Linear(1024, 1024)
       |
  SiLU (aka swish)
       |
  adarms_cond [B, 1024]      (final conditioning vector)
```

Note: `nnx.swish` is SiLU (`x * sigmoid(x)`), the same activation. Both names
are used interchangeably in JAX/Flax.

### Parameter count for time MLP

| Layer | Shape | Parameters |
|-------|-------|------------|
| `time_mlp_in.kernel` | [1024, 1024] | 1,048,576 |
| `time_mlp_in.bias` | [1024] | 1,024 |
| `time_mlp_out.kernel` | [1024, 1024] | 1,048,576 |
| `time_mlp_out.bias` | [1024] | 1,024 |
| **Total** | | **2,099,200** |

## 3. Contrast: pi0 (non-pi0.5) Timestep Path

When `self.pi05 = False`, the model takes a fundamentally different approach.
Source: `pi0.py:96-99` (layer definitions) and `pi0.py:171-178` (forward path).

```
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
```

```
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
```

| Aspect | pi0 (non-pi0.5) | pi0.5 |
|--------|-----------------|-------|
| Timestep injection | Concatenated with action tokens | Via adaRMSNorm conditioning |
| MLP input dim | 2048 (action + time concat) | 1024 (time only) |
| MLP output | Replaces action tokens | Produces `adarms_cond` |
| State token | Separate `state_proj` token prepended | No state token |
| `adarms_cond` | `None` (no adaptive norm) | `[B, 1024]` |
| Norm type | Standard RMSNorm for both experts | Standard for expert 0, adaptive for expert 1 |

The pi0 approach bakes timestep information directly into the token
representations before they enter the transformer. The pi0.5 approach keeps the
token representations clean and instead modulates the normalization at every
layer. The pi0.5 design eliminates the state token and avoids the 2x wider MLP
input, reducing computation while enabling more expressive per-layer conditioning.

## 4. Broadcast Semantics of `adarms_cond`

A critical design property: `adarms_cond` is NOT per-token. It is a single
vector per batch element.

```
  adarms_cond shape: [B, 1024]

  In each adaRMSNorm instance:
    modulation = Dense(cond)          # cond is [B, 1024] -> [B, 3072]
    modulation = modulation[:, None, :]  # -> [B, 1, 3072]
    scale, shift, gate = split3(...)  # each [B, 1, 1024]
```

The `[:, None, :]` unsqueeze at `gemma.py:129` inserts a length-1 sequence
dimension. When multiplied with `normed_inputs` of shape `[B, T, 1024]`, NumPy
broadcasting replicates the modulation across all `T` tokens:

```
  normed_inputs [B, T, 1024]
                  *
  (1 + scale)   [B, 1, 1024]   <-- broadcast across T
                  +
  shift         [B, 1, 1024]   <-- broadcast across T
```

This means every action token at every position receives the same scale, shift,
and gate from the timestep. The modulation is purely a function of "how noisy is
the current sample" -- it does not vary by token position.

### Layer broadcast

The `adarms_cond` list `[None, adarms_cond]` is passed to the `Module.__call__`
at `gemma.py:395`, which forwards it through the scanned layers at
`gemma.py:405`:

```
embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)
```

The scan configuration at `gemma.py:373` marks `adarms_cond` as `nn.broadcast`
(the 4th in_axes entry), meaning the same `adarms_cond` is passed to every layer
without modification.

```
  Layer 0:  adarms_cond [B, 1024]  ---> pre_attn_norm_1, pre_ffw_norm_1
  Layer 1:  adarms_cond [B, 1024]  ---> pre_attn_norm_1, pre_ffw_norm_1
  ...
  Layer 17: adarms_cond [B, 1024]  ---> pre_attn_norm_1, pre_ffw_norm_1
  Final:    adarms_cond [B, 1024]  ---> final_norm_1
```

However, each layer's `Dense` projection has independent weights (created via
`nn.compact` within each `RMSNorm` instance, with scanned parameters on axis 0
per `gemma.py:366: variable_axes={"params": 0}`). So while the input
conditioning vector is the same, the resulting scale/shift/gate differ per layer
because the projection weights differ.

## 5. The `adarms_cond` List Pattern

The conditioning is always passed as a Python list matching the expert count:

```python
adarms_cond = [None, adarms_cond]   # [expert_0_cond, expert_1_cond]
```

At `gemma.py:302-303` inside `Block.__call__`:

```
for i, x in enumerate(xs):
    if x is not None:
        x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])
```

- `i=0`: `adarms_cond[0] = None` --> standard RMSNorm, gate = None
- `i=1`: `adarms_cond[1] = adarms_cond` --> adaptive RMSNorm, gate = [B,1,D]

This indexing pattern is used at every normalization site in the block and in the
final norm.

## 6. End-to-End Data Flow

```
  User provides: scalar timestep t in [0.001, 1.0]
       |
       |  posemb_sincos(t, 1024, min_period=4e-3, max_period=4.0)
       |
       v
  time_emb [B, 1024]  (sinusoidal, captures multi-scale frequency info)
       |
       |  time_mlp_in: Linear(1024 -> 1024)
       |  SiLU
       |  time_mlp_out: Linear(1024 -> 1024)
       |  SiLU
       |
       v
  adarms_cond [B, 1024]
       |
       |  packaged as [None, adarms_cond]
       |
       v
  Module.__call__  --->  scan over 18 layers (broadcast)
       |
       +---> Block layer_k:
       |       |
       |       +-- pre_attention_norm_1(x, adarms_cond)
       |       |     |-> Dense_k_attn(adarms_cond) -> scale, shift, gate
       |       |     |-> normed * (1 + scale) + shift
       |       |     |-> gate -> _gated_residual
       |       |
       |       +-- pre_ffw_norm_1(x, adarms_cond)
       |             |-> Dense_k_ffw(adarms_cond) -> scale, shift, gate
       |             |-> normed * (1 + scale) + shift
       |             |-> gate -> _gated_residual
       |
       +---> final_norm_1(x, adarms_cond)
                |-> Dense_final(adarms_cond) -> scale, shift, gate
                |-> normed * (1 + scale) + shift
                |-> gate discarded (no residual after final norm)
```

---

## Key Takeaways

- The timestep encoding uses `posemb_sincos` with a frequency range tuned for
  the `[0.001, 1.0]` denoising interval: `min_period=4e-3`, `max_period=4.0`,
  producing a `[B, 1024]` vector from 512 sin/cos frequency bands.

- The time MLP is a simple two-layer network (`Linear -> SiLU -> Linear -> SiLU`)
  that maps `[B, 1024] -> [B, 1024]`, adding ~2.1M parameters.

- `adarms_cond` is NOT per-token: it is `[B, 1024]`, broadcast across all action
  tokens via unsqueeze. Every token at every position receives the same
  timestep modulation.

- `adarms_cond` is NOT per-layer: the same vector is passed (via `nn.broadcast`
  in the scan) to all 18 layers. Per-layer variation comes from each layer
  having its own independent `Dense` projection weights.

- Pi0 (non-pi0.5) takes a fundamentally different approach: it concatenates
  timestep embeddings with action tokens and processes them through a wider MLP,
  injecting time information into the token stream rather than into normalization.

- The list pattern `[None, adarms_cond]` cleanly separates expert conditioning:
  expert 0 always gets `None` (standard RMSNorm), expert 1 gets the timestep
  conditioning (adaptive RMSNorm).
