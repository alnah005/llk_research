# 3.2 Shared Attention Mechanism

## Context

The defining architectural choice in pi0.5 is that both experts share a single
attention computation per layer. Rather than running two separate self-attention
modules, the model computes Q/K/V independently for each expert, concatenates
the results along the sequence dimension, applies RoPE and grouped-query
attention (GQA) on the combined sequence, and then splits the output back to
each expert for its own output projection. This section traces the full
attention data-flow from input to output, including the KV cache strategy used
during iterative denoising at inference time.

---

## Per-Expert Q/K/V Computation

Each expert computes its own query, key, and value projections from its private
hidden state. Because `num_kv_heads (1) != num_heads (8)`, the code takes the
GQA path with separate `q_einsum` and `kv_einsum` modules:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 173-199

```python
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is None:
        continue
    # GQA path (num_kv_heads != num_heads)
    q_einsum = lora.Einsum(
        shape=(config.num_heads, config.width, config.head_dim),
        name=_name("q_einsum", i),  # "q_einsum" or "q_einsum_1"
        ...
    )
    q = q_einsum("BTD,NDH->BTNH", x)

    kv_einsum = lora.Einsum(
        shape=(2, config.num_kv_heads, config.width, config.head_dim),
        name=_name("kv_einsum", i),  # "kv_einsum" or "kv_einsum_1"
        ...
    )
    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
    qkvs.append((q, k, v))
```

The weight shapes for each expert:

```
Expert 0 (PaliGemma, width=2048):
    q_einsum.w   : [8, 2048, 256]     # 8 query heads, each projects 2048 -> 256
    kv_einsum.w  : [2, 1, 2048, 256]  # 2 = K+V, 1 KV head, projects 2048 -> 256

Expert 1 (Action, width=1024):
    q_einsum_1.w : [8, 1024, 256]     # 8 query heads, each projects 1024 -> 256
    kv_einsum_1.w: [2, 1, 1024, 256]  # 2 = K+V, 1 KV head, projects 1024 -> 256
```

Despite the different input widths (D=2048 vs D=1024), the output space is
identical: queries are `[B, T, 8, 256]` and keys/values are `[B, T, 1, 256]`.
This uniformity is what makes concatenation along the sequence axis valid.

---

## Concatenation Along the Sequence Dimension

After per-expert projection, the Q/K/V tensors are concatenated:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, line 201

```python
q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))
```

This produces combined tensors where axis 1 (the sequence/time axis) spans both
experts' tokens:

```
Before concatenation:
    Expert 0 q: [B, T_prefix, 8, 256]     Expert 1 q: [B, T_suffix, 8, 256]
    Expert 0 k: [B, T_prefix, 1, 256]     Expert 1 k: [B, T_suffix, 1, 256]
    Expert 0 v: [B, T_prefix, 1, 256]     Expert 1 v: [B, T_suffix, 1, 256]

After concatenation:
    q: [B, T_prefix + T_suffix, 8, 256]
    k: [B, T_prefix + T_suffix, 1, 256]
    v: [B, T_prefix + T_suffix, 1, 256]
```

---

## RoPE Application

Rotary Position Embeddings are applied to the concatenated Q and K tensors:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 203-206

```python
q = _apply_rope(q, positions=positions)
q *= self.configs[0].head_dim ** -0.5  # scale by 1/sqrt(256)
k = _apply_rope(k, positions=positions)
```

The `_apply_rope` function uses a maximum wavelength of 10,000:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 424-440

```python
def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return res.astype(x.dtype)
```

The `positions` tensor is `[B, T_prefix + T_suffix]` and is shared across both
experts. Both experts' tokens live in a single position space, with positions
computed via `cumsum(input_mask) - 1` (see Chapter 3.3). The key detail is that
RoPE is applied *after* concatenation, so each token's position encoding
reflects its global position in the combined prefix+suffix sequence.

The query scaling factor `head_dim ** -0.5 = 1/sqrt(256) = 0.0625` is applied
to queries (not to the attention logits), following the standard convention.

---

## Grouped-Query Attention (GQA)

With 8 query heads and 1 KV head, each KV head serves a group of 8 query heads.
The GQA computation reshapes Q into groups and uses a 5D einsum:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 216-217

```python
q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
```

Axis breakdown:

```
B = batch
T = query length (prefix + suffix tokens, or suffix only during denoising)
S = key/value length (may include cached prefix during denoising)
K = number of KV heads = 1
G = number of query heads per KV head = 8
H = head dimension = 256

q: [B, T, K=1, G=8, H=256]
k: [B, S, K=1, H=256]
  --> logits: [B, K=1, G=8, T, S]
```

The logits are computed in float32 regardless of the model's working precision
(bfloat16), via `preferred_element_type=jnp.float32`. After masking and softmax,
the output is computed and reshaped back:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 228-231

```python
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
```

The attention mask is applied as follows (line 226):

```python
big_neg = -2.3819763e38  # near -inf but avoids float overflow
masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
```

The mask shape is `[B, 1, T, S]` -- the `1` broadcasts across all KV groups.
The `None` inserted at dimension 2 broadcasts across G (the 8 query groups
within each KV head).

---

## Full Attention Data-Flow (ASCII)

```
Expert 0 hidden [B,Tp,2048]    Expert 1 hidden [B,Ts,1024]
        |                               |
   q_einsum [8,2048,256]          q_einsum_1 [8,1024,256]
   kv_einsum [2,1,2048,256]       kv_einsum_1 [2,1,1024,256]
        |                               |
  q0 [B,Tp,8,256]                 q1 [B,Ts,8,256]
  k0 [B,Tp,1,256]                 k1 [B,Ts,1,256]
  v0 [B,Tp,1,256]                 v1 [B,Ts,1,256]
        |                               |
        +------- concat(axis=1) --------+
                      |
              q [B,Tp+Ts,8,256]
              k [B,Tp+Ts,1,256]
              v [B,Tp+Ts,1,256]
                      |
               RoPE(positions)
              q *= 1/sqrt(256)
                      |
         (optional: prepend cached K,V)
                      |
          reshape q -> [B,T,K=1,G=8,H=256]
                      |
         einsum "BTKGH,BSKH->BKGTS"
                      |
              logits [B,1,8,T,S]
                      |
         mask + softmax (float32)
                      |
         einsum "BKGTS,BSKH->BTKGH"
                      |
          reshape -> [B,T,8,256]
                      |
        +------- split(axis=1) ---------+
        |                               |
  enc0 [B,Tp,8,256]              enc1 [B,Ts,8,256]
        |                               |
  attn_vec_einsum [8,256,2048]   attn_vec_einsum_1 [8,256,1024]
        |                               |
  out0 [B,Tp,2048]               out1 [B,Ts,1024]
```

---

## Output Splitting and Per-Expert Output Projection

After the shared attention computation, the encoded tensor is split back into
per-expert slices and each expert applies its own output projection:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 233-248

```python
out = []
start = 0
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is not None:
        end = start + x.shape[1]
        out_einsum = lora.Einsum(
            shape=(config.num_heads, config.head_dim, config.width),
            name=_name("attn_vec_einsum", i),
            ...
        )
        out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
        start = end
    else:
        out.append(None)
```

The split boundary is determined by each expert's sequence length
(`x.shape[1]`). Expert 0's output projection maps `[8, 256] -> 2048`, while
expert 1's maps `[8, 256] -> 1024`. Despite sharing the same attention
computation, each expert projects back to its own width.

---

## KV Cache During Denoising

At inference time, the model performs iterative denoising: the prefix (images +
language) is processed once to fill a KV cache, then only the suffix (action
tokens) is processed repeatedly at each denoising step.

**Step 1: Prefix-only forward pass to build the cache**

**File:** `/tmp/openpi/src/openpi/models/pi0.py`, lines 234-237

```python
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

Note `[prefix_tokens, None]` -- expert 1 is `None`, meaning only expert 0
generates Q/K/V. The returned `kv_cache` contains the prefix K and V tensors
across all 18 layers.

**Step 2: Suffix-only forward pass with cached prefix K/V**

**File:** `/tmp/openpi/src/openpi/models/pi0.py`, lines 261-267

```python
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens],  # only expert 1 is active
    mask=full_attn_mask,
    positions=positions,
    kv_cache=kv_cache,
    adarms_cond=[None, adarms_cond],
)
assert prefix_out is None
```

Now `[None, suffix_tokens]` means only expert 1 generates Q/K/V. Inside the
`Attention` module, the cached prefix K/V is prepended:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 211-214

```python
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    k = jnp.concatenate([cache_k, k], axis=1)
    v = jnp.concatenate([cache_v, v], axis=1)
```

The resulting dimensions during a denoising step:

```
Suffix-only (expert 1 active, expert 0 is None):
    q from expert 1:     [B, T_suffix, 8, 256]
    k from expert 1:     [B, T_suffix, 1, 256]
    v from expert 1:     [B, T_suffix, 1, 256]

After cache prepend:
    k: [B, T_prefix + T_suffix, 1, 256]
    v: [B, T_prefix + T_suffix, 1, 256]

Attention shape:
    q:      [B, T_suffix, 1, 8, 256]     (queries only from suffix)
    k:      [B, T_prefix+T_suffix, 1, 256] (keys from cached prefix + fresh suffix)
    logits: [B, 1, 8, T_suffix, T_prefix+T_suffix]
```

This means suffix tokens can attend to both cached prefix keys/values and their
own fresh keys/values, but the prefix K/V is never recomputed -- it stays frozen
from the initial forward pass. This is the standard prefix-caching optimization
that makes iterative denoising (typically 10 Euler steps) efficient.

---

## PyTorch Implementation

In the PyTorch dual-expert forward path, the same logic is implemented with
standard `nn.Linear` projections instead of custom Einsum modules:

**File:** `/tmp/openpi/src/openpi/models_pytorch/gemma_pytorch.py`, lines 157-194

```python
def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    # Concatenate across sequence dimension for joint attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
```

The PyTorch version uses HuggingFace Transformers' `apply_rotary_pos_emb` and
`eager_attention_forward` from the Gemma modeling module, but the core pattern
is identical: per-expert Q/K/V, concatenate, joint attention, split, per-expert
output projection.

For inference with KV cache, the PyTorch path uses the `past_key_values` idiom
from HuggingFace Transformers:

**File:** `/tmp/openpi/src/openpi/models_pytorch/pi0_pytorch.py`, lines 394-400

```python
_, past_key_values = self.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)
```

---

## Key Takeaways

- Each expert computes Q/K/V independently using its own width (2048 or 1024)
  but projects into the same head space (8 heads x 256 dim for Q, 1 head x 256
  dim for K/V).

- Q/K/V are concatenated along the sequence dimension, not the head dimension.
  This means tokens from both experts participate in a single global attention
  computation.

- RoPE is applied after concatenation on the combined Q and K, using shared
  position IDs and max_wavelength=10,000.

- GQA uses a 5D einsum (`BTKGH,BSKH->BKGTS`) where K=1 KV head serves G=8
  query head groups. Logits are computed in float32.

- During denoising, only the action expert generates fresh Q/K/V. The prefix
  K/V is cached from a one-time forward pass and prepended to the fresh
  suffix K/V at each denoising step.

- The output is split back by sequence position and each expert applies its own
  output projection to map from head space back to its private width.
