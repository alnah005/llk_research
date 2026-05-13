# 5.3 KV Cache Strategy

## Context

The KV cache is the mechanism that makes the 10-step denoising loop efficient. Without it,
every denoising step would need to reprocess all 968 prefix tokens through 18 transformer
layers -- a 20x overhead per step. This file details the exact cache structure, how it is
populated, how it is consumed, and the subtleties around the dual-expert architecture that
Tenstorrent engineers must understand to implement this correctly in TT-Metal.

---

## Cache Architecture Overview

```
  +-----------------------------------------------------------------+
  |                        KV Cache Structure                        |
  +-----------------------------------------------------------------+
  |                                                                  |
  |  Layer 0:  K [b, 968, 1, 256]    V [b, 968, 1, 256]            |
  |  Layer 1:  K [b, 968, 1, 256]    V [b, 968, 1, 256]            |
  |  Layer 2:  K [b, 968, 1, 256]    V [b, 968, 1, 256]            |
  |  ...                                                             |
  |  Layer 17: K [b, 968, 1, 256]    V [b, 968, 1, 256]            |
  |                                                                  |
  |  Total: 18 layers x 2 tensors x [b, 968, 1, 256]               |
  |  Per-batch memory: 18 * 2 * 968 * 1 * 256 * 2 bytes (bf16)     |
  |                  = 17.9 MB per batch element                    |
  +-----------------------------------------------------------------+
```

### Dimension Breakdown

| Axis | Size | Meaning |
|------|------|---------|
| layers | 18 | One K,V pair per transformer layer (depth of both Gemma-2B and Gemma-300M) |
| batch | b | Batch dimension |
| seq_len | 968 | Number of prefix tokens (3 * 256 image + 200 language) |
| num_kv_heads | 1 | Grouped Query Attention: 1 KV head shared by 8 query heads |
| head_dim | 256 | Per-head dimension |

Source: `gemma.py`, lines 69-87 (config definitions); lines 211-214 (cache concatenation).

---

## Cache Lifecycle: Three Phases

### Phase 1: Cache Population (prefix forward pass)

During the prefix-only forward pass, each transformer layer computes K and V for the 968
prefix tokens and stores them in the cache.

```python
# pi0.py, lines 234-237
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

Inside the transformer, at each layer the Attention module produces (k, v) and returns
them as the cache:

```python
# gemma.py, lines 211-214
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    k = jnp.concatenate([cache_k, k], axis=1)
    v = jnp.concatenate([cache_v, v], axis=1)
```

During the prefix pass, `kv_cache` is None so no concatenation happens. The output K, V
tensors from each layer become the cache.

Source: `gemma.py`, lines 248-249.

### Phase 2: Cache Consumption (denoising loop -- 10x)

During each denoising step, the suffix tokens generate new Q, K, V tensors. The new K, V
are concatenated with the cached prefix K, V to form the full key-value sequence:

```
  Per-layer attention during denoising:

  Suffix Q:     [b, 50, 8, 256]      (50 suffix query positions)
  Suffix K:     [b, 50, 1, 256]      (50 new suffix keys)
  Suffix V:     [b, 50, 1, 256]

  Cache K:      [b, 968, 1, 256]     (968 cached prefix keys)
  Cache V:      [b, 968, 1, 256]

  Concatenated:
  Full K:       [b, 1018, 1, 256]    (968 + 50)
  Full V:       [b, 1018, 1, 256]

  Attention:    Q @ K^T -> [b, 8, 50, 1018]   (GQA broadcast)
  Output:       softmax(scores) @ V -> [b, 50, 8, 256]
```

Source: `gemma.py`, lines 211-217.

```python
# gemma.py, lines 211-217
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    k = jnp.concatenate([cache_k, k], axis=1)
    v = jnp.concatenate([cache_v, v], axis=1)

q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
```

### Phase 3: No Cache Update (static, read-only)

A critical property: **the prefix KV cache is never updated during the denoising loop**.
Each denoising step reads the same cached prefix K, V but does not write new suffix K, V
back to the cache.

In the JAX code, the `step` function calls the LLM with `kv_cache=kv_cache` (the prefix
cache), and the returned cache from the suffix pass is discarded:

```python
# pi0.py, lines 261-268
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens],
    mask=full_attn_mask,
    positions=positions,
    kv_cache=kv_cache,            # read prefix cache
    adarms_cond=[None, adarms_cond],
)
# The returned cache (_) is discarded -- not fed back to next step
```

In the PyTorch code, `use_cache=False` is explicitly passed during the denoising step:

```python
# pi0_pytorch.py, lines 450-457
outputs_embeds, _ = self.paligemma_with_expert.forward(
    attention_mask=full_att_2d_masks_4d,
    position_ids=position_ids,
    past_key_values=past_key_values,    # read prefix cache
    inputs_embeds=[None, suffix_embs],
    use_cache=False,                     # do NOT update cache
    adarms_cond=[None, adarms_cond],
)
```

This is correct because the suffix changes every step (x_t is updated by the Euler step),
so there is nothing to cache from previous suffix passes.

---

## Dual-Expert KV Cache Subtlety

The pi0.5 architecture has two "experts" that share attention computation but have
separate weight matrices:

| Expert | Role | Width | Tokens |
|--------|------|-------|--------|
| Expert 0 (PaliGemma) | Vision-language backbone | 2048 | Prefix (968 tokens) |
| Expert 1 (Action) | Action prediction | 1024 | Suffix (50 tokens) |

### How Shared Attention Works

Despite different embedding dimensions, both experts share the same attention
hyperparameters: num_heads=8, num_kv_heads=1, head_dim=256. This means Q, K, V have the
same shape per token regardless of which expert produced them.

During the prefix-only forward pass, only Expert 0 is active. The cache stores K, V
computed from Expert 0's projection weights (kv_einsum_0).

During the denoising loop, only Expert 1 is active. The new K, V are computed from Expert
1's projection weights (kv_einsum_1). These are concatenated with Expert 0's cached K, V.

```
  Prefix pass:                    Denoising pass:
  Expert 0 weights               Expert 1 weights
  kv_einsum (no suffix)           kv_einsum_1 (no suffix)

  K_prefix [b, 968, 1, 256]      K_suffix [b, 50, 1, 256]
  V_prefix [b, 968, 1, 256]      V_suffix [b, 50, 1, 256]
                   |                      |
                   +-------> concat <-----+
                             |
                             v
                   K_full [b, 1018, 1, 256]
                   V_full [b, 1018, 1, 256]
```

**Key insight**: The prefix K, V were computed with Expert 0's projection weights, and the
suffix K, V are computed with Expert 1's weights. They live in the same head_dim=256
space after projection, so concatenation is mathematically valid. But they came from
different linear transformations -- this is by design, enabling the two experts to
specialize.

Source: `gemma.py`, lines 173-201 (per-expert Q/K/V projection); lines 443-450 (naming
convention: no suffix for expert 0, "_1" suffix for expert 1).

### Training vs. Inference Cache Behavior

During training, there is no KV cache. Both prefix and suffix tokens are processed in a
single forward pass:

```python
# pi0.py, lines 209-213 (training)
(prefix_out, suffix_out), _ = self.PaliGemma.llm(
    [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
)
```

During inference, the two-phase strategy (prefix cache fill + suffix-only denoising) is
functionally equivalent but much faster because the 968 prefix tokens are only processed
once instead of 11 times (1 + 10 denoising steps).

---

## PyTorch Cache Format Differences

The PyTorch implementation wraps the cache in HuggingFace's `past_key_values` format,
which differs from the JAX format:

| Aspect | JAX (gemma.py) | PyTorch (gemma_pytorch.py) |
|--------|----------------|---------------------------|
| Cache structure | Tuple of (K_all_layers, V_all_layers) | List of (K, V) tuples per layer |
| K shape per layer | [b, seq, num_kv_heads, head_dim] | [b, num_kv_heads, seq, head_dim] |
| Concatenation | Manual jnp.concatenate | HuggingFace handles internally |
| Cache return | Always returned as 2nd element | Controlled by use_cache flag |

In the PyTorch implementation, the prefix cache is computed via a standard HuggingFace
forward pass with `use_cache=True`:

```python
# pi0_pytorch.py, lines 394-400
_, past_key_values = self.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)
```

During denoising, the prefix cache is passed as `past_key_values` but only for the
PaliGemma language model path. When only the action expert runs (suffix-only mode), the
code directly uses `past_key_values` in the expert's forward call:

```python
# gemma_pytorch.py, lines 113-121
elif inputs_embeds[0] is None:
    suffix_output = self.gemma_expert.model.forward(
        inputs_embeds=inputs_embeds[1],
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
    )
```

---

## Memory Budget Analysis

For Tenstorrent deployment, the KV cache dominates inference memory. Here is the breakdown:

```
  KV Cache Memory per Batch Element (bf16):

  Per layer:  2 * 968 * 1 * 256 * 2 bytes = 991,232 bytes = 0.95 MB
  All layers: 18 * 0.95 MB = 17.1 MB

  With batch size 1:  17.1 MB
  With batch size 8:  136.8 MB
  With batch size 16: 273.5 MB
```

Compared to model weights:
```
  PaliGemma Gemma-2B weights: ~2B params * 2 bytes = ~4 GB
  Action Expert Gemma-300M:   ~300M params * 2 bytes = ~600 MB
  KV cache (b=1):             17.1 MB  (0.4% of model weights)
```

The cache is small relative to model weights, so it should comfortably fit in SRAM on
Tenstorrent hardware even with reasonable batch sizes.

---

## Implications for TT-Metal Implementation

### Static Allocation

All cache dimensions are known at model configuration time:
- Layers: 18 (fixed by config)
- Sequence length: 968 (3 cameras * 256 patches + max_token_len=200)
- KV heads: 1
- Head dim: 256

This allows pre-allocating the cache buffer once and reusing it across observations.

### Read-Only During Hot Loop

The cache is written once (prefix pass) and read 10 times (denoising loop) without
modification. This means:
- No write-back logic needed in the denoising kernel
- The cache can be placed in read-optimized memory
- No synchronization concerns between denoising steps regarding the cache

### Concatenation Is the Key Operation

Each denoising step concatenates suffix K, V with cached prefix K, V. On TT hardware,
this can be implemented as:
1. Physical concatenation (copy suffix K,V after prefix K,V in a contiguous buffer)
2. Virtual concatenation (pass two separate buffers with appropriate indexing)

Option 2 avoids the copy and is likely preferred for performance.

### No Suffix-to-Suffix Caching

Unlike autoregressive text generation where each new token's K, V is appended to the
cache, here the entire suffix is recomputed from scratch each step. The suffix cache from
step i is useless at step i+1 because x_t has changed. This simplifies the implementation
-- there is no growing cache to manage.

---

## Key Takeaways

1. **The KV cache stores prefix K, V across all 18 layers**: shape [b, 968, 1, 256] per
   tensor per layer, totaling ~17 MB per batch element in bf16. It is populated once and
   read 10 times.

2. **The cache is strictly read-only during the denoising loop**: suffix K, V are computed
   fresh each step and concatenated with the cached prefix K, V, but the cache itself is
   never modified.

3. **Dual-expert K, V coexist in the same head space**: prefix K, V come from Expert 0
   (PaliGemma) weights and suffix K, V from Expert 1 (action expert) weights. Both
   project to the same [num_kv_heads=1, head_dim=256] shape, making concatenation valid.

4. **All dimensions are static and known at config time**: this enables pre-allocation
   of fixed-size buffers on TT hardware with no dynamic memory management.

5. **The PyTorch cache format differs from JAX**: HuggingFace uses per-layer (K, V) tuples
   with shape [b, num_kv_heads, seq, head_dim] (note the transposed axes vs JAX). Any
   TT-Metal port must account for this layout difference.
