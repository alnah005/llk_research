# 5.2 Complete Inference Walkthrough

## Context

This file traces through the entire `sample_actions` method tensor-by-tensor, covering
both the JAX reference (`pi0.py`) and the PyTorch implementation (`pi0_pytorch.py`). For
Tenstorrent engineers, this is the definitive reference for understanding exactly what
computation happens, in what order, and with what tensor shapes at each stage.

All shapes assume the default pi0.5 configuration: PaliGemma (Gemma-2B) as the VLM
backbone, Gemma-300M as the action expert, 3 camera views, action_dim=32,
action_horizon=50, max_token_len=200.

---

## Phase 1: Prefix Embedding (Runs Once)

The prefix encodes all observation context: images from 3 cameras and a tokenized language
prompt. This runs once per observation and produces tokens that will be cached.

### Step 1a: Image Encoding via SigLIP

Each camera image is processed through the SigLIP vision encoder (So400m/14 variant) to
produce a grid of image tokens.

```
  Input:  image [b, 224, 224, 3]    (float32, range [-1, 1])
  Output: image_tokens [b, 256, 2048]

  SigLIP So400m/14:
    - Patch size: 14x14
    - Grid: 224/14 = 16 x 16 = 256 patches
    - Output dim: projected to paligemma_config.width = 2048
    - Pool type: "none" (all patches kept)
```

With 3 cameras, this produces 3 * 256 = 768 image tokens.

Source: `pi0.py`, lines 112-126; `pi0_pytorch.py`, lines 198-211.

### Step 1b: Language Token Embedding

The tokenized prompt is embedded through PaliGemma's Gemma embedder, which looks up
tokens in a 257,152-entry vocabulary table and scales by sqrt(embed_dim).

```
  Input:  tokenized_prompt [b, 200]    (int32, token IDs)
  Output: lang_tokens [b, 200, 2048]   (scaled by sqrt(2048))
```

Source: `pi0.py`, lines 128-133; `gemma.py`, lines 148-151; `pi0_pytorch.py`, lines 214-219.

### Step 1c: Concatenation and Mask Construction

All prefix tokens are concatenated along the sequence dimension:

```
  prefix_tokens = cat([img_0, img_1, img_2, lang_tokens], dim=1)
  Shape: [b, 768 + 200, 2048] = [b, 968, 2048]

  prefix_mask (padding mask): [b, 968]   (bool)
  prefix_ar_mask:             [968]       (bool, all False = bidirectional)
```

The ar_mask is all False for the prefix, meaning every prefix token can attend to every
other prefix token (bidirectional / prefix-LM attention).

Source: `pi0.py`, lines 134-137.

### Step 1d: Prefix-Only Forward Pass (Cache Fill)

The prefix tokens are fed through the full 18-layer transformer to populate the KV cache.

```
  Input:
    embedded:    [prefix_tokens, None]     -- only expert 0 (PaliGemma) is active
    mask:        [b, 1, 968, 968]          -- bidirectional attention
    positions:   [b, 968]                  -- cumsum of prefix_mask - 1

  Output:
    _           (prefix output, discarded)
    kv_cache    [18, b, 968, 1, 256] x 2   (K and V tensors)
```

Source: `pi0.py`, lines 234-237.

```python
# pi0.py, lines 234-237
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

```
  Prefix Forward Pass (diagram):

  prefix_tokens [b, 968, 2048]
       |
       v
  +-------------------------------------------+
  | Gemma Transformer (18 layers)             |
  | Expert 0 (PaliGemma): ACTIVE              |
  | Expert 1 (Action):    None (skipped)      |
  |                                           |
  | Each layer produces:                      |
  |   K [b, 968, 1, 256]  (1 KV head)        |
  |   V [b, 968, 1, 256]                     |
  +-------------------------------------------+
       |                    |
       v                    v
  prefix_out             kv_cache
  (discarded)            [18 layers] x [K, V]
                         each [b, 968, 1, 256]
```

**Important**: The KV cache stores keys and values for all 968 prefix tokens across all
18 layers. This cache is read-only for the rest of the inference -- it is never updated
during the denoising loop.

---

## Phase 2: Denoising Loop (Runs 10 Times)

Starting from x_1 ~ N(0, I) with shape [b, 50, 32], the loop refines the actions through
10 Euler steps.

### Step 2a: Suffix Embedding

Each iteration re-embeds the current noisy actions and the current timestep.

```
  Inputs:
    x_t       [b, 50, 32]     -- current noisy actions
    timestep  [b]              -- scalar time, broadcast to batch

  Processing Pipeline:

  1. action_in_proj(x_t)
     Linear(32, 1024)
     Output: action_tokens [b, 50, 1024]

  2. posemb_sincos(timestep, dim=1024, min=4e-3, max=4.0)
     Output: time_emb [b, 1024]

  3. time_mlp_in(time_emb)       Linear(1024, 1024)
     swish()
     time_mlp_out(...)           Linear(1024, 1024)
     swish()
     Output: adarms_cond [b, 1024]

  Final suffix outputs:
    suffix_tokens [b, 50, 1024]    -- action_tokens (no time mixed in for pi0.5)
    suffix_mask   [b, 50]          -- all True
    suffix_ar_mask [50]            -- [True, False, False, ..., False]
    adarms_cond   [b, 1024]        -- timestep conditioning for adaRMS
```

Source: `pi0.py`, lines 139-186; `pi0_pytorch.py`, lines 238-315.

**Critical detail about ar_mask**: The first action token has ar_mask=True, meaning prefix
tokens cannot attend to it (the boundary). The remaining 49 action tokens have
ar_mask=False, meaning they can attend to each other bidirectionally. Together with the
prefix, this creates a prefix-LM pattern: prefix tokens see only prefix; action tokens see
prefix + all action tokens.

```
  Attention pattern (prefix-LM):

                   prefix (968 tokens)     suffix (50 tokens)
                  |<--- all False --->|  |<- T,F,F,...,F ->|
                  +-------------------+--+------------------+
  prefix tokens   | bidirectional     | X  cannot see       |
  (968 tokens)    | attention         |    suffix tokens     |
                  +-------------------+--+------------------+
  suffix tokens   | can see all       |    bidirectional     |
  (50 tokens)     | prefix tokens     |    among actions     |
                  +-------------------+--+------------------+
```

### Step 2b: Attention Mask Construction for Suffix Pass

During the denoising loop, the suffix tokens generate queries, while keys/values come from
both the cached prefix and the current suffix.

```
  suffix_attn_mask:  make_attn_mask(suffix_mask, suffix_ar_mask)
                     Shape: [b, 50, 50]

  prefix_attn_mask:  repeat(prefix_mask, "b p -> b s p", s=50)
                     Shape: [b, 50, 968]
                     (every suffix token can attend to every valid prefix token)

  full_attn_mask:    cat([prefix_attn_mask, suffix_attn_mask], dim=-1)
                     Shape: [b, 50, 968 + 50] = [b, 50, 1018]
```

Source: `pi0.py`, lines 246-257.

```python
# pi0.py, lines 246-257
suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
assert full_attn_mask.shape == (
    batch_size,
    suffix_tokens.shape[1],
    prefix_tokens.shape[1] + suffix_tokens.shape[1],
)
```

### Step 2c: Position IDs for Suffix Tokens

Suffix positions continue from where the prefix left off:

```
  positions = sum(prefix_mask, dim=-1)[:, None] + cumsum(suffix_mask, dim=-1) - 1
  Shape: [b, 50]
  Values: [968, 969, 970, ..., 1017]   (if all prefix tokens are valid)
```

Source: `pi0.py`, line 259.

### Step 2d: Transformer Forward Pass (Suffix Only)

The suffix tokens are processed through the 18-layer transformer, reading from the
prefix KV cache but only computing on suffix tokens.

```
  Input:
    embedded:   [None, suffix_tokens]     -- only expert 1 (action expert) active
    mask:       [b, 50, 1018]             -- suffix queries, full KV range
    positions:  [b, 50]                   -- continuation positions
    kv_cache:   [18, b, 968, 1, 256] x 2  -- prefix cache (read-only)
    adarms_cond: [None, adarms_cond]      -- timestep conditioning for expert 1

  Processing per layer:
    1. adaRMSNorm on suffix tokens using adarms_cond
       - Produces scale, shift, gate from Dense(1024 -> 3072)
    2. Q/K/V projection (action expert weights)
       - Q: [b, 50, 8, 256]   (8 heads, 256 head_dim)
       - K: [b, 50, 1, 256]   (1 KV head)
       - V: [b, 50, 1, 256]
    3. K,V concatenated with cached prefix K,V
       - K: [b, 968+50, 1, 256] = [b, 1018, 1, 256]
       - V: [b, 1018, 1, 256]
    4. RoPE applied to Q and K using position IDs
    5. GQA attention: 8 query heads, 1 KV head (group size = 8)
    6. Output projection back to 1024 dims
    7. Gated residual connection (using gate from adaRMSNorm)
    8. adaRMSNorm -> FFN -> gated residual

  Output:
    suffix_out [b, 50, 1024]
```

Source: `pi0.py`, lines 261-268; `gemma.py`, lines 164-248, 283-333.

```
  Single Transformer Layer (action expert, pi0.5 mode):

  suffix_tokens [b, 50, 1024]     adarms_cond [b, 1024]
       |                                |
       v                                v
  +--adaRMSNorm---------+    Dense(1024, 3072)
  | var = mean(x^2)     |         |
  | x_normed = x/sqrt() |    split into 3
  | x = x*(1+scale)+shift|   scale, shift, gate
  +----------------------+    each [b, 1, 1024]
       |                           |
       v (gate saved)              |
  +--Q/K/V projection---+         |
  | Q [b,50,8,256]      |         |
  | K [b,50,1,256]      |         |
  | V [b,50,1,256]      |         |
  +----------------------+         |
       |                           |
  concat K,V with cache            |
  K [b,1018,1,256]                 |
  V [b,1018,1,256]                 |
       |                           |
  +--GQA Attention-------+        |
  | 8 query heads        |        |
  | 1 KV head            |        |
  | attn_mask [b,50,1018]|        |
  +----------------------+        |
       |                           |
  +--Output Projection--+         |
  | [b, 50, 1024]       |         |
  +----------------------+         |
       |                           |
  +--Gated Residual------+        |
  | x + y * gate         |<-------+
  +----------------------+
       |
       v
  +--adaRMSNorm + FFN----+  (same adaRMS pattern)
  | GeLU-gated FFN       |
  | 1024 -> 4096 -> 1024 |
  +--Gated Residual------+
       |
       v
  layer_output [b, 50, 1024]
```

### Step 2e: Action Output Projection

The last action_horizon tokens of the suffix output are projected back to action space:

```
  suffix_out[:, -50:]    [b, 50, 1024]
       |
  action_out_proj        Linear(1024, 32)
       |
       v
  v_t                    [b, 50, 32]    -- predicted velocity
```

Source: `pi0.py`, line 269; `pi0_pytorch.py`, lines 459-462.

### Step 2f: Euler Step

The predicted velocity updates the noisy actions:

```
  x_t = x_t + dt * v_t
      = x_t + (-0.1) * v_t
  time = time + dt
       = time - 0.1
```

Source: `pi0.py`, line 271; `pi0_pytorch.py`, lines 418-419.

---

## Complete Tensor Shape Summary

```
  +---------------------------+----------------------------+--------------------+
  | Tensor                    | Shape                      | Phase              |
  +---------------------------+----------------------------+--------------------+
  | input image (per camera)  | [b, 224, 224, 3]           | prefix embed       |
  | SigLIP output (per cam)   | [b, 256, 2048]             | prefix embed       |
  | tokenized prompt          | [b, 200]                   | prefix embed       |
  | language embeddings       | [b, 200, 2048]             | prefix embed       |
  | prefix_tokens             | [b, 968, 2048]             | prefix embed       |
  | prefix_mask               | [b, 968]                   | prefix embed       |
  | prefix_ar_mask            | [968]                      | prefix embed       |
  | prefix_attn_mask          | [b, 968, 968]              | prefix forward     |
  | positions (prefix)        | [b, 968]                   | prefix forward     |
  | KV cache K (per layer)    | [b, 968, 1, 256]           | prefix forward     |
  | KV cache V (per layer)    | [b, 968, 1, 256]           | prefix forward     |
  +---------------------------+----------------------------+--------------------+
  | x_t (noisy actions)       | [b, 50, 32]                | denoise loop       |
  | timestep                  | [b]                        | denoise loop       |
  | time_emb                  | [b, 1024]                  | suffix embed       |
  | adarms_cond               | [b, 1024]                  | suffix embed       |
  | action_tokens             | [b, 50, 1024]              | suffix embed       |
  | suffix_tokens             | [b, 50, 1024]              | suffix embed       |
  | suffix_mask               | [b, 50]                    | suffix embed       |
  | suffix_ar_mask            | [50]                       | suffix embed       |
  | suffix_attn_mask          | [b, 50, 50]                | denoise forward    |
  | prefix_attn_mask (repeat) | [b, 50, 968]               | denoise forward    |
  | full_attn_mask            | [b, 50, 1018]              | denoise forward    |
  | positions (suffix)        | [b, 50]                    | denoise forward    |
  | Q (per layer)             | [b, 50, 8, 256]            | denoise forward    |
  | K (per layer, suffix)     | [b, 50, 1, 256]            | denoise forward    |
  | V (per layer, suffix)     | [b, 50, 1, 256]            | denoise forward    |
  | K (concat w/ cache)       | [b, 1018, 1, 256]          | denoise forward    |
  | V (concat w/ cache)       | [b, 1018, 1, 256]          | denoise forward    |
  | suffix_out                | [b, 50, 1024]              | denoise forward    |
  | v_t (velocity)            | [b, 50, 32]                | action out proj    |
  | x_0 (clean actions)       | [b, 50, 32]                | final output       |
  +---------------------------+----------------------------+--------------------+
```

---

## JAX vs PyTorch: Structural Comparison of Inference

```
  JAX (pi0.py)                          PyTorch (pi0_pytorch.py)
  =====================================  =====================================
  sample_actions()                       sample_actions()
    |                                      |
    +-- embed_prefix()                     +-- embed_prefix()
    +-- make_attn_mask()                   +-- make_att_2d_masks()
    +-- llm([prefix, None])                +-- paligemma_with_expert.forward(
    |   -> kv_cache                        |       [prefix, None], use_cache=True)
    |                                      |   -> past_key_values
    |                                      |
    +-- jax.lax.while_loop(                +-- while time >= -dt/2:
    |     cond, step, (noise, 1.0))        |
    |                                      |
    |   step():                            |   denoise_step():
    |     +-- embed_suffix()               |     +-- embed_suffix()
    |     +-- make_attn_mask()             |     +-- make_att_2d_masks()
    |     +-- llm([None, suffix],          |     +-- paligemma_with_expert.forward(
    |     |       kv_cache=cache)          |     |       [None, suffix],
    |     +-- action_out_proj()            |     |       past_key_values=cache)
    |     +-- Euler step                   |     +-- action_out_proj()
    |                                      |     +-- return v_t
    |                                      |
    |                                      |   x_t = x_t + dt * v_t
    |                                      |   time += dt
    |                                      |
    v                                      v
  x_0 [b, 50, 32]                       x_0 [b, 50, 32]
```

---

## Key Takeaways

1. **The inference path has two distinct phases**: prefix embedding + cache fill (runs
   once) and the denoising loop (runs 10 times). The prefix is ~20x more tokens but only
   runs once; the suffix loop is the throughput bottleneck.

2. **Tensor shapes are deterministic**: 968 prefix tokens (3*256 image + 200 language),
   50 suffix tokens (action horizon), 1024-dim action expert, 2048-dim PaliGemma. These
   are fixed at model config time, enabling static memory allocation on TT hardware.

3. **The attention mask has prefix-LM structure**: prefix tokens attend bidirectionally
   among themselves; suffix tokens attend to all prefix tokens and bidirectionally among
   themselves; prefix tokens cannot see suffix tokens.

4. **adaRMS conditioning is the key difference from pi0**: instead of concatenating time
   with actions, pi0.5 uses the time MLP output to modulate every RMSNorm in the action
   expert via learned scale/shift/gate parameters.

5. **Each denoising step re-embeds x_t from scratch**: the action_in_proj, time embedding,
   and time MLP all re-execute every iteration. There is no caching of suffix computations
   between steps because x_t changes every step.
