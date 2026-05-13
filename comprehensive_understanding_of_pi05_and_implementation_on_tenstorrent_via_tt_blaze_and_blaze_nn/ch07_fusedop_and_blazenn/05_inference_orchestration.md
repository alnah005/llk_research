# 05 -- Inference Orchestration: Two-Phase Inference and Denoising Loop

## Context

Pi0.5 inference is fundamentally different from standard autoregressive LLM inference.
Instead of generating tokens one at a time, Pi0.5 generates a continuous action
trajectory through iterative **flow-matching denoising**. The inference has two
distinct phases:

1. **Prefix phase (once)**: Process all observation tokens (images + language prompt)
   through the full dual-expert backbone, producing a KV cache. The action expert
   receives no tokens in this phase (`[prefix_tokens, None]`).

2. **Denoising phase (10 iterations)**: Starting from Gaussian noise, iteratively
   refine the action trajectory. Each iteration processes only the action tokens
   through the action expert (`[None, suffix_tokens]`), reusing the VLM's KV cache
   from the prefix phase.

The reference implementation is in `openpi/models/pi0.py`, method `sample_actions()`
(lines 216-279).

## Key Takeaways

1. The prefix phase runs the full VLM pipeline (SiGLIP image encoding + Gemma
   backbone) **once** and caches the KV pairs. This is the expensive step (~500ms)
   but it only happens once per observation.

2. Each denoising step is lightweight -- only the action expert's suffix tokens
   pass through the backbone. The VLM expert is None, so its norms, projections,
   and FFN are all skipped. SDPA uses the cached KV from the prefix phase.

3. The denoising loop runs **10 steps** by default (configurable via `num_steps`).
   Each step: embed noisy actions + timestep -> forward through backbone with KV
   cache -> predict velocity -> Euler step.

4. Host-side control manages the denoising loop. The timestep, noise schedule,
   and Euler integration are all computed on CPU. Only the backbone forward pass
   runs on device.

5. The None-expert pattern (`[None, suffix_tokens]`) is critical for performance.
   Without it, the VLM's 2B parameters would be re-evaluated at every denoising
   step, making inference 10x slower.

## Phase 1: Prefix Encoding

```python
# From pi0.py sample_actions(), lines 233-237:
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
_, kv_cache = self.PaliGemma.llm(
    [prefix_tokens, None],   # <-- action expert is None
    mask=prefix_attn_mask,
    positions=positions,
)
```

### What happens in `embed_prefix`:

1. **Image encoding**: Each observation image (base, left wrist, right wrist) is
   processed by SiGLIP to produce image tokens at width=2048
2. **Language encoding**: Tokenized prompts are embedded via the Gemma embedder
3. **Concatenation**: Image tokens + language tokens are concatenated along the
   sequence dimension
4. **Attention mask**: Images use bidirectional attention (ar_mask=False), forming
   a prefix-LM pattern

### Prefix phase data flow

```
  base_image [B, 224, 224, 3]
  left_wrist [B, 224, 224, 3]   ──> [SiGLIP] ──> image_tokens [B, 3*256, 2048]
  right_wrist [B, 224, 224, 3]

  tokenized_prompt [B, 200] ──> [Embedder] ──> lang_tokens [B, 200, 2048]

  [concat] ──> prefix_tokens [B, ~968, 2048]
                    |
                    v
  [GemmaBackbone with [prefix_tokens, None]]
       |
       +──> Layer 0:
       |    - VLM: RMSNorm -> QKV -> SDPA -> O_proj -> ResAdd -> RMSNorm -> FFN -> ResAdd
       |    - Action: SKIPPED (None)
       |    - KV cache layer 0 stored
       |
       +──> Layer 1: same pattern
       |    ...
       +──> Layer 17: same pattern
       |
       v
  kv_cache: [18, B, ~968, num_kv_heads, head_dim]
  (stored for reuse in denoising phase)
```

### Blaze implementation of prefix phase

```python
class Pi05Module(Module):
    def encode_prefix(self, observation):
        """Phase 1: encode observation, return KV cache."""
        # Image encoding (on device)
        image_tokens = self.siglip(observation.images)

        # Language embedding (on device)
        lang_tokens = F.embedding(observation.tokenized_prompt,
                                  self.embedder.input_embedding)

        # Concatenate prefix tokens
        prefix_tokens = F.concat_seq(image_tokens, lang_tokens)

        # Run backbone with action expert = None
        _, kv_cache = self.backbone(
            vlm_tokens=prefix_tokens,
            action_tokens=None,
            positions=positions,
            attn_mask=prefix_attn_mask,
            adarms_cond=[None, None],  # no adaRMS in prefix phase
        )
        return kv_cache
```

## Phase 2: Denoising Loop

```python
# From pi0.py sample_actions(), lines 239-278:
def step(carry):
    x_t, time = carry
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
        observation, x_t, jnp.broadcast_to(time, batch_size)
    )
    # Build attention mask: suffix attends to prefix (via KV cache) + itself
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens],    # <-- VLM expert is None
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    assert prefix_out is None
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    return x_t + dt * v_t, time + dt

x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
```

### What happens in `embed_suffix`:

1. **Action projection**: Noisy actions `x_t [B, 50, 32]` projected to action expert
   width: `action_in_proj(x_t)` -> `[B, 50, 1024]`
2. **Timestep encoding**: Scalar timestep -> sincos positional encoding -> time MLP
   (two linear layers with SiLU) -> `adarms_cond [B, 1024]`
3. **No state token** (pi05 mode): Unlike pi0, pi0.5 encodes state as part of the
   language prompt, not as a separate continuous token

### Denoising step data flow

```
  Iteration i (time = 1.0 - i*dt, dt = -0.1):

  x_t [B, 50, 32] ──> [action_in_proj] ──> action_tokens [B, 50, 1024]

  time (scalar) ──> [posemb_sincos] ──> time_emb [B, 1024]
                          |
                          v
                    [time_mlp_in] ──> [SiLU] ──> [time_mlp_out] ──> [SiLU]
                          |
                          v
                    adarms_cond [B, 1024]

  [GemmaBackbone with [None, action_tokens], kv_cache, adarms_cond]
       |
       +──> Layer 0:
       |    - VLM: SKIPPED (None) -- KV from cache used in SDPA
       |    - Action: AdaRMSNorm(cond) -> QKV -> SDPA(+cached KV) -> O_proj
       |              -> GatedResidualAdd -> AdaRMSNorm(cond) -> FFN
       |              -> GatedResidualAdd
       |
       +──> Layer 1-17: same pattern
       |
       v
  suffix_out [B, 50, 1024]
       |
       v
  [action_out_proj] ──> v_t [B, 50, 32]  (predicted velocity)
       |
       v
  x_{t+dt} = x_t + dt * v_t   (Euler step, computed on host)
```

### Blaze implementation of denoising step

```python
class Pi05Module(Module):
    def denoise_step(self, x_t, time, observation, kv_cache):
        """Single denoising step. Returns updated x_t."""
        # Embed noisy actions
        action_tokens = F.linear(x_t, self.action_in_proj.weight)

        # Compute timestep conditioning
        time_emb = posemb_sincos(time, width=1024,
                                 min_period=4e-3, max_period=4.0)
        time_emb = F.linear(time_emb, self.time_mlp_in.weight)
        time_emb = F.silu(time_emb)
        time_emb = F.linear(time_emb, self.time_mlp_out.weight)
        adarms_cond = F.silu(time_emb)

        # Run backbone with VLM = None, using KV cache
        _, suffix_out = self.backbone(
            vlm_tokens=None,
            action_tokens=action_tokens,
            positions=positions,
            attn_mask=full_attn_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )

        # Project back to action space
        v_t = F.linear(suffix_out[:, -self.action_horizon:],
                        self.action_out_proj.weight)
        return v_t

    def sample_actions(self, observation, num_steps=10):
        """Full inference: prefix + denoising loop."""
        # Phase 1: encode prefix (once)
        kv_cache = self.encode_prefix(observation)

        # Phase 2: denoising loop (10 steps)
        dt = -1.0 / num_steps
        x_t = torch.randn(batch_size, self.action_horizon, self.action_dim)
        time = 1.0

        for step in range(num_steps):
            v_t = self.denoise_step(x_t, time, observation, kv_cache)
            x_t = x_t + dt * v_t    # Euler integration (host-side)
            time = time + dt

        return x_t  # denoised action trajectory
```

## Host-Side Control Flow

The denoising loop is orchestrated by the host CPU. The device executes the backbone
forward pass; everything else (noise initialization, timestep scheduling, Euler
integration, loop control) happens on the host.

```
Host-side orchestration timeline:

  t=0ms    [Host] Initialize noise x_1 ~ N(0,1)
  t=1ms    [Host] Prepare prefix inputs (images, language)
  t=2ms    [Device] Phase 1: encode_prefix()
  t=502ms  [Host] Receive KV cache from device
  t=503ms  [Host] Compute timestep t=1.0, embed suffix
  t=504ms  [Device] Phase 2, step 1: denoise_step(x_1, t=1.0)
  t=554ms  [Host] x_0.9 = x_1 + dt * v_1
  t=555ms  [Host] Compute timestep t=0.9, embed suffix
  t=556ms  [Device] Phase 2, step 2: denoise_step(x_0.9, t=0.9)
  t=606ms  [Host] x_0.8 = x_0.9 + dt * v_0.9
  ...
  t=1005ms [Host] x_0 = x_0.1 + dt * v_0.1  (final action trajectory)
  t=1006ms [Host] Return x_0

  Approximate total: ~1 second
  - Prefix: ~500ms (one-time)
  - Per denoise step: ~50ms (x10 = ~500ms)
```

The key insight is that **KV cache transfer is zero-cost** -- the cache stays on
device between the prefix phase and the denoising loop. Only the small suffix
tensors (`[B, 50, 1024]`) and the timestep conditioning (`[B, 1024]`) are
transferred per step.

## The None-Expert Pattern: Performance Impact

Without the None-expert pattern, each denoising step would process both experts:

```
Without None-expert (naive):
  Per step: VLM (2048 width, ~968 tokens) + Action (1024 width, 50 tokens)
  Compute: ~200ms per step (dominated by VLM FFN: [2048, 16384])
  10 steps: ~2000ms

With None-expert (Pi0.5):
  Per step: Action only (1024 width, 50 tokens) + SDPA with cached VLM KV
  Compute: ~50ms per step (action expert FFN: [1024, 4096] is 16x smaller)
  10 steps: ~500ms

Speedup: ~4x on denoising loop
```

The None-expert pattern saves compute because:
1. VLM norms/projections/FFN are all skipped (width=2048, 16K hidden -> 0 FLOPs)
2. Only the action expert's small FFN runs (width=1024, 4K hidden)
3. SDPA computes attention only for the 50 suffix query tokens against cached KV

## KV Cache Management

The KV cache from the prefix phase has shape:

```
kv_cache per layer: (K, V) where K, V are [B, S_prefix, num_kv_heads, head_dim]

For Pi0.5:
  B = 1 (typical robot inference)
  S_prefix = ~968 (3*256 image + ~200 language tokens)
  num_kv_heads = 1 (GQA with 1 KV head)
  head_dim = 256
  num_layers = 18

Total KV cache size:
  2 (K+V) * 18 layers * 1 * 968 * 1 * 256 * 2 bytes (BF16)
  = 2 * 18 * 968 * 256 * 2 = ~18 MB
```

This 18 MB lives in device DRAM and is accessed during each denoising step's SDPA.
The SDPA op concatenates the cached KV with the new suffix KV:

```python
# From gemma.py Attention.__call__():
if kv_cache is not None:
    cache_k, cache_v = kv_cache
    k = jnp.concatenate([cache_k, k], axis=1)  # [B, S_prefix+S_suffix, ...]
    v = jnp.concatenate([cache_v, v], axis=1)
```

In Blaze, this concatenation happens inside the SDPA op -- the cached KV tensors
are passed as additional inputs. The SDPA kernel reads cached KV from DRAM while
computing attention for the new suffix queries.

## Attention Mask Structure

The attention mask for the denoising step has a specific structure:

```
full_attn_mask [B, S_suffix, S_prefix + S_suffix]:

         prefix tokens (968)        suffix tokens (50)
        +------------------------+-------------------+
 suffix | 1 1 1 1 ... 1 1 1 1 1 | 1 0 0 0 ... 0 0 0 |  suffix[0]
 tokens | 1 1 1 1 ... 1 1 1 1 1 | 1 1 0 0 ... 0 0 0 |  suffix[1]
  (50)  | 1 1 1 1 ... 1 1 1 1 1 | 1 1 1 0 ... 0 0 0 |  suffix[2]
        | ...                     | ...                |  ...
        | 1 1 1 1 ... 1 1 1 1 1 | 1 1 1 1 ... 1 1 1 |  suffix[49]
        +------------------------+-------------------+

  - All suffix tokens attend to ALL prefix tokens (bidirectional)
  - Suffix tokens attend causally to each other (upper triangle)
  - First suffix token starts a new causal block (ar_mask=[True])
  - Remaining suffix tokens share mask (ar_mask=[False] * 49)
```

The first action token has `ar_mask=True`, creating a causal boundary: prefix tokens
cannot attend to action tokens (enforced during the prefix phase), but action tokens
can attend to all prefix tokens.

## ASCII Diagram: Complete Inference Pipeline

```
  PHASE 1: PREFIX ENCODING (once per observation)
  ================================================

  Images [3x224x224] ──> [SiGLIP] ──> img_tokens [768, 2048]
  Language [200] ──> [Embedder] ──> lang_tokens [200, 2048]
                          |
                     [concat] ──> prefix [968, 2048]
                          |
                     [Backbone: [prefix, None]]
                          |
                     kv_cache [18 layers]
                          |
                          v (stays on device)


  PHASE 2: DENOISING LOOP (10 iterations)
  ========================================

  x_t = noise [50, 32]
  for step in range(10):
      |
      v
      t = 1.0 - step * 0.1
      |
      +──> [posemb_sincos(t)] ──> [time_mlp] ──> adarms_cond [1024]
      |
      +──> [action_in_proj(x_t)] ──> action_tokens [50, 1024]
      |
      +──> [Backbone: [None, action_tokens], kv_cache, adarms_cond]
      |         |
      |         +──> Layer 0-17: Action expert only (VLM skipped)
      |         |    AdaRMSNorm -> QKV -> SDPA(+cache) -> O_proj
      |         |    -> GatedResAdd -> AdaRMSNorm -> FFN -> GatedResAdd
      |         |
      |         v
      |    suffix_out [50, 1024]
      |         |
      +──> [action_out_proj] ──> v_t [50, 32]
      |
      +──> x_{t-0.1} = x_t + (-0.1) * v_t     [host-side Euler step]
      |
      x_t = x_{t-0.1}  (update for next iteration)

  RESULT: x_0 [50, 32]  (denoised action trajectory)
```

## Source References

- `openpi/models/pi0.py:106-137` -- `embed_prefix()` method
- `openpi/models/pi0.py:139-186` -- `embed_suffix()` method with timestep MLP
- `openpi/models/pi0.py:216-279` -- `sample_actions()` with prefix + denoising loop
- `openpi/models/pi0.py:234-237` -- Prefix pass: `[prefix_tokens, None]`
- `openpi/models/pi0.py:261-268` -- Denoising pass: `[None, suffix_tokens]`
- `openpi/models/pi0.py:47-63` -- `posemb_sincos()` for timestep encoding
- `openpi/models/pi0.py:19-44` -- `make_attn_mask()` for prefix-LM attention
- `openpi/models/gemma.py:157-249` -- Attention class with KV cache support
- `openpi/models/pi0_config.py:18-48` -- Pi0Config with `pi05=True` settings
- `openpi/models/pi0.py:66-70` -- Pi0.__init__ with `self.pi05 = config.pi05`
- `openpi/models/pi0.py:93-99` -- Time MLP layers (pi0.5 path)
