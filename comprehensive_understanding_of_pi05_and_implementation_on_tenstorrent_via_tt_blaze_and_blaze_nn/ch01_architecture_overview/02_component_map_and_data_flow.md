# 02 -- Component Map and Data Flow

## Context

pi0.5 is composed of four major functional components: a SigLIP vision encoder, a PaliGemma Gemma-2B backbone (expert 0), an Action Expert Gemma-300M (expert 1), and a flow-matching denoiser loop. This file maps out each component, traces the end-to-end data flow from raw camera frames to final action outputs, and explains the prefix/suffix split, the shared-attention transformer pass, and the iterative Euler denoising procedure.

---

## 2.1 Four Major Components

```
+---------------------------------------------------------------------+
|                          pi0.5 Model                                |
|                                                                     |
|  +------------------+    +-------------------+    +---------------+ |
|  |  SigLIP So400m   |    |  PaliGemma 2B     |    | Action Expert | |
|  |  Vision Encoder  |    |  (Expert 0)       |    | 300M          | |
|  |                  |    |  Gemma backbone    |    | (Expert 1)    | |
|  |  ~400M params    |    |  ~2B params        |    | ~311M params  | |
|  |  27 layers       |    |  18 layers shared  |    | 18 layers     | |
|  |  width=1152      |    |  width=2048        |    | width=1024    | |
|  +--------+---------+    +---------+----------+    +-------+-------+ |
|           |                        |                       |         |
|           v                        v                       v         |
|  +------------------------------------------------------------------+|
|  |              Shared-Attention Transformer (18 layers)             ||
|  |         Experts 0 & 1 share K,V; each has own Q,FFN,Norms        ||
|  +------------------------------------------------------------------+|
|           |                                                          |
|           v                                                          |
|  +------------------+                                                |
|  | Flow-Matching    |   10 Euler steps, dt = -1/10                   |
|  | Denoiser Loop    |   x_{t+dt} = x_t + dt * v_t                   |
|  +------------------+                                                |
+---------------------------------------------------------------------+
```

### Component 1: SigLIP Vision Encoder

The SigLIP `So400m/14` model processes each camera image independently. It performs 14x14 patch extraction, adds sinusoidal 2D positional embeddings, and runs 27 transformer encoder layers. With `pool_type="none"`, all spatial tokens are preserved. A final linear head projects from SigLIP's native 1152-dim to PaliGemma's 2048-dim embedding space (`num_classes=paligemma_config.width`).

For a $224 \times 224$ image with $14 \times 14$ patches: $(224/14)^2 = 256$ tokens per image.

Source: `Pi0.__init__()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 81-90), `_Module.__call__()` in `/tmp/openpi/src/openpi/models/siglip.py`.

### Component 2: PaliGemma 2B (Expert 0)

The primary language-vision backbone. It provides the vocabulary embedder (`Embedder` with vocab size 257,152) used to embed tokenized prompts, and contributes one set of Q/K/V projections and FFN weights at each of the 18 shared transformer layers. In pi0.5, this expert does *not* use adaRMSNorm -- it uses standard RMSNorm with a learned scale.

Source: `gemma.Module.setup()` in `/tmp/openpi/src/openpi/models/gemma.py` (lines 350-381), initialization with `adarms=config.pi05` and `use_adarms=[False, True]` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 73-80).

### Component 3: Action Expert 300M (Expert 1)

A smaller Gemma variant (width 1024, MLP dim 4096) that processes action tokens. In pi0.5, every RMSNorm in this expert is replaced by adaRMSNorm, conditioned on the denoising timestep embedding. This expert shares the same 18-layer depth and attention geometry as PaliGemma, enabling the shared-attention mechanism.

Source: `get_config("gemma_300m")` in `/tmp/openpi/src/openpi/models/gemma.py` (lines 69-77).

### Component 4: Flow-Matching Denoiser

Not a separate neural network, but an iterative loop that calls the transformer backbone 10 times. Starting from Gaussian noise $x_1 \sim \mathcal{N}(0, I)$, each step computes a velocity field $v_t$ and updates:

$$
x_{t + \Delta t} = x_t + \Delta t \cdot v_t, \qquad \Delta t = -\frac{1}{N_{\text{steps}}}
$$

The loop runs from $t=1$ (pure noise) to $t \approx 0$ (clean actions), using `jax.lax.while_loop`.

Source: `Pi0.sample_actions()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 216-279).

---

## 2.2 End-to-End Data Flow: Prefix Construction

The prefix encodes everything the model knows *before* seeing the noisy actions. It is computed once and cached.

```
CAMERA IMAGES (3 x 224x224x3)             LANGUAGE INSTRUCTION
       |                                          |
       v                                          v
  +-----------+                          +------------------+
  |  SigLIP   |  x3 cameras             |  PaliGemma       |
  | So400m/14 |  256 tokens each         |  Embedder        |
  +-----------+                          +------------------+
       |                                          |
       | 3 x [B, 256, 2048]                       | [B, 200, 2048]
       v                                          v
  +---------------------------------------------------+
  |               Concatenate along seq dim            |
  |   image_tokens (768) + language_tokens (<=200)     |
  |           = prefix  ~968 tokens                    |
  +---------------------------------------------------+
       |
       |  ar_mask: all False (bidirectional attention within prefix)
       v
  prefix_tokens: [B, ~968, 2048]
  prefix_mask:   [B, ~968]       (True for valid, False for padding)
```

### Image tokens

Each of the three cameras produces 256 tokens. The image mask per camera is a scalar boolean broadcast to all 256 positions -- if a camera is absent, all its tokens are masked out. Total image tokens: $3 \times 256 = 768$.

### Language tokens

The tokenized prompt is embedded via `PaliGemma.llm.embed()`, producing up to `max_token_len=200` tokens at width 2048. A separate `tokenized_prompt_mask` marks which positions are real vs padding.

### Attention mask

All prefix tokens have `ar_mask=False`, meaning they can attend to each other bidirectionally (prefix-LM style). No prefix token can attend to suffix tokens.

Source: `Pi0.embed_prefix()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 106-137).

---

## 2.3 End-to-End Data Flow: Suffix Construction

The suffix is rebuilt at every denoising step because the noisy actions change.

```
NOISY ACTIONS [B, 50, 32]          TIMESTEP t (scalar per batch)
       |                                   |
       v                                   v
  +-------------+                  +-------------------+
  | action_in_  |                  | posemb_sincos()   |
  | proj        |                  | min=4e-3,max=4.0  |
  | 32 -> 1024  |                  +-------------------+
  +-------------+                          |
       |                                   | [B, 1024]
       | [B, 50, 1024]                     v
       |                           +-------------------+
       |                           | time_mlp_in(1024) |
       |                           | SiLU              |
       |                           | time_mlp_out(1024)|
       |                           | SiLU              |
       |                           +-------------------+
       |                                   |
       v                                   v
  action_tokens [B, 50, 1024]      adarms_cond [B, 1024]
       |                              (fed to every RMSNorm
       |                               in Action Expert)
       v
  suffix_tokens: [B, 50, 1024]
  suffix_mask:   [B, 50] (all True)
  ar_mask:       [True, False, False, ..., False]
                  ^first action token breaks causal boundary
```

### Noisy actions

At each denoising step, the current noisy actions $x_t \in \mathbb{R}^{B \times 50 \times 32}$ are projected from action_dim (32) to the Action Expert width (1024) via `action_in_proj`.

### Timestep embedding

The scalar timestep $t \in [0, 1]$ is encoded via sine-cosine positional embeddings (`posemb_sincos` with `min_period=4e-3`, `max_period=4.0`) into a 1024-dim vector, then refined by a two-layer MLP with SiLU activations. The result (`adarms_cond`) is passed to every adaRMSNorm in the Action Expert. (Note: the codebase uses `swish`, which is mathematically identical to SiLU: $\sigma(x) \cdot x$.)

### Attention mask for suffix

The first action token has `ar_mask=True`, meaning prefix tokens cannot attend to it (creating a causal boundary). The remaining 49 action tokens have `ar_mask=False`, so all action tokens can attend to each other bidirectionally. All action tokens can attend to all prefix tokens.

Source: `Pi0.embed_suffix()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 139-186).

---

## 2.4 Shared-Attention Transformer Pass

The prefix and suffix are concatenated and processed together through 18 transformer layers. Each layer runs separate weights for the two experts but shares the attention computation.

```
prefix_tokens [B, ~968, 2048]    suffix_tokens [B, 50, 1024]
       |                                 |
       v                                 v
  Expert 0 (PaliGemma)            Expert 1 (Action Expert)
  Q,K,V projections               Q,K,V projections
  [width=2048 -> heads=8,         [width=1024 -> heads=8,
   kv_heads=1, head=256]           kv_heads=1, head=256]
       |                                 |
       +------------ concat K,V ---------+
       |                                 |
       v                                 v
  +-----------------------------------------------+
  |  Joint self-attention over full sequence       |
  |  Q: [B, ~1018, 8, 256]                        |
  |  K,V: [B, ~1018, 1, 256]  (GQA, 1 KV head)   |
  |  Attention mask enforces prefix-LM pattern     |
  +-----------------------------------------------+
       |                                 |
       v                                 v
  Expert 0 output proj            Expert 1 output proj
  + RMSNorm residual              + adaRMSNorm residual (gated)
  + Expert 0 FFN                  + Expert 1 FFN
  + RMSNorm residual              + adaRMSNorm residual (gated)
       |                                 |
       v                                 v
  (to next layer)                 (to next layer)
```

Crucially:

- Both experts produce Q, K, V with the *same* head geometry: 8 heads, 1 KV head, head_dim 256.
- K and V from both experts are concatenated along the sequence dimension, so every token can attend to every other (subject to the attention mask).
- Each expert has its own output projection and FFN with its own width.
- Only Expert 1 receives adaRMSNorm conditioning in pi0.5; Expert 0 uses standard RMSNorm.

After all 18 layers, the last `action_horizon` (50) positions of the suffix output are extracted and projected back to the action dimension:

```python
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
# v_t shape: [B, 50, 32]
```

Source: `gemma.Block.__call__()` in `/tmp/openpi/src/openpi/models/gemma.py` (lines 284-333), `gemma.Attention.__call__()` (lines 158-249).

---

## 2.5 Denoising Loop: 10 Euler Steps

During inference (`sample_actions`), the model runs an ODE solver from $t=1$ to $t=0$:

```
t=1.0 (pure noise)
  |
  |  Step 1: embed suffix(x_1.0, t=1.0) -> v_t
  |           x_0.9 = x_1.0 + (-0.1) * v_t
  |
  |  Step 2: embed suffix(x_0.9, t=0.9) -> v_t
  |           x_0.8 = x_0.9 + (-0.1) * v_t
  |
  |  ...  (8 more steps)
  |
  |  Step 10: embed suffix(x_0.1, t=0.1) -> v_t
  |            x_0.0 = x_0.1 + (-0.1) * v_t
  v
t=0.0 (clean actions)  ->  output [B, 50, 32]
```

Key implementation details:

1. **KV cache.** The prefix is processed once; its K/V tensors are cached and reused for all 10 suffix passes. Only the suffix tokens generate new Q/K/V at each step.
2. **`jax.lax.while_loop`.** The loop is written as a while loop (not a for loop) to support dynamic step counts and XLA compilation.
3. **Convention note.** The code uses the diffusion convention where $t=1$ is noise and $t=0$ is the target, which is opposite to the pi0 paper's convention.
4. **Euler-step arithmetic.** The Euler update $x_{t+dt} = x_t + dt \cdot v_t$ is performed in float32 to avoid accumulating bfloat16 rounding error over 10 integration steps.

The overall compute cost of inference is dominated by these 10 transformer passes over the suffix, each requiring attention against the cached prefix.

Source: `Pi0.sample_actions()` in `/tmp/openpi/src/openpi/models/pi0.py` (lines 216-279).

---

## 2.6 Complete Data Flow Summary

```
Cameras (3x224x224)  Language Prompt   Robot State (discrete)
       |                   |                   |
       v                   v                   v
    SigLIP            Embedder           (tokenized into
   So400m/14          (vocab 257k)        language tokens)
       |                   |                   |
       | 768 tokens        | <=200 tokens      |
       +-------------------+-------------------+
                           |
                   PREFIX (~968 tokens, width 2048)
                           |
                   [Processed once, KV cached]
                           |
                           v
              +------------------------+
              |  FOR step = 1..10:     |
              |                        |
              |  Noisy actions ------> action_in_proj (32->1024)
              |  Timestep -----------> sincos -> time_mlp -> adaRMSNorm cond
              |                        |
              |  SUFFIX (50 tokens, width 1024)
              |                        |
              |  18-layer shared-attn transformer
              |  (prefix KV from cache + suffix KV)
              |                        |
              |  suffix_out[:, -50:] -> action_out_proj (1024->32)
              |                        |
              |  x_{t+dt} = x_t + dt * v_t
              +------------------------+
                           |
                           v
                  ACTIONS [B, 50, 32]  (float32)
```

---

## Token Budget Summary

| Token Group | Source | Count | Width | Notes |
|---|---|---|---|---|
| Image tokens (camera 1) | SigLIP So400m/14 | 256 | 2048 | $(224/14)^2 = 256$ spatial tokens |
| Image tokens (camera 2) | SigLIP So400m/14 | 256 | 2048 | Shared weights |
| Image tokens (camera 3) | SigLIP So400m/14 | 256 | 2048 | Shared weights |
| Language tokens | PaliGemma Embedder | $\leq 200$ | 2048 | Includes discrete state in pi0.5 |
| **Prefix total** | | **~968** | **2048** | Computed once, KV cached |
| Action tokens | `action_in_proj(x_t)` | 50 | 1024 | Recomputed per denoising step |
| **Suffix total** | | **50** | **1024** | Per-step, Action Expert only |
| **Full sequence** | | **~1018** | mixed | Prefix (2048) + suffix (1024) |

---

## Key Takeaways

- pi0.5 has four functional components: SigLIP (vision), PaliGemma 2B (language-vision backbone / expert 0), Action Expert 300M (expert 1), and the flow-matching denoiser loop.
- The prefix comprises 768 image tokens (3 cameras x 256 tokens) plus up to 200 language tokens, totaling roughly 968 tokens at width 2048.
- The suffix contains 50 action tokens at width 1024, with the denoising timestep injected through adaRMSNorm conditioning rather than concatenation.
- The 18-layer shared-attention mechanism lets both experts attend to each other's tokens while maintaining separate Q/K/V projections, output projections, and FFN weights.
- Inference runs 10 Euler denoising steps, each requiring a suffix-only transformer pass against cached prefix K/V, producing the final output of shape $[B, 50, 32]$.

---

**Next:** [`03_data_types_and_observation.md`](./03_data_types_and_observation.md)
