# 3.1 Expert Configs and Weight Layout

## Context

Pi0.5 runs two Gemma-family transformers in lock-step through a shared 18-layer
stack. The first expert is a full PaliGemma 2B model that processes images and
language. The second is a much smaller 300M-parameter action expert that
processes noisy action tokens for the flow-matching denoiser. Both experts must
share the same attention geometry so their Q/K/V tensors can be concatenated
along the sequence dimension. Everything else -- width, MLP dimension, embedding
table -- is allowed to differ.

This section documents the exact configurations, the weight naming convention
that allows seamless checkpoint loading, and how the shared embedder is
partitioned.

---

## Expert Configurations

The two experts are defined in `get_config()`:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 58-87

```
Expert 0 -- PaliGemma 2B (variant "gemma_2b"):
    width       = 2048
    depth       = 18
    mlp_dim     = 16384
    num_heads   = 8
    num_kv_heads = 1
    head_dim    = 256

Expert 1 -- Action Expert 300M (variant "gemma_300m"):
    width       = 1024
    depth       = 18
    mlp_dim     = 4096
    num_heads   = 8
    num_kv_heads = 1
    head_dim    = 256
```

The critical constraint is that `num_heads`, `num_kv_heads`, and `head_dim` must
match across experts. This is enforced by assertions at the top of the
`Attention` module:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 165-167

```python
assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)
```

What *does* differ between experts:

| Parameter  | Expert 0 (PaliGemma) | Expert 1 (Action) | Why it can differ |
|------------|---------------------:|-------------------:|-------------------|
| `width`    | 2048                 | 1024               | Only affects Q/K/V input projection and FFN; attention output is in head-space |
| `mlp_dim`  | 16384                | 4096               | Entirely per-expert; no cross-expert interaction |
| Embedder   | Yes (vocab 257152)   | No                 | Action expert receives projected continuous vectors, not token IDs |

The parameter counts work out to roughly 2B and 311M respectively, with the
difference driven almost entirely by the 4x width gap flowing through 18 layers
of FFN weights.

---

## Dimension Summary (ASCII)

```
                 Expert 0 (PaliGemma 2B)          Expert 1 (Action 300M)
                 ========================         ======================

  Input width:        2048                              1024
                        |                                 |
  Q projection:   [8, 2048, 256]                   [8, 1024, 256]
                    = 8 heads                         = 8 heads
                    each 256-d                        each 256-d
                        |                                 |
  KV projection:  [2, 1, 2048, 256]                [2, 1, 1024, 256]
                    K=1 kv head                       K=1 kv head
                        |                                 |
                        +------------ CONCAT -------------+
                                  (seq dim)
                                     |
                            Joint Attention
                           (8 heads, 256 dim)
                                     |
                        +------------ SPLIT --------------+
                        |                                 |
  Output proj:    [8, 256, 2048]                   [8, 256, 1024]
                        |                                 |
                     2048-d out                       1024-d out
```

---

## Weight Naming Convention

The `_name()` helper controls all parameter names in the dual-expert stack:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 443-451

```python
def _name(name, i):
    # we name layers like this because we want the first expert's weights
    # to have no suffix (e.g., "attn"), so that they can be loaded
    # seamlessly from the existing PaliGemma checkpoint. subsequent experts
    # will have a suffix (e.g., "attn_1") and their weights will be
    # initialized from scratch.
    if i == 0:
        return name
    return f"{name}_{i}"
```

This produces the following weight names across the two experts:

| Component            | Expert 0 name              | Expert 1 name                |
|----------------------|----------------------------|------------------------------|
| Q projection         | `q_einsum`                 | `q_einsum_1`                 |
| KV projection        | `kv_einsum`                | `kv_einsum_1`                |
| Output projection    | `attn_vec_einsum`          | `attn_vec_einsum_1`          |
| Pre-attention norm   | `pre_attention_norm`       | `pre_attention_norm_1`       |
| Pre-FFN norm         | `pre_ffw_norm`             | `pre_ffw_norm_1`             |
| FFN (MLP)            | `mlp`                      | `mlp_1`                      |
| Final norm           | `final_norm`               | `final_norm_1`               |

The zero-suffix convention is not cosmetic. When loading a pretrained PaliGemma
checkpoint, expert 0's weights match the original parameter names exactly (no
renaming needed). Expert 1's weights, which carry the `_1` suffix, are absent
from the PaliGemma checkpoint and are initialized from scratch. This is the
mechanism that allows fine-tuning to start from a pretrained VLM backbone while
adding a new expert with random initialization.

---

## Shared Embedder

The token embedding table lives exclusively in expert 0:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 354-358

```python
self.embedder = Embedder(
    vocab_size=PALIGEMMA_VOCAB_SIZE,  # 257_152
    embed_dim=self.configs[0].width,  # 2048
    name="embedder",
)
```

Only expert 0 uses this embedder -- it maps discrete token IDs (language tokens)
to 2048-dimensional vectors and scales them by `sqrt(2048)`. The action expert
never touches the embedding table; its inputs are continuous action vectors that
have been projected to 1024 dimensions by a separate `action_in_proj` linear
layer outside the transformer.

In the PyTorch implementation, this same separation appears as:

**File:** `/tmp/openpi/src/openpi/models_pytorch/gemma_pytorch.py`, lines 56-58

```python
self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
self.gemma_expert.model.embed_tokens = None  # action expert has no embedder
```

---

## Instantiation and Depth Constraint

The `Module` class enforces that all experts share the same depth:

**File:** `/tmp/openpi/src/openpi/models/gemma.py`, lines 352-353

```python
# all experts must have the same depth
assert all(config.depth == self.configs[0].depth for config in self.configs)
```

Both experts are always 18 layers deep. The layers are wrapped in Flax's
`nn.scan`, which vectorizes the 18 identical blocks into a single set of stacked
parameters with a leading layer dimension (shape `[18, ...]`). This is the
"scanned layers" pattern that enables efficient parameter handling in JAX.

The two configs are passed together at construction time from `pi0.py`:

**File:** `/tmp/openpi/src/openpi/models/pi0.py`, lines 73-79

```python
llm = nnx_bridge.ToNNX(
    _gemma.Module(
        configs=[paligemma_config, action_expert_config],
        embed_dtype=config.dtype,
        adarms=config.pi05,
    )
)
```

---

## Key Takeaways

- The two experts differ in width (2048 vs 1024) and MLP dimension (16384 vs
  4096) but share identical attention geometry (8 query heads, 1 KV head,
  head_dim=256). This shared geometry is what makes joint attention possible.

- Expert 0's weights use unsuffixed names (e.g., `q_einsum`) matching the
  pretrained PaliGemma checkpoint. Expert 1's weights get a `_1` suffix (e.g.,
  `q_einsum_1`) and are initialized from scratch.

- The embedding table belongs only to expert 0. The action expert receives
  pre-projected continuous vectors, never discrete tokens.

- Both experts must have the same depth (18 layers), enforced by assertion. The
  layers are scanned (vectorized) for efficient JAX compilation.
