# Chapter 3: Dual-Expert Transformer with Shared Attention

Pi0.5 fuses a 2B-parameter vision-language backbone (PaliGemma) with a lightweight
300M-parameter action expert inside a single transformer stack. The two experts
share every attention computation -- queries, keys, and values from both streams
are concatenated along the sequence dimension, passed through one joint
self-attention layer, and then split back so each expert receives only its own
slice of the output. Everything else -- embeddings, feed-forward networks,
normalization parameters -- stays private to each expert. This chapter walks
through every layer of that design, from weight naming conventions to the
attention mask that stitches the two sequences together.

---

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [01_expert_configs_and_weight_layout.md](./01_expert_configs_and_weight_layout.md) | PaliGemma 2B vs Action Expert 300M: dimensions, weight naming, shared embedder |
| 2 | [02_shared_attention_mechanism.md](./02_shared_attention_mechanism.md) | Per-expert Q/K/V projections, concatenation, RoPE, GQA einsum, output splitting, KV cache during denoising |
| 3 | [03_attention_masking_and_positions.md](./03_attention_masking_and_positions.md) | Prefix-LM mask via `mask_ar` cumsum trick, position computation, denoising KV cache mask |
| 4 | [04_feedforward_and_gated_residual.md](./04_feedforward_and_gated_residual.md) | GeGLU FFN (not SwiGLU), `_gated_residual`, full block data-flow, 18-layer scan |

---

## Key Takeaways (Chapter-Level)

- Two experts of different widths (2048 vs 1024) share identical attention
  geometry (8 query heads, 1 KV head, head_dim=256), enabling concatenation
  along the sequence axis for joint self-attention.

- Weight naming uses a zero-suffix convention: expert 0 weights carry no suffix
  (e.g., `q_einsum`), expert 1 weights get `_1` (e.g., `q_einsum_1`). This lets
  expert 0's checkpoint load directly from a pretrained PaliGemma model.

- The attention mask implements prefix-LM semantics: prefix tokens (images +
  language) attend bidirectionally to each other, while action tokens see
  everything but are not seen by the prefix.

- The feed-forward network uses GeGLU (GELU gating), not SwiGLU, matching the
  original Gemma architecture.

- Adaptive RMSNorm (adaRMS) injects diffusion timestep conditioning into the
  action expert's normalization layers, producing per-token scale, shift, and
  gate modulations.
