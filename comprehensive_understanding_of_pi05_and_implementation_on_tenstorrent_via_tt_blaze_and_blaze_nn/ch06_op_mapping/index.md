# Chapter 6: Mapping pi0.5 Ops to TT-Blaze

## Context

pi0.5 is a dual-expert vision-language-action (VLA) model built on PaliGemma (2B) + an action expert (300M). It uses a flow-matching diffusion head for continuous action prediction and a SigLIP So400m/14 vision encoder. The model has two distinct execution paths: a prefix path (image embedding + language tokens with full bidirectional attention) and a suffix path (noisy action tokens with causal attention and adaRMSNorm timestep conditioning). At inference, the prefix is computed once and cached, and the suffix runs iteratively in a denoising loop.

This chapter maps every pi0.5 operation to TT-Blaze's existing op inventory, classifying each as REUSE (use as-is), ADAPT (modify an existing op), or NEW (build from scratch).

## Key Architectural Properties of pi0.5

- **Dual-expert transformer**: PaliGemma (width=2048, depth=18, 8 heads, 1 KV head, head_dim=256) and action expert (width=1024, depth=18, 8 heads, 1 KV head, head_dim=256) share attention within each layer but have independent FFN weights.
- **Gated GELU FFN**: Both experts use `gelu(x @ W_gate) * (x @ W_up) @ W_down` -- GELU, not SiLU.
- **adaRMSNorm**: Action expert uses adaptive RMSNorm conditioned on a timestep embedding. Produces scale, shift, and gate modulation vectors.
- **SigLIP vision encoder**: 27-layer ViT with LayerNorm (not RMSNorm), GELU MLP, sincos2d positional embeddings, patch size 14x14.
- **Prefix-LM attention**: Image + language tokens use bidirectional (prefix) attention; action tokens use causal (suffix) attention within a single sequence. KV cache holds the prefix for iterative denoising.
- **No MoE**: Unlike DeepSeek V3, pi0.5 is dense throughout.

## Chapter Contents

| File | Tier | Scope |
|------|------|-------|
| [01_reusable_ops_inventory.md](01_reusable_ops_inventory.md) | REUSE | Ops usable as-is from TT-Blaze |
| [02_ops_requiring_adaptation.md](02_ops_requiring_adaptation.md) | ADAPT | Ops that need parameter/mode changes |
| [03_new_ops_required.md](03_new_ops_required.md) | NEW | Ops that must be built from scratch |
| [04_patterns_from_deepseek_and_glm.md](04_patterns_from_deepseek_and_glm.md) | Reference | Transferable patterns from DeepSeek V3 and GLM-5.1 |

## Summary Classification

| Op | Tier | Notes |
|----|------|-------|
| matmul / linear | REUSE | QKV projections, output projections, FFN dense layers, action_in_proj, action_out_proj |
| rmsnorm | REUSE | PaliGemma pre-attention and pre-FFN norms, final norm |
| rope | REUSE | RoPE for Q and K in shared attention |
| sdpa | ADAPT | Needs prefix-LM masking support (bidirectional prefix + causal suffix in one sequence) |
| embedding | REUSE | Token embedding for language inputs |
| residual_add | REUSE | Standard residual connections in PaliGemma expert |
| kv_cache_update | REUSE | KV cache for iterative denoising |
| q_branch / q_heads | REUSE | Q projection pipeline |
| mcast / gather | REUSE | Activation distribution and collection |
| clamped_silu | -- | Not used (pi0.5 uses GELU, not clamped SiLU) |
| gated_reduce | ADAPT | Must support GELU activation (currently defaults to SiLU) |
| broadcast_rmsnorm | ADAPT | Needs adaRMSNorm variant with scale/shift/gate conditioning |
| dense_mlp / dense_swiglu | ADAPT | Needs GELU variant instead of SiLU/SwiGLU |
| adaRMSNorm | NEW | Fused scale/shift/gate modulation from timestep embedding |
| gated_residual | NEW | `x + y * gate` pattern used by adaRMSNorm |
| layernorm | NEW | Required for SigLIP ViT (not RMSNorm) |
| gelu activation | NEW | Standalone GELU for SigLIP MLP and Gemma FFN |
| sinusoidal_timestep_embedding | NEW | Sincos positional encoding for scalar timestep |
| siglip_patch_embed | NEW | Conv2D-as-matmul for 14x14 patch extraction |
| dual_expert_qkv_concat_split | NEW | Concatenate multi-expert QKV, split post-attention |

## Key Takeaways

1. **GELU is the critical activation gap.** pi0.5's Gemma backbone uses GELU everywhere (FFN gating, SigLIP MLP). TT-Blaze's gated_reduce, dense_swiglu, and DRAM-streaming matmul all default to SiLU. The `gated_reduce` MicroOp already has a `use_gelu` compile-time flag, so the kernel supports it -- but the full pipeline (DenseMLP, DenseSwiGLU) needs to propagate this parameter.

2. **adaRMSNorm is the single most important new op.** It appears in every layer of the action expert (2 per layer = 36 instances per forward pass). It must produce 3 modulation vectors from a single conditioning input and apply scale/shift/gate to the normalized activation.

3. **Prefix-LM masking changes the SDPA contract.** The current SDPA decode op assumes causal-only masking. pi0.5 needs a hybrid mask where prefix tokens attend bidirectionally and suffix tokens attend causally. This affects both the prefill phase and the denoising iterations.

4. **The dense (no-MoE) pipeline simplifies things.** DenseMLP already exists and handles the `RMSNorm -> Mcast -> Matmul -> GatedReduce -> Mcast -> DownProj` pattern. Adapting it to GELU and adaRMSNorm is simpler than building MoE routing.

5. **SigLIP is a separate subsystem.** The vision encoder runs once per inference call (not in the denoising loop) and uses a different normalization (LayerNorm), a different activation (GELU), and a different positional embedding (sincos2d). It can be implemented as a standalone pipeline.
