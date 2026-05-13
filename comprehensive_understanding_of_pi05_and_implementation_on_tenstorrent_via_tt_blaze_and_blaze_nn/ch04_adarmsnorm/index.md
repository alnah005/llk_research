# Chapter 4: adaRMSNorm and Timestep Conditioning

Pi0.5 injects diffusion timestep information into the action expert through
adaptive RMSNorm (adaRMSNorm) -- a conditional normalization layer that modulates
the hidden activations with learned scale, shift, and gate signals derived from
the denoising timestep. Standard RMSNorm, used by the PaliGemma expert, applies
only a learned per-feature scale. adaRMSNorm extends this by projecting a
timestep conditioning vector into three modulation components, enabling the
network to dynamically adjust its normalization behavior at every layer as a
function of how much noise remains. This chapter traces the full pipeline: from
the raw scalar timestep through sinusoidal encoding and an MLP, into the 37
adaRMSNorm instances that span the 18-layer action expert, and finally into the
TT-Blaze mapping challenges that arise when lowering this operation onto
Tenstorrent hardware.

---

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [01_adarmsnorm_mechanics.md](./01_adarmsnorm_mechanics.md) | RMSNorm vs adaRMSNorm: math, zero-init, gated residuals, instance counting |
| 2 | [02_timestep_embedding_pipeline.md](./02_timestep_embedding_pipeline.md) | Sinusoidal positional encoding, time MLP, `adarms_cond` broadcast semantics |
| 3 | [03_adarmsnorm_ttblaze_mapping.md](./03_adarmsnorm_ttblaze_mapping.md) | Existing `rmsnorm` op gap, proposed FusedOp, decomposition strategies, dual-output challenge |

---

## Key Takeaways (Chapter-Level)

- adaRMSNorm is used exclusively by the action expert (expert 1). The PaliGemma
  expert (expert 0) uses standard RMSNorm with `cond=None` at every layer. The
  `use_adarms` flag pattern `[False, True]` controls this split.

- Zero-initialization of both the standard RMSNorm `scale` parameter and the
  adaptive `Dense` projection ensures that at training start, adaRMSNorm behaves
  identically to standard RMSNorm -- the scale is 1, the shift is 0, and the
  gate is 0 (meaning no residual contribution initially).

- The timestep conditioning vector `adarms_cond` is a single `[B, 1024]` tensor
  produced once per forward pass and broadcast unchanged across all action
  tokens and all 18 transformer layers.

- The gate output from adaRMSNorm modulates the residual connection:
  `x + y * gate`, replacing the standard `x + y`. This gives each layer a
  learnable mechanism to suppress or amplify its own contribution.

- TT-Blaze's existing `rmsnorm` MicroOp does not support the scale/shift/gate
  modulation. Lowering adaRMSNorm requires either a composed decomposition
  (rmsnorm + matmul + split + element-wise ops) or a custom fused LLK SFPU
  kernel, with the additional challenge that the operation produces two outputs
  (normed activations and gate).
