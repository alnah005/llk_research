# Comprehensive Understanding of pi0.5 and Implementation on Tenstorrent via TT-Blaze and blaze-nn

## Final Synthesized Plan

---

## Audience

This guide is written for **Tenstorrent ML infrastructure engineers and kernel developers** who need to bring Physical Intelligence's pi0.5 vision-language-action model to Tenstorrent silicon using the TT-Blaze kernel composition framework and blaze-nn's PyTorch-compatible Module system. Readers are expected to already understand:

- Transformer architecture fundamentals: multi-head attention, RMSNorm, RoPE, GQA/MQA, KV caching, gated FFN activations (SwiGLU), embedding layers
- TT-Blaze's programming model: MicroOp/FusedOp hierarchy, CBHandle chaining, FusedProgram composition, the compose/emit API, pipeline infrastructure
- blaze-nn basics: the Module base class, Parameter, functional ops (`F.linear`, `F.rmsnorm`, `F.rope`), graph tracing vs. compose tracing contexts
- Tenstorrent hardware concepts: Tensix cores, L1 SRAM, circular buffers, DRAM banks, NOC, tile formats (bfloat16, bfloat8_b)
- How existing TT-Blaze models (DeepSeek V3 B1, GLM-5.1) are structured: pipeline stages, weight providers, KV cache management
- Python and PyTorch at a professional level; basic JAX/Flax familiarity is helpful since the reference implementations exist in both

Readers are **not** expected to know:

- The pi0 or pi0.5 model architectures
- Vision-Language-Action (VLA) models
- SigLIP vision encoders or PaliGemma
- Dual-expert shared-attention mechanisms
- Adaptive RMSNorm (adaRMSNorm) with timestep conditioning
- Flow-matching / diffusion-style generative models for action generation

**After reading, the reader should be able to:** trace the complete pi0.5 inference data flow from image pixels to action output, identify which TT-Blaze ops can be reused vs. must be written, design a FusedOp decomposition for dual-expert shared attention and adaRMSNorm, express the full model hierarchy in blaze-nn, implement the flow-matching denoising loop with KV cache reuse on the TT-Blaze pipeline system, and reason about performance and memory trade-offs for real-time robotics inference.

---

## Chapter List

### Chapter 1: pi0.5 Architecture Overview

**Description:** Introduces the pi0.5 model end-to-end, covering its purpose as a vision-language-action model for robotic control, the three major subsystems (SigLIP vision encoder, PaliGemma 2B backbone, Action Expert 300M), and the high-level data flow from sensor inputs to continuous action outputs.

**Directory:** `ch01_architecture_overview/`

**Files:**

- **`01_model_landscape.md`**
  - What pi0.5 is: Physical Intelligence's VLA model for robotic manipulation -- takes images + language instruction + robot state as input, outputs a trajectory of 50 action steps each with 32 action dimensions
  - The pi0.5 lineage: pi0 (predecessor) vs. pi0.5 -- what pi0.5 adds (adaRMSNorm for timestep conditioning, discrete state input via language tokens, extended max_token_len from 48 to 200)
  - The `pi05: bool` flag in `Pi0Config` and how it gates architectural differences
  - Total parameter count breakdown: SigLIP ~400M, PaliGemma 2B ~2B, Action Expert ~311M, projection layers ~few M; total ~2.3B
  - Configuration summary: `action_dim=32`, `action_horizon=50`, `max_token_len=200`, `paligemma_variant="gemma_2b"`, `action_expert_variant="gemma_300m"`, `dtype=bfloat16`

- **`02_component_map_and_data_flow.md`**
  - The four major components: SigLIP vision encoder, PaliGemma Gemma-2B backbone (expert 0), Action Expert Gemma-300M (expert 1), flow-matching denoiser
  - End-to-end data flow diagram: images (3 cameras x 224x224x3) -> SigLIP -> 768 image tokens; language prompt -> tokenizer -> Gemma embedder -> up to 200 language tokens; prefix = [image_tokens, language_tokens] (~968 tokens)
  - Suffix construction: noisy actions -> `action_in_proj` -> 50 action tokens; timestep -> sinusoidal encoding -> time MLP -> adaRMSNorm conditioning
  - Dual-expert transformer: 18 shared-attention layers processing prefix (PaliGemma only) then suffix (Action Expert with KV cache)
  - Denoising loop: 10 Euler steps from noise to clean actions, producing output shape `[B, 50, 32]`

- **`03_data_types_and_observation.md`**
  - The `Observation` dataclass: images dict (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb at 224x224), image_masks, state (batch, 32), tokenized_prompt (batch, 200), tokenized_prompt_mask
  - The `Actions` type: float32 tensor of shape (batch, action_horizon=50, action_dim=32)
  - How pi0.5 differs from pi0 in observation handling: state is part of discrete language tokens rather than a continuous suffix input
  - Gemma config details: both experts share `depth=18`, `num_heads=8`, `num_kv_heads=1`, `head_dim=256`; they differ in `width` (2048 vs 1024) and `mlp_dim` (16384 vs 4096)
  - The bfloat16 precision strategy and where float32 is preserved (normalization variance computation, sinusoidal embeddings)

### Chapter 2: SigLIP Vision Encoder

**Description:** Details the SigLIP So400m/14 vision transformer that converts raw camera images into visual token sequences consumed by the PaliGemma backbone, and immediately maps its components to TT-Blaze primitives.

**Directory:** `ch02_siglip_encoder/`

**Files:**

- **`01_siglip_architecture.md`**
  - SigLIP So400m/14 configuration: width=1152, depth=27, num_heads=16, mlp_dim=4304, patch_size=14x14
  - Patch extraction via Conv2D (kernel=14, stride=14, VALID padding) producing `[B, 256, 1152]` where 256 = 16x16 spatial tokens from 224x224 input
  - Sinusoidal 2D positional embeddings (`posemb_sincos_2d`): computed from grid positions, not learned
  - Encoder: 27 Encoder1DBlock layers, each with LayerNorm -> MultiHeadDotProductAttention (16 heads, head_dim=72, no GQA, full bidirectional attention) -> residual -> LayerNorm -> MLP (GELU activation, hidden=4304) -> residual
  - Pool type "none": all 256 spatial tokens preserved (no CLS token, no pooling)
  - Linear projection head: Dense(1152 -> 2048) mapping SigLIP width to PaliGemma width
  - Three cameras processed independently with shared weights: 3 x 256 = 768 total image tokens
  - Key distinction from Gemma blocks: SigLIP uses **LayerNorm** (not RMSNorm), **GELU** (not SwiGLU/GeGLU), standard MHA (not GQA), no RoPE, no causal masking

- **`02_siglip_to_ttblaze.md`**
  - Conv2D patch embedding: can be expressed as reshape (extract 14x14 patches into `[B, 256, 588]`) + matmul with `[588, 1152]` weight matrix; avoids needing a dedicated Conv2D kernel
  - **LayerNorm is NOT RMSNorm**: LayerNorm includes mean subtraction and a bias term; requires a new `layernorm` MicroOp or decomposition into mean-subtract + rmsnorm + bias-add
  - GELU activation: existing `clamped_silu` will not work; need a GELU op or `matmul_fused_act` with GELU support
  - Standard multi-head attention (not GQA): 16 heads, head_dim=72, all-to-all attention with no masking -- the existing `sdpa` op must support non-causal mode
  - Sinusoidal 2D positional embeddings: precomputed on host and transferred as a constant tensor; added via `residual_add`
  - SigLIP runs once per inference (not per denoising step), so it can be a standalone pipeline stage or preprocessed on host
  - Strategy options: (a) run SigLIP entirely on host CPU and transfer image tokens, (b) implement as a TT-Blaze pipeline stage with new/adapted ops, (c) use ttnn high-level ops for SigLIP and TT-Blaze FusedOps for the transformer
  - Memory considerations: ~400M params (~800MB at bf16), 27 layers with layer-scan pattern to reduce L1 pressure

### Chapter 3: Dual-Expert Transformer with Shared Attention

**Description:** Deep dive into the core architectural innovation of pi0.5 -- how PaliGemma 2B and Action Expert 300M share a single attention computation across 18 layers while maintaining independent weights, including the critical prefix-LM attention masking pattern.

**Directory:** `ch03_dual_expert_transformer/`

**Files:**

- **`01_expert_configs_and_weight_layout.md`**
  - PaliGemma 2B config: width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256
  - Action Expert 300M config: width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
  - Critical shared dimensions: both experts share depth=18, num_heads=8, num_kv_heads=1, head_dim=256 -- required for shared attention via Q/K/V concatenation
  - Weight naming convention: expert 0 weights have no suffix (`attn`, `pre_attention_norm`), expert 1 weights have `_1` suffix (`attn_1`, `pre_attention_norm_1`), preserving PaliGemma checkpoint compatibility
  - Shared embedder: only expert 0 has the Embedder (vocab_size=257152, embed_dim=2048); expert 1 receives action tokens via `action_in_proj`

- **`02_shared_attention_mechanism.md`**
  - Per-expert Q/K/V computation: each expert independently computes Q, K, V from its own hidden states using its own projection weights
    - Expert 0: `q_einsum` shape [8, 2048, 256], `kv_einsum` shape [2, 1, 2048, 256]
    - Expert 1: `q_einsum_1` shape [8, 1024, 256], `kv_einsum_1` shape [2, 1, 1024, 256]
  - Concatenation along sequence dimension: `q = concat(q_0, q_1, dim=seq)`, same for K and V; creates a joint sequence where all tokens can attend to all others (subject to masking)
  - RoPE application: applied to concatenated Q and K using combined position IDs, `max_wavelength=10000`
  - GQA structure: `q` reshaped from `[B, T, N, H]` to `[B, T, K=1, G=8, H=256]`; attention logits via einsum `BTKGH,BSKH->BKGTS`
  - Output splitting: after attention, output is split by sequence position ranges; each expert's slice is projected back via its own `attn_vec_einsum` / `attn_vec_einsum_1` to its respective width
  - KV cache interaction: during denoising, only Action Expert tokens generate Q/K/V; cached prefix K/V (from PaliGemma) are concatenated with fresh suffix K/V for attention
  - In PyTorch: uses HuggingFace Gemma layers with `eager_attention_forward`, `apply_rotary_pos_emb`

- **`03_attention_masking_and_positions.md`**
  - The prefix-LM masking scheme: `mask_ar` array determines attention pattern
    - Prefix tokens (images + language): `mask_ar=0` (bidirectional among themselves)
    - First action token: `mask_ar=1` (causal break from prefix)
    - Remaining action tokens: `mask_ar=0` (bidirectional among themselves)
  - `make_attn_mask(input_mask, mask_ar)`: cumulative-sum trick to convert `mask_ar` to a 2D boolean attention matrix
  - Result: prefix tokens attend to all prefix tokens; action tokens attend to all prefix + all action tokens; prefix tokens do NOT attend to action tokens
  - Position computation: `positions = cumsum(input_mask) - 1`, handling padding correctly
  - During denoising with KV cache: combined mask `[B, suffix_len, prefix_len + suffix_len]` where prefix portion is `prefix_pad_masks` broadcast, suffix portion uses the prefix-LM mask
  - Suffix positions start from `sum(prefix_mask)` and increment, ensuring correct RoPE alignment with cached prefix

- **`04_feedforward_and_gated_residual.md`**
  - Per-expert FFN: Gemma uses **GeGLU** (GELU-gated linear unit), NOT SwiGLU
    - Expert 0 FFN: width=2048 -> `gating_einsum` [2, 2048, 16384] with GELU, then `linear` [16384, 2048]
    - Expert 1 FFN: width=1024 -> `gating_einsum` [2, 1024, 4096] with GELU, then `linear` [4096, 1024]
    - Formula: `output = (GELU(x @ W_gate) * (x @ W_up)) @ W_down`
  - The `_gated_residual(x, y, gate)` function: if `gate is None`, returns `x + y` (standard residual for PaliGemma); if gate present, returns `x + y * gate` (modulated residual for Action Expert)
  - Applied twice per block: once after attention (gate from pre-attention adaRMSNorm) and once after FFN (gate from pre-FFN adaRMSNorm)
  - The `Block.__call__()` flow: pre-attention RMSNorm per expert -> shared Attention -> gated residual -> pre-FFN RMSNorm per expert -> per-expert FFN (GeGLU) -> gated residual
  - The `Module.__call__()` flow: shared Embedder, `nn.scan` over 18 layers, per-expert final RMSNorm

### Chapter 4: adaRMSNorm and Timestep Conditioning

**Description:** Covers the adaptive RMSNorm mechanism unique to pi0.5, where the flow-matching timestep conditions every normalization layer in the Action Expert via learned scale/shift/gate modulation, and the complete conditioning signal pipeline.

**Directory:** `ch04_adarmsnorm/`

**Files:**

- **`01_adarmsnorm_mechanics.md`**
  - Standard RMSNorm (expert 0, `cond=None`): `normed = x * rsqrt(mean(x^2) + eps)`, then `output = normed * (1 + scale)` where `scale` is a learned per-feature parameter initialized to zero. Returns `(output, None)` -- no gate.
  - Adaptive RMSNorm (expert 1, `cond != None`): `modulation = Dense(cond, dim * 3)` where Dense is zero-initialized. Split into `(scale, shift, gate)` each of shape `[B, 1, D]`. Output: `normed * (1 + scale) + shift`. Returns `(output, gate)`.
  - Zero initialization ensures adaRMSNorm starts as standard RMSNorm at training init (scale=0, shift=0, gate=0), preserving pretrained behavior
  - The gate modulates residual connections: early in denoising (high noise), the model may use different sub-layer contributions than late in denoising (low noise) -- a form of adaptive computation depth
  - adaRMSNorm is applied at both pre-attention and pre-FFN positions in each of 18 layers, plus the final norm = 37 instances total for the Action Expert
  - Only the Action Expert uses adaRMSNorm (`use_adarms=[False, True]`); PaliGemma always uses standard RMSNorm

- **`02_timestep_embedding_pipeline.md`**
  - Timestep encoding: sinusoidal positional embedding via `posemb_sincos(t, dim=1024, min_period=4e-3, max_period=4.0)` producing `[B, 1024]`
  - Frequency computation: `period = min_period * (max_period / min_period) ^ fraction` where fraction is linearly spaced [0, 1] over D/2 dimensions
  - Time MLP (pi0.5 only): `time_emb -> time_mlp_in (Linear 1024->1024) -> SiLU -> time_mlp_out (Linear 1024->1024) -> SiLU` producing `adarms_cond` of shape `[B, 1024]`
  - The `adarms_cond` is NOT per-token: it is `(batch, 1024)`, broadcast across all action tokens and all 18 layers (same conditioning per layer, unlike DiT)
  - Contrast with pi0: in pi0, timestep is concatenated with action tokens via `action_time_mlp_in (Linear 2048->1024) -> SiLU -> action_time_mlp_out (Linear 1024->1024)`; no adaRMSNorm used
  - The `adarms_cond` list pattern: `[None, adarms_cond]` -- first element for PaliGemma (no conditioning), second for Action Expert

- **`03_adarmsnorm_ttblaze_mapping.md`**
  - Existing `rmsnorm` MicroOp computes standard RMSNorm but does NOT support scale/shift/gate modulation -- a new op is required
  - Proposed `AdaRMSNorm` FusedOp specification: inputs are `(x, cond, modulation_weights)`; outputs are `(normed_output, gate)`
  - Decomposition strategy A (composed): rmsnorm(x) -> matmul(cond, modulation_weights) -> split3 -> element-wise scale/shift/gate application
  - Decomposition strategy B (fused kernel): custom LLK SFPU kernel that fuses normalize + matmul + modulate in a single pass, avoiding intermediate CB reads/writes
  - The dual-output challenge: FusedOps typically return a single CBHandle, but adaRMSNorm must return both the normalized output and the gate; solution via split output CBs or tuple return
  - Memory layout: modulation_weights are `[1024, 3072]` per norm per layer; conditioning is `[B, 1024]` broadcast across cores
  - Comparison to `broadcast_rmsnorm`: similar structure but with additional shift and gate outputs

### Chapter 5: Flow-Matching Inference and the Denoising Loop

**Description:** Explains the flow-matching framework for action generation, the complete inference data flow with tensor shapes at every boundary, and the KV caching strategy that avoids redundant prefix computation across denoising steps.

**Directory:** `ch05_flow_matching/`

**Files:**

- **`01_flow_matching_theory.md`**
  - Flow matching as a generative framework: learn a velocity field `v(x_t, t)` that transports samples from noise distribution (t=1) to target distribution (t=0)
  - Training objective: given clean actions `a` and noise `n`, construct `x_t = t * n + (1-t) * a`, target velocity `u_t = n - a`, minimize `MSE(v_t, u_t)`
  - Inference: start from `x_1 ~ N(0, I)`, iterate `x_{t+dt} = x_t + dt * v(x_t, t)` with `dt = -1/num_steps`
  - Convention note: codebase uses `t=1` for noise and `t=0` for clean data (opposite of pi0 paper)
  - Training time sampling: `t ~ Beta(1.5, 1) * 0.999 + 0.001`, biasing toward higher noise levels
  - Default 10 denoising steps: balances action quality and latency for real-time robotics

- **`02_complete_inference_walkthrough.md`**
  - Step-by-step walkthrough of `sample_actions` with tensor shapes at every boundary:
    1. **Preprocess observation**: normalize images to [-1, 1]
    2. **Embed prefix**: SigLIP encodes 3 images -> `[B, 768, 2048]` image tokens; Gemma embedder encodes language -> `[B, 200, 2048]` (scaled by sqrt(2048)); concatenate -> `[B, ~968, 2048]` prefix tokens; build masks (`mask_ar=[0,...,0]` for bidirectional prefix)
    3. **Prefix forward pass**: `PaliGemma.llm([prefix_tokens, None], mask, positions)` -- only expert 0 runs; produces KV cache `(k, v)` each `[18, B, ~968, 1, 256]`
    4. **Initialize noise**: `x_1 = N(0, I)` of shape `[B, 50, 32]`, `time = 1.0`, `dt = -0.1`
    5. **Denoising loop** (10 iterations):
       a. `embed_suffix`: project `x_t` via `action_in_proj (Linear 32->1024)` -> `[B, 50, 1024]`; encode timestep via sinusoidal + time MLP -> `adarms_cond [B, 1024]`
       b. Suffix AR mask: `[True, False, ..., False]` (50 elements)
       c. Build combined mask: suffix self-attention `[B, 50, 50]` + prefix cross-attention `[B, 50, ~968]` -> `[B, 50, ~1018]`
       d. Suffix positions: `prefix_len + cumsum(suffix_mask) - 1`
       e. Forward: `PaliGemma.llm([None, suffix_tokens], mask, positions, kv_cache, adarms_cond=[None, adarms_cond])` -> suffix output
       f. Extract last 50 tokens, project via `action_out_proj (Linear 1024->32)` -> `v_t [B, 50, 32]`
       g. Euler step: `x_t = x_t + dt * v_t`, `time += dt`
    6. **Return**: `x_0 [B, 50, 32]` -- 50 action steps, each 32-dimensional

- **`03_kv_cache_strategy.md`**
  - KV cache dimensions: per layer K and V each `[B, ~968, 1, 256]` in bfloat16; 18 layers total = `18 * 2 * 968 * 256 * 2 bytes = ~17.9 MB` per batch element
  - Prefix caching: the prefix is processed once through the dual-expert transformer with only PaliGemma active; KV cache is written and then reused for all 10 denoising steps
  - Suffix K/V are NOT cached between denoising steps: they change at every step because the action tokens change; suffix K/V are computed fresh each step and concatenated with cached prefix K/V for SDPA
  - The dual-expert KV subtlety: during prefix forward, only PaliGemma (expert 0) generates K/V; during suffix forward, only Action Expert (expert 1) generates K/V; shared attention concatenates cached PaliGemma K/V with fresh Action Expert K/V
  - Comparison with LLM KV cache: LLM inference appends to the KV cache at each decode step; pi0.5 NEVER appends -- the prefix cache is static and read-only during denoising

- **`04_pytorch_implementation_differences.md`**
  - The PyTorch implementation uses HuggingFace Transformers `PaliGemmaForConditionalGeneration` and `GemmaForCausalLM` as backbones, patched with adaRMSNorm support via `transformers_replace`
  - `PaliGemmaWithExpertModel`: wraps both models, implements dual-expert forward by iterating through layers jointly
  - Key differences from JAX: attention mask format (PyTorch uses 4D float mask with -2.38e38 for masked positions vs JAX boolean mask), float64 precision for sinusoidal embedding on CPU, `torch.compile` with `max-autotune` mode
  - Weight format differences: HuggingFace `(out_features, in_features)` vs JAX einsum format -- affects how weights map to TT-Blaze
  - Gradient checkpointing support in PyTorch for training (not relevant for inference but useful context)

### Chapter 6: Mapping pi0.5 Ops to TT-Blaze

**Description:** Systematically maps every computational primitive in pi0.5 to existing TT-Blaze ops using a three-tier classification (REUSE / ADAPT / NEW), identifies gaps requiring new implementations, and integrates lessons from DeepSeek V3 and GLM-5.1 patterns.

**Directory:** `ch06_op_mapping/`

**Files:**

- **`01_reusable_ops_inventory.md`**
  - **matmul** / **matmul_cb** / **dram_streaming_matmul**: all linear projections -- Q/K/V, output, FFN gate/up/down, action_in_proj, action_out_proj, time MLP, SigLIP Dense layers, adaRMSNorm modulation Dense. Core workhorse; different shapes but same `[M,K]x[K,N]` pattern.
  - **rmsnorm** / **padded_rmsnorm**: standard RMSNorm for PaliGemma expert (expert 0) at pre-attention, pre-FFN, and final norm positions. Directly reusable.
  - **rope**: rotary position embedding for Q and K after concatenation. Same formulation as DeepSeek/Gemma. Directly reusable.
  - **sdpa** + **pre_sdpa** + **post_sdpa**: scaled dot-product attention for the joint attention computation. Supports GQA (num_kv_heads=1, num_heads=8). Needs evaluation for non-causal masking support (see adapted ops).
  - **embedding** / **embed_mcast**: language token embedding lookup (vocab_size=257152, embed_dim=2048). Directly reusable.
  - **residual_add**: standard residual connections `x + y` for PaliGemma expert. Directly reusable.
  - **kv_cache_update** / **kv_branch**: KV cache write for prefix caching. Directly reusable.
  - **q_branch** / **q_heads** / **create_q_heads**: query head manipulation for GQA (8 query heads, 1 KV head). Directly reusable.
  - **mcast** / **gather**: broadcasting and collection across core grids. Directly reusable.
  - **clamped_silu**: SiLU activation for time MLP. Directly reusable (may need unclamped variant).

- **`02_ops_requiring_adaptation.md`**
  - **gated_reduce / swiglu -> GeGLU**: Gemma FFN uses GELU gating, not SiLU. The existing `swiglu`/`gated_reduce` ops assume SiLU activation; need a configurable activation parameter or a new `geglu` variant. Check if `matmul_fused_act` supports GELU.
  - **sdpa for prefix-LM masking**: the existing SDPA supports causal masking; pi0.5 uses prefix-LM masking where prefix tokens attend bidirectionally. The prefill pass can use bidirectional attention; the decode pass needs the prefix-LM mask. This may require a custom mask parameter or a modified masking kernel.
  - **sdpa for dual-expert concatenated sequences**: during denoising, query length (50 suffix tokens) differs from KV length (968 prefix + 50 suffix = 1018). This is the standard KV-cached decode pattern, likely already supported.
  - **broadcast_rmsnorm**: CCL broadcast + RMSNorm fusion. Reusable for multi-device deployment of pi0.5, but cannot handle adaRMSNorm's scale/shift/gate modulation -- serves as a foundation for the new adaRMSNorm op.
  - **dense_mlp / dense_mlp_dram / dense_swiglu**: fused MLP ops assume SwiGLU; need GELU variant for Gemma's gated-GELU FFN.

- **`03_new_ops_required.md`**
  - **adaRMSNorm** (new FusedOp): extends RMSNorm with conditioning-based scale/shift/gate modulation. Inputs: activation tensor + conditioning vector (or pre-projected scale/shift/gate). Outputs: modulated activation + gate tensor. The conditioning Dense projection (`cond -> 3*width`) can be fused into this op or kept as a separate matmul. Must handle float32 variance computation.
  - **gated_residual** (new MicroOp): `x + y * gate` where gate is `[B, 1, width]` broadcast over sequence dimension. Could compose from `eltwise_mul` + `residual_add`, or fuse into a single kernel to avoid extra CB read/write.
  - **layernorm** (new MicroOp): full LayerNorm for SigLIP encoder -- includes mean subtraction, variance normalization, learned scale and bias. Distinct from RMSNorm which skips mean subtraction.
  - **gelu** (new activation): GELU activation for SigLIP MLP and potentially for Gemma FFN gating if `geglu` is decomposed. Currently only SiLU/clamped_silu exist.
  - **sinusoidal_timestep_embedding**: compute `sin/cos(t / period)` for logarithmically spaced periods. Small computation (`[B, 1024]`), likely cheaper to compute on host and transfer.
  - **siglip_patch_embed**: Conv2d with kernel=14, stride=14. Can be implemented as reshape + matmul (preferred) or dedicated kernel.
  - **dual_expert_qkv_concat_split**: orchestration pattern for (1) per-expert Q/K/V projection, (2) concatenation along sequence dimension, (3) post-attention split back to per-expert tensors. May be better as a composite pattern than a single FusedOp.

- **`04_patterns_from_deepseek_and_glm.md`**
  - **DeepSeek V3 dense layer pattern**: `RMSNorm -> Mcast -> KNSlicedMatmul (gate+up) -> GatedReduce -> Mcast -> DownProj -> Gather -> ReduceToOne -> ResidualAdd` -- directly transferable to pi0.5's Gemma FFN (with GELU substitution)
  - **DeepSeek V3 pipeline stage decomposition**: `EmbeddingStage -> DenseDecoderStage/MoEDecoderStage -> LMHeadStage` pattern maps to pi0.5's `SigLIPStage -> DualExpertDecoderStage -> ActionHeadStage`
  - **Weight provider pattern**: `CacheWeightProvider`, `StateDictWeightProvider`, `SyntheticWeightProvider` with safetensors loading, transform functions (transpose, slicing), and device tensor creation -- directly applicable
  - **KV cache management**: DeepSeek V3's `KVCacheUpdate` op and DRAM-streaming SDPA pattern transfer to pi0.5's prefix caching; pi0.5 is simpler (read-only cache during denoising)
  - **DRAM-streaming matmul**: triple-buffered DRAM read with column-major tile reordering -- directly usable for PaliGemma's large FFN weights (2048x16384)
  - **GLM-5.1 FusedOp patterns**: `glm5_fused_proj`, `glm5_q_branch`, `glm5_sdpa_post_sdpa` show how to fuse QKV projection with attention -- applicable to dual-expert attention design
  - **Key difference from existing models**: pi0.5 has NO MoE layers; its complexity comes from dual-expert shared attention (always-on, no routing) rather than expert routing

### Chapter 7: FusedOp Design and blaze-nn Expression

**Description:** Presents concrete FusedOp designs for pi0.5's novel components with CB chain analysis, then shows how to express the complete model as a blaze-nn Module hierarchy with functional ops, weight loading, and inference orchestration.

**Directory:** `ch07_fusedop_and_blazenn/`

**Files:**

- **`01_adarmsnorm_fusedop.md`**
  - **Composed design**: `Matmul(cond, modulation_weights) -> Split3Way -> RMSNorm(input) -> EltwiseMul(normed, 1+scale) -> ResidualAdd(scaled, shift) -> output; gate forwarded to caller`
  - **Fused kernel design**: custom MicroOp with new LLK SFPU kernel that fuses normalize + scale + shift in a single pass, avoiding intermediate CB reads/writes
  - CBHandle chain: `cond_cb -> matmul_cb (3*width) -> [scale_cb, shift_cb, gate_cb] (each width) -> rmsnorm_cb -> mul_cb -> add_cb -> output_cb; gate_cb forwarded to caller`
  - L1 memory estimate: for action expert width=1024, modulation weights are 1024 x 3072; intermediate CBs hold 1024-wide tiles
  - Recommended approach: composed design for initial implementation (faster development, easier debugging), fused kernel for optimization phase

- **`02_dual_expert_block_fusedop.md`**
  - **Option A -- Separate FusedOps sharing SDPA**: each expert has its own RMSNorm/adaRMSNorm -> Q/K/V FusedOp, outputs concatenated in L1, fed to shared SDPA FusedOp, output split, each expert has its own output projection -> gated residual -> FFN FusedOp
  - **Option B -- Monolithic DualExpertBlock FusedOp**: single large FusedOp encapsulating the entire block for both experts. Pro: holistic CB allocation optimization. Con: very complex, hard to debug.
  - **Option C -- Separate expert pipelines with synchronization**: expert 0 and expert 1 run on separate core grids, synchronize at attention concatenation point
  - **Recommended approach**: Option A with explicit CB management at concatenation/split boundaries. The existing DSA patterns from DeepSeek V3 provide precedent for multi-source attention fusion.
  - CB flow: Expert 0 RMSNorm writes to CB0, expert 0 QKV matmul reads CB0 writes to CB1. Expert 1 adaRMSNorm writes to CB2, expert 1 QKV matmul reads CB2 writes to CB3. Concatenate CB1+CB3 -> CB4 for SDPA. Split SDPA output back to per-expert CBs.
  - The concatenation challenge: Q/K/V from two experts have different input widths (2048 and 1024) but are projected to the SAME head_dim (256), so concatenation operates on equal-width tensors after projection
  - Alternative split: `DualExpertPreAttn` + `SharedSDPA` + `DualExpertPostAttn` for simpler CB management and debugging

- **`03_module_hierarchy_and_forward_pass.md`**
  - Top-level `Pi05Module(Module)` hierarchy:
    ```
    Pi05Module
    +-- siglip: SigLIPEncoder(Module)
    |   +-- patch_embed: PatchEmbedding(Module)
    |   +-- blocks: ModuleList([SigLIPBlock x 27])
    |   +-- head_proj: Linear(Module)
    +-- embedder: Embedder(Module)
    +-- layers: ModuleList([DualExpertBlock x 18])
    |   +-- pre_attn_norm_0: RMSNormModule
    |   +-- pre_attn_norm_1: AdaRMSNormModule
    |   +-- attn: DualExpertAttention(Module)
    |   +-- pre_ffn_norm_0: RMSNormModule
    |   +-- pre_ffn_norm_1: AdaRMSNormModule
    |   +-- ffn_0: GeGLUModule (width=2048, mlp_dim=16384)
    |   +-- ffn_1: GeGLUModule (width=1024, mlp_dim=4096)
    +-- final_norm_0: RMSNormModule
    +-- final_norm_1: AdaRMSNormModule
    +-- action_in_proj: Linear(Module)  # 32->1024
    +-- action_out_proj: Linear(Module) # 1024->32
    +-- time_mlp: TimeMLP(Module)       # sinusoidal + 2x Linear+SiLU
    ```
  - Existing blaze-nn functional ops usable: `F.linear`, `F.rmsnorm`, `F.rope`, `F.residual_add`, `F.mcast`, `F.gather`, `F.gated_reduce` (needs GELU mode)
  - New functional ops needed:
    - `F.ada_rmsnorm(input, gamma, cond, modulation_weight)` -> `(normed_output, gate)`
    - `F.gated_residual(x, y, gate)` -> `x + y * gate` (or `x + y` if gate is None)
    - `F.layernorm(input, gamma, beta, epsilon)` -> normalized output
    - `F.gelu(input)` -> GELU activation
    - `F.sdpa(q, k, v, mask, scale)` -> attention output
    - `F.concat(tensors, dim)` / `F.split(tensor, split_sizes, dim)` -> for dual-expert Q/K/V joining/splitting
    - `F.sinusoidal_embed(timestep, dim, min_period, max_period)` -> embedding vector (or host-side)
  - Each new functional op dispatches through `_dispatch()` to the active tracing context, mapping to corresponding TT-Blaze ops

- **`04_weight_loading_and_conversion.md`**
  - Source format: HuggingFace safetensors (PaliGemma + pi0.5 fine-tuned checkpoint)
  - Key name mappings:
    - PaliGemma: `paligemma.language_model.model.layers.{i}.self_attn.q_proj.weight` -> `layers.{i}.attn.q_weight_0`
    - Action Expert: `gemma_expert.model.layers.{i}.self_attn.q_proj.weight` -> `layers.{i}.attn.q_weight_1`
    - SigLIP: `paligemma.vision_tower.vision_model.encoder.layers.{i}.*` -> `siglip.blocks.{i}.*`
    - adaRMSNorm: `model.layers.{i}.input_layernorm.ada_dense.weight` -> `layers.{i}.pre_attn_norm_1.modulation_weight`
  - Weight format transformations:
    - HuggingFace Linear `(out_features, in_features)` -> TT-Blaze matmul `(in_features, out_features)` (transpose needed)
    - Gemma einsum weights `(num_heads, width, head_dim)` -> may need reshape for standard Linear format
    - SigLIP Conv2d `(out_channels, in_channels, kH, kW)` -> reshape to `(in_channels*kH*kW, out_channels)` for matmul-based patch extraction
    - GatedFFN: Flax stores `(2, width, mlp_dim)` gating_einsum; PyTorch stores separate `gate_proj` and `up_proj`
  - WeightProvider pattern: adapt DeepSeek V3's `StateDictWeightProvider` -- load safetensors, apply transforms, create device tensors
  - Quantization: bf16 primary; BFP8 quantization would reduce memory (~2.8 GB total) and DRAM streaming bandwidth significantly
  - LoRA support: reference supports LoRA variants; inference-only can merge LoRA weights; blaze-nn would need LoRA-aware linear layers only if fine-tuning on device

- **`05_inference_orchestration.md`**
  - Two-phase inference in blaze-nn:
    - **Phase 1 (Prefix)**: `siglip_encoder(images)` -> image tokens; `gemma_embedder(lang_tokens)` -> language embeddings; concatenate; run through `DualExpertTransformer` with PaliGemma expert only (`[prefix_tokens, None]`); capture KV cache
    - **Phase 2 (Denoising loop, 10 iterations)**: `action_in_proj(noisy_actions)` -> action embeddings; `timestep_mlp(sinusoidal_embed(t))` -> adaRMS conditioning; run through `DualExpertTransformer` with Action Expert only (`[None, suffix_tokens]`) + KV cache; `action_out_proj(output)` -> velocity; Euler step on host
  - Host-side loop control: the 10-step loop is orchestrated from the host, calling the denoising stage repeatedly with updated `x_t` and `time`
  - Handling the `None`-expert pattern: when one expert is None, skip its projections and FFN, pass only the active expert's tokens through shared attention (with cached K/V from the other expert)

### Chapter 8: Pipeline Architecture, Performance, and Implementation Roadmap

**Description:** Addresses how pi0.5 maps to Tenstorrent hardware at the system level, covering pipeline stage design, memory budgets, latency analysis for real-time robotics, and a phased implementation roadmap with technical risks.

**Directory:** `ch08_pipeline_performance_roadmap/`

**Files:**

- **`01_pipeline_stage_design.md`**
  - pi0.5 inference has a fundamentally different structure from autoregressive LLM decoding: no autoregressive token generation, instead a prefix-once-then-iterate pattern
  - Proposed pipeline stages:
    - **Stage 0 -- SigLIP Encoding**: process 3 camera images through SigLIP, produce 768 image tokens. Runs once per observation.
    - **Stage 1 -- Prefix Forward + KV Cache**: concatenate image + language tokens, run through 18 dual-expert layers (PaliGemma only), cache K/V. Runs once.
    - **Stage 2 -- Denoising Loop**: 10 iterations of suffix embedding + 18 dual-expert layers (Action Expert only with KV cache) + action output projection + Euler step. Latency-critical.
    - **Stage 3 -- Action Output**: extract final `[B, 50, 32]` actions and return to host.
  - The denoising loop challenge: three execution options
    - Option A (Host-orchestrated): host drives 10 iterations, sending suffix through pipeline each time. Simplest but incurs host-device roundtrip per step.
    - Option B (Device-loop): single program that loops 10 times over 18 layers internally. Minimizes host overhead but requires device-side loop control.
    - Option C (Unrolled pipeline): treat each denoising step as a pipeline stage. High device utilization but 10x weight replication.
    - **Recommended**: Option A with overlap of host-side timestep embedding computation and device-side transformer execution
  - Comparison to DeepSeek V3: DeepSeek uses 62 pipeline stages across 16+ devices; pi0.5 is simpler (18 layers, no MoE) but has the novel iterative decode pattern

- **`02_memory_and_latency_analysis.md`**
  - Parameter memory: PaliGemma ~4 GB (bf16), Action Expert ~622 MB (bf16), SigLIP ~800 MB (bf16), adaRMSNorm modulation weights ~226 MB, total ~5.6 GB. In BFP8: ~2.8 GB.
  - KV cache: ~17.9 MB per batch element for prefix at 968 tokens (compact due to GQA with 1 KV head). Fits easily in DRAM.
  - L1 budget: per Tensix core ~1.5 MB (Wormhole). Weights in DRAM with streaming to L1; KV cache in DRAM with SDPA accessing via DRAM reads; activations in L1/CB.
  - DRAM budget: 12 GB per device (Wormhole). Full model fits in DRAM of a single device.
  - Total inference compute: 1 SigLIP forward (3 images x 27 layers, ~400M params) + 1 prefix forward (18 layers, ~2B params, ~968 tokens) + 10 x denoising step (18 layers, ~311M params, ~50 tokens)
  - Latency target: 10-50 Hz for robotics; need total inference < 100ms for 10Hz, < 33ms for 30Hz
  - Bandwidth-bound analysis for denoising: Action Expert weights per layer ~34MB, streamed from DRAM 10 times = 180 layer reads; at ~200 GB/s DRAM bandwidth -> ~17ms for weight streaming alone
  - Optimization opportunities: reduce denoising steps (distillation to 1-4 steps), quantize to BFP8/BFP4, keep Action Expert weights in L1 across cores (~622MB distributed), batch multiple robots, fuse adaRMSNorm + attention to reduce CB round-trips

- **`03_implementation_roadmap.md`**
  - **Phase 1 -- Standalone op development** (highest priority):
    1. AdaRMSNorm FusedOp: implement and test against PyTorch reference
    2. GatedResidual MicroOp: implement and test
    3. GeGLU activation: adapt SwiGLU or MatmulFusedAct for GELU
    4. SinusoidalTimestepEmbedding: implement as host-side computation or device kernel
    5. Validate each op individually against PyTorch reference in `pi0_pytorch.py`
  - **Phase 2 -- Transformer block integration**:
    1. Single DualExpertBlock: compose one full block from Phase 1 ops + existing ops, test layer-by-layer against PyTorch
    2. 18-layer stack: compose all layers, verify end-to-end transformer output
    3. KV cache integration: implement prefix forward with KV cache write, suffix forward with KV cache read
  - **Phase 3 -- Vision encoder**:
    1. SigLIP Conv2d patch extraction: evaluate reshape+matmul vs. ttnn.conv2d vs. custom kernel
    2. SigLIP encoder blocks: compose 27 ViT blocks (requires LayerNorm, GELU, non-GQA attention)
    3. End-to-end SigLIP: test against HuggingFace SigLIP reference
  - **Phase 4 -- Full model integration**:
    1. Complete pi0.5 inference: SigLIP + prefix + 10-step denoising + action output
    2. Numerical validation: compare action outputs against PyTorch reference within acceptable tolerance
    3. blaze-nn Module implementation: express full model in blaze-nn and validate
  - **Phase 5 -- Performance optimization**:
    1. Pipeline stage decomposition and latency measurement
    2. CB size tuning, L1/DRAM partitioning optimization
    3. Denoising step fusion opportunities
    4. Real-time robotics benchmark: end-to-end latency measurement

- **`04_technical_risks_and_open_questions.md`**
  - **SDPA non-causal mask support**: the existing SDPA op may be hardcoded for causal masking; pi0.5 needs prefix-LM masking (bidirectional for prefix). Does the SDPA op support arbitrary mask patterns?
  - **Dual-expert width mismatch and tile alignment**: PaliGemma tokens are 2048-wide, Action Expert tokens are 1024-wide. How does this affect tile alignment and CB sizing on Tensix? The Q/K/V projections map to head_dim=256 which resolves the width mismatch for attention, but per-expert matmuls need different core grids.
  - **Conv2d for SigLIP**: the 14x14 stride-14 patch extraction is the only convolution in the entire model. Is the reshape+matmul approach sufficient, or does it lose accuracy/performance?
  - **Full attention in SigLIP**: SigLIP uses non-causal attention across 256 tokens per image. If the existing SDPA op only supports causal masking, SigLIP requires a different attention implementation.
  - **Variable-length prefix**: different observations may produce different effective prefix lengths (padded images, variable-length language). Does the KV cache and attention mask handling support dynamic prefix lengths?
  - **blaze-nn multi-branch tracing**: the current GraphTracingContext may assume a linear DAG; dual-expert computation with merge/split requires multi-branch tracing capability
  - **Numerical precision over 10 Euler steps**: adaRMSNorm modulation factors in bf16, accumulated through 10 ODE integration steps, may cause drift. Testing with fp32 accumulation may be necessary.
  - **Denoising loop on pipeline system**: TT-Blaze's pipeline infrastructure assumes forward pass through stages; the 10-step loop requires iterating the Action Expert stages. Is this handled like decode token generation, or does it require a new looping construct?

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **pi0.5** | Physical Intelligence's upgraded VLA model with adaRMSNorm timestep conditioning and discrete state input. Successor to pi0. |
| **VLA** | Vision-Language-Action model: takes visual observations + language instructions as input, produces continuous robot actions as output. |
| **PaliGemma / Expert 0** | The Gemma 2B backbone that processes prefix tokens (images + language). Also called the "VLM expert" or "backbone." Weights have no name suffix. |
| **Action Expert / Expert 1** | The Gemma 300M model that processes suffix tokens (actions) with adaRMSNorm conditioning. Weights have `_1` name suffix. |
| **Prefix** | The concatenation of all image tokens (from SigLIP) and language tokens (from Gemma embedder). Computed once per inference and cached. Up to ~968 tokens. |
| **Suffix** | The action tokens (from `action_in_proj` applied to noisy actions). Recomputed at each denoising step. 50 tokens in pi0.5. |
| **Dual-expert shared attention** | Joint attention where Q/K/V from both experts are concatenated along the sequence dimension, attention is computed jointly, and outputs are split back to per-expert tensors for per-expert output projection. |
| **adaRMSNorm** | Adaptive RMS normalization: timestep-conditioned normalization that applies learned scale, shift, and gate modulations derived from the flow-matching timestep embedding. Used only in the Action Expert. Returns `(normed_output, gate)`. |
| **Gated residual** | The residual connection pattern `x + y * gate` used with adaRMSNorm, where gate modulates how much of the sublayer output is added to the residual stream. Standard residual (`x + y`) is used when gate is None (PaliGemma). |
| **Flow matching** | A generative modeling framework where a neural network learns a velocity field that transports samples from a noise distribution to a data distribution along straight-line paths. |
| **Denoising step** | One Euler integration step in the flow-matching ODE: `x_{t+dt} = x_t + dt * v_t`. pi0.5 uses 10 steps by default (dt = -0.1). |
| **Prefix-LM masking** | Attention mask pattern where prefix tokens attend bidirectionally to each other, but suffix tokens use a causal boundary at the first action token. Prefix tokens cannot attend to suffix tokens. |
| **SigLIP** | Sigmoid Loss for Language-Image Pre-Training: the ViT-based vision encoder used in PaliGemma, variant So400m/14 (~400M params, patch size 14). Uses LayerNorm and GELU (not RMSNorm and SwiGLU). |
| **GeGLU** | Gated GELU activation used in Gemma's feed-forward network: `GELU(gate_proj(x)) * up_proj(x)`. Distinct from SwiGLU which uses SiLU. |
| **GQA** | Grouped Query Attention: 8 query heads share 1 KV head in both Gemma experts. |
| **KV cache** | Cached key and value tensors from the prefix forward pass, reused across all 10 denoising steps. Read-only during denoising. |
| **FusedOp** | A TT-Blaze composite operation built by chaining MicroOp `emit()` calls, compiled into a single persistent kernel program. |
| **MicroOp** | A TT-Blaze atomic operation backed by a C++ kernel header (LLK). The building block for FusedOps. |
| **CBHandle** | A TT-Blaze Python dataclass referencing a circular buffer, carrying metadata for inter-op data flow within FusedOps. |
| **CB** | Circular Buffer: L1 SRAM scratch space used by TT-Blaze MicroOps for intermediate tensor storage. |
| **blaze-nn** | A PyTorch-compatible Module interface for TT-Blaze that provides `Module`, `Parameter`, functional ops, and tracing for graph/compose execution paths. |
| **Action horizon** | The number of future action steps predicted per inference (50 in pi0.5). Each step is a 32-dimensional action vector. |

### Notation

- Tensor shapes use bracket notation: `[dim0, dim1, ...]`. Common dimensions: B=batch, S=sequence, D=model_dim, H=heads, DH=head_dim.
- Model hidden dimensions: PaliGemma width = 2048 (D_p), Action Expert width = 1024 (D_a), SigLIP width = 1152 (D_s).
- Parameter counts use M (million) and B (billion): 300M, 2B, 2.3B.
- Memory sizes use MB and GB.
- Code references cite specific source file paths and function/class names: `filename.py:ClassName.method_name()`.
- JAX source (pi0.py, gemma.py, siglip.py) is the canonical reference; PyTorch source (pi0_pytorch.py, gemma_pytorch.py) is the deployment reference.
- TT-Blaze source paths are relative to `/localdev/salnahari/testing_dir/tt-blaze/`.
- blaze-nn source paths are relative to `/localdev/salnahari/testing_dir/blaze-nn/`.
- Data types: bf16 = bfloat16, BFP8 = bfloat8_b, BFP4 = bfloat4_b, fp32 = float32.
- Expert indices: expert 0 = PaliGemma backbone (Gemma 2B), expert 1 = Action Expert (Gemma 300M).
- Op mapping uses three-tier classification: **REUSE** (use as-is), **ADAPT** (modify parameters or add mode), **NEW** (write from scratch).

### Formatting Rules

- Each chapter opens with a one-paragraph overview connecting it to the broader implementation effort.
- Each file begins with a one-paragraph "Context" summary and ends with a "Key Takeaways" bulleted list (3-5 items).
- Code references cite the source file with path and line numbers where relevant.
- Diagrams use ASCII art for data flow and tensor shape transformations.
- Code snippets from reference implementations (JAX and PyTorch) are quoted in fenced code blocks with language annotations.
- When comparing pi0 vs. pi0.5, use explicit "pi0:" / "pi0.5:" prefixes.
- When comparing JAX and PyTorch implementations, both are shown side-by-side with explicit mapping notes.

---

## Cross-Chapter Dependencies

```
Chapter 1 (Architecture Overview)
    -- foundational, no dependencies
    |
    +---> Chapter 2 (SigLIP Encoder)
    |         -- depends on Ch 1 for overall data flow and where SigLIP output feeds in
    |
    +---> Chapter 3 (Dual-Expert Transformer)
    |         -- depends on Ch 1 for model components and configs
    |         -- depends on Ch 2 for understanding image tokens that enter the transformer
    |
    +---> Chapter 4 (adaRMSNorm)
              -- depends on Ch 3 for block structure where norms are applied
              |
              v
         Chapter 5 (Flow-Matching Inference)
              -- depends on Ch 1, Ch 3, Ch 4 for full model forward pass
              -- depends on Ch 3 for shared attention and KV cache
              -- depends on Ch 4 for timestep conditioning during denoising
              |
              v
         Chapter 6 (Op Mapping)
              -- depends on Ch 2-5 for complete list of computational operations
              -- depends on Ch 4 for novel ops (adaRMSNorm, gated residual)
              -- depends on Ch 5 for inference-specific ops (KV cache, denoising loop)
              |
              v
         Chapter 7 (FusedOp Design + blaze-nn)
              -- depends on Ch 3-4 for model architecture to express as Modules
              -- depends on Ch 5 for inference orchestration pattern
              -- depends on Ch 6 for reusable vs. new op classification
              |
              v
         Chapter 8 (Pipeline, Performance, Roadmap)
              -- depends on all prior chapters
              -- synthesizes Ch 5 (denoising loop), Ch 6 (ops), Ch 7 (FusedOps/modules)
                 into pipeline design and performance analysis
```

| Chapter | Depends On | Nature of Dependency |
|---------|-----------|---------------------|
| Ch 2 (SigLIP) | Ch 1 (Overview) | Image token shape, observation format, how SigLIP output feeds into prefix |
| Ch 3 (Dual-Expert) | Ch 1, Ch 2 | Prefix token composition (image tokens from SigLIP + language tokens), expert config dimensions |
| Ch 4 (adaRMSNorm) | Ch 3 | Block structure, where norms are applied, gated residual mechanism, expert-specific weight naming |
| Ch 5 (Flow-Matching) | Ch 1, Ch 3, Ch 4 | Full model forward pass, dual-expert block during denoising, adarms_cond from timestep MLP, KV cache from shared attention |
| Ch 6 (Op Mapping) | Ch 2, Ch 3, Ch 4, Ch 5 | All model operations identified in Ch 2-5 are mapped to TT-Blaze ops |
| Ch 7 (FusedOp + blaze-nn) | Ch 3, Ch 4, Ch 5, Ch 6 | Module hierarchy mirrors architecture from Ch 3-4; functional ops reference Ch 6; forward pass follows Ch 5 inference flow; FusedOp designs use Ch 6 op classification |
| Ch 8 (Pipeline/Perf/Roadmap) | Ch 5, Ch 6, Ch 7 | Pipeline stages require understanding denoising loop (Ch 5), available ops (Ch 6), and module/FusedOp structure (Ch 7) |

### Reading Order Recommendations

1. **Sequential (recommended for first read):** Chapters 1 through 8 in order. Each chapter builds on the previous.
2. **Architecture focus:** Ch 1 -> Ch 2 -> Ch 3 -> Ch 4 -> Ch 5. Complete model understanding before hardware mapping.
3. **Kernel engineering focus:** Ch 3 -> Ch 4 -> Ch 6 -> Ch 7 (files 01-02). Focus on novel ops, FusedOp design, and what must be built.
4. **Implementation planning focus:** Ch 1 (skim) -> Ch 6 -> Ch 7 -> Ch 8. Jump to op mapping, blaze-nn design, and the roadmap.
5. **Performance engineering focus:** Ch 1 -> Ch 5 -> Ch 8. Understand data flow, denoising loop, and latency implications.

---

## Question Coverage Matrix

| # | Question | Primary Chapter(s) | Primary File(s) |
|---|----------|--------------------|-----------------|
| 1 | pi0.5 architecture end-to-end | Ch 1 | 01_model_landscape, 02_component_map_and_data_flow, 03_data_types_and_observation |
| 2 | Dual-expert shared attention | Ch 3 | 02_shared_attention_mechanism, 03_attention_masking_and_positions |
| 3 | adaRMSNorm | Ch 4 | 01_adarmsnorm_mechanics, 02_timestep_embedding_pipeline |
| 4 | Flow-matching inference | Ch 5 | 01_flow_matching_theory, 02_complete_inference_walkthrough |
| 5 | Complete inference data flow | Ch 5 | 02_complete_inference_walkthrough, 03_kv_cache_strategy |
| 6 | Reusable TT-Blaze ops | Ch 6 | 01_reusable_ops_inventory |
| 7 | New ops/FusedOps needed | Ch 6, Ch 7 | Ch6/03_new_ops_required, Ch7/01_adarmsnorm_fusedop, Ch7/02_dual_expert_block_fusedop |
| 8 | Dual-expert shared-attention mapping to FusedOp model | Ch 7 | 02_dual_expert_block_fusedop |
| 9 | Flow-matching denoising interaction with pipeline system | Ch 8 | 01_pipeline_stage_design |
| 10 | blaze-nn expression of full pi0.5 forward pass | Ch 7 | 03_module_hierarchy_and_forward_pass, 05_inference_orchestration |
| 11 | Weight loading requirements | Ch 7 | 04_weight_loading_and_conversion |
| 12 | Performance and memory considerations | Ch 8 | 02_memory_and_latency_analysis |
| 13 | Lessons from existing TT-Blaze models | Ch 6, Ch 8 | Ch6/04_patterns_from_deepseek_and_glm, Ch8/04_technical_risks_and_open_questions |

---

## Selection Rationale

This synthesized plan draws from the best elements of all five candidate plans based on the evaluation consensus:

### From Plan V1: File-level specificity and tensor shape detail
V1 had the most detailed file-level descriptions with concrete tensor shapes, weight naming conventions, parameter counts, and CB flow diagrams. The final plan adopts V1's approach of specifying exact shapes (e.g., `[B, 50, 32]`, `[18, B, ~968, 1, 256]`) and code references throughout every file description. V1's three FusedOp design options (A/B/C) with CB flow diagrams are preserved in Chapter 7.

### From Plan V2: Question coverage matrix and reading order recommendations
V2 was the only plan with an explicit question coverage matrix and multiple reading orders. Both are included in the final plan. V2's separation of op reuse analysis from FusedOp design influenced the split between Chapter 6 (op mapping with classification) and Chapter 7 (FusedOp design + blaze-nn). V2's PyTorch implementation differences file is included in Chapter 5.

### From Plan V3: Three-tier op classification and inline TT-Blaze mapping
V3's REUSE / ADAPT / NEW classification is adopted as the standard in Chapter 6. V3's approach of including a TT-Blaze mapping file within the adaRMSNorm chapter (Ch4 file 03) is preserved, along with V5's similar approach for SigLIP (Ch2 file 02). V3's dedicated attention masking file and feedforward+gated-residual file in the dual-expert chapter are adopted to keep related concepts together. V3's bandwidth-bound analysis for the denoising loop is included in Chapter 8.

### From Plan V4: Implementation roadmap and technical risk inventory
V4 was the only plan with a phased implementation roadmap and explicit technical risks. Both are adopted as Chapter 8 files 03 and 04. V4's five-phase approach (standalone ops -> block integration -> vision encoder -> full model -> performance optimization) provides the actionable structure the engineering team needs. V4's grouping of dual-expert attention and adaRMSNorm as "novel mechanisms" was considered but rejected in favor of separate chapters (Ch 3 and Ch 4) since each is complex enough to warrant dedicated treatment, consistent with the evaluation feedback that Ch 2 was overloaded in V4.

### From Plan V5: Inline TT-Blaze mapping per component and inference orchestration
V5's approach of including TT-Blaze mapping files within component chapters (SigLIP, adaRMSNorm) is adopted in Chapters 2 and 4, giving readers immediate hardware context while model details are fresh. V5's dedicated inference orchestration file (covering the two-phase prefix/denoising pattern and None-expert handling) is adopted as Chapter 7 file 05.

### Structural decisions resolving cross-plan conflicts

1. **SigLIP gets its own chapter (Ch 2)** rather than being folded into Ch 1 (as in V2, V4). While SigLIP is architecturally standard, it introduces new op requirements (LayerNorm, GELU, non-GQA attention, Conv2d) and represents a substantial pipeline stage. The evaluation consensus supported this approach.

2. **Attention masking is co-located with dual-expert attention (Ch 3)** rather than in Ch 1 (as in V1). The masking pattern is integral to understanding how dual-expert shared attention works and must be understood together.

3. **adaRMSNorm gets its own chapter (Ch 4)** with an embedded TT-Blaze mapping file, rather than being merged with dual-expert attention (as in V4). The evaluation noted V4's Ch 2 was overloaded by combining both topics.

4. **Lessons from existing models are integrated early (Ch 6)** rather than deferred to a final chapter (as in V1, V2, V5). The evaluation consensus noted that DeepSeek V3 and GLM-5.1 patterns directly inform op mapping and FusedOp design decisions, so they should be available when those decisions are made. A summary also appears in Chapter 8's technical risks section.

5. **FusedOp design and blaze-nn expression are combined in Ch 7** to avoid artificial separation between the kernel-level design and the Module-level expression that consumes those kernels. This consolidation allows the reader to see how FusedOp CBHandle chains map to blaze-nn functional ops in a single chapter.

6. **The implementation roadmap (Ch 8)** closes the guide with actionable engineering guidance, giving the team a concrete path from current state to working pi0.5 inference on Tenstorrent silicon.
