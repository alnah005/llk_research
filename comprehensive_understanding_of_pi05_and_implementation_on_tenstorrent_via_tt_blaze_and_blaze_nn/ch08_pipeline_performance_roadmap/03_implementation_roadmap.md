# 03 -- Implementation Roadmap

## Context

This section lays out a 5-phase plan for bringing pi0.5 to Tenstorrent silicon. Each phase is designed to derisk the next: Phase 1 validates individual ops, Phase 2 composes them into blocks, Phase 3 adds the vision encoder, Phase 4 integrates the full model, and Phase 5 optimizes for production. The phases are ordered by dependency -- each one builds on the outputs of the previous.

The roadmap assumes:
- A single Wormhole or Blackhole device as the initial target.
- Host-orchestrated looping (Strategy A from section 01) as the first pipeline approach.
- PyTorch reference weights converted from the openpi checkpoint format.
- blaze-nn as the primary tracing and module framework, with TT-Blaze FusedOps for performance-critical kernels.

## Phase Overview

```
+-----------------------------------------------------------------------+
|  Phase 1: Standalone Ops                                               |
|  Goal: Every atomic op in pi0.5 runs on device, bit-accurate vs ref   |
|  Duration: 2-3 weeks                                                   |
+-----------------------------------------------------------------------+
          |
          v
+-----------------------------------------------------------------------+
|  Phase 2: Transformer Blocks                                           |
|  Goal: Single-layer Gemma block (both experts) runs end-to-end        |
|  Duration: 2-3 weeks                                                   |
+-----------------------------------------------------------------------+
          |
          v
+-----------------------------------------------------------------------+
|  Phase 3: Vision Encoder                                               |
|  Goal: SigLIP So400m/14 produces correct image tokens                 |
|  Duration: 1-2 weeks                                                   |
+-----------------------------------------------------------------------+
          |
          v
+-----------------------------------------------------------------------+
|  Phase 4: Full Model Integration                                       |
|  Goal: End-to-end sample_actions matches reference output             |
|  Duration: 2-3 weeks                                                   |
+-----------------------------------------------------------------------+
          |
          v
+-----------------------------------------------------------------------+
|  Phase 5: Optimization                                                 |
|  Goal: Hit latency targets for real-time robot control                |
|  Duration: Ongoing                                                     |
+-----------------------------------------------------------------------+
```

## Phase 1: Standalone Ops (Weeks 1-3)

### Objective

Validate that every op type used in pi0.5 can execute on Tenstorrent hardware and produce numerically correct results when compared against the JAX/Flax reference implementation.

### Op Inventory

The following table lists every distinct op type in pi0.5, grouped by whether an existing TT-Blaze/blaze-nn implementation exists.

**Ops with existing implementations (reuse from DeepSeek):**

| Op | blaze-nn API | Source | Notes |
|---|---|---|---|
| Linear (matmul) | `F.linear` | `blaze_nn/functional.py:36` | Used everywhere |
| RMSNorm | `F.rmsnorm` | `blaze_nn/functional.py:46` | Standard (non-adaptive) variant |
| RoPE | `F.rope` | `blaze_nn/functional.py:71` | Rotary position embedding |
| Gated MLP | `F.gated_reduce` | `blaze_nn/functional.py:61` | SiLU/GELU activation + gate |
| Residual add | `F.residual_add` | `blaze_nn/functional.py:66` | Element-wise add |
| Multicast | `F.mcast` | `blaze_nn/functional.py:51` | Broadcast to cores |
| Gather | `F.gather` | `blaze_nn/functional.py:56` | Collect from cores |
| GQA attention | `_gqa_attention.py` | `tt-blaze/blaze/` | 8 query heads, 1 KV head |

**Ops requiring new implementation:**

| Op | Description | Reference Source | Priority |
|---|---|---|---|
| adaRMSNorm | RMSNorm with learned scale+shift+gate from conditioning | `gemma.py:113-131` | **P0** -- blocks Phase 2 |
| Conv2d (patch extract) | 14x14 kernel, stride 14, 3->1152 channels | `siglip.py:216-223` | **P0** -- blocks Phase 3 |
| SDPA with prefix-LM mask | Scaled dot-product attention with non-causal mask | `pi0.py:19-44` | **P0** -- blocks Phase 2 |
| Sincos positional embedding (2D) | Sinusoidal posemb for SigLIP patches | `siglip.py:27-37` | P1 -- can precompute on host |
| Sincos positional embedding (1D) | Timestep embedding for flow matching | `pi0.py:48-63` | P1 -- can precompute on host |
| SiLU (swish) activation | `nnx.swish` in time MLP | `pi0.py:165` | P1 -- element-wise, simple |
| Token embedding lookup | Table lookup from vocab 257,152 | `gemma.py:148-151` | P1 -- use existing embedding op |
| Attention mask construction | `make_attn_mask` cumsum + broadcast + logical_and | `pi0.py:19-44` | P2 -- can compute on host |

### Deliverables

1. **Test harness:** For each op, a Python test that:
   a. Runs the op on CPU with reference weights to get golden output.
   b. Runs the same op on device via blaze-nn or TT-Blaze.
   c. Compares outputs with configurable tolerance (default: PCC > 0.999 for bf16).

2. **adaRMSNorm Blaze MicroOp:** New kernel that implements:
   ```
   normed = x * rsqrt(mean(x^2) + eps)
   modulation = Dense(cond)                    # cond is timestep embedding
   scale, shift, gate = split(modulation, 3)
   output = normed * (1 + scale) + shift
   return output, gate                         # gate used for gated residual
   ```
   This is the single most important new op because it appears twice per layer (pre-attention and pre-FFN) across all 18 layers.

3. **Conv2d patch extraction test:** Validate that TTNN's `conv2d` or a custom Blaze kernel correctly implements the SigLIP patch extraction (14x14 kernel, stride 14, VALID padding, 3->1152 channels, input 224x224x3).

4. **SDPA mask test:** Validate that the SDPA kernel accepts a non-boolean mask (or converts a prefix-LM mask pattern) and produces correct attention outputs. Test both the prefix-fill case (bidirectional within prefix) and the denoising case (suffix attends to prefix+suffix, prefix cached in KV).

### Exit Criteria

- All ops pass PCC > 0.999 against JAX reference.
- adaRMSNorm, Conv2d, and masked SDPA have known-working implementations (even if not yet optimized).
- A clear mapping from each pi0.5 op to its TT implementation is documented.

## Phase 2: Transformer Blocks (Weeks 3-6)

### Objective

Compose standalone ops into working transformer blocks. The key challenge is the **dual-expert block** where PaliGemma (width=2048) and Action Expert (width=1024) share a single attention computation.

### Work Items

**2.1: Single-Expert Gemma Block**

Implement a standard Gemma transformer block with one expert (PaliGemma, width=2048). This is structurally identical to a DeepSeek transformer block minus the MoE routing.

```
Input: [B, T, 2048]
  |
  +-> RMSNorm -> Q/K/V projection -> RoPE -> GQA SDPA -> Out projection
  |              (8 heads, 1 KV,                         (2048 x 2048)
  |               head_dim=256)
  +-> residual add
  |
  +-> RMSNorm -> Gated MLP (2048 -> 16384 -> 2048, GELU)
  |
  +-> residual add
  |
Output: [B, T, 2048]
```

Validation: Compare layer output against JAX reference with same weights.

**2.2: Dual-Expert Gemma Block**

The pi0.5 signature block. Two experts with different widths share attention.

```
Input PaliGemma: [B, T_prefix, 2048]     Input Action: [B, T_suffix, 1024]
       |                                         |
       v                                         v
   RMSNorm(2048)                          adaRMSNorm(1024, cond)
       |                                         |
       v                                         v
   Q: [B,T_p,8,256]                       Q: [B,T_s,8,256]
   K: [B,T_p,1,256]                       K: [B,T_s,1,256]
   V: [B,T_p,1,256]                       V: [B,T_s,1,256]
       |         |                             |         |
       +----+----+----+-------+--------+-------+----+----+
            |         |                |            |
            v         v                v            v
        Q_cat: [B, T_p+T_s, 8, 256]   KV_cat: [B, T_p+T_s, 1, 256]
                     |
                     v
              Joint GQA SDPA
            (with prefix-LM mask)
                     |
                     v
              [B, T_p+T_s, 8, 256]
                     |
            +--------+--------+
            v                 v
    Out_PG: [B,T_p, 2048]   Out_Act: [B,T_s, 1024]
    (8x256 -> 2048)          (8x256 -> 1024)
            |                         |
            v                         v
     residual add              gated residual add (with gate from adaRMSNorm)
            |                         |
            v                         v
    RMSNorm(2048)              adaRMSNorm(1024, cond)
            |                         |
            v                         v
    MLP(2048->16384->2048)     MLP(1024->4096->1024)
            |                         |
            v                         v
     residual add              gated residual add
```

**Critical implementation detail:** The Q/K/V projections use *different weight matrices* per expert (different widths for the input dimension). The Q tensors have the same output shape (8 heads x 256 dim) regardless of input width, so they can be concatenated for joint attention. But the input projection shapes differ:

```
PaliGemma Q: [2048, 8*256] = [2048, 2048]
Action Q:    [1024, 8*256] = [1024, 2048]
```

The SDPA sees a single concatenated Q of shape `[B, T_p+T_s, 8, 256]` and a single concatenated K of shape `[B, T_p+T_s, 1, 256]`. After SDPA, the output is split back at the T_p boundary and projected through different output matrices.

**See Risk #2 in section 04 for the width-mismatch challenge this creates.**

**2.3: Stacked Blocks (18 layers)**

Once a single dual-expert block works, stack 18 of them with shared KV cache management. The JAX reference uses `nn.scan` to share weights across layers -- on TT hardware this means either:

(a) Sequentially executing 18 block instances that share the same weight buffers (streaming weights from DRAM per layer), or
(b) Having 18 distinct weight sets if L1/DRAM layout requires it.

Option (a) is the standard approach and matches the bandwidth analysis in section 02.

**2.4: KV Cache Fill + Reuse Test**

Validate the two-pass pattern:
1. Fill: Run 18 layers with PaliGemma input only, capture KV cache.
2. Reuse: Run 18 layers with Action Expert input only, reading from cached KV.
3. Compare combined output against JAX reference that does both passes.

This tests that KV cache storage and retrieval is correct across the two execution modes.

### Exit Criteria

- A single dual-expert block matches JAX reference output (PCC > 0.999).
- 18 stacked layers with KV cache fill + reuse produce correct outputs.
- Latency of one 18-layer denoising iteration is measured.

## Phase 3: Vision Encoder (Weeks 5-7)

### Objective

Bring up the SigLIP So400m/14 vision encoder on device.

### Work Items

**3.1: Conv2d Patch Extraction**

The first op in SigLIP is a 2D convolution that serves as patch extraction:
```python
nn.Conv(1152, (14, 14), strides=(14, 14), padding="VALID")
```

This is not a typical deep convolution -- it is a single-layer "mega-kernel" that converts a 224x224x3 image into a 16x16 grid of 1152-dimensional embeddings (256 patches).

Options:
- **TTNN conv2d:** Use the built-in TTNN convolution op. This should work but may not be optimized for this unusual kernel size.
- **Im2col + matmul:** Reshape the image into [256, 588] (256 patches, each 14x14x3=588 values) and multiply by [588, 1152] weight matrix. This converts the convolution into a standard matmul that blaze-nn handles natively.
- **Custom Blaze kernel:** Write a patch extraction kernel if neither above meets performance targets.

**Recommendation:** Start with im2col + matmul. It is the simplest path and the patch extraction runs only once per inference (not on the critical path).

**3.2: SigLIP Transformer Stack**

27 layers of standard bidirectional self-attention (no GQA -- full MHSA with 16 heads). This is simpler than the Gemma block because:
- No KV cache (bidirectional, runs once).
- No GQA (num_heads = num_kv_heads = 16).
- No gated MLP (standard MLP with GELU).
- No adaptive normalization (standard LayerNorm).

Each layer:
```
LayerNorm -> MHSA(16 heads, width 1152) -> residual -> LayerNorm -> MLP(4304) -> residual
```

The blaze-nn module system should handle this with standard `F.linear`, `F.rmsnorm` (or LayerNorm equivalent), and activation functions.

**3.3: Head Projection**

The final output of SigLIP is projected from width 1152 to PaliGemma's width 2048:
```python
nn.Dense(num_classes, ...)  # num_classes = paligemma_config.width = 2048
```

This is a simple linear layer.

**3.4: End-to-End SigLIP Test**

Feed a reference image through the device SigLIP and compare output tokens against the JAX reference.

### Exit Criteria

- SigLIP produces correct image tokens for test images (PCC > 0.999).
- Single-image encoding latency is measured.

## Phase 4: Full Model Integration (Weeks 7-10)

### Objective

Connect all stages into a working `sample_actions` pipeline that produces correct denoised actions.

### Work Items

**4.1: Prefix Embedding Pipeline**

Connect SigLIP output (Stage 1) with text embedding and attention mask construction (Stage 2). The `embed_prefix` function in `pi0.py` (lines 106-137) concatenates image tokens and text embeddings, constructs `input_mask` and `ar_mask`.

Decision point: Compute the attention mask on host or device?

```
Mask computation: make_attn_mask(input_mask, mask_ar)
  - cumsum on mask_ar: [prefix_len] -> [prefix_len]
  - outer comparison: [prefix_len] x [prefix_len] -> [prefix_len, prefix_len]
  - logical_and with valid_mask

Total mask size: [B, prefix_len, prefix_len] = [1, 968, 968] = ~1.9 MB at bool
```

**Recommendation:** Compute on host, transfer to device. The mask is small and computed once. This avoids implementing `cumsum` and `outer comparison` on device.

**4.2: Suffix Embedding Pipeline**

The `embed_suffix` function (lines 139-186) projects noisy actions, computes timestep embedding, and constructs the suffix attention mask.

For pi0.5: The adaRMSNorm conditioning signal `time_emb` flows through the time MLP (two linear layers with swish activation) and is passed as `adarms_cond` to every layer.

**4.3: Denoising Loop Integration**

Implement the host-orchestrated loop:

```python
# Stage 1: SigLIP
image_tokens = run_siglip(images)                    # device

# Stage 2: Prefix + KV fill
prefix_tokens = concat(image_tokens, text_tokens)    # host or device
prefix_mask = make_attn_mask(input_mask, ar_mask)     # host
kv_cache = run_prefix_fill(prefix_tokens, mask)       # device

# Stage 3: Denoising loop
x_t = noise                                           # host -> device
for step in range(num_steps):
    t = 1.0 - step / num_steps
    v_t = run_denoise_step(x_t, t, kv_cache)          # device
    x_t = x_t + dt * v_t                              # host or device

# Stage 4: Read out
actions = x_t                                         # device -> host
```

**4.4: Numerical Validation**

Compare the full pipeline output against the JAX reference `sample_actions` with the same:
- Input images (3 camera views)
- Text prompt
- Robot state
- Random seed (for initial noise `x_0`)

The outputs should match within bf16 tolerance. Note: Flow matching is iterative, so numerical errors compound across denoising steps. The tolerance after 10 steps may be looser than single-op tolerance. Recommended validation:
- Single step output PCC > 0.999
- After 5 steps PCC > 0.995
- After 10 steps PCC > 0.99
- Action MSE < 0.01 (application-level metric)

**4.5: Weight Conversion Pipeline**

Build a tool that:
1. Loads openpi checkpoint (JAX format via `orbax` or PyTorch format via `safetensors`).
2. Remaps parameter names to the TT module hierarchy.
3. Converts dtypes (float32 -> bf16, optionally -> BFP8).
4. Writes to device DRAM in the expected layout.

The reference code at `model.py:286-332` shows the JAX checkpoint loading path. The PyTorch path is at `model.py:243-247`.

### Exit Criteria

- Full `sample_actions` produces correct actions for reference inputs.
- End-to-end latency is measured and compared against section 02 estimates.
- Weight loading from checkpoint to device is automated.

## Phase 5: Optimization (Ongoing)

### Objective

Close the gap between measured latency and target latency (10-50 Hz control rate).

### Work Items (Priority Order)

**5.1: Profile and Identify Bottlenecks**
- Use TT-Blaze's per-op profiling to identify the top-5 latency contributors per denoising iteration.
- Measure actual DRAM bandwidth utilization vs. peak.
- Identify any unexpected data movement (e.g., PaliGemma weights being read during denoising).

**5.2: Validate Step Reduction**
- Run pi0.5 with num_steps = {10, 8, 5, 4, 3} and measure action quality on standard benchmarks.
- Determine minimum steps for acceptable policy performance.

**5.3: BFP8 Quantization**
- Convert weights to BFP8.
- Validate action quality.
- Measure latency improvement.

**5.4: L1 Weight Caching**
- Pin action expert attention weights in L1 across denoising iterations.
- Requires CB configuration changes to keep L1 allocations alive between program dispatches.

**5.5: Op Fusion**
- adaRMSNorm fusion (norm + Dense + split + modulate as single FusedOp).
- Attention QKV fusion (projections + SDPA).
- Gated MLP fusion (gate + up + activation + down).

**5.6: Device-Side Looping (Strategy B)**
- Implement step counter and conditional branching on device.
- Eliminate host round-trips during denoising.

**5.7: Batching**
- Enable multi-batch inference for multi-robot control.
- Validate that latency scales sub-linearly with batch size.

### Exit Criteria

- Measured latency meets target control frequency for the application.
- Action quality is validated on target robot manipulation tasks.

## Key Takeaways

- The roadmap has 5 phases spanning approximately 10-13 weeks for core bring-up (Phases 1-4), with optimization (Phase 5) ongoing thereafter.
- Phase 1 is the highest-risk phase because it confronts the three hardest unknowns: adaRMSNorm, Conv2d patch extraction, and prefix-LM masked SDPA.
- Phase 2 derisks the dual-expert block, which is architecturally unique to pi0.5 and has no direct precedent in the existing TT-Blaze model library.
- Host-orchestrated looping (Strategy A) is used throughout Phases 1-4. Device-side looping is a Phase 5 optimization, not a correctness requirement.
- Weight conversion from openpi checkpoints should be automated early (Phase 1-2) to avoid blocking integration testing.

---

**Next:** [`04_technical_risks_and_open_questions.md`](./04_technical_risks_and_open_questions.md)
