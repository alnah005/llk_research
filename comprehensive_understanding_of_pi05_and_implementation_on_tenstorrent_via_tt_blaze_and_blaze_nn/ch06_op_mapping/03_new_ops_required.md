# 03: New Ops Required (NEW Tier)

## Context

These ops do not exist in TT-Blaze's current inventory and must be built from scratch (or composed from existing MicroOps in new FusedOp configurations). They fall into three categories: (1) the adaRMSNorm and gated_residual ops that are specific to pi0.5's diffusion conditioning mechanism, (2) the SigLIP vision encoder ops that use LayerNorm and GELU, and (3) multi-expert coordination ops that handle the dual-expert architecture.

---

## 1. adaRMSNorm FusedOp

**What it computes:** Adaptive RMSNorm with timestep conditioning. Takes a hidden state `x` and a conditioning vector `cond` (the timestep embedding), produces a normalized+modulated output and a gate vector.

From `gemma.py`:
```python
# Step 1: Standard RMSNorm normalization
var = mean(x^2, axis=-1, keepdims=True)
normed = x * rsqrt(var + 1e-6)

# Step 2: Conditioning projection
modulation = Dense(x.shape[-1] * 3)(cond)   # (B, 3*D) from (B, D)
scale, shift, gate = split(modulation, 3, axis=-1)  # each (B, 1, D)

# Step 3: Apply modulation
output = normed * (1 + scale) + shift

# Step 4: Return output and gate (gate used in gated_residual)
return output, gate
```

**Where it appears in pi0.5:**
- Every layer of the action expert, twice per layer: `pre_attention_norm_1` and `pre_ffw_norm_1`.
- Also at the final norm: `final_norm_1`.
- Total: 2 * 18 + 1 = **37 instances per forward pass** (but with shared weights for the Dense projection).

**Proposed TT-Blaze implementation:**

This should be a `FusedOp` that composes:
1. `Matmul` (for the conditioning Dense projection: `cond @ W_mod -> modulation`)
2. A split/slice (partition `modulation` into scale, shift, gate)
3. `RMSNorm` (the normalization step on `x`)
4. Element-wise ops (multiply by `(1+scale)`, add `shift`)

```
compose():
    # Project conditioning vector to 3*D modulation
    mod_cb = Matmul.emit(f, cond_cb, W_mod, ...)  # (B, D) -> (B, 3*D)

    # Standard RMSNorm on hidden state
    normed_cb = RMSNorm.emit(f, x_cb, gamma=None, ...)  # no learned scale

    # Apply scale and shift (first 2/3 of modulation)
    # Gate (last 1/3) returned as side output
    output_cb = AdaRMSNormApply.emit(f, normed_cb, mod_cb, ...)
    gate_cb = ...  # extracted from mod_cb
    return output_cb, gate_cb
```

**Key design decisions:**
- The Dense projection `W_mod` has shape `(D, 3*D)` -- for D=1024 (action expert), this is `(1024, 3072)`. This fits in L1 as a standard matmul.
- The conditioning vector `cond` is the same for all tokens in the suffix (it's derived from the scalar timestep). So the Dense projection only runs once, producing a `(B, 3072)` vector that is broadcast to all token positions.
- The RMSNorm and modulation application are per-token, per-position operations.

**CB layout:**
- Input CBs: `x` (hidden state, 1xD per token), `cond` (conditioning, 1xD per batch), `W_mod` (DRAM weight)
- Output CBs: `output` (modulated hidden, 1xD per token), `gate` (gate vector, 1xD per token, persists for gated_residual)

**Effort:** Moderate. The individual components (matmul, rmsnorm, eltwise multiply/add) have existing kernels. The new work is:
- The 3-way split of the modulation vector.
- Coordinating the one-per-batch conditioning matmul with the per-token normalization.
- Managing the `gate` side output across the CB pipeline.

---

## 2. gated_residual

**What it computes:** `x + y * gate` -- a gated residual connection where the gate is produced by adaRMSNorm.

From `gemma.py`:
```python
def _gated_residual(x, y, gate):
    if gate is None:
        return x + y        # standard residual (PaliGemma expert)
    return x + y * gate     # gated residual (action expert)
```

**Where it appears in pi0.5:**
- After every attention and FFN block in the action expert.
- 2 * 18 = **36 instances per forward pass.**

**Proposed TT-Blaze implementation:**

This is a simple element-wise op: `output = x + y * gate`. It can be implemented as a MicroOp with a TRISC compute kernel that:
1. Reads `y` and `gate` tiles.
2. Computes `y * gate` (element-wise multiply).
3. Reads `x` tile.
4. Computes `x + (y * gate)` (element-wise add).
5. Writes result.

Alternatively, it can be composed from two existing ops:
```
temp = eltwise_mul(y, gate)
output = residual_add(x, temp)
```

But a fused single-op implementation saves one CB round-trip and one TRISC dispatch.

**CB layout:**
- Input CBs: `x` (residual), `y` (attention/FFN output), `gate` (from adaRMSNorm)
- Output CB: `output = x + y * gate`
- No scratch needed (fits in dest registers).

**Effort:** Low. The compute is two LLK calls (multiply, add) on 32x32 tiles. The CB management follows the same pattern as `residual_add` but with an extra input.

---

## 3. layernorm (for SigLIP)

**What it computes:** Standard LayerNorm: `(x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`. Unlike RMSNorm, this includes mean subtraction and a bias (beta) term.

**Where it appears in pi0.5:**
- SigLIP ViT encoder: `nn.LayerNorm` before every attention and MLP block (27 layers * 2 = 54 instances), plus a final `encoder_norm` (1 instance).
- Total: **55 instances** (but only during vision encoding, not in the denoising loop).

From `siglip.py`:
```python
class Encoder1DBlock(nn.Module):
    def __call__(self, x, deterministic=True):
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = nn.MultiHeadDotProductAttention(...)(y, y)
        x = x + y
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = MlpBlock(...)(y, deterministic)
        x = x + y
        return x
```

**Proposed TT-Blaze implementation:**

LayerNorm differs from RMSNorm in two ways:
1. It subtracts the mean: `x_centered = x - mean(x)`.
2. It uses variance (not just mean of squares): `var = mean(x_centered^2)`.
3. It has both `gamma` (scale) and `beta` (bias) parameters.

The TRISC compute path would be:
```
mean = reduce_sum(x) / D
x_centered = x - mean
var = reduce_sum(x_centered^2) / D
normed = x_centered * rsqrt(var + eps)
output = normed * gamma + beta
```

This is very similar to RMSNorm's compute path but with two extra ops (mean subtraction, bias addition).

**CB layout:**
- Input CBs: `x` (hidden state), `gamma` (scale), `beta` (bias)
- Output CB: `output`
- Scratch: `mean` (scalar per row), `var` (scalar per row)

**Effort:** Moderate. The normalization logic is close to RMSNorm but needs the mean subtraction path. The LLK library has `reduce_sum` and `sub` primitives. This could be implemented by extending the existing RMSNorm MicroOp with a mode flag, or as a standalone MicroOp.

---

## 4. gelu Activation (Standalone)

**What it computes:** `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))` or the tanh approximation: `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.

**Where it appears in pi0.5:**
- SigLIP ViT MLP blocks: `nn.gelu(x)` applied to the intermediate activation.
- Gemma FFN: `nn.gelu(ff_gate)` applied to the gate branch. (This is handled by gated_reduce's `use_gelu` flag in the fused pipeline, but a standalone GELU is needed for SigLIP.)
- Timestep MLP: `nnx.swish(time_emb)` -- this is actually SiLU/swish, not GELU. But the SigLIP path needs standalone GELU.

From `siglip.py`:
```python
class MlpBlock(nn.Module):
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)            # <-- standalone GELU activation
        x = nn.Dense(d)(x)
        return x
```

**Proposed TT-Blaze implementation:**

Two options:
1. **Use `matmul_fused_act`**: The `MatmulFusedAct` MicroOp already fuses activation with matmul output (currently supports sigmoid and SiLU). Adding a GELU mode here would handle SigLIP's `Dense -> GELU` pattern without a standalone op.

2. **Standalone element-wise op**: A MicroOp that applies GELU to each tile. Useful when GELU is not immediately following a matmul.

The Tenstorrent LLK library provides `gelu` as a unary compute primitive, so the TRISC kernel is straightforward.

**Effort:** Low. Either add a mode to `MatmulFusedAct` or create a thin MicroOp wrapper around the LLK `gelu` call.

---

## 5. sinusoidal_timestep_embedding

**What it computes:** Sinusoidal positional encoding for a scalar timestep value, producing a fixed-size embedding vector.

From `pi0.py`:
```python
def posemb_sincos(pos, embedding_dim, min_period, max_period):
    fraction = linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = einsum("i,j->ij", pos, 1.0 / period * 2 * pi)
    return concat([sin(sinusoid_input), cos(sinusoid_input)], axis=-1)
```

Called as:
```python
time_emb = posemb_sincos(timestep, 1024, min_period=4e-3, max_period=4.0)
```

**Where it appears in pi0.5:**
- Once per denoising step, producing the timestep embedding that feeds into the time_mlp and ultimately conditions the adaRMSNorm.
- The output shape is `(B, 1024)` -- a single vector per batch element.

**Proposed TT-Blaze implementation:**

This is a small, one-shot computation that runs once per denoising step. Two approaches:

1. **Host-side computation:** The sinusoidal encoding is cheap (1024 sin/cos evaluations) and the timestep changes each step. Computing on CPU and uploading the result as a tensor avoids allocating device compute for a trivial operation.

2. **Device-side MicroOp:** If the timestep is already on device (e.g., as part of a fused pipeline), a simple TRISC kernel can compute sin/cos over pre-stored period values.

**Recommendation:** Host-side computation is simpler and avoids device overhead. The `posemb_sincos` function has no data dependencies on device tensors (only the scalar timestep), so there's no latency penalty from computing it on CPU.

**Effort:** Minimal if done on host. Low if done as a device op.

---

## 6. siglip_patch_embed

**What it computes:** Patch extraction from an image via a strided convolution, equivalent to `Conv2D(width=1152, kernel=14x14, stride=14)`.

From `siglip.py`:
```python
x = nn.Conv(self.width, self.patch_size, strides=self.patch_size, padding="VALID")(image)
# image: (B, 224, 224, 3) -> x: (B, 16, 16, 1152)
n, h, w, c = x.shape
x = reshape(x, [n, h*w, c])  # (B, 256, 1152)
x = x + posemb_sincos_2d(h, w, 1152)  # add 2D sincos positional embedding
```

**Where it appears in pi0.5:**
- Once per inference call (not in the denoising loop), per input image (up to 3 images: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb).

**Proposed TT-Blaze implementation:**

The strided convolution with `kernel_size == stride` is equivalent to reshaping the image into non-overlapping patches and applying a linear projection:
```
# Reshape image into patches
patches = image.reshape(B, 16, 14, 16, 14, 3)  # (B, H_patches, patch_h, W_patches, patch_w, C)
patches = patches.transpose(0, 1, 3, 2, 4, 5)   # (B, 16, 16, 14, 14, 3)
patches = patches.reshape(B, 256, 588)            # 14*14*3 = 588

# Linear projection
patch_embeddings = patches @ W_conv  # (B, 256, 588) @ (588, 1152) -> (B, 256, 1152)
```

This can be implemented as:
1. A host-side reshape that converts the image tensor into patch vectors.
2. A standard matmul (`(B*256, 588) @ (588, 1152)`) on device.
3. An element-wise add for the positional embedding.

**Effort:** Low. The "convolution" is just a matmul after reshaping. The reshape can be done on host. The 2D sincos positional embedding is a fixed tensor that can be precomputed and stored.

---

## 7. dual_expert_qkv_concat_split

**What it computes:** Coordinates the dual-expert attention mechanism where two experts produce separate Q, K, V tensors that are concatenated along the sequence dimension before shared attention, then split back after attention.

From `gemma.py`, `Attention.__call__`:
```python
# Each expert produces Q, K, V independently
for i, (x, config) in enumerate(zip(xs, self.configs)):
    if x is None:
        continue
    q = q_einsum("BTD,NDH->BTNH", x)
    k, v = kv_einsum("BSD,2KDH->2BSKH", x)
    qkvs.append((q, k, v))

# Concatenate across experts along sequence dim
q, k, v = (concat(y, axis=1) for y in zip(*qkvs))

# Shared attention
logits = einsum("BTKGH,BSKH->BKGTS", q, k)
encoded = einsum("BKGTS,BSKH->BTKGH", probs, v)

# Split back per-expert
for i, (x, config) in enumerate(zip(xs, self.configs)):
    if x is not None:
        out_i = out_einsum("BTNH,NHD->BTD", encoded[:, start:end])
```

**Where it appears in pi0.5:**
- Every layer (18 layers). The PaliGemma expert has `T_prefix` tokens, the action expert has `T_suffix` tokens. They are concatenated to `T_prefix + T_suffix` for shared attention, then split.

**Proposed TT-Blaze implementation:**

This is fundamentally about managing the sequence dimension across two sets of Q, K, V projections. Two approaches:

**Approach A: Separate matmuls + explicit concat/split.**
1. Run Q/K/V matmul for expert 0 (PaliGemma) on its token range.
2. Run Q/K/V matmul for expert 1 (action expert) on its token range.
3. Concatenate K, V along the sequence dimension (Q is also concatenated but can be kept separate since attention queries are independent per position).
4. Run shared SDPA.
5. Split the encoded output by token range.
6. Run separate output projections per expert.

**Approach B: Use KV cache as implicit concat.**
During inference, the prefix K/V are already in the KV cache. The suffix K/V are computed fresh. SDPA naturally handles "cached K/V + new K/V" -- this is the standard KV-cache decode pattern. No explicit concat/split op is needed.

**Recommendation:** Approach B for inference. The dual-expert architecture maps naturally to the prefix-caching pattern: the PaliGemma expert fills the KV cache (prefix), and the action expert produces new K/V (suffix) that attend to the cached prefix. The SDPA op already handles this via its KV cache input.

For training or non-cached prefill, Approach A would be needed, but pi0.5 deployment on TT hardware is inference-only.

**Effort:** Low if using the KV cache approach (no new op needed). Moderate if explicit concat/split is required for training support.

---

## Summary Table

| Op | Compute Pattern | Instances/Forward | Effort |
|----|----------------|-------------------|--------|
| adaRMSNorm | RMSNorm + conditioning Dense + scale/shift/gate | 37 | Moderate |
| gated_residual | `x + y * gate` | 36 | Low |
| layernorm | Mean-centered normalization + gamma + beta | 55 (vision only) | Moderate |
| gelu | GELU activation | ~54 (vision) + fused in FFN | Low |
| sinusoidal_timestep_embedding | Sin/cos encoding of scalar | 1/step | Minimal |
| siglip_patch_embed | Strided conv as reshape + matmul | 1-3/inference | Low |
| dual_expert_qkv_concat_split | Sequence-dim concat for shared attention | 18 | Low (via KV cache) |

## Key Takeaways

1. **adaRMSNorm is the critical-path new op.** It runs 37 times per forward pass (in the denoising loop) and is on the latency-critical path. Every microsecond saved here multiplies by 37 * num_denoising_steps. A fused implementation (matmul + RMSNorm + modulation in one kernel) would be significantly faster than composing separate ops.

2. **gated_residual is trivial but high-frequency.** At 36 instances per forward pass, it should be fused into the adaRMSNorm + attention/FFN pipeline if possible, to avoid extra CB round-trips.

3. **SigLIP ops (layernorm, gelu, patch_embed) run once, outside the denoising loop.** They are important for end-to-end model support but are not on the latency-critical path. A less optimized implementation is acceptable for initial bring-up.

4. **The dual-expert architecture maps naturally to KV caching.** For inference, the two experts do not need an explicit concat/split op. The prefix-caching mechanism handles the sequence-dimension fusion implicitly. This is a significant simplification compared to what the training code suggests.

5. **The timestep embedding should be computed on the host.** It's a cheap, scalar-input function with no device-tensor dependencies. Sending it to the device as a pre-computed tensor avoids unnecessary kernel launches.

6. **Total new kernel work is approximately 3 substantive ops** (adaRMSNorm, gated_residual, layernorm) plus 2 thin wrappers (gelu activation, patch embed matmul). The sinusoidal embedding and dual-expert concat can be handled without new kernels.
