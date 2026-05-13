# Mapping SigLIP to TT-Blaze

## Context

This file analyzes the gap between the existing TT-Blaze / blaze-nn operator set and the operators required to run the SigLIP So400m/14 encoder on Tenstorrent hardware. The current blaze-nn `functional.py` (at `/localdev/salnahari/testing_dir/blaze-nn/blaze_nn/functional.py`) provides ops designed for Gemma-family decoder blocks: `rmsnorm`, `linear` (matmul), `rope`, `gated_reduce`, `residual_add`, `mcast`, `gather`, and `sliced_matmul`. SigLIP requires several ops that either do not exist in this set or behave differently from the Gemma-targeted versions.

The goal is to produce a concrete engineering plan: what new ops are needed, how existing ops can be adapted, and what execution strategy minimizes implementation effort while meeting latency targets.

---

## 1. Op-by-Op Gap Analysis

```
+------------------------+--------------------+----------------------------+----------+
| SigLIP Operation       | Existing blaze-nn  | Gap                        | Severity |
+------------------------+--------------------+----------------------------+----------+
| Conv2D patch embed     | (none)             | Decomposable to reshape +  | LOW      |
|   14x14, stride=14     |                    | matmul; no new kernel      |          |
+------------------------+--------------------+----------------------------+----------+
| LayerNorm              | rmsnorm            | Missing mean subtraction   | HIGH     |
|   (mean + var + bias)  |                    | and bias; NOT equivalent   |          |
+------------------------+--------------------+----------------------------+----------+
| GELU activation        | gated_reduce(silu) | Wrong function entirely;   | HIGH     |
|                        |                    | need GELU or fused op      |          |
+------------------------+--------------------+----------------------------+----------+
| Standard MHA           | (sdpa implied)     | Must support non-causal    | MEDIUM   |
|   16 heads, no mask    |                    | mode (no mask argument)    |          |
+------------------------+--------------------+----------------------------+----------+
| 2-layer MLP            | linear (matmul)    | Existing matmul works,     | LOW      |
|   (no gating)          |                    | but no fused matmul+GELU   |          |
+------------------------+--------------------+----------------------------+----------+
| Sinusoidal posemb add  | residual_add       | Existing op works as-is    | NONE     |
+------------------------+--------------------+----------------------------+----------+
| Residual connections   | residual_add       | Existing op works as-is    | NONE     |
+------------------------+--------------------+----------------------------+----------+
| Bias addition          | residual_add       | Can reuse for Dense bias   | NONE     |
|   (Dense layers)       |                    | terms, with broadcasting   |          |
+------------------------+--------------------+----------------------------+----------+
| RoPE                   | rope               | NOT NEEDED (SigLIP uses    | NONE     |
|                        |                    | additive sinusoidal posemb)|          |
+------------------------+--------------------+----------------------------+----------+
| Gated reduction        | gated_reduce       | NOT NEEDED (SigLIP MLP     | NONE     |
|   (SwiGLU)             |                    | has no gate projection)    |          |
+------------------------+--------------------+----------------------------+----------+
| Projection head        | linear (matmul)    | Existing op works as-is    | NONE     |
|   Dense(1152->2048)    |                    |                            |          |
+------------------------+--------------------+----------------------------+----------+
```

**Summary**: Two HIGH-severity gaps (LayerNorm, GELU), one MEDIUM gap (non-causal attention), and the rest can use existing ops or simple decompositions.

---

## 2. Conv2D Patch Embedding Decomposition

The SigLIP patch embedding is a Conv2D with kernel size equal to stride (14x14), which means patches are non-overlapping. This can be decomposed into a reshape followed by a matmul, avoiding the need for a general Conv2D kernel.

**Decomposition**:

```
Input: [B, 224, 224, 3]

Step 1 - Extract patches (reshape only, no compute):
  Reshape to [B, 16, 14, 16, 14, 3]      -- split H and W into (grid, patch)
  Transpose to [B, 16, 16, 14, 14, 3]     -- group grid dims together
  Reshape to [B, 256, 588]                 -- flatten patches (14*14*3 = 588)

Step 2 - Linear projection (matmul):
  Weight: [588, 1152]                      -- reshaped from Conv kernel [14,14,3,1152]
  Bias:   [1152]
  Output = patches @ weight + bias         -- [B, 256, 1152]
```

```
  [B, 224, 224, 3]
        |
   reshape + transpose          <-- host-side or tilize-time reshape
        |
  [B, 256, 588]
        |
   F.linear(weight=[588, 1152]) <-- existing matmul op
        |
   F.residual_add(bias=[1152])  <-- existing residual_add for bias
        |
  [B, 256, 1152]
```

**Implementation**: The reshape/transpose is a data layout transformation with no arithmetic. It can be done on the host before sending to the device, or folded into the tilize operation. The matmul and bias-add use existing blaze-nn ops directly.

The Conv2D kernel weights `[14, 14, 3, 1152]` are reshaped to `[588, 1152]` at weight-loading time (once, offline).

Source: `/tmp/openpi/src/openpi/models/siglip.py`, lines 216-226.

---

## 3. LayerNorm: Why RMSNorm is NOT Sufficient

The existing `rmsnorm` op in `functional.py` (line 46-48) computes:

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

SigLIP's `nn.LayerNorm` computes:

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

The differences:
1. **Mean subtraction**: LayerNorm subtracts the mean before normalization. RMSNorm does not.
2. **Variance vs RMS**: LayerNorm uses variance (mean of squared deviations from mean). RMSNorm uses mean of squares (no mean subtraction).
3. **Bias (beta)**: LayerNorm has a learnable additive bias. RMSNorm does not.

These are NOT numerically equivalent. Using RMSNorm in place of LayerNorm will produce wrong results.

### Option A: New `layernorm` MicroOp

Add a dedicated `layernorm` MicroOp to the blaze-nn functional API:

```python
def layernorm(input: TensorProxy, gamma: Any, beta: Any,
              *, epsilon: float = 1e-5, **kwargs) -> TensorProxy:
    """Full Layer Normalization (mean + variance + scale + bias)."""
    return _dispatch("layernorm", input, gamma, beta, epsilon=epsilon, **kwargs)
```

This is the cleanest approach. It maps to ttnn's `ttnn.layer_norm` which already exists in the ttnn op library and supports the full LayerNorm computation on device.

### Option B: Decomposition into Existing Ops

```
mean = reduce_mean(x, axis=-1)           -- new reduce op needed
x_centered = x - mean                    -- residual_add (with negated mean)
x_normed = rmsnorm(x_centered, gamma)    -- existing rmsnorm
output = x_normed + beta                 -- residual_add
```

This introduces a dependency on a `reduce_mean` op that also does not exist in blaze-nn. The decomposition adds 4 ops per LayerNorm call (55 total across the encoder), increasing dispatch overhead.

### Recommendation

**Option A (new `layernorm` op)** is strongly preferred. The op is well-defined, ttnn already supports it natively, and it avoids 55 x 3 = 165 extra op dispatches per forward pass. The additional op is also useful beyond SigLIP -- any future model with LayerNorm benefits.

Source: `/localdev/salnahari/testing_dir/blaze-nn/blaze_nn/functional.py`, lines 46-48.

---

## 4. GELU Activation

The existing blaze-nn activation support is limited to:
- `gated_reduce(activation="silu")` -- fused gate * SiLU(up), specific to SwiGLU MLP blocks

SigLIP uses GELU (`nn.gelu`), which is mathematically:
```
GELU(x) = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

This is commonly approximated as:
```
GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

### Option A: Standalone `gelu` Op

```python
def gelu(input: TensorProxy, **kwargs) -> TensorProxy:
    """Gaussian Error Linear Unit activation."""
    return _dispatch("gelu", input, **kwargs)
```

Maps to `ttnn.gelu` which is available in the ttnn op library.

### Option B: Fused `matmul_gelu`

For the SigLIP MLP up-projection, fuse the matmul and GELU into a single kernel:

```python
def matmul_fused_act(input, weight, *, activation="gelu", **kwargs):
    """Matrix multiply with fused activation."""
    return _dispatch("matmul_fused_act", input, weight, activation=activation, **kwargs)
```

This saves one full tensor read/write cycle per MLP block (27 blocks = 27 saved round-trips).

### Option C: Decompose GELU

Decompose GELU into elementary ops: multiply, erf (or tanh approximation), add, scale. This requires multiple dispatches and intermediate buffers, making it the worst option for performance.

### Recommendation

**Start with Option A** (standalone `gelu` op) for correctness. **Upgrade to Option B** (fused matmul+GELU) as a performance optimization once the basic pipeline is working. Option C should be avoided.

The SigLIP MLP forward pass with the new ops:

```
x = F.linear(x, W_up)          # [B, 256, 1152] -> [B, 256, 4304]
x = F.residual_add(x, b_up)    # add bias
x = F.gelu(x)                  # GELU activation
x = F.linear(x, W_down)        # [B, 256, 4304] -> [B, 256, 1152]
x = F.residual_add(x, b_down)  # add bias
```

Source: `/tmp/openpi/src/openpi/models/siglip.py`, lines 61-72.

---

## 5. Non-Causal Scaled Dot-Product Attention

SigLIP uses standard bidirectional (non-causal) self-attention. The Gemma-targeted `sdpa` implementation likely assumes causal masking (lower-triangular mask) since decoder models always need it.

For SigLIP:
- Attention matrix: `[B, 16, 256, 256]` -- all-to-all, no mask
- No RoPE -- Q and K are used directly from the linear projections
- Head dim = 72 (non-power-of-2, which may affect some hardware-optimized kernels)

### Required Changes

The `sdpa` kernel (or its TT-Blaze equivalent) must:

1. **Support an optional mask or no-mask mode**: Skip the causal mask application entirely. This is likely a flag or a `mask=None` code path.

2. **Handle head_dim=72**: Some SDPA implementations pad to power-of-2 head dims (64 or 128). Verify that head_dim=72 is supported or implement padding: pad K and V to head_dim=80 or 128, compute attention, then slice the output back to 72. The padding overhead is modest for 256-token sequences.

3. **No RoPE integration**: If the existing SDPA fuses RoPE application, ensure there is a code path that skips it.

### Attention Tensor Shapes

```
Q, K, V projections (each):
  Input:  [B, 256, 1152]
  Weight: [1152, 1152] (logically [1152, 16*72])
  Bias:   [1152]
  Output: [B, 256, 1152] -> reshape -> [B, 256, 16, 72] -> transpose -> [B, 16, 256, 72]

Attention scores:
  Q @ K^T / sqrt(72) -> [B, 16, 256, 256]

Softmax (no mask):
  softmax(scores, axis=-1) -> [B, 16, 256, 256]

Value aggregation:
  weights @ V -> [B, 16, 256, 72]

Output projection:
  reshape -> [B, 256, 1152]
  Dense(1152 -> 1152) + bias -> [B, 256, 1152]
```

The 256x256 attention matrix is small (256 KB per head at bf16, ~4 MB total for 16 heads). This fits entirely in Tensix L1 SRAM, making flash-attention unnecessary -- standard materialized attention is fine.

---

## 6. Sinusoidal Positional Embeddings: Host Precomputation

The positional embeddings are deterministic and input-independent. Strategy:

```
Host (one-time, at model load):
  1. Compute posemb_sincos_2d(16, 16, 1152) -> [1, 256, 1152] tensor
  2. Cast to bfloat16
  3. Transfer to device as a constant weight tensor

Device (per inference):
  1. After patch embedding matmul: x has shape [B, 256, 1152]
  2. F.residual_add(x, posemb)   -- broadcast [1, 256, 1152] over batch
```

The `residual_add` op already exists and handles this. No new op needed.

Source: `/tmp/openpi/src/openpi/models/siglip.py`, lines 27-37.

---

## 7. Pipeline Execution Strategy

SigLIP runs **once per inference call**, not per denoising step. In `pi0.py`, `embed_prefix()` (line 106-137) is called once, and its output is cached in the KV cache for all subsequent denoising iterations (`sample_actions`, lines 224-278). This makes SigLIP's latency less critical than the Gemma denoising loop -- it is a one-time cost amortized over 10 denoising steps.

### Strategy A: Host-Side SigLIP (Lowest Implementation Effort)

```
  Host CPU/GPU                          TT Device
  +--------------------------+          +-------------------------+
  | SigLIP forward pass      |          |                         |
  | (PyTorch or JAX)         |  ------> | Receive [B, 768, 2048]  |
  | -> [B, 768, 2048] tokens |  H2D     | image tokens            |
  +--------------------------+          | Feed into Gemma prefix  |
                                        +-------------------------+
```

- Run SigLIP entirely on the host using PyTorch/JAX
- Transfer the 768 x 2048 = 1.57M elements (~3.1 MB at bf16) to the device
- Zero new TT-Blaze ops required
- Latency: SigLIP forward pass on CPU (~50-100ms for 3 images) + H2D transfer (~0.1ms for 3MB)
- Acceptable if total inference budget is >200ms

### Strategy B: TT-Blaze Pipeline Stage (Best Latency)

```
  Host                     TT Device
  +----------+             +------------------------------------+
  | 3 images |  -------->  | Stage 1: SigLIP (new/adapted ops)  |
  | [224^2x3]|   H2D       | Stage 2: Gemma prefix pass         |
  +----------+             | Stage 3: Gemma denoising loop x10  |
                           +------------------------------------+
```

- Implement SigLIP as a TT-Blaze graph with the new ops (layernorm, gelu, non-causal sdpa)
- Requires implementing 2-3 new ops (see gap analysis above)
- Eliminates CPU compute and the 3 MB transfer is replaced by 3 x 224x224x3 x 2B = ~900KB raw image transfer (smaller)
- Best latency, but highest implementation cost

### Strategy C: Hybrid (Pragmatic Starting Point)

```
  Host                     TT Device
  +--------------------+   +----------------------------------+
  | Patch embed +      |   | 27x Encoder1DBlock (ttnn ops)    |
  | posemb (reshape +  |-->| Final LN + projection head       |
  | matmul on host)    |   | Gemma prefix + denoising         |
  +--------------------+   +----------------------------------+
```

- Do patch extraction and posemb addition on host (simple reshape + matmul, trivial on CPU)
- Run the 27 encoder blocks on TT using ttnn high-level ops (`ttnn.layer_norm`, `ttnn.gelu`, `ttnn.transformer.scaled_dot_product_attention`)
- Avoids writing blaze-nn FusedOps for SigLIP -- uses ttnn directly
- The Gemma blocks continue to use TT-Blaze FusedOps as before

### Recommendation

**Start with Strategy A** (host-side SigLIP) to unblock end-to-end model bring-up. The 3 MB transfer overhead is negligible compared to the Gemma denoising loop latency. Once the full pipeline works, **migrate to Strategy C** (hybrid) to reduce host CPU load. Strategy B (full TT-Blaze) is only justified if SigLIP latency becomes a bottleneck, which is unlikely given it runs once per inference.

Source: `/tmp/openpi/src/openpi/models/pi0.py`, lines 234-237 (prefix caching), lines 239-271 (denoising loop).

---

## 8. Memory Planning for On-Device Execution

If running SigLIP on TT hardware (Strategy B or C), memory layout matters.

### Weight Memory

```
Total params: ~414M at bf16 = ~829 MB

Per-layer breakdown (27 layers):
  Attention weights (Q,K,V,O): 4 * 1152 * 1152 * 2B = ~10.6 MB
  Attention biases:            4 * 1152 * 2B         = ~9.2 KB
  MLP weights (up, down):      2 * 1152 * 4304 * 2B  = ~19.8 MB
  MLP biases:                  (4304 + 1152) * 2B    = ~10.9 KB
  LayerNorm (2 sets):          2 * 2 * 1152 * 2B     = ~9.2 KB
  -------------------------------------------------------
  Total per layer:             ~30.4 MB
  Total 27 layers:             ~821 MB

Non-layer params:
  Patch embed kernel:          588 * 1152 * 2B        = ~1.4 MB
  Projection head:             1152 * 2048 * 2B       = ~4.7 MB
  Final LayerNorm:             2 * 1152 * 2B          = ~4.6 KB
```

### Layer-Scan Execution (L1 Pressure Reduction)

With `scan=True`, the execution pattern processes one layer at a time with the same code:

```
for layer_idx in range(27):
    weights = all_weights[layer_idx]   # slice axis 0
    x = encoder_block(x, weights)
```

This means only **one layer's weights (~30.4 MB)** need to be in L1 at any time, not all 27 layers (~821 MB). The weights for each layer are streamed from DRAM to L1 sequentially.

### Activation Memory

```
Primary activation: [B, 256, 1152] at bf16 = B * 589,824 bytes
  For B=1: ~576 KB (easily fits in L1)

Attention intermediates (per head, per layer):
  Q, K, V:  [B, 16, 256, 72] each = B * 589,824 bytes each
  Scores:   [B, 16, 256, 256] = B * 2,097,152 bytes
  Total per-layer attention: ~B * 5.8 MB

MLP intermediates:
  Up-projected: [B, 256, 4304] = B * 2,203,648 bytes
  After GELU:   same
  Total per-layer MLP: ~B * 4.2 MB

Peak per-layer activation: ~B * 10 MB
```

For B=1, peak activation is ~10 MB, well within L1 capacity. The combination of ~30 MB weights + ~10 MB activations = ~40 MB per layer pass is manageable with weight streaming from DRAM.

### DRAM Bandwidth Requirement

Each layer requires streaming ~30 MB of weights from DRAM. Over 27 layers:
```
Total DRAM reads: 27 * 30.4 MB = ~821 MB
At 256 GB/s DRAM bandwidth: 821 MB / 256 GB/s = ~3.2 ms
```

This is the theoretical minimum. Actual time will be higher due to compute overlap and dispatch overhead, but SigLIP should complete in well under 10ms on TT hardware at batch size 1.

---

## 9. New Ops Required: Summary

```
+------------------+----------+------------------------------------------+
| Op               | Priority | Notes                                    |
+------------------+----------+------------------------------------------+
| layernorm        | P0       | 55 calls per forward pass. Maps to       |
|                  |          | ttnn.layer_norm. Accepts gamma + beta.   |
+------------------+----------+------------------------------------------+
| gelu             | P0       | 27 calls (one per MLP block). Maps to    |
|                  |          | ttnn.gelu. Consider fused matmul+gelu.   |
+------------------+----------+------------------------------------------+
| sdpa (non-causal)| P1       | 27 calls. Existing sdpa needs a no-mask  |
|                  |          | code path. head_dim=72 support needed.   |
+------------------+----------+------------------------------------------+
```

P0 ops block any on-device execution. P1 can be worked around by decomposing attention into separate matmul + softmax + matmul steps if needed.

---

## 10. End-to-End SigLIP Data Flow on TT

```
  HOST                                       TT DEVICE (if Strategy B/C)
  ====                                       =========

  3x [224,224,3] images                      
        |                                    
  (Optional: patch extract on host)          
        |                                    
        v                                    
  H2D transfer --------------------------->  [B, 256, 588] or [B, 224, 224, 3]
                                                    |
                                             F.linear(W_patch) + F.residual_add(b_patch)
                                                    |
                                             F.residual_add(posemb)  -- constant tensor
                                                    |
                                             +----- loop 27x -----+
                                             | F.layernorm(g, b)   |
                                             | F.linear(W_q)       |
                                             | F.linear(W_k)       |
                                             | F.linear(W_v)       |
                                             | F.sdpa(Q,K,V)       | <-- no causal mask
                                             | F.linear(W_o)       |
                                             | F.residual_add(x)   |
                                             | F.layernorm(g, b)   |
                                             | F.linear(W_up)      |
                                             | F.gelu()            |
                                             | F.linear(W_down)    |
                                             | F.residual_add(x)   |
                                             +---------------------+
                                                    |
                                             F.layernorm(g, b)     -- final encoder_norm
                                                    |
                                             F.linear(W_head)      -- 1152->2048 projection
                                                    |
                                             [B, 256, 2048] image tokens
                                                    |
                                             (repeat for 3 cameras, shared weights)
                                                    |
                                             [B, 768, 2048] --> Gemma prefix
```

---

## Key Takeaways

- **Two new blaze-nn ops are required for on-device SigLIP execution**: `layernorm` (with mean subtraction and bias, distinct from `rmsnorm`) and `gelu` (distinct from `silu`/`gated_reduce`). Both map directly to existing ttnn primitives.

- **The Conv2D patch embedding decomposes cleanly into reshape + matmul**, avoiding any need for a general convolution kernel. The reshape can be done on the host or folded into tilization.

- **Start with host-side SigLIP execution (Strategy A)** to unblock end-to-end bring-up. The ~3 MB H2D transfer for image tokens is negligible. Migrate to on-device execution later if needed.

- **Layer-scan execution keeps memory pressure low**: only ~30 MB of weights per layer need to be in L1 at a time, with ~10 MB of activations. The full 27-layer forward pass requires ~821 MB of DRAM reads, achievable in ~3-5ms at full bandwidth.

- **Non-causal attention is the lowest-risk gap**: the existing SDPA kernel likely just needs a flag to skip causal masking. The 256x256 attention matrix is small enough for materialized (non-flash) attention, simplifying the implementation.
