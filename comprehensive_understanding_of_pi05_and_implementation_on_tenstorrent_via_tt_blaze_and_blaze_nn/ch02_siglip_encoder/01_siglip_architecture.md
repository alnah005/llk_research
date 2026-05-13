# SigLIP So400m/14 Architecture Deep Dive

## Context

The SigLIP (Sigmoid Loss for Language-Image Pre-training) vision encoder used in pi0.5 is the So400m/14 variant -- a ViT with ~400M parameters, 14x14 patch size, and 27 transformer layers. It is instantiated in `pi0.py` (line 82-90) with `variant="So400m/14"`, `pool_type="none"`, and `scan=True`. Understanding its exact architecture is a prerequisite for mapping it onto Tenstorrent hardware, since every op in the forward pass must have a TT-Blaze equivalent or decomposition.

This file walks through the forward pass top-to-bottom, with precise shapes, parameter counts, and source references.

---

## 1. Configuration

The `So400m/14` variant string is decoded by `decode_variant()` in `siglip.py` (lines 298-373):

```
width     = 1152        # embedding dimension
depth     = 27          # number of encoder blocks
mlp_dim   = 4304        # MLP hidden dimension (3.73x width, not 4x)
num_heads = 16          # attention heads
patch_size = (14, 14)   # patch extraction kernel
```

Derived values:
- `head_dim = width / num_heads = 1152 / 16 = 72`
- No GQA: `num_kv_heads = num_heads = 16` (standard MHA)
- Input image resolution: `224 x 224 x 3` (from `IMAGE_RESOLUTION` in `model.py`, line 47)
- Spatial grid: `224 / 14 = 16` patches per side, `16 x 16 = 256` total spatial tokens

Additional instantiation flags from `pi0.py` (lines 82-89):
- `num_classes = paligemma_config.width = 2048` (projection head output dim)
- `pool_type = "none"` (all 256 spatial tokens preserved)
- `scan = True` (layer-scan for memory efficiency)
- `dtype_mm = config.dtype = "bfloat16"` (compute dtype for matmuls)

Source: `/tmp/openpi/src/openpi/models/siglip.py`, lines 298-373; `/tmp/openpi/src/openpi/models/pi0.py`, lines 82-89.

---

## 2. Forward Pass Overview

```
  224x224x3 image
       |
  [Conv2D patch embed]  -- kernel=14, stride=14, VALID padding
       |
  [B, 16, 16, 1152]  -> reshape -> [B, 256, 1152]
       |
  [+ sincos2d posemb]   -- precomputed, not learned
       |
  [cast to bfloat16]
       |
  [27x Encoder1DBlock]  -- LayerNorm + MHA + LayerNorm + MLP (GELU)
       |
  [final LayerNorm]      -- encoder_norm
       |
  [pool_type="none"]     -- pass-through, all 256 tokens kept
       |
  [Dense 1152->2048]     -- projection head ("head")
       |
  [B, 256, 2048]         -- image tokens ready for PaliGemma
```

---

## 3. Patch Extraction (Conv2D Embedding)

Source: `siglip.py`, lines 216-227.

```python
x = nn.Conv(self.width, self.patch_size, strides=self.patch_size,
            padding="VALID", name="embedding", dtype=jnp.float32)(image)
```

This is a 2D convolution with:
- Input: `[B, 224, 224, 3]` (float32, explicitly cast at line 213)
- Kernel: `[14, 14, 3, 1152]` -- 14x14 spatial, 3 input channels, 1152 output channels
- Stride: `(14, 14)` -- non-overlapping patches
- Padding: VALID (no padding)
- Output: `[B, 16, 16, 1152]`
- Bias: yes (Flax Conv includes bias by default)

After convolution, reshape to `[B, 256, 1152]` (line 226):

```python
n, h, w, c = x.shape   # B, 16, 16, 1152
x = jnp.reshape(x, [n, h * w, c])  # [B, 256, 1152]
```

**Parameter count**: kernel = `14 * 14 * 3 * 1152 = 677,376` + bias = `1,152` = **678,528 parameters**.

**Key observation**: Because stride equals kernel size and padding is VALID, this Conv2D is mathematically equivalent to:
1. Extract non-overlapping 14x14x3 patches -> `[B, 256, 588]` (where 588 = 14*14*3)
2. Matmul with reshaped kernel `[588, 1152]` + bias

This decomposition eliminates the need for a dedicated Conv2D kernel on TT hardware.

---

## 4. Sinusoidal 2D Positional Embeddings

Source: `siglip.py`, lines 27-37 (`posemb_sincos_2d`), called at line 229.

```python
x = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)
```

With `posemb="sincos2d"` (decoded from variant), this calls `posemb_sincos_2d(h=16, w=16, width=1152)`:

```python
def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    y, x = jnp.mgrid[:h, :w]                    # grid indices [16,16] each
    omega = jnp.arange(width // 4) / (width // 4 - 1)  # [288] values in [0,1]
    omega = 1.0 / (temperature ** omega)         # frequency schedule
    y = jnp.einsum("m,d->md", y.flatten(), omega)  # [256, 288]
    x = jnp.einsum("m,d->md", x.flatten(), omega)  # [256, 288]
    pe = jnp.concatenate([sin(x), cos(x), sin(y), cos(y)], axis=1)  # [256, 1152]
    return pe[None, :, :]                        # [1, 256, 1152]
```

The positional embedding:
- Output shape: `[1, 256, 1152]`
- Is **not learned** -- computed deterministically from grid coordinates
- Uses the MoCo v3 convention: concatenate `[sin_x, cos_x, sin_y, cos_y]` where each is `width//4 = 288` dimensions
- Added in float32 (before the cast to bfloat16 at line 239)
- `width % 4 == 0` is asserted (1152 % 4 = 0, satisfied)

**For TT-Blaze**: This tensor is constant across all inputs. Precompute once on the host and transfer as a static weight. Apply via `residual_add`.

---

## 5. Encoder Block Structure (Encoder1DBlock)

Source: `siglip.py`, lines 75-108.

Each of the 27 encoder blocks follows a pre-norm residual pattern:

```
          x
          |
    [LayerNorm]
          |
    [Multi-Head Attention]  -- 16 heads, head_dim=72, bidirectional
          |
     + residual (x)
          |
          x
          |
    [LayerNorm]
          |
    [MLP: Dense(4304) -> GELU -> Dense(1152)]
          |
     + residual (x)
          |
          x
```

### 5.1 LayerNorm (NOT RMSNorm)

```python
y = nn.LayerNorm(dtype=self.dtype_mm)(x)
```

Flax `nn.LayerNorm` implements full Layer Normalization:
```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

This includes:
- **Mean subtraction** (RMSNorm omits this)
- **Learnable bias (beta)** in addition to scale (gamma)
- Parameters per LayerNorm: `gamma[1152] + beta[1152] = 2304`
- Total LayerNorm instances: 2 per block * 27 blocks + 1 final encoder_norm = **55 LayerNorm ops**

### 5.2 Multi-Head Dot-Product Attention

```python
y = nn.MultiHeadDotProductAttention(
    num_heads=self.num_heads,       # 16
    kernel_init=nn.initializers.xavier_uniform(),
    deterministic=deterministic,
    dtype=self.dtype_mm,            # bfloat16
)(y, y)  # self-attention: query=key=value=y
```

Attention configuration:
- **16 heads, head_dim = 1152/16 = 72** (Flax derives head_dim from width/num_heads)
- **Full bidirectional** attention -- no causal mask, no `mask` argument passed
- **Standard MHA** -- not grouped-query attention (GQA). All 16 heads have independent K and V projections
- Q, K, V projections: each `Dense(1152 -> 1152)` (reshaped as `[B, 256, 16, 72]`)
- Output projection: `Dense(1152 -> 1152)`

Parameter count per attention block:
```
Q projection:  1152 * 1152 + 1152 = 1,328,256
K projection:  1152 * 1152 + 1152 = 1,328,256
V projection:  1152 * 1152 + 1152 = 1,328,256
O projection:  1152 * 1152 + 1152 = 1,328,256
Total:         5,310,720 + 4,608 bias = ~5.3M per block
```

Attention computation (standard scaled dot-product):
```
scores = (Q @ K^T) / sqrt(72)     # [B, 16, 256, 256]
weights = softmax(scores)          # no mask applied
output = weights @ V               # [B, 16, 256, 72]
output = reshape -> [B, 256, 1152] # concatenate heads
output = O_proj(output)            # [B, 256, 1152]
```

The attention matrix is `[256, 256]` per head -- all spatial tokens attend to all others. This is relatively small (65K elements per head) compared to language model attention.

### 5.3 MLP Block (GELU Activation)

Source: `siglip.py`, lines 53-72.

```python
x = nn.Dense(4304, dtype=self.dtype_mm)(x)   # up-project
x = nn.gelu(x)                                # GELU activation
x = nn.Dense(1152, dtype=self.dtype_mm)(x)   # down-project
```

```
    [B, 256, 1152]
         |
    Dense(1152 -> 4304)  + bias
         |
    GELU activation
         |
    Dense(4304 -> 1152)  + bias
         |
    [B, 256, 1152]
```

This is a standard 2-layer MLP with GELU, **NOT** the gated SwiGLU/GeGLU pattern used in Gemma. There is no gate projection and no `gated_reduce` op.

Parameter count per MLP block:
```
Up:   1152 * 4304 + 4304 = 4,954,512
Down: 4304 * 1152 + 1152 = 4,962,360  (wait -- no, 4304*1152 = 4,958,208 + 1152 = 4,959,360)
```

Correcting:
```
Up:   1152 * 4304 = 4,958,208 + 4,304 bias = 4,962,512
Down: 4304 * 1152 = 4,958,208 + 1,152 bias = 4,959,360
Total: ~9.9M per block
```

**GELU** (Gaussian Error Linear Unit):
```
GELU(x) = x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

This is distinct from SiLU/Swish (`x * sigmoid(x)`) used in Gemma blocks. The existing `clamped_silu` or `gated_reduce(activation="silu")` ops in blaze-nn will NOT produce correct results for SigLIP.

---

## 6. Encoder Wrapper and Layer Scanning

Source: `siglip.py`, lines 111-161.

With `scan=True`, the encoder uses `nn.scan` (lines 126-145) to execute all 27 blocks with shared-structure weight stacking:

```python
x, scan_out = nn.scan(
    block,
    variable_axes={"params": 0},    # params stacked along axis 0
    split_rngs={"params": True, "dropout": True},
    in_axes=nn.broadcast,
    length=self.depth,              # 27
)(name="encoderblock", ...)(x, deterministic)
```

This means parameter tensors for all 27 layers are stacked along axis 0:
- Q weight: `[27, 1152, 16, 72]` instead of 27 separate `[1152, 16, 72]` tensors
- LayerNorm gamma: `[27, 1152]`
- etc.

After the 27 blocks, a final LayerNorm is applied (line 161):
```python
return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out
```

---

## 7. Pooling and Projection Head

Source: `siglip.py`, lines 253-289.

With `pool_type="none"` (line 266-267):
```python
elif self.pool_type == "none":
    pass  # no pooling, all 256 tokens preserved
```

No CLS token is used. No pooling is applied. All 256 spatial tokens proceed to the projection head.

The projection head maps from SigLIP width (1152) to PaliGemma width (2048):
```python
head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head")
```

Where `num_classes = paligemma_config.width = 2048` (set in `pi0.py` line 83).

- Input: `[B, 256, 1152]`
- Output: `[B, 256, 2048]`
- Parameters: `1152 * 2048 + 2048 = 2,361,344`

The `head_zeroinit=True` default means this layer's kernel is initialized to zeros, but at inference time we use the trained weights.

---

## 8. Three-Camera Processing

Source: `pi0.py`, lines 113-125.

```python
for name in obs.images:
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
    tokens.append(image_tokens)
```

The three cameras defined in `pi0_config.py` (lines 71-73):
- `base_0_rgb`
- `left_wrist_0_rgb`
- `right_wrist_0_rgb`

Each produces `[B, 256, 2048]` image tokens. These are concatenated along the sequence dimension:

```
Camera 1 (base):        [B, 256, 2048]  \
Camera 2 (left wrist):  [B, 256, 2048]   |-- concatenate --> [B, 768, 2048]
Camera 3 (right wrist): [B, 256, 2048]  /
```

All 768 image tokens have `ar_mask = False` (line 125), meaning they participate in **full bidirectional attention** -- image tokens can attend to all other image tokens and language tokens, but causal action tokens cannot attend backward to them.

**Shared weights**: All three cameras are processed by the same `self.PaliGemma.img` module. There is a single set of SigLIP weights, not three copies.

---

## 9. Parameter Count Summary

```
Component                  Per-block        Total (27 blocks)
-----------------------------------------------------------------
LayerNorm (2 per block)    2 * 2,304        124,416
Attention (Q,K,V,O)        ~5.31M           ~143.4M
MLP (up + down)            ~9.92M           ~267.9M
-----------------------------------------------------------------
Encoder blocks subtotal                     ~411.4M

Patch embedding (Conv2D)                    ~0.68M
Positional embedding                        0 (computed, not learned)
Final LayerNorm                             2,304
Projection head (1152->2048)                ~2.36M
-----------------------------------------------------------------
Total                                       ~414.5M parameters
```

At bfloat16 (2 bytes per param): **~829 MB** for the full SigLIP encoder.

---

## 10. Comparison: SigLIP vs Gemma Transformer Blocks

```
+--------------------+---------------------------+---------------------------+
| Feature            | SigLIP Encoder Block      | Gemma Decoder Block       |
+--------------------+---------------------------+---------------------------+
| Normalization      | LayerNorm (mean+var+bias) | RMSNorm (no mean, no bias)|
| Activation         | GELU                      | SwiGLU (SiLU-gated)       |
| MLP structure      | 2-layer (up, down)        | 3-layer (gate, up, down)  |
| Attention          | Standard MHA (16 heads)   | GQA (8 heads, 1 KV head)  |
| Head dim           | 72                        | 256                       |
| Position encoding  | Sinusoidal 2D (additive)  | RoPE (rotary, per-layer)  |
| Attention mask     | Bidirectional (full)      | Causal + prefix           |
| Bias terms         | Yes (all Dense + LN)      | No                        |
| Sequence length    | 256 (fixed)               | Variable (up to context)  |
+--------------------+---------------------------+---------------------------+
```

Every row in this table represents an op difference that affects TT-Blaze porting.

---

## Key Takeaways

- **SigLIP So400m/14 produces 256 spatial tokens per camera (768 total for 3 cameras)** at dimension 2048 after the projection head, forming the visual prefix for PaliGemma. The patch grid is fixed at 16x16 from 224x224 inputs.

- **The architecture is a standard ViT with no frills**: pre-norm residual blocks, standard MHA, 2-layer GELU MLP, sinusoidal positional embeddings. There is no CLS token, no pooling, no GQA, no gating.

- **Every major op differs from Gemma**: LayerNorm vs RMSNorm, GELU vs SiLU, standard MHA vs GQA, additive sinusoidal posemb vs RoPE, biases everywhere vs no biases. The blaze-nn ops built for Gemma cannot be reused without modification.

- **Layer scanning stacks all 27 layers' params along axis 0**, enabling a loop-based execution pattern that reduces L1 pressure -- only one layer's weights need to be resident at a time.

- **~414M parameters (~829 MB at bf16)** with a fixed, input-independent compute graph. The 256x256 attention matrix per head is small enough to fit entirely in SRAM, making SigLIP a good candidate for efficient on-device execution.
