# Worked Example: Multi-Head Attention

Multi-head attention (MHA) is the most shape-intensive pattern in LLM inference. It involves Q/K/V linear projections, a reshape into multi-head format, scaled dot-product attention (SDPA) with causal masking, and an output projection. Each of these stages introduces shape, broadcast, format, and padding challenges that stress every subsystem in TensorAdapter. This file traces MHA through the automated pipeline, identifies where the abstraction handles the complexity well, where gaps remain, and where the escape hatch system becomes necessary.

**What you will learn:**

- The MHA computation graph and its shape transformations at each stage
- Shape and broadcast challenges: the 4D `[B, H, S, S]` attention score tensor
- Format transitions: bfloat16 activations into HiFi4 + fp32 SDPA, back to bfloat16
- Quantitative softmax error analysis under different profiles
- Dest register pressure under fp32\_dest\_acc\_en
- Padding challenges: causal mask with NEG\_INF, non-aligned sequence lengths
- Four gaps in the current abstraction and how escape hatches bridge them
- Comparison with TT-Blaze's existing `DsaSparseAttention` and `DsaPipeline`

---

## The PyTorch Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(attn)
```

For LLaMA-7B: `hidden_dim = 4096`, `num_heads = 32`, `head_dim = 128`.

---

## The Automated Path

```python
model = MultiHeadAttention(4096, 32)
blaze_model = blaze.from_pytorch(
    model,
    sample_input=torch.randn(1, 128, 4096),  # [B=1, S=128, D=4096]
    device=mesh_device,
    precision="balanced",
)
output = blaze_model(input_tensor)
```

---

## Shape Transformations Through MHA

The attention pattern involves four distinct shape regimes:

### Regime 1: Q/K/V Projections (3 matmuls)

```text
Input:  [B=1, S=128, D=4096]
Q:      [1, 128, 4096] -> matmul -> [1, 128, 4096] -> reshape -> [1, 32, 128, 128]
K:      [1, 128, 4096] -> matmul -> [1, 128, 4096] -> reshape -> [1, 32, 128, 128]
V:      [1, 128, 4096] -> matmul -> [1, 128, 4096] -> reshape -> [1, 32, 128, 128]

ShapeDescriptor for Q/K/V output (before reshape):
  logical_shape  = (128, 4096)  (collapsed from 3D for matmul)
  padded_shape   = (128, 4096)  (both already 32-aligned)
  tile_grid      = (4, 128)
  format         = bfloat16
```

### Regime 2: Attention Scores ($QK^T$)

```text
Q: [1, 32, 128, 128]  (B, H, S, D_head)
K: [1, 32, 128, 128]  (B, H, S, D_head) -- transposed to [1, 32, 128, 128]

Per-head matmul: [S, D] x [D, S] -> [S, S]
  = [128, 128] x [128, 128] -> [128, 128]

ShapeDescriptor for scores:
  logical_shape   = (1, 32, 128, 128)
  Per-head 2D:    (128, 128)
  tile_grid       = (4, 4) per head
  total_tiles     = 16 per head, 512 total across 32 heads
  format          = bfloat16
  page_size       = 2,048 B
```

### Regime 3: Softmax (with causal mask)

```text
Mask:   [1, 1, 128, 128]  -- broadcast across H dimension
Scores: [1, 32, 128, 128]

Broadcasting (Chapter 8):
  Dim 1: 1 vs 32 -> mask broadcast across 32 heads
  The same mask tile is reused for all heads -- no extra L1

Padding strategy transition:
  Q/K/V matmul:    ZERO padding (mathematically neutral for dot product)
  Attention scores: NEG_INF padding for masked/padded positions
    exp(-inf) = 0 in softmax -> zero attention weight (correct)
    exp(0)    = 1 in softmax -> non-zero weight (INCORRECT if ZERO used)
```

### Regime 4: Attention Output + Output Projection

```text
AV: weights [1, 32, 128, 128] x V [1, 32, 128, 128] -> [1, 32, 128, 128]
Reshape: [1, 32, 128, 128] -> [1, 128, 4096]
O_proj: [1, 128, 4096] -> matmul -> [1, 128, 4096]
```

---

## Shape Challenges

### Challenge 1: The 4D Attention Tensor

The attention score tensor has shape `[B, H, S, S]` -- four dimensions where only the last two are tiled. The batch and head dimensions are "virtual" dimensions that map to core parallelism rather than tile geometry:

```text
ShapeDescriptor for QK^T scores:
  logical_shape   = (1, 32, 128, 128)
  padded_shape    = (1, 32, 128, 128)    # all dims tile-aligned
  tile_shape      = (32, 32)             # tiles on last 2 dims only
  tile_grid_shape = (1, 32, 4, 4)        # B=1, H=32, S/32=4, S/32=4
  total_tiles     = 1 * 32 * 4 * 4 = 512
```

### Challenge 2: Broadcasting the Causal Mask

The causal mask has shape `[1, 1, S, S]` and must be broadcast across the head dimension:

```text
Scores shape:     [1, 32, 128, 128]
Mask shape:       [1,  1, 128, 128]
Broadcast result: [1, 32, 128, 128]

The BroadcastResolver (Chapter 8) detects row-broadcast along dim 1.
In hardware: the same mask tile is reused for all 32 heads.
No additional L1 is consumed for the expanded dimension.
```

### Challenge 3: Non-Aligned Sequence Lengths

For sequence length $S = 37$ (not a multiple of 32):

```text
Q (after reshape):
  logical_shape   = (1, 32, 37, 128)
  padded_shape    = (1, 32, 64, 128)     # ceil(37/32)*32 = 64
  padding_strategy= "ZERO" for Q/K/V

QK^T scores:
  logical_shape   = (1, 32, 37, 37)
  padded_shape    = (1, 32, 64, 64)      # both S dims padded
  padding_strategy= "NEG_INF"            # softmax requires -inf for padding
```

The padding transition from ZERO (Q/K/V) to NEG\_INF (scores) is handled automatically by the padding system ([Chapter 5](../ch05_padding/)) at the matmul-to-softmax boundary.

---

## Format Transitions at the Softmax Boundary

The softmax safety override (from [Chapter 6, File 04](../ch06_data_formats/04_precision_profiles_and_user_overrides.md)) forces specific math settings regardless of the global profile:

```python
# From Chapter 6, File 04 -- enforced regardless of profile
REQUIRED_CONSTRAINTS = {
    "sdpa":    {"math_fidelity": "HiFi4", "math_approx_mode": False,
                "fp32_dest_acc_en": True},
    "softmax": {"math_fidelity": "HiFi4", "math_approx_mode": False,
                "fp32_dest_acc_en": True},
}
```

The format transitions under the "performance" profile:

```text
Q/K/V projections:
  Weights: bfloat8_b, Math: LoFi, math_approx=True
  Output: bfloat16

QK^T matmul (within SDPA):
  Both inputs: bfloat16
  Math: HiFi4 (safety override)

Softmax:
  REQUIRED: HiFi4, fp32_dest_acc_en=True, exact math
  This is NEVER overridable by profiles, overrides, or format_hints.

Attention-Value matmul:
  Math: HiFi4 (within SDPA safety scope)

Output projection:
  Weights: bfloat8_b, Math: LoFi (back to profile)
```

The math fidelity transitions from LoFi (projections) to HiFi4 (SDPA) and back to LoFi (output projection). This requires `reconfig()` phase boundaries in the `FusedProgram`.

---

## Quantitative Softmax Error Analysis

The softmax function is the most precision-sensitive operation in the attention pipeline. To quantify the impact of the safety override, consider the error propagation through softmax.

### Error Model

For softmax with input $x_i$ and output $p_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$, the relative error in $p_i$ due to an error $\delta$ in $x_i$ is approximately:

$$\frac{\delta p_i}{p_i} \approx \delta \cdot (1 - p_i)$$

For uniform attention over 128 positions ($p_i = 1/128 \approx 0.0078$), a 1-ulp error in bfloat16 ($\delta \approx 2^{-7} \approx 0.0078$) produces a relative error of $0.0078 \times 0.9922 \approx 0.77\%$ per position. Over 128 positions with independent errors:

$$\text{total redistribution} \leq 128 \times 0.0077 \approx 1.0\%$$

### Profile Comparison

| Profile | exp() Error | Accumulation | Total Softmax Error | Acceptable? |
|---------|------------|-------------|--------------------|-----------------------|
| performance (without safety) | ~2 ulp (poly approx) | bfloat16 dest (7-bit mantissa) | ~2.5% weight redistribution | Marginal -- attention drift |
| performance (with safety) | <1 ulp (exact) | fp32 dest (23-bit mantissa) | <0.1% weight redistribution | Yes |
| balanced | <1 ulp (exact) | bfloat16 dest | ~0.5% weight redistribution | Yes |
| accuracy | <1 ulp (exact) | fp32 dest | <0.01% weight redistribution | Yes (near-reference) |

The safety override reduces softmax error by approximately $25\times$ compared to the unprotected "performance" configuration. This is why the REQUIRED\_CONSTRAINTS registry enforces it unconditionally.

---

## Dest Register Pressure in Attention

Attention operations interact with the dest register capacity constraint. The interaction is non-obvious and can cause significant throughput reduction.

### The $QK^T$ Matmul Under fp32\_dest\_acc\_en

With `S=128` and `head_dim=128`, each head's score matrix is `[128, 128]` = `4 x 4` tiles:

```text
Without fp32_dest_acc_en (before entering SDPA scope):
  compute_subblock_w(4, fp32_dest_acc_en=False) -> max_dest=8
  subblock_w = 4 (fits in dest)
  Inner loop: 1 iteration

With fp32_dest_acc_en + dst_full_sync_en (SDPA safety):
  compute_subblock_w(4, fp32_dest_acc_en=True, dst_full_sync_en=True) -> max_dest=8
  subblock_w = 4 (fits in dest)
  Inner loop: 1 iteration
```

### The Footgun: fp32\_dest\_acc\_en Without dst\_full\_sync\_en

If a developer uses Level 2 overrides to set `fp32_dest_acc_en=True` without `dst_full_sync_en=True`:

```text
compute_subblock_w(4, fp32_dest_acc_en=True, dst_full_sync_en=False) -> max_dest=4
subblock_w = 4 (barely fits)
Inner loop: 1 iteration

But if S=256 -> 8x8 tiles -> subblock_w=8 needed:
  max_dest=4 -> subblock_w=4 -> inner loop: 2 iterations -> 50% throughput loss!
```

The precision profile system prevents this footgun by always coupling `fp32_dest_acc_en` with `dst_full_sync_en`. But if a developer bypasses the profile via Level 2 overrides without this coupling, the `max_dest` drops to 4, potentially halving throughput for larger sequence lengths.

---

## The SDPA Fusion

In practice, TT-Blaze does not implement $QK^T$, softmax, and $AV$ as three separate ops. The `SDPADecode` kernel fuses all three into a single kernel that never writes the full `[S, S]` attention matrix to L1:

```text
SDPADecode (fused):
  For each output tile row (S_out):
    For each K chunk:
      Compute partial QK^T scores
      Apply scale and mask
      Compute partial exp() and running max/sum
    Finalize softmax normalization
    Compute weighted sum with V tiles
    Write output tile

This avoids materializing the full [S, S] matrix in L1.
For S=2048: full matrix = 2048*2048*2 = 8 MB (far exceeds ~1.5 MB L1!)
Fused SDPA: ~few tiles at a time = ~10 KB
```

---

## Gaps in the Automated Path

The attention pattern exposes four gaps in the current TensorAdapter abstraction:

### Gap 1: Paged KV Cache

Production attention implementations use a paged KV cache where past key/value tensors are stored in pre-allocated DRAM pages. The cache uses indirect addressing: a page table maps logical sequence positions to physical DRAM addresses. This requires:

- Custom NOC data movement patterns to read non-contiguous pages
- A separate page table tensor that does not fit the standard `ShapeDescriptor` model
- Variable-length K/V tensors that change size on every decode step

**Escape hatch:** Drop to the raw `FusedProgram` API for the SDPA portion, or use `ExternalTensor` for the KV cache tensor. See [File 06](./06_escape_hatches_for_power_users.md).

### Gap 2: Grouped-Query Attention (GQA)

GQA (used in LLaMA-2 70B, Mistral) has fewer K/V heads than Q heads:

```text
Standard MHA:  Q: [1, 32, S, D]   K: [1, 32, S, D]   -> no broadcast
GQA (8 KV):   Q: [1, 32, S, D]   K: [1, 8, S, D]    -> broadcast K 4x
```

The `BroadcastResolver` ([Chapter 8](../ch08_broadcasting/)) handles simple head-dimension broadcasting, but production GQA implementations use optimized head-mapping patterns (interleaved vs. contiguous head groups) that affect NOC topology. The adapter's default broadcast inserts generic mcast nodes; hand-tuned code uses topology-aware multicast.

**Escape hatch:** Override the core grid via per-op `overrides`.

### Gap 3: Sparse Attention Patterns

Architectures like DeepSeek V3 use sparse attention (`DsaSparseAttention`) with block-diagonal attention masks. The standard SDPA pattern recognition does not identify sparse attention -- it falls back to dense SDPA.

**Escape hatch:** Use the existing `DsaSparseAttention` Blaze op via Level 2 bypass.

### Gap 4: Rotary Position Embeddings (RoPE)

RoPE applies a rotation to Q and K before the $QK^T$ matmul:

$$q_{\text{rot}} = q \cdot \cos(\theta) + \text{rotate\_half}(q) \cdot \sin(\theta)$$

The adapter handles the computation correctly (elementwise mul + add), but the position embeddings must be provided as additional inputs with proper broadcasting along the sequence dimension.

**Escape hatch:** Provide position embeddings as external inputs with shape hints.

### What Works vs. What Needs Escape Hatches

| Component | Adapter Handles | Escape Hatch Needed |
|-----------|----------------|---------------------|
| Q/K/V weight projections | Fully automatic | No |
| Weight format + fidelity | Profile + override | No |
| $QK^T$ matmul shapes | Automatic from head/seq dims | No |
| Causal mask broadcasting | BroadcastResolver | No |
| Softmax precision | Mandatory safety constraint | No |
| Attention-Value matmul | Automatic | No |
| Output projection | Fully automatic | No |
| Paged KV cache | -- | Yes: ExternalTensor |
| GQA head topology | -- | Yes: grid override |
| Sparse attention | -- | Yes: custom op |
| RoPE embedding input | Partial | Minor: shape hint |

---

## L1 Budget for MHA

For LLaMA-7B attention (hidden=4096, heads=32, head\_dim=128, S=128):

```text
Q/K/V Projections (fused, 3 matmuls):
  Weight CBs: 3 * DRAM-streaming (3 * 102 KB) = 306 KB
  Activation CB: ~256 KB
  Output CBs: 3 * ~50 KB = 150 KB
  Subtotal: ~712 KB

SDPA (fused QK^T + softmax + AV):
  Q tiles per head: 4 * 4 = 16 tiles = 32 KB
  K tiles per head: 16 tiles = 32 KB
  V tiles per head: 16 tiles = 32 KB
  Attention intermediate: ~4 tiles (streaming) = 8 KB
  Subtotal per head: ~104 KB
  [Heads parallelized across cores]

Output projection:
  Weight CB: 102 KB (DRAM-streaming)
  Activation CB: ~256 KB
  Output CB: ~50 KB
  Subtotal: ~408 KB

Peak (if Q/K/V fused with SDPA): ~712 + 104 = ~816 KB
Available L1: ~1,300 KB
L1 utilization: 816 / 1,300 = 62.8%
CB IDs: ~18 (across 3 phases, well under 64)
```

The attention pattern uses significantly more L1 than the gated MLP (62.8% vs. 27.5%) because of the three weight projection CBs. Under the performance profile with bfloat8\_b weights, the weight CBs are 47% smaller, bringing utilization to approximately 45%.

---

## Comparison: Adapter Path vs. DsaSparseAttention

The hand-tuned `DsaSparseAttention` op includes custom NCRISC orchestration, block-sparse mask evaluation, and fused online softmax normalization:

| Metric | Adapter Path (Dense SDPA) | DsaSparseAttention |
|--------|--------------------------|-------------------|
| Compute | $O(S^2 \cdot d)$ | $O(S \cdot d \cdot \text{nnz})$ (sparse) |
| L1 residency | Q streamed, K/V may spill | K/V kept in L1 via paging |
| Softmax | Standard two-pass (max + normalize) | Online single-pass |
| Dense perf gap | Baseline | 8-15% faster (NCRISC + softmax) |
| Sparse perf gap | Baseline | 50-200% faster (skip zero blocks) |

For dense attention (full causal mask, short sequences), the gap is 8-15% from NCRISC scheduling and softmax implementation differences. For sparse attention, the gap is fundamental.

---

## The Recommended Hybrid Approach

```python
# PROPOSED -- hybrid model for production attention
class LLaMADecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.mlp = GatedMLP(config.hidden_dim, config.intermediate_dim)

# Automated: MLP and norm via TensorAdapter
blaze_mlp = blaze.from_pytorch(
    decoder.mlp, sample_input=torch.randn(1, 4096),
    device=device, precision="performance",
)

# Manual: Attention via existing DsaPipeline
attn_program = build_dsa_pipeline(config, device)

# Hybrid execution
def decode_step(hidden, kv_cache):
    normed = blaze_norm(hidden)
    attn_out = attn_program.run(normed, kv_cache)  # manual path
    mlp_out = blaze_mlp(normed)                     # automated path
    return hidden + attn_out + mlp_out
```

This hybrid approach uses TensorAdapter for the 80% of ops that work well with automation (MLP, norm) while retaining manual control for the 20% that need specialized handling (attention with KV cache). This matches the 80/20 expectation from [Chapter 1, File 04](../ch01_the_gap/04_abstraction_goals_and_scope.md).

---

## Key Takeaways

- Multi-head attention is the most shape-intensive pattern in LLM inference. It involves 4D tensors, broadcasting (causal mask), format transitions (softmax safety override), and padding strategy changes (ZERO for matmul, NEG\_INF for softmax).
- The softmax safety override enforces HiFi4 + `fp32_dest_acc_en` + exact math regardless of profile. This reduces softmax error by $\sim 25\times$ compared to unprotected "performance" configuration. It is a hard constraint in the REQUIRED\_CONSTRAINTS registry.
- Dest register pressure under `fp32_dest_acc_en` without `dst_full_sync_en` can halve throughput for larger sequences. The precision profile system prevents this footgun by coupling the two flags.
- Four gaps remain: (1) paged KV cache requires custom NOC patterns, (2) GQA head topology requires topology-aware multicast, (3) sparse attention patterns require specialized kernels, (4) RoPE requires external position embedding inputs. Each gap has an escape hatch.
- The recommended production approach is a **hybrid model**: TensorAdapter for MLP and normalization, manual Blaze ops for attention with KV cache.
- L1 utilization for the attention block is 62.8% (balanced profile), compared to 27.5% for the gated MLP. Attention is the pattern most likely to trigger L1 pressure and automatic phase splitting.

## Source Files

- `blaze/ops/dsa_attention/op.py` -- `DsaSparseAttention` (production attention op)
- `blaze/ops/dsa_pipeline/op.py` -- `DsaPipeline` (multi-phase attention pipeline)
- `blaze/ops/matmul/op.py` -- `Matmul.emit()` (Q/K/V and output projections)
- `blaze/context.py` -- `ExternalTensor` (escape hatch for paged KV cache)
- `blaze/utils.py` -- `compute_subblock_w()` (dest capacity under fp32\_dest\_acc\_en)
- `blaze/compiler.py` -- `_get_compute_config()` (fidelity configuration)

---

**Next:** [`05_performance_implications_and_tradeoffs.md`](./05_performance_implications_and_tradeoffs.md)
