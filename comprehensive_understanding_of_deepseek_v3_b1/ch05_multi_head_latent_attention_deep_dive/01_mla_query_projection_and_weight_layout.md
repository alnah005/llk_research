# 5.1 MLA Query Projection and Weight Layout

This section traces every tensor shape transformation in the Q-path of DeepSeek V3's Multi-head Latent Attention (MLA) mechanism, from the raw hidden state entering the attention block through to the final Q heads ready for SDPA. We follow the data through RMSNorm, two successive matmuls with an intermediate normalization, a weight-shuffled split into QNOPE/QROPE heads, a batched matmul for QNOPE absorption, and RoPE for QROPE -- making the dimensional arithmetic explicit at every boundary. The KV-path shares the first normalization and mcast with the Q-path; its divergent path is covered in [Section 5.2](./02_compressed_kv_cache.md).

Source: `fused_ops/pre_sdpa/op.py`, `blitz_decode_weights.py`, `prepare_weights.py`

---

## 5.1.1 Architectural Context

MLA compresses the per-token KV cache from $40{,}960$ elements (standard MHA) to 576 elements — a $71\times$ element-level reduction, ~$134\times$ with BFP8 (see [Section 1.1.1](../ch01_architecture_overview_and_model_mapping/01_deepseek_v3_architecture_on_tenstorrent.md) for the full derivation). The consequence for the query path: a two-stage compression-decompression pipeline with latent dimension 1536, splitting into NOPE and ROPE head components. Under TP=2, each device handles 64 of the 128 heads.

---

## 5.1.2 End-to-End Q-Path Shape Trace

The Q-path implements the low-rank factorization that is the defining innovation of MLA. Instead of a single large $W_Q$ matrix, DeepSeek V3 uses a two-stage projection with an intermediate bottleneck:

$$x \xrightarrow{\text{RMSNorm}} \hat{x} \xrightarrow{W_{q\_a}} c_q \xrightarrow{\text{RMSNorm}} \hat{c}_q \xrightarrow{W_{q\_b}} [q_{\text{nope}} \| q_{\text{rope}}]$$

The full mathematical chain:

$$\mathbf{h}_\text{norm} = \text{RMSNorm}(\mathbf{x},\; \gamma_\text{attn}) \in \mathbb{R}^{1 \times 7168}$$

$$\mathbf{c}_q = \mathbf{h}_\text{norm} \cdot W_{q\_a} \in \mathbb{R}^{1 \times 1536}$$

$$\hat{\mathbf{c}}_q = \text{RMSNorm}(\mathbf{c}_q,\; \gamma_q) \in \mathbb{R}^{1 \times 1536}$$

$$\mathbf{q} = \hat{\mathbf{c}}_q \cdot W_{q\_b} \in \mathbb{R}^{1 \times 12288}$$

$$\mathbf{Q}_\text{nope},\; \mathbf{Q}_\text{rope} = \text{split}(\mathbf{q},\; [64 \times 128,\; 64 \times 64])$$

$$\mathbf{Q}'_\text{nope} = \text{BatchedMatmul}(\mathbf{Q}_\text{nope},\; W_{kv\_b1}) \in \mathbb{R}^{64 \times 1 \times 512}$$

$$\mathbf{Q}'_\text{rope} = \text{RoPE}(\mathbf{Q}_\text{rope}) \in \mathbb{R}^{64 \times 1 \times 64}$$

The final query for each head is the concatenation $[\mathbf{Q}'_\text{nope} \| \mathbf{Q}'_\text{rope}] \in \mathbb{R}^{1 \times 576}$, matching the 576-dim compressed KV cache entry. This concatenation is how MLA eliminates the need to expand the KV cache through $W_{kv\_b}$ at decode time -- the query is projected *into* the compressed space rather than the KV being expanded *out* of it.

### Tensor Shape Trace Through the Pipeline

The pipeline stages follow the same flow as [ch04 Section 4.2.2](../ch04_the_fused_op_library/02_attention_fused_ops.md); the table below adds per-core shard shapes and tile counts:

| Stage | Operation | Input Shape | Output Shape | Grid |
|-------|-----------|-------------|--------------|------|
| 1 | RMSNorm (attn\_norm) | $[1, 7168]$ | $[1, 7168]$ | Single core (12,9) |
| 2 | Mcast1 | $[1, 7168]$ | $[1, 7168]$ replicated | (12,9) $\to$ 13$\times$10 grid |
| 3 | Matmul1 (q\_a\_proj) | $[1, 7168] \times [7168, 1536]$ | $[1, 16]$ per core | 96 cores (12$\times$8) |
| 4 | Gather-Reduce | 96 $\times$ $[1, 16]$ | $[1, 1536]$ | 96 cores $\to$ (12,9) |
| 5 | RMSNorm2 (q\_norm) | $[1, 1536]$ | $[1, 1536]$ | Single core (12,9) |
| 6 | Mcast2 | $[1, 1536]$ | $[1, 1536]$ replicated | (12,9) $\to$ 96 cores |
| 7 | Matmul2 (q\_b\_proj) | $[1, 1536] \times [1536, 128]$ per core | $[1, 128]$ per core | 96 cores (8$\times$12) |
| 8a | Matmul3 (kv\_b1\_proj) | $[1, 128] \times [128, 512]$ per head | $[1, 512]$ per head | 64 Qnope cores (8$\times$8) |
| 8b | QRoPE | $[1, 128]$ per core (2 heads) | $[1, 128]$ per core | 32 Qrope cores (4$\times$8) |
| 9 | CreateQHeads | 64$\times[1,512]$ + 64$\times[1,64]$ | $[8, 576]$ per SDPA core | 8 SDPA input cores |

The total Q-path latent dimension at the Matmul2 output is $64 \times 128 + 64 \times 64 = 8192 + 4096 = 12288$, distributed across the 96-core grid at $128$ elements per core.

---

## 5.1.3 Weight Shape Catalog with Shard Layouts

All attention weights, their logical shapes, hardware tile formats, sharding strategies, and the core grids they occupy:

| Weight | Logical Shape | Tile | Sharding | Core Grid | Shard Shape | Tiles/Shard | Dtype | Fusion Group |
|--------|---------------|------|----------|-----------|-------------|-------------|-------|--------------|
| `q_a_proj` | $[7168, 1536]$ | $32 \times 32$ | WIDTH | $(0,0)$-$(11,7)$ [96 cores] | $[3584, 32]$ | 112 | BFP8 | `q_ab_kv_a` |
| `q_b_proj` | $[1536, 12288]$ | $32 \times 32$ | WIDTH | $(0,0)$-$(11,7)$ [96 cores] | $[1536, 128]$ | 192 | BFP8 | `q_ab_kv_a` |
| `kv_a_proj` | $[7168, 576]$ | $32 \times 32$ | WIDTH | $(0,8)$-$(8,9)$ [18 cores] | $[7168, 32]$ | 224 | BFP8 | `q_ab_kv_a` |
| `kv_b1_proj` | $[8192, 512]$ | $32 \times 32$ | HEIGHT | $(0,0)$-$(7,7)$ [64 cores] | $[128, 512]$ | 64 | BFP8 | `kv_b12` |
| `kv_b2_proj` | $[512, 8192]$ | $32 \times 32$ | HEIGHT | $(8,0)$-$(12,7)$ + $(0,8)$-$(11,9)$ [64 cores] | $[128, 512]$ | 64 | BFP8 | `kv_b12` |
| `o_proj` | $[8192, 7168]$ | $32 \times 32$ | WIDTH | $(0,0)$-$(11,7)$ + $(1,8)$-$(8,9)$ [112 cores] | $[8192, 64]$ | 512 | BFP8 | `o_proj_gate_mm_norms` |
| `attn_norm` ($\gamma_1$) | $[1, 7168]$ | $1 \times 32$ | Single core | $(12, 9)$ | $[1, 7168]$ | 224 | BF16 | `o_proj_gate_mm_norms` |
| `q_norm` ($\gamma_2$) | $[1, 1536]$ | $1 \times 32$ | Single core | $(12, 9)$ | $[1, 1536]$ | 48 | BF16 | `o_proj_gate_mm_norms` |
| `kv_norm` ($\gamma_{kv}$) | $[1, 512]$ | $1 \times 32$ | Single core | $(0, 8)$ | $[1, 512]$ | 16 | BF16 | `o_proj_gate_mm_norms` |
| `ffn_norm` ($\gamma_{ffn}$) | $[1, 7168]$ | $1 \times 32$ | Single core | $(12, 9)$ | $[1, 7168]$ | 224 | BF16 | `o_proj_gate_mm_norms` |

---

## 5.1.4 Weight Preparation Pipeline (HuggingFace to Hardware)

The transformation chain from HuggingFace checkpoint to on-device weights:

```
HuggingFace state_dict
  q_a_proj.weight: [1536, 7168]    (out_features, in_features)
  q_b_proj.weight: [24576, 1536]   (full TP, out_features, in_features)
  kv_a_proj_with_mqa.weight: [576, 7168]
  kv_b_proj.weight: [32768, 512]   (combined kv_b1 + kv_b2)
  o_proj.weight: [7168, 16384]     (full TP)
  input_layernorm.weight: [7168]
  q_a_layernorm.weight: [1536]
  kv_a_layernorm.weight: [512]

     | Transpose: .T for all matmul weights
     | Split: kv_b_proj -> kv_b1 [8192, 512] + kv_b2 [512, 8192]
     | TP slice: q_b_proj [1536, 24576] -> per-TP [1536, 12288]
     | Unsqueeze: norms [D] -> [1, D]
     v

Per-device logical shapes:
  q_a_proj:  [7168, 1536]
  q_b_proj:  [1536, 12288]
  kv_a_proj: [7168, 576]
  kv_b1:     [8192, 512]
  kv_b2:     [512, 8192]
  o_proj:    [8192, 7168]

     | Shuffle: q_a_proj -> K-interleave [3584, 3072]
     | Shuffle: q_b_proj -> interleaved QNOPE/QROPE columns
     | Shuffle: kv_a_proj -> shard reorder for KV cache branch layout
     | Convert: BFP8 quantization for matmul weights
     | Fuse: overlap into 3 fusion groups
     v

On-device fused tensors (WIDTH_SHARDED, L1)
```

### The kv_b_proj Split

The HF `kv_b_proj` weight $[32768, 512]$ is split at head-dim index 128 into `kv_b1` (K-NOPE) and `kv_b2` (V) via `_split_kv_b_proj` (see [Section 1.1.1](../ch01_architecture_overview_and_model_mapping/01_deepseek_v3_architecture_on_tenstorrent.md) for derivation). Per-device shapes after TP=2 shard:

| Weight | Full-model | Per-device (TP=2) | Post-shuffle |
|--------|-----------|-------------------|--------------|
| `kv_b1` | $[16384, 512]$ | $[8192, 512]$ | — (HEIGHT_SHARDED as-is) |
| `kv_b2` | $[512, 16384]$ | $[512, 8192]$ | $[8192, 512]$ via `shuffle_kv_b2` (HEIGHT_SHARDED) |

---

## 5.1.5 Fusion Groups (Overlapped Tensors)

Three fusion groups cover all attention weights (see [Section 4.1.5](../ch04_the_fused_op_library/01_fusion_principles_and_patterns.md) for the OverlappedTensor mechanism). This section details the per-core byte-level stacking:

**Group 1: `q_ab_kv_a`** -- Fuses `q_a_proj`, `q_b_proj`, and `kv_a_proj` into one buffer.

On the 96-core q_ab grid (cols 0-11, rows 0-7):

```
Core (x, y) in [0..11] x [0..7]:
  +-------------------------------------------+
  |  q_a_proj packed shard                    |  Offset 0
  |  (3584, shard_w) = q_a packed H/2, 2W    |
  +-------------------------------------------+
  |  q_b_proj shuffled shard                  |  Offset after q_a
  |  (1536, shard_w)                          |
  +-------------------------------------------+
```

On the 18-core kv_a grid (cols 0-8, rows 8-9):

```
Core (x, y) in [0..8] x [8..9]:
  +-------------------------------------------+
  |  kv_a_proj shard                          |  Offset 0
  |  (7168, 32)                               |
  +-------------------------------------------+
```

**Group 2: `kv_b12`** -- Fuses `kv_b1_proj` and `kv_b2_proj`.
- `kv_b1` on 64 cores (the 8x8 QNOPE grid), `kv_b2` on 64 different cores.
- Non-overlapping: no core holds both.

**Group 3: `o_proj_gate_mm_norms`** -- Fuses `o_proj`, `gate_mm`, and all four gamma vectors.

```
Core (12, 9) -- "gamma core":
  +--------------------------------------+
  |  attn_norm gamma  (1, 7168)  14336 B |  Offset 0
  +--------------------------------------+
  |  q_norm gamma     (1, 1536)   3072 B |  Offset 14336
  +--------------------------------------+
  |  ffn_norm gamma   (1, 7168)  14336 B |  Offset 17408
  +--------------------------------------+

Core (0, 8) -- "kv_norm core":
  +--------------------------------------+
  |  kv_norm gamma    (1, 512)    1024 B |  Offset 0
  +--------------------------------------+
```

---

## 5.1.6 The q_a_proj K-Split and Packed Weight Layout

The first matmul ($[1, 7168] \times [7168, 1536] \to [1, 1536]$) uses a K-split strategy to distribute the inner dimension across cores. The key insight is that `q_a_proj` is **packed** by interleaving K-halves:

```
Original q_a_proj: [7168, 1536]    (K=7168 rows, N=1536 columns)

K-interleave packing (shuffle_q_a):
  weights.reshape(2, 3584, 1536).permute(1, 0, 2).reshape(3584, 3072)

  Row 0    -> Packed row 0, left half
  Row 3584 -> Packed row 0, right half
  Row 1    -> Packed row 1, left half
  Row 3585 -> Packed row 1, right half
  ...

Packed shape: [3584, 3072]    (H/2 rows, 2*W columns)
```

**Why pack?** The packed layout allows the WIDTH_SHARDED matmul to compute both K-halves in a single pass per core. Each of the 96 cores gets a $[3584, 32]$ shard (112 tiles of $32 \times 32$). The first 48 cores handle the first K-half and the second 48 cores handle the second K-half, then results are reduced via gather-reduce on core (12,9).

### K-Split Reduction Detail

```
Matmul1 output: 96 cores, each [1, 16] (partial sum over K/2 rows)

K-split topology:
  Half 0: cores 0..47  (first 48 cores in row-major order)
  Half 1: cores 48..95 (second 48 cores)

  Core i (half 0) and core i+48 (half 1) produce partial sums
  for the same output columns.

Gather-Reduce on core (12, 9):
  Phase 1: receive half-0 partials from cores 0..47 -> accumulate into CB 7
  Phase 2: receive half-1 partials from cores 48..95 -> into CB 8
  TRISC: add corresponding tiles element-wise
  Result: [1, 1536] = 3 tiles of 16x32
```

The K-split approach is chosen because the output dimension (1536) is relatively small, so a pure N-split would leave each core with only 16 elements -- too little work to amortize the dispatch. K-splitting keeps each core's inner loop 112 tiles long, which is enough to saturate the Tensix FPU.

---

## 5.1.7 The q_b_proj Shuffled Weight Layout

The second matmul ($[1, 1536] \times [1536, 12288] \to [1, 12288]$) uses **shuffled weights** so the output lands directly on the correct QNOPE/QROPE cores without post-matmul data movement. This is the `shuffle_weights_for_interleaved_qnope_qrope()` function.

### Standard vs Shuffled Column Order

Without shuffling, q_b_proj output would be:
```
[QNOPE_head0..63 (8192 elements) | QROPE_head0..63 (4096 elements)]
```

With shuffling, the output is interleaved by row groups of 8 heads:
```
Row 0: [QNOPE_0..7 (1024 elems) | QROPE_0..7 (512 elems)]   = 1536 per row
Row 1: [QNOPE_8..15 (1024)      | QROPE_8..15 (512)]
...
Row 7: [QNOPE_56..63 (1024)     | QROPE_56..63 (512)]
```

Each row's 1536 elements are sharded across 12 cores:
- **Cols 0-7** (QNOPE): 8 cores x 128 elements = 1024 elements per row (1 head per core)
- **Cols 8-11** (QROPE): 4 cores x 128 elements = 512 elements per row (2 heads per core, 64 each)

### Shuffled Weight Construction

```python
def shuffle_weights_for_interleaved_qnope_qrope(weights, ...):
    K = weights.shape[0]                           # 1536
    qnope_weights = weights[:, :8192]              # [1536, 8192]
    qrope_weights = weights[:, 8192:12288]         # [1536, 4096]

    qnope_heads = qnope_weights.reshape(K, 64, 128)   # [1536, 64, 128]
    qrope_heads = qrope_weights.reshape(K, 64, 64)    # [1536, 64, 64]

    for row in range(8):   # 8 row groups of 8 heads each
        # Append QNOPE heads for this row
        shuffled_cols.append(qnope_heads[:, row*8:(row+1)*8, :].reshape(K, -1))  # [1536, 1024]
        # Append QROPE heads for this row
        shuffled_cols.append(qrope_heads[:, row*8:(row+1)*8, :].reshape(K, -1))  # [1536, 512]

    return torch.cat(shuffled_cols, dim=1)    # [1536, 12288] with shuffled columns
```

The result is a weight matrix where each row group's output block is:

$$[\underbrace{\text{QNOPE}_{8i} \ldots \text{QNOPE}_{8i+7}}_{8 \times 128 = 1024} \;|\; \underbrace{\text{QROPE}_{8i} \ldots \text{QROPE}_{8i+7}}_{8 \times 64 = 512}]$$

Each core on the 96-core grid receives $12288 / 96 = 128$ elements. The shuffle is applied at weight preparation time, so there is zero runtime overhead.

### Grid Layout After Matmul2 Output

```
     Col 0   1   2   3   4   5   6   7 |  8   9  10  11
Row 0  NP0 NP1 NP2 NP3 NP4 NP5 NP6 NP7| RP01 RP23 RP45 RP67    <- heads 0-7
Row 1  NP8 NP9 ... ... ... ... ... NP15| RP89 ...  ...  RP1415  <- heads 8-15
...
Row 7  NP56 ...                   NP63 | RP5657 ... RP6263       <- heads 56-63

Legend:
  NPn  = Qnope head n: [1, 128] = 4 tiles of 1x32
  RPab = Qrope heads a,b: [2, 64] = 4 tiles of 1x32 (2 heads packed)
```

---

## 5.1.8 The kv_a_proj Shard Reorder

The $W_{kv\_a}$ weight is $(7168, 576)$ width-sharded across 18 cores (a $9 \times 2$ grid at rows 8-9, columns 0-8). The output splits as $[512_{\text{nope}}, 64_{\text{rope}}]$ and the nope portion goes to the RMSNorm core while the rope portion goes to 2 K-RoPE cores. The shard ordering must place the correct output columns on the correct cores.

The reorder map from `QAB_KVA_PROJ_SingleDeviceOverlapSpec`:

```python
kv_a_proj_shard_order = (0, 1, 2, 3, 4, 5, 6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 17)
```

Shard 16 (the boundary shard containing the transition from NOPE to ROPE data) is moved to position 8 in the physical layout, placing it on core $(8, 8)$ -- the first K-RoPE core. This ensures the DKV matmul output naturally lands on the correct cores without post-matmul data movement.

---

## 5.1.9 QNOPE Absorption: Matmul3 with kv_b1_proj

After matmul2 produces the QNOPE heads $[64, 1, 128]$, a third matmul absorbs the KV decompression into the query side:

$$q_{\text{nope\_out}} = q_{\text{nope}} \cdot W_{kv\_b1} \quad : \quad [64, 1, 128] \times [64, 128, 512] \to [64, 1, 512]$$

This is a **batched matmul** across 64 heads, executed on the 8x8 QNOPE grid (64 cores, one per head):

```
QNOPE GRID: Cols 0-7 x Rows 0-7 (64 cores)

   Col 0     1     2     3     4     5     6     7
R0  H0     H1    H2    H3    H4    H5    H6    H7
R1  H8     H9    H10   H11   H12   H13   H14   H15
...
R7  H56    H57   H58   H59   H60   H61   H62   H63

Per-core:
  Input:   matmul2_output_cb [1, 128] = 4 tiles of 1x32
  Weights: matmul3_weights_cb (128, 512) = 4 x 16 tiles (from kv_b12 overlap)
  Compute: [1, 128] x [128, 512] = [1, 512]
  Output:  matmul3_output_cb [1, 512] = 16 tiles of 1x32
```

**Why does the query need kv_b1?** In standard attention, $Q \cdot K^T$ computes the attention score. In MLA, the KV cache stores the compressed representation $\mathbf{c}_{kv}$ (after normalization), not the expanded key. The attention score is:

$$\text{score} = \mathbf{Q}_\text{nope} \cdot W_{kv\_b1} \cdot \mathbf{c}_{kv}^T$$

By absorbing $W_{kv\_b1}$ into the query (Matmul3), the Flash MLA kernel can compute $\mathbf{Q}' \cdot \mathbf{c}_{kv}^T$ directly against the compressed cache -- no expansion needed at decode time. The input and weights are both already sharded on the same core, so there is no data movement for this stage -- it is purely compute-bound.

---

## 5.1.10 RoPE Application on QROPE and KROPE

Both the query rope component and the key rope component undergo Rotary Position Embedding. The implementation uses the Meta-style `rotate_half` formulation:

$$\text{RoPE}(\mathbf{x}, \text{pos}) = \mathbf{x} \odot \cos(\theta_\text{pos}) + \text{rotate\_half}(\mathbf{x}) \odot \sin(\theta_\text{pos})$$

where $\text{rotate\_half}$ swaps the two halves of the vector and negates the first half:

$$\text{rotate\_half}([x_0, \ldots, x_{d/2-1}, x_{d/2}, \ldots, x_{d-1}]) = [-x_{d/2}, \ldots, -x_{d-1}, x_0, \ldots, x_{d/2-1}]$$

### Implementation via `trans_mat`

Rather than implementing `rotate_half` as an explicit element rearrangement, the kernel implements it as a matrix multiplication with a pre-computed `trans_mat` tensor stored as a sharded tensor on the RoPE cores. This is a sparse $32 \times 32$ matrix that simultaneously permutes and negates the appropriate elements.

### Position-Indexed Cos/Sin Lookup

The cos and sin tables are stored in DRAM as tensors of shape $[\text{max\_seq\_len}, d_\text{rope}]$. At decode time, NCRISC reads a single row from each table, indexed by the current position:

1. NCRISC reads `position_ids_tensor` to determine the current sequence position.
2. NCRISC calculates the DRAM page offset: `page_offset = position_id * cos_sin_page_size`.
3. NCRISC reads one page from `cos_tensor_address + page_offset` into `cos_cb`.
4. NCRISC reads one page from `sin_tensor_address + page_offset` into `sin_cb`.
5. TRISC computes: `out = x * cos + (x @ trans_mat) * sin`.

### QROPE Configuration

On the 32 Qrope cores (columns 8-11, rows 0-7), each core processes 2 QROPE heads:

```
QROPE GRID: Cols 8-11 x Rows 0-7 (32 cores)

   Col 8     9    10    11
R0  H0,1   H2,3  H4,5  H6,7      <- 2 heads/core, 64 elements/head
R1  H8,9   H10,11 ...  H14,15
...
R7  H56,57 H58,59 H60,61 H62,63

Per core: qrope_head_dim_per_core_t = 2, qrope_num_heads_per_core = 2
```

### KROPE Configuration (KV Branch)

On the 2 KROPE cores at (8,8) and (8,9), each core processes 1 key rope vector:
- `krope_Wt = 1` (64 / 32 = 2, but only 1 tile per core)
- Input comes from `dkv_matmul_output_cb` (the rope portion of the DKV matmul output)
- Output goes to `k_rope_output_cb`

---

## 5.1.11 CreateQHeads: Assembly into SDPA Input Format

After QNOPE absorption and QROPE RoPE, the heads must be combined and tilized for FlashMLA. The target layout is $[1, 1, 64, 576]$ where 576 = 512 (nope) + 64 (rope) per head.

The **CreateQHeads** operation performs a 3-phase unicast from the 96 Q-path cores to 8 SDPA input cores:

| Source Row | Target Core | Heads |
|-----------|-------------|-------|
| Row 0 | (0, 1) | Heads 0-7 |
| Row 1 | (1, 1) | Heads 8-15 |
| Row 2 | (2, 1) | Heads 16-23 |
| Row 3 | (3, 1) | Heads 24-31 |
| Row 4 | (0, 2) | Heads 32-39 |
| Row 5 | (1, 2) | Heads 40-47 |
| Row 6 | (2, 2) | Heads 48-55 |
| Row 7 | (3, 2) | Heads 56-63 |

Mapping: `source_row r -> target core (r % 4, 1 + r // 4)`

The 3-phase transfer uses dedicated semaphores to avoid data corruption:

| Phase | Data | Senders | Semaphore |
|-------|------|---------|-----------|
| Phase 1 (NOPE first half) | 8 Qnope heads, first 256 bytes each | 8 Qnope cols 0-3 per row | `nope_phase1_semaphore` (reuses gather sem ID 2) |
| Phase 2 (NOPE second half) | 8 Qnope heads, second 256 bytes each | 8 Qnope cols 4-7 per row | `nope_phase2_semaphore` (reuses gather sem ID 3) |
| Phase 3 (ROPE) | 8 Qrope heads, 128 bytes each | 4 Qrope cols 8-11 per row | `rope_semaphore` (reuses mcast sem ID 0) |

The data arrives in an intermediate CB (`create_q_heads_receiver_in_cb`, index 31) in row-major order. TRISC then tilizes this data into the output CB (`create_q_heads_out_cb`, index 16) in the $8 \times 576$ tile format ($8 \times 32$ tiles, 18 tiles wide) that Flash MLA expects.

Per SDPA input core: 8 heads x (512 + 64) = 4608 elements, tiled as $\text{PNHt} = 1, \text{DHt} = 18$.

---

## 5.1.12 TP=2 Head Distribution

### Global vs. Per-Device Head Counts

| Parameter | Full Model | Per Device (TP=2) |
|-----------|-----------|-------------------|
| Total Q heads | 128 | 64 |
| $q\_b\_proj$ width | $1536 \times 24576$ | $1536 \times 12288$ |
| $kv\_b1\_proj$ height | $16384 \times 512$ | $8192 \times 512$ |
| $o\_proj$ height | $16384 \times 7168$ | $8192 \times 7168$ |
| Q NOPE elements | 16384 | 8192 ($= 64 \times 128$) |
| Q ROPE elements | 8192 | 4096 ($= 64 \times 64$) |

For TP=2 on a 4x2 mesh, mesh column 0 receives TP-slice 0 (heads 0-63) and mesh column 1 receives TP-slice 1 (heads 64-127). The per-TP fused tensors are concatenated along width and distributed via `ShardTensor2dMesh(dims=(None, 1))`.

### Per-Device Independence in the Q-Path

A critical design insight: the entire Q-path from RMSNorm1 through CreateQHeads executes **identically and independently** on each device after the CCL broadcast delivers $x$. There is no cross-device communication during the Q-path stages. Each device computes its own $q_a$ and $q_b$ using replicated $W_{q\_a}$ and its TP-local $W_{q\_b}$ slice, then processes its local 64-head Q tensor.

Cross-device coordination only occurs:
- **Before** the Q-path: CCL broadcast of $x$.
- **After** FlashMLA: `SDPAReduceToAll` merges partial attention statistics (Section 5.3).
- **After** output projection: CCL all-reduce of `o_proj` partial products (Section 5.4).

---

## 5.1.13 Tile Format Transitions Along the Q-Path

| Stage | CB | Tile Format | Elements per Tile | Bytes (BF16) |
|-------|----|-------------|-------------------|--------------|
| RMSNorm1 input/output | CB 0, 2 | $32 \times 32$ | 1024 | 2048 |
| Mcast1 dest / Matmul1 input | CB 5 | $1 \times 32$ | 32 | 64 |
| Matmul1 output / Gather-Reduce | CB 4, 7, 8 | $1 \times 32$ / $16 \times 32$ | varies | varies |
| RMSNorm2 | CB 7, 6, 9 | $16 \times 32$ | 512 | 1024 |
| Mcast2 dest / Matmul2 input | CB 10 | $1 \times 32$ | 32 | 64 |
| Matmul2-3 output | CB 12, 14 | $1 \times 32$ | 32 | 64 |
| QRoPE output | CB 15 | $1 \times 32$ | 32 | 64 |
| CreateQHeads output | CB 16 | $8 \times 32$ | 256 | 512 |

The critical reformat happens at the mcast boundaries: the sender produces $32 \times 32$ or $16 \times 32$ tiles, but the receiver interprets the same bytes as $1 \times 32$ tiles. This is a zero-copy reinterpretation -- the mcast sends raw bytes, and the destination CB is configured with a different tile format than the source. The total byte count is identical: for Mcast1, $7 \text{ tiles} \times 2048 \text{ B} = 14336 \text{ B} = 224 \text{ tiles} \times 64 \text{ B}$.

---

## 5.1.14 Full Q-Path Core Grid Diagram

```
     Col 0   1   2   3   4   5   6   7   8   9  10  11  12
     -------------------------------------------------------
R0  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  |     |
R1  | MM  |*RC* |*RC* |*RC* |*RC* | MM  | MM  | MM  | MM  | MM  | MM  | MM  |     |
R2  | MM  |*RC* |*RC* |*RC* |*RC* | MM  | MM  | MM  | MM  | MM  | MM  | MM  |     |
R3  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  |     |
R4  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  |     |
R5  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  |     |
R6  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  |     |
R7  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  | MM  |     |
R8  |     |     |     |     |     |     |     |     | KR  |     |     |     |     |
R9  |     |     |     |     |     |     |     |     | KR  | RN  |     |     |     |
     -------------------------------------------------------
                                                         (12,9) = RMSNorm
                                                                  GatherReduce
                                                                  Mcast Sender

  STAGE 1 (Matmul1 q_a_proj):
    ALL 96 cores [0..11] x [0..7] = MM (matmul with K-split)

  STAGE 2 (RMSNorm2 + Mcast2):
    Core (12, 9) computes RMSNorm2, then mcasts to all 96 cores

  STAGE 3 (Matmul2 q_b_proj):
    ALL 96 cores = MM (matmul, 4 tiles output/core)

  STAGE 4a (Matmul3 kv_b1_proj):        STAGE 4b (QRoPE):
    Cols 0-7, Rows 0-7 = 64 cores        Cols 8-11, Rows 0-7 = 32 cores

  STAGE 5 (CreateQHeads unicast):
    96 senders -> 8 receivers at *RC* positions

  Legend:
    MM  = Matmul core (Stages 1-3, then splits)
    *RC* = SDPA Input receiver core (Stage 5 target)
    KR  = K-RoPE core (KV branch, not Q-path)
    RN  = RMSNorm / Gather / Mcast core = (9, 9) physical
```

---

**Prev:** [Chapter 4 -- The Fused-Op Library](../ch04_the_fused_op_library/index.md) | **Next:** [Compressed KV Cache](02_compressed_kv_cache.md)
