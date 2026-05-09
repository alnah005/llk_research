# 5.2 Compressed KV Cache

This section traces the KV-path of MLA: how the hidden state is projected into a compressed 576-dimensional KV representation, split into NOPE and ROPE components, normalized and position-encoded, then written into the ND-sharded DRAM KV cache in BFP8 format. The KV-path runs concurrently with the Q-path (Section 5.1) inside the same `pre_sdpa` fused kernel, sharing the mcast'd normalized hidden state.

Source: `fused_ops/kv_cache_branch/op.py`, `micro_ops/kv_cache_update/op.py`, `fused_ops/pre_sdpa/op.py`

---

## 5.2.1 End-to-End KV-Path Shape Trace

The KV-path implements the second half of MLA's low-rank factorization. While Q uses a two-stage projection ($7168 \to 1536 \to 12288$), KV uses a single compression ($7168 \to 576$) and stores the compressed representation directly in the cache:

$$\hat{x} \xrightarrow{W_{kv\_a}} dkv \xrightarrow{\text{split}} [kv_{\text{nope}}, k_{\text{rope}}] \xrightarrow{\text{RMSNorm, RoPE}} \text{cache}[\text{pos}]$$

The critical insight: the Q-path decompresses the 576-dimensional latent to full 12288 dimensions via $W_{q\_b\_proj}$ and then recompresses Qnope via $W_{kv\_b1\_proj}$. But the KV cache stores only the 576-dimensional compressed form. This asymmetry -- expensive Q computation, cheap KV storage -- is the core of MLA's efficiency.

```
Stage 0 -- Normalized Hidden State (shared with Q-path)
  Shape: [1, 7168]
  Tiles:  224 x [1x32]  (already mcast to full grid during Q-path Stage 1)
  Layout: L1, replicated across matmul grid

     |  kv_a_proj matmul: [1, 7168] x [7168, 576] -> [1, 576]
     |  W_kv_a_proj: WIDTH_SHARDED on 18 cores ((0,8)-(8,9))
     |    Shard: [7168, 32] = 224 x [32x32] tiles per core
     |    18 shards x 32 = 576 output columns
     v

Stage 1 -- Compressed KV Vector (pre-split)
  Shape: [1, 576]        (kv_lora_rank=512 + rope_dim=64)
  Distribution: 18 cores, each holding [1, 32] = 1 x [1x32] tile
    Cores 0..15: 16 cores x 32 = 512 elements -> KNOPE
    Cores 16..17 ((8,8) and (8,9)): 2 cores x 32 = 64 elements -> KROPE

     |  Split is implicit via core assignment (no data movement)
     |  The kv_a_proj shard_order ensures correct core-to-component mapping
     v

Stage 2a -- KNOPE Path: Gather + RMSNorm
  Gather: 16 KNOPE cores send [1, 32] each via NOC_0 to core (0, 8)
  Input:  [1, 512]       (gathered from 16 matmul output cores)
  Core:   (0, 8) -- the kv_norm dedicated core

  RMSNorm computation:
    gamma_kv_norm: [1, 512], 1 x [16x32] tile, on core (0, 8)
    scalar: 1/sqrt(512) = 0.04419... (precomputed as float_to_uint32)
    output = x * rsqrt(variance + eps) * gamma

  Output: [1, 512]   (1 x [16x32] tile, on core (0,8))

Stage 2b -- KROPE Path: Rotary Position Embedding
  Input:  [1, 64]        (on 2 rope cores, 32 elements each)
  Cores:  (8, 8) and (8, 9)

  Per-core RoPE (same mechanism as QROPE):
    cos: [1, 32], sin: [1, 32] (fetched from DRAM at position_ids offset)
    trans_mat: [32, 32] (sharded, 1 tile per core)
    output = x * cos + (x @ trans_mat) * sin

  Output: [1, 64]   (2 x [1x32] tiles, on the 2 rope cores)

     |  Concatenation is implicit: NOPE on core (0,8), ROPE on cores (8,8)-(8,9)
     |  KV cache update reads from both locations
     v

Stage 3 -- Full KV Cache Entry
  Shape: [1, 576]        (512 NOPE + 64 ROPE = 576)
  Layout: Distributed across 3 cores: (0,8), (8,8), (8,9)
  Written to DRAM KV cache at position = cur_pos
```

---

## 5.2.2 DKV Matmul: The 9x2 Grid

The DKV matmul ($[1, 7168] \times [7168, 576] \to [1, 576]$) runs on the 18-core KV grid concurrently with the Q-path matmuls on the 96-core Q grid:

```
  DKV MATMUL GRID: 9x2 = 18 cores (Cols 0-8 x Rows 8-9)

     Col 0   1   2   3   4   5   6   7   8
  R8  DK0  DK1  DK2  DK3  DK4  DK5  DK6  DK7  KR     <- (8,8) = K-RoPE
  R9  DK9  DK10 DK11 DK12 DK13 DK14 DK15 DK16 KR     <- (8,9) = K-RoPE

  Note: Cores (8,8) and (8,9) have dual roles:
    - They ARE part of the dkv_matmul_weights_core_grid
    - They are ALSO krope_core_grid (K-RoPE)
    - The dkv_gather_sender_grid EXCLUDES them:
      dkv_gather_sender_grid = dkv_matmul_weights_core_grid - krope_core_grid
      = 16 sender cores (DK0-DK7, DK9-DK16)
```

### Core Role Differentiation

The 18 DKV matmul cores are tagged with four boolean compile-time flags:

| Flag | True Cores | False Cores | Purpose |
|------|-----------|-------------|---------|
| `is_dkv_matmul_core` | All 18 | Remaining grid | Run DKV matmul |
| `is_kv_rmsnorm_core` | (0, 8) only | All others | Run KV RMSNorm |
| `is_knope_core` | 16 NOPE cores | ROPE + others | Send gather to RMSNorm core |
| `is_krope_core` | (8,8), (8,9) | All others | Run K-ROPE |

### Per-Core Matmul

```
Activation (mcast'd):
  [1, 7168] = 224 x [1x32] tiles (same data as Q-path, already in L1 via mcast)

Weight W_kv_a_proj:
  Logical: [7168, 576], width-sharded on 18 cores
  Shard: [7168, 32] = 224 x [32x32] tiles per core
  Per-shard bytes (BFP8): 224 x 1088 = 243,712 bytes ~ 238 KB

Per-core output: 1 tile of 1x32 = 32 elements (64 bytes)
  K-loop: k_num_tiles = 7168 / 32 = 224 iterations
```

---

## 5.2.3 DKV Gather: 16 Cores to RMSNorm Core

After the DKV matmul, the 16 NOPE cores each hold a $[1, 32]$ result. These must be gathered to the RMSNorm core (0, 8) for normalization. This is a gather (concatenation), not a gather-reduce -- there is no K-split to sum.

```
  DKV Gather Topology (NOC0)

  16 sender cores (DK0-DK7, DK9-DK16):
  +------+  +------+  +------+       +------+
  | DK0  |  | DK1  |  | DK2  |  ...  |DK16  |
  | 1x32 |  | 1x32 |  | 1x32 |       | 1x32 |
  +--+---+  +--+---+  +--+---+       +--+---+
     |         |         |               |
     |    NOC0 unicast to receiver       |
     |    offset = sender_index * 64 B   |
     v         v         v               v
  +----------------------------------------------+
  |  Receiver: Core (0, 8) = dkv_rmsnorm         |
  |  kv_rmsnorm_input_cb (CB 25 in fused mode)   |
  |  16 tiles of 1x32 = [1, 512]                 |
  |  (reinterpreted as 1 tile of 16x32)          |
  +----------------------------------------------+
```

**Synchronization:** Senders use NOC0 with semaphore ID 2 (`gather_noc0_receiver_semaphore_id`). The receiver on BRISC waits for all 16 senders to signal, then reserves pages in `kv_rmsnorm_input_cb`. The sender grid coordinates are passed as compile-time args so each sender can compute its linear index and write offset.

---

## 5.2.4 KV RMSNorm: Single-Core Normalization

On core $(0, 8)$, the KV RMSNorm normalizes the gathered 512-element nope vector:

$$kv_{\text{nope}}' = \frac{kv_{\text{nope}}}{\sqrt{\frac{1}{512}\sum_{i=1}^{512} kv_{\text{nope},i}^2 + \epsilon}} \odot \gamma_{kv\_norm}$$

| Parameter | Value |
|-----------|-------|
| Input CB | `kv_rmsnorm_input_cb` (CB 25), 1 tile of $16 \times 32$ |
| Gamma CB | `kv_rmsnorm_gamma_cb` (CB 26), 1 tile of $16 \times 32$ |
| Output CB | `kv_rmsnorm_output_cb` (CB 27), 1 tile of $16 \times 32$ |
| Scalar | $1/\sqrt{512}$, packed as uint32 compile-time arg |
| Epsilon | $10^{-6}$, packed as uint32 compile-time arg |

The gamma vector $\gamma_{kv\_norm}$ is stored on core $(0, 8)$ within the `o_proj/gate/gamma` overlap group at byte offset 0 (it is the sole occupant of this core's shard).

**L1 cost for KV RMSNorm:** $3 \times 1024 = 3072$ bytes (3 CBs of $16 \times 32$).

---

## 5.2.5 K-RoPE: Rotary Embedding on 2 Cores

Simultaneously with KV RMSNorm, the 2 K-RoPE cores at $(8, 8)$ and $(8, 9)$ apply RoPE to their matmul output:

```
  K-RoPE Core Pipeline (per core)

  +-----------------------------------------------------------+
  |  Core (8, 8) or (8, 9)                                    |
  |                                                           |
  |  Input: dkv_matmul_output_cb (CB 24), 1 tile of 1x32     |
  |  NCRISC: Read cos[pos], sin[pos] from DRAM                |
  |          Read position from position_ids_tensor            |
  |                                                           |
  |  TRISC: rotate_half -> cos_mul -> sin_mul -> sum           |
  |  Output: krope_output_cb (CB 28), 1 tile of 1x32         |
  |                                                           |
  |  Wt = 1 tile, Ht = 1 head                                 |
  |  total_Wt = 2 (across 2 cores: start_tile_offset 0, 1)    |
  +-----------------------------------------------------------+
```

The `start_tile_offset` per-core compile-time descriptor gives core $(8, 8)$ offset 0 and core $(8, 9)$ offset 1, so each core knows which portion of the 64-element rope vector it handles.

**L1 cost per K-RoPE core:** ~2.6 KB (dominated by the $32 \times 32$ transformation matrix at 2048 bytes).

---

## 5.2.6 KV Cache Layout and ND Sharding

The KV cache is the only tensor in the attention path that lives in DRAM. It uses **ND sharding**, a Tenstorrent extension that shards a 4D tensor along multiple dimensions simultaneously:

```
Logical shape: [batch, 1, max_seq_len, 576]
  batch = 1 (single-batch decode)
  nkv = 1  (MLA: single KV head)
  max_seq_len = configurable (e.g., 8192, 16384)
  kv_dim = 576 (512 nope + 64 rope)

ND shard specification:
  Shard along dim 2 (seq_len) with shard_height = k_chunk_size (typically 128)
  Each shard: [1, 1, 128, 576]
```

### Physical DRAM Layout

```
8 DRAM banks, distributed by S-block proximity:
  Optimal bank order: (1, 3, 2, 0, 5, 7, 6, 4)

Per-bank pages: ceil(max_seq_len / k_chunk_size / 8) shards
Page size: k_chunk_size * kv_dim tiles in BFP8
         = 128 * 18 tiles * 1088 bytes = 2,506,752 bytes per shard
```

### Tile Layout Within Each DRAM Shard

```
  One DRAM shard (128 x 576 elements, BFP8):

  4 tile-rows x 18 tile-cols = 72 tiles of 32x32

  +-----+-----+-----+-----+-----+-- ... --+-----+-----+
  | T00 | T01 | T02 | ... | T015|         | T016| T017|
  +-----+-----+-----+-----+-----+-- ... --+-----+-----+
  | T10 | T11 | ...                        |     | T117|
  +-----+-----+-----+-----+-----+-- ... --+-----+-----+
  | T20 | ...                                    | T217|
  +-----+-----+-----+-----+-----+-- ... --+-----+-----+
  | T30 | T31 | ...                        |     | T317|
  +-----+-----+-----+-----+-----+-- ... --+-----+-----+

  Nope region: cols 0-15 (512 elements)
  Rope region: cols 16-17 (64 elements)
```

---

## 5.2.7 BFP8 Format and DRAM Savings

The KV cache stores data in **BFP8** (block floating point, 8-bit) format to minimize DRAM bandwidth during attention:

$$\text{BFP8 tile size} = 1088 \text{ bytes per } 32 \times 32 \text{ tile}$$
$$\text{BF16 tile size} = 2048 \text{ bytes per } 32 \times 32 \text{ tile}$$
$$\text{Compression ratio} = 2048 / 1088 \approx 1.88\times$$

### Sequence Length Budget

For a maximum sequence length of $S_{\max} = 8192$:

$$\text{KV cache size (BFP8)} = 1 \times 1 \times 8192 \times 576 \times \frac{1088}{1024} \approx 5.0 \text{ MB}$$

In BF16:

$$1 \times 1 \times 8192 \times 576 \times 2 = 9,437,184 \text{ bytes} \approx 9.0 \text{ MB}$$

The BFP8 format saves ~4 MB of DRAM per layer, or ~240 MB across 60 layers. This reduction is essential for fitting the full model's KV caches within the DRAM budget.

---

## 5.2.8 KV Cache Update: BFP8 Conversion and DRAM Write

After KV RMSNorm and K-RoPE complete, the results must be written to the KV cache in DRAM. The cache update requires a read-modify-write cycle because BFP8 tiles share exponents across rows. The `KVCacheUpdate` micro-op handles the format conversion.

### BFP8 Conversion Pipeline

```
  Step 1: READ existing tile column from DRAM
          NCRISC reads 16 BFP8 tiles (nope) + 2 tiles (rope)
          into kv_cache_input_cb

  Step 2: UNTILIZE BFP8 -> BF16
          TRISC converts tiles to row-major BF16
          in kv_cache_intermed_cb

  Step 3: OVERWRITE row at position
          BRISC copies nope data from kv_rmsnorm_output_cb
          and rope data from krope_output_cb
          into the correct row of the intermediate buffer

  Step 4: TILIZE BF16 -> BFP8
          TRISC converts back to BFP8 tiles
          in kv_cache_output_cb

  Step 5: WRITE modified tiles back to DRAM
          NCRISC writes 18 BFP8 tiles back
```

### Nope/Rope Core Split

The KV cache update runs on the merged grid of 3 cores:

```
  Core (0, 8) = is_nope_core:
    Input:  kv_rmsnorm_output_cb = [1, 512] normalized nope
    Reads:  16 BFP8 tiles from DRAM (nope column of cache)
    Writes: 16 BFP8 tiles back to DRAM after row replacement

  Core (8, 8) = is_rope_core:
    Input:  krope_output_cb = [1, 32] rope (first half)
    Reads:  1 BFP8 tile from DRAM (rope column 16)
    Writes: 1 BFP8 tile back

  Core (8, 9) = is_rope_core:
    Input:  krope_output_cb = [1, 32] rope (second half)
    Reads:  1 BFP8 tile from DRAM (rope column 17)
    Writes: 1 BFP8 tile back
```

### CB Layout for KV Cache Update

| CB | Index (fused) | Format | Size | Content |
|----|---------------|--------|------|---------|
| `kv_cache_input_cb` | 34 | BFP8 $32 \times 32$ | 16 tiles x 1088 = 17,408 B | Existing cache row from DRAM |
| `kv_cache_intermed_cb` | 33 | BF16 $32 \times 32$ | 17 tiles x 2048 = 34,816 B | Untilized intermediate |
| `kv_cache_output_cb` | 32 | BFP8 $32 \times 32$ | 16 tiles x 1088 = 17,408 B | Write to DRAM |

**Total L1 for KV cache update per core:** ~69 KB.

**Per-step DRAM bandwidth for KV cache update:**

$$\text{Read:} \quad 16 \times 1088 = 17,408 \text{ bytes}$$
$$\text{Write:} \quad 16 \times 1088 = 17,408 \text{ bytes}$$
$$\text{Total:} \quad 34,816 \text{ bytes} \approx 34 \text{ KB per decode step}$$

---

## 5.2.9 Standalone vs Fused KV Cache Branch

The `KVCacheBranch` class can operate standalone or as part of the `pre_sdpa` mega-fused op. In the fused configuration, the CB indices shift:

| Component | Standalone CB Index | Fused (pre\_sdpa) CB Index |
|-----------|-------------------|--------------------------|
| DKV matmul input | 6 | (shares matmul\_input\_cb = 5) |
| DKV matmul weights | 8 | 23 |
| DKV matmul output | 7 | 24 |
| KV RMSNorm input | 9 | 25 |
| KV RMSNorm gamma | 10 | 26 |
| KV RMSNorm output | 11 | 27 |
| K-ROPE output | 12 | 28 |
| K-ROPE cos | 0 | 29 |
| K-ROPE sin | 1 | 30 |
| KV cache input | -- | 34 |
| KV cache intermed | -- | 33 |
| KV cache output | -- | 32 |

In the fused mode, the KV cache update is integrated directly: after RMSNorm and RoPE complete, cores (0,8) and (8,8)-(8,9) perform the DRAM read-modify-write cycle without requiring a separate kernel launch.

---

## 5.2.10 The `cur_pos_tensor` Protocol

The position tracking uses a carefully designed protocol to coordinate between the KV cache write and the FlashMLA read:

### Position Synchronization

1. **Host** writes `cur_pos` to `position_ids_tensor` before each decode step.

2. **KV cache branch** reads `cur_pos` to determine:
   - DRAM page offset for cos/sin lookup (RoPE)
   - Cache write position: `kv_cache[0, 0, cur_pos, :]`

3. **After cache write completes**, the 3 KV cache update cores (1 nope + 2 rope) each signal the `kv_cache_cur_pos_ready_semaphore` (semaphore ID 5 in pre\_sdpa, ID 0 in standalone).

4. **FlashMLA cores wait** on this semaphore before starting attention. The semaphore is initialized to `kv_cache_cur_pos_ready_value = 3`, and each of the 3 KV cache update cores increments it. When it reaches the expected value, the position is confirmed ready.

5. **The position tensor** is also height-sharded across FlashMLA cores for direct L1 access during the attention computation -- no DRAM round-trip during the hot attention loop.

### Full-Grid Multicast for Position Readiness

The KV cache update kernel uses a full-device-grid multicast to broadcast the `cur_pos` value once the cache write completes. The multicast destination spans the full compute grid: `(0,0)` to `(device_grid_size.x-1, device_grid_size.y-1)`, ensuring all FlashMLA cores can safely read the cache.

---

## 5.2.11 FlashMLAOptimalGridNOC0: DRAM Bank-to-S-Block Mapping

The KV cache's ND shard placement across DRAM banks is co-designed with FlashMLA's S-block layout for optimal NOC locality. The optimal DRAM bank order is `(1, 3, 2, 0, 5, 7, 6, 4)`, minimizing NOC hop count between each S-block's cores and its assigned DRAM bank. See [Section 5.3.4](03_flash_mla_decode.md) for the full S-block-to-DRAM-bank mapping table with core coordinates.

---

## 5.2.12 Memory Hierarchy Summary: KV-Path

| Stage | DRAM Read | DRAM Write | L1 Peak (per core) | L1 Dominant |
|-------|----------|------------|--------------------:|-------------|
| DKV Matmul | 0 (weights pre-loaded) | 0 | ~252 KB | Weight shard |
| DKV Gather | 0 | 0 | ~3 KB (receiver) | Input CB |
| KV RMSNorm | 0 | 0 | ~3 KB | 3 CBs of $16 \times 32$ |
| K-RoPE | ~256 B (cos/sin) | 0 | ~2.6 KB | Trans mat |
| KV Cache Update | ~17 KB | ~17 KB | ~69 KB | BFP8/BF16 intermediates |

**Total DRAM bandwidth per decode step (KV-path):** ~35 KB read + ~17 KB write = ~52 KB. This is dwarfed by the Flash MLA DRAM reads (Section 5.3), confirming that the KV-path write is not the bottleneck.

---

**Prev:** [MLA Query Projection and Weight Layout](01_mla_query_projection_and_weight_layout.md) | **Next:** [Flash MLA Decode](03_flash_mla_decode.md)
