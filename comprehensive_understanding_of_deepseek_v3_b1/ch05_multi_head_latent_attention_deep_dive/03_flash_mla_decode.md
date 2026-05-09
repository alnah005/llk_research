# 5.3 Flash MLA Decode

This section is the heart of Chapter 5. It traces the Flash MLA decode kernel end-to-end: the chunked attention computation with online softmax, the 8-S-block parallelism scheme across 64 cores, the page-level double-buffered DRAM streaming pipeline, and the 3-step tree reduction that combines partial results across S-blocks. Every operation is mapped to the three RISC processors (NCRISC, BRISC, TRISC) with pseudocode for the critical compute path. For multi-device configurations, the section concludes with `SDPAReduceToAll`, which merges partial attention statistics across the device mesh using a ring-based fabric exchange.

Source: `micro_ops/flash_mla/op.py`, `unified_kernels/flash_mla.hpp`, `micro_ops/flash_mla/kernels/rt_args_common.hpp`, `micro_ops/sdpa_reduce_to_all/op.py`

---

## 5.3.1 Algorithm Overview: Flash Attention for MLA

The Flash MLA kernel computes scaled dot-product attention between a query tensor and a KV cache, where K and V share the same underlying tensor (the MLA latent):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

In the MLA setting:
- $Q \in \mathbb{R}^{[8, 576]}$ -- 8 query heads per core, each 576 elements (512 NOPE + 64 ROPE)
- KV cache $\in \mathbb{R}^{[1, 1, S, 576]}$ -- shared K/V per position
- $K$ is the full 576 elements (used for the dot product $QK^T$)
- $V$ is the first `head_dim_v = 512` elements (the NOPE portion, producing the 512-element attention output)

This means $V \subset K$ -- V is a prefix slice of K. FlashMLA exploits this by reading K tiles once from DRAM and using the same data for both the score and the value computations, eliminating a separate V read entirely.

### Inputs and Outputs

| Tensor | Shape | Format | Location |
|--------|-------|--------|----------|
| Q | $[1, 1, 64, 576]$ | BF16, $8 \times 32$ tiles | Height-sharded on 8 SDPA cores |
| KV Cache | $[1, 1, S_{max}, 576]$ | BFP8, $32 \times 32$ tiles | ND-sharded in DRAM (8 banks) |
| `cur_pos_tensor` | $[8]$ (int32) | Row-major | Height-sharded in L1, replicated per core |
| Output | $[1, 1, 64, 512]$ | BF16, $8 \times 32$ tiles | Height-sharded on 8 SDPA cores |

Note the output is $[1, 1, 64, 512]$, not $[1, 1, 64, 576]$ -- the ROPE dimensions are only needed for attention scores, not the output.

### Key Dimensional Parameters

| Symbol | Value | Meaning |
|--------|-------|---------|
| `total_q_heads` | 64 | Q heads per device |
| `num_q_heads_per_core` | 8 | Q heads per SDPA core (shard height) |
| `B` | 8 | Number of Q shards = output cores |
| `DH` | 576 | KV dimension (kvpe_dim) |
| `DHt` | 18 | $576 / 32$ tiles |
| `vDHt` | 16 | $512 / 32$ V dimension in tiles |
| `S` | $S_{max}$ | Maximum sequence length |
| `k_chunk_size` | 128 | Rows per KV cache chunk |
| `Sk_chunk_t` | 4 | $128 / 32$ tiles per chunk |
| `PNHt` | 1 | $8 / 8$ Q tile height ($8 \times 32$ tiny tiles) |
| `Q_TILE_HEIGHT` | 8 | Tiny tile height for Q |
| `K_TILE_HEIGHT` | 32 | Standard tile height for K/V |
| Scale | $1/\sqrt{576}$ | Attention scale factor |

---

## 5.3.2 Online Softmax with Max Tracking

The standard softmax computation $\text{softmax}(x_i) = e^{x_i} / \sum_j e^{x_j}$ requires two passes: one to find the normalizing constant and one to compute the output. Flash Attention avoids this with **online softmax** using running max and sum statistics, processing the sequence in chunks without ever materializing the full $S \times S$ attention matrix.

The Flash MLA compute kernel maintains three running statistics across K-chunks:

- $m \in \mathbb{R}^{[8, 1]}$ -- running row-wise maximum of $QK^T$ scores
- $s \in \mathbb{R}^{[8, 1]}$ -- running sum of $\exp(QK^T_{ij} - m_i)$ across positions
- $O \in \mathbb{R}^{[8, 512]}$ -- running weighted output accumulator

For each new K-chunk, the algorithm:

1. **Compute local scores:** $\text{scores} = Q \cdot K_{\text{chunk}}^T$ (raw dot products, no pre-scaling)
2. **Update max:** $m_{\text{new}} = \max(m_{\text{old}}, \max_j(\text{scores}_j))$
3. **Compute correction:** $P_{\text{old}} = \exp((m_{\text{old}} - m_{\text{new}}) \cdot \alpha)$ where $\alpha = 1/\sqrt{576}$
4. **Rescale running sums:** $s \leftarrow s \cdot P_{\text{old}}$, $O \leftarrow O \cdot P_{\text{old}}$
5. **Compute new exponentials:** $P_{\text{new}} = \exp((\text{scores} - m_{\text{new}}) \cdot \alpha)$
6. **Update sum:** $s \leftarrow s + \sum_j P_{\text{new},j}$
7. **Update output:** $O \leftarrow O + P_{\text{new}} \cdot V_{\text{chunk}}$

After all chunks: $\text{output} = O / s$ (element-wise division by the sum).

The max tracking ensures numerical stability by keeping the exponents from overflowing, and the correction factor $P_{\text{old}}$ rescales previously accumulated results to account for the updated maximum.

### Pseudocode: TRISC Compute Core

```
function compute_sdpa(Q, KV_cache, scale, k_chunk_start, k_chunk_end, stride):
    // Initialize statistics in DST registers
    m = -inf                    // [PNHt] tiles, max tracker
    s = 0                       // [PNHt] tiles, sum of exps
    O = 0                       // [PNHt x vDHt] tiles, output accumulator

    for k_chunk in range(k_chunk_start, k_chunk_end, stride):
        last = (k_chunk + stride >= k_chunk_end)

        // Wait for K-chunk data from NCRISC/BRISC pipeline
        cb_wait_front(cb_k_in, Sk_chunk_t * DHt)

        // Step 1: QK^T matmul - scores = Q @ K_chunk^T
        //   Q:      [PNHt, DHt]  tiny tiles (8x32)
        //   K_chunk: [Sk_chunk_t, DHt]  standard tiles (32x32), transposed
        //   scores:  [PNHt, Sk_chunk_t]  in DST registers
        sdpa_custom_mm_block(Q, K_chunk, transpose_k=true)

        // Step 2: Apply mask to last chunk if partial
        if last and mask_needed:
            apply_mask(scores, mask_cb)  // -inf for positions > cur_pos

        // Step 3: Online softmax update
        //   m_new = max(m, row_max(scores * scale))
        //   P_old = exp((m - m_new) * scale)
        //   O = O * P_old
        //   s = s * P_old
        //   P_new = exp((scores - m_new) * scale)
        //   s += row_sum(P_new)
        sdpa_update_max_and_rescale(scores, m, s, O, scale)

        // Step 4: V projection - O += P_new @ V_chunk
        //   P_new:   [PNHt, Sk_chunk_t]  in DST
        //   V_chunk:  [Sk_chunk_t, vDHt]  read from K buffer (first vDHt cols)
        //   Note: V is NOT transposed, read strided from K buffer
        sdpa_custom_mm_block(P_new, V_chunk, transpose_v=false)

        cb_pop_front(cb_k_in, Sk_chunk_t * DHt)

    // Step 5: Final normalization O = O / s
    if is_final_output:
        compute_sdpa_recip(O, s)
    else:
        // Pack m/s stats for tree reduction
        pack_tile(m_and_s, sdpa_ms_cb)

    // Pack output tiles
    for i in range(vDHt):
        pack_tile(O[i], sdpa_output_cb)
```

---

## 5.3.3 DST Register Layout

The TRISC compute kernel uses the DST accumulator register file extensively. The layout, computed from compile-time constants:

| Register Range | Symbol | Size | Contents |
|---------------|--------|------|----------|
| $[0, 16 \cdot v_{\text{DHt}})$ | `mm2_dst_offset` | $16 \times v_{\text{DHt}}$ tile slots | V accumulator $O$ ($\text{PNHt} \times v_{\text{DHt}}$ tiles) |
| $[16 \cdot v_{\text{DHt}}, 16 \cdot v_{\text{DHt}} + 2)$ | `max_dst_offset` | 2 tile slots | Running max $m$ (packed with sum $s$) |
| $[16 \cdot v_{\text{DHt}} + 2, 16 \cdot v_{\text{DHt}} + 16)$ | `sum_dst_offset` | 14 tile slots | Running sum $s$ |
| $[16 \cdot (v_{\text{DHt}} + 1), 16 \cdot (v_{\text{DHt}} + 2))$ | `corr_exp_dst_offset` | 16 tile slots | Correction exponent $P_{\text{old}}$ |
| $[16 \cdot (v_{\text{DHt}} + 2), ...)$ | `mm1_dst_offset` | $16 \times \text{Sk\_chunk\_t}$ tile slots | QK^T scores (overwritten each chunk) |

With $v_{\text{DHt}} = 16$ (512 elements / 32 per tile) and $\text{PNHt} = 1$ (8 heads packed into one $8 \times 32$ tile):
- $O$ occupies 16 tile slots
- $m, s$ share 1 tile slot
- $P_{\text{old}}$ occupies 1 tile slot
- Scores occupy 4 tile slots ($\text{Sk\_chunk\_t} = 4$ for 128/32)
- Total: 22 tile slots

The `compute_sdpa_chunk` function from `flash_mla.hpp` is a templated call that orchestrates this entire computation within the DST register file, avoiding L1 round-trips for intermediate results.

### Compute Configuration

FlashMLA uses specific compute settings for numerical stability:

- **`fp32_dest_acc_en`:** Enables FP32 accumulation in the destination register, critical for the online softmax where exponential values can span many orders of magnitude.
- **`math_approx_mode`:** Used for the `exp()` approximation in softmax.
- **`dst_size`:** Determined by FP32 mode and full-sync settings: 8 tiles for FP32 with full sync, 4 tiles for FP32 without, 16/8 for BF16.

The assertion `dst_size >= 8` ensures sufficient destination register space for the tiled matmul and reduction operations.

---

## 5.3.4 The S-Block Architecture: 8 Blocks, 8 Cores Each

The Flash MLA kernel distributes work across 8 **S-blocks**, each containing 8 compute cores. "S" stands for sequence-length parallelism: the $S$ positions in the KV cache are divided among S-blocks so each processes an independent subset of K-chunks.

### Physical Layout on the P150 Grid

The actual core assignments from `FlashMLAOptimalGridNOC0.BLOCKS`:

| S-Block | Cores (logical x,y) | DRAM Bank |
|---------|---------------------|-----------|
| S1 | (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2) | 1 |
| S2 | (0,3), (1,3), (2,3), (3,3), (0,4), (1,4), (2,4), (3,4) | 3 |
| S3 | (0,7), (1,7), (2,7), (3,7), (0,8), (1,8), (2,8), (3,8) | 2 |
| S4 | (0,9), (1,9), (2,9), (3,9), (0,0), (1,0), (2,0), (3,0) | 0 |
| S5 | (7,1), (8,1), (9,1), (10,1), (7,2), (8,2), (9,2), (10,2) | 5 |
| S6 | (7,4), (8,4), (9,4), (10,4), (7,5), (8,5), (9,5), (10,5) | 7 |
| S7 | (7,6), (8,6), (9,6), (10,6), (7,7), (8,7), (9,7), (10,7) | 6 |
| S8 | (7,9), (8,9), (9,9), (10,9), (7,0), (8,0), (9,0), (10,0) | 4 |

Each S-block is a $4 \times 2$ sub-grid spanning 4 columns and 2 rows. Key design properties:

1. **Left-right split:** S1-S4 occupy columns 0-3, S5-S8 occupy columns 7-10. The central columns (4-6) are left free for other operations in fused contexts.

2. **DRAM proximity:** Each S-block is placed near its assigned DRAM bank controller, minimizing NOC hop count for the dominant memory traffic pattern (streaming KV cache reads). The optimal bank order is `(1, 3, 2, 0, 5, 7, 6, 4)`.

3. **Wraparound S-blocks:** S4 and S8 have cores that span across the grid boundary (e.g., S4 includes columns 9 and 0). The NOC torus architecture supports wraparound multicast, making this topologically efficient.

### Work Distribution

Within each S-block, the 8 cores are assigned to the 8 Q-shards (batch elements). Core $b$ in S-block $s$ processes batch element $b$'s K-chunks from the portion assigned to S-block $s$:

$$\text{all\_cores}[b \times 8 + s] = \text{S-block}[s].\text{core}[b]$$

S1 is the "output" S-block: its cores hold the Q tensor shards and receive the final reduced output. Cores in S2-S8 are "worker" cores that assist with sequence parallelism.

### Sequence-Length Parallelism Example

For a sequence of length $S$ tokens, the KV cache contains $\lceil S / 128 \rceil$ chunks distributed round-robin across the 8 S-blocks:

```
  Work distribution for Q shard 0 (batch_idx = 0):

  S-block:  S1       S2       S3       S4       S5       S6       S7       S8
  Core:    (0,1)   (0,3)   (0,7)   (0,9)   (7,1)   (7,4)   (7,6)   (7,9)
  Chunks:  0,8,16  1,9,17  2,10,18 3,11,19 4,12,20 5,13,21 6,14,22 7,15,23
           ...     ...     ...     ...     ...     ...     ...     ...

  For seq_len = 3072 (24 chunks of 128):
    Each of the 8 cores processes 3 chunks
    Core 0 in S1: chunks 0, 8, 16
    Core 0 in S2: chunks 1, 9, 17
    etc.
```

---

## 5.3.5 K-Chunk Streaming: Double-Buffered DRAM Pipeline

The KV cache resides in DRAM. Reading it fast enough to keep the TRISC compute units busy is the primary performance bottleneck. Flash MLA uses a **double-buffered** K-chunk streaming pipeline with page-level overlap between DRAM reads and NOC multicasts.

### Buffer Setup

The K input circular buffer (`cb_k_in`) is sized for **two chunks**:

$$k\_\text{tiles} = \text{Sk\_chunk\_t} \times \text{DHt} \times 2 = 4 \times 18 \times 2 = 144 \text{ tiles}$$

Each chunk is $4 \times 18 = 72$ full tiles ($32 \times 32$), covering $[128, 576]$ elements. The double buffer allows one chunk to be read from DRAM while the previous chunk is consumed by compute.

### CB Layout for Double-Buffered K

```
cb_k_in (CB 1): 2 * Sk_chunk_t * DHt tiles = 2 * 72 = 144 BFP8 tiles

  +------------------+------------------+
  | Buffer A (72 t)  | Buffer B (72 t)  |
  |   76.5 KB BFP8   |   76.5 KB BFP8   |
  +------------------+------------------+

  Total: 144 * 1088 = 156,672 bytes ~ 153 KB
```

### Why BFP8 is Essential for the K Buffer

If the K buffer used BF16 instead of BFP8, each $32 \times 32$ tile would be 2,048 bytes instead of 1,088:

$$\text{Double-buffered K (BF16)} = 144 \times 2048 = 294,912 \text{ bytes} \approx 288 \text{ KB}$$

This would consume $288$ KB per core -- nearly $1.4\times$ the entire current CB budget. The BFP8 format saves $\sim 135$ KB of L1 per core on the K buffer alone, making double-buffering feasible within the L1 budget.

### Page-Level Pipelining

Rather than waiting for an entire K-chunk to arrive before multicasting, the NCRISC reader uses **page-level pipelining** with transaction IDs (TRIDs):

```
NCRISC (reader/sender):                    BRISC (multicast sender):
  for each K-chunk:                          for each K-chunk:
    reserve cb_k_in                            wait ncrisc_sync >= 1
    for page in 0..k_num_pages:                read k_write_ptr from shared mem
      noc_async_read with TRID                 for page in 0..k_num_pages:
      if prior TRID complete:                    wait ncrisc_sync >= page+1
        increment ncrisc_sync                    multicast page to S-block receivers
    swap sync pointers (double buffer)         signal receivers via semaphore
                                               swap sync pointers
```

The synchronization uses a shared memory location (`ncrisc_brisc_sync_semaphore_addr`) with an integer counter. NCRISC increments the counter as each page completes its DRAM read, and BRISC waits on the counter before multicasting that page. This allows the multicast to begin as soon as the first page arrives, overlapping DRAM latency with NOC transfer latency.

### Concrete Numbers

With `k_page_size = 16384` bytes (the NOC maximum) and BFP8 tiles:

$$k\_\text{chunk\_tiles} = 4 \times 18 = 72 \text{ tiles}$$
$$k\_\text{tile\_size} = 1088 \text{ bytes (BFP8 } 32 \times 32 \text{)}$$
$$k\_\text{total\_bytes} = 72 \times 1088 = 78,336 \text{ bytes per chunk}$$
$$k\_\text{num\_pages} = \lceil 78,336 / 16,384 \rceil = 5 \text{ pages}$$

So 5 NOC reads are pipelined per K-chunk, with BRISC multicasting each page as it completes.

### Double-Buffer Timing

```
  Time  ──────────────────────────────────────────────────────>

  NCRISC: [Read chunk 0 ][Read chunk 1 ][Read chunk 2 ]...
           into buf A     into buf B     into buf A

  TRISC:           [Compute chunk 0][Compute chunk 1][Compute chunk 2]...
                    from buf A       from buf B       from buf A

  BRISC:  [Q mcast + Setup]                          [Tree reduce]
```

---

## 5.3.6 The Three RISC Roles in Detail

### NCRISC: K Reader + Multicast Helper

The NCRISC processor has two roles based on `is_mcast_sender`:

**Sender cores (first batch element per S-block):**
1. Set up TRID-based async read pipeline from ND-sharded DRAM
2. Use `noc_async_read_one_packet_with_state_with_trid` for pipelined reads
3. Signal page completion to BRISC via shared counter
4. Iterate through assigned K-chunks with stride `num_cores_per_head`

**Receiver cores (non-sender):**
1. Signal readiness to sender via `receiver_ready_semaphore`
2. Wait on `mcast_semaphore` for each chunk
3. K data appears in `cb_k_in` via multicast (zero-copy into the same CB address)

**Wait for KV cache readiness:** Before entering the chunk loop, NCRISC waits on `kv_cache_cur_pos_ready_semaphore` to ensure the current token's KV data has been written. The initial value is 3, corresponding to the 3 KV cache update cores that must signal completion.

### BRISC: Q Distribution + K Multicast + Tree Reduction

BRISC has the most complex role, executing three phases:

**Phase 1 -- Q Distribution:**
- S1 (output) cores: already have Q in sharded `cb_q_in`; if sender, multicast Q input semaphore to all other cores
- Non-output cores: read Q from output core via unicast after semaphore signal
- Q size: $\text{PNHt} \times \text{DHt} = 1 \times 18 = 18$ tiles of $8 \times 32$ = 9,216 bytes per Q shard

**Phase 2 -- K Multicast:**
- Sender cores: wait for each page from NCRISC, then multicast to all 7 other cores in the S-block using `noc_async_write_multicast`
- Uses NOC0 for multicast, with the torus wraparound ensuring S-blocks like S4 (wrapping around column 0) work correctly
- The `physical_multicast_coords` method converts logical S-block coordinates to physical NOC coordinates for the multicast range

**Phase 3 -- Causal Mask Construction:**
- If the last K-chunk is partially valid, BRISC constructs a mask in L1: 0x00000000 for valid positions, 0xFF80FF80 (negative infinity in BF16) for invalid positions
- The mask is written directly to `cb_mask`

**Phase 4 -- Tree Reduction:**
- See Section 5.3.7

### TRISC: Attention Compute + Reduction

TRISC executes the core attention computation described in Section 5.3.2. Key implementation details:

- **DST register management:** Uses `tile_regs_acquire/commit/release` to manage the accumulator
- **Custom matmul block:** `sdpa_custom_mm_block_init_short<transpose_k>` initializes the hardware for the specific tile dimensions
- **Pack synchronization:** FPU-SFPU semaphores coordinate between the matmul and packing stages
- **Output routing:** Based on whether this core does tree reduction, the output goes to `cb_out_final` (direct output), `cb_interm_out` (intermediate for reduction), or `cb_out_o` (sender output for tree reduction)

---

## 5.3.7 Three-Step Tree Reduction

After each S-block's TRISC computes its partial attention result, the 8 partial results must be combined. Naive sequential reduction would take 7 steps. Flash MLA uses a **binary tree reduction** that completes in $\log_2(8) = 3$ steps.

### Reduction Tree Structure

From `FlashMLAOptimalGridNOC0.TREE_REDUCTION_ORDER`:

```python
TREE_REDUCTION_ORDER = (
    ((0, 1), (2, 3), (4, 5), (6, 7)),   # Step 1
    ((0, 2), (4, 6)),                     # Step 2
    ((0, 4),),                            # Step 3
)
```

Each pair $(d, s)$ means S-block $s$ sends its partial result to S-block $d$, which reduces them.

### Tree Reduction Data Flow Diagram

```
  Step 1 (4 parallel pairs):

  S2 core ─────NOC─────> S1 core     S4 core ─────NOC─────> S3 core
  (O2,m2,s2)             (O1,m1,s1)  (O4,m4,s4)             (O3,m3,s3)
                         reduce                               reduce
                         (O12,m12,s12)                        (O34,m34,s34)

  S6 core ─────NOC─────> S5 core     S8 core ─────NOC─────> S7 core
  (O6,m6,s6)             (O5,m5,s5)  (O8,m8,s8)             (O7,m7,s7)
                         reduce                               reduce
                         (O56,m56,s56)                        (O78,m78,s78)

  Step 2 (2 parallel pairs):

  S3 core ─────NOC─────> S1 core     S7 core ─────NOC─────> S5 core
  (O34,m34,s34)          (O12,m12,s12) (O78,m78,s78)        (O56,m56,s56)
                         reduce                               reduce
                         (O1234,...)                          (O5678,...)

  Step 3 (final):

  S5 core ─────NOC─────> S1 core
  (O5678,...)            (O1234,...)
                         reduce
                         (O_final, m_final, s_final)
                              |
                              v
                         O_output = O_final / s_final
                         Written to cb_out_final (output tensor)
```

### Role Assignment Per S-Block

| S-Block | Step 1 | Step 2 | Step 3 |
|---------|--------|--------|--------|
| S1 (idx 0) | receive from S2 | receive from S3 | receive from S5 |
| S2 (idx 1) | send to S1 | idle | idle |
| S3 (idx 2) | receive from S4 | send to S1 | idle |
| S4 (idx 3) | send to S3 | idle | idle |
| S5 (idx 4) | receive from S6 | receive from S7 | send to S1 |
| S6 (idx 5) | send to S5 | idle | idle |
| S7 (idx 6) | receive from S8 | send to S5 | idle |
| S8 (idx 7) | send to S7 | idle | idle |

S1 is the ultimate accumulator -- it receives in all 3 steps and never sends. After step 3, S1's output core holds the final attention result.

### Reduction Math: sdpa_tail

Each tree reduction step combines two partial results using the SDPA tail formula:

$$m = \max(m_1, m_2)$$
$$P_1 = \exp((m_1 - m) \cdot \alpha), \quad P_2 = \exp((m_2 - m) \cdot \alpha)$$
$$s = s_1 \cdot P_1 + s_2 \cdot P_2$$
$$O = O_1 \cdot P_1 + O_2 \cdot P_2$$

This is mathematically identical to the online softmax update but operates on pre-accumulated statistics rather than raw scores. The correction exponents $P_1, P_2$ rescale both outputs to a common max before summation. After the final reduction step on S1: $O_{\text{final}} = O / s$.

### Pseudocode: Tree Reduction (BRISC + TRISC)

```
// BRISC: data movement for tree reduction
for step in 0..num_tree_reduction_steps:
    role = tree_reduction_info[step].role_code
    partner = tree_reduction_info[step].partner

    if partner >= num_active_s_blocks:
        continue   // Partner has no work; skip

    if role == SENDER:
        wait cb_out_o and cb_out_ms ready
        write m/s stats to partner at cb_ms_in[step]
        write O tiles to partner at cb_out_in[step]
        increment partner's reducer_semaphore (bit-packed by step)
        break      // Sender is done after its step

    if role == RECEIVER:
        reserve cb_ms_in and cb_out_in
        spin-wait on reducer_semaphore for step's bit
        push back (data available for TRISC)

// TRISC: compute reduction
if num_cores_to_wait > 0:
    for i in 0..num_cores_to_wait-1:
        sdpa_tail(cb_ms_in, cb_interm_ms, ..., cb_out_in, cb_interm_out, ...)
    // Final reduction: output to cb_out_final with normalization
    sdpa_tail(cb_ms_in, cb_interm_ms, cb_out_ms, cb_out_in, cb_interm_out, cb_out_final)
```

### Semaphore Bit-Packing

The tree reduction uses a single semaphore word with **bit-packed step indicators**. Each step uses `bits_per_step = 1` bits:

$$\text{step\_semaphore\_inc}(\text{step}) = 1 \ll (\text{step} \times \text{bits\_per\_step})$$

This allows the receiver to check each step's completion independently without needing separate semaphores:

```c++
uint8_t step_sem = (sem_val >> (step * bits_per_step)) & step_mask;
if (step_sem >= 1) break;  // This step's data has arrived
```

### Multi-Step Buffer Isolation

A subtle correctness issue: tree reduction steps can complete out of order (e.g., S5's step-3 data might arrive before S3's step-2 data). To prevent data corruption, each step writes to a **separate buffer slot**:

$$\text{cb\_ms\_in offset for step } s = \text{base\_addr} + s \times \text{ms\_write\_size}$$
$$\text{cb\_out\_in offset for step } s = \text{base\_addr} + s \times \text{o\_write\_size}$$

The `cb_out_in` CB is sized for `out0_t * NUM_TREE_REDUCTION_STEPS` tiles and `cb_ms_in` for `PNHt * NUM_TREE_REDUCTION_STEPS` tiles, providing 3 independent buffer slots.

### Per-Step Data Transfers

| Step | Pairs | Transfer Size per Pair |
|------|-------|----------------------|
| 1 | (S2$\to$S1), (S4$\to$S3), (S6$\to$S5), (S8$\to$S7) | $(16 + 1) \times 512 = 8,704$ bytes |
| 2 | (S3$\to$S1), (S7$\to$S5) | Same |
| 3 | (S5$\to$S1) | Same |

With $\text{out0\_t} = \text{PNHt} \times \text{vDHt} = 1 \times 16 = 16$ tiles for the output and $\text{PNHt} = 1$ tile for stats, each transfer is 17 tiles at 512 bytes = 8,704 bytes.

---

## 5.3.8 Handling Variable Sequence Lengths

The K-chunk distribution adapts to the actual sequence length:

### Chunk Allocation (from `rt_args_common.hpp`)

```c++
uint32_t valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);  // Round up
uint32_t num_chunks = valid_seq_len / k_chunk_size;

if (num_cores_per_head > num_chunks) {
    // More cores than chunks: each active core gets 1 chunk
    k_chunk_start = core_num;
    k_chunk_end = k_chunk_start + (core_num < num_chunks ? 1 : 0);
} else {
    // Strided distribution
    chunks_per_core = num_chunks / num_cores_per_head;
    residuals = num_chunks % num_cores_per_head;
    num_for_core = chunks_per_core + (core_num < residuals ? 1 : 0);
    k_chunk_start = core_num;
    k_chunk_end = k_chunk_start + (num_for_core - 1) * num_cores_per_head + 1;
    // Kernel iterates: for (k = start; k < end; k += num_cores_per_head)
}
```

### Active S-Block Tracking

When `num_chunks < num_cores_per_head` (short sequences), some S-blocks have no work. The tree reduction handles this gracefully:

```c++
uint32_t num_active_s_blocks = min(k_num_chunks, num_cores_per_head);
// In tree reduction: skip if partner >= num_active_s_blocks
if (role_code != 0 && partner_s_block_idx >= num_active_s_blocks) continue;
```

For example, with `cur_pos = 200` ($N_{\text{chunks}} = 2$), only S-blocks 0 and 1 are active. The tree reduction degenerates to a single step: S2 sends to S1.

### Causal Mask for Partial Chunks

The last K-chunk may be partially valid. BRISC constructs a mask inline:

```c++
uint32_t num_unmasked = cur_pos % k_chunk_size + 1;
// Fill mask: 0x00000000 for valid, 0xFF80FF80 for invalid (-inf in BF16)
for (i = 0; i < num_unmasked / 2; i++)
    *mask_ptr++ = 0x00000000;
if (num_unmasked % 2 == 1)
    *mask_ptr++ = 0xFF800000;  // One valid, one masked
for (; i < k_chunk_size / 2; i++)
    *mask_ptr++ = 0xFF80FF80;  // Both masked
```

The BF16 value 0xFF80 is $-\infty$, ensuring masked positions contribute zero after the $\exp$ in the softmax.

---

## 5.3.9 Virtual Channel Assignment

Each S-block is assigned a virtual channel (VC) for NOC multicast to avoid contention:

```python
if s_block_idx < 4:
    vc = s_block_idx & 0x1        # S1:0, S2:1, S3:0, S4:1
else:
    vc = 2 + ((s_block_idx - 4) & 0x1)  # S5:2, S6:3, S7:2, S8:3
```

This assigns 4 VCs across 8 S-blocks (2 per VC), ensuring adjacent S-blocks use different VCs to minimize NOC contention. The left-side blocks (S1-S4) use VCs 0-1, and the right-side blocks (S5-S8) use VCs 2-3.

---

## 5.3.10 CB Budget Summary

| CB | Index | Tile Type | Size (tiles) | Per-Core Size | Role |
|----|-------|-----------|--------------|---------------|------|
| `cb_q_in` | 0 | $8 \times 32$ (tiny) | $\text{PNHt} \times \text{DHt} = 18$ | 9,216 B | Q input (sharded from CreateQHeads) |
| `cb_k_in` | 1 | $32 \times 32$ BFP8 | $\text{Sk\_chunk\_t} \times \text{DHt} \times 2 = 144$ | **156,672 B** | Double-buffered K chunks |
| `cb_mask` | 2 | $8 \times 32$ (tiny) | 1 | 512 B | Causal mask for partial chunks |
| `cb_ms_in` | 3 | $8 \times 32$ (tiny) | $\text{PNHt} \times \text{NUM\_STEPS} = 3$ | 1,536 B | Received m/s stats (tree reduction) |
| `cb_out_in` | 4 | $8 \times 32$ (tiny) | $v_{\text{DHt}} \times \text{NUM\_STEPS} = 48$ | 24,576 B | Received O data (tree reduction) |
| `cb_out_o` | 5 | $8 \times 32$ (tiny) | $\text{PNHt} \times v_{\text{DHt}} = 16$ | 8,192 B | Compute output O |
| `cb_out_ms` | 6 | $8 \times 32$ (tiny) | $\text{PNHt} = 1$ | 512 B | Compute output m/s stats |
| `cb_interm_out` | 7 | $8 \times 32$ (tiny) | $v_{\text{DHt}} = 16$ | 8,192 B | Intermediate O (aliased with cb_out_o) |
| `cb_interm_ms` | 8 | $8 \times 32$ (tiny) | $\text{PNHt} = 1$ | 512 B | Intermediate m/s (aliased with cb_out_ms) |
| `cb_out_final` | 9 | $8 \times 32$ (tiny) | per output spec | 8,192 B | Final sharded output |

Note: `cb_out_o`/`cb_interm_out` and `cb_out_ms`/`cb_interm_ms` share the same physical memory through CB aliasing (multiple format descriptors on the same buffer).

### L1 Budget Analysis

Summing all CBs per Flash MLA core:

| Category | Size |
|----------|-----:|
| Q shard (`cb_q_in`) | 9,216 B |
| Double-buffered K (`cb_k_in`) | **156,672 B** |
| Mask (`cb_mask`) | 512 B |
| Tree reduction input buffers | 26,112 B |
| Output/stats accumulators | 8,704 B |
| Final output (`cb_out_final`) | 8,192 B |
| **Total** | **209,408 B $\approx$ 204.5 KB** |

The double-buffered K CB dominates at $\sim 153$ KB, consuming $75\%$ of the total CB budget. This is the fundamental trade-off: larger K buffers enable better DRAM-compute overlap, but they consume scarce L1 space.

---

## 5.3.11 Semaphore Budget

| ID | Name | Purpose |
|----|------|---------|
| 0 | `reducer_semaphore` | Tree reduction arrival signaling (bit-packed) |
| 1 | `mcast_semaphore` | K-chunk multicast completion (sender to receivers) |
| 2 | `ncrisc_brisc_sync_semaphore` | Page-level DRAM/multicast overlap (+ 4 bytes for double-buffer state) |
| 3 | `receiver_ready_semaphore` | Double-buffer flow control (receivers signal readiness) |
| 4 | `q_input_mcast_semaphore` | Q input distribution (output core to workers) |
| 5 | `kv_cache_cur_pos_ready_semaphore` | KV cache write completion (init = 3, from KV cache update) |

In the fused `pre_sdpa` context, these are allocated as global semaphores at indices 7-12, after the 7 semaphores used by earlier stages.

---

## 5.3.12 DRAM Bandwidth Profile

The Flash MLA stage dominates the per-step DRAM bandwidth consumption. For a sequence length of $S$ positions:

**Per core:**

$$\text{K chunks read} = \frac{S / 8}{128} \text{ chunks}$$
$$\text{Bytes per chunk} = 72 \times 1088 = 78,336 \text{ B}$$
$$\text{Total per core} = \frac{S}{1024} \times 78,336$$

**Per Q shard (8 cores):**

$$\text{Total DRAM read per Q shard} = \frac{S}{128} \times 78,336 = S \times 612$$

**For $S = 4096$:**

$$\text{Total DRAM read per Q shard} = 4096 \times 612 = 2,506,752 \text{ B} \approx 2.4 \text{ MB}$$
$$\text{Total across 8 Q shards} = 8 \times 2.4 \text{ MB} = 19.2 \text{ MB}$$

**For $S = 8192$:**

$$\text{Total DRAM read per Q shard} = 8192 \times 612 = 5,013,504 \text{ B} \approx 4.8 \text{ MB}$$
$$\text{Total across 8 Q shards} = 8 \times 4.8 \text{ MB} = 38.4 \text{ MB}$$

However, thanks to the S-block locality optimization, each S-block reads from its nearest DRAM bank. With 8 DRAM banks and 8 S-blocks, the read bandwidth is evenly distributed across all banks, avoiding hot-spot contention.

**BFP8 savings:** In BF16, these numbers would be $1.88\times$ larger -- $\sim 72$ MB for $S = 8192$ instead of $\sim 38$ MB. The DRAM bandwidth savings from BFP8 directly translate to lower decode latency.

### Performance Characteristics

The Flash MLA decode kernel is compute-bound for long sequences and memory-bound for short sequences:

- **Long sequences ($S > 1024$):** Each core processes multiple K-chunks. DRAM reads are pipelined and hidden behind compute. The bottleneck is the $QK^T$ matmul at $8 \times 576 \times 128 \times 2 = 1.18M$ FLOPs per chunk.
- **Short sequences ($S \le 128$):** Only 1 chunk, distributed across 8 cores, with most cores idle. Tree reduction is minimal. DRAM read latency dominates.
- **Critical path for $S = 4096$:** 32 chunks, 4 per core, 3-step tree reduction. Total compute: $4 \times (QK^T + P \times V) = 4 \times (1.18M + 0.52M) = 6.8M$ FLOPs per core, plus reduction overhead.

---

## 5.3.13 SDPAReduceToAll: Cross-Device Attention Merging

With TP=2, each device computes attention for its local 64 heads over its portion of the KV cache. When the KV cache is sharded across devices (via `sdpa_per_device_chunk_size`), different devices own different portions of the sequence history. `SDPAReduceToAll` combines the partial attention statistics across the mesh so that each device holds the globally correct result.

### The Reduce-to-All Algorithm

Given $N$ devices, each holding local statistics $(\ell_d, s_d, m_d)$ -- where $\ell$ is the weighted output, $s$ is the sum of exponentials, and $m$ is the max logit -- the algorithm uses a ring-based exchange with 2 reduction rounds:

**Round 1:** Each device exchanges with its forward/backward neighbor:
- Type A workers (even `(row + worker_idx)`) send forward, receive backward.
- Type B workers (odd `(row + worker_idx)`) send backward, receive forward.

After Round 1, each device holds the reduction of 2 devices' statistics.

**Round 2:** The Round 1 results are exchanged in the opposite direction, producing the global reduction on every device.

```
  Round 1: Each device exchanges with its immediate neighbor
    Device 0 <--> Device 1    (bidirectional)
    Device 2 <--> Device 3    (bidirectional)

  Round 2: Merged pairs exchange
    Device 0 <--> Device 2    (via merged results)
    Device 1 <--> Device 3    (via merged results)

  Result: All 4 devices have the globally reduced (O, m, s)
```

The merge at each step uses the same online softmax formula as the tree reduction:

$$m_{new} = \max(m_1, m_2)$$
$$s_{new} = s_1 \cdot e^{(m_1 - m_{new}) \cdot \text{scale}} + s_2 \cdot e^{(m_2 - m_{new}) \cdot \text{scale}}$$
$$\ell_{new} = \ell_1 \cdot e^{(m_1 - m_{new}) \cdot \text{scale}} + \ell_2 \cdot e^{(m_2 - m_{new}) \cdot \text{scale}}$$

### Position-Aware Validity Masking

When `position_enabled = True` and `per_device_chunk_size > 0`, the reduction must account for devices that hold no valid KV data for the current position:

$$\text{valid}_d = \begin{cases} 1 & \text{if } \text{pos} \geq d \times \text{per\_device\_chunk\_size} \\ 0 & \text{otherwise} \end{cases}$$

The reduction kernel evaluates this at runtime. Invalid devices' partial results are excluded from the merge -- when one operand is invalid, the valid operand passes through unchanged. When both are invalid, the result passes through as-is. This handles the case where early tokens have not yet filled all devices' KV cache shards.

### Worker/Forwarder Architecture

SDPAReduceToAll uses 8 worker cores and 2 forwarder cores per device:

- **Workers** (8 cores on the FlashMLA S1 output grid): Hold local $(\ell, m, s)$ data. Each processes its output tiles through the reduction pipeline, with a double-buffered pipeline for overlap.

- **Forwarders** (2 cores): Relay data between workers and the TT-Fabric CCL mesh. Each forwarder handles a subset of workers (Type A and Type B alternating), with separate buffer slots for Round 1 and Round 2.

### Chunked L Transfer

The $\ell$ tensor (weighted output) is large -- $\text{out\_tiles} \times \text{page\_size}$ bytes -- so it is split into `num_l_chunks` pieces for transfer:

```python
max_tiles_per_chunk = 8
min_num_l_chunks = ceil(out_tiles / max_tiles_per_chunk)
num_l_chunks = max(min_num_l_chunks, 4)
```

Each chunk is sent as a separate fabric packet. The MS statistics ($m$ and $s$, packed into a single tile) are sent as the first slot, followed by the L chunks. This ensures the receiver can begin the reduction merge as soon as the first L chunk arrives.

### Forwarder Buffer Layout

The forwarder scratch buffer is sized for:

$$\text{slots\_per\_worker} = 1 + \text{num\_l\_chunks}$$
$$\text{slots\_per\_round} = \text{workers\_per\_type} \times \text{slots\_per\_worker}$$
$$\text{slot\_size} = \text{packet\_header\_size} + \text{l\_chunk\_size\_bytes}$$
$$\text{total} = 2 \times \text{slots\_per\_round} \times \text{slot\_size}$$

The factor of 2 accounts for Round 1 and Round 2 buffers at separate offsets.

### Forwarder Semaphores

The forwarder cores use 4 local semaphores (IDs 8-11 in the fused post_sdpa context):

| Semaphore | Direction | Purpose |
|-----------|-----------|---------|
| `fwd_r1_sem` | Forward, Round 1 | Workers signal forwarder that R1 data is ready |
| `fwd_r2_sem` | Forward, Round 2 | Workers signal forwarder that R2 data is ready |
| `bwd_r1_sem` | Backward, Round 1 | Workers signal forwarder that R1 data is ready |
| `bwd_r2_sem` | Backward, Round 2 | Workers signal forwarder that R2 data is ready |

These are bit-packed semaphores supporting up to 32 slots per round (`slots_per_round <= 32`), where each bit indicates whether a specific worker's slot data has been consumed by the forwarder.

Two global semaphores coordinate the fabric endpoints:
- `r1_recv_sem`: Receiver increments after writing R1 data, sender waits before reusing the buffer.
- `r2_recv_sem`: Same for R2.

### Scatter to Matmul4 Input

When `SDPAReduceToAll` is fused into `post_sdpa`, the reduced output is scattered directly to the matmul4 input cores. The scatter extracts individual rows from the $8 \times 32$ SDPA output tiles and writes them as $1 \times 32$ tiles on the kv_b2 matmul grid:

$$\text{scatter\_num\_rows} = 8 \quad \text{(tile height, matching Q\_TILE\_HEIGHT)}$$
$$\text{scatter\_src\_tile\_size} = 8 \times 32 \times 2 = 512 \text{ bytes (BF16 } 8 \times 32 \text{)}$$
$$\text{scatter\_dst\_tile\_size} = 1 \times 32 \times 2 = 64 \text{ bytes (BF16 } 1 \times 32 \text{)}$$

Each row corresponds to one head's 512-element attention output, destined for a specific core in the 64-core kv_b2 grid where it will be projected through $W_{kv\_b2}$.

---

## 5.3.14 Memory Hierarchy Summary

| Component | DRAM Read/Step | L1 Footprint/Core | Bottleneck |
|-----------|---------------|-------------------:|------------|
| K double-buffer | $\sim 627$ KB per core ($S=4096$) | $153$ KB | **DRAM BW** |
| Q shard | 0 (mcast from L1) | $9$ KB | -- |
| Tree reduction buffers | 0 (L1-to-L1) | $26$ KB | NOC BW |
| Output accumulator | 0 | $8.7$ KB | -- |
| Final output | 0 ($8$ KB write) | $8$ KB | -- |
| **Total per core** | **$\sim 627$ KB** | **$\sim 205$ KB** | **DRAM BW** |

The Flash MLA stage is the single largest DRAM bandwidth consumer in the entire decode step, and the BFP8 KV cache format is the most impactful optimization for reducing this cost.

---

**Prev:** [Compressed KV Cache](02_compressed_kv_cache.md) | **Next:** [Post-SDPA Output Projection](04_post_sdpa_output_projection.md)
