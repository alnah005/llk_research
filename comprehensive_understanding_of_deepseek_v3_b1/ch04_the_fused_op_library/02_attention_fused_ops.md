# 4.2 Attention Fused Ops

This section covers the four fused operations that implement the attention path of DeepSeek V3 B1: from the initial layer-norm broadcast through the full Multi-head Latent Attention (MLA) decode. The crown jewel is `pre_sdpa`, a 2564-line fused op that chains 10+ stages across 130 cores in a single dispatch.

Source: `models/demos/deepseek_v3_b1/fused_ops/{broadcast_rms,pre_sdpa,kv_cache_branch,post_sdpa}/op.py`

---

## 4.2.1 `broadcast_rms` -- CCL Broadcast + RMSNorm

**Source:** `broadcast_rms/op.py` (423 lines)

The simplest attention fused op. It chains a cross-device CCL broadcast with an RMSNorm, all on a **single core** per device.

### Composition Tree

```
broadcast_rms
  +-- [NCRISC+BRISC] CCL Broadcast (optional, skip_ccl mode)
  +-- [TRISC] RMSNorm: x * rsqrt(var + eps) * gamma
```

### Tensor Shape Trace

```
Input:  [1, 7168] (hidden state from previous layer)
  --> CCL Broadcast: [1, 7168] replicated to all devices
  --> RMSNorm: [1, 7168] --> [1, 7168]
Output: [1, 7168] (normalized hidden state)
```

### CB Layout (4 CBs)

| CB | Index | Tile Format | Role |
|----|-------|-------------|------|
| `input_cb` | 0 | $32 \times 32$ or $16 \times 32$ | RMSNorm input (tensor-backed: `intermediate_tensor` in CCL mode, `input_tensor` in skip-CCL) |
| `pkt_cb` | 1 | Raw | CCL broadcast staging buffer |
| `gamma_cb` | 2 | $32 \times 32$ or $16 \times 32$ | RMSNorm gamma (tensor-backed) |
| `output_cb` | 3 | $32 \times 32$ or $16 \times 32$ | RMSNorm output (tensor-backed) |

### RISC Assignment

| RISC | CCL Mode (`skip_ccl=False`) | Local Mode (`skip_ccl=True`) | Socket Mode |
|------|---------------------------|------------------------------|-------------|
| NCRISC | CCL broadcast reader + RMSNorm reader | RMSNorm reader (gamma + input) | RMSNorm reader |
| BRISC | CCL broadcast writer | Idle | Socket receiver (writes to `input_cb`) |
| TRISC | RMSNorm compute | RMSNorm compute | RMSNorm compute |

### Stage Pipeline Table

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Sync |
|-------|----------|------|-------------|--------------|------|
| 1 | CCL Broadcast | NCRISC + BRISC | `pkt_cb` (1) | `input_cb` (0) | Global semaphores (out_ready, barrier, secondary_sync) |
| 2 | RMSNorm | TRISC | `input_cb` (0), `gamma_cb` (2) | `output_cb` (3) | CB handshake: NCRISC signals input ready |

### Key Implementation Details

- **Tile reinterpretation:** The input tensor uses $1 \times 32$ tiles, but the RMSNorm compute kernel requires $32 \times 32$ or $16 \times 32$ tiles. For $K = 7168$: $7168 / 32 = 224$, $224 \bmod 32 = 0$, so the $32 \times 32$ interpretation is used, yielding $7168 / 1024 = 7$ tiles.
- **CCL topology:** Supports linear and torus topologies. Forward/backward hop counts are computed from `sender_row` and `ring_size`.
- **Socket mode:** When `use_socket=True`, BRISC reads from a D2D socket into `input_cb`, replacing the CCL broadcast path.
- **Per-device programs:** Uses `MeshProgramDescriptor` with per-device CCL routing args (fabric node IDs, forward/backward hop counts, sender/receiver roles).

---

## 4.2.2 `pre_sdpa` -- The Attention Mega-Fused Op (Deep Dive)

**Source:** `pre_sdpa/op.py` (2564 lines)

This is the most complex fused op in the entire model. It compresses the complete pre-attention computation -- from CCL broadcast of the hidden state through to the final FlashMLA decode output -- into a single kernel dispatch across the full device grid.

### Mathematical Operation

$$\text{attn\_norm} = \text{RMSNorm}(x, \gamma_1)$$
$$q\_a = \text{attn\_norm} \cdot W_{q\_a\_proj}$$
$$q\_b = \text{RMSNorm}(q\_a, \gamma_2) \cdot W_{q\_b\_proj}$$
$$q_{nope}, q_{rope} = \text{split}(q\_b)$$
$$q_{nope} = q_{nope} \cdot W_{kv\_b1\_proj}$$
$$q_{rope} = \text{RoPE}(q_{rope})$$
$$\text{SDPA output} = \text{FlashMLA}(\text{concat}(q_{nope}, q_{rope}), \text{KV cache})$$

Simultaneously on the KV branch:

$$dkv = \text{attn\_norm} \cdot W_{dkv}$$
$$kv, k_{rope} = \text{split}(dkv, [512, 64])$$
$$kv = \text{RMSNorm}(kv, \gamma_{dkv})$$
$$k_{rope} = \text{RoPE}(k_{rope})$$
$$\text{KV cache}[\text{pos}] = \text{concat}(kv, k_{rope})$$

### Core Grid Diagram (P150, 13x10 logical grid)

```
     Col 0   1   2   3   4   5   6   7   8   9  10  11  12
Row 0  [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [  ]
Row 1  [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [  ]
Row 2  [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [  ]
Row 3  [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [  ]
Row 4  [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [  ]
Row 5  [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [  ]
Row 6  [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [  ]
Row 7  [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [Q ] [  ]
Row 8  [DK] [DK] [DK] [DK] [DK] [DK] [DK] [DK] [KR] [DK] [  ] [  ] [  ]
Row 9  [DK] [DK] [DK] [DK] [DK] [DK] [DK] [DK] [KR] [RN] [  ] [  ] [  ]

Legend:
  Q  = q_ab grid (96 cores: cols 0-11 x rows 0-7)
       Stage 1: Matmul1 q_a_proj (all 96)
       Stage 2: Matmul2 q_b_proj (all 96)
       Stage 3: splits into Qnope (cols 0-7, 64 cores: Matmul3 kv_b1_proj)
                        and Qrope (cols 8-11, 32 cores: QRoPE)
  DK = DKV matmul core (kv_a_proj, 9x2 = 18 cores)
  KR = K-RoPE core (2 cores at (8,8) and (8,9))
  RN = DKV RMSNorm core (1 core at (9,9))
  [RN/sender] = RMSNorm/Input core at (12,9) -- mcast sender, gather receiver
```

**Note:** The q_ab grid is highly overlapped. All 96 cores participate in Matmul1 (q_a_proj), then all 96 in Matmul2 (q_b_proj with different weights), then split: cols 0-7 (64 cores) perform Matmul3 (kv_b1_proj for Qnope), cols 8-11 (32 cores) perform QRoPE.

### SDPA Input Grid

After Q head computation, results are unicast to an 8-core SDPA input grid:

```
     Col 0   1   2   3
Row 1  [S0] [S1] [S2] [S3]    <- rows 0-3 map here
Row 2  [S4] [S5] [S6] [S7]    <- rows 4-7 map here

Mapping: source_row -> target_core
  row 0 -> (0,1), row 1 -> (1,1), row 2 -> (2,1), row 3 -> (3,1)
  row 4 -> (0,2), row 5 -> (1,2), row 6 -> (2,2), row 7 -> (3,2)
```

Each SDPA input core receives 8 interleaved heads: $8 \times (512 + 64) = 4608$ elements.

### Core Grid Assignments

| Grid | Dimensions | Cores | Role |
|------|-----------|-------|------|
| `rmsnorm_core` | $(12, 9)$ | 1 | RMSNorm1, RMSNorm2, GatherReduce receiver, Mcast1/2 sender |
| `matmul_weights_core_grid` | $12 \times 8$ | 96 | Matmul1 (q\_a\_proj), K-split into 2 halves of 48 |
| `matmul2_weights_core_grid` | $12 \times 8$ | 96 | Matmul2 (q\_b\_proj) |
| `qnope_grid` | cols 0--7 $\times$ rows 0--7 | 64 | Matmul3 (kv\_b1\_proj) + QNOPE unicast sender |
| `qrope_grid` | cols 8--11 $\times$ rows 0--7 | 32 | QRoPE + QROPE unicast sender |
| `sdpa_input_grid` | cols 0--3 $\times$ rows 1--2 | 8 | CreateQHeads receiver + tilizer |
| `dkv_matmul_weights_core_grid` | $9 \times 2$ | 18 | DKV matmul |
| `dkv_rmsnorm_grid` | 1 core | 1 | KV RMSNorm |
| `krope_grid` | $(8,8)$--$(8,9)$ | 2 | K-RoPE |
| `mla_core_grid` | 64 cores | 64 | FlashMLA decode |

### Complete CB Map (45 CBs, indices 0--44)

| CB Index | Name | Tile Format | Role |
|----------|------|-------------|------|
| 0 | `input_cb` | $32 \times 32$ | Input tensor / CCL destination |
| 1 | `gamma_cb` | $32 \times 32$ | RMSNorm1 gamma (overlapped tensor) |
| 2 | `rmsnorm_output_cb` | $32 \times 32$ | RMSNorm1 output |
| 3 | `matmul_weights_cb` | native | Matmul1 weights (overlapped tensor) |
| 4 | `matmul_output_cb` | $1 \times 32$ | Matmul1 per-core output (1 tile) |
| 5 | `matmul_input_cb` | $1 \times 32$ | Mcast destination / Matmul1 + DKV input |
| 6 | `rmsnorm2_gamma_cb` | $16 \times 32$ | RMSNorm2 gamma (overlapped tensor) |
| 7 | `rmsnorm2_input_cb` | $16 \times 32$ | GatherReduce destination (3 tiles of $16 \times 32$) |
| 8 | `gather_reduce_half1_scratch_cb` | $16 \times 32$ | Scratch for K-split gather-reduce |
| 9 | `rmsnorm2_output_cb` | $16 \times 32$ | RMSNorm2 output / Mcast2 source |
| 10 | `matmul2_input_cb` | $1 \times 32$ | Mcast2 destination / Matmul2 input (48 tiles) |
| 11 | `matmul2_weights_cb` | native | Matmul2 weights (overlapped tensor) |
| 12 | `matmul2_output_cb` | $1 \times 32$ | Matmul2 output (4 tiles/core) |
| 13 | `matmul3_weights_cb` | native | Matmul3 weights (overlapped, kv\_b12 buffer) |
| 14 | `matmul3_output_cb` | $1 \times 32$ | Matmul3 output (16 tiles/core = $[1, 512]$) |
| 15 | `qrope_output_cb` | $1 \times 32$ | QRoPE output (4 tiles/core) |
| 16 | `create_q_heads_out_cb` | $8 \times 32$ | CreateQHeads output = MLA Q input |
| 17--22 | `qrope_cos_cb` .. `qrope_sin_interm_cb` | $1 \times 32$ | QRoPE cos/sin/trans\_mat/intermediates |
| 23 | `dkv_matmul_weights_cb` | native | DKV matmul weights (overlapped tensor) |
| 24 | `dkv_matmul_output_cb` | $1 \times 32$ | DKV matmul output (1 tile/core) |
| 25 | `kv_rmsnorm_input_cb` | $16 \times 32$ | KV RMSNorm input (gathered) |
| 26 | `kv_rmsnorm_gamma_cb` | $16 \times 32$ | KV RMSNorm gamma (overlapped tensor) |
| 27 | `kv_rmsnorm_output_cb` | $16 \times 32$ | KV RMSNorm output |
| 28 | `krope_output_cb` | $1 \times 32$ | K-RoPE output |
| 29--30 | `krope_cos_cb`, `krope_sin_cb` | $1 \times 32$ | K-RoPE cos/sin (DRAM) |
| 31 | `create_q_heads_receiver_in_cb` | $8 \times 32$ | CreateQHeads intermediate (row-major staging) |
| 32 | `kv_cache_output_cb` | $32 \times 32$ BFP8 | KV cache write output |
| 33 | `kv_cache_intermed_cb` | $32 \times 32$ BF16 | KV cache format conversion intermediate |
| 34 | `kv_cache_input_cb` | $32 \times 32$ BFP8 | KV cache read input |
| 35--43 | `mla_k_in_cb` .. `mla_out_final_cb` | various | FlashMLA (K input, mask, intermediates, output) |
| 44 | `bcast_pkt_cb` | native | CCL broadcast packet buffer |

### Semaphore Budget (10 skip-CCL, 13 with CCL)

| Sem Index | Name | Used By |
|-----------|------|---------|
| 0--2 | CCL out\_ready, barrier, secondary\_sync | CCL broadcast (skipped in single-device) |
| 3 | `mcast_data_sender_semaphore` | Mcast1, Mcast2 sender $\to$ receiver sync |
| 4 | `mcast_data_receiver_semaphore` | Mcast1, Mcast2 receiver ready signal |
| 5 | `gather_noc0_receiver_semaphore` | GatherReduce + DKV Gather (NOC0 senders) |
| 6 | `gather_noc1_receiver_semaphore` | GatherReduce (NOC1) |
| 7 | `mla_reducer_semaphore` | FlashMLA tree reduction |
| 8 | `mla_mcast_semaphore` | FlashMLA KV cache multicast |
| 9 | `mla_ncrisc_brisc_sync_semaphore` | FlashMLA NCRISC/BRISC synchronization |
| 10 | `mla_receiver_ready_semaphore` | FlashMLA receiver readiness |
| 11 | `mla_q_input_mcast_semaphore` | FlashMLA Q input multicast |
| 12 | `mla_kv_cache_cur_pos_ready_semaphore` | KV cache position sync with MLA |

### Stage Pipeline Table (Q-Path)

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|------|-------------|--------------|-----------|------|
| 1 | CCL Broadcast | NCRISC+BRISC | `bcast_pkt_cb` (44) | `input_cb` (0) | Sender core | Global sems 0--2 |
| 2 | RMSNorm1 | TRISC | `input_cb` (0), `gamma_cb` (1) | `rmsnorm_output_cb` (2) | Sender core | CB handshake |
| 3 | Mcast1 | BRISC (send) / NCRISC (recv) | `rmsnorm_output_cb` (2) | `matmul_input_cb` (5) | Sender $\to$ 129 receivers | Sems 3--4 |
| 4 | $W_{qa}$ Matmul (K-split) | TRISC | `matmul_input_cb` (5), `matmul_weights_cb` (3) | `matmul_output_cb` (4) | 96 cores ($8 \times 12$) | CB handshake |
| 5 | Gather-Reduce | NCRISC (send) / BRISC (recv) | `matmul_output_cb` (4) | `rmsnorm2_input_cb` (7) + scratch (8) | 96 $\to$ sender | Sems 5--6 |
| 6 | RMSNorm2 + Reduce | TRISC | `rmsnorm2_input_cb` (7), `rmsnorm2_gamma_cb` (6), scratch (8) | `rmsnorm2_output_cb` (9) | Sender core | CB handshake |
| 7 | Mcast2 | BRISC (send) / NCRISC (recv) | `rmsnorm2_output_cb` (9) | `matmul2_input_cb` (10) | Sender $\to$ 129 receivers | Reuse sems 3--4 |
| 8a | $W_{qb}$ Matmul2 | TRISC | `matmul2_input_cb` (10), `matmul2_weights_cb` (11) | `matmul2_output_cb` (12) | 96 cores ($8 \times 12$) | CB handshake |
| 8b | $W_{kv\_b1}$ Matmul3 | TRISC | `matmul2_output_cb` (12), `matmul3_weights_cb` (13) | `matmul3_output_cb` (14) | 64 Qnope cores (cols 0--7) | CB handshake |
| 8c | QRoPE | TRISC | `matmul2_output_cb` (12), cos/sin/trans\_mat (17--19) | `qrope_output_cb` (15) | 32 Qrope cores (cols 8--11) | CB handshake |
| 9 | CreateQHeads (3-phase unicast) | NCRISC (send) / BRISC (recv) / TRISC (tilize) | `matmul3_output_cb` (14), `qrope_output_cb` (15) | `create_q_heads_out_cb` (16) | 96 senders $\to$ 8 SDPA cores | 3 sems (reused 5, 6, 3) |
| 10 | Flash MLA Decode | All RISCs | `create_q_heads_out_cb` (16), KV cache | `mla_out_final_cb` (43) | 64 MLA cores | MLA semaphores 7--12 |

### Stage Pipeline Table (KV-Path, parallel with Q-Path stages 4+)

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|------|-------------|--------------|-----------|------|
| 4' | $W_{dkv}$ Matmul | TRISC | `matmul_input_cb` (5), `dkv_matmul_weights_cb` (23) | `dkv_matmul_output_cb` (24) | 18 KV cores ($9 \times 2$) | CB handshake |
| 5' | DKV Gather | NCRISC (send) / BRISC (recv) | `dkv_matmul_output_cb` (24) | `kv_rmsnorm_input_cb` (25) | 16 senders $\to$ 1 receiver | Sems 5--6 |
| 6' | KV RMSNorm | TRISC | `kv_rmsnorm_input_cb` (25), `kv_rmsnorm_gamma_cb` (26) | `kv_rmsnorm_output_cb` (27) | 1 KV-norm core | CB handshake |
| 7' | K-RoPE | TRISC | `dkv_matmul_output_cb` (24), cos/sin (29--30) | `krope_output_cb` (28) | 2 K-RoPE cores | CB handshake |
| 8' | KV Cache Write | BRISC | `kv_rmsnorm_output_cb` (27), `krope_output_cb` (28) | DRAM KV cache | KV-norm + K-RoPE cores | Position ID runtime arg |

### Key Implementation Details

- **K-split matmul with gather-reduce:** The $W_{qa}$ matmul splits K=7168 across 96 cores in two halves (48 cores each, `matmul_k_per_core = matmul_act_total_tiles // 2`). Each half computes a partial sum. The gather-reduce collects partial sums on the sender core: half-0 writes to `rmsnorm2_input_cb` (CB 7), half-1 writes to `gather_reduce_half1_scratch_cb` (CB 8). TRISC adds them element-wise and applies RMSNorm2.

- **OverlappedTensor usage:** `gamma_tensor`, `matmul_weights_tensor`, `rmsnorm2_gamma_tensor`, `matmul2_weights_tensor`, and `matmul3_weights_tensor` are all `OverlappedTensor` instances. The gamma and matmul weights share one fused buffer; KV-B1 and KV-B2 projections share another.

- **3-phase CreateQHeads:** Uses three semaphores (reusing addresses from completed gather/mcast stages) to synchronize unicast transfers: Phase 1 sends first 256 elements of each NOPE head, Phase 2 sends remaining elements, Phase 3 sends ROPE heads. Each SDPA input core's TRISC tilizes the interleaved $[8, 576]$ layout ($576 = 512_{\text{nope}} + 64_{\text{rope}}$) into $8 \times 32$ tiles.

- **Interleaved Q layout:** Matmul2 uses shuffled weights so that each row of the 8x12 grid produces interleaved Qnope + Qrope heads, avoiding a separate gather step.

- **FlashMLA:** 64 MLA cores use double-buffered K (`mla_k_in_cb`) for overlapping DRAM reads and compute, with tree reduction across S-blocks coordinated by `mla_reducer_semaphore`. NCRISC waits on `mla_kv_cache_cur_pos_ready_semaphore` to ensure the current position's KV data has been written before reading.

---

## 4.2.3 `kv_cache_branch` -- Standalone KV Branch

**Source:** `kv_cache_branch/op.py` (570 lines)

This fused op is the standalone version of the KV branch that also appears embedded in `pre_sdpa`. It fuses DKV matmul + gather + RMSNorm + RoPE + KV cache update.

### Core Grid Diagram (9x2 input grid)

```
     Col 0   1   2   3   4   5   6   7   8
Row 0  [MM] [MM] [MM] [MM] [MM] [MM] [MM] [MM] [RP]
Row 1  [MM] [MM] [MM] [MM] [MM] [MM] [MM] [MM] [RP]

Legend:
  MM = DKV matmul core (16 cores, knope senders)
  RP = RoPE core (2 cores at (8,0) and (8,1))
  [RN] = RMSNorm core (not shown, from gamma_tensor shard grid)
```

The DKV gather sender grid is the matmul grid minus the RoPE grid. The 16 nope cores send their outputs to the RMSNorm receiver core.

### CB Layout (13 CBs)

| CB | Index | Tile Format | Role |
|----|-------|-------------|------|
| `cos_cb` | 0 | Rope tile | RoPE cosine (NCRISC reads from DRAM) |
| `sin_cb` | 1 | Rope tile | RoPE sine (NCRISC reads from DRAM) |
| `trans_mat_cb` | 2 | $32 \times 32$ | RoPE transformation matrix (sharded) |
| `rotated_input_interm_cb` | 3 | $1 \times 32$ | RoPE intermediate |
| `cos_interm_cb` | 4 | $1 \times 32$ | RoPE cos intermediate |
| `sin_interm_cb` | 5 | $1 \times 32$ | RoPE sin intermediate |
| `dkv_matmul_input_cb` | 6 | $1 \times 32$ | DKV matmul input (224 tiles) |
| `dkv_matmul_output_cb` | 7 | $1 \times 32$ | DKV matmul per-core output |
| `dkv_matmul_weights_cb` | 8 | weight tile | DKV matmul weights (sharded) |
| `kv_rmsnorm_input_cb` | 9 | $16 \times 32$ | Gathered KV data (1 tile) |
| `kv_rmsnorm_gamma_cb` | 10 | $16 \times 32$ | KV RMSNorm gamma |
| `kv_rmsnorm_output_cb` | 11 | $16 \times 32$ | KV RMSNorm output |
| `k_rope_output_cb` | 12 | $1 \times 32$ | K-RoPE output |

### Semaphores (2 local)

| ID | Purpose |
|----|---------|
| 2 | Gather NOC0 receiver |
| 3 | Gather NOC1 receiver |

### Stage Pipeline Table

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|------|-------------|--------------|-----------|------|
| 1 | DKV Matmul | NCRISC + TRISC | `dkv_matmul_input_cb` (6), `dkv_matmul_weights_cb` (8) | `dkv_matmul_output_cb` (7) | $9 \times 2$ grid (18 cores) | CB handshake |
| 2 | DKV Gather (K-nope) | NCRISC (send) / BRISC (recv) | `dkv_matmul_output_cb` (7) | `kv_rmsnorm_input_cb` (9) | 16 senders $\to$ KV-norm core | Sem 2 |
| 3 | KV RMSNorm | TRISC | `kv_rmsnorm_input_cb` (9), `kv_rmsnorm_gamma_cb` (10) | `kv_rmsnorm_output_cb` (11) | 1 KV-norm core | CB handshake |
| 4 | K-RoPE | NCRISC + TRISC | `dkv_matmul_output_cb` (7), cos/sin (0,1), `trans_mat_cb` (2) | `k_rope_output_cb` (12) | 2 K-RoPE cores | CB handshake |
| 5 | KV Cache Write | BRISC | `kv_rmsnorm_output_cb` (11), `k_rope_output_cb` (12) | DRAM KV cache | KV-norm + K-RoPE cores | Runtime arg (position\_id) |

### Key Implementation Details

- **Role flags:** `is_dkv_matmul_core`, `is_kv_rmsnorm_core`, `is_knope_core`, `is_krope_core`.
- **Per-core `start_tile_offset`:** K-RoPE cores use `PerCoreCompileTimeDescriptor` to set per-core tile offsets for reading the correct position from the cos/sin DRAM tables.

---

## 4.2.4 `post_sdpa` -- Matmul Chain + CCL AllReduce

**Source:** `post_sdpa/op.py` (1802 lines)

The post-attention fused op chains two matmuls (kv\_b2 projection and o\_proj) with inter-device all-reduce, and optionally includes an SDPA reduce-to-all phase for multi-device attention.

### Tensor Shape Trace

```
Input:  [1, 512] per head from SDPA output (on 64 matmul4 cores)
  --> Matmul4 (kv_b2):  [1, 512] x [512, 128] -> [1, 128] per core
  --> Gather2: 64 cores -> [1, 8192] on (12,9)
  --> Mcast3: [1, 8192] -> 130 cores (13x10 grid)
  --> Matmul5 (o_proj): [1, 8192] x [8192, 64] -> [1, 64] per core
  --> Gather3: 112 cores -> [1, 7168] on (12,9)
  --> CCL AllReduce: sum across devices + residual add
Output: [1, 7168]
```

### Core Grid Diagram (13x10 = 130 cores)

```
     Col 0   1   2   3   4   5   6   7   8   9  10  11  12
Row 0  [D4] [M5] [M5] [D4] [M5] [M5] [M5] [D4] [M5] [D4] [M5] [M5] [P ]
Row 1  [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [P ]
Row 2  [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [P ]
Row 3  [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [P ]
Row 4  [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [P ]
Row 5  [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [P ]
Row 6  [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [P ]
Row 7  [M5] [D4] [M5] [M5] [D4] [M5] [D4] [M5] [M5] [D4] [M5] [M5] [P ]
Row 8  [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [P ]
Row 9  [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [M5] [CS] [G ]

Legend:
  D4 = DRAM worker (8 cores) -- receives mcast3, skips matmul4/5
  P  = Phantom (9 cores, col 12 rows 0-8) -- receives mcast3, skips matmul
  M5 = Matmul5 active core (112 cores, o_proj: 12x8 + 8x2 layout)
       Also serves as Matmul4 core where overlapped with kv_b2 grid
  CS = CCL sender core (11, 9) -- reads gather3 data, sends via fabric
  G  = Gather/CCL receiver core (12, 9) -- receives gathers, performs CCL reduce
```

### Matmul4 Sub-Grid (kv_b2, 64 cores: $5 \times 8 + 12 \times 2$)

```
     Col 0   1   2   3   4   5   6   7   8   9  10  11
Row 0  [--] [M4] [M4] [--] [M4] [M4] [M4] [--] [M4] [--] [M4] [M4]
Row 1  [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4]
...
Row 5  [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4]
Row 6  [--] [--] [--] [--] [--] [--] [--] [--] [--] [--] [--] [--]
Row 7  [--] [--] [--] [--] [--] [--] [--] [--] [--] [--] [--] [--]
Row 8  [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4]
Row 9  [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [M4] [--]
```

### CB Layout (14 base CBs + up to 11 SDPA CBs)

| CB | Index | Core(s) | Role |
|----|-------|---------|------|
| `matmul4_in0_cb` | 0 | 64 kv\_b2 cores | Matmul4 input (sharded SDPA output) |
| `matmul4_in1_cb` | 1 | 64 kv\_b2 cores | Matmul4 weights ($W_{kv\_b2}$, OverlappedTensor) |
| `matmul4_out_cb` | 2 | 64 kv\_b2 cores | Matmul4 output (4 tiles $1 \times 32$) |
| `gather2_dst_cb` | 3 | (12,9) | Gather2 destination = Mcast3 source |
| `matmul5_in0_cb` | 4 | 130 mcast cores | Mcast3 destination = Matmul5 input |
| `matmul5_in1_cb` | 5 | 112 o\_proj cores | Matmul5 weights ($W_{o\_proj}$, OverlappedTensor) |
| `matmul5_out_cb` | 6 | 112 o\_proj cores | Matmul5 output (2 tiles $1 \times 32$) |
| `gather3_dst_cb` | 7 | (12,9) | Gather3 destination = CCL local data |
| `ccl_sender_in_cb` | 8 | (11,9) | CCL sender reads gather3 output |
| `ccl_remote_data_cb` | 9 | (12,9) | CCL received remote data |
| `ccl_residual_cb` | 10 | (12,9) | Residual tensor (optional) |
| `ccl_temp_cb` | 11 | (12,9) | CCL compute scratch |
| `ccl_output_cb` | 12 | (12,9) | CCL final output |
| `ccl_packet_header_cb` | 13 | (11,9) + (12,9) | Fabric packet headers |

When the optional SDPA reduce-to-all is enabled, CBs 14--24 are added for the SDPA phase.

### Semaphores

| ID | Purpose |
|----|---------|
| 0--1 | Gather2: NOC0/NOC1 receiver |
| 2--3 | Mcast3: sender/receiver |
| 4--5 | Gather3: NOC0/NOC1 receiver |
| 6 | Gather3 completion (signals CCL sender) |
| Global | CCL semaphore1/semaphore2 (inter-device sync) |

### Stage Pipeline Table

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|------|-------------|--------------|-----------|------|
| 0 | SDPA Reduce-to-All (optional) | All RISCs | SDPA output | `matmul4_in0` (0) | SDPA workers + forwarders | Sems 8--11 |
| 1 | Matmul4 ($W_{kv\_b2}$) | NCRISC + TRISC | (0), (1) | (2) | 64 kv\_b2 cores | CB handshake |
| 2 | Gather2 | NCRISC (send) / BRISC (recv) | (2) | (3) | 64 $\to$ (12,9) | Sem 0 |
| 3 | Mcast3 | BRISC (send) / NCRISC (recv) | (3) | (4) | (12,9) $\to$ 129 receivers | Sems 2--3 |
| 4 | Matmul5 ($W_{o\_proj}$) | NCRISC + TRISC | (4), (5) | (6) | 112 o\_proj cores | CB handshake |
| 5 | Gather3 | NCRISC (send) / BRISC (recv) | (6) | (7) | 112 $\to$ (12,9) | Sems 4--5, completion sem 6 |
| 6 | CCL All-Reduce | NCRISC + TRISC | (7), (9), (10) | (12) | (12,9) recv + (11,9) send | Global sems + sem 6 |

### Role Flags (8 total: 6 base + 2 SDPA optional)

| Flag | Grid | Purpose |
|------|------|---------|
| `is_matmul4_core` | 64-core kv\_b2 grid | Matmul4 compute |
| `is_gather_receiver_core` | (12,9) | Gather2/3 destination + CCL receiver |
| `is_matmul5_core` | 112-core o\_proj grid | Matmul5 compute |
| `is_mcast3_receiver_core` | 129-core grid (130 minus sender) | Mcast3 data reception |
| `is_ccl_sender_core` | (11,9) | CCL fabric sender |
| `is_ccl_receiver_core` | (12,9) | CCL fabric receiver |
| `is_sdpa_worker_core` | SDPA worker grid (optional) | SDPA reduce-to-all workers |
| `is_sdpa_forwarder_core` | SDPA forwarder grid (optional) | SDPA reduce-to-all forwarders |

### Key Implementation Details

- **Non-rectangular matmul grids:** Both matmul4 ($5 \times 8 + 12 \times 2 = 64$ cores) and matmul5 ($12 \times 8 + 8 \times 2 = 112$ cores) use non-rectangular `CoreRangeSet` layouts derived from production weight overlap specs. Per-core `gather_sender_idx` values are assigned via `PerCoreCompileTimeDescriptor`.

- **SDPA buffer overlapping:** When `sdpa_kv_cache_buffer` is provided, matmul4\_out, matmul5\_in0, matmul5\_out, ccl\_sender\_in, and ccl\_temp CBs are overlapped into the SDPA KV cache buffer using `address_offset` stacking.

- **Inactive mcast cores:** The 8 DRAM workers and 9 phantom cores in the 130-core mcast3 grid receive the broadcast data (required for rectangular mcast) but have `is_matmul5_core=0`, so they skip the matmul and gather stages.

- **CCL all-reduce:** Uses a 2-device neighbor-exchange pattern. The sender core (11,9) reads from the adjacent gather core (12,9) and transmits via fabric. The receiver accumulates: `output = local_data + remote_data + residual`.

---

**Prev:** [Fusion Principles and Patterns](01_fusion_principles_and_patterns.md) | **Next:** [MoE Fused Ops](03_moe_fused_ops.md)
