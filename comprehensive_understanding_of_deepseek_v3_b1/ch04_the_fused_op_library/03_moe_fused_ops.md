# 4.3 MoE Fused Ops

This section covers the six fused operations that implement DeepSeek V3's Mixture-of-Experts (MoE) layer. The MoE path is the most resource-intensive part of the model, routing each token through 8 of 256 experts plus a shared expert. The fused ops here range from the top-level `moe` orchestrator (4442 lines) to the single-core `gated_local_reduce` (176 lines).

Source: `models/demos/deepseek_v3_b1/fused_ops/{moe,moe_routed_expert,shared_expert,gated_local_reduce,gated_local_reduce_down_proj,down_proj}/op.py`

---

## 4.3.1 `moe` -- The Top-Level MoE Orchestrator

**Source:** `moe/op.py` (4442 lines)

The `moe` fused op is the largest source file in the model. It composes the routed expert and shared expert paths into a single kernel dispatch, managing 17 global semaphores and CB reconfiguration between phases.

### Composition Tree

```
moe
├── [Routed Expert Path] (MoeRoutedExpertOp)
│   ├── rmsnorm (TRISC: sender core)
│   │   └── [1, 7168] → [1, 7168] normalized
│   ├── input_mcast (BRISC→NCRISC: sender → 130-core grid)
│   │   └── Broadcast [1, 7168] to all compute cores
│   ├── gate_matmul (NCRISC+TRISC: SRAM matmul with sigmoid)
│   │   └── [1, 7168] × [7168, N_routing] → [1, 256] with sigmoid
│   ├── gate_gather (NCRISC: compute cores → sender)
│   │   └── Collect [1, 256] (routing scores)
│   ├── deepseek_moe_gate (TRISC: sender core)
│   │   └── Top-8 selection: [1, 256] → 8 × (index, score)
│   ├── index_mcast (BRISC: sender → all compute cores)
│   ├── expert_scale_mcast (BRISC: sender → all compute cores)
│   ├── dram_streaming_matmul (NCRISC+TRISC: gate_proj with SiLU)
│   │   └── [1, 7168] × [7168, N_expert] → [1, N_expert] (indexed DRAM read)
│   ├── dram_streaming_matmul (NCRISC+TRISC: up_proj, no activation)
│   │   └── [1, 7168] × [7168, N_expert] → [1, N_expert] (indexed DRAM read)
│   ├── eltwise_mul (TRISC: SiLU(gate) × up × expert_scale)
│   ├── down_proj_gather (NCRISC: compute cores → sender)
│   ├── down_proj_mcast (BRISC: sender → 130-core grid)
│   ├── down_proj_matmul (NCRISC+TRISC: SRAM matmul)
│   ├── eltwise_add (TRISC: matmul_out + residual shard)
│   └── reduce_to_one_b1 (NCRISC+BRISC: optional, 4×2 mesh)
│       └── 3-level tree reduction across 8 devices
│
├── [Shared Expert Path] (MoeSharedExpertOp)
│   ├── activation_mcast (BRISC: sender → 130-core grid)
│   ├── kn_sliced_gate_matmul (NCRISC+TRISC: 64 A-cores)
│   ├── kn_sliced_up_matmul (NCRISC+TRISC: 64 B-cores)
│   ├── gate_gather (NCRISC: 64 A-cores → sender)
│   ├── up_gather (NCRISC: 64 B-cores → sender)
│   ├── gated_reduce (TRISC: sender core)
│   │   └── SiLU(sum(gate)) × sum(up) → [1, K_down]
│   ├── shared_down_mcast (BRISC: sender → 130-core grid)
│   ├── shared_down_matmul (NCRISC+TRISC: 112 cores)
│   ├── residual_add (TRISC: matmul_out + bias shard)
│   └── output_gather (NCRISC: 112 cores → sender)
│
└── [Final Combine]
    └── Routed result + Shared result → final output
```

### Architecture

The file defines three main classes:

| Class | Role |
|-------|------|
| `MoeOp` | Top-level orchestrator: composes both paths into a single `generic_op` call |
| `MoeRoutedExpertOp` | Context setup and build methods for the routed expert path |
| `MoeSharedExpertOp` | Context setup and build methods for the shared expert path |

Plus two key dataclasses:

| Dataclass | Role |
|-----------|------|
| `MoeContext` | Top-level context holding both routed and shared contexts, mesh info, reduce params, CB reconfig flags |
| `_MoeRoutedExpertContext` | All computed values for routed expert: grids, CBs, semaphores, setup dicts, per-core bank IDs |
| `_MoeSharedExpertContext` | All computed values for shared expert: A/B grids, gather params, face-view config |

### `MoeSem` -- Semaphore Index Constants (17 Global Semaphores)

| Index | Name | Purpose |
|-------|------|---------|
| 0 | `MCAST_SENDER` | Input mcast sender sync |
| 1 | `MCAST_DATA_RECEIVER` | Input mcast receiver ready |
| 2 | `DOWN_PROJ_GATHER` | Down-proj output gather |
| 3 | `RESIDUAL_MCAST_RECEIVER` | Residual data mcast (shared expert) |
| 4 | `AG_GATHER` | Shared expert gate-gather (A) receiver |
| 5 | `SHARED_DOWN_MCAST_RECEIVER` | Shared expert down-proj mcast |
| 6 | `BG_GATHER` | Shared expert up-gather (B) receiver |
| 7 | `SHARED_OUTPUT_MCAST_RECEIVER` | Shared expert output mcast |
| 8 | `OUTPUT_GATHER` | Shared expert output gather |
| 9 | `EXPERT_SCALE_MCAST_RECEIVER` | Expert scale broadcast |
| 10 | `INDEX_MCAST_RECEIVER` | Expert index broadcast |
| 11 | `DOWN_PROJ_MCAST_RECEIVER` | Down-proj input mcast |
| 12--15 | `REDUCE_WORKER_FABRIC_BASE` | Per-worker-slot fabric semaphores (4 slots) |
| 16 | `REDUCE_SYNC` | ReduceToOne synchronization |

Semaphore reuse is safe because stages are temporally ordered. For example, `MCAST_SENDER` (ID 0) is reused across the input activation mcast, the down-proj activation mcast, and the shared expert activation mcast -- each executes only after the prior stage completes.

### CB ID Management via `CircularBufferIdManager`

The MoE op is the primary consumer of `CircularBufferIdManager` (see Section 4.1.2). The routed and shared expert paths each require 20--30 CB indices, but they execute sequentially on overlapping core grids. The manager creates separate contexts:

```python
cb_id_manager = CircularBufferIdManager()
routed_ctx = cb_id_manager.create_context()   # Routed expert CBs
shared_ctx = cb_id_manager.create_context()   # Shared expert CBs
```

CBs with matching `(data_format, tile)` can share indices across contexts. The CB format groupings are:

| Tile Format | Example CBs | Count |
|-------------|-------------|-------|
| 1x32, bf16 | `gate_mm_input_cb`, `gate_proj_cb_out`, `up_proj_cb_mm_out`, `down_proj_gather_dst_cb`, `down_proj_mcast_dst_cb`, `down_proj_cb_out`, `residual_mcast_dst_cb` | 7+ |
| 16x16, bf16 | `mul_cb_in0`, `mul_cb_in1`, `mul_cb_out`, `gate_input_cb`, `mul_cb_scalar`, `mul_cb_scalar_src` | 6 |
| 32x32, bf16 | `add_cb_in0`, `add_cb_in1`, `add_cb_out`, reduce CBs | 8+ |
| Weight tile | `gate_proj_cb_in1` / `up_proj_cb_in1` (aliased), `down_proj_cb_in1`, `gate_mm_weights_cb` | 4--6 |

**Intentional alias:** `up_proj_cb_in1 = gate_proj_cb_in1` -- these sequential DRAM streaming matmuls share the same weight CB slot because they never overlap.

### CB Reconfiguration

When `reconfig_moe_cbs=True`, the kernel reads a reconfig tensor at startup to reprogram all CB read/write interfaces. This is necessary when the MoE op follows `pre_sdpa` (which leaves a completely different CB configuration in firmware):

```python
if ctx.reconfig_moe_cbs:
    cb_metadata = record_cb_metadata(all_cb_descriptors)
    reconfig_tensor = build_cb_reconfig_tensor(cb_metadata, full_device_grid, mesh_device)
```

The kernel's first action is to call `setup_local_cb_read_write_interfaces()` from the reconfig tensor.

### SDPA Buffer Overlap

Many MoE CB descriptors are initially created with `tensor=None` (deferred) and later overridden by `_overlap_cbs_with_sdpa_buffer`. This allows the MoE op's intermediate buffers to share physical memory with the SDPA KV-cache buffer (which is idle during MoE execution), reducing peak L1 usage.

---

## 4.3.2 `moe_routed_expert` -- 14-Stage Expert Pipeline (Deep Dive)

**Source:** `moe_routed_expert/op.py` (2150 lines)

This is the second most complex fused op, implementing the full routed expert computation from input multicast through cross-device reduce.

### Composition Tree

```
moe_routed_expert
├── mcast (BRISC: input activation to compute grid)
│   └── [1, K] → all compute cores
├── sram_matmul (NCRISC+TRISC: gate routing, fused sigmoid)
│   └── [1, K] × [K, 256] → [1, 256] with sigmoid
├── gather (NCRISC: compute cores → sender)
│   └── Collect [1, 256] routing scores
├── deepseek_moe_gate (TRISC: top-8 selection)
│   └── [1, 256] → 8 × (index, score)
├── index_mcast (BRISC: expert index to all cores)
├── scale_mcast (BRISC: expert scale to all cores)
├── dram_streaming_matmul (NCRISC+TRISC: gate_proj, fused SiLU)
│   └── [1, K] × [K, N_expert/core] → [1, N_expert/core]
├── dram_streaming_matmul (NCRISC+TRISC: up_proj)
│   └── [1, K] × [K, N_expert/core] → [1, N_expert/core]
├── eltwise_mul (TRISC: SiLU(gate) × up × scale)
│   └── Element-wise with 16×16 tile reinterpretation
├── down_proj_gather (NCRISC: compute → sender)
├── down_proj_mcast (BRISC: sender → all cores)
├── down_proj_matmul (NCRISC+TRISC: SRAM)
│   └── [1, K_down] × [K_down, N/core] → [1, N/core]
├── eltwise_add (TRISC: + residual)
└── reduce_to_one_b1 (NCRISC+BRISC: cross-device, optional)
    └── 3-level tree reduction: LEAF → ROOT3 → ROOT2 → ROOT1
```

### Full Stage Pipeline

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|------|-------------|--------------|-----------|------|
| 1 | Input Mcast | BRISC (send) / NCRISC (recv) | `input_cb` (0) | `gate_mm_input_cb` (1) | Sender $\to$ mcast grid | Sems 0, 1 |
| 2 | Gate MM + Sigmoid | NCRISC (weight read) + TRISC | `gate_mm_input_cb` (1), `gate_mm_weights_cb` (2) | `gate_mm_output_cb` (3) | Gate MM cores | CB handshake |
| 3 | Gate Gather | NCRISC (send) / BRISC (recv) | `gate_mm_output_cb` (3) | `gate_input_cb` (4) | Gate MM $\to$ sender | Sem 2 |
| 4 | Gate (top-K select) | TRISC | `gate_input_cb` (4), `gate_bias_cb` (5), `gate_indices_cb` (6) | `gate_output_cb` (7), `gate_output_indices_cb` (8) | Sender core | CB handshake |
| 5 | Index Mcast | BRISC (send) / NCRISC (recv) | `gate_output_indices_cb` (8) | `gate_proj_cb_index` (10) | Sender $\to$ mcast grid | Sem 10 |
| 6 | Expert Scale Mcast | BRISC (send) / NCRISC (recv) | `gate_output_cb` (7) | `mul_cb_scalar_src` (21) | Sender $\to$ mcast grid | Sem 9 |
| 7 | gate\_proj (DRAM streaming, +SiLU) | NCRISC (DRAM read) + TRISC | `gate_mm_input_cb` (1), `gate_proj_cb_in1` (9) | `gate_proj_cb_out` (11) | DRAM-optimal cores | Expert-indexed |
| 8 | up\_proj (DRAM streaming) | NCRISC (DRAM read) + TRISC | `gate_mm_input_cb` (1), `up_proj_cb_in1` (12) | `up_proj_cb_mm_out` (13) | DRAM-optimal cores | Expert-indexed |
| 9 | Fused Mul | TRISC | `mul_cb_in0` (14), `mul_cb_in1` (15), `mul_cb_scalar` (22) | `mul_cb_out` (16) | DRAM-optimal cores | CB aliasing (16x16 view) |
| 10 | down\_proj Gather | NCRISC (send) / BRISC (recv) | `mul_cb_out` (16) | `down_proj_gather_dst_cb` (17) | Compute $\to$ sender | Sem 2 |
| 11 | down\_proj Mcast | BRISC (send) / NCRISC (recv) | `down_proj_gather_dst_cb` (17) | `down_proj_mcast_dst_cb` (18) | Sender $\to$ mcast grid | Sems 0, 11 |
| 12 | down\_proj (SRAM matmul) | TRISC | `down_proj_mcast_dst_cb` (18), `down_proj_cb_in1` (19) | `down_proj_cb_out` (20) | Expert weight cores | CB handshake |
| 13 | Eltwise Add | TRISC | `add_cb_in0` (23), `add_cb_in1` (24) | `add_cb_out` (25) | Expert weight cores | Per-core index |
| 14 | ReduceToOne | NCRISC (fabric) + TRISC | `add_cb_out` (25) | `reduce_output_cb` (29) | Worker + fabric cores | Fabric sems 12--16 |

### CB Map (33 CBs, indices 0--32)

| CB | Index | Role |
|----|-------|------|
| `input_cb` | 0 | Input tensor (sharded on sender core) |
| `gate_mm_input_cb` | 1 | Mcast destination = gate MM + expert matmul input |
| `gate_mm_weights_cb` | 2 | Gate matmul weights (overlapped tensor) |
| `gate_mm_output_cb` | 3 | Gate matmul output with fused sigmoid |
| `gate_input_cb` | 4 | Gathered gate scores on sender (tensor-backed) |
| `gate_bias_cb` | 5 | Gate bias (tensor-backed) |
| `gate_indices_cb` | 6 | Gate indices (tensor-backed) |
| `gate_output_cb` | 7 | Gate output scores (tensor-backed) |
| `gate_output_indices_cb` | 8 | Gate output indices (tensor-backed) |
| `gate_proj_cb_in1` | 9 | DRAM gate\_proj weights working buffer (triple-buffered) |
| `gate_proj_cb_index` | 10 | Expert index (receives mcasted indices) |
| `gate_proj_cb_out` | 11 | gate\_proj output (tensor-backed) |
| `up_proj_cb_in1` | 12 | DRAM up\_proj weights working buffer |
| `up_proj_cb_mm_out` | 13 | up\_proj intermediate output |
| `mul_cb_in0` | 14 | up\_proj output aliased as $16 \times 16$ |
| `mul_cb_in1` | 15 | gate\_proj output aliased as $16 \times 16$ |
| `mul_cb_out` | 16 | Fused output: $\text{SiLU}(g) \times u \times s$ |
| `down_proj_gather_dst_cb` | 17 | Gathered fused output on sender |
| `down_proj_mcast_dst_cb` | 18 | Mcasted fused output on compute cores |
| `down_proj_cb_in1` | 19 | DRAM down\_proj weights working buffer |
| `down_proj_cb_out` | 20 | down\_proj output |
| `mul_cb_scalar_src` | 21 | Receives mcasted expert scale |
| `mul_cb_scalar` | 22 | Working buffer for scalar multiply ($16 \times 16$) |
| `add_cb_in0` | 23 | down\_proj output aliased as $32 \times 32$ for add |
| `add_cb_in1` | 24 | fused\_add (replicated on all cores) |
| `add_cb_out` | 25 | Final output |
| `reduce_received_cb_r1` | 26 | ReduceToOne round 1 receive buffer |
| `reduce_received_cb_r2` | 27 | ReduceToOne round 2 receive buffer |
| `reduce_received_cb_r3` | 28 | ReduceToOne round 3 receive buffer |
| `reduce_output_cb` | 29 | ReduceToOne final output |
| `reduce_scratch_cb` | 30 | ReduceToOne compute scratch |
| `reduce_packet_cb` | 31 | ReduceToOne packet buffer |
| `reduce_packet_header_cb` | 32 | ReduceToOne packet header (persistent, 96 bytes) |

### Core Grid Diagram

The routed expert operates on the standard $13 \times 10$ grid, with the sender/gather core at $(12, 9)$:

```
     Col 0   1   2   3   4   5   6   7   8   9  10  11  12
Row 0  [D ] [C ] [C ] [D ] [C ] [C ] [C ] [D ] [C ] [D ] [C ] [C ] [P ]
Row 1  [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [P ]
Row 2  [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [P ]
Row 3  [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [P ]
Row 4  [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [P ]
Row 5  [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [P ]
Row 6  [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [P ]
Row 7  [C ] [D ] [C ] [C ] [D ] [C ] [D ] [C ] [C ] [D ] [C ] [C ] [P ]
Row 8  [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [P ]
Row 9  [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [C ] [S ]

Legend:
  C = Compute core (gate_proj + up_proj + mul)
  D = DRAM worker (8 cores) -- receives mcast, skips matmul
  P = Phantom (9 cores, col 12 rows 0-8) -- receives mcast, skips matmul
  S = Sender/Gather/Mcast core (12, 9)
```

### Core Role Descriptors (from source code)

| Flag | Grid | Purpose |
|------|------|---------|
| `is_sender_core` | Single core $(12,9)$ | Input mcast sender, gate host, gather receiver |
| `is_mcast_grid_core` | Rectangular mcast grid | Input/down\_proj mcast receivers |
| `is_gate_mm_core` | Gate MM cores (overlapped tensor grid) | Gate matmul + sigmoid |
| `is_gate_proj_core` | DRAM-optimal cores | Expert matmuls + mul + add |
| `is_reduce_worker_core` | Same as gate\_proj (when reduce enabled) | ReduceToOne data source |
| `is_reduce_fabric_core` | Per-column fabric cores (set per-device) | ReduceToOne fabric forwarding |

### DRAM Streaming Matmul Details

The gate\_proj, up\_proj, and down\_proj stages all use DRAM streaming matmul with **expert-indexed weights**. The `setup_dram_matmul` helper computes:

- **Triple buffering:** `num_in1_buffers = 3` for overlapping DRAM reads with compute. While TRISC processes subblock $i$, NCRISC reads subblock $i+1$ and $i+2$ into the next buffer slots.
- **Subblock K:** The K dimension is split into `num_subblocks_k` chunks (4 for gate/up, 2 for down). Each subblock is `Kt // num_subblocks_k` tiles.
- **NOC page size:** Calculated via `get_max_page_size_and_num_pages` to respect the architecture's NOC burst limit (16384 bytes on Blackhole, 8192 bytes on Wormhole).
- **Expert indexing:** Each DRAM matmul core receives `gate_proj_index_offset = mesh_chip_id` to index into the correct expert's weight slab. The mcasted `gate_proj_cb_index` (CB 10) carries the expert index from the gate's top-K selection. The DRAM address is computed as `base_addr + expert_index * expert_stride`.

The `subblock_w` is computed to maximize DEST utilization:

| Config | `max_dest` |
|--------|-----------|
| `dst_full_sync_en=True`, FP32 | 8 |
| `dst_full_sync_en=True`, BF16 | 16 |
| `dst_full_sync_en=False`, FP32 | 4 |
| `dst_full_sync_en=False`, BF16 | 8 |

### Per-Core Bank ID and Virtual Channel Assignment

Each DRAM matmul core is mapped to an optimal DRAM bank via `device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)`. The bank ID and a conflict-free virtual channel (VC) are assigned per core:

```python
for idx, core in enumerate(gate_proj_optimal_workers):
    bank_id = core_to_bank_id[(core.x, core.y)]
    vc = bank_id & 0x3
    # Resolve VC conflicts for cores sharing the same NOC row
    for j in range(idx):
        prev_core = gate_proj_optimal_workers[j]
        if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
            vc = (vc + 1) & 0x3
            break
```

This produces `PerCoreCompileTimeDescriptor` entries: `gate_proj_bank_id`, `gate_proj_vc`, `up_proj_bank_id`, `up_proj_vc`, `down_proj_bank_id`, `down_proj_vc`.

### Fused Mul Stage (SiLU(gate) * up * scale) -- CB Aliasing

The `setup_eltwise_mul` helper creates aliased CB descriptors that view matmul output memory (in $1 \times 32$ tile format) as $16 \times 16$ tiles for the mul compute kernel:

```python
# Reinterpret 1x32 matmul output as 16x16 tiles
total_elements = M * per_core_n * tile_width
mul_num_tiles = math.ceil(total_elements / 256)   # 256 = 16*16

cb_in0_descriptor.total_size = mul_num_tiles * tile_16x16_size
cb_in0_descriptor.format_descriptors[0].tile = tile_16x16_desc
```

For $\text{per\_core\_n} = 8$ tiles of $1 \times 32$, there are $1 \times 8 \times 32 = 256$ elements, which equals 1 tile of $16 \times 16$. The compute is:

$$
\text{out}_{ij} = \text{SiLU}(\text{gate\_proj}_{ij}) \times \text{up\_proj}_{ij} \times s
$$

where $s$ is the normalized expert score from `mul_cb_scalar_src` (CB 21).

### Tensor Shape Trace

```
Input:           [1, K] on sender core (e.g., K = 7168)
  ↓ mcast (sender → 130-core grid)
All cores:       [1, K] replicated
  ↓ sram_matmul (gate routing, fused sigmoid)
Per core:        [1, K] × [K, per_core_N] → [1, per_core_N]
                 (total across cores: [1, 256] routing scores)
  ↓ gather (compute cores → sender)
On sender:       [1, 256] routing scores concatenated
  ↓ deepseek_moe_gate (top-8 selection)
Selected:        8 × (expert_index uint16, expert_score float16)
  ↓ index_mcast + scale_mcast
All cores:       expert_index + expert_scale replicated
  ↓ dram_streaming_matmul (gate_proj, fused SiLU)
Per core:        [1, K] × [K, N_expert/core] → [1, N_expert/core]
                 (triple-buffered: subblock_k × 3 tiles in weight CB)
  ↓ dram_streaming_matmul (up_proj)
Per core:        [1, K] × [K, N_expert/core] → [1, N_expert/core]
  ↓ eltwise_mul (SiLU(gate) × up × scale)
Per core:        [1, N_expert/core] reinterpreted as 16×16 tiles
  ↓ down_proj_gather (compute → sender)
On sender:       [1, K_down] concatenated
  ↓ down_proj_mcast (sender → all cores)
All cores:       [1, K_down] replicated
  ↓ down_proj_matmul (SRAM)
Per core:        [1, K_down] × [K_down, N/core] → [1, N/core]
  ↓ eltwise_add (+ residual shard)
Per core:        [1, N/core] + slice([1, K]) → [1, N/core]
  ↓ reduce_to_one_b1 (optional, cross-device)
Final:           [1, K] on ROOT1 device
```

### ReduceToOne -- Multi-Device Expert Aggregation

When running on a $4 \times 2$ mesh (8 devices), each device processes different experts. Their results must be summed. The `ReduceToOne` protocol uses a 3-round tree reduction:

#### Device Roles

The `get_reduce_device_role` function assigns roles based on mesh coordinates relative to the root:

| Role | Value | Row Criterion | Behavior |
|------|-------|--------------|----------|
| `MESH_LEAF` | 0 | Outer rows (0 or 3) | Sends to adjacent ROOT3/ROOT2 |
| `MESH_ROOT3` | 1 | Inner row opposite to ROOT1 | Receives from LEAF, sends to ROOT1's row |
| `MESH_ROOT2` | 2 | Same row as ROOT1, different column | Receives from ROOT3, sends to ROOT1 |
| `MESH_ROOT1` | 3 | Root coordinate | Receives all, produces final output |

#### Reduction Rounds

```
Round 1: LEAF (rows 0,3) --> ROOT3 (row 1 or 2)
         Semaphore: reduce_sem_round1
Round 2: ROOT3 --> ROOT1's row partner (ROOT2)
         Semaphore: reduce_sem_round2
Round 3: ROOT2 --> ROOT1
         Semaphore: reduce_sem_round3
```

Each round uses fabric connections between adjacent devices. Worker cores package their shard into a packet (96-byte header + payload), send to a fabric core, which forwards to the destination device.

#### Fabric Core Assignment

Each column of worker cores is assigned one fabric core placed at `(bottom_core.x + 1, bottom_core.y)`. Worker-to-fabric communication uses per-slot semaphores from `REDUCE_WORKER_FABRIC_BASE` (indices 12--15). Link assignment distributes traffic: columns in the first half use link 0, columns in the second half use link 1.

---

## 4.3.3 `shared_expert` -- 128-Core Gate/Up + Down Projection

**Source:** `shared_expert/op.py` (979 lines)

The shared expert computes:

$$\text{output} = \left(\text{SiLU}(\text{activation} \times W_{\text{gate}}) \odot (\text{activation} \times W_{\text{up}})\right) \times W_{\text{down}} + \text{bias}$$

Unlike the routed expert which uses DRAM streaming, the shared expert uses SRAM-resident KN-sliced matmul with $64 + 64$ A/B core pairs.

### Composition Tree

```
shared_expert
├── activation_mcast (BRISC: (12,9) → 130-core grid)
│   └── Broadcast [1, K_gate] to all cores
├── gate_matmul (NCRISC+TRISC: 64 A-cores, K-split)
│   └── [1, K_gate] × [K_gate/k_par, 32] → [1, 32] per core
├── up_matmul (NCRISC+TRISC: 64 B-cores, K-split)
│   └── [1, K_gate] × [K_gate/k_par, 32] → [1, 32] per core
├── gate_gather (NCRISC: 64 A-cores → (12,9))
│   └── Collect gate partial sums on sender
├── up_gather (NCRISC: 64 B-cores → (12,9))
│   └── Collect up partial sums on sender
├── gated_reduce (TRISC: on (12,9))
│   │  SiLU(sum(gate_partials)) × sum(up_partials)
│   └── [k_par tiles each] → [1, K_down]
├── down_mcast (BRISC: (12,9) → 130-core grid)
│   └── Broadcast [1, K_down] to all cores
├── residual_mcast (BRISC: (12,9) → 130-core grid)
│   └── Broadcast bias [1, N] to all cores
├── down_proj_matmul (NCRISC+TRISC: 112 cores)
│   └── [1, K_down] × [K_down, N/core] → [1, N/core]
├── residual_add (TRISC: 112 cores)
│   └── matmul_out + shard(bias) → [1, N/core]
└── output_gather (NCRISC: 112 cores → (12,9))
    └── Collect [1, N] on sender
```

### Dual A/B Core Grid Layout

The 128 compute cores are split into two groups on the $13 \times 10$ grid:

```
     Col 0   1   2   3   4   5   6   7   8   9  10  11  12
Row 0  [A ] [A ] [A ] [A ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [B ]
Row 1  [A ] [A ] [A ] [A ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [B ]
Row 2  [A ] [A ] [A ] [A ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [B ]
Row 3  [A ] [A ] [A ] [A ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [B ]
Row 4  [A ] [A ] [A ] [B ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [B ]
Row 5  [A ] [A ] [A ] [B ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [B ]
Row 6  [A ] [A ] [A ] [B ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [B ]
Row 7  [A ] [A ] [A ] [B ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [B ]
Row 8  [A ] [A ] [A ] [B ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [P ]
Row 9  [A ] [A ] [A ] [B ] [B ] [B ] [B ] [A ] [A ] [A ] [B ] [B ] [S ]

Legend:
  A = Gate compute core (64 cores) -- is_gate_compute_core
  B = Up compute core (64 cores)   -- is_up_compute_core
  P = Phantom/idle (col 12, row 8)
  S = Sender/Mcast/Gather core (12, 9) -- is_mcast_sender_core, is_gather_receiver_core
```

**A core assignment rules:**

- Rows 0--3: cols {0, 1, 2, 3, 7, 8, 9} = 7 per row, 28 total
- Rows 4--9: cols {0, 1, 2, 7, 8, 9} = 6 per row, 36 total
- Total: 28 + 36 = 64 A cores

**B core assignment rules:**

- Rows 0--3: cols {4, 5, 6, 10, 11, 12} = 6 per row, 24 total
- Rows 4--7: cols {3, 4, 5, 6, 10, 11, 12} = 7 per row, 28 total
- Row 8: cols {3, 4, 5, 6, 10, 11} = 6 (col 12 = phantom)
- Row 9: cols {3, 4, 5, 6, 10, 11} = 6 (col 12 = sender)
- Total: 24 + 28 + 6 + 6 = 64 B cores

### Stage Pipeline Table

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|------|-------------|--------------|-----------|------|
| 1 | Activation Mcast | BRISC (send) / NCRISC (recv) | `act_mcast_src_cb` (8) | `act_mcast_dst_cb` (9) | $(12,9) \to$ 129 receivers | Sems 0, 8 |
| 2a | Gate Matmul (A cores) | TRISC | `act_mcast_dst_cb` (9), `gu_weights_cb` (13) | `gu_out_cb` (14) | 64 A cores | CB handshake |
| 2b | Up Matmul (B cores) | TRISC | `act_mcast_dst_cb` (9), `gu_weights_cb` (13) | `gu_out_cb` (14) | 64 B cores | CB handshake |
| 3a | A-Gather (gate results) | NCRISC (send) / BRISC (recv) | `gu_out_cb` (14) | `group1_cb` (0) | 64 A $\to$ $(12,9)$ | Sems 4, 6 |
| 3b | B-Gather (up results) | NCRISC (send) / BRISC (recv) | `gu_out_cb` (14) | `group2_cb` (1) | 64 B $\to$ $(12,9)$ | Sems 5, 7 |
| 4 | Gated Reduce | TRISC | `group1_cb` (0), `group2_cb` (1) | `mcast_src_cb` (3) via `intermed_cb` (2) | $(12,9)$ | CB handshake |
| 5 | Mcast1 (down activation) | BRISC (send) / NCRISC (recv) | `mcast_src_cb` (3) | `mcast_dst_cb` (4) | $(12,9) \to$ 129 receivers | Sems 0, 1 |
| 6 | Mcast2 (residual/bias) | BRISC (send) / NCRISC (recv) | `residual_mcast_src_cb` (10) | `residual_mcast_dst_cb` (11) | $(12,9) \to$ 129 receivers | Sems 0, 1 (reused) |
| 7 | Down Proj Matmul | TRISC | `mcast_dst_cb` (4), `matmul_in1_cb` (5) | `matmul_out_cb` (6) | 112 matmul cores | CB handshake |
| 8 | Residual Add | TRISC | `matmul_out_cb` (6), `residual_mcast_dst_cb` (11) | `residual_add_out_cb` (12) | 112 matmul cores | CB handshake |
| 9 | Output Gather | NCRISC (send) / BRISC (recv) | `residual_add_out_cb` (12) | `gather_dst_cb` (7) | 112 $\to$ $(12,9)$ | Sems 2, 3 |

### CB Layout (15 CBs)

| CB | Index | Tile | Core(s) | Role |
|----|-------|------|---------|------|
| `group1_cb` | 0 | Face (16x16) or input | Sender | A-gather dest / reduce group1 |
| `group2_cb` | 1 | Face or input | Sender | B-gather dest / reduce group2 |
| `intermed_cb` | 2 | Face or input | Sender | Reduce intermediate (2 tiles) |
| `mcast_src_cb` | 3 | Face or input | Sender | Reduce output / Mcast1 source |
| `mcast_dst_cb` | 4 | Input tile | All 130 | Mcast1 dest / down matmul in0 |
| `matmul_in1_cb` | 5 | Weight tile | 112 matmul | Down proj weights |
| `matmul_out_cb` | 6 | 1x32 | 112 matmul | Down proj output |
| `gather_dst_cb` | 7 | 1x32 | Sender | Output gather destination |
| `act_mcast_src_cb` | 8 | Input tile | Sender | Activation mcast source |
| `act_mcast_dst_cb` | 9 | Input tile | All 130 | Activation mcast destination |
| `residual_mcast_src_cb` | 10 | 1x32 | Sender | Bias/residual mcast source |
| `residual_mcast_dst_cb` | 11 | 1x32 | All 130 | Bias/residual mcast destination |
| `residual_add_out_cb` | 12 | 1x32 | 112 matmul | Residual add output |
| `gu_weights_cb` | 13 | Weight tile | 128 compute | Gate/Up weights |
| `gu_out_cb` | 14 | Input tile | 128 compute | Gate/Up matmul output (1 tile) |

### Semaphore Map (9 local)

| ID | Name | Purpose |
|----|------|---------|
| 0 | `mcast_data_sender` | Shared: act mcast + mcast1 + mcast2 sender gate (reused across stages) |
| 1 | `mcast_data_receiver` | Mcast1 (down activation) receiver gate |
| 2 | `gather_noc0_receiver` | Output gather NOC0 receiver |
| 3 | `gather_noc1_receiver` | Output gather NOC1 receiver |
| 4 | `ag_receiver` | Gate (A) gather arrival on $(12,9)$ |
| 5 | `bg_receiver` | Up (B) gather arrival on $(12,9)$ |
| 6 | `ag_noc1_receiver` | Gate gather NOC1 |
| 7 | `bg_noc1_receiver` | Up gather NOC1 |
| 8 | `act_mcast_receiver` | Activation mcast receiver gate |

### Role Flags (8 roles, from source code)

| Flag | Grid | Cores |
|------|------|-------|
| `is_gate_compute_core` | 64 A-cores | Gate matmul + SiLU |
| `is_up_compute_core` | 64 B-cores | Up matmul |
| `is_gated_reduce_core` | $(12,9)$ | Gated reduce |
| `is_mcast_sender_core` | $(12,9)$ | All mcast operations |
| `is_mcast_receiver_core` | 129 cores | All mcast receivers |
| `is_matmul_core` | 112 cores | Down proj matmul |
| `is_gather_receiver_core` | $(12,9)$ | Output gather |
| `gather_use_per_core_sender_idx` | 112 cores | Non-rectangular gather indexing |

### Face-View in Gated Reduce

With `k_parallel = 8` and `n_parallel = 8`, each A/B core produces one $1 \times 32$ tile. Eight tiles from the 8 k-parallel cores at the same n-position pack into one $16 \times 16$ face:

$$8 \times (1 \times 32) = 8 \times 32 = 256 = 16 \times 16$$

The reduce kernel operates on single face tiles, halving the number of reduction iterations compared to operating on individual tiles.

### Dummy Tensors for A/B Gather

The A and B gathers write into **dummy tensors** allocated on the sender core $(12,9)$:

```python
ag_dummy_tensor = ttnn.from_torch(
    torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
    ..., memory_config=ag_dummy_mem,
)
```

These tensors back CB 0 (`group1_cb`) and CB 1 (`group2_cb`) on the sender core. They are deallocated immediately after `generic_op` returns to release L1 memory.

### Per-Core K-Offset

Each compute core processes a slice of the K dimension. `PerCoreCompileTimeDescriptor` encodes the K-offset:

```python
per_core_gu_k_offset = PerCoreCompileTimeDescriptor(
    named_compile_time_arg="gu_k_offset",
    core_values=[
        (core, (i // n_parallel) * k_per_core)
        for i, core in enumerate(a_cores_list)
    ] + [
        (core, (i // n_parallel) * k_per_core)
        for i, core in enumerate(b_cores_list)
    ],
    other_value=0,
)
```

### Tensor Shape Trace

```
Activation:      [1, K_gate] on (12,9)
  ↓ activation_mcast
All 130 cores:   [1, K_gate] replicated
  ↓ gate_matmul (64 A-cores, k_parallel × n_parallel layout)
Per A-core:      [1, k_per_core] × [k_per_core, 32] → [1, 32]
  ↓ up_matmul (64 B-cores, same layout)
Per B-core:      [1, k_per_core] × [k_per_core, 32] → [1, 32]
  ↓ gate_gather (64 A-cores → (12,9))
On (12,9):       64 × [1, 32] → [k_parallel, n_parallel × 32]
  ↓ up_gather (64 B-cores → (12,9))
On (12,9):       64 × [1, 32] → [k_parallel, n_parallel × 32]
  ↓ gated_reduce (on (12,9))
On (12,9):       SiLU(reduce_k(gate)) × reduce_k(up) → [1, K_down]
  ↓ down_mcast + residual_mcast
All 130 cores:   [1, K_down] + [1, N] replicated
  ↓ down_proj_matmul (112 cores)
Per core:        [1, K_down] × [K_down, N/112] → [1, N/112]
  ↓ residual_add (112 cores)
Per core:        [1, N/112] + bias_shard → [1, N/112]
  ↓ output_gather (112 → (12,9))
On (12,9):       112 × [1, N/112] → [1, N]
```

---

## 4.3.4 `gated_local_reduce` -- Single-Core Gated Reduction

**Source:** `gated_local_reduce/op.py` (176 lines)

The simplest MoE fused op. Runs on a single core and computes:

$$
\text{output} = \text{SiLU}\!\left(\sum_i g_1^{(i)}\right) \times \sum_j g_2^{(j)}
$$

### Composition Tree

```
gated_local_reduce
├── local_reduce_silu (TRISC: group 1)
│   └── Pairwise add + SiLU: N tiles → 1 tile
├── local_reduce (TRISC: group 2)
│   └── Pairwise add: M tiles → 1 tile
└── eltwise_mul (TRISC)
    └── intermed[0] × intermed[1] → output
```

### CB Layout (4 CBs)

| CB | Index | Role |
|----|-------|------|
| `in0_cb` | 0 | Group 1 input (tensor-backed) |
| `in1_cb` | 1 | Group 2 input (tensor-backed) |
| `intermed_cb` | 2 | Intermediate: 2 tiles (group1 result + group2 result) |
| `out_cb` | 3 | Final output (tensor-backed) |

### CB Lifecycle

```
Phase 1 (reduce group1 + SiLU):
  CB0[group1 input]  CB2[intermed, slot 0]
  NCRISC: setup_sharded_buffer(CB0)
  TRISC: pairwise add tiles, SiLU on result -> CB2 slot 0

Phase 2 (reduce group2):
  CB1[group2 input]  CB2[intermed, slot 1]
  NCRISC: setup_sharded_buffer(CB1)
  TRISC: pairwise add tiles -> CB2 slot 1

Phase 3 (multiply):
  CB2[intermed, both slots]  CB3[output]
  TRISC: CB2[0] * CB2[1] -> CB3
```

### Semaphores

**None.** Single-core operation, no inter-core coordination needed.

### Constraints

- Both group tile counts must be even and $\geq 2$ (pairwise reduction)
- All tensors must share the same dtype
- Uses `HiFi4` math fidelity with `math_approx_mode=True` for SiLU approximation

---

## 4.3.5 `gated_local_reduce_down_proj` -- Gather + Reduce + Matmul

**Source:** `gated_local_reduce_down_proj/op.py` (835 lines)

Fuses input gather from remote source cores, gated local reduce, down projection matmul with residual add, and output gather -- all in one dispatch.

### Composition Tree

```
gated_local_reduce_down_proj
├── input_gather_g1 (NCRISC: g1 source cores → (12,9))
│   └── Pull [1, 32] tiles from 16 source cores (gate partials)
├── input_gather_g2 (NCRISC: g2 source cores → (12,9))
│   └── Pull [1, 32] tiles from 16 source cores (up partials)
├── gated_reduce (TRISC: on (12,9))
│   │  For each K-tile position:
│   │    SiLU(reduce(group1[k])) × reduce(group2[k]) → mcast_src[k]
│   └── [tiles_per_k × k_num_tiles] → [1, K_down]
├── mcast1 (BRISC: (12,9) → 130-core grid)
│   └── Broadcast [1, K_down] reduced activation
├── mcast2 (BRISC: (12,9) → 130-core grid)
│   └── Broadcast residual/bias [1, N]
├── down_proj_matmul (NCRISC+TRISC: 112 cores)
│   └── [1, K_down] × [K_down, N/core] → [1, N/core]
├── residual_add (TRISC: 112 cores)
│   └── matmul_out + shard(residual) → [1, N/core]
└── output_gather (NCRISC: 112 cores → (12,9))
    └── Collect [1, N] on (12,9)
```

### Stage Pipeline Table

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|------|-------------|--------------|-----------|------|
| 1a | Input Gather g1 | NCRISC (send) / BRISC (recv) | `input_src_g1_cb` (8) | `group1_cb` (0) | g1 sources $\to$ $(12,9)$ | Sems 4, 6 |
| 1b | Input Gather g2 | NCRISC (send) / BRISC (recv) | `input_src_g2_cb` (9) | `group2_cb` (1) | g2 sources $\to$ $(12,9)$ | Sems 5, 7 |
| 2 | Gated Reduce | TRISC | `group1_cb` (0), `group2_cb` (1) | `mcast_src_cb` (3) via `intermed_cb` (2) | $(12,9)$ | CB handshake |
| 3 | Mcast1 (down activation) | BRISC (send) / NCRISC (recv) | `mcast_src_cb` (3) | `mcast_dst_cb` (4) | $(12,9) \to$ 129 receivers | Sems 0, 1 |
| 4 | Mcast2 (residual/bias) | BRISC (send) / NCRISC (recv) | `residual_add_mcast_src_cb` (10) | `residual_add_mcast_dst_cb` (11) | $(12,9) \to$ 129 receivers | Sems 0, 8 |
| 5 | Down Proj Matmul | TRISC | `mcast_dst_cb` (4), `matmul_in1_cb` (5) | `matmul_out_cb` (6) | 112 cores | CB handshake |
| 6 | Residual Add | TRISC | `matmul_out_cb` (6), `residual_add_mcast_dst_cb` (11) | `residual_add_out_cb` (12) | 112 cores | CB handshake |
| 7 | Output Gather | NCRISC (send) / BRISC (recv) | `residual_add_out_cb` (12) | `gather_dst_cb` (7) | 112 $\to$ $(12,9)$ | Sems 2, 3 |

### CB Layout (13 CBs)

| CB | Index | Tile | Core(s) | Role |
|----|-------|------|---------|------|
| `group1_cb` | 0 | Face or input | Sender $(12,9)$ | Group 1 gather dest / reduce input |
| `group2_cb` | 1 | Face or input | Sender $(12,9)$ | Group 2 gather dest / reduce input |
| `intermed_cb` | 2 | Face or input | Sender $(12,9)$ | Reduce intermediate (2 tiles) |
| `mcast_src_cb` | 3 | Face or input | Sender $(12,9)$ | Reduce output / Mcast1 source |
| `mcast_dst_cb` | 4 | Input tile | All 130 | Mcast1 dest / matmul in0 |
| `matmul_in1_cb` | 5 | Weight tile | 112 matmul | Down proj weights |
| `matmul_out_cb` | 6 | 1x32 | 112 matmul | Down proj output |
| `gather_dst_cb` | 7 | 1x32 | Sender $(12,9)$ | Output gather dest |
| `input_src_g1_cb` | 8 | Input tile | g1 source cores | Group 1 source data |
| `input_src_g2_cb` | 9 | Input tile | g2 source cores | Group 2 source data |
| `residual_add_mcast_src_cb` | 10 | 1x32 | Sender $(12,9)$ | Residual source |
| `residual_add_mcast_dst_cb` | 11 | 1x32 | All 130 | Residual dest |
| `residual_add_out_cb` | 12 | 1x32 | 112 matmul | Residual add output |

### Semaphore Map (9 semaphores)

| ID | Name | Purpose |
|----|------|---------|
| 0 | `mcast_data_sender` | Mcast1+2 sender gate (reused for both) |
| 1 | `mcast_data_receiver` | Mcast1 receiver gate |
| 2 | `gather_noc0_receiver` | Output gather arrival |
| 3 | `gather_noc1_receiver` | Reserved NOC1 |
| 4 | `ig_g1_receiver` | Input gather group1 arrival |
| 5 | `ig_g2_receiver` | Input gather group2 arrival |
| 6 | `ig_g1_noc1_receiver` | Input gather g1 NOC1 |
| 7 | `ig_g2_noc1_receiver` | Input gather g2 NOC1 |
| 8 | `mcast2_data_receiver` | Mcast2 (bias) receiver gate |

### Role Flags (8 roles, from source code)

| Flag | Grid | Cores |
|------|------|-------|
| `is_input_gather_sender_g1` | g1 source cores | Gate partial sum senders |
| `is_input_gather_sender_g2` | g2 source cores | Up partial sum senders |
| `is_gated_reduce_core` | $(12,9)$ | Gated reduce |
| `is_mcast_sender_core` | $(12,9)$ | All mcast operations |
| `is_mcast_receiver_core` | 129 cores | All mcast receivers |
| `is_matmul_core` | 112 matmul cores | Down proj matmul |
| `is_gather_receiver_core` | $(12,9)$ | Output gather |
| `gather_use_per_core_sender_idx` | 112 matmul cores | Non-rectangular gather indexing |

### Face-View Decision

The face-view optimization is selected based on the source tensor tile dimensions:

```python
if use_face_view is None:
    use_face_view = can_use_face_view(tile_h, tile_w, tiles_per_k, k_num_tiles)
```

When enabled, CBs 0--3 use $16 \times 16$ face tile descriptors. The result is `kernel_k_num_tiles = 1` and `mcast_src_num_pages = 1`, meaning each K-position's data is packed into a single face tile for the mcast source. Without face-view, gather destination sizes are `total_input_tiles = tiles_per_k * k_num_tiles`.

### Input Gather Sender Indexing

Each g1/g2 source core receives a per-core `ig_g1_sender_idx` / `ig_g2_sender_idx` via `PerCoreCompileTimeDescriptor`. The index computation depends on face-view mode:

- **With face-view:** `sender_idx = collection * k_num_tiles + k`
- **Without face-view:** `sender_idx = k * tiles_per_k + collection`

---

## 4.3.6 `down_proj` -- Mcast + Matmul + Add + Gather

**Source:** `down_proj/op.py` (531 lines)

The standalone down projection fused op. Implements the pattern $\text{Mcast1} + \text{Mcast2} + \text{Matmul} + \text{ResidualAdd} + \text{Gather}$ on the standard 130-core grid. This is the prototypical mcast-matmul-gather building block, reused by `shared_expert` and `gated_local_reduce_down_proj` as shared constants.

### Composition Tree

```
down_proj
├── mcast1 (BRISC: (12,9) → 130-core grid)
│   └── Broadcast input [1, K] to all cores
├── mcast2 (BRISC: (12,9) → 130-core grid)
│   └── Broadcast add_input [1, N] to all cores
├── matmul (NCRISC+TRISC: 112 cores)
│   └── [1, K] × [K, N/core] → [1, N/core]
├── residual_add (TRISC: 112 cores)
│   └── matmul_out + shard(add_input) → [1, N/core]
└── gather (NCRISC: 112 cores → (12,9))
    └── Collect [1, N] on (12,9)
```

### Core Grid Diagram

```
     Col 0   1   2   3   4   5   6   7   8   9  10  11  12
Row 0  [D ] [R ] [R ] [D ] [R ] [R ] [R ] [D ] [R ] [D ] [R ] [R ] [P ]
Row 1  [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [P ]
Row 2  [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [P ]
Row 3  [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [P ]
Row 4  [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [P ]
Row 5  [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [P ]
Row 6  [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [P ]
Row 7  [R ] [D ] [R ] [R ] [D ] [R ] [D ] [R ] [R ] [D ] [R ] [R ] [P ]
Row 8  [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [P ]
Row 9  [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [R ] [M ]

Legend:
  R = Matmul core (112 cores) -- mcast recv + matmul + add + gather send
  D = DRAM worker (8 cores at known positions) -- mcast recv only
  P = Phantom (9 cores, col 12 rows 0-8) -- mcast recv only
  M = Mcast sender + Gather receiver (12, 9)
```

**DRAM worker positions (logical):** $(0,0)$, $(0,3)$, $(0,7)$, $(0,9)$, $(7,1)$, $(7,4)$, $(7,6)$, $(7,9)$

**Core count breakdown:**
- $13 \times 10 = 130$ total grid cores
- minus 8 DRAM workers
- minus 9 phantoms (col 12, rows 0--8)
- minus 1 mcast/gather core $(12, 9)$
- = **112 matmul cores**

### Stage Pipeline Table

| Stage | Micro-Op | RISC | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|------|-------------|--------------|-----------|------|
| 1 | Mcast1 (activation) | BRISC (send) / NCRISC (recv) | `mcast_src_cb` (0) | `mcast_dst_cb` (1) | $(12,9) \to$ 129 receivers | Sems 0, 1 |
| 2 | Mcast2 (residual) | BRISC (send) / NCRISC (recv) | `residual_add_mcast_src_cb` (5) | `residual_add_mcast_dst_cb` (6) | $(12,9) \to$ 129 receivers | Sems 0, 4 |
| 3 | Matmul ($W_{\text{down}}$) | TRISC | `mcast_dst_cb` (1), `matmul_in1_cb` (2) | `matmul_out_cb` (3) | 112 cores | CB handshake |
| 4 | Residual Add | TRISC | `matmul_out_cb` (3), `residual_add_mcast_dst_cb` (6) | `residual_add_out_cb` (7) | 112 cores | CB handshake |
| 5 | Output Gather | NCRISC (send) / BRISC (recv) | `residual_add_out_cb` (7) | `gather_dst_cb` (4) | 112 $\to$ $(12,9)$ | Sems 2, 3 |

### CB Layout (8 CBs)

| CB | Index | Grid | Role |
|----|-------|------|------|
| `mcast_src_cb` | 0 | Sender $(12,9)$ | Input (tensor-backed) |
| `mcast_dst_cb` | 1 | All 130 | Mcast dest / matmul in0 |
| `matmul_in1_cb` | 2 | 112 matmul | Weights (tensor-backed) |
| `matmul_out_cb` | 3 | 112 matmul | Matmul output |
| `gather_dst_cb` | 4 | Sender $(12,9)$ | Output (tensor-backed) |
| `residual_add_mcast_src_cb` | 5 | Sender $(12,9)$ | Residual input (tensor-backed) |
| `residual_add_mcast_dst_cb` | 6 | All 130 | Residual mcast dest |
| `residual_add_out_cb` | 7 | 112 matmul | Residual add output |

### Semaphore Map (5 semaphores)

| ID | Name | Purpose | Notes |
|----|------|---------|-------|
| 0 | `mcast_data_sender` | Mcast1+2 sender gate | Reused across stages (sequential execution) |
| 1 | `mcast_data_receiver` | Mcast1 receiver gate | |
| 2 | `gather_noc0_receiver` | Gather arrival | |
| 3 | `gather_noc1_receiver` | Reserved NOC1 | |
| 4 | `mcast2_data_receiver` | Mcast2 receiver gate | |

### Role Flags (5 roles, from source code)

| Flag | Grid | Purpose |
|------|------|---------|
| `is_mcast_sender_core` | $(12,9)$ | Mcast sender + gather receiver |
| `is_mcast_receiver_core` | 129 cores | All mcast receivers |
| `is_matmul_core` | 112 cores | Down proj matmul + residual add |
| `is_gather_receiver_core` | $(12,9)$ | Output gather receiver |
| `gather_use_per_core_sender_idx` | 112 matmul cores | Non-rectangular gather indexing |

### Per-Core Gather Index

Because 112 matmul cores form a non-rectangular grid (DRAM workers create holes), each core needs an explicit gather sender index for computing its write offset on the receiver:

```python
matmul_cores_list = ttnn.corerange_to_cores(matmul_core_grid)
per_core_gather_idx = PerCoreCompileTimeDescriptor(
    named_compile_time_arg="gather_sender_idx",
    core_values=[(core, idx) for idx, core in enumerate(matmul_cores_list)],
    other_value=0,
)
```

### Shared Constants

`DownProj.MCAST_GATHER_CORE = (12, 9)`, `DownProj.MCAST_GRID_X = 13`, `DownProj.MCAST_GRID_Y = 10`, `DownProj.NUM_MATMUL_CORES = 112` are used by `shared_expert` and `gated_local_reduce_down_proj` as well. The `build_matmul_core_grid()` and `build_mcast_receiver_grid()` static methods are the canonical sources for these grid constructions.

### Tensor Shape Trace

```
Input:       [1, K] on (12,9)
  ↓ mcast1 (1 → 130)
All cores:   [1, K] replicated
Add input:   [1, N] on (12,9)
  ↓ mcast2 (1 → 130)
All cores:   [1, N] replicated
  ↓ matmul (112 cores)
Per core:    [1, K] × [K, N/112] → [1, N/112]
  ↓ residual_add (112 cores)
Per core:    [1, N/112] + shard([1, N]) → [1, N/112]
  ↓ gather (112 → (12,9))
Output:      [1, N] on (12,9)
```

---

**Prev:** [Attention Fused Ops](./02_attention_fused_ops.md) | **Next:** [Output Fused Ops](./04_output_fused_ops.md)
