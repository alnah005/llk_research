# 4.4 Output Fused Ops

This section covers the final stage of the DeepSeek V3 B1 inference pipeline: the language-model head projection and token sampling. The `lm_head_sampling` fused op projects the hidden state onto the vocabulary and performs a fused argmax to select the output token -- all in a single kernel dispatch across ~101 cores with mesh-wide reduction for multi-device deployments.

Source: `models/demos/deepseek_v3_b1/fused_ops/lm_head_sampling/op.py` (1217 lines)

---

## 4.4.1 `lm_head_sampling` -- CCL Broadcast + RMSNorm + Mcast + Matmul + Fused Argmax

### Mathematical Operation

$$
\text{scores} = \text{RMSNorm}(x, \gamma) \cdot W_{\text{vocab}}
$$

$$
\text{token\_id} = \arg\max(\text{scores})
$$

where $x \in \mathbb{R}^{1 \times K}$ is the final hidden state ($K = 7168$), $\gamma \in \mathbb{R}^{1 \times K}$ is the RMSNorm weight, and $W_{\text{vocab}} \in \mathbb{R}^{K \times N}$ is the vocabulary projection matrix width-sharded across matmul cores as $[K, N_{\text{per\_core}}]$.

The golden reference in `LMHeadSampling.golden()` implements tie-breaking: when multiple vocabulary entries share the maximum score, the one with the smallest index wins.

### Composition Tree

```
lm_head_sampling
├── ccl_broadcast (NCRISC+BRISC: optional, for multi-device)
│   └── Replicate [1, 7168] across mesh (same as broadcast_rms)
├── rmsnorm (TRISC: on sender core)
│   └── [1, 7168] → [1, 7168] normalized
├── mcast (BRISC: sender core → device grid)
│   └── Broadcast [1, 7168] to all matmul cores
├── matmul (NCRISC+TRISC: vocab projection)
│   └── [1, 7168] × [7168, N_per_core] → [1, N_per_core] per core
├── local_argmax (NCRISC: per-core)
│   │  Each core finds max score and corresponding global index
│   └── [1, N_per_core] → (max_score, max_index)
├── argmax_gather (NCRISC: all matmul cores → final core)
│   │  Collect (score, index) pairs from all cores
│   └── N_cores × (score, index) → single (score, index)
├── argmax_reduce (NCRISC: final core)
│   │  Find global maximum across all gathered pairs
│   └── (max_score, max_index) with tie-breaking on min index
├── mesh_argmax (NCRISC+BRISC: optional, multi-device)
│   │  Two-stage reduction across 4×2 mesh:
│   │  Stage 1: Column reduce (rows → target_row)
│   │  Stage 2: Row reduce (cols → target_col)
│   └── 8 × (score, index) → single (score, index) on final device
└── socket_output (BRISC: optional, D2H or D2D)
    └── Emit sampled token index via socket
```

### Tensor Shape Trace

```
Input:           [1, 7168] on sender core (single device) or sender device (mesh)
  ↓ ccl_broadcast (optional, NCRISC/BRISC)
All devices:     [1, 7168] replicated
  ↓ rmsnorm (TRISC, sender core per device)
Normalized:      [1, 7168]
  ↓ mcast (BRISC, sender → device grid)
All matmul cores: [1, 7168] replicated
  ↓ matmul (vocab projection, LoFi fidelity)
Per core:        [1, 7168] × [7168, N_per_core] → [1, N_per_core]
                 (e.g., N_per_core = 512 → 16 tiles of 1×32)
  ↓ local_argmax (per core)
Per core:        [1, N_per_core] scores + [1, N_per_core] indices → (max_val, max_idx)
                 (16 bytes: 4B score + 4B index + padding)
  ↓ argmax_gather (all cores → final core)
Final core:      N_senders × 16B → buffer of (score, index) pairs
  ↓ argmax_reduce (final core)
Device result:   (global_max_score, global_min_index_at_max)
  ↓ mesh_argmax stage 1 (optional: column reduce)
Target row:      mesh_rows × (score, index) → (score, index)
  ↓ mesh_argmax stage 2 (optional: row reduce)
Final device:    mesh_cols × (score, index) → (score, index)
  ↓ socket_output (optional)
Host/D2D:        Single uint32 token index
```

### Input/Output Tensors

| Tensor | Shape | Sharding | Purpose |
|--------|-------|----------|---------|
| `input_tensor_mesh` | $[1, K]$ | Height-sharded, 1 core | Hidden state from last decoder layer |
| `intermediate_tensor_mesh` | $[1, K]$ | Height-sharded, 1 core | CCL broadcast destination buffer |
| `gamma_tensor` | $[1, K]$ | Sharded, 1 core | RMSNorm weight |
| `vocab_tensor` | $[K, N]$ | Width-sharded, $C$ cores | LM head weights ($N_{\text{per\_core}} = N / C$) |
| `output_tensor` | $[1, N]$ | Width-sharded, $C$ cores | Matmul output scores |
| `indices_tensor` | $[1, N]$ | Width-sharded, $C$ cores | Pre-cached vocabulary indices (uint32) |
| `output_index_tensor` | $[1, 1]$ | 1 core | Final argmax result (uint32) |
| `fabric_scratch_tensor` | per-device | 1 core | Scratch buffer for mesh argmax slot exchange |

### CB Layout (11 CB indices)

| CB ID | Symbol | Backing | Core Range | Purpose |
|-------|--------|---------|------------|---------|
| 0 | `rmsnorm_input_cb` | Tensor (`input_tensor` or `intermediate_tensor`) | Sender core | RMSNorm input; in multi-device mode backed by the CCL intermediate buffer |
| 1 | `mcast_dst_cb` | Intermediate | All mcast grid cores | Mcast destination = matmul `in0`; sized at $\text{num\_tiles\_k} \times \text{tile\_size}$ bytes |
| 2 | `matmul_in1_cb` | Tensor (`vocab_tensor`) | Matmul cores | Vocab weight shards $[K, N_{\text{per\_core}}]$ |
| 3 | `argmax_winner_cb` | Intermediate (16 B) | Argmax cores | Local argmax winner: `{value, index, padding}` packed into 16 bytes |
| 4 | `argmax_gather_cb` | Intermediate ($16 \times S$ B) | Argmax cores | Gather buffer: $S$ winner pages from all senders ($S = \text{argmax\_num\_senders}$) |
| 5 | `argmax_indices_cb` | Tensor (`indices_tensor`) | Argmax cores | Pre-cached global vocabulary indices (uint32) |
| 6 | `argmax_socket_cb` | Intermediate (64 B) | Final core | Socket output staging buffer (conditional on `enable_socket_output`) |
| 7 | `rmsnorm_gamma_cb` | Tensor (`gamma_tensor`) | Sender core | RMSNorm gamma weights |
| 8 | `mcast_src_cb` | Intermediate | Sender core | RMSNorm output, consumed by mcast sender |
| 16 | `matmul_out_cb` | Tensor (`output_tensor`) | Matmul cores | Matmul output scores $[1, N_{\text{per\_core}}]$ |
| 30 | `bcast_pkt_cb` | Tensor (`input_tensor`) | Sender core | CCL broadcast packet buffer (multi-device only, omitted when `skip_ccl=True`) |

### Semaphore Budget

**Local semaphores:**

| ID | Symbol | Core Range | Purpose |
|----|--------|------------|---------|
| 0 | `mcast_data_sender_semaphore` | All mcast grid | Mcast sender synchronization |
| 1 | `mcast_data_receiver_semaphore` | All mcast grid | Mcast receiver ready signal |
| 2 | `argmax_receiver_semaphore` | Argmax cores | Intra-device argmax gather receiver |
| 3 | `argmax_local_ready_semaphore` | Argmax cores | Signals local argmax computation complete |

**Global semaphores for CCL broadcast:**

| Global Semaphore | Purpose |
|-----------------|---------|
| `out_ready_semaphore` | Signals CCL broadcast output is ready |
| `barrier_semaphore` | Ring barrier for broadcast synchronization |
| `secondary_sync_semaphore` | Secondary-axis synchronization |

**Global semaphores for mesh argmax:**

| Global Semaphore | Purpose |
|-----------------|---------|
| `global_semaphore` | Stage 1 mesh argmax: column-wise reduction |
| `global_stage2_semaphore` | Stage 2 mesh argmax: row-wise reduction |

### Compile-Time Core Roles (7 roles)

| Flag | Core Range | Purpose |
|------|-----------|---------|
| `is_input_core` | Sender core (1 core) | CCL receive/send, RMSNorm, mcast sender |
| `is_mcast_receiver_core` | Mcast grid minus sender | Mcast data receivers |
| `is_matmul_core` | Vocab weight shard grid (~100 cores) | LM head matmul |
| `is_rmsnorm_core` | Sender core (1 core) | RMSNorm compute (colocated with `is_input_core`) |
| `is_argmax_core` | Same as matmul grid | Local argmax + gather participation |
| `is_argmax_final_core` | 1 core | Final reduction winner, writes `output_index_tensor` |
| `is_argmax_mesh_sender_core` | Final core (conditional) | Cross-device argmax value sender (multi-device only) |

**Per-core scalar:**

| Descriptor | Values | Purpose |
|-----------|--------|---------|
| `argmax_sender_idx` | $0 \ldots S{-}1$ for each argmax core | Unique sender index for gather slot ordering |

### Compute Configuration

```python
trisc_compute_config = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=fp32_dest_acc_en,
    dst_full_sync_en=fp32_dest_acc_en,
)
```

The use of `MathFidelity.LoFi` is acceptable here because the argmax operation only needs relative ordering of scores, not precise absolute values. The NOC mode is `DM_DYNAMIC_NOC` -- both NCRISC and BRISC use dynamic NOC routing, which is necessary because the same kernel handles CCL broadcast (NCRISC fabric writes), mcast (BRISC hardware multicast), argmax gather (NCRISC unicast receives), and mesh argmax sends (BRISC fabric writes) on overlapping core grids.

### Stage-by-Stage Pipeline Table

| Stage | Micro-Op | Input CB(s) | Output CB(s) | Core Grid | Sync |
|-------|----------|-------------|--------------|-----------|------|
| 1 | CCL Broadcast | External fabric | CB 30 $\to$ CB 0 | Sender core (NCRISC+BRISC) | `out_ready_semaphore`, `barrier_semaphore`, `secondary_sync_semaphore` (global) |
| 2 | RMSNorm | CB 0, CB 7 | CB 8 | Sender core (TRISC) | CB handshake; `epsilon_packed`, `scalar_packed` as runtime args |
| 3 | Mcast | CB 8 | CB 1 | Sender $\to$ all cores (BRISC sender, NCRISC receivers) | Sems 0, 1 |
| 4 | Matmul | CB 1, CB 2 | CB 16 | Matmul cores (NCRISC+TRISC) | CB handshake |
| 5 | Local Argmax | CB 16, CB 5 | CB 3 | Argmax cores (NCRISC) | Per-core scan, no inter-core sync |
| 6 | Argmax Gather | CB 3 | CB 4 on final core | All argmax $\to$ final core (NCRISC) | Sem 2: final waits for $S-1$ pages; Sem 3: senders signal ready |
| 7 | Global Argmax | CB 4 | `output_index_tensor` | Final core (NCRISC) | Reduces all gathered winners, writes uint32 token ID |
| 8a | Mesh Argmax Stage 1 | Scratch buffer | Scratch on target row | Final core (BRISC fabric) | `global_semaphore`: target-row collects from all rows |
| 8b | Mesh Argmax Stage 2 | Scratch buffer | Scratch on target device | Final core on target row (BRISC fabric) | `global_stage2_semaphore`: target device collects from all columns |
| 9 | Socket Output | `output_index_tensor` or CB 6 | External socket | Final core (BRISC) | D2H or D2D socket transfer |

### Stage Details

#### Stage 1: CCL Broadcast

When `skip_ccl=False` (multi-device mode), the sender device broadcasts its $[1, K]$ input to all other devices using the CCL broadcast protocol (see Section 3.3.2). The broadcast uses 14 KB packets (`packet_size_bytes = 14336`) over a ring topology.

The ring routing calculates forward and backward hop counts from the sender's position:

```python
ring_index = row
num_targets_forward = ring_size - ring_index - 1
num_targets_backward = ring_index
```

Secondary-axis relay (`secondary_cluster_axis`) is enabled by default when `mesh_cols > 1`, allowing the sender to relay to the other column. Each non-sender device receives the broadcast into `intermediate_tensor`, which backs CB 0. In single-device mode (`skip_ccl = True`), CB 0 is backed directly by `input_tensor`.

#### Stage 2: RMSNorm

On the sender core, TRISC executes RMSNorm identically to `broadcast_rms` (Section 4.2.1), including the same tile-size autodetection:

```python
is_16x32_tile = (input_shape[1] // 32) % 32 != 0
rms_interpreted_tile = half_16x32_tile if is_16x32_tile else full_32x32_tile
```

For the standard $[1, 7168]$ input, $7168 / 32 = 224$ tiles, $224 \bmod 32 = 0$, so full $32 \times 32$ tiles are used, yielding $7168 / (32 \times 32) = 7$ tiles. The runtime args include pre-packed `epsilon_packed` and `scalar_packed` ($= 1/\sqrt{K}$) as uint32 bit-casts.

#### Stage 3: Mcast

BRISC on the sender core multicasts the RMSNorm output from CB 8 to CB 1 on all cores in the mcast grid. The mcast rectangle is the bounding box of the matmul grid union the sender core. When the sender is inside the mcast rectangle (`is_part_of_receiver_grid`), it self-copies the data to its own CB 1 in addition to multicasting.

#### Stage 4: Matmul

Each matmul core computes $[1, K] \times [K, N_{\text{per\_core}}]$ with `LoFi` math fidelity. The vocab projection does not require high-precision accumulation since the result feeds argmax, not further computation.

#### Stage 5: Local Argmax

Each argmax core (same grid as matmul) scans its $N_{\text{per\_core}}$ output scores in CB 16, comparing against the corresponding global indices in CB 5. The local winner (value + global index) is packed into CB 3 as a 16-byte page.

#### Stage 6: Intra-Device Gather

All argmax cores send their 16-byte winner pages to the final core via NOC unicast into CB 4, which has space for $S \times 16$ bytes. The `argmax_sender_idx` per-core compile-time descriptor determines each core's slot offset within the gather buffer.

The final core waits for `argmax_expected_remote_incs = S - 1` increments on semaphore 2, then scans all $S$ winner pages to find the global maximum. Tie-breaking selects the smallest index, matching the golden reference.

---

## 4.4.2 Mesh Argmax -- Cross-Device Reduction

When operating on a multi-device mesh, each device produces a local argmax winner. These must be reduced to a single global winner. The mesh argmax uses a **two-stage hierarchical reduction** targeting a designated device at `argmax_final_mesh_coord = (target_row, target_col)`.

### Mesh Roles

Each device is assigned roles based on its mesh coordinates relative to the target:

| Role | Condition | Stage 1 | Stage 2 |
|------|-----------|---------|---------|
| Stage 1 sender | `row != target_row` | Send to `(target_row, col)` | -- |
| Stage 1 receiver | `row == target_row` | Receive from all rows in same column | -- |
| Stage 2 sender | `row == target_row and col != target_col` | -- | Send to `(target_row, target_col)` |
| Stage 2 receiver | `row == target_row and col == target_col` | -- | Receive from all columns |

### Scratch Buffer Layout

The `fabric_scratch_tensor` on each device provides a slot array for incoming winner pages:

| Region | Offset | Size | Purpose |
|--------|--------|------|---------|
| Stage 1 slots | `0` | $R \times 16$ bytes | One slot per row ($R = \text{mesh\_rows}$) |
| Stage 2 slots | $R \times 16$ | $C \times 16$ bytes | One slot per column ($C = \text{mesh\_cols}$) |

Each sender writes its 16-byte winner page to a specific slot on the destination device:

- Stage 1 sender at row $r$ writes to slot offset $r \times 16$ on device $(target\_row, col)$
- Stage 2 sender at col $c$ writes to slot offset $(R \times 16) + c \times 16$ on device $(target\_row, target\_col)$

### Fabric Link Selection

Stage 1 senders select their fabric link index based on ring distance to the target to balance traffic across both available links:

```python
def _x_axis_link_idx_for_stage1_sender(sender_row_local):
    linear_distance = abs(sender_row_local - target_row)
    ring_distance = min(linear_distance, mesh_rows - linear_distance)
    max_ring_distance = mesh_rows // 2
    first_half_threshold = (max_ring_distance + 1) // 2
    return 0 if ring_distance <= first_half_threshold else 1
```

Devices closer to the target use link 0; devices farther away use link 1. This distributes fabric traffic across both links, preventing congestion on a single path.

### Global Semaphore Coordination

- `global_semaphore` (address `global_sem_addr`): Stage 1 receivers wait for `mesh_rows - 1` increments, one from each non-target row
- `global_stage2_semaphore` (address `global_stage2_sem_addr`): Stage 2 receiver waits for `mesh_cols - 1` increments, one from each non-target column

After receiving all slots, the target device's final core scans the slot array to determine the global argmax winner across all devices.

### Fabric Connection Setup

Mesh argmax senders require BRISC-side fabric connections. The setup is distinct from the CCL broadcast fabric (which uses NCRISC):

```python
fabric_rt_args = ttnn.setup_fabric_connection(
    src_fabric_node_id=mesh_device.get_fabric_node_id(coord),
    dst_fabric_node_id=mesh_device.get_fabric_node_id(dest_coord),
    link_idx=sender_link_idx,
    program_descriptor=program,
    worker_core=argmax_final_core,
)
```

---

## 4.4.3 Socket Output and Persistent Mode

### Socket Modes

The `lm_head_sampling` op supports three output socket modes:

| Mode | Value | Type | Purpose |
|------|-------|------|---------|
| None | 0 | -- | No socket; result stays in `output_index_tensor` on device |
| D2H | 1 | `ttnn.D2HSocket` | Device-to-host transfer of token ID |
| D2D | 2 | `ttnn.MeshSocket` | Device-to-device transfer for pipeline chaining |

And two input socket modes:

| Mode | Value | Type | Purpose |
|------|-------|------|---------|
| None | 0 | -- | Input from `input_tensor_mesh` or CCL broadcast |
| D2D | 2 | `ttnn.MeshSocket` | Input from another device via socket |

When socket output is active, the argmax final core on the emitting device writes the token ID through CB 6 (`argmax_socket_cb`, 64 bytes) to the socket endpoint. The emitting device is determined by `argmax_final_mesh_coord`. The socket page size is fixed at 64 bytes (minimum for socket transfer).

### Persistent Mode

When `persistent_mode=True`, the kernel runs in a persistent loop, executing the full CCL + RMSNorm + Mcast + Matmul + Argmax pipeline repeatedly without returning to the host. This is critical for minimizing latency in autoregressive decoding.

The persistent loop is gated by:

- `persistent_next_iter_semaphore`: A global semaphore that the host (or a preceding pipeline stage) increments to signal that new input data is ready
- `termination_semaphore`: A global semaphore that, when set to 1, causes the kernel to exit the persistent loop

The argmax final core on the emitting device uses a fabric connection to send a "next iteration ready" signal back to the sender device's input core, enabling a self-sustaining pipeline. Persistent mode is only active on the device that emits socket output (`emit_socket_on_this_device`), ensuring only one device runs the persistent loop.

---

## 4.4.4 Construction and Grid Summary

The `lm_head_sampling` op follows the standard construction pattern (Section 4.1.6): one `UnifiedKernelDescriptor` with 7 role descriptors and 1 per-core descriptor, producing a single program per mesh device. Like `broadcast_rms` and `pre_sdpa`, it builds a `MeshProgramDescriptor` with per-device programs:

```python
mesh_program_descriptor = ttnn.MeshProgramDescriptor()
for row in range(mesh_rows):
    for col in range(mesh_cols):
        coord = ttnn.MeshCoordinate(row, col)
        # ... per-device setup ...
        mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program
result = ttnn.generic_op(io_tensors, mesh_program_descriptor)
```

Each device's program contains device-specific CCL routing, device-specific mesh argmax roles, and identical matmul/argmax compute logic.

**Grid composition for a typical 100-core matmul grid:**

| Role | Cores | Stage |
|------|-------|-------|
| Input/RMSNorm/Mcast sender | 1 | CCL + RMSNorm + Mcast |
| Mcast receivers | ~100 | Mcast receive |
| Matmul | ~100 | Vocab projection |
| Argmax | ~100 | Local argmax + gather |
| Argmax final | 1 | Intra-device reduction + mesh send |

The mcast grid is the bounding box of the matmul grid union the sender core, typically resulting in ~101 cores. The matmul, argmax, and mcast receiver roles overlap on the same physical cores, differentiated by compile-time flags.

### Verification

The op includes extensive runtime validation of unified kernel role mappings:

```python
for group in kernel_result.groups:
    group_cores = ttnn.corerange_to_cores(group.core_range_set)
    if group.compile_time_arg_values.get("is_input_core", 0) == 1:
        input_role_cores.update((c.x, c.y) for c in group_cores)
    # ... same for mcast_receiver, matmul, argmax, argmax_final ...
```

Each role's actual core set is compared against the expected core set, raising `RuntimeError` on mismatch. This catches silent bugs where `UnifiedKernelDescriptor` mis-assigns role flags due to overlapping core ranges.

### End-to-End Data Movement

The `lm_head_sampling` fusion is particularly impactful for decode performance because:

1. The vocab projection produces a large intermediate ($[1, N_{\text{vocab}}]$ across all cores) that never needs to be materialized as a single tensor
2. The argmax reduction happens entirely in L1 using 16-byte winner packets
3. The mesh reduction uses point-to-point fabric writes (16 B per device pair) rather than full tensor exchange
4. In persistent mode, the dispatch overhead is zero for subsequent tokens

For a 128K vocabulary split across 112 matmul cores (~1143 scores per core), the total data moved for argmax is: $112 \times 16\text{ B} = 1792\text{ B}$ intra-device + $8 \times 16\text{ B} = 128\text{ B}$ inter-device -- trivial compared to the matmul compute.

---

**Prev:** [MoE Fused Ops](./03_moe_fused_ops.md) | **Next:** [Chapter 5 -- Multi-head Latent Attention Deep Dive](../ch05_multi_head_latent_attention_deep_dive/index.md)
