# 3.3 Communication and Infrastructure Micro-Ops

The micro-ops in this section span device boundaries. Where compute and data-movement ops operate within a single chip, the communication ops coordinate across the 2--8 devices in a Tenstorrent mesh (and in the case of the pipeline infrastructure, across multiple hosts). They use the **fabric** -- Tenstorrent's inter-device network -- along with global semaphores, D2D sockets, and `MeshProgramDescriptor` to deploy per-device programs across a mesh.

This section covers: **ccl_all_reduce**, **ccl_broadcast**, **reduce_to_one_b1**, **sdpa_reduce_to_all**, **d2d_exchange**, **host_io**, **pipeline_block**, and the deep-dive treatment of **deepseek_moe_gate**.

Source root: `models/demos/deepseek_v3_b1/micro_ops/`

---

## 3.3.1 ccl_all_reduce

**Source:** `micro_ops/ccl_all_reduce/op.py` | **Kernel headers:** `unified_kernels/all_reduce_sender.hpp`, `unified_kernels/all_reduce_receiver.hpp`

### Purpose

Multi-device all-reduce (sum) across a 2-device linear mesh. Each device exchanges its activation shard with its neighbor via fabric, then performs a local element-wise addition. Optionally fuses the next residual-add into the same kernel.

### Computation

$$\text{output} = \sum_{d=0}^{D-1} \text{input}_d + \text{residual} \quad \text{(if fused)}$$

### Data Flow Diagram

```
   Device 0                                        Device 1
   +----------------------------+                  +----------------------------+
   | Sender Core     Receiver   |                  | Sender Core     Receiver   |
   | (second_core)   Core       |                  | (second_core)   Core       |
   |                 (data_core)|                  |                 (data_core)|
   |                            |                  |                            |
   | NCRISC: read    NCRISC:    |                  | NCRISC: read    NCRISC:    |
   | input tensor    wait for   |                  | input tensor    wait for   |
   | -> CB0          fabric     |                  | -> CB0          fabric     |
   |                 data       |                  |                 data       |
   | BRISC: pack     -> CB1     |                  | BRISC: pack     -> CB1     |
   | fabric packet              |                  | fabric packet              |
   | send via        TRISC:     |                  | send via        TRISC:     |
   | routing plane   add(CB1,   |                  | routing plane   add(CB1,   |
   | -> link 0       CB2)       |                  | -> link 1       CB2)       |
   |                 -> CB3     |                  |                 -> CB3     |
   +--------+-------------------+                  +--------+-------------------+
            |           ^                                   |           ^
            +-----------+--- Fabric Link 0/1 ---------------+-----------+
                          (bidirectional exchange)
```

### CB Layout

| CB Index | Name | Backing | Format | Direction |
|----------|------|---------|--------|-----------|
| 0 | `src0_cb` | L1 (not tensor-backed) | bf16, tiny tiles (1x32) | Sender staging |
| 1 | `compute_cb_in1` | Sharded tensor (`intermediate_tensor`) | bf16, standard 32x32 | Remote data for compute |
| 2 | `compute_cb_in2` | Sharded tensor (`input_tensor`) reinterpreted | bf16, standard 32x32 | Local data for compute |
| 3 | `compute_cb_out` | Sharded tensor (`output_tensor`) reinterpreted | bf16, standard 32x32 | Reduction output |
| 4 | `packet_header_cb` | L1 (2 pages) | uint32, 32 bytes/page | Fabric packet headers |
| 5 | `packet_cb` | L1 | bf16 | Packet payload staging |
| 6 | `compute_cb_residual` | Sharded tensor (`residual_tensor`) | bf16, standard 32x32 | Optional fused residual |
| 7 | `compute_cb_temp` | L1 (scratch) | bf16, standard 32x32 | Scratch for fused residual compute |

### Compile-Time Args

**Sender (NCRISC):**

| Arg Name | Type | Description |
|----------|------|-------------|
| `cb0_id`, `num_tiles`, `tensor_page_size` | `uint32_t` | Source CB and sizing |
| `core_noc_x`, `core_noc_y` | `uint32_t` | Physical NOC coordinates of data core |

**Sender (BRISC):**

| Arg Name | Type | Description |
|----------|------|-------------|
| `packet_header_cb_id`, `packet_cb_id` | `uint32_t` | Packet CB indices |
| `payload_size_bytes`, `page_size_bytes` | `uint32_t` | Transfer sizing |
| `dst_num_hops`, `num_connections` | `uint32_t` | Fabric routing |

**Receiver (TRISC):**

| Arg Name | Type | Description |
|----------|------|-------------|
| `cb_in0`, `cb_in1`, `cb_out0` | `uint32_t` | Compute CB indices |
| `cb_residual`, `cb_temp`, `has_residual` | `uint32_t` | Residual fusion |
| `num_tiles` | `uint32_t` | Standard 32x32 tiles to process |

**Per-Core Descriptor:** `is_sender` -- 1 on sender core, 0 on receiver core.

### Runtime Args

| RISC | Core | Args | Description |
|------|------|------|-------------|
| NCRISC | Sender | `[input_tensor_addr]` | L1 address of input shard |
| BRISC | Sender | `[intermediate_addr, semaphore_addr, ...fabric_rt_args]` | Receiver's intermediate buffer + sync semaphore + fabric args |
| NCRISC | Receiver | `[semaphore_addr, ...fabric_rt_args]` | Sender's semaphore + fabric args |

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Sender: reads local tensor into CB0. Receiver: waits for fabric data. |
| BRISC | Sender: packs and sends fabric packets. |
| TRISC | Receiver: adds local (CB2) + remote (CB1) + optional residual (CB6) -> output (CB3). |

### Key Implementation Detail

**Tile format bridge:** Input tensors use tiny $1 \times 32$ tiles (optimized for per-token decode), but the compute reduction operates on standard $32 \times 32$ tiles for efficiency. CBs 2 and 3 are backed by the input/output tensor buffers but with overridden format descriptors specifying $32 \times 32$ tile dimensions. This reinterpretation eliminates a tile-format conversion pass.

**Fabric connection setup:** Each device uses `setup_routing_plane_connection` three times (sender NCRISC, sender BRISC, receiver NCRISC) with opposite link indices based on chip position. Two global semaphores provide symmetric synchronization: device 0 uses semaphore 1 for sending and semaphore 2 for receiving; device 1 uses the reverse.

**Optional fused residual add:** When `residual_tensor_mesh` is provided, TRISC computes $\text{output} = \text{local} + \text{remote} + \text{residual}$, eliminating a separate eltwise add kernel launch. The `has_residual` flag enables dead-code elimination of the residual path when unused.

### Composed Into

- **broadcast_rms** -- When preceded by ccl_all_reduce in the pipeline, the all-reduce output feeds directly into rmsnorm.
- **moe** -- All-reduce of routed expert outputs across devices.

---

## 3.3.2 ccl_broadcast

**Source:** `micro_ops/ccl_broadcast/op.py` | **Kernel header:** `unified_kernels/broadcast.hpp`

### Purpose

Multi-device broadcast from a single sender device to all other devices in the mesh. Supports both single-axis (column) and dual-axis (column + row) broadcast for 2D meshes.

### Computation

$$\text{output}_d = \text{input}_{\text{sender}} \quad \forall \; d \in \text{mesh}$$

### Data Flow Diagram (Dual-Axis Broadcast)

```
   Step 1: Sender broadcasts across secondary axis (row)
   =====================================================
   Sender (rs, cs) ---secondary_axis---> Secondary Sender (rs, 1-cs)

   Step 2: Both senders broadcast along primary axis (column)
   ==========================================================
   Sender (rs, cs)            Secondary Sender (rs, 1-cs)
        |                              |
        | forward (link 0)             | forward (link 0)
        v                              v
   (rs+1, cs)                    (rs+1, 1-cs)
        |                              |
        v                              v
   (rs+2, cs)                    (rs+2, 1-cs)
        |                              |
        v  backward (link 1)           v  backward (link 1)
   (rs-1, cs)                    (rs-1, 1-cs)
```

### CB Layout

| CB Index | Name | Backing | Format | Direction |
|----------|------|---------|--------|-----------|
| 0 | `src0_cb` | Sharded tensor (`input_tensor`) | Inherits | Source (sender) / Receive buffer (receiver) |

### Compile-Time Args

| Arg Name | RISC | Type | Description |
|----------|------|------|-------------|
| `cb0_id` | Both | `uint32_t` | CB index (0) |
| `num_pages_to_read` | Both | `uint32_t` | Pages per transfer |
| `is_sender` | Both | `uint32_t` | 1 on sender device only |
| `tensor0_page_size` | BRISC | `uint32_t` | Page size in bytes |
| `num_targets_forward_direction` | BRISC | `uint32_t` | Fan-out forward |
| `num_targets_backward_direction` | BRISC | `uint32_t` | Fan-out backward |
| `is_secondary_sender` | BRISC | `uint32_t` | Dual-axis secondary sender |
| `start_distance_in_hops_forward/backward` | BRISC | `uint32_t` | Multicast range control |
| `num_iterations` | Both | `uint32_t` | Repeat count (persistent mode) |

### Runtime Args

| RISC | Args | Description |
|------|------|-------------|
| NCRISC | `[output_addr, out_ready_sem_addr, wait_output_sem, reset_global_sem, noc_x, noc_y, ...]` | Output buffer and synchronization |
| NCRISC | `[...fabric_rt_args]` | Fabric connection args per direction |

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Writer on receiver: waits for data, signals readiness. |
| BRISC | Sender: multicasts data via fabric. |
| TRISC | No-op. |

### Key Implementation Detail

**Dual-axis broadcast:** For a 2D mesh, Phase 1 broadcasts across the secondary axis (row) to create a "secondary sender." Phase 2: both the primary and secondary sender broadcast down their respective columns along the primary axis. This converts an $O(R \times C)$ problem into $O(R + C)$ fabric hops.

**Three-semaphore protocol:** `out_ready` (data availability), `barrier` (consumption barrier for multi-iteration), `secondary_sync` (dual-axis handoff).

**Topology-aware distance:** For torus mode (sender at corner row), targets are split evenly between forward and backward directions, minimizing worst-case hop count. For linear mode, all targets go in one direction.

**Single-core per device** (unlike all-reduce which uses 2 cores).

### Composed Into

- **broadcast_rms** -- Broadcasts the all-reduce result to all devices before rmsnorm.

---

## 3.3.3 reduce_to_one_b1

**Source:** `micro_ops/reduce_to_one_b1/op.py` | **Kernel header:** `unified_kernels/reduce_to_one_b1.hpp`

### Purpose

Reduces data from all 8 devices in a $4 \times 2$ mesh to a single root device using a 3-level hierarchical reduction tree. The most complex multi-device communication op.

### Computation

$$\text{output}_{\text{root}} = \sum_{d=0}^{7} \text{input}_d$$

### Reduction Tree Diagram

```
   Level 1 (Leaf -> Adjacent Root)         Level 2 (ROOT3 -> ROOT2)
   ================================         ========================

   Row 0: LEAF
   [0,0]a0 ---> [1,0]a1 (ROOT2)            [2,0]a2 ---> [1,0]a1
   [0,1]b0 ---> [1,1]b1 (ROOT1)            [2,1]b2 ---> [1,1]b1

   Row 3: LEAF
   [3,0]a3 ---> [2,0]a2 (ROOT3)            Level 3 (ROOT2 -> ROOT1)
   [3,1]b3 ---> [2,1]b2 (ROOT3)            ========================
                                            [1,0]a1 ---> [1,1]b1

   ROOT1 final value = a0+a1+a2+a3+b0+b1+b2+b3
```

Device roles: `MESH_LEAF = 0`, `MESH_ROOT3 = 1`, `MESH_ROOT2 = 2`, `MESH_ROOT1 = 3`.

### CB Layout

| CB Index | Name | Backing | Format | Purpose |
|----------|------|---------|--------|---------|
| 0 | `local_cb` | Sharded tensor (`input_tensor`) | bf16, payload-sized pages | Input |
| 1 | `received_cb_r1` | Sharded tensor (`intermediate_r1`) | bf16 | Round 1 receive buffer |
| 2 | `output_cb` | Sharded tensor (`output_tensor`) | bf16 | Final output |
| 3 | `packet_cb` | L1 | bf16, slot-sized | Packet assembly staging |
| 4 | `packet_header_cb` | L1 (1 page) | uint32, 96 bytes | Persistent packet header |
| 5 | `received_cb_r2` | Sharded tensor (`intermediate_r2`) | bf16 | Round 2 receive buffer |
| 6 | `received_cb_r3` | Sharded tensor (`intermediate_r3`) | bf16 | Round 3 receive buffer |
| 7 | `scratch_cb` | L1 | bf16, 32x32 tiles | Compute scratch |

### Compile-Time Args (per RISC, selected)

| RISC | Arg Name | Type | Description |
|------|----------|------|-------------|
| NCRISC | `device_role` | `uint32_t` | 0--3 (LEAF/ROOT3/ROOT2/ROOT1) |
| NCRISC | `num_tiles`, `local_cb`, `received_cb_r1/r2/r3` | `uint32_t` | CB indices and tile counts |
| BRISC | `device_role`, `payload_size_bytes` | `uint32_t` | Role and transfer size |
| BRISC | `dst_fabric_node_chip_id`, `dst_fabric_node_mesh_id` | `uint32_t` | Fabric destination |
| BRISC | `num_workers`, `slot_size_bytes` | `uint32_t` | Worker grid and packet layout |
| TRISC | `device_role`, `num_tiles` | `uint32_t` | Role and compute size |

**Per-Core Descriptor:** `is_fabric_core` -- 1 on fabric helper cores (one per column).

### Runtime Args

| RISC | Core | Args | Description |
|------|------|------|-------------|
| NCRISC | All workers | `[sem_round1_addr, sem_round2_addr, sem_round3_addr]` | Global semaphore addresses |
| BRISC | Per worker | `[fabric_noc_x/y, slot_idx, worker_sem_addr, dst_l1_addr, dst_sem_addr, output_base_addr, shard_idx, socket_config_addr]` | Per-core routing and addressing |
| BRISC | Fabric cores | `[worker_sem_addr_0..N, ...fabric_connection_args]` | Worker semaphores + fabric args |

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Workers: wait for per-round semaphores, receive remote data. |
| BRISC | Workers: write payloads to fabric core staging CB, signal worker-fabric semaphore. Fabric cores: forward assembled packets via fabric. |
| TRISC | Workers: local addition of received data per round. |

### Key Implementation Detail

**Worker-Fabric Core Architecture:** Each device uses worker cores (from input tensor shard grid) + fabric cores (one per column, placed adjacent to the bottom worker). Workers write their payloads to slot positions in the fabric core's `packet_cb`, signal a worker-fabric semaphore, and the fabric core forwards the assembled packet. This column-to-fabric mapping enables multiple simultaneous fabric transfers without worker-to-worker NOC contention.

**D2D_0 aggregation output:** When `enable_d2d0_output=True`, ROOT1 runs a D2D_0 aggregator kernel on a separate core (auto-placed at a non-overlapping grid position). This core collects all 8 worker shards via D2D socket pairs and forwards downstream via a sender socket, enabling zero-copy pipeline handoff.

**Torus support:** When `is_torus=True` and the root is at row 0 or 3, LEAFs at rows 1 and 2 reverse their send direction, enabling shorter hop paths through the torus wrap.

**Four global semaphores:** `sem_round1` (LEAF -> ROOT3/ROOT2), `sem_round2` (ROOT3 -> ROOT2/ROOT1), `sem_round3` (ROOT2 -> ROOT1), `sem_exit` (final output ready).

### Composed Into

- **moe** -- Reduces routed expert outputs from all 8 devices to the root before final accumulation.

---

## 3.3.4 sdpa_reduce_to_all

**Source:** `micro_ops/sdpa_reduce_to_all/op.py` | **Kernel headers:** `unified_kernels/sdpa_reduce_forwarder.hpp`, `unified_kernels/sdpa_reduce_worker.hpp`

### Purpose

Cross-device reduction for Scaled Dot-Product Attention partial results. Reduces the $(L, S, M)$ tuples (attention output, sum-of-exponentials, max) across all devices in a torus/ring topology, producing the correct globally-normalized attention output on every device.

### Computation

Given partial results $(L_i, S_i, M_i)$ from device $i$:

$$M_{\text{new}} = \max(M_1, M_2)$$

$$S_{\text{new}} = S_1 \cdot e^{(M_1 - M_{\text{new}}) \cdot \alpha} + S_2 \cdot e^{(M_2 - M_{\text{new}}) \cdot \alpha}$$

$$L_{\text{new}} = L_1 \cdot e^{(M_1 - M_{\text{new}}) \cdot \alpha} + L_2 \cdot e^{(M_2 - M_{\text{new}}) \cdot \alpha}$$

Final normalization: $\text{output} = L_{\text{final}} / S_{\text{final}}$

### Data Flow Diagram (Two-Round Ring Reduction)

```
   Round 1: Neighbor exchange (distance=1)
   =========================================
   Device 0 <---> Device 1    (Type A: fwd send, bwd receive)
   Device 1 <---> Device 2    (Type B: bwd send, fwd receive)
   Device 2 <---> Device 3    (alternating pattern)
   Device 3 <---> Device 0    (ring wraps)

   After Round 1: each device has reduced(self, neighbor)

   Round 2: Distance-2 exchange
   ============================
   Device 0 <---> Device 2    (exchange Round 1 results)
   Device 1 <---> Device 3

   After Round 2: each device has reduced(all 4 devices)
   Final step: L_out = L_final / S_final (normalization)
```

### Forwarder Core Architecture

```
   Worker cores (shard_cores)           Forwarder cores (2 total)
   +------+------+------+------+       +----------+----------+
   | w0   | w1   | w2   | w3   |       | fwd_0    | fwd_1    |
   +------+------+------+------+       | (link 0) | (link 1) |
   | w4   | w5   | w6   | w7   |       +----------+----------+
   +------+------+------+------+
   Workers split across 2 links (N/2 per link)

   Per-forwarder buffer layout:
   +====================================================+
   | BRISC buffer: [R1 slots][R2 slots] (fwd direction) |
   | NCRISC buffer: [R1 slots][R2 slots] (bwd direction)|
   +====================================================+
```

### CB Layout

| CB Index | Name | Backing | Format | Direction |
|----------|------|---------|--------|-----------|
| 0 | `cb_local_l` | Sharded tensor (`input_l`) | Inherits | Local L (attention output) |
| 1 | `cb_local_ms` | Sharded tensor (`input_ms`) | Inherits | Local M+S packed |
| 2 | `cb_r1_neighbor_l` | Sharded tensor (`r1_recv_tensor`) | Inherits | Round 1 neighbor L |
| 3 | `cb_r1_neighbor_ms` | L1 | Inherits | Round 1 neighbor M+S |
| 4 | `cb_r1_result_l` | L1 | Inherits | Round 1 reduced L |
| 5 | `cb_r1_result_ms` | L1 | Inherits | Round 1 reduced M+S |
| 6 | `cb_r2_neighbor_l` | Sharded tensor (`r2_recv_tensor`) | Inherits | Round 2 neighbor L |
| 7 | `cb_r2_neighbor_ms` | L1 | Inherits | Round 2 neighbor M+S |
| 8 | `cb_l_out` | Sharded tensor (`output_l`) | Inherits | Final output L |
| 9 | `cb_ms_out` | L1 | Inherits | Final output M+S |
| 10 | `cb_packet_slot` | L1 (2 pages) | uint32 | Packet header staging |

### Compile-Time Args (selected)

| RISC | Arg Name | Type | Description |
|------|----------|------|-------------|
| NCRISC | `ms_tile_size_bytes`, `l_chunk_size_bytes`, `num_l_chunks` | `uint32_t` | Chunked transfer parameters |
| NCRISC | `position_enabled`, `per_device_chunk_size` | `uint32_t` | Position-aware validity masking |
| BRISC | `slot_size`, `l_chunk_size_bytes`, `num_l_chunks` | `uint32_t` | Forwarder slot layout |
| TRISC | `scale_fp32` | `uint32_t` | Attention scale as uint32 bit-pattern |
| TRISC | `final_reduction` | `uint32_t` | 1 to enable $L/S$ normalization |
| Forwarder | `fwd_slots_per_round`, `fwd_slot_size`, `fwd_r2_buffer_offset` | `uint32_t` | Buffer geometry |

**Per-Core Descriptor:** `is_worker` -- 1 on shard/compute cores, 0 on forwarder cores.

### Runtime Args

Extensive per-core args for both workers and forwarders, including: receive buffer addresses, fabric node IDs, semaphore addresses, NOC coordinates, slot addresses, and per-direction semaphore IDs. In position-enabled mode, compute cores also receive `[pos_addr, device_idx, r1_neighbor_idx, r2_neighbor_idx, r2_neighbor_r1_neighbor_idx]`.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Workers: receive data from forwarders, manage per-round CB handoffs. Forwarders (backward direction): receive from fabric. |
| BRISC | Workers: write payloads to forwarder slots. Forwarders (forward direction): send to fabric. |
| TRISC | Workers: online softmax reduction of (L, S, M) tuples per round, final L/S normalization. |

### Key Implementation Detail

**Type A/B worker alternation:** Workers are classified as "Type A" (forward-first) or "Type B" (backward-first) based on `(row + worker_idx) % 2`. Round 1: Type A sends forward, Type B sends backward. Round 2: directions swap. This ensures balanced utilization of both fabric links and achieves full all-reduce in exactly 2 rounds on a 4-device ring.

**Chunked L transfer:** The large L tensor is split into `num_l_chunks` (minimum 4) pieces, each fitting within the fabric maximum payload size. M+S metadata always fits in a single slot. Chunking enables pipelining: the forwarder can begin forwarding early L chunks while later chunks are still being produced.

**Position-aware masking:** When `position_enabled=True`, each device's validity is determined by `position_id >= device_idx * per_device_chunk_size`. Invalid devices' contributions are skipped in the reduction, enabling correct incremental KV cache processing.

### Composed Into

- **post_sdpa** -- Cross-device SDPA output reduction after local SDPA computation on each device.

---

## 3.3.5 d2d_exchange

**Source:** `micro_ops/d2d_exchange/op.py` | **Kernel:** `micro_ops/d2d_exchange/kernels/d2d_exchange.cpp`

### Purpose

Bidirectional data exchange between two cores (potentially on different devices) via D2D sockets. Acts as a relay stage in a multi-hop pipeline: receives from upstream, forwards to downstream.

### Computation

$$\text{downstream} \leftarrow \text{upstream}$$

Pure data forwarding with no arithmetic transformation.

### Data Flow Diagram

```
   Upstream Stage                    SocketInterface                    Downstream Stage
   (previous pipeline)               (relay node)                      (next pipeline)

        sender socket                                                   receiver socket
             |                   send_core         recv_core                  ^
             v                   +----------+     +----------+               |
   upstream_socket ---------> |  upstream  | --> | internal | --> downstream_socket
   (receiver on send_core)    |  receiver  |     |  sender  |    (sender on recv_core)
                              +----------+     +----------+

   If cross-device: fabric connections set up automatically
   based on comparing fabric_node_ids of src and dst
```

### CB Layout

| CB Index | Name | Backing | Format | Purpose |
|----------|------|---------|--------|---------|
| configurable | `packet_header_cb` | L1 (3 pages when fabric active) | uint32, `packet_header_size_bytes` | Fabric headers (only when cross-device) |

### Compile-Time Args

| Arg (positional) | Type | Description |
|------------------|------|-------------|
| 0 | `uint32_t` | Downstream socket config address |
| 1 | `uint32_t` | Upstream socket config address |
| 2 | `uint32_t` | Termination semaphore address |
| 3 | `uint32_t` | Page size |
| 4--6 | `uint32_t` | Fabric packet sizing |
| 7 | `uint32_t` | `packet_header_cb_index` |
| 8 | `bool` | `use_fabric_on_receiver` |
| 9 | `bool` | `use_fabric_on_sender` |

### Runtime Args

Fabric connection args appended for each link (2 forward + 1 backward) when fabric transfers are needed.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Receives from upstream socket, manages fabric backward direction. |
| BRISC | Sends to downstream socket, manages fabric forward direction. |
| TRISC | No-op. |

### Key Implementation Detail

**Persistent kernel with termination:** The `d2d_exchange` kernel runs in an infinite loop, polling the termination semaphore during each socket wait. When the host sets the semaphore to 1, the kernel exits within ~1000 device cycles.

**Same-device optimization:** When sender and receiver cores are on the same device, both kernels are combined into a single `ProgramDescriptor` with merged CBs (deduped by `buffer_index`). Cross-device transfers automatically use fabric connections with 2 forward links and 1 backward link.

**Multi-process support:** When `sender_mesh` and `receiver_mesh` are on different hosts, the `MeshSocket` API is used instead of `create_socket_pair`, enabling transparent cross-host communication.

### Composed Into

- **pipeline_block** -- Entry and exit socket interfaces for each pipeline stage.

---

## 3.3.6 host_io

**Source:** `micro_ops/host_io/op.py` | **Kernels:** `h2d_receiver.cpp`, `fused_h2d_receiver_embedding.cpp`, `d2h_sender.cpp`

### Purpose

Bidirectional host-device communication: H2D (Host-to-Device) receives tokens from the host and optionally performs an embedding lookup; D2H (Device-to-Host) sends output tokens back to the host.

### Computation

- H2D with embedding: $\text{output} = \text{Embedding}[\text{token\_id}]$
- D2H: $\text{host\_output} = \text{device\_output}$ (identity transfer)

### Data Flow Diagrams

**H2D Receiver:**
```
   Host CPU                         Device
   +--------+                       +---------------------------+
   | Token  | -- PCIe push -->      | H2D Socket (L1 FIFO)     |
   | tensor |   (HOST_PUSH mode)    |         |                 |
   +--------+                       |         v                 |
                                    | H2D Receiver Kernel       |
                                    |   if has_embedding:       |
                                    |     read token_id         |
                                    |     DRAM read embedding   |
                                    |     row -> CB             |
                                    |     forward embedding via |
                                    |     downstream socket     |
                                    |   else:                   |
                                    |     forward raw data      |
                                    +---------------------------+
```

**D2H Sender:**
```
   Device                           Host CPU
   +---------------------------+    +--------+
   | D2H Sender Kernel         |    | Output |
   |   wait for upstream data  |    | tensor |
   |   via upstream socket     |    +---^----+
   |   copy to D2H socket FIFO| -------+
   +---------------------------+  PCIe read
```

### CB Layout

| CB Index | Name | Backing | Format | Purpose |
|----------|------|---------|--------|---------|
| 0 (loopback) | `intermed_cb` | L1 | bf16 or uint32 | H2D-to-D2H loopback buffer |
| 2 (configurable) | `embedding_cb` | L1 | bf16, `embedding_page_size` | DRAM embedding read buffer |
| 1 (configurable) | `fabric_packet_header_cb` | L1 (3 pages) | uint32 | Fabric headers (cross-device) |

### Compile-Time Args

**H2D kernel:**

| Arg (positional) | Type | Description |
|------------------|------|-------------|
| 0 | `uint32_t` | H2D socket config address |
| 1 | `uint32_t` | Termination semaphore address |
| 2 | `uint32_t` | H2D page size (64 bytes for tokens) |
| 3 | `bool` | Device-pull mode flag |
| 4 | `bool` | Loopback mode |
| 5 | `uint32_t` | Downstream socket config addr (or CB index in loopback) |
| 11--13 | `uint32_t` | Embedding CB index, page size, buffer address (if fused) |
| 14+ | `uint32_t[]` | `TensorAccessorArgs` for DRAM embedding reads |

**D2H kernel:**

| Arg (positional) | Type | Description |
|------------------|------|-------------|
| 0 | `uint32_t` | D2H socket config address |
| 1 | `uint32_t` | Termination semaphore address |
| 2 | `uint32_t` | D2H page size |
| 3 | `bool` | Loopback mode |
| 4 | `uint32_t` | Upstream socket config addr (or CB index in loopback) |

### Runtime Args

Fabric connection args for H2D (2 forward links) and D2H (1 backward link) when cross-device forwarding is needed.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | D2H sender: reads from upstream socket, pushes to D2H. |
| BRISC | H2D receiver: reads from H2D socket, performs embedding lookup if configured, forwards downstream. |
| TRISC | No-op. |

### Key Implementation Detail

**Fused embedding:** The `fused_h2d_receiver_embedding.cpp` kernel reads a 64-byte token packet from the H2D socket, extracts the token ID, performs a DRAM embedding table lookup using `TensorAccessorArgs`, reads the embedding row into an L1 CB, and forwards the embedding (not the raw token ID) downstream. This fusion eliminates a round-trip through the host for embedding lookup, reducing latency by one PCIe transfer.

**Host-push mode:** H2D is configured with `H2DMode.HOST_PUSH` (faster for 64-byte token packets).

**Termination protocol:** Both kernels use polling loops with semaphore checks. `terminate()` calls `h2d_socket.barrier()` and `d2h_socket.barrier()` before setting the termination semaphore, ensuring all in-flight data is flushed.

### Composed Into

- **pipeline_block** -- Stage 0 manages HostInterface for token ingestion and output delivery.

---

## 3.3.7 pipeline_block

**Source:** `micro_ops/pipeline_block/op.py`

### Purpose

Encapsulates one stage of a multi-host inference pipeline. Manages the setup and lifecycle of D2D socket interfaces, host I/O (on the first stage), and data forwarding between pipeline stages across physical hosts.

### Computation

Not applicable -- `PipelineBlock` is an infrastructure orchestrator, not a compute op.

### Architecture

```
   Host
    |
    v
   [Stage 0: mesh_id=0]
   HostInterface (H2D + embedding + D2H)
   exit_socket_interface -----> [Stage 1: mesh_id=1]
                                entry_socket_interface
                                exit_socket_interface -----> [Stage 2: mesh_id=2]
                                                             ...
                                                             exit_socket_interface --+
                                                                                     |
   [Stage 0: entry_socket_interface] <----------------------------------------------+
   (loopback: last stage feeds back to first stage's D2H path)
```

### Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `upstream_d2d_socket_fifo_size` | `int` | FIFO depth for entry D2D socket |
| `downstream_d2d_socket_fifo_size` | `int` | FIFO depth for exit D2D socket |
| `upstream_d2d_socket_page_size` | `int` | Page size for entry transfers |
| `downstream_d2d_socket_page_size` | `int` | Must equal embedding size on stage 0 |
| `h2d_socket_fifo_size` | `int` | Host-to-device FIFO (stage 0 only) |
| `d2h_socket_fifo_size` | `int` | Device-to-host FIFO (stage 0 only) |
| `embedding_tensor` | `ttnn.Tensor` | DRAM embedding table (stage 0 only) |

### Lifecycle

```python
pipeline_block.run()         # Launch all socket interfaces (non-blocking persistent kernels)
pipeline_block.write_token(token_tensor)  # Stage 0 only: push token to H2D
pipeline_block.read_output(output_tensor) # Stage 0 only: pull from D2H
pipeline_block.terminate()   # Distributed barrier + semaphore-based shutdown
```

### Key Implementation Detail

**Pipeline topology discovery:** The constructor calls `generate_blitz_decode_pipeline(mesh_device)` to obtain a configuration array mapping physical ASIC locations to logical mesh coordinates. Each entry contains `entry_node_coord` and `exit_node_coord`. The pipeline forms a ring: the last stage's exit connects back to stage 0's entry via the loopback `entry_socket_interface`.

**Multi-process termination:** `terminate()` calls `ttnn.distributed_context_barrier()` before signaling termination semaphores, ensuring all processes have completed outstanding requests. Critical for multi-host deployments where stages run on different physical machines.

### Composed Into

- Top-level pipeline orchestration -- not consumed by any fused op, but wires together all fused ops into the full inference pipeline.

---

## 3.3.8 deepseek_moe_gate -- Deep Dive

**Source:** `micro_ops/deepseek_moe_gate/op.py` | **Kernel header:** `unified_kernels/deepseek_moe_gate.hpp`

`DeepseekMoeGateSingleCore` implements the hierarchical top-8 expert selection algorithm that is unique to DeepSeek V3's MoE architecture. It operates on a single $16 \times 16$ face of data -- 256 values representing scores for 256 experts organized as 16 groups of 16.

### The Hierarchical Selection Algorithm

See [Section 1.1.2](../ch01_architecture_overview_and_model_mapping/01_deepseek_v3_architecture_on_tenstorrent.md) for the model-level view of this gating algorithm. The hardware implementation uses a **two-level hierarchical selection** on a single $16 \times 16$ face:

**Level 1 -- Intra-Group Top-2:**
For each of the 16 groups (rows of the $16 \times 16$ matrix):
1. Apply sigmoid to scores (optional, controlled by `enable_sigmoid`).
2. Add bias: $\text{bias\_scores} = \text{scores} + \text{bias}$.
3. Sort bias_scores within the group (descending).
4. Take the top-2 bias scores and their corresponding **original** (pre-bias) scores and **global** indices.

**Level 2 -- Inter-Group Top-4:**
5. Sum each group's top-2 bias scores: $\text{top2\_sum}[g] = \text{bias}_{g,0} + \text{bias}_{g,1}$.
6. Sort the 16 group sums (descending).
7. Take the top-4 groups.
8. Flatten their top-2 entries: $4 \times 2 = 8$ candidates.

**Level 3 -- Final Top-8:**
9. From the 8 candidates, take the top-8 by bias score (trivially all 8, but the sort ensures descending order).
10. Map back to original scores (not bias scores) for the selected 8 experts.

**Normalization:**
11. Sum the 8 selected original scores: $D = \sum_{i=0}^{7} s_i + \epsilon$.
12. Normalize: $\hat{s}_i = \frac{s_i}{D} \times \text{scaling\_factor}$.

Default: $\text{scaling\_factor} = 2.5$, $\epsilon = 10^{-20}$.

### Data Flow Diagram

```
   CB 0 (input):          CB 1 (bias):          CB 3 (input_indices):
   [16,16] scores         [16,16] bias          [16,16] indices
   bf16                   bf16                  bf16
        |                      |                      |
        v                      v                      v
                  TRISC (compute):
                  1. Optional sigmoid on scores
                  2. Add bias for routing decisions
                  3. Per-group sort (16 groups of 16)
                  4. Top-2 sum ranking across groups
                  5. Top-4 groups -> 8 candidates
                  6. Final top-8, normalize, scale
                  (all data fits in a single 16x16 face)
                         |                |
                         v                v
               CB 2 (output):    CB 4 (output_indices):
               [1,16] scores     [1,16] indices
               (only first 8     (only first 8
                values used)      values used)
```

### CB Layout

| CB | Index | Role | Tile Shape | Backed by |
|----|-------|------|------------|-----------|
| input | 0 | Router logits | 16x16 | Sharded tensor |
| bias | 1 | Expert biases | 16x16 | Sharded tensor |
| output | 2 | Normalized top-8 scores | 1x16 | Sharded tensor |
| input_indices | 3 | Pre-computed expert indices | 16x16 | Sharded tensor |
| output_indices | 4 | Top-8 expert indices | 1x16 | Sharded tensor |

### Compile-Time Args

| RISC | Arg Name | Type | Description |
|------|----------|------|-------------|
| NCRISC | `moe_gate_input_cb`, `moe_gate_bias_cb`, `moe_gate_input_indices_cb` | `uint32_t` | Input CB indices (0, 1, 3) |
| BRISC | `moe_gate_output_cb`, `moe_gate_output_indices_cb` | `uint32_t` | Output CB indices (2, 4) |
| TRISC | All input and output CB indices | `uint32_t` | Full set for compute |
| TRISC | `moe_gate_eps` | `uint32_t` | Epsilon as uint32 bit-pattern ($10^{-20}$) |
| TRISC | `moe_gate_scaling_factor` | `uint32_t` | Scaling factor as uint32 bit-pattern (2.5) |
| TRISC | `moe_gate_enable_sigmoid` | `uint32_t` | 1 to apply sigmoid to input |

### Runtime Args

None.

### RISC Roles

| Processor | Role |
|-----------|------|
| NCRISC | Signals input, bias, and input_indices CBs ready. |
| BRISC | Waits for output and output_indices CBs. |
| TRISC | All gate computation: sigmoid, sort, hierarchical select, normalize. Uses custom LLK functions `deepseek_moe_gate_init` and `deepseek_moe_gate`. |

### Key Implementation Detail

**Single-face compute:** The entire 256-expert gating computation operates on a single $16 \times 16$ face of data. The hierarchical selection (per-group top-2, top-4 groups, global top-8) is implemented entirely in register-level operations in TRISC, avoiding any L1 scratch buffers for the sort. Input and output use different tile sizes ($16 \times 16$ vs $1 \times 16$), reflecting the 256-to-8 reduction ratio.

**Kernel compute sequence:**
1. Copy pre-transposed indices into DST register 1 via `copy_tile`.
2. Call `deepseek_moe_gate_init<enable_sigmoid>` and `deepseek_moe_gate<enable_sigmoid>` -- custom LLK functions that perform the entire hierarchical sort-and-reduce on a 16x16 face in hardware.
3. Pack DST register 0 (scores) to `output_cb` and DST register 1 (indices) to `output_indices_cb`.

**Float-to-uint32 packing:** The `float_to_uint32` utility packs `epsilon` and `scaling_factor` as raw IEEE 754 bit patterns into uint32 compile-time args, which the kernel reinterprets as floats.

### Compute Config

- Math fidelity: `HiFi4` (makes no practical difference since the operation is dominated by comparisons and reductions rather than multiply-accumulate)
- FP32 accumulation: disabled

### Core Grid

From `input_tensor.memory_config().shard_spec.grid`. All input tensors must share the same shard spec; output tensors must share the same (different) shard spec. The grid size equals the batch size.

### Composed Into

- **moe** -- Gate computation stage: after rmsnorm and mcast of the input activation, the gate matmul produces router logits which feed into deepseek_moe_gate for expert selection. The output (top-8 indices and weights) drives the expert routing for the entire MoE layer.

---

## Communication Summary

| Micro-Op | Devices | Topology | Synchronization | Single-Device Equivalent |
|----------|---------|----------|-----------------|--------------------------|
| `ccl_all_reduce` | 2 | Linear | 2 global semaphores | `gather` + `mcast` |
| `ccl_broadcast` | $N$ | Linear/Torus | 3 global semaphores | `mcast` |
| `reduce_to_one_b1` | 8 (4x2) | Tree | 4 global semaphores | Hierarchical `gather` |
| `sdpa_reduce_to_all` | 4+ | Ring/Torus | 4 local + 2 global | `gather_reduce` (ring) |
| `d2d_exchange` | 2 | Point-to-point | 1 global semaphore | Direct CB aliasing |
| `host_io` | 1 + host | PCIe | 1 global semaphore | -- |
| `pipeline_block` | multi-host | Pipeline | Distributed barrier | -- |
| `deepseek_moe_gate` | 1 | Single-core | None | -- |

All multi-device ops follow the `MeshProgramDescriptor` and fabric connection patterns described in [Section 2.1](../ch02_the_unified_kernel_system/01_unified_kernel_descriptor.md); `d2d_exchange`, `host_io`, and `pipeline_block` are persistent kernels terminated via global semaphore (see their individual sections above).

---

**Next:** [Chapter 4 — The Fused-Op Library](../ch04_the_fused_op_library/index.md)
