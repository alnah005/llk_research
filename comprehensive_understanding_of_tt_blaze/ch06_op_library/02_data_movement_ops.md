# 06.02 -- Data Movement Ops

[<-- Overview](01_op_library_overview.md) | [Next: Compute Ops -->](03_compute_ops.md)

Data movement ops transfer data between cores, between devices, and between L1 and DRAM. They are the plumbing that connects compute ops into end-to-end pipelines. Most are MicroOps with dedicated C++ kernels that orchestrate BRISC and NCRISC NOC transactions; a few (EmbedMcast, BroadcastRMSNorm, BroadcastRMSNormMcast) are FusedOps that compose simpler data-movement primitives.

---

## Intra-Device Data Movement

### Mcast

- **Class**: `Mcast` | **Type**: MicroOp
- **File**: `blaze/ops/mcast/op.py`
- **Purpose**: Multicast data from the sender core to all cores in the grid. This is the primary fan-out primitive -- every fused kernel that needs to broadcast activations, indices, or scores uses Mcast.
- **Inputs**: `src` (CBHandle or ttnn.Tensor on sender core)
- **Outputs**: `dst` (scratch CBHandle on `f.all_cores`)
- **Key kwargs**:
  - `prefix` -- CT arg namespace (enables multiple Mcasts per FusedProgram)
  - `sender_sem` -- optional pre-allocated sender semaphore
  - `dst_num_pages` -- override destination page count
  - `dst_tile_info` -- override destination tile format (e.g., face-view to full tiles)
- **Mechanism**: BRISC on the sender core performs a NOC multicast write; NCRISC on receiver cores signals readiness via semaphores. The dst CB is allocated with `balanced=False` to handle receivers that are not consumers.
- **Usage**: Appears in nearly every FusedOp (MoE mcast activations, MLA mcast after RMSNorm, attention mcast Q, etc.)

### Gather

- **Class**: `Gather` | **Type**: MicroOp
- **File**: `blaze/ops/gather/op.py`
- **Purpose**: Gather data from distributed cores to a single receiver (typically the sender core). The inverse of Mcast -- collects partial results from parallel compute cores into one location.
- **Inputs**: `src` (CBHandle on scattered cores)
- **Outputs**: `dst` (CBHandle or tensor on receiver core)
- **Key kwargs**:
  - `prefix` -- CT arg namespace
  - `row_major` -- core ordering for sender indexing (default True)
  - `dst_num_pages` -- override (default: `num_senders * src_num_pages`)
  - `receiver_core` -- optional override for gather destination (default: `f.sender_grid`)
  - `dst_cb`, `dst_tile`, `dst_page_size`, `dst_data_format` -- explicit destination CB control
- **Mechanism**: Each sender writes its data to the receiver via NOC unicast. NCRISC on senders initiates the transfer; BRISC on the receiver counts arrivals via two semaphores (noc0/noc1). Per-core `sender_idx` CT args establish the write offset.
- **Usage**: PostSdpa gather, MoE gate gather, SharedExpert gather, Q branch gather

### GatherReduce

- **Class**: `GatherReduce` | **Type**: MicroOp
- **File**: `blaze/ops/gather_reduce/op.py`
- **Purpose**: Dual-destination gather fused with element-wise addition. Senders are split into two halves; each half fills one region of a gather buffer on the receiver. TRISC reduces pairwise: `out[i] = half0[i] + half1[i]`.
- **Inputs**: `src` (CBHandle on scattered cores)
- **Outputs**: `output` (CBHandle on receiver)
- **Key kwargs**:
  - `half_num_cores` -- size of each half
  - `num_tiles` -- tiles per sender
  - `dst_tile_info` -- optional tile reinterpretation
  - `half0_cores` -- explicit first-half core list
- **Usage**: MLA PreSDPA q_a_proj reduction (K-sliced matmul with `k_half_split=True`)

### Scatter

- **Class**: `Scatter` | **Type**: MicroOp
- **File**: `blaze/ops/scatter/op.py`
- **Purpose**: Distribute data from source core(s) to multiple destination cores. The inverse of Gather -- each source writes a configurable number of tiles to each of its assigned destinations via NOC writes.
- **Inputs**: `src` (CBHandle or ttnn.Tensor on source cores)
- **Outputs**: `dst` (CBHandle on destination cores)
- **Key kwargs**:
  - `dest_mapping` -- `{CoreCoord: [CoreCoord, ...]}` explicit source-to-dest map
  - `tiles_per_dest` -- tiles each destination receives
- **Mechanism**: Source data must be contiguous-per-destination in CB order. BRISC on source cores performs NOC writes to destination L1 addresses. NCRISC on destinations waits for arrival semaphore then does `setup_sharded_buffer`.
- **Usage**: PostSdpa scatter (distributing SDPA output to post-grid cores)

### ScatterRaw

- **Class**: `ScatterRaw` | **Type**: MicroOp
- **File**: `blaze/ops/scatter_raw/op.py`
- **Purpose**: Distribute rows from a source core to destination cores with per-head replication. Designed for the D3-to-D4+D5 bridge in attention: source is untilized SDPA output on block0, destinations are post-grid cores.
- **Inputs**: `src_cb` (CBHandle on source core)
- **Outputs**: `dst` (CBHandle on dest cores)
- **Key kwargs**:
  - `dest_mapping` -- source-to-dest map
  - `num_tile_cols` -- tiles per row (v_dim / 32)
  - `num_heads` -- number of heads in source data
  - `cph` -- cores per head (replication factor)
  - `broadcast_mode` -- when True, every dest gets the full payload (replaces Mcast for non-all-cores broadcast)
  - `skip_src_init` -- skip `setup_sharded_buffer` when upstream already pushed data
- **Usage**: GLM 5.1 post-DSA scatter, attention head distribution

### Copy

- **Class**: `Copy` | **Type**: MicroOp
- **File**: `blaze/ops/copy/op.py`
- **Purpose**: Generic L1 data movement via NOC APIs. Generalizes CbFlush and TileCopy into a single op with configurable behavior: optional CB sync, local or remote copy, BRISC or NCRISC processor selection.
- **Inputs**: `src` (CBHandle or ttnn.Tensor)
- **Outputs**: `dst` (CBHandle for output tensor)
- **Key kwargs**:
  - `wait_src` -- `cb_wait_front` before reading (CbFlush equivalent)
  - `pop_src` / `push_dst` -- CB sync control
  - `processor` -- `Risc.NCRISC` or `Risc.BRISC`
  - `dest_core` -- for remote copy to another core
  - `src_addr` / `dst_addr` / `num_bytes` -- raw L1 address mode
- **Usage**: General-purpose data copy, CbFlush replacement, remote core copy

### TileCopy

- **Class**: `TileCopy` | **Type**: MicroOp
- **File**: `blaze/ops/tile_copy/op.py`
- **Purpose**: Copy tiles from a sharded source to an output tensor on NCRISC. Simpler than Copy -- no wait/pop options, no remote copy, purely local NCRISC-driven tile copy.
- **Inputs**: `src` (CBHandle or ttnn.Tensor)
- **Outputs**: `dst` (CBHandle for output tensor)
- **Key kwargs**:
  - `address_offset` / `total_size` / `page_size` -- for OverlappedView access (sub-tensor reads)
  - `core_ranges`, `data_format`, `tile` -- override source interpretation
- **Usage**: Weight tensor sub-view reads, tensor-to-CB initialization

### CbFlush

- **Class**: `CbFlush` | **Type**: MicroOp
- **File**: `blaze/ops/cb_flush/op.py`
- **Purpose**: Wait for a TRISC-written scratch CB, then copy to a tensor-backed output CB. Unlike TileCopy, CbFlush does `cb_wait_front` on the source before reading -- required when the source CB was filled by TRISC compute.
- **Inputs**: `src` (CBHandle, TRISC-written scratch)
- **Outputs**: `dst` (tensor-backed CBHandle)
- **Usage**: Flushing MoE gate scores to host-readable tensors

### Shard2CB

- **Class**: `Shard2CB` | **Type**: MicroOp
- **File**: `blaze/ops/shard2cb/op.py`
- **Purpose**: Fetch pages from a (potentially remote) sharded tensor into a local scratch CB via TensorAccessor NOC reads. Each active core reads `num_pages` pages at a per-core page offset. Bridges between Gather (collects to 1 receiver) and per-core consumption (needs data locally on each compute core).
- **Inputs**: `src_tensor` (sharded ttnn.Tensor)
- **Outputs**: `dst_cb` (pre-allocated scratch CBHandle)
- **Key kwargs**:
  - `num_pages` -- pages per core
  - `page_offsets` -- `{(x,y): start_page_index}` per-core offset
  - `active_grid` -- CoreRangeSet of active cores
- **Usage**: GLM Q branch absorption (distributing Q_nope from gather to per-head compute cores)

---

## Embedding

### Embedding

- **Class**: `Embedding` | **Type**: MicroOp
- **File**: `blaze/ops/embedding/op.py`
- **Purpose**: Token ID to DRAM weight row lookup. NCRISC reads a uint32 token ID from the input CB, uses it as a page index to read the corresponding embedding row from a DRAM-interleaved weight tensor. No TRISC compute.
- **Inputs**: `input_handle` (CBHandle or tensor with token IDs)
- **Outputs**: Scratch CBHandle with embedding row (1x32 storage tiles, byte-identical to RM)
- **Key kwargs**:
  - `weight_buffer_address` -- DRAM buffer address for embedding weights
  - `weight_page_size` -- bytes per embedding row
  - `cores` -- CoreRangeSet where embedding runs
- **Usage**: First op in inference pipelines, always on the sender core

### EmbedMcast

- **Class**: `EmbedMcast` | **Type**: FusedOp
- **File**: `blaze/ops/embed_mcast/op.py`
- **Purpose**: Embedding lookup on the sender core followed by Mcast to all cores. The canonical entry point for token embedding in decode.
- **Composition**: `Embedding.emit()` -> `Mcast.emit()`
- **Inputs**: `token_ids` (CBHandle or tensor)
- **Outputs**: CBHandle on all cores
- **Key kwargs**: `weight_buffer_address`, `weight_page_size`

---

## Inter-Device Communication (CCL)

### AllReduce

- **Class**: `AllReduce` | **Type**: MicroOp
- **File**: `blaze/ops/all_reduce/op.py`
- **Purpose**: Each device exchanges data with its neighbor and performs a local sum reduction. Supports optional residual add (fuses `x + reduced_output` for skip connections). Uses the TT fabric for inter-device transport.
- **Inputs**: `input` (tensor or CBHandle), `intermediate` (tensor or CBHandle), `output` (tensor)
- **Outputs**: CBHandle on receiver core with reduced result
- **Key kwargs**:
  - `cluster_axis` -- 0 for row-wise, 1 for column-wise reduction
  - `residual` -- optional residual tensor for fused add
- **Mechanism**: BRISC on sender core reads input data, writes to neighbor via fabric. NCRISC on receiver waits for fabric arrival. TRISC sums local + received data (+ optional residual). Uses `setup_fabric()` for fabric connection setup.
- **Usage**: PostSdpa final all-reduce, MoE ReduceToOne

### CclBroadcast

- **Class**: `CclBroadcast` | **Type**: MicroOp
- **File**: `blaze/ops/ccl_broadcast/op.py`
- **Purpose**: Multi-device fabric broadcast from a sender device to all devices in the mesh. Supports dual-axis broadcast on 2D meshes (e.g., 4x2 for TP=8), torus topology, and secondary senders for column-axis fanout.
- **Inputs**: `src` (CBHandle or tensor on sender device)
- **Outputs**: CBHandle on broadcast worker core
- **Key kwargs**:
  - `sender_coord` -- (row, col) of the sender device in the mesh
  - `secondary_cluster_axis` -- optional secondary broadcast axis
  - `num_links` -- fabric links per direction
  - `num_iterations` -- number of broadcast rounds
  - `is_torus` -- enable torus routing (wrap-around at mesh edges)
- **Mechanism**: Uses `compute_routing()` to determine each device's role (sender, secondary_sender, receiver), hop counts, and forwarding topology. Fabric connections are set up per-direction via `setup_fabric()`.
- **Usage**: BroadcastRMSNorm preamble (broadcasting hidden states before RMSNorm)

### ReduceToOne

- **Class**: `ReduceToOne` | **Type**: MicroOp
- **File**: `blaze/ops/reduce_to_one/op.py`
- **Purpose**: Multi-device reduction to a single root device using a hierarchical tree topology. Supports TP=2 (1x2), TP=4 (4x1), and TP=8 (4x2) mesh shapes with optional torus routing. The final result is written to a single output tensor on the root device.
- **Inputs**: `input` (tensor or CBHandle), `intermediate` (tensor), `output` (tensor)
- **Outputs**: CBHandle on output core (root device only)
- **Key kwargs**:
  - `root_mesh_coord` -- MeshCoordinate of the root device
  - `is_torus` -- enable torus topology
  - `sender_cores` -- list of cores to avoid (prevent NOC congestion with broadcast fabric)
  - `residual` -- optional residual tensor for fused add on root
- **Mechanism**: Uses `get_reduction_topology()` to classify each device as root, intermediate (ROOT2/ROOT3), or leaf. Leaf devices send to their nearest intermediate; intermediates reduce and forward to root. Fabric cores are placed adjacent to compute columns to avoid NOC conflicts.
- **Usage**: Final MoE reduction after routed + shared expert merge

---

## Barrier and Synchronization

### BarrierSender

- **Class**: `BarrierSender` | **Type**: MicroOp
- **File**: `blaze/ops/barrier_sender/op.py`
- **Purpose**: Send semaphore signals to receiver cores for intra-dispatch synchronization. Does not produce or consume data -- purely provides ordering guarantees between pipeline stages.
- **Inputs**: None (synchronization only)
- **Outputs**: None (`does_produce_output=False`)
- **Key kwargs**:
  - `sender_cores` -- list of cores that send barrier signals
  - `receiver_cores` -- list of cores to signal
  - `execute_on_ncrisc` / `execute_on_brisc` -- which RISC sends
  - `send_to_ncrisc` / `send_to_brisc` -- which RISC on receiver to signal
  - `ncrisc_semaphore_l1_addr` / `brisc_semaphore_l1_addr` -- semaphore addresses
- **Usage**: GLM Q branch (signal nope completion to head cores before Shard2CB)

### BarrierReceiver

- **Class**: `BarrierReceiver` | **Type**: MicroOp
- **File**: `blaze/ops/barrier_receiver/op.py`
- **Purpose**: Wait for semaphore signals from sender cores. The receiving counterpart to BarrierSender.
- **Inputs**: None
- **Outputs**: None (`does_produce_output=False`)
- **Key kwargs**:
  - `receiver_cores` -- list of waiting cores
  - `execute_on_ncrisc` / `execute_on_brisc` -- which RISC waits
  - `wait_value` -- semaphore threshold to unblock
  - `ncrisc_semaphore_l1_addr` / `brisc_semaphore_l1_addr` -- semaphore addresses
- **Usage**: GLM Q branch (head cores wait for nope data before absorption MatmulCB)

### PipelineStageSync

- **Class**: `PipelineStageSync` | **Type**: MicroOp
- **File**: `blaze/ops/pipeline_stage_sync/op.py`
- **Purpose**: Cross-device pipeline stage synchronization. One device's signalling core sends a fabric-routed semaphore increment to another device's stalling core, which blocks until the signal arrives. Supports multi-hop forwarding through intermediate devices.
- **Inputs**: None
- **Outputs**: None (`does_produce_output=False`)
- **Key kwargs**:
  - `src_device_mesh_coord` / `dst_device_mesh_coord` -- source and destination device coordinates
  - `signalling_core` / `stalling_core` -- logical core coordinates
  - `signalling_kernel_risc` / `stalling_kernel_risc` -- `Risc.NCRISC` or `Risc.BRISC` (not TRISC)
- **Mechanism**: Uses `build_signalling_path()` to compute the shortest route through the mesh (row-first, then column), identifies intermediate forwarding devices, and sets up fabric connections along the path.
- **Usage**: LMHeadSampling pipeline sync between attention output and LM head

### DMRiscHandshake

- **Class**: `DMRiscHandshake` | **Type**: MicroOp
- **File**: `blaze/ops/dm_risc_handshake/op.py`
- **Purpose**: Intra-core handshake between BRISC and NCRISC via a local semaphore. Used when one data-movement RISC needs to wait for the other to complete a local operation before proceeding.
- **Key kwargs**:
  - `cores` -- active cores
  - `signalling_risc` -- which RISC signals (the other waits)

---

## CB Lifecycle Utilities

### CbReconfig

- **Class**: `CbReconfig` | **Type**: MicroOp
- **File**: `blaze/ops/cb_reconfig/op.py`
- **Purpose**: Reconfigure CB interfaces at a phase boundary in multi-phase FusedPrograms. Marks a reconfig point where CB addresses are remapped on overlapping cores. The CT arg `cb_config_l1_addr` is a placeholder patched at build time by `_build_multi_phase()`.
- **Usage**: GlmQBranchFlash (phase boundary between Q branch and DsaSparseFlashDecode)

### CbScratchReset

- **Class**: `CbScratchReset` | **Type**: MicroOp
- **File**: `blaze/ops/cb_scratch_reset/op.py`
- **Purpose**: Reset a scratch CB's per-RISC FIFO pointers at a temporal-reuse boundary. Enables re-filling a CB that was previously consumed without reallocating it.
- **Key kwargs**: `cb_id` -- the CB to reset
- **Usage**: Temporal reuse patterns in multi-phase kernels

---

## Migration (DRAM/Host Transfers)

### MigrationOp / MigrationInterface

- **Class**: `MigrationOp` (MicroOp) + `MigrationInterface` (service class)
- **File**: `blaze/ops/migration/op.py`
- **Purpose**: Bidirectional DRAM-to-host and device-to-device data transfer via sockets with fabric transport support. Implements a full command-driven protocol with read/write/migrate commands, a 32-slot migration table for tracking in-flight transfers, and host-notified completion via D2H completion sockets.
- **Sockets per device**:
  - H2D command socket (64-byte command pages)
  - D2H data socket (device DRAM to host)
  - H2D data socket (host to device DRAM, DEVICE_PULL mode)
  - D2H completion socket (migration-id completion messages)
- **Key features**:
  - Per-bank parallel DRAM commands (`compute_dram_commands()`)
  - Cross-device fabric migration (`compute_migrate_commands()`)
  - 32-slot migration table with semaphore-based completion tracking
  - Termination via global semaphore
- **Usage**: KV cache migration, weight prefetching, model parallelism data redistribution

---

## Fused Data Movement Compositions

### BroadcastRMSNorm

- **Class**: `BroadcastRMSNorm` | **Type**: FusedOp
- **File**: `blaze/ops/broadcast_rmsnorm/op.py`
- **Purpose**: Fused CCL broadcast + RMSNorm. Broadcasts a tensor from a sender device to all devices in the mesh, then applies RMSNorm on the received data.
- **Composition**: `CclBroadcast.emit()` -> `RMSNorm.emit()`
- **Usage**: Input/post-attention layer norms in DeepSeek V3, used as the first stage of PreSDPA

### BroadcastRMSNormMcast

- **Class**: `BroadcastRMSNormMcast` | **Type**: FusedOp
- **File**: `blaze/ops/broadcast_rmsnorm_mcast/op.py`
- **Purpose**: Extends BroadcastRMSNorm with an Mcast to fan out the normalized result to all cores.
- **Composition**: `BroadcastRMSNorm.emit()` -> `Mcast.emit()`
- **Usage**: PreSDPA preamble (broadcast hidden states, normalize, distribute to Q and KV branches)

[<-- Overview](01_op_library_overview.md) | [Next: Compute Ops -->](03_compute_ops.md)
