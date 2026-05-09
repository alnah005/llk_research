# 8.2 Multi-Device Communication and Parallelism

This section traces the exact communication primitives that connect pipeline
stages and the intra-device parallelism strategies within each stage. The key
architectural insight is a two-level parallelism design: **pipeline parallelism**
between 4x2 mesh slices (covered in Section 8.1) and **tensor/expert
parallelism** within each 4x2 mesh slice. This section focuses on the D2D
exchange mechanism, the PipelineBlock orchestration layer, and the design
rationale for TP=2, EP=8, and the ReduceToOne device roles -- all of which are
co-designed with the 4x2 mesh geometry.

> **Cross-reference -> Chapter 5, Section 5.1**: How MLA attention uses TP=2
> across the 2-column mesh (per-device head distribution).

> **Cross-reference -> Chapter 6, Section 6.2**: How MoE uses EP=8 across all
> 8 devices (routed expert parallelism).

> **Cross-reference -> Chapter 3, Section 3.2**: CCL all-reduce and broadcast
> primitives that implement intra-mesh collective communication.

Source: `micro_ops/d2d_exchange/op.py` (360 lines), `micro_ops/pipeline_block/op.py` (197 lines), `micro_ops/host_io/op.py` (404 lines), `micro_ops/ccl_all_reduce/op.py` (498 lines), `micro_ops/reduce_to_one_b1/op.py`, `micro_ops/ccl_broadcast/op.py` (288 lines)

---

## 8.2.1 D2D Exchange: The Inter-Stage Data Transfer Primitive

The `SocketInterface` class (`micro_ops/d2d_exchange/op.py`, 360 lines) is the
fundamental building block for moving data between pipeline stages. It manages
a bidirectional data path between two cores -- potentially on different physical
devices across the fabric.

### Architecture

Each `SocketInterface` instance sets up three socket connections:

```
 D2D EXCHANGE: ONE STAGE'S DATA PATH
 ====================================

                        SocketInterface
         +-----------------------------------------+
         |                                         |
  From   |  +-------------+     +--------------+   |  To
  prev   |  | Upstream    |     | Downstream   |   |  next
  stage ---->| Socket      |---->| Socket       |---->  stage
         |  | (receiver)  |     | (sender)     |   |
         |  +-------------+     +--------------+   |
         |        ^                    |            |
         |        |   +------------+   |            |
         |        +---| Internal   |---+            |
         |            | Socket Pair|                |
         |            +------------+                |
         |                                         |
         |  Termination semaphore (global)         |
         +-----------------------------------------+
```

The three socket layers serve distinct purposes:

| Socket | Direction | Purpose |
|---|---|---|
| Upstream | Previous stage -> this stage's send_core | Receives activation from predecessor |
| Internal | send_core -> recv_core | Intra-stage forwarding (may cross devices) |
| Downstream | recv_core -> next stage's entry_node | Sends activation to successor |

**Design rationale -- three-layer socket architecture:** The internal socket
pair exists to decouple the upstream receive from the downstream send. Each
`d2d_exchange` kernel runs in a simple loop: receive from upstream, forward to
downstream. By splitting the send and receive paths into separate cores
(potentially on separate devices within the same mesh), the implementation
avoids contention on socket buffers and allows overlapping receive of the next
activation with sending the current one.

> **Source**: `micro_ops/d2d_exchange/op.py`, lines 41-149

### Local vs. Cross-Host Sockets

The `SocketInterface` constructor detects whether both sender and receiver
meshes are available on the local process:

> **Source**: `micro_ops/d2d_exchange/op.py`, lines 62-73:
> ```python
> if sender_mesh.get_mesh_device() and receiver_mesh.get_mesh_device():
>     assert sender_mesh.get_mesh_id() == receiver_mesh.get_mesh_id()
>     self.mesh_device = sender_mesh.get_mesh_device()
>     self.local_socket = True
> else:
>     self.mesh_device = (sender_mesh.get_mesh_device()
>                         if sender_mesh.get_mesh_device()
>                         else receiver_mesh.get_mesh_device())
>     self.local_socket = False
> ```

When `local_socket=True`, both sender and receiver programs run within the same
`MeshProgramDescriptor`. When `local_socket=False`, only one side's program is
created locally; the other side runs on a different MPI rank.

**Design rationale -- MeshSocket for cross-host:** When sender and receiver are
on different hosts, neither host has a `MeshDevice` handle for the other's mesh.
The `MeshSocket` abstraction uses explicit mesh IDs to establish the connection
through the fabric router, which handles cross-host routing transparently. The
`MeshWrapper` class (lines 24-38) encapsulates this duality: it holds either a
`MeshDevice` reference (local) or just a `mesh_id` integer (remote).

### Fabric-Aware Kernel Creation

The `_create_program()` method determines whether fabric connections are needed
by comparing fabric node IDs:

> **Source**: `micro_ops/d2d_exchange/op.py`, lines 183-184:
> ```python
> use_fabric_on_receiver = my_upstream_fabric_node_id != my_fabric_node_id
> use_fabric_on_sender = my_downstream_fabric_node_id != my_fabric_node_id
> ```

When fabric is required, the data is split across **2 forward links** and
**1 backward link** to maximize bandwidth:

> **Source**: `micro_ops/d2d_exchange/op.py`, lines 186-187:
> ```python
> num_fwd_links = 2
> num_bwd_links = 1
> ```

**Design rationale -- 2 forward links, 1 backward:** Forward links carry the
activation data (high bandwidth needed), while the backward link carries only
signaling/acknowledgment. Using 2 forward links doubles the available bandwidth
for the data path, which is the bottleneck in pipeline forwarding.

The page splitting formula:

```python
page_size_per_link = self.page_size // num_fwd_links       # 14336 // 2 = 7168
num_whole_fabric_packets_per_link = page_size_per_link // fabric_max_payload_size
partial_packet_size_per_link = page_size_per_link % fabric_max_payload_size
```

> **Source**: `micro_ops/d2d_exchange/op.py`, lines 193-197

The kernel `d2d_exchange.cpp` runs a polling loop that receives from the
upstream socket and forwards to the downstream socket, with a termination
semaphore check.

### Program Composition for Same-Device Stages

When sender and receiver are on the same physical device (same `device_coord`),
their programs are merged into a single `ProgramDescriptor` with combined
kernels and circular buffers:

> **Source**: `micro_ops/d2d_exchange/op.py`, lines 309-327

```python
if same_device:
    combined_program = ttnn.ProgramDescriptor(
        kernels=sender_program.kernels + receiver_program.kernels,
        semaphores=[],
        cbs=combined_cbs,
    )
```

This avoids the overhead of launching two separate programs on one chip.

### Termination Protocol

Clean shutdown is accomplished via a global semaphore:

> **Source**: `micro_ops/d2d_exchange/op.py`, lines 347-354:
> ```python
> def terminate(self, sync_devices):
>     ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)
>     if sync_devices:
>         ...
>         ttnn.synchronize_device(...)
> ```

Setting the semaphore to 1 causes the polling kernel to exit within
approximately 1000 device cycles (~1 us at ~1 GHz clock).

---

## 8.2.2 D2D Exchange Flow Diagram

The following diagram shows how data flows through a chain of three pipeline
stages, illustrating both intra-host and inter-host D2D exchange:

```
 D2D EXCHANGE CHAIN: 3 CONSECUTIVE PIPELINE STAGES
 ===================================================

 Stage N (Host A, Slice 0)      Stage N+1 (Host A, Slice 1)      Stage N+2 (Host B, Slice 0)
 +------------------------+    +------------------------+        +------------------------+
 |                        |    |                        |        |                        |
 | [Decoder Layer N]      |    | [Decoder Layer N+1]    |        | [Decoder Layer N+2]    |
 |     |                  |    |     ^            |     |        |     ^                  |
 |     v                  |    |     |            v     |        |     |                  |
 | +--------+  +--------+ |    | +--------+  +--------+|        | +--------+  +--------+ |
 | |upstream|  |internal | |    | |upstream|  |internal||        | |upstream|  |internal | |
 | |recvr   |->|  pair  | |    | |recvr   |->|  pair  ||        | |recvr   |->|  pair  | |
 | +--------+  +---+----+ |    | +---^----+  +---+----+|        | +---^----+  +---+----+ |
 |                 |       |    |     |            |    |        |     |            |     |
 |             +---v----+  |    | +---+----+  +---v---+|        | +---+----+  +---v---+  |
 |             |downstrm|  |    | |downstrm|  |downstr||        | |downstrm|  |downstr|  |
 |             |sender  |--------->|recvr  |  |sender |---------->|recvr  |  |sender |  |
 |             +--------+  |    | +--------+  +-------+|        | +--------+  +-------+  |
 |                         |    |                       |        |                        |
 +-------------------------+    +-----------------------+        +------------------------+
                                                        ^
     LOCAL SOCKETS (same host)                          |     FABRIC SOCKETS (cross-host)
     No fabric overhead                          MeshSocket with
                                                 fabric connections
                                                 (2 fwd + 1 bwd links)
```

Key observations:
- Intra-host transfers between slices on the same host use local socket pairs
  with no fabric overhead.
- Inter-host transfers create `MeshSocket` instances with explicit mesh IDs and
  use fabric connections for the actual data movement.
- The internal socket pair within each `SocketInterface` provides the
  sender-to-receiver bridge on the local mesh device.

---

## 8.2.3 PipelineBlock: Per-Stage Orchestration

The `PipelineBlock` class (`micro_ops/pipeline_block/op.py`, 197 lines)
encapsulates a single pipeline stage's complete D2D infrastructure. It manages
entry and exit `SocketInterface` instances, and for the first stage (mesh_id=0),
also manages the `HostInterface` for H2D/D2H communication.

### Construction Logic

The constructor queries the pipeline topology from the runtime:

> **Source**: `micro_ops/pipeline_block/op.py`, lines 53-57:
> ```python
> self.my_mesh_id = mesh_device.get_system_mesh_id()
> self.is_pipeline_start = self.my_mesh_id == 0
> pipeline_config = ttnn._ttnn.multi_device.experimental.generate_blitz_decode_pipeline(mesh_device)
> num_procs = int(ttnn.distributed_context_get_size())
> assert len(pipeline_config) == num_procs + 1
> ```

The `pipeline_config` is a list of `num_procs + 1` entries (one per stage plus
one for the loopback). Each entry contains `entry_node_coord` and
`exit_node_coord` that identify which device within the 4x2 mesh serves as the
entry point and exit point for that stage.

### First Stage (Embedding + Host I/O)

When `mesh_id == 0`, the PipelineBlock creates additional infrastructure:

```
 FIRST PIPELINE STAGE (mesh_id == 0)
 ====================================

                   Host
                    |
            write_tensor (H2D)
                    |
                    v
 +------------------------------------------+
 | PipelineBlock (Stage 0)                  |
 |                                          |
 |  HostInterface                           |
 |  +------------------------------------+  |
 |  | H2D socket (HOST_PUSH, 64 bytes)   |  |
 |  |     |                              |  |
 |  |     v                              |  |
 |  | fused_h2d_receiver_embedding.cpp   |  |
 |  |   token_id -> DRAM lookup ->       |  |
 |  |   embedding vector (bfloat16)      |  |
 |  |     |                              |  |
 |  |     v  (downstream socket)         |  |
 |  +-----+------------------------------+  |
 |        |                                  |
 |        v                                  |
 |  Exit SocketInterface -----> Stage 1      |
 |                                           |
 |  Entry SocketInterface <---- Stage N-1    |
 |        |   (loopback from last stage)     |
 |        v                                  |
 |  +------------------------------------+   |
 |  | D2H socket -> read_tensor (D2H)   |   |
 |  +------------------------------------+   |
 +-------------------------------------------+
              |
              v
            Host (read output)
```

The embedding tensor is fused into the H2D receiver kernel:

> **Source**: `micro_ops/pipeline_block/op.py`, lines 67 and 76-77:
> ```python
> token_size_bytes = 64  # Hardcode for now
> ...
> embedding_size_bytes = embedding_tensor.shape[3] * dtype_size(embedding_tensor.dtype)
> ```

The H2D page size is 64 bytes (enough for a batch of token IDs), and the
downstream D2D socket page size must equal the embedding vector size:

> **Source**: `micro_ops/pipeline_block/op.py`, lines 79-82:
> ```python
> assert downstream_d2d_socket_page_size == embedding_size_bytes, \
>     "Expected downstream D2D Socket Page size to be equal to embedding size"
> ```

**Design rationale -- fused embedding on stage 0:** The HostInterface on
stage 0 performs a DRAM table lookup to convert the token ID into an embedding
vector, which is then forwarded to stage 1 via D2D socket. This fusion
eliminates a separate embedding device-to-device transfer and reduces the
pipeline to a single kernel for the entire H2D-receive-embed-forward sequence.
The embedding tensor must be pre-loaded in DRAM on the device hosting stage 0.

### Intermediate Stages

For stages with `mesh_id > 0`, the PipelineBlock creates two SocketInterfaces:

> **Source**: `micro_ops/pipeline_block/op.py`, lines 134-157

The receiver mesh for the exit socket wraps around when at the last stage:

> **Source**: `micro_ops/pipeline_block/op.py`, line 156:
> ```python
> receiver_mesh=MeshWrapper(mesh_id=self.my_mesh_id + 1
>                           if self.my_mesh_id < num_procs - 1 else 0),
> ```

### Lifecycle Methods

| Method         | Behavior                                                       |
|----------------|----------------------------------------------------------------|
| `run()`        | Launches HostInterface (stage 0 only), then exit + entry D2D   |
| `terminate()`  | MPI barrier, then terminates all interfaces in order           |
| `write_token()`| Writes to H2D socket (stage 0 only)                           |
| `read_output()`| Reads from D2H socket (stage 0 only)                          |

The termination sequence uses an MPI barrier to ensure all outstanding pipeline
requests are completed before signaling shutdown:

> **Source**: `micro_ops/pipeline_block/op.py`, lines 168-178:
> ```python
> def terminate(self):
>     ttnn.distributed_context_barrier()
>     if self.is_pipeline_start:
>         self.host_io.terminate(False)
>         self.entry_socket_interface.terminate(False)
>         self.exit_socket_interface.terminate(True)
>     else:
>         self.entry_socket_interface.terminate(False)
>         self.exit_socket_interface.terminate(True)
> ```

Note: the `sync_devices` argument is `True` only for the last interface
terminated, ensuring all device programs complete before proceeding.

---

## 8.2.4 HostInterface: Bidirectional Host Communication

The `HostInterface` class (`micro_ops/host_io/op.py`, 404 lines) provides the
bridge between the host CPU and the device pipeline. It supports two modes:

### Loopback Mode (Testing)

In loopback mode, the H2D receiver and D2H sender communicate through a
circular buffer (CB) on the same core:

```
 LOOPBACK MODE (single core, single chip)
 =========================================

 Host                           Device Core (0,0)
  |                            +------------------+
  | write_tensor()             | H2D Receiver     |
  |--------------------------->|   (writer kernel) |
  |                            |       |           |
  |                            |       v  CB[0]    |
  |                            |       |           |
  |                            | D2H Sender        |
  | read_tensor()              |   (reader kernel) |
  |<---------------------------|                   |
  |                            +------------------+
```

> **Source**: `micro_ops/host_io/op.py`, lines 60-61 and 72-74:
> ```python
> self.loopback_mode = loopback_mode
> ...
> if loopback_mode:
>     if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]:
>         raise ValueError("Expected ... to be on the same core.")
> ```

### Socket Mode (Production)

In production mode, the H2D receiver forwards data downstream via a D2D socket,
and the D2H sender receives data from an upstream D2D socket:

```
 SOCKET MODE (cross-core, potentially cross-device)
 ===================================================

 Host                  H2D Core               Downstream Core
  |                   +------------+          +-------------+
  | write_tensor()    | H2D Recvr  |          |             |
  |------------------>| + Embedding|--------->| (next stage)|
  |                   +------------+  D2D     +-------------+
  |                   downstream socket
  |
  |                   D2H Core               Upstream Core
  |                   +------------+          +-------------+
  | read_tensor()     | D2H Sender |          |             |
  |<------------------| (reader)   |<---------| (prev stage)|
  |                   +------------+  D2D     +-------------+
  |                   upstream socket
```

> **Source**: `micro_ops/host_io/op.py`, lines 102-128

**Design rationale -- separate H2D and D2H cores:** The HostInterface supports
placing H2D and D2H on different devices within the mesh. This enables the H2D
entry point and D2H exit point to be on physically different devices -- critical
for the loopback topology where stage 0's H2D (entry) and stage N's D2H (exit)
may be on different corners of the mesh.

### Fused Embedding Kernel

When an embedding tensor is provided, the H2D receiver kernel is replaced with
a fused version that performs an inline DRAM table lookup:

> **Source**: `micro_ops/host_io/op.py`, lines 195-199:
> ```python
> kernel_source = (
>     "micro_ops/host_io/kernels/fused_h2d_receiver_embedding.cpp"
>     if self.has_embedding
>     else "micro_ops/host_io/kernels/h2d_receiver.cpp"
> )
> ```

The fused kernel receives a 64-byte token packet via H2D, extracts the token
ID, computes the DRAM address as `embedding_base + token_id * embedding_page_size`,
reads the embedding row into a CB, and forwards it downstream. The embedding
tensor must be DRAM-interleaved with row-major layout (asserted at lines 139-141).

> **Cross-reference -> Chapter 3, Section 3.2**: Embedding weight preparation
> and DRAM interleaving.

---

## 8.2.5 Intra-Device Parallelism

Within each 4x2 mesh (pipeline stage), multiple parallelism strategies operate
simultaneously during a decode step.

### MLA Tensor Parallelism (TP=2)

The Multi-head Latent Attention (MLA) computation is tensor-parallel across the
2 columns of the 4x2 mesh:

```
 MLA TENSOR PARALLELISM (TP=2 across columns)
 =============================================

       Column 0           Column 1
      +--------+         +--------+
 R0   | Dev 0  |         | Dev 1  |
      | Q/K/V  |         | Q/K/V  |    Each column computes
      | shard 0|         | shard 1|    half the attention heads
      +--------+         +--------+
 R1   | Dev 2  |         | Dev 3  |
      +--------+         +--------+
 R2   | Dev 4  |         | Dev 5  |
      +--------+         +--------+
 R3   | Dev 6  |         | Dev 7  |
      +--------+         +--------+
          |                   |
          +---AllReduce(2)----+
                  |
            Merged output
```

After the attention computation, a **2-device AllReduce** along `cluster_axis=1`
(the column axis) sums the partial results.

**Design rationale -- why TP=2 and not higher:**

1. **Hardware geometry:** The 4x2 mesh has 2 columns. The LINE dimension (cols)
   provides a direct device-to-device link between each row pair. TP=2 maps
   naturally to this 2-column layout.

2. **All-reduce cost:** With only 2 devices, the all-reduce is a single neighbor
   exchange -- no multi-hop routing needed.

3. **Activation size constraint:** For decode with B=1 and hidden_dim=7168,
   each TP shard holds 7168/2 = 3584 elements. The all-reduce payload is
   `3584 x bf16 = 7168 bytes` -- small enough for a single exchange.

4. **Weight memory:** TP=2 halves the weight memory per device for attention
   projections. TP=4 or TP=8 would reduce per-device weights further, but the
   all-reduce communication becomes more complex.

> **Cross-reference -> Chapter 4, Section 4.3**: Complete MLA pipeline
> (Pre-SDPA, Flash MLA, Post-SDPA).

### CCL AllReduce (2-device)

The `DeepseekMinimalAllReduce` class (`micro_ops/ccl_all_reduce/op.py`, 498 lines) implements neighbor exchange followed by local reduction with HiFi4/fp32 accumulation.

> **Cross-reference → Chapter 3, Section 3.3.1**: Full 2-device exchange diagram, CB layout, RISC roles, tile format bridge, and fused residual add capability.

In the MLA TP=2 context, AllReduce sums partial o_proj results from the two column-paired devices, with an optional fused residual add that saves one kernel launch per layer.

### MoE Expert Parallelism (EP=8)

Each MoE layer distributes its 256 routed experts across all 8 devices in the
4x2 mesh (32 experts per device). After expert computation, a ReduceToOne
operation gathers the results.

**Design rationale -- why EP=8:**

1. **Expert distribution:** 256 experts / 8 devices = 32 experts per device.
   With DeepSeek V3's top-8 routing, experts are distributed across multiple
   devices, requiring inter-device gather/scatter.

2. **Full mesh utilization:** EP=8 uses all devices in the 4x2 mesh. Unlike
   TP=2 which only pairs columns, EP=8 exploits both dimensions. This maximizes
   the aggregate DRAM bandwidth available for expert weight streaming -- the
   bottleneck for MoE decode.

> **Cross-reference -> Chapter 5, Section 5.2**: MoE gate, expert dispatch,
> and reduction.

### CCL Broadcast

The `DeepseekMinimalBroadcast` class (`micro_ops/ccl_broadcast/op.py`, 288 lines) broadcasts data from a sender device to all others via dual-axis broadcast.

> **Cross-reference → Chapter 3, Section 3.3.2**: Single-axis and dual-axis broadcast algorithms, RING topology multicast with `start_distance_in_hops`/`range_hops`.

In the MoE context, dual-axis broadcast distributes gate computation results (computed on one device) to all 8 devices before expert dispatch.

### ReduceToOne: 3-Level Reduction Tree

The `ReduceToOneB1` class (`micro_ops/reduce_to_one_b1/op.py`) implements a
multi-device reduce-to-one operation using a 3-level reduction tree on the 4x2
mesh.

> **Cross-reference → Chapter 3, Section 3.3.3**: The full reduction tree diagram, 4 device role constants (MESH_LEAF=0, MESH_ROOT3=1, MESH_ROOT2=2, MESH_ROOT1=3), role assignment via `get_device_role()` (lines 45-78), per-level data flow table, and CB layout.

**Design rationale — why 3-level tree on 4x2:** A flat all-reduce across 8
devices would require 7 sequential reduction steps. The 3-level tree completes
in 3 steps by exploiting the RING topology within each column for levels 1-2,
then the LINE topology between columns for level 3. The asymmetric root
placement at [1,1] rather than [0,0] minimizes the maximum hop distance from
any LEAF: the farthest LEAF [3,0] takes exactly 3 hops.

The ReduceToOne also supports a **D2D_0 aggregation output** where a dedicated
aggregator core on the ROOT1 device collects results from all 8 worker cores
via D2D sockets and forwards them to a downstream pipeline stage (`micro_ops/reduce_to_one_b1/op.py`, lines 103-212).

---

## 8.2.6 Complete Intra-Stage Communication Map

The following diagram summarizes all communication patterns within a single 4x2
mesh during one decode step:

```
 ALL COMMUNICATION WITHIN ONE 4x2 PIPELINE STAGE (MoE Layer)
 =============================================================

 Step 1: RMSNorm (local per-device, no communication)

 Step 2: MLA Attention
    Q/K/V projections -> Flash MLA -> Post-SDPA
    AllReduce(TP=2) across columns: col 0 <--> col 1

 Step 3: Residual add (local, fused with AllReduce)

 Step 4: RMSNorm (local per-device)

 Step 5: MoE Gate
    Compute on one device -> Broadcast to all 8

 Step 6: Expert Dispatch + Computation
    Each device processes its 32/256 experts locally

 Step 7: ReduceToOne (3-level tree)
    Level 1: rows 0,3 -> rows 1,2 (LEAF -> ROOT3/ROOT2)
    Level 2: row 2 -> row 1 (ROOT3 -> ROOT2/ROOT1)
    Level 3: col 0 -> col 1 (ROOT2 -> ROOT1)

 Step 8: Residual add (at ROOT1, then forward)

 Step 9: D2D Exchange -> next pipeline stage
```

---

## 8.2.7 Cross-Host Fabric Routing

When D2D exchange crosses host boundaries, the fabric router handles the
multi-hop path transparently. The key parameters in the textproto descriptor:

1. **Channel count:** Intra-pod connections have 8 channels; inter-pod
   connections drop to 2-4 channels. The reduced count is a physical constraint.

2. **Z-direction assignment:** Superpod inter-pod connections use
   `assign_z_direction: true`, telling the router to use the dedicated vertical
   links. This prevents inter-pod traffic from contending with intra-pod traffic.

3. **RELAXED policy:** All connections use `policy: RELAXED`, allowing the
   router to reorder packets and use any available path. Acceptable because D2D
   exchange operates on fixed-size pages with explicit synchronization.

> **Source**: `micro_ops/d2d_exchange/op.py`, lines 183-184

When fabric is needed, `ttnn.setup_fabric_connection()` populates the kernel
runtime args with the routing information for each link, enabling the kernel
to construct properly addressed fabric packets.

---

## 8.2.8 Fabric Link Utilization Summary

| Primitive | Scope | Topology | Devices | Links Used | Data Size (B=1) |
|---|---|---|---|---|---|
| D2D Exchange | Inter-stage | Point-to-point | 2 | 2 fwd + 1 bwd | 14 KB (hidden_dim bf16) |
| AllReduce (MLA) | Intra-stage | Column pair | 2 | 1 fwd + 1 bwd per device | 7 KB (hidden_dim/2 bf16) |
| Broadcast (MoE gate) | Intra-stage | Dual-axis | 8 | Multicast tree | ~14 KB (hidden_dim bf16) |
| ReduceToOne (MoE) | Intra-stage | 3-level tree | 8 | 3 sequential hops | 14 KB per partial sum |
| HostInterface H2D | Stage 0 | PCIe | 1 | PCIe | 64 bytes (token) |
| HostInterface D2H | Stage 0 | PCIe | 1 | PCIe | Varies (logits/token) |

**Design rationale -- why the 4x2 mesh drives everything:** The 4x2 geometry
with RING rows and LINE columns is the fundamental constraint. TP=2 aligns with
the 2-column LINE dimension. EP=8 uses all 8 devices. The 3-level ReduceToOne
tree exploits the 4-row RING for vertical reduction and the 2-column LINE for
the final cross-column step. Every parallelism and communication decision is a
direct consequence of this fixed hardware shape.

> **Cross-reference -> Chapter 6, Section 6.1**: Detailed kernel-level view
> of the dense FFN computation within one stage.
