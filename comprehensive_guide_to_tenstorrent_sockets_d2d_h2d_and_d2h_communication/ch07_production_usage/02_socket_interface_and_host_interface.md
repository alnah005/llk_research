# 7.2 -- Socket Interface and Host Interface

The pipeline runtime wraps raw socket APIs behind two abstraction layers: `SocketInterface` for D2D inter-mesh communication and `HostInterface` for H2D/D2H host I/O. These wrappers handle lifecycle differences between socket types, manage the distinction between local and remote meshes, and enforce the correct dispatch order within each pipeline stage.

This section documents both interfaces with annotated code patterns from `op.py`, compares them systematically, and provides decision tables for the `LoopbackConfig` modes that control autoregressive decode routing.

---

## 7.2.1 SocketInterface (D2D Communication)

### What Breaks: Direct MeshSocket Creation in a Pipeline

Creating `MeshSocket` objects directly (bypassing `SocketInterface`) causes lifecycle mismatches -- sockets are not torn down on stage reconfiguration, leaking L1 memory -- and coordinate confusion without `MeshWrapper` to distinguish local vs. remote mesh identifiers.

### Architecture

`SocketInterface` wraps `MeshSocket` and `MeshWrapper`:

```
  SocketInterface
  +--------------------------------------------+
  |  MeshWrapper                               |
  |    local_mesh_id, remote_mesh_id,          |
  |    is_sender (bool)                        |
  |                                            |
  |  MeshSocket                                |
  |    SocketConfig: connections, mem_config,   |
  |    sender_mesh_id, receiver_mesh_id,       |
  |    sender_rank, receiver_rank              |
  +--------------------------------------------+
```

### Annotated Construction and API

```python
# From op.py -- SocketInterface wraps MeshSocket for pipeline use
class SocketInterface:
    def __init__(self, mesh_device, socket_config, is_sender):
        self.socket = ttnn.MeshSocket(mesh_device, socket_config)
        self.is_sender = is_sender    # Track role for send/recv validation

    def send(self, tensor):
        assert self.is_sender, "Cannot send on a receiver socket"
        ttnn.experimental.send_async(tensor, self.socket)  # Non-blocking

    def recv(self, output_tensor):
        assert not self.is_sender, "Cannot recv on a sender socket"
        ttnn.experimental.recv_async(output_tensor, self.socket)
```

### What Breaks: MeshWrapper Sender/Receiver Reversal

If the `is_sender` flag is inverted, `SocketInterface` creates a sender socket on the receiver side and vice versa. `send_async()` completes instantly while `recv_async()` blocks forever with no error or timeout.

**Diagnosis**: Print `socket.get_socket_endpoint_type()` on both sides. Both will show the same endpoint type instead of complementary SENDER/RECEIVER.

### D2D Configuration and the 1:1 Core Constraint

```python
# Annotated: D2D socket configuration from op.py
connections = [
    ttnn.SocketConnection(
        MeshCoreCoord(exit_coord, core),   # sender core on exit chip
        MeshCoreCoord(entry_coord, core)   # receiver core on entry chip
    )
]
mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, d2d_fifo_size)
```

The `SocketConnection` contract: "Cannot reuse senders and receivers in a single socket context." Each connection is strictly 1:1. Sharing a sender or receiver core across connections in the same `SocketConfig` causes a validation error.

### What Breaks: Tensor Shape vs. Page Size Mismatch

If the tensor's byte size is not a multiple of `page_size`, the send writes a partial final page. The receiver hangs waiting for data or reads corrupted padding. **Rule**: `tensor_size_bytes % page_size == 0`. Minimum page size is 64 bytes (NOC word width on Blackhole).

---

## 7.2.2 HostInterface (H2D + D2H Communication)

### What Breaks: HostInterface on Non-Terminal Stages

Only terminal stages need host I/O (Stage 0 for H2D, Stage 4 for D2H). Creating a HostInterface on middle stages wastes L1 on unused buffers and risks core reuse violations if D2D and H2D sockets compete for the same core.

### Annotated Construction

```python
# From op.py -- HostInterface manages H2D + D2H lifecycle
class HostInterface:
    def __init__(self, mesh_device, h2d_core, d2h_core,
                 h2d_fifo_size, d2h_fifo_size):
        self.h2d_socket = ttnn.H2DSocket(
            mesh_device, MeshCoreCoord(h2d_core.device_coord, h2d_core.core_coord),
            BufferType.L1, h2d_fifo_size, H2DMode.HOST_PUSH
        ) if h2d_core else None  # HOST_PUSH: TLB-mapped PCIe posted writes
        self.d2h_socket = ttnn.D2HSocket(
            mesh_device, MeshCoreCoord(d2h_core.device_coord, d2h_core.core_coord),
            d2h_fifo_size
        ) if d2h_core else None  # Device NOC writes into pinned host memory
```

H2DSocket supports two modes -- HOST_PUSH (used by DeepSeek V3) and DEVICE_PULL (not used here). Both require vIOMMU.

All four `HostIoPlacement` cores (h2d_core, d2h_core, fwd_d2d_core, lb_d2d_core) must be distinct to avoid L1 and NOC contention between PCIe traffic and fabric traffic.

### What Breaks: H2D Without vIOMMU

Both H2D modes require vIOMMU. Without it, `H2DSocket` construction succeeds but the first `write()` silently corrupts memory or triggers an IOMMU fault. **Diagnosis**: Check `dmesg` for IOMMU errors.

### What Breaks: H2D Write Before Device Kernel Ready

Writing before the kernel calls `create_receiver_socket_interface()` lands data in L1 that the kernel may overwrite during initialization. Result: corrupted tokens, garbage output, no error message. **Fix**: Launch the device program and confirm kernel readiness before the first host write.

---

## 7.2.3 SocketInterface vs HostInterface Comparison

| Dimension | SocketInterface (D2D) | HostInterface (H2D/D2H) |
|-----------|-----------------------|--------------------------|
| Underlying socket type | MeshSocket | H2DSocket + D2HSocket |
| Transport | TT-Fabric (device-to-device) | PCIe (host-to-device / device-to-host) |
| Endpoints | Two mesh devices | Host CPU + one mesh device |
| FIFO location | L1 on sender and receiver cores | L1 (H2D) + host pinned memory (D2H) |
| Cross-host capability | Yes (via DistributedContext ranks) | No (single-host only) |
| Cross-process sharing | No (use DistributedContext) | Yes (export_descriptor/connect) |
| Bandwidth | TT-Fabric link BW | PCIe Gen4 (x8 or x1 depending on chip) |
| Page size minimum | 64 bytes | 64 bytes |
| vIOMMU required | No (device-only path) | Yes (all H2D/D2H modes) |
| Instantiated on stages | All except Combined | First, Last, Last+LB, Combined |

> **Note:** `export_descriptor`/`connect` is an H2D/D2H-only mechanism. D2D sockets use DistributedContext for cross-process coordination, not shared-memory descriptors.

---

## 7.2.4 LoopbackConfig Modes

The `LoopbackConfig` enum controls how the autoregressive decode loop routes data from the last pipeline stage back to the first.

### Mode Comparison Matrix

| Property | fabric_loopback | host_loopback | no_loopback |
|----------|-----------------|---------------|-------------|
| Loopback transport | D2D via TT-Fabric | D2H + H2D via host memory | No loopback |
| Loopback latency | Low (device-to-device) | High (device-host-device) | N/A |
| Host CPU involvement | None | Yes (reads D2H, writes H2D) | None |
| Extra sockets on Last stage | 1 loopback D2D send | 0 (uses existing D2H) | 0 |
| Extra sockets on First stage | 1 loopback D2D recv | 0 (uses existing H2D) | 0 |
| Cross-host loopback | Yes (D2D crosses host boundary) | Yes (each host handles its own) | N/A |
| Use case | Autoregressive decode (production) | Debug / fallback | Single-pass inference |

### LoopbackConfig Decision Table

```
  Does the model require autoregressive decode?
  |
  +-- NO --> no_loopback
  |          (Single-pass models: embeddings, classification)
  |
  +-- YES --> Is D2D fabric available between last and first mesh?
              |
              +-- YES --> fabric_loopback
              |           (Production autoregressive decode. Lowest latency.)
              |
              +-- NO --> host_loopback
                         (Fallback when fabric path unavailable or for debugging.)
```

### What Breaks: Loopback Core Contention

With `fabric_loopback`, Stage 4's exit chip has two D2D sockets (pipeline + loopback). If `lb_d2d_core` overlaps any other socket core, creation fails. **Fix**: Assign `lb_d2d_core` to a unique core in `HostIoPlacement`.

### What Breaks: Loopback Stall on Token Boundary

If `host_loopback` is used but the host sampling code does not call `h2d_socket.write()` before the next decode iteration begins, Stage 0 hangs waiting for input.

---

## 7.2.5 PipelineBlock Composition

`PipelineBlock` is the top-level orchestrator. It conditionally creates sockets based on stage configuration -- the same class handles all five configurations through conditional initialization, not subclassing.

### Construction Pattern

`PipelineBlock` inspects its stage configuration and creates only the sockets needed:

```python
# Simplified from op.py -- conditional socket creation driven by stage config
self.rank = int(ttnn.distributed_context_get_rank())
my_stages = [s for s in stages_metadata if s.rank == self.rank]
for stage in my_stages:
    config = pipeline_config[stage]
    if config.has_entry:      self.entry_socket = SocketInterface(..., is_sender=False)
    if config.has_exit:       self.exit_socket = SocketInterface(..., is_sender=True)
    if config.has_h2d or config.has_d2h:
        self.host_interface = HostInterface(mesh_device, h2d_core, d2h_core, ...)
    if loopback_config == LoopbackConfig.fabric_loopback:
        self.loopback_socket = SocketInterface(...)
```

### Interface Instantiation Per Stage Type

| Stage Type | SocketInterface (upstream) | SocketInterface (downstream) | SocketInterface (loopback) | HostInterface |
|------------|----------------------------|------------------------------|----------------------------|---------------|
| First+D2D | No | Yes | Yes (if fabric_loopback) | Yes (H2D, optional D2H) |
| Middle | Yes | Yes | No | No |
| Last+Loopback | Yes | No | Yes (send side) | Yes (D2H only) |
| Last | Yes | No | No | Yes (D2H only) |
| Combined | No | No | No | Yes (H2D + D2H) |

---

## 7.2.6 Dispatch Order Within a Pipeline Stage

When a `PipelineBlock` executes a single forward pass, the sockets are dispatched in a specific order. The critical rule: **receive before compute, compute before send**.

The op.py dispatch sequence for one token: (1) recv from D2D upstream or H2D, (2) compute model layers, (3) send via D2D downstream, loopback D2D, and/or D2H write.

### Dispatch Order Comparison

| Step | First+D2D | Middle | Last+Loopback | Last | Combined |
|------|-----------|--------|---------------|------|----------|
| 1 | H2D read (or loopback D2D recv) | Upstream D2D recv | Upstream D2D recv | Upstream D2D recv | H2D read |
| 2 | Compute | Compute | Compute | Compute | Compute |
| 3 | Downstream D2D send | Downstream D2D send | D2H write + Loopback D2D send | D2H write | D2H write |

### What Breaks: Socket Creation Order Dependencies

PipelineBlock creates sockets in order: D2D upstream, D2D downstream, then H2D/D2H. Cross-mesh D2D sockets block until the peer creates the matching endpoint. If hosts use different creation orders, both deadlock waiting for each other's D2D sockets. PipelineBlock prevents this by sequencing creation based on stage index.

### What Breaks: Distributed Context Size Mismatch

PipelineBlock reads the process count via `ttnn.distributed_context_get_size()` during construction. If the number of launched processes does not match the expected count in the pipeline configuration, socket endpoint addressing fails silently. The distributed context returns rank information for a smaller set of peers than the pipeline expects, causing later cross-mesh socket connections to time out.

---

## Key Takeaways

1. `SocketInterface` wraps `MeshSocket` + `MeshWrapper` for pipeline-aware D2D communication. Never create raw MeshSocket objects in pipeline code.

2. `HostInterface` manages H2D + D2H lifecycle and belongs only on terminal stages (First for H2D, Last for D2H). Both H2D modes require vIOMMU; HOST_PUSH is used exclusively in DeepSeek V3.

3. `SocketConnection` is strictly 1:1. Reusing a sender or receiver core within the same SocketConfig causes a validation error.

4. `HostIoPlacement` assigns four distinct cores for H2D, D2H, D2D forwarding, and loopback to prevent core reuse conflicts.

5. `LoopbackConfig` has three modes: `fabric_loopback` (production D2D path), `host_loopback` (debug/fallback via host memory), and `no_loopback` (single-pass inference). The mode comparison matrix provides the selection criteria.

6. Dispatch order within a stage is always recv-compute-send. Violating this sequence causes deadlocks or buffer overwrites.

7. Socket creation order across hosts must be identical. PipelineBlock enforces this via stage-index-based sequencing to avoid cross-host deadlocks.

---

**Previous:** [7.1 Pipeline Architecture and Stages](./01_pipeline_architecture_and_stages.md) | **Next:** [7.3 End-to-End Inference Walkthrough](./03_end_to_end_inference_walkthrough.md) | **Up:** [Chapter 7 Index](./index.md)
