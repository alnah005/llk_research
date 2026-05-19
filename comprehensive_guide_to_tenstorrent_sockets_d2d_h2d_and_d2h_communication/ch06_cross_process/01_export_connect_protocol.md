# 6.1 -- Export/Connect Protocol

When a serving architecture splits work across OS processes -- one process managing device resources, another feeding input tokens, a third consuming output logits -- the socket system needs a way to share connection state across process boundaries. The export/connect protocol provides this: the owner process serializes socket state into a flatbuffer descriptor in shared memory, and a connector process reconstructs a functional handle from that descriptor without needing a MetalContext.

---

## 6.1.1 The Cross-Process Problem

A `MetalContext` owns the device: it manages L1 allocations, PCIe TLB mappings, sub-device configurations, and kernel dispatch. Creating a second `MetalContext` for the same device in another process would conflict with the first. Yet the second process needs to transfer data over sockets that were allocated by the first. The constraint is precise: the connector process needs just enough information to issue PCIe transactions to the correct addresses, without any ability to reconfigure the device.

> **What breaks:** If a second process constructs a new H2DSocket targeting the same core, it gets independent counters (`bytes_sent`, `bytes_acked`). Two writers with independent counters will corrupt each other's data in the FIFO. There is no runtime guard -- the corruption is silent.

### Decision Tree: Should You Use Export/Connect?

```
  Do you need socket access from a different OS process?
  |
  +-- NO --> Use the socket directly. No export needed.
  |
  +-- YES --> Is this an H2D or D2H socket?
              |
              +-- H2D or D2H --> Use export_descriptor() + connect()
              |
              +-- D2D (MeshSocket) --> Cannot use export/connect.
                                       Use DistributedContext + ranks instead.
```

---

## 6.1.2 Protocol Steps

```
  OWNER PROCESS                         CONNECTOR PROCESS
  =============                         =================
  1. Create socket (H2D or D2H)
     - Allocates device FIFO, pins host memory, configures TLB
  2. set_page_size() -- must precede export
  3. export_descriptor(socket_id)
     - Serializes state to flatbuffer
     - Writes to /dev/shm/<socket_id>
                                        4. connect(socket_id, timeout_ms)
                |                           - Polls /dev/shm for file
           /dev/shm/<socket_id>             - Deserializes flatbuffer
                +-------------------------> - Maps PCIe resources (UMD only)
                                            - Returns socket handle
  5. Both operate on same FIFO
  6. Teardown: connector unmaps; owner unlinks /dev/shm, deallocates device memory
```

---

## 6.1.3 export_descriptor() and connect() APIs

```python
# Owner process
h2d = ttnn.H2DSocket(mesh_device, recv_core, buffer_type, fifo_size, h2d_mode)
h2d.set_page_size(page_size)
descriptor_path = h2d.export_descriptor("my_h2d_socket")

# Connector process (separate OS process, no MetalContext)
h2d_remote = ttnn.H2DSocket.connect("my_h2d_socket", timeout_ms=5000)
```

`export_descriptor()` serializes connection state into a flatbuffer, writes it to `/dev/shm/<socket_id>`, and returns the path. The `socket_id` must be unique across all sockets on the host.

`connect()` polls `/dev/shm/<socket_id>` until the file appears or `timeout_ms` expires, deserializes the flatbuffer, maps PCIe resources via UMD, and returns a socket handle. The handle does NOT require MetalContext.

> **What breaks:** Calling `export_descriptor()` before `set_page_size()` exports a descriptor with an unconfigured page size. The connector inherits this incomplete state and produces misaligned data in the FIFO, causing the device kernel to read garbage at page boundaries.

| `connect()` Condition | Behavior |
|----------------------|----------|
| File appears within timeout | Returns handle |
| File does not appear | Raises timeout exception |
| Invalid flatbuffer or PCIe unavailable | Raises exception |

---

## 6.1.4 Flatbuffer Contents by Socket Type

**H2D descriptor fields:**

| Field | Purpose |
|-------|---------|
| PCIe device ID | Identifies the target chip (BDF) |
| `recv_core` (MeshCoreCoord) | Target Tensix core on device |
| `h2d_mode` | HOST_PUSH or DEVICE_PULL |
| `fifo_base_addr`, `fifo_size`, `page_size` | FIFO geometry |
| `bytes_sent_l1_addr`, `bytes_acked_l1_addr` | Flow-control counter locations in device L1 |
| `pinned_host_addr` | Ack mirror (HOST_PUSH) or data FIFO (DEVICE_PULL) |
| `tlb_params`, PCIe BAR info | TLB window for raw PCIe writes |

**D2H descriptor fields:**

| Field | Purpose |
|-------|---------|
| PCIe device ID | Identifies the source chip |
| `sender_core` (MeshCoreCoord) | Source Tensix core |
| `pinned_fifo_addr`, `fifo_size`, `page_size` | Host-pinned FIFO geometry |
| `bytes_sent_host_addr` | Host-pinned address where device writes `bytes_sent` via NOC write |
| `bytes_acked_l1_addr` | Device L1 address where host writes `bytes_acked` via TLB write |
| `config_buffer_addr` | Device L1 config buffer address |
| `tlb_params` | TLB window for host-to-device ack writes |

**D2D: no descriptor.** D2D sockets do not use export/connect; cross-mesh communication is managed entirely by TT-Fabric.

The flatbuffer does NOT contain MetalContext state, host-side buffer pointers, tensor metadata, or authentication tokens.

---

## 6.1.5 PCIeCoreWriter for H2D Connectors

When an H2D connector calls `write()`, it does not use the same code path as the owner. The owner has full TLB management through MetalContext. The connector uses a `PCIeCoreWriter` -- a lightweight writer that performs TLB-mapped PCIe writes using only the BAR offset and TLB information extracted from the flatbuffer.

```
  H2D Owner write() path:              H2D Connector write() path:
  +---------------------+              +---------------------+
  | MetalContext         |              | PCIeCoreWriter      |
  | Full TLB management |              | BAR offset from     |
  | Device allocator     |              |   flatbuffer        |
  | write() -> TLB      |              | TLB info from       |
  +---------------------+              |   flatbuffer        |
                                       | write() -> raw PCIe |
                                       +---------------------+
```

The PCIeCoreWriter:
- Opens the PCIe device file (`/dev/tenstorrent/<N>`) using the BDF from the flatbuffer
- Maps the TLB window into the connector's address space
- Performs the same physical PCIe posted writes as the owner path
- Cannot allocate new device memory, reconfigure TLBs, or dispatch kernels
- Is stored as `pcie_writer_instance_` in the H2DSocket when `is_owner_ == false`

For D2H connectors, the situation is simpler: the connector maps the host-pinned FIFO region using the physical address from the descriptor. Reading `bytes_sent` and FIFO data is a local memory read with no PCIe traffic. The connector writes `bytes_acked` to device L1 via TLB write through the PCIe BAR.

> **What breaks:** If both the owner and the connector call `write()` concurrently without external synchronization, their `bytes_sent` updates will race. The FIFO is single-producer by design. Cross-process sharing is intended for handoff (owner stops writing, connector takes over), not concurrent writing.

---

## 6.1.6 Handshake Ordering Requirements

```
  Owner: create socket --> set_page_size() --> export_descriptor(id)
         |                                          |
         |                          /dev/shm/id becomes visible
         |                                          v
  Connector:                                  connect(id)
  ...data transfer...
  Connector: destroy handle  <-- MUST happen before owner destroys
  Owner: destroy socket
```

| Violation | Symptom |
|-----------|---------|
| Connector connects before owner exports | Timeout after `timeout_ms` |
| Owner exports before `set_page_size()` | Connector inherits page_size=0; silent corruption |
| Concurrent `write()` from both | `bytes_sent` race; FIFO corruption |
| Owner destructs while connector active | Segfault or silent corruption from freed memory |
| Owner re-configures after export | Connector silently operates on stale parameters |

> **What breaks:** The most insidious failure is the owner re-configuring after export. Since `export_descriptor()` is a snapshot, the connector's view diverges. There is no version check or invalidation protocol.

### Ordering Decision Tree

```
  Is the owner socket fully initialized (created + page_size set)?
  |
  +-- NO --> Do NOT call export_descriptor() yet.
  |
  +-- YES --> Has export_descriptor() been called?
              |
              +-- NO --> Call export_descriptor(). Signal connector.
              |
              +-- YES --> Is the connector done?
                          |
                          +-- NO --> Do NOT destroy the owner socket yet.
                          +-- YES --> Safe to destroy owner socket.
```

---

## 6.1.7 Flatbuffer File Lifecycle in /dev/shm

| Property | Value |
|----------|-------|
| Backing store | RAM (tmpfs), not disk |
| Persistence | Survives process exit; cleared on reboot |
| Creator / Unlinker | Owner process (unlinks in destructor if exported) |
| Permissions | Inherited from process umask (typically 0644) |

**Stale file risk:** If the owner crashes without running its destructor (SIGKILL, OOM), the `/dev/shm` file persists with addresses pointing to deallocated device memory. A connector calling `connect()` succeeds but operates on freed memory. See Section 6.3 for mitigation strategies.

---

## 6.1.8 Production Pattern: Barrier-Gated Export

In DeepSeek V3's PipelineBlock, the distributed context barrier ensures handshake ordering across ranks:

```python
# Process A (rank 0)
h2d = ttnn.H2DSocket(mesh_device, recv_core, BufferType.L1, fifo_size, H2DMode.HOST_PUSH)
h2d.set_page_size(page_size)
h2d.export_descriptor("stage0_h2d")
ttnn.distributed_context_barrier()   # All ranks synchronize here

# Process B (rank 1)
ttnn.distributed_context_barrier()   # Waits until rank 0 has exported
h2d_remote = ttnn.H2DSocket.connect("stage0_h2d", timeout_ms=5000)
```

The barrier guarantees that by the time Process B calls `connect()`, the descriptor file exists. The `timeout_ms` is a safety net for abnormal delays, not the primary synchronization mechanism.

---

## Key Takeaways

- `export_descriptor()` serializes socket state into a flatbuffer in `/dev/shm/<socket_id>`. The descriptor is a point-in-time snapshot; reconfiguration after export creates silent divergence.
- `connect()` reconstructs a socket handle from the flatbuffer using only UMD -- no MetalContext required.
- Only H2D and D2H sockets support export/connect; D2D sockets use DistributedContext and ranks over TT-Fabric.
- H2D connectors use `PCIeCoreWriter` for raw PCIe writes; D2H connectors map pinned memory and write `bytes_acked` to device L1 via TLB.
- Ordering is strict: initialize and export before connect; connector destroys before owner.
- The FIFO is single-producer by design -- cross-process sharing is a handoff, not a fan-in.
- Production deployments use distributed context barriers for handshake synchronization, with `timeout_ms` as a safety net.

---

**Previous:** [6.0 Chapter Index](./index.md) | **Next:** [6.2 Owner and Connector Lifecycle](./02_owner_connector_lifecycle.md) | **Up:** [Chapter Index](./index.md)
