# 3.2 -- Socket-to-Fabric Mapping

This section traces a D2D packet from sender to receiver, showing how `SocketConnection` maps to physical fabric routes, how multiple connections exploit parallel links, and how two flow control layers interact. It also covers bandwidth/latency analysis, PCIe topology, and failure modes.

---

## 3.2.1 D2D Packet Path: A Concrete Scenario

Consider a tensor transfer from core `(0, 0)` on device `(1, 0)` in Mesh A (rank 0) to core `(0, 0)` on device `(1, 2)` in Mesh B (rank 1). The two meshes are adjacent 4x4 sub-grids of a Galaxy, connected by MGD bridge links.

### Packet Trace

The runtime segments the tensor into pages (minimum 64 bytes, aligned to the 512-bit NOC word width). Each page follows this path:

1. **Sender core to local ethernet core** -- On-chip NOC from sender L1 to the ethernet core on chip `(1, 0)`.
2. **Intra-mesh X routing** -- Forwarded east through chips `(1, 1)`, `(1, 2)`, `(1, 3)` along the X dimension.
3. **MGD bridge crossing** -- At chip `(1, 3)` (MGD gateway), crosses to chip `(1, 4)` in Mesh B.
4. **Intra-mesh X routing (Mesh B)** -- Continues east to the destination column.
5. **Y routing (if needed)** -- Routes along the Y dimension if the destination row differs.
6. **Remote ethernet core to receiver core** -- On-chip NOC on the destination chip to the receiver core's L1 FIFO.

### Six-Stage Packet Path Table

| Stage | Location | Action | Transport |
|-------|----------|--------|-----------|
| 1. Source NOC | Sender chip, sender core -> ethernet core | NOC write from L1 to ethernet core buffer | On-chip NOC |
| 2. X-phase routing | Ethernet links along X dimension | Hop-by-hop forwarding through intermediate chips | Inter-chip ethernet |
| 3. MGD bridge | Edge chip -> bridge link -> remote edge chip | Cross mesh boundary (if inter-mesh) | Inter-mesh ethernet |
| 4. X-phase continued | Ethernet links in destination mesh | Continue X-phase forwarding (if needed) | Inter-chip ethernet |
| 5. Y-phase routing | Ethernet links along Y dimension | Turn from X to Y at the correct column | Inter-chip ethernet |
| 6. Destination NOC | Destination chip, ethernet core -> receiver core | NOC write from ethernet core buffer to receiver L1 | On-chip NOC |

### ASCII Visualization

```
  Sender Side (Mesh A)                          Receiver Side (Mesh B)
  +---------------------------+                 +---------------------------+
  | Chip (1,0)                |                 | Chip (1,2) [Mesh B]      |
  |                           |                 |                           |
  | +-------+    +----------+ |   Ethernet     | +----------+    +-------+ |
  | |Sender |    |Ethernet  | |   Links        | |Ethernet  |    |Recvr  | |
  | |Core   |--->|Core      |---> ... --->MGD--->|Core      |--->|Core   | |
  | |(0,0)  | NOC|          | |   (X-hops)     | |          | NOC|(0,0)  | |
  | +-------+    +----------+ |                 | +----------+    +-------+ |
  |                           |                 |                           |
  | L1 FIFO (send buffer)    |                 | L1 FIFO (recv buffer)    |
  +---------------------------+                 +---------------------------+
```

---

## 3.2.2 SocketConnection to Route Mapping

A `SocketConnection` specifies a sender `MeshCoreCoord` and a receiver `MeshCoreCoord`. The runtime translates these into a physical fabric route in two steps:

The `device_coord` field determines the inter-chip route. The runtime calls `get_fabric_node_id(endpoint, coord)` for both endpoints to obtain fabric node IDs, then computes the X-then-Y path. The `core_coord` field is resolved within the on-chip NOC only -- it determines the first and last hops but does not affect inter-chip routing.

### Mapping Summary

```
  SocketConnection
  +-------------------------------------+
  | sender:   MeshCoreCoord             |
  |   device_coord: (1, 0)  ---------> fabric route (inter-chip path)
  |   core_coord:   (0, 0)  ---------> NOC address (on-chip, first hop)
  |                                     |
  | receiver: MeshCoreCoord             |
  |   device_coord: (1, 2)  ---------> fabric route (inter-chip path)
  |   core_coord:   (0, 0)  ---------> NOC address (on-chip, last hop)
  +-------------------------------------+
```

### Cross-Mesh Connections in SPMD Mode

In SPMD mode, sender and receiver coordinates are interpreted relative to their respective meshes. The standard pattern from [Chapter 2](../ch02_d2d_mesh_socket/01_configuration_hierarchy.md) iterates over `MeshCoordinateRange(mesh_shape)` to create one connection per device, then sets `sender_rank=0` and `receiver_rank=1`. Because these ranks point to different meshes, coordinate `(1, 2)` on rank 0 maps to a physical chip in Mesh A, while the same `(1, 2)` on rank 1 maps to a different physical chip in Mesh B. The fabric routes packets across the boundary via MGD automatically.

---

## 3.2.3 Multi-Connection Parallelism

A single `SocketConfig` can contain multiple `SocketConnection` objects. When connections target different devices, their packets travel on different fabric links in parallel.

### Connection Topology Patterns

| Pattern | Description | Connections | Use Case |
|---------|-------------|-------------|----------|
| 1:1 single | One sender core to one receiver core | 1 | Control tokens, small metadata |
| 1:1 per device | One connection per mesh coordinate | N (one per device) | Sharded tensor transfer (standard pattern) |
| Many-to-one | Multiple senders, one receiver device | M | Gather/reduce from distributed sources |
| One-to-many | One sender device, multiple receivers | M | Broadcast/scatter to distributed targets |

### Parallel Fabric Links

When connections use different source and destination devices, the packets take different physical paths and do not compete for bandwidth:

```
  Connection 0: device (0,0) -> device (0,0)  [via path A]
  Connection 1: device (1,0) -> device (1,0)  [via path B]
  Connection 2: device (2,0) -> device (2,0)  [via path C]
  Connection 3: device (3,0) -> device (3,0)  [via path D]

  Paths A, B, C, D use different ethernet links --> full parallelism
```

When multiple connections share intermediate links, they share bandwidth via virtual channel multiplexing.

### Bisection Bandwidth

Bisection bandwidth measures aggregate bandwidth across the narrowest cut dividing the mesh in half. For a horizontal bisection of a 4x4 mesh (cutting between rows 1 and 2), the cut crosses 4 vertical links:

```
  Bisection bandwidth = 4 links x 12.5 GB/s = 50 GB/s per direction
```

Any traffic pattern crossing the midline is capped at 50 GB/s per direction. The same applies to a vertical bisection.

---

## 3.2.4 Dual-Layer Flow Control

D2D socket transfers involve two independent flow control mechanisms operating at different layers. Understanding their interaction is critical for diagnosing back-pressure and stalls.

- **Fabric-level (credits):** Each VC on each ethernet link maintains a credit count. A sender transmits only when it holds credits for the downstream buffer. Freed buffer space returns credits. This prevents overflow at any intermediate hop.
- **Socket-level (FIFO):** The socket maintains `bytes_sent` and `bytes_acked`. The sender writes when `bytes_sent - bytes_acked < fifo_size`. The receiver advances `bytes_acked` after consumption. This prevents application-level buffer overwrite.

### Interaction Diagram

```
  Sender Core                    Fabric Links              Receiver Core
  +------------------+          +---------------+         +------------------+
  | Socket Layer     |          |               |         | Socket Layer     |
  | bytes_sent: 8192 |          |               |         | bytes_acked: 4096|
  | fifo_size: 16384 |          |               |         | fifo_size: 16384 |
  |                  |          |               |         |                  |
  | Can send? YES    |          |               |         | Consumed 4096 B  |
  | (8192-4096<16384)|          |               |         | -> ack to sender |
  +--------+---------+          |               |         +--------+---------+
           |                    |               |                  ^
           v                    |               |                  |
  +--------+---------+          |               |         +--------+---------+
  | Fabric Layer     |   credit |               | credit  | Fabric Layer     |
  | Credits: 3       |--------->| hop 1 | hop 2 |-------->| Credits returned |
  | Can transmit? YES|          |               |         | Buffer freed     |
  +------------------+          +---------------+         +------------------+
```

When the receiver is slow to consume, `bytes_acked` falls behind --> the sender blocks at the socket layer while fabric credits flow normally. When a fabric link is congested, credits are exhausted and packets queue at intermediate hops --> `bytes_sent` stalls even though FIFO space is available, making the sender appear socket-blocked.

---

## 3.2.5 Bandwidth and Latency Analysis

Per-link bandwidth is ~12.5 GB/s per direction; bisection bandwidth for a 4x4 mesh is 50 GB/s per direction (4 links x 12.5 GB/s).

### Latency Formula

```
  t_total = t_source_noc + (N_hops x t_per_hop) + t_dest_noc + t_software
```

| Component | Scale | Notes |
|-----------|-------|-------|
| `t_source_noc` | Nanoseconds | Sender core to ethernet core |
| `N_hops x t_per_hop` | Tens of ns per hop | Inter-chip ethernet |
| `t_dest_noc` | Nanoseconds | Ethernet core to receiver core |
| `t_software` | Low microseconds | Packet headers, credit checks |

### Hardware Specs Affecting Transfer Performance

- **Page size minimum:** 64 bytes (512-bit NOC word width)
- **SRAM write alignment:** 16 bytes
- **AI clock:** 1.35 GHz (~0.74 ns cycle time)
- **L1 per core:** 1464 KB (shared with compute CBs; see [Chapter 5](../ch05_flow_control_and_memory/index.md))

---

## 3.2.6 PCIe Topology Brief

While D2D sockets use TT-Fabric ethernet links, H2D and D2H sockets use PCIe for host-device communication. This section provides a brief comparison; full PCIe socket coverage is in [Chapter 4](../ch04_host_device_sockets/index.md).

### D2D vs PCIe Comparison

| Property | D2D (TT-Fabric) | H2D / D2H (PCIe) |
|----------|-----------------|-------------------|
| Transport | Ethernet links between chips | PCIe lanes between host CPU and chip |
| Bandwidth | ~12.5 GB/s per link per direction | ~16 GB/s (Gen4 x8, 4 high-BW chips); ~2 GB/s (Gen4 x1, 28 chips) |
| Latency | Low (on-fabric, no host involvement) | Higher (crosses PCIe bus, host memory) |
| Direction | Device-to-device | Host-to-device or device-to-host |
| Flow control | Fabric credits + socket FIFO | Socket FIFO only (PCIe handles link-level) |
| High-BW chip designation | N/A | ASIC 6 (the 4 high-bandwidth PCIe chips) |
| vIOMMU required | No | Yes (all H2D/D2H modes, including HOST_PUSH) |
| Multi-hop routing | Yes (X-then-Y across mesh) | No (direct PCIe link to host) |

In a Blackhole Galaxy, the 4 high-bandwidth chips are designated **ASIC 6**. These chips manage TLB windows for host writes and NOC-to-PCIe bridges for device-initiated transactions. The remaining 28 chips (Gen4 x1, ~2 GB/s) are suitable only for low-rate control traffic. The **vIOMMU** is required for all H2D and D2H socket modes, including HOST_PUSH (which needs vIOMMU for control-path signaling even though data writes are host-initiated) -- without it, socket creation fails.

---

## 3.2.7 Failure Modes and Diagnosis

Socket-level stalls and errors can originate from the fabric layer, the PCIe layer, or configuration mismatches. Categorizing the failure source is the first step in diagnosis.

### Category 1: Fabric Failures

| Failure | Symptom | Resolution |
|---------|---------|------------|
| Ethernet link down | `send_async` hangs indefinitely | Hardware repair; check fabric link diagnostics |
| Link congestion | Throughput below 12.5 GB/s | Spread connections across different paths |
| Credit exhaustion | Sender stalls despite FIFO space | Check for slow consumers downstream |

### Category 2: PCIe Failures (H2D/D2H)

| Failure | Symptom | Resolution |
|---------|---------|------------|
| vIOMMU not enabled | Socket creation fails with IOMMU error (all modes, including HOST_PUSH) | Enable vIOMMU in BIOS/kernel |
| PCIe link degradation | Reduced H2D/D2H throughput | Check link speed via `lspci`; reseat card |
| Pinned memory exhaustion | Allocation failure at socket creation | Reduce FIFO sizes or concurrent sockets |

### Category 3: Configuration Mismatches

| Failure | Symptom | Resolution |
|---------|---------|------------|
| Fabric not initialized | `MeshSocket` constructor throws | Call `set_fabric_config` before `open_mesh_device` |
| Rank mismatch | Both processes become SENDER | Ensure identical `SocketConfig` across processes |
| Mismatched fabric config | Deadlock at `open_mesh_device` | All processes must use `FABRIC_2D` |
| Coordinate out of range | Constructor throws | Verify coordinates within `MeshShape` bounds |

### Stall Diagnosis: Socket vs Fabric

When a transfer stalls, use this table to narrow the cause:

| Observation | Likely Layer | Action |
|-------------|-------------|--------|
| `bytes_sent` == `bytes_acked` and both are less than expected | Socket layer: sender has not sent data | Check sender logic; verify `send_async` was called |
| `bytes_sent` >> `bytes_acked` | Socket layer: receiver is not consuming | Check receiver logic; verify `recv_async` and downstream compute |
| `bytes_sent` stalled, FIFO has space (`bytes_sent - bytes_acked < fifo_size`) | Fabric layer: packets not being delivered | Check fabric link status and credit flow |
| `bytes_sent` stalled, FIFO is full (`bytes_sent - bytes_acked == fifo_size`) | Socket layer: receiver back-pressure | Increase `fifo_size` or speed up receiver consumption |
| Transfer completes but data is wrong | Routing or coordinate error | Verify `MeshCoreCoord` values; check for `(row,col)` vs `(x,y)` swap |

---

## Key Takeaways

1. **A D2D packet traverses six stages:** sender NOC, X-phase ethernet routing, MGD bridge (if cross-mesh), continued routing, Y-phase routing, and destination NOC. The `device_coord` determines the inter-chip route; the `core_coord` determines the on-chip NOC endpoints.

2. **Connections on different devices use different fabric links.** Multi-connection sockets achieve parallelism by spreading traffic across independent paths. Bisection bandwidth of a 4x4 mesh is 50 GB/s per direction (4 links x 12.5 GB/s).

3. **Two flow control layers operate independently.** Fabric-level credits prevent link buffer overflow at each hop. Socket-level FIFO (bytes_sent / bytes_acked) prevents application buffer overwrite. A stall at either layer manifests as a blocked sender.

4. **PCIe sockets (H2D/D2H) use a fundamentally different transport.** They traverse the PCIe bus via ASIC 6 and require vIOMMU. Detailed coverage is in [Chapter 4](../ch04_host_device_sockets/index.md).

5. **Categorize failures into fabric, PCIe, or configuration.** Fabric failures show as credit exhaustion or link-level stalls. PCIe failures show as IOMMU errors or throughput degradation. Configuration failures show as constructor exceptions or rank mismatches.

---

**Previous:** [`01_fabric_topology_and_configuration.md`](./01_fabric_topology_and_configuration.md)
**Up:** [Chapter 3 Index](./index.md)
**Next chapter:** [Chapter 4 -- H2D and D2H Sockets](../ch04_host_device_sockets/index.md)
