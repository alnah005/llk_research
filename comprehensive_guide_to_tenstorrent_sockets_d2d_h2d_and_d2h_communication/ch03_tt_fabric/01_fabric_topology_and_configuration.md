# 3.1 -- Fabric Topology and Configuration

TT-Fabric is the inter-chip communication network that transports D2D socket data between mesh devices. The user never programs TT-Fabric directly -- socket configuration (covered in [Chapter 2](../ch02_d2d_mesh_socket/index.md)) is the interface. But understanding how TT-Fabric organizes chips, routes packets, and manages channels is essential for reasoning about performance, diagnosing stalls, and planning multi-mesh deployments. This section covers the fabric architecture from physical links up through routing and initialization.

---

## 3.1.1 TT-Fabric Architecture: Four-Layer Stack

TT-Fabric can be understood as a four-layer stack, each layer building on the one below:

```
  +-----------------------------------------------+
  | APPLICATION LAYER                              |
  | MeshSocket send_async / recv_async             |
  | SocketConnection defines endpoints             |
  | Socket-level FIFO flow control (bytes_sent/    |
  |   bytes_acked)                                 |
  +-----------------------------------------------+
  | CONNECTION LAYER                               |
  | Fabric node ID assignment                      |
  | MGD cross-mesh bridging                        |
  | Virtual channel multiplexing                   |
  +-----------------------------------------------+
  | ROUTING LAYER                                  |
  | FABRIC_2D dimension-ordered routing (X-then-Y) |
  | Routing tables on ethernet cores               |
  +-----------------------------------------------+
  | PHYSICAL LAYER                                 |
  | Blackhole chips, ethernet links                |
  | ~12.5 GB/s per link per direction              |
  | Credit-based link-level flow control           |
  +-----------------------------------------------+
```

### User vs Runtime Responsibility

| Concern | User Configures | Runtime Handles |
|---------|----------------|-----------------|
| Endpoints | `SocketConnection` with `MeshCoreCoord` | Maps coordinates to fabric node IDs and physical links |
| Topology | `ttnn.FabricConfig.FABRIC_2D` | Builds routing tables, discovers links, configures credits |
| Memory | `SocketMemoryConfig` (L1/DRAM, FIFO size) | Allocates fabric-level buffers on ethernet cores |
| Flow control | Socket-level FIFO (bytes_sent / bytes_acked) | Link-level credit-based flow control per hop |
| Routing | Not exposed | X-then-Y dimension-ordered routing |
| Multi-mesh | Not exposed | MGD gateway selection and cross-mesh forwarding |

---

## 3.1.2 Physical Layer: Blackhole Galaxy

The Galaxy system is built from **Blackhole** chips arranged in a 2D grid connected by ethernet links.

| Parameter | Value |
|-----------|-------|
| Chip generation | Blackhole |
| Chips per Galaxy | 32 |
| Physical grid | 4x8 (4 rows, 8 columns) |
| Ethernet bandwidth | ~12.5 GB/s per link per direction (~100 Gbps) |
| AI clock frequency | 1.35 GHz |
| L1 SRAM per core | 1464 KB |
| NOC word width | 512 bits (64 bytes) |
| Minimum page size | 64 bytes (one NOC word) |
| SRAM write alignment | 16 bytes |

Each chip connects to its immediate neighbors via dedicated ethernet links. Cross-mesh connections use additional links between edge chips of neighboring meshes.

```
  Blackhole Galaxy: 32 chips in 4x8 grid

  Col: 0    1    2    3    4    5    6    7
  R0  |0,0 |0,1 |0,2 |0,3 |0,4 |0,5 |0,6 |0,7 |
  R1  |1,0 |1,1 |1,2 |1,3 |1,4 |1,5 |1,6 |1,7 |
  R2  |2,0 |2,1 |2,2 |2,3 |2,4 |2,5 |2,6 |2,7 |
  R3  |3,0 |3,1 |3,2 |3,3 |3,4 |3,5 |3,6 |3,7 |
       <-- Mesh A (4x4) -->  <-- Mesh B (4x4) -->

  Adjacent chips connected by ~12.5 GB/s ethernet links
```

---

## 3.1.3 FABRIC_2D Routing: X-then-Y Dimension-Ordered Routing

`FABRIC_2D` implements **dimension-ordered routing**: packets first traverse the X dimension (columns), then the Y dimension (rows). This deterministic routing avoids deadlocks without requiring complex arbitration.

### Routing Example

A packet from chip `(0,0)` to chip `(2,3)`:

```
  Step 1: Route in X (columns 0 -> 3)
  (0,0) -> (0,1) -> (0,2) -> (0,3)       3 X-hops

  Step 2: Route in Y (rows 0 -> 2)
  (0,3) -> (1,3) -> (2,3)                 2 Y-hops

  Total: 3 + 2 = 5 hops
```

```
  +-----+-----+-----+-----+
  |*0,0*|>0,1>|>0,2>|>0,3 |   X-phase: move right
  +-----+-----+-----+--+--+
  | 1,0 | 1,1 | 1,2 |v1,3 |   Y-phase: move down
  +-----+-----+-----+--+--+
  | 2,0 | 2,1 | 2,2 |*2,3*|   Destination reached
  +-----+-----+-----+-----+
  | 3,0 | 3,1 | 3,2 | 3,3 |
  +-----+-----+-----+-----+

  * = source/destination    > = X-hop    v = Y-hop
```

### Hop Count Formula

For a packet traveling from `(r1, c1)` to `(r2, c2)`:

```
total_hops = |c2 - c1| + |r2 - r1|
```

This is the Manhattan distance. The X-then-Y ordering is fixed -- packets never take a shorter "diagonal" path.

Dimension-ordered routing guarantees deadlock freedom because circular dependencies in the link graph cannot form. The trade-off is longer paths than adaptive routing, but determinism simplifies flow control and bandwidth analysis.

---

## 3.1.4 Initialization Sequence: MeshShape and Fabric Setup

The fabric must be fully initialized before any `MeshSocket` is created. The initialization sequence is strict and must be followed by all processes in an SPMD deployment.

```python
# Source: test_multi_mesh.py

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)   # Step 1: before device open
mesh_shape = ttnn.MeshShape(4, 4)
device = ttnn.open_mesh_device(mesh_shape=mesh_shape)  # Step 2: fabric + context init
socket = ttnn.MeshSocket(device, socket_config)        # Step 3: safe to create sockets
```

| Stage | Call | Effect |
|-------|------|--------|
| 1 | `set_fabric_config(FABRIC_2D)` | Registers the desired topology. No hardware changes yet. |
| 2 | `open_mesh_device(mesh_shape)` | Discovers chips, builds the mesh grid, programs routing tables on ethernet cores, establishes credit-based flow control, initializes `DistributedContext` for SPMD |
| 3 | `MeshSocket(device, config)` | Resolves `SocketConnection` endpoints to fabric routes, allocates FIFO buffers, connects with peer |

```
  Rank 0                              Rank 1
  ============================        ============================
  set_fabric_config(FABRIC_2D)        set_fabric_config(FABRIC_2D)
  open_mesh_device(4x4)              open_mesh_device(4x4)
    |-- discover chips, build            |-- discover chips, build
    |   routing tables, init             |   routing tables, init
    |   DistributedContext               |   DistributedContext
    +---- fabric ready, synced ----------+
  MeshSocket(device, config)          MeshSocket(device, config)
    |-- resolve routes, connect          |-- resolve routes, connect
```

**Warning:** All processes must call `set_fabric_config` with the same argument. Mismatched configs cause a deadlock during `open_mesh_device`.

---

## 3.1.5 Fabric Node ID Assignment

Each chip in the mesh receives a unique **fabric node ID** that TT-Fabric uses internally for routing decisions. The runtime maintains `fabric_node_id_map_` -- an array of maps from `MeshCoordinate` to `FabricNodeId`, indexed by `SocketEndpoint` (one mapping per side: sender and receiver).

### get_fabric_node_id

The `get_fabric_node_id` function takes two parameters:

```cpp
FabricNodeId get_fabric_node_id(SocketEndpoint endpoint, const MeshCoordinate& coord);
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint` | `SocketEndpoint` | Whether this is the sender or receiver side |
| `coord` | `MeshCoordinate` | The logical device coordinate within the mesh |

The function looks up `coord` in `fabric_node_id_map_` and returns the corresponding fabric node ID. This ID is used internally when constructing packet headers and programming routing tables.

Fabric node IDs are assigned sequentially during `open_mesh_device` in row-major order: `(0,0)` -> 0, `(0,1)` -> 1, ..., `(3,3)` -> 15. The user does not interact with fabric node IDs directly -- the user specifies `MeshCoordinate` values in `SocketConnection`, and the runtime translates them via `get_fabric_node_id`.

---

## 3.1.6 MGD Cross-Mesh Bridging

When a socket connection spans two different meshes (e.g., Mesh A and Mesh B in a Galaxy), packets must cross the mesh boundary. **MGD (Mesh Gateway Device)** nodes are designated edge chips that bridge between meshes.

```
  Mesh A (4x4)        MGD Links         Mesh B (4x4)
  +--+--+--+--+                         +--+--+--+--+
  |  |  |  | G|========================|G |  |  |  |
  +--+--+--+--+   bridge ethernet      +--+--+--+--+
  |  |  |  | G|========================|G |  |  |  |
  +--+--+--+--+                         +--+--+--+--+
  |  |  |  |  |                         |  |  |  |  |
  +--+--+--+--+                         +--+--+--+--+
  G = MGD gateway chip
```

| Connection Type | MGD Required? | Path |
|----------------|---------------|------|
| Same mesh, same chip | No | NOC only |
| Same mesh, different chip | No | Intra-mesh ethernet links |
| Different meshes, same host | Yes | Intra-mesh links -> MGD -> bridge links -> MGD -> intra-mesh links |
| Different meshes, different hosts | Yes | Same as above, but bridge links cross host boundary |

### User Impact

MGD bridging is **transparent to the user**. The user specifies sender and receiver `MeshCoreCoord` values in `SocketConnection`; the runtime determines whether the connection crosses a mesh boundary and automatically routes through MGD nodes. The `distributed_context` (a `shared_ptr<DistributedContext>`, default `nullptr` in `SocketConfig`) provides the cross-host communication backbone when meshes span multiple hosts.

---

## 3.1.7 Fabric Channels and Virtual Channels

Each physical ethernet link supports multiple **virtual channels** (VCs) that allow concurrent transfers to share the same link without head-of-line blocking.

```
  Physical Ethernet Link (~12.5 GB/s per direction)
  +--------------------------------------------------+
  | VC 0:  Socket A data ------>                     |
  | VC 1:  Socket B data ------>                     |
  | VC 2:  Control traffic ----->                    |
  +--------------------------------------------------+
```

- **Bandwidth sharing with independent flow control:** Multiple VCs share the physical link bandwidth, but each VC has its own credit pool. A stalled VC does not block other VCs on the same link.
- **Runtime-managed:** The runtime assigns VCs during socket creation and manages per-VC credits. The user does not select channels or configure credit pools.

TT-Fabric's VC-level flow control and the socket-level FIFO flow control (bytes_sent / bytes_acked) operate independently. See [Section 3.2.4](./02_socket_to_fabric_mapping.md) for a detailed interaction diagram.

---

## Key Takeaways

1. **TT-Fabric is the transport layer; sockets are the API.** Users configure endpoints and memory through `SocketConfig`; the runtime handles routing, channel assignment, and link-level flow control.

2. **Blackhole Galaxy: 32 chips, 4x8 grid.** Each adjacent chip pair is connected by an ethernet link at ~12.5 GB/s per direction. AI clock runs at 1.35 GHz with 1464 KB L1 per core.

3. **FABRIC_2D uses deterministic X-then-Y routing.** Hop count equals Manhattan distance: `|delta_col| + |delta_row|`. This guarantees deadlock freedom.

4. **Initialization order is strict.** `set_fabric_config(FABRIC_2D)` must precede `open_mesh_device`, which must precede `MeshSocket` creation. All SPMD processes must use the same fabric config.

5. **MGD bridging is automatic.** Cross-mesh connections route through gateway chips without any user configuration. The user only specifies logical coordinates.

---

**Previous:** [`index.md`](./index.md)
**Next:** [`02_socket_to_fabric_mapping.md`](./02_socket_to_fabric_mapping.md)
**Up:** [Chapter 3 Index](./index.md)
