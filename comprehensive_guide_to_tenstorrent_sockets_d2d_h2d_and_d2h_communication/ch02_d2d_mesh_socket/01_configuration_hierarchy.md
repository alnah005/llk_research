# 2.1 -- Configuration Hierarchy

A D2D MeshSocket is fully defined by a single `SocketConfig` object. That object is not monolithic -- it is assembled from four layers of configuration structs, each responsible for one concern. This section walks through each layer bottom-up, provides field-by-field reference tables, explains the two `SocketConfig` constructor overloads, and concludes with a complete assembly walkthrough that builds a production-ready config from scratch.

---

## 2.1.1 The Building-Block Model

The four configuration structs form a composition hierarchy. Each layer answers a distinct question:

```
  Layer 4 (top)     SocketConfig
                    "Who are the endpoints, what memory do they use,
                     and which process owns each side?"
                         |
                    composes
                    |            |
  Layer 3      SocketMemory   SocketConnection[]
               Config         (vector)
               "Where do       "Which sender core
                buffers         talks to which
                live?"          receiver core?"
                    |                |
                 standalone      composes
                                     |
  Layer 2                    SocketConnection
                             "One sender-receiver
                              core pair"
                                  |
                              composes
                              |        |
  Layer 1 (bottom)    MeshCoreCoord  MeshCoreCoord
                      (sender)       (receiver)
                      "A specific core on a
                       specific device in the mesh"
```

Reading bottom-up: a `MeshCoreCoord` identifies one core on one device. A `SocketConnection` pairs a sender `MeshCoreCoord` with a receiver `MeshCoreCoord`. A `SocketMemoryConfig` specifies buffer placement and sizing. A `SocketConfig` bundles a vector of connections, a memory config, and process-identity fields (mesh IDs or SPMD ranks) into the single object that the `MeshSocket` constructor accepts.

---

## 2.1.2 Layer 1: MeshCoreCoord -- Naming a Core in a Mesh

`MeshCoreCoord` is the atomic addressing unit. It uniquely identifies a single Tensix core within a mesh device by combining two coordinates:

- **`device_coord`** (`MeshCoordinate`) -- identifies which device within the mesh grid
- **`core_coord`** (`CoreCoord`) -- identifies which core on that device

### C++ Definition (mesh_socket.hpp)

```cpp
struct MeshCoreCoord {
    MeshCoordinate device_coord = MeshCoordinate(0);
    CoreCoord core_coord = CoreCoord(0, 0);
};
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `device_coord` | `MeshCoordinate` | `MeshCoordinate(0)` | The device's position within the mesh. For a 4x4 mesh, valid coordinates range from `(0,0)` to `(3,3)`. The coordinate is logical (assigned by the runtime), not physical. |
| `core_coord` | `CoreCoord` | `CoreCoord(0, 0)` | The core's position within the device. This is a logical worker core coordinate. `(0, 0)` is typically the first available worker core. |

### Python Construction

```python
# Identify core (0, 0) on device at mesh coordinate (2, 1)
coord = ttnn.MeshCoreCoord(
    ttnn.MeshCoordinate(2, 1),    # device position in mesh
    ttnn.CoreCoord(0, 0)          # core position on device
)
```

### Coordinate Semantics

```
  Mesh (4x4 devices)
  +------+------+------+------+
  |(0,0) |(0,1) |(0,2) |(0,3) |   <-- MeshCoordinate(row, col)
  | core | core | core | core |
  | grid | grid | grid | grid |
  +------+------+------+------+
  |(1,0) |(1,1) |(1,2) |(1,3) |
  |      |      |      |      |
  +------+------+------+------+
  |  ...    ...    ...    ... |
  +------+------+------+------+

  Within device (0,0):
  +-------+-------+-------+---
  |(0,0)  |(1,0)  |(2,0)  |...   <-- CoreCoord(x, y)
  +-------+-------+-------+---
  |(0,1)  |(1,1)  |(2,1)  |...
  +-------+-------+-------+---
```

**Warning:** `MeshCoordinate` uses `(row, col)` order while `CoreCoord` uses `(x, y)` order. Swapping them silently addresses the wrong core and produces garbage data or hangs.

### Key Constraint

A single `MeshCoreCoord` can appear as a sender or receiver in at most one `SocketConnection` within a single socket context. The runtime enforces this at `MeshSocket` construction time: "Cannot reuse senders and receivers in a single socket context." If you need multiple sockets touching the same device, use different cores.

---

## 2.1.3 Layer 2: SocketConnection -- Binding Sender to Receiver

A `SocketConnection` establishes a strictly 1:1 mapping between one sender core and one receiver core. This is the fundamental channel primitive.

### C++ Definition

```cpp
struct SocketConnection {
    MeshCoreCoord sender_core;
    MeshCoreCoord receiver_core;
};
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `sender_core` | `MeshCoreCoord` | The core that will call `send_async`. Must not already appear as a sender or receiver in another connection within the same socket context. |
| `receiver_core` | `MeshCoreCoord` | The core that will call `recv_async`. Same uniqueness constraint as the sender. |

### Python Construction

```python
conn = ttnn.SocketConnection(
    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),  # sender
    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))   # receiver
)
```

### Why Connections Are 1:1

The 1:1 constraint reflects the FIFO data model described in [Chapter 1](../ch01_socket_overview/index.md). Each FIFO has exactly one writer and one reader. Multi-party patterns (one-to-many, many-to-one) are built by creating multiple connections within the same `SocketConfig` or by using multiple sockets.

```
VALID:                          INVALID (core reuse):

Core A ----> Core B             Core A ----> Core B
Core C ----> Core D             Core A ----> Core C   <-- A appears twice as sender
```

### Multiple Connections for Parallel Transfer

A common pattern iterates over all mesh coordinates to create one connection per device. For a 4x4 mesh, this produces 16 `SocketConnection` objects via `MeshCoordinateRange(mesh_shape)`:

```python
# Source: test_multi_mesh.py

sender_logical_coord = ttnn.CoreCoord(0, 0)
recv_logical_coord = ttnn.CoreCoord(0, 0)
socket_connections = []
for coord in ttnn.MeshCoordinateRange(mesh_shape):
    socket_connections.append(
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(coord, sender_logical_coord),
            ttnn.MeshCoreCoord(coord, recv_logical_coord)
        )
    )
```

For a 4x4 mesh, this produces 16 connections, one per device position:

```
  Sender Mesh (Rank 0)          Receiver Mesh (Rank 1)
  +----+----+----+----+         +----+----+----+----+
  |0,0 |0,1 |0,2 |0,3 |  --->  |0,0 |0,1 |0,2 |0,3 |
  +----+----+----+----+         +----+----+----+----+
  |1,0 |1,1 |1,2 |1,3 |  --->  |1,0 |1,1 |1,2 |1,3 |
  +----+----+----+----+         +----+----+----+----+
  |2,0 |2,1 |2,2 |2,3 |  --->  |2,0 |2,1 |2,2 |2,3 |
  +----+----+----+----+         +----+----+----+----+
  |3,0 |3,1 |3,2 |3,3 |  --->  |3,0 |3,1 |3,2 |3,3 |
  +----+----+----+----+         +----+----+----+----+
       core(0,0)                     core(0,0)
```

**Warning:** In this example, sender and receiver share the same `MeshCoreCoord` values, which may seem contradictory. This works because in the SPMD model, these coordinates are interpreted relative to different processes (ranks). The sender's coordinates resolve against the sender mesh, and the receiver's against the receiver mesh. They are not the same physical core.

---

## 2.1.4 Layer 3: SocketMemoryConfig -- Where Buffers Live

`SocketMemoryConfig` controls where the socket's data buffers are allocated and how large they are. It applies uniformly to all connections within a single `SocketConfig`.

### C++ Definition

```cpp
struct SocketMemoryConfig {
    BufferType socket_storage_type = BufferType::L1;
    uint32_t fifo_size = 0;
    std::optional<SubDeviceId> sender_sub_device = std::nullopt;
    std::optional<SubDeviceId> receiver_sub_device = std::nullopt;
};
```

### Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `socket_storage_type` | `BufferType` | `BufferType::L1` | Where FIFO buffers are allocated. `L1` places them in Tensix SRAM for lowest latency. `DRAM` places them in device DRAM for larger capacity. |
| `fifo_size` | `uint32_t` | `0` | Total FIFO buffer size in bytes. Determines the maximum amount of data in flight (unacknowledged) at any time. Larger values allow more pipeline overlap but consume more memory. |
| `sender_sub_device` | `optional<SubDeviceId>` | `nullopt` | If the device uses sub-device partitioning, this identifies which sub-device the sender buffer belongs to. Leave as `nullopt` for standard configurations. |
| `receiver_sub_device` | `optional<SubDeviceId>` | `nullopt` | Same as above, for the receiver side. |

### Python Construction

```python
# Source: test_multi_mesh.py

# L1-backed FIFO with 4096 bytes capacity
socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)
```

### Choosing L1 vs. DRAM

| Criterion | L1 | DRAM |
|-----------|----|------|
| **Latency** | Lowest (single-cycle SRAM access) | Higher (DRAM access latency) |
| **Capacity** | Limited (~1.5 MB per core, shared with compute) | Large (device DRAM, typically GBs) |
| **Use case** | Small, latency-sensitive transfers (activations, control tokens) | Large bulk transfers where capacity matters more than latency |
| **Compute contention** | FIFO competes with kernel scratch space | No contention with compute L1 |

For most inter-stage pipeline transfers, **L1 with a modest fifo_size (4096--16384 bytes)** is the standard choice. DRAM is reserved for scenarios where the FIFO must absorb large bursts without back-pressure. See [Chapter 5, File 1](../ch05_flow_control/01_memory_and_performance.md) for sizing guidance.

**Warning:** Setting `fifo_size` too small causes the FIFO to stall frequently, throttling throughput. Setting it too large wastes L1 capacity that compute kernels need. A good starting point is 2x the expected page size. Always set an explicit `fifo_size` for production workloads.

---

## 2.1.5 Layer 4: SocketConfig -- The Top-Level Assembly

`SocketConfig` is the top-level configuration object. It bundles connections, memory config, and endpoint identity into the single argument that the `MeshSocket` constructor accepts. It has **two constructor overloads** that correspond to the two D2D programming modes.

### C++ Definition

```cpp
struct SocketConfig {
    std::vector<SocketConnection> socket_connection_config;
    SocketMemoryConfig socket_mem_config;
    std::optional<tt::tt_fabric::MeshId> sender_mesh_id = std::nullopt;
    std::optional<tt::tt_fabric::MeshId> receiver_mesh_id = std::nullopt;
    multihost::Rank sender_rank{0};
    multihost::Rank receiver_rank{0};
    std::shared_ptr<multihost::DistributedContext> distributed_context = nullptr;
};
```

### Field Reference

| Field | Type | Default | Used In | Description |
|-------|------|---------|---------|-------------|
| `socket_connection_config` | `vector<SocketConnection>` | (required) | Both | The list of 1:1 core pairs. One entry per parallel FIFO channel. |
| `socket_mem_config` | `SocketMemoryConfig` | (required) | Both | Buffer placement and sizing, shared by all connections in this config. |
| `sender_mesh_id` | `optional<MeshId>` | `nullopt` | Intra-process | Identifies the sender mesh when both meshes are in the same process. |
| `receiver_mesh_id` | `optional<MeshId>` | `nullopt` | Intra-process | Identifies the receiver mesh. Set only for intra-process mode. |
| `sender_rank` | `Rank` | `0` | SPMD | The SPMD rank of the process that owns the sender side. |
| `receiver_rank` | `Rank` | `0` | SPMD | The SPMD rank of the process that owns the receiver side. |
| `distributed_context` | `shared_ptr<DistributedContext>` | `nullptr` | SPMD | The coordination backbone for rank-based pairing. Auto-initialized when mesh device is opened; leave as `nullptr` unless managing multiple independent contexts. |

### The Two Constructor Overloads

This is the **key decision point** for D2D socket programming. The overload you choose determines whether the socket operates within a single process or across SPMD processes.

```
  +---------------------------------------------+
  |          Which SocketConfig overload?        |
  +---------------------------------------------+
  |                                             |
  |   Both meshes in same process?              |
  |       |                    |                |
  |      YES                  NO                |
  |       |                    |                |
  |   Intra-process         SPMD                |
  |   (mesh_id overload)    (rank overload)     |
  |                                             |
  |   - Set sender_mesh_id   - Set sender_rank  |
  |   - Set receiver_mesh_id - Set receiver_rank|
  |   - No distributed_ctx   - Needs dist_ctx   |
  |   - Or use                - Both processes   |
  |     create_socket_pair()    run same program |
  +---------------------------------------------+
```

#### Overload 1: Intra-Process (mesh_id)

When both the sender mesh and receiver mesh exist within the same host process:

```python
# Intra-process: identify meshes by their MeshId
config = ttnn.SocketConfig(
    socket_connections,
    socket_mem_config,
    sender_mesh_id=sender_mesh.get_mesh_id(),
    receiver_mesh_id=receiver_mesh.get_mesh_id()
)
```

This overload sets `sender_mesh_id` and `receiver_mesh_id`. The runtime resolves both endpoints locally -- no inter-process coordination is needed.

#### Overload 2: Multi-Process SPMD (rank)

When sender and receiver are in different OS processes:

```python
# Source: test_multi_mesh.py

socket_config = ttnn.SocketConfig(
    socket_connections,      # vector of SocketConnection
    socket_mem_config,       # SocketMemoryConfig
    sender_rank,             # int: 0
    receiver_rank            # int: 1
)
```

This overload sets `sender_rank` and `receiver_rank`. The `DistributedContext` (auto-initialized when the mesh device is opened) provides the coordination backbone.

**Warning:** Mixing patterns -- setting both `mesh_id` and `rank` fields -- produces undefined behavior. Use one pattern or the other, never both.

| Scenario | Pattern | Why |
|----------|---------|-----|
| Single process, two mesh devices | Mesh IDs | Both meshes are local; no distributed context needed |
| Multi-process SPMD (one mesh per process) | Ranks | Each process only sees its own mesh; ranks identify peers |
| DeepSeek V3 multi-host pipeline | Ranks | Processes span hosts; SPMD is the only option |

---

## 2.1.6 Assembly Walkthrough: Building a Complete Config

This walkthrough builds a `SocketConfig` for a 4x4 mesh where every device connects core `(0, 0)` on the sender side to core `(0, 0)` on the receiver side, using L1 memory with a 4096-byte FIFO. The configuration targets a multi-process SPMD deployment (rank 0 sends, rank 1 receives).

### Step 1: Define the Mesh Shape

```python
mesh_shape = ttnn.MeshShape(4, 4)  # 4 rows, 4 columns = 16 devices
```

### Step 2: Build MeshCoreCoord Pairs (Layer 1)

```python
sender_logical_coord = ttnn.CoreCoord(0, 0)
recv_logical_coord = ttnn.CoreCoord(0, 0)
```

### Step 3: Assemble SocketConnections (Layer 2)

Iterate over every device coordinate in the mesh and create one `SocketConnection` per device:

```python
socket_connections = []
for coord in ttnn.MeshCoordinateRange(mesh_shape):
    socket_connections.append(
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(coord, sender_logical_coord),
            ttnn.MeshCoreCoord(coord, recv_logical_coord)
        )
    )
# Result: 16 SocketConnection objects (one per device in the 4x4 mesh)
```

The `MeshCoordinateRange` iterator yields all valid `MeshCoordinate` values for the given shape, visiting `(0,0)`, `(0,1)`, ..., `(3,3)`.

### Step 4: Configure Memory (Layer 3)

```python
socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)
```

This allocates a 4096-byte L1-backed FIFO for each connection. All 16 connections share the same memory configuration.

### Step 5: Compose into SocketConfig (Layer 4)

```python
socket_config = ttnn.SocketConfig(
    socket_connections,      # 16 connections
    socket_mem_config,       # L1, 4096 bytes
    0,                       # sender_rank = 0
    1                        # receiver_rank = 1
)
```

### The Complete Assembly Visualized

```
  MeshCoreCoord(device=(0,0), core=(0,0))  ---+
  MeshCoreCoord(device=(0,0), core=(0,0))  ---+--> SocketConnection[0]  --+
                                                                           |
  MeshCoreCoord(device=(0,1), core=(0,0))  ---+                           |
  MeshCoreCoord(device=(0,1), core=(0,0))  ---+--> SocketConnection[1]  --+
                                                                           |
  ...                                          ...                         |  16 connections
                                                                           |
  MeshCoreCoord(device=(3,3), core=(0,0))  ---+                           |
  MeshCoreCoord(device=(3,3), core=(0,0))  ---+--> SocketConnection[15] --+
                                                                           |
                                                                           v
  SocketMemoryConfig(L1, 4096)  ---+                                       |
                                   +--> SocketConfig(connections, mem,  <---+
  sender_rank=0, receiver_rank=1 --+     rank_0, rank_1)
```

### Step 6: Construct the MeshSocket

```python
device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
socket = ttnn.MeshSocket(device, socket_config)
```

The `MeshSocket` constructor reads the config, allocates FIFO buffers on each core, and establishes the TT-Fabric transport. After construction, the socket is ready for `ttnn.experimental.send_async` or `ttnn.experimental.recv_async` operations.

---

## 2.1.7 Common Configuration Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Cannot reuse senders and receivers in a single socket context" | Same `MeshCoreCoord` appears in two connections | Use distinct cores for each connection |
| Zero fifo_size | `SocketMemoryConfig` constructed with `fifo_size=0` | Set a nonzero FIFO size (minimum 4096 recommended) |
| Rank mismatch | `sender_rank` and `receiver_rank` do not match across processes | Ensure both processes use identical `SocketConfig` values |
| Missing DistributedContext | Using SPMD ranks but context is null | Verify that `ttnn.open_mesh_device` was called (context auto-initializes) |
| MeshCoordinate out of range | Device coordinate exceeds `MeshShape` bounds | Verify that coordinates are within `(0,0)` to `(rows-1, cols-1)` |
| Mixing mesh_id and rank fields | Both identification patterns set in same config | Use one pattern or the other, never both |

---

## 2.1.8 Validation Summary

| Level | Struct | Key Validation Rule |
|-------|--------|---------------------|
| 1 | MeshCoreCoord | device_coord within mesh bounds; core_coord is valid worker core |
| 2 | SocketConnection | sender and receiver endpoints valid; no core reuse across connections |
| 3 | SocketMemoryConfig | `fifo_size` > 0; `socket_storage_type` supported |
| 4 | SocketConfig | Non-empty connections; no core reuse; rank or mesh_id form (not both) |

Validation is performed at `MeshSocket` construction time. If any precondition is violated, the constructor throws. There is no deferred validation -- either the socket is fully valid at construction, or it fails immediately.

**Warning:** The no-core-reuse invariant is checked across the entire `socket_connection_config` vector. If you build connections from multiple sources and accidentally include overlapping cores, the error will surface only when the `MeshSocket` is constructed, not when the `SocketConfig` is assembled.

---

## Key Takeaways

1. **Four nested types.** `MeshCoreCoord` --> `SocketConnection` --> `SocketMemoryConfig` + connections --> `SocketConfig`. Each layer adds one dimension of the configuration.

2. **Connections are strictly 1:1.** One sender core, one receiver core, no sharing, no fan-out. Multi-core parallelism is achieved by adding more connections to the vector, not by reusing cores.

3. **Two identification patterns.** Use mesh IDs for intra-process setups (both meshes in one process). Use SPMD ranks for multi-process setups (one mesh per process). Never mix them.

4. **Memory config applies uniformly.** All connections in a `SocketConfig` share the same `SocketMemoryConfig`. If you need different FIFO sizes for different core pairs, create separate `SocketConfig` objects.

5. **Coordinate order matters.** `MeshCoordinate(row, col)` vs `CoreCoord(x, y)`. Getting these backwards is a common source of silent misconfiguration.

---

**Next:** [`02_send_recv_async_operations.md`](./02_send_recv_async_operations.md)
