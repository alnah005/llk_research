# 2.3 -- SPMD and Intra-Process Programming

D2D MeshSocket supports two fundamentally different programming modes depending on whether the sender and receiver meshes reside in the same OS process or in separate processes. This section covers both modes in depth: multi-process SPMD with rank-based coordination via `DistributedContext`, and intra-process communication via `create_socket_pair`. Each mode has distinct setup requirements, a different `SocketConfig` overload, and its own canonical code pattern.

---

## 2.3.1 The Two Modes at a Glance

```
  +---------------------+                    +---------------------+
  | SPMD (Multi-Process) |                   | Intra-Process       |
  +---------------------+                    +---------------------+
  | Two OS processes     |                    | One OS process      |
  | Each owns one mesh   |                    | Owns both meshes    |
  | Rank-based pairing   |                    | MeshId-based pairing|
  | DistributedContext   |                    | create_socket_pair  |
  | required             |                    | (no ranks needed)   |
  | FABRIC_2D transport  |                    | Local memory path   |
  +---------------------+                    +---------------------+
       |                                           |
       v                                           v
  SocketConfig(conns, mem,                   SocketConfig(conns, mem,
    sender_rank=0,                             sender_mesh_id=...,
    receiver_rank=1)                           receiver_mesh_id=...)
                                             -- OR --
                                             create_socket_pair(
                                               sender, receiver,
                                               base_config)
```

### Decision Criteria

| Question | If Yes | If No |
|----------|--------|-------|
| Are sender and receiver meshes in different OS processes? | Use SPMD with ranks | Use intra-process |
| Do you need multi-host deployment? | Must use SPMD | Either mode works |
| Are both meshes opened by the same process? | Use intra-process | Depends on process topology |
| Do you want the simplest possible setup for testing? | `create_socket_pair` | N/A |

---

## 2.3.2 FABRIC_2D Setup -- The Mandatory First Step

Before any D2D socket can be created, the fabric configuration must be set to `FABRIC_2D`. This is a process-global setting that must happen before device open.

```python
# Source: test_multi_mesh.py

# MUST call before open_mesh_device
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
```

### What FABRIC_2D Does

`FABRIC_2D` initializes a two-dimensional Ethernet routing fabric across all connected mesh devices. It enables both row-wise and column-wise routing, which D2D sockets require for transfers across arbitrary mesh coordinates.

```
  FABRIC_2D topology:

  Mesh A (4x4)                  Mesh B (4x4)
  +--+--+--+--+                +--+--+--+--+
  |  |  |  |  |  Ethernet     |  |  |  |  |
  |  |  |  |  |<=============>|  |  |  |  |
  |  |  |  |  |  links        |  |  |  |  |
  +--+--+--+--+                +--+--+--+--+

  TT-Fabric provides 2D routing across and within meshes.
  Sockets use this fabric for packet transport.
```

**Warning:** Calling `open_mesh_device` before `set_fabric_config` will open devices without fabric support. Any subsequent socket creation will fail with a fabric initialization error. There is no way to retroactively enable fabric after device open -- you must close and reopen. See [Chapter 3, File 1](../ch03_tt_fabric/01_fabric_topology.md) for details on fabric configuration options.

---

## 2.3.3 Distributed Context -- Automatic SPMD Coordination

When you open a mesh device in a multi-process SPMD environment, TT-Metal automatically initializes a `DistributedContext` that assigns each process a rank. There is no separate `init_distributed()` call required.

```python
# Source: test_multi_mesh.py

device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

# At this point, distributed context is already initialized
if not ttnn.distributed_context_is_initialized():
    raise ValueError("Distributed context not initialized")
```

### Querying Context State

| Function | Returns | Description |
|----------|---------|-------------|
| `ttnn.distributed_context_is_initialized()` | `bool` | Whether the context has been initialized |
| `ttnn.distributed_context_get_rank()` | `int` (as string) | This process's rank in the group |
| `ttnn.distributed_context_get_size()` | `int` (as string) | Total number of processes in the group |
| `ttnn.distributed_context_barrier()` | None | Block until all processes reach this point |

Note that `get_rank()` and `get_size()` return values that need to be cast to `int`:

```python
# Source: test_multi_mesh.py

if int(ttnn.distributed_context_get_size()) != 2:
    raise ValueError("This test requires 2 processes to run")
```

### How Ranks Are Assigned

The distributed context uses the process launch order (typically via `tt-run`, `mpirun`, or a similar launcher). Process 0 gets rank 0, process 1 gets rank 1, and so forth.

```
  tt-run --rank-binding 4x4_multi_mesh_rank_binding.yaml python test_multi_mesh.py

  Process 0:                    Process 1:
    rank = 0                      rank = 1
    mesh_device = Mesh A          mesh_device = Mesh B
    (4x4, 16 devices)            (4x4, 16 devices)
```

---

## 2.3.4 How Rank Determines Endpoint Type

When a `MeshSocket` is constructed with a rank-based `SocketConfig`, the socket compares the current process's rank against the `sender_rank` and `receiver_rank` fields:

```
  SocketConfig:
    sender_rank = 0
    receiver_rank = 1

  Process with rank 0 --> MeshSocket becomes SENDER
  Process with rank 1 --> MeshSocket becomes RECEIVER
```

This is why the same `SocketConfig` object can be used by both processes. The config is declarative; the MeshSocket constructor reads the local rank and assigns the appropriate endpoint type.

```
  Rank Resolution Flow:

  SocketConfig         +  Local Rank    =  Endpoint Type
  +-----------------+     +----------+     +------------------+
  | sender_rank: 0  |     | rank: 0  | --> | SENDER           |
  | receiver_rank: 1|     +----------+     +------------------+
  +-----------------+
  | sender_rank: 0  |     +----------+     +------------------+
  | receiver_rank: 1|     | rank: 1  | --> | RECEIVER         |
  +-----------------+     +----------+     +------------------+
```

If the local rank matches neither `sender_rank` nor `receiver_rank`, the socket construction will fail.

---

## 2.3.5 Canonical SPMD Code Pattern

The following is the canonical SPMD pattern extracted from `test_multi_mesh.py`. The key branching uses `ttnn.distributed_context_get_rank()` -- not `get_socket_endpoint_type()`.

```python
# Source: test_multi_mesh.py

def run_multiprocess_workload():
    torch.manual_seed(0)

    # --- Phase 1: Setup (identical on all processes) ---
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh_shape = ttnn.MeshShape(4, 4)
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    # --- Phase 2: Validation (identical on all processes) ---
    if not ttnn.distributed_context_is_initialized():
        raise ValueError("Distributed context not initialized")
    if int(ttnn.distributed_context_get_size()) != 2:
        raise ValueError("This test requires 2 processes to run")

    # --- Phase 3: Configuration (identical on all processes) ---
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
    socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)
    sender_rank = 0
    receiver_rank = 1
    socket_config = ttnn.SocketConfig(
        socket_connections, socket_mem_config, sender_rank, receiver_rank
    )

    # --- Phase 4: Data preparation (identical on all processes) ---
    torch_input = torch.randn(1, 1, 1024, 1024, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input, device=device, layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            device, dims=(2, 3), mesh_shape=mesh_shape
        ),
    )

    def torch_op_chain(tensor):
        return torch.exp(torch.nn.functional.relu(tensor))

    # --- Phase 5: Rank-aware execution (DIVERGES by rank) ---
    if int(ttnn.distributed_context_get_rank()) == 0:
        send_socket = ttnn.MeshSocket(device, socket_config)
        ttnn.experimental.send_async(ttnn.relu(ttnn_input), send_socket)
    else:
        recv_socket = ttnn.MeshSocket(device, socket_config)
        upstream_input = ttnn.allocate_tensor_on_device(
            ttnn_input.spec, device
        )
        ttnn.experimental.recv_async(upstream_input, recv_socket)
        # Downstream compute uses received data
        torch_tensor = ttnn.to_torch(
            ttnn.from_device(ttnn.exp(upstream_input)),
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                device, mesh_shape=mesh_shape, dims=(2, 3)
            ),
        )
        # Validate against PyTorch reference
        if torch.allclose(torch_tensor, torch_op_chain(torch_input)):
            print("Test passed: Output tensor matches expected tensor")
        else:
            raise ValueError("Test failed: Output tensor does not match expected tensor")

    # --- Phase 6: Synchronization and cleanup (identical) ---
    ttnn.distributed_context_barrier()
    ttnn.close_device(device)
```

### Phase Breakdown

| Phase | Description | Rank-Dependent? |
|-------|-------------|-----------------|
| 1. Setup | Fabric config, device open | No |
| 2. Validation | Context checks | No |
| 3. Configuration | Socket config assembly (16 connections via `MeshCoordinateRange`) | No |
| 4. Data preparation | Tensor creation and sharding | No |
| 5. Execution | Send or receive + compute | **Yes** |
| 6. Cleanup | Barrier + `ttnn.close_device(device)` | No |

Only Phase 5 branches on rank. All other phases run identical code on every process. This symmetry is the hallmark of SPMD design.

### Execution Flow Diagram

```
  Process 0 (rank=0)                     Process 1 (rank=1)
  ========================               ========================
  set_fabric_config(FABRIC_2D)           set_fabric_config(FABRIC_2D)
  open_mesh_device(4x4)                  open_mesh_device(4x4)
    |-- DistributedContext init              |-- DistributedContext init
  build socket_connections               build socket_connections
  build socket_config(rank 0->1)         build socket_config(rank 0->1)
  from_torch(input)                      from_torch(input)
        |                                       |
  rank==0? YES                           rank==0? NO
        |                                       |
  MeshSocket(config)                     MeshSocket(config)
    |-- process_host_ranks()                |-- process_host_ranks()
    |-- endpoint = SENDER                   |-- endpoint = RECEIVER
    |-- connect_with_peer()                 |-- connect_with_peer()
        |                                       |
  relu(input)                            allocate_tensor_on_device(spec)
  send_async(activated, socket)          recv_async(buffer, socket)
        |                                       |
        +------- TT-Fabric Ethernet --------->  |
        |                                  exp(buffer)
        |                                       |
  distributed_context_barrier()  <======>  distributed_context_barrier()
  close_device()                           close_device()
```

---

## 2.3.6 Intra-Process Mode: create_socket_pair

When both the sender mesh and receiver mesh are accessible within the same OS process, the SPMD machinery (ranks, `DistributedContext`) is unnecessary. The `create_socket_pair` static method provides a simpler alternative.

### Signature

```cpp
// Source: mesh_socket.hpp

static std::pair<MeshSocket, MeshSocket> MeshSocket::create_socket_pair(
    const std::shared_ptr<MeshDevice>& sender,
    const std::shared_ptr<MeshDevice>& receiver,
    const SocketConfig& base_config
);
```

### Behavior

1. Takes references to the sender mesh device and receiver mesh device, plus a base `SocketConfig`.
2. Returns a pair: the first element is the sender-side `MeshSocket`, the second is the receiver-side `MeshSocket`.
3. Both sockets are fully connected -- no additional `connect_with_peer()` step is needed.
4. The `SocketConfig` does not need `sender_rank`, `receiver_rank`, or `distributed_context` fields. It uses the mesh_id overload.

### Intra-Process Code Pattern

```python
# Both meshes are in the same process
sender_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4))
receiver_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4))

# Build connections and memory config
socket_connections = [...]
socket_mem_config = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096)

# Use mesh_id overload for SocketConfig
base_config = ttnn.SocketConfig(
    socket_connections,
    socket_mem_config,
    sender_mesh_id=sender_mesh.get_mesh_id(),
    receiver_mesh_id=receiver_mesh.get_mesh_id()
)

# Create paired sockets in one call
send_socket, recv_socket = ttnn.MeshSocket.create_socket_pair(
    sender_mesh, receiver_mesh, base_config
)

# Use them directly -- no rank branching needed
ttnn.experimental.send_async(tensor, send_socket)

upstream_buf = ttnn.allocate_tensor_on_device(tensor.spec, receiver_mesh)
ttnn.experimental.recv_async(upstream_buf, recv_socket)
result = ttnn.exp(upstream_buf)
```

### Key Differences from SPMD Mode

| Property | SPMD (rank-based) | Intra-Process (create_socket_pair) |
|----------|--------------------|------------------------------------|
| Number of OS processes | 2+ | 1 |
| SocketConfig overload | rank fields | mesh_id fields |
| Endpoint pairing | Runtime matches ranks across processes | `create_socket_pair` returns both ends |
| DistributedContext required | Yes (auto-initialized) | No |
| Code structure | Single program, branch on rank | Direct access to both socket objects |
| Multi-host capable | Yes | No (single process, single host) |

### Use Cases

| Scenario | Use create_socket_pair? |
|----------|------------------------|
| Unit testing D2D communication | Yes -- simplest setup |
| Single-process multi-mesh pipeline | Yes -- no SPMD overhead |
| Multi-process SPMD deployment | No -- use rank-based SocketConfig |
| Multi-host deployment | No -- different hosts require separate processes |

**Warning:** `create_socket_pair` is for intra-process use only. If you attempt to use it in a multi-process setup (passing a mesh that belongs to another process), the call will fail or produce invalid sockets.

---

## 2.3.7 Internal Connection Mechanics

Understanding what happens inside `MeshSocket` construction clarifies why the two modes exist.

### SPMD Path (process_host_ranks)

```
  MeshSocket(device, config)
       |
       v
  process_host_ranks()
       |-- Read local rank from DistributedContext
       |-- Compare with config.sender_rank and config.receiver_rank
       |-- Determine: am I SENDER or RECEIVER?
       |
       v
  connect_with_peer()
       |-- Exchange fabric routing info with the peer process
       |-- Establish TT-Fabric Ethernet channels
       |-- Allocate FIFO buffers per SocketMemoryConfig
       |
       v
  Socket ready for send_async / recv_async
```

### Intra-Process Path (process_mesh_ids)

```
  create_socket_pair(sender_dev, receiver_dev, config)
       |
       v
  populate_mesh_ids()
       |-- Read sender and receiver mesh devices
       |-- Resolve mesh IDs locally (same process)
       |-- No inter-process communication needed
       |
       v
  Allocate FIFO buffers on both meshes
  Wire sender buffers to receiver buffers
       |
       v
  Return (sender_socket, receiver_socket) pair
```

---

## 2.3.8 Testing Patterns

The test code in `test_multi_mesh.py` demonstrates important testing patterns:

### Pattern 1: End-to-End Validation with Torch Reference

The receiver computes a result on the received data and compares it against a PyTorch reference:

```python
# Source: test_multi_mesh.py

def torch_op_chain(tensor):
    return torch.exp(torch.nn.functional.relu(tensor))

# On receiver:
torch_tensor = ttnn.to_torch(
    ttnn.from_device(ttnn.exp(upstream_input)),
    mesh_composer=ttnn.ConcatMesh2dToTensor(
        device, mesh_shape=mesh_shape, dims=(2, 3)
    ),
)
if torch.allclose(torch_tensor, torch_op_chain(torch_input)):
    print("Test passed: Output tensor matches expected tensor")
else:
    raise ValueError("Test failed: Output tensor does not match expected tensor")
```

This validates the entire pipeline: tensor sharding, socket transfer, and downstream compute.

### Pattern 2: Process Count Assertion

Guard the test entry point with a check on the distributed context size:

```python
# Source: test_multi_mesh.py

if int(ttnn.distributed_context_get_size()) != 2:
    raise ValueError("This test requires 2 processes to run")
```

This fails fast with a clear error instead of cryptic socket errors when the wrong number of processes is launched.

### Pattern 3: Deterministic Seeding

Set a fixed random seed on all processes to ensure reproducible tensor contents:

```python
# Source: test_multi_mesh.py

torch.manual_seed(0)
```

When both sender and receiver generate the same `torch_input` from the same seed, the receiver can compute the expected output locally and validate against the received-and-computed result.

---

## 2.3.9 Advanced Patterns

### Multi-Stage SPMD Pipeline

For a pipeline with more than two stages, each pair of adjacent stages uses a separate socket with appropriate rank assignments:

```python
rank = int(ttnn.distributed_context_get_rank())

# Stage 0 -> Stage 1 socket
config_01 = ttnn.SocketConfig(connections, mem_config, sender_rank=0, receiver_rank=1)

# Stage 1 -> Stage 2 socket
config_12 = ttnn.SocketConfig(connections, mem_config, sender_rank=1, receiver_rank=2)

if rank == 0:
    socket_01 = ttnn.MeshSocket(device, config_01)
    ttnn.experimental.send_async(output, socket_01)
elif rank == 1:
    socket_01 = ttnn.MeshSocket(device, config_01)
    buf = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.experimental.recv_async(buf, socket_01)
    intermediate = ttnn.relu(buf)
    socket_12 = ttnn.MeshSocket(device, config_12)
    ttnn.experimental.send_async(intermediate, socket_12)
elif rank == 2:
    socket_12 = ttnn.MeshSocket(device, config_12)
    buf = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.experimental.recv_async(buf, socket_12)
    result = ttnn.exp(buf)

ttnn.distributed_context_barrier()
ttnn.close_device(device)
```

### Bidirectional Communication

Two sockets with swapped rank assignments enable bidirectional data flow:

```python
config_forward = ttnn.SocketConfig(connections, mem_config, sender_rank=0, receiver_rank=1)
config_backward = ttnn.SocketConfig(connections, mem_config, sender_rank=1, receiver_rank=0)
```

---

## 2.3.10 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgot `set_fabric_config` before device open | Socket creation fails: "fabric not initialized" | Move `set_fabric_config` to the top of the script, before any device open |
| Mismatched ranks in SocketConfig | Both processes become SENDER (or both RECEIVER) | Ensure `sender_rank` and `receiver_rank` are distinct and match your launcher |
| Forgot `distributed_context_barrier` before close | Hang on `close_device`, or crash on peer | Always barrier before close -- no exceptions |
| Using `create_socket_pair` in multi-process code | Invalid socket handle; crash on first send/recv | Use rank-based SocketConfig for multi-process |
| Rank count mismatch | Process opens device but has no peer | Ensure launcher spawns exactly the right number of processes |
| Different fabric configs across processes | Fabric initialization deadlock | All processes must call `set_fabric_config` with the same argument |
| Using `ttnn.close_mesh_device()` instead of `ttnn.close_device(device)` | API error | The correct teardown call is `ttnn.close_device(device)` |
| Mismatched SocketConfig across processes | Endpoint pairing failure | Both processes must construct identical `SocketConfig` values |

---

## 2.3.11 Summary: Choosing Your Mode

```
  Do sender and receiver
  live in different OS processes?
          |
     +----+----+
     |         |
    YES       NO
     |         |
  SPMD Mode   Intra-Process Mode
     |         |
     |    +----+----+
     |    |         |
     |  Need two  Just need
     |  separate  a quick
     |  socket    pair?
     |  objects?    |
     |    |       YES
     |    |         |
     |  Use mesh_id  Use
     |  overload   create_socket_pair
     |
  SocketConfig with ranks
  + DistributedContext
  + FABRIC_2D
  + branch on rank
```

---

## 2.3.12 DistributedContext Lifecycle

The distributed context has a specific lifecycle that must be respected:

```
  Process start
      |
  set_fabric_config()         <-- before device open
      |
  open_mesh_device()          <-- auto-initializes distributed context
      |
  distributed_context_is_initialized() == True
  distributed_context_get_rank() available
  distributed_context_get_size() available
      |
  ... create sockets, transfer data ...
      |
  distributed_context_barrier()   <-- MUST call before close
      |
  close_device(device)            <-- tears down context
      |
  distributed_context_is_initialized() == False
```

The `distributed_context_barrier()` before `close_device` is non-negotiable. When `close_device` executes, it tears down TT-Fabric connections. If process 0 closes while process 1 is still mid-transfer, the fabric writes target an invalid endpoint, resulting in a hang or crash.

---

## Key Takeaways

1. **SPMD is the production deployment model.** Multi-process, rank-based programming with `DistributedContext` is how D2D sockets are used in real multi-host pipelines. The canonical pattern branches on `ttnn.distributed_context_get_rank()`, not on `get_socket_endpoint_type()`.

2. **DistributedContext auto-initializes.** No explicit initialization call is needed -- `ttnn.open_mesh_device()` handles it. Verify with `distributed_context_is_initialized()`.

3. **create_socket_pair is the intra-process shortcut.** It returns a matched sender/receiver pair without SPMD overhead. Use it for single-process pipelines and unit tests.

4. **FABRIC_2D must be set before opening the mesh device.** Call `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)` before `ttnn.open_mesh_device()`. There is no recovery path without closing and reopening.

5. **Always call `ttnn.distributed_context_barrier()` then `ttnn.close_device(device)` at teardown.** This prevents peer crashes from mid-flight fabric disconnection.

---

**Previous:** [`02_send_recv_async_operations.md`](./02_send_recv_async_operations.md)
**Chapter index:** [`index.md`](./index.md)
**Next chapter:** [Chapter 3 -- TT-Fabric and Multi-Mesh Topology](../ch03_tt_fabric/index.md)
