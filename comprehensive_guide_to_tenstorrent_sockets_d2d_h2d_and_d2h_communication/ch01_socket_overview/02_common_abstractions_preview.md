# 1.2 -- Common Abstractions Preview

Although H2D, D2D, and D2H sockets differ in transport and direction, they share a set of fundamental abstractions: a page-oriented FIFO data model, credit-based flow control via `bytes_sent`/`bytes_acked` counters, non-blocking (async) operation semantics, SPMD rank-aware configuration for multi-process execution, a consistent five-phase connection lifecycle, and a cross-process sharing mechanism that splits cleanly between descriptor exchange (H2D/D2H) and SPMD ranks (D2D). This section previews each abstraction at a conceptual level; subsequent chapters cover the type-specific configuration and implementation details.

---

## 1.2.1 The FIFO Data Model

Every socket type models its channel as a **bounded, page-oriented FIFO** (circular buffer). The sender writes fixed-size pages into the FIFO; the receiver reads them out in order.

```
                   fifo_size (bytes)
        +------------------------------------+
        |                                    |
  write ---> [ page | page | page | ... ] ---> read
  ptr         ^^^^^^^^^^^^^^^^^^^^^^^^^^^       ptr
              pages in flight (unacked)
        +------------------------------------+
```

**Key FIFO properties:**

- **Page-granular.** All reads and writes operate on whole pages. Page size is fixed after configuration. Partial page writes are not supported.
- **Bounded depth.** The FIFO has a finite capacity. The number of pages that can be in flight equals the buffer size divided by the page size. The sender blocks or returns a back-pressure signal when the FIFO is full.
- **Ordered delivery.** Pages arrive at the consumer in the same order they were produced. No reordering occurs within a single socket.
- **Wrap-around.** When either pointer reaches the end of the buffer, it wraps to zero. The buffer is circular; no data is copied during wrap.

### Page Size and Memory Placement Per Socket Type

| Socket Type | How page size is set | FIFO location | Configured via |
|-------------|---------------------|---------------|----------------|
| H2D (HOST_PUSH) | `set_page_size(size)` | Device L1 | TLB window size |
| H2D (DEVICE_PULL) | `set_page_size(size)` | Pinned host memory | Host buffer allocation |
| D2D | Derived from tensor layout | Device L1 or DRAM | `SocketMemoryConfig(memory_type, fifo_size)` |
| D2H | `set_page_size(size)` | Pinned host memory | D2H constructor `fifo_size` parameter (controls pinned host memory FIFO allocation); `ExternalConfigBuffer` optionally controls the device-side config buffer address in L1 |

The FIFO model gives every socket type the same fundamental property: **the sender can write ahead of the receiver, up to the FIFO depth, without blocking.**

---

## 1.2.2 Flow Control Preview

Flow control prevents the sender from overwriting pages the receiver has not yet consumed. All three socket types use a **counter-based credit scheme**, though the counters live in different places depending on the transport:

```
  Sender side                     Receiver side
  +-----------+                   +-----------+
  |bytes_sent |----(data+meta)--->|bytes_sent |  (receiver's copy,
  |           |                   |           |   updated by sender
  |           |<--(ack/credit)----|bytes_acked|   or transport)
  +-----------+                   +-----------+

  Invariant:  bytes_sent - bytes_acked  <=  fifo_capacity
  Available credits = fifo_size - (bytes_sent - bytes_acked)
```

### How It Works

1. The sender increments `bytes_sent` by the page size each time it pushes a page into the FIFO.
2. The receiver increments `bytes_acked` after it has consumed a page and the FIFO slot is free.
3. The sender computes available credits as `fifo_size - (bytes_sent - bytes_acked)`. If credits are zero, the sender must wait.
4. Because both counters only increase monotonically, the scheme is safe under concurrent access without locks.

### Counter Locations Per Socket Type

| Socket type | `bytes_sent` location | `bytes_acked` mechanism |
|-------------|----------------------|------------------------|
| H2D | Host-side counter | Device writes ack after consuming page |
| D2D | Sender device L1 / DRAM | Receiver sends credit via TT-Fabric Ethernet |
| D2H | Device-side counter | Host advances read pointer after `read()` |

### Back-Pressure Behavior

**Back-pressure** occurs when `bytes_sent - bytes_acked == fifo_capacity`. At that point:

- **H2D:** The host `write` call stalls until the device acknowledges consumption.
- **D2D:** The `send_async` op respects credits internally; the program can continue issuing other ops.
- **D2H:** The device stalls its NOC write until the host calls `read()` to free space.

---

## 1.2.3 Non-Blocking (Async) Semantics

All socket operations are designed to be **non-blocking at the API level**. The call returns immediately, enqueueing work to be performed by the runtime or hardware. A separate `barrier()` call blocks until all previously enqueued operations have completed.

| Operation | Socket Type | Behavior |
|-----------|-------------|----------|
| `write` | H2D | Queues page for transfer; returns immediately |
| `send_async` | D2D | Initiates Fabric send; returns immediately |
| `recv_async` | D2D | Initiates Fabric receive; returns immediately |
| `has_data` | D2H | Returns `true`/`false` without blocking |
| `read` | D2H | Copies available pages; does not wait for more |
| `pages_available` | D2H | Returns count of ready pages without blocking |
| `discard_pending_pages` | D2H | Drops unconsumed pages without reading them |
| `barrier` | H2D, D2D, D2H | **Blocking exception** -- waits for all in-flight ops |

### The Barrier Contract

`barrier()` is the explicit synchronization point. It guarantees that all data written before the barrier has been acknowledged by the receiver. Pipeline authors place barriers at stage boundaries -- for example, after writing all tokens via H2D and before launching the first compute kernel that depends on those tokens.

```
  Time --->

  Host:    write  write  write  barrier ........... (blocked)
                                   |
  Device:  ....   consume consume  consume  ack --> barrier returns
```

Between `write` calls and the `barrier`, the host is free to perform other work (prepare the next batch, poll D2H sockets, etc.).

### D2D Pipeline Overlap

The non-blocking model enables overlapping computation with communication:

```python
# D2D pipeline overlap (pseudocode)
ttnn.experimental.send_async(stage_1_output, socket)   # enqueue send -- returns immediately
result = ttnn.matmul(local_data, weights)               # compute while send is in flight
ttnn.experimental.recv_async(stage_2_input, socket)     # enqueue receive
# ... stage_2_input is valid after the runtime resolves the dependency
```

### D2H Code Example

```cpp
// Non-blocking query: does the FIFO contain at least one page?
bool ready = d2h_socket.has_data();

// Non-blocking read: copies available pages out of the FIFO
d2h_socket.read(dest_ptr, num_pages);

// Blocks until all in-flight data has arrived
d2h_socket.barrier();

// Additional non-blocking queries
size_t n = d2h_socket.pages_available();
d2h_socket.discard_pending_pages();
```

---

## 1.2.4 SPMD and Rank-Aware Configuration Preview

Multi-process SPMD (Single Program, Multiple Data) execution is the standard deployment model for multi-chip Tenstorrent systems. All processes run the same program binary but branch on their **rank** to determine their role. Sockets integrate with SPMD through **rank-aware configuration**.

```
  Process 0 (rank=0)              Process 1 (rank=1)
  +------------------+            +------------------+
  | MeshDevice(4x4)  |            | MeshDevice(4x4)  |
  |                   |            |                   |
  |  SocketConfig:    |            |  SocketConfig:    |
  |   sender_rank=0   |   D2D     |   sender_rank=0   |
  |   receiver_rank=1 |---------->|   receiver_rank=1  |
  |                   |  Ethernet |                   |
  +------------------+            +------------------+
```

The `SocketConfig` object carries:

- `connections` -- list of `SocketConnection` (sender/receiver core pairs).
- `socket_memory_config` -- `SocketMemoryConfig(memory_type, fifo_size)` where `memory_type` is `L1` or `DRAM`.
- `sender_rank` / `receiver_rank` -- integer rank IDs that map to SPMD processes.
- `DistributedContext` -- provides rank assignment, mesh topology (e.g., `MeshShape(4, 4)` for a 4x4 mesh), and the fabric type (e.g., `FABRIC_2D`). It is the coordination backbone for SPMD socket programs.

### Inter-Process D2D Example (from test_multi_mesh.py)

```python
# Both processes execute this same program
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(4, 4),
    mesh_type=ttnn.MeshType.FABRIC_2D
)

config = ttnn.SocketConfig(
    connections=connections,
    memory_config=ttnn.SocketMemoryConfig(
        memory_type=ttnn.BufferType.L1,
        fifo_size=4096
    ),
    sender_rank=0,
    receiver_rank=1
)

# Branch on rank
if rank == 0:
    # Sender: compute then send
    output = ttnn.relu(input_tensor)
    ttnn.experimental.send_async(output, config)
elif rank == 1:
    # Receiver: receive then compute
    buffer = ttnn.allocate_tensor(shape, layout, mesh)
    ttnn.experimental.recv_async(buffer, config)
    result = ttnn.exp(buffer)
```

**Key properties of the SPMD model:**

- **Symmetric code, asymmetric roles.** The same source file is launched N times. Each instance checks `rank` to decide whether it is a sender or receiver.
- **Rank-based socket matching.** The runtime uses the `sender_rank` / `receiver_rank` fields to pair the two ends of a socket across processes.
- **Intra-process shortcut.** For D2D sockets where both endpoints are in the same process, `create_socket_pair()` returns a matched sender/receiver pair without requiring rank configuration.

---

## 1.2.5 Socket Lifecycle Phases

Every socket, regardless of type, passes through the same five phases:

```
  +-------------------+
  | 1. CREATION       |    Allocate socket object, establish
  |                   |    endpoint identity. For D2D: create_socket_pair
  |                   |    (intra-process) or allocate both sides
  |                   |    independently (inter-process).
  +--------+----------+
           |
           v
  +--------+----------+
  | 2. CONFIGURATION  |    Set page_size, memory config (L1/DRAM,
  |                   |    fifo_size), rank assignment. For cross-
  |                   |    process H2D/D2H: export_descriptor / connect.
  +--------+----------+
           |
           v
  +--------+-----------+
  | 3. DATA TRANSFER   |    Producer calls write / send_async.
  |                    |    Consumer calls read / recv_async / has_data.
  |                    |    Flow control governs throughput.
  +--------+-----------+
           |
           v
  +--------+-----------+
  | 4. SYNCHRONIZATION |    barrier() ensures all in-flight data
  |                    |    has been delivered and acknowledged.
  |                    |    Phases 3 and 4 may repeat (multiple
  |                    |    batches with barriers between them).
  +--------+-----------+
           |
           v
  +--------+-----------+
  | 5. TEARDOWN        |    Drain remaining data, issue final
  |                    |    barrier, close socket handles, free
  |                    |    buffers. Cross-process descriptors
  |                    |    in /dev/shm are cleaned up.
  +-------------------+
```

### Per-Socket Lifecycle Details

| Phase | H2D | D2D | D2H |
|-------|-----|-----|-----|
| **Create** | Open socket, pin host memory, vIOMMU setup | `create_socket_pair()` or `SocketConfig` with ranks | Open socket, pin host memory, vIOMMU setup |
| **Configure** | `set_page_size(size)` | `SocketMemoryConfig(L1/DRAM, fifo_size)` | `set_page_size(size)`, `ExternalConfigBuffer` |
| **Transfer** | `write()` | `send_async()`, `recv_async()` | `read()`, `has_data()` |
| **Synchronize** | `barrier()` | `barrier()` | `barrier()` |
| **Teardown** | `barrier()` then release | `distributed_context_barrier()` then `close_mesh_device()` | `barrier()` then release |

**Important warnings:**

- **Skipping CONFIGURATION** (e.g., forgetting `set_page_size`) results in undefined behavior.
- **Skipping TEARDOWN** leaks pinned host memory or device L1/DRAM allocations.
- **For D2D sockets in multi-process SPMD**, a `distributed_context_barrier()` must be called before closing the mesh device. This ensures both processes have finished all socket operations before resources are released.

---

## 1.2.6 Cross-Process Sharing Preview

H2D and D2H sockets support **cross-process sharing** through a descriptor-based mechanism that uses `/dev/shm` (POSIX shared memory) as the transport. D2D sockets use a fundamentally different mechanism -- SPMD ranks -- because both endpoints are on-device.

### H2D / D2H: Descriptor Exchange

```
  Process A (owner)                  Process B (connector)
  +----------------+                 +----------------+
  | Create socket  |                 |                |
  | export_        |                 |                |
  |   descriptor() |--flatbuffer---->| connect(desc)  |
  |                |  via /dev/shm   |                |
  +----------------+                 +----------------+
         |                                  |
         v                                  v
     Owns the                  Shares the same
     underlying                underlying socket
     socket                    (reads/writes to
                               same FIFO)
```

The flow is:

1. **Process A** creates the socket and calls `export_descriptor()`, which serializes the socket's connection metadata (buffer addresses, page size, flow control counter locations, PCIe BAR mappings) into a flatbuffer and writes it to a file in `/dev/shm`.
2. **Process B** calls `connect(descriptor_path)` with the path to the shared-memory file, reconstructing a handle to the same underlying socket. No `MetalContext` is needed -- it operates on raw PCIe mappings.
3. Both processes can now operate on the socket. The FIFO flow-control counters are in shared memory, so mutual exclusion is handled by atomic counter updates.

This mechanism enables architectures where one process manages the device while another supplies or consumes data -- for example, a frontend process handles token encoding (H2D) while a backend process runs the model.

### D2D: SPMD Rank-Based Coordination

D2D sockets do **not** use `export_descriptor` / `connect`. Cross-process D2D is handled by the SPMD rank system:

- Both processes create sockets with matching `SocketConfig` (same `sender_rank`, `receiver_rank`, and connection list).
- `DistributedContext` manages rank assignment and socket pairing through its own inter-process communication mechanisms.
- The runtime matches the two ends based on the rank fields.

This clean split reflects the physical reality: host-side sockets (H2D/D2H) share CPU-addressable memory via PCIe, while device-side sockets (D2D) share the Ethernet fabric.

---

## Key Takeaways

1. **Page-oriented FIFO is the universal data model.** Every socket type moves fixed-size pages through a bounded circular buffer, giving uniform semantics regardless of transport. The FIFO location varies (device L1, DRAM, or pinned host memory), but the abstraction is the same.
2. **Credit-based flow control prevents data loss.** The `bytes_sent` / `bytes_acked` invariant ensures the sender never overwrites unconsumed data, with back-pressure propagating naturally through the counter gap. No locks are needed.
3. **All APIs are non-blocking by default.** `send_async`, `recv_async`, `write`, and `read` return immediately; explicit `barrier()` calls provide synchronization when needed. This enables overlapping computation with communication.
4. **SPMD rank fields in `SocketConfig` unify multi-process coordination.** The same program runs on every process; the `sender_rank` / `receiver_rank` fields determine each process's role. `DistributedContext` provides the coordination backbone.
5. **Cross-process sharing uses `/dev/shm` flatbuffer descriptors for H2D/D2H, and SPMD ranks + `DistributedContext` for D2D.** H2D/D2H descriptor exchange bypasses `MetalContext`; D2D rank matching uses the fabric's own coordination. Do not confuse the two mechanisms.

---

**Next:** [Chapter 2 -- D2D MeshSocket Configuration](../ch02_d2d_mesh_socket/index.md)
