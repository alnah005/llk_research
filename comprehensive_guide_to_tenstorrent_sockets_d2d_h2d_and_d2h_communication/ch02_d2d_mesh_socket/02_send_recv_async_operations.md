# 2.2 -- send_async and recv_async Operations

With a `SocketConfig` assembled and a `MeshSocket` constructed, the next step is moving data. D2D MeshSocket provides two primary data-transfer operations -- `ttnn.experimental.send_async` and `ttnn.experimental.recv_async` -- plus synchronization via `ttnn.distributed_context_barrier()`. This section covers the semantics of each operation, the critical pre-allocation requirement for receiving, endpoint type queries, ordering guarantees, and the synchronization model.

---

## 2.2.1 Operation Overview

D2D data transfer uses two asymmetric operations. The sender pushes a tensor; the receiver must have a buffer ready before the push arrives. Both operations live in the `ttnn.experimental` namespace.

```
  Sender (rank 0)                          Receiver (rank 1)
  +-----------------+                      +-----------------+
  |                 |                      |                 |
  | tensor = relu(x)|                      | buf = allocate  |
  |                 |                      |   _tensor_on_   |
  | send_async(     |   TT-Fabric         |   device(spec)  |
  |   tensor,       |---ethernet--------->|                 |
  |   socket)       |                      | recv_async(     |
  |                 |                      |   buf, socket)  |
  |                 |                      |                 |
  | (returns        |                      | (returns        |
  |  immediately)   |                      |  immediately)   |
  +-----------------+                      +-----------------+
```

```python
ttnn.experimental.send_async(tensor, socket)
ttnn.experimental.recv_async(tensor, socket)
```

---

## 2.2.2 send_async: Sender Semantics

```python
ttnn.experimental.send_async(tensor: ttnn.Tensor, socket: ttnn.MeshSocket) -> None
```

### Preconditions

| # | Precondition | Rationale |
|---|-------------|-----------|
| 1 | `socket.get_socket_endpoint_type() == SocketEndpoint.SENDER` | Only sender sockets can transmit |
| 2 | `tensor` is allocated on the same mesh device that the socket is bound to | Data must be local to the sending device |
| 3 | The tensor's data layout is compatible with the receiver's pre-allocated buffer | Shape and dtype must match |

### Postconditions

- The transfer is enqueued on the device. `send_async` returns immediately.
- The data will be transmitted through the TT-Fabric transport layer (see [Chapter 3](../ch03_tt_fabric/index.md)) to the receiver's FIFO buffer.
- The tensor can be the result of a prior compute operation (e.g., `ttnn.relu(input)`) -- the runtime respects on-device data dependencies and will not read the tensor until the compute completes.

### Behavior Details

| Property | Value |
|----------|-------|
| **Blocking?** | No. Returns immediately after enqueuing the transfer command. |
| **Ownership** | The runtime takes a reference to the tensor's device buffer. The caller must not free or overwrite the tensor until the transfer completes. |
| **Ordering** | FIFO within a single socket. If you call `send_async` twice on the same socket, the first tensor is guaranteed to arrive before the second. |
| **Endpoint requirement** | The socket must have `SocketEndpoint.SENDER` type. Calling `send_async` on a receiver socket is an error. |

### Non-Blocking Overlap

`send_async` enqueues the transfer and returns. The caller can immediately issue additional compute operations:

```python
ttnn.experimental.send_async(stage_output, socket)  # enqueue send
local_result = ttnn.matmul(other_data, weights)      # compute overlaps with send
```

The send and the matmul can execute concurrently because `send_async` does not wait for acknowledgment.

### Typical Pattern

```python
# Source: test_multi_mesh.py

# Compute on sender, then send
ttnn.experimental.send_async(ttnn.relu(ttnn_input), send_socket)
```

---

## 2.2.3 recv_async: Receiver Semantics

```python
ttnn.experimental.recv_async(tensor: ttnn.Tensor, socket: ttnn.MeshSocket) -> None
```

### Preconditions

| # | Precondition | Rationale |
|---|-------------|-----------|
| 1 | `socket.get_socket_endpoint_type() == SocketEndpoint.RECEIVER` | Only receiver sockets can accept data |
| 2 | `tensor` is already allocated on the receiver's mesh device | The runtime writes into this buffer; it does not allocate |
| 3 | `tensor` has sufficient size and matching spec to hold the incoming data | A mismatch produces silent data corruption |

### Postconditions

- The receive is enqueued on the device. `recv_async` returns immediately.
- When the corresponding `send_async` data arrives, it is written directly into `tensor`.
- Subsequent TTNN operations that consume the tensor will be scheduled after the receive completes, thanks to the runtime's dependency tracking.

### The Pre-Allocation Requirement

This is the most critical constraint for correct D2D receiver code. The receiver **must** allocate the destination tensor before calling `recv_async`. The runtime does not allocate memory on behalf of the receiver.

```python
# Source: test_multi_mesh.py

# CORRECT: pre-allocate, then receive
upstream_input = ttnn.allocate_tensor_on_device(ttnn_input.spec, device)
ttnn.experimental.recv_async(upstream_input, recv_socket)

# WRONG: no pre-allocation -- recv_async has nowhere to write
# ttnn.experimental.recv_async(???, recv_socket)  # what buffer?
```

Note that `ttnn_input.spec` is a property access, not a method call -- there are no parentheses.

### Why Pre-Allocation Is Mandatory

| Reason | Explanation |
|--------|-------------|
| **Zero-copy receive** | The TT-Fabric DMA writes directly into the destination L1 address. There is no intermediate staging area. Pre-allocation ensures the buffer exists at a known address before data arrives. |
| **Deterministic memory layout** | Compute kernels that consume the received data can be compiled against the known buffer address. The receive buffer address is fixed at allocation time, not at transfer time. |
| **No runtime allocation during transfer** | Memory allocation is a host-side operation that could introduce latency and non-determinism. By requiring pre-allocation, the receive path is purely a hardware data-movement operation. |

### Obtaining the Correct Spec

The receiver must allocate a tensor with the same spec (shape, layout, data type, memory config) as the tensor the sender will transmit:

```python
# Derive from an existing tensor's spec (property, not method call)
upstream_input = ttnn.allocate_tensor_on_device(reference_tensor.spec, device)
```

**Warning:** If the spec of the pre-allocated buffer does not match the incoming tensor (different shape, dtype, or layout), the behavior is undefined. The runtime does not perform a cross-process spec check. You will get corrupted data with no error message.

---

## 2.2.4 Endpoint Type Queries

Every constructed `MeshSocket` is exactly one of `SocketEndpoint.SENDER` or `SocketEndpoint.RECEIVER`, determined at construction time.

### get_socket_endpoint_type

```python
endpoint = socket.get_socket_endpoint_type()
# Returns: SocketEndpoint.SENDER or SocketEndpoint.RECEIVER
```

```cpp
// C++ equivalent (mesh_socket.hpp)
SocketEndpoint get_socket_endpoint_type() const;
// Returns: SocketEndpoint::SENDER or SocketEndpoint::RECEIVER
```

The endpoint type is determined based on the `SocketConfig` fields:

| Config Form | Current Context | Endpoint Type |
|-------------|-----------------|---------------|
| Rank-based | Current rank == `sender_rank` | `SENDER` |
| Rank-based | Current rank == `receiver_rank` | `RECEIVER` |
| Mesh-ID-based | Current mesh == `sender_mesh_id` | `SENDER` |
| Mesh-ID-based | Current mesh == `receiver_mesh_id` | `RECEIVER` |

This method is useful for writing generic code that operates on a socket without knowing its role in advance:

```python
if socket.get_socket_endpoint_type() == ttnn.SocketEndpoint.SENDER:
    ttnn.experimental.send_async(tensor, socket)
else:
    ttnn.experimental.recv_async(tensor, socket)
```

**Warning:** Calling `send_async` on a `RECEIVER` socket or `recv_async` on a `SENDER` socket is undefined behavior. Always check the endpoint type or structure your code so that senders and receivers are handled on separate code paths.

---

## 2.2.5 Buffer Access

The `MeshSocket` exposes two buffers for advanced use cases (diagnostics, custom kernels, memory inspection).

### get_data_buffer

```python
data_buf = socket.get_data_buffer()  # returns MeshBuffer
```

Returns the `MeshBuffer` that holds the FIFO data. This is the buffer where `send_async` writes source data and `recv_async` receives destination data. The buffer's type (L1 or DRAM) and size match the `SocketMemoryConfig`.

**Warning:** `get_data_buffer()` can only be queried on receiver sockets. This restriction comes directly from the source code: "Access the data-buffer associated with the socket on the receiver mesh. Can only be queried for receiver sockets."

### get_config_buffer

```python
config_buf = socket.get_config_buffer()  # returns MeshBuffer
```

Returns the `MeshBuffer` that holds the socket's internal control state -- flow control counters, connection metadata, and fabric routing information. This buffer is managed entirely by the runtime; modifying it directly leads to undefined behavior.

### When to Use Buffer Access

| Use Case | Buffer | Notes |
|----------|--------|-------|
| Inspect FIFO occupancy for debugging | `get_config_buffer()` | Read-only; parse counter layout per runtime version |
| Pass buffer address to a custom kernel | `get_data_buffer()` | Receiver-only; kernel must respect FIFO protocol |
| Memory accounting | Both | Query `size()` to track L1/DRAM consumption |

For standard send/receive workflows, you do not need to access these buffers directly. The `send_async` and `recv_async` operations handle all buffer interactions.

---

## 2.2.6 Synchronization

Because both `send_async` and `recv_async` are non-blocking, the programmer must synchronize explicitly when data visibility is required.

### Distributed Context Barrier

The primary synchronization mechanism for multi-process sockets is the distributed context barrier:

```python
# Source: test_multi_mesh.py

ttnn.distributed_context_barrier()
```

This call blocks until all processes in the distributed context have reached the barrier. After the barrier returns:

- All `send_async` calls issued before the barrier have completed transmission.
- All `recv_async` calls issued before the barrier have received their data.
- Pre-allocated tensors on receiver ranks contain valid data.

### Implicit Device-Level Dependency Tracking

In many cases, **explicit barriers are unnecessary** because the TTNN runtime tracks data dependencies:

```python
# Source: test_multi_mesh.py

ttnn.experimental.recv_async(upstream_input, recv_socket)
result = ttnn.exp(upstream_input)  # runtime ensures recv completes before exp starts
```

The runtime knows that `ttnn.exp` reads `upstream_input`, which is the target of a pending `recv_async`. It inserts the necessary synchronization internally. The test code in `test_multi_mesh.py` relies on exactly this pattern: `ttnn.exp(upstream_input)` is called without an explicit barrier, and the runtime ensures correctness.

### When Explicit Synchronization Is Needed

| Scenario | Method | Why |
|----------|--------|-----|
| Before teardown | `ttnn.distributed_context_barrier()` | Ensure all processes finished before closing devices |
| Reading tensor from host | `ttnn.distributed_context_barrier()` | Host reads bypass the device dependency tracker |
| Reusing send buffer | Wait for transfer completion | Overwriting the tensor before send completes corrupts the transfer |
| Intra-process, single device | `device.synchronize()` | Blocks until all ops on that device (including transfers) complete |

**Warning:** Using `device.synchronize()` in a multi-process setting does not guarantee that the remote sender has completed its transfer. Always use `ttnn.distributed_context_barrier()` when the receiver needs data from a sender in a different process.

---

## 2.2.7 Ordering Guarantees

### Within a Single Socket: FIFO

Data sent through a single socket arrives in order. If the sender calls:

```python
ttnn.experimental.send_async(tensor_A, socket)
ttnn.experimental.send_async(tensor_B, socket)
```

The receiver will observe `tensor_A`'s data before `tensor_B`'s data. This follows from the FIFO model -- pages are delivered in the order they were enqueued.

```
  Connection (core 0,0 -> core 0,0):

    send A  -->  send B  -->  send C
                                        recv A  -->  recv B  -->  recv C
                                        (guaranteed order)
```

### Across Multiple Sockets: No Ordering

There are **no ordering guarantees** between different sockets. If a sender transmits data through socket X and socket Y, the receiver may observe the data from Y before X, even if X was sent first.

```
  Socket 1 (core 0,0 -> core 0,0):  send X  ------------>  recv X
  Socket 2 (core 0,1 -> core 0,1):  send Y  -->  recv Y

  Y may arrive before X even though X was sent first.
  No cross-socket ordering is guaranteed.
```

If cross-socket ordering matters, you must enforce it at the application level using barriers or explicit sequencing.

### Ordering Summary

| Guarantee | Scope | Description |
|-----------|-------|-------------|
| Per-socket FIFO | Single socket | Sends and receives on the same socket pair are matched in order |
| No cross-socket ordering | Multiple sockets | Sends on socket A and socket B have no relative ordering guarantee |
| Barrier is global | All ranks | After barrier, all pre-barrier operations across all sockets are complete |

### Send-Receive Timing Flexibility

In an SPMD program, the sender and receiver execute independently. There is no implicit synchronization between the `send_async` call on rank 0 and the `recv_async` call on rank 1. The FIFO buffers absorb timing differences:

- **recv_async called first:** The receiver's buffer is registered; data arrives when the sender sends.
- **send_async called first:** Data enters the FIFO; when the receiver calls `recv_async`, the data drains immediately.
- **Simultaneous:** The FIFO handles concurrent access.

---

## 2.2.8 Multi-Connection Parallelism

When a `SocketConfig` contains multiple `SocketConnection` entries, `send_async` and `recv_async` automatically distribute the tensor across all connection cores:

```
  SocketConfig with 16 connections (4x4 mesh):
    core (0,0) <-> core (0,0)    \
    core (0,1) <-> core (0,1)     |-- send_async distributes
    ...                            |   tensor shards across
    core (3,3) <-> core (3,3)    /    all 16 channels

  Effective bandwidth = single_link_bw * num_connections
                        (up to fabric saturation)
```

The caller does not need to manually shard the tensor. The runtime handles the decomposition and reassembly.

**Warning:** If the tensor is too small to be meaningfully split across all connections, some cores will have zero-length shards. This may cause the transfer to hang or produce incorrect results.

---

## 2.2.9 Complete Send/Recv Example

The following example shows the full lifecycle from `test_multi_mesh.py`:

```python
# Source: test_multi_mesh.py

# Both processes execute this code; rank determines behavior
if int(ttnn.distributed_context_get_rank()) == 0:
    # Sender: construct socket and dispatch
    send_socket = ttnn.MeshSocket(device, socket_config)
    ttnn.experimental.send_async(ttnn.relu(ttnn_input), send_socket)
else:
    # Receiver: construct socket, pre-allocate, receive, then compute
    recv_socket = ttnn.MeshSocket(device, socket_config)
    upstream_input = ttnn.allocate_tensor_on_device(ttnn_input.spec, device)
    ttnn.experimental.recv_async(upstream_input, recv_socket)

    # Runtime ensures recv completes before exp starts (dependency tracking)
    torch_tensor = ttnn.to_torch(
        ttnn.from_device(ttnn.exp(upstream_input)),
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            device, mesh_shape=mesh_shape, dims=(2, 3)
        ),
    )

# Synchronize all processes before teardown
ttnn.distributed_context_barrier()
ttnn.close_device(device)
```

The key sequence for the receiver is:

1. Construct the `MeshSocket` (allocates internal buffers)
2. Allocate the destination tensor with `allocate_tensor_on_device` using `ttnn_input.spec` (property)
3. Call `recv_async` (returns immediately)
4. Use the tensor in subsequent compute (the runtime ensures data dependencies are respected on-device)
5. Synchronize with the sender via `distributed_context_barrier()`

---

## 2.2.10 Common Mistakes and Diagnostics

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Calling `send_async` on receiver endpoint | Runtime error | Check `get_socket_endpoint_type()` |
| Calling `recv_async` without pre-allocating | Crash or undefined behavior | Always `allocate_tensor_on_device` first |
| Mismatched tensor specs between sender and receiver | Silent data corruption | Log and compare specs on both sides before transfer |
| Using `input_tensor.spec()` with parentheses | Error (spec is a property) | Use `input_tensor.spec` without parentheses |
| Reusing send buffer before transfer completes | Data corruption in flight | Wait for completion before reusing |
| Different send/recv call counts across iterations | Deadlock | Ensure paired send/recv counts match |

### Debugging Checklist

1. Print `socket.get_socket_endpoint_type()` on both processes.
2. Print tensor specs on sender and receiver; confirm they match.
3. Verify `ttnn.distributed_context_barrier()` is called before reading results or closing devices.
4. Check that `get_active_cores()` returns the expected core set.
5. Verify that no core appears in two different socket configs.

---

## Key Takeaways

1. **Both operations are non-blocking.** `ttnn.experimental.send_async` enqueues; `ttnn.experimental.recv_async` registers a target buffer. Neither blocks. Use `ttnn.distributed_context_barrier()` to synchronize across processes.

2. **Pre-allocation is mandatory for recv_async.** You must call `ttnn.allocate_tensor_on_device` with a matching spec before calling `recv_async`. This enables zero-copy DMA directly into the receive buffer.

3. **FIFO ordering is guaranteed per socket; no ordering across sockets.** Multiple sends on the same socket arrive in order. Use barriers to enforce cross-socket ordering when required.

4. **The runtime tracks on-device dependencies.** In most cases you can call compute operations on a received tensor without an explicit barrier -- the runtime ensures the receive completes first. Explicit barriers are needed before host reads or device teardown.

5. **get_data_buffer() is receiver-only.** The data buffer access method can only be called on receiver sockets, per the source code restriction.

---

**Next:** [`03_spmd_and_intra_process.md`](./03_spmd_and_intra_process.md)
