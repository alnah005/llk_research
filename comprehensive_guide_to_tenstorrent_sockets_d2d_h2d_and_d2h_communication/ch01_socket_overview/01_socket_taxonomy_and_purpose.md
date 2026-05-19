# 1.1 -- Socket Taxonomy and Purpose

Tenstorrent's socket system provides three distinct but architecturally related primitives -- H2D (host-to-device), D2D (device-to-device via MeshSocket), and D2H (device-to-host) -- that together form a unified, FIFO-based communication layer for streaming data across PCIe and Ethernet boundaries in multi-chip systems. Each socket type maps to a distinct physical transport, carries its own memory-placement rules, and exposes a uniform asynchronous API, so pipeline authors can compose all three types within a single application without switching programming models. This section defines the taxonomy, compares the three types side-by-side, locates them in the software stack, contrasts them with CCL collectives and direct memory access, presents a decision tree for choosing the right socket, and illustrates the composition pattern with a real-world DeepSeek V3 pipeline example.

---

## 1.1.1 The Problem Sockets Solve

Modern Tenstorrent deployments place multiple mesh devices on a single host or across hosts. Three categories of data movement arise:

1. **Inter-mesh tensor transfer (D2D).** A tensor produced on Mesh A must be consumed by a kernel on Mesh B. The path runs over TT-Fabric ethernet links between the two meshes. Without sockets, this requires direct TT-Fabric ethernet commands and manual buffer management.
2. **Host-to-device streaming (H2D).** The host CPU must inject input data -- tokenized prompts, activations, control signals -- into device L1 memory so that device kernels can consume it. Without sockets, this means manual PCIe TLB manipulation and ring-buffer bookkeeping.
3. **Device-to-host streaming (D2H).** Device cores must emit results -- output tokens, intermediate diagnostics -- back to host memory for post-processing or forwarding. Without sockets, this requires hand-rolled NOC write tracking and ad-hoc synchronization.

Without sockets, each direction requires a different programming model with hand-rolled address management, manual ring-buffer bookkeeping, and ad-hoc synchronization. Sockets replace that with a single conceptual model: a **unidirectional FIFO** between a producer and a consumer, with the runtime managing buffer allocation, flow control counters, and transport-specific details.

---

## 1.1.2 The Three Socket Types

Every data path in a Tenstorrent pipeline falls into one of three categories based on the *source* and *sink* of the transfer:

```
 +-----------+         +-----------+         +-----------+
 |   HOST    | --H2D-->|  DEVICE A | --D2D-->|  DEVICE B |
 |  (CPU /   |         | (Tensix   |         | (Tensix   |
 |   DRAM)   |<--D2H-- |   cores)  |         |   cores)  |
 +-----------+         +-----------+         +-----------+
```

### H2D -- Host-to-Device (H2DSocket)

H2D sockets inject data from host memory into device L1 or DRAM. Two transfer modes are available depending on who initiates the copy:

| Sub-mode | Who initiates the transfer? | Transport |
|----------|-----------------------------|-----------|
| `HOST_PUSH` | Host CPU writes via PCIe TLB (Translation Lookaside Buffer) windows directly into device L1 | PCIe TLB mapped writes |
| `DEVICE_PULL` | Device NOC engine reads from pinned host memory over PCIe | PCIe NOC reads |

Both modes require vIOMMU support, but for different reasons: HOST_PUSH requires vIOMMU for safe PCIe TLB-based writes to device memory, while DEVICE_PULL requires vIOMMU for device-initiated PCIe NOC reads from pinned host memory. The socket exposes a page-oriented FIFO tracked by `bytes_sent` and `bytes_acked` counters.

**Endpoints:** The producer is the host CPU (user-space process). The consumer is one or more Tensix cores on the device.

**API surface:** `set_page_size`, `write`, `barrier`

**Cross-process support:** A socket created in one process can be exported via `export_descriptor()`, which serializes connection state into a flatbuffer stored in `/dev/shm`. A second process calls `connect()` with the descriptor path, obtaining a handle that bypasses `MetalContext` entirely -- it operates on raw PCIe mappings.

**Typical use case:** Streaming tokenized input from a host inference server into device L1 for a prefill kernel.

### D2D -- Device-to-Device (MeshSocket)

```
  Mesh A (sender cores)                  Mesh B (receiver cores)
 +---------------------+               +---------------------+
 | Tensix core 0       |  TT-Fabric    | Tensix core 0       |
 |   L1 send buffer  --+--> ethernet --+-->  L1 recv buffer  |
 | Tensix core 1       |    links      | Tensix core 1       |
 |   L1 send buffer  --+--> ....   --+-->  L1 recv buffer    |
 +---------------------+               +---------------------+
```

MeshSocket transfers tensors between mesh devices over TT-Fabric Ethernet links. Each connection is strictly 1:1 and is described by a `SocketConnection` that names a sender core and a receiver core using `MeshCoreCoord` (which pairs a `MeshCoordinate` for the device within the mesh with a `CoreCoord` for the core within the device).

**Key configuration types:**

- `SocketConnection` -- binds one `sender_core` to one `receiver_core`.
- `SocketMemoryConfig` -- selects `buffer_type` (L1 or DRAM), `fifo_size`, and optional `sub_device_id` values for sender and receiver.
- `SocketConfig` -- aggregates a vector of `SocketConnection`s, a `SocketMemoryConfig`, sender/receiver identification (via `sender_rank` / `receiver_rank`), and a `DistributedContext`.

**API surface (Python TTNN):**

```python
ttnn.experimental.send_async(tensor, socket)
ttnn.experimental.recv_async(tensor, socket)
ttnn.experimental.barrier(socket)
```

**Cross-process support:** D2D sockets coordinate across processes via `DistributedContext` and rank-based identification using SPMD (Single Program, Multiple Data) execution. Within a single process, `create_socket_pair` wires up both ends without descriptor exchange.

**Typical use case:** Pipelining stages of a large model across two 4x4 meshes, where each stage's output tensor is the next stage's input.

### D2H -- Device-to-Host (D2HSocket)

```
  Device Mesh                           Host CPU
 +-------------------+                 +-------------------+
 | Tensix core       |   PCIe NOC      |                   |
 |   L1 send buffer --+--------------->| pinned host mem   |
 |                   |  device writes   |   (ring buffer)   |
 +-------------------+                 +-------------------+
```

D2H sockets let a device core push results into pinned host memory via PCIe NOC writes. The device core updates `bytes_sent`; the host polls `bytes_sent` and manages its local read pointer.

**Endpoints:** The producer is a Tensix core on the device. The consumer is the host CPU.

**Memory note:** An `ExternalConfigBuffer` can be used for off-allocator L1, allowing the socket's device-side buffer to reside in a region of L1 not managed by TT-Metal's standard allocator.

**API surface (host side):** `set_page_size`, `has_data`, `read`, `barrier`, `pages_available`, `discard_pending_pages`

**Cross-process support:** Same `export_descriptor()` / `connect()` pattern as H2DSocket.

**Typical use case:** Streaming decoded output tokens from device back to the host for delivery to a client.

---

## 1.1.3 Comparison Table

| Property | H2D (H2DSocket) | D2D (MeshSocket) | D2H (D2HSocket) |
|----------|------------------|-------------------|------------------|
| **Direction** | Host --> Device | Device --> Device | Device --> Host |
| **Physical transport** | PCIe (TLB or NOC) | TT-Fabric Ethernet | PCIe NOC writes |
| **Initiator (data mover)** | Host CPU or Device NOC | Device NOC | Device NOC |
| **Sub-modes** | HOST_PUSH, DEVICE_PULL | None | None |
| **Device-side memory** | L1 (HOST_PUSH) or pinned host mem (DEVICE_PULL) | L1 or DRAM (`SocketMemoryConfig`) | L1 (`ExternalConfigBuffer` supported) |
| **Connection topology** | 1:1 | 1:1 (`SocketConnection`) | 1:1 |
| **Send API** | `write` | `send_async` | Device kernel writes via NOC |
| **Receive API** | Device kernel reads from L1 | `recv_async` | `read`, `has_data`, `pages_available` |
| **Synchronization** | `barrier` | `barrier` | `barrier` |
| **Cross-process** | `export_descriptor` / `connect` via `/dev/shm` flatbuffer | `DistributedContext` + SPMD ranks | `export_descriptor` / `connect` via `/dev/shm` flatbuffer |
| **vIOMMU required** | Yes | No | Yes |
| **Typical use case** | Token injection, weight streaming | Inter-stage pipeline, SPMD scatter/gather | Result extraction, logit readback |

---

## 1.1.4 Position in the Software Stack

```
 HOST PROCESS(ES)
 +-------------------------------------------------------------+
 |  User Application / Pipeline (e.g., DeepSeek V3)            |
 |  +-------+    +-----------+    +-------+                    |
 |  | H2D   |    | D2D       |    | D2H   |   <-- Socket API   |
 |  | write  |    | send_async|    | read  |                    |
 |  +---+---+    +-----+-----+    +---+---+                    |
 |      |              |              |                         |
 |  +---v--------------v--------------v---+                    |
 |  |        TT-Metalium Runtime          |                    |
 |  |  Buffer allocation, command queue,  |                    |
 |  |  kernel dispatch                    |                    |
 |  +---+--------------+--------------+---+                    |
 +------|--------------|--------------|-----------+            |
        |              |              |                         |
 -------+--------------+--------------+-------------------------+
        |              |              |       PHYSICAL LAYER
   +----v----+    +----v----+    +----v----+
   |  PCIe   |    |TT-Fabric|    |  PCIe   |
   |  TLB /  |    |Ethernet |    |  NOC    |
   |  NOC    |    | Links   |    | Writes  |
   +---------+    +---------+    +---------+
        |              |              |
   +----v----+    +----v----+    +----v----+
   | Device  |    | Device  |    | Pinned  |
   |   L1    |    | L1/DRAM |    | Host    |
   |  Buffer |    | Buffer  |    | Memory  |
   +---------+    +---------+    +---------+
```

**Key observations:**

- Sockets sit *below* the TTNN op interface but *above* raw fabric and PCIe primitives. They are a runtime abstraction that imposes structure (FIFO ordering, flow control, page-oriented access) on top of the transport.
- A single `SocketConfig` object can bundle multiple `SocketConnection` entries together with a `SocketMemoryConfig` and rank identifiers, making it the canonical unit of configuration for any socket type.
- The application layer (e.g., DeepSeek V3's `PipelineBlock`) composes all three socket types: H2D for token injection, D2D `SocketInterface` for inter-stage transfers, and D2H for result extraction.

---

## 1.1.5 Sockets vs. CCL vs. Direct Memory Access

Sockets, CCL (Collective Communication Library), and direct memory access each serve different communication patterns:

| Criterion | Sockets | CCL Collectives | Direct Memory Access |
|-----------|---------|-----------------|---------------------|
| **Communication pattern** | Point-to-point, 1:1 streaming | Collective N:N (all-reduce, all-gather, etc.) | Single transfer, explicit address |
| **Topology** | Explicit endpoint addressing | Library manages topology | Manual coordination |
| **Lifetime** | Persistent across many transfers | Per-invocation | Per-invocation |
| **Data semantics** | Raw byte/page stream (or tensors) | Typed tensor with reduction semantics | Raw address + size |
| **Flow control** | Built-in (FIFO, bytes_sent/acked) | Managed by collective algorithm | Manual |
| **Multi-process** | Native SPMD support | Multi-device, single-process typical | Manual coordination |
| **Host involvement** | H2D/D2H require host-side API | Purely device-side | Host-initiated |
| **Scheduling** | Caller-driven (send/recv/write/read) | Runtime-scheduled | Synchronous, one-shot |
| **Use when...** | Directed pipeline flows, streaming I/O | Symmetric collectives (gradient sync, tensor-parallel all-gather) | One-shot DMA, low-level kernel setup |

**Rule of thumb:**
- Use **sockets** when you need a persistent, low-overhead, flow-controlled channel between two specific endpoints -- pipeline stages, host I/O streaming, or cross-mesh activation forwarding.
- Use **CCL** when the communication pattern is a well-known collective (all-reduce, all-gather, reduce-scatter) across a uniform group of devices.
- Use **direct memory access** when you need a single, one-shot transfer at a known address with no flow control overhead (e.g., writing weights, reading a final result).

---

## 1.1.6 Socket Selection Decision Tree

```
                        What are the endpoints?
                       /          |             \
                  Host-->Dev   Dev-->Dev      Dev-->Host
                     |            |               |
                    H2D          D2D             D2H
                     |            |               |
             Latency critical?  Same process?   Need low
              /        \        /      \       latency?
            Yes        No     Yes      No       /    \
             |          |      |       |      Yes    No
         HOST_PUSH  DEVICE_   Use     Use      |      |
         (TLB       PULL    create_  Socket   Poll   barrier
          writes,   (NOC    socket_  Config   with    after
          host      reads,  pair()   with     read()  batch
          drives)   device  (intra-  SPMD
                    drives, process) ranks +
                    needs            Distributed
                    vIOMMU)          Context
                                    (inter-
                                     process)
```

**Reading the tree:**

1. Start by identifying which two endpoint *types* are involved (host or device).
2. The endpoint pair directly selects the socket type (H2D, D2D, or D2H).
3. Within each type, secondary questions guide sub-mode and API choices:
   - **H2D:** HOST_PUSH for simpler host-driven transfers; DEVICE_PULL for overlap with device compute (requires vIOMMU).
   - **D2D:** `create_socket_pair` for intra-process setups; `SocketConfig` with SPMD ranks + `DistributedContext` for inter-process.
   - **D2H:** Polling with `has_data()`/`read()` for low-latency consumption; `barrier()` for batch-style processing.

### Quick Reference

- **Injecting tokens or weights from host** --> H2D (HOST_PUSH for simplicity, DEVICE_PULL for overlap with compute)
- **Moving activations between pipeline stages on different chips** --> D2D MeshSocket
- **Extracting logits or results back to host** --> D2H
- **Broadcasting the same tensor to all chips** --> CCL AllGather (not sockets)
- **Summing gradients across all chips** --> CCL AllReduce (not sockets)
- **One-shot weight upload or final result readback** --> Direct memory access (not sockets)

---

## 1.1.7 Real-World Composition: DeepSeek V3 Pipeline

The DeepSeek V3 `PipelineBlock` demonstrates all three socket types working together in a multi-stage inference pipeline:

```
  Host                Chip 0          Chip 1       ...   Chip N           Host
  +----+   H2D     +--------+  D2D  +--------+        +--------+  D2H   +----+
  |    |---------->| Stage 0 |------>| Stage 1 |-...-->| Stage N |------>|    |
  |Host|  token    |        |tensor |        |        |        | result |Host|
  |    |  inject   +--------+  xfer +--------+        +--------+ extract|    |
  +----+                                                                 +----+
```

Each `PipelineBlock` stage config specifies:

- **Input socket type** (H2D for the first stage, D2D for subsequent stages).
- **Output socket type** (D2D for intermediate stages, D2H for the final stage).
- **`SocketInterface`** objects that wrap `SocketConfig` with directional semantics.

This composition pattern -- H2D at ingress, a chain of D2D links, D2H at egress -- is the canonical architecture for streaming inference pipelines on Tenstorrent hardware. All three socket types share the same FIFO-based programming model, so the pipeline author composes them without switching paradigms.

---

## Key Takeaways

1. **Three types, one system.** H2D, D2D, and D2H sockets form a unified communication layer; the type is determined entirely by the source/sink endpoint pair. All connections are 1:1; multi-party patterns are built by composing multiple 1:1 links.
2. **Sockets are point-to-point; CCL is collective.** Use sockets for directed pipeline flows and host I/O streaming. Use CCL for symmetric reductions and gathers. Use direct memory access for one-shot transfers.
3. **H2D offers two transfer modes.** HOST_PUSH (host-driven, simpler) and DEVICE_PULL (device-driven, overlaps with compute, requires vIOMMU).
4. **Cross-process mechanisms differ by type.** H2D and D2H use `export_descriptor`/`connect` via `/dev/shm` flatbuffers. D2D uses SPMD ranks + `DistributedContext` -- it does *not* use descriptor exchange.
5. **Composition is the design pattern.** Production models like DeepSeek V3 compose all three socket types in a single pipeline, using H2D at ingress, D2D between stages, and D2H at egress.

---

**Next:** [`02_common_abstractions_preview.md`](./02_common_abstractions_preview.md)
