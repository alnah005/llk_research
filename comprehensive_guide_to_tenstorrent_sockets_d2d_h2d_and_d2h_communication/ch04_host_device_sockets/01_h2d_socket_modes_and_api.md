# 4.1 -- H2D Socket Modes and API

An H2D socket streams data from a host CPU process into device L1 or DRAM on one or more Tensix cores. The socket presents a page-oriented FIFO to both sides: the host writes pages in, the device kernel consumes them. Two distinct transport modes are available -- HOST_PUSH and DEVICE_PULL -- each with a different physical data path, latency profile, and throughput characteristic. This section traces the bytes through both modes, explains when to choose each, documents the full API surfaces, and covers the socket lifecycle.

---

## 4.1.1 H2D Overview

Every H2D socket connects exactly one host-side producer to one device-side consumer core. The host creates the socket, configures a page size, and writes pages. The device kernel creates a `SocketReceiverInterface`, waits for pages, processes them, and acknowledges consumption. A pair of counters -- `bytes_sent` and `bytes_acked` -- track progress and prevent the host from overwriting data the device has not yet consumed.

```
  Host CPU process                        Device (Tensix core)
 +------------------+                    +----------------------+
 |  H2DSocket       |     PCIe bus       |  SocketReceiver-     |
 |   write()  ------+---> (TLB or NOC) --+->  Interface         |
 |   barrier()      |                    |   socket_wait_for_   |
 |                  |                    |     pages()          |
 +------------------+                    +----------------------+
         |                                        |
         +--- bytes_sent / bytes_acked -----------+
                    (flow control)
```

The choice of HOST_PUSH vs DEVICE_PULL determines who initiates the PCIe data transfer, where intermediate buffers live, and where each counter is placed.

---

## 4.1.2 HOST_PUSH Mode

In HOST_PUSH mode, the host CPU writes data directly into the device L1 FIFO through a PCIe TLB (Translation Lookaside Buffer) window. This is a host-initiated posted write -- the CPU pushes bytes across the PCIe bus and they land in device L1 without any device-side initiation.

### Follow the Bytes: HOST_PUSH

```
  Host user buffer          PCIe TLB window           Device L1 FIFO
 +----------------+        +---------------+         +----------------+
 | page data      | --1--> | mapped to     | --2-->  | page lands in  |
 |                |        | device L1     |         | circular FIFO  |
 +----------------+        +---------------+         +----------------+
                                                            |
  Host pinned memory        PCIe (NOC write)          3. Device kernel
 +----------------+  <--4-- +---------------+            processes page
 | bytes_acked    |         | device writes |            then updates
 | (polled by     |         | ack counter   |            bytes_acked
 |  host)         |         | to host       |            in device L1
 +----------------+         +---------------+         +----------------+
```

Step-by-step:

1. **Host writes `bytes_sent` to device L1.** The host increments its local `bytes_sent` tracking and writes the updated value into device L1 via a TLB-mapped PCIe write. The device can read this counter directly from its own L1.
2. **Host writes page data to device L1 FIFO.** The host pushes the page contents through the same TLB window into the circular FIFO region in device L1. This is a PCIe posted write -- the host does not wait for a device-side acknowledgement of the data landing.
3. **Device kernel consumes the page.** The device kernel calls `socket_wait_for_pages()`, which spins on the `bytes_sent` counter in L1. Once sufficient pages are available, the kernel reads the data directly from L1, processes it, and calls `socket_pop_pages()` followed by `socket_notify_sender()`.
4. **Device updates `bytes_acked` in device L1, then mirrors the value to host-pinned memory.** The device kernel first writes the updated `bytes_acked` into its own L1 (the canonical location, same as DEVICE_PULL), then the NOC engine issues a write through PCIe that deposits the same value into host-pinned memory. The host polls this pinned-memory mirror to determine how much FIFO space has been freed.

**Why HOST_PUSH needs vIOMMU:** Step 4 is a device-initiated PCIe transaction (NOC write to host memory). The vIOMMU translates the device's physical address into the correct host virtual address for the pinned memory region.

### Typical use case

Token injection in interactive LLM serving. The host receives a user prompt, tokenizes it, and pushes tokens into device L1 with minimal latency. The small payload size (a few hundred bytes per token batch) makes the TLB write overhead negligible compared to the transfer itself.

---

## 4.1.3 DEVICE_PULL Mode

In DEVICE_PULL mode, the host writes data into a pinned host memory FIFO and signals the device. The device NOC engine then reads the data from host memory into device L1 over PCIe. This is a device-initiated read -- the device pulls bytes from host memory.

### Follow the Bytes: DEVICE_PULL

```
  Host user buffer          Host pinned FIFO          Device L1
 +----------------+        +----------------+        +----------------+
 | page data      | --1--> | staged in      |        |                |
 |                |        | pinned memory  | --3--> | NOC read pulls |
 +----------------+        +----------------+        | data into L1   |
                                                     +----------------+
                                                            |
  (Host writes bytes_sent                             4. Device kernel
   to device L1 via TLB)                                 processes page
         --2-->                                          updates
                                                         bytes_acked
  Host reads bytes_acked   <--5-- TLB read               in device L1
  from device L1 when             (PCIe RT)
  checking FIFO space
```

Step-by-step:

1. **Host writes page data to pinned host memory FIFO.** The page contents are copied into a pinned (non-pageable) region of host DRAM that is accessible to the device over PCIe.
2. **Host writes `bytes_sent` to device L1.** The host updates the `bytes_sent` counter in device L1 via a TLB-mapped PCIe write, signaling the device that new data is available in the host FIFO.
3. **Device NOC engine reads data from host memory.** The device kernel detects the updated `bytes_sent`, then issues one or more NOC read transactions to pull the page data from host-pinned memory into device L1. Multiple reads can be pipelined -- the device can issue the next read before the previous one completes, overlapping PCIe latency with processing.
4. **Device updates `bytes_acked` in device L1.** After consuming the data, the device kernel updates `bytes_acked` in its own L1. Since both `bytes_sent` and `bytes_acked` reside in device L1 in this mode, the device can check available space without any PCIe round trip. The host reads `bytes_acked` from device L1 via TLB read when it needs to check FIFO space.

**Pipelining advantage:** DEVICE_PULL can pipeline multiple outstanding NOC reads over PCIe, amortizing the PCIe round-trip latency across many pages. This makes it significantly more efficient for large bulk transfers.

### Typical use case

Loading large weight tensors or training data batches. The host stages a large buffer in pinned memory and the device pulls it in at sustained PCIe bandwidth using pipelined reads.

---

## 4.1.4 Mode Selection Guide

| Criterion | HOST_PUSH | DEVICE_PULL |
|-----------|-----------|-------------|
| Who initiates transfer | Host CPU | Device NOC engine |
| Latency for small payloads | Lower (single TLB write) | Higher (NOC read round trip) |
| Throughput for large payloads | Limited by single TLB write path | Higher (pipelined NOC reads) |
| Host memory required | None (writes directly to device L1) | Pinned host FIFO (fifo_size bytes) |
| bytes_acked read path | Host polls pinned memory (fast) | Host reads device L1 via TLB (slower) |
| Best for | Token injection, control signals, interactive serving | Tensor loading, training data, bulk transfers |
| Production example | `pipeline_block` in `op.py` | Large activation prefetch |

**Rule of thumb:** If the transfer is latency-sensitive and small (under ~4 KB), use HOST_PUSH. If throughput-sensitive and large (over ~64 KB), use DEVICE_PULL. For sizes in between, benchmark both.

---

## 4.1.5 API Surface

### Constructor

```python
# Source: h2d_socket.hpp / ttnn Python bindings
h2d = ttnn.H2DSocket(
    mesh_device,                    # MeshDevice handle
    recv_core=MeshCoreCoord(...),   # Target Tensix core on device
    buffer_type=BufferType.L1,      # L1 or DRAM
    fifo_size=fifo_size,            # Circular FIFO size in bytes
    h2d_mode=H2DMode.HOST_PUSH     # HOST_PUSH or DEVICE_PULL
)
```

Allocates the device-side FIFO, pins host memory (for DEVICE_PULL or HOST_PUSH acknowledgement), and sets up PCIe TLB mappings.

### set_page_size

```python
# Source: h2d_socket.hpp
h2d.set_page_size(page_size_bytes)
```

Sets the page granularity for subsequent `write()` calls. Minimum 64 bytes (NOC word width), 16-byte aligned, FIFO size must be an exact multiple. Call before the first `write()`.

### write

```python
# Source: h2d_socket.hpp
h2d.write(data, num_pages)
```

Writes `num_pages` pages from `data` into the device FIFO. Blocks if the FIFO is full. In HOST_PUSH, data flows through the TLB window. In DEVICE_PULL, data is copied to pinned host FIFO and `bytes_sent` is updated in device L1.

### barrier

```python
# Source: h2d_socket.hpp
h2d.barrier()
```

Blocks the host until the device has acknowledged all previously written pages (i.e., `bytes_acked == bytes_sent`).

### export_descriptor and connect

```python
# Source: h2d_socket.hpp
# Owner process:
descriptor_path = h2d.export_descriptor(socket_id)

# Connector process (different OS process):
h2d_remote = ttnn.H2DSocket.connect(socket_id, timeout_ms=5000)
```

`export_descriptor()` serializes connection state (TLB mappings, FIFO addresses, counters) into a flatbuffer in `/dev/shm`. A second process calls `connect()` with the same `socket_id` and receives a handle that operates on raw PCIe mappings without `MetalContext`, enabling multi-process serving architectures.

---

## 4.1.6 vIOMMU Requirement

Both modes require vIOMMU because both involve device-initiated PCIe transactions: HOST_PUSH needs it for the `bytes_acked` NOC write to host memory (step 4); DEVICE_PULL needs it for the NOC read of page data from host memory (step 3). Verify vIOMMU is active:

```bash
dmesg | grep -i iommu
# Expected: lines showing IOMMU enabled and devices mapped
```

**Warning:** If vIOMMU is not enabled, H2D socket creation will fail or produce silent data corruption. There is no software fallback.

---

## 4.1.7 Device Kernel: SocketReceiverInterface

The device-side kernel uses `SocketReceiverInterface` to consume pages from the H2D FIFO.

```cpp
// Source: socket_receiver_interface.hpp
// In device kernel code:

auto receiver = create_receiver_socket_interface();
set_receiver_socket_page_size(receiver, page_size_bytes);

// Main receive loop:
while (has_work) {
    socket_wait_for_pages(receiver, num_pages);   // Spin until num_pages available
    // ... read page data from L1 FIFO ...
    socket_pop_pages(receiver, num_pages);         // Advance read pointer
    socket_notify_sender(receiver);                // Update bytes_acked
}
```

| Function | Purpose |
|----------|---------|
| `create_receiver_socket_interface()` | Allocates and initializes the receiver state from socket metadata in L1 |
| `set_receiver_socket_page_size(iface, size)` | Sets the page size; must match the host-side `set_page_size()` call |
| `socket_wait_for_pages(iface, n)` | Spins until at least `n` pages are available (checks `bytes_sent` in L1) |
| `socket_pop_pages(iface, n)` | Advances the FIFO read pointer by `n` pages |
| `socket_notify_sender(iface)` | Writes updated `bytes_acked` so the host knows space has been freed |
| `update_socket_config(iface)` | Reloads socket configuration (used after dynamic reconfiguration) |

**Warning:** `socket_notify_sender()` must be called after `socket_pop_pages()`, not before. Calling it before pop would advertise free space that the kernel is still reading, risking data corruption if the host writes new data into that region.

---

## 4.1.8 Lifecycle and Teardown

H2D sockets follow RAII semantics. The destructor behavior depends on whether the socket was created as an owner or obtained via `connect()`.

| Step | Owner | Connector |
|------|-------|-----------|
| 1 | Wait for device ack (`bytes_acked == bytes_sent`) | Wait for device ack |
| 2 | Unpin host memory / release TLB mappings | Unmap shared memory from `connect()` |
| 3 | Unlink `/dev/shm` descriptor (if exported) | No-op (only owner unlinks) |
| 4 | Deallocate device FIFO and counter memory | No-op (only owner deallocates) |

**Warning:** Destroying an H2D socket while the device kernel is still calling `socket_wait_for_pages()` will cause the kernel to hang indefinitely. Always ensure the device kernel has exited its receive loop before destroying the host-side socket.

---

## Key Takeaways

- H2D sockets offer two modes: HOST_PUSH (low-latency, host-initiated TLB writes) and DEVICE_PULL (high-throughput, device-initiated pipelined NOC reads).
- In HOST_PUSH, the host writes both data and `bytes_sent` directly into device L1; the device updates `bytes_acked` in device L1 (canonical) and mirrors it to host-pinned memory via NOC write through PCIe for fast host polling.
- In DEVICE_PULL, the host stages data in pinned memory and writes `bytes_sent` to device L1; the device pulls data via NOC read and maintains `bytes_acked` in its own L1.
- Both modes require vIOMMU because both involve device-initiated PCIe transactions.
- The device kernel uses `SocketReceiverInterface` with a wait-pop-notify loop.
- Cross-process sharing uses `export_descriptor()` / `connect()` via flatbuffer descriptors in `/dev/shm`.

---

**Previous:** [`index.md`](./index.md) | **Next:** [`02_d2h_socket_api_and_external_config_buffer.md`](./02_d2h_socket_api_and_external_config_buffer.md) | **Up:** [`index.md`](./index.md)
