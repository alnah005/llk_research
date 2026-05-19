# 4.2 -- D2H Socket API and ExternalConfigBuffer

A D2H socket streams data from a device Tensix core back to the host CPU. Unlike H2D sockets, which offer two transfer modes, D2H has a single transport: the device kernel pushes data into host-pinned memory via NOC writes through PCIe. The host polls for pages, reads them, and acknowledges consumption. This section traces the data path, documents both APIs, explains ExternalConfigBuffer, and covers the lifecycle.

---

## 4.2.1 D2H Overview

Every D2H socket connects one device-side sender core to one host-side consumer process. The device kernel produces pages using a `SocketSenderInterface`. The host-side `D2HSocket` object reads pages from a pinned-memory FIFO and acknowledges consumption so the device can reuse FIFO slots.

```
  Device (Tensix core)                       Host CPU process
 +----------------------+                   +------------------+
 |  SocketSender-       |     PCIe bus      |  D2HSocket       |
 |   Interface          |                   |                  |
 |   socket_push_  -----+---> NOC write --->|   has_data()     |
 |     pages()          |    through PCIe   |   read()         |
 |                      |                   |   barrier()      |
 +----------------------+                   +------------------+
         |                                          |
         +--- bytes_sent / bytes_acked -------------+
                    (flow control)
```

The D2H direction is the mirror image of H2D HOST_PUSH, but with roles reversed: the device is the writer and the host is the reader.

---

## 4.2.2 Transport Mechanism: Follow the Bytes

```
  Device L1 FIFO          PCIe (NOC write)        Host pinned FIFO
 +----------------+      +----------------+      +----------------+
 | page produced  | --1->| device NOC     | --2->| page lands in  |
 | by kernel      |      | writes to host |      | pinned memory  |
 +----------------+      +----------------+      +----------------+
                                                        |
  Device L1               PCIe (TLB write)         3. Host reads
 +----------------+  <-4- +----------------+          page from
 | bytes_acked    |       | host writes    |          pinned mem
 | (read by       |       | ack back to    |
 |  device)       |       | device L1      |
 +----------------+       +----------------+
```

Step-by-step:

1. **Device kernel produces a page in L1.** The kernel writes output data into the sender FIFO region in device L1, then calls `socket_push_pages()` to advance the write pointer.
2. **Device NOC writes page data and `bytes_sent` to host-pinned memory.** The device NOC engine issues a posted write through PCIe that deposits the page contents into the host-pinned FIFO. The device also writes the updated `bytes_sent` counter to a separate location in host-pinned memory. From the host's perspective, both the data and the progress counter appear in local (pinned) memory -- no PCIe read is needed to detect new data.
3. **Host reads page from pinned memory.** The host calls `has_data()` or `pages_available()` to check whether new pages have arrived (by reading `bytes_sent` from pinned memory -- a local memory read with no PCIe traffic). When pages are available, `read()` copies the data from the pinned FIFO into the caller's buffer.
4. **Host writes `bytes_acked` to device L1.** When `read()` is called with `notify_sender=True` (the default), the host writes the updated `bytes_acked` counter back to device L1 via a TLB-mapped PCIe write. The device kernel uses L1 cache invalidation to read the updated counter and determine how much FIFO space has been freed.

**Why D2H needs vIOMMU:** Step 2 is a device-initiated PCIe transaction (NOC write from device to host memory). The vIOMMU translates the device physical address to the correct host address for the pinned-memory region.

---

## 4.2.3 Host-Side API

### Constructor (standard)

```python
# Source: d2h_socket.hpp / ttnn Python bindings
d2h = ttnn.D2HSocket(
    mesh_device,                      # MeshDevice handle
    sender_core=MeshCoreCoord(...),   # Source Tensix core on device
    fifo_size=fifo_size               # Circular FIFO size in bytes
)
```

Allocates the host-pinned FIFO, device L1 metadata, and configures PCIe mappings.

### Constructor (with ExternalConfigBuffer)

```python
# Source: d2h_socket.hpp
d2h = ttnn.D2HSocket(
    mesh_device,
    sender_core=MeshCoreCoord(...),
    fifo_size=fifo_size,
    external_config=ExternalConfigBuffer(address=l1_address)
)
```

See Section 4.2.4 for when and why to use this form.

### set_page_size

```python
# Source: d2h_socket.hpp
d2h.set_page_size(page_size_bytes)
```

Sets the page granularity. Constraints are the same as H2D: minimum 64 bytes, 16-byte aligned, FIFO size must be an exact multiple. Call before the first `read()`.

**Warning:** If alignment causes the internal read pointer to wrap, `set_page_size()` blocks until the device sends enough data to cover the alignment adjustment. This can stall the host thread if called mid-stream while the device has not yet produced sufficient pages.

### has_data and pages_available

```python
# Source: d2h_socket.hpp
if d2h.has_data():          # True if bytes_sent > bytes_acked (at least one page ready)
    n = d2h.pages_available()  # Number of complete pages available for reading
```

Both are non-blocking local memory reads (polling `bytes_sent` in host-pinned memory) -- zero PCIe traffic.

### read

```python
# Source: d2h_socket.hpp
d2h.read(data, num_pages, notify_sender=True)
```

Reads `num_pages` from the pinned-memory FIFO into the caller's buffer `data`. If fewer than `num_pages` are available, `read()` blocks until the device pushes enough data.

The `notify_sender` parameter controls acknowledgement: `True` (default) immediately writes `bytes_acked` to device L1, freeing FIFO space; `False` defers the acknowledgement for batching -- see Section 4.3.6 in [`03_host_device_flow_control.md`](./03_host_device_flow_control.md).

**Warning:** Setting `notify_sender=False` without eventually acknowledging will cause the device kernel to stall when the FIFO fills. The device's `socket_reserve_pages()` will spin indefinitely waiting for free space.

### barrier and discard_pending_pages

```python
# Source: d2h_socket.hpp
d2h.barrier()                  # Block until all sent pages are read and acknowledged
d2h.discard_pending_pages()    # Rebase bytes_acked = bytes_sent, discard unread pages
```

`barrier()` drains the FIFO completely. `discard_pending_pages()` is for error recovery (timeouts, cancelled requests, rejected speculative decode) -- it fast-forwards `bytes_acked` without touching the data region, making the FIFO appear empty to both sides.

### export_descriptor and connect

```python
# Source: d2h_socket.hpp
# Owner process:
descriptor_path = d2h.export_descriptor(socket_id)

# Connector process:
d2h_remote = ttnn.D2HSocket.connect(socket_id, timeout_ms=5000)
```

Same semantics as H2D: owner serializes to `/dev/shm`, connector gets a handle on raw PCIe mappings without `MetalContext`.

---

## 4.2.4 ExternalConfigBuffer

By default, the D2H constructor allocates device L1 space for socket metadata (socket descriptor, `bytes_acked` counter location, downstream encoding). When multiple D2H sockets share a sender core or the kernel needs precise L1 layout control, pre-allocate this region and pass it via `ExternalConfigBuffer`:

```python
# Source: d2h_socket.hpp
ext_config = ExternalConfigBuffer(address=l1_address)  # L1 address on the sender core
d2h = ttnn.D2HSocket(mesh_device, sender_core, fifo_size, external_config=ext_config)
```

The underlying struct has a single field: `uint32_t address` -- the starting L1 byte address where socket metadata will be placed. The caller must ensure this region does not overlap with other L1 allocations.

### How much space is needed

```python
# Source: d2h_socket.hpp
size = ttnn.D2HSocket.required_config_buffer_size()
```

`required_config_buffer_size()` returns the total bytes needed for:

| Component | Description |
|-----------|-------------|
| `sender_socket_md` | Socket metadata descriptor |
| `bytes_acked` counter | Acknowledgement counter location |
| `sender_downstream_encoding` | Routing info for multi-hop paths |

Each component is individually rounded up to `L1_ALIGNMENT`. The sum is the minimum allocation.

### When to use ExternalConfigBuffer

- **Multiple sockets on one core:** prevents metadata overlap when two D2H sockets share a sender core.
- **Deterministic L1 layout:** kernel needs compile-time knowledge of metadata addresses.
- **Memory budgeting:** co-locating socket metadata with other structures under L1 pressure.

---

## 4.2.5 Device Kernel: SocketSenderInterface

The device-side kernel uses `SocketSenderInterface` to produce pages for the D2H FIFO.

```cpp
// Source: socket_sender_interface.hpp
// In device kernel code:

auto sender = create_sender_socket_interface();
set_sender_socket_page_size(sender, page_size_bytes);

// Main send loop:
while (has_output) {
    socket_reserve_pages(sender, num_pages);   // Spin until FIFO has space
    // ... write output data into L1 FIFO region ...
    socket_push_pages(sender, num_pages);       // NOC write to host + advance pointer
    socket_notify_receiver(sender);             // Update bytes_sent in host memory
}

// After all pages sent:
socket_barrier(sender);  // Wait for host to ack all pages
```

| Function | Purpose |
|----------|---------|
| `create_sender_socket_interface()` | Allocates and initializes sender state from socket metadata in L1 |
| `set_sender_socket_page_size(iface, size)` | Sets page size; must match host-side `set_page_size()` |
| `socket_reserve_pages(iface, n)` | Spins until at least `n` pages of FIFO space are free (checks `bytes_acked` in L1 via cache invalidation) |
| `socket_push_pages(iface, n)` | NOC writes `n` pages from L1 to host-pinned memory; advances write pointer |
| `socket_notify_receiver(iface)` | Writes `bytes_sent` to host-pinned memory so the host detects new data |
| `socket_barrier(iface)` | Waits until host has acknowledged all sent pages |
| `update_socket_config(iface)` | Reloads socket configuration after dynamic reconfiguration |

**Warning:** `socket_notify_receiver()` must be called after `socket_push_pages()`, not before. Notifying before the data has been written would cause the host to read stale or partial data from pinned memory.

---

## 4.2.6 Lifecycle and Teardown

D2H sockets follow RAII semantics. Teardown behavior depends on ownership:

| Step | Owner (standard) | Owner (ExternalConfigBuffer) | Connector |
|------|-------------------|------------------------------|-----------|
| 1 | Wait for host to read all pages | Same | Wait for outstanding reads |
| 2 | Unpin host memory, release PCIe mappings | Same | Unmap shared memory |
| 3 | Unlink `/dev/shm` descriptor if exported | Same | No-op |
| 4 | Deallocate device L1 FIFO + metadata | Does NOT free ExternalConfigBuffer region | No-op |

Cross-process pattern: the owner calls `export_descriptor(socket_id)`, then a separate process calls `D2HSocket.connect(socket_id)` to obtain a handle that bypasses `MetalContext`. Both processes can read from the socket; only the owner manages device resources.

---

## Key Takeaways

- D2H sockets use a single transport mode: device NOC writes push data through PCIe into host-pinned memory.
- The host reads `bytes_sent` from local pinned memory (zero PCIe traffic) and writes `bytes_acked` back to device L1 via TLB write.
- `notify_sender=False` in `read()` enables batched acknowledgement for higher throughput; `discard_pending_pages()` enables error recovery by fast-forwarding the ack counter.
- `ExternalConfigBuffer` controls where socket metadata lives in device L1 -- essential for shared sender cores or deterministic layout.
- The device kernel uses `SocketSenderInterface` with a reserve-push-notify loop that mirrors the receiver's wait-pop-notify pattern.
- vIOMMU is required because the device issues NOC writes through PCIe to deposit data in host memory.

---

**Previous:** [`01_h2d_socket_modes_and_api.md`](./01_h2d_socket_modes_and_api.md) | **Next:** [`03_host_device_flow_control.md`](./03_host_device_flow_control.md) | **Up:** [`index.md`](./index.md)
