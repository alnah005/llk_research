# Chapter 4.3 -- Socket Abstractions

## Context

The mesh API (Section 4.1) and connection managers (Section 4.2) provide the raw machinery for sending packets across the fabric. However, they operate at the granularity of individual packets with explicit route setup and flow control. This section ascends the transparency spectrum to examine the **socket layer** -- a higher-level abstraction that provides FIFO-based, flow-controlled channels between pairs of cores, potentially on different chips.

The socket API spans three layers: the **device-side kernel interface** (`socket_api.h`) providing `SocketSenderInterface` and `SocketReceiverInterface`; the **host-side MeshSocket** (`mesh_socket.hpp`) managing cross-device socket configuration and memory allocation; and the **TTNN-level FabricSocket** (`fabric_socket.hpp`) providing tensor-level `send()`/`recv()` semantics through the `ISocket` interface. Together, these represent the closest existing approach to "transparent" cross-chip data movement -- a sender pushes pages into a logical FIFO, and the runtime handles the physical delivery to the receiver's L1 memory.

---

## 4.3.1 Device-Side Data Structures

The device-side socket interface is built on metadata structures defined in `hostdev/socket.h`. There are distinct sender and receiver structures, each with variants for device-to-device (D2D), host-to-device (H2D), and device-to-host (D2H) communication.

### Sender-Side Metadata

The `sender_socket_md` structure is the on-device config buffer for a sender core:

```cpp
struct sender_socket_md {
    uint32_t bytes_sent = 0;
    uint32_t num_downstreams = 0;
    uint32_t write_ptr = 0;
    uint32_t downstream_bytes_sent_addr = 0;
    uint32_t downstream_fifo_addr = 0;
    uint32_t downstream_fifo_total_size = 0;
    uint32_t is_d2h = 0;
};
// Followed in memory by:
//   uint32_t bytes_acked_array[num_downstreams]
//   sender_downstream_encoding[num_downstreams]
```

The L1 layout after the metadata contains two variable-length arrays: a `bytes_acked` array (one `uint32_t` per downstream receiver, each L1-aligned) and a `sender_downstream_encoding` array encoding each receiver's location.

### Per-Downstream Encoding: A Proto-Global Address

Each downstream receiver is identified by a `sender_downstream_encoding` union:

```cpp
struct d2d_sender_socket_md {
    uint32_t downstream_mesh_id;
    uint32_t downstream_chip_id;
    uint32_t downstream_noc_y;
    uint32_t downstream_noc_x;
};

struct d2h_sender_socket_md {
    uint32_t bytes_sent_addr_hi;
    uint32_t data_addr_hi;
    uint32_t pcie_xy_enc;
};

union sender_downstream_encoding {
    d2h_sender_socket_md d2h;
    d2d_sender_socket_md d2d;
};
```

The D2D encoding is essentially a **global core address**: `(mesh_id, chip_id, noc_x, noc_y)` uniquely identifies any core in a multi-mesh system. This is the closest the current system comes to a flat global address for cross-chip targeting -- and the first place in TT-Metal where such a complete cross-device coordinate appears as a single data structure.

### Receiver-Side Metadata

```cpp
struct receiver_socket_md {
    uint32_t bytes_sent;
    uint32_t read_ptr;
    uint32_t fifo_addr;
    uint32_t fifo_total_size;
    uint32_t bytes_acked;
    uint32_t is_h2d;
    union {
        h2d_socket_md h2d;
        d2d_recv_socket_md d2d;
    };
};
```

The D2D receiver variant stores the upstream source identity:

```cpp
struct d2d_recv_socket_md {
    uint32_t upstream_mesh_id;
    uint32_t upstream_chip_id;
    uint32_t upstream_noc_y;
    uint32_t upstream_noc_x;
    uint32_t upstream_bytes_acked_addr;
};
```

This allows the receiver to send acknowledgments back to the sender's `bytes_acked` location via fabric.

### Runtime Interface Structs

The `SocketSenderInterface` and `SocketReceiverInterface` are the working copies used by kernel code:

```cpp
struct SocketSenderInterface {
    uint32_t config_addr;
    uint32_t write_ptr;
    uint32_t bytes_sent;
    uint32_t bytes_acked_base_addr;
    uint32_t page_size;
    uint32_t num_downstreams;
    uint32_t downstream_fifo_total_size;
    uint32_t downstream_fifo_curr_size;
    uint32_t downstream_fifo_addr;
    uint32_t downstream_bytes_sent_addr;
    uint32_t is_d2h;
    union {
        D2HSocketInterface d2h;
        D2DSocketSendInterface d2d;
    };
};

struct SocketReceiverInterface {
    uint32_t config_addr;
    uint32_t read_ptr;
    uint32_t bytes_acked;
    uint32_t bytes_sent_addr;
    uint32_t page_size;
    uint32_t fifo_addr;
    uint32_t fifo_total_size;
    uint32_t fifo_curr_size;
    uint32_t is_h2d;
    union {
        H2DSocketInterface h2d;
        D2DSocketRecvInterface d2d;
    };
};
```

These are created from the on-device config buffer via factory functions:

```cpp
SocketSenderInterface socket = create_sender_socket_interface(config_addr);
SocketReceiverInterface socket = create_receiver_socket_interface(config_addr);
```

## 4.3.2 Socket Variants: D2D, H2D, and D2H

The socket system supports three communication patterns through the `is_d2h` and `is_h2d` flags:

| Variant | Direction | Transport | Key Fields |
|---------|-----------|-----------|------------|
| **D2D** | Device to Device | Fabric (Ethernet) | `downstream_mesh_id`, `downstream_chip_id`, `downstream_noc_x/y` |
| **D2H** | Device to Host | PCIe | `bytes_sent_addr_hi`, `data_addr_hi`, `pcie_xy_enc` |
| **H2D** | Host to Device | PCIe | `bytes_acked_addr_lo/hi`, `data_addr_lo/hi`, `pcie_xy_enc` |

The D2D variant is fabric-aware: its notification functions (`fabric_socket_notify_receiver`, `fabric_socket_notify_sender`) construct fabric packets with unicast inline writes to deliver `bytes_sent` or `bytes_acked` updates across chips. The H2D/D2H variants bypass the fabric entirely, using PCIe for data transfer and acknowledgment.

## 4.3.3 Device-Side Sender Flow

The sender operates in a strict three-step protocol for each batch of pages:

### 1. Page Size Configuration

```cpp
void set_sender_socket_page_size(SocketSenderInterface& socket, uint32_t page_size);
```

This aligns the write pointer to the new page size, computes the page-aligned FIFO size (dropping any remainder), and handles wrapping. If alignment causes the write pointer to cross the FIFO boundary, the function adjusts `bytes_sent` to account for the dead space.

### 2. Reserve Pages (Flow Control)

```cpp
void socket_reserve_pages(const SocketSenderInterface& socket, uint32_t num_pages);
```

This is the backpressure mechanism. It spin-waits on each downstream's `bytes_acked` pointer until:

$$
\text{free} = \text{fifoTotalSize} - (\text{bytesSent} - \text{bytesAcked}) \geq \text{numPages} \times \text{pageSize}
$$

The check iterates over all `num_downstreams` entries, ensuring that **every** downstream has sufficient space. This is a blocking call -- the kernel cannot proceed until all downstreams are ready.

### 3. Push Pages

```cpp
void socket_push_pages(SocketSenderInterface& socket, uint32_t num_pages);
```

Advances `write_ptr` and `bytes_sent` by `num_pages * page_size`. The write pointer wraps circularly within the page-aligned portion of the FIFO. When wrapping occurs, any alignment waste is added to `bytes_sent` so that the sender and receiver stay synchronized.

### 4. Notify Receiver

For D2D sockets over fabric:

```cpp
void fabric_socket_notify_receiver(
    const SocketSenderInterface& socket,
    WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* fabric_header_addr);
```

For each downstream, this function:
1. Reads the downstream encoding to get the target's NOC coordinates and mesh/chip IDs.
2. Calls `fabric_set_unicast_route()` to set the packet header's routing fields.
3. Encodes a `NocUnicastInlineWriteCommandHeader` targeting the downstream's `bytes_sent_addr` with the current `bytes_sent` value.
4. Sends the header-only packet via the fabric connection.

The notification is an inline write (32-bit value embedded in the packet header, no payload) carrying the updated `bytes_sent` value to the receiver's `bytes_sent` field.

For local (same-chip) sockets, `socket_notify_receiver()` uses a direct `noc_inline_dw_write`.

### 5. Barrier

```cpp
void socket_barrier(const SocketSenderInterface& socket);
```

Blocks until all downstreams have acknowledged all sent data: `bytes_sent == *bytes_acked_ptr` for every downstream.

## 4.3.4 Device-Side Receiver Flow

### 1. Wait for Pages

```cpp
bool socket_wait_for_pages(
    const SocketReceiverInterface& socket,
    uint32_t num_pages,
    uint32_t early_exit_iter_count = 0);
```

Spin-waits until `*bytes_sent_ptr - bytes_acked >= num_bytes`. The optional `early_exit_iter_count` enables non-blocking polling -- if the data has not arrived after the specified number of iterations, returns `false`.

### 2. Pop Pages

```cpp
void socket_pop_pages(SocketReceiverInterface& socket, uint32_t num_pages);
```

Advances `read_ptr` and `bytes_acked`, handling wrap-around.

### 3. Notify Sender (ACK)

For D2D sockets over fabric:

```cpp
void fabric_socket_notify_sender(
    const SocketReceiverInterface& socket,
    WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* fabric_header_addr);
```

Sends the current `bytes_acked` value to the upstream sender's `upstream_bytes_acked_addr` via a fabric inline write. There is also a stateful variant (`fabric_socket_notify_sender_stateful`) that skips route computation when the header is pre-configured.

### 4. Circular Buffer Integration

```cpp
void assign_local_cb_to_socket(const SocketReceiverInterface& socket, uint32_t cb_id);
```

This maps a socket's FIFO region onto a local circular buffer interface (`LocalCBInterface`), allowing TRISC compute cores to consume socket data through the standard CB API. Compute kernels treat socket data as if it were a local circular buffer, with no awareness of the cross-chip origin.

This is a particularly important GSRP building block: it demonstrates that data arriving over the fabric can be presented to compute kernels through the same interface as local data. GSRP extends this idea to make **all remote L1** accessible through the same mechanism.

## 4.3.5 `MeshCoreCoord`: Cross-Device Core Addressing

The host-side socket layer introduces a unified addressing scheme for cores across a mesh:

```cpp
struct MeshCoreCoord {
    MeshCoordinate device_coord;  // Position in the mesh grid
    CoreCoord core_coord;         // Core position within the device
};
```

A `MeshCoreCoord` uniquely identifies any core on any device in the mesh. It is the host-level equivalent of the device-side `(mesh_id, chip_id, noc_x, noc_y)` tuple.

## 4.3.6 `MeshSocket`: Host-Side Socket Management

The `MeshSocket` class in `mesh_socket.hpp` manages the lifecycle of a socket pair:

```cpp
class MeshSocket {
public:
    MeshSocket(const std::shared_ptr<MeshDevice>& device, const SocketConfig& config);

    static std::pair<MeshSocket, MeshSocket> create_socket_pair(
        const std::shared_ptr<MeshDevice>& sender,
        const std::shared_ptr<MeshDevice>& receiver,
        const SocketConfig& base_config);

    std::shared_ptr<MeshBuffer> get_data_buffer() const;
    std::shared_ptr<MeshBuffer> get_config_buffer() const;
    const SocketConfig& get_config() const;
    SocketEndpoint get_socket_endpoint_type() const;  // SENDER or RECEIVER
    std::vector<MeshCoreCoord> get_active_cores() const;
};
```

### `SocketConfig`

The `SocketConfig` fully specifies a socket:

```cpp
struct SocketConfig {
    std::vector<SocketConnection> socket_connection_config;
    SocketMemoryConfig socket_mem_config;
    std::optional<MeshId> sender_mesh_id;
    std::optional<MeshId> receiver_mesh_id;
    multihost::Rank sender_rank{0};
    multihost::Rank receiver_rank{0};
    std::shared_ptr<multihost::DistributedContext> distributed_context;
};
```

The `socket_connection_config` is a vector of `SocketConnection` entries, each mapping a sender `MeshCoreCoord` to a receiver `MeshCoreCoord`. Each socket connection is strictly 1:1.

### `SocketMemoryConfig`

```cpp
struct SocketMemoryConfig {
    BufferType socket_storage_type = BufferType::L1;
    uint32_t fifo_size = 0;
    std::optional<SubDeviceId> sender_sub_device;
    std::optional<SubDeviceId> receiver_sub_device;
};
```

The `socket_storage_type` can be `L1` or `DRAM`, controlling where the FIFO buffer is allocated.

### Multi-Host Support

The `sender_mesh_id`/`receiver_mesh_id` and `sender_rank`/`receiver_rank` fields support sockets that span multiple hosts in a distributed MPI-based configuration. The `distributed_context` provides the MPI-like communication primitives needed for initial handshaking between hosts.

### Socket Pair Creation

The static `create_socket_pair` method creates matched sender/receiver sockets:

1. Populates mesh IDs from the sender and receiver `MeshDevice` instances.
2. Creates the data buffer (allocated on the receiver mesh).
3. Creates config buffers (on both sender and receiver meshes).
4. Writes socket metadata (downstream encodings, FIFO addresses, etc.) to the config buffers.
5. Returns the pair as `(sender_socket, receiver_socket)`.

## 4.3.7 `FabricSocket` and `ISocket`: Tensor-Level Communication

The `FabricSocket` in `ttnn/api/ttnn/distributed/fabric_socket.hpp` wraps a `MeshSocket` and implements the `ISocket` interface for tensor-level send/recv:

```cpp
class ISocket {
public:
    virtual void send(const ttnn::Tensor& tensor) = 0;
    virtual void recv(ttnn::Tensor& tensor) = 0;
    virtual tt::tt_metal::distributed::multihost::Rank get_rank() const = 0;
    virtual std::shared_ptr<DistributedContext> get_distributed_context() const = 0;
};

class FabricSocket : public ISocket {
public:
    FabricSocket(const MeshSocket& mesh_socket);

    void send(const ttnn::Tensor& tensor) override;
    void recv(ttnn::Tensor& tensor) override;

    static std::unique_ptr<FabricSocket> create(
        const std::shared_ptr<MeshDevice>& mesh_device,
        multihost::Rank sender_rank,
        multihost::Rank receiver_rank,
        SocketConfig socket_config);
};
```

This is the highest-level abstraction: the user specifies sender and receiver ranks, and the socket handles tensor serialization, page management, and fabric routing transparently. A developer can call `socket.send(tensor)` without knowing anything about packet headers, connection management, or flow control.

## 4.3.8 D2H and H2D Socket Classes

### `D2HSocket` (Device-to-Host)

The `D2HSocket` provides streaming from a device core to host memory:

```cpp
class D2HSocket {
public:
    D2HSocket(const std::shared_ptr<MeshDevice>& mesh_device,
              const MeshCoreCoord& sender_core,
              uint32_t fifo_size);

    void set_page_size(uint32_t page_size);
    void read(void* data, uint32_t num_pages, bool notify_sender = true);
    void barrier(std::optional<uint32_t> timeout_ms = std::nullopt);
    uint32_t get_config_buffer_address() const;
};
```

The D2H socket allocates pinned host memory (via `PinnedMemory`) and maps it for DMA access. The device kernel writes data to the pinned buffer via PCIe NOC writes and signals completion by writing `bytes_sent`. The host side reads from the pinned buffer and writes `bytes_acked` back to the device. This requires vIOMMU to be enabled.

### `H2DSocket` (Host-to-Device)

The `H2DSocket` supports two transfer modes:

```cpp
enum class H2DMode : uint8_t {
    HOST_PUSH,    // Host writes to device L1 via TLB-mapped PCIe writes
    DEVICE_PULL,  // Host writes to pinned memory; device pulls via PCIe NOC reads
};

class H2DSocket {
public:
    H2DSocket(const std::shared_ptr<MeshDevice>& mesh_device,
              const MeshCoreCoord& recv_core,
              BufferType buffer_type,
              uint32_t fifo_size,
              H2DMode h2d_mode);

    void set_page_size(uint32_t page_size);
    void write(void* data, uint32_t num_pages);
    void barrier(std::optional<uint32_t> timeout_ms = std::nullopt);
};
```

Both modes use the same flow-control protocol (`bytes_sent`/`bytes_acked`), but differ in the data path: `HOST_PUSH` uses UMD TLB writes, while `DEVICE_PULL` uses pinned memory with device-initiated reads.

## 4.3.9 Boilerplate Comparison

The socket layer dramatically reduces boilerplate compared to the raw mesh API:

| Task | Raw mesh API (Files 01-02) | Socket API |
|------|---------------------------|------------|
| Specify destination | `dst_dev_id`, `dst_mesh_id`, NOC addr | `SocketConnection` config (host-side, once) |
| Manage connection | `open()` / `close()` per kernel | Implicit in socket create/destroy |
| Flow control | `wait_for_empty_write_slot()` per send | `socket_reserve_pages()` (FIFO-level) |
| Deliver acknowledgment | Manual semaphore management | `socket_notify_sender()` / `_receiver()` |
| Compute route | `fabric_set_unicast_route()` per send | Implicit (stored in downstream encoding) |
| Handle page alignment | Manual | `set_sender_socket_page_size()` handles wrap |

**Estimated device-side boilerplate for socket-based kernel:** 10-15 lines (create interface, set page size, reserve/push/notify loop). Compare with 25-45 lines for raw mesh API.

## GSRP Implications

The socket abstraction is the closest existing component to the GSRP vision. The `downstream_encoding` — `(mesh_id, chip_id, noc_x, noc_y)` — is essentially a **global address at core granularity**. The remaining step is extending this to byte-level addressing and making resolution automatic. The `assign_local_cb_to_socket` function is a proof of concept for GSRP transparency: compute kernels consume fabric-delivered data through standard circular buffers without knowing the data's origin. The principal gaps are: point-to-point (not address-based) routing, no read path, static topology, and explicit kernel-level page management.

## Key Takeaways

- The `sender_downstream_encoding` structure forms a de facto global core address — the closest existing primitive to a GSRP address.
- The reserve/push/notify sender flow with `bytes_sent`/`bytes_acked` tracking provides FIFO-based flow control across D2D, H2D, and D2H variants.
- `MeshSocket` handles host-side lifecycle; `FabricSocket` provides tensor-level `send()`/`recv()` via the `ISocket` interface; `assign_local_cb_to_socket` bridges sockets to compute kernels transparently.

## Source Code References

| File | Key Contents |
|------|-------------|
| `tt_metal/hw/inc/hostdev/socket.h` | `sender_socket_md`, `receiver_socket_md`, `SocketSenderInterface`, `SocketReceiverInterface`, `sender_downstream_encoding`. |
| `tt_metal/hw/inc/api/socket_api.h` | Device-side socket functions: `create_sender_socket_interface`, `socket_reserve_pages`, `socket_push_pages`, `socket_notify_receiver`, `fabric_socket_notify_receiver`, `fabric_socket_notify_sender`, `assign_local_cb_to_socket`. |
| `tt_metal/api/tt-metalium/experimental/sockets/mesh_socket.hpp` | `MeshSocket`, `MeshCoreCoord`, `SocketConfig`, `SocketConnection`, `SocketMemoryConfig`. |
| `ttnn/api/ttnn/distributed/fabric_socket.hpp` | `FabricSocket` implementing `ISocket` for tensor-level communication. |
| `ttnn/api/ttnn/distributed/isocket.hpp` | `ISocket` abstract interface. |
| `tt_metal/api/tt-metalium/experimental/sockets/d2h_socket.hpp` | `D2HSocket`: device-to-host streaming with pinned memory. |
| `tt_metal/api/tt-metalium/experimental/sockets/h2d_socket.hpp` | `H2DSocket`: host-to-device streaming with `HOST_PUSH` and `DEVICE_PULL` modes. |

---

**Next:** [`04_mesh_buffer_and_mesh_device.md`](./04_mesh_buffer_and_mesh_device.md)
