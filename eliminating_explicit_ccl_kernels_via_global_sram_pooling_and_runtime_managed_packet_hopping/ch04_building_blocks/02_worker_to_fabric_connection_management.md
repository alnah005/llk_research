# Chapter 4.2 -- Worker-to-Fabric Connection Management

## Context

Chapter 3 described how Ethernet Data Movers (EDMs) manage channels, flow control, and packet forwarding. File 01 of this chapter showed the mesh fabric API's send primitives, each of which requires a "sender interface" and a pre-opened connection to the fabric. This section descends one level on the transparency spectrum to examine **how Tensix worker cores discover, establish, and manage connections to the fabric routers**. Three classes form this layer: `WorkerToFabricEdmSenderImpl` (the low-level adapter), `FabricConnectionManager` (the V1 forward/backward model), and `RoutingPlaneConnectionManager` (the V2 multi-plane model). On the host side, `append_fabric_connection_rt_args` and `append_routing_plane_connection_manager_rt_args` prepare the runtime arguments that these device-side classes consume.

This layer is the most boilerplate-heavy component that a GSRP runtime would need to absorb: connection lifecycle management accounts for an estimated 30-50 lines of device-side code and 15-25 lines of host-side code per kernel that communicates over the fabric.

---

## 4.2.1 `WorkerToFabricEdmSenderImpl`: The Core Adapter

The `WorkerToFabricEdmSenderImpl` template is the fundamental bridge between a Tensix worker and a single EDM channel. It is defined in `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` and parameterized on two compile-time values:

```cpp
template <bool I_USE_STREAM_REG_FOR_CREDIT_RECEIVE, uint8_t EDM_NUM_BUFFER_SLOTS = 0>
struct WorkerToFabricEdmSenderImpl { ... };
```

| Parameter | Purpose |
|-----------|---------|
| `I_USE_STREAM_REG_FOR_CREDIT_RECEIVE` | When `false` (worker path), free-slot credits are tracked via a volatile L1 pointer. When `true` (EDM-to-EDM path), credits are tracked via hardware stream registers. |
| `EDM_NUM_BUFFER_SLOTS` | When non-zero, enables compile-time-known buffer slot counts for tighter codegen. When zero, the slot count is discovered at runtime. |

The common worker alias is:

```cpp
using WorkerToFabricEdmSender = WorkerToFabricEdmSenderImpl<false, 0>;
```

### Key Fields

The adapter maintains the following state:

```
edm_buffer_addr             -- Current write-target address in EDM L1
edm_buffer_base_addr        -- Base of the EDM channel buffer ring
buffer_size_bytes           -- Size of each buffer slot
num_buffers_per_channel     -- Number of slots in the ring
buffer_slot_write_counter   -- Monotonic write counter (worker only)
buffer_slot_index           -- Current slot index (mod num_buffers)
edm_noc_x, edm_noc_y       -- NOC coordinates of the target EDM core
edm_buffer_local_free_slots_read_ptr  -- Local copy of EDM's read counter
edm_buffer_remote_free_slots_update_addr -- Stream reg addr for credit updates
worker_teardown_addr        -- L1 address for teardown ACK signaling
```

### Flow Control Protocol

The flow control is a producer-consumer ring buffer protocol:

1. **Write pointer** (`buffer_slot_write_counter.counter`): Incremented by the worker each time it sends a packet.
2. **Read pointer** (`*edm_buffer_local_free_slots_read_ptr`): Updated by the EDM when it consumes a buffer slot. The EDM writes directly into the worker's L1 (or stream register) via NOC.

The space check is:

```cpp
template <size_t num_slots = 1>
FORCE_INLINE bool edm_has_space_for_packet() const {
    invalidate_l1_cache();
    if constexpr (!I_USE_STREAM_REG_FOR_CREDIT_RECEIVE) {
        auto used_slots = this->buffer_slot_write_counter.counter
                        - *this->edm_buffer_local_free_slots_read_ptr;
        return used_slots < this->num_buffers_per_channel;
    } else {
        return get_ptr_val(worker_credits_stream_id) >= num_slots;
    }
}
```

The difference between write counter and read counter gives the number of occupied slots. When this is less than `num_buffers_per_channel`, there is room for at least one more packet. The `invalidate_l1_cache()` call is necessary because the EDM updates the read pointer asynchronously via NOC writes.

After sending a payload, the adapter advances the buffer slot index with wrap-around and decrements the downstream EDM's credit counter via a stream register write.

### Building from Runtime Args (Tensix Path)

On Tensix cores, the adapter reads connection parameters from the pre-populated L1 connection info table:

```cpp
template <ProgrammableCoreType my_core_type>
static WorkerToFabricEdmSenderImpl build_from_args(std::size_t& arg_idx) {
    if constexpr (my_core_type == ProgrammableCoreType::TENSIX) {
        tt_l1_ptr tensix_fabric_connections_l1_info_t* connection_info =
            reinterpret_cast<tt_l1_ptr tensix_fabric_connections_l1_info_t*>(
                MEM_TENSIX_FABRIC_CONNECTIONS_BASE);
        uint32_t eth_channel = get_arg_val<uint32_t>(arg_idx++);
        const auto conn = &connection_info->read_only[eth_channel];
        // Extract: edm_noc_x/y, buffer_base_addr, num_buffers,
        //          handshake_addr, buffer_size_bytes, etc.
    }
    // ...
}
```

The key insight is that the runtime arg for a Tensix worker is a **single uint32** -- the Ethernet channel index. All other connection metadata is pre-populated in the L1 connection table at `MEM_TENSIX_FABRIC_CONNECTIONS_BASE`.

## 4.2.2 The L1 Connection Table

Every Tensix core has a `tensix_fabric_connections_l1_info_t` structure at a fixed L1 address:

| Architecture | Base Address | Total Size | Split Point |
|-------------|-------------|------------|-------------|
| Wormhole | `MEM_TENSIX_FABRIC_CONNECTIONS_BASE` | 656 bytes | Offset 400 (`read_write` section) |
| Blackhole | `MEM_TENSIX_FABRIC_CONNECTIONS_BASE` | 656 bytes | Offset 400 (`read_write` section) |

The structure is split into a **read-only** section (populated by the control plane at fabric initialization) and a **read-write** section (updated during connection lifecycle). The read-only section contains per-channel records with:

| Field | Description |
|-------|-------------|
| `edm_direction` | Which direction this channel faces. |
| `edm_noc_x`, `edm_noc_y` | NOC coordinates of the connected EDM core. |
| `edm_buffer_base_addr` | Base of the EDM's channel buffer in its L1. |
| `num_buffers_per_channel` | Number of buffer slots. |
| `buffer_size_bytes` | Size of each slot. |
| `edm_connection_handshake_addr` | Address of the connection handshake semaphore. |
| `edm_worker_location_info_addr` | Where the EDM stores `EDMChannelWorkerLocationInfo`. |
| `buffer_index_semaphore_id` | Write-pointer tracking semaphore. |
| `worker_free_slots_stream_id` | Stream register ID for credit-based flow control. |

This table is the fabric's **device-side directory service**. Without it, every kernel would need 10+ runtime args per fabric connection.

## 4.2.3 Connection Lifecycle: `open()` and `close()`

The open/close handshake establishes a live connection between the worker and the EDM channel.

**Opening a connection** (`open()` or split `open_start()`/`open_finish()`):

1. **Read the EDM's write counter** -- The worker reads `edm_copy_of_wr_counter_addr` from the EDM's L1 to synchronize its local write counter. This uses `worker_teardown_addr` as a temporary read-back location.
2. **Read the EDM's read counter** -- Fetches the current free-slot count from the EDM's `EDMChannelWorkerLocationInfo`.
3. **Register the worker's semaphore address** -- Writes the address of the worker's local free-slots pointer into the EDM's `worker_semaphore_address` field.
4. **Register the worker's teardown address** -- Writes `worker_teardown_addr` so the EDM can send a teardown ACK.
5. **Write the worker's NOC coordinates** -- So the EDM can target NOC writes back to this core.
6. **Signal connection open** -- Writes `open_connection_value` to `edm_connection_handshake_l1_addr`.

After `noc_async_read_barrier()` completes, the worker initializes its local write counter from the read-back value and starts using the connection.

**Closing a connection** (`close()` or split `close_start()`/`close_finish()`):

1. Write the current write counter to the EDM's buffer index location.
2. Write `close_connection_request_value` to the handshake address.
3. Spin-wait on `worker_teardown_addr` until the EDM writes `1` (ACK).
4. Issue a write barrier and reset the teardown address.

The split API allows overlapping close operations across multiple connections:

```cpp
// Open all connections with overlapped start:
for (auto& conn : connections) conn.open_start();
// Then wait for all to finish:
for (auto& conn : connections) conn.open_finish();
```

This naturally overlaps the handshake latency across all connections.

## 4.2.4 `FabricConnectionManager`: The 1D-Era Manager

The `FabricConnectionManager` is the V1 connection abstraction, designed for linear (1D) fabric topologies with at most two connections: forward and backward.

```cpp
class FabricConnectionManager final {
    static constexpr uint8_t FORWARD_CONNECTION_FLAG_MASK  = 0x01;
    static constexpr uint8_t BACKWARD_CONNECTION_FLAG_MASK = 0x02;

    WorkerToFabricEdmSender forward_fabric_sender;
    WorkerToFabricEdmSender backward_fabric_sender;
    uint8_t connection_flags;
};
```

Key characteristics:

- **Two fixed directions**: `has_forward_connection()` and `has_backward_connection()` check single bits in `connection_flags`.
- **Build modes**: The `BuildFromArgsMode` enum controls whether connections are automatically opened during construction.
- **Symmetric open/close**: Opens forward first, then backward; closes in the same order.

| Mode | Behavior |
|------|----------|
| `BUILD_ONLY` | Construct senders but do not open connections. |
| `BUILD_AND_OPEN_CONNECTION` | Build, then fully open (blocking). |
| `BUILD_AND_OPEN_CONNECTION_START_ONLY` | Build, then `open_start()` all. Caller must call `open_finish()` later. |

The forward/backward model maps naturally to a ring or linear chain topology but becomes awkward for 2D mesh communication where a core may need to send in up to four directions.

## 4.2.5 `RoutingPlaneConnectionManager`: The 2D-Era Manager

The `RoutingPlaneConnectionManager` is the V2 replacement, supporting up to 4 simultaneous connections with per-connection destination tracking:

```cpp
class RoutingPlaneConnectionManager final {
public:
    static constexpr std::size_t MaxConnections = TT_FABRIC_MAX_ROUTING_PLANE_CONNECTIONS;  // 4
    using Sender = WorkerToFabricEdmSender;

    struct ConnectionSlot {
        Sender sender;
        uint8_t tag;
#ifdef FABRIC_2D
        uint16_t dst_dev_id;
        uint16_t dst_mesh_id;
#endif
    };
    // ...
private:
    std::array<ConnectionSlot, MaxConnections> slots_{};
    uint32_t num_active_;
};
```

Each `ConnectionSlot` contains:

| Field | Description |
|-------|-------------|
| `sender` | The underlying `WorkerToFabricEdmSender` for this connection. |
| `tag` | A user-defined tag for grouping connections (e.g., by routing plane or direction). |
| `dst_dev_id` | (2D mode) The target device ID within the mesh. |
| `dst_mesh_id` | (2D mode) The target mesh ID for multi-mesh topologies. |

In 2D fabric mode, the manager also stores `ew_dim` (east-west dimension), `my_mesh_id`, and `my_chip_id` -- used by `fabric_set_unicast_route()` to compute routing fields from the source position relative to the destination.

### Per-Connection Destination Tracking in 2D Mode

When compiled with `FABRIC_2D`, each slot carries destination identity. The `build_from_args` factory reads these from runtime args:

```cpp
#if defined(FABRIC_2D)
    mgr.ew_dim = get_arg_val<uint32_t>(arg_idx++);
    mgr.my_chip_id = get_arg_val<uint32_t>(arg_idx++);
    mgr.my_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    for (uint32_t i = 0; i < num_connections_to_build; i++) {
        auto& conn = mgr.slots_[i];
        conn.dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
        conn.dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(arg_idx++));
    }
#endif
```

This means the manager knows the global identity of each connection's destination, enabling the mesh API's connection-manager overloads (File 01) to automatically stamp the correct route into each packet header.

### Tag-Based Filtering

Each slot carries a `tag` field that enables selective iteration:

```cpp
template <typename Fn>
inline void for_each_with_tag(uint32_t tag, Fn&& fn) {
    for (uint32_t i = 0; i < num_active_; ++i) {
        if (slots_[i].tag == tag) {
            fn(slots_[i].sender, i, slots_[i].tag);
        }
    }
}
```

Tags allow a kernel to partition its connections by purpose. For example, in a 2D allreduce, connections tagged `0` might go east/west while connections tagged `1` go north/south.

### Bulk Open/Close

The manager delegates lifecycle operations to all active senders:

```cpp
template <bool SEND_CREDIT_ADDR = false>
inline void open() {
    open_start<SEND_CREDIT_ADDR>();  // all senders in parallel
    open_finish();                    // barrier for all
}
inline void close() {
    close_start();   // all senders in parallel
    close_finish();  // barrier for all
}
```

This naturally overlaps the handshake latency across all 4 possible connections.

## 4.2.6 Host-Side Connection Setup APIs

The host side is responsible for populating the runtime args that the device-side managers consume. Two key APIs in `tt_metal/api/tt-metalium/experimental/fabric/fabric.hpp` handle this:

### `append_fabric_connection_rt_args`

```cpp
template <typename ProgramOrDescriptor = Program>
void append_fabric_connection_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    ProgramOrDescriptor& worker_program_or_desc,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    CoreType core_type = CoreType::WORKER);
```

This API appends the connection-specific runtime args for a **single** fabric connection. It takes the source and destination as `FabricNodeId` values and a link index. Constraints: for 1D fabric, source and destination must be physically adjacent and on the same mesh.

### `append_routing_plane_connection_manager_rt_args`

```cpp
template <typename ProgramOrDescriptor>
void append_routing_plane_connection_manager_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const std::vector<FabricNodeId>& dst_nodes,
    const std::vector<uint32_t>& connection_link_indices,
    ProgramOrDescriptor& worker_program_or_desc,
    KernelHandle& kernel_id,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    FabricApiType api_type = FabricApiType::Linear,
    CoreType core_type = CoreType::WORKER);
```

This API handles **multiple** connections at once. If `connection_link_indices` is empty, valid links are auto-selected. The `FabricApiType` parameter (`Linear` or `Mesh`) controls which device-side API the kernel will use, setting the appropriate compile-time environment variable.

A convenience overload accepts `eth_chan_directions` instead of explicit `FabricNodeId` destinations:

```cpp
template <typename ProgramOrDescriptor>
uint32_t append_routing_plane_connection_manager_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const std::vector<eth_chan_directions>& attempted_directions,
    const std::vector<uint32_t>& connection_link_indices,
    ProgramOrDescriptor& worker_program_or_desc,
    KernelHandle& kernel_id,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    FabricApiType api_type = FabricApiType::Linear,
    CoreType core_type = CoreType::WORKER);
```

### Supporting Utilities

| Function | Purpose |
|----------|---------|
| `get_forwarding_link_indices(src, dst)` | Returns which Ethernet links on `src` can reach `dst`. |
| `get_fabric_node_id_from_physical_chip_id(chip_id)` | Converts a physical chip ID to a `FabricNodeId`. |
| `get_active_fabric_eth_routing_planes_in_direction(node, dir)` | Lists active routing planes for a given direction. |
| `get_neighbor_eth_directions(src, dst)` | Returns the Ethernet channel directions to reach `dst` from `src`. |
| `get_physical_mesh_shapes()` | Returns per-mesh-ID physical mesh shapes. |

### `SetFabricConfig`: Global Fabric Configuration

Before device creation, the host must call `SetFabricConfig`:

```cpp
void SetFabricConfig(
    FabricConfig fabric_config,
    FabricReliabilityMode reliability_mode = ...,
    std::optional<uint8_t> num_routing_planes = std::nullopt,
    FabricTensixConfig fabric_tensix_config = FabricTensixConfig::DISABLED,
    FabricUDMMode fabric_udm_mode = FabricUDMMode::DISABLED,
    FabricManagerMode fabric_manager = FabricManagerMode::DEFAULT,
    FabricRouterConfig router_config = FabricRouterConfig{});
```

This configures the number of routing planes, reliability mode, and whether Tensix-based fabric routing is enabled. The number of routing planes directly determines how many simultaneous connections a `RoutingPlaneConnectionManager` can establish.

## 4.2.7 Boilerplate Accounting

To put a concrete number on the developer burden, here is the minimal host+device code required for a kernel that sends data to two remote destinations:

**Host side (~20 lines):**
```
1. Determine src/dst FabricNodeIds
2. Determine link indices (get_forwarding_link_indices)
3. Create kernel, set compile-time args
4. Build runtime args vector
5. Call append_routing_plane_connection_manager_rt_args(...)
6. Set runtime args on kernel
```

**Device side (~25 lines):**
```
1. auto mgr = RoutingPlaneConnectionManager::build_from_args<BUILD_AND_OPEN_CONNECTION>(arg_idx, 2);
2. auto route_id = PacketHeaderPool::allocate_header_n(2);
3. // ... set up NocUnicastCommandHeader per destination
4. fabric_unicast_noc_unicast_write(mgr, route_id, src_addr, size, cmd_hdr);
5. // ... repeat for each send
6. mgr.close();
```

Total: approximately **45 lines** of connection management code for a simple two-destination send pattern. In a GSRP model, all of this would be absorbed by the runtime.

## GSRP Implications

The connection infrastructure is already remarkably close to what a GSRP runtime needs internally. The L1 connection table is effectively a device-side routing directory, and the `RoutingPlaneConnectionManager` already multiplexes across routing planes. The missing piece is **automatic connection selection based on a global address** rather than explicit developer specification. A GSRP runtime would pre-establish persistent connections (eliminating per-kernel open/close latency), derive destinations from global addresses (eliminating static `dst_dev_id`/`dst_mesh_id`), and manage multi-plane credit flow transparently. Connection management accounts for ~45 lines of per-kernel boilerplate — the largest single source — all of which a GSRP runtime would absorb.

## Key Takeaways

- `WorkerToFabricEdmSenderImpl` implements write-pointer/read-pointer ring buffer flow control; Tensix cores discover connections via the 656-byte L1 table at `MEM_TENSIX_FABRIC_CONNECTIONS_BASE`.
- `FabricConnectionManager` (V1, 1D) and `RoutingPlaneConnectionManager` (V2, 2D with up to 4 slots and tag-based grouping) represent two generations of connection management with increasing automation.
- Host-side `append_*_rt_args` APIs abstract link selection and direction computation, but device-side kernels still bear ~45 lines of connection lifecycle boilerplate per cross-chip communication pattern.

## Source Code References

| File | Key Contents |
|------|-------------|
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | `WorkerToFabricEdmSenderImpl`: flow control, open/close handshake, send methods. |
| `tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp` | `RoutingPlaneConnectionManager`: multi-slot connection management with 2D destination tracking. |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp` | `FabricConnectionManager`: V1 forward/backward connection manager. |
| `tt_metal/api/tt-metalium/experimental/fabric/fabric.hpp` | Host APIs: `append_fabric_connection_rt_args`, `append_routing_plane_connection_manager_rt_args`, `SetFabricConfig`. |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h` | `MEM_TENSIX_FABRIC_CONNECTIONS_BASE` definition (Wormhole). |
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` | `MEM_TENSIX_FABRIC_CONNECTIONS_BASE` definition (Blackhole). |

---

**Next:** [`03_socket_abstractions.md`](./03_socket_abstractions.md)
