# Chapter 4.1 -- Mesh Fabric API and Device-Side Primitives

## Context

The previous chapters established the low-level plumbing: Chapter 2 covered the NOC architecture and L1 memory map, while Chapter 3 detailed the Ethernet link layer, EDM channels, packet headers, and the control plane that computes routing tables. This section climbs one level higher and examines the **device-side API** that Tensix worker kernels use to send data across the fabric. The mesh API, defined in `tt_metal/fabric/hw/inc/mesh/api.h`, provides a library of `FORCE_INLINE` functions that abstract packet construction, route setup, and payload submission into single-call operations.

This API sits at the bottom of the **transparency spectrum** -- it abstracts packet construction into callable functions but still requires the developer to explicitly supply destination device IDs, mesh IDs, packet headers, and sender interfaces. Understanding exactly what these primitives automate -- and what they leave to the programmer -- is essential to evaluating how far the current stack is from a fully transparent Global SRAM Pool (GSRP). Every line of boilerplate in this API represents a responsibility that a GSRP runtime would need to absorb.

---

## 4.1.1 The Core Routing Abstraction: `fabric_set_unicast_route`

Every mesh fabric API call ultimately bottlenecks through a single routing primitive:

```cpp
template <bool called_from_router = false,
          eth_chan_directions my_direction = eth_chan_directions::COUNT>
bool fabric_set_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id = MAX_NUM_MESHES);
```

This function translates a logical destination -- specified as `(dst_dev_id, dst_mesh_id)` -- into the concrete route vector embedded in the packet header. It consults the L1-resident routing tables at `ROUTING_PATH_BASE_2D` and `ROUTING_TABLE_BASE` (populated by the control plane during fabric initialization, as covered in Chapter 3). The route is encoded as a series of per-hop direction entries in `packet_header->routing_fields`.

For inter-mesh traffic (when `dst_mesh_id` differs from the local mesh), the function first resolves the inter-mesh path before computing the intra-mesh route on the destination mesh. The destination is packed into a single 32-bit field:

```cpp
packet_header->dst_start_node_id = ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
```

This `(mesh_id, device_id)` pair is the closest thing the current system has to a "global device address." It identifies a chip uniquely across the entire multi-mesh topology, but it does **not** encode a specific L1 address within that chip -- that remains the caller's responsibility via NOC command headers.

The multicast variant `fabric_set_mcast_route` additionally encodes four-direction hop counts (e, w, n, s) into the header, enabling the packet to be replicated at each intermediate router.

## 4.1.2 The Two-Phase Send Protocol

A critical implementation detail that recurs throughout the fabric stack is the **two-phase send pattern**. When `fabric_unicast_noc_unicast_write` sends a packet, it issues:

```cpp
client_interface->wait_for_empty_write_slot();
client_interface->send_payload_without_header_non_blocking_from_address(src_addr, size);
client_interface->send_payload_flush_non_blocking_from_address(
    (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
```

The first call blocks until the downstream EDM has a free buffer slot (credit-based flow control). The second call writes the payload data into the EDM's buffer slot, starting after the header region. The third call writes the header into the beginning of the same buffer slot and flushes.

This ordering is intentional: the EDM uses the header's arrival as a **packet-complete signal**. If the header arrived before the payload, the EDM might attempt to forward an incomplete packet. By sending the payload first and the header last, the system guarantees atomicity at the packet level without requiring explicit transaction markers.

For header-only operations (e.g., `fabric_unicast_noc_unicast_atomic_inc`), only the header send is needed:

```cpp
client_interface->wait_for_empty_write_slot();
client_interface->send_payload_flush_non_blocking_from_address(
    (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
```

The `src_addr` must remain valid and unmodified until the flush completes. In practice, kernels use circular buffers or double-buffering to ensure source data stability.

## 4.1.3 Unicast Write Primitives

The primary unicast write entry point is:

```cpp
template <typename FabricSenderType, bool SetRoute = true>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    uint32_t size,
    tt::tt_fabric::NocUnicastCommandHeader noc_unicast_command_header);
```

This function performs the complete four-step sequence: route setting, header population, flow-control wait, and two-phase payload submission. The `NocUnicastCommandHeader` encodes the destination NOC address on the remote chip, including the NOC x/y coordinates and L1 address.

A second overload accepts a `RoutingPlaneConnectionManager&` and a `route_id` instead of explicit sender/device/mesh parameters:

```cpp
template <bool SetRoute = true>
FORCE_INLINE void fabric_unicast_noc_unicast_write(
    RoutingPlaneConnectionManager& connection_manager,
    uint8_t route_id,
    uint32_t src_addr,
    uint32_t size,
    NocUnicastCommandHeader noc_unicast_command_header);
```

This iterates via `PacketHeaderPool::for_each_header(route_id, ...)`, extracting `dst_dev_id` and `dst_mesh_id` from each `ConnectionSlot` in the manager. The programmer does not need to know which physical EDM connection maps to which destination -- the connection manager handles the mapping.

### Developer Boilerplate for a Single Unicast Write

A kernel developer must currently produce roughly the following to send data to one remote core:

1. Build a `RoutingPlaneConnectionManager` from runtime args (~1 call, but the host must first call `append_routing_plane_connection_manager_rt_args`).
2. Allocate packet headers via `PacketHeaderPool::allocate_header_n(N)` -- returns a route ID.
3. Open the connection: `connection_manager.open()`.
4. Construct a `NocUnicastCommandHeader` encoding the destination NOC address.
5. Call `fabric_unicast_noc_unicast_write(...)`.
6. Close the connection: `connection_manager.close()`.

**Estimated boilerplate: 15-25 lines of device code, plus 10-15 lines of host code** for runtime arg setup.

## 4.1.4 The Three-Tier API Pattern: Base, Set-State, With-State

The mesh API systematically provides every operation in three variants to reduce per-packet overhead in tight loops:

| Variant | Suffix | Purpose |
|---------|--------|---------|
| **Base** | (none) | Full setup per call -- stamps route, populates header, sends payload. |
| **Set-state** | `_set_state` | One-time header pre-configuration for the stateful variant. |
| **With-state** | `_with_state` | Reuses a pre-configured header, updating only fields specified by an `UpdateMask` bitmask. |

For example, the unicast write family contains:

- `fabric_unicast_noc_unicast_write` -- base
- `fabric_unicast_noc_unicast_write_set_state<UpdateMask>` -- cold-path initialization
- `fabric_unicast_noc_unicast_write_with_state<UpdateMask>` -- stateful hot path

The `UpdateMask` is a compile-time bitmask enum (e.g., `UnicastWriteUpdateMask`) that controls which header fields are re-written on each call. When the destination does not change between iterations, only the payload size or NOC address might need updating.

### Inner Loop Optimization Example

To illustrate the performance benefit, consider a kernel that sends 1000 pages to the same destination:

```cpp
// Base API -- redundant work per iteration
for (uint32_t i = 0; i < 1000; i++) {
    fabric_unicast_noc_unicast_write(
        &sender, pkt_hdr, dst_dev, dst_mesh,
        src_addr + i * page_size, page_size, cmd_hdr);
}
```

With the stateful API, the route and command header are set once:

```cpp
// Set-state (cold path, once):
fabric_unicast_noc_unicast_write_set_state<
    UnicastWriteUpdateMask::NocUnicastCommandHeader>(
    pkt_hdr, dst_dev, dst_mesh, cmd_hdr, page_size);

// Stateful (hot path, per iteration -- only update NOC address):
for (uint32_t i = 0; i < 1000; i++) {
    NocUnicastCommandHeader updated_hdr{noc_addr + i * page_size};
    fabric_unicast_noc_unicast_write_with_state<
        UnicastWriteUpdateMask::NocUnicastCommandHeader>(
        &sender, pkt_hdr, src_addr + i * page_size, updated_hdr);
}
```

The stateful variant eliminates redundant `fabric_set_unicast_route` calls and only updates the fields specified by `UpdateMask`, reducing per-iteration overhead.

## 4.1.5 Operation Catalog

The mesh API provides the following operation families, each in base/set-state/with-state form and each with both direct-sender and connection-manager overloads:

| Operation | Payload | Semaphore | Description |
|-----------|---------|-----------|-------------|
| `fabric_unicast_noc_unicast_write` | Yes | No | Bulk data write to remote L1/DRAM. |
| `fabric_unicast_noc_unicast_atomic_inc` | No | Yes | Increment a remote semaphore. |
| `fabric_unicast_noc_scatter_write` | Yes (multi-chunk) | No | Write payload split across up to 4 destination chunks. |
| `fabric_unicast_noc_fused_scatter_write_atomic_inc` | Yes (multi-chunk) | Yes | Scatter write + semaphore increment in one packet. |
| `fabric_unicast_noc_fused_unicast_with_atomic_inc` | Yes | Yes | Bulk write + semaphore increment in one packet. |
| `fabric_unicast_noc_unicast_inline_write` | No (32-bit in header) | No | Write a single 32-bit value via header encoding. |
| `fabric_multicast_noc_unicast_write` | Yes | No | Unicast write replicated across a rectangular chip range. |
| `fabric_multicast_noc_unicast_atomic_inc` | No | Yes | Atomic increment replicated across a chip range. |
| `fabric_multicast_noc_scatter_write` | Yes (multi-chunk) | No | Scatter write with multicast routing. |

Combined with the four calling conventions (direct, route-based, set-state, with-state) described in Section 4.1.4, this yields approximately 30+ distinct function templates in the API, though most are thin wrappers around the same core logic.

### Fused Operations

The fused operations deserve special attention because they demonstrate how multi-step remote interactions can be collapsed into a single packet traversal.

`fabric_unicast_noc_fused_unicast_with_atomic_inc` sends data to one address and atomically increments a semaphore at a second address, all in a single fabric packet. The `NocUnicastAtomicIncFusedCommandHeader` encodes both the data destination address and the semaphore address plus increment value. The destination router unpacks this into two NOC transactions: a bulk write for the data and an atomic increment for the semaphore.

The fused scatter write variant `fabric_unicast_noc_fused_scatter_write_atomic_inc` is even more powerful: it splits the payload into up to 4 scatter chunks plus a semaphore increment, all in one packet. The set-state variant includes a compile-time assertion:

```cpp
static_assert(
    has_flag(UpdateMask,
             UnicastFusedScatterWriteAtomicIncUpdateMask::Flush),
    "When setting state, the Flush field must be updated to set "
    "the chunk encoding for the semaphore increment chunk");
```

These fused operations reduce cross-chip round trips from 2 (data + signal) to 1, cutting latency in half for signaled-write patterns. For a GSRP, fused write-plus-signal would be the primitive underlying "write to remote SRAM and notify the consumer."

## 4.1.6 `PacketHeaderPool`: Per-RISC Header Management

Every mesh API call requires a pre-allocated packet header in L1. The `PacketHeaderPool` class manages a fixed pool carved from `MEM_PACKET_HEADER_POOL_BASE`:

```cpp
class PacketHeaderPool {
    static constexpr uint32_t POOL_BASE = MEM_PACKET_HEADER_POOL_BASE;
    static constexpr uint32_t POOL_SIZE = MEM_PACKET_HEADER_POOL_SIZE;
    static constexpr uint32_t HEADER_SIZE = PACKET_HEADER_MAX_SIZE;
    // Split per-RISC to avoid contention between DM0 and DM1:
    static constexpr uint32_t POOL_SIZE_PER_RISC = POOL_SIZE / MaxDMProcessorsPerCoreType;

    static volatile tt_l1_ptr PACKET_HEADER_TYPE* allocate_header(uint8_t num = 1);
    static uint8_t allocate_header_n(uint8_t num_headers);  // returns route_id
    static void for_each_header(uint8_t route_id, Func&& func);
    static void reset();
};
```

Key design points:

| Method | Purpose |
|--------|---------|
| `allocate_header(n)` | Allocates `n` contiguous headers and registers them under a new route ID. |
| `allocate_header_n(n)` | Same as above but returns the route ID directly. |
| `for_each_header(route_id, func)` | Iterates over all headers in a route, invoking `func(header_ptr, index)`. |
| `reset()` | Resets the bump pointer and route counter, allowing reuse across loop iterations. |

The route ID concept is central to the mesh API: when a kernel needs to send the same payload to multiple EDM connections, it allocates a group of headers under a single route ID. The `for_each_header` iterator then pairs each header with the corresponding connection slot.

The pool does not support deallocation of individual headers -- only a full `reset()` clears everything. On pool exhaustion, an intentional infinite loop with a diagnostic `DPRINT` message halts the core rather than silently corrupting memory.

## 4.1.7 Multicast Support via `MeshMcastRange`

Multicast in the 2D mesh fabric is expressed through per-direction hop counts:

```cpp
struct MeshMcastRange {
    uint8_t e;  // east hops
    uint8_t w;  // west hops
    uint8_t n;  // north hops
    uint8_t s;  // south hops
};
```

When a packet carries a `MeshMcastRange`, each intermediate router along the specified directions replicates the packet before forwarding, implementing a spine-and-branch multicast tree (N/S spine, E/W branches). This is distinct from the NOC-level multicast described in Chapter 2 -- it operates at the **chip level**, multicasting across devices rather than across cores within a single device.

The route-level multicast variants accept a `const MeshMcastRange* ranges` array, providing per-connection-slot multicast ranges. This enables asymmetric multicast patterns where different routing planes fan out in different directions.

**Concrete example:** Multicasting a weight update from device (2,2) to a 4x3 subset (rows 0-3, columns 1-3) of a 4x4 mesh:

```cpp
MeshMcastRange ranges = {.e = 1, .w = 1, .n = 2, .s = 1};
fabric_multicast_noc_unicast_write(
    &sender, packet_header,
    dst_dev_id, dst_mesh_id, ranges,
    src_addr, data_size, noc_unicast_command_header);
```

The packet is first routed to the anchor device. At each intermediate router, if the remaining hop count in any direction is positive, the router replicates the packet and decrements the hop count before forwarding the copy.

## 4.1.8 Address Generation Integration (`addrgen_api.h`)

The `addrgen_api.h` layer (declared in `tt_metal/fabric/hw/inc/linear/addrgen_api.h` and re-exported by the mesh namespace) bridges TT-Metal's address generator types with fabric packet headers:

```cpp
template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_write(
    uint32_t packet_payload_size,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const uint32_t id,
    const AddrGenType& d,
    uint32_t offset = 0);
```

This function:
1. Calls `d.get_noc_addr(id, offset, edm_to_local_chip_noc)` to resolve the logical page `id` to a 64-bit NOC address.
2. On Wormhole, performs a coordinate system correction (virtual to NOC0) since Wormhole does not support virtual coordinates for DRAM cores.
3. Fills `pkt_hdr->to_noc_unicast_write(NocUnicastCommandHeader{noc_address}, page_size)`.

A type trait `is_addrgen<T>` detects whether a type provides `get_noc_addr(id)`:

```cpp
template <typename T, typename = void>
struct is_addrgen : std::false_type {};

template <typename T>
struct is_addrgen<T, std::void_t<decltype(std::declval<const T&>().get_noc_addr(0))>>
    : std::true_type {};
```

Supported address generator types include `InterleavedAddrGen`, `InterleavedAddrGenFast`, `ShardedAddrGen`, and `TensorAccessor`.

The addrgen API also provides fused operation helpers:

```cpp
template <typename AddrGenType>
FORCE_INLINE void to_noc_fused_unicast_write_atomic_inc(
    uint32_t page_size,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const NocUnicastAtomicIncCommandHeader& atomic_inc_spec,
    const uint32_t id,
    const AddrGenType& d,
    uint32_t offset = 0);
```

And scatter helpers:

```cpp
template <typename AddrGenType>
FORCE_INLINE void to_noc_unicast_scatter_write(
    uint32_t page_size,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    const uint32_t id0, const uint32_t id1,
    const AddrGenType& d,
    uint32_t offset0 = 0, uint32_t offset1 = 0);
```

A Wormhole-specific detail: the `get_noc_address` helper performs coordinate translation from virtual to NOC0 coordinates, because Wormhole does not support virtual coordinates for DRAM cores:

```cpp
#if defined(ARCH_WORMHOLE)
    auto noc_address_components = get_noc_address_components(noc_address);
    auto noc_addr = safe_get_noc_addr(
        noc_address_components.first.x,
        noc_address_components.first.y,
        noc_address_components.second,
        edm_to_local_chip_noc);
    noc_address = noc_addr;
#endif
```

The addrgen integration means that kernel authors working with standard buffer layouts do not need to manually compute NOC addresses. However, these generators only resolve **local-device** addresses. The cross-device routing must still be specified separately.

**GSRP implication:** A unified global address space would merge the addrgen's local-address resolution with the fabric route's device-selection into a single lookup:
$$
\text{globalAddr}(\text{pageId}) \rightarrow (\text{meshId}, \text{devId}, \text{nocAddr})
$$
Today these are two separate steps requiring two separate data structures.

## 4.1.9 Payload Size Constraints

All mesh API operations are bounded by `FABRIC_MAX_PACKET_SIZE`, which limits the maximum payload that can be sent in a single fabric operation. For larger transfers, the kernel must split the data into multiple packets and issue multiple API calls.

The packet size constraint has a direct impact on GSRP design: if a GSRP transparently splits large memory accesses into fabric-sized packets, it must track per-packet completion (via fused atomic increments) and manage the ordering guarantees that the application expects.

## 4.1.10 The `FabricSenderType` Abstraction

All mesh API functions are templated on `FabricSenderType` rather than being hardcoded to `WorkerToFabricEdmSender`. A compile-time check enforces the interface contract:

```cpp
[[maybe_unused]] CheckFabricSenderType<FabricSenderType> check;
```

This allows the same API to work with different sender implementations (e.g., MUX-based senders for relay configurations) without changing the mesh API layer. The type erasure is entirely compile-time, with no virtual dispatch overhead.

For the connection-manager overloads, the sender type is extracted from the slot:

```cpp
auto& slot = connection_manager.get(i);
fabric_unicast_noc_unicast_write<decltype(slot.sender), SetRoute>(
    &slot.sender, packet_header, slot.dst_dev_id, slot.dst_mesh_id, ...);
```

A GSRP runtime could introduce its own sender type (e.g., `GsrpFabricSender`) that integrates global address resolution without modifying the existing mesh API.

## 4.1.11 What the Programmer Still Handles Manually

Despite the mesh API's abstractions, the kernel developer is responsible for:

| Responsibility | Details |
|----------------|---------|
| **Route computation** | Determining `dst_dev_id` and `dst_mesh_id` from the logical operation semantics. |
| **NOC address construction** | Building the `NocUnicastCommandHeader` with the correct x/y/address for the remote core. |
| **Connection setup** | Initializing the `RoutingPlaneConnectionManager` via `build_from_args()` at kernel startup. |
| **Header pool management** | Calling `PacketHeaderPool::allocate_header()` and `reset()` at appropriate times. |
| **Flow control integration** | Deciding when to call `wait_for_empty_write_slot()` vs. using non-blocking sends. |
| **Multicast range calculation** | Computing the E/W/N/S hop counts for the desired target set. |
| **Packet size constraints** | Ensuring payloads fit within `FABRIC_MAX_PACKET_SIZE`. |

## GSRP Implications

The mesh fabric API is the **most explicit** layer in the cross-chip communication stack. A per-send operation requires approximately 6 distinct API calls (route setup, header allocation, destination specification, payload send, flow control wait, header commit). A GSRP runtime would collapse all of this into a single `global_write(global_addr, local_src, size)`. The three principal gaps are: (1) address translation is explicit — the programmer must supply `(dst_mesh_id, dst_dev_id, noc_x, noc_y, l1_addr)`, (2) route selection is static per packet with no adaptive routing, and (3) the API is write-only with no remote read counterpart.

## Key Takeaways

- `fabric_set_unicast_route` is the core routing bottleneck: every operation flows through it, consulting L1-resident routing tables to translate `(dst_dev_id, dst_mesh_id)` into per-hop route vectors.
- The three-tier API pattern (base / set-state / with-state) with compile-time `UpdateMask` enables zero-overhead inner-loop optimization; `PacketHeaderPool` manages per-RISC header allocation with route-based grouping.
- The addrgen integration provides page-level address translation (including Wormhole coordinate correction) but does not extend to cross-device address resolution.

## Source Code References

| File | Key Contents |
|------|-------------|
| `tt_metal/fabric/hw/inc/mesh/api.h` | All mesh API functions: unicast/multicast writes, atomic increments, scatter, fused operations, inline writes, stateful variants. |
| `tt_metal/fabric/hw/inc/mesh/addrgen_api.h` | Mesh-namespace re-export of addrgen integration. |
| `tt_metal/fabric/hw/inc/linear/addrgen_api.h` | Address generator helpers: `to_noc_unicast_write`, `to_noc_fused_unicast_write_atomic_inc`, `to_noc_unicast_scatter_write`. |
| `tt_metal/fabric/hw/inc/packet_header_pool.h` | `PacketHeaderPool` class: bump allocator, route ID management, `for_each_header`. |
| `tt_metal/fabric/hw/inc/tt_fabric_api.h` | `fabric_set_unicast_route`, `fabric_set_mcast_route`. |
| `tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp` | `RoutingPlaneConnectionManager`, `ConnectionSlot` with per-slot `dst_dev_id`/`dst_mesh_id`. |
| `tt_metal/fabric/hw/inc/api_common.h` | Common types: `CheckFabricSenderType`, `UpdateMask` enums. |

---

**Next:** [`02_worker_to_fabric_connection_management.md`](./02_worker_to_fabric_connection_management.md)
