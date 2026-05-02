# 03 -- Packet Header Formats and Routing

## Context

File 02 described the EDM channel architecture -- how packets flow between sender and receiver channels. This file examines what is inside those packets: the header formats that carry routing decisions, NOC command fields, and payload metadata. TT-Fabric uses **source routing**, where the complete route (sequence of per-hop forwarding decisions) is encoded in the packet header by the sender. Intermediate routers do not perform routing table lookups; they simply read the next hop from the packet's route buffer and forward accordingly. The reader should understand the three header families (`PacketHeader` for dynamic 1D, `LowLatencyPacketHeaderT` for low-latency 1D, `HybridMeshPacketHeaderT` for 2D mesh), the 2-bit and 4-bit per-hop encoding schemes, the `NocCommandFields` union that specifies what to do when the packet arrives, and the compressed routing structures used for 2D route computation. Understanding these formats is essential for GSRP because any transparent address translation layer must produce correctly-formed headers that the existing router firmware can forward without modification.

---

## 1. Packet Header Type Hierarchy

TT-Fabric defines a CRTP (Curiously Recurring Template Pattern) hierarchy of packet headers:

```
PacketHeaderBase<Derived>           (40 + 4 bytes base)
  |
  +-- PacketHeader                  (64 bytes, dynamic 1D routing)
  |
  +-- LowLatencyPacketHeaderT<N>    (48-64 bytes, low-latency 1D routing)
  |
  +-- HybridMeshPacketHeaderT<N>    (80-128 bytes, 2D mesh routing)
       |
       +-- UDMHybridMeshPacketHeader  (base + 16 bytes UDM control)
```

The active packet header type is selected at compile time via the `ROUTING_MODE` define. The following table enumerates all header variants with their sizes and maximum hop counts:

| Header Type | Routing Mode | Total Size | Max Hops |
|-------------|-------------|:----------:|:--------:|
| `PacketHeader` | Dynamic 1D | 64 B | 30 |
| `LowLatencyPacketHeaderT<0>` | 1D Low-Latency (16-hop) | 48 B | 16 |
| `LowLatencyPacketHeaderT<1>` | 1D Low-Latency (32-hop) | 64 B | 32 |
| `LowLatencyPacketHeaderT<2>` | 1D Low-Latency (48-hop) | 64 B | 48 |
| `LowLatencyPacketHeaderT<3>` | 1D Low-Latency (64-hop) | 64 B | 64 |
| `HybridMeshPacketHeaderT<19>` | 2D Mesh (19-hop) | 80 B | 19 |
| `HybridMeshPacketHeaderT<35>` | 2D Mesh (35-hop, default) | 96 B | 35 |
| `HybridMeshPacketHeaderT<51>` | 2D Mesh (51-hop) | 112 B | 51 |
| `HybridMeshPacketHeaderT<67>` | 2D Mesh (67-hop, max) | 128 B | 67 |
| `UDMHybridMeshPacketHeader` | 2D + UDM control | 112 B | 35 |

---

## 2. `PacketHeaderBase` -- The Common Foundation

All packet headers share a common base structure:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
template <typename Derived>
struct PacketHeaderBase {
    NocCommandFields command_fields;   // 40 bytes: union of all NOC command types
    uint16_t payload_size_bytes;       // 2 bytes: payload size (excluding header)
    NocSendType noc_send_type;         // 1 byte: what NOC operation to perform at destination
    uint8_t src_ch_id;                 // 1 byte: source channel ID (for credit tracking)
    // Total base: 44 bytes
};
```

### 2.1 `NocSendType` -- Destination Operations

The `noc_send_type` field tells the destination router what NOC operation to perform when delivering the packet:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
enum NocSendType : uint8_t {
    NOC_UNICAST_WRITE         = 0,  // Write payload to a single core's L1
    NOC_UNICAST_INLINE_WRITE  = 1,  // Write a 4-byte value inline (no payload)
    NOC_UNICAST_ATOMIC_INC    = 2,  // Atomic increment at target address
    NOC_FUSED_UNICAST_ATOMIC_INC = 3,  // Write payload + atomic increment
    NOC_UNICAST_SCATTER_WRITE = 4,  // Scatter payload chunks to multiple addresses
    NOC_MULTICAST_WRITE       = 5,  // Write payload to a rectangle of cores
    NOC_MULTICAST_ATOMIC_INC  = 6,  // Atomic increment on a rectangle of cores
    NOC_UNICAST_READ          = 7,  // Read from target (requires UDM mode)
};
```

> **Note:** Values 5 (`NOC_MULTICAST_WRITE`) and 6 (`NOC_MULTICAST_ATOMIC_INC`) have known bugs in the current codebase.

### 2.2 `ChipSendType` -- Inter-Chip Routing Mode

```cpp
enum ChipSendType : uint8_t {
    CHIP_UNICAST   = 0,   // Deliver to a single destination chip
    CHIP_MULTICAST = 1,   // Deliver to a range of chips
};
```

### 2.3 `NocCommandFields` -- The 40-Byte Command Union

The `NocCommandFields` union carries all the information needed to execute the destination NOC operation:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
union NocCommandFields {
    NocUnicastCommandHeader unicast_write;            // 8 bytes: {noc_address}
    NocUnicastCommandHeader unicast_read;              // 8 bytes: {noc_address}
    NocUnicastInlineWriteCommandHeader unicast_inline_write;  // 16 bytes: {noc_address, value}
    NocMulticastCommandHeader mcast_write;             // 8 bytes: {address, x_start, y_start, size_x, size_y}
    NocUnicastAtomicIncCommandHeader unicast_seminc;   // 16 bytes: {noc_address, val, flush}
    NocUnicastAtomicIncFusedCommandHeader unicast_seminc_fused; // 24 bytes
    NocMulticastAtomicIncCommandHeader mcast_seminc;   // 12 bytes: {address, val, coords, sizes}
    NocUnicastScatterCommandHeader unicast_scatter_write; // 40 bytes (max union member)
};
static_assert(sizeof(NocCommandFields) == 40);
```

Key struct sizes (verified by `static_assert`):

| Struct | Size | Key Fields |
|--------|:----:|--------|
| `NocUnicastCommandHeader` | 8 B | `noc_address` (64-bit) |
| `NocMulticastCommandHeader` | 8 B | `address` (32-bit), `noc_x_start`, `noc_y_start`, `mcast_rect_size_x`, `mcast_rect_size_y` |
| `NocUnicastInlineWriteCommandHeader` | 16 B | `noc_address` (64-bit), `value` (32-bit) |
| `NocUnicastAtomicIncCommandHeader` | 16 B | `noc_address` (64-bit), `val` (32-bit), `flush` (bool) |
| `NocUnicastAtomicIncFusedCommandHeader` | 24 B | `noc_address`, `semaphore_noc_address`, `val`, `flush` |
| `NocMulticastAtomicIncCommandHeader` | 12 B | `address`, `val`, `noc_x_start`, `noc_y_start`, `size_x`, `size_y` |
| `NocUnicastScatterCommandHeader` | 40 B | `noc_address[4]`, `chunk_size[3]`, `chunk_count`, `chunk_encoding` |

The scatter write header is the largest at 40 bytes, which dictates the union size. It supports up to 4 destination addresses with per-chunk encoding:

```cpp
enum NocScatterWriteChunkEncoding : uint8_t {
    CHUNK_ENCODING_NOP            = 0,
    CHUNK_ENCODING_UNICAST_WRITE  = 1,
    CHUNK_ENCODING_SEMINC_NO_FLUSH = 2,
    CHUNK_ENCODING_SEMINC_FLUSH   = 3,
};
```

---

## 3. `PacketHeader` -- Dynamic 1D Routing (64 B)

The simplest packet header uses an 8-bit `RoutingFields` value for compact 1D routing:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
struct PacketHeader : public PacketHeaderBase<PacketHeader> {
    ChipSendType chip_send_type;    // 1 byte
    RoutingFields routing_fields;   // 1 byte
    uint8_t padding0[18];           // 18 bytes (pad to 64B total)
};
// sizeof(PacketHeader) = 64 bytes
```

For **unicast**: `value = LAST_CHIP_IN_MCAST_VAL | distance_in_hops` where `LAST_CHIP_IN_MCAST_VAL = 0x10`.

For **multicast**: `value = (range_hops << 4) | start_distance_in_hops`.

Maximum hops: $(2^4 - 1) + (2^4 - 1) = 30$ hops.

---

## 4. `LowLatencyPacketHeaderT<N>` -- Low-Latency 1D Routing (48-64 B)

The low-latency header uses a 2-bit per-hop encoding that gives the router a direct instruction for each hop:

### 4.1 2-Bit Per-Hop Encoding

Each hop consumes 2 bits from the routing fields:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
struct RoutingFieldsConstants {
    struct LowLatency {
        static constexpr uint32_t FIELD_WIDTH = 2;
        static constexpr uint32_t NOOP           = 0b00;  // No action
        static constexpr uint32_t WRITE_ONLY     = 0b01;  // Write to local chip, stop forwarding
        static constexpr uint32_t FORWARD_ONLY   = 0b10;  // Forward to next hop, no local write
        static constexpr uint32_t WRITE_AND_FORWARD = 0b11; // Write locally AND forward
        static constexpr uint32_t BASE_HOPS = 16;         // Hops per 32-bit word
    };
};
```

A single 32-bit word encodes 16 hops. The router reads the two LSBs, performs the indicated action, then right-shifts the value by 2 bits for the next hop.

### 4.2 Extension Words for Longer Routes

The `LowLatencyRoutingFieldsT` template supports 0-3 extension words beyond the base word:

```cpp
template <uint32_t ExtensionWords = 1>
struct LowLatencyRoutingFieldsT {
    uint32_t value;                          // Active routing field (read first)
    uint32_t route_buffer[ExtensionWords];   // Extension storage
};
```

| ExtensionWords | Total Words | Max Hops | Header Size |
|:--------------:|:-----------:|:--------:|:-----------:|
| 0 | 1 | 16 | 48 B |
| 1 | 2 | 32 | 64 B |
| 2 | 3 | 48 | 64 B |
| 3 | 4 | 64 | 64 B |

### 4.3 Encoding Examples

**Unicast 3 hops** (ExtensionWords=0):

```
Router reads LSB-first:
  Hop 0 (bits 0-1): FORWARD_ONLY  = 0b10
  Hop 1 (bits 2-3): FORWARD_ONLY  = 0b10
  Hop 2 (bits 4-5): WRITE_ONLY    = 0b01
  Result: value = 0b01'10'10 = 0x1A
```

$$\text{route} = \underbrace{10}_{hop 0} \; \underbrace{10}_{hop 1} \; \underbrace{01}_{hop 2} = 0\text{x}1\text{A}$$

**Multicast: 3 hops away, 2-chip range** (start=3, range=2):

```
  Hop 0 (bits 0-1): FORWARD_ONLY       = 0b10
  Hop 1 (bits 2-3): FORWARD_ONLY       = 0b10
  Hop 2 (bits 4-5): WRITE_AND_FORWARD  = 0b11
  Hop 3 (bits 6-7): WRITE_ONLY         = 0b01
  Result: value = 0b01'11'10'10 = 0x7A
```

### 4.4 Sparse Multicast

The `SparseMulticastRoutingCommandHeader` uses a hop bitmask for non-contiguous multicast targets:

```cpp
template <typename HEADER_TYPE>
struct SparseMulticastRoutingCommandHeader {
    static constexpr uint32_t max_num_hops = get_max_num_hops<HEADER_TYPE>::value;
    using HopMaskType = /* uint8_t, uint16_t, uint32_t, or uint64_t based on max_num_hops */;
    HopMaskType hop_mask;
};
```

Each set bit in `hop_mask` indicates a chip that should receive the packet. At each hop where the mask bit is set, the router performs WRITE_AND_FORWARD. At unset hops, FORWARD_ONLY. At the last set bit, WRITE_ONLY.

> **Note:** Sparse multicast is currently only supported for `LowLatencyPacketHeaderT<0>` (16-hop mode).

---

## 5. `HybridMeshPacketHeaderT<N>` -- 2D Mesh Routing (80-128 B)

The 2D packet header supports routing across a mesh or torus topology using a per-hop route buffer:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
template <int RouteBufferSize = 35>
struct HybridMeshPacketHeaderT : PacketHeaderBase<HybridMeshPacketHeaderT<RouteBufferSize>> {
    LowLatencyMeshRoutingFields routing_fields;  // 4 bytes
    uint8_t route_buffer[RouteBufferSize];        // Variable: 19-67 bytes
    union {
        struct { uint16_t dst_start_chip_id; uint16_t dst_start_mesh_id; };
        uint32_t dst_start_node_id;
    };                                            // 4 bytes
    union {
        uint16_t mcast_params[4];                 // 8 bytes: hops per direction
        uint64_t mcast_params_64;
    };
    uint8_t is_mcast_active;                      // 1 byte
} __attribute__((packed, aligned(16)));
```

### 5.1 `LowLatencyMeshRoutingFields` -- Routing State

```cpp
struct LowLatencyMeshRoutingFields {
    union {
        uint32_t value;          // For fast increment / inline dword write
        struct {
            uint16_t hop_index;          // Current position in route_buffer
            uint8_t branch_east_offset;  // Offset for mcast east branch
            uint8_t branch_west_offset;  // Offset for mcast west branch
        };
    };
};
```

The `hop_index` tracks which byte in `route_buffer` the current router should read. Each router increments `hop_index` by 1 before forwarding, so the next router reads the next entry.

### 5.2 4-Bit Per-Hop Direction Encoding

Each byte in the `route_buffer` uses a 4-bit direction encoding (the upper 4 bits are reserved):

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
struct RoutingFieldsConstants {
    struct Mesh {
        static constexpr uint8_t NOOP          = 0b0000;  // Stop routing
        static constexpr uint8_t FORWARD_EAST  = 0b0001;
        static constexpr uint8_t FORWARD_WEST  = 0b0010;
        static constexpr uint8_t FORWARD_NORTH = 0b0100;
        static constexpr uint8_t FORWARD_SOUTH = 0b1000;

        // Combinations for multicast branching:
        static constexpr uint8_t WRITE_AND_FORWARD_EW  = 0b0011;
        static constexpr uint8_t WRITE_AND_FORWARD_NS  = 0b1100;
        static constexpr uint8_t WRITE_AND_FORWARD_NE  = 0b0101;
        // ... and more combinations up to WRITE_AND_FORWARD_NSEW = 0b1111
    };
};
```

The 4-bit encoding allows each hop to specify **multiple simultaneous directions** for multicast branching. When a router sees `WRITE_AND_FORWARD_NE` (0b0101), it writes locally AND forwards copies both NORTH and EAST, using the `branch_east_offset` to split the route buffer for the east copy.

### 5.3 2D Unicast Encoding

The `encode_2d_unicast` function generates route buffer entries for unicast. The encoding follows **NS-first, then EW** (dimension-ordered routing):

1. `ns_hops` entries of NS forward direction
2. `ew_hops` entries of EW forward direction
3. 1 entry of the **opposite** direction as a termination signal

Example: 2 hops South, 1 hop East:

```
route_buffer[0] = FORWARD_SOUTH (0b1000)
route_buffer[1] = FORWARD_SOUTH (0b1000)
route_buffer[2] = FORWARD_EAST  (0b0001)
route_buffer[3] = FORWARD_WEST  (0b0010)  <- opposite = stop signal
route_buffer[4..N] = NOOP
```

The use of the opposite direction as a termination signal is an elegant encoding trick: the router checks whether the next entry is the opposite of its own direction, and if so, knows this is the final hop. This avoids needing a separate "this is the last hop" field.

### 5.4 `compressed_route_2d_t` -- Compact 2D Route Representation

For storing routes in the per-core routing table, a 4-byte compressed format is used:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
struct compressed_route_2d_t {
    // Bit field layout (32 bits total):
    //   [6:0]   NS_HOPS       (7 bits, max 127)
    //   [13:7]  EW_HOPS       (7 bits, max 127)
    //   [14]    NS_DIR        (1 bit: 0=North, 1=South)
    //   [15]    EW_DIR        (1 bit: 0=West, 1=East)
    //   [22:16] TURN_POINT    (7 bits: hop index where NS->EW turn occurs)
    uint32_t data;

    // Device-side accessors:
    uint8_t get_ns_hops() const;
    uint8_t get_ew_hops() const;
    uint8_t get_ns_direction() const;
    uint8_t get_ew_direction() const;
    uint8_t get_turn_point() const;
};
static_assert(sizeof(compressed_route_2d_t) == 4);
```

This compressed format allows storing routes for 256 destination chips in $256 \times 4 = 1024$ bytes -- fitting exactly within the `ROUTING_PATH_SIZE_2D` allocation in L1.

---

## 6. UDM Packet Header Extension

The Unified Data Movement (UDM) mode extends the 2D mesh header with control fields for remote read operations:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
struct UDMWriteControlHeader {
    uint8_t src_chip_id;
    uint16_t src_mesh_id;
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint8_t risc_id;
    uint8_t transaction_id;
    uint8_t posted;
    uint8_t initial_direction;
} __attribute__((packed));  // 9 bytes

struct UDMReadControlHeader {
    uint8_t src_chip_id;
    uint16_t src_mesh_id;
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint32_t src_l1_address;
    uint32_t size_bytes;
    uint8_t risc_id;
    uint8_t transaction_id;
    uint8_t initial_direction;
} __attribute__((packed));  // 16 bytes

struct UDMHybridMeshPacketHeader : public HybridMeshPacketHeader {
    UDMControlFields udm_control;  // 16 bytes (union of write and read headers)
};
```

> **Key Insight:** The `UDMReadControlHeader` carries all the information needed for a remote read and response -- including the source chip's identity (`src_chip_id`, `src_mesh_id`, `src_noc_x`, `src_noc_y`) and a return address (`src_l1_address`). This is a **direct building block for GSRP remote reads**, which require exactly this request/response pattern. Cross-chip reads require UDM mode because the standard fabric is unidirectional (fire-and-forget writes).

---

## 7. Route Construction on Device

### 7.1 2D Unicast Route Construction

The device-side `fabric_set_unicast_route` function converts a `(dst_dev_id, dst_mesh_id)` pair into a complete route buffer:

```cpp
// tt_metal/fabric/hw/inc/tt_fabric_api.h
template <bool called_from_router, eth_chan_directions my_direction>
bool fabric_set_unicast_route(
    volatile tt_l1_ptr HybridMeshPacketHeader* packet_header,
    uint16_t dst_dev_id, uint16_t dst_mesh_id) {
    // 1. Set destination node ID
    packet_header->dst_start_node_id =
        ((uint32_t)dst_mesh_id << 16) | (uint32_t)dst_dev_id;
    // 2. Look up routing table from L1
    // 3. Handle inter-mesh routing (redirect to exit node)
    // 4. Decode compressed_route_2d_t to route_buffer
    // 5. Set branch offsets based on NS/EW decomposition
}
```

### 7.2 Direction Lookup

The next-hop direction for a given destination is retrieved from the routing table:

```cpp
// tt_metal/fabric/hw/inc/tt_fabric_api.h
inline eth_chan_directions get_next_hop_router_direction(
    uint32_t dst_mesh_id, uint32_t dst_dev_id) {
    tt_l1_ptr routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    if (dst_mesh_id == routing_table->my_mesh_id) {
        return static_cast<eth_chan_directions>(
            routing_table->intra_mesh_direction_table.get_original_direction(dst_dev_id));
    } else {
        return static_cast<eth_chan_directions>(
            routing_table->inter_mesh_direction_table.get_original_direction(dst_mesh_id));
    }
}
```

---

## 8. Compile-Time Routing Mode Selection

The active packet header type and routing mode are selected via compile-time defines injected into kernel builds:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
#ifndef ROUTING_MODE
    #define PACKET_HEADER_TYPE tt::tt_fabric::LowLatencyPacketHeader
#else
    #if (ROUTING_MODE & (ROUTING_MODE_1D | ROUTING_MODE_LINE)) != 0
        #if (ROUTING_MODE & ROUTING_MODE_LOW_LATENCY) != 0
            #define PACKET_HEADER_TYPE tt::tt_fabric::LowLatencyPacketHeader
        #else
            #define PACKET_HEADER_TYPE tt::tt_fabric::PacketHeader
        #endif
    #elif (ROUTING_MODE & (ROUTING_MODE_2D | ROUTING_MODE_MESH)) != 0
        #if (ROUTING_MODE & ROUTING_MODE_LOW_LATENCY) != 0
            #define PACKET_HEADER_TYPE tt::tt_fabric::HybridMeshPacketHeader
        #endif
    #endif
#endif
```

This means the router firmware is compiled for a specific header type -- a single binary does not handle multiple header formats simultaneously.

---

## Key Takeaways

- **TT-Fabric uses source routing: the complete route is in the packet header.** The sender pre-computes all per-hop forwarding decisions and encodes them in the routing fields. Intermediate routers read the next hop from the packet and forward without consulting a routing table. This makes forwarding fast (the router reads 2-4 bits per hop) but puts the routing computation burden on the sender.
- **1D routing uses a 2-bit per-hop encoding with 4 actions.** `NOOP`, `WRITE_ONLY`, `FORWARD_ONLY`, and `WRITE_AND_FORWARD` provide complete control over unicast, contiguous multicast, and sparse multicast patterns. The encoding fits 16 hops per 32-bit word, with extension words scaling up to 64 hops.
- **2D routing uses a 4-bit per-hop encoding supporting 4 cardinal directions.** The direction bits can be OR-combined to express multicast branching at any hop. The `branch_east_offset` and `branch_west_offset` fields enable route buffer splitting for 2D multicast.
- **The `NocCommandFields` union (40 bytes) is the largest part of every header.** It carries the complete specification of the destination NOC operation -- from simple unicast writes (8 bytes used) to complex scatter writes with per-chunk encoding (40 bytes used). The union size is fixed regardless of which command type is active.
- **The `compressed_route_2d_t` packs a 2D route into 4 bytes** using 7-bit hop counts, 1-bit directions, and a 7-bit turn point. This compressed format is what gets stored in the per-core routing table (1024 bytes for 256 destinations). Route buffers in packet headers are the expanded form decoded from these compressed entries.

## Source Code References

- `tt_metal/fabric/fabric_edm_packet_header.hpp` -- `PacketHeaderBase`, `PacketHeader`, `LowLatencyPacketHeaderT`, `HybridMeshPacketHeaderT`, `UDMHybridMeshPacketHeader`, `NocSendType`, `ChipSendType`, all `Noc*CommandHeader` structs, `RoutingFields`, `LowLatencyRoutingFieldsT`, `LowLatencyMeshRoutingFields`, `SparseMulticastRoutingCommandHeader`, `ROUTING_MODE` selection logic
- `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` -- `RoutingFieldsConstants` (LowLatency and Mesh), `compressed_route_2d_t`, routing encoding functions (`encode_1d_unicast`, `encode_1d_multicast`, `encode_1d_sparse_multicast`, `encode_2d_unicast`), `routing_table_t`, `direction_table_t`
- `tt_metal/fabric/hw/inc/tt_fabric_api.h` -- Device-side packet construction: `fabric_set_unicast_route`, `fabric_set_mcast_route`, `fabric_set_route`, `get_next_hop_router_direction`

---

[**Previous:** `02_erisc_datamover_and_channel_architecture.md`](./02_erisc_datamover_and_channel_architecture.md) | [**Next:** `04_control_plane_and_routing_tables.md`](./04_control_plane_and_routing_tables.md)
