# 02 -- ERISC Datamover and Channel Architecture

## Context

File 01 covered the physical Ethernet link layer -- the raw `eth_send_packet` primitives, channel synchronization, and the handshake protocol. This file moves up the stack to the EDM (ERISC Data Mover) kernel firmware that runs on each active Ethernet core. The EDM is the persistent fabric router: it accepts packets from local worker cores, forwards packets from upstream EDMs, delivers packets to their local destinations, and relays packets to the next hop. The reader should understand the two-sender-plus-one-receiver channel architecture, the `StaticSizedSenderEthChannel` buffer slot management, the worker-to-router connection lifecycle, and the "speedy path" optimization. The legacy per-CCL-operation EDM kernel path was covered in Chapter 1, File 04 -- this file focuses exclusively on the persistent fabric router architecture.

---

## 1. EDM Kernel Architecture Overview

The fabric EDM kernel (`fabric_erisc_router.cpp`) is a persistent kernel that runs for the lifetime of the fabric on each active Ethernet core. It implements a store-and-forward packet router with the following structure:

```
EDM Kernel on a Single Ethernet Core:

  Sender Channel 0 (S0) -- Accepts packets from local Tensix workers
  Sender Channel 1 (S1) -- Accepts forwarded packets from upstream EDM receiver
  Receiver Channel  (Rx) -- Receives packets from remote chip's Ethernet link

  S0 and S1 merge traffic into the remote chip's Rx channel.
  Rx delivers locally and/or forwards to the next hop's S1 channel.
```

The critical distinction between the two sender channels:

| Channel | Source | Purpose |
|---------|--------|---------|
| **Sender 0 (S0)** | Local Tensix worker cores | Ingress point for new packets entering the fabric |
| **Sender 1 (S1)** | Upstream EDM receiver on the same chip | Forwarding point for multi-hop packets |

### 1.1 Data Flow Diagram

The following diagram shows how two EDM kernels on connected Ethernet cores form a bidirectional link:

```
 +-----------------------+           +-----------------------+
 |    Sender Channel 0   |           |    Receiver Channel   |
 |   +----------------+  |           |   +----------------+  |
 |   |                +--+---+-------+-->|                |  |
 |   |                |  |   |       |   |                |  |
 |   +----------------+  |   |       |   +----------------+  |
 |    Sender Channel 1   |   |       |    Sender Channel 1   |
 |   +----------------+  |   |       |   +----------------+  |
 |   |                +--+---+       |   |                |  |
 |   |                |  |       +---+---+                |  |
 |   +----------------+  |       |   |   +----------------+  |
 |    Receiver Channel   |       |   |    Sender Channel 0   |
 |   +----------------+  |       |   |   +----------------+  |
 |   |                |  |       |   |   |                |  |
 |   |                +<-+-------+---+---+                |  |
 |   +----------------+  |           |   +----------------+  |
 +-----------------------+           +-----------------------+
      Chip A, Eth Core i                  Chip B, Eth Core j
```

### 1.2 Multi-Chip Fabric Topology

For a linear fabric across multiple chips, the pattern extends: each chip's receiver channel forwards packets to the next chip's sender channel 1 via an intra-chip NOC transfer:

```
      CHIP 0                  CHIP 1                  CHIP 2
 +-------------+         +-------------+         +-------------+
 | S0  S1  Rx  |         | S0  S1  Rx  |         | S0  S1  Rx  |
 |  |   |   ^  |         |  |   |   ^  |         |  |   |   ^  |
 |  +---+---+  |         |  +---+---+  |         |  +---+---+  |
 |      |   |  |         |      |   |  |         |      |   |  |
 |      +---+--+---------+-->---+---+--+---------+-->---+   |  |
 |          |  |         |          |  |         |          |  |
 |  +---<---+--+---------+--<---+---+--+---------+--<---+   |  |
 |  |       |  |         |  |       |  |         |  |       |  |
 |  S0  S1  Rx |         |  S0  S1  Rx |         |  S0  S1  Rx |
 +-------------+         +-------------+         +-------------+
```

Each chip has two EDM kernel instances per link direction -- one for each side of the Ethernet connection. Traffic flows in both directions simultaneously.

### 1.3 2D Fabric Extension

In 2D fabric mode (`FABRIC_2D`), the architecture extends to support up to 4 downstream EDM connections per router direction, plus local worker ingress:

```cpp
// tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
constexpr uint32_t MAX_DOWNSTREAM_EDM_COUNT = 4;
```

Each direction maintains a mapping from compact index to downstream direction:

```cpp
// tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
#if defined(FABRIC_2D)
constexpr uint32_t edm_index_to_edm_direction[eth_chan_directions::COUNT][MAX_DOWNSTREAM_EDM_COUNT] = {
    {WEST, NORTH, SOUTH, Z},   // EAST router
    {EAST, NORTH, SOUTH, Z},   // WEST router
    {EAST, WEST, SOUTH, Z},    // NORTH router
    {EAST, WEST, NORTH, Z},    // SOUTH router
    {EAST, WEST, NORTH, SOUTH} // Z router
};
#endif
```

---

## 2. `StaticSizedSenderEthChannel`

The sender channels use a fixed-size circular buffer of packet slots. The `StaticSizedSenderEthChannel` template implements this buffer management.

### 2.1 Template Parameters

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS>
class StaticSizedSenderEthChannel : public SenderEthChannelInterface<...> {
    std::array<size_t, NUM_BUFFERS> buffer_addresses;
    std::size_t cached_next_buffer_slot_addr;
    BufferIndex next_packet_buffer_index;
};
```

`HEADER_TYPE` is the packet header type (e.g., `LowLatencyPacketHeader` for 1D, `HybridMeshPacketHeader` for 2D). `NUM_BUFFERS` defines the number of buffer slots per channel -- typically 4 or 8 depending on the available L1 space and the channel buffer size.

### 2.2 Buffer Slot Layout

Each channel occupies a contiguous region of ERISC L1, divided into equal-sized buffer slots:

```
Channel Base Address
+--------+--------+--------+--------+
| Slot 0 | Slot 1 | Slot 2 | Slot 3 |   (NUM_BUFFERS = 4)
+--------+--------+--------+--------+

Each Slot:
+------------------+
| Header (48-128B) |   <- Packet header (type-dependent size)
+------------------+
| Payload          |   <- Variable-size payload
| (up to max_size) |
+------------------+
```

### 2.3 Initialization

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
FORCE_INLINE void init_impl(
    size_t channel_base_address,
    size_t max_eth_payload_size_in_bytes,
    size_t header_size_bytes) {
    this->next_packet_buffer_index = BufferIndex{0};
    for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
        this->buffer_addresses[i] =
            channel_base_address + i * max_eth_payload_size_in_bytes;
        // Zero-initialize header region
        for (size_t j = 0; j < sizeof(HEADER_TYPE) / sizeof(uint32_t); j++) {
            reinterpret_cast<volatile uint32_t*>(this->buffer_addresses[i])[j] = 0;
        }
    }
    cached_next_buffer_slot_addr = this->buffer_addresses[0];
}
```

Buffer addresses are pre-computed at initialization time and stored in the `buffer_addresses` array. The `cached_next_buffer_slot_addr` avoids an array lookup on every packet send.

### 2.4 Buffer Slot Advancement

The `wrap_increment` function advances the buffer index with circular wrapping:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp
template <size_t LIMIT, typename T>
FORCE_INLINE auto wrap_increment(T val) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + 1) & (static_cast<T>(LIMIT - 1));  // Bitwise AND for wrap
    } else {
        return (val == static_cast<T>(LIMIT - 1)) ? 0 : val + 1;
    }
}
```

When `NUM_BUFFERS` is a power of 2, the wrap uses a bitwise AND instead of a comparison and branch -- saving approximately 2-3 cycles per packet on the ERISC's RISC-V core. This is why buffer counts are typically chosen as powers of 2.

---

## 3. Receiver Channel

The receiver channel accepts packets from the Ethernet link and performs one of three actions based on the packet's routing information:

| Action | When | Operation |
|--------|------|-----------|
| **Write to local chip** | Packet's destination is this chip | NOC write from ERISC L1 to target Tensix core's L1 |
| **Forward to next hop** | Packet has remaining hops | Intra-chip NOC write to next-hop EDM's sender channel 1 (S1) |
| **Write and forward** | Multicast: this chip is in range AND more hops remain | Both of the above operations |

The receiver channel uses `StaticSizedEthChannelBuffer` with similar buffer slot indexing as sender channels. The flow is reversed: the remote sender writes into the receiver's buffer slots, and the local ERISC reads from them.

### 3.1 Receiver Processing Loop

The EDM router's main loop (in `fabric_erisc_router.cpp`) continuously polls both sender and receiver channels. For the receiver channel:

1. Check if a packet has arrived (buffer slot occupied)
2. Read the packet header to determine forwarding action
3. If writing locally: issue NOC write to target core, wait for completion
4. If forwarding: copy packet to the next-hop EDM's S1 buffer slot via Ethernet
5. Send acknowledgment credits back to the remote sender

### 3.2 Sender Channel 1 (S1) -- The Forwarding Path

Sender channel 1 is structurally identical to sender channel 0 but serves a different purpose: it receives forwarded packets from the local receiver channel of a different ERISC core on the same chip. This intra-chip forwarding uses the NOC (not Ethernet) since both ERISC cores are on the same die.

The forwarding path introduces a key latency component:

$$T_{\text{hop}} = T_{\text{eth receive}} + T_{\text{noc write to S1}} + T_{\text{router decision}} + T_{\text{eth send}}$$

Each hop adds this latency to the end-to-end delivery time. The store-and-forward model means the entire packet must be received before it can be forwarded.

---

## 4. Worker-to-Router Connection Interface

For a Tensix worker core to send packets through the fabric, it must establish a connection to a local EDM sender channel 0. The connection protocol is defined by a set of constants and a handshake procedure.

### 4.1 Connection States

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp
namespace tt::tt_fabric::connection_interface {

static constexpr uint32_t unused_connection_value = 0;
static constexpr uint32_t open_connection_value = 1;
static constexpr uint32_t close_connection_request_value = 2;

// Default stream register ID for credit notification
static constexpr uint32_t sender_channel_0_free_slots_stream_id = 22;

}  // namespace
```

### 4.2 Connection Protocol

The connection protocol between a worker and an EDM sender channel is asymmetric: the worker initiates, the EDM passively detects:

```
Worker (Initiator):                     EDM Sender Channel 0 (Passive):

1. Read EDM's buffer_index address     Continuously polls connection
   (to know current write position)    semaphore for state changes

2. Write worker core X/Y (NOC 0)
   to EDM's worker_location_info

3. Write worker flow control
   semaphore L1 address

4. Write open_connection_value (1)     Detects open_connection_value
   to EDM's connection semaphore       -> begins monitoring worker
                                       -> starts sending credits

   --- Connection established ---

   Worker sends packets to EDM
   buffer slots, tracked by credits

   --- Teardown ---

5. Write close_connection_request (2)  Detects close_connection_request
   to EDM's connection semaphore       -> drains remaining packets
                                       -> returns to idle polling
```

> **Critical Constraint:** Only one worker can be connected to a given EDM sender channel at a time. If multiple workers try to connect simultaneously, the behavior is undefined (likely a hang). This is a fundamental limitation that GSRP must address -- either through multiplexing (MUX cores, covered in File 05) or through pre-reserved per-core fabric connections.

### 4.3 `fabric_connection_info_t` Structure

The `fabric_connection_info_t` struct stored in every Tensix core's L1 provides the addressing information needed to connect to an EDM channel:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
struct fabric_connection_info_t {
    uint32_t edm_buffer_base_addr;             // Base address of EDM channel buffers in ERISC L1
    uint32_t edm_connection_handshake_addr;     // Address of connection semaphore
    uint32_t edm_worker_location_info_addr;     // Where to write worker X/Y and semaphore addr
    uint32_t buffer_index_semaphore_id;         // Semaphore for buffer index tracking
    uint16_t buffer_size_bytes;                 // Size of each buffer slot
    uint8_t edm_direction;                      // Direction of this EDM (EAST/WEST/etc.)
    uint8_t edm_noc_x;                          // NOC X coordinate of the ERISC core
    uint8_t edm_noc_y;                          // NOC Y coordinate of the ERISC core
    uint8_t num_buffers_per_channel;            // Number of buffer slots
    uint16_t worker_free_slots_stream_id;       // Stream register ID for credit notification
} __attribute__((packed));
```

Total size: 24 bytes per connection entry. With 16 possible endpoints per core, the full `tensix_fabric_connections_l1_info_t` occupies:

```cpp
struct tensix_fabric_connections_l1_info_t {
    static constexpr uint8_t MAX_FABRIC_ENDPOINTS = 16;
    fabric_connection_info_t read_only[MAX_FABRIC_ENDPOINTS];    // 16 * 24 = 384 B
    uint32_t valid_connections_mask;                               // 4 B (bitmask)
    uint32_t padding_0[3];                                        // 12 B (alignment)
    fabric_aligned_connection_info_t read_write[MAX_FABRIC_ENDPOINTS]; // 16 * 16 = 256 B
};
// Total: 384 + 16 + 256 = 656 B per Tensix core
```

### 4.4 Fabric Connection Synchronization

For multi-RISC scenarios (where multiple RISC-V processors on a single Tensix core might access the same fabric connection), a synchronization region is provided:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
struct fabric_connection_sync_t {
    uint32_t lock;         // Spinlock (0 = unlocked, 1 = locked)
    uint32_t initialized;  // 0 = not initialized, 1 = initialized
    // Connection object follows at offset 8 (128 bytes reserved)
};

static constexpr uint32_t FABRIC_CONNECTION_OBJECT_OFFSET = 8;
static constexpr uint32_t FABRIC_CONNECTION_OBJECT_SIZE = 128;
// Total sync region: 8 + 128 = 136 B, padded to 144 B
```

### 4.5 Credit-Based Flow Control (Worker-to-EDM)

Once connected, the worker uses a credit-based protocol to avoid overwriting EDM buffer slots:

1. **EDM advertises free slots** by writing a count to the stream register identified by `worker_free_slots_stream_id` (default: stream 22).
2. **Worker reads the stream register** to determine how many packets it can send.
3. **Worker writes packet** to the next buffer slot address.
4. **Worker decrements its local credit count.** When credits reach zero, it must wait for the EDM to process packets and send new credits.

Stream register IDs for all sender channels:

```cpp
// tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
static constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS>
    sender_channel_free_slots_stream_ids = {
    22, 23, 24, 25, 26, 27, 28, 29, 0  // Last entry unused
};
```

---

## 5. The "Speedy Path" Optimization

The EDM router firmware includes a "speedy path" optimization for the common case where:
- The sender channel has only one active buffer slot to process
- The Ethernet TXQ is immediately available
- No flow control contention exists

In this case, the router bypasses the full scheduling logic and immediately issues the `eth_send_packet` call. This optimization saves approximately 9 cycles per send operation -- a meaningful saving given that the ERISC RISC-V runs at relatively low clock speeds.

The speedy path is particularly beneficial for small, latency-sensitive packets (e.g., semaphore increments, inline writes) where the router processing overhead would otherwise dominate the end-to-end latency.

---

## 6. Elastic Channel Buffers (Future)

The codebase includes stub implementations for elastic (dynamically-sized) channel buffers:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
template <typename HEADER_TYPE>
class ElasticSenderEthChannel : public SenderEthChannelInterface<...> {
    // TODO: Issue #26311 - Dynamic buffer allocation logic
};

template <typename HEADER_TYPE>
class ElasticEthChannelBuffer : public EthChannelBufferInterface<...> {
    // TODO: Issue #26311
};

constexpr bool USE_STATIC_SIZED_CHANNEL_BUFFERS = true;  // Currently hardcoded
```

Elastic channels would allow runtime adjustment of buffer sizes based on traffic patterns -- larger buffers for bulk transfers, smaller buffers for many concurrent small messages. This is tracked as a future enhancement but is not yet functional.

---

## 7. GSRP Implications

The EDM channel architecture provides both enablers and constraints for transparent cross-chip access:

| Building Block | Constraint |
|---------------|------------|
| Sender channel 0's worker injection protocol provides a template for transparent GSRP packet injection | Only one worker can connect to sender channel 0 at a time -- severe contention for multi-core GSRP |
| Sender channel 1's forwarding path is already content-agnostic -- it forwards based on routing headers without inspecting payloads | Workers must explicitly construct packet headers; GSRP needs a shim layer to automate this |
| Credit-based flow control via stream registers provides fine-grained, low-overhead backpressure | Explicit open/close connection handshake adds latency; GSRP needs persistent connections |
| Power-of-2 optimized `wrap_increment` and `ChannelBufferPointer` provide low-overhead buffer management | Fixed 2-sender + 1-receiver channel count limits concurrent traffic streams per link |

The MUX mode (covered in File 05) addresses the single-worker bottleneck by aggregating many workers into one sender channel. GSRP should leverage MUX rather than requiring direct worker-to-EDM connections.

---

## Key Takeaways

- **The EDM has exactly two sender channels and one receiver channel per Ethernet link.** Sender channel 0 accepts new packets from local workers; sender channel 1 accepts forwarded packets from upstream receivers. Both feed into the remote chip's single receiver channel, which decides whether to deliver locally, forward, or both.
- **Buffer slots are fixed-size and pre-allocated at initialization.** The `StaticSizedSenderEthChannel` template pre-computes all buffer addresses, and the `wrap_increment` function uses power-of-2 bitwise optimization when possible. This zero-allocation design is essential for deterministic latency on the ERISC's limited RISC-V processor.
- **The worker-to-EDM connection protocol is a software handshake.** Workers discover EDM channel addresses from pre-populated L1 tables (`tensix_fabric_connections_l1_info_t`, 656 bytes per Tensix core), write their identity and semaphore addresses, then set `open_connection_value = 1`. Only one worker can connect to a given sender channel at a time.
- **Credit-based flow control prevents buffer overflow.** The EDM advertises free buffer slots via stream registers, and workers track their local credit count. When credits are exhausted, the worker must wait -- this backpressure mechanism propagates congestion from the fabric back to the application.
- **The "speedy path" saves ~9 cycles per packet** by bypassing scheduling logic when the common case (single pending packet, TXQ available) is met. For latency-sensitive traffic like semaphore operations, this optimization is significant.

## Source Code References

- `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` -- EDM kernel main loop, sender/receiver channel processing, forwarding logic, `edm_index_to_edm_direction`
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` -- `StaticSizedSenderEthChannel`, `StaticSizedEthChannelBuffer`, `ElasticSenderEthChannel`, channel buffer layout
- `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp` -- Connection state constants: `open_connection_value`, `close_connection_request_value`, `sender_channel_0_free_slots_stream_id`
- `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp` -- `wrap_increment` with power-of-2 optimization, `BufferIndex`, `BufferPtr`, `normalize_ptr`, `distance_behind`
- `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` -- `fabric_connection_info_t`, `tensix_fabric_connections_l1_info_t`, `fabric_connection_sync_t`

---

[**Previous:** `01_ethernet_link_layer.md`](./01_ethernet_link_layer.md) | [**Next:** `03_packet_header_formats_and_routing.md`](./03_packet_header_formats_and_routing.md)
