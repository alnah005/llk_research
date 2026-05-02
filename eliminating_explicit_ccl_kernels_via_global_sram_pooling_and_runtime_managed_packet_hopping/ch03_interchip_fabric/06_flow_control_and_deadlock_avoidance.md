# 06 -- Flow Control and Deadlock Avoidance

## Context

The previous five files documented the hardware link layer (File 01), the EDM channel architecture (File 02), packet header formats (File 03), control plane routing (File 04), and topology abstractions (File 05). This final file in the chapter addresses two critical correctness and performance concerns for any store-and-forward network: **flow control** (preventing buffer overflow when senders produce faster than receivers consume) and **deadlock avoidance** (preventing circular wait conditions that halt the entire fabric). The reader should understand the credit-based flow control between senders and receivers at both the worker-to-EDM and EDM-to-EDM levels, the buffer pointer management (`BufferIndex`, `BufferPtr`, `wrap_increment`), the dateline concept for ring/torus topologies, the bubble flow control mechanism, and the `CoordinatedEriscContextSwitchState` for link retraining coordination. For GSRP (Chapters 6-7), the deadlock avoidance mechanisms documented here become especially important because transparent bidirectional traffic introduces new circular dependency risks that the current unidirectional mechanisms do not cover.

---

## 1. Credit-Based Flow Control Overview

TT-Fabric uses credit-based flow control at every hop. No packet is sent unless the receiver has advertised available buffer space. This prevents packet loss and provides backpressure propagation from congested links back to the traffic sources.

The flow control operates at three levels:

```
Level 1: Worker -> EDM Sender Channel 0
  Credits: stream register-based free slot count
  Granularity: per buffer slot

Level 2: EDM Sender -> EDM Receiver (across Ethernet)
  Credits: completion counters + ack counters
  Granularity: per buffer slot

Level 3: EDM Receiver -> Local Chip (NOC write)
  Flow: receiver-side only, no credits needed
  (NOC write is posted and guaranteed to complete)
```

---

## 2. Worker-to-EDM Flow Control

Workers discover available buffer slots via stream registers on the ERISC core. The full protocol (stream register IDs, credit advertisement, and the read-write-decrement cycle) is documented in File 02, Section 4.5.

The stream register approach is preferred over L1 semaphore polling because stream registers are accessible via a dedicated hardware path that does not contend with NOC traffic.

---

## 3. EDM-to-EDM Flow Control (Across Ethernet)

Flow control between sender and receiver channels on connected Ethernet cores uses two types of credits, with two implementation variants selected at compile time.

The EDM uses a two-level acknowledgment protocol (completion credit + ack credit) described in File 01, Section 3.2. At the implementation level, there are two variants selected at compile time:

### 3.1 Stream Register-Based Implementation (Default)

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp
struct ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender {
    FORCE_INLINE void send_completion_credit(uint8_t src_id, uint32_t n) {
        remote_update_ptr_val<receiver_txq_id>(
            sender_channel_packets_completed_stream_ids[src_id], n);
    }

    FORCE_INLINE void send_ack_credit(uint8_t src_id) {
        remote_update_ptr_val<receiver_txq_id>(
            sender_channel_packets_ack_stream_ids[src_id], 1);
    }
};
```

`remote_update_ptr_val` uses the Ethernet link to remotely increment a stream register on the peer ERISC core. This is a single 16-byte packet over the Ethernet link -- the minimum transfer unit. Stream register-based credits have lower latency for single-TXQ configurations and avoid L1 cache invalidation overhead.

### 3.3 Counter-Based Implementation (Multi-TXQ Mode)

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp
struct ReceiverChannelCounterBasedResponseCreditSender {
    FORCE_INLINE void send_completion_credit(uint8_t src_id, uint32_t num_completions) {
        completion_counters[src_id] += num_completions;
        completion_counters_base_ptr[src_id] = completion_counters[src_id];
        update_sender_side_credits();
    }

    FORCE_INLINE void send_ack_credit(uint8_t src_id) {
        ack_counters[src_id]++;
        ack_counters_base_ptr[src_id] = ack_counters[src_id];
        update_sender_side_credits();
    }

    // Maintains local copy to avoid L1 load on increment
    std::array<uint32_t, NUM_SENDER_CHANNELS> completion_counters;
    std::array<uint32_t, NUM_SENDER_CHANNELS> ack_counters;

private:
    FORCE_INLINE void update_sender_side_credits() const {
        internal_::eth_send_packet_bytes_unsafe(
            receiver_txq_id,
            local_receiver_credits_base_address,
            to_senders_credits_base_address,
            total_number_of_receiver_to_sender_credit_num_bytes);
    }
};
```

The counter-based variant maintains local copies of counters to save an L1 read on every increment, batching multiple credits into a single Ethernet transfer. It is better suited for multi-TXQ configurations.

### 3.4 Compile-Time Selection

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp
using ReceiverChannelResponseCreditSender = typename std::conditional_t<
    multi_txq_enabled,
    ReceiverChannelCounterBasedResponseCreditSender,
    ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender>;
```

### 3.5 Sender-Side Credit Reception

On the sender side, credits from the receiver are consumed via complementary structures:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp

// Stream register-based:
struct SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver {
    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        return get_ptr_val(to_sender_packets_acked_stream);
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) {
        increment_local_update_ptr_val(to_sender_packets_acked_stream, -num_acks);
    }
};

// Counter-based:
struct SenderChannelFromReceiverCounterBasedCreditsReceiver {
    template <bool RISC_CPU_DATA_CACHE_ENABLED>
    FORCE_INLINE uint32_t get_num_unprocessed_acks_from_receiver() {
        router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
        return *acks_received_counter_ptr - acks_received_and_processed;
    }

    FORCE_INLINE void increment_num_processed_acks(size_t num_acks) {
        acks_received_and_processed += num_acks;
    }
};
```

The key difference: counter-based credits require an L1 cache invalidation to get fresh values (the receiver overwrites L1 via Ethernet), while stream register-based credits use hardware registers that are inherently coherent. The compile-time selection mirrors the receiver side:

```cpp
using SenderChannelFromReceiverCredits = std::conditional_t<
    multi_txq_enabled,
    SenderChannelFromReceiverCounterBasedCreditsReceiver,
    SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver>;
```

### 3.6 Flow Control Invariants

The flow control maintains these invariants at all times:

1. **Sender never overflows receiver:** `wrptr - completion_ptr <= NUM_BUFFERS`
2. **Receiver never underflows:** Only reads from slots where `ackptr > wr_sent_ptr`
3. **Credits are monotonically increasing:** Both ack and completion counters only increment, making overflow detection trivial
4. **Hop-by-hop backpressure:** When a receiver's buffers fill, it stops sending ack credits to upstream senders, which stops them from sending more packets, propagating backpressure upstream to the source

---

## 4. Buffer Pointer Management

### 4.1 `BufferIndex` and `BufferPtr` Strong Types

The flow control system uses named types to prevent accidental confusion between buffer indices and buffer pointers:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp
using BufferIndex = NamedType<uint8_t, struct BufferIndexType>;
using BufferPtr = NamedType<uint8_t, struct BufferPtrType>;
```

`BufferPtr` can exceed `NUM_BUFFERS` (it wraps at `2 * NUM_BUFFERS`), while `BufferIndex` is always in `[0, NUM_BUFFERS)`. The double-width pointer enables distinguishing full from empty: when `rd_ptr == wr_ptr`, the buffer is empty; when `distance_behind(rd_ptr, wr_ptr) == NUM_BUFFERS`, it is full. The `normalize_ptr` function converts:

```cpp
template <uint8_t NUM_BUFFERS>
FORCE_INLINE auto normalize_ptr(BufferPtr ptr) -> BufferIndex {
    constexpr bool is_size_pow2 = (NUM_BUFFERS & (NUM_BUFFERS - 1)) == 0;
    if constexpr (is_size_pow2) {
        return BufferIndex{static_cast<uint8_t>(ptr.get() & (NUM_BUFFERS - 1))};
    } else {
        bool normalize = ptr >= NUM_BUFFERS;
        uint8_t normalized = ptr.get() - static_cast<uint8_t>(normalize * NUM_BUFFERS);
        return BufferIndex{normalized};
    }
}
```

### 4.2 `distance_behind` -- Occupancy Computation

To determine how many buffer slots are occupied between a trailing (consumer) pointer and a leading (producer) pointer:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp
template <uint8_t NUM_BUFFERS>
FORCE_INLINE uint8_t distance_behind(
    const BufferPtr& trailing, const BufferPtr& leading) {
    constexpr bool is_pow2 = is_power_of_2(NUM_BUFFERS);
    constexpr uint8_t ptr_wrap_mask = (2 * NUM_BUFFERS) - 1;
    if constexpr (is_pow2) {
        return (leading - trailing) & ptr_wrap_mask;  // Single AND
    } else {
        bool gte = leading >= trailing;
        return gte ? leading - trailing
                   : (2 * NUM_BUFFERS) - (trailing - leading);
    }
}
```

Power-of-2 buffer counts reduce the distance calculation to a single AND instruction.

### 4.3 `wrap_increment` Power-of-2 Optimizations

The `wrap_increment` family provides specialized fast paths for each buffer count (see File 02, Section 2.4 for the full implementation). The `wrap_increment_n` variant extends this to multi-slot advancement:

```cpp
template <size_t LIMIT, typename T>
FORCE_INLINE auto wrap_increment_n(T val, uint8_t increment) -> T {
    if constexpr (is_power_of_2(LIMIT)) {
        return (val + increment) & (LIMIT - 1);
    } else {
        T new_val = val + increment;
        return (new_val >= LIMIT) ? static_cast<T>(new_val - LIMIT) : new_val;
    }
}
```

### 4.4 `OutboundReceiverChannelPointers`

For tracking the state of the remote receiver from the sender's perspective:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp
template <uint8_t RECEIVER_NUM_BUFFERS>
struct OutboundReceiverChannelPointers {
    uint32_t slot_size_bytes;
    // ... pointers tracking remote receiver state ...
};
```

---

## 5. Deadlock Avoidance

### 5.1 The Deadlock Problem

In a ring or torus topology, circular dependencies can arise when multiple chips simultaneously try to send data around the ring in the same direction. If every buffer along the ring is full, no chip can make progress:

```
Chip 0 -> Chip 1 -> Chip 2 -> Chip 3 -> Chip 0
  [FULL]    [FULL]    [FULL]    [FULL]

Circular wait: each chip is waiting for the next chip
to free a buffer, but none can.
```

This is impossible in a linear (non-ring) topology because the endpoint chip always has somewhere to drain traffic. But ring and torus topologies create cycles.

### 5.2 Dateline Concept

TT-Fabric uses the **dateline** concept (borrowed from interconnect network theory) to break cycles. A dateline is a designated link in the ring where packets are not allowed to cross in both directions simultaneously -- effectively cutting the ring into a line for deadlock analysis purposes.

The dateline works by separating traffic into two virtual channels based on whether the packet has crossed the dateline:

```
    Chip 0 ---- Chip 1 ---- Chip 2 ---- Chip 3
      |                                    |
      +------------- Dateline ------------+
                 (uses VC 1)

    Normal traffic: VC 0
    Traffic crossing dateline: VC 1

    Since VC 0 and VC 1 have separate buffer resources,
    cycles in one VC cannot block the other.
```

### 5.3 `need_deadlock_avoidance_support` Query

The `FabricContext` provides a per-direction query for whether deadlock avoidance is needed:

```cpp
// tt_metal/fabric/fabric_context.hpp
bool FabricContext::need_deadlock_avoidance_support(eth_chan_directions direction) const;
```

This returns `true` for directions that participate in a ring or torus wrap-around:

| FabricConfig | Directions Requiring Avoidance |
|-------------|-------------------------------|
| `FABRIC_1D_RING` | All directions in the ring |
| `FABRIC_2D` (mesh) | None (acyclic, no cycles possible) |
| `FABRIC_2D_TORUS_X` | EAST and WEST (X-axis wrap) |
| `FABRIC_2D_TORUS_Y` | NORTH and SOUTH (Y-axis wrap) |
| `FABRIC_2D_TORUS_XY` | All four directions |

### 5.4 Bubble Flow Control

Bubble flow control is an alternative deadlock avoidance strategy where at least one buffer slot in the ring is always kept empty (the "bubble"). This prevents the ring from becoming completely full, ensuring that packets can always make progress:

$$N_{\text{usable buffers per hop}} = N_{\text{total buffers}} - 1$$

The `FabricContext` tracks whether bubble flow control is enabled:

```cpp
// tt_metal/fabric/fabric_context.hpp
bool is_bubble_flow_control_enabled() const { return bubble_flow_control_enabled_; }
```

Bubble flow control is simpler than dateline-based virtual channel separation but wastes one buffer slot per hop. For routers with few buffer slots, this waste can significantly reduce throughput.

### 5.5 Virtual Channel Separation via Routing Planes

The current implementation uses separate routing planes as virtual channels for deadlock avoidance:

| Architecture | Max Routing Planes | Typical Usage |
|:------------:|:-----------------:|---------------|
| Wormhole | 4 | 2 planes active (forward + backward) |
| Blackhole | 2 | 1 plane per direction typically |

Each routing plane uses a separate set of Ethernet channels, providing physical isolation. Traffic on different planes cannot interfere with each other, preventing cross-plane deadlocks. The per-direction routing plane count is dynamically determined:

```cpp
// tt_metal/fabric/control_plane.cpp
void ControlPlane::initialize_dynamic_routing_plane_counts(
    const IntraMeshConnectivity& intra_mesh_connectivity,
    FabricConfig fabric_config,
    FabricReliabilityMode reliability_mode) {
    // For each direction, find the minimum link count across all chips
    // That minimum becomes the number of routing planes for that direction
}
```

### 5.6 Coordinated ERISC Context Switching

A separate but related concern is coordinating context switches between the two ERISC cores on a link (relevant for Blackhole's dual RISC-V cores):

```cpp
// tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp
enum class CoordinatedEriscContextSwitchState : uint32_t {
    NORMAL_EXECUTION = 0,
    RETRAIN_INTENT = 1,      // Master signals intent to retrain
    INTENT_ACK = 2,          // Subordinate acknowledges
    RETRAIN_COMPLETE = 3,    // Master signals completion
    COMPLETE_ACK = 4,        // Subordinate acknowledges completion
};
```

This 5-state protocol ensures that both ERISCs on a link coordinate gracefully during link retraining events (triggered by `RouterCommand::RETRAIN` from File 01), preventing data corruption when one core pauses Ethernet traffic for retraining.

---

## 6. `FabricReliabilityMode` and Link Health

The `FabricReliabilityMode` enum (STRICT / RELAXED / DYNAMIC) controls how the fabric handles link failures at initialization time; see File 04, Section 2.1 for the full definition and behavior.

From a flow control perspective, `DYNAMIC_RECONFIGURATION` combined with `RouterCommand::RETRAIN` (File 01) would enable hot-swapping of failed links -- a critical capability for large-scale deployments and for any production GSRP deployment where link failures should not invalidate the global address space.

---

## 7. End-to-End Flow Control Diagram

Combining all levels, the end-to-end flow control for a packet traversing 3 chips:

```
Source Worker (Chip 0)
    |
    | [1] Check stream reg 22 for credits
    | [2] Write packet to EDM S0 buffer slot
    v
EDM S0 (Chip 0 ERISC)
    |
    | [3] Read packet header, check Ethernet credits
    | [4] eth_send_packet to remote Rx
    v
======= Ethernet Link =======
    |
EDM Rx (Chip 1 ERISC)
    | [5] Receive packet
    | [6] Send completion credit back to S0 via Ethernet
    | [7] Read routing fields -> FORWARD_ONLY
    | [8] NOC write to next-hop EDM S1
    | [9] Send ack credit back to S0 via Ethernet
    v
EDM S1 (Chip 1 ERISC, different core)
    |
    | [10] Check Ethernet credits for next link
    | [11] eth_send_packet to remote Rx
    v
======= Ethernet Link =======
    |
EDM Rx (Chip 2 ERISC)
    | [12] Receive packet
    | [13] Send completion credit back to S1
    | [14] Read routing fields -> WRITE_ONLY
    | [15] NOC write to target Tensix core L1
    | [16] Send ack credit back to S1
    v
Target Tensix Core (Chip 2)
    | Packet payload now in local L1
```

At every hop, credits prevent buffer overflow, and at every routing decision, the per-hop encoding from the packet header determines the action (FORWARD_ONLY, WRITE_ONLY, or WRITE_AND_FORWARD per File 03).

---

## 8. GSRP Implications

| Aspect | Current State | GSRP Challenge |
|--------|--------------|----------------|
| Backpressure scope | Per-hop only | End-to-end backpressure needed for remote reads across multiple hops |
| Congestion detection | Implicit via credit depletion | Explicit congestion signals needed for GSRP traffic management |
| Traffic priority | No priority mechanism | GSRP latency-sensitive accesses would compete with bulk CCL transfers |
| Ordering | Within a single connection, in-order | Cross-connection ordering may be needed for GSRP consistency |
| Bidirectional deadlocks | Not applicable (unidirectional store-and-forward) | Chip A reading from Chip B while Chip B reads from Chip A creates circular wait |
| Request-response cycles | Not applicable (write-only traffic) | UDM read responses sharing resources with requests can form cycles |

The current deadlock avoidance mechanisms (dateline, bubble flow control, virtual channel separation) are designed for **unidirectional** store-and-forward traffic. GSRP's bidirectional, request-response traffic patterns would require additional mechanisms -- either separate virtual channels for requests and responses, epoch-based synchronization barriers, or routing plane partitioning by traffic class (see Chapters 5 and 7).

---

## Key Takeaways

- **Credit-based flow control operates at every hop with two credit levels.** Completion credits signal "I received the data" (freeing the sender's transmit buffer), while ack credits signal "I finished processing" (confirming the receiver's buffer is fully free). This two-level protocol enables pipelining: the sender can reuse its transmit buffer after completion, while the receiver may still be forwarding.
- **Two flow control implementations exist: counter-based (L1) and stream register-based (hardware).** Counter-based credits batch updates efficiently for multi-TXQ configurations. Stream register-based credits have lower latency for single-TXQ configurations. The compile-time `multi_txq_enabled` flag selects between them via `std::conditional_t`.
- **Power-of-2 buffer counts enable branchless pointer management.** The `wrap_increment`, `distance_behind`, and `normalize_ptr` functions all provide single-instruction fast paths for power-of-2 buffer counts, saving 2-3 cycles per packet on the ERISC RISC-V. The `BufferIndex` and `BufferPtr` strong types prevent index/pointer confusion.
- **Dateline-based virtual channel separation prevents deadlocks in ring/torus topologies.** The `need_deadlock_avoidance_support(direction)` query tells the router whether a given direction requires VC separation. Bubble flow control is an alternative that wastes one buffer slot per hop but avoids VC complexity. Non-wrapping mesh topologies are inherently deadlock-free under dimension-ordered routing.
- **GSRP introduces deadlock risks not covered by current mechanisms.** Bidirectional request-response traffic (e.g., cross-chip reads via UDM) can create circular dependencies between request and response paths. Addressing this will require either separate request/response virtual channels, routing plane partitioning by traffic class, or epoch-based synchronization, as explored in Chapters 5 and 7.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp` | `ReceiverChannelCounterBasedResponseCreditSender`, `ReceiverChannelStreamRegisterFreeSlotsBasedCreditSender`, `SenderChannelFromReceiverCounterBasedCreditsReceiver`, `SenderChannelFromReceiverStreamRegisterFreeSlotsBasedCreditsReceiver`, `ReceiverChannelResponseCreditSender` (type alias), `SenderChannelFromReceiverCredits` (type alias) |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp` | `BufferIndex`, `BufferPtr`, `wrap_increment()`, `wrap_increment_n()`, `normalize_ptr()`, `distance_behind()`, `ChannelBufferPointer`, `OutboundReceiverChannelPointers` |
| `tt_metal/fabric/fabric_context.hpp` | `FabricContext::need_deadlock_avoidance_support()`, `is_bubble_flow_control_enabled()`, `is_wrap_around_mesh()`, hop limit constants |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | `CoordinatedEriscContextSwitchState` (5-state protocol), `is_spine_direction()` |
| `tt_metal/fabric/control_plane.cpp` | `ControlPlane::initialize_dynamic_routing_plane_counts()` |
| `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp` | `FabricReliabilityMode` (STRICT/RELAXED/DYNAMIC RECONFIGURATION) |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp` | `sender_channel_0_free_slots_stream_id` (worker-to-EDM credit stream) |

---

[**Previous:** `05_fabric_topology_and_mesh_graph.md`](./05_fabric_topology_and_mesh_graph.md) | [**Next:** Chapter 4 -- Existing Building Blocks for Cross-Chip Communication](../ch04_building_blocks/01_mesh_api_device_primitives.md)
