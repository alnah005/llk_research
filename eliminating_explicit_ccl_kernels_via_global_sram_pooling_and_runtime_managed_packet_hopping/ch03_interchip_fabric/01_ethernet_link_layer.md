# 01 -- Ethernet Link Layer

## Context

Chapter 2 documented the on-chip NOC and established the fundamental constraint: **the NOC is chip-local and cannot cross chip boundaries.** This chapter picks up where that constraint left off. This first file examines the hardware-level Ethernet link that physically connects pairs of Ethernet cores across chips, the RISC-V processor (ERISC) that manages each link endpoint, the raw packet transmission primitives, and the handshake protocol that establishes a link before payload traffic can flow. The Ethernet L1 memory layout was covered in Chapter 2, File 02 -- this file focuses on the link-layer interface and signaling protocol rather than the memory map. The reader should understand the physical link characteristics, the `eth_send_packet` primitive, the channel synchronization data structures, and the EDM handshake state machine that brings a link from cold-start to ready-for-traffic.

---

## 1. Ethernet Core Specifications

Each Tenstorrent chip has a set of dedicated Ethernet cores whose sole purpose is inter-chip data movement. These cores have no Tensix compute pipeline (no FPU, no SFPU) -- they contain only a RISC-V processor (ERISC) and local L1 SRAM.

| Parameter | Wormhole (WH) | Blackhole (BH) |
|-----------|:-------------:|:--------------:|
| Ethernet cores per chip | 16 | 14 active + varying idle |
| ERISC processor | Single RISC-V | Dual RISC-V (primary + subordinate) |
| L1 SRAM per core | 256 KB (minus 32 B barrier) | 512 KB |
| Usable L1 (with routing FW) | ~151 KB | ~358.8 KB |
| Bidirectional links | Yes | Yes |
| Ethernet word size | 16 B | 16 B |

> **Key Insight:** The Ethernet core is the only hardware bridge between chips. Every byte that crosses a chip boundary must pass through an Ethernet core's L1 SRAM, be processed by its ERISC firmware, and be transmitted via the Ethernet link hardware. There is no DMA bypass.

### 1.1 Physical Connectivity Model

On Wormhole, the 16 Ethernet cores connect to neighboring chips in up to 5 logical directions:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
enum eth_chan_directions : std::uint8_t {
    EAST = 0,
    WEST = 1,
    NORTH = 2,
    SOUTH = 3,
    Z = 4,        // Inter-shelf / vertical stacking
    COUNT = 5,
};
```

- **EAST/WEST**: Horizontal links within a mesh row
- **NORTH/SOUTH**: Vertical links within a mesh column
- **Z**: Inter-mesh links (e.g., between Galaxy trays)

Each Ethernet core is physically wired to exactly one remote Ethernet core on an adjacent chip -- the direction assignment is determined by the physical board wiring and discovered by the control plane during topology initialization.

### 1.2 Link Characteristics

The Ethernet link between two ERISC cores operates as a bidirectional channel with the following properties:

- **Minimum transfer granularity:** 16 bytes (`PACKET_WORD_SIZE_BYTES = 16`). All Ethernet transfers must be 16-byte aligned and sized in multiples of 16 bytes.
- **Maximum packet payload:** Bounded by channel buffer size (configurable, typically 4-16 KB per buffer slot). The historical 1500-byte maximum referenced in the plan spec refers to a legacy constraint; current TT-Fabric configurations use larger buffers.
- **Per-link bandwidth (WH):** Approximately 12.5 GB/s per link, with 4 links typically available per neighbor direction.
- **Utilization:** For small payloads (near the 16 B minimum), the per-packet overhead of channel synchronization reduces effective utilization to approximately 91%. For larger payloads (4096+ bytes), utilization approaches line rate.

---

## 2. The `eth_send_packet` Primitive

At the absolute lowest level, all inter-chip data movement reduces to calls to `internal_::eth_send_packet` and its variants. This is the hardware-level instruction that pushes data from local Ethernet L1 to the remote Ethernet core's L1 across the physical link.

### 2.1 Packet Transmission Interface

```cpp
// tt_metal/hw/inc/internal/ethernet/dataflow_api.h

// Safe variant: may context-switch to run_routing() if TXQ is busy
internal_::eth_send_packet(
    0,                    // Queue ID (always 0 for current implementations)
    src_addr >> 4,        // Source address in 16-byte words
    dst_addr >> 4,        // Destination address in 16-byte words (on remote core)
    num_words             // Number of 16-byte words to send
);

// Unsafe variant: guaranteed not to context-switch (used for latency-critical paths)
internal_::eth_send_packet_unsafe(
    0,                    // Queue ID
    src_addr >> 4,        // Source word address
    dst_addr >> 4,        // Destination word address
    num_words             // Word count
);

// Byte-granularity unsafe variant
internal_::eth_send_packet_bytes_unsafe(
    0,                    // Queue ID
    src_addr,             // Source byte address
    dst_addr,             // Destination byte address
    num_bytes             // Byte count (must be multiple of 16)
);
```

All addresses are divided by 16 (right-shifted by 4) because the hardware operates in 16-byte word granularity. The "unsafe" variants skip the TXQ busy-wait loop -- they assume the caller has already verified that the transmit queue is not busy.

### 2.2 TXQ (Transmit Queue) Interface

The Ethernet hardware has a transmit queue that accepts one command at a time:

```cpp
// tt_metal/hw/inc/internal/ethernet/dataflow_api.h
FORCE_INLINE bool eth_txq_is_busy() {
    return internal_::eth_txq_is_busy(0);
}

FORCE_INLINE void wait_for_eth_txq_cmd_space(uint32_t wait_min = 0) {
    uint32_t count = 0;
    while (eth_txq_is_busy()) {
        if (count == wait_min) {
            run_routing();  // Yield to routing firmware while waiting
            count = 0;
        } else {
            count++;
        }
    }
}
```

> **Key Insight:** The `run_routing()` call during TXQ busy waits is a critical design pattern. ERISC cores cooperatively multitask between user-level Ethernet operations and fabric routing firmware. Every blocking wait invokes `run_routing()` to ensure the router continues processing packets even when the application-level code is stalled. This cooperative scheduling model is both a strength (no preemption overhead) and a constraint (long-running operations without `run_routing()` calls can stall the fabric).

### 2.3 Higher-Level Send Functions

The dataflow API provides several wrapper functions over `eth_send_packet`:

| Function | Behavior |
|----------|----------|
| `eth_send_bytes(src, dst, n)` | Send `n` bytes in 16-byte increments, update `bytes_sent` counter |
| `eth_send_bytes_over_channel_payload_only(src, dst, n)` | Send payload only, no completion signal |
| `eth_send_bytes_over_channel_payload_only_unsafe(src, dst, n)` | Payload only, no context switch |
| `eth_send_bytes_over_channel(src, dst, n, ch)` | Full send: payload + completion signal on channel `ch` |
| `eth_send_payload_complete_signal_over_channel(ch, n)` | Send only the completion signal (after payload was already sent) |

The split between "payload only" and "completion signal" enables a two-phase send protocol: the sender first transmits the data payload, then separately sends the completion notification. This allows the sender to pipeline multiple data transfers before signaling completion.

---

## 3. Channel Synchronization Data Structures

Each Ethernet link uses a shared data structure (`erisc_info`) with multiple channels for coordinating sender and receiver.

### 3.1 `eth_channel_sync_t`

The fundamental synchronization unit is a pair of counters per channel:

```
eth_channel_sync_t:
  bytes_sent    (uint32_t)  -- Set by sender to indicate payload size, cleared by receiver when done
  receiver_ack  (uint32_t)  -- Set by receiver to acknowledge receipt
```

The `erisc_info->channels[N]` array provides up to `MAX_NUM_CONCURRENT_TRANSACTIONS` channels (typically 8) for multiplexing traffic over a single Ethernet link.

### 3.2 Two-Level Acknowledgment Protocol

The two-level acknowledgment protocol works as follows:

1. **First level (receiver_ack):** The receiver sets `receiver_ack = 1` after it has received the data into its local buffer. At this point, the sender knows the data arrived but the receiver may still be processing it.
2. **Second level (bytes_sent = 0):** The receiver clears `bytes_sent` after it has fully consumed the data and freed its buffer. At this point, the sender can safely reuse the buffer slot.

```cpp
// tt_metal/hw/inc/internal/ethernet/dataflow_api.h

// Check first-level ack (data received by peer)
FORCE_INLINE bool eth_is_receiver_channel_send_acked(uint32_t channel) {
    return erisc_info->channels[channel].receiver_ack != 0;
}

// Check second-level done (data fully consumed by peer)
FORCE_INLINE bool eth_is_receiver_channel_send_done(uint32_t channel) {
    return erisc_info->channels[channel].bytes_sent == 0;
}
```

### 3.3 Receiver-Side Synchronization

```cpp
// tt_metal/hw/inc/internal/ethernet/dataflow_api.h

// Check if payload data is available on a channel
FORCE_INLINE bool eth_bytes_are_available_on_channel(uint8_t channel) {
    return erisc_info->channels[channel].bytes_sent != 0;
}

// Send first-level ack to sender
FORCE_INLINE void eth_receiver_channel_ack(
    uint32_t channel, uint32_t eth_transaction_ack_word_addr) {
    erisc_info->channels[channel].receiver_ack = 1;
    internal_::eth_send_packet(0,
        eth_transaction_ack_word_addr >> 4,
        ((uint32_t)(&(erisc_info->channels[channel].receiver_ack))) >> 4, 1);
}

// Send second-level completion to sender (clears bytes_sent)
FORCE_INLINE void eth_receiver_channel_done(uint32_t channel) {
    send_eth_receiver_channel_done(&(erisc_info->channels[channel]));
}
```

### 3.4 Adaptive Wait Behavior

All blocking wait functions in the Ethernet dataflow API follow the same pattern: they poll a local L1 address with `invalidate_l1_cache()` and, after a configurable number of idle iterations (`wait_min`), call `run_routing()` to yield processing time to the routing firmware. The `wait_for_bytes_on_channel_sync_addr` function adds an optimization: if the `bytes_sent` counter changes during the wait (indicating an in-progress transfer), the function resets its idle counter to avoid unnecessary context switches.

---

## 4. The EDM Handshake Protocol

Before any fabric payload traffic can flow over an Ethernet link, the two ERISC cores at each end must complete a handshake to verify mutual readiness. This is handled by the EDM handshake protocol defined in `edm_handshake.hpp`.

### 4.1 `EDMStatus` State Machine

The EDM kernel progresses through a defined sequence of states:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
enum EDMStatus : uint32_t {
    STARTED                    = 0xA0B0C0D0,  // EDM kernel has started
    REMOTE_HANDSHAKE_COMPLETE  = 0xA1B1C1D1,  // Handshake with remote complete
    LOCAL_HANDSHAKE_COMPLETE   = 0xA2B2C2D2,  // Local handshake setup complete
    READY_FOR_TRAFFIC          = 0xA3B3C3D3,  // Ready to send/receive packets
    TERMINATED                 = 0xA4B4C4D4,  // EDM shutting down
};
```

The initialization sequence adds finer-grained postcodes for debugging:

```
INITIALIZATION_STARTED        (0xB0C0D0E0)
  -> TXQ_INITIALIZED          (0xB1C1D1E1)
  -> STREAM_REG_INITIALIZED   (0xB2C2D2E2)
  -> DOWNSTREAM_EDM_SETUP_STARTED (0xB3C3D3E3)
  -> EDM_VCS_SETUP_COMPLETE   (0xB4C4D4E4)
  -> WORKER_INTERFACES_INITIALIZED (0xB6C6D6E6)
  -> ETHERNET_HANDSHAKE_COMPLETE (0xB7C7D7E7)
  -> VCS_OPENED               (0xB8C8D8E8)
  -> ROUTING_TABLE_INITIALIZED (0xB9C9D9E9)
  -> INITIALIZATION_COMPLETE  (0xBACADAEA)
```

### 4.2 `handshake_info_t` Data Structure

The handshake uses a 32-byte aligned data structure:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp
struct handshake_info_t {
    uint32_t local_value;        // Bytes 0-3: Updated by remote with MAGIC_HANDSHAKE_VALUE
    uint16_t neighbor_mesh_id;   // Bytes 4-5: Peer's mesh_id
    uint8_t neighbor_device_id;  // Byte 6: Peer's device_id
    uint8_t padding0;            // Byte 7: Alignment padding
    uint32_t padding[2];         // Bytes 8-15: 16B alignment
    uint32_t scratch[4];         // Bytes 16-31: Outgoing identity
};
```

Total size: 32 bytes (exactly 2 Ethernet words).

### 4.3 Handshake Protocol Sequence

The handshake involves a "master" (sender-side) and a "subordinate" (receiver-side) role. Both sides prepare their identity in the `scratch` buffer and then exchange it:

```
1. Both sides initialize:
   - local_value = 0
   - scratch[0] = MAGIC_HANDSHAKE_VALUE (0xAA)
   - scratch[1] = (my_mesh_id) | (my_device_id << 16)

2. Master (sender_side_handshake):
   - Repeatedly sends scratch[0..3] to subordinate's local_value[0..3]
     via eth_send_packet
   - Polls own local_value for MAGIC_HANDSHAKE_VALUE
   - While waiting, periodically calls run_routing()

3. Subordinate (receiver_side_handshake):
   - Polls local_value for MAGIC_HANDSHAKE_VALUE
   - Once seen, sends own scratch[0..3] to master's local_value[0..3]

4. Handshake complete: both sides know each other's mesh_id and device_id
```

The magic value `0xAA` was chosen because it is non-zero (distinguishing it from the initialization state) and has a simple bit pattern for debugging. The handshake timeout is deliberately very long (`A_LONG_TIMEOUT_BEFORE_CONTEXT_SWITCH = 1,000,000,000` cycles) to avoid premature timeouts on slow link initialization.

### 4.4 Identity Exchange

A subtle but important feature of the handshake is identity exchange. When the master sends its `scratch` buffer to the subordinate's `handshake_info_t`, the little-endian memory layout causes `scratch[1]` to overwrite bytes 4-7 of the subordinate's struct:

- `scratch[1]` lower 16 bits -> `neighbor_mesh_id` (bytes 4-5)
- `scratch[1]` bits 16-23 -> `neighbor_device_id` (byte 6)

After the handshake, each side knows the `(mesh_id, device_id)` of its link partner. This identity information is used by the control plane for topology discovery and by the routing firmware for packet forwarding decisions.

### 4.5 Deprecated Legacy Handshake

The codebase retains a deprecated split handshake in `erisc::datamover::handshake::deprecated` namespace. This older protocol was designed for non-persistent EDM kernels (launched and torn down per CCL operation). It splits the handshake into `start` and `finish` phases so that the master can initiate the handshake early while other EDM initialization runs in parallel. This protocol is marked for removal once legacy CCL operations are fully migrated to the persistent fabric router model (see Chapter 1, File 04 for the legacy-to-fabric transition).

---

## 5. Termination Protocol and Router State Management

The EDM supports a termination signal enum and a host-controlled router state manager:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
enum TerminationSignal : uint32_t {
    KEEP_RUNNING             = 0,
    GRACEFULLY_TERMINATE     = 1,  // Deprecated, non-functional
    IMMEDIATELY_TERMINATE    = 2
};
```

Termination is controlled by the `RouterStateManager`, which provides a host-writable command register and a device-writable state register:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
struct RouterStateManager {
    RouterState state;        // 4B, written by device, read by host
    uint8_t padding0[12];
    RouterCommand command;    // 4B, written by host, read by device
    uint8_t padding1[12];
};
```

The 16-byte padding between `state` and `command` ensures they reside on separate cache lines, preventing false sharing between host and device writes.

The `RouterCommand` enum includes:

| Command | Value | Description |
|---------|:-----:|-------------|
| `RUN` | 0 | Normal operation: process messages and credits |
| `PAUSE` | 1 | Stop processing messages (transition state) |
| `DRAIN` | 3 | Accept but drop messages (pipe to `/dev/null`) |
| `RETRAIN` | 4 | Attempt link retrain |

The `PAUSE` -> `DRAIN` -> `RETRAIN` sequence enables live link recovery without full fabric teardown, supporting the `DYNAMIC_RECONFIGURATION` reliability mode.

---

## 6. NOC Integration on Ethernet Cores

Ethernet cores are full participants in the on-chip NOC. Tensix cores access Ethernet cores through the NOC using `NOC_XY_ADDR(eth_x, eth_y, l1_offset)`. Key operations performed via NOC between Tensix and Ethernet cores:

1. **Write packet to ERISC L1:** A Tensix worker writes a fully formed packet (header + payload) into a specific buffer slot on the ERISC core's L1 via `noc_async_write`.
2. **Signal packet availability:** The worker increments a stream register or semaphore on the ERISC core to notify it that a new packet is ready.
3. **Read flow control credits:** The worker reads stream registers on the ERISC core to check for available buffer slots before writing.

All blocking waits on Ethernet cores include `run_routing()` for cooperative scheduling:

```cpp
// tt_metal/hw/inc/internal/ethernet/dataflow_api.h
FORCE_INLINE
void eth_noc_async_read_barrier() {
    while (!ncrisc_noc_reads_flushed(noc_index)) {
        run_routing();
    }
    invalidate_l1_cache();
}

FORCE_INLINE
void eth_noc_async_write_barrier() {
    while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
        run_routing();
    }
}
```

This NOC-mediated interaction is the bridge between the chip-local world (Tensix + NOC) and the inter-chip world (Ethernet). The worker-to-ERISC connection protocol and flow control are detailed in File 02.

---

## 7. GSRP Implications

The Ethernet link layer provides both building blocks and constraints for GSRP:

| Building Block | Constraint |
|---------------|------------|
| Bidirectional point-to-point links support both write (push) and read-response (pull) data flows | Cooperative `run_routing()` scheduling means any address translation logic on ERISC competes for the same processor cycles as routing |
| 16B granularity matches L1 alignment requirements and NOC word sizes | Single RISC-V on WH means routing, translation, and forwarding all share one processor |
| 8 logical channels per link enable concurrent traffic classes | Single-entry TXQ creates a bottleneck when many cores target the same remote chip |
| Identity exchange in handshake provides the foundation for global address resolution | Store-and-forward model requires buffering in limited Ethernet L1 (256 KB on WH) |
| `RouterStateManager` enables live link management for dynamic fabric reconfiguration | No hardware address interception -- all routing logic is in software |

BH's dual RISC-V per Ethernet core is a significant structural advantage: the subordinate RISC-V could handle address translation while the primary continues routing (see Chapter 6-7 for design exploration).

---

## Key Takeaways

- **Every inter-chip byte passes through ERISC L1.** The `eth_send_packet` primitive copies data from local ERISC L1 to remote ERISC L1 in 16-byte word granularity. There is no DMA bypass -- the ERISC firmware must orchestrate all transfers, making it the bottleneck for cross-chip communication throughput.
- **The two-level acknowledgment protocol enables pipelined transfers.** The first-level ack (`receiver_ack`) frees the sender's buffer while the receiver still processes; the second-level completion (`bytes_sent = 0`) confirms full consumption. This decoupling is essential for maintaining link utilization.
- **The EDM handshake exchanges identity (mesh_id, device_id) as a side effect.** The 32-byte `handshake_info_t` structure doubles as an identity exchange mechanism, letting each ERISC discover its link partner's topology coordinates. This eliminates the need for a separate discovery protocol.
- **Cooperative multitasking via `run_routing()` is pervasive.** Every blocking wait in the Ethernet dataflow API periodically yields to the routing firmware. This is how a single ERISC core handles both sending and receiving on the same link -- it cannot use interrupts, so it polls with cooperative yields.
- **The `RouterStateManager` provides host-side control over link state.** The `PAUSE`/`DRAIN`/`RETRAIN` command sequence enables live link management without tearing down the entire fabric, which is the basis for the `FabricReliabilityMode::DYNAMIC_RECONFIGURATION` mode.

## Source Code References

- `tt_metal/hw/inc/internal/ethernet/dataflow_api.h` -- Ethernet link-layer primitives: `eth_send_bytes`, `eth_wait_for_bytes`, `eth_receiver_done`, `eth_txq_is_busy`, `eth_noc_async_read_barrier`, `eth_noc_async_write_barrier`, channel synchronization functions
- `tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp` -- EDM handshake protocol: `handshake_info_t`, `sender_side_handshake`, `receiver_side_handshake`, `MAGIC_HANDSHAKE_VALUE`
- `tt_metal/fabric/fabric_edm_packet_header.hpp` -- `EDMStatus` state machine, `TerminationSignal` enum
- `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` -- `RouterStateManager`, `RouterCommand`, `eth_chan_directions`, `PACKET_WORD_SIZE_BYTES`

---

[**Previous:** Chapter 2 -- Per-Chip Memory Architecture and the NOC](../../ch02_memory_and_noc/04_buffer_types_and_circular_buffers.md) | [**Next:** `02_erisc_datamover_and_channel_architecture.md`](./02_erisc_datamover_and_channel_architecture.md)
