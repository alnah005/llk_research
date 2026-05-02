# 02 -- Ethernet L1 Layout

## Context

The previous file documented the Tensix core L1 memory map -- the compute-side memory that holds user data and kernel code. This file examines the L1 SRAM on Ethernet cores, which are structurally different: they have no Tensix compute pipeline (no FPU/SFPU), they run a dedicated RISC-V processor (ERISC), and their primary role is moving data between chips via Ethernet links. The Ethernet core L1 layout is critical for the GSRP proposal because any address translation tables, routing firmware extensions, or transparent packet injection logic must fit within the Ethernet core's constrained L1 space. The reader should understand the two Ethernet core types (Active ERISC and Idle ERISC), how routing firmware shifts the unreserved base address, and what space constraints limit the feasibility of firmware-level address translation on Ethernet cores.

---

## 1. Physical Ethernet L1 Parameters

| Parameter | Wormhole (WH) | Blackhole (BH) |
|-----------|:-------------:|:--------------:|
| L1 size per Ethernet core | 256 KB (262,144 B) minus 32 B barrier = 262,112 B | 512 KB (524,288 B) |
| Number of Ethernet cores per chip | 16 | 14 (active) + varying idle |
| ERISC processor | Single RISC-V per Ethernet core | Dual RISC-V per Ethernet core (primary + subordinate) |
| Maximum loadable address | `0x3DC00` (252,928) | Up to `MEM_ERISC_MAX_SIZE` (~455 KB) |
| Syseng reserved (top of L1) | None (32 B barrier at top) | 64 KB at top of L1 |

> **Key Insight:** Blackhole's dual RISC-V per Ethernet core provides significantly more processing headroom for firmware-level address translation. Where WH has a single ERISC that must handle both routing and any translation logic sequentially, BH can dedicate the subordinate RISC-V to translation while the primary handles routing -- a structural advantage for the GSRP design.

---

## 2. Wormhole Ethernet L1 Layout

### 2.1 Active ERISC (AERISC) Memory Map

The Active ERISC runs on Ethernet cores that are actively performing fabric routing. Its L1 is divided into three major regions: the loadable region (bottom), the config space (middle), and the fabric router reserved area (top).

```
WH Active ERISC L1 (256 KB - 32 B = 262,112 B usable)
+-------------------------------------------------------------+
| 0x00000  Epoch Q / Firmware Base (0x9040)                   |
|          APP firmware: 32 KB                                 |
+-------------------------------------------------------------+
| 0x09000  L1_EPOCH_Q_BASE                                    |
+-------------------------------------------------------------+
| 0x11000  ROUTING_FW_RESERVED_BASE                           |
|          Routing firmware: 28 KB                             |
|          (present only when ROUTING_FW_ENABLED)              |
+-------------------------------------------------------------+
| 0x18000  ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE           |
|          User-available space (~151 KB with routing FW)      |
|          (~179 KB without routing FW)                        |
+-------------------------------------------------------------+
| 0x3DC00  MAX_L1_LOADING_ADDR (config space starts here)     |
|          ERISC_APP_ROUTING_INFO: 48 B                        |
|          ERISC_APP_SYNC_INFO: 288 B                          |
|          AERISC Mailbox: 5,072 B                             |
|          Kernel config: 640 B                                |
+-------------------------------------------------------------+
|          Fabric router reserved: 3,088 B                     |
|            - Telemetry: 160 B                                |
|            - Routing table: 2,576 B                          |
|              (base 516 + paths 1024 + exit nodes 1024 + 12) |
+-------------------------------------------------------------+
| 0x3FFE0  ERISC_BARRIER (32 B at very top)                   |
+-------------------------------------------------------------+
```

### 2.2 `ERISC_L1_UNRESERVED_BASE` Shift

The key address that determines how much L1 is available for user kernels on an Ethernet core is `ERISC_L1_UNRESERVED_BASE`. This value depends on whether routing firmware is enabled:

```cpp
// tt_metal/hw/inc/internal/tt-1xx/wormhole/eth_l1_address_map.h
#ifdef ROUTING_FW_ENABLED
    static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE =
        ROUTING_ENABLED_ERISC_L1_UNRESERVED_BASE;  // 0x18000 (98,304)
#else
    static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE =
        ROUTING_FW_RESERVED_BASE;                    // 0x11000 (69,632)
#endif
```

| Mode | `ERISC_L1_UNRESERVED_BASE` | Unreserved Size |
|------|:------------------------:|:----------------:|
| Without routing FW | `0x11000` (69,632) | 183,296 B (~179 KB) |
| With routing FW | `0x18000` (98,304) | 154,624 B (~151 KB) |

On the host side, `ERISC_L1_UNRESERVED_BASE` always resolves to the routing-firmware-disabled value (`0x11000`) because the host cannot know at compile time whether routing firmware is present. The actual available space is determined at runtime for device-side code.

### 2.3 Idle ERISC (IERISC) Memory Map

Idle ERISC cores are Ethernet cores not actively participating in fabric routing but available for user kernels (e.g., the legacy `erisc_async_datamover` CCL path). Their layout differs:

```
WH Idle ERISC L1
+-------------------------------------------------------------+
| 0x00000  IERISC Reserved1: 1,024 B                          |
+-------------------------------------------------------------+
| 0x01020  IERISC Reserved2: 4,064 B at offset 4,128          |
+-------------------------------------------------------------+
| 0x02000  IERISC Mailbox: 5,072 B                            |
+-------------------------------------------------------------+
| 0x033D0  IERISC Firmware: 16 KB                             |
+-------------------------------------------------------------+
|          IERISC Routing table: 2,576 B                      |
|          (positioned after firmware, 32-byte aligned)        |
+-------------------------------------------------------------+
| MEM_IERISC_MAP_END                                          |
|          IERISC kernel space: 24 KB max                      |
|          (user kernel code loaded here)                      |
+-------------------------------------------------------------+
```

IERISC has a smaller firmware footprint (16 KB vs 32 KB for AERISC) but also less total available space because its L1 shares the same 256 KB physical size. The IERISC also carries a routing table for fabric participation even when idle -- this supports configurations where idle Ethernet cores can assist with routing.

---

## 3. Blackhole Ethernet L1 Layout

### 3.1 Active ERISC (AERISC) Memory Map

Blackhole restructures the Ethernet L1 layout significantly: the config space and fabric router reservations are allocated from the top of L1 downward, and the loadable region grows upward from the bottom. With 512 KB total and dual RISC-V processors, there is substantially more room.

**Bottom-up (firmware and user space):**

```
BH Active ERISC L1 (512 KB = 524,288 B)
+-------------------------------------------------------------+
| 0x00000  ERISC Reserved1: 256 B                             |
+-------------------------------------------------------------+
| 0x00100  AERISC Mailbox: 12,768 B                           |
+-------------------------------------------------------------+
| 0x032E0  AERISC L1 Inline: 64 B (NOC inline write scratch)  |
+-------------------------------------------------------------+
| 0x03320  AERISC Void Launch Flag: 16 B                       |
+-------------------------------------------------------------+
| 0x03330  AERISC Firmware (primary): 24 KB                    |
+-------------------------------------------------------------+
| 0x09330  Subordinate AERISC Firmware: 24 KB                  |
+-------------------------------------------------------------+
| 0x0F330  MEM_AERISC_MAP_END (62,256)                         |
|          Kernel config: 25 KB                                |
+-------------------------------------------------------------+
| 0x15740  ERISC_L1_UNRESERVED_BASE (87,872)                  |
|                                                              |
|          User-available space: ~358.8 KB                     |
|                                                              |
+-------------------------------------------------------------+
```

**Top-down (system and fabric reserved):**

```
+-------------------------------------------------------------+
| 0x80000  End of L1 (512 KB)                                  |
+-------------------------------------------------------------+
| 0x70000  Syseng Reserved: 64 KB                             |
+-------------------------------------------------------------+
| 0x6FFC0  ERISC Barrier: 64 B                                 |
+-------------------------------------------------------------+
| 0x6FF90  ERISC_APP_ROUTING_INFO: 48 B                        |
+-------------------------------------------------------------+
| 0x6FE70  ERISC_APP_SYNC_INFO: 288 B                         |
+-------------------------------------------------------------+
| 0x6F260  FABRIC_ROUTER_RESERVED_BASE (455,264)              |
|          Fabric router reserved: 3,088 B                     |
|            - Telemetry: 160 B                                |
|            - Routing table: 2,576 B                          |
+-------------------------------------------------------------+
```

### 3.2 `ERISC_L1_UNRESERVED_BASE` on Blackhole

On Blackhole, the unreserved base is computed differently from Wormhole. Because BH uses base firmware and places the config space after the firmware map end, the computation is:

```cpp
// tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_l1_address_map.h
static constexpr std::uint32_t ERISC_L1_UNRESERVED_BASE =
    (MEM_AERISC_MAP_END + MEM_ERISC_KERNEL_CONFIG_SIZE + 63) & ~63;
    // = (62,256 + 25,600 + 63) & ~63 = 87,872 = 0x15740
```

There is no conditional `ROUTING_FW_ENABLED` shift on BH because the routing firmware reservation is handled differently -- it is placed at the top of L1 (above the usable region) rather than consuming low-address space. This means the unreserved base is fixed regardless of fabric configuration.

### 3.3 Available Space Comparison

| Metric | WH (with routing FW) | WH (no routing FW) | BH |
|--------|:-------------------:|:------------------:|:--:|
| Total Ethernet L1 | 262,112 B (~256 KB) | 262,112 B (~256 KB) | 524,288 B (512 KB) |
| Unreserved base | `0x18000` (98,304) | `0x11000` (69,632) | `0x15740` (87,872) |
| Upper bound | `0x3DC00` (252,928) | `0x3DC00` (252,928) | `0x6F260` (455,264) |
| **Usable space** | **154,624 B (~151 KB)** | **183,296 B (~179 KB)** | **367,392 B (~358.8 KB)** |

Blackhole provides **more than double** the usable Ethernet L1 compared to Wormhole with routing firmware enabled. This is the primary reason BH is a more natural fit for GSRP: there is enough room in the Ethernet L1 for routing tables, telemetry, and potentially additional address translation tables.

### 3.4 Idle ERISC on Blackhole

Blackhole's Idle ERISC layout mirrors the active layout but with a different memory organization. Both idle ERISC processors (primary and subordinate) have their own firmware regions:

```
BH Idle ERISC L1
+-------------------------------------------------------------+
| 0x00000  Reserved1: 256 B                                   |
| 0x00100  IERISC Mailbox: 12,768 B                           |
|          IERISC L1 Inline: 64 B                              |
|          IERISC Firmware (primary): 24 KB                    |
|          IERISC Subordinate Firmware: 24 KB                  |
+-------------------------------------------------------------+
|          IERISC Routing Table: 2,576 B (32-byte aligned)     |
+-------------------------------------------------------------+
| MEM_IERISC_MAP_END                                           |
|          Kernel space: 24 KB max                             |
+-------------------------------------------------------------+
```

The idle ERISC also carries a routing table for potential fabric participation, maintaining the same routing table structure used by active ERISC and Tensix cores.

---

## 4. Fabric Router Reserved Regions

On both architectures, the fabric router firmware reserves a block of L1 for its operational state:

### 4.1 Common Structure (3,088 B on both WH and BH)

| Component | Size (Bytes) | Description |
|-----------|:----------:|-------------|
| Telemetry | 160 | `FabricTelemetry` struct: counters, postcodes (4 B), scratch (28 B), packet/error counters |
| Routing table | 2,576 | Same structure as Tensix routing table: base(516) + paths(1024) + exit nodes(1024) + padding(12) |
| Remaining reserved | 352 | Router state, command interface, alignment padding |

### 4.2 Routing Table Layout

The AERISC routing table uses the same 3-component structure as the Tensix routing table (see File 01 Section 4.2): base header (516 B) + routing paths (1,024 B) + exit node table (1,024 B), totaling 2,576 B. The routing table is replicated identically on both Tensix and Ethernet cores, enabling cores to independently compute routes without cross-core communication.

For fabric operation, the control plane writes the routing table to every active ERISC core at `MEM_AERISC_ROUTING_TABLE_BASE`. The ERISC router firmware reads this table to determine the next hop for each packet.

### 4.3 Telemetry and Debug Addresses

| Address | Name | Architecture |
|:-------:|------|:------------:|
| `MEM_AERISC_FABRIC_TELEMETRY_BASE` | Fabric telemetry struct (160 B) | Both |
| `MEM_AERISC_FABRIC_POSTCODES_BASE` | Router postcode (4 B) | Both |
| `MEM_AERISC_FABRIC_SCRATCH_BASE` | Scratch space (28 B) | Both |
| `MEM_AERISC_FABRIC_ROUTER_STATE_BASE` | Router state | Both |
| `MEM_AERISC_FABRIC_ROUTER_COMMAND_BASE` | Router command interface | Both |

These addresses are used by the fabric watcher and telemetry systems to monitor router health and debug routing issues.

---

## 5. Constraints for Address Translation Tables

For the GSRP proposal (detailed in Chapter 6), one candidate architecture involves placing address translation tables on Ethernet cores. The key constraints are:

### 5.1 Space Budget

On Wormhole with routing firmware enabled, only ~151 KB of Ethernet L1 is available for user kernels and data. The fabric router already consumes 3,088 B for its reserved region. Adding an address translation table would further reduce this:

| Translation Table Size | Entries | WH Remaining | BH Remaining |
|:---------------------:|:------:|:-----------:|:-----------:|
| 4 KB (256 entries x 16 B) | 256 | ~147 KB | ~354.8 KB |
| 16 KB (1024 entries x 16 B) | 1,024 | ~135 KB | ~342.8 KB |
| 64 KB (4096 entries x 16 B) | 4,096 | ~87 KB | ~294.8 KB |

On WH, even a 64 KB translation table would leave only ~87 KB for the ERISC kernel, router buffers, and in-transit packet data -- likely insufficient for the EDM firmware's operational needs (the EDM uses multiple buffer slots per channel for pipelined forwarding). On BH, a 64 KB table is feasible within the ~358.8 KB available.

### 5.2 Firmware Processing Budget

The ERISC processor must handle packet forwarding at line rate. On Wormhole, the single RISC-V must:
1. Receive incoming Ethernet packets
2. Parse the route buffer in the packet header
3. Make forwarding decisions (write-local, forward, or both)
4. Manage flow control credits

Adding address translation lookup to this critical path risks increasing per-packet latency. On Blackhole, the subordinate RISC-V could handle translation while the primary handles routing, mitigating this concern.

### 5.3 Alignment Requirements

All L1 allocations on Ethernet cores must respect the following alignment:

- **16-byte alignment** for NOC writes and reads to/from Ethernet L1 (same as Tensix)
- **32-byte alignment** for routing table placement (enforced by `& ~31` alignment in the memory map)
- **Ethernet word size** of 16 bytes (`eth_word_size_bytes = 16` in `EriscDatamoverConfig`) -- the minimum transfer granularity for Ethernet transactions

Any translation table structure would need to be 16-byte aligned and sized in multiples of 16 bytes for efficient NOC access.

### 5.4 The `ERISC_L1_UNRESERVED_BASE` Shift Complication

The conditional `ERISC_L1_UNRESERVED_BASE` (which changes depending on `ROUTING_FW_ENABLED` on WH) is a complication for GSRP: the available address range is not fixed at compile time for all configurations. Any GSRP metadata placed on Ethernet cores would need to either:

1. Use the conservative (routing-enabled) base address, wasting space when routing is disabled
2. Have a runtime-configurable base address, adding indirection

On BH, this complication does not exist because the unreserved base is fixed regardless of routing firmware state.

---

## 6. The Dual-Firmware Model on Blackhole

Blackhole Ethernet cores run two independent RISC-V processors:

| Processor | Role | Firmware Size | Local Memory |
|-----------|------|:------------:|:------------:|
| Primary ERISC | Main routing firmware, packet forwarding | 24 KB | 8 KB (minus base FW local) |
| Subordinate ERISC | Additional processing capacity | 24 KB | 8 KB |

Both processors share the same L1 address space but have independent instruction execution. The subordinate has its own reset PC, firmware base, and local memory region. This dual-processor model is unique to BH and provides the processing headroom needed for firmware-level address translation:

- **Primary ERISC:** Continues to handle packet forwarding at line rate
- **Subordinate ERISC:** Could run address translation logic, connection management, or telemetry without impacting forwarding latency

On WH, the single ERISC must time-multiplex all of these functions, making it fundamentally more constrained for GSRP firmware extensions.

---

## 7. Kernel Config Space

Both WH and BH Ethernet cores have a dedicated kernel config region that holds compile-time arguments, runtime arguments, semaphore addresses, and circular buffer configurations for user-loaded ERISC kernels:

| Chip | Kernel Config Size | Location |
|------|--------------------|----------|
| WH | `(96 * 4) + (16 * 16)` = 640 B | Above mailbox, below fabric router reserved |
| BH | 25 KB | Above `MEM_AERISC_MAP_END` (included in `ERISC_L1_UNRESERVED_BASE` calculation) |

The much larger kernel config space on BH reflects the expectation that BH ERISC kernels will be more complex, with more runtime arguments and larger circular buffer configurations.

---

## Key Takeaways

- **Ethernet L1 is the bottleneck for GSRP firmware extensions.** At 256 KB on WH (with ~151 KB usable after routing FW), there is limited space for address translation tables. BH's 512 KB (with ~358.8 KB usable) provides substantially more room, making it the more viable target for firmware-assisted translation.
- **`ERISC_L1_UNRESERVED_BASE` shifts by 28 KB when routing firmware is enabled on WH.** This conditional shift means that systems running the TT-Fabric router have less Ethernet L1 available for user kernels than those without it. On BH, the fabric router reservation is at the top of L1 rather than consuming low-address space, so the unreserved base does not shift.
- **BH's dual RISC-V per Ethernet core is a structural advantage for GSRP.** The subordinate processor can run translation logic without impacting the primary router's packet forwarding throughput. This hardware capability does not exist on WH and cannot be emulated in software.
- **The routing table is replicated on both Tensix and Ethernet cores.** Every Tensix core has a 2,576 B routing table at `MEM_TENSIX_ROUTING_TABLE_BASE`, and every ERISC core has an identical 2,576 B routing table. This redundancy enables cores to independently compute routes without cross-core communication, but it consumes memory on every core in the system.
- **The top-down memory allocation on BH Ethernet cores** (system reserved growing downward from 512 KB, firmware growing upward from 0) creates a contiguous user region in the middle. This is architecturally clean for hosting additional metadata, unlike WH where the config space creates a fragmented layout.

## Source Code References

- `tt_metal/hw/inc/internal/tt-1xx/wormhole/eth_l1_address_map.h` -- Wormhole Ethernet L1 address map (`eth_l1_mem::address_map`)
- `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h` -- WH AERISC/IERISC memory map constants
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_l1_address_map.h` -- Blackhole Ethernet L1 address map
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` -- BH AERISC/IERISC/subordinate memory layout

---

**Next:** [`03_noc_architecture_and_addressing.md`](./03_noc_architecture_and_addressing.md)
