# 7.1 Transparent Packet Injection

## Context

Chapters 3 and 4 documented how Tensix worker cores currently inject packets into the
fabric: the worker explicitly opens a connection to an EDM sender channel (Chapter 3,
Section 2), constructs a packet header with per-hop routing fields (Chapter 3, Section 3),
and issues a two-phase send through the mesh API (Chapter 4, Section 1). Chapter 6 proposed
the Global SRAM Pool (GSRP) abstraction, selected the Fat Fabric API as the Phase 1 target
(Section 6.4), and designed an address translation layer that decomposes a 64-bit
`GlobalAddress` into a destination tuple and route at ~13 cycles per cached lookup
(Section 6.5). This section designs the **packet injection mechanism** -- the runtime
component that converts a Tensix core's write to a remote global address into a
correctly-formed fabric packet without explicit kernel involvement. Three candidate
approaches are evaluated (compiler-inserted shim, firmware trap, and hardware NOC extension),
the write/read path asymmetry is analyzed in detail, and the persistent connection pool and
MUX integration requirements are specified.

---

## 7.1.1 The Injection Problem

Today, a cross-chip write from a Tensix core requires approximately 25--35 lines of
device-side code (Chapter 1, Section 2; Chapter 4, Section 2). These lines perform six
distinct functions:

```
gsrp_write(dest, src, size)
  |
  1. Address decomposition       dest -> {mesh_id, chip_id, noc_x, noc_y, l1_offset}
  2. Locality check              Is (mesh_id, chip_id) == local?
  3. Route resolution             Lookup compressed_route_2d_t from L1 table
  4. Connection selection          Pick available sender from connection pool
  5. Header construction           Fill NocUnicastCommandHeader + routing fields
  6. Two-phase payload send        Payload first, header last (atomicity)
```

Steps [3] through [6] constitute **packet injection** -- they transform a logical write
intent into a physical fabric packet. Today, the kernel developer performs all of them
manually (Chapter 4, Section 2 estimated ~45 lines of boilerplate). The GSRP runtime
absorbs all of them behind a single function call.

---

## 7.1.2 Approach A: Compiler-Inserted Shim Code

### Mechanism

The TTNN compiler or a pre-compilation pass inserts shim code around every `gsrp_write` /
`gsrp_read` call. The shim is a `FORCE_INLINE` library function that performs the full
injection sequence using the existing mesh API.

```
Kernel Source:                        After Compiler Shim Insertion:

gsrp_write(dest, src, sz);           {
                                        auto dec = gsrp_decompose(dest);
                                        if (dec.is_local()) {
                                          noc_async_write(dec.noc_addr, src, sz);
                                        } else {
                                          auto& conn = gsrp_pool_select(dec);
                                          auto* hdr = gsrp_fill_header(dec);
                                          conn.wait_for_empty_write_slot();
                                          conn.send_payload(src, sz);
                                          conn.send_header(hdr);
                                        }
                                      }
```

### Injection Pipeline

```
+-------------------+     +-------------------+     +-------------------+
| Tensix Core       |     | Mesh API (L1)     |     | EDM Sender Ch 0   |
|                   |     |                   |     |                   |
| gsrp_write() ---->+---->| route_lookup()    +---->| buffer_slot[n]    |
| (shim-expanded)   |     | header_fill()     |     | eth_send_packet() |
|                   |     | two_phase_send()  |     |                   |
+-------------------+     +-------------------+     +-------------------+
       Tensix RISC              L1 Tables              ERISC Router
```

### Cycle-Level Cost Analysis

Using the address translation layer from Chapter 6, Section 5:

| Step | Operation | Cycles |
|------|-----------|:------:|
| Address decomposition | 5 shifts + 5 masks | ~3 |
| Locality check | 2 comparisons | ~1 |
| Route cache lookup | Hash-indexed L1 load (8-entry cache, Section 6.5.5) | ~2 |
| Connection pool select | Counter increment + table read | ~3 |
| Header fill | 6--8 stores to packet header fields | ~4 |
| **Total CPU overhead** | | **~13 cycles** |

At ~1 GHz Blackhole core clock, 13 cycles is 13 ns. For a 4 KB transfer (~328 ns on the
fabric at 12.5 GB/s per link), the overhead is:

$$\frac{13}{328} \approx 4.0\%$$

This meets the $< 5\%$ throughput target for transfers $\geq$ 4 KB (Section 6.5.5).

### Implementation

```cpp
// tt_metal/fabric/hw/inc/gsrp/gsrp_shim.hpp (proposed)
FORCE_INLINE void gsrp_write(GlobalAddress dest, uint32_t local_src, uint32_t size) {
    auto dec = gsrp_decompose(dest);  // ~3 cycles
    if (dec.chip_id == MY_CHIP_ID && dec.mesh_id == MY_MESH_ID) {
        // Local fast path: direct NOC write (~4 ns)
        uint64_t noc_addr = get_noc_addr(dec.noc_x, dec.noc_y, dec.l1_offset);
        noc_async_write(local_src, noc_addr, size);
    } else {
        // Remote path: fabric injection (~13 cycles + fabric latency)
        gsrp_chip_route route = gsrp_route_cache_lookup(dec.chip_id);
        auto& conn = gsrp_connection_pool_select(route);
        auto* hdr = gsrp_fill_header(dec, route);
        conn.wait_for_empty_write_slot();
        conn.send_payload_without_header_non_blocking_from_address(local_src, size);
        conn.send_payload_flush_non_blocking_from_address(
            (uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
    }
}
```

The locality check adds only ~4 cycles to every access but saves the full fabric injection
overhead for local writes -- a common case in operations that mix local and remote accesses
(e.g., a Tensix core writing to both local and remote portions of a sharded tensor).

For static placement (where the destination chip is known at compile time), the compiler
can eliminate the runtime locality check entirely and emit either a direct `noc_async_write`
or a direct `gsrp_write` without the branch.

> **Key Insight:** The compiler-inserted shim approach requires zero firmware changes and
> zero hardware modifications. It composes entirely with the existing mesh API (Chapter 4,
> Section 1) and EDM channel architecture (Chapter 3, Section 2). The ~13-cycle per-access
> CPU overhead meets the $< 5\%$ throughput target for transfers $\geq$ 4 KB.

### Evaluation

| Criterion | Assessment |
|-----------|-----------|
| Kernel transparency | Partial: programmer calls `gsrp_write()` explicitly |
| CPU overhead | Low (~13 cycles per injection, ~4.0% for 4 KB) |
| Firmware changes | None (uses existing EDM protocol) |
| Hardware changes | None |
| Risk | Low: library code, no EDM modifications |
| Read support | Requires separate `gsrp_read()` implementation |
| Time to production | 3--6 months |

---

## 7.1.3 Approach B: Firmware Trap on Out-of-Range NOC Addresses

### Mechanism

Tensix cores issue a standard `noc_async_write` to a global address that falls in a
reserved "trap region" of ERISC L1. The ERISC firmware detects the write (via doorbell
polling) and converts it into a fabric packet.

```
+-------------------+     +-------------------+     +-------------------+
| Tensix Core       |     | ERISC L1          |     | EDM Router        |
|                   |     | (Trap Region)     |     |                   |
| noc_async_write() +---->| doorbell[n] = {   +---->| route_resolve()   |
| (to ERISC trap)   |     |   global_addr,    |     | header_construct()|
|                   |     |   src_addr,       |     | fabric_send()     |
|                   |     |   size, flags     |     |                   |
+-------------------+     | }                 |     +-------------------+
       Tensix RISC        +-------------------+           ERISC RISC
                               Trap Buffer
```

### Trap Entry Format (32 bytes, 16-byte aligned)

```
Offset  Size  Field
0x00    8 B   global_addr       -- 64-bit GlobalAddress
0x08    4 B   src_l1_addr       -- Source data address in Tensix L1
0x0C    4 B   size              -- Transfer size in bytes
0x10    4 B   flags             -- Operation type (write/read/atomic)
0x14    4 B   doorbell          -- Incremented to signal new request
0x18    8 B   reserved          -- Alignment padding
```

### Contention Analysis

With 140 Tensix cores and 14 active ERISC cores on Blackhole (Chapter 3, Section 1),
each ERISC handles requests from ~10 Tensix cores. The trap processing pipeline takes
~500--2,500 ns per entry (Section 6.5.3), yielding:

$$\text{ERISC trap throughput} = \frac{1}{0.5\text{--}2.5\ \mu\text{s}} = 0.4\text{--}2.0\ \text{M requests/s per ERISC}$$

For 14 ERISCs: 5.6--28 M requests/s system-wide. The M/D/1 queuing model (Section 6.5.3)
shows average response times of ~1.5 $\mu$s at moderate load, rising to ~4.5 $\mu$s under
heavy load.

For a 4 KB transfer (~328 ns fabric time), the extra 500--2,500 ns represents 150--760%
overhead -- far exceeding the $< 5\%$ target.

### Where Firmware Traps Make Sense

The firmware trap path is viable for **read requests**, where the ERISC must mediate anyway
(see Section 7.1.5). The ERISC naturally serves as the request/response coordinator because
it already runs the fabric router and can issue reads on behalf of remote requesters.

### Evaluation

| Criterion | Assessment |
|-----------|-----------|
| Kernel transparency | High: Tensix writes to a local NOC address |
| CPU overhead | Moderate (Tensix) + High (ERISC): ~500--2,500 ns |
| Firmware changes | Significant: new trap handler in EDM |
| Hardware changes | None |
| Risk | Moderate: ERISC contention and latency |
| Read support | Natural: ERISC can mediate round-trip |
| Time to production | 6--12 months |

---

## 7.1.4 Approach C: Hardware NOC Extension

A future-silicon approach where the NOC detects writes to global addresses (identified by
reserved high-order bits) and routes them to a hardware translation unit that generates
fabric packets without firmware intervention.

### Blackhole Extension Points

BH has two existing hardware features that could serve as foundations:

1. **`noc_address_translation_table` registers** -- Used for coordinate virtualization.
   These registers remap virtual NOC coordinates to physical coordinates. An extension
   could add entries that remap out-of-range coordinates to a trap ERISC.

2. **`NOC_SEC_FENCE_RANGE` registers** -- Used for security fencing. These define address
   ranges that trigger exceptions when accessed. An extension could repurpose the exception
   path to redirect writes to ERISC.

Neither mechanism currently supports cross-chip translation, but both establish the
precedent that the NOC interface includes address-range-based interception logic.

### Hardware TLB Design

Section 6.5.4 specified the hardware TLB parameters (8--16 entries, 1--2 cycle lookup,
95--99% hit rate). **Verdict:** Not feasible on WH or BH. Included to inform next-generation
(Quasar) silicon design requirements. The design parameters establish the hardware interface
contract that software approaches A and B must approximate.

---

## 7.1.5 Three-Approach Comparison

Section 6.4.4 presented a weighted decision matrix for the three candidate architectures.
Extending that analysis with injection-specific criteria (per-access overhead at 30% weight,
read-path support at 20% weight), the compiler-inserted shim scores highest for writes
(2.45/3.00) and the firmware trap scores highest for reads (read-path support: 3/3).

> **Recommendation:** The Phase 1 GSRP should use **Approach A (compiler-inserted shim)**
> for the write path and **Approach B (firmware trap)** for the read path, creating a
> **hybrid injection model**. Approach A handles the performance-critical write path with
> minimal overhead (~13 cycles), while Approach B leverages the ERISC's natural position as
> the fabric gateway for request-response reads. Approach C (hardware NOC extension) should
> inform next-generation silicon (Quasar) requirements.

---

## 7.1.6 Write/Read Path Asymmetry

The TT-Fabric is fundamentally **push-oriented**: data moves from sender to receiver via
posted writes. The `NocSendType` enum reflects this:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
enum NocSendType : uint8_t {
    NOC_UNICAST_WRITE         = 0,  // Posted write (fire-and-forget)
    NOC_UNICAST_INLINE_WRITE  = 1,  // Posted inline write
    NOC_UNICAST_ATOMIC_INC    = 2,  // Posted atomic
    // ...
    NOC_UNICAST_READ          = 7,  // Read request (requires response)
};
```

### Write Path (Posted)

```
Requester         Fabric          Target
    |                |               |
    |--- WRITE PKT ->|-- hop(s) ---->|
    |                |               |- NOC write to L1
    |                |               |
    | (fire-and-forget: no response needed)
```

The requester issues the packet and continues execution. Completion is inferred from
credit flow: when the EDM sends back an ack credit, the requester knows the buffer slot
is free for reuse. The actual write to remote L1 is asynchronous.

### Read Path (Request/Response)

```
Requester         Fabric          Target ERISC     Target Core
    |                |               |                |
    |--- READ REQ -->|-- hop(s) ---->|                |
    |                |               |-- NOC read --->|
    |                |               |<-- data -------|
    |                |               |                |
    |<-- READ RSP ---|<- hop(s) -----|                |
    |- NOC write     |               |                |
    |  to local L1   |               |                |
```

| Requirement | Write Path | Read Path |
|-------------|:----------:|:---------:|
| Response channel | Not needed | Must reserve return path |
| ERISC read handler | Not needed | New firmware component |
| Round-trip latency | Single direction | 2x fabric hops + ERISC processing |
| Deadlock risk | Low (unidirectional) | High (request/response cycle) |
| Credit accounting | Single direction | Both directions |

### UDM Read Control Headers as Building Blocks

The existing `UDMReadControlHeader` (Chapter 3, Section 3.6) already carries the fields
needed for a read request/response:

```cpp
// tt_metal/fabric/fabric_edm_packet_header.hpp
struct UDMReadControlHeader {
    uint8_t src_chip_id;
    uint16_t src_mesh_id;
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint32_t src_l1_address;    // Where to write the response data
    uint32_t size_bytes;
    uint8_t risc_id;
    uint8_t transaction_id;     // For matching responses to requests
    uint8_t initial_direction;
} __attribute__((packed));  // 16 bytes
```

The GSRP read path extends this with a completion signaling mechanism:

```cpp
// Proposed GSRP read request extension
struct GsrpReadRequestHeader : public UDMReadControlHeader {
    uint32_t completion_counter_addr;  // L1 address for atomic-inc on completion
    uint8_t  completion_counter_val;   // Increment value (typically 1)
    uint8_t  padding[3];
};  // 24 bytes total
```

### Read Latency Model

Section 6.5.6 established that a single-hop remote read takes ~10--21 $\mu$s (dominated by
two fabric transits), yielding a remote/local ratio of ~2,500--5,250x compared to the ~4 ns
local L1 read. This NUMA-like cliff is exposed to programmers via Section 6.1, Requirement 3
(performance visibility).

### Multi-Hop Read Scaling

For an $h$-hop read, each direction contributes $h$ store-and-forward delays:

```math
T_{\text{read}}(h) \approx 2h \times T_{\text{hop}} + T_{\texttt{erisc\_process}}
```

| Hops | Estimated Read Latency |
|:----:|:----------------------:|
| 1 | 10--21 $\mu$s |
| 2 | 20--42 $\mu$s |
| 4 | 40--84 $\mu$s |
| 8 | 80--168 $\mu$s |

> **Key Insight:** The write/read asymmetry is not just a latency difference -- it is an
> architectural difference. Writes use the existing unidirectional fabric pipeline. Reads
> require bidirectional traffic, consuming fabric capacity in both directions and introducing
> new deadlock risks (Section 7.3). GSRP should strongly favor write-based communication
> patterns: the producer writes data to the consumer's L1 rather than the consumer reading
> from the producer's L1. This "push model" aligns with how existing CCL operations work
> (all-gather pushes data to receivers).

---

## 7.1.7 Adapting `FabricEriscDatamoverConfig` for Transparent Injection

The current `FabricEriscDatamoverConfig` (defined in
`tt_metal/fabric/erisc_datamover_builder.hpp`) allocates sender and receiver channels with
fixed buffer counts and sizes. Transparent injection requires three adaptations:

### Persistent Connection Pool

Today, each worker opens and closes a connection per kernel invocation (Chapter 4,
Section 2.3), incurring ~1--5 $\mu$s overhead (6 NOC transactions per open, 4 per close).
Under GSRP, connections must be **persistent** -- opened at fabric initialization time and
kept alive for the duration of the program.

```cpp
// Proposed extension to FabricEriscDatamoverConfig
struct FabricEriscDatamoverConfig {
    // ... existing fields ...

    // GSRP persistent connection pool
    uint32_t gsrp_persistent_connection_count = 0;  // 0 = disabled
    bool gsrp_auto_open_connections = false;
    // When enabled, sender channel 0 maintains persistent connections
    // that survive across kernel launches
};
```

### Connection Pool Design

The per-core connection pool builds on the `RoutingPlaneConnectionManager` (Chapter 4,
Section 2.5):

```
Connection Pool (per Tensix core, L1 resident):
+------+----------+----------+-------------------+--------+
| slot | dst_mesh | dst_chip | sender_handle_ptr | active |
| 4b   | 8b       | 8b       | 32b (L1 pointer)  | 1b     |
+------+----------+----------+-------------------+--------+
Entries: 4--16 (configurable).  Per-entry: 8 bytes.  Total: 32--128 bytes.
```

Connection selection exploits temporal locality:

```cpp
FORCE_INLINE WorkerToFabricEdmSender* gsrp_conn_pool_select(
    uint8_t dst_mesh, uint8_t dst_chip)
{
    // Fast path: check last-used connection (temporal locality)
    if (pool_[last_used_].dst_mesh == dst_mesh &&
        pool_[last_used_].dst_chip == dst_chip &&
        pool_[last_used_].active) {
        return pool_[last_used_].sender;
    }
    // Slow path: linear scan (pool is small, 4--16 entries)
    for (uint8_t i = 0; i < pool_size_; i++) {
        if (pool_[i].dst_mesh == dst_mesh &&
            pool_[i].dst_chip == dst_chip &&
            pool_[i].active) {
            last_used_ = i;
            return pool_[i].sender;
        }
    }
    // Miss: evict LRU and open new connection
    return gsrp_conn_pool_open_new(dst_mesh, dst_chip);
}
```

ML workloads exhibit strong temporal locality: a core typically communicates with 2--4
destinations for the entire training step (Section 6.5.5).

### Multi-Worker Sender Channel via MUX

The single-worker-per-EDM-channel constraint (Chapter 3, Section 2) is a bottleneck for
GSRP. With 140 Tensix cores and 14 ERISC sender channels, at most 14 cores can have
direct EDM connections simultaneously.

The MUX mode (`FabricTensixConfig::MUX`) addresses this by dedicating Tensix cores as
traffic aggregators:

```
Workers (8--16 per MUX)         MUX Tensix Core          ERISC Sender Channel 0
+--------+                      +---------------+         +------------------+
| Core 0 |-- NOC write -------->|               |         |                  |
| Core 1 |-- NOC write -------->| Aggregate     |-- NOC ->| EDM Sender Ch 0  |
|  ...   |-- NOC write -------->| into fabric   |  write  |                  |
| Core 15|-- NOC write -------->| packets       |         |                  |
+--------+                      +---------------+         +------------------+
```

| Strategy | Workers per ERISC | Throughput | L1 cost |
|----------|:-----------------:|:----------:|:-------:|
| Single connection (current) | 1 | 1x | Minimal |
| MUX aggregation | 8--16 | ~0.9x (mux overhead) | 1 Tensix core |
| Routing plane distribution | 4 | ~3.8x | 4 connection slots |
| MUX + plane distribution | 32--64 | ~3.5x | 4 Tensix cores |

> **Recommendation:** For Phase 1, use routing plane distribution (4 workers per ERISC).
> For Phase 2, add MUX aggregation to scale to 32+ workers per direction. GSRP should
> require MUX mode for configurations where more than $N_{\text{ERISC sender channels}}$
> cores need concurrent fabric access.

### Buffer Sizing for Mixed Traffic

When GSRP traffic shares EDM channels with explicit CCL traffic, buffer partitioning
prevents starvation:

```cpp
enum class ChannelPartitionMode : uint8_t {
    SHARED = 0,         // All traffic shares all buffer slots
    SPLIT_GSRP_CCL = 1, // Reserve N slots for GSRP, M for CCL
};
```

Under `SPLIT_GSRP_CCL` mode, GSRP traffic is guaranteed buffer slots even when CCL
operations saturate the channel.

---

## 7.1.8 Per-Core Resource Summary

| Resource | Write Path (Approach A) | Read Path (Approach B) | Total |
|----------|:-----------------------:|:----------------------:|:-----:|
| Route cache | 512 B | Shared | 512 B |
| Connection pool | 128 B | Shared | 128 B |
| Chip route table | 248 B (32-chip) | Shared | 248 B |
| Segment registers | 128 B | Shared | 128 B |
| Read response buffer | -- | 4,096 B (1 outstanding read) | 4,096 B |
| Read completion counter | -- | 4 B | 4 B |
| **Per-core total** | **1,016 B** | **4,100 B** | **5,116 B** |

The 5,116 B total is within the 16 KB budget from Section 6.1 and leaves 10.9 KB for
additional Phase 2 features (multiple outstanding reads, congestion counters, extended
route caches).

---

## GSRP Implications

The transparent injection design confirms that the GSRP write path is implementable on
current hardware using a compiler-inserted shim with ~13 cycles overhead per remote access.
The read path requires ERISC firmware extension but leverages the existing
`UDMReadControlHeader` format and `NocSendType::NOC_UNICAST_READ`. The hybrid approach
(shim for writes, firmware trap for reads) provides the best trade-off between performance
and feasibility. The per-core L1 budget of ~5.1 KB (write + read paths combined) is within
the 16 KB target and accounts for a single outstanding read; supporting multiple outstanding
reads would increase this by ~4 KB per additional slot. The persistent connection model
eliminates the 10-statement open/close handshake that accounts for the largest portion of
current boilerplate. MUX aggregation and routing plane distribution solve the
single-worker-per-channel bottleneck, enabling all 140 BH Tensix cores to access the fabric
through 14 ERISC channels.

---

## Key Takeaways

- **The compiler-inserted shim (Approach A) is the recommended write-path injection
  mechanism**, achieving ~13 cycles per remote access with zero firmware or hardware changes
  by composing the existing mesh API with an L1 route cache and connection pool.

- **The firmware trap (Approach B) is the recommended read-path mechanism**, leveraging the
  ERISC's natural position as fabric gateway. Read latency is ~10--21 $\mu$s per single-hop
  access (~2,500x local), reinforcing the GSRP's NUMA-aware design.

- **Persistent connections eliminate the open/close handshake overhead**, removing the
  largest single source of per-kernel boilerplate (~10 of the ~45 lines identified in
  Chapter 4, Section 2.7). Connections survive across kernel launches.

- **The `UDMReadControlHeader` and `NocSendType::NOC_UNICAST_READ` are direct building
  blocks** for the GSRP read path, requiring only a completion-counter extension
  (`GsrpReadRequestHeader`) to support round-trip reads.

- **MUX aggregation is essential for multi-core GSRP workloads** on Blackhole, enabling
  8--16 workers to share a single EDM sender channel through a dedicated Tensix core.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `tt_metal/fabric/fabric_edm_packet_header.hpp` | `NocSendType` (including `NOC_UNICAST_READ`), `UDMReadControlHeader`, `UDMWriteControlHeader`, `NocUnicastCommandHeader`, `PacketHeaderBase`, `HybridMeshPacketHeaderT` |
| `tt_metal/fabric/erisc_datamover_builder.hpp` | `FabricEriscDatamoverConfig`, sender/receiver channel layout, buffer sizing, NOC VC assignment |
| `tt_metal/fabric/hw/inc/mesh/api.h` | `fabric_unicast_noc_unicast_write()`, two-phase send protocol, `fabric_set_unicast_route()` |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | `WorkerToFabricEdmSenderImpl`, `edm_has_space_for_packet()`, connection open/close protocol |
| `tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp` | `RoutingPlaneConnectionManager`, `MaxConnections = 4` |
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | `compressed_route_2d_t`, `routing_table_t`, `fabric_connection_info_t`, `fabric_connection_sync_t` |
| `tt_metal/fabric/fabric_tensix_builder.hpp` | `FabricTensixConfig::MUX`, MUX core builder |

---

**Next:** [`02_flow_control_and_congestion.md`](./02_flow_control_and_congestion.md)
