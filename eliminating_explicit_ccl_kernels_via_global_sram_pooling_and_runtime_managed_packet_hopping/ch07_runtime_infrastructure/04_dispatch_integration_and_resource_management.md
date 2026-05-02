# 7.4 Dispatch Integration and Resource Management

## Context

The previous three sections of this chapter designed the transparent injection mechanism
(Section 7.1), flow control extensions (Section 7.2), and deadlock avoidance strategies
(Section 7.3). This final section addresses two practical concerns that determine whether
the GSRP can be deployed on real hardware: **dispatch integration** (how the existing
host-side command queue and dispatch system interacts with transparent cross-chip traffic)
and **resource management** (how the scarce L1 SRAM is partitioned between GSRP metadata,
application data, and dispatch infrastructure). The reader should be familiar with the
`DispatchMemMap` class (Chapter 2, Section 4), the `MeshDevice`/`MeshCommandQueue`
infrastructure (Chapter 4, Section 4), the `MeshSocket` API (Chapter 4, Section 3), and the
L1 memory layout from Chapter 2, Section 1. This section also covers telemetry and debugging
extensions, coexistence with explicit CCL operations, and system-level resource accounting for
a 32-chip Blackhole Galaxy.

---

## 7.4.1 Dispatch System Architecture

The TT-Metal dispatch system moves data and commands from the host to device cores through
a hierarchy of specialized cores:

```
Host (PCIe)
  |
  v
MeshCommandQueue          -- Mesh-level dispatch (coordinates across devices)
  |
  v
Per-Device CommandQueue   -- Per-chip hardware command queue
  |
  v
Prefetcher Core           -- Dedicated Tensix core that fetches commands from host
  |
  v
Dispatch Core             -- Dedicated Tensix core that executes commands
  |
  v
Worker Cores              -- Run user kernels (80-140 per chip)
```

The `MeshCommandQueue` (defined in `tt_metal/api/tt-metalium/mesh_command_queue.hpp`)
provides the mesh-level interface:

```cpp
// tt_metal/api/tt-metalium/mesh_command_queue.hpp
class MeshCommandQueue {
public:
    virtual void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) = 0;
    virtual void enqueue_write_mesh_buffer(
        const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) = 0;
    virtual void enqueue_read_mesh_buffer(
        void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) = 0;
    virtual MeshEvent enqueue_record_event(...) = 0;
    virtual void enqueue_wait_for_event(const MeshEvent& sync_event) = 0;
    virtual void finish(...) = 0;
};
```

The dispatch system manages per-device L1 layout via `DispatchMemMap`
(`tt_metal/impl/dispatch/dispatch_mem_map.hpp`), which defines command queue addresses,
prefetch buffers, and dispatch buffer regions.

---

## 7.4.2 GSRP-Dispatch Integration Points

The GSRP runtime must integrate with the dispatch system at four points:

| Integration Point | Current Owner | GSRP Extension |
|-------------------|:------------:|----------------|
| **Program launch** | `MeshWorkload::enqueue()` | Initialize GSRP connection pool and route cache |
| **Runtime arg setup** | `append_routing_plane_connection_manager_rt_args()` | Replace with GSRP connection pool descriptor |
| **Barrier sync** | `MeshCommandQueue::finish()` | Add GSRP epoch barriers |
| **Program teardown** | `MeshWorkload` destructor | Flush GSRP credits, close persistent connections |

### MeshBuffer Integration

The `MeshBuffer` abstraction (Chapter 4, Section 4) allocates buffers at consistent L1
addresses across the mesh. GSRP extends this by automatically computing `GlobalAddress`
values for each shard:

```cpp
// Proposed extension to MeshBuffer
class MeshBuffer {
public:
    // Existing:
    static std::shared_ptr<MeshBuffer> create(
        const MeshDevice& mesh_device, const DeviceLocalBufferConfig& config);

    // GSRP extension: return GlobalAddress for a specific shard
    GlobalAddress gsrp_address(MeshCoordinate shard_coord) const {
        auto [mesh_id, chip_id] = device_to_fabric_node(shard_coord);
        auto [noc_x, noc_y] = buffer_core_coord();
        return GlobalAddress::from_components(
            mesh_id, chip_id, noc_x, noc_y, buffer_address());
    }

    // GSRP extension: write the shard address table to all cores
    void populate_gsrp_address_table(MeshCommandQueue& cq) const;
};
```

The `populate_gsrp_address_table()` method distributes a compact address map (8 bytes per
shard, see Chapter 6, Section 5) to every Tensix core in the mesh. For a 32-chip system
with per-chip sharding, this is 256 bytes per core -- a one-time cost at program setup.

### MeshSocket Coordination

The `MeshSocket` API (Chapter 4, Section 3) and GSRP serve complementary roles:

| Aspect | GSRP Fat Fabric API | MeshSocket |
|--------|:-------------------:|:----------:|
| Level | Device-side kernel API | Host-side tensor API |
| Granularity | Byte-level L1 access | Tensor-level buffer send/recv |
| Address model | GlobalAddress (64-bit) | Named socket endpoints |
| Use case | Fine-grained kernel communication | Structured pipeline stages |

GSRP and MeshSocket can coexist but must coordinate their use of fabric resources:

| Resource | MeshSocket usage | GSRP usage | Resolution |
|----------|:----------------:|:----------:|:----------:|
| EDM sender channel 0 | Exclusive (per-socket) | Shared (connection pool) | GSRP reserves Plane 0 |
| Routing planes | Per-socket allocation | Round-robin across planes | Resource manager arbitration |
| Packet header pool | Per-socket headers | Per-core headers | Separate pools |
| L1 buffers | Socket-managed CBs | GSRP-visible regions | Separate L1 regions |

The GSRP runtime registers its connection pool with the fabric resource manager during
initialization. MeshSocket creation queries the resource manager and avoids planes
already reserved for GSRP.

### Trace Replay Compatibility

TT-Metal's trace system (`MeshTraceDescriptor`) records command sequences for replay. GSRP
operations must be trace-compatible:

- **`gsrp_write()`:** Since the injection is performed by device-side code (not host
  commands), GSRP writes are automatically captured within the kernel's trace.
- **Route table updates:** If topology changes between trace capture and replay, route
  tables must be re-populated. The trace replay system should invoke
  `populate_gsrp_address_table()` after route table recomputation.
- **Connection pool state:** Persistent connections are stateful. Trace replay must verify
  that connections are open before replaying traces that contain GSRP operations.

---

## 7.4.3 GSRP Initialization Sequence

### Host-Side Setup

Before any GSRP-enabled kernel runs, the host must prepare the per-core GSRP metadata:

```
Host-Side GSRP Initialization Sequence:
  1. Query ControlPlane for mesh topology and routing tables
  2. Compute per-chip route tables (gsrp_chip_route, Section 6.5.2)
  3. For each chip in the mesh:
     a. Write route table to L1 at GSRP_ROUTE_TABLE_BASE (248 B for 32 chips)
     b. Write connection pool descriptors (128 B)
     c. Write GSRP configuration (admission limits, VC assignments, etc.)
  4. Open persistent fabric connections on each Tensix core
  5. Signal initialization complete via global semaphore
```

**Initialization cost estimate:**

| Step | Per-Chip Cost | 32-Chip Galaxy |
|------|:------------:|:--------------:|
| Route table write (140 cores x 248 B) | ~2.8 $\mu$s | ~90 $\mu$s |
| Connection pool write (140 cores x 128 B) | ~1.4 $\mu$s | ~45 $\mu$s |
| Connection open (4 connections per core) | ~50 $\mu$s | ~1.6 ms |
| Global semaphore barrier | ~30 $\mu$s | ~30 $\mu$s |
| **Total initialization** | | **~1.8 ms** |

This is a one-time cost per program, amortized over the entire training job.

### Kernel-Side Initialization

When a GSRP-enabled kernel starts, it performs a lightweight local initialization:

```cpp
// Per-kernel GSRP initialization (device side)
FORCE_INLINE void gsrp_kernel_init() {
    // 1. Read GSRP config from pre-populated L1
    gsrp_config = *(volatile gsrp_config_t*)GSRP_CONFIG_BASE;

    // 2. Initialize route cache (zero all valid bits)
    for (uint8_t i = 0; i < GSRP_ROUTE_CACHE_ENTRIES; i++) {
        gsrp_route_cache[i].valid = 0;
    }

    // 3. Read connection pool state (already opened by host)
    gsrp_pool = *(volatile gsrp_connection_pool_t*)GSRP_POOL_BASE;
}
```

Cost: ~30 cycles. This replaces the ~45 lines of per-kernel connection management code
identified in Chapter 4, Section 2.7.

---

## 7.4.4 L1 Resource Management

### Current L1 Budget

Chapter 2, Section 1 documented the Blackhole L1 layout. The fabric-related reserved
regions are:

| Region | Size | Contents |
|--------|:----:|----------|
| Routing table | 2,576 B | `routing_l1_info_t`: base (516 B) + paths (1,024 B) + exit nodes (1,024 B) + padding (12 B) |
| Fabric connections | 656 B | `tensix_fabric_connections_l1_info_t`: 16 RO entries + mask + 16 RW entries |
| Packet header pool | 3,456 B | `PACKET_HEADER_MAX_SIZE` (144 B) x `NUM_PACKET_HEADERS` (24) |
| Connection sync/lock | 144 B | Lock (4 B) + initialized (4 B) + connection (128 B) + padding (8 B) |
| Fabric counters | 32 B | 3 barrier types x 2 DMs x 4 B + 8 B padding |
| **Existing fabric subtotal** | **6,864 B** | |

> **Note on packet header pool size:** The plan spec estimates ~2.3 KB for packet headers,
> but the codebase uses `PACKET_HEADER_MAX_SIZE = 144` (including the UDM header extension)
> and `NUM_PACKET_HEADERS = 24`, yielding 3,456 B. This section uses the codebase-accurate
> value.

The `MEM_MAP_END` constant marks the boundary between reserved and user-available L1. On
Blackhole, `MEM_MAP_END` is approximately 40 KB from the L1 base (0x09DE0 = 40,416 bytes),
including firmware, mailbox, counters, and fabric regions. The remaining ~1,496 KB is
available for user data.

### GSRP L1 Overhead

Section 7.1.8 established the per-core GSRP cost. Here is the complete breakdown including
resources from Sections 7.1--7.4:

| GSRP Component | Size | Source |
|----------------|:----:|--------|
| Chip route table (32-chip) | 248 B | Sec 6.5.2, 7.1 |
| Route cache (8 entries) | 512 B | Sec 6.5.5, 7.1 |
| Connection pool (16 entries) | 128 B | Sec 6.4.1, 7.1 |
| Segment registers (8 entries) | 128 B | Sec 6.5, 7.1 |
| Read response buffer (1 outstanding) | 4,096 B | Sec 7.1.5 |
| Read completion counter | 4 B | Sec 7.1.5 |
| Per-destination credit table | 64 B | Sec 7.2.4 |
| Rate limiter state | 16 B | Sec 7.2.6 |
| GSRP configuration | 32 B | Sec 7.4.3 |
| Telemetry counters | 48 B | Sec 7.4.7 |
| Export region config | 12 B | Sec 7.4.5 |
| **GSRP subtotal** | **5,288 B** | |

**Combined L1 impact:**

The existing fabric metadata (6,864 B) is already included within `MEM_MAP_END` (40,416 B).
GSRP metadata is added on top of `MEM_MAP_END`:

| Configuration | MEM\_MAP\_END | + GSRP | Total Reserved | Available L1 |
|:-------------|:--------------:|:------:|:--------------:|:------------:|
| BH Phase 1 (write-only) | 40,416 B | 1,016 B | ~40.5 KB | ~1,495.5 KB |
| BH Phase 2 (full GSRP) | 40,416 B | 5,288 B | ~44.6 KB | ~1,491.4 KB |

The Phase 1 GSRP metadata (1,016 B, write-path only) is 0.07% of BH L1. The full Phase 2
deployment (5,288 B) is 0.35% -- well within the 16 KB budget from Section 6.1.

> **Key Insight:** The GSRP L1 overhead is negligible relative to total L1. Phase 1 adds
> only 1,016 bytes (0.07% of BH L1). The real L1 trade-off is the globally-visible region
> partitioning (Section 7.4.5), which trades private capacity for remote accessibility -- a
> workload-specific tuning knob, not a fixed cost.

---

## 7.4.5 L1 Partitioning: Globally Visible vs. Locally Private

The GSRP address space design (Chapter 6, Section 2) introduced the concept of partitioning
each core's L1 into globally-visible and core-private regions.

### Memory Map

```
L1 Memory Map (Blackhole Tensix, GSRP-enabled):

0x0000 +----------------------------------+
       | System Reserved (~39.5 KB)       |
       | (firmware, mailbox, zeros)        |
0x09DE0+----------------------------------+  <- MEM_MAP_END
       | Fabric Reserved (~6.7 KB)        |
       | (routing table, connections,      |
       |  packet headers, sync)            |
       +----------------------------------+
       | GSRP Metadata (~5.2 KB)          |
       | (route cache, conn pool, credits, |
       |  read response buffer, config)    |
       +----------------------------------+  <- GSRP_EXPORT_BASE
       | GSRP-Exportable Region           |
       | (globally addressable, remote     |
       |  cores can read/write here)       |
       |                                   |
       | Size: configurable per workload   |
       +----------------------------------+  <- GSRP_PRIVATE_BASE
       | Locally Private Region            |
       | (compute buffers, CBs, stack)     |
       | NOT accessible via GSRP           |
       +----------------------------------+
0x180000                                     <- End of L1 (1,536 KB)
```

### Per-Workload Sizing Guidelines

| Workload type | Globally-visible | Core-private | Rationale |
|:-------------|:----------------:|:------------:|-----------|
| Data parallel (all-reduce) | 10--25% | 75--90% | Few remote accesses, large local working sets |
| Tensor parallel (all-gather) | 25--40% | 60--75% | Moderate remote data (activation shards) |
| Pipeline parallel (P2P) | 10--20% | 80--90% | Small P2P buffers, large local compute |
| MoE (expert routing) | 40--60% | 40--60% | Dynamic remote access, many expert buffers |
| KV-cache sharing (inference) | 50--70% | 30--50% | Heavy remote reads of KV cache |

The partition is configured per-program via `GsrpAllocatorConfig`, with
`gsrp_global_size = 0` providing zero overhead when GSRP is unused:

```cpp
// Proposed extension to AllocatorConfig
struct GsrpAllocatorConfig {
    uint32_t gsrp_global_base;     // Start of globally-visible region
    uint32_t gsrp_global_size;     // Size of globally-visible region
    uint32_t gsrp_private_base;    // Start of core-private region
    // Invariant: gsrp_global_base + gsrp_global_size == gsrp_private_base
};
```

### Exportable Region Validation

The GSRP-exportable region is a **host-configured** attribute set during `gsrp_initialize`.
It is NOT enforced by hardware (no memory protection on Tensix L1) but is checked by the
GSRP firmware on the destination ERISC before delivering a remote write:

```cpp
// Proposed destination-side validation on ERISC
struct gsrp_export_config_t {
    uint32_t export_base;   // Start of exportable region in L1
    uint32_t export_size;   // Size of exportable region
    uint32_t export_flags;  // R/W permissions, owner mesh/chip
};
// Size: 12 bytes per core (written to ERISC during initialization)

FORCE_INLINE bool is_address_in_export_region(uint32_t l1_offset) {
    return (l1_offset >= export_config.export_base) &&
           (l1_offset < export_config.export_base + export_config.export_size);
}
```

Writes to addresses outside the exportable region are **silently dropped** with a telemetry
counter increment (`gsrp_export_violations` in Section 7.4.7) for debugging.

---

## 7.4.6 Dispatch Coordination for Epoch Barriers

The relaxed-epoch barrier model (Section 7.3.5) requires the dispatch system to coordinate
barrier execution across all chips. Two approaches:

### Host-Driven Barriers

The host issues barrier commands via `MeshCommandQueue`:

```cpp
// Host-side barrier implementation
void gsrp_epoch_barrier(MeshCommandQueue& cq) {
    // 1. Record an event on all devices
    MeshEvent barrier_event = cq.enqueue_record_event();

    // 2. Wait for the event on all devices
    cq.enqueue_wait_for_event(barrier_event);

    // 3. Ensure all fabric traffic has drained
    cq.finish();

    // 4. Optionally: update route tables if topology changed
    if (route_table_dirty) {
        repopulate_gsrp_route_tables(cq);
    }
}
```

**Latency:** The host round-trip adds ~10--50 $\mu$s on top of the inter-chip barrier
latency. For epoch durations of 200+ $\mu$s, this is acceptable.

### Device-Driven Barriers

For lower latency, the barrier can be implemented entirely on-device using the distributed
semaphore protocol from Section 7.3.5:

```cpp
// Device-side barrier (runs on each Tensix core)
FORCE_INLINE void gsrp_device_epoch_barrier() {
    // Phase 1: Local chip barrier
    noc_semaphore_inc(local_barrier_sem, 1);
    noc_semaphore_wait(local_barrier_sem, NUM_ACTIVE_CORES);

    // Phase 2: Designated barrier core broadcasts to all chips
    if (my_core_xy == BARRIER_MASTER_CORE) {
        for (uint8_t dst = 0; dst < NUM_CHIPS; dst++) {
            if (dst == MY_CHIP) continue;
            GlobalAddress remote_barrier = GlobalAddress::from_components(
                MY_MESH, dst, BARRIER_CORE_X, BARRIER_CORE_Y,
                GLOBAL_BARRIER_SEM_OFFSET);
            gsrp_atomic_inc(remote_barrier, 1);
        }
        noc_semaphore_inc(global_barrier_sem, 1);
    }

    // Phase 3: Wait for all chips
    noc_semaphore_wait(global_barrier_sem, NUM_CHIPS);

    // Phase 4: Reset for next epoch
    if (my_core_xy == BARRIER_MASTER_CORE) {
        noc_semaphore_set(global_barrier_sem, 0);
    }
    noc_semaphore_set(local_barrier_sem, 0);
}
```

**Latency:** Device-driven barriers eliminate the host round-trip, reducing barrier latency
to the inter-chip fabric latency alone (~10--30 $\mu$s for 8--32 chip systems).

---

## 7.4.7 Telemetry and Debugging

Transparent cross-chip transfers are inherently harder to debug than explicit CCL operations
because the programmer does not write the communication code. The GSRP must provide
comprehensive observability.

### Per-Core GSRP Telemetry Counters

Add a lightweight counter block to each Tensix core's GSRP metadata region:

```
GSRP Telemetry Block (per core, 48 bytes):
+------------------------------------------+
| gsrp_writes_issued      : uint32_t       |  4 B
| gsrp_writes_local       : uint32_t       |  4 B (fast path hits)
| gsrp_writes_remote      : uint32_t       |  4 B
| gsrp_reads_issued       : uint32_t       |  4 B
| gsrp_reads_completed    : uint32_t       |  4 B
| gsrp_write_stall_cycles : uint32_t       |  4 B (flow control waits)
| gsrp_read_stall_cycles  : uint32_t       |  4 B
| gsrp_credit_exhaustions : uint32_t       |  4 B
| gsrp_route_cache_hits   : uint32_t       |  4 B
| gsrp_route_cache_misses : uint32_t       |  4 B
| gsrp_bytes_sent         : uint64_t       |  8 B
+------------------------------------------+
Total: 48 bytes per core
```

These counters are updated by the GSRP injection shim (Section 7.1) at negligible cost
(~1 cycle per counter increment). They are readable by the host via NOC reads or via
the `FabricTelemetryReader` infrastructure.

### Per-ERISC GSRP Telemetry

Add counters to each ERISC for GSRP-specific traffic:

```
GSRP ERISC Telemetry (per ERISC, 32 bytes):
+------------------------------------------+
| gsrp_packets_forwarded   : uint32_t      |  4 B
| gsrp_read_requests_served: uint32_t      |  4 B
| gsrp_read_responses_sent : uint32_t      |  4 B
| gsrp_congestion_events   : uint32_t      |  4 B (ECN bit set)
| gsrp_export_violations   : uint32_t      |  4 B (writes to non-export region)
| gsrp_credit_updates_sent : uint32_t      |  4 B
| gsrp_admission_rejects   : uint32_t      |  4 B
| reserved                 : uint32_t      |  4 B
+------------------------------------------+
```

### Watcher Integration

The watcher system monitors NOC transactions and validates memory access patterns. Under
GSRP, a core can appear hung if it is blocked on flow control, read completion, or an
epoch barrier. The watcher should recognize these GSRP-specific stall conditions:

| Check | Description | Action on violation |
|-------|-------------|-------------------|
| Address range | `GlobalAddress.l1_offset < MEM_L1_SIZE` | Assert + log |
| Export region | Destination offset within GSRP-visible range | Assert + log |
| Route validity | Cached route has not been invalidated | Warn + force re-lookup |
| Credit overflow | `per_dst_outstanding <= MAX_PER_DST` | Assert + log |
| Connection state | Connection pool entry is in OPEN state | Assert + log |
| GSRP stall detection | Distinguish GSRP stalls from kernel bugs | Report stall type |

These checks are gated behind `WATCHER_ENABLED` and have zero cost in production builds.

### DPrint Integration

GSRP operations can be traced via DPrint with a dedicated format:

```cpp
#ifdef GSRP_DEBUG
    DPRINT << "gsrp_write: " << "M" << ga.mesh_id()
           << ":D" << ga.chip_id()
           << ":C(" << ga.noc_x() << "," << ga.noc_y() << ")"
           << ":0x" << HEX() << ga.l1_offset()
           << " size=" << DEC() << size << ENDL();
#endif
```

The `M<mesh>:D<chip>:C(x,y):0x<offset>` notation matches the plan convention for
`GlobalAddress` display, making debug output immediately parseable.

### Packet Recording Extension

The `FabricPacketRecorder` annotates GSRP packets with their origin:

```cpp
// Proposed GSRP packet annotation
struct GsrpPacketAnnotation {
    uint8_t  source_type;       // 0=explicit CCL, 1=GSRP write, 2=GSRP read
    uint8_t  source_core_x;    // Originating Tensix core
    uint8_t  source_core_y;
    uint8_t  priority;          // GSRP priority class
    uint64_t global_addr;       // Target GlobalAddress
    uint32_t timestamp;         // Cycle counter at injection
};
```

This annotation enables post-mortem questions like "which core generated this packet?"
and "was this a GSRP or CCL transfer?" -- critical for diagnosing performance regressions
when both traffic types coexist.

---

## 7.4.8 Coexistence with Explicit CCL Operations

GSRP does not replace explicit CCL operations (Chapter 1). The two coexist on the same
fabric:

```
Traffic Mix on Fabric:
+------------------------------------------------------------------+
|  Explicit CCL Traffic          |  GSRP Traffic                   |
|  (all-gather, reduce-scatter,  |  (gsrp_write, gsrp_read,       |
|   all-reduce)                  |   gsrp_atomic_inc)              |
|                                |                                 |
|  - Pre-computed routes         |  - Runtime-resolved routes      |
|  - Pre-allocated connections   |  - Persistent connection pool   |
|  - Deterministic scheduling    |  - Dynamic scheduling           |
+------------------------------------------------------------------+
```

### Resource Isolation

| Resource | Explicit CCL | GSRP |
|----------|:------------:|:----:|
| Connections | Per-op, open/close | Persistent pool |
| Routing planes | Dedicated (per MeshSocket) | Plane 0 (reserved) |
| Packet headers | Per-op allocation | Pre-allocated pool |
| Flow control | Per-connection credits | Pool-wide credits |

The key coordination point is **routing plane allocation**: GSRP reserves plane 0 for
persistent connections, leaving planes 1+ available for explicit CCL. On Wormhole (4
planes), this provides 3 planes for CCL operations -- no loss compared to the current
default. On Blackhole (2 planes), GSRP shares plane 0 with one direction of CCL traffic,
managed by the VC-based priority mechanism from Section 7.2.5.

### Priority Between Traffic Types

When explicit CCL and GSRP traffic compete for the same Ethernet link:

- CCL bulk operations (all-gather, reduce-scatter) use VC 1 (normal priority).
- GSRP latency-sensitive traffic (semaphore increments, small control) uses VC 0 (high
  priority).
- GSRP bulk data transfers share VC 1 with CCL.
- The rate limiter (Section 7.2.6) caps GSRP injection rate to a configurable fraction of
  link bandwidth (default: 50%).

### MeshWorkload Integration

A `MeshWorkload` can contain both CCL operations and GSRP-enabled kernels:

```cpp
// Host-side program setup
MeshWorkload workload(mesh_device);

// Traditional CCL operation (explicit)
auto all_gather_program = create_all_gather_program(input_tensor, dim, cluster_axis);
workload.add_program(all_gather_program, mesh_coordinates);

// GSRP-enabled kernel (implicit communication)
auto compute_program = create_gsrp_compute_program(weights, activations);
workload.add_program(compute_program, mesh_coordinates);

mesh_command_queue.enqueue_mesh_workload(workload, /*blocking=*/false);
```

The dispatch system launches both programs. The CCL program uses explicit fabric
connections; the GSRP program uses the persistent connection pool. Both share the same
underlying Ethernet links, with flow control and priority queuing managing the contention.

---

## 7.4.9 System-Level Resource Accounting

### Per-Chip ERISC Resources

The GSRP read handler (Section 7.1.5) runs on each active ERISC core:

| ERISC Resource | Current Usage | GSRP Addition | Total |
|----------------|:------------:|:-------------:|:-----:|
| Router firmware | ~32 KB | ~2 KB (read handler) | ~34 KB |
| Read staging buffer | 0 | 4,096 B | 4,096 B |
| GSRP telemetry | 0 | 32 B | 32 B |
| Export config cache | 0 | 12 B per remote core | ~1,680 B (140 cores) |
| **ERISC L1 addition** | | | **~7.8 KB** |

On Blackhole ERISC (512 KB L1), 7.8 KB is 1.5% -- negligible.

### Multiple Outstanding Reads

Section 7.1.5 allocated 4,096 bytes for a single outstanding read response buffer. For
read-heavy workloads, multiple outstanding reads improve utilization:

| Outstanding Reads | Read Buffer L1 | Total GSRP Overhead | % of BH L1 |
|:-----------------:|:--------------:|:-------------------:|:----------:|
| 1 | 4,096 B | 5,288 B | 0.34% |
| 2 | 8,192 B | 9,384 B | 0.61% |
| 4 | 16,384 B | 17,576 B | 1.14% |
| 8 | 32,768 B | 33,960 B | 2.21% |

Even 8 outstanding reads (34 KB) represents only 2.2% of BH L1 -- feasible for read-heavy
workloads that benefit from request pipelining.

### System-Wide Summary (32-chip Blackhole Galaxy)

| Resource | Per-Chip | 32-Chip Galaxy |
|----------|:--------:|:--------------:|
| Tensix L1 for GSRP metadata | 5,288 B x 140 cores = 723 KB | 22.6 MB |
| ERISC L1 for GSRP handler | 7.8 KB x 14 ERISCs = 109 KB | 3.4 MB |
| MUX cores (if 4 per chip) | 4 cores + 96 KB L1 | 128 cores + 3 MB |
| **Total memory overhead** | ~928 KB | **~29 MB** |
| **Total compute overhead** | 4 cores (2.9%) | 128 cores (2.9%) |

The 29 MB system-wide memory overhead is:

$$\frac{29\ \text{MB}}{6{,}720\ \text{MB (total Galaxy L1)}} \approx 0.43\%$$

This confirms that the GSRP's resource footprint is practical at scale.

---

## GSRP Implications

The dispatch integration and resource analysis confirms three practical findings:

1. **L1 overhead is not a blocking concern.** The GSRP metadata (1--5.3 KB depending on
   phase) is a fraction of a percent of total L1. The real trade-off is the globally-visible
   region partitioning, which is a workload-specific configuration choice -- not a fixed cost.

2. **The dispatch system already provides the right integration points.**
   `MeshWorkload::enqueue()` distributes per-program state, `MeshBuffer::create()` allocates
   consistent-address buffers, and `MeshCommandQueue` coordinates cross-device operations.
   GSRP initialization fits naturally within these existing patterns with a one-time cost of
   ~1.8 ms for a 32-chip Galaxy.

3. **Telemetry is achievable with minimal overhead.** The 48-byte per-core and 32-byte
   per-ERISC telemetry blocks provide per-transfer observability. Watcher and DPrint
   integration enables production debugging. The combined telemetry L1 cost (48 + 12 = 60
   bytes per core) is negligible.

The coexistence model with explicit CCL operations is essential: GSRP does not replace CCL
but complements it. Bulk collectives should continue to use explicit CCL for their superior
pipeline efficiency. GSRP targets the cases where explicit CCL is burdensome: point-to-point
transfers, irregular communication patterns, and latency-sensitive synchronization.

---

## Key Takeaways

- **GSRP initialization integrates through `MeshCommandQueue` with ~1.8 ms one-time setup
  cost** for a 32-chip Galaxy, amortized across the entire program lifetime. Per-kernel
  initialization reduces from ~45 lines of boilerplate to a single `gsrp_kernel_init()` call
  (~30 cycles).

- **The per-core GSRP L1 overhead is ~5.3 KB (0.35% of BH L1)**, bringing the total
  fabric + GSRP overhead to ~12.1 KB (0.79%). The existing fabric metadata (6.8 KB: routing
  ~2.6 KB + connections ~656 B + packet headers 3,456 B) is the dominant L1 consumer; GSRP
  adds only ~77% on top of that. The combined total is well within the 16 KB budget from
  Chapter 6, Section 1.

- **L1 partitioning into "globally visible" and "locally private" regions** provides a
  software-enforced safety boundary. The destination ERISC validates that incoming GSRP
  writes target the exportable region, silently dropping violations and incrementing a
  telemetry counter.

- **Explicit CCL and GSRP coexist via resource partitioning.** GSRP reserves routing plane 0
  for persistent connections; remaining planes are available for CCL operations. The VC-based
  priority mechanism ensures that latency-sensitive GSRP traffic is not delayed by bulk CCL
  transfers.

- **System-wide overhead for a 32-chip Galaxy is ~29 MB of L1 (~0.43%) and 128 MUX cores
  (2.9%)**, confirming that the GSRP is practical at the scale where multi-chip communication
  complexity is highest.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `tt_metal/api/tt-metalium/mesh_command_queue.hpp` | `MeshCommandQueue`, `enqueue_mesh_workload()`, `enqueue_write_mesh_buffer()`, `enqueue_record_event()`, `finish()` |
| `tt_metal/api/tt-metalium/mesh_device.hpp` | `MeshDevice`, `MeshDeviceView`, submesh creation |
| `tt_metal/api/tt-metalium/mesh_buffer.hpp` | `MeshBuffer::create()`, `REPLICATED`/`SHARDED` layouts, consistent-address allocation |
| `tt_metal/api/tt-metalium/experimental/sockets/mesh_socket.hpp` | `MeshSocket`, `SocketConfig` for structured cross-device communication |
| `tt_metal/impl/dispatch/dispatch_mem_map.hpp` | `DispatchMemMap`, `prefetch_q_size()`, `dispatch_buffer_base()`, command queue L1 addresses |
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` | `MEM_MAP_END`, `MEM_TENSIX_ROUTING_TABLE_BASE`, `MEM_TENSIX_FABRIC_CONNECTIONS_BASE`, `MEM_PACKET_HEADER_POOL_BASE`, `MEM_L1_SIZE` (1,536 KB), `PACKET_HEADER_MAX_SIZE` (144 B), `NUM_PACKET_HEADERS` (24) |
| `tt_metal/api/tt-metalium/experimental/fabric/fabric_telemetry.hpp` | `FabricTelemetrySnapshot`, `FabricTelemetryBandwidthCounters`, `FabricTelemetryRouterState` |
| `tt_metal/api/tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp` | Host-side telemetry reading infrastructure |
| `tt_metal/fabric/fabric_telemetry_converter.hpp` | `unpack_bandwidth()`, `unpack_static_info_from_hal()` |
| `tt_metal/fabric/fabric_tensix_builder.hpp` | `FabricTensixDatamoverConfig`, MUX core configuration, `FabricTensixConfig::MUX` |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | `WATCHER_ENABLED` conditional compilation, debug validation paths |

---

**Previous:** [`03_deadlock_avoidance_strategies.md`](./03_deadlock_avoidance_strategies.md) | **Next:** [Chapter 8 -- Performance Analysis, Hardware Gap Assessment, and Roadmap](../ch08_performance_and_roadmap/index.md)
