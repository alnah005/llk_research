# 6.5 Address Translation Layer

## Context

Section 6.4 selected the Fat Fabric API as the Phase 1 implementation target and identified
the address translation layer as its performance-critical inner loop. When a kernel calls
`gsrp_write(dest, src, size)`, the runtime must resolve the 64-bit `GlobalAddress` into a
concrete fabric destination and route -- all within the overhead budget of $< 5\%$ throughput
loss (Section 6.1). This section designs the translation machinery in detail, evaluating
three implementation options (host-computed L1 tables, ERISC firmware translation, and
hardware-assisted translation), introducing route caching as the key optimization, and
specifying the read-path protocol that enables bidirectional GSRP access.

The reader should be familiar with the flat 64-bit address encoding from Section 6.2, the
existing `routing_l1_info_t` data structures from Chapter 3 (Section 4), and the
`fabric_set_unicast_route` function from Chapter 4 (Section 1).

---

## 6.5.1 What Translation Must Accomplish

Given a `GlobalAddress`, the translation layer must produce five outputs:

| Output | Type | Purpose |
|--------|------|---------|
| `dst_mesh_id` | `uint16_t` | Target mesh (for inter-mesh routing) |
| `dst_dev_id` | `uint16_t` | Target device within mesh |
| `noc_xy` | `uint32_t` | NOC coordinates on target chip |
| `l1_offset` | `uint32_t` | Byte offset in target core's L1 |
| `route` | `compressed_route_2d_t` | Pre-computed route from source to destination |

The first four are directly extractable from the `GlobalAddress` encoding (Section 6.2,
decomposition cost: ~3 cycles via shifts and masks). The fifth -- the route -- requires a
lookup. The locality dispatch table (Section 6.2.3) then selects the transport; this
section focuses on the route-resolution cost for the `REMOTE_CHIP`/`REMOTE_MESH` cases.

### Cost Breakdown (Uncached, Remote Path)

| Translation step | Operation | Cycles (estimated) |
|-----------------|-----------|:-----------------:|
| Address decomposition + locality check | Per Section 6.2 | ~4 |
| Route table lookup | 1--2 L1 reads | ~5--10 |
| Connection pool lookup | 1 L1 read + 1 increment | ~3--5 |
| Header population | 6--8 stores | ~6--8 |
| **Total translation** | | **~18--27 cycles** |

At ~1 GHz Blackhole core clock, 27 cycles is 27 ns. For a 4 KB transfer (~328 ns on the
fabric at 12.5 GB/s), the overhead is $\frac{27}{328} \approx 8.2\%$, exceeding the $< 5\%$
target. This motivates **route caching** (Section 6.5.5), which reduces the lookup to
~13 cycles and brings 4 KB overhead to ~4.0%.

---

## 6.5.2 Option A: Host-Computed L1 Translation Tables

### Design

The host pre-computes the full translation table during program initialization and writes it
to each Tensix core's L1. The device-side translation is a simple indexed load.

**Table structure (8 bytes per entry):**

```cpp
struct gsrp_chip_route {
    uint32_t route_word;       // Packed per-hop routing bits (Ch 3 format)
    uint8_t  hop_count;        // Hops to destination
    uint8_t  header_type;      // 0 = LowLatency, 1 = HybridMesh
    uint8_t  exit_erisc_x;     // Local ERISC core for injection
    uint8_t  exit_erisc_y;     // (selects which Ethernet link)
};  // 8 bytes per entry
```

**Host-side construction (using ControlPlane data):**

```cpp
std::vector<gsrp_chip_route> build_gsrp_route_table(
    const ControlPlane& cp,
    chip_id_t local_chip, MeshId local_mesh)
{
    auto all_nodes = cp.get_all_fabric_nodes();
    std::vector<gsrp_chip_route> table(MAX_CHIPS_PER_MESH);

    for (auto& node : all_nodes) {
        if (node.mesh_id != local_mesh) continue;
        auto first_hop = cp.get_first_hop_direction(
            local_chip, node.chip_id, local_mesh, node.mesh_id);
        table[node.chip_id] = {
            .route_word = cp.get_compressed_route(local_chip, node.chip_id),
            .hop_count = cp.get_hop_count(local_chip, node.chip_id),
            .header_type = (cp.get_hop_count(local_chip, node.chip_id) > 2) ? 1 : 0,
            .exit_erisc_x = select_erisc_for_direction(first_hop).x,
            .exit_erisc_y = select_erisc_for_direction(first_hop).y
        };
    }
    return table;
}
```

### Table Sizing

| Configuration | Entries | Table size |
|--------------|:-------:|:----------:|
| 8-chip BH (T3K) | 7 | 56 B |
| 32-chip BH Galaxy | 31 | 248 B |
| 64-chip (2 Galaxies) | 63 | 504 B |

### Device-Side Lookup

```cpp
FORCE_INLINE gsrp_chip_route gsrp_route_lookup(GlobalAddress addr) {
    uint8_t chip = addr.chip_id();
    return ((volatile gsrp_chip_route*)GSRP_ROUTE_TABLE_BASE)[chip];
}
```

Lookup cost: index computation (1 cycle) + L1 load (3--4 cycles) = **~5 cycles**.

### Epoch Transition

At epoch boundaries, the host updates the tables. For 31 entries $\times$ 8 B at 12.5 GB/s:

$$T_{\text{table\_update}} = \frac{248\ \text{B}}{12.5\ \text{GB/s}} \approx 0.02\ \mu\text{s per core}$$

For 140 cores per chip (parallelized across NOC): ~3 $\mu$s per chip -- negligible.

### Address-Based Routing Extension

The existing `fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id)` requires the
caller to supply chip-level identifiers extracted manually. GSRP extends this with an
address-based variant that accepts a `GlobalAddress` directly:

```cpp
FORCE_INLINE void fabric_set_unicast_route_from_addr(
    HybridMeshPacketHeader* header,
    GlobalAddress addr)
{
    gsrp_chip_route route = gsrp_route_lookup(addr);
    header->routing.value = route.route_word;
    header->hop_count = route.hop_count;
}
```

This wrapper is the single integration point between the GSRP address space and the
existing mesh API. It composes `gsrp_route_lookup` (above) with the standard header
population, eliminating the need for kernel code to manually decompose addresses into
`dst_dev_id`/`dst_mesh_id` pairs.

---

## 6.5.3 Option B: ERISC Firmware Translation

### Design

Instead of distributing translation tables to every Tensix core, centralize translation on
ERISC. Tensix cores send lightweight requests to ERISC, which resolves routes using its own
routing database and issues fabric transfers on behalf of the Tensix core.

```
Tensix Core                     ERISC
    |                              |
    |-- [global_addr, src, size] ->|  (NOC write to ERISC trap buffer)
    |                              |-- translate global_addr
    |                              |-- resolve route
    |                              |-- issue fabric_unicast_noc_unicast_write
    |                              |
    |<---- [completion signal] ----|  (atomic inc on Tensix semaphore)
```

### Overhead Analysis

| Phase | Latency |
|-------|:-------:|
| Tensix $\to$ ERISC (NOC write) | ~100--200 ns |
| ERISC poll + route computation | ~500--2,500 ns |
| ERISC $\to$ Fabric (Ethernet send) | Same as direct path |
| **Total overhead vs. direct** | **~600--2,700 ns** |

For a 4 KB transfer (~328 ns fabric time), this is 183--823% overhead -- unacceptable.
For a 64 KB transfer (~5.2 $\mu$s), it is 12--52% -- still high.

### Queuing Model (M/D/1)

Modeling the ERISC as an M/D/1 queue with 140 Tensix cores sharing 14 active ERISCs
(Chapter 3, Section 1):

- Arrival rate per ERISC: $\lambda = 14 \times 10^6 / 14 \approx 1.0 \times 10^6$ req/s
- Service rate: $\mu \approx 1.5 \times 10^6$ req/s
- Utilization: $\rho = 0.67$

Average response time: $T \approx 1.5\ \mu\text{s}$ at moderate load; $\sim$4.5 $\mu$s
under heavy load ($\rho \to 0.9$).

> **Key Insight:** ERISC firmware translation is viable only for large transfers ($\geq$
> 32 KB) where the extra NOC round-trip is amortized. For the common case, Tensix-side
> L1 tables (Option A) are mandatory.

---

## 6.5.4 Option C: Hardware-Assisted Translation (Next-Gen Silicon)

A hardware translation unit on each Tensix core intercepts NOC transactions targeting global
addresses and performs translation in silicon.

### Proposed TLB Design

| Parameter | Value |
|-----------|:-----:|
| Entries | 8--16 (fully associative) |
| Entry size | 32 bytes |
| Total TLB area | 256--512 bytes |
| Lookup latency | 1--2 cycles (hit) |
| Miss penalty | ~15 ns (L1 page table walk) |
| Estimated hit rate | 95--99% (ML workloads have high locality) |

ML workloads exhibit strong temporal locality in cross-chip access patterns:

| Workload | Typical remote destinations | TLB entries needed |
|----------|:--------------------------:|:------------------:|
| Data parallelism (ring) | 2 | 2 |
| Tensor parallelism | 4--8 | 4--8 |
| Pipeline parallelism | 1--2 | 2 |
| MoE (expert routing) | 2--16 | 8--16 |

Even the most demanding pattern fits in a 16-entry TLB.

### Blackhole Extension Points

BH has `noc_address_translation_table` registers for coordinate virtualization and
`NOC_SEC_FENCE_RANGE` registers for address validation. Neither supports cross-chip
translation today, but they establish a precedent -- the NOC interface already has
translation mechanisms. Extending them for GSRP would be an incremental silicon change
for the Quasar generation.

**Verdict:** Not feasible on WH or BH. Included to inform next-gen silicon design.

---

## 6.5.5 Route Caching

Route caching is the single most impactful optimization. In ML workloads, a kernel
typically communicates with 2--8 remote destinations for the entire program. Computing the
route once and reusing it eliminates the routing table lookup from the critical path.

### Per-Core Route Cache Design

```
Route Cache (per Tensix core):
+----------+---------+-------------------+-------------------+
| dst_chip | valid   | noc_unicast_hdr   | compressed_route  |
| 8 bits   | 1 bit   | 64 bits           | 40 bytes          |
+----------+---------+-------------------+-------------------+
Entry size: 56 bytes (padded to 64 bytes)
```

| Parameter | Value |
|-----------|:-----:|
| Entries | 8 |
| Total size | 512 bytes |
| Hash-indexed lookup | ~2 cycles (3-bit hash of dst_chip) |
| Fill cost (on miss) | ~10 cycles (routing table lookup) |

### Cached Translation Cost

| Step | Cycles |
|------|:------:|
| Address decomposition | 3 |
| Locality check | 1 |
| Route cache lookup (hash-indexed) | 2 |
| Connection pool selection | 3 |
| Header fill (from cached values) | 4 |
| **Total (cache hit)** | **~13 cycles** |

### Overhead with Cache

| Transfer size | Translation (cycles) | Fabric time (ns) | Overhead |
|:------------:|:-------------------:|:-----------------:|:--------:|
| 256 B | 13 | 20 | 65% (batch required) |
| 1 KB | 13 | 82 | 16% |
| 4 KB | 13 | 328 | **4.0%** |
| 16 KB | 13 | 1,311 | 1.0% |
| 64 KB | 13 | 5,242 | 0.25% |

For transfers $\geq$ 4 KB, the GSRP meets the $< 5\%$ throughput target. For smaller
transfers, batching is required (see Section 6.5.7).

### Cache Invalidation

Route cache entries are invalidated when: (1) the program ends (per-program state),
(2) the fabric is reconfigured (ControlPlane signals flush), or (3) the application
explicitly calls `gsrp_invalidate_route_cache()`. In practice, invalidation is rare --
routing tables are static for the duration of a training job.

---

## 6.5.6 Read Path Translation

Write translation is straightforward: the Tensix core translates, constructs a packet, and
injects. Read translation requires a request-response protocol.

### Read Request Flow

```
1. Requester (Tensix core, chip 0):
   - Translates global_addr to (dst_chip, core_xy, l1_offset)
   - Constructs READ_REQUEST packet containing:
     * Target global address
     * Requester's (chip_id, core_xy, l1_offset) for the response
     * Read size
   - Sends READ_REQUEST via fabric

2. Responder (ERISC, destination chip 3):
   - Receives READ_REQUEST
   - Reads data from target core's L1 via NOC
   - Constructs READ_RESPONSE packet containing the data
   - Sends READ_RESPONSE back to requester via reverse fabric route

3. Receiver (ERISC, requester's chip 0):
   - Receives READ_RESPONSE
   - Writes data to requester's L1 via NOC
   - Increments requester's read-completion counter via NOC atomic
```

### ERISC Read-Handler Firmware (Pseudocode)

```cpp
void handle_gsrp_read_request(const GsrpReadRequestHeader* req) {
    // 1. Read data from local core's L1
    uint64_t src_noc = get_noc_addr(
        req->target_core_x, req->target_core_y, req->target_l1_offset);
    noc_async_read(src_noc, local_read_buffer, req->size);
    noc_async_read_barrier();

    // 2. Construct response packet
    HybridMeshPacketHeader response;
    fabric_set_unicast_route(&response, req->requester_chip_id,
                             req->requester_mesh_id);
    NocUnicastCommandHeader noc_cmd;
    noc_cmd.noc_x = req->requester_core_x;
    noc_cmd.noc_y = req->requester_core_y;
    noc_cmd.addr  = req->requester_l1_offset;
    noc_cmd.size  = req->size;
    response.noc_unicast_command_header = noc_cmd;

    // 3. Send response data
    edm_channel_send(local_read_buffer, req->size, &response);

    // 4. Send completion signal
    NocUnicastAtomicIncCommandHeader ack;
    ack.noc_x = req->requester_core_x;
    ack.noc_y = req->requester_core_y;
    ack.addr  = req->requester_completion_counter;
    ack.val   = 1;
    // ... send via fabric ...
}
```

### Read Latency Model

For a single-hop read of $N$ bytes:

| Component | Estimate |
|-----------|:--------:|
| Translation (Option A) | ~6 cycles (~6 ns) |
| Request fabric (1 hop) | 5--10 $\mu$s |
| ERISC processing | ~0.5 $\mu$s |
| NOC read on remote chip | ~0.1 $\mu$s |
| Response fabric (1 hop) | 5--10 $\mu$s |
| NOC write to requester L1 | ~0.1 $\mu$s |
| **Total single-hop read** | **10--21 $\mu$s** |

Compare with local L1 read: ~4 ns. The remote/local ratio is ~2,500:1, reinforcing the
NUMA model: remote reads should be avoided when possible. The GSRP makes them **possible**
but does not make them **cheap**.

---

## 6.5.7 Batching and Coalescing

Small remote writes ($<$ 4 KB) are common in synchronization-heavy patterns. The GSRP
provides a batching mechanism that amortizes translation cost:

```cpp
void gsrp_write_batch_begin(GlobalAddress base);
void gsrp_write_batch_add(uint32_t offset, uint32_t src, uint32_t size);
void gsrp_write_batch_commit();  // Coalesces and sends as one packet
```

| Batch size | Individual overhead | Batched overhead |
|:----------:|:------------------:|:----------------:|
| 4 writes $\times$ 64 B (256 B) | $4 \times 65\%$ = catastrophic | 4.0% |
| 8 writes $\times$ 128 B (1 KB) | $8 \times 16\%$ = catastrophic | 1.0% |

---

## 6.5.8 Recommended Configuration (Phase 1, Blackhole)

| Component | Choice | Size |
|-----------|--------|:----:|
| Chip-level translation table | Option A (host-computed) | 248 B |
| Route cache | 8-entry hash-indexed | 512 B |
| Connection pool | 16-entry | 128 B |
| Segment registers | 8 registers | 128 B |
| **Total per-Tensix GSRP metadata** | | **1,016 B** |

The 1,016-byte total is well within the 16 KB budget -- leaving ample room for Phase 2
additions (read response buffers, congestion counters, extended route caches).

---

## 6.5.9 Translation Option Comparison

| Option | Per-access cycles | L1 (Tensix) | L1 (ERISC) | FW changes | HW changes |
|--------|:-----------------:|:-----------:|:----------:|:----------:|:----------:|
| **A: Host tables** | 18--27 (uncached) | 248 B--3 KB | None | None | None |
| **A + Route cache** | **13** | 760 B--3.5 KB | None | None | None |
| **B: ERISC firmware** | 600--2,700 ns | ~32 B | 2--4 KB | Moderate | None |
| **A+B hybrid** | 13 (small) / 0.6 $\mu$s (large) | 760 B--3.5 KB | 2--4 KB | Moderate | None |
| **C: Hardware TLB** | 1--2 (hit) / 15 ns (miss) | 0 B | None | Minimal | **Extensive** |

---

## Key Takeaways

- **Host-computed L1 translation tables (Option A) with an 8-entry route cache provide the
  best balance** of per-access overhead (~13 cycles), L1 cost (~1,016 bytes total per core),
  and implementation complexity (no firmware or hardware changes for the write path).

- **Route caching is the single most impactful optimization**: it reduces translation from
  18--27 cycles to ~13 cycles, bringing 4 KB transfer overhead from 8.2% to 4.0% and
  meeting the $< 5\%$ throughput target without batching.

- **The read path requires ERISC firmware extension** for request-response round trips, with
  single-hop latency of 10--21 $\mu$s (~2,500$\times$ local access) -- reinforcing that the
  GSRP provides a NUMA-aware abstraction, not transparent shared memory.

## GSRP Implications

The address translation layer completes the design of the Fat Fabric API's critical path.
A `gsrp_write(dest, src, size)` call now has a fully specified implementation: decompose the
address (3 cycles), check locality (1 cycle), look up the cached route (2 cycles), select a
connection (3 cycles), populate the header (4 cycles), and issue the two-phase send via the
existing EDM protocol. The 1,016-byte per-core footprint leaves 15 KB of headroom within the
16 KB budget for Phase 2 additions. The ERISC read-request handler is the single most
important firmware deliverable for Phase 2: it unlocks read access across the global address
space, transforming the GSRP from a write-only mechanism to a full bidirectional memory
abstraction.

## Source Code References

| File | Relevance |
|------|-----------|
| `tt_metal/fabric/hw/inc/mesh/api.h` | `fabric_set_unicast_route`, `fabric_unicast_noc_unicast_write` |
| `tt_metal/fabric/routing_table_generator.cpp` | BFS routing, `routing_l1_info_t`, `compressed_route_2d_t` |
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | `compressed_route_2d_t`, `routing_table_t` |
| `tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp` | Per-worker connection management |
| `tt_metal/fabric/control_plane.cpp` | Routing table generation and distribution |
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` | L1 addresses of routing data |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | ERISC router -- target for read-handler |

---

**Previous:** [`04_candidate_architectures.md`](./04_candidate_architectures.md) | **Next:** [Chapter 7 -- Runtime Infrastructure for Transparent Cross-Chip Data Movement](../ch07_runtime_infrastructure/index.md)
