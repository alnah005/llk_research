# 7.3 Deadlock Avoidance Strategies

## Context

Section 7.2 addressed flow control and congestion -- the performance side of multi-hop data
movement. This section addresses the **correctness** side: deadlock avoidance. The reader
should be familiar with the existing dateline-based virtual channel separation and bubble
flow control for ring/torus topologies (Chapter 3, Section 6), the BSP epoch model from
Graphcore as a deadlock-free synchronization framework (Chapter 5, Section 3), and the GSRP
consistency model with epoch barriers (Chapter 6, Section 3). The current fabric was designed
for **unidirectional, store-and-forward writes** -- the EDM routes packets from source to
destination in one direction at a time. GSRP introduces **bidirectional, request-response
traffic** that creates circular dependency risks not covered by the existing mechanisms. This
section catalogs those risks, evaluates four mitigation strategies, and proposes a layered
composite solution combining virtual channel partitioning, response-priority scheduling,
epoch-based barriers, and buffer reservation.

---

## 7.3.1 Current Deadlock Avoidance Mechanisms

The existing deadlock avoidance mechanisms are documented in Chapter 3, Section 6:

- **Dateline-based VC separation** breaks ring cycles by switching from VC0 to VC1 at a
  designated dateline link. The `need_deadlock_avoidance_support(direction)` query identifies
  directions requiring this treatment.
- **Bubble flow control** reserves one buffer slot per hop as an alternative to VC separation
  ($N_{\text{usable}} = N_{\text{total}} - 1$).
- **Dimension-ordered routing (DOR)** guarantees deadlock freedom for non-wrapping 2D meshes
  by enforcing NS-first-then-EW ordering.
- **Routing plane separation** provides physical isolation (WH: 4 planes, BH: 2 planes).

The key property for GSRP analysis: **non-wrapping 2D meshes (`FABRIC_2D`) are inherently
deadlock-free under DOR**, while ring/torus topologies require dateline VC separation or
bubble flow control.

---

## 7.3.2 New Deadlock Risks Under GSRP

GSRP introduces three categories of deadlock risk not present in the current unidirectional
model.

### Risk 1: Request-Response Circular Dependency

The most dangerous new deadlock scenario arises from bidirectional reads. Cross-chip reads
(Section 7.1.6) require a request packet and a response packet. If both directions share
the same channel resources, a circular dependency can form:

```
Chip A                        Chip B
+--------+                    +--------+
| Core 0 |-- READ_REQ ------>| ERISC  |
|        |                    |  |     |
|        |<-- [blocked] ----  |  | issues NOC read
|        |                    |  | builds response
| ERISC  |<-- READ_REQ ------| Core 3 |
|  |     |                    |        |
|  | issues NOC read          |        |
|  | builds response          |        |
|  +-- [blocked] ----------->|        |
+--------+                    +--------+

Circular wait:
  A's ERISC needs EDM buffer to send response to B
  B's ERISC needs EDM buffer to send response to A
  Both EDM buffers are full with the other's request
```

This is a classic request-response deadlock. In the current fabric, it cannot occur because
there are no cross-chip reads -- all traffic is posted writes.

### Risk 2: Multi-Hop Transitive Dependencies

In a multi-chip mesh, transitive dependencies can form through shared intermediate buffers:

```
Chip 0 writes to Chip 1 via Chip 3 (path: 0->3->1)
Chip 1 writes to Chip 2 via Chip 3 (path: 1->3->2)
Chip 2 writes to Chip 0 via Chip 3 (path: 2->3->0)

If Chip 3's forwarding buffers are shared across all paths:
  0->3 buffer full (waiting for 3->1 to drain)
  1->3 buffer full (waiting for 3->2 to drain)
  2->3 buffer full (waiting for 3->0 to drain)
  Circular wait through shared buffers on Chip 3.
```

This scenario is impossible under dimension-ordered routing in a non-wrapping mesh
(Chapter 3, Section 3.5) because the turn constraint prevents cycles. However, for torus
topologies or optimized non-DOR routing, it requires explicit avoidance.

### Risk 3: Mixed GSRP + CCL Deadlock

GSRP writes sharing channels with explicit CCL traffic can create inter-class deadlocks:

```
CCL all-gather (ring): ... -> Chip 2 -> Chip 3 -> Chip 4 -> ...
GSRP write: Chip 4 -> Chip 3 -> Chip 2

If the CCL ring and GSRP write share the same EDM channel buffers,
the GSRP write competes with the CCL forward for buffer space.
A cycle can form through the shared buffer pool.
```

---

## 7.3.3 Solution 1: Virtual Channel Partitioning by Traffic Class

### Design

Assign separate virtual channels to different traffic classes so they cannot form cross-class
dependencies. Each VC has independent buffer slots and credit tracking:

```
Ideal 4-VC Scheme (Wormhole with 4 routing planes):
  VC 0: Forward writes (lower chip_id -> higher chip_id)
  VC 1: Reverse writes (higher chip_id -> lower chip_id)
  VC 2: Read requests
  VC 3: Read responses

Each VC has its own buffer pool. No cross-VC contention.
All three risk classes eliminated.
```

### Practical 2-VC Scheme (Blackhole)

The current hardware supports only 2 VCs (`MAX_NUM_VCS = 2`). The 4-VC ideal must be
compressed:

| VC | Traffic Class | Buffer Slots | Purpose |
|:--:|:-------------:|:------------:|---------|
| 0 | CCL + GSRP writes + read requests | 4 | Standard push traffic + requests |
| 1 | Read responses + credit updates | 4 | Terminal traffic (generates no further deps) |

The separation principle: VC 1 carries only **terminal traffic** -- packets that generate
no further network requests. Read responses are terminal (the response is delivered to the
requester and consumed). This means VC 1 can never be blocked by a dependency on VC 0,
breaking the circular dependency.

### Buffer Cost

With 8 total buffer slots per channel, splitting into 2 VCs of 4 each halves the per-VC
depth. For power-of-2 optimization (Chapter 3, Section 6), 4 slots per VC is ideal
(`wrap_increment` uses bitwise AND with mask 0x3).

Throughput verification: at 12.5 GB/s Ethernet bandwidth and ~1 $\mu$s per-hop latency:

$$BDP = 12.5\ \text{GB/s} \times 1\ \mu\text{s} = 12.5\ \text{KB}$$

Four 4 KB slots (16 KB) exceed the BDP, so 4 slots per VC are sufficient for full
throughput per-VC.

---

## 7.3.4 Response-Priority Scheduling: The Key 2-VC Mechanism

On Blackhole with only 2 VCs, read requests and read responses may share a VC (when all
traffic must be compressed into 2 VCs). The critical mechanism that makes this safe is
**response-priority scheduling**:

```cpp
// Proposed scheduling extension for fabric_erisc_router.cpp
// In the receiver channel processing loop:
FORCE_INLINE bool is_response_packet(const PacketHeader* hdr) {
    return hdr->noc_send_type == NOC_UNICAST_WRITE &&
           hdr->flags & GSRP_READ_RESPONSE_FLAG;
}

// Process responses before requests when both are pending:
if (has_pending_response && has_pending_request) {
    process_response_first();  // Responses drain without generating new traffic
}
```

### Why This Works

A read response is a **terminal operation**: once delivered to the requester's L1 (via NOC
write), it is consumed. It generates no further network traffic. A read request, by contrast,
generates a response -- creating a new dependency.

By prioritizing responses over requests within the EDM scheduler:

1. Responses always drain, even under heavy load.
2. Requests may be delayed, but since responses always make progress, every request
   eventually receives its response.
3. The circular wait from Risk 1 is broken: A's response to B is prioritized over B's new
   request, so A's response buffer drains, freeing capacity for B's response.

> **Key Insight:** Response-priority scheduling within a shared request/response VC is the
> critical mechanism that enables deadlock-free bidirectional GSRP on 2-VC hardware
> (Blackhole). Without it, request-response cycles can form. With it, responses always drain
> because they are terminal operations that generate no further dependencies.

---

## 7.3.5 Solution 2: Epoch-Based Barrier Synchronization

### The Relaxed BSP Model

Drawing on Graphcore's BSP model (Chapter 5, Section 3), epoch barriers provide structural
deadlock freedom by ensuring that all cross-chip traffic from epoch $k$ completes before
epoch $k+1$ begins. Unlike strict BSP (which prohibits all communication during compute),
the GSRP relaxed BSP model allows:

| Property | Strict BSP (Graphcore) | Relaxed BSP (GSRP) |
|----------|:---------------------:|:------------------:|
| Compute during exchange | No | Yes |
| Local NOC during exchange | No | Yes |
| Remote writes during compute | No | Yes (GSRP writes) |
| Remote reads during compute | No | Conditional (if VC-separated) |
| Barrier frequency | Every superstep | At epoch boundaries only |

```
Epoch N                    Barrier          Epoch N+1
+-----------------------+  | |  +-----------------------+
| Compute + GSRP writes |  | |  | Compute + GSRP writes |
| GSRP reads begin      |  | |  | Read prior epoch data |
| All writes/reads must |  | |  |                       |
| complete before barrier|  | |  |                       |
+-----------------------+  | |  +-----------------------+
                           ^ ^
                     All-chip sync
                     (GlobalSemaphore atomic incs)
```

### Epoch Barrier Implementation

Using the existing `GlobalSemaphore` and `NocUnicastAtomicIncCommandHeader` (Chapter 2,
Section 4):

```cpp
// GSRP epoch barrier (per core)
FORCE_INLINE void gsrp_epoch_barrier() {
    // Phase 1: Flush all pending GSRP writes
    gsrp_write_barrier();  // Wait for all local write credits to return

    // Phase 2: Signal local completion to barrier coordinator
    noc_semaphore_inc(gsrp_barrier_semaphore_addr, 1);

    // Phase 3: Wait for all chips to reach barrier
    noc_semaphore_wait(gsrp_barrier_semaphore_addr, expected_count);

    // Phase 4: Reset barrier for next epoch
    noc_semaphore_set(gsrp_barrier_semaphore_addr, 0);
}
```

For cross-chip synchronization, the barrier coordinator is an ERISC core that aggregates
per-chip completion signals:

```
Per-chip completion:
  140 Tensix cores signal local ERISC -> ERISC aggregates -> sends 1 fabric
                                          completion msg per chip

Cross-chip synchronization:
  All N chips send completion to barrier master (e.g., Chip 0 ERISC)
  Master broadcasts "barrier done" to all chips
  Total: 2 * (N-1) fabric messages + local NOC broadcasts
```

### Barrier Cost

| Configuration | Estimated Barrier Cost | Compute Phase (typical) | Overhead |
|:-------------:|:---------------------:|:----------------------:|:--------:|
| 8-chip BH | ~10 $\mu$s | ~500 $\mu$s | 2.0% |
| 32-chip BH Galaxy | ~30 $\mu$s | ~500 $\mu$s | 6.0% |
| 32-chip BH Galaxy | ~30 $\mu$s | ~2 ms | 1.5% |

Assumptions: tree-based barrier reduction (ERISC aggregation per chip, then
chip-to-chip broadcast), ~5 $\mu$s per fabric hop. For training steps of ~10 ms, the 30
$\mu$s barrier represents 0.3% overhead -- negligible. For inference with sub-ms latency
targets, barriers should be placed only at natural synchronization points (layer boundaries,
attention boundaries) to amortize the cost.

**Epoch placement.** Natural epoch boundaries align with TTNN operation boundaries (matmul,
attention, MLP layers). A typical transformer layer has 4--8 operations, yielding 4--8
epochs per layer. At 200--500 $\mu$s per operation, the 10--30 $\mu$s barrier overhead
represents 2--6%.

> **Key Insight:** The epoch barrier is not a performance mechanism -- it is a **safety
> mechanism**. It guarantees that any transitive dependency chains that might form during
> an epoch are broken at the barrier. The cost is proportional to the drain time (dependent
> on the longest outstanding GSRP transfer) plus the synchronization time (dependent on
> chip count). The relaxed BSP model minimizes this cost by allowing overlap within epochs.

---

## 7.3.6 Solution 3: Request-Reply Channel Separation

### Design

Dedicate separate physical resources (routing planes) for request and reply traffic:

```
Routing Plane 0: Requests (GSRP writes + read requests)
  +----> Direction: Source -> Destination

Routing Plane 1: Replies (read responses + credit updates)
  <----+ Direction: Destination -> Source
```

Since requests and replies use separate routing planes with independent buffer resources,
they cannot contend for the same slots. This is the strongest guarantee against
request-response deadlocks.

### Routing Plane Availability

| Architecture | Total Planes | Needed for CCL | Available for GSRP Req/Rsp Split |
|:------------:|:------------:|:--------------:|:--------------------------------:|
| Wormhole | 4 | 2 | 2 (1 request + 1 reply) |
| Blackhole | 2 | 1 | 1 (shared -- use VC-based split instead) |

On Wormhole with 4 planes, dedicating 2 to requests and 2 to replies is natural. On
Blackhole with only 2 planes, dedicating 1 to each leaves no room for CCL separation.
The Blackhole workaround is intra-plane VC separation (Section 7.3.3) combined with
response-priority scheduling (Section 7.3.4).

---

## 7.3.7 Extending `need_deadlock_avoidance_support`

The existing `FabricContext::need_deadlock_avoidance_support(direction)` considers only
topology-based deadlock risks (ring/torus wrap-around). GSRP introduces traffic-pattern-based
risks that are independent of topology.

### Proposed Extension

```cpp
// Proposed extension to FabricContext
// tt_metal/fabric/fabric_context.hpp

enum class DeadlockAvoidanceReason : uint8_t {
    TOPOLOGY_WRAP_AROUND = 0x01,    // Existing: ring/torus wrap
    GSRP_BIDIRECTIONAL   = 0x02,    // New: GSRP read requests create cycles
    GSRP_MIXED_TRAFFIC   = 0x04,    // New: GSRP + CCL sharing channels
};

struct DeadlockAvoidanceInfo {
    bool needed;
    DeadlockAvoidanceReason reasons;
    uint8_t recommended_vc_count;    // Minimum VCs for deadlock freedom
    bool epoch_barrier_recommended;  // True if VC count is insufficient
};

DeadlockAvoidanceInfo need_deadlock_avoidance_support_v2(
    eth_chan_directions direction,
    bool gsrp_enabled,
    bool gsrp_reads_enabled) const;
```

### Decision Logic

```
need_deadlock_avoidance_support_v2(direction, gsrp, gsrp_reads):
  reasons = 0
  vc_count = 1
  epoch_needed = false

  if (topology is ring/torus for this direction):
    reasons |= TOPOLOGY_WRAP_AROUND
    vc_count = max(vc_count, 2)

  if (gsrp_enabled):
    if (gsrp_reads_enabled):
      reasons |= GSRP_BIDIRECTIONAL
      vc_count = max(vc_count, 2)      // Separate requests from responses
    if (ccl_traffic_active):
      reasons |= GSRP_MIXED_TRAFFIC
      if (max_available_vcs < 3):       // BH: only 2 VCs
        epoch_needed = true             // Need epoch barriers as supplement

  return {reasons != 0, reasons, vc_count, epoch_needed}
```

For `FABRIC_2D` (non-wrapping mesh), the current function returns `false` (no avoidance
needed for unidirectional traffic). With GSRP bidirectional traffic enabled, it would return
`{true, GSRP_BIDIRECTIONAL, 2, false}` -- indicating that VC partitioning is sufficient.

---

## 7.3.8 Buffer Reservation Strategies

Buffer reservation prevents a single traffic class from consuming all available buffer slots,
which would cause starvation-based deadlocks.

### Strategy A: Per-Destination Credit Pools

Reserve a minimum number of buffer slots per destination, guaranteeing that at least one
packet to each destination can always be accepted:

| Pool strategy | Slots per dest | Total reserved | Throughput loss |
|:-------------:|:--------------:|:--------------:|:---------------:|
| 1 reserved + 4 shared | 1 | 4 (for 4 dests) | ~0% (shared absorbs) |
| 2 reserved + 2 shared | 2 | 8 (for 4 dests) | ~25% (shared reduced) |
| Fully reserved | 2 | 8 (for 4 dests) | ~50% (no sharing) |

**Advantage:** Simple, predictable. **Disadvantage:** Wastes capacity when traffic is
asymmetric (one destination active, three idle).

### Strategy B: Shared Pool with Admission Control

All buffer slots are shared, but an admission controller limits how many slots any single
traffic class or destination can occupy:

```cpp
FORCE_INLINE bool admit_gsrp_packet(uint8_t traffic_class, uint8_t free_slots) {
    if (traffic_class == RESPONSE) {
        return free_slots > 0;  // Responses always admitted if any slot free
    } else {
        return free_slots > RESPONSE_RESERVATION;  // Requests need headroom
    }
}
```

With `RESPONSE_RESERVATION = 2` and `NUM_BUFFERS = 8`:
- Requests are admitted when $\text{free} > 2$ (at most 6 request slots used).
- Responses are admitted when $\text{free} > 0$ (always, unless truly full).

This guarantees that at least 2 slots are available for responses at any time, breaking
the request-response deadlock.

### Comparison

| Strategy | Deadlock-free? | Max throughput | Complexity | ERISC overhead |
|----------|:--------------:|:--------------:|:----------:|:--------------:|
| Per-dest credit pools | Yes (with VCs) | 75--100% | Moderate | ~5 cycles/pkt |
| Shared + admission | Yes | 75--85% | Low | ~3 cycles/pkt |
| VC separation only | Yes | 100% per VC | Moderate | ~2 cycles/pkt |
| Epoch barriers only | Yes (at barrier) | 100% between barriers | Low | 0 cycles/pkt |

> **Recommendation:** Use shared pool with admission control for Phase 1 (write-only, fewer
> deadlock constraints). Transition to shared + admission combined with VC separation in
> Phase 2 when request-reply traffic requires stricter isolation.

---

## 7.3.9 Combined Layered Deadlock Avoidance Architecture

No single strategy optimally addresses all risks on all hardware. The recommended approach
combines strategies in a layered fashion:

```
+================================================================+
| Layer 3: Epoch Barriers (safety net)                           |
|   - Guarantees all traffic drains at epoch boundaries          |
|   - Catches any deadlock not prevented by Layers 0-2           |
|   - Cost: ~10-30 us per barrier (amortized over epoch)         |
+================================================================+
| Layer 2: Admission Control (per-packet)                        |
|   - Reserves buffer slots for response traffic                 |
|   - Prevents buffer starvation under heavy load                |
|   - Cost: ~3 cycles per packet on ERISC                        |
+================================================================+
| Layer 1: VC / Routing Plane Separation + Response Priority     |
|   - Write/CCL traffic on VC0, Response traffic on VC1          |
|   - Response-priority scheduling within shared VCs             |
|   - Breaks request-response circular dependencies              |
|   - Cost: halves per-VC buffer depth (4 slots per VC with 8)   |
+================================================================+
| Layer 0: Existing Dateline + Bubble + DOR (unchanged)          |
|   - Handles topology-specific wrap-around cycles               |
|   - DOR prevents turn-based cycles in 2D mesh                  |
|   - No changes needed for GSRP                                 |
+================================================================+
```

### Phase 1 Implementation (Write-Only GSRP)

- **Layer 0:** Unchanged (existing dateline/bubble for ring/torus, DOR for mesh).
- **Layer 1:** Not needed (no read requests, no responses).
- **Layer 2:** Per-destination credit limits (Section 7.2.4) prevent starvation.
- **Layer 3:** Epoch barriers at natural synchronization points (TTNN operation boundaries).

Since Phase 1 only supports `gsrp_write()` (no reads), Risk 1 does not apply. Risk 2
(transitive write cycles) is prevented by DOR routing on non-wrapping meshes and by
the existing dateline mechanism on torus topologies. Risk 3 (mixed traffic) is addressed
by the VC-based traffic separation from Section 7.2.5 (VC 0 for high-priority GSRP,
VC 1 for bulk/CCL).

### Phase 2 Implementation (Bidirectional GSRP with Reads)

- **Layer 0:** Unchanged.
- **Layer 1:** VC partitioning + response-priority scheduling (Section 7.3.4).
- **Layer 2:** Shared-pool admission control with `RESPONSE_RESERVATION = 2`.
- **Layer 3:** Epoch barriers with reduced frequency (relaxed BSP).

---

## 7.3.10 Formal Deadlock Freedom Argument

### Phase 1 (Write-Only, DOR Routing)

**Claim:** Under DOR routing on a 2D mesh, the GSRP write path is deadlock-free.

**Argument:**
1. DOR routing imposes a total order on dimensions (NS before EW). This eliminates routing
   cycles in acyclic (non-torus) topologies.
2. Within an epoch, writes are unidirectional from each source. Bidirectional writes exist
   globally but use physically distinct buffer pools (different chips' sender channels):
   - Forward writes (A->B) use sender channel resources on A.
   - Reverse writes (B->A) use sender channel resources on B.
3. At intermediate forwarding nodes, DOR ensures that if A->B goes NS then EW, and B->A
   also goes NS then EW (with reversed directions), their paths do not form a cycle in the
   buffer dependency graph.
4. At the epoch barrier, all in-flight packets drain -- the barrier acts as a "bubble"
   in the time domain, breaking any potential temporal dependency.

**Caveat for torus topologies:** The existing dateline mechanism handles topology-level
cycles. The epoch barrier provides an additional safety layer.

### Phase 2 (Read+Write, VC Separation + Response Priority)

**Claim:** Adding VC-based request/response separation with response-priority scheduling
preserves deadlock freedom for the read path.

**Argument:**
1. Read responses occupy VC 1 (or have strict priority within a shared VC). Since responses
   are terminal operations (they generate no further network traffic), they cannot create
   new dependencies.
2. Read requests occupy VC 0 (shared with writes). The request path is deadlock-free by
   the same DOR argument as the write path.
3. Since responses always drain (response priority guarantees progress), every outstanding
   request eventually receives its response.
4. Admission control limits per-destination buffer occupancy, preventing a single
   destination's requests from consuming all buffer slots.
5. No circular dependency can span both request and response traffic because response
   traffic is terminal and always makes progress.

---

## GSRP Implications

The deadlock analysis confirms that transparent bidirectional GSRP traffic is safe under a
well-chosen combination of existing and new mechanisms. The key design decisions:

1. **DOR routing is sufficient for write-only GSRP on 2D mesh topologies** -- no additional
   mechanism is needed beyond the existing infrastructure.
2. **Response-priority scheduling is the critical new mechanism for 2-VC hardware
   (Blackhole)** -- it breaks request-response cycles by ensuring terminal (response) traffic
   always makes progress.
3. **Epoch barriers are the safety net** -- they guarantee periodic draining of all in-flight
   traffic, catching any dependency chain not broken by Layers 0--2. The relaxed BSP model
   preserves compute-communication overlap within epochs.
4. **The `need_deadlock_avoidance_support` API should be extended** with a
   `DeadlockAvoidanceInfo` structure that enables automatic avoidance configuration by the
   control plane based on active workload characteristics.
5. **Request-reply buffer separation is the minimum viable mechanism for read support** --
   dedicating 2 of 8 buffer slots to responses provides deadlock freedom with only 25%
   capacity reduction for the response channel.

---

## Key Takeaways

- **GSRP introduces three deadlock risk classes not present in the current fabric:**
  (1) request-response circular dependencies when chip A reads from chip B while chip B reads
  from chip A, (2) multi-hop transitive dependency chains through shared forwarding buffers,
  and (3) mixed GSRP + CCL resource deadlocks when both traffic types share EDM channels.

- **Response-priority scheduling is the key mechanism for 2-VC hardware (Blackhole):** since
  read responses are terminal (they generate no further traffic), prioritizing them over
  requests within a shared VC guarantees that responses always drain, breaking
  request-response cycles.

- **Epoch-based barriers (relaxed BSP model) provide a structural deadlock-freedom guarantee**
  at ~10--30 $\mu$s per barrier for 8--32 chip systems. With compute phases $\geq$ 500
  $\mu$s, the overhead is 2--6% -- acceptable when aligned with natural TTNN operation
  boundaries.

- **A four-layer defense provides incrementally deployable protection:** Layer 0 (existing
  dateline/DOR) handles topology cycles, Layer 1 (VC separation + response priority) handles
  request-response cycles, Layer 2 (admission control) prevents starvation, and Layer 3
  (epoch barriers) provides a global safety net.

- **The formal deadlock freedom argument shows that DOR + response priority + admission
  control is sufficient for all identified risk classes** on 2-VC Blackhole hardware, with
  epoch barriers providing defense in depth.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `tt_metal/fabric/fabric_context.hpp` | `FabricContext::need_deadlock_avoidance_support(direction)`, `is_bubble_flow_control_enabled()`, `is_wrap_around_mesh()`, routing plane count queries |
| `tt_metal/fabric/control_plane.cpp` | `ControlPlane::initialize_dynamic_routing_plane_counts()`, routing table generation, BFS shortest-path |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | EDM router main loop (scheduling logic target for response priority), `CoordinatedEriscContextSwitchState` |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | `StaticSizedSenderEthChannel<HEADER, NUM_BUFFERS>`, buffer slot layout, VC depth |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp` | `BufferPtr`, `BufferIndex`, `distance_behind()`, `wrap_increment()` -- used for admission control |
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | `compressed_route_2d_t` (DOR encoding), `RoutingFieldsConstants::Mesh` direction encodings |
| `tt_metal/api/tt-metalium/global_circular_buffer.hpp` | `GlobalSemaphore` -- primitive for epoch barrier implementation |
| `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp` | `FabricReliabilityMode`, `FabricConfig` enum (mesh, torus variants) |

---

**Previous:** [`02_flow_control_and_congestion.md`](./02_flow_control_and_congestion.md) | **Next:** [`04_dispatch_integration_and_resource_management.md`](./04_dispatch_integration_and_resource_management.md)
