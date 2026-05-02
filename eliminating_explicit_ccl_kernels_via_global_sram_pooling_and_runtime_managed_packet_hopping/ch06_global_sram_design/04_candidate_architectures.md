# 6.4 Candidate Architectures

## Context

The preceding sections established the GSRP's goal (Section 6.1), defined a 64-bit global
address encoding (Section 6.2), and specified a release consistency model with epoch barriers
(Section 6.3). This section presents three concrete candidate architectures for implementing
the GSRP at the RT-Assisted transparency level on Blackhole hardware. Each candidate is
evaluated against the success criteria from Section 6.1 and the comparative lessons from
Chapter 5. The analysis concludes with a weighted decision matrix and a recommended layered
hybrid architecture.

---

## 6.4.1 Candidate 1: Fat Fabric API

### Concept

Extend the existing mesh fabric API (`tt_metal/fabric/hw/inc/mesh/api.h`) with higher-level
primitives that accept `GlobalAddress` arguments and internally handle address decomposition,
route resolution, connection selection, and flow control. Communication remains **explicit in
kernel code** -- the programmer writes `gsrp_write(dest, src, size)` instead of the 30--50
line manual sequence -- but the mechanics are fully automated.

### API Surface

```cpp
// Core GSRP write: fire-and-forget posted write to global address
FORCE_INLINE void gsrp_write(
    GlobalAddress dest,         // Encoded per Section 6.2
    uint32_t local_src_addr,    // L1 address of source data
    uint32_t size);             // Transfer size (16-byte aligned)

// GSRP write with release semantics
FORCE_INLINE void gsrp_write_release(
    GlobalAddress dest,
    uint32_t local_src_addr,
    uint32_t size,
    GlobalAddress flag_addr,    // Release flag target
    uint32_t flag_value);

// GSRP read: blocks until data arrives in local buffer
FORCE_INLINE void gsrp_read(
    GlobalAddress src,          // Remote source address
    uint32_t local_dst_addr,    // L1 address for received data
    uint32_t size);

// GSRP read with acquire semantics
FORCE_INLINE void gsrp_read_acquire(
    GlobalAddress src,
    uint32_t local_dst_addr,
    uint32_t size,
    GlobalAddress flag_addr,    // Acquire flag to wait on
    uint32_t expected_value);

// Atomic increment at global address
FORCE_INLINE void gsrp_atomic_inc(
    GlobalAddress dest,
    uint32_t increment);

// Epoch barrier
FORCE_INLINE void gsrp_epoch_barrier();
```

### Internal Flow for `gsrp_write()`

```
gsrp_write(dest, src, size)
  |
  +-- 1. Decompose dest: extract mesh_id, chip_id, noc_x, noc_y, l1_offset
  |       Cost: 5 shifts + masks (~3 cycles)
  |
  +-- 2. Locality check: if (mesh_id == MY_MESH && chip_id == MY_CHIP)
  |       -> local: noc_async_write(noc_xy, l1_offset, src, size)
  |       Cost: 1 comparison (~1 cycle)
  |
  +-- 3. Route resolution: lookup route to (mesh_id, chip_id) in L1 table
  |       Reuses existing ROUTING_TABLE_BASE / ROUTING_PATH_BASE_2D tables
  |       Cost: 1-2 L1 reads (~5-10 cycles)
  |
  +-- 4. Connection selection: pick available connection from pool
  |       Round-robin across routing planes for load balancing
  |       Cost: 1 counter increment + table read (~3-5 cycles)
  |
  +-- 5. Header construction: fill NocUnicastCommandHeader with (noc_x, noc_y, offset)
  |       Fill routing fields from resolved route
  |       Cost: 4-6 stores (~4-6 cycles)
  |
  +-- 6. Two-phase send: payload then header (existing EDM protocol)
  |       wait_for_empty_write_slot()  -- may block on flow control
  |       send_payload_non_blocking(src, size)
  |       send_header_flush(header)
  |       Cost: dominated by fabric latency, not CPU
  |
  Total CPU overhead: ~16-25 cycles per gsrp_write (before fabric latency)
```

### Connection Pool

The Fat Fabric API maintains a per-core connection pool: pre-opened fabric connections
across available routing planes.

```
Connection Pool Table (per core):
+----------+---------+---------+-----------------------+
| pool_idx | dst_mesh| dst_chip| connection_handle_ptr |
| 4 bits   | 8 bits  | 8 bits  | 32 bits (L1 pointer)  |
+----------+---------+---------+-----------------------+
Entry size: 8 bytes.  Pool size (16 entries): 128 bytes.
```

For a 32-chip system, 16 entries cover 16 destination chips. If a destination is not in
the pool, the API falls back to a slower path that dynamically selects a connection.

**L1 cost per connection:**

| Component | Size | Notes |
|-----------|:----:|-------|
| Connection struct | 16 B | Pointers + metadata |
| Sender interface state | ~64 B | `WorkerToFabricEdmSenderImpl` working state |
| Packet header | ~144 B | `HybridMeshPacketHeader` (BH) |
| **Per-connection total** | ~224 B | |
| **Pool (4 connections)** | ~896 B | |

### Evaluation

| Criterion | Fat Fabric API | Target (Sec 6.1) | Assessment |
|-----------|:--------------:|:-----------------:|:----------:|
| Lines per transfer | 1--3 | $\leq$ 5 | **Meets** |
| CPU overhead per access | ~16--25 cycles | -- | Low |
| Throughput overhead | ~1--2% | $< 5\%$ | **Meets** |
| L1 metadata | ~192 B (pool) + ~4 KB (table) | $\leq$ 16 KB | **Well under** |
| Firmware changes | None (uses existing EDM) | -- | Minimal risk |
| Read support | Requires request-response (new) | -- | Partial |
| Debugging | API-level tracing via `FabricTelemetry` | 100% logged | **Meets** |

### Strengths

1. **Minimal risk.** Builds directly on existing mesh API; no firmware or hardware changes
   for the write path.
2. **Performance visibility preserved.** Kernels explicitly call `gsrp_write()`, making
   remote accesses syntactically distinct from local writes.
3. **Incremental adoption.** Existing CCL kernels can be migrated one operation at a time.

### Weaknesses

1. **Still explicit.** Programmers must insert `gsrp_write()` calls manually.
2. **Read path is difficult.** Remote reads require a request packet and response -- the
   current fabric is push-oriented with no standard pull protocol.
3. **No compiler integration.** The TTNN compiler cannot auto-optimize placement or
   communication patterns.

---

## 6.4.2 Candidate 2: Address-Translation Shim (ERISC)

### Concept

Insert a thin firmware layer on each ERISC that intercepts Tensix-issued NOC transactions
targeting out-of-range addresses and converts them to fabric packets. Tensix kernels issue
writes to a "trap region" in ERISC L1; the shim detects these writes and routes them
through the fabric.

### Architecture

```
Tensix Core                  ERISC (with shim)            Remote Chip
+-----------+               +------------------+         +----------+
| Kernel    |               | Shim Firmware    |         | Target   |
| issues    |  NOC write    | 1. Intercept     | Fabric  | Core     |
| noc_write +-------------->+    non-local addr +-------->+ receives |
| (to ERISC |  to ERISC     | 2. Translate     | packet  | NOC      |
|  trap buf)|  L1 trap      | 3. Route resolve |         | write    |
+-----------+               +------------------+         +----------+
```

### Trap Entry Format

Each trap entry (doorbell protocol) occupies 32 bytes in ERISC L1:

```
Trap Entry (32 bytes, 16-byte aligned):
+------------------+------------------+
| global_addr (8B) | src_addr (4B)    |
+------------------+------------------+
| size (4B)        | flags (4B)       |
+------------------+------------------+
| doorbell (4B)    | padding (8B)     |
+------------------+------------------+
```

The Tensix kernel writes payload data to a staging area in ERISC L1, then writes the trap
entry header + increments the doorbell. The ERISC shim detects the doorbell change and
processes the entry.

### Contention Analysis

With 140 Tensix cores per BH chip and 14 active ERISC cores (Chapter 3, Section 1):

$$\text{Tensix-to-ERISC ratio} = \frac{140}{14} = 10\ \text{Tensix cores per ERISC}$$

If each Tensix core issues GSRP writes at 1 GB/s, each ERISC must handle $10 \times 1 =
10$ GB/s of trap traffic. The per-ERISC Ethernet link bandwidth is ~12.5 GB/s -- the
ERISC becomes a bottleneck at ~125% Tensix utilization (i.e., no bottleneck at typical rates).

**Mitigation:** Use MUX cores (Chapter 3, Section 5) to aggregate traffic from 8--16 Tensix
cores before hitting the ERISC, reducing per-ERISC contention.

### Evaluation

| Criterion | Addr-Translation Shim | Target (Sec 6.1) | Assessment |
|-----------|:---------------------:|:-----------------:|:----------:|
| Lines per transfer | 2--4 | $\leq$ 5 | **Meets** |
| CPU overhead (Tensix) | ~8--12 cycles + ERISC latency | -- | Moderate |
| Throughput overhead | 3--8% (extra NOC hop + FW) | $< 5\%$ | **Marginal** |
| L1 metadata (Tensix) | ~32 B (trap address) | $\leq$ 16 KB | Negligible |
| L1 metadata (ERISC) | ~2--4 KB (trap buf + tables) | -- | Within budget |
| Firmware changes | **Significant** (new ERISC shim) | -- | High risk |
| Read support | Natural (ERISC can mediate) | -- | **Good** |
| Debugging | Shim can log all intercepted transfers | 100% logged | **Meets** |

### Strengths

1. **Near-transparent.** Kernels use a pattern close to standard NOC writes.
2. **Centralized translation.** ERISC handles address translation, keeping Tensix L1
   overhead near zero.
3. **Read support is natural.** ERISC can issue fabric read requests and write responses
   back to the requesting Tensix core's L1.

### Weaknesses

1. **ERISC bottleneck.** The 10:1 Tensix-to-ERISC ratio creates contention when multiple
   cores simultaneously target remote chips.
2. **Extra hop latency.** Every remote access traverses the intra-chip NOC to ERISC,
   adding ~100--500 ns.
3. **Firmware complexity.** Adding a shim to the EDM router increases firmware complexity
   and risks impacting router performance.

---

## 6.4.3 Candidate 3: Compiler-Inserted Communication

### Concept

A TTNN graph compiler pass analyzes the data flow graph, identifies cross-chip tensor
dependencies, and automatically inserts GSRP transfer operations. Kernels contain zero
cross-chip communication code; the compiler handles everything.

### Compiler Pass Architecture

```
TTNN Operation Graph
  |
  v
[Placement Pass] -- assigns ops to (mesh_id, chip_id, core_xy)
  |
  v
[Dependency Analysis] -- identifies cross-chip tensor edges
  |
  v
[Communication Insertion] -- inserts gsrp_write / gsrp_read / barrier
  |
  v
[Scheduling Pass] -- overlaps communication with computation
  |
  v
[Code Generation] -- emits kernel code with embedded GSRP calls
```

### Cross-Chip Dependency Detection

The compiler identifies a cross-chip dependency when:

$$\text{placement}(\text{producer\_op}) \neq \text{placement}(\text{consumer\_op})$$

For each such dependency, the compiler selects a communication pattern:

| Dependency pattern | Inserted communication | Algorithm |
|-------------------|----------------------|-----------|
| 1-to-1 (pipeline) | `gsrp_write` + `gsrp_read_acquire` | Point-to-point |
| 1-to-N (broadcast) | `gsrp_multicast_write` | Fabric multicast |
| N-to-1 (gather) | N $\times$ `gsrp_write` + barrier | Ordered writes |
| N-to-N (all-to-all) | Ring or tree algorithm | Collective |
| Reduction | `gsrp_write` + local reduce kernel | Hierarchical |

### Evaluation

| Criterion | Compiler-Inserted | Target (Sec 6.1) | Assessment |
|-----------|:-----------------:|:-----------------:|:----------:|
| Lines per transfer (user) | 0 | $\leq$ 5 | **Exceeds** |
| CPU overhead per access | Same as emission target | -- | Depends |
| Throughput overhead | 2--10% (may not match hand-tuned) | $< 5\%$ | **Risk** |
| L1 metadata | Same as emission target | $\leq$ 16 KB | Depends |
| Firmware changes | None (emits existing API calls) | -- | No risk |
| Compiler complexity | **High** | -- | Major investment |
| Time to production | 12--24 months | -- | Slowest |
| Debugging | Compiler IR visible | 100% logged | **Meets** |

### Strengths

1. **Zero programmer burden.** Cross-chip communication invisible to kernel developers.
2. **Global optimization.** Compiler overlaps communication with computation, batches small
   transfers, and selects optimal algorithms with full graph visibility.
3. **Portability.** Same TTNN graph runs on any mesh; compiler adapts to topology.

### Weaknesses

1. **Throughput risk.** Compiler-generated communication may not match hand-optimized CCL
   for well-studied patterns. Even NVIDIA retains explicit NCCL despite massive investment.
2. **Long tail complexity.** Dynamic patterns (MoE routing, speculative decoding) require
   increasingly complex compiler analysis.
3. **Opacity.** Compiler-inserted communication is harder to debug and profile.

---

## 6.4.4 Weighted Decision Matrix

| Criterion | Weight | Fat Fabric API | Addr-Translation Shim | Compiler-Inserted |
|-----------|:------:|:--------------:|:---------------------:|:-----------------:|
| Boilerplate reduction | 20% | 2 (0.40) | 2 (0.40) | 3 (0.60) |
| Throughput overhead | 25% | 3 (0.75) | 1 (0.25) | 2 (0.50) |
| L1 overhead | 10% | 2 (0.20) | 3 (0.30) | 2 (0.20) |
| Read path support | 10% | 1 (0.10) | 3 (0.30) | 2 (0.20) |
| Firmware risk | 15% | 3 (0.45) | 1 (0.15) | 3 (0.45) |
| Time to production | 10% | 3 (0.30) | 2 (0.20) | 1 (0.10) |
| Debugging visibility | 10% | 3 (0.30) | 2 (0.20) | 1 (0.10) |
| **Weighted Total** | **100%** | **2.50** | **1.80** | **2.15** |

Scoring: 3 = best, 2 = good, 1 = marginal (multiplied by weight).

> **Recommendation:** The Fat Fabric API scores highest (2.50/3.00) and should be the
> **Phase 1 implementation target**. The Compiler-Inserted approach is the **Phase 3
> target**, building on the Fat Fabric API as its emission backend. The
> Address-Translation Shim is a **contingency option** if future silicon (Quasar)
> provides better interception mechanisms.

---

## 6.4.5 Hybrid Architecture: The Recommended Path

The three candidates are not mutually exclusive. The recommended architecture is a layered
hybrid where each layer adds value independently:

```
Layer 3 (Phase 3): TTNN Compiler Pass
  |  Detects cross-chip deps, inserts communication
  |  Emission target: Layer 2 or Layer 1
  v
Layer 2 (Phase 2): Fat Fabric API
  |  gsrp_write(), gsrp_read(), gsrp_atomic_inc()
  |  Address decomposition, route resolution, connection pool
  v
Layer 1 (existing): Mesh Fabric API
  |  fabric_unicast_noc_unicast_write(), etc.
  |  Packet construction, EDM send, flow control
  v
Layer 0 (existing): EDM / Fabric Hardware
  |  Ethernet links, store-and-forward routing
```

| Layer | Alone | With layer above |
|-------|-------|-----------------|
| Layer 1 (Mesh API) | ~15 lines/transfer | -- |
| Layer 2 (Fat Fabric API) | 1--3 lines/transfer | Compiler-generated calls |
| Layer 3 (Compiler) | N/A (needs target) | 0 lines/transfer |

**Design principle:** Layer 2 must be complete and self-sufficient before Layer 3 is
attempted. A well-designed Fat Fabric API makes the compiler pass a code generator that
emits `gsrp_write`/`gsrp_read` calls rather than raw fabric primitives -- dramatically
reducing compiler complexity.

### Escape Hatch

For performance-critical paths where GSRP overhead is unacceptable, kernels can drop to
Layer 1 (existing mesh API) or raw EDM operations. This is analogous to inline assembly
in C -- a safety valve for expert users. The GSRP explicitly preserves this path.

---

## Key Takeaways

- **Three candidate architectures span the trade-off space** between programmer convenience
  and implementation complexity: the Fat Fabric API (lowest risk, best throughput at ~1--2%
  overhead), the Address-Translation Shim (best read support but 10:1 ERISC contention),
  and Compiler-Inserted Communication (zero programmer burden, 12--24 month timeline).

- **The Fat Fabric API scores highest on the weighted decision matrix (2.50/3.00)** due to
  its low firmware risk, best-in-class throughput overhead, and fastest time to production
  (3--6 months). It should be the Phase 1 implementation target.

- **A layered hybrid architecture** (Mesh API $\to$ Fat Fabric API $\to$ Compiler Pass)
  delivers incremental value at each phase, allows expert users to drop to lower layers,
  and ensures the compiler pass has a well-tested emission target.

## GSRP Implications

The candidate architecture analysis confirms that the GSRP is not a single mechanism but a
layered system. The Fat Fabric API's `gsrp_write(dest, src, size)` must resolve the
`GlobalAddress` into a destination tuple, look up the route, select a connection, and issue
the fabric transfer -- all within the $< 5\%$ throughput overhead target. This makes the
address translation inner loop the performance-critical path of the entire GSRP design.

## Source Code References

| File | Relevance |
|------|-----------|
| `tt_metal/fabric/hw/inc/mesh/api.h` | Existing mesh API that Fat Fabric API extends |
| `tt_metal/fabric/hw/inc/mesh/adapter.h` | `WorkerToFabricEdmSenderImpl` connection interface |
| `tt_metal/fabric/hw/inc/mesh/routing.h` | `fabric_set_unicast_route()` and route primitives |
| `tt_metal/fabric/hw/inc/tt_fabric_api.h` | `fabric_unicast_noc_unicast_write()` and related |
| `tt_metal/api/tt-metalium/control_plane.hpp` | `ControlPlane` route computation for host setup |

---

**Previous:** [`03_coherence_and_consistency_models.md`](./03_coherence_and_consistency_models.md) | **Next:** [`05_address_translation_layer.md`](./05_address_translation_layer.md)
