# 8.3 Explicit vs Implicit Tradeoff Matrix

## Context

Sections 8.1 and 8.2 quantified the raw performance characteristics of the TT-Fabric:
bandwidth across topologies, latency per hop, and the incremental cost of GSRP's address
translation layer. This section broadens the evaluation to six dimensions -- throughput,
latency, resource utilization, programmability, debuggability, and scalability -- comparing
the explicit CCL model (Chapter 1) against the proposed GSRP model (Chapters 6--7) in a
structured tradeoff matrix. It identifies the break-even point where GSRP overhead becomes
negligible, maps each major collective operation onto the GSRP model to assess per-collective
suitability, and provides a visual scoring summary to guide the hybrid architecture
recommendation.

---

## 8.3.1 Six-Dimension Comparison

| Dimension | Explicit CCL | GSRP (Runtime-Assisted) | Winner |
|-----------|-------------|--------------------------|--------|
| **Throughput** | Near-link-rate for pipelined ring collectives; ~11.8 B/c sustained (1-hop) | ~12.4 B/c (1-hop, 4096 B); no ring pipelining | Explicit CCL |
| **Latency** | First byte: ~330 ns (1-hop); pipeline fill: 1-hop latency | First byte: ~350--380 ns (1-hop); +13 cycles translation | Explicit CCL (marginal) |
| **Resource utilization** | Worker cores dedicated to comm; Tensix compute cycles consumed by CCL kernels | Fabric cores handle comm; Tensix cores freed for compute | **GSRP** |
| **Programmability** | 30--50 lines per cross-chip transfer; topology-aware | 1--3 lines per access; topology-hidden | **GSRP** |
| **Debuggability** | Deterministic traces; explicit packet construction visible in dprint/watcher | Non-deterministic timing; requires extended telemetry (Chapter 7, Section 4) | Explicit CCL |
| **Scalability** | Requires per-topology kernel variants; $O(T \times C)$ code paths | Topology-independent; single code path scales to arbitrary mesh size | **GSRP** |

### Visual Scoring Summary

Scoring each dimension on a 1--5 scale (5 = best):

```
Dimension           CCL Score    GSRP Score
----------------------------------------------
Throughput              5            3
Latency                 4            3
Resource utilization    2            5
Programmability         2            5
Debuggability           5            4
Scalability             3            5
----------------------------------------------
Total                  21/30        25/30
```

GSRP leads on four of six dimensions; CCL leads on throughput and debuggability. The
throughput advantage is decisive for ring-pipelined collectives but irrelevant for
point-to-point and irregular patterns.

---

## 8.3.2 Detailed Throughput Analysis

### Explicit CCL Throughput Model

Explicit CCL achieves high throughput through **ring pipelining**: data is divided into
$N-1$ chunks (for $N$ chips), and each chunk is forwarded to the next chip while the
previous chunk is being processed. The steady-state throughput approaches the per-link
bandwidth:

$$\text{BW}_{\text{CCL}} \approx \text{BW}_{\text{link}} \cdot \frac{N-1}{N}$$

For an 8-chip ring at ~12 B/c per link:

$$\text{BW}_{\text{CCL}} \approx 12 \times \frac{7}{8} = 10.5\ \text{B/c effective}$$

### GSRP Throughput Model

GSRP issues individual remote writes without cross-chip pipelining awareness. Each write
pays the full end-to-end latency. For sustained throughput, the sender must have multiple
writes in flight (pipelining at the packet level rather than the chunk level):

$$\text{BW}_{\text{GSRP}} \approx \text{BW}_{\text{link}} \cdot \frac{1}{1 + \frac{t_{\text{overhead}}}{t_{\text{transfer}}}}$$

For 4096 B packets with ~13 ns overhead on ~370 ns transfer time:

$$\text{BW}_{\text{GSRP}} \approx 12.8 \times \frac{1}{1 + 0.035} \approx 12.4\ \text{B/c}$$

### Throughput Comparison Table

| Transfer Pattern | Explicit CCL (B/c) | GSRP (B/c) | Gap |
|:----------------|:-------------------:|:-----------:|:---:|
| Single write, 1 hop | 12.8 | 12.4 | 3% |
| Single write, 4 hops | 6.4 | 6.2 | 3% |
| All-gather (8 chips, ring) | 10.5 (pipelined) | 6.2 (per-write) | **40%** |
| Point-to-point bulk | 12.8 | 12.4 | 3% |
| Irregular scatter | 8--10 (scheduling) | 8--10 (dynamic) | **~0%** |

> **Key Insight:** GSRP matches explicit CCL for point-to-point and irregular patterns
> (within 3--5%), but falls significantly behind for ring-pipelined collectives (40%+ gap).
> The pipelining advantage is the single strongest argument for retaining explicit CCL
> for bulk collectives.

---

## 8.3.3 Resource Utilization Comparison

| Resource | Explicit CCL | GSRP | Advantage |
|----------|:------------:|:----:|:---------:|
| Worker cores for comm | 1--8 per chip | 0 | GSRP saves 1--8 cores |
| L1 for comm buffers | 64--256 KB per chip | 0 (uses fabric buffers) | GSRP |
| L1 for metadata | ~6.9 KB (existing fabric) | ~12.2 KB (fabric + GSRP Phase 2) | CCL (marginal) |
| ERISC utilization | Shared | Higher load | CCL |
| Compute/comm overlap | Manual (async CCL) | Automatic | GSRP |

The existing fabric metadata totals **6,864 B** per core; GSRP Phase 2 adds **5,288 B**,
for a combined **~12.2 KB** per core (0.79% of BH L1, 0.83% of WH L1). See Chapter 7,
Section 4.4 for the full itemized breakdown of both components.

> **Recommendation:** For models that are compute-bound and L1-constrained, the worker core
> and L1 savings from GSRP (1--8 freed cores, 64--256 KB reclaimed L1) can translate
> directly into higher compute throughput.

---

## 8.3.4 Programmability Comparison

### Lines of Code per Transfer

```cpp
// Explicit CCL: ~30-50 lines per cross-chip transfer
// 1. Discover topology and sender/receiver config
// 2. Build command stream with slice descriptors
// 3. Select worker cores
// 4. Configure circular buffers
// 5. Set up global semaphores
// 6. Launch sender and receiver kernels
// 7. Handle topology-specific edge cases

// GSRP: 3 lines
GlobalAddress remote_dest = GlobalAddress::encode(target_mesh, target_chip,
                                                   target_core_x, target_core_y,
                                                   target_l1_offset);
gsrp_write(remote_dest, local_src_addr, transfer_size);
gsrp_fence_release();
```

| Aspect | Explicit CCL | GSRP |
|--------|:------------:|:----:|
| Lines of code per transfer | 30--50 | 1--3 |
| Topology awareness required | Yes | No |
| New topology = new code | Yes | No (runtime handles) |
| Custom optimization possible | Full control | Limited |
| Error surface area | Large (many parameters) | Small (address + size) |
| Learning curve | Steep (weeks) | Moderate (days) |

---

## 8.3.5 Debuggability Comparison

| Tool | Explicit CCL Support | GSRP Support (Projected) |
|------|---------------------|--------------------------|
| **Watcher** | Full: validates NOC writes, detects hangs | Needs extension: track GSRP writes, remote address violations |
| **dprint** | Full: worker kernels print packet contents, routes | Needs extension: instrument GSRP shim to log translated addresses |
| **Fabric Telemetry** (`FabricTelemetrySnapshot`) | Full: per-ERISC bandwidth counters, packet counts | Compatible: GSRP traffic uses same fabric path |
| **Packet Recorder** (`fabric_packet_recorder.hpp`) | Full: records packet headers at each hop | Compatible: GSRP packets have standard headers |
| **Telemetry Converter** (`fabric_telemetry_converter.hpp`) | Full: `unpack_bandwidth()`, HAL-based static info | Compatible |
| **Determinism** | Deterministic (compile-time schedule) | Non-deterministic (runtime arbitration) |

GSRP does not break existing fabric debugging tools -- GSRP packets traverse the same EDM
infrastructure and appear in telemetry and packet recordings. The new challenge is
**attribution**: mapping a fabric packet back to the source-level `gsrp_write()` call. The
GSRP telemetry block proposed in Chapter 7, Section 4 (48 B per core for counters and
stall tracking) addresses this.

---

## 8.3.6 Scalability Comparison

| Metric | Explicit CCL | GSRP |
|--------|:------------:|:----:|
| Code paths per topology | $O(T \times C)$ | $O(1)$ -- single implementation |
| Ring to torus migration | New kernel variants | Runtime route table update |
| 8-chip to 32-chip scaling | Re-tune chunk sizes, buffers, links | Transparent (routing tables expand) |
| Multi-mesh support | New inter-mesh forwarding code | Routing table includes inter-mesh paths |
| Adding new collective | Full implementation (~1000+ lines) | Not needed (point-to-point primitives compose) |

---

## 8.3.7 Connection Amortization Crossover

For workloads with repeated transfers to the same destinations, GSRP's persistent
connection pool provides a structural advantage:

```
Connection Cost Crossover:

  Cost
  (us)
  100 |  * CCL (per-op connection setup)
      |   *
   50 |    *
      |     *
   25 |      *
      |       *  -- Crossover at ~120 transfers --
   10 |  *-------*---------*
      | GSRP (persistent pool)
    0 +----+----+----+----+----+----+----
      1   25   50  100  120  200  500   Transfers

  CCL: ~3 us per connection setup x N ops = 3N us
  GSRP: ~1 us first setup + 0.005 us per subsequent = ~1 + 0.005N us
```

For workloads with more than ~120 repeated transfers (common in training loops), GSRP's
connection pool saves significant overhead. This crossover point is reached within the
first few training iterations for most production workloads.

---

## 8.3.8 Break-Even Analysis

The overhead gap between explicit CCL and GSRP depends on message size:

```
GSRP Overhead vs. Message Size:

  Overhead
  (%)
  15 |*
  12 | *
  10 |  *
   8 |    *
   5 |       * <-- 5% threshold at ~3 KB
   3 |           *
   2 |                * <-- 2% threshold at ~8 KB
   1 |                        *
   0 +--+----+----+----+-----+---------+
    16B  256B 1KB  4KB  8KB  16KB  64KB
            Message Size
```

The break-even thresholds are ~3 KB (at 5% overhead), ~8 KB (at 2%), and ~17 KB (at 1%) --
see Section 8.2.6 for the full derivation. For workloads that move tensors (typically
64 KB to several MB), GSRP performs within noise of explicit CCL on a per-transfer basis.
The remaining disadvantage is the lack of ring pipelining for multi-chip collectives.

---

## 8.3.9 Per-Collective Mapping to GSRP

### All-Gather

| Metric | Explicit CCL | GSRP |
|--------|:------------:|:----:|
| Throughput | ~10.5 B/c (8-chip ring, pipelined) | ~6.2 B/c (per write, 4 hops avg) |
| Programmability | ~50 lines per collective | ~10 lines (loop + gsrp_write) |
| Verdict | **Explicit CCL for large tensors** | GSRP acceptable for small tensors (<64 KB) |

### Reduce-Scatter

| Metric | Explicit CCL | GSRP |
|--------|:------------:|:----:|
| Throughput | ~10 B/c (8-chip ring, with reduction) | ~5--6 B/c (no pipeline reduction) |
| Reduction overlap | Yes (compute during transfer) | No (transfer then compute) |
| Verdict | **Explicit CCL strongly preferred** | |

### All-Reduce (= Reduce-Scatter + All-Gather)

| Metric | Explicit CCL | GSRP |
|--------|:------------:|:----:|
| Throughput | ~9 B/c (fused, 8-chip ring) | ~4--5 B/c (two-phase) |
| Verdict | **Explicit CCL strongly preferred** | |

### Point-to-Point (Send/Recv)

| Metric | Explicit CCL (Socket) | GSRP |
|--------|:--------------------:|:----:|
| Throughput | ~12.8 B/c (1-hop) | ~12.4 B/c (1-hop) |
| API simplicity | Good (socket abstraction) | Slightly simpler (direct address) |
| Verdict | **Equivalent** | |

### All-to-All (MoE Expert Dispatch)

| Metric | Explicit CCL | GSRP |
|--------|:------------:|:----:|
| Throughput | ~8--10 B/c | ~8--10 B/c |
| Programmability | Complex (data-dependent routing, ~200+ lines) | Simple (per-element gsrp_write, ~15 lines) |
| Dynamic patterns | Requires rebuild per batch shape | Handles natively |
| Verdict | **GSRP preferred** | |

### Broadcast

| Metric | Explicit CCL | GSRP |
|--------|:------------:|:----:|
| Throughput | ~11.7 B/c (multicast linear) | ~11.7 B/c (multicast same path) |
| Verdict | **Equivalent** (both use multicast) | |

---

## 8.3.10 Summary Decision Matrix

| Collective / Pattern | Explicit CCL | GSRP | Recommended |
|---------------------|:------------:|:----:|:-----------:|
| All-gather (bulk) | Strong | Weak | Explicit CCL |
| Reduce-scatter (bulk) | Strong | Weak | Explicit CCL |
| All-reduce (bulk) | Strong | Weak | Explicit CCL |
| Point-to-point | Good | Good | Either |
| Broadcast | Good | Good | Either |
| All-to-all / MoE | Complex | Simple | **GSRP** |
| KV-cache sharing | N/A (custom) | Natural fit | **GSRP** |
| Activation checkpointing | Complex | Natural fit | **GSRP** |
| Dynamic routing | Not supported | Native | **GSRP** |

---

## Key Takeaways

- **Explicit CCL wins for bulk ring-pipelined collectives** (all-gather, reduce-scatter,
  all-reduce) by 40--50% in throughput due to pipeline overlap that GSRP cannot replicate.

- **GSRP wins for irregular, dynamic, and point-to-point patterns** (MoE, KV-cache,
  activation checkpointing) through simpler code and equivalent performance.

- **The break-even message size is ~3 KB (at 5%) and ~8 KB (at 2%)**: above these
  thresholds, GSRP per-access overhead becomes negligible.

- **Resource savings from GSRP** (1--8 worker cores freed, 64--256 KB L1 reclaimed) may
  offset throughput differences for compute-bound workloads.

- **Connection amortization crossover at ~120 transfers**: beyond this, GSRP's persistent
  pool saves significant overhead compared to per-operation CCL connection setup.

- **The recommended approach is hybrid**: retain explicit CCL for the 3--4 bulk collectives,
  use GSRP for everything else.

## GSRP Implications

The tradeoff matrix confirms the hybrid model proposed in Chapter 6, Section 4. GSRP should
not attempt to replace explicit CCL for all-gather or reduce-scatter -- the ring pipelining
advantage is fundamental, not an implementation detail. Instead, GSRP provides maximum value
by eliminating the need for custom kernels in irregular communication patterns (MoE dispatch,
KV-cache sharing, dynamic activation routing) where explicit CCL offers no pipelining benefit.
The six-dimension scoring (CCL 21/30 vs. GSRP 25/30) shows that GSRP is the superior model
when weighting all engineering dimensions equally.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp` | `ExecuteAllGather::invoke` -- explicit CCL API surface |
| `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp` | `ExecuteReduceScatter::invoke` -- explicit CCL for reduce-scatter |
| `tt_metal/api/tt-metalium/experimental/fabric/fabric_telemetry.hpp` | `FabricTelemetrySnapshot` -- telemetry for both CCL and GSRP |
| `tt_metal/fabric/fabric_telemetry_converter.hpp` | `unpack_bandwidth()` -- bandwidth telemetry extraction |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | `WorkerToFabricEdmSenderImpl` -- worker-to-fabric interface |
| `tt_metal/api/tt-metalium/experimental/sockets/mesh_socket.hpp` | `MeshSocket` -- existing P2P abstraction for comparison |

---

**Previous:** [`02_latency_model.md`](./02_latency_model.md) | **Next:** [`04_wormhole_gap_analysis.md`](./04_wormhole_gap_analysis.md)
