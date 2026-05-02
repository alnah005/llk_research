# 8.7 Phased Adoption Roadmap

## Context

The preceding sections established the quantitative performance envelope (Sections 8.1--8.2),
the explicit-vs-implicit tradeoff landscape (Section 8.3), the hardware gap analyses for WH
and BH (Sections 8.4--8.5), and the workload suitability rankings (Section 8.6). This final
section synthesizes all findings into a phased adoption roadmap from the current explicit CCL
model toward increasingly transparent communication. Each phase delivers incremental,
standalone value. Decision gates between phases ensure the next investment is justified by
measured results from the previous phase. The reader should be familiar with the three
candidate architectures (Chapter 6, Section 4), the hybrid architecture recommendation
(Chapter 6, Section 4.5), the runtime infrastructure design (Chapter 7), and the industry
finding that no production system has achieved transparent global SRAM with Ethernet-class
links (Chapter 5, Section 4).

---

## 8.7.1 Roadmap Overview

```
Phase 0        Phase 1          Phase 2           Phase 3         Phase 4
(Today)        (Near-term)      (FW-assisted)     (Next-gen Si)   (Full integration)
              SW-only, WH/BH   FW changes, BH    Quasar tt-2xx   Compiler + HW
+---------+   +------------+   +-------------+   +------------+  +-------------+
| Explicit|   | Named      |   | Transparent |   | HW-Assisted|  | Compiler    |
| CCL     |-->| Remote     |-->| Global      |-->| GSRP       |->| CCL         |
| Kernels |   | Memory     |   | Write       |   |            |  | Elimination |
+---------+   +------------+   +-------------+   +------------+  +-------------+
  |              |                  |                 |               |
  |  DG1         |  DG2            |  DG3            |  DG4          |
  |  Programmer  |  Throughput     |  Silicon         |  Compiler     |
  |  productivity|  within 5%     |  availability    |  coverage     |
```

---

## 8.7.2 Phase 0: Today -- Explicit CCL + Fabric Routing

### Current State

Workers explicitly construct packet headers, manage connections, and orchestrate multi-chip
data movement via the CCL kernel model (Chapter 1).

| Aspect | Status |
|--------|--------|
| Communication model | Explicit kernel code (30--50 lines per transfer) |
| Routing | Source routing via pre-computed packet headers |
| Connection lifecycle | Open/close per operation |
| Supported collectives | All-gather, reduce-scatter, all-reduce, broadcast, all-to-all |
| Fabric API | `fabric_unicast_noc_unicast_write()` and related primitives (Chapter 4, Section 1) |
| Debugging | Watcher, DPrint, packet recorder, fabric telemetry |
| Hardware | WH (production), BH (incoming) |

### What Phase 0 Delivers

Phase 0 is the performance baseline. All subsequent phases are measured against it:
- Throughput: ~12.8 B/c (BH), ~5--12 B/c (WH) depending on topology (Section 8.1).
- Per-packet overhead: ~9--15 cycles (Section 8.2.4).
- Lines per transfer: 30--50 (Chapter 1, Section 2).
- Collective implementation effort: weeks per new pattern.

---

## 8.7.3 Phase 1: Named Remote Memory (Software-Only, Current Hardware)

### Timeline: 3--6 months (8--10 engineering-weeks of firmware work)

### Deliverables

| Deliverable | Description | Hardware | Effort |
|:------------|:-----------|:--------:|:------:|
| `GlobalAddress` type | 64-bit encoded: mesh_id, chip_id, noc_xy, l1_offset | WH + BH | Low |
| `gsrp_write()` | Posted write with automatic route resolution | WH + BH | Medium |
| `gsrp_write_release()` | Write + atomic increment for release semantics | WH + BH | Low |
| `gsrp_atomic_inc()` | Cross-chip atomic increment via global address | WH + BH | Low |
| `gsrp_epoch_barrier()` | Cross-chip barrier via distributed semaphore | WH + BH | Medium |
| Connection pool | Persistent connections across program invocations | WH + BH | Medium |
| `MeshBuffer.gsrp_address()` | Global address computation for MeshBuffer shards | WH + BH | Low |
| GSRP telemetry | Per-core counters (48 B) + per-ERISC counters (32 B) | WH + BH | Low |
| TTNN auto-insert pass (prototype) | Graph pass detecting cross-chip tensor deps | Host SW | High |

### Architecture

Phase 1 implements the Fat Fabric API (Chapter 6, Section 4.1):

```
Phase 1 Stack:

  Application Kernel
    |
    v
  Fat Fabric API (Layer 2)
    gsrp_write(dest, src, size)
    gsrp_write_release(dest, src, size, flag, val)
    gsrp_atomic_inc(dest, inc)
    |
    |  Address decomposition (~4 cycles)
    |  Route cache lookup (~5 cycles)
    |  Connection pool select (~3 cycles)
    |  Header population (~1 cycle, cached)
    |  Total: ~13 cycles
    v
  Mesh Fabric API (Layer 1, existing)
    fabric_unicast_noc_unicast_write()
    |
    v
  EDM / Fabric Hardware (Layer 0, existing)
```

### Target Workloads

| Workload | Phase 1 Coverage | Notes |
|:---------|:----------------:|:------|
| MoE expert dispatch | Full | Write-only, per-token gsrp_write() |
| Activation checkpointing | Full | Write to remote chip L1, retrieve via write-back |
| Pipeline parallel (P2P) | Full | gsrp_write() replaces socket for simple cases |
| KV-cache sharing | Partial | Push-based only (no gsrp_read yet) |

### Success Criteria

| Metric | Target | Measurement Method |
|:-------|:------:|:-------------------|
| Lines per transfer | <=5 | Code review of migrated workloads |
| Throughput overhead | <5% for 4096 B+ | Benchmark vs. Phase 0 on same workload |
| L1 overhead | <2 KB per core (Phase 1) | `MEM_MAP_END` delta measurement |
| New collective time | <3 days | Developer time tracking |
| Route cache hit rate | >90% | GSRP telemetry counters |

### Decision Gate 1 (DG1): Phase 1 to Phase 2

```
DG1 Criteria:

[_] At least 2 workloads (MoE + activation checkpoint) running on GSRP Phase 1
[_] Measured throughput overhead < 5% for bulk transfers (4096 B+)
[_] Developer feedback confirms > 5x productivity improvement
[_] No regressions in explicit CCL workloads running on the same fabric
[_] Galaxy-scale (32-chip) GSRP routing verified on BH

Proceed to Phase 2 when all criteria met.
```

### Escape Hatch

GSRP primitives are optional. Existing CCL kernels continue to work unchanged on the
same fabric. If Phase 1 overhead exceeds 5%, GSRP serves as a convenience layer for
non-performance-critical paths only.

---

## 8.7.4 Phase 2: Transparent Global Write (Firmware-Assisted, WH/BH)

### Timeline: 6--12 months after Phase 1 (12--16 engineering-weeks of firmware work)

### Deliverables

| Deliverable | Description | Hardware | Effort |
|:------------|:-----------|:--------:|:------:|
| `gsrp_read()` | Cross-chip read via request-response protocol | BH primary, WH limited | High |
| `gsrp_read_acquire()` | Read with acquire semantics (wait for flag) | BH + WH | Medium |
| `gsrp_prefetch()` | Asynchronous read hint for latency hiding | BH + WH | Medium |
| Address-Translation Shim | ERISC-1 handles address translation on BH | **BH only** | High |
| Export region validation | ERISC validates writes to globally-visible L1 | BH + WH | Low |
| Dynamic route caching | Automatic route table updates on topology change | WH + BH | Medium |
| Extended ControlPlane API | Global address registration, route pre-computation | Host SW | Medium |
| Full GSRP telemetry | Per-core (48 B) + per-ERISC (32 B) telemetry blocks | WH + BH | Low |

### Architecture (BH)

```
Phase 2 Stack (BH):

  Application Kernel
    |
    |  Option A: explicit gsrp_write() (same as Phase 1)
    |  Option B: NOC write to ERISC trap buffer (near-transparent)
    v
  Address-Translation Shim (BH subordinate RISC-V / ERISC-1)
    Intercept non-local NOC writes
    Translate GlobalAddress to destination tuple
    Resolve route from L1 table
    Issue fabric packet via primary RISC-V
    |
    v
  Mesh Fabric API (Layer 1, existing)
    |
    v
  EDM / Fabric Hardware (Layer 0, existing)
```

### New Capability: Cross-Chip Reads

```
gsrp_read() Flow:

1. Requester builds GsrpReadRequestHeader
2. Request packet sent to destination ERISC (N hops)
3. ERISC-1 reads target L1 via NOC (~200-500 ns)
4. Response packet sent back to requester (N hops)
5. Requester writes data to local L1
6. Completion counter incremented

Round-trip: ~2.0-2.3x write latency (Section 8.2.8)
```

### Target Workloads (New)

| Workload | Phase 2 Coverage | Notes |
|:---------|:----------------:|:------|
| KV-cache sharing | **Full** | gsrp_read() + prefetch for inference |
| Remote parameter serving | Full | Read weights from parameter server chip |
| Distributed attention | Full | Read K/V from remote chips |
| Small tensor parallel | Partial | Sub-64 KB all-gather via GSRP |

### Success Criteria

| Metric | Target | Measurement Method |
|:-------|:------:|:-------------------|
| Read latency overhead | <15% vs. push equivalent | Benchmark |
| Shim processing time (BH) | <30 cycles per packet | ERISC telemetry |
| ERISC-0 utilization | No degradation from Phase 0 | Router telemetry |
| Read-heavy workload perf | Within 10% of hand-optimized | KV-cache benchmark |

### Decision Gate 2 (DG2): Phase 2 to Phase 3

```
DG2 Criteria:

[_] BH Address-Translation Shim demonstrates < 5% throughput overhead
[_] gsrp_read() latency within 2.5x of gsrp_write() (1-hop)
[_] KV-cache sharing workload runs successfully on GSRP Phase 2
[_] NOC_SEC_FENCE_RANGE investigation completed (determines Phase 2.5 HW assist)
[_] Coexistence with CCL verified under concurrent traffic load

Proceed to Phase 3 when Quasar silicon is available.
```

### Escape Hatch

The Address-Translation Shim can be disabled per-ERISC, reverting to Phase 1 behavior.
`gsrp_read()` can be replaced by push-based `gsrp_write()` patterns at the cost of
increased programming complexity.

---

## 8.7.5 Phase 3: Hardware-Assisted GSRP (Next-Generation Silicon)

### Timeline: 12--24 months (contingent on Quasar availability)

### Deliverables

| Deliverable | Description | Hardware |
|:------------|:-----------|:--------:|
| HW address translation | Quasar `noc_address_translation_table` integration | **Quasar only** |
| Transparent NOC interception | NOC writes to global prefix auto-routed to fabric | Quasar |
| Extended NOC address space | 64-bit addresses with global prefix detection | Quasar |
| In-network reduction (prototype) | ERISC firmware performs reduction on forwarded packets | BH + Quasar |
| TTNN compiler pass (production) | Automatic cross-chip communication insertion | Host SW |

### Architecture

```
Phase 3 Stack (Quasar):

  Application Kernel
    |
    |  Standard noc_async_write(global_addr, src, size)
    |  (no gsrp_ prefix needed -- transparent)
    v
  [HW Mask Table] -- detects global address prefix (16 entries)
    |
    v
  [HW Endpoint Table] -- translates chip_id to physical (x,y) (1024 entries)
    |
    v
  [HW Routing Table] -- selects route to remote chip (32 entries)
    |
    v
  Fabric ERISC (automatic forwarding)
    |
    v
  Remote Chip Delivery (with address rebase to local L1 offset)
```

### TTNN Compiler Pass (Production Quality)

```
TTNN Compiler Pass:

  Input: Operation graph with per-chip placement
  |
  +-- 1. Identify cross-chip tensor edges
  |
  +-- 2. For each edge:
  |       - If bulk regular pattern: emit explicit CCL collective
  |       - If irregular/dynamic: emit gsrp_write()/gsrp_read()
  |       - If small/latency-sensitive: emit batched gsrp_write()
  |
  +-- 3. Schedule communication to overlap with computation
  |
  +-- 4. Insert epoch barriers at natural sync points
  |
  Output: Annotated graph with embedded communication ops
```

### Decision Gate 3 (DG3): Phase 3 to Phase 4

```
DG3 Criteria:

[_] Quasar silicon available and HW translation tables validated
[_] TTNN compiler pass covers > 60% of communication patterns automatically
[_] HW-assisted GSRP throughput matches explicit CCL within 3% for bulk transfers
[_] No deadlock incidents in extended testing (>1000 hours multi-chip training)
[_] In-network reduction prototype evaluated

Proceed to Phase 4 when compiler infrastructure is ready.
```

### Escape Hatch

Quasar HW translation tables can be disabled, falling back to Phase 2 firmware-assisted
translation. The TTNN compiler pass can be bypassed per-operation by annotating the graph.

---

## 8.7.6 Phase 4: Full Integration -- Hybrid CCL Elimination

### Timeline: 24+ months (after Phase 3 validation)

### Deliverables

| Deliverable | Description |
|:------------|:-----------|
| Compiler-driven CCL elimination | TTNN automatically selects GSRP vs. explicit CCL per-op |
| Hybrid runtime | GSRP for irregular patterns + explicit CCL for bulk collectives |
| Performance auto-tuner | Runtime selects algorithm (ring/tree/mesh-aware) per transfer |
| Full topology abstraction | Same TTNN graph runs on T3K, Galaxy, multi-Galaxy without changes |

### Compiler Decision Logic

```python
# TTNN Compiler Communication Strategy Selector (conceptual)

for each cross_chip_data_movement D:
    if D.is_collective() and D.transfer_size > 64_KB:
        strategy = EXPLICIT_CCL
        reason = "bulk collective, pipeline efficiency"
    elif D.is_dynamic_pattern():
        strategy = GSRP
        reason = "data-dependent routing"
    elif D.transfer_size < 4_KB and D.count > 120:
        strategy = GSRP
        reason = "small, frequent -- connection amortization"
    elif D.is_point_to_point() and D.destinations <= 4:
        strategy = GSRP
        reason = "simple P2P, programmability"
    else:
        strategy = EXPLICIT_CCL
        reason = "default: proven performance"
```

### Hybrid Model at Steady State

```
Phase 4 Steady-State Architecture:

+---------------------------------------------------------+
|  TTNN Model                                             |
|                                                         |
|  [FFN Layer]--GSRP(MoE dispatch)-->[Expert Chips]       |
|  [Attention]--CCL(all-gather)----->[All Chips]          |
|  [Gradient]---CCL(reduce-scatter)-->[All Chips]         |
|  [KV-Cache]---GSRP(read)---------->[Inference Chips]    |
|  [Ckpt]-------GSRP(write)--------->[Storage Chips]      |
|                                                         |
|  Compiler selects strategy per operation.               |
|  Both share fabric via routing plane partitioning.      |
+---------------------------------------------------------+
```

### Decision Gate 4 (DG4): Completion

```
DG4 Criteria:

[_] > 80% of production workloads run without hand-written CCL kernels
[_] Remaining 20% (bulk collectives) use auto-generated CCL
[_] New communication patterns expressible in < 10 lines of GSRP API
[_] No workload regressions relative to hand-optimized explicit CCL baseline
[_] Developer productivity measured: >= 50% reduction in comm kernel LOC

Phase 4 is the long-term steady state.
```

### Escape Hatch

The compiler's CCL elimination can be overridden per-operation by annotating the TTNN graph
with explicit collective specifications. Manual `gsrp_write()`/`gsrp_read()` calls remain
available for patterns the compiler cannot optimize.

---

## 8.7.7 Phase Timeline and Resource Estimates

```
Timeline (months from start):

  Month:  0    3    6    9    12   15   18   21   24   27   30
          |----|----|----|----|----|----|----|----|----|----|
  Phase 0 |====|
  Phase 1      |=========|
  DG1                     *
  Phase 2                 |==============|
  DG2                                   *
  Phase 3                               |=================|
  DG3                                                     *
  Phase 4                                         |=======...
```

| Phase | Duration | Engineering Effort | Key Risk |
|:-----:|:--------:|:------------------:|:---------|
| Phase 1 | 3--6 months | 2--3 engineers, ~8--10 eng-weeks FW | Low (SW-only) |
| Phase 2 | 6--12 months | 3--4 engineers, ~12--16 eng-weeks FW | Moderate (FW changes) |
| Phase 3 | 12--24 months | 4--6 engineers | High (silicon dependency) |
| Phase 4 | 6--12 months | 3--5 engineers | Moderate (compiler complexity) |

Note: Engineering effort estimates are concurrent-person counts, not FTE-months. Phase 1
requires 2--3 engineers working in parallel for 3--6 months.

---

## 8.7.8 Risk Mitigation

| Risk | Phase | Mitigation |
|:-----|:-----:|:-----------|
| GSRP overhead exceeds 5% | Phase 1 | Fallback to explicit CCL; GSRP as convenience layer only |
| BH subordinate RISC-V contention | Phase 2 | Limit shim to low-rate traffic; use MUX aggregation |
| Quasar delays | Phase 3 | Continue with FW-only GSRP (Phase 2 extended) |
| Compiler cannot cover irregular patterns | Phase 4 | Retain explicit GSRP API as primary for dynamic patterns |
| Deadlocks from bidirectional GSRP traffic | Phase 2+ | Epoch barriers at layer boundaries (Chapter 7, Section 3) |
| L1 pressure from GSRP metadata | Phase 1+ | Configurable per-program allocation; zero overhead when unused |
| Read handler degrades router throughput (WH) | Phase 2 | Limit WH to 1 outstanding read; defer multi-read to BH |

---

## 8.7.9 Value Delivery per Phase

| Phase | Standalone Value | Cumulative Value |
|:-----:|:-----------------|:-----------------|
| Phase 1 | 10x fewer lines for P2P, MoE, checkpoint; topology-agnostic addressing | Foundation for all subsequent phases |
| Phase 2 | Read support enables KV-cache sharing; near-transparent write on BH | Full GSRP API complete (write + read) |
| Phase 3 | Zero-line cross-chip writes on Quasar; compiler handles 60%+ of patterns | Hardware-speed GSRP for most workloads |
| Phase 4 | Automatic CCL selection; new topologies require zero kernel changes | Full communication abstraction |

> **Key Insight:** Each phase delivers **standalone value** independent of subsequent phases.
> Phase 1 alone justifies the investment by reducing MoE expert dispatch from ~200+ lines to
> ~10 lines. Even if Phase 3+ is never reached (e.g., Quasar delays), Phases 1--2 provide
> lasting productivity improvements for 60--70% of communication patterns.

---

## Key Takeaways

- **The roadmap spans 5 phases (0--4) over 24+ months**, progressing from software-only
  primitives (Phase 1) through firmware-assisted transparency (Phase 2) to hardware-assisted
  GSRP (Phase 3) and compiler-driven CCL elimination (Phase 4). Each phase has explicit
  decision gates tied to measured performance criteria.

- **Phase 1 is the lowest-risk, highest-impact starting point.** It requires only software
  changes (~8--10 weeks of FW work, low risk), runs on both WH and BH, and delivers a 10x
  reduction in per-transfer code volume for MoE and activation checkpointing workloads.

- **The hybrid model persists through all phases.** Even at Phase 4, bulk collectives
  (all-reduce, large all-gather) use explicit CCL for optimal pipelining. GSRP handles
  irregular, dynamic, and latency-sensitive patterns. The compiler selects the best model
  per-operation.

- **Escape hatches at every phase** ensure that GSRP investment is never wasted: each phase's
  output is a production-usable layer that can stand alone if subsequent phases are delayed
  or deprioritized.

- **Quasar's hardware address translation tables are the key enabler for Phase 3+**, providing
  the silicon support for fully transparent NOC-to-fabric address interception. Phase 2 on BH
  (firmware-assisted) validates the protocols that Phase 3 will accelerate in hardware.

## GSRP Implications

The phased roadmap confirms the central thesis of this guide: eliminating explicit CCL kernels
is not a binary transition but a progressive layering of abstraction. The GSRP does not
replace the fabric -- it builds on top of it. The Fat Fabric API (Phase 1) is a thin wrapper
around existing mesh API primitives. The Address-Translation Shim (Phase 2) is a firmware
extension, not a replacement. The Quasar hardware tables (Phase 3) accelerate the same
protocol. At each layer, the explicit CCL path remains available as a performance escape
hatch, ensuring that the system never sacrifices peak throughput for transparency.

## Source Code References

| File | Relevance |
|------|-----------|
| `tt_metal/fabric/hw/inc/mesh/api.h` | Layer 1 mesh API that Phase 1 Fat Fabric API wraps |
| `tt_metal/api/tt-metalium/mesh_buffer.hpp` | `MeshBuffer` for Phase 1 `gsrp_address()` extension |
| `tt_metal/api/tt-metalium/mesh_command_queue.hpp` | `MeshCommandQueue` for GSRP initialization integration |
| `tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_address_translation_tables.hpp` | Phase 3 Quasar HW translation tables |
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h` | `NOC_SEC_FENCE_RANGE` for Phase 2 BH investigation |
| `tt_metal/fabric/control_plane.cpp` | `ControlPlane` for Phase 2 global address registration |
| `tt_metal/fabric/fabric_context.hpp` | `FabricContext` for topology queries across all phases |
| `ttnn/cpp/ttnn/operations/ccl/` | Explicit CCL operations retained across all phases |

---

**Previous:** [`06_workload_suitability.md`](./06_workload_suitability.md)

---

**End of guide.** Return to [Guide Index](../index.md)
