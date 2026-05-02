# 6.1 Problem Statement and Transparency Levels

## Context

Chapters 1 through 5 established the full landscape: the explicit CCL programming model
that demands 30--50 lines of boilerplate per cross-chip transfer (Chapter 1), the per-core
L1 SRAM layout and chip-local NOC addressing (Chapter 2), the inter-chip Ethernet fabric
with source-routed packet headers and store-and-forward EDM firmware (Chapter 3), the
existing building blocks -- mesh API, sockets, MeshBuffer -- that partially abstract away
packet construction (Chapter 4), and the comparative evidence from Cerebras, NVIDIA, and
Graphcore that no production system has achieved fully transparent global SRAM across
Ethernet-class links (Chapter 5). This section formulates the precise problem that a Global
SRAM Pool (GSRP) must solve, quantifies the aggregate memory at stake, and introduces a
three-level transparency taxonomy that structures the design space explored in the remainder
of Chapter 6.

---

## 6.1.1 Goal Definition

The Global SRAM Pool (GSRP) is a proposed abstraction that satisfies three requirements:

1. **Addressability.** Every byte of L1 SRAM across every Tensix core in every chip of a
   multi-chip mesh can be named by a single 64-bit address.

2. **Accessibility.** A Tensix kernel running on any core can read from or write to any
   named address, including addresses that reside on remote chips, without constructing
   packet headers, managing fabric connections, or computing routes.

3. **Performance visibility.** The system exposes whether a given access is local
   (intra-chip) or remote (inter-chip), enabling compilers and expert programmers to make
   placement decisions that respect the NUMA-like latency hierarchy.

Requirement 3 distinguishes the GSRP from a fully transparent shared-memory model. Chapter 5
established that hiding the latency cliff between intra-chip NOC access ($< 100$ ns) and
inter-chip Ethernet access ($5$--$20$ $\mu$s per hop) leads to 11--100x performance
degradation (NVIDIA UVM). The GSRP therefore aims for **mechanical transparency** (automating
the how) while preserving **semantic transparency** (revealing the where).

### What the GSRP Eliminates

| Boilerplate category | Lines per transfer (Ch 1) | GSRP eliminates? | Mechanism |
|---------------------|:------------------------:|:-----------------:|-----------|
| Packet header construction | 5--8 | Yes | Runtime fills headers from global address |
| Route computation (`fabric_set_unicast_route`) | 3--5 | Yes | Runtime resolves route from address |
| Connection setup (`WorkerToFabricEdmSender`) | 8--12 | Yes | Connection pool managed by runtime |
| Flow control (`wait_for_empty_write_slot`) | 2--3 | Yes | Runtime manages credits transparently |
| Destination address encoding (`NocUnicastCommandHeader`) | 4--6 | Yes | Derived from global address |
| Synchronization (semaphores, barriers) | 3--5 | Partially | Epoch barriers replace ad-hoc sync |
| Local buffer management (CBs, double-buffering) | 5--10 | No | Remains application responsibility |

The GSRP targets the first five categories -- approximately 25--35 lines per cross-chip
transfer -- while leaving local buffer management to the programmer.

### What the GSRP Does NOT Do

- **Replace DRAM.** The GSRP pools L1 SRAM only. DRAM remains a separate tier.
- **Automatic data placement.** Deciding which data lives on which chip is a compiler or
  application concern. The GSRP provides the transport, not the placement policy.
- **Cache coherence.** L1 is scratchpad, not cache. There are no hardware caches to keep
  coherent (Section 6.3 addresses consistency instead).
- **Device-side dynamic allocation.** `MeshBuffer::create()` is host-initiated. The GSRP
  does not require a distributed malloc.
- **Eliminate all collective operations.** Reduction operations (all-reduce, reduce-scatter)
  inherently combine data movement with computation. GSRP provides the data movement
  substrate; reduction logic remains in compute kernels or library functions.

---

## 6.1.2 Quantifying the Aggregate SRAM Pool

### Per-Core L1 Budget (Blackhole)

| Region | Bytes | Source |
|--------|------:|--------|
| Total L1 per Tensix core | 1,572,864 (1,536 KB) | `dev_mem_map.h` (BH) |
| System reserved (MEM\_MAP\_END) | 40,416 (~39.5 KB) | `0x09DE0` (mailbox, firmware, routing, connections, headers, counters) |
| **Usable L1 per core** | **~1,532,448** (~1,496.5 KB) | |

### System-Level Aggregation

| Configuration | Chips | Tensix/chip | Usable L1/core | Aggregate Usable |
|---------------|------:|------------:|----------------:|-----------------:|
| BH single chip | 1 | 140 | ~1,497 KB | ~205 MB |
| BH 8 chips (T3K-equivalent) | 8 | 140 | ~1,497 KB | ~1.64 GB |
| BH 32 chips (Galaxy) | 32 | 140 | ~1,497 KB | ~6.52 GB |
| WH T3K (8 chips) | 8 | 80 | ~1,432 KB | ~894 MB |
| WH Galaxy (32 chips) | 32 | 80 | ~1,432 KB | ~3.58 GB |

Raw aggregate for 32-chip BH:

$$S_{\text{raw}} = 32 \times 140 \times 1{,}536 \text{ KB} \approx 6.56 \text{ GB}$$

### Realistic Exportable Estimate

Not all unreserved L1 is available for remote access. Each core needs local working space
for compute buffers, circular buffers, and stack. A practical GSRP design would expose only
a configurable **export region** -- 25--50% of unreserved L1. With GSRP metadata overhead
of ~15 KB per core (midpoint estimate, validated in Section 6.5):

$$S_{\text{net}} = 32 \times 140 \times (1{,}505 - 15) \text{ KB} \approx 6.17 \text{ GB}$$

At 50% export ratio:

$$S_{\text{exportable}} \approx 3.1 \text{ GB of globally addressable SRAM}$$

```
 Aggregate SRAM Budget (32-chip BH Galaxy)
 +---------------------------------------------------------+
 | Total L1:  6.56 GB (32 x 140 x 1,536 KB)               |
 |  +-----------------------------------------------------+|
 |  | Reserved (firmware, routing, headers): ~173 MB (~2.6%)||
 |  +-----------------------------------------------------+|
 |  | Local working set (compute buffers):  ~3.1 GB (~47%) ||
 |  +-----------------------------------------------------+|
 |  | GSRP-exportable region:              ~3.1 GB (~47%)  ||
 |  +-----------------------------------------------------+|
 |  | GSRP metadata:                       ~85 MB (~1.3%)  ||
 |  +-----------------------------------------------------+|
 +---------------------------------------------------------+
```

Even at 50% export, ~3.1 GB of distributed SRAM is comparable to Graphcore's IPU Mk2 at
897 MB per chip, and substantially exceeds a single Cerebras WSE core's 49 KB. The value
proposition is clear for workloads whose working sets fit in this envelope -- KV caches,
activation checkpoints, expert routing tables, and small-to-medium tensor shards.

### Bandwidth Budget

The aggregate capacity is useful only if the interconnect can sustain the traffic. From
Chapter 3 (Section 1):

| Metric | Value |
|--------|-------|
| Per-link Ethernet bandwidth | ~12.5 GB/s |
| Links per neighbor direction (WH/BH) | 4 |
| BH Galaxy bisection bandwidth (estimated) | ~200--400 GB/s |

For a 32-chip Galaxy all-gather of a 256 MB tensor:

$$t_{\text{all-gather}} \geq \frac{248 \text{ MB}}{300 \text{ GB/s}} \approx 0.83 \text{ ms}$$

This is well within the ~5.6 ms epoch budget (Chapter 5, Section 4), confirming that the
fabric bandwidth is not the fundamental bottleneck -- the software complexity of utilizing
it is.

---

## 6.1.3 Why Not Just Improve the CCL Library?

A natural counter-argument is that the explicit CCL model can be made more ergonomic without
introducing a GSRP. The answer depends on the workload mix:

| Communication pattern | CCL improvement sufficient? | GSRP adds value? |
|----------------------|---------------------------|-------------------|
| Bulk all-gather (ring) | Yes -- already pipelined | Marginal -- adds overhead |
| Bulk reduce-scatter | Yes -- chunk-sequential | Marginal |
| All-reduce (fused) | Yes -- with op fusion API | Marginal |
| Point-to-point (fixed) | Partially -- MeshSocket | Yes -- simplifies setup |
| Irregular all-to-all (MoE) | No -- patterns are data-dependent | **Strong** -- runtime routing |
| KV-cache read sharing | No -- requires fine-grained reads | **Strong** -- read path |
| Dynamic activation routing | No -- destinations unknown at compile time | **Strong** |
| Metadata/control word exchange | Overkill to launch a CCL op | **Strong** -- lightweight API |

The rightmost column identifies workloads where GSRP delivers value that CCL improvements
cannot match. These are precisely the irregular, fine-grained, or data-dependent patterns
increasingly common in modern LLM inference (MoE models, speculative decoding, dynamic
batching) that the explicit CCL model handles poorly because each pattern requires a new,
hand-written kernel.

---

## 6.1.4 Three Levels of Transparency

### Level A: Hardware-Transparent (Cache-Coherent NUMA Analogy)

A Tensix core issues a standard NOC read or write to a 64-bit address. If the address
falls outside the local chip's range, hardware intercepts and transparently routes through
the fabric.

```
// Level A pseudocode -- identical to local access
noc_async_write(src_local, global_addr, size);  // HW intercepts if remote
noc_async_write_barrier();
```

**Hardware requirements:**

| Requirement | Current WH/BH Status | Gap |
|-------------|:-------------------:|:---:|
| NOC address interception | Not supported | Silicon change |
| Automatic route translation | Not supported | Hardware TLB |
| Transparent reads (round-trip) | Not supported | Hardware protocol |
| HW flow control | Not supported | Hardware credit mgmt |

**Verdict:** Not feasible on WH or BH. Relevant as a next-gen silicon target.

### Level B: Runtime-Assisted (Explicit Address, Implicit Routing)

Kernels use a global address type. The runtime translates it into the correct fabric
packet and route. The kernel is aware the access may be remote but does not manage transport.

```cpp
// Level B pseudocode
GlobalAddress dest = make_global_addr(/*mesh*/0, /*chip*/7, /*x*/5, /*y*/3, 0x2000);
gsrp_async_write(dest, local_src, 4096);
gsrp_write_barrier();
```

**Hardware requirements:**

| Requirement | Current WH/BH Status | Gap |
|-------------|:-------------------:|:---:|
| Pre-computed routing tables | Exists (`ROUTING_PATH_BASE_2D`) | Need global-addr-indexed variant |
| Connection pool in L1 | Exists (`MEM_TENSIX_FABRIC_CONNECTIONS_BASE`) | Need multi-dest pool |
| Packet header pool | Exists | None |
| Two-phase send protocol | Exists | None |

**Verdict:** Achievable on current WH/BH hardware with software/firmware changes only.

### Level C: Library-Mediated (High-Level APIs)

Kernels use named abstractions -- `MeshSocket`, `MeshBuffer` -- that internally handle
all cross-chip mechanics.

```cpp
// Level C pseudocode
gsrp_region_t kv_cache = gsrp_open_region("kv_cache_shard_5");
gsrp_region_read(local_buf, kv_cache, /*offset=*/0, /*size=*/4096);
```

**Verdict:** Already substantially available via existing MeshSocket and MeshBuffer. The
lowest-hanging fruit for immediate delivery.

### Transparency Level Comparison

| Criterion | Level A (HW) | Level B (Runtime) | Level C (Library) |
|-----------|:-----------:|:-----------------:|:-----------------:|
| Kernel awareness | None | Address has chip ID | Named remote objects |
| Per-access overhead | ~0 cycles (HW) | ~15--25 cycles | ~50--100 cycles |
| Silicon requirement | Next-gen | Current (WH/BH) | Current (WH/BH) |
| Implementation timeline | 2--4 years | 6--12 months | 3--6 months |
| Read support | HW round-trip | FW request/response | Library-level |

---

## 6.1.5 Mapping to TT Hardware Generations

| Hardware generation | Best-fit level | Rationale |
|--------------------|:--------------:|-----------|
| Wormhole (current) | Lib-Mediated (C) | 1D routing limits RT-Assisted generality |
| Blackhole (current) | RT-Assisted (B) | 2D routing, dual ERISC, larger Ethernet L1 |
| Quasar (future) | HW-Transparent (A) | Opportunity for NOC global-address recognition |

### The Incremental Path

```
Phase 0 (Today):       Explicit CCL kernels
Phase 1 (Near-term):   Level C -- extend MeshBuffer/MeshSocket
Phase 2 (6-12 mo):     Level B -- gsrp_write/gsrp_read with FW translation
Phase 3 (Next-gen):    Level A -- hardware-transparent GSRP
```

Each phase subsumes the previous one. Level C primitives can call Level B APIs internally.
Level B APIs can leverage Level A mechanisms when available.

---

## 6.1.6 Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Boilerplate reduction | Lines of cross-chip code per kernel | $\leq 5$ (from 30--50) |
| Throughput overhead | Bandwidth loss vs. hand-optimized CCL | $< 5\%$ for bulk ($\geq$ 64 KB) |
| Latency overhead | Additional per-access latency | $< 100$ ns ($< 2\%$ of single-hop) |
| L1 overhead | Per-core GSRP metadata | $\leq 16$ KB (1% of BH L1) |
| Correctness | Consistency model violations | Zero (under specified model) |
| Debuggability | Transfer visibility in telemetry | 100% of GSRP transfers logged |

---

## 6.1.7 Scope Boundaries

1. **Write-path first.** Cross-chip reads require a request/response protocol and are
   treated as Phase 2 (aligns with TT-Fabric's posted-write model).
2. **Tensix-to-Tensix only.** DRAM, host memory, and Ethernet L1 are excluded.
3. **Single mesh first.** Multi-mesh is supported by the address encoding but deferred.
4. **No hardware modifications.** All mechanisms must work on current WH/BH silicon.
5. **Coexistence with explicit CCL.** The GSRP augments, not replaces, optimized CCL
   kernels for bulk collectives.

---

## Key Takeaways

- **The GSRP targets mechanical transparency -- automating packet construction, routing,
  and connection management -- while preserving performance visibility** so that programmers
  distinguish local from remote accesses. This avoids the performance cliffs that doomed
  fully transparent approaches (Chapter 5, NVIDIA UVM).

- **A 32-chip BH Galaxy yields approximately 6.2 GB of usable GSRP, with ~3.1 GB
  practically exportable as globally addressable memory** -- a meaningful pool for KV
  caches, activations, and expert routing tables.

- **Three transparency levels form an incremental adoption path** (C then B then A), with
  Level B on Blackhole as the primary design target and Level C as the immediately
  achievable stepping stone.

## GSRP Implications

The problem statement establishes that GSRP is not a universal replacement for explicit CCL
but a complementary mechanism targeting the long tail of cross-chip data movement. The
aggregate SRAM calculation confirms the pool is large enough to be practically useful
(~3.1 GB exportable on a 32-chip BH Galaxy). The scope boundary of write-path-first
reflects the hardware reality that TT-Fabric is optimized for posted writes and the
comparative lesson from Chapter 5 that demand-driven remote reads carry significant
latency penalties.

---

**Next:** [`02_address_space_design.md`](./02_address_space_design.md)
