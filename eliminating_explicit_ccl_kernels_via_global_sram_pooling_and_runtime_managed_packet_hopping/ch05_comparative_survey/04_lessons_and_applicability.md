# 5.4 Lessons and Applicability to Tenstorrent's GSRP Design

## Context

The preceding three sections surveyed how Cerebras (wafer-scale, no chip boundaries),
NVIDIA (high-bandwidth links with explicit software collectives), and Graphcore (BSP model
with compiler-scheduled exchange) handle the tension between implicit and explicit inter-chip
communication. This section synthesizes the quantitative findings across all three
architectures, maps them to Tenstorrent's specific hardware constraints, and identifies the
practical paths forward for a Global SRAM Pool (GSRP) design.

The reader should have read Sections 5.1 through 5.3. This section also references
TT-Fabric infrastructure from Chapter 3 and the existing building blocks from Chapter 4.

## Cross-Architecture Quantitative Summary

Per-architecture memory and bandwidth data are tabulated in Sections 5.1--5.3. The key
cross-cutting insight: Tenstorrent's Tensix cores have the **largest per-core SRAM** of any
architecture surveyed ($2.5\times$ Graphcore, $6\times$ NVIDIA, $31\times$ Cerebras), making
a "mostly-local with efficient remote fallback" GSRP model the natural fit.

### Communication Model Comparison

| Property | Cerebras | NVIDIA | Graphcore | TT-Metal (Current) |
|---------|---------|--------|----------|-------------------|
| Communication visibility | Fully implicit | Explicit (NCCL) | Compiler-inserted | Explicit (CCL kernels) |
| Routing model | HW dimension-ordered | SW (NCCL algorithms) | Compiler-scheduled | SW source routing |
| Synchronization | Dataflow (implicit) | Stream-based (explicit) | BSP barriers | Semaphores (explicit) |
| Compute-comm overlap | Inherent (pipelined) | Yes (async streams) | No (phase separation) | Yes (async CCL) |
| Deadlock avoidance | DOR guarantees | NCCL scheduling | BSP guarantees | Dateline + VC |
| Dynamic routing | No (static mapping) | Limited (NCCL topology) | No (static schedule) | Yes (runtime routes) |
| Compilation cost | Hours | N/A (library) | 10--30 min | N/A (explicit) |

## The Central Finding: No Production System Has Transparent Global SRAM With Ethernet Links

Across all three architectures surveyed, plus Tenstorrent's current model, a consistent
pattern emerges:

> **No production multi-chip AI accelerator has achieved fully transparent global SRAM at
> scale using Ethernet-class (or equivalent bandwidth) inter-chip links.** Each architecture
> that achieves communication transparency does so through mechanisms unavailable to
> Tenstorrent: physical elimination of chip boundaries (Cerebras), or strict compiler-enforced
> phase separation (Graphcore).

The only architecture with truly transparent global SRAM is Cerebras, and it achieves this by
eliminating chip boundaries -- a physical property, not a software one. Graphcore achieves
programmer transparency but not runtime transparency: the compiler statically schedules all
communication. NVIDIA, with the highest inter-chip bandwidth of any multi-chip system, still
relies on explicit collective libraries.

This finding directly constrains the GSRP design space: a Tenstorrent GSRP that attempts to
provide fully transparent, demand-driven remote SRAM access across Ethernet links is
architecturally unprecedented. CUDA Unified Memory is the closest analog, and its
$11\times$--$100\times$ performance penalty (Section 5.2) demonstrates why demand-driven
approaches fail for performance-critical workloads.

## The NUMA Analogy

The correct analogy for a TT GSRP is not a flat shared memory but a **Non-Uniform Memory
Access (NUMA)** system, where:

- Each chip's L1 is a "NUMA node"
- Local L1 access is fast (~1 cycle)
- Remote L1 access is possible but expensive (~5,000+ cycles for a multi-hop round trip)
- The programmer or runtime should maximize locality

CPU NUMA systems have decades of experience with this model. The key lessons:

1. **First-touch placement** (allocating data on the first node that accesses it) works well
   for many workloads.
2. **Explicit binding** (pinning data to specific nodes) is necessary for performance-critical
   code.
3. **Migration** (moving data between nodes) should be infrequent and batched.
4. **NUMA-oblivious code** suffers 2--3$\times$ slowdowns on 2-socket systems and
   5--10$\times$ on 4+ socket systems.

For a TT GSRP at 32--128 chips, the NUMA penalty for oblivious access would be far more
severe -- ~100--1000$\times$ for latency-sensitive patterns. The GSRP must make the hierarchy
visible, not hide it. This is the distinction between **mechanical transparency** (automating
packet construction, routing, and flow control) and **performance transparency** (making it
visible when an access crosses a chip boundary).

## The Programming Model Gap: Quantified

The gap between TT-Metal's current model and the surveyed architectures can be measured in
lines of communication code required to move a 4 KB tensor slice between chips:

| Platform | Lines of communication code | What the programmer writes |
|----------|---------------------------|--------------------------|
| Graphcore (Poplar) | 0 | Tensor declaration + mapping hint |
| NVIDIA (NCCL) | 5--10 | Communicator init, send/recv, sync |
| TT-Metal (current) | 30--50 | Sender construction, connection open, packet header, NOC setup, payload send, flow control, semaphore signal, connection close |

This $6$--$10\times$ code volume difference is the concrete target for GSRP. Each line
represents a decision the developer must make or a protocol step the developer must execute
correctly. A successful GSRP reduces this to a single `remote_write(global_addr, local_src,
size)` call (Path B) or zero lines (Path A, compiler-inserted).

## Two Practical Paths Forward

The comparative survey identifies two viable paths for reducing the explicit communication
burden. These are not mutually exclusive; a mature system would implement both.

### Path A: Compiler-Assisted Communication Insertion (Graphcore-like)

**Approach:** A TTNN graph compiler pass analyzes the operation graph, identifies cross-chip
tensor dependencies, and automatically inserts fabric communication operations at shard
boundaries.

**Quantitative feasibility for TT-Metal:**

| Constraint | Cerebras | Graphcore | TT-Metal |
|-----------|---------|----------|---------|
| Per-core memory | 48 KB (forces decomposition) | 624 KB (moderate) | 1,464--1,536 KB (large) |
| Operations per core | 1--few (dataflow) | ~tens per superstep | Many (full kernel) |
| Communication granularity | Per-core, per-op | Per-tile, per-superstep | Per-chip, per-operation |
| Compilation complexity | Very high ($> 1$ hr) | High (10--30 min) | Moderate (seconds--minutes) |

Tenstorrent's large per-core memory means the compiler only needs to: (1) determine which
tensor shards reside on which chips (from MeshBuffer's SHARDED layout), (2) identify
operations consuming shards from other chips, and (3) insert fabric operations using mesh API
primitives. This is a substantially simpler problem than Graphcore's full BSP scheduling.

**Limitations:** Only works for operations with statically determinable communication
patterns. Dynamic patterns (MoE expert selection, data-dependent routing) cannot be resolved
at compile time.

### Path B: Library + Hardware Assist (NVIDIA-like)

**Approach:** Extend the existing mesh API and fabric infrastructure with increasingly
powerful primitives that hide packet construction and routing, while keeping communication
initiation explicit.

**Concrete steps for TT-Metal:**

| Step | What it provides | Foundation in current codebase |
|-----|-----------------|------------------------------|
| Global address API | `remote_write(global_addr, data, size)` | Mesh API `fabric_unicast_noc_unicast_write` (Ch 4.1) |
| Automatic route resolution | Caller provides `(mesh_id, device_id, core_xy, offset)`, runtime resolves route | `fabric_set_unicast_route` + routing tables (Ch 3.4) |
| Transparent flow control | Backpressure handled by runtime, not kernel | `WorkerToFabricEdmSenderImpl` credit-based flow (Ch 4.2) |
| Named remote regions | Pre-mapped remote L1 regions accessible by name | `MeshBuffer` consistent-address allocation (Ch 4.4) |
| In-network reduction | ERISC firmware performs reduction on forwarded packets | No current support (firmware extension needed) |

## TT-Fabric + Mesh API as the Foundation for Path B

The survey reveals that Tenstorrent's existing infrastructure is better positioned for
Path B than might be apparent from the current explicit CCL model:

| Required capability | Current state | Gap |
|--------------------|--------------|-----|
| Per-packet route computation | `fabric_set_unicast_route` (Ch 4.1) | None -- exists |
| Multi-chip addressing | `FabricNodeId` (mesh_id + chip_id) | Needs core_xy + L1 offset |
| Cross-chip write | `fabric_unicast_noc_unicast_write` (Ch 4.1) | None -- exists |
| Cross-chip atomic | `fabric_unicast_noc_unicast_atomic_inc` (Ch 4.1) | None -- exists |
| Cross-chip multicast | `MeshMcastRange` + multicast API (Ch 4.1) | None -- exists |
| Consistent-address allocation | `MeshBuffer::create()` (Ch 4.4) | Need global address registry |
| Named socket abstraction | `SocketSenderInterface`, `MeshSocket` (Ch 4.3) | Need address-based variant |
| Flow control | Credit-based via stream registers (Ch 4.2) | Need transparent wrapper |
| In-network reduction | Not supported | Firmware extension needed |
| Address translation | Not supported | L1 tables or firmware needed |

Out of 10 required capabilities, **6 already exist**, 2 need extensions to existing
infrastructure, and only 2 require new mechanisms. This positions Path B as the lower-risk,
incremental approach.

## BSP-Inspired Epoch Barriers for Deadlock Avoidance

The Graphcore survey (Section 5.3) highlighted BSP's structural deadlock avoidance. A GSRP
introduces new deadlock risks: implicit bidirectional cross-chip traffic creates potential
circular dependencies that the current unidirectional traffic model avoids.

### Epoch-Based Barrier Model

```
   +---------------+     +------------------+     +---------+
   | Epoch N       | --> | Cross-chip sync  | --> | Epoch   | --> ...
   | (compute +    |     | (barrier ensures |     | N+1     |
   |  cross-chip   |     |  all Epoch N     |     |         |
   |  transfers)   |     |  transfers       |     |         |
   |               |     |  are visible)    |     |         |
   +---------------+     +------------------+     +---------+
```

Unlike strict BSP, this model allows **overlapping communication with computation within
an epoch**. The barrier only guarantees that all cross-chip transfers initiated in epoch $N$
are visible to all consumers before epoch $N+1$ begins. This preserves TT-Metal's async
communication advantage while providing BSP-like deadlock guarantees.

### Epoch Duration Analysis

For a TT-Fabric barrier (global semaphore increment + all-chip acknowledgment):

$$t_{\text{barrier, TT}} \approx 20\text{--}100\text{ }\mu\text{s}$$

(higher than Graphcore due to store-and-forward Ethernet latency versus direct IPU-Link).
To keep barrier overhead below $5\%$:

$$t_{\text{epoch}} > 20 \times t_{\text{barrier}} = 20 \times 100\text{ }\mu\text{s} = 2\text{ ms}$$

A 2 ms epoch corresponds to substantial computation -- on the order of a full transformer
layer. This granularity is practical: most collective operations in training loops already
operate at layer boundaries.

### When Epoch Barriers Apply

| Workload Pattern | Epoch Barrier Applicable? | Rationale |
|-----------------|--------------------------|-----------|
| Transformer inference (layer-by-layer) | Yes | Natural layer boundaries |
| Tensor-parallel all-gather/reduce-scatter | Yes | Per-operation communication at known graph points |
| Data-parallel gradient sync | Partial | All-reduce is a natural sync point, but overlap with backward pass is desired |
| MoE dispatch (dynamic routing) | No | Data-dependent communication, no predictable epoch |
| Pipeline parallelism (micro-batches) | No | Asynchronous pipeline stages, barriers would serialize |

## What the Survey Rules Out: Anti-Pattern Catalog

Equally important to what the survey recommends is what it rules out. Drawing on the failures
and limitations observed across all three architectures:

### Anti-Pattern 1: Fully Hardware-Transparent Global SRAM

Cerebras achieves this through wafer-scale integration. NVIDIA attempted a software
approximation with UVM and demonstrated $11\times$--$100\times$ overhead. **Conclusion:** A TT
GSRP should not aim for full hardware transparency where a NOC write to a "remote" L1
address is automatically intercepted and forwarded. The goal is to eliminate *boilerplate*,
not *awareness* of data locality.

### Anti-Pattern 2: Pure BSP Execution Model

Graphcore demonstrated BSP's strengths but also its costs: no compute-communication overlap,
strict phase separation, and 20--40% throughput loss for overlap-critical workloads.
**Conclusion:** Use BSP principles selectively (epoch barriers, compiler-scheduled exchange)
without adopting the full BSP execution model.

### Anti-Pattern 3: In-Network Compute as Primary Strategy

NVIDIA's SHARP is effective but narrowly scoped (FP addition for all-reduce). TT's Ethernet
link bandwidth is the primary bottleneck -- reducing link traversals matters more than
accelerating the reduction arithmetic. **Conclusion:** In-network reduction on ERISC is a
useful optimization but not a substitute for a proper GSRP communication strategy.

### Anti-Pattern 4: Single-Algorithm Communication

NCCL's success comes from maintaining a portfolio of algorithms (ring, tree, direct,
NVSwitch-offload) and selecting the best one per invocation. **Conclusion:** A GSRP that
uses a single communication strategy will underperform across the range of message sizes and
topologies that production workloads require.

### Anti-Pattern 5: Hardware-Only Solution Requiring Silicon Changes

Cerebras's transparency requires wafer-scale integration; NVLink's requires custom SerDes
and coherence protocols. **Conclusion:** Tenstorrent's GSRP should be implementable on
current Wormhole/Blackhole hardware through firmware and software changes, with optional
hardware support in future generations.

## Architecture-to-Path Mapping

| Workload pattern | Best path | Rationale | Architecture reference |
|-----------------|----------|-----------|----------------------|
| Bulk collectives (all-gather, reduce-scatter) | Path B (explicit library) | NCCL demonstrates explicit outperforms transparent for bulk patterns | NVIDIA |
| Tensor parallelism boundaries | Path A (compiler-inserted) or Path B | Graphcore handles similar patterns well | Graphcore |
| Pipeline parallelism (point-to-point) | Path B (MeshSocket, already exists) | Low benefit from additional abstraction | NVIDIA |
| MoE / irregular all-to-all | Path B (global address API) | Dynamic patterns benefit from runtime flexibility | NVIDIA + TT |
| KV-cache sharing (inference) | Path A or B (read-heavy) | Relaxed consistency sufficient | Cerebras (uniform latency) |
| Activation checkpointing | Path B (named remote regions) | Pre-mapped regions with bulk transfer | Graphcore (liveness analysis) |

**No single approach dominates across all workload patterns.** The optimal GSRP is a hybrid.

## Per-Core SRAM Overhead Estimate

Adding GSRP infrastructure increases the L1 reserved space per Tensix core:

| GSRP component | Estimated L1 cost per core | Purpose |
|---|---|---|
| Address translation table | 2--8 KB | Map global addresses to (chip, core, offset) |
| Remote access connection state | 1--2 KB | Track outstanding remote operations |
| Response buffers (for reads) | 4--16 KB | Hold remote read responses |
| Route cache | 1--4 KB | Cache recently used routes |
| **Total GSRP overhead** | **8--30 KB** | |

At the upper end, 30 KB per core is about 2% of L1 -- manageable but not negligible. The
GSRP design should make metadata allocation configurable per-program, so programs that do not
use cross-chip communication pay zero overhead. The existing `AllocatorConfig` mechanism
(Chapter 2.1) provides a model for this.

## Risk Assessment

Four risks deserve explicit attention, each drawn from the surveyed architectures:

| Risk | Source lesson | Mitigation |
|------|-------------|------------|
| **Abstraction leakage under load** -- transparent APIs degrade 10--100x under pathological access patterns | NVIDIA UVM | Make remote accesses syntactically distinct from local; provide performance lint for hot-loop remote access |
| **Compiler complexity spiral** -- initial patterns work well but long tail requires disproportionate effort | Graphcore Poplar | Maintain explicit API (Layer 1) as first-class, not fallback; expert users can always drop down |
| **L1 memory pressure** -- existing fabric uses ~5.5 KB/core; GSRP adds 8--30 KB more | All architectures | Configurable per-program metadata allocation; zero overhead when cross-chip comms unused |
| **Debugging opacity** -- implicit communication is hard to trace, profile, and debug | Cerebras, Graphcore | Extend `FabricTelemetry` and packet recording to log GSRP-initiated transfers from the start |

## Design Principles for TT's GSRP

Drawing on the full comparative analysis:

### Principle 1: Automate Mechanics, Preserve Semantics (Mechanical vs. Performance Transparency)

The GSRP should eliminate the boilerplate of cross-chip communication (packet construction,
route computation, connection management, flow control) while preserving the semantic
structure of collective operations. Remote accesses should be syntactically distinct from
local accesses at the API level. **Source:** NVIDIA's NCCL evolution, Cerebras's latency
uniformity vs. TT's latency discontinuity.

### Principle 2: Hierarchical Communication

Exploit the bandwidth hierarchy (intra-core L1 > intra-chip NOC > inter-chip Ethernet) by
performing local aggregation before cross-chip transfer. **Source:** NVIDIA's hierarchical
all-reduce, Graphcore's placement optimizer.

### Principle 3: Epoch Barriers at Natural Synchronization Points

Use BSP-style barriers at transformer layer boundaries, microbatch boundaries, and pipeline
stage boundaries to drain cross-chip traffic and prevent deadlocks. Allow unrestricted
communication within epochs. **Source:** Graphcore's BSP deadlock-freedom guarantee.

### Principle 4: Pre-Computed Transfer Schedules

For communication patterns that repeat across training iterations, compute the transfer
schedule once and reuse for all iterations. **Source:** Graphcore's pre-arranged exchange,
NCCL's connection caching, CUDA Graphs.

### Principle 5: Scale Algorithms, Not Mechanisms

Decompose multi-chip operations into local aggregation phases and inter-region transfer
phases. Support multiple algorithm strategies per collective pattern (ring, tree,
mesh-aware). **Source:** NCCL's auto-tuner.

## Phased Delivery Roadmap

| Phase | Timeline | Key deliverables | Value |
|-------|----------|-----------------|-------|
| **1: Enriched Fabric API** | 3--6 months | `GlobalAddress` type, `remote_write`/`remote_read`, connection pooling, `MeshBuffer` address resolution | Eliminates lowest-level boilerplate |
| **2: Collective Library** | 6--12 months | Shared algorithm selection library (ring/tree/halving-doubling), topology-aware selection via `MeshGraph`, epoch barrier API | Eliminates hand-written CCL kernels for standard patterns |
| **3: Compiler-Assisted** | 12--24 months | TTNN graph pass for cross-chip dependency detection, automatic collective insertion, communication-aware placement optimizer | Practical transparency for standard ML workloads |

Each phase delivers standalone value and builds the foundation for the next.

## Key Takeaways

- **No production system has achieved transparent global SRAM across Ethernet-class
  interconnects.** The optimal GSRP is workload-dependent: explicit collectives for bulk
  patterns, compiler-inserted communication for static tensor parallelism, and
  library-mediated access for dynamic/irregular patterns.

- **Two practical paths (compiler-assisted and library+HW-assisted) are not mutually
  exclusive.** The existing TT-Fabric mesh API provides 6 of 10 required capabilities,
  positioning the library path as the lower-risk starting point.

- **The NUMA analogy is the correct mental model**: the GSRP provides mechanical transparency
  (automating packet construction and routing) while preserving performance visibility
  (making chip-boundary crossings explicit). BSP-inspired epoch barriers at ~2 ms
  granularity provide structural deadlock avoidance at $< 5\%$ overhead.

## GSRP Implications

Chapter 6 should pursue a **layered system**: explicit CCL kernels for bulk collectives
(bottom), a global address API for irregular patterns (middle), and optional compiler-assisted
communication insertion (top). Epoch-based barriers provide the synchronization backbone.
What remains to be designed -- address translation, epoch barriers, and TTNN compiler
integration -- is explored in Chapters 6 and 7.

## Further Reading

- Kumar, S., et al. (2020). "Scale MLPerf-0.6 models on Google TPU-v3 Pods."
  *arXiv:1909.09756*. Describes Google TPU's approach to inter-chip communication.

- Abts, D., et al. (2020). "Think Fast: A Tensor Streaming Processor (TSP) for Accelerating
  Deep Learning Workloads." *ISCA'20*. Groq's TSP architecture provides another example of
  compiler-scheduled deterministic communication.

- Dally, W. J., and Towles, B. (2004). *Principles and Practices of Interconnection
  Networks.* Morgan Kaufmann. Foundational reference for mesh, torus, and crossbar
  topologies, including deadlock avoidance and routing algorithms.

- Ben-Nun, T., and Hoefler, T. (2019). "Demystifying Parallel and Distributed Deep Learning:
  An In-Depth Concurrency Analysis." *ACM Computing Surveys*, 52(4).

- Ivanov, A., et al. (2021). "Data Movement Is All You Need: A Case Study on Optimizing
  Transformers." *MLSys 2021*. Quantifies the communication bottleneck in transformer
  training.

- Valiant, L. G. (1990). "A Bridging Model for Parallel Computation." *Communications of
  the ACM*, 33(8), 103-111. Foundational BSP paper for epoch-based barrier design.

- Cho, M., et al. (2019). "BlueConnect: Decomposing All-Reduce for Deep Learning on
  Heterogeneous Network Hierarchy." *MLSys 2019*. Hierarchical all-reduce decomposition.

---

**Next:** [Chapter 6 -- The Global SRAM Pool Abstraction: Design Space Exploration](../ch06_global_sram_design/index.md)
