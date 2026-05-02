# 5.3 Graphcore IPU: Bulk Synchronous Parallel and Compiler-Scheduled Exchange

## Context

The previous two sections examined architectures at opposite extremes: Cerebras eliminates
chip boundaries entirely, while NVIDIA bridges them with high-bandwidth hardware but retains
explicit software collectives. Graphcore's Intelligence Processing Unit (IPU) occupies a
distinctive middle ground that is arguably the most relevant comparator for Tenstorrent's
GSRP ambitions. Like Tenstorrent, Graphcore builds discrete chips with per-core SRAM and no
hardware cache coherence. Like Tenstorrent, Graphcore must move data between chips using
serialized links (IPU-Links). But unlike Tenstorrent's current model of explicit CCL
kernels, Graphcore uses a **compiler-driven Bulk Synchronous Parallel (BSP) model** where
all inter-core and inter-chip communication is automatically scheduled by the compiler into
explicit exchange phases.

This section provides a quantitative analysis of the IPU architecture and its BSP
communication model, focusing on the aspects most relevant to the GSRP question: How does
the compiler determine what data to move, when, and where? What is the performance cost of
the BSP synchronization boundary? And how does the SRAM-only constraint shape the design?

The reader should be familiar with TT-Metal's explicit CCL kernel model (Chapter 1) and the
circular buffer / semaphore-based synchronization primitives (Chapter 2, Section 4). This
section does not assume prior Graphcore knowledge.

## IPU Architecture: Quantitative Overview

Graphcore has shipped two generations of IPU processors plus a third-generation design in the
Bow IPU. The core architectural constants have remained stable across generations:

| Parameter | IPU Mk1 (C2) | IPU Mk2 (GC200) | Bow IPU (Bow-2000) |
|-----------|-------------|-----------------|-------------------|
| Year | 2018 | 2020 | 2022 |
| Process node | TSMC 16 nm | TSMC 7 nm | TSMC 7 nm + WoW |
| Tile count | 1,216 | 1,472 | 1,472 |
| Threads per tile | 6 | 6 | 6 |
| Total threads | 7,296 | 8,832 | 8,832 |
| Per-tile SRAM | 256 KB | 624 KB | 624 KB |
| Aggregate SRAM | 304 MB | 897 MB | 897 MB |
| Clock speed | ~1.2 GHz | ~1.325 GHz | ~1.85 GHz (boosted) |
| FP16 FLOPS | 31.1 TFLOPS | 62.4 TFLOPS | 89.6 TFLOPS |
| IPU-Links per chip | 10 | 10 | 10 |
| IPU-Link bandwidth (aggregate) | 128 GB/s | 320 GB/s | 320 GB/s |
| On-chip exchange BW | 8 TB/s | 47.5 TB/s | 47.5 TB/s |
| DRAM | None | None | None |

### Comparison With Tenstorrent and Other Architectures

| Metric | Graphcore IPU Mk2 | TT Wormhole | TT Blackhole | Cerebras WSE-2 |
|--------|------------------|------------|-------------|---------------|
| Compute units | 1,472 tiles | 80 Tensix | 140 Tensix | 850,000 cores |
| Per-unit SRAM | 624 KB | 1,464 KB | 1,536 KB | 48 KB |
| Aggregate SRAM (1 chip) | 897 MB | ~114 MB | ~210 MB | 40 GB |
| External DRAM | None | Yes | Yes | MemoryX (external) |
| Inter-chip BW (aggregate) | 320 GB/s | ~200 GB/s | Higher | N/A (on-wafer) |
| Intra-chip exchange BW | 47.5 TB/s | NOC-limited | NOC-limited | 220 Pb/s |

Three observations stand out:

1. **The IPU's per-tile SRAM (624 KB) is the closest analog to Tensix L1** among the
   architectures surveyed -- closer to Tenstorrent than to Cerebras. This means the IPU
   faces a similar per-core memory capacity, making the compiler's job more analogous.

2. **The IPU has no DRAM.** All model weights, activations, and optimizer states must fit in
   SRAM. This is a more extreme constraint than Tenstorrent (which has per-chip DRAM) and
   forces aggressive model parallelism across IPUs for large models.

3. **The intra-chip exchange bandwidth (47.5 TB/s) dwarfs inter-chip bandwidth (320 GB/s)**
   by $\sim$$148\times$. This ratio is even more extreme than TT-Metal's intra-chip NOC
   versus inter-chip Ethernet ratio, and it directly shapes the BSP model's design.

> **TT Takeaway:** The IPU's per-tile SRAM (624 KB) and Tensix L1 (1,464--1,536 KB) are in
> the same order of magnitude, making Graphcore the most directly applicable architectural
> reference for GSRP design. Both have enough local memory for meaningful working sets,
> unlike Cerebras's 48 KB per core. The key difference is that TT has DRAM as a fallback,
> allowing the GSRP to focus on communication transparency rather than capacity.

## The Bulk Synchronous Parallel (BSP) Execution Model

The IPU's defining architectural choice is the BSP execution model. Every program executes
as a sequence of **supersteps**, each consisting of three phases:

```
   +-----------+     +-----------+     +-----------+
   |  Compute  | --> | Exchange  | --> |  Barrier  | --> [Next superstep]
   +-----------+     +-----------+     +-----------+
        |                  |                 |
   All tiles           All tiles          Global
   compute on          exchange data      synchronization
   local SRAM          via on-chip        ensures all
   independently       interconnect       exchanges complete
                       and IPU-Links      before next compute
```

### Phase 1: Compute

Each tile executes its assigned computation using only its local 624 KB SRAM. There is no
remote memory access during the compute phase -- the hardware enforces this. Each tile has 6
hardware threads that share the SRAM and execute in a barrel-threaded fashion. With 1,472
tiles, this provides 8,832 concurrent threads, all operating on local data.

The compute phase duration is determined by the slowest tile. Load imbalance across tiles
directly translates to wasted compute cycles, making the compiler's partitioning decisions
critical for performance.

### Phase 2: Exchange

After all tiles complete their compute phase, the exchange phase begins:

- **Intra-chip exchange** uses a hardware crossbar/exchange fabric that provides 47.5 TB/s
  aggregate bandwidth. Every tile can simultaneously send data to and receive data from any
  other tile on the same chip.

- **Inter-chip exchange** uses IPU-Links (320 GB/s aggregate per chip) to move data between
  chips.

The exchange phase is **compiler-scheduled**: the compiler determines exactly which bytes
move from which tile to which tile, and in what order. There are no runtime decisions.

### Phase 3: Barrier Synchronization

A global barrier ensures all exchange operations have completed before the next compute phase
begins.

### BSP Cost Model

The BSP model provides a clean cost model for predicting execution time:

$$T_{\text{superstep}} = w + h \cdot g + l$$

where:
- $w$ = maximum local computation time across all tiles
- $h$ = maximum number of messages sent or received by any tile (the "h-relation")
- $g$ = the cost per message (determined by network bandwidth)
- $l$ = the barrier synchronization cost

Compute and network utilization within a superstep are:

$$\eta_{\text{compute}} = \frac{w}{w + h \cdot g + l}$$

$$\eta_{\text{network}} = \frac{h \cdot g}{w + h \cdot g + l}$$

For a superstep to be efficient, one of these must dominate. If $w \approx h \cdot g$, both
resources are at roughly 50% utilization, which is wasteful.

### Quantitative Cost of BSP Synchronization

The barrier synchronization has a non-trivial cost:

$$t_{\text{barrier}} \approx 3\text{--}5\text{ }\mu\text{s (intra-chip)}$$

For a multi-IPU system (e.g., 4 IPUs in an IPU-Machine):

$$t_{\text{barrier}} \approx 10\text{--}20\text{ }\mu\text{s (inter-chip)}$$

For a 16-IPU POD:

$$t_{\text{barrier}} \approx 30\text{--}50\text{ }\mu\text{s}$$

These barrier costs are amortized over the compute phase duration. For a compute phase of
$100\text{ }\mu\text{s}$, a $20\text{ }\mu\text{s}$ barrier represents $20\%$ overhead.
For a compute phase of $1\text{ ms}$, the overhead drops to $2\%$.

### Load Imbalance Amplification

BSP's strict phase separation means that the slowest tile determines the compute phase
duration. If tile load is imbalanced, the wasted time compounds:

$$T_{\text{waste}} = (T_{\text{max-tile}} - T_{\text{avg-tile}}) \times N_{\text{supersteps}}$$

For a workload with 1000 supersteps and 10% load imbalance, the cumulative waste is
significant. The Poplar compiler invests substantial effort in balanced partitioning to
minimize this effect -- a concern that any GSRP with epoch-based barriers would share.

### Comparison With TT-Metal Synchronization

| Mechanism | IPU BSP | TT-Metal CCL |
|----------|---------|-------------|
| Synchronization model | Global barrier per superstep | Per-operation semaphores |
| Granularity | Coarse (entire superstep) | Fine (per-collective) |
| Communication trigger | Compiler-scheduled | Explicit kernel code |
| Overlap with compute | None (phases are exclusive) | Yes (async CCL + compute) |
| Deadlock avoidance | Guaranteed (BSP structure) | Manual (dateline, virtual channels) |
| Communication pattern | Static (compile-time) | Dynamic (runtime-selectable) |

The most significant tradeoff is the **inability to overlap communication with computation**
in the standard BSP model. TT-Metal's async CCL operations (Chapter 1, Section 2) can
overlap data transfer with subsequent compute kernels, hiding communication latency.

The overlap cost is workload-dependent:

| Workload Pattern | BSP Impact | Overlap Gain (TT model) |
|-----------------|------------|-------------------------|
| Large matmul with all-gather | Low (compute dominates) | Modest (~10-20%) |
| Attention with KV cache exchange | Medium (exchange significant) | Significant (~30-50%) |
| Sparse operations with irregular exchange | High (barrier cost amplified) | Large (varies) |

> **TT Takeaway:** BSP's strict phase separation is both its greatest strength (deadlock-free
> by construction) and its greatest weakness (no compute-communication overlap). For a TT
> GSRP, a pure BSP approach would be a step backward from the current model's overlap
> capability. A better fit is a **relaxed BSP** or **epoch-based** model where compute
> is allowed to proceed on locally available data, with barriers only at points where remote
> data is actually consumed. This preserves overlap while providing BSP's deadlock guarantees.

## Compiler-Scheduled Communication: The Poplar Framework

Graphcore's Poplar compiler is responsible for converting a user's computation graph into
a BSP schedule. The programmer writes in Poplar's graph API (or through PyTorch/TensorFlow
frontends that lower to Poplar):

```cpp
// Programmer specifies computation, not communication
poplar::Tensor a = graph.addVariable(poplar::FLOAT, {1024, 1024}, "a");
poplar::Tensor b = graph.addVariable(poplar::FLOAT, {1024, 1024}, "b");
poplar::Tensor c = popops::matMul(graph, a, b, prog, "matmul");
```

The compiler then: (1) partitions the computation across tiles, (2) inserts exchange
operations wherever a tile needs data from another tile, (3) schedules the exchange
operations into exchange phases, and (4) optimizes the schedule to minimize exchange duration.

### The Programming Model Gap: Zero Lines of Communication Code

Poplar represents the maximal end of the transparency spectrum: zero lines of communication
code. The gap with TT-Metal's current 30--50 lines per cross-chip transfer is quantified in
Section 5.4.

### Communication Volume Analysis

The compiler's primary optimization objective is minimizing the **exchange volume**. For a
matrix multiplication $C = A \times B$ where $A$ is $M \times K$ and $B$ is $K \times N$
partitioned row-wise across $T$ tiles:

$$V_{\text{exchange}} = K \times N \times \text{element size}$$

The compiler explores partitioning strategies (2D tiling, output-stationary, weight-
stationary) to minimize $V_{\text{exchange}}$ while respecting the per-tile SRAM constraint
of 624 KB.

### Live Variable Tracking

One of the compiler's most challenging tasks is **live variable analysis** across supersteps.
Because there is no DRAM, every tensor that will be needed in a future superstep must be kept
alive in SRAM. The compiler tracks liveness and makes explicit decisions about:

- **Recomputation vs. storage:** If re-deriving a value is cheaper than keeping it in SRAM,
  the compiler inserts recomputation (rematerialization).
- **Spilling to remote tiles:** If a tile runs out of SRAM, the compiler "spills" data to a
  tile with available capacity, inserting exchange operations to retrieve it later.
- **Temporal scheduling:** Operations are ordered to minimize peak live memory usage.

The memory liveness analysis can be formalized: at any program point $p$, the set of live
tensors $\mathcal{L}(p)$ must satisfy:

$$\sum_{t \in \mathcal{L}(p)} \text{size}(t) \leq 624 \text{ KB} \quad \forall \text{ tiles}$$

This constraint is analogous to register allocation in a traditional compiler, but operating
at the scale of kilobytes per tile across thousands of tiles.

### Compilation Time

The comprehensive nature of the compiler's optimization leads to significant compilation
times:

| Model size | Compilation time (approximate) |
|-----------|-------------------------------|
| Small CNN (ResNet-50) | 2--5 minutes |
| Medium transformer (BERT-Large) | 10--30 minutes |
| Large LLM (GPT-3 scale) | 1--4 hours |

These compile times are comparable to Cerebras's CSoft compiler (Section 5.1) and reflect
the fundamental complexity of the communication scheduling problem.

> **TT Takeaway:** The Poplar compiler demonstrates that compiler-driven communication
> insertion is viable for production ML workloads on discrete-chip SRAM hardware. For
> Tenstorrent, the feasibility depends on whether TTNN's operation graph provides sufficient
> information to determine communication patterns (it does for standard operations) and
> whether compilation time is acceptable. Because Tensix cores have $2.5\times$ more SRAM
> than IPU tiles and TT operates at chip granularity (not tile granularity), the search space
> is orders of magnitude smaller, suggesting compile times of seconds to tens of seconds
> rather than minutes to hours.

## IPU-Link: The Inter-Chip Interconnect

IPU-Links connect multiple IPUs into larger systems. The Mk2 IPU has 10 IPU-Link ports, each
providing 32 GB/s bidirectional bandwidth. The 10 links provide 320 GB/s aggregate --
comparable to Wormhole's 16 Ethernet links at ~200 GB/s aggregate.

| Interconnect | Links per chip | Per-link BW (bidir) | Aggregate BW | Topology |
|-------------|---------------|--------------------|--------------|---------| 
| IPU Mk2 IPU-Link | 10 | 32 GB/s | 320 GB/s | All-to-all (4 IPUs) |
| TT Wormhole Ethernet | 16 | ~12.5 GB/s | ~200 GB/s | 2D mesh/torus |
| NVLink 4.0 (H100) | 18 | 50 GB/s | 900 GB/s | NVSwitch crossbar |

Graphcore systems scale through hierarchical aggregation: 4 IPUs in an IPU-Machine (3.6 GB
SRAM, all-to-all via IPU-Links), 16 IPUs in a POD16 (14.3 GB, 2-tier fabric), up to 256
IPUs in a POD256 (229 GB, multi-tier). The multi-tier topology introduces non-uniform
exchange costs, and the compiler accounts for this by preferring to place communicating
operations on IPUs within the same machine.

## BSP as a Deadlock Avoidance Strategy

One of the BSP model's most valuable properties is **guaranteed deadlock freedom**. Because
communication and computation are strictly separated into phases, and a global barrier ensures
all communication completes before the next compute phase, circular dependencies cannot form.

Consider the classic deadlock scenario in TT-Fabric (Chapter 3, Section 6): chip A sends to
chip B via an intermediate hop, while chip B simultaneously sends to chip A via the same hop.
If both paths share buffer resources, a circular wait can occur.

In BSP, this scenario is impossible:
1. All tiles stop computing simultaneously.
2. All exchange operations execute according to the compiler's static schedule.
3. The schedule is conflict-free by construction.
4. A barrier confirms completion before computation resumes.

### Quantitative Comparison of Deadlock Strategies

| Strategy | Deadlock freedom | Bandwidth overhead | Latency overhead | Complexity |
|---------|-----------------|-------------------|-----------------|-----------|
| BSP barriers (Graphcore) | Guaranteed | 0% (no extra traffic) | $10$--$50\text{ }\mu$s per barrier | Compiler |
| Dateline + VC (TT-Fabric) | Topology-dependent | ~0% | ~0 (amortized) | Firmware |
| DOR routing (Cerebras) | Guaranteed (for 2D mesh) | 0% | 0% | Hardware |
| NCCL scheduling (NVIDIA) | Software-guaranteed | 0% | Algorithm-dependent | Library |

> **TT Takeaway:** BSP's value for TT is not as a complete execution model but as a
> **deadlock avoidance strategy** for the GSRP's exchange phases. Epoch barriers at natural
> synchronization points (layer boundaries, microbatch boundaries) provide deadlock-free
> windows for compiler-scheduled cross-chip transfers, while allowing unconstrained local
> computation and communication between barriers. This hybrid approach captures BSP's safety
> guarantee without sacrificing overlap.

## The SRAM-Only Constraint and Its Consequences

The IPU's lack of DRAM is its most distinctive architectural decision. Every byte of model
state must reside in the aggregate SRAM:

| Model | Parameter size (FP16) | IPUs required (SRAM only) |
|-------|---------------------|-------------------------|
| BERT-Large (340M) | 680 MB | 1 IPU (897 MB available) |
| GPT-2 (1.5B) | 3 GB | 4 IPUs (3.6 GB available) |
| LLaMA-7B | 14 GB | 16 IPUs (14.3 GB available) |
| LLaMA-70B | 140 GB | 256 IPUs (229 GB available) |

The SRAM-only constraint forces Graphcore to use more IPUs per model than other architectures
that can offload weights to DRAM. For Tenstorrent, the presence of per-chip DRAM means that
a GSRP does not need to solve the "fit everything in SRAM" problem. The GSRP targets making
cross-chip L1 access transparent, not avoiding DRAM.

BSP also requires **double buffering** for exchanged data ($M_{\text{exchange}} = 2 \times
\sum_t \text{received data}(t)$), which is significant on a 624 KB tile. On Tensix with
1,536 KB, the relative overhead is smaller but still meaningful given existing L1 reservations
(~5.5 KB fabric overhead per core, Chapter 2.1).

## Graphcore's Software Stack and Market Lessons

Graphcore's two-layer software stack -- low-level Poplar (graph compiler API where
programmers define compute vertices and the compiler handles exchange insertion) and
high-level PopART (accepts ONNX models, generates Poplar graphs automatically) -- is
instructive for GSRP: it validates a model with a lower-level API for expert users (analogous
to TT-Metal's mesh API) and a higher-level API where the compiler handles everything
(analogous to a TTNN graph pass).

Graphcore demonstrated SRAM-only training at scale and compiler-managed communication for
standard workloads, but struggled with dynamic workloads (MoE, variable sequence lengths)
under static BSP scheduling. The commercial challenges (acquisition by SoftBank in 2024) do
not invalidate the technical lessons. For Tenstorrent, the key takeaway is the **compiler
investment required**: Graphcore invested hundreds of person-years in Poplar. A pragmatic
approach starts with enriched API primitives and progressively adds compiler passes.

## Key Takeaways

- **BSP achieves programmer-transparent communication at the cost of strict phase
  separation.** The IPU's per-tile SRAM (624 KB) is the closest analog to Tensix L1
  (1,464--1,536 KB), making Graphcore the most directly applicable architectural reference
  for GSRP design.

- **Graphcore's compiler solves the same problems GSRP would need to solve** -- partitioning,
  communication scheduling, and liveness analysis -- but TT's coarser granularity (per-chip
  vs. per-tile) reduces the problem scale and compilation time significantly.

- **BSP provides structural deadlock freedom that can be selectively adopted** as epoch
  barriers at natural synchronization points, without sacrificing TT-Metal's
  compute-communication overlap capability.

## GSRP Implications

The Graphcore IPU validates the "compiler-inserted communication" candidate architecture
(Chapter 6): a TTNN graph pass identifying cross-chip dependencies and emitting fabric
operations targeting Chapter 4's mesh API primitives. The key adaptation: unlike strict BSP,
a GSRP epoch should allow overlapping communication with computation, using barriers only to
guarantee visibility of prior-epoch transfers -- a relaxed-BSP model that preserves
TT-Metal's async advantage while borrowing BSP's deadlock-freedom guarantee.

## Further Reading

- Jia, Z., et al. (2019). "Dissecting the Graphcore IPU Architecture via Microbenchmarking."
  *arXiv:1912.03413*. Detailed microbenchmark analysis of IPU Mk1 tile architecture,
  exchange bandwidth, and synchronization costs.

- Graphcore. (2020). "Graphcore Colossus Mk2 IPU." Graphcore product documentation for the
  GC200 processor covering tile architecture, memory layout, and IPU-Link specifications.

- Luk, C. K., et al. (2020). "A Taxonomy of Deep Learning Workloads on Graphcore's IPU."
  *IEEE International Symposium on Workload Characterization*. Analyzes BSP efficiency
  across different neural network architectures.

- Valiant, L. G. (1990). "A Bridging Model for Parallel Computation." *Communications of
  the ACM*, 33(8), 103-111. The foundational BSP paper that defines the superstep model.

- Knowles, S. (2017). "Graphcore: Building the IPU." *Hot Chips 29*. Architecture
  presentation covering the IPU design philosophy.

- Graphcore. (2022). "Bow IPU Processor." Product documentation covering Wafer-on-Wafer
  packaging, clock speed improvements, and compatibility with Mk2 software stack.

---

**Next:** [`04_lessons_and_applicability.md`](./04_lessons_and_applicability.md)
