# 5.1 Cerebras Wafer-Scale Engine: The No-Boundary Ideal

## Context

Chapters 1 through 4 have established the Tenstorrent programming model, per-chip memory
architecture, inter-chip fabric, and existing building blocks. The central challenge this guide
addresses -- making cross-chip SRAM access transparent -- arises precisely because chip
boundaries exist. Cerebras Systems offers the most radical counterfactual: a processor where
chip boundaries do not exist at all. Their Wafer-Scale Engine (WSE) places hundreds of
thousands of cores on a single silicon wafer, interconnected by a hardware-routed 2D mesh
with no off-die links. This section provides a quantitative examination of the WSE
architecture, extracts the design principles that enable its "zero-overhead global SRAM"
property, and evaluates which lessons are transferable to a system that must span chip
boundaries via Ethernet.

The reader should already be familiar with TT-Metal's dual-NOC interconnect and L1 memory
layout (Chapter 2) and the packet-hopping inter-chip fabric (Chapter 3). This section does
not assume prior knowledge of Cerebras hardware.

## WSE Architecture: Three Generations at a Glance

Cerebras has shipped three generations of wafer-scale processors. The quantitative progression
illustrates both the scaling trajectory and the architectural constants that underpin the
design.

| Parameter | WSE-1 (2019) | WSE-2 (2021) | WSE-3 (2024) |
|-----------|-------------|-------------|-------------|
| Process node | TSMC 16 nm | TSMC 7 nm | TSMC 5 nm |
| Die area | 46,225 mm$^2$ | 46,225 mm$^2$ | 46,225 mm$^2$ |
| Core count | 400,000 | 850,000 | 900,000 |
| On-wafer SRAM | 18 GB | 40 GB | 44 GB |
| SRAM per core | ~45 KB | ~48 KB | ~49 KB |
| Fabric bandwidth (aggregate) | 100 Pb/s | 220 Pb/s | 214 Pb/s |
| Memory bandwidth (to external) | 9.6 TB/s | 20 TB/s | 21 TB/s |
| Interconnect links per core | 4 (N/S/E/W) | 4 (N/S/E/W) | 4 (N/S/E/W) |
| Transistor count | 1.2 T | 2.6 T | 4 T |

All three generations share the same physical constraint: the entire wafer is $300\text{ mm}$
in diameter ($\sim$46,225 mm$^2$ usable area). Each generation packs more cores and more SRAM
per unit area by leveraging a smaller process node while keeping the wafer size constant.

### Comparison With Tenstorrent Scale

To appreciate what these numbers mean for the GSRP question, consider the aggregate SRAM
available in a comparable Tenstorrent multi-chip deployment:

| Configuration | Core count | Aggregate L1 SRAM | Interconnect type |
|---------------|-----------|-------------------|-------------------|
| Cerebras WSE-3 | 900,000 | 44 GB | On-wafer 2D mesh |
| TT Wormhole 1-chip | 80 Tensix | ~114 MB usable | Intra-chip NOC |
| TT Wormhole T3K (8 chips) | 640 Tensix | ~914 MB usable | Ethernet (16 links/chip) |
| TT Blackhole 1-chip | 140 Tensix | ~210 MB usable | Intra-chip NOC |
| TT Blackhole Galaxy (32 chips) | 4,480 Tensix | ~6.7 GB usable | Ethernet |

The WSE-3's 44 GB of on-wafer SRAM is roughly $6.5\times$ the aggregate L1 of a 32-chip
Blackhole Galaxy system. More critically, the WSE accesses all of that SRAM through a
homogeneous on-die interconnect, while the Galaxy must traverse Ethernet links whose
per-link bandwidth ($\sim$12.5 GB/s for Wormhole, higher for Blackhole) is orders of
magnitude lower than the per-core fabric bandwidth on the WSE.

## The 2D Mesh Interconnect

Each WSE core connects to its four cardinal neighbors (north, south, east, west) via
dedicated hardware links. The aggregate on-wafer bandwidth is staggering -- 220 Pb/s for
WSE-2 -- but the per-link number is what matters for latency analysis.

### Per-Core Bandwidth Calculation

With 850,000 cores on WSE-2, each having 4 bidirectional links:

$$B_{\text{per-link}} = \frac{220 \times 10^{15} \text{ B/s}}{850{,}000 \times 4 \times 2} \approx 32.4 \text{ GB/s per direction per link}$$

This per-link bandwidth is roughly $2.6\times$ the per-Ethernet-link bandwidth on Wormhole
($\sim$12.5 GB/s) -- a meaningful but not overwhelming advantage on a per-link basis. The
real advantage is structural: every core has these links, there are no bottleneck points
where traffic must funnel through a small number of Ethernet cores, and the links are on-die
with sub-microsecond latency.

> **TT Takeaway:** The WSE's per-link bandwidth advantage over TT Ethernet ($2.6\times$) is
> smaller than commonly assumed. The decisive advantage is not raw bandwidth but the absence
> of chip boundaries -- every core connects to every neighbor at silicon speed, with no
> serialization overhead, no Ethernet framing, and no software-managed credit flow. A GSRP
> cannot replicate this physical property but can approximate the programming convenience by
> automating the packet construction and routing that the WSE handles in hardware.

### Latency Characteristics

On-wafer communication latency is governed by the number of mesh hops. For a 2D mesh with
$N$ cores arranged in a $\sqrt{N} \times \sqrt{N}$ grid:

- **WSE-2** ($\sqrt{850{,}000} \approx 922$): maximum Manhattan distance $\approx 1{,}844$
  hops. At single-cycle-per-hop forwarding, this translates to $\sim 1{,}844$ cycles. At a
  core clock of $\sim$850 MHz, maximum cross-wafer latency is approximately:

$$t_{\text{max}} = \frac{1{,}844}{850 \times 10^6} \approx 2.2 \text{ } \mu\text{s}$$

- **Tenstorrent T3K** (8 Wormhole chips in a 2$\times$4 arrangement): a cross-system
  transfer requires traversing up to 3 Ethernet hops. Each hop involves store-and-forward
  through an ERISC router. With measured bandwidth of $\sim$6.4 B/cycle for 4096 B packets
  over an 8-chip half-ring (from TT-Metal benchmark golden data), the per-hop latency for
  a 4 KB payload is:

$$t_{\text{hop}} \approx \frac{4096}{6.4 \times 10^9 / f_{\text{clk}}} \text{ cycles}$$

  With additional routing overhead, typical inter-chip latencies are in the range of
  $5$--$20\text{ }\mu\text{s}$ per hop depending on payload size and congestion.

The key quantitative insight: WSE's worst-case cross-wafer latency ($\sim$2 $\mu$s) is
comparable to or better than a single Ethernet hop on Tenstorrent hardware. This is the
fundamental reason why Cerebras can treat all SRAM as uniformly accessible without
performance cliffs.

### The Latency Cliff Problem

The key architectural difference is not the absolute latency but the **discontinuity**. On
the WSE, latency scales linearly and predictably with Manhattan distance:

$$t_{\text{access}}(d) = t_{\text{local}} + d \times t_{\text{hop}}$$

There is no point where latency jumps by an order of magnitude. On a multi-chip TT system,
every chip boundary introduces a latency cliff:

$$\frac{t_{\text{inter-chip}}}{t_{\text{intra-chip}}} \approx 10\text{x} - 100\text{x}$$

This ratio makes transparent remote access fundamentally different from local access. A GSRP
that hides this discontinuity behind a uniform API risks creating performance surprises.

> **TT Takeaway:** The WSE's uniform latency is its greatest advantage for transparent global
> SRAM, and it is the one advantage that a multi-chip GSRP fundamentally cannot replicate.
> This argues against a purely transparent model on TT hardware and in favor of a
> **latency-aware** model -- one where the programmer or compiler has visibility into whether
> an access is local or remote, even if the mechanics of the remote access are automated. The
> mesh API's existing requirement that callers supply `dst_dev_id` and `dst_mesh_id` may
> actually be a feature, not a limitation.

### Routing Model

The WSE uses hardware dimension-ordered routing (DOR). Each core's router inspects the
destination coordinates and forwards the packet first along one axis (e.g., X) until
aligned, then along the other axis (Y). This is implemented entirely in hardware -- no
firmware, no software intervention, no routing tables in SRAM.

Contrast with TT-Fabric's source routing model (Chapter 3, Section 3): the entire route is
encoded in the packet header as a sequence of per-hop direction entries. This makes
TT-Fabric more flexible (arbitrary routes, multicast branching, 2D turn points) but imposes
per-packet overhead for route computation and consumes packet header space proportional to
hop count.

| Routing property | Cerebras WSE | TT-Fabric |
|-----------------|-------------|-----------|
| Routing decision location | Each hop (HW) | Source (SW/FW) |
| Route computation cost | Zero (combinational logic) | L1 table lookup at source |
| Header overhead per hop | None | 4 bits (2D) or 2 bits (1D) |
| Maximum hops | ~1,844 (WSE-2) | 63 (1D) / 67 (2D) |
| Multicast support | Row/column broadcast | Flexible 2D multicast ranges |
| Deadlock avoidance | DOR guarantees | Dateline + virtual channels |

## Per-Core SRAM: 48 KB and Its Implications

Each WSE-2 core has approximately 48 KB of local SRAM. This is a significant constraint --
it is roughly $30\times$ smaller than a Tensix core's L1 on Wormhole (1,464 KB) or Blackhole
(1,536 KB). The architectural choice is deliberate: Cerebras prioritizes core count and
aggregate bandwidth over per-core memory capacity.

### Memory Density Comparison

| Architecture | Per-core SRAM | Core count (single device) | Aggregate SRAM |
|-------------|--------------|--------------------------|----------------|
| Cerebras WSE-2 | 48 KB | 850,000 | 40 GB |
| Cerebras WSE-3 | ~49 KB | 900,000 | 44 GB |
| TT Wormhole | 1,464 KB | 80 Tensix | ~114 MB |
| TT Blackhole | 1,536 KB | 140 Tensix | ~210 MB |
| NVIDIA H100 (SM) | 256 KB shared mem | 132 SMs | ~33 MB (shared) |
| NVIDIA H100 (L2) | -- | -- | 50 MB (global) |

The 48 KB per-core constraint means that most tensors do not fit on a single core. Cerebras's
compiler must decompose every operation across many cores, with intermediate results flowing
through the mesh. The programming model is fundamentally **dataflow**: the compiler maps a
computational graph onto physical cores, with each core executing one or a few operations and
forwarding results to downstream cores via the mesh.

This is a critical difference from TT-Metal's model, where a Tensix core has enough L1 to
hold meaningful working sets (entire tiles, multiple circular buffer slots) and perform
multi-step computations locally before communicating results.

> **TT Takeaway:** Tenstorrent's $30\times$ larger per-core SRAM (1,536 KB vs. 48 KB) means
> that a GSRP could be designed around a model where most accesses are local, with remote
> accesses being relatively infrequent and tolerable at higher latency. This "mostly-local
> with efficient remote fallback" model is fundamentally different from the WSE's "uniformly
> distributed" model and may be better suited to TT's hardware constraints.

## Compiler Complexity: The Hidden Cost of Transparency

The WSE achieves programmer-transparent communication at the cost of compiler complexity.
The Cerebras Software Platform (CSoft) compiler must solve several hard problems:

1. **Graph partitioning.** Decompose the computational graph into sub-graphs that fit within
   48 KB per core. For a large language model, this means distributing attention heads,
   feed-forward layers, and embedding tables across hundreds of thousands of cores.

2. **Placement.** Assign each sub-graph to a physical core location on the wafer such that
   communication between connected sub-graphs is local (minimizing hop count). This is a
   variant of the quadratic assignment problem -- NP-hard in general.

3. **Routing.** Allocate bandwidth on mesh links for all inter-core data flows without
   creating congestion hot spots. With 850,000+ cores and potentially millions of data flow
   edges, this is a massive constraint satisfaction problem.

4. **Scheduling.** Order operations across cores to ensure data dependencies are satisfied
   while maximizing pipeline utilization. Since there is no explicit synchronization
   primitive (no barriers, no semaphores), the compiler must statically guarantee that
   data arrives before it is consumed.

5. **Memory allocation.** Pack the 48 KB per core with input buffers, output buffers,
   weights, and intermediate values. Buffer overflows require spilling to neighboring cores,
   which introduces additional communication.

The compilation time for large models can be substantial -- reportedly hours for full-scale
LLM mappings. This is the price of transparency: the communication complexity that
programmers avoid is absorbed by the compiler.

> **TT Takeaway:** If Tenstorrent pursued a compiler-driven approach to GSRP, the Cerebras
> experience suggests that the compiler complexity is substantial but tractable for regular
> workloads. The key difference is that a TT compiler would additionally need to reason about
> the latency asymmetry between intra-chip and inter-chip transfers -- a dimension that the
> Cerebras compiler does not face. Tenstorrent's coarser granularity (per-chip, not
> per-core) would reduce the search space by orders of magnitude compared to Cerebras.

## Congestion Management and Bandwidth Allocation

Even though the WSE's mesh interconnect is hardware-managed, congestion remains a first-order
concern. The 2D mesh topology has a fundamental bandwidth limitation: the bisection bandwidth
scales as $O(\sqrt{N})$ where $N$ is the number of cores, while many ML communication
patterns (all-reduce, all-to-all) generate $O(N)$ traffic.

The aggregate bandwidth of the central links is:

$$B_{\text{bisection}} = \sqrt{N} \times B_{\text{link}}$$

while the total traffic volume for an all-to-all of $S$ bytes per core is:

$$V_{\text{total}} = N \times S$$

The time to complete the all-to-all is bounded by the bisection bandwidth:

$$T_{\text{all-to-all}} \geq \frac{N \times S}{\sqrt{N} \times B_{\text{link}}} = \sqrt{N} \times \frac{S}{B_{\text{link}}}$$

This $\sqrt{N}$ scaling is a fundamental limitation of 2D mesh topologies, shared by
Tenstorrent's multi-chip mesh.

The Cerebras compiler mitigates congestion through: (1) locality-aware placement that
co-locates communicating operations on nearby cores, (2) dimension-ordered routing that
creates predictable flow patterns, (3) traffic shaping that staggers communication to
prevent simultaneous sends, and (4) hierarchical reduction trees where cores reduce locally
with neighbors first.

These strategies are directly applicable to TT's GSRP. The existing 2D routing in
`HybridMeshPacketHeader` with per-hop direction encoding parallels the WSE's dimension-ordered
routing, and the `ccl_command_stream` pipeline supports staggered collective execution.

> **TT Takeaway:** Even with hardware-transparent routing, the WSE compiler must actively
> manage traffic patterns to avoid mesh bisection bottlenecks. For TT's GSRP, this means
> that the runtime or compiler cannot simply fire off remote accesses and hope the fabric
> handles congestion -- it must schedule transfers to avoid overloading Ethernet links, which
> are the mesh bisection bandwidth bottleneck. The existing fabric flow control (Chapter 3.6)
> provides reactive backpressure, but proactive traffic scheduling would deliver better
> performance.

## WSE External Memory and the MemoryX Subsystem

One often-overlooked aspect of the WSE architecture is that 40--44 GB of on-wafer SRAM,
while large by accelerator standards, is not sufficient for the full parameter set of modern
LLMs. A 70B-parameter model in FP16 requires $\sim$140 GB, far exceeding on-wafer capacity.

Cerebras addresses this with the MemoryX subsystem -- an external memory fabric that streams
model weights onto the wafer:

- **MemoryX nodes** contain high-bandwidth DRAM and connect to the WSE via high-speed
  off-wafer links.
- **SwarmX** is the interconnect fabric between MemoryX nodes and the WSE, providing
  bandwidth-matched streaming of weights to the wafer edge.
- The wafer edge cores act as I/O interfaces, receiving weight tiles from MemoryX and
  distributing them across the wafer mesh.

This transforms the WSE from a "pure global SRAM" system into a **hierarchical memory
system** with two tiers:

| Memory tier | Cerebras | Tenstorrent (32-chip BH) |
|------------|---------|-------------------------|
| Fast SRAM | 44 GB (on-wafer, uniform) | ~6.7 GB (L1, per-chip) |
| Slow DRAM | MemoryX (TB-scale, off-wafer) | Per-chip DRAM (varies by board) |
| SRAM access latency | $< 3\text{ }\mu\text{s}$ worst case | $< 100\text{ ns}$ local, $5$--$20\text{ }\mu\text{s}$ remote |
| DRAM access latency | Varies (off-wafer streaming) | ~hundreds of ns (local DRAM) |

The fact that even Cerebras -- with 44 GB of on-wafer SRAM -- still requires an external
memory hierarchy reinforces that a GSRP for Tenstorrent would operate within similar memory
tier constraints. The GSRP would pool L1 across chips ($\sim$6.7 GB for a 32-chip BH system)
to reduce explicit communication, not to eliminate the need for DRAM.

> **TT Takeaway:** Cerebras's weight streaming demonstrates that even a system designed around
> global SRAM needs a hierarchical memory strategy when models exceed SRAM capacity. For TT's
> GSRP, the hierarchy is L1 (local) $\to$ L1 (remote, via fabric) $\to$ DRAM (local) $\to$
> DRAM (remote). The GSRP should provide primitives for prefetching across this hierarchy:
> `prefetch_remote_l1(global_addr, local_buf, size)` that triggers an asynchronous fabric
> transfer, allowing compute to proceed on previously-prefetched data.

## Yield and Fault Tolerance

A full wafer at advanced process nodes will inevitably contain defective cores. Cerebras
addresses this with redundancy: the wafer is manufactured with more cores than are needed, and
the compiler maps around defective regions. The routing hardware can forward packets around
dead cores.

This fault tolerance is native to the wafer-scale approach and does not apply directly to
Tenstorrent's multi-chip model, where each chip is a known-good die. However, the concept of
**routing around unavailable resources** has an analog: in a TT-Fabric mesh, if one Ethernet
link fails or is congested, the routing tables could be regenerated to use alternative paths.
The existing `ControlPlane` routing table generator (Chapter 3, Section 4) already performs
BFS-based shortest-path computation and could be extended with link-failure awareness.

## Key Takeaways

- **Cerebras achieves global SRAM transparency through physical elimination of chip
  boundaries.** Ethernet interconnects cannot replicate the WSE's $\sim$32 GB/s per-link,
  $< 3\text{ }\mu\text{s}$ worst-case latency properties -- this establishes the theoretical
  upper bound on GSRP transparency that a multi-chip system cannot reach.

- **Tenstorrent's $30\times$ larger per-core SRAM** enables a "mostly-local with efficient
  remote fallback" model rather than the fine-grained dataflow model that 48 KB per core
  forces. Even Cerebras requires external memory (MemoryX) for large models, confirming
  that a GSRP pooling ~6.7 GB of L1 would supplement, not replace, the DRAM tier.

- **The WSE's compiler-driven communication** and congestion management demonstrate that
  software must actively schedule traffic even with hardware-transparent routing -- a lesson
  that applies doubly to TT's bandwidth-constrained Ethernet fabric.

## GSRP Implications

The WSE establishes that a Tenstorrent GSRP must provide **mechanical transparency**
(automating packet construction, routing, and flow control) while preserving **performance
transparency** (making chip-boundary crossings visible). The Cerebras compiler's static
graph decomposition suggests a lighter-weight analog for TTNN: a graph pass inserting fabric
operations at shard boundaries, targeting the mesh API primitives from Chapter 4.

## Further Reading

- Lauterbach, G. (2021). "The Path to a Wafer-Scale Processor." *IEEE Micro*, 41(4), 52-59.
  Covers the engineering challenges of wafer-scale integration including yield, power
  delivery, and cooling.

- Rocki, K., et al. (2020). "Fast Stencil-Code Computation on a Wafer-Scale Processor."
  *SC20: International Conference for High Performance Computing*. Demonstrates the
  dataflow mapping of stencil computations onto the WSE.

- Cerebras Systems. (2024). "Cerebras WSE-3 Technical Overview." Cerebras product
  documentation covering the third-generation wafer-scale engine specifications.

- Lie, S. (2022). "Cerebras Architecture Deep Dive: First Look Inside the Hardware/Software
  Co-Design for Deep Learning." *IEEE Micro*, 42(3). Discusses the co-design of the WSE
  hardware and CSoft compiler.

- Selig, J., et al. (2022). "CEREBRAS-GPT: Open Compute-Optimal Language Models Trained on
  the Cerebras Wafer-Scale Cluster." Demonstrates scaling behavior on WSE-based clusters.

- Selig, J., et al. (2023). "Weight Streaming: Unlocking Large Models on CS-2." Cerebras
  technical report covering the MemoryX/SwarmX external memory architecture.

---

**Next:** [`02_nvidia_nvlink_nccl.md`](./02_nvidia_nvlink_nccl.md)
