# 5.2 NVIDIA NVLink, NVSwitch, and NCCL: High-Bandwidth Links With Explicit Software

## Context

The previous section examined Cerebras's wafer-scale approach, where chip boundaries are
eliminated entirely. This section examines the opposite end of the spectrum: NVIDIA's GPU
ecosystem, where chip boundaries are prominent but bridged by progressively higher-bandwidth
hardware links. Despite massive investment in interconnect hardware -- NVLink, NVSwitch,
NVLink-C2C -- NVIDIA's collective communication library (NCCL) remains explicit software
that programmers or frameworks must invoke. This persistence of explicit collectives, even
with the industry's highest interconnect bandwidth, carries direct lessons for whether a
Tenstorrent GSRP can realistically eliminate explicit CCL kernels.

The reader should be familiar with the TT-Metal CCL programming model (Chapter 1) and the
inter-chip fabric architecture (Chapter 3). This section does not assume deep NVIDIA GPU
knowledge beyond general awareness of the CUDA programming model.

## NVLink Evolution: A Quantitative History

NVIDIA has invested five generations of hardware into GPU-to-GPU interconnects. The bandwidth
progression tells a story of relentless scaling:

| Generation | Year | Per-link BW (bidirectional) | Links per GPU | Aggregate per GPU | Signaling |
|-----------|------|---------------------------|--------------|-------------------|-----------|
| NVLink 1.0 | 2016 (P100) | 40 GB/s | 4 | 160 GB/s | NVLink |
| NVLink 2.0 | 2017 (V100) | 50 GB/s | 6 | 300 GB/s | NVLink |
| NVLink 3.0 | 2020 (A100) | 50 GB/s | 12 | 600 GB/s | NVLink |
| NVLink 4.0 | 2022 (H100) | 50 GB/s | 18 | 900 GB/s | NVLink |
| NVLink 5.0 | 2024 (B200) | 100 GB/s | 18 | 1,800 GB/s | NVLink-C2C |

NVLink 5.0 doubles the per-link bandwidth to 100 GB/s bidirectional, yielding 1,800 GB/s
aggregate per GPU (or per die in the dual-die B200 configuration). The H100's 18 links at
50 GB/s provide 900 GB/s aggregate -- roughly $72\times$ the per-link bandwidth of a single
Wormhole Ethernet link ($\sim$12.5 GB/s) and $4.5\times$ the aggregate bandwidth of all 16
Wormhole Ethernet links combined ($\sim$200 GB/s bidirectional).

### Bandwidth Context With Tenstorrent

| Interconnect | Per-link BW | Links per device | Aggregate BW | Latency |
|-------------|------------|-----------------|-------------|---------|
| NVLink 4.0 (H100) | 50 GB/s | 18 | 900 GB/s | ~1 $\mu$s |
| NVLink 5.0 (B200) | 100 GB/s | 18/die (36 total) | 1,800 GB/s (chip) | ~1 $\mu$s |
| TT Wormhole Ethernet | ~12.5 GB/s | 16 | ~200 GB/s | $5$--$20\text{ }\mu$s |
| TT Blackhole Ethernet | Higher (est.) | TBD | Higher | TBD |
| Cerebras on-wafer | ~32 GB/s/link | 4/core $\times$ 850K | 220 Pb/s | $< 3\text{ }\mu$s |

> **TT Takeaway:** Even with NVLink 5.0 providing 1,800 GB/s per GPU -- roughly $9\times$
> Wormhole's aggregate Ethernet bandwidth -- NVIDIA has not eliminated explicit collectives.
> This is the strongest industry evidence that bandwidth alone does not make transparent
> communication viable. The persistence of NCCL reflects fundamental algorithmic advantages
> of explicit orchestration that no amount of bandwidth can replace.

## NVSwitch: All-to-All Connectivity

NVSwitch is a dedicated switch chip that provides all-to-all NVLink connectivity within a
node. Without NVSwitch, GPUs must route traffic through intermediate GPUs (multi-hop), which
halves effective bandwidth and introduces latency. NVSwitch eliminates this by providing
direct crossbar connectivity.

### NVSwitch Generations

| Generation | Year | Ports | Per-port BW | Aggregate BW | GPUs supported |
|-----------|------|-------|------------|-------------|---------------|
| NVSwitch 1.0 | 2018 (DGX-2) | 18 | 50 GB/s | 900 GB/s | 16 (V100) |
| NVSwitch 2.0 | 2020 (DGX A100) | 36 | 50 GB/s | 1,800 GB/s | 8 (A100) |
| NVSwitch 3.0 | 2022 (DGX H100) | 64 | 50 GB/s | 3,200 GB/s | 8 (H100) |
| NVSwitch 4.0 | 2024 (GB200 NVL72) | 128 | 100 GB/s | 12,800 GB/s | 72 (B200 dies) |

The NVSwitch 4.0 in the GB200 NVL72 system connects 72 Blackwell GPU dies (36 B200 chips,
each containing 2 dies) with all-to-all bandwidth, creating a single NVLink domain of
$72 \times 1{,}800 = 129.6$ TB/s aggregate bisection bandwidth.

### The NVL72 Scaling Inflection Point

The NVL72 represents a qualitative shift: 72 GPU dies in a single all-to-all domain is
large enough to run most training workloads without inter-node communication. This changes
the communication problem from "how to communicate across bandwidth tiers" to "how to
communicate efficiently within a very large flat domain." The inflection point is that at 72
dies, the crossbar's $O(N)$ bisection bandwidth begins to encounter practical limits --
further scaling requires multi-level NVSwitch hierarchies or inter-node interconnects.

For TT-Fabric's 2D mesh topology, the analogous inflection occurs earlier. The mesh's
$O(\sqrt{N})$ bisection bandwidth means that at 32 chips (a $4 \times 8$ mesh), the
bisection is only $\sqrt{32} \approx 5.7$ links wide, providing $\sim$71 GB/s bisection
bandwidth. Congestion amplification in the mesh center becomes a first-order concern for
all-to-all patterns.

### Comparison With TT-Fabric Topology

| Property | NVSwitch (crossbar) | TT-Fabric (2D mesh) |
|---------|-------------------|-------------------|
| Any-to-any latency | 1 hop (uniform) | 1--$N$ hops (distance-dependent) |
| Bisection bandwidth | Full crossbar ($N \times B_{\text{link}}$) | $O(\sqrt{N} \times B_{\text{link}})$ |
| Scalability | Limited by switch radix | Scales with mesh dimension |
| Cost per endpoint | High (dedicated switch ASIC) | Low (Ethernet on each chip) |
| Multi-hop overhead | None | Store-and-forward per hop |

## NCCL: The Persistence of Explicit Collectives

Despite the NVLink/NVSwitch hardware investment, NVIDIA's collective communication library
(NCCL) remains an explicit software layer. Programmers (or frameworks like PyTorch, which
calls NCCL through `torch.distributed`) must explicitly invoke collectives:

```python
# Explicit all-reduce in PyTorch using NCCL backend
torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
```

NCCL implements ring, tree, and collnet (in-network) algorithms, choosing among them based
on message size, GPU count, and topology.

### Why Explicit Collectives Persist

The persistence of NCCL despite massive hardware investment is instructive. Several factors
contribute:

**1. Performance predictability.** Explicit collectives allow the programmer (or framework)
to pipeline communication with computation. NCCL's `ncclGroupStart`/`ncclGroupEnd` API
enables fusing multiple collectives into a single operation. Transparent remote memory
access would sacrifice this control.

**2. Algorithm diversity.** Different collective patterns have different optimal algorithms
depending on message size:

| Message size | Preferred algorithm | Bandwidth utilization | Latency |
|-------------|--------------------|-----------------------|---------|
| $< 256$ KB | Tree (recursive halving-doubling) | Moderate | $O(\log N)$ |
| $256$ KB -- $4$ MB | Hybrid | Blends latency and BW | Mixed |
| $> 4$ MB | Ring or double-ring | Near-optimal | $O(N)$ |

No single transparent mechanism can match this adaptive behavior. A runtime that
transparently routes individual cache lines or pages would default to a "naive" algorithm
equivalent to direct send for all sizes.

**3. Topology awareness.** NCCL constructs rings and trees that respect the physical
topology -- ensuring that NVLink-connected GPUs communicate directly while InfiniBand-
connected GPUs use different algorithms. This topology-aware scheduling is inherently an
optimization problem that requires global knowledge.

**4. Overlap with computation.** NCCL operations are asynchronous with respect to compute
streams. Frameworks issue collectives early and overlap the data transfer with subsequent
layers' computation. This overlap is the primary mechanism for hiding communication latency
in large-scale training and would be lost with synchronous remote memory access.

> **TT Takeaway:** The persistence of NCCL despite NVLink's transparent access capability
> (GPUDirect P2P load/store) is the strongest industry evidence that explicit collective
> communication provides performance benefits that transparent access cannot match. For TT's
> GSRP, this argues for a model where the common collective patterns (all-gather,
> reduce-scatter, all-reduce) remain highly optimized library operations, while the GSRP
> focuses on automating the **boilerplate** -- packet construction, route computation,
> connection management -- rather than eliminating explicit communication entirely.

### NCCL Performance Characteristics

For a DGX H100 system (8 GPUs, NVSwitch):

| Collective | 1 MB message | 128 MB message | 1 GB message |
|-----------|-------------|---------------|-------------|
| AllReduce (bus BW) | ~200 GB/s | ~430 GB/s | ~450 GB/s |
| AllGather (bus BW) | ~250 GB/s | ~500 GB/s | ~520 GB/s |
| ReduceScatter (bus BW) | ~240 GB/s | ~480 GB/s | ~500 GB/s |

These bus bandwidth numbers approach the theoretical NVLink peak (900 GB/s) for large
messages. The gap between 1 MB and 1 GB reflects the latency overhead that dominates small
transfers -- the same overhead that would affect every GSRP access.

### Message Size Sensitivity

NCCL's efficiency degrades for small messages due to per-collective startup cost. The
efficiency as a function of message size can be approximated as:

$$\eta(S) = \frac{S}{S + S_0}$$

where $S_0$ is the effective startup overhead (in bytes-equivalent). For NVLink collectives,
$S_0 \approx 256\text{ KB}$, meaning messages below 256 KB achieve less than 50% bandwidth
efficiency. For TT's Ethernet fabric with higher per-operation overhead, $S_0$ would be
correspondingly larger, making the case for bulk transfers even stronger.

## NVSwitch In-Network Reduction: Hardware-Assisted Collectives

Starting with NVSwitch 3.0 (H100), NVIDIA introduced in-network reduction (SHARP/NVLS),
enabling the switch itself to perform reduction operations. Rather than implementing
all-reduce as a multi-step ring algorithm, in-network reduction allows:

1. Each GPU sends its partial gradient to the NVSwitch.
2. The NVSwitch performs the reduction in hardware (FP32/FP16/BF16 addition).
3. The NVSwitch broadcasts the reduced result back to all GPUs.

This reduces all-reduce latency from $O(N)$ to $O(1)$ for the network traversal component.
For 8 GPUs:

$$\text{Ring all-reduce data volume per GPU} = 2 \times \frac{7}{8} \times M = 1.75M$$
$$\text{SHARP all-reduce data volume per GPU} = 2M$$

The per-GPU data volume is slightly higher with SHARP for small GPU counts, but the latency
is dramatically lower because only a single round trip is needed instead of $2(N-1) = 14$
steps. For larger GPU counts ($N > 8$), the $O(1)$ latency advantage dominates.

### Relevance to TT-Fabric

TT-Fabric's current architecture does not include in-network computation capability. The
ERISC routers perform store-and-forward but do not modify packet payloads. Adding in-network
reduction would require extending ERISC firmware to perform arithmetic on forwarded packets.
The ERISC cores have RISC-V processors and local SRAM (256 KB on Wormhole, 512 KB on
Blackhole) that could accumulate partial reductions before forwarding.

> **TT Takeaway:** In-network reduction on ERISC is a lower-risk enhancement than a full
> GSRP -- it does not require changes to the programming model or address space, only
> firmware-level logic in the EDM forwarding path. It could serve as a stepping stone:
> demonstrating that ERISC cores can do more than store-and-forward, building the foundation
> for more sophisticated runtime-managed communication.

## CUDA Unified Memory: Transparent Remote Access and Its Costs

CUDA Unified Memory (UM) provides a page-migration-based mechanism for transparent access to
memory across GPUs and between GPU and host CPU:

```cpp
// Allocate unified memory accessible from any GPU or CPU
cudaMallocManaged(&ptr, size);
// Access from any GPU -- page faults trigger migration
kernel<<<grid, block>>>(ptr);
```

When a GPU accesses a page that resides on another GPU (or the CPU), a page fault is
generated, the page is migrated to the accessing GPU, and the access is retried.

### Performance Reality

The transparency comes with severe performance penalties:

| Access pattern | Unified Memory BW | Explicit cudaMemcpy BW | Ratio |
|---------------|-------------------|----------------------|-------|
| Sequential (large) | ~40 GB/s (H100 NVLink) | ~450 GB/s (NVLink) | 0.09x |
| Random (page faults) | ~2 GB/s | N/A (not applicable) | -- |
| First-touch (cold) | ~5 GB/s | ~450 GB/s (pre-staged) | 0.01x |

The $11\times$ bandwidth penalty for sequential access and $>100\times$ penalty for random
access make Unified Memory unsuitable for performance-critical paths. The overhead comes from
page fault latency ($\sim$20--50 $\mu$s per fault), page granularity mismatch (4 KB or 64 KB
pages for 4-byte accesses), thrashing under multi-GPU contention, and TLB pressure from
large unified allocations.

### The Hybrid Model: Prefetch Hints

The most effective use of CUDA UVM in practice relies heavily on
`cudaMemPrefetchAsync` -- programmer-inserted hints that trigger page migration before the
data is needed. This creates a **hybrid model** that is nominally transparent but practically
explicit.

| Access Pattern | UVM Performance vs. Explicit | Root Cause |
|---------------|------------------------------|------------|
| Sequential, large blocks | 70-90% of explicit | Page prefetch effective |
| Strided, regular | 40-70% of explicit | Partial page utilization, TLB pressure |
| Random, fine-grained | 10-30% of explicit | Page fault storms, thrashing |
| Read-mostly, replicated | 80-95% of explicit | Read duplication effective |

> **TT Takeaway:** CUDA UVM's experience strongly validates the hybrid approach for TT's
> GSRP. Pure transparency degrades under irregular access; pure explicitness requires too
> much programmer effort. The sweet spot is a system that automates the mechanics (addressing,
> routing, flow control) while allowing or requiring hints about data movement timing.
> Extending `MeshBuffer` with prefetch semantics -- `prefetch_remote(global_addr, size)` --
> would follow NVIDIA's validated pattern.

### Direct Analogy to GSRP

CUDA Unified Memory is the closest existing analogy to what a fully transparent GSRP would
look like on Tenstorrent hardware. The lessons are cautionary:

| UM property | GSRP analog | Risk level |
|-----------|------------|-----------|
| Page fault on remote access | Trap on out-of-range NOC address | High latency |
| Page migration | L1-to-L1 data transfer via fabric | Bandwidth waste |
| 4 KB page granularity | Ethernet minimum packet (16 B) to L1 page | Granularity mismatch |
| Thrashing | Ping-pong between chips | Bandwidth collapse |
| TLB pressure | Address translation table in L1 | L1 capacity loss |

## GPUDirect P2P: Hardware-Transparent Remote Memory

GPUDirect Peer-to-Peer allows one GPU to directly access another GPU's memory over NVLink
without staging through CPU memory. The accessing GPU issues a load/store that is translated
to a remote memory transaction by the NVLink hardware:

| Access Type | Latency | Bandwidth |
|------------|---------|-----------|
| Local HBM | ~300-600 ns | ~3.3 TB/s (H100) |
| Remote P2P via NVLink | ~1-2 us | ~450 GB/s (H100, per direction) |
| Remote via PCIe Gen5 | ~5-10 us | ~64 GB/s |

Despite the availability of transparent P2P access, NCCL does not use individual P2P
loads/stores for collective operations. Bulk DMA transfers achieve near-peak bandwidth while
P2P loads achieve only a fraction, due to coalescing inefficiency, TLB pressure, and weak
ordering guarantees.

> **TT Takeaway:** GPUDirect P2P demonstrates that hardware-transparent remote memory access
> and explicit bulk transfers can coexist in the same system, serving different use cases.
> For TT's GSRP, this suggests a **dual-mode design**: (1) a "bulk mode" for collective-style
> transfers where the runtime pre-computes routes and uses large DMA-style fabric transfers,
> and (2) a "scalar mode" for occasional fine-grained remote access (e.g., reading a single
> control word from a remote chip) where simplicity outweighs the performance overhead. The
> existing fabric API (Chapter 4.1) already provides bulk mode; the gap is providing a
> convenient scalar mode without requiring full packet header construction for small transfers.

## CUDA Graphs: Amortizing Setup Cost

CUDA Graphs allow an application to record a sequence of kernel launches and memory
operations into a graph, then replay that graph repeatedly with minimal host-side overhead.
NCCL integrates with CUDA Graphs to capture collective communication operations.

The significance for GSRP is the **amortization of setup cost**. In the traditional NCCL
model, each collective invocation requires topology lookup, algorithm selection, buffer
registration, and kernel launch with communication parameters. With CUDA Graph capture, these
steps happen once during the recording phase. Subsequent replays skip all setup.

This maps directly to TT-Metal's execution model, where `MeshWorkload` programs are compiled
once and executed repeatedly via `MeshCommandQueue`. The communication plan (routes, buffer
addresses, flow control parameters) can be embedded in the compiled program and reused across
iterations without per-iteration recomputation. The lesson: the cost of explicit
communication management can be amortized to near-zero for iterative workloads, which
describes the vast majority of deep learning training.

## Multi-Node Scaling: The Two-Tier Hierarchy

Beyond the NVLink domain (single node), NVIDIA systems use InfiniBand for inter-node
communication. This creates a two-tier hierarchy:

| Tier | Interconnect | Bandwidth | Latency | Collective library |
|-----|-------------|-----------|---------|-------------------|
| Intra-node | NVLink + NVSwitch | 900 GB/s | ~1 $\mu$s | NCCL (NVLink transport) |
| Inter-node | InfiniBand NDR | 400 Gb/s (~50 GB/s) | ~1--2 $\mu$s | NCCL (IB transport) |

The $18\times$ bandwidth drop from intra-node NVLink (900 GB/s) to inter-node InfiniBand
(50 GB/s) is the NVIDIA analog of Tenstorrent's chip-boundary penalty. NCCL handles this by
using topology-aware algorithms that minimize cross-node traffic: hierarchical all-reduce
(reduce within each node first, then across nodes), and pipeline parallelism placement that
co-locates pipeline stages on the same node where possible.

This two-tier approach is directly applicable to TT-Fabric. A GSRP could adopt hierarchical
communication: use the intra-chip NOC for local L1 access (fast tier) and the inter-chip
fabric for remote L1 access (slow tier), with the runtime or compiler minimizing cross-chip
traffic through placement-aware scheduling.

## Key Takeaways

- **NVIDIA's 900+ GB/s NVLink bandwidth has not eliminated explicit collectives.** NCCL
  persists because algorithmic optimization, computation overlap, and topology-aware
  scheduling provide performance benefits that no amount of bandwidth replaces. CUDA Unified
  Memory's $11\times$--$100\times$ penalty for demand-driven page migration is the strongest
  evidence against fully transparent remote SRAM.

- **In-network reduction (SHARP/NVLS) and CUDA Graphs** demonstrate complementary strategies:
  hardware-assisted collectives reduce all-reduce latency from $O(N)$ to $O(1)$, while graph
  capture amortizes explicit communication setup to near-zero for iterative workloads.

- **The two-tier NVLink/InfiniBand hierarchy maps directly to TT-Metal's NOC/Ethernet
  hierarchy**, suggesting a GSRP should adopt hierarchical algorithms and support both
  **bulk mode** (pre-computed collectives) and **scalar mode** (fine-grained remote access).

## GSRP Implications

The NVIDIA ecosystem rules out fully transparent, demand-driven remote SRAM as a CCL
replacement. The practical GSRP template is the CUDA UVM hybrid: a global address space with
explicit prefetch/staging directives. The remaining gap between TT-Metal's mesh API and an
NCCL-equivalent abstraction is primarily address translation and automatic flow control.
In-network reduction on ERISC could deliver outsized returns for all-reduce without
requiring the full GSRP infrastructure.

## Further Reading

- NVIDIA. (2022). "NVIDIA H100 Tensor Core GPU Architecture Whitepaper." Covers NVLink 4.0,
  NVSwitch 3.0, and the DGX H100 system architecture.

- Jeaugey, S. (2017). "NCCL 2.0." *GTC 2017*. NVIDIA presentation covering NCCL's ring
  and tree algorithms, topology detection, and performance characteristics.

- Li, A., et al. (2020). "Evaluating Modern GPU Interconnect: PCIe, NVLink, NV-SLI,
  NVSwitch, and GPUDirect." *IEEE Transactions on Parallel and Distributed Systems*,
  31(1), 94-110.

- NVIDIA. (2024). "NVIDIA Blackwell Architecture Technical Brief." Covers NVLink 5.0,
  NVSwitch 4.0, and the GB200 NVL72 architecture.

- Awan, A. A., et al. (2018). "Efficient Large Message Broadcast using NCCL and CUDA-Aware
  MPI for Deep Learning." *EuroMPI'18*. Analyzes NCCL collective performance and
  optimization strategies.

- Ganguly, D., et al. (2019). "Interplay between Hardware Prefetcher and Page Eviction
  Policy in CPU-GPU Unified Virtual Memory." *ISCA*. Analyzes UVM page migration behavior.

- NVIDIA Corporation. "CUDA C++ Programming Guide: Unified Memory." Developer documentation
  covering managed memory allocation, prefetch hints, and access counters.

---

**Next:** [`03_graphcore_ipu_exchange.md`](./03_graphcore_ipu_exchange.md)
