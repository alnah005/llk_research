# Chapter 8: Performance Analysis, Hardware Gap Assessment, and Roadmap

This chapter provides the quantitative conclusion to the guide. It grounds the GSRP feasibility assessment in measured benchmark data, builds latency models for cross-chip data movement, presents a comprehensive explicit-vs-implicit tradeoff matrix, identifies per-architecture hardware gaps and firmware requirements for Wormhole and Blackhole, assesses workload suitability across six parallelism patterns, and proposes a phased adoption roadmap from today's explicit CCL model toward increasingly transparent communication.

## Files

| # | File | Description |
|---|------|-------------|
| 1 | [`01_measured_fabric_bandwidth.md`](./01_measured_fabric_bandwidth.md) | Actual bandwidth measurements from TT-Metal benchmark golden data: BH unicast ~12.8 B/cycle, WH half-ring ~6.4 B/cycle, WH full-ring ~5.2 unicast / ~5.6 multicast. Multi-link scaling, hop degradation, bidirectional penalties, and GSRP overhead impact. |
| 2 | [`02_latency_model.md`](./02_latency_model.md) | Per-hop latency decomposition (~671 ns single-hop), multi-hop estimates for T3K/Galaxy/multi-Galaxy. GSRP translation overhead (~13 cycles cached). Read/write asymmetry (~2.0--2.3x). Break-even analysis: <5% overhead at ~3 KB, <2% at ~8 KB. Connection setup amortization. |
| 3 | [`03_explicit_vs_implicit_tradeoff_matrix.md`](./03_explicit_vs_implicit_tradeoff_matrix.md) | Six-dimension comparison (throughput, latency, resource utilization, programmability, debuggability, scalability). Per-collective mapping. Break-even message sizes. Recommendation: hybrid model retaining explicit CCL for 3--4 bulk collectives, GSRP for everything else. |
| 4 | [`04_wormhole_gap_analysis.md`](./04_wormhole_gap_analysis.md) | WH hardware assessment: 80 Tensix, 1,464 KB L1, 16 Ethernet links, 1D routing. Capability matrix across GSRP phases. Missing: HW address translation, cross-chip read protocol. Minimum firmware changes: ~8--10 weeks Phase 1. |
| 5 | [`05_blackhole_gap_analysis.md`](./05_blackhole_gap_analysis.md) | BH hardware assessment: 140 Tensix, 14 ERISC (dual RISC-V), 1,536 KB L1, native 2D routing, `NOC_SEC_FENCE_RANGE`. BH closes every WH gap. Quasar `noc_address_translation_table` (Phase 3 enabler, Quasar-only). |
| 6 | [`06_workload_suitability.md`](./06_workload_suitability.md) | Assessment across six parallelism patterns: data parallel, tensor/model parallel, pipeline parallel, MoE, KV-cache sharing, activation checkpointing. Five-criteria scoring. Production model analysis (LLaMA, Mixtral). Best fit: MoE and KV-cache. |
| 7 | [`07_phased_adoption_roadmap.md`](./07_phased_adoption_roadmap.md) | Five-phase roadmap (Phase 0--4) with decision gates (DG1--DG4), escape hatches, timeline, resource estimates, and risk mitigation. Phase 1: Named Remote Memory (software-only, 8--10 weeks). Phase 2: Transparent Global Write (firmware-assisted). Phase 3: Hardware-Assisted GSRP (Quasar). Phase 4: Full compiler integration. |

## Reading Guide

Files 01--02 provide the quantitative foundation (bandwidth and latency data). File 03 synthesizes the tradeoff analysis. Files 04--05 are architecture-specific gap analyses (read the one relevant to your hardware first). File 06 maps workloads to GSRP suitability. File 07 is the actionable conclusion -- the phased roadmap with concrete deliverables and decision gates. Readers short on time can read files 03 and 07 for the key conclusions.

## Key Source Directories

- `tests/tt_metal/microbenchmarks/ethernet/golden/` -- Bandwidth benchmark golden data
- `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h` -- WH memory map constants
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` -- BH memory map constants
- `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp` -- FabricConfig variants
