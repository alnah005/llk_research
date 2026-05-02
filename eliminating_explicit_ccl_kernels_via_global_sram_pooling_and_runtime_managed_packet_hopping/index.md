# Eliminating Explicit CCL Kernels via Global SRAM Pooling and Runtime-Managed Packet Hopping

A research guide exploring whether explicit Collective Communication Library (CCL) kernels between Tenstorrent devices can be eliminated by treating all L1 SRAMs across chips as a global pool with runtime-managed packet hopping.

## Key Finding

No production multi-chip AI accelerator has achieved fully transparent global SRAM at scale with Ethernet-class interconnects. The practical path for Tenstorrent is a **hybrid model**: retain explicit CCL for the 3--4 bulk ring-pipelined collectives (all-gather, reduce-scatter, all-reduce) where pipelining provides 40--50% throughput advantage, and use the Global SRAM Pool (GSRP) abstraction for everything else -- MoE expert dispatch, KV-cache sharing, activation checkpointing, point-to-point transfers, and irregular communication patterns where GSRP matches explicit CCL performance within 3--5% while reducing code from 30--50 lines to 1--3 lines per transfer.

## Chapters

| # | Chapter | Directory | Files | Description |
|---|---------|-----------|:-----:|-------------|
| 1 | [The Current CCL Programming Model](./ch01_current_ccl_model/index.md) | `ch01_current_ccl_model/` | 4 | Baseline: how explicit CCL kernels work today in TT-Metal. The problem statement. |
| 2 | [Per-Chip Memory Architecture and the NOC](./ch02_memory_and_noc/index.md) | `ch02_memory_and_noc/` | 4 | L1 SRAM layout, dual-NOC system, address encoding, buffer types. Hardware foundation. |
| 3 | [Inter-Chip Communication](./ch03_interchip_fabric/index.md) | `ch03_interchip_fabric/` | 6 | Ethernet links, EDM firmware, packet headers, control plane, routing, flow control. |
| 4 | [Existing Building Blocks](./ch04_building_blocks/index.md) | `ch04_building_blocks/` | 4 | Mesh API primitives, worker-fabric connections, sockets, MeshBuffer/MeshDevice. |
| 5 | [Comparative Survey](./ch05_comparative_survey/index.md) | `ch05_comparative_survey/` | 4 | Cerebras WSE, NVIDIA NVLink/NCCL, Graphcore IPU -- lessons for Tenstorrent. |
| 6 | [Global SRAM Pool Design](./ch06_global_sram_design/index.md) | `ch06_global_sram_design/` | 5 | Problem statement, address space, consistency model, candidate architectures, translation layer. |
| 7 | [Runtime Infrastructure](./ch07_runtime_infrastructure/index.md) | `ch07_runtime_infrastructure/` | 4 | Transparent injection, flow control, deadlock avoidance, dispatch integration. |
| 8 | [Performance Analysis and Roadmap](./ch08_performance_and_roadmap/index.md) | `ch08_performance_and_roadmap/` | 7 | Measured bandwidth, latency model, tradeoff matrix, WH/BH gap analysis, workload suitability, phased roadmap. |

**Total: 38 content files across 8 chapters.**

## Reading Paths

- **Full path (newcomers to fabric):** Read chapters 1 through 8 sequentially. If unfamiliar with Tensix L1 and NOC addressing, read Chapter 2 before Chapter 1.
- **Fast path (engineers familiar with TT-Metal internals):** Start at Chapter 5, reference earlier chapters as needed. Then read Chapters 6--8 for the GSRP proposal and assessment.
- **Executive summary:** Read the guide-level index (this file), then Chapter 8 files 03 (tradeoff matrix) and 07 (phased roadmap).

## Conventions

See [`plan.md`](./plan.md) for full terminology, notation conventions, formatting rules, and cross-chapter dependency table.
