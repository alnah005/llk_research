# Chapter 5: Comparative Survey -- How Other Architectures Handle This Problem

This chapter reviews how three other multi-chip and wafer-scale architectures handle the tension between explicit and implicit inter-chip communication: Cerebras WSE (hardware-transparent wafer-scale SRAM), NVIDIA NVLink/NCCL (high-bandwidth links with explicit software collectives), and Graphcore IPU (BSP model with compiler-scheduled exchange phases). The synthesis file maps the lessons learned to Tenstorrent's specific constraints and hardware capabilities, directly informing the GSRP design candidates in Chapter 6 and the deadlock avoidance strategies in Chapter 7.

## Files

| # | File | Description |
|---|------|-------------|
| 1 | [`01_cerebras_wafer_scale.md`](./01_cerebras_wafer_scale.md) | Cerebras WSE: 850,000+ cores on a single wafer, ~40 GB on-wafer SRAM, hardware-routed inter-core communication with no chip boundaries. Lessons: zero-overhead global SRAM requires no chip boundaries; the compiler bears significant complexity; uniform latency matters more than transparency per se. |
| 2 | [`02_nvidia_nvlink_nccl.md`](./02_nvidia_nvlink_nccl.md) | NVLink/NVSwitch (900 GB/s bidirectional), NCCL ring/tree algorithms, NVSwitch in-network reduction, CUDA Unified Memory. Key lesson: even with massive hardware investment, explicit collective libraries persist for workload-specific optimization; transparent remote memory carries significant overhead. |
| 3 | [`03_graphcore_ipu_exchange.md`](./03_graphcore_ipu_exchange.md) | Graphcore IPU: BSP model (compute then exchange), SRAM-only execution, compiler-scheduled all-to-all exchange. Key lesson: BSP provides natural synchronization boundaries that avoid deadlocks; compiler-driven communication insertion hides complexity; epoch-based synchronization is powerful but constraining. |
| 4 | [`04_lessons_and_applicability.md`](./04_lessons_and_applicability.md) | Synthesis across all three architectures mapped to Tenstorrent constraints. Key finding: no production multi-chip AI accelerator has achieved fully transparent global SRAM at scale with Ethernet-class interconnects. The practical path is increasingly powerful library primitives with hardware assist. |

## Reading Guide

Each architecture file (01--03) is self-contained and can be read independently. File 04 synthesizes the cross-cutting lessons and should be read last. Readers focused on the GSRP design can read file 04 alone for the key takeaways, referencing the individual architecture files for supporting evidence.

## Key Source Directories

This chapter is primarily based on published architecture papers and documentation rather than TT-Metal source code. Source references are cited within each file.
