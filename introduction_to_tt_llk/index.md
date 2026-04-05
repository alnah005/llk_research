# Introduction to TT-LLK

This guide is a comprehensive introduction to **TT-LLK (Tenstorrent Low-Level Kernels)**, the header-only C++17 library that programs Tenstorrent Tensix cores at the lowest software layer. It is written for software engineers and op writers who need to write, modify, or optimize LLK kernels, and for engineers working at higher levels of the Tenstorrent stack who want to understand what the LLK layer does and how it maps to hardware.

---

## How to Use This Guide

| Your Goal | Recommended Path | Key Deep Links |
|:----------|:-----------------|:---------------|
| Understand what TT-LLK is and where it fits | Ch 1 | [Software Stack Position](ch1_what_is_tt_llk/software_stack_position.md), [Repository Structure](ch1_what_is_tt_llk/repository_structure.md) |
| Learn the hardware you are programming | Ch 2 then Ch 3 | [Core Components](ch2_tensix_core_architecture/core_components.md), [Data Flow](ch2_tensix_core_architecture/data_flow.md), [Tiles and Faces](ch3_data_organization/tiles_and_faces.md) |
| Write your first kernel | Ch 4, Ch 7, Ch 8 | [Unpack Operations](ch4_unpack_math_pack_pipeline/unpack_operations.md), [Writing Tests](ch7_test_infrastructure/writing_tests.md), [Eltwise Binary Walkthrough](ch8_kernel_development_patterns/eltwise_binary_walkthrough.md) |
| Implement an SFPU activation function | Ch 5 then Ch 8 | [SFPU Architecture](ch5_sfpu_operations/sfpu_architecture.md), [SFPU Operations Catalog](ch5_sfpu_operations/sfpu_operations_catalog.md), [API Conventions](ch8_kernel_development_patterns/api_conventions.md) |
| Port a kernel to a different chip | Ch 6 | [Architecture Comparison](ch6_hardware_targets/architecture_comparison.md), [LLK API Differences](ch6_hardware_targets/llk_api_differences.md) |
| Write and run a performance test | Ch 7 | [Performance Testing](ch7_test_infrastructure/performance_testing.md), [Running and Debugging](ch7_test_infrastructure/running_and_debugging.md) |
| Understand data formats and fidelity trade-offs | Ch 3 | [Data Formats](ch3_data_organization/data_formats.md), [Format Conversion](ch3_data_organization/format_conversion.md), [Math Fidelity](ch3_data_organization/math_fidelity.md) |
| Understand MOP programming and matmul internals | Ch 8 | [API Conventions](ch8_kernel_development_patterns/api_conventions.md), [Matrix Multiply Walkthrough](ch8_kernel_development_patterns/matmul_walkthrough.md) |

---

## Chapter Index

| # | Chapter | Description | Key Concepts |
|:-:|:--------|:------------|:-------------|
| 1 | [Ch 1 -- What is TT-LLK?](ch1_what_is_tt_llk/index.md) | Introduces TT-LLK and its position in the Tenstorrent software stack. | Software stack (TT-Forge, TT-Metalium, TT-LLK), repository layout, hardware targets |
| 2 | [Ch 2 -- Tensix Core Architecture](ch2_tensix_core_architecture/index.md) | Explains the internal structure of a single Tensix Core. | L1 SRAM, five RISC-V processors (BRISC, NCRISC, TRISC0/1/2), Tensix Engine, Source A/B and Dest registers |
| 3 | [Ch 3 -- Data Organization](ch3_data_organization/index.md) | Covers tiles, faces, data formats, and format conversion. | 32x32 tiles, 16x16 faces, `DataFormat`, block floating point, gaskets, `MathFidelity` |
| 4 | [Ch 4 -- The Unpack-Math-Pack Pipeline](ch4_unpack_math_pack_pipeline/index.md) | Describes the three-stage compute pipeline and its synchronization. | Unpack/math/pack stages, TRISC0/1/2, `DstSync`, semaphores, MOP programming |
| 5 | [Ch 5 -- SFPU Operations](ch5_sfpu_operations/index.md) | Covers the programmable SIMD engine for element-wise operations. | SFPU load/store model, `VectorMode`, `SfpuType`, unary/binary/ternary interfaces |
| 6 | [Ch 6 -- Hardware Target Differences](ch6_hardware_targets/index.md) | Compares LLK implementations across Wormhole B0, Blackhole, and Quasar. | Scoped vs. unscoped enums, `SfpuType` divergence, file naming differences, boot modes |
| 7 | [Ch 7 -- Test Infrastructure](ch7_test_infrastructure/index.md) | Explains how to write, run, and debug LLK tests. | Python + C++ test pairs, `TestConfig`, `pytest`, `PerfConfig`, hardware perf counters |
| 8 | [Ch 8 -- Kernel Development Patterns](ch8_kernel_development_patterns/index.md) | Practical guidance for writing kernels, with full walkthroughs. | Naming conventions, `addr_mod_t`, `ckernel_template`, LLTT replay, matmul tiling |

---

## Quick Reference

| API / Concept | What It Does | Where to Learn More |
|:--------------|:-------------|:--------------------|
| `llk_unpack_A` / `llk_unpack_AB` | DMA tile data from L1 into Source A (or A and B) registers with format conversion | [Unpack Operations](ch4_unpack_math_pack_pipeline/unpack_operations.md) |
| `llk_math_eltwise_binary` | Execute element-wise add, sub, or mul on Source A and Source B data via the FPU | [Math Operations](ch4_unpack_math_pack_pipeline/math_operations.md) |
| `llk_math_matmul` | Execute matrix multiply on Source A and Source B tiles | [Matrix Multiply Walkthrough](ch8_kernel_development_patterns/matmul_walkthrough.md) |
| `llk_math_eltwise_unary_sfpu` | Execute an SFPU operation (activation, transcendental, etc.) on Dest register data | [SFPU Architecture](ch5_sfpu_operations/sfpu_architecture.md) |
| `llk_pack` | DMA results from the Destination register back to L1 with format conversion | [Pack Operations](ch4_unpack_math_pack_pipeline/pack_operations.md) |
| `DstSync` (`SyncHalf` / `SyncFull`) | Controls destination register double-buffering mode | [Synchronization](ch4_unpack_math_pack_pipeline/synchronization.md) |
| `DataFormat` | Enum specifying tile data encoding (Float16, BFP8, Float32, etc.) | [Data Formats](ch3_data_organization/data_formats.md) |
| `MathFidelity` | Controls FPU multiplication phases, trading accuracy for throughput | [Math Fidelity](ch3_data_organization/math_fidelity.md) |
| `ckernel_template` | Programs MOP loops for efficient repeated instruction sequences | [API Conventions](ch8_kernel_development_patterns/api_conventions.md) |
| `addr_mod_t` | Configures address modifier registers for register file pointer management | [API Conventions](ch8_kernel_development_patterns/api_conventions.md) |
| `TestConfig` | Python-side configuration object for LLK tests (formats, templates, runtimes) | [Writing Tests](ch7_test_infrastructure/writing_tests.md) |
| `VectorMode` | Controls which faces of a tile the SFPU processes | [SFPU Architecture](ch5_sfpu_operations/sfpu_architecture.md) |

---

## Prerequisites

- **C++ proficiency** -- at least C++17, including templates and `constexpr`.
- **Computer architecture basics** -- registers, memory hierarchies, SIMD concepts.
- **Conceptual ML knowledge** -- matrix multiply, element-wise operations, reductions.
- **Some embedded/hardware-adjacent experience** -- memory-mapped registers, DMA.
- **No prior Tenstorrent knowledge required** -- the guide introduces Tensix hardware, the Tensix ISA, RISC-V internals, and Tenstorrent-specific data formats from scratch.

---

## Source Code Location

TT-LLK is an open-source project hosted on GitHub:

- **Repository:** [https://github.com/tenstorrent/tt-llk](https://github.com/tenstorrent/tt-llk)
- **Common headers (shared across targets):** `common/inc/` -- SFPU implementations, `ckernel_defs`, data format definitions.
- **Per-architecture LLK implementations:** `<arch>/llk_lib/` where `<arch>` is one of `wormhole_b0`, `blackhole`, or `quasar`.
- **Per-architecture hardware headers:** `<arch>/inc/` -- register definitions, ISA wrappers, architecture-specific constants.
- **Tests:** `tests/` -- Python test drivers and C++ kernel source files.
