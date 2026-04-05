# How TT-Metal Uses TT-LLK: A Developer's Guide

This guide explains how the TT-Metal runtime consumes the TT-LLK (Low-Level Kernel) library -- from build-system wiring and API layering through op dispatch and hardware targeting. It is intended for developers working on TT-Metal compute kernels, TTNN operations, or the LLK library itself.

---

## How to Use This Guide

| Your Goal | Recommended Path | Direct Links |
|---|---|---|
| Understand the overall system before diving in | Ch 1 then Ch 3 | [Architecture Overview](ch1_architecture_overview/index.md), [LLK Library Structure](ch3_llk_library_structure/index.md) |
| Write or modify a compute kernel | Ch 5, then Ch 4 for internals | [Compute Kernel API](ch5_compute_kernel_api/index.md), [LLK API Wrappers](ch4_llk_api_wrappers/index.md) |
| Add a new TTNN op end-to-end | Ch 6 then Ch 8 | [Op-to-Kernel Mapping](ch6_op_to_kernel_mapping/index.md), [Extending and Debugging](ch8_extending_and_debugging/index.md) |
| Fix a build or JIT compilation issue | Ch 2, then Ch 7 | [Build System Integration](ch2_build_system_integration/index.md), [Hardware Target Selection](ch7_hardware_target_selection/index.md) |
| Port an op to a new architecture (e.g., Quasar) | Ch 7, then Ch 3 and Ch 4 | [Hardware Target Selection](ch7_hardware_target_selection/index.md), [LLK Library Structure](ch3_llk_library_structure/index.md), [LLK API Wrappers](ch4_llk_api_wrappers/index.md) |
| Debug a failing compute kernel | Ch 8, with Ch 5 as reference | [Extending and Debugging](ch8_extending_and_debugging/index.md), [Compute Kernel API](ch5_compute_kernel_api/index.md) |

---

## Chapter Index

| # | Chapter | Description | Key Topics |
|---|---|---|---|
| 1 | [Architecture Overview](ch1_architecture_overview/index.md) | Bird's-eye view of TT-Metal and TT-LLK relationship | RISC-V cores (Tensix), tile-based execution, submodule boundary, data flow |
| 2 | [Build System Integration](ch2_build_system_integration/index.md) | Build system: submodule, CMake, JIT compilation, code generation | CMake firmware build, JIT include paths, code generation, `generate_bank.py` |
| 3 | [LLK Library Structure](ch3_llk_library_structure/index.md) | TT-LLK internal structure and API surface across hardware targets | Per-architecture directories, `_llk_*_()` functions, SFPU intrinsics, `common/` utilities |
| 4 | [LLK API Wrappers](ch4_llk_api_wrappers/index.md) | TT-Metal's wrapper layer, I/O layer, and SFPU extensions | Operand abstraction, `init`/`pack`/`unpack` wrappers, SFPU dispatch macros |
| 5 | [Compute Kernel API](ch5_compute_kernel_api/index.md) | User-facing Compute Kernel API and LLK mapping | `tile_work`, `acquire_dst`, `release_dst`, elementwise/matmul/reduce APIs |
| 6 | [Op-to-Kernel Mapping](ch6_op_to_kernel_mapping/index.md) | Host-side ops configuring and dispatching compute kernels | TTNN op structure, `CreateKernel()`, compile-time defines, `KernelHandle` |
| 7 | [Hardware Target Selection](ch7_hardware_target_selection/index.md) | Architecture selection via HAL at build and JIT time | HAL, arch enum, per-target include sets, Quasar TLS-style divergence |
| 8 | [Extending and Debugging](ch8_extending_and_debugging/index.md) | Adding new ops, debugging, end-to-end compilation flow | New-op walkthrough, kernel printf, watcher, compilation pipeline |

---

## Quick Reference

| API / Concept | What It Does | Where to Learn More |
|---|---|---|
| `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release` | Lock, commit, wait, and release the destination register tile array for compute | [Ch 5](ch5_compute_kernel_api/index.md) |
| `mul_tiles`, `add_tiles`, `sub_tiles` | Elementwise binary tile operations | [Ch 5](ch5_compute_kernel_api/index.md) |
| `matmul_tiles` | Tile-level matrix multiply | [Ch 5](ch5_compute_kernel_api/index.md) |
| `_llk_unpack_*_()` / `_llk_pack_*_()` | Raw LLK unpack and pack functions (not called directly by kernels) | [Ch 3](ch3_llk_library_structure/index.md), [Ch 4](ch4_llk_api_wrappers/index.md) |
| `SFPU_OP_FUNC` / sfpu dispatch | Macro-driven dispatch to special-function unit operations | [Ch 4](ch4_llk_api_wrappers/index.md) |
| `CreateKernel()` | Host API to register a compute kernel with compile-time args | [Ch 6](ch6_op_to_kernel_mapping/index.md) |
| HAL (`tt::tt_metal::hal`) | Hardware Abstraction Layer selecting arch-specific paths at runtime | [Ch 7](ch7_hardware_target_selection/index.md) |
| JIT include path injection | Build system wiring that resolves the correct LLK headers per arch | [Ch 2](ch2_build_system_integration/index.md), [Ch 7](ch7_hardware_target_selection/index.md) |
| `generate_bank.py` | Code-generation script producing LLK dispatch tables | [Ch 2](ch2_build_system_integration/index.md) |

---

## Prerequisites

- Familiarity with C++ and basic CMake concepts.
- A cloned `tt-metal` repository with submodules initialized (`git submodule update --init --recursive`).
- General awareness of tile-based hardware execution (the guide introduces Tenstorrent specifics in Chapter 1).

---

## Source Code Locations

| Component | Path inside `tt-metal` |
|---|---|
| TT-LLK submodule | `tt_metal/third_party/tt_llk/` |
| Per-arch LLK implementations | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole,quasar}/` |
| TT-Metal ckernel wrappers | `tt_metal/hw/ckernels/{wormhole_b0,blackhole,quasar}/metal/` |
| Compute Kernel API headers | `tt_metal/hw/inc/api/compute/` |
| TTNN operations | `ttnn/cpp/ttnn/operations/` |
| Firmware build scripts | `tt_metal/hw/firmware/` |
