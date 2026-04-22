# Pros and Cons of Writing TT-Metal Kernels Using LLK

A practical guide for compute kernel authors and op developers evaluating the strengths and pain points of the current TT-Metal kernel-authoring workflow built on TT-LLK, covering the full stack from TRISC macros through the Compute Kernel API to SFPU extensions and cross-architecture portability.

---

## How to Use This Guide

| If you want to... | Start here |
|---|---|
| Understand how a single kernel source becomes three TRISC binaries | [Ch 1 — TRISC Macro System](ch1_trisc_macro_system/index.md) |
| Know what the Compute Kernel API handles for you and where it does not | [Ch 2 — Compute Kernel API Abstraction](ch2_compute_kernel_api_abstraction/index.md) |
| Evaluate whether defines-as-configuration is the right parameterization model | [Ch 3 — Defines as Configuration](ch3_defines_as_configuration/index.md) |
| Avoid deadlocks and protocol bugs in the tile processing loop | [Ch 4 — Tile Processing Pattern](ch4_tile_processing_pattern/index.md) |
| Assess compile-time and readability costs of template-heavy APIs | [Ch 5 — Template-Heavy API Design](ch5_template_heavy_api_design/index.md) |
| Find out what debugging tools exist and what gaps remain | [Ch 6 — Debugging Tools](ch6_debugging_tools/index.md) |
| Add a new SFPU operation end-to-end and understand the friction | [Ch 7 — SFPU Extension Path](ch7_sfpu_extension_path/index.md) |
| Write portable kernels across Blackhole, Wormhole B0, and Quasar | [Ch 8 — Portability and Improvement Opportunities](ch8_portability_and_improvement_opportunities/index.md) |

---

## Chapter Index

| # | Chapter | Description | Key Topics |
|---|---------|-------------|------------|
| 1 | [TRISC Macro System](ch1_trisc_macro_system/index.md) | Advantages and pitfalls of the MATH/PACK/UNPACK macro system that gates code to specific processor pipelines | Single-source authoring, compile-time elimination, invisible code paths, cross-TRISC synchronization gaps |
| 2 | [Compute Kernel API Abstraction](ch2_compute_kernel_api_abstraction/index.md) | How well the user-facing Compute Kernel API abstracts LLK complexity, and where the abstraction leaks | Descriptive API names, state_configure guard, manual reconfig calls, CB index management, DST register protocol |
| 3 | [Defines as Configuration](ch3_defines_as_configuration/index.md) | Pros and cons of preprocessor defines as the primary kernel parameterization mechanism | Zero-cost specialization, JIT build cache, `#ifdef` forests, implicit contracts, combinatorial binary explosion |
| 4 | [Tile Processing Pattern](ch4_tile_processing_pattern/index.md) | Ergonomics and common mistakes in the acquire/compute/commit/wait/pack/release tile processing loop | Pipeline synchronization, predictable structure, deadlock from missed release, CB count mismatches, tile index errors |
| 5 | [Template-Heavy API Design](ch5_template_heavy_api_design/index.md) | Trade-offs of pervasive C++ template usage in LLK and the Compute Kernel API | Compile-time specialization, JIT compile cost, deep error traces, macro-template interaction |
| 6 | [Debugging Tools](ch6_debugging_tools/index.md) | Effectiveness and gaps of kernel debugging tools | LLK_ASSERT, fake_kernels_target, generated file inspection, watcher, DPRINT (available on all three TRISCs with timing-perturbation caveats) |
| 7 | [SFPU Extension Path](ch7_sfpu_extension_path/index.md) | End-to-end friction of adding a new SFPU operation across six integration layers | Per-architecture duplication, sfpu_split_includes bottleneck, macro-based registration, testing overhead, header sprawl, SFPU_OP_CHAIN_0 code injection |
| 8 | [Portability and Improvement Opportunities](ch8_portability_and_improvement_opportunities/index.md) | Architecture divergence impact on kernel portability, and highest-impact improvement proposals | Quasar exclusions, CB API fragmentation, RAII guards, define schemas, automated reconfig, unified SFPU registration |

---

## Quick Reference

| Pattern / Issue | What It Is | Where to Learn More |
|---|---|---|
| `MATH(x)` / `PACK(x)` / `UNPACK(x)` macros | Gate code to a specific TRISC processor; expand to nothing on non-matching TRISCs | [Ch 1 — Advantages](ch1_trisc_macro_system/advantages.md), [Ch 1 — Pitfalls](ch1_trisc_macro_system/pitfalls.md) |
| `tile_regs_acquire` / `commit` / `wait` / `release` | Four-step destination register protocol; misordering causes silent hangs | [Ch 4 — Common Mistakes](ch4_tile_processing_pattern/common_mistakes.md) |
| `reconfig_data_format()` calls | Manual hardware reconfiguration required when alternating operations on different data formats | [Ch 2 — Abstraction Leaks](ch2_compute_kernel_api_abstraction/abstraction_leaks.md) |
| `#ifdef` / defines-as-configuration | Host-side program factory passes preprocessor defines to specialize kernel behavior at compile time | [Ch 3 — Advantages](ch3_defines_as_configuration/advantages_of_defines.md), [Ch 3 — Disadvantages](ch3_defines_as_configuration/disadvantages_of_defines.md) |
| `SFPU_OP_CHAIN_0` fusion pattern | Preprocessor code-injection mechanism for fusing SFPU operations after binary/unary ops | [Ch 2 — Abstraction Leaks](ch2_compute_kernel_api_abstraction/abstraction_leaks.md), [Ch 7 — Friction Points](ch7_sfpu_extension_path/friction_points.md) |
| `sfpu_split_includes.h` bottleneck | Centralized file that every new SFPU operation must modify, causing merge conflicts | [Ch 7 — Friction Points](ch7_sfpu_extension_path/friction_points.md) |
| CB synchronization mismatches | `cb_wait_front` / `cb_pop_front` / `cb_push_back` counts must match across TRISCs; mismatches cause deadlocks | [Ch 4 — Common Mistakes](ch4_tile_processing_pattern/common_mistakes.md) |
| `#ifndef ARCH_QUASAR` guards | Large sections of the Compute API are excluded on Quasar, breaking kernel portability | [Ch 8 — Architecture Divergence](ch8_portability_and_improvement_opportunities/architecture_divergence.md) |
| JIT error messages pointing to generated files | Compile errors reference `chlkc_math.cpp` line numbers instead of original kernel source | [Ch 6 — Gaps and Limitations](ch6_debugging_tools/gaps_and_limitations.md) |
| Template error propagation | Type mismatches at the Compute API level produce deeply nested traces through 4+ header layers | [Ch 5 — Developer Friction](ch5_template_heavy_api_design/developer_friction.md) |

---

## Prerequisites

- **C++ proficiency**: Intermediate knowledge of C++17 templates, preprocessor macros, and `constexpr`.
- **Tile-based accelerator concepts**: Familiarity with circular buffers, destination registers, and data movement patterns.
- **Tensix core architecture**: Basic understanding of the three TRISCs (unpack, math, pack) and the model where a single kernel source is compiled three times.
- **API stack awareness**: Familiarity with the four-layer API stack (raw LLK -> LLK API wrappers -> Compute Kernel API -> kernel `.cpp`), as covered in the companion guide [TT-LLK Usage in TT-Metal](../tt_llk_usage_in_tt_metal/).

---

## Source Code Locations

| Repository | Local Path | Description |
|---|---|---|
| TT-LLK | `/localdev/salnahari/testing_dir/tt-llk` | Low-level kernel library with per-architecture SFPU implementations, LLK API wrappers, and LLK lib functions |
| TT-Metal | `/localdev/salnahari/testing_dir/tt-metal` | Higher-level runtime containing the Compute Kernel API (`hw/inc/api/compute/`), kernel sources (`tt_metal/kernels/`, `ttnn/cpp/ttnn/operations/*/kernels/`), and host-side program factories |
