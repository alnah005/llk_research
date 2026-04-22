# Pros and Cons of LLK Integration with TT-Metal

This guide provides a comprehensive analysis of how TT-LLK is integrated into TT-Metal via its submodule-based, header-only compilation model, covering architectural trade-offs, code duplication, API boundaries, testing gaps, and alternative approaches. It is intended for TT-Metal and TT-LLK engineers -- op writers, build system maintainers, and architects -- who work at or near the integration boundary between the two repositories.

---

## How to Use This Guide

| Reader Goal | Recommended Path | Direct Links |
|---|---|---|
| Understand the integration from scratch | Ch1 -> Ch2 -> Ch3 (sequential) | [Architecture overview](./ch1_integration_architecture_overview/index.md), [Advantages](./ch2_advantages/index.md), [Disadvantages](./ch3_disadvantages/index.md) |
| Evaluate whether to refactor the integration model | Ch3 -> Ch5 -> Ch8 | [Pain points](./ch3_disadvantages/index.md), [Duplication analysis](./ch5_architecture_duplication/index.md), [Alternative patterns](./ch8_alternative_patterns/index.md) |
| Assess API stability and abstraction quality | Ch4 -> Ch6 | [API boundary analysis](./ch4_api_boundary/index.md), [Template trade-offs](./ch6_template_tradeoffs/index.md) |
| Improve testing at the LLK/Metal boundary | Ch7 -> Ch4 | [Testing strategy](./ch7_testing_strategy/index.md), [Leaky abstractions](./ch4_api_boundary/index.md) |
| Reduce per-architecture maintenance cost | Ch5 -> Ch8 | [Architecture duplication](./ch5_architecture_duplication/index.md), [Code generation alternative](./ch8_alternative_patterns/index.md) |
| Understand build system and JIT compilation | Ch1 -> Ch3 -> Ch6 | [JIT pipeline](./ch1_integration_architecture_overview/index.md), [Build complexity](./ch3_disadvantages/index.md), [Compilation impact](./ch6_template_tradeoffs/index.md) |

---

## Chapter Index

| # | Title | Description | Key Topics |
|---|-------|-------------|------------|
| 1 | [Integration Architecture Overview](./ch1_integration_architecture_overview/index.md) | Establishes the structural relationship between TT-LLK and TT-Metal, from submodule to compiled kernel. | Three-layer header architecture, git submodule mechanics, JIT compilation pipeline, HAL include path wiring |
| 2 | [Advantages of the Current Integration Model](./ch2_advantages/index.md) | Documents why the header-only, build-time-compiled model is a deliberate and effective design. | Zero-cost abstractions via inlining, LTO and build-time specialization, atomic cross-repo updates, source-level debugging |
| 3 | [Disadvantages and Pain Points](./ch3_disadvantages/index.md) | Catalogues concrete friction, risks, and maintenance burdens with severity ratings. | Include path explosion (14+ paths), submodule update friction, implicit `_llk_*` API contracts, `#ifdef TRISC_*` fragility, developer ergonomics |
| 4 | [API Abstraction Boundary Analysis](./ch4_api_boundary/index.md) | Evaluates the quality of the three-layer API boundary and where it leaks. | Layer 1/2/3 call chain, leaky abstractions, SFPU boundary ambiguity, compile-time globals piercing API surface |
| 5 | [Per-Architecture Code Duplication](./ch5_architecture_duplication/index.md) | Quantifies the duplicated code across wormhole_b0, blackhole, and quasar in both LLK and Metal. | Near-clone wormhole/blackhole directories, quasar structural divergence, Metal-side duplication multiplier, shared `common/` is only 2 files |
| 6 | [Template Design Trade-offs](./ch6_template_tradeoffs/index.md) | Examines the compilation-time cost and binary-size implications of heavy template parameterization. | Up to 6 template parameters per function, 26 `if constexpr` branches in a single file, JIT cache effectiveness, runtime parameterization trade-offs |
| 7 | [Cross-Boundary Testing Strategy](./ch7_testing_strategy/index.md) | Surveys how LLK and Metal test compute operations independently and identifies blind spots. | LLK standalone tests (Layer 1), Metal integration tests (Layer 3), no submodule-update CI gate, five identified testing gaps |
| 8 | [Alternative Integration Patterns](./ch8_alternative_patterns/index.md) | Explores three alternative models and evaluates them against evidence from earlier chapters. | Compiled library model (unsuitable), interface stabilization (best incremental path), monorepo absorption, template-based code generation |

---

## Quick Reference

| Finding / Recommendation | Source | Assessment |
|---|---|---|
| Header-only integration enables full inlining and cross-module LTO -- critical for resource-constrained Tensix RISC-V cores | Ch2, Ch8 | Keep the header-only compilation model; a compiled library alternative would impose unacceptable performance penalties |
| Include path configuration involves 14+ directories assembled in `build.cpp` (marked `TODO(pgk) this list is insane`) | Ch1, Ch3 | High-severity pain point; simplification should be a priority |
| The `_llk_*` Layer 1 API has no formal contract -- Metal wrappers call functions that are not declared as stable | Ch3, Ch4 | High severity; introduce a versioned `llk_api.h` contract at the Layer 1/Layer 2 boundary |
| Wormhole_b0 and blackhole `llk_lib/` directories are near-clones (28 files each, ~6,600-7,600 lines) | Ch5 | Significant maintenance burden; template-based code generation could reduce duplication long-term |
| Quasar diverges structurally from wormhole/blackhole (different file names, 10 unique files, 15 missing files) | Ch5 | Mechanical porting of fixes across architectures is infeasible today |
| LLK tests validate Layer 1 in isolation; Metal tests validate Layer 3 -- nobody tests the Layer 1-to-Layer 2 boundary | Ch7 | Add targeted integration tests and a submodule-update CI gate |
| Interface stabilization (versioned API contract) offers the best risk/reward ratio for incremental improvement | Ch8 | Recommended first step before considering monorepo absorption or code generation |
| Template parameter counts (up to 6 per function) and pervasive `if constexpr` branching increase JIT compile time | Ch6 | JIT caching mitigates repeat builds; first-build cost remains high but is the correct trade-off for binary size |
| The SFPU kernel layer (158 files in Metal) depends on LLK infrastructure with unclear ownership | Ch4 | Clarify ownership boundary between LLK and Metal for SFPU code |

---

## Prerequisites

Readers should have:

- **C++ proficiency**: Familiarity with templates, `if constexpr`, inline functions, and header-only library design.
- **Build system knowledge**: Working understanding of CMake, cross-compilation toolchains, and link-time optimization (LTO).
- **Tensix architecture basics**: Awareness of the unpack/math/pack pipeline and the TRISC (Tensix RISC-V) processor model.
- **Repository access**: Ability to browse [tt-metal](https://github.com/tenstorrent/tt-metal) and [tt-llk](https://github.com/tenstorrent/tt-llk) source code.
- **Familiarity with at least one repo**: Deep knowledge of both repositories is not required, but readers should have working experience with either TT-Metal or TT-LLK.

---

## Source Code Locations

| Component | Repository | Path |
|---|---|---|
| LLK core headers (Layer 1) | [tt-llk](https://github.com/tenstorrent/tt-llk) | `tt_llk_<arch>/llk_lib/` |
| LLK shared headers | [tt-llk](https://github.com/tenstorrent/tt-llk) | `common/` (`llk_assert.h`, `tensor_shape.h`) |
| Metal LLK API wrappers (Layer 2) | [tt-metal](https://github.com/tenstorrent/tt-metal) | `tt_metal/hw/ckernels/<arch>/metal/llk_api/` |
| Compute Kernel API (Layer 3) | [tt-metal](https://github.com/tenstorrent/tt-metal) | `tt_metal/hw/inc/api/compute/` |
| SFPU kernels | [tt-metal](https://github.com/tenstorrent/tt-metal) | `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/` |
| JIT build system | [tt-metal](https://github.com/tenstorrent/tt-metal) | `tt_metal/jit_build/build.cpp`, `tt_metal/jit_build/genfiles.cpp` |
| HAL layer (per-arch) | [tt-metal](https://github.com/tenstorrent/tt-metal) | `tt_metal/llrt/hal/tt-1xx/`, `tt_metal/llrt/hal/tt-2xx/` |
| Submodule mount point | [tt-metal](https://github.com/tenstorrent/tt-metal) | `tt_metal/third_party/tt_llk/` |
| LLK standalone tests | [tt-llk](https://github.com/tenstorrent/tt-llk) | `tests/` |
| Metal compute tests | [tt-metal](https://github.com/tenstorrent/tt-metal) | `tests/tt_metal/tt_metal/test_kernels/compute/` |
