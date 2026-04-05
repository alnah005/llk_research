# Chapter 7: Hardware Target Selection

TT-Metal must compile compute kernels that run directly on Tenstorrent silicon -- Wormhole, Blackhole, or Quasar. Each architecture has its own LLK implementation inside the `tt-llk` repository (`tt_llk_wormhole_b0/`, `tt_llk_blackhole/`, `tt_llk_quasar/`), its own `ckernels/` shim layer inside TT-Metal, and its own HAL initialization code. This chapter explains the machinery that selects the correct LLK paths for a given hardware target at both CMake build time and JIT compile time.

Chapter 2 introduced the HAL and the JIT include mechanism. This chapter focuses specifically on **what differs across architectures** in the selection logic, how the CMake firmware build iterates over architectures, and how Quasar's newer "TLS-style" build diverges from the Wormhole/Blackhole pattern.

## Sections

1. [**HAL Architecture Dispatch**](./hal_architecture_dispatch.md) -- How the HAL constructor selects per-architecture initialization, and how each architecture's `HalJitBuildQuery` subclass returns different LLK include paths, defines, and compiler flags.

2. [**Firmware Compilation**](./firmware_compilation.md) -- How `tt_metal/hw/CMakeLists.txt` iterates over architectures to compile firmware objects with architecture-specific `-I` paths pointing into `tt_llk_*` and `ckernels/`.

3. [**Quasar Divergence**](./quasar_divergence.md) -- How Quasar's tt-2xx generation breaks from the Wormhole/Blackhole pattern: different HAL base class, TLS-style build loop, different CPU targets, a smaller LLK API surface, and `ARCH_QUASAR` guards throughout the Compute API.
