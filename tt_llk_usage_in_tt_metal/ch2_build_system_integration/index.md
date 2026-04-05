# Chapter 2: Build System Integration

This chapter explains how TT-Metal's build system references, includes, and compiles TT-LLK headers for each hardware target. LLK code never ships as a pre-compiled library; instead, it is compiled from source into every device kernel binary. The build system therefore needs to wire the correct architecture-specific LLK headers into three distinct compilation contexts:

1. **Firmware object compilation** -- static firmware images built at CMake configure/build time.
2. **JIT kernel compilation** -- user compute kernels compiled on-the-fly at runtime.
3. **Fake-kernel validation** -- a CMake target that compiles all in-tree kernels at build time to catch syntax and include errors early.

Each context assembles its own set of `-I` include paths that reference the TT-LLK submodule, selecting the correct architecture-specific subdirectory (`tt_llk_wormhole_b0/`, `tt_llk_blackhole/`, or `tt_llk_quasar/`).

## Sections

- [`submodule_and_cmake.md`](./submodule_and_cmake.md) -- How TT-LLK is vendored as a git submodule and integrated into CMake targets (`jitapi`, firmware objects).
- [`jit_compilation_pipeline.md`](./jit_compilation_pipeline.md) -- How the JIT build environment constructs include paths at runtime, with per-architecture specialization via HAL.
- [`per_kernel_code_generation.md`](./per_kernel_code_generation.md) -- How each compute kernel is split into four TRISC source files, how simplified syntax is transformed, and how `chlkc_descriptors.h` feeds LLK configuration.
