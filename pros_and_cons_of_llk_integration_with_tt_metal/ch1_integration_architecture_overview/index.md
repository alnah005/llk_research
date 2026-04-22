# Chapter 1: Integration Architecture Overview

## Learning Objectives

After reading this chapter you will be able to:

1. Trace the full path from an LLK header in the `tt-llk` repository to a compiled device kernel inside TT-Metal.
2. Describe the three-layer header architecture that separates hardware-specific math from the public compute API.
3. Explain how the HAL layer and JIT build system wire architecture-specific include paths at compile time.
4. Identify where the `common/` directory in TT-LLK fits into the dependency chain.

## Subtopics

| File | Description |
|------|-------------|
| [`submodule_mechanics.md`](./submodule_mechanics.md) | How TT-LLK is pinned as a git submodule, how CMakeLists.txt exposes its headers, and the include-path complexity this creates. |
| [`jit_compilation_pipeline.md`](./jit_compilation_pipeline.md) | The JIT build pipeline: `JitBuildEnv`, `JitBuildState`, generated TRISC source files, `chlkc_descriptors.h`, and build caching. |

---

## High-Level Integration Diagram

```
tt-metal  (host repo)
 |
 +-- tt_metal/
 |    +-- third_party/
 |    |    +-- tt_llk/              <-- git submodule  (.gitmodules, branch = main)
 |    |         +-- common/         <-- shared headers: llk_assert.h, tensor_shape.h
 |    |         +-- tt_llk_wormhole_b0/
 |    |         |    +-- llk_lib/   <-- Layer 1: core LLK functions (_llk_*)
 |    |         +-- tt_llk_blackhole/
 |    |         +-- tt_llk_quasar/
 |    |
 |    +-- hw/
 |    |    +-- ckernels/<arch>/metal/
 |    |    |    +-- llk_api/        <-- Layer 2: Metal LLK API wrappers (llk_*)
 |    |    |    +-- llk_api/llk_sfpu/
 |    |    |    +-- common/
 |    |    |    +-- llk_io/
 |    |    +-- inc/api/compute/     <-- Layer 3: public Compute Kernel API
 |    |
 |    +-- jit_build/
 |    |    +-- build.cpp            <-- JitBuildEnv::init(), JitBuildState
 |    |    +-- genfiles.cpp         <-- TRISC source generation, chlkc_descriptors.h
 |    |
 |    +-- llrt/hal/
 |         +-- tt-1xx/wormhole/     <-- HAL: per-arch includes, defines, source lists
 |         +-- tt-1xx/blackhole/
 |         +-- tt-2xx/quasar/
 |
 +-- .gitmodules                    <-- submodule declarations
```

## The Three-Layer Header Architecture

The integration stacks three distinct layers of headers, each providing a progressively higher level of abstraction.

### Layer 1 -- LLK Core (`_llk_*` functions)

**Location:** `[tt-llk] tt_llk_<arch>/llk_lib/`

These are the lowest-level math primitives. Functions use the `_llk_` prefix and operate directly on hardware registers. Example: `_llk_math_eltwise_binary_init_<>()` in [`tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h) configures address modes and FPU parameters via `ckernel_template`.

### Layer 2 -- Metal LLK API Wrappers (`llk_*` functions)

**Location:** `[tt-metal] tt_metal/hw/ckernels/<arch>/metal/llk_api/`

These wrappers live in TT-Metal and call Layer 1. They strip the leading underscore: `llk_math_eltwise_binary_init()` calls `_llk_math_eltwise_binary_init_<>()`. This layer is where Metal inserts default tile shapes (e.g., `ckernel::DEFAULT_TENSOR_SHAPE`) and provides overloads accepting operand IDs. See [`llk_math_binary_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_binary_api.h) which includes `llk_math_eltwise_binary.h` from Layer 1.

### Layer 3 -- Compute Kernel API

**Location:** `[tt-metal] tt_metal/hw/inc/api/compute/`

This is the public API that kernel authors call. Functions live in the `ckernel` namespace and are documented with parameter tables. For example, [`eltwise_binary.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/inc/api/compute/eltwise_binary.h) conditionally includes `llk_math_binary_api.h` (under `#ifdef TRISC_MATH`) and `llk_unpack_AB_api.h` (under `#ifdef TRISC_UNPACK`), bridging down to Layer 2.

The conditional inclusion is key: each TRISC processor (UNPACK, MATH, PACK) sees only its own subset of Layer 2 headers, controlled by defines like `TRISC_MATH` emitted during JIT compilation.

## How the HAL Wires Include Paths

The Hardware Abstraction Layer at `[tt-metal] tt_metal/llrt/hal/` provides a `HalJitBuildQueryInterface` that returns per-architecture include directories, defines, and source file lists. For Wormhole, this is implemented in [`wh_hal.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp) (lines 101-134), which pushes paths including:

- `tt_metal/hw/ckernels/wormhole_b0/metal/common`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_io`
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc`
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib` (Layer 1)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api` (Layer 2, for COMPUTE cores)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu` (for COMPUTE cores)

These are consumed by `JitBuildState` (see [`build.cpp:340-345`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L340)) which iterates `jit_build_query.includes(params)` and prepends `-I<root>/` to form compiler flags.

## Role of the `common/` Directory in TT-LLK

The `[tt-llk] common/` directory contains two shared headers:

- **`llk_assert.h`** -- Assertion macros for LLK code, enabled via `-DENABLE_LLK_ASSERT` (see [`build.cpp:259-261`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L259)).
- **`tensor_shape.h`** -- Tensor shape definitions used across all architectures.

The `common/` directory is included in the JIT environment via `JitBuildEnv::init()` at [`build.cpp:277`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L277):

```cpp
root_ + "tt_metal/third_party/tt_llk/common",
```

It is also globbed into the CMake `jitapi` INTERFACE target at [`CMakeLists.txt:172`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/CMakeLists.txt#L172):

```cmake
file(
    GLOB_RECURSE TT_LLK_HEADERS
    third_party/tt_llk/common/*.h
    ...
)
```

This dual-path inclusion (CMake for IDE/packaging, JIT for runtime compilation) is a recurring pattern in the integration.

---

**Next:** [`submodule_mechanics.md`](./submodule_mechanics.md)
