# Plan: Pros and Cons of LLK Integration with TT-Metal

## Audience

This guide is intended for **TT-Metal and TT-LLK engineers** -- op writers, build system maintainers, and architects -- who need to understand how TT-LLK is consumed by TT-Metal and evaluate the trade-offs of the current integration model. Readers are expected to be familiar with C++ template metaprogramming, CMake build systems, and the Tensix compute architecture (unpack/math/pack pipeline). They should have a working knowledge of at least one of the two repositories, but not necessarily deep familiarity with the integration boundary between them.

---

## Chapter List

### Chapter 1: Integration Architecture Overview

**Description:** Establishes the structural relationship between TT-LLK and TT-Metal, tracing the full path from LLK header to compiled device kernel.

**Directory:** `ch1_integration_architecture_overview`

**Files:**

- `index.md`
  - High-level diagram of the integration: TT-LLK as a git submodule at `tt_metal/third_party/tt_llk/`
  - The three-layer header architecture: LLK core (`tt_llk_<arch>/llk_lib/`), Metal LLK API wrappers (`tt_metal/hw/ckernels/<arch>/metal/llk_api/`), and the compute kernel API (`tt_metal/hw/inc/api/compute/`)
  - How the HAL layer (`tt_metal/llrt/hal/`) wires include paths per architecture at JIT compile time (wh_hal.cpp, bh_hal.cpp, qa_hal.cpp each push architecture-specific include directories)
  - Role of the `common/` directory in TT-LLK (shared `tensor_shape.h`, `llk_assert.h`)

- `submodule_mechanics.md`
  - How `.gitmodules` pins TT-LLK at `tt_metal/third_party/tt_llk` on the `main` branch
  - Version synchronization: submodule commit pinning, update workflow, risk of drift
  - CMakeLists.txt integration: TT-LLK headers are declared as `INTERFACE` file sets in `tt_metal/CMakeLists.txt` via `GLOB_RECURSE` over `third_party/tt_llk/`
  - The `TODO(pgk) this list is insane` comment in `build.cpp:268` as evidence of include path complexity

- `jit_compilation_pipeline.md`
  - How `JitBuildEnv::init()` assembles compiler flags, include paths (including `tt_metal/third_party/tt_llk/common`), and defines
  - How `JitBuildState` queries the HAL for per-architecture includes, defines, and source files
  - The generated files pipeline: `genfiles.cpp` producing `chlkc_unpack.cpp`, `chlkc_math.cpp`, `chlkc_pack.cpp` with TRISC-specific prologs
  - `chlkc_descriptors.h` generation: data format emission, tile dimension emission, the `#include "llk_defs.h"` injection for math kernels
  - Build caching via FNV1a hashing of all compilation parameters

### Chapter 2: Advantages of the Current Integration Model

**Description:** Analyzes the concrete benefits that the submodule-based, header-only, build-time-compiled integration provides.

**Directory:** `ch2_advantages`

**Files:**

- `index.md`
  - Summary of key advantages with cross-references to detailed sections

- `header_only_benefits.md`
  - Zero-cost abstraction: all LLK functions are `inline`, enabling the compiler to specialize per template instantiation with no function call overhead
  - Template specialization allows compile-time selection of behavior (e.g., `EltwiseBinaryType`, `BroadcastType`, `MathFidelity` as template parameters)
  - No separate library build step: LLK headers are compiled directly into each kernel, avoiding ABI compatibility concerns between LLK and Metal
  - Simplified deployment: no shared library to package or version separately for device targets

- `build_time_specialization.md`
  - How the RISC-V cross-compiler (`riscv-tt-elf-g++`) compiles LLK headers with full visibility into the kernel context
  - LTO (`-flto=auto`) across LLK and kernel code: dead code elimination and cross-function optimization
  - Architecture-specific define injection (`-DARCH_WORMHOLE`, etc.) enabling `#ifdef`-based hardware adaptation
  - The `-O3` optimization level for compute kernels vs `-Os` for other processors, demonstrating fine-grained control

- `tight_coupling_benefits.md`
  - Atomic updates: changing LLK and Metal in the same PR when APIs evolve
  - Direct source-level debugging: op writers can step into LLK code during kernel development
  - No versioning ceremony: no need to publish LLK releases, manage semantic versioning, or maintain backward compatibility guarantees
  - The `ENABLE_LLK_ASSERT` mechanism in `llk_assert.h`: conditional compilation allows LLK assertions to be toggled from Metal's runtime options (`rtoptions.get_llk_asserts()`)

### Chapter 3: Disadvantages and Pain Points

**Description:** Documents concrete problems, friction, and risks observed in the current integration.

**Directory:** `ch3_disadvantages`

**Files:**

- `index.md`
  - Summary of pain points with severity assessment

- `build_system_complexity.md`
  - The include path explosion: `JitBuildEnv::init()` hardcodes 8+ base include directories, then the HAL adds 6-8 more per architecture -- totaling 14+ include paths per kernel compilation
  - The `fake_kernels_target/CMakeLists.txt` duplicates include path logic for IDE support, creating a second source of truth
  - Triple duplication of include path configuration: `build.cpp` (base), HAL files (per-arch), and `fake_kernels_target` (IDE)
  - SFPI toolchain management complexity: version checking, download, fallback logic spanning 100+ lines in `hw/CMakeLists.txt`

- `coupling_and_synchronization.md`
  - Submodule update friction: bumping LLK in Metal requires updating the pinned commit, which can conflict across branches
  - Implicit API contracts: the Metal `llk_api/` wrappers (e.g., `llk_math_binary_api.h`) call internal `_llk_*` prefixed functions from LLK (e.g., `_llk_math_eltwise_binary_init_<>`) -- these are not formally declared as a stable API surface
  - The `#ifdef TRISC_MATH` / `#ifdef TRISC_UNPACK` / `#ifdef TRISC_PACK` conditional compilation pattern in `api/compute/eltwise_binary.h`: fragile coupling between Metal's compute API and the TRISC processor model
  - Global mutable state in LLK tests (e.g., `unp_cfg_context`, `pack_sync_tile_dst_ptr` as globals in test sources) suggesting tight coupling to execution model

- `developer_ergonomics.md`
  - Deep include chains: a compute kernel author writes `#include "api/compute/eltwise_binary.h"`, which includes `llk_math_binary_api.h`, which includes `llk_math_eltwise_binary.h`, which includes `ckernel_include.h`, `ckernel_ops.h`, `cmath_common.h`, etc.
  - Macro-heavy kernel authoring: `ELTWISE_OP`, `ELTWISE_OP_TYPE`, `SFPU_OP_CHAIN_0`, `PACK_RELU`, `DST_ACCUM_MODE` are compile-time defines injected by the build system, not visible in source
  - Legacy vs simplified kernel syntax: `genfiles.cpp` detects and transforms between `namespace NAMESPACE { void MAIN { } }` and `void kernel_main()`, with a deprecation warning for the old style
  - Error messages from deep template instantiation failures in LLK headers are notoriously difficult to diagnose

### Chapter 4: API Abstraction Boundary Analysis

**Description:** Evaluates the quality of the abstraction boundary between LLK and TT-Metal, identifying leaky abstractions and unclear ownership.

**Directory:** `ch4_api_boundary`

**Files:**

- `index.md`
  - Overview of the three API layers and their intended responsibilities

- `three_layer_api.md`
  - Layer 1 (LLK Core): `_llk_*` prefixed functions in `tt_llk_<arch>/llk_lib/` -- hardware register manipulation, address mode configuration, low-level tile operations
  - Layer 2 (Metal LLK API): `llk_*` prefixed functions in `hw/ckernels/<arch>/metal/llk_api/` -- wraps Layer 1 with Metal-specific defaults (e.g., `DEFAULT_TENSOR_SHAPE`, `DST_SYNC_MODE`)
  - Layer 3 (Compute Kernel API): `api/compute/` headers -- user-facing functions like `binary_op_init_common()`, `binary_tiles_init()`, `pack_tile()`, `tile_regs_acquire()`
  - Assessment: Layer 2 is thin (mostly forwarding with Metal globals injected), raising the question of whether it adds value or just adds indirection

- `leaky_abstractions.md`
  - The `UNPACK()`, `MATH()`, `PACK()` macros in compute API: op writers must annotate which TRISC processor executes each call, leaking the hardware threading model
  - `DST_SYNC_MODE` and `DST_ACCUM_MODE` as compile-time globals that LLK functions depend on: these are Metal concepts that leak into LLK's API surface
  - LLK functions expecting raw data format integers (`std::uint32_t srca_data_format`) rather than typed enums, requiring casts at the boundary
  - The `llk_io/` directory in Metal's ckernels: I/O operations (circular buffer management) are Metal-owned but live alongside LLK-owned code, blurring ownership

- `sfpu_boundary.md`
  - The SFPU kernel layer: 90+ `ckernel_sfpu_*.h` files in Metal's `llk_api/llk_sfpu/` (~10,123 lines) -- these are Metal-authored, not part of TT-LLK, but use LLK infrastructure
  - This represents a significant body of "LLK-style" code that lives in Metal, creating ambiguity about where new SFPU ops should be authored
  - The `experimental/` directory in both repos: unstable APIs under development, with no clear promotion path to stable

### Chapter 5: Per-Architecture Code Duplication

**Description:** Quantifies and analyzes the per-architecture code duplication in TT-LLK and its impact on TT-Metal.

**Directory:** `ch5_architecture_duplication`

**Files:**

- `index.md`
  - Summary statistics and key findings

- `duplication_analysis.md`
  - TT-LLK code volume by architecture: wormhole_b0 (7,646 lines in llk_lib + 6,239 in common/inc), blackhole (6,615 + 6,412), quasar (4,210 + 5,482)
  - File-level divergence: wormhole and blackhole have nearly identical file lists but quasar diverges significantly (e.g., `llk_math_eltwise_binary_broadcast.h` instead of `llk_math_eltwise_binary_sfpu.h`, `llk_unpack_binary_operands.h` instead of `llk_unpack_AB.h`)
  - Metal's mirroring: `hw/ckernels/` has matching per-architecture directories (`wormhole_b0/`, `blackhole/`, `quasar/`) with their own `llk_api/` wrappers
  - The HAL layer also duplicates per-architecture logic (wh_hal.cpp, bh_hal.cpp, qa_hal.cpp) including nearly identical include path lists

- `maintenance_burden.md`
  - Bug fixes must be applied N times (once per architecture) unless they happen to be in the small `common/` directory (2 files: `llk_assert.h`, `tensor_shape.h`)
  - Adding a new LLK operation requires creating files in each architecture directory plus corresponding Metal API wrappers
  - Quasar's structural divergence (different naming conventions, different operation decomposition) means fixes cannot be mechanically ported
  - The instructions directory (`instructions/assembly.yaml`) is per-architecture, reflecting genuine hardware differences, but the llk_lib layer often contains similar algorithmic logic with minor register-level differences

### Chapter 6: Template Design Trade-offs

**Description:** Analyzes the template-heavy API design and its implications for compilation time, binary size, and maintainability.

**Directory:** `ch6_template_tradeoffs`

**Files:**

- `index.md`
  - Overview of template usage patterns and their trade-offs

- `compilation_impact.md`
  - Template parameter explosion: functions like `_llk_math_eltwise_binary_<EltwiseBinaryType, BroadcastType, SyncType, is_fp32_dest_acc_en, MathFidelity, EltwiseBinaryReuseDestType>` have 6 template parameters, each with multiple valid values
  - Each unique combination generates a separate function instantiation, compiled from scratch in each kernel
  - Header-only + templates means the full LLK implementation is parsed and type-checked for every translation unit (every `chlkc_math.cpp`, `chlkc_unpack.cpp`, `chlkc_pack.cpp`)
  - The JIT build system mitigates this somewhat via build caching (`JitBuildState::build_state_matches()`), but first-time compilation for a new parameter combination is expensive
  - Metal uses `-O3` for compute kernels and `-flto=auto`, which increases compile time further

- `binary_size_and_alternatives.md`
  - Template instantiations produce specialized code: good for performance on a resource-constrained RISC-V core, but each kernel binary includes only the instantiations it uses
  - The `constexpr` and `if constexpr` patterns in LLK code (e.g., `is_high_fidelity(math_fidelity) ? 1 : 0`) enable compile-time pruning
  - Alternative: runtime-parameterized functions would reduce compilation cost but increase binary size and runtime overhead on the Tensix RISC-V cores
  - The current approach is likely correct for the target hardware, but creates pain for development iteration speed

### Chapter 7: Cross-Boundary Testing Strategy

**Description:** Evaluates the testing approaches in both repos and identifies gaps at the integration boundary.

**Directory:** `ch7_testing_strategy`

**Files:**

- `index.md`
  - Overview of testing philosophy in each repo

- `llk_standalone_testing.md`
  - TT-LLK has its own test infrastructure: Python test harness (`tests/python_tests/`), C++ test sources (`tests/sources/`), hardware-specific tests (`tests/hw_specific/`)
  - Tests call `_llk_*` functions directly (Layer 1), bypassing Metal's Layer 2 and Layer 3 wrappers
  - Uses tt-exalens for hardware interaction, SFPI compiler for building -- independent of Metal's build system
  - Performance tests (e.g., `perf_eltwise_binary_fpu.py`, `perf_matmul.py`) validate LLK in isolation

- `metal_integration_testing.md`
  - Metal tests compute kernels through the Layer 3 API (`api/compute/`), exercising the full three-layer stack
  - No systematic integration tests that verify the LLK submodule works correctly after an update
  - The `fake_kernels_target/` in Metal's JIT build provides compile-time validation for IDE support, but does not test runtime correctness

- `testing_gaps.md`
  - Gap 1: No contract tests between Layer 1 (`_llk_*`) and Layer 2 (`llk_*`) -- if LLK changes internal function signatures, Metal discovers breakage only at build time
  - Gap 2: LLK standalone tests and Metal tests may test the same operations with different parameters, leading to undetected corner cases
  - Gap 3: No automated compatibility check when updating the LLK submodule in Metal
  - Gap 4: SFPU operations authored in Metal (`llk_sfpu/`) are not tested by TT-LLK's test suite at all
  - Gap 5: Performance regression testing across the boundary -- LLK performance tests run in isolation, but Metal's kernel compilation overhead is not measured alongside

### Chapter 8: Alternative Integration Patterns

**Description:** Explores alternative integration models that could address identified weaknesses, with trade-off analysis for each.

**Directory:** `ch8_alternative_patterns`

**Files:**

- `index.md`
  - Summary of alternatives and evaluation criteria

- `compiled_library_model.md`
  - Alternative: pre-compile LLK into static libraries (per architecture, per optimization level)
  - Pros: faster kernel JIT compilation, clearer API boundary, version independence
  - Cons: loss of cross-module LTO and inlining, ABI stability requirements, larger binary sizes from unused instantiations, significant refactoring effort
  - Assessment: likely unsuitable due to Tensix RISC-V resource constraints and the performance criticality of inlining

- `interface_stabilization.md`
  - Alternative: formalize the Layer 1/Layer 2 boundary with a versioned API contract
  - Define a `llk_api.h` header with stable function signatures; internal `_llk_*` functions become implementation details
  - Introduce API version checking at build time
  - Pros: enables independent development cadences, clearer ownership
  - Cons: added ceremony, risk of abstraction ossification, may limit hardware-specific optimizations

- `monorepo_and_code_generation.md`
  - Alternative: absorb LLK into TT-Metal's monorepo, eliminating submodule friction
  - Pros: atomic commits across the boundary, simplified CI, no version synchronization
  - Cons: loss of LLK's independent testing capability, larger repo, harder for LLK-focused contributors
  - Alternative: use code generation to produce per-architecture LLK implementations from a shared template, reducing duplication
  - Pros: single source of truth for algorithmic logic, per-arch specialization via configuration
  - Cons: significant upfront investment, code generation tooling maintenance, harder to debug generated code

---

## Conventions

1. **Terminology:**
   - "LLK" always refers to the TT-LLK library (the `tt-llk` repository or its submodule copy).
   - "Metal" always refers to TT-Metal (the `tt-metal` repository).
   - "Layer 1" = LLK core functions (`_llk_*` prefix), "Layer 2" = Metal LLK API wrappers (`llk_*` prefix), "Layer 3" = Compute Kernel API (`api/compute/`).
   - "Architecture" or "arch" refers to a hardware target: wormhole_b0, blackhole, or quasar.
   - "TRISC" refers to the three compute RISC-V processors (T0/unpack, T1/math, T2/pack).

2. **File References:**
   - All file paths are relative to their respective repository roots unless qualified with `[tt-llk]` or `[tt-metal]`.
   - When referencing code that exists in both repos, specify which repo and the exact path.

3. **Notation:**
   - Template parameter lists are written in angle brackets: `<EltwiseBinaryType, BroadcastType, MathFidelity>`.
   - Compile-time defines are written in `SCREAMING_SNAKE_CASE` and prefixed with `-D` when showing compiler flags.
   - Line counts and statistics are approximate and based on the repository state at the time of writing.

4. **Formatting:**
   - Each pro or con must include at least one concrete code reference or build system artifact as evidence.
   - Trade-off discussions use a consistent "Pros / Cons / Assessment" structure.
   - Diagrams use Mermaid syntax.
   - Code snippets are kept minimal -- show only the relevant lines, not entire files.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Nature of Dependency |
|---------|-----------|---------------------|
| Ch2 (Advantages) | Ch1 (Architecture Overview) | Requires understanding of the three-layer model and JIT compilation pipeline |
| Ch3 (Disadvantages) | Ch1 (Architecture Overview) | References include path mechanics and build system structure |
| Ch4 (API Boundary) | Ch1 (Architecture Overview), Ch2 (Advantages) | Builds on the three-layer model; references benefits of tight coupling |
| Ch5 (Architecture Duplication) | Ch1 (Architecture Overview) | References per-architecture directory structure and HAL configuration |
| Ch6 (Template Trade-offs) | Ch2 (Advantages), Ch3 (Disadvantages) | Extends the header-only benefits/costs discussion with concrete template analysis |
| Ch7 (Testing Strategy) | Ch1 (Architecture Overview), Ch4 (API Boundary) | References the API layers to identify which layers are tested where |
| Ch8 (Alternative Patterns) | Ch2-Ch7 (All preceding analysis) | Synthesizes all identified pros/cons to evaluate alternatives |
