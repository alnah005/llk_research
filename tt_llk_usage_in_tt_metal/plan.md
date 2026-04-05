# TT-LLK Usage in TT-Metal: Guide Plan

## 1. Audience

**Primary audience:** Engineers and developers who work on or across the TT-Metal and TT-LLK codebases -- including op writers, compute kernel authors, and build-system maintainers.

**Assumed knowledge:**
- Intermediate C++ (templates, preprocessor, cross-compilation concepts).
- Basic familiarity with accelerator programming concepts (kernels, tiles, circular buffers).
- General awareness of what Tenstorrent hardware does (Tensix cores, RISC-V processors), but NOT deep knowledge of the Tensix ISA.
- Basic CMake literacy.

**NOT assumed:**
- Prior knowledge of how TT-Metal and TT-LLK are wired together.
- Familiarity with the LLK API surface or its naming conventions.
- Understanding of TT-Metal's JIT compilation pipeline.

---

## 2. Chapter List

---

### Chapter 1: `ch1_architecture_overview`

**Description:** Provides a bird's-eye view of how TT-Metal and TT-LLK relate as a consumer-library pair, covering both repository layouts and the conceptual model that motivates the design.

**Files:**

#### `index.md`
- Chapter introduction and reading order.

#### `codebase_layout.md`
- Top-level directory structure of TT-LLK: `common/` (cross-arch headers: `llk_assert.h`, `tensor_shape.h`), and per-arch directories `tt_llk_blackhole/`, `tt_llk_wormhole_b0/`, `tt_llk_quasar/`. Each arch dir contains `common/inc/` (ckernel infrastructure), `llk_lib/` (LLK implementations), `instructions/` (assembly instruction definitions).
- TT-LLK is a header-only library; no compiled libraries are shipped. Consumers include headers directly.
- Key TT-Metal directories that touch LLK: `tt_metal/third_party/tt_llk/` (the submodule), `tt_metal/hw/ckernels/<arch>/metal/` (Metal-side API wrappers), `tt_metal/hw/inc/api/compute/` (user-facing compute kernel API), `tt_metal/jit_build/` (JIT compilation pipeline), `tt_metal/kernels/compute/` and `ttnn/cpp/ttnn/operations/*/device/kernels/compute/` (device-side kernel source files).
- Diagram: four-layer stack from TTNN ops down to raw LLK.

#### `key_concepts.md`
- Tensix core architecture: five RISC-V processors, three TRISCs (T0=unpack, T1=math, T2=pack) controlling the Tensix Engine. Why the same kernel source gets compiled three times with different `TRISC_*` defines.
- Tile-based computation: data flows L1 -> circular buffers -> unpack to source registers -> FPU/SFPU -> destination registers -> pack back to L1.
- The `MATH(x)`, `PACK(x)`, `UNPACK(x)` macros (defined in `hw/inc/api/compute/compute_kernel_api.h` and `common_globals.h`): expand to `x` on the matching TRISC, expand to nothing on others.
- Submodule relationship: `.gitmodules` entry points `tt_metal/third_party/tt_llk` to `https://github.com/tenstorrent/tt-llk.git` on branch `main`. The submodule is NOT built by CMake (`tt_metal/third_party/CMakeLists.txt` only adds `umd`). Version pinning via submodule commit.

---

### Chapter 2: `ch2_build_system_integration`

**Description:** Explains how TT-Metal's CMake build system and JIT compilation pipeline wire TT-LLK headers into firmware and kernel compilation.

**Files:**

#### `index.md`
- Chapter introduction.

#### `cmake_offline_build.md`
- `tt_metal/hw/CMakeLists.txt`: the offline (ahead-of-time) firmware build using the SFPI RISC-V cross-compiler (`riscv-tt-elf-g++`).
- Include paths in `GPP_INCLUDES` (lines 354-380): for each architecture, adds `-I` for `tt_metal/third_party/tt_llk/tt_llk_<arch>/common/inc`, `tt_metal/third_party/tt_llk/tt_llk_<arch>/llk_lib`, `tt_metal/hw/ckernels/<arch>/metal/common`, `tt_metal/hw/ckernels/<arch>/metal/llk_io`.
- Compile flags: `-std=c++17`, `-flto=auto`, `-ffast-math`, `-Os`, architecture-specific CPU flags (`-mcpu=tt-wh`, `-mcpu=tt-bh`).
- Old-style (tt-1xx: Wormhole/Blackhole) vs. TLS-style (tt-2xx: Quasar) build loops.

#### `jit_build_pipeline.md`
- `tt_metal/jit_build/build.cpp`: the runtime JIT kernel compilation pipeline.
- `JitBuildEnv::init()` (lines 91-303): Sets base include paths including `tt_metal/third_party/tt_llk/common`. Configures SFPI compiler path, flags, and defines (`-DTENSIX_FIRMWARE`, `-DLOCAL_MEM_EN=0`, optional profiler/watcher/debug defines).
- `JitBuildState` constructor (lines 305-435): Queries the HAL via `HalJitBuildQueryInterface` for architecture-specific includes, defines, flags, source files. Compute kernels (TENSIX + COMPUTE processor class) get `-O3` optimization and the SFPI include directory.
- `compile_one()` (lines 464-523): Assembles full compilation command: compiler + optimization level + cflags + includes + per-kernel defines (via `settings->process_defines`) + compile-time args (`-DKERNEL_COMPILE_TIME_ARGS=...` and `-DKERNEL_COMPILE_TIME_ARG_MAP=...`).
- Build cache: `build_state_hash_` computed from all compilation parameters; stale binaries are invalidated when any parameter changes.

#### `hal_arch_includes.md`
- Per-architecture HAL files that populate include paths at JIT time:
  - `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp` (lines 103-136): always adds `ckernels/blackhole/metal/common`, `ckernels/blackhole/metal/llk_io`, `third_party/tt_llk/tt_llk_blackhole/common/inc`, `third_party/tt_llk/tt_llk_blackhole/llk_lib`. For COMPUTE processors: also `ckernels/blackhole/metal/llk_api`, `ckernels/blackhole/metal/llk_api/llk_sfpu`.
  - `tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp` (lines 101-134): same pattern for `wormhole_b0`.
  - `tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp` (lines 222-257): same pattern for `quasar`, but uses `tt-2xx` internal include paths.
- Per-arch defines: `ARCH_BLACKHOLE`, `ARCH_WORMHOLE`, `ARCH_QUASAR`.
- Common defines from `hal_1xx_common.cpp`: `COMPILE_FOR_TRISC=N`, `UCK_CHLKC_UNPACK`/`UCK_CHLKC_MATH`/`UCK_CHLKC_PACK`, `NAMESPACE=chlkc_*`.

---

### Chapter 3: `ch3_llk_api_layer_architecture`

**Description:** Describes the multi-layered header architecture from raw TT-LLK primitives up to the user-facing compute kernel API.

**Files:**

#### `index.md`
- Chapter introduction and the "layer cake" diagram.

#### `layer_overview.md`
- Four-layer architecture:
  1. **TT-LLK `llk_lib/`** -- raw implementations with underscore-prefixed names (e.g., `_llk_math_eltwise_binary_init_()`, `_llk_math_eltwise_binary_()`). These emit Tensix instructions via ckernel primitives.
  2. **TT-LLK `common/inc/`** -- ckernel infrastructure: `ckernel_template.h`, `ckernel_ops.h`, `cmath_common.h`, `cpack_common.h`, `cunpack_common.h`. Address mode configuration, instruction encoding, register management.
  3. **TT-Metal `ckernels/<arch>/metal/llk_api/`** -- API wrappers providing `llk_math_eltwise_binary_init()`, `llk_math_eltwise_binary()`, etc. These add operand shape handling, operand ID translation, and cleaner calling conventions.
  4. **TT-Metal `hw/inc/api/compute/`** -- user-facing API with descriptive names (`binary_op_init_common()`, `matmul_tiles()`, `reduce_tile()`, `pack_tile()`). These compose UNPACK + MATH + PACK calls.

#### `tt_llk_layer.md`
- Files in `tt_llk_<arch>/llk_lib/`: `llk_math_eltwise_binary.h`, `llk_math_matmul.h`, `llk_math_reduce.h`, `llk_pack.h`, `llk_unpack_AB.h`, `llk_unpack_A.h`, `llk_math_eltwise_unary_datacopy.h`, `llk_math_eltwise_unary_sfpu.h`, etc.
- Example walkthrough: `llk_math_eltwise_binary.h` -- includes `ckernel_ops.h` and `cmath_common.h`, defines `eltwise_binary_configure_addrmod()` (sets address modes for the FPU), the operation template selects `TT_OP_ELWADD`/`TT_OP_ELWSUB`/`TT_OP_ELWMUL` Tensix instructions.
- Files in `common/inc/sfpu/`: SFPU-specific implementations (`ckernel_sfpu_exp.h`, `ckernel_sfpu_gelu.h`, `ckernel_sfpu_recip.h`, etc.) that use SFPU intrinsics for vector operations.
- `llk_defs.h`: common definitions and enum types used across LLK files.

#### `metal_api_wrapper_layer.md`
- Files in `ckernels/<arch>/metal/llk_api/`: `llk_math_binary_api.h`, `llk_math_matmul_api.h`, `llk_pack_api.h`, `llk_unpack_AB_api.h`, `llk_unpack_A_api.h`, etc.
- Wrapper pattern: each file includes its `llk_lib` counterpart (e.g., `llk_math_binary_api.h` includes `llk_math_eltwise_binary.h`), then provides a public function that resolves operand info and delegates to the underscore-prefixed function.
- Example: `llk_math_eltwise_binary_init_with_operands()` calls `get_operand_id()`, `get_operand_tensor_shape()`, then `_llk_math_eltwise_binary_init_()`.
- `llk_param_structs.h`: parameter structures for configuration passing.
- `llk_sfpu_types.h`: SFPU type definitions.
- `llk_sfpu/` subdirectory: per-SFPU-op wrappers, significantly extended beyond TT-LLK's base set (Metal adds `ckernel_sfpu_addcdiv.h`, `ckernel_sfpu_binary_pow.h`, `ckernel_sfpu_cbrt.h`, `ckernel_sfpu_celu.h`, and dozens more).
- `llk_io/` layer: `llk_io.h`, `llk_io_pack.h`, `llk_io_unpack.h`, `llk_operands.h`, `llk_outputs.h` -- manage circular buffer interactions for pack/unpack operations.

#### `user_facing_compute_api.md`
- Files in `hw/inc/api/compute/`: `matmul.h`, `eltwise_binary.h`, `reduce.h`, `tile_move_copy.h`, `pack.h`, `tilize.h`, `untilize.h`, `bcast.h`, `softmax.h`, `layernorm.h`, `cb_api.h`, `reg_api.h`, etc.
- Master include files: `compute_kernel_api.h` conditionally includes all LLK API headers gated by `TRISC_MATH`, `TRISC_PACK`, `TRISC_UNPACK`. `common_globals.h` provides the simpler version for headers that need lighter includes.
- Example: `eltwise_binary.h::binary_op_init_common()` orchestrates all three pipelines: `UNPACK(llk_unpack_hw_configure, llk_unpack_AB_init)`, `MATH(llk_math_pack_sync_init, llk_math_hw_configure)`, `PACK(llk_pack_hw_configure, llk_pack_init, llk_pack_dest_init)`.
- The `ALWI` macro (`inline __attribute__((always_inline))`) ensures these wrappers are always inlined for performance.

---

### Chapter 4: `ch4_kernel_compilation_pipeline`

**Description:** How TT-Metal compiles a single compute kernel `.cpp` file into three separate TRISC binaries (unpack, math, pack).

**Files:**

#### `index.md`
- Chapter introduction.

#### `trisc_code_generation.md`
- `tt_metal/jit_build/genfiles.cpp` function `jit_build_genfiles_triscs_src()`: Takes a single kernel `.cpp` file and generates four files:
  - `chlkc_unpack.cpp` (with `#define TRISC_UNPACK`)
  - `chlkc_math.cpp` (with `#define TRISC_MATH`)
  - `chlkc_pack.cpp` (with `#define TRISC_PACK`)
  - `chlkc_isolate_sfpu.cpp` (with `#define TRISC_ISOLATE_SFPU`, used by Quasar)
- Each file has a prolog (`#define TRISC_* \n #include "defines_generated.h"`) followed by the kernel source.
- Two syntax modes: "simplified" (`void kernel_main()`) is preferred and is auto-transformed to legacy format via `simple_kernel_syntax::transform_to_legacy_syntax()` which wraps the source in `namespace chlkc_math { void math_main() { ... } }` (and similarly for unpack/pack).
- Legacy syntax (`namespace NAMESPACE { void MAIN { ... } }`) still supported but deprecated.

#### `chlkc_list_entry_point.md`
- `tt_metal/hw/ckernels/<arch>/metal/common/chlkc_list.h`: the compilation entry point that ties everything together.
- Includes `ckernel.h`, `ckernel_gpr_map.h`, `llk_param_structs.h`.
- Conditionally includes `chlkc_descriptors.h` (generated data format config) and the generated `chlkc_math.cpp` / `chlkc_pack.cpp` / `chlkc_unpack.cpp` based on `UCK_CHLKC_MATH`, `UCK_CHLKC_PACK`, `UCK_CHLKC_UNPACK` defines (set by HAL).
- Defines `run_kernel()` which calls `chlkc_math::math_main()`, `chlkc_pack::pack_main()`, or `chlkc_unpack::unpack_main()`.

#### `defines_and_data_formats.md`
- `defines_generated.h`: Generated per-kernel by `genfiles.cpp`, contains user-specified defines from `ComputeConfig.defines` (e.g., `#define ELTWISE_OP_TYPE EltwiseBinaryType::ELWADD`, `#define MATH_FIDELITY MathFidelity::HiFi4`).
- `chlkc_descriptors.h`: Generated by `jit_build_genfiles_descriptors()`, contains data format arrays (`unpack_src_format[]`, `unpack_dst_format[]`, `pack_src_format[]`, `pack_dst_format[]`), tile dimensions, and math configuration. Uses the `tt_hlk_desc` structure to carry CB data formats from host to kernel compilation.
- Compilation is per-kernel: each `CreateKernel()` call triggers independent JIT compilation with its own set of defines and data formats.

---

### Chapter 5: `ch5_op_to_llk_mapping`

**Description:** How TT-Metal and TTNN operations map to specific LLK API calls through compute kernel files.

**Files:**

#### `index.md`
- Chapter introduction and summary mapping table.

#### `mapping_pattern.md`
- General pattern: A TTNN op program factory calls `tt_metal::CreateKernel()` with a compute kernel `.cpp` path, a `ComputeConfig` containing defines that select the specific operation variant, and runtime/compile-time args for parameterization.
- The compute kernel `.cpp` includes user-facing API headers (e.g., `api/compute/eltwise_binary.h`), which internally call LLK API wrappers, which call raw LLK implementations.
- Key defines that act as operation selectors: `ELTWISE_OP_TYPE`, `ELTWISE_OP`, `ELTWISE_OP_INIT`, `SFPU_OP_CHAIN_0`, `REDUCE_OP`, `REDUCE_DIM`, `MATH_FIDELITY`, `DST_ACCUM_MODE`, `PACK_RELU`.

#### `eltwise_ops.md`
- Binary eltwise: `ttnn/.../binary/device/element_wise_multi_core_program_factory.cpp` -> `eltwise_binary_kernel.cpp` -> `api/compute/eltwise_binary.h` -> `llk_math_eltwise_binary_init_with_operands()` + `llk_math_eltwise_binary()` (via `_llk_math_eltwise_binary_()` which emits `TT_OP_ELWADD`/`TT_OP_ELWSUB`/`TT_OP_ELWMUL` Tensix instructions).
- Unary eltwise (SFPU): Uses headers in `api/compute/eltwise_unary/` (exp, gelu, relu, etc.). Maps to `llk_math_eltwise_unary_sfpu` functions. The `sfpu_split_includes.h` mechanism includes the appropriate SFPU operation chain.
- The `SFPU_OP_CHAIN_0` macro pattern for fusing SFPU operations after binary/unary ops.

#### `matmul_ops.md`
- Simple matmul: `bmm.cpp` -> `api/compute/matmul.h` -> `mm_init()` calls `llk_math_matmul_init()` + `llk_unpack_AB_matmul_init()`. `matmul_tiles()` calls `llk_math_matmul()` + `llk_unpack_AB_matmul()`.
- Large block matmul: `bmm_large_block_zm.cpp` uses `matmul_block()` for blocked matmul with configurable block dimensions.
- Fused variants: `bmm_large_block_zm_fused_bias_activation.cpp` adds bias and activation fusion.
- Blackhole-specific: `matmul_block_math_dynamic_throttle()` in `matmul.h` -- firmware-controlled dynamic throttling via `MEM_L1_ARC_FW_SCRATCH` to manage power/thermal.

#### `other_ops.md`
- Reduce: `api/compute/reduce.h` -> `llk_math_reduce_*`, `llk_unpack_AB_reduce_*`. Parameterized by `REDUCE_OP` and `REDUCE_DIM`.
- Tilize/Untilize: `api/compute/tilize.h` / `untilize.h` -> `llk_unpack_tilize_*`, `llk_pack_untilize_*`.
- Tile move/copy: `api/compute/tile_move_copy.h` -> `llk_math_eltwise_unary_datacopy_*`, `llk_unpack_A_*`.
- Broadcast: `api/compute/bcast.h` -> binary eltwise with `BroadcastType` template parameter.
- Pack: `api/compute/pack.h` -> `llk_pack()`.

---

### Chapter 6: `ch6_hardware_target_configuration`

**Description:** How TT-Metal configures LLK for different hardware targets (Blackhole, Wormhole B0, Quasar).

**Files:**

#### `index.md`
- Chapter introduction.

#### `arch_differences.md`
- Directory structure per arch: parallel trees in TT-LLK (`tt_llk_blackhole/`, etc.) and TT-Metal ckernels (`hw/ckernels/blackhole/`, etc.).
- Compiler CPU flags: `-mcpu=tt-wh` (Wormhole), `-mcpu=tt-bh` (Blackhole). Quasar uses `tt-qsr32` and `tt-qsr64` CPU targets.
- Wormhole/Blackhole use "old-style" tt-1xx build path. Quasar uses "TLS-style" tt-2xx build path with Thread Local Storage and separate linker scripts (`script_tng.ld` vs. `main.ld`).
- Quasar has fourth TRISC file: `chlkc_isolate_sfpu.cpp` for SFPU isolation.

#### `arch_defines_and_guards.md`
- `ARCH_BLACKHOLE`, `ARCH_WORMHOLE`, `ARCH_QUASAR` defines added by per-arch HAL `defines()` methods.
- Conditional compilation in compute API headers: `#ifndef ARCH_QUASAR` guards in `compute_kernel_api.h` exclude binary, SFPU, reduce, tilize/untilize APIs on Quasar (not yet implemented). `#ifdef ARCH_BLACKHOLE` for dynamic matmul throttling.
- Quasar-specific: `thread_local` qualifiers for RTA base pointers (`rta_l1_base`, `crta_l1_base`) and watcher counters.
- Quasar's smaller `llk_api/` directory reflects a newer, in-development API surface.

---

### Chapter 7: `ch7_runtime_parameters`

**Description:** How TT-Metal passes runtime and compile-time parameters to LLK kernels, and how data formats are plumbed through.

**Files:**

#### `index.md`
- Chapter introduction.

#### `compile_time_args.md`
- Compile-time arguments via `ComputeConfig.compile_args` (positional) and `ComputeConfig.named_compile_args` (by name).
- Injected as `-DKERNEL_COMPILE_TIME_ARGS=v0,v1,...` and `-DKERNEL_COMPILE_TIME_ARG_MAP=...` defines during JIT compilation (in `compile_one()`, `build.cpp` lines 482-506).
- Accessed in kernels via `get_compile_time_arg_val(index)` and `get_named_compile_time_arg_val("name")` (from `hw/inc/api/compile_time_args.h`).
- Additional defines via `ComputeConfig.defines` map -> written to `defines_generated.h` and also passed as `-D` flags.

#### `runtime_args.md`
- Runtime arguments via `SetRuntimeArgs()` API on the host. Written to L1 memory at `rta_l1_base`.
- Accessed in kernels via `get_arg_val<T>(index)` (defined in `hw/inc/api/compute/common.h`).
- Common runtime args via `SetCommonRuntimeArgs()` -> `crta_l1_base`, accessed via `get_common_arg_val<T>()`.
- Example: `eltwise_binary.cpp` reads `per_core_block_cnt` and `per_core_block_size` as runtime args at indices 0 and 1.
- Watcher asserts check runtime arg bounds when `WATCHER_ENABLED` is defined.

#### `data_format_plumbing.md`
- Data formats configured per circular buffer (CB) on the host via `CircularBufferConfig`.
- During JIT compilation, `genfiles.cpp` generates `chlkc_descriptors.h` containing format arrays: `unpack_src_format[]`, `unpack_dst_format[]`, `pack_src_format[]`, `pack_dst_format[]`.
- These arrays are consumed by LLK `hw_configure` and `init` functions to configure unpackers and packers for the correct data format conversion.
- `tt_hlk_desc` structure (from `jit_build/hlk_desc.hpp`) carries the CB data format array from host to code generation.
- Math fidelity (`MATH_FIDELITY`) and FP32 dest accumulation (`DST_ACCUM_MODE`) are compile-time template parameters that affect LLK behavior.

---

### Chapter 8: `ch8_abstraction_layers_and_extension`

**Description:** The wrappers and abstraction boundaries between user-facing APIs and raw LLK calls, and practical guidance for extending the system.

**Files:**

#### `index.md`
- Chapter introduction.

#### `abstraction_boundaries.md`
- Four clear boundaries:
  1. **TTNN op Python/C++ API -> Program factory** (host-side): selects kernel paths and defines.
  2. **Program factory -> `CreateKernel()`** (host-to-device boundary): specifies kernel file, `ComputeConfig` with defines/args, core ranges.
  3. **Compute kernel `.cpp` -> `api/compute/*.h`** (kernel author boundary): user code calls descriptive functions like `matmul_tiles()`, `binary_op_init_common()`.
  4. **User-facing API -> LLK API wrappers -> raw LLK implementations** (internal boundary): `matmul_tiles()` -> `llk_math_matmul()` -> `_llk_math_matmul_()`.
- The MATH/PACK/UNPACK macro system as a compile-time abstraction.
- `state_configure()` as a reconfiguration guard avoiding redundant hardware reconfig.

#### `adding_new_ops.md`
- To add a new SFPU operation: (1) add `ckernel_sfpu_<op>.h` in TT-LLK `common/inc/sfpu/`, (2) add wrapper in `ckernels/<arch>/metal/llk_api/llk_sfpu/`, (3) add user-facing header in `hw/inc/api/compute/eltwise_unary/`, (4) wire into `sfpu_split_includes.h`, (5) create compute kernel `.cpp`, (6) create program factory in TTNN.
- To add a new FPU operation: modify LLK lib (`llk_math_*.h`), add `_api.h` wrapper, add compute API header, create kernel, create factory.
- The `experimental/` directories in both TT-LLK `llk_lib/experimental/` and TT-Metal `ckernels/<arch>/metal/llk_api/experimental/` for in-progress features.

#### `testing_and_debugging.md`
- TT-Metal LLK tests: `tests/tt_metal/tt_metal/llk/` -- test individual LLK operations through TT-Metal's CreateKernel/Program infrastructure.
- TT-LLK standalone tests: `tests/` directory in the TT-LLK repo with its own Python test suite and hardware-specific tests.
- `ENABLE_LLK_ASSERT` define (from `rtoptions.get_llk_asserts()`) enables assertions in LLK code.
- The `fake_kernels_target` CMake target compiles kernels at build time, catching errors without hardware.
- Debugging tips: inspect generated files (`chlkc_descriptors.h`, `chlkc_math.cpp`, `defines_generated.h`) in the JIT build cache directory.

---

## 3. Conventions

1. **File paths** are always relative to the repository root of the referenced codebase. Paths starting with `tt_metal/` refer to the TT-Metal repo; paths starting with `tt_llk_*` or `common/` refer to the TT-LLK repo (or equivalently the submodule at `tt_metal/third_party/tt_llk/`).
2. **Architecture names**: "Blackhole" or `blackhole`, "Wormhole B0" or `wormhole_b0`, "Quasar" or `quasar` in prose. Directory names use the lowercase forms. The CMake variable `ARCH_ALIAS` maps `wormhole` to `wormhole_b0`.
3. **LLK function naming**: Functions prefixed with `_` and suffixed with `_` (e.g., `_llk_math_eltwise_binary_()`) are internal implementations in `llk_lib/`. Functions without underscores (e.g., `llk_math_eltwise_binary()`) are in the `llk_api/` wrapper layer. Functions with descriptive names (e.g., `binary_op_init_common()`, `matmul_tiles()`) are the user-facing compute API.
4. **TRISC identifiers**: TRISC0 = Unpack, TRISC1 = Math, TRISC2 = Pack. Preprocessor guards: `TRISC_UNPACK`, `TRISC_MATH`, `TRISC_PACK`. HAL defines: `UCK_CHLKC_UNPACK`, `UCK_CHLKC_MATH`, `UCK_CHLKC_PACK`.
5. **Code examples** show actual function signatures from the codebase with file path annotations. Inline comments mark which layer each function belongs to.
6. **Diagrams** use text-based ASCII art for portability. Each chapter describing a flow includes at least one diagram.
7. **Cross-repo references**: When a file exists in both repos (because TT-LLK is vendored into TT-Metal via submodule), always clarify whether the path is in TT-LLK standalone or in TT-Metal's `third_party/tt_llk/`.
8. **Terms**: "Compute API" refers to `hw/inc/api/compute/*.h`. "LLK API" refers to `ckernels/<arch>/metal/llk_api/*.h`. "LLK lib" refers to `tt_llk_<arch>/llk_lib/*.h`. "Ckernel headers" refers to `tt_llk_<arch>/common/inc/*.h`.

---

## 4. Cross-Chapter Dependencies

| Chapter | Depends On | Reason |
|---------|-----------|--------|
| Ch2 (Build System) | Ch1 (Architecture Overview) | Needs understanding of directory structure, submodule location, and TRISC model. |
| Ch3 (API Layer Architecture) | Ch1 (Architecture Overview), Ch2 (Build System) | Needs to know where headers live and how include paths are resolved. |
| Ch4 (Kernel Compilation) | Ch2 (Build System), Ch3 (API Layers) | Needs JIT pipeline context and understanding of which headers get included per TRISC. |
| Ch5 (Op-to-LLK Mapping) | Ch3 (API Layers), Ch4 (Kernel Compilation) | Needs the full layer stack and compilation model to trace from op to LLK call. |
| Ch6 (Hardware Targets) | Ch2 (Build System), Ch3 (API Layers) | Needs the include path mechanism and conditional compilation system. |
| Ch7 (Runtime Parameters) | Ch4 (Kernel Compilation), Ch5 (Op Mapping) | Needs the code generation and op plumbing context to understand how args flow. |
| Ch8 (Abstraction & Extension) | All previous chapters | Synthesizes the full picture and discusses extension points across all layers. |

---

## Key Source Files Reference

This table lists the most important files for each topic, to guide writers during content creation.

| Topic | Key Files (TT-Metal) | Key Files (TT-LLK) |
|-------|----------------------|---------------------|
| Submodule wiring | `.gitmodules`, `tt_metal/third_party/CMakeLists.txt` | -- |
| Offline firmware build | `tt_metal/hw/CMakeLists.txt` (lines 354-380 for includes) | -- |
| JIT build environment | `tt_metal/jit_build/build.cpp` (`JitBuildEnv::init`, `JitBuildState` ctor, `compile_one`) | -- |
| HAL include paths | `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp`, `wormhole/wh_hal.cpp`, `tt-2xx/quasar/qa_hal.cpp` | -- |
| HAL common defines | `tt_metal/llrt/hal/tt-1xx/hal_1xx_common.cpp`, `tt-2xx/hal_2xx_common.cpp` | -- |
| Code generation | `tt_metal/jit_build/genfiles.cpp` | -- |
| TRISC entry point | `tt_metal/hw/ckernels/<arch>/metal/common/chlkc_list.h` | -- |
| LLK API wrappers | `tt_metal/hw/ckernels/<arch>/metal/llk_api/*.h` | -- |
| LLK IO layer | `tt_metal/hw/ckernels/<arch>/metal/llk_io/*.h` | -- |
| SFPU extensions | `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/*.h` | -- |
| User compute API | `tt_metal/hw/inc/api/compute/*.h` (esp. `compute_kernel_api.h`, `common_globals.h`) | -- |
| LLK implementations | -- | `tt_llk_<arch>/llk_lib/llk_math_*.h`, `llk_pack*.h`, `llk_unpack_*.h` |
| Ckernel primitives | -- | `tt_llk_<arch>/common/inc/ckernel*.h`, `cmath_common.h`, `cpack_common.h` |
| SFPU base implementations | -- | `tt_llk_<arch>/common/inc/sfpu/ckernel_sfpu_*.h` |
| Shared compute kernels | `tt_metal/kernels/compute/*.cpp` | -- |
| Op-specific compute kernels | `ttnn/cpp/ttnn/operations/*/device/kernels/compute/*.cpp` | -- |
| Op program factories | `ttnn/cpp/ttnn/operations/*/device/*_program_factory.cpp` | -- |
| LLK tests in Metal | `tests/tt_metal/tt_metal/llk/` | -- |
| LLK standalone tests | -- | `tests/` |
