# JIT Compilation Pipeline

This section traces how a compute kernel goes from source code to a compiled device binary at runtime, focusing on the role of LLK headers throughout the pipeline.

## Overview

The JIT pipeline has four stages:

1. **Environment setup** (`JitBuildEnv::init`) -- assembles toolchain paths, global flags, defines, and the base include path list.
2. **State construction** (`JitBuildState`) -- queries the HAL for per-architecture includes, defines, and source files; computes a cache key.
3. **Generated files** (`genfiles.cpp`) -- produces TRISC-specific `.cpp` files and the `chlkc_descriptors.h` header.
4. **Compilation and caching** -- compiles, links, and caches the result keyed by an FNV1a hash.

## Stage 1: JitBuildEnv::init()

**Source:** [`tt_metal/jit_build/build.cpp:91-299`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L91)

`JitBuildEnv::init()` is called once per device configuration. It sets up:

### Toolchain Selection (lines 104-130)

The method locates the RISC-V cross compiler (`riscv-tt-elf-g++`) by searching two paths in order:

1. `<root>/runtime/sfpi` (local development)
2. `/opt/tenstorrent/sfpi` (system install)

If `TT_METAL_CCACHE_KERNEL_SUPPORT` is set, `ccache` is prepended to the compiler command.

### Compiler Flags (lines 147-265)

Base flags: `-std=c++17 -flto=auto -ffast-math -fno-exceptions`

Conditional defines are appended based on runtime options:

- `-DROUTING_FW_ENABLED` -- when routing firmware is enabled (line 252)
- `-DLIGHTWEIGHT_KERNEL_ASSERTS` -- lightweight asserts (line 256)
- `-DENABLE_LLK_ASSERT` -- enables assertions inside LLK Layer 1 code (line 260)
- `-DDISABLE_SFPLOADMACRO` -- disables SFP load macros (line 264)

The LLK assert define is particularly important: it bridges a runtime option in Metal to a compile-time behavior in the LLK submodule, using the `llk_assert.h` header from `[tt-llk] common/`.

### Base Include Paths (lines 268-285)

The static include list (the "insane" list discussed in [`submodule_mechanics.md`](./submodule_mechanics.md)) includes `tt_metal/third_party/tt_llk/common` as a global include for all architectures. These paths are formatted into `-I` flags and stored in `this->includes_`.

### Build Key (lines 291-298)

An FNV1a hash is computed over the build key, architecture, compiler flags, linker flags, defines, and the sfpi version file contents. This `build_key_` determines the output directory layout:

```cpp
this->out_firmware_root_ = fmt::format("{}{}/firmware/", this->out_root_, build_key_);
this->out_kernel_root_ = fmt::format("{}{}/kernels/", this->out_root_, build_key_);
```

**Source:** [`build.cpp:291-301`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L291)

## Stage 2: JitBuildState Construction

**Source:** [`tt_metal/jit_build/build.cpp:305-434`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L305)

`JitBuildState` is constructed per processor class (BRISC, NCRISC, TRISC COMPUTE, etc.). It inherits the environment's base flags and extends them with architecture- and core-type-specific settings from the HAL.

### HAL Query

The constructor builds a `HalJitBuildQueryInterface::Params` struct (lines 330-335) and queries:

- **`jit_build_query.includes(params)`** (line 342) -- returns architecture-specific include directories. For Wormhole COMPUTE, this adds the Layer 1 LLK path (`tt_llk_wormhole_b0/llk_lib`) and Layer 2 wrapper path (`hw/ckernels/wormhole_b0/metal/llk_api`). See [`wh_hal.cpp:101-134`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp#L101).
- **`jit_build_query.defines(params)`** (line 349) -- returns architecture-specific defines (e.g., `ARCH_WORMHOLE`).
- **`jit_build_query.srcs(params)`** (line 370) -- returns source files to compile.
- **`jit_build_query.common_flags(params)`** (line 361) -- additional compiler/linker flags.

### COMPUTE-Specific Behavior

For COMPUTE processors (lines 317-322), the constructor makes two notable adjustments:

1. **Optimization level** is set to `O3` (vs `Os` for other processors) -- math kernels need peak performance.
2. **`process_defines_at_compile_`** is set to `false` -- defines are deferred to a later stage, allowing per-TRISC customization.
3. The sfpi compiler's own include directory (`gpp_include_dir_`) is added to the include path.

### Build State Hash (lines 419-433)

A second FNV1a hash captures all effective compilation parameters (compiler path, flags, defines, includes, linker script, source list, optimization levels). This hash is stored in a `.build_state` file in the output directory. On subsequent builds, `build_state_matches()` (line 439) compares the stored hash against the current one. A mismatch triggers recompilation.

```cpp
tt::FNV1a hasher;
hasher.update(env_.gpp_);
hasher.update(cflags_);
hasher.update(defines_);
hasher.update(includes_);
hasher.update(lflags_);
hasher.update(linker_script_);
hasher.update(extra_link_objs_);
for (const auto& src : srcs_) {
    hasher.update(src);
}
hasher.update(default_compile_opt_level_);
hasher.update(default_linker_opt_level_);
build_state_hash_ = hasher.digest();
```

**Source:** [`build.cpp:420-433`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L420)

## Stage 3: Generated Files

**Source:** [`tt_metal/jit_build/genfiles.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/genfiles.cpp)

### TRISC Source File Generation

The function `jit_build_genfiles_triscs_src()` (line 171) takes a single kernel source file and produces four TRISC-specific source files:

| Generated File | TRISC Processor | Purpose |
|----------------|-----------------|---------|
| `chlkc_unpack.cpp` | TRISC_UNPACK | Unpacking tiles from circular buffers |
| `chlkc_math.cpp` | TRISC_MATH | Math operations (eltwise, matmul, reduce) |
| `chlkc_pack.cpp` | TRISC_PACK | Packing results back to circular buffers |
| `chlkc_isolate_sfpu.cpp` | TRISC_ISOLATE_SFPU | Isolated SFPU operations (Quasar) |

Each generated file is prefixed with a "prolog" that defines the processor identity (e.g., `TRISC_UNPACK`), which controls which Layer 2 headers are included via `#ifdef` guards in Layer 3 headers.

The generator supports two kernel syntax styles:

- **Simplified syntax:** `void kernel_main()` -- the generator transforms this into the legacy namespace structure using `simple_kernel_syntax::transform_to_legacy_syntax()`, splitting the single function into per-TRISC entry points (`unpack_main`, `math_main`, `pack_main`).
- **Legacy syntax:** `namespace NAMESPACE { void MAIN() { ... } }` -- used as-is but triggers a deprecation warning (line 188).

### chlkc_descriptors.h Generation

The function `generate_all_descriptors()` (line 488) produces `chlkc_descriptors.h`, which encodes the data formats and tile dimensions for each TRISC processor:

```cpp
const string descriptors_path = options.path + "chlkc_descriptors.h";
```

The generated header uses conditional compilation blocks to emit only the relevant data for each TRISC:

- **`#if defined(UCK_CHLKC_MATH)`** -- emits math fidelity and approximation mode (via `emit_math_scalar_descriptors`, line 504).
- **`#if !defined(UCK_CHLKC_PACK)`** -- emits unpack data formats and tile dimensions (lines 507-509).
- **`#if !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_UNPACK)`** -- emits pack data formats and tile dimensions (lines 512-514).
- **Combined block** (line 517) -- emits compute scalar descriptors shared across TRISC processors.

This generated header is the bridge between Metal's runtime configuration (data formats chosen by the user) and the LLK functions that need compile-time constants for register configuration. The data format values (`compute_data_formats()` at line 499) flow from the user's `tt_hlk_desc` through this header and into LLK Layer 1 templates.

**Source:** [`genfiles.cpp:488-525`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/genfiles.cpp#L488)

## Stage 4: Compilation and Caching

### Compilation

`JitBuildState::compile_one()` (line 464) runs the cross compiler in the output directory:

```cpp
string cmd{"cd " + out_dir + " && " + env_.gpp_};
```

Each source file (both HAL-provided firmware sources and the generated `chlkc_*.cpp` files) is compiled to a `.o` object.

### Build Caching via FNV1a Hashing

The caching strategy uses two levels of hashing:

1. **Environment-level `build_key_`** (line 298) -- determines the output directory. If the architecture, global flags, or sfpi version change, the entire output tree is different.
2. **State-level `build_state_hash_`** (line 433) -- determines whether cached objects within that directory are still valid. This captures HAL-populated flags that may change between code revisions without changing the environment.

On each build invocation, `build_state_matches()` reads the `.build_state` file. If the stored hash differs from the current `build_state_hash_`, all objects are recompiled. Otherwise, individual object files are checked for staleness using `need_compile()` (line 545), which compares modification times.

After successful compilation, `write_build_state_hash()` (line 458) persists the new hash:

```cpp
void JitBuildState::write_build_state_hash(const string& out_dir) const {
    jit_build::utils::FileRenamer tmp(out_dir + string(BUILD_STATE_HASH_FILE));
    std::ofstream file(tmp.path());
    file << build_state_hash_;
}
```

This means that any change to an LLK header -- whether from a submodule update or a local edit -- will not automatically invalidate the cache unless it also changes a flag or source file path captured by the hash. The hash does not incorporate header file contents, only the compiler invocation parameters. Header-level change detection relies on the compiler's own dependency tracking and modification time checks.

---

**Next:** [Chapter 2 -- Advantages of the Current Integration Model](../ch2_advantages/index.md)
