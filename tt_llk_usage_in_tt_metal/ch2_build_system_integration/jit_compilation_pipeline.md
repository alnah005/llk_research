# JIT Compilation Pipeline

When a user submits a compute kernel to TT-Metal at runtime, it is compiled just-in-time (JIT) by invoking the device compiler. The JIT build environment must assemble the correct set of include paths so that the kernel source can `#include` LLK headers. This happens in two stages: a base set of includes common to all architectures, and a per-architecture set injected by the Hardware Abstraction Layer (HAL).

## Base Include Paths

The `JitBuildEnv` constructor in `tt_metal/jit_build/build.cpp` (lines 269-285) sets up a base set of include directories that apply to every JIT-compiled kernel regardless of architecture:

```cpp
// tt_metal/jit_build/build.cpp, lines 268-285
// Includes
// TODO(pgk) this list is insane
std::vector<std::string> includeDirs = {
    ".",
    "..",
    root_,
    root_ + "ttnn",
    root_ + "ttnn/cpp",
    root_ + "tt_metal",
    root_ + "tt_metal/hw/inc",
    root_ + "tt_metal/third_party/tt_llk/common",
    root_ + "tt_metal/hostdevcommon/api",
    root_ + "tt_metal/api/"};

std::ostringstream oss;
for (const auto& includeDir : includeDirs) {
    oss << "-I" << includeDir << " ";
}
this->includes_ = oss.str();
```

The key LLK path here is `tt_metal/third_party/tt_llk/common`, which provides the architecture-independent LLK headers. The architecture-specific LLK paths (`tt_llk_wormhole_b0/`, etc.) are NOT set here -- they come from the HAL, as described next.

## Per-Architecture Includes via HAL

Each hardware architecture registers its own include paths through a HAL initialization function. These functions are called at runtime when the system detects which hardware is present. The HAL populates a `HalJitBuildConfig` struct with include paths that the JIT compiler appends to the base set above.

### Wormhole B0

```cpp
// tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp, lines 105-121
includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/common");
includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_io");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole/noc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib");

switch (params.core_type) {
    case HalProgrammableCoreType::TENSIX:
        switch (params.processor_class) {
            case HalProcessorClassType::COMPUTE:
                includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_api");
                includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu");
                break;
            case HalProcessorClassType::DM: break;
        }
        includes.push_back("tt_metal/hw/firmware/src/tt-1xx");
        break;
```

### Blackhole

```cpp
// tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp, lines 107-136
includes.push_back("tt_metal/hw/ckernels/blackhole/metal/common");
includes.push_back("tt_metal/hw/ckernels/blackhole/metal/llk_io");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole/noc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib");

switch (params.core_type) {
    case HalProgrammableCoreType::TENSIX:
        switch (params.processor_class) {
            case HalProcessorClassType::COMPUTE:
                includes.push_back("tt_metal/hw/ckernels/blackhole/metal/llk_api");
                includes.push_back("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu");
                break;
            case HalProcessorClassType::DM: break;
        }
        break;
    case HalProgrammableCoreType::ACTIVE_ETH: {
        includes.push_back("tt_metal/hw/inc/ethernet");
        break;
    }
    case HalProgrammableCoreType::IDLE_ETH: break;
    // ...
}
includes.push_back("tt_metal/hw/firmware/src/tt-1xx");  // line 135: OUTSIDE the switch
return includes;
```

**Structural difference from Wormhole:** On Blackhole, the `tt_metal/hw/firmware/src/tt-1xx` include is pushed **after** the `switch` statement (line 135), meaning it applies to **all** core types (TENSIX, ACTIVE_ETH, IDLE_ETH). On Wormhole, the equivalent `includes.push_back("tt_metal/hw/firmware/src/tt-1xx")` appears **inside** the `TENSIX` case (line 123 of `wh_hal.cpp`) and separately inside `IDLE_ETH`, so `ACTIVE_ETH` cores on Wormhole do not receive this path. This matters because the `tt-1xx` firmware source directory contains RISC-V startup code and firmware headers that Blackhole makes available to all programmable core types uniformly.

### Quasar

```cpp
// tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp, lines 226-244
includes.push_back("tt_metal/hw/ckernels/quasar/metal/common");
includes.push_back("tt_metal/hw/ckernels/quasar/metal/llk_io");
includes.push_back("tt_metal/hw/inc/internal");
includes.push_back("tt_metal/hw/inc/internal/tt-2xx");
includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar");
includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar/quasar_defines");
includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar/noc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_quasar/");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_quasar/llk_lib");

switch (params.core_type) {
    case HalProgrammableCoreType::TENSIX:
        switch (params.processor_class) {
            case HalProcessorClassType::COMPUTE:
                includes.push_back("tt_metal/hw/ckernels/quasar/metal/llk_api");
                includes.push_back("tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu");
                break;
            case HalProcessorClassType::DM: break;
        }
        break;
```

Note that Quasar adds an extra include for the `tt_llk_quasar/` root directory itself (line 234), in addition to the `common/inc` and `llk_lib` subdirectories.

## COMPUTE-Only LLK API Includes

A critical pattern visible across all three architectures: **only TENSIX cores with `HalProcessorClassType::COMPUTE` receive the `llk_api` and `llk_api/llk_sfpu` include paths.** Data-movement processors (`HalProcessorClassType::DM`) and Ethernet cores (`ACTIVE_ETH`, `IDLE_ETH`) do not get these paths because they do not run compute kernels and have no use for the LLK math/unpack/pack API.

The two COMPUTE-only include directories serve different roles:

| Path | Purpose |
|---|---|
| `hw/ckernels/${ARCH}/metal/llk_api` | Metal-layer wrapper functions (`llk_unpack_*`, `llk_math_*`, `llk_pack_*`) that call into the LLK library |
| `hw/ckernels/${ARCH}/metal/llk_api/llk_sfpu` | SFPU (Special Function Processing Unit) kernel wrappers for transcendental and custom math operations |

The common includes that **all** core types receive are:

| Path | Purpose |
|---|---|
| `hw/ckernels/${ARCH}/metal/common` | Shared ckernel infrastructure (register definitions, base types) |
| `hw/ckernels/${ARCH}/metal/llk_io` | I/O helpers for circular buffer access |
| `third_party/tt_llk/tt_llk_${ARCH}/common/inc` | Architecture-specific LLK common headers |
| `third_party/tt_llk/tt_llk_${ARCH}/llk_lib` | Core LLK library implementations |

## Build Key and Caching

After assembling the include paths, the JIT build environment computes a hash-based build key (lines 291-298) that incorporates the architecture, compiler flags, linker flags, defines, and SFPI version. This key is used to cache compiled binaries so that identical kernel configurations are not recompiled:

```cpp
// tt_metal/jit_build/build.cpp, lines 291-298
tt::FNV1a hasher;
hasher.update(build_key);
hasher.update(enchantum::to_underlying(this->arch_));
hasher.update(cflags_);
hasher.update(lflags_);
hasher.update(defines_);
hasher.update(sfpi_version_contents);
build_key_ = hasher.digest();
```

Note that the include paths (`this->includes_`) are **not** part of this first hash. The `build_key_` hash is used only for **output directory routing** -- choosing which directory compiled artifacts are written to. It does not, by itself, determine whether a cached build is still valid.

Cache invalidation is handled by a **second hash**, `build_state_hash_`, computed in the `JitBuildState` constructor (lines 419-434). This hash covers a broader set of parameters and **does** include `includes_`:

```cpp
// tt_metal/jit_build/build.cpp, lines 419-434
{
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
}
```

Before every JIT compilation, `build_state_matches()` (line 684) checks whether the stored `build_state_hash_` in the output directory matches the current one. If any hashed parameter has changed -- including include paths -- the cached objects are invalidated and a full recompilation is triggered. This means the build system uses a two-level hashing scheme:

| Level | Hash field | Purpose | Includes `includes_`? |
|---|---|---|---|
| `JitBuildEnv` | `build_key_` | Output directory routing | No |
| `JitBuildState` | `build_state_hash_` | Cache invalidation | **Yes** |

Changes to include paths (e.g., adding a new `-I` directory or modifying HAL-provided paths) **do** invalidate the build cache through `build_state_hash_`, ensuring stale binaries are never reused.

---

**Next:** [`per_kernel_code_generation.md`](./per_kernel_code_generation.md)
