# HAL Architecture Dispatch

The Hardware Abstraction Layer (HAL) is the single runtime object that tells the JIT compiler which LLK directories to include, which architecture defines to set, and which CPU target flags to use. This section traces the dispatch from the `Hal` constructor down to the per-architecture `includes()` and `defines()` methods that ultimately select the correct TT-LLK implementation.

## The `Hal` Constructor Switch

When TT-Metal opens a device, it constructs a `Hal` object with the detected `tt::ARCH` enum. The constructor in `tt_metal/llrt/hal.cpp` (line 34) dispatches immediately:

```cpp
// tt_metal/llrt/hal.cpp:34-51
Hal::Hal(
    tt::ARCH arch,
    bool is_base_routing_fw_enabled,
    bool enable_2_erisc_mode,
    uint32_t profiler_dram_bank_size_per_risc_bytes) :
    arch_(arch) {
    switch (this->arch_) {
        case tt::ARCH::WORMHOLE_B0:
            initialize_wh(is_base_routing_fw_enabled, profiler_dram_bank_size_per_risc_bytes);
            break;
        case tt::ARCH::QUASAR:
            initialize_qa(profiler_dram_bank_size_per_risc_bytes);
            break;
        case tt::ARCH::BLACKHOLE:
            initialize_bh(enable_2_erisc_mode, profiler_dram_bank_size_per_risc_bytes);
            break;
        default: break;
    }
}
```

Each `initialize_*` function lives in its own file under `tt_metal/llrt/hal/`:

| Architecture | File | Initializer | JIT Query Class |
|---|---|---|---|
| Wormhole | `tt-1xx/wormhole/wh_hal.cpp` | `initialize_wh()` | `HalJitBuildQueryWormhole` |
| Blackhole | `tt-1xx/blackhole/bh_hal.cpp` | `initialize_bh()` | `HalJitBuildQueryBlackHole` |
| Quasar | `tt-2xx/quasar/qa_hal.cpp` | `initialize_qa()` | `HalJitBuildQueryQuasar` |

Wormhole and Blackhole both inherit from `hal_1xx::HalJitBuildQueryBase` (declared in `tt-1xx/hal_1xx_common.hpp`), while Quasar inherits from `hal_2xx::HalJitBuildQueryBase` (declared in `tt-2xx/hal_2xx_common.hpp`). The `1xx` vs `2xx` split reflects Tenstorrent's hardware generation boundary.

## How `includes()` Selects LLK Paths

Each `HalJitBuildQuery*` subclass overrides `includes()` to return the architecture-specific include directories that the JIT compiler passes as `-I` flags. These paths control which LLK headers and which `ckernels/` shim layer the compiled kernel sees.

### Wormhole

From `wh_hal.cpp` lines 101-135:

```cpp
std::vector<std::string> includes(const Params& params) const override {
    std::vector<std::string> includes;

    // Common includes for all core types
    includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/common");
    includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_io");
    includes.push_back("tt_metal/hw/inc/internal/tt-1xx");
    includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole");
    includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines");
    includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole/noc");
    includes.push_back("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc");
    includes.push_back("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib");

    // For TENSIX Compute cores, also include the LLK API shim
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
        // ...
    }
    return includes;
}
```

### Blackhole

From `bh_hal.cpp` lines 103-137. The paths point to `blackhole/` and `tt_llk_blackhole/`, but the structure differs from Wormhole in one important way: Wormhole pushes `firmware/src/tt-1xx` inside the TENSIX case only (so it applies only to Tensix cores), while Blackhole pushes it *after* the switch statement, meaning it applies to all core types:

```cpp
includes.push_back("tt_metal/hw/ckernels/blackhole/metal/common");
includes.push_back("tt_metal/hw/ckernels/blackhole/metal/llk_io");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines");
includes.push_back("tt_metal/hw/inc/internal/tt-1xx/blackhole/noc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib");
```

### Quasar

From `qa_hal.cpp` lines 222-257. Quasar uses `tt-2xx` internal paths and adds an extra root-level include for the `tt_llk_quasar/` directory itself:

```cpp
includes.push_back("tt_metal/hw/ckernels/quasar/metal/common");
includes.push_back("tt_metal/hw/ckernels/quasar/metal/llk_io");
includes.push_back("tt_metal/hw/inc/internal");
includes.push_back("tt_metal/hw/inc/internal/tt-2xx");
includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar");
includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar/quasar_defines");
includes.push_back("tt_metal/hw/inc/internal/tt-2xx/quasar/noc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc");
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_quasar/");          // root of quasar LLK
includes.push_back("tt_metal/third_party/tt_llk/tt_llk_quasar/llk_lib");
```

Note the extra `tt_metal/hw/inc/internal` (without `tt-1xx` or `tt-2xx` suffix) and the inclusion of the `tt_llk_quasar/` root directory -- neither of which appear in the Wormhole/Blackhole paths.

## How `defines()` Sets the Architecture Macro

Each subclass appends a single architecture-identifying preprocessor define. This define propagates into every LLK header and every Compute API header, enabling `#ifdef`/`#ifndef` guards to select architecture-specific code paths.

| Architecture | Define pushed | Source location |
|---|---|---|
| Wormhole | `ARCH_WORMHOLE` | `wh_hal.cpp:146` |
| Blackhole | `ARCH_BLACKHOLE` | `bh_hal.cpp:141` |
| Quasar | `ARCH_QUASAR` | `qa_hal.cpp:262` |

Blackhole's `defines()` also conditionally pushes `ENABLE_2_ERISC_MODE` and `PHYSICAL_AERISC_ID=<N>` for Active Ethernet cores. Wormhole pushes `ENABLE_IRAM` and `COOPERATIVE_ERISC` for its Active ETH case. Quasar's defines are simpler -- just the base class defines plus `ARCH_QUASAR`.

## How `common_flags()` Selects the CPU Target

The `-mcpu` flag tells the RISC-V compiler which instruction set extensions are available on the target silicon. Each architecture uses distinct CPU target names:

| Architecture | Compute (TENSIX) | Data Movement / Other |
|---|---|---|
| Wormhole | `-mcpu=tt-wh-tensix` | `-mcpu=tt-wh` |
| Blackhole | `-mcpu=tt-bh-tensix` | `-mcpu=tt-bh` |
| Quasar | `-mcpu=tt-qsr32-tensixbh` | `-mcpu=tt-qsr64-rocc` |

Quasar is notably different: it uses a 32-bit RISC-V for Tensix compute cores (`tt-qsr32`) and a 64-bit RISC-V for data movement cores (`tt-qsr64`). This dual-ISA design is reflected throughout the build system (see [Firmware Compilation](./firmware_compilation.md) for the CMake-level handling).

## Summary of the Include Path Matrix

For a Tensix Compute kernel, the full set of LLK-relevant include paths per architecture:

| Path component | Wormhole | Blackhole | Quasar |
|---|---|---|---|
| `ckernels/*/metal/common` | `wormhole_b0` | `blackhole` | `quasar` |
| `ckernels/*/metal/llk_io` | `wormhole_b0` | `blackhole` | `quasar` |
| `ckernels/*/metal/llk_api` | `wormhole_b0` | `blackhole` | `quasar` |
| `ckernels/*/metal/llk_api/llk_sfpu` | `wormhole_b0` | `blackhole` | `quasar` |
| `tt_llk_*/common/inc` | `tt_llk_wormhole_b0` | `tt_llk_blackhole` | `tt_llk_quasar` |
| `tt_llk_*/llk_lib` | `tt_llk_wormhole_b0` | `tt_llk_blackhole` | `tt_llk_quasar` |
| Internal HW inc | `tt-1xx/wormhole` | `tt-1xx/blackhole` | `tt-2xx/quasar` |
| Firmware src | `firmware/src/tt-1xx` | `firmware/src/tt-1xx` | `firmware/src/tt-2xx` |

The key insight: every architecture gets its own complete set of LLK includes. There is no runtime fallback or shared path -- the HAL selects the full set at JIT time based on the detected hardware.

---

**Next:** [`firmware_compilation.md`](./firmware_compilation.md)
