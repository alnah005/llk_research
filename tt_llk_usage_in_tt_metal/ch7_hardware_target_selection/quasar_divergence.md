# Quasar Divergence

Quasar is Tenstorrent's tt-2xx generation hardware, and its integration with TT-LLK diverges substantially from the Wormhole/Blackhole (tt-1xx) pattern. This section catalogs the specific differences: a separate HAL base class, a distinct build loop, dual CPU targets, a smaller LLK API surface, and pervasive `ARCH_QUASAR` guards indicating features not yet ported.

## Different HAL Base Class

Wormhole and Blackhole JIT query classes both inherit from `hal_1xx::HalJitBuildQueryBase`, defined in `tt_metal/llrt/hal/tt-1xx/hal_1xx_common.hpp`:

```cpp
// tt_metal/llrt/hal/tt-1xx/hal_1xx_common.hpp
namespace tt::tt_metal::hal_1xx {

// Wormhole and Blackhole share many common build options.
class HalJitBuildQueryBase : public HalJitBuildQueryInterface {
public:
    HalJitBuildQueryBase(const Hal& hal) : hal_(hal) {}
    std::vector<std::string> defines(const Params& params) const override;
    std::vector<std::string> srcs(const Params& params) const override;
    std::string target_name(const Params& params) const override;
    std::string weakened_firmware_target_name(const Params& params) const override;
protected:
    const Hal& hal_;
};

}  // namespace tt::tt_metal::hal_1xx
```

Quasar has its own mirror class in `tt_metal/llrt/hal/tt-2xx/hal_2xx_common.hpp`:

```cpp
// tt_metal/llrt/hal/tt-2xx/hal_2xx_common.hpp
namespace tt::tt_metal::hal_2xx {

class HalJitBuildQueryBase : public HalJitBuildQueryInterface {
public:
    HalJitBuildQueryBase(const Hal& hal) : hal_(hal) {}
    std::vector<std::string> defines(const Params& params) const override;
    std::vector<std::string> srcs(const Params& params) const override;
    std::string target_name(const Params& params) const override;
    std::string weakened_firmware_target_name(const Params& params) const override;
protected:
    const Hal& hal_;
};

}  // namespace tt::tt_metal::hal_2xx
```

While the interface is the same, the `2xx` base provides Quasar-specific default implementations for `defines()`, `srcs()`, and `target_name()`. The implementations in `hal_2xx_common.cpp` and `hal_1xx_common.cpp` differ in which source files they return and what defines they inject.

## Dual CPU Targets: tt-qsr32 and tt-qsr64

Quasar uses two fundamentally different RISC-V ISAs within a single Tensix core:

| Processor class | CPU target | ISA | Usage |
|---|---|---|---|
| `COMPUTE` (trisc) | `tt-qsr32` | 32-bit RISC-V | Math/unpack/pack compute engines |
| `DM` (data movement) | `tt-qsr64` | 64-bit RISC-V with RoCC extensions | BRISC, NCRISC, Ethernet |

This is set in `HalJitBuildQueryQuasar::common_flags()` at `qa_hal.cpp` line 278:

```cpp
std::string cflags =
    params.processor_class == HalProcessorClassType::DM
        ? "-mcpu=tt-qsr64-rocc "
        : "-mcpu=tt-qsr32-tensixbh ";
cflags += "-fno-extern-tls-init ";
cflags += "-ftls-model=local-exec ";
```

The extra TLS (Thread-Local Storage) flags (`-fno-extern-tls-init`, `-ftls-model=local-exec`) are unique to Quasar and reflect its use of hardware TLS for multi-threaded RISC-V cores. This is why the CMake build loop for Quasar is called the "TLS-style" loop.

The `link_objs()` method also reflects this duality, prefixing every object file with the CPU name:

```cpp
// qa_hal.cpp:203-219
std::string_view cpu = params.processor_class == HalProcessorClassType::DM
    ? "tt-qsr64" : "tt-qsr32";
std::string_view dir = "runtime/hw/lib/quasar";
objs.push_back(fmt::format("{}/{}-crt0-tls.o", dir, cpu));
```

In contrast, Wormhole and Blackhole link objects without a CPU prefix (e.g., just `tmu-crt0.o`) because all processors in those architectures use the same ISA width.

## Quasar's `firmware_is_kernel_object()` Returns True

A subtle but significant difference: Quasar's JIT query returns `true` for `firmware_is_kernel_object()` (line 291), while both Wormhole and Blackhole return `false`. This changes how firmware ELFs are linked and loaded -- on Quasar, firmware and kernel code share the same object format, reflecting the unified TLS memory model.

## Quasar's Complex Linker Flags

Wormhole and Blackhole return empty linker flags from `linker_flags()`. Quasar returns an extensive set of `--defsym` flags that define memory region bases and sizes for the linker script. For compute processors alone, there are four different base address configurations (one per `trisc0`-`trisc3`), each with firmware and kernel variants. This complexity stems from Quasar's more elaborate memory map with TLS-local memory regions.

## Smaller LLK API Surface in ckernels

The `ckernels/*/metal/llk_api/` directory contains the architecture-specific LLK API shim headers. Quasar's directory is markedly smaller:

**Wormhole/Blackhole** (`ckernels/wormhole_b0/metal/llk_api/` and `ckernels/blackhole/metal/llk_api/`) -- 19+ header files:
- `llk_math_binary_api.h`, `llk_math_binary_sfpu_api.h`, `llk_math_common_api.h`, `llk_math_matmul_api.h`
- `llk_math_reduce_api.h`, `llk_math_reduce_custom_api.h`, `llk_math_transpose_dest_api.h`
- `llk_math_unary_datacopy_api.h`, `llk_math_unary_sfpu_api.h`
- `llk_pack_api.h`, `llk_param_structs.h`, `llk_sfpu_types.h`
- `llk_unpack_AB_api.h`, `llk_unpack_AB_matmul_api.h`, `llk_unpack_AB_reduce_api.h`, `llk_unpack_AB_reduce_custom_api.h`
- `llk_unpack_A_api.h`, `llk_unpack_common_api.h`, `llk_unpack_reduce_api.h`
- `llk_unpack_tilize_api.h`, `llk_unpack_untilize_api.h`
- Plus an `llk_sfpu/` subdirectory

**Quasar** (`ckernels/quasar/metal/llk_api/`) -- 7 header files:
- `llk_math_common_api.h`, `llk_math_matmul_api.h`, `llk_math_unary_datacopy_api.h`
- `llk_pack_api.h`
- `llk_unpack_AB_matmul_api.h`, `llk_unpack_A_api.h`, `llk_unpack_common_api.h`

The missing files (binary ops, reduce, tilize/untilize, SFPU, transpose_dest, etc.) indicate operations not yet ported to the Quasar LLK. The reduced function signatures in the files that do exist also differ -- for example, Quasar's `llk_wait_for_free_tiles()` and `llk_push_tiles()` take fewer template parameters than their Wormhole/Blackhole counterparts.

## Quasar-Specific `chlkc_list.h`

Each architecture has its own `chlkc_list.h` in `ckernels/*/metal/common/`. This file defines the `run_kernel()` function that dispatches to the user's compute kernel. Quasar's version at `ckernels/quasar/metal/common/chlkc_list.h` differs from the Wormhole/Blackhole version in two ways:

1. **Extra kernel type.** Quasar supports `UCK_CHLKC_ISOLATE_SFPU`, a fourth kernel type not present in the tt-1xx versions:
   ```cpp
   #ifdef UCK_CHLKC_ISOLATE_SFPU
   #include "chlkc_descriptors.h"
   #include "chlkc_isolate_sfpu.cpp"
   #endif
   ```

2. **Namespace-qualified calls.** Quasar uses `ckernel::zeroacc()` and `ckernel::zerosrc()` with explicit namespace qualification, while Wormhole/Blackhole call the unqualified `zeroacc()` and `zerosrc()`.

3. **Return type.** Quasar uses `std::uint32_t run_kernel()` while Wormhole/Blackhole use `uint run_kernel()`.

## `ARCH_QUASAR` Guards in the Compute API

There are 21 files under `tt_metal/hw/inc/` that contain `#ifndef ARCH_QUASAR` or `#ifdef ARCH_QUASAR` guards. These guards appear in the public Compute API headers, meaning many compute operations are effectively no-ops on Quasar. The affected files include:

- `compute/tile_move_copy.h` -- `copy_tile_to_dst_init_short()`, `copy_tile_init()`, `copy_tile()`, `copy_block_matmul_partials()` are all guarded out
- `compute/matmul.h` -- matmul operations
- `compute/cb_api.h` -- circular buffer operations
- `compute/common.h` -- common compute initialization
- `compute/compute_kernel_api.h` -- compute kernel API
- `compute/compute_kernel_hw_startup.h` -- `compute_kernel_hw_startup()` is guarded out
- `compute/pack.h` -- pack operations
- `compute/reconfig_data_format.h` -- data format reconfiguration
- `compute/reg_api.h` -- register API
- `compute/eltwise_unary/eltwise_unary.h` -- unary elementwise operations
- `hostdev/dev_msgs.h` -- device message definitions
- `internal/firmware_common.h` -- firmware init differences (e.g., Quasar uses thread-local `rta_l1_base`)
- `internal/circular_buffer_interface.h` -- CB interface differences
- `internal/circular_buffer_init.h` -- CB initialization
- `internal/dataflow/dataflow_api_addrgen.h`, `dataflow_api_common.h` -- dataflow differences
- `internal/debug/stack_usage.h` -- stack usage tracking
- `internal/hw_thread.h` -- hardware thread management
- `internal/tensix_functions.h` -- Tensix-specific function definitions
- `experimental/circular_buffer.h` -- experimental CB API with different template signatures
- `debug/dprint_tile.h` -- tile debug printing

A typical pattern looks like this (from `tile_move_copy.h`):

```cpp
ALWI void copy_tile_init(uint32_t cbid, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    copy_tile_to_dst_init_short(cbid, 0, false, call_line);
#endif  // TODO: AM; add Quasar implementation
}
```

The `TODO` comments confirm these are intentional gaps in the port, not architectural impossibilities. As the Quasar LLK implementation matures, these guards will be replaced with Quasar-specific implementations that call into `tt_llk_quasar/`.

## Different Template Signatures

Where Quasar does implement an LLK function, its API often has simpler template signatures. The `circular_buffer.h` example illustrates this clearly:

```cpp
// Wormhole/Blackhole:
PACK((llk_wait_for_free_tiles<false, false, false>(cb_id_, num_pages)));
PACK((llk_push_tiles<false, false>(cb_id_, num_pages)));

// Quasar:
PACK((llk_wait_for_free_tiles(cb_id_, num_pages)));
PACK((llk_push_tiles(cb_id_, num_pages)));
```

The Wormhole/Blackhole versions carry template parameters for features like untilize mode and accumulation mode. Quasar's LLK functions drop these template parameters, either because the features are not yet supported or because the hardware handles them differently.

## Summary

Quasar's divergence from the tt-1xx generation is structural, not just parametric. It is not simply a matter of different include paths -- the HAL base class is different (`hal_2xx` vs `hal_1xx`), the build system uses a fundamentally different loop structure, the CPU targets are dual-ISA (32-bit + 64-bit), the LLK API surface is a subset of what Wormhole/Blackhole provide, and many Compute API functions are guarded out entirely. As the Quasar port matures, the `#ifndef ARCH_QUASAR` guards will shrink and the `ckernels/quasar/metal/llk_api/` directory will grow to match its tt-1xx counterparts.

---

**Next:** [Chapter 8 -- Extending and Debugging](../ch8_extending_and_debugging/index.md)
