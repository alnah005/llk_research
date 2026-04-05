# Submodule Layout and CMake Integration

## TT-LLK as a Git Submodule

TT-LLK is vendored into TT-Metal as a git submodule at `tt_metal/third_party/tt_llk/`. The `.gitmodules` entry is:

```ini
# .gitmodules (tt-metal repo root)
[submodule "tt_metal/third_party/tt_llk"]
    path = tt_metal/third_party/tt_llk
    url = https://github.com/tenstorrent/tt-llk.git
    branch = main
```

After checkout, the submodule directory contains the cross-architecture `common/` tree and per-architecture directories:

```
tt_metal/third_party/tt_llk/
    common/                        # Architecture-independent headers
    tt_llk_wormhole_b0/
        common/inc/                # Wormhole-specific common includes
        llk_lib/                   # Wormhole LLK implementation
    tt_llk_blackhole/
        common/inc/
        llk_lib/
    tt_llk_quasar/
        common/inc/
        llk_lib/
```

## The `jitapi` Interface Target

The `jitapi` CMake target is an `INTERFACE` library that aggregates all headers the JIT compiler needs to see. LLK headers are gathered into this target via a recursive glob.

```cmake
# tt_metal/CMakeLists.txt, lines 159-176
add_library(jitapi INTERFACE)

file(GLOB_RECURSE COMPUTE_KERNEL_API include/*)
file(GLOB_RECURSE FABRIC_HW_INC fabric/hw/inc/*)
file(
    GLOB_RECURSE TT_LLK_HEADERS
    third_party/tt_llk/common/*.h
    third_party/tt_llk/tt_llk_wormhole_b0/**/*.h
    third_party/tt_llk/tt_llk_blackhole/**/*.h
    third_party/tt_llk/tt_llk_quasar/**/*.h
)
```

These headers -- along with compute kernel API headers, fabric headers, and various other device headers -- are then attached to the `jitapi` target via `target_sources(... FILE_SET jit_api ...)` starting at line 178. This ensures CMake tracks them as part of the interface, though they are not compiled directly (header verification is disabled because they target the device, not the host):

```cmake
# tt_metal/CMakeLists.txt, lines 161-166
set_target_properties(
    jitapi
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
```

## Firmware Object Compilation

When building firmware objects (the static firmware images flashed onto Tensix cores), the `tt_metal/hw/CMakeLists.txt` file constructs `-I` include paths directly. The LLK-specific paths appear in the architecture-specific include list:

```cmake
# tt_metal/hw/CMakeLists.txt, lines 367-379
# Architecture specific include paths
list(
    APPEND
    GPP_INCLUDES
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}/${ARCH_ALIAS}_defines
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}/noc
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${ARCH}/overlay
    -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/umd/device/${ARCH}
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_ALIAS}/metal/common
    -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH_ALIAS}/metal/llk_io
    -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_ALIAS}/common/inc
    -I${PROJECT_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_ALIAS}/llk_lib
)
```

The `${ARCH_ALIAS}` variable resolves to the architecture-specific name (e.g., `wormhole_b0`, `blackhole`, `quasar`). This means the firmware build gets:

| Variable value | LLK include path (common/inc) | LLK include path (llk_lib) |
|---|---|---|
| `wormhole_b0` | `tt_llk_wormhole_b0/common/inc` | `tt_llk_wormhole_b0/llk_lib` |
| `blackhole` | `tt_llk_blackhole/common/inc` | `tt_llk_blackhole/llk_lib` |
| `quasar` | `tt_llk_quasar/common/inc` | `tt_llk_quasar/llk_lib` |

These includes precede the common ones (listed earlier in the same CMake function, lines 355-365) that cover the non-architecture-specific hardware headers. The common includes provide paths like `tt_metal/hw/inc`, `tt_metal/api`, and `tt_metal/hw/firmware/src/tt-1xx`.

Note that the firmware build uses `custom_command` invocations (line 387 onward) rather than standard CMake targets, because it cross-compiles for the Tenstorrent RISC-V ISA using a custom toolchain (`-mcpu=tt-wh` or equivalent). The `-I` paths are passed as raw strings to the compiler invocation rather than through `target_include_directories`.

---

**Next:** [`jit_compilation_pipeline.md`](./jit_compilation_pipeline.md)
