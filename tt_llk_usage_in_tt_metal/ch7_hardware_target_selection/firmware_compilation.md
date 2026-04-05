# Firmware Compilation

While the HAL drives JIT-time include path selection (covered in [HAL Architecture Dispatch](./hal_architecture_dispatch.md)), the CMake build system in `tt_metal/hw/CMakeLists.txt` handles ahead-of-time compilation of firmware objects. This file contains two distinct architecture iteration loops -- an "old-style" loop for Wormhole and Blackhole, and a "TLS-style" loop for Quasar -- each of which wires up LLK include paths differently.

## Architecture Lists

The file begins by defining the architectures and their associated processor types:

```cmake
# tt_metal/hw/CMakeLists.txt:1-28
set(ARCHS
    wormhole
    blackhole
)

set(TLS_ARCH_CPUS quasar:tt-qsr32:tt-qsr64)
# name:cpu
set(TLS_PROCS
    trisc:tt-qsr32
    dm:tt-qsr64
)
```

`ARCHS` is the list for Wormhole and Blackhole (the tt-1xx generation). `TLS_ARCH_CPUS` encodes Quasar's architecture name along with its two CPU targets in a colon-delimited string. `TLS_PROCS` maps processor names to their respective CPU ISA targets.

## Old-Style Loop (Wormhole and Blackhole)

Starting at line 324, the CMake file iterates over `ARCHS` to compile firmware objects (substitutes, TDMA, NOC, CRT0, etc.):

```cmake
# tt_metal/hw/CMakeLists.txt:324
foreach(ARCH IN LISTS ARCHS)
    set(HW_OBJ_SRC_PAIRS
        "substitutes.o:tt_metal/hw/toolchain/substitutes.cpp"
        "tdma_xmov.o:tt_metal/hw/firmware/src/tt-1xx/tdma_xmov.c"
        "noc.o:tt_metal/hw/firmware/src/tt-1xx/${ARCH}/noc.c"
        "tmu-crt0.o:tt_metal/hw/toolchain/tmu-crt0.S"
    )
    set(ARCH_ALIAS ${ARCH})

    if(ARCH STREQUAL "wormhole")
        set(ARCH_ALIAS wormhole_b0)
        list(APPEND HW_OBJ_SRC_PAIRS
            "wh-iram-start.o:tt_metal/hw/toolchain/wh-iram-start.S"
            "wh-iram-trampoline.o:tt_metal/hw/toolchain/wh-iram-trampoline.S"
        )
    endif()
```

The `ARCH_ALIAS` variable is important: the CMake `ARCH` variable is `wormhole` (the directory name under `hw/inc/internal/tt-1xx/`), but the LLK and ckernels directories use `wormhole_b0`. The alias remapping at line 335 bridges this gap.

### LLK Include Paths in the Old-Style Loop

Lines 368-379 append architecture-specific include paths that point directly into the LLK and ckernels directories using `ARCH_ALIAS`:

```cmake
# tt_metal/hw/CMakeLists.txt:368-379
list(APPEND GPP_INCLUDES
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

For Wormhole this expands to:
- `-I.../tt_llk_wormhole_b0/common/inc`
- `-I.../tt_llk_wormhole_b0/llk_lib`
- `-I.../ckernels/wormhole_b0/metal/common`
- `-I.../ckernels/wormhole_b0/metal/llk_io`

For Blackhole:
- `-I.../tt_llk_blackhole/common/inc`
- `-I.../tt_llk_blackhole/llk_lib`
- `-I.../ckernels/blackhole/metal/common`
- `-I.../ckernels/blackhole/metal/llk_io`

### Per-Architecture Compiler Flags

Each architecture gets its own `-mcpu` flag set earlier in the file:

```cmake
# tt_metal/hw/CMakeLists.txt:299-300
set(GPP_FLAGS_wormhole -mcpu=tt-wh)
set(GPP_FLAGS_blackhole -mcpu=tt-bh)
```

These are selected via `${GPP_FLAGS_${ARCH}}` at line 346, meaning the firmware objects for each architecture are compiled with the correct instruction set.

### Object File Output

Each architecture's firmware objects are placed in `runtime/hw/lib/${ARCH}/` (line 351). These objects are later linked by the JIT build system via the `link_objs()` method on the HAL query class. For example, `HalJitBuildQueryWormhole::link_objs()` references `runtime/hw/lib/wormhole/tmu-crt0.o`, `runtime/hw/lib/wormhole/noc.o`, etc.

## TLS-Style Loop (Quasar)

Starting at line 406, a separate loop handles Quasar using the `TLS_ARCH_CPUS` list:

```cmake
# tt_metal/hw/CMakeLists.txt:406-407
foreach(ARCH_CPU IN LISTS TLS_ARCH_CPUS)
    string(REPLACE ":" ";" CPUS ${ARCH_CPU})
    list(GET CPUS 0 ARCH)
    list(POP_FRONT CPUS)
```

This parses `quasar:tt-qsr32:tt-qsr64` into `ARCH=quasar` and `CPUS=(tt-qsr32 tt-qsr64)`.

### Key Differences from Old-Style

1. **Dual CPU targets.** Each object file is compiled once per CPU target, producing output like `tt-qsr32-substitutes.o` and `tt-qsr64-substitutes.o` (line 454: `set(HW_OBJ_FILE "${CPU}-${HWOBJ}")`).

2. **tt-2xx internal paths.** The include paths reference `tt-2xx` instead of `tt-1xx`:
   ```cmake
   # tt_metal/hw/CMakeLists.txt:413-428
   set(GPP_INCLUDES
       ...
       -I${PROJECT_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-2xx
       -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-2xx/${ARCH}
       -I${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-2xx/${ARCH}/noc
       -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH}/metal/common
       -I${PROJECT_SOURCE_DIR}/tt_metal/hw/ckernels/${ARCH}/metal/llk_io
   )
   ```
   Note that the TLS loop does **not** include `tt_llk_*/common/inc` or `tt_llk_*/llk_lib` in the firmware build -- those paths are only added at JIT time by `HalJitBuildQueryQuasar::includes()`.

3. **No ARCH_ALIAS remapping.** Unlike the Wormhole case where `ARCH=wormhole` maps to `ARCH_ALIAS=wormhole_b0`, Quasar uses `quasar` everywhere consistently.

4. **CPU-conditional defines.** The 32-bit Tensix target gets an extra define:
   ```cmake
   if(CPU STREQUAL "tt-qsr32")
       list(APPEND GPP_FLAGS -DCOMPILE_FOR_TRISC)
   endif()
   ```

5. **Per-object CPU filtering.** Some objects are only needed for specific CPUs. The NOC object, for example, is restricted to `tt-qsr64` only:
   ```cmake
   set(HW_OBJ_PATTERNS
       "substitutes.o:tt_metal/hw/toolchain/substitutes.cpp"
       "crt0-tls.o:tt_metal/hw/toolchain/crt0-tls.S"
       "crt0.o:tt_metal/hw/toolchain/crt0.S"
       "noc.o:tt_metal/hw/firmware/src/tt-2xx/quasar/noc.c:tt-qsr64"
   )
   ```
   The third colon-delimited field limits which CPUs build that object.

### How the JIT Links These Objects

`HalJitBuildQueryQuasar::link_objs()` (in `qa_hal.cpp` lines 203-219) selects objects using the CPU name prefix:

```cpp
std::string_view cpu = params.processor_class == HalProcessorClassType::DM
    ? "tt-qsr64" : "tt-qsr32";
std::string_view dir = "runtime/hw/lib/quasar";
objs.push_back(fmt::format("{}/{}-crt0-tls.o", dir, cpu));
if (params.is_fw) {
    objs.push_back(fmt::format("{}/{}-crt0.o", dir, cpu));
}
```

This matches the CMake naming convention of `${CPU}-${HWOBJ}`, ensuring the correct ISA variant is linked for each processor class.

## Linker Script Generation

Both loop styles also generate architecture-specific linker scripts. The old-style loop (lines 170-240) preprocesses `main.ld` with architecture defines. The TLS-style loop (lines 242-293) preprocesses `script_tng.ld` with `ARCH_QUASAR` and CPU-specific defines.

The linker scripts themselves are selected at JIT time by the HAL's `linker_script()` method, which returns paths like `runtime/hw/toolchain/wormhole/firmware_brisc.ld` or `runtime/hw/toolchain/quasar/firmware_dm.ld`.

---

**Next:** [`quasar_divergence.md`](./quasar_divergence.md)
