# Single Source File, Three Kernels

## The One-File-Three-Compiles Pattern

Every LLK compute kernel is written as a **single C++ source file** that contains three logically separate programs -- one for each TRISC thread. The file is compiled three times by the SFPI cross-compiler (`riscv-tt-elf-g++`), each time with a different preprocessor define active, producing three independent ELF binaries:

| Compilation Pass | Define Set | Output Binary | Tensix Thread |
|:-----------------|:-----------|:--------------|:--------------|
| 1 | `LLK_TRISC_UNPACK` | `unpack.elf` | T0 / TRISC0 |
| 2 | `LLK_TRISC_MATH` | `math.elf` | T1 / TRISC1 |
| 3 | `LLK_TRISC_PACK` | `pack.elf` | T2 / TRISC2 |

The test infrastructure's compilation pipeline passes two independent `-D` flags per compilation: `-DLLK_TRISC_UNPACK` (or `_MATH` / `_PACK`) to select the `#ifdef` block, and `-DCOMPILE_FOR_TRISC=N` (0, 1, or 2) for thread-index-aware code in `ckernel.h`. These are separate defines — `COMPILE_FOR_TRISC` does not set or imply the `LLK_TRISC_*` macros. The same source file is fed to the compiler three times; only the active `#ifdef` block determines which code ends up in the resulting binary.

A fourth binary, `brisc.elf`, is compiled separately from the boot firmware source (`tests/helpers/src/brisc.cpp`) with `-DLLK_BOOT_MODE_BRISC` set. BRISC manages the lifecycle of the three TRISC threads and is not part of the `#ifdef`-guarded structure.

## Structure of `matmul_test.cpp`

The canonical matmul kernel source file (`tests/sources/matmul_test.cpp`) follows a strict structure. Every LLK kernel source must match this pattern.

### Common Preamble (Outside All `#ifdef` Blocks)

The top of the file contains includes and global variable declarations that are shared by all three threads:

```cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
```

These includes and globals are compiled into **all three** ELF files. They must be present in every LLK kernel source.

### Three `#ifdef` Blocks, Each Defining `run_kernel()`

After the preamble, three mutually exclusive `#ifdef` blocks define the per-thread kernel logic:

```cpp
#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // ... unpack logic ...
}
#endif

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // ... math logic ...
}
#endif

#ifdef LLK_TRISC_PACK
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // ... pack logic ...
}
#endif
```

Each block defines the same function signature -- `void run_kernel(RUNTIME_PARAMETERS params)` -- but with entirely different bodies and includes. The `RUNTIME_PARAMETERS` macro (defined in `build.h`, discussed below) expands to `[[maybe_unused]] const struct RuntimeParams&`, so each `run_kernel` receives the same parameter struct by const reference.

Only one `#ifdef` block is active per compilation pass. This means each ELF binary sees exactly one definition of `run_kernel()`, avoiding any multiple-definition errors. The TRISC firmware entry point (`trisc.cpp`) calls this `run_kernel()` function after completing boot-time initialization.

## Mandatory Global Variables

Three global variables must be declared in the common preamble of every LLK kernel source:

```cpp
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
```

These are not arbitrary; they are declared as `extern` in the ckernel infrastructure header `ckernel_globals.h` and are referenced by name from within the LLK header implementations. Omitting any of them will cause a linker error because the LLK headers reference them as external symbols even in compilation passes where the variable is not logically needed.

### `unp_cfg_context`

Tracks the current unpacker configuration context (0 or 1). The Tensix unpacker hardware supports **double-buffered configuration contexts** -- while one context is actively being used to unpack data, the other context can be programmed with the configuration for the next unpack operation. After each unpack operation, `switch_config_context()` in `cunpack_common.h` toggles this variable:

```cpp
inline void switch_config_context(std::uint32_t &unp_cfg_context)
{
    unp_cfg_context = 1 - unp_cfg_context;
    if (unp_cfg_context == 0)
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    }
    else
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
    }
}
```

The variable is also read directly in `_llk_unpack_AB_matmul_()` to decide which configuration register to update for the next tile's base address:

```cpp
if (0 == unp_cfg_context)
{
    TTI_REG2FLOP(1, 0, 0, 0,
        THCON_SEC1_REG3_Base_address_ADDR32 - THCON_CFGREG_BASE_ADDR32,
        p_gpr_unpack::TMP0);
}
else
{
    TTI_REG2FLOP(1, 0, 0, 0,
        THCON_SEC1_REG3_Base_cntx1_address_ADDR32 - THCON_CFGREG_BASE_ADDR32,
        p_gpr_unpack::TMP0);
}
```

This context ping-pong is how the unpacker hides configuration latency behind active data movement. The full double-buffered protocol is documented in [Section 03](./03_inter_thread_synchronization.md).

### `pack_sync_tile_dst_ptr`

Tracks the tile pointer for the packer's destination register synchronization. It is reset to zero in `_llk_pack_dest_init_()`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_(const std::uint32_t face_r_dim = FACE_R_DIM, const bool narrow_tile = false)
{
    tensix_sync();
    reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst>(face_r_dim, narrow_tile);
    packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}
```

### `math_sync_tile_dst_index`

Tracks the current tile index within the math thread's destination section. It is reset to zero in `_llk_math_dest_section_done_()`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    set_math_semaphores();
    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        dest_section_flip();
    }
}
```

All three variables must be initialized to zero and must be defined at global scope so they are visible to the inlined LLK header code in all three compilation contexts.

Note that two additional state variables -- `cfg_state_id` (which tracks the active configuration register bank) and `dest_offset_id` (which tracks the active destination register half) -- are declared in `ckernel.h` and defined by the ckernel infrastructure. They do not appear in the kernel source but are equally critical to the synchronization machinery.

## Mandatory Header Includes

### Common Headers (All Three Threads)

Every kernel source must include these headers outside the `#ifdef` blocks:

| Header | Purpose |
|:-------|:--------|
| `ckernel.h` | Core ckernel infrastructure: semaphore operations, configuration register access, `tensix_sync()`, mailbox operations, memory primitives |
| `llk_defs.h` | Tile constants (`FACE_R_DIM`, `TILE_R_DIM`), data format enums, the `DstSync` enum |
| `params.h` | Brings in `build.h` (auto-generated per variant), defines the `RuntimeParams` struct, and provides the `RUNTIME_PARAMETERS` macro |

The `llk_memory_checks.h` header is also included in `matmul_test.cpp`; it provides `is_valid_L1_address()` for runtime address validation.

### Thread-Specific Headers

Each `#ifdef` block adds headers specific to the operation that thread performs. These headers are placed inside the `#ifdef` blocks because they depend on thread-specific Tensix hardware state and would conflict if compiled together. For example, `llk_math_common.h` pulls in `cmath_common.h`, which uses `namespace ckernel::math`, while `llk_pack_common.h` pulls in `cpack_common.h`, which uses `namespace ckernel::packer`. Including both in the same compilation unit would create ambiguities and redefinitions.

**Unpack thread (`LLK_TRISC_UNPACK`):**

| Header | Purpose |
|:-------|:--------|
| `llk_unpack_AB_matmul.h` | MatMul-specific unpack functions: `_llk_unpack_AB_matmul_init_()`, `_llk_unpack_AB_matmul_()` |
| `llk_unpack_common.h` | Shared unpack configuration: `_llk_unpack_hw_configure_()` |

**Math thread (`LLK_TRISC_MATH`):**

| Header | Purpose |
|:-------|:--------|
| `llk_math_common.h` | Shared math functions: `_llk_math_hw_configure_()`, `_llk_math_wait_for_dest_available_()`, `_llk_math_dest_section_done_()`, `_llk_math_pack_sync_init_()` |
| `llk_math_matmul.h` | MatMul-specific math: `_llk_math_matmul_init_()`, `_llk_math_matmul_()` |

**Pack thread (`LLK_TRISC_PACK`):**

| Header | Purpose |
|:-------|:--------|
| `llk_pack.h` | Pack operation: `_llk_pack_()` |
| `llk_pack_common.h` | Shared pack configuration: `_llk_pack_hw_configure_()`, `_llk_pack_init_()`, `_llk_pack_dest_init_()`, `_llk_packer_wait_for_math_done_()`, `_llk_pack_dest_section_done_()` |

## The `RUNTIME_PARAMETERS` Macro and `struct RuntimeParams`

### The Macro

The `RUNTIME_PARAMETERS` macro is defined inside the auto-generated `build.h` header:

```cpp
#define RUNTIME_PARAMETERS  [[maybe_unused]] const struct RuntimeParams&
```

This expands the function signature `void run_kernel(RUNTIME_PARAMETERS params)` into:

```cpp
void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
```

The `[[maybe_unused]]` attribute silences compiler warnings when some kernel sections do not use all fields of the struct. The const reference avoids copying the struct.

### The `RuntimeParams` Struct

The `RuntimeParams` struct is also defined in `build.h`, generated per test variant by the Python test infrastructure. Its layout depends on which runtime parameters the test Python file specifies. For a matmul test variant, a typical generated struct looks like:

```cpp
struct RuntimeParams {
    std::uint32_t TILE_SIZE_PACK;
    std::uint32_t TILE_SIZE_UNPACK_A;
    std::uint32_t TILE_SIZE_UNPACK_B;
    std::uint32_t CT_DIM;
    std::uint32_t RT_DIM;
    std::uint32_t KT_DIM;
    std::uint32_t TILE_CNT;
    std::uint32_t num_faces_A;
    std::uint32_t num_faces_B;
    std::uint32_t buffer_A[...];
    std::uint32_t buffer_B[...];
    std::uint32_t buffer_Res[...];
    FormatConfig formats;
};
```

Three fields are always present:

1. **`TILE_SIZE_PACK`** -- The tile size in bytes for the packer (output format).
2. **`TILE_SIZE_UNPACK_A`** -- The tile size in bytes for unpacker 0 (operand A).
3. **`TILE_SIZE_UNPACK_B`** -- The tile size in bytes for unpacker 1 (operand B).

The remaining fields are test-specific. The Python `generate_runtime_args_struct()` method in `TestConfig` constructs the struct definition by iterating over the `runtimes` list passed by the test. Each `RuntimeParameter` subclass provides a `convert_to_struct_fields()` method that returns the C++ type and name for its field(s). The `FormatConfig formats` field bundles all data format values needed by the three threads (see Chapter 2 for format details).

The struct is populated on the host side, written to a known L1 address via `ttexalens`, and then copied into local memory by the TRISC firmware at boot time (see the next section on the boot sequence).

When `TestConfig.SPEED_OF_LIGHT` mode is active (the `--speed-of-light` pytest flag), all parameters are treated as compile-time constants and the `RuntimeParams` struct becomes an empty struct, eliminating any runtime overhead from parameter lookups.

## The Auto-Generated `build.h` Header

The `build.h` header is generated per test variant by the Python infrastructure and placed in the variant's build directory. It is included transitively through `params.h`. A typical `build.h` for a matmul variant contains:

```cpp
#pragma once

#include <array>
#include <type_traits>

#include "operand.h"
#include "llk_defs.h"
#include "llk_sfpu_types.h"
#include "perf.h"
#include "tensix_types.h"

#define RUNTIME_PARAMETERS  [[maybe_unused]] const struct RuntimeParams&

constexpr bool l1_acc_en = false;
constexpr bool unpack_to_dest = false;
constexpr bool is_fp32_dest_acc_en = false;
constexpr auto MATH_FIDELITY = ckernel::MathFidelity::HiFi4;

struct RuntimeParams {
    // ... fields as described above ...
};
```

The key elements are the `RUNTIME_PARAMETERS` macro, compile-time `constexpr` values (used as template arguments to LLK API functions), and the `RuntimeParams` struct definition. The `build.h` file is what makes each test variant unique: the same kernel source file can be compiled with different `build.h` files to test different format combinations, fidelity settings, and tile dimensions. For the complete `build.h` generation pipeline, see Chapter 5, Section 02.

## Architectural Conditionals Within the Kernel

The matmul kernel source contains `#ifdef ARCH_BLACKHOLE` conditionals to handle API differences between architectures. For example, in the pack thread:

```cpp
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_<false, false, false>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
    _llk_pack_init_<false, false>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>();
#endif
```

The Blackhole pack API takes an additional `tilize` template parameter (the third `false`) on `_llk_pack_hw_configure_` and `_llk_pack_init_`, while the Wormhole version lacks it. On the other hand, the Blackhole `_llk_pack_dest_init_` takes two template parameters (no third boolean), while Wormhole takes three. These differences are fully handled by the `#ifdef`, so the same source file compiles correctly for both architectures.

## The Compilation Model

The test infrastructure compiles each TRISC binary through a synthetic compilation unit. Rather than passing the kernel `.cpp` file directly to the compiler, it constructs a synthetic source on stdin:

```cpp
#include "matmul_test.cpp"  // The kernel source
#include "trisc.cpp"        // The TRISC firmware entry point
```

This means the kernel source and the TRISC firmware are compiled together as a single translation unit, enabling the compiler to inline LLK function bodies directly into the firmware binary. The compiler receives this synthetic unit along with the thread-specific defines (`-DLLK_TRISC_UNPACK`, `-DARCH_BLACKHOLE`, `-DTENSIX_FIRMWARE`, etc.).

The result is four ELF files -- one for each active RISC-V core on the Tensix tile. These are loaded into their respective L1 memory regions by the test runner and execution begins with BRISC releasing the TRISC cores from reset. For the complete compilation command structure and linker script details, see Chapter 6.

---

**Next:** [`02_boot_sequence_and_firmware.md`](./02_boot_sequence_and_firmware.md)
