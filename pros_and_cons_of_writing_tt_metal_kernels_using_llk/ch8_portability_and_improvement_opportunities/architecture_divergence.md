# Architecture Divergence

The Compute Kernel API (`tt_metal/hw/inc/api/compute/*.h`) is the primary surface through which kernel authors interact with the LLK stack. Its purpose is to provide a stable, architecture-independent interface that hides the differences between Wormhole B0, Blackhole, and Quasar. This section examines how well that promise holds up in practice.

## Quantitative Overview

A search for `ARCH_QUASAR` references across the compute API directory reveals the scale of the problem:

| File | `#ifndef ARCH_QUASAR` guards | `#ifdef ARCH_QUASAR` / `#else` branches | "TODO: AM; add Quasar" stubs |
|------|----:|----:|----:|
| `matmul.h` | **11** | 0 | **9** |
| `pack.h` | **8** | 0 | **7** |
| `reconfig_data_format.h` | **6** | 0 | **6** |
| `tile_move_copy.h` | **5** | 0 | **4** |
| `compute_kernel_hw_startup.h` | **4** | 0 | **2** |
| `compute_kernel_api.h` | **3** | 0 | 0 |
| `cb_api.h` | **4** | 1 | **2** |
| `common.h` | 0 | 2 | 0 |
| `reg_api.h` | **2** | 0 | 0 |
| `eltwise_unary/eltwise_unary.h` | **2** | 0 | **1** |
| **Total** | **45** | 3 | **31** |

**45 `#ifndef ARCH_QUASAR` guards** across 10 files, with **31 of those** being TODO stubs that simply compile to empty function bodies on Quasar. Only a handful of functions (like `cb_reserve_back`, `cb_push_back`, `pack_tile`, `copy_tile`, `unary_op_init_common`) provide genuine Quasar-specific alternative implementations rather than stubs.

## Case Study: `matmul.h` -- 11 Guards in One File

The matmul API header (`tt_metal/hw/inc/api/compute/matmul.h`) is the most heavily guarded file, with **11 `#ifndef ARCH_QUASAR`** blocks. Every single public function in the file is affected:

```
// tt_metal/hw/inc/api/compute/matmul.h

ALWI void mm_init(...)                           // #ifndef ARCH_QUASAR with #else branch
ALWI void matmul_tiles(...)                      // #ifndef ARCH_QUASAR with #else branch
ALWI void matmul_tiles_math(...)                 // #ifndef ARCH_QUASAR -- TODO stub
ALWI void mm_init_short(...)                     // #ifndef ARCH_QUASAR -- TODO stub
ALWI void mm_init_short_with_dt(...)             // #ifndef ARCH_QUASAR -- TODO stub
ALWI void mm_block_init(...)                     // #ifndef ARCH_QUASAR -- TODO stub
ALWI void matmul_block(...)                      // #ifndef ARCH_QUASAR -- TODO stub
ALWI void mm_block_init_short(...)               // #ifndef ARCH_QUASAR -- TODO stub
ALWI void mm_block_init_short_with_dt(...)       // #ifndef ARCH_QUASAR -- TODO stub
ALWI void mm_block_init_short_with_both_dt(...)  // #ifndef ARCH_QUASAR -- TODO stub
ALWI void matmul_block_math_dynamic_throttle(...)// #ifndef ARCH_QUASAR (Blackhole-only)
```

Only `mm_init` and `matmul_tiles` have genuine Quasar implementations. The remaining 9 functions are dead code on Quasar. A kernel author calling `mm_block_init` on Quasar will get a function that silently does nothing -- no compile error, no runtime error, just incorrect behavior.

The divergence is not just about missing implementations. Even the two functions that *do* work on Quasar have different API signatures. Compare `mm_init`:

```cpp
// Wormhole/Blackhole path:
UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(in1_cb_id, in0_cb_id)));
MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));

// Quasar path:
UNPACK((llk_unpack_hw_configure(in1_cb_id, in0_cb_id)));  // no template params
MATH((llk_math_matmul_init<MATH_FIDELITY>()));            // fewer template params, no args
PACK(/* no pack_dest_init call */);                        // missing entirely
```

The underlying LLK functions have different template parameter lists and different function argument lists on Quasar. This means the `#ifdef` branching cannot be eliminated by simply providing matching implementations -- the API contracts themselves diverge at the LLK layer.

## Quasar's 4th TRISC: TRISC3

Quasar introduces a fundamental architectural change: **4 TRISC cores** instead of the traditional 3. The Quasar project parameters confirm this:

```cpp
// tt_llk_quasar/common/inc/ckernel_proj_params.h
#define TRISC_COUNT  0x00000004  // = 4 in decimal
```

The 4th core, TRISC3, is dedicated to SFPU (Special Function Processing Unit) operations. It has its own:

- **Firmware slot:** `MEM_TRISC3_FIRMWARE_BASE` in the memory map
- **Memory layout:** dedicated code, data, stack, and loader regions in linker scripts (`tests/helpers/ld/memory.quasar.ld`)
- **Debug ID:** `TRISC3 = 0x0006'0000` in the debug register enumeration
- **Device enumeration:** `RiscCore.TRISC3 = 4` in the Python test framework

This has deep implications for the TRISC macro system discussed in Chapter 1. The existing `MATH(x)` / `PACK(x)` / `UNPACK(x)` macros map to 3 TRISCs. On Quasar, SFPU work that previously ran on TRISC_MATH can potentially run on TRISC3 instead. The LLK test infrastructure already supports this -- there is a dedicated `sfpu.ld` linker script that maps to `TRISC3_CODE` memory regions.

The Compute Kernel API has **no mechanism** to express this. There is no `SFPU(x)` macro. Kernel authors targeting Quasar's 4th TRISC must work outside the standard API entirely.

## CB API Fragmentation: Runtime Assert-Fails

The circular buffer API (`cb_api.h`) takes a different approach from the matmul API: instead of silently compiling to empty stubs, two functions **assert-fail at runtime** on Quasar:

```cpp
// tt_metal/hw/inc/api/compute/cb_api.h

ALWI uint32_t get_tile_address(uint32_t cb_id, uint32_t tile_index) {
#ifndef ARCH_QUASAR
    // ... full implementation using mailbox synchronization ...
#else
    ASSERT(false && "get_tile_address is not implemented for ARCH_QUASAR");
    return 0;
#endif
}

ALWI uint32_t read_tile_value(uint32_t cb_id, uint32_t tile_index, uint32_t element_offset) {
#ifndef ARCH_QUASAR
    // ... full implementation ...
#else
    ASSERT(false && "read_tile_value is not implemented for ARCH_QUASAR");
    return 0;
#endif
}
```

These functions compile successfully on Quasar but crash at runtime. This is arguably worse than the silent-stub approach used in `matmul.h` because it creates time-bombs in kernel code: the problem only manifests during hardware execution, not during development or compilation.

Meanwhile, the core CB operations (`cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back`) do have proper Quasar implementations, but with different underlying LLK template parameters:

```cpp
// cb_reserve_back: Wormhole/Blackhole
PACK((llk_wait_for_free_tiles<false, false, false>(cbid, ntiles)));

// cb_reserve_back: Quasar
PACK((llk_wait_for_free_tiles(cbid, ntiles)));  // no template params
```

## The `compute_kernel_api.h` Include Guard Structure

The top-level header that most kernels include (`compute_kernel_api.h`) uses `#ifndef ARCH_QUASAR` to exclude entire categories of LLK API headers:

```cpp
// tt_metal/hw/inc/api/compute/compute_kernel_api.h

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_matmul_api.h"
#include "llk_math_unary_datacopy_api.h"
#ifndef ARCH_QUASAR                        // <--- entire categories excluded
#include "llk_math_binary_api.h"
#include "llk_math_unary_sfpu_api.h"
#include "llk_math_binary_sfpu_api.h"
#include "llk_math_reduce_api.h"
#endif
// ...

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_A_api.h"
#ifndef ARCH_QUASAR                        // <--- entire categories excluded
#include "llk_unpack_AB_api.h"
#include "llk_unpack_reduce_api.h"
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_untilize_api.h"
#endif
```

On Quasar, the following LLK API categories are **completely unavailable**:
- Binary math operations (`llk_math_binary_api.h`)
- Unary SFPU operations (`llk_math_unary_sfpu_api.h`)
- Binary SFPU operations (`llk_math_binary_sfpu_api.h`)
- Reduce operations (`llk_math_reduce_api.h`)
- Binary unpack operations (`llk_unpack_AB_api.h`)
- Reduce unpack operations (`llk_unpack_reduce_api.h`)
- Tilize/untilize operations (`llk_unpack_tilize_api.h`, `llk_unpack_untilize_api.h`)

This means that any kernel using binary operations, reductions, tilize, or untilize operations **cannot compile on Quasar at all**. The exclusion happens at the include level, so the compile error is an undefined-symbol error rather than a clear "not supported on this architecture" message.

## Impact on Kernel Authors

The practical consequence is that writing a portable kernel across Wormhole, Blackhole, and Quasar requires one of two strategies:

**Strategy 1: `#ifdef` guards in the kernel source.** The kernel author mirrors the same `#ifndef ARCH_QUASAR` pattern used in the API headers, wrapping architecture-specific code paths in their own kernel:

```cpp
void MAIN {
#ifndef ARCH_QUASAR
    mm_block_init(in0_cb, in1_cb, out_cb, 0, ct_dim, rt_dim, kt_dim);
    // ... block matmul logic ...
#else
    mm_init(in0_cb, in1_cb, out_cb);
    // ... tile-by-tile matmul fallback ...
#endif
}
```

**Strategy 2: Separate source files per architecture.** Maintain entirely separate kernel `.cpp` files (e.g., `matmul_wh.cpp`, `matmul_quasar.cpp`) and select the correct one at the build system level.

Neither strategy is satisfactory. Strategy 1 produces code that is hard to read and test. Strategy 2 creates maintenance burden when the shared logic (which is often the majority of the kernel) needs to change. The deeper issue is that the Compute Kernel API -- which was supposed to prevent exactly this situation -- has become a pass-through for architecture-specific `#ifdef` blocks rather than a genuine abstraction layer.

## Beyond Quasar: Blackhole-Specific Divergence

Quasar is the most extreme case, but Blackhole also introduces its own `#ifdef` paths. The `matmul_block_math_dynamic_throttle` function exists only under `#ifdef ARCH_BLACKHOLE` and implements firmware-controlled dynamic throttle adjustment via a shared L1 memory flag:

```cpp
// tt_metal/hw/inc/api/compute/matmul.h (Blackhole-only)
volatile tt_l1_ptr uint32_t* throttle_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
```

This pattern -- where architecture-specific performance features are wired into the public API -- further erodes the abstraction boundary.

---

**Next:** [`improvement_opportunities.md`](./improvement_opportunities.md)
