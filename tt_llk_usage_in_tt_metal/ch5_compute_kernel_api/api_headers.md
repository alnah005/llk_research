# API Headers

This section describes the organization of the Compute Kernel API headers, the dispatch macros that route code to the correct TRISC thread, and the inlining discipline that keeps these wrappers zero-overhead.

## Directory Layout

All user-facing compute API headers live under:

```
tt_metal/hw/inc/api/compute/
```

The directory contains approximately 40+ headers. A representative listing:

```
compute_kernel_api.h          -- master include (LLK includes + macros)
compute_kernel_hw_startup.h   -- one-time HW init
common.h                      -- runtime args, coordinate helpers
common_globals.h              -- ALWI macro, MATH/UNPACK/PACK macros
cb_api.h                      -- circular buffer operations
reg_api.h                     -- DST register acquire/release
pack.h                        -- pack_tile, pack_reconfig_*
matmul.h                      -- mm_init, matmul_tiles, matmul_block
eltwise_binary.h              -- add_tiles, sub_tiles, mul_tiles
reduce.h                      -- reduce_init, reduce_tile
tile_move_copy.h              -- copy_tile, copy_tile_to_dst_init_short
bcast.h                       -- broadcast binary ops
softmax.h                     -- softmax-specific helpers
reconfig_data_format.h        -- data format reconfiguration
sentinel/
  compute_kernel_sentinel.h   -- state_configure() implementation
```

## The Master Include: `compute_kernel_api.h`

The file `tt_metal/hw/inc/api/compute/compute_kernel_api.h` is the central entry point. It pulls in all LLK API headers, conditioned on which TRISC thread is being compiled:

```cpp
// tt_metal/hw/inc/api/compute/compute_kernel_api.h, lines 17-59

#define ALWI inline __attribute__((always_inline))

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_matmul_api.h"
#include "llk_math_unary_datacopy_api.h"
#ifndef ARCH_QUASAR
#include "llk_math_binary_api.h"
#include "llk_math_unary_sfpu_api.h"
#include "llk_math_binary_sfpu_api.h"
#include "llk_math_reduce_api.h"
#endif
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#include "llk_io_pack.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_A_api.h"
#ifndef ARCH_QUASAR
#include "llk_unpack_AB_api.h"
#include "llk_unpack_reduce_api.h"
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_untilize_api.h"
#endif
#include "llk_io_unpack.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
```

### How the Dispatch Macros Work

The `MATH(x)`, `UNPACK(x)`, and `PACK(x)` macros are the key mechanism that makes a single source file compile into three different TRISC binaries:

- When compiled with `-DTRISC_MATH`, `MATH(x)` expands to `x` and `UNPACK(x)` / `PACK(x)` expand to nothing.
- When compiled with `-DTRISC_UNPACK`, only `UNPACK(x)` expands.
- When compiled with `-DTRISC_PACK`, only `PACK(x)` expands.

This means a wrapper function like `matmul_tiles()` contains calls to both `UNPACK(...)` and `MATH(...)`, but after preprocessing each TRISC binary contains only its own relevant code.

### The `ALWI` Macro

```cpp
#define ALWI inline __attribute__((always_inline))
```

Nearly all Compute API wrapper functions are declared `ALWI`. (A notable exception is `init_bcast` in `bcast.h`, which lacks the `ALWI` annotation.) This ensures the compiler always inlines the wrapper at the call site -- there is zero function-call overhead. On RISC-V firmware where code size and branch-prediction resources are limited, this is critical: the wrapper layer adds no runtime cost, and the generated code is as if the kernel author had written the raw LLK calls directly.

The `ALWI` macro is defined in both `compute_kernel_api.h` (line 17) and `common_globals.h` (line 7), ensuring it is available regardless of which include path a per-operation header uses.

## Per-Operation Headers

Each per-operation header includes only the LLK API headers it needs, gated by TRISC defines. For example:

### `matmul.h`

```cpp
// tt_metal/hw/inc/api/compute/matmul.h, lines 7-15
#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_common_api.h"
#endif
```

### `eltwise_binary.h`

```cpp
// tt_metal/hw/inc/api/compute/eltwise_binary.h, lines 8-15
#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#include "llk_unpack_A_api.h"
#endif
```

### `reduce.h`

```cpp
// tt_metal/hw/inc/api/compute/reduce.h, lines 7-15
#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_reduce_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_reduce_api.h"
#endif
```

### `tile_move_copy.h`

```cpp
// tt_metal/hw/inc/api/compute/tile_move_copy.h, lines 7-16
#include "api/compute/common_globals.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif
```

### `pack.h`

```cpp
// tt_metal/hw/inc/api/compute/pack.h, lines 7-8
#include "common_globals.h"
```

Pack operations only need the `PACK()` macro and the pack LLK APIs that are already included via `common_globals.h` (which includes `llk_pack_api.h` when `TRISC_PACK` is defined).

### `cb_api.h`

```cpp
// tt_metal/hw/inc/api/compute/cb_api.h, lines 11-16
#ifdef TRISC_PACK
#include "llk_io_pack.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_io_unpack.h"
#endif
```

Circular buffer operations split across UNPACK (for consumer-side `cb_wait_front` / `cb_pop_front`) and PACK (for producer-side `cb_reserve_back` / `cb_push_back`).

## How Wrapper Functions Are Structured

Every wrapper follows the same pattern:

1. An `ALWI` function with user-friendly parameters (CB IDs, tile indices).
2. Inside the body, `UNPACK(...)`, `MATH(...)`, and/or `PACK(...)` calls that invoke LLK functions.
3. Template parameters for compile-time specialization (operation type, broadcast dimension, math fidelity).

Example -- `add_tiles()` from `eltwise_binary.h` (line 170):

```cpp
ALWI void add_tiles(uint32_t icb0, uint32_t icb1,
                     uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE,
          MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
          icb0, icb1, idst, true)));
}
```

When compiled for TRISC0 (unpack), only the `llk_unpack_AB` call remains. When compiled for TRISC1 (math), only the `llk_math_eltwise_binary` call remains. When compiled for TRISC2 (pack), the function body is empty -- packing is handled separately via `pack_tile()`.

## Summary of Include Dependencies

```
compute_kernel_api.h
  |-- chlkc_list.h, ckernel.h, ckernel_include.h
  |-- TRISC_MATH:   llk_math_common_api.h, llk_math_matmul_api.h,
  |                  llk_math_unary_datacopy_api.h, llk_math_binary_api.h,
  |                  llk_math_unary_sfpu_api.h, llk_math_binary_sfpu_api.h,
  |                  llk_math_reduce_api.h
  |-- TRISC_PACK:   llk_pack_api.h, llk_io_pack.h
  |-- TRISC_UNPACK: llk_unpack_common_api.h, llk_unpack_AB_matmul_api.h,
  |                  llk_unpack_A_api.h, llk_unpack_AB_api.h,
  |                  llk_unpack_reduce_api.h, llk_unpack_tilize_api.h,
  |                  llk_unpack_untilize_api.h, llk_io_unpack.h
  |
  +-- Per-op headers (matmul.h, eltwise_binary.h, etc.)
       each includes a subset of the above, plus common.h / sentinel
```

---

**Next:** [`operation_to_llk_mapping.md`](./operation_to_llk_mapping.md)
