# Chapter 1: The TRISC Macro System

## Introduction

Every compute kernel in TT-Metal is compiled **three separate times** -- once for each of the three TRISC processors that share a Tensix core:

| TRISC Thread | Preprocessor Define | Role |
|---|---|---|
| TRISC0 | `TRISC_UNPACK` | Reads tiles from circular buffers into source registers |
| TRISC1 | `TRISC_MATH` | Executes FPU/SFPU math on tiles in the destination register |
| TRISC2 | `TRISC_PACK` | Writes tiles from the destination register back to circular buffers |

The kernel author writes **one source file**. Three macros -- `MATH(x)`, `PACK(x)`, and `UNPACK(x)` -- gate individual statements so that each TRISC binary contains only the code relevant to its pipeline stage. The definitions live in [`compute_kernel_api.h`](../../tt-metal/tt_metal/hw/inc/api/compute/compute_kernel_api.h):

```cpp
// tt_metal/hw/inc/api/compute/compute_kernel_api.h, lines 19-59

#ifdef TRISC_MATH
#include "llk_math_common_api.h"
// ... math LLK headers ...
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
// ... unpack LLK headers ...
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
```

When `TRISC_MATH` is defined, `MATH(x)` expands to `x` and `PACK(x)` / `UNPACK(x)` expand to nothing (and vice versa for the other two). The higher-level Compute API functions (like `add_tiles`, `matmul_block`, `pack_tile`) use these macros internally, so most kernel authors never need to invoke the macros directly -- but complex kernels still reach for them regularly.

A companion attribute, `ALWI` (defined as `inline __attribute__((always_inline))` at line 17), ensures that every Compute API helper is fully inlined into the caller, eliminating function-call overhead on the RISC-V cores.

## How the Compilation Model Works

The build system compiles the same `.cpp` file three times with different `-D` flags:

1. **TRISC0 build:** `-DTRISC_UNPACK` -- only `UNPACK(...)` statements survive.
2. **TRISC1 build:** `-DTRISC_MATH` -- only `MATH(...)` statements survive.
3. **TRISC2 build:** `-DTRISC_PACK` -- only `PACK(...)` statements survive.

Shared code (loop control, `constexpr` constants, `cb_wait_front` / `cb_pop_front` dispatched by `#ifdef` inside `cb_api.h`) compiles into all three binaries and keeps the three processors synchronized through circular-buffer semaphores.

For a comprehensive description of how this compilation model integrates with the LLK library and the JIT pipeline, see the [companion guide's Chapter 5: Compute Kernel API](../../tt_llk_usage_in_tt_metal/ch5_compute_kernel_api/index.md) and [Chapter 2: Build System Integration](../../tt_llk_usage_in_tt_metal/ch2_build_system_integration/index.md).

## What This Chapter Evaluates

This chapter analyzes the **developer experience** of writing kernels under this macro system:

- [**`advantages.md`**](./advantages.md) -- What the single-source, macro-gated model gets right: authoring simplicity, compile-time elimination, implicit pipeline parallelism, and zero-overhead inlining.

- [**`pitfalls.md`**](./pitfalls.md) -- Where the model creates friction: invisible code elimination, debugging confusion, macro hygiene issues, and the absence of cross-TRISC static analysis.

The evaluation uses concrete examples from production kernels including [`bmm_large_block_zm_fused_bias_activation.cpp`](../../tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp), [`eltwise_binary.cpp`](../../tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary.cpp), [`compute_common.hpp`](../../tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp) (SDPA), and [`layernorm.cpp`](../../tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp).
