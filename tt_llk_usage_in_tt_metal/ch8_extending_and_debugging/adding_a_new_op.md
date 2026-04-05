# Adding a New Operation

This section walks through the full sequence of changes required to add a new operation to the TT-LLK / TT-Metal stack. The process differs depending on whether the operation is an **SFPU operation** (executed on the SFPU vector engine within the math TRISC) or a **non-SFPU operation** (a matrix/unpack/pack operation that uses fixed-function hardware blocks). We cover the SFPU case in full detail using the existing `relu` family as a reference, then outline the non-SFPU path.

## Adding a New SFPU Operation

Suppose we want to add a hypothetical operation called `my_activation`. The changes span four repositories/directories and six layers.

### Step 1: Implement the SFPI Kernel in TT-LLK

Create a file `ckernel_sfpu_my_activation.h` in the per-architecture SFPU directory:

```
tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_my_activation.h
tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_my_activation.h
```

This file contains the raw SFPI implementation. Follow the pattern of existing files like [`ckernel_sfpu_relu.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h):

```cpp
// tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_my_activation.h
#pragma once

#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_load_config.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void _calculate_my_activation_(const int iterations, std::uint32_t param0) {
    // param0 reinterpreted as a float threshold
    vFloat threshold = Converter::as_float(param0);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        vFloat val = dst_reg[0];
        // ... apply activation logic using SFPI intrinsics ...
        dst_reg[0] = val;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

Key points:
- The function is templated on `APPROXIMATION_MODE` (controlled by `ComputeConfig::math_approx_mode` on the host).
- The `iterations` parameter is always 8 for a full 16x16 face (8 iterations, each processing 2 rows of 16 elements (32 datums per iteration, 256 per face)).
- Use `dst_reg[0]` to load from the destination register, compute in-place, and store back.
- The `Converter::as_float()` utility reinterprets a `uint32_t` as an IEEE 754 float, needed because kernel parameters are passed as unsigned integers.

### Step 2: Add the LLK Math Wrapper in `llk_lib/`

If your new operation is a standard unary SFPU op, you typically do **not** need a new file in `llk_lib/`. The generic dispatcher [`llk_math_eltwise_unary_sfpu_params.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h) handles the face-iteration logic for all unary SFPU ops via a callable template parameter:

```cpp
// tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h (line 13-73)
template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func, std::uint32_t dst_index,
    int vector_mode = static_cast<int>(VectorMode::RC), Args&&... args)
{
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ...>()), "...");
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    // ... iterates over faces based on VectorMode, calling sfpu_func(args...) per face ...
}
```

The init function is already generic in [`llk_math_eltwise_unary_sfpu_init.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h):

```cpp
template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init() {
    _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
}
```

If your operation needs a custom init (e.g. to load lookup tables into SFP registers), add a new init function and use the `SFPU_INIT_KERNEL_CALL` macro pattern. If it needs completely different face-iteration logic (like `topk`), you would add a dedicated `llk_math_my_activation_sfpu.h` in `llk_lib/`.

You must also add a new entry to the `SfpuType` enum in `llk_sfpu_types.h` if your op needs a distinct init path:

```
tt-llk/tt_llk_wormhole_b0/llk_lib/llk_sfpu_types.h
```

### Step 3: Add the LLK API Wrapper in TT-Metal's `ckernels/`

In TT-Metal, create a Metal-specific wrapper that bridges the LLK math layer to the Compute API. This lives under `ckernels/{arch}/metal/llk_api/llk_sfpu/`:

```
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_my_activation.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_my_activation.h
```

Follow the pattern of [`ckernel_sfpu_relu.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h) (the Metal-side version, not the LLK one):

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_my_activation.h
#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_my_activation(const uint param0) {
    _calculate_my_activation_<APPROXIMATION_MODE>(ITERATIONS, param0);
}

}  // namespace sfpu
}  // namespace ckernel
```

This thin wrapper serves two purposes:
1. It provides a fixed `ITERATIONS` default (8), matching the macros' expectations.
2. It bridges the include path -- the Metal `llk_sfpu/` directory is on the JIT include path, while the raw TT-LLK `sfpu/` directory is included transitively through `ckernel_sfpu.h`.

### Step 4: Add the Compute API Header

Create the user-facing API header in `tt_metal/hw/inc/api/compute/eltwise_unary/`:

```
tt_metal/hw/inc/api/compute/eltwise_unary/my_activation.h
```

This is what kernel authors actually `#include` and call. Follow the pattern of [`relu.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/inc/api/compute/eltwise_unary/relu.h):

```cpp
// tt_metal/hw/inc/api/compute/eltwise_unary/my_activation.h
#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_my_activation.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Performs element-wise my_activation on each element of a tile in DST.
 *
 * | Argument   | Description                        | Type     |
 * |------------|------------------------------------|----------|
 * | tile_index | Index of tile in DST register      | uint32_t |
 * | param0     | Activation parameter (as uint)     | uint32_t |
 */
ALWI void my_activation_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_my_activation, RC, APPROX, idst, param0));
}

/**
 * Initializes my_activation. Must be called before my_activation_tile.
 */
ALWI void my_activation_tile_init() {
    MATH(SFPU_UNARY_KERNEL_INIT(my_activation, APPROX));
}

}  // namespace ckernel
```

The macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN` (from [`llk_math_eltwise_unary_sfpu_macros.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h) line 50-52) expands to:

```cpp
_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
    ckernel::sfpu::calculate_my_activation<APPROXIMATE>, dst_index,
    (int)VectorMode::RC, param0)
```

The available macro variants are documented in the macros header (lines 10-96). Choose the one matching your parameter count:

| Macro | Parameters | Use Case |
|-------|-----------|----------|
| `SFPU_UNARY_NO_PARAM_KERNEL_FN` | None | Ops like `abs`, `sign` |
| `SFPU_UNARY_ONE_PARAM_KERNEL_FN` | 1 uint | Ops like `relu_min`, `relu_max` |
| `SFPU_UNARY_TWO_PARAM_KERNEL_FN` | 2 uint | Ops like `clamp` |
| `SFPU_UNARY_THREE_PARAM_KERNEL_FN` | 3 uint | Ops like `addcdiv` |

### Step 5: Write the Device Kernel

Create a compute kernel `.cpp` that uses the Compute API. This file lives under `ttnn/cpp/ttnn/operations/` alongside the operation's host code:

```cpp
// ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/my_activation.cpp
#include "compute_kernel_api/eltwise_unary/my_activation.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/cb_api.h"

void kernel_main() {
    uint32_t num_tiles = get_compile_time_arg_val(0);
    uint32_t param0 = get_compile_time_arg_val(1);

    my_activation_tile_init();
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb::c_in0, 1);
        copy_tile_to_dst(cb::c_in0, 0, 0);
        my_activation_tile(0, param0);
        pack_tile(0, cb::c_out0);
        cb_pop_front(cb::c_in0, 1);
        cb_push_back(cb::c_out0, 1);
    }
}
```

Because this uses the simplified `kernel_main()` syntax, the JIT system will automatically transform it into three TRISC namespaces (`chlkc_unpack`, `chlkc_math`, `chlkc_pack`) via the `simple_kernel_syntax::transform_to_legacy_syntax()` function in [`genfiles.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/genfiles.cpp) (lines 108-133).

### Step 6: Create the Host-Side Factory

The host-side operation factory registers the kernel with the program and passes compile-time arguments. This lives in the TTNN operations tree:

```cpp
// In your operation's program_factory.cpp
auto compute_kernel_id = CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/my_activation.cpp",
    core_range,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = true,
        .compile_args = {num_tiles, param0_as_uint},
    });
```

The [`ComputeConfig`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/api/tt-metalium/kernel_types.hpp) struct (lines 76-98 of `kernel_types.hpp`) controls:
- `math_fidelity` -- sets `MATH_FIDELITY` in `chlkc_descriptors.h`
- `fp32_dest_acc_en` -- sets `DST_ACCUM_MODE` in `chlkc_descriptors.h`
- `math_approx_mode` -- sets `APPROX` in `chlkc_descriptors.h`
- `compile_args` -- become `KERNEL_COMPILE_TIME_ARGS` on the command line
- `defines` -- become `#define` entries in `defines_generated.h`

## Summary: The Full Layer Stack

Here is every file touched, from bottom to top, for an SFPU operation:

| Layer | Repository | Path Pattern | Example |
|-------|-----------|-------------|---------|
| SFPI kernel | tt-llk | `tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_*.h` | `ckernel_sfpu_relu.h` |
| LLK math dispatcher | tt-llk | `tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu*.h` | `llk_math_eltwise_unary_sfpu_params.h` |
| Metal LLK API wrapper | tt-metal | `hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_*.h` | `ckernel_sfpu_relu.h` |
| Compute API header | tt-metal | `hw/inc/api/compute/eltwise_unary/*.h` | `relu.h` |
| Device kernel | tt-metal | `ttnn/cpp/ttnn/operations/.../kernels/*.cpp` | (per-op) |
| Host factory | tt-metal | `ttnn/cpp/ttnn/operations/.../program_factory.cpp` | (per-op) |

## Adding a Non-SFPU Operation

Non-SFPU operations (e.g. matmul, reduce, tilize) follow a similar pattern but go through different LLK layers:

1. **LLK implementation** -- Add files in `tt-llk/tt_llk_{arch}/llk_lib/` (e.g. `llk_unpack_my_op.h`, `llk_math_my_op.h`, `llk_pack_my_op.h`). Non-SFPU operations typically have separate unpack, math, and pack implementations because they drive the fixed-function hardware blocks directly.

2. **LLK API wrapper** -- Add `llk_math_my_op_api.h` (and unpack/pack equivalents) in `tt_metal/hw/ckernels/{arch}/metal/llk_api/`. These call the underscore-prefixed LLK functions and handle Metal-specific concerns (the `DST_SYNC_MODE`, `DST_ACCUM_MODE` globals, `LLK_ASSERT` checks).

3. **Compute API header** -- Add `tt_metal/hw/inc/api/compute/my_op.h`. This provides the user-facing functions like `my_op_init()` and `my_op_tiles()`, guarded by `#ifdef TRISC_MATH` / `TRISC_UNPACK` / `TRISC_PACK` as appropriate.

4. **Device kernel and host factory** -- Same pattern as SFPU operations.

The key difference is that non-SFPU operations need separate implementations for each TRISC processor (unpack, math, pack), while SFPU operations only touch the math TRISC -- the unpack and pack paths use the generic `copy_tile` / `pack_tile` infrastructure.

---

**Next:** [`debugging_llk_issues.md`](./debugging_llk_issues.md)
