# Step-by-Step Walkthrough: Adding a New SFPU Operation

This walkthrough traces the end-to-end path for adding a new SFPU operation. We use the existing `abs` and `activations` (hardsigmoid, softsign, celu, softshrink) operations as concrete reference points, showing the real files you would need to create or modify.

---

## Step 1: TT-LLK SFPU Implementation

**Repository:** TT-LLK
**Path:** `tt_llk_<arch>/common/inc/sfpu/ckernel_sfpu_<op>.h`

This is where the actual math lives. You write a templated function using SFPI vector intrinsics that operates on the destination register file, one tile row at a time.

For `abs` on Blackhole, the implementation is minimal:

```cpp
// tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_abs.h

#pragma once
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_abs_(const int iterations) {
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::abs(v);
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
```

The naming convention uses a leading and trailing underscore (`_calculate_<op>_`) in the LLK repo. The function template takes `APPROXIMATION_MODE` and `ITERATIONS` as compile-time parameters and a runtime `iterations` count.

**Critical detail:** This file must be duplicated per architecture. The same `ckernel_sfpu_abs.h` exists in both `tt_llk_blackhole/common/inc/sfpu/` and `tt_llk_wormhole_b0/common/inc/sfpu/`. Currently both architecture directories contain 52 SFPU headers each. The Quasar architecture has a much smaller set (14 files) because it is newer and only has the most essential operations ported.

For operations where the math differs between architectures (due to different instruction support, precision trade-offs, or hardware capabilities), you may need genuinely different implementations. But for many operations like `abs`, the code is identical across architectures.

---

## Step 2: Metal SFPU Wrapper

**Repository:** TT-Metal
**Path:** `tt_metal/hw/ckernels/<arch>/metal/llk_api/llk_sfpu/`

This layer wraps the raw LLK SFPU function in a Metal-compatible calling convention. Two things happen here:

1. The `_calculate_<op>_` template gets a wrapper named `calculate_<op>` (dropping the underscores).
2. An `llk_math_eltwise_unary_sfpu_<op>` function is defined that calls into the shared dispatch machinery.

For `abs` on Blackhole:

```cpp
// tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
```

And the LLK math wrapper:

```cpp
// tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h

#pragma once
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_abs.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_abs<APPROXIMATE>, dst_index, vector_mode);
}

} // namespace ckernel
```

The dispatch function `_llk_math_eltwise_unary_sfpu_params_` handles the boilerplate of setting up the SFPU execution context, selecting the right tile dimensions, and calling your function pointer.

**This directory is large.** The Blackhole `llk_sfpu/` directory contains 159 header files. The Wormhole directory contains 158 (the difference is `llk_math_eltwise_binary_sfpu_mul_int32.h`, which exists only on Blackhole). Both must be kept in sync when adding a new operation.

---

## Step 3: Compute API Header

**Repository:** TT-Metal
**Path:** `tt_metal/hw/inc/api/compute/eltwise_unary/<op>.h`

This is the user-facing API that compute kernels call. It wraps the LLK math function in a `<op>_tile()` / `<op>_tile_init()` pair, conditioned on `TRISC_MATH` to ensure it only compiles on the math RISC-V core.

For the `activations` group (hardsigmoid, softsign, celu, softshrink):

```cpp
// tt_metal/hw/inc/api/compute/eltwise_unary/activations.h

#pragma once
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_activations.h"
#endif

namespace ckernel {

ALWI void hardsigmoid_tile(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid<APPROX, ckernel::ActivationType::Hardsigmoid>(idst)));
}

ALWI void hardsigmoid_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_hardsigmoid_init<APPROX>()));
}

// ... softsign_tile, celu_tile, softshrink_tile follow the same pattern

} // namespace ckernel
```

The `ALWI` macro expands to `__attribute__((always_inline))`. The `MATH((...))` macro gates compilation to the math RISC core only. The `APPROX` template parameter is a global compile-time constant set by the program factory.

There are currently **54 header files** in the `eltwise_unary/` directory (53 operation headers plus the `sfpu_split_includes.h` dispatcher).

---

## Step 4: `sfpu_split_includes` Wiring

**Repository:** TT-Metal
**Path:** `tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h`

This file is the conditional include dispatcher. It exists because including all 53+ SFPU headers at once would blow the instruction memory budget on the RISC-V cores. Instead, each operation gets a preprocessor guard:

```cpp
// sfpu_split_includes.h (excerpt)

#if SFPU_OP_ACTIVATIONS_INCLUDE
#include "api/compute/eltwise_unary/activations.h"
#endif

#if SFPU_OP_THRESHOLD_INCLUDE
#include "api/compute/eltwise_unary/threshold.h"
#endif

// ... 44 more conditional blocks
```

To register a new operation, you must:

1. Choose or create a guard macro name (e.g., `SFPU_OP_MYOP_INCLUDE`).
2. Add a `#if` / `#include` / `#endif` block to this file.
3. Wire the macro name into the host-side `get_macro_definition()` function (Step 6).

The file currently contains **46 conditional include blocks** (some blocks serve multiple related ops). Every new SFPU operation must add an entry here. This makes the file a serialization point for all SFPU-related pull requests.

---

## Step 5: Compute Kernel

**Repository:** TT-Metal
**Path:** `tt_metal/kernels/compute/eltwise_sfpu.cpp`

The standard unary SFPU compute kernel is a generic template that does not need modification for new operations. Instead, it uses the `SFPU_OP_CHAIN_0` macro for code injection:

```cpp
// tt_metal/kernels/compute/eltwise_sfpu.cpp

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            cb_wait_front(tt::CBIndex::c_0, 1);
            copy_tile(tt::CBIndex::c_0, 0, 0);

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, tt::CBIndex::c_16);
            cb_pop_front(tt::CBIndex::c_0, 1);
            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_16, per_core_block_dim);
    }
}
```

The `SFPU_OP_CHAIN_0` macro is not defined in any header. It is injected at compile time via `-D` flags by the program factory. When expanded, it becomes a sequence of init and function calls like:

```
hardsigmoid_tile_init(); hardsigmoid_tile(0);
```

This same `SFPU_OP_CHAIN_0` mechanism is used in multiple kernel files across the codebase -- `eltwise_sfpu.cpp`, `eltwise_binary.cpp`, and several TTNN-specific kernel files (at least 7 files reference this macro).

For most new operations, you do not need to touch this file. However, if your operation requires a different kernel structure (custom looping, multiple inputs, etc.), you would write a new compute kernel.

---

## Step 6: Program Factory and Host Defines

**Repository:** TT-Metal
**Path:** `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`

This is the host-side C++ code that wires everything together. Two functions must be updated:

### 6a. `get_macro_definition()` -- Maps `UnaryOpType` to the `sfpu_split_includes.h` guard macro:

```cpp
std::string get_macro_definition(UnaryOpType op_type) {
    switch (op_type) {
        // ...
        case UnaryOpType::SOFTSHRINK:
        case UnaryOpType::SOFTSIGN:
        case UnaryOpType::HARDSIGMOID:
        case UnaryOpType::CELU: return "SFPU_OP_ACTIVATIONS_INCLUDE";
        // ...
    };
}
```

### 6b. `get_op_init_and_func()` -- Returns the init/function string pair that becomes the `SFPU_OP_CHAIN_0` content:

```cpp
case UnaryOpType::HARDSIGMOID:
    return {"hardsigmoid_tile_init();", fmt::format("hardsigmoid_tile({});", idst)};
```

A third function, `get_block_defines()`, assembles the full `SFPU_OP_CHAIN_0` define from individual init+func pairs, but it is a generic dispatcher that iterates over the op chain -- it does not contain per-op logic and does not need modification when adding a new SFPU op.

The program factory in `unary_program_factory.cpp` then passes these defines to the compute kernel compilation:

```cpp
std::map<std::string, std::string> unary_defines =
    utils::get_block_defines(args.op_chain, "0", "0", input.dtype());
// ...
auto eltwise_unary_kernel_id = CreateKernel(program, ..., {
    .defines = unary_defines
});
```

This is where the `SFPU_OP_CHAIN_0` macro value and the `SFPU_OP_<NAME>_INCLUDE=1` flags are injected as compile-time definitions for the device kernel.

---

## Summary: Files Touched for a New SFPU Op

For a simple new unary SFPU operation across two architectures (Wormhole B0 and Blackhole), the minimum set of files to create or modify is:

| # | File | Action |
|---|------|--------|
| 1 | `tt-llk: tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_<op>.h` | Create |
| 2 | `tt-llk: tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_<op>.h` | Create |
| 3 | `tt-metal: hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_<op>.h` | Create |
| 4 | `tt-metal: hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_<op>.h` | Create |
| 5 | `tt-metal: hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_<op>.h` | Create |
| 6 | `tt-metal: hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_<op>.h` | Create |
| 7 | `tt-metal: hw/inc/api/compute/eltwise_unary/<op>.h` | Create |
| 8 | `tt-metal: hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | Modify |
| 9 | `tt-metal: ttnn/.../unary/common/unary_op_utils.cpp` | Modify (2 functions) |
| 10 | `tt-metal: hw/CMakeLists.txt` | Modify (add header to install list) |

That is **7 new files and 3 modified files** across two repositories, minimum. Adding Quasar support would add 2-4 more files.

---

**Next:** [`friction_points.md`](./friction_points.md)
