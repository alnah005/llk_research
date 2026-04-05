# Codebase Layout

This document maps the directory structures of both TT-LLK and TT-Metal, showing how the two codebases connect at the submodule boundary and how layers build on top of each other.

## 1. TT-LLK Repository Structure

The TT-LLK repository (`tt_metal/third_party/tt_llk/`) is organized primarily by hardware architecture. Each architecture directory contains the same three subdirectories.

```
tt_llk/
├── common/                       # Shared headers across all architectures
│   ├── llk_assert.h
│   └── tensor_shape.h
├── tt_llk_blackhole/             # Blackhole-specific LLK implementation
│   ├── common/inc/               # Architecture-specific common includes
│   ├── instructions/             # Assembly instruction descriptors (assembly.yaml)
│   └── llk_lib/                  # Core LLK library headers
│       ├── llk_math_common.h
│       ├── llk_math_matmul.h
│       ├── llk_math_eltwise_binary.h
│       ├── llk_math_eltwise_unary_sfpu.h
│       ├── llk_math_eltwise_unary_datacopy.h
│       ├── llk_math_reduce.h
│       ├── llk_pack.h
│       ├── llk_pack_common.h
│       ├── llk_pack_untilize.h
│       ├── llk_unpack_A.h
│       ├── llk_unpack_AB.h
│       ├── llk_unpack_AB_matmul.h
│       ├── llk_unpack_common.h
│       ├── llk_unpack_tilize.h
│       ├── llk_unpack_untilize.h
│       └── ...
├── tt_llk_wormhole_b0/           # Wormhole B0-specific (same subdirectory layout)
│   ├── common/inc/
│   ├── instructions/
│   └── llk_lib/
├── tt_llk_quasar/                # Quasar-specific (same subdirectory layout)
│   ├── common/inc/
│   ├── instructions/
│   └── llk_lib/
├── docs/
├── tests/
└── infra/
```

### Naming Convention for LLK Internal Functions

All functions defined in `llk_lib/` files use the **internal** naming convention with leading underscores and trailing underscores:

```cpp
// File: tt_llk_blackhole/llk_lib/llk_math_common.h, line 37
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(const std::uint32_t srca_data_format,
                                     const std::uint32_t srcb_data_format)
```

The pattern is `_llk_<subsystem>_<operation>_()`. These are the raw hardware-manipulating functions that directly issue Tensix instructions (e.g., `TT_OP_UNPACR`, `cfg_reg_rmw_tensix`, `TTI_STALLWAIT`).

## 2. TT-Metal Directories That Consume LLK

### 2.1 Submodule Mount Point: `tt_metal/third_party/tt_llk/`

This is a Git submodule pointing to the TT-LLK repository. The entire LLK source tree is available here.

### 2.2 CKernel API Wrappers: `tt_metal/hw/ckernels/`

This directory contains the **wrapper layer** that bridges TT-Metal's compute kernel API to the raw LLK functions. It is organized per architecture:

```
tt_metal/hw/ckernels/
├── blackhole/metal/
│   ├── common/
│   │   └── chlkc_list.h
│   ├── llk_api/                  # LLK API wrappers (llk_* functions)
│   │   ├── llk_math_common_api.h
│   │   ├── llk_math_matmul_api.h
│   │   ├── llk_math_unary_datacopy_api.h
│   │   ├── llk_math_binary_api.h
│   │   ├── llk_math_unary_sfpu_api.h
│   │   ├── llk_math_reduce_api.h
│   │   ├── llk_pack_api.h
│   │   ├── llk_unpack_A_api.h
│   │   ├── llk_unpack_AB_api.h
│   │   ├── llk_unpack_AB_matmul_api.h
│   │   ├── llk_unpack_common_api.h
│   │   ├── llk_unpack_reduce_api.h
│   │   ├── llk_unpack_tilize_api.h
│   │   ├── llk_unpack_untilize_api.h
│   │   ├── llk_sfpu/             # SFPU operation implementations
│   │   └── ...
│   └── llk_io/                   # Circular-buffer I/O integration
│       ├── llk_io.h
│       ├── llk_io_pack.h
│       ├── llk_io_unpack.h
│       ├── llk_operands.h
│       └── llk_outputs.h
├── wormhole_b0/metal/            # Same structure for Wormhole B0
└── quasar/metal/                 # Same structure for Quasar
```

**Wrapper functions** use the public naming convention (no leading/trailing underscores) and call through to the internal LLK functions. For example:

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_common_api.h, line 25
template <bool is_fp32_dest_acc_en>
inline void llk_math_hw_configure(const std::uint32_t srca_operand,
                                   const std::uint32_t srcb_operand) {
    std::uint32_t srca_operand_id = get_operand_id(srca_operand);
    std::uint32_t srcb_operand_id = get_operand_id(srcb_operand);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(
        unpack_dst_format[srca_operand_id], unpack_dst_format[srcb_operand_id]);
}
```

The wrapper's job is to **translate operand IDs into data formats** and other Metal-specific concepts, then call the corresponding `_llk_*_()` internal function.

### 2.3 Compute Kernel Public API: `tt_metal/hw/inc/api/compute/`

This directory contains the **user-facing Compute Kernel API** -- the functions that kernel authors call directly. These are organized by operation:

```
tt_metal/hw/inc/api/compute/
├── compute_kernel_api.h          # Master header, defines MATH()/PACK()/UNPACK() macros
├── compute_kernel_hw_startup.h   # Hardware initialization
├── matmul.h                      # matmul_tiles(), mm_init(), mm_init_short()
├── tile_move_copy.h              # copy_tile(), copy_tile_init()
├── eltwise_binary.h              # Binary element-wise operations
├── eltwise_binary_sfpu.h         # Binary SFPU element-wise operations
├── pack.h                        # pack_tile(), pack_tile_block()
├── cb_api.h                      # cb_wait_front(), cb_pop_front(), cb_reserve_back(), cb_push_back()
├── reg_api.h                     # tile_regs_acquire(), tile_regs_commit(), tile_regs_wait(), tile_regs_release()
├── reduce.h                      # Reduction operations
├── tilize.h                      # Tilize operations
├── untilize.h                    # Untilize operations
├── reconfig_data_format.h        # Data format reconfiguration
├── bcast.h                       # Broadcast operations
├── eltwise_unary/                # Unary element-wise operations (one file per op)
│   ├── eltwise_unary.h
│   ├── exp.h
│   ├── relu.h
│   ├── sqrt.h
│   ├── gelu.h
│   └── ... (50+ operation headers)
└── experimental/                 # Experimental APIs
```

### 2.4 JIT Build System: `tt_metal/jit_build/`

The JIT build system compiles compute kernels at runtime. The key file for understanding TRISC compilation is:

```
tt_metal/jit_build/
├── genfiles.cpp                  # Generates per-TRISC source files from a single kernel
├── genfiles.hpp
├── build.cpp / build.hpp         # JIT compilation driver
├── jit_build_options.cpp/.hpp    # Compiler flags and defines
└── ...
```

### 2.5 TTNN Operations: `ttnn/cpp/ttnn/operations/`

TTNN operations sit at the top of the stack. They define compute kernels as `.cpp` files that include headers from `tt_metal/hw/inc/api/compute/`. For example:

```
ttnn/cpp/ttnn/operations/
├── matmul/                       # Matrix multiplication
├── eltwise/
│   ├── binary/device/kernels/compute/eltwise_binary_kernel.cpp
│   ├── unary/device/kernels/compute/...
│   └── ternary/device/kernels/compute/...
├── normalization/
├── pool/
├── conv/
└── ...
```

## 3. Layering Diagram

The following diagram shows how the layers connect, from TTNN operations down to the raw LLK hardware primitives:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TTNN Operations                             │
│  ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/   │
│  ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/           │
│                                                                     │
│  Kernel authors write: kernel_main() using Compute API functions    │
│  e.g. mm_init(), matmul_tiles(), pack_tile(), cb_wait_front()      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ #include
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Compute Kernel Public API                        │
│  tt_metal/hw/inc/api/compute/                                      │
│                                                                     │
│  Defines MATH(), PACK(), UNPACK() macros that gate code to each    │
│  TRISC processor. Functions here wrap llk_*() calls.               │
│  e.g. matmul_tiles() calls:                                        │
│    UNPACK((llk_unpack_AB_matmul(...)))                              │
│    MATH((llk_math_matmul<...>(...)))                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ #include (via TRISC-specific guards)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLK API Wrappers (per-arch)                      │
│  tt_metal/hw/ckernels/{blackhole,wormhole_b0,quasar}/metal/llk_api/│
│                                                                     │
│  Wrapper functions: llk_<subsystem>_<operation>()                   │
│  Translate Metal operand IDs → data formats, then delegate to       │
│  internal _llk_*_() functions.                                      │
│  e.g. llk_math_hw_configure<>() calls _llk_math_hw_configure_<>()  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ #include
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLK Library (submodule, per-arch)                 │
│  tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0,quasar}/│
│  └── llk_lib/                                                       │
│                                                                     │
│  Internal functions: _llk_<subsystem>_<operation>_()                │
│  Direct hardware manipulation: Tensix instructions, register        │
│  configuration, MOP templates, address mode programming.            │
│  e.g. _llk_math_hw_configure_<>() writes ALU config registers       │
└─────────────────────────────────────────────────────────────────────┘
```

### How a Call Traverses the Stack

Taking `matmul_tiles()` as a concrete example:

1. **TTNN compute kernel** calls `matmul_tiles(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, idst)`.

2. **Compute Kernel API** (`tt_metal/hw/inc/api/compute/matmul.h`, line 137) expands this into TRISC-gated calls:
   ```cpp
   UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index)));
   MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)));
   ```

3. **LLK API Wrapper** (e.g., `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h`) translates operand IDs and calls the internal function:
   ```cpp
   _llk_math_matmul_<...>(...)
   ```

4. **LLK Library** (`tt_llk_blackhole/llk_lib/llk_math_matmul.h`) programs Tensix MOP templates and issues hardware instructions.

---

**Next:** [`key_concepts.md`](./key_concepts.md)
