# Directory Layout

This section maps out the physical directory structure of the TT-LLK repository (located at `tt_metal/third_party/tt_llk/` within tt-metal). Every path below is relative to that submodule root unless otherwise noted.

## Top-level overview

```
tt_llk/
├── common/                        # Shared cross-architecture headers
│   ├── llk_assert.h
│   └── tensor_shape.h
├── tt_llk_blackhole/              # Blackhole architecture
│   ├── common/inc/                # Ckernel infrastructure headers
│   │   ├── sfpu/                  # SFPU intrinsic wrappers (52 files)
│   │   ├── ckernel.h
│   │   ├── cmath_common.h
│   │   ├── cpack_common.h
│   │   ├── cunpack_common.h
│   │   └── ... (18 headers total)
│   ├── llk_lib/                   # LLK function implementations (28 files)
│   │   └── experimental/          # Experimental LLK functions (13 files)
│   └── instructions/              # Instruction encodings
│       └── assembly.yaml
├── tt_llk_wormhole_b0/            # Wormhole B0 architecture
│   ├── common/inc/                # Ckernel infrastructure headers
│   │   ├── sfpu/                  # SFPU intrinsic wrappers (52 files)
│   │   ├── ckernel.h
│   │   ├── cmath_common.h
│   │   ├── cpack_common.h
│   │   ├── cunpack_common.h
│   │   └── ... (18 headers total)
│   ├── llk_lib/                   # LLK function implementations (28 files)
│   │   └── experimental/          # Experimental LLK functions (4 files)
│   └── instructions/
│       └── assembly.yaml
├── tt_llk_quasar/                 # Quasar architecture
│   ├── common/inc/                # Ckernel infrastructure headers
│   │   ├── sfpu/                  # SFPU intrinsic wrappers (14 files)
│   │   ├── internal/              # Internal helpers
│   │   │   └── circular_buffer_interface.h
│   │   ├── ckernel.h
│   │   ├── cmath_common.h
│   │   ├── cpack_common.h
│   │   ├── cunpack_common.h
│   │   └── ... (19 headers total)
│   ├── llk_lib/                   # LLK function implementations (23 files)
│   └── instructions/
│       └── assembly.yaml
├── docs/
├── tests/
├── infra/
└── CLAUDE.md, README.md, ...
```

## Shared `common/` directory

The repo root contains a `common/` directory with exactly two headers that are shared across all architectures:

**`common/llk_assert.h`** -- Provides the `LLK_ASSERT` macro used throughout the LLK codebase. It has three modes depending on compile-time defines:

1. When `ENABLE_LLK_ASSERT` and `ENV_LLK_INFRA` are both set, it triggers an `ebreak` instruction on the RISC-V core (for standalone LLK infrastructure testing).
2. When only `ENABLE_LLK_ASSERT` is set (the tt-metal path), it delegates to tt-metal's own `ASSERT` macro via `api/debug/assert.h`.
3. When `ENABLE_LLK_ASSERT` is not defined, asserts compile away entirely via `(void)sizeof((condition))` -- a zero-overhead pattern that still forces the compiler to type-check the expression.

```cpp
// common/llk_assert.h, lines 12-13
#define LLK_ASSERT(condition, message) \
    do                                 \
    {                                  \
        if (UNLIKELY(!(condition)))    \
        {                              \
            asm volatile("ebreak");    \
        }                              \
    } while (0)
```

**`common/tensor_shape.h`** -- Defines the `TensorShape` struct and associated constants used across all LLK operations to describe tile geometry. This is a packed 4-byte struct containing `face_r_dim`, `face_c_dim`, `num_faces_r_dim`, and `num_faces_c_dim`:

```cpp
// common/tensor_shape.h, lines 44-74
struct __attribute__((packed)) TensorShape
{
    std::uint8_t face_r_dim;      ///< Row dimension of each face (typically 16)
    std::uint8_t face_c_dim;      ///< Column dimension of each face (always 16 for HW)
    std::uint8_t num_faces_r_dim; ///< Number of faces in row dimension
    std::uint8_t num_faces_c_dim; ///< Number of faces in column dimension
    // ...
};

constexpr TensorShape DEFAULT_TENSOR_SHAPE = {MAX_FACE_R_DIM, MAX_FACE_C_DIM,
                                               MAX_NUM_FACES_R_DIM, MAX_NUM_FACES_C_DIM};
```

The default tile is 32x32 = 4 faces of 16x16. Other configurations like 32x16 (2 faces) or 16x16 (1 face) are supported for narrow/partial tiles.

## Per-architecture directories

Each architecture directory (`tt_llk_blackhole/`, `tt_llk_wormhole_b0/`, `tt_llk_quasar/`) follows the same three-subdirectory pattern, although the contents differ between architectures.

### `common/inc/` -- Ckernel infrastructure headers

These headers define the low-level compute kernel ("ckernel") infrastructure that LLK functions are built on. They include register manipulation primitives, address mode configuration, template-based microcode operations (MOPs), and data format handling. The key files present in all architectures are:

| Header | Purpose |
|--------|---------|
| `ckernel.h` | Top-level ckernel include; pulls in register definitions and base types. |
| `ckernel_defs.h` | Constants for tile dimensions (`TILE_R_DIM`, `FACE_R_DIM`, `FACE_C_DIM`), data format enums, broadcast types. |
| `ckernel_ops.h` | Inline functions that emit Tensix instruction opcodes (e.g., `TT_OP_MVMUL`, `TT_OP_UNPACR`). |
| `ckernel_template.h` | The `ckernel_template` class -- configures and runs MOP (micro-operation) sequences. |
| `ckernel_addrmod.h` | Address modifier configuration (`addr_mod_t`) for controlling how SrcA, SrcB, and Dest pointers advance. |
| `ckernel_instr_params.h` | Instruction parameter constants used across the instruction set. |
| `cmath_common.h` | Math engine infrastructure -- FPU configuration, math reset/wait functions, destination register management. |
| `cpack_common.h` | Packer engine infrastructure -- pack format configuration, destination address management, L1 write setup. |
| `cunpack_common.h` | Unpacker engine infrastructure -- unpack format configuration, source address management, L1 read setup. |

Blackhole and Wormhole B0 share the same set of 18 `common/inc/` headers (with identical file names but architecture-specific implementations). They include additional files not present in Quasar:

- `ckernel_common_ops.h` -- Common arithmetic/logic operations
- `ckernel_debug.h` -- Debug utilities
- `ckernel_globals.h` -- Global state variables
- `ckernel_mutex_guard.h` -- Mutex primitives for Tensix core synchronization
- `ckernel_structs.h` -- Struct definitions for hardware register layouts
- `ckernel_xmov.h` -- Cross-move instruction support

Quasar has 19 headers in `common/inc/` but uses a different set reflecting Quasar's distinct hardware design. Unique to Quasar are:

- `ckernel_dest.h` -- Destination register management specific to Quasar
- `ckernel_pcbuf.h` -- PC buffer management
- `ckernel_proj_params.h` -- Projection parameters
- `ckernel_risc_atomics.h` -- RISC-V atomic operations
- `ckernel_riscv_debug.h` -- RISC-V debug support
- `ckernel_trisc_common.h` -- TRISC common utilities
- `ckernel_vector.h` -- Vector operation support
- `internal/circular_buffer_interface.h` -- Circular buffer interface

### `common/inc/sfpu/` -- SFPU intrinsic wrappers

The SFPU (Special Function Processing Unit) subdirectory contains headers that wrap the SFPU's vector instruction set into C++ inline functions. Each file typically implements one mathematical operation (e.g., `exp`, `sqrt`, `sigmoid`, `gelu`).

Blackhole and Wormhole B0 each have **52 SFPU files** with identical names, covering a broad set of operations. Quasar has only **14 SFPU files** -- a minimal set of core operations:

| Quasar SFPU files (14) | Notable Blackhole/Wormhole-only SFPU files |
|------------------------|---------------------------------------------|
| `ckernel_sfpu_add.h` | `ckernel_sfpu_binary.h` |
| `ckernel_sfpu_exp.h` | `ckernel_sfpu_binary_bitwise.h` |
| `ckernel_sfpu_gelu.h` | `ckernel_sfpu_cdf.h` |
| `ckernel_sfpu_lrelu.h` | `ckernel_sfpu_converter.h` |
| `ckernel_sfpu_recip.h` | `ckernel_sfpu_cumsum.h` |
| `ckernel_sfpu_relu.h` | `ckernel_sfpu_dropout.h` |
| `ckernel_sfpu_rsqrt.h` | `ckernel_sfpu_elu.h` |
| `ckernel_sfpu_sigmoid.h` | `ckernel_sfpu_hardtanh.h` |
| `ckernel_sfpu_silu.h` | `ckernel_sfpu_log.h` |
| `ckernel_sfpu_sqrt.h` | `ckernel_sfpu_polyval.h` |
| `ckernel_sfpu_square.h` | `ckernel_sfpu_quant.h` |
| `ckernel_sfpu_tanh.h` | `ckernel_sfpu_topk.h` |
| `ckernel_sfpu_typecast_fp16b_uint16.h` | `ckernel_sfpu_trigonometry.h` |
| `ckernel_sfpu_typecast_int32_fp32.h` | `ckernel_sfpu_welfords.h` |

The SFPU files in TT-LLK represent the hardware-level intrinsic wrappers. TT-Metal's `llk_api/llk_sfpu/` directory extends this set significantly (see [`arch_differences.md`](./arch_differences.md) for details).

### `llk_lib/` -- LLK function implementations

This is the core of TT-LLK. Every file in `llk_lib/` provides the internal `_llk_<operation>_()` functions that perform the actual hardware programming for compute operations. These are the functions that tt-metal's `llk_api/` wrappers call into.

Blackhole and Wormhole B0 have **28 files** each with identical names:

```
llk_lib/
├── llk_defs.h
├── llk_math_common.h
├── llk_math_eltwise_binary.h
├── llk_math_eltwise_binary_sfpu.h
├── llk_math_eltwise_binary_sfpu_params.h
├── llk_math_eltwise_ternary_sfpu.h
├── llk_math_eltwise_ternary_sfpu_params.h
├── llk_math_eltwise_unary_datacopy.h
├── llk_math_eltwise_unary_sfpu.h
├── llk_math_eltwise_unary_sfpu_params.h
├── llk_math_matmul.h
├── llk_math_reduce.h
├── llk_math_transpose_dest.h
├── llk_math_welfords_sfpu.h
├── llk_math_welfords_sfpu_params.h
├── llk_memory_checks.h
├── llk_pack.h
├── llk_pack_common.h
├── llk_pack_rows.h
├── llk_pack_untilize.h
├── llk_unpack_A.h
├── llk_unpack_AB.h
├── llk_unpack_AB_matmul.h
├── llk_unpack_AB_reduce.h
├── llk_unpack_common.h
├── llk_unpack_reduce.h
├── llk_unpack_tilize.h
└── llk_unpack_untilize.h
```

Quasar has **23 files** with a distinct naming scheme (see [`arch_differences.md`](./arch_differences.md) for a detailed comparison).

### `instructions/` -- Low-level instruction encodings

Each architecture directory contains an `instructions/` directory with an `assembly.yaml` file that defines the complete instruction set for that architecture's Tensix core. This YAML file specifies:

- Instruction mnemonics and binary encodings (`op_binary` fields)
- Instruction types: `LOCAL_CREGS`, `PC_MODIFYING`, `COMPUTE`, `COMMON_CREGS`
- Execution resources: `SYNC`, `MATH`, `PACK`, `UNPACK`, etc.
- Argument bit-field layouts (start bit, width, type)
- Functional coverage annotations (`fcov` tags)

```yaml
# tt_llk_blackhole/instructions/assembly.yaml, lines 19-22 (example structure)
SOME_INSTR:
  instrn_type: LOCAL_CREGS
  ex_resource: SYNC
  op_binary: 0xa0
```

The `assembly.yaml` files differ between architectures to reflect each Tensix variant's instruction set. The instruction definitions in these files correspond to the `TT_OP_*` macros used in `ckernel_ops.h` and throughout `llk_lib/`.

## How tt-metal references TT-LLK

Within tt-metal, the per-architecture ckernel directories mirror the TT-LLK structure:

```
tt_metal/hw/ckernels/
├── blackhole/metal/llk_api/      # 21 wrapper headers + llk_sfpu/ (159 files)
├── wormhole_b0/metal/llk_api/    # 21 wrapper headers + llk_sfpu/ (158 files)
└── quasar/metal/llk_api/         # 7 wrapper headers (no llk_sfpu/ yet)
```

Each `llk_api/` wrapper file includes the corresponding TT-LLK `llk_lib/` header and provides the public `llk_*()` functions. For example, `llk_math_matmul_api.h` includes `llk_math_matmul.h` from TT-LLK:

```cpp
// tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h, line 7
#include "llk_math_matmul.h"
```

The `llk_sfpu/` subdirectories in tt-metal add a large number of SFPU operation wrappers (158-159 files for Blackhole/Wormhole B0) beyond what TT-LLK's `common/inc/sfpu/` provides. This is where tt-metal defines the full set of compute operations that kernel authors can use.

---

**Next:** [`api_naming_conventions.md`](./api_naming_conventions.md)
