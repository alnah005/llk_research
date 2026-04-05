# API Naming Conventions

The LLK API uses a strict two-tier naming convention that separates hardware-level implementation functions from the higher-level wrappers that kernel authors interact with. This section documents the naming scheme, function categories, and key operations.

## Two-tier naming: internal vs. wrapper functions

### Internal functions (`_llk_<operation>_()`)

Internal functions are defined in TT-LLK's `llk_lib/` directory. They carry a **leading underscore** and a **trailing underscore**, and they operate on raw hardware parameters (face dimensions, data format IDs, raw addresses):

```cpp
// tt_llk_blackhole/llk_lib/llk_math_matmul.h, lines 591-600
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_init_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t transpose      = 0,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1)
```

```cpp
// tt_llk_blackhole/llk_lib/llk_math_matmul.h, lines 628-629
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_(std::uint32_t dst_index,
    const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1)
```

```cpp
// tt_llk_blackhole/llk_lib/llk_pack.h, lines 428-429
template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
```

```cpp
// tt_llk_blackhole/llk_lib/llk_unpack_A.h, lines 195
inline void _llk_unpack_A_init_(...)
```

Each internal function corresponds to a single hardware programming step: configuring address modifiers, setting up MOP sequences, or executing the actual compute/unpack/pack instruction sequence.

### Wrapper functions (`llk_<operation>()`)

Wrapper functions are defined in tt-metal's `llk_api/` directory. They have **no leading underscore** and **no trailing underscore**. Their role is to translate operand IDs, output IDs, and circular buffer handles into the raw parameters that the internal functions need:

```cpp
// tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h, lines 13-31
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    const bool partial_face = (in0_tile_r_dim < FACE_R_DIM);

    _llk_math_matmul_init_<math_fidelity, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim,
        partial_face, transpose, ct_dim, rt_dim);
}
```

The pattern is consistent: the wrapper calls `get_operand_id()`, `get_operand_tile_r_dim()`, `get_operand_face_r_dim()`, `get_operand_num_faces()`, or the output-side equivalents (`get_output_id()`, `get_output_face_r_dim()`, etc.) to resolve hardware parameters, then delegates to the corresponding `_llk_*_()` internal function.

Similarly for unpack:

```cpp
// tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_api.h, lines 34-60
template <BroadcastType BType, bool acc_to_dest, ...>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces = 0,
    const std::uint32_t within_face_16x16_transpose = 0,
    const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    // ... validation ...
    _llk_unpack_A_init_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces, within_face_16x16_transpose,
        face_r_dim, num_faces,
        operand_unpack_src_format, operand_unpack_dst_format);
}
```

And for pack:

```cpp
// tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h, lines 40-51
template <bool is_fp32_dest_acc_en>
inline void llk_pack_hw_configure(std::uint32_t pack_output) {
    const std::uint32_t output_id = get_output_id(pack_output);
    const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
    const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
    const std::uint32_t num_faces = get_output_num_faces(output_id);
    // ...
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
        pack_src_format[output_id], pack_dst_format[output_id],
        tile_size, face_r_dim, tile_c_dim, num_faces, partial_face, narrow_tile, 0);
}
```

## Function lifecycle pattern

Most operations follow a three-function lifecycle:

| Function | Purpose | Example |
|----------|---------|---------|
| `_llk_<op>_init_()` / `llk_<op>_init()` | One-time configuration: sets up address modifiers, MOP sequences, and hardware state. | `_llk_math_matmul_init_()` |
| `_llk_<op>_()` / `llk_<op>()` | Per-tile execution: runs the actual compute/unpack/pack operation. | `_llk_math_matmul_()` |
| `_llk_<op>_uninit_()` / `llk_<op>_uninit()` | Teardown: restores hardware state if needed. Often a no-op. | `_llk_math_matmul_uninit_()` |

Additional variants exist for specific configuration steps:

- `_llk_<op>_mop_config_()` -- Configures the MOP (micro-operation program) template for the operation.
- `_llk_<op>_hw_configure_()` -- Performs one-time hardware register configuration.
- `_llk_<op>_configure_addrmod_()` -- Sets up address modifier registers.
- `_llk_<op>_reconfig_data_format_()` -- Reconfigures data format without full re-initialization.

## Function categories

LLK functions are organized into three categories corresponding to the three TRISC processors on each Tensix core:

### Math functions (`llk_math_*`) -- TRISC1

These drive the FPU/SFPU math engine. They read from SrcA/SrcB registers and write to Dest:

| Wrapper file (in `llk_api/`) | Internal file (in `llk_lib/`) | Operation |
|-------------------------------|-------------------------------|-----------|
| `llk_math_matmul_api.h` | `llk_math_matmul.h` | Matrix multiply (MVMUL instruction) |
| `llk_math_binary_api.h` | `llk_math_eltwise_binary.h` | Element-wise binary ops (add, sub, mul) |
| `llk_math_unary_datacopy_api.h` | `llk_math_eltwise_unary_datacopy.h` | Data movement through math pipe (copy/transpose) |
| `llk_math_unary_sfpu_api.h` | `llk_math_eltwise_unary_sfpu.h` | Unary SFPU operations (exp, sqrt, sigmoid, etc.) |
| `llk_math_binary_sfpu_api.h` | `llk_math_eltwise_binary_sfpu.h` | Binary SFPU operations |
| `llk_math_reduce_api.h` | `llk_math_reduce.h` | Reduction operations (row/column reduce) |
| `llk_math_transpose_dest_api.h` | `llk_math_transpose_dest.h` | In-place transpose within Dest register |
| `llk_math_common_api.h` | `llk_math_common.h` | Shared math infrastructure (dest acquire/release, wait) |

Additional internal files with no direct wrapper (used via the SFPU wrappers):
- `llk_math_eltwise_unary_sfpu_params.h` -- SFPU parameter structures
- `llk_math_eltwise_binary_sfpu_params.h` -- Binary SFPU parameter structures
- `llk_math_eltwise_ternary_sfpu.h` / `_params.h` -- Ternary SFPU operations
- `llk_math_welfords_sfpu.h` / `_params.h` -- Welford's online algorithm for variance

### Unpack functions (`llk_unpack_*`) -- TRISC0

These drive the unpacker engines that read tiles from L1 circular buffers into SrcA/SrcB registers:

| Wrapper file (in `llk_api/`) | Internal file (in `llk_lib/`) | Operation |
|-------------------------------|-------------------------------|-----------|
| `llk_unpack_A_api.h` | `llk_unpack_A.h` | Unpack single operand into SrcA |
| `llk_unpack_AB_api.h` | `llk_unpack_AB.h` | Unpack two operands into SrcA and SrcB |
| `llk_unpack_AB_matmul_api.h` | `llk_unpack_AB_matmul.h` | Unpack for matmul (specialized A+B path) |
| `llk_unpack_AB_reduce_api.h` | `llk_unpack_AB_reduce.h` | Unpack for reduce operations |
| `llk_unpack_tilize_api.h` | `llk_unpack_tilize.h` | Unpack with row-major to tile-order conversion |
| `llk_unpack_untilize_api.h` | `llk_unpack_untilize.h` | Unpack with tile-order to row-major conversion |
| `llk_unpack_reduce_api.h` | `llk_unpack_reduce.h` | Unpack with reduction scaling |
| `llk_unpack_common_api.h` | `llk_unpack_common.h` | Shared unpack infrastructure |

### Pack functions (`llk_pack*`) -- TRISC2

These drive the packer engine that writes tiles from the Dest register back to L1 circular buffers:

| Wrapper file (in `llk_api/`) | Internal file (in `llk_lib/`) | Operation |
|-------------------------------|-------------------------------|-----------|
| `llk_pack_api.h` | `llk_pack.h` + `llk_pack_common.h` | Pack tiles from Dest to L1 |
| (included via `llk_pack_api.h`) | `llk_pack_untilize.h` | Pack with detilization (tile to row-major) |
| (included via `llk_pack_api.h`) | `llk_pack_rows.h` | Pack partial rows |

Note that `llk_pack_api.h` is a single wrapper that covers all pack operations. It includes `llk_pack.h`, `llk_pack_common.h`, `llk_pack_rows.h`, and `llk_pack_untilize.h` from TT-LLK.

### Support files

Both TT-LLK and tt-metal include additional support files:

- `llk_defs.h` (in `llk_lib/`) -- Common definitions and enums shared across LLK functions.
- `llk_memory_checks.h` (in `llk_lib/`) -- Runtime memory validation utilities.
- `llk_param_structs.h` (in `llk_api/`) -- Parameter structures used by the wrappers.
- `llk_sfpu_types.h` (in `llk_api/`) -- Type definitions for SFPU operations.

## Key operation names

The following table summarizes the core operation names used throughout the LLK naming scheme:

| Operation name | Subsystem | Description |
|----------------|-----------|-------------|
| `matmul` | math, unpack | Matrix multiplication via MVMUL |
| `eltwise_binary` | math, unpack | Element-wise binary (add/sub/mul) via FPU |
| `eltwise_binary_sfpu` | math | Element-wise binary via SFPU |
| `eltwise_unary_datacopy` | math | Data copy/move through math pipe |
| `eltwise_unary_sfpu` | math | Unary SFPU (exp, sqrt, sigmoid, etc.) |
| `eltwise_ternary_sfpu` | math | Ternary SFPU (where, addcmul, etc.) |
| `reduce` | math, unpack | Row/column/scalar reduction |
| `transpose_dest` | math | In-place transpose in Dest register |
| `pack` | pack | Pack tile from Dest to L1 |
| `pack_untilize` | pack | Pack with detilization |
| `pack_rows` | pack | Pack partial rows from Dest |
| `unpack_A` | unpack | Unpack single operand to SrcA |
| `unpack_AB` | unpack | Unpack two operands to SrcA+SrcB |
| `unpack_AB_matmul` | unpack | Matmul-specialized dual unpack |
| `unpack_tilize` | unpack | Unpack with tilization |
| `unpack_untilize` | unpack | Unpack with detilization |

## Template parameters

LLK functions make extensive use of C++ template parameters for compile-time specialization. Common template parameters include:

| Parameter | Type | Description |
|-----------|------|-------------|
| `MathFidelity` | enum | `LoFi`, `HiFi2`, `HiFi3`, `HiFi4` -- controls FPU iteration count |
| `BroadcastType` | enum | `NONE`, `ROW`, `COL`, `SCALAR` -- broadcast mode for binary ops |
| `DstSync` | enum | Destination synchronization mode |
| `DstTileShape` | enum | Tile shape variant (e.g., `Tile32x32`) |
| `EltwiseBinaryReuseDestType` | enum | Controls whether Dest is reused as an input |
| `is_fp32_dest_acc_en` | bool | Whether Dest accumulator uses FP32 precision |
| `acc_to_dest` | bool | Accumulate-to-destination mode |
| `unpack_to_dest` | bool | Direct unpack to Dest register (bypass SrcA/SrcB) |
| `untilize` | bool | Enable detilization during pack |

These template parameters allow the compiler to generate specialized code paths at compile time, eliminating branches in the hot path of tile processing.

---

**Next:** [`arch_differences.md`](./arch_differences.md)
