# Architecture Differences

Blackhole, Wormhole B0, and Quasar all follow the same high-level directory structure, but the contents diverge significantly. This section documents the concrete differences in file naming, API surface, and SFPU coverage across the three architectures.

## Blackhole vs. Wormhole B0: same names, different implementations

Blackhole and Wormhole B0 have **identical file sets** in both `llk_lib/` (28 files each, plus an `experimental/` subdirectory -- 13 files in Blackhole, 4 in Wormhole B0) and `common/inc/` (18 headers each, plus 52 SFPU files each). Every file name matches one-to-one between the two architectures:

```
tt_llk_blackhole/llk_lib/llk_math_matmul.h
tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h
```

However, the implementations inside these files differ. Each architecture's Tensix variant has different register layouts, instruction timings, address modifier behaviors, and MOP configurations. For example, the `matmul_configure_addrmod` helper in `llk_math_matmul.h` sets architecture-specific address modifier increments based on the tile geometry and fidelity level.

The same applies to `common/inc/` headers: both architectures provide `ckernel.h`, `ckernel_ops.h`, `cmath_common.h`, `cpack_common.h`, `cunpack_common.h`, and the rest of the 18-header set, but the register offsets, instruction opcodes, and hardware constants differ.

Because the file names are identical, the build system selects the correct architecture at compile time via include paths. When tt-metal builds a kernel for Blackhole, it includes from `tt_llk_blackhole/`; for Wormhole B0, from `tt_llk_wormhole_b0/`. The wrapper files in tt-metal's `llk_api/` are similarly duplicated per architecture:

```
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h
```

Both wrapper sets (21 files each, not counting `llk_sfpu/`) have the same file names and near-identical logic.

## Quasar: distinct naming scheme

Quasar's `llk_lib/` has **23 files** (vs. 28 for Blackhole/Wormhole B0) and uses a fundamentally different naming scheme for several operations. The table below shows the key naming divergences:

| Blackhole / Wormhole B0 | Quasar | Notes |
|--------------------------|--------|-------|
| `llk_unpack_A.h` | `llk_unpack_unary_operand.h` | Quasar uses a generic "unary operand" concept with a `UNP_SEL` template parameter selecting unpacker A, B, or Dest |
| `llk_unpack_AB.h` | `llk_unpack_binary_operands.h` | Parallel rename |
| `llk_unpack_AB_matmul.h` | `llk_unpack_matmul.h` | Dropped the `AB_` prefix |
| `llk_unpack_AB_reduce.h` | `llk_unpack_reduce.h` | Dropped the `AB_` prefix (both architectures have `llk_unpack_reduce.h`) |
| `llk_unpack_untilize.h` | *(not present)* | Quasar does not yet support untilize unpack |
| *(not present)* | `llk_unpack_binary_broadcast_operands.h` | Quasar has explicit broadcast variants |
| *(not present)* | `llk_unpack_unary_broadcast_operands.h` | Quasar has explicit broadcast variants |
| `llk_pack.h` | `llk_pack.h` | Same name |
| *(not present)* | `llk_pack_matmul.h` | Quasar has a dedicated matmul packing path |
| `llk_pack_rows.h` | *(not present)* | Quasar does not have pack_rows |
| *(not present)* | `llk_srcs_tdma.h` | Quasar-specific: SrcS TDMA (transfer DMA) operations |
| `llk_math_eltwise_binary_sfpu.h` | *(not present)* | Quasar does not yet support binary/ternary SFPU |
| `llk_math_eltwise_ternary_sfpu.h` | *(not present)* | Same as above |
| `llk_math_welfords_sfpu.h` | *(not present)* | Welford's algorithm not yet ported |
| `llk_math_transpose_dest.h` | *(not present)* | Dest transpose not yet available on Quasar |
| *(not present)* | `llk_math_eltwise_binary_broadcast.h` | Quasar has explicit binary broadcast math |
| *(not present)* | `llk_math_eltwise_unary_sfpu_common.h` | Quasar splits common SFPU logic into its own file |
| *(not present)* | `llk_math_unary_broadcast.h` | Quasar has explicit unary broadcast math |

### Quasar's generic unpack design

The most significant architectural difference is Quasar's `llk_unpack_unary_operand.h`. Instead of separate `llk_unpack_A.h` and `llk_unpack_B.h` files, Quasar uses a single template parameterized by `UNP_SEL`:

```cpp
// tt_llk_quasar/llk_lib/llk_unpack_unary_operand.h, lines 24-25
template <std::uint32_t UNP_SEL, bool IS_32b_DEST_EN>
inline void _llk_unpack_unary_operand_mop_config_(
    const std::uint32_t buf_desc_id, const std::uint32_t num_tiles)
```

Where `UNP_SEL` can be `p_unpacr::UNP_A`, `p_unpacr::UNP_B`, or `p_unpacr::UNP_DEST`. This reflects Quasar's more symmetric hardware design, where the unpacker resources are more interchangeable.

### Quasar's dedicated matmul pack and SrcS TDMA

Quasar introduces `llk_pack_matmul.h`, which handles packing the results of matmul operations differently from generic tile packing. It uses a subblock-aware MOP configuration:

```cpp
// tt_llk_quasar/llk_lib/llk_pack_matmul.h, lines 23-24
inline void _llk_pack_matmul_mop_config_(
    const std::uint32_t buf_desc_id, const std::uint32_t subblock_r_dim,
    const std::uint32_t subblock_c_dim, const std::uint32_t num_subblocks_c_dim)
```

The `llk_srcs_tdma.h` file is entirely unique to Quasar and provides functions for configuring the SrcS (source stream) TDMA engine. This includes both unpack-side and pack-side configuration:

```cpp
// tt_llk_quasar/llk_lib/llk_srcs_tdma.h, lines 21-22
template <std::uint8_t INSTRN_COUNT, std::uint8_t INSTRN_LOOP_COUNT>
inline void _llk_unpack_srcs_config_()

// tt_llk_quasar/llk_lib/llk_srcs_tdma.h, lines 30-31
template <std::uint8_t INSTRN_COUNT, std::uint8_t INSTRN_LOOP_COUNT>
inline void _llk_pack_srcs_config_()
```

## Quasar's smaller `llk_api/` in tt-metal

The wrapper layer in tt-metal reflects Quasar's still-developing support. While Blackhole and Wormhole B0 each have **21 wrapper files** in `llk_api/`, Quasar has only **7**:

| Quasar `llk_api/` files | Blackhole/Wormhole B0 counterpart |
|--------------------------|----------------------------------|
| `llk_math_common_api.h` | `llk_math_common_api.h` |
| `llk_math_matmul_api.h` | `llk_math_matmul_api.h` |
| `llk_math_unary_datacopy_api.h` | `llk_math_unary_datacopy_api.h` |
| `llk_pack_api.h` | `llk_pack_api.h` |
| `llk_unpack_AB_matmul_api.h` | `llk_unpack_AB_matmul_api.h` |
| `llk_unpack_A_api.h` | `llk_unpack_A_api.h` |
| `llk_unpack_common_api.h` | `llk_unpack_common_api.h` |

Missing from Quasar's `llk_api/` (operations not yet wrapped for use in tt-metal):

- `llk_math_binary_api.h` -- Element-wise binary math
- `llk_math_binary_sfpu_api.h` -- Binary SFPU
- `llk_math_reduce_api.h` -- Reductions
- `llk_math_reduce_custom_api.h` -- Custom reductions
- `llk_math_transpose_dest_api.h` -- Dest transpose
- `llk_math_unary_sfpu_api.h` -- Unary SFPU
- `llk_unpack_AB_api.h` -- Generic AB unpack
- `llk_unpack_AB_reduce_api.h` -- Reduce unpack
- `llk_unpack_AB_reduce_custom_api.h` -- Custom reduce unpack
- `llk_unpack_reduce_api.h` -- Reduce-specific unpack
- `llk_unpack_tilize_api.h` -- Tilize
- `llk_unpack_untilize_api.h` -- Untilize
- `llk_param_structs.h` -- Parameter structures
- `llk_sfpu_types.h` -- SFPU type definitions

This means that as of the current codebase, Quasar in tt-metal supports matmul, unary datacopy, and basic pack/unpack, but does not yet expose element-wise binary operations, SFPU operations, reductions, tilize/untilize, or dest transpose through the standard LLK wrapper API.

## SFPU coverage: TT-LLK vs. tt-metal

The SFPU file counts reveal a major difference between what TT-LLK provides at the hardware-intrinsic level and what tt-metal exposes to kernel authors.

### TT-LLK `common/inc/sfpu/` (hardware intrinsics)

| Architecture | File count | Examples |
|-------------|------------|----------|
| Blackhole | 52 | `ckernel_sfpu_exp.h`, `ckernel_sfpu_gelu.h`, `ckernel_sfpu_topk.h`, `ckernel_sfpu_welfords.h` |
| Wormhole B0 | 52 | Identical file set to Blackhole |
| Quasar | 14 | `ckernel_sfpu_exp.h`, `ckernel_sfpu_gelu.h`, `ckernel_sfpu_recip.h`, `ckernel_sfpu_tanh.h` |

These files wrap raw SFPU vector instructions into C++ functions. They implement the core mathematical kernels that run on the SFPU's SIMD pipeline.

### tt-metal `llk_api/llk_sfpu/` (extended operations)

| Architecture | File count | Includes |
|-------------|------------|----------|
| Blackhole | 159 | 90+ `ckernel_sfpu_*.h` + 65+ `llk_math_eltwise_*_sfpu_*.h` |
| Wormhole B0 | 158 | Nearly identical to Blackhole |
| Quasar | *(no `llk_sfpu/` directory)* | SFPU wrappers not yet added |

The tt-metal `llk_sfpu/` directory dramatically extends the set of available SFPU operations. It contains two kinds of files:

1. **`ckernel_sfpu_*.h` files** (approximately 90 in Blackhole) -- These are additional SFPU intrinsic implementations that go beyond what TT-LLK provides. Many are tt-metal-specific operations like `ckernel_sfpu_addcdiv.h`, `ckernel_sfpu_binary_pow.h`, `ckernel_sfpu_erfinv.h`, `ckernel_sfpu_softplus.h`, `ckernel_sfpu_i0.h` (Bessel function), etc.

2. **`llk_math_eltwise_*_sfpu_*.h` files** (approximately 65) -- These are the LLK-level wrappers that plug each SFPU operation into the standard `_llk_math_eltwise_unary_sfpu_` / `_llk_math_eltwise_binary_sfpu_` framework. For example:
   - `llk_math_eltwise_unary_sfpu_exp2.h`
   - `llk_math_eltwise_unary_sfpu_sigmoid.h`
   - `llk_math_eltwise_binary_sfpu_binop.h`
   - `llk_math_eltwise_ternary_sfpu_where.h`

Operations present in tt-metal's `llk_sfpu/` but absent from TT-LLK's `common/inc/sfpu/` include (partial list):

| Category | Operations |
|----------|-----------|
| Activation functions | `softplus`, `softshrink`, `softsign`, `celu`, `selu`, `prelu`, `hardmish`, `xielu` |
| Comparison/logic | `binary_comp`, `unary_comp`, `logical_not`, `signbit` |
| Math extensions | `cbrt`, `log1p`, `expm1`, `erfinv`, `erf_erfc`, `i0`, `i1` |
| Integer arithmetic | `div_int32`, `div_int32_floor`, `mul_int32`, `remainder_int32`, `rsub_int32`, `gcd`, `lcm` |
| Bitwise operations | `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `left_shift`, `right_shift` |
| Binary extensions | `binary_pow`, `binary_fmod`, `binary_max_min`, `logsigmoid`, `xlogy`, `copy_dest_values` |
| Ternary operations | `addcdiv`, `addcmul`, `lerp`, `where` |
| Utility | `rand`, `identity`, `mask`, `int_sum`, `tiled_prod`, `conversions` |

This layering is intentional: TT-LLK provides the stable, hardware-validated core set of SFPU operations, while tt-metal extends it with higher-level operations that may compose multiple SFPU primitives or implement PyTorch-specific semantics.

## `common/inc/` header differences

Beyond file naming, the `common/inc/` headers differ in meaningful ways between architectures:

### Blackhole/Wormhole B0 unique headers (not in Quasar)

- `ckernel_common_ops.h` -- Shared arithmetic/logic helper functions
- `ckernel_debug.h` -- Debug mode utilities
- `ckernel_globals.h` -- Global state variables (operand formats, tile dimensions)
- `ckernel_mutex_guard.h` -- Hardware mutex primitives for TRISC synchronization
- `ckernel_structs.h` -- Register layout struct definitions
- `ckernel_xmov.h` -- Cross-move instruction wrappers

### Quasar unique headers (not in Blackhole/Wormhole B0)

- `ckernel_dest.h` -- Destination register management (Quasar has a different dest register architecture)
- `ckernel_pcbuf.h` -- PC buffer management for instruction sequencing
- `ckernel_proj_params.h` -- Projection parameter configuration
- `ckernel_risc_atomics.h` -- RISC-V atomic operations (Quasar uses a different RISC-V core)
- `ckernel_riscv_debug.h` -- RISC-V-specific debug support
- `ckernel_trisc_common.h` -- Common TRISC utilities (replaces multiple BH/WH headers)
- `ckernel_vector.h` -- Vector operation support
- `internal/circular_buffer_interface.h` -- CB interface for Quasar's memory subsystem

These differences reflect that Quasar is a significantly different Tensix variant with a redesigned RISC-V subsystem, memory hierarchy, and instruction scheduling model.

## Summary table

| Dimension | Blackhole | Wormhole B0 | Quasar |
|-----------|-----------|-------------|--------|
| `llk_lib/` files | 28 | 28 (same names) | 23 (different names) |
| `common/inc/` headers | 18 | 18 (same names) | 19 (different set) |
| `common/inc/sfpu/` files | 52 | 52 (same names) | 14 |
| `instructions/assembly.yaml` | Yes | Yes | Yes |
| tt-metal `llk_api/` wrappers | 21 | 21 | 7 |
| tt-metal `llk_sfpu/` files | 159 | 158 | 0 (no directory) |
| Matmul support | Yes | Yes | Yes |
| Eltwise binary (FPU) | Yes | Yes | In `llk_lib/` only, no wrapper |
| Eltwise SFPU (unary) | Yes | Yes | Structurally different: Quasar has `llk_math_eltwise_unary_sfpu_common.h` (not `_sfpu.h`), with a different interface; no tt-metal wrapper |
| Eltwise SFPU (binary/ternary) | Yes | Yes | Not in `llk_lib/` |
| Reduce | Yes | Yes | In `llk_lib/` only, no wrapper |
| Tilize/untilize | Yes | Yes | Tilize only |
| Dest transpose | Yes | Yes | No |
| SrcS TDMA | No | No | Yes (`llk_srcs_tdma.h`) |
| Dedicated matmul pack | No | No | Yes (`llk_pack_matmul.h`) |

---

**Next:** [Chapter 4 — LLK API Wrappers](../ch4_llk_api_wrappers/index.md)
