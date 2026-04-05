# SFPU Extensions: `llk_api/llk_sfpu/`

## Location and scale

```
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/   (158 files)
```

TT-Metal extends the SFPU (Special Function Processing Unit) operation set far beyond what tt-llk provides natively. While tt-llk ships approximately 52 SFPU kernel files in `tt_llk_wormhole_b0/common/inc/sfpu/`, TT-Metal adds 158 files under `llk_sfpu/` that implement additional math functions needed by TTNN operations.

## Organization: two file types

The `llk_sfpu/` directory contains two kinds of files:

### 1. `ckernel_sfpu_*.h` -- raw SFPU microcode

These implement the actual SFPU computation using the SFPI intrinsic infrastructure (`sfpi::dst_reg`, `sfpi::vFloat`, `sfpi::vInt`, etc.) described in Chapter 3. Each file defines a `calculate_*()` function template in the `ckernel::sfpu` namespace.

Examples of Metal-specific SFPU kernels:

| File | Operation | Description |
|------|-----------|-------------|
| `ckernel_sfpu_celu.h` | CELU activation | `alpha * (exp(x/alpha) - 1)` for x < 0 |
| `ckernel_sfpu_cbrt.h` | Cube root | Newton's method with magic constant approximation |
| `ckernel_sfpu_addcdiv.h` | Ternary addcdiv | `a + value * (b / c)` |
| `ckernel_sfpu_binary_pow.h` | Binary power | `x^y` for two input tensors |
| `ckernel_sfpu_erfinv.h` | Inverse error function | Numerical approximation of `erf^{-1}(x)` |
| `ckernel_sfpu_softplus.h` | Softplus activation | `(1/beta) * log(1 + exp(beta * x))` |
| `ckernel_sfpu_gcd.h` | Greatest common divisor | Integer GCD via SFPU |
| `ckernel_sfpu_cumsum.h` | Cumulative sum | Running sum across tile elements |
| `ckernel_sfpu_topk.h` | Top-K | Partial sorting in SFPU registers |

### 2. `llk_math_eltwise_*_sfpu_*.h` -- LLK wrapper/dispatch

These wire the raw SFPU kernels into the LLK dispatch framework. They define `llk_math_eltwise_{unary,binary,ternary}_sfpu_*()` and corresponding `*_init()` functions that set up SFPU state and call through `_llk_math_eltwise_*_sfpu_params_()`.

## How an SFPU extension is structured

### Example: CELU (Continuously Differentiable Exponential Linear Unit)

**Step 1: The SFPU microcode** (`ckernel_sfpu_celu.h`)

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_celu.h, lines 10-42

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_celu(uint32_t param0, uint32_t param1) {
    // param0 = alpha, param1 = alpha_recip (both FP16_B encoded)
    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        v_if(v < sfpi::vConst0) {
            sfpi::vFloat exp_val = _sfpu_exp_21f_<true>(v * alpha_recip);
            sfpi::vFloat result = alpha * (exp_val - sfpi::vConst1);
            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
            }
            sfpi::dst_reg[0] = result;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
```

Key patterns visible here:
- The `ITERATIONS` template parameter controls how many 16-element rows of the destination register to process (typically 8, covering one face of 128 elements).
- `sfpi::dst_reg[0]` reads the current element, and `sfpi::dst_reg++` advances to the next row.
- `v_if`/`v_endif` provides predicated execution -- only negative values are transformed, positive values pass through unchanged.
- `_sfpu_exp_21f_()` is reused from tt-llk's exponential implementation.
- The `is_fp32_dest_acc_en` guard handles FP16_B rounding when not in FP32 accumulation mode.

**Step 2: The LLK wrapper** (`llk_math_eltwise_unary_sfpu_activations.h`)

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/
//   llk_math_eltwise_unary_sfpu_activations.h, lines 42-58

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_celu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::celu, APPROXIMATE>();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_celu(
    uint dst_index, uint32_t alpha, uint32_t alpha_recip,
    int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        [](uint32_t alpha, uint32_t alpha_recip) {
            ckernel::sfpu::calculate_celu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>(
                alpha, alpha_recip);
        },
        dst_index, vector_mode, alpha, alpha_recip);
}
```

The wrapper:
1. Calls `llk_math_eltwise_unary_sfpu_init<>()` to configure the SFPU hardware (stall settings, address modes).
2. Wraps the compute function in a lambda and passes it to `_llk_math_eltwise_unary_sfpu_params_()` from tt-llk, which handles the face iteration pattern based on `VectorMode`.

### Example: Cube root (more complex math)

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h, lines 17-71

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_cube_root() {
    sfpi::vFloat negative_third_256 = -0x1.555556p-10f;
    sfpi::vFloat magic = 1418472267.0f / 256.0f + 8388608.0f;

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::vFloat x = sfpi::abs(a);

        // Bit-level initial approximation using magic constant 0x548c2b4b
        sfpi::vFloat f = sfpi::int32_to_float(sfpi::reinterpret<sfpi::vInt>(x), 0);
        f = f * negative_third_256 + magic;
        sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(f) << 8);

        // Newton-Raphson refinement
        if constexpr (is_fp32_dest_acc_en) {
            // Two refinement iterations for FP32 precision
            sfpi::vFloat c = (x * y) * (y * y);
            y = y * (c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1)
                     + sfpi::vConstFloatPrgm0);
            sfpi::vFloat d = x * (y * y);
            // ... second iteration ...
            sfpi::dst_reg[0] = y;
        } else {
            // Single refinement for FP16_B
            sfpi::vFloat d = x * (y * y);
            sfpi::vFloat c = d * y;
            sfpi::vFloat t = c * (sfpi::vConstFloatPrgm2 * c + sfpi::vConstFloatPrgm1)
                             + sfpi::vConstFloatPrgm0;
            d = sfpi::setsgn(d, a);
            y = d * (t * t);
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
        }
        sfpi::dst_reg++;
    }
}
```

This demonstrates more advanced SFPU programming techniques:
- **Bit-level manipulation**: Using `sfpi::reinterpret<>` to treat floats as integers for the magic-constant initial approximation.
- **Programmable constants**: `sfpi::vConstFloatPrgm0`/`1`/`2` are set during `cube_root_init()` and loaded into SFPU constant registers.
- **Precision-dependent paths**: FP32 mode gets an extra Newton-Raphson iteration for higher accuracy.

### Example: Addcdiv (ternary operation)

Ternary operations (three inputs) use a different dispatch path:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_addcdiv.h, lines 14-41

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format,
          int ITERATIONS>
inline void calculate_addcdiv(
    const uint dst_index_in0, const uint dst_index_in1,
    const uint dst_index_in2, const uint dst_index_out,
    const uint value) {
    constexpr uint dst_tile_size_sfpi = 32;
    const sfpi::vFloat value_float = Converter::as_float(value);

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 + (in1 * value_float * _sfpu_reciprocal_<2>(in2));
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}
```

Ternary SFPU operations access multiple tiles in the destination register simultaneously by using offset indexing (`dst_index_in0 * dst_tile_size_sfpi`). The dispatch wrapper uses `_llk_math_eltwise_ternary_sfpu_params_()`:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/
//   llk_math_eltwise_ternary_sfpu_addcdiv.h, lines 12-23

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format,
          int ITERATIONS = 8>
inline void llk_math_eltwise_ternary_sfpu_addcdiv(
    uint dst_index0, uint dst_index1, uint dst_index2, uint odst, uint value,
    int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_addcdiv<APPROXIMATE, is_fp32_dest_acc_en, data_format, ITERATIONS>,
        dst_index0, dst_index1, dst_index2, odst, vector_mode, value);
}
```

## The dispatch infrastructure

All SFPU extensions rely on the generic dispatch function `_llk_math_eltwise_unary_sfpu_params_()` from tt-llk (`tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`):

```cpp
// tt-llk: tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h, lines 13-73

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Callable&& sfpu_func, std::uint32_t dst_index,
    int vector_mode = static_cast<int>(VectorMode::RC), Args&&... args) {

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    VectorMode mode = static_cast<VectorMode>(vector_mode);
    if (mode == VectorMode::R) {
        // Row vector: process Face0 + Face1 (first row of each), skip Face2 + Face3
        for (int face = 0; face < 2; face++) {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        // Skip 4 more sub-faces to advance past Face2 + Face3
        // ...
    } else if (mode == VectorMode::C) {
        // Column vector: process Face0 + Face2 (all rows)
        for (int face = 0; face < 2; face++) {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
            // Advance past full face (4 sub-faces of 8 rows each)
            // ...
        }
    } else if (mode == VectorMode::RC) {
        // Full tile: process all 4 faces
        for (int face = 0; face < 4; face++) {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    } else {
        // Single face
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}
```

This function:
1. Sets the destination register write address for the given `dst_index`.
2. Stalls until the SFPU is available.
3. Calls the SFPU function once per face (or subset of faces for row/column vector modes), advancing the read/write cursor between faces using `TTI_SETRWC`.
4. Clears state and waits for the SFPU to finish.

The variadic template with perfect forwarding (`Callable&&`, `Args&&...`) allows any compute function with any parameter signature to be dispatched through this single infrastructure.

## Macro-based dispatch helpers

For common patterns, `llk_math_eltwise_unary_sfpu_macros.h` provides macros that reduce boilerplate:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/
//   llk_math_eltwise_unary_sfpu_macros.h (selected macros)

// No-parameter kernel
#define SFPU_UNARY_NO_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                \
        ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE)

// One-parameter kernel
#define SFPU_UNARY_ONE_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(                         \
        ckernel::sfpu::FN<APPROXIMATE>, DST_IDX, (int)VectorMode::MODE, PARAM0)

// Init with custom callback
#define SFPU_INIT_KERNEL_CALL(OP, INIT_CB, APPROXIMATE) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP, APPROXIMATE>(INIT_CB<APPROXIMATE>)

// Comparison with zero
#define SFPU_ZERO_KERNEL(OP, MODE, APPROXIMATE, DST_IDX) \
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(   \
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::OP>, \
        DST_IDX, (int)VectorMode::MODE, 8);
```

There are over 25 macro variants handling different combinations of template parameters, runtime arguments, data formats, and iteration counts. These are used extensively by the simpler SFPU extension wrappers.

## What tt-llk provides vs. what Metal adds

### SFPU operations in tt-llk (52 files in `tt_llk_wormhole_b0/common/inc/sfpu/`)

Core math: `exp`, `log`, `sqrt`, `rsqrt`, `recip`, `sigmoid`, `tanh`, `gelu`, `relu`, `abs`, `sign`, `square`, `elu`, `hardtanh`, `clamp`, `dropout`, `topk`, `typecast`, `trigonometry`, `exp2`, `cumsum`, `binary`, `binary_bitwise`, `shift`, `quant`, `mul_int`, `add_int`, `sub_int`, `max_pool_indices`, `reduce`, `reshuffle_rows`, `activations`, `comp`, `converter`, `fill`, `is_fp16_zero`, `isinf_isnan`, `load_config`, `negative`, `polyval`, `rounding_ops`, `tanh_derivative`, `threshold`, `welfords`, `where`, `silu`, `square`, `cast_fp32_to_fp16a`, `add_top_row`, `ema`, `reduce_custom`, `cdf`, `rsqrt_compat`.

### SFPU operations added by Metal (90+ unique compute kernels)

The Metal additions fall into several categories:

**Activation functions**: `celu`, `softplus`, `softshrink`, `softsign`, `hardmish` (hard mish), `selu`, `prelu`, `logsigmoid`, `sigmoid_appx` (fast sigmoid approximation), `xielu`.

**Advanced math**: `cbrt` (cube root), `erfinv` (inverse error function), `erf_erfc`, `log1p`, `expm1`, `exp2`, `i0` (Bessel I0), `i1` (Bessel I1).

**Binary operations via SFPU**: `binary_pow`, `binary_fmod`, `binary_max_min`, `binary_comp`, `rdiv` (reverse divide), `rpow` (reverse power), `remainder`.

**Integer operations**: `div_int32`, `div_int32_floor`, `mul_int32`, `remainder_int32`, `rsub_int32`, `int_sum`, `left_shift`, `right_shift`.

**Bitwise operations**: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`.

**Ternary operations**: `addcdiv` (`a + v*b/c`), `addcmul` (`a + v*b*c`), `lerp` (linear interpolation), `where` (conditional select).

**Comparison/logic**: `unary_comp`, `logical_not`, `signbit`, `heaviside`.

**Data manipulation**: `copy_dest_values`, `identity`, `mask`, `conversions`, `rand`, `tiled_prod`, `reduce` (SFPU-based reduction).

**Specialized**: `topk`, `cumsum`, `reshuffle_rows`, `add_top_row`, `alt_complex_rotate90`, `max_pool_indices`, `quant`.

## Init functions and programmable constants

Many SFPU extensions require initialization to load programmable constants into the SFPU constant registers:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_cbrt.h, lines 73-78

template <bool APPROXIMATION_MODE>
inline void cube_root_init() {
    sfpi::vConstFloatPrgm0 = 0x1.c09806p0f;
    sfpi::vConstFloatPrgm1 = -0x1.403e6cp0f;
    sfpi::vConstFloatPrgm2 = 0x1.04cdb2p-1f;
}
```

These programmable constants (`vConstFloatPrgm0` through `vConstFloatPrgm2`) are hardware registers in the SFPU that can be loaded once and reused across all iterations, avoiding the overhead of per-iteration constant loading.

The init function is called through the wrapper's `*_init()` method:

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_ternary_sfpu_addcdiv_init() {
    _llk_math_eltwise_ternary_sfpu_init_<SfpuType::addcdiv>();
    sfpu::init_addcdiv<APPROXIMATE>();
}
```

## Complete file listing

For reference, the 158 files in `llk_sfpu/` break down as:

- **90 `ckernel_sfpu_*.h` files**: Raw SFPU compute kernels
- **68 `llk_math_*_sfpu_*.h` dispatch/wrapper files** (66 `llk_math_eltwise_*` + 2 non-eltwise)
  - `llk_math_eltwise_unary_sfpu_*.h` (39 files): Unary operation wrappers
  - `llk_math_eltwise_binary_sfpu_*.h` (23 files): Binary operation wrappers
  - `llk_math_eltwise_ternary_sfpu_*.h` (4 files): Ternary operation wrappers
  - `llk_math_ema_sfpu_entry.h`, `llk_math_welfords_sfpu_entry.h`: Specialized entry points

This extensive SFPU extension set is what allows TTNN to map the full breadth of PyTorch operations down to efficient Tensix compute kernels.

---

**Next:** [Chapter 5 -- Compute Kernel API](../ch5_compute_kernel_api/index.md)
