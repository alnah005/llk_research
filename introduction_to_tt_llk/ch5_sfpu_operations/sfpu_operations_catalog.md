# SFPU Operations Catalog

## Overview

The `SfpuType` enum (defined in `tests/helpers/include/llk_sfpu_types.h`) enumerates every SFPU operation supported by TT-LLK on WH/BH. Each operation is implemented as a C++ template function in a corresponding header under `tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_*.h`, and the SFPU infrastructure that aggregates all these headers is `tt_llk_wormhole_b0/common/inc/ckernel_sfpu.h`.

Most SFPU operations are parameterized on `bool APPROXIMATION_MODE`, which selects between a faster approximate implementation and a more accurate one. The `APPROXIMATE` flag is typically derived from `MathFidelity` settings at the LLK API layer.

## Activation Functions

These implement the nonlinear activation functions commonly used in neural networks.

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `tanh` | `ckernel_sfpu_tanh.h` | Hyperbolic tangent |
| `hardtanh` | `ckernel_sfpu_hardtanh.h` | Hard tanh (clamped to [-1, 1]) |
| `gelu` | `ckernel_sfpu_gelu.h` | Gaussian Error Linear Unit |
| `gelu_derivative` | `ckernel_sfpu_gelu.h` | Derivative of GELU |
| `gelu_appx` | `ckernel_sfpu_gelu.h` | Approximate GELU |
| `sigmoid` | `ckernel_sfpu_sigmoid.h` | Sigmoid (logistic function) |
| `sigmoid_appx` | `ckernel_sfpu_sigmoid.h` | Approximate sigmoid |
| `silu` | `ckernel_sfpu_silu.h` | Sigmoid Linear Unit (x * sigmoid(x)) |
| `relu_max` | `ckernel_sfpu_relu.h` | ReLU with max threshold |
| `relu_min` | `ckernel_sfpu_relu.h` | ReLU with min threshold |
| `lrelu` | `ckernel_sfpu_activations.h` | Leaky ReLU |
| `prelu` | `ckernel_sfpu_activations.h` | Parametric ReLU |
| `elu` | `ckernel_sfpu_elu.h` | Exponential Linear Unit |
| `celu` | `ckernel_sfpu_activations.h` | Continuously Differentiable ELU |
| `softplus` | `ckernel_sfpu_activations.h` | Softplus (smooth ReLU) |
| `hardsigmoid` | `ckernel_sfpu_activations.h` | Hard sigmoid approximation |
| `tanh_derivative` | `ckernel_sfpu_tanh_derivative.h` | Derivative of tanh |

## Exponential and Logarithmic Functions

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `exponential` | `ckernel_sfpu_exp.h` | Natural exponential (e^x). Uses the `exp_21f` algorithm from Moroz et al. 2022. |
| `exp2` | `ckernel_sfpu_exp2.h` | Base-2 exponential (2^x) |
| `exp_with_base` | `ckernel_sfpu_exp.h` | Exponential with arbitrary base |
| `expm1` | `ckernel_sfpu_exp.h` | exp(x) - 1 (numerically stable near zero) |
| `log` | `ckernel_sfpu_log.h` | Natural logarithm |
| `log_with_base` | `ckernel_sfpu_log.h` | Logarithm with arbitrary base |
| `log1p` | `ckernel_sfpu_log.h` | log(1 + x) (numerically stable near zero) |

## Power and Root Functions

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `sqrt` | `ckernel_sfpu_sqrt.h` | Square root |
| `rsqrt` | `ckernel_sfpu_rsqrt.h` | Reciprocal square root (1/sqrt(x)) |
| `square` | `ckernel_sfpu_square.h` | Square (x^2) |
| `power` | `ckernel_sfpu_binary.h` | Power (x^y), implemented as a binary SFPU op via exp(y * ln(x)) |
| `reciprocal` | `ckernel_sfpu_recip.h` | Reciprocal (1/x), uses Newton-Raphson refinement |

## Trigonometric Functions

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `sine` | `ckernel_sfpu_trigonometry.h` | Sine |
| `cosine` | `ckernel_sfpu_trigonometry.h` | Cosine |
| `tan` | `ckernel_sfpu_trigonometry.h` | Tangent |
| `asin` | `ckernel_sfpu_trigonometry.h` | Arcsine |
| `acos` | `ckernel_sfpu_trigonometry.h` | Arccosine |
| `atan` | `ckernel_sfpu_trigonometry.h` | Arctangent |
| `atanh` | `ckernel_sfpu_trigonometry.h` | Inverse hyperbolic tangent |
| `asinh` | `ckernel_sfpu_trigonometry.h` | Inverse hyperbolic sine |
| `acosh` | `ckernel_sfpu_trigonometry.h` | Inverse hyperbolic cosine |

## Comparison Operations

These operations compare each element against zero or against another value, producing boolean-like results.

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `equal_zero` | `ckernel_sfpu_comp.h` | x == 0 |
| `not_equal_zero` | `ckernel_sfpu_comp.h` | x != 0 |
| `less_than_zero` | `ckernel_sfpu_comp.h` | x < 0 |
| `greater_than_zero` | `ckernel_sfpu_comp.h` | x > 0 |
| `less_than_equal_zero` | `ckernel_sfpu_comp.h` | x <= 0 |
| `greater_than_equal_zero` | `ckernel_sfpu_comp.h` | x >= 0 |
| `heaviside` | `ckernel_sfpu_comp.h` | Heaviside step function |
| `unary_ne` | `ckernel_sfpu_comp.h` | Unary not-equal comparison |
| `unary_eq` | `ckernel_sfpu_comp.h` | Unary equal comparison |
| `unary_gt` | `ckernel_sfpu_comp.h` | Unary greater-than comparison |
| `unary_lt` | `ckernel_sfpu_comp.h` | Unary less-than comparison |
| `unary_ge` | `ckernel_sfpu_comp.h` | Unary greater-or-equal comparison |
| `unary_le` | `ckernel_sfpu_comp.h` | Unary less-or-equal comparison |

## Special Math Functions

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `erf` | `ckernel_sfpu_activations.h` | Gauss error function |
| `erfc` | `ckernel_sfpu_activations.h` | Complementary error function |
| `erfinv` | `ckernel_sfpu_activations.h` | Inverse error function |
| `i0` | `ckernel_sfpu_activations.h` | Modified Bessel function of the first kind, order 0 |
| `isfinite` | `ckernel_sfpu_isinf_isnan.h` | Test for finite value |
| `isinf` | `ckernel_sfpu_isinf_isnan.h` | Test for infinity |
| `isposinf` | `ckernel_sfpu_isinf_isnan.h` | Test for positive infinity |
| `isneginf` | `ckernel_sfpu_isinf_isnan.h` | Test for negative infinity |
| `isnan` | `ckernel_sfpu_isinf_isnan.h` | Test for NaN |
| `logical_not_unary` | `ckernel_sfpu_comp.h` | Logical NOT |

> **Note:** The header `ckernel_sfpu_cdf.h` provides internal CDF helper functions (e.g., `_calculate_sfpu_cdf_appx_()`) that are consumed by other operations such as GELU, but `cdf` is not a standalone `SfpuType` enum value.

## Integer Arithmetic Operations

These operations work on data interpreted as integers rather than floating-point values. They use `InstrModLoadStore::INT32` for load/store format.

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `add_int32` | `ckernel_sfpu_add_int.h` | 32-bit integer addition |
| `mul_int32` | `ckernel_sfpu_mul_int.h` | 32-bit integer multiplication |
| `mul_uint16` | `ckernel_sfpu_mul_int.h` | 16-bit unsigned integer multiplication |
| `add1` | `ckernel_sfpu_add_int.h` | Add 1 (increment) |
| `quant_int32` | `ckernel_sfpu_quant.h` | Quantize to int32 |
| `requant_int32` | `ckernel_sfpu_quant.h` | Requantize int32 (change scale) |
| `dequant_int32` | `ckernel_sfpu_quant.h` | Dequantize from int32 to float |

## Bitwise Operations

These operate on the bit representation of values, treating them as integers.

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `bitwise_xor` | `ckernel_sfpu_binary_bitwise.h` | Bitwise XOR |
| `bitwise_and` | `ckernel_sfpu_binary_bitwise.h` | Bitwise AND |
| `bitwise_or` | `ckernel_sfpu_binary_bitwise.h` | Bitwise OR |
| `bitwise_not` | `ckernel_sfpu_binary_bitwise.h` | Bitwise NOT |
| `left_shift` | `ckernel_sfpu_shift.h` | Left bit shift |
| `right_shift` | `ckernel_sfpu_shift.h` | Right bit shift |

## Reduction and Aggregation Operations

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `reduce` | `ckernel_sfpu_reduce.h` | General reduction within a tile |
| `cumsum` | `ckernel_sfpu_cumsum.h` | Cumulative sum |
| `tiled_prod` | `ckernel_sfpu_activations.h` | Tiled product (product reduction) |
| `add_top_row` | `ckernel_sfpu_add_top_row.h` | Add the top row to all other rows |
| `topk_local_sort` | `ckernel_sfpu_topk.h` | Local sorting phase of top-K algorithm |
| `topk_merge` | `ckernel_sfpu_topk.h` | Merge phase of top-K algorithm |
| `topk_rebuild` | `ckernel_sfpu_topk.h` | Rebuild phase of top-K algorithm |
| `unary_max` | `ckernel_sfpu_comp.h` | Unary max reduction |
| `unary_min` | `ckernel_sfpu_comp.h` | Unary min reduction |
| `unary_max_int32` | `ckernel_sfpu_comp.h` | Unary max reduction (int32) |
| `unary_min_int32` | `ckernel_sfpu_comp.h` | Unary min reduction (int32) |
| `unary_max_uint32` | `ckernel_sfpu_comp.h` | Unary max reduction (uint32) |
| `unary_min_uint32` | `ckernel_sfpu_comp.h` | Unary min reduction (uint32) |

## Type Conversion Operations

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `typecast` | `ckernel_sfpu_typecast.h` | General type cast between data formats |
| `cast_fp32_to_fp16a` | `ckernel_sfpu_cast_fp32_to_fp16a.h` | Cast from Float32 to Float16 (fp16a) |

## Other Operations

| `SfpuType` Value | Implementation Header | Description |
|---|---|---|
| `abs` | `ckernel_sfpu_abs.h` | Absolute value |
| `sign` | `ckernel_sfpu_sign.h` | Sign function (-1, 0, or +1) |
| `signbit` | `ckernel_sfpu_sign.h` | Extract sign bit |
| `negative` / `neg` | `ckernel_sfpu_negative.h` | Negate value |
| `clamp` | `ckernel_sfpu_clamp.h` | Clamp to range [min, max] |
| `dropout` | `ckernel_sfpu_dropout.h` | Dropout (randomly zero elements) |
| `fill` | `ckernel_sfpu_fill.h` | Fill with constant value |
| `where` | `ckernel_sfpu_where.h` | Conditional select (ternary SFPU op) |
| `threshold` | `ckernel_sfpu_threshold.h` | Threshold function |
| `mask` | `ckernel_sfpu_is_fp16_zero.h` | Mask based on fp16 zero detection |
| `floor` | `ckernel_sfpu_rounding_ops.h` | Floor (round toward negative infinity) |
| `ceil` | `ckernel_sfpu_rounding_ops.h` | Ceiling (round toward positive infinity) |
| `remainder` | `ckernel_sfpu_rounding_ops.h` | Remainder after division |
| `fmod` | `ckernel_sfpu_rounding_ops.h` | Floating-point modulo |
| `reshuffle_rows` | `ckernel_sfpu_reshuffle_rows.h` | Reshuffle rows within a tile |
| `max` / `min` | `ckernel_sfpu_binary.h` | Element-wise max/min (binary SFPU) |
| `max_int32` / `min_int32` | `ckernel_sfpu_binary.h` | Element-wise max/min for int32 (binary SFPU) |
| `max_uint32` / `min_uint32` | `ckernel_sfpu_binary.h` | Element-wise max/min for uint32 (binary SFPU) |

## Implementation Pattern

Every SFPU operation follows the same structural pattern. Taking exponential as an example, the implementation in `ckernel_sfpu_exp.h` defines:

1. **A core computation function** (e.g., `_sfpu_exp_()`) that operates on SFPI vector registers and implements the mathematical algorithm. These use SFPI intrinsics like `sfpi::vFloat`, `sfpi::setexp()`, `sfpi::exexp()`, and predicated control flow.

2. **An iteration wrapper** (e.g., `_calculate_exponential_()`) that loops over rows within a face (typically `ITERATIONS = 8` for 8 rows of the SFPU's SIMD width), loading from `sfpi::dst_reg[]`, computing, and storing back.

3. **An init function** (e.g., `_init_exponential_()`) that loads any required constants (like polynomial coefficients) into SFPU LREGs.

The LLK API layer then calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>()` with the iteration wrapper as the callable, and the params function handles face iteration via `VectorMode`.

---

**Next:** [`sfpu_binary_operations.md`](./sfpu_binary_operations.md)
