# Defines as LLK Configuration

The `defines` field in `ComputeConfig` is the primary mechanism by which host-side TTNN operations control which LLK functions a compute kernel will invoke. Each entry in the `std::map<std::string, std::string>` becomes a `#define` in a generated header (`hlk_defines_generated.h`) that is automatically included during kernel compilation. This lets a single kernel source file serve many different operations.

## How Defines Flow from Host to Device

1. The host-side program factory populates a `std::map<std::string, std::string>`.
2. This map is passed as `ComputeConfig{.defines = ...}` to `CreateKernel()`.
3. The TT-Metal build system emits `hlk_defines_generated.h` containing one `#define` per map entry.
4. The compute kernel's compilation unit includes this generated header.
5. Preprocessor conditionals and macro expansions in the kernel source select LLK paths accordingly.

Each unique combination of defines produces a distinct compiled kernel binary. This means switching from `ELWADD` to `ELWMUL` does not require editing kernel source -- only changing the defines map on the host side.

## Eltwise Binary Op Type Selection

The most fundamental use of defines is selecting which binary operation to perform. The function `utils::get_defines()` in `ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp` (line 38) builds the defines map based on `BinaryOpType`:

```cpp
std::map<std::string, std::string> get_defines(
    BinaryOpType op_type,
    const std::optional<tt::tt_metal::DataType> input_dtype,
    const std::optional<tt::tt_metal::DataType> output_dtype,
    const std::optional<std::vector<EltwiseUnaryWithParam>>& fused_activations,
    const std::optional<EltwiseUnaryWithParam>& input_tensor_a_activation) {

    std::map<std::string, std::string> defines;
    std::string op_name = "sub_tiles";
    std::string op_binary_type = "EltwiseBinaryType::ELWSUB";
    // ...

    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_binary_type = "EltwiseBinaryType::ELWSUB";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        // ... more cases
    }

    defines["ELTWISE_OP"] = op_name;
    defines["ELTWISE_OP_TYPE"] = op_binary_type;
    // ...
    return defines;
}
```

The kernel (`eltwise_binary_kernel.cpp`, line 36) then uses these defines to call LLK:

```cpp
binary_tiles_init<false, ELTWISE_OP_TYPE>(cb_inp0, cb_inp1);
// ...
ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);
```

When `BinaryOpType::ADD` is selected, the preprocessor expands this to:

```cpp
binary_tiles_init<false, EltwiseBinaryType::ELWADD>(cb_inp0, cb_inp1);
// ...
add_tiles(cb_inp0, cb_inp1, i, i, i);
```

## Fused SFPU Operations via `SFPU_OP_CHAIN_0`

The defines mechanism also supports fusing SFPU operations after the primary binary op. When `fused_activations` is provided (e.g., a RELU after an ADD), the host-side code injects an `SFPU_OP_CHAIN_0` define:

From `binary_op_utils.cpp` (line 167):

```cpp
if (fused_activations.has_value()) {
    if (op_type == BinaryOpType::ADD and fused_activations->size() == 1 and
        fused_activations->at(0).type() == UnaryOpType::RELU and
        not input_tensor_a_activation.has_value()) {
        defines["PACK_RELU"] = "1";
    } else {
        defines.merge(
            ttnn::operations::unary::utils::get_block_defines(
                *fused_activations, "0", idst));
    }
}
```

The kernel checks for these defines at compile time (line 119):

```cpp
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);

#ifdef SFPU_OP_INIT_0
    SFPU_OP_INIT_0
    SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
    SFPU_OP_CHAIN_0
#endif
}
```

When `SFPU_OP_CHAIN_0` is defined to, say, `gelu_tile_init(); gelu_tile(i);`, the SFPU GELU activation runs on each tile immediately after the binary operation, all within the same DST register -- avoiding a round-trip through a circular buffer.

## Pre-Scaling via `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_INIT_PRE_IN1_0`

Some composite operations require transforming inputs before the binary op. For example:

- **LOGADDEXP**: `log(exp(a) + exp(b))` -- needs `exp()` applied to both inputs before the add.
- **DIV**: `a / b = a * recip(b)` -- needs `recip()` on the second input before the multiply.
- **RSUB**: `b - a = -a + b` -- needs `neg()` on the first input before the add.

The host-side code sets `SFPU_OP_INIT_PRE_IN0_0` and/or `SFPU_OP_INIT_PRE_IN1_0` defines:

```cpp
case BinaryOpType::LOGADDEXP:
    defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN0_0"));
    defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN1_0"));
    op_name = "add_tiles";
    op_binary_type = "EltwiseBinaryType::ELWADD";
    defines.merge(get_defines(UnaryOpType::LOG, std::nullopt, "0", idst));
    break;

case BinaryOpType::DIV:
    defines.merge(get_defines(UnaryOpType::RECIP, std::nullopt, "PRE_IN1_0"));
    op_name = "mul_tiles";
    op_binary_type = "EltwiseBinaryType::ELWMUL";
    break;
```

The kernel then conditionally applies these pre-scaling operations to the input tiles before the main binary computation (lines 48-105 of `eltwise_binary_kernel.cpp`).

## Typecast Fusion

When input and output data types differ and require a typecast (e.g., BFLOAT16 input to INT32 output), the host injects an `SFPU_OP_CHAIN_0` that calls `typecast_tile()`:

```cpp
if (is_typecast(*input_dtype, *output_dtype)) {
    auto in_dataformat = static_cast<uint32_t>(
        datatype_to_dataformat_converter(input_dtype.value()));
    auto out_dataformat = static_cast<uint32_t>(
        datatype_to_dataformat_converter(output_dtype.value()));
    defines.insert(
        {"SFPU_OP_CHAIN_0",
         fmt::format("typecast_tile_init<{0}u, {1}u>(); typecast_tile<{0}u, {1}u>(i);",
                     in_dataformat, out_dataformat)});
}
```

This is a powerful pattern: arbitrary C++ code is injected as a preprocessor define, expanding inline within the kernel's tile loop.

## Reduction Op and Dimension Selection

Reduction operations use two defines -- `REDUCE_OP` and `REDUCE_DIM` -- to configure the LLK `reduce_tile()` function. From `reduce_op.cpp` (line 20):

```cpp
std::map<std::string, std::string> get_defines(
    tt::tt_metal::ReduceOpMath reduce_op, tt::tt_metal::ReduceOpDim reduce_dim) {
    std::map<std::string, std::string> defines;
    bool do_max = reduce_op == tt::tt_metal::ReduceOpMath::MAX;
    std::string reduce_dim_str;
    switch (reduce_dim) {
        case ReduceOpDim::W: reduce_dim_str = "ReduceDim::REDUCE_ROW"; break;
        case ReduceOpDim::H: reduce_dim_str = "ReduceDim::REDUCE_COL"; break;
        case ReduceOpDim::HW: reduce_dim_str = "ReduceDim::REDUCE_SCALAR"; break;
    }
    defines["REDUCE_OP"] = (do_max ? "PoolType::MAX" : "PoolType::SUM");
    defines["REDUCE_DIM"] = reduce_dim_str;
    return defines;
}
```

These defines are consumed by the `reduce_init()` and `reduce_tile()` LLK functions in the compute kernel, selecting between sum-reduction and max-reduction, and between row, column, and scalar reduction modes.

## `PACK_RELU` -- Hardware-Level Activation

A special optimization for `ADD + RELU`: instead of running RELU as an SFPU operation, the packer hardware can apply it directly:

```cpp
if (op_type == BinaryOpType::ADD and fused_activations->size() == 1 and
    fused_activations->at(0).type() == UnaryOpType::RELU) {
    defines["PACK_RELU"] = "1";
}
```

The kernel then configures the packer (line 40):

```cpp
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
```

This avoids SFPU overhead entirely by using the pack unit's built-in clamping functionality.

## Summary of Common Defines

| Define | Source | Effect |
|--------|--------|--------|
| `ELTWISE_OP` | `binary_op_utils.cpp` | Function name: `add_tiles`, `sub_tiles`, `mul_tiles` |
| `ELTWISE_OP_TYPE` | `binary_op_utils.cpp` | Template parameter: `EltwiseBinaryType::ELWADD`, etc. |
| `SFPU_OP_CHAIN_0` | Unary/binary utils | Inline SFPU code executed after primary op |
| `SFPU_OP_INIT_PRE_IN0_0` | `binary_op_utils.cpp` | Pre-scaling init for input 0 |
| `SFPU_OP_FUNC_PRE_IN0_0` | `binary_op_utils.cpp` | Pre-scaling function for input 0 |
| `SFPU_OP_INIT_PRE_IN1_0` | `binary_op_utils.cpp` | Pre-scaling init for input 1 |
| `SFPU_OP_FUNC_PRE_IN1_0` | `binary_op_utils.cpp` | Pre-scaling function for input 1 |
| `PACK_RELU` | `binary_op_utils.cpp` | Hardware-level RELU in pack unit |
| `REDUCE_OP` | `reduce_op.cpp` | `PoolType::MAX` or `PoolType::SUM` |
| `REDUCE_DIM` | `reduce_op.cpp` | `ReduceDim::REDUCE_ROW`, `REDUCE_COL`, `REDUCE_SCALAR` |

---

**Next:** [`runtime_args_and_compile_time_args.md`](./runtime_args_and_compile_time_args.md)
