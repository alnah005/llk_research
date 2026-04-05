# Binary and Multi-Operand SFPU Operations

## Overview

While most SFPU operations are **unary** (one input tile, one output tile), TT-LLK also supports **binary** (two-input) and **ternary** (three-input) SFPU operations. These are distinct from the FPU's element-wise binary operations (`EltwiseBinaryType::ELWADD`, `ELWMUL`, etc.) -- FPU binary ops use the hardware multiply-accumulate datapath on Source A and Source B registers, while binary SFPU ops perform their computations programmatically within the SFPU, reading multiple tiles from the Destination register.

The key files for multi-operand SFPU operations:

- `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu.h` -- Binary SFPU lifecycle functions
- `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h` -- Binary SFPU face iteration
- `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_ternary_sfpu.h` -- Ternary SFPU lifecycle functions
- `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_ternary_sfpu_params.h` -- Ternary SFPU face iteration
- `tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h` -- Core binary SFPU computation
- `tt_llk_wormhole_b0/llk_lib/llk_math_welfords_sfpu.h` -- Welford's algorithm (specialized multi-tile SFPU)

## Binary SFPU Operations

### The BinaryOp Enum

The `BinaryOp` enum (defined in `tt_llk_wormhole_b0/common/inc/ckernel_defs.h`) lists the supported binary SFPU operations:

```cpp
enum class BinaryOp : std::uint8_t
{
    ADD           = 0,
    SUB           = 1,
    MUL           = 2,
    DIV           = 3,
    RSUB          = 4,   // Reverse subtract (in1 - in0)
    POW           = 5,
    XLOGY         = 6,   // x * log(y)
    RSHFT         = 7,   // Right shift
    LSHFT         = 8,   // Left shift
    LOGICAL_RSHFT = 9,   // Logical (unsigned) right shift
    ADD_TOP_ROW   = 10
};
```

### Programming Model

Binary SFPU operations differ from unary ones in a critical way: they take **three Destination register tile indices** -- `dst_index_in0`, `dst_index_in1`, and `dst_index_out`. Both input tiles and the output tile must all reside in the Destination register simultaneously.

The function signature in `_llk_math_eltwise_binary_sfpu_params_()` (from `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary_sfpu_params.h`):

```cpp
template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args);
```

The face iteration pattern is identical to the unary case -- the same `VectorMode` logic with `SETRWC` advancement applies. The difference is that the callable receives all three tile indices so it can load from two different tiles and store to a third.

### Core Implementation

The core binary computation in `ckernel_sfpu_binary.h` uses `sfpi::dst_reg[]` to access tiles at different offsets:

```cpp
template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_out)
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0    = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1    = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)      { result = in0 + in1; }
        else if constexpr (BINOP == BinaryOp::SUB)  { result = in0 - in1; }
        else if constexpr (BINOP == BinaryOp::MUL)  { result = in0 * in1; }
        else if constexpr (BINOP == BinaryOp::DIV)  { result = in0 * _sfpu_reciprocal_<2>(in1); }
        else if constexpr (BINOP == BinaryOp::RSUB) { result = in1 - in0; }
        else if constexpr (BINOP == BinaryOp::POW)  { result = _calculate_sfpu_binary_power_(in0, in1); }
        // ...

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}
```

Each tile occupies `dst_tile_size_sfpi = 32` rows in the Destination register when accessed via SFPI. The multiplication by this constant converts a tile index into a row offset, allowing the SFPU to read from two separate tiles and write to a third in a single pass.

Note that `BinaryOp::DIV` is implemented as multiplication by the reciprocal (`_sfpu_reciprocal_<2>(in1)`), and `BinaryOp::POW` uses the `_calculate_sfpu_binary_power_()` helper which computes `exp(y * ln(x))` with special-case handling for negative bases with integer exponents.

### Address Modifier Configuration

Binary SFPU operations configure address modifiers similarly to unary operations. Most use `ADDR_MOD_7` with zero-increment. Operations like `mul_int32`, `mul_uint16`, `max`, `min`, `max_int32`, `min_int32`, `max_uint32`, and `min_uint32` additionally configure `ADDR_MOD_6` with `dest.incr = 2` for their specialized two-element-at-a-time access patterns.

## Ternary SFPU Operations

Ternary SFPU operations take **four** Destination register tile indices: three inputs and one output. The primary ternary operation is `SfpuType::where`, which implements conditional selection (analogous to `numpy.where` or `torch.where`).

The lifecycle and face iteration logic in `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_ternary_sfpu.h` and `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_ternary_sfpu_params.h` follow the same pattern as unary and binary SFPU, with the callable signature extended to four tile indices:

```cpp
template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_ternary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,   // condition
    std::uint32_t dst_index_in1,   // true-branch values
    std::uint32_t dst_index_in2,   // false-branch values
    std::uint32_t dst_index_out,   // output
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args);
```

The face iteration uses the same `VectorMode::R`, `VectorMode::C`, `VectorMode::RC` patterns with `SETRWC` advancement. The `where` operation's address modifier configures `ADDR_MOD_6` with `dest.incr = 2`.

## Welford's Online Algorithm

The file `tt_llk_wormhole_b0/llk_lib/llk_math_welfords_sfpu.h` implements a specialized SFPU operation for **Welford's online algorithm**, which computes running mean and variance in a single pass with superior numerical stability compared to naive two-pass or sum-of-squares approaches.

Welford's algorithm maintains three running accumulators:
- **count** -- number of samples seen
- **mean** -- running mean
- **M2** -- running sum of squared deviations from the current mean

For each new sample `x`, the update is:
```
count = count + 1
delta = x - mean
mean  = mean + delta / count
delta2 = x - mean
M2    = M2 + delta * delta2
```

The final variance is `M2 / count` (population) or `M2 / (count - 1)` (sample).

The implementation in `tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_welfords.h` uses a precomputed reciprocal lookup table (`_load_recip_of_idx_()`) to avoid expensive division operations in the inner loop, loading `1/(idx+1)` values into SFPU LREG7 for the `delta / count` step.

### Welford's Lifecycle

The Welford's SFPU has a slightly different initialization from standard unary/binary SFPU:

```cpp
inline void _llk_math_welfords_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    welfords_sfpu_configure_addrmod();
    math::reset_counters(p_setrwc::SET_ABD_F);
    _program_welfords_replay_buffer_();  // Additional: programs replay buffer
}
```

The call to `_program_welfords_replay_buffer_()` sets up an instruction replay buffer that allows the SFPU to efficiently re-execute the Welford update sequence across multiple samples without re-fetching instructions. This optimization is important because Welford's is typically called in a tight loop over many input tiles.

The remaining lifecycle functions (`_llk_math_welfords_sfpu_start_`, `_llk_math_welfords_sfpu_done_`, `_llk_math_welfords_sfpu_inc_dst_face_addr_`) follow the same pattern as unary and binary SFPU operations: set Destination address, stall for SFPU readiness, execute, clear address, and wait for completion.

## Summary: Choosing Between FPU and SFPU Binary Operations

| Aspect | FPU Binary Eltwise | SFPU Binary |
|---|---|---|
| Operand source | Source A + Source B registers | Destination register (two tiles) |
| Precision | 19-bit source format | 32-bit float |
| Operations | `ELWADD`, `ELWSUB`, `ELWMUL`, `ELWDIV`, `ELWLESS` | `BinaryOp::ADD`, `SUB`, `MUL`, `DIV`, `RSUB`, `POW`, `XLOGY`, shifts |
| Performance | Hardware multiply-accumulate datapath | Programmable SFPU microcode |
| Use case | High-throughput element-wise math | Operations requiring 32-bit precision, power, shifts, or complex multi-operand logic |

FPU binary operations are faster because they use the dedicated hardware datapath. SFPU binary operations provide more flexibility and higher precision, and are the only option for operations like power, xlogy, and bitwise shifts that have no FPU hardware equivalent.

---

**Next:** [Chapter 6 -- Hardware Target Differences](../ch6_hardware_targets/index.md)
