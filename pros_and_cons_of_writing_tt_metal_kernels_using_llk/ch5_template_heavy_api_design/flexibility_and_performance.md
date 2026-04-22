# Flexibility and Performance

Templates are not incidental to LLK's design -- they are the mechanism by which
hardware-specific code generation happens. Each template instantiation compiles to a
distinct function body, and `if constexpr` branches ensure that only the relevant
hardware instructions survive into the final binary.

## Compile-time specialization per operation variant

The eltwise binary path demonstrates this clearly. A single function
`_llk_math_eltwise_binary_standard_` handles addition, subtraction, and multiplication
across four broadcast modes and multiple fidelity levels, all through template dispatch:

```cpp
// tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity = MathFidelity::LoFi>
inline void _llk_math_eltwise_binary_standard_(
    const ckernel::TensorShape &tensor_shape,
    std::uint32_t dst_index)
```

When `eltwise_binary_type == ELWMUL` and `math_fidelity == MathFidelity::HiFi4`, the
compiler emits the high-fidelity multiply path with its multi-phase fidelity loop. When
`eltwise_binary_type == ELWADD` and `bcast_type == BroadcastType::COL`, it emits the
column-broadcast addition path with per-face-row MOP calls and B-register clearing. The
other branches are eliminated entirely -- no runtime dispatch, no branch misprediction,
no dead code in the binary.

The `constexpr` values computed from template parameters feed directly into hardware
configuration:

```cpp
// tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    constexpr std::uint32_t fidelity_increment =
        is_high_fidelity(math_fidelity) ? 1 : 0;
    constexpr std::uint8_t srcb_incr =
        (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL)
            ? MAX_FPU_ROWS : 0;
    // ...
}
```

These `constexpr` expressions resolve at compile time, meaning the address modifier
registers are programmed with literal constants, not computed values. On Tensix, where
register configuration instructions have fixed encoding widths, this is not just an
optimization -- it is often a correctness requirement.

## SFPU function template parameters

SFPU operations carry the pattern further. The exponential function has six template
parameters that control algorithm selection:

```cpp
// tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h

template <bool APPROXIMATION_MODE, bool SCALE_EN, int ITERATIONS,
          bool FAST_APPROX, bool SKIP_POSITIVE_CHECK, bool CLAMP_NEGATIVE = true>
void _calculate_exponential_(const std::uint16_t exp_base_scale_factor)
```

When `FAST_APPROX && APPROXIMATION_MODE && CLAMP_NEGATIVE` are all true, the function
emits a hand-scheduled SFPLOADMACRO pipeline that processes 256 elements per call (8 iterations × 32 elements) with
clamped negative inputs. When `FAST_APPROX` is false, it falls back to a scalar loop
calling `_calculate_exponential_piecewise_`. These are completely different algorithms
sharing a single entry point, selected entirely at compile time.

Similarly, the reciprocal function selects between three precision tiers:

```cpp
// tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations)
{
    if constexpr (APPROXIMATION_MODE)
    {
        _calculate_reciprocal_fast_7b_(iterations);      // ~7-bit, 1 cycle/32 elements
    }
    else if constexpr (is_fp32_dest_acc_en)
    {
        _calculate_reciprocal_fast_24b_5c_(iterations);  // ~24-bit, 5 cycles/32 elements
    }
    else
    {
        _calculate_reciprocal_fast_8b_3c_(iterations);   // ~8-bit, 3 cycles/32 elements
    }
}
```

Each path is a fundamentally different implementation -- different instruction sequences,
different pipeline scheduling, different register allocation. Templates make it possible
to package all three behind one function name without any runtime cost.

## Interaction with the JIT pipeline via constexpr compile-time arguments

The tt-metal JIT build system bridges the gap between host-side configuration and
template instantiation. When a kernel is created via `CreateKernel`, the host passes
compile-time arguments as a vector of `uint32_t` values. The build system injects these
as a preprocessor define:

```cpp
// tt-metal/tt_metal/jit_build/build.cpp

defines += fmt::format("-DKERNEL_COMPILE_TIME_ARGS={} ",
                       fmt::join(values, ","));
```

On the device side, these values become a `constexpr` array:

```cpp
// tt-metal/tt_metal/hw/inc/api/compile_time_args.h

constexpr auto kernel_compile_time_args =
    make_array<std::uint32_t>(KERNEL_COMPILE_TIME_ARGS);

template <uint32_t Idx>
constexpr uint32_t get_ct_arg() {
    static_assert(Idx < kernel_compile_time_args.size(),
                  "Index out of range");
    return kernel_compile_time_args[Idx];
}

#define get_compile_time_arg_val(arg_idx) get_ct_arg<arg_idx>()
```

Because `kernel_compile_time_args` is `constexpr`, values retrieved via
`get_compile_time_arg_val` are compile-time constants. They can be used as template
arguments, in `if constexpr` branches, or in `static_assert` checks. The JIT system
effectively turns host-side runtime values into device-side compile-time constants,
enabling per-kernel template specialization without requiring the kernel author to
manually instantiate templates.

Global defines like `MATH_FIDELITY` and `DST_ACCUM_MODE` work the same way -- they are
injected by the JIT pipeline and consumed as template arguments deep inside the LLK stack.
The compute API layer uses them to collapse the template parameter space:

```cpp
// tt-metal/tt_metal/hw/inc/api/compute/eltwise_binary.h

MATH((llk_math_eltwise_binary_init_with_operands<
    eltwise_binary_type, NONE, MATH_FIDELITY>(icb0, icb1, acc_to_dest)));
```

Here `MATH_FIDELITY` is a macro that expands to a `MathFidelity` enum value set during
JIT compilation. The kernel author writes `add_tiles_init(cb0, cb1)` and the template
machinery, fed by the JIT pipeline, produces a function body specialized for the exact
fidelity and accumulation mode of that particular program.

## The net effect

The template-heavy design means that each compiled kernel binary contains exactly the code
it needs -- no dispatch tables, no unused algorithm variants, no runtime configuration
overhead. For a device with limited instruction memory and no branch predictor, this is a
significant advantage. The cost is paid at compile time, not at runtime, which is the
right trade-off for production inference workloads where the same kernel may execute
billions of times.

---

**Next:** [`developer_friction.md`](./developer_friction.md)
