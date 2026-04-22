# Binary Size, Compile-Time Pruning, and Alternatives

## Template Specialization Produces Tight Binaries

Despite the high compilation cost, template specialization is beneficial for the **output**:
each kernel binary contains only the exact code paths it needs. The compiler eliminates all
dead branches at compile time, producing small, fully specialized RISC-V binaries that fit
within the constrained SRAM of Tenstorrent's Tensix cores.

The mechanism is `if constexpr`, which appears **26 times** in a single file like
[`llk_math_eltwise_binary.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h).
For example, the operation-type dispatch:

```cpp
// tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h, lines 56-67
template <EltwiseBinaryType eltwise_binary_type>
inline auto eltwise_binary_func(...) {
    if constexpr (eltwise_binary_type == ELWADD) {
        return TT_OP_ELWADD(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
    } else if constexpr (eltwise_binary_type == ELWSUB) {
        return TT_OP_ELWSUB(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
    } else {
        return TT_OP_ELWMUL(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
    }
}
```

When `eltwise_binary_type` is `ELWADD`, the compiler discards the `ELWSUB` and `ELWMUL`
branches entirely. No runtime branch, no dead code in the binary.

## Nested `if constexpr` Creates Deep Pruning Trees

The pattern is not limited to top-level dispatch. Inside
[`_llk_math_eltwise_binary_standard_`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h#L188-L282),
the broadcast type, fidelity level, and operation type are all checked via nested
`if constexpr`:

```cpp
// llk_math_eltwise_binary.h, lines 204-230 (abbreviated)
if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB)) {
    if constexpr (src_b_bcast_type == BroadcastType::COL) {
        // COL broadcast loop
    } else {
        ckernel_template::run();
        if constexpr (src_b_bcast_type == BroadcastType::SCALAR) {
            TTI_SETRWC(p_setrwc::CLR_B, ...);
        }
    }
} else if constexpr (eltwise_binary_type == ELWMUL) {
    if constexpr (src_b_bcast_type == BroadcastType::COL) {
        // COL broadcast with fidelity handling
        if constexpr (high_fidelity) { ... }
    }
}
```

Each level of nesting eliminates another set of code paths. A kernel that performs
`ELWADD` with `BroadcastType::NONE` at `LoFi` will have all multiply logic, all column
broadcast loops, and all high-fidelity register management stripped out.

## `constexpr` Values Drive Hardware Configuration

Beyond `if constexpr` branching, the templates also compute hardware configuration constants
at compile time:

```cpp
// llk_math_eltwise_binary.h, lines 25-26
constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
constexpr std::uint8_t srcb_incr =
    (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL)
        ? MAX_FPU_ROWS : 0;
```

These values are folded into immediate operands of the generated instructions, eliminating
any runtime computation for register configuration.

## The Dispatcher Pattern: `if constexpr` as Architecture

The top-level `_llk_math_eltwise_binary_` function
([`llk_math_eltwise_binary.h:571-589`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h#L571-L589))
uses `if constexpr` to dispatch between the standard and dest-reuse implementations:

```cpp
template <..., EltwiseBinaryReuseDestType binary_reuse_dest = ...>
inline void _llk_math_eltwise_binary_(...) {
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::NONE) {
        _llk_math_eltwise_binary_standard_<...>(...);
    } else {
        _llk_math_eltwise_binary_with_dest_reuse_<...>(...);
    }
}
```

This is not polymorphism -- it is compile-time code selection. The "wrong" branch is never
compiled into the output. The pattern is repeated throughout the LLK codebase: templates and
`if constexpr` together form a **compile-time dispatch architecture** that replaces what would
traditionally be virtual functions or switch statements.

## Alternative: Runtime-Parameterized Functions

The obvious alternative to the template-heavy design would be runtime-parameterized functions:

```cpp
// Hypothetical runtime-dispatch alternative (NOT the actual design)
void llk_math_eltwise_binary(
    EltwiseBinaryType type,
    BroadcastType bcast,
    DstSync sync,
    bool fp32_acc,
    MathFidelity fidelity,
    EltwiseBinaryReuseDestType reuse,
    uint32_t dst_index) {
    if (type == ELWADD) { ... }
    else if (type == ELWSUB) { ... }
    ...
}
```

This would:

- **Reduce compilation cost** dramatically: one function compiled once, no template instantiation.
- **Increase binary size**: all code paths would be present in the binary, even unused ones.
  On a RISC-V core with limited SRAM, this may push kernels past size limits.
- **Add runtime overhead**: branch mispredictions and inability to fold constants into
  immediates. On a core without branch prediction hardware, every runtime branch is a
  pipeline stall.
- **Lose `static_assert` checking**: compile-time validation like
  `static_assert(binary_reuse_dest != EltwiseBinaryReuseDestType::NONE, ...)`
  ([`llk_math_eltwise_binary.h:393`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h#L393))
  would have to become runtime assertions or be removed entirely.

## Why the Current Approach Is Likely Correct

The template-heavy design is the right choice for the target hardware, for several reasons:

1. **Tensix RISC-V cores have no branch predictor.** Every runtime branch costs cycles.
   Compile-time elimination of branches produces straight-line code that the core can
   execute without stalls.

2. **SRAM is limited.** Each kernel binary must fit in the core's local memory. Template
   specialization ensures only the necessary code is present. A runtime-parameterized
   function carrying all 960 possible code paths would be substantially larger.

3. **Kernels are JIT-compiled on the host.** The host CPU (x86-64) is powerful and has
   abundant memory. Spending extra seconds on template instantiation at `-O3` on the host
   is a reasonable trade to produce optimal device code.

4. **JIT caching amortizes the cost.** As shown in
   [`compilation_impact.md`](./compilation_impact.md), Metal's build cache and optional
   `ccache` integration mean the compilation cost is paid only once per unique kernel
   configuration.

## The Pain Point: Development Iteration Speed

The cost of this design is borne by developers, not by end users:

- **First-time builds are slow.** A fresh build with no cache must compile every kernel
  variant from scratch at `-O3 -flto=auto`.
- **Changing an LLK header invalidates many cached objects.** Because the implementation
  is header-only, a single change to `llk_math_eltwise_binary.h` forces recompilation of
  every kernel that includes it (transitively, through `llk_math_binary_api.h`).
- **Template error messages are dense.** A type error in a 6-parameter template produces
  deeply nested compiler diagnostics that are difficult to read.

These are development-experience costs, not runtime costs. They are the price paid for
the performance characteristics of the generated device code.

---

**Next:** [Chapter 7 -- Cross-Boundary Testing Strategy](../ch7_testing_strategy/index.md)
