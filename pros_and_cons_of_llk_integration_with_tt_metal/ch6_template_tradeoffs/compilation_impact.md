# Compilation Impact of Template-Heavy Design

## Template Parameter Explosion

The LLK math library relies on deeply nested template functions where each combination of
parameters produces a distinct instantiation. The most complex example is the core eltwise
binary dispatch function in
[`llk_math_eltwise_binary.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h):

```cpp
// tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h, lines 571-578
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity                   = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_(
    const ckernel::TensorShape &tensor_shape,
    std::uint32_t dst_index,
    const bool clear_fp32_dst_acc = false)
```

This single function has **6 template parameters**. A rough combinatorial count for the
non-defaulted parameters yields:

| Parameter | Values | Count |
|-----------|--------|-------|
| `EltwiseBinaryType` | `ELWMUL`, `ELWDIV`, `ELWADD`, `ELWSUB`, `ELWLESS` | 5 |
| `BroadcastType` | `NONE`, `COL`, `ROW`, `SCALAR` | 4 |
| `DstSync` | `SyncHalf`, `SyncFull` | 2 |
| `is_fp32_dest_acc_en` | `true`, `false` | 2 |
| `MathFidelity` | `LoFi`, `HiFi2`, `HiFi3`, `HiFi4` | 4 |
| `EltwiseBinaryReuseDestType` | `NONE`, `DEST_TO_SRCA`, `DEST_TO_SRCB` | 3 |

The theoretical maximum is 5 x 4 x 2 x 2 x 4 x 3 = **960 instantiations**. In practice a given
kernel only uses one or two combinations, but the compiler must still parse the entire template
definition and all of its `if constexpr` branches for each instantiation.

## The Metal Wrapper Adds Another Layer

Metal's
[`llk_math_binary_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h)
wraps the LLK function with its own template:

```cpp
// tt-metal/.../llk_math_binary_api.h, lines 45-61
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(uint dst_index, const bool clear_fp32_dst_acc = true) {
    ...
    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,           // injected from a global #define
        is_fp32_dest_acc_en,
        math_fidelity,
        binary_reuse_dest>(...);
}
```

This means Metal re-exposes 5 template parameters and adds `DST_SYNC_MODE` as a compile-time
define, effectively creating a two-layer template chain. The compiler must instantiate both the
wrapper and the LLK implementation for every unique call site.

## Header-Only Means Full Re-parsing

Both LLK and the Metal `llk_api/` wrappers are **header-only**. There are no precompiled
`.cpp` files in the LLK `llk_lib/` directory -- every symbol is defined in a `.h` file included
transitively by the kernel source. This means:

- Every translation unit that uses `llk_math_eltwise_binary` pulls in the full include chain:
  [`llk_math_binary_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h)
  includes
  [`llk_math_eltwise_binary.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h),
  which includes `ckernel_template.h`, `cmath_common.h`, `ckernel_ops.h`, and so on.
- The compiler parses, type-checks, and potentially instantiates the **entire LLK implementation**
  for every kernel object file, even though each kernel typically uses only one combination.

## JIT Build Caching Mitigates Repeat Costs

Metal's JIT build infrastructure in
[`build.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp)
provides object-level caching. The system:

1. Hashes all compilation parameters (compiler path, flags, defines, includes, source files,
   and optimization levels) into a `build_state_hash_`
   ([`build.cpp:419-434`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L419-L434)).
2. On each build, checks whether the stored hash matches the current parameters
   ([`build.cpp:439-444`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L439-L444)).
3. Skips compilation for cache hits, logging `"JIT build cache hit"`
   ([`build.cpp:566`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L566)).

This means the second and subsequent runs with identical kernel configurations are fast. But
**first-time compilation** -- or any change to a define, include path, or compiler flag --
invalidates the cache and forces a full rebuild.

Additionally, the environment variable `TT_METAL_CCACHE_KERNEL_SUPPORT` enables `ccache` as a
compiler wrapper
([`build.cpp:105-107`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L105-L107)),
which can share cached compilations across different build directories.

## Aggressive Optimization Flags Compound the Problem

Compute kernels targeting the Tensix processor are compiled at **`-O3`** with **`-flto=auto`**:

```cpp
// tt-metal/tt_metal/jit_build/build.cpp, lines 147, 317-320
string common_flags = "-std=c++17 -flto=auto -ffast-math -fno-exceptions ";
...
if (build_config.core_type == HalProgrammableCoreType::TENSIX &&
    build_config.processor_class == HalProcessorClassType::COMPUTE) {
    this->default_compile_opt_level_ = "O3";
    this->default_linker_opt_level_ = "O3";
```

Non-compute processors (data movement, ethernet) default to `-Os`
([`build.cpp:314-315`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L314-L315)).

The combination of `-O3` (aggressive inlining and loop transformations) with `-flto=auto`
(link-time optimization across all object files) and deeply nested templates creates a
worst-case scenario for compile time. The compiler must:

1. Parse the full header-only LLK tree for each source file.
2. Instantiate all referenced template specializations at `-O3`.
3. Emit LTO-compatible IR instead of final machine code.
4. At link time, re-optimize across all translation units.

For a single kernel this is manageable. For a full model compilation with dozens of unique
kernel configurations, the cumulative cost is significant.

---

**Next:** [`binary_size_and_alternatives.md`](./binary_size_and_alternatives.md)
