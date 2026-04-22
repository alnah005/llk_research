# Chapter 6: Template Design Trade-offs

## Summary

The LLK and Metal compute-kernel APIs are heavily template-parameterized. Functions like
[`_llk_math_eltwise_binary_`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h)
carry up to **6 template parameters**, and the Metal-side wrappers in
[`llk_math_binary_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h)
add their own. Because the entire LLK implementation is header-only, every kernel translation
unit re-parses, type-checks, and instantiates the full template tree from scratch.

This chapter examines the compilation-time cost of that design, its effect on binary size,
and the alternatives that were (implicitly) rejected.

## Sub-pages

| File | Topic |
|------|-------|
| [`compilation_impact.md`](./compilation_impact.md) | Template parameter explosion and its effect on JIT compile time |
| [`binary_size_and_alternatives.md`](./binary_size_and_alternatives.md) | Binary size implications and why runtime parameterization was not chosen |

## Key Observations

1. **Template parameter counts are high.** The core dispatch function
   `_llk_math_eltwise_binary_` in LLK has 6 template parameters:
   `EltwiseBinaryType`, `BroadcastType`, `DstSync`, `bool is_fp32_dest_acc_en`,
   `MathFidelity`, and `EltwiseBinaryReuseDestType`
   ([`llk_math_eltwise_binary.h:571-577`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h#L571-L577)).
   The Metal wrapper `llk_math_eltwise_binary` in
   [`llk_math_binary_api.h:45-51`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h#L45-L51)
   forwards 5 of these and injects `DST_SYNC_MODE` from a global define.

2. **`if constexpr` is pervasive.** A single file like `llk_math_eltwise_binary.h` contains
   **26 `if constexpr` branch points** that prune dead code paths at compile time. This
   is essential for keeping generated RISC-V binaries small but forces the compiler to parse
   every branch before eliminating unreachable ones.

3. **JIT caching helps -- but only on repeat builds.** Metal's JIT build system
   ([`build.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp))
   hashes all compilation parameters and caches object files. Cache hits skip recompilation
   entirely. First-time or cache-miss builds pay the full template-instantiation cost.

4. **Compiler flags amplify the cost.** Compute kernels (Tensix) are compiled at `-O3` with
   `-flto=auto`
   ([`build.cpp:147,319-320`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L147)).
   Link-time optimization across template-heavy translation units is expensive.

5. **The trade-off is deliberate.** On resource-constrained RISC-V cores with no OS and
   limited SRAM, fully specialized code with zero runtime dispatch overhead is the correct
   default. The cost is borne at compile time on the host, not at runtime on the device.
