# Header-Only Benefits

## 2.1 Zero-Cost Abstraction via `inline` Functions

Every LLK function exposed to kernel authors is declared `inline`. This is not merely a hint -- on the Tensix RISC-V cores where code size and call overhead are critical, `inline` ensures the compiler can fold each LLK call directly into the kernel's instruction stream, eliminating function-call prologues, epilogues, and register spills.

**Evidence -- `[tt-llk] tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h`:**

```cpp
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod()
{
    constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
    constexpr std::uint8_t srcb_incr =
        (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL) ? MAX_FPU_ROWS : 0;
    // ... address-modifier configuration elided
}
```

The function is a template parameterized on `EltwiseBinaryType`, `BroadcastType`, and `MathFidelity`. Because these are all compile-time values, the compiler resolves every `if constexpr` and `constexpr` expression at compile time. The resulting machine code contains only the instructions for the *exact* combination the kernel requested -- no branches, no unused paths.

## 2.2 Template Specialization for Compile-Time Behavior Selection

LLK uses C++ template parameters as a type-safe alternative to `#define`-based configuration. A single function template such as `eltwise_binary_func` covers addition, subtraction, and multiplication:

**Evidence -- `[tt-llk] tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h`:**

```cpp
template <EltwiseBinaryType eltwise_binary_type>
inline auto eltwise_binary_func(
    std::uint8_t clr_src, std::uint8_t acc_to_dest,
    std::uint8_t broadcast_type, std::uint8_t addr_mod)
{
    if constexpr (eltwise_binary_type == ELWADD)
        return TT_OP_ELWADD(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
    else if constexpr (eltwise_binary_type == ELWSUB)
        return TT_OP_ELWSUB(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
    else
        return TT_OP_ELWMUL(clr_src, acc_to_dest, broadcast_type, addr_mod, 0);
}
```

The same pattern is repeated in the Layer 2 API wrappers. For example, the Metal-side wrapper at `[tt-metal] hw/ckernels/wormhole_b0/metal/llk_api/llk_math_binary_api.h` exposes:

```cpp
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(const std::uint32_t acc_to_dest = 0);
```

Four template parameters give the compiler up to `5 x 4 x 4 x 3 = 240` potential specializations for this one init function, yet each kernel instantiates only the one it needs. Dead-code elimination (discussed in [`build_time_specialization.md`](./build_time_specialization.md)) removes everything else.

Similarly, `MathFidelity` as a template parameter controls fidelity-related address modifier configuration at compile time:

```cpp
// From llk_math_matmul.h
template <MathFidelity math_fidelity, int THROTTLE_LEVEL>
inline void matmul_configure_addrmod(
    const bool transpose,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM, ...);
```

## 2.3 No Separate Library Build Step

Because LLK is header-only, there is no `libllk.a` or `libllk.so` to build, link against, or distribute. The headers are compiled directly into each kernel translation unit by the JIT build system.

This eliminates an entire class of integration problems:

- **No ABI compatibility concerns.** Different kernels can be compiled with different flags (e.g., different `MathFidelity` settings) without worrying about calling into a pre-compiled library that was built with a conflicting ABI.
- **No link-order issues.** Since everything is inlined at compile time, there is no separate link step for the LLK "library" -- only the final link of the kernel ELF.
- **No stale-library bugs.** It is impossible for a kernel to accidentally link against an older version of a pre-compiled LLK library, because the library does not exist.

## 2.4 Simplified Deployment for Device Targets

The Tensix RISC-V cores run bare-metal firmware with no dynamic linker, no shared-library loader, and no filesystem. In this environment, every piece of code that a kernel needs must be statically linked into its ELF binary.

A header-only design is a natural fit: the JIT compiler (`riscv-tt-elf-g++`) includes LLK headers via `-I` paths set up in `[tt-metal] jit_build/build.cpp` (line 269-285), and the resulting object files contain all the LLK logic the kernel requires. There is no shared library to package, load, or version for the device target.

The include path setup in `build.cpp` demonstrates this directly:

```cpp
std::vector<std::string> includeDirs = {
    // ...
    root_ + "tt_metal/third_party/tt_llk/common",
    // ...
};
```

The arch-specific LLK headers (e.g., `tt_llk_wormhole_b0/llk_lib/`) are added through the HAL's `includes()` query, and the CMake fake-kernels target shows the full set of include directories referencing `tt_llk_${ARCH_NAME_LOWER}/llk_lib` (see `[tt-metal] jit_build/fake_kernels_target/CMakeLists.txt`, lines 80-87).

---

**Next:** [`build_time_specialization.md`](./build_time_specialization.md)
