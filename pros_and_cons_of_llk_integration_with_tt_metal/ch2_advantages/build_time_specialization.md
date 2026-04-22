# Build-Time Specialization

## 2.5 The RISC-V Cross-Compiler with Full Visibility

LLK headers are not compiled in isolation -- they are compiled as part of each kernel's translation unit by the RISC-V cross-compiler `riscv-tt-elf-g++`. This means the compiler has simultaneous visibility into both the LLK implementation and the kernel code that calls it, enabling optimizations that would be impossible with a pre-compiled library.

**Evidence -- `[tt-metal] jit_build/build.cpp`, lines 112-130:**

```cpp
const std::array<std::string, 2> sfpi_roots = {
    this->root_ + "runtime/sfpi", "/opt/tenstorrent/sfpi"};

// ...
auto gxx = sfpi_roots[i] + "/compiler/bin/riscv-tt-elf-g++";
if (std::filesystem::exists(gxx)) {
    this->gpp_ += gxx + " ";
    // ...
}
```

The build system locates `riscv-tt-elf-g++` from the SFPI toolchain distribution and uses it for all kernel compilations. Because LLK headers are included via `-I` paths (not linked as a pre-built artifact), the compiler sees every `inline` function body and every `constexpr` expression in the same compilation context as the kernel that invokes them.

## 2.6 Link-Time Optimization (LTO) Across LLK and Kernel Code

The JIT build system enables LTO unconditionally for all kernel compilations:

**Evidence -- `[tt-metal] jit_build/build.cpp`, line 147:**

```cpp
string common_flags = "-std=c++17 -flto=auto -ffast-math -fno-exceptions ";
```

The `-flto=auto` flag instructs the compiler to emit LLVM/GCC intermediate representation instead of final machine code at compile time. At link time, the linker merges IR from all translation units and performs whole-program optimization.

For LLK specifically, this means:

- **Dead code elimination:** If a kernel only calls `llk_math_eltwise_binary` with `EltwiseBinaryType::ELWADD`, any unreachable paths for `ELWSUB` or `ELWMUL` that survived template instantiation are eliminated at link time.
- **Cross-function inlining:** Even if a function was not declared `inline` or was too large for the compiler to inline during the initial compilation pass, LTO can inline it at link time when the full call graph is visible.
- **Constant propagation across translation units:** Values known at compile time in the kernel source (e.g., tile dimensions, data format enums) can propagate into LLK function bodies during LTO, enabling further simplification.

The combination of `-flto=auto` with `-ffast-math` is particularly powerful for LLK math operations, as it allows the compiler to reorder and simplify floating-point operations across LLK and kernel boundaries.

## 2.7 Architecture-Specific Define Injection

The HAL (Hardware Abstraction Layer) injects architecture-specific preprocessor defines at JIT compile time. These defines allow LLK code to adapt its behavior to the target hardware using `#ifdef` guards.

**Evidence -- `[tt-metal] llrt/hal/tt-1xx/wormhole/wh_hal.cpp`, line 146:**

```cpp
defines.push_back("ARCH_WORMHOLE");
```

**Evidence -- `[tt-metal] llrt/hal/tt-1xx/blackhole/bh_hal.cpp`, line 141:**

```cpp
defines.push_back("ARCH_BLACKHOLE");
```

These defines are collected by the HAL's `defines()` query and injected into the compiler command line by `JitBuildState` (see `[tt-metal] jit_build/build.cpp`, lines 347-353). LLK code and kernel code can then use `#ifdef ARCH_WORMHOLE` or `#ifdef ARCH_BLACKHOLE` to select hardware-specific code paths.

**Evidence of usage -- `[tt-metal] third_party/tt_llk/tests/sources/eltwise_binary_test.cpp`:**

```cpp
#ifdef ARCH_BLACKHOLE
    // Blackhole-specific test configuration
#endif
```

The CMake fake-kernels target in `[tt-metal] jit_build/fake_kernels_target/CMakeLists.txt` (lines 139-153) demonstrates the same pattern for offline compilation:

```cmake
if(ARCH_NAME_LOWER MATCHES "wormhole")
    target_compile_definitions(jit_kernels_index PRIVATE
        ARCH_WORMHOLE
        NUM_DRAM_BANKS=12
        NUM_L1_BANKS=64)
elseif(ARCH_NAME_LOWER MATCHES "blackhole")
    target_compile_definitions(jit_kernels_index PRIVATE
        ARCH_BLACKHOLE
        NUM_DRAM_BANKS=8
        NUM_L1_BANKS=140)
endif()
```

This mechanism ensures that the *same* LLK header file can serve multiple hardware generations without runtime dispatch overhead -- the preprocessor resolves all `#ifdef` branches before the compiler even sees the code.

## 2.8 Differentiated Optimization Levels per Processor

The JIT build system applies different optimization levels depending on the target processor class:

**Evidence -- `[tt-metal] jit_build/build.cpp`, lines 314-320:**

```cpp
default_compile_opt_level_("Os"),
default_linker_opt_level_("Os") {
    // ...
    if (build_config.core_type == HalProgrammableCoreType::TENSIX &&
        build_config.processor_class == HalProcessorClassType::COMPUTE) {
        this->default_compile_opt_level_ = "O3";
        this->default_linker_opt_level_ = "O3";
```

| Processor | Optimization | Rationale |
|-----------|-------------|-----------|
| Tensix COMPUTE (TRISC2) | `-O3` | Math-heavy LLK code benefits from aggressive loop unrolling, vectorization, and instruction scheduling. Code size is secondary to throughput. |
| All other processors (BRISC, NCRISC, ERISC) | `-Os` | These processors run dataflow and dispatch code where instruction-cache pressure matters more than raw throughput. `-Os` optimizes for size. |

This is a direct consequence of the header-only model: because LLK code is compiled as part of each kernel, the build system can apply the *optimal* optimization level for the processor that will execute it. A pre-compiled LLK library would have to pick a single optimization level at library-build time, which would be suboptimal for at least some targets.

The optimization level can also be overridden per-kernel via `JitBuildSettings`:

```cpp
// build.cpp, lines 507-510
if (settings && !settings->get_compiler_opt_level().empty()) {
    cmd += fmt::format("-{} ", settings->get_compiler_opt_level());
} else {
    cmd += fmt::format("-{} ", this->default_compile_opt_level_);
}
```

---

**Next:** [`tight_coupling_benefits.md`](./tight_coupling_benefits.md)
