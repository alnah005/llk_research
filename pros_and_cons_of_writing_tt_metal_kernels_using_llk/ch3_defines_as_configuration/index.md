# Chapter 3: Preprocessor Defines as Configuration

Preprocessor defines are the primary mechanism for parameterizing TT-Metal compute kernels. Rather than passing configuration at runtime or using template parameters, the system generates a header file (`defines_generated.h`) from a host-side `std::map<std::string, std::string>` and includes it at the top of every compiled kernel. This chapter examines the advantages and disadvantages of this approach.

## The Define Flow

The configuration pipeline has three stages:

### 1. Host Program Factory Sets Defines

Each operation's program factory populates a `std::map<std::string, std::string>` that captures the kernel variant to compile. For example, in the eltwise binary program factory:

```cpp
// ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp

std::map<std::string, std::string> eltwise_defines = utils::get_defines(
    op_type, a.dtype(), output.dtype(), fused_activations, operation_attributes.input_tensor_a_activation);

// ...

auto eltwise_binary_kernel_id = tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
    all_device_cores,
    tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .defines = eltwise_defines});
```

The `ComputeConfig` struct (defined in `tt_metal/api/tt-metalium/kernel_types.hpp`) carries the defines map alongside other configuration like math fidelity and compile-time arguments:

```cpp
// tt_metal/api/tt-metalium/kernel_types.hpp

struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    // ...
    // Each unique combination of defines will produce a unique compiled instantiation
    std::map<std::string, std::string> defines;
    // ...
};
```

### 2. JIT Build Emits `defines_generated.h`

During JIT compilation, `genfiles.cpp` iterates the defines map and writes each entry as a `#define` directive:

```cpp
// tt_metal/jit_build/genfiles.cpp (lines 241-243)

settings.process_defines([&gen_defines_file](const string& define, const string& value) {
    gen_defines_file << "#define " << define << " " << value << endl;
});
```

Each TRISC source file (unpack, math, pack) begins with:

```cpp
#define TRISC_MATH          // (or TRISC_UNPACK / TRISC_PACK)
#include "defines_generated.h"
```

### 3. Kernel Source Uses `#ifdef` Branches

The kernel reads these defines at compile time to select behavior. The same kernel source file serves many different operation variants depending on which defines are active.

## Build Cache and Hashing

Each unique combination of defines, compile-time arguments, and kernel source produces a distinct hash. The `Kernel::compute_hash()` method (in `tt_metal/impl/kernels/kernel.cpp`) feeds every define key-value pair into an FNV-1a hash:

```cpp
// tt_metal/impl/kernels/kernel.cpp (lines 365-390)

uint64_t Kernel::compute_hash() const {
    tt::FNV1a hasher;
    for (const auto& [define, value] : this->defines_) {
        hasher.update(define);
        hasher.update(value);
    }
    // ... also hashes named_compile_time_args_, source, compile_time_args_ ...
    return hasher.digest();
}
```

The `JitBuildCache` (in `tt_metal/jit_build/jit_build_cache.cpp`) uses this hash to avoid redundant compilations: if a kernel with the same hash has already been built, the cached binary is reused.

## Chapter Contents

- [**Advantages of Defines**](./advantages_of_defines.md) -- Zero-cost abstraction, single-source polymorphism, build cache deduplication, and host-device decoupling.
- [**Disadvantages of Defines**](./disadvantages_of_defines.md) -- Readability, implicit contracts, combinatorial explosion, error opacity, and refactoring hazards.
