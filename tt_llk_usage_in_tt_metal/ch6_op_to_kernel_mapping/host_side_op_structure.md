# Host-Side Op Structure

This section describes how TTNN operations are organized on disk and how their host-side factory files configure and dispatch device-side compute kernels that use LLK.

## Directory Layout

Each TTNN operation follows a consistent directory structure under `ttnn/cpp/ttnn/operations/`. The key convention is:

```
ttnn/cpp/ttnn/operations/<category>/device/
    <op>_device_operation.hpp        # Operation type definitions
    <op>_device_operation.cpp        # Validation, shape computation
    factory/                         # (or inline in device/) Program factory files
        <variant>_program_factory.cpp
    kernels/
        compute/                     # Device-side compute kernels (use LLK)
            <kernel>.cpp
        dataflow/                    # Device-side reader/writer kernels (use NOC)
            reader_*.cpp
            writer_*.cpp
```

For example, the matmul operation:

```
ttnn/cpp/ttnn/operations/matmul/device/
    matmul_device_operation.hpp
    matmul_device_operation.cpp
    factory/
        matmul_multicore_reuse_mcast_2d_program_factory.cpp
        matmul_multicore_reuse_mcast_1d_program_factory.cpp
        matmul_multicore_reuse_program_factory.cpp
        matmul_multicore_program_factory.cpp
        ...
    kernels/
        compute/
            bmm_large_block_zm.cpp
            bmm_large_block_zm_fused_bias_activation.cpp
            ...
        dataflow/
            reader_bmm_tile_layout_in0_sender_padding.cpp
            reader_bmm_tile_layout_in1_receiver_writer_padding.cpp
            ...
```

And eltwise binary:

```
ttnn/cpp/ttnn/operations/eltwise/binary/device/
    binary_device_operation.hpp
    element_wise_multi_core_program_factory.cpp
    broadcast_height_multi_core_program_factory.cpp
    ...
    kernels/
        compute/
            eltwise_binary_kernel.cpp
            bcast_h.cpp
            bcast_w.cpp
            ...
        dataflow/
            reader_binary_interleaved_start_id.cpp
            ...
```

## The Program Factory Pattern

Each program factory file is responsible for:

1. **Allocating circular buffers** on device (the shared SRAM regions between data movement and compute).
2. **Creating data movement kernels** (reader and writer) via `tt_metal::CreateKernel()` with a `DataMovementConfig`.
3. **Building a defines map** that configures which LLK functions the compute kernel will use.
4. **Creating the compute kernel** via `tt_metal::CreateKernel()` with a `ComputeConfig`.
5. **Setting runtime arguments** via `tt_metal::SetRuntimeArgs()` for per-core data like tensor addresses and offsets.

## The `CreateKernel()` Call

The critical bridge between host and device is `tt_metal::CreateKernel()`. For compute kernels, it takes four arguments:

```cpp
tt_metal::CreateKernel(
    program,          // The tt_metal::Program being constructed
    kernel_path,      // Path to the device-side .cpp file
    core_range,       // Which cores to run this kernel on
    ComputeConfig{    // Configuration struct
        .math_fidelity = ...,
        .fp32_dest_acc_en = ...,
        .math_approx_mode = ...,
        .compile_args = ...,
        .defines = ...,
        .named_compile_args = ...
    }
);
```

A real example from `element_wise_multi_core_program_factory.cpp` (line 175):

```cpp
auto eltwise_binary_kernel_id = tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
    all_device_cores,
    tt_metal::ComputeConfig{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .defines = eltwise_defines
    });
```

And from `matmul_multicore_reuse_mcast_2d_program_factory.cpp` (line 848):

```cpp
tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
    "bmm_large_block_zm_fused_bias_activation.cpp",
    all_cores_with_work,
    tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_kernel_args,
        .defines = mm_kernel_defines,
        .named_compile_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in1", tt::CBIndex::c_1},
            {"cb_bias", tt::CBIndex::c_3},
            {"cb_out", tt::CBIndex::c_4},
            {"cb_intermed0", tt::CBIndex::c_5},
            {"cb_in0_intermediate", tt::CBIndex::c_8},
            {"cb_in1_intermediate", tt::CBIndex::c_9},
            {"cb_in0_transposed", tt::CBIndex::c_10},
        }});
```

## The `ComputeConfig` Struct

Defined in `tt_metal/api/tt-metalium/kernel_types.hpp` (line 76):

```cpp
struct ComputeConfig {
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    KernelBuildOptLevel opt_level = KernelBuildOptLevel::O3;
};
```

Each field has a specific role in shaping LLK behavior:

| Field | Effect on LLK |
|-------|---------------|
| `math_fidelity` | Controls FPU precision (HiFi4, HiFi3, HiFi2, LoFi). Affects the number of FPACK recombination passes. |
| `fp32_dest_acc_en` | When true, DST accumulator registers use FP32 instead of FP16. Critical for matmul accuracy. |
| `math_approx_mode` | When true, SFPU functions use fast approximations (e.g., for GELU, EXP). |
| `defines` | Preprocessor defines injected into the kernel. Selects LLK function variants. |
| `compile_args` | Integer arguments accessible at compile time via `get_compile_time_arg_val(idx)`. |
| `named_compile_args` | Named integer arguments accessible via `get_named_compile_time_arg_val("name")`. |
| `opt_level` | Compiler optimization level (defaults to O3 for compute kernels). |

## Three Kernels Per Core

Every Tensix core runs three RISC-V processors simultaneously:

- **BRISC** (RISCV_0): Typically runs the reader data movement kernel.
- **NCRISC** (RISCV_1): Typically runs the writer data movement kernel.
- **TRISC** (compute): Runs the compute kernel that invokes LLK functions.

The program factory creates all three kernels. The data movement kernels fill and drain circular buffers. The compute kernel consumes tiles from input CBs, processes them through LLK operations, and produces tiles into output CBs. This decoupled producer-consumer model is what makes the CB-based synchronization (covered in Chapter 3) essential.

---

**Next:** [`defines_as_llk_configuration.md`](./defines_as_llk_configuration.md)
