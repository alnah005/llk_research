# Chapter 6: Op-to-Kernel Mapping

This chapter explains how TT-Metal's host-side operations (in TTNN) configure, compile, and dispatch compute kernels that ultimately invoke LLK functions on device. It covers the full path from a high-level TTNN op down to the preprocessor defines and compile-time arguments that parameterize LLK template instantiations.

## Sections

1. [**Host-Side Op Structure**](./host_side_op_structure.md) -- How TTNN operations are organized on disk, with factory files that call `CreateKernel()` to bind device-side compute kernels.

2. [**Defines as LLK Configuration**](./defines_as_llk_configuration.md) -- How host-side `defines` maps become preprocessor macros in compute kernels, selecting LLK operation types, fidelity levels, and fused SFPU chains.

3. [**Runtime Args and Compile-Time Args**](./runtime_args_and_compile_time_args.md) -- The two mechanisms for passing parameters to device kernels: compile-time args (baked into the binary) and runtime args (written to L1 per launch).

4. [**Concrete Examples**](./concrete_examples.md) -- Walkthrough of three real operations (eltwise binary, matmul, and reduction) showing the complete host-to-device configuration chain.

## Prerequisites

This chapter assumes familiarity with:
- The LLK API layers covered in Chapters 3 and 4 (tile operations, init functions, CB management).
- The compute kernel anatomy from Chapter 5 (acquire/commit/wait/release cycle, `kernel_main()`).
- The SFPU pipeline from Chapter 4 (how SFPU operations are initialized and invoked).

## Key Insight

The central mechanism connecting host-side ops to device-side LLK usage is the `ComputeConfig` struct (defined in `tt_metal/api/tt-metalium/kernel_types.hpp`). This struct carries `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`, `defines`, `compile_args`, and `named_compile_args` -- all of which shape how LLK templates are instantiated at compile time. The `defines` map is especially powerful: it injects preprocessor macros that select between different LLK function variants (e.g., `add_tiles` vs. `mul_tiles`) without changing the kernel source code.
