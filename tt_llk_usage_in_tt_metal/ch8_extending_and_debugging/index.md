# Chapter 8: Extending and Debugging

This final chapter provides practical guidance for developers who need to add new operations to the TT-LLK / TT-Metal stack, debug issues that arise during LLK integration, and understand the end-to-end compilation flow from host-side `CreateKernel()` to a binary loaded into device L1 memory.

Chapters 1 through 7 covered the architecture, build system, library structure, API wrappers, Compute API surface, operation-to-kernel mapping, and hardware target selection. This chapter ties those layers together with step-by-step walkthroughs of concrete development tasks.

## Sections

1. [**Adding a New Operation**](./adding_a_new_op.md) -- Step-by-step instructions for adding a new SFPU operation (from the low-level SFPI kernel in `tt-llk` up to the user-facing Compute API in TT-Metal), and an outline of the equivalent flow for non-SFPU operations.

2. [**Debugging LLK Issues**](./debugging_llk_issues.md) -- How to enable `LLK_ASSERT` runtime assertions, how to inspect the generated files (`chlkc_descriptors.h`, `chlkc_math.cpp`, `defines_generated.h`), how to use the `fake_kernels_target` CMake target to catch compile errors without hardware, and key preprocessor defines to verify.

3. [**Compilation Flow End to End**](./compilation_flow_end_to_end.md) -- The complete path from a host-side `CreateKernel()` call with a `ComputeConfig` through JIT code generation, GCC cross-compilation with architecture-specific include paths, and binary loading onto device TRISC processors.
