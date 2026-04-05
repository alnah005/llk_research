# Chapter 8: Kernel Development Patterns

## Overview

This final chapter brings together the concepts from the preceding seven chapters into practical guidance for writing LLK kernels. It covers the naming conventions and API patterns that every kernel must follow, the compile-time and runtime parameter systems, the MOP (Macro Operation Pipe) programming model, and the LLTT instruction-replay mechanism. Two complete kernel walkthroughs -- element-wise binary and matrix multiply -- demonstrate how the unpack/math/pack pipeline, address modifiers, fidelity phases, and test parametrization all connect in real code.

## Learning Objectives

After reading this chapter, you will be able to:

1. Recognize and apply the `_llk_<stage>_<operation>_<variant>_<suffix>_()` naming convention and explain how template parameters, runtime parameters, and global state variables interact.
2. Program MOP loops using `ckernel_template` and `ckernel_unpack_template`, configure address modifiers via `addr_mod_t`, and understand when LLTT replay buffers are used instead of direct MOP loops.
3. Trace the complete unpack/math/pack flow of `eltwise_binary_test.cpp` and explain how the Python test parametrizes formats, broadcast types, tile dimensions, and fidelity.
4. Describe the matrix multiply kernel structure, including inner-loop tiling across `CT_DIM`/`RT_DIM`/`KT_DIM`, fidelity phases, throttle levels, and register reuse strategies.

## Subtopics

1. [**API Conventions**](./api_conventions.md) -- Naming patterns, template and runtime parameter conventions, global state variables, the `ckernel` namespace, MOP programming with `ckernel_template`, LLTT instruction replay, and the `LLK_ASSERT` system.

2. [**Element-wise Binary Walkthrough**](./eltwise_binary_walkthrough.md) -- A line-by-line walkthrough of `tests/sources/eltwise_binary_test.cpp`, covering the unpack, math, and pack sections, MOP configuration for add/sub/mul, broadcast handling, and the Python test harness that parametrizes this kernel.

3. [**Matrix Multiply Walkthrough**](./matmul_walkthrough.md) -- The structure of `tests/sources/matmul_test.cpp`, inner-loop tiling, fidelity phases via `matmul_configure_addrmod`, throttle levels, LLTT replay buffer usage, and register reuse optimization.
