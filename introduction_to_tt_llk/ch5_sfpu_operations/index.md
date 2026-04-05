# Chapter 5: SFPU Operations

## Overview

The **Scalar Floating Point Unit (SFPU)** is a programmable SIMD engine embedded within the Tensix FPU, controlled by the TRISC1 math thread. While the FPU handles matrix multiplies and element-wise operations at 19-bit precision through fixed microcode sequences, the SFPU provides a fully programmable path for arbitrary element-wise computations at full 32-bit floating-point precision.

The SFPU is essential for implementing activation functions (GELU, sigmoid, tanh), transcendental math (exp, log, trigonometry), comparison operations, integer arithmetic, and many other operations that do not map to the FPU's fixed-function datapath. Nearly every non-matmul compute kernel uses the SFPU in some capacity.

## Learning Objectives

After reading this chapter, you will be able to:

1. Describe the SFPU's load/store architecture and its relationship to the Destination register.
2. Explain how `VectorMode` controls which faces of a tile are processed by an SFPU operation.
3. Trace the face iteration pattern in `_llk_math_eltwise_unary_sfpu_params_()` and the role of `SETRWC` instructions.
4. Identify the major categories of SFPU operations from the `SfpuType` enum.
5. Locate the implementation header for any given SFPU operation in `common/inc/sfpu/ckernel_sfpu_*.h`.
6. Distinguish between unary, binary, and ternary SFPU operation interfaces and their respective LLK headers.
7. Explain the purpose of Welford's online algorithm implementation for numerically stable variance computation.

## Subtopics

1. [**SFPU Architecture**](./sfpu_architecture.md) -- The SFPU's role within the FPU, its load/store programming model, 32-bit precision, input requirements via the Destination register, `VectorMode` control, and the face iteration pattern.

2. [**SFPU Operations Catalog**](./sfpu_operations_catalog.md) -- A comprehensive listing of all SFPU operations organized by category: activation functions, exponential/logarithmic, power/root, trigonometric, comparison, special math, integer, bitwise, reduction, type conversion, and miscellaneous.

3. [**Binary and Multi-Operand SFPU Operations**](./sfpu_binary_operations.md) -- Binary SFPU operations (two-operand computations distinct from FPU element-wise), ternary SFPU operations, the `BinaryOp` enum, and Welford's online algorithm for variance.
