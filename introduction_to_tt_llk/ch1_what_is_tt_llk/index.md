# Chapter 1: What is TT-LLK?

## Overview

This chapter introduces TT-LLK (Tenstorrent Low-Level Kernels), the foundational software layer for programming Tenstorrent AI chips. You will learn what TT-LLK is, where it sits in the Tenstorrent software stack, and how the repository is organized.

## Learning Objectives

After reading this chapter, you will be able to:

1. Explain what TT-LLK is and the role it plays as the first software layer above Tensix hardware.
2. Describe how TT-LLK relates to TT-Metalium and TT-Forge in the Tenstorrent software stack.
3. Navigate the repository structure, identify per-architecture directories, and understand the naming conventions used for header files.
4. Distinguish the three hardware targets (Wormhole B0, Blackhole, Quasar) and explain why each has its own implementation directory.

## Subtopics

1. [**Software Stack Position**](./software_stack_position.md) -- TT-LLK as a header-only C++17 library, its relationship to TT-Metalium and TT-Forge, and how it encapsulates the Tensix ISA into callable C++ functions.

2. [**Repository Structure**](./repository_structure.md) -- Top-level layout, per-architecture directory structure, naming conventions, shared headers, and the three hardware targets.

## Key Terms Introduced

| Term | Definition |
|:-----|:-----------|
| **Datum** | A single data element within a tensor. |
| **Tile** | A 32x32 block of datums, the fundamental unit of LLK computation. |
| **Tensix Engine** | The coprocessor controlled by TRISC processors, optimized for matrix computations. |
| **TRISC0/1/2** | The three RISC-V processors that control unpack, math, and pack respectively. |
| **FPU** | Floating Point Unit, the primary matrix computation engine within the Tensix Engine. |
| **SFPU** | Scalar Floating Point Unit, a SIMD engine for specialized operations beyond FPU capabilities. |
