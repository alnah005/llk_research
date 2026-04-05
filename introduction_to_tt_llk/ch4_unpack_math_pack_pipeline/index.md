# Chapter 4: The Unpack-Math-Pack Pipeline

## Overview

Every compute operation on a Tensix core follows a three-stage pipeline: **unpack**, **math**, and **pack**. Each stage runs on a dedicated RISC-V processor (TRISC0, TRISC1, TRISC2 respectively), and the stages execute concurrently on different tiles. This chapter explains how data flows through the pipeline, the LLK API surface for each stage, and the synchronization mechanisms that keep the three threads coordinated.

The pipeline operates as follows:

```
L1 Memory
    |
    v
[ UNPACK ]  (TRISC0)  -- DMA from L1 to Source A / Source B registers, with format conversion
    |
    v
[  MATH  ]  (TRISC1)  -- FPU/SFPU computation on register data, result to Destination register
    |
    v
[  PACK  ]  (TRISC2)  -- DMA from Destination register to L1, with format conversion
    |
    v
L1 Memory
```

## Learning Objectives

After reading this chapter, you will be able to:

1. Describe the three pipeline stages and identify which TRISC thread controls each.
2. Explain how unpackers move tile data from L1 into Source A and Source B registers, including format conversion.
3. List the key LLK unpack API headers and describe when each variant is used.
4. Describe how the FPU executes matrix multiply, element-wise, and reduction operations on source register data.
5. List the key LLK math API headers and explain the `addr_mod_t` and `ckernel_template` configuration mechanisms.
6. Explain how the packer moves results from the Destination register back to L1.
7. Describe the `DstSync` modes (`SyncHalf` and `SyncFull`) and how destination register double buffering works.
8. Trace the semaphore-based synchronization between the three TRISC threads.

## Subtopics

1. [**Unpack Operations**](./unpack_operations.md) -- How unpackers DMA tile data from L1 into source registers, the two unpacker channels, format conversion, context switching, and the LLK unpack API headers.

2. [**Math Operations**](./math_operations.md) -- FPU and SFPU computation on source register data, the four FPU cell operations, address modifier configuration, MOP programming, and the LLK math API headers.

3. [**Pack Operations**](./pack_operations.md) -- How the packer DMAs results from the Destination register back to L1, format conversion during pack, ReLU integration, and the LLK pack API headers.

4. [**Synchronization**](./synchronization.md) -- `DstSync` modes, destination register double buffering, semaphore-based coordination between unpack/math/pack, stall instructions, and the `ckernel_template` class.

## Key Terms Introduced

| Term | Definition |
|:-----|:-----------|
| Unpacker | Hardware DMA engine that transfers tile data from L1 into Source A or Source B registers. |
| Source A / Source B | Register files that hold input data for math operations. |
| Destination Register | Register file that holds output data from math operations and input data for the packer. |
| FPU | Floating Point Unit; executes matrix multiply and element-wise operations on 16x16 face data. |
| SFPU | Scalar Floating Point Unit; executes transcendental and other complex unary operations. |
| Packer | Hardware DMA engine that transfers data from the Destination register to L1. |
| MOP | Micro-Op Program; a small loop programmed into hardware registers that executes unpack/math/pack instruction sequences. |
| `DstSync` | Enum controlling whether destination register operates in double-buffered (`SyncHalf`) or single-buffered (`SyncFull`) mode. |
| Context Switching | The unpacker's mechanism for double-buffering configuration state, allowing TRISC0 to program the next tile while the current tile is being unpacked. |

---

**Next:** [`unpack_operations.md`](./unpack_operations.md)
