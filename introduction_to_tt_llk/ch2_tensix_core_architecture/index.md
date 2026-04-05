# Chapter 2: Tensix Core Architecture

## Overview

This chapter explains the hardware architecture that LLKs program. Every Tenstorrent chip is built from a grid of identical Tensix Cores, and understanding a single core's internal structure is essential for writing efficient low-level kernels. You will learn about the four major components of a Tensix Core, the roles of its five RISC-V processors, how the Tensix Engine executes instructions from three concurrent threads, and the data flow path that every tile follows from L1 memory through computation and back.

## Learning Objectives

After reading this chapter, you will be able to:

1. Identify the four major components of a Tensix Core and explain each one's role.
2. Distinguish the five RISC-V processors by name and describe which ones are relevant to LLK development.
3. Explain the three-way threaded architecture of the Tensix Engine, including its per-thread frontends and shared backend execution units.
4. Trace the data flow of a tile through the core compute pipeline: L1 to source registers, through the FPU or SFPU, to the destination register, and back to L1.
5. Describe the properties of the Source A, Source B, and Destination register files.

## Subtopics

1. [**Core Components**](./core_components.md) -- The four major parts of a Tensix Core: L1 SRAM, two NoC routers, five RISC-V processors, and the Tensix Engine.

2. [**RISC-V Processors**](./riscv_processors.md) -- The five 32-bit RISC-V cores (BRISC, NCRISC, TRISC0, TRISC1, TRISC2) and their responsibilities.

3. [**Tensix Engine**](./tensix_engine.md) -- The three-way threaded coprocessor with independent frontend pipelines and shared backend execution units.

4. [**Data Flow**](./data_flow.md) -- The compute pipeline from L1 through unpackers, register files, FPU/SFPU, packers, and back to L1.

## Key Terms Introduced

| Term | Definition |
|:-----|:-----------|
| **L1 SRAM** | Local memory within each Tensix Core that stores input/output tensors and program code for all RISC-V processors. |
| **NoC Router** | Network on Chip router that manages data movement between Tensix Cores and DRAM. |
| **BRISC** | Board RISC-V processor, handles NOC communication and board setup. |
| **NCRISC** | NOC RISC-V processor, handles NOC communication. |
| **Tensix ISA** | The custom instruction set executed by the Tensix Engine, distinct from the RISC-V ISA used by the host cores. |
| **Source A / Source B** | Double-buffered source operand registers that hold unpacked tile data for the FPU. |
| **Destination Register (Dest)** | Output register for FPU results and input/output for SFPU, configurable to hold 4--16 tiles. |
