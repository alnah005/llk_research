# Chapter 1: How Hardware Debuggers Work -- cuda-gdb and the Reference Architecture

This chapter establishes the conceptual vocabulary and reference architecture for hardware kernel debuggers. By analyzing how cuda-gdb, rocgdb, Intel GT Debugger, and Arm DS attach to accelerator cores, we extract the common patterns that any Tenstorrent debugger must replicate, adapt, or deliberately replace. The central argument is that a **capture-and-replay** paradigm -- serializing kernel state for offline deterministic debugging -- offers a more practical path for Tenstorrent than the traditional live-attach model that GPU debuggers rely on.

## Prerequisites

No prior experience with cuda-gdb internals, RISC-V Debug Specification (v0.13), JTAG debugging protocols, or GDB Remote Serial Protocol is assumed. Familiarity with basic GDB concepts (breakpoints, watchpoints, single-stepping) and the Tensix five-RISC architecture (BRISC, NCRISC, TRISC0/1/2, L1 SRAM, circular buffers) is expected. See the guide-level audience description for the full prerequisites list.

## Chapter Contents

1. [GPU Debugger Architectures](./01_gpu_debugger_architectures.md)
   - The layered architecture of cuda-gdb: frontend, debug API, driver interface, and SM hardware
   - Core cuda-gdb operations and how they map to SM debug hardware
   - The "focus" model for selecting GPU, SM, warp, and lane
   - Comparative analysis: rocgdb, Intel GT Debugger, Arm DS
   - Common patterns extracted across all hardware debuggers
   - Which patterns Tenstorrent hardware provides today
   - Where the live-debug model breaks down for Tenstorrent

2. [Lessons and Requirements for Tenstorrent](./02_lessons_and_requirements_for_tenstorrent.md)
   - Translation of GPU debugger capabilities into Tenstorrent-specific requirements
   - The key insight: all hardware debuggers require either debug circuitry or instrumented traps
   - Definition of the "capture and replay" alternative and the capture envelope concept
   - Decision matrix: hardware JTAG vs. firmware-cooperative vs. capture-and-replay
   - Gap analysis summary: cuda-gdb features vs. Tenstorrent capabilities
   - Requirements derived from the analysis (R1-R7)

## Navigation

- **Next Chapter:** [Chapter 2 -- Complete State Inventory for Kernel Capture](../ch2_state_inventory/index.md)
