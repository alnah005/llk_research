# Chapter 7: Test Infrastructure

## Overview

The TT-LLK repository includes a purpose-built test infrastructure for writing, compiling, running, and validating low-level kernels on Tenstorrent hardware. Every test is a pair of files -- a Python file that drives parametrization, stimuli generation, golden computation, compilation, execution, and validation, and a C++ file that contains the kernel code exercising the LLK API. The Python side is built on `pytest`, while the C++ side is compiled by a custom RISC-V toolchain (SFPI) into ELF binaries that run directly on Tensix cores.

This chapter walks through the full test lifecycle: how tests are structured, how to write new ones, how to run and debug them, and how to measure kernel performance.

## Learning Objectives

After reading this chapter, you will be able to:

1. Describe the two-file (Python + C++) anatomy of an LLK test and explain the role of each side.
2. Construct a `TestConfig` object with the correct `formats`, `templates`, `runtimes`, and `variant_stimuli` arguments.
3. Write a C++ kernel file with the required `#ifdef LLK_TRISC_UNPACK` / `LLK_TRISC_MATH` / `LLK_TRISC_PACK` sections.
4. Set up the test environment, invoke `pytest` with the correct flags, and interpret build artifacts.
5. Write a performance test using `PerfConfig`, instrument C++ code with `ZONE_SCOPED` / `PROFILER_SYNC`, and read the resulting performance reports.
6. Use hardware performance counters to collect cycle-level metrics from Tensix cores.

## Subtopics

1. [**Test Architecture Overview**](./test_architecture_overview.md) -- The two-file test model, the compilation pipeline from Python parametrization through SFPI compilation to ELF generation, and the auto-generated `build.h` and `params.h` headers.

2. [**Writing Tests**](./writing_tests.md) -- How to write both the Python test function (parametrization, stimuli, golden, validation) and the C++ kernel file (mandatory includes, globals, three `run_kernel` functions). Covers `TestConfig`, `StimuliConfig`, template vs. runtime parameters, and format-specific tolerances.

3. [**Running and Debugging**](./running_and_debugging.md) -- Environment setup, `pytest` invocation, parallel compilation and execution, custom CLI flags, build artifact locations, L1 memory layouts, and code coverage collection.

4. [**Performance Testing**](./performance_testing.md) -- Writing performance tests with `PerfConfig`, C++ profiling macros, hardware performance counters, derived efficiency metrics, and performance report generation.
