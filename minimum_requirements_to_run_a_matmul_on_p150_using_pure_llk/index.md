# Minimum Requirements to Run a MatMul on P150 Using Pure LLK

A comprehensive research guide tracing every layer of the software and hardware stack required to execute a matrix multiplication on a Tenstorrent P150 card using the Low-Level Kernel (LLK) library directly, without TT-Metal or any higher-level framework.

**Source repository:** `tt-llk` (Tenstorrent Low-Level Kernel library)
**Target hardware:** P150 PCIe card (Blackhole architecture)
**Total scope:** 8 chapters, 25 files, ~9,550 lines

---

## Table of Contents

### Chapter 1: P150 and the Blackhole Architecture
*What is a P150, and how does it map to the Blackhole chip?*

- [1.1 P150 Chip Identity](./ch1_p150_and_blackhole/01_p150_chip_identity.md) -- PCIe card to silicon mapping, device detection, architecture selection
- [1.2 Tensix Core and Dataflow](./ch1_p150_and_blackhole/02_tensix_core_and_dataflow.md) -- 5 RISC-V cores, 3-phase coprocessor pipeline, MVMUL instruction
- [1.3 LLK Software Position](./ch1_p150_and_blackhole/03_llk_software_position.md) -- Where LLK sits in the stack, header-only C++17 library

### Chapter 2: Data Formats, Tiles, and Memory
*How data is shaped and stored for the Tensix compute engine.*

- [2.1 Tile Geometry and Faces](./ch2_data_formats_and_memory/01_tile_geometry_and_faces.md) -- 32x32 tiles, 16x16 faces, face layout for MVMUL
- [2.2 Data Format Pipeline](./ch2_data_formats_and_memory/02_data_format_pipeline.md) -- Float16, Float16_b, Float32, Bfp8_b; unpack/math/pack format conversions
- [2.3 L1 Memory Layout and Addressing](./ch2_data_formats_and_memory/03_l1_memory_layout_and_addressing.md) -- SrcA/SrcB/Dest register files, L1 SRAM allocation

### Chapter 3: Kernel Architecture
*How a single C++ source becomes three cooperating firmware images.*

- [3.1 Single Source, Three Kernels](./ch3_kernel_architecture/01_single_source_three_kernels.md) -- Three-compilation pattern, `#ifdef LLK_TRISC_*` guards, `run_kernel()`
- [3.2 Boot Sequence and Firmware](./ch3_kernel_architecture/02_boot_sequence_and_firmware.md) -- BRISC command loop, `do_crt0()`, `device_setup()`, TRISC startup
- [3.3 Inter-Thread Synchronization](./ch3_kernel_architecture/03_inter_thread_synchronization.md) -- Hardware semaphores, `DstSync::SyncHalf`, acquire/release protocol

### Chapter 4: MatMul LLK API Call Sequences
*The exact sequence of LLK function calls for each TRISC thread during matmul.*

- [4.1 Unpack Thread API Sequence](./ch4_matmul_api_calls/01_unpack_thread_calls.md) -- `_llk_unpack_AB_matmul_init_`, tile loading, format configuration
- [4.2 Math Thread API Sequence](./ch4_matmul_api_calls/02_math_thread_calls.md) -- `_llk_math_matmul_init_`, MOP programming, fidelity phases, MVMUL replay
- [4.3 Pack Thread API Sequence](./ch4_matmul_api_calls/03_pack_thread_calls.md) -- `_llk_pack_init_`, dest-to-L1 packing, format conversion
- [4.4 Minimum Call Summary](./ch4_matmul_api_calls/04_minimum_call_summary.md) -- The irreducible set of 15 LLK API calls across 3 threads

### Chapter 5: Compile-Time Configuration
*Preprocessor defines, `build.h` generation, and the C++ structs that configure the matmul kernel.*

- [5.1 Required Preprocessor Defines](./ch5_compile_time_config/01_required_preprocessor_defines.md) -- Master reference table of all `-D` flags
- [5.2 build.h Generation and Constexpr](./ch5_compile_time_config/02_build_h_generation_and_constexpr.md) -- Python-to-C++ bridge, 8-phase header generation, FormatConfig, RuntimeParams
- [5.3 Matmul-Specific Config Structs](./ch5_compile_time_config/03_matmul_specific_config_structs.md) -- `addr_mod_t`, `ckernel_template`, MathFidelity, MOP programming

### Chapter 6: Compilation, Linking, and ELF Generation
*From source code to loadable ELF binaries.*

- [6.1 Toolchain and Dependencies](./ch6_compilation/01_toolchain_and_dependencies.md) -- SFPI cross-compiler (`riscv-tt-elf-g++`), ISA extensions, system dependencies
- [6.2 Compilation Commands and Flags](./ch6_compilation/02_compilation_commands_and_flags.md) -- `build_kernel_part()` trace, reconstructed compiler command, BRISC vs TRISC
- [6.3 Linker Scripts and Memory Regions](./ch6_compilation/03_linker_scripts_and_memory_regions.md) -- `memory.blackhole.ld`, `sections.ld`, L0/L1 layout, `tmu-crt0.S`

### Chapter 7: Host-Side Execution Workflow
*How the Python test harness drives the hardware from detection to result readback.*

- [7.1 Host-Device Communication](./ch7_host_execution/01_host_device_communication.md) -- tt-exalens API, ARC messages, soft-reset register control, coordinate system
- [7.2 Step-by-Step Execution](./ch7_host_execution/02_step_by_step_execution.md) -- 10-step workflow, BRISC command protocol, device_setup(), mailbox polling
- [7.3 What TT-Metal Replaces](./ch7_host_execution/03_what_tt_metal_replaces.md) -- Division of labor, minimum requirements checklist, advantages of pure LLK

### Chapter 8: Test Infrastructure and Alternatives
*The complete test suite and whether LLK-free matmul on P150 is possible.*

- [8.1 Matmul Test Survey](./ch8_test_infrastructure/01_matmul_test_survey.md) -- All 10 test variants, master reference table, experimental headers
- [8.2 Python Test Harness](./ch8_test_infrastructure/02_python_test_harness.md) -- test_matmul.py walkthrough, format inference, build caching, golden comparison
- [8.3 Alternatives to LLK](./ch8_test_infrastructure/03_alternatives_to_llk.md) -- 9 alternative approaches analyzed; conclusion: no LLK-free hardware matmul path exists

---

## Chapter Line Counts

| Chapter | Topic | Lines |
|---------|-------|------:|
| Ch1 | P150 and Blackhole Architecture | 487 |
| Ch2 | Data Formats, Tiles, and Memory | 977 |
| Ch3 | Kernel Architecture | 1,454 |
| Ch4 | MatMul LLK API Sequences | 1,761 |
| Ch5 | Compile-Time Configuration | 1,225 |
| Ch6 | Compilation, Linking, ELF Generation | 1,201 |
| Ch7 | Host-Side Execution Workflow | 1,452 |
| Ch8 | Test Infrastructure and Alternatives | 993 |
| **Total** | | **9,550** |

## Quality Control

Every chapter was produced using a multi-architect pipeline:
1. **4 independent generators** produced competing versions
2. **4 independent evaluators** ranked and critiqued each version
3. **1 synthesizer** merged the best content from all versions
4. **Agent B (factual critic)** verified every technical claim against the tt-llk source code
5. **Agent C (compressor)** identified and removed cross-chapter duplication

All chapters have been factually approved by Agent B with corrections applied.

## Key Conclusions

1. **Minimum LLK API surface**: 15 function calls across 3 TRISC threads (5 unpack + 4 math + 6 pack) constitute the irreducible kernel-side requirement.

2. **Minimum host-side requirements**: 12 operations from architecture detection through result readback, all performed via the tt-exalens Python library.

3. **No LLK-free path exists**: Every software path to the Tensix FPU's MVMUL instruction passes through LLK. The hardware requires specific unpacker/packer/FPU configuration that LLK encapsulates; reproducing this from scratch would mean reimplementing LLK.

4. **What "pure LLK" means**: Approximately 200 lines of Python infrastructure code replace what TT-Metal provides, trading automation for direct register-level control and cycle-accurate inspection.
