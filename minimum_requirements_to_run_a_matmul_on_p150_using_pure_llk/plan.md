# Final Plan: Minimum Requirements to Run a MatMul on P150 Using Pure LLK

---

## Audience

This guide is written for **firmware engineers, kernel developers, and low-level systems programmers** who want to execute a matrix multiplication on a Tenstorrent P150 card using only the LLK (Low-Level Kernel) header-only library, bypassing TT-Metal's runtime abstractions entirely.

**Prerequisites the reader should have:**
- Working knowledge of C++ (C++17), including templates, constexpr, and preprocessor macros.
- Familiarity with embedded or bare-metal programming concepts: linker scripts, cross-compilation, ELF binaries, memory-mapped I/O.
- A conceptual understanding of matrix multiplication at the tile/block level.
- General awareness of RISC-V architecture (not expert-level).
- Access to a P150 card (Blackhole architecture) or the tt-exalens simulator environment.

**No prior experience with Tenstorrent hardware, the Tensix ISA, the LLK API, or TT-Metal is assumed.** The guide builds that knowledge from scratch.

---

## Chapter List

### Chapter 1: P150 Hardware Identity and the Blackhole Architecture

**Description:** Establishes which chip architecture the P150 corresponds to in the LLK codebase, introduces the Tensix core model, and positions LLK in the software stack.

**Directory:** `ch1_p150_and_blackhole/`

**Files:**

- `01_p150_chip_identity.md`
  - Identifies the P150 as a Blackhole-based product and maps it to `tt_llk_blackhole/` in the codebase.
  - Explains the `ARCH_BLACKHOLE` preprocessor define and its role in compile-time architecture selection.
  - Distinguishes Blackhole from Wormhole B0 and Quasar architectures (the three variants in tt-llk), including key Blackhole-specific differences: `-mcpu=tt-bh` / `-mcpu=tt-bh-tensix` compiler flags, clock speed (1350 cycles per microsecond vs. 1000 for Wormhole), destination clock gating control register, extra parameters in `ZEROACC` instruction, and pack API differences (`_llk_pack_hw_configure_` taking a third `tilize` template parameter).
  - References: `tests/python_tests/helpers/chip_architecture.py` (ChipArchitecture enum), `tests/python_tests/helpers/test_config.py` (`setup_arch()` method), `setup_testing_env.sh`.
  - Lists the LLK source tree structure for Blackhole: `tt_llk_blackhole/llk_lib/` (API headers), `tt_llk_blackhole/common/inc/` (ckernel infrastructure).

- `02_tensix_core_and_dataflow.md`
  - Describes the Tensix Core structure: five RISC-V cores (BRISC, NCRISC, TRISC0/1/2), the Tensix Engine as a multi-threaded coprocessor with three concurrent threads (T0=Unpack, T1=Math, T2=Pack), L1 SRAM, and routers.
  - Explains the shared backend execution units: Sync Unit, Unpackers, Matrix Unit/FPU, Packers, Vector Unit/SFPU, Configuration Unit.
  - Traces the specific data path for matrix multiplication: L1 -> Unpacker -> Source Registers (SrcA, SrcB) -> FPU (MVMUL instruction) -> Destination Register -> Packer -> L1.
  - Describes double buffering of Source registers and its role in sustaining throughput.
  - References: `CLAUDE.md`, `docs/llk/l2/top_level_overview.md` (Figures 1-3).

- `03_llk_software_position.md`
  - Positions LLK as the first software layer above hardware, below TT-Metalium and TT-Forge.
  - Explains that TT-LLK is a header-only library -- it produces no standalone binary, only C++ headers consumed by TRISC firmware.
  - Clarifies the distinction between "pure LLK" (using the TT-LLK test infrastructure to compile and dispatch kernels without TT-Metal) versus using LLK headers within TT-Metal's runtime.
  - References: `docs/llk/l1/intro.md`, `README.md`.

---

### Chapter 2: Tile Formats, Data Formats, and Memory Layout

**Description:** Covers the data organization, tile geometry, numerical formats, format conversion pipeline, and L1 memory map that must be understood before any LLK matmul API calls make sense.

**Directory:** `ch2_data_formats_and_memory/`

**Files:**

- `01_tile_geometry_and_faces.md`
  - Defines the canonical 32x32 tile composed of four 16x16 faces (F0-F3) stored in row-major order within each face.
  - Documents constants: `TILE_R_DIM` (32), `TILE_C_DIM` (32), `FACE_R_DIM` (16), `FACE_C_DIM` (16) from `ckernel_defs.h`.
  - Covers non-standard tile dimensions supported for matmul (16x32, 32x16) via `num_faces` parameter (1, 2, or 4) and `partial_face` mode.
  - Notes the assertion: "16x16 by 16x16 matmul is not supported" in both math and unpack init functions.
  - Explains how matmul operates at the face level: `D[8,16] = B[8,16] * A[16,16]` inner loop, iterated across faces and tiles.
  - Defines block dimensions: `ct_dim` (column-tile dimension), `rt_dim` (row-tile dimension), `kt_dim` (inner/accumulation dimension in tiles).

- `02_data_format_pipeline.md`
  - Enumerates supported data formats from the `DataFormat` enum in `tensix_types.h`: Float16, Float16_b, Float32, Bfp8_b, Bfp4_b, Int8, Tf32, Fp8_e4m3, etc.
  - Documents tile sizes per data format: Float16/Float16_b = 128 words/tile (2048 bytes header+data), Bfp8_b = 68 words/tile (with shared exponents), Float32 = 256 words/tile.
  - Traces the format conversion pipeline: L1 source format -> unpacker gasket conversion -> Source register format (limited precision, up to 19-bit) -> FPU computation -> Destination register format (configurable 16-bit or 32-bit) -> packer gasket conversion -> L1 destination format.
  - Documents the `FormatConfig` struct with its 11 fields (`unpack_A_src`, `unpack_B_src`, `unpack_A_dst`, `unpack_B_dst`, `math`, `pack_src`, `pack_dst`, etc.) and how each maps to a pipeline stage.
  - Discusses math fidelity (`MathFidelity::LoFi`, `HiFi2`, `HiFi3`, `HiFi4`): LoFi = 1 phase (fastest), HiFi4 = 4 phases (most precise), and the fidelity_increment mechanism.
  - Explains destination register accumulator modes: 16-bit (up to 16 tiles in SyncFull, 8 in SyncHalf) vs. 32-bit FP32 accumulation (`is_fp32_dest_acc_en`, halving capacity to 8/4 tiles), and the `DstSync::SyncHalf` / `SyncFull` synchronization modes.

- `03_l1_memory_layout_and_addressing.md`
  - Presents the complete Blackhole L1 memory layout from the linker script:
    - 0x00000-0x03FFF: BRISC code (16K)
    - 0x04000-0x04FFF: BRISC C-runtime data
    - 0x05000-0x08FFF: TRISC0 (Unpack) code (16K)
    - 0x09000-0x097FF: TRISC0 C-runtime data
    - 0x09800-0x0D7FF: TRISC1 (Math) code (16K)
    - 0x0D800-0x0DFFF: TRISC1 C-runtime data
    - 0x0E000-0x10FFF: TRISC2 (Pack) code (16K)
    - 0x11000-0x117FF: TRISC2 C-runtime data
    - 0x20000: Runtime arguments struct (1K)
    - 0x21000+: Stimuli/data space (~1.3 MB)
  - Explains LOCAL_MEM_BASE (0xFFB00000) for per-core private data memory (3584 bytes data + 512 bytes stack on Blackhole).
  - Details the L1 address transformation: `L1_ADDRESS(physical_addr) = (physical_addr >> 4) - 1`, the 16-byte granularity requirement, and the off-by-one hardware convention (from `llk_memory_checks.h`).
  - Explains the `Operand` class (from `tests/helpers/include/operand.h`) for address arithmetic: `base_addr + index * tile_size`.
  - Covers the `StimuliConfig` class that computes L1 addresses for buffer_A, buffer_B, and buffer_Res starting from 0x21000.
  - Explains matmul operand mapping: in0/inA goes to SrcB, in1/inB goes to SrcA (the hardware convention is D = B * A, not A * B).
  - References: `tests/helpers/ld/memory.blackhole.ld`, `docs/tests/infra_architecture.md`, `tests/helpers/include/operand.h`, `tests/python_tests/helpers/stimuli_config.py`.

---

### Chapter 3: The Three-Thread Kernel Architecture

**Description:** Explains how a single C++ source file is compiled three times (once per TRISC thread), covers the required source file structure, the BRISC/TRISC boot sequence, and all inter-thread synchronization mechanisms.

**Directory:** `ch3_kernel_architecture/`

**Files:**

- `01_single_source_three_kernels.md`
  - Shows the pattern from `matmul_test.cpp`: one `.cpp` file with three `#ifdef` blocks (`LLK_TRISC_UNPACK`, `LLK_TRISC_MATH`, `LLK_TRISC_PACK`), each defining a `void run_kernel(RUNTIME_PARAMETERS params)` function.
  - Explains that the SFPI compiler compiles this file three times with different defines, producing `unpack.elf`, `math.elf`, and `pack.elf`.
  - Documents mandatory global variables: `unp_cfg_context`, `pack_sync_tile_dst_ptr`, `math_sync_tile_dst_index`.
  - Documents mandatory header includes per section: `ckernel.h`, `llk_defs.h`, `params.h`, and the thread-specific LLK headers.
  - Explains the `RUNTIME_PARAMETERS` macro and `struct RuntimeParams` -- how Python-side parameters become C++ struct fields.
  - Describes the auto-generated `build.h` header: compile-time constants (`is_fp32_dest_acc_en`, `MATH_FIDELITY`), `FormatConfig` struct, and `RuntimeParams` struct definition.
  - References: `tests/sources/matmul_test.cpp`, `docs/tests/getting_started.md`.

- `02_boot_sequence_and_firmware.md`
  - Traces the boot path: `_start()` -> `do_crt0()` (sets global pointer, stack pointer, initializes `.bss`, copies `.loader_init` to `.ldm_data`, runs global constructors) -> `main()`.
  - Documents TRISC `main()`: copies runtime parameters from L1 (`__runtime_args_start`), zeros register file, resets `cfg_state_id` and `dest_offset_id`, then calls `run_kernel(temp_args)`.
  - In BRISC boot mode (default for Blackhole): BRISC runs first, enters polling loop, receives `START_TRISCS` command, executes `device_setup()`, then clears TRISC soft reset.
  - Explains `device_setup()`: `TTI_ZEROACC` to clear all accumulators, `TTI_SFPENCC` to enable CC stack, `TTI_SFPCONFIG` to initialize SFPU constant registers, semaphore initialization via `t6_semaphore_init()`.
  - Explains the mailbox signaling protocol: each TRISC writes `KERNEL_COMPLETE` (0xFF) to its designated mailbox address upon `run_kernel()` return.
  - References: `tests/helpers/include/boot.h`, `tests/helpers/src/brisc.cpp`, `tests/helpers/src/trisc.cpp`.

- `03_inter_thread_synchronization.md`
  - Documents the three hardware semaphores initialized in `device_setup()`:
    - `UNPACK_TO_DEST` (semaphore 2): coordinates unpack and math access to Source registers. Initial value 0, max value 1.
    - `MATH_DONE` / `MATH_PACK` (semaphore 7): coordinates math completion and pack readiness. Initial value 0, max value 1.
    - `PACK_DONE` (semaphore 4): signals pack completion. Initial value 0, max value 1.
  - Describes the `UNPACK_SYNC` mechanism within the unpack thread: `semaphore_post(semaphore::UNPACK_SYNC)` for context acquire and `t6_semaphore_get(semaphore::UNPACK_SYNC)` for context release. Explains the double-buffered context switching (`unp_cfg_context`) that allows overlapped unpack and math operations.
  - Details the math-pack synchronization: `math_dest_wait()` (math waits for pack to release destination half), `set_math_semaphores()` (math signals pack), `dest_section_flip()` (alternates destination halves in SyncHalf mode).
  - Details the pack-side pattern: `_llk_packer_wait_for_math_done_()` blocks until math signals, then pack tiles, then `_llk_pack_dest_section_done_()` clears dest half, decrements semaphore, flips packer dest offset.
  - Covers `tensix_sync()` -- the full pipeline drain barrier used during initialization and cleanup.
  - Discusses the `TTI_STALLWAIT` instruction variants: `STALL_CFG`/`STALL_UNPACK`/`STALL_MATH` with `THCON`/`TRISC_CFG`/`UNPACK0` targets, ensuring configuration writes complete before operations begin.
  - References: `tests/helpers/include/boot.h`, `tt_llk_blackhole/llk_lib/llk_math_common.h`, `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`, `tt_llk_blackhole/llk_lib/llk_pack_common.h`.

---

### Chapter 4: Minimal LLK API Call Sequences for MatMul

**Description:** Presents the exact, ordered sequence of LLK API calls required on each of the three TRISC threads to execute a tiled matrix multiplication, with parameter-by-parameter explanation.

**Directory:** `ch4_matmul_api_calls/`

**Files:**

- `01_unpack_thread_calls.md`
  - Complete API sequence for TRISC0 (Unpack thread):
    1. `_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(...)` -- configures unpacker hardware with source/destination formats, face dimensions, number of faces, tile sizes.
    2. `_llk_unpack_AB_matmul_init_<>(transpose, ct_dim, rt_dim, kt_dim, face_r_dim_A, face_r_dim_B, num_faces_A, num_faces_B, partial_face_A, partial_face_B)` -- programs the unpack MOP (Micro-Operation Program) via replay buffer, configures haloize mode for transpose, sets address counters, stores kt_dim in GPR.
    3. Loop over kt_dim: `_llk_unpack_AB_matmul_<>(base_addr_a, base_addr_b, tile_idx_a, tile_idx_b, tile_size_a, tile_size_b, ...)` -- waits for context, configures addresses, posts semaphore, triggers unpacker, runs MOP, gets semaphore, switches context.
  - Explains the mapping: in0/inA -> SrcB (via SEC1), in1/inB -> SrcA (via SEC0).
  - Documents the double-buffered context mechanism (unp_cfg_context flipping between 0 and 1).
  - References: `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`, `tests/sources/matmul_test.cpp` (UNPACK block).

- `02_math_thread_calls.md`
  - Complete API sequence for TRISC1 (Math thread):
    1. `_llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(tile_r_dim, tile_c_dim, ..., transpose, ct_dim, rt_dim)` -- configures address modifier registers and programs the MOP (replay buffer with MVMUL instruction sequences). Explains the face-level addressing: `D[8,16] = B[8,16] * A[16,16]`.
    2. `_llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>()` -- initializes math/pack synchronization (resets dest pointers, initializes MATH_PACK and PACK_DONE semaphores).
    3. `_llk_math_hw_configure_<is_fp32_dest_acc_en>(srca_format, srcb_format)` -- configures FP32 accumulation, INT8 math mode, and ZEROACC behavior.
    4. `_llk_math_wait_for_dest_available_<DstSync::SyncHalf>()` -- waits for destination register half to become available from packer.
    5. Loop over kt_dim: `_llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(dst_index, ct_dim, rt_dim)` -- executes the matrix multiply using `ckernel_template::run()`, handling register reuse (SrcA/SrcB clearing) and fidelity phases.
    6. `_llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>()` -- signals packer that destination data is ready, flips dest section.
  - Explains the reuse_a vs. reuse_b strategy: when ct_dim >= rt_dim, SrcA is reused (reuse_a=true), otherwise SrcB is reused.
  - Explains MathFidelity: LoFi (1 phase), HiFi2 (2 phases), HiFi3 (3 phases), HiFi4 (4 phases) and the inner_loops/fidelity_increment mechanism.
  - References: `tt_llk_blackhole/llk_lib/llk_math_matmul.h`, `tests/sources/matmul_test.cpp` (MATH block).

- `03_pack_thread_calls.md`
  - Complete API sequence for TRISC2 (Pack thread):
    1. `_llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(pack_src_format, pack_dst_format, tile_size, ...)` -- configures packer hardware (Blackhole variant takes three template booleans: is_fp32_dest_acc_en, untilize, tilize).
    2. `_llk_pack_init_<false, false, false>(pack_dst_format, ...)` -- programs pack MOP and address modifiers (Blackhole variant with 3 template params).
    3. `_llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>()` -- initializes pack destination synchronization (resets dest read pointer, initializes PACK_DONE and MATH_PACK semaphores).
    4. `_llk_packer_wait_for_math_done_()` -- blocks until math thread signals completion via MATH_PACK semaphore.
    5. Loop over TILE_CNT: `_llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(tile_index, L1_ADDRESS(buffer_Res[i]))` -- packs tiles from destination register back to L1.
    6. `_llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>()` -- clears destination register section, signals math thread it can write again, flips dest offset.
  - Notes the Blackhole-specific API differences vs. Wormhole (third template parameter).
  - References: `tt_llk_blackhole/llk_lib/llk_pack.h`, `tt_llk_blackhole/llk_lib/llk_pack_common.h`, `tests/sources/matmul_test.cpp` (PACK block).

- `04_minimum_call_summary.md`
  - Summary table: minimum API call count per thread for a 1x1x1 (single-tile) matmul vs. a general RT_DIM x CT_DIM x KT_DIM matmul.
  - Matmul dimension parameter reference: `ct_dim` = number of output tile columns, `rt_dim` = number of output tile rows, `kt_dim` = inner dimension (tiles accumulated), `TILE_CNT` = rt_dim * ct_dim.
  - Quick-reference showing which init calls are one-time vs. which calls are per-iteration.

---

### Chapter 5: Compile-Time Configuration, Preprocessor Defines, and build.h

**Description:** Documents every preprocessor define, compile-time constant, configuration macro, and the build.h generation system needed for a matmul kernel to compile correctly.

**Directory:** `ch5_compile_time_config/`

**Files:**

- `01_required_preprocessor_defines.md`
  - **Architecture define:** `-DARCH_BLACKHOLE` (required for P150/Blackhole path selection throughout LLK headers).
  - **Thread identification:** `-DLLK_TRISC_UNPACK`, `-DLLK_TRISC_MATH`, `-DLLK_TRISC_PACK` -- exactly one must be defined per compilation unit. `-DCOMPILE_FOR_TRISC=0/1/2` for additional thread-index-aware code.
  - **Boot mode:** `-DLLK_BOOT_MODE_BRISC` (for BRISC boot on Blackhole, the default and only supported mode).
  - **Environment defines:** `-DTENSIX_FIRMWARE`, `-DENV_LLK_INFRA`, `-DENABLE_LLK_ASSERT` -- gate firmware mode, LLK infrastructure paths, and assertion checking.
  - **Optional defines:** `RUNTIME_FORMATS` (when format config is runtime rather than compile-time), `LLK_PROFILER`, `COVERAGE`, `SPEED_OF_LIGHT`.
  - References: `tests/python_tests/helpers/test_config.py` (setup_arch method, ARCH_DEFINE).

- `02_build_h_generation_and_constexpr.md`
  - Explains the auto-generated `build.h` file that the test infrastructure produces per variant.
  - Essential constexpr values for matmul:
    - `MATH_FIDELITY` -- a `constexpr MathFidelity` value (LoFi/HiFi2/HiFi3/HiFi4) controlling the number of fidelity phases in the MVMUL instruction replay.
    - `is_fp32_dest_acc_en` -- `constexpr bool` controlling 32-bit vs. 16-bit destination register mode, affects tile capacity.
    - `dest_sync` -- `DstSync::SyncHalf` or `DstSync::SyncFull`.
    - `THROTTLE_LEVEL` -- `constexpr int` (0-5) controlling matmul compute throughput throttling by inserting NOPs between MVMUL instructions.
  - Format configuration: either compile-time (`FORMATS_CONFIG_STRUCT_COMPILETIME` with `FormatConfig` struct) or runtime (`RUNTIME_FORMATS` define with FormatConfig read from L1).
  - The `RuntimeParams` struct definition: `TILE_SIZE_PACK`, `TILE_SIZE_UNPACK_A`, `TILE_SIZE_UNPACK_B`, FormatConfig, stimuli buffer addresses, matmul dimensions (CT_DIM, RT_DIM, KT_DIM, TILE_CNT), face counts.
  - The `RUNTIME_PARAMETERS` macro that expands to `const struct RuntimeParams&`.
  - Explains how `TemplateParameter` and `RuntimeParameter` Python abstract classes map to C++ code generation.
  - References: `tests/helpers/include/params.h`, `docs/tests/infra_architecture.md`, `docs/tests/getting_started.md`.

- `03_matmul_specific_config_structs.md`
  - `addr_mod_t` -- address modifier configuration for SRCA/SRCB/DEST increment/clear/carry-reset per MVMUL step.
  - `ckernel_template` -- MOP programming with outer/inner loops and replay buffer.
  - `ckernel_unpack_template` -- unpack MOP programming with context-dependent replay.
  - `#ifndef HF / #define HF 0 / #endif` in `llk_math_matmul.h` -- controls high-fidelity path.
  - `DstTileShape` enum and non-standard tile dimension handling.

---

### Chapter 6: Toolchain, Compilation Pipeline, and ELF Generation

**Description:** Covers the complete toolchain requirements, compilation command structure, linker scripts, and the process of producing the three kernel ELF files plus the BRISC firmware ELF.

**Directory:** `ch6_compilation/`

**Files:**

- `01_toolchain_and_dependencies.md`
  - **SFPI compiler toolchain:** `riscv-tt-elf-g++`, `riscv-tt-elf-objdump`, `riscv-tt-elf-objcopy` from `tests/sfpi/compiler/bin/`, downloaded automatically by `setup_testing_env.sh` from the `tenstorrent/sfpi` repository.
  - **Hardware-specific headers from tt-metal:** `cfg_defines.h`, `tensix.h`, `tensix_types.h`, `dev_mem_map.h`, `core_config.h`, `risc_attribs.h` -- downloaded by `setup_testing_env.sh` from `tt-metal` repository. These define register addresses, configuration register bitfield macros (RMW patterns), and memory map constants.
  - **System packages required:** `curl`, `cmake`, `build-essential`, `libyaml-cpp-dev`, `libhwloc-dev`, `libzmq3-dev`, `git-lfs`, `xxd`, `wget`.
  - **Python dependencies:** Python 3.8/3.10, `ttexalens` library, `torch` (for golden comparison), standard scientific Python packages from `tests/requirements.txt`.
  - **Firmware requirement:** Original Blackhole firmware must be flashed on the P150. The LLK test infrastructure does NOT replace chip firmware; it loads kernel code into L1 alongside existing firmware structures.
  - References: `tests/setup_testing_env.sh`, `tests/setup_external_testing_env.sh`, `tests/requirements.txt`, `tests/README.md`.

- `02_compilation_commands_and_flags.md`
  - **Compiler flags:** `-mcpu=tt-bh-tensix` (compute cores), `-mcpu=tt-bh` (non-compute/BRISC), `-std=c++17`, `-O3`, `-ffast-math`, `-nostdlib`, `-fno-use-cxa-atexit`, `-DTENSIX_FIRMWARE`, `-DENV_LLK_INFRA`, `-DENABLE_LLK_ASSERT`, `-DARCH_BLACKHOLE`.
  - **Linker flags:** `-Wl,-z,max-page-size=16`, `-Wl,-z,common-page-size=16`, `-nostartfiles`.
  - **Include paths (in order):** `tests/helpers/include/`, `tests/hw_specific/blackhole/inc/`, `tt_llk_blackhole/llk_lib/`, `tt_llk_blackhole/common/inc/`, `tt_llk_blackhole/common/inc/sfpu/`, `common/`, variant build directory (for `build.h`).
  - **Compilation command:** One invocation per thread, using stdin piping to concatenate `#include <test_source.cpp>` and `#include <trisc.cpp>`, producing `unpack.elf`, `math.elf`, `pack.elf`.
  - **BRISC firmware:** Compiled separately as `brisc.elf` from `brisc.cpp`, manages TRISC lifecycle on Blackhole via command polling loop.
  - References: `tests/python_tests/helpers/test_config.py` (build_kernel_part, OPTIONS_ALL, OPTIONS_LINK, INCLUDES).

- `03_linker_scripts_and_memory_regions.md`
  - **Linker script chain:** `memory.blackhole.ld` (defines L1 code/data regions for BRISC and TRISC0/1/2), per-thread alias scripts (`unpack.ld` -> TRISC0 regions, `math.ld` -> TRISC1 regions, `pack.ld` -> TRISC2 regions), and `sections.ld` (section placement rules).
  - Explains the TRISC L0 private memory layout: 3584 bytes data + 512 bytes stack.
  - Documents the `__instrn_buffer` address (0xFFE40000) used for replay buffer instructions.
  - Documents startup assembly: `tmu-crt0.S` -- entry point setup before C++ code runs.
  - References: `tests/helpers/ld/memory.blackhole.ld`, `tests/helpers/ld/unpack.ld`, `tests/helpers/ld/math.ld`, `tests/helpers/ld/pack.ld`, `tests/helpers/ld/sections.ld`.

---

### Chapter 7: Host-Side Execution Workflow

**Description:** Consolidates the complete host-side sequence required to initialize the device, load data and kernel ELFs, start computation, wait for completion, and read results -- the end-to-end "how to actually run this" chapter.

**Directory:** `ch7_host_execution/`

**Files:**

- `01_host_device_communication.md`
  - The `tt-exalens` library: the host-side communication layer providing `write_to_device`, `read_from_device`, `write_words_to_device`, `read_word_from_device`, `load_elf`, `parse_elf`.
  - Device initialization: `tt_exalens_init.init_ttexalens()` for hardware, or `init_ttexalens_remote()` for simulator mode.
  - ARC messages: `GO_BUSY` to activate the device, `GO_IDLE` when done. References `_send_arc_message()` in `device.py`.
  - Card reset via `tt-smi -r` when needed.
  - References: `tests/python_tests/helpers/device.py`, `tests/python_tests/helpers/hardware_controller.py`.

- `02_step_by_step_execution.md`
  - Numbered step-by-step host-side workflow:
    1. Detect chip architecture via `tt-smi` or `ttexalens` context.
    2. Set soft reset on all cores: `set_tensix_soft_reset(1)`.
    3. Load `brisc.elf` onto BRISC core: `load_elf(elf_file, location, risc_name="brisc")`.
    4. Release BRISC from reset: `set_tensix_soft_reset(0, [RiscCore.BRISC])`.
    5. Write runtime parameters to L1 at 0x20000 using `write_to_device()`.
    6. Write stimuli (input tiles, pre-tilized in 32x32 tile format) to L1 starting at 0x21000.
    7. Load TRISC ELFs: `load_elf(elf_file, location, risc_name="trisc0/1/2")`.
    8. Issue BRISC command to start TRISCs: `commit_brisc_command(location, BriscCmd.START_TRISCS)`.
    9. Poll mailbox addresses for `KERNEL_COMPLETE` (0xFF): unpack at offset 0, math at offset 4, pack at offset 8 from 0x1FFB8.
    10. Read result tiles from L1 at the result buffer addresses.
  - Documents the `BriscCommandState` protocol: BRISC polls command buffer, executes START_TRISCS (device_setup + clear_trisc_soft_reset), returns to IDLE.
  - Notes that input data must be pre-tilized (32x32 tile format) before writing to L1.
  - References: `tests/python_tests/helpers/test_config.py` (write_runtimes_to_L1, run_elf_files, wait_for_tensix_operations_finished), `tests/python_tests/helpers/device.py`.

- `03_what_tt_metal_replaces.md`
  - What TT-Metal normally provides that pure LLK must handle manually:
    - Device initialization and reset management
    - L1 memory allocation and layout
    - Circular buffer management
    - Kernel dispatch and scheduling across multiple cores
    - Data movement (NOC) for getting data to/from DRAM to L1
    - Multi-core coordination and routing
    - High-level op decomposition
  - What you gain by using pure LLK: direct control over every register configuration, ability to experiment with custom MOP programs, no overhead from runtime abstractions, single-core deterministic execution, and the ability to test individual LLK API behaviors in isolation.
  - Emphasizes that LLK tests operate on a single Tensix core -- no multi-core orchestration is available.
  - References: `tests/python_tests/helpers/device.py`, `README.md`.

---

### Chapter 8: Existing Test Cases, References, and Alternatives

**Description:** Surveys all matmul-related test cases in the TT-LLK repository as practical references, and evaluates alternative approaches that bypass LLK entirely.

**Directory:** `ch8_references_and_alternatives/`

**Files:**

- `01_matmul_test_survey.md`
  - **Primary matmul test:** `tests/sources/matmul_test.cpp` -- the canonical example showing all three TRISC thread implementations with the standard LLK matmul API. Python driver at `tests/python_tests/test_matmul.py` demonstrates format combinations (Float16_b, Float16, Float32, Bfp8_b), all four fidelity levels, destination accumulation modes, and multi-tile dimension sweeps.
  - **Custom matmul (no MOP):** `tests/sources/matmul_custom_test.cpp` with `tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h` -- experimental variant that issues MVMUL instructions directly without MOP replay buffers.
  - **Math-only matmul test:** `tests/sources/math_matmul_test.cpp` -- extended matmul with support for non-standard tile dimensions, partial faces, transpose, throttle levels.
  - **Unpack-specific matmul test:** `tests/sources/unpack_matmul_test.cpp` -- focused on unpacker behavior for matmul.
  - **Matmul with pack untilize:** `tests/sources/matmul_pack_untilize_test.cpp` -- matmul output in row-major format.
  - **Matmul with unpack tilize:** `tests/sources/matmul_unpack_tilize_test.cpp` -- matmul with row-major input conversion.
  - **Matmul performance tests:** `tests/sources/matmul_perf.cpp`, `tests/sources/math_matmul_perf.cpp` and their Python drivers.
  - **Matmul with SFPU:** `tests/sources/matmul_and_unary_sfpu_test.cpp` -- matmul followed by SFPU operation.
  - **Quasar matmul:** `tests/sources/quasar/matmul_quasar_test.cpp` -- architecture-specific differences (4 TRISC threads, different boot mode, TDMA descriptors).
  - **Experimental headers:** `llk_math_matmul_custom_no_mop.h` and `llk_unpack_AB_matmul_custom.h` in `tt_llk_blackhole/llk_lib/experimental/`.
  - **Fused operation config:** `tests/python_tests/fuser_config/fpu_matmul.yaml` -- YAML configuration for fused matmul tests.

- `02_python_test_harness.md`
  - Walks through `tests/python_tests/test_matmul.py` as the canonical Python-side reference:
    - Format selection, matmul dimension generation (`generate_matmul_dimension_combinations`), stimuli generation, tilization (`tilize_block`).
    - Golden comparison with `MatmulGolden`.
    - `TestConfig` construction with `MATH_FIDELITY`, `NUM_FACES`, `TILE_COUNT`, `CRK_TILE_DIMM` parameters.
    - Result readback and assertion (`passed_test`).
  - Covers `test_math_matmul.py`, `test_matmul_custom.py`, `test_unpack_matmul.py` as additional references.
  - Test infrastructure overview: `conftest.py` / `pytest` integration, `StimuliConfig` for input generation.
  - References: `tests/python_tests/test_matmul.py`, `tests/python_tests/helpers/matmul_sweep.py`.

- `03_alternatives_to_llk.md`
  - **Host-side CPU computation:** Bypass the Tensix engine entirely; compute matmul on the host CPU. No LLK involvement, but does not use the accelerator.
  - **TT-Metal TTNN layer:** `ttnn.matmul()` abstracts LLK, handling multi-core dispatch, data movement, and format conversion. Still uses LLK internally.
  - **TT-Forge / PyBuda:** High-level compiler frameworks that compile ML models to Tensix instructions. Use LLK under the hood via TT-Metal.
  - **Direct Tensix ISA programming:** Write raw assembly using `TTI_MVMUL` and other instructions without LLK wrappers. Possible but extremely error-prone -- LLK is essentially C++ wrappers around these instructions.
  - **SFPU-only path:** SFPU cannot perform matrix multiplication natively (no matrix unit access); not a practical matmul alternative.
  - **Non-Tensix execution paths:** BRISC and NCRISC RISC-V cores lack access to the FPU/Matrix Unit. Could theoretically compute matmul in software, but orders of magnitude slower.
  - **Simulator path:** Using tt-exalens with `--run-simulator` to run kernels without physical hardware. Still uses LLK but does not require a P150 card.
  - **Conclusion:** No true LLK-free hardware matmul path exists. The Matrix Unit (FPU) is the only hardware unit on the Tensix core capable of matrix multiplication, and LLK is the software layer that programs it. Any hardware-accelerated matmul on P150 necessarily passes through the same MVMUL instructions that LLK wraps.

---

## Conventions

### Terminology
- **P150:** Tenstorrent's PCIe accelerator card based on the Blackhole ASIC.
- **Blackhole:** The chip architecture (`ARCH_BLACKHOLE`), mapped to `tt_llk_blackhole/` in the LLK repository.
- **Tensix Core:** The full processing unit including RISC-V cores, Tensix Engine, L1 memory, and routers.
- **Tensix Engine:** The coprocessor within a Tensix Core that executes Tensix ISA instructions (unpack, math, pack threads).
- **TRISC0/1/2:** The three "TRISC" RISC-V cores that control the Tensix Engine threads (Unpack, Math, Pack respectively).
- **BRISC:** "Boot RISC" -- the supervisor RISC-V core that manages device initialization and TRISC lifecycle.
- **LLK:** Low-Level Kernel -- the header-only C++ library that wraps Tensix ISA instructions into callable functions.
- **MOP:** Micro-Operation Program -- a pre-programmed instruction sequence stored in a replay buffer, executed by the Tensix Engine's template mechanism.
- **FPU / Matrix Unit:** The floating-point unit that executes MVMUL (matrix-vector multiply) instructions.
- **SFPU:** Special FPU -- a 32-lane SIMD vector unit for non-linear operations, operates on the Destination register.
- **SrcA / SrcB:** Source register files (double-buffered) holding unpacked tile data for FPU operations.
- **Dest:** Destination register file storing FPU/SFPU output, double-buffered in SyncHalf mode.
- **Face:** A 16x16 sub-block of a 32x32 tile (tiles have 4 faces: F0, F1, F2, F3).
- **Tile:** A 32x32 data element block unless otherwise qualified (e.g., "16x32 tile").
- **ct_dim / rt_dim / kt_dim:** Column-tile, row-tile, and K-tile dimensions of the matmul block operation.
- **DstSync:** Destination register synchronization mode (SyncHalf or SyncFull).
- **ttexalens:** The Python library providing host-to-device communication (`read_from_device`, `write_to_device`, `load_elf`).
- **SFPI:** The RISC-V cross-compiler toolchain from Tenstorrent for Tensix firmware.
- **build.h:** Auto-generated header containing compile-time and runtime parameter definitions for a test variant.

### Notation
- LLK function names are written with their leading underscores: `_llk_math_matmul_init_<>()`.
- Template parameters are shown in angle brackets: `<MATH_FIDELITY, THROTTLE_LEVEL>`.
- Preprocessor defines are written in `ALL_CAPS`: `ARCH_BLACKHOLE`, `LLK_TRISC_MATH`.
- `TTI_` prefix indicates a Tensix ISA instruction macro (e.g., `TTI_MVMUL`, `TTI_SEMWAIT`).
- L1 addresses are given in hexadecimal with the `0x` prefix.
- File paths are relative to the tt-llk repository root unless otherwise noted.
- Hardware register names use their exact names from `cfg_defines.h` (e.g., `ALU_ACC_CTRL_Fp32_enabled_RMW`).
- Tile dimensions are written as RxC (e.g., 32x32, 16x16).

### Architecture-Specific Annotations
- [BH] prefix indicates statements that apply only to Blackhole (P150). Statements without a qualifier apply to all architectures.

### Formatting Rules
- Code blocks use C++ syntax highlighting for kernel code and Python for host-side scripts.
- Each API function is documented with its signature, template parameters, runtime parameters, and a brief description of the hardware operation it triggers.
- Each chapter begins with a short summary of what the reader will learn and a "Prerequisites" note listing which earlier chapters must be read first.
- Cross-references between chapters use the format "see Chapter N, Section Title".
- L1 memory addresses and layout diagrams use monospaced tables.

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Reason |
|---------|-----------|--------|
| Ch 2 (Data Formats and Memory) | Ch 1 (P150 and Blackhole) | Tile sizes, data format enums, and L1 layout are architecture-specific; Chapter 1 establishes which architecture is in use and introduces the dataflow model. |
| Ch 3 (Kernel Architecture) | Ch 1, Ch 2 | The three-thread model references the Tensix core from Ch 1; source structure references tile formats and L1 layout from Ch 2; synchronization references the data path from Ch 1. |
| Ch 4 (API Call Sequences) | Ch 2, Ch 3 | API calls reference tile formats and data formats from Ch 2; they operate within the kernel structure and synchronization framework from Ch 3. |
| Ch 5 (Compile-Time Config) | Ch 3, Ch 4 | Configuration defines drive the kernel source structure (Ch 3) and parameterize the API calls (Ch 4). |
| Ch 6 (Compilation) | Ch 1, Ch 2, Ch 5 | Compilation uses architecture-specific flags (Ch 1), linker scripts reference memory layout (Ch 2), and the compilation consumes build.h from Ch 5. |
| Ch 7 (Host Execution) | Ch 2, Ch 3, Ch 6 | Host workflow writes data to L1 addresses from Ch 2, loads ELFs produced by Ch 6, and uses the boot/mailbox protocol from Ch 3. |
| Ch 8 (References and Alternatives) | All prior chapters | Test cases are concrete instantiations of the patterns described in Ch 3-4; alternatives contextualize the full stack knowledge from Ch 1-7. |

---

## Selection Rationale

This final plan is a **hybrid** that draws from all four candidate plans, selecting the strongest elements from each while addressing the weaknesses identified in all four evaluations. Here is what was chosen and why:

### From Plan v1: Deep API Documentation and Comprehensive Test Survey
- **Chapter 4 (API Call Sequences)** adopts Plan v1's exceptionally detailed parameter-by-parameter API documentation, which was cited by all evaluators as the deepest and most precise across all plans. The face-level addressing explanation (`D[8,16] = B[8,16] * A[16,16]`), MOP replay buffer structure, and reuse_a logic are all preserved from v1's Chapter 3.
- **Chapter 8 test survey** draws from Plan v1's Chapter 7, which covers 12 test files -- more than any other plan. The experimental custom matmul and Quasar comparison are included.
- **Conventions section** adopts Plan v1's comprehensive 27-term glossary as the foundation, which was rated strongest across all plans.
- **Synchronization details** (semaphore numbers, tensix_sync, TTI_STALLWAIT variants) come from Plan v1's Chapter 4, which was the only plan to provide this level of semaphore-protocol depth.

### From Plan v2: Host-Side Execution Workflow
- **Chapter 7 (Host-Side Execution)** is modeled directly on Plan v2's Chapter 7 Section 02, which was cited by all four evaluators as having the "most actionable and clearly structured host-side guide." The 10-step numbered workflow is adopted wholesale.
- **Python test harness walkthrough** (Chapter 8, file 02) is drawn from Plan v2's Chapter 8 Section 02, a unique addition among the plans that helps readers actually run existing tests.
- **"What TT-Metal replaces" section** (Chapter 7, file 03) is adapted from Plan v2's Chapter 7 Section 03, which was praised for explicitly enumerating what TT-Metal normally provides.

### From Plan v3: Software Stack Positioning, Consolidated Kernel Architecture, and Conventions
- **Chapter 1, file 03 ("LLK Software Position")** is taken from Plan v3, which was the only plan to include a dedicated discussion of LLK's position in the software stack. All evaluators noted this as a valuable unique addition.
- **Chapter 3 structure** (combining kernel source, boot sequence, and synchronization into one "Kernel Architecture" chapter) follows Plan v3's Chapter 4, which was praised for creating a cohesive chapter that ties together the compilation model, boot flow, and thread coordination. This directly addresses the "synchronization fragmented" criticism leveled at Plans v2 and v4.
- **"[BH]" architecture qualifier convention** is adopted from Plan v3's conventions, which was noted as a useful formatting decision by the evaluator.
- **Firmware requirements** are integrated into Chapter 6 (toolchain) following the evaluator's suggestion to merge Plan v3's standalone firmware file into the dependency inventory.

### From Plan v4: Dataflow-First Pedagogy and API Summary Table
- **Chapter 1 file 02 and Chapter 2 structure** are influenced by Plan v4's approach of explaining the full Tensix dataflow pipeline and matmul computation model before any code discussion. Plan v4 was the only plan to dedicate standalone content to the dataflow model, which all evaluators praised as "the best conceptual foundation." The DstSync model, fp32 accumulation effects, and block dimension explanations are incorporated into Chapter 2.
- **Chapter 4, file 04 (Minimum Call Summary Table)** adopts Plan v4's unique "summary table: minimum call count per thread for a 1x1x1 matmul vs. general RTxCTxKT matmul," which was noted as a concrete, actionable reference absent from all other plans.

### Structural Decisions Addressing Cross-Plan Weaknesses

All four evaluations identified three recurring problems. This hybrid addresses each:

1. **Data formats must precede API calls.** Plans v2 and v4 were criticized for placing data format treatment too late (after API calls). This plan places data formats, tile geometry, and memory layout in Chapter 2 -- before the kernel architecture (Ch 3) and API calls (Ch 4) -- following the approach of Plans v1 and v3.

2. **Host-side execution must be consolidated.** Plans v1, v3, and v4 were all criticized for fragmenting the host-side workflow across multiple chapters. This plan dedicates Chapter 7 entirely to host-side execution, with a clear 10-step workflow in a single file, plus supporting files for device communication details and the TT-Metal gap analysis.

3. **Synchronization must be consolidated.** Plans v2 and v4 were criticized for scattering synchronization across multiple chapters. This plan consolidates all synchronization content in Chapter 3, file 03, following Plan v3's approach. The API calls in Chapter 4 reference synchronization but do not re-explain it; they cross-reference Chapter 3.

### Chapter Count and Ordering
The final plan uses 8 chapters ordered from foundational to advanced: hardware identity (Ch 1) -> data formats and memory (Ch 2) -> kernel architecture including boot and sync (Ch 3) -> API calls (Ch 4) -> compile-time config (Ch 5) -> compilation toolchain (Ch 6) -> host-side execution (Ch 7) -> references and alternatives (Ch 8). This ordering ensures every concept is introduced before it is used, and the practical "how to run" chapter comes near the end when the reader has full context.
