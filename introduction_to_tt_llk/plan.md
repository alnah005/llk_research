# Introduction to TT-LLK: Guide Plan

## Audience

**Primary audience:** Software engineers and op writers who need to write, modify, or optimize low-level kernels for Tenstorrent Tensix cores. They are expected to have:

- Proficiency in C++ (at least C++17, including templates and constexpr)
- Basic understanding of computer architecture concepts (registers, memory hierarchies, SIMD)
- Familiarity with machine learning operations at a conceptual level (matrix multiply, elementwise ops, reductions)
- Some exposure to embedded or hardware-adjacent programming (memory-mapped registers, DMA)

**Secondary audience:** Engineers working at higher levels of the Tenstorrent stack (TT-Metal, TT-Forge) who want to understand what the LLK layer does and how it maps to hardware, without necessarily writing LLK code themselves.

**Not assumed:** Prior knowledge of Tenstorrent hardware, the Tensix ISA, RISC-V internals, or the specific data formats used by Tenstorrent chips.

---

## Chapter List

### Chapter 1: What is TT-LLK?

**Description:** Introduces TT-LLK as the foundational software layer for programming Tenstorrent AI chips, explains its role in the software stack, and orients the reader to the repository structure.

**Directory:** `ch1_what_is_tt_llk/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to subtopics

- `software_stack_position.md`
  - TT-LLK as a header-only C++17 library of compute primitives for Tensix cores
  - Relationship to TT-Metalium (runtime SDK that compiles and dispatches kernels using LLKs) and TT-Forge (compiler that bridges ML frameworks to hardware via LLKs)
  - Diagram: ML Framework -> TT-Forge -> TT-Metal -> TT-LLK -> Tensix Hardware
  - LLK as "first software layer above hardware" -- encapsulates Tensix ISA into callable C++ functions
  - Standalone development and testing capability independent of TT-Metal

- `repository_structure.md`
  - Top-level layout: `tt_llk_wormhole_b0/`, `tt_llk_blackhole/`, `tt_llk_quasar/`, `common/`, `tests/`, `docs/`
  - Per-architecture directory structure: `common/inc/` (ckernel headers, SFPU implementations), `instructions/` (assembly YAML), `llk_lib/` (LLK API headers)
  - Naming conventions: `llk_unpack_*.h`, `llk_math_*.h`, `llk_pack_*.h`, `ckernel_*.h`, `cmath_common.h`, `cpack_common.h`, `cunpack_common.h`
  - The `common/` directory at repo root containing shared headers (`llk_assert.h`, `tensor_shape.h`)
  - The three hardware targets: Wormhole B0, Blackhole, Quasar -- and why each has its own implementation directory

- `compression_analysis.md`

---

### Chapter 2: Tensix Core Architecture

**Description:** Explains the hardware architecture that LLKs program -- the Tensix Core with its five RISC-V processors, three-way threaded coprocessor (Tensix Engine), and L1 memory.

**Directory:** `ch2_tensix_core_architecture/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to subtopics

- `core_components.md`
  - Tensix Core as the fundamental unit in a Tenstorrent chip grid
  - Four major components: L1 SRAM, two NoC routers, five RISC-V processors, Tensix Engine
  - L1 memory: stores input/output tensors and program code for all RISC-V cores
  - The two routers for inter-core and DRAM data movement (managed by BRISC/NCRISC, outside LLK scope)

- `riscv_processors.md`
  - Five 32-bit in-order single-issue RISC-V cores: B (BRISC), T0 (TRISC0), T1 (TRISC1), T2 (TRISC2), NC (NCRISC)
  - BRISC and NCRISC: handle NOC communication and board setup -- not directly used by LLK compute
  - Three TRISCs: control the Tensix Engine through Tensix ISA instructions
  - TRISC0 controls unpackers (data movement L1 -> source registers)
  - TRISC1 controls FPU and SFPU (math operations)
  - TRISC2 controls packers (data movement destination register -> L1)
  - Each TRISC pushes instructions to its corresponding coprocessor thread

- `tensix_engine.md`
  - Three-way threaded coprocessor with independent frontend pipelines per thread and shared backend execution units
  - Thread model: T0 (UNPACK), T1 (MATH), T2 (PACK) -- each with its own instruction stream
  - Shared backend execution units: Sync Unit, Unpackers, Matrix Unit (FPU), Packers, Vector Unit (SFPU), Scalar Unit (ThCon), Configuration Unit, Mover, Miscellaneous Unit
  - In-order frontend per thread, out-of-order cross-thread execution in shared backend
  - Distinction between Tensix ISA (coprocessor instructions) and RISC-V ISA (host core instructions)

- `data_flow.md`
  - The core compute pipeline: L1 -> Unpack -> Source Registers -> FPU -> Destination Register -> Pack -> L1
  - Alternative SFPU path: Destination Register -> SFPU -> Destination Register
  - Source A and Source B registers: double-buffered, store up to 19-bit data elements, one tile per bank
  - Destination register: stores FPU results and SFPU operands/results, configurable 4-16 tiles, supports 32-bit elements
  - Data flow diagram with all register files and execution units labeled

- `compression_analysis.md`

---

### Chapter 3: Data Organization

**Description:** Covers how data is structured in L1 memory (tiles, faces, data formats) and how format conversions happen during unpack and pack operations.

**Directory:** `ch3_data_organization/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to subtopics

- `tiles_and_faces.md`
  - Tile: the fundamental data unit for LLK operations -- 32x32 datums
  - Four 16x16 faces per tile (F0-F3), stored in row-major order within each face
  - `TensorShape` struct: `face_r_dim`, `face_c_dim`, `num_faces_r_dim`, `num_faces_c_dim`
  - Tile dimension constants: `TILE_R_DIM` (32), `TILE_C_DIM` (32), `FACE_R_DIM` (16), `FACE_C_DIM` (16)
  - Tiny tiles (non-32x32): `num_faces` parameter (1, 2, or 4), `face_r_dim` for partial faces
  - Row-major order vs. tile order storage in L1

- `data_formats.md`
  - Complete list of supported `DataFormat` enum values: Float32, Float16 (FP16a), Float16_b (BFP16/FP16b), Bfp8, Bfp8_b, Bfp4, Bfp4_b, Bfp2, Bfp2_b, Lf8, Int8, UInt8, UInt16, Int32, UInt32, Tf32, Fp8_e4m3, MxFp8R, MxFp8P
  - Which formats are supported in L1 (all), Source registers (limited subset, up to 19-bit), and Destination register (up to 32-bit)
  - Block floating point (BFP) formats: shared exponent across a block of mantissas
  - `InstrModLoadStore` enum: how SFPU load/store instructions specify format (FP32, FP16A, FP16B, INT32, INT8, etc.)
  - `GetSfpLoadStoreInstrMod<DataFormat>()` template for selecting correct SFPU instruction modifier

- `format_conversion.md`
  - Automatic format conversion in unpacker gaskets (L1 format -> register format)
  - Automatic format conversion in packer gaskets (register format -> L1 format)
  - FP32 destination accumulation (`is_fp32_dest_acc_en` template parameter) for higher precision intermediates
  - `IS_BFP_FORMAT()` macro for detecting block floating point formats
  - Stochastic rounding modes (`StochRndType`: None, Fpu, Pack, All) and their effect on precision

- `math_fidelity.md`
  - `MathFidelity` enum: LoFi (0), HiFi2 (2), HiFi3 (3), HiFi4 (4)
  - How fidelity controls number of multiplication phases for full precision
  - Trade-off between accuracy and performance
  - `is_high_fidelity()` helper and fidelity increment in address modifier configuration

- `compression_analysis.md`

---

### Chapter 4: The Unpack-Math-Pack Pipeline

**Description:** Explains the three-stage compute pipeline in detail -- how unpack, math, and pack operations work together, including synchronization between the three TRISC threads.

**Directory:** `ch4_unpack_math_pack_pipeline/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to subtopics

- `unpack_operations.md`
  - Unpack = DMA from L1 to source/destination registers with format conversion
  - Two unpackers: Unpacker 0 (Source A and Destination register), Unpacker 1 (Source B)
  - LLK unpack API headers and their purposes:
    - `llk_unpack_A.h`: Single tile to Source A (unary operations)
    - `llk_unpack_AB.h`: Two tiles to Source A and Source B (binary operations)
    - `llk_unpack_AB_matmul.h`: Optimized matmul unpacking with register reuse
    - `llk_unpack_reduce.h`: Tile to Source A + scalar to Source B
    - `llk_unpack_tilize.h`: Row-major to tile format conversion during unpack
    - `llk_unpack_untilize.h`: Tile to row-major conversion during unpack
  - Common API pattern: `_llk_unpack_hw_configure_<>()`, `_llk_unpack_*_init_<>()`, `_llk_unpack_*_<>()`, `_llk_unpack_*_uninit_<>()`
  - Key template parameters: `BroadcastType`, `acc_to_dest`, `EltwiseBinaryReuseDestType`, `unpack_to_dest`
  - Context switching and double buffering: `wait_for_next_context()`, `switch_config_context()`
  - Semaphore-based synchronization: `semaphore_post(UNPACK_SYNC)`, `t6_semaphore_get(UNPACK_SYNC)`

- `math_operations.md`
  - Math = FPU/SFPU computation on register data
  - FPU: matrix of multiply-accumulate cells operating on Source A x Source B -> Destination
  - Four FPU cell operations: accumulated dot product, accumulated eltwise add, accumulated eltwise mul, eltwise add
  - LLK math API headers:
    - `llk_math_eltwise_binary.h`: Element-wise add/sub/mul (`EltwiseBinaryType` enum: ELWADD, ELWSUB, ELWMUL, ELWDIV, ELWLESS)
    - `llk_math_eltwise_unary_datacopy.h`: Source A/B to Destination copy (`DataCopyType`: A2D, B2D)
    - `llk_math_eltwise_unary_sfpu.h`: SFPU unary operations on Destination register data
    - `llk_math_matmul.h`: Matrix multiply (D = B * A with source register tiling)
    - `llk_math_reduce.h`: Row/column/scalar reduction (`ReduceDim`, `PoolType`: SUM, AVG, MAX, MIN)
    - `llk_math_transpose_dest.h`: Transpose in destination register
  - Common API pattern: `_llk_math_hw_configure_<>()`, `_llk_math_*_init_<>()`, `_llk_math_*_<>()`, `_llk_math_dest_section_done_<>()`
  - Address modifier configuration: `addr_mod_t` struct with `.srca`, `.srcb`, `.dest`, `.fidelity` fields
  - MOP (Micro-Op Program) configuration via `ckernel_template`

- `pack_operations.md`
  - Pack = DMA from Destination register to L1 with format conversion
  - Single packer controlled by TRISC2
  - LLK pack API headers:
    - `llk_pack.h`: Tile from Destination to L1, maintaining tile format
    - `llk_pack_untilize.h`: Tile from Destination to L1, converting to row-major
    - `llk_pack_rows.h`: Row-based packing
  - Common API pattern: `_llk_pack_hw_configure_<>()`, `_llk_pack_*_init_<>()`, `_llk_pack_*_<>()`, `_llk_pack_dest_section_done_<>()`
  - Zero output flag (`zero_output` template parameter)
  - ReLU integration in packer (`ReluType`: NO_RELU, ZERO_RELU, MIN_THRESHOLD_RELU, MAX_THRESHOLD_RELU)

- `synchronization.md`
  - `DstSync` enum: `SyncHalf` (double-buffered destination) vs. `SyncFull` (single-buffered)
  - Destination register double buffering: `dest_section_flip()`, `math_dest_wait()`
  - Semaphore-based sync between unpack/math/pack threads
  - `_llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>()`
  - `_llk_math_wait_for_dest_available_<Dst>()` and `_llk_math_dest_section_done_<Dst, is_fp32_dest_acc_en>()`
  - Stall instructions: `TTI_STALLWAIT` with `p_stall::STALL_MATH`, `p_stall::STALL_SFPU`, `p_stall::STALL_UNPACK`, `p_stall::STALL_CFG`
  - The `ckernel_template` class: programming MOP loops with outer/inner loop counts, start/end/loop operations

- `compression_analysis.md`

---

### Chapter 5: SFPU Operations

**Description:** Deep dive into the Scalar Floating Point Unit -- its architecture, programming model, and the extensive catalog of supported operations.

**Directory:** `ch5_sfpu_operations/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to subtopics

- `sfpu_architecture.md`
  - SFPU as a 32-lane SIMD engine within the FPU, controlled by TRISC1
  - Load/store architecture: data must be loaded from Destination register to SFPU internal registers, operated on, then stored back
  - 32-bit input calculations (higher precision than FPU's 19-bit source registers)
  - Input operands must reside in Destination register (via FPU datacopy from Source A/B, or direct unpack from L1 via Unpacker 0)
  - `VectorMode` enum: None, R (row), C (column), RC (all faces), RC_custom
  - Face iteration pattern in `_llk_math_eltwise_unary_sfpu_params_()`: iterating over faces with `SETRWC` for dest pointer advancement

- `sfpu_operations_catalog.md`
  - Complete `SfpuType` enum listing with categories:
    - **Activation functions:** tanh, hardtanh, gelu, gelu_derivative, sigmoid, silu, relu, lrelu, elu, celu, softplus, hardsigmoid
    - **Exponential/logarithmic:** exponential, exp2, exp_with_base, expm1, log, log_with_base, log1p
    - **Power/root:** sqrt, rsqrt, square, power, reciprocal
    - **Trigonometric:** sine, cosine, tan, asin, acos, atan, atanh, asinh, acosh
    - **Comparison:** equal_zero, not_equal_zero, less_than_zero, greater_than_zero, etc., heaviside
    - **Special math:** erf, erfc, erfinv, i0, cdf (via related files)
    - **Integer operations:** add_int32, mul_int32, quant_int32, requant_int32, dequant_int32
    - **Bitwise operations:** bitwise_xor, bitwise_and, bitwise_or, bitwise_not, left_shift, right_shift
    - **Reduction/aggregation:** reduce, cumsum, tiled_prod, add_top_row, topk_local_sort, topk_merge, topk_rebuild
    - **Type conversion:** typecast, cast_fp32_to_fp16a
    - **Other:** abs, sign, signbit, negative, clamp, dropout, fill, where, threshold, floor, ceil, remainder, fmod, rounding_ops, reshuffle_rows
  - Each SFPU op implemented as a header in `common/inc/sfpu/ckernel_sfpu_*.h`

- `sfpu_binary_operations.md`
  - Binary SFPU operations (two-operand operations in SFPU, distinct from FPU binary eltwise)
  - `llk_math_eltwise_binary_sfpu.h` and `llk_math_eltwise_binary_sfpu_params.h`
  - Ternary SFPU operations: `llk_math_eltwise_ternary_sfpu.h` (e.g., where operation)
  - Welford's online algorithm: `llk_math_welfords_sfpu.h` for numerically stable variance computation

- `compression_analysis.md`

---

### Chapter 6: Hardware Target Differences

**Description:** Compares the three chip architectures (Wormhole B0, Blackhole, Quasar) and how their LLK implementations differ.

**Directory:** `ch6_hardware_targets/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to subtopics

- `architecture_comparison.md`
  - Shared API surface: all three targets implement the same high-level LLK operations (unpack, math, pack)
  - Wormhole B0 and Blackhole: very similar, both have `ckernel_xmov.h`, `ckernel_mutex_guard.h`, `ckernel_debug.h` -- identical common/inc structure
  - Quasar: significantly different architecture -- uses `ckernel_pcbuf.h` (PC buffer), `ckernel_vector.h`, `ckernel_dest.h`, `ckernel_risc_atomics.h`, `ckernel_trisc_common.h`
  - Quasar lacks BRISC core (no BRISC elf in L1 layout)
  - Quasar enum differences: `EltwiseBinaryType`, `BroadcastType`, `ReduceDim`, etc. are `enum class` (scoped) vs. unscoped enums in WH/BH
  - Quasar has a reduced `SfpuType` enum (16 operations) vs. WH/BH (100+ operations)
  - Quasar has different LLK API file organization: `llk_unpack_unary_operand.h` and `llk_unpack_binary_operands.h` instead of `llk_unpack_A.h` / `llk_unpack_AB.h`

- `llk_api_differences.md`
  - Wormhole B0 and Blackhole: near-identical `llk_lib/` contents (same file names, same API signatures)
  - Quasar unique files: `llk_srcs_tdma.h`, `llk_pack_matmul.h`, `llk_math_eltwise_binary_broadcast.h`, `llk_math_unary_broadcast.h`, `llk_unpack_unary_broadcast_operands.h`, `llk_unpack_binary_broadcast_operands.h`
  - Quasar missing from WH/BH: `llk_pack_rows.h`, `llk_math_transpose_dest.h`, `llk_math_welfords_sfpu.h`, binary/ternary SFPU param files
  - Quasar `llk_math_eltwise_unary_sfpu_common.h` replaces the split sfpu/sfpu_params pattern
  - Boot mode differences: WH/BH default to `BootMode.BRISC`, Quasar defaults to `BootMode.TRISC`
  - Instruction definitions: `assembly.yaml` files per architecture

- `compression_analysis.md`

---

### Chapter 7: Test Infrastructure

**Description:** Explains how to write, run, and debug LLK tests using the Python/C++ test framework, including the compilation pipeline, stimuli generation, and validation methodology.

**Directory:** `ch7_test_infrastructure/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to subtopics

- `test_architecture_overview.md`
  - Test = Python file (parametrization, stimuli, golden, validation) + C++ file (kernel code using LLK API)
  - Python side: pytest-based, `@parametrize` decorator for variant generation, `TestConfig` object as central configuration
  - C++ side: three `#ifdef` sections (`LLK_TRISC_UNPACK`, `LLK_TRISC_MATH`, `LLK_TRISC_PACK`), each with `void run_kernel(RUNTIME_PARAMETERS params)`
  - Compilation: SFPI compiler (custom RISC-V toolchain) produces `unpack.elf`, `math.elf`, `pack.elf`
  - `build.h`: auto-generated header with template parameters and format configuration per variant
  - `params.h`: contains `RUNTIME_PARAMETERS` macro and `struct RuntimeParams`

- `writing_tests.md`
  - Python test file naming: `test_*.py` in `tests/python_tests/`
  - `TestConfig` constructor arguments: `test_name`, `formats`, `templates`, `runtimes`, `variant_stimuli`, `boot_mode`, `profiler_build`, `unpack_to_dest`, `dest_acc`, `l1_acc`, etc.
  - `StimuliConfig`: generating and configuring input stimuli (buffer_A, buffer_B, formats, tile counts, dimensions)
  - Golden generators: `generate_stimuli()` and `get_golden_generator()` from `helpers/golden_generators.py`
  - Template parameters vs. runtime parameters: compile-time constexpr values vs. L1-loaded struct fields
  - `TemplateParameter` and `RuntimeParameter` abstract classes: `convert_to_cpp()`, `convert_to_struct_fields()`
  - Validation: `passed_test()` using `torch.isclose()` with format-specific tolerances (e.g., Float16: atol=0.05/rtol=0.05, Bfp8_b: atol=0.1/rtol=0.2, Int32: exact match)
  - C++ kernel structure: mandatory includes (`ckernel.h`, `llk_defs.h`), global variables (`unp_cfg_context`, `pack_sync_tile_dst_ptr`, `math_sync_tile_dst_index`), three `run_kernel` functions

- `running_and_debugging.md`
  - Environment setup: `setup_testing_env.sh` (Docker) or `setup_external_testing_env.sh` (external)
  - Running tests: `pytest test_file.py`, parallel compilation with `--compile-producer -n 10`, parallel execution with `--compile-consumer -n 10`
  - Custom pytest flags: `--coverage`, `--speed-of-light`, `--compile-producer`/`--compile-consumer`, `--no-debug-symbols`, `--logging-level`, `--detailed-artefacts`
  - Build artifacts location: `/tmp/tt-llk-build/` with variant hash directories
  - L1 memory layouts: performance vs. debug layouts, address ranges for TRISC code/data, stimuli space, runtime arguments
  - Code coverage: compiling with `--coverage`, generating reports with `generate_coverage_report.sh`

- `performance_testing.md`
  - Performance test naming: `perf_*.py`
  - `ProfilerConfig` object: extends TestConfig with `run_types` (performance markers)
  - C++ profiling macros: `ZONE_SCOPED("name")` and `PROFILER_SYNC()`
  - Hardware performance counters: five counter banks (INSTRN_THREAD, FPU, TDMA_UNPACK, L1, TDMA_PACK)
  - Performance counter modes, control registers, and output registers
  - Performance report generation and interpretation

- `compression_analysis.md`

---

### Chapter 8: Kernel Development Patterns

**Description:** Provides practical guidance on writing LLK kernels -- common patterns, API conventions, and worked examples demonstrating how the concepts from previous chapters come together.

**Directory:** `ch8_kernel_development_patterns/`

**Files:**

- `index.md`
  - Chapter overview and learning objectives
  - Navigation to subtopics

- `api_conventions.md`
  - Naming convention: `_llk_<stage>_<operation>_<variant>_<suffix>_()` where stage is unpack/math/pack, suffix is hw_configure/init/uninit or omitted for the main call
  - Template parameter conventions: `BroadcastType`, `MathFidelity`, `DstSync`, `is_fp32_dest_acc_en`, `EltwiseBinaryType`, etc. as compile-time parameters
  - Runtime parameter conventions: L1 addresses, tile counts, face dimensions passed via `params` struct
  - Global state variables: `unp_cfg_context` (unpacker config context), `pack_sync_tile_dst_ptr`, `math_sync_tile_dst_index`
  - The `ckernel` namespace and sub-namespaces (`ckernel::unpacker`, `ckernel::packer`, `ckernel::math`)
  - MOP programming: `ckernel_template` class with `program()` and `run()`, address modifier system (`addr_mod_t`)
  - LLTT (Low-Level Trace/replay): `lltt::record()`, `lltt::replay_insn()` for instruction replay buffers
  - LLK assert system: `LLK_ASSERT()` macro for validation of parameters at runtime

- `eltwise_binary_walkthrough.md`
  - Complete walkthrough of `eltwise_binary_test.cpp` as a reference kernel
  - UNPACK section: `_llk_unpack_hw_configure_()`, `_llk_unpack_AB_init_()`, loop calling `_llk_unpack_AB_()`
  - MATH section: `_llk_math_pack_sync_init_()`, `_llk_math_hw_configure_()`, `_llk_math_eltwise_binary_init_()`, loop with `_llk_math_wait_for_dest_available_()`, `_llk_math_eltwise_binary_()`, `_llk_math_dest_section_done_()`
  - PACK section: corresponding pack init and loop structure
  - How Python test parametrizes formats, broadcast types, tile dimensions, and number of tiles
  - How `build.h` template parameters (e.g., `BROADCAST_TYPE`, `ELTWISE_BINARY_TYPE`, `dest_sync`) connect Python to C++

- `matmul_walkthrough.md`
  - Matrix multiply kernel structure: `_llk_unpack_AB_matmul_init_()` for optimized register reuse
  - Inner loop tiling: how Source A and Source B tiles are paired for the D = B * A computation
  - Fidelity phases: how `matmul_configure_addrmod<MathFidelity, THROTTLE_LEVEL>()` sets up multi-phase multiplication
  - Throttle level concept for pipeline flow control
  - Address modifier cycling through faces: `ADDR_MOD_0` through `ADDR_MOD_5` for different reset/increment patterns

- `compression_analysis.md`

---

## Conventions

### Terminology

- **Datum** -- A single data element within a tensor.
- **Tile** -- A 32x32 block of datums, the fundamental unit of LLK computation. Contains 4 faces.
- **Face** -- A 16x16 block of datums, one quarter of a tile. Faces are labeled F0-F3 in row-major order.
- **Tiny tile** -- A tile with fewer than 4 faces or reduced face dimensions (e.g., 16x32, 32x16, 16x16).
- **Source A / Source B** -- Double-buffered source operand registers fed by Unpacker 0 and Unpacker 1 respectively.
- **Destination register (Dest)** -- Output register for FPU results and input/output for SFPU operations.
- **FPU** -- Floating Point Unit, the primary matrix computation engine.
- **SFPU** -- Scalar Floating Point Unit, a 32-lane SIMD engine for complex math operations.
- **MOP** -- Micro-Op Program, a small loop programmed via `ckernel_template` to execute repeated Tensix instructions.
- **Gasket** -- Hardware-accelerated data format conversion unit in unpackers and packers.
- **TRISC0/1/2** -- The three RISC-V processors that control unpack, math, and pack respectively.
- **Tensix Engine** -- The coprocessor controlled by TRISCs, executing Tensix ISA instructions.
- **Tensix ISA** -- The custom instruction set for the Tensix coprocessor, distinct from RISC-V.

### Code Formatting

- All C++ code snippets use the exact function signatures and naming from the repository.
- Template parameters are shown with their type and default values (e.g., `template <BroadcastType BType = BroadcastType::NONE>`).
- Hardware register names preserve their exact casing from the codebase (e.g., `THCON_SEC0_REG2_Haloize_mode_RMW`).
- Enum values are shown with their fully qualified names where disambiguation is needed.

### Notation

- File paths are relative to the repository root (e.g., `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h`).
- When describing behavior common to Wormhole B0 and Blackhole, use "WH/BH" as shorthand.
- When describing Quasar-specific behavior, explicitly note "Quasar only".
- Hardware addresses use hexadecimal notation with `0x` prefix (e.g., `0x00005000`).
- Bit fields use the format `[high:low]` (e.g., `[16:8]` for counter_sel).

### Formatting Rules

- Each chapter opens with a brief (2-3 sentence) summary and a list of what the reader will learn.
- Code examples are drawn from actual test source files and API headers in the repository.
- Diagrams reference the existing images in `docs/llk/l1/images/` and `docs/llk/l2/images/` where available.
- Tables are used for enum value listings, API parameter descriptions, and comparison matrices.
- Internal cross-references use relative Markdown links (e.g., `[synchronization](../ch4_unpack_math_pack_pipeline/synchronization.md)`).

---

## Cross-Chapter Dependencies

| Chapter | Depends On | Concepts Referenced |
|---------|-----------|-------------------|
| Ch1 (What is TT-LLK) | None | Foundational -- introduces all terms used later |
| Ch2 (Tensix Core Architecture) | Ch1 | References repository structure from Ch1 to locate architecture-specific code |
| Ch3 (Data Organization) | Ch2 | Uses register file descriptions from Ch2 (Source A/B, Destination) to explain format constraints |
| Ch4 (Unpack-Math-Pack Pipeline) | Ch2, Ch3 | Builds on hardware components from Ch2 and data format knowledge from Ch3 |
| Ch5 (SFPU Operations) | Ch2, Ch3, Ch4 | Requires understanding of Destination register (Ch2), data formats (Ch3), and the math stage of the pipeline (Ch4) |
| Ch6 (Hardware Targets) | Ch1, Ch2, Ch4, Ch5 | Compares architecture differences (Ch2), API variations (Ch4), and SFPU support (Ch5) across targets introduced in Ch1 |
| Ch7 (Test Infrastructure) | Ch3, Ch4 | Uses data format knowledge (Ch3) for stimuli configuration and API patterns (Ch4) for kernel structure |
| Ch8 (Kernel Development Patterns) | All previous | Synthesizes all prior chapters into practical worked examples |
