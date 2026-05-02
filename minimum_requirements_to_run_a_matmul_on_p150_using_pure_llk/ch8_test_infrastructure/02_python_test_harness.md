# Chapter 8 -- Putting It All Together: Test Infrastructure and Alternatives

## 8.2 The Python Test Harness

### 8.2.1 Harness Architecture Overview

Every matmul test in tt-llk follows a single execution model orchestrated by the Python test harness. The harness is not a thin wrapper -- it handles the complete lifecycle from stimulus generation through compilation, device programming, execution, data collection, and validation. Understanding this harness is essential because it defines the minimum host-side workflow for running any LLK kernel, including matmul.

The core components are:

1. **`TestConfig`** (`helpers/test_config.py`) -- the central orchestrator that owns compilation, ELF loading, runtime parameter serialization, and device interaction.
2. **`StimuliConfig`** (`helpers/stimuli_config.py`) -- manages input/output tensor allocation in L1, stimulus data serialization, and result collection.
3. **`MatmulGolden`** (`helpers/golden_generators.py`) -- produces reference tensors via PyTorch matrix multiplication with format-aware precision modeling.
4. **Parameter dataclasses** (`helpers/test_variant_parameters.py`) -- typed wrappers for compile-time template parameters and runtime parameters.
5. **`matmul_sweep.py`** (`helpers/matmul_sweep.py`) -- generates the cross-product of all valid test configurations (covered in Section 8.1.10).

### 8.2.2 Walkthrough of `test_matmul.py`

The canonical `test_matmul.py` is the simplest complete matmul test. Its execution proceeds through six phases:

**Phase 1: Parametrization.** The `@parametrize` decorator expands the test into a cross-product of four math fidelities (LoFi, HiFi2, HiFi3, HiFi4) and all format-aware matmul combinations. Each combination is a tuple of (`FormatConfig`, `DestAccumulation`, `(input_A_dimensions, input_B_dimensions)`). The `generate_format_aware_matmul_combinations()` function enforces hardware constraints:

- When `is_dest_acc_needed(fmt)` returns true (format outlier like Float16_b input to Float16 output), maximum tiles is 4 (32-bit dest register halves capacity).
- When `dest_acc=Yes`, maximum tiles is 4 regardless of format.
- Otherwise, maximum tiles is 8 (16-bit dest, `DstSync::SyncHalf`).

**Phase 2: Stimulus generation.** The `generate_stimuli()` function creates random input tensors sized to the requested dimensions in the specified data format. Both inputs use the same format (`formats.input_format`). The function returns flat tensors and tile counts. Stimuli are then tilized via `tilize_block()`, which rearranges row-major data into the face-order layout (F0, F1, F2, F3) expected by the hardware. Bfp8_b inputs skip Python-side tilization because the block-float exponent sharing requires format-aware handling.

**Phase 3: Golden generation.** `MatmulGolden.__call__()` performs matrix multiplication in PyTorch with format-aware rounding that models the FPU's limited-precision multiplication at the requested fidelity level. The `tilize=True` flag causes the golden output to be tilized after computation, matching the tile-order layout the hardware will produce.

**Phase 4: TestConfig construction.** This is where the Python-to-device bridge is built:

```python
configuration = TestConfig(
    "sources/matmul_test.cpp",       # C++ kernel source path
    formats,                          # FormatConfig (input/output formats)
    templates=[MATH_FIDELITY(math_fidelity)],   # Compile-time params
    runtimes=[
        NUM_FACES(),                  # Runtime params serialized to L1
        TILE_COUNT(matmul_dims.output_tile_cnt),
        CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
    ],
    variant_stimuli=StimuliConfig(
        tilized_A.flatten(), formats.input_format,  # Input A data + format
        tilized_B.flatten(), formats.input_format,  # Input B data + format
        formats.output_format,                       # Output format
        tile_count_A=tile_cnt_A,                    # Tile counts for L1 alloc
        tile_count_B=tile_cnt_B,
        tile_count_res=matmul_dims.output_tile_cnt,
    ),
    dest_acc=dest_acc,
)
```

The `TestConfig` constructor performs several critical tasks:
- Runs the **data format inference model** via `data_formats()`, which determines the eleven internal format fields based on the input/output formats and hardware architecture.
- Detects **format outliers** and auto-promotes `dest_acc` to `Yes` when required.
- Computes **tile sizes** for pack/unpack based on the data format (Bfp8_b = 68 words, Float32 = 256 words, others = 128 words).
- Generates the **runtime parameter struct** layout -- a C struct definition and a `struct.pack` format string that will be used to serialize parameters into L1 memory.

> **Cross-reference:** [Chapter 6, Section 1](../ch6_compilation/) describes the `build.h` header generation. [Chapter 4, Section 1](../ch4_matmul_api_calls/) documents the data format inference model.

**Phase 5: Execution.** The `configuration.run(workers_tensix_coordinates)` call triggers the full build-and-run pipeline (detailed in Sections 8.2.7 and 8.2.10 below).

**Phase 6: Validation.** The result tensor is read from L1, converted to a PyTorch tensor, and compared against the golden tensor using `passed_test()`, which applies format-appropriate tolerance thresholds (PCC-based comparison with format-dependent error budgets).

### 8.2.3 StimuliConfig -- L1 Memory Layout

`StimuliConfig` is the bridge between Python-side tensors and the hardware's L1 SRAM. Its constructor computes the contiguous memory layout:

```
L1 memory: [A tiles | B tiles | Result tiles]
            ^                                  ^
         buf_a_addr                        buf_res_addr + result_size
```

The base address depends on the build mode: `0x21000` for standard builds, `0x70000` for coverage builds (to avoid conflicting with the coverage data region). Note that runtime parameters use a separate, lower address (`0x20000` standard, `0x6E000` coverage). Each tile's byte size is computed from its data format using `calculate_tile_size_bytes()`. For example, Float16 tiles occupy `32 * 32 * 2 = 2048` bytes; Float32 tiles occupy `32 * 32 * 4 = 4096` bytes; Bfp8_b tiles occupy 68 * 4 = 272 bytes (shared exponent format).

The `write()` method dispatches to format-specific pack functions (`pack_fp16`, `pack_bfp16`, `pack_fp32`, `pack_bfp8_b`, etc.), then calls `write_to_device(location, addr, packed_data)` to DMA the bytes into L1. The `collect_results()` method reverses the process after execution:

```python
read_data = read_from_device(location, self.buf_res_addr, num_bytes=read_bytes_cnt)
res_from_L1 = unpack_res_tiles(read_data, self.stimuli_res_format, ...)
```

The runtime parameters struct (`RUNTIME_PARAMETERS`) includes buffer pointer arrays (`buffer_A[]`, `buffer_B[]`, `buffer_Res[]`) whose values are computed from the StimuliConfig addresses. These arrays allow the C++ kernel to address individual tiles in L1 by index.

### 8.2.4 Template vs. Runtime Parameters

The test harness maintains a strict separation between compile-time and runtime parameters:

**Template parameters** (passed via `templates=[]`) are emitted as `constexpr` or `#define` directives in `build.h`. They become part of the variant hash and trigger recompilation when changed. Examples:
- `MATH_FIDELITY(MathFidelity.HiFi4)` emits `constexpr MathFidelity MATH_FIDELITY = MathFidelity::HiFi4;`
- `DEST_SYNC(DestSync.Full)` emits `constexpr DstSync dest_sync = DstSync::SyncFull;`
- `THROTTLE_LEVEL(3)` emits `constexpr int THROTTLE_LEVEL = 3;`

**Runtime parameters** (passed via `runtimes=[]`) are serialized into the `RuntimeParams` struct and written to L1 before execution. They do not affect compilation. Examples:
- `TILE_COUNT(tile_cnt)` adds `std::uint32_t TILE_CNT;` to the struct
- `CRK_TILE_DIMM(ct, rt, kt)` adds `std::uint32_t CT_DIM; std::uint32_t RT_DIM; std::uint32_t KT_DIM;`
- `NUM_FACES(4, 4, 4)` adds face count fields

The C++ kernel accesses template parameters directly as constants and runtime parameters through the `params` reference: `params.CT_DIM`, `params.KT_DIM`, etc.

**Summary of parameters used across the matmul test suite:**

| Parameter Class | Type | Fields Set | Used By |
|:---|:---|:---|:---|
| `MATH_FIDELITY(fidelity)` | Template | `MATH_FIDELITY` | All matmul tests |
| `THROTTLE_LEVEL(level)` | Template | `THROTTLE_LEVEL` | math_matmul, unpack_matmul, perf |
| `DEST_SYNC(mode)` | Template | `dest_sync` | math_matmul, unpack_matmul |
| `STOCHASTIC_ROUNDING(mode)` | Template | `STOCHASTIC_RND` | unpack_matmul (sweeps Fpu/Pack/All), math_matmul (No only) |
| `APPROX_MODE(mode)` | Template | `APPROX_MODE` | matmul_and_unary_sfpu |
| `MATH_OP(op)` | Template | `SFPU_UNARY_OPERATION` | matmul_and_unary_sfpu |
| `NUM_FACES(n, nA, nB)` | Runtime | `num_faces`, `num_faces_A`, `num_faces_B` | All matmul tests |
| `TILE_COUNT(n)` | Runtime | `TILE_CNT` | All matmul tests |
| `CRK_TILE_DIMM(c, r, k)` | Runtime | `CT_DIM`, `RT_DIM`, `KT_DIM` | All matmul tests |
| `IN_TILE_DIMS(r0, c0, r1, c1)` | Runtime | `in0_tile_r_dim`, etc. | math_matmul, unpack_matmul |
| `DEST_INDEX(idx)` | Runtime | `DST_INDEX` | math_matmul, unpack_matmul |
| `PARTIAL_FACE(...)` | Runtime | `PARTIAL_FACE_A/B`, `PARTIAL_FACE_MATH/PACK` | math_matmul, unpack_matmul |
| `LOOP_FACTOR(n)` | Runtime | `LOOP_FACTOR` | perf tests |

### 8.2.5 Architecture Detection and Multi-Arch Support

The test harness automatically detects the connected chip architecture and configures the entire compilation and execution pipeline accordingly. `TestConfig.setup_arch()` queries the chip via `get_chip_architecture()` and sets:

| Setting | Wormhole | Blackhole | Quasar |
|---|---|---|---|
| `ARCH_COMPUTE` | `-mcpu=tt-wh-tensix` | `-mcpu=tt-bh-tensix` | `-mcpu=tt-bh-tensix` (fallback) |
| `ARCH_DEFINE` | `-DARCH_WORMHOLE` | `-DARCH_BLACKHOLE` | `-DARCH_QUASAR` |
| `ARCH_LLK_ROOT` | `tt_llk_wormhole_b0` | `tt_llk_blackhole` | `tt_llk_quasar` |
| `KERNEL_COMPONENTS` | `[unpack, math, pack]` | `[unpack, math, pack]` | `[unpack, math, pack, sfpu]` |
| Boot mode | BRISC (default) | BRISC (default) | TRISC only |

The Quasar architecture is notable for having four TRISC cores (adding a dedicated SFPU core) and requiring TRISC boot mode. The LLK headers themselves are architecture-specific: `tt_llk_blackhole/llk_lib/llk_math_matmul.h` contains Blackhole-specific address-mode calculations, while `tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h` contains Wormhole variants. The include path set by `TestConfig.INCLUDES` ensures the correct headers are selected at compile time.

### 8.2.6 Data Format Inference Model

A critical piece of the harness is the data format inference model (`helpers/data_format_inference.py`, called via `data_formats()`). Given user-specified input and output formats, it determines eleven internal format fields that the hardware pipeline needs:

1. **`unpack_A_src` / `unpack_B_src`** -- the format of data as it sits in L1 memory (matches the user's input format).
2. **`unpack_A_dst` / `unpack_B_dst`** -- the format of data after unpacker gasket conversion, as it will appear in Source A / Source B registers.
3. **`math`** -- the format the FPU operates in (Float16_b or Float32 depending on configuration).
4. **`pack_src`** -- the format of data in the destination register that the packer reads.
5. **`pack_dst`** -- the format of data after packer gasket conversion, as written to L1 (matches the user's output format).
6. **Scalar variants** (`unpack_S_src`, `unpack_S_dst`, `pack_S_src`, `pack_S_dst`) -- used for reduce operations and bias terms; set to default values for matmul.

The inference model also handles **format outliers**: certain input-to-output format conversions (e.g., Float16_b input to Float16 output, Bfp8_b input to Float16 output) require 32-bit destination accumulation because the format conversion cannot be done accurately in 16-bit. The model auto-promotes `dest_acc` for these cases and adjusts the internal format chain accordingly.

### 8.2.7 The `TestConfig` Compilation Pipeline in Detail

The `build_elfs()` method implements a sophisticated compilation pipeline:

1. **Shared artefacts** are built once per session: `brisc.elf` (the BRISC firmware) and optionally `coverage.o`. A `FileLock` prevents concurrent builds.

2. **Per-variant artefacts** are built in a variant-specific directory under `/tmp/tt-llk-build/sources/matmul_test.cpp/<sha256_hash>/`. The hash includes the test source name, all template parameters, format configuration, dest accumulation mode, boot mode, and profiler/coverage flags.

3. The **compilation command** for each TRISC is approximately:
   ```
   riscv-tt-elf-g++ -mcpu=tt-bh-tensix -O3 -std=c++17 -ffast-math
     -nostdlib -fno-use-cxa-atexit -Werror -Wall ...
     -DTENSIX_FIRMWARE -DENV_LLK_INFRA -DENABLE_LLK_ASSERT -DARCH_BLACKHOLE
     -DLLK_BOOT_MODE_BRISC -DRUNTIME_FORMATS -DCOMPILE_FOR_TRISC=1
     -DLLK_TRISC_MATH
     -I<variant_dir> -I../tt_llk_blackhole/llk_lib ...
     -T memory.blackhole.ld -T math.ld -T sections.ld
     -x c++ - -lc -o math.elf
   ```
   The source is piped via stdin (`-x c++ -`), combining `#include <matmul_test.cpp>` with `#include <trisc.cpp>` (the firmware entry point).

4. The **three-way compilation** (unpack, math, pack) is parallel via `ThreadPoolExecutor`, reducing build time by approximately 3x on multi-core hosts.

> **Cross-reference:** [Chapter 6, Section 2](../ch6_compilation/) covers the linker scripts and memory regions. [Chapter 6, Section 3](../ch6_compilation/) explains the separate text sections for each TRISC.

### 8.2.8 Variant Hashing and Build Caching

The test harness uses SHA-256 hashing to enable aggressive build caching. `generate_variant_hash()` collects all compilation-relevant fields from the `TestConfig` instance -- the test source name, template parameters, format configuration, dest accumulation mode, boot mode, profiler/coverage flags, and L1-to-L1 iteration count -- and hashes them into a 64-character hex string.

Fields that do not affect compilation are excluded from the hash:
- Runtime parameters (they do not change the compiled binary)
- Stimuli data (written to L1, not compiled)
- The variant hash itself

The hash determines the build directory path: `/tmp/tt-llk-build/sources/<test_name>/<hash>/`. If a `.build_complete` marker file exists at that path, the build is skipped entirely. A `FileLock` prevents race conditions when multiple pytest workers build the same variant concurrently.

This caching is critical for the parametrized tests. `test_matmul.py` generates hundreds of parameter combinations, but many share the same compiled binary (differing only in runtime parameters like tile counts and CRK dimensions). The hash collapses these into a smaller set of unique compilations.

### 8.2.9 How `test_math_matmul.py` Extends the Baseline

The `test_math_matmul.py` test introduces several extensions over the baseline:

1. **Throttle levels** -- values 1 through 5 insert progressively more NOP instructions between MVMUL operations, reducing effective compute throughput to between 73% and 33% of maximum. Throttle level 0 means no throttling (tiny tiles only run at level 0).

2. **Face-level data generation** -- instead of `generate_stimuli()`, it uses `generate_face_matmul_data()` which generates per-face data with correct zeroing for sub-4-face modes. In 2-face mode, input 0 (SrcB) uses faces f0,f1 (horizontal layout) while input 1 (SrcA) uses faces f0,f2 (vertical layout).

3. **L1 memory view conversion** -- `convert_to_l1_view()` rearranges the tilized tensor to match the actual L1 memory layout when tiles have non-standard dimensions, accounting for padding and face ordering.

4. **Per-tile comparison for sub-4-face modes** -- when `num_faces < 4`, only the active face elements within each tile are compared against the golden, ignoring padding in unused faces.

5. **Extended template parameters** -- `THROTTLE_LEVEL`, `DEST_SYNC` are compile-time. `PARTIAL_FACE`, `IN_TILE_DIMS`, `DEST_INDEX`, `UNPACK_TRANS_FACES`, `UNPACK_TRANS_WITHIN_FACE` are runtime.

### 8.2.10 Device Interaction via tt-exalens

The test harness communicates with the Tenstorrent hardware exclusively through the `ttexalens` Python library. The key functions used are:

- **`load_elf(elf_file, location, risc_name, ...)`** -- writes an ELF binary to the appropriate L1 text section for a specific RISC core. The `risc_name` parameter (`"trisc0"`, `"trisc1"`, `"trisc2"`) selects the target.
- **`write_to_device(location, address, data)`** -- writes raw bytes to L1 at a specified address. Used for runtime parameters and stimuli data.
- **`read_from_device(location, address, num_bytes)`** -- reads raw bytes from L1. Used for result collection and coverage data extraction.
- **`write_words_to_device(location, address, [word_list])`** -- writes 32-bit words to L1. Used for TRISC start addresses and BRISC commands.
- **`read_word_from_device(location, address)`** -- reads a single 32-bit word from L1. Used for mailbox polling.

ELF loading follows an architecture-specific protocol:
- **Wormhole:** ELFs are loaded while TRISCs are in reset. BRISC reads start addresses from L1 and releases TRISCs via a command protocol (`BriscCmd.UPDATE_START_ADDR_CACHE_AND_START` or `BriscCmd.START_TRISCS`).
- **Blackhole:** ELFs are loaded, then TRISC soft reset is deasserted directly via `set_tensix_soft_reset(0, [RiscCore.TRISC0], location)`.
- **Quasar:** Similar to Blackhole but with 4 TRISCs and `neo_id=0`.

> **Cross-reference:** [Chapter 7, Section 2](../ch7_host_execution/02_step_by_step_execution.md) provides the detailed ELF loading and boot sequence.

### 8.2.11 End-to-End Pytest Flow

```
pytest test_matmul.py
  |
  v
conftest.py::pytest_configure()
  -> TestConfig.setup_build()
  -> check_hardware_headers()
  -> init tt-exalens / simulator
  |
  v
conftest.py::pytest_sessionstart()
  -> send GO_BUSY ARC message
  |
  v
@parametrize generates test variants
  -> 4 fidelities x N format/dimension combos
  |
  v
For each variant:
  1. generate_stimuli()         -> random input tensors
  2. generate_tile_dims()       -> rt/ct/kt dimensions
  3. MatmulGolden()             -> expected output
  4. tilize_block()             -> tilize inputs for L1
  5. TestConfig(...)            -> build configuration object
  6. configuration.run()        -> compile + load + execute + read
  7. passed_test()              -> compare result vs golden
  |
  v
conftest.py::pytest_sessionfinish()
  -> send GO_IDLE ARC message
  -> combine perf reports
  -> process coverage artifacts (if --coverage)
```

### 8.2.12 Performance Test Harness (`PerfConfig`)

Performance tests use `PerfConfig` instead of `TestConfig`. The `PerfConfig` class extends `TestConfig` with multi-run-type support: a single test configuration is executed in five isolation modes (documented in Section 8.1.8).

Performance tests use a `LOOP_FACTOR` parameter (typically 16 for `perf_matmul` or 1024 for `perf_math_matmul`) that causes the kernel to repeat the operation multiple times within a single execution, providing stable cycle-count measurements. Results are collected via profiler data buffers at fixed L1 addresses (`0x16B000` through `0x16D000`, one per TRISC).

The `perf_report` fixture aggregates results across all parametrized combinations and generates a structured performance report.

### 8.2.13 Fused Test Patterns

Two tests demonstrate fused multi-operation pipelines:

**`test_matmul_unpack_tilize.py`** runs a two-pass pipeline using `L1_to_L1_iterations=2`:
- Pass 1: Unpack-tilize (row-major to tile-order in L1), datacopy to dest, pack to L1 intermediate buffers.
- Pass 2: Unpack-matmul from intermediate buffers, math matmul, pack result.

The two passes share the same TRISC ELFs but use different format configurations (one per L1-to-L1 iteration). Inter-TRISC synchronization uses hardware semaphores: the packer signals `semaphore::PACK_DONE` after writing tilized data, and the unpacker waits on that semaphore before starting the matmul unpack.

**`test_zzz_matmul_and_unary_sfpu.py`** (the `zzz_` prefix ensures it runs last) chains matmul with a unary SFPU operation. The matmul result in the destination register is packed to an intermediate L1 buffer, then unpacked and processed by an SFPU function (Abs, Celu, Cos, Hardsigmoid, Log, Reciprocal, Sin, Sqrt, Square). This validates that the destination register contents after matmul are correctly accessible to the SFPU pipeline.

> **Cross-reference:** [Chapter 5, Section 3](../ch5_compile_time_config/) describes how the three TRISCs coordinate via semaphores and the DstSync mechanism. [Chapter 2, Section 3](../ch2_data_formats_and_memory/) covers the destination register and its role in the SFPU data path.

### 8.2.14 Speed-of-Light Mode

When `SPEED_OF_LIGHT` is enabled, all runtime parameters are promoted to compile-time constants, stimuli addresses are hardcoded, and format configurations are baked in. The `RuntimeParams` struct becomes empty. This eliminates runtime overhead from parameter reads and enables the compiler to fully unroll loops and inline constants, producing the fastest possible kernel -- the "speed of light" for a given configuration.

### 8.2.15 Code Coverage Support

When `--coverage` is enabled, the harness adds `-fprofile-arcs -ftest-coverage -fprofile-info-section -DCOVERAGE` to the compilation flags and uses an extended linker script (`memory.blackhole.debug.ld`) that reserves additional L1 space for GCC coverage counters. After kernel execution, `read_coverage_data_from_device()` extracts the coverage stream from the `__coverage_start` symbol in each ELF, writing it to `.stream` files. The `process_coverage_run_artefacts()` function post-processes all streams using `riscv-tt-elf-gcov-tool merge-stream` and `lcov` to produce unified coverage reports.

Coverage builds use different L1 addresses: runtime parameters shift from `0x20000` to `0x6E000`, and stimuli shift from `0x21000` to `0x70000`, to avoid conflicting with the coverage data region.

### 8.2.16 The `conftest.py` Test Infrastructure

**File:** `tests/python_tests/conftest.py`

The `conftest.py` provides pytest infrastructure shared by all tests:

**Device Management:**
- `pytest_sessionstart`: sends `GO_BUSY` ARC message to activate the device.
- `pytest_sessionfinish`: sends `GO_IDLE`, combines perf reports, processes coverage artifacts, and stops the exalens server.
- `pytest_configure`: initializes ExaLens (device communication layer), sets up build directories, and validates hardware headers.

**Simulator Support:**
- `--run-simulator` option starts an ExaLens server backed by the tt-simulator instead of real hardware.
- `--reset-simulator-per-test` restarts the simulator between tests for full isolation.

**Test Execution Options:**

| CLI Option | Effect |
|:-----------|:-------|
| `--coverage` | Enables code coverage `.info` file generation |
| `--compile-producer` | Compile-only mode (produce ELFs, do not run) |
| `--compile-consumer` | Run-only mode (consume pre-compiled ELFs) |
| `--detailed-artefacts` | Extra debug flags in compilation |
| `--no-debug-symbols` | Omit `-g` to save disk space |
| `--speed-of-light` | Convert all runtime parameters to compile-time |
| `--logging-level` | Set loguru log level (TRACE through CRITICAL) |

**Worker Coordinates:**
- `workers_tensix_coordinates` fixture maps pytest-xdist worker IDs to Tensix core coordinates (row, col) via `divmod(int(worker_id[2:]), 8)`, enabling parallel test execution across multiple cores.

**Error Reporting:**
- `pytest_runtest_makereport` provides custom error formatting for `LLKAssertException` (hardware assertion hits), `TimeoutError` (Tensix hangs), and standard `AssertionError` (golden comparison failures).

### 8.2.17 The Validation Pipeline

Result validation uses `passed_test()` from `helpers/utils.py`, which computes Pearson Correlation Coefficient (PCC) between the golden and result tensors. The acceptance threshold is format-dependent:

- Float32: near-exact match expected
- Float16 / Float16_b: relaxed threshold accounting for 16-bit precision loss
- Bfp8_b: further relaxed for the 8-bit mantissa shared-exponent format
- With stochastic rounding: additional tolerance for non-deterministic rounding
- With L1-to-L1 iterations > 1: accumulated precision loss is factored in

For sub-4-face modes, each tile is validated independently by slicing out only the `in0_tile_r_dim * in1_tile_c_dim` active elements per tile, ignoring the zero-padded remainder of the 32x32 tile storage.

### 8.2.18 Key Takeaway: The Harness as Minimum Requirements Specification

The Python test harness serves a dual purpose. Beyond testing, it constitutes the most complete reference for the minimum host-side workflow required to run an LLK matmul kernel. The sequence -- architecture detection, format inference, `build.h` generation, RISC-V cross-compilation, runtime parameter serialization, stimuli write to L1, ELF loading, TRISC reset deassertion, mailbox polling, and result readback -- represents the irreducible set of host operations. Any alternative host framework (TT-Metal, a custom driver, etc.) must perform equivalent operations.

The fact that `TestConfig.run()` encapsulates this entire sequence in a single method call should not obscure its complexity. Under the hood, it involves:
- 1 header file generation (containing ~50 lines of format constants, struct definitions, and template parameters)
- 3 parallel RISC-V compilations (each ~2 seconds on a modern host)
- 1 L1 write of serialized runtime parameters (~100 bytes for a typical matmul)
- 2 L1 writes for input stimuli (~4 KB per 32x32 Float16 tile)
- 3 ELF loads via tt-exalens (each ~10 KB)
- 1 BRISC command (to release TRISCs from reset)
- Polling of 3 mailbox addresses until all TRISCs complete
- 1 L1 read for result data

This sequence, distilled from the test harness, is the practical answer to "what does it take to run a matmul on P150."

> **Cross-reference:** [Chapter 7, Section 2](../ch7_host_execution/02_step_by_step_execution.md) provides a step-by-step walkthrough of this execution sequence from the host perspective. [Chapter 1](../ch1_introduction/) introduces the conceptual model that this harness implements.

---

**Previous:** [8.1 -- Matmul Test Survey](./01_matmul_test_survey.md) | **Next:** [8.3 -- Alternatives to LLK](./03_alternatives_to_llk.md)
