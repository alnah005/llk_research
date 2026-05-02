# Chapter 8 -- Putting It All Together: Test Infrastructure and Alternatives

## 8.1 Matmul Test Survey

### 8.1.1 Scope and Organization of Tests

The tt-llk repository contains a comprehensive suite of matmul-related tests organized across two layers: Python test drivers under `tests/python_tests/` and C++ kernel sources under `tests/sources/`. Every test follows the same structural pattern -- a Python file parametrizes inputs, generates stimuli and golden references using PyTorch, constructs a `TestConfig` object pointing to a C++ kernel source, compiles the kernel for the target architecture, writes data to device L1 memory, executes the kernel on a Tensix core, reads results back, and asserts correctness against the golden tensor.

There are eleven matmul-related files in total: seven functional test drivers, two performance benchmarks, one helper module for sweep generation, and one Quasar-specific test.

### 8.1.2 Master Reference Table

The following table captures every matmul test variant, cross-referencing the Python driver, C++ kernel, tested features, distinguishing LLK API calls, and primary value as a reference.

| # | Python Driver | C++ Kernel | Formats | Tile Geometry | Special Features | Key LLK API Differences | Reference Value |
|---|---|---|---|---|---|---|---|
| 1 | `test_matmul.py` | `matmul_test.cpp` (119 lines) | Float16_b, Float16, Float32, Bfp8_b | Multi-tile (ct x rt x kt), 32x32 | Boot mode testing, dest acc toggle | Standard MOP-based `_llk_math_matmul_init_` / `_llk_math_matmul_` | Canonical starting point; simplest full-pipeline matmul |
| 2 | `test_math_matmul.py` | `math_matmul_test.cpp` (155 lines) | Float16_b, Float16, Float32 | Multi-tile + tiny tiles (1--16 rows) | Throttle 1--5, DstSync Full/Half, transpose, partial faces, dest index. Nightly. | `THROTTLE_LEVEL` template on init and exec, variable `dest_sync` | Most comprehensive math-layer test |
| 3 | `test_unpack_matmul.py` | `unpack_matmul_test.cpp` (151 lines) | Float16_b, Float16, Float32 | Multi-tile + tiny tiles | Stochastic rounding (Fpu, Pack, All), face modes (1/2/4). Nightly. | `_llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>()` in unpack | Primary stochastic rounding validation vehicle |
| 4 | `test_matmul_custom.py` | `matmul_custom_test.cpp` (123 lines) | Float16_b, Float16, Float32, Bfp8_b | Multi-tile, 32x32 | No-MOP experimental; Blackhole-only | `_llk_math_matmul_init_no_mop_` / `_llk_math_matmul_no_mop_` | MOP-free replay-based execution |
| 5 | `test_matmul_pack_untilize.py` | `matmul_pack_untilize_test.cpp` (90 lines) | Float16_b, Float16, Float32 (Bfp8_b output skipped) | Single tile 32x32 | Pack with untilize; row-major output | `_llk_pack_untilize_init_` / `_llk_pack_untilize_`; `_llk_math_reconfig_remap_` on BH | Row-major output path |
| 6 | `test_matmul_unpack_tilize.py` | `matmul_unpack_tilize_test.cpp` (197 lines) | Float16_b, Float16, Float32 (same in/out) | Single tile 32x32 | Two-pass fused: tilize then matmul; semaphore sync | `_llk_unpack_tilize_init_` / `_llk_unpack_tilize_`; `_llk_math_eltwise_unary_datacopy_`; semaphore ops | Fused multi-stage pipeline |
| 7 | `test_zzz_matmul_and_unary_sfpu.py` | `matmul_and_unary_sfpu_test.cpp` (160 lines) | Float16, Float16_b, Float32, Bfp8_b | Single tile 32x32 | Matmul then SFPU; skipped on both Blackhole and Wormhole | `_llk_math_eltwise_unary_sfpu_init_` / `_llk_math_eltwise_unary_sfpu_start_` | Fused matmul + activation |
| 8 | `perf_matmul.py` | `matmul_perf.cpp` (250 lines) | Float16_b, Float16, Float32, Bfp8_b | Multi-tile, kt_dim 1--32 | Isolation modes, profiler zones, LOOP_FACTOR(16) | `PERF_RUN_TYPE` compile-time; mock functions for isolation | Cycle-count measurement |
| 9 | `perf_math_matmul.py` | `math_matmul_perf.cpp` (276 lines) | Float16_b, Float16, Float32, Bfp8_b | Multi-tile + tiny tiles | Extended perf: tiny tiles, throttle, partial faces | Same perf infra + `IN_TILE_DIMS`, `PARTIAL_FACE`, `DST_INDEX` | Tiny-tile perf regression |
| 10 | `quasar/test_matmul_quasar.py` | `quasar/matmul_quasar_test.cpp` (134 lines) | Float16, Float16_b | Multi-tile, 32x32 | Quasar arch; TDMA descriptors; ImpliedMathFormat; DstSync Full/Half | `_llk_unpack_matmul_init_` / `_llk_unpack_matmul_`; `_llk_math_matmul_block_`; `_llk_pack_matmul_init_` / `_llk_pack_matmul_` | Forward-looking Quasar reference |

> **Note on stochastic rounding:** The `test_math_matmul.py` Python driver passes a `STOCHASTIC_ROUNDING` template parameter but only uses `StochasticRounding.No`. The C++ kernel `math_matmul_test.cpp` does not call `_llk_unpack_configure_stoch_rnd_`. Actual stochastic rounding sweeps (Fpu, Pack, All) are exclusive to `test_unpack_matmul.py`, whose C++ kernel (`unpack_matmul_test.cpp`) contains the `_llk_unpack_configure_stoch_rnd_<STOCHASTIC_RND>()` call at line 39.

### 8.1.3 The Two-Layer Test Architecture

Each matmul test consists of a Python driver and a C++ kernel source. The C++ source is a single file that contains three `run_kernel()` functions, each conditionally compiled under a different preprocessor guard:

- `#ifdef LLK_TRISC_UNPACK` -- the unpacker kernel, compiled as `unpack.elf` for TRISC0
- `#ifdef LLK_TRISC_MATH` -- the math kernel, compiled as `math.elf` for TRISC1
- `#ifdef LLK_TRISC_PACK` -- the packer kernel, compiled as `pack.elf` for TRISC2

The Python driver does not directly call these functions. Instead, it constructs a `TestConfig` object that references the C++ source file path as a string (e.g., `"sources/matmul_test.cpp"`). The `TestConfig.build_elfs()` method compiles the source three times with different `-DLLK_TRISC_*` defines, producing three separate ELF files. The `TestConfig.run()` method loads all three ELFs onto the target Tensix core and starts execution.

Each C++ kernel function takes a `RUNTIME_PARAMETERS params` argument, which is a macro expanding to `const struct RuntimeParams&`. The `RuntimeParams` struct is auto-generated by the Python harness in `build.h` and contains fields for tile sizes, format configurations, buffer pointers, tile counts, and operation-specific parameters (CRK dimensions, face counts, transpose flags, etc.). The struct layout matches a `struct.pack` format string used to serialize values from Python into binary data written to a fixed L1 address.

> **Cross-reference:** [Chapter 6](../ch6_compilation/) describes the `build.h` generation process. [Chapter 7](../ch7_host_execution/) covers the runtime parameter serialization and L1 write.

### 8.1.4 File Inventory

The complete list of matmul-related source files in the repository is:

**Python test drivers** (under `tests/python_tests/`):
- `test_matmul.py` -- baseline end-to-end matmul
- `test_math_matmul.py` -- extended math-layer matmul (nightly)
- `test_matmul_custom.py` -- Blackhole-only no-MOP variant
- `test_unpack_matmul.py` -- unpacker-focused matmul (nightly)
- `test_matmul_pack_untilize.py` -- matmul with pack-untilize
- `test_matmul_unpack_tilize.py` -- unpack-tilize into matmul
- `test_zzz_matmul_and_unary_sfpu.py` -- fused matmul + SFPU
- `perf_matmul.py` -- end-to-end performance
- `perf_math_matmul.py` -- math-layer performance
- `helpers/matmul_sweep.py` -- dimension/configuration sweep generator
- `quasar/test_matmul_quasar.py` -- Quasar-specific matmul

**C++ kernel sources** (under `tests/sources/`):
- `matmul_test.cpp` -- 119 lines, baseline three-TRISC matmul
- `math_matmul_test.cpp` -- 155 lines, extended with throttle/transpose/tiny-tiles
- `matmul_custom_test.cpp` -- 123 lines, no-MOP math variant
- `unpack_matmul_test.cpp` -- 151 lines, stochastic rounding + extended unpack
- `matmul_pack_untilize_test.cpp` -- 90 lines, pack-untilize output
- `matmul_unpack_tilize_test.cpp` -- 197 lines, two-pass fused pipeline
- `matmul_and_unary_sfpu_test.cpp` -- 160 lines, matmul + SFPU fusion
- `matmul_perf.cpp` / `math_matmul_perf.cpp` -- performance kernel variants (250 / 276 lines)
- `quasar/matmul_quasar_test.cpp` -- 134 lines, Quasar variant

**LLK matmul headers** (per architecture):
- `tt_llk_blackhole/llk_lib/llk_math_matmul.h` -- 676 lines, FPU matmul logic
- `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` -- unpacker matmul logic
- `tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h` -- no-MOP variant
- Equivalent headers exist under `tt_llk_wormhole_b0/` and `tt_llk_quasar/`

### 8.1.5 C++ Kernel Source Structure: `matmul_test.cpp`

The baseline `matmul_test.cpp` is a 119-line file that cleanly demonstrates the three-TRISC structure. Each `run_kernel()` function follows a strict pattern:

**TRISC0 (Unpack):**
1. `_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(...)` -- programs unpacker gaskets with source/destination data formats, face dimensions, face counts, and tile sizes for both inputs A and B.
2. `_llk_unpack_AB_matmul_init_<>(...)` -- initializes the matmul-specific unpack configuration: transpose mode, CRK tile dimensions, face dimensions, and partial-face flags.
3. Loop over K dimension: `_llk_unpack_AB_matmul_<>(...)` -- issues UNPACR instructions that move tiles from L1 into Source A and Source B registers with matmul-aware address striding (each K iteration advances the tile pointer by `CT_DIM` tiles in input B).

**TRISC1 (Math):**
1. `_llk_math_matmul_init_<MATH_FIDELITY>(...)` -- programs address modification registers and the MOP replay buffer for the specified fidelity level and tile dimensions.
2. `_llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>()` -- initializes the destination register synchronization protocol.
3. `_llk_math_hw_configure_<is_fp32_dest_acc_en>(...)` -- programs FPU format configuration.
4. `_llk_math_wait_for_dest_available_<DstSync::SyncHalf>()` -- waits for the destination register bank to be available (packer has finished reading the other bank).
5. Loop over K dimension: `_llk_math_matmul_<MATH_FIDELITY>(...)` -- executes the MOP (issuing MVMUL instructions) for each K iteration, accumulating results in the destination register.
6. `_llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>()` -- signals that the math phase is complete; the destination bank can be read by the packer.

**TRISC2 (Pack):**
1. `_llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(...)` -- programs packer gaskets with source/destination formats and tile sizes. The Blackhole variant has an additional template parameter for tilize mode.
2. `_llk_pack_init_<false, false, false>(...)` -- initializes packer strides and output format.
3. `_llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>()` -- initializes the destination register read-side synchronization.
4. `_llk_packer_wait_for_math_done_()` -- waits for TRISC1 to signal completion.
5. Loop over output tiles: `_llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, L1_ADDRESS(params.buffer_Res[i]))` -- packs each tile from the destination register to its L1 output buffer address.
6. `_llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>()` -- signals that the pack phase is complete.

> **Cross-reference:** [Chapter 4](../ch4_matmul_api_calls/) provides detailed documentation of each LLK function called in these kernels. [Chapter 3, Section 2](../ch3_kernel_architecture/) explains the DstSync::SyncHalf protocol.

### 8.1.6 The Custom No-MOP Matmul Variant

The `matmul_custom_test.cpp` is architecturally significant because it demonstrates that the MOP (Macro-Operation Processor) replay buffer is an optimization, not a requirement. The standard `llk_math_matmul.h` programs a replay buffer with a sequence of 16 `TTI_MVMUL` instructions that the hardware replays automatically via `ckernel_template::run()`. The custom no-MOP variant in `experimental/llk_math_matmul_custom_no_mop.h` instead manages fidelity phases and tile reuse under direct software control:

**Standard matmul (`matmul_test.cpp`):**
```cpp
#include "llk_math_matmul.h"
_llk_math_matmul_init_<MATH_FIDELITY>(...);   // Programs MOP
_llk_math_matmul_<MATH_FIDELITY>(...);         // Runs MOP via ckernel_template::run()
```

**No-MOP custom (`matmul_custom_test.cpp`):**
```cpp
#include "experimental/llk_math_matmul_custom_no_mop.h"
_llk_math_matmul_init_no_mop_<MATH_FIDELITY>(...);  // Loads replay buffer only
_llk_math_matmul_no_mop_<MATH_FIDELITY>(...);        // Calls lltt::replay() in software loop
```

The no-MOP path calls `matmul_configure_mop_custom()` which loads a replay buffer but does not program a `ckernel_template`. Instead, `_llk_math_matmul_no_mop_()` explicitly calls `lltt::replay()` in a software loop, managing fidelity phases and tile reuse (`CLR_A` / `CLR_B`) manually. The `_no_mop_` init variant still programs the same address modification registers (ADDR_MOD_0 through ADDR_MOD_5) -- the FPU face-traversal logic is identical. The difference is in execution only.

This test is Blackhole-only (`get_chip_architecture() == ChipArchitecture.WORMHOLE` triggers a skip) because the no-MOP experimental header is only provided under `tt_llk_blackhole/llk_lib/experimental/`. Even this "custom" variant still includes:

- `llk_unpack_AB_matmul.h` for the unpack phase
- `llk_pack.h` and `llk_pack_common.h` for the pack phase
- `ckernel_include.h`, `ckernel_ops.h`, and `cmath_common.h` for register definitions

This validates that matmul correctness does not depend on the MOP hardware -- the same results are produced whether the instruction sequence is replayed from hardware or issued from software.

> **Cross-reference:** [Chapter 5, Section 2](../ch5_compile_time_config/) documents the `TTI_MVMUL` instruction and MOP replay buffer programming.

### 8.1.7 Quasar-Specific Test

The Quasar architecture uses a fundamentally different unpack and pack model based on TDMA (Time-Division Memory Access) buffer descriptors. The unpack section uses `tdma_descriptor_t` structures:

```cpp
tdma_descriptor_t tdma_desc_src_a;
tdma_desc_src_a.buf_desc.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
tdma_desc_src_a.buf_desc.f.format = static_cast<uint8_t>(formats.unpack_A_src);
_configure_buf_desc_table_(tdma_desc_src_a.buf_desc_id, tdma_desc_src_a.buf_desc);
_llk_unpack_hw_configure_<ckernel::p_unpacr::UNP_B>(tdma_desc_src_a);
_llk_unpack_matmul_init_<UNPACK_TRANSPOSE_FACES>(...);
```

The math section uses `_llk_math_matmul_block_` (block-level matmul) and `_llk_math_set_dvalid_` (explicit dvalid signaling) instead of the standard `_llk_math_matmul_` and `_llk_math_dest_section_done_`. The pack section uses `_llk_pack_matmul_init_` and `_llk_pack_matmul_` instead of the generic pack functions. Synchronization uses `set_up_dest_dvalid_per_thread` instead of the SyncHalf/SyncFull mechanism.

The Quasar test uses `BootMode.TRISC` (no BRISC), introduces `ImpliedMathFormat`, and tests both `DstSync::Half` and `DstSync::Full`. Float16 and Float16_b only. All parameters are compile-time (templates list, empty runtimes).

> **Cross-reference:** [Chapter 3, Section 1](../ch3_kernel_architecture/) describes the BRISC vs. TRISC boot modes. The Quasar test is the only matmul test that boots directly via TRISC, reflecting the Quasar architecture's lack of BRISC boot path. Quasar LLK calls are not compatible with the P150 (Blackhole); this test exists for forward-looking architecture support.

### 8.1.8 Performance Benchmarks

Performance tests use `@pytest.mark.perf` and require the `perf_report` fixture rather than `workers_tensix_coordinates`. They use `PerfConfig` (a subclass of `TestConfig`) which extends the build/run pipeline with multi-run-type support. A single parametrized test case is executed in five isolation modes:

| Mode | Unpack | Math | Pack | Purpose |
|---|---|---|---|---|
| `L1_TO_L1` | Runs real | Runs real | Runs real | Full-pipeline end-to-end throughput |
| `UNPACK_ISOLATE` | Runs real | Mock | Returns early | Unpack-only performance |
| `MATH_ISOLATE` | Mock | Runs real | Returns early | Math-only performance |
| `PACK_ISOLATE` | Returns early | Returns early | Runs real | Pack-only performance |
| `L1_CONGESTION` | Runs real | Mock | Runs real | L1 bandwidth contention |

Mock functions (`_perf_unpack_matmul_mock`, `_perf_math_matmul_mock`) generate synthetic synchronization signals so the isolated engine sees realistic stall patterns. The `LOOP_FACTOR` parameter causes the kernel to repeat the matmul operation many times within a single ELF execution (16 for `perf_matmul`, 1024 for `perf_math_matmul`), ensuring that cycle-count measurements are dominated by steady-state throughput rather than one-time initialization overhead. Both tests wrap phases in profiler zones (`ZONE_SCOPED("INIT")`, `ZONE_SCOPED("TILE_LOOP")`) for cycle-accurate measurement.

`math_matmul_perf.cpp` additionally supports tiny tiles, partial faces, throttle levels, and configurable `DST_INDEX` -- combining the parameter space of `math_matmul_test.cpp` with profiling.

> **Cross-reference:** Section 8.2.12 provides detailed documentation of the `PerfConfig` class.

### 8.1.9 Global State and Synchronization Variables

Every matmul C++ kernel source begins with three global variables:

```cpp
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
```

These are not arbitrary -- they are required by the LLK infrastructure:

- **`unp_cfg_context`** -- tracks which unpacker configuration context (bank) is active. The unpacker hardware supports double-buffered configuration registers, allowing one configuration to be active while the next is being programmed. This variable selects the active bank (0 or 1).
- **`pack_sync_tile_dst_ptr`** -- used by the `DstSync` mechanism in the packer. It tracks which destination register tile the packer should read next, synchronized with the math thread.
- **`math_sync_tile_dst_index`** -- the math-side counterpart, tracking which destination register tile the math engine is writing to.

These variables must be defined at file scope because they are referenced by multiple LLK functions across the unpack/math/pack phases. They form part of the implicit contract between the kernel source and the LLK library.

Some tests define additional global constants. For example, `matmul_pack_untilize_test.cpp` defines:

```cpp
const std::uint32_t ct_dim  = 1;
const bool UNTILIZE         = true;
std::uint32_t face_size     = 128;
std::uint32_t tile_size     = 16 * 16 * 4;
const ckernel::DstSync sync = ckernel::DstSync::SyncHalf;
```

These compile-time constants override the template parameters that would normally come from `build.h`, allowing the simpler single-tile tests to avoid runtime parameter passing entirely.

### 8.1.10 Dimension Sweep Details

The dimension combinations generated by `generate_matmul_dimension_combinations()` produce a carefully constrained set of test cases. For `max_tiles=8` (the default for 16-bit destination with no accumulation), the generated combinations include:

| mt_dim | nt_dim | kt_dim | Input A | Input B | Output Tiles |
|--------|--------|--------|---------|---------|-------------|
| 1 | 1 | 1 | [32, 32] | [32, 32] | 1 |
| 1 | 1 | 2 | [32, 64] | [64, 32] | 1 |
| 1 | 1 | 3 | [32, 96] | [96, 32] | 1 |
| 1 | 1 | 4 | [32, 128] | [128, 32] | 1 |
| 2 | 1 | 1 | [64, 32] | [32, 32] | 2 |
| 1 | 2 | 1 | [32, 32] | [32, 64] | 2 |
| 2 | 2 | 1 | [64, 32] | [32, 64] | 4 |
| 2 | 4 | 1 | [64, 32] | [32, 128] | 8 |

The constraint is `mt_dim * nt_dim <= max_tiles`, ensuring the output tile count fits in a single destination register bank. The K dimension is varied independently (default range 1--4) because K does not increase the output tile count -- it only increases the number of accumulation iterations.

For tiny tiles, `generate_matmul_tiny_tiles_combinations()` produces configurations where input 0 has sub-32 row heights while maintaining 32-column width:

| in0_tile_r_dim | in0_tile_c_dim | in1_tile_r_dim | in1_tile_c_dim | Faces |
|----------------|----------------|----------------|----------------|-------|
| 1 | 32 | 32 | 32 | in0: 2-face, in1: 4-face |
| 2 | 32 | 32 | 32 | in0: 2-face, in1: 4-face |
| 4 | 32 | 32 | 32 | in0: 2-face, in1: 4-face |
| 8 | 32 | 32 | 32 | in0: 2-face, in1: 4-face |
| 16 | 32 | 32 | 32 | in0: 2-face, in1: 4-face |

In 2-face mode, input 0 uses a horizontal layout (faces f0, f1 -- 16 rows by 32 columns) where only `in0_tile_r_dim` rows contain data and the rest are zero. The math path sets `partial_face_math=true` when `in0_tile_r_dim < 16`, which changes the address-mode register programming in `matmul_configure_addrmod()`.

### 8.1.11 Experimental Headers

The `tt_llk_blackhole/llk_lib/experimental/` directory contains 13 alternative LLK implementations that are not part of the stable API:

| Header | Purpose |
|:-------|:--------|
| `llk_math_matmul_custom_no_mop.h` | MOP-free matmul via replay buffer (Section 8.1.6) |
| `llk_unpack_AB_matmul_custom.h` | Custom unpack path for matmul |
| `llk_math_eltwise_binary_custom.h` | Custom binary eltwise math |
| `llk_math_eltwise_unary_datacopy_custom.h` | Custom datacopy math |
| `llk_math_reduce_custom.h` | Custom reduce math |
| `llk_math_reduce_runtime_custom.h` | Runtime-configurable reduce |
| `llk_math_mul_reduce_scalar.h` | Scalar multiply-reduce |
| `llk_pack_custom.h` | Custom pack path |
| `llk_unpack_A_custom.h` | Custom single-operand unpack |
| `llk_unpack_AB_reduce_custom.h` | Custom reduce unpack |
| `llk_unpack_AB_reduce_custom_runtime.h` | Runtime-configurable reduce unpack |
| `llk_unpack_AB_sub_bcast_col_custom.h` | Custom column-broadcast subtract unpack |
| `llk_unpack_mul_reduce_scalar.h` | Scalar multiply-reduce unpack |

These headers are the mechanism for extending LLK without modifying the stable library. They follow the same API patterns but bypass the standard MOP programming, giving full control over instruction sequencing.

### 8.1.12 Fused Operation Configuration

**Config file:** `tests/python_tests/fuser_config/fpu_matmul.yaml`

The fuser config YAML defines multi-operation pipelines for code generation. The matmul fuser config specifies:

```yaml
operations:
  - src_a: "input_A"
    src_b: "input_B"
    output: "matmul_out"
    src_a_dims: [256, 256]
    src_b_dims: [256, 256]
    output_dims: [256, 256]
    input_format: "Float16_b"
    output_format: "Float16_b"
    math:
      - type: "Fpu"
        operation: "Matmul"
        unpacker: "MatmulUnpacker"
    packer: "Packer"
    math_fidelity: "HiFi2"
    block_size: [64, 128]
```

This declarative format drives the fused test code generator, which produces C++ kernel code from YAML specifications. The `block_size` field (`[64, 128]`) controls the tile blocking factor for the generated kernel, corresponding to 2x4 tile blocks.

### 8.1.13 Architectural Coverage Matrix

Across all tests, the following architectural features are exercised:

| Feature | Tests That Cover It |
|---|---|
| Multi-tile matmul (ct_dim > 1 or rt_dim > 1) | test_matmul, test_math_matmul, test_unpack_matmul, perf_matmul |
| K-dimension accumulation (kt_dim > 1) | All except pack_untilize and unpack_tilize (single-tile) |
| Tiny tiles (in0_tile_r_dim < 32) | test_math_matmul, test_unpack_matmul, perf_math_matmul |
| Transpose (faces or within-face) | test_math_matmul, test_unpack_matmul |
| Stochastic rounding (Fpu, Pack, All modes) | test_unpack_matmul |
| Dest accumulation (32-bit dest) | All tests |
| DstSync::Full | test_math_matmul, perf_math_matmul, quasar test |
| Throttle levels | test_math_matmul, perf_math_matmul |
| Pack untilize | test_matmul_pack_untilize |
| Unpack tilize + matmul fusion | test_matmul_unpack_tilize |
| Matmul + SFPU fusion | test_zzz_matmul_and_unary_sfpu (skipped on BH/WH) |
| MOP-free (custom no-MOP) | test_matmul_custom (Blackhole only) |
| Quasar architecture | quasar/test_matmul_quasar |

### 8.1.14 What Is Not Tested

The following configurations do not appear in the test suite:

- **16x16 by 16x16 matmul** -- explicitly rejected by `LLK_ASSERT` in `llk_math_matmul.h` ("16x16 by 16x16 matmul is not supported"). No dedicated math path exists for single-face-by-single-face multiplication.
- **Transpose with input 1 dimensions 32x16** -- also rejected by assertion ("Transpose with input 1 dimensions 32x16 not supported").
- **Multi-core matmul** -- all tests target a single Tensix core. Multi-core tiling and data distribution are the responsibility of higher-level frameworks (see Section 8.3).
- **Bfp8_b output with pack untilize** -- explicitly skipped ("Pack untilize does not support Bfp8_b").
- **Wormhole custom no-MOP matmul** -- only `matmul_custom_test.cpp` exists, and it is Blackhole-specific.

### 8.1.15 Choosing the Right Reference

| Goal | Start Here | Section |
|:-----|:-----------|:--------|
| Minimal working matmul | `matmul_test.cpp` | 8.1.5 |
| Custom inner loop / no MOP | `matmul_custom_test.cpp` | 8.1.6 |
| Tiny tiles or transpose | `math_matmul_test.cpp` | Variant #2 in master table |
| Stochastic rounding validation | `unpack_matmul_test.cpp` | Variant #3 in master table |
| Row-major output | `matmul_pack_untilize_test.cpp` | Variant #5 in master table |
| Row-major input | `matmul_unpack_tilize_test.cpp` | Variant #6 in master table |
| Performance measurement | `matmul_perf.cpp` | 8.1.8 |
| Fused matmul + activation | `matmul_and_unary_sfpu_test.cpp` | Variant #7 in master table |
| Multi-pass pipeline with semaphores | `matmul_unpack_tilize_test.cpp` | Variant #6 in master table |

For a first-time P150 matmul, start with `matmul_test.cpp`. It is the simplest kernel that exercises all three pipeline stages with the minimum number of LLK calls, and its Python driver demonstrates the complete end-to-end flow from stimuli generation through golden comparison.

---

**Previous:** [Chapter 7 -- What TT-Metal Replaces](../ch7_host_execution/03_what_tt_metal_replaces.md) | **Next:** [8.2 -- Python Test Harness](./02_python_test_harness.md)
