# Running and Debugging

This section covers environment setup, test execution, custom pytest flags, build artifact locations, L1 memory layouts, and code coverage collection.

## Environment Setup

Before running tests, the SFPI compiler toolchain, platform headers, and Python dependencies must be installed. The setup procedure depends on your environment.

### LLK Docker Image

If you are using the official `tt-llk` Docker image:

```bash
cd tests/
./setup_testing_env.sh
```

### External Environment

If you are using a different Docker image or a bare-metal setup:

```bash
cd tests/
source ./setup_external_testing_env.sh
```

The `source` command is required so that environment variables (compiler paths, Python virtualenv activation) persist in your current shell session.

Both scripts install the latest [SFPI](https://github.com/tenstorrent/sfpi) compiler version and all necessary platform headers. When the environment is correctly initialized, the infrastructure auto-detects the underlying hardware (Wormhole, Blackhole, or derivative).

### Requirements

The test host must have a Tenstorrent device installed:

- **Wormhole** (or any derivative)
- **Blackhole** (or any derivative)

The device must be flashed with the original firmware.

## Running Tests

### Basic Invocation

Navigate to `tests/` or `tests/python_tests/` and run:

```bash
pytest test_my_kernel.py
```

This compiles and executes all variants serially.

### Parallel Compilation and Execution

For tests with many variants, split the process into a compile phase and an execution phase to leverage parallelism:

```bash
# Phase 1: Compile all variants using 10 CPU cores
pytest --compile-producer -n 10 -x test_my_kernel.py

# Phase 2: Execute compiled variants on 10 Tensix tiles in parallel
pytest --compile-consumer -n 10 -x test_my_kernel.py
```

The `-n` flag controls the parallelism level (uses `pytest-xdist`). The `-x` flag stops on the first failure. On Wormhole and Blackhole, each parallel worker executes on a separate Tensix tile.

Pre-compiled tests can be re-executed as many times as needed without recompilation. This is useful when iterating on runtime parameters or stimuli generation functions. However, if any compile-time parameter changes (template parameters, format configuration, or other `TestConfig` arguments that affect `build.h`), you must recompile by running the `--compile-producer` command again. Failing to do so typically results in a `TTException` about missing ELF files or a stimuli assertion error.

## Custom Pytest Flags

The test infrastructure exposes several custom command-line flags:

| Flag | Description |
|------|-------------|
| `--coverage` | Compile with coverage counters, link with debug L1 layout, and collect coverage data after kernel execution. Coverage data is stored per-variant and merged into `/tmp/tt-llk-build/merged_coverage.info`. |
| `--compile-producer` | Only compile enumerated test variants (do not execute). |
| `--compile-consumer` | Only execute already-compiled test variants. If `--coverage` is also passed, merges coverage data at the end. |
| `--speed-of-light` | Treat all parameters passed to `TestConfig` or `PerfConfig` as compile-time arguments. |
| `--test-order-file <path>` | Execute tests in the exact order listed in the file. Useful for debugging flaky tests. |
| `--skip-codegen` | Skip C++ code generation for fused tests and use existing files. |
| `--no-debug-symbols` | Compile ELFs without debug symbols (`-g` flag) to save disk space. |
| `--logging-level <LEVEL>` | Set loguru log level. Valid values: `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Overrides the `LOGURU_LEVEL` environment variable. Default: `INFO`. |
| `--detailed-artefacts` | Add `-save-temps=obj -fdump-tree-all -fdump-rtl-all -v` compilation flags for compiler-level debug information. Useful when debugging SFPI compiler issues. |

## Build Artifacts

The default build directory is `/tmp/tt-llk-build/`. Artifacts are organized to mirror the `test_name` path provided to `TestConfig` or `PerfConfig`:

```
/tmp/tt-llk-build/
  sources/
    eltwise_binary_test/
      <variant_hash_1>/
        build.h            # Auto-generated header for this variant
        unpack.elf         # Compiled Unpack kernel
        math.elf           # Compiled Math kernel
        pack.elf           # Compiled Pack kernel
      <variant_hash_2>/
        build.h
        unpack.elf
        math.elf
        pack.elf
  merged_coverage.info     # Merged coverage data (when --coverage is used)
```

The variant hash is a deterministic hash of all compile-time arguments. Variants with identical compile-time configuration share a single set of ELFs. The `build.h` file in each variant directory is a human-readable record of the exact configuration used.

Path-related static variables on `TestConfig` are initialized during `pytest_configure` when `TestConfig.setup_build` is called:

| Variable | Description |
|----------|-------------|
| `TestConfig.DEFAULT_ARTEFACTS_PATH` | `/tmp/tt-llk-build/` |
| `TestConfig.ARTEFACTS_DIR` | Root artifacts directory |
| `TestConfig.SHARED_DIR` | Shared build artifacts |
| `TestConfig.SHARED_OBJ_DIR` | Compiled object files |
| `TestConfig.SHARED_ELF_DIR` | Compiled ELF files |
| `TestConfig.COVERAGE_INFO_DIR` | Per-variant coverage data |
| `TestConfig.LLK_ROOT` | Root of the tt-llk repository |
| `TestConfig.TOOL_PATH` | SFPI toolchain location |
| `TestConfig.GXX` | SFPI C++ compiler path |
| `TestConfig.OBJDUMP` | SFPI objdump path |
| `TestConfig.OBJCOPY` | SFPI objcopy path |
| `TestConfig.GCOV` | SFPI gcov path |
| `TestConfig.GCOV_TOOL` | SFPI gcov tool path |

## L1 Memory Layouts

Wormhole and Blackhole have different L1 memory layouts because their RISC-V cores have different local-memory sizes (e.g., Wormhole TRISC local memory is 2K while Blackhole TRISC local memory is 4K). The tables below show the Wormhole layout; Blackhole differences are noted after each table. Quasar is similar to Wormhole except that it lacks a BRISC core, so no space is reserved for a BRISC ELF. The infrastructure provides two layout variants for each architecture: **performance** and **debug**.

### Performance Layout

The performance layout uses local data memory (L0) of RISC-V cores for stack and data, yielding better runtime performance. Linker scripts are at `tests/helpers/ld/memory.*.ld` (e.g., `memory.wormhole.ld`, `memory.blackhole.ld`).

**TRISC L0 (per core):**

| Address Range | Content |
|--------------|---------|
| 0xFFB00000 - 0xFFB00700 | Local data memory |
| 0xFFB00700 - 0xFFB00800 | Stack |

**Wormhole L1 (performance):**

| Address Range | Content |
|--------------|---------|
| 0x00000000 - 0x00003FFF | BRISC code (16K) |
| 0x00004000 - 0x00004FFF | BRISC C-runtime data (4K) |
| 0x00005000 - 0x00008FFF | TRISC0 (Unpack) code (16K) |
| 0x00009000 - 0x000097FF | TRISC0 (Unpack) C-runtime data (2K) |
| 0x00009800 - 0x0000D7FF | TRISC1 (Math) code (16K) |
| 0x0000D800 - 0x0000DFFF | TRISC1 (Math) C-runtime data (2K) |
| 0x0000E000 - 0x00011FFF | TRISC2 (Pack) code (16K) |
| 0x00012000 - 0x000127FF | TRISC2 (Pack) C-runtime data (2K) |
| 0x00020000 - 0x000203FF | Runtime arguments struct (1K) |
| 0x00021000 - 0x00169FFF | Stimuli space |
| 0x0016A000 - 0x0016AFF3 | Performance counters data |
| 0x0016AFF4 - 0x0016AFFF | Profiler barrier |
| 0x0016B000 - 0x0016B3FF | TRISC0 (Unpack) perf counter memory |
| 0x0016C000 - 0x0016C3FF | TRISC1 (Math) perf counter memory |
| 0x0016D000 - 0x0016D3FF | TRISC2 (Pack) perf counter memory |
| 0x0016DFF0 - 0x0016DFFB | TRISC[0,1,2] start addresses buffer (WH only) |

**Blackhole L1 differences (performance):** Blackhole uses larger C-runtime data regions (8K for BRISC, 4K per TRISC) because `BRISC_LOCAL_MEM_LENGTH = 8K` and `TRISC_LOCAL_MEM_LENGTH = 4K` (versus 4K and 2K on Wormhole). This shifts all subsequent regions to higher addresses. Additionally, Blackhole has no NCRISC IMEM. The RUNTIME_ARGS region is at the same address (0x20000) with the same 1K size. See `memory.blackhole.ld` for exact addresses.

### Debug Layout

The debug layout places everything in L1 (no local data memory usage), enabling GCOV coverage collection at the cost of runtime performance. Linker scripts are at `tests/helpers/ld/memory.*.debug.ld` (e.g., `memory.wormhole.debug.ld`, `memory.blackhole.debug.ld`).

**TRISC L0 (per core, debug):**

| Address Range | Content |
|--------------|---------|
| 0xFFB00000 - 0xFFB00800 | Stack |

**Wormhole L1 (debug):**

| Address Range | Content |
|--------------|---------|
| 0x00000000 - 0x00007FFF | BRISC code (32K) |
| 0x00008000 - 0x0000FFFF | BRISC data memory (32K) |
| 0x00010000 - 0x00011FFF | BRISC GCOV memory (8K) |
| 0x00012000 - 0x00021FFF | TRISC0 (Unpack) code (64K) |
| 0x00022000 - 0x00029FFF | TRISC0 (Unpack) data memory (32K) |
| 0x0002A000 - 0x00031FFF | TRISC0 (Unpack) GCOV memory (32K) |
| 0x00032000 - 0x00041FFF | TRISC1 (Math) code (64K) |
| 0x00042000 - 0x00049FFF | TRISC1 (Math) data memory (32K) |
| 0x0004A000 - 0x00051FFF | TRISC1 (Math) GCOV memory (32K) |
| 0x00052000 - 0x00061FFF | TRISC2 (Pack) code (64K) |
| 0x00062000 - 0x00069FFF | TRISC2 (Pack) data memory (32K) |
| 0x0006A000 - 0x00071FFF | TRISC2 (Pack) GCOV memory (32K) |
| 0x0006E000 - 0x0006E3FF | Runtime arguments struct (1K) |
| 0x00070000 - 0x00169FFF | Stimuli space |
| 0x0016A000 - 0x0016AFF3 | Performance counters data |
| 0x0016AFF4 - 0x0016AFFF | Profiler barrier |
| 0x0016B000 - 0x0016D3FF | Perf counter memory (same as performance layout) |
| 0x0016DFF0 - 0x0016DFFB | TRISC[0,1,2] start addresses buffer (WH only) |

**Blackhole L1 differences (debug):** Blackhole debug uses the same code and data region sizes as Wormhole debug, but GCOV regions are smaller (8K instead of 32K). Additionally, Blackhole debug adds per-TRISC RUNTIME_ARGS regions (1K each) between each TRISC's data memory and GCOV memory. The global RUNTIME_ARGS is also at 0x6E000 with 1K size. See `memory.blackhole.debug.ld` for exact addresses.

Key addresses are abstracted by these Python/C++ variables:

| Variable | Location | Purpose |
|----------|----------|---------|
| `StimuliConfig.STIMULI_L1_ADDRESS` | `stimuli_config.py` | Start of stimuli space |
| `RUNTIME_PARAMS_TYPE __runtime_args_start[]` | `trisc.cpp` | Runtime arguments in C++ |
| `TestConfig.RUNTIME_ADDRESS` | `test_config.py` | Runtime arguments in Python |
| `ProfilerConfig.THREAD_BUFFER` | `profiler.py` | Per-TRISC perf counter memory |
| `TestConfig.TRISC_START_ADDRS` | `test_config.py` | TRISC start addresses buffer (WH only) |

## Code Coverage

Coverage tracks which lines of LLK API code are exercised during kernel execution. It uses GCC's GCOV mechanism, adapted for the bare-metal RISC-V environment.

### Collecting Coverage

Pass `--coverage` to both the compilation and execution phases:

```bash
# Compile with coverage instrumentation
pytest --compile-producer --coverage -n 10 -x test_my_kernel.py

# Execute and collect coverage data
pytest --compile-consumer --coverage -n 10 -x test_my_kernel.py
```

When `--coverage` is active, the infrastructure:

1. Adds coverage counters to the kernel binary.
2. Generates `.gcno` counter tables for every ELF file.
3. Links every ELF with the debug L1 layout (to accommodate GCOV memory regions).
4. Pulls coverage data from the device after kernel execution.
5. Merges all per-variant coverage data into `/tmp/tt-llk-build/merged_coverage.info`.

Coverage cannot be collected on performance tests because the instrumentation overhead would invalidate the timing measurements.

### Generating the HTML Report

After collecting coverage data, run the report generator:

```bash
cd tests/
./generate_coverage_report.sh
```

This produces an HTML report in `coverage_report/` (two directories above `tests/` -- i.e., next to the `tt-llk` repo root). Open it with a browser or use the [Live Server VS Code extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) for convenient browsing. The report shows per-file, per-folder, and per-line coverage information.

---

**Next:** [`performance_testing.md`](./performance_testing.md)
