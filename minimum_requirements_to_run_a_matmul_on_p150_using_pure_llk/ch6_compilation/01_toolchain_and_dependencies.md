# Chapter 6: Toolchain, Compilation Pipeline, and ELF Generation

## 6.1 Toolchain and Dependencies

Building LLK matmul kernels for the P150 (Blackhole architecture) requires a cross-compilation toolchain, hardware-specific headers, system packages, and Python libraries. The `tests/setup_testing_env.sh` script automates most of the provisioning, but understanding each dependency is essential for reproducing the build outside the test infrastructure. This section complements the three-thread compilation pattern introduced in Chapter 3, Section 1 and the preprocessor / `build.h` generation discussed in Chapter 5, Section 2.

---

### 6.1.1 The SFPI Cross-Compiler Toolchain

The core toolchain is a RISC-V GCC cross-compiler from the [tenstorrent/sfpi](https://github.com/tenstorrent/sfpi) repository. It targets the Tensix RISC-V cores with custom extensions for Tenstorrent's SFPU (Special Function Processing Unit). The toolchain binaries reside under `tests/sfpi/compiler/bin/` after installation.

#### Toolchain Binaries

| Binary | Purpose |
|:-------|:--------|
| `riscv-tt-elf-g++` | C++ cross-compiler for Tensix cores. Supports `-mcpu=tt-bh-tensix` (compute/TRISC cores) and `-mcpu=tt-bh` (BRISC). |
| `riscv-tt-elf-objdump` | Disassembler for inspecting generated ELFs. Used during debugging to verify instruction sequences. |
| `riscv-tt-elf-objcopy` | Binary manipulation tool. Used to extract profiler metadata sections from ELFs (`-O binary -j .profiler_meta`). |
| `riscv-tt-elf-gcov` | Code coverage data processor. Used only when building with `-fprofile-arcs -ftest-coverage`. |
| `riscv-tt-elf-gcov-tool` | Coverage stream merging tool (`merge-stream` subcommand). Processes coverage data read back from device memory. |

#### Toolchain Path Resolution

The Python test infrastructure resolves these paths in `TestConfig.setup_paths()`:

```python
# tests/python_tests/helpers/test_config.py
TestConfig.TOOL_PATH = TestConfig.LLK_ROOT / "tests/sfpi/compiler/bin"
TestConfig.GXX     = str((TestConfig.TOOL_PATH / "riscv-tt-elf-g++").absolute())
TestConfig.OBJDUMP = str((TestConfig.TOOL_PATH / "riscv-tt-elf-objdump").absolute())
TestConfig.OBJCOPY = str((TestConfig.TOOL_PATH / "riscv-tt-elf-objcopy").absolute())
```

Every compilation command uses these absolute paths. There is no reliance on `$PATH` or system-installed cross-compilers.

#### Installation via `setup_testing_env.sh`

The setup script reads `tests/sfpi-info.sh` -- a helper script that sources the data file `tests/sfpi-version` -- to determine the correct SFPI release version and download URL. The `sfpi-version` file is the single source of truth for version metadata:

```
sfpi_repo='https://github.com/tenstorrent/sfpi'
sfpi_version='7.36.0'
sfpi_build='441'
sfpi_hashtype='sha256'
sfpi_x86_64_debian_txz_hash='0815caa286eac3c6244f27f69edfe4747cca2c66bd59dc9ca2da8dbad7d669ae'
```

The installation process:

1. Checks if the currently-installed version (stored in `tests/sfpi/sfpi.version`) matches the required version.
2. Downloads the release tarball from the tenstorrent/sfpi GitHub releases.
3. Verifies the SHA-256 hash against the value in `sfpi-version`.
4. Extracts the toolchain to `tests/sfpi/`.
5. Writes the version string to `tests/sfpi/sfpi.version`.

If the correct version is already installed, the script exits early with no download.

#### Custom RISC-V Extensions

The `riscv-tt-elf-g++` compiler is not a standard RISC-V toolchain. It includes Tenstorrent-specific ISA extensions that generate instructions the standard `riscv64-unknown-elf-gcc` would reject:

- **SFPU instructions:** `SFPMOV`, `SFPMAD`, `SFPEXEXP`, `SFPSETCC`, etc. These operate on the Special Function Processing Unit's 32-wide SIMD vector registers (LREGs). The compiler maps C++ operations in `sfpi::vFloat` / `sfpi::vInt` types to these hardware instructions.
- **Tensix intrinsics:** The `TTI_*` macro family (e.g., `TTI_MVMUL`, `TTI_ZEROACC`, `TTI_SFPENCC`, `TTI_UNPACR`) generates Tensix-specific instruction encodings that are unknown to generic RISC-V assemblers.
- **MOP triggers:** The `TT_MOP()` and `TT_MOP_CFG()` macros generate instructions that interact with the hardware Matrix Operation Pipeline.

The `-mcpu=tt-bh-tensix` flag enables the full Tensix extension set. The `-mcpu=tt-bh` flag (used for BRISC) enables only the base extensions without compute instructions, because BRISC does not have access to the Tensix compute engine.

#### ELF Format

The toolchain produces 32-bit little-endian RISC-V ELFs (`elf32-littleriscv`). These are loaded by the host-side `ttexalens.load_elf()` function, which parses the ELF program headers to determine load addresses and writes each segment to the appropriate L1 memory region via the NOC.

#### Verifying the Toolchain

After installation, verify the toolchain supports Blackhole targets:

```bash
tests/sfpi/compiler/bin/riscv-tt-elf-g++ --version
tests/sfpi/compiler/bin/riscv-tt-elf-g++ -mcpu=tt-bh-tensix -x c++ -E - < /dev/null
```

The `-mcpu=tt-bh-tensix` flag will fail if the toolchain does not support the Blackhole Tensix ISA extensions -- the quickest way to catch a version mismatch.

---

### 6.1.2 LLK Library Headers

The LLK library itself (`tt_llk_blackhole/`) provides the API headers that kernel code includes. These are not downloaded -- they are part of the tt-llk repository. The key directories:

| Directory | Contents |
|:----------|:---------|
| `tt_llk_blackhole/llk_lib/` | Top-level LLK API headers: `llk_unpack_AB_matmul.h`, `llk_math_matmul.h`, `llk_pack.h`, `llk_pack_common.h`, `llk_math_common.h`, `llk_unpack_common.h`. These are the headers included by kernel source files inside `#ifdef` blocks. |
| `tt_llk_blackhole/common/inc/` | Core ckernel infrastructure: `ckernel.h` (semaphores, register access, memory primitives), `ckernel_defs.h` (tile constants, enums), `ckernel_template.h` (MOP programming), `ckernel_globals.h` (global state variables). |
| `tt_llk_blackhole/common/inc/sfpu/` | SFPU operation headers for transcendental functions. |
| `tt_llk_blackhole/instructions/` | Low-level instruction encoding macros (`TT_OP_*` builders). |
| `common/` | Cross-architecture headers shared between Blackhole and Wormhole: `llk_assert.h`, `tensor_shape.h`. |

The test infrastructure's include headers (`tests/helpers/include/`) provide the glue between the LLK library and the test framework:

| Header | Purpose |
|:-------|:--------|
| `boot.h` | CRT startup (`do_crt0()`), device setup (`device_setup()`), soft reset control, mailbox helpers. Central to every kernel's startup. |
| `params.h` | Includes `build.h` and defines the `RuntimeParams` access pattern. |
| `operand.h` | Operand address definitions for L1 tile buffers. |
| `ckernel_helper.h` | Wrapper utilities around ckernel primitives. |
| `perf.h` | Performance counter addresses and configuration. |
| `profiler.h` | Profiler zone macros (`ZONE_SCOPED`), barrier synchronization. |
| `counters.h` | Performance counter L1 memory layout. |
| `llk_sfpu_types.h` | SFPU type definitions shared between Python and C++. |

> **Cross-reference:** Chapter 3, Section 1 documents which headers each `#ifdef` block requires. Chapter 5, Section 2 covers the auto-generated `build.h` that `params.h` includes.

---

### 6.1.3 Hardware-Specific Headers from tt-metal

The LLK kernel code depends on a set of low-level hardware definition headers maintained in the [tt-metal](https://github.com/tenstorrent/tt-metal) repository. These are downloaded by `setup_testing_env.sh` and placed in `tests/hw_specific/blackhole/inc/`.

#### Downloaded Headers

| Header | Content |
|:-------|:--------|
| `cfg_defines.h` | Configuration register bitfield macros. Defines `*_ADDR32` address constants and `*_MASK` bitmask constants for all Tensix config registers. Enables the Read-Modify-Write (RMW) patterns used throughout the LLK codebase (e.g., `cfg_ptr[DISABLE_RISC_BP_Disable_main_ADDR32] |= DISABLE_RISC_BP_Disable_main_MASK`). |
| `tensix.h` | Core Tensix hardware definitions. Base addresses for register blocks (`TENSIX_CFG_BASE`), debug register offsets (`RISCV_DEBUG_REG_SOFT_RESET_0`, `RISCV_DEBUG_REG_DEST_CG_CTRL`), and hardware constants. |
| `tensix_types.h` | Type definitions and enums for Tensix hardware: data formats, tile shapes, destination register parameters. Included in every `build.h` via the auto-generated header (see Chapter 5, Section 2). |
| `dev_mem_map.h` | Device memory map constants. Defines base addresses for L1 regions, mailbox locations, and firmware structures. |
| `core_config.h` | Per-core configuration parameters. Core-specific memory sizes, TRISC reset PC addresses, and soft-reset mask values. |
| `risc_attribs.h` | Downloaded separately into `tests/hw_specific/blackhole/inc/internal/`. Defines the `tt_reg_ptr` attribute qualifier, `TT_ALWAYS_INLINE`, and section placement macros used by `boot.h` and firmware code. |

#### Download Source

All headers are fetched from the tt-metal `main` branch:

```
https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/
  tt_metal/hw/inc/internal/tt-1xx/blackhole/<header>
```

The `risc_attribs.h` file comes from a different path:

```
https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/
  tt_metal/hw/inc/internal/risc_attribs.h
```

Stamp files (`.headers_downloaded`, `.sfpu_downloaded`) prevent redundant downloads. The script verifies that all expected header files physically exist even when the stamp is present, recovering gracefully from partial downloads.

#### SFPU Headers from tt-metal

In addition to the core hardware headers, the setup script downloads SFPU implementation headers from tt-metal's `ckernels/blackhole/metal/llk_api/llk_sfpu/` directory. These are placed in `tests/hw_specific/blackhole/metal_sfpu/` via a sparse `git clone`:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/tenstorrent/tt-metal.git tt-metal-temp
git -C tt-metal-temp sparse-checkout set \
    tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu
cp tt-metal-temp/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/*.h \
    tests/hw_specific/blackhole/metal_sfpu/
```

These provide higher-level SFPU operation implementations (`llk_sfpu_exp.h`, `llk_sfpu_sqrt.h`, etc.) that supplement the LLK library's built-in set.

---

### 6.1.4 System Package Dependencies

The build environment requires the following system packages:

| Package | Purpose |
|:--------|:--------|
| `build-essential` | GCC, make, and standard C/C++ development headers for host-side compilation. |
| `cmake` | Build system generator. Used by some dependency builds and test infrastructure. |
| `curl`, `wget` | HTTP clients for downloading the SFPI toolchain and tt-metal headers. |
| `tar`, `xz-utils` | Extracting the SFPI `.txz` archive. |
| `git-lfs` | Git Large File Storage. Required for sparse checkout of SFPU headers from tt-metal. |
| `libyaml-cpp-dev` | YAML parsing library. Used by tt-exalens and device configuration. |
| `libhwloc-dev` | Hardware locality library. Used by UMD (User-Mode Driver) for PCIe topology discovery. |
| `libzmq3-dev` | ZeroMQ messaging library. Used by tt-exalens for device communication. |
| `xxd` | Hex dump utility. Used for binary data inspection during debugging. |

Install all at once on Debian/Ubuntu:

```bash
sudo apt-get install -y \
    build-essential cmake curl wget tar xz-utils git-lfs xxd \
    libyaml-cpp-dev libhwloc-dev libzmq3-dev
```

---

### 6.1.5 Python Dependencies

#### Python Version

The test infrastructure requires Python 3.8 or 3.10. Python is the orchestration language -- it generates `build.h` headers, invokes the cross-compiler, loads ELFs to device memory via tt-exalens, and compares kernel output against golden reference values.

#### Key Python Packages

From `tests/requirements.txt`:

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `tt-exalens` | 0.3.11 | Core device interaction library. Provides `load_elf()`, `read_from_device()`, `write_to_device()`, and `parse_elf()` functions for loading kernel ELFs into L1 and reading results. |
| `tt-umd` | 0.9.3.260313 | User-Mode Driver. Low-level PCIe communication layer beneath tt-exalens. |
| `tt-smi` | 3.1.1 | System Management Interface. Used by `setup_testing_env.sh` to detect chip architecture (`get_chip_architecture()` parses `tt-smi -ls` output). |
| `torch` | 2.9.1 | PyTorch. Used for golden reference computation -- the test infrastructure runs `torch.matmul()` on CPU to generate expected output for comparison. |
| `numpy` | 2.2.6 | Array operations for stimuli generation and result comparison. |
| `pytest` | 8.4.2 | Test framework. All LLK tests are pytest test cases. |
| `filelock` | 3.20.3 | File-based locking for concurrent builds. Multiple test workers share compiled artefacts through lock-protected directories. |

The packages are installed from a combination of PyPI, test PyPI (for Tenstorrent packages), and the PyTorch CPU wheel index:

```bash
pip install -r tests/requirements.txt
```

> **Note:** `tt-exalens` and `tt-umd` are sourced from the Tenstorrent test PyPI index (`https://test.pypi.org/simple/`), and `torch` uses the CPU-only wheel index (`https://download.pytorch.org/whl/cpu`). The requirements file includes the necessary `--extra-index-url` directives.

---

### 6.1.6 Firmware Requirement

The LLK test infrastructure does **not** replace chip firmware. The original Blackhole firmware must already be flashed on the P150 before running any LLK tests. The test infrastructure operates by:

1. Loading kernel ELF code into specific L1 memory regions (defined by the linker scripts -- see Section 6.3).
2. Releasing TRISC cores from soft reset to execute the loaded code.
3. Polling mailbox addresses for completion signals.

The firmware structures (reset vectors, debug registers, soft reset control, NOC fabric initialization, clock configuration) must already be in place for this to work. The `device_setup()` function in `boot.h` performs hardware initialization that depends on the firmware having configured the Tensix core into a known state:

```cpp
// From boot.h -- depends on firmware-initialized debug register
ckernel::reg_write(RISCV_DEBUG_REG_DEST_CG_CTRL, 0);  // Blackhole-specific
TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0, 1, 0); // Clear accumulator
TTI_SFPENCC(3, 0, 0, 10);                              // Enable CC stack
```

BRISC's main loop (in `brisc.cpp`) also assumes firmware-level structures are functional -- it reads and writes config registers, controls TRISC soft reset, and uses mailbox addresses that are part of the firmware's L1 layout.

The consequence: if the chip's firmware is corrupted or missing, the LLK tests will fail at device communication, not at compilation. The fix is to re-flash the firmware using Tenstorrent's firmware tools (`tt-flash`), which is outside the scope of tt-llk.

> **Cross-reference:** Chapter 3, Section 2 covers the boot sequence in detail, including how BRISC releases the three TRISC cores from reset and the mailbox signaling protocol.

---

### 6.1.7 Directory Structure Summary

After `setup_testing_env.sh` completes, the relevant directory structure is:

```
tt-llk/
  common/                          # Shared headers (llk_assert.h, tensor_shape.h)
  tt_llk_blackhole/
    llk_lib/                       # LLK API headers (llk_math_matmul.h, llk_pack.h, ...)
    common/inc/                    # ckernel headers (ckernel.h, ckernel_ops.h, ...)
    common/inc/sfpu/               # SFPU ckernel headers
  tests/
    sfpi/
      compiler/
        bin/
          riscv-tt-elf-g++
          riscv-tt-elf-objdump
          riscv-tt-elf-objcopy
          riscv-tt-elf-gcov
          riscv-tt-elf-gcov-tool
      sfpi.version                 # "7.36.0"
    sfpi-version                   # Pinned SFPI release data
    sfpi-info.sh                   # SFPI version metadata script (sources sfpi-version)
    setup_testing_env.sh           # One-command environment setup
    requirements.txt               # Python package manifest
    hw_specific/
      blackhole/
        inc/
          cfg_defines.h
          tensix.h
          tensix_types.h
          dev_mem_map.h
          core_config.h
          internal/
            risc_attribs.h
        metal_sfpu/
          *.h                      # SFPU implementation headers
    helpers/
      include/                     # boot.h, params.h, operand.h, etc.
      ld/                          # Linker scripts (see Section 6.3)
      src/                         # brisc.cpp, trisc.cpp, coverage.cpp
      tmu-crt0.S                   # Startup assembly (see Section 6.3)
    python_tests/
      helpers/test_config.py       # Build orchestration (TestConfig)
```

---

### 6.1.8 Dependency Acquisition Sequence

The complete sequence to go from a bare checkout to a build-ready state:

```
1. Clone tt-llk repository
2. Install system packages (apt-get install ...)
3. Install Python dependencies (pip install -r tests/requirements.txt)
4. Ensure P150 firmware is flashed (tt-flash)
5. Run tests/setup_testing_env.sh:
   a. Detect chip architecture via tt-smi (or CHIP_ARCH env var override)
   b. Download cfg_defines.h, tensix.h, tensix_types.h,
      dev_mem_map.h, core_config.h, risc_attribs.h
   c. Download SFPU operation headers via sparse git clone
   d. Install pre-commit hooks
   e. Download and extract SFPI toolchain (sfpi 7.36.0)
6. Verify: riscv-tt-elf-g++ --version
7. Build environment is ready -- pytest can compile and run kernels
```

After step 5, the `riscv-tt-elf-g++` compiler, all headers, and all linker scripts are in place. Section 6.2 details the exact compiler invocations that turn source into ELF files.

Verify headers and Python dependencies with spot checks:

```bash
test -f tests/hw_specific/blackhole/inc/cfg_defines.h && echo "OK" || echo "MISSING"
python3 -c "from ttexalens.tt_exalens_lib import load_elf; print('tt-exalens OK')"
```

---

*Previous: [Chapter 5 -- Matmul-Specific Config Structs](../ch5_final/03_matmul_specific_config_structs.md)*
| *Next: [6.2 -- Compilation Commands and Flags](02_compilation_commands_and_flags.md)*
