# 6.2 Compilation Commands and Flags

This section traces the exact compiler invocations that produce the four ELF files needed to run a matmul on P150. The focus is on the Python build system orchestration: how `TestConfig.build_elfs()` assembles the command, how `build_kernel_part()` uses stdin piping to create a synthetic translation unit, and how the linker flags place code and data into the correct L1 regions. All information is extracted directly from `test_config.py`.

---

### 6.2.1 The Two-Phase Build Pipeline

The build splits into two phases:

1. **Shared artefacts** -- `brisc.elf` (and optionally `coverage.o`) are built once and reused across all test variants.
2. **Per-variant ELFs** -- `unpack.elf`, `math.elf`, and `pack.elf` are built once per unique combination of template parameters, data formats, and compile options.

The entry point is `TestConfig.build_elfs()`, which calls `build_shared_artefacts()` first, then launches parallel compilation of the three kernel ELFs:

```
TestConfig.run()
    |
    v
TestConfig.build_elfs()
    |
    +-- build_shared_artefacts()       [once per test session]
    |       |
    |       +-- compile brisc.elf      [from brisc.cpp]
    |       +-- compile coverage.o     [if coverage enabled]
    |
    +-- generate_build_header()        [writes build.h to variant dir]
    |
    +-- ThreadPoolExecutor(max_workers=3)
            |
            +-- build_kernel_part("unpack")  --> unpack.elf
            +-- build_kernel_part("math")    --> math.elf
            +-- build_kernel_part("pack")    --> pack.elf
```

Each variant gets its own directory keyed by a SHA-256 digest of all compilation-affecting parameters (template values, data formats, profiler flags):

```
/tmp/tt-llk-build/
  shared/
    elf/
      brisc.elf
    obj/
      coverage.o          (only if --coverage)
  <test_name>/
    <variant_hash>/
      build.h             (generated config header -- see Chapter 5, Section 2)
      elf/
        unpack.elf
        math.elf
        pack.elf
      .build_complete      (stamp file: variant build is done)
```

---

### 6.2.2 Architecture Selection Flags

The SFPI toolchain supports multiple Tenstorrent architectures through the `-mcpu` flag. For Blackhole, two distinct CPU targets are used:

| Flag | Used For | Meaning |
|:-----|:---------|:--------|
| `-mcpu=tt-bh-tensix` | TRISC threads (unpack, math, pack) | Enables Tensix compute-specific instruction encodings (SFPU instructions, TTI macros, MOP triggers, dest register access). |
| `-mcpu=tt-bh` | BRISC firmware | General RISC-V for Blackhole without Tensix compute extensions. BRISC does not have access to the Tensix compute engine. |

These are set during `setup_arch()`:

```python
# test_config.py lines 229-230
case ChipArchitecture.BLACKHOLE:
    TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-bh"
    TestConfig.ARCH_COMPUTE     = "-mcpu=tt-bh-tensix"
    TestConfig.ARCH_DEFINE      = "-DARCH_BLACKHOLE"
```

---

### 6.2.3 Common Compiler Flags (`OPTIONS_ALL`)

The `OPTIONS_ALL` string applies to every compilation -- both BRISC and TRISC threads:

```python
# test_config.py line 331
TestConfig.OPTIONS_ALL = f"{debug_flag}-O3 -std=c++17 -ffast-math"
```

| Flag | Purpose |
|:-----|:--------|
| `-g` | Emit debug symbols (DWARF). Debug sections are placed at address 0 in the ELF via the `0 :` syntax in `sections.ld` -- they are retained for host-side `objdump` inspection but are not loaded to device memory. Can be disabled with `no_debug_symbols=True`. |
| `-O3` | Maximum optimization. Critical for kernel performance -- the LLK headers are heavily templated and rely on aggressive inlining and constant propagation to eliminate dead branches from compile-time `constexpr` conditionals. Code must fit in 16 KB. |
| `-std=c++17` | C++17 standard. Required for `constexpr if`, structured bindings, `std::array` deduction guides, and `[[maybe_unused]]` attributes used throughout the LLK library. |
| `-ffast-math` | Relaxed floating-point semantics. Allows the compiler to reorder and simplify FP operations. Acceptable because the Tensix FPU has its own precision model (controlled by `MathFidelity` -- see Chapter 5) and the SFPU handles special values in hardware. |

---

### 6.2.4 Compile-Only Flags (`INITIAL_OPTIONS_COMPILE`)

These flags apply during compilation (not to the BRISC link step):

```python
# test_config.py lines 342-347
TestConfig.INITIAL_OPTIONS_COMPILE = (
    "-nostdlib -fno-use-cxa-atexit -Werror -Wall "
    "-fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti "
    "-Wunused-parameter -Wfloat-equal -Wpointer-arith "
    "-Wnull-dereference -Wredundant-decls -Wuninitialized "
    "-Wmaybe-uninitialized "
    f"-DTENSIX_FIRMWARE -DENV_LLK_INFRA -DENABLE_LLK_ASSERT "
    f"{TestConfig.ARCH_DEFINE} "
    f"{'-DSPEED_OF_LIGHT' if TestConfig.SPEED_OF_LIGHT else ''}"
)
```

#### Bare-Metal Flags

| Flag | Purpose |
|:-----|:--------|
| `-nostdlib` | Do not link the standard C library. The kernel runs on bare metal with no OS. The ckernel namespace and `boot.h` provide all runtime support. |
| `-fno-use-cxa-atexit` | Disables `__cxa_atexit` registration for static destructors. The kernels never return from `main()` (they loop forever after signaling completion), so no destructor cleanup is needed. |
| `-fno-asynchronous-unwind-tables` | Suppresses `.eh_frame` generation. Exception unwinding is not used; these tables would waste code space. |
| `-fno-exceptions` | Disables C++ exception support entirely. No `try`/`catch` is possible in kernel code. `LLK_ASSERT` uses `ebreak` instead. |
| `-fno-rtti` | Disables run-time type information. No `dynamic_cast` or `typeid`. |
| `-nostartfiles` | (In `OPTIONS_LINK`.) See Section 6.2.5. |

#### Warning Flags

All warnings are promoted to errors with `-Werror`. The additional warnings (`-Wall`, `-Wunused-parameter`, `-Wfloat-equal`, `-Wpointer-arith`, `-Wnull-dereference`, `-Wredundant-decls`, `-Wuninitialized`, `-Wmaybe-uninitialized`) enforce strict code quality.

#### Preprocessor Defines

The compilation command includes fixed preprocessor defines (`-DTENSIX_FIRMWARE`, `-DENV_LLK_INFRA`, `-DENABLE_LLK_ASSERT`, `-DARCH_BLACKHOLE`) and optional defines (`-DRUNTIME_FORMATS`, `-DSPEED_OF_LIGHT`, `-DLLK_PROFILER`, `-DCOVERAGE`). Each thread compilation also receives per-thread defines (`-DCOMPILE_FOR_TRISC=N`, `-DLLK_TRISC_UNPACK/MATH/PACK`) and a boot-mode define (`-DLLK_BOOT_MODE_BRISC`). See Chapter 5, Section 5.1 for the definitive reference on all preprocessor defines, their conditions, and their effects. The per-thread flag differences are shown in the concrete compilation commands in Section 6.2.8.

---

### 6.2.5 Linker Flags (`OPTIONS_LINK`)

```python
# test_config.py line 341
TestConfig.OPTIONS_LINK = (
    "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 "
    "-nostartfiles -Wl,--trace"
)
```

| Flag | Purpose |
|:-----|:--------|
| `-Wl,-z,max-page-size=16` | Sets maximum page alignment to 16 bytes. The default ELF page size of 4096 bytes would waste L1 SRAM. Tensix cores have no MMU, so page alignment has no hardware meaning beyond the linker's placement decisions. |
| `-Wl,-z,common-page-size=16` | Sets common page alignment to 16 bytes. Works in conjunction with `max-page-size` to minimize padding between ELF segments. |
| `-nostartfiles` | Do not link the toolchain's default CRT startup files (`crt0.o`, `crti.o`, `crtn.o`). The custom `_start` function in `boot.h` provides the entry point. |
| `-Wl,--trace` | Print the name of each input file as the linker processes it. Useful for debugging link order issues (output goes to stderr). |


---

### 6.2.6 Include Paths

The include path order matters because multiple directories contain headers with the same name (e.g. `dev_mem_map.h` exists in both `helpers/include/` and `hw_specific/blackhole/inc/`). The order as constructed in `setup_compilation_options()`:

```python
# test_config.py lines 348-359
TestConfig.INCLUDES = [
    "-Isfpi/include",                                    # 1. SFPI intrinsic headers
    "-I../tt_llk_blackhole/llk_lib",                     # 2. LLK API (llk_math_matmul.h, etc.)
    "-I../tt_llk_blackhole/common/inc",                  # 3. ckernel core (ckernel.h, ckernel_ops.h)
    "-I../tt_llk_blackhole/common/inc/sfpu",             # 4. ckernel SFPU headers
    "-I../common",                                       # 5. Shared headers (llk_assert.h)
    f"-I{TestConfig.HEADER_DIR}",                        # 6. hw_specific/blackhole/inc/ (cfg_defines.h, tensix.h)
    f"-Ihw_specific/{TestConfig.ARCH.value}",            # 7. hw_specific/blackhole/ (arch-level config)
    f"-Ihw_specific/{TestConfig.ARCH.value}/metal_sfpu", # 8. SFPU op headers from tt-metal
    "-Ifirmware/riscv/common",                           # 9. Firmware common headers (if present)
    "-Ihelpers/include",                                 # 10. Test infrastructure headers (boot.h, params.h)
]
```

Additionally, `build_kernel_part()` prepends three more paths per compilation:

| Additional Path | Contents |
|:----------------|:---------|
| `-I{TestConfig.TESTS_WORKING_DIR}` | Resolves to `tests/`. Allows `#include <sources/matmul_test.cpp>` via the stdin piping pattern. |
| `-I{TestConfig.RISCV_SOURCES}` | Resolves to `tests/helpers/src/`. Allows `#include <trisc.cpp>`. |
| `-I{VARIANT_DIR}` | Resolves to the variant build directory (e.g., `/tmp/tt-llk-build/<test>/<hash>/`). Contains the generated `build.h` (see Chapter 5, Section 2). |

The include order matters: SFPI headers must come first, then LLK library headers, then hardware definitions, then test infrastructure headers.

---

### 6.2.7 The stdin Piping Technique

The most unusual aspect of the build system is how it feeds source code to the compiler. Rather than passing a `.cpp` file path on the command line, `build_kernel_part()` pipes a synthetic two-line source on stdin:

```python
# test_config.py lines 1014-1018
run_shell_command(
    compile_command,
    TestConfig.TESTS_WORKING_DIR,
    (f"#include  <{self.test_name}>\n"   # e.g. #include <python_tests/test_matmul.cpp>
     "#include  <trisc.cpp>\n"),           # Always included -- TRISC main() entry point
)
```

The compile command ends with `-x c++ -` which tells the compiler to read C++ source from stdin. What the compiler actually receives (for a matmul test):

```cpp
#include  <sources/matmul_test.cpp>
#include  <trisc.cpp>
```

This achieves two things:

1. **The test source** is included first, bringing in the `#ifdef LLK_TRISC_UNPACK` / `LLK_TRISC_MATH` / `LLK_TRISC_PACK` blocks that define the `run_kernel()` function for each thread.

2. **The trisc runtime** (`tests/helpers/src/trisc.cpp`) is included second, providing the `_start` entry point, `main()`, mailbox setup, and the call to `run_kernel()`.

The angle-bracket include (`<...>`) works because the test working directory and the RISCV sources directory are both on the `-I` include path. The compiler resolves `<sources/matmul_test.cpp>` via `-I{TestConfig.TESTS_WORKING_DIR}` and `<trisc.cpp>` via `-I{TestConfig.RISCV_SOURCES}`.

**Why stdin piping?** This technique avoids creating a temporary concatenated source file on disk. It also means the compiler sees a single translation unit containing both the test-specific kernel code and the runtime framework, which is necessary because `trisc.cpp` calls `run_kernel()` -- a function defined in the test source. Without this single-TU approach, the linker would need to resolve `run_kernel` as an external symbol, requiring a separate compilation and link step.

> **Cross-reference:** Chapter 3, Section 1 explains the single-source, three-compiles pattern that this stdin piping implements.

---

### 6.2.8 The Complete TRISC1 (Math) Compilation Command

Assembling all the pieces, the fully expanded command for the math thread:

```bash
/path/to/tt-llk/tests/sfpi/compiler/bin/riscv-tt-elf-g++ \
    -mcpu=tt-bh-tensix \
    -g -O3 -std=c++17 -ffast-math \
    -I/path/to/tt-llk/tests \
    -I/path/to/tt-llk/tests/helpers/src \
    -I/tmp/tt-llk-build/sources/matmul_test/<variant-hash> \
    -Isfpi/include \
    -I../tt_llk_blackhole/llk_lib \
    -I../tt_llk_blackhole/common/inc \
    -I../tt_llk_blackhole/common/inc/sfpu \
    -I../common \
    -I/path/to/tt-llk/tests/hw_specific/blackhole/inc \
    -Ihw_specific/blackhole \
    -Ihw_specific/blackhole/metal_sfpu \
    -Ifirmware/riscv/common \
    -Ihelpers/include \
    -nostdlib -fno-use-cxa-atexit -Werror -Wall \
    -fno-asynchronous-unwind-tables -fno-exceptions -fno-rtti \
    -Wunused-parameter -Wfloat-equal -Wpointer-arith \
    -Wnull-dereference -Wredundant-decls -Wuninitialized -Wmaybe-uninitialized \
    -DTENSIX_FIRMWARE -DENV_LLK_INFRA -DENABLE_LLK_ASSERT -DARCH_BLACKHOLE \
    -DLLK_BOOT_MODE_BRISC \
    -DCOMPILE_FOR_TRISC=1 \
    -DLLK_TRISC_MATH \
    -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 \
    -nostartfiles -Wl,--trace \
    -T/path/to/tt-llk/tests/helpers/ld/memory.blackhole.ld \
    -T/path/to/tt-llk/tests/helpers/ld/math.ld \
    -T/path/to/tt-llk/tests/helpers/ld/sections.ld \
    -x c++ - \
    -lc \
    -o /tmp/tt-llk-build/sources/matmul_test/<variant-hash>/elf/math.elf
```

With stdin input:

```cpp
#include  <sources/matmul_test.cpp>
#include  <trisc.cpp>
```

The three TRISC compilations differ only in:
- `-DCOMPILE_FOR_TRISC=N` (0, 1, or 2)
- `-DLLK_TRISC_UNPACK` / `-DLLK_TRISC_MATH` / `-DLLK_TRISC_PACK`
- `-T.../unpack.ld` / `-T.../math.ld` / `-T.../pack.ld`
- Output filename: `unpack.elf` / `math.elf` / `pack.elf`

---

### 6.2.9 The BRISC Compilation Command

BRISC firmware is compiled separately as a **shared artefact** -- it does not depend on the test variant and is built once per test session. The command:

```python
# test_config.py lines 753-758
compile_command = (
    f"{TestConfig.GXX} {TestConfig.ARCH_NON_COMPUTE} {TestConfig.OPTIONS_ALL} "
    f"{TestConfig.OPTIONS_LINK} {local_non_coverage} "
    f'{"-DCOVERAGE " if TestConfig.WITH_COVERAGE else ""}'
    f"-T{local_memory_layout_ld} "
    f"-T{TestConfig.LINKER_SCRIPTS / 'brisc.ld'} "
    f"-T{TestConfig.LINKER_SCRIPTS}/sections.ld "
    f"-o {shared_elf_dir / 'brisc.elf'} "
    f"{TestConfig.RISCV_SOURCES / 'brisc.cpp'}"
)
```

Key differences from TRISC compilation:

| Aspect | TRISC | BRISC |
|:-------|:------|:------|
| CPU target | `-mcpu=tt-bh-tensix` | `-mcpu=tt-bh` |
| Source input | stdin pipe (`-x c++ -`) | Direct file (`brisc.cpp`) |
| Thread defines | `-DLLK_TRISC_UNPACK/MATH/PACK`, `-DCOMPILE_FOR_TRISC=N` | None |
| Linker alias script | `unpack.ld` / `math.ld` / `pack.ld` | `brisc.ld` |
| Output location | `<variant>/elf/unpack.elf` etc. | `shared/elf/brisc.elf` |
| Build frequency | Once per variant | Once per session (shared) |

BRISC does not use the stdin piping technique because `brisc.cpp` contains its own `_start` entry point and `main()` function. It does not include a kernel source file -- its sole purpose is the boot-and-poll loop described in Chapter 3, Section 2.

---

### 6.2.10 Parallel Compilation and Build Locking

All three TRISC threads are compiled in parallel using `ThreadPoolExecutor(max_workers=3)`. Since each thread compiles to a different ELF with different linker aliases, there are no file conflicts.

Build deduplication uses a two-level scheme: a `FileLock` keyed on the variant hash prevents concurrent builds of the same variant, and a `.build_complete` stamp file provides a fast-path skip when the variant is already built. A similar pattern protects `build_shared_artefacts()` with a separate lock file (`/tmp/tt-llk-build-shared.lock`).

---

### 6.2.11 The `-lc` Flag and Minimal C Library

The TRISC compilation command includes `-lc`, which links a minimal C library provided by the SFPI toolchain. This supplies:

- `memcpy`, `memset`, `memmove` -- used by the compiler for struct copies and array initialization
- Basic integer math routines (`__muldi3`, `__divdi3`) for operations the hardware does not natively support

The library does **not** provide I/O, heap allocation, or any POSIX functionality. Combined with `-nostdlib` and `-nostartfiles`, the resulting binary is truly bare-metal.

---

### 6.2.12 ELF Inspection with objdump and objcopy

After compilation, verify section placement and code size:

```bash
riscv-tt-elf-objdump -h math.elf                  # List sections with sizes and addresses
riscv-tt-elf-objdump -d -S math.elf > math.dis    # Disassemble with source annotations
riscv-tt-elf-objdump -t math.elf | grep _start    # Verify entry point address

# Check total code fits within 16K
riscv-tt-elf-objdump -h math.elf | grep -E '\.init|\.text|l1_data' | \
    awk '{sum += strtonum("0x"$3)} END {printf "Code: %d bytes (%.1f%% of 16K)\n", sum, sum/16384*100}'

# Extract profiler metadata section
riscv-tt-elf-objcopy -O binary -j .profiler_meta math.elf math.meta.bin
```

---

### 6.2.13 Common Build Errors and Fixes

| Error | Cause | Fix |
|:------|:------|:----|
| `riscv-tt-elf-g++: No such file or directory` | SFPI toolchain not installed. | Run `bash tests/setup_testing_env.sh`. Verify `tests/sfpi/compiler/bin/riscv-tt-elf-g++` exists. |
| `fatal error: cfg_defines.h: No such file or directory` | tt-metal headers not downloaded. | Run `setup_testing_env.sh` or check `tests/hw_specific/blackhole/inc/`. Delete `.headers_downloaded` stamp to force re-download. |
| `fatal error: build.h: No such file or directory` | Variant build directory not created or `build.h` not generated. | Ensure `TestConfig.build_elfs()` runs first. The variant directory is `/tmp/tt-llk-build/<test>/<hash>/`. See Chapter 5, Section 2 for `build.h` generation. |
| `undefined reference to 'run_kernel'` | Missing test source or `#ifdef` guard mismatch. | Verify the test source defines `run_kernel()` inside the correct `#ifdef LLK_TRISC_*` block for each thread. |
| `section '.text' will not fit in region 'TRISC1_CODE'` | Kernel code exceeds 16K. | Reduce code size: ensure `-O3`, enable `SPEED_OF_LIGHT`, simplify the kernel, or split complex operations. |
| `section '.ldm_data' will not fit in region 'TRISC1_LOCAL_DATA_MEM'` | Static data exceeds 3584 bytes of private local memory. | Move large constants to L1 (use `l1_data` section attribute), reduce `constexpr` arrays, or load data from L1 at runtime. |
| `undefined symbol: __cxa_atexit` | Missing `-fno-use-cxa-atexit` or global C++ object with non-trivial destructor. | Ensure `-fno-use-cxa-atexit` is set. Avoid global variables with destructors. |
| `relocation truncated to fit: R_RISCV_JAL` | Jump target out of range. | Ensure `-Wl,-z,max-page-size=16` is set. Reduce code size. |
| `cannot find -lgcov` | Coverage build requested but gcov library not available. | Use coverage builds only with the full SFPI toolchain. For non-coverage builds, omit `-fprofile-arcs -ftest-coverage`. |

---

*Previous: [6.1 -- Toolchain and Dependencies](01_toolchain_and_dependencies.md)*
| *Next: [6.3 -- Linker Scripts and Memory Regions](03_linker_scripts_and_memory_regions.md)*
