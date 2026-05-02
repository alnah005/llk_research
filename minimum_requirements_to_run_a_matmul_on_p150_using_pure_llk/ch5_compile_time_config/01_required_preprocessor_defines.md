# Chapter 5: Compile-Time Configuration, Preprocessor Defines, and `build.h`

## 5.1 Required Preprocessor Defines

Every LLK kernel compilation requires a precise set of `-D` flags passed to the
RISC-V cross-compiler. Omit one and the build either fails outright or produces
firmware that silently misbehaves on the Tensix core. This section provides the
definitive reference for every preprocessor define in the tt-llk build system.

---

### Master Reference Table

| Define | Required? | Set By | Effect |
|--------|-----------|--------|--------|
| `-DARCH_BLACKHOLE` | **Yes** | `setup_arch()` | Selects Blackhole architecture. Controls clock constants, register layouts, ISA opcodes, cache-control paths. |
| `-DLLK_TRISC_UNPACK` | **Yes** (one per ELF) | `build_kernel_part()` | Compiles unpack TRISC (thread 0). Sets mailbox offset=0, guards unpacker `run_kernel()`. |
| `-DLLK_TRISC_MATH` | **Yes** (one per ELF) | `build_kernel_part()` | Compiles math TRISC (thread 1). Sets mailbox offset=4, guards math `run_kernel()`. |
| `-DLLK_TRISC_PACK` | **Yes** (one per ELF) | `build_kernel_part()` | Compiles pack TRISC (thread 2). Sets mailbox offset=8, guards packer `run_kernel()`. |
| `-DCOMPILE_FOR_TRISC=N` | **Yes** (WH/BH) | `build_kernel_part()` | Numeric index (0/1/2) for per-thread config register offsets. Not used on Quasar. |
| `-DLLK_BOOT_MODE_BRISC` | **Yes** (BH) | `resolve_compile_options()` | BRISC manages TRISC lifecycle. Alternative `-DLLK_BOOT_MODE_TRISC` is Quasar-only. |
| `-DTENSIX_FIRMWARE` | **Yes** | `setup_compilation_options()` | Targets Tensix RISC-V cores (not host). Without it, headers select host-side stubs. |
| `-DENV_LLK_INFRA` | **Yes** | `setup_compilation_options()` | Selects LLK assert path (`ebreak`). Without it, asserts route to tt-metal's `ASSERT()`. |
| `-DENABLE_LLK_ASSERT` | **Yes** | `setup_compilation_options()` | Master switch for `LLK_ASSERT`. Without it, asserts compile to `(void)sizeof(cond)`. |
| `-DRUNTIME_FORMATS` | Conditional | `build_kernel_part()` | Reads `FormatConfig` from L1 at runtime. Omit for compile-time format embedding. |
| `-DLLK_PROFILER` | Optional | `resolve_compile_options()` | Enables profiling. Allocates ring buffers at `0x16B000`. Mutually exclusive with COVERAGE. |
| `-DCOVERAGE` | Optional | `resolve_compile_options()` | Enables gcov instrumentation. Shifts mailbox from `0x1FFB8` to `0x6DFB8`. |
| `-DSPEED_OF_LIGHT` | Optional | `setup_compilation_options()` | All runtime params become `constexpr`. `RuntimeParams` struct becomes empty. |
| `-DFUSED_TEST` | Situational | `fused_generator.py` | Marks fused multi-op kernel from the fuser infrastructure. |

> **Cross-reference:** Chapter 1 covers the P150-to-Blackhole mapping; Chapter 3, Section 1 details the three-compilation pattern.

---

### 5.1.1 Architecture Selection

The architecture define is set in `TestConfig.setup_arch()`:

```python
case ChipArchitecture.BLACKHOLE:
    TestConfig.ARCH_DEFINE = "-DARCH_BLACKHOLE"
    TestConfig.ARCH_LLK_ROOT = "tt_llk_blackhole"
```

It gates Blackhole-specific code such as `ARCH_CYCLE_MICRO_SECOND = 1350` in
`brisc.cpp`, the 5-argument `ZEROACC` instruction form in `boot.h`, and the
`RISCV_DEBUG_REG_DEST_CG_CTRL` cache-control write.

### 5.1.2 TRISC Thread Identity

Each TRISC thread compiles the same source with a different define. On Quasar
a fourth thread exists (`-DLLK_TRISC_ISOLATE_SFPU`). The build system sets
these in `build_kernel_part()`:

```python
trisc_define = "ISOLATE_SFPU" if name == "sfpu" else name.upper()
compile_command = f"-DLLK_TRISC_{trisc_define} ..."
```

In `trisc.cpp`, these control the mailbox offset and kernel dispatch:

```cpp
#if defined(LLK_TRISC_UNPACK)
constexpr std::uint32_t mailbox_offset = 0;
#elif defined(LLK_TRISC_MATH)
constexpr std::uint32_t mailbox_offset = sizeof(std::uint32_t);
#elif defined(LLK_TRISC_PACK)
constexpr std::uint32_t mailbox_offset = 2 * sizeof(std::uint32_t);
#else
#error "No TRISC define set"
#endif
```

The test source file (e.g., `matmul_test.cpp`) uses these to select which
`run_kernel()` body is compiled into each ELF:

```cpp
#ifdef LLK_TRISC_UNPACK
void run_kernel(RUNTIME_PARAMETERS params) { /* unpack path */ }
#endif
#ifdef LLK_TRISC_MATH
void run_kernel(RUNTIME_PARAMETERS params) { /* math path */ }
#endif
#ifdef LLK_TRISC_PACK
void run_kernel(RUNTIME_PARAMETERS params) { /* pack path */ }
#endif
```

The numeric index `-DCOMPILE_FOR_TRISC=N` is set alongside the named define,
computed via `TestConfig.KERNEL_COMPONENTS.index(name)` where
`KERNEL_COMPONENTS = ["unpack", "math", "pack"]`. It is consumed by low-level
hardware register selection headers (e.g., `cfg_defines.h`) to pick
thread-specific configuration register addresses. On Quasar this define is
omitted because the hardware uses a different register-selection mechanism.

### 5.1.3 Boot Mode

On Blackhole, the BRISC (Baby RISC) core runs an infinite command loop
(`brisc.cpp`) that handles TRISC reset, start-address caching, and mailbox
initialization. On Quasar, TRISC0 takes ownership of device setup:

```python
OPTIONS_COMPILE += (
    "-DLLK_BOOT_MODE_TRISC "
    if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR
    else "-DLLK_BOOT_MODE_BRISC "
)
```

On the C++ side, `brisc.cpp` wraps its entire body in
`#ifdef LLK_BOOT_MODE_BRISC`. And `trisc.cpp` uses the TRISC boot mode to have
TRISC0 perform device setup:

```cpp
#if defined(LLK_TRISC_UNPACK) && defined(LLK_BOOT_MODE_TRISC)
    device_setup();
    clear_trisc_soft_reset();  // Release TRISC1 and TRISC2
#endif
```

For P150 (Blackhole), the boot mode is always BRISC.

> **Cross-reference:** Chapter 3 covers the boot sequence and BRISC/TRISC lifecycle management.

### 5.1.4 Firmware, Infrastructure, and Assertions

These three defines are always present, hard-coded in `INITIAL_OPTIONS_COMPILE`.
The assertion chain from `common/llk_assert.h` has three levels:

```cpp
#ifdef ENABLE_LLK_ASSERT
  #ifdef ENV_LLK_INFRA
    // Standalone LLK: halt core with ebreak
    #define LLK_ASSERT(condition, message) \
        do { if (UNLIKELY(!(condition))) { asm volatile("ebreak"); } } while (0)
  #else
    // tt-metal: use tt-metal's ASSERT macro
    #include "api/debug/assert.h"
    #define LLK_ASSERT(condition, message) ASSERT(condition)
  #endif
#else
    // Disabled: type-check but never execute
    #define LLK_ASSERT(condition, message) ((void)sizeof((condition)))
#endif
```

### 5.1.5 Optional Defines

These defines are conditionally added based on test configuration:

| Define | Condition | Set in | Effect |
|---|---|---|---|
| `-DRUNTIME_FORMATS` | `compile_time_formats == False` | `build_kernel_part()` | Tells kernel code to read `FormatConfig` from the `RuntimeParams` struct at runtime instead of using `constexpr` values from `build.h`. |
| `-DLLK_PROFILER` | `profiler_build == ProfilerBuild.Yes` | `resolve_compile_options()` | Enables cycle-accurate profiling instrumentation. `trisc.cpp` initializes profiler ring buffers at `0x16B000` and sets up barriers. |
| `-DCOVERAGE` | `coverage_build == CoverageBuild.Yes` | `resolve_compile_options()` | Enables gcov code-coverage instrumentation. Shifts mailbox base from `0x1FFB8` to `0x6DFB8` and adds `-fprofile-arcs -ftest-coverage -fprofile-info-section`. |
| `-DSPEED_OF_LIGHT` | `TestConfig.SPEED_OF_LIGHT == True` | `setup_compilation_options()` | Converts all runtime parameters to compile-time constants for maximum performance. Eliminates the `RuntimeParams` struct entirely. |

The `RUNTIME_FORMATS` define controls whether format configuration is read from
L1 at runtime or baked into `build.h` as `constexpr` values:

```python
if not self.compile_time_formats:
    optional_kernel_flags += " -DRUNTIME_FORMATS"
```

When active, kernel code reads formats from the runtime struct:
```cpp
#ifdef RUNTIME_FORMATS
    const FormatConfig& formats = params.formats;
#endif
```

The `SPEED_OF_LIGHT` mode promotes all `RuntimeParameter` instances to `TemplateParameter`, collapsing `RuntimeParams` to an empty struct (see Section 5.2.6 for the full mechanism).

> **Cross-reference:** Section 5.2 covers the compile-time vs. runtime format distinction in detail.

---

### 5.1.6 Minimal Compiler Invocation for a P150 Matmul

```bash
riscv-tt-elf-g++ \
  -mcpu=tt-bh-tensix \
  -O3 -std=c++17 -ffast-math -nostdlib \
  -DARCH_BLACKHOLE \
  -DTENSIX_FIRMWARE \
  -DENV_LLK_INFRA \
  -DENABLE_LLK_ASSERT \
  -DLLK_BOOT_MODE_BRISC \
  -DLLK_TRISC_MATH \
  -DCOMPILE_FOR_TRISC=1 \
  -DRUNTIME_FORMATS \
  -I<variant_dir> \
  ...
```

For unpack, replace `-DLLK_TRISC_MATH -DCOMPILE_FOR_TRISC=1` with
`-DLLK_TRISC_UNPACK -DCOMPILE_FOR_TRISC=0`. For pack,
`-DLLK_TRISC_PACK -DCOMPILE_FOR_TRISC=2`. The BRISC ELF uses `-mcpu=tt-bh`
(non-compute) and needs only `-DARCH_BLACKHOLE -DLLK_BOOT_MODE_BRISC`.

---

### 5.1.7 Interaction Between Defines

- **`LLK_PROFILER` and `COVERAGE` are mutually exclusive.** The build system
  raises `RuntimeError` if both are requested.

- **`SPEED_OF_LIGHT` implies `compile_time_formats=True`**, so
  `-DRUNTIME_FORMATS` is NOT passed. All `RuntimeParameter` instances become
  `TemplateParameter` instances.

- **`COVERAGE` changes the memory layout.** It selects `memory.blackhole.debug.ld`
  and shifts the mailbox base from `0x1FFB8` to `0x6DFB8`.

- **`COMPILE_FOR_TRISC` must match the `LLK_TRISC_*` define.** The build
  system guarantees this by computing the index from the component name.

---

**Previous:** [Chapter 4 -- LLK API Sequences](../ch4_matmul_api_calls/04_minimum_call_summary.md) | **Next:** [`02_build_h_generation_and_constexpr.md`](./02_build_h_generation_and_constexpr.md)
