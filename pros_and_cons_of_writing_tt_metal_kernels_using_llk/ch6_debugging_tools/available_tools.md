# Available Debugging Tools

## LLK_ASSERT

`LLK_ASSERT(condition, message)` is the primary runtime assertion mechanism inside LLK library code. It is defined in `tt-llk/common/llk_assert.h` and operates in one of three modes depending on build-time defines:

### Mode 1: TT-LLK infrastructure (`ENABLE_LLK_ASSERT` + `ENV_LLK_INFRA`)

When running inside the standalone TT-LLK test infrastructure, the assertion compiles to a RISC-V `ebreak` instruction:

```cpp
// tt-llk/common/llk_assert.h:11-20
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

#define LLK_ASSERT(condition, message) \
    do                                 \
    {                                  \
        if (UNLIKELY(!(condition)))    \
        {                              \
            asm volatile("ebreak");    \
        }                              \
    } while (0)
```

The `ebreak` instruction causes the TRISC core to halt immediately. This is the most direct form of the assertion -- it stops execution at the exact point of failure. However, the only externally visible effect is that the core stops making progress; there is no message transmitted to the host.

### Mode 2: TT-Metal ASSERT (`ENABLE_LLK_ASSERT` without `ENV_LLK_INFRA`)

When compiled inside TT-Metal, `LLK_ASSERT` delegates to the TT-Metal `ASSERT` macro:

```cpp
// tt-llk/common/llk_assert.h:26-27
#include "api/debug/assert.h"
#define LLK_ASSERT(condition, message) ASSERT(condition)
```

The TT-Metal `ASSERT` macro itself has two sub-modes depending on `WATCHER_ENABLED`:

**With watcher enabled** (`tt_metal/hw/inc/api/debug/assert.h:12-47`): The assertion writes the failing line number and thread index into a mailbox in L1 memory, then enters an infinite spin loop. The host-side watcher thread periodically reads this mailbox and reports the failure with a line number.

```cpp
// tt_metal/hw/inc/api/debug/assert.h:12-17
inline void assert_and_hang(uint32_t line_num, ...) {
    debug_assert_msg_t tt_l1_ptr* v = GET_MAILBOX_ADDRESS_DEV(watcher.assert_status);
    if (v->tripped == DebugAssertOK) {
        v->line_num = line_num;
        v->tripped = assert_type;
        v->which = internal_::get_hw_thread_idx();
    }
    // ...
    while (1) { ; }
}
```

**With lightweight asserts** (`LIGHTWEIGHT_KERNEL_ASSERTS`): Falls back to a bare `ebreak`, identical to the TT-LLK infra behavior.

**With neither**: The `ASSERT` macro compiles to nothing.

### Mode 3: Disabled (default)

When `ENABLE_LLK_ASSERT` is not defined -- which is the default in TT-Metal release builds -- the macro compiles to a `sizeof` expression:

```cpp
// tt-llk/common/llk_assert.h:35
#define LLK_ASSERT(condition, message) ((void)sizeof((condition)))
```

This is a deliberate design choice. The `sizeof` creates an unevaluated context: the condition is fully type-checked and name-resolved by the compiler, but generates zero runtime code. This means the assertion still catches typos and type errors at compile time, even when disabled.

### Enabling LLK asserts in TT-Metal

LLK asserts are controlled by the environment variable `TT_METAL_LLK_ASSERTS`. When set, the JIT build system adds `-DENABLE_LLK_ASSERT` to the kernel compile flags:

```cpp
// tt_metal/jit_build/build.cpp:259-261
if (rtoptions.get_llk_asserts()) {
    this->defines_ += "-DENABLE_LLK_ASSERT ";
}
```

### Effectiveness assessment

LLK_ASSERT is a well-designed mechanism that balances zero-cost disabled mode (via `sizeof`) with informative failure when enabled (via watcher mailbox). The `sizeof` trick in disabled mode is particularly valuable -- it ensures assertions are never silently broken by refactoring, even in production builds. The main weakness is that the `message` parameter in `LLK_ASSERT(condition, message)` is never actually used in any of the three modes; it exists solely as documentation for the developer reading the source.

---

## fake_kernels_target

The `fake_kernels_target` is a CMake build target that compiles all kernel source files at CMake configure/build time against a set of hardcoded stub definitions, rather than waiting for runtime JIT compilation. It is defined in:

`tt_metal/jit_build/fake_kernels_target/CMakeLists.txt`

### How it works

The target collects all `.cpp` and `.cc` files from kernel directories across TT-Metal and TTNN:

```cmake
# tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:1-42
file(GLOB_RECURSE TT_METAL_KERNELS "${CMAKE_SOURCE_DIR}/tt_metal/kernels/*.cpp" ...)
file(GLOB_RECURSE TTNN_OPERATION_KERNELS ...)
```

It compiles them using the device-side toolchain (`-mcpu=tt-wh`, `-std=c++20`) but with a special prelude header (`fake_jit_prelude.h`) force-included via `-include`:

```cmake
# tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:113
-include "${CMAKE_SOURCE_DIR}/tt_metal/jit_build/fake_kernels_target/fake_jit_prelude.h"
```

The prelude header provides hardcoded values for constants that are normally generated at runtime by the JIT system. For example, it supplies stub values for `chlkc_descriptors.h` constants like data format arrays and tile dimensions:

```cpp
// tt_metal/jit_build/fake_kernels_target/fake_jit_prelude.h:17-21
constexpr bool DST_ACCUM_MODE = false;
#define DST_SYNC_MODE DstSync::SyncHalf
constexpr bool APPROX = true;
constexpr std::int32_t MATH_FIDELITY = 255;
```

It also provides dummy compile-time arguments:

```cmake
# tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:136
"KERNEL_COMPILE_TIME_ARGS=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
```

By default the target pretends kernels are compiled for the MATH TRISC core:

```cmake
# tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:124-125
TRISC_MATH=1
NAMESPACE=chlkc_math
```

### What it catches

- Syntax errors in kernel source files
- Type errors and undefined symbol references
- Missing includes
- Template instantiation failures
- Misuse of LLK APIs that fail `static_assert` checks

### What it cannot catch

- Errors that depend on the actual compile-time arguments passed at runtime (since the fake target uses hardcoded dummy values)
- Mismatches between data format descriptors and the actual circular buffer configuration
- Errors that only manifest in the UNPACK, PACK, or ISOLATE_SFPU TRISC contexts (since the default build is MATH-only, leaving three of four TRISC contexts unverified)
- Any runtime behavior -- the kernels are never executed

### Effectiveness assessment

The fake_kernels_target is a valuable CI gatekeep. It catches compile-time errors in kernels without requiring hardware or a full Metal runtime. However, its reliance on hardcoded stub values means that a kernel which compiles successfully against the fake target may still fail during actual JIT compilation when real descriptor values are substituted. The single-TRISC-context limitation (MATH by default) also means that pack-, unpack-, or isolate_sfpu-specific code paths are not verified (three of four TRISC contexts are missing).

---

## Generated file inspection

When TT-Metal JIT-compiles a compute kernel, it generates several intermediate files that can be inspected for debugging. These files are written to the kernel's output directory under the JIT build root.

### chlkc_math.cpp, chlkc_unpack.cpp, chlkc_pack.cpp, chlkc_isolate_sfpu.cpp

The JIT system generates four C++ source files, one per TRISC processor context. The generation logic is in `tt_metal/jit_build/genfiles.cpp:171-231`. Each file contains:

1. A `#define` for the TRISC type (e.g., `#define TRISC_MATH`)
2. An `#include "defines_generated.h"` directive
3. The user's kernel source, either included via `#include` (for file-based kernels) or inlined directly (for source-code kernels)

For kernels using the simplified `kernel_main()` syntax, the source is transformed into the legacy `namespace NAMESPACE { void MAIN() { ... } }` format. A `#line` directive is emitted to preserve original file line numbers in compiler diagnostics:

```cpp
// tt_metal/jit_build/genfiles.cpp:209-216
if (kernel_src.source_type_ == KernelSource::FILE_PATH) {
    line_directive = "#line 1 \"" + kernel_src.path_.string() + "\"\n";
}
unpack_src = line_directive +
    simple_kernel_syntax::transform_to_legacy_syntax(kernel_content, "chlkc_unpack", "unpack_main");
```

### defines_generated.h

This file contains all user-defined `#define` values added via the `add_define()` API on kernel objects. It is generated at `tt_metal/jit_build/genfiles.cpp:235-246`:

```cpp
const string generated_defines_fname = out_dir + "defines_generated.h";
settings.process_defines([&gen_defines_file](const string& define, const string& value) {
    gen_defines_file << "#define " << define << " " << value << endl;
});
```

### chlkc_descriptors.h

This header contains the data format and tile dimension arrays that describe circular buffer configurations. It is generated by `generate_all_descriptors()` at `tt_metal/jit_build/genfiles.cpp:488-524`. The content is split by TRISC context using `#if defined(UCK_CHLKC_MATH)` guards:

- Math scalar descriptors (`MATH_FIDELITY`, `APPROX`) — emitted by `emit_math_scalar_descriptors`, visible only to MATH
- Compute-wide scalar descriptors (`DST_ACCUM_MODE`, `DST_SYNC_MODE`) — emitted by `emit_compute_scalar_descriptors`, visible to all four TRISC contexts (MATH, PACK, UNPACK, ISOLATE_SFPU)
- Unpack source/destination format arrays (visible to UNPACK, MATH, and ISOLATE_SFPU)
- Pack source/destination format arrays (visible to PACK and ISOLATE_SFPU)
- Tile dimension arrays for both unpack and pack

### How these files aid debugging

These generated files are the actual source that the device-side compiler sees. When a JIT compilation fails, the compiler error messages reference line numbers in these files. Inspecting them reveals:

- Whether the correct defines were propagated from the host program
- Whether data format arrays match expectations
- How the kernel source was transformed (especially for simplified syntax kernels)

### Where to find them

The generated files are written to a directory derived from the kernel name under the JIT output root. The exact path depends on the build configuration but typically follows the pattern:

```
<build_root>/kernels/<full_kernel_name>/chlkc_math.cpp
<build_root>/kernels/<full_kernel_name>/defines_generated.h
<build_root>/kernels/<full_kernel_name>/chlkc_descriptors.h
```

### How generated files are consumed

The `chlkc_list.h` file ties everything together. It includes `chlkc_descriptors.h` and the generated TRISC source file, then defines the `run_kernel()` entry point that dispatches to the appropriate namespace. On Blackhole and Wormhole, `chlkc_list.h` includes 3 generated files (`chlkc_math.cpp`, `chlkc_pack.cpp`, `chlkc_unpack.cpp`). Only Quasar's `chlkc_list.h` includes all 4, adding `chlkc_isolate_sfpu.cpp`:

```cpp
// tt_metal/hw/ckernels/blackhole/metal/common/chlkc_list.h:14-17
#ifdef UCK_CHLKC_MATH
#include "chlkc_descriptors.h"
#include "chlkc_math.cpp"
#endif
```

---

## Watcher infrastructure

The watcher is a host-side monitoring system that polls device cores for status information during execution. It is gated behind `WATCHER_ENABLED` and provides several debugging capabilities.

### Waypoints

Waypoints are 4-byte status codes that device firmware writes to a dedicated L1 mailbox. The watcher thread on the host reads these periodically to determine what each core is currently doing. Defined in `tt_metal/hw/inc/api/debug/waypoint.h`:

```cpp
// tt_metal/hw/inc/api/debug/waypoint.h:36-38
template <uint32_t x>
inline void write_debug_waypoint(volatile tt_l1_ptr uint32_t* debug_waypoint) {
    debug_waypoint[internal_::get_hw_thread_idx()] = x;
}
```

Usage in kernel code: `WAYPOINT("ABCD")` writes the 4-character tag to the mailbox. This is useful for identifying which phase of execution a core reached before it stalled.

### Assert mailbox

When `WATCHER_ENABLED` is active, the `ASSERT` macro (and by extension `LLK_ASSERT` in TT-Metal mode) writes the failing line number and hardware thread index into the `watcher.assert_status` mailbox before hanging. The host-side watcher server (`tt_metal/impl/debug/watcher_server.cpp`) reads this mailbox and reports the failure.

### NOC address sanitization

The watcher provides runtime bounds checking for NOC (Network-on-Chip) transactions. Defined in `tt_metal/hw/inc/internal/debug/sanitize.h`, this checks:

- Whether target NOC coordinates are valid
- Whether memory offsets are within legal ranges
- Whether the target core type is appropriate for the transaction

Invalid NOC addresses are logged to the watcher mailbox and cause the core to hang with diagnostic information available to the host.

### Hang detection

The `WatcherServer` class (`tt_metal/impl/debug/watcher_server.hpp`) runs a polling thread that periodically reads status from all active cores. If a core's waypoint has not changed for an extended period, the watcher reports it as potentially stalled. The server also checks the assert mailbox and NOC sanitization mailboxes on each poll.

The watcher logs are written to a file and include per-core status, waypoint values, and any assertion or sanitization failures.

---

## DPRINT

`DPRINT` is a device-side print facility that allows kernels to emit formatted text to a ring buffer in L1, which the host-side print server then reads and displays. It is defined in `tt_metal/hw/inc/api/debug/dprint.h` and gated behind `DEBUG_PRINT_ENABLED`:

```cpp
// tt_metal/hw/inc/api/debug/dprint.h:41-47
#if defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
#define DPRINT DebugPrinter()
#else
#define DPRINT \
    if (0)     \
    DebugPrinter()
#endif
```

### Usage in data movement kernels

Data movement kernels (BRISC, NCRISC) can use `DPRINT` directly:

```cpp
DPRINT << "value = " << my_variable << ENDL();
```

### Usage in compute kernels

Compute kernels run on the three TRISC processors. DPRINT is available on all three via context-gated macros:

```cpp
// tt_metal/hw/inc/api/debug/dprint.h:49-65
#ifdef UCK_CHLKC_UNPACK
#define DPRINT_UNPACK(x) x
#else
#define DPRINT_UNPACK(x)
#endif

#ifdef UCK_CHLKC_MATH
#define DPRINT_MATH(x) x
#else
#define DPRINT_MATH(x)
#endif

#ifdef UCK_CHLKC_PACK
#define DPRINT_PACK(x) x
#else
#define DPRINT_PACK(x)
#endif
```

Usage example from TT-Metal programming examples:

```cpp
DPRINT_MATH(DPRINT << "Hello from MATH core" << ENDL());
DPRINT_UNPACK(DPRINT << "Hello from UNPACK core" << ENDL());
DPRINT_PACK(DPRINT << "Hello from PACK core" << ENDL());
```

### Enabling DPRINT

DPRINT is enabled via the `TT_METAL_DPRINT_CORES` environment variable, which specifies which cores should have print enabled. The JIT build system adds `-DDEBUG_PRINT_ENABLED` to compile flags when enabled:

```cpp
// tt_metal/jit_build/build.cpp:219-221
if (rtoptions.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
    this->defines_ += "-DDEBUG_PRINT_ENABLED ";
}
```

### Effectiveness assessment

DPRINT is the closest thing to `printf` debugging available on device. It supports formatted output including hex, decimal, floats, and strings. The ring buffer mechanism means that if the buffer fills up, the device stalls waiting for the host to drain it, which can alter timing behavior. For compute kernels, the `DPRINT_MATH`/`DPRINT_UNPACK`/`DPRINT_PACK` wrappers correctly gate output to the appropriate TRISC context, preventing code compiled for one TRISC from accidentally emitting output intended for another.

---

**Next:** [`gaps_and_limitations.md`](./gaps_and_limitations.md)
