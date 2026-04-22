# Gaps and Limitations of Current Debugging Tools

The tools described in [available_tools.md](./available_tools.md) represent real progress in making Tensix kernels debuggable. But each tool has constraints that, in practice, leave significant blind spots. This section evaluates those gaps honestly.

---

## 1. LLK_ASSERT is opt-in and off by default

The most fundamental limitation of `LLK_ASSERT` is that it is disabled in standard TT-Metal builds. The environment variable `TT_METAL_LLK_ASSERTS=1` must be explicitly set to activate assertions:

```cpp
// tt_metal/llrt/rtoptions.cpp:1267
case EnvVarID::TT_METAL_LLK_ASSERTS: this->enable_llk_asserts = true; break;
```

This means that the majority of kernel executions -- CI runs, benchmarks, production workloads -- proceed with no runtime validation of LLK invariants. The `sizeof` fallback in disabled mode preserves compile-time checking, which is valuable, but the runtime checks that catch actual misuse (wrong `num_faces`, invalid tile dimensions, etc.) are silent.

**Impact on developers:** A kernel author who carefully adds `LLK_ASSERT` guards will not see them fire unless they remember to set the environment variable. Bugs that violate LLK invariants may produce silently incorrect results or mysterious hangs, with no indication that an assertion would have caught the problem.

**What would improve this:** A debug build profile where LLK asserts are on by default, or a CI pipeline that routinely runs tests with assertions enabled.

### The message parameter is unused

The `message` argument to `LLK_ASSERT(condition, message)` is never transmitted to the host in any of the three modes. In TT-LLK infra mode, the macro issues a bare `ebreak`. In TT-Metal mode, it delegates to `ASSERT(condition)` which only records the line number. In disabled mode, the message is discarded entirely.

This means diagnostic messages like:

```cpp
LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4,
           "num_faces must be 1, 2, or 4");
```

exist only as source-code documentation. The developer debugging a failed assertion sees a line number but not the message. They must manually look up the source to understand what was being checked.

---

## 2. fake_kernels_target cannot validate runtime correctness or define consistency

The fake kernels target provides hardcoded stub values for all JIT-generated constants. The compile-time arguments are fixed at twenty `1` values:

```cmake
# tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:136
"KERNEL_COMPILE_TIME_ARGS=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
```

Data format arrays are filled with `255`, tile dimensions with `32`, and other descriptors with generic defaults:

```cpp
// tt_metal/jit_build/fake_kernels_target/fake_jit_prelude.h:23-27
constexpr unsigned char pack_src_format[32] = {
    255, 255, 255, 255, ...
};
```

### Consequence: false positives and false negatives

- **False negatives (missed errors):** A kernel that uses `get_compile_time_arg_val(5)` to select between two code paths will always take the path corresponding to argument value `1`. If the other path has a bug, it goes undetected.
- **False positives (spurious errors):** A kernel with a `static_assert` that requires a compile-time argument to be greater than `1` will fail the fake build even though it works correctly in practice. This can lead to developers ignoring fake build failures.

### Single TRISC context

The fake target only defines `TRISC_MATH=1` and `NAMESPACE=chlkc_math`. The `TRISC_PACK`, `TRISC_UNPACK`, and `TRISC_ISOLATE_SFPU` alternatives are commented out or absent:

```cmake
# tt_metal/jit_build/fake_kernels_target/CMakeLists.txt:124-128
# By default pretend that kernel is compiled for MATH core
TRISC_MATH=1
NAMESPACE=chlkc_math
# TRISC_PACK=1
# NAMESPACE=chlkc_pack
```

Code that is conditionally compiled for the pack, unpack, or isolate_sfpu TRISC contexts -- which includes all pack and unpack LLK API calls as well as isolated SFPU operations -- is never verified by the fake build. Three of the four TRISC contexts are missing from this build target.

---

## 3. DPRINT limitations in compute kernels

DPRINT is available on compute kernels (MATH, UNPACK, PACK TRISCs) via the `DPRINT_MATH`, `DPRINT_UNPACK`, and `DPRINT_PACK` wrappers. This is a significant capability. However, there are practical limitations:

### Timing perturbation

DPRINT operates via a ring buffer in L1. When the buffer is full, the device core stalls until the host drains it. In compute kernels that are tightly synchronized with data movement kernels through semaphores and circular buffers, this stall can change the relative timing of operations and mask or introduce bugs. A hang that only occurs without DPRINT -- or only occurs *with* it -- is extremely difficult to diagnose.

### No structured data inspection

DPRINT supports basic types (integers, floats, strings, hex formatting) but has no built-in support for dumping tile contents, register states, or circular buffer metadata in a structured way. Inspecting the 1024 elements of a tile requires manual loops and formatted output, which is tedious and consumes significant ring buffer space.

The `dprint_tensix_pack.h` and `dprint_tensix_unpack.h` helpers provide some structured output for pack/unpack register configurations, but these are specialized utilities rather than general-purpose inspection tools.

### Opt-in per core

DPRINT must be enabled for specific cores via `TT_METAL_DPRINT_CORES`. In a multi-core workload, enabling DPRINT on all cores may overwhelm the host print server or cause widespread timing perturbation. The developer must guess which core is misbehaving before they can add print statements -- a chicken-and-egg problem.

---

## 4. Hang diagnosis: watcher identifies the stalled core but cannot explain why

The watcher's hang detection works by polling waypoints. If a core's waypoint has not changed for an extended period, it is flagged as stalled. The watcher report includes:

- Which core is stalled
- The last waypoint value (a 4-character tag)
- Whether an assert was tripped (with line number)
- Whether a NOC sanitization error occurred

### What the watcher cannot tell you

- **Why the core is waiting.** A core stalled at waypoint `"WAIT"` tells you it reached a wait point, but not what condition it is waiting on. Is it waiting for a semaphore? A circular buffer to drain? A NOC transaction to complete? The waypoint does not carry enough information to distinguish these cases.

- **Deadlock analysis.** If core A is waiting on core B and core B is waiting on core A, the watcher reports both as stalled but does not identify the circular dependency. The developer must manually reconstruct the dependency chain from the waypoint values and kernel source.

- **The state of LLK internal registers.** When a compute kernel hangs inside an LLK function (e.g., during an unpack or math operation), the watcher has no visibility into the state of the Tensix configuration registers, the tile pipeline, or the SFPU. The developer cannot determine whether the hang is caused by a misconfigured register or a logic error in the kernel.

- **Root cause for "no waypoint change" scenarios.** A core that hangs between two waypoints could be stuck anywhere in the intervening code. Without finer-grained instrumentation, the developer can only narrow the problem to a potentially large code region.

---

## 5. No single-step debugging for TRISCs

There is no interactive debugger (GDB, LLDB, or equivalent) that can single-step through code running on TRISC processors. The RISC-V cores on Tensix do not expose a standard debug interface that would support breakpoints, register inspection, or memory watches from the host.

This means that the primary debugging methodology for compute kernels is:

1. Add `DPRINT` statements or `LLK_ASSERT` checks
2. Recompile (JIT)
3. Re-run the workload
4. Inspect output

Each iteration requires a full re-execution, which can take seconds to minutes depending on the workload. This is a significant productivity bottleneck compared to host-side development where an interactive debugger can inspect state without re-running.

The `ebreak` instruction used by `LLK_ASSERT` does halt the core, but there is no attached debugger to inspect the register file or memory at the point of the break. The halt is effectively a "crash" -- the only information extracted is the line number (if watcher is enabled).

---

## 6. Error messages reference generated files, not original source

When a JIT compilation fails, the compiler error messages reference line numbers in the generated files (`chlkc_math.cpp`, `chlkc_pack.cpp`, `chlkc_unpack.cpp`). The JIT system does emit `#line` directives for kernels using the simplified `kernel_main()` syntax:

```cpp
// tt_metal/jit_build/genfiles.cpp:213-214
if (kernel_src.source_type_ == KernelSource::FILE_PATH) {
    line_directive = "#line 1 \"" + kernel_src.path_.string() + "\"\n";
}
```

This helps map errors back to the original kernel file. However:

- **Legacy syntax kernels** (using `namespace NAMESPACE { void MAIN { ... } }`) that are included via `#include` already preserve file references. But those using `SOURCE_CODE` type (inline kernel source) have no original file to reference.
- **Errors in generated headers** -- such as `defines_generated.h` or `chlkc_descriptors.h` -- report line numbers within those generated files. The developer must locate the generated files under the build root and manually correlate the error with their host-side configuration code.
- **Template instantiation errors** deep inside LLK headers can produce error chains that reference multiple generated files and LLK library headers, making the actual root cause difficult to identify.

### Practical example

A typical error chain might look like:

```
.../chlkc_math.cpp:5: In function 'void chlkc_math::math_main()':
  .../llk_math_eltwise_binary.h:42: error: static_assert failed: "block_ct_dim must be < 128"
```

The developer must understand that `chlkc_math.cpp:5` is a generated file, trace through the `#include` chain, and determine which host-side `add_define()` or compile-time argument led to the invalid instantiation. This is manageable for experienced developers but presents a steep learning curve for newcomers.

---

## Summary: debugging capability matrix

| Scenario | Available tool | Effectiveness |
|----------|---------------|---------------|
| Invalid LLK API parameter at runtime | `LLK_ASSERT` | Good when enabled; invisible when disabled (default) |
| Kernel does not compile | JIT error + generated file inspection | Adequate, but errors reference generated files |
| Kernel compiles but produces wrong output | `DPRINT` | Usable but labor-intensive; no structured tile dump |
| Kernel hangs | Watcher waypoints + assert mailbox | Identifies stalled core; cannot explain root cause |
| NOC address error | Watcher NOC sanitization | Good; reports exact invalid address |
| Deadlock between cores | Watcher (partial) | Reports stalled cores; no dependency analysis |
| Register misconfiguration in Tensix | None | No visibility into hardware register state |
| Build-time kernel validation | `fake_kernels_target` | Catches syntax/type errors; limited by hardcoded stubs |

The tooling that exists is functional and in some cases well-engineered (the `sizeof` trick in `LLK_ASSERT`, the watcher mailbox protocol). But the overall debugging experience for LLK kernel development remains closer to embedded firmware debugging than modern software development. The absence of interactive debugging, the opt-in nature of runtime assertions, and the indirection through generated files all contribute to a steep debugging curve.

---

**Next:** [Chapter 7 -- SFPU Extension Path](../ch7_sfpu_extension_path/index.md)
