# Developer Ergonomics

## Deep Include Chains

A compute kernel author writes a single include, but that include chains through multiple levels of headers before reaching the actual LLK implementation. The chain for a pack operation looks like this:

1. **Kernel source** includes `compute_kernel_api.h`
2. `[tt-metal] tt_metal/hw/inc/api/compute/compute_kernel_api.h` -- gates on `#ifdef TRISC_PACK` and includes `llk_pack_api.h`
3. `[tt-metal] tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h` -- includes `chlkc_list.h`, `ckernel.h`, `ckernel_defs.h`, `ckernel_template.h`, `cpack_common.h`, `ckernel_globals.h`, and six `llk_*.h` headers from LLK
4. `[tt-llk] tt_llk_wormhole_b0/llk_lib/llk_pack_common.h` -- includes `ckernel.h`, `ckernel_defs.h`, `ckernel_instr_params.h`, `cpack_common.h`, `llk_defs.h`
5. Each of those headers includes further internal headers (`ckernel_structs.h`, `risc_attribs.h`, etc.)

This means a kernel author's single `#include "compute_kernel_api.h"` expands through 5+ levels of headers spanning both the `[tt-metal]` and `[tt-llk]` repositories. When a compilation error occurs deep in this chain, the error message references a file the kernel author has never seen and may not know how to find.

## Macro-Heavy Kernel Authoring

Kernel behavior is configured not through function parameters or template arguments visible in the kernel source, but through preprocessor defines injected by the build system at compile time. Metal's host-side code passes these defines via `ComputeConfig::defines`. For example, from `[tt-metal] tt_metal/programming_examples/distributed/4_distributed_trace_and_events/distributed_trace_and_events.cpp` (lines 100-107):

```cpp
std::map<std::string, std::string> binary_defines = {
    {"ELTWISE_OP", op_id_to_op_define[eltwise_op_index]},
    {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op_index]}};
auto eltwise_binary_kernel = tt_metal::CreateKernel(
    *program,
    "tt_metal/kernels/compute/eltwise_binary.cpp",
    cores_for_program,
    tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});
```

Similarly, `PACK_RELU` is injected as a define in matmul factory code (e.g., `[tt-metal] ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`, line 562):

```cpp
mm_kernel_defines["PACK_RELU"] = "1";
```

Note that `DST_ACCUM_MODE` is generated at JIT time by `[tt-metal] tt_metal/jit_build/genfiles.cpp` (line 473) as a `constexpr bool`, not a preprocessor define:

```cpp
"constexpr bool DST_ACCUM_MODE = {};\n"
```

This means `DST_ACCUM_MODE` participates in the type system and benefits from compile-time checking, unlike the preprocessor defines above.

These build-injected defines have several drawbacks:

- **Invisible to the kernel author**: The kernel source references `ELTWISE_OP` or `PACK_RELU` but never defines them -- their values come from the host-side C++ code, making the kernel unreadable in isolation.
- **No type safety**: Preprocessor defines like `ELTWISE_OP` and `PACK_RELU` bypass the type system entirely. A typo in a define name silently results in an undefined macro rather than a compile error.
- **Combinatorial explosion**: The number of possible define combinations grows multiplicatively. Each distinct combination triggers a separate JIT compilation, increasing compile time and cache pressure.

## Legacy vs Simplified Kernel Syntax

Metal supports two syntactic styles for compute kernels, and the build system must detect and transform between them. This logic lives in `[tt-metal] tt_metal/jit_build/genfiles.cpp` (lines 60-134).

### The two styles

**Legacy syntax** (being deprecated):
```cpp
namespace NAMESPACE {
void MAIN() {
    // kernel body
}
}
```

**Simplified syntax** (preferred):
```cpp
void kernel_main() {
    // kernel body
}
```

### The detection and transformation logic

The `simple_kernel_syntax` namespace in `genfiles.cpp` implements detection via regex:

```cpp
const std::regex kernel_main_pattern(R"(\bvoid\s+kernel_main\s*\(\s*\)\s*\{)");
```

The detection has three heuristics (lines 88-106):
1. Check if `kernel_main()` exists in the source.
2. Check if legacy markers (`namespace NAMESPACE` or `void MAIN`) are present.
3. If both exist, assume legacy syntax (mixed kernel with multiple entrypoints).
4. If multiple `kernel_main()` definitions exist, throw an error.

When simplified syntax is detected, the build system transforms it to legacy syntax by splitting the source at `void kernel_main()`, wrapping the function body in the appropriate namespace, and renaming the function. This happens four times -- once per TRISC processor (lines 216-220):

```cpp
unpack_src = simple_kernel_syntax::transform_to_legacy_syntax(
    kernel_content, "chlkc_unpack", "unpack_main");
math_src = simple_kernel_syntax::transform_to_legacy_syntax(
    kernel_content, "chlkc_math", "math_main");
pack_src = simple_kernel_syntax::transform_to_legacy_syntax(
    kernel_content, "chlkc_pack", "pack_main");
isolate_sfpu_src = simple_kernel_syntax::transform_to_legacy_syntax(
    kernel_content, "chlkc_isolate_sfpu", "isolate_sfpu_main");
```

When legacy syntax is detected, a deprecation warning is emitted (lines 188-192):

```cpp
log_warning(
    tt::LogBuildKernels,
    "Compute kernel '{}' uses deprecated 'namespace NAMESPACE {{ void MAIN {{ }} }}' syntax. "
    "Please migrate to simplified 'void kernel_main() {{ }}' syntax.",
    settings.get_full_kernel_name());
```

### Problems with this coexistence

- **Regex-based source transformation**: The build system parses C++ source with regex, which is inherently fragile. Comments or strings containing `void kernel_main()` could confuse detection.
- **Two code paths through the build**: The simplified and legacy paths generate output differently (inlined transformed content vs. `#include` directive), so a bug may manifest in only one style.
- **Migration burden**: Until all kernels are migrated, both paths must be maintained and tested. The deprecation warning exists but there is no enforcement mechanism to prevent new legacy-style kernels from being added.

## Error Messages from Deep Template Instantiation

The LLK functions are heavily templated. For example, a single pack call in `[tt-metal] tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h` expands to:

```cpp
_llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, untilize>(tile_index, pack_tile_addr);
```

Where `DST_SYNC_MODE` is itself a macro-defined type, and `is_fp32_dest_acc_en` and `untilize` are template booleans. When a type mismatch or constraint violation occurs deep in this instantiation chain, the compiler error message references the internal LLK header (e.g., `llk_pack_common.h` inside `[tt-llk]`) rather than the kernel source the developer wrote. Combined with the 5+ level include chain, the developer must mentally traverse:

1. Their kernel source
2. The compute API header
3. The Metal llk_api wrapper
4. The LLK library header
5. The LLK internal implementation header

to understand why a compilation failed. The `#line` directive added for simplified-syntax kernels (genfiles.cpp, lines 213-215) helps with line numbers but does not simplify the template backtrace.

---

**Next:** [Chapter 4 -- API Abstraction Boundary Analysis](../ch4_api_boundary/index.md)
