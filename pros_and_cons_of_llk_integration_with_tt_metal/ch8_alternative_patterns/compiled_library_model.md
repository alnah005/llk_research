# Alternative: Compiled Library Model

## Overview

In this alternative, LLK would be pre-compiled into static libraries (`.a` archives) for each
supported architecture (wormhole_b0, blackhole, quasar) and each relevant optimization level
(`-O3` for compute, `-Os` for other processors). Metal's JIT compilation pipeline would then
link against these pre-built libraries instead of including LLK headers directly.

This is the most structurally different alternative to the current integration model. It would
replace the header-only, compile-time integration documented in
[Ch1](../ch1_integration_architecture_overview/index.md) with a traditional library dependency.

## How It Would Work

1. **Build phase:** A separate build step (either in LLK's CI or as a pre-build stage in
   Metal) compiles each `_llk_*` function in
   `[tt-llk] tt_llk_<arch>/llk_lib/` into object files using `riscv-tt-elf-g++`. Because
   LLK functions are heavily templated, this step would need to explicitly instantiate all
   required template combinations -- for example, every valid combination of
   `<EltwiseBinaryType, BroadcastType, DstSync, bool is_fp32_dest_acc_en, MathFidelity, EltwiseBinaryReuseDestType>` for
   `_llk_math_eltwise_binary_`.

2. **Distribution:** The resulting `.a` files (one per architecture, per optimization level)
   are either committed to Metal's repo, published to an artifact store, or fetched during
   Metal's build.

3. **Consumption:** Metal's JIT pipeline (`JitBuildState` in
   `[tt-metal] tt_metal/jit_build/build.cpp`) would pass `-l` and `-L` flags to the
   cross-compiler instead of the current 14+ include paths. The Layer 2 wrappers in
   `[tt-metal] hw/ckernels/<arch>/metal/llk_api/` would include a minimal header declaring
   function signatures rather than pulling in the full implementation.

## Pros

### Faster kernel JIT compilation

The current model re-parses and re-compiles the full LLK header chain for every kernel
translation unit (`chlkc_math.cpp`, `chlkc_unpack.cpp`, `chlkc_pack.cpp`). As documented
in Ch6, deeply templated headers like `llk_math_eltwise_binary.h` (which includes
`ckernel_include.h`, `ckernel_ops.h`, `ckernel_template.h`, `cmath_common.h`, and more) are
processed from scratch each time. Pre-compiled libraries would eliminate this redundant
parsing, reducing JIT compilation latency for iterating op writers.

### Clearer API boundary

Today, the Layer 1/Layer 2 boundary is implicit: Metal's `llk_math_binary_api.h` directly
`#include`s `llk_math_eltwise_binary.h` from LLK and calls `_llk_math_eltwise_binary_init_<>`
(see Ch4, [leaky abstractions](../ch4_api_boundary/leaky_abstractions.md)). With a compiled
library, the boundary would be enforced by the linker. Metal could only call functions
declared in a public header, making the API surface explicit and auditable.

### Version independence

Pre-built libraries could be versioned and distributed independently. Metal could pin a
specific library version (e.g., `libllk_blackhole_v2.3.a`) without needing to synchronize
git submodule commits. This would eliminate the submodule update friction documented in
Ch3 ([coupling and synchronization](../ch3_disadvantages/coupling_and_synchronization.md)).

## Cons

### Loss of cross-module LTO and inlining

This is the critical problem. As documented in Ch2
([header-only benefits](../ch2_advantages/header_only_benefits.md)), the current model allows
the RISC-V cross-compiler to inline LLK functions directly into kernel code, eliminating
function call overhead and enabling dead code elimination across the LLK/Metal boundary.

With a pre-compiled library, the compiler cannot see LLK function bodies when compiling
kernel code. Even with LTO (`-flto=auto`, applied as a common flag to all RISC-V kernel compilations), cross-library
LTO is fragile and does not guarantee the same optimization quality as source-level inlining.

On the Tensix RISC-V cores, this matters acutely:
- Instruction memory is limited. Function call prologues/epilogues consume space.
- There is no virtual memory or instruction cache hierarchy to hide call overhead.
- The `constexpr` and `if constexpr` patterns used throughout LLK (e.g.,
  `is_high_fidelity(math_fidelity) ? 1 : 0` in `llk_math_eltwise_binary.h`) rely on
  compile-time constant propagation that would be lost across a library boundary.

### ABI stability requirements

Template functions cannot be pre-compiled without explicit instantiation. The build system
would need to enumerate every valid template parameter combination and instantiate each one.
For a function like `_llk_math_eltwise_binary_<>` with 6 template parameters (Ch6,
[compilation impact](../ch6_template_tradeoffs/compilation_impact.md)), the combinatorial
explosion is significant.

Any new template parameter value added in Metal (e.g., a new `EltwiseBinaryType` enumerator)
would require a coordinated LLK library rebuild and release. This negates much of the
"version independence" benefit.

### Larger binary sizes from unused instantiations

Pre-compiled libraries must instantiate all template combinations that any kernel might need.
A given kernel uses only a small subset of these. The linker can strip unused symbols from
static archives, but only at the object file granularity -- if a single object file contains
multiple instantiations, all are included. The current header-only model produces only the
exact instantiations each kernel requires, which is optimal for the memory-constrained
Tensix targets.

### Significant refactoring effort

Converting LLK from header-only to a compiled library would require:
- Modifying every file in `[tt-llk] tt_llk_<arch>/llk_lib/` (approximately 25+ headers per
  architecture) to separate declarations from definitions.
- Creating explicit template instantiation lists for each architecture.
- Reworking Metal's `JitBuildEnv::init()` and the HAL layer to pass linker flags instead of
  include paths.
- Updating `[tt-metal] tt_metal/jit_build/genfiles.cpp` to generate kernel code that calls
  library functions rather than including full implementations.
- Adding a library build and distribution pipeline to CI.

## Assessment

**Verdict: likely unsuitable.**

The compiled library model fundamentally conflicts with the performance requirements of the
Tensix RISC-V target. The benefits of the current header-only model (Ch2) -- inlining,
compile-time constant propagation, dead code elimination, zero call overhead -- are not
luxuries but necessities given the instruction memory constraints of the target hardware.

The JIT compilation latency problem is real (Ch3, Ch6), but it is better addressed through
build caching improvements (which Metal already implements via `JitBuildState`) and
precompiled headers than through a compiled library model that sacrifices runtime
performance.

This alternative would be more viable if the target hardware had a more generous instruction
memory budget, an instruction cache, or if LLK functions were less performance-critical.
None of these conditions hold for Tensix cores today.

---

**Next:** [`interface_stabilization.md`](./interface_stabilization.md)
