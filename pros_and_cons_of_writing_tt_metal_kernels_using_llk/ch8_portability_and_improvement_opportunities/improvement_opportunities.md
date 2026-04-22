# Improvement Opportunities

This section synthesizes the friction points identified across all previous chapters into nine concrete improvement proposals. Each proposal is grounded in specific evidence from the TT-LLK and TT-Metal codebases and is ordered roughly by estimated impact-to-effort ratio.

---

## 1. RAII-Based DST Register Management

**Problem (Chapters 1, 4):** The DST register acquire/release protocol is the most error-prone aspect of kernel authoring. Every compute operation must be bracketed by `tile_regs_acquire()` / `tile_regs_commit()` on the MATH thread and `tile_regs_wait()` / `tile_regs_release()` on the PACK thread. Missing a release causes a deadlock. The old `acquire_dst()` / `release_dst()` pair is deprecated but still present, creating confusion:

```cpp
// tt_metal/hw/inc/api/compute/reg_api.h
[[deprecated("Use tile_regs_acquire() instead")]]
ALWI void acquire_dst() { ... }

[[deprecated("Use tile_regs_release() instead")]]
ALWI void release_dst() { ... }
```

**Proposal:** Introduce a `DstRegScope` RAII guard that acquires DST on construction and commits/releases on destruction:

```cpp
// Proposed addition to reg_api.h
struct DstRegScope {
    ALWI DstRegScope() { tile_regs_acquire(); }
    ALWI ~DstRegScope() { tile_regs_commit(); }
    DstRegScope(const DstRegScope&) = delete;
    DstRegScope& operator=(const DstRegScope&) = delete;
};

struct DstPackScope {
    ALWI DstPackScope() { tile_regs_wait(); }
    ALWI ~DstPackScope() { tile_regs_release(); }
    DstPackScope(const DstPackScope&) = delete;
    DstPackScope& operator=(const DstPackScope&) = delete;
};
```

Kernel code would change from:

```cpp
tile_regs_acquire();
// ... math operations ...
tile_regs_commit();
```

to:

```cpp
{
    DstRegScope dst;
    // ... math operations ...
}  // automatic commit
```

This eliminates an entire class of deadlock bugs. The RAII objects are zero-cost after inlining (the `ALWI` attribute ensures this) and provide automatic cleanup on scope exit even in the presence of early returns.

---

## 2. CB Synchronization Validation

**Problem (Chapters 1, 4):** Mismatched `cb_wait_front` / `cb_pop_front` / `cb_reserve_back` / `cb_push_back` calls between TRISCs are the most common source of silent hangs in kernel development. As documented in Chapter 1, there is no cross-TRISC static analysis to detect these mismatches. The CB API documentation itself warns about the constraints:

```
// tt_metal/hw/inc/api/compute/cb_api.h
// Important note: number of tiles used in all cb_* calls must evenly divide
// the cb size and must be the same number in all cb_wait_front calls in the
// same kernel.
```

**Proposal:** Add a compile-time or debug-mode runtime validator that tracks CB operations per thread:

- **Compile-time (ideal):** A `constexpr` analysis pass that counts CB push/pop operations per TRISC thread and flags mismatches. This is feasible because most kernel loops have statically determinable iteration counts.
- **Debug-mode runtime (pragmatic):** Shadow counters in debug builds that track cumulative push/pop/wait/reserve calls per CB index and assert when invariants are violated (e.g., pop count exceeds push count, or push count does not evenly divide CB size). These counters would be compiled out in release builds via `#ifdef DEBUG`.

Even a basic debug-mode counter that catches "popped more tiles than were pushed" would prevent hours of debugging silent hangs.

---

## 3. Define Schema and Contract Documentation

**Problem (Chapter 3):** The Compute Kernel API depends on a large number of preprocessor defines (`DST_ACCUM_MODE`, `MATH_FIDELITY`, `APPROX`, `UnpackToDestEn`, `A2D`, `MATH_ROWS`, `MM_THROTTLE`, etc.) that are injected by the build system. There is no single document or schema that lists all required defines, their valid values, and which API functions depend on them. Kernel authors discover requirements through compile errors or by reading implementation source code.

**Proposal:** Create a machine-readable schema file (YAML or JSON) that declares every build-system-injected define:

```yaml
defines:
  DST_ACCUM_MODE:
    type: bool
    values: [0, 1]
    required_by: [pack_tile, tile_regs_release, tile_regs_commit, mm_init, ...]
    injected_by: "tt_metal/hw/CMakeLists.txt"

  MATH_FIDELITY:
    type: enum
    values: [MathFidelity::LoFi, MathFidelity::HiFi2, MathFidelity::HiFi3, MathFidelity::HiFi4]
    required_by: [matmul_tiles, mm_init, mm_block_init, ...]
    injected_by: "Program::compile_compute_kernel()"
```

This schema would serve three purposes:
1. **Documentation** for kernel authors.
2. **Validation** at build time -- the build system can check that all required defines are set before compiling a kernel.
3. **Tooling foundation** for IDE integration (autocomplete, hover docs).

---

## 4. Automated Data Format Reconfiguration

**Problem (Chapters 2, 5):** Switching between operations that use different data formats (e.g., from eltwise unary to matmul) requires manual calls to `reconfig_data_format_srca`, `reconfig_data_format_srcb`, and/or `pack_reconfig_data_format`. The `reconfig_data_format.h` file alone contains 6 nearly identical functions, all wrapped in `#ifndef ARCH_QUASAR` guards (all 6 are TODO stubs on Quasar). Forgetting a reconfig call produces silently wrong results rather than an error.

**Proposal:** Extend the `state_configure` mechanism (which already tracks last-configured CB IDs to avoid redundant reconfiguration) to automatically trigger data format reconfiguration when the operand data formats change:

```cpp
// Currently, state_configure only checks CB IDs:
// state_configure(in1_cb_id, in0_cb_id, out_cb_id, call_line);

// Proposed: state_configure also compares data formats and
// automatically calls reconfig when they differ, eliminating
// the need for manual reconfig_data_format_* calls.
```

This would remove an entire category of boilerplate from kernel code and eliminate the "forgot to reconfigure" class of silent-corruption bugs. The `state_configure` function already has the CB IDs available; it just needs to also compare the data format fields in the CB interface structures.

---

## 5. Unified SFPU Registration (Replacing `sfpu_split_includes.h`)

**Problem (Chapter 7):** Adding a new SFPU operation requires wiring a new `#if SFPU_OP_<NAME>_INCLUDE` guard into `sfpu_split_includes.h`. This file currently has **46 conditional include blocks**, each gating a single header. The guards are defined by the build system, making it impossible to understand at the source level which SFPU ops are available. A missed guard means the operation silently does not exist.

```cpp
// tt_metal/hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h
#if SFPU_OP_ISINF_ISNAN_INCLUDE
#include "api/compute/eltwise_unary/isinf_isnan.h"
#endif
// ... 45 more blocks ...
```

**Proposal:** Replace the conditional-include pattern with a registry-based approach:

- **Option A (minimal change):** Replace `sfpu_split_includes.h` with a single `#include` of all SFPU headers, and use `static_assert` or `SFINAE` to ensure unused operations produce zero code. Modern compilers already eliminate unused `inline` functions; the conditional includes are an optimization from an era when compile times were more constrained.
- **Option B (structural change):** Introduce an SFPU operation registry that uses a macro or `constexpr` table to declare available operations. New operations register themselves via a single-line declaration rather than requiring edits to a central header:

```cpp
// In each SFPU op header:
REGISTER_SFPU_OP(abs, abs_tile_init, abs_tile);

// The registry auto-generates the sfpu_split_includes equivalent
```

Either option eliminates the manual wiring step that Chapter 7 identified as one of the six required steps for adding a new SFPU op.

---

## 6. Architecture Abstraction Layer

**Problem (this chapter):** The Compute Kernel API has 45 `#ifndef ARCH_QUASAR` guards across 10 files, with 31 TODO stubs. The underlying issue is that the LLK API functions themselves have different signatures across architectures (different template parameters, different function arguments). The `#ifdef` blocks in the Compute API are a direct consequence of LLK-layer signature divergence.

**Proposal:** Introduce a thin architecture abstraction layer between the Compute Kernel API and the LLK API:

```
Kernel .cpp  -->  Compute Kernel API  -->  Arch Abstraction Layer  -->  LLK API
                  (stable interface)       (per-arch impl files)       (arch-specific)
```

The abstraction layer would consist of per-architecture implementation files (e.g., `arch_impl_wormhole.h`, `arch_impl_blackhole.h`, `arch_impl_quasar.h`) that are selected at build time. Each file would implement the same set of `arch_*` functions with architecture-appropriate LLK calls:

```cpp
// arch_impl_wormhole.h
ALWI void arch_math_matmul_init(uint32_t in0, uint32_t in1, uint32_t transpose) {
    llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0, in1, transpose);
}

// arch_impl_quasar.h
ALWI void arch_math_matmul_init(uint32_t in0, uint32_t in1, uint32_t transpose) {
    llk_math_matmul_init<MATH_FIDELITY>();
}
```

The Compute Kernel API functions would then call `arch_math_matmul_init` instead of directly calling `llk_math_matmul_init`, eliminating all `#ifdef ARCH_QUASAR` blocks from the API headers. This is a significant refactor but would pay for itself by making every future architecture addition a localized change rather than a scattered edit across 9+ files.

---

## 7. Improved Error Diagnostics (Source Maps)

**Problem (Chapters 1, 6):** When a compile error or runtime failure occurs in a compute kernel, the error message references the generated per-TRISC source file (`chlkc_math.cpp`, `chlkc_pack.cpp`, `chlkc_unpack.cpp`) rather than the original kernel `.cpp` file. Line numbers do not correspond to the author's source. Macro expansion errors from the `MATH()` / `PACK()` / `UNPACK()` wrappers are especially opaque -- a missing inner parenthesis in `PACK((llk_pack<DST_ACCUM_MODE, false>(dst, cb)))` produces errors that point to macro internals.

**Proposal:** Generate `#line` directives in the per-TRISC output files that map back to the original kernel source:

```cpp
// Generated chlkc_math.cpp
#line 42 "ttnn/operations/matmul/device/kernels/compute/bmm.cpp"
llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst);
```

This is standard C/C++ practice for code generators and preprocessors. The `#line` directive costs nothing at runtime and makes compiler errors, debugger step-through, and `dprint` line references all point to the original source file. The code generation step that produces `chlkc_*.cpp` files would need to track source origin and emit these directives.

---

## 8. Lightweight SFPU Testing Harness

**Problem (Chapter 7):** Testing a new SFPU operation requires either running on hardware or setting up the full TT-Metal test infrastructure. There is no lightweight way to test SFPU math kernels in isolation. The TT-LLK repository has its own test framework (under `tests/`), but it requires hardware access and a complex build setup.

**Proposal:** Create a host-side SFPU testing harness that:

1. **Compiles SFPU math functions for x86** using a compatibility shim that maps SFPI vector intrinsics to standard C++ float operations.
2. **Provides a `DstRegMock`** that simulates the 16-tile DST register as a flat array.
3. **Includes golden-value comparisons** against reference implementations (e.g., `std::exp`, `std::sqrt`, `std::tanh`).

The TT-LLK SFPU functions are written in SFPI, which is a thin abstraction over vector register operations. Most SFPI operations map directly to scalar float equivalents. A compatibility header that provides `vFloat` as a wrapper around `float` and maps `dst_reg[i]` to array accesses would allow SFPU functions to compile and run on a host CPU with no hardware dependency.

This would enable:
- Unit testing of SFPU operations in CI without hardware
- Rapid iteration during SFPU development (seconds instead of minutes per test cycle)
- Numerical accuracy validation against arbitrary-precision reference implementations

---

## 9. Reducing Template Depth

**Problem (Chapter 5):** The LLK API uses deeply nested templates, with some call chains reaching 4+ levels of template instantiation. For example, a `pack_tile` call expands through:

```
pack_tile<out_of_order_output>  // Compute API
  -> llk_pack<DST_ACCUM_MODE, out_of_order_output, false>  // LLK API wrapper
    -> _llk_pack_<...>  // LLK implementation
      -> ckernel_template::run<...>  // Hardware instruction template
```

Each template parameter multiplies the number of instantiations, increasing compile times and binary size. The `ckernel_template` system alone can generate dozens of specializations per operation.

**Proposal:** Audit template parameters to identify which ones are truly needed at compile time versus which could be runtime parameters:

- **Keep as templates:** Parameters that affect instruction encoding (e.g., `MATH_FIDELITY`, which selects different hardware opcodes).
- **Convert to runtime:** Parameters that only affect control flow or register addresses (e.g., `out_of_order_output` in `pack_tile`, which just changes a pointer offset calculation). These can be `if constexpr` branches if the compiler can prove the branch is predictable, or simply regular `if` statements -- on an in-order embedded core, a predictable branch has near-zero cost.
- **Reduce arity:** Combine related boolean flags into a single enum or bitfield. The LLK pack functions take up to 3 boolean template parameters (`DST_ACCUM_MODE`, `untilize`, `zero_output`); a `PackConfig` enum with named values would be more readable and produce the same number of instantiations.

Even a modest reduction (e.g., converting 2 out of 4 template parameters to runtime) would halve the number of template instantiations and noticeably improve kernel compile times.

---

## Summary: Priority Matrix

| # | Proposal | Impact | Effort | Chapters Referenced |
|---|----------|--------|--------|---------------------|
| 1 | RAII DST register management | High (eliminates deadlocks) | Low | 1, 4 |
| 2 | CB synchronization validation | High (eliminates silent hangs) | Medium | 1, 4 |
| 3 | Define schema/contract | Medium (documentation + tooling) | Low | 3 |
| 4 | Automated data format reconfig | Medium (eliminates silent corruption) | Medium | 2, 5 |
| 5 | Unified SFPU registration | Medium (reduces SFPU onboarding steps) | Medium | 7 |
| 6 | Architecture abstraction layer | Very High (enables portability) | High | 2, 8 |
| 7 | Source map diagnostics | Medium (improves debug experience) | Low-Medium | 1, 6 |
| 8 | SFPU testing harness | Medium (enables host-side testing) | Medium | 7 |
| 9 | Reducing template depth | Low-Medium (compile time improvement) | Medium | 5 |

The highest-impact, lowest-effort improvement is **RAII DST register management** (proposal 1): it is a small, additive change that eliminates the single most common kernel bug. The highest-impact overall improvement is the **architecture abstraction layer** (proposal 6): it addresses the root cause of the portability problems documented throughout this chapter, but requires significant refactoring effort.

---

**End of guide.** Return to [Guide Index](../index.md)
