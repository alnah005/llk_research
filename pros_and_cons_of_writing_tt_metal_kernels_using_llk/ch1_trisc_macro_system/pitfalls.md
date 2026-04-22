# Pitfalls of the TRISC Macro System

## 1. Invisible Code Elimination

The macro system's greatest strength -- compile-time elimination -- is also its most insidious pitfall. When reading a kernel source file, **you cannot tell which lines actually execute on which TRISC** without mentally evaluating the preprocessor state.

Consider this init function from `eltwise_binary.h`:

```cpp
// tt_metal/hw/inc/api/compute/eltwise_binary.h, lines 31-43

ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb, ...) {
    state_configure(icb0, icb1, ocb, call_line);

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}
```

An experienced developer recognizes the `MATH(...)` / `PACK(...)` / `UNPACK(...)` pattern. But at the **call site** in `layernorm.cpp` (line 77), you just see:

```cpp
// ttnn/.../normalization/layernorm/device/kernels/compute/layernorm.cpp, line 77

binary_op_init_common(cb_in, cb_inb, cb_x);
```

Nothing in this line tells the reader that it configures hardware on all three TRISCs. If the reader is tracing a bug in the packer, they must look inside `binary_op_init_common` to discover that `llk_pack_init` is called there. The indirection through `ALWI`-inlined functions makes this worse -- the macro elimination is hidden behind abstraction layers.

The problem compounds in production kernels. In `bmm_large_block_zm_fused_bias_activation.cpp`, there are 21 `PACK(...)` calls scattered across 500 lines of code. Many are inside nested `#ifdef` blocks:

```cpp
// ttnn/.../matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp, lines 316-329

#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                                PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif

#ifdef PACKER_L1_ACC
#ifdef FUSE_BIAS
                                if (block == 0) {
                                    PACK((llk_pack_reconfig_l1_acc(0)));
                                } else {
                                    PACK((llk_pack_reconfig_l1_acc(1)));
                                }
#else
                                PACK((llk_pack_reconfig_l1_acc(0)));
#endif
#endif
```

A reader trying to understand the packer behavior must mentally evaluate **both** the TRISC macros **and** the feature-flag `#ifdef` nesting simultaneously. The interaction between these two preprocessor layers creates a combinatorial explosion of possible code paths that is extremely difficult to reason about statically.

**Impact:** New developers frequently misattribute bugs to the wrong TRISC pipeline because they cannot easily trace which code runs where.

## 2. Debugging Confusion: Line Numbers Do Not Match

When a TRISC binary crashes or produces incorrect results, the developer uses debug printing (`DPRINT`) or hardware watchpoints to identify the failing location. The reported line numbers correspond to the **preprocessed** source, not the original `.cpp` file.

The problem has two dimensions:

**Generated wrapper files shift line numbers.** The C preprocessor itself preserves line-number correspondence by emitting blank lines for eliminated code, so `MATH`/`PACK`/`UNPACK` expanding to nothing does not cause a mismatch. The actual source of confusion is the JIT compilation pipeline: `simple_kernel_syntax::transform_to_legacy_syntax()` inlines the transformed kernel source directly into a generated file (e.g., `chlkc_math.cpp`, `chlkc_pack.cpp`, `chlkc_unpack.cpp`). The preamble prepended by `build_trisc_prolog()` is minimal -- just a `#define` for the TRISC type and one `#include "defines_generated.h"` -- but it still shifts line numbers. For `FILE_PATH` kernels, the JIT pipeline mitigates this by emitting a `#line 1 "original.cpp"` directive (see `genfiles.cpp` line 214), which causes the compiler to report errors relative to the original source file. However, `SOURCE_CODE` type kernels (where the source is passed as a string rather than a file path) do not get this `#line` directive, so compiler errors report line numbers relative to the generated wrapper file. In that case, a reported "line 142" may correspond to line 95 in the actual kernel source, and the mapping changes whenever the wrapper boilerplate is modified.

**Template-heavy LLK calls produce opaque error messages.** Even when `#line` directives preserve file-level attribution, template errors remain problematic. Since `ALWI` forces inlining, compiler errors inside LLK functions like `llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>` are reported at the call site in the Compute API header, not at the kernel source line where `add_tiles` was called. The developer must trace through multiple layers of always-inlined functions to find the actual error location.

**Impact:** Debugging sessions are slower than they should be. Developers frequently resort to binary-search commenting-out of code blocks to isolate issues, rather than trusting line-number information from the toolchain.

## 3. Macro Hygiene: Commas and Complex Expressions

The `MATH(x)` / `PACK(x)` / `UNPACK(x)` macros take a single argument `x`. The C preprocessor treats commas as argument separators. This means any expression containing a comma **must be wrapped in extra parentheses** to be treated as a single macro argument.

The codebase has adopted a convention of double parentheses for all LLK calls:

```cpp
// tt_metal/hw/inc/api/compute/eltwise_binary.h, line 148

MATH((llk_math_eltwise_binary<ELWMUL, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
    icb0, icb1, idst, true)));
```

The outer parentheses belong to `MATH(...)`. The inner parentheses group the entire template call -- including its comma-separated template arguments -- into a single preprocessor token. Without the inner parentheses, the preprocessor would interpret the first comma in the template argument list as separating `MATH` arguments, producing a compile error about too many macro arguments.

This double-parenthesis convention is not enforced by any tooling. It is a learned pattern. When an author forgets the inner parentheses with a simple call, it may work fine:

```cpp
PACK(llk_pack_relu_config(ReluType::ZERO_RELU));  // Works: single function arg, no template commas
```

But with templates, the omission causes a confusing preprocessor error:

```cpp
// WRONG -- preprocessor sees 5 arguments to PACK():
PACK(llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(args));
```

The inconsistency between calls that need double-parens and calls that do not makes the convention fragile.

A more subtle variant appears in the DeepSeek unified kernels, where multi-statement blocks use double braces:

```cpp
// models/demos/deepseek_v3_b1/unified_kernels/eltwise_add.hpp, lines 152-155

UNPACK(({
    cb_in1_base_rd_ptr = get_local_cb_rd_ptr(CTArgs::cb_in1);
    update_local_cb_rd_ptr(CTArgs::cb_in1, cb_in1_base_rd_ptr + offset_aligned);
}));
```

Here the `({...})` is a GCC statement expression -- a non-standard extension -- used to smuggle multiple statements through a single-argument macro. This works but is obscure, compiler-specific, and would break under a standards-conforming preprocessor.

**Impact:** New contributors regularly encounter cryptic preprocessor errors when writing TRISC-gated code. The double-parenthesis convention adds visual noise and must be learned by example rather than from documentation.

## 4. No Cross-TRISC Static Analysis

The three TRISC binaries are compiled **independently**. No existing tooling verifies that the unpack, math, and pack stages are **synchronized** -- that is, that they agree on:

- The number of tiles processed per iteration.
- The order of circular-buffer operations.
- Which CBs are used for input vs. output.

Consider the SDPA kernel's `compute_common.hpp` where the author mixes Compute API calls (which internally contain UNPACK+MATH) with direct `MATH(...)` calls:

```cpp
// ttnn/.../transformer/sdpa/device/kernels/compute/compute_common.hpp, lines 880-883

    sub_tiles(in0_cb, in1_cb, i, i, 0);          // Contains UNPACK + MATH
    MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));  // MATH only
    pack_tile(0, out_cb);                          // PACK only
```

The `sub_tiles` call drives both UNPACK and MATH. The `exp_tile_first_column` drives only MATH. The `pack_tile` drives only PACK. If an author adds an extra `MATH(...)` call that changes the destination register without a corresponding packer adjustment, there is no compile-time or link-time error -- the kernel silently produces wrong results.

Similarly, in `bmm_large_block_zm_fused_bias_activation.cpp` the packer reconfiguration calls (`PACK((pack_reconfig_data_format(...)))`) must stay synchronized with the math pipeline's output format. If someone modifies the math path without updating the corresponding `PACK(...)` calls, the mismatch manifests as silent data corruption at runtime.

The `#ifdef TRISC_MATH` blocks used in SDPA for custom SFPI code (line 889) illustrate another gap:

```cpp
// ttnn/.../transformer/sdpa/device/kernels/compute/compute_common.hpp, lines 889-902

#ifdef TRISC_MATH
/**
 * The custom SFPI LLK function computes the following operation:
 * cur_max = max(prev_max, worker_max)
 * ...
 */
template <bool SDPA_EXP_APPROX_MODE>
void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    // ... direct sfpi:: register manipulation ...
```

This function is only visible to the MATH TRISC. The caller (`fused_max_sub_exp_add_tile`) is invoked inside a `MATH(...)` macro. But nothing verifies that the tiles the function operates on were actually unpacked by TRISC0 and will be packed by TRISC2. The correctness guarantee is entirely in the developer's head.

**Impact:** Pipeline synchronization bugs are among the hardest to diagnose in TT-Metal kernel development. They manifest as intermittent data corruption, hangs (when one TRISC waits on a CB that another never fills), or silent numerical errors. A static analysis tool that could verify cross-TRISC CB flow invariants would catch an entire class of bugs that currently require manual code review.

## 5. Difficulty Estimating Per-TRISC Code Size and Balance

Because code elimination is invisible, it is hard to estimate the **relative code size** of each TRISC binary from the source. A kernel that appears balanced in source lines may produce wildly unbalanced binaries.

In `bmm_large_block_zm_fused_bias_activation.cpp`, the PACK TRISC receives 21 macro-gated calls (mostly reconfiguration), while the MATH TRISC receives only the `matmul_block` inner loop (which is large due to inlined LLK code). The UNPACK TRISC receives the CB wait/pop calls. The source gives no visual indication of this imbalance.

Pipeline performance depends on balancing the three TRISCs. If one TRISC's binary is significantly larger or takes significantly more cycles than the others, it becomes the bottleneck and the other two stall. Without tooling to show per-TRISC instruction counts from the source, performance tuning requires examining the compiled binaries directly.

**Impact:** Developers cannot easily predict from source code which TRISC stage is the performance bottleneck. This makes performance optimization an empirical process of compiling, measuring, and iterating, rather than something that can be reasoned about from the source.

---

**Next:** [Chapter 2 -- Compute Kernel API Abstraction](../ch2_compute_kernel_api_abstraction/index.md)
