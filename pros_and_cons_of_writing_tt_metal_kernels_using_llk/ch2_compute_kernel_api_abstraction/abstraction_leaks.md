# Abstraction Leaks

The Compute Kernel API provides a clean surface for simple operations, but production kernels routinely break through the abstraction. This section documents five categories of leaks, with evidence from real kernels in the TT-Metal codebase.

## 1. Manual `reconfig_data_format()` / `pack_reconfig_data_format()` Calls

**The problem:** When a kernel operates on circular buffers with different data formats (e.g., BFloat16 inputs, Float32 intermediates), it must manually reconfigure the unpacker and math unit to match each new operand. The Compute API provides `reconfig_data_format()` and its variants, but gives the developer full responsibility for calling them at the right time, with the right arguments.

**Scale of the issue:** The layernorm sharded kernel (`layernorm_sharded.cpp`) contains **18 calls** to `reconfig_data_format` variants. The SDPA flash decode kernel (`sdpa_flash_decode.cpp`) contains **32 calls**. The groupnorm sharded kernel (`groupnorm_sharded_v2.cpp`) contains **6 calls**.

Here is a representative excerpt from `layernorm_sharded.cpp`:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, lines 110-142
#ifdef FUSE_PRE_ADD
    reconfig_data_format_srcb(cb_in0, cb_in1);
    add_tiles_init(cb_in0, cb_in1);
    // ... compute block ...
#ifndef RMSNORM
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_scaler);
#else
    reconfig_data_format(cb_in0, cb_in, cb_in1, cb_in);
#endif
    cb_wait_front(cb_in, num_tiles_per_block);
#else
#ifndef RMSNORM
    reconfig_data_format_srcb(cb_in0, cb_scaler);
#endif
#endif
```

This pattern continues throughout the kernel. After computing partial reductions:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, line 164
reconfig_data_format_srca(cb_in, cb_ex_external);
```

After computing variance:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, lines 253-259
#if defined RMSNORM and not defined FUSED_PRE_ADD
    reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
#else
    if constexpr (FLOAT32_DTYPE) {
        reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
    }
#endif
```

Before the gamma multiply:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, line 389
reconfig_data_format(cb_im, cb_gamma);
```

Before the beta add:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, line 430
reconfig_data_format(cb_fusion, cb_beta);
pack_reconfig_data_format(cb_out);
```

**Why this is an abstraction leak:** The `reconfig_data_format` functions exist in the Compute API layer (`tt_metal/hw/inc/api/compute/reconfig_data_format.h`), but they expose the internal structure of the hardware pipeline. The developer must understand:

- That SRCA and SRCB are separate source registers with independently configurable data formats.
- That the math unit and unpacker must both be reconfigured in lockstep.
- That the packer has its own separate data format configuration (`pack_reconfig_data_format`).
- That two-argument vs. four-argument variants differ in whether they check the old format first.

The API provides the *mechanism* for reconfiguration but no *policy* about when it is needed. A developer who forgets a reconfig call gets silent data corruption, not a compile error.

## 2. CB Index Management: Manual `tt::CBIndex::c_0` Through `c_31` Tracking

**The problem:** Every compute kernel manually declares its circular buffer mapping as a block of `constexpr` assignments. There is no type system, no allocation scheme, and no checking that indices do not conflict.

From `layernorm_sharded.cpp`:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, lines 61-84
constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_eps = tt::CBIndex::c_3;
constexpr uint32_t cb_scaler_global = tt::CBIndex::c_4;
constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
constexpr uint32_t cb_beta = tt::CBIndex::c_6;
constexpr uint32_t cb_x = tt::CBIndex::c_24;
// ...
constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
constexpr uint32_t cb_ex = tt::CBIndex::c_9;
constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_11;
constexpr uint32_t cb_ex2 = tt::CBIndex::c_12;
constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_13;
constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;
constexpr uint32_t cb_xmm2 = cb_x;  // alias!
constexpr uint32_t cb_ex2pe = tt::CBIndex::c_20;
constexpr uint32_t cb_fusion = tt::CBIndex::c_18;
constexpr uint32_t cb_out = tt::CBIndex::c_16;
```

The SDPA flash decode kernel is even more extreme, using **28 distinct CB indices** (up to 29 with the conditional `c_8`):

```cpp
// File: ttnn/.../sdpa_flash_decode.cpp, lines 75-104
constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
// ... through ...
constexpr uint32_t cb_exp_max_diff_2 = tt::CBIndex::c_22;
constexpr uint32_t cb_out_accumulate_im_2 = tt::CBIndex::c_23;
// ... and more through c_31
```

**Why this is an abstraction leak:** The CB index is a hardware resource. It determines the physical memory layout on the Tensix core. The Compute Kernel API treats CB indices as opaque `uint32_t` values, but the developer must manually coordinate them with the reader and writer kernels on the same core. A mismatch between the compute kernel's `cb_gamma = tt::CBIndex::c_5` and the reader kernel's mapping causes silent data corruption.

Note the aliasing pattern: `constexpr uint32_t cb_xmm2 = cb_x;` on line 81 of `layernorm_sharded.cpp`. This is deliberate reuse of the same physical CB for different logical purposes at different points in the kernel. The API has no concept of CB lifetimes or reuse safety.

## 3. DST Register Management: Fully Manual Protocol With No Safety

**The problem:** The DST register is a shared, 16-tile buffer that serves as the handoff point between the MATH and PACK hardware threads. The Compute Kernel API exposes this as a four-function protocol:

```cpp
// File: tt_metal/hw/inc/api/compute/reg_api.h
ALWI void tile_regs_acquire() {  // MATH thread acquires DST
    MATH((llk_math_wait_for_dest_available()));
}
ALWI void tile_regs_commit() {   // MATH thread releases DST to PACK
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
}
ALWI void tile_regs_wait() {     // PACK thread waits for MATH to commit
    PACK((llk_packer_wait_for_math_done()));
}
ALWI void tile_regs_release() {  // PACK thread releases DST back to MATH
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}
```

Every compute loop in every kernel must follow the exact sequence: `acquire -> compute -> commit -> wait -> pack -> release`. Here is the pattern repeated in `layernorm_sharded.cpp`:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, lines 117-128 (one of many instances)
tile_regs_acquire();
for (uint32_t w = 0; w < subblock_w; w++) {
    index = w + index_subblock_w_offset + index_h_offset;
    add_tiles(cb_in0, cb_in1, index, index, w);
}
tile_regs_commit();
tile_regs_wait();
for (uint32_t i = 0; i < subblock_w; i++) {
    pack_tile(i, cb_in);
}
tile_regs_release();
```

This exact six-step pattern appears **11 times** in `layernorm_sharded.cpp` and **dozens of times** across complex kernels. It is pure boilerplate that every kernel must replicate verbatim.

**Why this is an abstraction leak:** The DST register protocol is the most fundamental hardware synchronization mechanism on the Tensix compute engine, and the API does nothing to encapsulate it. There are several specific problems:

- **No RAII or scope guards.** A missed `tile_regs_release()` deadlocks the core. A missed `tile_regs_acquire()` writes to unowned memory. Neither error is caught at compile time.
- **No bounds checking on DST indices.** The developer writes `pack_tile(i, cb_in)` where `i` must be less than 16 (the DST register size). Nothing enforces this.
- **The protocol is identical in every kernel.** The acquire-compute-commit-wait-pack-release pattern never varies, yet it must be manually written each time. A higher-level API could provide `with_dst_tiles(n, [&](auto dst) { ... })` or similar scoped abstractions.

## 4. Template Parameters Injected via Preprocessor Defines

**The problem:** Many Compute API functions depend on template parameters that are not passed by the kernel author but are injected by the build system as preprocessor defines. The most pervasive examples:

- **`DST_ACCUM_MODE`** -- Controls whether the destination register accumulates in Float32 or BFloat16. Used in nearly every LLK call.
- **`MATH_FIDELITY`** -- Controls the precision of math operations (LoFi, HiFi2, HiFi3, HiFi4).
- **`REDUCE_OP`** and **`REDUCE_DIM`** -- Must be defined before including the reduce header.

These are configured through `#define` statements at the top of kernel files:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, lines 5-9
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL
```

They flow through the Compute API into LLK calls as template arguments:

```cpp
// File: tt_metal/hw/inc/api/compute/eltwise_binary.h, line 148
MATH((llk_math_eltwise_binary<ELWMUL, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
    icb0, icb1, idst, true)));
```

**Why this is an abstraction leak:** These defines create implicit dependencies between the kernel source code and the build system. The developer must know:

- That `REDUCE_OP` and `REDUCE_DIM` must be defined *before* `#include "api/compute/reduce.h"`, or the code will not compile.
- That `DST_ACCUM_MODE` is injected by the build infrastructure and controls accumulation precision globally.
- That `MATH_FIDELITY` is similarly injected and affects the behavior of all math operations.
- That changing a `#define` at the top of the file changes the behavior of API functions hundreds of lines below with no visible connection.

The `constexpr` compile-time arguments (`get_compile_time_arg_val()`) are a similar pattern -- the kernel source references argument indices by number, with the actual values injected by the host program:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, lines 20-35
constexpr uint32_t is_top_row = get_compile_time_arg_val(0);
constexpr uint32_t do_gamma = get_compile_time_arg_val(1);
constexpr uint32_t do_beta = get_compile_time_arg_val(2);
constexpr uint32_t num_blocks_first_stage = get_compile_time_arg_val(3);
// ... indices must match the host-side SetRuntimeArgs() call exactly
```

The coupling between kernel source and host-side argument ordering is entirely convention-based.

## 5. `SFPU_OP_CHAIN` Preprocessor Code Injection

**The problem:** For fused activation functions (e.g., applying GELU after a layernorm), TT-Metal uses a preprocessor code injection system. The host-side code generates `#define` statements that expand to actual C++ function calls inside the kernel:

```cpp
// File: ttnn/.../unary_op_utils.cpp, lines 958-968
std::string init_def = fmt::format("SFPU_OP_CHAIN_{}_INIT_{}", block_id, i);
std::string func_def = fmt::format("SFPU_OP_CHAIN_{}_FUNC_{}", block_id, i);
block_define += init_def + " " + func_def + " ";
// ...
block_defines[fmt::format("SFPU_OP_CHAIN_{}", block_id)] = block_define;
```

This generates defines like `SFPU_OP_CHAIN_0` that expand to init/function call pairs. The kernel consumes them as raw token expansion:

```cpp
// File: ttnn/.../eltwise_sfpu.cpp, lines 31-33
#ifdef SFPU_OP_CHAIN_0
    SFPU_OP_CHAIN_0
#endif
```

The layernorm kernels use a similar but distinct pattern with `SFPU_OP_INIT_ACTIVATION` and `SFPU_OP_FUNC_ACTIVATION`:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, lines 360-368
#ifdef SFPU_OP_INIT_ACTIVATION
    if constexpr (!(do_gamma == 1 || do_beta == 1)) {
        SFPU_OP_INIT_ACTIVATION
        SFPU_OP_FUNC_ACTIVATION
    }
#endif
```

**Why this is an abstraction leak:** This is not just an abstraction leak -- it is an abstraction inversion. The host-side C++ code is generating preprocessor defines that contain compute kernel API calls, which are then token-pasted into the kernel source. The developer cannot:

- **See the actual code being executed.** The defines are generated at compile time by `unary_op_utils.cpp` on the host side. Looking at the kernel source alone, `SFPU_OP_CHAIN_0` is opaque.
- **Debug the generated code easily.** Errors in the macro expansion appear as cryptic template failures in LLK headers.
- **Compose operations safely.** The layernorm kernel must check `if constexpr (!(do_gamma == 1 || do_beta == 1))` to decide *where* to inject the activation, because the activation must be applied after the last mathematical transform. This conditional placement logic is spread across the kernel rather than being encapsulated.

## The Cumulative Effect

These five categories of leaks interact. A developer writing a layernorm kernel must simultaneously:

1. Track 15+ CB indices and ensure they match the reader/writer kernels.
2. Insert `reconfig_data_format()` calls at every transition point between operands of different formats.
3. Manually write the acquire-commit-wait-pack-release DST protocol around every compute block.
4. Define `REDUCE_OP`, `REDUCE_DIM`, `BCAST_LLKOP`, and `BCAST_DIM` before including headers.
5. Account for preprocessor-injected activation functions at the correct position in the computation.

The result is a 462-line kernel (`layernorm_sharded.cpp`) where the actual mathematical logic -- E[x], Var(x), normalize, scale, shift -- could be expressed in perhaps 30 lines of pseudocode. The remaining 430+ lines are hardware orchestration that the Compute Kernel API was supposed to abstract away.

---

**Next:** [Chapter 3 -- Defines as Configuration](../ch3_defines_as_configuration/index.md)
