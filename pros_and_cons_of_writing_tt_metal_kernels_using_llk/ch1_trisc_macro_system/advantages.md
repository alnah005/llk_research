# Advantages of the TRISC Macro System

## 1. Single-Source Authoring

The most immediate benefit is that one `.cpp` file describes the **complete compute pipeline** -- unpack, math, and pack -- in a single reading order. A developer can understand the full data flow of an operation without switching between three separate files.

Consider the `add_tiles` function from the Compute API:

```cpp
// tt_metal/hw/inc/api/compute/eltwise_binary.h, lines 170-174

ALWI void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, true)));
}
```

Reading this single function, a developer immediately sees that an element-wise add requires (1) unpacking two tiles from CBs into source registers, then (2) invoking the FPU binary operation. The sequential listing maps directly to the hardware pipeline.

The same principle scales to full kernels. In `eltwise_binary.cpp`, the entire compute path for a binary operation with optional activations fits in ~110 lines:

```cpp
// ttnn/.../eltwise/binary_ng/device/kernels/compute/eltwise_binary.cpp, lines 48-57

        tile_regs_acquire();
        BINARY_OP(cb_post_lhs, cb_post_rhs, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
```

The kernel author does not need to reason about three separate programs. The acquire/compute/commit/wait/pack/release pattern reads like sequential code, even though three processors execute it concurrently.

**Why this matters:** Alternative architectures (e.g., CUDA's separate host/device split, or manually partitioned pipeline stages) require the developer to mentally reassemble data flow across multiple files. The single-source model reduces that cognitive load significantly for the common case.

## 2. Compile-Time Elimination

The `MATH(x)` / `PACK(x)` / `UNPACK(x)` macros expand to **nothing** on non-matching TRISCs. This means:

- No runtime branching cost -- eliminated code is not present in the binary at all.
- No dead-code warnings -- the preprocessor removes the statement before the compiler sees it.
- The resulting RISC-V binary for each TRISC is minimal and contains only its pipeline stage.

A clear example comes from the hardware startup function:

```cpp
// tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h, lines 41-51

ALWI void compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init<false, false, false>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>(ocb)));
    // ...
}
```

When compiled for TRISC0 (`-DTRISC_UNPACK`), only the `llk_unpack_hw_configure` call survives. The five `MATH(...)` and `PACK(...)` calls vanish entirely. There is no `if (thread_id == MATH) { ... }` runtime dispatch -- the binary simply does not contain the irrelevant code.

**Why this matters:** On small RISC-V cores with limited instruction cache, binary size directly affects performance. Compile-time elimination produces tight, per-TRISC binaries without any runtime overhead for pipeline dispatch.

## 3. Implicit Pipeline Parallelism

The three TRISCs run **concurrently** on separate hardware threads. The macro system lets the author write what looks like sequential code, while the hardware executes it as a three-stage pipeline.

The synchronization mechanism is built into the circular-buffer API. Functions like `cb_wait_front` and `cb_reserve_back` contain TRISC-specific implementations gated by `#ifdef`:

```cpp
// tt_metal/hw/inc/api/compute/cb_api.h, line 44

ALWI void cb_wait_front(uint32_t cbid, uint32_t ntiles) { UNPACK((llk_wait_tiles(cbid, ntiles))); }
```

`cb_wait_front` only executes on TRISC0 (unpack). TRISC1 and TRISC2 skip the call entirely. Similarly, `cb_reserve_back` and `cb_push_back` only execute on TRISC2 (pack). The CB semaphores handle cross-TRISC synchronization in hardware.

This means a kernel loop like the one in `bmm_large_block_zm_fused_bias_activation.cpp` (lines 233-401) implicitly distributes work across three processors:

```cpp
// ttnn/.../matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp

for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
    // ... [control flow shared by all TRISCs] ...

    cb_wait_front(in0_cb_id, in0_block_num_tiles);   // UNPACK only
    cb_wait_front(in1_cb_id, in1_block_num_tiles);   // UNPACK only

    // ... [matmul_block calls: UNPACK + MATH internally] ...

    cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);  // PACK only
    // ... [pack_tile_block: PACK only] ...
    cb_push_back(mm_out_cb_id, out_subblock_num_tiles);     // PACK only
}
```

The author writes a single loop. The preprocessor and CB synchronization turn it into a pipelined execution where TRISC0 is unpacking the next block while TRISC1 computes the current one and TRISC2 packs the previous result.

**Why this matters:** Explicit pipeline management (manually coding three communicating threads) is error-prone and verbose. The macro system makes the common case -- a three-stage pipeline -- look like straight-line code. The parallel execution is an emergent property of the compilation model rather than something the author must manually orchestrate.

## 4. ALWI Always-Inline Attribute Ensures No Call Overhead

Every Compute API function is marked with the `ALWI` attribute:

```cpp
// tt_metal/hw/inc/api/compute/compute_kernel_api.h, line 17

#define ALWI inline __attribute__((always_inline))
```

This guarantees that the compiler inlines every API function at every call site. Combined with compile-time elimination, this means:

- No function-call overhead (no stack frame setup, no register save/restore).
- The compiler can optimize across the inlined LLK calls (constant folding, dead store elimination, register allocation across the full pipeline stage).
- Template parameters (like `MATH_FIDELITY`, `DST_ACCUM_MODE`, `BroadcastType`) are resolved at compile time and propagated through the inlined code.

The LLK layer itself uses the same pattern. In the LLK repo, `TT_ALWAYS_INLINE` serves the same purpose:

```cpp
// tt_llk_blackhole/common/inc/ckernel.h, line 49

#define TT_ALWAYS_INLINE inline __attribute__((always_inline))
```

The entire call chain -- from `add_tiles()` through `llk_math_eltwise_binary()` down to the hardware register writes -- is fully inlined into the kernel's `kernel_main()` function.

**Why this matters:** On the Tensix RISC-V cores, function calls are expensive relative to the tight inner loops of tile processing. The always-inline guarantee means that the Compute API abstraction adds zero runtime cost. The author gets a readable, well-documented API surface with the performance characteristics of hand-written register manipulations.

## 5. Seamless Mixing of Shared and TRISC-Specific Code

The macro system naturally supports code that must run on all three TRISCs alongside code specific to one. Consider the mailbox read pattern in `bmm_large_block_zm_fused_bias_activation.cpp`:

```cpp
// ttnn/.../matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp, lines 209-214

UNPACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
MATH(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
PACK(is_batch_valid = (bool)mailbox_read(ckernel::ThreadId::BriscThreadId);)
if (!is_batch_valid) {
    continue;
}
```

All three TRISCs need to read the mailbox and skip invalid batches in sync. The author writes three macro-gated lines followed by shared control flow. The pattern is explicit about what runs where, while keeping the overall logic in one place.

Similarly, the `#if defined(TRISC_UNPACK) || defined(TRISC_MATH)` guard in `bcast.h` (line 34) allows code that must run on two of three TRISCs:

```cpp
// tt_metal/hw/inc/api/compute/bcast.h, lines 34-54

#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = ...;
    UNPACK((llk_unpack_A_init<bcast_type, ...>(...)));
    MATH((llk_math_eltwise_unary_datacopy_init<...>(icb)));
#endif
```

The shared `dst_format` computation runs on both UNPACK and MATH, while the actual LLK calls inside are still macro-gated to their respective TRISCs. This composability is a genuine strength of the preprocessor approach.

**Why this matters:** Real hardware operations do not always decompose cleanly into three independent stages. The macro system gives the author fine-grained control over which code runs where, without sacrificing the single-source property.

---

**Next:** [`pitfalls.md`](./pitfalls.md)
