# Pattern Ergonomics

The acquire/commit/wait/pack/release tile processing pattern provides several genuine advantages for kernel developers. This section examines each benefit with evidence from the codebase.

## 1. The Pipeline Is Explicit

The four-call protocol makes the hardware pipeline visible in source code. Every kernel clearly shows when the MATH thread owns the DST register versus when the PACK thread owns it:

```cpp
// From tt_metal/kernels/compute/eltwise_binary.cpp
tile_regs_acquire();           // MATH takes ownership of DST

for (uint32_t i = 0; i < per_core_block_size; ++i) {
    ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);
}
tile_regs_commit();            // MATH hands DST to PACK

tile_regs_wait();              // PACK takes ownership of DST
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    pack_tile(i, cb_out0);
}
tile_regs_release();           // PACK releases DST back to MATH
```

The definitions in `reg_api.h` make the separation concrete. `tile_regs_acquire` compiles to a call only on the MATH thread (`llk_math_wait_for_dest_available`), while `tile_regs_wait` compiles only on the PACK thread (`llk_packer_wait_for_math_done`):

```cpp
// From tt_metal/hw/inc/api/compute/reg_api.h
ALWI void tile_regs_acquire() {
    MATH((llk_math_wait_for_dest_available()));
}

ALWI void tile_regs_wait() {
    PACK((llk_packer_wait_for_math_done()));
}
```

This means the protocol is not just a convention but is enforced at compile time -- the MATH macro strips pack-only calls and vice versa, so each thread only sees the calls relevant to it. A developer reading the source sees both halves of the handshake in sequence, which directly maps to the hardware execution model.

## 2. Predictable Structure Across All Kernels

Nearly every compute kernel follows the same skeleton, regardless of operation complexity. This consistency means that once a developer learns the pattern from a simple kernel like eltwise binary, they can navigate layernorm, groupnorm, or matmul kernels without learning a fundamentally new structure.

Consider the inner loop structure visible across three different operation types:

**Layernorm (reduce stage)**:
```cpp
// From layernorm_sharded.cpp, lines 150-161
for (uint32_t i = 0; i < block_h; i++) {
    tile_regs_acquire();
    for (uint32_t w = 0; w < num_reduce_tiles_per_block_h; w++) {
        reduce_tile<...>(cb_in, cb_scaler, w + index_h_offset, scaler0, dst0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_ex_partial);
    tile_regs_release();
    index_h_offset += block_w;
}
```

**Groupnorm (mask and reduce)**:
```cpp
// From groupnorm_sharded_v2.cpp, lines 192-209
for (uint32_t j = 0; j < num_subblocks_w; ++j) {
    tile_regs_acquire();
    for (uint32_t w = 0; w < subblock_w; ++w) {
        mul_tiles(cb_in0, cb_input_mask, index, index_mask, w);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < subblock_w; ++i) {
        pack_tile(i, cb_x);
    }
    tile_regs_release();
}
```

**BMM (bias addition stage)**:
```cpp
// From bmm_large_block_zm_fused_bias_activation.cpp, lines 425-455
tile_regs_acquire();
for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
    for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
        add_tiles_bcast_rows(mm_partials_cb_id, bias_cb_id, i, bcast_tile_idx, i);
    }
}
tile_regs_commit();
// ...
tile_regs_wait();
for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
    pack_tile(i, untilize_mode_out_cb_id);
}
tile_regs_release();
```

The shape is always the same: acquire, compute in a loop, commit, wait, pack in a loop, release. The only variation is which math operation fills the compute loop and how many tiles are packed.

## 3. Helper Aliases Show Developers Create Shorthand

Several kernel files define helper functions that wrap the four-phase protocol into shorter aliases, which reveals both that the pattern is well-understood and that developers find the raw calls verbose.

**The `ACQ()`/`REL()` pattern using the deprecated API**:

```cpp
// From layernorm.cpp (lines 28-29) and compute_depthwise_conv1d.cpp (lines 23-24)
ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }
```

This older shorthand wraps both the MATH and PACK halves into a single call pair, which is simpler but hides the pipeline staging. The `acquire_dst` / `release_dst` functions are themselves now deprecated in favor of the four-call protocol (see the `[[deprecated]]` attributes in `reg_api.h`).

**The `ACQ()`/`REL()` pattern using the new API**:

```cpp
// From rmsnorm_pre_allgather.cpp (lines 21-28)
ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}
```

This variant collapses the four calls into two, merging the MATH acquire with the PACK wait and the MATH commit with the PACK release. This works for kernels that do not need to overlap MATH and PACK execution, and it makes the code significantly more compact:

```cpp
// From compute_depthwise_conv1d.cpp (lines 53-56)
ACQ();
mul_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
pack_tile(0, eltwise_mul_partials_cb_cb_id);
REL();
```

These aliases appear in at least five separate kernel files:
- `compute_depthwise_conv1d.cpp`
- `rmsnorm_pre_allgather.cpp` and `rmsnorm_pre_allgather_2d.cpp`
- `rmsnorm_post_allgather.cpp`
- `layernorm_pre_allgather_2d.cpp`
- `layernorm.cpp`

The fact that each file redefines these aliases independently (rather than importing from a shared header) hints at a missing standard abstraction.

## 4. Flexibility: Interleaving Operations Between Acquire and Commit

The four-phase protocol allows multiple math operations to run between a single `tile_regs_acquire()` and `tile_regs_commit()`, enabling complex compute sequences within one DST ownership window.

**Multiple operations in one acquire/commit window (layernorm rsqrt)**:

```cpp
// From layernorm_sharded.cpp, lines 318-328
tile_regs_acquire();
add_tiles_init(cb_ex2, cb_eps);
add_tiles(cb_ex2, cb_eps, i, 0, dst0);     // Var + eps
tile_regs_wait();                            // Note: interleaved wait
rsqrt_tile_init<LEGACY_RSQRT>();
rsqrt_tile<LEGACY_RSQRT>(dst0);             // 1/sqrt(Var + eps)
tile_regs_commit();
tile_regs_wait();
pack_tile(dst0, cb_ex2pe);
cb_push_back(cb_ex2pe, 1);
tile_regs_release();
```

This example is notable because it calls `tile_regs_wait()` twice — once between the `add_tiles` and `rsqrt_tile` operations, and again before `pack_tile`. Since `tile_regs_wait()` is defined as `PACK((llk_packer_wait_for_math_done()))`, it is completely stripped on the MATH thread — the MATH thread simply never sees it. The interleaving works because of the underlying semaphore mechanism: `tile_regs_wait()` calls `_llk_packer_wait_for_math_done_()`, which issues a `SEMWAIT(STALL_ON_ZERO)` — a non-consuming wait that blocks until the semaphore is non-zero but does not decrement it. `tile_regs_commit()` increments the semaphore, and `tile_regs_release()` decrements it. So the first `tile_regs_wait()` (line 157) blocks the PACK thread until `tile_regs_commit()` (line 160) fires and increments the semaphore. The second `tile_regs_wait()` (line 161) passes immediately because the semaphore is still non-zero — it has not yet been decremented by `tile_regs_release()`. This is not DST double-buffering; it is a consequence of the non-consuming semaphore wait semantics that allow multiple waits to succeed against a single commit.

**Accumulation across an entire inner dimension (BMM)**:

```cpp
// From bmm_large_block_zm_fused_bias_activation.cpp, lines 262-301
tile_regs_acquire();
if (enable_reload) {
    reload_from_cb_to_dst(...);  // reload partial sums
}
for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
    matmul_block(in0_cb_id, in1_cb_id, ...);  // accumulate into DST
}
// Optional SFPU activation fused inside the same acquire window
#if not defined FUSE_BIAS and defined SFPU_OP_INIT_ACTIVATION
for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
    SFPU_OP_FUNC_ACTIVATION
}
#endif
tile_regs_commit();
```

Here, the entire inner dimension reduction plus an optional activation runs within one DST ownership window, avoiding unnecessary DST handoffs.

**SDPA: DST held across an entire chunk computation (with custom synchronization)**:

The DeepSeek SDPA kernel in `sdpa_compute.cpp` holds `tile_regs_acquire()` across the entire multi-chunk flash attention computation, but it does **not** use the standard four-phase protocol for MATH/PACK synchronization. Instead, it relies on a custom `t6_semaphore` mechanism and hardware stalls:

```cpp
// From models/demos/deepseek_v3_b1/micro_ops/sdpa/kernels/sdpa_compute.cpp
MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
PACK(ckernel::t6_semaphore_init(SFPU_FPU, 0, 1));
// ...
tile_regs_acquire();
for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
    compute_sdpa_chunk<...>(...);  // entire QK matmul, softmax, OV matmul
}
// Pack occurs BEFORE tile_regs_commit, synchronized via semaphores:
for (uint32_t i = 0; i < num_tiles_v; i += 2) {
    PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
    pack_tile(mm2_dst_tile_offset + i, cb_out);
    pack_tile(mm2_dst_tile_offset + i + 1, cb_out);
    PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
}
tile_regs_commit();
// tile_regs_wait();   <-- commented out in the actual source
tile_regs_release();
```

This kernel demonstrates that at scale, the standard handshake can be insufficient. The `tile_regs_wait()` call is commented out, pack operations execute before `tile_regs_commit()`, and the MATH/PACK coordination is handled by low-level `t6_semaphore` waits rather than the four-phase protocol. This is a case where the developer has outgrown the standard pattern and reached for hardware-level synchronization primitives directly.

---

**Next:** [`common_mistakes.md`](./common_mistakes.md)
