# Common Mistakes

The tile processing pattern's reliance on manual protocol adherence creates several classes of errors that are difficult to diagnose. This section catalogs the most common failure modes, with evidence from the codebase.

## 1. Protocol Violations Causing Silent Hangs

The four-phase protocol (`tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release`) is a semaphore-based handshake. If any call is missing or misplaced, the kernel silently deadlocks -- neither thread can proceed, and there is no error message, timeout, or stack trace.

**Forgetting `tile_regs_release`**: If the PACK thread never calls `tile_regs_release()`, the MATH thread's next `tile_regs_acquire()` blocks forever waiting for DST to become available. The kernel hangs with no diagnostic output.

**Forgetting `tile_regs_commit`**: If the MATH thread never calls `tile_regs_commit()`, the PACK thread's `tile_regs_wait()` blocks forever. Again, silent hang.

The underlying mechanism in `reg_api.h` makes the failure mode clear:

```cpp
ALWI void tile_regs_acquire() {
    MATH((llk_math_wait_for_dest_available()));  // blocks until PACK releases
}

ALWI void tile_regs_commit() {
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));  // signals PACK
}
```

These are bare semaphore operations with no timeout, no deadlock detection, and no assertion that the matching call will eventually arrive. The only recovery from a hang is to kill the process and inspect the kernel source manually.

## 2. No RAII or Scoped Guard for Destination Registers

Every DST acquire/release sequence is manually paired. There is no C++ RAII wrapper or scoped guard that ensures release happens on all code paths. Consider the conditional paths in the BMM kernel:

```cpp
// From bmm_large_block_zm_fused_bias_activation.cpp, lines 262-367
tile_regs_acquire();
// ... ~40 lines of matmul computation ...

if (last_out) {
    // ... SFPU activation ...
    tile_regs_commit();
    cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
    tile_regs_wait();
    // ... pack tiles ...
    tile_regs_release();    // release on path A
    cb_push_back(...);
} else {
    tile_regs_commit();
    // ... reserve and pack ...
    tile_regs_wait();
    // ... pack tiles ...
    tile_regs_release();    // release on path B
    cb_push_back(...);
}
```

Both branches must independently remember to call all four protocol functions in the correct order. If a future edit adds an early return or a third branch, missing the release is easy and the result is a silent hang. A scoped guard like the following does not exist in the codebase:

```cpp
// Hypothetical -- does not exist
{
    auto dst_lock = acquire_dst_guard();  // calls tile_regs_acquire()
    // ... compute ...
}  // destructor calls tile_regs_commit() + tile_regs_wait() + pack + tile_regs_release()
```

The ACQ/REL helper pattern discussed in the ergonomics section partially mitigates this but still relies on manual pairing.

## 3. CB Synchronization Bugs: Wait/Pop Count Mismatches

Circular buffer synchronization requires that `cb_wait_front` and `cb_pop_front` counts match exactly, and that `cb_reserve_back` and `cb_push_back` counts match exactly. A mismatch deadlocks the kernel with no error message.

**The wait-without-pop pattern**: In the groupnorm kernel, the CB is used in-place with a pop/reserve/push cycle inside a subblock loop:

```cpp
// From groupnorm_sharded_v2.cpp, lines 277-291
for (uint32_t j = 0; j < num_subblocks_w; j++) {
    tile_regs_acquire();
    for (uint32_t w = 0; w < subblock_w; w++) {
        sub_tiles_bcast_scalar(cb_x, cb_ex_global, index, 0, w);
    }
    tile_regs_commit();
    cb_pop_front(cb_x, subblock_w);       // pop subblock_w tiles
    cb_reserve_back(cb_x, subblock_w);    // reserve subblock_w tiles
    tile_regs_wait();
    for (uint32_t k = 0; k < subblock_w; k++) {
        pack_tile(k, cb_x);
    }
    cb_push_back(cb_x, subblock_w);       // push subblock_w tiles
    tile_regs_release();
    cb_wait_front(cb_x, block_hw);        // wait for all tiles again
}
```

This works correctly but is extremely fragile. The `cb_pop_front(cb_x, subblock_w)` call happens *before* `tile_regs_wait()`, which means the CB slot is freed before the pack has started. The pattern relies on the fact that the same CB is used for both input and output (in-place operation). If a developer changes `subblock_w` or `block_hw` without updating all the matching calls, the kernel deadlocks silently.

**The push-before-pop ordering hazard**: The layernorm kernel shows a pattern where `cb_push_back` is called before the corresponding `cb_wait_front`:

```cpp
// From layernorm_sharded.cpp, lines 131-137
cb_push_back(cb_in, num_tiles_per_block);
// ... reconfig ...
cb_wait_front(cb_in, num_tiles_per_block);
```

This ordering is correct (push makes tiles available, then wait reads them), but reversing these two lines would cause a deadlock. The correctness depends on the developer understanding that `push_back` signals the consumer side of the CB while `wait_front` reads from the consumer side.

## 4. Tile Index Management: Manual DST Index Tracking

The DST register file has 16 tile slots, and kernel code must manually track which slot each tile occupies. This tracking is done through integer variables and offset arithmetic with no bounds checking.

**Subblock offset arithmetic in layernorm**:

```cpp
// From layernorm_sharded.cpp, lines 116-129
index_subblock_w_offset = 0;
for (uint32_t j = 0; j < num_subblocks_w; j++) {
    tile_regs_acquire();
    for (uint32_t w = 0; w < subblock_w; w++) {
        index = w + index_subblock_w_offset + index_h_offset;
        add_tiles(cb_in0, cb_in1, index, index, w);  // DST index is 'w'
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < subblock_w; i++) {
        pack_tile(i, cb_in);                          // must match 'w' above
    }
    tile_regs_release();
    index_subblock_w_offset += subblock_w;
}
index_h_offset += block_w;
```

The third argument to `add_tiles` (`w`) is the DST register index. The first argument to `pack_tile` (`i`) must match. Both are derived from the same loop variable, but they are in separate loops with separate iterators (`w` and `i`). An off-by-one error in either loop would pack the wrong tile or access DST out of bounds, and neither condition produces a compile-time or runtime error.

**Dual-tile indexing in SFPU binary kernel**:

```cpp
// From eltwise_binary_sfpu_kernel.cpp, lines 110-165
tile_regs_acquire();
tile_regs_wait();
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    copy_tile(cb_inp0, i, i * 2);       // even DST slots
}
for (uint32_t i = 0; i < per_core_block_size; ++i) {
    copy_tile(cb_inp1, i, i * 2 + 1);   // odd DST slots
    BINARY_SFPU_OP                       // operates on pairs
    pack_tile(i * 2, cb_out0);           // pack even slot
}
tile_regs_commit();
tile_regs_release();
```

This kernel uses the `i * 2` / `i * 2 + 1` scheme to interleave two inputs into adjacent DST slots. The SFPU binary operation implicitly expects operands at `dst[i*2]` and `dst[i*2+1]`. If `per_core_block_size` exceeds 8 (half of 16 DST slots), the `i * 2 + 1` index would overflow the DST register with no bounds check.

## 5. Nested Acquire/Commit Is Undefined with No Warning

The DST register protocol does not support nesting. Calling `tile_regs_acquire()` twice without an intervening `tile_regs_commit()` is undefined behavior -- it may silently corrupt state or deadlock, depending on the semaphore implementation.

There is no compile-time check, no runtime assertion, and no documentation warning. The definitions in `reg_api.h` are simple semaphore operations:

```cpp
ALWI void tile_regs_acquire() {
    MATH((llk_math_wait_for_dest_available()));
}
```

If a developer wraps a helper function that internally calls `tile_regs_acquire()`/`tile_regs_commit()`, and then calls that helper inside an outer acquire/commit block, the behavior is undefined. The `reload_from_cb_to_dst` helper in the BMM kernel demonstrates this risk:

```cpp
// From bmm_large_block_zm_fused_bias_activation.cpp, lines 81-102
FORCE_INLINE void reload_from_cb_to_dst(...) {
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
    // Copies directly into DST -- no acquire/commit here
    copy_block_matmul_partials(mm_partials_cb_id, ...);
    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
    // Reconfigure for matmul
    mm_block_init_short_with_dt(...);
}
```

This function is called *inside* an existing `tile_regs_acquire()` block (line 262-264 of the same file). It works correctly because it does not itself call acquire/commit. But the function's contract -- that it must only be called while DST is already acquired -- is implicit and undocumented. If someone added acquire/commit calls inside this helper, the outer kernel would break.

## 6. The Deprecated vs. New API Split

The codebase contains two parallel APIs for DST management:

| Old API (deprecated) | New API |
|---|---|
| `acquire_dst()` | `tile_regs_acquire()` + `tile_regs_wait()` |
| `release_dst()` | `tile_regs_commit()` + `tile_regs_release()` |

The old API combines MATH and PACK synchronization into single calls. It is marked `[[deprecated]]` in `reg_api.h` with a reference to GitHub issue #5868, but it remains actively used in production kernels:

- The SDPA `compute_common.hpp` uses `acquire_dst()`/`release_dst()` throughout
- The `layernorm.cpp` kernel defines `ACQ()`/`REL()` wrappers around the deprecated API

The `rmsnorm_pre_allgather.cpp` kernel shows the migration pattern, where `ACQ()` now wraps the new four-call API:

```cpp
ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}
```

But `layernorm.cpp` still uses the old form:

```cpp
ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }
```

This inconsistency means developers reading different kernels will see different patterns for the same underlying operation, and may inadvertently mix the two APIs.

## Summary of Failure Modes

| Mistake | Symptom | Diagnosis Difficulty |
|---|---|---|
| Missing `tile_regs_release()` | Silent hang | High -- no error message |
| Missing `tile_regs_commit()` | Silent hang | High -- no error message |
| CB wait/pop count mismatch | Silent deadlock | High -- no error message |
| DST index out of bounds | Silent corruption | Very high -- wrong results only |
| Nested acquire/commit | Undefined behavior | Very high -- may work sometimes |
| Mixing deprecated/new API | Potential protocol violation | Medium -- compiler warnings |

The fundamental issue across all these failure modes is the same: the protocol relies entirely on the developer to maintain invariants that the toolchain does not enforce. There are no scoped guards, no runtime assertions, and no deadlock detection. When things go wrong, the only symptom is a hang or silently wrong output.

---

**Next:** [Chapter 5 -- Template-Heavy API Design](../ch5_template_heavy_api_design/index.md)
