# Abstraction Strengths

The Compute Kernel API gets several things right. When it works, it genuinely shields kernel authors from the three-pipeline complexity of the Tensix compute engine.

## 1. Descriptive Function Names That Map to Domain Concepts

The API surface reads like a tensor math library, not a hardware driver. Compare the Compute API call to the LLK calls it replaces:

**What the developer writes (Compute Kernel API):**
```cpp
// File: tt_metal/hw/inc/api/compute/eltwise_binary.h
ALWI void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, true)));
}
```

A single `add_tiles()` call replaces what would otherwise require the developer to know:
- That unpack must be called on the UNPACK RISC to stage tile data from CBs into source registers.
- That the math operation runs on a separate RISC thread and needs the `ELWADD` opcode, the accumulation mode, the math fidelity setting, and a reuse-dest policy.
- That these calls are dispatched to different hardware threads via the `UNPACK((...))` and `MATH((...))` macros.

The naming convention is consistent across the API: `mul_tiles()`, `sub_tiles()`, `mul_tiles_bcast_cols()`, `sub_tiles_bcast_scalar()`, `reduce_tile()`. A developer familiar with tensor operations can read kernel code and understand the data flow without consulting the LLK reference.

Similarly, shorthand init functions like `mul_tiles_init()`, `add_tiles_init()`, and `sub_tiles_init()` map cleanly to their operation:

```cpp
// File: tt_metal/hw/inc/api/compute/eltwise_binary.h
ALWI void mul_tiles_init(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    binary_tiles_init<true, ELWMUL>(icb0, icb1, false, call_line);
}
```

These wrappers eliminate the need to remember template parameters like `ELWMUL` or `true` for `full_init`.

## 2. Single Calls That Orchestrate All Three Pipelines

The most valuable abstraction the API provides is `binary_op_init_common()`, which initializes all three hardware pipelines in one call:

```cpp
// File: tt_metal/hw/inc/api/compute/eltwise_binary.h
ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
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

Without this function, every kernel would need to manually call seven LLK functions across three pipeline macros, passing the correct accumulation mode to each. This is the pattern that every layernorm, groupnorm, and SDPA kernel begins with:

```cpp
// File: ttnn/.../layernorm_sharded.cpp, line 86
binary_op_init_common(cb_in0, cb_in0, cb_x);
```

```cpp
// File: ttnn/.../groupnorm_sharded_v2.cpp, line 155
binary_op_init_common(cb_in0, cb_in0, cb_in);
```

One line replaces a page of boilerplate. This is the API working as intended.

## 3. `state_configure()` -- Avoiding Redundant Hardware Reconfiguration

Hardware reconfiguration on the Tensix compute engine is not free. Changing the data format for the unpacker or math unit involves writing to configuration registers, which stalls the pipeline. The `state_configure()` system tracks the current configuration state and skips reconfiguration when the requested state matches the current one:

```cpp
// File: tt_metal/hw/inc/api/compute/sentinel/compute_kernel_sentinel.h
template <Operand operand = Operand::SRCA>
ALWI void state_configure(uint32_t cb, uint32_t call_line) {
    ComputeKernelSentinel::instance().reconfigure_single_operand<operand>(cb, call_line);
}
```

This is embedded in every init function. When a kernel calls `mul_tiles_init(cb_x, cb_x)` followed by `add_tiles_init(cb_x, cb_x)` on the same operands, the second call's `state_configure` can detect that SRCA and SRCB are already configured for `cb_x` and skip the hardware register writes.

This is a meaningful optimization that would be error-prone if left to the kernel author. The sentinel pattern effectively implements a "configuration cache" for the three-pipeline system.

## 4. Doxygen-Style Parameter Documentation

The API headers include structured documentation tables for every public function. For example:

```cpp
// File: tt_metal/hw/inc/api/compute/eltwise_binary.h
/**
 * Performs element-wise multiplication C=A*B of tiles in two CBs at given
 * indices and writes the result to the DST register at index dst_tile_index.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * Note: `acquire_dst` is the deprecated name; the current API uses
 * `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release`.
 * This call is blocking and is only available on the compute engine.
 *
 * | Argument       | Description                                              | Type     | Valid Range                                    | Required |
 * |----------------|----------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in0_cb_id      | The identifier of the circular buffer (CB) containing A  | uint32_t | 0 to 31                                        | True     |
 * | in1_cb_id      | The identifier of the circular buffer (CB) containing B  | uint32_t | 0 to 31                                        | True     |
 * | in0_tile_index | The index of tile A within the first CB                  | uint32_t | Must be less than the size of the CB           | True     |
 * | in1_tile_index | The index of tile B within the second CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result C        | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
```

This structured format specifies types, valid ranges, and whether parameters are required. The `pack.h` header is especially thorough, documenting the relationship between `cb_reserve_back`, `pack_tile`, and `cb_push_back` and the write pointer semantics:

```cpp
// File: tt_metal/hw/inc/api/compute/pack.h
/**
 * Copies a single tile from the DEST register buffer at a specified index to a
 * specified CB at a given index. For the out_tile_index to be valid for this
 * call, cb_reserve_back(n) has to be called first to reserve at least some
 * number n > 0 of tiles in the output CB. ...
 *
 * A typical use case is first the producer ensures that there is a number of
 * tiles available in the buffer via cb_reserve_back, then the producer uses
 * the pack_tile call to copy a tile from one of DEST slots to a slot in
 * reserved space and finally cb_push_back is called to announce visibility of
 * the reserved section of the circular buffer to the consumer.
 */
```

This documentation also captures the deprecation of old APIs with actionable migration guidance:

```cpp
// File: tt_metal/hw/inc/api/compute/reg_api.h
[[deprecated("Use tile_regs_acquire() instead")]]
ALWI void acquire_dst() { ... }
```

The combination of structured parameter tables, usage examples in docstrings, and deprecation annotations makes the Compute Kernel API one of the better-documented layers in the TT-Metal stack.

## Summary

The Compute Kernel API succeeds in four areas:

1. **Naming** -- Function names map to domain operations, not hardware mechanics.
2. **Pipeline consolidation** -- Single calls like `binary_op_init_common()` replace multi-pipeline boilerplate.
3. **State caching** -- `state_configure()` prevents redundant hardware reconfiguration.
4. **Documentation** -- Structured parameter tables and usage notes make the API self-documenting.

These strengths hold well for simple kernels: an element-wise add, a basic reduction, a tile copy. The problems begin when kernels need to change data formats mid-computation, manage many circular buffers simultaneously, or inject custom SFPU operations -- which is precisely what production kernels need to do.

---

**Next:** [`abstraction_leaks.md`](./abstraction_leaks.md)
