# Three-Layer API Architecture

The compute kernel stack is organized into three layers. This section traces a concrete call chain through all three to illustrate how they interact, and then assesses whether the middle layer pulls its weight.

## Layer 1: LLK Core (`_llk_*`)

**Location:** [`tt_llk_wormhole_b0/llk_lib/`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_wormhole_b0/llk_lib) (and equivalent directories for Blackhole and Quasar).

**Characteristics:**

- All functions use the `_llk_` prefix with leading underscore (e.g., `_llk_unpack_AB_init_`, `_llk_math_eltwise_binary_`, `_llk_pack_dest_init_`).
- ~148 inline functions across ~7,646 lines of header code (Wormhole B0 count).
- Operates directly on hardware registers, MOP (micro-operation) configurations, and address modes.
- Takes raw parameters -- data format integers, raw addresses, tile shapes -- with no operand or circular buffer abstraction.

Example from [`llk_lib/llk_unpack_AB.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_wormhole_b0/llk_lib/llk_unpack_AB.h):

```cpp
template <BroadcastType BType>
inline void _llk_unpack_AB_init_(
    const ckernel::TensorShape tensor_shape,
    const std::uint32_t transpose = 0) {
    // Directly configures unpacker hardware registers
    ...
}
```

## Layer 2: Metal LLK API (`llk_*`)

**Location:** [`hw/ckernels/wormhole_b0/metal/llk_api/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api) (21 header files, ~2,187 lines total).

**Characteristics:**

- Functions use the `llk_` prefix without the leading underscore (e.g., `llk_unpack_AB_init`, `llk_math_eltwise_binary_init`).
- ~116 inline functions that wrap the corresponding `_llk_*` calls.
- Primary job: resolve Metal operand IDs to tensor shapes, inject Metal-specific defaults (`ckernel::DEFAULT_TENSOR_SHAPE`), and reference compile-time globals (`DST_SYNC_MODE`, `DST_ACCUM_MODE`).

Example from [`llk_api/llk_unpack_AB_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_AB_api.h):

```cpp
template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0) {
    const std::uint32_t operandA_id = get_operand_id(operandA);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operandA_id);
    _llk_unpack_AB_init_<BType>(tensor_shape, transpose);
}
```

The pattern is consistent: look up the operand, extract a shape or format, forward to Layer 1.

Example from [`llk_api/llk_math_binary_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_binary_api.h):

```cpp
template <...>
inline void llk_math_eltwise_binary_init(const std::uint32_t acc_to_dest = 0) {
    _llk_math_eltwise_binary_init_<...>(ckernel::DEFAULT_TENSOR_SHAPE, acc_to_dest);
}
```

Here `DEFAULT_TENSOR_SHAPE` is the only value Layer 2 contributes. The entire function body is a single forwarding call.

## Layer 3: Compute Kernel API

**Location:** [`hw/inc/api/compute/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/api/compute) (105 header files, ~10,505 lines total).

**Characteristics:**

- Public API surface for op kernel authors: `binary_op_init_common()`, `binary_tiles_init()`, `pack_tile()`, `tile_regs_acquire()`, `tile_regs_commit()`, `tile_regs_release()`.
- Orchestrates calls across the three TRISC processors using `UNPACK()`, `MATH()`, `PACK()` macros.
- Injects compile-time configuration (`DST_ACCUM_MODE`, `MATH_FIDELITY`, `BroadcastType::NONE`) that Layer 2 functions consume.

Example from [`api/compute/eltwise_binary.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/inc/api/compute/eltwise_binary.h):

```cpp
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

## Complete Call Chain

Tracing a single operation from user code to hardware:

```
Layer 3:  binary_op_init_common(icb0, icb1, ocb)
            |
            +-- UNPACK( llk_unpack_AB_init<NONE>(icb0, icb1) )   [Layer 2]
            |     +-- _llk_unpack_AB_init_<NONE>(tensor_shape)   [Layer 1]
            |           +-- (hardware register writes)
            |
            +-- MATH( llk_math_pack_sync_init<DST_ACCUM_MODE>() ) [Layer 2]
            |     +-- _llk_math_pack_sync_init_()                  [Layer 1]
            |
            +-- PACK( llk_pack_init(ocb) )                         [Layer 2]
                  +-- _llk_pack_init_(...)                         [Layer 1]
```

## Assessment: Does Layer 2 Justify Its Existence?

Layer 2 is remarkably thin. The majority of its 116 functions follow a two-step pattern:

1. Resolve an operand CB index to a tensor shape or data format via `get_operand_id()` / `get_operand_tensor_shape()`.
2. Forward all arguments plus the resolved value to the corresponding `_llk_*` function.

This raises a design question: **Layer 2 exists primarily to inject Metal's operand abstraction and compile-time globals into LLK calls.** If those globals and the operand lookup were instead supplied by Layer 3 (the compute API), Layer 2 could potentially be eliminated, reducing the total indirection from three layers to two.

**Pros of keeping Layer 2:**

- It isolates LLK (Layer 1) from any knowledge of Metal's circular buffer / operand model.
- It provides a single place to inject defaults like `DEFAULT_TENSOR_SHAPE`, making it easy to change those defaults without modifying LLK.

**Cons of keeping Layer 2:**

- At ~2,187 lines of mostly mechanical forwarding, it adds maintenance burden with little logic of its own.
- Each new LLK function requires a corresponding wrapper in Layer 2, creating lockstep changes across repos.
- The forwarding pattern is so uniform that it could be auto-generated, which suggests the abstraction may not be earning its keep.

---

**Next:** [`leaky_abstractions.md`](./leaky_abstractions.md)
