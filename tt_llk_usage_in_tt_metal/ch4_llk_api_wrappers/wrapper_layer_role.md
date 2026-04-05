# The Wrapper Layer: `llk_api/` Headers

## Location and scope

```
tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/
    llk_math_binary_api.h
    llk_math_binary_sfpu_api.h
    llk_math_common_api.h
    llk_math_matmul_api.h
    llk_math_reduce_api.h
    llk_math_reduce_custom_api.h
    llk_math_transpose_dest_api.h
    llk_math_unary_datacopy_api.h
    llk_math_unary_sfpu_api.h
    llk_pack_api.h
    llk_param_structs.h
    llk_sfpu_types.h
    llk_unpack_AB_api.h
    llk_unpack_AB_matmul_api.h
    llk_unpack_AB_reduce_api.h
    llk_unpack_AB_reduce_custom_api.h
    llk_unpack_A_api.h
    llk_unpack_common_api.h
    llk_unpack_reduce_api.h
    llk_unpack_tilize_api.h
    llk_unpack_untilize_api.h
    llk_sfpu/                          ← 158 SFPU extension files (covered in sfpu_extensions.md)
```

Each `*_api.h` file provides the `llk_<operation>()` functions that the Compute Kernel API calls. These are **not** the same as the `_llk_<operation>_()` functions in tt-llk -- they are a thin layer that resolves operand metadata and delegates downward.

## The pattern: resolve, then delegate

Every wrapper follows the same structure:

1. Accept a high-level **operand identifier** (an integer index into the circular buffer table).
2. Call helper functions from `llk_operands.h` or `llk_outputs.h` to extract tile geometry and format information.
3. Delegate to the corresponding `_llk_<operation>_()` function from tt-llk's `llk_lib/` headers.

### Example: `llk_math_matmul_init()`

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h, lines 13-32

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void llk_math_matmul_init(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1) {
    const std::uint32_t in0_id = get_operand_id(operandA);
    const std::uint32_t in1_id = get_operand_id(operandB);

    const std::uint32_t in0_tile_r_dim = get_operand_tile_r_dim(in0_id);
    const std::uint32_t in0_tile_c_dim = get_operand_tile_c_dim(in0_id);
    const std::uint32_t in1_tile_r_dim = get_operand_tile_r_dim(in1_id);
    const std::uint32_t in1_tile_c_dim = get_operand_tile_c_dim(in1_id);

    const bool partial_face = (in0_tile_r_dim < FACE_R_DIM);

    _llk_math_matmul_init_<math_fidelity, THROTTLE_LEVEL>(
        in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim,
        partial_face, transpose, ct_dim, rt_dim);
}
```

The caller provides `operandA` and `operandB` as abstract identifiers. The wrapper resolves each operand's row/column tile dimensions via `get_operand_tile_r_dim()` and `get_operand_tile_c_dim()`, determines whether partial faces are in play, then forwards everything to `_llk_math_matmul_init_()` from `llk_math_matmul.h` (included on line 7).

The actual execution wrapper follows the same pattern:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h, lines 34-43

template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0, uint32_t num_faces = 4>
inline void llk_math_matmul(const uint dst_index, const std::uint32_t ct_dim = 1,
                            const std::uint32_t rt_dim = 1) {
    static_assert(num_faces == 4,
        "num_faces other than 4 is not supported in llk_math_matmul");
    LLK_ASSERT((ckernel::math::get_dest_max_matmul_tiles(dst_index, ct_dim, rt_dim) <
         get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    _llk_math_matmul_<math_fidelity, THROTTLE_LEVEL>(dst_index, ct_dim, rt_dim);
}
```

Here the wrapper adds a `static_assert` constraint and a runtime `LLK_ASSERT` bounds check before calling through.

## Operand ID translation

The operand resolution helpers live in `llk_operands.h` (for unpack-side inputs) and `llk_outputs.h` (for pack-side outputs).

### Input operands (`llk_operands.h`)

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_operands.h, lines 11-47

inline uint32_t get_operand_id(uint32_t operand) { return (operand); }

inline const uint32_t get_operand_src_format(const std::uint32_t operand_id) {
    return unpack_src_format[operand_id];
}
inline const uint32_t get_operand_dst_format(const std::uint32_t operand_id) {
    return unpack_dst_format[operand_id];
}
inline const uint32_t get_operand_num_faces(const std::uint32_t operand_id) {
    return (uint32_t)unpack_tile_num_faces[operand_id];
}
inline const uint32_t get_operand_face_r_dim(const std::uint32_t operand_id) {
    return (uint32_t)unpack_tile_face_r_dim[operand_id];
}
inline const uint32_t get_operand_tile_r_dim(const std::uint32_t operand_id) {
    return (uint32_t)unpack_tile_r_dim[operand_id];
}
inline const uint32_t get_operand_tile_c_dim(const std::uint32_t operand_id) {
    return (uint32_t)unpack_tile_c_dim[operand_id];
}
```

In TT-Metal's implementation, `get_operand_id()` is an identity function -- the operand index passed by the Compute Kernel API maps directly to a position in the global `unpack_*` arrays. These arrays (`unpack_src_format[]`, `unpack_tile_r_dim[]`, etc.) are populated at kernel compile time from the program's circular buffer configuration.

A more complex helper constructs a `TensorShape` struct:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_operands.h, lines 41-47

inline ckernel::TensorShape get_operand_tensor_shape(const std::uint32_t operand_id) {
    return ckernel::TensorShape{
        unpack_tile_face_r_dim[operand_id],
        ckernel::MAX_FACE_C_DIM,
        unpack_num_faces_r_dim[operand_id],
        unpack_num_faces_c_dim[operand_id]};
}
```

This is used by wrappers that need the full tile shape, such as `llk_math_eltwise_binary_init_with_operands()`.

### Output operands (`llk_outputs.h`)

The pack side mirrors this with `get_output_id()`, `get_output_face_r_dim()`, `get_output_num_faces()`, etc., reading from the `pack_*` global arrays:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_outputs.h, lines 10-43

inline uint32_t get_output_id(uint32_t output) { return (output); }

inline const uint32_t get_output_num_faces(const std::uint32_t output_id) {
    return (uint32_t)pack_tile_num_faces[output_id];
}
inline const uint32_t get_output_face_r_dim(const std::uint32_t output_id) {
    return (uint32_t)pack_tile_face_r_dim[output_id];
}
inline const uint32_t get_output_partial_face(const std::uint32_t output_id) {
    return (uint32_t)pack_partial_face[output_id];
}
inline const uint32_t get_output_narrow_tile(const std::uint32_t output_id) {
    return (uint32_t)pack_narrow_tile[output_id];
}
```

The `OUTPUT_BASE_ID` constant is 16, reflecting that output circular buffers start at index 16 in the unified CB address space.

## Subsystem-specific wrappers

### Unpack wrappers (TRISC0)

The unpack wrappers handle address computation from circular buffer read pointers. For example, `llk_unpack_A()`:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_A_api.h, lines 63-86

template <BroadcastType BType, bool acc_to_dest, EltwiseBinaryReuseDestType binary_reuse_dest,
          bool unpack_to_dest>
inline void llk_unpack_A(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;

    // ... LLK_ASSERT for configuration correctness ...

    WAYPOINT("UPAW");
    _llk_unpack_A_<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        address, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
    WAYPOINT("UPAD");
}
```

Key points:
- The tile's L1 address is computed from `fifo_rd_ptr` (the circular buffer read pointer) plus a page-size-based offset.
- The `-1` on the base address accounts for the hardware's 1-based addressing convention.
- `WAYPOINT` macros emit profiling markers for kernel performance analysis.
- The `_llk_unpack_A_()` call passes the resolved address along with source/destination data formats.

For dual-input operations, `llk_unpack_AB()` resolves two operands and computes two addresses:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_AB_api.h, lines 28-91

template <BroadcastType BType>
inline void llk_unpack_AB(
    const std::uint32_t operandA, const std::uint32_t operandB,
    const std::uint32_t tile_index_a, const std::uint32_t tile_index_b,
    const std::uint32_t bcast_row_idx = 0) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_a = get_local_cb_interface(operandA_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_a = get_local_cb_interface(operandA_id).fifo_page_size * tile_index_a;
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = get_local_cb_interface(operandB_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address_b = get_local_cb_interface(operandB_id).fifo_page_size * tile_index_b;
    std::uint32_t address_b = base_address_b + offset_address_b;

    // ... row broadcast address adjustment for BType == BroadcastType::ROW ...

    _llk_unpack_AB_<BType>(address_a, address_b);
}
```

The wrapper also handles row-broadcast address adjustment, computing byte offsets within tile faces when `bcast_row_idx > 0`.

### Math wrappers (TRISC1)

Math wrappers are generally thinner since the FPU does not interact with circular buffers. The primary wrapper concerns are:
- Template parameter forwarding (`MathFidelity`, `EltwiseBinaryType`, `BroadcastType`)
- Destination register bounds checking
- Operand-to-`TensorShape` conversion for non-square tile support

The binary elementwise wrapper provides two overloads -- one that assumes a default 32x32 tile shape and one that resolves tile shape from operand IDs:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_binary_api.h, lines 19-25

// Version with no operand (assumes default 32x32 tile)
template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type,
          MathFidelity math_fidelity, EltwiseBinaryReuseDestType binary_reuse_dest>
inline void llk_math_eltwise_binary_init(const std::uint32_t acc_to_dest = 0) {
    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type,
        math_fidelity, binary_reuse_dest>(ckernel::DEFAULT_TENSOR_SHAPE, acc_to_dest);
}

// Version with operands
template <...>
inline void llk_math_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A, const std::uint32_t operand_B,
    const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id = get_operand_id(operand_A);
    const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
    _llk_math_eltwise_binary_init_<...>(tensor_shape, acc_to_dest);
}
```

### Pack wrappers (TRISC2)

Pack wrappers are the most complex because they must handle output address computation, format-dependent tile sizing, and synchronization with the math engine. `llk_pack()` demonstrates this:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h, lines 151-167

template <bool is_fp32_dest_acc_en, bool out_of_order_output, bool untilize>
inline void llk_pack(std::uint32_t tile_index, std::uint32_t output,
                     std::uint32_t output_tile_index = 0) {
    LLK_ASSERT((tile_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE,
                DstTileShape::Tile32x32>()), "");

    std::uint8_t output_id = get_output_id(output);

    static_assert((!(untilize && out_of_order_output)) &&
                  "untilize out of order packing is not supported!");

    std::uint32_t pack_tile_addr = get_output_tile_address<out_of_order_output, untilize>(
        output_id, output_tile_index);

    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, untilize>(tile_index, pack_tile_addr);
}
```

The address computation itself is in `get_output_tile_address()`, which handles both in-order (sequential write pointer) and out-of-order (random access) packing modes:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h, lines 116-149

template <bool out_of_order_output, bool untilize>
inline std::uint32_t get_output_tile_address(std::uint8_t output_id,
                                              std::uint32_t output_tile_index) {
    std::uint32_t pack_tile_addr;
    if constexpr (out_of_order_output) {
        pack_tile_addr = get_local_cb_interface(output_id).fifo_wr_ptr +
            (std::uint32_t)(get_local_cb_interface(output_id).fifo_page_size)
                * output_tile_index - 1;
    } else {
        // In-order: use the running write-tile pointer
        pack_tile_addr = get_local_cb_interface(output_id).fifo_wr_ptr +
            get_local_cb_interface(output_id).fifo_wr_tile_ptr - 1;
        get_local_cb_interface(output_id).fifo_wr_tile_ptr +=
            get_local_cb_interface(output_id).fifo_page_size;
    }
    return pack_tile_addr;
}
```

### Common wrappers

`llk_math_common_api.h` provides cross-cutting functions used by all math operations. The hardware configuration wrapper resolves both operands' data formats:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_common_api.h, lines 24-30

template <bool is_fp32_dest_acc_en>
inline void llk_math_hw_configure(const std::uint32_t srca_operand,
                                   const std::uint32_t srcb_operand) {
    std::uint32_t srca_operand_id = get_operand_id(srca_operand);
    std::uint32_t srcb_operand_id = get_operand_id(srcb_operand);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(
        unpack_dst_format[srca_operand_id], unpack_dst_format[srcb_operand_id]);
}
```

Similarly, `llk_unpack_common_api.h` provides `llk_unpack_hw_configure()` which extracts face dimensions, tile sizes, and formats from both operands before forwarding to `_llk_unpack_hw_configure_()`.

## Data format reconfiguration

A significant portion of the wrapper code handles **runtime data format reconfiguration** -- switching the hardware between different numerical formats without a full reinit. Both unpack and pack sides provide `llk_*_reconfig_data_format()` functions:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_common_api.h, lines 66-80

template <bool is_fp32_dest_acc_en, bool to_from_int8, p_dim_stride_target dim_stride_target>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    const std::uint32_t srca_operand_id = get_operand_id(srca_new_operand);
    const std::uint32_t num_faces = get_operand_num_faces(srca_operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(srca_operand_id);
    const std::uint32_t tile_size = get_local_cb_interface(srca_operand_id).fifo_page_size;

    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, to_from_int8, dim_stride_target>(
        unpack_src_format[srca_operand_id], unpack_dst_format[srca_operand_id],
        tile_size, face_r_dim, num_faces);
}
```

The conditional variants compare old vs. new operand formats and only reconfigure when they differ, avoiding unnecessary hardware register writes:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_common_api.h, lines 100-114

template <...>
inline void llk_unpack_reconfig_data_format_srca(
    const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    if (should_reconfigure_cbs(srca_old_operand, srca_new_operand)) {
        llk_unpack_reconfig_data_format_srca<...>(srca_new_operand);
    }
}
```

## Include chain

Each wrapper file includes its corresponding tt-llk implementation header. The include graph for matmul looks like:

```
llk_math_matmul_api.h          (TT-Metal wrapper)
    ├── llk_math_common_api.h  (TT-Metal common wrapper)
    │       ├── llk_math_common.h  (tt-llk)
    │       ├── llk_operands.h     (TT-Metal)
    │       └── llk_io.h           (TT-Metal)
    └── llk_math_matmul.h      (tt-llk implementation)
            └── _llk_math_matmul_init_()
            └── _llk_math_matmul_()
```

This structure ensures that each wrapper has access to both the Metal-specific operand resolution infrastructure and the tt-llk internal functions it delegates to.

## Summary of wrapper responsibilities

| Responsibility | Where it happens |
|---|---|
| Operand ID lookup | `get_operand_id()` / `get_output_id()` in `llk_operands.h` / `llk_outputs.h` |
| Tile dimension queries | `get_operand_tile_r_dim()`, `get_operand_face_r_dim()`, `get_operand_num_faces()` |
| Tensor shape construction | `get_operand_tensor_shape()` returning `ckernel::TensorShape` |
| L1 address computation | `get_local_cb_interface().fifo_rd_ptr` (unpack) / `fifo_wr_ptr` (pack) |
| Bounds checking | `LLK_ASSERT` on `dst_index` against `get_dest_max_tiles()` |
| Profiling instrumentation | `WAYPOINT("UPAW")` / `WAYPOINT("UPAD")` markers |
| Format-conditional reconfiguration | `llk_*_reconfig_data_format*()` comparing old vs. new operand formats |
| Delegation to tt-llk | Direct call to `_llk_<operation>_()` with resolved parameters |

---

**Next:** [`llk_io_layer.md`](./llk_io_layer.md)
