# Math Operations

## What Math Does

The math stage is the second stage of the compute pipeline. It performs **FPU and SFPU computation** on data sitting in the Source A and Source B register files, writing results into the **Destination register**. The math stage is controlled by **TRISC1**.

## The FPU

The FPU (Floating Point Unit) operates on 16x16 faces. It contains a matrix of multiply-accumulate cells that process data from Source A and Source B, producing results in the Destination register.

The FPU supports four cell-level operations:

| Operation | Description | Instruction |
|:----------|:-----------|:------------|
| Accumulated dot product | D += B * A (matrix multiply) | `MVMUL` |
| Accumulated element-wise add | D += A + B | `ELWADD` |
| Accumulated element-wise multiply | D += A * B | `ELWMUL` |
| Element-wise add | D = A + B (non-accumulated) | `ELWADD` with clear |

The FPU processes data in chunks of `MAX_FPU_ROWS` (8 rows) at a time. For a full 16-row face, the inner loop runs twice (16 / 8 = 2). The `MVMUL` instruction computes D[8,16] = B[8,16] * A[16,16], pairing columns of A with rows of B.

## The SFPU

The SFPU (Scalar Floating Point Unit) handles complex unary operations that the FPU cannot perform natively, such as exponential, reciprocal, square root, and other transcendental functions. SFPU operations are covered in detail in [Chapter 5 -- SFPU Operations](../ch5_sfpu_operations/index.md).

## LLK Math API Headers

The math API is split across several header files in `tt_llk_wormhole_b0/llk_lib/`:

### `llk_math_eltwise_binary.h` -- Element-Wise Binary Operations

Implements element-wise add, subtract, and multiply between two tiles. The `EltwiseBinaryType` enum selects the operation:

```cpp
enum EltwiseBinaryType
{
    ELWMUL,   // element-wise multiply
    ELWDIV,   // element-wise divide
    ELWADD,   // element-wise add
    ELWSUB,   // element-wise subtract
    ELWLESS,  // element-wise less-than
};
```

Key functions:

```cpp
template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type,
          MathFidelity math_fidelity>
inline void eltwise_binary_configure_addrmod();

template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type,
          MathFidelity math_fidelity>
inline void eltwise_binary_configure_mop_standard(
    const std::uint32_t acc_to_dest, const ckernel::TensorShape &tensor_shape);

template <EltwiseBinaryType eltwise_binary_type, BroadcastType src_b_bcast_type,
          DstSync Dst, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void _llk_math_eltwise_binary_standard_(
    const ckernel::TensorShape &tensor_shape, std::uint32_t dst_index);
```

The MOP outer loop count depends on broadcast type: for `BroadcastType::COL`, the MOP processes one row of faces (`num_faces_c_dim` faces) and the runtime calls it `num_faces_r_dim` times; for other types, the MOP processes all faces in a single call.

For `ELWMUL` with high math fidelity, the MOP runs multiple fidelity phases, each incrementing the fidelity counter through the address modifier. For `ELWADD`/`ELWSUB`, fidelity has no effect since addition is exact in a single pass.

### `llk_math_eltwise_unary_datacopy.h` -- Source to Destination Copy

Copies data from Source A or Source B into the Destination register. The `DataCopyType` enum selects the direction:

```cpp
enum DataCopyType
{
    A2D,  // Source A to Destination
    B2D,  // Source B to Destination
};
```

This is the math-stage component of datacopy operations. When `unpack_to_dest` is `true` and 32-bit formats are in use, the unpack stage writes directly to the Destination register and the math stage merely coordinates via `math_unpack_to_dest_math_ready()` and `math_unpack_to_dest_tile_ready()`.

For broadcast datacopy (e.g., `BroadcastType::ROW` with 32-bit data), the implementation uses `MOVD2B` / `MOVB2D` instructions to move data between Destination and Source B registers, performing the broadcast through repeated row writes.

### `llk_math_eltwise_unary_sfpu.h` -- SFPU Unary Operations

Provides the framework for SFPU-based unary operations. The `_llk_math_eltwise_unary_sfpu_start_` function sets the destination write address and switches to the SFPU address modifier base (ADDR_MOD 4-7), then stalls until the FPU is idle:

```cpp
template <DstSync Dst>
inline void _llk_math_eltwise_unary_sfpu_start_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_unary_sfpu_done_()
{
    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}
```

The SFPU uses `ADDR_MOD_7` as its primary address modifier (with zero increments by default) to avoid conflicting with FPU address modifiers 0-3.

### `llk_math_matmul.h` -- Matrix Multiply

Implements tile-level matrix multiplication. The FPU executes `MVMUL` instructions: D[8,16] = B[8,16] * A[16,16].

The address modifier configuration (`matmul_configure_addrmod`) is complex because it must handle:
- Inner loop: Source B and Destination advance by 8 rows, Source A stays fixed (the full 16x16 face is consumed by hardware).
- Outer loop transition: Source A advances to the next face, Source B and Destination reset, fidelity increments.
- Tile shape variations: 16x32, 32x16, and 32x32 inputs each require different address modifier strides.

The matmul MOP also supports throttle levels for managing power consumption, configured via the `THROTTLE_LEVEL` template parameter.

### `llk_math_reduce.h` -- Row/Column/Scalar Reduction

Implements reductions along rows, columns, or across the entire tile. The `ReduceDim` and `PoolType` enums control the operation:

```cpp
enum ReduceDim { REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR };
enum PoolType  { SUM, AVG, MAX, MIN };
```

The math kernel (`_llk_math_reduce_`) processes faces in dimension-specific order. For `REDUCE_ROW`, it pools across face columns, then transposes the result (using `MOVD2B` / `TRNSPSRCB` / `MOVB2D` sequences to transpose in Source B). For `REDUCE_COL`, it pools across face rows without transpose. For `REDUCE_SCALAR`, it pools all faces into a single value, then transposes and re-pools.

Pool operations use `GAPOOL` (average pool) or `GMPOOL` (max pool) instructions, with high-fidelity variants running multiple phases via a `ckernel_template` MOP.

### `llk_math_transpose_dest.h` -- Transpose in Destination Register

Performs a transpose of data already resident in the Destination register.

## Common API Pattern

Math API headers follow a consistent pattern:

| Function | Purpose |
|:---------|:--------|
| `_llk_math_hw_configure_<>()` | One-time hardware configuration: set Source A/B data formats, enable FP32 accumulation and INT8 math mode. |
| `_llk_math_*_init_<>()` | Configure address modifiers and program the MOP for the specific operation. |
| `_llk_math_*_<>()` | Execute the math operation: set destination write address, wait for source data valid, run the MOP, clear counters. |
| `_llk_math_dest_section_done_<>()` | Signal completion to the packer and flip the destination buffer (in `SyncHalf` mode). |

## Address Modifier Configuration

The FPU uses **address modifiers** (`addr_mod_t`) to control how Source A, Source B, and Destination register pointers advance after each instruction. Each modifier specifies increment, clear, and carry-to-carry-register behavior for each register file:

```cpp
addr_mod_t {
    .srca = {.incr = MAX_FPU_ROWS},   // Source A advances by 8 rows
    .srcb = {.incr = MAX_FPU_ROWS},   // Source B advances by 8 rows
    .dest = {.incr = MAX_FPU_ROWS},   // Destination advances by 8 rows
}.set(ADDR_MOD_0);

addr_mod_t {
    .srca     = {.incr = 0, .clr = 1},
    .srcb     = {.incr = 0, .clr = 1},
    .dest     = {.incr = 0, .clr = 0, .cr = 1},
    .fidelity = {.incr = fidelity_increment},
}.set(ADDR_MOD_2);
```

Up to 8 address modifiers (ADDR_MOD_0 through ADDR_MOD_7) can be programmed. The `.fidelity` field controls fidelity phase cycling for high-fidelity math. The `.bias` field is used in matmul for bias accumulation.

Typical address modifier assignments:
- **ADDR_MOD_0**: Standard per-iteration increment (8 rows for Source A/B/Dest).
- **ADDR_MOD_1**: Transition between faces or tile dimensions.
- **ADDR_MOD_2 / ADDR_MOD_3**: End-of-face operations with fidelity increment or clear.
- **ADDR_MOD_5 / ADDR_MOD_6**: Matmul-specific full reset with fidelity handling.
- **ADDR_MOD_7**: SFPU default (zero increments, no conflict with FPU modifiers).

## MOP Configuration via `ckernel_template`

Every math operation programs a **Micro-Op Program (MOP)** using the `ckernel_template` class, which encapsulates a hardware instruction loop that executes on the Tensix engine without TRISC1 intervention. For the full MOP structure, constructor API, and last-iteration replacement behavior, see [synchronization.md -- The `ckernel_template` Class](./synchronization.md#the-ckernel_template-class).

As a concrete example, an element-wise add MOP for a 4-face tile with 16-row faces:

```cpp
// outerloop = 4 (faces), innerloop = 2 (16 rows / 8 rows per FPU batch)
ckernel_template tmp(4, 2, TT_OP_ELWADD(0, acc_to_dest, broadcast_type, ADDR_MOD_0, 0));
tmp.set_end_op(TT_OP_SETRWC(CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
tmp.program();
```

---

**Next:** [`pack_operations.md`](./pack_operations.md)
