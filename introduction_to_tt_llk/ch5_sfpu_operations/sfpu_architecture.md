# SFPU Architecture

## The SFPU Within the FPU

The Scalar Floating Point Unit (SFPU) is a SIMD engine embedded inside the Tensix FPU, controlled entirely by the TRISC1 math thread. While the main FPU datapath handles matrix multiplies and element-wise operations through fixed microcode (configured via `ckernel_template`), the SFPU provides a programmable execution path where arbitrary per-element computations can be expressed in C++ using the SFPI (SFPU Programming Interface) intrinsics.

The key architectural distinction: the FPU operates on 19-bit source register data (the truncated format used by Source A and Source B), while the SFPU performs its computations at full **32-bit floating-point precision**. This makes the SFPU the correct choice for operations that require higher numeric fidelity, such as transcendental functions, activation functions, and integer arithmetic.

The SFPU is programmed using SFPI, a C++-embedded DSL defined in `sfpi.h`. SFPI provides vector types (`sfpi::vFloat`, `sfpi::vInt`, `sfpi::vUInt`) and predicated control flow (`v_if`/`v_else`/`v_endif`) that compile down to native SFPU instructions like `SFPLOAD`, `SFPSTORE`, `SFPMAD`, `SFPSETCC`, and others.

## Load/Store Architecture

The SFPU follows a **load/store architecture**. Data does not flow directly from L1 or source registers into the SFPU. Instead, the programming model requires three steps:

1. **Load** -- Data is loaded from the Destination register into SFPU internal registers (LREGs). The SFPU reads from Dest via `sfpi::dst_reg[]` indexing, which compiles to `SFPLOAD` instructions.

2. **Compute** -- The SFPU performs arithmetic on its internal registers using SFPI operations. These map to instructions like `SFPMAD` (multiply-add), `SFPSETCC` (set condition code), `SFPMOV`, `SFPABS`, `SFPSHFT`, and others.

3. **Store** -- Results are written back from SFPU internal registers to the Destination register via assignment to `sfpi::dst_reg[]`, which compiles to `SFPSTORE` instructions.

This means that **input operands must reside in the Destination register** before an SFPU operation can process them. There are two ways to get data into Dest:

- **FPU datacopy from Source A or Source B** -- Use `DataCopyType::A2D` or `DataCopyType::B2D` to copy data from a source register into Dest via the FPU. This is the standard path when data has been unpacked into source registers.

- **Direct unpack from L1 via Unpacker 0** -- When `UnpackToDestEn` is set to `true`, Unpacker 0 can write directly into the Destination register, bypassing source registers entirely. This is controlled by the `UnpackToDestEn`/`UnpackToDestDis` constants defined in `tt_llk_wormhole_b0/llk_lib/llk_defs.h`.

## Data Format Handling: InstrModLoadStore

When the SFPU loads from or stores to the Destination register, the `SFPLOAD` and `SFPSTORE` instructions use a modifier that specifies the data format interpretation. The `InstrModLoadStore` enum (defined in `tt_llk_wormhole_b0/llk_lib/llk_defs.h`) maps data formats to instruction modifiers:

| InstrModLoadStore Value | Numeric Code | Data Formats |
|---|---|---|
| `DEFAULT` | 0 | Bfp8, Bfp4, Bfp2 (and _b variants), Lf8 |
| `FP16A` | 1 | Float16 |
| `FP16B` | 2 | Float16_b (bfloat16) |
| `FP32` | 3 | Float32, Tf32 |
| `INT32` | 4 | Int32, UInt32 |
| `INT8` | 5 | Int8, UInt8 |
| `LO16` | 6 | UInt16 |
| `HI16` | 7 | (high 16 bits) |
| `INT32_2S_COMP` | 12 | Int32 two's complement |
| `INT8_2S_COMP` | 13 | Int8 two's complement |
| `LO16_ONLY` | 14 | Low 16 bits only |
| `HI16_ONLY` | 15 | High 16 bits only |

The compile-time function `GetSfpLoadStoreInstrMod<DataFormat>()` resolves the correct modifier for a given `DataFormat` template parameter.

## VectorMode: Controlling Face Iteration

A 32x32 tile consists of four 16x16 **faces**, arranged as:

```
+----------+----------+
| Face 0   | Face 1   |
| (top-L)  | (top-R)  |
+----------+----------+
| Face 2   | Face 3   |
| (bot-L)  | (bot-R)  |
+----------+----------+
```

The `VectorMode` enum (defined in `tt_llk_wormhole_b0/llk_lib/llk_defs.h`) controls which faces an SFPU operation processes:

```cpp
enum VectorMode
{
    None      = 0,   // Single face only (no iteration)
    R         = 1,   // Row vector: Face 0 + Face 1 (top row)
    C         = 2,   // Column vector: Face 0 + Face 2 (left column)
    RC        = 4,   // All four faces
    RC_custom = 6,   // Custom iteration pattern
    Invalid   = 0xFF,
};
```

- **`VectorMode::RC`** (the default) processes all four faces, which is the common case for element-wise operations on full tiles.
- **`VectorMode::R`** processes only the top two faces (Face 0 and Face 1), used for row-vector operations.
- **`VectorMode::C`** processes only the left two faces (Face 0 and Face 2), used for column-vector operations.
- **`VectorMode::None`** processes a single face with no iteration, used for custom iteration patterns or when the caller manages face advancement externally.

## Face Iteration Pattern

The core face iteration logic lives in `_llk_math_eltwise_unary_sfpu_params_()` (defined in `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`). This function is a template that accepts any SFPU callable and dispatches it across the appropriate faces based on `VectorMode`.

The pattern for `VectorMode::RC` (all four faces):

```cpp
for (int face = 0; face < 4; face++)
{
    std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
}
```

After each face is processed, two `SETRWC` (Set Read/Write Counter) instructions advance the Destination register pointer by 8 rows each (16 rows total), which moves the pointer to the start of the next face. Since each 16x16 face occupies 16 rows in the Dest register, two increments of 8 cover one complete face.

For `VectorMode::R` (row vector, Face 0 and Face 1):

```cpp
for (int face = 0; face < 2; face++)
{
    sfpu_func(args...);
    TTI_SETRWC(..., 8, ...);  // +8
    TTI_SETRWC(..., 8, ...);  // +8 = 16 rows (next face)
}
// Skip the remaining 2 faces
TTI_SETRWC(...);  // x4 to advance past Face 2 and Face 3
TTI_SETRWC(...);
TTI_SETRWC(...);
TTI_SETRWC(...);
```

For `VectorMode::C` (column vector, Face 0 and Face 2):

```cpp
for (int face = 0; face < 2; face++)
{
    sfpu_func(args...);
    TTI_SETRWC(..., 8, ...);  // +8
    TTI_SETRWC(..., 8, ...);  // +8
    TTI_SETRWC(..., 8, ...);  // +8
    TTI_SETRWC(..., 8, ...);  // +8 = 32 rows (skip one face)
}
```

In the column case, after processing Face 0, the pointer advances by 32 rows (4 x 8), skipping over Face 1 entirely to land on Face 2. This correctly handles the non-contiguous face layout for column vectors.

## SFPU Lifecycle Functions

Each SFPU operation class follows a standard lifecycle, defined in `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`:

1. **`_llk_math_eltwise_unary_sfpu_init_<SfpuType>()`** -- One-time initialization. Calls `sfpu::_init_sfpu_config_reg()` to configure SFPU control registers, sets up address modifiers via `eltwise_unary_sfpu_configure_addrmod<sfpu_op>()`, and resets read/write counters with `math::reset_counters(p_setrwc::SET_ABD_F)`.

2. **`_llk_math_eltwise_unary_sfpu_start_<DstSync>(dst_index)`** -- Per-tile setup. Sets the Destination write address to the target tile and stalls until the SFPU is ready (`TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH)`).

3. **`_llk_math_eltwise_unary_sfpu_params_(...)`** -- Executes the actual SFPU computation across faces, as described above.

4. **`_llk_math_eltwise_unary_sfpu_done_()`** -- Per-tile cleanup. Clears the Destination register address and waits for the SFPU to finish (`TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)`).

The address modifier configuration deserves attention. Most SFPU operations use `ADDR_MOD_7` with zero-increment for all three register files (srca, srcb, dest), because the SFPU manages its own addressing through `SFPLOAD`/`SFPSTORE` rather than the FPU's automatic address advancement. Some operations like `topk_local_sort`, `typecast`, and `signbit` configure `ADDR_MOD_6` with `dest.incr = 2` or `dest.incr = 32` for specialized access patterns.

---

**Next:** [`sfpu_operations_catalog.md`](./sfpu_operations_catalog.md)
