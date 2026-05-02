# Data Formats, the Format Conversion Pipeline, and Math Fidelity

## Supported Data Formats

Every tile stored in L1 has an associated **data format** that specifies how each element is encoded. The `DataFormat` enum in `tensix_types.h` defines the hardware-level encodings. The Python test infrastructure maintains a parallel enum in `format_config.py` with explicit hardware integer values for Blackhole:

| Format | Enum Value | Encoding | Bytes/Element | Notes |
|:-------|:-----------|:---------|:--------------|:------|
| Float32 | 0 | E8M23 | 4 | Full IEEE 754 single precision |
| Float16 | 1 | E5M10 | 2 | IEEE half precision ("A-format" exponent) |
| Bfp8 | 2 | 8-bit shared exp (A-family), 8-bit mantissa | ~1.06 | Block FP8 with "A" exponent |
| Tf32 | 4 | E8M10 | 3 (in L1) | TensorFloat-32; 19-bit precision |
| Float16_b | 5 | E8M7 | 2 | Brain float / BF16 ("B-format" exponent) |
| Bfp8_b | 6 | 8-bit shared exp, 8-bit mantissa | ~1.06 | Block FP8 with "B" exponent |
| Bfp4_b | 7 | 8-bit shared exp, 4-bit mantissa | ~0.56 | Block FP4 with "B" exponent |
| Int32 | 8 | 32-bit signed integer | 4 | |
| UInt16 | 9 | 16-bit unsigned integer | 2 | |
| Int8 | 14 | 8-bit signed integer | 1 | |
| UInt32 | 24 | 32-bit unsigned integer | 4 | |
| Fp8_e4m3 | 26 | E4M3 | 1 | FP8 per OCP spec (Blackhole-specific enum value) |
| UInt8 | 30 | 8-bit unsigned integer | 1 | |

The critical distinction between "A-format" and "B-format" families is the exponent width:

- **A-formats** (Float16, Bfp8, Bfp4, Bfp2, Lf8) use a 5-bit exponent. They all convert to Float16 in the Source registers. Note: Fp8_e4m3 has a 4-bit exponent (E4M3) and is **not** included in `IS_A_FORMAT()`, but it follows the A-format unpacker path and converts to Float16 in the Source registers.
- **B-formats** (Float16_b, Bfp8_b, Bfp4_b, Bfp2_b, Float32, Tf32) use an 8-bit exponent. They convert to Float16_b or Tf32 in the Source registers.

This distinction is enforced by the hardware: you cannot cross-convert from a B-format input to an A-format register output, or vice versa. The helper functions `IS_A_FORMAT()` and `IS_BFP_FORMAT()` in `ckernel_defs.h` test for these families.

Note on BFP shared exponents: for all BFP formats (both A-family and B-family), the shared exponents are stored as **8-bit (1-byte) values** in L1. The "A" vs "B" distinction refers to the exponent encoding family (5-bit vs 8-bit exponent width in the register format they convert to), not the storage width of the shared exponent bytes in L1. Every BFP tile stores 64 exponent bytes (one 8-bit exponent per 16 elements).

A hardware detail: the unpacker and packer configuration registers use only the **bottom 4 bits** of the format value. The comment in `ckernel_defs.h` explains:

```cpp
// The reason we keep only the bottom 4 bits is because the HW only has 4 bits
// to represent the dataformat.
constexpr std::uint32_t DATA_FORMAT_BIT_COUNT = 4;
constexpr std::uint32_t DATA_FORMAT_CONFIG_MASK = (1 << DATA_FORMAT_BIT_COUNT) - 1;
```

Higher bits serve other purposes (like distinguishing Fp8_e4m3 from other 8-bit formats via a separate config bit, `Unp_LF8_4b_exp`).

## Tile Sizes Per Data Format

Different data formats produce different tile sizes in L1 memory. The `format_tile_sizes` dictionary in `llk_params.py` gives byte counts for a standard 32x32 tile (1024 elements):

| Format | Tile Size (bytes) | Calculation |
|:-------|:------------------|:------------|
| Float32 / Int32 / UInt32 | 4096 | $1024 \times 4$ |
| Float16 / Float16_b / Int16 / UInt16 | 2048 | $1024 \times 2$ |
| Tf32 | 3072 | $1024 \times 3$ |
| Bfp8_b | 1088 | $1024 \times 1$ (data) + $64$ (shared exponents) |
| Bfp4_b | 576 | $512$ (data) + $64$ (shared exponents) |
| Int8 / UInt8 / Fp8_e4m3 | 1024 | $1024 \times 1$ |

For Block Floating Point (BFP) formats, a **shared exponent** is stored for every 16 elements. A 32x32 tile has 1024 elements, so $1024 / 16 = 64$ exponent bytes, plus the per-element mantissa bytes. The hardware requires a minimum of 16 exponent bytes even for smaller tiles.

The firmware-side function `GET_L1_HEADERLESS_TILE_SIZE` in `ckernel_defs.h` returns tile size in **16-byte units** (the hardware's L1 addressing granularity):

```cpp
constexpr static std::uint32_t GET_L1_HEADERLESS_TILE_SIZE(std::uint32_t format)
{
    switch (masked_data_format(format))
    {
        case (to_underlying(DataFormat::Float32)):   return (4096 >> 4);    // 256
        case (to_underlying(DataFormat::Float16)):
        case (to_underlying(DataFormat::Float16_b)): return (2048 >> 4);    // 128
        case (to_underlying(DataFormat::Bfp8)):
        case (to_underlying(DataFormat::Bfp8_b)):    return ((1024 >> 4) + (64 >> 4)); // 68
        case (to_underlying(DataFormat::Bfp4)):
        case (to_underlying(DataFormat::Bfp4_b)):    return ((512 >> 4) + (64 >> 4));  // 36
        case (to_underlying(DataFormat::Int8)):
        case (to_underlying(DataFormat::Fp8_e4m3)):  return (1024 >> 4);   // 64
        // ...
    }
}
```

These values are used directly by the unpacker and packer to stride through multi-tile buffers in L1. For example, Float16 tiles return 128, meaning $128 \times 16 = 2048$ bytes per tile.

## The Format Conversion Pipeline

Data does not stay in a single format as it flows through the Tensix Engine. It passes through a multi-stage conversion pipeline, with format transformations performed by hardware circuits called **gaskets** at each boundary:

```
L1 (source format)
    |
    v  [Unpacker gasket conversion]
Source Register (up to 19-bit precision)
    |
    v  [FPU computation -- MVMUL]
Destination Register (16-bit or 32-bit)
    |
    v  [Packer gasket conversion]
L1 (destination format)
```

Each stage:

1. **L1 Source Format** (`unpack_A_src`, `unpack_B_src`): The format of the tile as stored in L1. This is what the unpacker reads -- whatever format the host wrote the data in.

2. **Unpacker Gasket Conversion**: Each unpacker (Unpacker 0 for SrcA, Unpacker 1 for SrcB) contains a gasket circuit that converts from the L1 format to the source register format. The gasket handles decompression of BFP shared exponents, conversion between exponent families, and widening or narrowing of mantissa fields.

3. **Source Register Format** (`unpack_A_dst`, `unpack_B_dst`): The internal representation in Source A and Source B registers. The registers have a maximum precision of **19 bits** (TF32: 1 sign + 8 exponent + 10 mantissa). Without FP32 destination accumulation, they hold 16-bit data (Float16 or Float16_b).

4. **FPU Computation**: The FPU reads from Source A and Source B and writes results to the Destination register. On Blackhole, the ALU format is inferred from the source register data formats -- the math thread does not need to program explicit ALU format registers.

5. **Destination Register Format**: Configurable as 16-bit (default) or 32-bit FP32 (when `is_fp32_dest_acc_en = true`). See "Destination Register Accumulator Modes" below.

6. **Packer Gasket Conversion**: The packer reads from the Destination register and converts to the output L1 format. This may involve truncation, rounding, or recompression into a block floating point format.

7. **L1 Destination Format** (`pack_dst`): The format of the result tile written back to L1.

### Supported Unpacker Format Conversions

Not all conversions are supported by the hardware gasket. The valid source-to-register format conversions include:

- **Float32 in L1** can unpack to: TF32 (when FP32 dest acc enabled), Float16_b, Float16
- **Float16_b in L1** can unpack to: TF32 (when FP32 dest acc enabled), Float16_b
- **Float16 in L1** can unpack to: Float16
- **Bfp8_b in L1** can unpack to: TF32 (when FP32 dest acc enabled), Float16_b
- **Bfp4_b in L1** can unpack to: Float16_b, Bfp8_b
- **Int8 in L1** can unpack to: TF32 (when FP32 dest acc enabled), Float16_b, Int8
- **Fp8_e4m3 in L1** can unpack to: Float16

Key constraints:

- Float16_b (bfloat16, 8-bit exponent) **cannot** be converted to Float16 (IEEE half, 5-bit exponent) or vice versa. The unpacker has no cross-exponent-width conversion path.
- Int32 **cannot** be unpacked to Source registers -- it can only go directly to the Destination register via `unpack_to_dest` mode.
- TF32 as a register output is only available when `is_fp32_dest_acc_en` is true.

These conversion rules are implemented as validation functions in `cunpack_common.h`: `is_unpacker_format_conversion_supported_fp32_acc()` and `is_unpacker_format_conversion_supported_dest()`.

## The FormatConfig Struct

The test infrastructure wraps all format choices into a single `FormatConfig` dataclass (defined in `format_config.py`). It has 11 fields, each mapping to a specific stage of the pipeline:

| Field | Pipeline Stage | Description |
|:------|:---------------|:------------|
| `unpack_A_src` | L1 input for Unpacker 0 | Format of in1/inB tiles in L1 (loaded to SrcA) |
| `unpack_A_dst` | SrcA register output | Format after unpacker gasket conversion |
| `unpack_B_src` | L1 input for Unpacker 1 | Format of in0/inA tiles in L1 (loaded to SrcB) |
| `unpack_B_dst` | SrcB register output | Format after unpacker gasket conversion |
| `unpack_S_src` | L1 input for S-path | Source format for optional S-operand path |
| `unpack_S_dst` | S-path register output | Register format for S-operand |
| `math` | FPU/math functions | Format used for `_llk_math_*` function configuration |
| `pack_src` | Packer input | Format the packer reads from the Destination register |
| `pack_dst` | L1 output | Format of result tiles written to L1 |
| `pack_S_src` | Packer S-path input | Source format for S-path packing |
| `pack_S_dst` | Packer S-path output | Destination format for S-path packing |

The C++ counterpart (emitted into the generated `build.h` header) mirrors these fields:

```cpp
struct FormatConfig
{
    std::uint32_t unpack_A_src;
    std::uint32_t unpack_B_src;
    std::uint32_t unpack_S_src;
    std::uint32_t unpack_A_dst;
    std::uint32_t unpack_B_dst;
    std::uint32_t unpack_S_dst;
    std::uint32_t math;
    std::uint32_t pack_src;
    std::uint32_t pack_dst;
    std::uint32_t pack_S_src;
    std::uint32_t pack_S_dst;
};
```

For a simple matmul with Float16_b inputs and Float16_b output, a typical `FormatConfig` would set:
- `unpack_A_src = unpack_B_src = Float16_b` (L1 input)
- `unpack_A_dst = unpack_B_dst = Float16_b` (register format)
- `math = Float16_b`
- `pack_src = Float16_b`, `pack_dst = Float16_b` (L1 output)

When `same_src_format=True` (the default), `unpack_B_src` and `unpack_B_dst` mirror `unpack_A_src` and `unpack_A_dst`. The S-path fields default to matching the A-path.

## Math Fidelity

The Tensix FPU uses a limited number of mantissa bits per multiplication pass. To achieve full precision for a given data format, multiple **fidelity phases** may be required. The `MathFidelity` enum in `llk_defs.h` controls this tradeoff:

```cpp
enum class MathFidelity : std::uint8_t
{
    LoFi  = 0,   // 1 phase  -- fastest, lowest precision
    HiFi2 = 2,   // 2 phases
    HiFi3 = 3,   // 3 phases
    HiFi4 = 4    // 4 phases -- slowest, highest precision
};
```

The enum values are not sequential -- there is no `HiFi1`. The value equals the number of inner loop iterations in the MOP template. The helper function `is_high_fidelity` in `cmath_common.h` distinguishes LoFi from all higher modes:

```cpp
inline constexpr bool is_high_fidelity(const MathFidelity math_fidelity_desc)
{
    return math_fidelity_desc != MathFidelity::LoFi;
}
```

### How Fidelity Phases Work

In LoFi mode, each `MVMUL` instruction computes a single partial dot product using a limited-width mantissa multiply. In high-fidelity modes, the hardware executes multiple passes over the same source data. Each pass computes a portion of the mantissa product, and the results are accumulated in the Destination register.

The replay buffer sequence always ends with ADDR_MOD_5, which includes a `fidelity_increment`:

```cpp
constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;

addr_mod_t {
    .srca     = {.incr = 0, .clr = 1, .cr = 1},
    .srcb     = {.incr = 0, .clr = 1, .cr = 1},
    .dest     = {.incr = 0, .clr = 1, .cr = 1},
    .fidelity = {.incr = fidelity_increment, .clr = 0},
}.set(ADDR_MOD_5);
```

After each complete tile-face pass, ADDR_MOD_5 resets all register pointers and increments the fidelity phase counter. The MOP's inner loop count is set to the number of fidelity phases:

```cpp
constexpr std::uint32_t inner_loops = high_fidelity ? to_underlying(math_fidelity) : 1;
ckernel_template tmp(1, inner_loops, lltt::replay_insn(...));
```

For HiFi4, the replay sequence runs 4 times per tile, with the fidelity phase incrementing each iteration. The FPU accumulates results across phases in the Destination register, yielding maximum precision at 4x the computation time of LoFi.

**Practical guidance:**
- **LoFi** is sufficient when input precision is low (e.g., BFP8, FP8).
- **HiFi2 or HiFi3** provides a good tradeoff for Float16 inputs.
- **HiFi4** maximizes accuracy when combined with FP32 destination accumulation.

## Destination Register Accumulator Modes

The Destination register's capacity and precision are configurable. As introduced in Chapter 1, the register has a fixed physical size:

```cpp
static constexpr std::uint32_t DEST_FACE_WIDTH         = 16;
static constexpr std::uint32_t DEST_FACE_HEIGHT        = 16;
static constexpr std::uint32_t DEST_REGISTER_FULL_SIZE = 64 * DEST_FACE_HEIGHT;  // = 1024 rows
static constexpr std::uint32_t DEST_REGISTER_HALF_SIZE = DEST_REGISTER_FULL_SIZE / 2;  // = 512 rows
```

### 16-bit vs. 32-bit Accumulation

The template parameter `is_fp32_dest_acc_en` controls whether the Destination register stores elements as 16-bit or 32-bit values:

- **16-bit mode** (`is_fp32_dest_acc_en = false`): The default. Each element occupies 16 bits. This provides maximum tile capacity.

- **32-bit FP32 mode** (`is_fp32_dest_acc_en = true`): Each element occupies 32 bits (full IEEE FP32). This halves the tile capacity but enables high-precision accumulation -- essential when accumulating many partial dot products along a long K dimension.

The mode is configured through hardware register writes:

```cpp
cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
```

When FP32 accumulation is enabled, the Source registers can also carry TF32 (19-bit) data rather than the standard 16-bit Float16 or Float16_b.

### DstTileShape and Tile Capacity

The Destination register uses a tile shape enum to determine how many tiles fit:

```cpp
enum DstTileShape
{
    Tile32x32 = 0,
    Tile32x16 = 1,  // also covers Tile16x32
    Tile16x16 = 2
};

constexpr std::uint32_t DstTileSizeLog2[3] = {
    6,  // Tile32x32: 64 rows per tile (2^6)
    5,  // Tile32x16/16x32: 32 rows per tile (2^5)
    4   // Tile16x16: 16 rows per tile (2^4)
};
```

The `get_dest_max_tiles` function computes capacity based on sync mode, accumulation mode, and tile shape:

```cpp
template <DstSync SYNC_MODE, bool ACCUM_MODE, DstTileShape TILE_SHAPE>
constexpr std::uint32_t get_dest_max_tiles()
{
    constexpr std::uint32_t DEST_REGISTER_SIZE =
        SYNC_MODE == DstSync::SyncHalf
            ? (ACCUM_MODE ? DEST_REGISTER_HALF_SIZE >> 1 : DEST_REGISTER_HALF_SIZE)
            : (ACCUM_MODE ? DEST_REGISTER_FULL_SIZE >> 1 : DEST_REGISTER_FULL_SIZE);

    return DEST_REGISTER_SIZE >> DstTileSizeLog2[static_cast<int>(TILE_SHAPE)];
}
```

For 32x32 tiles, the capacities are:

| Mode | Sync Mode | Tile Capacity |
|:-----|:----------|:--------------|
| 16-bit accumulation | SyncFull | 16 tiles |
| 16-bit accumulation | SyncHalf | 8 tiles per half |
| 32-bit FP32 accumulation | SyncFull | 8 tiles |
| 32-bit FP32 accumulation | SyncHalf | 4 tiles per half |

Note: for matmul, `_llk_math_matmul_` uses `DstTileShape::Tile32x32` when setting the destination write address, regardless of actual input tile dimensions. The face iteration pattern in the replay buffer always writes to destination addresses as if the output tile were 32x32.

### DstSync: SyncHalf vs. SyncFull

Building on the double-buffering concept introduced in Chapter 1, the `DstSync` enum controls how the Math and Pack threads share the Destination register:

```cpp
enum DstSync
{
    SyncHalf = 0,
    SyncFull = 1,
};
```

**SyncFull**: The entire Destination register is available to the Math thread. The Pack thread waits until Math signals completion before reading any tiles. The MATH_PACK semaphore is initialized to 1 (only one token -- math fills all, then pack drains all):

```cpp
TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
```

**SyncHalf**: The Destination register is split into two halves. The Math thread writes to one half while the Pack thread reads from the other. The `dest_offset_id` variable tracks which half is active:

```cpp
inline void update_dest_offset_id()
{
    dest_offset_id = 1 - dest_offset_id;
}

inline std::uint32_t get_dest_buffer_base()
{
    return (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
}
```

The semaphore is initialized to 2 (both halves can be in flight simultaneously):

```cpp
TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
```

When the Math thread finishes writing tiles to the current half, it calls `_llk_math_dest_section_done_`, which signals completion and flips the active half:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    set_math_semaphores();
    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        dest_section_flip();
    }
}
```

For a minimal matmul processing a single output tile, `SyncFull` is the simpler choice. For throughput-optimized multi-tile processing, `SyncHalf` allows the pack thread to write out one tile while the math thread computes the next.

---

**Next:** [`03_l1_memory_layout_and_addressing.md`](./03_l1_memory_layout_and_addressing.md)
