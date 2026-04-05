# Data Formats

## The `DataFormat` Enum

The `DataFormat` enum (defined in `tensix_types.h`, referenced throughout LLK headers) lists every numeric encoding that the hardware can store and convert between. Each value carries an integer code used in hardware configuration registers.

### Wormhole B0 Formats

The following formats are referenced by the WH LLK codebase (in `tt_llk_wormhole_b0/llk_lib/llk_defs.h` via `GetSfpLoadStoreInstrMod`):

| Format | Notation | Description |
|:-------|:---------|:------------|
| `Float32` | E8M23 | IEEE 754 single precision, 32 bits |
| `Float16` | E5M10 (FP16a) | IEEE 754 half precision, 16 bits |
| `Float16_b` | E8M7 (BFP16 / FP16b) | Bfloat16, 16 bits with 8-bit exponent |
| `Bfp8` | BFP8 | Block floating point, 8-bit mantissa, 5-bit shared exponent |
| `Bfp8_b` | BFP8_b | Block floating point, 8-bit mantissa, 8-bit shared exponent |
| `Bfp4` | BFP4 | Block floating point, 4-bit mantissa, 5-bit shared exponent |
| `Bfp4_b` | BFP4_b | Block floating point, 4-bit mantissa, 8-bit shared exponent |
| `Bfp2` | BFP2 | Block floating point, 2-bit mantissa, 5-bit shared exponent |
| `Bfp2_b` | BFP2_b | Block floating point, 2-bit mantissa, 8-bit shared exponent |
| `Lf8` | LF8 | Low-precision 8-bit float |
| `Int8` | INT8 | 8-bit signed integer (sign-magnitude) |
| `UInt8` | UINT8 | 8-bit unsigned integer |
| `UInt16` | UINT16 | 16-bit unsigned integer |
| `Int32` | INT32 | 32-bit signed integer |
| `UInt32` | UINT32 | 32-bit unsigned integer |
| `Tf32` | E8M10 (TF32) | TensorFloat-32, stored in 32-bit container with lower 13 mantissa bits zeroed |
| `Fp8_e4m3` | E4M3 | 8-bit float with 4-bit exponent and 3-bit mantissa |
| `MxFp8R` | E5M2 + block exp | Microscaling FP8 with round-to-nearest semantics |
| `MxFp8P` | E4M3 + block exp | Microscaling FP8 with pack semantics |

Newer architectures (Blackhole/Quasar) extend this list with additional microscaling formats (`MxFp6R`, `MxFp6P`, `MxFp4`, `MxInt8`, `MxInt4`, `MxInt2`) and packed register-only formats (`MxFp4_2x_A`, `MxFp4_2x_B`).

## Format Support by Hardware Location

Not every format is valid in every part of the pipeline. The three relevant storage locations have different width constraints:

| Location | Max Datum Width | Supported Formats |
|:---------|:----------------|:------------------|
| **L1 memory** | Any | All `DataFormat` values. L1 is byte-addressable storage with no format restrictions. |
| **Source registers** (SrcA, SrcB) | Up to 19 bits | TF32 (when FP32 dest accumulation is enabled), Float16_b (BF16), Float16 (FP16), and integer register formats derived from Int8/UInt8/UInt16. Source registers cannot hold full 32-bit values. |
| **Destination register** | Up to 32 bits | Float32, Float16_b, Float16, Int32, UInt16, Int8/UInt8 (and narrower types widened to the register container). The Destination register can be configured for either 16-bit or 32-bit element storage. |

The unpacker gasket automatically converts from the L1 format to the register format during unpack. The packer gasket converts from the register format back to the L1 format during pack. These conversions are described in [`format_conversion.md`](./format_conversion.md).

## Block Floating Point (BFP) Formats

BFP formats store data compactly by sharing a single exponent across a block of mantissas. In Tenstorrent's implementation, each block of 16 datums shares one exponent byte. This means a BFP tile includes both a mantissa section and an exponent section.

The `IS_BFP_FORMAT()` function (in `tt_llk_wormhole_b0/common/inc/ckernel_defs.h`) identifies BFP formats:

```cpp
constexpr static bool IS_BFP_FORMAT(std::uint32_t format)
{
    switch (masked_data_format(format))
    {
        case (to_underlying(DataFormat::Bfp8)):
        case (to_underlying(DataFormat::Bfp8_b)):
        case (to_underlying(DataFormat::Bfp4)):
        case (to_underlying(DataFormat::Bfp4_b)):
        case (to_underlying(DataFormat::Bfp2)):
        case (to_underlying(DataFormat::Bfp2_b)):
            return true;
        default:
            return false;
    };
}
```

The `_b` suffix variants (e.g., `Bfp8_b`) use an 8-bit shared exponent, while the non-`_b` variants use a 5-bit shared exponent. This distinction is encoded in the format's bit pattern and affects the exponent section size in L1 tile storage.

For BFP formats, the L1 tile size includes both the mantissa data and one exponent byte per 16 datums. For example, a 4-face tile in `Bfp8` stores 1024 mantissa bytes plus 64 exponent bytes (1024 / 16 = 64).

## The `InstrModLoadStore` Enum

When the SFPU needs to load data from or store data to the Destination register, it uses `SFPLOAD`/`SFPSTORE` instructions. These instructions take a modifier that specifies the data format of the operand. The `InstrModLoadStore` enum (in `tt_llk_wormhole_b0/llk_lib/llk_defs.h`) defines these modifiers:

```cpp
enum InstrModLoadStore
{
    DEFAULT       = 0,   // BFP formats, Lf8
    FP16A         = 1,   // Float16
    FP16B         = 2,   // Float16_b (bfloat16)
    FP32          = 3,   // Float32, Tf32
    INT32         = 4,   // Int32, UInt32
    INT8          = 5,   // Int8, UInt8
    LO16          = 6,   // UInt16
    HI16          = 7,
    INT32_2S_COMP = 12,
    INT8_2S_COMP  = 13,
    LO16_ONLY     = 14,
    HI16_ONLY     = 15
};
```

## `GetSfpLoadStoreInstrMod<DataFormat>()`

The `GetSfpLoadStoreInstrMod` template function maps a `DataFormat` to the correct `InstrModLoadStore` value for SFPU instructions. This is resolved at compile time:

```cpp
template <DataFormat format>
constexpr InstrModLoadStore GetSfpLoadStoreInstrMod()
{
    switch (format)
    {
        case DataFormat::Float32:   return InstrModLoadStore::FP32;
        case DataFormat::Float16:   return InstrModLoadStore::FP16A;
        case DataFormat::Float16_b: return InstrModLoadStore::FP16B;
        case DataFormat::Int8:      return InstrModLoadStore::INT8;
        case DataFormat::UInt8:     return InstrModLoadStore::INT8;
        case DataFormat::UInt16:    return InstrModLoadStore::LO16;
        case DataFormat::Int32:     return InstrModLoadStore::INT32;
        case DataFormat::UInt32:    return InstrModLoadStore::INT32;
        case DataFormat::Tf32:      return InstrModLoadStore::FP32;
        // All BFP and Lf8 formats -> DEFAULT (0)
        default:                    return InstrModLoadStore::DEFAULT;
    }
}
```

This mapping is important for SFPU kernel authors: the instruction modifier must match the data format that is currently stored in the Destination register, or the loaded values will be misinterpreted.

---

**Next:** [`format_conversion.md`](./format_conversion.md)
