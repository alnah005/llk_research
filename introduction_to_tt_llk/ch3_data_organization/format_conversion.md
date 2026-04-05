# Format Conversion

## Overview

Data rarely stays in the same format throughout the compute pipeline. Tiles stored in L1 may use compact formats like `Bfp8` to save memory, but the Source registers and Destination register require specific widths. The hardware handles these conversions automatically through **gaskets** -- dedicated conversion logic in the unpacker and packer datapaths.

## Unpacker Gasket: L1 Format to Register Format

When the unpacker reads a tile from L1, the unpacker gasket converts each datum from the L1 storage format (`in_data_format` in the tile descriptor) to the register format (`out_data_format` in the unpack config). This conversion is configured during `_llk_unpack_hw_configure_` and validated by `is_unpacker_format_conversion_supported_fp32_acc()` in `tt_llk_wormhole_b0/common/inc/cunpack_common.h`.

The supported conversions depend on the destination of the unpack:

**Unpack to Source registers (SrcA / SrcB):**
- Float32 in L1 can convert to TF32 (when FP32 dest accumulation is enabled), Float16_b, or Float16.
- Float16 / Float16_b in L1 can convert to Float16 or Float16_b (including cross-conversion between the two).
- BFP formats in L1 can convert to Float16 or Float16_b.
- Int8 / UInt8 in L1 can convert to TF32 (when FP32 dest accumulation is enabled) or to integer register format.
- Int32 in L1 can convert to Int32 (passed through to Destination register via unpack-to-dest).

**Unpack to Destination register (unpack-to-dest mode):**
- Float32 in L1 can convert to Float32, Float16_b, or Float16.
- Int32 in L1 can convert to Int32.
- Other formats follow the same rules as Source register unpack for their supported output formats.

The key constraint is that **Source registers are limited to 19-bit data elements** in current architectures. This is why Float32 data in L1 must be narrowed to TF32, Float16_b, or Float16 when targeting Source registers. Full 32-bit precision is only preserved when unpacking directly to the Destination register.

The `configure_unpack_AB` function (in `tt_llk_wormhole_b0/common/inc/cunpack_common.h`) sets up both unpackers. It configures stride registers based on the output format width, sets the tile descriptor's `in_data_format` field, and programs the unpack config's `out_data_format` field. The function asserts that the requested conversion is supported before proceeding.

## Packer Gasket: Register Format to L1 Format

The packer performs the reverse conversion. It reads data from the Destination register in the register format (`in_data_format` in pack config) and converts it to the L1 storage format (`out_data_format` in pack config).

The packer configuration (in `tt_llk_wormhole_b0/common/inc/cpack_common.h`) includes a `dest_rd_ctrl` register that controls how data is read from the Destination register:

- `PCK_DEST_RD_CTRL_Read_32b_data`: set when packing from a 32-bit format or when FP32 dest accumulation is enabled.
- `PCK_DEST_RD_CTRL_Read_unsigned`: set when the output format is unsigned (e.g., `UInt8`).
- `PCK_DEST_RD_CTRL_Read_int8`: set for Int8 output when not in 32-bit mode.
- `PCK_DEST_RD_CTRL_Round_10b_mant`: set when reading FP32 dest data for Float16 output, to round the mantissa to 10 bits.

For BFP output formats, the packer additionally writes the shared exponent section. The `exp_section_size` configuration field is set to the number of faces (or 1 for partial-face BFP packing).

## FP32 Destination Accumulation

By default, the Destination register stores data in 16-bit elements (Float16 or Float16_b). When the `is_fp32_dest_acc_en` template parameter is set to `true`, the Destination register is configured to store 32-bit elements instead.

FP32 accumulation is beneficial when:
- The computation involves many accumulation steps (e.g., matrix multiplication with a large inner dimension) and intermediate precision matters.
- The input data is natively 32-bit (Float32 or Int32) and you want to avoid premature truncation.
- The output requires full 32-bit precision.

The trade-off is that 32-bit destination mode halves the number of tiles that fit in the Destination register. In 16-bit mode the register can hold up to 16 tiles; in 32-bit mode it holds up to 8.

FP32 accumulation affects both the unpacker and packer:
- The unpacker allows TF32 as a Source register output format (the 19-bit format that preserves maximum precision for FPU operations that accumulate into FP32 dest).
- The packer reads 32-bit values from the Destination register and converts them to the L1 output format.

The relevant configuration is set in `configure_unpack_AB`:

```cpp
// FP32 accumulation and SFPU to read dest as FP32
static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_FORMAT_SPEC_REG0_SrcA_ADDR32);
static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32);
```

## BFP-Specific Packing Behavior

The `IS_BFP_FORMAT()` function (see [`data_formats.md`](./data_formats.md) for definition and tile-size calculations) gates BFP-specific logic in the pack and unpack paths. In the packer, partial-face packing with BFP formats requires packing one face at a time (setting `PACKCNT` to 1) so that the exponent section aligns correctly with the mantissa data.

## Stochastic Rounding

Format conversions that reduce precision (e.g., Float32 to Float16_b) must round the result. By default, the hardware uses **round-to-nearest-even**. Stochastic rounding is an alternative that adds random perturbation before rounding, which reduces systematic bias across many operations.

The `StochRndType` enum (in `tt_llk_wormhole_b0/llk_lib/llk_defs.h`) controls where stochastic rounding is applied:

```cpp
enum struct StochRndType
{
    None = 0,    // No stochastic rounding; default round-to-nearest-even
    Fpu  = 1,    // Stochastic rounding on every FPU accumulation
    Pack = 2,    // Stochastic rounding in gasket and packer format conversion
    All  = 0xf,  // Stochastic rounding in FPU, gasket, and packer
};
```

The three rounding enable bits are configured in `configure_unpack_AB` via the ALU rounding mode register:

- `ALU_ROUNDING_MODE_Fpu_srnd_en`: enables stochastic rounding in the FPU during accumulation.
- `ALU_ROUNDING_MODE_Gasket_srnd_en`: enables stochastic rounding in the packer gasket (dest format to pack source format conversion).
- `ALU_ROUNDING_MODE_Packer_srnd_en`: enables stochastic rounding in the packer (pack source format to pack destination format conversion).

`StochRndType::Pack` enables both the gasket and packer rounding bits. `StochRndType::All` enables all three.

Stochastic rounding is particularly useful for training workloads where many small updates are accumulated. Without it, small gradient values can be systematically rounded to zero in low-precision formats.

---

**Next:** [`math_fidelity.md`](./math_fidelity.md)
