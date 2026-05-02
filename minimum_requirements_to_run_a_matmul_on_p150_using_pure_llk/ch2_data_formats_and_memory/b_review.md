# Agent B Review: Chapter 2 — Pass 1

1. **File:** `02_data_format_pipeline.md`, line 25
   **Error:** The chapter states: "A-formats (Float16, Bfp8, Bfp4, Bfp2, Lf8, Fp8_e4m3) use a 5-bit exponent." Fp8_e4m3 has a 4-bit exponent (E4M3), not 5-bit. Furthermore, `IS_A_FORMAT()` in `ckernel_defs.h` (line 191-204) does not include Fp8_e4m3; it only covers Lf8, Float16, Bfp8, Bfp4, and Bfp2. While the hardware does treat Fp8_e4m3 in the same unpacker conversion family as A-formats (unpacks to Float16, confirmed in `cunpack_common.h`), listing it under "use a 5-bit exponent" is factually wrong and would mislead someone checking `IS_A_FORMAT()` or reasoning about exponent widths.
   **Fix:** Remove Fp8_e4m3 from the parenthetical A-format list and note it separately, e.g.: "Fp8_e4m3 (E4M3, 4-bit exponent) follows the A-format unpacker conversion path (unpacks to Float16) but is not classified as an A-format by `IS_A_FORMAT()` and is distinguished via the `Unp_LF8_4b_exp` config bit."

2. **File:** `03_l1_memory_layout_and_addressing.md`, lines 5-9, 326, 358
   **Error:** The chapter states L1 size is 1.5 MB (0x180000), citing `tensix_types.h`, and uses 0x180000 as the "End of L1" throughout (the memory map diagram, the summary table, and the available-space calculation of ~1.37 MB). However, `is_valid_L1_address()` in `llk_memory_checks.h` computes `L1_REGION_END = L1_ADDRESS(MEM_L1_BASE + MEM_L1_SIZE)` using `MEM_L1_SIZE` from `dev_mem_map.h`, which is `1464 * 1024 = 0x16E000` (1,499,136 bytes). A reader who allocates tile buffers up to 0x180000 based on this chapter will hit address validation failures at runtime. The actual usable L1 ceiling for tile data is 0x16E000, giving ~1.30 MB from 0x21000, not ~1.37 MB.
   **Fix:** Replace 0x180000 / 1.5 MB with 0x16E000 / ~1.43 MB (from `MEM_L1_SIZE` in `dev_mem_map.h`), or at minimum note the discrepancy: `tensix_types.h` defines `L1_SIZE = 0x180000` but the address validation boundary (`L1_REGION_END`) is derived from `MEM_L1_SIZE = 0x16E000` in `dev_mem_map.h`. The summary table, memory diagram, and available-space calculation must all be updated accordingly.

3. **File:** `02_data_format_pipeline.md`, line 11
   **Error:** The Bfp8 encoding is described as "5-bit shared exp, 8-bit mantissa." The shared exponent in all BFP formats is stored as a full 8-bit byte (one byte per 16 elements). The "5" in "A-format" refers to the exponent width of the per-element format family, not the width of the shared exponent. A reader implementing a custom packer could incorrectly store only 5 bits for the shared exponent, producing corrupt tiles. The Bfp8_b row correctly says "8-bit shared exp" but only because the B-format exponent width happens to coincide with the byte size; both A and B BFP variants use 8-bit (1-byte) shared exponents.
   **Fix:** Change the Bfp8 encoding column to something like "8-bit shared exp + 8-bit mantissa (A-format family)" and Bfp8_b to "8-bit shared exp + 8-bit mantissa (B-format family)" to make clear the shared exponent is always 8 bits. The A/B distinction is in how the exponent is interpreted, not its storage size.

# Agent B Review: Chapter 2 — Pass 2

**No feedback — chapter approved.**

All three issues from Pass 1 have been resolved in the current text:
- Fp8_e4m3 is now correctly separated from the A-format list with its own note (02_data_format_pipeline.md, line 25).
- The L1 size distinction between physical (0x180000) and firmware-validated (0x16E000) is now clearly stated, with the memory diagram and summary table using the correct validated limit (03_l1_memory_layout_and_addressing.md, lines 12-17, 334, 368).
- BFP shared exponent storage is now explicitly clarified as always 8-bit for both A and B families (02_data_format_pipeline.md, line 30).

Verified all key claims against tt-llk source: tile geometry constants, face layout, matmul init signatures and assertions, MathFidelity enum values, MVMUL inner loop counts, address modifier configurations, operand-to-register mapping, L1_ADDRESS transformation, Operand class, StimuliConfig address computation, format tile sizes, DstTileSizeLog2 values, dest register capacities, DstSync semaphore initialization, and unpacker format conversion validation functions. All match the source code.
