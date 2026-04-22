# Agent B Review: Chapter 2 — Compute Kernel API Abstraction — Pass 1

1. **SDPA flash decode CB count is wrong (abstraction_leaks.md, Section 2).** The chapter claims the SDPA flash decode kernel uses "23 distinct CB indices." Counting the actual `constexpr` declarations at lines 75-104 of `sdpa_flash_decode.cpp` yields 28 distinct `tt::CBIndex` values (c_0 through c_7, c_10 through c_14, c_16 through c_18, c_20 through c_31 excluding c_15 and c_19). An additional CB (c_8) is used conditionally at line 148, bringing the total to 29. The stated number should be corrected to 28 (or 29 if including the conditional index).

# Agent B Review: Chapter 2 — Pass 2

No feedback — chapter approved.

The CB count fix from Pass 1 has been correctly applied. The chapter now states "28 distinct CB indices (up to 29 with the conditional c_8)" which matches the source. All other factual claims verified against the tt-metal codebase:

- reconfig_data_format call counts: 18 in layernorm (confirmed), 32 in SDPA (confirmed), 6 in groupnorm (confirmed).
- layernorm_sharded.cpp is 462 lines (confirmed).
- tile_regs_acquire appears 11 times in layernorm_sharded.cpp (confirmed).
- All quoted code snippets and line number references match the source files.
- binary_op_init_common at line 86 of layernorm and line 155 of groupnorm (confirmed).
- SFPU_OP_CHAIN generation at lines 959-968 of unary_op_utils.cpp (confirmed; chapter says 958-968 which is off by one on the start but includes the enclosing for-loop, acceptable).
