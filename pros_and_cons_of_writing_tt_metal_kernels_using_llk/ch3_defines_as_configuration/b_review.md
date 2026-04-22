# Agent B Review: Chapter 3 — Defines as Configuration — Pass 1

1. **advantages_of_defines.md, Section 2: eltwise_binary_kernel.cpp line count is wrong.** The chapter states "This single 141-line file handles..." but `eltwise_binary_kernel.cpp` is 140 lines (`wc -l` returns 140). Off by one.

2. **advantages_of_defines.md, Section 4: layernorm factory snippet uses wrong condition variable and omits a guard.** The chapter shows `if (b.has_value())` as the condition for setting `FUSE_PRE_ADD`, but the actual code at line 358 of `layernorm_op_multi_core.cpp` uses `if (fuse_pre_add)`. Additionally, the chapter shows `compute_defines.emplace_back("FUSE_PRE_ADD", "1")` as unconditionally inside that block, but the actual code wraps it in `if (!use_welford)` (line 360). This is a meaningful omission since it misrepresents when the compute define is set.

3. **disadvantages_of_defines.md, Section 5: sharded_layernorm_factory_helpers.cpp snippet uses wrong variable names.** The chapter (lines 142-144) shows `if (fuse_pre_add)`, `if (do_gamma)`, `if (do_beta)` but the actual code at lines 485-493 uses `if (has_b)`, `if (has_gamma)`, `if (has_beta)`. These are fabricated variable names that do not appear in the source file.

4. **disadvantages_of_defines.md, Section 5: sharded_layernorm_factory_helpers.cpp RMSNORM compute define has an additional condition.** The chapter (line 152) shows `if (rms_norm)` for the compute RMSNORM define, but the actual code at line 507 is `if (rms_norm && !use_welford)`. The `!use_welford` guard is omitted, which matters because it shows the define contract is even more nuanced than the chapter claims -- the same flag produces different define sets depending on the algorithm variant.

5. **advantages_of_defines.md, Section 2: LOGADDEXP snippet line reference is off by one.** The chapter cites "lines 96-101" for the `case BinaryOpType::LOGADDEXP:` block, but the `case` statement is at line 95, not 96. Lines 96-101 are the body only. Minor, but the snippet visually includes the `case` line while the cited range excludes it.

# Agent B Review: Chapter 3 — Pass 2
No feedback — chapter approved.
