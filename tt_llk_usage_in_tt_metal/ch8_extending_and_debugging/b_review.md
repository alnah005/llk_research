# Agent B Review: Chapter 8 — Extending and Debugging — Pass 1

1. **compilation_flow_end_to_end.md lines 239-241: Summary diagram shows `-Os` for unpack and pack TRISCs, but all three TRISCs use `-O3`.** The source (`build.cpp` line 314-319) sets the default opt level to `Os`, then overrides to `O3` for `HalProcessorClassType::COMPUTE`. All three TRISCs (TRISC0/1/2) belong to the COMPUTE processor class (confirmed in `wh_hal_tensix.cpp` lines 91-128 where they are grouped under the `// COMPUTE` section of `processor_classes`). Therefore `chlkc_unpack.cpp` and `chlkc_pack.cpp` are compiled with `-O3`, not `-Os`. The `-Os` default only applies to data movement processors (BRISC/NCRISC). The diagram should show `-O3` for all three lines. The prose at line 86-87 ("compute TRISC build states use `-O3`... while data movement processors default to `-Os`") is technically correct but the diagram contradicts it.

2. **debugging_llk_issues.md lines 93-94 and compilation_flow_end_to_end.md line 248 (debugging checklist): Wrong environment variable name.** The chapter references `TT_METAL_LOG_KERNELS_COMPILATION_COMMANDS` but the actual env var in `rtoptions.cpp` line 625 is `TT_METAL_LOG_KERNELS_COMPILE_COMMANDS` (no "ATION" suffix). Using the documented name would silently do nothing, leaving the developer without the compilation command output they need.

3. **adding_a_new_op.md line 50: "The `iterations` parameter is always 8 for a full 16x16 face (8 rows of 2 datums each)" is numerically wrong.** A 16x16 face contains 256 elements. The SFPU processes 2 rows of 16 elements (32 datums) per iteration via SIMD. So 8 iterations x 32 datums = 256 datums per face. The parenthetical "8 rows of 2 datums each" implies only 16 datums total, which is off by 16x. A correct description would be something like "8 iterations, each processing 2 rows of 16 elements (32 datums)".

# Agent B Review: Chapter 8 — Extending and Debugging — Pass 2

No feedback — chapter approved.
