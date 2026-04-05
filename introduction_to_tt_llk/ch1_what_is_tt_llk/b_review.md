# Agent B Review: Chapter 1 — What is TT-LLK? — Pass 1

1. **File:** `repository_structure.md`, line 137. **Error:** The guide states the test infrastructure includes "the SFPI compiler toolchain (in `tests/sfpi/`)". No `tests/sfpi/` directory exists in the repository. The actual test subdirectories are `helpers/`, `hw_specific/`, `python_tests/`, and `sources/`. There are only standalone files `tests/sfpi-info.sh` and `tests/sfpi-version` at the top level of `tests/`. **Fix:** Remove the claim about `tests/sfpi/` or replace it with an accurate description, e.g., "SFPI version tracking files (`sfpi-info.sh`, `sfpi-version`) at the tests root" and mention `tests/sources/` which contains the actual test source files.

2. **File:** `software_stack_position.md`, line 89. **Error:** The guide says the test environment "supports both Wormhole and Blackhole hardware platforms", omitting Quasar. The repository contains `tests/hw_specific/quasar/`, `tests/sources/quasar/`, and a dedicated `tests/run_quasar_regression.sh` script, demonstrating that Quasar is a supported test platform. A reader would incorrectly conclude that LLK tests cannot target Quasar hardware. **Fix:** Change to "supports Wormhole, Blackhole, and Quasar hardware platforms" (or at minimum note that Quasar support exists).

# Agent B Review: Chapter 1 — What is TT-LLK? — Pass 2

1. **File:** `repository_structure.md`, line 89. **Error:** The `llk_math_reduce.h` table entry describes "Reduction operations (max, average) via FPU". The actual `PoolType` enum in `tt_llk_wormhole_b0/llk_lib/llk_defs.h` defines four reduction types: `SUM`, `AVG`, `MAX`, and `MIN`. Omitting SUM and MIN from what reads as an exhaustive parenthetical list is materially misleading — a reader would not know that sum-reductions (critical for softmax, layer normalization, loss computation) are supported. **Fix:** Change to "Reduction operations (sum, average, max, min) via FPU".

# Agent B Review: Chapter 1 — What is TT-LLK? — Pass 3

1. **File:** `repository_structure.md`, lines 80-98 (llk_lib tables). **Error:** The Math headers table lists only six headers and states SFPU operations are unary (`llk_math_eltwise_unary_sfpu.h`). The actual `tt_llk_wormhole_b0/llk_lib/` directory contains `llk_math_eltwise_binary_sfpu.h`, `llk_math_eltwise_ternary_sfpu.h`, and `llk_math_welfords_sfpu.h` (plus their corresponding `*_params.h` files), as well as `llk_math_transpose_dest.h`. These represent fundamentally different operation categories — binary SFPU (element-wise SFPU ops on two inputs), ternary SFPU (three inputs), and Welford's online variance algorithm. A kernel developer consulting this table would incorrectly conclude that all SFPU operations are unary only and that no variance-computation primitive exists. The Pack and Unpack tables similarly omit `llk_pack_rows.h` and `llk_unpack_AB_reduce.h`. **Fix:** Add at minimum `llk_math_eltwise_binary_sfpu.h`, `llk_math_eltwise_ternary_sfpu.h`, and `llk_math_welfords_sfpu.h` to the Math headers table, and `llk_pack_rows.h` and `llk_unpack_AB_reduce.h` to their respective tables. Alternatively, add a note that the tables show representative headers and are not exhaustive.

# Agent B Review: Chapter 1 — What is TT-LLK? — Pass 4

**No feedback — chapter approved.**
