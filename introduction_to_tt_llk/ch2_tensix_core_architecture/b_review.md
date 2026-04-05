# Agent B Review: Chapter 2 — Tensix Core Architecture — Pass 1

1. **Thread numbering contradiction between table and diagram in `riscv_processors.md`.**
   The table on lines 6-12 correctly lists Thread IDs from the codebase (`ckernel_defs.h`): BRISC=0, TRISC0(Unpack)=1, TRISC1(Math)=2, TRISC2(Pack)=3. However, the diagram on lines 22-24 labels them "Thread 0 (UNPACK)", "Thread 1 (MATH)", "Thread 2 (PACK)". This directly contradicts the table. A developer looking up `UnpackThreadId` in the codebase would find the value 1, not 0. The same incorrect T0/T1/T2 labeling carries into `tensix_engine.md` (line 13 table column "Thread" shows T0, T1, T2). Either renumber the diagram/table to use the actual codebase Thread IDs (1, 2, 3), or explicitly state that "T0/T1/T2" is a chapter-local shorthand distinct from the codebase `ThreadId` enum values.

2. **SFPU described as "32-lane SIMD engine" in `tensix_engine.md` (line 59) without source support.**
   The source-of-truth documents (`l1/intro.md`, `l2/top_level_overview.md`) do not specify the SFPU lane count, and the codebase headers in `tt_llk_wormhole_b0/common/inc/` do not contain a constant confirming 32 lanes. If this number is architecture-specific (e.g., different between Wormhole and Blackhole), stating it as a universal fact could mislead a developer targeting a different chip. Either cite the source for 32 lanes or qualify it with the architecture (e.g., "32 lanes on Wormhole B0").

# Agent B Review: Chapter 2 — Tensix Core Architecture — Pass 2

1. **Wrong file name `llk_math_eltwise_matmul.h` in `riscv_processors.md` (line 45).**
   The text says TRISC1's key headers include `llk_math_eltwise_matmul.h` for matrix multiplication. This file does not exist in the codebase. The actual file is `llk_math_matmul.h` (present under `tt_llk_wormhole_b0/llk_lib/`, `tt_llk_blackhole/llk_lib/`, and `tt_llk_quasar/llk_lib/`). A developer searching for the cited file name would find nothing. Rename to `llk_math_matmul.h`.

# Agent B Review: Chapter 2 — Tensix Core Architecture — Pass 3

1. **Wrong header files cited for TTI macros in `riscv_processors.md` (line 62).**
   The text states that `TTI_SETC16(...)`, `TTI_STALLWAIT(...)`, and similar macros are "defined in the `ckernel_instr_params.h` and `ckernel.h` headers." In the codebase, these macros are actually `#define`d in `ckernel_ops.h` (e.g., `tt_llk_wormhole_b0/common/inc/ckernel_ops.h` lines 488 and 812, and equivalently in the Blackhole variant). `ckernel_instr_params.h` defines the parameter constants (like `p_stall::STALL_UNPACK`) used as arguments to these macros, and `ckernel.h` merely calls them -- neither file contains the macro definitions. A developer trying to trace how Tensix ISA instructions are encoded would look in the wrong files. Change to `ckernel_ops.h`.

# Agent B Review: Chapter 2 — Tensix Core Architecture — Pass 4

**No feedback — chapter approved.**
