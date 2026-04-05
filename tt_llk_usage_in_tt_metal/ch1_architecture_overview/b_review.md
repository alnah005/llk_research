# Agent B Review: Chapter 1 — Architecture Overview — Pass 1

1. **File:** `key_concepts.md`, approx. line 19. **Error:** The text states the JIT build system generates "three separate `.cpp` files from the same kernel source." In reality, `genfiles.cpp` (lines 196-199, 228-231) generates **four** files: `chlkc_unpack.cpp`, `chlkc_math.cpp`, `chlkc_pack.cpp`, and `chlkc_isolate_sfpu.cpp`. A fourth prolog `isolate_sfpu_prolog = build_trisc_prolog("TRISC_ISOLATE_SFPU")` is also created. Chapter 2 of the plan explicitly references "four generated files," so this contradicts what a reader is told here. **Fix:** Change "three separate `.cpp` files" to "four separate `.cpp` files (three for the TRISC processors, plus a fourth for SFPU isolation)" and mention the `chlkc_isolate_sfpu.cpp` file.

# Agent B Review: Chapter 1 — Architecture Overview — Pass 2

No feedback — chapter approved.
