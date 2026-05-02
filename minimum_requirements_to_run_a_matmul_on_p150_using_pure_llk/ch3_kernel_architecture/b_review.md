# Agent B Review: Chapter 3 — Pass 1

1. **File:** `03_inter_thread_synchronization.md`, **Lines:** 396-412
   **Error:** The `_llk_pack_dest_section_done_()` code snippet shows the **Wormhole** implementation (3-argument `TT_ZEROACC` with a `CLR_HALF_32B`/`CLR_HALF` ternary), not the Blackhole implementation. The actual Blackhole source (`tt_llk_blackhole/llk_lib/llk_pack_common.h:48`) uses a 5-argument call: `TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2)`, where `is_fp32_dest_acc_en` is a separate parameter rather than being folded into the clear mode selection.
   **Fix:** Replace the `SyncHalf` branch in the code block with the Blackhole 5-argument form. Remove the `constexpr std::uint32_t CLEAR_MODE` ternary and show the direct `TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2)` call.

2. **File:** `02_boot_sequence_and_firmware.md`, **Line:** 17
   **Error:** The text states that `_start()` "is defined identically in both `brisc.cpp` and `trisc.cpp`". This is false. The `trisc.cpp` version carries an additional `no_profile_instrument_function` attribute and includes an `#ifdef COVERAGE` block calling `gcov_dump()` between `main()` and the infinite loop. The `brisc.cpp` version has neither.
   **Fix:** Change "defined identically" to something like "follows the same general structure" and note the `trisc.cpp`-specific differences (extra attribute and coverage support).

# Agent B Review: Chapter 3 — Pass 2

1. **File:** `01_single_source_three_kernels.md`, **Line:** 13
   **Type:** Factual error
   **Error:** The text states that the `-DCOMPILE_FOR_TRISC=N` flag "in turn sets the corresponding `LLK_TRISC_*` macro." This is false. In the actual build infrastructure (`tests/python_tests/helpers/test_config.py`, lines 991-1007), `-DCOMPILE_FOR_TRISC=N` and `-DLLK_TRISC_{UNPACK|MATH|PACK}` are two independent `-D` flags passed separately on the same compiler command line. `COMPILE_FOR_TRISC` is used in `ckernel.h` only for conditional inclusion of `lltt.h` and the `load_replay_buf` template; it never defines or triggers the `LLK_TRISC_*` macros.
   **Fix:** Replace "which in turn sets the corresponding `LLK_TRISC_*` macro" with something like "alongside the corresponding `-DLLK_TRISC_UNPACK`, `-DLLK_TRISC_MATH`, or `-DLLK_TRISC_PACK` flag. The `LLK_TRISC_*` define controls the `#ifdef` block selection, while `COMPILE_FOR_TRISC` enables TRISC-specific features in `ckernel.h`."

No other factual errors, critical coherence issues, or critical structural gaps found in this pass. The Pass 1 fixes (ZEROACC argument mismatch and `_start()` identity claim) have been correctly applied.

# Agent B Review: Chapter 3 — Pass 3

1. **File:** `02_boot_sequence_and_firmware.md`, **Lines:** 393-398; also `03_inter_thread_synchronization.md`, **Lines:** 500-505
   **Type:** Factual error (appears in two locations)
   **Error:** Both sections quote the `store_blocking()` inline assembly as using `andi %[raw], %[raw], 0` as the data-dependency instruction. The actual source code (`tt_llk_blackhole/common/inc/ckernel.h`, line 207) uses `and x0, x0, %[raw]`. These are different RISC-V instructions: `andi %[raw], %[raw], 0` is an AND-immediate that zeros the `raw` register, while `and x0, x0, %[raw]` is a register AND that writes to the hardwired zero register `x0`, creating a read dependency on `raw` without modifying it. The functional intent (pipeline stall via data dependency) is the same, but the actual instruction and its side effects differ.
   **Fix:** In both locations, replace `"andi %[raw], %[raw], 0\n\t"` with `"and x0, x0, %[raw]"` to match the source. Update the surrounding prose accordingly -- the dependency is created by reading `raw` into a computation targeting `x0`, not by zeroing `raw`.

No other factual errors, critical coherence issues, or critical structural gaps found in this pass. All prior fixes from Pass 1 and Pass 2 have been correctly applied. The semaphore definitions, boot sequence, ZEROACC parameters, unpack sync flow, math-pack protocol, STALLWAIT semantics, and compilation model all verified correctly against the tt-llk source.

# Agent B Review: Chapter 3 — Pass 4

1. **File:** `02_boot_sequence_and_firmware.md`, **Line 400**; also `03_inter_thread_synchronization.md`, **Line 508**
   **Type:** Factual error (residual from Pass 3 -- partial fix)
   **Error:** The Pass 3 fix corrected the inline assembly code blocks to show `and x0, x0, %[raw]`, but the accompanying prose in both locations still refers to "the `andi` instruction creates a data dependency." The actual instruction is `and` (R-type, register-register), not `andi` (I-type, register-immediate). These are distinct RISC-V instructions.
   **Fix:** In both files, change `` the `andi` instruction creates a data dependency `` to `` the `and` instruction creates a data dependency ``.

2. **File:** `03_inter_thread_synchronization.md`, **Lines 435-444**
   **Type:** Factual error in code snippet
   **Error:** The `select_packer_dest_registers()` code snippet shows `if constexpr (Dst == DstSync::SyncHalf)` as the condition. The actual source (`tt_llk_blackhole/common/inc/cpack_common.h`, lines 658-671) uses the inverse structure: `if constexpr (Dst == DstSync::SyncFull) { TTI_WRCFG(...); } else { TT_WRCFG(...); }`. The chapter's version omits the `SyncFull` branch entirely and inverts the conditional guard, making the code snippet not match the source.
   **Fix:** Replace the snippet with the actual two-branch structure from the source, or at minimum change the guard to `else` (the `SyncHalf` path) and add a comment noting the `SyncFull` branch is omitted for brevity.

No other factual errors, critical coherence issues, or critical structural gaps found. All other prior fixes from Passes 1-3 have been correctly applied. Verified against tt-llk source: semaphore definitions, boot sequence, device_setup order, ZEROACC parameters, store_blocking assembly (code blocks), unpack sync flow, math-pack protocol, STALLWAIT semantics, compilation model, compiler name, mailbox addresses, and TRISC boot mode all match.

# Agent B Review: Chapter 3 — Pass 5

1. **File:** `01_single_source_three_kernels.md`, **Lines 138-146**
   **Type:** Wrong architecture variant
   **Error:** The `_llk_pack_dest_init_()` template showed the Wormhole signature (`template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>`) with `_llk_init_packer_dest_offset_registers_<Dst, untilize>(face_r_dim, narrow_tile)`. The Blackhole source (`tt_llk_blackhole/llk_lib/llk_pack_common.h:77-78`) has only two template parameters and calls `_llk_init_packer_dest_offset_registers_<Dst>(face_r_dim, narrow_tile)` without `untilize`.
   **Status:** Fixed by Agent A. Now correctly shows `template <DstSync Dst, bool is_fp32_dest_acc_en>` and `_llk_init_packer_dest_offset_registers_<Dst>(face_r_dim, narrow_tile)`.

All Pass 4 fixes verified:
- `select_packer_dest_registers` (file 03, lines ~431-449): Correctly shows `SyncFull` first with `TTI_WRCFG`/`DEST_OFFSET_LO`, then `else` with `TT_WRCFG`/`get_packer_dest_offset_index()`. Matches `cpack_common.h:658-671`.
- `store_blocking` assembly (files 02 and 03): Both show `and x0, x0, %[raw]`. Matches `ckernel.h:207`.
- Prose references (files 02 and 03): Both say "`and` instruction". No residual `andi`.

**APPROVED** -- All factual issues from Passes 1-5 have been resolved. No new issues found.
