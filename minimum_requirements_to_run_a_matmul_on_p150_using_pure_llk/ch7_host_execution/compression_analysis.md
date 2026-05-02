# Chapter 7 Compression Analysis

## CRUCIAL Findings

1. **`device_setup()` function reproduced three times across files**
   - **File 2** (`02_step_by_step_execution.md`), lines 526-546: Full `device_setup()` code listing with inline comments and a 5-item explanatory list (items 1-5).
   - **File 3** (`03_what_tt_metal_replaces.md`), lines 123-144: Identical full `device_setup()` code listing with inline comments, plus a near-identical 5-item explanatory list (items 1-5, same wording).
   - **Cross-chapter**: Ch3 `02_boot_sequence_and_firmware.md`, lines 223-274, contains the authoritative coverage of `device_setup()` with per-step code snippets and explanations.
   - **Recommendation**: File 3 should remove its `device_setup()` code block and explanatory list entirely, replacing them with a one-sentence summary and cross-reference to File 2 Section 7.2.10 (or Ch3 Section 3.2). File 2 should keep its listing as the Ch7-authoritative version but add a cross-reference note that Ch3 covers the firmware perspective.
   - **Estimated savings**: ~40 lines (20 in File 2's duplicate explanation, 20 in File 3's full duplication).

2. **`clear_trisc_soft_reset()` code block reproduced three times**
   - **File 2** (`02_step_by_step_execution.md`), lines 559-567: Full `clear_trisc_soft_reset()` function with read-modify-write and readback-verify loop.
   - **Cross-chapter**: Ch3 `02_boot_sequence_and_firmware.md`, lines 280-293: Identical function listing and explanation.
   - **Cross-chapter**: Ch3 also includes `set_triscs_soft_reset()` (lines 297-307), which File 2 does not duplicate -- good.
   - **Recommendation**: File 2 should summarize in prose ("BRISC clears bits 12-14 via a read-modify-write with readback verify") and cross-reference Ch3 for the full code.
   - **Estimated savings**: ~10 lines.

3. **BRISC command loop reproduced across File 2 and Ch3**
   - **File 2** (`02_step_by_step_execution.md`), lines 365-391: BRISC `main()` command loop with `switch` statement, `invalidate_data_cache()`, `ckernel::wait()`.
   - **Cross-chapter**: Ch3 `02_boot_sequence_and_firmware.md`, lines 146-195: Nearly identical BRISC `main()` command loop listing.
   - Both include the `BriscCommandState` enum table (File 2 lines 311-319; Ch3 lines 214-221).
   - Both include the `commit_store()` function (File 2 lines 453-461; Ch3 lines 200-210).
   - **Recommendation**: File 2's value-add is the host-side perspective (the `commit_brisc_command` Python code and the handshake timing diagram). The BRISC-side C++ should be summarized with a cross-reference to Ch3 rather than fully reproduced. Keep the handshake diagram (lines 396-434) as it is unique to File 2.
   - **Estimated savings**: ~30 lines (remove the full BRISC `main()` loop listing and `commit_store` from File 2, replace with summary + cross-ref).

4. **Mailbox addresses and layout table appears four times**
   - **File 2** (`02_step_by_step_execution.md`), lines 64-76: Mailbox layout table with addresses.
   - **File 2** (`02_step_by_step_execution.md`), lines 293-307: Mailbox layout in the BRISC command protocol deep dive (different format, same data).
   - **File 3** (`03_what_tt_metal_replaces.md`), lines 78-87: L1 address layout including mailbox addresses.
   - **Cross-chapter**: Ch3 `02_boot_sequence_and_firmware.md`, lines 128-141: Mailbox address definitions from BRISC code.
   - **Recommendation**: Keep the table in File 2 Section 7.2.2 (lines 64-76) as the canonical Ch7 mailbox reference. Remove the second mailbox listing in File 2 lines 293-307 and replace with "See Section 7.2.2 for the complete mailbox layout." File 3's L1 layout (lines 78-87) is sufficiently different in purpose (showing the full address map) to keep.
   - **Estimated savings**: ~15 lines.

5. **`copy_runtimes_from_L1()` code block reproduced three times**
   - **File 2** (`02_step_by_step_execution.md`), lines 135-139: `copy_runtimes_from_L1()` function code.
   - **Cross-chapter**: Ch3 `02_boot_sequence_and_firmware.md`, lines 337-353: Same function with more context.
   - **Cross-chapter**: Ch5 `02_build_h_generation_and_constexpr.md`, lines 238-243: Same function.
   - **Cross-chapter**: Ch6 `03_linker_scripts_and_memory_regions.md`, lines 282-287: Same snippet.
   - **Recommendation**: File 2 can keep a brief inline code block since it is part of the step-by-step walkthrough, but add a cross-reference to Ch3 Section 3.2 as the authoritative source.
   - **Estimated savings**: 0 lines (keep as-is, it is short and contextual).

## MINOR Findings

1. **Architecture table duplicated between File 1 and File 2**
   - File 1 lines 181-185: Table of architecture -> boot mode, RISC cores, CPU flags, kernel components.
   - File 2 lines 47-53: Partial restatement of same information (boot mode, compiler target, arch define, data format enum for Blackhole).
   - **Recommendation**: File 2's version is specialized to the P150 context and reads as a recap, not a full duplication. Acceptable as-is, but could be shortened to one sentence with a cross-reference to File 1 Section 7.1.3.
   - **Estimated savings**: ~5 lines.

2. **ARC GO_BUSY description appears in File 1 and File 2**
   - File 1 lines 189-231: Full ARC message protocol with code, table, and explanation.
   - File 2 lines 82-88: Brief recap of GO_BUSY purpose and effect.
   - **Recommendation**: File 2's treatment is already minimal (7 lines) with a cross-reference. This is acceptable.
   - **Estimated savings**: 0 lines.

3. **TRISC `main()` code listing near-duplicated**
   - File 2 lines 578-601: TRISC `main()` code from `trisc.cpp`.
   - Ch3 `02_boot_sequence_and_firmware.md`, lines 310-418: Much more detailed coverage of the same TRISC `main()` code.
   - **Recommendation**: File 2's version is trimmed and contextual for the execution walkthrough. Acceptable, but could be shortened further by removing steps 2-4 (copy_runtimes, zero regfile, reset_cfg) which are Ch3 material, and keeping only `run_kernel()` + mailbox write.
   - **Estimated savings**: ~8 lines.

4. **Soft-reset bit positions table duplicated**
   - File 1 lines 308-315: Hardware bit positions across architectures (BRISC=11, TRISC0=12, etc.).
   - File 2 lines 225-228: Restates that bits 11-14 are set/cleared.
   - Ch3 lines 283-284: `TRISC_SOFT_RESET_MASK = 0x7000` (bits 12-14).
   - **Recommendation**: File 1 has the authoritative table. File 2's inline mention is contextual and brief. No action needed.
   - **Estimated savings**: 0 lines.

5. **`setup_arch()` match/case block in File 1**
   - File 1 lines 152-175: Full `setup_arch()` match/case code block.
   - Ch1 `01_p150_chip_identity.md`, lines 73-78: Same code snippet (smaller).
   - **Recommendation**: File 1 provides the comprehensive version with all three architectures. Ch1 shows only Blackhole. No redundancy within Ch7, and the cross-chapter overlap is minimal. No action needed.
   - **Estimated savings**: 0 lines.

6. **File 2 Section 7.2.14 (Complete Execution Timeline) partially redundant with Section 7.2.1 overview table**
   - File 2 lines 26-39: 10-step overview table.
   - File 2 lines 737-778: Complete execution timeline (ASCII diagram) that covers the same 10 steps plus multi-variant behavior.
   - **Recommendation**: The timeline adds substantial value (showing temporal ordering, BRISC persistence, multi-variant behavior). The overview table adds navigational value. Both serve distinct purposes. No action needed.
   - **Estimated savings**: 0 lines.

7. **`write_pipeline_operands_to_l1()` code reproduced in File 2 and File 3**
   - File 2 lines 157-167: Address allocation code for writing operands.
   - File 3 lines 60-72: Nearly identical code block.
   - **Recommendation**: File 3 should replace its code block with a cross-reference to File 2 Section 7.2.6.
   - **Estimated savings**: ~12 lines.

## Cross-Chapter Duplication

1. **`device_setup()` function and explanation**
   - Duplicated in: File 2 (7.2.10), File 3 (7.3.5), Ch3 (3.2)
   - Authoritative version: **Ch3 `02_boot_sequence_and_firmware.md`** (firmware perspective, includes `#ifdef` variants)
   - Ch7 secondary authority: **File 2 Section 7.2.10** (host-triggered execution context)
   - File 3 version: Should be removed and cross-referenced.

2. **BRISC command loop and `BriscCommandState` enum**
   - Duplicated in: File 2 (7.2.9), Ch3 (3.2)
   - Authoritative version: **Ch3 `02_boot_sequence_and_firmware.md`** (firmware code walkthrough)
   - Ch7 value-add: File 2's handshake timing diagram and host-side `commit_brisc_command` Python code are unique. The BRISC C++ loop listing is redundant.

3. **`clear_trisc_soft_reset()` and `set_triscs_soft_reset()`**
   - Duplicated in: File 2 (7.2.10), Ch3 (3.2)
   - Authoritative version: **Ch3 `02_boot_sequence_and_firmware.md`**

4. **`commit_store()` function**
   - Duplicated in: File 2 (7.2.9), Ch3 (3.2)
   - Authoritative version: **Ch3 `02_boot_sequence_and_firmware.md`**

5. **`copy_runtimes_from_L1()` function**
   - Duplicated in: File 2 (7.2.5), Ch3 (3.2), Ch5 (5.2.5), Ch6 (6.3.4)
   - Authoritative version: **Ch5 `02_build_h_generation_and_constexpr.md`** (covers the struct layout and serialization round-trip)

6. **TRISC `main()` boot sequence**
   - Duplicated in: File 2 (7.2.11), Ch3 (3.2)
   - Authoritative version: **Ch3 `02_boot_sequence_and_firmware.md`** (detailed step-by-step firmware walkthrough)

7. **Architecture detection (`get_chip_architecture()`) code**
   - Duplicated in: File 1 (7.1.3), Ch1 (1.1)
   - Authoritative version: **File 1 Section 7.1.3** (includes both detection paths and full code). Ch1 provides a shorter overview. Acceptable cross-chapter overlap since both chapters have different angles.

8. **L1 memory map / address layout**
   - Duplicated in: File 2 (7.2.6, lines 192-197), File 3 (7.3.3, lines 78-87), Ch6 (6.3.2, lines 82-96)
   - Authoritative version: **Ch6 `03_linker_scripts_and_memory_regions.md`** (derived from actual linker scripts)
   - File 3's version adds the data region above `0x21000`, which is unique context. File 2's version is a minimal single-tile example. Both are acceptable but could cross-reference Ch6.

## Summary

- **Total line count**: 1524 lines (440 + 784 + 300)
- **File 2 share**: 784 lines (51.4% of chapter)
- **Estimated compressible lines**: ~120 lines (approximately 8% of the chapter)
  - CRUCIAL: ~95 lines (device_setup x2, clear_trisc_soft_reset, BRISC loop duplication, double mailbox layout)
  - MINOR: ~25 lines (write_pipeline_operands, TRISC main trimming)
- **Target line count after compression**: ~1400 lines

### Recommendations (priority order)

1. **File 3 Section 7.3.5**: Remove the full `device_setup()` code block and its 5-item explanation list. Replace with a 2-3 sentence summary and cross-reference to File 2 Section 7.2.10 and Ch3 Section 3.2. Saves ~25 lines.

2. **File 2 Section 7.2.9**: Remove the BRISC-side C++ command loop listing (lines 365-391) and `commit_store()` listing (lines 453-461). Replace with a prose summary and cross-reference to Ch3. Keep the host-side Python code, the handshake timing diagram, and the protocol analysis, which are unique to Ch7. Saves ~30 lines.

3. **File 2 Section 7.2.10**: Remove the `clear_trisc_soft_reset()` code block (lines 559-567). Replace with a one-sentence description and cross-reference to Ch3. Saves ~10 lines.

4. **File 2 Section 7.2.9**: Remove the second mailbox layout listing (lines 293-307). Add a brief note referencing Section 7.2.2 for the mailbox layout. Saves ~15 lines.

5. **File 3 Section 7.3.3**: Remove the `write_pipeline_operands_to_l1()` code block (lines 60-72). Replace with cross-reference to File 2 Section 7.2.6. Saves ~12 lines.

6. **File 2 Section 7.2.11**: Trim the TRISC `main()` listing to show only `run_kernel()` + mailbox write + tensix_sync. Remove the copy_runtimes/zero_regfile/reset_cfg steps and cross-reference Ch3. Saves ~8 lines.

### File 2 balance assessment

File 2 at 784 lines is justified given that it is the central execution walkthrough chapter. After applying the above recommendations, it would shrink to approximately 720 lines (47% of chapter), which is a healthier balance. The unique content in File 2 -- the host-side Python code, the handshake timing diagram, the polling logic, and the complete execution timeline -- cannot be found elsewhere and should be preserved in full.
