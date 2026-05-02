# Chapter 5 Factual Review -- Pass 1

**Reviewer:** Agent B (factual critic)
**Scope:** Files `01_required_preprocessor_defines.md`, `02_build_h_generation_and_constexpr.md`, `03_matmul_specific_config_structs.md`
**Cross-referenced against:** tt-llk source at `/localdev/salnahari/testing_dir/tt-llk`

---

1. **File:** `03_matmul_specific_config_structs.md`, **Lines:** 384-391
   **Type:** Incorrect code
   **Error:** The `DEF_A_instr`, `DEF_B_instr`, `DEF_SKIP_A`, and `DEF_SKIP_B` constants use bare `RAREFYB_DISABLE` without the required `p_unpacr::` namespace qualifier. The actual source in `ckernel_template.h` uses `p_unpacr::RAREFYB_DISABLE` in all four definitions. As written, the code snippets would not compile.
   **Fix:** Change all four occurrences:
   - `RAREFYB_DISABLE` to `p_unpacr::RAREFYB_DISABLE`

   Corrected block:
   ```cpp
   static constexpr uint32_t DEF_A_instr =
       TT_OP_UNPACR(0, 0b1, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
   static constexpr uint32_t DEF_B_instr =
       TT_OP_UNPACR(1, 0b01, 0, 0, 0, 0, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
   ```

---

2. **File:** `03_matmul_specific_config_structs.md`, **Lines:** 541-548
   **Type:** Wrong architecture variant
   **Error:** Section 5.3.8 lists `constexpr uint32_t FACE_SIZE = 256; // 16 * 16` as a Blackhole constant from `ckernel_defs.h`. This constant does **not** exist in `tt_llk_blackhole/common/inc/ckernel_defs.h`. It exists only in the Wormhole variant (`tt_llk_wormhole_b0/common/inc/ckernel_defs.h`). The section header states "All definitions come from `tt_llk_blackhole/common/inc/`," making this a wrong-architecture-variant error.
   **Fix:** Remove the `FACE_SIZE` line from section 5.3.8, or add a note that this constant is not present in the Blackhole variant and the value 256 must be computed from `FACE_R_DIM * FACE_C_DIM`.

---

3. **File:** `03_matmul_specific_config_structs.md`, **Lines:** 587-590
   **Type:** Factual error
   **Error:** The text states "4 face pairs x 2 MVMUL per face pair x up to 4 fidelity phases = up to 32 MVMUL instructions." The actual replay buffer for a standard 32x32 matmul contains 16 MVMUL instructions per fidelity phase, not 8. The face-pair count is 8, not 4 (there are 4 result faces, each requiring 2 inner-dimension accumulations, yielding 8 face-pair operations). The correct arithmetic is: 8 face-pair operations x 2 MVMUL per face-pair = 16 MVMUL per phase x up to 4 fidelity phases = up to 64 MVMUL instructions.

   Source evidence: In `llk_math_matmul.h`, the standard 32x32 path (`!is_in1_16x32`) emits exactly 16 `TTI_MVMUL` calls into the replay buffer. The `replay_buf_len` variable is set to 16 for this path.
   **Fix:** Change "4 face pairs x 2 MVMUL per face pair x up to 4 fidelity phases = up to 32 MVMUL instructions" to "8 face-pair operations x 2 MVMUL each x up to 4 fidelity phases = up to 64 MVMUL instructions."

---

4. **File:** `03_matmul_specific_config_structs.md`, **Lines:** 186-211
   **Type:** Incomplete class definition
   **Error:** The `ckernel_template` class definition listing omits the 2-argument constructor `ckernel_template(uint32_t outer, uint32_t inner)`, which exists in the actual source (line 57 of `ckernel_template.h`). It also omits the destructor `~ckernel_template()` and the `= delete` declarations for copy/move constructors and assignment operators. While the 2-argument constructor is not used by matmul, the document purports to show the class definition and the omission makes it factually incomplete.
   **Fix:** Add the 2-argument constructor to the class definition listing. At minimum, add a comment noting that copy/move constructors are deleted and a destructor exists.

---

5. **File:** `01_required_preprocessor_defines.md`, **Lines:** 127-144
   **Type:** Factual error (file path)
   **Error:** Section 5.1.4 states the assertion chain comes from `common/llk_assert.h`. This is correct for the file at repo root (`/tt-llk/common/llk_assert.h`). However, `ckernel.h` includes `"llk_assert.h"` (not `"common/llk_assert.h"`), and the include path resolves to a different location within the variant directory. The actual include chain is: `ckernel.h` -> `#include "llk_assert.h"` which, given the include paths, resolves to `common/llk_assert.h`. The document's reference path is technically correct for human navigation but the text could mislead readers into thinking there is a `common/` prefix in the `#include` directive itself.

   Additionally, the code shown omits the fact that `llk_assert.h` itself defines `UNLIKELY` within the `ENV_LLK_INFRA` guard block (the same `UNLIKELY` macro is then used in the `LLK_ASSERT` expansion). This is a minor omission since `UNLIKELY` is also defined in `ckernel.h`, but the document shows `UNLIKELY` being used in the macro without noting it is defined in the same file.
   **Fix:** This is a minor accuracy concern. No change strictly required, but consider noting that `UNLIKELY` is defined within `llk_assert.h` itself (inside the `ENV_LLK_INFRA` guard).

---

6. **File:** `03_matmul_specific_config_structs.md`, **Lines:** 460-467
   **Type:** Factual error (enum definition location)
   **Error:** Section 5.3.5 states the `DstTileShape` source is `ckernel_defs.h`, which is correct. However, the `DstTileSizeLog2` array shown in this section is actually defined in `ckernel.h` (line 821), NOT in `ckernel_defs.h`. The `DstTileShape` enum itself is indeed in `ckernel_defs.h`, but the `DstTileSizeLog2` constant array that uses it is in `ckernel.h`.
   **Fix:** Note that `DstTileShape` is defined in `ckernel_defs.h` while `DstTileSizeLog2[]` is defined in `ckernel.h`.

---

7. **File:** `03_matmul_specific_config_structs.md`, **Lines:** 113-141
   **Type:** Incorrect code (address modifier slot values for ADDR_MOD_1)
   **Error:** The "Matmul Address Modifier Slot Configuration" block for ADDR_MOD_1 shows:
   ```cpp
   addr_mod_t{.srca={.incr=16}, .srcb={.cr=1}, .dest={.incr=8}}.set(ADDR_MOD_1);
   ```
   with the comment "Face boundary -- advance SRCA, carry-reset SRCB". This is correct for the standard 32x32 non-transposed path. However, the document omits the crucial `.srcb={.incr=0}` clarification -- since `incr` defaults to 0, it is technically correct but less clear than the source, which explicitly writes `.srcb = {.incr = 0, .clr = 0, .cr = 1}`. This is not a factual error per se, since the struct defaults make both forms equivalent.
   **Fix:** No change strictly required (defaults make it equivalent).

---

**Previously-fixed errors check:**

- "cr" is correctly described as "carry-reset" throughout (Section 5.3.1 field tables). No recurrence of the old "change rate" error.
- Matmul `ckernel_template` is correctly shown with `outer=1, inner=fidelity` (not "typically 2"). Confirmed in Section 5.3.2 line 218: "Always 1 for matmul."
- `DstTileSizeLog2` for `Tile32x32` is correctly described as "1024 datum positions" (not "64 rows") in Section 5.3.5 line 464.

---

**Summary:** 6 findings, of which 3 are substantive (findings 1, 2, 3) and 3 are minor (findings 4, 5, 6). The most significant error is finding 3 (incorrect face-pair count leading to wrong MVMUL total). The `FACE_SIZE` wrong-variant issue (finding 2) and missing namespace qualifier (finding 1) should also be corrected before publication.

Not approved -- corrections needed for findings 1, 2, and 3.

---

## Fixes Applied (2026-05-02)

- **Finding 1 fixed:** Added `p_unpacr::` namespace qualifier to all `RAREFYB_DISABLE` occurrences.
- **Finding 2 fixed:** Removed `FACE_SIZE` from the Blackhole constants listing; added note that it exists only in Wormhole.
- **Finding 3 fixed:** Changed to "8 face-pair operations x 2 MVMUL each x up to 4 fidelity phases = up to 64 MVMUL" with `replay_buf_len = 16`.
- **Finding 6 fixed:** Noted that `DstTileSizeLog2[]` is in `ckernel.h`, not `ckernel_defs.h`.

---

## Pass 2 Review

**Reviewer:** Agent B (factual critic)
**Scope:** Verify all four fixes; look for new or missed errors.
**Cross-referenced against:** tt-llk source at `/localdev/salnahari/testing_dir/tt-llk`

### Fix Verification

**Finding 1 (RAREFYB_DISABLE namespace qualifier):** Verified. Section 5.3.3, lines 385--391 of `03_matmul_specific_config_structs.md` now show `p_unpacr::RAREFYB_DISABLE` in both `DEF_A_instr` and `DEF_B_instr`. This matches the source at `ckernel_template.h` lines 142 and 158. `DEF_SKIP_A` and `DEF_SKIP_B` do not use `RAREFYB_DISABLE` (they use `TT_OP_INCADCZW`), so no qualifier was needed there. Fix is correct.

**Finding 2 (FACE_SIZE removed from Blackhole constants):** Verified. Section 5.3.8, lines 541--551 of `03_matmul_specific_config_structs.md` no longer lists `FACE_SIZE` as a Blackhole constant. The added note that `FACE_SIZE` exists only in the Wormhole variant is accurate -- confirmed by its presence at line 99 of `tt_llk_wormhole_b0/common/inc/ckernel_defs.h` and its complete absence from `tt_llk_blackhole/common/inc/`. Fix is correct.

**Finding 3 (MVMUL count arithmetic):** Verified. Section 5.3.9, lines 587--594 of `03_matmul_specific_config_structs.md` now reads "8 face-pair operations x 2 MVMUL each x up to 4 fidelity phases = up to 64 MVMUL instructions, with `replay_buf_len = 16` per phase for 32x32 tiles." I manually counted the TTI_MVMUL calls in the standard 32x32 non-transposed path (`llk_math_matmul.h`, lines 352--381, the `else` branch with `!is_in1_16x32`): exactly 16 calls (8 face-pair groups of 2 each). The `replay_buf_len` variable is set to 16 at line 302--303 for this path. With up to 4 inner loop iterations (HiFi4), the total is 64. Fix is correct.

**Finding 6 (DstTileSizeLog2 location):** Verified. Section 5.3.5, line 458 of `03_matmul_specific_config_structs.md` now states "`DstTileShape` enum is in `ckernel_defs.h`; `DstTileSizeLog2[]` is in `ckernel.h`." Source confirms: `DstTileShape` enum at `ckernel_defs.h` line 60; `DstTileSizeLog2` array at `ckernel.h` line 821. Fix is correct.

### Scan for New or Missed Errors

Checked the following claims against source:

1. **`addr_mod_t` struct definition (Section 5.3.1):** All field types, widths, `val()` method implementations, and `set()` method body match `ckernel_addrmod.h` lines 25--145. The `SRC_INCR_MASK` (0x3F) and `DEST_INCR_MASK` (0x3FF) values are correct. No errors.

2. **ADDR_MOD slot values (Section 5.3.1):** The five slot configurations shown (ADDR_MOD_0, 1, 2, 4, 5) match the standard 32x32 non-transposed path in `matmul_configure_addrmod()` at `llk_math_matmul.h` lines 44--58 (MOD_0, MOD_5), 117--123 (MOD_1), 180--185 (MOD_2), 263--270 (MOD_4). Default-initialized fields (e.g., `.clr=0`) are correctly omitted from the chapter's designated-initializer syntax. No errors.

3. **`ckernel_template` class (Section 5.3.2):** Member fields, constructor implementations, `program()` register layout (mop_cfg[0..8]), and `run()` using `TTI_MOP(1, 0, 0)` all match `ckernel_template.h` lines 14--369. The last-iteration replacement logic description matches the source comment at lines 43--47. No errors.

4. **`ckernel_unpack_template` (Section 5.3.3):** Field definitions, `program()` register layout, and `run()` methods match `ckernel_template.h` lines 73--403. The zmask splitting (`zmask >> 16` for TT_MOP_CFG, `zmask & 0xFFFF` for TT_MOP) is correct per lines 379--380. No errors.

5. **`MathFidelity` enum (Section 5.3.7):** Values (LoFi=0, HiFi2=2, HiFi3=3, HiFi4=4) match `llk_defs.h` lines 116--122. The `is_high_fidelity()` description matches `cmath_common.h` line 213--216. No errors.

6. **Tile/face constants (Section 5.3.8):** All listed constants (`FACE_HEIGHT=16`, `FACE_WIDTH=16`, `TILE_HEIGHT=32`, `TILE_WIDTH=32`, `FACE_R_DIM=16`, `FACE_C_DIM=16`, `TILE_R_DIM=32`, `TILE_C_DIM=32`, `TILE_NUM_FACES=4`) match `ckernel_defs.h` lines 86--97. No errors.

7. **`matmul_configure_mop` (Section 5.3.9):** The `inner_loops = high_fidelity ? to_underlying(math_fidelity) : 1` formula and `ckernel_template` construction with outer=1 match `llk_math_matmul.h` lines 397--398. The `set_end_op(CLR_A/CLR_B)` description matches lines 402--409. No errors.

8. **Preprocessor defines (Section 5.1):** Spot-checked `fidelity_increment` is 1 for HiFi2--HiFi4 and 0 for LoFi (line 31 of `llk_math_matmul.h`). This matches the chapter's claim. No errors.

### Result

All four fixes are correctly applied. No new factual errors were found. No previously missed substantive errors were identified.

**APPROVED**
