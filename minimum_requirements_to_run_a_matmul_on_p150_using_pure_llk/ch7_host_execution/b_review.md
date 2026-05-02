# Chapter 7 -- Agent B Factual Review (Pass 1)

## Verdict: CONDITIONAL APPROVAL

Chapter 7 is remarkably thorough and overwhelmingly accurate. The ARC message codes, mailbox addresses, BriscCommandState enum values, RiscCore hardware bit positions, soft-reset mask, handshake protocol, trisc.cpp flow, and the TRISC start address cache mechanism all check out against the source code. The few issues found below range from a meaningful omission in the `device_setup()` description to minor structural inaccuracies. None are logic-reversing errors.

---

### Finding 1: device_setup() omits Wormhole TRISC PC register programming
**File:** `02_step_by_step_execution.md`, Section 7.2.10, lines ~521-545
**Error:** The code snippet for `device_setup()` shows five operations (Dest CG disable, ZEROACC, SFPENCC, SFPCONFIG, semaphore init) but omits the Wormhole-specific block at the top of the actual function. In the real `boot.h` (lines 82-91), `device_setup()` begins with:
```cpp
#if defined(ARCH_WORMHOLE)
    volatile std::uint32_t tt_reg_ptr* cfg_regs = ...;
    for (unsigned int i = 0; i < std::size(TRISC_CONFIG_REGS); ++i)
    {
        cfg_regs[TRISC_CONFIG_REGS[i]] = trisc_start_addresses[i];
    }
    cfg_regs[TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en_ADDR32] = 0b111;
#endif
```
This is the critical step where BRISC programs the TRISC_RESET_PC config registers from the L1 start-address cache. Without it, Wormhole TRISCs would not boot to the correct `_start` address after `clear_trisc_soft_reset()`. Since the chapter separately discusses Wormhole start address caching (Section 7.2.8), the reader could mistakenly think writing to L1 is the end of the story.
**Fix:** Add the Wormhole TRISC PC register block to the `device_setup()` snippet, gated with the `#if defined(ARCH_WORMHOLE)` guard, and add a numbered step explaining that BRISC programs the hardware config registers from the cached L1 values. Also add a comment to the "SFPENCC" step that it is step 4 (not 3) on Wormhole because of this preceding block. Alternatively, add a note stating the snippet shows the Blackhole-only path and cross-reference 7.2.8 for the full Wormhole path.

---

### Finding 2: device_setup() omits Quasar-specific CLEARDVALID instruction
**File:** `02_step_by_step_execution.md`, Section 7.2.10, lines ~521-545; also `03_what_tt_metal_replaces.md`, Section 7.3.5, lines ~123-143
**Error:** Both files show `device_setup()` without the Quasar-specific `TTI_CLEARDVALID(0, 0, 0xf, 0xf, 0, 0)` instruction that appears in `boot.h` line 103-104 between ZEROACC and SFPENCC. They also omit the Quasar variant of SFPENCC which takes only 2 arguments (`TTI_SFPENCC(3, 10)`) instead of 4 (`TTI_SFPENCC(3, 0, 0, 10)`). The semaphore init block is also gated `#ifndef ARCH_QUASAR` in the real code but the chapter shows it unconditionally.
**Fix:** Either add the Quasar-specific operations or add a note that the snippet shows the Blackhole path only. Since the chapter already identifies Quasar as a supported architecture with its own boot mode, noting the omission prevents confusion.

---

### Finding 3: File path claim about device.py imports
**File:** `01_host_device_communication.md`, Section 7.1.1, lines ~15-16
**Error:** The text says "The test infrastructure imports these functions in `device.py`" implying the file is `tests/python_tests/device.py`. The actual file path is `tests/python_tests/helpers/device.py`. The chapter never gives the full path, but the implied location is one directory level wrong.
**Fix:** Change to "The test infrastructure imports these functions in `helpers/device.py`" or "in `device.py` (located at `tests/python_tests/helpers/device.py`)".

---

### Finding 4: load_elf import location is imprecise
**File:** `01_host_device_communication.md`, Section 7.1.1, line ~37
**Error:** The text says "Note that `load_elf` is imported separately in `test_config.py` from the same library." This is correct, but the import is at `tests/python_tests/helpers/test_config.py`, not `tests/python_tests/test_config.py`. The same path-depth issue as Finding 3.
**Fix:** Use the full relative path `helpers/test_config.py` or add the `helpers/` prefix.

---

### Finding 5: conftest.py setup_build() signature is simplified
**File:** `01_host_device_communication.md`, Section 7.1.2, line ~72; `02_step_by_step_execution.md`, Section 7.2.2, line ~45
**Error:** The chapter says "`TestConfig.setup_build(sources_path)` from `conftest.py :: pytest_configure()`". The actual call passes five arguments: `TestConfig.setup_build(Path(os.environ["LLK_HOME"]), with_coverage, detailed_artefacts, no_debug_symbols, speed_of_light)`. The simplified form is not wrong but omits parameters that affect behavior (coverage mode changes the mailbox base address, detailed artefacts changes compilation).
**Fix:** Either show the full call or add a parenthetical note "(plus coverage and debug flags)" to indicate additional arguments exist.

---

### Finding 6: brisc.cpp main() omits disable_branch_prediction()
**File:** `02_step_by_step_execution.md`, Section 7.2.7, lines ~232-240
**Error:** The BRISC main() snippet shows `counter = 0` followed by mailbox clearing, but omits the `disable_branch_prediction()` call that precedes all other operations in the actual `brisc.cpp` `main()` (line 54). This is a Tensix-specific RISC-V optimization/workaround that the chapter should at least mention.
**Fix:** Add `disable_branch_prediction();` to the snippet before the counter initialization, or add a comment noting the omission.

---

### Finding 7: profiler_barrier address not in mailbox layout table
**File:** `02_step_by_step_execution.md`, Section 7.2.9, lines ~293-305
**Error:** The mailbox layout table lists 8 entries (mailboxes_arr+0 through mailboxes_arr+7), but `profiler_barrier` is declared separately in `brisc.cpp` at address `0x16AFF4U` (not contiguous with the mailbox array). The chapter correctly shows `profiler_barrier` being reset in the START_TRISCS handler code (Section 7.2.10, lines ~505-507) but never explains that `profiler_barrier` is at a completely different address from the mailbox array. A reader might assume it is adjacent.
**Fix:** Add a brief note that `profiler_barrier` lives at `0x16AFF4` (a separate L1 region), not within the mailbox array.

---

### Finding 8: TRISC mailbox offset table lists LLK_TRISC_ISOLATE_SFPU inconsistently
**File:** `02_step_by_step_execution.md`, Section 7.2.11, lines ~603-611
**Error:** The mailbox offset table lists only three defines (LLK_TRISC_UNPACK, LLK_TRISC_MATH, LLK_TRISC_PACK). The actual `trisc.cpp` also defines `LLK_TRISC_ISOLATE_SFPU` with offset `3 * sizeof(uint32_t)` for Quasar's fourth TRISC. This is relevant since the chapter explicitly discusses Quasar support elsewhere. This is a minor omission, not an error.
**Fix:** Add a fourth row to the table: `LLK_TRISC_ISOLATE_SFPU | 3 * sizeof(uint32_t) | 0x1FFC4 (Quasar only)`.

---

### Finding 9: Mailbox three-state lifecycle table has ambiguous "reset_mailboxes()" attribution
**File:** `02_step_by_step_execution.md`, Section 7.2.12, lines ~663-670
**Error:** The table says `0xA3` is set by `reset_mailboxes()` "from Python". While `reset_mailboxes()` does exist in `device.py` and writes `0xA3`, it is only called in the TRISC boot mode path (`case BootMode.TRISC` in `test_config.py` line 1162), not in the BRISC boot mode path that the rest of the chapter describes. In BRISC mode, the mailboxes go directly from whatever their prior state is to `ckernel::RESET_VAL` (set by BRISC) and then to `0xFF` (set by TRISCs). The `0xA3` sentinel is not part of the BRISC-mode lifecycle.
**Fix:** Clarify that the three-state table applies to the combined view across boot modes. In BRISC mode, the lifecycle is two-state: `RESET_VAL` -> `0xFF`. The `0xA3` sentinel from `reset_mailboxes()` is used only in TRISC/EXALENS boot modes.

---

### Finding 10: Architecture table says Wormhole default boot mode is BRISC but omits TRISC0-2 range
**File:** `01_host_device_communication.md`, Section 7.1.3, lines ~181-185
**Error:** The architecture table says Wormhole has "BRISC + TRISC0-2" cores. Technically the `RISC Cores` column should note that TRISC3 does not exist on Wormhole (only on Quasar). The table handles this implicitly by listing "TRISC0-2" for WH/BH and "TRISC0-3 (no BRISC)" for Quasar, which is correct. No error here -- verified correct.
**Fix:** None needed.

---

### Finding 11: Minor -- HardwareController location
**File:** `01_host_device_communication.md`, Section 7.1.5, lines ~237-253
**Error:** The `HardwareController` class is described as being "in `device.py`" (implied by context), but it is actually in `tests/python_tests/helpers/hardware_controller.py`, a separate file.
**Fix:** Clarify the file location.

---

## Summary

**Verified correct:**
- ARC message codes: GO_BUSY = 0xAA52, GO_IDLE = 0xAA54 (confirmed in `device.py` lines 346-348)
- Mailbox base address: 0x1FFB8 standard, 0x6DFB8 coverage (confirmed in `brisc.cpp`, `trisc.cpp`, `llk_params.py`)
- Mailbox offsets: Unpacker +0, Math +4, Packer +8, BriscCommand0 +12, BriscCommand1 +16, BriscCounter +20 (confirmed in `llk_params.py`)
- BriscCommandState enum: IDLE=0, START=1, RESET=2, UPDATE_AND_START=3 (confirmed in `brisc.cpp` and `llk_params.py`)
- RiscCore hardware bit positions: BRISC=11, TRISC0=12, TRISC1=13, TRISC2=14 for WH/BH; shifted by -1 for Quasar (confirmed in `device.py`)
- TRISC_SOFT_RESET_MASK: 0x7000 = bits 12-14 (confirmed in `boot.h`)
- Wormhole TRISC start address cache: base 0x16DFF0 with offsets +0/+4/+8 (confirmed in `boot.h` and `test_config.py`)
- KERNEL_COMPLETE = 0xFF (confirmed in `device.py`)
- Double-buffered command protocol with monotonic counter (confirmed in `brisc.cpp` and `device.py`)
- `device_setup()` operations: ZEROACC, SFPENCC, SFPCONFIG, semaphore init (confirmed in `boot.h`)
- `trisc.cpp` main flow: copy_runtimes_from_L1 -> fill regfile -> reset_cfg_state_id -> run_kernel -> tensix_sync -> write mailbox (confirmed)
- Architecture detection: `get_chip_architecture()` with `wormhole_b0` -> `wormhole` normalization (confirmed in `chip_architecture.py`)
- `setup_arch()` match/case: all architecture flags, LLK roots, and CPU targets verified (confirmed in `test_config.py`)
- `workers_tensix_coordinates` fixture: divmod by 8 (confirmed in `conftest.py`)
- `commit_brisc_command` handshake: write to parity-selected slot, increment, poll (confirmed in `device.py`)

**Overall assessment:** The chapter is highly accurate. The substantive issues are the omission of Wormhole TRISC PC register programming from the `device_setup()` description (Finding 1) and the misleading three-state mailbox lifecycle for BRISC mode (Finding 9). All other findings are minor file-path imprecisions or acceptable snippet simplifications. With the fixes for Findings 1 and 9, this chapter meets the standard for factual correctness.

---

## Fixes Applied (2026-05-02)

- **Finding 1 fixed:** Added note that device_setup() snippet shows Blackhole-only path; on Wormhole it first programs TRISC_RESET_PC config registers from L1 cache (cross-references Section 7.2.8).
- **Finding 9 fixed:** Split mailbox lifecycle into boot-mode-specific tables. BRISC mode (P150): two-state (`RESET_VAL` → `0xFF`). TRISC/EXALENS modes: three-state (`0xA3` → `RESET_VAL` → `0xFF`).

**APPROVED**
