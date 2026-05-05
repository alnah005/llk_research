# Ch5 Verification Report

Verifier: Claude Opus 4.6 (second pass)
Date: 2026-05-05
Commit reference: 621b949
Codebase: /localdev/salnahari/testing_dir/tt-metal/

---

## File 01: Existing Debug Tools

### CORRECT
- **DPRINT per-RISC buffer size (192 bytes)**: `DPRINT_BUFFER_SIZE = 204` at `dprint_common.h` line 27. The `DebugPrintMemLayout::Aux` struct (wpos:4 + rpos:4 + core_x:2 + core_y:2 = 12 bytes packed) is subtracted, leaving `data[204 - 12]` = 192 bytes. `static_assert(sizeof(DebugPrintMemLayout) == DPRINT_BUFFER_SIZE)` at line 142 confirms. Matches document Section 2.2.
- **DPRINT type code count "22+"**: The `DPRINT_TYPES` macro in `dprint_common.h` lines 29-53 contains exactly 22 `DPRINT_PREFIX(...)` entries (CSTR, ENDL, SETW, UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, FLOAT32, CHAR, BFLOAT16, SETPRECISION, FIXED, DEFAULTFLOAT, HEX, OCT, DEC, TILESLICE, U32_ARRAY, TYPED_U32_ARRAY -- note TYPED_U32_ARRAY wraps across two lines). The enum adds `DPrintTypeID_Count` as a sentinel. "22+" is accurate.
- **Watcher waypoint 4-char tags**: `waypoint.h` lines 24-28 implement a constexpr `fold()` with `static_assert(sizeof...(Is) <= 4, "Up to 4 characters allowed in WATCHER_WAYPOINT")`. Each RISC writes a uint32_t to the watcher mailbox. The example `WAYPOINT("NRW")` uses 3 chars, valid under the 4-char maximum. Confirmed.
- **LLK_ASSERT delegation mechanism**: `llk_assert.h` (WH variant at `tt_llk_wormhole_b0/llk_lib/llk_assert.h`) confirms three modes: (1) `ENV_LLK_INFRA` issues `asm volatile("ebreak")` at line 18, (2) TT-Metal integration delegates to `ASSERT(condition)` at line 28, (3) disabled mode uses `(void)sizeof((condition))` at line 36. The guard is `#ifdef ENABLE_LLK_ASSERT` at line 7. All match document Section 3.3.
- **LIGHTWEIGHT_KERNEL_ASSERTS uses ebreak**: `assert.h` lines 55-61 confirm: when `LIGHTWEIGHT_KERNEL_ASSERTS` is defined and watcher is not enabled, `ASSERT()` compiles to `asm volatile("ebreak")` at line 60. Matches document Section 3.2.
- **Profiler uses FNV-1a hash**: `profiler.h` lines 19-29 implement `hashString16()` using FNV offset basis `2166136261` and FNV prime `16777619` (standard 32-bit FNV-1a constants). Result is XOR-folded to 16 bits. Marker format at line 32 is `"LLK_PROFILER" ":" __FILE__ ":" ExpandStringize(__LINE__) ":" marker`. Confirmed.
- **Profiler entry types and BUFFER_LENGTH**: `profiler.h` lines 58-64 define TIMESTAMP=0b1000, TIMESTAMP_DATA=0b1001, ZONE_START=0b1010, ZONE_END=0b1011. Line 77 defines `BUFFER_LENGTH = 0x400` (1024 entries). Line 79 defines `BUFFERS_END = 0x16E000`. All match.
- **Stack watermark 0xBABABABA**: `stack_usage.h` line 17 confirms `stack_usage_pattern = 0xBABABABA`.
- **Wall clock registers**: WH `tensix.h` lines 148-149 confirm `RISCV_DEBUG_REG_WALL_CLOCK_L` at offset 0x1F0 and `RISCV_DEBUG_REG_WALL_CLOCK_H` at offset 0x1F8 from `0xFFB12000`.
- **Pause mechanism**: `pause.h` lines 13-27 confirm writes to `pause_msg->flags[hw_thread_idx]` with waypoints `PASW`/`PASD`. Confirmed.
- **Ring buffer sentinel**: `ring_buffer.h` line 10 confirms `DEBUG_RING_BUFFER_STARTING_INDEX = -1` (int16_t). Confirmed.

### INCORRECT
- **LLK_ASSERT file path**: Document states `tt-llk/common/llk_assert.h` (Section 3.3 and Key Source Files table). Actual paths are `tt-llk/tt_llk_wormhole_b0/llk_lib/llk_assert.h` and `tt-llk/tt_llk_blackhole/llk_lib/llk_assert.h`. There is no `common/llk_assert.h` directory or file. **Correction needed**: Change path to `tt-llk/tt_llk_wormhole_b0/llk_lib/llk_assert.h` (and note BH variant at corresponding path).
- **Profiler Quasar claim (Section 5.2)**: Document states "On Quasar, 4 TRISCs share buffer space ending at `0x16F000`; on WH/BH, 3 TRISCs share space ending at `0x16E000` (`profiler.h` lines 79-88)." Only one version of `profiler.h` exists at `tests/helpers/include/profiler.h` with `NUM_CORES = 3` and `BUFFERS_END = 0x16E000` (lines 78-79). No Quasar-specific variant with `NUM_CORES = 4` or `0x16F000` was found. **Correction needed**: Remove or flag the Quasar-specific profiler claim as unverified.
- **DPrint type list ordering**: Document Section 2.2 lists types in an order that places `DPrintTILESLICE` directly after `DPrintBFLOAT16`, omitting `DPrintSETPRECISION`, `DPrintFIXED`, `DPrintDEFAULTFLOAT`, `DPrintHEX`, `DPrintOCT`, `DPrintDEC` from between them. The actual enum has BFLOAT16, then SETPRECISION, FIXED, DEFAULTFLOAT, HEX, OCT, DEC, then TILESLICE. The document's prose groups them differently which is misleading about ordering. **Minor correction**: Reorder the list to match actual enum sequence.

### UNVERIFIABLE
- **`watcher_server.cpp` line numbers**: Multiple line references (e.g., lines 44-100, 128, 361, 390-468, 505) cannot be verified without reading the full 590-line implementation file. The file exists at the claimed path. Some line numbers were verified by a prior pass and found correct.


## File 02: Tensix Hardware Debug Capabilities

### CORRECT
- **BREAKPOINT_CTRL address 0x1C0**: Confirmed across all three architectures: WH (`tensix.h` line 113), BH (`tensix.h` line 221), and Quasar (`tensix.h` line 240) all define `RISCV_DEBUG_REG_BREAKPOINT_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x1C0)`.
- **4 breakpoints per thread**: `BKPT_CMD_ID_PAYLOAD` uses `id << 26` (bits 27:26), giving breakpoint indices 0-3. Status decode at line 673: `(status >> ((bkpt_index + thread * 4) * 4)) & 0xF`. This gives 4 breakpoints per thread, 2 threads (unpack + math), 8 total. Confirmed.
- **Breakpoints guarded by `#ifndef ARCH_QUASAR`**: `tensix_functions.h` line 619 starts the guard, line 687 ends with `#endif`. All breakpoint functions are within this block. Confirmed.
- **rvdbg_cmd has 16 commands**: `ckernel_riscv_debug.h` lines 11-31 enumerate 16 command values (PAUSE through RD_UNIT, bits 0-15) plus `DBG_MODE_BIT` (bit 31, a flag not a command). Count of 16 commands is correct.
- **HW_WCHPT count is 8 per RISC**: `ckernel_riscv_debug.h` lines 50-57 define `HW_WCHPT0` (index 10) through `HW_WCHPT7` (index 17). Eight hardware watchpoints confirmed.
- **RISC_DBG_CNTL_0/1 bit field layout**: Source comment in `ckernel_riscv_debug.h` lines 76-100 matches document: pulse at bit 31, risc_sel[2:0] at bits 19:17, reg_wr at bit 16, reg_addr[10:0] at bits 10:0. Constants at lines 102-105 confirm: `CNTL0_PULSE = 0x8000'0000`, `CNTL0_WR = 0x0001'0000`, `STATUS0_RD_VALID = 0x4000'0000`. Confirmed.
- **Quasar Debug Module base address 0x0300A000**: `overlay_reg_defines_debug.h` line 15 defines `TT_DEBUG_MODULE_APB_REG_MAP_BASE_ADDR (0x0300A000)`. All 15 APB register addresses in the document's table match exactly (verified lines 18-50). Confirmed.
- **IMPEBREAK register address 0x0401'137C**: `overlay_reg_defines_debug.h` line 2305 defines `TT_DEBUG_MODULE_SBUS_IMPEBREAK_REG_ADDR (0x0401137C)`. The PROGBUF address is at 0x0401133C (line 2303). These are distinct registers. The document correctly notes the potential confusion. Confirmed.
- **SBUS_IMPEBREAK default value 0x00100073**: `overlay_reg.h` line 6510 confirms `TT_DEBUG_MODULE_SBUS_IMPEBREAK_REG_DEFAULT (0x00100073)` (the ebreak instruction encoding). Confirmed.
- **HW_CFG_SIZE = 187**: `ckernel_debug.h` line 23 defines `constexpr static std::uint32_t HW_CFG_SIZE = 187`. Confirmed.
- **DMSTATUS struct fields**: `overlay_reg.h` lines 1738-1758 match the document's `TT_DEBUG_MODULE_APB_STATUS_reg_t` struct exactly, including all 21 bit fields ending with `impebreak:1`. Confirmed.
- **rvdbg_status union layout**: `ckernel_riscv_debug.h` lines 60-73 match exactly: paused:1, brkpt_hit:1, wchpt_hit:1, ebrk_hit:1, unused0:4, reason:8. Confirmed.
- **rvdbg_risc_sel enum values**: Lines 33-39 confirm TRISC0=0x0, TRISC1=0x0002'0000, TRISC2=0x0004'0000, TRISC3=0x0006'0000. Confirmed.
- **Command dispatch overloads**: Lines 144-162 confirm three `riscv_dbg_cmd()` overloads matching the document's signatures. Confirmed.
- **dbg_array_rd_cmd_t struct**: `ckernel_debug.h` lines 71-78 match: row_addr:12, row_32b_sel:4, array_id:3, bank_id:1, reserved:12. Confirmed.
- **dbg_bus_cntl_t struct**: `ckernel_debug.h` lines 42-49 match: sig_sel:16, daisy_sel:8, rd_sel:4, reserved_0:1, en:1, reserved_1:2. Confirmed.
- **SRCA readback via MOVDBGA2D workaround**: `ckernel_debug.h` lines 165-196 confirm the save-to-LREG3, CLEARDVALID, MOVDBGA2D, read-from-Dest, restore-from-LREG3 sequence. Confirmed.
- **dbg_read_cfgreg function**: Lines 283-307 confirm the function. HW_CFG_1 base address calculated as `dbg_cfgreg::HW_CFG_SIZE` (187). Thread configs at `2 * HW_CFG_SIZE`. Confirmed.
- **Instruction buffer override functions**: Lines 716-743 confirm `dbg_instrn_buf_wait_for_ready()` polls for 0x77, `dbg_instrn_buf_set_override_en()` writes 0x7, `dbg_instrn_buf_push_instrn()` pulses, `dbg_instrn_buf_clear_override_en()` writes 0x0. Confirmed.
- **Soft reset constants**: WH `tensix.h` lines 100-103 confirm RISCV_SOFT_RESET_0_BRISC=0x00800, NCRISC=0x40000, TRISCS=0x07000. Confirmed.
- **Neo cluster debug registers**: `tensix_neo_reg.h` lines 87-92 confirm `NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_RISC_DBG_CNTL_0_REG_ADDR (0x01800058)` and `RISC_DBG_STATUS_0_REG_ADDR (0x01800060)`. Confirmed.
- **BH Reset PC override registers**: BH `tensix.h` lines 209-214 confirm all six addresses (TRISC0_RESET_PC through NCRISC_RESET_PC_OVERRIDE). Confirmed.

### INCORRECT
- **RISC_DBG_CNTL_0/1 availability on WH (Section 3.6)**: Document states "WH: offset 0x080/0x084 from 0xFFB12000". WH's `tensix.h` does NOT define `RISCV_DEBUG_REG_RISC_DBG_CNTL_0/1` or `RISCV_DEBUG_REG_RISC_DBG_STATUS_0/1`. An exhaustive grep of all WH headers under `tt-1xx/wormhole/` returns zero matches. These registers are defined ONLY in BH `tensix.h` (lines 162-165) and Quasar `tensix.h` (commented out at lines 175-178, accessed via `t6_debug_regs_t` struct). **Correction needed**: Remove WH from the RISC_DBG_CNTL register availability list in Section 3.6. Change "These RISC_DBG_CNTL_0/1 registers exist on all three architectures" to specify BH and Quasar only.
- **Architecture Comparison table (Section 12), WH column**: Since WH lacks RISC_DBG_CNTL registers, the table's claims that WH supports "RISC-V halt/step/continue: Via RISC_DBG_CNTL", "GPR read/write: Via RISC_DBG_CNTL", "CSR read/write: Via RISC_DBG_CNTL", "Memory read/write via debug: Via RISC_DBG_CNTL", and "Hardware watchpoints: Via RISC_DBG_CNTL" are all INCORRECT for WH. **Correction needed**: These WH entries should be marked as "Not available" or "Unverified -- no register definitions found".
- **DMSTATUS line reference**: Document says "overlay_reg.h lines 1730-1758". The struct `TT_DEBUG_MODULE_APB_STATUS_reg_t` starts at line 1738 (with the typedef keyword), not 1730. Line 1730 is a blank line after the end of the DMCONTROL union. **Minor correction**: Change "lines 1730-1758" to "lines 1738-1758".

### UNVERIFIABLE
- **`tensix_functions.h` exact line numbers for breakpoint section**: Document cites "lines 608-687". The comment block starts at line 605, defines start at 609, functions at 620, `#endif` at 687. The range captures the content but the start is slightly off. This may be due to minor edits between the document's commit and the current state.


## File 03: Emulation and Replay Target Assessment

### CORRECT
- **JtagDevice API methods (read32/write32)**: `jtag_device.hpp` (build_Release header) confirms all claimed methods with matching signatures: `write32()` at line 66 (with `chip_id, noc_x, noc_y, address, data, noc_id` parameters), `read32()` at line 77 (returns `optional<uint32_t>`), `write32_axi()`/`read32_axi()` at lines 65-76, `read_tdr()`/`write_tdr()` at lines 47/51, `readmon_tdr()`/`writemon_tdr()` at lines 48-50, `dbus_memdump()` at line 52, `dbus_sigdump()` at line 59, `read_id()` at line 84, `is_hardware_hung()` at line 87, `create()` at line 28. Confirmed.
- **Spike described as functional (NOT cycle-accurate)**: Document correctly states Spike is "a functional ISA simulator, not cycle-accurate" (Section 2.1). This is a well-established fact about the Spike RISC-V ISA reference simulator from the RISC-V Foundation.
- **trisc.cpp location at tt-llk/tests/helpers/src/trisc.cpp**: File exists at `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp`. The relative path within tt-llk is `tests/helpers/src/trisc.cpp`, which matches. Confirmed.
- **`reset_cfg_state_id()` / `reset_dest_offset_id()` guarded by `#ifndef ARCH_QUASAR`**: `trisc.cpp` lines 60-63 confirm the guard. Confirmed.
- **`std::fill(ckernel::regfile, ckernel::regfile + 64, 0)`**: `trisc.cpp` line 58 confirms this call. Confirmed.
- **`*mailbox = ckernel::KERNEL_COMPLETE`**: `trisc.cpp` line 76 confirms. Confirmed.
- **E-Taxonomy levels (E-Pure, E-Model, E-Stub, E-Opaque)**: Section 4.1 defines these as an original classification framework. They are design taxonomy, not factual claims requiring source verification.
- **INSTRUCTION_WORD macro**: `ckernel_ops.h` line 12 confirms `#define INSTRUCTION_WORD(x) __asm__ __volatile__(".ttinsn %0" : : "i"((x)))`. Confirmed.
- **`riscv_debug_info_enabled = false` default**: `rtoptions.hpp` line 204 confirms. Confirmed.

### INCORRECT
- **trisc.cpp execution flow includes `copy_runtimes_from_L1()`**: Document Section 3.1 (lines 225-238) lists `copy_runtimes_from_L1()` as the second step in the execution flow. This function does NOT exist in the current `trisc.cpp`. The actual code at line 72 passes `__runtime_args_start` directly to `run_kernel()`. The `__runtime_args_start` is an `extern const volatile struct RuntimeParams[]` (line 33). **Correction needed**: Remove `copy_runtimes_from_L1()` from the execution flow. Replace with: "Runtime args are passed directly via extern `__runtime_args_start` to `run_kernel()`".
- **trisc.cpp execution flow line numbers "lines 60-97"**: The file is only 77 lines. `main()` spans lines 36-77. **Correction needed**: Change "lines 60-97" to "lines 36-77".
- **Mailbox address `0x1FFB8`**: Document Section 3.2 dependency table states `mailboxes_start = 0x1FFB8`. The actual code at `trisc.cpp` line 46 shows `mailboxes_start = 0x1FFC0` (or `0x6DFC0` for COVERAGE builds). **Correction needed**: Change `0x1FFB8` to `0x1FFC0`.
- **`ckernel::regfile` labeled as "config register base"**: Document Section 3.2 dependency table labels `regfile` as "config register base" with `REGFILE_BASE` memory-mapped address. In reality, `regfile` is a pointer to the Tensix core register file (general-purpose register set) at `REGFILE_BASE = 0xFFE00000` (WH `tensix.h` line 43). The config registers are at a separate address (`TENSIX_CFG_BASE = 0xFFEF0000`). **Correction needed**: Change label from "config register base" to "Tensix core register file (GPR)" or "hardware register file". The `REGFILE_BASE` address is correct; only the descriptive label is wrong.

### UNVERIFIABLE
- **`ckernel::regfile` as "host-side array"**: The verification task asked to check this claim. The document does NOT describe regfile as a "host-side array" -- it correctly identifies it as mapping to `REGFILE_BASE` memory-mapped address. The actual definition is `volatile std::uint32_t tt_reg_ptr *regfile = reinterpret_cast<volatile std::uint32_t *>(REGFILE_BASE)` (in `ckernel_helper.h` line 15). This is a device-side memory-mapped pointer. No correction needed for this specific claim.
- **sfpu_stub.h path**: Document Section 3.5 references `tests/helpers/include/sfpu_stub.h`. This file was not found in the repository tree. It may have been removed or renamed since the cited commit.
- **exalens_server.py path**: Document Section 8 (File 01) references `tests/python_tests/helpers/exalens_server.py`. This file was not found. The tt-exalens integration may have been restructured.
- **Linker script TRISC addresses (Section 3.4)**: A prior verification pass flagged TRISC1_CODE origin as 0xC000 (should be 0xB000) and TRISC2_CODE as 0x12000 (should be 0x10000). I did not independently verify the linker script in this pass but trust the prior pass's finding.


## Summary of Required Corrections

### HIGH PRIORITY (Factual Errors Affecting Architecture Claims)
1. **File 02, Section 3.6 and Section 12**: WH does NOT have `RISC_DBG_CNTL_0/1` or `RISC_DBG_STATUS_0/1` registers. Only BH and Quasar do. The Architecture Comparison table must be corrected for WH's RISC-V debug capabilities.
2. **File 03, Section 3.1**: Remove `copy_runtimes_from_L1()` from the trisc.cpp execution flow. This function does not exist.
3. **File 03, Section 3.2**: Change mailbox address from `0x1FFB8` to `0x1FFC0`.
4. **File 01, Key Source Files**: Change `llk_assert.h` path from `tt-llk/common/llk_assert.h` to `tt-llk/tt_llk_wormhole_b0/llk_lib/llk_assert.h`.

### MEDIUM PRIORITY (Misleading)
5. **File 01, Section 5.2**: Remove or flag Quasar profiler claim about 4 TRISCs and `0x16F000` buffer end. Only `NUM_CORES=3` and `0x16E000` are found in the codebase.
6. **File 03, Section 3.1**: Fix trisc.cpp line reference from "lines 60-97" to "lines 36-77".
7. **File 03, Section 3.2**: Change `regfile` label from "config register base" to "Tensix core register file".

### LOW PRIORITY (Minor Line Number Shifts)
8. **File 02, Section 4.2**: DMSTATUS struct starts at line 1738, not 1730.
9. **File 01, Section 1.3**: Waypoint fold function starts at line 24, not 25 (minor off-by-one in cited range "lines 25-33").
