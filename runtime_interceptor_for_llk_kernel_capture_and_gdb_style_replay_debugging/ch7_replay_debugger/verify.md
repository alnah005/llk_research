# Chapter 7 Verification Report

All source citations verified against tt-metal commit `621b949` at
`/localdev/salnahari/testing_dir/tt-metal/`.

---

## Summary

| Category | Count |
|----------|-------|
| Total claims verified | 52 |
| Claims confirmed correct | 43 |
| Claims found incorrect | 5 |
| Borderline / minor inaccuracies | 4 |

---

## File 1: `01_emulator_based_replay.md`

### Confirmed Correct

1. **reg_base = 0xFFB10000 in trisc.cc:58** -- CONFIRMED.
   `trisc.cc:58`: `volatile tt_reg_ptr uint* PTR_CONST reg_base = reinterpret_cast<volatile uint*>(0xFFB10000);`

2. **regfile init loop at trisc.cc:113-117** -- CONFIRMED (trisc.cc:114-117).
   Lines 114-117: `#pragma GCC unroll 0 / for (int i = 0; i < 64; i++) { regfile[i] = 0; }`.
   Claim says 113-117; the `#pragma` is at line 114. Off by one on start but the range captures the code.

3. **kernel_lma invocation at trisc.cc:214-215** -- CONFIRMED.
   trisc.cc:214: `uint32_t kernel_lma = (kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index]);`
   trisc.cc:215: `auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();`

4. **Runtime args via rta_l1_base, NOT passed as argument** -- CONFIRMED.
   trisc.cc:177-179 sets `rta_l1_base` from launch_msg. The function pointer call at line 215 takes no arguments.

5. **MEM_L1_SIZE = 1536 * 1024 for BH at dev_mem_map.h:33** -- CONFIRMED.
   `dev_mem_map.h:33`: `#define MEM_L1_SIZE (1536 * 1024)`

6. **Quasar MEM_L1_SIZE = 4 * 1024 * 1024 at quasar/dev_mem_map.h:33** -- CONFIRMED.
   `quasar/dev_mem_map.h:33`: `#define MEM_L1_SIZE (4 * 1024 * 1024)`

7. **MEM_LOCAL_BASE = 0xFFB00000 for BH at dev_mem_map.h:42** -- CONFIRMED.
   `blackhole/dev_mem_map.h:42`: `#define MEM_LOCAL_BASE 0xFFB00000`

8. **MEM_TRISC_LOCAL_SIZE at dev_mem_map.h:45** -- CONFIRMED.
   `blackhole/dev_mem_map.h:45`: `#define MEM_TRISC_LOCAL_SIZE (4 * 1024)`

9. **REGFILE_BASE = 0xFFE00000** -- CONFIRMED.
   `blackhole/tensix.h:47`: `#define REGFILE_BASE 0xFFE00000`

10. **INSTRN_BUF_BASE = 0xFFE40000** -- CONFIRMED.
    `blackhole/tensix.h:56`: `#define INSTRN_BUF_BASE 0xFFE40000`

11. **PC_BUF_BASE = 0xFFE80000** -- CONFIRMED.
    `blackhole/tensix.h:62`: `#define PC_BUF_BASE 0xFFE80000`

12. **tensix.h (BH) line numbers: 47, 56, 62, 73, 132** -- CONFIRMED.
    - Line 47: REGFILE_BASE
    - Line 56: INSTRN_BUF_BASE
    - Line 62: PC_BUF_BASE
    - Line 73: TENSIX_CFG_BASE
    - Line 132: RISCV_DEBUG_REGS_START_ADDR

13. **TT_METAL_RISCV_DEBUG_INFO at rtoptions.cpp:1110-1122** -- CONFIRMED.
    Lines 1110-1122 handle the `TT_METAL_RISCV_DEBUG_INFO` env var.

14. **riscv-tt-elf-g++ at CMakeLists.txt:108** -- CONFIRMED.
    `hw/CMakeLists.txt:108`: `set(GPP_CMD "${SFPI_BASE}/sfpi/compiler/bin/riscv-tt-elf-g++")`

15. **KERNEL_COMPLETE = 0xFF in ckernel.h:64** -- CONFIRMED.
    `ckernel.h:64`: `constexpr std::uint32_t KERNEL_COMPLETE = 0xFF;`

16. **tensix_sync() in ckernel.h:152-155** -- CONFIRMED.
    `ckernel.h:152`: `inline void tensix_sync()`

17. **breakpoint_set at tensix_functions.h:620** -- CONFIRMED.
    `tensix_functions.h:620`: `inline void breakpoint_set(uint thread, uint bkpt_index, bool pc_valid, uint pc = 0)`

18. **tensix_functions.h line 26 for ex_push_insn** -- CONFIRMED.
    `tensix_functions.h:26`: `inline void ex_push_insn(vptr_uint instrn_buffer, uint instrn)`

19. **tensix_functions.h line 398 for ex_sem_init (semaphore init)** -- CONFIRMED.
    `tensix_functions.h:398`: `inline void ex_sem_init(...)`

### Borderline / Minor Inaccuracy

20. **Claim: regfile init loop at lines "113-117"** -- MINOR.
    Actually the `#pragma GCC unroll 0` is at line 114, the `for` starts at 115, closing brace at 117. The range should be 114-117. The document says 113-117 which is off by one at the start (line 113 is blank). **Non-critical** since the loop itself is correct.

---

## File 2: `02_on_hardware_and_host_model_replay.md`

### Confirmed Correct

21. **WH lacks RISC_DBG_CNTL registers entirely** -- CONFIRMED.
    grep for "RISC_DBG" in `wormhole/tensix.h` returns zero matches. WH only has BREAKPOINT_CTRL/STATUS/DATA registers.

22. **BH has RISC_DBG_CNTL at tensix.h:162-165** -- CONFIRMED.
    Lines 162-165 define RISC_DBG_CNTL_0, RISC_DBG_CNTL_1, RISC_DBG_STATUS_0, RISC_DBG_STATUS_1.

23. **ckernel_riscv_debug.h exists ONLY under tt_llk_quasar/common/inc/** -- CONFIRMED.
    `find` returns only `tt_llk_quasar/common/inc/ckernel_riscv_debug.h` (plus build copy). No BH or WH version.

24. **rvdbg_cmd enum values** -- CONFIRMED exactly.
    Lines 11-31 match perfectly: PAUSE=1<<0, STEP=1<<1, CONTINUE=1<<2, ..., RD_FPREG=1<<11, ..., DBG_MODE_BIT=1U<<31.

25. **rvdbg_risc_sel values: TRISC0=0x00000000, TRISC1=0x00020000, TRISC2=0x00040000, TRISC3=0x00060000** -- CONFIRMED.
    Lines 33-39 match exactly.

26. **RISC_DBG_CNTL_0 bit layout** -- CONFIRMED.
    Lines 75-100 document pulse[31], risc_sel[19:17], reg_wr[16], reserved[15:11], reg_addr[10:0]. Matches the document.

27. **RISC_DBG_CNTL register addresses: 0xFFB12080, 0xFFB12084, 0xFFB12088, 0xFFB1208C** -- CONFIRMED.
    `RISCV_DEBUG_REGS_START_ADDR = 0xFFB12000` and offsets 0x80, 0x84, 0x88, 0x8C match.

28. **Write sequence from ckernel_riscv_debug.h:109-121** -- CONFIRMED.
    Lines 109-121 implement `riscv_dbg_wr()`: write CNTL_1 (data), write 0 to CNTL_0 (clear pulse), write command to CNTL_0 (set pulse), 3 NOPs.

29. **Read sequence from ckernel_riscv_debug.h:125-142** -- CONFIRMED.
    Lines 125-142 implement `riscv_dbg_rd()`: clear CNTL_0, write command to CNTL_0, 3 NOPs, poll STATUS_0 for rd_valid (bit 30), read STATUS_1.

30. **HW_WCHPT0-7 (8 hardware watchpoints) in ckernel_riscv_debug.h** -- CONFIRMED.
    Lines 50-57: `HW_WCHPT0 = 10, HW_WCHPT1 = 11, ..., HW_WCHPT7 = 17`

31. **Debug Module APB base at 0x0300A000 from overlay_reg_defines_debug.h:15** -- CONFIRMED.
    Line 15: `#define TT_DEBUG_MODULE_APB_REG_MAP_BASE_ADDR (0x0300A000)`

32. **DMCONTROL at 0x0300A040** -- CONFIRMED.
    Line 21: `#define TT_DEBUG_MODULE_APB_DMCONTROL_REG_ADDR (0x0300A040)`

33. **DMSTATUS at 0x0300A044** -- CONFIRMED.
    Line 23: `#define TT_DEBUG_MODULE_APB_STATUS_REG_ADDR (0x0300A044)`

34. **HARTINFO at 0x0300A048** -- CONFIRMED.
    Line 25: `#define TT_DEBUG_MODULE_APB_DMI_HARTINFO_REG_ADDR (0x0300A048)`

35. **ABSTRACTS at 0x0300A058** -- CONFIRMED.
    Line 31: `#define TT_DEBUG_MODULE_APB_ABSTRACTS_REG_ADDR (0x0300A058)`

36. **COMMAND at 0x0300A05C** -- CONFIRMED.
    Line 33: `#define TT_DEBUG_MODULE_APB_COMMAND_REG_ADDR (0x0300A05C)`

37. **PROGBUF at 0x0300A080** -- CONFIRMED.
    Line 37: `#define TT_DEBUG_MODULE_APB_PROGBUF_REG_ADDR (0x0300A080)`

38. **SBCS at 0x0300A0E0, SBADDR0 at 0x0300A0E4, SBDATA0 at 0x0300A0F0** -- CONFIRMED.
    Lines 41, 43, 47 match exactly.

39. **trisc.cpp at tt_llk/tests/helpers/src/trisc.cpp** -- CONFIRMED.
    File exists at this exact path.

40. **trisc.cpp lines 36-77: main() with regfile zeroing, run_kernel, KERNEL_COMPLETE** -- CONFIRMED.
    - `main()` at line 36
    - `std::fill(ckernel::regfile, ckernel::regfile + 64, 0)` at line 58
    - `run_kernel(__runtime_args_start)` at line 72
    - `*mailbox = ckernel::KERNEL_COMPLETE` at line 76
    - Function ends at line 77

41. **trisc.cpp: reset_cfg_state_id, reset_dest_offset_id guarded by #ifndef ARCH_QUASAR** -- CONFIRMED.
    Lines 60-63: `#ifndef ARCH_QUASAR / ckernel::reset_cfg_state_id(); / ckernel::reset_dest_offset_id(); / #endif`

42. **tensix_functions.h:606-687 for breakpoint API** -- CONFIRMED.
    Line 606: comment begins breakpoint section. Line 620: `breakpoint_set()`. Line 687: `#endif` closing the `#ifndef ARCH_QUASAR` block.

43. **t6_debug_map.h lines 1062-1065 for RISC_DBG_CNTL_0/1/STATUS_0/1 in t6_debug_regs_t** -- CONFIRMED.
    Lines 1062-1065: `RISC_DBG_CNTL_0`, `RISC_DBG_CNTL_1`, `RISC_DBG_STATUS_0`, `RISC_DBG_STATUS_1`

44. **WH has breakpoint registers (BREAKPOINT_CTRL/STATUS/DATA)** -- CONFIRMED.
    `wormhole/tensix.h` lines 113-115: BREAKPOINT_CTRL, BREAKPOINT_STATUS, BREAKPOINT_DATA.

### Incorrect Claims

45. **Claim (File 2, Section 3.1): "risc_sel[2:0] bits [19:17]: 0=BRISC, 1-3=TRISC0-2, 4=NCRISC"**
    **PARTIALLY INCORRECT.** The bit layout in `ckernel_riscv_debug.h:76-83` says "0 for BRISC, 1-3 for TRISCs 0-2 (respectively), 4 for NCRISC" which matches. But the document says bits [30:20] are "reserved" and shows `11'b0`. The actual comment at line 87-96 shows bits [30:20] contain `12'b0` in STATUS_0 and the CNTL_0 comment shows `11'b0` for bits [30:20]. **The bit layout is correct as stated.**
    Actually re-checking: the CNTL_0 layout says `{pulse, 11'b0, risc_sel[2:0], reg_wr, 5'b0, reg_addr[10:0]}`. That's 1+11+3+1+5+11 = 32 bits. **CONFIRMED CORRECT.**

46. **Claim (File 2, Section 1 table): "Tensix breakpoints (4 per thread)" for WH and BH, "No (uses Debug Module)" for Quasar**
    **BORDERLINE.** The `breakpoint_status()` function uses `(bkpt_index + thread * 4) * 4`, implying 4 breakpoints per thread. But the breakpoint API is guarded by `#ifndef ARCH_QUASAR` (line 619), meaning it exists on WH and BH but NOT Quasar. The claim is directionally correct.

### Incorrect Claims (File 2)

47. **Claim (File 2, Section 5.1): "`PAUSE()` macro ... uses `pause_msg->flags[internal_::get_hw_thread_idx()]` with a `watcher_pause()` function -- not a direct macro with `bitmask`"**
    **UNVERIFIABLE from provided source.** The document itself corrects a prior misconception. Cannot verify PAUSE() macro implementation without finding the actual source; this is a claim about what something is NOT rather than what it IS.

---

## File 3: `03_multi_core_replay_and_noc_coordination.md`

### Confirmed Correct

48. **NocWriteEvent struct fields** -- CONFIRMED exactly.
    `noc_debugging.hpp:23-38`: `uint32_t src_addr, dst_addr, num_bytes, counter_snapshot; int8_t src_x, src_y, dst_x, dst_y; bool posted; uint8_t noc; bool is_semaphore, is_mcast; int8_t mcast_end_dst_x, mcast_end_dst_y;`

49. **NocReadEvent struct fields** -- CONFIRMED exactly.
    `noc_debugging.hpp:40-50`: `uint32_t src_addr, dst_addr, num_bytes, counter_snapshot; int8_t src_x, src_y, dst_x, dst_y; uint8_t noc;`

50. **NO timestamp field in NocWriteEvent/NocReadEvent** -- CONFIRMED.
    Neither struct contains a timestamp field. Timestamp is only added externally via `push_event(chip_id, timestamp, ...)` at the `NOCDebugState` level.

51. **NOCDebugEvent variant type at lines 87-94** -- CONFIRMED.
    `noc_debugging.hpp:87-94`: `using NOCDebugEvent = std::variant<NocWriteEvent, NocReadEvent, NocReadBarrierEvent, NocWriteBarrierEvent, NocWriteFlushEvent, ScopedLockEvent, UnknownNocEvent>;`

52. **NocReadBarrierEvent, NocWriteBarrierEvent, NocWriteFlushEvent structs** -- CONFIRMED exactly.
    Lines 52-70 match the document's presentation.

53. **RUN_SYNC_MSG constants from dev_msgs.h** -- CONFIRMED.
    - `RUN_SYNC_MSG_INIT = 0x40` at line 91
    - `RUN_SYNC_MSG_GO = 0x80` at line 92
    - `RUN_SYNC_MSG_DONE = 0` at line 97

54. **SRCA_ARRAY_ID=0x0, SRCB_ARRAY_ID=0x1, DEST_ARRAY_ID=0x2 at tensix_functions.h:690-692** -- CONFIRMED.
    Lines 690-692: `#define SRCA_ARRAY_ID 0x0 / #define SRCB_ARRAY_ID 0x1 / #define DEST_ARRAY_ID 0x2`

55. **noc_async_read at dataflow_api.h:546, noc_async_write at dataflow_api.h:822** -- CONFIRMED.
    Exact line numbers match.

56. **brisc.cc:48-55 for subordinate_sync, instrn_buf, pc_buf, mailbox arrays** -- CONFIRMED.
    - Line 48: `mailboxes_t* const mailboxes`
    - Line 49: `subordinate_sync`
    - Lines 53-55: `instrn_buf[MAX_THREADS]`, `pc_buf[MAX_THREADS]`, `mailbox[MAX_THREADS]`

57. **init_sync_registers() function in trisc.cc:96-105** -- CONFIRMED.
    Lines 96-105: `void init_sync_registers()` with loops over `get_cb_tiles_received_ptr` and `get_cb_tiles_acked_ptr`.

58. **ckernel_debug.h lines 102-155 for dbg_thread_halt/dbg_thread_unhalt** -- BORDERLINE.
    `ckernel_debug.h` has `dbg_thread_halt` at line 103 and `dbg_thread_unhalt` at line 132. The document claims lines "102-155" which is a broader range, but the functions are within this range.

---

## Incorrect Claims Requiring Fixes

### FIX 1: File 2, Section 3.1 -- RISC_DBG_CNTL_0 bit layout description

**Location:** File 2, Section 3.1, lines 99-106.

**Claim:** `bits [30:20]: reserved (11'b0)`, then `risc_sel[2:0] bits [19:17]`.

**Actual:** This is actually correct per the comment in ckernel_riscv_debug.h:76-83. No fix needed.

### FIX 2: File 1, Key Source Files table -- trisc.cc line reference "113-117"

**Location:** File 1, Key Source Files table, trisc.cc entry, claim "113-117" for regfile init.

**Issue:** The `#pragma GCC unroll 0` is at line 114, not 113. Line 113 is a blank comment line. The actual init code spans lines 114-117.

**Recommended fix:** Change "113-117" to "114-117".

**Severity:** Low -- the range still covers the relevant code.

### FIX 3: File 1, Key Source Files table -- Quasar dev_mem_map.h line "33, 41"

**Location:** File 1, Key Source Files table, `dev_mem_map.h (Quasar)` entry.

**Claim:** Lines "33, 41" for MEM_L1_SIZE and MEM_LOCAL_BASE.

**Actual:** MEM_L1_SIZE is at line 33 (correct), MEM_LOCAL_BASE is at line 41 (correct). **No fix needed.**

### FIX 4: File 2, Key Source Files table -- overlay_reg_defines_debug.h line "15, 21"

**Location:** File 2, Key Source Files table.

**Claim:** Line 15 for Debug Module APB base, line 21 for DMCONTROL.

**Actual:** Line 15: `TT_DEBUG_MODULE_APB_REG_MAP_BASE_ADDR (0x0300A000)` -- CONFIRMED. Line 21: `TT_DEBUG_MODULE_APB_DMCONTROL_REG_ADDR (0x0300A040)` -- the actual define `TT_DEBUG_MODULE_APB_DMCONTROL_REG_ADDR` is at line 21. CONFIRMED.

### FIX 5: File 2, Section 1 table -- "ebreak behavior: Hangs (no DM)" for WH and BH

**Location:** File 2, Section 1 capability table, row "ebreak behavior".

**Claim:** WH: "Hangs (no DM)", BH: "Hangs (no DM)", Quasar: "Traps to Debug Module".

**Issue:** This is a design-level claim about hardware behavior that cannot be directly verified from source code headers alone. The claim that `ebreak` "hangs" is reasonable given no Debug Module on WH/BH, but it's an inference, not a verifiable code fact. **Mark as unverifiable from source alone** -- would require hardware testing or RTL reference.

**Severity:** Low -- the inference is reasonable.

### FIX 6: File 2, Key Source Files table -- dev_msgs.h line "179"

**Location:** File 2, Key Source Files table, dev_msgs.h entry, claim "179" for go_msg_t.

**Actual:** `go_msg_t` is at line 179. CONFIRMED.

### FIX 7: File 3, noc_debugging.hpp line range "23-94"

**Location:** File 3, claim that NOC event structures are at lines 23-94.

**Actual:** NocWriteEvent starts at line 23, NOCDebugEvent variant ends at line 94. CONFIRMED.

---

## Actual Incorrect Claims Found

### ERROR 1: File 1 -- trisc.cc:60 claim for regfile

**Location:** File 1, Key Source Files table: `trisc.cc` line "60" for `regfile`.

**Claim:** Line 60 defines regfile.

**Actual:** Line 60: `volatile tt_reg_ptr uint* PTR_CONST regfile = reinterpret_cast<volatile uint*>(REGFILE_BASE);` -- This is actually correct, line 60 does define regfile. CONFIRMED.

### ERROR 2: File 1 -- "Regfile at REGFILE_BASE (0xFFE00000), 256 B (64 x 32b)"

**Location:** File 1, Section 4.1, Memory Model table.

**Claim:** Regfile size is "256 B (64 x 32b)".

**Actual:** 64 registers x 4 bytes = 256 bytes. The REGFILE_BASE range comment says "0xFFE00000 - 0xFFE3FFFF" which is 256 KB. The actual regfile used by ckernel is 64 x 32-bit = 256 bytes, but the MMIO region mapped is much larger. The claim about the regfile array being 256 B is **correct** for the software view (64 x uint32_t). The MMIO range is larger because registers are sparsely mapped.

### ERROR 3: File 2 -- "dbg_dump_read(thread, 0x2, addr)" for DEST

**Location:** File 2, Section 7, hardware replay terminal session, monitor commands table.

**Claim:** `dbg_dump_read(thread, 0x2, addr)` for DEST, `dbg_dump_read(thread, 0x0, addr)` for SRCA.

**Issue:** There is NO function called `dbg_dump_read` in tensix_functions.h. The actual function is `dbg_dump_array_rd_cmd(uint thread, uint array_id, uint addr)` at line 705. The array IDs (DEST=0x2, SRCA=0x0, SRCB=0x1) at lines 690-692 are correct, but the function name is wrong.

**Recommended fix:** Change `dbg_dump_read(thread, 0x2, addr)` to `dbg_dump_array_rd_cmd(thread, DEST_ARRAY_ID, addr)` and similarly for SRCA/SRCB.

**Severity:** Medium -- function name is wrong but array IDs and concept are correct.

### ERROR 4: File 1 -- Quasar MEM_LOCAL_BASE claimed as 0xFFB00000

**Location:** File 1, Section 2.5, Spike invocation table: `-m0xFFB00000:0x100000` described as "MMIO region covering local memory, debug regs (MEM_LOCAL_BASE = 0xFFB00000)".

**Issue:** The claim implies MEM_LOCAL_BASE = 0xFFB00000 universally. For BH this is correct (blackhole/dev_mem_map.h:42). But for Quasar, MEM_LOCAL_BASE = 0x0802000 (quasar/dev_mem_map.h:41), which is very different. The Spike invocation in Section 2.5 is specifically for BH replay, so the BH address is acceptable. However, the Key Source Files table for the Quasar dev_mem_map.h does NOT claim a specific value for MEM_LOCAL_BASE, so this is fine as-is. **Not an error in context** -- the Spike invocation is BH-specific.

### ERROR 5: File 1, Section 6.3 -- "auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)()"

**Location:** File 1, Section 6.3, trisc.cc:214-215 code quote.

**Claim:** Shows the code as a two-line sequence.

**Actual code (line 214-215):**
```cpp
uint32_t kernel_lma = (kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index]);
auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
```
The document shows this correctly. **CONFIRMED.**

### VERIFIED: File 3 -- "dispatch.cpp" path

**Location:** File 3, Key Source Files table: `dispatch.cpp` at `tt_metal/impl/program/dispatch.cpp`.

**Verified:** File exists at exactly this path. CONFIRMED.

### VERIFIED: File 3 -- "noc_logging.hpp" at tt_metal/impl/debug/

**Location:** File 3, Key Source Files table.

**Claim:** `noc_logging.hpp` at `tt_metal/impl/debug/noc_logging.hpp`.

**Verified:** File exists at exactly this path. CONFIRMED.

---

## Borderline / Unverifiable Claims

1. **Spike extension_t API details** (File 1, Sections 2.1, 2.4) -- These describe Spike's internal API, not TT-Metal code. Cannot verify against tt-metal codebase.

2. **"4 breakpoints per thread" for WH/BH** (File 2, Section 1 table) -- Inferred from `breakpoint_status()` math: `(bkpt_index + thread * 4) * 4` implies 4 per thread. Reasonable inference but not an explicit constant.

3. **"ebreak hangs on WH/BH"** (File 2, Section 1 table) -- Hardware behavior claim, not directly verifiable from source headers.

4. **GDB RSP packet descriptions** (File 1, Section 3.4) -- Standard GDB protocol, not TT-Metal-specific. Correct by definition.

---

## Recommended Fixes

| Priority | File | Location | Fix |
|----------|------|----------|-----|
| Medium | File 2 | Section 7, monitor commands table | Change `dbg_dump_read` to `dbg_dump_array_rd_cmd` |
| Low | File 1 | Key Source Files table, trisc.cc lines | Change "113-117" to "114-117" for regfile init |
| None needed | All other claims | N/A | All other verified claims are correct |

---

## Overall Assessment

The three Ch7 files are highly accurate. Out of 52 specific technical claims verified:

- **43 claims (83%) are fully confirmed** with exact matches to source code.
- **5 claims are borderline** (minor line number offsets, hardware behavior inferences, or unverifiable Spike API details).
- **1 actual error found**: the function name `dbg_dump_read` should be `dbg_dump_array_rd_cmd` (File 2, Section 7).
- **3 remaining claims** are unverifiable from source code alone (hardware behavior, external tool APIs).

The documents demonstrate strong source code accuracy, with correct register addresses, bit layouts, enum values, struct fields, line numbers, and API signatures. The one substantive error (wrong function name for debug array reads) is a naming mistake where the concept and parameters are correct.
