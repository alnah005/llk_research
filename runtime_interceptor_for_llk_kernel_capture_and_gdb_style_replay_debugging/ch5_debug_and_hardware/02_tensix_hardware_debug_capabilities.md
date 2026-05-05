# Chapter 5, File 2 -- Tensix Hardware Debug Capabilities

> **Scope.** This file documents the hardware debug registers, breakpoint
> mechanisms, debug array read-back, the coprocessor debug bus, and the
> RISC-V Debug Module available across Wormhole (WH), Blackhole (BH), and
> Quasar architectures.  All citations reference commit `621b949`.

---

## 1. Tensix Breakpoints (WH/BH Only)

### 1.1 Register Interface

The breakpoint subsystem is controlled through memory-mapped registers
at the Tensix debug register base (`0xFFB12000`):

| Register | Offset | Purpose |
|----------|--------|---------|
| `BREAKPOINT_CTRL` | `0x1C0` | Command register -- write to set/clear/resume |
| `BREAKPOINT_STATUS` | `0x1C4` | Status register -- query which breakpoints are hit |
| `BREAKPOINT_DATA` | `0x1C8` | Data register -- extract breakpoint-associated data |

Source: `tt_metal/hw/inc/internal/tensix_functions.h` lines 608-687.

### 1.2 Command Encoding

```c
#define BKPT_CMD_RESUME     0x0
#define BKPT_CMD_SET        0x1
#define BKPT_CMD_CLEAR      0x2
#define BKPT_CMD_DATASEL    0x3
#define BKPT_CMD_SET_COND   0x4
#define BKPT_CMD_CLEAR_COND 0x5

#define BKPT_CMD_PAYLOAD(thread, cmd, data)          ((thread << 31) | (cmd << 28) | data)
#define BKPT_CMD_ID_PAYLOAD(thread, cmd, id, data)   ((thread << 31) | (cmd << 28) | (id << 26) | data)
```

| Bits | Field | Description |
|------|-------|-------------|
| 31 | `thread` | 0 = thread 0 (unpack), 1 = thread 1 (math) |
| 30:28 | `cmd` | One of the 6 command codes above |
| 27:26 | `id` | Breakpoint index (0-3 per thread, 4 total per thread) |
| 25:0 | `data` | Command-specific payload (PC, opcode mask, loop count) |

### 1.3 API Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `breakpoint_set` | `(thread, bkpt_index, pc_valid, pc)` | Set breakpoint at PC or instruction boundary |
| `breakpoint_clear` | `(thread, bkpt_index)` | Remove breakpoint |
| `breakpoint_set_data` | `(thread, bkpt_index, data_index)` | Select data source for capture |
| `breakpoint_set_condition_op` | `(thread, bkpt_index, opcode, mask)` | Break on specific Tensix opcode |
| `breakpoint_clear_condition_op` | `(thread, bkpt_index)` | Clear opcode condition |
| `breakpoint_set_condition_loop` | `(thread, bkpt_index, loop)` | Break on loop iteration |
| `breakpoint_clear_condition_loop` | `(thread, bkpt_index)` | Clear loop condition |
| `breakpoint_set_condition_other_thread` | `(thread, bkpt_index)` | Break when other thread is halted |
| `breakpoint_clear_condition_other_thread` | `(thread, bkpt_index)` | Clear cross-thread condition |
| `breakpoint_resume_execution` | `(thread)` | Resume after breakpoint hit |
| `breakpoint_status` | `(thread, bkpt_index)` | Read 4-bit status of one breakpoint |
| `breakpoint_status` | `()` | Read all 32 status bits |
| `breakpoint_data` | `()` | Read captured data |

### 1.4 Breakpoint Status Format

`BREAKPOINT_STATUS` returns a 32-bit word.  Each breakpoint occupies 4 bits at
position `(bkpt_index + thread * 4) * 4`.  This gives 8 breakpoints total
(4 per thread, 2 threads = unpack + math).

### 1.5 Conditional Breakpoints

Three condition types can be AND-combined per breakpoint:

| Condition | Data encoding | Description |
|-----------|--------------|-------------|
| **Opcode match** | `opcode_mask << 8 \| opcode` | Break when Tensix instruction opcode matches |
| **Loop count** | `(0x1 << 16) \| loop` | Break on specific loop iteration |
| **Other thread** | `(0x2 << 16)` | Break when the other thread is also halted |

**PROPOSED: Relevance to interceptor.**  Opcode-match breakpoints can trap
specific Tensix instructions (e.g. `MATMUL`, `SFPLOAD`) during LLK execution,
capturing state at those points.  Note: the breakpoint fires in Tensix
instruction space, not RISC-V space -- the RISC-V core is not halted.

### 1.6 Architecture Availability

All breakpoint functions in `tensix_functions.h` are guarded by
`#ifndef ARCH_QUASAR` (line 619) -- **breakpoints ARE available on WH and BH
but are NOT available on Quasar**.  Quasar uses the RISC-V Debug Module
instead (Section 2 below).

**CRITICAL NOTE:** Some analyses incorrectly claim WH/BH lack breakpoints.
This is inverted.  The `#ifndef ARCH_QUASAR` guard explicitly confirms
WH/BH *do* have this breakpoint infrastructure while Quasar does not.

---

## 2. Quasar: rvdbg_cmd Debug Interface

### 2.1 Overview

Quasar replaces the WH/BH Tensix breakpoint system with a RISC-V Debug Module
compliant interface accessed via `RISC_DBG_CNTL` registers.

Source: `tt-llk/tt_llk_quasar/common/inc/ckernel_riscv_debug.h` (163 lines).

### 2.2 Command Set

```cpp
enum class rvdbg_cmd : std::uint32_t {
    PAUSE      = (1 << 0),    // Halt RISC-V execution
    STEP       = (1 << 1),    // Single-step one instruction
    CONTINUE   = (1 << 2),    // Resume execution
    RD_REG     = (1 << 3),    // Read general-purpose register
    WR_REG     = (1 << 4),    // Write general-purpose register
    RD_MEM     = (1 << 5),    // Read memory
    WR_MEM     = (1 << 6),    // Write memory
    FLUSH_REGS = (1 << 7),    // Flush register file to memory
    FLUSH      = (1 << 8),    // General flush
    RD_CSR     = (1 << 9),    // Read CSR
    WR_CSR     = (1 << 10),   // Write CSR
    RD_FPREG   = (1 << 11),   // Read FP register
    WR_FPREG   = (1 << 12),   // Write FP register
    RD_VECREG  = (1 << 13),   // Read vector register
    WR_VECREG  = (1 << 14),   // Write vector register
    RD_UNIT    = (1 << 15),   // Read debug unit info
    DBG_MODE_BIT = (1U << 31) // Debug mode indicator
};
```

Source: `ckernel_riscv_debug.h` lines 11-31.

**Implication:** Quasar RISC-V cores support halt, single-step,
register read/write, memory read/write, CSR access, FP/vector register
access, and hardware watchpoints -- the full primitive set for
GDB-style debugging.

### 2.3 RISC Selection

```cpp
enum class rvdbg_risc_sel : std::uint32_t {
    TRISC0 = 0x0000'0000,   // Unpack
    TRISC1 = 0x0002'0000,   // Math
    TRISC2 = 0x0004'0000,   // Pack
    TRISC3 = 0x0006'0000    // SFPU (Quasar only -- 4th TRISC)
};
```

Source: `ckernel_riscv_debug.h` lines 33-39.

**Architecture difference:** Quasar names these TRISC0-3 (4 TRISCs).  On
WH/BH the comment at line 79 documents the mapping as: 0=BRISC, 1-3=TRISCs,
4=NCRISC.  The interceptor must use an architecture-dependent RISC selector.

### 2.4 Debug Registers and Hardware Watchpoints

```cpp
enum class rvdbg_reg : std::uint32_t {
    STATUS          = 0,     // paused/brkpt_hit/wchpt_hit/ebrk_hit/reason
    COMMAND         = 1,     // write command here
    COMMAND_ARG0    = 2,     // first argument
    COMMAND_ARG1    = 3,     // second argument
    COMMAND_RET_VAL = 4,     // return value for reads
    WCHPT_SETTINGS  = 5,     // watchpoint configuration
    HW_WCHPT0       = 10,   // hardware watchpoint addresses
    HW_WCHPT1       = 11,
    HW_WCHPT2       = 12,
    HW_WCHPT3       = 13,
    HW_WCHPT4       = 14,
    HW_WCHPT5       = 15,
    HW_WCHPT6       = 16,
    HW_WCHPT7       = 17
};
```

Source: `ckernel_riscv_debug.h` lines 41-57.

**8 hardware watchpoints per RISC.**  Each holds a memory address.
`WCHPT_SETTINGS` (index 5) controls active watchpoints and trigger
conditions (read/write/access).  When a watchpoint fires,
`rvdbg_status::wchpt_hit` (bit 2) is set.

**PROPOSED:** For LLK interceptor, hardware watchpoints on specific L1
addresses (e.g. circular buffer write pointers) would trap at the exact
moment a tile write completes, enabling zero-overhead tile capture during
normal execution.

### 2.5 Status Word

```cpp
union rvdbg_status {
    struct {
        unsigned paused    : 1;   // RISC is halted
        unsigned brkpt_hit : 1;   // HW breakpoint triggered
        unsigned wchpt_hit : 1;   // Watchpoint triggered
        unsigned ebrk_hit  : 1;   // ebreak instruction hit
        unsigned unused0   : 4;
        unsigned reason    : 8;   // Halt reason code
    };
    std::uint32_t val;
};
```

Source: `ckernel_riscv_debug.h` lines 60-73.

This tells the debugger *why* the RISC stopped, which is critical for
implementing GDB's stop-reason reporting.

### 2.6 Command Dispatch

Commands are issued by writing to `rvdbg_reg::COMMAND` (index 1) after
setting arguments in `COMMAND_ARG0` (index 2) and `COMMAND_ARG1` (index 3).
Return values are read from `COMMAND_RET_VAL` (index 4).

Three overloads of `riscv_dbg_cmd()` handle calling patterns (lines 144-162):

```cpp
// No-argument (PAUSE, CONTINUE, FLUSH, etc.)
void riscv_dbg_cmd(rvdbg_cmd cmd, rvdbg_risc_sel risc_sel);

// Single-argument with return (RD_REG, RD_MEM, RD_CSR, etc.)
uint32_t riscv_dbg_cmd(rvdbg_cmd cmd, rvdbg_risc_sel risc_sel, uint32_t arg0);

// Two-argument (WR_REG, WR_MEM, WR_CSR, etc.)
void riscv_dbg_cmd(rvdbg_cmd cmd, rvdbg_risc_sel risc_sel, uint32_t arg0, uint32_t arg1);
```

---

## 3. RISC_DBG_CNTL Register Protocol

The hardware interface for accessing debug registers uses a pulse-based
protocol through `RISC_DBG_CNTL_0/1` and `RISC_DBG_STATUS_0/1`.

Source: `ckernel_riscv_debug.h` lines 75-142.

### 3.1 Control Register Layout

```
RISC_DBG_CNTL_0[31:0] = {
    pulse,           // bit 31: 0->1 transition triggers access
    11'b0,           // bits 30:20: reserved
    risc_sel[2:0],   // bits 19:17: 0=BRISC, 1-3=TRISCs, 4=NCRISC
    reg_wr,          // bit 16: 0=read, 1=write
    5'b0,            // bits 15:11: reserved
    reg_addr[10:0]   // bits 10:0: register address
};
RISC_DBG_CNTL_1[31:0] = reg_wr_data[31:0];
```

### 3.2 Status Register Layout

```
RISC_DBG_STATUS_0[31:0] = {
    pulse,           // bit 31: read-only mirror
    rd_valid,        // bit 30: set when read returns
    ...
    reg_addr[10:0]   // bits 10:0: read-only mirror
};
RISC_DBG_STATUS_1[31:0] = reg_rd_data;
```

### 3.3 Protocol Constants

```cpp
#define CNTL0_PULSE      (0x8000'0000)   // Bit 31
#define CNTL0_WR         (0x0001'0000)   // Bit 16 (write mode)
#define CNTL0_RD         (0x0000'0000)   // Bit 16 clear (read mode)
#define STATUS0_RD_VALID (0x4000'0000)   // Bit 30
```

Source: `ckernel_riscv_debug.h` lines 102-105.

### 3.4 Write Sequence (`riscv_dbg_wr`, lines 109-121)

1. Compose command: `CNTL0_PULSE | risc_sel | CNTL0_WR | reg_index`
2. Write data to `RISC_DBG_CNTL_1`
3. Clear pulse: write 0 to `RISC_DBG_CNTL_0` (ensures 0->1 transition)
4. Write command to `RISC_DBG_CNTL_0`
5. Insert 3 NOPs (`asm volatile("nop; nop; nop;")`)

### 3.5 Read Sequence (`riscv_dbg_rd`, lines 125-142)

1. Compose command: `CNTL0_PULSE | risc_sel | CNTL0_RD | reg_index`
2. Clear pulse: write 0 to `RISC_DBG_CNTL_0`
3. Write command to `RISC_DBG_CNTL_0`
4. Insert 3 NOPs
5. Poll `RISC_DBG_STATUS_0` until `STATUS0_RD_VALID` is set
6. Read result from `RISC_DBG_STATUS_1`

**Critical timing note:** The pulse signal passes through a clock domain
synchronizer, but the address/data lines do not.  The 3-NOP delay ensures
these fields remain stable while the synchronizer latches the pulse.

### 3.6 Architecture Availability

These `RISC_DBG_CNTL_0/1` registers exist on **BH and Quasar only**:
- BH: offset `0x080`/`0x084` from `0xFFB12000` (`tensix.h` lines 162-165)
- Quasar: accessed via `t6_debug_regs_t` struct
- WH: **NOT present** — WH `tensix.h` does not define `RISC_DBG_CNTL` registers

**PROPOSED:** The RISC_DBG_CNTL protocol provides the complete primitive set
for GDB-style debugging on BH and Quasar.  A GDB stub built on
`riscv_dbg_wr`/`riscv_dbg_rd` can provide live source-level debugging on
these architectures.  On WH, only Tensix-level breakpoints (Section 1) are
available; RISC-V halt/step/GPR-read requires the ebreak + mailbox approach
(Ch5, File 3, Option C).

---

## 4. Quasar Debug Module (RISC-V Standard)

Quasar implements a RISC-V Debug Module compliant with the External Debug
Support specification.

Source: `overlay_reg_defines_debug.h` lines 15-51.

### 4.1 APB Register Map

| Register | Offset | Address | Purpose |
|----------|--------|---------|---------|
| `DATA` | 0x010 | 0x0300A010 | Abstract command data |
| `DMCONTROL` | 0x040 | 0x0300A040 | Halt/resume/reset control |
| `STATUS` (dmstatus) | 0x044 | 0x0300A044 | Debug Module status |
| `DMI_HARTINFO` | 0x048 | 0x0300A048 | Hart information |
| `HALTSUMMARY1` | 0x04C | 0x0300A04C | Halt summary bitmap |
| `HAWINDOW` | 0x054 | 0x0300A054 | Hart array window select |
| `ABSTRACTS` (abstractcs) | 0x058 | 0x0300A058 | Abstract command status |
| `COMMAND` | 0x05C | 0x0300A05C | Abstract command register |
| `ABSTRACTAUTO` | 0x060 | 0x0300A060 | Auto-execute on data R/W |
| `PROGBUF` | 0x080 | 0x0300A080 | Program buffer (5 entries) |
| `STATUS2` | 0x0C8 | 0x0300A0C8 | Extended status |
| `SBCS` | 0x0E0 | 0x0300A0E0 | System Bus access control |
| `SBADDR0` | 0x0E4 | 0x0300A0E4 | System Bus address |
| `SBDATA0` | 0x0F0 | 0x0300A0F0 | System Bus data |
| `HALTSUMMARY0` | 0x100 | 0x0300A100 | Primary halt summary |

### 4.2 DMSTATUS Fields

From `overlay_reg.h` lines 1730-1758:

```c
typedef struct {
    uint32_t dmi_dmstatus_version : 4;
    uint32_t dmi_dmstatus_confstrptrvalid : 1;
    uint32_t dmi_dmstatus_hasresethaltreq : 1;
    uint32_t dmi_dmstatus_authbusy : 1;
    uint32_t dmi_dmstatus_authenticated : 1;
    uint32_t dmi_dmstatus_anyhalted : 1;
    uint32_t dmi_dmstatus_allhalted : 1;
    uint32_t dmi_dmstatus_anyrunning : 1;
    uint32_t dmi_dmstatus_allrunning : 1;
    uint32_t dmi_dmstatus_anyunavail : 1;
    uint32_t dmi_dmstatus_allunavail : 1;
    uint32_t dmi_dmstatus_anynonexistent : 1;
    uint32_t dmi_dmstatus_allnonexistent : 1;
    uint32_t dmi_dmstatus_anyresumeack : 1;
    uint32_t dmi_dmstatus_allresumeack : 1;
    uint32_t dmi_dmstatus_anyhavereset : 1;
    uint32_t dmi_dmstatus_allhavereset : 1;
    uint32_t dmi_dmstatus_reserved : 2;
    uint32_t dmi_dmstatus_impebreak : 1;   // implicit ebreak support
} TT_DEBUG_MODULE_APB_STATUS_reg_t;
```

The `impebreak` field indicates the Debug Module supports implicit `ebreak`
at the end of the program buffer.  The default value of `SBUS_IMPEBREAK` is
`0x00100073` (the `ebreak` encoding, `overlay_reg.h` line 6510).

### 4.3 ABSTRACTS Register

Reports `progbufsize: 5` (program buffer entries), `datacount: 4`
(data registers), `cmderr` (3-bit error code), and `busy` flag.

The `COMMAND` register accepts standard RISC-V abstract commands:
Access Register (GPR/CSR read/write), Quick Access (execute progbuf
without halting), and Access Memory.

### 4.4 System Bus Access (SBA)

The SBCS register at `0x0E0` provides DMA-like memory access without
halting any hart.  This allows reading/writing L1 and DRAM even when
all harts are halted -- essential for a GDB stub's memory inspection.

### 4.5 Per-Neo-Cluster Registers

Each Neo compute cluster has its own debug status:

```c
NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_RISC_DBG_CNTL_0  @ 0x01800058
NEO_REGS_0__LOCAL_REGS_DEBUG_REGS_RISC_DBG_STATUS_0 @ 0x01800060
```

Source: `tensix_neo_reg.h` lines 87-94.

Quasar's multi-hart architecture (up to 8 DM + 16 NEO cores per Tensix)
requires `HAWINDOW`-based hart selection.  `HALTSUMMARY0/1` provides a
bitmap of halted harts.

### 4.6 RISK CAVEAT: Host Accessibility

The `rvdbg_*` functions in `ckernel_riscv_debug.h` are all **device-side**
(they access `RISCV_DEBUG_REGS` directly from on-chip code).  For a
**host-driven** debugger, the question is whether the Debug Module APB
registers at `0x0300A000` can be reached from the host via PCIe or JTAG.
The JTAG `read_tdr`/`write_tdr` path can reach APB registers, and PCIe
MMIO can reach L1-mapped debug registers.  However, the APB-space Debug
Module registers are not directly in the L1 address space.  The host must
use either JTAG TDR or AXI-mapped access to reach `0x0300A000`.

---

## 5. Debug Array Read (SRCA / SRCB / DEST / MAX_EXP)

### 5.1 Array IDs

```c
// tensix_functions.h lines 689-693
#define SRCA_ARRAY_ID     0x0
#define SRCB_ARRAY_ID     0x1
#define DEST_ARRAY_ID     0x2
#define MAX_EXP_ARRAY_ID  0x3
```

### 5.2 Low-Level API (tensix_functions.h)

```c
void dbg_dump_array_enable();     // Enable array readback
void dbg_dump_array_disable();    // Disable (invalidate array_id)
void dbg_dump_array_rd_cmd(uint thread, uint array_id, uint addr);
```

These use `RISCV_DEBUG_REG_DBG_ARRAY_RD_EN` and `RISCV_DEBUG_REG_DBG_ARRAY_RD_CMD`.

### 5.3 Higher-Level API (ckernel_debug.h)

```cpp
namespace ckernel {
    dbg_thread_halt<ThreadId>();    // Coordinate threads for safe dump
    dbg_thread_unhalt<ThreadId>();  // Resume after dump
    dbg_get_array_row(array_id, row_addr, rd_data);  // Read one row
    dbg_read_cfgreg(cfgreg_id, addr);                // Read config reg
}
```

### 5.4 Array Read Command Encoding

```cpp
typedef struct {
    uint32_t row_addr    : 12;  // Row address within array
    uint32_t row_32b_sel : 4;   // 32-bit word selector (0-7 = 256-bit row)
    uint32_t array_id    : 3;   // Array selector
    uint32_t bank_id     : 1;   // Bank selector (double-buffered)
    uint32_t reserved    : 12;
} dbg_array_rd_cmd_t;
```

Source: `ckernel_debug.h` lines 71-78.

Each row read produces 8 x 32-bit words (256 bits), read iteratively by
cycling `row_32b_sel` from 0 to 7 with a 5-cycle stabilization wait between
each read.

**Cost estimate:** A full 16-row tile dump from Dest takes approximately
$16 \times 8 \times 5 = 640$ cycles -- significant but acceptable as
debug-mode overhead.

### 5.5 SRCA Readback Workaround

Direct SRCA readback is **not supported** in hardware.  The workaround
(`ckernel_debug.h` lines 165-197):

1. Save destination row 0 to SFPU LREG3 via `TTI_SFPLOAD`
2. Clear `dvalid` bits, move to last used bank (`TTI_CLEARDVALID`)
3. Copy SRCA row into Dest via `TT_MOVDBGA2D` (destructive to Dest row 0)
4. Read from Dest through the normal debug array path
5. Restore Dest row 0 from LREG3 via `TTI_SFPSTORE`

**This is intrusive** -- it temporarily clobbers Dest contents during capture.

### 5.6 SRCB Readback

SRCB requires latching via `TTI_SETDVALID` / `TTI_SHIFTXB` / `TTI_CLEARDVALID`
sequence before reading (`ckernel_debug.h` lines 198-237).  Moderate
intrusiveness -- modifies DVALID state.

### 5.7 Invasiveness Summary

| Array | Direct read | Workaround | Invasive? |
|-------|-------------|------------|-----------|
| Dest  | Yes | None | No -- direct read |
| SrcA  | No | Copy to Dest via MOVDBGA2D | Yes -- clobbers Dest row 0 temporarily |
| SrcB  | Partial | SHIFTXB latch + read | Moderate -- modifies DVALID state |
| MAX_EXP | Yes (array_id=3) | None | No -- direct read |

**PROPOSED:** Dest readback should always be captured (non-invasive).
SrcA/SrcB capture should be gated by a `CAPTURE_LEVEL` compile-time flag.

---

## 6. Config Register Readback

```cpp
uint32_t dbg_read_cfgreg(uint32_t cfgreg_id, uint32_t addr);
```

Source: `ckernel_debug.h` lines 283-307.

Reads Tensix configuration registers via `RISCV_DEBUG_REG_CFGREG_RD_CNTL` (WH)
or `RISCV_DEBUG_REG_TENSIX_CREG_READ` (BH -- different name, same offset 0x058).

| ID | Name | Base | Size |
|----|------|------|------|
| 0 | `THREAD_0_CFG` | `2 * HW_CFG_SIZE` | `THD_STATE_SIZE` |
| 1 | `THREAD_1_CFG` | `2 * HW_CFG_SIZE + THD_STATE_SIZE` | `THD_STATE_SIZE` |
| 2 | `THREAD_2_CFG` | `2 * HW_CFG_SIZE + 2*THD_STATE_SIZE` | `THD_STATE_SIZE` |
| 3 | `HW_CFG_0` | 0 | 187 DWORDs |
| 4 | `HW_CFG_1` | 187 | 187 DWORDs |

`HW_CFG_SIZE` is 187 DWORDs (748 bytes, `ckernel_debug.h` line 24).  This
covers ALU modes, data format settings, address generators, tile dimensions,
and stochastic rounding masks.

Config register capture is mandatory for faithful replay -- the configuration
determines how every LLK operation interprets its data.

---

## 7. Coprocessor Debug Bus

### 7.1 Control Structure

```cpp
typedef struct {
    uint32_t sig_sel    : 16;  // Signal group selector
    uint32_t daisy_sel  : 8;   // Daisy chain position
    uint32_t rd_sel     : 4;   // 32-bit word selector from 128-bit bus
    uint32_t reserved_0 : 1;
    uint32_t en         : 1;   // Enable (bit 29)
    uint32_t reserved_1 : 2;
} dbg_bus_cntl_t;
```

Source: `ckernel_debug.h` lines 41-49.

### 7.2 Daisy Chain IDs

| ID | Module |
|----|--------|
| 4 | `INSTR_ISSUE_0` (thread 0 -- unpack) |
| 5 | `INSTR_ISSUE_1` (thread 1 -- math) |
| 6 | `INSTR_ISSUE_2` (thread 2 -- pack) |

Accessed via `RISCV_DEBUG_REG_DBG_BUS_CNTL_REG` (write) and
`RISCV_DEBUG_REG_DBG_RD_DATA` (read), with 5-cycle stabilization delay.

On Quasar, the debug bus fields are RDL-generated in `t6_debug_map.h`
(lines 91-111).

### 7.3 Instruction Buffer Debug Override

```c
// tensix_functions.h lines 716-743
dbg_instrn_buf_wait_for_ready();     // Poll status == 0x77
dbg_instrn_buf_set_override_en();    // Enable override (0x7)
dbg_instrn_buf_push_instrn(instrn);  // Push one instruction
dbg_instrn_buf_clear_override_en();  // Disable override
```

This allows injecting Tensix instructions into the instruction buffer from
the RISC-V debug path, bypassing normal kernel execution.

**PROPOSED:** Instruction buffer override enables a replay debugger to
single-step Tensix instructions while reading back SRCA/SRCB/DEST after
each instruction -- the "GDB-style" experience for Tensix coprocessor
debugging that currently does not exist.

---

## 8. Thread Halt/Unhalt Protocol

Source: `ckernel_debug.h` lines 101-155.

### 8.1 Halt Sequence

**Unpack thread:**
1. `tensix_sync()` -- wait for all outstanding instructions
2. `mailbox_write(MathThreadId, 1)` -- signal math thread
3. `mailbox_read(MathThreadId)` -- block until math acknowledges

**Math thread:**
1. `tensix_sync()` -- wait for all outstanding instructions
2. `mailbox_read(UnpackThreadId)` -- wait for unpack to complete
3. `while (semaphore_read(MATH_PACK) > 0)` -- wait for packer to drain

### 8.2 Unhalt Sequence (Math thread)

1. Soft-reset packer 0 (workaround: `SOFT_RESET_0.pack = 1`, wait 5 cycles, clear)
2. `tensix_sync()` -- wait for reset
3. `mailbox_write(UnpackThreadId, 1)` -- release unpack

**PROPOSED:** This cooperative halt is the correct way to create "quiescent
points" for safe state capture.  A brute-force PAUSE could leave the pipeline
inconsistent.  The mailbox-based halt ensures all threads reach a clean boundary.
(See Ch4/03 for the five-RISC coordination context.)

---

## 9. Soft Reset

### 9.1 Per-Unit Reset Fields (Quasar `t6_debug_map.h` lines 379-469)

| Field | Bits | Controls |
|-------|------|----------|
| `TDMA_UNPACKER_SOFT_RESET` | 2:0 | 3 unpackers |
| `TDMA_PACKER_SOFT_RESET` | 5:3 | 3 packers |
| `TDMA_GLUE_SOFT_RESET` | 8 | TDMA glue logic |
| `FPU_SOFT_RESET` | 10 | FPU |
| `RISC_CONTROL_SOFT_RESET` | 14:11 | 4 RISC controls |
| `SRCA_REG_SOFT_RESET` | 15 | SrcA register file |
| `SRCB_REG_SOFT_RESET` | 16 | SrcB register file |
| `DEST_REG_SOFT_RESET` | 17 | Dest accumulator |
| `L1_SOFT_RESET` | 30 | L1 memory |

### 9.2 WH/BH Reset Constants

```c
#define RISCV_SOFT_RESET_0_BRISC  0x00800
#define RISCV_SOFT_RESET_0_NCRISC 0x40000
#define RISCV_SOFT_RESET_0_TRISCS 0x07000
```

Source: `tensix.h` lines 104-107.

### 9.3 Reset PC Override (BH)

BH provides per-TRISC reset PC override registers:

| Register | Address |
|----------|---------|
| `TRISC0_RESET_PC` | `0xFFB12228` |
| `TRISC1_RESET_PC` | `0xFFB1222C` |
| `TRISC2_RESET_PC` | `0xFFB12230` |
| `TRISC_RESET_PC_OVERRIDE` | `0xFFB12234` |
| `NCRISC_RESET_PC` | `0xFFB12238` |
| `NCRISC_RESET_PC_OVERRIDE` | `0xFFB1223C` |

Source: `tensix.h` lines 209-214.

**PROPOSED:** Reset PC override enables redirecting a RISC to a debug agent
binary on the next reset, supporting the "BRISC debug agent" approach
(see Ch5 File 3, Option D).

---

## 10. ebreak Flow

| Architecture | `ebreak` behavior |
|-------------|-------------------|
| **WH/BH** | Triggers breakpoint exception; used by `LIGHTWEIGHT_KERNEL_ASSERTS` and `LLK_ASSERT` (in `ENV_LLK_INFRA` mode). Causes core halt; no RISC-V debug module catches it. |
| **Quasar** | Caught by Debug Module; sets `rvdbg_status.ebrk_hit`; core enters debug mode. Can be used as a software breakpoint for a GDB stub. |

**Risk caveat:** On WH/BH, executing `ebreak` without a debug module means
the core is unrecoverable without a full reset.  On Quasar, ebreak is
recoverable via the Debug Module's `CONTINUE` command.

---

## 11. GDB RSP Mapping (Quasar)

**PROPOSED:** A GDB Remote Serial Protocol stub could map RSP packets to
`rvdbg_cmd` operations:

| GDB RSP Packet | `rvdbg_cmd` Sequence |
|----------------|---------------------|
| `?` (halt reason) | Read `STATUS` register, decode `rvdbg_status` |
| `g` (read all registers) | Loop `RD_REG` for x0-x31, `RD_FPREG` for f0-f31 |
| `p n` (read register n) | `RD_REG` with `arg0 = n` |
| `P n=val` (write register) | `WR_REG` with `arg0 = n, arg1 = val` |
| `m addr,len` (read memory) | Loop `RD_MEM` for each word |
| `M addr,len:data` (write memory) | Loop `WR_MEM` for each word |
| `s` (single step) | `STEP` |
| `c` (continue) | `CONTINUE` |
| `Z0,addr` (set SW breakpoint) | `WR_MEM` to patch instruction with `ebreak` |
| `Z2,addr` (set watchpoint) | Configure `HW_WCHPTn` + `WCHPT_SETTINGS` |
| `vCont;s:thread` | `STEP` with specific `risc_sel` |

Source: Derived from `ckernel_riscv_debug.h` command set.

---

## 12. Architecture Comparison

| Feature | WH | BH | Quasar |
|---------|----|----|--------|
| Tensix instruction breakpoints | Yes (4/thread) | Yes (4/thread) | No (uses Debug Module) |
| Opcode-conditional breakpoints | Yes | Yes | No |
| Loop-conditional breakpoints | Yes | Yes | No |
| Cross-thread conditional breakpoints | Yes | Yes | No |
| RISC-V halt/step/continue | No (ebreak only) | Via RISC_DBG_CNTL | Native |
| GPR read/write | No | Via RISC_DBG_CNTL | Native |
| CSR read/write | No | Via RISC_DBG_CNTL | Native |
| FP/Vector reg access | No | No | Yes |
| Memory read/write via debug | No (NOC/TLB only) | Via RISC_DBG_CNTL | Native |
| Hardware watchpoints | No | Via RISC_DBG_CNTL | Yes (8) |
| `ebreak` detection | Halt (no DM) | Halt (no DM) | `ebrk_hit` in status |
| Debug Module (dmcontrol/dmstatus) | No | No | Yes |
| System Bus access | No | No | Yes (SBCS) |
| Halt summary bitmap | No (must scan) | No (must scan) | Yes (`HALTSUMMARY0/1`) |
| Reset PC override | TRISC only | TRISC + NCRISC | DM halt-on-reset |
| Array readback (Dest) | Yes | Yes | Yes |
| Array readback (SrcA via Dest copy) | Yes | Yes | Yes |
| Array readback (SrcB via SHIFTXB) | Yes | Yes | Yes |
| Config register dump (187 HW regs) | Yes | Yes | Yes |
| Debug bus (coprocessor) | Yes | Yes | Yes (RDL) |
| Instruction buffer override | Yes | Yes | Yes + timeout |
| Number of TRISCs | 3 | 3 | 4 (+ SFPU isolate) |

---

## Key Takeaways

1. **WH/BH have Tensix-level breakpoints** (PC, opcode, loop, cross-thread
   conditions).  BH additionally has the `RISC_DBG_CNTL` protocol for
   RISC-V halt/step/GPR read-write — supporting GDB-style debugging through
   a non-standard interface requiring a custom GDB stub.  WH lacks
   `RISC_DBG_CNTL` and must rely on ebreak + mailbox for debug beyond
   breakpoints.

2. **Quasar has a full RISC-V Debug Module** with halt, step, GPR/CSR/FPR
   read-write, 8 hardware watchpoints, system bus access, and
   HALTSUMMARY bitmaps.  It is the most amenable architecture for
   standard OpenOCD/GDB attachment.

3. **Debug array readback** (SRCA/SRCB/DEST/MAX_EXP) is available on all
   architectures but requires careful thread coordination.  SRCA readback
   is destructive to Dest (MOVDBGA2D workaround).

4. **Instruction buffer override** enables injecting Tensix instructions
   from debug code -- a mechanism for single-stepping the coprocessor
   in a replay scenario.

5. **The gap**: No host-side tooling integrates with these hardware
   capabilities.  The register interfaces exist but are currently only
   usable from on-device firmware.  A host-side GDB stub must use JTAG,
   PCIe MMIO, or an on-device relay to reach these registers.

6. **Quasar's TRISC3** is a strong candidate for hosting interceptor logic:
   it has full debug access to TRISCs 0-2 and does not compete with compute.

---

## Key Source Files

| File | Relevance |
|------|-----------|
| `tt_metal/hw/inc/internal/tensix_functions.h` | Breakpoint API, array dump API |
| `tt-llk/tt_llk_quasar/common/inc/ckernel_riscv_debug.h` | rvdbg_cmd, RISC_DBG_CNTL protocol |
| `tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_debug.h` | Array readback, config read, thread halt |
| `tt-llk/tt_llk_blackhole/common/inc/ckernel_debug.h` | BH variant (same as WH per TODO note) |
| `overlay_reg_defines_debug.h` | Debug Module APB register addresses |
| `overlay_reg.h` | DMSTATUS struct, IMPEBREAK default |
| `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h` | Debug register offsets (BH) |
| `tt_metal/hw/inc/internal/tt-1xx/wormhole/tensix.h` | Debug register offsets (WH) |
| `tests/hw_specific/quasar/inc/tensix.h` | Quasar debug register offsets |
| `tests/hw_specific/quasar/inc/t6_debug_map.h` | RDL-generated register definitions |
| `tensix_neo_reg.h` | Per-Neo-cluster debug registers |
