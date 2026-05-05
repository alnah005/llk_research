# Chapter 7, File 2 -- On-Hardware and Host-Model Replay

> **Scope.** This file designs two alternative replay strategies beyond
> emulator-based replay (File 01): (a) replaying a captured snapshot on
> actual Tensix hardware using debug registers and a GDB RSP bridge, and
> (b) replaying by compiling the kernel source with a host compiler
> against functional stubs and debugging under native x86 GDB.  It also
> introduces the BRISC-as-debug-agent concept with a concrete state
> machine design.  All designs are **PROPOSED** unless explicitly noted
> as existing infrastructure.
>
> All source citations reference commit `621b949`.

**Cross-References:**
- Ch 2 (state inventory: regfile[64], config registers, L1, CBs, semaphores)
- Ch 4 (5-RISC execution model, dispatch protocol, `go_msg_t` launch)
- Ch 5 File 2 (Tensix hardware debug: WH breakpoints, BH RISC_DBG_CNTL registers, Quasar Debug Module and `rvdbg_cmd`)
- Ch 5 File 3 (three-tier strategy, E-taxonomy, host functional model `trisc.cpp`, GDB-attach options)
- Ch 6 File 3 (`.ttksnap` FlatBuffer schema, `KernelInvocationSnapshot` root type)
- Ch 8 (implementation phases -- forward reference)

---

## Part A: On-Hardware Replay

### 1. Architecture Overview

On-hardware replay writes a captured snapshot back to a physical Tensix
core's L1, loads the kernel binary, and controls execution through
hardware debug primitives.  This provides perfect fidelity at the cost
of requiring device access and exclusive core ownership.

The hardware debug capabilities vary by architecture:

| Capability | Wormhole | Blackhole | Quasar |
|-----------|----------|-----------|--------|
| Tensix breakpoints (4 per thread) | Yes | Yes | No (uses Debug Module) |
| `RISC_DBG_CNTL_0/1` registers | **No** | **Yes** (raw registers) | **Yes** (registers + API) |
| `rvdbg_cmd` software API | No | **No** (must port) | **Yes** |
| Full RISC-V Debug Module | No | No | Yes (at `0x0300A000`) |
| Hardware watchpoints | No | No | Yes (8 via HW_WCHPT0-7) |
| Single-step instruction | No | Yes (via RISC_DBG_CNTL) | Yes (Debug Module) |
| `ebreak` behavior | Hangs (no DM) | Hangs (no DM) | Traps to Debug Module |

**Critical architecture distinction:** The `RISC_DBG_CNTL_0/1` hardware
registers exist on both Blackhole and Quasar.  However, the software
abstraction layer (`rvdbg_cmd`, `rvdbg_risc_sel`, `riscv_dbg_wr`,
`riscv_dbg_rd`) defined in `ckernel_riscv_debug.h` exists **only** under
`tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/`.  There is no
corresponding file under `tt_llk_blackhole/`.  BH has the raw registers
at `tensix.h:162-165` but lacks the convenience API.  On BH, the debug
agent must implement its own register access functions using the raw
RISC_DBG_CNTL register protocol (straightforward to port from Quasar).

**Wormhole has no RISC_DBG_CNTL at all.**  Only `ebreak` + mailbox
debugging is available on WH.

---

### 2. On-Hardware Debug Agent Alternatives

#### 2.1 Three Strategies

| Strategy | Description | Required HW |
|----------|-------------|------------|
| **A: Debug registers via PCIe** | Host writes RISC_DBG_CNTL_0/1 via TLB-mapped NOC writes | BH/Quasar with RISC_DBG_CNTL |
| **B: ebreak + mailbox protocol** | Kernel compiled with `ebreak` at debug points; trap handler saves state to L1 mailbox | All architectures |
| **C: BRISC-as-debug-agent** | Dedicated BRISC firmware controls TRISCs via breakpoint API and L1 inspection | All architectures |

#### 2.2 Scoring Matrix

| Criterion | Weight | A: Debug Regs/PCIe | B: ebreak+Mailbox | C: BRISC Agent |
|-----------|--------|-------------------|-------------------|----------------|
| Architecture coverage | 4 | 2 (BH/Quasar only) | 5 (all) | 4 (all, breakpoint API) |
| Step/inspect fidelity | 5 | 5 (hardware native) | 3 (GPR only, no CSR) | 4 (breakpoints + L1 + regfile) |
| Intrusiveness | 4 | 5 (non-invasive) | 2 (recompilation) | 3 (BRISC occupied) |
| GDB RSP completeness | 4 | 5 (all packets) | 3 (limited) | 4 (most packets) |
| Latency per command | 3 | 3 (PCIe round-trip) | 2 (poll delay) | 3 (L1 poll delay) |
| Implementation effort | 3 | 3 | 4 (trap handler) | 2 (agent firmware) |
| **Weighted total** | | **93** | **74** | **81** |

> **PROPOSED: Phased strategy.** Phase 3: Strategy B (ebreak+mailbox) as
> a quick prototype.  Phase 4: Strategy C (BRISC-as-agent) for richer
> debugging.  Phase 5: Strategy A (debug registers) when Quasar hardware
> access is validated.

---

### 3. The RISC_DBG_CNTL Register Protocol

The RISC_DBG_CNTL register pair provides the foundation for on-hardware
debugging on BH and Quasar.

#### 3.1 Register Bit Layout

From `ckernel_riscv_debug.h` lines 76-96 (Quasar):

```
RISC_DBG_CNTL_0[31:0] = {
    pulse,              // bit [31]: 0->1 transition triggers access
    11'b0,              // bits [30:20]: reserved
    risc_sel[2:0],      // bits [19:17]: 0=BRISC, 1-3=TRISC0-2, 4=NCRISC
    reg_wr,             // bit [16]: 0=read, 1=write
    5'b0,               // bits [15:11]: reserved
    reg_addr[10:0]      // bits [10:0]: register index
};

RISC_DBG_CNTL_1[31:0] = reg_wr_data[31:0];

RISC_DBG_STATUS_0[31:0] = {
    pulse,              // bit [31]: read-only mirror
    rd_valid,           // bit [30]: set when read completes
    ...
};

RISC_DBG_STATUS_1[31:0] = reg_rd_data (latched when rd_valid goes high);
```

The register addresses on BH (`tensix.h:162-165`):

| Register | Address |
|----------|---------|
| `RISC_DBG_CNTL_0` | `0xFFB12080` |
| `RISC_DBG_CNTL_1` | `0xFFB12084` |
| `RISC_DBG_STATUS_0` | `0xFFB12088` |
| `RISC_DBG_STATUS_1` | `0xFFB1208C` |

#### 3.2 Access Protocol

The write sequence from `ckernel_riscv_debug.h:109-121`:

```
1. Write value to RISC_DBG_CNTL_1
2. Write 0 to RISC_DBG_CNTL_0          (clear pulse bit)
3. Write command to RISC_DBG_CNTL_0     (set pulse bit + risc_sel + WR + addr)
4. Execute 3 NOPs                       (synchronizer crossing delay)
```

The read sequence from `ckernel_riscv_debug.h:125-142`:

```
1. Write 0 to RISC_DBG_CNTL_0          (clear pulse bit)
2. Write command to RISC_DBG_CNTL_0     (set pulse bit + risc_sel + RD + addr)
3. Execute 3 NOPs                       (synchronizer crossing delay)
4. Poll RISC_DBG_STATUS_0 until rd_valid (bit 30) is set
5. Read RISC_DBG_STATUS_1               (latched read data)
```

The 3-NOP delay is required because the pulse signal passes through a
synchronizer.  For host-side access via PCIe, the PCIe round-trip latency
(~1 us) exceeds this requirement, so no explicit delay is needed from the
debug agent.

---

### 4. The `rvdbg_cmd` Interface (Quasar Only)

The `rvdbg_cmd` enumeration is defined in
`tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/ckernel_riscv_debug.h`
lines 11-31:

```cpp
enum class rvdbg_cmd : std::uint32_t {
    PAUSE      = (1 << 0),     // Halt the RISC core
    STEP       = (1 << 1),     // Single-step one instruction
    CONTINUE   = (1 << 2),     // Resume execution
    RD_REG     = (1 << 3),     // Read integer register
    WR_REG     = (1 << 4),     // Write integer register
    RD_MEM     = (1 << 5),     // Read memory
    WR_MEM     = (1 << 6),     // Write memory
    FLUSH_REGS = (1 << 7),     // Flush register state
    FLUSH      = (1 << 8),     // Flush pipeline
    RD_CSR     = (1 << 9),     // Read CSR
    WR_CSR     = (1 << 10),    // Write CSR
    RD_FPREG   = (1 << 11),    // Read FP register
    WR_FPREG   = (1 << 12),    // Write FP register
    RD_VECREG  = (1 << 13),    // Read vector register
    WR_VECREG  = (1 << 14),    // Write vector register
    RD_UNIT    = (1 << 15),    // Read debug unit info
    DBG_MODE_BIT = (1U << 31)  // Debug mode indicator
};
```

Each command targets a specific RISC via `rvdbg_risc_sel`
(lines 33-39): TRISC0 = `0x00000000`, TRISC1 = `0x00020000`,
TRISC2 = `0x00040000`, TRISC3 = `0x00060000`.  Note that the
`rvdbg_risc_sel` enum covers only TRISC0-3.  At the raw register level,
`risc_sel` bits 19:17 encode 0=BRISC, 1-3=TRISC0-2, 4=NCRISC.

### 4.1 GDB RSP to rvdbg_cmd Mapping (Quasar)

| GDB RSP Packet | rvdbg_cmd | Notes |
|----------------|-----------|-------|
| `g` (read all regs) | `RD_REG` x 32 | Read x0-x31 sequentially |
| `G XX...` (write all regs) | `WR_REG` x 32 | Write x0-x31 |
| `m addr,len` | `RD_MEM` | Read memory at address |
| `M addr,len:data` | `WR_MEM` | Write memory at address |
| `c` (continue) | `CONTINUE` | Resume execution |
| `s` (step) | `STEP` | Single-step one instruction |
| `Ctrl-C` (break) | `PAUSE` | Halt execution |
| `p fpreg` | `RD_FPREG` | Quasar FP register access |
| `Z0,addr` (sw breakpoint) | `WR_MEM` + `ebreak` | Write ebreak at addr |
| `Z1,addr,len` (hw watchpoint) | Write `HW_WCHPT0-7` | 8 hardware watchpoints |
| `?` (halt reason) | Read `rvdbg_status` | paused/brkpt_hit/wchpt_hit/ebrk_hit |

---

### 5. Architecture-Specific Implementations

#### 5.1 Wormhole (WH): ebreak + Mailbox Only

**WH does NOT have RISC_DBG_CNTL registers.**  The only debug mechanisms
are Tensix breakpoints (at the coprocessor instruction level, not RISC-V
instruction level) and `ebreak`.

The `PAUSE()` macro in TT-Metal (see Ch 5, File 1) implements a
spin-on-mailbox pattern.  The actual implementation uses
`pause_msg->flags[internal_::get_hw_thread_idx()]` with a
`watcher_pause()` function -- not a direct macro with `bitmask`.

| Capability | WH (ebreak+mailbox) | BH (RISC_DBG_CNTL) | Quasar (rvdbg_cmd + DM) |
|-----------|---------------------|---------------------|-------------------------|
| Halt | Cooperative only | PAUSE via register | PAUSE command |
| Step | Not supported | STEP via register | STEP command |
| Register read | From L1 mailbox | RD_REG via register | RD_REG/RD_FPREG/RD_CSR |
| Memory read | Direct NOC read | RD_MEM via register | RD_MEM or System Bus |
| Breakpoint | Software ebreak | Hardware watchpoint | 8 HW watchpoints |
| Intrusiveness | Modifies kernel binary | Non-intrusive | Non-intrusive |

**Recommendation:** On WH, prefer Tier 1 emulator replay (File 01) over
on-hardware debug.  Reserve on-hardware WH debug for cases where emulator
replay diverges from hardware behavior.

#### 5.2 Blackhole (BH): RISC_DBG_CNTL Registers

BH has the full `RISC_DBG_CNTL_0/1` register pair at `0xFFB12080` and
`0xFFB12084` (tensix.h:162-163).  The register protocol described in
Section 3 works on BH.  However, the `rvdbg_cmd` / `riscv_dbg_wr` /
`riscv_dbg_rd` software abstraction from `ckernel_riscv_debug.h` must
be ported from Quasar -- it does not exist in the BH LLK tree.

#### 5.3 Quasar: Full Debug Module

Beyond `rvdbg_cmd`, Quasar provides a RISC-V Debug Specification v0.13
compliant Debug Module at APB base address `0x0300A000` (from
`overlay_reg_defines_debug.h:15`):

| Register | Offset | Address | Function |
|----------|--------|---------|----------|
| `DMCONTROL` | `0x40` | `0x0300A040` | Halt/resume/reset individual harts |
| `DMSTATUS` | `0x44` | `0x0300A044` | Hart state (halted/running) |
| `HARTINFO` | `0x48` | `0x0300A048` | Hart capabilities |
| `ABSTRACTS` | `0x58` | `0x0300A058` | Abstract command status |
| `COMMAND` | `0x5C` | `0x0300A05C` | Execute abstract command (reg access) |
| `PROGBUF` | `0x80` | `0x0300A080` | Program buffer for arbitrary execution |
| `SBCS` | `0xE0` | `0x0300A0E0` | System Bus Access control |
| `SBADDR0` | `0xE4` | `0x0300A0E4` | System Bus address |
| `SBDATA0` | `0xF0` | `0x0300A0F0` | System Bus data |

The Debug Module interface is compatible with OpenOCD's RISC-V target
driver, enabling standard GDB connectivity without custom agent software.

---

### 6. PROPOSED: BRISC-as-Debug-Agent (Deep Dive)

#### 6.1 Architecture

```
 Host GDB  <-- RSP over TCP -->  Host-side GDB Stub
                                      |
                                 NOC read/write (PCIe)
                                      |
                                 BRISC (debug agent FW)
                                      |
                            +----+----+----+----+
                            |    |    |    |    |
                          NCRISC T0   T1   T2  (user kernels)
                            |    |    |    |
                            +----+----+----+
                              L1 SRAM (shared)
```

BRISC has NOC access for host communication and can reach all TRISC debug
interfaces through the Tensix breakpoint API (`tensix_functions.h:606-687`)
on WH/BH and through RISC_DBG_CNTL on BH/Quasar.

#### 6.2 Command Protocol

```cpp
// PROPOSED: debug_agent_protocol.h
enum class DbgAgentCmd : uint32_t {
    IDLE          = 0x00,
    HALT          = 0x01,
    RESUME        = 0x02,
    SINGLE_STEP   = 0x03,
    READ_REG      = 0x04,
    WRITE_REG     = 0x05,
    READ_MEM      = 0x06,
    WRITE_MEM     = 0x07,
    READ_CSR      = 0x08,
    SET_BREAKPOINT   = 0x10,
    CLEAR_BREAKPOINT = 0x11,
    READ_STATUS   = 0x20,
    DETACH        = 0xFF,
};

struct DbgMailbox {
    volatile DbgAgentCmd    command;
    volatile uint32_t       trisc_index;
    volatile uint32_t       arg0;
    volatile uint32_t       arg1;
    volatile DbgAgentStatus status;
    volatile uint32_t       result;
    volatile uint32_t       halt_reason;
    volatile uint32_t       current_pc;
};
```

#### 6.3 Advantages Over Host-Driven Debug

| Advantage | Explanation |
|-----------|-------------|
| Lower latency | BRISC accesses debug registers locally (~10 ns) vs host via PCIe (~1 us) |
| Batch operations | Host sends "read all 32 GPRs" as one command; BRISC reads locally |
| Reduced PCIe traffic | One NOC round-trip per batch vs 6+ PCIe transactions per register read |

#### 6.4 Limitations

- **BH/Quasar only** for register read/write (WH lacks RISC_DBG_CNTL)
- **Occupies BRISC**: Cannot run data-movement kernel simultaneously
- **`rvdbg_risc_sel` covers TRISCs only**: BRISC/NCRISC targeting requires
  raw register writes with `risc_sel` bits 19:17 = 0 or 4

#### 6.5 Latency Characteristics

| Step | Latency |
|------|---------|
| Host PCIe write to L1 (command) | ~1 us |
| BRISC polling detection | 0-1 us |
| RISC_DBG_CNTL register access | ~10 ns (3 NOPs + synchronizer) |
| RISC_DBG_STATUS poll | ~10-50 ns |
| BRISC write result to L1 | ~10 ns |
| Host PCIe read from L1 (result) | ~1 us |
| **Total per command** | **~2-4 us** |

For GDB `g` command (33 registers): ~100 us total -- acceptable for
interactive debugging.

---

### 7. PROPOSED: Hardware Replay Terminal Session

```
$ tt-kernel-replay hw-replay matmul_core3_4.ttksnap \
    --device 0 --core 3,4 --risc trisc1 --exclusive

tt-kernel-replay: Architecture: WORMHOLE_B0 (matches device 0)
tt-kernel-replay: Exclusive access granted (PID 12345)
tt-kernel-replay: Loading L1 (487 KB) via PCIe ... [ok]
tt-kernel-replay: Installing ebreak trap handler (416 bytes at 0x1F800)
tt-kernel-replay: Entry breakpoint at run_kernel (0xB1A0)
tt-kernel-replay: Core (3,4) launched, hit entry breakpoint.
tt-kernel-replay: GDB RSP on port 1234

$ riscv-tt-elf-gdb trisc1.elf -ex "target remote :1234"
0x0000b1a0 in run_kernel (args=0x20000) at bmm_large_block.cpp:42
(gdb) break llk_math_matmul
(gdb) continue
Breakpoint 1, llk_math_matmul<...>() at llk_math_matmul.h:127
(gdb) monitor dest-dump 0 4
DEST row 0:  0x3F80 0x0000 0x0000 0x0000 ... (16 values)
DEST row 1:  0x0000 0x3F80 0x0000 0x0000 ...
```

Custom `monitor` commands bridge standard GDB to Tensix-specific state:

| Command | Description | Hardware API |
|---------|------------|-------------|
| `monitor dest-dump <row> <n>` | Read DEST array rows | `dbg_dump_array_rd_cmd(thread, DEST_ARRAY_ID, addr)` |
| `monitor srca-dump <row> <n>` | Read SRCA array | `dbg_dump_array_rd_cmd(thread, SRCA_ARRAY_ID, addr)` |
| `monitor cfg-dump <start> <n>` | Read HW config registers | Coprocessor debug bus |
| `monitor sem-dump` | All 16 semaphore values | L1 read at sem addresses |
| `monitor cb-status` | CB read/write pointers | L1 read at CB bases |

---

## Part B: Host Functional Model Replay

### 8. Architecture Overview

Host functional model replay takes a fundamentally different approach:
instead of executing the RISC-V binary, it recompiles the kernel source
code with a host compiler (g++ or clang++) and links it against stub
libraries that model Tensix hardware in software.

### 9. Existing Infrastructure: `trisc.cpp`

The TT-LLK test infrastructure already provides host-side kernel execution
via `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp`.  This
file (lines 36-77) provides:

1. **Host-side `main()`**: Entry point that calls `run_kernel()`
2. **Host-side regfile**: `std::fill(ckernel::regfile, ckernel::regfile + 64, 0)` (line 58)
3. **Config state reset**: `reset_cfg_state_id()`, `reset_dest_offset_id()` guarded by `#ifndef ARCH_QUASAR`
4. **Completion signaling**: `*mailbox = ckernel::KERNEL_COMPLETE` (line 76)

The host-side `ckernel::regfile` is declared in
`tt_metal/third_party/tt_llk/tests/helpers/include/ckernel_helper.h`
as a 64-entry array at `REGFILE_BASE`.  In the host test environment,
this resolves to a host-allocated memory region.

**Important:** The actual device firmware (`trisc.cc`) has a different
execution model than `trisc.cpp`.  The firmware reads `launch_msg`,
computes `kernel_lma` from `kernel_text_offset`, and calls via function
pointer: `reinterpret_cast<uint32_t (*)()>(kernel_lma)()`.  Runtime
args are accessed via `rta_l1_base` from the launch message, not passed
as a function argument.  The host model in `trisc.cpp` simplifies this
to `run_kernel(__runtime_args_start)`.  The replay host model should
mirror the `trisc.cpp` simplified path while being aware of this
discrepancy.

### 10. Host Model Alternatives

| Strategy | Description | GDB Support |
|----------|-------------|-------------|
| **A: trisc.cpp extension** | Extend existing LLK test harness to load .ttksnap | Full (host gdb) |
| **B: Standalone kernel runner** | New binary compiles kernel source with host compiler + stubs | Full (host gdb) |
| **C: LLK interpreter** | Parse and interpret LLK API calls from trace log | Trace-level only |

### 10.1 Scoring Matrix

| Criterion | Weight | A: trisc.cpp Extension | B: Standalone Runner | C: Interpreter |
|-----------|--------|----------------------|---------------------|----------------|
| Implementation effort | 4 | 4 (extend existing) | 3 (new tool) | 2 (parse+interpret) |
| GDB debugging fidelity | 5 | 5 (full source debug) | 5 (full source debug) | 1 (no stepping) |
| Snapshot integration | 4 | 3 (test harness format) | 5 (.ttksnap native) | 3 (trace format) |
| Architecture coverage | 3 | 4 (WH/BH via defines) | 4 (WH/BH/Quasar) | 5 (any) |
| Maintenance with LLK changes | 3 | 5 (same build system) | 3 (separate build) | 2 (parser updates) |
| **Weighted total** | | **93** | **89** | **55** |

> **PROPOSED: Recommendation.** Extend `trisc.cpp` (Strategy A) for
> Phase 3.  Build the standalone runner (Strategy B) in Phase 4 if the
> LLK test build system proves too coupled.

### 11. PROPOSED: Host-Model Build and Replay

#### 11.1 Host Compilation

```bash
# PROPOSED: Host compilation command
g++ -std=c++17 -g -O0 -fsanitize=address,undefined \
    -DARCH_BLACKHOLE -DCOMPILE_FOR_TRISC -DLLK_TRISC_MATH \
    -I${TT_METAL}/tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc \
    -I${TT_METAL}/tt_metal/hw/inc/internal/tt-1xx/blackhole \
    -I${TT_METAL}/tt_metal/hw/inc/api \
    -I${TT_METAL}/tt_metal/third_party/tt_llk/tests/helpers/include \
    -include host_replay_stubs.h \
    -o kernel_replay \
    extracted_kernel.cpp \
    host_replay_main.cpp
```

#### 11.2 Host Stub Implementations

```cpp
// PROPOSED: host_stubs.cpp
namespace ckernel {
    alignas(64) volatile uint32_t regfile_storage[64] = {};
    volatile uint32_t* regfile = regfile_storage;
    alignas(64) volatile uint32_t reg_base_storage[1024] = {};
    volatile uint32_t* reg_base = reg_base_storage;
    alignas(64) volatile uint32_t pc_buf_storage[16] = {};
    volatile uint32_t* pc_buf_base = pc_buf_storage;
    uint32_t cfg_state_id = 0;
    uint32_t dest_offset_id = 0;
    constexpr uint32_t KERNEL_COMPLETE = 0xFF;

    // tensix_sync() -- on host, a compiler barrier suffices
    inline void tensix_sync() { asm volatile("" ::: "memory"); }
    void reset_cfg_state_id() { cfg_state_id = 0; }
    void reset_dest_offset_id() { dest_offset_id = 0; }
}
```

#### 11.3 Host-Model Terminal Session

```
$ tt-kernel-replay host-model eltwise_snap.ttksnap --core trisc1 --gdb

tt-kernel-replay: Generating stubs (virtual_l1, tensix_stubs, cb_stubs)
tt-kernel-replay: Compiling with g++ -g -O0 -fsanitize=address,undefined ...
tt-kernel-replay: Launching under GDB ...

(gdb) break run_kernel
(gdb) run eltwise_snap.ttksnap
Breakpoint 1, run_kernel (args=0x7ffff6a20000) at eltwise_unary_sfpu.cpp:15
(gdb) break llk_math_eltwise_unary_sfpu_exponential
(gdb) continue
Breakpoint 2, llk_math_eltwise_unary_sfpu_exponential(dst_index=0)
(gdb) print ckernel::regfile[0]@64
$1 = {0, 0, 0, ...}
(gdb) step
90      float val = dest_read_float(dst_index, face, row);
```

#### 11.4 AddressSanitizer Catching L1 Buffer Overflow

```
$ tt-kernel-replay host-model matmul_snap.ttksnap --sanitize

==12345==ERROR: AddressSanitizer: heap-buffer-overflow at 0x7f0000100000
    #0 virtual_l1::read32(unsigned int) virtual_l1.cpp:42
    #1 llk_unpack_A(unsigned int, unsigned int) tensix_stubs.cpp:118
    #2 run_kernel(unsigned int*) bmm_large_block.cpp:67

tt-kernel-replay: L1 buffer overflow detected! Kernel read beyond L1 bounds.
```

Standard memory sanitizers catch L1 access violations that silently
corrupt data on hardware.

---

## Part C: Comparison of All Three Replay Targets

### 12. Three-Way Scoring Matrix

| Criterion | Weight | Emulator (File 01) | On-Hardware | Host Model |
|-----------|--------|-------------------|-------------|------------|
| Hardware fidelity | 4 | 3 (ISA only) | 5 (exact) | 2 (stubs) |
| GDB debugging UX | 5 | 4 (Spike GDB) | 3 (latency) | 5 (native GDB) |
| Setup complexity | 3 | 3 (Spike + plugin) | 2 (device + agent) | 5 (host compiler) |
| Hardware requirement | 3 | 5 (none) | 1 (exclusive device) | 5 (none) |
| Iteration speed | 4 | 3 (reload + restart) | 2 (device reload) | 5 (recompile + run) |
| SFPU accuracy | 3 | 2 (stubs) | 5 (exact) | 2 (stubs) |
| Multi-RISC support | 2 | 3 (multi-hart) | 3 (one-at-a-time) | 2 (single thread) |
| **Weighted total** | | **84** | **78** | **90** |

### 13. Recommended Usage by Bug Type

| Bug Type | Primary Target | Secondary Target |
|----------|---------------|-----------------|
| Wrong tile in output | Host model | Emulator |
| Incorrect loop count | Host model | Emulator |
| CB pointer corruption | Emulator | On-hardware |
| Semaphore deadlock | Emulator (5-RISC) | On-hardware |
| SFPU precision error | On-hardware | Host model (IEEE approx) |
| NOC data corruption | On-hardware | N/A |
| Config register bug | Emulator | On-hardware |

### 14. Comprehensive Fidelity Table

| State Category (Ch 2) | Emulator L0 | Emulator L2 | Host Model | HW (WH/BH) | HW (Quasar) |
|---|:-:|:-:|:-:|:-:|:-:|
| RISC-V scalar code | Exact | Exact | Exact (x86) | Exact | Exact |
| CB pointer management | No | Modeled | Modeled | Exact | Exact |
| FPU matmul | No | Approximate | Approximate | Exact | Exact |
| SFPU operations | No | Approximate | Approximate | Exact | Exact |
| BFP8/BFP4 rounding | No | Differs | Differs | Exact | Exact |
| NOC transactions | No | No | No | Real | Real |
| DEST/SRCA/SRCB arrays | No | Modeled | Modeled | Readable | Readable |
| Hardware watchpoints | No | No | No | 4/thread | 8/RISC |
| FP register access | Limited | Limited | N/A | No | Via RD_FPREG |

**Iteration speed:** Host model ~1 sec startup.  Emulator ~2-3 sec.
HW replay ~5 sec.  RTL simulator 60-600 sec startup.

---

## 15. Error Messages for Hardware and Host-Model Replay

| Condition | Message |
|-----------|---------|
| Arch mismatch | `ERROR: Snapshot=BLACKHOLE, Device 0=WORMHOLE_B0. Use matching device or: tt-kernel-replay emulate snap.ttksnap` |
| Device busy | `ERROR: Device 0 in use (PID 54321). Use --device <other>, wait, or replay in emulator/host-model mode.` |
| Compiler error | `ERROR: g++ failed: 'ckernel_ops.h' not found. Set TT_METAL_HOME or use --include-path.` |
| Trap handler overlap | `ERROR: Debug mailbox (0x1F800-0x1F9A0) overlaps kernel stack. Re-capture with smaller stack or use --no-trap-handler on Quasar.` |

---

## Key Source Files

| File | Path (relative to TT-Metal root) | Lines | Relevance |
|------|----------------------------------|-------|-----------|
| `tensix_functions.h` | `tt_metal/hw/inc/internal/tensix_functions.h` | 606-687 | WH/BH breakpoint API |
| `ckernel_riscv_debug.h` | `tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/ckernel_riscv_debug.h` | 11-31, 33-39, 76-100, 109-142 | `rvdbg_cmd`, `rvdbg_risc_sel`, RISC_DBG_CNTL protocol, `riscv_dbg_wr`/`riscv_dbg_rd` |
| `tensix.h` (BH) | `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h` | 162-165 | RISC_DBG_CNTL_0/1 and STATUS_0/1 register addresses |
| `overlay_reg_defines_debug.h` | `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/meta/registers/overlay_reg_defines_debug.h` | 15, 21 | Debug Module APB base, DMCONTROL |
| `t6_debug_map.h` | `tt_metal/hw/inc/internal/tt-2xx/quasar/t6_debug_map.h` | 1062-1065 | `t6_debug_regs_t` struct |
| `trisc.cc` | `tt_metal/hw/firmware/src/tt-1xx/trisc.cc` | 58, 113-117, 214-215 | reg_base, regfile init loop, kernel_lma invocation |
| `trisc.cpp` | `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp` | 36-77 | Host-side TRISC entry point |
| `dev_msgs.h` | `tt_metal/hw/inc/hostdev/dev_msgs.h` | 179 | `go_msg_t`, `launch_msg_t`, sync protocol |
| `rtoptions.cpp` | `tt_metal/llrt/rtoptions.cpp` | 1110-1122 | `TT_METAL_RISCV_DEBUG_INFO` |
| `dump.h` | `tt_metal/third_party/tt_llk/tests/helpers/include/dump.h` | -- | `llk::debug::tensix_dump` state dump protocol |

---

## Key Takeaways

- **On-hardware replay provides exact fidelity** but requires different
  implementation strategies per architecture: raw RISC_DBG_CNTL registers
  on BH (must port `rvdbg_cmd` API from Quasar), full `rvdbg_cmd` + Debug
  Module on Quasar, and only ebreak+mailbox on WH.

- **The host functional model scores highest overall** (90/130) for
  debugging compute kernel logic.  It provides native GDB, the fastest
  iteration, zero hardware dependency, and builds on existing `trisc.cpp`
  infrastructure.  AddressSanitizer/UBSan integration catches L1 access
  violations invisible on hardware.

- **WH hardware debugging is limited**: without `RISC_DBG_CNTL` registers,
  WH cannot support non-intrusive halt/step/register-read.  Prefer Tier 1
  emulator or host model replay on WH.

- **The BRISC-as-debug-agent concept reduces PCIe round-trips** by
  batching debug register operations on-device (~2-4 us per command vs
  ~10 us per PCIe round-trip), but requires custom firmware development.

- **Architecture-dependent debug paths require a clean `DebugTransport`
  abstraction**: the same GDB RSP commands map to different primitives
  on WH (ebreak+mailbox) vs BH (RISC_DBG_CNTL raw registers) vs Quasar
  (rvdbg_cmd API or Debug Module APB).
