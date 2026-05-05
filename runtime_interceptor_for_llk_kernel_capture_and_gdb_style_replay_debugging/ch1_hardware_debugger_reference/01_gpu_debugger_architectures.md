# GPU Debugger Architectures

This file dissects the layered architecture of cuda-gdb and three peer debuggers (rocgdb, Intel GT Debugger, Arm DS), extracts the universal patterns they share, and identifies which of those patterns Tenstorrent's Tensix architecture provides or lacks today. The analysis is oriented toward a specific question: *what would a Tenstorrent debugger need to replicate, and where does the "capture and replay" paradigm offer a shorter path than replicating the full live-debug stack?*

---

## 1. cuda-gdb: The Four-Layer Reference Architecture

cuda-gdb is the most mature accelerator debugger in production and serves as our reference. Its architecture is organized into four layers, each adding a level of abstraction between the developer and the silicon.

### 1.1 Layer 1 -- SM Debug Hardware

At the bottom of the stack, each NVIDIA Streaming Multiprocessor (SM) contains dedicated debug circuitry:

- **Trap handler**: a firmware routine invoked when an SM warp encounters a debug exception (software breakpoint via `BPT.TRAP` instruction, hardware breakpoint match, or single-step completion). The trap handler saves warp context and signals the driver.
- **Warp-level halt/resume**: the SM can halt individual warps (groups of 32 threads executing in SIMT lockstep) without halting the entire GPU. This is fundamental to cuda-gdb's ability to freeze a specific thread while the rest of the kernel continues.
- **Debug registers**: per-SM registers for breakpoint addresses, single-step enable, exception cause, and warp status. These are accessible from the driver via the GPU's debug interface (typically memory-mapped through the PCIe BAR).
- **Memory access ports**: the debug unit can read and write GPU global memory, shared memory, local memory, and register files on behalf of the host debugger, even while the kernel is paused.
- **Single-step circuitry**: a "step count" register allows the debug unit to advance a warp by exactly $n$ instructions (typically $n = 1$) before re-trapping.

### 1.2 Layer 2 -- CUDA Driver Debug Interface

The CUDA driver (`libcuda.so`) implements debug API requests by communicating with the GPU through memory-mapped I/O (MMIO) registers:

- **Debug mode transition**: writing to GPU control registers puts the device in a state where SM debug units are active and warps can be individually halted.
- **Breakpoint installation**: software breakpoints replace original instructions with `BPT.TRAP`; the driver maintains a shadow copy for single-step-past-breakpoint logic. Hardware breakpoints use address comparator registers (typically 4-8 per SM).
- **Memory access routing**: the driver resolves virtual addresses to physical addresses using the GPU page table, then reads/writes through BAR windows or DMA side-channels.
- **Warp scheduling control**: issuing halt/resume commands through the SM's warp scheduler control registers.

### 1.3 Layer 3 -- CUDA Debug API (libcudbg)

The `libcudbg` library provides a C API that translates high-level debug operations into driver-level commands:

| API Function | Purpose | Hardware Mechanism |
|---|---|---|
| `cudbgAttach()` | Attach debugger to a CUDA context | Enables debug mode; installs trap handlers on all SMs |
| `cudbgDetach()` | Detach from CUDA context | Disables debug mode; removes trap handlers |
| `cudbgSetBreakpoint(addr)` | Set breakpoint at instruction address | Writes `BPT.TRAP` into SM instruction cache or sets HW comparator |
| `cudbgSingleStep(dev, sm, warp)` | Single-step one instruction | Sets step-count register to 1; resumes warp; re-halts |
| `cudbgReadRegister(dev, sm, warp, lane, reg)` | Read per-lane register | Side-channel read from SM register file while warp is halted |
| `cudbgWriteRegister(...)` | Write per-lane register | Side-channel write to SM register file |
| `cudbgReadMemory(dev, sm, warp, addr, len)` | Read device memory | Translates virtual address via GPU page table; MMIO/DMA read |
| `cudbgWriteMemory(...)` | Write device memory | Same path, write direction |
| `cudbgResumeWarps(mask)` | Resume halted warps | Clears halt flag on selected warps |

The addressing model is hierarchical: every operation targets a specific `(device, SM, warp, lane)` tuple. This is the "focus" concept (Section 2 below).

### 1.4 Layer 4 -- GDB Frontend

The developer interacts through standard GDB commands augmented with CUDA-specific extensions:

- `cuda device 0 sm 3 warp 7 lane 0` -- set focus
- `info cuda kernels` -- list active kernels
- Standard `break`, `step`, `next`, `print`, `backtrace` commands work on device code

The key architectural insight is that **each layer translates a small, well-defined set of abstract operations** (halt, resume, step, read register, read memory, set breakpoint) into the next layer's primitives. The interface between layers is narrow and stable, even as GPU hardware evolves.

### Layer Diagram

```
+------------------------------------------------------------------+
|  Layer 4: GDB Frontend (host process)                            |
|  Standard GDB CLI / MI + CUDA extensions                         |
|  "Focus" model: select GPU -> SM -> warp -> lane                 |
+------------------------------------------------------------------+
                          |
                          | GDB -> libcudbg API calls
                          v
+------------------------------------------------------------------+
|  Layer 3: CUDA Debug API (libcudbg.so)                           |
|  cudbgAttach / cudbgSetBreakpoint / cudbgSingleStep              |
|  cudbgReadRegister / cudbgReadMemory                             |
|  Hierarchical (device, SM, warp, lane) addressing                |
+------------------------------------------------------------------+
                          |
                          | Internal driver IOCTLs / register writes
                          v
+------------------------------------------------------------------+
|  Layer 2: CUDA Driver Debug Interface (libcuda.so)               |
|  Trap instruction insertion / BPT.TRAP                           |
|  Warp scheduler control via PCIe BAR registers                   |
|  Memory space routing (shared / local / global)                  |
+------------------------------------------------------------------+
                          |
                          | PCIe MMIO to GPU registers
                          v
+------------------------------------------------------------------+
|  Layer 1: SM Hardware Debug Unit (silicon)                       |
|  Per-warp halt/resume control bits                               |
|  Trap handler entry on BPT.TRAP                                  |
|  Register file debug read/write paths                            |
|  Single-step mode bit                                            |
+------------------------------------------------------------------+
```

---

## 2. The Focus Model: Selecting GPU, SM, Warp, and Lane

cuda-gdb introduces a hierarchical "focus" concept for navigating the massively parallel execution state:

$$\text{Focus} = (\text{device}, \text{SM}, \text{warp}, \text{lane})$$

Each level narrows the scope:

1. **Device** -- which GPU (multi-GPU systems)
2. **SM** -- which Streaming Multiprocessor (up to 144 on H100)
3. **Warp** -- which warp within the SM (up to 64 resident warps)
4. **Lane** -- which thread within the warp (0-31)

Breakpoints are set at a kernel address and hit by any warp that reaches that PC. The debugger then freezes the hitting warp and allows inspection. Other warps may continue or be frozen, depending on configuration.

> **Pattern extracted**: A hardware debugger for a massively parallel processor requires a hierarchical focus model that narrows from device down to the smallest schedulable execution unit.

The equivalent hierarchy for Tenstorrent:

$$\text{Focus}_{\text{Tensix}} = (\text{device}, \text{core}_{(x,y)}, \text{RISC} \in \{\text{BRISC}, \text{NCRISC}, \text{TRISC0}, \text{TRISC1}, \text{TRISC2}[, \text{TRISC3}]\})$$

> **Architecture note:** Wormhole and Blackhole have 5 RISC-V processors per core (BRISC, NCRISC, TRISC0-2). Quasar extends this to 6 processors by adding TRISC3 (`NUM_TRISC_CORES=4`). The focus model must be architecture-aware: the set of valid RISC targets varies by chip generation.

| cuda-gdb Level | Tenstorrent Equivalent |
|---|---|
| Device | Device (chip) |
| SM | Tensix core at $(x, y)$ coordinates |
| Warp | RISC-V processor (BRISC, NCRISC, TRISC0-2 on WH/BH; TRISC0-3 on Quasar) |
| Lane | N/A (Tensix RISCs are scalar; SFPI vector ops use a vector register file accessed via `RD_VECREG`/`WR_VECREG`) |

This mapping is significant: GPU debuggers must handle SIMT divergence (lanes within a warp may be at different PCs), while Tenstorrent debuggers deal with five independent scalar processors per core (six on Quasar, which adds TRISC3). The Tenstorrent model is architecturally simpler in the SIMT dimension but more complex in the inter-processor coordination dimension.

---

## 3. Comparative Analysis

### 3.1 rocgdb (AMD ROCm Debugger)

rocgdb debugs kernels on AMD GPUs (CDNA/RDNA architectures). Its architecture parallels cuda-gdb but with key AMD-specific mechanisms:

**Wavefront granularity.** AMD GPUs execute in wavefronts (32 threads natively on RDNA, with 32 or 64 supported; 64 threads on CDNA). The debug granularity is the wavefront, analogous to NVIDIA's warp.

**CWSR (Compute Wave Save/Restore).** When a wavefront is preempted or hits a trap, AMD hardware saves the entire wavefront state (registers, EXEC mask, VCC, program counter, SGPR/VGPR contents) to a system memory buffer. This is a hardware-assisted form of state serialization.

> CWSR is significant for our purposes because it demonstrates that **serializing execution state to memory is a viable debugging primitive on production accelerator hardware**. The Tenstorrent capture-and-replay approach generalizes this concept to the entire kernel invocation.

**AQLPROFILE trap handlers.** For software breakpoints, rocgdb uses `s_trap` instructions that invoke a kernel-mode trap handler. The handler saves context to the CWSR save area and signals the host via a shared-memory doorbell.

**Debug API.** AMD's `amd-dbgapi` library provides the equivalent of NVIDIA's `libcudbg`: `amd_dbgapi_process_attach()`, `amd_dbgapi_wave_stop()`, `amd_dbgapi_wave_resume()`, `amd_dbgapi_read_register()`, `amd_dbgapi_read_memory()`.

### 3.2 Intel GT Debugger (Intel GPU Debug)

Intel's GPU debugger for Xe/Arc architectures uses a distinctive **System Routine (SIP) injection** model:

**EU Debug Mode.** Intel GPUs have a per-Execution-Unit debug mode. When enabled, EUs can be halted, single-stepped, and inspected through MMIO registers.

**System Routine (SIP).** When an EU hits an exception, it vectors to a pre-loaded SIP -- a small firmware routine that:

1. Saves EU thread state (GRF, ARF, flags, IP) to a designated surface in GPU memory.
2. Signals the host via a shared-memory notification mechanism.
3. Waits for the host to modify saved state and issue a resume.

This is a **software-cooperative** debug mechanism: the hardware provides a minimal halt mechanism, and software handles the complexity of state capture.

> Intel's SIP model is the closest architectural precedent for a potential Tenstorrent approach where BRISC acts as a debug agent controlling the TRISCs. See Chapter 5, `02_tensix_hardware_debug_capabilities.md` and Chapter 7, `02_on_hardware_and_host_model_replay.md`.

**Thread-level granularity.** Intel GPUs can halt individual hardware threads within an EU (each EU has 7-8 threads), providing finer-grained control than NVIDIA's warp-level halt.

### 3.3 Arm DS (Development Studio) with CoreSight

Arm DS debugs Arm cores using the CoreSight debug infrastructure -- the standard reference for JTAG-based RISC core debugging:

**JTAG/SWD halt-mode debugging.** The Arm Debug Architecture supports halting a core via JTAG/SWD by writing to the core's Debug Halt Control and Status Register (DHCSR). Once halted, all registers and memory are accessible through the Debug Access Port (DAP).

**Non-invasive trace.** CoreSight supports non-invasive debug via ETM (Embedded Trace Macrocell), which emits execution traces without halting the core -- a "flight recorder" capability for post-mortem analysis.

**Cross Trigger Interface (CTI).** CoreSight's CTI enables synchronized halt across multiple cores: when one core hits a breakpoint, a CTI trigger can simultaneously halt all other cores. This is particularly relevant for Tenstorrent's multi-RISC-per-Tensix architecture.

**No software cooperation required.** Unlike GPU debuggers, Arm halt-mode debugging requires zero cooperation from the running software. The JTAG interface can halt a core at any point, even in interrupt handlers or firmware.

The Arm model is particularly relevant because Tensix cores contain RISC-V processors, and the RISC-V Debug Specification (v0.13) was designed with CoreSight as a reference. The Quasar architecture's Debug Module (`TT_DEBUG_MODULE_APB_DMCONTROL_reg_t` with `haltreq`, `resumereq`, `setresethaltreq`) directly implements this standard.

### 3.4 Structured Comparison

| Dimension | cuda-gdb (NVIDIA) | rocgdb (AMD) | Intel GT Debugger | Arm DS |
|---|---|---|---|---|
| **Execution unit** | Warp (32 threads) | Wavefront (32 or 64 threads) | HW thread (8-16 lanes) | Core (1 thread per debug context) |
| **Debug interface** | MMIO via driver | IOCTL via KFD | MMIO via L0 driver | JTAG / SWD (external) |
| **Breakpoint mechanism** | `BPT.TRAP` instruction patch | `s_trap` + trap handler | EU exception + SIP | Hardware BPU (no code modification) |
| **Halt granularity** | Per-warp | Per-wavefront | Per-thread within EU | Per-core |
| **State save method** | Debug unit reads register file | Trap handler saves to CWSR buffer | SIP saves to surface | JTAG reads directly from core |
| **Software cooperation** | Minimal (driver sets debug mode) | Required (trap handler loaded) | Required (SIP must be present) | None (pure external control) |
| **Hardware watchpoints** | Not directly exposed | 4 per CU | Limited (varies by gen) | 2-8 DWT units |
| **Multi-unit sync** | Device-wide suspend/resume | Process-level stop | Attention signal bitmask | CTI (Cross Trigger Interface) |
| **Host connection** | PCIe (driver in-band) | PCIe (driver in-band) | PCIe (driver in-band) | JTAG/SWD (out-of-band) |

| Halt Aspect | cuda-gdb | rocgdb | Intel GT | Arm DS |
|---|---|---|---|---|
| **Halt type** | Cooperative (trap) | Cooperative (trap handler) | Hybrid (HW debug mode + SIP) | Preemptive (external halt) |
| **Can halt hung code** | No | No | No | Yes |
| **Impact on other units** | Per-warp; others continue | Per-wave; others continue | Per-thread or per-EU | Optional (CTI can sync) |

---

## 4. Common Patterns Extracted

Across all four debuggers, four universal patterns emerge:

### Pattern 1: Hardware Debug Unit with Halt/Resume/Step

Every debugger relies on a hardware mechanism to **halt** execution at a specific point, **resume** execution, and **single-step** by one instruction. This requires either dedicated debug circuitry or a cooperative software mechanism (trap handler, SIP injection) that achieves the same effect.

### Pattern 2: Driver-Level Debug API

A host-side library abstracts the hardware debug registers into structured operations: read/write registers by name, read/write memory by address, set/clear breakpoints, query execution status. This API handles address translation, breakpoint shadow management, multi-unit coordination, and hardware-revision abstraction.

- cuda-gdb: `libcudbg` (`cudbgAttach`, `cudbgSetBreakpoint`, ...)
- rocgdb: `amd-dbgapi` (`amd_dbgapi_process_attach`, `amd_dbgapi_wave_stop`, ...)
- Intel GT: Level Zero Debug API (`zetDebugAttach`, `zetDebugReadRegisters`, ...)
- Arm DS: RDDI/CADI API

### Pattern 3: Host-Side GDB Integration Layer

All four present their capabilities through a GDB-compatible interface, typically via the GDB Remote Serial Protocol (RSP):

| RSP Packet | Purpose |
|---|---|
| `g` / `G` | Read / Write all general registers |
| `p` / `P` | Read / Write a specific register |
| `m` / `M` | Read / Write memory |
| `c` | Continue execution |
| `s` | Single-step |
| `Z0` / `z0` | Set / Remove software breakpoint |
| `Z1` / `z1` | Set / Remove hardware breakpoint |
| `Z2` / `z2` | Set / Remove write watchpoint |
| `?` | Query halt reason |
| `H` | Set thread for subsequent operations |

A GDB stub for Tenstorrent RISC-V cores would translate these standard RSP packets into Tensix-specific operations. On Quasar, this maps naturally to the `rvdbg_cmd` enumeration (see Section 6.1).

### Pattern 4: Cooperative or Preemptive Halt Semantics

- **Preemptive halt** (Arm CoreSight, Quasar Debug Module): an external signal forces the core to halt regardless of what it is executing. Most reliable but requires dedicated hardware.
- **Cooperative halt** (cuda-gdb `BPT.TRAP`, rocgdb `s_trap`, Intel SIP, TT-Metal `ebreak`): the core encounters a trap instruction or enters a trap handler, which then cooperatively halts. Requires software support but works with simpler hardware.

---

## 5. What Tenstorrent Hardware Provides Today

### 5.1 The Quasar Debug Register Interface

The Quasar debug register interface (`ckernel_riscv_debug.h`) is worth examining closely because it provides nearly all the primitives needed for a GDB stub. The `rvdbg_cmd` enumeration:

```cpp
enum class rvdbg_cmd : std::uint32_t {
    PAUSE      = (1 << 0),   // Halt the target RISC-V core
    STEP       = (1 << 1),   // Single-step one instruction
    CONTINUE   = (1 << 2),   // Resume execution
    RD_REG     = (1 << 3),   // Read integer register
    WR_REG     = (1 << 4),   // Write integer register
    RD_MEM     = (1 << 5),   // Read memory
    WR_MEM     = (1 << 6),   // Write memory
    FLUSH_REGS = (1 << 7),   // Flush register state
    FLUSH      = (1 << 8),   // Flush pipeline
    RD_CSR     = (1 << 9),   // Read Control/Status Register
    WR_CSR     = (1 << 10),  // Write CSR
    RD_FPREG   = (1 << 11),  // Read floating-point register
    WR_FPREG   = (1 << 12),  // Write floating-point register
    RD_VECREG  = (1 << 13),  // Read vector register
    WR_VECREG  = (1 << 14),  // Write vector register
    RD_UNIT    = (1 << 15),  // Read unit-specific data
};
```

The interface operates through a register-mapped protocol:

- **RISC_DBG_CNTL_0**: 32-bit control register with fields for `pulse` (triggers access), `risc_sel` (3-bit field selecting the target processor), `reg_wr` (0=read, 1=write), and `reg_addr` (11-bit register address). The source comment in `ckernel_riscv_debug.h` documents `risc_sel` as "0 for BRISC, 1-3 for TRISCs 0-2, 4 for NCRISC."
- **RISC_DBG_CNTL_1**: 32-bit write data register.
- **RISC_DBG_STATUS_0**: mirrors control fields plus a `rd_valid` flag indicating read completion.
- **RISC_DBG_STATUS_1**: holds the read data value.

> **Note on `rvdbg_risc_sel` vs `RISC_DBG_CNTL_0::risc_sel` encoding conflict.** The `rvdbg_risc_sel` enum shown above encodes only the four compute processors: TRISC0=0x00000000 (risc_sel=0), TRISC1=0x00020000 (risc_sel=1), TRISC2=0x00040000 (risc_sel=2), TRISC3=0x00060000 (risc_sel=3). It contains no entries for BRISC or NCRISC. Meanwhile, the `RISC_DBG_CNTL_0` comment assigns risc_sel=0 to BRISC and risc_sel=4 to NCRISC -- meaning the `rvdbg_risc_sel::TRISC0` bit pattern occupies the same slot the comment assigns to BRISC. The `rvdbg_risc_sel` enum covers only the compute processors in Quasar's NEO model (which has four TRISCs rather than three). The `RISC_DBG_CNTL_0` comment describing "0=BRISC...4=NCRISC" may be inherited from a different access mode or architecture revision where the same register field addresses all five processors. An implementer building a GDB stub to target BRISC or NCRISC via this interface would find no `rvdbg_risc_sel` value for them; **targeting BRISC/NCRISC via the `rvdbg_risc_sel` path is an open question requiring hardware validation**.

The `rvdbg_status` union provides halt-reason reporting:

```cpp
union rvdbg_status {
    struct {
        unsigned paused    : 1;  // Core is paused
        unsigned brkpt_hit : 1;  // Breakpoint was hit
        unsigned wchpt_hit : 1;  // Watchpoint was hit
        unsigned ebrk_hit  : 1;  // ebreak instruction was hit
    };
    std::uint32_t val;
};
```

Hardware watchpoints are available via `HW_WCHPT0` through `HW_WCHPT7` (8 per RISC, since they are set per-processor via `riscv_dbg_wr(risc_sel, ...)`).

The `rvdbg_cmd` commands map directly to GDB RSP operations:

| GDB RSP Packet | `rvdbg_cmd` Command | Description |
|---|---|---|
| `c` (continue) | `CONTINUE` | Resume execution |
| `s` (single-step) | `STEP` | Execute one instruction, re-halt |
| `g` (read registers) | `RD_REG` (iterated) | Read general-purpose register by index |
| `G` (write registers) | `WR_REG` (iterated) | Write general-purpose register by index |
| `m` (read memory) | `RD_MEM` | Read memory at address |
| `M` (write memory) | `WR_MEM` | Write memory at address |
| `Z2`/`Z3` (watchpoint) | `HW_WCHPT0`-`HW_WCHPT7` | Set hardware watchpoint (8 available) |
| Custom | `RD_CSR` / `WR_CSR` | Read/write Control and Status Registers |
| Custom | `RD_FPREG` / `WR_FPREG` | Read/write floating-point registers |
| Custom | `RD_VECREG` / `WR_VECREG` | Read/write vector (SFPI) registers |

Quasar also provides a RISC-V Debug Module with `dmcontrol` registers (`haltreq`, `resumereq`, `setresethaltreq`, `hartsello` for hart selection) and `dmstatus` registers (`anyhalted`, `allhalted`, `anyrunning`, `allrunning`, `impebreak`). This is a compliant RISC-V Debug Specification v0.13 interface.

> **Critical open question:** Whether these registers are accessible from the host via PCIe (through L1 memory-mapped writes) or only from on-chip (another RISC-V core on the same Tensix) is an open question requiring hardware validation. See Chapter 8, `03_open_questions_and_validation_experiments.md`.

### 5.2 The Wormhole/Blackhole Breakpoint Infrastructure

On Wormhole and Blackhole, the breakpoint API in `tensix_functions.h` provides a different but complementary set of capabilities:

```cpp
// Set a hardware breakpoint on a coprocessor thread at a specific PC
breakpoint_set(uint thread, uint bkpt_index, bool pc_valid, uint pc);
// Resume execution after breakpoint
breakpoint_resume_execution(uint thread);
// Read breakpoint status
breakpoint_status(uint thread, uint bkpt_index);
// Read data captured at breakpoint
breakpoint_data();
```

These operate on the **coprocessor threads** (the Tensix tile engine threads), not the RISC-V cores themselves. They are accessed via `RISCV_DEBUG_REG_BREAKPOINT_CTRL`. The `#ifndef ARCH_QUASAR` guard indicates these APIs exist only for Wormhole/Blackhole.

Conditional breakpoints are supported:
- `breakpoint_set_condition_op(thread, bkpt_index, opcode)` -- break on specific Tensix opcode
- `breakpoint_set_condition_loop(thread, bkpt_index, loop)` -- break on loop iteration count
- `breakpoint_set_condition_other_thread(thread, bkpt_index)` -- break when another thread hits a breakpoint (analogous to Arm's CTI)

### 5.3 Debug Array and Coprocessor Bus Access

Beyond the breakpoint infrastructure, Tenstorrent provides inspection capabilities through the debug array and coprocessor debug bus (via `ckernel_debug.h`):

- **Debug array read**: `dbg_dump_array_enable()` followed by `dbg_dump_array_rd_cmd(thread, array_id, addr)` (from `tensix_functions.h`) reads from SRCA (0x0), SRCB (0x1), DEST (0x2), and MAX_EXP (0x3) register files. This provides visibility into the tile math engine's register state. Note that `ckernel_debug.h` provides the higher-level `dbg_get_array_row(array_id, row_addr, rd_data)`, which handles the SRCA indirection (SRCA cannot be read directly; the function copies the SRCA row into DEST via `TT_MOVDBGA2D`, saves and restores the original DEST contents via SFPU LREG3, then reads from DEST).
- **Config register dump**: `dbg_read_cfgreg(cfgreg_id, addr)` reads per-thread config registers (THREAD_0_CFG through THREAD_2_CFG) and hardware config state (HW_CFG_0, HW_CFG_1 with `HW_CFG_SIZE` = 187 registers).
- **Debug bus**: `dbg_bus_cntl_t` enables monitoring instruction issue via a daisy-chained debug bus.

These operate at the **coprocessor level** (the Tensix tile engine), complementing the RISC-V debug register interface by providing visibility into the compute datapath.

### 5.4 Pattern Coverage Summary

| Pattern | Wormhole/Blackhole | Quasar | Gap |
|---|---|---|---|
| Hardware halt/resume/step | Coprocessor breakpoints via `RISCV_DEBUG_REG_BREAKPOINT_CTRL` | Full: `rvdbg_cmd` PAUSE/STEP/CONTINUE + DM `haltreq`/`resumereq` | No RISC-V core halt on WH/BH; Quasar halt is core-on-core (host accessibility open question) |
| Register read/write | Debug array for SRCA/SRCB/DEST; config register dump | Full: `RD_REG`/`WR_REG`, `RD_FPREG`/`WR_FPREG`, `RD_VECREG`/`WR_VECREG`, `RD_CSR`/`WR_CSR` | WH/BH: no general-purpose RISC-V register read |
| Memory read/write | L1 via NOC from host; JTAG `read32`/`write32` | Same + `rvdbg_cmd::RD_MEM`/`WR_MEM` | Works, but concurrent L1 access while kernel runs is unsafe |
| Driver-level debug API | **None** | **None** | Must be built |
| GDB RSP stub | **None** | **None** | Must be built; `riscv-tt-elf-gdb` exists but has no target |
| Trap handler with state capture | `ebreak` triggers exception; default handler hangs | Same + `ebrk_hit` flag in `rvdbg_status` | No state-capturing trap handler; `ebreak` is currently fatal |
| Cooperative halt | `PAUSE()` macro (spin-wait on Watcher mailbox) | Same | Functional but no register/memory state capture on halt |

---

## 6. Where the Live-Debug Model Breaks Down for Tenstorrent

Despite the promising hardware debug primitives described above, building a full cuda-gdb-equivalent live debugger for Tenstorrent faces fundamental challenges:

### 6.1 The Multi-RISC Coordination Problem

A Tensix core runs five independent RISC-V processors (BRISC, NCRISC, TRISC0-2) on Wormhole/Blackhole, or six on Quasar (BRISC, NCRISC, TRISC0-3, where `NUM_TRISC_CORES=4`). These processors coordinate through shared L1 SRAM, circular buffer pointers, and hardware semaphores. Halting one RISC (e.g., TRISC1 for math debugging) while the others continue running creates an inconsistent state:

- BRISC/NCRISC continue moving data into circular buffers, potentially overwriting data the debugger is inspecting.
- TRISC0 (unpack) may signal TRISC1 (math) via semaphore that new data is ready; if TRISC1 is halted, the semaphore blocks TRISC0, which blocks BRISC, cascading into a deadlock.
- The dispatch system may time out waiting for kernel completion, killing the entire device.

cuda-gdb can halt individual warps because warps are *independent* -- they share an SM but do not synchronize at the hardware level the way Tensix RISCs do. The Tensix coordination model makes surgical halting fundamentally harder.

### 6.2 The Dispatch Timeout Problem

TT-Metal's dispatch infrastructure expects kernels to complete within a timeout. When a core is halted for debugging, the dispatch system (specifically `cq_prefetch.cpp` / `cq_dispatch.cpp`) may interpret the stall as a hang and trigger a device reset. The debugger must either:

1. Disable dispatch timeouts (risking actual hangs going undetected), or
2. Inform the dispatch system that a debug session is in progress (requiring dispatch pipeline modifications).

Neither NVIDIA nor AMD faces this problem because their drivers are debug-aware -- the debug API callback informs the driver that a debug session is active.

### 6.3 The Multi-Core Cascade

Tenstorrent programs typically span dozens or hundreds of Tensix cores connected by NOC. Halting one core may stall NOC transactions that other cores depend on, causing a cascade of timeouts across the mesh. cuda-gdb avoids this because GPU cores do not communicate directly during kernel execution (inter-SM communication goes through global memory with explicit synchronization).

### 6.4 The Missing Software Stack

Even if the hardware primitives are sufficient, building the four-layer debug stack from scratch requires approximately 12-18 person-weeks for a minimal implementation, with significant open questions about whether halting actually works reliably in practice. This motivates the capture-and-replay alternative explored in the remainder of this guide.

---

## 7. The Debugging Capability Spectrum

The four debuggers span a spectrum from full hardware debug to fully software-mediated approaches. This spectrum is useful for positioning the Tenstorrent design options:

```
Full Hardware Debug          Hybrid                         Full Software
(Arm DS / JTAG)       (cuda-gdb / rocgdb)              (Capture & Replay)
     |                       |                                |
     |  JTAG halt,           |  HW debug unit + SW trap       |  No live attach;
     |  register scan,       |  handler, driver API,          |  serialize all state,
     |  memory read          |  host GDB integration          |  replay in emulator
     |  via debug probe      |                                |  or host model
     |                       |                                |
     v                       v                                v
  Highest fidelity      Production standard             No hardware needed
  Slowest iteration     Moderate overhead               Fastest iteration
  Requires probe        Requires driver work            Requires capture infra
```

Tenstorrent's hardware sits at different points depending on the architecture:
- **Quasar**: capable of reaching "Hybrid" given the RISC-V Debug Module and `rvdbg_cmd` interface.
- **Wormhole/Blackhole**: breakpoint hardware exists but lacks RISC-V register read/write granularity for full hybrid debugging; better suited for cooperative or capture-and-replay approaches.
- **All architectures**: "Full Software" (capture-and-replay) is available regardless of hardware generation, making it the most broadly applicable strategy.

---

## Key Takeaways

- **All production accelerator debuggers share four patterns**: (1) a hardware halt/resume/step mechanism, (2) a structured driver-level debug API, (3) GDB integration via RSP, and (4) cooperative or preemptive halt semantics. Tenstorrent hardware provides patterns 1 and 4 in partial form but lacks patterns 2 and 3 entirely.

- **The live-debug model faces fundamental obstacles on Tenstorrent** due to multi-RISC coordination (halting one of the 5 or 6 tightly-coupled RISCs cascades into deadlocks), dispatch timeout interference, and multi-core NOC dependencies -- challenges that GPU debuggers avoid because their execution units are more independent.

- **Quasar's `rvdbg_cmd` interface provides the complete set of GDB RSP primitives** (PAUSE, STEP, CONTINUE, RD_REG, WR_REG, RD_MEM, WR_MEM, plus 8 hardware watchpoints), making a traditional hardware debugger feasible on Quasar if the registers prove host-accessible via PCIe. Wormhole/Blackhole breakpoint APIs target coprocessor threads rather than RISC-V cores and are guarded by `#ifndef ARCH_QUASAR`.

- **AMD's CWSR mechanism and Intel's SIP injection** validate that serializing execution state to memory is a proven production technique for accelerator debugging, providing precedent for the capture-and-replay approach developed in the rest of this guide.

---

**Next:** [`02_lessons_and_requirements_for_tenstorrent.md`](./02_lessons_and_requirements_for_tenstorrent.md)
