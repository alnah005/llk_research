# Chapter 5: Debug Infrastructure, Hardware Capabilities, and Emulation Feasibility

## Purpose

This chapter surveys all existing TT-Metal debug tools, investigates the
hardware-level debug features on Tensix RISC-V cores (including the Quasar
debug register interface and coprocessor debug bus), and evaluates RISC-V
emulators as replay targets — establishing what is already available and what
is physically possible before the interceptor is designed in Chapter 6.

## Key Findings

1. **No existing tool supports interactive stepping or full state capture.**
   All nine debug mechanisms (Watcher, DPRINT, Assert/Pause, Inspector,
   Profiler, NOC logging, data collection, ring buffer, tt-exalens) are
   polling-based, one-way, post-mortem, or fatal. None provide breakpoints,
   conditional halts, or register-level inspection from the host.

2. **Tensix hardware DOES support GDB-style debugging primitives.** WH/BH
   provide 4 breakpoints per thread with PC-match and conditional modes
   (opcode, loop, cross-thread) via `tensix_functions.h`. Quasar provides a
   full RISC-V debug register interface (`rvdbg_cmd`: PAUSE/STEP/CONTINUE,
   GPR/CSR/FPR/VEC read-write, 8 hardware watchpoints per RISC, memory
   read-write) plus a spec-compliant Debug Module (dmcontrol/dmstatus).

3. **Host accessibility of Quasar debug registers is an open question.**
   The RISC_DBG_CNTL_0/1 pulse-based protocol enables cross-core debug access,
   but whether host PCIe writes can reach these registers requires hardware
   validation.

4. **Three-tier debug strategy recommended.** Tier 1: Host functional model
   replay (from `trisc.cpp` infrastructure, ~2 weeks effort, full GDB, no
   hardware needed). Tier 2: ebreak + mailbox GDB protocol on real hardware
   (~4 weeks). Tier 3: Quasar Debug Module / JTAG-based hardware attach
   (contingent on host accessibility validation).

5. **Spike/QEMU are feasible for scalar RISC-V but cannot model the Tensix
   coprocessor.** SFPU, tile math, CB hardware semantics, and NOC transactions
   are all lost. The host functional model (compile with host compiler +
   stub headers) provides better fidelity for LLK debugging.

## Chapter Contents

### [5.1 Existing Debug Tools](01_existing_debug_tools.md)

Watcher system (waypoints, NOC sanitization, assert, pause, ring buffer),
DPRINT (per-RISC buffers, typed wire format, tile printing), Assert/Pause
mechanisms (ASSERT, LIGHTWEIGHT_KERNEL_ASSERTS, LLK_ASSERT, PAUSE), Inspector
(RPC, compile events, DWARF), Profiler (Tracy, LLK zone profiler), NOC
logging (NocWriteEvent/NocReadEvent), data collection, tt-exalens, firmware
debug stubs, and gap summary table.

### [5.2 Tensix Hardware Debug Capabilities](02_tensix_hardware_debug_capabilities.md)

WH/BH breakpoint infrastructure (4 per thread, conditional modes), Quasar
`rvdbg_cmd` enumeration (16 commands), `rvdbg_status` union, 8 hardware
watchpoints (HW_WCHPT0-7), RISC_DBG_CNTL_0/1 pulse protocol with bit-field
layout, Quasar Debug Module (dmcontrol/dmstatus/abstractcs/progbuf/SBA),
debug array readback (SRCA via DEST workaround, SRCB via latch, DEST direct),
coprocessor debug bus (dbg_bus_cntl_t, HW_CFG_SIZE=187), thread halt/unhalt
protocol, ebreak exception flow, GDB RSP mapping, architecture comparison.

### [5.3 Emulation and Replay Target Assessment](03_emulation_and_replay_target_assessment.md)

JTAG infrastructure (JtagDevice API, bandwidth analysis), RISC-V emulators
(Spike, QEMU, riscvOVPsim comparison), host functional model (`trisc.cpp`
deep analysis with execution flow and hardware dependency table),
E-Pure/E-Model/E-Stub/E-Opaque emulation taxonomy, concrete Spike emulator
workflow (4-phase with command lines), RTL simulation infrastructure
(RtlSimulationChip, Versim descriptors), four GDB-attach options (A-D)
with weighted scoring matrix, seven-dimensional decision matrix across six
targets, three-tier debug strategy with transition criteria,
TT_METAL_RISCV_DEBUG_INFO and riscv-tt-elf-gdb, effort estimates.

## Cross-References

- **Ch1 (Hardware Debugger Reference):** The four GDB-attach options (Section 7)
  implement the "cooperative halt" and "capture-and-replay" patterns identified
  in Ch1's GPU debugger analysis.
- **Ch2 (State Inventory):** The emulation taxonomy maps each state category
  from Ch2 to an emulation difficulty level (E-Pure through E-Opaque).
- **Ch3 (LightMetal Foundation):** The JTAG side-channel provides an alternative
  to LightMetal's command-queue-based capture for post-mortem L1 readback.
- **Ch4 (Dispatch and Coordination):** The ebreak + mailbox protocol (Option C)
  builds on Ch4's understanding of the BRISC master / subordinate sync model.
- **Ch6 (proposed):** Hardware capabilities documented here constrain the
  interceptor's capture scope and trigger mechanisms.
- **Ch7 (proposed):** The three replay targets (emulator, hardware, host model)
  define the replay debugger's architecture.

## Key Source Files

| File | Path | Relevance |
|------|------|-----------|
| `watcher_server.hpp/cpp` | `tt_metal/impl/debug/watcher_server.hpp` | Watcher polling, mailbox decode |
| `dprint_server.hpp` | `tt_metal/impl/debug/dprint_server.hpp` | DPRINT host server |
| `dprint.h` | `tt_metal/hw/inc/api/debug/dprint.h` | Device-side DPRINT API |
| `waypoint.h` | `tt_metal/hw/inc/api/debug/waypoint.h` | Waypoint 4-char tag API |
| `assert.h` | `tt_metal/hw/inc/api/debug/assert.h` | Assert, ebreak mechanism |
| `pause.h` | `tt_metal/hw/inc/api/debug/pause.h` | PAUSE macro |
| `llk_assert.h` | `tt-llk/tt_llk_wormhole_b0/llk_lib/llk_assert.h` | LLK_ASSERT delegation |
| `tensix_functions.h` | `tt_metal/hw/inc/internal/tensix_functions.h` | WH/BH breakpoints, debug array |
| `ckernel_riscv_debug.h` | (Quasar) | rvdbg_cmd, rvdbg_status, HW_WCHPT |
| `ckernel_debug.h` | (Quasar) | dbg_bus_cntl_t, debug bus |
| `overlay_reg.h` | (Quasar) | Debug Module registers |
| `jtag_device.hpp` | `tt_metal/third_party/umd/device/api/umd/device/jtag/jtag_device.hpp` | JTAG access |
| `noc_debugging.hpp` | `tt_metal/hw/inc/hostdev/noc_debugging.hpp` | NOC event logging |
| `trisc.cpp` | `tt-llk/tests/helpers/src/trisc.cpp` | Host functional model |

## Notation

- **PROPOSED:** Designs not yet implemented in TT-Metal.
- File paths relative to TT-Metal repository root.
- WH/BH breakpoint APIs are guarded by `#ifndef ARCH_QUASAR`.
- Quasar debug APIs are in separate header files from `tt-2xx/quasar/`.
