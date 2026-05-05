# Lessons and Requirements for Tenstorrent

This file translates the GPU debugger patterns identified in `01_gpu_debugger_architectures.md` into concrete Tenstorrent requirements, articulates the fundamental insight that motivates the capture-and-replay approach, defines the capture-and-replay alternative in concrete terms, and presents a decision matrix and gap analysis that frames the remainder of this guide.

---

## 1. Translation of GPU Debugger Capabilities to Tenstorrent Requirements

Each core debugger capability has a specific meaning when applied to the Tensix architecture. This section makes those meanings concrete.

### 1.1 "Attach to BRISC on Core (3,4)"

In cuda-gdb, `cudbgAttach()` connects the debugger to a CUDA context on a specific GPU. The Tenstorrent equivalent requires:

1. **Identifying the target core** by its NOC coordinate $(x, y)$ within the chip's core grid.
2. **Selecting the RISC-V processor** via the `RISC_DBG_CNTL_0::risc_sel` field or `rvdbg_risc_sel` enum (encoding details and a known conflict between these two are documented in `01_gpu_debugger_architectures.md` Section 5.1).
3. **Establishing a communication channel** (PCIe + NOC, JTAG, or mailbox protocol).
4. **Ensuring dispatch coordination** so the dispatch system does not timeout-kill the core during the debug session.

**Current infrastructure:** The HAL (`tt_metal/llrt/hal/`) provides coordinate translation. The Watcher system already polls specific cores by coordinate. `JtagDevice::read32(chip_id, noc_x, noc_y, address)` provides raw access.

### 1.2 "Inspect SRCA Register Bank for TRISC1"

This is a compute-datapath-specific operation with no direct GPU analog. On Tenstorrent:

- **What it means**: Read the contents of the SRCA register array while debugging a kernel running on TRISC1 (the math thread). SRCA holds the first operand tile data after unpacking.
- **What it requires**: The coprocessor must be halted (no instructions in flight modifying SRCA), and the debug array read path must be enabled.
- **How it works today (Wormhole/Blackhole)**: On Wormhole/Blackhole, this uses the multi-step SRCA-via-DEST workaround described in `01_gpu_debugger_architectures.md` Section 5.3. Config registers are readable via `dbg_read_cfgreg` (see same section).
- **How it works on Quasar**: `rvdbg_cmd::RD_REG`, `RD_FPREG`, and `RD_VECREG` provide direct register access without the indirection required on Wormhole/Blackhole.
- **Limitation**: The `dbg_get_array_row` function must run on a TRISC (it uses Tensix instructions). It cannot be called from the host directly. A debug agent on BRISC could invoke these functions if the coprocessor thread is halted, but this requires careful coordination.

### 1.3 "Single-Step Through Math Operations"

- **On Quasar**: `rvdbg_cmd::STEP` single-steps one RISC-V instruction. Combined with `RD_REG`, `RD_MEM`, and `RD_CSR`, this provides a full GDB single-step experience.
- **On Wormhole/Blackhole**: There is no explicit single-step API for the RISC-V cores. Single-stepping would require setting a breakpoint at the next instruction, resuming, and waiting for the breakpoint hit -- unreliable due to variable-length instructions and branches.

> This asymmetry -- Quasar has a clean single-step primitive; Wormhole/Blackhole do not -- is one of the strongest arguments for prioritizing the capture-and-replay approach, which works identically across all architectures.

### 1.4 "Inspect Local Memory"

- **What it means**: Read L1 SRAM contents for a specific Tensix core (the equivalent of reading GPU shared memory).
- **From the host via PCIe**: Configure a TLB window or use the NOC-based read path. This works even while the core is running, but may see inconsistent data.
- **From the host via JTAG**: `JtagDevice::read32(chip_id, noc_x, noc_y, l1_addr)` provides a side-channel access that does not go through the dispatch system.
- **On Quasar**: `rvdbg_cmd::RD_MEM` reads memory from the perspective of the target RISC, which may differ from a raw L1 read if the core has an L1 data cache.
- **Open question**: Whether L1 can be read reliably while a kernel is actively executing depends on whether L1 is dual-ported and whether read-modify-write races are possible. See Chapter 8, `03_open_questions_and_validation_experiments.md`.

---

## 2. The Key Insight from All Hardware Debuggers

> **Every production hardware kernel debugger requires either (a) dedicated debug circuitry that can halt, step, and inspect the processor without modifying the running program, or (b) an instrumented software trap mechanism that cooperatively halts the processor and exposes its state. There is no third option for live debugging.**

Tenstorrent's position is mixed:

- **Quasar** has both: a RISC-V Debug Module (hardware halt/resume/step via `dmcontrol`) and a software trap mechanism (`ebreak` + trap handler). This is comparable to Arm CoreSight and could support a traditional GDB-attached debugger.
- **Wormhole/Blackhole** have partial support: coprocessor breakpoints exist but RISC-V core debug (halt/step/register-read for BRISC/NCRISC/TRISCs) is limited. The `ebreak` mechanism exists but with only a default hang handler.
- **All architectures** lack the middle layers: no structured debug API, no GDB RSP stub, no debug session management.

Building a full live debugger stack (GDB stub, debug API, session management, multi-core coordination) is a substantial engineering effort. Combined with the five-RISC coordination problem, dispatch timeout interference, and multi-core cascade described in `01_gpu_debugger_architectures.md`, this leads to the central question of this guide:

> Is there an alternative to building a full live debugger that still gives engineers the ability to interactively debug kernel logic?

---

## 3. The Capture-and-Replay Alternative

The answer is **yes**: instead of attaching to a running core in real-time, **serialize all state needed to reproduce a kernel invocation** and **replay it in a controllable environment** where standard debugging tools already work.

### 3.1 How Capture-and-Replay Works

```
+-------------------+     +-------------------+     +-------------------+
|   CAPTURE PHASE   |     |   TRANSFER PHASE  |     |    REPLAY PHASE   |
|                   |     |                   |     |                   |
| Runtime hooks     |     | .ttksnap file     |     | Load snapshot     |
| intercept kernel  | --> | (FlatBuffer with  | --> | into controllable |
| dispatch, capture |     | binaries, L1 data,|     | environment:      |
| all state into    |     | configs, args)    |     | - RISC-V emulator |
| snapshot          |     |                   |     | - Host func model |
|                   |     |                   |     | - Real hardware   |
|                   |     |                   |     |   via debug regs  |
|                   |     |                   |     |                   |
|                   |     |                   |     | Debug with GDB    |
+-------------------+     +-------------------+     +-------------------+
```

**Capture phase**: At kernel dispatch time, host-side interceptor hooks serialize the compiled kernel binaries, L1 memory contents, circular buffer configurations, runtime arguments, semaphore state, and all configuration into a self-contained snapshot file (`.ttksnap`). This is an extension of TT-Metal's existing LightMetal capture system. See Chapter 3, `02_gaps_for_kernel_level_debugging.md`.

**Replay phase**: The snapshot is loaded into one of three replay targets:

1. **RISC-V emulator** (e.g., Spike or QEMU): The kernel ELF is loaded, L1 memory is pre-populated from the snapshot, and a standard GDB stub (built into Spike/QEMU) provides full interactive debugging -- breakpoints, single-stepping, variable inspection, stack traces. See Chapter 7, `01_emulator_based_replay.md`.

2. **Host functional model**: The kernel source is recompiled with a host C++ compiler against stub headers that model L1, circular buffers, and semaphores. The TT-LLK test infrastructure (`tt-llk/tests/helpers/src/trisc.cpp`) already demonstrates this pattern: `ckernel::regfile` is a host-side array, `run_kernel(temp_args)` executes the kernel function directly, and the entire program runs under standard host GDB. See Chapter 7, `02_on_hardware_and_host_model_replay.md`.

3. **Real hardware via debug registers**: The snapshot's L1 contents and binaries are written to a target core, and the Quasar debug register interface (`rvdbg_cmd`) is used for GDB-style control. This provides full hardware fidelity but requires hardware access. See Chapter 7, `02_on_hardware_and_host_model_replay.md`.

### 3.2 The Capture Envelope

A **capture envelope** is the minimal complete set of data that enables bit-exact replay of a single kernel invocation on a single Tensix core. It contains:

- **Compiled binaries**: ELF images for each active RISC-V processor (up to 5 on Wormhole/Blackhole, up to 6 on Quasar which adds TRISC3), including DWARF debug symbols if compiled with `-g`.
- **Runtime configuration**: the `launch_msg_t` / `kernel_config_msg_t` structures, runtime arguments per core, circular buffer configurations, semaphore initial values.
- **Memory contents**: L1 SRAM contents at the moment of kernel launch, including tile data in circular buffers, firmware-initialized regions, and mailbox state.
- **Build metadata**: compile-time arguments, preprocessor defines, HLK descriptor, architecture identifier, build key.

The total size of a capture envelope ranges from approximately 200 KB (single-TRISC compute kernel with minimal CB data) to 2 MB (full five-RISC kernel with populated circular buffers), compressible to 50-500 KB with LZ4. See Chapter 2 for the detailed state inventory.

### 3.3 Why Capture-and-Replay Works for LLK Debugging

LLK debugging has characteristics that make this approach particularly effective:

1. **Deterministic compute.** LLK kernels perform tile-level math operations (matmul, eltwise, reduce) that are deterministic given the same inputs. If you capture the inputs, you can reproduce the outputs exactly (modulo FPU precision differences in emulation).

2. **Self-contained TRISC execution.** The most common debugging scenario targets TRISC1 (math). TRISC1's inputs come from TRISC0 (unpack) via circular buffers in L1. If the snapshot captures L1 contents, TRISC1 can be replayed in isolation by pre-populating semaphores at values that allow it to proceed without waiting for TRISC0.

3. **Small state footprint.** A single-core capture envelope is 200 KB - 2 MB -- small enough to capture thousands of invocations and transmit over a network for remote debugging.

4. **Failure reproduction.** The most common workflow is: "the kernel crashed / produced wrong output; I need to reproduce and diagnose it." Capture-and-replay makes this trivial: capture the failing invocation, replay it as many times as needed, with full GDB support, on any machine.

### 3.4 What Capture-and-Replay Gives Up

The trade-off is explicit:

- **No live observation of timing-dependent behavior**: Race conditions between RISC-V cores, NOC transaction ordering, L1 bank contention cannot be reproduced deterministically.
- **No observation of state that was not captured**: The state inventory (Chapter 2) is critical.
- **Emulation gaps**: RISC-V emulators do not model the Tensix SFPU (custom vector unit), hardware data format conversions (BFloat16, block float), or the tile engine pipeline.

---

## 4. Decision Matrix: Three Debugging Strategies

| Criterion | Strategy A: Hardware JTAG Debugging | Strategy B: Firmware-Cooperative Debugging | Strategy C: Capture-and-Replay |
|---|---|---|---|
| **Mechanism** | OpenOCD + J-Link via JTAG; halt core, read registers directly | BRISC firmware as debug agent; halts TRISCs via semaphore/rvdbg; communicates with host via mailbox | Intercept dispatch, serialize state to `.ttksnap`, replay in Spike/QEMU with GDB stub |
| **Hardware fidelity** | Perfect -- real hardware | High -- real hardware, but debug agent overhead | Approximate -- no SFPU, NOC timing, L1 bank contention |
| **Hardware required** | Device + JTAG probe (J-Link) | Device only | Device for capture only; none for replay |
| **Intrusiveness** | Low -- JTAG is side-channel | Medium -- BRISC used as debug agent | Low -- capture at dispatch time only |
| **Multi-core support** | One core at a time via JTAG | One core at a time (BRISC controls local TRISCs) | Scalable -- multiple emulator instances |
| **Breakpoint types** | Hardware only (limited count) | Software (`ebreak`) + hardware watchpoints | Unlimited software breakpoints in emulator |
| **Single-step latency** | ~10-100 us (PCIe round-trip) | ~1-10 us (on-chip communication) | <1 us (emulator) |
| **Supports offline debugging?** | No | No | Yes |
| **Reproducibility** | Low (timing perturbation) | Low (debug agent alters timing) | High (deterministic inputs) |
| **CI/CD integration** | Impractical (hardware-specific) | Difficult (device contention) | Natural (capture on failure, attach artifact) |
| **Multi-developer** | One developer per device | One developer per device | Unlimited (snapshot files are shareable) |
| **Architecture support** | Quasar (via DM), WH/BH (via coprocessor breakpoints) | Quasar (`rvdbg_cmd`), WH/BH (partial) | All architectures (architecture-independent) |
| **Existing infrastructure** | `JtagDevice`, `riscv-tt-elf-gdb`, `-g` flag | `ebreak`, Watcher polling, `rvdbg_cmd` (Quasar) | LightMetal capture, `trisc.cpp` test harness, Spike |
| **Risk** | High: debug reg accessibility uncertain; device state corruption possible | Medium: dispatch timeouts; five-RISC deadlock | Low: all components are proven technologies |
| **Implementation effort** | ~8-12 pw (OpenOCD target + GDB RSP) | ~10-14 pw (debug firmware + RSP + agent) | ~10-15 pw (capture hooks + snapshot format + emulator + host model) |

### Recommendation

The three strategies are not mutually exclusive. The recommended approach is a phased combination:

1. **Start with Strategy C** (capture and replay): it provides the broadest coverage, requires no hardware for replay, and delivers the fastest iteration cycle for the most common scenario -- investigating LLK compute kernel logic errors.

2. **Add Strategy B** (firmware-cooperative) for cases requiring hardware-fidelity debugging: NOC interaction issues, timing-sensitive failures, SFPU numerical accuracy investigation.

3. **Investigate Strategy A** (JTAG/hardware debug) as a long-term investment for Quasar, contingent on validating that the Debug Module registers are host-accessible via PCIe.

Strategy C's capture infrastructure (FlatBuffer snapshot schema, dispatch interception hooks, L1 readback) is a prerequisite for Strategy B's firmware-cooperative approach as well: the debug agent on BRISC would use the same snapshot format to send captured state to the host. This makes capture-and-replay the highest-leverage first investment.

---

## 5. Gap Analysis Summary

The following table maps cuda-gdb features against current Tenstorrent capabilities and identifies what the capture-and-replay system must provide:

| cuda-gdb Feature | Tenstorrent Hardware Capability | Current TT-Metal Debug Infrastructure | Gap | Addressed In |
|---|---|---|---|---|
| **Attach to running kernel** | Quasar `dmcontrol.haltreq`; WH/BH `ebreak` hang | Watcher polls core status; Inspector tracks program lifecycle | No host-initiated attach API | Ch. 6 `01_interceptor_architecture_and_hooks.md` |
| **Software breakpoints** | `ebreak` instruction on all RISC-V cores | `LLK_ASSERT` uses `ebreak`; `PAUSE()` macro spins on flag | `ebreak` is fatal (hangs); no resumable breakpoints | Ch. 7 `01_emulator_based_replay.md` (emulator provides unlimited breakpoints) |
| **Hardware breakpoints** | Quasar: 8 watchpoints; WH/BH: coprocessor breakpoints with conditional triggers | None exposed to users | No host API to set/clear | Ch. 5 `02_tensix_hardware_debug_capabilities.md` |
| **Single-step** | Quasar: `rvdbg_cmd::STEP`; WH/BH: no RISC-V single-step | None | Fundamental gap on WH/BH; Quasar step not exposed to host | Ch. 7 (emulator provides step natively) |
| **Register read (GPR)** | Quasar: `rvdbg_cmd::RD_REG` for x0-x31; WH/BH: none for RISC-V GPRs | None | Requires trap handler on WH/BH to save GPRs to L1 | Ch. 7 (emulator provides GPR read natively) |
| **Register read (compute)** | All: `dbg_get_array_row` for SRCA/SRCB/DEST; config register dump | DPRINT can print tile data (one-way) | APIs exist but device-side only, not host-accessible | Ch. 5 `02_tensix_hardware_debug_capabilities.md` |
| **Memory read/write** | All: L1 via NOC from host; JTAG; Quasar `RD_MEM`/`WR_MEM` | Watcher reads mailbox addresses | Works, but no structured "snapshot L1 region" API | Ch. 6 `01_interceptor_architecture_and_hooks.md` |
| **Watchpoints** | Quasar: `HW_WCHPT0`-`HW_WCHPT7` (8 per RISC); WH/BH: none | None exposed | Quasar HW exists but no host API | Ch. 7 `02_on_hardware_and_host_model_replay.md` |
| **Conditional breakpoints** | WH/BH: `breakpoint_set_condition_op`, `_loop`, `_other_thread` | None exposed | Rich HW exists but completely unused | Ch. 5 `02_tensix_hardware_debug_capabilities.md` |
| **Stack trace / backtrace** | DWARF unwind info in ELF when compiled with `-g` | `riscv-tt-elf-gdb` can parse DWARF; Watcher tracks stack usage | ELF with DWARF exists but not preserved; no mechanism for stack walk | Ch. 7 (emulator provides full backtrace); Ch. 6 (ELF preserved in snapshot) |
| **Kernel state capture** | N/A (not a cuda-gdb feature) | LightMetal captures host API calls; no L1, binaries, or semaphore state | LightMetal provides ~40% of needed capture | Ch. 3 `02_gaps_for_kernel_level_debugging.md`; Ch. 6 `03_snapshot_schema_and_serialization.md` |
| **Multi-core coordinated debug** | Quasar: `dmcontrol.hartsello`; WH/BH: `breakpoint_set_condition_other_thread` | None | No multi-core debug coordination | Ch. 7 `03_multi_core_replay_and_noc_coordination.md` |
| **Non-invasive trace** | Debug bus on WH/BH; no ETM equivalent | Watcher waypoints; DPRINT; Profiler (Tracy) | No instruction-level trace; requires live execution | Not addressable by capture-and-replay |

### Critical Gaps by Category

The gaps cluster into three categories:

**Category A -- Hardware exists, host API missing (6 gaps)**:
- Resumable breakpoints, single-step, GPR read, hardware watchpoints, conditional breakpoints, cross-RISC triggers.
- Resolution: build a driver-level debug API wrapping the hardware registers (significant effort per Pattern 2).
- Alternative: capture-and-replay makes these available in the emulator/host model without building the on-device API.

**Category B -- Partial infrastructure exists, needs extension (3 gaps)**:
- L1 memory snapshot (reads possible, no API), debug symbol retention (ELFs and GDB exist, no target), config register inspection (device-side macros exist).
- Resolution: extend existing tools with host-side wrappers and capture hooks.
- Alternative: capture-and-replay captures this state at dispatch time.

**Category C -- Fundamental capability absent (2 gaps)**:
- Non-intrusive instruction trace (no hardware equivalent to Arm ETM).
- Host-initiated attach to a running kernel without prior instrumentation.
- Resolution for trace: not achievable without hardware changes.
- Resolution for attach: Quasar Debug Module's `dmcontrol.haltreq` could enable this, pending hardware validation.

> Category A and B issues constitute the vast majority of LLK bugs (estimated >90% based on the types of issues that Watcher, DPRINT, and assert mechanisms currently catch). Category C issues are rare and specialized, typically arising only in hardware bring-up or performance optimization contexts.

---

## 6. The Spectrum from Polling to Interactive Debugging

It is useful to place the existing TT-Metal debug tools and the proposed capture-and-replay system on a spectrum of debug capability:

```
Polling/Post-Mortem                                          Interactive/Live
      |                                                            |
      v                                                            v
  Watcher    DPRINT    Assert/    Debug Ring    Capture-    Hardware    GDB Stub
  (polls     (printf   ebreak     Buffer        and-Replay  Replay     (live
  mailboxes  from      (fatal     (32 entries   (offline    (on-device  attach,
  every      device,   halt,      per core,     GDB via    with debug  step,
  250ms)     one-way)  no state   post-mortem)  emulator)  registers)  inspect)
                       capture)
```

The existing tools cluster at the left end: polling-based (Watcher), one-directional (DPRINT), fatal (assert/ebreak), or limited in capacity (debug ring buffer). **Capture-and-replay bridges to the right side of the spectrum** by enabling interactive GDB sessions, but does so offline rather than live. A full GDB stub (rightmost) would complete the spectrum but is the most expensive to build and the most risky.

---

## 7. Requirements Derived from This Analysis

Based on the GPU debugger study and gap analysis, the following requirements guide the rest of this guide:

**R1 -- State inventory completeness.** Before designing the interceptor, we must enumerate every piece of state that constitutes a reproducible kernel invocation. This is the subject of Chapter 2.

**R2 -- Extend LightMetal, do not replace it.** LightMetal already captures approximately 40% of the needed state (program structure, kernel file paths, CB configs, runtime args, buffer data) and provides the FlatBuffer serialization infrastructure. The interceptor should extend this foundation. This is analyzed in Chapter 3.

**R3 -- Hook the dispatch path at the right level.** The interceptor must capture state after all configuration is resolved (post-`finalize_offsets()`) but before it is serialized to the raw command stream. The dispatch path analysis in Chapter 4 identifies the optimal hook point.

**R4 -- Capture L1 memory contents.** The largest gap in current infrastructure is the absence of L1 memory snapshots. The interceptor must read L1 via NOC before kernel launch. This is designed in Chapter 6.

**R5 -- Retain compiled ELFs with debug symbols.** The build system must be modified to retain JIT-compiled kernel ELFs with `-g` debug info, either always (when Inspector is active) or on-demand (when capture is enabled). This is addressed in Phase 0 of the implementation roadmap (Chapter 8).

**R6 -- Support three replay targets.** The snapshot format must be self-contained enough to support replay on a RISC-V emulator, a host-compiled functional model, and on-hardware via debug registers. Each target is designed in Chapter 7.

**R7 -- Failure-triggered capture.** The system must automatically capture kernel state when Watcher detects a failure (assert, timeout, NOC violation), using post-mortem L1 reads via NOC. This is the highest-value use case and is designed in Chapter 6.

---

## Key Takeaways

- **The hardware analysis in `01_gpu_debugger_architectures.md` establishes that Tenstorrent hardware partially covers debug patterns 1 and 4 but lacks patterns 2 and 3**, and that multi-RISC coordination makes live debugging harder than on GPUs. That analysis motivates the capture-and-replay alternative developed in this file.

- **Capture-and-replay is the recommended primary strategy** because it is architecture-independent, builds on existing LightMetal and TT-LLK test infrastructure, supports offline debugging without hardware, and produces artifacts (`.ttksnap` snapshots) that can feed all three replay targets as the debug ecosystem matures. Over 90% of LLK debugging scenarios are addressable by this approach.

- **The gap analysis clusters missing capabilities into three categories**: hardware-exists-but-no-host-API (Category A, 6 gaps), partial-infrastructure-needs-extension (Category B, 3 gaps), and fundamental-capability-absent (Category C, 2 gaps). Capture-and-replay addresses Categories A and B without building on-device APIs.

- **Seven requirements (R1-R7) derived from this analysis drive the remaining chapters**: state inventory completeness, LightMetal extension, dispatch path hook selection, L1 memory capture, ELF retention, three replay targets, and failure-triggered capture.

---

**Next:** [Chapter 2 -- Complete State Inventory for Kernel Capture](../ch2_state_inventory/index.md)
