# Gap Analysis with Severity Scoring and Developer Impact Assessment

This file consolidates every gap identified across Chapters 1-7 into a single
risk-rated inventory, scored by severity (Critical / High / Medium / Low) with
impact assessment, developer pain quantification, closure steps, and
cross-references to the chapter where each gap was originally identified.
All source citations reference commit `621b949`.

> **Cross-references:** Ch 1 (GPU debugger requirements R1-R7), Ch 2 (8 state
> categories, capture envelope sizes), Ch 3 (LightMetal ~34% weighted coverage,
> 10 gaps), Ch 4 (dispatch hook analysis, H1+H4 dual-hook architecture),
> Ch 5 (9 debug tools, none interactive; three-tier debug strategy; WH lacks
> `RISC_DBG_CNTL`), Ch 6 (interceptor design, `.ttksnap` schema, 93% Ch2
> coverage), Ch 7 (three replay targets: emulator, on-hardware, host model).

---

## 1. Severity Scoring Framework

Each gap is scored on two dimensions and also on developer pain impact:

| Dimension | Definition |
|-----------|-----------|
| **Impact** | How severely the gap blocks the capture-and-replay debugging workflow if left unresolved (Blocking / Degraded / Cosmetic) |
| **Likelihood** | Probability that a typical debugging session encounters this gap (Certain / Probable / Occasional / Unlikely) |

The severity matrix:

| | Blocking Impact | Degraded Impact | Cosmetic Impact |
|---|---|---|---|
| **Certain** | CRITICAL | HIGH | MEDIUM |
| **Probable** | CRITICAL | HIGH | LOW |
| **Occasional** | HIGH | MEDIUM | LOW |
| **Unlikely** | HIGH | LOW | LOW |

### Developer Pain Scoring

Each gap is additionally scored on three pain points (1-5 scale):

| Pain Point | Description | Weight |
|-----------|-------------|--------|
| **P1 -- Debugging time** | Time from observing a failure to understanding root cause | 0.35 |
| **P2 -- Reproduction difficulty** | Effort to recreate the exact conditions that triggered a bug | 0.35 |
| **P3 -- Information loss** | State that existed at failure time but is unrecoverable after the fact | 0.30 |

**Composite score** = P1 x 0.35 + P2 x 0.35 + P3 x 0.30.

---

## 2. State Capture Gaps

These gaps concern the completeness of the kernel invocation snapshot relative
to the Ch 2 state inventory.

### GAP-SC-1: No L1 Memory Snapshot in LightMetal (CRITICAL)

- **Source:** Ch 3, File 2 (Gap 2); Ch 2, File 3
- **Description:** LightMetal's `CreateCircularBufferCommand` captures CB
  configuration (indices, sizes, data formats) but not the actual tile data
  present in L1 circular buffers at the moment of kernel dispatch. For
  compute kernel debugging, the tile data in input CBs is the primary input
  and must be present for any replay to produce meaningful results.
- **Impact:** Blocking -- without input tile data, no replay target (emulator,
  host model, or hardware) can reproduce the kernel's computation.
- **Likelihood:** Certain -- every deep capture requires tile data.
- **Pain Scores:** P1=5, P2=5, P3=5. **Composite: 5.00** (highest of all gaps).
- **Developer scenario:** When debugging "matmul produces NaN on core (3,4)",
  the developer cannot replay the compute kernel at all without input tile data.
  They must instead replay the entire program or manually construct synthetic input,
  which can take hours. L1 contents are volatile -- once the core resets, the data
  is gone permanently.
- **Current Mitigation:** None in the existing codebase. The interceptor must
  add L1 readback via UMD `read_from_device()` or PCIe/TLB-mapped reads.
- **PROPOSED Resolution:** Phase 2 of the implementation plan adds L1 snapshot
  capability to the interceptor (Ch 6, File 1, Section 4.3). The
  `L1RegionSnapshot` table in the `.ttksnap` schema (Ch 6, File 3, Section 3)
  stores sparse L1 regions with LZ4 compression.

**How to close:**

| Step | Description | Owner | Effort |
|------|-------------|-------|--------|
| SC1-A | Add `read_from_device()` calls in the interceptor hook (site B, `tt_metal/impl/program/dispatch.cpp:1890`) to read L1 regions for each active CB on the target core. Use `tt::tt_metal::detail::ReadFromDeviceL1()` or the UMD path. | Runtime Eng | 1 pw |
| SC1-B | Define a sparse L1 representation: capture only address ranges corresponding to active CBs (from `CircularBufferConfig::locally_allocated_address_` and `total_size`), semaphore slots (16 x 4 bytes), and runtime args region. | Runtime Eng | 0.5 pw |
| SC1-C | Implement LZ4 compression for captured L1 regions before serialization. Tile data compresses 2-4x with LZ4. | Infra Eng | 0.5 pw |
| SC1-D | Add `L1RegionSnapshot` FlatBuffer table to the schema (see Ch 6, `03_snapshot_schema_and_serialization.md`). | Infra Eng | 0.5 pw |

- **Effort Estimate:** 2.5 pw (Phase 2).
- **Validation:** Capture L1 for a single-core matmul, compare captured CB data against expected tile values. Pass criterion: bit-exact match for all active CB tiles.
- **Cross-references:** Ch 2, `03_memory_contents_and_tile_data.md`; Ch 3, `02_gaps_for_kernel_level_debugging.md`, Section 2; Ch 6, `03_snapshot_schema_and_serialization.md`.

### GAP-SC-2: Compiled ELF Binaries Not Captured by LightMetal (CRITICAL)

- **Source:** Ch 3, File 2 (Gap 1); Ch 2, File 1
- **Description:** `CreateKernelCommand` in LightMetal stores the kernel source
  file path (`file_name` at `command.fbs:79`), not the compiled ELF binaries. Replay currently
  requires the same source tree and a recompilation step. For offline debugging
  or cross-machine triage, this is a hard blocker.
- **Impact:** Blocking -- without compiled binaries, the snapshot is not
  self-contained; the replay target cannot load kernel code.
- **Likelihood:** Certain -- every capture must include binaries.
- **Pain Scores:** P1=4, P2=3, P3=5. **Composite: 3.95**.
- **Current Mitigation:** JIT build cache retains ELFs on the same machine,
  but the cache key is an FNV1a hash that is not stored in the LightMetal
  binary and binaries are not portable across machines.
- **PROPOSED Resolution:** Phase 1 adds `KernelBinaryBlob` to the `.ttksnap`
  schema. The interceptor reads binaries via `Kernel::binaries(build_key)`
  (`kernel.hpp:197`) at the primary hook.

**How to close:**

| Step | Description | Owner | Effort |
|------|-------------|-------|--------|
| SC2-A | When capture mode is active, force `KernelBuildOptLevel::O0` and `-g` for all kernels in the target program. Modify `JitBuildOptions` to accept a `force_debug` flag. | Runtime Eng | 1 pw |
| SC2-B | After `JitBuildState::build()` completes, read the ELF binary from `Kernel::binaries_` (the `ll_api::memory` objects) and serialize into `KernelBinaryBlob` FlatBuffer tables. | Runtime Eng | 1 pw |
| SC2-C | Add an auxiliary hook at `Inspector::program_kernel_compile_finished` (`inspector.hpp:39-43`) to record the full defines map, compile args, and build key alongside the binary. | Debug Eng | 0.5 pw |

- **Effort Estimate:** 2.5 pw (Phases 0-1).
- **Cross-references:** Ch 2, `01_compiled_binary_and_build_state.md`; Ch 3, `02_gaps_for_kernel_level_debugging.md`, Section 1.

### GAP-SC-3: No Single-Kernel Dispatch Granularity (HIGH)

- **Source:** Ch 3, File 2 (Gap 5); Ch 4, File 2
- **Description:** LightMetal captures at program granularity
  (`EnqueueProgramCommand`), not individual kernel invocations. A matmul
  program may contain reader, writer, and compute kernel groups across
  dozens of cores. Debugging a single TRISC1 math bug requires isolating
  one kernel on one core.
- **Impact:** Degraded -- the user can capture the full program but must
  manually extract the relevant kernel state, increasing triage time from
  minutes to hours.
- **Likelihood:** Certain -- most debug sessions target a single core.
- **Pain Scores:** P1=4, P2=3, P3=3. **Composite: 3.35**.
- **Current Mitigation:** None.
- **PROPOSED Resolution:** The `CaptureFilter` in the interceptor
  (Ch 6, File 1, Section 3.2) supports `(program_id, core_coord)` targeting.
  The `TT_METAL_KERNEL_CAPTURE_CORE` env var (Ch 6, File 2) enables this.
  Uses `ProgramImpl::get_kernel_groups()` to iterate kernel groups and match
  against the target `core_coord`.
- **Effort Estimate:** Included in Phase 1 (filter logic ~0.5 pw).

### GAP-SC-4: Semaphore Runtime State Not Captured (HIGH)

- **Source:** Ch 3, File 2 (Gap 3); Ch 2, File 4
- **Description:** LightMetal records semaphore initial values from
  `CreateSemaphoreCommand` but not the volatile values at the moment of a
  specific dispatch. Inter-RISC and inter-core semaphore values are critical
  for replay of multi-RISC coordination. The `data_collection.hpp` system
  tracks `DISPATCH_DATA_SEMAPHORE` sizes (enum value 1 at
  `data_collection.hpp:21`) only for dispatch profiling.
- **Impact:** Degraded -- single-TRISC replay can work by pre-setting
  semaphores to "unblocked" values, but multi-RISC replay will deadlock or
  produce incorrect synchronization.
- **Likelihood:** Probable -- occurs whenever debugging data movement or
  inter-RISC coordination bugs.
- **Pain Scores:** P1=3, P2=4, P3=4. **Composite: 3.65**.
- **PROPOSED Resolution:** Phase 2 L1 snapshot includes semaphore addresses
  (addresses are known from `ProgramConfig::sem_offset`). The
  `SemaphoreSnapshot` table captures per-semaphore `value_at_capture`.
- **Effort Estimate:** Included in Phase 2.

### GAP-SC-5: NOC Transaction History Not Recorded (HIGH)

- **Source:** Ch 3, File 2 (Gap 4); Ch 4, File 4; Ch 2, File 4
- **Description:** The sequence of NOC reads/writes performed by BRISC/NCRISC
  during kernel execution is not captured. Without this, inter-core data
  dependencies cannot be faithfully replayed. The existing `NocWriteEvent`/`NocReadEvent`
  structs in `noc_debugging.hpp` use `int8_t` coordinates (`src_x/src_y/dst_x/dst_y`),
  `uint32_t` addresses (`src_addr/dst_addr`), and do NOT contain a timestamp field --
  timestamps are passed as separate `uint64_t` parameters to `handle_write_event()`
  and `handle_read_event()`.
- **Impact:** Degraded for single-TRISC replay (TRISCs do not issue NOC
  transactions directly). Blocking for full-core or multi-core replay.
- **Likelihood:** Occasional -- only relevant for data movement kernel
  debugging, not compute-only LLK debugging.
- **Current Mitigation:** `NocWriteEvent`/`NocReadEvent` in
  `noc_debugging.hpp` can log NOC transactions, but this is a separate debug
  feature not integrated with the capture system.
- **PROPOSED Resolution:** Phase 4 integrates NOC logging with capture.
  NOC transaction log serialization must carry timestamps externally since
  they are not part of the event structs.
- **Effort Estimate:** 2-3 pw within Phase 4.

### GAP-SC-6: Hardware Register State Not Captured (MEDIUM)

- **Source:** Ch 3, File 2 (Gap 6); Ch 2, File 4
- **Description:** Tensix config registers (HW_CFG_SIZE=187 per thread),
  SFPU register file, FPU rounding mode, and destination register file
  contents are not captured. These affect compute results.
- **Impact:** Degraded -- host functional model and emulator replay will use
  default config register values, which are correct for standard kernel
  launches but may miss non-default FPU modes.
- **Likelihood:** Occasional -- most kernels use default FPU config.
- **Current Mitigation:** Config registers are initialized by firmware
  from compile-time defines; capturing the defines (which the interceptor
  does) provides an indirect record.
- **PROPOSED Resolution:** Debug array readback via
  `dbg_dump_array_enable()`, `dbg_dump_array_rd_cmd()`, and `dbg_dump_array_to_l1()`
  from `tensix_functions.h` (starting at line 697) can read SRCA/SRCB/DEST.
  Note: there is NO `dbg_dump_read()` function -- the actual API uses the
  three functions listed above. Full config register dump via coprocessor debug
  bus (`dbg_bus_cntl_t`) is feasible but adds ~5 ms per core (187 register reads).
  Deferred to Phase 4.
- **Effort Estimate:** 1-2 pw within Phase 4.

### GAP-SC-7: Missing Configuration Fields in LightMetal Schema (MEDIUM)

- **Source:** Ch 3, File 2 (Gap 7); Ch 2, File 1
- **Description:** `KernelConfig` in `program_types.fbs` omits
  `named_compile_args`, `hlk_desc` tile metadata (data formats, tile dims,
  HLK args), `opt_level`, and `noc_mode` for `EthernetConfig`.
- **Impact:** Degraded -- replay compilation may produce different binaries if
  these fields differ from defaults.
- **Likelihood:** Probable -- `hlk_desc` fields are set for every compute
  kernel.
- **Current Mitigation:** The interceptor captures compiled binaries directly
  (GAP-SC-2 resolution), so missing compile-time fields do not block replay
  of the captured binary. They block re-compilation for source-level patches.
- **PROPOSED Resolution:** Add missing fields to `KernelBinaryBlob` metadata
  in Phase 1. Low incremental effort (~0.5 pw).

### GAP-SC-8: Launch Message and Go Signal Not Captured (MEDIUM)

- **Source:** Ch 3, File 2 (Gap 9); Ch 2, File 2
- **Description:** `launch_msg_t` and `go_msg_t` contain processor enables,
  NOC configuration, memory offsets, and synchronization fields essential for
  replay. LightMetal does not serialize these structures. The firmware in
  `trisc.cc` reads `launch_msg` from `mailboxes->launch[launch_msg_rd_ptr]`,
  computes `kernel_lma = (kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index])`,
  and calls via function pointer `reinterpret_cast<uint32_t (*)()>(kernel_lma)()`.
  Runtime args are accessed via the `rta_l1_base` global variable (NOT passed as
  a function argument to the kernel).
- **Impact:** Degraded -- the replay engine must reconstruct launch messages
  from CB/RTA/kernel configuration, which is possible but error-prone.
- **Likelihood:** Certain -- every replay needs these.
- **PROPOSED Resolution:** The `.ttksnap` schema includes
  `ProgramConfigSnapshot` with all 10 fields from `ProgramConfig`. The primary
  hook at Site B (`tt_metal/impl/program/dispatch.cpp:1890`) has direct access
  to populated `launch_msg_t` structures.
- **Effort Estimate:** Included in Phase 1 serialization.

### GAP-SC-9: No Versioning Metadata (LOW)

- **Source:** Ch 3, File 2 (Gap 10)
- **Description:** LightMetal binary has a TODO for git hash and versioning.
  Without this, snapshots from different TT-Metal versions may be
  misinterpreted.
- **PROPOSED Resolution:** `CaptureMetadata` table in `.ttksnap` schema
  includes `tt_metal_git_hash`, `schema_version_major/minor`, `compiler_version`,
  `architecture`, `l1_size_bytes` (Ch 6, File 3, Section 3).
- **Effort Estimate:** ~0.5 pw in Phase 1.

---

## 3. Debug Infrastructure Gaps

These gaps concern the tools and hardware capabilities needed for GDB-style
interactive debugging during replay.

### GAP-DI-1: No GDB Stub for Tensix RISC-V Cores (CRITICAL)

- **Source:** Ch 1, File 2 (R3); Ch 5, File 2; Ch 5, File 3
- **Description:** No existing mechanism translates GDB Remote Serial Protocol
  (RSP) commands to Tensix debug register operations. This is the central
  missing piece for interactive debugging on real hardware.
- **Impact:** Blocking for on-hardware replay (Tier 2/3 debug). Not blocking
  for emulator or host model replay (Tier 1).
- **Likelihood:** Certain for hardware debug workflows.
- **Pain Scores:** P1=5, P2=2, P3=4. **Composite: 3.65**.
- **Current Mitigation:** `riscv-tt-elf-gdb` exists in the toolchain for
  symbol-level debugging, and `TT_METAL_RISCV_DEBUG_INFO` (`rtoptions.hpp:204`)
  compiles with `-g`. But no target backend connects GDB to live hardware.
- **PROPOSED Resolution:** Phase 4 implements a GDB RSP stub mapping
  g/G/m/M/c/s/Z packets to architecture-specific debug operations:
  - **Quasar:** Use `rvdbg_cmd` operations (PAUSE, STEP, CONTINUE, RD_REG,
    WR_REG, RD_MEM, WR_MEM) via RISC_DBG_CNTL_0/1 plus 8 hardware
    watchpoints (HW_WCHPT0-7 in `rvdbg_reg` enum of `ckernel_riscv_debug.h`).
  - **BH:** Use raw RISC_DBG_CNTL registers or ebreak+mailbox protocol.
  - **WH:** ebreak+mailbox only (no RISC_DBG_CNTL registers).
- **Effort Estimate:** 4-6 pw in Phase 4.
- **Risk Register Entry:** R-DI-1. Contingent on GAP-DI-2 resolution.

### GAP-DI-2: Quasar/BH Debug Register Host Accessibility Unvalidated (CRITICAL)

- **Source:** Ch 5, File 2, Section on RISC_DBG_CNTL; Ch 5 Key Finding 3
- **Description:** The `RISC_DBG_CNTL_0/1` pulse-based protocol enables
  cross-core debug access on Quasar (`T6_DEBUG_REGS__RISC_DBG_CNTL_0` in
  `t6_debug_map.h`, lines 220-223) and Blackhole (`RISCV_DEBUG_REG_RISC_DBG_CNTL_0`
  at `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h:162-163`), but whether
  host PCIe writes can reach these registers has not been validated.

  **Architecture-specific register status:**
  - **Wormhole B0:** No `RISC_DBG_CNTL` defines exist in
    `tt-1xx/wormhole/tensix.h` (confirmed by codebase search). WH is limited
    to breakpoint infrastructure via `RISCV_DEBUG_REG_BREAKPOINT_CTRL` and
    debug array readback.
  - **Blackhole:** `RISCV_DEBUG_REG_RISC_DBG_CNTL_0` at offset `0x80` and
    `RISCV_DEBUG_REG_RISC_DBG_CNTL_1` at offset `0x84` are defined as active
    (not commented) macros at `tt-1xx/blackhole/tensix.h:162-163`. BH has the
    raw register interface but does NOT have the `rvdbg_cmd` software API
    (which exists only in `ckernel_riscv_debug.h` under `tt_llk_quasar`).
  - **Quasar:** The `ckernel_riscv_debug.h` file provides the full `rvdbg_cmd`
    software API (PAUSE, STEP, CONTINUE, RD_REG, WR_REG, RD_MEM, WR_MEM,
    RD_FPREG, WR_FPREG, RD_VECREG, WR_VECREG, and more). However, the
    `RISC_DBG_CNTL_0/1` defines in Quasar's `tensix.h` are **commented out**
    (lines 175-178). Quasar accesses the debug module through the
    `RISCV_DEBUG_REGS` memory-mapped struct referenced in
    `ckernel_riscv_debug.h`, not through the `tensix.h` defines.

- **Impact:** Blocking for Tier 3 debug (hardware GDB attach via debug
  registers). Does not block Tier 1 (host model) or Tier 2 (ebreak+mailbox).
- **Likelihood:** Certain -- must be validated before Phase 4 hardware
  replay can proceed.
- **PROPOSED Resolution:** Validation Experiment VE-1 (Section 3 of
  `03_open_questions_and_validation_experiments.md`).
- **Go/No-Go:** Phase 3->4 gate.

### GAP-DI-3: No Tensix Functional Emulator (HIGH)

- **Source:** Ch 1, File 2 (R4); Ch 5, File 3
- **Description:** No off-the-shelf RISC-V emulator (Spike, QEMU,
  riscvOVPsim) can model Tensix-specific behavior: SFPU custom ISA
  extensions (SFPI), tile math units, circular buffer hardware semantics
  (stream scratch registers), NOC transactions, or block-float data format
  handling.
- **Impact:** Degraded -- scalar RISC-V control flow is emulatable, but
  any SFPU instruction triggers an illegal instruction exception. LLK math
  functions are ~80% SFPU instructions by count.
- **Likelihood:** Certain for math kernel debugging.
- **Current Mitigation:** `trisc.cpp` host functional model
  (`tt-llk/tests/helpers/src/trisc.cpp`) provides a partial solution by
  compiling LLK kernels with a host compiler against stub headers, replacing
  SFPU intrinsics with C++ float operations. The model uses `ckernel::regfile`
  as a host-side register array (line 58), calls `run_kernel(__runtime_args_start)`
  (line 72), and writes `ckernel::KERNEL_COMPLETE` to the mailbox (line 76)
  to signal completion.
- **PROPOSED Resolution:** Tier 1 debug strategy (host functional model)
  is the recommended first target (Ch 5 Key Finding 4). Phase 3 adapts
  `trisc.cpp` infrastructure to load `.ttksnap` snapshots. Spike/QEMU
  serve scalar control-flow debugging only.
- **Effort Estimate:** 6-8 pw in Phase 3 (replay engine including host model).

### GAP-DI-4: WH Lacks RISC_DBG_CNTL Registers (HIGH)

- **Source:** Ch 5, File 2; `tt-1xx/wormhole/tensix.h` (no RISC_DBG_CNTL
  defines, confirmed by codebase search)
- **Description:** Wormhole B0 does not provide `RISC_DBG_CNTL_0/1`
  registers for cross-core debug access. The only hardware debug primitives
  are the 4-per-thread breakpoint infrastructure via
  `RISCV_DEBUG_REG_BREAKPOINT_CTRL` (also available on BH and Quasar --
  note that Quasar defines `BREAKPOINT_CTRL` at `tensix.h:240` but the
  convenience functions in `tensix_functions.h` are guarded by
  `#ifndef ARCH_QUASAR`) and the debug array readback.
- **Impact:** Degraded -- on-hardware GDB-style debugging on WH requires
  the ebreak+mailbox protocol (Ch 5, File 3, Option C) instead of the
  cleaner debug register interface. Single-stepping is cooperative, not
  preemptive.
- **Likelihood:** Certain for WH-based development (most current deployments).
- **PROPOSED Resolution:** Phase 4 implements architecture-aware GDB stub:
  ebreak+mailbox on WH, debug registers on BH/Quasar.
- **Effort Estimate:** Additional 1-2 pw in Phase 4 for WH-specific path.

### GAP-DI-5: Existing Debug Tools Are Non-Interactive (MEDIUM)

- **Source:** Ch 5, File 1; Ch 5 Key Finding 1
- **Description:** All nine existing debug mechanisms (Watcher, DPRINT,
  Assert/Pause, Inspector, Profiler, NOC logging, data collection, ring
  buffer, tt-exalens) are polling-based, one-way, post-mortem, or fatal.
  None support breakpoints, conditional halts, or register-level inspection
  from the host.
- **Impact:** Degraded -- developers rely on printf-debugging (DPRINT) and
  post-mortem analysis, which is slow and imprecise for subtle compute bugs.
- **Likelihood:** Certain -- this is the status quo.
- **PROPOSED Resolution:** The capture-and-replay system is specifically
  designed to address this gap by enabling offline interactive debugging.

### GAP-DI-6: Debug Symbol Quality for JIT Kernels (MEDIUM)

- **Source:** Ch 5, File 3; Ch 2, File 1
- **Description:** JIT-compiled kernel ELFs contain DWARF debug info only
  when `TT_METAL_RISCV_DEBUG_INFO=1` is set (`rtoptions.hpp:204`, default `false`)
  and `KernelBuildOptLevel::O0` is used. Default optimization (`O2`) strips
  most debug info. Heavily inlined LLK functions may not have frame pointers,
  making stack traces unreliable. `LLK_ASSERT` is defined at
  `tt_llk_wormhole_b0/llk_lib/llk_assert.h` (also at
  `tt_llk_blackhole/llk_lib/llk_assert.h`), NOT at `tt-llk/common/`.
- **Impact:** Degraded -- breakpoints on source lines work, but variable
  inspection may show `<optimized out>` for many values.
- **Likelihood:** Probable -- developers often debug optimized builds to
  reproduce production issues.
- **PROPOSED Resolution:** Phase 0 foundation work. Consider
  `-Og` (optimize for debug) as a middle ground. When Inspector is active,
  automatically enable `TT_METAL_RISCV_DEBUG_INFO=1`.
- **Effort Estimate:** 0.5-1 pw in Phase 0.

---

## 4. Replay Fidelity Gaps

These gaps concern differences between the replay environment and real
hardware execution.

### GAP-RF-1: SFPU/SFPI Emulation Gap (HIGH)

- **Source:** Ch 5, File 3; Ch 7, File 1
- **Description:** Tenstorrent's SFPI custom RISC-V ISA extension for
  special-function unit operations has no open-source emulator. Spike and
  QEMU will trap on SFPI instructions with illegal-instruction exceptions.
  The host functional model uses C++ float operations that approximate but
  do not exactly match Tensix FPU behavior (different rounding, different
  denorm handling, different block-float precision).
- **Impact:** Degraded -- control flow through SFPU-heavy math kernels
  cannot be traced in a standard emulator. The host model provides
  logical equivalence but not bit-exact results.
- **Likelihood:** Certain for math kernel debugging.
- **PROPOSED Resolution:** Accept functional-level fidelity for Tier 1
  debugging. Bit-exact debugging requires Tier 2/3 (on-hardware replay).
  Document known divergences in replay tool output.

### GAP-RF-2: CB Hardware Semantics in Replay (HIGH)

- **Source:** Ch 4, File 3 (Key Finding 3); Ch 5, File 3
- **Description:** Circular buffer `tiles_received`/`tiles_acked` counters
  live in hardware stream scratch registers, not L1. Host replay must
  replace these with shared atomics or a software CB model.
- **Impact:** Degraded -- multi-RISC replay coordination will differ from
  hardware behavior. Single-TRISC replay with pre-populated CBs is not
  affected.
- **Likelihood:** Probable -- any multi-RISC replay hits this.
- **PROPOSED Resolution:** Phase 3 replay engine implements software CB
  model with shared atomics for tile counters. Phase 4 extends to
  multi-RISC coordination.
- **Effort Estimate:** 2-3 pw in Phase 3.

### GAP-RF-3: Non-Deterministic Behavior in Replay (MEDIUM)

- **Source:** Ch 2, File 4; Ch 4, File 4
- **Description:** NOC arbitration timing, L1 bank contention, and
  inter-core message ordering are non-deterministic. Replay cannot
  reproduce these effects without cycle-accurate simulation.
- **PROPOSED Resolution:** Accept non-determinism for initial phases.
  Document that timing-dependent bugs require on-hardware debugging
  (Tier 2/3). Consider future integration with RTL simulation for cycle-accuracy.

### GAP-RF-4: Dispatch Infrastructure Timeout During Debug (MEDIUM)

- **Source:** Ch 5, File 3 (validation question)
- **Description:** Halting a core for GDB-style debugging may cause the
  dispatch infrastructure to time out, triggering Watcher asserts
  (`WatcherServer::killed_due_to_error()` at `watcher_server.hpp:34`)
  or command queue errors. The go/completion mailbox protocol has timeouts.
- **PROPOSED Resolution:** Disable Watcher and dispatch timeouts during
  debug sessions. Use dedicated device mode (exclusive access) for
  hardware replay. Implement `TT_METAL_DEBUG_SESSION=1` env var.
- **Effort Estimate:** ~1 pw in Phase 4.

---

## 5. Toolchain and Integration Gaps

### GAP-TI-1: No Snapshot CLI Tool (MEDIUM)

- **Source:** Ch 6, File 1 (Section on `tt-kernel-capture`)
- **Description:** No command-line tool exists for inspecting, validating,
  or replaying `.ttksnap` files.
- **PROPOSED Resolution:** Phase 3 delivers `tt-kernel-capture` CLI with
  `inspect` and `replay` subcommands. Phase 5 adds VS Code integration.
- **Effort Estimate:** 2-3 pw in Phase 3.

### GAP-TI-2: LightMetal Issue #24955 Rearchitecture Risk (MEDIUM)

- **Source:** Ch 3, Key Dependencies
- **Description:** Multiple LightMetal replay handlers are currently
  disabled or throwing (Issue #24955). The kernel capture system's FlatBuffer
  extensions must not conflict with the ongoing LightMetal rearchitecture.
- **Current Mitigation:** The `.ttksnap` format uses a standalone schema
  (`kernel_snapshot.fbs`) with file_identifier `"TKSN"` distinct from
  LightMetal's `"LMTL"`, reducing coupling.
- **PROPOSED Resolution:** Coordinate with LightMetal maintainers before
  Phase 1 implementation.

### GAP-TI-3: Spike Licensing for Commercial Use (LOW)

- **Source:** Ch 5, File 3; Ch 7, File 1
- **Description:** Spike is BSD-licensed (permissive). Embedding it in a
  Tenstorrent product requires legal review.
- **PROPOSED Resolution:** Legal review during Phase 3. BSD-3-Clause is
  straightforward for commercial use.

---

## 6. Gap Priority Summary

| ID | Gap | Severity | P1 | P2 | P3 | Composite | Phase | Effort (pw) |
|----|-----|----------|----|----|----|-----------|-------|-------------|
| GAP-SC-1 | No L1 memory snapshot | CRITICAL | 5 | 5 | 5 | **5.00** | 2 | 2.5 |
| GAP-SC-2 | No ELF binary capture | CRITICAL | 4 | 3 | 5 | **3.95** | 0-1 | 2.5 |
| GAP-DI-1 | No GDB stub for Tensix | CRITICAL | 5 | 2 | 4 | **3.65** | 4 | 4-6 |
| GAP-DI-2 | Debug reg accessibility | CRITICAL | -- | -- | -- | -- | Pre-4 | VE-1 |
| GAP-SC-4 | Semaphore runtime state | HIGH | 3 | 4 | 4 | **3.65** | 2 | incl. |
| GAP-SC-3 | No single-kernel granularity | HIGH | 4 | 3 | 3 | **3.35** | 1 | 0.5 |
| GAP-SC-5 | NOC transaction history | HIGH | 2 | 4 | 4 | **3.30** | 4 | 2-3 |
| GAP-DI-3 | No Tensix functional emulator | HIGH | 3 | 3 | 2 | **2.70** | 3 | 6-8 |
| GAP-DI-4 | WH lacks RISC_DBG_CNTL | HIGH | -- | -- | -- | -- | 4 | 1-2 |
| GAP-RF-1 | SFPU/SFPI emulation gap | HIGH | -- | -- | -- | -- | 3 | Accept |
| GAP-RF-2 | CB hardware semantics | HIGH | -- | -- | -- | -- | 3 | 2-3 |
| GAP-SC-6 | Hardware register state | MEDIUM | -- | -- | -- | -- | 4 | 1-2 |
| GAP-SC-7 | Missing config fields | MEDIUM | -- | -- | -- | -- | 1 | 0.5 |
| GAP-SC-8 | Launch message capture | MEDIUM | -- | -- | -- | -- | 1 | incl. |
| GAP-DI-5 | Non-interactive debug tools | MEDIUM | -- | -- | -- | -- | All | N/A |
| GAP-DI-6 | Debug symbol quality | MEDIUM | -- | -- | -- | -- | 0 | 0.5-1 |
| GAP-RF-3 | Non-deterministic replay | MEDIUM | -- | -- | -- | -- | Accept | N/A |
| GAP-RF-4 | Dispatch timeout during debug | MEDIUM | -- | -- | -- | -- | 4 | 1 |
| GAP-TI-1 | No snapshot CLI tool | MEDIUM | -- | -- | -- | -- | 3 | 2-3 |
| GAP-TI-2 | LightMetal rearch risk | MEDIUM | -- | -- | -- | -- | 1 | Coord. |
| GAP-SC-9 | No versioning metadata | LOW | -- | -- | -- | -- | 1 | 0.5 |
| GAP-TI-3 | Spike licensing | LOW | -- | -- | -- | -- | 3 | Legal |

> **Critical path:** GAP-SC-2 (Phase 0-1) -> GAP-SC-1 (Phase 2) -> GAP-DI-3
> (Phase 3, host model) -> GAP-DI-2 validation -> GAP-DI-1 (Phase 4).
> The host functional model path (Tier 1) avoids the hardware debug
> register dependency entirely and delivers value earliest.

---

## 7. Gap Interaction Matrix

Some gaps compound each other. The following interactions are noteworthy:

| Gap A | Gap B | Interaction |
|-------|-------|-------------|
| GAP-SC-2 (no binaries) | GAP-DI-6 (debug symbols) | Even after capturing binaries, without debug symbols (-g), GDB cannot map addresses to source lines. Both must be resolved for useful debugging. |
| GAP-SC-1 (no L1 snapshot) | GAP-SC-4 (no semaphore state) | Semaphore values live in L1; resolving GAP-SC-1 (L1 snapshot) implicitly resolves GAP-SC-4 if semaphore addresses are known (from `ProgramConfig::sem_offset`). |
| GAP-DI-2 (debug reg accessibility) | GAP-DI-1 (no GDB stub) | The GDB stub architecture depends on which debug register interface is available. If GAP-DI-2 is resolved negatively (no PCIe access to debug regs), the GDB stub must use the ebreak+mailbox protocol exclusively. |
| GAP-DI-3 (no emulator) | GAP-RF-1 (SFPU gap) | Both point to the same root cause: Tensix custom hardware has no software model. The host functional model (trisc.cpp) is the pragmatic resolution for both. |
| GAP-SC-5 (no NOC log) | GAP-RF-3 (non-determinism) | Even with NOC transaction capture, replay of NOC timing is non-deterministic. The transaction log provides ordering but not exact cycle timing. |
| GAP-RF-2 (CB semantics) | GAP-SC-4 (semaphore state) | CB tile counters and semaphores both implement cross-RISC synchronization. The software CB model in the replay engine must handle both consistently. |

---

## 8. Architecture-Specific Gap Summary

| Gap | Wormhole B0 | Blackhole | Quasar |
|-----|-------------|-----------|--------|
| L1 snapshot (GAP-SC-1) | ~1 MB L1, PCIe readback | ~1.5 MB L1, PCIe readback | ~4 MB L1, readback unverified |
| Debug registers (GAP-DI-2) | **Not present** (no `RISC_DBG_CNTL` defines in `wormhole/tensix.h`) | `RISC_DBG_CNTL_0/1` defined, active macros (`tensix.h:162-163`), host accessibility unverified; raw register interface only (no `rvdbg_cmd` API) | Full `rvdbg_cmd` software API via `ckernel_riscv_debug.h`; `RISC_DBG_CNTL` defines commented out in `tensix.h:175-178` (uses Debug Module struct instead); host accessibility unverified |
| GDB stub (GAP-DI-1) | ebreak+mailbox only | ebreak+mailbox or debug registers | rvdbg_cmd or Debug Module |
| Breakpoints | 4 per thread via `BREAKPOINT_CTRL` | 4 per thread via `BREAKPOINT_CTRL` | 4 per thread via `BREAKPOINT_CTRL` (`tensix.h:240`) + 8 watchpoints (HW_WCHPT0-7); note: convenience functions in `tensix_functions.h` guarded by `#ifndef ARCH_QUASAR` |
| SFPU emulation (GAP-RF-1) | Same gap across all | Same gap across all | 4 compute processors per Tensix engine amplify complexity |
| Snapshot size | ~50 KB - 1 MB per core | ~50 KB - 1.5 MB per core | ~50 KB - 4 MB per core |
| NEO cluster (Quasar only) | N/A | N/A | 4 engines x 4 procs = 16 per cluster; multiplies capture effort |

---

## 9. Coverage Analysis: LightMetal vs. Interceptor vs. Full Replay

| State Category (Ch 2) | LightMetal | Interceptor (.ttksnap) | Full Replay | Gap |
|----------------------|------------|----------------------|-------------|-----|
| 1. Kernel ELF binaries | File path only | Full ELF with debug symbols | Same | SC-2 |
| 2. Compile-time args/defines | Partial | Full (defines, named args, hlk_desc) | Same | SC-2 |
| 3. Runtime args | Yes | Yes (per-core, per-processor) | Same | -- |
| 4. CB configuration | Yes | Yes + allocated L1 addresses | Same | -- |
| 5. CB tile data (L1 contents) | **No** | Yes (targeted or full L1 dump) | Same | SC-1 |
| 6. Semaphore state | **No** | Initial + runtime values | Same | SC-4 |
| 7. Launch/go messages | **No** | Full `launch_msg_t` + `go_msg_t` | Same | SC-8 |
| 8. NOC transaction history | **No** | **No** (optional in Phase 4) | Yes | SC-5 |
| 9. Hardware register state | **No** | SRCA/SRCB/DEST via debug array (BH/Quasar) | Full register file | SC-6 |
| 10. Environmental | **No** | Yes (`CaptureMetadata`) | Same | -- |

**Quantitative summary:**
- LightMetal: ~34% coverage (weighted, per Ch 3)
- Interceptor (Phase 1+2): ~93% coverage (per Ch 6)
- Full Replay (Phase 4+): ~98% coverage (NOC history and full register state)

---

## 10. Developer Workflow Pain Analysis

### Scenario 1: "Matmul produces NaN on core (3,4)"

**Today (without interceptor):** Developer iterates through DPRINT insertions,
attempts standalone test, fails to reproduce due to missing CB/semaphore state.
**Total: ~2 hours.**

**With interceptor (Phases 0-3):** Capture on assert, replay under GDB, find
root cause. **Estimated: 15-30 minutes.** Time savings: 70-90%.

### Scenario 2: "Intermittent hang in multi-core eltwise"

**Today:** Developer waits for hang reproduction (~60 min), adds DPRINT for
semaphore values, cannot determine which core should have posted the semaphore,
fails to reproduce deterministically. **Total: ~5.5 hours.**

**With interceptor:** Automatic capture on hang detection, semaphore values in
snapshot, replay under GDB. **Estimated: 30-60 minutes.** Time savings: 80-90%.

### Scenario 3: "Numerical mismatch in new LLK op"

**Today:** Developer adds DPRINTs after each LLK call, writes host-side
reference implementation, compares step-by-step against DPRINT output.
**Total: ~3.5 hours.**

**With interceptor:** Capture, replay in host functional model under GDB, set
breakpoints in LLK source, inspect `ckernel::regfile`. **Estimated: 20-45
minutes.** Time savings: 70-85%.

---

## 11. Industry Comparison

| Capability | cuda-gdb | rocgdb | TT-Metal (Current) | TT-Metal (Post-Phase 3) |
|-----------|---------|--------|-------------------|------------------------|
| Breakpoints | Hardware, per-warp | Software, per-wavefront | None (ASSERT only) | Hardware (BH/Quasar), software (emulator) |
| Single-step | Yes, warp-level | Yes, wavefront-level | No | Yes (emulator, host model) |
| Register inspect | Full register file | Full register file | Watcher waypoints only | Full via GDB (emulator), partial via debug array (BH/Quasar) |
| Memory inspect | Global/shared/local | Global/LDS | DPRINT (manual) | Full L1 snapshot + GDB memory commands |
| Kernel isolation | Per-kernel launch | Per-dispatch | No | `.ttksnap` per-invocation |
| Failure capture | Coredump on GPU exception | ROCm error report | Watcher assert + DPRINT | Automatic `.ttksnap` on failure |
| Offline replay | NVIDIA Nsight captures | RGP capture | LightMetal (~34%) | `.ttksnap` (~93%) |

---

## Key Takeaways

- **Four CRITICAL gaps** form the blocking path: ELF binary capture (GAP-SC-2),
  L1 memory snapshot (GAP-SC-1), GDB stub implementation (GAP-DI-1), and
  debug register host accessibility validation (GAP-DI-2). The first
  two are resolved by Phase 1-2 interceptor work; the latter two are Phase 4
  and contingent on hardware validation.

- **The host functional model (Tier 1) bypasses 5 of the 22 gaps entirely**
  (GAP-DI-1, GAP-DI-2, GAP-DI-4, GAP-RF-4, and partially GAP-RF-1),
  making it the highest-value, lowest-risk starting point. This aligns with
  Ch 5's three-tier debug strategy recommendation.

- **The highest-impact gap (GAP-SC-1, L1/CB snapshot) has the highest
  composite pain score (5.00) and is tractable within 2.5 person-weeks.**
  It is also the single highest-leverage gap to close because it unblocks both
  emulator-based replay and host-model replay, and its infrastructure is
  reused by semaphore capture (GAP-SC-4).

- **Five of seven top-priority gaps are closed by Phases 0-2 (9-14 pw total),
  delivering ~93% state coverage.** The remaining gaps (GDB stub, functional
  emulator) require more effort but are needed to make captured state
  interactively debuggable.

- **Architecture differences materially affect gap closure strategy.** WH
  requires ebreak+mailbox for on-hardware GDB; BH provides raw RISC_DBG_CNTL
  registers (but no rvdbg_cmd API); Quasar has the full rvdbg_cmd software API
  with 8 watchpoints. The implementation must account for all three paths.

- **The three debugging scenarios analyzed show potential time savings of
  70-90%** (from hours to minutes) once Phases 0-3 are complete.
