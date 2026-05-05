# Phased Implementation Plan with Risk Registers and Value Milestones

This file presents a six-phase implementation plan (Phases 0-5) for the
runtime interceptor and replay debugging system, with per-phase risk
registers, mitigation strategies, go/no-go criteria, dependencies, effort
estimates, and value-unlocked milestones. Total estimated effort: 20-29
person-weeks for Phases 0-4. All source citations reference commit `621b949`.

> **Cross-references:** Ch 2 (state inventory, capture envelope sizes),
> Ch 3 (LightMetal ~34% coverage, 8-phase incremental adoption path),
> Ch 4 (H1+H4 dual-hook architecture, ~2 us capture overhead),
> Ch 5 (three-tier debug strategy, trisc.cpp host model),
> Ch 6 (interceptor design, `.ttksnap` schema, capture modes),
> Ch 7 (three replay targets), `01_gap_analysis.md` (22 gaps, 4 CRITICAL).

---

## 1. Phase Overview

| Phase | Title | Effort (pw) | Key Deliverable | Primary Gaps Addressed | Developer Capability Unlocked |
|-------|-------|-------------|-----------------|----------------------|-------------------------------|
| 0 | Foundation | 2-3 | Debug-ready build infrastructure | GAP-DI-6 | "I can find the exact binary that crashed" |
| 1 | Minimal Capture | 4-6 | `.ttksnap` files with binaries + config | GAP-SC-2, GAP-SC-3, GAP-SC-7, GAP-SC-8, GAP-SC-9 | "I can capture a specific kernel's full config" |
| 2 | L1 Snapshot and Failure Capture | 4-6 | L1 memory dumps, Watcher-triggered capture | GAP-SC-1, GAP-SC-4 | "I can capture the exact state at failure time" |
| 3 | Replay Engine | 6-8 | `tt-kernel-capture` CLI, host model replay | GAP-DI-3, GAP-RF-1, GAP-RF-2, GAP-TI-1 | "I can step through my kernel under GDB" |
| 4 | Multi-Core and Hardware Replay | 4-6 | Multi-RISC replay, hardware GDB stub | GAP-DI-1, GAP-DI-4, GAP-SC-5, GAP-SC-6 | "I can debug on real hardware with GDB" |
| 5 | Integration and UX | Ongoing | VS Code adapter, CI integration | GAP-DI-5, GAP-TI-2 | "Debugging is part of my normal workflow" |
| | **Total Phases 0-4** | **20-29** | | | |

---

## 2. Assignee Profiles

| Profile | Abbreviation | Skills Required |
|---------|-------------|-----------------|
| **Runtime Engineer** | Runtime Eng | TT-Metal dispatch path (`dispatch.cpp`, `program_impl.hpp`, `HWCommandQueue`), L1 memory layout, UMD device read/write APIs, LightMetal capture infrastructure, JIT compilation pipeline. |
| **Debug Tooling Engineer** | Debug Eng | GDB internals (RSP protocol, DWARF parsing), RISC-V ISA and debug specification, emulator integration (Spike plugin model), `ckernel_riscv_debug.h` and `tensix_functions.h` debug register interfaces, `trisc.cpp` host model. |
| **Infrastructure Engineer** | Infra Eng | FlatBuffer schema design and code generation, CLI tool development, CI/CD integration, compression libraries (LZ4/zstd), VS Code extension development, storage management. |

---

## 3. Phase 0 -- Foundation (2-3 pw)

### 3.1 Objectives

Ensure that every JIT-compiled kernel binary retains debug symbols and that
the build infrastructure supports the interceptor's requirements, without
changing any runtime behavior.

### 3.2 Tasks

| ID | Task | Owner | Effort | Depends On |
|----|------|-------|--------|------------|
| P0-T1 | Force Debug Symbols in Capture Mode: When `TT_METAL_KERNEL_CAPTURE` is set, modify `JitBuildOptions` to force `-g` and consider `-Og` instead of `-O2`. Currently `TT_METAL_RISCV_DEBUG_INFO` at `rtoptions.hpp:204` defaults to `false`. When Inspector is active and env var is not explicitly set, auto-enable (partially supported at `rtoptions.cpp:1332-1334`). | Runtime Eng | 0.5 pw | -- |
| P0-T2 | ELF Retention in Capture Cache: Create `$TT_METAL_HOME/generated/kernel_captures/` directory. Hard-link or copy each compiled ELF with metadata encoding `<program_id>_<build_key>_<processor>.elf`. Implement age-based eviction (keep last 100, configurable via `TT_METAL_CAPTURE_RETENTION`). | Infra Eng | 0.75 pw | P0-T1 |
| P0-T3 | Inspector Compile-State Recording: Extend `Inspector::program_kernel_compile_finished` (`inspector.hpp:39-43`) to record: full defines map (`Kernel::defines_`), compile args, named compile args, `hlk_desc`, build key hash, source path, optimization level. Store as JSON sidecar. | Debug Eng | 0.5 pw | P0-T2 |
| P0-T4 | DWARF Quality Audit: Compile 5 representative kernels (matmul, eltwise_unary, eltwise_binary, reduce, softmax) with `-g -O0` using `riscv-tt-elf-g++` (path: `build.cpp:119`). Load each into `riscv-tt-elf-gdb`. Test breakpoints on `llk_unpack_AB`, `llk_math_matmul`, `llk_pack`. Document results. | Debug Eng | 0.5 pw | P0-T1 |

**Phase 0 Total: 2.25 pw**

### 3.3 Risk Register

| ID | Risk | Probability | Impact | Severity | Mitigation |
|----|------|------------|--------|----------|------------|
| R0-1 | Debug symbols increase build cache size by 3-5x | High | Medium | HIGH | Make debug retention opt-in via env var; default off. Use `-Og` which is smaller than `-O0 -g`. Add cache eviction policy based on total size. |
| R0-2 | `-Og` produces different behavior than `-O2` | Medium | Low | LOW | Document known differences. `-Og` is designed for debugging with minimal behavioral changes. |
| R0-3 | Inspector extension conflicts with existing consumers | Low | Medium | LOW | Additive-only changes to Inspector API. New fields are optional. |

### 3.4 Go/No-Go Criteria for Phase 1

- [ ] `riscv-tt-elf-gdb` can load a JIT-compiled ELF with debug symbols and display LLK source line numbers.
- [ ] `Kernel::binaries(build_key)` (`kernel.hpp:197`) returns valid `ll_api::memory` objects with complete binary data.
- [ ] `TT_METAL_KERNEL_CAPTURE` env var is parsed without errors.
- [ ] Build cache size with debug symbols is measured and within 2x of baseline.

### 3.5 Value Unlocked

> **Before:** When a kernel asserts, the developer must manually enable debug
> symbols, recompile (potentially producing a different binary), and hope the
> bug reproduces.
>
> **After:** The exact ELF binary that crashed is automatically preserved with
> full debug symbols and build metadata. The developer can immediately load it
> into `riscv-tt-elf-gdb` for disassembly and source mapping.

---

## 4. Phase 1 -- Minimal Capture (4-6 pw)

### 4.1 Objectives

Implement the core interceptor that produces self-contained `.ttksnap`
snapshot files capturing compiled binaries, runtime configuration, and
CB/semaphore configuration (but not yet L1 memory contents) at kernel
dispatch time.

### 4.2 Tasks

| ID | Task | Owner | Effort | Depends On |
|----|------|-------|--------|------------|
| P1-T1 | `kernel_snapshot.fbs` FlatBuffer Schema: Implement the schema from Ch 6, File 3: `CaptureMetadata`, `KernelBinaryBlob`, `CircularBufferSnapshot` (config only), `SemaphoreSnapshot` (initial values), `RuntimeArgsSnapshot`, `ProgramConfigSnapshot`, `TensixCoreSnapshot`, `KernelInvocationSnapshot` root type with file_identifier `"TKSN"`. | Infra Eng | 1 pw | -- |
| P1-T2 | Primary Hook at Site B: Insert capture call in `program_dispatch::assemble_device_commands()` at `tt_metal/impl/program/dispatch.cpp:1890`. The hook iterates `ProgramImpl::get_kernel_groups()`, reads `Kernel::binaries_`, serializes `ProgramConfig`, CB configs, semaphore initial values, runtime args, `launch_msg_t`, and `go_msg_t`. | Runtime Eng | 2 pw | P1-T1 |
| P1-T3 | Capture Target Selector / `CaptureFilter`: Pattern-matching filter supporting kernel name glob, program ID, core coordinate, dispatch count. Env var: `TT_METAL_KERNEL_CAPTURE_PATTERN`, `_PROGRAM`, `_CORE`. Plus programmatic API: `program.enable_capture()`. | Runtime Eng | 1 pw | P1-T2 |
| P1-T4 | Snapshot Inspection CLI: `tt-kernel-capture inspect <path.ttksnap>` that reads a snapshot and prints: program ID, kernel names, architecture, core coordinates, binary sizes, CB configs, runtime arg counts, capture timestamp, git hash. | Infra Eng | 1 pw | P1-T1 |
| P1-T5 | Binary Deduplication: Hash-based dedup (FNV1a or xxHash) for ELF binary deduplication within a snapshot. Shared-binary kernel groups store the binary once with references. Reduces 64-core snapshot from ~64x to ~1x binary storage. | Infra Eng | 0.5 pw | P1-T1 |

**Phase 1 Total: 5.5 pw**

### 4.3 Risk Register

| ID | Risk | Probability | Impact | Severity | Mitigation |
|----|------|------------|--------|----------|------------|
| R1-1 | Dispatch path perturbation causes timing-sensitive tests to fail | Medium | High | HIGH | Compile-time gating via `TT_ENABLE_KERNEL_CAPTURE` (default off in release). ~2 us inline overhead (Ch 4 Key Finding 5) is below dispatch noise floor. Background serialization thread for I/O. |
| R1-2 | `Kernel::binaries(build_key)` returns empty for certain kernel types | Medium | High | HIGH | Phase 0 P0-T2 ensures ELF retention. Add assertion that binary data is non-empty. Fall back to recording `file_name` path if unavailable. |
| R1-3 | FlatBuffer schema conflicts with LightMetal rearchitecture (Issue #24955) | Medium | Medium | MEDIUM | Use standalone schema file (`kernel_snapshot.fbs`) with distinct namespace and file_identifier (`"TKSN"` vs `"LMTL"`). Coordinate with LightMetal team. |
| R1-4 | Thread-safety issues with multi-CQ dispatch | Low | High | HIGH | Use `thread_local` depth counter (same pattern as `TraceScope`). Primary hook is called under the command queue mutex. Add mutex to shared snapshot output buffer. |
| R1-5 | Snapshot sizes exceed storage budget for CI | Medium | Medium | MEDIUM | Default to MetadataOnly mode (~300 B per dispatch). `TT_METAL_KERNEL_CAPTURE_MAX=100` caps total captures. Total-size budget default 512 MB. |

### 4.4 Go/No-Go Criteria for Phase 2

- [ ] Produce a valid `.ttksnap` file for a single-core eltwise compute kernel containing: 3 TRISC ELF binaries, CB configurations, runtime args, semaphore initial values, launch message, and capture metadata.
- [ ] FlatBuffer verifier confirms all fields non-null.
- [ ] Binary deduplication reduces 64-core matmul snapshot to <400 KB (Ch 6 estimate: ~384 KB).
- [ ] Capture overhead <5 us per dispatch for MetadataOnly mode.
- [ ] No test regressions in TT-Metal CI when compiled in but env var unset.

### 4.5 Value Unlocked

> **Before:** When a bug is reported, the developer has no self-contained record
> of what was running. They must reproduce the full workload from scratch.
>
> **After:** A `.ttksnap` file captures the complete kernel configuration --
> binaries, args, CB config, semaphore initial values -- for any targeted kernel
> invocation. Shareable, attachable to bug reports, inspectable on any machine.

---

## 5. Phase 2 -- L1 Snapshot and Failure-Triggered Capture (4-6 pw)

### 5.1 Objectives

Add the ability to capture actual L1 memory contents (tile data in circular
buffers, semaphore runtime values) and implement failure-triggered automatic
capture via Watcher integration.

### 5.2 Tasks

| ID | Task | Owner | Effort | Depends On |
|----|------|-------|--------|------------|
| P2-T1 | L1 Memory Readback: Read L1 regions via UMD `read_from_device()` or PCIe TLB-mapped reads. Capture CB data regions (addresses from `CircularBufferConfig::address()`, sizes from `total_size()`), semaphore values (from `ProgramConfig::sem_offset`). Sparse representation: only non-zero L1 pages. | Runtime Eng | 1.5 pw | P1-T2 |
| P2-T2 | Sparse L1 and LZ4 Compression: `L1RegionSnapshot` with explicit `(start_address, size, data)` tuples and LZ4 compression per region. Target: single-core matmul under 200 KB compressed vs 1,500 KB full L1. | Infra Eng | 1 pw | P2-T1 |
| P2-T3 | Semaphore Runtime State: Read all 16 semaphore L1 addresses per core. Store in `SemaphoreSnapshot` with `value_at_capture`. Compute "unlocked" semaphore vector for single-TRISC replay. | Runtime Eng + Debug Eng | 1 pw | P2-T1 |
| P2-T4 | Failure-Triggered Capture: Integrate with `WatcherServer::killed_due_to_error()` (`watcher_server.hpp:34`). On assert/timeout, auto-capture post-mortem L1 state. Tag as `capture_mode: FailureTriggered`. | Runtime Eng | 1 pw | P2-T1 |
| P2-T5 | `CAPTURE_BREAKPOINT()` Device-Side Macro: Write sentinel to `debug_ring_buf_msg_t` data[31] mailbox. Watcher polling detects sentinel and initiates deep L1 snapshot. Kernel spins awaiting host resume. | Runtime Eng | 0.5 pw | P2-T1 |
| P2-T6 | Rolling Ring Buffer: FIFO of last N lightweight snapshots (MetadataOnly) in memory. On trigger, flush to disk alongside deep snapshot. 512 MB cap with FIFO eviction. | Runtime Eng | 0.5-1 pw | P1-T2 |

**Phase 2 Total: 5.5-6 pw**

### 5.3 Risk Register

| ID | Risk | Probability | Impact | Severity | Mitigation |
|----|------|------------|--------|----------|------------|
| R2-1 | L1 readback during active dispatch corrupts data | High | High | CRITICAL | Read L1 before writing the go signal (capture at Site B is before `write_program_command_sequence`). For failure capture, core is already halted. Add memory barrier verification. |
| R2-2 | L1 readback latency (~75 ms for full 1.5 MB) blocks dispatch | High | High | CRITICAL | Background L1 read on separate thread. Dispatch path records capture request and continues. Latency acceptable for failure-triggered (post-mortem). |
| R2-3 | `CAPTURE_BREAKPOINT()` sentinel conflicts with Watcher mailbox values | Medium | Medium | MEDIUM | Use distinct sentinel (e.g., `0xCAPB0000`). Add to Watcher's known-values table. Test with Watcher enabled/disabled. |
| R2-4 | Ring buffer memory exceeds budget under high dispatch rate | Medium | Medium | MEDIUM | 512 MB hard cap. At 300 B per MetadataOnly: ~1.7M snapshots. Auto-downgrade if pressure detected. |
| R2-5 | Post-mortem L1 contains corrupted data from failing kernel | High | Medium | HIGH | Document that post-mortem reflects failure state. Pair with ring buffer's pre-failure history for context. |

### 5.4 Go/No-Go Criteria for Phase 3

- [ ] `.ttksnap` with L1 data for a single-core matmul; tile data matches expected input values.
- [ ] Failure-triggered capture fires on `TT_ASSERT` and produces valid post-mortem snapshot.
- [ ] `CAPTURE_BREAKPOINT()` pauses kernel and allows host L1 capture without device reset.
- [ ] L1 readback latency <100 ms for full BH L1 (1.5 MB) via PCIe.
- [ ] Ring buffer retains last N snapshots and flushes on trigger.

### 5.5 Value Unlocked

> **Before:** When a kernel crashes, the developer knows *that* it crashed (from
> Watcher) and *what assert fired*, but not *what data was being processed* or
> *what the semaphore state was*. The data is gone by the time they investigate.
>
> **After:** The system automatically captures complete L1 state at failure time.
> `CAPTURE_BREAKPOINT()` enables "printf-free" state inspection at any kernel point.
> Rolling ring buffer provides "time-travel" context.

**Cumulative state coverage:** ~93% of Ch 2 inventory.

---

## 6. Phase 3 -- Replay Engine (6-8 pw)

### 6.1 Objectives

Build the replay system that loads `.ttksnap` files and enables GDB-style
interactive debugging. Tier 1 (host functional model) is the primary target;
Spike-based emulator replay is secondary.

### 6.2 Tasks

| ID | Task | Owner | Effort | Depends On |
|----|------|-------|--------|------------|
| P3-T1 | Snapshot Loader Library: Deserialize `.ttksnap` into `ReplayContext` with per-core binary blobs, L1 regions, CB metadata, semaphore vectors, runtime args, launch messages. | Infra Eng | 1 pw | P1-T1, P2-T2 |
| P3-T2 | Host Functional Model Replay: Adapt `trisc.cpp` infrastructure (`tt-llk/tests/helpers/src/trisc.cpp`). Extract kernel source + defines from snapshot, compile with host `g++` against LLK stub headers, link against "virtual L1" library loaded from snapshot. The host model uses `ckernel::regfile` as host-side register array, calls `run_kernel(__runtime_args_start)`, writes `ckernel::KERNEL_COMPLETE` to mailbox on completion. | Debug Eng | 2 pw | P3-T1, P0-T3 |
| P3-T3 | Host Model GDB Integration: `--gdb` flag launches compiled kernel under host GDB with breakpoint at `run_kernel`. Auto-generate `.gdbinit` with `symbol-file`, common LLK breakpoints, tile data pretty-printers. | Debug Eng | 0.5 pw | P3-T2 |
| P3-T4 | Spike Emulator with Tensix MMIO Plugin: Custom plugin mapping L1 address range (`0x0` to `0x180000` for WH) to snapshot data. CB pointer semantics: intercept writes to CB control addresses, maintain read/write pointers. Semaphore stubs return "unlocked" values. SFPU stub: trap and NOP-replace (control-flow-only). | Debug Eng | 3 pw | P3-T1 |
| P3-T5 | Spike GDB Integration: Connect Spike's built-in GDB stub to `riscv-tt-elf-gdb`. Verify breakpoints, single-step, register/memory read. Add convenience launch script. | Debug Eng | 0.5 pw | P3-T4 |
| P3-T6 | `tt-kernel-capture` CLI: Unified CLI with `inspect`, `validate`, `replay`, and `list` subcommands. `replay` supports `--mode={host-model|spike|hardware}`, `--core`, `--gdb`, `--gdb-port`. Phase 3 implements `host-model` and `spike`; `hardware` is a stub. | Infra Eng | 1 pw | P3-T2, P3-T4 |

**Phase 3 Total: 8 pw**

### 6.3 Risk Register

| ID | Risk | Probability | Impact | Severity | Mitigation |
|----|------|------------|--------|----------|------------|
| R3-1 | Host model diverges from hardware for LLK math | High | Medium | HIGH | Document known divergences (float precision, denorm, block-float rounding). Add comparison mode. Accept functional-level correctness for control-flow debugging. |
| R3-2 | `trisc.cpp` tightly coupled to LLK test harness | Medium | High | HIGH | Factor out minimal runtime (`regfile`, `mailbox_t`, virtual L1) into standalone `libttreplay`. Estimated 2 pw of phase budget. |
| R3-3 | Spike traps on first SFPI instruction | High | Medium | HIGH | Add trap handler that logs and NOP-replaces SFPI instructions. Label Spike replay as "control-flow-only." Prefer host model for math debugging. |
| R3-4 | Snapshot contains insufficient state for replay | Medium | High | HIGH | `validate` subcommand checks every Ch 2 category. Fail gracefully with clear messages. |
| R3-5 | GDB symbol resolution fails for JIT kernels | Medium | Medium | MEDIUM | Snapshot records ELF path. `tt-kernel-capture replay` searches build cache by hash. Add `--elf-dir` override. |

### 6.4 Go/No-Go Criteria for Phase 4

- [ ] Developer can capture a failing matmul TRISC1 kernel, run `tt-kernel-capture replay --gdb`, and set a breakpoint in `llk_math_eltwise_unary()` with source-level stepping.
- [ ] Spike-based replay runs a scalar RISC-V kernel (no SFPI) to completion with correct results. GDB breakpoints work.
- [ ] Software CB model correctly coordinates TRISC0/1/2 for unpack->math->pack pipeline in host model.
- [ ] `tt-kernel-capture inspect` displays snapshot contents.
- [ ] End-to-end: capture on machine A, copy `.ttksnap` to machine B (no hardware), replay under GDB.

### 6.5 Value Unlocked -- The "15-Minute Bug" Test

> **Before:** The developer has a `.ttksnap` file with complete kernel state
> but can only inspect it. To understand *why* the kernel produced bad output,
> they must still reason statically or add DPRINTs.
>
> **After:** The developer can replay the captured kernel under GDB and
> interactively debug -- breakpoints in LLK source, single-step through
> unpack/math/pack, inspect tile values and local variables.

**Phase 3 acceptance criterion:** A kernel developer who hits a matmul assert
can go from "I see the assert in Watcher output" to "I have identified the
root cause in GDB" in under 15 minutes, without re-running the workload,
without modifying kernel source, and without exclusive hardware access.

---

## 7. Phase 4 -- Multi-Core and Hardware Replay (4-6 pw)

### 7.1 Objectives

Extend the replay system to support multi-RISC coordination (all 5 RISCs on
a single Tensix core) and implement hardware GDB stub for on-device debugging.
This phase is contingent on the GAP-DI-2 validation experiment.

### 7.2 Tasks

| ID | Task | Owner | Effort | Depends On |
|----|------|-------|--------|------------|
| P4-T1 | Hardware Accessibility Experiment (VE-1): From host, write to `RISC_DBG_CNTL_0` via UMD. Read `RISC_DBG_STATUS_0/1`. Determine host access feasibility for BH and Quasar. | Runtime Eng + Debug Eng | 1 pw | Hardware access |
| P4-T2 | GDB RSP Server: Minimal server handling `?`, `g/G`, `m/M`, `c`, `s`, `Z0/z0`, `Z1/z1`, `qSupported`, `qfThreadInfo`. Pluggable backend: `DebugTarget::read_register()`, `write_register()`, `read_memory()`, `write_memory()`, `halt()`, `resume()`, `step()`. | Debug Eng | 2 pw | P4-T1 |
| P4-T3 | Architecture-Specific Backends: **BH:** Map RSP to `RISC_DBG_CNTL_0/1` register protocol (halt->PAUSE, resume->CONTINUE, step->STEP, read_register->set CNTL_0 with reg_addr and reg_wr=0 then read STATUS_1). **Quasar:** Map to `rvdbg_cmd` enum. If P4-T1 negative: implement BRISC-based debug agent receiving commands via L1 mailbox. **WH:** ebreak+mailbox protocol. | Debug Eng | 2-3 pw | P4-T1, P4-T2 |
| P4-T4 | Multi-RISC Host Model Replay: Extend host model to run all 5 RISCs as threads with software CB model for synchronization. Use `launch_msg_t.enables` bitmask for active processors. | Debug Eng | 1 pw | P3-T2 |
| P4-T5 | NOC Transaction Log Integration: Capture `NocWriteEvent`/`NocReadEvent` from `noc_debugging.hpp` when `TT_METAL_NOC_DEBUG=1` alongside capture mode. Note: event structs use `int8_t` coordinates and `uint32_t` addresses; timestamps are NOT in the structs but passed separately to handlers. Serialize into `NocTransactionLog` in `.ttksnap`. | Runtime Eng | 0.5-1 pw | P2-T1 |

**Phase 4 Total: 4-6 pw** (within spec range when deferring features to keep scope bounded)

**Scope adjustment based on VE-1:**
- VE-1 positive (all tests pass, <100 us latency): Full debug register backends. 4-5 pw.
- VE-1 partial (PAUSE works, RD_REG fails): Hybrid halt + L1 save/restore. ~5 pw.
- VE-1 negative (PCIe cannot reach debug registers): ebreak+mailbox for all architectures. Reduce P4-T3 to ~1.5 pw. Total 3.5-4 pw.

### 7.3 Risk Register

| ID | Risk | Probability | Impact | Severity | Mitigation |
|----|------|------------|--------|----------|------------|
| R4-1 | Quasar debug registers not accessible from host PCIe | Medium | High | CRITICAL | **Go/No-Go gate.** If VE-1 fails, drop debug register path. Fall back to ebreak+mailbox. Reduces scope but limits to cooperative single-stepping. |
| R4-2 | Halting a core causes dispatch timeout | High | High | CRITICAL | "Debug mode" disabling Watcher timeouts, extending mailbox polling to infinity, exclusive device access. `TT_METAL_DEBUG_SESSION=1` env var. |
| R4-3 | ebreak+mailbox single-step latency too high | High | Medium | HIGH | Estimate: ~10-50 us per step via mailbox round-trip. At 50 us/step, 1000 instructions = 50 ms -- acceptable. If >100 us, batch with breakpoint-and-continue. |
| R4-4 | Multi-RISC replay deadlocks | Medium | High | HIGH | Deadlock detection (timeout on CB wait). `--timeout-ms` flag. Log CB state at deadlock. Test with known-good kernels first. |
| R4-5 | NOC replay ordering differs from hardware | High | Medium | HIGH | Use "logical ordering" (happens-before from semaphore signals) not "physical ordering" (cycle-level). Document limitation (GAP-RF-3). |

### 7.4 Go/No-Go Criteria for Phase 5

- [ ] VE-1 experiment completed: host accessibility confirmed or denied; scope adjusted.
- [ ] Hardware GDB stub connects to single TRISC on real hardware, supports `break`, `continue`, `step`, `info registers`, `x` (memory examine).
- [ ] Multi-RISC host model runs 3-TRISC unpack->math->pack to completion with correct output.
- [ ] If VE-1 positive: single-step on hardware, inspect variable, verify match with host model.

### 7.5 Value Unlocked

> **Before:** GDB works in emulation and host model, but hardware-specific bugs
> require falling back to DPRINT.
>
> **After:** The developer can attach GDB to a real Tensix core on BH or Quasar,
> set hardware breakpoints, single-step, and inspect registers with full fidelity.

**Architecture-specific outcomes:**

| Architecture | On-Hardware GDB | Limitations |
|-------------|----------------|-------------|
| Wormhole B0 | ebreak+mailbox (software protocol) | Slower step (~1 ms per step), no hardware watchpoints |
| Blackhole | `RISC_DBG_CNTL` register protocol | Fast step (~10 us), 4 breakpoints per thread |
| Quasar | Full `rvdbg_cmd` + Debug Module | Full GDB feature set, 8 watchpoints (pending VE-1 validation) |

---

## 8. Phase 5 -- Integration and UX (Ongoing)

### 8.1 Deliverables

| ID | Deliverable | Description | Effort |
|----|-------------|-------------|--------|
| P5-D1 | VS Code Debug Adapter Protocol (DAP) | DAP adapter wrapping `tt-kernel-capture replay`. Launch config for `.ttksnap`. Source mapping, variable view, tile data visualization. | 2-3 pw |
| P5-D2 | CI/CD Failure Capture | `TT_METAL_KERNEL_CAPTURE=on-assert` in CI. Auto-upload `.ttksnap` as test artifact. `tt-kernel-capture replay --from-ci <run-id>`. | 1-2 pw |
| P5-D3 | Inspector/tt-exalens Integration | Serve capture data via Inspector RPC. Link from Inspector program view to `.ttksnap`. | 1 pw |
| P5-D4 | Snapshot Comparison Tool | `tt-kernel-capture diff snap1.ttksnap snap2.ttksnap` -- field-by-field diff highlighting runtime args, CB data, binary hashes. | 1 pw |
| P5-D5 | Documentation and Training | Developer guide, API reference, tutorials (single-core eltwise, multi-core matmul, failure triage). | 1 pw |

### 8.2 Risk Register

| ID | Risk | Probability | Impact | Severity | Mitigation |
|----|------|------------|--------|----------|------------|
| R5-1 | LightMetal rearch breaks schema compatibility | Medium | Medium | MEDIUM | Standalone `.ttksnap` with distinct identifier. Monitor Issue #24955. Schema migration tool. |
| R5-2 | CI capture storage costs escalate | Medium | Medium | MEDIUM | Default failure-triggered in CI (zero overhead until failure). Compress with zstd for archival. 30-day TTL, 1 GB per test suite. |
| R5-3 | Developer adoption low due to workflow complexity | Medium | High | HIGH | Single-command: `TT_METAL_KERNEL_CAPTURE=on-assert ./test_matmul`. Auto-open VS Code. Minimize configuration. |

---

## 9. Critical Path Analysis

```
Phase 0 (2-3 pw)
    |
    v
Phase 1 (4-6 pw) -----> [Go/No-Go: valid .ttksnap files]
    |
    v
Phase 2 (4-6 pw) -----> [Go/No-Go: L1 snapshots, failure capture]
    |                         |
    v                         v
Phase 3 (6-8 pw)         VE-1: debug reg validation (1 pw)
    |                         |
    v                         v
    [Go/No-Go: host model  [Go/No-Go: hardware
     replay works]          accessibility confirmed?]
    |                         |
    +----------+--------------+
               |
               v
         Phase 4 (4-6 pw) -- scope adjusted by VE-1 result
               |
               v
         Phase 5 (ongoing)
```

**Critical path for earliest debugging value:** Phase 0 -> Phase 1 ->
Phase 2 -> Phase 3. This delivers host functional model replay (Tier 1
debug) in 16-23 pw. The host model path does not depend on any hardware
validation experiments.

**Critical path for hardware debugging:** Phase 0 -> Phase 1 -> Phase 2 ->
VE-1 experiment -> Phase 4. VE-1 should start during Phase 2 to avoid
serializing the critical path.

---

## 10. Staffing Model

| Role | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|---------|---------|---------|---------|---------|
| Runtime engineer | 1 | 1 | 1 | 0.5 | 0.5 |
| Debug tooling engineer | 1 | 1 | 1 | 1 | 1 |
| LLK/kernel engineer | 0 | 0 | 0 | 1 | 0.5 |
| Hardware validation engineer | 0 | 0 | 0 | 0 | 0.5 |
| **Total headcount** | **2** | **2** | **2** | **2.5** | **2.5** |

**Minimum viable team:** 1 engineer through Phases 0-2 (sequential), expanding
to 2 for Phase 3. Delivers capture system and one replay target in ~15 pw elapsed.

**Recommended team:** 2 engineers starting at Phase 1 (runtime eng on hooks,
tooling eng on schema/CLI). Achieves 4-6 pw phase in ~3 calendar weeks.

---

## 11. Risk-Adjusted Timeline

| Phase | Optimistic | Expected | Pessimistic | Primary Risk |
|-------|-----------|----------|-------------|-------------|
| 0 | 1.5 pw | 2.5 pw | 4 pw | Build cache conflicts with JIT refactoring |
| 1 | 3 pw | 5 pw | 8 pw | FlatBuffer schema iteration; hook stability |
| 2 | 3 pw | 5 pw | 8 pw | L1 readback latency; Watcher integration |
| 3 | 5 pw | 7 pw | 10 pw | CB/semaphore stubbing; host model divergence |
| 4 | 3 pw | 5 pw | 8 pw | Quasar debug reg host access; WH ebreak stability |
| **Total (0-4)** | **15.5 pw** | **24.5 pw** | **38 pw** | |

The expected total of 24.5 pw is within the 20-29 pw spec range.

---

## 12. Effort Budget Reconciliation

| Item | Ch 3/5/6 Estimate | This Plan | Reconciliation |
|------|-------------------|-----------|----------------|
| LightMetal schema extensions | 25-39 pd (Ch 3) | Phase 1: 4-6 pw (20-30 pd) | Aligned. Standalone schema reduces integration overhead. |
| Host functional model | ~2 pw (Ch 5) | Phase 3: 2 pw of 6-8 pw budget | Aligned. Ch 5 covers trisc.cpp adaptation only; Phase 3 also includes CB model and CLI. |
| ebreak+mailbox GDB | ~4 pw (Ch 5) | Phase 4: 4-6 pw | Aligned. Phase 4 also includes multi-RISC and NOC. |
| Quasar debug module | Contingent (Ch 5) | Phase 4: conditional 2 pw | Aligned. Contingent on VE-1 validation. |
| Interceptor capture | 1-2 pw estimated (Ch 6) | Phase 1 hook: ~1.5 pw | Aligned. Bulk of Phase 1 is schema and serialization. |
| Total Phases 0-4 | 20-29 pw | 20-29 pw | Consistent with plan spec. |

---

## 13. Quarterly Milestones (Assuming 2 Engineers)

| Quarter | Milestones | Cumulative Effort |
|---------|-----------|-------------------|
| Q1 (weeks 1-6) | Phase 0 complete. Phase 1 complete. First `.ttksnap` files produced. VE-1 started. | 6-9 pw |
| Q2 (weeks 7-12) | Phase 2 complete. L1 snapshots and failure capture working. VE-1 results available. | 10-15 pw |
| Q3 (weeks 13-20) | Phase 3 complete. Host model replay with GDB. `tt-kernel-capture` CLI. | 16-23 pw |
| Q4 (weeks 21-28) | Phase 4 complete (scope per VE-1). Hardware GDB stub. Multi-RISC replay. | 20-29 pw |

---

## 14. Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 0 | % of kernel compilations with debug ELFs retained | 100% when Inspector active |
| 0 | Time to find binary for reported crash | < 30 seconds |
| 1 | `.ttksnap` generation success rate | > 99% for targeted captures |
| 1 | Average `.ttksnap` size (single-core, compressed) | < 50 KB |
| 2 | Failure-triggered capture success rate | > 95% of Watcher failures |
| 2 | L1 snapshot capture latency | < 100 ms per core |
| 3 | Time from `.ttksnap` to "GDB at kernel entry" | < 60 seconds |
| 3 | % of LLK bugs debuggable via host model | > 70% |
| 4 | On-hardware GDB step latency (BH) | < 100 us per step |
| 4 | Multi-RISC deadlock reproduction rate | > 80% |

---

## Key Takeaways

- **The critical risk is GAP-DI-2 (debug register host accessibility).**
  VE-1 should start during Phase 2 to avoid serializing the critical path.

- **The host functional model delivers the highest value per effort.**
  Phase 3 enables full GDB debugging without any hardware dependency.

- **Per-phase go/no-go criteria prevent wasted effort.** Each phase has
  measurable acceptance criteria before proceeding.

- **Phase 0-2 (9-14 pw) deliver 93% state coverage and automatic failure
  capture.** This is the minimum investment for transformative improvement.

- **Phase 3 is the single highest-ROI phase** because it converts captured
  state from "inspectable artifact" to "interactively debuggable session."
  The "15-minute bug" test is the measurable outcome.

- **20-29 pw total is achievable with 2 engineers in 6-7 months.**
  The critical path through Phases 0-3 (host model) is 16-23 pw (~5 months).
  Phase 4 adds 4-6 pw for hardware debugging.

- **The total investment replaces a debugging paradigm (iterative printf)
  that costs kernel developers 2-5 hours per non-trivial bug** with one
  that takes 15-30 minutes -- a 4-10x productivity improvement.
