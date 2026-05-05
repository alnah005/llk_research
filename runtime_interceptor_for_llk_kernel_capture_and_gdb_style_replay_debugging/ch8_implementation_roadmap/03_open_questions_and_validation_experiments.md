# Open Questions and Validation Experiments

This file enumerates the open questions identified across Chapters 1-7,
ranked by blocking severity (how urgently each must be answered before
implementation can proceed), and proposes concrete validation experiments
with expected outcomes, probability estimates, failure modes, and fallback
strategies. All source citations reference commit `621b949`.

> **Cross-references:** Ch 2 (state inventory completeness), Ch 3 (LightMetal
> Issue #24955), Ch 4 (`tt_metal/impl/program/dispatch.cpp:1890` hook point),
> Ch 5 (Quasar debug registers, three-tier strategy, trisc.cpp host model),
> Ch 6 (interceptor design, `.ttksnap` schema), Ch 7 (three replay targets),
> `01_gap_analysis.md` (22 gaps, 4 CRITICAL),
> `02_phased_implementation_plan.md` (Phases 0-5, go/no-go criteria).

---

## 1. Question Prioritization Framework

Each question is classified into three tiers:

| Tier | Definition | Action |
|------|-----------|--------|
| **Tier 1: Plan-Blocking** | An unfavorable answer fundamentally alters the plan or blocks a phase | Must resolve before gated phase begins |
| **Tier 2: Scope-Affecting** | An unfavorable answer changes scope or performance targets but not feasibility | Should resolve 1 phase ahead of when results are needed |
| **Tier 3: Design Refinement** | Affects design details but does not gate any phase | Resolve opportunistically |

Within each tier, questions are additionally scored on:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Phase impact** | 0.40 | Which phase is blocked? Earlier = higher impact. |
| **Scope of alternatives** | 0.30 | How much rework if answer is unfavorable? |
| **Cost to resolve** | 0.30 | Effort for the validation experiment. |

---

## 2. Tier 1: Plan-Blocking Questions

### OQ-1: Can Quasar RISC_DBG_CNTL Registers Be Accessed from Host via PCIe?

- **Context:** The `RISC_DBG_CNTL_0/1` pulse-based protocol
  (`T6_DEBUG_REGS__RISC_DBG_CNTL_0` in `t6_debug_map.h`, lines 220-223)
  enables cross-core debug access. The `rvdbg_cmd` enumeration provides
  PAUSE/STEP/CONTINUE/RD_REG/WR_REG/RD_MEM/WR_MEM operations (Ch 5,
  File 2). The `riscv_dbg_wr` and `riscv_dbg_rd` functions write to
  `RISCV_DEBUG_REGS->RISC_DBG_CNTL_0/1` (lines 113-115 of
  `ckernel_riscv_debug.h`). But all existing usage is from firmware running
  on a neighboring RISC-V core. Whether the host can write to these
  registers via PCIe TLB-mapped access is undocumented.
- **Why Critical:** This determines whether hardware GDB-style debugging
  on BH/Quasar can use the clean register-based protocol (Tier 3) or must
  fall back to the ebreak+mailbox protocol (Tier 2). The GDB stub
  architecture in Phase 4 depends directly on this answer.
- **Blocking:** Phase 4, deliverables P4-T2 and P4-T3.
- **Resolution Path:** Validation Experiment VE-1.
- **Expected Timeline:** 1 pw; should start during Phase 2.

### OQ-2: Does Blackhole's RISC_DBG_CNTL_0/1 Work Identically to Quasar's?

- **Context:** Blackhole defines `RISCV_DEBUG_REG_RISC_DBG_CNTL_0` and
  `RISCV_DEBUG_REG_RISC_DBG_CNTL_1` (`tt-1xx/blackhole/tensix.h:162-163`)
  as active macros, but BH does NOT have the `rvdbg_cmd` software API
  (which exists only in `ckernel_riscv_debug.h` under `tt_llk_quasar`).
  The surrounding register map may differ from Quasar's.
- **Why Critical:** If BH and Quasar debug register protocols differ, the
  GDB stub needs architecture-specific implementations, increasing Phase 4
  effort by 1-2 pw.
- **Blocking:** Phase 4, deliverable P4-T3.
- **Resolution Path:** Code analysis of BH vs Quasar register maps + VE-1
  extended to BH hardware.

### OQ-3: Can L1 Memory Be Read Reliably While a Kernel Is Running?

- **Context:** The L1 snapshot in Phase 2 reads L1 memory before the kernel
  starts (at Site B, before go signal). But failure-triggered post-mortem
  capture reads L1 while the kernel is halted on an assert. The question is
  whether L1 reads via PCIe are consistent when a RISC-V core is actively
  writing to L1.
- **Why Critical:** If L1 reads are not consistent, captured tile data may
  be corrupted, rendering snapshots unreliable.
- **Blocking:** Phase 2, deliverable P2-T1.
- **Resolution Path:** Validation Experiment VE-2.

---

## 3. Tier 2: Scope-Affecting Questions

### OQ-4: How Can SFPI Instructions Be Functionally Modeled for Host Replay?

- **Context:** `trisc.cpp` host model compiles LLK kernels with a host
  compiler, replacing SFPI intrinsics with C++ float operations. This
  provides functional equivalence but not bit-exact results (Ch 5, File 3).
  The question is how large the divergence is for production kernels.
- **Why Important:** If host model output diverges significantly, developers
  will not trust the replay tool, undermining adoption.
- **Blocking:** Phase 3 quality gate.
- **Resolution Path:** Validation Experiment VE-3.

### OQ-5: What Is the Minimal trisc.cpp Extraction for Standalone Replay?

- **Context:** Phase 3 deliverable P3-T2 requires factoring out the minimal
  runtime from `trisc.cpp` (`tt-llk/tests/helpers/src/trisc.cpp`) and the
  LLK test harness into a standalone library. The file uses
  `ckernel::regfile` as a host-side array (line 58), `mailbox_t` as host
  pointers, and `run_kernel(temp_args)` for execution (line 72). The
  question is how many headers and stubs must be extracted.
- **Why Important:** If extraction requires >50 files with circular
  dependencies, Phase 3 effort increases significantly.
- **Blocking:** Phase 3, deliverable P3-T2.
- **Resolution Path:** Dependency analysis + Validation Experiment VE-4.

### OQ-6: Does the JIT Build Cache Need Modification for Debug Symbol Retention?

- **Context:** Phase 0 deliverable P0-T2 requires retaining debug symbols
  in cached ELFs. Adding `-g` changes the FNV1a hash, so debug and release
  builds have separate cache entries, potentially doubling cache size.
- **Why Important:** Excessive cache growth in CI/CD could cause disk pressure.
- **Blocking:** Phase 0, deliverable P0-T2.
- **Resolution Path:** Measure cache size impact. Consider separate partitions.

### OQ-7: Can the Coprocessor Be Halted Independently of the RISC-V Cores?

- **Context:** If halting a RISC-V core via debug registers also stalls the
  Tensix coprocessor pipeline, in-flight tile operations may be incomplete.
  If it continues independently, DEST registers may be overwritten.
- **Why Important:** Affects hardware replay fidelity (Phase 4) but not
  host model replay (Phase 3).
- **Blocking:** Phase 4 hardware register readback accuracy.
- **Resolution Path:** Validation Experiment VE-5.

### OQ-8: How Does the Debug Register Interface Interact with Dispatch Timeouts?

- **Context:** Halting a core for debugging may cause dispatch timeout
  (`WatcherServer::killed_due_to_error()` at `watcher_server.hpp:34`).
- **Why Important:** Affects usability of on-hardware debugging.
- **Blocking:** Phase 4 usability.
- **Resolution Path:** Implement `TT_METAL_DEBUG_SESSION=1` env var
  disabling timeouts. Test with existing `PAUSE()` macro.

### OQ-9: Does the dispatch.cpp Hook Point Remain Stable?

- **Context:** The interceptor architecture (Ch 6) hooks into
  `assemble_device_commands()` at `dispatch.cpp:1890`. These are specific
  implementation details, not stable APIs.
- **Why Important:** If hooks break on every major TT-Metal release, the
  interceptor becomes a maintenance burden.
- **Blocking:** Phase 1 and all subsequent phases.
- **Resolution Path:** Review git log for dispatch.cpp over last 6 months.
  Discuss with dispatch team whether an official callback API is feasible.

---

## 4. Tier 3: Design Refinement Questions

### OQ-10: How Should Quasar NEO Cluster Scaling Affect Snapshot Design?

- **Context:** NEO clusters: 4 engines x 4 compute processors = 16 per cluster.
  Snapshot sizes and replay complexity scale by 16x.
- **Resolution:** Ensure `.ttksnap` schema supports variable processor counts
  via `TensixCoreSnapshot::processor_snapshots` vector. Add `processor_count`
  to `CaptureMetadata`.

### OQ-11: Licensing Implications of Embedding Spike

- **Context:** Spike is BSD-3-Clause (permissive). QEMU is GPLv2 (copyleft).
- **Resolution:** Legal review during Phase 3. BSD-3-Clause is straightforward.

### OQ-12: Should the Snapshot Format Support Cross-Architecture Replay?

- **Context:** `.ttksnap` captured on WH may be replayed on host model even
  without matching hardware. But L1 size, processor count, ISA extensions differ.
- **Resolution:** Add `architecture` field to metadata. Warn on mismatch.
  Accept best-effort cross-architecture replay.

### OQ-13: How Should Non-Deterministic Behavior Be Documented in Replay Output?

- **Context:** NOC arbitration timing, L1 bank contention are non-deterministic (GAP-RF-3).
- **Resolution:** Add `--strict-mode` flag that warns on non-deterministic
  operations. Log divergences for review.

---

## 5. Validation Experiments

### VE-1: Quasar/Blackhole Debug Register Host Accessibility (CRITICAL)

**Objective:** Determine whether host PCIe writes can reach Tensix debug
registers (`RISC_DBG_CNTL_0/1`) and execute `rvdbg_cmd` operations.

**Prerequisites:**
- Access to Blackhole or Quasar hardware
- UMD `write_to_device()` / `read_from_device()` API access
- Knowledge of Tensix debug register addresses for target architecture

**Experiment Steps:**

1. **Setup:** Write a minimal C++ program using UMD API to access a single
   Tensix core's debug register space. Use the address offset from
   `RISCV_DEBUG_REGS_START_ADDR` (defined in `tensix.h`).

2. **Test 1 -- Register visibility:** Write a known pattern to
   `RISCV_DEBUG_REG_RISC_DBG_CNTL_0` (at `RISCV_DEBUG_REGS_START_ADDR + 0x80`)
   via PCIe. Read back `RISCV_DEBUG_REG_RISC_DBG_STATUS_0`
   (at `RISCV_DEBUG_REGS_START_ADDR + 0x88`). Verify the status register
   reflects the command.

3. **Test 2 -- PAUSE command:** Issue `rvdbg_cmd::PAUSE` targeting TRISC0.
   Read `RISC_DBG_STATUS_0` and check the `paused` bit. Then issue
   `rvdbg_cmd::CONTINUE` and verify the core resumes.

4. **Test 3 -- Register read:** Issue `rvdbg_cmd::RD_REG` for GPR x1 (ra)
   of a halted TRISC0. Read the data from `RISC_DBG_STATUS_1`
   (at `RISCV_DEBUG_REGS_START_ADDR + 0x8C`). Verify the value matches a
   known register value (set by a test kernel that writes a sentinel to x1
   then spins).

5. **Test 4 -- Latency measurement:** Time 1000 iterations of PAUSE +
   RD_REG + CONTINUE cycles. Report mean and P99 latency per operation.

**Expected Outcomes:**

| Outcome | Probability | Implication |
|---------|-------------|-------------|
| **Positive: All tests pass, latency <100 us** | 40% | Phase 4 proceeds with full debug register GDB stub. Best-case scenario. |
| **Partial: Registers accessible but latency >1 ms** | 20% | Hardware GDB works but stepping is slow. Batch operations (breakpoint-and-continue) preferred. |
| **Partial: PAUSE works but RD_REG fails** | 15% | Core can be halted but register inspection requires L1 save/restore by firmware. Hybrid approach. |
| **Negative: PCIe writes do not reach debug registers** | 25% | Phase 4 hardware GDB falls back to ebreak+mailbox for all architectures. No debug register path. Reduce Phase 4 scope by ~2 pw. |

**Failure Mode and Fallback:**
If negative, the GDB stub uses ebreak+mailbox exclusively. The ebreak handler
saves GPR state to L1 (known address), the host reads L1 via PCIe (known to
work), and resumes by writing a continue flag to the mailbox. This is Tier 2
debug and adds ~50 us per single-step.

**Effort:** 1 pw. Start during Phase 2.

---

### VE-2: L1 Read Consistency During Active Dispatch (CRITICAL)

**Objective:** Verify that host PCIe reads of L1 memory produce consistent
data when the dispatch path has written configuration but the kernel has
not yet started.

**Experiment Steps:**

1. **Setup:** Write a test program that creates a compute kernel with known
   runtime args and CB configurations. Insert a delay between
   `assemble_device_commands()` and `write_program_command_sequence()` (or
   use a debug breakpoint at `dispatch.cpp:1890`).

2. **Test 1 -- Read CB config region:** After `finalize_offsets()` populates
   the program config buffer, read the CB configuration region of L1 via
   PCIe. Compare with the host-side `CircularBufferConfig` values.

3. **Test 2 -- Read runtime args region:** Read the runtime args region of
   L1. Compare with `Kernel::core_to_runtime_args_` values.

4. **Test 3 -- Concurrent write + read:** Have the dispatch path write
   kernel binaries to L1 while simultaneously reading L1 from a separate
   host thread. Check for partial reads (torn data).

5. **Test 4 -- Post-assert read:** Trigger a kernel assert. After Watcher
   detects the assert, read full L1. Verify contents are stable.

**Expected Outcomes:**

| Outcome | Probability | Implication |
|---------|-------------|-------------|
| **Positive: Reads consistent in all tests** | 70% | Phase 2 L1 snapshot proceeds as designed. Capture at Site B is safe. |
| **Partial: Test 3 shows torn reads during concurrent writes** | 25% | L1 snapshot must be taken strictly before or after dispatch writes, not concurrently. This is the natural behavior of the Site B hook. |
| **Negative: Reads inconsistent even without concurrent writes** | 5% | L1 read path has caching/coherence issue. Investigate TLB configuration. May require flush operations. |

**Effort:** 0.5 pw. Part of Phase 2.

---

### VE-3: Host Functional Model Fidelity Assessment (HIGH)

**Objective:** Quantify the numerical divergence between host functional
model (`trisc.cpp`) output and real hardware output for representative LLK
operations.

**Experiment Steps:**

1. **Setup:** Select 5 representative LLK operations: eltwise_unary (exp),
   eltwise_binary (add), matmul (BF16), reduce (sum), topk. These cover
   SFPU-heavy, FPU-heavy, and mixed workloads.

2. **Test:** For each operation:
   a. Run on real hardware. Capture output tiles.
   b. Capture input tiles via Phase 2 L1 snapshot.
   c. Run the same kernel on the host functional model with captured inputs.
   d. Compare: max absolute error, mean absolute error, and percentage of
      tiles with >1 ULP divergence.

3. **Analyze:** Categorize divergences: rounding mode, denorm handling,
   block-float precision loss, SFPI approximation.

**Expected Outcomes:**

| Outcome | Probability | Implication |
|---------|-------------|-------------|
| **Good: <1% tiles diverge by >1 ULP for BF16** | 50% | Host model trustworthy for most debugging. Document that bit-exact requires hardware replay. |
| **Acceptable: 1-10% diverge, max error <0.1%** | 35% | Useful for control-flow debugging and gross error detection. Not for precision debugging. |
| **Poor: >10% significant divergence** | 15% | Host model unreliable for math. Prioritize hardware replay (Phase 4). Investigate specific SFPI operations. |

**Effort:** 1 pw. Start during Phase 3.

---

### VE-4: trisc.cpp Standalone Extraction Feasibility (HIGH)

**Objective:** Determine the complexity of extracting the minimal `trisc.cpp`
runtime into a standalone replay library.

**Experiment Steps:**

1. **Dependency analysis:** Starting from
   `tt-llk/tests/helpers/src/trisc.cpp`, trace all `#include` directives
   recursively. Count unique headers. Identify circular dependencies.

2. **Minimal build test:** Attempt to compile with host `g++ -std=c++17`
   and minimum LLK headers. Record missing symbols. Categorize:
   (a) easily stubbed hardware intrinsics, (b) complex dependencies, (c)
   architecture-specific paths.

3. **Prototype:** Build minimal standalone program: load hardcoded L1 image,
   populate `ckernel::regfile` and CB pointers, call simple LLK kernel via
   `run_kernel()`, read output from virtual L1.

**Expected Outcomes:**

| Outcome | Probability | Implication |
|---------|-------------|-------------|
| **Simple: <30 headers, <10 stubs** | 40% | Phase 3 P3-T2 effort is ~2 pw as estimated. |
| **Moderate: 30-80 headers, 10-30 stubs** | 45% | P3-T2 increases to ~3 pw. Need automated stub generation. |
| **Complex: >80 headers, circular deps** | 15% | Alternative: compile against full LLK tree with host flags rather than extracting minimal runtime. |

**Effort:** 0.5 pw. Start during Phase 2.

---

### VE-5: Coprocessor Halt Behavior Under RISC-V Core Halt (MEDIUM)

**Objective:** Determine whether halting a RISC-V core via debug registers
also halts the Tensix coprocessor.

**Experiment Steps:**

1. **Setup:** Run a compute kernel performing long-running tile math. After
   start, issue PAUSE to TRISC1 via debug registers (or ebreak).

2. **Observation 1:** Read DEST register file via debug array
   (`dbg_dump_array_enable()`, `dbg_dump_array_rd_cmd()`,
   `dbg_dump_array_to_l1()` from `tensix_functions.h` starting at line 697).
   Verify values are stable while core is halted.

3. **Observation 2:** Issue STEP. Read DEST again. Verify exactly one
   instruction's worth of state change.

4. **Observation 3:** Time the halt-resume cycle. Measure if there is
   additional latency beyond the RISC-V core halt.

**Expected Outcomes:**

| Outcome | Probability | Implication |
|---------|-------------|-------------|
| **Coprocessor halts synchronously** | 60% | Clean debug state. Best case for hardware debugging. |
| **Coprocessor finishes current op then halts** | 30% | Register file at "safe point" between operations. Usable. |
| **Coprocessor continues independently** | 10% | Register state unpredictable at halt. Must use "capture before launch" approach. |

**Effort:** 0.5 pw. Part of Phase 4.

---

### VE-6: Capture Overhead Benchmark (MEDIUM)

**Objective:** Measure actual interceptor overhead across capture modes to
validate Ch 4's ~2 us estimate and Ch 6's overhead model.

**Experiment Steps:**

1. **Baseline:** Tight dispatch loop (1000 iterations, single-core eltwise)
   without capture. Measure wall time and per-dispatch latency.

2. **MetadataOnly:** Enable `TT_METAL_KERNEL_CAPTURE=metadata`. Measure delta.

3. **Selective (no L1):** Enable selective with `TT_METAL_KERNEL_CAPTURE_L1=0`.
   Includes binary serialization and FlatBuffer construction.

4. **Selective (with L1):** Enable `TT_METAL_KERNEL_CAPTURE_L1=1`. Measure
   L1 readback latency separately from serialization.

5. **Failure-triggered (no failure):** Enable `on-assert`, run normally.
   Verify zero overhead.

**Expected Outcomes:**

| Mode | Expected Overhead | Acceptable Threshold |
|------|-------------------|---------------------|
| MetadataOnly | <1 us per dispatch | <5 us |
| Selective (no L1) | 2-5 us inline + background I/O | <10 us inline |
| Selective (with L1) | 5-80 ms per snapshot | <200 ms |
| Failure-triggered (no failure) | 0 | 0 |

**Effort:** 0.5 pw. Part of Phase 1.

---

### VE-7: SFPU Bug Frequency and Host Model Coverage (MEDIUM)

**Objective:** Determine what percentage of real LLK bugs require SFPU-level
fidelity that the host model cannot provide.

**Experiment Steps:**

1. Review last 50 LLK bug reports from the kernel team.
2. Classify each as: scalar logic, data format, SFPU-specific, or synchronization.
3. Determine: for each category, would host model replay have been sufficient
   to identify root cause?

**Expected Outcomes:**

| Outcome | Probability | Implication |
|---------|-------------|-------------|
| **<20% require SFPU fidelity** | 60% | Host model covers vast majority. SFPU model is deferrable. |
| **20-40% require SFPU fidelity** | 30% | Host model still valuable but invest in SFPI intrinsic stubs (add 2 pw to Phase 3). |
| **>40% require SFPU fidelity** | 10% | Reprioritize: accelerate Phase 4 hardware replay. Host model demoted to logic-only debugging. |

**Effort:** 0.5 pw. Run during Phase 1-2.

---

## 6. Question-to-Phase Mapping

| Question | Tier | Must Resolve Before | Experiment |
|----------|------|-------------------|------------|
| OQ-1 (Quasar debug reg access) | 1 | Phase 4 start | VE-1 |
| OQ-2 (BH vs Quasar debug regs) | 1 | Phase 4 start | VE-1 extended |
| OQ-3 (L1 read consistency) | 1 | Phase 2 start | VE-2 |
| OQ-4 (SFPI host model fidelity) | 2 | Phase 3 quality gate | VE-3 |
| OQ-5 (trisc.cpp extraction) | 2 | Phase 3 start | VE-4 |
| OQ-6 (Build cache debug symbols) | 2 | Phase 0 completion | Measurement |
| OQ-7 (Coprocessor halt) | 2 | Phase 4 design | VE-5 |
| OQ-8 (Dispatch timeout) | 2 | Phase 4 design | Impl + test |
| OQ-9 (Hook stability) | 2 | Phase 1 start | Code review |
| OQ-10 (Quasar NEO scaling) | 3 | Phase 1 schema | Analysis |
| OQ-11 (Spike licensing) | 3 | Phase 3 distribution | Legal review |
| OQ-12 (Cross-arch replay) | 3 | Phase 5 | Design decision |
| OQ-13 (Non-determinism docs) | 3 | Phase 5 | Documentation |

---

## 7. Experiment Schedule

```
Phase 0 (weeks 1-3):
  VE-6 (capture overhead) -- validates Phase 1 design assumptions
  OQ-6 measurement (build cache impact)
  OQ-9 code review (dispatch.cpp hook stability)

Phase 1 (weeks 4-9):
  VE-6 continued with actual interceptor
  VE-7 (SFPU bug frequency) -- de-risks Phase 3 scope

Phase 2 (weeks 7-12):
  VE-1 (debug register access) -- start early, results needed for Phase 4
  VE-2 (L1 read consistency) -- required for Phase 2 deliverables
  VE-4 (trisc.cpp extraction) -- de-risks Phase 3

Phase 3 (weeks 13-20):
  VE-3 (host model fidelity) -- validates Phase 3 quality

Phase 4 (weeks 21-28):
  VE-5 (coprocessor halt) -- informs hardware debug design
```

**Total experiment effort: ~5 pw (~25 person-days)**, spread across Phases 0-3.
The most critical experiments (VE-1, VE-2) total ~1.5 pw and should be completed
before Phase 4 design decisions are needed.

---

## 8. Decision Trees for Major Unknowns

### Debug Register Accessibility (OQ-1 + OQ-2)

```
VE-1 Result?
  |
  +-- Positive (all tests pass, <100 us latency)
  |     |
  |     +-- BH and Quasar both work?
  |     |     Yes --> Full debug register GDB stub (Phase 4 full scope)
  |     |     No  --> Architecture-specific stub: debug regs on one,
  |     |            ebreak+mailbox on the other
  |     |
  |     +-- Latency >1 ms?
  |           Yes --> Breakpoint-and-continue mode (batch operations)
  |           No  --> Interactive single-step supported
  |
  +-- Partial (PAUSE works, RD_REG fails)
  |     --> Hybrid: halt via debug regs, inspect via L1 save/restore
  |         Firmware debug stub saves GPRs to known L1 address on halt
  |
  +-- Negative (PCIe cannot reach debug registers)
        --> ebreak+mailbox for all architectures (Tier 2 only)
            Reduce Phase 4 scope by ~2 pw
            Hardware debug ceiling: cooperative single-step only
```

### Host Model Fidelity (OQ-4)

```
VE-3 Result?
  |
  +-- Good (<1% divergence)
  |     --> Host model is primary debug target
  |         Market as "source-level LLK debugger"
  |
  +-- Acceptable (1-10% divergence)
  |     --> Host model for control-flow debugging
  |         Add comparison mode (host vs hardware) for precision work
  |         Prioritize Phase 4 hardware replay for precision bugs
  |
  +-- Poor (>10% divergence)
        --> Host model unreliable for math debugging
            Reprioritize: accelerate Phase 4 hardware replay
            Host model demoted to "structural/logic debugging only"
```

### Hook Stability (OQ-9)

```
E3 / Code Review Result?
  |
  +-- Stable (< 1 breaking change per quarter)
  |     --> Proceed with dispatch.cpp hook. Budget 0.5 pw/quarter maintenance.
  |
  +-- Unstable (frequent refactoring)
        --> Negotiate official callback API with dispatch team
            (similar to Inspector::program_kernel_compile_finished)
            Add as Phase 1 prerequisite task
```

---

## 9. Decision Points: Where Experiment Results Change the Plan

| Experiment | Favorable Result | Unfavorable Result | Plan Change |
|-----------|-----------------|-------------------|-------------|
| VE-1 (Quasar host access) | Host can write RISC_DBG_CNTL | Host cannot reach registers | Phase 4: add BRISC debug agent firmware (+2-3 pw) |
| VE-2 (L1 readback) | < 200 ms, coherent reads | Torn reads or > 500 ms | Phase 2: add halt-before-read protocol (+1-2 pw) |
| VE-3 (Host model fidelity) | < 1% tile divergence | > 10% divergence | Phase 3: invest in SFPI stubs (+2 pw) or accelerate Phase 4 |
| VE-4 (trisc.cpp extraction) | < 30 headers, < 10 stubs | > 80 headers, circular deps | Phase 3: compile against full LLK tree instead of extracting |
| VE-5 (Coprocessor halt) | Halts synchronously | Continues independently | Phase 4: document limitation or add explicit halt command |
| VE-6 (Capture overhead) | < 5 us MetadataOnly | > 10 us | Phase 1: optimize serialization path or default to on-assert only |
| VE-7 (SFPU bugs) | < 20% require SFPU fidelity | > 40% require SFPU | Phase 3: add SFPI stubs (+2 pw) |

---

## Key Takeaways

- **Three CRITICAL questions (OQ-1, OQ-2, OQ-3) must be answered before
  their respective phases can proceed.** OQ-3 (L1 read consistency) gates
  Phase 2 and should be answered first. OQ-1/OQ-2 (debug register access)
  gate Phase 4 and should be validated during Phase 2 to avoid serializing
  the critical path.

- **VE-1 (debug register access) is the highest-leverage experiment.** It
  resolves 3 open questions (OQ-1, OQ-2, OQ-10) and determines the scope of
  Phase 4 (4-6 pw of effort). Starting it during Phase 2 provides ~8 weeks
  of lead time before Phase 4 decisions are needed.

- **The validation experiments are designed with explicit fallback strategies
  for every negative outcome.** No single experimental failure kills the
  project. The worst case (OQ-1 negative + OQ-4 poor) limits the system to
  cooperative hardware debugging (ebreak+mailbox) with a less-precise host
  model, which still delivers significant value over the status quo of
  printf-debugging (DPRINT) and post-mortem analysis.

- **The experiment schedule ensures no phase is blocked by an experiment
  that has not yet started.** Each experiment runs 1-2 phases ahead of when
  its results are needed.

- **Total validation effort (~5 pw) is small relative to the 20-29 pw
  implementation** and should be front-loaded to reduce risk for the most
  expensive phases (3 and 4).

- **The decision trees define a plan that adapts, not a fixed plan.**
  Both favorable and unfavorable outcomes lead to concrete next steps with
  explicit effort adjustments.
