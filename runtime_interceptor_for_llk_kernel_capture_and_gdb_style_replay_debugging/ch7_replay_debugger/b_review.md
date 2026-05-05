# Chapter 7 -- B-Level Quality Review

**Reviewer:** B-Level Structural Quality Reviewer
**Date:** 2026-05-05
**Files Reviewed:**
- `01_emulator_based_replay.md` (719 lines)
- `02_on_hardware_and_host_model_replay.md` (623 lines)
- `03_multi_core_replay_and_noc_coordination.md` (558 lines)
- `index.md` (64 lines)

**Compared Against:**
- `ch5_debug_and_hardware/index.md`
- `ch6_interceptor_design/index.md`

---

## 1. Criteria Checklist

### Criterion 1: PROPOSED Prefix

All proposed (non-existing) designs must be clearly marked with **PROPOSED**.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | Scope block states "All designs are **PROPOSED** unless explicitly noted as existing infrastructure." Individual sections (2.3 Recommendation, 3.3 Phased approach, Sections 8, 9, 13) are additionally labeled PROPOSED. Spike extension class, `tt-kernel-replay` CLI, VS Code DAP, and terminal session all carry PROPOSED tags. |
| 02 | **PASS** | Scope block carries the same blanket PROPOSED disclaimer. Individual sections (Section 2.2 Phased strategy, Section 6 BRISC-as-Debug-Agent, Section 7 terminal session, Section 10.1 Recommendation, Section 11 host-model build) all labeled PROPOSED. Existing infrastructure (`trisc.cpp`, `rvdbg_cmd`, `ckernel_riscv_debug.h`) correctly noted as existing. |
| 03 | **PASS** | Scope block carries the blanket PROPOSED disclaimer. Section 2.5 Phased approach, Section 4.3 CBSyncModel, Section 7 terminal session all labeled PROPOSED. Existing NOC event structures (`noc_debugging.hpp`) correctly noted as existing. |

### Criterion 2: Cross-References

Cross-chapter references must cite specific chapter/file/section numbers.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | Cross-references block cites Ch 2, Ch 4, Ch 5 File 3, Ch 6 File 3, Ch 8. Body text references "(Ch 2, File 3)" for memory layout, "(Ch 2, File 4)" for register layout, "(Ch 4, File 3)" for NOC transaction property, "(Ch 6, File 3)" for `.ttksnap` schema and `KernelBinaryBlob.base_address`, "Ch 5, File 3, Section 4" for E-taxonomy, "Ch 5, File 3" for E-Opaque. All are specific. |
| 02 | **PASS** | Cross-references block cites Ch 2, Ch 4, Ch 5 File 2, Ch 5 File 3, Ch 6 File 3, Ch 8. Body references "Ch 5, File 1" for PAUSE macro. All specific. |
| 03 | **PASS** | Cross-references block cites Ch 2, Ch 4, Ch 5 File 1, Ch 5 File 3, Ch 6 File 2, Ch 6 File 3, Ch 8. Body references "(Ch 4, File 3)" for TRISC NOC property, "(Ch 5, File 3)" for E-Opaque, "(Ch 4, File 1)" for dispatch pipeline. All specific. Internal cross-references to File 01 and File 02 are present. |

### Criterion 3: Commit Citation

Commit `621b949` must be referenced.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | Scope block: "All source citations reference commit `621b949`." Terminal session output includes `TT-Metal: 621b949`. |
| 02 | **PASS** | Scope block: "All source citations reference commit `621b949`." |
| 03 | **PASS** | Scope block: "All source citations reference commit `621b949`." |

### Criterion 4: Key Takeaways

Each file must end with a Key Takeaways section.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | "Key Takeaways" section at line 694 with 6 bullet points. Covers Spike recommendation, GDB integration, memory model, SFPU handling, single-TRISC replay, and fidelity levels. |
| 02 | **PASS** | "Key Takeaways" section at line 599 with 5 bullet points. Covers on-hardware fidelity, host model scoring, WH limitations, BRISC-as-agent, and DebugTransport abstraction. |
| 03 | **PASS** | "Key Takeaways" section at line 521 with 7 bullet points. Covers independent replay, 90/10 rule, NOC recording, 5-RISC coordination, snapshot consistency, NOC modes, and scaling. |

### Criterion 5: Scoring Matrices

Alternatives must be evaluated with weighted scoring matrices.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | 4 scoring matrices: emulator engine (Section 2.2), GDB integration (Section 3.2), memory model (Section 4.2), SFPU handling (Section 5.1). All have Weight columns and weighted totals. |
| 02 | **PASS** | 3 scoring matrices: on-hardware debug agent (Section 2.2), host model alternatives (Section 10.1), three-way comparison (Section 12). All have Weight columns and weighted totals. |
| 03 | **PASS** | 1 scoring matrix: multi-core replay strategy (Section 2.5). Has Weight column and weighted totals. |

### Criterion 6: Line Counts

Files should be within 400-800 lines each.

| File | Lines | Pass/Fail | Notes |
|------|-------|-----------|-------|
| 01 | 719 | **PASS** | Within range (400-800). |
| 02 | 623 | **PASS** | Within range (400-800). |
| 03 | 558 | **PASS** | Within range (400-800). |

Total: 1,900 lines. Index correctly reports this total.

### Criterion 7: Consistency with Ch5/Ch6

Architecture capability claims must match Ch5. Schema references must match Ch6.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | E-taxonomy references (E-Pure/E-Model/E-Stub/E-Opaque) match Ch5 File 3 nomenclature. Three-tier debug strategy matches Ch5 index (Tier 1: host/emulator, Tier 2: on-hardware, Tier 3: RTL). `trisc.cpp` path matches Ch5 index. SFPU/NOC as E-Opaque matches Ch5. |
| 02 | **PASS** | Architecture capability table (Section 1) is consistent with Ch5 index: WH has breakpoints but no RISC_DBG_CNTL, BH has RISC_DBG_CNTL but no rvdbg_cmd, Quasar has full Debug Module + rvdbg_cmd + 8 watchpoints. `ckernel_riscv_debug.h` as Quasar-only matches Ch5. `tensix_functions.h` breakpoint API referenced consistently. `.ttksnap` FlatBuffer schema and `KernelInvocationSnapshot` root type match Ch6 File 3 description. `file_identifier "TKSN"` matches Ch6. |
| 03 | **PASS** | `NocWriteEvent`/`NocReadEvent` from `noc_debugging.hpp` matches Ch5 index ("NOC logging (NocWriteEvent/NocReadEvent)"). Capture mode references (single-core trigger, multi-core burst) match Ch6 File 2 description. `.ttksnap` per-core `CoreSnapshot` matches Ch6 File 3. |

**Minor inconsistency found:** Ch5 index lists the `noc_debugging.hpp` path as `tt_metal/hw/inc/hostdev/noc_debugging.hpp` while Ch7 File 02 lists it as `tt_metal/impl/debug/noc_debugging.hpp` and Ch7 File 03 also lists `tt_metal/impl/debug/noc_debugging.hpp`. The Ch7 index lists the same `tt_metal/impl/debug/noc_debugging.hpp` path. This is an inconsistency with Ch5 -- one of the two paths is wrong. See Recommended Fixes.

**Minor inconsistency found:** Ch6 index references `TensixCoreSnapshot` as a schema type but Ch7 File 03 refers to `CoreSnapshot` (Section 6.3 cross-ref line 20: "per-core `CoreSnapshot`"). These should use the same name. See Recommended Fixes.

### Criterion 8: Key Source Files Table

Each file must include a Key Source Files table.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | "Key Source Files" table at line 678 with 9 entries. Includes path, line numbers, and relevance. |
| 02 | **PASS** | "Key Source Files" table at line 582 with 10 entries. Includes path, line numbers, and relevance. |
| 03 | **PASS** | "Key Source Files" table at line 505 with 9 entries. Includes path, line numbers, and relevance. |

### Criterion 9: Error Message Examples

Error messages must include condition/context/recovery.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | Section 10 provides 6 error messages in a table with Condition and Message columns. Messages include condition context and recovery actions (e.g., "Update tt-kernel-replay or re-capture", "Set TT_METAL_RISCV_DEBUG_INFO=1 before workload", "Re-capture the kernel invocation", "Ensure the Tensix extension plugin is loaded"). |
| 02 | **PASS** | Section 15 provides 4 error messages in a table with Condition and Message columns. Messages include condition, context, and recovery (e.g., "Use matching device or: tt-kernel-replay emulate", "Use --device <other>, wait, or replay in emulator/host-model mode", "Set TT_METAL_HOME or use --include-path", "Re-capture with smaller stack or use --no-trap-handler on Quasar"). |
| 03 | **PASS** | Section 5 includes a completion-detection timeout warning with diagnostic context (per-core PC, semaphore state, identification of which core failed to increment). While not in a formal error-message table format, it provides condition, context, and recovery guidance implicitly. |

**Minor note on File 03:** The error/warning examples in File 03 are embedded in prose (Section 5, dispatch completion detection) rather than in a dedicated error message table. This is adequate but less structured than Files 01 and 02. See Recommended Fixes.

### Criterion 10: Terminal Session Examples

Realistic terminal sessions must be included.

| File | Pass/Fail | Notes |
|------|-----------|-------|
| 01 | **PASS** | Section 9 provides a detailed two-terminal session: `tt-kernel-replay inspect` + `tt-kernel-replay emulate` in terminal 1, then `riscv-tt-elf-gdb` session in terminal 2 with breakpoints, `info locals`, memory inspection, and stepping. Section 2 also includes a 3-terminal Spike+OpenOCD+GDB session. Realistic with actual addresses, register values, and kernel names. |
| 02 | **PASS** | Section 7 provides hardware replay terminal session (`tt-kernel-replay hw-replay` + GDB with custom `monitor` commands). Section 11.3 provides host-model terminal session (`tt-kernel-replay host-model` + GDB). Section 11.4 provides AddressSanitizer output. All realistic with addresses, kernel names, and diagnostic output. |
| 03 | **PASS** | Section 7 provides multi-core hang debugging terminal session (`tt-kernel-replay emulate` with `--cores all --noc-mode sync` + GDB backtrace showing stuck semaphore wait). Includes diagnostic output from the replay tool identifying the missing semaphore increment. |

---

## 2. Inconsistencies Found

### 2.1 `noc_debugging.hpp` Path Discrepancy (Ch5 vs Ch7)

- **Ch5 index** (Key Source Files table): `tt_metal/hw/inc/hostdev/noc_debugging.hpp`
- **Ch7 index** (Key Source Files table): `tt_metal/impl/debug/noc_debugging.hpp`
- **Ch7 File 02** (Key Source Files table): `tt_metal/impl/debug/noc_debugging.hpp`
- **Ch7 File 03** body text (line 77): `tt_metal/impl/debug/noc_debugging.hpp`

One of these paths is incorrect. The Ch7 files are internally consistent with each other but differ from Ch5. This needs verification against the actual source tree at commit `621b949`.

### 2.2 `CoreSnapshot` vs `TensixCoreSnapshot` Naming

- **Ch6 index** (Section 6.3 description): references `TensixCoreSnapshot`
- **Ch7 File 03** (cross-references, line 20): references `CoreSnapshot`

These should use the same FlatBuffer type name. Based on Ch6 which defines the schema, `TensixCoreSnapshot` is likely the canonical name.

### 2.3 `trisc.cpp` Path Discrepancy (Ch7 File 01 vs Ch7 File 02)

- **Ch7 File 01** Key Source Files table: `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp`
- **Ch7 File 02** body text (line 398-399): `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp`
- **Ch7 index** Key Source Files table: `tt-llk/tests/helpers/src/trisc.cpp`

The index uses a shorter path (`tt-llk/`) while the files use the full path (`tt_metal/third_party/tt_llk/`). Both likely resolve to the same file via symlink or submodule, but the index should use the full canonical path for consistency.

### 2.4 Weighted Total Denominator Ambiguity

- File 01 Section 2.2 scoring matrix: weighted total 87/80/85 -- but the maximum possible score is not stated
- File 02 Section 12: "Host model scores highest overall (90/130)" -- the "/130" is provided here
- File 03 Key Takeaways: "scoring 98/130 versus 70 for trace-driven and 59 for lockstep" -- correct

File 01 scoring matrices do not state the maximum possible score. While the weights and scores make this calculable, explicitly stating "/MAX" (e.g., "87/110") would improve clarity. This is a minor stylistic inconsistency, not an error.

---

## 3. Recommended Fixes

### Fix 1 (Minor): `noc_debugging.hpp` Path

Verify the correct path at commit `621b949` and update either Ch5 or Ch7 files to be consistent. Both locations should use the same canonical path.

### Fix 2 (Minor): `CoreSnapshot` -> `TensixCoreSnapshot`

In File 03, cross-references line 20, change `CoreSnapshot` to `TensixCoreSnapshot` to match the Ch6 schema definition.

### Fix 3 (Minor): Index `trisc.cpp` Path

In `index.md`, Key Source Files table, change `tt-llk/tests/helpers/src/trisc.cpp` to `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp` to match the files.

### Fix 4 (Optional): File 03 Error Message Table

Add a dedicated error message table in File 03 (similar to Files 01 and 02) consolidating the warning/error messages currently embedded in prose. Candidates:
- The completion timeout warning in Section 5
- NOC mode mismatch (no NOC log for logged mode)
- Core coordinate out of snapshot range

### Fix 5 (Optional): Explicit Maximum Scores in File 01

Add `/MAX` denominators to weighted totals in File 01 scoring matrices for immediate readability, matching the pattern used in Files 02 and 03 Key Takeaways.

---

## 4. Overall Quality Assessment

**Rating: PASS -- High Quality**

All three files meet all 10 review criteria. The synthesis is thorough, well-structured, and internally consistent. The PROPOSED/existing distinction is carefully maintained. Cross-references are specific to chapter/file/section granularity. Scoring matrices are present for all major design decisions with appropriate weight justifications. Terminal sessions are realistic and demonstrate actual debugging workflows with plausible addresses, kernel names, and register values.

The five inconsistencies identified are all minor (path variations, naming differences, stylistic denominator omission) and none affect technical correctness or reader comprehension. The recommended fixes are all low-effort editorial corrections.

**Strengths:**
- Excellent progression from single-TRISC (File 01) to on-hardware/host-model (File 02) to multi-core (File 03) -- each file builds naturally on the previous
- Architecture-dependent capability tables are consistent across all files and match Ch5
- The 90/10 scoping guidance (File 03) provides practical prioritization
- Error messages include actionable recovery steps
- Terminal sessions demonstrate end-to-end workflows, not just isolated commands
- Implementation priority table (File 03, Section 11) ties everything together with effort estimates

**No blocking issues. Files are ready for integration.**
