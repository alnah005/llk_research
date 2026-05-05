# Chapter 8: Implementation Roadmap, Gap Analysis, and Open Questions -- Index

**Synthesized from 5 competing generator outputs (v1-v5), evaluated by independent reviewers.**

**Best scores:** v3 = 5.0/5.0 (zero errors, best traceability), v1 = 4.8/5.0 (most file path citations), v4 = 4.8/5.0 (best task granularity)

**Synthesis strategy:** v3 primary for all files, supplemented with v4 task breakdowns, v5 developer-centric framing, and v1 verified file paths.

---

## Files

| # | File | Lines | Description |
|---|------|------:|-------------|
| 1 | `01_gap_analysis.md` | 653 | 22 gaps across 5 dimensions with severity scoring, developer pain quantification, architecture-specific gap comparison (WH/BH/Quasar), gap interaction matrix, gap-to-chapter traceability |
| 2 | `02_phased_implementation_plan.md` | 465 | Phases 0-5 with effort estimates (20-29 pw total for P0-P4), per-phase deliverables, risk registers, go/no-go criteria, staffing model, incremental value curve |
| 3 | `03_open_questions_and_validation_experiments.md` | 594 | 13 open questions in 3 tiers, 7 validation experiments with procedures, decision trees, experiment priority matrix, scheduling aligned to phases |

**Total: 1,712 lines across 3 files**

---

## Key Findings

1. **22 gaps identified, 4 CRITICAL**: (a) No L1 snapshot capability, (b) No GDB stub for Tensix RISC-V, (c) Compiled ELF binaries not captured (GAP-SC-2), (d) Quasar debug register host accessibility unvalidated

2. **Total effort: 20-29 person-weeks** for Phases 0-4:
   - Phase 0 Foundation (2-3 pw): Debug symbols, ELF retention, compile metadata
   - Phase 1 Minimal Capture (4-6 pw): `.ttksnap` schema, interceptor hook, selective triggers
   - Phase 2 L1 Snapshot (4-6 pw): L1 dump, failure-triggered capture, ring buffer
   - Phase 3 Replay Engine (6-8 pw): Host model + emulator GDB replay
   - Phase 4 Hardware Replay (4-6 pw): On-hardware GDB stub, multi-core, NOC recording

3. **Phase 3 is the highest-ROI phase** — converts captured state from "inspectable artifact" to "interactively debuggable session"

4. **VE-6 (capture overhead) should be completed before Phase 1; VE-2 and VE-3 are scheduled during Phases 2 and 3 respectively** — VE-6 alone is 1-2 person-days, resolving the highest-impact capture feasibility unknown

5. **Quasar debug register host accessibility (Q1/E1) is the single most consequential open question** — favorable answer enables high-quality on-hardware GDB; unfavorable adds 2-3 pw for firmware workaround

6. **Estimated 4-10x productivity improvement** for debugging workflows (from 2-5 hours per non-trivial bug to 15-30 minutes)

---

## Key Source Files Referenced

| File | Path | Relevance |
|------|------|-----------|
| `dispatch.cpp` | `tt_metal/impl/program/dispatch.cpp:1890` | Primary interceptor hook (assemble_device_commands) |
| `ckernel_riscv_debug.h` | `tt_llk_quasar/common/inc/` | rvdbg_cmd API (Quasar only) |
| `tensix.h` (BH) | `tt-1xx/blackhole/tensix.h:162-165` | RISC_DBG_CNTL registers (BH only) |
| `trisc.cpp` | `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp` | Host functional model |
| `noc_debugging.hpp` | `tt_metal/impl/debug/noc_debugging.hpp` | NocWriteEvent/NocReadEvent |
| `inspector.hpp` | `tt_metal/impl/debug/inspector/inspector.hpp:39-43` | program_kernel_compile_finished callback |
| `rtoptions.hpp` | `tt_metal/llrt/rtoptions.hpp:204` | riscv_debug_info_enabled |
| `lightmetal_capture.hpp` | `tt_metal/impl/lightmetal/lightmetal_capture.hpp:38` | LightMetalCaptureContext |

---

## Cross-Chapter Dependencies

- **Ch 1**: GPU debugger requirements (R1-R7) define success criteria
- **Ch 2**: State inventory (8 categories) defines what must be captured — gap analysis measures coverage
- **Ch 3**: LightMetal ~34% coverage establishes baseline; gaps drive Phases 1-2
- **Ch 4**: Dispatch hook architecture (H1+H4) provides interceptor integration points
- **Ch 5**: Debug hardware capabilities (by architecture) determine Phase 4 implementation strategy
- **Ch 6**: Interceptor design and `.ttksnap` schema are the input to Phase 1 implementation
- **Ch 7**: Three replay targets (emulator, hardware, host model) are Phase 3-4 deliverables
