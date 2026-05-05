# Cross-Chapter Consistency Review -- Chapter 6: Interceptor Design

**Reviewer:** Agent B (Cross-Chapter Consistency)
**Scope:** Ch6 files (index.md, 01, 02, 03) checked against Ch1--Ch5 indexes
**Date:** 2026-05-05

---

## 1. Cross-Chapter Consistency: Do Ch6 Claims Match Ch1--Ch5?

### 1.1 Ch1 (Hardware Debugger Reference) Alignment

**PASS.** Ch6 index.md states the interceptor implements the "capture-and-replay pattern identified in Ch1's decision matrix as the alternative to live hardware attach." Ch1 index.md confirms that the central argument is that "a capture-and-replay paradigm ... offers a more practical path for Tenstorrent than the traditional live-attach model." The framing is consistent.

Ch6 File 01 Section 14 (Architecture Portability) states Wormhole B0 lacks `RISC_DBG_CNTL` registers. Ch5 index.md confirms: "WH/BH provide 4 breakpoints per thread ... Quasar provides a full RISC-V debug register interface (`rvdbg_cmd`)," and separately that `RISC_DBG_CNTL` is documented under Quasar. Ch1 does not contradict this. However, see Note 1A below.

> **Note 1A (Minor):** Ch6 File 01 Section 14 states "Blackhole" has `rvdbg_cmd + Debug Module`, implying BH has `RISC_DBG_CNTL`. Ch5 index.md says "WH/BH provide 4 breakpoints per thread with PC-match and conditional modes via `tensix_functions.h`" but attributes the full debug register interface (`rvdbg_cmd`, Debug Module, `RISC_DBG_CNTL`) to Quasar. The Ch5 notation section says "WH/BH breakpoint APIs are guarded by `#ifndef ARCH_QUASAR`" and "Quasar debug APIs are in separate header files from `tt-2xx/quasar/`." This suggests the `rvdbg_cmd` + Debug Module may be Quasar-only, not BH. Ch6's architecture portability table may overstate BH capabilities. **Recommend verifying Ch5 Section 2 detail files to confirm BH vs. Quasar feature boundaries.**

### 1.2 Ch2 (State Inventory) Alignment

**PASS.** Ch6 claims 93% coverage of the Ch2 state inventory (13 of 14 categories). Ch2 identifies four state categories (compiled binaries, runtime configuration, memory contents, implicit/coordination state). Ch6 File 03 Section 13 provides a detailed mapping table showing each Ch2 category against schema locations, with only "NOC transient state" excluded as "physically uncapturable from host." This exclusion is consistent with Ch2 File 04's coverage of NOC dependencies and Ch4's observation that NOC transactions are transient.

Ch2 gives per-core snapshot sizes of "~50 KB -- 1.5 MB." Ch6's worked examples in File 03 Section 11 give single-core eltwise at ~22 KB and single-core with full L1 at ~600 KB (for WH) to ~1.5 MB. These ranges are consistent.

Ch2's capture envelope size for a 64-core matmul (deduplicated, BF16) is "~2--8 MB" uncompressed. Ch6 File 03 Section 11.1 gives the 64-core matmul without tile data at ~384 KB, with CB tile data at ~3.5 MB, and with full L1 dump at ~96.4 MB. The ~3.5 MB figure for CB tile data falls within Ch2's "~2--8 MB" range. No contradiction.

> **Note 2A (Minor):** Ch6 File 01 Section 9 states `capture_program()` captures ProgramConfig with 7 fields (rta_offset, sem_offset, sem_size, cb_offset, cb_size, kernel_text_offset, kernel_text_size), omitting dfb_offset, dfb_size, and local_cb_size. However, Ch6 File 03 Section 3 (`ProgramConfigSnapshot` table) correctly includes all 10 fields. The File 01 code snippet is incomplete relative to File 03's schema. This is an internal Ch6 inconsistency flagged in Section 3 below but also relevant to Ch2 alignment because Ch2 File 02 defines the full ProgramConfig.

### 1.3 Ch3 (LightMetal Foundation) Alignment

**PASS.** Ch6 correctly describes the relationship with LightMetal:

- Ch6 index.md states the interceptor "reuses LightMetal's FlatBuffer infrastructure (`base_types.fbs`, `program_types.fbs`) and macro gating pattern, but uses a standalone context rather than extending `LightMetalCaptureContext`." Ch3 index.md confirms `LightMetalCaptureContext` is a singleton (lines 38-88 of the header), and Ch3 Section 2 identifies gaps including no compiled binary capture and no L1/CB data snapshots -- exactly the gaps Ch6 fills.

- Ch6 File 01 Section 4.1 explains why a standalone context is preferred over extending LightMetal: thread safety, schema independence, selective capture, and different lifecycles. This aligns with Ch3's description of LightMetal as a whole-session capture system.

- Ch6 File 03 Section 2 lists the existing schema types reused from `base_types.fbs`, `program_types.fbs`, `buffer_types.fbs`, `command.fbs`, and `light_metal_binary.fbs`. These match Ch3's inventory.

> **Note 3A (Informational):** Ch3 Section 3 proposes a `KernelSnapshotCommand` and `DeepTraceScope` mechanism. Ch6 takes a different architectural direction (standalone `KernelSnapshotCaptureContext` instead of `DeepTraceScope`). This is not a contradiction -- Ch6 explicitly states it prefers the standalone design -- but readers should note that Ch3's proposed extension path is partially superseded by Ch6's design. The `KernelSnapshotCommand` union member addition to `command.fbs` (Ch6 File 03 Section 4) is consistent with Ch3's proposal.

### 1.4 Ch4 (Dispatch and Coordination) Alignment

**PASS WITH NOTES.**

Ch4 recommends H4 (`update_program_dispatch_commands` at dispatch.cpp:2127) as the primary hook with a weighted score of 4.85/5.00. Ch6 instead selects Site B (`assemble_device_commands` at dispatch.cpp:1890) as the primary hook, with H4 as a secondary hook for delta capture on the cached path. This is a deliberate design divergence, not an inconsistency -- Ch6 File 01 Section 2.3 provides a detailed rationale for why Site B is preferred for the kernel snapshot use case (state is in structured C++ form vs. serialized command bytes). However, the divergence deserves explicit acknowledgment.

> **Note 4A (Significant):** Ch4 recommends H4 as the primary hook; Ch6 promotes Site B (which corresponds roughly to H3 in Ch4's framework, inside `assemble_device_commands`). Ch6 index.md cross-reference says "Hook point selection directly uses the interception point analysis from Ch4 File 2," but Ch6 actually reaches a different conclusion. The cross-reference should be more precise: Ch6 uses Ch4's analysis framework but selects a different primary hook for the kernel snapshot use case. This is defensible but should be stated more clearly.

> **Note 4B (Minor):** Ch4 gives the command sequence size as "~13 KB" for a typical dispatch. Ch6 File 01 Section 10 and File 02 Section 11 both reference this same "~13 KB" figure for `ProgramCommandSequence` copy cost. Consistent.

> **Note 4C (Minor):** Ch4 states single-core compute replay requires "~33 KB." Ch6 File 03 gives single-core eltwise at ~22 KB (no CB data) or ~38 KB (with 4-tile CB data). The ~33 KB figure from Ch4 falls between these, which is reasonable given different tile assumptions. No contradiction.

### 1.5 Ch5 (Debug and Hardware) Alignment

**PASS.**

- Ch6 File 02 Section 4 (`CAPTURE_BREAKPOINT()`) reuses the `debug_ring_buf_msg_t` mailbox and describes integration with the Watcher polling thread. Ch5 index.md confirms the Watcher system's polling-based architecture and the `PAUSE()` macro pattern. Ch6's comparison table (File 02 Section 4.4) correctly differentiates `CAPTURE_BREAKPOINT()` from `PAUSE()`.

- Ch6 File 02 Section 6 integrates with `WatcherServer::killed_due_to_error()` (line 34) and `exception_message()` (line 36). Ch5 confirms these are declared in `watcher_server.hpp`.

- Ch6 File 01 Section 14 states WH lacks `RISC_DBG_CNTL` registers. Ch5 confirms this architecture distinction (see Note 1A above for the BH question).

- Ch6 File 02 Section 10.1 notes Watcher disables DMA because the DMA library is not thread-safe. This aligns with Ch5 Section 1's description of Watcher's constraints.

---

## 2. Forward References to Ch7/Ch8

Ch6 index.md makes two forward references:

1. **Ch7 (proposed):** "The `.ttksnap` format is consumed by all three replay targets (emulator, on-hardware, host functional model)." Also: "WH's lack of `RISC_DBG_CNTL` registers means on-hardware replay (Ch7) must use ebreak+mailbox on WH vs debug registers on BH/Quasar."

2. **Ch8 (proposed):** "Effort estimates for interceptor implementation feed into the phased roadmap."

These forward references are appropriately qualified as "(proposed)" and do not overclaim. Ch5 already describes the three replay targets (Tier 1: host functional model, Tier 2: ebreak+mailbox, Tier 3: Quasar Debug Module/JTAG), so the Ch7 forward reference is grounded in Ch5's analysis.

**PASS.** Forward references are well-founded and appropriately tentative.

---

## 3. Internal Consistency within Ch6's 3 Files

### 3.1 Hook Point Terminology

**PASS.** All three files consistently refer to:
- "Site B" or "primary hook" for `assemble_device_commands` at dispatch.cpp:1890
- "H4-update" or "secondary hook" for `update_program_dispatch_commands` at dispatch.cpp:2127
- "Hook A" for post-compilation Inspector callback

### 3.2 Capture Mode Naming

**PASS.** The three modes are consistently named across all files:
- `MetadataOnly` (File 01: enum value 1, File 02: described in trigger table, File 03: `CaptureMode_MetadataOnly`)
- `Selective` (File 01: enum value 2, File 02: described with filters, File 03: `CaptureMode_Selective`)
- `FailureTriggered` (File 01: enum value 3, File 02: Watcher integration, File 03: `CaptureMode_FailureTriggered`)

File 03's FlatBuffer `CaptureMode` enum adds a fourth value `Breakpoint = 3`, which is not present in File 01's `KernelCaptureConfig::Mode` enum. This is acceptable because the FlatBuffer schema captures finer-grained trigger provenance than the host-side config enum, and File 02 describes `CAPTURE_BREAKPOINT()` as operating within the `FailureTriggered` mode. No functional inconsistency.

### 3.3 ProgramConfig Fields

> **Note 6A (Internal Inconsistency):** File 01 Section 6.2 (`capture_program()` code) captures only 7 of the 10 ProgramConfig fields (omits `dfb_offset`, `dfb_size`, `local_cb_size`). File 03 Section 3 defines `ProgramConfigSnapshot` with all 10 fields. The index.md key findings correctly state "all 10 fields including dfb_offset/dfb_size." The File 01 code snippet should be updated to include all 10 fields for consistency.

### 3.4 Size Estimates

Cross-checking size estimates across files:

| Metric | File 01 | File 02 | File 03 | Consistent? |
|--------|---------|---------|---------|:-----------:|
| MetadataOnly per invocation | ~300 B | ~300 B | ~300 B | Yes |
| Single-core selective (no L1) | ~95 KB | ~55 KB | ~22 KB | See Note 6B |
| 64-core matmul (no L1, deduplicated) | ~395 KB | ~5 MB (binaries only) | ~384 KB | See Note 6C |
| Ring buffer 16 entries (metadata) | -- | ~5 KB | ~5 KB | Yes |
| L1 readback WH (1 MB) | ~2.5 ms | ~2.5 ms | -- | Yes |
| L1 readback BH (1.5 MB) | ~3.8 ms | ~3.8 ms | -- | Yes |

> **Note 6B (Minor Inconsistency):** File 01 gives single-core selective (no L1) as ~95 KB (Section 8). File 02 gives it as ~55 KB (Section 5.2). File 03 gives single-core eltwise as ~22 KB (Section 11.2). These differences are partly explained by different workload assumptions (matmul with more binaries vs. eltwise with fewer), but File 01 and File 02 both claim to describe "single-core compute" without specifying the workload. The discrepancy between ~55 KB and ~95 KB should be reconciled or clarified with explicit workload labels.

> **Note 6C (Minor Inconsistency):** File 02 Section 5.2 gives "Per invocation (64-core matmul): ~5 MB (binaries only, no L1)" for the ring buffer calculation. File 01 gives ~395 KB and File 03 gives ~384 KB for 64-core matmul with deduplication. The ~5 MB figure in File 02 appears to be the non-deduplicated size (64 cores x ~84 KB binaries = ~5.25 MB), while Files 01 and 03 assume deduplication. File 02 should clarify whether this ring buffer estimate assumes deduplication or not.

### 3.5 Overhead Estimates

> **Note 6D (Minor Inconsistency):** File 01 Section 10 gives Selective (single core, no L1) overhead as "~15-20 us (~35%)." File 02 Section 11 gives "Selective (no L1)" overhead as "~15-50 ms." These are three orders of magnitude apart (microseconds vs. milliseconds). File 02 likely means milliseconds for the full scope including binary serialization and FlatBuffer building, while File 01 means microseconds for a lightweight selective capture. The units difference is jarring and should be harmonized, or the scopes should be explicitly distinguished.

### 3.6 Disabled-Mode Overhead

> **Note 6E (Trivial):** File 01 Section 10 gives disabled overhead as "~1 ns" for the `is_active()` atomic load. File 02 Section 11 gives disabled overhead as "~5 ns (singleton check + atomic load)." The difference (1 ns vs. 5 ns) reflects whether the singleton access cost is included. Both are negligible, but the index.md states "<1% overhead" for MetadataOnly rather than disabled, so there is no conflict at the summary level.

### 3.7 Cross-References Between Files

All three files correctly reference each other:
- File 02 references File 01 for `KernelSnapshotCaptureContext` class and primary hook
- File 03 references File 01 for primary hook and capture context, and File 02 for capture modes and ring buffer
- All files reference their respective sections in other chapters consistently

**PASS** with the notes above.

---

## 4. Terminology Alignment

| Term | Ch1--Ch5 Usage | Ch6 Usage | Aligned? |
|------|---------------|-----------|:--------:|
| `assemble_device_commands` | Ch4: one of six hook points (H3 area) | Ch6: "Site B," primary hook | Yes (renamed for clarity) |
| `update_program_dispatch_commands` | Ch4: H4, recommended primary hook | Ch6: "H4-update," secondary hook | Yes (role changed, name preserved) |
| `LightMetalCaptureContext` | Ch3: singleton, whole-session capture | Ch6: reference pattern, not extended | Yes |
| `ProgramConfig` | Ch2: L1 memory layout, Ch4: offsets | Ch6: captured in `ProgramConfigSnapshot` | Yes |
| `KernelGroup` | Ch4: dispatch grouping, launch messages | Ch6: iterated for capture, serialized | Yes |
| `ll_api::memory` | Ch3: binary data format | Ch6: source of ELF data for binary pool | Yes |
| `debug_ring_buf_msg_t` | Ch5: ring buffer mailbox | Ch6: `data[31]` for CAPTURE_BREAKPOINT | Yes |
| `PAUSE()` macro | Ch5: halt mechanism | Ch6: compared with CAPTURE_BREAKPOINT | Yes |
| `killed_due_to_error()` | Ch5: Watcher failure signal | Ch6: trigger for failure capture | Yes |
| `.ttksnap` | Not in Ch1--Ch5 | Ch6: new file format | N/A (new term) |
| `KernelSnapshotCaptureContext` | Not in Ch1--Ch5 | Ch6: new singleton class | N/A (new term) |
| `CAPTURE_BREAKPOINT()` | Not in Ch1--Ch5 | Ch6: new device-side macro | N/A (new term) |
| `Tensix` / `TRISC0/1/2` / `BRISC` / `NCRISC` | Consistent across all chapters | Consistent | Yes |
| `Wormhole B0` / `Blackhole` / `Quasar` | Consistent across all chapters | Consistent | Yes |

> **Note 7A (Minor):** Ch4 uses `.ttlk` as the proposed capture format name. Ch6 uses `.ttksnap`. These are different proposed formats for different scopes (Ch4's `.ttlk` captures a tiered dispatch record; Ch6's `.ttksnap` captures a single-invocation FlatBuffer snapshot). The relationship between these two formats is not explained in Ch6. If they are intended to coexist, a brief note clarifying their respective roles would help.

**PASS** on terminology alignment.

---

## 5. Summary of All Notes

| ID | Severity | Description |
|----|----------|-------------|
| 1A | Minor | Ch6 may overstate Blackhole's debug capabilities (rvdbg_cmd + Debug Module) relative to Ch5's Quasar-specific attribution. Verify against Ch5 Section 2 detail. |
| 2A | Minor | File 01 `capture_program()` code omits 3 of 10 ProgramConfig fields that File 03 schema includes. See also 6A. |
| 3A | Informational | Ch3's proposed `DeepTraceScope` extension path is partially superseded by Ch6's standalone context design. Not a conflict, but worth noting for readers. |
| 4A | Significant | Ch4 recommends H4 as primary hook; Ch6 selects Site B instead. The Ch6 index cross-reference should clarify this is a deliberate divergence from Ch4's recommendation, not a direct application of it. |
| 4B | -- | Consistent (~13 KB command sequence size). |
| 4C | -- | Consistent (single-core sizes compatible within assumptions). |
| 6A | Minor | Internal: File 01 captures 7 ProgramConfig fields; File 03 schema has all 10. Code should match schema. |
| 6B | Minor | Internal: Single-core size estimates vary (~22 KB, ~55 KB, ~95 KB) across files without clear workload labels. |
| 6C | Minor | Internal: 64-core ring buffer estimate in File 02 (~5 MB) appears non-deduplicated; Files 01/03 assume deduplication (~384 KB). Should clarify. |
| 6D | Minor | Internal: Selective overhead given as ~15-20 us in File 01 but ~15-50 ms in File 02. Three orders of magnitude discrepancy in units. |
| 6E | Trivial | Internal: Disabled overhead 1 ns (File 01) vs. 5 ns (File 02). Negligible. |
| 7A | Minor | Ch4 proposes `.ttlk` format; Ch6 proposes `.ttksnap`. Relationship not clarified. |

---

## 6. Verdict

**PASS WITH NOTES.**

Chapter 6 is broadly consistent with Chapters 1--5. The interceptor design correctly builds on the state inventory (Ch2), reuses LightMetal infrastructure without improper coupling (Ch3), leverages the dispatch path analysis (Ch4), and integrates with existing debug tools (Ch5) as described. The capture-and-replay paradigm from Ch1 is faithfully implemented.

The most significant note (4A) is the divergence from Ch4's primary hook recommendation. Ch6 provides a well-reasoned justification for selecting Site B over H4, but the cross-reference in the index should be reworded to acknowledge the divergence rather than implying direct adoption of Ch4's recommendation.

Internal consistency within Ch6 is good at the architectural level but has several minor numerical discrepancies (Notes 6B, 6C, 6D) that should be harmonized in a revision pass. The ProgramConfig field omission in File 01 (Note 6A) is a straightforward fix.

No blocking issues were found. The chapter is ready for use as a foundation for Ch7/Ch8 development.
