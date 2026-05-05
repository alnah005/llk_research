# Ch5 Cross-Chapter Consistency Review (Agent B)

**Reviewer:** Agent B (Cross-Chapter Consistency)
**Date:** 2026-05-05
**Scope:** Cross-reference Ch5 (all 3 content files + index) against Ch1-Ch4
indexes and content, check internal consistency within Ch5, verify terminology
alignment, assess forward references, and identify gaps.

---

## Cross-Chapter Issues

1. **Ch5 File 2, Section 2.3 vs Ch1 b_review Issue 1 (risc_sel conflict).**
   Ch5 File 2 (line ~141-153) presents the `rvdbg_risc_sel` enum with
   TRISC0=0x00000000, TRISC1=0x00020000, TRISC2=0x00040000, TRISC3=0x00060000,
   and then states at line 152: "On WH/BH the comment at line 79 documents the
   mapping as: 0=BRISC, 1-3=TRISCs, 4=NCRISC." The Ch1 b_review (Pass 1,
   Issue 1) already flagged that the `rvdbg_risc_sel` enum value TRISC0
   occupies the same bit pattern the RISC_DBG_CNTL_0 comment assigns to BRISC,
   and the enum has no entries for BRISC or NCRISC. Ch5 File 2 reproduces this
   same conflicting information without acknowledging the inconsistency or
   citing the Ch1 resolution. The text at line 153 ("The interceptor must use
   an architecture-dependent RISC selector") is correct but insufficient --
   an implementer still cannot determine the correct bit pattern for targeting
   BRISC on Quasar. **Recommendation:** Add the explicit note recommended in
   the Ch1 b_review: the `rvdbg_risc_sel` enum covers only the four compute
   processors (TRISC0-3) in Quasar's NEO model, and targeting BRISC/NCRISC
   via this interface is an open question requiring hardware validation.

2. **Ch5 File 2, Section 1.4 vs Ch1 b_review Issue 2 (watchpoint count unit).**
   Ch5 File 2, Section 2.4 (line ~178) correctly states "8 hardware watchpoints
   per RISC." However, the Ch5 index (line 23) says "8 hardware watchpoints per
   RISC" which is also correct. This is consistent. No issue here -- noting for
   completeness that Ch5 correctly resolved the "per core" vs "per RISC"
   ambiguity that Ch1 b_review flagged.

3. **Ch5 index line 18 vs Ch1 b_review Issue 2 (five vs six RISCs).**
   Ch5 index line 18 says "WH/BH provide 4 breakpoints per thread" without
   mentioning how many threads/RISCs exist. Ch5 File 2, Section 12 architecture
   comparison (line ~714) correctly notes "Number of TRISCs: 3 (WH/BH), 4
   (Quasar)." However, the Ch5 File 3 trace bundle (Section 5.2, lines
   ~413-425) lists exactly 3 TRISC ELFs (`trisc0_kernel.elf`, `trisc1_kernel.elf`,
   `trisc2_kernel.elf`) and 3 local memory dumps (`local_mem_trisc0.bin`,
   `local_mem_trisc1.bin`, `local_mem_trisc2.bin`). The Ch1 b_review (Pass 2,
   Issue 2 and Pass 3, Issue 1) identified that Quasar has 6 RISC-V processors
   (BRISC, NCRISC, TRISC0-TRISC3). The trace bundle format in Ch5 File 3 is
   hardcoded to 3 TRISCs, which would silently drop TRISC3 state on Quasar.
   **Recommendation:** Either parameterize the trace bundle to include
   `trisc3_kernel.elf` and `local_mem_trisc3.bin` when targeting Quasar, or
   add an explicit note that the bundle format shown is WH/BH-specific and
   needs extension for Quasar.

4. **Ch5 File 3, Section 5.2 cross-reference "(Ch 2, Ch 3)" is vague.**
   Line ~402 says "Using the interceptor described in **(Ch 2, Ch 3)**." The
   interceptor design is actually proposed for Ch6 (as stated elsewhere in
   Ch5). Ch2 provides the state inventory and Ch3 analyzes LightMetal gaps.
   The interceptor itself is not "described" in Ch2 or Ch3 -- they provide
   the requirements. Ch4 identifies interception points. This cross-reference
   is misleading. **Recommendation:** Change to "Using the state inventory
   from Ch2 and the capture schema extensions from Ch3, implemented by the
   interceptor proposed in Ch6."

5. **Ch5 File 3, Section 10.3 cross-reference "(Ch 2, Ch 3)" repeated.**
   Line ~1007 states "The interceptor **(Ch 2, Ch 3)** should ensure
   TT_METAL_RISCV_DEBUG_INFO=1." Same issue as above -- the interceptor is
   designed in Ch6, not Ch2/Ch3. **Recommendation:** Same fix as issue 4.

6. **Ch5 File 1 capture envelope size vs Ch2 estimates.** Ch2 index states
   "Total per-core snapshot: ~50 KB -- 1.5 MB" and provides detailed tables.
   Ch4 states "Single-core compute replay requires ~33 KB." Ch5 File 3,
   Section 5.2 shows the trace bundle including a 1 MB L1 snapshot, 3 x 4 KB
   local memory dumps, config registers, runtime args, CB config, and
   semaphore state -- which would total well over 1 MB uncompressed. There
   is no contradiction per se, but Ch5 does not reconcile its bundle size
   with Ch2's envelope estimates or Ch4's 33 KB figure. The 33 KB figure
   from Ch4 explicitly excludes the full L1 dump, while Ch5's bundle includes
   it. **Recommendation:** Add a note in Ch5 File 3 Section 5.2 clarifying
   that the full L1 snapshot is optional (for Tier 2/3 only) and that a
   Tier 1 capture aligns with Ch4's ~33 KB estimate.

7. **Ch5 File 2, Section 8 thread halt protocol vs Ch4 five-RISC model.**
   Ch5 File 2 (Section 8) describes a cooperative halt between unpack and
   math threads using mailboxes, and references "Ch4/03 for the five-RISC
   coordination context." Ch4 Section 4.3 describes the BRISC-coordinated
   subordinate sync protocol using `subordinate_sync` mailbox and
   `RUN_SYNC_MSG_*` constants. The Ch5 halt protocol (mailbox_write from
   unpack to math, then semaphore drain for pack) is a different protocol
   from Ch4's BRISC subordinate sync. Ch5 presents this as coming from
   `ckernel_debug.h`, which is a debug-specific halt, while Ch4 describes
   the normal execution coordination. The two protocols are complementary
   but Ch5 does not explicitly distinguish them. **Recommendation:** Add a
   clarifying note that the Section 8 halt protocol is a debug-specific
   cooperative halt (between threads within the Tensix coprocessor) distinct
   from Ch4's BRISC-coordinated subordinate sync (which governs the overall
   kernel launch/completion lifecycle).

8. **Ch5 File 1 Section 5.2 LLK profiler buffer addresses vs Ch2 memory maps.**
   Ch5 File 1 (lines ~291-292) states Quasar TRISCs share buffer space ending
   at `0x16F000` and WH/BH share space ending at `0x16E000`. Ch2 provides
   L1 memory maps for all architectures. These profiler buffer addresses fall
   within the L1 address range but Ch2's memory map tables should be checked
   for consistency with these addresses. The addresses are plausible but not
   explicitly cross-referenced. **Recommendation:** Minor -- no action needed
   unless Ch2 memory maps show a conflict at these addresses.

## Internal Consistency Issues

1. **Ch5 File 1 gap table vs File 2 capabilities.** File 1 Section 10 gap
   summary table (line ~402-411) shows tt-exalens as "Yes" for per-instruction
   stepping, breakpoints (HW/SW), tile data, coprocessor config dump, and
   source-level debug. File 2 documents the hardware debug capabilities that
   underpin tt-exalens. These are consistent. No issue.

2. **Ch5 File 3 Section 7.2 (Option B) vs File 2 Section 3.** File 3
   states Option B is "LOW on WH/BH" because "Baby RISC-V GPRs are not
   exposed via any memory-mapped register." File 2 Section 3 documents the
   `RISC_DBG_CNTL_0/1` protocol which can read GPRs via `riscv_dbg_rd`.
   However, File 2 Section 3.6 notes these registers exist on all three
   architectures (WH/BH/Quasar). This seems contradictory: if RISC_DBG_CNTL
   can read registers on WH/BH, why does Option B say GPRs are not accessible?
   The resolution is that `riscv_dbg_rd/wr` are **device-side** functions
   (one RISC reading another RISC's registers), while Option B requires
   **host-side** access. File 2, Section 4.6 ("RISK CAVEAT") makes this
   distinction for Quasar, and File 2 Key Takeaway 5 states the registers
   "are currently only usable from on-device firmware." But this crucial
   distinction is not stated clearly in Option B's rationale. An implementer
   reading Option B's claim that "GPRs are not exposed" and then reading
   File 2's RISC_DBG_CNTL documentation would be confused.
   **Recommendation:** In File 3 Section 7.2 (Option B), change "Baby
   RISC-V GPRs are not exposed via any memory-mapped register" to "Baby
   RISC-V GPRs are not exposed to the **host** via any PCIe-accessible
   memory-mapped register (the RISC_DBG_CNTL interface documented in
   File 2 Section 3 is device-side only)."

3. **Ch5 File 3 effort estimates: Section 7.3.6 vs Section 12.** Section
   7.3.6 (line ~813) estimates Option C at "4-6 engineer-weeks (prototype),
   8-12 weeks (production)." Section 12 (line ~1060) estimates "Total for
   Option C GDB stub: ~3 person-weeks (prototype)." The prototype estimate
   disagrees: 4-6 weeks vs 3 weeks. **Recommendation:** Reconcile these
   estimates. If the Section 12 estimate excludes the trap handler (listed
   separately as 1 person-week), note that explicitly. Even then, 3+1=4
   person-weeks vs 4-6 engineer-weeks is only marginally consistent.

4. **Ch5 File 3 Section 8.1 vs Section 7.3.6 effort for ebreak option.**
   Section 8.1 (line ~877) estimates "Real HW via ebreak: 500-1,500 lines,
   2-4 weeks." Section 7.3.6 estimates "4-6 engineer-weeks (prototype)."
   The 2-4 weeks in Section 8.1 is more consistent with the 3-week estimate
   in Section 12, but conflicts with Section 7.3.6's 4-6 weeks.
   **Recommendation:** Use consistent ranges across all three locations.

5. **Ch5 File 2, Section 6 config register HW_CFG_SIZE=187 vs File 3
   capture bundle.** File 2 (line ~517) states HW_CFG_SIZE is 187 DWORDs
   (748 bytes). File 3's trace bundle (line ~424) lists `config_regs.bin`
   as "187 HW config registers + 3 thread configs." These are consistent.
   No issue.

## Terminology Alignment

1. **"Baby RISC-V" vs "RISC-V" vs "Tensix RISC-V."** Ch5 uses all three
   terms. File 3 Section 7.3.1 uses "Baby RISC-V cores" (line ~663), File 3
   Section 10.2 uses "Baby RISC-V architecture" (line ~1000), while File 2
   uses "RISC-V" generically and "Tensix RISC-V" in the Spike comparison
   table. Ch1 uses "RISC-V processors." Ch4 uses "RISC-V" throughout. The
   term "Baby RISC-V" is the Tenstorrent-internal name for the proprietary
   RISC-V variant. While not a factual error, the inconsistent terminology
   could confuse readers about whether these refer to the same processors.
   **Recommendation:** Minor. Consider standardizing on "Tensix RISC-V"
   with "(Baby RISC-V)" as a parenthetical on first use.

2. **"hart" vs "RISC" vs "processor" vs "thread."** Ch5 File 2 uses
   "thread" (as in "4 breakpoints per thread," "thread 0 = unpack, thread 1
   = math"), "RISC" (as in "8 watchpoints per RISC"), and "hart" (as in
   "up to 24 harts per Tensix on Quasar"). Ch4 uses "RISC" consistently.
   Ch2 uses "RISC-V processor." The Tensix "thread" in the breakpoint
   context (thread 0 = unpack, thread 1 = math) refers to Tensix coprocessor
   threads, NOT RISC-V harts. This is a critical distinction that Ch5
   File 2 Section 1.2 does clarify (bit 31: "0 = thread 0 (unpack), 1 =
   thread 1 (math)"), but the overloaded use of "thread" could cause
   confusion with RISC-V harts or GDB threads. **Recommendation:** Add a
   brief note in File 2 Section 1 clarifying that "thread" in the Tensix
   breakpoint context means a Tensix coprocessor instruction stream, not a
   RISC-V hart.

3. **"ebreak" capitalization.** Consistently lowercase throughout Ch5,
   matching RISC-V specification convention. Consistent with Ch4. No issue.

4. **"Option C" label consistency.** Ch5 File 3 labels the ebreak + mailbox
   approach as "Option C" throughout. The Ch5 index cross-references section
   also references "Option C." Ch4 does not reference these option labels
   (as expected, since they are introduced in Ch5). Consistent within Ch5.

5. **"Tier 1/2/3" overloading.** Ch5 File 1 Section 3 defines three tiers
   of assert mechanisms (Tier 1: Watcher Asserts, Tier 2: Lightweight Kernel
   Asserts, Tier 3: LLK Assert). Ch5 File 3 Section 9 defines a three-tier
   debug strategy (Tier 1: Host model, Tier 2: HW ebreak, Tier 3:
   Cycle-accurate simulator). Ch5 File 3 Section 5.2's CaptureHeader uses
   `capture_level` with TIER1/TIER2/TIER3. These are three different
   tier numbering schemes within the same chapter. **Recommendation:**
   Rename the assert tiers to avoid collision -- e.g., "Assert Level 1/2/3"
   or "Assert Tier" vs "Debug Tier" -- or add a disambiguation note.

## Forward References

1. **Ch6 (proposed interceptor design).** Ch5 references Ch6 in the index
   cross-references (line ~87: "Hardware capabilities documented here
   constrain the interceptor's capture scope and trigger mechanisms") and
   in File 1 (line ~425: "This gap motivates the runtime interceptor design
   (Ch6)"). These are reasonable forward references. Ch5 establishes what
   hardware primitives exist (breakpoints, debug arrays, watchpoints,
   RISC_DBG_CNTL), and Ch6 would logically use these to design the
   interceptor's capture triggers. The constraint relationship is valid.

2. **Ch7 (proposed replay debugger).** Ch5 index (line ~88: "The three
   replay targets define the replay debugger's architecture") and File 1
   (line ~426: "the replay debugger design (Ch7)"). Ch5 File 3 extensively
   evaluates replay targets (Spike, host model, hardware, simulator) and
   proposes the three-tier strategy. Ch7 would logically build on this
   assessment to specify the replay architecture. Valid forward reference.

3. **Forward reference quality.** The forward references are appropriately
   constrained -- Ch5 does not make assumptions about Ch6/Ch7 designs but
   instead establishes the "possibility space" that those chapters will
   operate within. This is good practice.

## Gap Analysis

1. **Ch2 hidden state not addressed in Ch5 emulation assessment.** Ch2
   Section 4 (implicit/hidden state) identifies firmware state, inter-RISC
   sync primitives, and Quasar NEO cluster state as capture requirements.
   Ch5 File 3's E-taxonomy (Section 4.2) covers semaphores (E-Stub), NOC
   transactions (E-Opaque), and config registers (E-Model), but does not
   explicitly classify the inter-RISC synchronization state from Ch2's
   `subordinate_sync` mailbox or Ch4's `RUN_SYNC_MSG_*` protocol. Since
   the host functional model replaces the firmware launch sequence entirely,
   this state might be implicitly unnecessary for Tier 1 replay. However,
   for Tier 2 (hardware replay), the kernel must be launched through the
   normal BRISC coordination, and the interceptor must capture the launch
   message state. **Recommendation:** Add a row to the E-taxonomy table
   for "BRISC subordinate sync / launch_msg_t" classified as E-Stub for
   Tier 1 (replaced by host harness) and E-Opaque for Tier 2 (must be
   replayed exactly as captured).

2. **Ch3 LightMetal schema extensions not referenced in Ch5.** Ch3
   Section 3 proposes `KernelSnapshotCommand`, `L1SnapshotCommand`,
   `SemaphoreSnapshotCommand`, `RegisterStateCommand`, and
   `KernelOutputSnapshotCommand` as FlatBuffer extensions. Ch5 File 3's
   trace bundle format (Section 5.2) defines a completely separate
   file-based capture format (`metadata.json`, `*.elf`, `*.bin`). There is
   no discussion of whether the trace bundle should use Ch3's FlatBuffer
   schema or a standalone format. **Recommendation:** Add a note in Ch5
   File 3 Section 5.2 explaining the relationship between the file-based
   trace bundle shown and Ch3's proposed FlatBuffer schema. Either the
   bundle is an export format derived from FlatBuffer captures, or it is
   an alternative format -- either way, the relationship should be stated.

3. **Ch4 hook points (H1-H6) not referenced in Ch5.** Ch4 identifies six
   interception points and recommends the H1+H4 dual-hook architecture.
   Ch5's capture workflow (File 3, Section 5.2) describes capturing "LLK
   API calls" and "Tensix config register state" but does not specify which
   of Ch4's hook points would be used. For a chapter titled "Debug
   Infrastructure," it would be useful to connect the hardware capabilities
   (Ch5) with the software interception points (Ch4).
   **Recommendation:** Add a brief note in File 3 Section 5.2 linking the
   capture phase to Ch4's recommended H4 hook point.

4. **tt-exalens coverage vs Ch1 GPU debugger patterns.** Ch1 establishes
   that GPU debuggers follow a layered architecture (frontend, debug API,
   driver, hardware). Ch5 File 1 documents tt-exalens as having silicon and
   simulator backends with breakpoints, callstack unwinding, and register
   access. However, Ch5 does not map tt-exalens against Ch1's extracted
   patterns to assess how close it already comes to a "Tenstorrent cuda-gdb
   equivalent." This would be valuable context for Ch6/Ch7.
   **Recommendation:** Minor. Consider adding a paragraph in File 1
   Section 8 mapping tt-exalens capabilities to Ch1's common debugger
   patterns.

## Verdict

**PASS WITH NOTES**

Ch5 is largely consistent with Ch1-Ch4 and internally coherent. The most
significant issues are:

- The Quasar TRISC3 omission in the trace bundle format (Cross-Chapter
  Issue 3) repeats a pattern flagged in Ch1's review and could lead to
  data loss on Quasar captures.
- The internal effort estimate inconsistency for Option C (Internal
  Consistency Issue 3-4) would confuse an implementer trying to plan work.
- The "Tier" terminology overloading within a single chapter (Terminology
  Issue 5) is a readability concern.
- Two cross-references to "(Ch 2, Ch 3)" incorrectly attribute the
  interceptor design to those chapters instead of Ch6 (Cross-Chapter
  Issues 4-5).
- The GPR accessibility claim in Option B could confuse readers who also
  read File 2's RISC_DBG_CNTL documentation (Internal Consistency Issue 2).

None of these rise to the level of FAIL -- there are no fundamental
architectural contradictions or incorrect hardware claims that would lead
an implementer to build something broken. The issues are primarily about
precision, disambiguation, and cross-reference accuracy.
