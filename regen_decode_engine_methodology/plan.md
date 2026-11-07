# Final Plan — Guide: Regen Decode Engine Methodology (records-as-data persistent interpreter)

Synthesized by the Plan Synthesizer from candidate plans v1–v6 and their evaluations.

---

## 1. Selection Rationale

**What was chosen from which plans, and why:**

- **Skeleton (all six plans, converged):** 8 chapters, foundational→advanced — Ch1 orientation; Ch2–7 map
  one-to-one onto briefing Questions 1–6; Ch8 is the fenced porting chapter (Q7). Every evaluator confirmed
  this ordering and one-primary-owner coverage; no plan deviated structurally, so the skeleton is adopted
  as-is.
- **Ch1 from v6 + v2 + v1 + v4 + v5 (hybrid):** v6's `anatomy-of-a-token` walkthrough (evaluated "the best
  orientation file in the field" — eval-v5; eval-v6's own weakness list confirms it needed more); v2/v4's
  file-by-file **source map** of the 1.2 MB tree (eval-v1 flagged its absence in v1 as a real gap against the
  "contribute without reading the source" bar); v1's **model-shape + DRAM-bound arithmetic** file (eval-v3 and
  eval-v6 both demanded it); v5's **measured-envelope-as-accounting-targets** framing plus v3's
  **citation-discipline worked examples** in that same file.
- **Ch2 from v5/v3/v2 + all plans:** dedicated files for step records, interpreter loop, **lead/helper team
  jobs** (eval-v2 demanded promotion out of the loop file — "every other plan gives it a dedicated file"),
  read-ahead rings, and a dedicated **elimination-argument file** (v2 `no-phases` / v3 `host-elimination` /
  v5 `no-host-in-loop`; eval-v4 called its absence v4's biggest Q1 weakness). The Ch2-ring vs Ch4-strip-claim
  boundary is pinned per eval-v2 mod 4.
- **Ch3 from v1/v4/v5 + v6:** dedicated **`op-bodies` file** (eval-v3, eval-v6: "a port contributor cannot
  compose or port an op without it"), v4's **mechanism-in-Ch3 / performance-in-Ch4 split** (eval-v4's
  duplication fence called "the clearest of any plan"), v6's **engine-vs-LLK give-up/gain table** explicitly
  built as a Ch8 input, and the smoke-gate/oracle-machinery evidence that eval-v1 found missing in v1.
- **Ch4 from v4 + v2 + v3-fix:** v4's dedicated **NoC-ID budgeting** treatment and its five-file shape
  (eval-v4: "the only plan that gives NoC-ID budgeting its own file, matching the question's wording");
  v2's **`batch-economics` placement with the correct causal chain** (routed-expert unions destroy strip
  reuse) — chosen over v5's transport-rounds framing by evals 2, 3, and 5; the file carries the **full MTP
  recovery numbers** (eval-v1 and eval-v4 both flagged MTP as missing); v3's rename fix: **prefetch**-windows,
  never "prefill"-windows (eval-v3 mod 3).
- **Ch5 from v4 + v1-ordering:** v4's **latency compendium** (one table of every measured transport number
  with host context and per-row [MEAS] citations — eval-v4: "better than v2 scattering them"), but v1's file
  order (both transport mechanism files *before* the selection rule, per eval-v1's note that v4 front-loads
  topology before the reader has seen either transport).
- **Ch6 from v1/v2/v4 split:** backend-open-and-load, TTQ/requant, placement, record-builder-and-serving, and
  CPU oracle as **five separate files** — eval-v5 mod 1 required splitting v5's overloaded single file
  ("cannot hold both within the plan's own 500-line cap"). Strip-format ownership boundary Ch4↔Ch6 pinned
  per eval-v5 mod 5.
- **Ch7 from v1 + v4:** v1's explicit **determinism chain** (pinned toolchain → static_assert-pinned
  instruction counts → deterministic image layout → replayable records; eval-v1: "the most explicit causal
  argument of any plan") combined with v4's checked-in-generated-code and smoke-gate invariants.
- **Ch8 from v4 + v3 + v2 + v5:** v4's **contributor onboarding** (task→chapter→source map, with port traps)
  plus v3/v4's **rederivation playbook** (invariant → tt-metal mechanism → re-measure, with worked examples)
  plus v2's **re-measure rule** ("a port never inherits regen's registry values" — eval-v2: "the strongest
  single sentence in any plan").

**Evaluation fixes incorporated (all of them):**
1. MTP/batching economics as a real dedicated file with the full registry set (38.7 t/s/user, 40 distinct,
   ~3.8×, 38.7→53.1 = +37%, acceptance 0.892) — evals 1, 3, 4, 5. Placed in Ch4 (v2's causal chain).
2. Ch1 source-map file — evals 1, 2, 4.
3. Ch1 token walkthrough — evals 1, 2, 5 (borrowed from v6/v3).
4. Ch1 model-shape/workload file with DRAM-bound arithmetic — evals 3, 6.
5. Ch8 contributor checklist — evals 1, 2, 6.
6. Lead/helper-teams as a dedicated Ch2 file — eval 2.
7. Dedicated elimination-argument file in Ch2 — evals 1 (implicit), 4.
8. Dedicated op-bodies file in Ch3 — evals 3, 6.
9. Window-terminology disambiguation (SFPU replay window / DRAM prefetch window / PCIe BAR window) — evals
   1, 2, 3, 5, 6.
10. Explicit non-duplication/number-ownership rules — evals 1, 6.
11. Verify-first flags on every non-registry number (layer counts, GB/token figures, smoke PASS lines, load
    times, `--prefill-chunk`, byte-class shares, soak counts) — evals 2, 3, 4, 5, 6.
12. Rename prefill-windows → prefetch-windows — eval 3.
13. Smoke-gate + oracle-comparison machinery in the correctness file — eval 1.
14. Engine-vs-LLK tradeoff file in Ch3 as a Ch8 input — evals 4, 6.
15. Split Ch6's overloaded host-driver file — evals 5, 6.
16. Latency compendium in Ch5 — eval 4.
17. Ch6↔Ch1 record-builder duplication fence; Ch2-ring vs Ch4-claims boundary — evals 2, 6.
18. Numbered file prefixes, per-file abstracts + scope lines, ~500-line cap — evals 2, 3, 5, 6.
19. Single evidence tags only (no hybrid [MEAS/INFER]) — eval 6.
20. Port re-measure checklist — eval 2.
21. Determinism chain in Ch7 — eval 1.
22. Prefetch-window qualifier in glossary — eval 1.

---

## 2. Audience

TT/tt-metal engineers and program agents bringing up the GLM-5.3-Flash decode port (Path 2,
`doc/research/regen_kernel_port.md`) who have **not** read regen's ~1.2 MB source tree.

**They already know:**
- Tensix/Blackhole hardware at the tt-metal level: core grid, BRISC/TRISC/ERISC split, L1/LDM, NoC, DRAM
  channels, tiles, BFP4/BFP8/BF16.
- tt-metal workflow concepts: device open, kernel launch, buffers/alloc, circular buffers, llk, traces,
  program cache.
- General LLM decode architecture and the GLM-5.3-Flash model shape at config level (KDA, MLA/DSA, mHC,
  288-expert MoE) — the guide carries a workload digest but does not re-teach the model.

**Not assumed:** any regen exposure — no step records, raw MMIO programming, SFPU replay windows, strip
streaming, peer-BAR DMA, TTQ, or the sfpi pin. Every chapter opens from the reader's tt-metal mental model
and contrasts regen's approach with what tt-metal provides.

**Success bar:** a reader who finishes understands regen's engine well enough to contribute to the tt-metal
port without reading the 1.2 MB source — i.e. can locate any subsystem in the tree via the source map,
re-derive regen's measured decisions, and argue stays/comes/new for any port question.

---

## 3. Chapter list (8 chapters, foundational → advanced)

File names inside each chapter are numbered (`01-…`) to make reading order mechanical. Every chapter is an
`index.md` plus 2–6 topic files. `[verify]` marks numbers NOT in the orchestrator registry: the writer MUST
locate and verify each in the cited notes/source file before writing it, then cite the file (and line where
useful).

### Chapter 1 — The Engine at a Glance (`ch1_engine_at_a_glance`)

One-sentence: the mental model of regen's decode engine — one persistent BRISC interpreter per Tensix core
executing step records as data, with the host out of the token loop — plus the workload, the source map, the
measured envelope, and one token walked end-to-end.

Files:
- `index.md` — chapter purpose, reading paths (port contributor / perf engineer / new agent), glossary of
  fixed terms (see Conventions), links to all files, forward map to Chapters 2–8.
  - Primary evidence: `README.md`; briefing registry.
- `01-mental-model.md` — the core inversion vs a conventional tt-metal stack: records-as-data executed by
  one persistent interpreter per core; no phase boundaries, no per-step dispatch, no llk, no CBs, TRISCs
  idle, host absent from the token loop; a side-by-side contrast table (conventional host-dispatch
  kernel-launch vs regen). The five design principles as a checklist, each with one paragraph and a forward
  pointer to its chapter. Headline payoff numbers as motivation: 165.3 t/s = 6.05 ms/token single-user;
  plan0h 178.9 t/s = 5.59 ms [MEAS, notes/post-rc/glm-rc1.md lines 136-139]; README context-decay table
  165.9 → 128.4 t/s [MEAS, README.md "Performance"].
  - Primary evidence: `src/brisc.cpp` (record loop structure), `src/constants.h`, `README.md`,
    `notes/post-rc/glm-rc1.md`.
- `02-anatomy-of-a-token.md` — walk ONE decode token end-to-end at 50,000 ft: record fetch → op bodies
  (matvec, norms, router, SwiGLU, mHC, KDA, MLA, rounds) → cross-chip collectives → token emit; where each of
  the 6.05 ms goes at the roofline level (decode reads ~2.10 GB/chip/token [verify in README.md
  "Roofline"/notes]; DRAM busy 79-95% of the token [MEAS, notes/glm-decode-roofline.md §5 lines 108,
  177-181] — cited here, owned by Ch4). Chapters 2–5 each open by locating their subsystem on this walk.
  - Primary evidence: `src/brisc.cpp`, `src/constants.h`, `notes/glm-decode-roofline.md`, `README.md`.
- `03-workload-context.md` — the GLM-5.3-Flash layer map (34 KDA / 11 MLA+DSA layers, dense FFN at layers
  0-2, 42 MoE layers with 288 experts top-8 plus shared, mHC blocks, untied head [verify all counts in
  notes/glm-inventory.md before writing]) and the arithmetic showing decode is DRAM-bandwidth-bound:
  per-token byte totals, only 8 of 288 experts active per token per layer, BFP strip packing (4.5 / 8.5
  bits/weight [verify in notes/glm-inventory.md "Packing assumed"]) — the motivation substrate for Ch4 and
  the decoder ring for Ch3's op names.
  - Primary evidence: `notes/glm-inventory.md`, `notes/glm-scores.md`, `README.md`, `doc`-side model context.
- `04-source-map.md` — the file-by-file map of the 1.2 MB tree with line counts and responsibilities
  (`brisc.cpp` 23,665 — device interpreter; `tt_backend.cpp` 38,808 — host backend; `constants.h` — records,
  command ids, L1 `_ADDR` map; `erisc.cpp` 609 + `erisc.ld` + `erisc_entry.S`; `fw.h` 84 — MMIO register
  layer; `kernels.sfpu` + `tools/sfpu.py`; `requant.cpp`; `glm.cpp`; `main.cpp`/`serve.cpp`; `brisc.ld`/
  `brisc_entry.S`; `ckernel_ops.h`; `rules.py`/`make.py`; `tests/tt_kernels.cpp`), which file owns which
  subsystem, where each guide chapter points into it, which anchor symbols to search for (e.g.
  `TtGlmStepRecord` in `constants.h`, op tables in `ckernel_ops.h`), what can safely never be read (bench
  harness paths, proxy/sim paths) and why, and a notes-corpus map (~80 files → which chapters cite which).
  After this file, a reader never opens the big files linearly.
  - Primary evidence: the source tree inventory itself (read-only listing), `README.md`, `docs/README.md`.
- `05-measured-envelope-and-evidence.md` — the measured envelope stated UP FRONT as the set of numbers the
  rest of the guide must account for: 165.3 t/s = 6.05 ms; plan0h 178.9 t/s = 5.59 ms [MEAS,
  notes/post-rc/glm-rc1.md lines 136-139]; 447 GB/s per chip = 89% of the 500 GB/s peak inside streaming
  windows [MEAS, notes/glm-decode-roofline.md §5 lines 108, 177-181]; weight files hot3-meas 245.7 GB
  (5153 hot experts = 42.6% at BFP8) vs plan0h 180.9 GB all-BFP4 [MEAS/FACT, README.md "Weights"]; image
  budget ≤ 0x16000 = 88 KB [FACT, src/brisc.ld]. Establishes the [MEAS]/[EST]/[INFER]/[FACT] citation
  discipline with two or three worked examples (how to cite, how to verify a non-registry number, why hybrid
  tags are banned).
  - Primary evidence: briefing registry; `README.md`; `notes/post-rc/glm-rc1.md`; `notes/glm-decode-roofline.md`.

Questions covered: none primarily (shared foundation); previews Q1–Q7 and anchors every later chapter.

### Chapter 2 — The Record-Driven Execution Model (`ch2_record_driven_execution`)

One-sentence: how `TtGlmStepRecord` step lists plus one persistent per-lead BRISC interpreter loop, with
lead/helper team jobs and DRAM read-ahead rings, turn decoding into data interpretation — eliminating phase
boundaries, per-step dispatch, and the host from the token loop.

Files:
- `index.md` — chapter purpose; Q1 in scope; the elimination argument stated as three testable claims
  (no phases / no per-step dispatch / no host in the token loop) that `05-eliminating-the-runtime.md`
  substantiates; the exact Ch2-vs-Ch4 boundary: ring geometry and record-stream prefetch live HERE;
  two-strips-in-flight, NoC-ID budget, and window rates live in Ch4.
  - Primary evidence: `src/constants.h`, `src/brisc.cpp`.
- `01-step-records.md` — the `TtGlmStepRecord` format and the record/step/trace/progress struct family in
  `src/constants.h`: field layout, command-id space, embedded operand addresses (the L1 memory map `_ADDR`
  symbols), how a whole decode step is encoded as one flat per-core list, image build flags gating record
  variants, and why records-as-data beats code-as-program (patchable, host-auditable, no device recompiles,
  extensible without re-flashing). [INFER, src/constants.h] per structure.
  - Primary evidence: `src/constants.h`; cross-ref `src/main.cpp` (builder side, owned by Ch6).
- `02-interpreter-loop.md` — the per-lead persistent loop in `src/brisc.cpp`: fetch → decode command id →
  dispatch to op body → advance; the single resident program per core; what state persists across tokens and
  where it lives in L1; trace/progress records as the in-band observability mechanism; contrast with a
  conventional launch-per-kernel model; the "record list IS the trace" insight (no capture-vs-replay
  distinction). [INFER, src/brisc.cpp].
  - Primary evidence: `src/brisc.cpp` (record loop, trace/progress record paths), `src/constants.h`.
- `03-lead-helper-teams.md` — lead vs helper core roles and team jobs: which core leads a step, how helpers
  are assigned work within a step, mailbox/relay handshakes between peers, how the record stream is shared or
  relayed to helpers, how team structure substitutes for tt-metal's per-op grid dispatch, and how leads
  interface with cross-chip rounds (forward pointer to Ch5). Notes are largely silent here — [INFER,
  src/brisc.cpp]/[INFER, src/constants.h] tagging is mandatory and strict.
  - Primary evidence: `src/brisc.cpp` (team job structures, role branches), `src/constants.h` (team/step
    structs).
- `04-read-ahead-rings.md` — DRAM read-ahead rings feeding the interpreter its own records: ring structure
  and slot geometry, how far ahead the loop prefetches, claims-ahead pacing at the RECORD level, how the
  ring absorbs DRAM latency so fetch never gates compute, and the pointer (not restatement) to Ch4's strip
  claims-ahead — the same mechanism family generalized to weight streams.
  - Primary evidence: `src/brisc.cpp` (ring construction), `src/constants.h` (ring structs/_ADDR symbols).
- `05-eliminating-the-runtime.md` — the three eliminations as one argued claim: (a) no phase boundaries —
  records chain continuously, no init/forward boundary; (b) no per-step dispatch — the interpreter is
  already resident, nothing is launched per step; (c) no host in the token loop — the host builds/loads
  records once per batch configuration and reads results, nothing per token. Contrast with a
  dispatch-per-step engine's cost structure; connect to the measured envelope (no dispatch-overhead term in
  the roofline) tagged [INFER, notes/glm-decode-roofline.md] — the connection is an inference, the roofline
  numbers are [MEAS]. Sets up Ch8's stays/comes split.
  - Primary evidence: `src/brisc.cpp`, `src/main.cpp`/`src/serve.cpp` (boundary of host involvement),
    `notes/glm-decode-roofline.md`.

Questions covered: **Q1 (primary)**.

### Chapter 3 — Raw-MMIO Compute and SFPU Replay (`ch3_raw_mmio_sfpu_replay`)

One-sentence: how the engine drives Tensix compute directly — cfg register writes at 0xFFEF0000, the
instruction port at 0xFFE40000, SFPU replay windows compiled by `tools/sfpu.py` — with no llk, no circular
buffers, and the TRISCs idle, plus the correctness disciplines that make it safe.

Files:
- `index.md` — chapter purpose; Q2 in scope; the "engine as its own runtime" thesis: raw MMIO as a
  *methodology choice* (deterministic, dispatch-free, CB-free compute), and the safety framing — raw MMIO is
  only viable with the disciplines in `04`/`05`; the mechanism-vs-performance fence: op MECHANISM lives here,
  op PERFORMANCE lives in Ch4's ledger.
  - Primary evidence: `src/fw.h`, `src/brisc.cpp`.
- `01-mmio-register-layer.md` — the register layer from `src/fw.h` (84 lines, the whole layer):
  `TENSIX_CFG_BASE` 0xFFEF0000, `TENSIX_INST_BASE` 0xFFE40000, `NOC0/NOC1_BASE` 0xFFB20000/0xFFB30000
  [FACT, src/fw.h]; what a cfg-word write is mechanically (data formats, tile sizes, unpack/pack state,
  accumulator setup), how the engine sequences cfg writes per op — doing what llk/ckernel init code would
  otherwise do — and how BAR0 windows in `tt_backend.cpp` expose the register space at load time.
  - Primary evidence: `src/fw.h`, `src/brisc.cpp` (config layer), `src/tt_backend.cpp` (BAR0 windows).
- `02-instruction-port.md` — driving compute through the instruction port at 0xFFE40000: what an
  instruction write is, how the interpreter streams packed instructions, how operand setup is done with plain
  L1 stores, what "TRISCs idle" means operationally (BRISC issues everything, Tensix executes; a choice,
  not a limitation), and why there are no CBs (data flows L1/DRAM→compute directly; readers own all
  synchronization — forward pointer to Ch4). Distinguish [FACT] addresses from [INFER] sequencing claims.
  - Primary evidence: `src/fw.h`, `src/brisc.cpp` (instruction layer), `src/ckernel_ops.h`.
- `03-sfpu-replay-windows.md` — the SFPU pipeline end-to-end: `src/kernels.sfpu` fragment source →
  `tools/sfpu.py` fragment compiler → emitted `kernels_sfpu.inc` **SFPU replay windows** with
  static_assert-pinned instruction counts; replay semantics (a window is a pre-compiled, count-pinned
  instruction stream the interpreter replays into the SFPU pipe), which decode ops consume windows (exp,
  silu, sigmoid-class activations), and the fragment ISA subset the compiler enforces; the sfpi version pin
  (regen 7.76.0, GCC 15.1; re-verified under 7.86.0 with the xtttensixbh march extension [FACT, rules.py;
  cross-check regen_kernel_port.md M0]).
  - Primary evidence: `src/kernels.sfpu`, `tools/sfpu.py`, `src/constants.h` (image build flags),
    `rules.py`.
- `04-op-bodies.md` — the op bodies as compositions of cfg writes + replay windows + NoC moves, using the
  op/ckernel tables in `src/ckernel_ops.h` as the auditable menu: matvec, RMSNorm/norms, router, SwiGLU
  (clamped), mHC, KDA recurrence, MLA/indexer + absorbed attention, MoE combine, and the "rounds" pattern;
  register/accumulator conventions per op family; 2–3 worked examples at the record-command level (one norm,
  the router, one KDA recurrence step). Mechanism only — each op's PERFORMANCE ledger entry belongs to Ch4.
  - Primary evidence: `src/brisc.cpp` (op bodies), `src/ckernel_ops.h`, `src/constants.h` (command ids);
    op semantics cross-ref Ch1 `03-workload-context.md`.
- `05-correctness-discipline.md` — the safety net in lieu of llk, with the full evidence chain:
  static_assert-pinned instruction counts (codegen drift becomes a BUILD failure, not silent silicon
  corruption), deterministic window sizing, startup probes (the SrcA bank test and siblings) validating
  MMIO assumptions before the loop starts, the smoke gate (`tests/tt_kernels.cpp`; PASS-line counts
  [verify in README.md §4 before writing]) as the post-build invariant, and how the `glm.cpp` CPU oracle
  closes the loop end-to-end (mechanics owned by Ch6 `05-cpu-oracle.md`); how failures manifest (probe
  abort / assert halt, not wrong numbers). Checklist form for a port engineer.
  - Primary evidence: `src/kernels.sfpu` (static_asserts), `tools/sfpu.py`, `src/brisc.cpp` (startup
    probes), `tests/tt_kernels.cpp`, `README.md` §4.
- `06-what-no-llk-means.md` — the full no-llk/no-CB contract as an argued trade: what regen gives up (CB
  flow control, maintained llk math, portability, tested abstractions) vs what it gains (no reconfig
  stalls, no CB sync races, exact instruction budgets, byte-stable windows); a comparison table
  engine-vs-llk that Ch8's stays/comes/new analysis reuses verbatim.
  - Primary evidence: `src/brisc.cpp` vs llk concepts; `src/fw.h`; cross-ref `notes/` for any measured
    reconfig/CB-cost comparisons found (writer verifies).

Questions covered: **Q2 (primary)**; instruction-count pins recur in Ch7.

### Chapter 4 — DRAM Streaming Discipline (`ch4_dram_streaming_discipline`)

One-sentence: the strip-claims-ahead, two-strips-in-flight, near-channel streaming pattern — with prefetch
windows and NoC-ID budgeting — that holds DRAM at 447 GB/s = 89% of peak inside windows and took the engine
from 9.3 to 6.05 ms/token.

Files:
- `index.md` — chapter purpose; Q3 in scope; the roofline thesis (decode on this model is a DRAM problem,
  per Ch1 `03-workload-context.md`): streaming discipline IS the performance methodology; the headline
  numbers up front with citations; the ownership fence: ALL DRAM throughput numbers live in this chapter.
  - Primary evidence: `notes/glm-decode-roofline.md`, `notes/glm-decode-ops.md`.
- `01-roofline-and-byte-budget.md` — why decode is DRAM-bound: per-class byte inventories from
  `glm-inventory.md`/`glm-scores.md` turned into a per-chip byte budget (class-share percentages
  [verify in notes/glm-inventory.md before writing]); the per-op time table and sync analysis from
  `glm-decode-roofline.md`; the plan requirement that DRAM stay busy 79-95% of the token and 447 GB/s = 89%
  of the 500 GB/s peak inside windows [MEAS, notes/glm-decode-roofline.md §5 lines 108, 177-181]; why the
  plan is stated as a DRAM-busy fraction, not a FLOP roofline.
  - Primary evidence: `notes/glm-decode-roofline.md`, `notes/glm-inventory.md`, `notes/glm-scores.md`.
- `02-strip-claims-and-reader-placement.md` — the streaming mechanics: the strip as the unit of transfer
  (geometry owned here as the DRAM currency/reader contract: 16-row BFP strips; byte formats cross-referenced
  to Ch6, not restated), claims-ahead scheduling, exactly two strips in flight per reader and WHY two is
  the number (bandwidth vs L1 footprint), near-channel reader placement (readers pinned near the DRAM
  channel their strips live in), and how claims-ahead overlaps fetch with compute — tying back to Ch2's
  read-ahead rings without duplicating them.
  - Primary evidence: `src/brisc.cpp` (NoC layer, claim protocol), `src/tt_backend.cpp` (placement side),
    `notes/glm-decode-roofline.md`.
- `03-prefetch-windows-and-noc-id-budget.md` — DRAM **prefetch windows** (always qualified — see
  Conventions): how traffic is batched into windows around attention/round phases, window sizing so compute
  never starves, effective per-window rates; and NoC-ID budgeting from `notes/noc-ids.md`: outstanding
  transaction IDs as a finite scheduling resource, allocation across weight streams / record fetch /
  cross-chip relays, and the failure mode when budgeting is wrong. NOTE: this chapter is decode-streaming
  only; prefill-phase design is out of guide scope and fenced out by name.
  - Primary evidence: `notes/noc-ids.md`, `notes/glm-decode-roofline.md`, `src/brisc.cpp` (NoC layer).
- `04-optimization-ledger.md` — the 9.3 → 6.05 ms/token ledger as the worked example of how the discipline
  was won: base 9295/9925 us (plan0h/hotE) → 8312/9021 us through the named change-by-change table, ending at
  165.3 t/s = 6.05 ms and plan0h 178.9 t/s = 5.59 ms [MEAS, notes/glm-decode-ops.md "Changes" table;
  notes/post-rc/glm-rc1.md lines 136-139]; each ledger entry classified as a streaming, placement, compute
  (Ch3 pointer), or sync fix — the op-performance companion to Ch3's mechanism catalog.
  - Primary evidence: `notes/glm-decode-ops.md`, `notes/post-rc/glm-rc1.md`, `notes/glm-decode-roofline.md`.
- `05-batch-and-mtp-economics.md` — why the same discipline punishes batching and how MTP recovers it, told
  as ONE causal story: batching on one group unions routed-expert sets across users, destroying per-strip
  reuse and multiplying per-user bytes (~3.8× loss vs one user per group [MEAS, notes/glm-glx-layouts.md
  §6]); B=8 plain ≈ 38.7 t/s/user, B=4 distinct-prompt 40 t/s/user [MEAS, notes/glm-glx-mtp2.md results
  table]; MTP at B=8 greedy lifts 38.7 → 53.1 t/s/user = +37% with draft-1 acceptance 0.892 greedy (0.876
  at T=1 sampled; 0.802-0.84 at long context) [MEAS, notes/glm-glx-mtp2.md lines 119, 122-129]; the
  transport-round multiplication of speculative decoding mentioned only as a cross-referenced secondary
  angle (owned by Ch5). Sets up the serving-shape decision a porter faces.
  - Primary evidence: `notes/glm-glx-mtp2.md`, `notes/glm-glx-mtp3.md`, `notes/glm-glx-layouts.md`,
    `notes/glm-batch.md`.

Questions covered: **Q3 (primary)**; strip format facts shared with Ch6 under the pinned boundary.

### Chapter 5 — Cross-Chip Transports (`ch5_cross_chip_transports`)

One-sentence: the two transports — device-originated PCIe peer-BAR DMA (peer BAR regions, lead relays) vs
the ERISC Ethernet message service (rings, credits, binomial trees, restart-correct counters) — their
mechanics, their measured latencies per host, and the selection rule for which carries which collective.

Files:
- `index.md` — chapter purpose; Q4 in scope; the communication requirements of a multi-chip decode round
  (who must talk to whom per layer kind — router bits, MoE combine, barriers); the ownership fence: ALL
  transport latency numbers live in this chapter. Mechanism files first, decision rule after.
  - Primary evidence: `notes/glm-ccl.md`, `notes/glm-eth.md`, `src/erisc.cpp`.
- `01-pcie-peer-bar-dma.md` — device-originated PCIe peer-BAR DMA: peer BAR region setup (host side in
  `tt_backend.cpp`, device side in `brisc.cpp`), the lead-relay pattern carrying results between chips with
  no host CPU in the path (Ch2's no-host rule applied to I/O), placement of peer windows in the L1/DRAM map,
  and why device-originated beats host-relayed for the token loop; measured latencies: 0.50 us same-socket
  / 0.55 us cross-socket one-way, 16 B all-to-all ~1.02 us, mesh barrier ~1.74 us [MEAS, notes/glm-ccl.md
  lines 36, 211, 555]; host/socket-topology dependence made explicit.
  - Primary evidence: `src/brisc.cpp` (peer BAR/relay), `src/tt_backend.cpp` (BAR windows), `notes/glm-ccl.md`.
- `02-erisc-message-service.md` — the ERISC Ethernet message service from `src/erisc.cpp` (609 lines) +
  `src/erisc.ld` + `src/erisc_entry.S`: `TtEthMsg` format (struct in `constants.h`), send/receive rings,
  credit flow control, binomial tree collectives, restart-correct counters (why counters must survive
  collective restarts without double-counting), and how the BRISC interpreter hands messages to ERISC and
  receives completions; reliability model (credits + counters) contrasted with PCIe; design notes from
  `glm-eth.md`.
  - Primary evidence: `src/erisc.cpp`, `src/erisc.ld`, `src/erisc_entry.S`, `src/constants.h`,
    `notes/glm-eth.md`.
- `03-rounds-and-transport-selection.md` — how decode rounds use each transport: which steps of the Ch1
  token walk cross chips; in-situ ERISC rounds 2.84-3.5 us (lean tree post) [MEAS, notes/glm-glx-ethdec.md
  lines 57-58]; the decision rule — which collectives (small latency-critical all-to-all vs larger/tree
  flows; LoudBox same/cross-socket vs Galaxy cabled-Ethernet fabric realities) ride which transport per
  host, how `topology.*` feeds the choice, and how the lead chooses or the record encodes it. Keeps
  LoudBox and Galaxy latency classes distinct (transport latencies are host-specific). Cross-references (not
  restates) Ch4's batching economics as the demand side.
  - Primary evidence: `notes/glm-ccl.md`, `notes/glm-eth.md`, `notes/glm-glx-ethdec.md`,
    `notes/glm-glx-layouts.md`, `src/topology.{h,cpp}`.
- `04-latency-compendium.md` — ONE table of every measured transport number with its host context and
  per-row [MEAS] citations: the PCIe figures above, ERISC in-situ rounds 2.84-3.5 us, any additional
  per-link or per-host figures the writer verifies in `glm-ccl.md`/`glm-glx-ethdec.md`/`glm-eth.md`
  ([verify] each before writing); a short "what the port must re-measure on its own host" pointer (owned by
  Ch8's checklist).
  - Primary evidence: `notes/glm-ccl.md`, `notes/glm-glx-ethdec.md`, `notes/glm-eth.md`.

Questions covered: **Q4 (primary)**.

### Chapter 6 — Host Driver, Weights Pipeline, and the CPU Oracle (`ch6_host_driver_weights_oracle`)

One-sentence: what the host actually does — open devices, build the record list, quantize and load weights,
and verify — via `tt_backend.cpp`/`main.cpp`/`serve.cpp`, the TTQ container with its requant classes,
channel-local placement, and `glm.cpp` as the correctness oracle; everything except the token loop.

Files:
- `index.md` — chapter purpose; Q5 in scope; the division of labor (host owns setup, weights, admission,
  verification; device owns the token loop — recap pointer to Ch2); the ownership fence: ALL weights-format
  numbers live here; the strip-format boundary: Ch4 owns strip GEOMETRY as the reader contract, this
  chapter owns requant-CLASS selection and hot-expert policy; each links, neither restates.
  - Primary evidence: `README.md`, `src/requant.cpp`, `src/tt_backend.cpp`.
- `01-backend-open-and-load.md` — `src/tt_backend.cpp`/`tt_backend.h`/`topology.{h,cpp}`: KMD/ioctl open,
  BAR0 windows, firmware load, device enumeration/topology, the bench harness and proxy/sim paths (what they
  are and why the guide's reader can skip them), the dedicated-host/legacy-KMD ownership discipline
  [verify exact rules in README.md §1], and load-time facts (cached load time [verify in README.md before
  writing]) — the level an engineer needs to run and instrument the engine, not a feature manual.
  - Primary evidence: `src/tt_backend.cpp`, `src/tt_backend.h`, `src/topology.{h,cpp}`, `README.md`,
    `docs/README.md`.
- `02-ttq-container-and-requant.md` — the TTQ container from `src/requant.cpp`/`requant.h`: requant classes
  per tensor group and WHY (BFP4 strips for routed experts, BFP8 strips for the rest, BF16 router and
  indexer gate, source-precision small tensors, hot experts), imatrix-weighted BFP4 exponents, hot-expert
  selection from calibration texts, strip byte formats from `glm-bfp8.md` (geometry cross-referenced to
  Ch4); the two production files: hot3-meas 245.7 GB (5153 hot experts = 42.6% of experts at BFP8) vs
  plan0h 180.9 GB all-BFP4 [MEAS/FACT → cite as two tags in two claims: sizes MEAS README.md "Weights",
  formats FACT src/requant.cpp]; quality trade tables [verify in README.md].
  - Primary evidence: `src/requant.cpp`, `src/requant.h`, `README.md` "Weights", `notes/glm-weights.md`,
    `notes/glm-bfp8.md`.
- `03-channel-local-placement.md` — the weights loader: reading the TTQ, computing channel-local strip
  placement per chip so Ch4's near-channel readers see sequential streams (the contract between host
  placement and device reading — closes the loop Ch2/Ch4 opened, with a back-pointer, no restatement),
  replication decisions (e.g. f_a/g_a, MLA latents [verify in notes/glm-inventory.md/glm-layout.md]), and
  capacity accounting (users × context table [verify in README.md]).
  - Primary evidence: `src/tt_backend.cpp` (placement/loader), `notes/glm-layout.md`,
    `notes/glm-inventory.md`.
- `04-record-builder-and-serving.md` — `src/main.cpp` (1,445 lines) / `src/serve.cpp` (5,200 lines): how
  the host compiles model + weights + topology into `TtGlmStepRecord` lists (feeding Ch2's interpreter — the
  builder side of the record format; the FORMAT itself is owned by Ch2), rounds construction, the serving
  loop (sessions/slots), prefill chunking (`--prefill-chunk` value [verify in source before writing]), MTP
  orchestration at the host edge (economics owned by Ch4), and the exact boundary where host involvement
  ends per token.
  - Primary evidence: `src/main.cpp`, `src/serve.cpp`, `src/constants.h`.
- `05-cpu-oracle.md` — `src/glm.h`/`glm.cpp` (490 lines) as the correctness oracle: what the CPU reference
  computes, how device output is checked against it (per-op and end-to-end; fake-quantized reference parity
  [verify claim wording in README.md "Token similarity"]), how exactness/soak rounds are structured (counts
  [verify in README.md Galaxy section before writing]) and how the smoke suite wires oracle comparisons
  (`TT_LAB_CHECK` [verify in source/README]); how a port engineer reuses the oracle for divergence
  bisection (per-layer-kind I/O comparison pattern) — the reference standard Ch8's playbook reuses.
  - Primary evidence: `src/glm.h`, `src/glm.cpp`, `tests/tt_kernels.cpp`, `README.md`.

Questions covered: **Q5 (primary)**.

### Chapter 7 — Build Methodology and Determinism (`ch7_build_determinism`)

One-sentence: how `rules.py`/`make.py`, the sfpi pin, recipe flags, linker scripts, and per-image budgets
like the 88 KB ARC-scratch ceiling make every build instruction-level deterministic — the pillar that makes
regen's measured numbers citable and reproducible.

Files:
- `index.md` — chapter purpose; Q6 in scope; determinism stated as an engineering contract, not hygiene:
  records assume pinned instruction counts, replay windows assume stable layout — and the thesis that
  reproducible binaries are what make the notes corpus's numbers citable at all.
  - Primary evidence: `rules.py`, `make.py`, `README.md`.
- `01-toolchain-and-recipes.md` — the pinned toolchain: riscv-tt-elf-g++ sfpi (regen pin 7.76.0, GCC 15.1)
  [FACT, rules.py], march/mabi recipe flags, `make.py`/`rules.py` recipe structure, image assembly from
  fragments (including `tools/sfpu.py`'s checked-in `kernels_sfpu.inc` output); WHY the pin is load-bearing
  (instruction-count static_asserts are compiler-version contracts — Ch3's pins); the port's re-verification
  that the static_asserts pass under sfpi 7.86.0 with the xtttensixbh march extension as evidence the pins
  are auditable and movable under control [FACT, rules.py; cross-check regen_kernel_port.md M0].
  - Primary evidence: `rules.py`, `make.py`, `tools/sfpu.py`, `src/kernels.sfpu`.
- `02-image-layout-and-budgets.md` — device image layout from `src/brisc.ld`/`src/brisc_entry.S`: image ≤
  0x16000 = 88 KB (ARC scratch ceiling, linker assert) [FACT, src/brisc.ld], binary ~68 KB class, stack in
  the 8 KB LDM at 0xFFB02000 [FACT, src/brisc_entry.S]; the parallel ERISC image budget (`erisc.ld`); how
  the linker assert turns a budget into a BUILD-time gate (budgets fail the build, not the run); how budget
  pressure fed back into the methodology (records as data, replay windows — back-references to Ch2/Ch3).
  - Primary evidence: `src/brisc.ld`, `src/brisc_entry.S`, `src/erisc.ld`, `src/erisc_entry.S`.
- `03-determinism-chain.md` — the end-to-end composition argument: pinned toolchain + fixed flags +
  static_assert-pinned instruction counts (Ch3) + linker-asserted budgets + checked-in generated code +
  byte-identical release binaries (sha256 discipline [verify in README.md before writing]) ⇒ byte-stable
  images and reproducible measurements; what a contributor must never change casually (window lengths,
  record layouts, image flags) and how to re-derive a pin when a toolchain must move; the forward pointer
  to Ch8: which guarantees a tt-metal build must replicate to keep regen's evidence discipline meaningful.
  - Primary evidence: `rules.py`, `src/kernels.sfpu` (static_asserts), `README.md`, `docs/README.md`.

Questions covered: **Q6 (primary)**; static_assert pins back-referenced from Ch3.

### Chapter 8 — Porting to tt-metal: Stays, Comes, New (`ch8_tt_metal_port`)

One-sentence: the port analysis — what the tt-metal harness supplies (open, launch, alloc, evidence) vs what
the engine must own, regen's decisions classified as load-bearing vs incidental, the re-derivation method,
the measured gates, and the contributor's entry points; the guide's ONLY porting commentary.

Files:
- `index.md` — chapter purpose; Q7 in scope; the separation contract: Chapters 1–7 documented regen
  as-built; this chapter alone interprets; the stays/comes/new vocabulary defined; primary evidence is
  regen source + notes, with `regen_kernel_port.md` used ONLY as cross-check, cited
  `[cross-check, regen_kernel_port.md]` and never presented as regen-internal evidence.
  - Primary evidence: `doc/research/regen_kernel_port.md` (cross-check only); Chapters 1–7.
- `01-stays-comes-new.md` — the §2 split restated from regen's side: what STAYS as regen engine concepts
  (record interpreter, SFPU replay philosophy, streaming discipline, TTQ pipeline), what COMES from tt-metal
  (device open, kernel launch, memory alloc, watcher/tracy evidence machinery, locks), what is NEW (bridge
  layers, dispatch integration, trace/megakernel harness per the bring-up program's constraints); per item:
  which chapter of this guide explains the source mechanism it keeps or replaces.
  - Primary evidence: regen_kernel_port.md §2 (cross-check); Chapters 2–7 as primary grounding.
- `02-load-bearing-vs-incidental.md` — the decision-by-decision audit of Chapters 2–7's pillars:
  load-bearing (records-as-data for dispatch elimination; claims-ahead two-strips-in-flight for 89% DRAM;
  channel-local placement co-design; instruction-count determinism; device-originated transports;
  oracle-based verification) vs incidental to regen's host (exact BAR addresses and BAR0 window code,
  legacy-KMD/ioctl specifics, LoudBox topology constants, the specific 88 KB ceiling value, sfpi version
  number itself) — with the test "does the 6.05 ms/token roofline argument survive if tt-metal replaces
  this?" applied to each entry, each citing the chapter that established it and naming the tt-metal
  constraint forcing re-derivation. Consumes Ch3's engine-vs-llk table as an input.
  - Primary evidence: Chapters 2–7; `notes/glm-decode-roofline.md` (the roofline argument being tested).
- `03-rederivation-playbook.md` — HOW to re-derive a regen decision under tt-metal constraints: the method
  (identify the invariant the decision protects → find the tt-metal mechanism that protects it → re-measure
  with regen's own oracle-and-ledger style, tagged per the evidence rules), with two worked examples (e.g.
  dispatch elimination via a persistent megakernel; DRAM-busy budget restated under tt-metal's allocator),
  each pointing back to the chapter file that stated the invariant.
  - Primary evidence: Chapters 2–7 invariant statements; `src/glm.cpp` oracle method; `notes/glm-decode-ops.md`
    ledger style.
- `04-port-gates-and-re-measure.md` — the measured port gates as of the briefing (M0 static_asserts under
  sfpi 7.86.0/xtttensixbh; H1; M-ERISC) [cross-check, regen_kernel_port.md], what each validated about the
  methodology's transferability, mapped back to the chapter whose claim it checks; and the RE-MEASURE
  CHECKLIST: the numbers a port must measure on its own host and NEVER inherit from regen's registry —
  transport latencies (PCIe per socket topology; ERISC per fabric), DRAM window rate, the t/s ladder,
  smoke-gate results — restating the evidence-tag rules for port-side writing.
  - Primary evidence: regen_kernel_port.md (cross-check); briefing registry (for the list of what to
    re-measure).
- `05-contributor-checklist.md` — the actionable close-out: a task→chapter→primary-source map ("I want to
  add an op → Ch3 `04-op-bodies.md` + `src/ckernel_ops.h`; change strip layout → Ch4 `02` + `src/requant.cpp`
  and Ch6 `03`; debug a numerical divergence → Ch6 `05-cpu-oracle.md` + `tests/tt_kernels.cpp`; move the
  toolchain pin → Ch7 `01` + `rules.py`; touch transports → Ch5 + `src/erisc.cpp`/`notes/glm-ccl.md");
  the port traps (don't port incidental addresses; don't break instruction-count pins; respect the
  bring-up program's traced-execution and device-lock rules on the tt-metal side); the minimum reading set
  for a new contributor (Ch1, Ch2, Ch3, Ch4, Ch7 + this chapter); pointers back into Ch1's source map for
  the rare deep dives.
  - Primary evidence: Chapters 1–7 file structure; `src/` anchors per task row.

Questions covered: **Q7 (primary)**, argued from Chapters 1–7.

---

## 4. Conventions

### Terminology

Fixed glossary, defined at first use in `ch1/index.md`, bolded and linked on first use per chapter; regen's
own identifiers keep exact source spelling (`TtGlmStepRecord`, `_ADDR` symbols, `hot3-meas`, `plan0h`,
`SrcA bank test`).

- *record* — one `TtGlmStepRecord` step; *interpreter* — the persistent per-lead BRISC loop; *lead*/*helper*
  — team roles; *team job*; *op body*; *strip* — a 16-row BFP weight strip (geometry owned by Ch4, class
  policy by Ch6); *claim / claims-ahead*; *round* — a cross-chip collective phase; *ring*; *TTQ*; *requant
  class*; *hot expert*; *peer-BAR DMA*; *ERISC service*; *oracle* — the `glm.cpp` CPU reference.

**"Window" is ALWAYS qualified — three distinct senses, never bare:**
1. **SFPU replay window** (Ch3) — a pre-compiled, instruction-count-pinned SFPU sequence emitted by
   `tools/sfpu.py` into `kernels_sfpu.inc` and replayed by the interpreter.
2. **DRAM prefetch window** (Ch4) — a scheduled burst of DRAM reads batching strip traffic around
   compute/round phases (never "prefill window"; prefill-phase design is out of guide scope).
3. **PCIe BAR window** (Ch5/Ch6) — a host-mapped BAR0/peer-BAR address aperture onto device or peer
   memory.

### Notation

- Hex addresses as in source (`0xFFE40000`), citing the defining file on first appearance
  [FACT, src/fw.h]. Throughput in GB/s with the peak it is a fraction of; latencies in us; sizes in bytes
  unless suffixed; regen identifiers in backticks.
- All bandwidth/latency arithmetic shown as equations where it supports a claim (e.g. $447/500 = 89\%$),
  not asserted.

### Evidence tags (every factual/numbered claim carries exactly ONE)

- `[MEAS, notes/<file>.md]` / `[MEAS, README.md]` / `[MEAS, src/<file>:line]` — measured on
  silicon/bench. Registry numbers cited verbatim (never re-rounded, never invented). Any number NOT in the
  orchestrator registry must be located and verified in notes/source by the writer BEFORE writing, then
  cited with file (and line where useful). In this plan, `[verify …]` marks such items for the writer.
- `[FACT, src/<file>]` — structural facts (addresses, struct layouts, linker asserts, pins).
- `[EST, <file>]` — estimate recorded in the corpus, cited to it.
- `[INFER, src/<file>]` — derived from code reading, never presented as measured; used liberally and
  strictly in the low-notes-evidence files (Ch2 teams, Ch4 placement, Ch5 mechanics).
- **Hybrid tags (e.g. `[MEAS/INFER]`) are banned.** If a claim has a measured part and an inferred part,
  split it into two tagged claims.
- Port material: `[cross-check, regen_kernel_port.md]`, only in Ch8 or as clearly-marked cross-checks;
  never presented as regen-internal evidence.

### LaTeX

Inline `$...$`, display `$$...$$`; ```math blocks when `\texttt{}`-with-underscores or `\!` is needed;
never underscores inside `\text{}`.

### Navigation and formatting

- Every file: one H1, opens with a 2–3 sentence abstract + a scope line ("covers Question N, part X"); ends
  with a navigation footer `← prev file | ↑ chapter index | next file →` plus a link to the guide root;
  chapter indexes link all files and all eight chapter indexes.
- All cross-references are clickable relative links into the regen repo (read-only; repo-relative paths
  like `src/brisc.cpp`, `notes/glm-ccl.md`; `brisc.cpp` never cited without a symbol or section anchor when
  a claim depends on it). Tables for ≥4-item comparisons and for all measured-number lists (with a
  citation column or per-row tags). Per-file length cap ~500 lines. Long source excerpts are linked, not
  pasted.

### Non-duplication / ownership rules (deliberate)

- **Ch1 cites, never restates**: headline numbers appear once in Ch1 with citations; the owning chapter
  carries the full exposition (Ch1 `02`'s roofline decomposition is a pointer-style preview, owned by Ch4).
- Transport latencies ONLY in Ch5. DRAM throughput/window rates and the optimization ledger ONLY in Ch4.
  TTQ/weights numbers ONLY in Ch6. Toolchain/budget facts ONLY in Ch7. Port interpretation ONLY in Ch8.
- The oracle's ROLE is introduced in Ch3 `05` (as correctness consumer); its MECHANICS live once in Ch6
  `05`.
- Op MECHANISM only in Ch3 `04`; op PERFORMANCE only in Ch4 `04`.
- Record FORMAT only in Ch2 `01`; record PRODUCTION only in Ch6 `04`.
- Strip GEOMETRY as the DRAM reader contract owned by Ch4 `02`; requant-CLASS selection and hot-expert
  policy owned by Ch6 `02`; each links, neither restates.
- Ch2 `04` owns ring geometry and record-stream prefetch only; two-strips-in-flight, NoC-ID budget, and
  window rates are Ch4's (Ch2 defers by name).
- Batch/MTP numbers ONLY in Ch4 `05`; Ch5 and Ch6 cross-reference them.
- Ch1–7 may point to Ch8 ("see Ch8 §…") but must not argue port decisions.

---

## 5. Cross-chapter dependencies (one-way back-references only; no cycles)

- **Ch1 → all**: mental model, token walk, workload context, source map, measured envelope, glossary.
  Every later chapter assumes its vocabulary; Ch2–5 each open by locating their subsystem on Ch1's token
  walk.
- **Ch2 → Ch3, 4, 5, 6**: the interpreter loop is the substrate Ch3's MMIO writes execute on, the consumer
  of Ch4's rings, the initiator of Ch5's transports, and the target of Ch6's record builder. Ch2 `05`'s
  elimination argument sets up Ch8.
- **Ch3 → Ch4**: streaming must respect raw-MMIO's no-CB ownership of data movement (readers own all
  synchronization). Ch3 `06`'s engine-vs-llk table is a declared input to Ch8 `02`.
- **Ch2+Ch3 → Ch4**: claims-ahead generalizes Ch2's rings; op timing vocabulary from Ch3.
- **Ch4 → Ch6** (deliberate, declared forward reference, resolved on reading Ch6): near-channel reading
  requires channel-local placement, whose loader half lives in Ch6 `03`; Ch6 back-references Ch4 — the
  pairing is a split, not a cycle, and both files name their half.
- **Ch2+Ch4 → Ch5**: transports serve interpreter rounds and budget NoC IDs jointly with streaming
  readers; Ch5 cites, never restates, Ch4's throughput numbers.
- **Ch2+Ch3+Ch4+Ch1 → Ch6**: the record list only makes sense given Ch2's format, Ch3's op bodies, Ch4's
  placement demand, Ch1's workload digest.
- **Ch3 → Ch7** (and Ch2's structs feed image layout): determinism exists to protect Ch3's instruction
  pins; Ch7 references them, never vice versa beyond forward pointers.
- **Ch1–Ch7 → Ch8**: terminal chapter; every stays/comes/new and load-bearing entry cites the chapter that
  established its fact. No chapter depends on Ch8.

---

## 6. Question coverage map

| # | Question | Primary chapter | Secondary mentions (pointers only) |
|---| --- | --- | --- |
| 1 | Record-driven execution model (records, per-lead interpreter loop, DRAM read-ahead rings, lead/helper team jobs; eliminates phase boundaries, per-step dispatch, host) | **Ch2** (`ch2_record_driven_execution`, all files; `05-eliminating-the-runtime.md` explicitly) | Ch1 `01-mental-model.md`; Ch6 `04-record-builder-and-serving.md` (builder origin) |
| 2 | Raw-MMIO compute (cfg writes, instruction port 0xFFE40000, SFPU replay windows via tools/sfpu.py; no llk/CBs, TRISCs idle; static_asserts, instruction-count pins, SrcA bank probe) | **Ch3** (`ch3_raw_mmio_sfpu_replay`, all files) | Ch7 `01`/`03` (pins as determinism gates) |
| 3 | DRAM streaming discipline (strip claims-ahead, two strips in flight, near-channel placement, prefetch windows, NoC-ID budgeting; 9.3 → 6.05 ms/token) | **Ch4** (`ch4_dram_streaming_discipline`, all files) | Ch6 `03` (placement supply side); Ch1 `02`/`03` (byte inventory + walk) |
| 4 | Two cross-chip transports (PCIe peer-BAR DMA vs ERISC message service; when each; measured latencies per host) | **Ch5** (`ch5_cross_chip_transports`, all files) | Ch2 `03` (lead relays); Ch4 `05` (batching demand side, cross-referenced) |
| 5 | Host driver and weights pipeline (TTQ format, requant classes, channel-local placement, glm.cpp oracle) | **Ch6** (`ch6_host_driver_weights_oracle`, all files) | Ch2 `01` (what the builder emits); Ch3 `05` (oracle's correctness role) |
| 6 | Build methodology (sfpi pin, recipe flags, linker scripts, 88 KB ceiling; instruction-level determinism) | **Ch7** (`ch7_build_determinism`, all files) | Ch3 `03` (count pins); Ch3 `05` (smoke gate) |
| 7 | tt-metal port (harness-supplied vs engine-owned; stays/comes/new; load-bearing vs incidental) | **Ch8** (`ch8_tt_metal_port`, all files) | Argued from Ch1–Ch7; `regen_kernel_port.md` as cross-check only |

Every question has exactly one primary chapter; Ch1 is the shared foundation. All 7 questions covered; no
chapter duplicates another's owned content.

---

## Standing constraints honored

- 8 chapters maximum: exactly 8.
- Success bar: reader contributes to the tt-metal port without reading the 1.2 MB source — served by the
  Ch1 source map, per-file primary-evidence routing, Ch8 rederivation playbook + contributor checklist.
- Every measured number carries a `[MEAS, notes/<file>.md]`-style registry citation or is verified by the
  writer in the notes before writing (`[verify]` flags throughout; hybrid tags banned).
- regen source tree is read-only; the guide only links into it.
- Porting commentary confined to Ch8; regen methodology primary throughout.
- LaTeX and navigation-footer rules per the deep-planning constraints.
