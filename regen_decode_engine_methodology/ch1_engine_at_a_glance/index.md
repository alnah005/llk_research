# Chapter 1 — The Engine at a Glance

regen's decode engine inverts the conventional host-dispatch stack: the host compiles model, weights, and topology into flat step-record lists once, and one persistent BRISC interpreter per Tensix core executes those records as data, token after token, with no per-step dispatch and no host in the token loop. This chapter is the shared foundation for the whole guide — the mental model, one token walked end-to-end, the workload arithmetic that makes decode DRAM-bound, a file-by-file map of the ~1.2 MB source tree, and the measured envelope the rest of the guide must account for.

**Scope:** shared foundation for Questions 1–7. No question is owned here; every later chapter opens from this one. Primary source: the read-only regen tree at `/home/alex/mpi-shfs/tenstorrent/regen-glm5.3-flash` — all `src/...` and `notes/...` citations in this guide are relative to that root.

**Link convention.** Links into the regen repo are written with repo-relative paths as their text and resolve relative to this guide's root (`llk_research/regen_decode_engine_methodology/`), e.g. [src/brisc.cpp](../../../regen-glm5.3-flash/src/brisc.cpp).

## Files in this chapter

| File | What it gives you |
| --- | --- |
| [`01-mental-model.md`](./01-mental-model.md) | The core inversion vs a conventional tt-metal stack, a side-by-side contrast table, the five design principles, and the headline payoff numbers. |
| [`02-anatomy-of-a-token.md`](./02-anatomy-of-a-token.md) | One decode token walked end-to-end at 50,000 ft: record fetch → op bodies → cross-chip rounds → token emit, and where the 6.05 ms goes at roofline level. |
| [`03-workload-context.md`](./03-workload-context.md) | The GLM-5.3-Flash layer map, strip packing, the per-token byte budget, and the arithmetic showing decode is DRAM-bandwidth-bound. |
| [`04-source-map.md`](./04-source-map.md) | The file-by-file map of the source tree with verified line counts, anchor symbols, the notes-corpus map, and what you can safely never read. |
| [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md) | The measured envelope stated up front as the numbers the guide must account for, plus the citation discipline with worked examples. |

## Reading paths

- **Port contributor** (Ch8-bound): [`01-mental-model.md`](./01-mental-model.md) → [`02-anatomy-of-a-token.md`](./02-anatomy-of-a-token.md) → [`04-source-map.md`](./04-source-map.md) → [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md) → [Ch2](../ch2_record_driven_execution/index.md) → [Ch3](../ch3_raw_mmio_sfpu_replay/index.md) → [Ch7](../ch7_build_determinism/index.md) → [Ch8](../ch8_tt_metal_port/index.md).
- **Performance engineer**: [`01-mental-model.md`](./01-mental-model.md) → [`02-anatomy-of-a-token.md`](./02-anatomy-of-a-token.md) → [`03-workload-context.md`](./03-workload-context.md) → [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md) → [Ch4](../ch4_dram_streaming_discipline/index.md) → [Ch5](../ch5_cross_chip_transports/index.md). Use [`04-source-map.md`](./04-source-map.md) to find the note that owns any number you want to re-measure.
- **New agent**: read this chapter in order, then [Ch2](../ch2_record_driven_execution/index.md) straight through; the glossary below covers every fixed term later chapters use without re-defining.

## Glossary

Fixed terms, defined once here, bolded and linked on first use per chapter. regen's own identifiers keep exact source spelling (`TtGlmStepRecord`, `_ADDR` symbols, `hot3-meas`, `plan0h`, `SrcA bank test`).

| Term | Meaning | Defined/owned by |
| --- | --- | --- |
| **record** | One `TtGlmStepRecord` step: a 256-byte flat command word-set (command id, sync class, operand args, prefetch descriptor); the data-encoded unit of device work. Format owned by Ch2. | [Ch2 `01-step-records.md`](../ch2_record_driven_execution/01-step-records.md) |
| **interpreter** | The persistent per-lead BRISC loop that fetches, decodes, and runs records; the single resident program per core. | [Ch2 `02-interpreter-loop.md`](../ch2_record_driven_execution/02-interpreter-loop.md) |
| **lead / helper** | The two core roles of a team: the lead runs the step and the rounds; helpers take posted jobs (streaming, matvec shares). | [Ch2 `03-lead-helper-teams.md`](../ch2_record_driven_execution/03-lead-helper-teams.md) |
| **team job** | A posted unit of helper work inside a step (e.g. the resident-prefetch job the interpreter posts from a record's prefetch field). | [Ch2 `03-lead-helper-teams.md`](../ch2_record_driven_execution/03-lead-helper-teams.md) |
| **op body** | The BRISC code implementing one record command: matvec, norms, router, SwiGLU, mHC, KDA, MLA, rounds. | [Ch3 `04-op-bodies.md`](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md) |
| **strip** | A 16-row BFP weight strip, the unit of DRAM transfer. Geometry (reader contract) owned by Ch4; requant-class policy owned by Ch6. | [Ch4 `02-strip-claims-and-reader-placement.md`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md) |
| **claim / claims-ahead** | A reader's reservation of upcoming strips; the pacing mechanism that keeps strips in flight while compute runs. | [Ch4 `02-strip-claims-and-reader-placement.md`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md) |
| **round** | A cross-chip collective phase inside the token walk (router logits, MoE combine, barriers, all-gathers). | [Ch5 `03-rounds-and-transport-selection.md`](../ch5_cross_chip_transports/03-rounds-and-transport-selection.md) |
| **ring** | A circular read-ahead buffer. Two distinct senses, never conflated: the DRAM **record ring** feeding the interpreter its own records, and the ERISC **send/receive rings** of the message service. | [Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md) + [Ch5 `02-erisc-message-service.md`](../ch5_cross_chip_transports/02-erisc-message-service.md) |
| **TTQ** | regen's weight container format produced by `src/requant.cpp`. | [Ch6 `02-ttq-container-and-requant.md`](../ch6_host_driver_weights_oracle/02-ttq-container-and-requant.md) |
| **requant class** | The per-tensor-group quantization choice: BFP4 strips (routed experts), BFP8 strips, BF16 strips (router, indexer compressor gate), source precision. | [Ch6 `02-ttq-container-and-requant.md`](../ch6_host_driver_weights_oracle/02-ttq-container-and-requant.md) |
| **hot expert** | One of the routed experts stored at BFP8 instead of BFP4 in the `hot3-meas` weight file, chosen by calibration (the experts BFP4 damages most). | [Ch6 `02-ttq-container-and-requant.md`](../ch6_host_driver_weights_oracle/02-ttq-container-and-requant.md) |
| **peer-BAR DMA** | Device-originated PCIe writes into a peer chip's BAR aperture; one of the two cross-chip transports. | [Ch5 `01-pcie-peer-bar-dma.md`](../ch5_cross_chip_transports/01-pcie-peer-bar-dma.md) |
| **ERISC service** | The Ethernet-core message service: `TtEthMsg` rings, credits, binomial trees, restart-correct counters. | [Ch5 `02-erisc-message-service.md`](../ch5_cross_chip_transports/02-erisc-message-service.md) |
| **oracle** | The `glm.cpp` CPU reference model whose output device results are checked against. | [Ch6 `05-cpu-oracle.md`](../ch6_host_driver_weights_oracle/05-cpu-oracle.md) |
| `hot3-meas` / `plan0h` | The two production weight files: 245.7 GB (hot experts at BFP8) / 180.9 GB (all-BFP4 experts). | [Ch6 `02-ttq-container-and-requant.md`](../ch6_host_driver_weights_oracle/02-ttq-container-and-requant.md) |
| LoudBox / Galaxy | The 8-chip single-host serving configuration / the 32-chip cabled-Ethernet host. Transport latencies are host-specific and kept distinct throughout. | [Ch5 `03-rounds-and-transport-selection.md`](../ch5_cross_chip_transports/03-rounds-and-transport-selection.md) |

**"Window" is ALWAYS qualified — three distinct senses, never bare:**

| Term | Sense | Owned by |
| --- | --- | --- |
| **SFPU replay window** | A pre-compiled, instruction-count-pinned SFPU sequence emitted by `tools/sfpu.py` into `kernels_sfpu.inc` and replayed by the interpreter. | [Ch3 `03-sfpu-replay-windows.md`](../ch3_raw_mmio_sfpu_replay/03-sfpu-replay-windows.md) |
| **DRAM prefetch window** | A scheduled burst of DRAM reads batching strip traffic around compute/round phases. (Prefill-phase design is out of guide scope; "prefill window" is never used.) | [Ch4 `03-prefetch-windows-and-noc-id-budget.md`](../ch4_dram_streaming_discipline/03-prefetch-windows-and-noc-id-budget.md) |
| **PCIe BAR window** | A host-mapped BAR0/peer-BAR address aperture onto device or peer memory. | [Ch5 `01-pcie-peer-bar-dma.md`](../ch5_cross_chip_transports/01-pcie-peer-bar-dma.md) |

## Forward map — where the questions land

| # | Question | Chapter |
| --- | --- | --- |
| 1 | Record-driven execution model (records, interpreter loop, read-ahead rings, lead/helper teams) | [Ch2 — The Record-Driven Execution Model](../ch2_record_driven_execution/index.md) |
| 2 | Raw-MMIO compute and SFPU replay (no llk, no CBs, TRISCs idle) | [Ch3 — Raw-MMIO Compute and SFPU Replay](../ch3_raw_mmio_sfpu_replay/index.md) |
| 3 | DRAM streaming discipline (strip claims-ahead, prefetch windows, NoC-ID budgeting; 9.3 → 6.05 ms) | [Ch4 — DRAM Streaming Discipline](../ch4_dram_streaming_discipline/index.md) |
| 4 | The two cross-chip transports and their measured latencies | [Ch5 — Cross-Chip Transports](../ch5_cross_chip_transports/index.md) |
| 5 | Host driver, weights pipeline (TTQ, requant, placement), CPU oracle | [Ch6 — Host Driver, Weights Pipeline, and the CPU Oracle](../ch6_host_driver_weights_oracle/index.md) |
| 6 | Build methodology and determinism (sfpi pin, budgets, byte-stable images) | [Ch7 — Build Methodology and Determinism](../ch7_build_determinism/index.md) |
| 7 | The tt-metal port: stays, comes, new | [Ch8 — Porting to tt-metal](../ch8_tt_metal_port/index.md) |

## Evidence conventions

Every claim in this guide carries exactly one evidence tag — `[MEAS, ...]`, `[FACT, ...]`, `[EST, ...]`, or `[INFER, ...]` — and measured numbers are cited verbatim from the verified registry or the notes. The rules, the envelope table, and the worked examples live in [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md).

---

**Next:** [`01-mental-model.md`](./01-mental-model.md)
