# Chapter 4 — DRAM Streaming Discipline

Decode on GLM-5.3-Flash is a DRAM problem (the byte arithmetic lives in [Ch1 `03-workload-context.md`](../ch1_engine_at_a_glance/03-workload-context.md)), and this chapter owns the discipline that solves it: **[strips](./02-strip-claims-and-reader-placement.md)** read **claims-ahead** with exactly two strips in flight per reader, readers pinned **near** the DRAM channel their bytes live in, **[DRAM prefetch windows](./03-prefetch-windows-and-noc-id-budget.md)** batching strip traffic around the serial phases, and **NoC-ID budgeting** keeping every outstanding-read class sound. The discipline holds DRAM at **447 GB/s per chip = 89% of the 500 GB/s peak inside windows** [MEAS, notes/glm-decode-roofline.md (line 108)] and took the engine from 9.3 to 6.05 ms/token [MEAS, notes/glm-decode-ops.md (lines 11, 18)].

**Scope:** covers **Question 3** (primary) — strip claims-ahead, two strips in flight, near-channel placement, prefetch windows, NoC-ID budgeting, and the 9.3 → 6.05 ms/token ledger, plus the batch/MTP economics the same discipline dictates. **Ownership fence:** ALL DRAM throughput numbers, the optimization ledger, and the batch/MTP economics numbers live in this chapter; other chapters cite, never restate. Strip *geometry* is Ch4 `02`'s (the reader contract); requant-*class* selection, hot-expert policy, and weight-file sizes are [Ch6](../ch6_host_driver_weights_oracle/index.md)'s. Primary sources: [notes/glm-decode-roofline.md](../../../regen-glm5.3-flash/notes/glm-decode-roofline.md), [notes/glm-decode-ops.md](../../../regen-glm5.3-flash/notes/glm-decode-ops.md), [notes/noc-ids.md](../../../regen-glm5.3-flash/notes/noc-ids.md), [notes/glm-bfp8.md](../../../regen-glm5.3-flash/notes/glm-bfp8.md), [src/brisc.cpp](../../../regen-glm5.3-flash/src/brisc.cpp), [src/tt_backend.cpp](../../../regen-glm5.3-flash/src/tt_backend.cpp).

## The headline numbers this chapter owns

| Number | Value | Citation |
| --- | --- | --- |
| Single-user decode, hot3-meas | 165.3 t/s = 6.05 ms/token | [MEAS, notes/post-rc/glm-rc1.md (line 136); repeats 165.29/165.30 (line 139)] |
| Single-user decode, plan0h | 178.9 t/s = 5.59 ms/token | [MEAS, notes/post-rc/glm-rc1.md (line 137)] |
| Inside DRAM-streaming windows | 447 GB/s per chip = 89% of the 500 GB/s peak | [MEAS, notes/glm-decode-roofline.md (line 108)] |
| The 5 ms plan requires DRAM busy | 79% (plan0h) to 95% (hotE) of the token, at 447 GB/s | [MEAS, notes/glm-decode-roofline.md (lines 25–26)] |
| Optimization ledger | base 9295/9925 us (plan0h/hotE) down to 8312/9021 us | [MEAS, notes/glm-decode-ops.md (lines 11, 18)] |
| Batch penalty, one group | B=8 plain ≈ 38.7 t/s/user; B=4 distinct-prompt 40.3 t/s/user; ~3.8× vs one user per group | [MEAS, notes/glm-glx-mtp2.md (line 126); notes/glm-glx-layouts.md (lines 265, 270)] |
| MTP recovery | 38.7 → 53.1 t/s/user = +37% at draft-1 acceptance 0.892 greedy (0.876 T=1 sampled; 0.802–0.84 long-context) | [MEAS, notes/glm-glx-mtp2.md (lines 119–120, 126, 147)] |

## Files in this chapter

| File | What it gives you |
| --- | --- |
| [`01-roofline-and-byte-budget.md`](./01-roofline-and-byte-budget.md) | The per-chip byte budget, the DRAM floors, the sync chain, the 9.3 ms decomposition, and why the plan is stated as a DRAM-busy fraction (79–95%) rather than a FLOP roofline. |
| [`02-strip-claims-and-reader-placement.md`](./02-strip-claims-and-reader-placement.md) | The strip as the unit of transfer (geometry = the reader contract), the claims-ahead protocol, why exactly two strips in flight, and near-channel reader placement — with the placement and claims measurements. |
| [`03-prefetch-windows-and-noc-id-budget.md`](./03-prefetch-windows-and-noc-id-budget.md) | DRAM prefetch windows (always qualified), the four learned scheduling rules, the DRAM-idle map, the measured moves (column semantics pinned), effective window rates, and the NoC-ID budget with its failure modes. |
| [`04-optimization-ledger.md`](./04-optimization-ledger.md) | The 9.3 → 6.05 ms/token ledger: the ranked plan, the change-by-change table (9295/9925 → 8312/9021 us), the phases of cuts, and where it landed (165.3 t/s = 6.05 ms; plan0h 178.9 t/s = 5.59 ms). |
| [`05-batch-and-mtp-economics.md`](./05-batch-and-mtp-economics.md) | Why the same discipline punishes batching (routed-expert unions destroy per-strip reuse; ~3.8× loss; B=8 plain ≈ 38.7 t/s/user, B=4 distinct 40.3) and how draft-1 MTP recovers it (38.7 → 53.1 t/s/user = +37% at 0.892 acceptance). |

## Reading paths

- **Straight through** (recommended): the files are ordered — budget → mechanics → scheduling → ledger → economics.
- **Port contributor** short path: [`01`](./01-roofline-and-byte-budget.md) → [`02`](./02-strip-claims-and-reader-placement.md) → [`04`](./04-optimization-ledger.md), then [Ch8](../ch8_tt_metal_port/index.md).
- **Serving-shape question** (batch vs MTP vs DP): [`05`](./05-batch-and-mtp-economics.md) alone, with [Ch5](../ch5_cross_chip_transports/index.md) as the follow-up.

Prior chapters' hooks: the byte budget is [Ch1 `03`](../ch1_engine_at_a_glance/03-workload-context.md)'s (cited, never restated here); the record ring that claims-ahead generalizes is [Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md)'s; op *mechanism* is [Ch3 `04-op-bodies.md`](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md)'s while op *performance* is this chapter's.

## Evidence conventions

Evidence tags per [Ch1 `05-measured-envelope-and-evidence.md`](../ch1_engine_at_a_glance/05-measured-envelope-and-evidence.md) — one tag per claim, no hybrids. Where the corpus states a design rule learned from a measured failure, the *rule* is quoted with the measured citation that taught it. **Table-reading key:** in every table of this chapter, `plan0h` / `hotE-meas` / `hotM-meas` / `hot3-meas` are weight files and `F` / `H` / `A` / `D` are boxes (LoudBox hosts) — a box letter is never a weight class and a weight name is never a box; cells the notes leave blank are em-dashes, never zeros. [`03`](./03-prefetch-windows-and-noc-id-budget.md) pins the canonical treatment. Line anchors cited in this chapter were verified against the read-only tree; where a note's table and a source comment disagree, the in-tree source is authoritative and both are cited.

**Terminology reminder (from [Ch1](../ch1_engine_at_a_glance/index.md) glossary):** "window" is always qualified. This chapter owns **DRAM prefetch windows**; SFPU replay windows are [Ch3 `03-sfpu-replay-windows.md`](../ch3_raw_mmio_sfpu_replay/03-sfpu-replay-windows.md)'s and PCIe BAR windows are [Ch5](../ch5_cross_chip_transports/index.md)'s. Prefill-phase design is out of guide scope and is fenced out by name wherever it borders this chapter.

---

**Next:** [`01-roofline-and-byte-budget.md`](./01-roofline-and-byte-budget.md)
