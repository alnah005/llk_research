# Chapter 5 — Cross-Chip Transports

A multi-chip decode token is stitched together by **cross-chip rounds** — every layer kind needs the 8 chips of a group to exchange slices mid-token — and regen carries every one of them device-originated, with no host CPU in the path. This chapter owns the two transports that carry those rounds — **[peer-BAR DMA](./01-pcie-peer-bar-dma.md)** (device-originated PCIe writes into peer **PCIe BAR windows**, relayed by the chip roots and, for gathers, by the leads) and the **[ERISC service](./02-erisc-message-service.md)** (the Ethernet-core message service: `TtEthMsg` rings, credits, binomial trees, restart-correct counters) — plus the rule that decides which collective rides which transport on which host, and the compendium of every measured transport latency.

**Scope:** covers Question 4 (primary) — the two cross-chip transports, when each is used, and their measured latencies on the different hosts. On [Ch1's token walk](../ch1_engine_at_a_glance/02-anatomy-of-a-token.md) this chapter owns stage 4 — "cross-chip rounds, throughout". Primary sources: the read-only regen tree at `/home/alex/mpi-shfs/tenstorrent/regen-glm5.3-flash` — all `src/...` and `notes/...` citations are relative to that root; links resolve from the guide root per the [Ch1 link convention](../ch1_engine_at_a_glance/index.md). Every line anchor in this chapter was re-verified in-tree before writing.

## What a decode round needs, per layer kind

One **round** is an all-gather of equal per-chip slices: each chip's **[lead](../ch2_record_driven_execution/03-lead-helper-teams.md)** column computes a slice, every chip ends with all 8 slices in place, then a release flag frees the buffers ([`03-rounds-and-transport-selection.md`](./03-rounds-and-transport-selection.md) walks the machinery). Who must talk to whom is fixed by the model's tensor-parallel split; the byte sizes come straight from the record's descriptors:

| Round | Who contributes / who takes | Slice per chip | Citation |
| --- | --- | ---: | --- |
| mHC site record rounds (×90/token — one per site: 45 FFN-site + 45 attention-site) | every chip's root; all leads take (the sites need the full 4 streams) | `TT_GLM_HC_RECORD` = 4 × 512 × 2 + 128 = 4224 B | [FACT, src/constants.h:2617-2621] |
| KDA heads (×34/token) | every chip; leads **and** helpers take | `TT_KDA_HEADS` × 256 = 8 × 256 = 2048 B | [FACT, src/brisc.cpp:17715; src/constants.h:3275] |
| MoE Ra — router logits (×42/token) | every chip (each holds 288/8 = 36 expert logits); leads take | `TT_GLM_MOE_RA_BYTES` = 36 × 4 = 144 B | [FACT, src/constants.h:2790-2791] |
| MoE Rb — shared + routed hidden (×42/token) | every chip; helpers take | `TT_GLM_MOE_SLOTS` × `TT_GLM_MOE_HIDDEN` × 2 = 9 × 256 × 2 = 4608 B = `TT_GLM_ROUND_MAX_SLICE` | [FACT, src/constants.h:2780-2781, 2983-2984] |
| `GLM_ALLGATHER` (×27/token, dense list) | every chip's first lead; varies | up to `TT_GLM_ROUND_MAX_SLICE` = 4608 B | [FACT, src/constants.h:2983-2984; src/brisc.cpp:17475] |
| Index gather (index list, per MLA layer) | every chip; leads take | `TT_GLM_INDEX_RECORD_BYTES` = 16 + 512 × 6 = 3088 B | [FACT, src/constants.h:3359; src/brisc.cpp:17120] |

Per-token round *counts* — 235 dense / 257 index op-internal global rounds, 561/693 step chip barriers, 429/473 cross-chip PCIe rounds in all — are [MEAS, notes/glm-decode-roofline.md §2 (lines 58-70)]; the walk-level exposition of that census is [Ch1 `02-anatomy-of-a-token.md`](../ch1_engine_at_a_glance/02-anatomy-of-a-token.md)'s, cited not restated. The chip-internal `allgather_barrier` that closes most steps is a NoC exchange among the 8 leads of ONE chip — not a cross-chip transport; it is [Ch2 `03-lead-helper-teams.md`](../ch2_record_driven_execution/03-lead-helper-teams.md)'s. The old step-ending cross-chip mesh sync is gone: the round-count parity scheme replaced it ([`03`](./03-rounds-and-transport-selection.md)) [FACT, notes/glm-ccl.md (lines 149-158)].

The largest round a GLM step can ask for is 4608 B per chip [FACT, src/constants.h:2983-2984] — the entire decode round population is small and latency-critical, and that fact drives most of the selection rule. The bulk flows (token-row gathers, prefill gathers) are a different size class and pick their transport by a measured crossover ([`03`](./03-rounds-and-transport-selection.md)).

## The ownership fence this chapter keeps

- **ALL transport latency and transport-bandwidth numbers live here.** PCIe peer-write, all-to-all, barrier, `glm_round` and ERISC round latencies, per-link and per-chip transport bandwidths, and the selection economics appear once, in this chapter ([`04-latency-compendium.md`](./04-latency-compendium.md) is the single table). Other chapters cite, never restate.
- The fence cuts the other way too: DRAM throughput and prefetch-window rates, the 9.3 → 6.05 ms/token optimization ledger, and the batch/MTP economics numbers are [Ch4 — DRAM Streaming Discipline](../ch4_dram_streaming_discipline/index.md)'s ([`04`](../ch4_dram_streaming_discipline/04-optimization-ledger.md), [`05`](../ch4_dram_streaming_discipline/05-batch-and-mtp-economics.md)); the demand-side coupling (batching and speculative decoding multiply rounds) is cross-referenced in [`03`](./03-rounds-and-transport-selection.md) and owned by Ch4 `05`.
- The **lead**/**helper** team mechanics and the "root relays rounds" role are [Ch2 `03-lead-helper-teams.md`](../ch2_record_driven_execution/03-lead-helper-teams.md)'s; this chapter starts where that file stops — *what the relay rides*.
- Record **format** is [Ch2 `01`](../ch2_record_driven_execution/01-step-records.md)'s; the host builder that constructs rounds is [Ch6 `04`](../ch6_host_driver_weights_oracle/04-record-builder-and-serving.md)'s; the host open/load path around the load-time facts cited here is [Ch6 `01`](../ch6_host_driver_weights_oracle/01-backend-open-and-load.md)'s.
- The ERISC image budget as a *build gate* is [Ch7 `02-image-layout-and-budgets.md`](../ch7_build_determinism/02-image-layout-and-budgets.md)'s; the budget value is stated here only as the message service's footprint fact.
- What a port must **re-measure** is [Ch8 `04-port-gates-and-re-measure.md`](../ch8_tt_metal_port/04-port-gates-and-re-measure.md)'s checklist; the compendium points at it and lists nothing on its behalf.

Terminology reminder (from the [Ch1 glossary](../ch1_engine_at_a_glance/index.md)): "window" is always qualified. This chapter owns **PCIe BAR windows** (host-mapped BAR0/peer-BAR address apertures onto device or peer memory); SFPU replay windows are [Ch3 `03`](../ch3_raw_mmio_sfpu_replay/03-sfpu-replay-windows.md)'s and DRAM prefetch windows are [Ch4 `03`](../ch4_dram_streaming_discipline/03-prefetch-windows-and-noc-id-budget.md)'s. "Ring" also splits: the DRAM record ring is [Ch2 `04`](../ch2_record_driven_execution/04-read-ahead-rings.md)'s; the ERISC send/receive rings are [`02`](./02-erisc-message-service.md)'s — rings, never windows.

## Files in this chapter

Mechanism files first, decision rule after, compendium last — the order the guide plan pins (you see both transports before the selection rule that trades them):

| File | What it gives you |
| --- | --- |
| [`01-pcie-peer-bar-dma.md`](./01-pcie-peer-bar-dma.md) | Device-originated PCIe peer-BAR DMA: the BAR0 window layout, host and device setup, the root/lead relay pattern, peer windows in the L1/DRAM map, why it beats host-relayed I/O, and the LoudBox latency anchors (the registry is [`04`](./04-latency-compendium.md)'s). |
| [`02-erisc-message-service.md`](./02-erisc-message-service.md) | The ERISC message service from `src/erisc.cpp` + `erisc.ld` + `erisc_entry.S`: `TtEthMsg` format, send/receive rings, credits, binomial trees, restart-correct counters, the interpreter handoff, and the reliability model. |
| [`03-rounds-and-transport-selection.md`](./03-rounds-and-transport-selection.md) | How decode rounds use each transport, the in-situ round costs per host, the selection rule — knob, thresholds, `topology.*` discovery, LoudBox vs Galaxy classes — and the gather crossover. |
| [`04-latency-compendium.md`](./04-latency-compendium.md) | ONE table of every measured transport number with its host context and per-row `[MEAS]` citations, plus the pointer to the port's re-measure checklist. |

**Reading order:** [`01`](./01-pcie-peer-bar-dma.md) → [`02`](./02-erisc-message-service.md) → [`03`](./03-rounds-and-transport-selection.md) → [`04`](./04-latency-compendium.md). A reader who only wants a number goes straight to [`04`](./04-latency-compendium.md) and follows its citations back out.

## Evidence conventions

Evidence tags per [Ch1 `05-measured-envelope-and-evidence.md`](../ch1_engine_at_a_glance/05-measured-envelope-and-evidence.md) — exactly one tag per claim, no hybrids. The transport mechanics are lightly covered by the notes, so mechanism claims lean on strict `[FACT, src/...]` for what the code pins and `[INFER, src/...]` for behavior derived from reading it; every measured number is `[MEAS, notes/<file>.md (line)]` with the line re-verified in-tree.

**Host caution, stated once and kept throughout:** transport latencies are host-specific — the LoudBox numbers (boxes like a10u28: 8 × p150b, each on its own Gen5 **x16** root port, chips 0-3 on NUMA 0 and 4-7 on NUMA 1, 2 × EPYC 9124 [MEAS, notes/glm-ccl.md (lines 10-11)]) and the Galaxy numbers (bh-glx-120-b08u02 / c01u20: 32 chips in cube groups behind **x1** PCIe links [MEAS, notes/glm-glx-ethdec.md (lines 3-4)]) are distinct classes, never averaged or mixed. Every table below carries a host column.

---

← [guide root](../index.md)

**Next:** [`01-pcie-peer-bar-dma.md`](./01-pcie-peer-bar-dma.md)
