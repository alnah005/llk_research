# Roofline and Byte Budget

Decode moves 1.76–2.13 GB per chip per token — about 3.5× Qwen's 0.51 GB [EST, notes/glm-decode-roofline.md (line 24); the bytes are the 1763.8/2134 MB placed totals below] — so the floor is set by DRAM bytes, not compute. This file turns the per-class byte budget into per-chip floors, states the measured 9.3 ms decomposition that the streaming discipline had to attack, and derives the plan's form: keep DRAM busy for 79–95% of the token, because inside its streaming windows the engine already runs at 89% of peak.

**Scope:** covers Question 3, part 1 — the roofline and the byte budget. The per-class byte budget is [Ch1 `03-workload-context.md`](../ch1_engine_at_a_glance/03-workload-context.md)'s (cited here, never restated; the class totals below are cross-cited from it); the prefetch machinery that answers this roofline is [`02`](./02-strip-claims-and-reader-placement.md) and [`03`](./03-prefetch-windows-and-noc-id-budget.md)'s; the ledger that executed the plan is [`04`](./04-optimization-ledger.md)'s. The measured base is one LoudBox (8 BH chips), real weights, decode positions 31–63, with 1 us = 1350 cycles [MEAS, notes/glm-decode-roofline.md (lines 3–5)].

## From classes to per-chip bytes

[Ch1 `03-workload-context.md`](../ch1_engine_at_a_glance/03-workload-context.md) owns the config-derived per-class budget: 13.61 GB per token over all classes, with KDA projections at 36.5% and routed experts at 35.0% of decode bytes [EST, notes/glm-inventory.md (lines 120–121, 130)]. Placed on the 8-chip layout the total becomes 14.10 GB per token — the plan's 13.61 GB plus the replicated f_a/g_a (0.26 GB), strip padding, and the replicated latent reads (0.19 GB) [EST, notes/glm-inventory.md (lines 136–137)].

This file owns the *per-chip realization*: the loader's own strip shares, dumped at load [MEAS, notes/glm-decode-roofline.md (lines 3–6, 30)]:

| Phase (per token per chip) | MB/token/chip | Busiest / mean channel | Citation |
| --- | ---: | ---: | --- |
| KDA q/k/v/b/f_a/g_a (209 strips, 4 K parts, BFP8) x34 | 494.8 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 34)] |
| KDA f_b, g_b x34 | 9.5 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 35)] |
| KDA o_proj x34 | 151.5 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 36)] |
| MLA in (q_a, kv_a, wk, weights_proj; 18 strips on 18 helpers) x11 | 13.8 | 2.67 | [MEAS, notes/glm-decode-roofline.md (line 37)] |
| MLA indexer compressor gate (BF16, replicated) x11 | 11.5 | 2.00 | [MEAS, notes/glm-decode-roofline.md (line 38)] |
| MLA q_b + wq_b x11 | 46.0 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 39)] |
| MLA W_UK^T, W_UV (chip's 8 heads) x11 | 24.5 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 40)] |
| MLA o_proj x11 | 98.0 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 41)] |
| Dense gate/up, down x3 | 60.2 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 42)] |
| Router (BF16 rows) x42 | 16.5 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 43)] |
| Shared expert x42 | 140.4 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 44)] |
| Routed experts, 8 x (gate/up 1.18 + down 0.59 MB) BFP4 x42 | 594.5 | 1.00 (per expert) | [MEAS, notes/glm-decode-roofline.md (line 45)] |
| LM head | 84.3 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 46)] |
| mHC statics, MoE/MLA statics, step records | 18.2 | 1.00 | [MEAS, notes/glm-decode-roofline.md (line 47)] |
| **Total plan0h** | **1763.8** | | [MEAS, notes/glm-decode-roofline.md (line 48)] |
| hotE-meas adds (70% of routed reads at BFP8) | +370 | | [MEAS, notes/glm-decode-roofline.md (line 49)] |

The rows sum to the stated total: $\sum \text{rows} = 1763.8$ MB. The hottest weight file (`hotE-meas`; requant class and hot-expert policy are [Ch6 `02`](../ch6_host_driver_weights_oracle/02-ttq-container-and-requant.md)'s) adds +370 MB per token per chip on top of plan0h's 1763.8, i.e. 2134 MB/chip [MEAS, notes/glm-decode-roofline.md (lines 48–49)]. The production file `hot3-meas`'s own budget is separate: $1763.8 + (932 - 595) \approx 2101$ MB $\approx 2.10$ GB/chip [MEAS, notes/glm-batch.md (lines 30–31); README.md "Roofline"] — the 4.27/4.77 ms floors below are `hotE-meas`'s. Latent and index reads are under 0.1 MB at position < 64 (23 MB/token/chip at 2048) and KDA state, conv history and mHC streams are L1-resident — 0 DRAM [MEAS, notes/glm-decode-roofline.md (lines 51, 54)].

**Channel balance.** The busiest channel costs only 0.07 ms over the balanced floor (3.60 vs 3.53 ms): the skews are MLA `in` (18 strips on 18 helpers, 2.7× on channels 0–2) and the replicated BF16 gate (2×), together ~5 us per MLA layer [MEAS, notes/glm-decode-roofline.md (lines 52–53)]. So the placement is already near-balanced; imbalance is not the problem.

## The floors

$$\frac{1763.8\ \text{MB}}{500\ \text{GB/s}} = 3.53\ \text{ms} \qquad (284\ \text{t/s}), \qquad \frac{1763.8\ \text{MB}}{447\ \text{GB/s}} = 3.95\ \text{ms} \quad (253\ \text{t/s})$$

| Floor (per token, device) | ms | t/s | Citation |
| --- | ---: | ---: | --- |
| DRAM floor, plan0h (1764 MB/chip), 500 GB/s | 3.53 | 284 | [EST, notes/glm-decode-roofline.md (line 16)] |
| Same at the 447 GB/s the DRAM windows reach | 3.95 | 253 | [EST, notes/glm-decode-roofline.md (line 17)] |
| DRAM floor, hotE-meas (2134 MB/chip), 500 / 447 GB/s | 4.27 / 4.77 | 234 / 210 | [EST, notes/glm-decode-roofline.md (line 18)] |
| DRAM floor, down_exps BFP8 + hot gate/up (2052 MB/chip), 500 / 447 | 4.10 / 4.59 | 244 / 218 | [EST, notes/glm-decode-roofline.md (line 19)] |
| Sync floor of today's step structure | ~1.5 | | [EST, notes/glm-decode-roofline.md (line 20)] |
| Sync floor of the minimum round set | 0.41–0.47 | | [EST, notes/glm-decode-roofline.md (line 21)] |

(The floors are the note's own computed projections, tagged as estimates; the 447 GB/s rate inside them is the measured window rate below. The two sync-floor rows use the roofline's own per-round constants; the per-transport latency compendium is [Ch5 `04-latency-compendium.md`](../ch5_cross_chip_transports/04-latency-compendium.md)'s, not restated here.) Against these floors, the measured starting point was 9.30 ms device / 9.49 ms with host (105 t/s) on plan0h and 9.93 / 10.12 ms (99 t/s) on hotE-meas [MEAS, notes/glm-decode-roofline.md (lines 12–15)].

## Why the plan is a DRAM-busy fraction, not a FLOP roofline

- **Compute is not the binding resource.** Weight MAC per decode token is 16.74G [EST, notes/glm-inventory.md (line 133)]. Inside a DRAM-streaming window, a KDA helper's compute is "555 BFP8 tiles x 7.2 cycles = 3 us per helper" against a 40.5K-cycle matvec [MEAS, notes/glm-decode-roofline.md (line 113)], and a helper's share of a channel (~58 GB/s over 12 helpers) "delivers a BFP8 tile every ~70 cycles, so every shape is DRAM bound" [MEAS, notes/glm-bfp8.md (lines 207–208)]. The synchronized strip bench agrees: KDA q/k/v/b/f_a/g_a runs at 457 GB/s = 91% of 500 [MEAS, notes/glm-bfp8.md (line 216)] and the LM head at 461 GB/s = 92% [MEAS, notes/glm-bfp8.md (line 230)]; in the 9.3 ms profile the LM head matvec measured 182.7 us for 84.3 MB — at its floor [MEAS, notes/glm-decode-roofline.md (line 169)].
- **So the plan is stated as occupancy.** "GLM at 5 ms has to keep DRAM streaming for 79% (plan0h at 447 GB/s) to 95% (hotE at 447 GB/s) of the token" [MEAS, notes/glm-decode-roofline.md (lines 25–26)]; equivalently, the 5 ms target is 71% of plan0h's 500 GB/s floor and 85% of hotE's [EST, notes/glm-decode-roofline.md (line 23)]. A FLOP roofline would hide the real constraint: there is no FLOP term anywhere in the 9.3 ms.

## The sync chain per token

| Per token | Dense list | Index list (P >= 2051) | Citation |
| --- | ---: | ---: | --- |
| GLM_STEPS steps | 561 | 693 | [MEAS, notes/glm-decode-roofline.md (line 60)] |
| Step chip barriers | 561 | 693 | [MEAS, notes/glm-decode-roofline.md (line 61)] |
| Step mesh syncs (0-byte exchanges) | 194 | 216 | [MEAS, notes/glm-decode-roofline.md (line 62)] |
| Op-internal global rounds | 235 | 257 | [MEAS, notes/glm-decode-roofline.md (line 63)] |
| Cross-chip PCIe rounds in all | 429 | 473 | [MEAS, notes/glm-decode-roofline.md (line 64)] |

Measured per-instance costs: step mesh sync 4.4–5.4 us on chip 0 (194 per token = 0.96 ms; ~3.6 us is the round itself, ~1.4 us skew) [MEAS, notes/glm-decode-roofline.md (line 73)]; op-internal rounds 0.6–2.9 us (0.31 ms per token) [MEAS, notes/glm-decode-roofline.md (line 74)]; step bookkeeping 0.34 us record fetch and copy plus ~0.81 us acknowledged cycles write-back (~0.45 ms per token) [MEAS, notes/glm-decode-roofline.md (lines 77–78)]; chip barrier 0.42 us (0.23 ms per token) [MEAS, notes/glm-decode-roofline.md (line 79)]. These are the token-time sync terms the ledger owns; which transport carries which round is [Ch5 `03-rounds-and-transport-selection.md`](../ch5_cross_chip_transports/03-rounds-and-transport-selection.md)'s — counted here, never priced here.

## Where the 9.3 ms goes

Whole-token trace (per-step stamps on every chip plus firmware event traces on every core; instrumented device time 9.43 ms against 9.30, +1.5% [MEAS, notes/glm-decode-roofline.md (line 386)]). Each op's time splits into fetch, chip barrier, mesh sync, DRAM-streaming windows, op rounds, and lead serial work [MEAS, notes/glm-decode-roofline.md (lines 90–99)]:

| Component (count) | Measured ms | DRAM-streaming windows | Op rounds | Lead serial | DRAM floor @500 (plan0h / hotE) | Sync floor (rounds × 2 us) | Gap to max(floors) | Citation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MoE (42) | 2.559 | 1.696 | 0.107 | 0.479 | 1.503 / 2.243 | 0.168 | 1.06 | [MEAS, notes/glm-decode-roofline.md (line 92)] |
| KDA (34) | 2.099 | 1.197 | 0.074 | 0.634 | 1.312 | 0.068 | 0.79 | [MEAS, notes/glm-decode-roofline.md (line 93)] |
| mHC sites (90) | 1.785 | 0 | 0.087 | 1.184 | 0.030 | 0.180 | 1.61 | [MEAS, notes/glm-decode-roofline.md (line 94)] |
| MLA + indexer (11) | 1.673 | 0.714 | 0.035 | 0.634 | 0.388 | 0.044 | 1.29 | [MEAS, notes/glm-decode-roofline.md (line 95)] |
| mHC expand (90) | 0.503 | 0 | 0 | 0.445 | 0 | 0 | 0.50 | [MEAS, notes/glm-decode-roofline.md (line 96)] |
| Dense FFN (3) | 0.261 | 0.152 | 0.007 | 0.063 | 0.120 | 0.006 | 0.14 | [MEAS, notes/glm-decode-roofline.md (line 97)] |
| Head + argmax (1) | 0.245 | 0.181 | 0.003 | 0.043 | 0.169 | 0.004 | 0.08 | [MEAS, notes/glm-decode-roofline.md (line 98)] |
| **total** | **9.124** | 3.941 | 0.313 | 3.483 | 3.524 / 4.264 | 0.470 | | [MEAS, notes/glm-decode-roofline.md (line 99)] |

(The table omits the fetch, chip-barrier and mesh-sync columns of the note's original, which is at §3 lines 90–99; the sync-floor column is that original's, restored verbatim. "Gap to max(floors)" is measured − max(DRAM floor, sync floor) — for mHC sites $1.785 - 0.180 = 1.61$, the only row where the sync floor, not the DRAM floor, is the binding one.) Plus ~0.17 ms of cycle write-backs between stamps and the first step's GO skew: 9.30 ms device [MEAS, notes/glm-decode-roofline.md (line 101)]. Ranked by absolute gap: mHC sites 1.61, MLA 1.29, MoE 1.06, KDA 0.79, expand 0.50, dense 0.14, head 0.08 ms [MEAS, notes/glm-decode-roofline.md (line 104)].

## Diagnosis: idle DRAM, not slow DRAM

| Where the 9.3 ms goes | ms | share | Citation |
| --- | ---: | ---: | --- |
| DRAM streaming (447 GB/s inside windows) | 3.94 | 42% | [MEAS, notes/glm-decode-roofline.md (line 177)] |
| Lead serial work, few-core stages, team dispatch and team skew | 3.48 | 38% | [MEAS, notes/glm-decode-roofline.md (line 178)] |
| Syncs: mesh 0.96, bookkeeping ~0.45, op rounds 0.31, chip barriers 0.23 | ~1.95 | 21% | [MEAS, notes/glm-decode-roofline.md (line 179)] |

"Decode is bound by serial work per step, with DRAM idle, not by DRAM bandwidth or by the CCL" [MEAS, notes/glm-decode-roofline.md (line 173)]: "Not DRAM rate: inside the windows DRAM runs at 89% of peak and the busiest channel is within 2% of balance" [MEAS, notes/glm-decode-roofline.md (line 181)]. The largest serial items are the mHC site's Sinkhorn and norm on the leads (1.18 ms), KDA's post-projection chain (0.63 ms), MLA's many small lead-bound steps (0.63 ms), expand (0.45 ms), and MoE routing and team skew (0.48 ms) — "None of it overlaps DRAM, because no step reads the next step's weights" [MEAS, notes/glm-decode-roofline.md (lines 184–189)].

**Inside the windows** the discipline already works: 1764 MB moved in 3.94 ms is

$$\frac{1764\ \text{MB}}{3.94\ \text{ms}} = 447\ \text{GB/s} = \frac{447}{500} = 89\% \text{ of peak}$$

[MEAS, notes/glm-decode-roofline.md (line 108)], with helpers spending 80–85% of their window time waiting on DRAM chunks, ~10% in unpack/MVMUL [MEAS, notes/glm-decode-roofline.md (line 109)]. NoC issue cost is negligible (~30 cycles per command, a few hundred per strip) [MEAS, notes/glm-decode-roofline.md (lines 113–114)]. "**So the windows are DRAM bound; the problem is that DRAM is idle the other 58% of the token.**" [MEAS, notes/glm-decode-roofline.md (line 115)].

That sentence is the whole chapter's thesis in one line: the *rate* problem was already solved by strip streaming (this file's tables); the *occupancy* problem — filling the other 58% — is what claims-ahead generalization ([`02`](./02-strip-claims-and-reader-placement.md)), DRAM prefetch windows ([`03`](./03-prefetch-windows-and-noc-id-budget.md)), and the ledger ([`04`](./04-optimization-ledger.md)) attacked.

---

← [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`02-strip-claims-and-reader-placement.md`](./02-strip-claims-and-reader-placement.md)
