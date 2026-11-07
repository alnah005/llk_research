# Batch and MTP Economics

The streaming discipline rewards reading each weight strip once and keeping DRAM busy; batching B *independent* users on one 8-chip group breaks exactly that — every MoE layer must read the **union** of their routed experts, so per-user bytes multiply while DRAM does not. Measured, batching 4 users on one group loses ~3.8× against one user per group, and draft-1 MTP (speculative decoding) is what recovers per-user throughput at batch: at B=8 greedy it lifts 38.7 to 53.1 t/s/user (+37%) at 0.892 draft acceptance. This file tells the two halves as one causal story and sets up the serving-shape decision a porter faces.

**Scope:** covers Question 3, part 5 — batch and MTP economics; all batch/MTP numbers live here. The transport-round costs of speculative decoding are [Ch5](../ch5_cross_chip_transports/index.md)'s (cross-referenced only); the batch image's build and firmware ceiling are out of scope here (glm-batch's own design sections); the per-user state/KDA-rollback machinery is recorded in the same notes but is not restated. Measured host context: Galaxy 8-chip groups unless a row says otherwise.

## Why batching breaks the discipline

At B=1, a token reads only 8 of 288 routed experts per MoE layer — 594.5 MB/token/chip of routed bytes on plan0h [MEAS, notes/glm-decode-roofline.md (line 45)] — and every other class is a fixed stream shared by nothing. Batch B rows in one step and the routed phase must read the **union** of the B users' expert sets:

| B | Expected union of routed experts per layer (imatrix model) | Routed MB/chip/step (plan0h) | Citation |
| --- | ---: | ---: | --- |
| 1 | 8.0 | 595 | [EST, notes/glm-batch.md (line 48)] |
| 8 | 58.1 uniform; 54.6 / 54.8 on real text mixes | 4150 | [EST, notes/glm-batch.md (line 51)] |

The union model is arithmetic from routing marginals — "Expert e is missed with probability prod_u (1 - p_u(e)), which is exact in expectation for independent users" [EST, notes/glm-batch.md (lines 39–40)] — and silicon route dumps confirm it: B=8 K=1 rows measured 54.6 experts per layer against 53.8 for independent draws [MEAS, notes/glm-batch.md (line 182)]. Identical prompts touch 8 experts at any B — only *correlated* rows keep the union small [FACT, notes/glm-batch.md (line 42)]. So at B=8:

$$\frac{4150}{595} \approx 6.97\times \text{ the routed bytes, for } 8\times \text{ the users} \quad \Longrightarrow \quad \text{per-user reuse} \approx 1 - \frac{4150}{8 \times 595} \approx 13\%$$

Expert strips amortize across users only where routes coincide; 8 independent users share barely 13% of routed bytes [INFER, notes/glm-batch.md (lines 48–58)]. The non-routed classes amortize perfectly, which is exactly why the union dominates.

## The measured penalty

On one Galaxy group (hot3-meas, batch image) [MEAS, notes/glm-glx-layouts.md (lines 259–266)]:

| B | Same-prompt step / t/s per user | Distinct-prompt step / t/s per user | Citation |
| --- | --- | --- | --- |
| 1 | 6.92 ms / 144.5 | — | [MEAS, notes/glm-glx-layouts.md (line 263)] |
| 4 | 19.13 ms / 52.3 | 24.82 ms / **40.3** | [MEAS, notes/glm-glx-layouts.md (line 265)] |
| 8 | 20.16 ms / 49.6 | — | [MEAS, notes/glm-glx-layouts.md (line 266)] |

Against one user per group — DP4, measured at 151 t/s/user (150.0–151.0 with all four groups running) [MEAS, notes/glm-glx-layouts.md (line 51; §7 line 290)] — batching 4 users on one group loses ~3.8×: "For 3-4 users, batching on one group loses 3.8x against one user per group. Even at its model it cannot win: 4 distinct users' union (30 experts) is 4.6 GB per chip per step, a 10.3 ms DRAM floor against DP4's 6.6 ms per token" [MEAS, notes/glm-glx-layouts.md (lines 270–272)]. The verdict line: "Batching only pays for aggregate throughput at B of 8 or more" [MEAS, notes/glm-glx-layouts.md (line 275)].

On the later f1 engine (Galaxy, hot3), plain B=8 measured 25.86 ms per step = **38.7 t/s/user** [MEAS, notes/glm-glx-mtp2.md (lines 122–126)]; a batched step costs "about 17.5 ms fixed plus about 0.3 ms per row, and about 2 ms per extra 15 experts of union. The fixed part is the batch engine's per-step cost (about 12 steps per layer at 4.5 us or more, plus gathers)" [EST, notes/glm-glx-layouts.md (line 268)]. (The LoudBox B=8 record, for host contrast, is 36.2 t/s/user with 289.8 total on hot3-meas and 41.8 / 334.2 on plan0h [MEAS, notes/post-rc/glm-rc1.md (lines 136–137)].)

## Why MTP recovers it

**What MTP is here:** draft-1 speculative decoding using the model's own MTP layer (layer 45) as the draft head, "run 3 from batch commands, no new firmware for the layer" [FACT, notes/glm-glx-mtp2.md (lines 21–22)]; the host loop is greedy only, "one post per step with every user's 1 or 2 rows and their MTP rows; accept when the first row's greedy token equals the draft" [FACT, notes/glm-glx-mtp2.md (lines 55–56)].

**The reuse batching lacks, a user's own draft rows have.** A user's consecutive tokens share experts: 4 consecutive tokens touch 22.9 experts against 29.6 for independent rows (ratio 0.77) [MEAS, notes/glm-batch.md (line 179)]; "Sixteen of one user's tokens touch 61% of what 16 independent rows touch, but across 8 users the saving is only 12-14%" [MEAS, notes/glm-batch.md (lines 189–190)]. So each user's second (draft) row re-covers mostly the same experts as its main row — the union grows far more slowly than a second independent user's rows would [INFER, notes/glm-batch.md (lines 179–190)] — while the *arithmetic* of the step (weights read once for all rows) still amortizes.

**The measured recovery** (m11; host ms, medians over steps where every user decodes; hot3-meas) [MEAS, notes/glm-glx-mtp2.md (lines 122–129)]:

| Shape | f1 ms | MTP ms | tokens/user step | f1 t/s/user | MTP t/s/user | Citation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B = 2 | 17.21 | 21.93 | 1.878 | 58.1 | 85.6 | [MEAS, notes/glm-glx-mtp2.md (line 124)] |
| B = 4 | 20.42 | 26.82 | 1.875 | 49.0 | 69.9 | [MEAS, notes/glm-glx-mtp2.md (line 125)] |
| B = 8 | 25.86 | 36.38 | 1.932 | **38.7** | **53.1** | [MEAS, notes/glm-glx-mtp2.md (line 126)] |
| B = 16 | 36.68 | 56.34 | 1.896 | 27.3 | 33.7 | [MEAS, notes/glm-glx-mtp2.md (line 127)] |

At B=8 greedy the lift is $53.1 / 38.7 = 1.372$, i.e. **+37%** per user, at 1.932 tokens per user step for +41% step time ("Step time +27/31/41/54% at B = 2/4/8/16 … most of it is each user's second main row" [MEAS, notes/glm-glx-mtp2.md (lines 131–132)]). The exchange rate is set by acceptance:

| Acceptance (draft 1) | Value | Citation |
| --- | --- | --- |
| Greedy, 8 templated prompts | **0.892** (reference 0.869) | [MEAS, notes/glm-glx-mtp2.md (line 119)] |
| Greedy, short raw prompts | 0.876–0.917 | [MEAS, notes/glm-glx-mtp2.md (line 119)] |
| Long context | 0.840 at 7 x 60K; 0.802 at 1 x 222K | [MEAS, notes/glm-glx-mtp2.md (line 120)] |
| Sampled, T = 1, top_p 0.95 | **0.876** (reference 0.866) | [MEAS, notes/glm-glx-mtp2.md (line 147)] |

The sampled path was later folded into one post (removing the second host round trip): sampled MTP at B=8 then measured 36.78/36.81 ms = **50.2/50.1 t/s/user** at T=1 [MEAS, notes/glm-glx-mtp3.md (line 69)].

The batch image's exactness contract keeps all of this comparable: batch rows bit-match single-user decode and each user's state is independent, so speculative decoding can run on top [FACT, notes/glm-batch.md (lines 8–9)]; MTP's greedy picks equal the plain engine's at B = 2/4/8/16, with forced-reject variants exact [MEAS, notes/glm-glx-mtp2.md (lines 115–117)].

## The transport angle — cross-reference only

Speculative decoding multiplies the cross-chip rounds a step pays (draft rows ride the same collectives as main rows, and the sampled path adds host sync points). Those round costs, and which transport carries which collective, are [Ch5 `03-rounds-and-transport-selection.md`](../ch5_cross_chip_transports/03-rounds-and-transport-selection.md)'s; the per-transport latency compendium is [Ch5 `04-latency-compendium.md`](../ch5_cross_chip_transports/04-latency-compendium.md)'s. No transport numbers are asserted here.

## The serving-shape decision

Assembling the owned numbers (all Galaxy, hot3-meas class):

| Serving shape | t/s per user | Aggregate t/s | Citation |
| --- | ---: | ---: | --- |
| One user per group (DP4), 4 groups | 151 (150.0–151.0) | ≈604 | [MEAS, notes/glm-glx-layouts.md (line 51; §7 line 290)] |
| B=4 distinct on one group | 40.3 | 161–209 | [MEAS, notes/glm-glx-layouts.md (line 265)] |
| B=8 plain (f1) | 38.7 | 309 | [MEAS, notes/glm-glx-mtp2.md (line 126)] |
| B=8 MTP greedy | 53.1 | 425 | [MEAS, notes/glm-glx-mtp2.md (line 126)] |
| B=8 MTP sampled (one post) | 50.2/50.1 | | [MEAS, notes/glm-glx-mtp3.md (line 69)] |

The DP4 aggregate cell (4 × 151) is arithmetic on the measured per-user rate, not a measured aggregate [INFER, notes/glm-glx-layouts.md (line 51)].

The discipline's own arithmetic decides the shape: for per-user rate at 3–4 users, one user per group wins outright (the ~3.8× loss is structural — the union floor, not engine overhead); for aggregate throughput, batch at B ≥ 8, with MTP as the per-user recovery (38.7 → 53.1 = +37%) [MEAS, notes/glm-glx-layouts.md (lines 270, 275); notes/glm-glx-mtp2.md (line 126)]. Even MTP at B=2 (85.6) stays well below DP4's 151 per user, so MTP does not flip the small-B decision; it makes large-B serving per-user tolerable [INFER, notes/glm-glx-mtp2.md (lines 122–129)]. A porter inherits none of these rates — the DRAM budget, the union behavior on real traffic, and the acceptance on real prompts are all re-measure items ([Ch8 `04-port-gates-and-re-measure.md`](../ch8_tt_metal_port/04-port-gates-and-re-measure.md) owns the checklist).

---

← [`04-optimization-ledger.md`](./04-optimization-ledger.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [Chapter 5 — Cross-Chip Transports](../ch5_cross_chip_transports/index.md)
