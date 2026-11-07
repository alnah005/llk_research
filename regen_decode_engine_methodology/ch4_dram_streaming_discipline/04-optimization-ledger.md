# The Optimization Ledger

The 9.3 → 6.05 ms/token story is a ledger: every change bit-exact ("generated text is identical in every run" [MEAS, notes/glm-decode-ops.md (lines 1–3)]), measured as whole-token device-time medians over positions 31–63 on interleaved rounds, and accepted only on whole-token time [MEAS, notes/glm-decode-roofline.md (lines 323–325)]. This file owns the ledger: the ranked plan, the change-by-change table from 9295/9925 us (plan0h/hotE) down to 8312/9021, the phases of cuts that followed, and where it landed — 165.3 t/s = 6.05 ms on hot3-meas and 178.9 t/s = 5.59 ms on plan0h.

**Scope:** covers Question 3, part 4 — the ledger as the worked example of how the streaming discipline was won. Op *mechanism* is [Ch3 `04-op-bodies.md`](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md)'s (compute entries point there); the prefetch-window machinery many entries use is [`03`](./03-prefetch-windows-and-noc-id-budget.md)'s; claims and placement are [`02`](./02-strip-claims-and-reader-placement.md)'s; the roofline that planned it is [`01`](./01-roofline-and-byte-budget.md)'s. **Column discipline throughout:** every table's columns pin their weight file *and* their box (F/H/A/D as the notes label them); deltas compose per column and are never mixed across columns — the measured-moves table in [`03`](./03-prefetch-windows-and-noc-id-budget.md) is the canonical example. **Reading rule:** the phases below are overlapping slices on different builds, boxes and bases — not one linear A/B chain; no cross-phase additivity is implied.

## Method

- Bit-exact changes, two interleaved rounds against the previous build on the same box, medians over positions 31–63 of `chat-france -n 48` [MEAS, notes/glm-decode-ops.md (lines 1–3)].
- "Rank and accept moves on whole-token device time only; the profile tool refuses a file whose step sums miss it by more than 10 us" [MEAS, notes/glm-decode-roofline.md (lines 323–325)].
- Whole-token acceptance exists because per-step accounting missed 45 us/token on `glm-mla7r` and 210 on `glm-mla6`, and mis-ranked move 1 at −37 [MEAS, notes/glm-decode-roofline.md (lines 323–324)].

## The ranked plan (roofline §5)

Savings against 9.30 ms device (plan0h), each estimated from the measured segments of [`01`](./01-roofline-and-byte-budget.md)'s decomposition, roughly additive [EST, notes/glm-decode-roofline.md (lines 193–197)]:

| # | Item | Saving ms | Class | Citation |
| --- | --- | ---: | --- | --- |
| 1 | Prefetch across steps (helpers read the next op's static weights while the lead runs the site/expand/routing/rounds) | 0.8 | streaming | [EST, notes/glm-decode-roofline.md (line 198)] |
| 2 | Remove the step mesh syncs (one global round counter for RX parity) | 0.75 | sync | [EST, notes/glm-decode-roofline.md (line 199)] |
| 3 | mHC site: comb (Sinkhorn) off the critical path; drop the no-op team post | 0.6 | sync/compute | [EST, notes/glm-decode-roofline.md (line 200)] |
| 4 | Fuse expand into the next site, or cut it to ~1 us of SFPU | 0.36 | compute | [EST, notes/glm-decode-roofline.md (line 201)] |
| 5 | MLA: zero query tiles at load; multicast heads/latents; flash fixed cost 15 → ~5 us; fuse small steps | 0.45 | compute/sync | [EST, notes/glm-decode-roofline.md (line 202)] |
| 6 | Step bookkeeping: prefetch record i+1 during op i; cycles write-back only when profiling | 0.35 | sync | [EST, notes/glm-decode-roofline.md (line 203)] |
| 7 | MoE: routing 5 → ~2 us; claim-based shares against team skew; combine 2 → 1 us | 0.33 | streaming/compute | [EST, notes/glm-decode-roofline.md (line 204)] |
| 8 | KDA: fold f_b/g_b into the projection pass; multicast input writes | 0.29 | compute/streaming | [EST, notes/glm-decode-roofline.md (line 205)] |
| 9 | Head argmax 30 → ~3 us on helpers | 0.03 | compute | [EST, notes/glm-decode-roofline.md (line 206)] |
| | **Sum** | **~4.0** | | [EST, notes/glm-decode-roofline.md (line 207)] |

(The Class column is this guide's reading of the note's items, using the spec's four-class vocabulary — streaming / placement / compute / sync — plus their compounds [INFER, notes/glm-decode-roofline.md (lines 198–206)].) Projected: $9.3 - 4.0 \approx 5.3$ ms device with plan0h, ~5.5 ms with host, ~180 t/s [EST, notes/glm-decode-roofline.md (line 209)]. The routed experts sit on the chain — "they cannot be read before routing" — so the hot-expert delta stays on the critical path; measured: hotE-meas costs +0.63 ms (MoE 61.0 → 75.7 us per layer; 370 MB in 0.62 ms), and part of the extra reads fill today's first-strip and team-skew slack, "so expect +0.6-0.8 ms after them: ~5.9-6.1 ms device with hotE" [MEAS, notes/glm-decode-roofline.md (lines 210–213)]. Past the plan, 5 ms with hotE "needs DRAM busy for ~95% of the token at today's window rate, i.e. a weight-streaming schedule where every serial piece runs under the next phase's reads" [EST, notes/glm-decode-roofline.md (lines 216–217)].

## Phase 1 — the Changes table

Base `a7dda80`, device us per token, each row cumulative over the one above [MEAS, notes/glm-decode-ops.md (lines 9–21)]:

| Change | plan0h | hotE-meas | Class | Citation |
| --- | ---: | ---: | --- | --- |
| Base a7dda80 | 9295 / 9296 | 9925 / 9916 | — | [MEAS, notes/glm-decode-ops.md (line 11)] |
| Finish post/comb under the block's helper wait (Sinkhorn off the critical path) | 8838 / 8843 | 9471 / 9472 | sync | [MEAS, notes/glm-decode-ops.md (line 12)] |
| Expand from LReg coefficients and fixed Dst offsets | 8591 / 8586 | 9221 / 9221 | compute | [MEAS, notes/glm-decode-ops.md (line 13)] |
| Fuse the expand into the next site | 8494 / 8484 | 9125 / 9129 | compute | [MEAS, notes/glm-decode-ops.md (line 14)] |
| MoE shared down into the prefetch slot while the leads route | 8397 / 8397 | 9044 / 9040 | streaming | [MEAS, notes/glm-decode-ops.md (line 15)] |
| Head argmax on the helpers | 8369 / 8374 | 9020 / 9022 | compute | [MEAS, notes/glm-decode-ops.md (line 16)] |
| MoE combine as one replayed body | 8342 / 8348 | 8980 / 8980 | compute (replay: [Ch3 `03`](../ch3_raw_mmio_sfpu_replay/03-sfpu-replay-windows.md)) | [MEAS, notes/glm-decode-ops.md (line 17)] |
| Replay the expand's block body | 8312 / 8326 | 9021 / 9050 | compute (replay) | [MEAS, notes/glm-decode-ops.md (line 18)] |

(The Class column is again this guide's classification [INFER, notes/glm-decode-ops.md (lines 12–18)] — the table's own columns are measured.) The plan0h column composes: $9295 - 983 = 8312$ us. **Column caveat, verbatim from the note:** "Each row is cumulative over the one above. The last row's second column is hotM-meas, against the row above re-measured on hotM-meas" [MEAS, notes/glm-decode-ops.md (lines 20–21)] — the hotE chain ends at 8980 through row 7; the final row's 9021/9050 is a hotM-meas measurement and does not chain into the hotE column. The registry's ledger line is: base 9295/9925 us (plan0h/hotE) down to 8312/9021 through this table [MEAS, notes/glm-decode-ops.md (lines 11, 18)].

## Phase 2 — measured cuts (roofline §6)

On `glm-dops` (decode-ops' mHC commits on a7dda80), median device time over positions 31–60 [MEAS, notes/glm-decode-roofline.md (lines 225–233)]:

| Set | plan0h ms | hotE-meas ms | Citation |
| --- | ---: | ---: | --- |
| glm-dops | 8.489 | 9.129 | [MEAS, notes/glm-decode-roofline.md (line 232)] |
| + no cycles write-back, step window, KDA prefetch, inputs fused, prefetch split | 7.356 | 7.973 | [MEAS, notes/glm-decode-roofline.md (line 233)] |

Steps on G, plan0h: the first three commits 8.599 → 8.170, inputs fused into their steps 7.991, the split prefetch 7.45; "KDA went from ~59 to 34.5 us per layer, record fetches from 191 to 32 us per token, and ~127 input steps per token went away. The text is unchanged." [MEAS, notes/glm-decode-roofline.md (lines 235–237)]. The prefetch-split discipline those commits learned is [`03`](./03-prefetch-windows-and-noc-id-budget.md)'s four rules.

## Phase 3 — the idle map and the moves

`glm-mla6` reached 6.40 ms per token on plan0h [MEAS, notes/glm-decode-roofline.md (line 255)]; on `glm-mla7r` the whole-token base measured 6100 (plan0h, box F) / 6092, 6091 (plan0h, box H) / 6697 (hot3-meas, box H) [MEAS, notes/glm-decode-roofline.md (line 301)]. The overlap moves — query/value/router first strips into the stage from idle windows — are [`03`](./03-prefetch-windows-and-noc-id-budget.md)'s: the table, its column semantics, the per-column composition, and the dropped moves all live there. Bottom line per column: plan0h F $6100 \to 5924$ us, plan0h H $6092 \to 5939$ (second run $6091 \to 5938$), and on hot3-meas H only move 4 was measured, $6697 \to 6607 / 6627$ us [MEAS, notes/glm-decode-roofline.md (lines 301–304)].

## Phase 4 — step boundaries and bookkeeping (roofline §8)

On `glm-hd` (data cache off since 31776c13), each row adding one change, device us/token [MEAS, notes/glm-decode-roofline.md (lines 327–340)]:

| Change | plan0h | hot3-meas | Step | Citation |
| --- | ---: | ---: | ---: | --- |
| glm-hd | 6149, 6158 | 6774 (one clean run) | — | [MEAS, notes/glm-decode-roofline.md (line 335)] |
| + prefetch jobs' args staged clear of the next step (race fix) | 6176, 6180 | 6804, 6798 | +25 | [MEAS, notes/glm-decode-roofline.md (line 336)] |
| + no barrier after an op that ends with the leads met, or before a GLM_MATVEC | 6112, 6118 | 6738, 6722 | −64, −71 | [MEAS, notes/glm-decode-roofline.md (line 337)] |
| + no barrier where the steps up to the next meet touch no L1 a peer still uses | 6082 | 6699, 6698 | −33 | [MEAS, notes/glm-decode-roofline.md (line 338)] |
| + record ring read ahead at window restarts, each step copies only its own arg words | 6005, 6005 | 6616, 6617 | −76, −84 | [MEAS, notes/glm-decode-roofline.md (line 339)] |
| + 4-word copies, record control words loaded before the first store | 5978, 5987 | 6598, 6603 | −23, −18 | [MEAS, notes/glm-decode-roofline.md (line 340)] |

The two record-ring rows are [Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md)'s measured wins (cited there, owned here as ledger entries); the phase composes to $6149 - 171 = 5978$ us on plan0h and $6774 - 176 = 6598$ us on hot3-meas. The data-cache lesson underneath it: with the cache off, "an L1 store between L1 loads makes each later load wait out the full L1 latency: about 10 cycles a word when loads and stores alternate" — the step's 48-word args copy and staged job copies paid it on every step [MEAS, notes/glm-decode-roofline.md (lines 342–348)].

## The placement + claims line

The streaming mechanics of [`02`](./02-strip-claims-and-reader-placement.md) landed as one stack (whole-token device us; box F, four interleaved rounds, each row cumulative) [MEAS, notes/glm-decode-ops.md (lines 219–231)]:

| Build | hot3-meas | plan0h | Citation |
| --- | ---: | ---: | --- |
| placement 9c7e9362 | 6953 / 6966 / 6965 / 6959 | 6364 / 6364 / 6362 / 6362 | [MEAS, notes/glm-decode-ops.md (line 224)] |
| claims ahead 660d6b25 | 6973 / 6904 / 6972 / 6906 | 6309 / 6385 / 6306 / 6307 | [MEAS, notes/glm-decode-ops.md (line 225)] |
| two strips in flight a9ca9001 | 6920 / 7012 / 6919 / 6926 | 6328 / 6326 / 6318 / 6324 | [MEAS, notes/glm-decode-ops.md (line 226)] |
| downs claimed b2cb77c1 | 6892 / 6888 / 6893 / 6888 | 6317 / 6316 / 6315 / 6310 | [MEAS, notes/glm-decode-ops.md (line 227)] |

"The middle two rows swing by 60-90 us between rounds on hot3; the stack lands together: −72 hot3-meas, −48 plan0h" [MEAS, notes/glm-decode-ops.md (lines 229–231)]. On the line at `head 28aedc1f` (two interleaved rounds): head 6769/6848 hot3, 6228/6157 plan0h; + placement 6747/6752, 6154/6156; + placement, claims, KDA slices on NoC 1: 6638/6725 hot3, 6053/6059 plan0h — and without the placement 6829/6831, 6163/6161, so "the claims need each channel's helpers near it" [MEAS, notes/glm-decode-ops.md (lines 254–263)].

Two further ledger entries on that line:

- **Routed downs started in the MoE middle job** (`4dca2e02`): each helper claims its first routed down strip with its gate/up claims and starts its reads when its gate/up strips run out, "so its DRAM channel no longer idles between the middle job and the down job" — about −90 us on hot3 and +7 on plan0h [MEAS, notes/glm-decode-ops.md (lines 265–275)].
- **mHC sites and the blocks after them**: leads share the post/comb logits with their squares (−7/−7 hot3, −12/−13 plan0h), GO writes reprogram only the coordinate, index key norm from word loads, and the site helper job posted before the expand (−90/−88 hot3, −87/−89 plan0h) — "The first three took 40 us off the sites and put 13 on KDA and 15 on MoE. The last starts every block's prefetch about 1.7 us earlier." [MEAS, notes/glm-decode-ops.md (lines 300–307)].

## Where it landed

Single-user production decode (2026-10-10, device time per token): **6.0 ms with plan0h, 6.6 ms with hot3** [MEAS, notes/glm-batch.md (line 11)]. RC1, B=1 [MEAS, notes/post-rc/glm-rc1.md (lines 134–140)]:

| .ttq | B=1 | Citation |
| --- | --- | --- |
| hot3-meas | 165.3 t/s (6.05 ms) | [MEAS, notes/post-rc/glm-rc1.md (line 136)] |
| plan0h | 178.9 t/s (5.59 ms) | [MEAS, notes/post-rc/glm-rc1.md (line 137)] |

with repeats 165.29 / 165.30 t/s [MEAS, notes/post-rc/glm-rc1.md (line 139)]. The arithmetic:

$$\frac{1000}{165.3} = 6.05\ \text{ms/token}, \qquad \frac{1000}{178.9} = 5.59\ \text{ms/token}$$

The hot-expert delta narrowed along the ledger: +0.63 ms when the base was 9.3 ms [MEAS, notes/glm-decode-roofline.md (lines 210–211)], ~+0.46 ms at the end ($6.05 - 5.59$) — consistent with the roofline's prediction that items 1, 7 and 8 would absorb part of the hot reads' first-strip and team-skew slack [INFER, notes/glm-decode-roofline.md (lines 212–213)].

## What the ledger teaches

1. **The discipline is occupancy work.** Every large entry either moved reads into a DRAM-idle window (prefetch slot, middle-job downs, the [`03`](./03-prefetch-windows-and-noc-id-budget.md) moves) or removed serial/sync time so windows could lengthen — the plan's classes (streaming / placement / compute / sync) are one budget.
2. **Bit-exactness is the acceptance harness.** Text identity on every change made 9.3 → 6.05 a performance-only story [MEAS, notes/glm-decode-ops.md (lines 1–3)].
3. **Whole-token time is the only currency.** Per-step accounting mis-ranked moves [MEAS, notes/glm-decode-roofline.md (lines 323–325)]; refused runs are left out and said so.
4. **Columns are contracts.** Weight file and box are part of every number; deltas compose within a column only — the measured-moves table's F/H semantics ([`03`](./03-prefetch-windows-and-noc-id-budget.md)) made that a hard rule, and the hotM-meas row of the Changes table is the same lesson in Phase 1 [MEAS, notes/glm-decode-roofline.md (lines 299–305); notes/glm-decode-ops.md (lines 20–21)].
5. **The floor still stands.** 6.05 ms against a 4.77 ms hotE floor at 447 GB/s [MEAS, notes/glm-decode-roofline.md (line 18)]: reaching the 5 ms class needs DRAM busy ~95% of the token — a weight-streaming schedule where every serial piece runs under the next phase's reads [EST, notes/glm-decode-roofline.md (lines 216–217)].

---

← [`03-prefetch-windows-and-noc-id-budget.md`](./03-prefetch-windows-and-noc-id-budget.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`05-batch-and-mtp-economics.md`](./05-batch-and-mtp-economics.md)
