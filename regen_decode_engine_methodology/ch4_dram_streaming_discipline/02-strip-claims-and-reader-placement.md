# Strip Claims and Reader Placement

A **strip** — a 16-row BFP slice of a weight matrix — is the unit of DRAM transfer, and every reader is tuned around it: each helper claims strips **ahead** of the one it is projecting, keeps exactly **two strips in flight**, and sits **near** the DRAM channel its strips live in, so a channel's 12 helpers see sequential streams at 63 GB/s per channel instead of NoC-throttled fragments. This file owns strip *geometry* as the reader contract and the claims/placement measurements; it is the mechanism half of the 89%-of-peak window rate that [`01`](./01-roofline-and-byte-budget.md) established.

**Scope:** covers Question 3, part 2 — strip claims-ahead and near-channel reader placement. Ring geometry and record-stream prefetch are [Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md)'s (this file ties back, never restates); the NoC-ID *budget* behind every "in flight" here is [`03`](./03-prefetch-windows-and-noc-id-budget.md)'s; which tensor group gets which byte format is [Ch6 `02-ttq-container-and-requant.md`](../ch6_host_driver_weights_oracle/02-ttq-container-and-requant.md)'s — geometry only here. The ledger rows these mechanisms produced are [`04`](./04-optimization-ledger.md)'s.

## The strip as the unit of transfer

A BFP strip is 16 rows × K, one shared 8-bit exponent per 16 consecutive K values of a row (packing arithmetic: [Ch1 `03-workload-context.md`](../ch1_engine_at_a_glance/03-workload-context.md)) [MEAS, notes/glm-bfp8.md (lines 8–12)]. Strip *byte geometry* is the reader's contract:

$$\text{strip bytes} = \text{columns} \times \begin{cases} 32 & \text{BF16}\\ 17 & \text{BFP8}\\ 9 & \text{BFP4} \end{cases}$$

[FACT, src/brisc.cpp:1499-1502]. Strips are chunked for the NoC: "chunks of 16 KiB of data, each its tile-major exponents … then its tile-major section rows. A chunk is 2048 BFP4 columns or 1024 BFP8 columns" [FACT, notes/glm-bfp8.md (lines 10–12)] — so a 4096-column BFP8 strip is 69,632 B, and the helper stage holds **three** such buffers: "strip stage 3 x 69632 B … from 0x20000" [FACT, notes/glm-bfp8.md (line 53)]. "Three strip buffers: up to two strips can be outstanding (`strip_ahead`), and the third belongs exclusively to compute" [FACT, src/brisc.cpp:1507]. The 1024-column chunk also makes every 1024-column K slice a strip of its own, so a K-split matvec reads its slice as one contiguous range — the layout splits KDA's K = 4096 projections in four [FACT, notes/glm-bfp8.md (lines 12–15)].

## The reader contract: channel-local shares

Placement (the loader half of this contract) is [Ch6 `03-channel-local-placement.md`](../ch6_host_driver_weights_oracle/03-channel-local-placement.md)'s; the reader half:

- **Static matrices** (`TtGlmStrips`): "Helper w of W projects slice w % parts of team w / parts's contiguous strip range, stored contiguously in local DRAM channel w % 8 after the slices of the channel's lower helpers" [FACT, notes/glm-bfp8.md (lines 59–60)]. A helper's strips are one contiguous DRAM range in its own channel — a sequential stream.
- **Routed experts** (`TtGlmExpertStrips`): each expert stored once per chip; "Slice q of strip l of expert e is piece n = l * parts + q, in channel n % 8" [FACT, notes/glm-bfp8.md (lines 74–75)]. Each expert's strips interleave over channels n % 8, and each channel's 12 helpers take that channel's strips, "so every channel carries the same bytes: 1.18 MB per chip on plan0h, 1.44 MB on hotM-meas" [MEAS, notes/glm-decode-ops.md (lines 44–45)].
- **Local share beats contiguous share**: the contiguous share (team t takes virtual strips [V·t/T, V·(t+1)/T) wherever they lie) measured 6% slower on gate/up and even on down, "so only the local share remains" [MEAS, notes/glm-bfp8.md (lines 96–98)].

## Claims-ahead: the protocol

A **claim** is a NoC 1 atomic (ID 14, [FACT, notes/noc-ids.md (line 22)]) to a per-channel-group counter: "the group's entry k (as in `glm_expert_share`) goes to the k-th claim on the group's count `counter` on `coord`, so the helpers the channel serves first take more strips" [FACT, src/brisc.cpp:14892-14895]. The loop [FACT, src/brisc.cpp:14913-14960]:

1. `glm_claims_start` takes the first claimed strip into stage buffer 0 [FACT, src/brisc.cpp:14939-14942].
2. `glm_claim_take` waits the claim, issues the *next* claim, and starts the strip's reads into stage buffer 0 or 1 [FACT, src/brisc.cpp:14913-14936].
3. `glm_claims_send` projects every taken strip and takes the next into the buffer just freed — "Each claimed strip claims the next and starts its reads into strip stage buffer 0 or 1, so two strips are in flight while one projects" [FACT, src/brisc.cpp:14893-14897].

Why claims at all: with static shares, "helpers of one channel with the same bytes finished 7.5-28 us after the selection: the order the channel served its 12 readers. The lead waits for the channel's drain, 0.4-0.5 us after its last helper" [MEAS, notes/glm-decode-ops.md (lines 47–48)] — claims turn a fixed share into work-conserving self-scheduling, so no helper idles while its channel still has strips.

**Measured effect** (in-op stamps, every chip, us from the lead's MoE start) [MEAS, notes/glm-decode-ops.md (lines 198–208)]:

| Stamp | plan0h base / placed / claims ahead | hot3-meas base / placed / claims ahead |
| --- | --- | --- |
| gate/up GB/s per channel from the selection | 55.8 / 56.2 / 58.4 | 57.2 / 58.0 / 59.1 |
| routed gate/up window (us) | 21.1 / 21.0 / 20.2 | 29.8 / 29.4 / 28.9 |
| routed gate/up end (lead wait end) | 37.3 / 36.5 / 35.5 | 46.2 / 43.8 / 43.2 |

"One channel streams 63-64 GB/s in the sweep bench; the routed gate|up reached 56-58 before the claims moved ahead. With one strip in flight per helper, each strip paid its claim's round trip to lead 0 and its first read's latency before the channel saw it, plus the last strip's projection and send at the tail" [MEAS, notes/glm-decode-ops.md (lines 209–211)]. Claims ahead (`660d6b25`): "the first claim leaves before the selection, and each strip claims the next before it projects" [MEAS, notes/glm-decode-ops.md (line 212)]. The routed downs claimed the same way (`b2cb77c1`), "first two strips read before the hidden values arrive. The earlier attempt (one strip per claim, no read-ahead) lost 34-137 us; with two strips in flight it wins" [MEAS, notes/glm-decode-ops.md (lines 216–217)].

## Two strips in flight — why two is the number

Three constraints pin the depth:

- **L1 footprint.** The stage is three buffers (two outstanding + one computing) [FACT, src/brisc.cpp:1507]; the helper's L1 also carries the resident range R (157,632 B) and the prefetch slot L (69,632 B) [MEAS, notes/glm-decode-roofline.md (line 264)] — a fourth strip does not fit the budget the prefetch machinery needs.
- **Channel behavior at depth 3.** "Strips read ahead of the one computing: two, or one above 40 KiB, where a third outstanding strip per helper slows the channel (a BFP8 LM head runs at 76% of 500 GB/s with two ahead)" [FACT, src/brisc.cpp:1531-1535]. Measured: with two strips ahead, "a helper kept up to three 68 KiB strips (2.4 MB per channel) outstanding and the BFP8 LM head ran at 76% (220.9 us); the same bytes as 34 KiB strips took 182.0 us … `strip_ahead` now reads one strip ahead above 40 KiB: the LM head takes 182.8 us (92%), 38 us less per token" [MEAS, notes/glm-bfp8.md (lines 242–246)]. So *two in flight* is the sweet spot at the 36–40 KiB strip sizes; deeper read-ahead pays only below 40 KiB.
- **Claim granularity beats read-ahead on the hot path.** "Two or three routed gate|up strips per claim, read ahead within a claim … against one: plan0h 6391 / 6395 against 6444 / 6444 (two) and 6542 / 6548 (three); hotM 7084 / 7005 against 7199 / 7205 and 7215 / 7225 … Claim granularity matters more than read-ahead here" [MEAS, notes/glm-decode-ops.md (lines 38–40)]. The winning shape is one strip per claim with two claims in flight; and even the read-ahead threshold matters: dropping it from 40 KiB to 32 KiB (the 36 KiB cold expert strips) cost plan0h +16/+16 and swung hotM −84/+91 [MEAS, notes/glm-decode-ops.md (lines 34–35)].

## Near-channel reader placement

The host builds each chip's grid with the NoC 0 endpoint *nearest* each Tensix row: "The NoC 0 endpoint of `channel` nearest Tensix row `row`" — `dram_coord_near` [FACT, src/tt_backend.cpp:1125-1126], stored per channel into `grid.dram` [FACT, src/tt_backend.cpp:4743-4744]. The rationale: "responses travel along the endpoint's row first, so spreading readers over endpoints spreads DRAM traffic over NoC rows" [FACT, src/tt_backend.cpp:1118-1119].

**The sweep bench** (all 8 channels busy at once; per-channel median / worst, GB/s) [MEAS, notes/glm-decode-ops.md (lines 169–176)]:

| Placement | Per channel | Per chip | Citation |
| --- | ---: | ---: | --- |
| near, 1 MB per reader, 1–12 readers | 63.0–63.8 / 61–63 | 503–510 | [MEAS, notes/glm-decode-ops.md (line 170)] |
| decode (helper w on channel w % 8) | 50–62 / 42–59 | 385–491 | [MEAS, notes/glm-decode-ops.md (line 169)] |
| far | 13–42 | 125–344 | [MEAS, notes/glm-decode-ops.md (line 171)] |

Each channel alone streams 62.7–63.9 GB/s from 1 to 12 readers, near or far, on either NoC — "With all 8 busy the readers' NoC paths decide it" [MEAS, notes/glm-decode-ops.md (lines 163–164)]. Once placement is right, "request size (8 or 16 KB), reads in flight (2–8) and a static request VC change little" [MEAS, notes/glm-decode-ops.md (line 176)]; on NoC 0 the near deal puts each channel's twelve readers in the rows of its endpoints, so responses run 1–9 hops east (4.6 on average against 12.3 for the decode numbering) [MEAS, notes/glm-decode-ops.md (lines 177–178)]; permuting channel slots within each lead row is *worse* than row order (457–471 GB/s per chip) [MEAS, notes/glm-decode-ops.md (line 179)].

**Renumbering the helpers** to the near deal (`9c7e9362`) measured, per token (us) [MEAS, notes/glm-decode-ops.md (lines 181–191)]:

| Component | plan0h F r1 / r2 | plan0h A | hot3-meas A |
| --- | ---: | ---: | ---: |
| total | −32.0 / −28.6 | −27.2 | −28.8 |
| MoE | −46.6 / −46.0 | −44.6 | −61.6 |
| dense FFN | −19.9 / −20.3 | −19.9 | −19.8 |
| head | −16.5 / −15.6 | −16.5 | −15.9 |
| MLA + indexer | +23.2 / +22.3 | +21.7 | +21.9 |
| KDA | +26.1 / +27.2 | +28.8 | +28.6 |
| mHC sites | +2.1 / +1.0 | +2.5 | +6.7 |

(`F` and `A` are the note's box labels — LoudBox hosts; `r1`/`r2` are the two interleaved rounds. The table is complete: no rows are omitted.) By step: MLA query −10, MLA gate −7, dense gate/up −11, dense down −4; MLA value +29.5 — every helper reads the whole latent from lead 0 — and KDA +27 (its layer check rose from 61147 to 62027 cycles) [MEAS, notes/glm-decode-ops.md (lines 192–193)]. The placement wins net −27 to −32 us even while giving back on MLA, KDA and the mHC sites — and it is a *co-design*, not a free win: matching each lead's workers to the channel helpers nearest the lead made KDA worse still (+72) and the MoE gain smaller (−33) [MEAS, notes/glm-decode-ops.md (lines 195–196)]. The placement is also a *precondition* for claims: "The claims need each channel's helpers near it: without the placement the line loses to head" [MEAS, notes/glm-decode-ops.md (lines 258–263)].

**A complementary NoC-path fix.** After the near-channel placement, the KDA slice helpers' sequenced posts queued on NoC 0 behind other helpers' o_proj prefetch responses and the heads round slipped; posted on NoC 1's static VC, "the slices land 0.05 us after they finish; the layer check went from 61-62k cycles to 58.7k" — whole-token 6842/6848 hot3-meas, 6264/6271 plan0h [MEAS, notes/glm-decode-ops.md (lines 232–243)]. Channel placement and NoC-path placement are two halves of the same rule: keep each reader's requests off congested paths [INFER, notes/glm-decode-ops.md (lines 232–243)].

## Tie-back: claims-ahead generalizes the record ring

[Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md) owns the record ring: keep the next records in flight so the interpreter never waits on its own stream. Claims-ahead is the same mechanism family applied to weight strips — keep the next strips in flight so compute never waits on its stream — with two differences the measurements forced: strips are big enough that *depth* is a measured resource (two, not thirty-one-ahead), and strips are shared across a channel's 12 readers, so the pacing unit is a claim on a channel counter, not a private cursor [INFER, src/brisc.cpp:14913-14960].

---

← [`01-roofline-and-byte-budget.md`](./01-roofline-and-byte-budget.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`03-prefetch-windows-and-noc-id-budget.md`](./03-prefetch-windows-and-noc-id-budget.md)
