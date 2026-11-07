# Prefetch Windows and the NoC-ID Budget

A **DRAM prefetch window** (always qualified — never bare, never "prefill window") is a scheduled burst of DRAM reads that batches strip traffic around the serial phases of the token, so DRAM stays busy while leads run sites, routing and rounds. This file owns the window-scheduling rules, the DRAM-idle map that motivates them, the measured moves, the effective per-window rates — and the **NoC transaction-ID budget**, the finite resource (16 IDs per NoC) that every "in flight" in this chapter is accounted against.

**Scope:** covers Question 3, part 3 — prefetch windows and NoC-ID budgeting. Decode-streaming only: prefill-phase design is out of guide scope and is fenced out by name (regen's prefill work lives in [notes/prefill-design.md](../../../regen-glm5.3-flash/notes/prefill-design.md), outside this guide). The other window senses — SFPU replay windows ([Ch3 `03`](../ch3_raw_mmio_sfpu_replay/03-sfpu-replay-windows.md)) and PCIe BAR windows ([Ch5](../ch5_cross_chip_transports/index.md)/[Ch6](../ch6_host_driver_weights_oracle/index.md)) — are never conflated with this one. Strip/claims mechanics are [`02`](./02-strip-claims-and-reader-placement.md)'s; the ledger that consumed these windows is [`04`](./04-optimization-ledger.md)'s; transport latencies are [Ch5](../ch5_cross_chip_transports/index.md)'s; the record-ring pacing that window restarts protect is [Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md)'s.

## What a DRAM prefetch window is

Inside a step, the interpreter can post DRAM reads whose consumer is a *later* step: a **resident burst** into the resident range R, a **stage read** into the strip stage S, or a **slot prefetch** into the slot L [MEAS, notes/glm-decode-roofline.md (lines 261–264)]. The plan's item 1 states the shape: "helpers read the next op's static weights while the lead runs the site, expand, routing and rounds. KDA projections during the attention site …; router and shared expert during the FFN site; MLA q_b/o_proj during the flash; the LM head's first strips during layer 44. Needs a helper L1 budget of ~60-150 KB for a full-share stage" [MEAS, notes/glm-decode-roofline.md (line 198)]. A step whose op will post a burst can also set its `window` flag to restart the record ring ahead of the burst — that pacing contract is [Ch2 `04-read-ahead-rings.md`](../ch2_record_driven_execution/04-read-ahead-rings.md)'s.

## The demand: where DRAM idles

On a traced 6.26 ms token (plan0h, `glm-mla6`, with a trace-only scratch: every helper stamps each matvec with its DRAM bytes, each prefetch issue, and the moment its NoC ID 1 or 2 drains while idle) helpers read 1728.6 MB per token per chip — matvec 1090, resident bursts 604, slot 35 MB — which is $\frac{1.7286\ \text{GB}}{447\ \text{GB/s}} = 3.87$ ms of streaming; "DRAM is idle for 2.29 ms of the 6.26 ms token (2.16 ms below 10%)" [MEAS, notes/glm-decode-roofline.md (lines 255–262)]. The idle time, by kind [MEAS, notes/glm-decode-roofline.md (lines 267–276)]:

| Idle kind | us/token | L1 free for prefetches | Static reads that fit there | Citation |
| --- | ---: | --- | --- | --- |
| MLA steps (11) | 818 | S from the low rank to the query; R from the gate to the flash | q_b + wq_b (4.2 MB); W_UK, W_UV (2.2 MB) in the flash | [MEAS, notes/glm-decode-roofline.md (line 269)] |
| KDA after the projections (34) | 491 | R from the projections to the FFN site | the next layer's attention input | [MEAS, notes/glm-decode-roofline.md (line 270)] |
| Step boundaries (345) | 444 | — | none | [MEAS, notes/glm-decode-roofline.md (line 271)] |
| mHC sites (90) | 424 | S, L | the MoE router and shared expert | [MEAS, notes/glm-decode-roofline.md (line 272)] |
| MoE downs and combine (42) | 217 | half of R | the routed downs | [MEAS, notes/glm-decode-roofline.md (line 273)] |
| MoE routing and Ra (42) | 160 | S, L | the router (0.39 MB, on the chain) | [MEAS, notes/glm-decode-roofline.md (line 274)] |
| Dense FFN (3) | 80 | S | gate/up, partly | [MEAS, notes/glm-decode-roofline.md (line 275)] |
| Head | 25 | R, S | the LM head's first strips | [MEAS, notes/glm-decode-roofline.md (line 276)] |

Helper L1 carries three prefetch regions: R, the resident range (157,632 B); S, the strip stage (3 x 69,632 B, free between a helper's matvecs); L, the prefetch slot (69,632 B) [FACT, notes/glm-decode-roofline.md (line 264)]. First estimates put the overlap moves at about 0.5–0.6 ms [EST, notes/glm-decode-roofline.md (line 290)]; the 1.7 ms they cannot reach is serial work with no static weights left to read under it — and the routed experts (1.33 ms of DRAM-bound time) cannot move before routing [MEAS, notes/glm-decode-roofline.md (lines 290–292)] — the same demand-signal property [Ch1 `03-workload-context.md`](../ch1_engine_at_a_glance/03-workload-context.md) states for the byte budget.

## The four rules, each taught by a measured failure

From the phase-2 prefetch work [MEAS, notes/glm-decode-roofline.md (lines 239–251)]:

1. **A prefetch burst queues every later small DRAM read behind it.** The first KDA prefetch (~15 MB per chip, ~2 MB per channel, issued at once by 96 helpers) delayed lead 0's 16-byte cycles write-back by ~25 us, then the next step's record read by ~18 us, then the attention site's 20 KB statics by ~16 us. Fix: before posting a burst, land or move to L1 every small read the steps up to its consumer need; the step window restarts at the posting step, the site posts only after its statics land, nothing writes DRAM per step [MEAS, notes/glm-decode-roofline.md (lines 240–244)].
2. **Size a burst to its idle window.** A burst longer than the window just moves the wait into the consumer: the attention site's ~15 us window left KDA waiting ~13 us for a ~30 us burst. Splitting it over the FFN and attention sites removed the wait without slowing the MoE [MEAS, notes/glm-decode-roofline.md (lines 245–247)].
3. **Give a resident prefetch its own NoC ID.** Waits are per ID, so sharing the slot prefetch's ID 1 makes every slot wait also wait for the burst; resident reads use ID 2, which decode leaves to the leads [MEAS, notes/glm-decode-roofline.md (lines 248–249)]. (The ID allocation behind this rule is below.)
4. **Place resident data where no op between the posting step and the consumer writes.** KDA shares sit past GLM_MOE's helper work (38,960 B), "leaving 157,632 B for a 156,672 B maximum share" [MEAS, notes/glm-decode-roofline.md (lines 250–251)].

## Chain delay, and how to rank a move

A burst sits in the DRAM channel FIFOs ahead of every later small read on the chain, *whatever its NoC ID*, so a move's net saving is the critical DRAM time it removes minus the delay to the chain reads queued behind it — about the burst's backlog per channel at 63 GB/s per channel [MEAS, notes/glm-decode-roofline.md (lines 317–319)]. Record-window reads are among those delayed reads: a step with `window` set reads its 16 records before its op, behind any burst in flight [MEAS, notes/glm-decode-roofline.md (lines 319–321)].

The methodological rule this forced: **rank and accept moves on whole-token device time only.** An earlier per-step dump timed each step after its window read and so missed about 45 us/token on `glm-mla7r` and 210 on `glm-mla6`; it ranked move 1 at −37, which whole-token timing later showed as a loss; the profile tool refuses a trace whose step sums miss the device line by more than 10 us [MEAS, notes/glm-decode-roofline.md (lines 323–325)]. (The tool also leaves out the ~1-in-5 runs where lead 0's trace ring misbehaves, 75–80 us slow [MEAS, notes/glm-decode-roofline.md (lines 307–308)] — refused runs are left out and said so.)

## The measured moves — exact column semantics

Whole-token device time (median us/token over positions 30–63, `glm-mla7r` with the profile commit) [MEAS, notes/glm-decode-roofline.md (lines 296–298)]. **Column discipline:** the columns carry BOTH a weight class and a box — `plan0h F | plan0h H | hot3-meas H`, where `plan0h` and `hot3-meas` are weight files and `F`/`H` are **box labels** (the LoudBox hosts the runs executed on), *not* weight classes. Cells the note leaves blank are shown as em-dashes, never as zeros — **only move 4 was measured on hot3-meas** [MEAS, notes/glm-decode-roofline.md (lines 299–305)]:

| Change (cumulative) | plan0h F | plan0h H | hot3-meas H | Where (plan0h) | Citation |
| --- | ---: | ---: | ---: | --- | --- |
| base | 6100 | 6092, 6091 | 6697 | | [MEAS, notes/glm-decode-roofline.md (line 301)] |
| + 3: query's first strips into S from the gate | −24 | −16 | — | MLA −19 | [MEAS, notes/glm-decode-roofline.md (line 302)] |
| + 5': value's first strips and o_proj's upper half from the flash, behind its gather | −68 | −60 | — | MLA −70 | [MEAS, notes/glm-decode-roofline.md (line 303)] |
| + 4: router's strips into S from the FFN site, ahead of its burst | −84 | −77, −77 | −90, −70 | MoE −21 | [MEAS, notes/glm-decode-roofline.md (line 304)] |
| + 4, without the gate step's window restart | — | −83, −81 | −86, −51 | not kept | [MEAS, notes/glm-decode-roofline.md (line 305)] |

Each column composes **independently** — F deltas with the F base, H deltas with the H base, never mixed:

$$\text{plan0h F: } 6100 - 24 - 68 - 84 = 6100 - 176 = 5924\ \text{us}$$

$$\text{plan0h H: } 6092 - 16 - 60 - 77 = 6092 - 153 = 5939\ \text{us} \quad (6091 \to 5938)$$

The hot3-meas H column has no cells for moves 3 and 5', so no cumulative chain exists there; move 4 alone measured −90/−70 on hot3-meas H, i.e. $6697 - 90 = 6607$ or $6697 - 70 = 6627$ us [MEAS, notes/glm-decode-roofline.md (lines 301–304)] — the composed endpoints are this guide's arithmetic over the note's per-move deltas; the note itself reports only the deltas.

Dropped, with the reasons [MEAS, notes/glm-decode-roofline.md (lines 310–315)]: move 1 (the next KDA layer's first upper eighth from the KDA o_proj phase) — with 3, +61 (KDA +129, MoE −42), +23 without its window restart: "its burst queues the KDA's own later reads"; move 2 (routed downs' stage reads straight after each helper's gate/up) — +125: "The downs' bursts sat in the channel FIFOs ahead of the routed gate|up tail"; moves 6 and 7 parked (under 0.02 ms each).

## Effective window rates

- **Aggregate inside windows:** 447 GB/s per chip = 89% of the 500 GB/s peak [MEAS, notes/glm-decode-roofline.md (line 108)] — [`01`](./01-roofline-and-byte-budget.md)'s tables own the decomposition.
- **Per shape, synchronized strip bench** ("one synchronized pass of every helper of 8 chips, 20 repeats") [MEAS, notes/glm-bfp8.md (lines 211–233)]:

| Shape | GB/s | of 500 | Citation |
| --- | ---: | ---: | --- |
| KDA q/k/v/b/f_a/g_a, BFP8 | 457 | 91% | [MEAS, notes/glm-bfp8.md (line 216)] |
| KDA q/k/v/b/f_a/g_a, BFP4 | 443 | 89% | [MEAS, notes/glm-bfp8.md (line 217)] |
| dense down, BFP8 | 397 | 79% | [MEAS, notes/glm-bfp8.md (line 228)] |
| LM head, BFP8 / BFP4 | 461 / 458 | 92% / 92% | [MEAS, notes/glm-bfp8.md (lines 230–231)] |
| router, all rows, BF16 | 398 | 80% | [MEAS, notes/glm-bfp8.md (line 233)] |

- **In situ vs bench:** the MoE middle job moves shared gate/up 2.2 MB, the shared down's prefetch 1.1 MB and the routed gate/up 9.4 / 11.5 MB "from 3.5 us to the drain at about 410 GB/s per chip, against 465-500 in the strip benches" [MEAS, notes/glm-decode-ops.md (lines 51–52)] — first-strip latency, team skew and round overhead are the difference [INFER, notes/glm-decode-ops.md (lines 51–52)].
- **Per channel:** one channel streams 63–64 GB/s in the sweep bench; the routed gate/up reached 56–58 GB/s per channel before the claims moved ahead (58.4 with claims; [`02`](./02-strip-claims-and-reader-placement.md) owns the claims table) [MEAS, notes/glm-decode-ops.md (lines 209, 203)].
- **A window that is not DRAM-idle does not pay.** The selection-wait window is "core idle, not DRAM idle": each channel still carries 4 shared gate/up strips (272 KB) plus 140 KB of shared down — 412 KB, 6.5–7.9 us at 52–63 GB/s per channel — so "new prefetch there is not worth it", and "routing faster than the shared reads only moves the routed gate|up's start onto them" [MEAS, notes/glm-decode-roofline.md (lines 374–376); notes/glm-decode-ops.md (lines 108–111)].

## The NoC-ID budget

**The resource.** "Each NIU counts requests in flight per transaction ID, 16 IDs per NoC" — `REQS_OUTSTANDING` counts reads and non-posted writes until their acks return — "and a wait on an ID waits for all of them, so on a core an ID serves one role at a time" [FACT, notes/noc-ids.md (lines 3–4); src/brisc.cpp:354]. An acked multicast counts one request but takes one ack per destination tile, so its ID never drains and any wait on it hangs — on box a08u32, strip chunk IDs that reached 15 hung exactly this way; `c8811acb` kept them below 12, and an audit moved every ID behind one allocation [FACT, notes/noc-ids.md (lines 5–7)].

**The decode allocation** (current source; the full audit, including the prefill-image rows, is in [notes/noc-ids.md](../../../regen-glm5.3-flash/notes/noc-ids.md) — prefill allocations are out of guide scope, fenced by name) [FACT, src/brisc.cpp:354–373]:

| IDs | Role | NoC 0 | NoC 1 | Citation |
| --- | --- | --- | --- | --- |
| 0 | general | default write tag; `noc_read_to_l1_async` | all `noc1_write_from_l1_async` writes | [FACT, notes/noc-ids.md (line 17)] |
| 1–3 | general, prefetch | per-command reads; slot prefetch ID 1, resident prefetch ID 2 (issued in one job, a later job waits) | general reads (`noc_read_alternate`; prefill produce roles out of scope) | [FACT, notes/noc-ids.md (line 18); src/brisc.cpp:359–361] |
| 4–12 | strip staging | `noc_chunk_id(b, c)`: buffer b lands chunk c on `4 + 3b + min(c, 2)` — three chunk IDs per buffer | none | [FACT, src/brisc.cpp:362–364] |
| 13 | sequence / flight | `post_sequenced` flag writes | flight recorder posts | [FACT, notes/noc-ids.md (line 21)] |
| 14 | step ring / claims | GLM leads' step-record ring reads, outstanding across steps | expert claim atomics (`noc1_claim_issue`) | [FACT, notes/noc-ids.md (line 22)] |
| 15 | acked casts | `NocCastId` — never read or waited on | same | [FACT, notes/noc-ids.md (line 23)] |

**Note-vs-source state, flagged:** the note's own printed allocation table still shows the *earlier audit-time* form — strips on IDs 4–9 at `4 + 2b + min(c, 1)`, "10–12 unused while strips take two IDs per buffer" [FACT, notes/noc-ids.md (lines 19–20)] — a two-IDs-per-buffer layout the current source comment supersedes with three per buffer on 4–12; the source is authoritative here, and the audit itself moved the sequence ID to 13 to sit "above every strip layout up to three IDs per buffer (4-12)" [FACT, notes/noc-ids.md (line 94)]. (In the current image ID 12 also carries the glm-batch prefill matmul prefetch — a prefill-image role, out of scope here [FACT, src/brisc.cpp:364].) Strip staging granularity is itself a measured tradeoff: "each buffer waits on three chunk IDs instead of four. That cost hot3 about 5 us per token (6595.2 to 6600.4 us, a08u32, 4 rounds) and plan0h nothing; two IDs per buffer cost hot3 5 and plan0h 8. Fewer, larger chunks per strip, so three IDs cover a whole strip, might win it back" [MEAS, notes/glm-decode-ops.md (lines 116–118)].

**IDs are not raw integers.** `NocId` has a private constructor; its only sources are range-checked functions — compile-time `static_assert` for constants, runtime checks (stop `0xBADB2000`) in the TT_NOC_CHECK images for computed indices [FACT, notes/noc-ids.md (lines 9–13); src/brisc.cpp:376–399].

**What stays in flight across steps** (the classes that make every "in flight" in this chapter sound) [FACT, notes/noc-ids.md (lines 58–71)]: a slot prefetch issued in an earlier job until `slot_matvec` waits for it (ID 1); a resident prefetch until `glm_resident_matvec` waits (ID 2) — "a re-posted prefetch with no consumer piles up on the ID, and the budget stops the core past 255"; router stage strips issued in the site job with the MoE up job's strips (the strip staging IDs); `post_sequenced` flag writes (ID 13); and one outstanding claim atomic per claim on NoC 1 ID 14. "No two roles share an ID on one core outside the two exceptions, and the TT_NOC_CHECK images enforce that" [FACT, notes/noc-ids.md (line 71)].

**Failure mode 1 — the undraining ID.** The acked-multicast trap above (a08u32); the audit's other finds include chunk IDs (4 + 4b + c before the fix) reaching 12 (sequence) and 15 — "the a08u32 hang" [FACT, notes/noc-ids.md (lines 5–7, 87)]. (The Qwen and GLM *prefill* collisions are recorded in the same list — fenced by name: prefill.)

**Failure mode 2 — the 8-bit counter.** "An ID's outstanding counter holds 8 bits, so on silicon a wait after more than 255 requests can return while some are still in flight. … Reads and unicast writes both count. Bursts past 255 pace with `noc_pace_outstanding` (drain below 127 every 128)" [FACT, notes/noc-ids.md (lines 98–101)].

**The check.** TT_NOC_CHECK (ttsim images) stops a core with `0xBADB0000 | NoC << 4 | ID` (more than 255 requests between counter reads), `0xBADB8000 | …` (a request of one role on an ID that still has another's), or `0xBADB2000 | …` (a computed ID outside its role's range) [FACT, notes/noc-ids.md (lines 113–121)]. Silicon images leave the check out — its runtime flag cost GLM decode about 50 us per token [FACT, src/brisc.cpp:343–344].

**A related acceptance race, same audit family.** `noc1_write_from_l1_async` does not wait for its command to be accepted, and a write counts as outstanding only once it is, so a NoC 1 write wait polled right after it can pass before the write leaves. With the KDA sends on NoC 1 and the old `noc1_wait_writes`, helpers ended jobs with partials still unissued and every lead stopped at the next MLA flash's argument check (0xBAD6) in 4 of 13 plan0h runs on F; with the acceptance waits, 0 of 12 [MEAS, notes/glm-decode-ops.md (lines 248–252)]. This is the readers-own-synchronization discipline of [Ch3 `02-instruction-port.md`](../ch3_raw_mmio_sfpu_replay/02-instruction-port.md) at the NoC layer: the writer must wait for *acceptance*, not just issue.

**ERISC is outside this budget.** The Ethernet cores have their own NIU, "so their IDs do not interact with the Tensix IDs" [FACT, notes/noc-ids.md (line 75)] — the message-service ring's IDs are [Ch5 `02-erisc-message-service.md`](../ch5_cross_chip_transports/02-erisc-message-service.md)'s.

## Why the budget is a streaming-discipline concern

Every mechanism in this chapter is an "outstanding across steps" pattern: slot prefetches outlive their posting job, claims outlive the projection, resident bursts outlive the site, and the record ring (Ch2) outlives the step. The NoC-ID budget is what lets all of them coexist without one class's wait accidentally coupling to another's — the scheduler's side of the same discipline that claims-ahead provides on the data side [INFER, notes/noc-ids.md (lines 58–71)].

---

← [`02-strip-claims-and-reader-placement.md`](./02-strip-claims-and-reader-placement.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`04-optimization-ledger.md`](./04-optimization-ledger.md)
