# Read-Ahead Rings

Each lead's records arrive through a DRAM **[ring](./index.md)** in its own L1: a circular buffer of 32 record slots that the interpreter fills with non-blocking bulk reads on a dedicated NoC ID and consumes under a two-watermark protocol, so the fetch of upcoming records overlaps the op of the current one. This file owns the ring's geometry, its reader, its pacing contract, its window-restart rationale, and the measured wins; the same mechanism family generalized to weight **[strips](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)** is [Chapter 4](../ch4_dram_streaming_discipline/index.md)'s, deferred by name throughout.

**Scope:** covers Question 1, part 4 — ring geometry and record-stream prefetch only. Two-strips-in-flight, strip claims-ahead, the NoC-ID *budget* across reader classes, and all DRAM window rates are [Ch4 `02-strip-claims-and-reader-placement.md`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md) and [Ch4 `03-prefetch-windows-and-noc-id-budget.md`](../ch4_dram_streaming_discipline/03-prefetch-windows-and-noc-id-budget.md)'s; this file claims none of them.

## Why a ring: records are DRAM traffic too

`GLM_STEPS` tells every lead to run `count` records "from its own DRAM channel (the lead's index)" [FACT, src/constants.h:2385-2386] — on the order of half a thousand to a thousand 256-byte fetches per token (561 steps on the dense list, 693 on the index list [MEAS, notes/glm-decode-roofline.md (line 60)]). Fetched naively, each step would pay a full DRAM round trip before its op could even copy its args. The ring is the fix: keep the next records resident ahead of execution, so by the time step $i$ runs, its record is already in L1 and reads for records after it are in flight [INFER, src/brisc.cpp:21044-21178].

## Geometry

The ring's constants are one block of the L1 map [FACT, src/constants.h:3344-3347]:

```text
#define TT_GLM_STEP_WINDOW 16u
#define TT_GLM_STEP_WINDOW_ADDR ((TT_GLM_MLA_LATENT_PENDING_END + 63u) / 64u * 64u)
#define TT_GLM_STEP_RING (2u * TT_GLM_STEP_WINDOW)
#define TT_GLM_STEP_WINDOW_END (TT_GLM_STEP_WINDOW_ADDR + TT_GLM_STEP_RING * 256u)
```

So: `TT_GLM_STEP_WINDOW` = 16 records and `TT_GLM_STEP_RING` = 2 × 16 = 32 slots [FACT, src/constants.h:3344-3346], each slot one 256-byte `TtGlmStepRecord` [FACT, src/constants.h:2428] — $32 \times 256 = 8192$ bytes of L1. "Record r in slot r % TT_GLM_STEP_RING" [FACT, src/brisc.cpp:21044-21045]: the ring is indexed modulo its capacity, so a contiguous run of reads always lands contiguously in L1 and a read never splits a record [INFER, src/brisc.cpp:21049-21056]. The ring sits after the interpreter's persistent state region — resident streams, histories, pending mixes ([`02-interpreter-loop.md`](./02-interpreter-loop.md)) — one planned L1 layout, not an allocator's output [INFER, src/constants.h:3337-3347].

Two of the map's static_asserts pin the shape [FACT, src/constants.h:3372-3375]: the ring ends below the control block, and the whole ring fits one NoC bulk transfer — `TT_GLM_STEP_RING * 256 <= TT_NOC_MAX_BULK_BYTES` (16384) [FACT, src/constants.h:3373; src/constants.h:1446]. The placement side makes every ring read local: each lead reads its own DRAM channel's copy of the list — replicated into every channel of every chip at placement, per [`01-step-records.md`](./01-step-records.md)'s replication contract — resolved through its `TtGrid` word (`grid.dram[grid.worker]`) [FACT, src/brisc.cpp:21154-21155].

## The reader

`glm_step_ring_read` "Reads records first.. of the `count` in the list, up to TT_GLM_STEP_RING - 1 past record i, into their slots of the step ring ... on the ring's NoC ID, without waiting for them. Returns the end of the records it read." [FACT, src/brisc.cpp:21044-21047] — every read a `noc_read_to_l1_async_id` from the lead's own DRAM channel, chunked only where the slot index wraps [FACT, src/brisc.cpp:21048-21056]. Three properties carry the design [INFER, src/brisc.cpp:21044-21056]:

- **Up to 31 records past the current one** can be in flight — ring capacity minus one, the bound the steady state never exceeds.
- **The reads never block the caller** — issued and left in flight; the *consumer* waits, and only when it must.
- **Fetches travel as multi-record bulk transfers** — up to 8 KB per post — so the record stream does not dribble the NoC.

## The ring's NoC ID

Record reads ride a reserved ID, defined in the NoC ID table as "GLM leads' step-record ring (`noc_step_ring_id`)" [FACT, src/brisc.cpp:366], with the role comment: "Step-record reads that stay in flight across steps, until a later step or the token's end waits for them" [FACT, src/brisc.cpp:464-468]. That isolation is what makes "in flight across steps" sound: ring reads are counted and drained as a class of their own, so an in-flight record read is invisible to every other wait on the core, and a step's compute never accidentally consumes or waits on the record stream's outstanding transactions [INFER, src/brisc.cpp:464-468]. The allocation of IDs across weight streams, gathers, and relays — the budget question — is [Ch4 `03-prefetch-windows-and-noc-id-budget.md`](../ch4_dram_streaming_discipline/03-prefetch-windows-and-noc-id-budget.md)'s.

## Pacing: claims-ahead at the record level

`glm_step_list` tracks two cursors — "Records below `ready` have landed; those from `ready` to `end` are in flight." [FACT, src/brisc.cpp:21156-21157] — and per step [FACT, src/brisc.cpp:21160-21173]:

1. If the step's record has landed ($i < \text{ready}$), run it — no fetch, no wait at all.
2. If it is in flight, issue a fill up to ring capacity when nothing is outstanding, then wait — once — on the ring's NoC ID (`noc_wait_outstanding_id` + `FENCE`) and set `ready = end`: one wait covers a whole batch of posted reads, never one wait per record.
3. If the record's `window` flag is set, issue the next fill *before* running the op [FACT, src/brisc.cpp:21169-21172].

The builder paces refills by marking records: "a record with `window` set requests every record up to TT_GLM_STEP_RING - 1 past itself before its op (so at least its TT_GLM_STEP_WINDOW records land ahead of its prefetch)" [FACT, src/constants.h:2388-2389]. At list start, one full-capacity fill covers the first 32 records [FACT, src/brisc.cpp:21160-21163]; mid-list, only window-flagged records trigger refills, so a windowed record placed where the ring is about to drain restarts the read-ahead just in time [INFER, src/brisc.cpp:21169-21172].

### Why a restart exists: burst protection, not latency hiding

The ring's latency-hiding is already handled by the steady state; the `window` restart exists for a different reason. A prefetch burst queues every later small DRAM read behind it on the channel — the roofline measured the first KDA prefetch (~15 MB per chip, issued at once by 96 helpers) delaying lead 0's 16-byte cycles write-back by ~25 us, the next step's record read by ~18 us [MEAS, notes/glm-decode-roofline.md (line 240)]. The learned rule: "Before posting a burst, land or move to L1 every small read the steps up to its consumer need: here the step window restarts at the posting step" [MEAS, notes/glm-decode-roofline.md (lines 242-244)]. So a step whose op will post a large DRAM prefetch first pushes the *future record reads* ahead of its own burst [INFER, src/brisc.cpp:21169-21171].

The measurements around that rule quantify both sides. Dropping the restart at the KDA step cost +23 us — "its burst queues the KDA's own later reads" [MEAS, notes/glm-decode-roofline.md (lines 311-312)]. The gate step's restart row measured "not kept" — its deltas (-83, -81 / -86, -51) sit within noise of the kept variant's (-84 / -90, -70) [MEAS, notes/glm-decode-roofline.md (lines 304-305)]; read strictly, that row shows the gate step's restart was neither clearly winning nor clearly losing, so the build kept the simpler list [INFER, notes/glm-decode-roofline.md (lines 304-305)].

## The measured wins

The ring was not born whole — the roofline's step-bookkeeping table measures each piece on silicon (median device us/token, plan0h / hot3-meas) [MEAS, notes/glm-decode-roofline.md §8 (lines 333-340)]:

| Change | plan0h | hot3-meas | Step |
| --- | ---: | ---: | ---: |
| `glm-hd` base | 6149, 6158 | 6774 | — [MEAS, notes/glm-decode-roofline.md §8 (line 335)] |
| prefetch jobs' args staged clear of the next step (race fix) | 6176, 6180 | 6804, 6798 | +25 [MEAS, notes/glm-decode-roofline.md §8 (line 336)] |
| record ring read ahead at window restarts, each step copies only its own arg words | 6005, 6005 | 6616, 6617 | $-76, -84$ [MEAS, notes/glm-decode-roofline.md §8 (line 339)] |
| 4-word copies, record control words loaded before the first store | 5978, 5987 | 6598, 6603 | $-23, -18$ [MEAS, notes/glm-decode-roofline.md §8 (line 340)] |

Window-restart read-ahead alone bought 76-84 us/token, and the arg-copy discipline another 18-23 — with the L1 effect spelled out: with the data cache off, a store between loads makes each later load wait out the full L1 latency, about 10 cycles a word when loads and stores alternate [MEAS, notes/glm-decode-roofline.md §8 (lines 341-343)]. The phase-2 cut set that carried these changes — "no cycles write-back, step window, KDA prefetch, inputs fused, prefetch split" — took record fetches from 191 to 32 us per token [MEAS, notes/glm-decode-roofline.md (lines 233, 236)]. Against the pre-fix baseline, the per-component table's fetch column totaled 0.191 ms at its 9.124 ms base [MEAS, notes/glm-decode-roofline.md (line 99)] — the same story per layer kind.

## Does fetch ever gate compute?

The measured residual: record fetch and copy costs 0.34 us per step [MEAS, notes/glm-decode-roofline.md (line 77)]. Against the baseline token — 561 steps in 9.30 ms [MEAS, notes/glm-decode-roofline.md (lines 12, 60)] — the average step is

$$\frac{9300\ \mu\text{s}}{561} \approx 16.6\ \mu\text{s}, \qquad \frac{0.34}{16.6} \approx 2\%$$

so the fetch+copy bookkeeping is ~2% of a step, and the *wait* for it is less still whenever the read-ahead has done its job [INFER, notes/glm-decode-roofline.md (lines 12, 60, 77)]. The only way the loop stalls on records is if compute catches the in-flight frontier — the wait at [src/brisc.cpp:21164](../../../regen-glm5.3-flash/src/brisc.cpp) fires only then, and by then the next block has been in flight for up to 31 steps' worth of compute [INFER, src/brisc.cpp:21160-21167]. One deliberate limit: routed-expert bytes behind a router step cannot be read before the routing is known; the demand-signal analysis is [Ch4 `01-roofline-and-byte-budget.md`](../ch4_dram_streaming_discipline/01-roofline-and-byte-budget.md)'s [INFER, notes/glm-decode-roofline.md].

## The same family, one pointer

Three prefetch mechanisms in this chapter's scope share one shape — claim ahead, let reads live in flight, drain only at a horizon [INFER, src/brisc.cpp:21044-21178]: the record ring (this file), the `TtTeamJob.next` hook ([`03-lead-helper-teams.md`](./03-lead-helper-teams.md)), and the record's own `prefetch` descriptor, whose staged args "neither the next step's args nor the list's result overwrite" [FACT, src/constants.h:2429-2432] ([`02-interpreter-loop.md`](./02-interpreter-loop.md)). The word "ring" also names two structures that are NOT this one: the ERISC send/receive rings of the message service ([Ch5 `02-erisc-message-service.md`](../ch5_cross_chip_transports/02-erisc-message-service.md)) and the trace-event ring ([`02-interpreter-loop.md`](./02-interpreter-loop.md)).

---

← [`03-lead-helper-teams.md`](./03-lead-helper-teams.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md)
