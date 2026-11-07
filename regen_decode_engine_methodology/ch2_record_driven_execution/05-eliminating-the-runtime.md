# Eliminating the Runtime

The chapter's thesis, argued as one claim: because the model's control flow is records and the interpreter is resident, decode has no phase boundaries, nothing is dispatched per step, and the host is outside the token loop — and the measured roofline confirms it, because the token's cost decomposition contains no dispatch-overhead term to remove. This file states each elimination against its source anchor, prices the residual, marks the exact boundary of host involvement per token (including its one deliberate, record-encoded exception), contrasts the cost structure with a dispatch-per-step engine, and hands the stays/comes question to Chapter 8.

**Scope:** covers Question 1's elimination argument — the "how does it eliminate phase boundaries, per-step dispatch, and the host from the token loop" clause, resolved. The mechanisms were established in [`01`](./01-step-records.md)-[`04`](./04-read-ahead-rings.md); production serving and the record builder are [Ch6 `04-record-builder-and-serving.md`](../ch6_host_driver_weights_oracle/04-record-builder-and-serving.md)'s; the roofline's throughput exposition and the optimization ledger are [Ch4](../ch4_dram_streaming_discipline/index.md)'s.

## Claim (a): no phase boundaries

Records chain continuously, inside one command:

- **Between steps**, the only structure is the record's own `sync` field — `TT_GLM_SYNC_MET` (none) or `TT_GLM_SYNC_CHIP` (one chip-leads barrier) [FACT, src/constants.h:2412-2413] — and mesh syncs are themselves record-encoded steps of the list [INFER, src/brisc.cpp]. There is no init pass, no "forward phase" marker, and no boundary the host can observe between steps [INFER, src/brisc.cpp:21153-21179].
- **Between tokens**, `glm_steps` iterates the token count with a plain device-side loop (the `glm_steps` section of [`02-interpreter-loop.md`](./02-interpreter-loop.md)), and the records re-parameterize themselves for each running token ([`01-step-records.md`](./01-step-records.md)). The state that token $t$'s last step left in L1 — streams, KDA and MLA histories, pending mixes — is exactly what token $t+1$'s first record reads [FACT, src/constants.h:3332-3343]. The only first-position special cases live host-side (e.g. resetting MLA index state at position 0) [FACT, src/tt_backend.cpp:31280-31284] — data setup, not a device phase [INFER, src/tt_backend.cpp:31280-31284].
- **Between commands**, the one true boundary is the closing barrier-and-exchange of `brisc_main`'s command loop [FACT, src/brisc.cpp:23647-23663] — once per posted command, not per step or per token: a `GLM_STEPS` post of $n$ tokens crosses it once [INFER, src/brisc.cpp:23633-23663].

The proof that sync is data, and that the flat chain is real, is that regen *edited synchronization out of the running system by rewriting lists*: the builder classifies commands that already leave the chip's leads met (`glm_command_meets`) and drops trailing barriers (`glm_skip_barrier`) with no device-code change [FACT, src/tt_backend.cpp:23296-23310, 23337-23343] — and those list edits measured $-64$/$-71$ and $-33$ us/token on silicon [MEAS, notes/glm-decode-roofline.md §8 (lines 337-338)]. A boundary you can delete with a data edit was never a boundary [INFER, src/tt_backend.cpp:23296-23343]. On [Chapter 1's token walk](../ch1_engine_at_a_glance/02-anatomy-of-a-token.md), this is why the 45 layers, the head, and the loop back to layer 0 appear as one continuous record stream [INFER, src/brisc.cpp:21153-21199].

## Claim (b): no per-step dispatch

Nothing is launched per step. The interpreter is the single resident program per core ([`02-interpreter-loop.md`](./02-interpreter-loop.md)); a step's "dispatch" is one indirect call through the command table — `if (command == TT_COMMAND_GLM_STEPS || !run_command(command))` [FACT, src/brisc.cpp:21137-21139] — plus the team post the op body itself decides to make ([`03-lead-helper-teams.md`](./03-lead-helper-teams.md)). What dispatch remains is measured on silicon:

| Per-step device cost | Measured |
| --- | ---: |
| Record fetch and copy | 0.34 us [MEAS, notes/glm-decode-roofline.md (line 77)] |
| Acknowledged cycles write-back (with the fetch, "step bookkeeping") | ~0.45 ms per 561-step token [MEAS, notes/glm-decode-roofline.md lines 77-78] |
| An empty team post (an mHC site's no-op job) | 0.6 us [MEAS, notes/glm-decode-roofline.md lines 119-120] |
| Chip barrier closing a step | 0.42 us [MEAS, notes/glm-decode-roofline.md (line 79)] |

Dispatch did not become free — it became *device-side bookkeeping*, priced in the table above and folded into the roofline's serial term [INFER, notes/glm-decode-roofline.md].

## Claim (c): no host in the token loop

The host's involvement is bounded to the token's **edges**, and the interior of the loop runs device-alone [INFER, src/brisc.cpp:21153-21199]:

1. **Write the token's data — the data writers.** `glm_write_streams` — "Writes each chip's slice of the streams [4][4096] to its leads' resident slice" [FACT, src/tt_backend.cpp:24398-24405] — or `glm_write_embedding` — "Writes each chip's slice of the token's embedding [4096] to its leads' first resident stream" [FACT, src/tt_backend.cpp:24407-24414], e.g. at the per-position check loop [FACT, src/tt_backend.cpp:29708-29714].
2. **Post at most one command — the command post.** `glm_post_steps` — "Posts GLM_STEPS `args` for the list. A failure names each lead's running step." [FACT, src/tt_backend.cpp:24429-24441] — and a single post carries `tokens` positions: `glm_run_rows` "Runs the list for `tokens` positions from `position` with one GLM_STEPS command" [FACT, src/tt_backend.cpp:24447-24452], so even the per-token post is a host policy choice, not a mechanism requirement [INFER, src/tt_backend.cpp:24447-24452].
3. **Read the result.** The argmax candidate comes back "from chip 0's first lead" [FACT, src/tt_backend.cpp:24003-24005]; the streams return via `glm_read_streams` [FACT, src/tt_backend.cpp:24418-24423].

The two anchors are deliberately distinct: the *writers* (`glm_write_streams` / `glm_write_embedding`) feed the device's L1; the *post* (`glm_post_steps`) is the command. In the single-list check path the two sit back to back per position — write the streams, then run one post [FACT, src/tt_backend.cpp:29708-29714] — but they are separate mechanisms: the writer is data placement, the post is one command for the whole list. Conflating them is the classic misreading of this boundary [INFER, src/tt_backend.cpp:24398-24452].

The lists themselves were built and placed once per configuration — `glm_token_lists` builds the dense/index lists in the stepper's constructor [FACT, src/tt_backend.cpp:31124-31126, 31204], and `glm_place_steps` replicates them per channel by the placement contract [`01-step-records.md`](./01-step-records.md) owns — not per token (the builder is [Ch6 `04`](../ch6_host_driver_weights_oracle/04-record-builder-and-serving.md)'s). The generation path shows the per-position shape: `glm_lone_picks` — "one GLM_STEPS post per position" — embedding row, write, post, read [FACT, src/tt_backend.cpp:31589-31599].

**Two device-side facts keep even the edges thin.** The batch image's `TT_COMMAND_GLM_BATCH_EMBED` reads the token's embedding row from DRAM itself and can take its token from the chips' own argmax candidates, so even the embed/argmax edge work leaves the host [FACT, src/constants.h:3469-3478]. And the one place host decisions enter mid-post is itself a record: where sampled MTP needs host randomness, the list carries a `TT_COMMAND_GLM_HOST_SYNC` step — "where the host joins a post" — an explicit arrive/wait protocol with a timeout stop, with `resolve` running mid-post inside its wait in the `one_post` serving shape [FACT, src/constants.h:3486-3492; src/tt_backend.cpp:34008-34012]. That is the sole in-post host role, it is opt-in per record, and being a record is exactly the point: the schedule *names* its own exception [INFER, src/constants.h:3486-3492]. Sampling and MTP economics are [Ch4 `05-batch-and-mtp-economics.md`](../ch4_dram_streaming_discipline/05-batch-and-mtp-economics.md)'s and [Ch6 `04`](../ch6_host_driver_weights_oracle/04-record-builder-and-serving.md)'s.

Measured at the roofline baseline: the device token is 9.30 ms; device + host is 9.49 ms, the host edges being "embed 0.05, streams 0.09, post and argmax read 0.05" [MEAS, notes/glm-decode-roofline.md (lines 12-13)]:

$$9.49 - 9.30 = 0.19\ \text{ms} = 0.05 + 0.09 + 0.05$$

— about 2% of the token, all of it at the edges [INFER, notes/glm-decode-roofline.md (lines 12-13)]. So the claim survives its precision: no host *in* the token loop — the loop's interior is device-alone, its edges shrink to data writes plus one command post, and the single mid-post join that exists is itself a record [INFER, src/brisc.cpp:21153-21199].

## The roofline confirmation

The roofline's diagnosis of the baseline token:

> "Decode is bound by serial work per step, with DRAM idle, not by DRAM bandwidth or by the CCL" [MEAS, notes/glm-decode-roofline.md (line 173)]

and its three-term decomposition of the 9.3 ms token [MEAS, notes/glm-decode-roofline.md (lines 177-179)]:

| Term | ms | Share |
| --- | ---: | ---: |
| DRAM streaming (447 GB/s inside windows) | 3.94 | 42% |
| Lead serial work, few-core stages, team dispatch and team skew | 3.48 | 38% |
| Syncs: mesh 0.96, bookkeeping ~0.45, op rounds 0.31, chip barriers 0.23 | ~1.95 | 21% |

The connection to this chapter is an inference, so it is tagged as one: **none of the three terms is a host-dispatch term.** In a launch-per-op engine the serial-work row would contain host round trips and launch overheads; here the serial row is lead-side compute, the dispatch that does appear is the sub-microsecond device-side kind priced above, and the host exists only in the 0.19 ms edge [INFER, notes/glm-decode-roofline.md (lines 173-179)]. That absence is not an accident of measurement — it is the structural consequence of claims (a)-(c): the cost class a dispatch-per-step runtime would add per step has no machinery left to charge [INFER, notes/glm-decode-roofline.md]. The runtime was removed *before* the profiling began, so there is nothing left whose removal the ledger could credit; what the ledger did fight — the serial and sync terms — is [Ch4 `04-optimization-ledger.md`](../ch4_dram_streaming_discipline/04-optimization-ledger.md)'s story, not a dispatch story [INFER, notes/glm-decode-roofline.md].

## Cost-structure contrast

| Per-token item | Dispatch-per-step engine (assumed background) | regen interpreter |
| --- | --- | --- |
| Op launches | ~hundreds (one per op) | 0 |
| Per-step host cost | Round trip + launch per op | None — host only at token edges |
| Per-step device cost | Runtime bookkeeping behind each launch | 0.34 us fetch/copy + arg rewrites [MEAS, notes/glm-decode-roofline.md (line 77)] |
| Grid formation | Per-op runtime dispatch | Load-time teams; one posted word per helper [FACT, src/brisc.cpp:3337-3348] |
| Phase structure | Init/forward/completion phases | One continuous record stream per post [FACT, src/brisc.cpp:21189-21190] |
| Host synchronization | barrier/trace waits per op | 1 sentinel poll per command [FACT, src/tt_backend.cpp:6420-6440] |
| Trace mitigation | Capture then replay | The record list IS the trace ([`02-interpreter-loop.md`](./02-interpreter-loop.md)) |

The conventional column is the audience's tt-metal background as this guide assumes it, not a claim about regen [INFER, src/brisc.cpp].

## What this hands to Chapter 8

The eliminations are the load-bearing core of the port question: records-as-data for dispatch elimination is a candidate "stay" only insofar as the roofline argument survives without a per-step host term — a port that reintroduces a dispatch term changes the shape of the budget, not just its constants [INFER, notes/glm-decode-roofline.md]. The *invariants* — no per-step launch, no host sync in the loop's interior, state resident across tokens — must survive under any harness; the *mechanisms* that protect them here (sentinel posts, GO word writes, the ring's reserved NoC ID) are candidates for re-derivation, not verbatim porting [INFER, src/brisc.cpp]. The audit that applies that test, decision by decision, is [Ch8 `02-load-bearing-vs-incidental.md`](../ch8_tt_metal_port/02-load-bearing-vs-incidental.md)'s; this chapter supplies its premises without arguing port decisions [INFER, src/brisc.cpp:21108-21199].

---

← [`04-read-ahead-rings.md`](./04-read-ahead-rings.md) | ↑ [`index.md`](./index.md) | ↑ [guide root](../index.md)

**Next:** [Chapter 3 — Raw-MMIO Compute and SFPU Replay](../ch3_raw_mmio_sfpu_replay/index.md)
