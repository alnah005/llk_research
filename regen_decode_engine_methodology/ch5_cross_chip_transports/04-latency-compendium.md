# The Transport Latency Compendium

Every measured cross-chip transport number in the corpus, in ONE table: what was measured, its value, and the host it was measured on — because transport latencies are host-specific and a row lifted out of its host context is wrong. This is the chapter's single numbers table; the mechanism files cite into it rather than carrying their own registries, other chapters cite this file when they need a transport number, and the port's re-measure checklist draws from here.

**Scope:** covers Question 4, part: the measured latency/bandwidth evidence for both transports across the hosts the corpus covers. DRAM throughput and prefetch-window rates are NOT here — they are [Ch4 — DRAM Streaming Discipline](../ch4_dram_streaming_discipline/index.md)'s; the batch/MTP economics are [Ch4 `05`](../ch4_dram_streaming_discipline/05-batch-and-mtp-economics.md)'s. Primary evidence: [notes/glm-ccl.md](../../../regen-glm5.3-flash/notes/glm-ccl.md), [notes/glm-eth.md](../../../regen-glm5.3-flash/notes/glm-eth.md), [notes/glm-glx-ethdec.md](../../../regen-glm5.3-flash/notes/glm-glx-ethdec.md), [notes/glm-glx-layouts.md](../../../regen-glm5.3-flash/notes/glm-glx-layouts.md). Every row's line anchor was re-verified in-tree before writing; values are quoted verbatim, never re-rounded.

## Host key

| Tag | Host | Fabric facts that set its latency class |
| --- | --- | --- |
| a10u28 | LoudBox-class box: 8 × p150b, one Gen5 **x16** root port each, 2 × EPYC 9124 (chips 0-3 / 4-7 split across the two sockets), AICLK 1.35 GHz (1 us = 1350 cycles); Ethernet links an 8-ring + 2 chords, *not* a 3-cube | [MEAS, notes/glm-ccl.md (lines 3-19)] |
| box C | LoudBox 3-cube box (the ccl-branch message-service bench host); a08u28 is the same class [MEAS, notes/glm-glx-ethdec.md (line 154)] | [MEAS, notes/glm-eth.md (lines 9-11)] |
| b08u02 | Galaxy bh-glx-120: 32 chips in cube groups, **x1** PCIe links (cube A bench host) | [MEAS, notes/glm-glx-ethdec.md (lines 3-4)] |
| c01u20 | Galaxy bh-glx-120 at a 75 W limit (layouts host) — a different Galaxy host under a power limit | [MEAS, notes/glm-glx-layouts.md (lines 7-8)] |

Round sizes are per chip unless noted. Rows 1-5 are chip-local primitives, included because every transport decomposition (the mesh barrier, the round phase stamps) is written in their terms; they are not themselves cross-chip numbers.

## The table

| # | Class | Measured | Host | Value | Citation |
| ---: | --- | --- | --- | --- | --- |
| 1 | chip-local context | NoC write issue, steady state, FIFO off / FIFO on | a10u28 | 0.022 / 0.011 us (30.3 / 15.3 cycles) | [MEAS, notes/glm-ccl.md (lines 30, 32)] |
| 2 | chip-local context | Acked 16 B NoC write, lead to lead, any distance / 16 B read | a10u28 | 0.13 / 0.20 us (175 / 267 cycles) | [MEAS, notes/glm-ccl.md (lines 33-34)] |
| 3 | chip-local context | Chip sync (`allgather_barrier`, 8 leads) | a10u28 | 0.30 us (402 cycles) | [MEAS, notes/glm-ccl.md (line 35)] |
| 4 | chip-local context | Per-step bookkeeping (256 B record read + 48-word copy + acked write-back) | a10u28 | 0.81 us (1097 cycles) | [MEAS, notes/glm-ccl.md (line 44)] |
| 5 | PCIe, point | **Peer write, one way, same socket / cross socket (relaxed = strict window)** | a10u28 | **0.50 / 0.55 us** | [MEAS, notes/glm-ccl.md (line 36)] |
| 6 | PCIe, point | Same write from every lead of the PCIe-aligned column (placement does not change latency) | a10u28 | 0.501 / 0.551 us | [MEAS, notes/glm-ccl.md (line 211)] |
| 7 | PCIe, point | Peer read, 16 B, same / cross socket | a10u28 | 0.96 / 1.06 us | [MEAS, notes/glm-ccl.md (line 37)] |
| 8 | PCIe, point | Inbound writes to one tile serialize at (the destination-tile limit) | a10u28 | ~0.13 us each | [MEAS, notes/glm-ccl.md (lines 74-78)] |
| 9 | PCIe, collective | 8-chip all-to-all of 16 B lines, one root to/from 7 roots | a10u28 | 1.46 us | [MEAS, notes/glm-ccl.md (line 38)] |
| 10 | PCIe, collective | Same, each line landing on a different lead of the receiver | a10u28 | 1.02 us | [MEAS, notes/glm-ccl.md (line 39)] |
| 11 | PCIe, collective | All-to-all by shift count k = 1…7, root to root, per repeat | a10u28 | 0.59 → 1.46 us | [MEAS, notes/glm-ccl.md (line 64)] |
| 12 | PCIe, barrier | Mesh barrier, lead p writes chip + p + 1's lead column (multicast window) | a10u28 | 1.74 us | [MEAS, notes/glm-ccl.md (line 42)] |
| 13 | PCIe, barrier | Mesh barrier, root writes 7 peer roots, multicasts a release | a10u28 | 1.77 us | [MEAS, notes/glm-ccl.md (line 41)] |
| 14 | PCIe, round | `glm_round`, 1 contributor, 256 B / 1 KB / 4 KB per chip | a10u28 | 2.15 / 2.26 / 2.93 us | [MEAS, notes/glm-ccl.md (line 43)] |
| 15 | PCIe, round (removed) | Old step mesh sync (barrier + courier exchange + PCIe-read fence), bench / in situ | a10u28 | 4.51 us (6089 cycles) / 4.4-5.4 us | [MEAS, notes/glm-ccl.md (lines 40, 92-96)] |
| 16 | PCIe, round, in situ | Whole round chain per token (235 dense / 257 index rounds), critical path | a10u28, real-weight chat | 506 / 550 us (+561 chip syncs = 0.23 ms; mean wait 680 us) | [MEAS, notes/glm-ccl.md (lines 222-224)] |
| 17 | PCIe, round, in situ | Per kind, critical us/round: MoE / FFN site / attention site / KDA / MLA gathers | a10u28 | 2.52 / 2.12 / 1.80 / 2.18 / 1.4-2.1 us | [MEAS, notes/glm-ccl.md (lines 226-232)] |
| 18 | PCIe, round, in situ | Round with delivery, 16 B, lockstep | a10u28 | ~1.85 us | [MEAS, notes/glm-ccl.md (line 234)] |
| 19 | PCIe, protocol change | Step mesh syncs → round-count RX parity (06b6c9d), real-weight chat | box C class | saves 0.82-0.88 ms per token, identical md5s | [MEAS, notes/glm-ccl.md (lines 160-171)] |
| 20 | PCIe, round | `glm_round` back to back, 192 B → 4608 B; GLM's chain of 257 rounds | box C | 2.30 / 2.34 / 2.38 / 2.52 / 2.99 / 3.10 us; 0.73 ms per token | [MEAS, notes/glm-eth.md (lines 318, 393-394)] |
| 21 | PCIe, gather | `GLM_GATHER` fit / rate sent per chip | a10u28 | ~5 us + 0.33 us per KB = 21-23 GB/s | [MEAS, notes/glm-ccl.md (lines 357-361)] |
| 22 | PCIe, gather | B=32 device vs host-copied cmd 146: logits / routed hidden / shared hidden | a10u28 | 7.2 vs 7.8 / 49.5 vs 191.5 (3.9x) / 8.3 vs 20.7 us | [MEAS, notes/glm-ccl.md (lines 357-358)] |
| 23 | ERISC, link | **Raw link, one way, as first reported (16 B-1 KB)** — see reading note A | a10u28 (ccl bench) | 0.67 us | [MEAS, notes/glm-ccl.md (line 45)] |
| 24 | ERISC, link | ERISC-to-ERISC ping, one way as reported, 16 B / 1 KB / 4 KB / 16 KB / 64 KB | a10u28 | 0.672 / 0.676 / 0.749 / 1.004 / 2.000 us | [MEAS, notes/glm-ccl.md (line 126)] |
| 25 | ERISC, link | **True one-way, traced (the row-23/24 figure was this path's round trip less the turnaround, halved)** | box C | **0.39 us** | [MEAS, notes/glm-eth.md (lines 314, 408-412)] |
| 26 | ERISC, link | In-board / cabled link, one way | Galaxy (dsv41) | 0.34 / 0.39 us (~+0.05 us per hop) | [MEAS, notes/glm-eth.md (lines 410-412, 444)] |
| 27 | ERISC, link | ERISC stream, one link, 4 KB+ transfers | a10u28 | 49.15 GB/s | [MEAS, notes/glm-ccl.md (line 127)] |
| 28 | ERISC, link | Tensix root to its own ERISC1 and back, 16 B | a10u28 | 0.61 us round trip | [MEAS, notes/glm-ccl.md (line 128)] |
| 29 | ERISC, forwarder | Tensix to Tensix through ERISC forwarders, 16 B, old / lean | a10u28 | 1.84 / 1.77 us one way | [MEAS, notes/glm-ccl.md (lines 129-130)] |
| 30 | ERISC, msg service | Lead L1 to lead L1, one hop, 16 B / 1 KB / 4 KB / 8 KB | box C | 0.95-1.02 / 1.05 / 1.14 / 1.41 us | [MEAS, notes/glm-eth.md (line 313)] |
| 31 | ERISC, msg service | Each relayed hop (2 hops 1.68 / 3 hops 2.34 at 16 B) | box C | +0.66 us | [MEAS, notes/glm-eth.md (line 315)] |
| 32 | ERISC, msg service | Relay between sibling ERISCs, uncontended / median in a round | box C | 0.30-0.35 / ~0.43 us | [MEAS, notes/glm-eth.md (lines 416-418)] |
| 33 | ERISC, msg service | 8-chip all-gather round, root-triggered, 192 → 4608 B per chip (p99 3.07-3.57) | box C | 2.77 → 3.31 us | [MEAS, notes/glm-eth.md (line 316)] |
| 34 | ERISC, msg service | Same rounds back to back | box C | 2.80 → 3.27 us | [MEAS, notes/glm-eth.md (line 317)] |
| 35 | ERISC, msg service | Fit of the triggered round | box C | 2.73 us + 0.130 us per KB | [MEAS, notes/glm-eth.md (line 319)] |
| 36 | ERISC, msg service | One link per neighbour pair instead of two, 192 / 2048 / 4608 B | box C | 3.75 / 3.73 / 3.74 us (+0.5 to +0.95; slowest chip's p99 doubles to 7.1-8.8) | [MEAS, notes/glm-eth.md (lines 320, 397-399)] |
| 37 | ERISC, round | Trees, first version (three `eth_msg_post`, plain order) | b08u02 cube A | 2.92 → 3.60 us | [MEAS, notes/glm-glx-ethdec.md (line 57)] |
| 38 | ERISC, round | **Trees, lean tree post, back to back, 192 → 4608 B** | b08u02 cube A | **2.84 / 2.88 / 2.91 / 3.07 / 3.50 / 3.53 us** | [MEAS, notes/glm-glx-ethdec.md (line 58)] |
| 39 | ERISC, round | Trees, lean tree post + balanced order | b08u02 cube A | 2.80 → 3.48 us | [MEAS, notes/glm-glx-ethdec.md (line 59)] |
| 40 | ERISC, round | Lean post: round start to three-posts-ended | b08u02 cube A | 0.33 us (vs 0.66 first version) | [MEAS, notes/glm-glx-ethdec.md (line 61)] |
| 41 | ERISC, round, in situ | Trees, balanced order, far chip over PCIe under 1 KB, by decode size bucket (0-511 B → 4.6 KB) | b08u02 cube A | 2.84 / 2.70 / 3.61 / 4.39 / 3.73 / 4.34 us | [MEAS, notes/glm-glx-ethdec.md (line 68)] |
| 42 | PCIe, round, in situ | LoudBox PCIe rounds, same buckets, same decode | LoudBox | 2.20 / 2.08 / 2.78 / 2.69 / 2.85 / 3.51 us | [MEAS, notes/glm-glx-ethdec.md (line 69)] |
| 43 | ERISC, round, in situ | Rounds per token in situ | c01u20 cube A | 257 per token, 3.4-4.5 us each; 3 KB rounds reach 6-7.5 us | [MEAS, notes/glm-glx-layouts.md (line 96)] |
| 44 | ERISC, round | Round tail after last arrival (auto rounds, 0.5-4.6 KB; release is 0.21-0.33 of it) | b08u02 cube A | 0.38-0.50 us | [MEAS, notes/glm-glx-ethdec.md (lines 176-192)] |
| 45 | ERISC, far-chip | Far slice over PCIe for rounds under 1 KB | b08u02 | 0.4-0.5 us shorter; up to 3 KB no better | [MEAS, notes/glm-glx-ethdec.md (lines 70-72); FACT, src/tt_backend.cpp:5965-5970] |
| 46 | ERISC, handoff | Lead-to-lead ping, 16 B: 1 hop (in-tray / tray) / 2 hops / 3 hops | c01u20 cubes | 1.020 / 1.087 / 1.745 / 2.406 us | [MEAS, notes/glm-glx-layouts.md (lines 132-137)] |
| 47 | ERISC, handoff | Links between groups vs inside one | c01u20 | agree to within 0.01 us | [MEAS, notes/glm-glx-layouts.md (line 140)] |
| 48 | ERISC, handoff | Ring-PP4 stage hop, 4 KB slice: hop / wake | c01u20 | 2.1-2.2 us measured + ~1 us wake (the wake is modelled in the note) ≈ 3.2 us; 45 hops = 0.14 ms per token | [MEAS, notes/glm-glx-layouts.md (lines 146-148); EST, notes/glm-glx-layouts.md (line 150)] |
| 49 | ERISC, EP paths | Dispatch 8 KB at 2 / 3 hops (both links); return 1 KB × 2 | c01u20 | 2.39 / 3.37 us; 1.95-2.02 us (round trip 4.78 us at 2 hops) | [MEAS, notes/glm-glx-layouts.md (lines 209-214)] |
| 50 | Transport, engine | Decode t/s by round transport, cube A alone | b08u02 | PCIe 99.4 t/s; eth or auto 142.3 t/s; four groups at once 141.7 / 139.8 / 142.4 / 140.3 | [MEAS, notes/glm-glx-ethdec.md (line 12)] |
| 51 | Transport, engine | LoudBox decode t/s, rounds knob unset / with `eth` on its own cube | LoudBox | 165.0 / 161-162 t/s | [MEAS, notes/glm-glx-ethdec.md (line 13)] |
| 52 | Transport, engine | Galaxy minus LoudBox token, round-transport share | b08u02 vs LoudBox | +224 us of a +943 us total (+238 GDDR and ~+480 harvest skew are not transport facts — the DRAM-rate cause is Ch4's) | [MEAS, notes/glm-glx-ethdec.md (lines 14-23)] |
| 53 | Bandwidth, link | ERISC stream, one link | a10u28 | 49.15 GB/s | [MEAS, notes/glm-ccl.md (line 127)] |
| 54 | Bandwidth, link | All-gather per link each way (ERISC loop bound; 205 GB/s per chip with no DRAM) | box C | ~34 GB/s | [MEAS, notes/glm-eth.md (lines 15-16, 152-158)] |
| 55 | Bandwidth, per chip | Ethernet all-gather, DRAM to DRAM, 6 feeders / without feeders | box C | 195 GB/s sent per chip, bit-exact / 97.7 GB/s | [MEAS, notes/glm-eth.md (lines 11, 16)] |
| 56 | Bandwidth, per chip | Smallest gather (512 B per chip), lead trigger to all landed / fixed cost per gather | box C | 8.0 us / ~7 us | [MEAS, notes/glm-eth.md (lines 13-14, 138)] |
| 57 | Bandwidth, per chip | All-gather re-measured on the base commit, 4 MB/channel; Ethernet channel-slice / stride layouts | box C class | 172.8 GB/s per chip; 81-86 / 171 GB/s per chip | [MEAS, notes/glm-eth.md (lines 401-403); notes/glm-ccl.md (lines 558, 623)] |
| 58 | Fit | PCIe gather fit vs Ethernet gather fit (with feeders) | a10u28 / a08u28 | 5 us + 0.33 us/KB vs 15 us + 0.037 us/KB | [MEAS, notes/glm-ccl.md (lines 397-398, 559)] |
| 59 | Crossover | Gather crossover, PCIe vs Ethernet (no feeders in parens) | LoudBox class | ~34 KB per chip (~7 KB without feeders) | [MEAS, notes/glm-ccl.md (lines 399-405, 561)] |
| 60 | Crossover | Round crossover on x1 links: Ethernet wins from | Gen4 x1 / Gen5 x1 | 428 B / 764 B per chip; by 5x-10x at 4 KB | [MEAS, notes/glm-eth.md (lines 395-396)] |
| 61 | Topology penalty | 2x4 ladder (a Galaxy PCIe bus group) needs a 4th hop | Galaxy | ~+0.65 us per round | [MEAS, notes/glm-eth.md (lines 435-437)] |

## Reading notes — the rows that commonly get misquoted

- **(A) Rows 23-25 are the same link measured two ways.** The 0.67/0.672 us "one way" of `glm-ccl.md` was later traced and shown to be "this round trip less the turnaround, so it was a round trip"; the traced one-way link time is 0.39 us [MEAS, notes/glm-eth.md (lines 408-412)]. Both rows stay because both were used in decisions — the LoudBox verdict against Ethernet for decode was drawn against the 0.67 reading and holds against 0.39 too, since a whole PCIe peer write costs 0.50 us and an 8-chip gather needs 3 hops on the cube [MEAS, notes/glm-ccl.md (lines 116-118)]. **A porter re-measuring must state which figure they mean.**
- **(B) Rows 38 and 41 are different measurements — never conflate them.** Row 38 is a back-to-back *bench* row (the guide plan's compressed "2.84-3.5" loses the 3.53 endpoint; the row keeps the note's own six points); row 41 is the *in-decode* mean per size bucket on the least-waiting chip (2.84 at 0-511 B through 4.34 at 4.6 KB); row 43 is a *different Galaxy host under a power limit* — a class, not a contradiction of row 41. Rows 41-42 are the only like-for-like in-situ comparison between the transports in the corpus.
- **(C) Row 15 is a deleted mechanism.** The step mesh sync no longer runs in decode — the round-count parity replaced it [MEAS, notes/glm-ccl.md (lines 149-158)]; the row stays because it is the measured reason the removal was worth row 19's 0.82-0.88 ms.
- **(D) Rows 50-52 are engine-level consequences, not transport primitives.** The per-token decomposition (row 52) fences the +238 us GDDR row to [Ch4](../ch4_dram_streaming_discipline/index.md) by name — the DRAM-rate exposition and its numbers are deliberately not restated here. And row 50's savings did not by themselves move t/s, "because the round waits on the slow chips absorb them" [MEAS, notes/glm-glx-ethdec.md (line 71)].
- **(E) No LoudBox row applies to a Galaxy host, and no Galaxy row applies to a LoudBox.** The selection consequences are [`03-rounds-and-transport-selection.md`](./03-rounds-and-transport-selection.md)'s.

## Estimates recorded in the corpus (fenced)

The cross-box (four cabled LoudBoxes) exchange latencies below are estimates assembled from measured parts — "None of this is measured end to end" [EST, notes/glm-eth.md (lines 272-283)]:

| Exchange | Estimate | Citation |
| --- | ---: | --- |
| Dispatch, 8 KB to each peer box, one way | ~1.9 us | [EST, notes/glm-eth.md (lines 272-275)] |
| Return, 16 KB, one way | ~2.1 us | [EST, notes/glm-eth.md (lines 272-275)] |
| Relayed hop to the diagonal box | +0.9-1.2 us | [EST, notes/glm-eth.md (line 277)] |
| Whole exchange, all-direct vs one relay | ~2 us vs 3-3.5 us | [EST, notes/glm-eth.md (lines 278-281)] |

Across hosts over PCIe or the host network: "tens of microseconds per exchange: not viable" [MEAS, notes/glm-eth.md (line 292)].

## What the port must re-measure (pointer only)

Nothing in this table transfers: the same/cross-socket split is a property of the host's NUMA topology, the x16-vs-x1 crossover is a property of the PCIe link widths `mesh_pcie_link_width()` reads from sysfs, the ERISC rows are a property of the fabric's cube-ness and cabling, and even the LoudBox-vs-Galaxy class split was decided by exactly these measurements [FACT, src/tt_backend.cpp:5912-5963]. The re-measure rule ("a port never inherits regen's registry values") and the checklist that applies it — PCIe per socket topology, ERISC per fabric, DRAM prefetch-window rate, the t/s ladder, smoke-gate results — are owned by [Ch8 `04-port-gates-and-re-measure.md`](../ch8_tt_metal_port/04-port-gates-and-re-measure.md); this table is that checklist's input, never its output. regen's own benches are the template: `TT_LAB_BENCH=sync` for the PCIe rows, `TT_LAB_BENCH=ethmsg` for the message-service rows [FACT, notes/glm-ccl.md (lines 5-6); notes/glm-eth.md (lines 306-310)].

---

← [`03-rounds-and-transport-selection.md`](./03-rounds-and-transport-selection.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [Chapter 6 — Host Driver, Weights Pipeline, and the CPU Oracle](../ch6_host_driver_weights_oracle/index.md)
