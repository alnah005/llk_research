# Rounds and Transport Selection

Every collective the token walk triggers becomes one call — `glm_round` — and that one call can carry the slice over either transport, per round, at run time: PCIe peer-BAR writes, ERISC message-service trees, or a mix. The choice is not free-form and not encoded in the records: the host resolves a rule from the box's PCIe link width and cube topology into a byte threshold, plants it in each first lead's `TtEthMsgLead` at image start, and the lead applies it per round. This file walks the round machinery, states the measured per-host round costs, gives the selection rule, and shows how `topology.*` feeds it — LoudBox and Galaxy kept distinct, because every latency in it is host-specific.

**Scope:** covers Question 4, part: how decode rounds use each transport, the in-situ ERISC round numbers, and the decision rule — which collectives ride which transport per host, how `topology.*` feeds the choice, and how the lead chooses or the record encodes it. The two transports' mechanics are [`01`](./01-pcie-peer-bar-dma.md)/[`02`](./02-erisc-message-service.md)'s; the measured-value registry is [`04-latency-compendium.md`](./04-latency-compendium.md)'s, cited by row here, never restated. The demand side — why batching and speculative decoding multiply what these transports carry — is [Ch4 `05-batch-and-mtp-economics.md`](../ch4_dram_streaming_discipline/05-batch-and-mtp-economics.md)'s, cross-referenced here, never restated. Primary evidence: [notes/glm-ccl.md](../../../regen-glm5.3-flash/notes/glm-ccl.md), [notes/glm-eth.md](../../../regen-glm5.3-flash/notes/glm-eth.md), [notes/glm-glx-ethdec.md](../../../regen-glm5.3-flash/notes/glm-glx-ethdec.md), [notes/glm-glx-layouts.md](../../../regen-glm5.3-flash/notes/glm-glx-layouts.md), [src/topology.h](../../../regen-glm5.3-flash/src/topology.h), [src/topology.cpp](../../../regen-glm5.3-flash/src/topology.cpp), [src/brisc.cpp](../../../regen-glm5.3-flash/src/brisc.cpp), [src/tt_backend.cpp](../../../regen-glm5.3-flash/src/tt_backend.cpp).

## Which steps of the walk cross chips

The round census per token — mHC site record rounds ×90 (one per site), KDA heads ×34, MoE Ra and Rb ×42 each, `GLM_ALLGATHER` ×27, plus the index-list rounds — is [Ch1 `02-anatomy-of-a-token.md`](../ch1_engine_at_a_glance/02-anatomy-of-a-token.md)'s; the slice sizes are [this chapter's index](./index.md)'s. One structural fact belongs here because it shapes transport choice: **the per-layer rounds cannot merge** — "each needs the previous one's result" — so a token pays ~235/257 serial round latencies (dense/index), and per-round latency is the whole transport question [MEAS, notes/glm-ccl.md (lines 222, 235)]. No decode round exceeds `TT_GLM_ROUND_MAX_SLICE` = 4608 B per chip (the MoE hidden gather is the largest) [FACT, src/constants.h:2983-2984] — the entire decode round population is small and latency-critical. And there is exactly one choke point: "No GLM op writes to another chip except through `glm_round`" [FACT, notes/glm-ccl.md (lines 157-158)].

## The shared round machinery

`glm_round_with` is transport-agnostic at its core [FACT, src/brisc.cpp:17007-17056]:

1. The root reads the round count at `TT_GLM_ROUND_COUNT_ADDR`, bumps it, and picks that round's RX parity — every chip runs the same rounds in the same order, so the counts agree and no mesh sync is needed between steps [FACT, src/brisc.cpp:17009-17013; MEAS, notes/glm-ccl.md (lines 149-158)].
2. It decides `eth` from the slice size: `eth = bytes >= TtEthMsgLead::eth_min` [FACT, src/brisc.cpp:17017-17018].
3. `round_exchange_tagged` runs with a `post` parameter: PCIe by default, `glm_round_post_eth` when `eth` — and the post's return mask says which chips still get the slice over PCIe, so **a round may mix transports** [FACT, src/brisc.cpp:17020-17038; notes/glm-glx-ethdec.md (lines 25-33)].
4. Delivery, the stale-line check and the release are identical either way — the arrival machinery is [`02`](./02-erisc-message-service.md)'s round-integration point [FACT, notes/glm-glx-ethdec.md (lines 24-29)].

The critical chip's phases, cycles from entering `glm_round`, show where a round's time goes [MEAS, notes/glm-ccl.md (lines 238-244)]:

| Phase | MoE round | Attention-site round |
| --- | ---: | ---: |
| Stage seq written and fenced | 164 | 162 |
| 7 PCIe sends accepted | 739 | 786 |
| Peers' slices polled and delivered | 2913 | 2012 |
| Sends and deliveries gone | 2992 | 2074 |
| Seq lines checked | 3122 | 2204 |
| Release multicast out | 3424 | 2421 |

Sends cost about 75 cycles each — the limit is "downstream of the NIU, at the PCIe tile, when the destination address changes" [MEAS, notes/glm-ccl.md (lines 249-251)]. The old step-ending mesh sync (a barrier + courier exchange + PCIe-read fence per step, 4.51 us bench / 4.4-5.4 us in situ, ~0.96 ms per token on chip 0) was deleted in favor of the count-picked parity, saving 0.82-0.88 ms per token with identical outputs (md5 `dfb95f7f` / `79dd8dfa` on both sides of interleaved runs) [MEAS, notes/glm-ccl.md (lines 40, 92-96, 149-171)] — its ledger entry is [Ch4 `04`](../ch4_dram_streaming_discipline/04-optimization-ledger.md)'s; the mechanism is this chapter's.

## Measured round cost, per host

The measured values are [`04-latency-compendium.md`](./04-latency-compendium.md)'s registry, cited by row here and never restated. **LoudBox, PCIe (a10u28, real-weight chat):** the in-situ round chain and its per-kind per-round costs are rows 16-18; the bench `glm_round` series is row 14, and box C's back-to-back series — with GLM's chain of 257 rounds per token — is row 20. **Galaxy cube, ERISC trees (b08u02, cube A):** the lean-tree-post bench series is row 38, with its first-version and balanced-order companions in rows 37 and 39 and the lean post's three-posts-ended stamp in row 40; in decode, the same trees with the far chip over PCIe under 1 KB, bucketed by size against LoudBox PCIe on the same buckets, are rows 41-42 — the corpus's only like-for-like in-situ comparison between the transports (never conflate rows 38 and 41, per 04 note B) — and a later profile on a *different* Galaxy host under a power limit is row 43, a class, not a contradiction of the buckets; the caveat that keeps the sub-4.6 KB savings from being over-read is 04 note D. **What the transport choice is worth, end to end:** same code, same model, opposite verdicts — the transport choice is a property of the host: on the Galaxy, decode on cube A on Ethernet rounds is bit-exact with PCIe rounds (md5 `383e072f`) and the t/s margin is row 50's; on the LoudBox the sign flips and the unset knob wins (row 51) [MEAS, notes/glm-glx-ethdec.md (lines 8-13)]. The Galaxy token is 943 us of device time longer than the LoudBox's (6972 vs 6029 us), and three causes account for all of it — Ethernet rounds against x16 PCIe rounds (this chapter's term), GDDR rate (a DRAM-rate cause — the exposition is [Ch4](../ch4_dram_streaming_discipline/index.md)'s), and right-half-harvest skew — in row 52's decomposition [MEAS, notes/glm-glx-ethdec.md (lines 14-23, 90)].

## The decision rule

Who chooses what, in one place:

| Decision | Owner | Mechanism | Citation |
| --- | --- | --- | --- |
| Transport *class* for GLM rounds | host, at image start | `TT_LAB_GLM_ROUND` = `pcie`/`eth`/`auto`; `auto` reads the mesh's narrowest PCIe link width from sysfs: **x1 (a Galaxy) → every round over Ethernet; otherwise (a LoudBox's x16) → all over PCIe** — "a LoudBox's x16 PCIe rounds beat its Ethernet ones at every size" | [FACT, src/tt_backend.cpp:5885-5889, 5912-5963] |
| Byte threshold | host, encoded in the lead's L1 | `eth_min` written into `TtEthMsgLead` at every GLM image start (`~0u` = none, 0 = all, or a byte count); under `auto` the width read from sysfs `current_link_width` at each device's root port — "the host side of its link, so nothing reads the chip" — picks it | [FACT, src/tt_backend.cpp:6043-6054; src/constants.h:1755-1759] |
| Far-chip exception | host → lead | `far_pcie` (default 1024 B with `auto`): slices under it reach the antipode by one PCIe write instead of a 3-hop tree copy — "rounds under 1 KB 0.4-0.5 us shorter, 2-3 KB no better" (and up to 3 KB no better than up to 1 KB) | [FACT, src/tt_backend.cpp:5965-5970; MEAS, notes/glm-glx-ethdec.md (lines 70-72)] |
| Per-round choice | **the lead, at run time** | `glm_round_with` compares the round's bytes to `eth_min`; the *record encodes the round* (bytes, seq), never the transport | [FACT, src/brisc.cpp:17017-17032] |
| Recursive doubling | host, default off | `rd_max` = 0 — RD loses to the trees ([`02`](./02-erisc-message-service.md)) | [MEAS, notes/glm-glx-ethdec.md (lines 152-171)] |
| Gather transport (batch image) | host, image-wide | `TT_LAB_GLM_GATHER` picks `pcie` or `eth`; the eth build maps in-place `GLM_GATHER`s onto `GLM_PREFILL_GATHER` and its layout rules "rule out every B=32 decode shape, which stay on PCIe" — and as built it has run only in the simulator, never on silicon | [FACT, notes/glm-ccl.md (lines 480-502, 609-614)] |

**Does the record encode the transport? No.** The record encodes the round's byte size and targets (via the op body's `glm_round` call); the transport falls out of size-meets-threshold, with the thresholds in lead L1 rewritten by the host per image [INFER, src/brisc.cpp:17007-17040; src/tt_backend.cpp:6043-6064]. The schedule is transport-agnostic data — which is exactly why "every GLM round can run over the Ethernet message service, bit-exact" and why a round may mix transports [MEAS, notes/glm-glx-ethdec.md (lines 8-11, 25-33)]. What the image *does* gate is the code: only the decode image carries the Ethernet-round path (`glm_eth_rounds` = decode image only) [FACT, src/brisc.cpp:16959-16964]. Per-transport, per-size profiling is always available via `TtGlmRoundProfile` [FACT, src/constants.h:2998-3003].

## The economic spine and the two crossovers

The rule's economics are measured, with the arithmetic shown:

$$t_{\text{PCIe}} \approx 5\ \text{us} + 0.33\ \text{us per KB of a chip's slice}, \qquad t_{\text{Eth}} \approx 15\ \text{us} + 0.037\ \text{us per KB}$$

[MEAS, notes/glm-ccl.md (lines 397-398, 559)]. The fixed-vs-slope crossover:

$$5 + 0.33\,k = 15 + 0.037\,k \;\Longrightarrow\; k = \frac{10}{0.293} \approx 34\ \text{KB}$$

— "the crossover is a slice of about 34 KB per chip, or about 7 KB against Ethernet without feeders (7 us fixed)" [MEAS, notes/glm-ccl.md (lines 399-405)]. Below it PCIe wins on latency — "so decode stays on PCIe: every B=32 decode gather is latency-bound or close to it" [MEAS, notes/glm-ccl.md (lines 561-562)]; above it Ethernet's per-link rate takes over (49 GB/s raw per link, ~34 GB/s each way in the all-gather loop, 195 GB/s sent per chip with 6 feeders, against `GLM_GATHER`'s 21-23 GB/s per chip over PCIe) [MEAS, notes/glm-ccl.md (line 557); notes/glm-eth.md (lines 11, 152-158)] — which is why **prefill needs Ethernet**: on PCIe the 2048-row chunk's gathers alone take 268 ms against Ethernet's 55 ms [MEAS, notes/glm-ccl.md (lines 509-519, 570-576)]. On a Galaxy's x1 links the *round* crossover collapses the other way — "Ethernet wins from 428 B per chip at Gen4 and 764 B at Gen5, and by 5x (Gen5) to 10x (Gen4) at 4 KB" [MEAS, notes/glm-eth.md (lines 395-396)] — so every decode round (144 B to 4608 B) sits in Ethernet's winning band, which is the whole reason `auto` keys on link width [FACT, src/tt_backend.cpp:5885-5963].

On the LoudBox, the same ranking read at round level: box C's Ethernet rounds measured 2.77-3.31 us root-triggered against PCIe `glm_round`'s 2.30-3.10 at the same sizes [MEAS, notes/glm-eth.md (lines 316-318)], and on a LoudBox's 3-cube the trees measure 2.73-2.91 us — "PCIe beats the cube on the LoudBox by 0.2-0.5 us: one hop of x16 to every peer against three hops" [MEAS, notes/glm-eth.md (lines 394-395); notes/glm-glx-ethdec.md (lines 150-158)].

## What `topology.*` feeds the choice

The ERISC service needs a specific fabric — an 8-chip **3-cube**: "exactly two links between [neighbour pairs], seen from both ends", `TT_ETH_LINKS` = 2 per pair, 6 links per chip [FACT, src/topology.h:10-19; src/constants.h:1528-1531]. The discovery and cover machinery:

| Function | What it decides | Citation |
| --- | --- | --- |
| `eth_links` / `eth_cube` | reads each ETH tile's base-firmware boot results and matches remote ASIC ids to the mesh's chips; `eth_cube` "searches the labelings of the chips for one under which every pair of labels one bit apart has exactly two links, and fails otherwise" | [FACT, src/tt_backend.cpp:5247-5337] |
| `cube_neighbours(a, b)` | whether two devices are 3-cube neighbours: "exactly two links between them, seen from both ends" | [FACT, src/topology.h:16-19; src/topology.cpp:28-34] |
| `find_cubes` | "every set of eight devices that holds a 3-cube whose 12 neighbour pairs are cube neighbours" | [FACT, src/topology.h:22-23] |
| `cube_cover` | "the most disjoint cubes ... at most `most`: of the largest such sets, the one with the most groups of `preferred`" (known-good groups, in order) | [FACT, src/topology.h:22-29] |
| `galaxy_known_groups` | a 32-chip Galaxy's known covers: "the four 2x2x2 blocks over the d <-> d + 16 cables (the groups Galaxy runs default to), then the four over the d <-> d + 8 cables" | [FACT, src/topology.h:38-40; src/topology.cpp:190-197] |

`serve --groups auto` takes the cover and prints it as a `TT_LAB_GROUPS` line [FACT, src/main.cpp:160-190]; the ERISC service itself starts on the mesh's discovered cube at the first GLM image start when `eth_min` asks for it [FACT, src/tt_backend.cpp:6011-6037]. Fabric realities that move the numbers:

- **Not every 8-chip set is a cube.** a10u28's links are "an 8-ring (2-0-6-4-5-7-1-3) plus the chords 0-1 and 6-7, with a diameter of 4" — the all-gather "runs only on boxes A and C" there [MEAS, notes/glm-ccl.md (lines 15-19, 563)]. A Galaxy PCIe bus group is not a cube either: "rows of 4 with no wraparound, rungs m <-> m + 4. That is diameter 4, so a round needs a 4th hop" (compendium row 61), and its row paths relay twice in one direction, "which this ring layout cannot do" — so Galaxy groups run the service on the *cube* groups (the d ↔ d + 16 blocks) [MEAS, notes/glm-eth.md (lines 435-441)].
- **Two links per pair matter.** With one link per neighbour pair, rounds slow and the slowest chip's p99 doubles — the measured penalty is compendium row 36 [MEAS, notes/glm-eth.md (lines 317, 320, 397-399)].
- **Cables cost a little more than in-board links** — about +0.05 us per hop, compendium row 26 [MEAS, notes/glm-eth.md (line 444)].
- **The fabric is uniform across group borders:** "links between groups behave like links inside one, to within 0.01 us" — the hop-count ping ladder behind it is compendium rows 46-47 [MEAS, notes/glm-glx-layouts.md (lines 132-141)].

So the selection pipeline composes: *link inventory → cube cover → per-host latency class → `eth_min`/`far_pcie` in the lead's L1 → per-round lead choice* [INFER, src/topology.h; src/tt_backend.cpp:5885-6062].

## Two rulings the ownership table does not derive

- **Chip barriers and step syncs stay on-chip** — NoC atomics among one chip's leads, never a cross-chip transport, on either host ([Ch2 `03-lead-helper-teams.md`](../ch2_record_driven_execution/03-lead-helper-teams.md)'s exchange; its cost is compendium row 3) [MEAS, notes/glm-ccl.md (line 35)].
- **The end-of-command courier exchange stays PCIe** — the root-to-root ring with its read-back fence ([`01`](./01-pcie-peer-bar-dma.md)) [FACT, src/brisc.cpp:3466-3470].

Every measured value behind the rule is a host-specific measurement; a port inherits none of them — the re-measure rule and checklist are [Ch8 `04-port-gates-and-re-measure.md`](../ch8_tt_metal_port/04-port-gates-and-re-measure.md)'s, with [`04-latency-compendium.md`](./04-latency-compendium.md) as that checklist's reference table.

## The demand side — cross-references only

Why the transports move what they move is [Ch4](../ch4_dram_streaming_discipline/index.md)'s economics: batching multiplies the bytes each round and gather carries, and speculative decoding multiplies the *count* of rounds — draft rows ride the same collectives as main rows, so the rounds per accepted token scale with the draft depth [INFER, notes/glm-decode-roofline.md §2] — and those economics are [Ch4 `05-batch-and-mtp-economics.md`](../ch4_dram_streaming_discipline/05-batch-and-mtp-economics.md)'s, which defers the transport-round costs here. This chapter's answer to that deferral, in its own numbers only: the transport cost of an extra row is the round-size term of the two fits above — about $+0.33$ us per KB per round on the LoudBox for the bytes a row adds, plus per-round fixed costs that do not grow with rows until the 4608 B slice cap forces shape changes [INFER, notes/glm-ccl.md (lines 397-405); src/constants.h:2983-2984].

---

← [`02-erisc-message-service.md`](./02-erisc-message-service.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`04-latency-compendium.md`](./04-latency-compendium.md)
