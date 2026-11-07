# PCIe Peer-BAR DMA

The first transport is not a network stack at all: a Tensix core issues a plain NoC write whose destination is a 64-bit peer-chip address, the write crosses PCIe through a host-mapped peer **PCIe BAR window**, and it lands in the peer's L1 or DRAM — with no host CPU, no kernel crossing, and no device driver in the per-token path. The host's entire job is load-time plumbing (one ioctl per chip pair, a window table, a published peer map); from then on every round, barrier, and gather crosses chips on writes the device itself originates. This file covers the BAR0 window layout, the host and device sides of the setup, the relay pattern that carries results between chips, where the peer windows sit in the address map, why device-originated beats host-relayed, and the LoudBox latency anchors.

**Scope:** covers Question 4, part: device-originated PCIe peer-BAR DMA — peer BAR region setup, the lead-relay pattern, placement of the peer windows in the L1/DRAM map, why device-originated beats host-relayed, and the measured per-socket latencies. The ERISC service is [`02-erisc-message-service.md`](./02-erisc-message-service.md)'s; the selection rule that decides between the two is [`03-rounds-and-transport-selection.md`](./03-rounds-and-transport-selection.md)'s; the lead/root relay *boundary* ("the lead initiates, the root relays") is [Ch2 `03-lead-helper-teams.md`](../ch2_record_driven_execution/03-lead-helper-teams.md)'s. Primary evidence: [src/brisc.cpp](../../../regen-glm5.3-flash/src/brisc.cpp), [src/tt_backend.cpp](../../../regen-glm5.3-flash/src/tt_backend.cpp), [src/constants.h](../../../regen-glm5.3-flash/src/constants.h), [notes/glm-ccl.md](../../../regen-glm5.3-flash/notes/glm-ccl.md).

## What "device-originated" means

The PCIe tile is a NoC endpoint like any other: a NoC write whose target coordinate is `TT_PCIE_COORD` = $19 \,|\, (24 \ll 6)$ exits the chip over PCIe instead of the die fabric [FACT, src/constants.h:140]. Every cross-chip send in the decode path is therefore issued by BRISC firmware as a `noc_write` — the same register sequence as a local L1 write, with a 64-bit destination address that lands in a peer chip's BAR0 [FACT, src/brisc.cpp:2674-2699]. The host's entire role is load-time: program the windows, publish the peer addresses, and never touch the path again [INFER, src/tt_backend.cpp:3276-3307, 4711-4733]. That is [Ch2 `05-eliminating-the-runtime.md`](../ch2_record_driven_execution/05-eliminating-the-runtime.md)'s no-host rule (c) applied to I/O: no cross-chip exchange inside the token loop ever visits the host.

## The address layer: BAR0 windows

Every chip's BAR0 is mapped at `0x100000000` and carved into 2 MB TLB **PCIe BAR windows** whose configs live at BAR0 offset `0x1FC00000` [FACT, src/tt_backend.cpp:69-72]. The host points a fixed run of them at exactly the apertures peers need [FACT, src/tt_backend.cpp:70-90; src/constants.h:144-152]:

| BAR0 window | Offset from the peer's BAR0 base | Points at | Ordering | Citation |
| --- | ---: | --- | --- | --- |
| 0-15 (two per lead) | 2 MB each | each lead's L1, plus its register window | strict | [FACT, src/constants.h:335-337] |
| 16-17 | — | the retargetable scratch pair (host-side reads) | strict | [FACT, src/tt_backend.cpp:1316-1320] |
| 18 (`multicast_window`) | 0x2400000 | multicast loads over the lead column | relaxed | [FACT, src/constants.h:422] |
| **19 (`peer_round_window`)** | **`TT_PEER_ROUND_WINDOW` = 0x2600000** | **the chip root's L1 from address 0** — "the root's BAR0 window that rounds write peer slices through, relaxed so that each write does not wait for the one before it" | **relaxed** | [FACT, src/tt_backend.cpp:76-78, 1290-1292; src/constants.h:144-146] |
| 20 (`lead_cast_window`) | 0x2800000 | the lead column as a strict multicast (command posts) | strict | [FACT, src/tt_backend.cpp:79-80, 1431-1434] |
| **21-116 (`peer_dram_window` + 96)** | **`TT_PEER_DRAM_WINDOW` = 0x2A00000 + channel offset** | the **GLM gather arena** in DRAM: 12 relaxed windows per channel × 8 channels onto a 24 MB 2 MB-aligned arena per channel | relaxed | [FACT, src/constants.h:147-152; src/tt_backend.cpp:81-86, 1940-1953] |

In all, `bar0_windows` = 21 + 96 = 117 windows and `peer_bar_bytes` = 117 × 2 MB of peer aperture per chip pair; the window-number arithmetic is pinned by `static_assert`s — window 19 × 2 MB = 0x2600000 and window 21 × 2 MB = 0x2A00000 [FACT, src/tt_backend.cpp:78, 83]. The relaxed/strict choice on window 19 is a bandwidth decision, not a formality: data through a *strict* window "runs at a few GB/s, against ~36 GB/s inbound at the root through the relaxed round window" [MEAS, notes/glm-ccl.md (lines 188-192)] — and inbound serialization is the PCIe tile, not the window's ordering mode (posted ordering mode 2 changed nothing; mode 3 "dropped two chips on box A" and stays unused [MEAS, notes/glm-ccl.md (lines 258-269, 604-607)]).

## Host side: three things, all before any command runs

1. **Map each peer's BAR0 into every chip's DMA space.** For every ordered pair (src, dst), `map_peer_bars` issues `TENSTORRENT_IOCTL_MAP_PEER_BAR` for dst's BAR0 at the window run the process allocated (`peer_bar_offset` = `window_base` × 2 MB, length `peer_bar_bytes`), and stores the returned DMA address; "KMD owns this mapping until the source device fd is closed" [FACT, src/tt_backend.cpp:3276-3303]. Where the KMD hands back an address at or above $2^{58}$, the mapping goes through an outbound iATU region instead — `map_dma_iatu` programs region `dst` of `src`'s iATU (registers at BAR1 UC + `0x1000 + region * 0x200`) and returns the region's base with bit 60 set, because "BH NoC window 4 translates through outbound iATU before reaching the IOMMU"; a region collision is refused — an iATU region left on by a client killed without cleanup "only a reset clears" [FACT, src/tt_backend.cpp:3242-3274, 3250-3255].
2. **Publish the peer table to the device.** `make_grid` writes each worker's `TtGrid` with `peer[2 * p]`/`peer[2 * p + 1]` = the low/high halves of "the PCIe address of each other chip's root L1" [FACT, src/tt_backend.cpp:4711-4733; src/constants.h:155-159]. So every Tensix core carries, for its whole uptime, the 64-bit doorbell of all seven peers.
3. **Point the gather-arena windows.** `configure_glm_gather_arena` re-points the 96 relaxed DRAM windows of every chip at the arena offset once per load, and clears the arena's seq lines; the arena must sit 2 MB-aligned and inside the channel's usable bytes, or load is refused [FACT, src/tt_backend.cpp:1940-1968; notes/glm-ccl.md (lines 317-321)].

Nothing in steps 1-3 costs a token anything — the ioctls, iATU programs, and `TtGrid` writes happen at load, before the first record runs [INFER, src/tt_backend.cpp:3276-3307, 4711-4733].

## Device side: the round send path, step by step

The core exchange is `round_exchange_tagged` — an all-gather of each chip root's staged slice through `[slice | seq]` slots, with a pluggable *post* so the same body can send over PCIe, over Ethernet, or mixed ([`03`](./03-rounds-and-transport-selection.md)) [FACT, src/brisc.cpp:2646-2659]. Its PCIe send loop [FACT, src/brisc.cpp:2660-2699]:

1. The root stages its chip's slice at `TT_GLM_ROUND_STAGE_ADDR` with the trailing seq word, and fences [FACT, src/brisc.cpp:2662-2665; src/constants.h:2030].
2. The slice (plus its 16-byte line) is chunked at `TT_NOC_MAX_BULK_BYTES` and, **for each chunk, written once per peer**: the command buffer's target address is reprogrammed to `grid.peer[chip] + TT_PEER_ROUND_WINDOW + rx + offset`, the return-address high word to `TT_PCIE_COORD`, and the write issued on the round's own NoC ID (`noc_id<3>()`) [FACT, src/brisc.cpp:2670-2699; src/constants.h:140].
3. The peer order is **by shift** — chip $(\text{self} + k) \bmod 8$ for $k = 1..7$ — "so each line waits behind the sender's earlier ones, so no chip should be every root's last" [FACT, src/brisc.cpp:2687-2689]. "Writes to one peer stay in order, so its seq lands last" [FACT, src/brisc.cpp:2665-2667].
4. The peers' slices *arrive in the root's own L1* — every peer wrote them into this chip's round RX slots through this chip's PCIe BAR window 19 — so the root polls their seq lines locally and multicasts each slice to the lead column (`deliver`) as it arrives, in arrival order [FACT, src/brisc.cpp:2701-2716].
5. The send ID is drained, and a release-flag multicast ends the round [FACT, src/brisc.cpp:2717-2719, 2748-2754].

`glm_round_with` wraps this body with the decode round's state: the RX parity comes from the root's running count at `TT_GLM_ROUND_COUNT_ADDR`, the count rides word 1 of the seq line, and after the round a stale-line check stops the root with `0xBADA0000 | chip` if any peer's line holds another seq or count [FACT, src/brisc.cpp:17007-17053]. Two RX parities of per-chip `[slice | seq]` slots make the reuse rule sound: "a chip cannot then start round N + 2, which reuses N's parity, before every peer has sent N + 1, and a peer sends N + 1 only after it has delivered N" [MEAS, notes/glm-ccl.md (lines 100-107)] — the round-count parity that let the step mesh syncs be deleted ([`03`](./03-rounds-and-transport-selection.md)).

## The relay pattern: who originates the writes

- **Rounds — the chip root relays.** "Every worker of every chip; worker 0 is the chip root that relays rounds" [FACT, src/constants.h:154-155]. The lead that runs the op body calls `glm_round` and the root's relay is inside it — the interpreter's step never sees the transport [INFER, src/brisc.cpp:17007-17056].
- **Gathers — the leads relay.** `GLM_GATHER` (cmd 147) moves token rows between chips with no host involvement, replacing the host copies of `glm_host_gather` [FACT, notes/glm-ccl.md (lines 306-309)]: "Lead l of 0-6 reads the chip's slice from DRAM into two L1 halves of 32 KB, then writes it to chip + l + 1 in 16 KB pieces. The next half's reads overlap the previous half's writes" [MEAS, notes/glm-ccl.md (lines 320-329)]. Ordering on the bulk path is a measured fact, not an assumption: a 16 B seq line written *after* a channel's pieces never landed before them — "0 in 390k pieces" across 16-196 KB — while a seq carried inside the payload was reordered 30-107 times per run of 11k-134k pieces, "so a seq written after the pieces is enough. A seq carried inside the payload is not" [MEAS, notes/glm-ccl.md (lines 329-338)]. With `peer_stride`, chip c sends a *different* part to each peer (the all-to-all / K-split reduce-scatter transport), the gather tile interleaving the peers' pieces "so all 7 PCIe paths stay busy" [MEAS, notes/glm-ccl.md (lines 466-474)]. A single **gather tile** (role `TT_ROLE_GLM_GATHER`, the lead column's tile in a row without a lead) can drive the whole gather off the leads via `GLM_GATHER_START`/`GLM_GATHER_WAIT`, matching the 7 leads within 3 us at 128 KB per chip [FACT, src/constants.h:3050-3062; MEAS, notes/glm-ccl.md (lines 426-451)].
- **At command close, the courier exchange.** `exchange_across_chips` — the once-per-post all-gather of the command loop's close — is the same idea at end-of-command grain: "each courier owns one incoming staging buffer. Ready releases that buffer, not the live computation buffer" [FACT, src/brisc.cpp:3462-3466]. Its ordering discipline is instructive of the transport's reliability model: payload and span are drained, then a **PCIe read of the peer serves as a fence** before the seq is published, and "peer BAR windows must use Default mode: the read-back fences payload completion before notification" [FACT, src/brisc.cpp:3466-3470]. GLM steps no longer run it per step [FACT, notes/glm-ccl.md (lines 152-158)], but it remains the end-of-command exchange and the prefill/batch path's mechanism [FACT, src/brisc.cpp:23655, 4454, 4652].

## Peer windows in the L1/DRAM map

The peer-round PCIe BAR window covers the whole root L1, so the round machinery defines its own slots inside it [FACT, src/constants.h:2030, 2983-2996]:

| L1 (root, peer-visible through window 19) | Address | Meaning |
| --- | ---: | --- |
| GLM round stage | `TT_GLM_ROUND_STAGE_ADDR` 0x144000 | the root's outgoing slice + seq + count |
| GLM round RX slots | `TT_GLM_ROUND_RX_ADDR` | two parities of 8 per-chip `[slice \| seq]` slots, sized by `TT_GLM_ROUND_MAX_SLICE` = 4608 B |
| Round count | `TT_GLM_ROUND_COUNT_ADDR` | rounds since image start; picks the parity |
| Message-service lead block | `TT_GLM_ETH_MSG_LEAD_ADDR` 0x157400 | `TtEthMsgLead` + per-ERISC credits and post staging ([`02`](./02-erisc-message-service.md)) |

Gather traffic bypasses L1 for the 24 MB-per-channel arena in DRAM (windows 21-116) [FACT, notes/glm-ccl.md (lines 317-321); src/tt_backend.cpp:1947-1953]. NoC-ID budgeting across these relay writers and the streaming readers is jointly owned with [Ch4 `03-prefetch-windows-and-noc-id-budget.md`](../ch4_dram_streaming_discipline/03-prefetch-windows-and-noc-id-budget.md).

## Why device-originated beats host-relayed

The comparison is measured, not argued: `GLM_GATHER` replaced a host-copy path (`glm_host_gather`, batch `GLM_BATCH_GATHER` cmd 146) that issued the same moves from the host [FACT, notes/glm-ccl.md (lines 306-309)] — and the measured B=32 head-to-head, device against host copies on the three gathers, is [`04-latency-compendium.md`](./04-latency-compendium.md) row 22 [MEAS, notes/glm-ccl.md (lines 357-358)].

Three reasons the device path wins, each with a number behind it:

- **No host scheduling in the round.** Run from the host one command at a time, "the chips start about 12 us apart. The slowest chip's time then includes that skew" [MEAS, notes/glm-ccl.md (line 362)] — an order of magnitude above an entire PCIe round. The counterfactual is stark: the token has 235 (dense) / 257 (index) op-internal rounds [MEAS, notes/glm-ccl.md (line 222)], so even a 10 us host relay per round would add

$$235 \times 10\ \text{us} \approx 2.35\ \text{ms per token}$$

against the 506/550 us the device-originated path actually costs — before the host's own scheduling jitter [INFER, notes/glm-ccl.md (lines 222, 36)]. And across hosts, relay is out of class outright: "Across hosts over PCIe or the host network costs tens of microseconds per exchange: not viable" [MEAS, notes/glm-eth.md (line 292)].
- **The relay is a spread all-to-all, not N serial hops.** Each root writes all 7 peers in parallel; only inbound serialization at each receiver's PCIe tile bounds it [MEAS, notes/glm-ccl.md (lines 59-78)]. The one restructure that was tried did **not** pay: spreading round traffic over the receiving chip's leads was measured and dropped — for small slices the forward hop costs more than the ~0.44 us of root inbound serialization it saves, and for larger slices the strict lead PCIe BAR windows serialize the data [MEAS, notes/glm-ccl.md (lines 175-207)]. The root-relay shape is the measured winner on this hardware [INFER, notes/glm-ccl.md (lines 175-207)].
- **What the host does at load is not nothing, but it is not per-token either.** The invariant this file states is device-originated transports, not BAR0 window indices — the port-side question is [Ch8 `02-load-bearing-vs-incidental.md`](../ch8_tt_metal_port/02-load-bearing-vs-incidental.md)'s [INFER, src/tt_backend.cpp:74-90].

## Measured latencies — anchors, then the compendium

Back-to-back bench, no skew, 1.35 GHz clock (1 us = 1350 cycles) [MEAS, notes/glm-ccl.md (lines 3-5, 28)]. The single registry of measured transport numbers — chip-local primitives, peer reads, barriers, `glm_round` by size, the in-situ round chain — is [`04-latency-compendium.md`](./04-latency-compendium.md)'s table; this file keeps only the anchors its argument leans on:

| Anchor | Value | Compendium row |
| --- | ---: | --- |
| **PCIe peer write, one way, same / cross socket (relaxed = strict window)** | **0.50 / 0.55 us** | row 5 — [MEAS, notes/glm-ccl.md (line 36)] |
| `glm_round`, 1 contributor, 256 B / 1 KB / 4 KB per chip | 2.15 / 2.26 / 2.93 us | row 14 — [MEAS, notes/glm-ccl.md (line 43)] |
| 8-chip all-to-all of 16 B lines, one root to/from 7 roots | 1.46 us | row 9 — [MEAS, notes/glm-ccl.md (line 38)] |

Two structural facts behind the all-to-all row, both measured:

- **What the sender does does not matter.** NoC 0 or 1, posted or non-posted, static or dynamic VC, one transaction ID per shift, a separate peer-round window per shift, relaxed or strict — all stay within 1.47-1.50 us at 7 shifts [MEAS, notes/glm-ccl.md (lines 67-71)].
- **The destination tile is what matters.** "Inbound PCIe writes to one tile serialize at about 0.13 us each" — the 175-cycle NoC write-acknowledge time, so the PCIe tile keeps one write to a given tile in flight; spread over 7 destination tiles, the same 7 lines finish in 1.02 us [MEAS, notes/glm-ccl.md (lines 74-78)]. The arithmetic the design leans on:

$$1.46 - 1.02 = 0.44\ \text{us saved by spreading}, \qquad 7 \times 0.13 = 0.91\ \text{us of unoverlapped inbound time}$$

In situ — real-weight chat on a10u28, each chip root stamping its rounds — the per-token round chain, its per-kind costs and the deleted step mesh sync are compendium rows 15-17 ([`04`](./04-latency-compendium.md)); the sync's removal is [`03`](./03-rounds-and-transport-selection.md)'s story.

## Host and socket dependence, made explicit

Every number above is a property of *this host's* PCIe fabric; the topology it came from — a10u28: 8 × p150b, one Gen5 x16 root port each, chips 0-3 / 4-7 split across two EPYC 9124 sockets — is [`04-latency-compendium.md`](./04-latency-compendium.md)'s host key [MEAS, notes/glm-ccl.md (lines 10-12)]. The same/cross-socket split is exactly the NUMA 0/1 split: a peer write crossing the socket between the EPYCs pays 0.55 us against 0.50 inside one, and a read 1.06 against 0.96 [MEAS, notes/glm-ccl.md (lines 36-37)] — 0.05 us on writes, 0.10 on reads. **Lead placement is latency-neutral** — "Every lead ... measures 0.501 / 0.551 us one way to a peer, and an acked NoC write takes 175 cycles to any lead" — so which column leads is free to be decided by the DRAM-reader placement ([Ch4 `02`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)'s), not by the transport [MEAS, notes/glm-ccl.md (line 211)]. A Galaxy's chips sit on x1 links, where these numbers are void — the per-byte cost dominates and the round crosses to Ethernet; that comparison, with the x1 numbers, is [`03-rounds-and-transport-selection.md`](./03-rounds-and-transport-selection.md)'s. A port inherits none of these rows: they are re-measure items on the port's own host ([Ch8 `04-port-gates-and-re-measure.md`](../ch8_tt_metal_port/04-port-gates-and-re-measure.md) owns the checklist).

---

← [`index.md`](./index.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`02-erisc-message-service.md`](./02-erisc-message-service.md)
