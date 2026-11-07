# The ERISC Message Service

The second transport is a firmware service on the Ethernet cores: `src/erisc.cpp` (609 lines) runs on ERISC1 of each linked ETH tile, next to the vendor base firmware on ERISC0, and moves variable-length `TtEthMsg` messages from a chip's first lead to the first leads of the other cube chips — L1 to L1, no DRAM in the path, under credit flow control, along binomial-tree routes, with counters engineered so that a restarted collective never reuses or double-counts a slot. The same image runs an all-gather mode for prefill-scale flows; this file covers the message service (`TT_ETH_MODE_MSG`) and the substrate both modes share, then contrasts the reliability model with PCIe's.

**Scope:** covers Question 4, part: the ERISC Ethernet message service — `TtEthMsg` format, send/receive rings, credit flow control, binomial tree collectives, restart-correct counters, the interpreter handoff, and the reliability model. The one link-latency anchor appears here with its citation; the link registry is [`04-latency-compendium.md`](./04-latency-compendium.md)'s; which rounds ride the service on which host is [`03-rounds-and-transport-selection.md`](./03-rounds-and-transport-selection.md)'s; the image budget as a build gate is [Ch7 `02-image-layout-and-budgets.md`](../ch7_build_determinism/02-image-layout-and-budgets.md)'s. Primary evidence: [src/erisc.cpp](../../../regen-glm5.3-flash/src/erisc.cpp), [src/erisc.ld](../../../regen-glm5.3-flash/src/erisc.ld), [src/erisc_entry.S](../../../regen-glm5.3-flash/src/erisc_entry.S), [src/constants.h](../../../regen-glm5.3-flash/src/constants.h), [src/brisc.cpp](../../../regen-glm5.3-flash/src/brisc.cpp), [notes/glm-eth.md](../../../regen-glm5.3-flash/notes/glm-eth.md).

## The substrate: image, modes, link

- **Image.** `src/erisc.ld` places the whole image at ETH L1 `0x60000` (`TT_ETH_FW_ADDR`) — `.text`, `.rodata`, `.data`, `.bss` in one flat run [FACT, src/erisc.ld; src/constants.h:1537]. The entry stub is three instructions: set the stack to the 8 KB LDM at `0xFFB02000`, call `erisc_main`, park [FACT, src/erisc_entry.S]. The host loads it on ERISC1 only — "next to the base firmware on ERISC0, which is never touched" [FACT, notes/glm-ccl.md (lines 581-583)] — and a host-side build assert caps the image below 0x70000, where the base firmware's L1 begins: `TT_ETH_FW_ADDR + sizeof(erisc_program) <= 0x70000` [FACT, src/tt_backend.cpp:156-159].
- **Modes, chosen at load.** `erisc_main` dispatches on `TtEthConfig::mode`: `TT_ETH_MODE_GATHER` runs the all-gather service (DRAM-to-DRAM, fed by Tensix feeder tiles — the bandwidth mode prefill uses), `TT_ETH_MODE_MSG` runs the message service; "the host loads one mode or the other (`start_eth_msg` / `start_eth_gather`, sharing the claim, load and release code), and quiesce and close stop either" [FACT, src/erisc.cpp:596-603; src/constants.h:1566-1569; notes/glm-eth.md (lines 322-324)].
- **ERISC1 can use only NoC 1** — "base firmware drives NoC 0 on ERISC0" — so all ERISC-issued NoC traffic rides NoC 1's static VC [FACT, notes/glm-eth.md (lines 141-144); src/erisc.cpp:117-133].
- **The link.** One link is driven by the Ethernet TX queue at `ETH_TXQ0` 0xFFB90000: one `eth_send` writes source, destination, size, then a start-data command, spinning on the queue's ongoing bit — reading CMD before STATUS to space the polls (a Blackhole erratum, "BH-55") [FACT, src/erisc.cpp:11-21, 41-48]. Transfers on one queue arrive in order [FACT, src/erisc.cpp:57-58] — the ordering property the whole protocol leans on. The raw link's cost: **0.39 us one-way ERISC to ERISC, traced** [MEAS, notes/glm-eth.md (lines 314, 408-410)] — the 0.67-vs-0.39 correction story, the in-board/cabled split, and the porter warning that ends it are [`04-latency-compendium.md`](./04-latency-compendium.md) reading note A and rows 23-26.

**Abort discipline.** The host owns an abort line (`TT_ETH_ABORT_ADDR`) that "only the host writes (never a lead, so a lead still running cannot overwrite it with a trigger)" [FACT, src/constants.h:1710-1712]. On `TT_ETH_STOP` there, `abort_gather` issues nothing more, waits at most ~100 ms of AICLK (135000000 cycles) for the TX queue, NoC 1's command buffers and IDs 11-15 to drain, reports `TT_ETH_DONE_ABORTED` and parks; one that does not drain parks *without* done (`TT_ETH_LIVE_STUCK`) and "the host leaves it running" — never reset busy [FACT, src/erisc.cpp:85-107; notes/glm-ccl.md (lines 588-589)].

## The `TtEthMsg` format

A slot is `[payload | 16-byte line | TtEthMsg]`, and the trailer lands last [FACT, src/constants.h:1637-1639]:

```c
struct TtEthMsg   { uint32_t tag, dst, line, info; };  // src/constants.h:1730
struct TtEthMsgLine { uint32_t seq, words[3]; };        // src/constants.h:1744
```

- `tag` = "the nonce plus the ring's count of messages before this one, plus 1" — validity and staleness in one word: a stale slot never matches a live one [FACT, src/constants.h:1726-1729].
- `info` = payload bytes (a multiple of 16, up to `TT_ETH_MSG_PAYLOAD` = 8192) | route << 16 [FACT, src/constants.h:1728-1729, 1641].
- `dst` and `line` are lead L1 addresses, 16-byte aligned, "the same on every chip that takes the message": the payload lands at `dst`, then the 16-byte line at `line` — the line *is* the receiver's completion signal, carrying the round's seq plus sender-chosen words [FACT, src/constants.h:1727-1730, 1744-1745].

Ordering is by construction: the producer writes the payload, then the line, then the `TtEthMsg`, "on the same NoC and static VC, and the link lands a transfer in order" — so the trailer's arrival proves the payload's [FACT, src/constants.h:1637-1639; notes/glm-eth.md (lines 328-330)]. `static_assert(TT_GLM_ROUND_MAX_SLICE <= TT_ETH_MSG_PAYLOAD)` guarantees the largest GLM round slice (4608 B) fits one message [FACT, src/constants.h:3013-3015].

## Rings

Each ERISC (dimension d, link l) has `TT_ETH_MSG_RINGS` = 3 TX and 3 RX rings of `TT_ETH_MSG_SLOTS` = 7 slots, slot stride 8192 + 64 = 8256 bytes; TX slot i of ring k goes to the peer's RX slot i of ring k in one link transfer [FACT, src/constants.h:1633-1644; notes/glm-eth.md (lines 324-326)]. The geometry is pinned by a `static_assert` that the whole message area fits below the recursive-doubling area [FACT, src/constants.h:1701-1705]:

$$3 \times 7 \times 8256 = 173{,}376\ \text{bytes per ring set}, \qquad \text{TX at } 0x4000,\ \text{RX ends at } 0x58\text{A}80$$

matching the note's "3 TX and 3 RX rings of 7 slots with up to 8 KB of payload (ETH L1 0x4000-0x58A80)" [FACT, notes/glm-eth.md (lines 324-325)]. The ring layout encodes the relay topology:

- **TX ring 0** is filled by the chip's first lead — the only ring a Tensix writes into directly.
- **TX ring k** (k = 1, 2) is filled by the ERISC on link l of dimension d + k (mod 3): when a message must travel on, the receiving ERISC relays it into the next dimension's sibling ring rather than bouncing through a Tensix [FACT, src/constants.h:1633-1639; src/erisc.cpp:556-560].

Ring sizing is a latency decision, not a memory one: "Ring sizes cover two rounds of the all-gather, so a round never waits on a credit", and routes must keep the relay rings' producers acyclic (dimension order does) "for a full ring never to deadlock" [FACT, notes/glm-eth.md (lines 343-344)]. The service loop polls in a fixed order — relays and deliveries ("RX rings first since relays carry the longest paths"), then sends, then frees and credits [FACT, src/erisc.cpp:538-590].

## Credits: the flow-control protocol

There is no retransmission anywhere in this service; correctness is bounded buffers plus in-order evidence. The key insight is what "free" means: **a slot is not free when its message is consumed; it is free once the writes that read it have left the chip** — and the NIU provides that knowledge for free, via NoC 1's in-order count of non-posted writes sent [FACT, notes/glm-eth.md (lines 335-336); src/erisc.cpp:250-256]. Three movements follow:

1. **Frees.** The ERISC tracks each taken message in a `MSG_PENDING` = 32-entry pending queue of (ring, writes-through-last-use), and `msg_free` drains entries whose writes have been sent — `static_assert(TT_ETH_MSG_RINGS * TT_ETH_MSG_SLOTS <= MSG_PENDING)` so every slot of every ring can be pending at once [FACT, src/erisc.cpp:255-258, 339-341, 421-438].
2. **Credits to the peer.** `msg_credit` sends the freed counts of all three rings over the link as one 16-byte credit line — when idle (every 8th idle pass, "so an idle pass stays short"), "or sooner once a ring nears exhaustion": a ring within 2 slots of full (`freed - credited >= TT_ETH_MSG_SLOTS - 2`) forces a credit even while busy [FACT, src/erisc.cpp:440-451, 567-571; notes/glm-eth.md (lines 335-337)].
3. **Credits to producers.** `msg_forward` passes the peer's per-ring credit to each TX ring's producer: "the lead for ring 0, the sibling of dimension dim + k for ring k" — the lead's credit line in `TtEthMsgLead` staging, the sibling's relay credit line [FACT, src/erisc.cpp:458-474; src/constants.h:1645-1648]. The BRISC side mirrors the rule: `eth_msg_post` spins "while count - credit >= TT_ETH_MSG_SLOTS" before staging the next message [FACT, src/brisc.cpp:2374-2380].

So the bounded resource is *slots in flight per ring*: 7 outstanding messages per ring per link, enforced at both ends [INFER, src/brisc.cpp:2374-2380; src/erisc.cpp:421-474]. Lazy credits are a measured design: the first build "sent a credit line after every message" and ran rounds at 3.8-4.0 us; idle-time credits and one status word per event brought rounds to 2.8-3.3 us [MEAS, notes/glm-eth.md (lines 423-427)]. One guard protects the NIU counters: "At most 6 writes per message: keep `NOC_MSG_ID`'s 8-bit outstanding count from wrapping" — the loop stalls while the ID holds 128 or more outstanding [FACT, src/erisc.cpp:379-381]. The ERISC NIU is its own, outside the Tensix NoC-ID budget of [Ch4 `03`](../ch4_dram_streaming_discipline/03-prefetch-windows-and-noc-id-budget.md) — the service uses ID 11 for posted credit lines and ID 12 for relays/deliveries, with a `static_assert` pinning the whole 11-15 span below the NIU's 16-ID limit [FACT, src/erisc.cpp:28-31; notes/noc-ids.md (line 75)].

## Binomial trees: relay then deliver

A route names up to two more dimensions to cross after the first, encoded by `tt_eth_msg_route(hops, first, second, tree)` [FACT, src/constants.h:1732-1741]. `TT_ETH_MSG_TREE` (0x40) turns a path into a **binomial broadcast**: "on a tree, every chip it reaches does [take it], and passes it on along each later dimension of the route, each copy keeping the dimensions after its own" [FACT, src/constants.h:1734-1736]. `TT_ETH_MSG_NEAR` (0x80) stops a two-dimension tree's first copy early — "every chip but the one across all three dimensions takes the message" — the far-chip hook that hands the antipode back to PCIe ([`03`](./03-rounds-and-transport-selection.md)) [FACT, src/constants.h:1737-1739; src/brisc.cpp:16971-16981]. `msg_receive` is where the tree lives [FACT, src/erisc.cpp:365-424]:

1. **Admission.** A message is taken once its tag matches, *and* only when "each sibling it goes on to has a free slot" — the receiving ERISC checks every relay ring's credit before taking, so a relay never overruns a sibling [FACT, src/erisc.cpp:366-375].
2. **Relays, furthest first.** "The copy that goes furthest first": the first relay keeps the route's later dimensions (unless `TT_ETH_MSG_NEAR`), the second, on a tree, crosses the last one. Each relay is two writes into the sibling's TX slot — `[payload | line]`, then a fresh `TtEthMsg` staged from a per-dimension ring of 4 stage slots [FACT, src/erisc.cpp:376-404; src/constants.h:1652].
3. **Delivery.** The payload to the lead's `dst`, then the line to `line`, "on NoC 1's static VC 1, so the line lands after the payload" — ordering by link, not by fence [FACT, src/erisc.cpp:405-410; notes/glm-eth.md (lines 332-334)].

A GLM round posts its slice as three binomial trees, one per cube dimension, in the lead's posting order (`TtEthMsgLead::order`, the longest path first), on the link of the chip's label parity [FACT, src/brisc.cpp:2405-2457; notes/glm-glx-ethdec.md (lines 27-29)]. The balanced `order` (labels 2-5 post from dims 2, 0, 1, the others from 1, 0, 2) is measured to matter: "so no ERISC moves more than 3 messages a round (plain: 4)" [FACT, src/tt_backend.cpp:5818-5833; notes/glm-glx-ethdec.md (lines 42-44)].

## How the interpreter hands messages over and back

The **[interpreter](../ch2_record_driven_execution/02-interpreter-loop.md)** side is entirely first-lead work:

- **Post.** `eth_msg_post` stages `[line | TtEthMsg]` in its own L1, checks the ERISC's credit line for a free slot, and issues two NoC 1 writes into the ERISC's TX slot (payload, then the 32-byte trailer) — the writes "stay in flight: the poster waits for them to leave (`noc1_wait_writes`) before it posts to the same ERISC again, which reuses the ERISC's stage" [FACT, src/brisc.cpp:2369-2404]. `eth_msg_post_tree` is the three-post round form [FACT, src/brisc.cpp:2405-2457].
- **Round integration.** `glm_round_with` calls the round's usual exchange with an Ethernet `post`: `glm_round_post_eth` posts the slice as trees and "returns the mask of chips the slice still goes to over PCIe" — none for a full Ethernet round, the antipode for a far-PCIe round, all of them for a PCIe round; **a round may mix transports** [FACT, src/brisc.cpp:16971-16981, 17020-17038; notes/glm-glx-ethdec.md (lines 25-33)].
- **Completion is the line, not a callback.** The ERISCs deliver payload-then-line "exactly where a PCIe round's write lands" — the arrival poll, delivery multicast, RX parity and release are the *same code* as the PCIe round [FACT, notes/glm-glx-ethdec.md (lines 24-29); src/brisc.cpp:17007-17056]. The lead's wait is the ordinary `wait_sequence` on the line's seq [INFER, src/brisc.cpp:17028-17031].
- **Config.** The host starts the service once at the first GLM image start and stops it at close; it "cannot run together with `TT_LAB_ETH_GATHER`" [FACT, src/tt_backend.cpp:6008-6037; notes/glm-glx-ethdec.md (lines 36-39)].

## Restart-correct counters

The counter discipline answers one question, and answers it differently on the two transports: **after a restart, how does a receiver tell a live collective from a stale slot, without counting any collective twice?**

**PCIe side — counters restart WITH the image.** "Job sequences restart with the image, so no round RX, contribution or QSA count slot may keep an exact match" — at every image start the host zeroes the GLM round RX slots *and the round count after them*, in one act [FACT, src/tt_backend.cpp:5021-5031]. The runtime mechanism those counters serve — count at `TT_GLM_ROUND_COUNT_ADDR` picks the RX parity and rides word 1 of the seq line, stale lines stop the root — is [`01`](./01-pcie-peer-bar-dma.md)'s round-send walk, which also carries the safety quote. What is restart-unique here is the *joint* clearing: clearing the count is safe *precisely because* the slots it names are cleared with it — the count's own reuse argument lives in [`01`](./01-pcie-peer-bar-dma.md) [INFER, src/tt_backend.cpp:5021-5031].

**ERISC side — counters survive ACROSS image starts.** The service is never reset at an image switch: "image switches never reset ERISC1" [FACT, notes/glm-ccl.md (lines 585-588)]. Instead, at every GLM image start the host rewrites each chip's `TtEthMsgLead` and "resumes each ERISC's ring 0 from the ERISC's published sent count, because every round of the previous image finished on every chip" [FACT, src/tt_backend.cpp:6043-6054; notes/glm-glx-ethdec.md (lines 36-38)] — `posted[i]` read back from the ERISC's published `TtEthMsgStatus::sent`, slot = posted % `TT_ETH_MSG_SLOTS`, credit lines matched to it, so the lead's producer counts and the ERISC's ring counts stay exactly in step across the restart [FACT, src/tt_backend.cpp:5839-5862]. The failure this prevents: a counter that rewound to zero would mint `nonce + 0 + 1` again — a tag a stale slot from before the restart may still hold, so the ERISC would take a dead message or the lead would reuse a live slot [INFER, src/erisc.cpp:342-345].

Two flanks close the remaining restart classes:

- **A firmware load restarts counting — and changes the nonce.** Tags count from "a nonce shared by both ends of the link" per firmware load [FACT, src/constants.h:1556; notes/glm-eth.md (line 72)], so a restarted *load* gets a fresh nonce space and "no slot may keep a tag from an earlier load, which a new nonce could match" [FACT, src/tt_backend.cpp:5629]. A restarted *image* keeps counting under the same nonce — resume, never reset [INFER, src/tt_backend.cpp:5839-5862].
- **A retried collective defaces its consumed slots.** `wait_take` marks a consumed slot "with the round of the sequence's job, which no sequence uses (offsets within a round start at 1), so a taken slot left unwritten across any wrap matches none" [FACT, src/brisc.cpp:2604-2608] — a stale or half-written slot can never satisfy a new wait [INFER, src/brisc.cpp:2604-2608].

The shared principle: a slot's identity is (nonce, monotone count); counts never reset while any slot they name can still be read; and consumed slots are written with a value no future wait will match [INFER, src/erisc.cpp; src/brisc.cpp:2604-2608].

## Recursive doubling: built, measured, off

A fourth posting shape lives in the same firmware: recursive doubling with ERISC-level aggregation (`TT_ETH_RD_ADDR`) — ERISC (d, l) sends a block of $2^d$ records gathered from the lead and earlier siblings, behind a `TtEthRdLine` trailer that lands last [FACT, src/erisc.cpp:478-537; src/constants.h:1672-1694]. It needs **no credits at all**: "each link carries every other round at most, and a chip starts a round only once the round before has landed on every chip" [FACT, src/constants.h:1679-1682] — the same argument as PCIe's RX parities. It loses to the trees (its critical path is the trees' longest path, and at the third step it carries 4 records instead of 1), so it stays off by default (`rd_max` = 0) [MEAS, notes/glm-glx-ethdec.md (lines 152-171)].

## The reliability model, contrasted with PCIe

| Concern | ERISC message service | PCIe peer-BAR DMA ([`01`](./01-pcie-peer-bar-dma.md)) |
| --- | --- | --- |
| Flow control | explicit credits: 7 slots/ring, freed by NIU writes-sent evidence, forwarded producer-to-producer [FACT, src/erisc.cpp:428-474] | none needed inbound; the receiver's two RX parities + round-count ordering are the whole buffer discipline [FACT, src/brisc.cpp:17007-17018] |
| Ordering evidence | transfer-in-order per link + trailer-lands-last + NIU in-order non-posted counts [FACT, src/erisc.cpp:57-58, 250-256] | per-peer write order "so its seq lands last", plus the read-back fence in the courier path [FACT, src/brisc.cpp:2665-2667, 3466-3470] |
| Stale detection | nonce tags, never reused within a load; consumed slots defaced [FACT, src/constants.h:1726-1729; src/brisc.cpp:2604-2608] | seq **and** count match; stale-line stop `0xBADA0000 \| chip` [FACT, src/brisc.cpp:17045-17049] |
| Restart discipline | counts continue across image starts; producer resumes from the ERISC's published counts [FACT, src/tt_backend.cpp:6043-6064] | counts and slots cleared together at image start [FACT, src/tt_backend.cpp:5021-5031] |
| Failure containment | abort drains then parks with a named done/stuck phase; a busy ERISC is never reset [FACT, src/erisc.cpp:85-107] | a wedged NoC or stalled ARC drops the chip off PCIe — a box-level failure the halt-all teardown addresses [MEAS, notes/glm-ccl.md (lines 599-608)] |
| ID budget coupling | ERISC's own NIU, IDs 11-15 — "their IDs do not interact with the Tensix IDs" [FACT, notes/noc-ids.md (line 75)] | paced per NoC ID under the [Ch4](../ch4_dram_streaming_discipline/03-prefetch-windows-and-noc-id-budget.md) budget [FACT, notes/noc-ids.md (lines 58-71)] |

Design notes recorded in `glm-eth.md` that a porter must not lose: where a round's time goes — "a round is the 3-hop path (2.34 us at 16 B: one post, 3 links, 2 relays, one delivery) plus about 0.45 us of ERISC contention, plus about 0.13 us per KB of slice" [MEAS, notes/glm-eth.md (lines 419-421)]; and ERISC-direct delivery into helper rectangles was measured and *rejected* — the multicast form "is the pattern that hung the NoC" and the safe forms cost as much as they save [MEAS, notes/glm-glx-ethdec.md (lines 192-195)]. The sibling-relay cost and the ERISC loop's per-link cap are registry rows ([`04`](./04-latency-compendium.md) rows 32 and 54).

---

← [`01-pcie-peer-bar-dma.md`](./01-pcie-peer-bar-dma.md) | ↑ [chapter index](./index.md) | ↑ [guide root](../index.md)

**Next:** [`03-rounds-and-transport-selection.md`](./03-rounds-and-transport-selection.md)
