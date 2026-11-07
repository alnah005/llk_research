# The Measured Envelope and the Evidence Discipline

The rest of this guide must account for a small set of measured numbers, stated here up front with their citations. This file also establishes the citation discipline every chapter follows — one evidence tag per claim, measured numbers verbatim from the registry or verified in the notes before writing, hybrid tags banned — with worked examples of how to cite, how to verify, and how to split a compound claim.

**Scope:** shared foundation (no question owned). Each number below carries its citation here; the owning chapter carries the full exposition (the "owned by" column).

## The envelope

All rows: single-user decode on one LoudBox, `--tiles 64` [MEAS, README.md "Performance"], unless stated.

| # | Quantity | Value | Citation | Owned by |
| --- | --- | --- | --- | --- |
| 1 | Single-user decode, `hot3-meas` | 165.3 t/s = 6.05 ms/token | [MEAS, notes/post-rc/glm-rc1.md lines 136-139] | Ch4 `04` |
| 2 | Single-user short context, `plan0h` | 178.9 t/s = 5.59 ms/token | [MEAS, notes/post-rc/glm-rc1.md lines 136-139] | Ch4 `04` |
| 3 | Repeat stability of row 1 | 165.29 / 165.30 t/s, back to back | [MEAS, notes/post-rc/glm-rc1.md lines 139] | Ch4 `04` |
| 4 | Context decay, `hot3-meas` | 165.9 t/s at 48-token context, declining to 128.4 t/s at 262,014 tokens | [MEAS, README.md "Performance"] | Ch1 (stated here; substrate in [`03-workload-context.md`](./03-workload-context.md)) |
| 5 | Streaming-window rate | 447 GB/s per chip = 89% of the 500 GB/s peak, inside DRAM-streaming windows | [MEAS, notes/glm-decode-roofline.md §5, lines 108, 177-181] | Ch4 `01` |
| 6 | DRAM-busy plan requirement | DRAM busy 79% (`plan0h`) to 95% (`hotE`) of the token for the 5 ms plan | [MEAS, notes/glm-decode-roofline.md §5, lines 108, 177-181] | Ch4 `01` |
| 7 | Per-token DRAM read | 2.10 GB/chip/token (16.8 GB for the box) | [MEAS, README.md "Roofline"] | Ch4 `01` |
| 8 | Weight file, `hot3-meas` | 245.7 GB | [MEAS, README.md "Weights"] | Ch6 `02` |
| 9 | Weight file, `plan0h` | 180.9 GB (routed experts all BFP4) | [MEAS, README.md "Weights"] | Ch6 `02` |
| 10 | Hot experts | 5153 hot experts = 42.6% of experts, stored at BFP8 (in `hot3-meas`) | [MEAS, README.md "Weights"] | Ch6 `02` |
| 11 | Requant class policy | Routed experts BFP4 with hot experts at BFP8; router and indexer compressor gate BF16 strips; every other matrix BFP8; small parameters at source precision | [FACT, src/requant.cpp:115-117] | Ch6 `02` |
| 12 | BFP4 exponents | imatrix-weighted in both files | [FACT, README.md "Weights"] | Ch6 `02` |
| 13 | Device image ceiling | Image $\le$ 0x16000 = 88 KB, enforced by a linker ASSERT (ARC scratch ceiling) | [FACT, src/brisc.ld:28] | Ch7 `02` |
| 14 | BRISC stack | Set to 0xFFB02000 (the 8 KB LDM) at entry | [FACT, src/brisc_entry.S:9] | Ch7 `02` |
| 15 | MMIO addresses | Instruction port 0xFFE40000; config base 0xFFEF0000; NOC0 0xFFB20000 | [FACT, src/fw.h:19-20, 16] | Ch3 `01` |
| 16 | NOC1 base | 0xFFB30000 | [FACT, src/brisc.cpp:694] | Ch3 `01` |
| 17 | Toolchain pin | riscv-tt-elf-g++, sfpi 7.76.0 | [FACT, rules.py:12] | Ch7 `01` |

Rows 10 and 11 are deliberately separate claims about the same subject — see worked example 1 below.

## What this chapter does NOT carry

Per the guide's ownership rules, everything else is a pointer: transport latencies (PCIe one-way, all-to-all, mesh barrier, ERISC in-situ rounds) live only in [Ch5 `04-latency-compendium.md`](../ch5_cross_chip_transports/04-latency-compendium.md); the 9.3 → 6.05 ms optimization ledger, the per-op time tables, and the window-rate exposition live only in [Ch4](../ch4_dram_streaming_discipline/index.md); TTQ internals, requant-class selection detail, and hot-expert selection mechanics live only in [Ch6 `02`](../ch6_host_driver_weights_oracle/02-ttq-container-and-requant.md); toolchain and image-budget detail beyond rows 13-14, 17 live only in [Ch7](../ch7_build_determinism/index.md); batch/MTP economics live only in [Ch4 `05`](../ch4_dram_streaming_discipline/05-batch-and-mtp-economics.md); and the re-measure checklist — which of these numbers a port must measure on its own host and never inherit — lives only in [Ch8 `04`](../ch8_tt_metal_port/04-port-gates-and-re-measure.md).

## The evidence discipline

Every factual or numbered claim in this guide carries **exactly one** evidence tag:

- `[MEAS, notes/<file>.md]` / `[MEAS, README.md]` / `[MEAS, src/<file>:line]` — measured on silicon/bench. Registry numbers are cited verbatim: never re-rounded, never invented.
- `[FACT, src/<file>]` — structural facts: addresses, struct layouts, linker asserts, pins, code comments stating policy. The form `[FACT, notes/<file>.md]` is used the same way for structural facts verified in the corpus (layer maps, sync-chain counts, model shapes).
- `[EST, <file>]` — an estimate or derived budget recorded in the corpus (e.g. a computed floor), cited to it.
- `[INFER, src/<file>]` — derived from code reading, never presented as measured; used strictly where the notes are silent.
- `[cross-check, regen_kernel_port.md]` — port-side material, only in [Ch8](../ch8_tt_metal_port/index.md), never presented as regen-internal evidence.

Three rules bind them: (1) exactly one tag per claim — hybrid tags (a measured tag on one side, a structural or derived tag on the other, joined in one bracket) are banned; a claim with a measured part and an inferred part is two claims. (2) A number not in the orchestrator's registry must be located and verified by the writer in the notes or source **before** writing, then cited with file (and line where useful). (3) When a claim is anchored to a line number, the anchor must be verified — a guide that claims verified anchors must have verified every one of them.

### Worked example 1 — splitting a compound claim (the canonical citation error)

The hot-expert fact is two claims, and gluing them is the canonical failure:

> **WRONG:** "5153 hot experts (42.6% of experts) at BFP8 [FACT, src/requant.cpp:115-117]."

`src/requant.cpp:115-117` states only the *policy* — "routed experts BFP4 except the hot experts at BFP8, the router and the indexer compressor gate BF16 strips, every other matrix BFP8 ... small parameters exactly as the source stores them" — and nowhere the count. The right way is rows 10 and 11 above: the measured count carries `[MEAS, README.md "Weights"]` (it is independently recorded for the hotE-meas file in [MEAS, notes/glm-bfp8.md:171]), and the policy carries `[FACT, src/requant.cpp:115-117]` as a separate claim. One tag per claim, split wherever a sentence mixes measurement with structure.

### Worked example 2 — anchor precision (a doc comment is not its definition)

When a claim depends on a line anchor, the anchor must name the exact line the thing is on:

- The interpreter loop `glm_step_list` is **defined at `src/brisc.cpp:21153`**; the line above (`src/brisc.cpp:21151-21152`) is its doc comment ("Runs the `count` steps of the list at `steps`..."). Citing 21152 as the definition misdirects the reader to the comment. The same rule holds at file grain — `NOC1_BASE` and `NOC0_BASE` live in *different* files ([`04-source-map.md`](./04-source-map.md)'s anchor table carries the pair).

The verification method this chapter used: `grep -n` the symbol, read the exact line and its neighbors, then cite. All anchors in [`04-source-map.md`](./04-source-map.md) and this file were produced that way; any file that claims its anchors were verified must have verified every one of them.

### Worked example 3 — MEAS vs EST: computed inventories are not measurements

[`notes/glm-inventory.md`](../../../regen-glm5.3-flash/notes/glm-inventory.md) derives its per-class byte totals from the model config and the quantization plan, so its tables are estimates recorded in the corpus — cited `[EST, notes/glm-inventory.md "Totals"]` in [`03-workload-context.md`](./03-workload-context.md). The roofline note's per-chip bytes, by contrast, come from the loader's own strip shares dumped at load on real weights, with every number reproduced within 1% on a second box [MEAS, notes/glm-decode-roofline.md (lines 4-6)]. Same subject (decode bytes), different evidence classes — and the tag must follow the evidence, not the rhetorical convenience. For the same reason the DRAM floors (3.53 / 4.27 ms) are `[EST, notes/glm-decode-roofline.md "Headline"]`: they are arithmetic over measured bytes, recorded in the corpus, not themselves silicon measurements.

## Registry-verbatim writing

A registry number is quoted with its registry citation and never transformed: "about 450 GB/s", "roughly 90%", or "447.2 GB/s" are all violations — the first two re-round, the last invents precision. The fraction may be *shown* as arithmetic, not re-asserted as a different number: $447/500 = 89\%$. When two artifacts disagree, stop and resolve before writing either.

---

← [`04-source-map.md`](./04-source-map.md) | ↑ [`index.md`](./index.md) | ↑ [guide root](../index.md)

**Next:** [Chapter 2 — The Record-Driven Execution Model](../ch2_record_driven_execution/index.md)
