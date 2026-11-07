# Source Map

The regen tree is ~1.2 MB of device source plus a large notes corpus, and almost none of it should be read linearly. This file maps every primary-evidence file to its responsibility, line count, anchor symbols, and the chapters that cite it; after this file you locate subsystems by anchor, never by scrolling. It also maps the notes corpus and lists what you can safely never read.

**Scope:** shared foundation (no question owned). The primary source root is `/home/alex/mpi-shfs/tenstorrent/regen-glm5.3-flash` (read-only; linked below as `../../../regen-glm5.3-flash/...`). Every line count in the tables was verified with `wc -l` against that tree, and every anchor in the anchor table was re-verified line-by-line while writing this chapter.

## Source file map

| File | Lines | Responsibility | Cited by |
| --- | ---: | --- | --- |
| [`src/brisc.cpp`](../../../regen-glm5.3-flash/src/brisc.cpp) | 23,665 | The device interpreter: MMIO/config layer, NoC layer, SFPU replay, op bodies (matvec, norms, router, SwiGLU, mHC, KDA, MLA, rounds), lead/helper roles, team jobs, record loop, trace/progress records | Ch2, Ch3, Ch4, Ch5 |
| [`src/tt_backend.cpp`](../../../regen-glm5.3-flash/src/tt_backend.cpp) | 38,808 | Host backend: ioctl/KMD open, BAR0 windows, firmware load, channel-local strip placement, weights loader, bench harness, proxy/sim paths | Ch5, Ch6 |
| [`src/constants.h`](../../../regen-glm5.3-flash/src/constants.h) | 3,652 | `TtGlmStepRecord` and the record/step/trace/progress struct family, command ids, the L1 memory map (`_ADDR` symbols), `TtEthMsg`, image build flags, team-job kinds | Ch2, Ch3, Ch5 |
| [`src/erisc.cpp`](../../../regen-glm5.3-flash/src/erisc.cpp) | 609 | The ERISC Ethernet message service: rings, credits, binomial trees, restart-correct counters | Ch5 |
| [`src/erisc.ld`](../../../regen-glm5.3-flash/src/erisc.ld) + [`src/erisc_entry.S`](../../../regen-glm5.3-flash/src/erisc_entry.S) | 22 + 12 | ERISC image layout and entry | Ch5, Ch7 |
| [`src/fw.h`](../../../regen-glm5.3-flash/src/fw.h) | 84 | The whole MMIO register layer: `TENSIX_CFG_BASE` 0xFFEF0000, `TENSIX_INST_BASE` 0xFFE40000, `NOC0_BASE` 0xFFB20000, instruction/PHYS_WR helpers | Ch3 |
| [`src/kernels.sfpu`](../../../regen-glm5.3-flash/src/kernels.sfpu) | 1,314 | SFPU fragment source; the instruction-count static_asserts that pin the emitted windows live at `src/brisc.cpp:65-69` | Ch3, Ch7 |
| [`tools/sfpu.py`](../../../regen-glm5.3-flash/tools/sfpu.py) | 1,895 | The fragment compiler: emits SFPU replay windows into `kernels_sfpu.inc` | Ch3, Ch7 |
| [`src/requant.cpp`](../../../regen-glm5.3-flash/src/requant.cpp) + [`src/requant.h`](../../../regen-glm5.3-flash/src/requant.h) | 1,138 + 115 | TTQ requant pipeline: BFP4/BFP8 strip classes, imatrix exponents, hot-expert selection | Ch6 |
| [`src/glm.h`](../../../regen-glm5.3-flash/src/glm.h) / [`src/glm.cpp`](../../../regen-glm5.3-flash/src/glm.cpp) | 154 / 490 | The CPU reference model — the correctness oracle | Ch6 |
| [`src/main.cpp`](../../../regen-glm5.3-flash/src/main.cpp) / [`src/serve.cpp`](../../../regen-glm5.3-flash/src/serve.cpp) | 1,445 / 5,200 | Host driver: record-list builder, rounds construction, serving loop, sessions | Ch6 |
| [`src/brisc.ld`](../../../regen-glm5.3-flash/src/brisc.ld) + [`src/brisc_entry.S`](../../../regen-glm5.3-flash/src/brisc_entry.S) | 29 + 12 | Device image layout; the 88 KB linker-asserted budget; entry code | Ch7 |
| [`src/ckernel_ops.h`](../../../regen-glm5.3-flash/src/ckernel_ops.h) | 1,072 | Auto-generated op/ckernel instruction tables — the auditable menu op bodies compose from | Ch3 |
| [`rules.py`](../../../regen-glm5.3-flash/rules.py) / [`make.py`](../../../regen-glm5.3-flash/make.py) | 206 / 617 | Build recipes: sfpi pin, march/mabi flags, image assembly | Ch7 |
| [`src/tt_backend.h`](../../../regen-glm5.3-flash/src/tt_backend.h) | 334 | Backend interface types | Ch6 |
| [`src/topology.h`](../../../regen-glm5.3-flash/src/topology.h) / [`src/topology.cpp`](../../../regen-glm5.3-flash/src/topology.cpp) | 40 / 197 | Device enumeration and topology | Ch5, Ch6 |
| [`src/tt_matvec.cpp`](../../../regen-glm5.3-flash/src/tt_matvec.cpp) | 366 | The CPU proxy matvec used by host-side checks | Ch6 |
| [`src/ioctl.h`](../../../regen-glm5.3-flash/src/ioctl.h) | 561 | KMD ioctl definitions | Ch6 |
| [`tests/tt_kernels.cpp`](../../../regen-glm5.3-flash/tests/tt_kernels.cpp) | 45 | Front end of the smoke/bench driver: dispatches into the backend's `run_tt_kernel_check` / `run_tt_bench` | Ch3, Ch6 |
| [`README.md`](../../../regen-glm5.3-flash/README.md) / [`docs/README.md`](../../../regen-glm5.3-flash/docs/README.md) | — / 808 | Run/serve instructions and the measured tables ("Performance", "Weights", "Roofline") / build-run documentation | Ch1, Ch6, Ch7 |

`src/` also holds host-side API surface files (`chat.cpp`, `sampling.cpp`, `grammar.cpp`, `session.cpp`, `gguf.cpp`, `model.cpp`, `media*.cpp`, `qwen.cpp` and siblings) that serve the chat/OpenAI-compatible layer; **skip** them — they are outside this guide's primary-evidence set except where Ch6 discusses the serving loop.

## Verified anchors

Every anchor below was re-verified against the tree while writing this chapter — cite these line numbers as-is; do not shift them without re-checking:

| Anchor | Location |
| --- | --- |
| Peer BAR windows (`TT_PEER_ROUND_WINDOW` 0x2600000, `TT_PEER_DRAM_WINDOW` 0x2A00000) | `src/constants.h:146, 149` |
| `TtGlmStrips` | `src/constants.h:2167` |
| `TT_COMMAND_GLM_MATVEC` | `src/constants.h:2325` |
| `TtGlmSteps` | `src/constants.h:2404` |
| `TT_COMMAND_GLM_STEPS` (command id 124) | `src/constants.h:2403` |
| Sync classes (`TT_GLM_SYNC_MET` / `TT_GLM_SYNC_CHIP`) | `src/constants.h:2412-2413` |
| `TtGlmStepRecord` | `src/constants.h:2414` (its 256-byte size is asserted at `src/constants.h:2428`) |
| Record-ring contract comment (`window` flag) | `src/constants.h:2385-2388` |
| `TT_BFP8_STRIP_BYTES` | `src/constants.h:1502` |
| `TtEthMsg` | `src/constants.h:1730` |
| `TT_GLM_STEP_WINDOW_ADDR` / `TT_GLM_STEP_RING` | `src/constants.h:3345-3346` |
| `NOC1_BASE` (0xFFB30000) | `src/brisc.cpp:694` (NOC0's 0xFFB20000 lives in `src/fw.h:16`) |
| `glm_step_ring_read` (record read-ahead) | `src/brisc.cpp:21047` |
| `run_command` dispatch (decode image; the prefill image's copy is at :16256) | `src/brisc.cpp:16085, 21137` |
| `glm_step` | `src/brisc.cpp:21108` |
| `glm_step_list` (the interpreter's list walk) | `src/brisc.cpp:21153` (its doc comment is at `21151-21152`) |
| `glm_steps` (device token loop) | `src/brisc.cpp:21182-21191` |
| `glm_batch` | `src/brisc.cpp:21218` |
| Instruction-count static_asserts (SFPU replay windows) | `src/brisc.cpp:65-69` |
| Image-size ASSERT (88 KB ceiling) | `src/brisc.ld:28` |
| Stack pointer (`li sp, 0xFFB02000`) | `src/brisc_entry.S:9` |
| Quantization-plan comment + `GlmQuant` | `src/requant.cpp:115-117` / `:118` |
| sfpi toolchain pin | `rules.py:12` |

## The notes corpus

The corpus is the primary measured evidence. Counts, verified by listing the tree [FACT, notes/]: **81 `.md` files in all — 74 top-level (62 `glm-*`, 10 `qwen-*`, and 2 others: `noc-ids.md` and `prefill-design.md`), 5 under `post-rc/` (`glm-rc1.md`, `glm-serve.md`, `glm-integ.md`, `glm-moe.md`, `glm-prefill.md`), and 2 under `pending/` (`glm-crash1.md`, `glm-dram-noc.md`)**. `notes/roofline/` holds 16 Python analysis scripts (no notes); `notes/pending/` holds open investigations, not settled evidence.

Which chapters cite which (core files):

| Notes | Cited by |
| --- | --- |
| [`README.md`](../../../regen-glm5.3-flash/README.md) (repo root) — performance, weights, roofline, capacity, smoke | Ch1, Ch6, Ch7 |
| [`notes/glm-inventory.md`](../../../regen-glm5.3-flash/notes/glm-inventory.md), [`notes/glm-scores.md`](../../../regen-glm5.3-flash/notes/glm-scores.md) — per-op byte inventories, packing, layer map | Ch1, Ch4, Ch6 |
| [`notes/glm-decode-roofline.md`](../../../regen-glm5.3-flash/notes/glm-decode-roofline.md) — DRAM windows, per-op time table, sync chains, idle map | Ch1, Ch2, Ch4 |
| [`notes/glm-decode-ops.md`](../../../regen-glm5.3-flash/notes/glm-decode-ops.md) — the op-by-op optimization ledger (9.3 → 6.05 ms) | Ch4 |
| [`notes/glm-batch.md`](../../../regen-glm5.3-flash/notes/glm-batch.md), [`glm-glx-layouts.md`](../../../regen-glm5.3-flash/notes/glm-glx-layouts.md), [`glm-glx-mtp2.md`](../../../regen-glm5.3-flash/notes/glm-glx-mtp2.md), [`glm-glx-mtp3.md`](../../../regen-glm5.3-flash/notes/glm-glx-mtp3.md) — batching and MTP economics | Ch4 `05` |
| [`notes/noc-ids.md`](../../../regen-glm5.3-flash/notes/noc-ids.md) — NoC-ID budgeting | Ch4 `03` |
| [`notes/glm-ccl.md`](../../../regen-glm5.3-flash/notes/glm-ccl.md), [`glm-eth.md`](../../../regen-glm5.3-flash/notes/glm-eth.md), [`glm-glx-ethdec.md`](../../../regen-glm5.3-flash/notes/glm-glx-ethdec.md) — transports and latencies | Ch5 |
| [`notes/glm-weights.md`](../../../regen-glm5.3-flash/notes/glm-weights.md), [`glm-bfp8.md`](../../../regen-glm5.3-flash/notes/glm-bfp8.md), [`glm-layout.md`](../../../regen-glm5.3-flash/notes/glm-layout.md) — TTQ, strip formats, replication | Ch6 |
| [`notes/post-rc/glm-rc1.md`](../../../regen-glm5.3-flash/notes/post-rc/glm-rc1.md) — RC1 headline numbers | Ch1, Ch4, Ch6 |
| `glm-stability.md`, `glm-integration.md`, `glm-kda.md`, `glm-mhc-moe.md`, `glm-mla.md`, `glm-perf-model.md` — per-subsystem stability, integration and op notes | Ch3, Ch4, Ch5 |
| [`docs/README.md`](../../../regen-glm5.3-flash/docs/README.md), [`docs/router-top4.md`](../../../regen-glm5.3-flash/docs/router-top4.md) | Ch6, Ch7 |
| [`notes/prefill-design.md`](../../../regen-glm5.3-flash/notes/prefill-design.md) | out of guide scope (decode only); named here only as the fence |
| `notes/qwen-*` (10 files) | predecessor Qwen engine notes; not primary evidence for this guide — **skip** unless a GLM note cites one |

The remaining top-level `glm-*` files (Galaxy operations, serving audits, incidents, worklists) are cited by the chapter that needs each one; nothing in this guide depends on reading them first.

## What you can safely never read

- **The bench harness and proxy/sim paths in `src/tt_backend.cpp`** — they exist to drive standalone benchmarks, the bit-accurate host-side device proxy, and the simulated backend (`tests/tt_kernels.cpp`'s `LIBTTSIM` mode); the GLM decode path never executes the proxy or sim branches, so a port contributor can skip them without losing any GLM evidence [INFER, src/tt_backend.cpp]. The smoke gate's *results* are evidence (Ch3 `05`); the harness itself is not.
- **`notes/pending/`** — working notes, not part of the citable evidence base.
- **Vision code** (`vision*`) — the multimodal encoder runs as a separate worker; checkpoint import feeds `requant` only.
- **Prefill-phase design** (`notes/prefill-design.md`, the `TtGlmPrefill*` structs and prefill image branches) — explicitly out of this guide's scope; decode-only readers skip them.

What you cannot skip: `src/brisc.cpp`'s anchor neighborhoods (via the anchor table above, never linearly), `src/constants.h`'s struct family, and the four core notes (`glm-decode-roofline.md`, `glm-decode-ops.md`, `glm-inventory.md`, `post-rc/glm-rc1.md`).

## Navigating the big files

Never read `src/brisc.cpp` or `src/tt_backend.cpp` start to finish. The recipe that works: search the anchor symbol, read the doc comment immediately above the definition, then the body — e.g. `glm_step_list`'s comment at `brisc.cpp:21151-21152` explains the ring contract before the definition at `brisc.cpp:21153`. Good searches from this chapter: `TtGlmStepRecord` in `src/constants.h`, the op tables in `src/ckernel_ops.h`, `GlmQuant` in `src/requant.cpp`, `TtEthMsg` in `src/constants.h`.

---

← [`03-workload-context.md`](./03-workload-context.md) | ↑ [`index.md`](./index.md) | ↑ [guide root](../index.md)

**Next:** [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md)
