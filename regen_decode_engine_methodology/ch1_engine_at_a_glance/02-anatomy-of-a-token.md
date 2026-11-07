# Anatomy of a Token

One decode token walked end-to-end at 50,000 ft: the host hands over an embedding, leads fetch records through a read-ahead ring, op bodies run the 45 layers with cross-chip rounds interleaved, and the head emits a token — all on the device. This file locates every subsystem on that walk and states, at roofline level, where the 6.05 ms goes; Chapters 2–5 each open by finding their subsystem here.

**Scope:** shared foundation (no question owned). The roofline decomposition is cited here as a pointer-style preview; the full exposition — per-op table, ledger, window rates — is owned by [Ch4](../ch4_dram_streaming_discipline/index.md).

## The walk at 50,000 ft

**0. Host edge (once per token, off the device loop).** The embedding row is read on the host and costs no device DRAM [FACT, notes/glm-inventory.md "Packing assumed" (line 23)]; at the roofline-note baseline the host edges of a token cost about 0.05 ms embedding, 0.09 ms input streams, and 0.05 ms post and argmax read [MEAS, notes/glm-decode-roofline.md "Headline" (line 13)]. The record lists for the whole decode step are already resident from load time; the host's per-token involvement ends at handing the token's inputs to the device (the exact boundary is Ch2 `05`'s elimination argument; the builder that produced the lists is Ch6 `04`'s).

**1. Record fetch.** Each lead reads its next `TtGlmStepRecord` from its own DRAM channel into an L1 **[ring](./index.md)** at `TT_GLM_STEP_WINDOW_ADDR`, of `TT_GLM_STEP_RING` = 2 × `TT_GLM_STEP_WINDOW` slots [FACT, src/constants.h:3345-3346]; `glm_step_ring_read` fills the ring ahead of the interpreter [FACT, src/brisc.cpp:21047]. A record with its `window` flag set requests every record up to the ring's capacity past itself before its op runs [FACT, src/constants.h:2388], so the fetch of step $i+1$ overlaps the op of step $i$ inside `glm_step_list` [FACT, src/brisc.cpp:21153].

**2. Decode and dispatch.** The interpreter decodes the record's command id and dispatches to its **[op body](./index.md)** through `run_command` [FACT, src/brisc.cpp:21137]. Each record carries its own sync class — none (`TT_GLM_SYNC_MET`), or a chip barrier (`TT_GLM_SYNC_CHIP`, an `allgather_barrier` of the chip's leads) [FACT, src/constants.h:2412-2413] — and may broadcast its input to every helper of the lead's team. The op bodies are raw MMIO compositions; the model semantics behind them are [`03-workload-context.md`](./03-workload-context.md)'s.

**3. The layers — 45 of them.** For each layer of the 45-layer stack [FACT, notes/glm-inventory.md "Layer map"], the list runs the same shape:

- **mHC site before attention** — one of 90 mHC blocks per token (two per layer), over the `TtGlmHc*` struct family (`TtGlmHcSite` at `src/constants.h:2673`) [FACT, src/constants.h:2632-2703], each with its own record round.
- **Attention**, by layer kind:
  - **KDA layer (34 layers):** projections (`TT_COMMAND_GLM_MATVEC` at `src/constants.h:2325` [FACT, src/constants.h:2325]) — q/k/v/b/f_a/g_a, o_proj matvecs streamed from DRAM — the gated recurrence over per-head state, and the heads round across chips.
  - **MLA + DSA layer (11 layers):** low-rank projections, query build, the absorbed-attention flash (`TT_COMMAND_GLM_MLA_FLASH` at `src/constants.h:2555` [FACT, src/constants.h:2555]) over the latent cache, the value path, o_proj; on the index list, the query gather and `GLM_INDEX_TOP` per MLA layer.
- **mHC site before the FFN** (with the expand).
- **The FFN**, by layer kind:
  - **Dense SwiGLU (layers 0-2).**
  - **MoE (42 layers):** router matvec → Ra round (router logits) → top-8 routing on the leads → routed gate/up + shared expert → Rb round (shared + routed hidden) → routed downs → combine (`TT_COMMAND_GLM_MOE` at `src/constants.h:2743` [FACT, src/constants.h:2743]).

**4. Cross-chip rounds, throughout.** The census of op-internal rounds in one token: mHC site record rounds ×90, KDA heads ×34, MoE Ra and Rb (router logits / shared + routed hidden) ×42 each, and `GLM_ALLGATHER` ×27 (MLA low rank and heads, dense hidden, head streams and candidates; `TT_COMMAND_GLM_ALLGATHER` at `src/constants.h:2473` [FACT, src/constants.h:2473]); the index list adds the query gather and `GLM_INDEX_TOP`'s record round per MLA layer [FACT, notes/glm-decode-roofline.md §2 (lines 68-70)].

**5. Head and emit.** The final norm feeds the untied LM head ([154880, 4096]) [FACT, notes/glm-inventory.md "Head and embedding" (line 112)] matvec streaming over the chips, then argmax picks the token on a lead (`TT_COMMAND_GLM_ARGMAX` at `src/constants.h:2483` [FACT, src/constants.h:2483]). The chosen token's row moves to host memory via `TT_COMMAND_GLM_MOVE` with `TT_GLM_TO_HOST` [FACT, src/constants.h:2457-2465], and the interpreter moves to the next token's first record with no phase break. Per-token state updates land alongside: KDA recurrence state, MLA latents, and pooled indexer keys [EST, notes/glm-inventory.md "Per-user cache and state"].

**6. Loop.** `glm_steps` iterates the token count on the device — `for (uint32_t t = 0; t < tokens; ++t) glm_step_list(...)` [FACT, src/brisc.cpp:21189-21190]. No host round trip exists between tokens.

## Two step-list variants

The host emits two list variants and the engine switches by context: the **dense list** at short context, and the **index list** once $P \ge 2051$ selected positions, when MLA's selection reads the top-512 pooled keys — 2048 positions plus the open tail of up to 3, at most 2051 selected positions in all — rather than a dense list [FACT, notes/glm-inventory.md "Per-user cache and state" (line 163)]. The index list adds the query gather and `GLM_INDEX_TOP` per MLA layer, so every structural count below differs per variant [FACT, notes/glm-decode-roofline.md §2].

## The sync chain per token

Stated per chain class, per list variant — both lists exist in production, so a count quoted for one variant alone would be wrong for the other [FACT, notes/glm-decode-roofline.md §2 (lines 58-66)]:

| Chain class | Dense list | Index list ($P \ge 2051$) |
| --- | ---: | ---: |
| `GLM_STEPS` steps | 561 | 693 |
| Step chip barriers (allgather_barrier of the 8 leads) | 561 | 693 |
| Step mesh syncs (0-byte exchange_across_chips) | 194 | 216 |
| Op-internal global rounds | 235 | 257 |
| Cross-chip PCIe rounds in all | 429 | 473 |

## Where the 6.05 ms goes — preview, owned by Ch4

At roofline level, the token is a byte-moving problem with a compute tail:

- **Bytes.** `hot3-meas` decode reads 2.10 GB per chip per token — 16.8 GB across the box; at long context the rows count weights only, so they are lower bounds [MEAS, README.md "Roofline"].
- **The rate and the plan's constraint.** Inside DRAM-streaming windows the engine sustains 447 GB/s per chip; the plan's DRAM-busy constraint and the full window-rate exposition are [Ch4 `01-roofline-and-byte-budget.md`](../ch4_dram_streaming_discipline/01-roofline-and-byte-budget.md)'s (envelope rows 5-6 in [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md)).
- **The baseline decomposition.** Where the pre-optimization baseline token went, per component — the only per-component split the notes measure; regen's per-step firmware profiling of the 9.30 ms `plan0h` baseline device token [MEAS, notes/glm-decode-roofline.md §3 (lines 88-99)]:

| Component (count per token) | Measured ms |
| --- | ---: |
| MoE layers (42) | 2.559 |
| KDA layers (34) | 2.099 |
| mHC sites (90) | 1.785 |
| MLA + indexer layers (11) | 1.673 |
| mHC expand (90) | 0.503 |
| Dense FFN layers (3) | 0.261 |
| Head + argmax (1) | 0.245 |
| **Component total** (plus ~0.17 ms write-backs and first-step skew → 9.30 ms device) | **9.124** |

- **How the gap was closed.** The optimization ledger took the engine from 9.3 ms/token to 6.05 ms/token (base 9295/9925 us, `plan0h`/`hotE`, down to 8312/9021 us through the named change table) [MEAS, notes/glm-decode-ops.md "Changes" table] — the ledger itself is [Ch4 `04-optimization-ledger.md`](../ch4_dram_streaming_discipline/04-optimization-ledger.md)'s subject and is not restated here.

## Where the chapters land on this walk

| Walk stage | Chapter that owns it |
| --- | --- |
| 0 — host edge: record builder, weights load, admission | [Ch6](../ch6_host_driver_weights_oracle/index.md) |
| 1 — record fetch, ring, interpreter loop, teams | [Ch2](../ch2_record_driven_execution/index.md) |
| 2 — op bodies over raw MMIO, SFPU replay windows | [Ch3](../ch3_raw_mmio_sfpu_replay/index.md) |
| 2-3 — weight streams inside the ops, strip claims, prefetch windows, NoC-ID budget | [Ch4](../ch4_dram_streaming_discipline/index.md) |
| 4 — rounds, transports, latency classes | [Ch5](../ch5_cross_chip_transports/index.md) |
| 5 — oracle checking of what the head emits | [Ch6 `05-cpu-oracle.md`](../ch6_host_driver_weights_oracle/05-cpu-oracle.md) |
| all — the image that runs it, and its budgets | [Ch7](../ch7_build_determinism/index.md) |
| all — porting the walk to tt-metal | [Ch8](../ch8_tt_metal_port/index.md) |

---

← [`01-mental-model.md`](./01-mental-model.md) | ↑ [`index.md`](./index.md) | ↑ [guide root](../index.md)

**Next:** [`03-workload-context.md`](./03-workload-context.md)
