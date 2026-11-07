# Workload Context

GLM-5.3-Flash at decode time is a 45-layer hybrid stack — 34 KDA layers, 11 MLA+DSA layers, 42 MoE layers of 288 experts — whose per-token byte budget dwarfs its compute. This file carries the layer map, the BFP strip packing, the per-class byte budget, and the arithmetic that shows decode is DRAM-bandwidth-bound; it is the motivation substrate for [Ch4](../ch4_dram_streaming_discipline/index.md) and the decoder ring for [Ch3](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md)'s op names.

**Scope:** shared foundation (no question owned). Strip GEOMETRY as a reader contract is Ch4 `02`'s; requant-CLASS selection and hot-expert policy are Ch6 `02`'s — this file uses both, links both, restates neither.

## The model at decode time

Layer map [FACT, notes/glm-inventory.md "Layer map" (lines 27-30)]:

| Component | Count / shape | Notes |
| --- | --- | --- |
| Layers | 45 (indices 0-44) | KDA at 0-2, 4-6, ..., 40-42, 44; MLA+DSA at 3, 7, ..., 43 |
| KDA layers | 34 | hidden 4096, 64 heads × 128; depthwise conv q/k/v, kernel 4 |
| MLA + DSA layers | 11 | q_lora 1536, kv_lora 512, qk 256, v 256; indexer selects from pooled keys |
| Dense SwiGLU FFN | 3 (layers 0-2) | intermediate 12288; clamped SwiGLU |
| MoE layers | 42 (layers 3-44) | 288 routed experts, top-8, plus 1 shared expert; width 2048 |
| mHC blocks | 2 per layer = 90 | 4 streams of 4096; Sinkhorn comb |
| Positional encoding | NoPE everywhere | qk_rope_head_dim 0 — no RoPE in MLA or the indexer |
| LM head | untied [154880, 4096] | the head matvec streams across chips [FACT, notes/glm-inventory.md "Head and embedding" (line 112)] |
| Text parameters | 313,326,811,966 (≈313.3B) | matches the BF16 release count when MTP + vision are added [EST, notes/glm-inventory.md (lines 5-7)] |
| Device parameters | 312,692M (≈312.7B) | the text count minus the host-read embedding (634.4M) [EST, notes/glm-inventory.md "Totals" (line 130)] |

The engine checks the geometry it was built against at load: `requant` requires `expert_used_count` = 8, `expert_feed_forward_length` = 2048, `leading_dense_block_count` = 3, and `hyper_connection.count` = 4 from the model metadata [FACT, src/requant.cpp:105-112].

## The MoE shape

Each MoE layer has 288 routed experts of width 2048 with top-8 routing plus one shared expert [FACT, notes/glm-inventory.md "MoE layer" (line 82)]. Routing is sigmoid scores with a correction bias applied for the choice only, top 8, weights taken from the unbiased scores, normalized and scaled by 2.5 [FACT, notes/glm-inventory.md "MoE layer" (lines 92-94)]. Per token per layer only

$$8/288 \approx 2.8\%$$

of experts are active — 201M of 7,248M routed parameters per layer, and 8.46B of 304.4B across the 42 MoE layers per token [FACT, notes/glm-inventory.md "MoE layer" (line 94)].

Two consequences the later chapters cash out:

- Which 8-of-288 subset a token needs is unknown until the router runs — so routed-expert bytes sit on the critical chain and cannot be prefetched ahead of routing [INFER, notes/glm-decode-roofline.md §5]. This is the demand signal Ch4's streaming discipline answers; the batching consequence (routed-expert unions destroying per-strip reuse) is [Ch4 `05-batch-and-mtp-economics.md`](../ch4_dram_streaming_discipline/05-batch-and-mtp-economics.md)'s.
- Everything non-routed is a fixed per-token stream, whatever the routing does — see the byte budget below.

## Strip packing

Weights travel as BFP **[strips](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md)**: 16 rows × K, one shared 8-bit exponent per 16 consecutive K values of a row [FACT, notes/glm-inventory.md "Packing assumed" (lines 13-14)]. The packing arithmetic:

$$\text{BFP4}: \frac{16 \times 4 + 8}{16} = 4.5\ \text{bits/weight} \quad (9/16\ \text{B}), \qquad \text{BFP8}: \frac{16 \times 8 + 8}{16} = 8.5\ \text{bits/weight} \quad (17/16\ \text{B})$$

[FACT, notes/glm-inventory.md "Packing assumed" (lines 15-18)]. Which tensor group gets which class is requant policy — envelope row 11 in [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md), owned by [Ch6 `02-ttq-container-and-requant.md`](../ch6_host_driver_weights_oracle/02-ttq-container-and-requant.md).

## The per-token byte budget

Per-class decode reads under the quantization plan (all-layers-of-kind totals), config-derived arithmetic recorded in the corpus — an estimate, not a silicon measurement [EST, notes/glm-inventory.md "Totals" (lines 118-130)]:

| Class | B/weight | Decode GB/token | Share of decode bytes |
| --- | ---: | ---: | ---: |
| KDA projections | 17/16 | 4.972 | 36.5% |
| Routed experts (8 of 288 per layer) | 9/16 | 4.756 | 35.0% |
| MLA projections | 17/16 | 1.373 | 10.1% |
| Shared expert | 17/16 | 1.123 | 8.3% |
| LM head | 17/16 | 0.674 | 5.0% |
| Dense FFN | 17/16 | 0.481 | 3.5% |
| DSA indexer | 17/16 | 0.087 | 0.6% |
| mHC fn | 2 | 0.071 | 0.5% |
| Router | 17/16 | 0.053 | 0.4% |
| Small parameters | ≤ 4 | 0.016 | 0.1% |
| **Device total (plan)** | | **13.61** | |

Placed on the real 8-chip layout the same budget is 14.10 GB per token — the plan's 13.61 GB plus replicated f_a/g_a, strip padding, and replicated latent reads [EST, notes/glm-inventory.md "Totals" (lines 136-137)]. The loader's own dump of the real placement gives the per-chip figure the roofline uses: 1763.8 MB/token/chip with `plan0h` weights, and `hotE`-class weights (the hot3-meas byte class, its BFP8 hot experts moving 70% of routed reads to BFP8) add +370 MB [MEAS, notes/glm-decode-roofline.md §1 (lines 28-49)].

Note the inversion the table encodes: non-routed classes are 2.6% of parameters but 65% of decode bytes [EST, notes/glm-inventory.md "Totals" (line 135)] — the fixed per-token stream dominates even against a 288-expert MoE, because only 8 experts fire; the routed-expert 35% is the part that depends on routing.

## Why decode is DRAM-bound

The byte side, against the 500 GB/s per-chip peak the roofline analysis uses: the placed bytes per chip give

$$\frac{14.10\ \text{GB}}{8\ \text{chips}} \approx 1.76\ \text{GB/chip/token}, \qquad \frac{1764\ \text{MB}}{500\ \text{GB/s}} = 3.53\ \text{ms (plan0h)}, \qquad \frac{2134\ \text{MB}}{500\ \text{GB/s}} = 4.27\ \text{ms (hotE)}$$

as DRAM floors — arithmetic over measured bytes recorded in the corpus, not itself a measurement [EST, notes/glm-decode-roofline.md "Headline" (lines 16, 18)]. GLM moves 1.76-2.13 GB per chip per token, about $3.5\times$ Qwen's 0.51 GB, so its floor is 3.5-4.3 ms, not 2-2.5 ms [EST, notes/glm-decode-roofline.md "Headline" (lines 23-25)].

The compute side is nowhere near limiting: weight work is 16.74 GMAC/token [EST, notes/glm-inventory.md "Totals" (line 133)] — $2 \times 16.74\ \text{G} = 33.5$ GFLOP — against an 8-chip peak of 5312 TFLOPS [FACT, README.md "Roofline"]:

$$\frac{33.5 \times 10^{9}\ \text{FLOP}}{5312 \times 10^{12}\ \text{FLOP/s}} \approx 6.3\ \mu\text{s}$$

— a factor of ~560 below the 3.53 ms bandwidth floor. Bytes take milliseconds, FLOPs take microseconds, and decode is a DRAM problem. Realized at the measured short-context rate, the box sustains $2.10\ \text{GB} \times 165.9\ \text{t/s} \times 8\ \text{chips} \approx 2787$ GB/s — about 68% of the 4096 GB/s aggregate peak [MEAS, README.md "Roofline"] (arithmetic shown; the inputs carry the citation). This is why the engine's performance methodology is a streaming discipline rather than a matmul-optimization effort ([Ch4](../ch4_dram_streaming_discipline/index.md)).

## Per-user state (the other DRAM term)

Fixed per user: KDA state 34 × 64 heads × 128 × 128 FP32 = 136.0 MiB, KDA conv history 4.78 MiB, indexer tail 16.5 KiB, mHC streams 32 KiB [EST, notes/glm-inventory.md "Per-user cache and state" (lines 151-158)]. Per position: the replicated MLA latent and pooled indexer key add 1088 B/layer, 11,968 B over the 11 MLA layers, replicated to 1419 B per channel per position [EST, notes/glm-inventory.md "Per-user cache and state" (lines 143-149)]. These stay small at short context — latent and index reads add 23 MB/token/chip once positions reach 2048 [MEAS, notes/glm-decode-roofline.md §1] — and they are the term behind the context-decay table in [`05-measured-envelope-and-evidence.md`](./05-measured-envelope-and-evidence.md).

## Decoder ring for the op names

The names the step lists and notes use for the walk's stages, listed here so [Ch3 `04-op-bodies.md`](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md)'s reader already knows the model-side semantics. The component breakdown is the corpus's inventory [FACT, notes/glm-inventory.md]; each op's mechanism is Ch3 `04`'s, its performance Ch4 `04`'s.

- **mHC block**: unweighted RMSNorm over the flattened 16384 (4 streams of 4096), the $[24, 16384]$ fn projection, pre/post sigmoid mixing, a softmax comb, then 20 Sinkhorn passes [FACT, notes/glm-inventory.md "mHC" (lines 103-106)].
- **KDA layer**: q/k/v projections (qkv width 8192), depthwise conv1d (kernel 4), the gated-decay recurrence over a $64 \times 128 \times 128$ FP32 state (about 4 MAC per state element), f_a/g_a and f_b/g_b low-rank parts, o_proj [FACT, notes/glm-inventory.md "KDA layer" (lines 37-51)].
- **MLA + DSA layer**: q_a/q_b low-rank query path, kv_a + kv_b absorbed attention (W_UK^T, W_UV), o_proj, and the DSA indexer — compressor gate, pooled keys; selection capped at 2051 selected positions (see [`02-anatomy-of-a-token.md`](./02-anatomy-of-a-token.md)'s list variants).
- **MoE layer**: router (Ra round, top-8 scan), shared expert, routed gate/up (Rb round), downs, combine.
- **Dense FFN**: the clamped SwiGLU — gate clamped at 10, up clamped to $[-10, 10]$, then $\mathrm{silu}(\mathrm{gate}) \times \mathrm{up}$ [FACT, notes/glm-inventory.md "Dense FFN" (line 80)].
- **Head**: LM head matvec + argmax; **whole step list**: `GLM_STEPS`; **batch/serving path**: `GLM_BATCH`.

---

← [`02-anatomy-of-a-token.md`](./02-anatomy-of-a-token.md) | ↑ [`index.md`](./index.md) | ↑ [guide root](../index.md)

**Next:** [`04-source-map.md`](./04-source-map.md)
