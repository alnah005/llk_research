# Chapter 2 — The Record-Driven Execution Model

How `TtGlmStepRecord` step lists plus one persistent per-lead BRISC **[interpreter](./02-interpreter-loop.md)** loop — with **[lead](./03-lead-helper-teams.md)**/**[helper](./03-lead-helper-teams.md)** team jobs and DRAM read-ahead **[rings](./04-read-ahead-rings.md)** — turn decoding into data interpretation: the host compiles a whole decode step into a flat per-core list of 256-byte **[records](./01-step-records.md)** once, and the device walks those lists token after token with no phase boundaries, no per-step dispatch, and no host inside the token loop. This chapter owns the record format, the loop chain, the teams, the rings, and the elimination argument that closes them into one thesis.

**Scope:** covers Question 1 (the record-driven execution model) of the guide's seven. On [Chapter 1's token walk](../ch1_engine_at_a_glance/02-anatomy-of-a-token.md) this chapter owns stage 1 (record fetch through the ring) and stage 2 (decode, dispatch, and the team jobs each step fans into), plus the device-side loop that chains tokens. Primary source: the read-only regen tree at `/home/alex/mpi-shfs/tenstorrent/regen-glm5.3-flash` — all `src/...` and `notes/...` citations are relative to that root; links resolve from the guide root per the [Ch1 link convention](../ch1_engine_at_a_glance/index.md). Every line anchor in this chapter was re-verified in-tree before writing.

## The elimination argument, stated as three testable claims

The chapter's thesis is one argument with three parts, each falsifiable against source or measurement, each substantiated in [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md):

| # | Claim | What would falsify it | Established in |
| --- | --- | --- | --- |
| 1 | **No phase boundaries** — records chain continuously inside one `TT_COMMAND_GLM_STEPS` post; sync is per-record data; the device moves from one token's last record to the next token's first with no init/forward break | A per-phase barrier or re-initialization step in the token walk | [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md) §(a) |
| 2 | **No per-step dispatch** — the interpreter is the single resident program per core; nothing is launched per step, and the per-step device bookkeeping is measured at 0.34 us | A launch, post, or host round-trip inside one record's execution | [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md) §(b) |
| 3 | **No host in the token loop** — the host builds/loads records once per configuration, then per token writes the input, posts at most one command, and reads the result; the interior of the loop runs device-alone | Host work proportional to the step count | [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md) §(c) |

## The Ch2–Ch4 boundary (pinned)

This chapter and [Ch4 — DRAM Streaming Discipline](../ch4_dram_streaming_discipline/index.md) deliberately split one mechanism family:

- **Owned here (Ch2 `04`):** the *record ring's* geometry and record-stream prefetch — slot count, window size, lookahead distance, the ring's NoC ID, and claims-ahead pacing at the *record* level.
- **Owned by Ch4, deferred by name:** strip **claims-ahead**, two-strips-in-flight, near-channel reader placement, the NoC-ID *budget* across reader classes, and all DRAM window rates ([Ch4 `02-strip-claims-and-reader-placement.md`](../ch4_dram_streaming_discipline/02-strip-claims-and-reader-placement.md), [Ch4 `03-prefetch-windows-and-noc-id-budget.md`](../ch4_dram_streaming_discipline/03-prefetch-windows-and-noc-id-budget.md)). Ch2 `04` points at these; it never restates them.

Other fences, each deferred by name wherever it appears: op *mechanism* (what each command id computes) is [Ch3 `04-op-bodies.md`](../ch3_raw_mmio_sfpu_replay/04-op-bodies.md)'s; record *production* (the host builder that emits these lists) is [Ch6 `04-record-builder-and-serving.md`](../ch6_host_driver_weights_oracle/04-record-builder-and-serving.md)'s — this chapter owns only the FORMAT (the split pinned in the guide plan); the rounds some records trigger, and every transport latency, are [Ch5](../ch5_cross_chip_transports/index.md)'s; the image budget and toolchain determinism are [Ch7](../ch7_build_determinism/index.md)'s; and the port-side interpretation of all of it is [Ch8](../ch8_tt_metal_port/index.md)'s, which `05` supplies premises to without arguing port decisions.

## Files in this chapter

| File | What it gives you |
| --- | --- |
| [`01-step-records.md`](./01-step-records.md) | The `TtGlmStepRecord` format, field by field, and the records-as-data argument. |
| [`02-interpreter-loop.md`](./02-interpreter-loop.md) | The persistent chain that walks the lists, plus its in-band observability. |
| [`03-lead-helper-teams.md`](./03-lead-helper-teams.md) | Lead/helper roles, the job protocol, the sentinel discipline, share math. |
| [`04-read-ahead-rings.md`](./04-read-ahead-rings.md) | The DRAM record ring: geometry, pacing, window restarts, measured wins. |
| [`05-eliminating-the-runtime.md`](./05-eliminating-the-runtime.md) | The three eliminations argued as one claim against the measured roofline. |

## Evidence conventions

Evidence tags per [Ch1 `05-measured-envelope-and-evidence.md`](../ch1_engine_at_a_glance/05-measured-envelope-and-evidence.md); where the corpus is silent — notably team mechanics — this chapter tags strictly `[INFER, src/...]` and never presents code reading as measurement.

---

**Next:** [`01-step-records.md`](./01-step-records.md)
