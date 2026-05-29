# Chapter 3: PipelineSimulator Timing Model

This chapter provides a deep dive into the per-token timestamp simulation engine
that models systolic pipeline latency, throughput cap, and backpressure without a
tick thread.  The `PipelineSimulator` is the primary backend for benchmarking
Pi0.5 decode throughput without access to physical hardware.

## Sections

1. [Simulator Design](01_simulator_design.md)
   PipelineSimulator class architecture: the per-token timestamp model, three
   timing invariants, `inject()` and `read_result()` hot paths, the rationale
   for eliminating the tick thread (with a comparative analysis of tick-thread,
   `sleep_until`, and busy-wait approaches), and `PipelineSimulatorConfig`
   parameter mapping.

2. [Batch Prefill Behavior](02_batch_prefill_behavior.md)
   The `batchPrefill` flag and its effect on non-last prefill tokens, the
   `EMPTY_TOKEN` sentinel definition, why Pi0.5 hardware requires this mode
   (with contrast to the generic `simulated_pipeline_runner.cpp`), how it
   affects TTFT measurement and backpressure, and the config propagation path
   through `DecodeScheduler`.

3. [Token Model and Speculative Decode](03_token_model_and_spec_decode.md)
   The synthetic token generation model (accept/reject paths),
   `safeVocabModulus` arithmetic for long-generation safety (with formal
   verification and worked example), fixed `tokenId` deterministic mode,
   relevance to Pi0.5's single-output-token operation, condition variable
   notification in `read_result()`, and a summary table of all token generation
   modes.

---

**Previous:** [Chapter 2 -- JSON Configuration Schema and Parsing](../ch2_json_configuration_schema_and_parsing/index.md)

**Next:** [Chapter 4 -- Bulk H2D/D2H Socket Protocol](../ch4_bulk_h2d_d2h_socket_protocol/index.md)
