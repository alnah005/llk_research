# Comprehensive Guide to tt-llm-engine

This guide provides a complete, source-level walkthrough of [tt-llm-engine](https://github.com/tt-asaigal/tt-llm-engine), the host-side control-plane scheduling engine for streaming LLM inference on Tenstorrent hardware. It covers every subsystem from architectural foundations through threading models, scheduling algorithms, speculative decode, pipeline abstractions, session lifecycle, transport tooling, and build system design decisions.

The guide is written for systems engineers and infrastructure developers who build, extend, or integrate with production LLM inference serving stacks. Readers should have working knowledge of C++20, multi-threaded systems programming, and LLM inference concepts. No prior knowledge of the tt-llm-engine codebase or Tenstorrent-specific APIs is assumed.

---

## Chapters

| # | Chapter | Focus | Key Topics |
|---|---------|-------|------------|
| 1 | [Architecture and Layering](ch1_architecture_and_layering/index.md) | Foundations | System position, IS/PM ownership contract, component map, six key invariants |
| 2 | [Threading and Data Structures](ch2_threading_and_data_structures/index.md) | Internals | Three-thread model, FreeIdPool, UserTable, DecodeStaging, BoundedQueue, lock-free design |
| 3 | [Scheduling Algorithm](ch3_scheduling_algorithm/index.md) | Algorithm | Writer priority, chunked prefill, decode loopback, completion and EOS write-back |
| 4 | [Speculative Decode](ch4_speculative_decode/index.md) | Protocol | MTP pair injection, four result cases, SpecDecodeState machine, throughput analysis |
| 5 | [Pipeline and Wire Format](ch5_pipeline_and_wire_format/index.md) | Abstraction | PipelineInterface, MockPipeline, PipelineSimulator, SocketPipeline, 256-byte page layout |
| 6 | [Lifecycle, IS Integration, and KV Tiering](ch6_lifecycle_is_integration_kv_tiering/index.md) | Integration | UserState FSM, multi-turn CONTINUE, deferred cancellation, three-queue model, KV tiers |
| 7 | [Transport and Tools](ch7_transport_and_tools/index.md) | Tooling | Mooncake transport, loopback kernel/benchmark, DeepSeek inference runner |
| 8 | [Build System and Design Decisions](ch8_build_system_and_design_decisions/index.md) | Build/Design | Two-library architecture, CMake, integration methods, 16 architectural design decisions |

## Reading Paths

### Quick Start (IS Integrators)

For developers integrating with the DecodeScheduler API without needing internal scheduling mechanics:

**Ch1** (full) &rarr; **Ch6** (session_state_machine, multi_turn_continue, is_integration_patterns) &rarr; **Ch8** (build_system, integration_methods)

Covers: architectural context, ALLOCATE/SUBMIT/CONTINUE/CANCEL lifecycle, three-queue communication, build and linking. ~5 content files.

### Deep Dive (Scheduler Developers)

For developers extending the scheduler, adding policies, or debugging concurrency:

**Ch1** &rarr; **Ch2** &rarr; **Ch3** &rarr; **Ch4** &rarr; **Ch5** &rarr; **Ch6** &rarr; **Ch7** &rarr; **Ch8**

Full sequential reading. Each chapter builds on the previous. ~40 content files.

### Hardware/Transport (Kernel and Transport Developers)

For developers working on device kernels, wire format, or transport integration:

**Ch1** (system_position, component_map) &rarr; **Ch5** (full, especially wire_format, socket_pipeline) &rarr; **Ch7** (loopback_kernel, loopback_microbenchmark, mooncake_transport) &rarr; **Ch8** (build_system)

Covers: host-device interface, page layout, socket protocol, loopback reference kernel, hardware build. ~10 content files.

## Question-to-Chapter Map

| Question | Chapter | Key Files |
|----------|---------|-----------|
| Where does tt-llm-engine sit in the stack? | Ch1 | system_position.md, ownership_contract.md |
| How does the three-thread model work? | Ch2 | threading_model.md, lock_free_design.md |
| What data structures does the scheduler use? | Ch2 | free_id_pool.md, user_table.md, decode_staging.md |
| How does the Writer prioritize work? | Ch3 | writer_priority_and_decode.md, chunked_prefill.md |
| How does speculative decode work? | Ch4 | result_classification.md, pair_injection.md, state_machine_detail.md |
| What is the PipelineInterface contract? | Ch5 | pipeline_interface.md |
| What is the host-device wire format? | Ch5 | wire_format.md |
| How does a user session lifecycle work? | Ch6 | session_state_machine.md, deferred_cancellation.md |
| How do I integrate with the three-queue API? | Ch6 | is_integration_patterns.md |
| What is the KV cache tiering design? | Ch6 | kv_cache_tiering.md |
| How does Mooncake transport work? | Ch7 | mooncake_transport.md |
| How do I run the loopback benchmark? | Ch7 | loopback_microbenchmark.md, loopback_kernel.md |
| How do I run DeepSeek inference? | Ch7 | deepseek_inference_runner.md |
| How do I build the project? | Ch8 | build_system.md, integration_methods.md |
| What design tradeoffs were made? | Ch8 | design_decisions.md |

## Terminology

| Term | Definition |
|------|-----------|
| **IS** | Inference Server -- the upstream HTTP/gRPC stack that owns client sessions and tokenization |
| **PM** / **DS** | Pipeline Manager / Decode Scheduler -- the tt-llm-engine scheduling component |
| **Pipeline** | The hardware inference pipeline, modeled as D stages with one-token-per-tick throughput |
| **slot_id** / **user_id** / **uid** | Internal KV cache index allocated by PM. Range [0, max_users). All three names refer to the same integer |
| **D** | Number of pipeline stages (e.g., 64 for 16-device x 4-stage) |
| **TPOT** | Time Per Output Token |
| **TTFT** | Time To First Token |
| **MTP** | Multi-Token Prediction -- the speculative decode mechanism |
| **Hot path** | The Writer and Reader thread loops that run at pipeline tick frequency |

## Conventions

- Source file paths are relative to the tt-llm-engine repository root.
- Memory ordering annotations appear in parentheses: (relaxed), (acquire), (release), (acq_rel).
- Thread ownership is marked with brackets: [API], [Writer], [Reader].
- Features not yet implemented are marked: **Status: Planned**.
- Code blocks use `cpp`, `cmake`, or `bash` language identifiers.

## Methodology

This guide was produced using a deep planning research workflow: 4 competing generators per chapter, 4 independent evaluators, 1 synthesizer selecting the best elements, a factual critic (Agent B) verifying every source reference against the actual codebase, and a compressor (Agent C) checking for redundancy. The synthesized plan that structured all chapters is in [plan.md](plan.md).

---

[Start Reading: Chapter 1 &rarr;](ch1_architecture_and_layering/index.md)
