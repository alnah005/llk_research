# Chapter 7: Transport and Tools

[Prev: Ch6 -- Lifecycle, IS Integration, and KV Tiering](../ch6_lifecycle_is_integration_kv_tiering/index.md) | [Next: Ch8 -- Build System and Design Decisions](../ch8_build_system_and_design_decisions/index.md)

---

This chapter covers the practical tooling layer of tt-llm-engine: the programs, kernels, and transport integrations that exercise the scheduling stack outside of a full model deployment. The tools form a progression -- each level adds capabilities on top of the previous, while sharing common infrastructure patterns (SocketConfig, H2D/D2H socket setup, the DecodeScheduler API, and the wire format documented in [Chapter 5](../ch5_pipeline_and_wire_format/wire_format.md)). By reading the tools in order, you see the lifecycle protocol from [Chapter 6](../ch6_lifecycle_is_integration_kv_tiering/session_state_machine.md) exercised at increasing levels of realism.

## The Complexity Ladder

```
                              COMPLEXITY LADDER
                              =================

  Level 4: Mooncake Transport          Network-layer transport for
           (TCP / RDMA)                distributed / disaggregated
           No scheduler dependency     deployment (future KV migration)
           +--------------------------+
           | TestEngine RAII wrapper   |
           | TCP + RDMA backends       |
           | Mooncake Store Put/Get    |
           +--------------------------+

  Level 3: DeepSeek Inference Runner   Production inference with real
           (deepseek_inference_runner)  tokenization, batching, multi-turn,
           +--------------------------+ speculative decode
           | tokenizers-cpp (Rust)     |
           | DeepSeek chat template    |
           | Batch processing          |
           | Pipelined teardown        |
           +--------------------------+

  Level 2: Loopback Benchmark          First full-stack test: launcher +
           (launcher + connector)       connector exercising the real
           +--------------------------+ DecodeScheduler over device sockets
           | MeshDevice + H2D/D2H     |
           | SocketPipeline backend    |
           | Multi-user, multi-turn    |
           | TTFT/ITL/TPOT metrics     |
           +--------------------------+

  Level 1: Loopback Kernel             Minimal echo pipeline -- the
           (pipeline_loopback.cpp)      "hello world" of tt-llm-engine
           +--------------------------+ device interaction
           | Read H2D page            |
           | Compute token response   |
           | Write D2H page           |
           | Sentinel exit protocol   |
           +--------------------------+
```

## Tool Capability Comparison

| Capability | Loopback Kernel | Loopback Benchmark | DeepSeek Runner | Mooncake Transport |
|---|---|---|---|---|
| **Where it runs** | On-device (Tensix) | Two host processes + device | Host process + device | Host only (no device) |
| **Wire format** | Non-DeepSeek (default) | Non-DeepSeek (default) | DeepSeek MD format | N/A (raw DMA) |
| **DecodeScheduler** | No | Yes (via SocketPipeline) | Yes (via SocketPipeline) | No |
| **Multi-user** | No (single-stream) | Yes (configurable) | Yes (configurable) | N/A |
| **Multi-turn** | No | Yes (CONTINUE) | Yes (CONTINUE) | N/A |
| **Real tokenization** | No | No (synthetic tokens) | Yes (tokenizers-cpp) | N/A |
| **Speculative decode** | No | No | Yes (configurable per-user) | N/A |
| **Batching** | No | No | Yes (pipelined teardown) | N/A |
| **Benchmark output** | No | Yes (TTFT/ITL/TPOT) | Yes (TTFT/ITL/TPOT + spec stats) | N/A |

## Reading Paths

**Production user**: Start with [deepseek_inference_runner.md](deepseek_inference_runner.md), which covers real model inference with tokenization, multi-turn conversations, and benchmarking. This is the primary entry point for anyone running DeepSeek on Tenstorrent hardware.

**Test/development**: Start with [loopback_kernel.md](loopback_kernel.md) to understand the simplest possible pipeline backend, then read [loopback_microbenchmark.md](loopback_microbenchmark.md) for the two-process testing setup. The loopback tools are the predecessor to the DeepSeek runner and share much of its structure.

**Transport/networking**: Read [mooncake_transport.md](mooncake_transport.md) for the standalone Mooncake integration that will eventually underpin distributed KV cache migration (planned, documented in Ch6's KV tiering section).

## Relationship to Prior Chapters

| Concept | Source Chapter | How This Chapter Uses It |
|---------|---------------|--------------------------|
| System architecture, IS/PM ownership | Ch1 | Tools implement the IS-side of the three-queue model |
| PipelineInterface, SocketPipeline | Ch5 | Connector and runner instantiate DecodeScheduler with SocketConfig |
| Wire format (default + DeepSeek) | Ch5 | Loopback kernel implements the default wire format; DeepSeek runner uses `use_deepseek_md_format=true` |
| ALLOCATE/SUBMIT/CONTINUE/CANCEL lifecycle | Ch6 | All tools drive the full request lifecycle |
| Multi-turn CONTINUE semantics | Ch6 | Connector and runner issue CONTINUE after each turn's completion |
| Deferred cancellation, pipelined teardown | Ch6 | DeepSeek runner's `allocate_one_with_eviction` drains prior-batch CANCELs |
| Speculative decode | Ch4 | DeepSeek runner assigns spec-decode users and reports accept/reject stats |

## Chapter Files

| File | Description |
|---|---|
| [loopback_kernel.md](loopback_kernel.md) | Level 1: The on-device echo kernel -- simplest possible device interaction |
| [loopback_microbenchmark.md](loopback_microbenchmark.md) | Level 2: The loopback benchmark -- full-stack test with launcher + connector |
| [deepseek_inference_runner.md](deepseek_inference_runner.md) | Level 3: Production DeepSeek inference with tokenization, batching, spec decode |
| [mooncake_transport.md](mooncake_transport.md) | Level 4: Mooncake transport integration for distributed deployment |

## Source Files

| Path | Role |
|------|------|
| `kernels/pipeline_loopback.cpp` | On-device loopback kernel |
| `tools/pipeline/dummy_pipeline_launcher.cpp` | Kernel launcher process |
| `tools/pipeline/dummy_pipeline_connector.cpp` | Scheduler connector process |
| `tools/decode/deepseek_inference_runner.cpp` | DeepSeek inference runner |
| `include/tt_llm_engine/transport/mooncake/mooncake_test_engine.hpp` | TestEngine RAII wrapper |
| `tests/transport/mooncake/test_mooncake_transport.cpp` | Transport test suite |
| `tests/transport/mooncake/test_mooncake_store.cpp` | Store test suite |
| `tests/transport/mooncake/mooncake_test_fixture.hpp` / `.cpp` | Shared test fixture |
| `tests/transport/mooncake/mooncake_test_port_util.hpp` | Ephemeral endpoint generator |
| `tests/transport/mooncake/mooncake_transport_test_policies.hpp` | TCP/RDMA test policy traits |
| `cmake/mooncake.cmake` | CMake integration for Mooncake submodule |
| `docs/transport/mooncake.md` | User-facing Mooncake setup docs |
| `run_mooncake_tests.sh` | Test runner script |

---

[Prev: Ch6 -- Lifecycle, IS Integration, and KV Tiering](../ch6_lifecycle_is_integration_kv_tiering/index.md) | [Next: Ch8 -- Build System and Design Decisions](../ch8_build_system_and_design_decisions/index.md)
