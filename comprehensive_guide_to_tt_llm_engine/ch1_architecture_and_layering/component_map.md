# Component Map

tt-llm-engine is organized into a small number of well-defined components, each occupying a distinct namespace and directory subtree. This file catalogs every component, its implementation status, the two library artifacts produced by the build system, the namespace hierarchy, the physical directory layout, and the key constants that size the system.

## Component Status Table

| Component | Status | Description | Library | Primary Source |
|-----------|--------|-------------|---------|----------------|
| **Decode scheduler** | Available | Per-token decode scheduling, user slot management, chunked prefill interleaving, speculative decode acceptance, EOS/cancellation handling, multi-turn continuation. Three dedicated threads (Writer, Reader, API handler). | Core + Full | `src/scheduler/decode/decode_scheduler.cpp` |
| **Pipeline transport** | Available | Abstract `PipelineInterface` with three implementations: `MockPipeline` (in-process testing), `PipelineSimulator` (timing-accurate model), `SocketPipeline` (real H2D/D2H sockets). | Core (Mock, Simulator) / Full (Socket) | `include/tt_llm_engine/pipeline/` |
| **Mooncake transport** | Available | Standalone transport testing for Mooncake Transfer Engine (TCP, RDMA) and Mooncake Store. No scheduler dependency. | Separate library (`TtLlmEngine::Mooncake::TransferEngine`) | `include/tt_llm_engine/transport/mooncake/`, `tests/transport/mooncake/` |
| **Prefill scheduler** | Planned | Dedicated chunked prefill admission, queueing, and interleaving with in-flight decode. Currently, prefill scheduling is embedded in the decode scheduler's writer loop. | -- | -- |
| **KV-cache migration scheduler** | Planned | Prefill-to-decode KV migration for disaggregated architectures. Hot/warm/cold tiering and eviction of per-user KV caches across memory tiers. API contract specified in `docs/scheduler/decode.md`; not yet implemented in source. | -- | -- |

### How Future Components Slot In

The directory and namespace structure is designed for additive growth. When the prefill scheduler and KV-cache migration scheduler are implemented, they will occupy new directories without restructuring existing code:

```
include/tt_llm_engine/scheduler/
    decode/         <-- exists today
    prefill/        <-- planned
    kv_migration/   <-- planned

src/scheduler/
    decode/         <-- exists today
    prefill/        <-- planned
    kv_migration/   <-- planned
```

Each new component will be a separate shared library target or will be linked into the existing library artifacts, following the same pattern as the decode scheduler.

## Two Library Artifacts

The build system produces two shared libraries with distinct dependency footprints. This split allows the scheduling logic to be developed, tested, and integrated without requiring Tenstorrent hardware or the tt-metalium SDK.

| Library | File | CMake Target | Alias | Dependencies | Description |
|---------|------|-------------|-------|--------------|-------------|
| **Core** | `libtt_llm_engine_core.so` | `tt_llm_engine_core` | `TtLlmEngine::Core` | `Threads::Threads` only | Scheduling logic, `MockPipeline`, `PipelineSimulator`. Always built. No hardware dependencies. |
| **Full** | `libtt_llm_engine.so` | `tt_llm_engine` | `TtLlmEngine::Full` | `TT::Metalium`, `Threads::Threads` | Everything in Core plus `SocketPipeline`, which communicates with Tenstorrent hardware via tt-metalium H2D/D2H sockets. Defines `DS_HAS_METALIUM=1`. Only built when `TT::Metalium` is available. |

The split serves two purposes:

1. **Development velocity.** The Core library builds and tests without Tenstorrent hardware or the tt-metal toolchain. You can run the full mock test suite on any Linux machine with a C++20 compiler.

2. **Deployment flexibility.** Upstream projects that only need the scheduling algorithm (e.g., for simulation, benchmarking, or integration testing) link `TtLlmEngine::Core`. Production deployments that drive real hardware link `TtLlmEngine::Full`.

Both libraries export the same decode scheduler source (`src/scheduler/decode/decode_scheduler.cpp`). The Full library additionally compiles `src/pipeline/socket_pipeline.cpp` and links against `TT::Metalium`. When linking against `TtLlmEngine::Full`, the preprocessor macro `DS_HAS_METALIUM=1` is defined automatically, enabling conditional compilation:

```cpp
#ifdef DS_HAS_METALIUM
#include <tt_llm_engine/pipeline/socket_pipeline.hpp>
// SocketPipeline available
#else
#include <tt_llm_engine/pipeline/pipeline_interface.hpp>
// MockPipeline or PipelineSimulator only
#endif
```

The pipeline backend is selected at construction time via a `PipelineConfig` variant:

```cpp
using PipelineConfig = std::variant<MockConfig, SocketConfig, PipelineSimulatorConfig>;
```

The `DecodeScheduler` constructor accepts a `PipelineConfig` and uses `std::visit` to instantiate the correct `PipelineInterface` implementation. Attempting to use `SocketConfig` when linked against the Core library triggers an assertion failure at runtime. This means the scheduling algorithm is completely decoupled from the transport layer -- the same scheduling code runs against mock, simulated, or real hardware backends.

For build system details, see [Chapter 8 -- Build System](../ch8_build_system_and_design_decisions/build_system.md).

## Namespace-to-Header Mapping

The C++ namespace structure mirrors the `include/` directory layout. The table below maps each qualified public type to its defining header. Private implementation types (PIMPL internals) are annotated in the Full Directory Layout below.

| Qualified Type | Header |
|----------------|--------|
| `tt_llm_engine::scheduler::decode::DecodeScheduler` | `scheduler/decode/decode_scheduler.hpp` |
| `tt_llm_engine::scheduler::decode::SchedulerParams` | `scheduler/decode/decode_types.hpp` |
| `tt_llm_engine::scheduler::decode::ISRequest` | `scheduler/decode/decode_types.hpp` |
| `tt_llm_engine::scheduler::decode::SchedulerResponse` | `scheduler/decode/decode_types.hpp` |
| `tt_llm_engine::scheduler::decode::OutputMessage` | `scheduler/decode/decode_types.hpp` |
| `tt_llm_engine::scheduler::decode::GenerationParams` | `scheduler/decode/decode_types.hpp` |
| `tt_llm_engine::scheduler::decode::UserState` | `scheduler/decode/decode_types.hpp` |
| `tt_llm_engine::scheduler::decode::RequestType` | `scheduler/decode/decode_types.hpp` |
| `tt_llm_engine::pipeline::PipelineInterface` | `pipeline/pipeline_interface.hpp` |
| `tt_llm_engine::pipeline::InjectDescriptor` | `pipeline/pipeline_types.hpp` |
| `tt_llm_engine::pipeline::ResultDescriptor` | `pipeline/pipeline_types.hpp` |
| `tt_llm_engine::pipeline::TokenType` | `pipeline/pipeline_types.hpp` |
| `tt_llm_engine::pipeline::PipelineConfig` | `pipeline/pipeline_types.hpp` |
| `tt_llm_engine::pipeline::MockConfig` / `SocketConfig` / `PipelineSimulatorConfig` | `pipeline/pipeline_types.hpp` |
| `tt_llm_engine::pipeline::MockPipeline` | `pipeline/mock_pipeline.hpp` |
| `tt_llm_engine::pipeline::PipelineSimulator` | `pipeline/pipeline_simulator.hpp` |
| `tt_llm_engine::pipeline::SocketPipeline` | `pipeline/socket_pipeline.hpp` |
| `tt_llm_engine::pipeline::INVALID_SLOT` / `EMPTY_TOKEN` | `pipeline/pipeline_types.hpp` |
| `tt_llm_engine::transport::mooncake_test::TestEngine` | `transport/mooncake/mooncake_test_engine.hpp` |

Sentinel constants `INVALID_SLOT` and `EMPTY_TOKEN` are defined in `tt_llm_engine::pipeline` and re-exported into `tt_llm_engine::scheduler::decode` via `using` declarations in `decode_types.hpp`.

## Full Directory Layout

Paths under `include/` are the **public API** (stable headers, mirroring the namespace). Paths under `src/` are **private implementation** (not exposed to consumers; included via relative paths within the library). Private headers in `src/` are never installed or exported.

```
tt-llm-engine/
+-- include/tt_llm_engine/            # PUBLIC API
|   +-- tt_llm_engine.hpp                 # Umbrella header
|   +-- pipeline/                         # tt_llm_engine::pipeline
|   |   +-- pipeline_interface.hpp            # PipelineInterface abstract base
|   |   +-- pipeline_types.hpp                # PipelineConfig, InjectDescriptor, ResultDescriptor, TokenType, sentinels
|   |   +-- mock_pipeline.hpp                 # MockPipeline (in-process testing)
|   |   +-- pipeline_simulator.hpp            # PipelineSimulator (timing-accurate model)
|   |   +-- socket_pipeline.hpp               # SocketPipeline (real H2D/D2H, requires DS_HAS_METALIUM)
|   |   +-- wire_format.hpp                   # Host <-> device page serialization
|   +-- scheduler/
|   |   +-- decode/
|   |       +-- decode_scheduler.hpp          # DecodeScheduler (PIMPL)
|   |       +-- decode_types.hpp              # SchedulerParams, ISRequest, SchedulerResponse, OutputMessage, GenerationParams, UserState, RequestType
|   +-- transport/
|       +-- mooncake/
|           +-- mooncake_test_engine.hpp      # Mooncake TestEngine RAII wrapper
|
+-- src/                              # PRIVATE IMPLEMENTATION
|   +-- common/                           # Shared scheduling primitives
|   |   +-- bounded_queue.hpp                 # BoundedQueue<T> (mutex ring buffer)
|   |   +-- free_id_pool.hpp                  # FreeIdPool (multi-word bitmap allocator)
|   |   +-- user_table.hpp                    # UserTable, PromptTable, CancelBitmap
|   +-- pipeline/
|   |   +-- socket_pipeline.cpp               # SocketPipeline::Impl
|   +-- scheduler/
|       +-- decode/
|           +-- decode_scheduler.cpp          # DecodeScheduler::Impl (Writer, Reader, API threads)
|           +-- decode_staging.hpp            # DecodeStaging, DecodeStagingEntry (FIFO + generation counters)
|           +-- prefill_queue.hpp             # PrefillQueue (mutex-protected deque)
|           +-- spec_decode_state.hpp         # SpecDecodeState (reader-thread-only)
|
+-- tests/                            # Test binaries (mirror src/ layout)
|   +-- scheduler/decode/
|   |   +-- test_decode_scheduler.cpp         # Mock-pipeline scheduler tests
|   |   +-- test_decode_scheduler_device.cpp  # Device integration tests
|   +-- transport/mooncake/
|       +-- test_mooncake_transport.cpp       # TCP/RDMA transport tests
|       +-- test_mooncake_store.cpp           # Mooncake Store Put/Get tests
|       +-- mooncake_test_fixture.cpp         # Shared test fixture
|
+-- tools/                            # CLI binaries
|   +-- pipeline/
|   |   +-- dummy_pipeline_launcher.cpp       # Device kernel launcher (loopback)
|   |   +-- dummy_pipeline_connector.cpp      # Scheduler-driven benchmark client
|   +-- decode/
|       +-- deepseek_inference_runner.cpp      # DeepSeek model runner
|       +-- prompts.json                       # Sample multi-turn prompts
|
+-- kernels/                          # Device kernels
|   +-- pipeline_loopback.cpp                 # On-device loopback echo kernel
|
+-- cmake/                            # CMake support files
|   +-- tt-llm-engine-config.cmake.in         # Package config template
|   +-- mooncake.cmake                        # Mooncake submodule build integration
|
+-- docs/                             # Architecture documentation
|   +-- overview.md                           # Engine layering and component overview
|   +-- scheduler/
|   |   +-- decode.md                         # Decode scheduler internals
|   +-- transport/
|       +-- README.md                         # Transport overview
|       +-- mooncake.md                       # Mooncake transport documentation
|
+-- third_party/                      # Vendored dependencies
|   +-- boost/                                # Boost headers (for Mooncake)
|   +-- gflags/                               # Command-line flags
|   +-- glog/                                 # Google logging
|   +-- mooncake/                             # Mooncake Transfer Engine submodule
|   +-- msgpack/                              # MessagePack serialization headers
|   +-- yaml-cpp/                             # YAML parser
|   +-- zstd/                                 # Zstandard compression
|
+-- tt-metal/                         # tt-metal submodule (pinned)
+-- CMakeLists.txt                    # Top-level build configuration
+-- build.sh                          # Convenience build script
+-- setup.sh                          # Combined tt-metal + engine build
+-- run_mooncake_tests.sh             # Mooncake test runner
+-- LICENSE                           # Apache 2.0
```

### Naming Conventions

Include paths mirror qualified C++ names: `#include <tt_llm_engine/scheduler/decode/decode_scheduler.hpp>` exposes `tt_llm_engine::scheduler::decode::DecodeScheduler`. Tests mirror the source tree: `src/scheduler/decode/decode_scheduler.cpp` has its tests at `tests/scheduler/decode/test_decode_scheduler.cpp`.

## Build Targets Summary

| Target | Type | Requires Metalium | Requires Rust | Description |
|--------|------|-------------------|---------------|-------------|
| `tt_llm_engine_core` | Shared library | No | No | Core scheduling library |
| `tt_llm_engine` | Shared library | Yes | No | Full library with SocketPipeline |
| `test_decode_scheduler` | Test binary | No | No | Mock-pipeline scheduler tests |
| `test_decode_scheduler_device` | Test binary | Yes | No | Device integration tests |
| `test_mooncake_transport` | Test binary | No | No | Mooncake transport tests |
| `test_mooncake_store` | Test binary | No | No | Mooncake Store tests |
| `dummy_pipeline_launcher` | Tool | Yes | No | Device loopback kernel launcher |
| `dummy_pipeline_connector` | Tool | Yes | No | Scheduler-driven benchmark client |
| `deepseek_inference_runner` | Tool | Yes | Yes | DeepSeek model runner with tokenizer |

For build system details, see [Chapter 8](../ch8_build_system_and_design_decisions/build_system.md).

## Key Constants and Defaults

These constants, defined in `include/tt_llm_engine/scheduler/decode/decode_types.hpp` and `include/tt_llm_engine/pipeline/pipeline_types.hpp`, set the defaults for the entire system:

| Constant | Value | Source | Meaning |
|----------|-------|--------|---------|
| `DEFAULT_MAX_USERS` | 64 | `decode_types.hpp` | Default maximum concurrent users (configurable via `SchedulerParams::max_users`) |
| `DEFAULT_MAX_SEQ_LEN` | 131072 (128K) | `decode_types.hpp` | Default maximum sequence length per user |
| `DEFAULT_CHUNK_SIZE` | 24 | `decode_types.hpp` | Default prefill chunk size before round-robin rotation |
| `DEFAULT_EOS_TOKEN` | 1 | `decode_types.hpp` | Default EOS token ID (overridable via `SchedulerParams::eos_token`) |
| `INVALID_SLOT` | `UINT32_MAX` | `pipeline_types.hpp` | Sentinel for "no slot" / "shutdown" / "allocation failed" |
| `EMPTY_TOKEN` | `UINT32_MAX` | `pipeline_types.hpp` | Sentinel for "no token" (prefill_token_id on last prefill, spec prediction when unavailable) |

All are `static constexpr` values. `INVALID_SLOT` and `EMPTY_TOKEN` are defined in `pipeline_types.hpp` and re-exported into the `scheduler::decode` namespace via `using` declarations.

---

**Next:** [`key_invariants.md`](./key_invariants.md)
