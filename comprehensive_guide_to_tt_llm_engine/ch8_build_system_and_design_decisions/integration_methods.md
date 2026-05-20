# Integration Methods

[< Previous: Build System](build_system.md) | [Next: Design Decisions >](design_decisions.md)

---

This section documents the two methods for integrating tt-llm-engine into an upstream project -- CMake `add_subdirectory` (recommended) and `find_package` (installed package) -- along with the runtime environment requirements for device operation. Both methods produce the same target names (`TtLlmEngine::Core` and `TtLlmEngine::Full`), so consumer CMakeLists can be written once and work with either integration method.

## The Two CMake Targets

Before choosing an integration method, understand what you are linking against:

| CMake Target | Shared Library | What It Contains | When Available |
|---|---|---|---|
| `TtLlmEngine::Core` | `libtt_llm_engine_core.so` | DecodeScheduler, MockPipeline, PipelineSimulator, all scheduling logic. No hardware dependencies. | Always |
| `TtLlmEngine::Full` | `libtt_llm_engine.so` | Everything in Core plus SocketPipeline (H2D/D2H socket communication with Tenstorrent devices). Defines `DS_HAS_METALIUM=1`. | Only when `TT::Metalium` is found at configure time |

These aliases are defined in `CMakeLists.txt`:

```cmake
add_library(TtLlmEngine::Core ALIAS tt_llm_engine_core)
```

```cmake
add_library(TtLlmEngine::Full ALIAS tt_llm_engine)
```

Source: `CMakeLists.txt`, lines 78 and 123.

## Choosing Core vs Full

| Scenario | Target | Why |
|---|---|---|
| Unit testing scheduler logic | `TtLlmEngine::Core` | MockPipeline provides deterministic in-process behavior; no hardware needed |
| Performance modeling | `TtLlmEngine::Core` | PipelineSimulator provides timing-accurate systolic simulation without hardware |
| Custom PipelineInterface implementation | `TtLlmEngine::Core` | Implement the five virtual methods (see [Chapter 5](../ch5_pipeline_and_wire_format/pipeline_interface.md)) against your own backend |
| Production deployment on Tenstorrent hardware | `TtLlmEngine::Full` | SocketPipeline communicates with device-resident models via H2D/D2H sockets |
| Loopback microbenchmark | `TtLlmEngine::Full` | dummy_pipeline_connector links Full to exercise the real socket path (see [Chapter 7](../ch7_transport_and_tools/loopback_microbenchmark.md)) |
| DeepSeek inference runner | `TtLlmEngine::Full` | Requires SocketPipeline for device communication, plus optional tokenizer support |

## What Headers to Include

Public headers live under `include/tt_llm_engine/` and mirror the C++ namespace hierarchy. The minimum set for a consumer:

```cpp
// The scheduler API -- ALLOCATE, SUBMIT, CONTINUE, CANCEL lifecycle
#include <tt_llm_engine/scheduler/decode/decode_scheduler.hpp>

// Types: ISRequest, SchedulerResponse, OutputMessage, SchedulerParams,
// GenerationParams, RequestType, UserState
#include <tt_llm_engine/scheduler/decode/decode_types.hpp>

// Pipeline configs: MockConfig, PipelineSimulatorConfig, SocketConfig
#include <tt_llm_engine/pipeline/pipeline_types.hpp>
```

For a typical IS integration that only uses the Core library with MockPipeline for development, these three headers are sufficient. The `decode_scheduler.hpp` header forward-declares `PipelineInterface` and includes `decode_types.hpp` and `pipeline_types.hpp` transitively.

When linking `TtLlmEngine::Full`, you may additionally include:

```cpp
#ifdef DS_HAS_METALIUM
#include <tt_llm_engine/pipeline/socket_pipeline.hpp>
#endif
```

The `DS_HAS_METALIUM` macro is automatically defined as a PUBLIC compile definition on the `tt_llm_engine` target (source: `CMakeLists.txt`, line 112), so it propagates to any target that links `TtLlmEngine::Full` without manual flag management.

**Public header directory layout:**

```
include/tt_llm_engine/
    tt_llm_engine.hpp             # Umbrella header
    pipeline/
        pipeline_interface.hpp
        pipeline_types.hpp        # configs, descriptors, sentinels, TokenType
        mock_pipeline.hpp
        socket_pipeline.hpp       # requires DS_HAS_METALIUM
        pipeline_simulator.hpp
        wire_format.hpp
    scheduler/
        decode/
            decode_scheduler.hpp  # DecodeScheduler class
            decode_types.hpp      # SchedulerParams, ISRequest, ...
```

Private implementation headers under `src/` (e.g., `src/common/bounded_queue.hpp`, `src/common/free_id_pool.hpp`, `src/scheduler/decode/decode_staging.hpp`) are not part of the public API and are not installed.

## Method 1: CMake add_subdirectory (Recommended)

This method embeds tt-llm-engine's source tree directly into the upstream build via a git submodule. It is recommended because it avoids a separate install step and ensures all targets are directly available.

### Directory Layout

```
your-project/
+-- external/
|   +-- tt-llm-engine/          <-- submodule (contains tt-metal/ as nested submodule)
+-- src/
+-- CMakeLists.txt
```

Source: `README.md`, lines 215-220.

### Setup

```bash
git submodule add git@github.com:tt-asaigal/tt-llm-engine.git external/tt-llm-engine
git submodule update --init --recursive
```

Source: `README.md`, lines 223-224.

### Core-Only Integration

For Core-only usage (MockPipeline, PipelineSimulator, or a custom `PipelineInterface` implementation), no tt-metal dependency is needed:

```cmake
cmake_minimum_required(VERSION 3.24)
project(your_project LANGUAGES CXX)

add_subdirectory(external/tt-llm-engine)

add_executable(your_app src/main.cpp)
target_link_libraries(your_app PRIVATE TtLlmEngine::Core)
```

Configure and build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)
```

Source: `README.md`, lines 229-235.

In this configuration, CMake will not find `TT::Metalium` (unless the parent project makes it available), so only `libtt_llm_engine_core.so` is produced. The Full library, device tests, and hardware tool targets are all silently skipped. The configure output will show:

```
-- TT::Metalium not found -- building tt_llm_engine_core only (mock pipeline)
```

Source: `CMakeLists.txt`, line 165.

### Full Integration (with SocketPipeline)

To build the Full library, tt-metal must be built first and its CMake config made discoverable. Because tt-metal requires a specific toolchain (clang-20) and the Ninja generator, it cannot be added via `add_subdirectory` from an arbitrary parent project. Instead, it must be pre-built using its own `build_metal.sh` script:

```bash
# From your upstream project root:
cd external/tt-llm-engine/tt-metal && ./build_metal.sh && cd -

# Configure with tt-metal paths:
cmake -S . -B build \
  -DTT_METAL_SOURCE_DIR=$(pwd)/external/tt-llm-engine/tt-metal \
  -DCMAKE_PREFIX_PATH="$(pwd)/external/tt-llm-engine/tt-metal/build/lib/cmake;$(pwd)/external/tt-llm-engine/tt-metal/build_Release/_deps/nlohmann_json-build"
```

Source: `README.md`, lines 240-247.

The parent project's CMakeLists then links the Full target:

```cmake
add_subdirectory(external/tt-llm-engine)

add_executable(your_app src/main.cpp)
target_link_libraries(your_app PRIVATE TtLlmEngine::Full)
```

Source: `README.md`, lines 253-257.

Two CMake variables are required:

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `TT_METAL_SOURCE_DIR` | Path to tt-metal source tree. Needed for socket headers (`d2h_socket.hpp`, `h2d_socket.hpp`) not yet in the installed package. | `$(pwd)/external/tt-llm-engine/tt-metal` |
| `CMAKE_PREFIX_PATH` | Semicolon-separated list of directories where CMake searches for packages. Must include tt-metal's install prefix and nlohmann_json. | `<tt-metal>/build/lib/cmake;<tt-metal>/build_Release/_deps/nlohmann_json-build` |

tt-llm-engine's CMakeLists automatically runs `find_package(TT-Metalium QUIET CONFIG)` using the provided `CMAKE_PREFIX_PATH`. The `if(NOT TARGET TT::Metalium)` guard handles the case where a parent project already made the target available via its own `add_subdirectory` or `find_package`.

## Method 2: find_package (Installed tt-llm-engine)

For projects that prefer to consume a pre-installed tt-llm-engine rather than building from source, the package can be installed and discovered via `find_package`.

### Installing tt-llm-engine

For Core-only install (no hardware):

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=/opt/tt-llm-engine
cmake --build build -j$(nproc)
cmake --install build
```

For Full install (with hardware):

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=/opt/tt-llm-engine \
  -DTT_METAL_SOURCE_DIR=$(pwd)/tt-metal \
  -DCMAKE_PREFIX_PATH="$(pwd)/tt-metal/build/lib/cmake;$(pwd)/tt-metal/build_Release/_deps/nlohmann_json-build"
cmake --build build -j$(nproc)
cmake --install build
```

Source: `README.md`, lines 265-268.

### Consumer CMakeLists

```cmake
cmake_minimum_required(VERSION 3.24)
project(your_project LANGUAGES CXX)

find_package(tt-llm-engine REQUIRED CONFIG)

add_executable(your_app src/main.cpp)
target_link_libraries(your_app PRIVATE TtLlmEngine::Core)
# or: target_link_libraries(your_app PRIVATE TtLlmEngine::Full)
```

Configure with the install prefix on `CMAKE_PREFIX_PATH`:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/tt-llm-engine
```

Source: `README.md`, lines 272-284.

### Package Config Template

The `find_package` support is implemented via a config-file package. The template at `cmake/tt-llm-engine-config.cmake.in` handles transitive dependency resolution:

```cmake
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(Threads)

set(TT_LLM_ENGINE_HAS_FULL @TT_LLM_ENGINE_HAS_FULL@)
if(TT_LLM_ENGINE_HAS_FULL)
    find_dependency(TT-Metalium CONFIG)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/tt-llm-engine-targets.cmake")

check_required_components(tt-llm-engine)
```

Source: `cmake/tt-llm-engine-config.cmake.in`, full file (20 lines).

Key design points:

1. **`Threads` is always required** -- `find_dependency(Threads)` runs unconditionally because the Core library links `Threads::Threads` publicly.

2. **TT-Metalium is conditionally required** -- The `TT_LLM_ENGINE_HAS_FULL` variable is substituted at install time from the boolean set in `CMakeLists.txt` (line 318). If the Full library was installed, `find_dependency(TT-Metalium CONFIG)` ensures the consumer gets a clear error at `find_package(tt-llm-engine)` time rather than a confusing "target TT::Metalium not found" link error later.

3. **Consumers must still put tt-metal's install prefix on CMAKE_PREFIX_PATH** -- the config file re-finds TT-Metalium but does not embed the path. This keeps the installed package relocatable.

Version compatibility uses `SameMajorVersion` (source: `CMakeLists.txt`, line 339).

### Installed File Layout

```
${CMAKE_INSTALL_PREFIX}/
+-- include/
|   +-- tt_llm_engine/                    # Public headers
|       +-- pipeline/
|       |   +-- pipeline_interface.hpp
|       |   +-- pipeline_types.hpp
|       |   +-- mock_pipeline.hpp
|       |   +-- socket_pipeline.hpp       # Only useful with Full
|       |   +-- pipeline_simulator.hpp
|       |   +-- wire_format.hpp
|       +-- scheduler/
|       |   +-- decode/
|       |       +-- decode_scheduler.hpp
|       |       +-- decode_types.hpp
|       +-- tt_llm_engine.hpp             # Umbrella header
+-- lib/
|   +-- libtt_llm_engine_core.so -> libtt_llm_engine_core.so.0
|   +-- libtt_llm_engine_core.so.0 -> libtt_llm_engine_core.so.0.1.0
|   +-- libtt_llm_engine_core.so.0.1.0
|   +-- libtt_llm_engine.so -> ...        # Only if Full was built
|   +-- cmake/
|       +-- tt-llm-engine/
|           +-- tt-llm-engine-config.cmake
|           +-- tt-llm-engine-config-version.cmake
|           +-- tt-llm-engine-targets.cmake
|           +-- tt-llm-engine-targets-relwithdebinfo.cmake
```

Test binaries and tool executables are not installed. They are development-only artifacts.

## Integration Method Comparison

| Aspect | add_subdirectory | find_package |
|--------|-----------------|-------------|
| **Setup complexity** | Add submodule + recursive init | Build, install, point `CMAKE_PREFIX_PATH` |
| **Source access** | Full source in your tree | Headers only |
| **Rebuild on change** | Automatic (CMake detects source changes) | Must reinstall after changes |
| **Build isolation** | Shares your project's CMake cache/options | Fully isolated build |
| **Best for** | Active development, CI pipelines | Production deployments, binary distribution |
| **Mooncake support** | Can enable with `DS_ENABLE_MOONCAKE=ON` | Mooncake targets not installed (test-only) |

## Using the Library from C++

The decode scheduler exposes a four-request lifecycle: ALLOCATE, SUBMIT, (stream tokens), CANCEL (or wait for completion). A minimal single-turn example using the Core library with MockPipeline:

```cpp
#include <tt_llm_engine/scheduler/decode/decode_scheduler.hpp>
#include <thread>

namespace sched = tt_llm_engine::scheduler::decode;
namespace pl    = tt_llm_engine::pipeline;

int main() {
    sched::DecodeScheduler scheduler(pl::MockConfig{});
    scheduler.start();

    // 1. ALLOCATE -- ask the scheduler for a free user slot.
    sched::ISRequest alloc{};
    alloc.type = sched::RequestType::ALLOCATE;
    alloc.request_id = 1;
    while (!scheduler.push_request(alloc)) std::this_thread::yield();

    sched::SchedulerResponse resp{};
    while (!scheduler.try_pop_response(resp)) std::this_thread::yield();
    if (resp.error_code != 0) {
        return 1;  // no free slots
    }
    const uint32_t slot_id = resp.slot_id;

    // 2. SUBMIT -- hand over the prompt and start prefill + decode.
    sched::ISRequest submit{};
    submit.type = sched::RequestType::SUBMIT;
    submit.request_id = 2;
    submit.slot_id = slot_id;
    submit.tokens = {100, 200, 300};
    submit.gen.max_new_tokens = 64;
    while (!scheduler.push_request(submit)) std::this_thread::yield();

    // 3. Drain output until is_complete fires.
    for (;;) {
        sched::OutputMessage out;
        if (!scheduler.try_pop_output(out)) {
            std::this_thread::yield();
            continue;
        }
        if (out.is_complete) break;
    }

    scheduler.stop();
}
```

Source: `README.md`, lines 292-342.

### Key API Types

| Type | Header | Description |
|------|--------|-------------|
| `DecodeScheduler` | `decode_scheduler.hpp` | Main scheduler class. Constructed with a `PipelineConfig` variant and optional `SchedulerParams`. |
| `ISRequest` | `decode_types.hpp` | Request from IS to PM. Fields: `type` (ALLOCATE/SUBMIT/CONTINUE/CANCEL), `request_id`, `slot_id`, `tokens` vector, `gen` (GenerationParams). |
| `SchedulerResponse` | `decode_types.hpp` | Response from PM to IS. Fields: `request_id`, `slot_id`, `error_code` (0 = success, 1 = no free slots). |
| `OutputMessage` | `decode_types.hpp` | Streaming output from PM to IS. Fields: `slot_id`, `token_id`, `is_complete`, `ctx_exhausted`, `tokens_generated`, `generation`. |
| `SchedulerParams` | `decode_types.hpp` | Configuration: `max_users` (default 64), `chunk_size` (default 24), `max_seq_len` (default 128K), `eos_token`, thinking-phase token IDs, CPU affinity settings. |
| `GenerationParams` | `decode_types.hpp` | Per-request generation settings: `max_new_tokens`, `spec_decode`, `ignore_eos`, `temperature`, `top_p`, `top_k`. |
| `UserState` | `decode_types.hpp` | Enum: `INACTIVE`, `PREFILL`, `DECODE`, `COMPLETE`. |
| `RequestType` | `decode_types.hpp` | Enum: `ALLOCATE`, `SUBMIT`, `CONTINUE`, `CANCEL`. |

### Queue Communication Model

The three non-blocking queue methods correspond to the three bounded queues described in [Chapter 6: IS Integration Patterns](../ch6_lifecycle_is_integration_kv_tiering/is_integration_patterns.md):

| Method | Queue | Direction | Capacity |
|--------|-----------|-------|---------|
| `push_request()` | request_queue (`BoundedQueue<ISRequest>`) | IS -> PM | `2 * max_users` |
| `try_pop_response()` | response_queue (`BoundedQueue<SchedulerResponse>`) | PM -> IS | `2 * max_users` |
| `try_pop_output()` | output_queue (`BoundedQueue<OutputMessage>`) | PM -> IS | `256 * max_users` |

All three return `bool` and never block. This is a deliberate design choice: the PM never blocks on IS, and IS never calls PM functions directly; all interaction is through bounded queues.

### Pipeline Backend Selection

The `PipelineConfig` variant determines which pipeline backend is used:

```cpp
// MockPipeline (Core library, no hardware):
sched::DecodeScheduler scheduler(pl::MockConfig{});

// PipelineSimulator (Core library, timing-accurate):
sched::DecodeScheduler scheduler(pl::PipelineSimulatorConfig{
    .num_stages = 64,
    .stage_duration_us = 44
});

// SocketPipeline (Full library, real hardware):
sched::DecodeScheduler scheduler(pl::SocketConfig{
    .h2d_socket_id = "deepseek_h2d",
    .d2h_socket_id = "deepseek_d2h",
    .use_deepseek_md_format = true
});
```

Source: `include/tt_llm_engine/pipeline/pipeline_types.hpp`, lines 68-97.

## Runtime Environment

### TT_METAL_RUNTIME_ROOT

When running any code that uses `TtLlmEngine::Full` (or the device test binary) outside the tt-metal source directory, the `TT_METAL_RUNTIME_ROOT` environment variable must be set:

| Variable | Required For | Points To | Purpose |
|----------|-------------|-----------|---------|
| `TT_METAL_RUNTIME_ROOT` | Device tests, Full library usage, tool binaries | tt-metal repo root (directory containing `tt_metal/`) | Metalium runtime uses this to find kernel sources, firmware, and the SFPI compiler |

Source: `README.md`, lines 662-665.

Example:

```bash
export TT_METAL_RUNTIME_ROOT=$(pwd)/tt-metal
export TT_VISIBLE_DEVICES=0
LD_LIBRARY_PATH=build-full build-full/test_decode_scheduler_device
```

Source: `README.md`, lines 193-195.

### TT_VISIBLE_DEVICES

For the loopback microbenchmark (see [Ch7](../ch7_transport_and_tools/loopback_microbenchmark.md)), restricting the device scope to a single chip keeps startup fast:

```bash
export TT_VISIBLE_DEVICES=0
```

Source: `README.md`, line 375.

### LD_LIBRARY_PATH

Because the libraries are shared objects and are not installed to a system path during development, `LD_LIBRARY_PATH` must include the build directory:

```bash
LD_LIBRARY_PATH=build-standalone build-standalone/test_decode_scheduler
```

Source: `README.md`, line 183. This is only needed for development builds. An installed package places libraries in standard paths.

---

[< Previous: Build System](build_system.md) | [Next: Design Decisions >](design_decisions.md)
