# 1.3 Build System and Component Map

This section maps every source file to its role in the framework, explains the CMake build targets and compile-time feature gating, and provides a namespace reference.

## Source File Map

The Pi0.5 benchmarking framework spans source files in three areas of the `tt-llm-engine` repository: the example applications, the engine library (headers and `src/` implementation files), and device kernels.

### Application Code

| File | Role |
|------|------|
| `examples/pi05_pipeline_runner.cpp` | Main host-side driver (1,463 lines). Contains `PhaseConfig`, `UserSession`, `PipelineConfig`, JSON parser, `BulkH2DChannel`, `BulkD2HChannel`, `DeviceLauncher`, and the `main()` function with all three operating modes. Single source file compiled into two binaries via the `PI05_HAS_SOCKETS` preprocessor guard. |
| `examples/pi05_device_launcher.cpp` | Device-side process (234 lines). Creates `MeshDevice`, allocates H2D/D2H sockets, configures circular buffers, compiles and launches Tensix kernels, then blocks on `Finish()` until the kernels exit. |

### Configuration Files

| File | Mode | Key Differences |
|------|------|-----------------|
| `examples/pi05_config_sim.json` | Simulation-only | No `"socket"` block. Multi-stage phases (4+20+6 stages). 8 users, 20 turns. |
| `examples/pi05_config.json` | Full socket | `"socket"` block with 4 socket IDs, `"mode"` absent (defaults to `"full"`). Multi-stage phases. 8 users, 20 turns. |
| `examples/pi05_config_hybrid.json` | Hybrid / bulk-only | `"socket"` block with `"mode": "bulk_only"`, only bulk socket IDs (no token sockets). Single-stage phases. 1 user, 400 turns, 25 Gbps PCIe bandwidth. |

### Kernel Code

| File | Role |
|------|------|
| `kernels/pipeline_loopback.cpp` | Token loopback kernel for core (0,0). Receives token pages from the H2D socket, performs identity transformation, and sends them back via the D2H socket. Used only in full socket mode. |
| `kernels/pi05_bulk_passthrough.cpp` | Bulk passthrough kernel for core (1,0) or (0,0) in bulk-only mode. Receives pixel payloads via H2D, discards pixel data, generates synthetic action output (incrementing bfloat16 pattern), and sends it via D2H. Supports both `HOST_PUSH` and `DEVICE_PULL` modes. Handles sentinel-based shutdown protocol. |
| `kernels/pcie_noc_utils.h` | Shared header with PCIe/NOC utility functions used by the device kernels. |

### Engine Library Headers

| File | Role |
|------|------|
| `include/tt_llm_engine/pipeline/pipeline_types.hpp` | Type definitions: `InjectDescriptor`, `ResultDescriptor`, `MockConfig`, `SocketConfig`, `PipelineSimulatorConfig`, and the `PipelineConfig` variant (`std::variant<MockConfig, SocketConfig, PipelineSimulatorConfig>`). |
| `include/tt_llm_engine/pipeline/pipeline_interface.hpp` | Abstract `PipelineInterface` base class defining the `inject()`/`read_result()` contract that all pipeline backends implement. |
| `include/tt_llm_engine/pipeline/pipeline_simulator.hpp` | `PipelineSimulator` -- CPU-side systolic pipeline timing model. Implements `PipelineInterface` with per-token timestamp scheduling. Header-only. |
| `include/tt_llm_engine/pipeline/socket_pipeline.hpp` | `SocketPipeline` -- real socket-based pipeline backend. Connects to H2D/D2H token sockets on the device. Only available when `TT::Metalium` is linked. |
| `include/tt_llm_engine/pipeline/mock_pipeline.hpp` | `MockPipeline` -- minimal latency-only mock. Used by unit tests, not by the Pi0.5 runner. |
| `include/tt_llm_engine/pipeline/wire_format.hpp` | Wire format constants (e.g., `PAGE_SIZE_BYTES = 256` for token sockets). |
| `include/tt_llm_engine/scheduler/decode/decode_scheduler.hpp` | `DecodeScheduler` -- the main scheduling engine. Accepts a `PipelineConfig` variant in its constructor and creates the appropriate `PipelineInterface` implementation internally. |
| `include/tt_llm_engine/scheduler/decode/decode_types.hpp` | Scheduler type definitions: `SchedulerParams`, `UserState`, `RequestType`, `ISRequest`, `SchedulerResponse`, `OutputMessage`, `GenerationParams`. |

### Engine Library Implementation Files (`src/`)

| File | Compiled Into | Role |
|------|---------------|------|
| `src/scheduler/decode/decode_scheduler.cpp` | Both `libtt_llm_engine_core.so` and `libtt_llm_engine.so` | `DecodeScheduler` implementation. Core scheduling logic, pipeline backend creation based on `PipelineConfig` variant. |
| `src/pipeline/socket_pipeline.cpp` | `libtt_llm_engine.so` only | `SocketPipeline` implementation. Real hardware socket pipeline backend. Not compiled into `libtt_llm_engine_core.so` (no Metalium dependency in the core library). |

## CMake Build Targets

The Pi0.5-related targets are defined in `CMakeLists.txt`, organized in three tiers based on hardware dependency.

### Tier 1: No Hardware Dependency (Always Built)

```cmake
# CMakeLists.txt, ~lines 56-78
add_library(tt_llm_engine_core SHARED
    ${TLE_SCHEDULER_DECODE_SOURCES}    # src/scheduler/decode/decode_scheduler.cpp
)
target_link_libraries(tt_llm_engine_core PUBLIC Threads::Threads)
```

```cmake
# CMakeLists.txt, ~line 172-174
add_executable(pi05_pipeline_runner examples/pi05_pipeline_runner.cpp)
target_link_libraries(pi05_pipeline_runner PRIVATE tt_llm_engine_core Threads::Threads)
```

`pi05_pipeline_runner` links only `tt_llm_engine_core`, which contains the `DecodeScheduler` and (transitively via headers) the `PipelineSimulator`. No Metalium, no sockets. This is the simulation-only binary, viable in CI without hardware on any Linux system with a C++20 compiler and pthreads.

### Tier 2: Metalium Available (Multi-Chip Capable)

```cmake
# CMakeLists.txt, ~lines 85-123
if(TARGET TT::Metalium)
    add_library(tt_llm_engine SHARED
        ${TLE_SCHEDULER_DECODE_SOURCES}    # same scheduler code
        ${TLE_PIPELINE_SOURCES}            # + src/pipeline/socket_pipeline.cpp
    )
    target_link_libraries(tt_llm_engine PUBLIC TT::Metalium Threads::Threads)
    target_compile_definitions(tt_llm_engine PUBLIC DS_HAS_METALIUM=1)
endif()
```

```cmake
# CMakeLists.txt, ~lines 177-180
if(TARGET tt_llm_engine)
    add_executable(pi05_pipeline_runner_device examples/pi05_pipeline_runner.cpp)
    target_link_libraries(pi05_pipeline_runner_device PRIVATE tt_llm_engine Threads::Threads)
    target_compile_definitions(pi05_pipeline_runner_device PRIVATE PI05_HAS_SOCKETS=1)
endif()
```

`pi05_pipeline_runner_device` compiles the **same source file** (`pi05_pipeline_runner.cpp`) but links `tt_llm_engine` (which includes `SocketPipeline`) and defines `PI05_HAS_SOCKETS=1`. This enables the `#ifdef PI05_HAS_SOCKETS` code blocks containing `BulkH2DChannel`, `BulkD2HChannel`, `DeviceLauncher`, and the full/hybrid socket mode logic.

### Tier 3: Direct Metalium (Device Launcher)

```cmake
# CMakeLists.txt, ~lines 184-194
if(TARGET TT::Metalium)
    add_executable(pi05_device_launcher examples/pi05_device_launcher.cpp)
    target_link_libraries(pi05_device_launcher PRIVATE TT::Metalium Threads::Threads)
    target_compile_definitions(pi05_device_launcher PRIVATE
        DS_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    )
    if(TT_METAL_SOURCE_DIR)
        target_include_directories(pi05_device_launcher SYSTEM PRIVATE
            ${TT_METAL_SOURCE_DIR}/tt_metal/api)
    endif()
endif()
```

`pi05_device_launcher` links directly against `TT::Metalium` (not `tt_llm_engine`). It uses the Metalium API directly to create devices, allocate sockets, create programs, and launch kernels. The `DS_KERNEL_DIR` compile definition tells it where to find the Tensix kernel source files at runtime.

### Target Dependency Graph

```
TT::Metalium (find_package or parent scope)
    |
    +-- libtt_llm_engine.so
    |       sources: decode_scheduler.cpp + socket_pipeline.cpp
    |       defines: DS_HAS_METALIUM=1
    |       |
    |       +-- pi05_pipeline_runner_device
    |               source: pi05_pipeline_runner.cpp
    |               defines: PI05_HAS_SOCKETS=1
    |
    +-- pi05_device_launcher
            source: pi05_device_launcher.cpp
            defines: DS_KERNEL_DIR="..."

libtt_llm_engine_core.so  (always built, no Metalium)
    sources: decode_scheduler.cpp
    |
    +-- pi05_pipeline_runner
            source: pi05_pipeline_runner.cpp
            (no PI05_HAS_SOCKETS, no DS_HAS_METALIUM)
```

### Target Summary

| Target | Source | When Built | Link Libraries | Compile Definitions |
|--------|--------|------------|----------------|---------------------|
| `tt_llm_engine_core` | `decode_scheduler.cpp` | Always | `Threads::Threads` | (none) |
| `tt_llm_engine` | `decode_scheduler.cpp` + `socket_pipeline.cpp` | When `TARGET TT::Metalium` exists | `TT::Metalium`, `Threads::Threads` | `DS_HAS_METALIUM=1` |
| `pi05_pipeline_runner` | `pi05_pipeline_runner.cpp` | Always | `tt_llm_engine_core`, `Threads::Threads` | (none) |
| `pi05_pipeline_runner_device` | `pi05_pipeline_runner.cpp` | When `TARGET tt_llm_engine` exists | `tt_llm_engine`, `Threads::Threads` | `PI05_HAS_SOCKETS=1` |
| `pi05_device_launcher` | `pi05_device_launcher.cpp` | When `TARGET TT::Metalium` exists | `TT::Metalium`, `Threads::Threads` | `DS_KERNEL_DIR="<src>/kernels"` |

## Compile-Time Feature Gating

### `PI05_HAS_SOCKETS` (Pi0.5-Specific)

The `PI05_HAS_SOCKETS` macro gates all socket-dependent code in `pi05_pipeline_runner.cpp`. Defined only on `pi05_pipeline_runner_device`. When undefined (simulation-only build), the compiler excludes:

- System headers: `<sys/prctl.h>`, `<sys/types.h>`, `<sys/wait.h>`, `<unistd.h>`
- Metalium socket headers: `h2d_socket.hpp`, `d2h_socket.hpp`
- `DistributedContext` header
- `BulkH2DChannel` class
- `BulkD2HChannel` class
- `DeviceLauncher` class
- The entire full-socket and hybrid code paths in `main()`

The guard appears in three locations:

```cpp
// examples/pi05_pipeline_runner.cpp

// Line 35-43: Conditional includes
#ifdef PI05_HAS_SOCKETS
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <tt-metalium/distributed_context.hpp>
#endif

// Line 496-727: Conditional class definitions
#ifdef PI05_HAS_SOCKETS
class BulkH2DChannel { ... };
class BulkD2HChannel { ... };
class DeviceLauncher { ... };
#endif

// Line 859-1198: Conditional main() code path
#ifdef PI05_HAS_SOCKETS
        if (pipeline_config.use_sockets) {
            // Full socket or hybrid mode
            // ...
        } else
#endif
        if (pipeline_config.use_sockets) {
            // Compiled without sockets but config has socket block -- error
            throw std::runtime_error("...");
        }
```

When the binary is compiled **without** `PI05_HAS_SOCKETS` but the user provides a config with a `"socket"` block, the runtime produces a clear error:

```
Config has socket section but binary was compiled without PI05_HAS_SOCKETS.
Use pi05_pipeline_runner_device for socket mode.
```

### `DS_HAS_METALIUM` (Library-Wide)

The `DS_HAS_METALIUM` macro is defined as a **public** compile definition on `libtt_llm_engine.so`. It enables `SocketPipeline` as a viable `PipelineConfig` variant option inside `DecodeScheduler`. When this macro is not defined (i.e., in `libtt_llm_engine_core.so`), passing a `SocketConfig` to the `DecodeScheduler` constructor will fail at runtime because the `SocketPipeline` backend is not compiled in.

This macro propagates to all targets that link `tt_llm_engine` (including `pi05_pipeline_runner_device`) via CMake's `PUBLIC` visibility on `target_compile_definitions`.

### `DS_KERNEL_DIR` (Device Launcher)

The `DS_KERNEL_DIR` definition is injected only into `pi05_device_launcher` and resolves to the absolute path of the `kernels/` directory at build time:

```cmake
target_compile_definitions(pi05_device_launcher PRIVATE
    DS_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)
```

It is used in `pi05_device_launcher.cpp` to locate kernel source files for JIT compilation:

```cpp
// examples/pi05_device_launcher.cpp, ~line 44-46
#ifndef DS_KERNEL_DIR
#define DS_KERNEL_DIR "kernels"
#endif
```

The fallback `"kernels"` value (relative path) provides a default for standalone compilation or testing, but the CMake-injected absolute path is what is used in production builds.

Kernel paths are constructed by concatenation:

```cpp
// examples/pi05_device_launcher.cpp, ~line 179, 197
CreateKernel(program, std::string(DS_KERNEL_DIR) + "/pipeline_loopback.cpp", ...);
CreateKernel(program, std::string(DS_KERNEL_DIR) + "/pi05_bulk_passthrough.cpp", ...);
```

## Link Dependencies and Their Implications

### tt_llm_engine_core (libtt_llm_engine_core.so)

The "core" library contains all hardware-independent components:

- `PipelineSimulator` -- CPU-side timing model (header-only, included transitively)
- `MockPipeline` -- trivial latency mock (header-only)
- `DecodeScheduler` -- scheduling engine (accepts any `PipelineConfig` variant)
- All scheduler types and pipeline type definitions

This library has **no dependency on TT::Metalium or tt-metal**. It compiles and runs on any Linux system with a C++20 compiler and pthreads. This is what makes `pi05_pipeline_runner` (simulation-only) viable in CI without hardware.

### tt_llm_engine (libtt_llm_engine.so)

The "full" library extends `tt_llm_engine_core` with hardware-dependent components:

- `SocketPipeline` -- real socket-based pipeline backend (compiled from `src/pipeline/socket_pipeline.cpp`)
- Integration with `TT::Metalium` for device management
- `DS_HAS_METALIUM=1` public define, enabling `SocketConfig` handling in `DecodeScheduler`

It is only built when `TT::Metalium` is available (CMake guard: `if(TARGET TT::Metalium)`), and in turn `pi05_pipeline_runner_device` is only built when `tt_llm_engine` exists.

### TT::Metalium

The Metalium runtime is a direct dependency of `pi05_device_launcher` only. It provides:

- `MeshDevice` -- device topology management
- `H2DSocket` / `D2HSocket` -- host-device socket creation and descriptor export
- `Program`, `CreateKernel`, `CreateCircularBuffer` -- kernel compilation and launch APIs
- `MeshWorkload`, `EnqueueMeshWorkload`, `Finish` -- workload execution

The host-side runner (`pi05_pipeline_runner_device`) does NOT link `TT::Metalium` directly. It links `tt_llm_engine`, which provides the `SocketPipeline` wrapper. The raw `H2DSocket`/`D2HSocket` includes in the runner (for `BulkH2DChannel`/`BulkD2HChannel`) come from `tt_llm_engine`'s transitive includes.

## Namespace Map

| Namespace | Alias in Runner | Contents |
|-----------|:---:|---------|
| `tt_llm_engine::pipeline` | `pl` | `PipelineConfig` variant, `PipelineSimulatorConfig`, `SocketConfig`, `MockConfig`, `PipelineInterface`, `PipelineSimulator`, `SocketPipeline`, `InjectDescriptor`, `ResultDescriptor`, `INVALID_SLOT`, `EMPTY_TOKEN`, `TokenType` |
| `tt_llm_engine::scheduler::decode` | `pm` | `DecodeScheduler`, `SchedulerParams`, `ISRequest`, `SchedulerResponse`, `OutputMessage`, `RequestType`, `UserState`, `GenerationParams`, `MAX_USERS` |
| `tt::tt_metal::distributed` | (used directly) | `H2DSocket`, `D2HSocket`, `H2DMode` -- socket creation/connection |
| `tt::tt_metal::distributed::multihost` | (used directly) | `DistributedContext`, `Color`, `Key` -- MPI coordination |
| `tt::tt_metal` | (used in launcher) | `MeshDevice`, `MeshDeviceConfig`, `MeshShape`, `MeshCoordinate`, `CoreCoord`, `Program`, `CreateKernel`, `CreateCircularBuffer`, `CircularBufferConfig`, `DataMovementConfig`, `MeshWorkload`, `EnqueueMeshWorkload`, `Finish` |
| `(anonymous)` | -- | All Pi0.5-specific types in `pi05_pipeline_runner.cpp`: `PhaseConfig`, `PipelineConfig` (not the variant -- the JSON-parsed struct), `UserSession`, `Stats`, `SocketConfigParams`, `PixelPayloadConfig`, `BulkTransferMetrics`, `BulkH2DHeader`, `BulkD2HHeader`, `BulkH2DChannel`, `BulkD2HChannel`, `DeviceLauncher` |

Note the naming collision: `PipelineConfig` exists in two contexts:
- `tt_llm_engine::pipeline::PipelineConfig` -- the `std::variant<MockConfig, SocketConfig, PipelineSimulatorConfig>` type alias (in `pipeline_types.hpp`).
- The anonymous-namespace `PipelineConfig` struct in `pi05_pipeline_runner.cpp` -- the JSON-parsed configuration container holding `phases`, `socket`, `pixel`, etc.

These do not conflict because the anonymous-namespace version is local to the translation unit and is never passed to library APIs. The `DecodeScheduler` constructor receives the variant type, not the JSON struct.

---

**Next:** [Chapter 2 -- JSON Configuration Schema and Parsing](../ch2_json_configuration_schema_and_parsing/index.md)
