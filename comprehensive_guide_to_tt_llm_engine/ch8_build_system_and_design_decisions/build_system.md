# Build System

[< Previous: Chapter Overview](index.md) | [Next: Integration Methods >](integration_methods.md)

---

The tt-llm-engine build system is a single-file CMake project (`CMakeLists.txt`, 346 lines) that produces two shared-library variants -- Core (always built) and Full (conditional on TT::Metalium) -- plus test binaries, tool executables, and optional Mooncake transport targets. Conditional compilation is driven by CMake options and automatic target detection, not by explicit feature flags in source code. Two shell scripts (`build.sh` and `setup.sh`) wrap the CMake invocations for common workflows.

## Two Library Variants

The central architectural decision in the build system is the split into two shared libraries. This mirrors the `PipelineInterface` abstraction described in [Ch5](../ch5_pipeline_and_wire_format/pipeline_interface.md): the Core library contains all scheduling logic and software-only pipeline backends (MockPipeline, PipelineSimulator), while the Full library adds the hardware-backed SocketPipeline.

| Library | CMake Target | Alias | Shared Object | Dependencies | When Built |
|---------|-------------|-------|---------------|-------------|------------|
| **Core** | `tt_llm_engine_core` | `TtLlmEngine::Core` | `libtt_llm_engine_core.so` | `Threads::Threads` | Always |
| **Full** | `tt_llm_engine` | `TtLlmEngine::Full` | `libtt_llm_engine.so` | `TT::Metalium`, `Threads::Threads` | Only when `TARGET TT::Metalium` exists |

Source: `CMakeLists.txt`, lines 56-78 (Core), lines 85-123 (Full).

### Core Library

The Core library is unconditionally built. It compiles the decode scheduler implementation and exposes the full public API surface:

```cmake
add_library(tt_llm_engine_core SHARED
    ${TLE_SCHEDULER_DECODE_SOURCES}
)
target_include_directories(
    tt_llm_engine_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(tt_llm_engine_core PUBLIC Threads::Threads)
```

Source: `CMakeLists.txt`, lines 56-67.

Key points:
- PUBLIC include directory uses generator expressions for correct behavior in both build-tree and install-tree contexts.
- PRIVATE `src/` directory is for internal headers (`bounded_queue.hpp`, `free_id_pool.hpp`, `user_table.hpp`, `decode_staging.hpp`, etc.). These are not exposed to consumers.
- Only dependency is `Threads::Threads` (pthreads). No hardware, network, or third-party library dependencies.

The source grouping is defined at the top of the file:

```cmake
set(TLE_SCHEDULER_DECODE_SOURCES
    src/scheduler/decode/decode_scheduler.cpp
)
```

Source: `CMakeLists.txt`, lines 31-33.

The `EXPORT_NAME Core` property (line 76) ensures the install target is exported as `TtLlmEngine::Core`:

```cmake
set_target_properties(
    tt_llm_engine_core
    PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 0
        EXPORT_NAME Core
)
add_library(TtLlmEngine::Core ALIAS tt_llm_engine_core)
```

Source: `CMakeLists.txt`, lines 69-78.

### Full Library

The Full library is only built when `TT::Metalium` is available. Detection uses a two-step process:

```cmake
if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium QUIET CONFIG)
endif()
```

Source: `CMakeLists.txt`, lines 42-44.

This means `TT::Metalium` can arrive either from a parent project's `add_subdirectory` (the target is already in scope) or from a prior `find_package` (found via `CMAKE_PREFIX_PATH`). The `QUIET` flag prevents a hard error when Metalium is absent -- the build simply skips the Full library.

The Full library compiles both the scheduler sources and the SocketPipeline:

```cmake
add_library(
    tt_llm_engine
    SHARED
    ${TLE_SCHEDULER_DECODE_SOURCES}
    ${TLE_PIPELINE_SOURCES}
)
```

Source: `CMakeLists.txt`, lines 86-91.

Where `TLE_PIPELINE_SOURCES` is:

```cmake
set(TLE_PIPELINE_SOURCES
    src/pipeline/socket_pipeline.cpp
)
```

Source: `CMakeLists.txt`, lines 34-36.

The Full library re-compiles `decode_scheduler.cpp` rather than linking against the Core library. This is intentional: the Full library defines `DS_HAS_METALIUM=1` as a public compile definition (line 112), which would cause ODR violations if a single `.cpp` file were compiled with different definitions in Core and Full.

```cmake
target_compile_definitions(tt_llm_engine PUBLIC DS_HAS_METALIUM=1)
```

Source: `CMakeLists.txt`, line 112.

This enables consumers to use conditional compilation:

```cpp
#ifdef DS_HAS_METALIUM
#include <tt_llm_engine/pipeline/socket_pipeline.hpp>
#else
#include <tt_llm_engine/pipeline/pipeline_interface.hpp>
#endif
```

Source: `README.md`, lines 350-357.

### TT_METAL_SOURCE_DIR Workaround

The `d2h_socket.hpp` and `h2d_socket.hpp` headers are not yet part of the installed TT-Metalium package. When building against an installed package, `TT_METAL_SOURCE_DIR` must point at the tt-metal source tree:

```cmake
set(TT_METAL_SOURCE_DIR "" CACHE PATH
    "Path to tt-metal source tree (needed until socket headers are installed)")
```

Source: `CMakeLists.txt`, line 50.

When set, the directory is added as a system include path:

```cmake
if(TT_METAL_SOURCE_DIR)
    target_include_directories(tt_llm_engine SYSTEM PUBLIC
        $<BUILD_INTERFACE:${TT_METAL_SOURCE_DIR}/tt_metal/api>)
endif()
```

Source: `CMakeLists.txt`, lines 102-104.

The `SYSTEM` qualifier suppresses compiler warnings from these external headers.

## CMake Options

| Option | Type | Default | Description | Source |
|--------|------|---------|-------------|--------|
| `DS_ENABLE_TSAN` | `BOOL` | `OFF` | Enable ThreadSanitizer (`-fsanitize=thread -g`) | `CMakeLists.txt`, line 11 |
| `DS_BUILD_TESTS` | `BOOL` | `ON` | Build test binaries (fetches Google Test) | `CMakeLists.txt`, line 12 |
| `DS_ENABLE_MOONCAKE` | `BOOL` | `OFF` | Build with Mooncake transport (`third_party/mooncake` submodule) | `CMakeLists.txt`, line 13 |
| `DS_MOONCAKE_WITH_RDMA` | `BOOL` | `OFF` | Build Mooncake's RDMA transport (implies `DS_ENABLE_MOONCAKE`) | `CMakeLists.txt`, line 14 |
| `DS_NO_TOKENIZER` | `BOOL` | `OFF` | Skip Rust/tokenizers-cpp build for `deepseek_inference_runner` | `CMakeLists.txt`, line 140 |
| `TT_METAL_SOURCE_DIR` | `PATH` | `""` | Path to tt-metal source tree for socket headers | `CMakeLists.txt`, line 50 |

### TSAN Support

Thread Sanitizer is enabled via `DS_ENABLE_TSAN`:

```cmake
if(DS_ENABLE_TSAN)
    add_compile_options(
        -fsanitize=thread
        -g
    )
    add_link_options(-fsanitize=thread)
endif()
```

Source: `CMakeLists.txt`, lines 16-22.

This adds the flags globally -- all targets (libraries, tests, tools) are instrumented. The `-g` flag ensures debug info is present for meaningful TSAN reports. The `build.sh` script exposes this via the `TSAN` environment variable:

```bash
TSAN="${TSAN:-OFF}"
```

Source: `build.sh`, line 6.

## Test Targets

All tests are gated behind `DS_BUILD_TESTS` (default ON). Google Test is fetched via `FetchContent`:

```cmake
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0
    FIND_PACKAGE_ARGS
        NAMES
        GTest
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
```

Source: `CMakeLists.txt`, lines 172-183.

The `FIND_PACKAGE_ARGS` clause means CMake will first try a system-installed Google Test before downloading.

### Test Target Map

| Test Binary | CMake Target | Links | When Built | CTest Name |
|-------------|-------------|-------|------------|------------|
| `test_decode_scheduler` | `test_decode_scheduler` | `tt_llm_engine_core`, `gtest`, `gtest_main` | Always (when tests enabled) | `DecodeSchedulerMock` |
| `test_decode_scheduler_device` | `test_decode_scheduler_device` | `tt_llm_engine`, `gtest`, `gtest_main` | When `TT::Metalium` available | `DecodeSchedulerDevice` |
| `test_mooncake_transport` | `test_mooncake_transport` | `TtLlmEngine::Mooncake::TransferEngine`, `gtest`, `gtest_main` | When `DS_ENABLE_MOONCAKE=ON` | `MooncakeTransport` |
| `test_mooncake_store` | `test_mooncake_store` | `TtLlmEngine::Mooncake::TransferEngine`, `TtLlmEngine::Mooncake::Store`, `gtest`, `gtest_main` | When `DS_ENABLE_MOONCAKE=ON` | `MooncakeStore` |

Source: `CMakeLists.txt`, lines 186-285.

The mock test (`test_decode_scheduler`) links against the Core library only -- it exercises the full scheduler stack using `MockPipeline` or `PipelineSimulator` without any hardware. This is the primary regression gate. The device test links against the Full library and requires `TT::Metalium` at both build and run time.

An umbrella target aggregates all available tests:

```cmake
add_custom_target(tt_llm_engine_tests DEPENDS test_decode_scheduler)
if(TARGET test_decode_scheduler_device)
    add_dependencies(tt_llm_engine_tests test_decode_scheduler_device)
endif()
if(TARGET test_mooncake_transport)
    add_dependencies(tt_llm_engine_tests test_mooncake_transport)
endif()
if(TARGET test_mooncake_store)
    add_dependencies(tt_llm_engine_tests test_mooncake_store)
endif()
```

Source: `CMakeLists.txt`, lines 276-285.

Both `build.sh` build modes target `tt_llm_engine_tests` for the build step.

### Mooncake Test Configuration

Mooncake tests have CTest labels for selective execution:

```cmake
if(DS_MOONCAKE_WITH_RDMA)
    set_tests_properties(MooncakeTransport PROPERTIES LABELS "mooncake;tcp;rdma")
else()
    set_tests_properties(MooncakeTransport PROPERTIES LABELS "mooncake;tcp")
endif()
```

Source: `CMakeLists.txt`, lines 237-240.

RDMA test cases are conditionally compiled via a preprocessor define:

```cmake
if(DS_MOONCAKE_WITH_RDMA)
    target_compile_definitions(test_mooncake_transport PRIVATE MOONCAKE_HAS_RDMA=1)
endif()
```

Source: `CMakeLists.txt`, lines 233-234.

## Tool Targets

All tool binaries require the Full library (and thus `TT::Metalium`). They are built inside the `if(TARGET TT::Metalium)` block.

| Tool Binary | CMake Target | Links | Additional Dependencies | Source |
|-------------|-------------|-------|------------------------|--------|
| `dummy_pipeline_launcher` | `dummy_pipeline_launcher` | `TT::Metalium`, `Threads::Threads` | Kernel directory define | `CMakeLists.txt`, lines 126-133 |
| `dummy_pipeline_connector` | `dummy_pipeline_connector` | `tt_llm_engine`, `Threads::Threads` | -- | `CMakeLists.txt`, lines 136-137 |
| `deepseek_inference_runner` | `deepseek_inference_runner` | `tt_llm_engine`, `Threads::Threads`, `tokenizers_cpp` | Rust/cargo | `CMakeLists.txt`, lines 156-157 |

### Kernel Directory Define

The `dummy_pipeline_launcher` and `test_decode_scheduler_device` targets receive a compile definition pointing to the kernel source directory:

```cmake
target_compile_definitions(dummy_pipeline_launcher PRIVATE
    DS_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
)
```

Source: `CMakeLists.txt`, lines 128-130.

This allows the launcher to locate the `pipeline_loopback.cpp` kernel at runtime without a runtime path search.

### Tokenizer Build (tokenizers-cpp)

The `deepseek_inference_runner` requires `tokenizers-cpp`, which wraps HuggingFace's Rust-based tokenizers crate. Building it requires `cargo` in the PATH. The build system probes for cargo and conditionally builds:

```cmake
option(DS_NO_TOKENIZER "Disable tokenizer support (skip Rust/tokenizers-cpp build)" OFF)
if(NOT DS_NO_TOKENIZER)
    find_program(CARGO_EXECUTABLE cargo HINTS "$ENV{HOME}/.cargo" PATH_SUFFIXES "bin")
endif()
if(CARGO_EXECUTABLE AND NOT DS_NO_TOKENIZER)
    include(FetchContent)
    FetchContent_Declare(
        tokenizers_cpp
        GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
        GIT_TAG acbdc5a27ae01ba74cda756f94da698d40f11dfe  # main as of 2026-05-07
    )
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
    FetchContent_MakeAvailable(tokenizers_cpp)
    add_executable(deepseek_inference_runner tools/decode/deepseek_inference_runner.cpp)
    target_link_libraries(deepseek_inference_runner PRIVATE tt_llm_engine Threads::Threads tokenizers_cpp)
    message(STATUS "Rust/cargo found -- building deepseek_inference_runner")
else()
    message(WARNING "Rust/cargo not found -- skipping deepseek_inference_runner build")
endif()
```

Source: `CMakeLists.txt`, lines 140-161.

Key details:

- The `GIT_TAG` is pinned to a known-good SHA (`acbdc5a27ae01ba74cda756f94da698d40f11dfe`), not a branch name. Upstream pushes breaking changes through main, including msgpack/cmake-policy churn.
- The `CMAKE_POLICY_VERSION_MINIMUM` workaround addresses a compatibility issue with tokenizers-cpp's vendored msgpack, which sets `cmake_minimum_required` below CMake 4.x's floor.
- The `DS_NO_TOKENIZER` option provides an explicit opt-out, useful when cargo is installed but the tokenizer build is not desired.

## Mooncake Build Integration

Mooncake integration is isolated in `cmake/mooncake.cmake` (348 lines), included unconditionally from the main `CMakeLists.txt`:

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/mooncake.cmake)
```

Source: `CMakeLists.txt`, line 27.

The file begins with an early return guard:

```cmake
if(NOT DS_ENABLE_MOONCAKE)
    return()
endif()
```

Source: `cmake/mooncake.cmake`, lines 19-21.

### Submodule Validation

The Mooncake cmake file validates that all required submodules are initialized before proceeding. It checks eight separate paths:

```cmake
set(_MOONCAKE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mooncake")
if(NOT EXISTS "${_MOONCAKE_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "DS_ENABLE_MOONCAKE=ON but the submodule ...")
endif()
foreach(_dep glog gflags yaml-cpp)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/${_dep}/CMakeLists.txt")
        message(FATAL_ERROR "Vendored dependency third_party/${_dep} is not initialized. ...")
    endif()
endforeach()
```

Source: `cmake/mooncake.cmake`, lines 23-56.

The validated dependencies are: `mooncake`, `glog`, `gflags`, `yaml-cpp`, `zstd`, `boost`, `msgpack`, and `yalantinglibs` (Mooncake's own submodule).

### Vendored Dependencies via OVERRIDE_FIND_PACKAGE

Mooncake's C++ dependencies are not available via system packages in the target environment (no sudo). They are vendored as submodules under `third_party/` and wired into Mooncake's `find_package()` calls via CMake's `OVERRIDE_FIND_PACKAGE` mechanism (requires CMake >= 3.24):

```cmake
FetchContent_Declare(gflags
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gflags"
    OVERRIDE_FIND_PACKAGE)
FetchContent_Declare(glog
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glog"
    OVERRIDE_FIND_PACKAGE)
FetchContent_Declare(yaml-cpp
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/yaml-cpp"
    OVERRIDE_FIND_PACKAGE)
```

Source: `cmake/mooncake.cmake`, lines 103-111.

| Vendored Dependency | Source Location | Why Vendored |
|--------------------|-----------------|----|
| gflags | `third_party/gflags` | Not available via system packages (no sudo) |
| glog | `third_party/glog` | Not available via system packages (no sudo) |
| yaml-cpp | `third_party/yaml-cpp` | Not available via system packages (no sudo) |
| yalantinglibs | `third_party/mooncake/extern/yalantinglibs` | Mooncake's own submodule, header-only; provides ASIO headers |
| zstd | `third_party/zstd/build/cmake` | libzstd-dev not available (no sudo) |
| Boost | `third_party/boost` | Modular superproject layout (`libs/*/include/`) |
| msgpack-cxx | `third_party/msgpack` | Required by Mooncake Store headers |

### Mooncake Feature Gating

Several Mooncake features are explicitly disabled to minimize the build footprint:

```cmake
set(USE_ETCD OFF CACHE BOOL "" FORCE)
set(USE_HTTP OFF CACHE BOOL "" FORCE)         # drops libcurl
set(WITH_STORE_RUST OFF CACHE BOOL "" FORCE)  # drops Rust binding build
set(WITH_P2P_STORE OFF CACHE BOOL "" FORCE)
set(WITH_METRICS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_STORE ON CACHE BOOL "" FORCE)
set(WITH_TE ON CACHE BOOL "" FORCE)
```

Source: `cmake/mooncake.cmake`, lines 65-73.

Mooncake is added with `EXCLUDE_FROM_ALL` to prevent it from polluting the default build targets:

```cmake
add_subdirectory("${_MOONCAKE_ROOT}" "${CMAKE_BINARY_DIR}/_mooncake" EXCLUDE_FROM_ALL)
```

Source: `cmake/mooncake.cmake`, line 249.

### Vendor Include Injection

Instead of a directory-scoped `include_directories()` that would pollute every target, vendor includes are injected only into Mooncake's own targets:

```cmake
set(_MC_ALL_TARGETS
    transfer_engine transport
    tcp_transport rdma_transport rpc_communicator
    base
    mooncake_store mooncake_master mooncake_client
    mooncake_common asio_shared)
foreach(_tgt ${_MC_ALL_TARGETS})
    if(TARGET ${_tgt})
        target_include_directories(${_tgt} PRIVATE ${_MC_VENDOR_INCLUDES})
    endif()
endforeach()
```

Source: `cmake/mooncake.cmake`, lines 258-268.

### Target Aliasing

After `add_subdirectory`, Mooncake's internal targets are aliased under a stable namespace:

```cmake
add_library(TtLlmEngine::Mooncake::TransferEngine ALIAS transfer_engine)
if(TARGET mooncake_store)
    add_library(TtLlmEngine::Mooncake::Store ALIAS mooncake_store)
endif()
```

Source: `cmake/mooncake.cmake`, lines 343-345.

This insulates consumers from Mooncake's internal target naming. If Mooncake renames its targets, only `cmake/mooncake.cmake` needs updating.

### Compatibility Shims

The Mooncake cmake file includes several compatibility workarounds for vendored dependencies that do not match assumptions in Mooncake's own CMakeLists:

1. **gflags namespace shim**: Mooncake's `master.cpp` uses `google::CommandLineFlagInfo` which is `gflags::CommandLineFlagInfo` in the vendored version. A generated compatibility header maps the namespace via `using` declarations (lines 305-323):

```cmake
set(_GFLAGS_COMPAT_SHIM "${CMAKE_BINARY_DIR}/_mooncake_gflags_google_compat.h")
file(WRITE "${_GFLAGS_COMPAT_SHIM}" [==[
#pragma once
#include <gflags/gflags.h>
namespace google {
  using gflags::CommandLineFlagInfo;
  using gflags::GetCommandLineFlagInfo;
  ...
}
]==])
target_compile_options(mooncake_master PRIVATE
    "-include" "${_GFLAGS_COMPAT_SHIM}")
```

2. **glog build-dir ordering**: glog's build directory contains a bare `config.h` that conflicts with Mooncake's own `config.h`. The glog build dir is injected only AFTER Mooncake's own includes to preserve priority (lines 274-279).

3. **gflags force-include**: `mooncake_store`'s `real_client.cpp` uses `DEFINE_bool`/`DEFINE_int32` without including `<gflags/gflags.h>`. A force-include is added (lines 297-299).

4. **ASIO headers from yalantinglibs**: `libasio-dev` is not available, so the bundled copy from yalantinglibs at `include/ylt/thirdparty/` is used (lines 188-204).

5. **xxhash from zstd**: `libxxhash-dev` is absent, but zstd bundles `xxhash.h`. The include path is pointed at zstd's `lib/common/` directory (lines 161-184).

## Build Scripts

### build.sh

The primary build script supports two modes:

| Mode | Flag | Build Directory | Artifacts |
|------|------|----------------|-----------|
| Standalone | `--standalone` | `build-standalone/` | `libtt_llm_engine_core.so`, `test_decode_scheduler` |
| With Metal | `--with-metal` | `build-full/` | Both libraries, all tests, all tools |

Source: `build.sh`, lines 66-166.

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JOBS` | `$(nproc)` | Parallel build jobs |
| `TSAN` | `OFF` | Maps to `DS_ENABLE_TSAN` |
| `NO_TOKENIZER` | `OFF` | Maps to `DS_NO_TOKENIZER` |
| `MOONCAKE` | `OFF` | Maps to `DS_ENABLE_MOONCAKE` |
| `MOONCAKE_RDMA` | `OFF` | Maps to `DS_MOONCAKE_WITH_RDMA` |
| `TT_METAL_SOURCE_DIR` | `./tt-metal` (bundled submodule) | tt-metal source root |
| `CMAKE_PREFIX_PATH` | Auto-derived from `TT_METAL_SOURCE_DIR` | CMake search paths |

Source: `build.sh`, lines 5-9.

The `--with-metal` mode performs validation before building:

1. Checks that `TT_METAL_SOURCE_DIR` points to an initialized submodule (or set explicitly)
2. Verifies that tt-metal has been built (looks for `tt-metaliumConfig.cmake` or `tt-metalium-config.cmake` in `build/lib/cmake/tt-metalium/`)
3. Auto-derives `CMAKE_PREFIX_PATH` if not set

Source: `build.sh`, lines 92-119.

The `--with-metal` mode builds targets sequentially: first the umbrella test target, then each tool individually:

```bash
cmake --build "${BUILD_DIR}" --target tt_llm_engine_tests -j "${JOBS}"
cmake --build "${BUILD_DIR}" --target dummy_pipeline_launcher -j "${JOBS}"
cmake --build "${BUILD_DIR}" --target dummy_pipeline_connector -j "${JOBS}"
if [[ "${NO_TOKENIZER}" != "ON" ]]; then
    cmake --build "${BUILD_DIR}" --target deepseek_inference_runner -j "${JOBS}"
fi
```

Source: `build.sh`, lines 136-141.

The `MOONCAKE_RDMA` variable auto-enables `MOONCAKE` when set:

```bash
if [[ "${MOONCAKE_RDMA}" == "ON" && "${MOONCAKE}" != "ON" ]]; then
    echo "WARN: MOONCAKE_RDMA=ON requires MOONCAKE=ON; enabling MOONCAKE." >&2
    MOONCAKE="ON"
fi
```

Source: `build.sh`, lines 11-14.

### setup.sh

The setup script is a higher-level wrapper that combines tt-metal and tt-llm-engine builds:

| Mode | Flag | Description |
|------|------|-------------|
| All (default) | `--all` | Build tt-metal, then tt-llm-engine with metal backend |
| Metal only | `--metal-only` | Build tt-metal only |
| DS standalone | `--ds-standalone` | Build tt-llm-engine core only |
| DS full | `--ds-full` | Build tt-llm-engine with metal backend (tt-metal must already be built) |

Source: `setup.sh`, lines 72-92.

The `build_metal()` function delegates to tt-metal's own build script:

```bash
build_metal() {
    echo "=== Building tt-metal ==="
    if [[ ! -d "${TT_METAL_DIR}" ]]; then
        echo "ERROR: tt-metal directory not found at ${TT_METAL_DIR}" >&2
        exit 1
    fi
    (cd "${TT_METAL_DIR}" && ./build_metal.sh)
}
```

Source: `setup.sh`, lines 41-50.

The `build_pm_full()` function sets the required environment variables and delegates to `build.sh --with-metal`:

```bash
build_pm_full() {
    export TT_METAL_SOURCE_DIR="${TT_METAL_DIR}"
    export CMAKE_PREFIX_PATH="${TT_METAL_DIR}/build/lib/cmake;${TT_METAL_DIR}/build_Release/_deps/nlohmann_json-build"
    TSAN="${TSAN}" JOBS="${JOBS}" NO_TOKENIZER="${NO_TOKENIZER}" \
        MOONCAKE="${MOONCAKE}" MOONCAKE_RDMA="${MOONCAKE_RDMA}" \
        "${DS_DIR}/build.sh" --with-metal
}
```

Source: `setup.sh`, lines 58-68.

## Install Rules

The install rules produce a `find_package(tt-llm-engine CONFIG)` consumable package. Both library variants are exported under the `TtLlmEngine::` namespace:

```cmake
install(
    TARGETS tt_llm_engine_core
    EXPORT tt-llm-engine-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
```

Source: `CMakeLists.txt`, lines 296-305.

If the Full library was built, it is added to the same export set:

```cmake
if(TARGET tt_llm_engine)
    install(
        TARGETS tt_llm_engine
        EXPORT tt-llm-engine-targets
        ...
    )
    set(TT_LLM_ENGINE_HAS_FULL ON)
else()
    set(TT_LLM_ENGINE_HAS_FULL OFF)
endif()
```

Source: `CMakeLists.txt`, lines 307-321.

The `TT_LLM_ENGINE_HAS_FULL` variable is substituted into the package config template (see [integration_methods.md](integration_methods.md)).

Public headers are installed as a directory:

```cmake
install(DIRECTORY include/tt_llm_engine DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
```

Source: `CMakeLists.txt`, line 323.

The export targets file and package config are installed to the standard CMake package location:

```cmake
install(
    EXPORT tt-llm-engine-targets
    FILE tt-llm-engine-targets.cmake
    NAMESPACE TtLlmEngine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tt-llm-engine
)
```

Source: `CMakeLists.txt`, lines 325-330.

Version compatibility uses `SameMajorVersion`:

```cmake
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/tt-llm-engine-config-version.cmake
    COMPATIBILITY SameMajorVersion
)
```

Source: `CMakeLists.txt`, lines 337-340.

## Complete Build Dependency Graph

```
TtLlmEngine::Core (always)
  +-- Threads::Threads

TtLlmEngine::Full (when TT::Metalium available)
  +-- TT::Metalium
  +-- Threads::Threads

test_decode_scheduler (when DS_BUILD_TESTS=ON)
  +-- TtLlmEngine::Core
  +-- gtest, gtest_main

test_decode_scheduler_device (when DS_BUILD_TESTS=ON + TT::Metalium)
  +-- TtLlmEngine::Full
  +-- gtest, gtest_main

dummy_pipeline_launcher (when TT::Metalium)
  +-- TT::Metalium
  +-- Threads::Threads

dummy_pipeline_connector (when TT::Metalium)
  +-- TtLlmEngine::Full
  +-- Threads::Threads

deepseek_inference_runner (when TT::Metalium + cargo)
  +-- TtLlmEngine::Full
  +-- Threads::Threads
  +-- tokenizers_cpp (FetchContent)

test_mooncake_transport (when DS_ENABLE_MOONCAKE=ON)
  +-- TtLlmEngine::Mooncake::TransferEngine
  +-- gtest, gtest_main

test_mooncake_store (when DS_ENABLE_MOONCAKE=ON)
  +-- TtLlmEngine::Mooncake::TransferEngine
  +-- TtLlmEngine::Mooncake::Store
  +-- gtest, gtest_main
```

## C++ Standard and Compiler Requirements

The project requires C++20 with no extensions:

```cmake
cmake_minimum_required(VERSION 3.24)
project(tt_llm_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```

Source: `CMakeLists.txt`, lines 1-6.

C++20 is needed for designated initializers (used throughout in struct construction), `std::atomic<T>` on aggregate types, and `<bit>` header utilities. Extensions are disabled for portability across GCC and Clang. The README specifies "GCC 11+, Clang 14+" as minimum compiler versions (`README.md`, line 43).

## Troubleshooting: Common Build Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `tt-metal submodule not initialized` | Submodule at `./tt-metal/` has no content | `git submodule update --init --recursive` |
| `tt-metal does not appear to be built` | No `tt-metaliumConfig.cmake` found | `cd tt-metal && ./build_metal.sh && cd ..` |
| `TT::Metalium not found -- building tt_llm_engine_core only` | tt-metal not built or `CMAKE_PREFIX_PATH` not set | Set `TT_METAL_SOURCE_DIR` and `CMAKE_PREFIX_PATH`, or use `./build.sh --with-metal` which auto-detects |
| `Rust/cargo not found -- skipping deepseek_inference_runner` | No Rust toolchain in PATH | Install via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` (>= 1.80 required) |
| `Compatibility with CMake < 3.5 has been removed` | tokenizers-cpp's vendored msgpack triggers CMake policy error | Delete `build-full/CMakeCache.txt` and reconfigure. The `CMAKE_POLICY_VERSION_MINIMUM` workaround should handle this automatically. |
| `invalid metadata` or `requires rustc 1.79 or newer` | System-packaged Rust (via apt) is too old and takes precedence over rustup | Ensure `~/.cargo/bin` comes first in PATH. Delete `build-full/CMakeCache.txt` to clear the cached `cargo` path. |
| `Ninja does not match Unix Makefiles` | Stale tt-metal build directory from a different generator | `rm -rf tt-metal/build_Release && cd tt-metal && ./build_metal.sh` |
| `DS_ENABLE_MOONCAKE=ON but the submodule at ... is not initialized` | Mooncake or a vendored dependency submodule not initialized | `git submodule update --init --recursive` |
| `target 'transfer_engine' was not produced` | Mooncake submodule version mismatch or configuration issue | Check Mooncake's CMakeLists for current target names; update `cmake/mooncake.cmake` |
| `glog/export.h not found` during Mooncake build | glog build directory include path ordering issue | This should be handled by the per-target patching in `cmake/mooncake.cmake`. If it persists, clean the build directory. |

---

[< Previous: Chapter Overview](index.md) | [Next: Integration Methods >](integration_methods.md)
