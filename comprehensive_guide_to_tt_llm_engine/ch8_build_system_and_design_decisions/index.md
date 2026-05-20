# Chapter 8: Build System and Design Decisions

[< Previous: Chapter 7 -- Transport and Tools](../ch7_transport_and_tools/index.md)

---

This chapter documents the CMake build system that produces the two library variants of tt-llm-engine, the build and setup scripts that drive the build, the two integration methods available to consumers, and the key design decisions and tradeoffs that shaped the engine's architecture across all subsystems.

## Quick Reference: Common Tasks

| Task | Command / Steps |
|------|----------------|
| **Build standalone (no hardware)** | `./build.sh --standalone` |
| **Build with tt-metal backend** | `git submodule update --init --recursive && cd tt-metal && ./build_metal.sh && cd .. && ./build.sh --with-metal` |
| **Build everything in one step** | `./setup.sh` (or `./setup.sh --all`) |
| **Build tt-metal only** | `./setup.sh --metal-only` |
| **Build with Mooncake transport** | `MOONCAKE=ON ./build.sh --standalone` |
| **Build with RDMA transport** | `MOONCAKE=ON MOONCAKE_RDMA=ON ./build.sh --standalone` |
| **Build with ThreadSanitizer** | `TSAN=ON ./build.sh --standalone` |
| **Build without tokenizer (skip Rust)** | `NO_TOKENIZER=ON ./build.sh --with-metal` |
| **Run mock tests (no hardware)** | `LD_LIBRARY_PATH=build-standalone build-standalone/test_decode_scheduler` |
| **Run device tests (requires hardware)** | `export TT_METAL_RUNTIME_ROOT=$(pwd)/tt-metal && LD_LIBRARY_PATH=build-full build-full/test_decode_scheduler_device` |
| **Run filtered tests** | `LD_LIBRARY_PATH=build-full build-full/test_decode_scheduler --gtest_filter="*Prefill*"` |
| **Install for find_package** | `cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/opt/tt-llm-engine && cmake --build build -j$(nproc) && cmake --install build` |
| **Integrate via add_subdirectory** | Add as submodule, then `add_subdirectory(external/tt-llm-engine)` and link `TtLlmEngine::Core` or `TtLlmEngine::Full` |

## Chapter Files

| File | Contents |
|------|----------|
| [build_system.md](build_system.md) | CMake build system: library targets, options, conditional compilation, build scripts, test/tool targets, Mooncake integration, install rules |
| [integration_methods.md](integration_methods.md) | How to integrate tt-llm-engine via `add_subdirectory` or `find_package`, runtime environment, C++ usage example |
| [design_decisions.md](design_decisions.md) | Sixteen key design decisions with source-level rationale, rejected alternatives, and cross-references |

## Build Artifacts Summary

| Build Mode | Output Directory | Libraries | Test Binaries | Tools |
|------------|-----------------|-----------|---------------|-------|
| `--standalone` | `build-standalone/` | `libtt_llm_engine_core.so` | `test_decode_scheduler` | -- |
| `--with-metal` | `build-full/` | `libtt_llm_engine_core.so`, `libtt_llm_engine.so` | `test_decode_scheduler`, `test_decode_scheduler_device` | `dummy_pipeline_launcher`, `dummy_pipeline_connector`, `deepseek_inference_runner`* |
| `--standalone` + `MOONCAKE=ON` | `build-standalone/` | `libtt_llm_engine_core.so` | `test_decode_scheduler`, `test_mooncake_transport`, `test_mooncake_store` | -- |

\* `deepseek_inference_runner` is only built when Rust/cargo is available and `DS_NO_TOKENIZER` is not `ON`.

## Environment Variables Reference

| Variable | Used By | Default | Description |
|----------|---------|---------|-------------|
| `TT_METAL_SOURCE_DIR` | `build.sh`, CMake | `./tt-metal` (submodule) | Path to tt-metal source tree. Required for `--with-metal` if not using the bundled submodule. |
| `CMAKE_PREFIX_PATH` | `build.sh`, CMake | Auto-derived from `TT_METAL_SOURCE_DIR` | Semicolon-separated list of CMake search paths for installed packages (tt-metalium, nlohmann_json). |
| `TT_METAL_RUNTIME_ROOT` | Runtime | -- (must be set) | Path to tt-metal repo root. Required when running device tests or any `TtLlmEngine::Full` code outside the tt-metal directory. The metalium runtime uses it to locate kernel sources, firmware, and the SFPI compiler. |
| `TT_VISIBLE_DEVICES` | Runtime | all devices | Restricts which Tenstorrent devices are visible. Set to `0` for single-chip testing. |
| `JOBS` | `build.sh`, `setup.sh` | `$(nproc)` | Number of parallel build jobs. |
| `TSAN` | `build.sh`, `setup.sh` | `OFF` | `ON`/`OFF` -- enable ThreadSanitizer for all tt-llm-engine targets. |
| `NO_TOKENIZER` | `build.sh` | `OFF` | `ON`/`OFF` -- skip Rust/tokenizers-cpp build (disables `deepseek_inference_runner`). |
| `MOONCAKE` | `build.sh`, `setup.sh` | `OFF` | `ON`/`OFF` -- build with Mooncake transport (Transfer Engine + Store). |
| `MOONCAKE_RDMA` | `build.sh`, `setup.sh` | `OFF` | `ON`/`OFF` -- include Mooncake's RDMA transport (implies `MOONCAKE=ON`). |

## Relationship to Prior Chapters

This chapter references specific mechanisms from every prior chapter:

- **Ch1 (Architecture):** The two-library architecture (Core vs Full) and the IS/PM ownership split are the top-level design decisions that drive the build structure.
- **Ch2 (Threading and Data Structures):** Fixed-size pre-allocated data structures, bitmap allocators, bounded queues, lock-free hot path -- all design decisions documented here with references to their implementations.
- **Ch3 (Scheduling):** Decode priority over prefill, chunked prefill size, FIFO vs priority scheduling.
- **Ch4 (Speculative Decode):** Spec decode bandwidth cost, no KV rollback, defer_complete.
- **Ch5 (Pipeline):** The `PipelineInterface` abstraction that enables the Core/Full split; `DS_HAS_METALIUM` compile define.
- **Ch6 (Lifecycle):** KV cache retention on COMPLETE, deferred cancellation, generation counters.
- **Ch7 (Transport and Tools):** Mooncake build integration, tool binary dependencies, tokenizer build.

## Reading Paths

- **IS integrators:** Read [build_system.md](build_system.md) (library variants table and CMake options) then [integration_methods.md](integration_methods.md) (linking guide).
- **Scheduler developers:** Read [design_decisions.md](design_decisions.md) in full, cross-referencing the Ch2-Ch4 files it points to.
- **DevOps / build engineers:** Read [build_system.md](build_system.md) in full, including the Mooncake and TSAN sections, then [integration_methods.md](integration_methods.md).

---

[Next: Build System >](build_system.md)
