# Comprehensive Understanding of the Pi0.5 Simulation and Benchmarking Framework in tt-llm-engine

A deep technical guide to the Pi0.5 pipeline simulation, bulk socket protocol, device kernel architecture, and benchmark orchestration infrastructure inside tt-llm-engine. Written for systems engineers who need to run, debug, extend, or replace the Pi0.5 benchmarking stack.

---

## How to Use This Guide

| Your goal | Recommended reading path |
|---|---|
| I want to run the benchmark | [Ch 7](ch7_metrics_reporting_and_benchmark_orchestration/index.md) (run_hybrid.sh) --> [Ch 2](ch2_json_configuration_schema_and_parsing/index.md) (config) --> [Ch 1](ch1_architecture_overview_and_operating_modes/index.md) (modes) |
| I want to understand the architecture | [Ch 1](ch1_architecture_overview_and_operating_modes/index.md) --> [Ch 2](ch2_json_configuration_schema_and_parsing/index.md) --> [Ch 3](ch3_pipelinesimulator_timing_model/index.md) |
| I want to understand the device-side kernel | [Ch 5](ch5_device_launcher_and_kernel_architecture/index.md) --> [Ch 4](ch4_bulk_h2d_d2h_socket_protocol/index.md) |
| I want to know what could go wrong | [Ch 8](ch8_design_critique_failure_modes_and_gotchas/index.md) --> [Ch 4](ch4_bulk_h2d_d2h_socket_protocol/index.md) (race conditions) |
| I want to build a production kernel | [Ch 8](ch8_design_critique_failure_modes_and_gotchas/index.md) (production differences) --> [Ch 5](ch5_device_launcher_and_kernel_architecture/index.md) --> [Ch 4](ch4_bulk_h2d_d2h_socket_protocol/index.md) |

---

## Chapter Index

| Chapter | Title | Description |
|---|---|---|
| 1 | [Architecture Overview and Operating Modes](ch1_architecture_overview_and_operating_modes/index.md) | The three-phase pipeline model (vision, text, denoise), three operating modes (simulation, full socket, hybrid/bulk-only), and how all components compose into the end-to-end system. |
| 2 | [JSON Configuration Schema and Parsing](ch2_json_configuration_schema_and_parsing/index.md) | The JSON configuration schema that drives the pipeline runner, the hand-rolled recursive-descent parser, and numerical walkthroughs of the three reference configs. |
| 3 | [PipelineSimulator Timing Model](ch3_pipelinesimulator_timing_model/index.md) | The per-token timestamp simulation engine that models systolic pipeline latency, throughput cap, and backpressure without a tick thread. |
| 4 | [Bulk H2D/D2H Socket Protocol](ch4_bulk_h2d_d2h_socket_protocol/index.md) | The dedicated bulk socket pair for pixel-frame and action-tensor transfers, including wire protocol, page-alignment rules, sentinel shutdown, and concurrency hazards. |
| 5 | [Device Launcher and Kernel Architecture](ch5_device_launcher_and_kernel_architecture/index.md) | The two-process split between host runner and device launcher, from fork() through kernel execution, covering the bulk passthrough kernel and NOC chunking utilities. |
| 6 | [Decode Scheduler Integration](ch6_decode_scheduler_integration/index.md) | How Pi0.5 wires its multi-turn robotics inference loop into the DecodeScheduler via ALLOCATE/SUBMIT/EVICT, the skip_eos_writeback optimization, and per-user session management. |
| 7 | [Metrics, Reporting, and Benchmark Orchestration](ch7_metrics_reporting_and_benchmark_orchestration/index.md) | Every metric computed by the benchmark, the exact formulas and code paths, output formats per mode, and the run_hybrid.sh end-to-end orchestration script. |
| 8 | [Design Critique, Failure Modes, and Gotchas](ch8_design_critique_failure_modes_and_gotchas/index.md) | Critical evaluation of design decisions, catalog of failure modes, behavioral differences between passthrough and production kernels, and operational gotchas for real hardware. |

---

## Quick Reference

| Concept / Value | Where to learn more |
|---|---|
| Three operating modes (simulation, full socket, hybrid/bulk-only) | [Ch 1](ch1_architecture_overview_and_operating_modes/index.md) |
| `BULK_PAGE_SIZE = 4096` | [Ch 4](ch4_bulk_h2d_d2h_socket_protocol/index.md) |
| `output_tokens = 1` semantics | [Ch 2](ch2_json_configuration_schema_and_parsing/index.md), [Ch 6](ch6_decode_scheduler_integration/index.md) |
| PipelineSimulator invariants | [Ch 3](ch3_pipelinesimulator_timing_model/index.md) |
| Sentinel shutdown protocol | [Ch 4 -- Shutdown Protocol](ch4_bulk_h2d_d2h_socket_protocol/03_shutdown_protocol.md) |
| `run_hybrid.sh` workflow | [Ch 7 -- run_hybrid.sh](ch7_metrics_reporting_and_benchmark_orchestration/04_run_hybrid_orchestration.md) |

---

## Prerequisites

- **C++20** -- the codebase uses `std::span`, `std::jthread`, designated initializers, and `<format>` throughout.
- **Basic PCIe/DMA concepts** -- page alignment, host-to-device transfers, NOC routing.
- **Pipeline/systolic-array intuition** -- understanding of multi-stage pipelines with per-stage latency and throughput caps.
- **Familiarity with the Pi0.5 model** is helpful but not required; the guide explains all Pi0.5-specific semantics (vision/text/denoise phases, action-token outputs) as they arise.

---

## Source Code Location

All source code referenced in this guide lives in:

```
/localdev/salnahari/testing_dir/tt-llm-engine
```
