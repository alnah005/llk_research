# Chapter 2: JSON Configuration Schema and Parsing

The Pi0.5 pipeline runner is driven entirely by a JSON configuration file that defines pipeline phases, socket parameters, pixel payload dimensions, and runtime tunables. This chapter documents the configuration schema in full, walks through the hand-rolled recursive-descent parser that consumes it, and provides numerical walkthroughs for each of the three reference configuration files shipped with the project.

## Chapter Contents

| # | File | Description |
|---|------|-------------|
| 1 | [`01_config_schema_reference.md`](./01_config_schema_reference.md) | Complete schema reference for `PipelineConfig`, `SocketConfigParams`, and `PixelPayloadConfig` -- field types, defaults, validation rules, and CLI override precedence |
| 2 | [`02_custom_json_parser.md`](./02_custom_json_parser.md) | Anatomy of the ~120-line hand-rolled recursive-descent JSON parser: design rationale, function-by-function walkthrough, `skip_json_value` forward-compatibility mechanism, comma-handling strategies, and known limitations |
| 3 | [`03_reference_configs_walkthrough.md`](./03_reference_configs_walkthrough.md) | Numerical walkthroughs of `pi05_config.json` (full socket mode), `pi05_config_hybrid.json` (bulk-only hybrid), and `pi05_config_sim.json` (simulation-only), including effective stage counts, weighted-average timing derivations, and the single-action-token output model |

## Prerequisites

This chapter assumes familiarity with the three-phase pipeline model (vision / text / denoise) and the `PhaseConfig` struct introduced in [Chapter 1](../ch1_architecture_overview_and_operating_modes/index.md). The flattening math that converts multi-phase configurations into a single `PipelineSimulatorConfig` is derived here and consumed by the timing model documented in [Chapter 3](../ch3_pipelinesimulator_timing_model/index.md).

## Key Source Files

| File | Role |
|------|------|
| `examples/pi05_pipeline_runner.cpp` | Config structs, parser, CLI argument handling, validation |
| `examples/pi05_config.json` | Full socket mode reference config (8 users, 20 turns) |
| `examples/pi05_config_hybrid.json` | Bulk-only hybrid mode reference config (1 user, 400 turns) |
| `examples/pi05_config_sim.json` | Simulation-only reference config (8 users, 20 turns) |

---

**Previous:** [Chapter 1 -- Pi0.5 Architecture and Pipeline Model](../ch1_architecture_overview_and_operating_modes/index.md)
**Next:** [Chapter 3 -- PipelineSimulator Timing Model](../ch3_pipelinesimulator_timing_model/index.md)
