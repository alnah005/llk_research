# Chapter 5: Pipeline and Wire Format

## The Abstraction Boundary

Every interaction between the `DecodeScheduler` and the hardware (or any test double standing in for it) flows through a single polymorphic seam: `PipelineInterface`. The scheduler never knows -- and never needs to know -- whether tokens are being processed by a real Tenstorrent device over metalium sockets, a timing-accurate simulator, or a trivial in-memory queue used by a unit test. This chapter documents that contract, the three concrete backends that fulfill it, the variant-based configuration system that selects among them, and the binary page layout that crosses the host-device boundary.

The pipeline abstraction is the **sole I/O boundary** in the engine's critical path. The Writer thread calls `inject()` and the Reader thread calls `read_result()` (details in [PipelineInterface](./pipeline_interface.md)). Because the scheduler is fully decoupled from the transport layer, the entire decode algorithm (Ch3) and the speculative decode protocol (Ch4) can be tested at full fidelity without hardware.

## The Testing Pyramid

The pipeline layer forms a clean testing pyramid:

```
                    +---------------------+
                    |   SocketPipeline    |   Production / HW integration
                    |   (real sockets)    |
                    +----------+----------+
                               |
                   +-----------+-----------+
                   |  PipelineSimulator    |   Timing-accurate integration
                   |  (backpressure,       |   tests, CI without HW
                   |   per-token timing)   |
                   +-----------+-----------+
                               |
              +----------------+----------------+
              |          MockPipeline           |   Unit tests, token-model
              |  (instant, queue-based,         |   verification, fuzz
              |   deterministic inject log)     |
              +----------------+----------------+
```

| Layer | Latency | Backpressure | Timing Fidelity | Hardware Required |
|-------|---------|-------------|-----------------|-------------------|
| **MockPipeline** | Zero (or injected) | No | None | No |
| **PipelineSimulator** | Modelled | Yes (stage-limited) | Sub-microsecond | No |
| **SocketPipeline** | Real | Yes (socket flow control) | Real | Yes |

Every scheduler test in the codebase uses one of the first two layers. The speculative-decode state machine (Ch4), the chunked-prefill pipeline (Ch3), and the writer-priority scheduler (Ch3) are all fully testable on a laptop without any accelerator hardware.

## Selecting a Backend: The PipelineConfig Variant

Backend selection is a compile-time-safe, runtime decision:

```cpp
using PipelineConfig = std::variant<MockConfig, SocketConfig, PipelineSimulatorConfig>;
```

The engine entry point uses `std::visit` to construct the appropriate concrete pipeline:

```cpp
auto pipeline = std::visit(overloaded{
    [](const MockConfig& cfg) -> std::unique_ptr<PipelineInterface> {
        return std::make_unique<MockPipeline>(
            cfg.latency_min_us, cfg.latency_max_us, cfg.seed, cfg.accept_rate);
    },
    [](const SocketConfig& cfg) -> std::unique_ptr<PipelineInterface> {
        return std::make_unique<SocketPipeline>(
            cfg.h2d_socket_id, cfg.d2h_socket_id,
            cfg.connect_timeout_ms, cfg.use_deepseek_md_format);
    },
    [](const PipelineSimulatorConfig& cfg) -> std::unique_ptr<PipelineInterface> {
        return std::make_unique<PipelineSimulator>(
            cfg.num_stages, cfg.stage_duration_us, cfg.decode_token_id,
            cfg.accept_rate, cfg.seed, cfg.safe_vocab_base, cfg.safe_vocab_modulus);
    },
}, config);
```

Three properties of this pattern: (1) **Exhaustiveness** -- adding a new config alternative without handling it is a compile error. (2) **No downcasting** -- once constructed, the scheduler holds only a `std::unique_ptr<PipelineInterface>`. (3) **Single point of selection** -- this is the only place in the codebase that knows which backends exist.

## Chapter Map

| Section | What It Covers |
|---------|---------------|
| [PipelineInterface](./pipeline_interface.md) | The abstract contract: five virtual methods, descriptor structs, sentinel values, two-phase shutdown protocol |
| [MockPipeline](./mock_pipeline.md) | In-process queue-based backend for unit testing with deterministic token model, accept-rate control, and inject log |
| [PipelineSimulator](./pipeline_simulator.md) | Timing-accurate systolic pipeline model with backpressure, throughput caps, and safe-vocab modular arithmetic |
| [SocketPipeline](./socket_pipeline.md) | Real hardware backend over tt-metalium H2D/D2H sockets with PIMPL encapsulation and sentinel-based shutdown |
| [Wire Format](./wire_format.md) | 256-byte page layout for both default and DeepSeek metadata formats, serialization, and deserialization |

---

| | Navigation | |
|:---|:---:|---:|
| [Chapter 4: Speculative Decode](../ch4_speculative_decode/index.md) | [Table of Contents](../index.md) | [PipelineInterface](./pipeline_interface.md) |
