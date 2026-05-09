# Chapter 8 -- Multi-Device Scaleout, Demo Runtime, and Full Decode Trace

Chapters 1-7 built up every subsystem in isolation: the unified kernel system, individual micro-ops and fused ops, the attention and MoE deep dives, and weight preparation. This chapter is the capstone. It shows how all those pieces compose into a working system: how the 63-stage pipeline maps onto Galaxy pod hardware, how devices communicate, how the demo runtime orchestrates inference, and what a complete decode step looks like end-to-end with tensor shapes at every boundary.

The chapter is organized in four sections:

1. **Pipeline parallelism and configuration** -- 63 logical stages mapped to physical hosts via snake-order traversal, YAML/textproto configuration files, mesh graph descriptors, and superpod topology.
2. **Multi-device communication and parallelism** -- D2D exchange sockets, pipeline blocks, host I/O with HOST_PUSH/DEVICE_PULL modes, intra-device TP=2/EP=8, and CCL collective operations.
3. **Demo inference runtime** -- the CLI entry point, weight loading dispatch, the generation loop, prefill/decode lifecycle, and the termination protocol.
4. **Full decode step data flow** -- an annotated trace from host token write through embedding, attention, MoE, LM head, argmax, and D2H return, with tensor shapes and data sizes at every stage boundary.

Source root: `models/demos/deepseek_v3_b1/`

---

## Contents

### [8.1 Pipeline Parallelism and Configuration](./01_pipeline_parallelism_and_configuration.md)

| Topic | Key Concepts |
|-------|-------------|
| Pipeline topology | 63 stages: embedding (0), dense (1-3), MoE (4-61), LM head (62) |
| Host traversal | Snake-order across Galaxy pod: u08 → u02 → u03 → u07 (or reversed) |
| Configuration files | Per-topology YAML (mesh shape, host map) + textproto (mesh graph descriptor) |
| `generate_blitz_decode_pipeline_configs.py` | Generates both YAML and textproto from topology definition |
| Superpod scaling | 2-pod (126 stages) and superpod (315 stages) multi-pod configurations |
| Design rationale | Why pipeline parallelism over tensor parallelism for inter-host scaleout |

### [8.2 Multi-Device Communication and Parallelism](./02_multi_device_communication_and_parallelism.md)

| Topic | Key Concepts |
|-------|-------------|
| D2D exchange | `SocketInterface` with upstream/downstream/internal sockets; termination semaphore |
| Pipeline block | `PipelineBlock` encapsulates per-stage D2D forwarding; stage 0 adds `HostInterface` |
| Host I/O | HOST_PUSH (host writes, device reads) vs DEVICE_PULL (host pre-writes, device pulls at own pace) |
| Intra-device parallelism | MLA TP=2 (column-parallel), MoE EP=8 (full 4x2 mesh) |
| CCL operations | AllReduce (neighbor exchange), Broadcast (dual-axis), ReduceToOne (3-level tree) |
| Loopback ring | Token loops from stage 62 back to stage 0 via D2D fabric, avoiding PCIe round-trip |

### [8.3 Demo Inference Runtime](./03_demo_inference_runtime.md)

| Topic | Key Concepts |
|-------|-------------|
| CLI entry point | `demo/cli.py`: mesh construction, weight loading dispatch, model construction |
| Weight loading | Per-mesh-ID dispatch to loader functions; fast dispatch for MoE expert bulk loading |
| `ModelLike` protocol | 4-method interface (`start`/`prefill`/`decode_step`/`stop`) for model-agnostic runner |
| Generation loop | `demo/runner.py`: tokenization, streaming output, clean shutdown |
| Prefill/decode lifecycle | IDLE → PREFILL → DECODE state machine; `_switch_to_decode()` with sync_devices |
| Termination protocol | 4-step barrier+semaphore+sync sequence; HOST_PUSH vs DEVICE_PULL differences |

### [8.4 Full Decode Step Data Flow](./04_full_decode_step_data_flow.md)

| Topic | Key Concepts |
|-------|-------------|
| End-to-end trace | Host → H2D → embedding → attention → MoE → LM head → argmax → D2H → host |
| Tensor shape annotations | Shapes at every stage boundary (e.g., $[1, 7168]$ hidden state, $[1, 576]$ compressed KV) |
| MoE block data flow | Gate → broadcast → expert dispatch → ReduceToOne at integration level |
| Residual connection algebra | Algebraic trace through attention and MoE residual streams |
| Pipeline bottleneck analysis | D2D transfer time vs compute time per stage; critical path identification |
| Hardware-software co-design | 10-row summary table mapping each software decision to its hardware constraint |

---

**Previous:** [Chapter 7 -- Weight Preparation and Blitz Caching](../ch07_weight_preparation_and_blitz_caching/index.md)
