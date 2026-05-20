# System Position

tt-llm-engine is the host-side control-plane scheduling engine that connects Inference Server stacks to persistent inference workloads running on Tenstorrent hardware. It does not own the HTTP frontend, and it does not own the model. It is the bridge between the two -- a scheduling layer that drives token traffic at pipeline-tick granularity across host-to-device (H2D) and device-to-host (D2H) sockets.

## The Three-Layer Stack

The Tenstorrent inference deployment consists of three distinct layers, each with different lifetimes, timescales, and responsibilities:

```
 +--------------------------------------------------------------+
 |  Layer 1: Inference Server (IS)                              |
 |                                                              |
 |  tt-inference-server, custom HTTP/gRPC server,               |
 |  batching frontend, WebSocket gateway, etc.                  |
 |                                                              |
 |  Owns: HTTP endpoints, TLS, JSON parsing, tokenization,     |
 |        chat history, session management, auth, rate limiting |
 |  Timescale: milliseconds                                     |
 +----------------------------+---------------------------------+
                              |  ISRequest / SchedulerResponse / OutputMessage
                              |  (bounded queues -- fixed-size messages)
                              v
 +--------------------------------------------------------------+
 |  Layer 2: tt-llm-engine (this repository)                    |
 |                                                              |
 |  tt_llm_engine::scheduler::decode   (Decode Scheduler)       |
 |  tt_llm_engine::scheduler::prefill  (planned)                |
 |  tt_llm_engine::scheduler::kv_migration (planned)            |
 |  tt_llm_engine::pipeline            (PipelineInterface)      |
 |                                                              |
 |  Owns: user_id allocation, prompt storage, prefill/decode    |
 |        scheduling, pipeline injection, in-flight tracking,   |
 |        KV cache lifecycle execution                          |
 |  Timescale: microseconds                                     |
 +----------------------------+---------------------------------+
                              |  H2D / D2H sockets
                              |  (256-byte pages -- serialized InjectDescriptor / ResultDescriptor)
                              v
 +--------------------------------------------------------------+
 |  Layer 3: Persistent Inference Workload                      |
 |                                                              |
 |  Model running on Tenstorrent cluster hardware,              |
 |  launched via tt-metal directly or through tt-blaze           |
 |                                                              |
 |  Owns: model weights, KV cache memory, attention compute,    |
 |        MTP prediction heads, device-side sampling            |
 |  Timescale: sub-microsecond (pipeline stage period)          |
 +--------------------------------------------------------------+
```

Each layer is independently deployable. The IS can run on a different machine from the engine. The engine can run on the host CPU while the model occupies the accelerator cluster. Communication between layers uses explicit, bounded interfaces with no shared memory outside of the queue structures.

Each layer boundary is a well-defined protocol surface:

- **IS <-> tt-llm-engine:** Three bounded queues carrying fixed-size messages (`ISRequest`, `SchedulerResponse`, `OutputMessage`). No shared memory. No direct function calls. See [Chapter 6 -- IS Integration Patterns](../ch6_lifecycle_is_integration_kv_tiering/is_integration_patterns.md).

- **tt-llm-engine <-> Device Workload:** H2D and D2H sockets exchanging 256-byte page buffers (`InjectDescriptor` / `ResultDescriptor` serialized as `PageBuffer = std::array<uint32_t, 64>`). See [Chapter 5 -- Wire Format](../ch5_pipeline_and_wire_format/wire_format.md).

## The Bridge Metaphor

The defining characteristic of tt-llm-engine is that **the model is long-lived**. A typical deployment launches the model workload once -- it stays resident on the Tenstorrent cluster (a "semi-hardened" deployment). The model process creates H2D and D2H sockets and then waits for traffic. tt-llm-engine **attaches** to these pre-existing sockets and drives token traffic for many concurrent users without ever launching or tearing down the model.

This is the "bridge" metaphor from the source documentation (`docs/overview.md`):

> This repo is the **bridge** between an Inference Server Stack and a persistent / semi-hardened inference workload running on Tenstorrent hardware. The inference workload itself is launched separately -- and exposes H2D/D2H sockets that this engine drives at token granularity.

The practical implications of this architecture are:

1. **No model lifecycle management.** The engine assumes the model is already running and its sockets are ready; it never compiles, loads, or unloads models.
2. **Socket-level attachment.** The `SocketPipeline` connects to pre-existing named H2D/D2H sockets -- it does not create them (see [Chapter 5](../ch5_pipeline_and_wire_format/index.md) for connection details).
3. **No model graph knowledge.** The engine operates exclusively on token IDs, positions, and slot IDs; model architecture and KV cache geometry are determined by the model launch.
4. **Independent restarts.** Either the engine or the model can restart independently, though in-flight tokens are lost and active users must be re-allocated.

## Semi-Hardened Deployment Pattern

The deployment pattern is "semi-hardened": the model is launched once and stays resident on the Tenstorrent cluster for the duration of a service lifetime. This is driven by model load times (minutes for large models) versus request latencies (milliseconds to seconds).

- **Hardened** in the sense that the model is launched once and runs continuously, serving many user sessions without reloading. KV caches persist across requests within a user session (enabling multi-turn conversations) and even across sessions when the slot is retained in COMPLETE state.

- **Semi** in the sense that the model is not yet a firmware-level appliance. It is still a user-space process launched via `tt-metal` or `tt-blaze`, running on the host and delegating compute to the accelerator cluster. The engine itself is designed with a hardware deployment path in mind (see [Key Invariants](./key_invariants.md)), but neither the engine nor the model currently runs as embedded firmware.

```
Timeline:
  |--- model launch (minutes) ---|-- tt-llm-engine connects --|
  |                               |                            |
  |                               |-- user requests stream --> |
  |                               |-- tokens generated ------> |
  |                               |                            |
  |                               |-- engine restarts -------->|
  |                               |-- re-connects to sockets ->|
  |                               |                            |
  |------------- model stays resident for days/weeks -------->|
```

This semi-hardened model is visible in the practical tooling:

- The `dummy_pipeline_launcher` tool opens a `MeshDevice`, creates H2D/D2H sockets, compiles and launches a loopback kernel, and then *waits* -- it does not drive traffic itself.
- The `dummy_pipeline_connector` connects to those pre-existing sockets, instantiates a `DecodeScheduler`, and drives the traffic.
- The DeepSeek inference runner uses the same pattern: the model pipeline is launched with `--launch-only` in one terminal, and the runner connects to the exported sockets in another.

The engine does not attempt to recover from device failures -- if the model crashes, the sockets break, and the engine's `SocketPipeline` reports failure. Recovery is handled at a higher level (orchestration layer re-launches the model, engine re-connects).

## Upstream IS Stacks

tt-llm-engine is designed to work with multiple IS implementations:

- **[tt-inference-server](https://github.com/tenstorrent/tt-inference-server)** -- Tenstorrent's reference inference server with OpenAI-compatible HTTP endpoints. Not a build dependency -- it communicates with the engine purely through the three bounded queues.
- **Custom HTTP/gRPC servers** -- Any server that implements the `ISRequest`/`SchedulerResponse`/`OutputMessage` queue protocol can serve as the IS layer.
- **Batching frontends** -- Systems that aggregate requests from multiple clients and submit them as a batch.

The engine does not impose any particular API shape on the IS. All communication flows through three bounded queues that carry fixed-size messages. The IS is free to expose whatever external API it chooses -- REST, gRPC, WebSocket, SSE -- and translate internally.

## Downstream Device Workloads

The model workload can be launched through:

- **[tt-metal](https://github.com/tenstorrent/tt-metal)** -- Tenstorrent's low-level programming model and runtime. Provides device management, kernel compilation, socket APIs (`H2DSocket`, `D2HSocket`), and the metalium runtime. The `SocketPipeline` backend connects to sockets created by `tt-metal` programs. This is the current production path.
- **[tt-blaze](https://github.com/tenstorrent/tt-blaze)** -- A higher-level runtime built on `tt-metal`. Provides a more ergonomic model authoring experience while compiling down to the same device kernels and exposing the same H2D/D2H socket interface.

Both paths produce the same interface visible to tt-llm-engine: H2D and D2H sockets that accept 256-byte pages containing serialized `InjectDescriptor` and `ResultDescriptor` messages (see [Chapter 5](../ch5_pipeline_and_wire_format/index.md) for the wire format details).

tt-llm-engine has a compile-time dependency on tt-metalium only for the Full library variant (`libtt_llm_engine.so`). The Core library (`libtt_llm_engine_core.so`) contains all scheduling logic and the `MockPipeline`/`PipelineSimulator` backends, and can be built, tested, and integrated without any Tenstorrent hardware or software dependencies beyond a C++20 compiler and pthreads.

## Repository URL Transitional Note

The repository currently lives at `github.com/tt-asaigal/tt-llm-engine` while the transfer to `github.com/tenstorrent/tt-llm-engine` is pending organization permissions. After the transfer, GitHub maintains a permanent redirect from the old URL, so existing `git clone` and `git submodule add` commands continue to work. To canonicalize after transfer: `sed -i 's|tt-asaigal/tt-llm-engine|tenstorrent/tt-llm-engine|g'`.

---

**Next:** [`ownership_contract.md`](./ownership_contract.md)
