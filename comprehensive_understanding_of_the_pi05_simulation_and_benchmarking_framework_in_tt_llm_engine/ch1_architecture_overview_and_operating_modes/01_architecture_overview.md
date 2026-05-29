# 1.1 Architecture Overview

## What is Pi0.5?

Pi0.5 is a Vision-Language-Action (VLA) model designed for real-time robotic control. Its neural architecture comprises three sequential stages:

1. **SigLIP vision encoder** -- ingests camera frames (typically 224x224x3, 3 frames) and produces visual embeddings.
2. **Gemma text backbone** -- fuses the visual embeddings with a tokenized instruction prompt to produce language-grounded representations.
3. **Euler flow-matching denoise loop** -- iteratively denoises an action distribution over multiple steps (typically 5) to produce a 50x32 bfloat16 action tensor (3,200 bytes) that drives the robot's actuators.

Each inference request follows a distinct pattern: a large pixel payload flows host-to-device (H2D), the three neural phases execute on-device, and a small action tensor flows device-to-host (D2H). This asymmetric data flow -- large H2D, small D2H, multi-phase on-device compute -- motivates a dedicated benchmarking framework rather than reusing generic LLM serving infrastructure.

The **Pi0.5 benchmarking framework** (`pi05_pipeline_runner`) exists to:

- Model the multi-phase pipeline timing without requiring real model weights or full hardware.
- Validate end-to-end data flow (pixel ingestion, token scheduling, action output) across different hardware configurations.
- Measure key latency metrics (TTFT, ITL, TPOT) and bulk transfer bandwidth under concurrent multi-user robotics workloads.
- Provide a CI-friendly simulation mode that exercises scheduler logic without any hardware dependency.

## Essential Tenstorrent Concepts

Before diving into the framework, a brief primer on the hardware abstractions used throughout this guide:

| Concept | Description |
|---------|-------------|
| **MeshDevice** | Multi-chip topology manager. Creates and manages a grid of Tenstorrent accelerator chips. In full socket mode, `MeshDevice::create(MeshDeviceConfig{MeshShape{1,1}})` opens a mesh; in bulk-only mode, `MeshDevice::create_unit_mesh(0)` opens a single device. |
| **Tensix cores** | RISC-V-based compute units on each Tenstorrent chip. Each core has five RISC-V processors (two data movement, three compute). The Pi0.5 device launcher places the token loopback kernel on core `(0,0)` and the bulk passthrough kernel on core `(1,0)` (or `(0,0)` in bulk-only mode). |
| **NOC (Network-on-Chip)** | The on-chip interconnect that moves data between Tensix cores, DRAM banks, and the PCIe endpoint. Kernel configurations specify which NOC channel to use (e.g., `NOC::RISCV_0_default`). |
| **Circular buffers** | L1 SRAM data staging areas on each Tensix core. Configured with `CircularBufferConfig`, they provide page-aligned FIFOs for socket data movement. The token path uses 256-byte pages; the bulk path uses 4,096-byte pages. |
| **H2D/D2H sockets** | Host-device communication channels built on top of circular buffers. `H2DSocket` transfers data from the host CPU to a Tensix core; `D2HSocket` transfers data back. Sockets support two modes: `HOST_PUSH` (host writes directly into device L1) and `DEVICE_PULL` (device reads from host memory via NOC). Socket descriptors are exported to `/dev/shm/` as `.bin` files for cross-process connection. |
| **DistributedContext (MPI)** | Multi-host coordination layer. Required for full socket mode where multi-chip topology discovery uses MPI. Not needed in bulk-only or simulation modes. |

## The Three Operating Modes

The framework supports three operating modes, selected by the combination of the JSON config's `"socket"` block and the compile-time `PI05_HAS_SOCKETS` flag.

### Mode 1: Simulation-Only

**Backend:** `PipelineSimulator` inside `DecodeScheduler`
**Hardware required:** None
**Config example:** `pi05_config_sim.json` (no `"socket"` block)

In this mode, the entire pipeline runs on the CPU. The `PipelineSimulator` enforces latency, throughput, and backpressure invariants that mirror real hardware (see [Chapter 1.2](./02_three_phase_pipeline_model.md) for details).

No sockets, no device launcher, no MPI. The `DecodeScheduler` receives a `PipelineSimulatorConfig` variant and creates the simulator internally.

**Use case:** Development iteration, CI pipelines, scheduler algorithm validation, latency modeling without hardware access.

### Mode 2: Full Socket

**Backend:** `SocketPipeline` for the token path + `BulkH2DChannel`/`BulkD2HChannel` for pixel/action data
**Hardware required:** Multi-chip Tenstorrent topology with MPI
**Config example:** `pi05_config.json` (has `"socket"` block with `"mode": "full"` or mode omitted)

Full socket mode exercises the complete data path:

1. **DistributedContext** is initialized via MPI (`DistributedContext::create(argc, argv)`).
2. **DeviceLauncher** fork/execs `pi05_device_launcher`, which creates a `MeshDevice`, allocates four sockets (2 token + 2 bulk) on two Tensix cores, compiles and launches kernels, and exports socket descriptors to `/dev/shm/`.
3. The host connects `BulkH2DChannel` and `BulkD2HChannel` to the bulk sockets for pixel ingestion and action output.
4. `DecodeScheduler` is constructed with a `SocketConfig` variant, creating a `SocketPipeline` backend that connects to the token-path H2D/D2H sockets.

```cpp
// examples/pi05_pipeline_runner.cpp, ~line 869-902
if (!bulk_only) {
    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);
    // ...
}
// DecodeScheduler backend depends on mode
if (bulk_only) {
    // ... PipelineSimulatorConfig ...
} else {
    pl::SocketConfig socket_cfg{
        pipeline_config.socket.h2d_token_socket_id,
        pipeline_config.socket.d2h_token_socket_id,
        pipeline_config.socket.connect_timeout_ms,
    };
    mgr = std::make_unique<pm::DecodeScheduler>(socket_cfg, pm::SchedulerParams{
        .max_users = num_users,
        .skip_eos_writeback = true,
    });
}
```

**Use case:** Integration testing on multi-chip hardware, end-to-end PCIe bandwidth validation, full-stack smoke tests before deploying real model weights.

### Mode 3: Hybrid / Bulk-Only

**Backend:** `PipelineSimulator` for the token path + real `BulkH2DChannel`/`BulkD2HChannel` over PCIe
**Hardware required:** Single Tenstorrent chip (e.g., one N150); multi-chip topology NOT required
**Config example:** `pi05_config_hybrid.json` (`"socket"` block with `"mode": "bulk_only"`)

This mode is designed for machines with disconnected N150 accelerators where the multi-chip `SocketPipeline` is unavailable (topology discovery would crash), but single-chip PCIe still works. The token scheduling path uses `PipelineSimulator` (same CPU-side timing model as simulation mode), while the bulk pixel/action path uses real hardware sockets.

Key implementation details:

- `DeviceLauncher` runs with `--bulk-only`, creating only 2 bulk sockets on a single core `(0,0)`.
- `TT_VISIBLE_DEVICES=0` is set to restrict UMD to device 0, preventing topology discovery across disconnected chips.
- `MeshDevice::create_unit_mesh(0)` is used instead of `MeshDevice::create(MeshDeviceConfig{...})`.
- No MPI / `DistributedContext` initialization.

```cpp
// examples/pi05_device_launcher.cpp, ~line 77-99
bool bulk_only = has_flag(argc, argv, "--bulk-only");
if (bulk_only) {
    const char* visible = std::getenv("TT_VISIBLE_DEVICES");
    if (!visible) {
        setenv("TT_VISIBLE_DEVICES", "0", 1);
    }
}
if (!bulk_only) {
    DistributedContext::create(argc, argv);
    // ...
}
```

**Use case:** Single-chip hardware validation of PCIe bulk transfer paths, measuring real H2D/D2H bandwidth without requiring a full multi-chip mesh, development on machines with disconnected accelerators, IOMMU testing.

## Decision Matrix

| Criterion | Simulation-Only | Full Socket | Hybrid / Bulk-Only |
|-----------|:-:|:-:|:-:|
| Hardware required | None | Multi-chip + MPI | Single chip |
| Token path | `PipelineSimulator` | `SocketPipeline` (real sockets) | `PipelineSimulator` |
| Bulk H2D/D2H path | None | Real PCIe sockets | Real PCIe sockets |
| Device launcher | None | `pi05_device_launcher` (full) | `pi05_device_launcher --bulk-only` |
| MPI / DistributedContext | No | Yes | No |
| CI-friendly | Yes | No (needs hardware) | No (needs hardware) |
| Measures PCIe bandwidth | No | Yes | Yes |
| Exercises scheduler logic | Yes | Yes | Yes |
| Binary | `pi05_pipeline_runner` | `pi05_pipeline_runner_device` | `pi05_pipeline_runner_device` |
| Config distinguisher | No `"socket"` block | `"socket"` block, mode absent or `"full"` | `"socket"` block with `"mode": "bulk_only"` |

## Architecture Diagrams

### Simulation-Only Mode

```
┌─────────────────────────────────────────────────────────────┐
│                       HOST PROCESS                          │
│              (pi05_pipeline_runner)                          │
│                                                             │
│  ┌──────────────────────┐    ┌───────────────────────────┐  │
│  │   Main Loop          │    │   DecodeScheduler         │  │
│  │                      │    │                           │  │
│  │  push_request(ALLOC) │───>│  ┌─────────────────────┐  │  │
│  │  push_request(SUBMIT)│    │  │ PipelineSimulator    │  │  │
│  │  try_pop_output()  <─│────│  │                     │  │  │
│  │                      │    │  │ num_stages=54       │  │  │
│  │  UserSession[] state │    │  │ stage_duration=780us│  │  │
│  │  TTFT/ITL/TPOT stats │    │  │ batch_prefill=true  │  │  │
│  │                      │    │  └─────────────────────┘  │  │
│  └──────────────────────┘    └───────────────────────────┘  │
│                                                             │
│  No sockets.  No device launcher.  No hardware.             │
└─────────────────────────────────────────────────────────────┘
```

### Full Socket Mode

```
┌──────────────────────────────────────────────────────────────┐
│                        HOST PROCESS                          │
│            (pi05_pipeline_runner_device, MPI)                 │
│                                                              │
│  ┌─────────────────┐   ┌──────────────────────────────────┐  │
│  │   Main Loop      │   │     DecodeScheduler              │  │
│  │                  │   │     (SocketPipeline backend)      │  │
│  │  push_request()──│──>│                                  │  │
│  │  try_pop_output()<│──│  Token H2D ──> SocketPipeline    │  │
│  │                  │   │  Token D2H <── SocketPipeline    │  │
│  │  BulkH2DChannel──│───│──────────────────────────────┐   │  │
│  │  BulkD2HChannel<─│───│──────────────────────────┐   │   │  │
│  └─────────────────┘   └──────────────────────────│───│───┘  │
│                                                   │   │      │
│  ┌────────────────────────────────────────────────│───│───┐  │
│  │   DeviceLauncher (fork/exec)                   │   │   │  │
│  │   └─> pi05_device_launcher                     │   │   │  │
│  │       Descriptors: /dev/shm/tt_{h2d,d2h}_*.bin │   │   │  │
│  └────────────────────────────────────────────────│───│───┘  │
└───────────────────────────────────────────────────│───│──────┘
                                                    │   │
                          PCIe                      │   │
                                                    │   │
┌───────────────────────────────────────────────────│───│──────┐
│                   TENSTORRENT DEVICE              │   │      │
│                   (MeshDevice 1x1)                │   │      │
│                                                   v   │      │
│  Core (0,0): Token loopback kernel                    │      │
│    H2D Socket (256B pages) ─┐                         │      │
│    D2H Socket (256B pages) <┘                         │      │
│                                                       v      │
│  Core (1,0): Bulk passthrough kernel                         │
│    H2D Socket (4096B pages, DEVICE_PULL) ─┐                  │
│    D2H Socket (4096B pages)              <┘                  │
│    CB[0]: CircularBuffer for staging                         │
└──────────────────────────────────────────────────────────────┘
```

### Hybrid / Bulk-Only Mode

```
┌──────────────────────────────────────────────────────────────┐
│                        HOST PROCESS                          │
│            (pi05_pipeline_runner_device, no MPI)              │
│                                                              │
│  ┌─────────────────┐   ┌──────────────────────────────────┐  │
│  │   Main Loop      │   │     DecodeScheduler              │  │
│  │                  │   │     (PipelineSimulator backend)   │  │
│  │  push_request()──│──>│                                  │  │
│  │  try_pop_output()<│──│  Token path: CPU simulation      │  │
│  │                  │   │  (no token sockets)               │  │
│  │  BulkH2DChannel──│───│──────────────────────────────┐   │  │
│  │  BulkD2HChannel<─│───│──────────────────────────┐   │   │  │
│  └─────────────────┘   └──────────────────────────│───│───┘  │
│                                                   │   │      │
│  ┌────────────────────────────────────────────────│───│───┐  │
│  │   DeviceLauncher (fork/exec)                   │   │   │  │
│  │   └─> pi05_device_launcher --bulk-only         │   │   │  │
│  │       TT_VISIBLE_DEVICES=0                     │   │   │  │
│  │       Descriptors: /dev/shm/tt_{h2d,d2h}_*.bin │   │   │  │
│  └────────────────────────────────────────────────│───│───┘  │
└───────────────────────────────────────────────────│───│──────┘
                                                    │   │
                          PCIe                      │   │
                                                    │   │
┌───────────────────────────────────────────────────│───│──────┐
│                   TENSTORRENT DEVICE              │   │      │
│              (MeshDevice::create_unit_mesh(0))    │   │      │
│              Single chip, no topology discovery   v   │      │
│                                                       v      │
│  Core (0,0): Bulk passthrough kernel ONLY                    │
│    H2D Socket (4096B pages, DEVICE_PULL) ─┐                  │
│    D2H Socket (4096B pages)              <┘                  │
│    CB[0]: CircularBuffer for staging                         │
│                                                              │
│    (No token loopback kernel -- tokens are simulated)        │
└──────────────────────────────────────────────────────────────┘
```

## Mode Selection Flow in Code

The mode is determined at two levels: the JSON config (runtime) and the `PI05_HAS_SOCKETS` preprocessor guard (compile time). The following pseudocode shows the unified selection logic:

```cpp
// examples/pi05_pipeline_runner.cpp, lines ~847-1203
// Step 1: Config parsing sets pipeline_config.use_sockets based on "socket" block presence
if (pipeline_config.use_sockets) {
    // Step 2: Compile-time guard
    #ifdef PI05_HAS_SOCKETS
        // Step 3: Runtime mode selection
        const bool bulk_only = (pipeline_config.socket.mode == "bulk_only");
        if (bulk_only) {
            // Hybrid mode: PipelineSimulator + real bulk sockets
        } else {
            // Full socket mode: SocketPipeline + real bulk sockets + MPI
        }
    #endif
    // Step 4: If PI05_HAS_SOCKETS not defined but config has sockets:
    // throws: "binary was compiled without PI05_HAS_SOCKETS"
}
// Step 5: No socket block -> simulation-only mode
```

If the binary lacks `PI05_HAS_SOCKETS`, providing a socket config produces a clear compile-mismatch error (see [Section 1.3](./03_build_system_and_component_map.md)).

## Data Flow Summary

In all modes, a single inference turn follows this sequence:

1. **Pixel ingestion (H2D bulk):** 224x224x3x3 = 451,584 bytes of pixel data + up to 4,096 bytes of action history, sent as a `BulkH2DHeader` + payload padded to 4,096-byte page boundaries.
2. **Token scheduling:** A prompt of `input_tokens` synthetic token IDs is submitted to the `DecodeScheduler`, which drives them through the pipeline backend (simulated or socket-based).
3. **Multi-phase pipeline execution:** The pipeline processes 54 effective stages across vision, text, and denoise phases (see [Section 1.2](./02_three_phase_pipeline_model.md) for per-phase timing).
4. **Action output (D2H bulk):** A `BulkD2HHeader` + 3,200 bytes of bfloat16 action tensor is read from the device.
5. **Multi-turn loop:** The action output from turn $n$ becomes the action history input for turn $n+1$, repeating for `max_turns` rounds per user.

---

**Next:** [`02_three_phase_pipeline_model.md`](./02_three_phase_pipeline_model.md)
