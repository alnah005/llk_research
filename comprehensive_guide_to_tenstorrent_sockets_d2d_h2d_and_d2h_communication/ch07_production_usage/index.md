# Chapter 7: Production Usage -- DeepSeek V3 Multi-Host Pipeline

Every socket type, configuration pattern, and lifecycle rule from Chapters 1-6 converges in a single production system: the DeepSeek V3 multi-host inference pipeline. This chapter traces how `PipelineBlock` composes D2D MeshSockets, H2D sockets, and D2H sockets into a distributed pipeline spanning multiple Galaxy systems -- each Galaxy containing two 4x4 meshes of Blackhole chips -- and delivers autoregressive token generation across hosts connected by TT-Fabric.

The treatment follows **failure-mode pedagogy**: for every production pattern, we show what breaks when the pattern is violated, explain why, then present the correct approach. Decision tables and comparison matrices provide structured references for configuration choices and debugging.

## Contents

| # | File | Focus |
|---|------|-------|
| 1 | [Pipeline Architecture and Stages](./01_pipeline_architecture_and_stages.md) | Galaxy topology, five-stage decomposition, PipelineBlock, StageMetadata, PipelineConfigEntry, autoregressive loop |
| 2 | [Socket Interface and Host Interface](./02_socket_interface_and_host_interface.md) | SocketInterface for D2D, HostInterface for H2D/D2H, HostIoPlacement, LoopbackConfig, dispatch order |
| 3 | [End-to-End Inference Walkthrough](./03_end_to_end_inference_walkthrough.md) | Token trace from initialization through prefill, decode, and result extraction across meshes and hosts |
| 4 | [Troubleshooting and Debugging](./04_troubleshooting_and_debugging.md) | Hang/deadlock patterns, data corruption, connection errors, performance diagnosis, teardown, CCL comparison |

## Prerequisites

- **Chapters 1-4:** MeshSocket, H2DSocket, D2HSocket fundamentals
- **Chapter 5:** FIFO sizing, flow control counters, backpressure model
- **Chapter 6:** Cross-process export/connect protocol, DistributedContext

## Key Types at a Glance

| Type | Role | Defined in |
|------|------|------------|
| `PipelineBlock` | Top-level orchestrator for one pipeline stage | `pipeline_block/op.py` |
| `StageMetadata` | Per-stage rank and mesh_id assignment | `pipeline_block/op.py` |
| `PipelineConfigEntry` | Maps a stage to entry/exit chip coordinates | `pipeline_block/op.py` |
| `HostIoPlacement` | Core assignments: h2d_core, d2h_core, fwd_d2d_core, lb_d2d_core | `pipeline_block/op.py` |
| `SocketInterface` | Wraps D2D MeshSocket with MeshWrapper for pipeline use | `pipeline_block/op.py` |
| `HostInterface` | Wraps H2D + D2H socket lifecycle for host I/O | `pipeline_block/op.py` |
| `LoopbackConfig` | Enum: fabric_loopback, host_loopback, no_loopback | `pipeline_block/op.py` |

## Hardware Context

A Galaxy system comprises two 4x4 meshes of Blackhole chips (32 chips total). Each mesh has 4 high-bandwidth PCIe links (Gen4 x8) and 28 low-bandwidth links (Gen4 x1). DeepSeek V3's Mixture-of-Experts architecture is partitioned into five pipeline stages that span these meshes, with each stage mapped to specific chip coordinates via `PipelineConfigEntry`.

## Where This Chapter Fits

```
Chapter 1  Socket System Overview and Architecture
Chapter 2  D2D MeshSocket Configuration
Chapter 3  TT-Fabric and Multi-Mesh Topology
Chapter 4  H2D and D2H Sockets -- Host-Device Streaming
Chapter 5  Flow Control, Memory Configuration, and Performance
Chapter 6  Cross-Process Socket Sharing and Distributed Context
Chapter 7  Production Usage -- DeepSeek V3 Multi-Host Pipeline   <-- you are here
```

This is the capstone chapter. It introduces no new socket APIs; instead, it demonstrates how every concept from the preceding chapters combines in a real production pipeline.

## Quick Reference: Where to Find Each Decision Table

| Decision | Section | Table Name |
|----------|---------|------------|
| Which stage type for this mesh? | 7.1.2 | Stage Type Decision Table |
| Which sockets does this stage need? | 7.1.3 | Socket Composition Matrix |
| Which loopback mode? | 7.2.4 | LoopbackConfig Decision Table |
| SocketInterface vs HostInterface? | 7.2.3 | Interface Comparison Matrix |
| Prefill vs decode behavior? | 7.3.5 | Prefill vs Decode Comparison |
| What is causing this hang? | 7.4.2 | Hang Diagnosis Decision Tree |
| What is causing this corruption? | 7.4.3 | Corruption Source Comparison |

---

**Previous:** [Chapter 6](../ch06_cross_process/index.md) | **Next:** [7.1 Pipeline Architecture and Stages](./01_pipeline_architecture_and_stages.md) | **Up:** [Guide Index](../index.md)
