# Chapter 5: Flow Control, Memory Configuration, and Performance

Every socket in the Tenstorrent system -- D2D, H2D (HOST_PUSH and DEVICE_PULL), and D2H -- shares a single flow control primitive: a circular FIFO governed by two 64-bit monotonic counters. The previous chapters introduced this model per socket type. This chapter unifies the treatment into a single parameterized model and turns it into a decision framework: given your workload, how should you size the FIFO, where should the buffer live, and how should you bind it to a sub-device?

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_unified_circular_fifo_model.md`](./01_unified_circular_fifo_model.md) | The parameterized FIFO model across all socket types, per-type counter placement, blocking behavior, D2D dual-layer flow control, backpressure propagation, deadlock avoidance decision tree |
| 2 | [`02_memory_placement_and_fifo_sizing.md`](./02_memory_placement_and_fifo_sizing.md) | L1 vs DRAM decision framework, sub-device binding, FIFO sizing methodology with decision trees, PCIe alignment requirements, socket reuse patterns |

## What You Will Learn

After completing this chapter you will be able to:

- Describe the circular FIFO invariants that hold across all four socket modes (D2D, HOST_PUSH, DEVICE_PULL, D2H) and explain why the counters are 64-bit monotonically increasing values rather than modular pointers.
- Look up exactly where `bytes_sent` and `bytes_acked` reside for any mode, who writes them, and how the other party reads them.
- Use the backpressure propagation model to predict where a multi-stage pipeline will stall.
- Apply the deadlock avoidance decision tree to detect and mitigate circular socket dependencies.
- Choose between L1 and DRAM buffer placement using the storage decision framework.
- Size a FIFO given a target page size and pipeline depth, using the sizing formula and rules of thumb.
- Select correct PCIe and NOC alignment for page sizes and FIFO capacities.
- Reuse sockets across iterations and recognize when recreation is unavoidable.

## Prerequisites

- [Chapter 1](../ch01_socket_overview/index.md) -- socket taxonomy, FIFO model preview
- [Chapter 2](../ch02_d2d_mesh_socket/index.md) -- `SocketMemoryConfig` fields, `SocketConfig` assembly
- [Chapter 3](../ch03_tt_fabric/index.md) -- dual-layer flow control (fabric credits + socket FIFO)
- [Chapter 4](../ch04_host_device_sockets/index.md) -- H2D modes, D2H transport, per-mode counter placement

## Where This Chapter Fits

```
Chapter 1  Socket System Overview and Architecture
Chapter 2  D2D MeshSocket Configuration
Chapter 3  TT-Fabric and Multi-Mesh Topology
Chapter 4  H2D and D2H Sockets -- Host-Device Streaming
Chapter 5  Flow Control, Memory Configuration, and Performance   <-- you are here
Chapter 6  Cross-Process Socket Sharing and Distributed Context
Chapter 7  Production Usage -- DeepSeek V3 Multi-Host Pipeline
```

Chapters 2-4 introduced flow control inline with each socket type. This chapter pulls those threads into one place and adds the sizing, placement, and alignment guidance that was deferred earlier. Chapter 7 applies these decisions in the DeepSeek V3 production pipeline, where upstream and downstream D2D socket FIFO and page sizes are constructor parameters tuned for that specific workload.

## The Core Insight

All four socket modes reduce to one question: is there room in the FIFO for the producer to write, or data in the FIFO for the consumer to read? The answer is always computed from the same two counters. What varies is where those counters live, how they cross address-space boundaries, and what transport carries the data bytes. This chapter treats those differences as three parameters of a single model -- FIFO location, data transport, and counter transport -- rather than four separate mechanisms.

**Warning:** Incorrect FIFO sizing or memory placement can cause silent throughput degradation or outright deadlock. A FIFO that is too small starves the pipeline; a FIFO that is too large wastes scarce L1 (only ~1464 KB per core on Blackhole). Circular socket dependencies with symmetric sizing can deadlock. Use the decision frameworks in this chapter before deploying any multi-stage socket pipeline.

---

*Next: [5.1 Unified Circular FIFO Model](01_unified_circular_fifo_model.md)*
