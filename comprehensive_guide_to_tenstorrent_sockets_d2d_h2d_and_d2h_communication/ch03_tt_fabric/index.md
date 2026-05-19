# Chapter 3 -- TT-Fabric and Multi-Mesh Topology

This chapter explains the transport layer that makes D2D sockets possible. When a `send_async` call dispatches a tensor across mesh boundaries, the data travels through TT-Fabric -- Tenstorrent's inter-chip ethernet routing network. The user configures what to send and where; TT-Fabric determines how packets are routed, buffered, and delivered. Understanding this boundary between user responsibility and runtime responsibility is essential for diagnosing performance bottlenecks, reasoning about latency, and scaling to multi-Galaxy deployments.

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_fabric_topology_and_configuration.md`](./01_fabric_topology_and_configuration.md) | TT-Fabric architecture, Blackhole Galaxy physical layer, FABRIC_2D routing, initialization sequence, fabric node IDs, MGD bridging, virtual channels |
| 2 | [`02_socket_to_fabric_mapping.md`](./02_socket_to_fabric_mapping.md) | D2D packet path, connection-to-route mapping, multi-connection parallelism, dual-layer flow control, bandwidth/latency analysis, PCIe topology, failure modes |

## What You Will Learn

After completing this chapter you will be able to:

- Describe TT-Fabric's four-layer architecture and the role each layer plays in D2D packet delivery.
- Identify the physical characteristics of a Blackhole Galaxy system: 32 chips in a 4x8 grid, ~12.5 GB/s per link per direction.
- Explain FABRIC_2D dimension-ordered (X-then-Y) routing and calculate hop counts for arbitrary source-destination pairs.
- Trace the initialization sequence from `set_fabric_config` through `open_mesh_device` to a ready fabric.
- Map a `SocketConnection` to the physical fabric route its packets will traverse.
- Calculate aggregate bisection bandwidth for a given mesh topology.
- Distinguish between fabric-level credit-based flow control and socket-level FIFO flow control, and explain how they interact.

## Prerequisites

- [Chapter 1: Socket System Overview and Architecture](../ch01_socket_overview/index.md) -- socket taxonomy, FIFO model, lifecycle phases
- [Chapter 2: D2D MeshSocket Configuration](../ch02_d2d_mesh_socket/index.md) -- SocketConfig, SocketConnection, MeshCoreCoord, SPMD programming
- Familiarity with Tenstorrent mesh device concepts: `MeshDevice`, `MeshCoordinate`, `MeshShape`
- Basic networking concepts: routing, hops, bandwidth, latency, flow control

## Where This Chapter Fits

```
Chapter 1  Socket System Overview and Architecture
Chapter 2  D2D MeshSocket Configuration
Chapter 3  TT-Fabric and Multi-Mesh Topology                 <-- you are here
Chapter 4  H2D and D2H Sockets -- Host-Device Streaming
Chapter 5  Flow Control, Memory Configuration, and Performance
Chapter 6  Cross-Process Socket Sharing and Distributed Context
Chapter 7  Production Usage -- DeepSeek V3 Multi-Host Pipeline
```

## Key Concepts at a Glance

```
  User Responsibility                    Runtime Responsibility
  (SocketConfig, SocketConnection)       (TT-Fabric, routing, channels)
  +---------------------------------+    +---------------------------------+
  | WHAT: tensor data               |    | HOW: X-then-Y routing           |
  | WHERE: sender/receiver coords   | -> | WHERE (physical): ethernet link |
  | WHEN: send_async / recv_async   |    | BUFFERING: credit-based flow    |
  | FIFO: socket-level flow control |    | CHANNELS: virtual channel mux   |
  +---------------------------------+    +---------------------------------+
```

**Warning:** TT-Fabric must be initialized before any `MeshSocket` is created. The required call order is: `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)` then `ttnn.open_mesh_device(...)`. Reversing this order or omitting `set_fabric_config` results in a fabric initialization error with no recovery path short of closing and reopening all devices.

---

**Next:** [`01_fabric_topology_and_configuration.md`](./01_fabric_topology_and_configuration.md)
