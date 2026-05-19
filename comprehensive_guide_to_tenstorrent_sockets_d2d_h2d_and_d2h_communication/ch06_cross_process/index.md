# Chapter 6: Cross-Process Socket Sharing and Distributed Context

Production LLM serving on Tenstorrent hardware rarely fits inside a single process. A typical deployment splits work across multiple OS processes: one process per pipeline stage, each owning a subset of devices, with H2D and D2H sockets bridging the host-device boundary at ingress and egress points. The cross-process socket sharing protocol -- built on `export_descriptor()`, `connect()`, and the DistributedContext -- is what makes this architecture possible without forcing every process to hold a full MetalContext.

This chapter covers the export/connect protocol mechanics, the asymmetric owner-connector ownership model, and the security and operational concerns that arise when raw physical addresses transit through shared memory.

## Scope

This chapter applies to **H2D and D2H sockets only**. D2D sockets operate entirely within device fabric and do not use `export_descriptor()` or `connect()`. MeshSocket-based D2D communication uses DistributedContext and rank-based addressing, which is covered in Chapter 5.

```
  Which socket types support cross-process sharing?

  +-- D2D (MeshSocket) -----> NO export/connect. Uses DistributedContext + ranks.
  |
  +-- H2D (H2DSocket) ------> YES. export_descriptor() + connect().
  |
  +-- D2H (D2HSocket) ------> YES. export_descriptor() + connect().
```

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_export_connect_protocol.md`](./01_export_connect_protocol.md) | Flatbuffer serialization, `/dev/shm` transport, `PCIeCoreWriter`, handshake ordering |
| 2 | [`02_owner_connector_lifecycle.md`](./02_owner_connector_lifecycle.md) | Role separation, ordering constraints, failure modes, three-barrier shutdown |
| 3 | [`03_security_and_operational_considerations.md`](./03_security_and_operational_considerations.md) | `/dev/shm` attack surface, vIOMMU requirements, tmpfs lifecycle, multi-host limits |

## Prerequisites

Before reading this chapter, ensure familiarity with:

- **Chapter 4.1:** H2D socket modes, HOST_PUSH and DEVICE_PULL transport paths
- **Chapter 4.2:** D2H socket API, ExternalConfigBuffer, SocketSenderInterface
- **Chapter 5:** Unified FIFO model, counter placement per socket mode
- **vIOMMU:** All H2D and D2H modes require vIOMMU; cross-process sharing inherits this requirement

## Key Concepts at a Glance

| Concept | One-sentence definition |
|---------|------------------------|
| Owner | The process that creates the socket, allocates device resources, and controls teardown |
| Connector | A process that attaches to an exported socket; operates on raw PCIe mappings without MetalContext |
| `export_descriptor()` | Serializes socket state to a flatbuffer file in `/dev/shm/<socket_id>` |
| `connect()` | Reconstructs a socket handle from the flatbuffer; requires only UMD, not MetalContext |
| `PCIeCoreWriter` | The low-level writer used by H2D connectors to push data via TLB-mapped PCIe writes |
| DistributedContext | Multi-host coordination layer; auto-initialized on device open; used by MeshSocket, not by export/connect |

## Where This Chapter Fits

```
Chapter 1  Socket System Overview and Architecture
Chapter 2  D2D MeshSocket Configuration
Chapter 3  TT-Fabric and Multi-Mesh Topology
Chapter 4  H2D and D2H Sockets -- Host-Device Streaming
Chapter 5  Flow Control, Memory Configuration, and Performance
Chapter 6  Cross-Process Socket Sharing and Distributed Context   <-- you are here
Chapter 7  Production Usage -- DeepSeek V3 Multi-Host Pipeline
```

Chapter 4 introduced `export_descriptor` and `connect` as API methods. This chapter explains the protocol underneath, defines the contract both sides must honor, and covers the operational reality of running it in production. Chapter 7 applies these patterns in the DeepSeek V3 PipelineBlock architecture, where HostInterface manages H2D/D2H lifecycle and SocketInterface handles D2D connections between pipeline stages.

---

**Previous:** [Chapter 5](../ch05_flow_control_and_memory/index.md) | **Next:** [6.1 Export/Connect Protocol](./01_export_connect_protocol.md) | **Up:** [Guide Index](../index.md)
