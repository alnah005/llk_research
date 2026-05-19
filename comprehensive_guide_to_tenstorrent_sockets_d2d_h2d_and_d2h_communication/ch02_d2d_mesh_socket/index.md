# Chapter 2 -- D2D MeshSocket Configuration

This chapter provides a complete walkthrough of the D2D MeshSocket configuration data model, its asynchronous send/receive operations, and the two programming modes (intra-process and multi-process SPMD) that determine how socket endpoints are paired. The chapter treats configuration as a layered assembly process -- four structs compose bottom-up into a single `SocketConfig` -- and then shows how that config drives runtime behavior through `send_async`, `recv_async`, and rank-aware branching.

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_configuration_hierarchy.md`](./01_configuration_hierarchy.md) | Configuration building blocks: MeshCoreCoord, SocketConnection, SocketMemoryConfig, SocketConfig. Field-by-field tables, two constructor overloads, assembly walkthrough |
| 2 | [`02_send_recv_async_operations.md`](./02_send_recv_async_operations.md) | `ttnn.experimental.send_async` / `ttnn.experimental.recv_async` semantics, pre-allocation requirement, endpoint type queries, buffer access, synchronization, ordering guarantees |
| 3 | [`03_spmd_and_intra_process.md`](./03_spmd_and_intra_process.md) | SPMD rank-aware programming with DistributedContext, intra-process `create_socket_pair`, FABRIC_2D setup, canonical code patterns |

## What You Will Learn

After completing this chapter you will be able to:

- Describe the four configuration structs (`MeshCoreCoord`, `SocketConnection`, `SocketMemoryConfig`, `SocketConfig`) and explain how each composes into the next.
- Build a complete `SocketConfig` from scratch for a given mesh topology.
- Choose between the two `SocketConfig` constructor overloads: mesh_id-based (intra-process) and rank-based (multi-process SPMD).
- Explain the pre-allocation requirement for `recv_async` and why it exists.
- Use `ttnn.experimental.send_async` and `ttnn.experimental.recv_async` to move tensors between mesh devices with correct synchronization.
- Write SPMD programs that branch on `ttnn.distributed_context_get_rank()` and coordinate through `DistributedContext`.
- Use `create_socket_pair` for intra-process D2D communication without SPMD overhead.

## Prerequisites

- [Chapter 1: Socket System Overview and Architecture](../ch01_socket_overview/index.md) -- socket taxonomy, FIFO model, lifecycle phases
- Familiarity with Tenstorrent mesh device concepts: `MeshDevice`, `MeshCoordinate`, `CoreCoord`, `MeshShape`
- Basic understanding of TTNN tensor operations and the `ttnn.experimental` namespace
- Awareness of multi-process execution models (SPMD)

## Where This Chapter Fits

```
Chapter 1  Socket System Overview and Architecture
Chapter 2  D2D MeshSocket Configuration                     <-- you are here
Chapter 3  TT-Fabric and Multi-Mesh Topology
Chapter 4  H2D and D2H Sockets -- Host-Device Streaming
Chapter 5  Flow Control, Memory Configuration, and Performance
Chapter 6  Cross-Process Socket Sharing and Distributed Context
Chapter 7  Production Usage -- DeepSeek V3 Multi-Host Pipeline
```

## Key Concepts at a Glance

```
MeshCoreCoord          -- WHERE: a specific core on a specific device
    |
    v
SocketConnection       -- WHO: sender core <-> receiver core (1:1 binding)
    |
    v
SocketMemoryConfig     -- HOW: buffer type, FIFO size, sub-device placement
    |
    v
SocketConfig           -- COMPLETE: connections + memory + routing (mesh_ids or ranks)
    |
    v
MeshSocket             -- RUNTIME: instantiated channel ready for send/recv
```

**Warning:** D2D MeshSocket does NOT use `export_descriptor` / `connect` patterns. Those belong to the cross-process model covered in [Chapter 6](../ch06_cross_process/index.md). D2D uses either SPMD ranks with DistributedContext or intra-process mesh IDs with `create_socket_pair`.

---

**Next:** [`01_configuration_hierarchy.md`](./01_configuration_hierarchy.md)
