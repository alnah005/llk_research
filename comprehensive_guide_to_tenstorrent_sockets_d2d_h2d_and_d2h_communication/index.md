# Comprehensive Guide to Tenstorrent Sockets: D2D, H2D, and D2H Communication

A systems engineering reference for building multi-mesh, multi-host inference pipelines on Tenstorrent Blackhole hardware using the socket communication API.

---

## Chapters

| # | Chapter | Directory | Topic |
|---|---------|-----------|-------|
| 1 | [Socket System Overview](ch01_socket_overview/index.md) | `ch01_socket_overview/` | Socket taxonomy (D2D, H2D, D2H), purpose, and common abstractions preview |
| 2 | [D2D MeshSocket Configuration](ch02_d2d_mesh_socket/index.md) | `ch02_d2d_mesh_socket/` | SocketConfig hierarchy, send/recv async operations, SPMD model, intra-process sockets |
| 3 | [TT-Fabric and Multi-Mesh Topology](ch03_tt_fabric/index.md) | `ch03_tt_fabric/` | Fabric topology, FABRIC_2D routing, initialization, MGD bridging, socket-to-fabric mapping |
| 4 | [H2D and D2H Sockets](ch04_host_device_sockets/index.md) | `ch04_host_device_sockets/` | HOST_PUSH and DEVICE_PULL modes, D2H transport, ExternalConfigBuffer, device kernel APIs |
| 5 | [Flow Control, Memory, and Performance](ch05_flow_control_and_memory/index.md) | `ch05_flow_control_and_memory/` | Unified FIFO model, counter placement map, blocking, backpressure, FIFO sizing, alignment |
| 6 | [Cross-Process Socket Sharing](ch06_cross_process/index.md) | `ch06_cross_process/` | export_descriptor/connect protocol, owner/connector lifecycle, security, multi-host |
| 7 | [Production Usage -- DeepSeek V3](ch07_production_usage/index.md) | `ch07_production_usage/` | PipelineBlock architecture, SocketInterface/HostInterface, end-to-end walkthrough, troubleshooting |

## Reading Paths

**New to Tenstorrent sockets:** Read Chapters 1-5 in order. Chapter 1 introduces the taxonomy, Chapters 2-4 cover each socket type in depth, and Chapter 5 unifies the flow control model.

**Building a multi-process serving pipeline:** Start with Chapter 6 (cross-process sharing), then Chapter 7 (DeepSeek V3 production patterns). Refer back to Chapters 2-4 for API details.

**Debugging a socket stall or performance issue:** Jump to Chapter 7, Section 4 (troubleshooting). Use Chapter 5's counter placement map and blocking behavior tables to diagnose.

**Understanding DeepSeek V3's pipeline:** Read Chapter 7 directly. It cross-references earlier chapters for foundational concepts.

## Hardware Context

This guide targets the **Blackhole Galaxy** system:

| Parameter | Value |
|-----------|-------|
| Chips per Galaxy | 32 (4x8 physical grid, typically 2 meshes of 4x4) |
| Ethernet bandwidth | ~12.5 GB/s per link per direction |
| PCIe (high-BW, ASIC 6) | 4 ports, Gen4 x8, ~16 GB/s each |
| PCIe (low-BW) | 28 ports, Gen4 x1, ~2 GB/s each |
| AI clock | 1.35 GHz |
| L1 SRAM per core | 1464 KB |
| NOC word width | 512 bits (64 bytes, minimum page size) |
| SRAM write alignment | 16 bytes |

## Key Concepts

All three socket types share a single flow control primitive: a circular FIFO governed by two 64-bit monotonically increasing counters (`bytes_sent` and `bytes_acked`). What varies across types is where the FIFO data lives, how bytes move, and where each counter is placed. Chapter 5 provides the complete counter placement map.

**vIOMMU is required** for all H2D and D2H socket modes, including HOST_PUSH. Verify with `dmesg | grep -i iommu` before running socket workloads.

## Prerequisites

- Familiarity with Tenstorrent Tensix cores, L1 SRAM, NOC, and PCIe
- Basic understanding of TT-Metal runtime and the TTNN Python API
- For Chapters 6-7: familiarity with multi-process SPMD deployment patterns

---

*This guide was generated using the deep-planning research methodology with parallel generation, evaluation, synthesis, and critic review.*
