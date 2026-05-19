# Chapter 4 -- H2D and D2H Sockets: Host-Device Streaming

H2D and D2H sockets form a matched pair that streams data between the host CPU and Tenstorrent device cores over PCIe. Unlike D2D MeshSockets (Chapter 2), which traverse TT-Fabric ethernet links between meshes, host-device sockets operate entirely within a single host-device PCIe connection. H2D moves data from host memory into device L1 or DRAM; D2H moves results back from device cores into host-pinned memory. Both directions share the same page-oriented FIFO abstraction and `bytes_sent`/`bytes_acked` flow control, but their transport mechanisms, counter placement, and API shapes differ in ways this chapter explains precisely.

## H2D vs D2H at a Glance

| Property | H2D | D2H |
|----------|-----|-----|
| Direction | Host CPU -> Device core | Device core -> Host CPU |
| Modes | HOST_PUSH, DEVICE_PULL | Single transport (device NOC write) |
| Data transport | TLB write (push) or NOC read (pull) | NOC write through PCIe |
| Host API object | `H2DSocket` | `D2HSocket` |
| Device kernel interface | `SocketReceiverInterface` | `SocketSenderInterface` |
| Host writes data? | Yes (push) or stages in pinned RAM (pull) | No -- reads from pinned buffer |
| Cross-process | `export_descriptor` / `connect` | `export_descriptor` / `connect` |
| ExternalConfigBuffer | Not applicable | Optional (pre-allocated L1) |

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_h2d_socket_modes_and_api.md`](./01_h2d_socket_modes_and_api.md) | HOST_PUSH vs DEVICE_PULL data paths, mode selection guide, H2DSocket API, SocketReceiverInterface, lifecycle |
| 2 | [`02_d2h_socket_api_and_external_config_buffer.md`](./02_d2h_socket_api_and_external_config_buffer.md) | D2H transport mechanism, D2HSocket API, ExternalConfigBuffer, SocketSenderInterface, lifecycle |
| 3 | [`03_host_device_flow_control.md`](./03_host_device_flow_control.md) | Shared FIFO model, counter placement map, H2D and D2H flow control paths, batching, error recovery, PCIe bandwidth |

## What You Will Learn

After completing this chapter you will be able to:

1. Describe the HOST_PUSH and DEVICE_PULL data paths and explain when to choose each.
2. Trace the byte-level journey of data through an H2D socket from host memory to device L1.
3. Trace the byte-level journey of data through a D2H socket from device L1 to host-pinned memory.
4. Identify exactly where each flow control counter lives, who writes it, and who reads it.
5. Use `ExternalConfigBuffer` to pre-allocate D2H socket metadata in device L1.
6. Apply `notify_sender` batching and `discard_pending_pages` for production throughput and error recovery.
7. Explain why vIOMMU is required for all H2D and D2H socket modes.

## Prerequisites

- Familiarity with Tenstorrent hardware: Tensix cores, L1 SRAM, NOC, PCIe TLBs, pinned host memory
- Chapters 1-3 of this guide (socket taxonomy, D2D MeshSocket, TT-Fabric basics)
- Basic understanding of TT-Metal runtime and the TTNN Python API

## Where This Chapter Fits

```
Chapter 1  Socket System Overview and Architecture
Chapter 2  D2D MeshSocket Configuration
Chapter 3  TT-Fabric and Multi-Mesh Topology
Chapter 4  H2D and D2H Sockets -- Host-Device Streaming   <-- you are here
Chapter 5  Flow Control, Memory Configuration, and Performance
Chapter 6  Cross-Process Socket Sharing and Distributed Context
Chapter 7  Production Usage -- DeepSeek V3 Multi-Host Pipeline
```

**Warning:** All H2D and D2H socket modes require vIOMMU to be enabled on the host system. Any mode that involves a device-initiated PCIe transaction (NOC read in DEVICE_PULL, NOC write in D2H, acknowledgement write in HOST_PUSH) requires IOMMU address translation. Verify with `dmesg | grep -i iommu` before running socket workloads.

---

**Previous:** [`../ch03_tt_fabric/index.md`](../ch03_tt_fabric/index.md) | **Next:** [`01_h2d_socket_modes_and_api.md`](./01_h2d_socket_modes_and_api.md) | **Up:** [`../plan.md`](../plan.md)
