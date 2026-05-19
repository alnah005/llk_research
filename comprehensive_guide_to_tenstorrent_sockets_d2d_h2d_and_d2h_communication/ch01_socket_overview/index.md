# Chapter 1 -- Socket System Overview and Architecture

This chapter introduces Tenstorrent's three socket types as a unified communication system for moving data between hosts and devices in multi-mesh deployments. It establishes a taxonomy of data-flow directions and transport mechanisms, provides a selection guide with real-world pipeline context, and previews the common abstractions that every socket type shares.

## Contents

| # | File | Topic |
|---|------|-------|
| 1 | [`01_socket_taxonomy_and_purpose.md`](./01_socket_taxonomy_and_purpose.md) | Three socket types taxonomy, comparison table, full-stack diagram, sockets vs CCL vs direct memory access, decision tree, DeepSeek V3 pipeline composition |
| 2 | [`02_common_abstractions_preview.md`](./02_common_abstractions_preview.md) | FIFO model, flow control preview, non-blocking semantics, SPMD preview, five-phase lifecycle, cross-process sharing preview |

## What You Will Learn

After completing this chapter you will be able to:

- Name the three socket types (H2D, D2D, D2H) and explain when each is used.
- Place sockets in the Tenstorrent software stack relative to TT-Fabric, TTNN, and CCL.
- Choose the correct socket type for a given data-movement scenario using the decision tree.
- Describe the FIFO-based flow-control model common to all socket types.
- Outline the lifecycle stages every socket connection passes through.
- Explain how cross-process sharing differs between H2D/D2H (descriptor exchange) and D2D (SPMD ranks).

## Prerequisites

- Familiarity with Tenstorrent hardware concepts: mesh devices, Tensix cores, L1 SRAM, DRAM, NOC, PCIe
- Basic understanding of TT-Metal runtime and the TTNN Python API
- Awareness of multi-device and multi-process programming patterns

## Where This Chapter Fits

```
Chapter 1  Socket System Overview and Architecture         <-- you are here
Chapter 2  D2D MeshSocket Configuration
Chapter 3  TT-Fabric and Multi-Mesh Topology
Chapter 4  H2D and D2H Sockets — Host-Device Streaming
Chapter 5  Flow Control, Memory Configuration, and Performance
Chapter 6  Cross-Process Socket Sharing and Distributed Context
Chapter 7  Production Usage — DeepSeek V3 Multi-Host Pipeline
```

---

**Next:** [`01_socket_taxonomy_and_purpose.md`](./01_socket_taxonomy_and_purpose.md)
