# Chapter 7: Runtime Infrastructure for Transparent Cross-Chip Data Movement

This chapter specifies the runtime components needed to make packet hopping transparent to application kernels, building on the GSRP design from Chapter 6. It covers the transparent packet injection mechanism (how a Tensix write to a global address becomes a fabric packet), flow control extensions for implicit traffic, deadlock avoidance strategies for bidirectional cross-chip communication, and the dispatch integration and resource management that ties everything into the existing TT-Metal runtime. The quantitative assessment of these mechanisms appears in Chapter 8.

## Files

| # | File | Description |
|---|------|-------------|
| 1 | [`01_transparent_packet_injection.md`](./01_transparent_packet_injection.md) | Designs write-path injection via compiler-inserted shim (~13 cycles overhead) and read-path injection via ERISC firmware trap. Evaluates three approaches (compiler shim, FW trap, HW extension). Write/read asymmetry: writes are posted, reads require round-trip (~2.0--2.3x write latency). Multi-hop scaling model. |
| 2 | [`02_flow_control_and_congestion.md`](./02_flow_control_and_congestion.md) | Extends existing three-level flow control (Chapter 3, Section 6) for GSRP traffic: dynamic credit allocation, per-destination tracking, priority queuing, local aggregation via Tensix MUX, rate limiting, end-to-end flow control, and congestion detection/mitigation. |
| 3 | [`03_deadlock_avoidance_strategies.md`](./03_deadlock_avoidance_strategies.md) | Analyzes new deadlock risks from transparent bidirectional traffic. Solutions: VC partitioning by traffic class (request vs. reply), epoch-based barrier synchronization, request-reply channel separation. Formal safety argument for the layered defense strategy. |
| 4 | [`04_dispatch_integration_and_resource_management.md`](./04_dispatch_integration_and_resource_management.md) | Integration with dispatch system (`MeshCommandQueue`, hardware command queue). L1 resource accounting: existing fabric 6,864 B + GSRP Phase 2 5,288 B = ~12.2 KB per core (0.77--0.83% of L1). Galaxy-scale system overhead: ~0.47% of total L1. Telemetry, debugging, and CCL coexistence. |

## Reading Guide

Read sequentially: file 01 establishes how data gets injected into the fabric, file 02 ensures the fabric doesn't become congested, file 03 prevents deadlocks, and file 04 integrates everything into the runtime. Files 01 and 04 contain the key resource budget numbers referenced by Chapter 8.

## Key Source Directories

- `tt_metal/fabric/impl/kernels/edm_fabric/` -- EDM router kernel (injection point)
- `tt_metal/fabric/hw/inc/edm_fabric/` -- Flow control, channel architecture
- `tt_metal/fabric/control_plane.cpp` -- Routing table generation
- `tt_metal/impl/dispatch/` -- Dispatch memory map, command queue
- `tt_metal/api/tt-metalium/mesh_device.hpp` -- MeshDevice integration
