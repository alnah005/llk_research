# Chapter 2: Per-Chip Memory Architecture and the NOC

This chapter documents the hardware foundation upon which any "global SRAM pool" abstraction must be built: the L1 SRAM layout of each Tensix and Ethernet core, the dual-NOC interconnect that moves data within a single chip, the 64-bit address encoding that identifies every byte on-die, and the buffer and circular buffer abstractions that the runtime builds on top of these hardware primitives. Chapter 1 established the explicit CCL programming model we aim to replace; this chapter answers the question "what does the per-chip hardware actually look like?" so that subsequent chapters on inter-chip fabric (Chapter 3), existing building blocks (Chapter 4), and the GSRP design (Chapter 6) can reason about concrete address spaces, reserved regions, alignment constraints, and available capacity.

## Files

| # | File | Description |
|---|------|-------------|
| 1 | [`01_tensix_l1_layout.md`](./01_tensix_l1_layout.md) | Detailed L1 SRAM memory map for Wormhole (1464 KB, 80 cores) and Blackhole (1536 KB, 140 cores). Reserved regions from boot code through packet header pool, firmware bases, `MEM_MAP_END` derivation, `L1_UNRESERVED_BASE`. The `AllocatorConfig` model, buffer types (`L1`, `L1_SMALL`, `DRAM`, `SYSTEM_MEMORY`, `TRACE`), `TensorMemoryLayout`, and `L1BankingAllocator` with bank shuffling. |
| 2 | [`02_ethernet_l1_layout.md`](./02_ethernet_l1_layout.md) | Ethernet core L1 layout: 256 KB on WH (minus 32 B barrier), 512 KB on BH. Active ERISC vs Idle ERISC memory maps. Routing firmware reserved regions, kernel config space, fabric router reserved regions. How `ERISC_L1_UNRESERVED_BASE` shifts with routing firmware. Constraints for address translation tables, including space budget and firmware processing capacity. |
| 3 | [`03_noc_architecture_and_addressing.md`](./03_noc_architecture_and_addressing.md) | Dual-NOC system (NOC0, NOC1), 3-port router architecture (NIU, X, Y) with XY routing, register-space layout, coordinate virtualization (WH virtual Tensix at (18,18); BH at (1,2)), 64-bit address encoding, NOC transaction types, the `experimental::Noc` class, BH security fence registers. Key constraint: the NOC is chip-local. |
| 4 | [`04_buffer_types_and_circular_buffers.md`](./04_buffer_types_and_circular_buffers.md) | Circular buffers with `LocalCBInterface`, `RemoteCircularBuffer` (sender/receiver with pages-sent/pages-acked flow control), `GlobalCircularBuffer` (multi-sender/multi-receiver shared L1 buffer), `GlobalSemaphore`, the dispatch system (`DispatchMemMap`, command queue addresses), data flow paths, and the layered software stack summary. |

## Reading Guide

This chapter is foundational -- it introduces L1 layout and NOC addressing concepts referenced throughout the rest of the guide. Readers new to Tenstorrent hardware should read this chapter sequentially: the memory map (file 01) provides the physical foundation, the Ethernet layout (file 02) introduces the constraints specific to inter-chip communication cores, the NOC (file 03) explains how data moves within a chip, and the buffer types (file 04) show how the runtime manages the available memory. Readers already familiar with the per-chip architecture can proceed directly to Chapter 1 for the problem statement, then return here as needed. Each file builds on the previous, and all files connect hardware details back to GSRP feasibility.

## Key Source Directories

- `tt_metal/hw/inc/internal/tt-1xx/wormhole/` -- Wormhole device memory maps and NOC parameters
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/` -- Blackhole device memory maps and NOC parameters
- `tt_metal/hw/inc/experimental/` -- Experimental NOC API (`noc.h`)
- `tt_metal/hw/inc/api/` -- Device-side API headers (`remote_circular_buffer.h`)
- `tt_metal/impl/allocator/` -- L1 banking allocator and allocator configuration
- `tt_metal/api/tt-metalium/` -- Host-side buffer types, global circular buffer, global semaphore
- `tt_metal/impl/dispatch/` -- Dispatch memory map
