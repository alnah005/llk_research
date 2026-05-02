# Chapter 6: The Global SRAM Pool Abstraction -- Design Space Exploration

This chapter proposes and analyzes candidate architectures for presenting all L1 SRAM across a multi-chip mesh as a unified addressable memory. It begins with a precise problem statement distinguishing three transparency levels, then designs the global address encoding, proposes a relaxed consistency model, evaluates three candidate architectures (Fat Fabric API, Address-Translation Shim, Compiler-Inserted Communication), and designs the address translation layer that underpins all three. The design choices made here are implemented in Chapter 7 (runtime infrastructure) and evaluated in Chapter 8 (performance analysis).

## Files

| # | File | Description |
|---|------|-------------|
| 1 | [`01_problem_statement_and_transparency_levels.md`](./01_problem_statement_and_transparency_levels.md) | Defines the goal (eliminate application-level cross-chip data movement management), distinguishes three transparency levels (hardware-transparent Level A, runtime-assisted Level B, library-mediated Level C), and calculates the total addressable SRAM: 32-chip BH Galaxy = ~6.7 GB aggregate. |
| 2 | [`02_address_space_design.md`](./02_address_space_design.md) | Global address encoding: `{mesh_id, chip_id, core_xy, local_l1_offset}` packed into 64 bits. Evaluates encoding strategies (NOC address extension, software translation table, segment registers), alignment constraints, and how `MeshBuffer`'s consistent-address property is a stepping stone. |
| 3 | [`03_coherence_and_consistency_models.md`](./03_coherence_and_consistency_models.md) | Argues full cache coherence is unnecessary (no hardware caches on L1). Proposes relaxed/release consistency with explicit fences using existing atomic primitives. Memory ownership models: SWMR, epoch-based transfer, producer-consumer rings. BSP epoch model (from Chapter 5) as a synchronization framework. |
| 4 | [`04_candidate_architectures.md`](./04_candidate_architectures.md) | Three candidate architectures evaluated: (1) Fat Fabric API extending the mesh API with `remote_read`/`remote_write`/`remote_atomic`; (2) Address-Translation Shim on ERISC intercepting out-of-range NOC transactions; (3) Compiler-Inserted Communication via TTNN graph pass. Weighted decision matrix across 7 criteria. |
| 5 | [`05_address_translation_layer.md`](./05_address_translation_layer.md) | Address translation design: host-computed L1 tables, firmware-level ERISC translation, hardware-assisted TLB (Quasar-only). Route caching (8-entry per-core cache reducing translation from 18--27 to ~13 cycles). Overhead analysis: lookup cost per packet vs. amortized cost for bulk transfers. Break-even at ~3 KB (5% overhead). |

## Reading Guide

Read sequentially: file 01 frames the problem, file 02 designs the address space, file 03 establishes consistency guarantees, file 04 evaluates candidate architectures, and file 05 designs the translation mechanism. Readers focused on implementation can prioritize files 04 and 05. The three transparency levels from file 01 are referenced throughout Chapters 7 and 8.

## Key Source Directories

- `tt_metal/fabric/hw/inc/mesh/api.h` -- Existing mesh API (basis for Fat Fabric API)
- `tt_metal/fabric/control_plane.cpp` -- Routing table generation (basis for address translation)
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` -- L1 memory map constants
- `tt_metal/api/tt-metalium/mesh_buffer.hpp` -- MeshBuffer consistent-address allocation
