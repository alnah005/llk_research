# Chapter 8: Portability and Improvement Opportunities

The previous seven chapters examined the LLK-based kernel authoring experience from multiple angles: the TRISC macro system (Chapter 1), the Compute Kernel API abstraction (Chapter 2), define-driven configuration (Chapter 3), the tile processing pattern (Chapter 4), template-heavy API design (Chapter 5), debugging tools (Chapter 6), and the SFPU extension path (Chapter 7). A recurring theme across all of them is **architecture-specific divergence** -- the degree to which kernel code must change when targeting different Tenstorrent hardware generations.

This final chapter consolidates that theme into a direct examination of portability barriers, then synthesizes the pain points identified throughout the guide into a prioritized list of improvement opportunities.

## Why Portability Matters Now

The Tenstorrent hardware family is expanding. Wormhole B0 and Blackhole share a broadly compatible Tensix architecture, but Quasar introduces fundamental changes: a 4th TRISC core (TRISC3 for dedicated SFPU work), 8 DM cores, a TLS-style build model, different template parameter signatures across the LLK API, and 4 MB of L1 memory. Kernel code that was portable between Wormhole and Blackhole frequently cannot compile -- let alone run -- on Quasar without significant modification.

The Compute Kernel API layer (`tt_metal/hw/inc/api/compute/*.h`) was designed to isolate kernel authors from these differences. In practice, this isolation is incomplete: **45 `#ifndef ARCH_QUASAR` guards** appear across 9 files in the compute API directory alone, and **31 TODO comments** reading `"TODO: AM; add Quasar implementation"` mark functions that simply do nothing on Quasar. The result is that kernel portability requires either manual `#ifdef` guards or maintaining separate per-architecture source files.

## Chapter Contents

- [**Architecture Divergence**](./architecture_divergence.md) -- A quantitative analysis of per-architecture `#ifdef` fragmentation across the Compute Kernel API, with `matmul.h` (11 guards) as the primary case study. Covers Quasar's 4th TRISC, CB API stubs that assert-fail at runtime, and the practical impact on kernel authors.
- [**Improvement Opportunities**](./improvement_opportunities.md) -- Nine concrete, evidence-backed proposals for reducing friction in the kernel authoring experience: RAII-based DST management, CB synchronization validation, define schema contracts, automated data format reconfiguration, unified SFPU registration, an architecture abstraction layer, improved error diagnostics, a lightweight SFPU testing harness, and reducing template depth.
