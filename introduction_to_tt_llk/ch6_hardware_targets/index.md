# Chapter 6: Hardware Target Differences

## Overview

TT-LLK supports three Tenstorrent chip architectures: **Wormhole B0**, **Blackhole**, and **Quasar**. While all three targets expose the same high-level LLK operations (unpack, math, pack), the underlying implementations differ significantly. Wormhole B0 and Blackhole are nearly identical in their LLK layer -- sharing the same file names, API signatures, and common infrastructure headers. Quasar, as a newer architecture, diverges substantially: it uses scoped enums, a reduced SFPU operation set, different file naming conventions, a different boot model (no BRISC core), and unique headers for features like PC buffers and vector operations.

Understanding these differences is essential when writing portable kernels, debugging architecture-specific failures, or extending the LLK layer for a particular target.

## Learning Objectives

After reading this chapter, you will be able to:

1. Identify which `common/inc/` headers are shared between WH/BH and which are unique to Quasar.
2. Explain why Quasar uses `enum class` (scoped enums) while WH/BH use unscoped enums, and the impact on code portability.
3. Describe the difference in `SfpuType` coverage between WH/BH (112 operations) and Quasar (16 operations).
4. Map the WH/BH unpack file naming convention (`llk_unpack_A.h`, `llk_unpack_AB.h`) to the Quasar equivalents (`llk_unpack_unary_operand.h`, `llk_unpack_binary_operands.h`).
5. Explain why Quasar defaults to `BootMode.TRISC` while WH/BH default to `BootMode.BRISC`.
6. Locate the `assembly.yaml` instruction definition files for each architecture.

## Subtopics

1. [**Architecture Comparison**](./architecture_comparison.md) -- A side-by-side comparison of the three chip targets at the `common/inc/` header level: shared headers, WH/BH-specific headers (xmov, mutex guard, debug), and Quasar-specific headers (PC buffer, vector, dest, RISC atomics). Covers the scoped vs. unscoped enum differences and `SfpuType` divergence.

2. [**LLK API Differences**](./llk_api_differences.md) -- How the `llk_lib/` directories differ across targets: identical file sets in WH/BH, Quasar-unique files (TDMA, matmul pack, broadcast variants), files present in WH/BH but absent in Quasar, the unified SFPU common header, boot mode defaults, and per-architecture instruction definitions.
