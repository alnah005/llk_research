# Chapter 3: Data Organization

## Overview

This chapter covers how data is structured in L1 memory for consumption by the Tensix Engine. Every LLK operation works on tiles -- fixed-size blocks of data that are decomposed into faces. Understanding these structures, the data formats that encode them, and the automatic format conversions that happen at unpack and pack time is a prerequisite for writing or debugging any LLK kernel.

## Learning Objectives

After reading this chapter, you will be able to:

1. Describe what a tile and a face are, their default dimensions, and how they map to L1 memory.
2. Explain the `TensorShape` struct and how it encodes tile geometry for LLK operations.
3. List the supported `DataFormat` values and identify which formats are valid in L1, Source registers, and the Destination register.
4. Distinguish block floating point (BFP) formats from standard floating point formats and explain their shared-exponent structure.
5. Trace the automatic format conversions performed by unpacker and packer gaskets.
6. Define FP32 destination accumulation and explain when it is beneficial.
7. Describe the four `MathFidelity` levels and the accuracy-versus-performance trade-off they represent.

## Subtopics

1. [**Tiles and Faces**](./tiles_and_faces.md) -- The fundamental data unit (32x32 tile), its decomposition into 16x16 faces, the `TensorShape` struct, and tile-order versus row-major storage.

2. [**Data Formats**](./data_formats.md) -- The `DataFormat` enum, block floating point formats, the `InstrModLoadStore` enum for SFPU, and which formats are valid in each hardware register file.

3. [**Format Conversion**](./format_conversion.md) -- Automatic format conversion in unpacker and packer gaskets, FP32 destination accumulation, BFP detection, and stochastic rounding modes.

4. [**Math Fidelity**](./math_fidelity.md) -- The `MathFidelity` enum, how fidelity controls multiplication phases, and the trade-off between accuracy and throughput.

## Key Terms Introduced

| Term | Definition |
|:-----|:-----------|
| Tile | A 32x32 block of datums; the fundamental unit of data for LLK operations. |
| Face | A 16x16 sub-block of a tile; four faces compose a standard tile. |
| `TensorShape` | A packed struct describing tile geometry via face dimensions and face counts. |
| Block Floating Point (BFP) | A family of data formats where a group of mantissas shares a single exponent. |
| Gasket | Hardware-accelerated format conversion logic in the unpacker or packer. |
| Math Fidelity | A setting that controls how many multiplication phases the FPU uses, trading accuracy for speed. |
| Stochastic Rounding | A rounding mode that adds random noise to reduce systematic bias during format conversion. |

---

**Next:** [`tiles_and_faces.md`](./tiles_and_faces.md)
