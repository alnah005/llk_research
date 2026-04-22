# Chapter 5: Per-Architecture Code Duplication

## Summary

TT-LLK maintains separate, nearly-complete copies of its kernel library for each hardware
architecture -- wormhole_b0, blackhole, and quasar. TT-Metal then mirrors this per-architecture
split in its own `hw/ckernels/` tree and again in its HAL layer. The result is a multiplication
of code that inflates maintenance costs and makes cross-architecture bug fixes error-prone.

## Key Statistics

| Metric | wormhole_b0 | blackhole | quasar |
|--------|-------------|-----------|--------|
| `llk_lib/` lines (excl. experimental) | 7,646 | 6,615 | 4,210 |
| `common/inc/` top-level lines | 6,239 | 6,412 | 5,482 |
| `common/inc/sfpu/` lines | 9,774 | 10,092 | 658 |
| `llk_lib/` file count (excl. experimental) | 28 | 28 | 23 |
| `common/inc/` file count | 70 | 70 | 34 |
| Metal `llk_api/` wrapper lines | 12,310 | 12,284 | 593 |
| HAL per-arch lines | 921 | 1,012 | 1,161 |
| `instructions/assembly.yaml` lines | 3,006 | 3,571 | 7,050 |

The truly shared code in the repository-level [`common/`](https://github.com/tenstorrent/tt-llk/tree/main/common) directory consists of exactly **2 files** (`llk_assert.h` and `tensor_shape.h`).

## Key Findings

1. **Wormhole and blackhole are near-clones.** Their `llk_lib/` directories have identical file
   lists (28 files each, excluding experimental). Their `common/inc/` directories also share the
   same 70-file layout. See [`duplication_analysis.md`](./duplication_analysis.md) for details.

2. **Quasar diverges structurally.** It uses different file naming conventions (e.g.,
   `llk_unpack_unary_operand.h` instead of `llk_unpack_A.h`), has 10 files with no wormhole/blackhole
   counterpart, and is missing 15 files present in those architectures. This makes mechanical
   porting of fixes infeasible.

3. **Metal multiplies the duplication.** Each Metal architecture directory under
   [`hw/ckernels/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels)
   contains its own `llk_api/` wrappers (12,310 lines for wormhole_b0, 12,284 for blackhole).
   The HAL layer adds another ~3,094 lines split across per-arch subdirectories.

4. **Bug fixes must be applied N times.** Any fix to LLK code outside the 2-file `common/`
   directory must be manually replicated across architectures in LLK, then again in Metal's
   wrapper and HAL layers. See [`maintenance_burden.md`](./maintenance_burden.md) for analysis.

## Files in This Chapter

- [`duplication_analysis.md`](./duplication_analysis.md) -- Code volume breakdown, file-level divergence, and Metal's mirroring
- [`maintenance_burden.md`](./maintenance_burden.md) -- Impact on bug fixes, new operations, and porting
