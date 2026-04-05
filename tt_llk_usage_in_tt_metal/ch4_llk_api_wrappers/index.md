# Chapter 4: LLK API Wrappers

This chapter examines the **wrapper layer** that TT-Metal places between the Compute Kernel API (the interface kernel authors write against) and the raw `_llk_*_()` functions provided by the tt-llk library. This layer lives under `tt_metal/hw/ckernels/{wormhole_b0,blackhole,quasar}/metal/` and serves three purposes:

1. **Operand abstraction** -- Translating high-level operand/output identifiers into the concrete tile dimensions, data formats, and circular-buffer addresses that tt-llk requires.
2. **Circular-buffer I/O** -- Managing producer/consumer synchronization (tile receive/ack counters, write-pointer advancement) across the three TRISC processors.
3. **SFPU operation extensions** -- Providing 90+ additional SFPU math kernels beyond what tt-llk ships, using the same SFPI intrinsic infrastructure.

## Chapter contents

| File | Topic |
|------|-------|
| [`wrapper_layer_role.md`](./wrapper_layer_role.md) | The `llk_api/` wrapper headers: how `llk_<op>()` delegates to `_llk_<op>_()` with operand/tile metadata resolution |
| [`llk_io_layer.md`](./llk_io_layer.md) | The `llk_io/` layer: circular-buffer wait/push/pop, tile pointer management, TRISC synchronization |
| [`sfpu_extensions.md`](./sfpu_extensions.md) | Metal-specific SFPU extensions in `llk_api/llk_sfpu/`: 158 files adding operations like `celu`, `cbrt`, `addcdiv`, and more |

## Key directories

- **Wrapper API headers:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/` (21 files)
- **I/O layer:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/` (5 files)
- **SFPU extensions:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` (158 files)
- **Blackhole variant:** `tt_metal/hw/ckernels/blackhole/metal/llk_api/` (identical structure, plus an `experimental/` directory)

The architecture directories (`wormhole_b0`, `blackhole`, `quasar`) share an almost identical file set. Differences are minor -- Blackhole adds an `experimental/` subdirectory under `llk_api/`. The discussion below uses Wormhole B0 paths unless noted otherwise.

## Relationship to other chapters

- Chapter 3 covered the tt-llk `llk_lib/` headers that define the `_llk_*_()` internal functions these wrappers call.
- Chapter 3 covered the SFPU intrinsic infrastructure (`sfpi::dst_reg`, `sfpi::vFloat`, etc.) that the SFPU extensions in this chapter build upon.
- Chapter 5 will cover the Compute Kernel API that sits above this wrapper layer -- the public functions kernel authors invoke (e.g., `matmul_tiles()`, `add_tiles()`).
