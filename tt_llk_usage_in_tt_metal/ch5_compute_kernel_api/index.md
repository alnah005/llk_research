# Chapter 5: Compute Kernel API

This chapter documents the user-facing Compute Kernel API -- the set of C++ functions that device-side compute kernels call to perform tile-level operations on Tenstorrent hardware. These API functions serve as the primary interface between kernel authors and the underlying Low-Level Kernel (LLK) library.

Every function described here lives under:

```
tt_metal/hw/inc/api/compute/
```

and compiles into one of the three TRISC threads (TRISC0 = unpack, TRISC1 = math, TRISC2 = pack). A single source file is compiled three times -- once per TRISC -- with the preprocessor defines `TRISC_UNPACK`, `TRISC_MATH`, and `TRISC_PACK` selecting which LLK calls survive in each binary.

## What This Chapter Covers

1. **[API Headers](./api_headers.md)** -- How `compute_kernel_api.h` and the per-operation headers are organized, what the `MATH()` / `UNPACK()` / `PACK()` dispatch macros do, and how the `ALWI` inlining attribute works.

2. **[Operation-to-LLK Mapping](./operation_to_llk_mapping.md)** -- A detailed, per-function mapping of every major Compute API function to the specific LLK calls it invokes on each TRISC thread. Covers matmul, eltwise binary, reduce, tile copy, broadcast, and packing.

3. **[Init and Tile Pattern](./init_and_tile_pattern.md)** -- The canonical control-flow pattern that every compute kernel follows: hardware startup, operation init, the acquire/compute/commit/wait/pack/release loop, and the role of `state_configure()` in avoiding redundant reconfigurations.

## Prerequisites

This chapter assumes familiarity with:

- The three-TRISC execution model (see Chapter 1).
- LLK API layers: `_llk_*` primitives and `llk_*` wrappers (see Chapters 3 and 4).
- Circular buffers (CBs) as the tile transport mechanism between data-movement and compute kernels.

## Key Source Files

| File | Purpose |
|------|---------|
| [`tt_metal/hw/inc/api/compute/compute_kernel_api.h`](../../tt-metal/tt_metal/hw/inc/api/compute/compute_kernel_api.h) | Master include; pulls in all LLK API headers gated by TRISC defines; defines `MATH()`, `UNPACK()`, `PACK()`, and `ALWI` |
| [`tt_metal/hw/inc/api/compute/matmul.h`](../../tt-metal/tt_metal/hw/inc/api/compute/matmul.h) | Matrix multiply wrappers |
| [`tt_metal/hw/inc/api/compute/eltwise_binary.h`](../../tt-metal/tt_metal/hw/inc/api/compute/eltwise_binary.h) | Element-wise add/sub/mul wrappers |
| [`tt_metal/hw/inc/api/compute/reduce.h`](../../tt-metal/tt_metal/hw/inc/api/compute/reduce.h) | Reduce (row, col, scalar) wrappers |
| [`tt_metal/hw/inc/api/compute/tile_move_copy.h`](../../tt-metal/tt_metal/hw/inc/api/compute/tile_move_copy.h) | Tile copy / datacopy wrappers |
| [`tt_metal/hw/inc/api/compute/bcast.h`](../../tt-metal/tt_metal/hw/inc/api/compute/bcast.h) | Broadcast binary operation wrappers |
| [`tt_metal/hw/inc/api/compute/pack.h`](../../tt-metal/tt_metal/hw/inc/api/compute/pack.h) | Pack tile from DST to CB |
| [`tt_metal/hw/inc/api/compute/cb_api.h`](../../tt-metal/tt_metal/hw/inc/api/compute/cb_api.h) | Circular buffer wait/pop/reserve/push |
| [`tt_metal/hw/inc/api/compute/reg_api.h`](../../tt-metal/tt_metal/hw/inc/api/compute/reg_api.h) | DST register acquire/release/commit/wait |
| [`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`](../../tt-metal/tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h) | One-time hardware configuration at kernel start |
| [`tt_metal/hw/inc/api/compute/common_globals.h`](../../tt-metal/tt_metal/hw/inc/api/compute/common_globals.h) | Shared `ALWI`, `MATH()`, `UNPACK()`, `PACK()` macro definitions |
| [`tt_metal/hw/inc/api/compute/sentinel/compute_kernel_sentinel.h`](../../tt-metal/tt_metal/hw/inc/api/compute/sentinel/compute_kernel_sentinel.h) | `state_configure()` -- CB state tracking to avoid redundant reconfiguration |
