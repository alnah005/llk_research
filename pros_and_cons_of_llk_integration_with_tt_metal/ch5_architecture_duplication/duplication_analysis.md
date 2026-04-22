# Duplication Analysis

## LLK Code Volume by Architecture

Each architecture directory in TT-LLK
([`tt_llk_wormhole_b0/`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_wormhole_b0),
[`tt_llk_blackhole/`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_blackhole),
[`tt_llk_quasar/`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_quasar))
contains three subdirectories: `llk_lib/`, `common/`, and `instructions/`. Each is a
self-contained copy of the kernel library for that hardware target.

### `llk_lib/` -- Core LLK Implementations

| Architecture | Files (excl. experimental) | Lines (excl. experimental) | Experimental files | Experimental lines |
|---|---|---|---|---|
| wormhole_b0 | 28 | 7,646 | 4 | 877 |
| blackhole | 28 | 6,615 | 13 | 2,285 |
| quasar | 23 | 4,210 | 0 | 0 |

Source: `tt_llk_wormhole_b0/llk_lib/`, `tt_llk_blackhole/llk_lib/`, `tt_llk_quasar/llk_lib/`
(counted with `find ... -type f -not -path '*/experimental/*' | xargs wc -l`).

### `common/inc/` -- Shared Ckernel Infrastructure

Each architecture's `common/inc/` directory holds ckernel headers and an `sfpu/` subdirectory
for SFPU (Special Function Processing Unit) implementations.

| Architecture | Top-level files | Top-level lines | sfpu files | sfpu lines | Total lines |
|---|---|---|---|---|---|
| wormhole_b0 | 18 | 6,239 | 52 | 9,774 | 16,013 |
| blackhole | 18 | 6,412 | 52 | 10,092 | 16,504 |
| quasar | 19 | 5,482 | 14 | 658 | 6,268* |

*Quasar also has an `internal/` subdirectory with 1 file (128 lines):
[`circular_buffer_interface.h`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_quasar/common/inc/internal/circular_buffer_interface.h).

### `instructions/assembly.yaml` -- Instruction Definitions

| Architecture | Lines |
|---|---|
| wormhole_b0 | 3,006 |
| blackhole | 3,571 |
| quasar | 7,050 |

Quasar's instruction file is roughly twice the size of the others, reflecting genuine
hardware-level differences in its instruction set.

### Grand Totals per Architecture

| Architecture | Total lines (all subdirs) |
|---|---|
| wormhole_b0 | 27,542 |
| blackhole | 28,975 |
| quasar | 17,528 |
| **Combined** | **74,045** |

The cross-architecture [`common/`](https://github.com/tenstorrent/tt-llk/tree/main/common)
directory at the repository root contains exactly 2 files (`llk_assert.h`, `tensor_shape.h`),
meaning virtually all LLK code is architecture-specific.

## File-Level Divergence

### Wormhole vs. Blackhole: Near-Identical File Lists

The non-experimental `llk_lib/` file listings for wormhole_b0 and blackhole are **identical** --
the same 28 files with the same names:

```
llk_defs.h                          llk_pack.h
llk_math_common.h                   llk_pack_common.h
llk_math_eltwise_binary.h           llk_pack_rows.h
llk_math_eltwise_binary_sfpu.h      llk_pack_untilize.h
llk_math_eltwise_binary_sfpu_params.h  llk_unpack_A.h
llk_math_eltwise_ternary_sfpu.h     llk_unpack_AB.h
llk_math_eltwise_ternary_sfpu_params.h  llk_unpack_AB_matmul.h
llk_math_eltwise_unary_datacopy.h   llk_unpack_AB_reduce.h
llk_math_eltwise_unary_sfpu.h       llk_unpack_common.h
llk_math_eltwise_unary_sfpu_params.h   llk_unpack_reduce.h
llk_math_matmul.h                   llk_unpack_tilize.h
llk_math_reduce.h                   llk_unpack_untilize.h
llk_math_transpose_dest.h           llk_memory_checks.h
llk_math_welfords_sfpu.h
llk_math_welfords_sfpu_params.h
```

Their `common/inc/` directories are also identical in structure: 18 top-level headers
and 52 sfpu headers with matching names.

### Quasar: Structural Divergence

Quasar's `llk_lib/` has a substantially different file set compared to wormhole_b0/blackhole.

**Files present only in quasar** (10 files):
- `llk_math_eltwise_binary_broadcast.h`
- `llk_math_eltwise_unary_sfpu_common.h`
- `llk_math_unary_broadcast.h`
- `llk_pack_matmul.h`
- `llk_srcs_tdma.h`
- `llk_unpack_binary_broadcast_operands.h`
- `llk_unpack_binary_operands.h`
- `llk_unpack_matmul.h`
- `llk_unpack_unary_broadcast_operands.h`
- `llk_unpack_unary_operand.h`

**Files present in wormhole_b0/blackhole but absent in quasar** (15 files):
- `llk_math_eltwise_binary_sfpu.h`, `llk_math_eltwise_binary_sfpu_params.h`
- `llk_math_eltwise_ternary_sfpu.h`, `llk_math_eltwise_ternary_sfpu_params.h`
- `llk_math_eltwise_unary_sfpu.h`, `llk_math_eltwise_unary_sfpu_params.h`
- `llk_math_transpose_dest.h`
- `llk_math_welfords_sfpu.h`, `llk_math_welfords_sfpu_params.h`
- `llk_pack_rows.h`
- `llk_unpack_A.h`, `llk_unpack_AB.h`, `llk_unpack_AB_matmul.h`, `llk_unpack_AB_reduce.h`
- `llk_unpack_untilize.h`

**Shared files** (13 of quasar's 23): `llk_defs.h`, `llk_math_common.h`,
`llk_math_eltwise_binary.h`, `llk_math_eltwise_unary_datacopy.h`, `llk_math_matmul.h`,
`llk_math_reduce.h`, `llk_memory_checks.h`, `llk_pack.h`, `llk_pack_common.h`,
`llk_pack_untilize.h`, `llk_unpack_common.h`, `llk_unpack_reduce.h`, `llk_unpack_tilize.h`.

The naming conventions differ too. Quasar uses `llk_unpack_unary_operand.h` where
wormhole_b0/blackhole use `llk_unpack_A.h`, and `llk_unpack_binary_operands.h` where they
use `llk_unpack_AB.h`. Quasar also has TDMA source management (`llk_srcs_tdma.h`) and
broadcast-specific files that have no counterpart in the other architectures.

In `common/inc/`, quasar has only 34 files versus 70 for wormhole_b0/blackhole. It is
missing most of the SFPU operation headers (14 sfpu files vs. 52) and introduces unique
headers like `ckernel_dest.h`, `ckernel_pcbuf.h`, `ckernel_proj_params.h`,
`ckernel_risc_atomics.h`, and `ckernel_vector.h`.

## Metal's Mirroring of Per-Architecture Directories

TT-Metal's [`hw/ckernels/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels)
directory replicates the per-architecture split:

```
tt_metal/hw/ckernels/
    wormhole_b0/metal/llk_api/   (21 files + llk_sfpu/ subdir, 12,310 lines total)
    blackhole/metal/llk_api/     (21 files + llk_sfpu/ subdir + experimental/, 12,284 lines total)
    quasar/metal/llk_api/        (7 files, 593 lines total)
```

Each `llk_api/` directory contains `*_api.h` wrapper headers that call into the vendored
LLK code. The wormhole_b0 and blackhole wrappers are structurally identical (21 files each,
~12,300 lines), while quasar has only 7 wrapper files at 593 lines, reflecting its smaller
and differently-structured LLK surface.

The wormhole_b0 and blackhole `llk_api/` wrappers expose the same API surface:
```
llk_math_binary_api.h          llk_unpack_A_api.h
llk_math_binary_sfpu_api.h     llk_unpack_AB_api.h
llk_math_common_api.h          llk_unpack_AB_matmul_api.h
llk_math_matmul_api.h          llk_unpack_AB_reduce_api.h
llk_math_reduce_api.h          llk_unpack_AB_reduce_custom_api.h
llk_math_reduce_custom_api.h   llk_unpack_common_api.h
llk_math_transpose_dest_api.h  llk_unpack_reduce_api.h
llk_math_unary_datacopy_api.h  llk_unpack_tilize_api.h
llk_math_unary_sfpu_api.h      llk_unpack_untilize_api.h
llk_pack_api.h
```

Quasar exposes only 7 of these:
`llk_math_common_api.h`, `llk_math_matmul_api.h`, `llk_math_unary_datacopy_api.h`,
`llk_pack_api.h`, `llk_unpack_AB_matmul_api.h`, `llk_unpack_A_api.h`, `llk_unpack_common_api.h`.

### The HAL Layer Also Duplicates Per-Architecture Logic

Metal's Hardware Abstraction Layer under
[`llrt/hal/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/llrt/hal)
groups architectures into families and then further splits by specific chip:

```
llrt/hal/
    tt-1xx/
        hal_1xx_common.cpp, hal_1xx_common.hpp
        wormhole/   (7 files, 921 lines)
        blackhole/  (7 files, 1,012 lines)
    tt-2xx/
        hal_2xx_common.cpp, hal_2xx_common.hpp
        quasar/     (7 files, 1,161 lines)
```

Each per-chip HAL directory contains the same structural pattern of 7 files:
`*_hal.cpp`, `*_hal.hpp`, `*_hal_active_eth.cpp`, `*_hal_eth_asserts.hpp`,
`*_hal_idle_eth.cpp`, `*_hal_tensix.cpp`, `*_hal_tensix_asserts.hpp`.

Combined, the HAL per-architecture code totals 3,094 lines across the three chip directories,
plus family-level common files. The family grouping (tt-1xx vs. tt-2xx) provides some
code sharing between wormhole and blackhole, but each chip still requires its own
implementation files.

---

**Next:** [`maintenance_burden.md`](./maintenance_burden.md)
