# Maintenance Burden

## Bug Fixes Must Be Applied N Times per Architecture

The repository-level [`common/`](https://github.com/tenstorrent/tt-llk/tree/main/common)
directory contains exactly 2 files:

- `llk_assert.h`
- `tensor_shape.h`

Every other LLK source file lives inside an architecture-specific directory
(`tt_llk_wormhole_b0/`, `tt_llk_blackhole/`, or `tt_llk_quasar/`). This means any bug fix
to LLK code outside those 2 shared files must be manually applied to up to 3 separate copies.

Consider a fix to `llk_math_matmul.h`, which exists in all three architectures:

| Copy | Path | Lines |
|---|---|---|
| 1 | [`tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h) | 874 |
| 2 | [`tt_llk_blackhole/llk_lib/llk_math_matmul.h`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_blackhole/llk_lib/llk_math_matmul.h) | varies |
| 3 | [`tt_llk_quasar/llk_lib/llk_math_matmul.h`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_quasar/llk_lib/llk_math_matmul.h) | 356 |

Even for the 13 files that share names across all three architectures, the implementations
differ enough that a patch cannot be blindly `cp`'d -- the quasar version of `llk_math_matmul.h`
is 356 lines versus wormhole_b0's 874 lines, indicating substantially different internals.

After the LLK fix lands, Metal's `llk_api/` wrappers may also need corresponding updates in:

| Metal copy | Path |
|---|---|
| 1 | [`hw/ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_matmul_api.h) |
| 2 | [`hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h) |
| 3 | [`hw/ckernels/quasar/metal/llk_api/llk_math_matmul_api.h`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/quasar/metal/llk_api/llk_math_matmul_api.h) |

A single logical bug fix thus requires touching up to **6 files** across 2 repositories.

## Adding a New LLK Operation

Introducing a new LLK operation (e.g., a new math kernel) requires creating files in each
architecture directory, plus corresponding Metal wrappers. The minimum set of touchpoints:

1. **LLK `llk_lib/`** -- Create the new `llk_math_*.h` (or `llk_unpack_*.h`, `llk_pack_*.h`)
   in each of:
   - `tt_llk_wormhole_b0/llk_lib/`
   - `tt_llk_blackhole/llk_lib/`
   - `tt_llk_quasar/llk_lib/`

2. **LLK `common/inc/`** -- If the operation needs new ckernel or SFPU support, add headers in:
   - `tt_llk_wormhole_b0/common/inc/` (and/or `sfpu/`)
   - `tt_llk_blackhole/common/inc/` (and/or `sfpu/`)
   - `tt_llk_quasar/common/inc/` (and/or `sfpu/`)

3. **Metal `llk_api/` wrappers** -- Create `llk_*_api.h` wrappers in:
   - `hw/ckernels/wormhole_b0/metal/llk_api/`
   - `hw/ckernels/blackhole/metal/llk_api/`
   - `hw/ckernels/quasar/metal/llk_api/`

4. **Metal HAL** -- If the operation affects hardware configuration, update per-arch HAL files
   under `llrt/hal/tt-1xx/{wormhole,blackhole}/` and `llrt/hal/tt-2xx/quasar/`.

For wormhole_b0 and blackhole, steps 1-3 can often be done by copying files between the two
(given their near-identical structure), then adjusting architecture-specific register addresses
or timing parameters. For quasar, the implementation must be written from scratch due to its
different file organization and naming conventions.

## Quasar's Structural Divergence Prevents Mechanical Porting

The differences between quasar and the other architectures go beyond content -- they are
structural:

**Different naming conventions.** Quasar uses descriptive names like
`llk_unpack_unary_operand.h` and `llk_unpack_binary_operands.h` where wormhole_b0/blackhole use
abstract names like `llk_unpack_A.h` and `llk_unpack_AB.h`. This means automated scripts
that look for files by name will fail to find the corresponding quasar file.

**Different decomposition of functionality.** Quasar splits broadcast operations into
dedicated files (`llk_math_eltwise_binary_broadcast.h`, `llk_math_unary_broadcast.h`,
`llk_unpack_binary_broadcast_operands.h`, `llk_unpack_unary_broadcast_operands.h`) that do
not exist in the other architectures. Conversely, quasar lacks the SFPU parameter files
(`*_sfpu_params.h`) and the ternary SFPU files present in wormhole_b0/blackhole.

**Different hardware abstractions.** Quasar introduces TDMA source management
([`llk_srcs_tdma.h`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_quasar/llk_lib/llk_srcs_tdma.h)),
a PCBuf interface
([`ckernel_pcbuf.h`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_quasar/common/inc/ckernel_pcbuf.h)),
RISC atomics
([`ckernel_risc_atomics.h`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_quasar/common/inc/ckernel_risc_atomics.h)),
and vector operations
([`ckernel_vector.h`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_quasar/common/inc/ckernel_vector.h))
that have no counterpart in wormhole_b0/blackhole. These reflect genuine hardware differences
in the quasar chip architecture.

**Dramatically different SFPU coverage.** Quasar has 14 SFPU operation headers (658 lines)
versus 52 headers (9,774-10,092 lines) in wormhole_b0/blackhole. The SFPU operations that
exist in quasar use different names (e.g., `ckernel_sfpu_add.h` vs. the absent
`ckernel_sfpu_binary.h`) and cannot be assumed to share implementations.

These structural differences mean that a developer porting a fix from wormhole_b0 to quasar
must understand both architectures' file organizations, locate the conceptually equivalent
code (if it exists), and adapt the fix to quasar's different abstractions. This cannot be
automated with simple file-level tooling.

## The Instructions Directory Reflects Genuine Hardware Differences

Each architecture's `instructions/assembly.yaml` defines the hardware instruction set:

| Architecture | Lines |
|---|---|
| wormhole_b0 | 3,006 |
| blackhole | 3,571 |
| quasar | 7,050 |

Quasar's instruction file is more than twice the size of wormhole_b0's, reflecting a
genuinely different and more complex instruction set. This is one area where per-architecture
duplication is **unavoidable** -- different chips have different instructions, and no amount
of abstraction can eliminate this divergence.

The instruction files are consumed by codegen tooling to produce C++ headers, so their
per-architecture split is architecturally justified. However, they contribute to the overall
impression of scale: the `instructions/` directories alone account for 13,627 lines across
the three architectures.

---

**Next:** [Chapter 6 -- Template Design Trade-offs](../ch6_template_tradeoffs/index.md)
