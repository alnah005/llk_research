# Repository Structure

The `tt-llk` repository is organized around hardware targets: each supported Tenstorrent chip architecture has its own top-level directory containing a complete set of LLK headers. A shared `common/` directory at the repository root holds cross-architecture utilities, while `tests/` and `docs/` provide the validation and documentation infrastructure.

## Top-Level Layout

```
tt-llk/
  tt_llk_wormhole_b0/    # LLK implementation for Wormhole B0
  tt_llk_blackhole/       # LLK implementation for Blackhole
  tt_llk_quasar/          # LLK implementation for Quasar
  common/                 # Shared headers used across all architectures
  tests/                  # Test infrastructure and test cases
  docs/                   # Documentation (L1, L2, L3 levels)
  infra/                  # Build and CI infrastructure
  README.md
  CONTRIBUTING.md
```

## Per-Architecture Directory Structure

Each architecture directory (`tt_llk_wormhole_b0/`, `tt_llk_blackhole/`, `tt_llk_quasar/`) follows the same internal layout with three subdirectories:

```
tt_llk_<arch>/
  common/
    inc/                  # Ckernel headers and SFPU implementations
      sfpu/               # SFPU operation implementations (per-function headers)
      ckernel_*.h         # Core kernel infrastructure headers
      cmath_common.h      # Shared math utility functions
      cpack_common.h      # Shared packing utility functions
      cunpack_common.h    # Shared unpacking utility functions
  instructions/           # Assembly instruction definitions (YAML)
    assembly.yaml         # Tensix ISA instruction encodings for this architecture
  llk_lib/                # LLK API headers (the public interface)
    llk_unpack_*.h        # Unpack operation APIs
    llk_math_*.h          # Math operation APIs
    llk_pack*.h           # Pack operation APIs
    llk_defs.h            # Common LLK definitions and enumerations
```

### `common/inc/` -- Ckernel Headers

This subdirectory contains the internal implementation machinery. The headers here are not the public API; they are the lower-level building blocks that the `llk_lib/` headers call into. Key files include:

| File | Purpose |
|:-----|:--------|
| `ckernel.h` | Top-level include for ckernel infrastructure |
| `ckernel_defs.h` | Hardware constants, register addresses, configuration values |
| `ckernel_template.h` | Templated instruction emission helpers |
| `ckernel_ops.h` | Low-level Tensix instruction wrappers |
| `ckernel_addrmod.h` | Address mode configuration for Tensix instructions |
| `ckernel_globals.h` | Global state shared across ckernel functions |
| `ckernel_sfpu.h` | SFPU instruction dispatch and configuration |
| `cmath_common.h` | Common math functions shared across math LLKs |
| `cpack_common.h` | Common packing functions shared across pack LLKs |
| `cunpack_common.h` | Common unpacking functions shared across unpack LLKs |

The `sfpu/` subdirectory contains per-operation SFPU implementations as individual headers (e.g., `ckernel_sfpu_exp.h`, `ckernel_sfpu_gelu.h`, `ckernel_sfpu_binary.h`). Each file implements one SFPU operation using the SFPI (SFPU Programming Interface) intrinsics.

### `instructions/` -- Assembly Definitions

Each architecture directory contains an `assembly.yaml` file that defines the Tensix ISA instruction encodings for that chip. These YAML files specify instruction formats, field positions, and valid operand ranges. They serve as the ground truth for how C++ instruction-emission code maps to binary instruction words.

### `llk_lib/` -- LLK API Headers

This is the **public interface** of TT-LLK. The headers here define the functions that kernel authors (and TT-Metal) call directly. They follow a consistent naming convention organized by pipeline stage:

**Unpack headers** (TRISC0):

| Header | Description |
|:-------|:------------|
| `llk_unpack_A.h` | Unpack a single tile from L1 into Source A (unary operations) |
| `llk_unpack_AB.h` | Unpack two tiles into Source A and Source B (binary operations) |
| `llk_unpack_AB_matmul.h` | Optimized unpack for matrix multiplication with register reuse |
| `llk_unpack_reduce.h` | Unpack a tile into Source A and a scalar into Source B (reductions) |
| `llk_unpack_tilize.h` | Convert row-major data to tile format during unpack |
| `llk_unpack_untilize.h` | Convert tile data to row-major format during unpack |
| `llk_unpack_AB_reduce.h` | Unpack two tiles for fused binary-reduce operations |
| `llk_unpack_common.h` | Shared unpack utilities and configuration |

**Math headers** (TRISC1):

| Header | Description |
|:-------|:------------|
| `llk_math_eltwise_unary_datacopy.h` | Transfer a tile from Source A/B to the destination register via FPU |
| `llk_math_eltwise_unary_sfpu.h` | Execute unary SFPU operations on destination register data |
| `llk_math_eltwise_unary_sfpu_params.h` | Parameter definitions for unary SFPU math operations |
| `llk_math_eltwise_binary.h` | Element-wise add/sub/mul on Source A and Source B via FPU |
| `llk_math_eltwise_binary_sfpu.h` | Execute binary SFPU operations on destination register data |
| `llk_math_eltwise_binary_sfpu_params.h` | Parameter definitions for binary SFPU math operations |
| `llk_math_eltwise_ternary_sfpu.h` | Execute ternary SFPU operations on destination register data |
| `llk_math_eltwise_ternary_sfpu_params.h` | Parameter definitions for ternary SFPU math operations |
| `llk_math_matmul.h` | Matrix multiplication between Source A and Source B tiles |
| `llk_math_reduce.h` | Reduction operations (SUM, AVG, MAX, MIN) via FPU |
| `llk_math_transpose_dest.h` | Transpose tile data within the destination register |
| `llk_math_welfords_sfpu.h` | Welford's online algorithm for variance/standard deviation via SFPU |
| `llk_math_welfords_sfpu_params.h` | Parameter definitions for Welford's SFPU operations |
| `llk_memory_checks.h` | Memory bounds checking and validation utilities |
| `llk_math_common.h` | Shared math utilities and configuration |

**Pack headers** (TRISC2):

| Header | Description |
|:-------|:------------|
| `llk_pack.h` | Pack a tile from the destination register to L1 in tile format |
| `llk_pack_untilize.h` | Pack a tile from the destination register to L1 in row-major format |
| `llk_pack_rows.h` | Pack individual rows from the destination register to L1 |
| `llk_pack_common.h` | Shared pack utilities and configuration |

## The `common/` Directory (Repository Root)

At the repository root, the `common/` directory contains headers shared across all architecture targets:

- **`llk_assert.h`** -- A portable assertion macro (`LLK_ASSERT`) that adapts to the execution environment. When running under the LLK test infrastructure, it triggers a RISC-V breakpoint (`ebreak`). When running under TT-Metal, it delegates to TT-Metal's own assertion system. When assertions are disabled, the condition is still compiled (for type-checking) but never executed.

- **`tensor_shape.h`** -- Defines the `TensorShape` struct used across all LLK operations to describe tile dimensions. A tile is composed of faces arranged in a grid; the default shape is 32x32 (four 16x16 faces). This header also defines key constants like `MAX_FACE_R_DIM` (16), `MAX_FACE_C_DIM` (16), `MAX_TILE_R_DIM` (32), and `MAX_TILE_C_DIM` (32).

These shared headers are architecture-independent and provide foundational types and utilities that all three hardware targets rely on.

## The Three Hardware Targets

TT-LLK supports three Tenstorrent chip architectures, each with its own implementation directory:

### Wormhole B0 (`tt_llk_wormhole_b0/`)

Wormhole B0 is a mature, widely deployed Tenstorrent ASIC. Its LLK implementation is the most complete and serves as the reference for new feature development.

### Blackhole (`tt_llk_blackhole/`)

Blackhole is a newer-generation Tenstorrent ASIC. It shares the same directory layout, header file names, and API surface as Wormhole B0, but with chip-specific instruction encodings and hardware constants.

### Quasar (`tt_llk_quasar/`)

Quasar is a distinct architecture with a noticeably different internal structure. While it follows the same three-subdirectory layout (`common/inc/`, `instructions/`, `llk_lib/`), the `common/inc/` directory contains different header files (e.g., `ckernel_dest.h`, `ckernel_pcbuf.h`, `ckernel_vector.h` instead of WH/BH's `ckernel_common_ops.h`, `ckernel_xmov.h`). The `llk_lib/` API also uses different naming conventions in some cases (e.g., `llk_unpack_unary_operand.h` instead of `llk_unpack_A.h`, `llk_unpack_binary_operands.h` instead of `llk_unpack_AB.h`). These differences reflect Quasar's distinct hardware design.

### Why Separate Directories?

Each chip has its own Tensix ISA encoding, different hardware constants (register addresses, buffer sizes, configuration bit fields), and potentially different capabilities. Keeping separate implementation directories ensures that:

1. Chip-specific optimizations are not constrained by other architectures.
2. Instruction encodings in `assembly.yaml` exactly match each chip's hardware.
3. New architectures can be added by creating a new top-level directory without modifying existing ones.
4. Build systems can cleanly select the correct headers by pointing an include path at the appropriate `tt_llk_<arch>/` directory.

## Tests and Documentation

The `tests/` directory contains the standalone test infrastructure, including Python test scripts (in `tests/python_tests/`), hardware-specific test configurations (in `tests/hw_specific/`), helper headers and linker scripts (in `tests/helpers/`), test source files (in `tests/sources/`), and SFPI version tracking files (`sfpi-info.sh`, `sfpi-version`) at the tests root.

The `docs/` directory organizes documentation into three levels of increasing detail:
- **L1** (`docs/llk/l1/`) -- High-level introduction for broad audiences.
- **L2** (`docs/llk/l2/`) -- Architectural overview of the Tensix Core and engine.
- **L3** (`docs/llk/l3/`) -- Programming model details for kernel developers.

---

**Next:** [Chapter 2 -- Tensix Core Architecture](../ch2_tensix_core_architecture/index.md)
