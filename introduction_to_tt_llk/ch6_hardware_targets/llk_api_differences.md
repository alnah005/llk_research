# LLK API Differences

## Overview

While the `common/inc/` layer captures chip-level infrastructure differences, the `llk_lib/` directory is where the divergence manifests at the API level -- the headers that test authors and kernel developers interact with directly. Wormhole B0 and Blackhole share identical `llk_lib/` contents, while Quasar uses a different file organization, different naming conventions, and includes operations unique to its architecture.

## WH/BH: Identical LLK Libraries

The `tt_llk_wormhole_b0/llk_lib/` and `tt_llk_blackhole/llk_lib/` directories contain exactly the same set of files with the same names. These 28 files (plus the `experimental/` subdirectory) define the complete LLK API for both architectures:

**Core infrastructure:**
- `llk_defs.h` -- Enum and type definitions
- `llk_math_common.h` -- Shared math thread setup and helpers
- `llk_pack_common.h` -- Shared pack thread setup and helpers
- `llk_unpack_common.h` -- Shared unpack thread setup and helpers
- `llk_memory_checks.h` -- Runtime memory validation

**Unpack operations:**
- `llk_unpack_A.h` -- Single-operand (unary) unpack
- `llk_unpack_AB.h` -- Dual-operand (binary) unpack
- `llk_unpack_AB_matmul.h` -- Matrix multiply unpack (A and B operands)
- `llk_unpack_AB_reduce.h` -- Reduce-with-two-operands unpack
- `llk_unpack_reduce.h` -- Reduction unpack
- `llk_unpack_tilize.h` -- Row-major to tile-order conversion during unpack
- `llk_unpack_untilize.h` -- Tile-order to row-major conversion during unpack

**Math operations:**
- `llk_math_eltwise_binary.h` -- FPU element-wise binary (add, mul, sub)
- `llk_math_eltwise_binary_sfpu.h` -- Binary SFPU operations
- `llk_math_eltwise_binary_sfpu_params.h` -- Binary SFPU parameter/face iteration
- `llk_math_eltwise_ternary_sfpu.h` -- Ternary SFPU operations
- `llk_math_eltwise_ternary_sfpu_params.h` -- Ternary SFPU parameter/face iteration
- `llk_math_eltwise_unary_datacopy.h` -- Data copy through math pipeline
- `llk_math_eltwise_unary_sfpu.h` -- Unary SFPU operations
- `llk_math_eltwise_unary_sfpu_params.h` -- Unary SFPU parameter/face iteration
- `llk_math_matmul.h` -- Matrix multiplication
- `llk_math_reduce.h` -- Reduction operations (row, column, scalar)
- `llk_math_transpose_dest.h` -- In-place transpose within Destination register
- `llk_math_welfords_sfpu.h` -- Welford's online algorithm for variance
- `llk_math_welfords_sfpu_params.h` -- Welford's face iteration parameters

**Pack operations:**
- `llk_pack.h` -- Standard tile packing
- `llk_pack_common.h` -- Shared pack helpers (listed above but also a pack file)
- `llk_pack_rows.h` -- Row-granularity packing
- `llk_pack_untilize.h` -- Untilize during pack

## Quasar: Different Organization

Quasar's `tt_llk_quasar/llk_lib/` contains 23 files with a noticeably different structure. The differences fall into three categories: renamed files, unique files, and absent files.

### Renamed Unpack Files

The most visible naming difference is in the unpack layer. Where WH/BH use letter-based names (`A` for single operand, `AB` for two operands), Quasar uses descriptive names:

| WH/BH | Quasar | Purpose |
|--------|--------|---------|
| `llk_unpack_A.h` | `llk_unpack_unary_operand.h` | Single-operand unpack |
| `llk_unpack_AB.h` | `llk_unpack_binary_operands.h` | Dual-operand unpack |
| `llk_unpack_AB_matmul.h` | `llk_unpack_matmul.h` | Matrix multiply unpack |

Quasar drops the `AB_` prefix pattern entirely, opting for operation-descriptive names instead.

### Quasar-Unique Files

These files exist only in `tt_llk_quasar/llk_lib/` and have no WH/BH equivalent:

- **`llk_srcs_tdma.h`** -- TDMA (Time-Division Multiple Access) source management. Quasar uses a TDMA mechanism for coordinating access to shared source operand buffers that does not exist on WH/BH.
- **`llk_pack_matmul.h`** -- A dedicated packing path for matrix multiply results. On WH/BH, matmul results are packed through the standard `llk_pack.h` path; Quasar separates this into its own header.
- **`llk_math_eltwise_binary_broadcast.h`** -- Element-wise binary operations with built-in broadcast semantics. On WH/BH, broadcasting is handled within `llk_math_eltwise_binary.h` via the `BroadcastType` template parameter; Quasar factors this out into a separate file.
- **`llk_math_unary_broadcast.h`** -- Unary broadcast operations, another broadcast-specific factoring that WH/BH handles inline.
- **`llk_unpack_unary_broadcast_operands.h`** -- Unpack path for unary broadcast operations.
- **`llk_unpack_binary_broadcast_operands.h`** -- Unpack path for binary broadcast operations.

The broadcast-specific files reflect a design decision in Quasar to separate broadcast variants into their own compilation units rather than handling them through template specialization within the base operation files.

### Files Present in WH/BH but Absent in Quasar

These WH/BH files have no Quasar equivalent:

- **`llk_pack_rows.h`** -- Row-granularity packing is not supported on Quasar.
- **`llk_math_transpose_dest.h`** -- In-place Destination transpose is not available on Quasar.
- **`llk_math_welfords_sfpu.h`** and **`llk_math_welfords_sfpu_params.h`** -- Welford's online variance algorithm is not implemented for Quasar.
- **`llk_math_eltwise_binary_sfpu.h`** and **`llk_math_eltwise_binary_sfpu_params.h`** -- Binary SFPU operations are not available on Quasar (consistent with its reduced `SfpuType` enum).
- **`llk_math_eltwise_ternary_sfpu.h`** and **`llk_math_eltwise_ternary_sfpu_params.h`** -- Ternary SFPU operations are likewise absent.
- **`llk_math_eltwise_unary_sfpu_params.h`** -- The separate params file for unary SFPU is replaced by the unified common header (see below).
- **`llk_unpack_AB_reduce.h`** -- Reduce-with-two-operands unpack is not present on Quasar.
- **`llk_unpack_untilize.h`** -- Untilize-during-unpack is not available on Quasar.
- **`experimental/`** -- Quasar has no experimental subdirectory.

### Unified SFPU Common Header

On WH/BH, unary SFPU operations are split across two files:
- `llk_math_eltwise_unary_sfpu.h` -- Lifecycle functions (init, configure, execute)
- `llk_math_eltwise_unary_sfpu_params.h` -- Face iteration and parameter handling

Quasar replaces both with a single file:
- **`llk_math_eltwise_unary_sfpu_common.h`** -- Contains both the lifecycle functions and the face iteration logic in one header.

This consolidation aligns with Quasar's smaller `SfpuType` surface area (16 operations vs. 112), making the split less necessary.

## Boot Mode Differences

The boot mode determines how a kernel is launched on the chip. This is defined in `tests/python_tests/helpers/device.py`:

```python
class BootMode(Enum):
    BRISC = "brisc"
    TRISC = "trisc"
    EXALENS = "exalens"
    DEFAULT = "default"

CHIP_DEFAULT_BOOT_MODES = {
    ChipArchitecture.WORMHOLE: BootMode.BRISC,
    ChipArchitecture.BLACKHOLE: BootMode.BRISC,
    ChipArchitecture.QUASAR: BootMode.TRISC,
}
```

- **WH/BH default to `BootMode.BRISC`**: The BRISC core boots first, loads the TRISC ELF images, and orchestrates kernel launch. This is the standard two-stage boot where BRISC acts as a coordinator.
- **Quasar defaults to `BootMode.TRISC`**: Since Quasar has no BRISC core, kernels boot directly on the TRISC cores. There is no BRISC ELF in the Quasar L1 memory layout.

The test infrastructure resolves `BootMode.DEFAULT` to the appropriate architecture-specific mode at runtime (in `tests/python_tests/helpers/test_config.py`), so most test code does not need to handle this distinction explicitly.

## Instruction Definitions: `assembly.yaml`

Each architecture defines its instruction set in an `assembly.yaml` file located at:

- `tt_llk_wormhole_b0/instructions/assembly.yaml`
- `tt_llk_blackhole/instructions/assembly.yaml`
- `tt_llk_quasar/instructions/assembly.yaml`

These YAML files specify every Tensix instruction available on the architecture: the instruction type (e.g., `LOCAL_CREGS`, `PC_MODIFYING`, `COMPUTE`, `COMMON_CREGS`), the binary opcode (`op_binary`), execution resource (`ex_resource`), and detailed argument definitions including bit positions and field types. The instruction sets differ across architectures, reflecting their different hardware capabilities.

## Summary Table

| Aspect | Wormhole B0 | Blackhole | Quasar |
|--------|-------------|-----------|--------|
| `common/inc/` unique headers | xmov, mutex_guard, debug, common_ops, globals, structs | (identical to WH) | pcbuf, vector, dest, risc_atomics, trisc_common, proj_params, riscv_debug |
| `llk_lib/` file count | 28 + experimental/ | 28 + experimental/ | 23 |
| Enum style | Unscoped enums | Unscoped enums | `enum class` (scoped) |
| `SfpuType` operations | 112 | 112 | 16 |
| BRISC core | Yes | Yes | No |
| TRISC cores | 3 (TRISC0-2) | 3 (TRISC0-2) | 4 (TRISC0-3) |
| Default boot mode | `BootMode.BRISC` | `BootMode.BRISC` | `BootMode.TRISC` |
| Broadcast handling | Inline via template params | Inline via template params | Separate `*_broadcast*.h` files |
| Binary/ternary SFPU | Supported | Supported | Not available |
| Welford's algorithm | Supported | Supported | Not available |

---

**Next:** [Chapter 7 -- Test Infrastructure](../ch7_test_infrastructure/index.md)
