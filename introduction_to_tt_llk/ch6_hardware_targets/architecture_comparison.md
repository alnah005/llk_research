# Architecture Comparison

## Overview

The three TT-LLK hardware targets -- Wormhole B0, Blackhole, and Quasar -- share a common compute abstraction (the Tensix core with unpack/math/pack threads) but differ in their low-level infrastructure. The clearest picture of these differences emerges from comparing the `common/inc/` directories, where chip-specific ckernel headers live.

## Shared API Surface

All three targets implement the same fundamental LLK operations: tile unpacking, FPU/SFPU math, and packing. The high-level lifecycle pattern (init, configure, execute, reconfigure) is consistent across architectures. Each target provides these common headers in `common/inc/`:

| Header | Purpose |
|--------|---------|
| `ckernel.h` | Top-level ckernel include |
| `ckernel_addrmod.h` | Address modifier configuration |
| `ckernel_defs.h` | Architecture-specific constants and defines |
| `ckernel_gpr_map.h` | General-purpose register mapping |
| `ckernel_include.h` | Master include aggregation |
| `ckernel_instr_params.h` | Instruction parameter encoding |
| `ckernel_ops.h` | Low-level ckernel operation wrappers |
| `ckernel_sfpu.h` | SFPU instruction primitives |
| `ckernel_template.h` | Templated ckernel utilities |
| `cmath_common.h` | Common math thread helpers |
| `cpack_common.h` | Common pack thread helpers |
| `cunpack_common.h` | Common unpack thread helpers |
| `sfpu/` | Directory of SFPU kernel implementations |

## Wormhole B0 and Blackhole: Near-Identical Targets

Wormhole B0 (`tt_llk_wormhole_b0/common/inc/`) and Blackhole (`tt_llk_blackhole/common/inc/`) have **identical file listings**. Beyond the shared headers above, both targets include:

- **`ckernel_xmov.h`** -- Cross-move instruction helpers for data movement between register files.
- **`ckernel_mutex_guard.h`** -- RAII-style mutex guard for synchronizing access to shared hardware resources between BRISC and TRISC threads.
- **`ckernel_debug.h`** -- Debug utilities for dumping register state and inserting breakpoints.
- **`ckernel_common_ops.h`** -- Common operation building blocks shared across math operations.
- **`ckernel_globals.h`** -- Global variable declarations for ckernel state.
- **`ckernel_structs.h`** -- Structure definitions for hardware register layouts.

These two architectures also share the same `llk_defs.h` enum definitions (see below) and the same `llk_lib/` file organization.

## Quasar: A Different Architecture

Quasar (`tt_llk_quasar/common/inc/`) diverges significantly. It **lacks** `ckernel_xmov.h`, `ckernel_mutex_guard.h`, `ckernel_debug.h`, `ckernel_common_ops.h`, `ckernel_globals.h`, and `ckernel_structs.h`. In their place, Quasar introduces several unique headers:

- **`ckernel_pcbuf.h`** -- PC buffer management. Quasar uses a PC (Program Counter) buffer mechanism for instruction dispatch that does not exist on WH/BH.
- **`ckernel_vector.h`** -- Vector operation primitives specific to Quasar's vector execution model.
- **`ckernel_dest.h`** -- Explicit Destination register management, handling dest access patterns that WH/BH manage differently.
- **`ckernel_risc_atomics.h`** -- Atomic operation support for RISC cores. Quasar provides hardware atomics for inter-thread synchronization.
- **`ckernel_trisc_common.h`** -- Common utilities for TRISC cores. Since Quasar lacks a BRISC core entirely, all compute runs on TRISC threads.
- **`ckernel_proj_params.h`** -- Project-level parameter definitions specific to the Quasar architecture.
- **`ckernel_riscv_debug.h`** -- RISC-V debug facilities (replaces the WH/BH `ckernel_debug.h` with a different interface).
- **`internal/`** -- A subdirectory for internal implementation headers not present in WH/BH.

### No BRISC Core on Quasar

A fundamental architectural difference: Quasar **does not have a BRISC core**. On WH/BH, the BRISC (Boot RISC) core handles kernel bootstrapping and data movement coordination. The `RiscCore` enum in `tests/python_tests/helpers/device.py` makes this explicit:

```python
RiscCore.BRISC: -1 if is_quasar else 11,  # -1 = invalid/nonexistent
```

Quasar compensates with four TRISC cores (`TRISC0` through `TRISC3`) instead of WH/BH's three (`TRISC0` through `TRISC2`). The `get_all_cores()` function in `tests/python_tests/helpers/device.py` returns different core lists per architecture:

```python
# Quasar: four TRISC cores, no BRISC
[RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2, RiscCore.TRISC3]

# WH/BH: BRISC + three TRISC cores
[RiscCore.BRISC, RiscCore.TRISC0, RiscCore.TRISC1, RiscCore.TRISC2]
```

## Enum Style Differences

One of the most pervasive differences between the targets is the enum declaration style. Comparing `tt_llk_wormhole_b0/llk_lib/llk_defs.h` and `tt_llk_quasar/llk_lib/llk_defs.h` reveals a systematic shift from unscoped to scoped enums.

### WH/BH: Unscoped Enums

WH/BH use traditional C-style unscoped enums, where enumerators leak into the enclosing `ckernel` namespace:

```cpp
// tt_llk_wormhole_b0/llk_lib/llk_defs.h
enum ReduceDim { REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR };
enum EltwiseBinaryType { ELWMUL, ELWDIV, ELWADD, ELWSUB, ELWLESS };
enum BroadcastType { NONE = 0x0, COL = 0x1, ROW = 0x2, SCALAR = 0x3 };
enum DstSync { SyncHalf = 0, SyncFull = 1 };
enum DataCopyType { A2D, B2D };
enum PoolType { SUM, AVG, MAX, MIN };
enum ReluType { NO_RELU, ZERO_RELU, MIN_THRESHOLD_RELU, MAX_THRESHOLD_RELU };
```

### Quasar: Scoped `enum class`

Quasar uses C++11 `enum class` (scoped enums) with explicit underlying types for all the same logical enumerations:

```cpp
// tt_llk_quasar/llk_lib/llk_defs.h
enum class ReduceDim : std::uint8_t { REDUCE_ROW, REDUCE_COL, REDUCE_SCALAR };
enum class EltwiseBinaryType : std::uint8_t { ELWMUL, ELWADD, ELWSUB };
enum class BroadcastType : std::uint8_t { NONE, COL, ROW, SCALAR };
enum class DstSync : std::uint8_t { SyncHalf, SyncFull };
enum class DataCopyType : std::uint8_t { A2D, B2D };
enum class PoolType : std::uint8_t { SUM, AVG, MAX };
enum class ReluType : std::uint8_t { NO_RELU, ZERO_RELU, MIN_THRESHOLD_RELU, MAX_THRESHOLD_RELU };
```

Notable value differences beyond the scoping:

- **`EltwiseBinaryType`**: WH/BH includes `ELWDIV` and `ELWLESS`; Quasar does not.
- **`PoolType`**: WH/BH includes `MIN`; Quasar does not.
- **`StochRndType::All`**: WH/BH sets this to `0xf`; Quasar sets it to `3`.

Both targets use `enum class` for `EltwiseBinaryReuseDestType` and `MathFidelity`, so those are consistent.

### Quasar-Exclusive Types

Quasar's `llk_defs.h` also includes a `ReluConfig` struct -- a type-safe wrapper around `ReluType` plus a threshold value -- that does not exist in the WH/BH `llk_defs.h`. This struct provides factory methods (`ReluConfig::none()`, `ReluConfig::zero()`, `ReluConfig::min_threshold(t)`, `ReluConfig::max_threshold(t)`) and handles the hardware bit-packing internally.

### WH/BH-Exclusive Types

WH/BH `llk_defs.h` defines several types absent from Quasar:

- **`VectorMode`** -- Controls SFPU face iteration (`None`, `R`, `C`, `RC`, `RC_custom`).
- **`TileDim`** -- Tile dimension indexing (`R_IDX`, `C_IDX`).
- **`src_op_id_e`**, **`local_op_id_e`**, **`out_op_id_e`** -- Operand slot identifiers.
- **`InstrModLoadStore`** -- SFPLOAD/SFPSTORE instruction modifier encoding for different data formats.
- **`GetSfpLoadStoreInstrMod<DataFormat>()`** -- Compile-time mapping from `DataFormat` to `InstrModLoadStore`.

## SfpuType: 112 vs. 16 Operations

The `SfpuType` enum is the most dramatic divergence. WH/BH define 112 SFPU operations in `tests/helpers/include/llk_sfpu_types.h` (guarded by `#ifndef ARCH_QUASAR`), covering the full range of activation functions, transcendentals, comparisons, integer arithmetic, bitwise operations, and specialized kernels like TopK and Welford reduction.

Quasar defines only 16 operations directly in `tt_llk_quasar/llk_lib/llk_defs.h`:

```cpp
enum class SfpuType : std::uint32_t {
    tanh, gelu, exponential, reciprocal, sqrt, rsqrt,
    relu, lrelu, relumin, relumax, stochround,
    typecast, add, square, sigmoid, silu
};
```

This reduced set represents the core activations and mathematical primitives that Quasar currently supports in hardware. The `using SfpuType = ckernel::SfpuType;` at the bottom of Quasar's `llk_defs.h` brings it into the global namespace for compatibility with the test infrastructure.

---

**Next:** [`llk_api_differences.md`](./llk_api_differences.md)
