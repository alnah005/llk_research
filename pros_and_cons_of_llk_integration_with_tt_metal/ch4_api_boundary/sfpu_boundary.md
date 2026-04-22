# SFPU Boundary Ambiguity

The SFPU (Special Function Processing Unit) kernel layer is the largest single body of code at the API boundary between LLK and Metal. It is authored and maintained by Metal, not by tt-llk, yet it depends heavily on LLK infrastructure. This creates a persistent source of ownership confusion.

## Scale of the SFPU Layer

**Location:** [`hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu)

The directory contains **158 header files** totaling **10,123 lines** (Wormhole B0). These break down into two categories:

### Low-Level SFPU Kernels (`ckernel_sfpu_*.h`) -- 90 files

These implement the actual SFPU math operations. Examples:

- `ckernel_sfpu_abs.h`
- `ckernel_sfpu_exp.h`
- `ckernel_sfpu_gelu.h`
- `ckernel_sfpu_binary.h`
- `ckernel_sfpu_topk.h`
- `ckernel_sfpu_cumsum.h`
- `ckernel_sfpu_typecast.h`

Each file contains the inner-loop SFPU register manipulation for one operation, using primitives from the LLK `ckernel` namespace (e.g., `sfpu_load`, `sfpu_store`, dest register access patterns).

### LLK-Style Wrappers (`llk_math_eltwise_*_sfpu_*.h`) -- 68 files

These wrap the `ckernel_sfpu_*.h` implementations in the standard LLK calling convention. Examples:

- `llk_math_eltwise_unary_sfpu_abs.h`
- `llk_math_eltwise_unary_sfpu_exp2.h`
- `llk_math_eltwise_binary_sfpu_shift.h`
- `llk_math_eltwise_ternary_sfpu_lerp.h`
- `llk_math_ema_sfpu_entry.h`

These files follow the same `llk_*` naming convention as Layer 2 wrappers, calling into `_llk_*` functions defined in tt-llk's [`llk_lib/llk_math_eltwise_unary_sfpu.h`](https://github.com/tenstorrent/tt-llk/blob/main/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h) and related headers.

## The Ownership Problem

The SFPU layer sits in a unique position:

| Aspect | Owner |
|--------|-------|
| File location | Metal (`hw/ckernels/<arch>/metal/llk_api/llk_sfpu/`) |
| Naming convention | LLK (`llk_*` and `ckernel_*` prefixes) |
| Infrastructure used | LLK (SFPU register access, dest register management, MOP config) |
| New op authoring | Metal team (no corresponding directory in tt-llk) |

This creates ambiguity in several practical scenarios:

1. **Where should a new SFPU op be authored?** The `ckernel_sfpu_*.h` files live in Metal, but they depend on LLK's SFPU infrastructure. If LLK changes its SFPU register access patterns, all 90 `ckernel_sfpu_*.h` files may need updating -- but they are not in the LLK repo.

2. **Who reviews SFPU changes?** The code uses `llk_` prefixes and LLK calling conventions, suggesting LLK ownership, but the files are in Metal's tree. A developer unfamiliar with the history would reasonably assume these are LLK-owned.

3. **Architecture portability.** The SFPU directory is duplicated per architecture (Wormhole B0, Blackhole, Quasar). When a new architecture is added, all 158 files need to be evaluated for portability -- but since they are in Metal, not LLK, the architecture bring-up team must coordinate across repo boundaries.

## LLK's Own SFPU Infrastructure

For comparison, tt-llk's [`llk_lib/`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_wormhole_b0/llk_lib) contains the generic SFPU dispatch infrastructure:

- `llk_math_eltwise_unary_sfpu.h` -- the `_llk_math_eltwise_unary_sfpu_*` functions that configure SFPU execution mode, dest register targeting, and face iteration.
- `llk_math_eltwise_unary_sfpu_params.h` -- parameter structures for SFPU dispatch.
- `llk_math_eltwise_binary_sfpu.h` and `llk_math_eltwise_ternary_sfpu.h` -- binary and ternary SFPU dispatch.

These LLK-owned files define the *framework* for SFPU execution, but the actual *operations* (the 90 `ckernel_sfpu_*.h` files) are authored in Metal. The framework/operation split crosses the repo boundary.

## The `experimental/` Directories

Both repos contain `experimental/` directories with unstable APIs:

**tt-llk:** [`tt_llk_wormhole_b0/llk_lib/experimental/`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_wormhole_b0/llk_lib/experimental)
- `llk_math_reduce_custom.h`
- `llk_math_reduce_runtime_custom.h`
- `llk_unpack_AB_reduce_custom.h`
- `llk_unpack_AB_reduce_custom_runtime.h`

**Metal:** [`hw/inc/api/compute/experimental/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/api/compute/experimental)
- `matmul_custom.h`
- `mul_reduce_scalar.h`
- `pack_custom.h`
- `sdpa_sub_custom.h`

**Issues with the experimental directories:**

- There is no documented promotion path. No file in either `experimental/` directory indicates when or how an API graduates to the stable set.
- The LLK experimental files (reduce_custom) have corresponding Metal experimental files, but neither repo references the other's experimental status.
- If a Metal experimental API depends on an LLK experimental API, both must be promoted together, but there is no mechanism to enforce this coordination.
- The naming convention `_custom` / `_runtime_custom` does not follow the standard `_llk_` / `llk_` layering, further blurring the boundary.

## Summary

The SFPU layer represents the largest single area of boundary ambiguity between LLK and Metal. At 10,123 lines across 158 files (per architecture), it is larger than the Layer 2 LLK API wrappers (2,187 lines) and comparable in size to the Layer 3 compute API (10,505 lines). Its ownership is effectively split: Metal authors the operations, LLK provides the execution framework, and neither repo has clear authority over the combined surface area.

---

**Next:** [Chapter 5 -- Per-Architecture Code Duplication](../ch5_architecture_duplication/index.md)
