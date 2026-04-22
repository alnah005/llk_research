# Chapter 7: The SFPU Extension Path

Adding a new SFPU (Special Function Processing Unit) operation to TT-Metal is one of the most common tasks a kernel developer faces. It is also one of the most revealing tests of the LLK architecture: the process touches both the TT-LLK library and the TT-Metal framework, crosses repository boundaries, and requires coordinating changes across at least six distinct locations.

This chapter walks through the end-to-end experience of adding a new SFPU operation, then catalogs the friction points that make the process harder than it needs to be.

## The Six-Step Path

Adding a new SFPU op requires modifications in a strict dependency order. Each step feeds into the next:

| Step | Layer | Repository | Key File(s) |
|------|-------|------------|-------------|
| 1 | SFPU math kernel | TT-LLK | `tt_llk_<arch>/common/inc/sfpu/ckernel_sfpu_<op>.h` |
| 2 | Metal SFPU wrapper | TT-Metal | `hw/ckernels/<arch>/metal/llk_api/llk_sfpu/` |
| 3 | Compute API header | TT-Metal | `hw/inc/api/compute/eltwise_unary/<op>.h` |
| 4 | `sfpu_split_includes` wiring | TT-Metal | `hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` |
| 5 | Compute kernel | TT-Metal | `tt_metal/kernels/compute/eltwise_sfpu.cpp` (via `SFPU_OP_CHAIN_0`) |
| 6 | Program factory + host defines | TT-Metal | `ttnn/cpp/.../unary/common/unary_op_utils.cpp` |

The chain runs from bare SFPI vector intrinsics at the bottom to Python-callable TTNN operations at the top. Missing any link in the chain results in either a compile error or a silently absent operation.

## Chapter Contents

- [**Step-by-Step Walkthrough**](./step_by_step_walkthrough.md) -- A concrete trace through all six steps using real file paths and code from the `abs` and `activations` operations as reference examples.
- [**Friction Points**](./friction_points.md) -- An analysis of the six major pain points in this process: per-architecture duplication, the `sfpu_split_includes.h` bottleneck, macro-based registration, header sprawl, lack of a lightweight test harness, and `SFPU_OP_CHAIN_0` code injection.
