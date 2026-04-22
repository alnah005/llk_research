# Chapter 2: Compute Kernel API -- The Abstraction Layer Over LLK

## The Four-Layer Stack

When a developer writes a compute kernel for TT-Metal, they interact with a stack of four distinct layers:

| Layer | Location | Role |
|-------|----------|------|
| **Compute Kernel API** | `tt_metal/hw/inc/api/compute/*.h` | User-facing functions: `add_tiles()`, `mul_tiles()`, `pack_tile()`, etc. |
| **LLK API** | `tt-llk/.../llk_*.h` | Per-pipeline orchestration: `llk_math_eltwise_binary()`, `llk_unpack_AB()`, `llk_pack()` |
| **LLK Implementation** | `tt-llk/.../ckernel_*.h` | Register-level operations: writes to UNPACK/MATH/PACK config registers |
| **Hardware** | Tensix RISC cores | Three TRISC threads (UNPACK, MATH, PACK) executing concurrently |

The Compute Kernel API is the boundary that most kernel authors are supposed to stay within. Its purpose is to collapse the complexity of coordinating three hardware pipelines into single, domain-meaningful function calls. This chapter evaluates how well it achieves that goal -- and where it fails.

## What This Chapter Covers

- [**Abstraction Strengths**](./abstraction_strengths.md) -- Where the API genuinely simplifies kernel development: descriptive naming, pipeline orchestration in single calls, the `state_configure()` guard, and Doxygen-style documentation.

- [**Abstraction Leaks**](./abstraction_leaks.md) -- Where the abstraction breaks down and developers must reason about hardware details the API was supposed to hide: manual `reconfig_data_format()` calls, raw CB index management, fully manual DST register protocols, preprocessor-injected template parameters, and `SFPU_OP_CHAIN` code injection.

## The Central Question

The Compute Kernel API sits at a design tension point. Make it too thin, and every kernel author must understand UNPACK/MATH/PACK pipeline synchronization. Make it too thick, and you lose the performance control that justifies writing hardware kernels in the first place. The question is not whether the API *should* abstract everything -- it is whether the current boundary between "abstracted" and "exposed" is drawn at the right place.

As we will see, the API does an excellent job on the "happy path" (simple binary ops, straightforward reductions), but the abstraction degrades significantly for production kernels like layernorm, groupnorm, and SDPA, where the developer ends up managing hardware state that the API was ostensibly designed to hide.
