# Chapter 4: API Abstraction Boundary Analysis

This chapter evaluates the quality of the abstraction boundary between LLK and Metal. The compute kernel stack is organized into three API layers, each with a distinct intended responsibility. In practice, several of these boundaries leak implementation details upward, creating coupling that complicates maintenance, onboarding, and architecture portability.

## The Three API Layers

| Layer | Location | Prefix / Convention | Intended Role |
|-------|----------|---------------------|---------------|
| 1 -- LLK Core | [`tt_llk_<arch>/llk_lib/`](https://github.com/tenstorrent/tt-llk/tree/main/tt_llk_wormhole_b0/llk_lib) | `_llk_*` (leading underscore) | Hardware register manipulation, address mode config, low-level tile ops |
| 2 -- Metal LLK API | [`hw/ckernels/<arch>/metal/llk_api/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api) | `llk_*` (no leading underscore) | Wraps Layer 1 with Metal-specific defaults and operand resolution |
| 3 -- Compute Kernel API | [`hw/inc/api/compute/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/inc/api/compute) | User-facing names (`binary_op_init_common`, `pack_tile`, `tile_regs_acquire`) | Public API for op kernel authors |

## Sections

- [**Three-Layer API Architecture**](./three_layer_api.md) -- Detailed walkthrough of each layer, with code examples showing the call chain from Layer 3 down to Layer 1.
- [**Leaky Abstractions**](./leaky_abstractions.md) -- Identifies places where hardware details, compile-time globals, and raw data formats pierce through the API surface.
- [**SFPU Boundary Ambiguity**](./sfpu_boundary.md) -- Examines the 158-file SFPU kernel layer that lives in Metal but depends on LLK infrastructure, and the unclear ownership it creates.
