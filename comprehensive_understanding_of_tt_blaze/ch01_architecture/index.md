# Chapter 1 -- Architecture and Position in the Stack

This chapter establishes where TT-Blaze sits in the Tenstorrent software stack, explains the core design decisions that motivated its creation, and provides a structural map of the entire repository.

## Key Takeaways

- Blaze is a **composition and dispatch layer** that sits above TT-Metal's program descriptor infrastructure and below model-level code. It is a peer to TT-Forge, but targets handwritten C++ kernels rather than compiler-generated ones.
- The project depends on a pinned `tt-metal` submodule on the `blaze-metal` branch, which provides C++20 kernel compilation and named compile-time / runtime argument infrastructure not yet on TT-Metal main.
- Blaze provides two complementary APIs -- a declarative **graph API** for high-level fusion specification, and an imperative **composition API** for low-level kernel wiring -- both of which produce the same internal representation (`BlazeGraph`).
- The op library is auto-discovered: adding a new `BlazeOp` subclass in `blaze/ops/<name>/op.py` is sufficient to make it available as `blaze.<name>(...)` (graph API) and `blaze.<name>.emit(...)` (composition API).

## Reading Order

1. [**Stack Position**](./01_stack_position.md) -- Where Blaze sits relative to TT-Metal, TT-LLK, and TT-Forge; the `blaze-metal` submodule contract; and why Blaze exists.
2. [**Repository Map**](./02_repository_map.md) -- Top-level directory layout, the `blaze/` package internals, build system, and key infrastructure modules.
3. [**The Two-API Design**](./03_two_api_design.md) -- The graph API (`blaze.fuse()`), the composition API (`FusedProgram` + `emit()`), how they converge on a `BlazeGraph`, and when to use each.

## Files in This Chapter

| File | Description |
|------|-------------|
| [`01_stack_position.md`](./01_stack_position.md) | Stack placement, relationships to Metal/LLK/Forge, and project motivation |
| [`02_repository_map.md`](./02_repository_map.md) | Directory structure, build system, and key infrastructure modules |
| [`03_two_api_design.md`](./03_two_api_design.md) | Graph API, composition API, their convergence on BlazeGraph, and usage guidance |
