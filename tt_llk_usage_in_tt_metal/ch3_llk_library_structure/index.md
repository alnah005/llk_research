# Chapter 3: LLK Library Structure

This chapter describes the internal organization of the TT-LLK repository and how its API surface is laid out across hardware targets. Understanding this structure is essential for navigating the codebase, locating the correct implementation for a given architecture, and comprehending the division of responsibility between TT-LLK (the submodule) and TT-Metal (the consumer).

TT-LLK lives inside tt-metal as a Git submodule at `tt_metal/third_party/tt_llk/`. It provides per-architecture directories for Blackhole, Wormhole B0, and Quasar, each containing hardware-specific LLK function implementations, ckernel infrastructure headers, SFPU intrinsic wrappers, and low-level instruction encodings. A small shared `common/` directory at the repo root supplies cross-architecture utilities.

TT-Metal then layers its own `llk_api/` wrapper files on top of these raw LLK implementations. The wrappers translate high-level operand identifiers into the low-level parameters that TT-LLK functions expect. This two-tier design -- internal `_llk_*()` functions in TT-LLK, public `llk_*()` wrappers in TT-Metal -- is the central organizational principle of the library.

## In this chapter

| File | Description |
|------|-------------|
| [`directory_layout.md`](./directory_layout.md) | Physical directory structure of the TT-LLK repository, per-architecture directories, and shared headers. |
| [`api_naming_conventions.md`](./api_naming_conventions.md) | Two-tier naming scheme (`_llk_*` vs `llk_*`), function categories (math, unpack, pack), and key operation names. |
| [`arch_differences.md`](./arch_differences.md) | How Blackhole, Wormhole B0, and Quasar differ in file naming, API coverage, and SFPU support. |
