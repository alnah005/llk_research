# Chapter 6: Debugging Tools for LLK Kernel Development

Debugging kernels that run on Tenstorrent's Tensix cores is fundamentally different from debugging host-side software. There is no interactive debugger, no standard output stream, and no operating system to catch segmentation faults. When something goes wrong inside a compute kernel, the typical symptom is a silent hang or incorrect numerical output -- neither of which points directly to a root cause.

This chapter evaluates the debugging tools that are currently available for LLK kernel development, assesses their effectiveness, and identifies the gaps that remain. The goal is not merely to catalog what exists, but to give kernel developers a realistic picture of what they can and cannot diagnose with today's infrastructure.

## Tools covered

| Tool | Section | Purpose |
|------|---------|---------|
| `LLK_ASSERT(condition, message)` | [available_tools.md](./available_tools.md#llk_assert) | Runtime and compile-time assertion inside LLK code |
| `fake_kernels_target` | [available_tools.md](./available_tools.md#fake_kernels_target) | CMake target for build-time syntax and type checking of kernels |
| Generated file inspection | [available_tools.md](./available_tools.md#generated-file-inspection) | Examining `chlkc_math.cpp`, `defines_generated.h`, `chlkc_descriptors.h` |
| Watcher infrastructure | [available_tools.md](./available_tools.md#watcher-infrastructure) | Host-side polling for waypoints, asserts, NOC sanitization, hang detection |
| `DPRINT` | [available_tools.md](./available_tools.md#dprint) | Device-side printf for data movement and compute kernels |

## Gaps and limitations

Each tool above has significant constraints that affect day-to-day kernel development. These are analyzed in detail in [`gaps_and_limitations.md`](./gaps_and_limitations.md), covering topics such as:

- Assertions that are off by default and opt-in per build
- The inability of the fake kernels target to validate runtime correctness
- Hang diagnosis that identifies *where* a stall occurs but not *why*
- Error messages that reference generated files instead of original source
- The absence of single-step debugging for TRISC processors

Understanding both the tools and their limitations is essential for anyone writing or maintaining LLK kernels in the TT-Metal stack.
