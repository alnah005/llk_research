# Chapter 7: Cross-Boundary Testing Strategy

## Summary

LLK and Metal each maintain independent test suites that validate compute operations at
different layers of the stack. LLK tests exercise the Layer 1 `_llk_*` functions directly
on hardware (or emulator) via its own Python/C++ harness and
[tt-exalens](https://github.com/tenstorrent/tt-exalens). Metal tests exercise operations
through the Layer 3 TTNN API, which calls down through the Layer 2 wrappers and ultimately
into the LLK submodule. Because the two suites were designed independently, there is no
shared contract that ensures they agree on parameter ranges, correctness thresholds, or
performance baselines -- creating blind spots at the integration boundary.

This chapter surveys how each repo tests compute operations, then catalogues the gaps that
arise from the submodule integration model.

## Sub-pages

| File | Topic |
|------|-------|
| [`llk_standalone_testing.md`](./llk_standalone_testing.md) | LLK's own test infrastructure: Python harness, C++ test sources, hardware-specific tests |
| [`metal_integration_testing.md`](./metal_integration_testing.md) | How Metal tests compute kernels through the full three-layer stack |
| [`testing_gaps.md`](./testing_gaps.md) | Five identified gaps at the cross-boundary testing interface |

## Key Observations

1. **LLK tests call Layer 1 directly.** C++ test sources such as
   [`tests/sources/eltwise_binary_test.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/eltwise_binary_test.cpp)
   invoke `_llk_unpack_AB_init_`, `_llk_unpack_hw_configure_`, and other `_llk_*` functions
   without any Metal wrapper in the call path. This validates LLK in isolation but says
   nothing about whether Metal's Layer 2 wrapper passes the correct arguments.

2. **Metal tests call Layer 3.** Compute test kernels like
   [`tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block.cpp)
   use the `copy_tile` / `pack_tile` Layer 3 API. TTNN-level tests in
   [`tests/ttnn/unit_tests/operations/eltwise/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/unit_tests/operations/eltwise)
   go even higher. Neither suite isolates the Layer 1-to-Layer 2 boundary.

3. **No submodule-update gate exists.** When the `tt_llk` submodule pointer is bumped in
   Metal, no CI job re-runs a targeted subset of Metal compute tests to verify that the
   new LLK commit is compatible. The
   [`fake_kernels_target/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/jit_build/fake_kernels_target)
   build catches compile errors but not runtime regressions.

4. **SFPU operations authored in Metal are invisible to LLK tests.** Metal's
   [`hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu)
   directory contains 158 SFPU header files (e.g., `ckernel_sfpu_silu.h`,
   `ckernel_sfpu_cumsum.h`). These are never compiled or tested by LLK's test suite.
