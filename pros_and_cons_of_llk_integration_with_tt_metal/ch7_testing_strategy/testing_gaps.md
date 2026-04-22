# Testing Gaps at the Integration Boundary

## Overview

The combination of LLK's standalone test suite and Metal's end-to-end test suite leaves
five distinct gaps at the integration boundary. Each gap represents a class of defect that
could be introduced by a submodule update and go undetected until it surfaces in production
workloads.

## Gap 1: No Contract Tests Between Layer 1 and Layer 2

LLK tests call `_llk_*` functions directly from C++ test sources like
[`tests/sources/eltwise_binary_test.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/eltwise_binary_test.cpp),
which invokes `_llk_unpack_hw_configure_<>()` and `_llk_unpack_AB_init_<>()` with
specific template arguments and parameter orderings. Metal's Layer 2 wrappers (e.g., the
`llk_math_eltwise_binary` wrapper in
[`llk_math_binary_api.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h))
also call these same `_llk_*` functions, but with their own argument-passing conventions.

There is no shared contract test that verifies the Layer 2 wrapper passes arguments in the
exact format that Layer 1 expects. If LLK changes the semantics of a parameter (e.g.,
reordering arguments, changing the meaning of a template parameter value), LLK's own tests
would be updated accordingly, but Metal's Layer 2 wrappers would silently break. The
[`fake_kernels_target`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/jit_build/fake_kernels_target)
catches signature changes but not semantic changes.

## Gap 2: Parameter Space Divergence

LLK and Metal test the same operations but may sweep entirely different parameter ranges.
For example, LLK's
[`test_zzz_eltwise_binary.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/test_zzz_eltwise_binary.py)
parametrizes over data formats (`Bfp8_b`, `Float16_b`, `Float32`, etc.), tile dimensions
(from the `SUPPORTED_TILE_SIZES` set), broadcast types, and math fidelity levels -- using
the parameter definitions in
[`helpers/test_variant_parameters.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/helpers/test_variant_parameters.py).

Metal's TTNN-level eltwise tests like
[`test_unary.py`](https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/unit_tests/operations/eltwise/test_unary.py)
parametrize over tensor shapes and compare against PyTorch golden outputs, but they
operate at a higher abstraction level where tile dimensions and math fidelity are
implicit. A configuration that LLK tests but Metal does not (or vice versa) creates a
coverage gap where regressions in one parameter combination are invisible.

## Gap 3: No Automated Compatibility Check on Submodule Update

When a developer updates the `tt_llk` submodule pointer in Metal, there is no dedicated
CI gate that:

1. Runs LLK's own test suite against the new commit to confirm standalone correctness.
2. Runs a targeted subset of Metal compute tests (e.g., the tests under
   [`tests/tt_metal/tt_metal/test_kernels/compute/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/tt_metal/tt_metal/test_kernels/compute))
   to confirm integration correctness.
3. Compares the results of (1) and (2) to ensure agreement.

The
[`fake_kernels_target/CMakeLists.txt`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/fake_kernels_target/CMakeLists.txt)
compiles all kernel files with LLK include paths, catching compile-time breakage. But it
uses hardcoded constants from
[`fake_jit_prelude.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/fake_kernels_target/fake_jit_prelude.h)
(e.g., `MATH_FIDELITY = 255`, all format arrays set to `255`), which means it tests only
one artificial configuration. Runtime regressions, performance changes, and bugs that
manifest under specific format/fidelity combinations are not caught.

## Gap 4: Metal-Authored SFPU Operations are Untested by LLK

Metal defines 158 SFPU operation headers in
[`hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu),
including operations like `ckernel_sfpu_silu.h`, `ckernel_sfpu_cumsum.h`,
`ckernel_sfpu_dropout.h`, `ckernel_sfpu_elu.h`, and many more. These files live in Metal's
repository, not in the LLK submodule.

LLK's test suite under
[`tests/sources/`](https://github.com/tenstorrent/tt-llk/tree/main/tests/sources)
tests SFPU operations that are defined within LLK itself (e.g.,
[`eltwise_unary_sfpu_test.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/eltwise_unary_sfpu_test.cpp)),
but it has no visibility into Metal-authored SFPU operations. If an LLK change modifies
the SFPU infrastructure (e.g., register allocation, convergence behavior in
`ckernel_sfpu.h`), the Metal-authored SFPU operations could break without any LLK test
detecting it.

Metal's own test for SFPU operations,
[`eltwise_sfpi.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpi.cpp),
uses macro injection (`SFPI_OP_AND_PACK`) to test individual operations, but coverage
depends on which operations the host-side test driver configures. There is no systematic
sweep that covers all 158 SFPU operations after an LLK submodule update.

## Gap 5: No Cross-Boundary Performance Regression Testing

LLK maintains its own performance benchmarks through `perf_*.py` files such as
[`perf_matmul.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/perf_matmul.py),
[`perf_eltwise_binary_fpu.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/perf_eltwise_binary_fpu.py),
and
[`perf_reduce.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/perf_reduce.py),
which measure cycle counts using hardware performance counters via the
[`helpers/perf.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/helpers/perf.py)
module.

Metal has its own performance and benchmark suites under
[`tests/ttnn/perf_tests/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/perf_tests) and
[`tests/ttnn/benchmark/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/benchmark).

However, there is no mechanism to:

1. Compare LLK's isolated cycle counts against the cycle counts observed when the same
   operation runs through Metal's full stack.
2. Detect performance regressions that appear only when LLK functions are called through
   the Layer 2 wrappers (e.g., due to additional overhead from wrapper logic or different
   compiler optimization behavior when LLK headers are included in a larger translation
   unit).
3. Track performance trends across submodule updates to establish whether a given LLK
   bump improves, degrades, or is neutral to Metal's end-to-end throughput.

---

**Next:** [Chapter 8 -- Alternative Integration Patterns](../ch8_alternative_patterns/index.md)
