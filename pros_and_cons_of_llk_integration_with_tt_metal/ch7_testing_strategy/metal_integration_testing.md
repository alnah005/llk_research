# Metal Integration Testing

## Overview

Metal tests compute operations through the full three-layer stack: Layer 3 (user-facing
API) calls Layer 2 (Metal wrappers), which calls Layer 1 (LLK `_llk_*` functions). Unlike
LLK's standalone tests, Metal's test suite never isolates the Layer 1-to-Layer 2 boundary.
This means that when a Metal compute test passes, it confirms that the entire stack works
end-to-end for that particular configuration, but it cannot pinpoint whether a failure
originates in LLK, the Metal wrapper, or the host-side dispatch logic.

## Test Organization

Metal's test hierarchy is spread across several directories:

- **Low-level Metal tests** at
  [`tests/tt_metal/tt_metal/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/tt_metal/tt_metal) --
  C++ test programs that use the `tt_metal` host API directly. Examples include
  [`test_untilize_eltwise_binary.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_untilize_eltwise_binary.cpp),
  [`test_matmul_large_block.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_matmul_large_block.cpp), and
  [`test_transpose_hc.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_transpose_hc.cpp).

- **Compute test kernels** at
  [`tests/tt_metal/tt_metal/test_kernels/compute/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/tt_metal/tt_metal/test_kernels/compute) --
  device-side kernel code that uses the Layer 3 API. Files like
  [`eltwise_copy_block.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block.cpp)
  call `copy_tile()`, `pack_tile()`, `acquire_dst()`, and `release_dst()` -- all Layer 3
  constructs that eventually dispatch to LLK through the Layer 2 wrappers.

- **TTNN unit tests** at
  [`tests/ttnn/unit_tests/operations/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/unit_tests/operations) --
  Python tests that exercise the highest-level API. Subdirectories cover
  [`eltwise/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/unit_tests/operations/eltwise),
  [`matmul/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/unit_tests/operations/matmul),
  [`reduce/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/unit_tests/operations/reduce),
  and others. For example,
  [`test_unary.py`](https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/unit_tests/operations/eltwise/test_unary.py)
  tests unary operations via `ttnn` function calls, comparing results against PyTorch
  golden outputs.

- **Nightly and performance suites** at
  [`tests/ttnn/nightly/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/nightly),
  [`tests/ttnn/perf_tests/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/perf_tests), and
  [`tests/ttnn/sweep_tests/`](https://github.com/tenstorrent/tt-metal/tree/main/tests/ttnn/sweep_tests).

## How Compute Kernels are Tested

A typical Metal compute test kernel uses the Layer 3 API exclusively. For example, in
[`eltwise_copy_block.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_block.cpp):

```cpp
void kernel_main() {
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    // ...
    for (uint32_t block = 0; block < num_blocks; ++block) {
        acquire_dst();
        cb0.wait_front(block_num_tiles);
        for (uint32_t t = 0; t < block_num_tiles; ++t) {
            copy_tile(tt::CBIndex::c_0, t, t);
        }
        // ...
    }
}
```

Similarly,
[`eltwise_sfpi.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tests/tt_metal/tt_metal/test_kernels/compute/eltwise_sfpi.cpp)
uses macro-injected SFPU operations (`SFPI_OP_AND_PACK`) defined via `add_define` in the
test's host-side configuration. The `SFPI_OP_AND_PACK` macro expands to calls like `exp_tile`,
`gelu_tile`, or `recip_tile` -- Layer 3 functions that map through Layer 2 into the LLK.

None of these test kernels call `_llk_*` functions directly. The Layer 2 wrapper is always
in the call path, meaning these tests validate the combined behavior of all three layers.

## The `fake_kernels_target` Compile-Time Check

The directory
[`tt_metal/jit_build/fake_kernels_target/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/jit_build/fake_kernels_target)
provides a compile-time validation mechanism. Its
[`CMakeLists.txt`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/fake_kernels_target/CMakeLists.txt)
globs all kernel `.cpp` files across `tt_metal/kernels/`, `tt_metal/impl/dispatch/kernels/`,
`tt_metal/fabric/impl/kernels/`, and `ttnn/cpp/ttnn/operations/*/kernels/` (as well as `*/kernels_ng/`), then compiles
them into a single `jit_kernels_index` object library. The include paths explicitly
reference the LLK submodule:

```cmake
"${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/common"
"${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_NAME_LOWER}/common/inc"
"${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_NAME_LOWER}/llk_lib"
```

A companion header,
[`fake_jit_prelude.h`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/fake_kernels_target/fake_jit_prelude.h),
supplies hardcoded constants (e.g., `DST_ACCUM_MODE`, `MATH_FIDELITY`, pack/unpack format
arrays) that would normally come from the generated `chlkc_descriptors.h`. This allows
the build to succeed without a real JIT context.

**What this catches:** Compile errors from API signature changes, missing headers, or type
mismatches after an LLK submodule update.

**What this misses:** Runtime correctness, performance regressions, and any bug that
manifests only with specific parameter combinations (since the fake prelude uses a single
hardcoded configuration with all format values set to `255` and `MATH_FIDELITY = 255`).

## No Targeted Submodule-Update Tests

There is no CI workflow in Metal that specifically runs when the `tt_llk` submodule pointer
changes. The workflows found under the submodule's own
[`.github/workflows/`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/third_party/tt_llk/.github/workflows)
(e.g., `check-stale.yml`, `collect-test-durations.yml`) are LLK-internal CI jobs, not
Metal integration checks. When a developer bumps the submodule, the standard Metal CI
pipeline runs, but it executes the full test suite without any prioritization of compute
kernel tests that are most likely to be affected by LLK changes.

---

**Next:** [`testing_gaps.md`](./testing_gaps.md)
