# LLK Standalone Testing

## Overview

TT-LLK maintains a self-contained test infrastructure under
[`tests/`](https://github.com/tenstorrent/tt-llk/tree/main/tests) that validates Layer 1
`_llk_*` functions on real hardware or an emulator -- entirely independent of Metal's build
system, runtime, or APIs. The infrastructure consists of three main components: a Python
test harness driven by pytest, C++ test source files that call LLK functions directly, and
hardware-specific configuration headers downloaded per architecture.

## Python Test Harness

The entry point is
[`tests/python_tests/conftest.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/conftest.py),
which initializes the `LLK_HOME` environment variable, starts a
[tt-exalens](https://github.com/tenstorrent/tt-exalens) server for hardware interaction,
and provides pytest fixtures for device setup. Key imports include:

- `helpers.exalens_server.ExalensServer` -- manages the tt-exalens subprocess lifecycle
  ([`tests/python_tests/helpers/exalens_server.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/helpers/exalens_server.py))
- `helpers.device` -- provides low-level device read/write via `ttexalens.tt_exalens_lib`
  functions such as `read_from_device`, `write_to_device`, `load_elf`, and `parse_elf`
  ([`tests/python_tests/helpers/device.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/helpers/device.py))
- `helpers.test_config.TestConfig` -- orchestrates test execution including ELF loading,
  golden comparison, and coverage reporting
  ([`tests/python_tests/helpers/test_config.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/helpers/test_config.py))

The helper module directory at
[`tests/python_tests/helpers/`](https://github.com/tenstorrent/tt-llk/tree/main/tests/python_tests/helpers)
contains over 40 Python modules covering golden generation, stimulus creation, data format
handling, performance measurement, and tilize/untilize utilities.

Functional test files follow the naming convention `test_*.py` and include tests such as:

- [`test_zzz_eltwise_binary.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/test_zzz_eltwise_binary.py) --
  sweeps elementwise binary operations across data formats, tile dimensions, broadcast
  types, and math fidelity settings
- [`test_zzz_eltwise_unary_sfpu.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/test_zzz_eltwise_unary_sfpu.py) --
  validates unary SFPU operations
- [`test_unpack_matmul.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/test_unpack_matmul.py) --
  tests the unpack-to-matmul pipeline

## C++ Test Sources

The C++ test source files under
[`tests/sources/`](https://github.com/tenstorrent/tt-llk/tree/main/tests/sources)
contain the actual kernel code that runs on the Tensix cores. Each file is structured
around the three TRISC processors (UNPACK, MATH, PACK), guarded by `#ifdef` blocks:

```cpp
// From tests/sources/eltwise_binary_test.cpp
#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_AB.h"
void run_kernel(RUNTIME_PARAMETERS params) {
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(...);
    _llk_unpack_AB_init_<BROADCAST_TYPE>(...);
    // ...
}
#endif
```

These tests call `_llk_*` functions (Layer 1) directly. There is no `kernel_main()`
entry point, no circular buffer API, and no `copy_tile` / `pack_tile` abstraction --
those are all Layer 2/3 concepts from Metal. The test sources include:

- **Functional tests** (`*_test.cpp`): Correctness validation, e.g.,
  [`eltwise_binary_test.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/eltwise_binary_test.cpp),
  [`matmul_test.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/matmul_test.cpp),
  [`pack_untilize_test.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/pack_untilize_test.cpp)
- **Performance tests** (`*_perf.cpp`): Cycle-count measurement, e.g.,
  [`matmul_perf.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/matmul_perf.cpp),
  [`eltwise_binary_fpu_perf.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/eltwise_binary_fpu_perf.cpp),
  [`eltwise_unary_sfpu_perf.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/eltwise_unary_sfpu_perf.cpp)
- **Architecture-specific tests** (under `sources/quasar/`): e.g.,
  [`eltwise_binary_test.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/quasar/eltwise_binary_test.cpp),
  [`isolate_sfpu_square_quasar_test.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sources/quasar/isolate_sfpu_square_quasar_test.cpp)

## Hardware-Specific Configuration

The
[`tests/hw_specific/`](https://github.com/tenstorrent/tt-llk/tree/main/tests/hw_specific)
directory holds per-architecture header files (e.g., `cfg_defines.h`, `tensix.h`,
`tensix_types.h`) that are downloaded at test setup time. The
[`setup_testing_env.sh`](https://github.com/tenstorrent/tt-llk/blob/main/tests/setup_testing_env.sh)
script detects the chip architecture via `tt-smi` and fetches the appropriate headers.

## Build Toolchain

LLK tests use the SFPI compiler toolchain to build test kernels. The script
[`tests/sfpi-info.sh`](https://github.com/tenstorrent/tt-llk/blob/main/tests/sfpi-info.sh)
determines the SFPI release version and architecture. Linker scripts under
[`tests/helpers/ld/`](https://github.com/tenstorrent/tt-llk/tree/main/tests/helpers/ld)
(e.g., `math.ld`, `unpack.ld`, `pack.ld`, `sfpu.ld`) define the memory layout for each
TRISC core. Boot and initialization code lives in
[`tests/helpers/src/brisc.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/helpers/src/brisc.cpp)
and
[`tests/helpers/src/trisc.cpp`](https://github.com/tenstorrent/tt-llk/blob/main/tests/helpers/src/trisc.cpp).

This entire toolchain is independent of Metal's CMake build system. LLK tests can be run
on a machine that has never built or installed TT-Metal.

## Performance Testing

Performance test files (prefixed `perf_`) under
[`tests/python_tests/`](https://github.com/tenstorrent/tt-llk/tree/main/tests/python_tests)
measure cycle counts for individual operations in isolation:

- [`perf_matmul.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/perf_matmul.py)
- [`perf_eltwise_binary_fpu.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/perf_eltwise_binary_fpu.py)
- [`perf_eltwise_unary_sfpu.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/perf_eltwise_unary_sfpu.py)
- [`perf_reduce.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/perf_reduce.py)
- [`perf_unpack_tilize.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/perf_unpack_tilize.py)

These use the `PerfConfig` and `PerfReport` helpers from
[`tests/python_tests/helpers/perf.py`](https://github.com/tenstorrent/tt-llk/blob/main/tests/python_tests/helpers/perf.py)
to capture hardware performance counters. The results validate LLK in isolation but are
never compared against Metal's end-to-end performance numbers.

---

**Next:** [`metal_integration_testing.md`](./metal_integration_testing.md)
