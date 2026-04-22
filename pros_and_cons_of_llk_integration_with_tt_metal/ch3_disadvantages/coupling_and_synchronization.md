# Coupling and Synchronization

## Submodule Update Friction

LLK is integrated into Metal as a Git submodule, as declared in `[tt-metal] .gitmodules`:

```
[submodule "tt_metal/third_party/tt_llk"]
    path = tt_metal/third_party/tt_llk
    url = https://github.com/tenstorrent/tt-llk.git
```

Bumping the LLK version in Metal requires updating the pinned submodule commit in `tt_metal/third_party/tt_llk`. This creates several friction points:

- **Branch conflicts**: Two Metal PRs that each bump LLK to different commits will conflict on the submodule pointer, even if the LLK changes themselves are independent. Standard Git merge cannot resolve submodule pointer conflicts automatically.
- **Atomic cross-repo changes**: When a Metal change requires a corresponding LLK change (e.g., adding a new parameter to an `_llk_*` function), the developer must land the LLK PR first, then update the submodule pointer in Metal. This serializes work that should be parallel.
- **CI bisection difficulty**: A single Metal commit that bumps the LLK submodule can pull in dozens of LLK commits, making `git bisect` less effective for isolating regressions.

## Implicit API Contracts: The `_llk_*` Pattern

Metal's `llk_api/` wrapper headers call into LLK's internal `_llk_*`-prefixed functions. For example, `[tt-metal] tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h` contains calls such as:

```cpp
_llk_pack_mop_config_<untilize, zero_output>(...);
_llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(...);
_llk_pack_init_<untilize, zero_output>(...);
_llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, untilize>(tile_index, pack_tile_addr);
_llk_pack_untilize_init_<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(...);
```

These `_llk_*` functions are defined in `[tt-llk]` headers like `tt_llk_wormhole_b0/llk_lib/llk_pack_common.h`. The leading underscore conventionally signals "internal/private," yet Metal depends on their exact signatures, template parameters, and semantics. There is no formal API contract -- no version number, no stability guarantee, no deprecation process. Any signature change in LLK immediately breaks Metal's wrapper layer.

This is not hypothetical: the `llk_api/` directory contains 183 files across Wormhole and Blackhole that each directly invoke `_llk_*` functions, making the coupling surface area enormous.

## The `#ifdef TRISC_*` Conditional Compilation Pattern

Compute kernels are compiled four times -- once for each TRISC processor (unpack, math, pack, isolate_sfpu). The fourth compilation targets the Quasar-specific `isolate_sfpu` path. The selection is controlled by preprocessor defines injected by the build system. In `[tt-metal] tt_metal/jit_build/genfiles.cpp` (lines 196-199):

```cpp
const string unpack_prolog = build_trisc_prolog("TRISC_UNPACK");
const string math_prolog = build_trisc_prolog("TRISC_MATH");
const string pack_prolog = build_trisc_prolog("TRISC_PACK");
const string isolate_sfpu_prolog = build_trisc_prolog("TRISC_ISOLATE_SFPU");
```

These defines then gate which code paths are active throughout the header tree. For example, `[tt-metal] tt_metal/hw/inc/api/compute/compute_kernel_api.h` (lines 19-52):

```cpp
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_matmul_api.h"
// ... more math includes
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
// ...
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
// ... more unpack includes
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
```

This pattern is fragile for several reasons:

1. **Hidden compilation variants**: The same source file compiles to four entirely different translation units. A change to shared code (e.g., a struct definition) may break only one of the four compilations.
2. **Macro redefinition of `MAIN`**: The `MAIN` macro is defined differently depending on which `TRISC_*` is active. This means `void MAIN` in a kernel expands to `void math_main()`, `void pack_main()`, or `void unpack_main()` depending on context -- a source of confusion.
3. **Pervasive spread**: A grep across `[tt-metal] tt_metal/hw/inc/api/compute/` reveals `TRISC_MATH`, `TRISC_UNPACK`, or `TRISC_PACK` guards in at least 15 header files including `bcast.h`, `cb_api.h`, `compute_kernel_api.h`, `reduce_custom.h`, `tile_move_copy.h`, and `common_globals.h`.

The `fake_kernels_target` also reveals this coupling: `[tt-metal] tt_metal/jit_build/fake_kernels_target/CMakeLists.txt` (lines 123-124) must pick a single TRISC mode for IDE support, defaulting to `TRISC_MATH=1`, which means IDE code navigation only works correctly for the math path.

## Global Mutable State in LLK Tests

LLK test files rely on global mutable variables for cross-function communication. For example, `[tt-llk] tests/sources/matmul_test.cpp` (lines 15-17):

```cpp
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
```

These globals are declared in `[tt-llk] tt_llk_wormhole_b0/common/inc/ckernel_globals.h` (lines 12-17) as `extern`:

```cpp
extern std::uint32_t cfg_state_id;
extern std::uint32_t unp_cfg_context;
extern std::uint32_t pack_sync_tile_dst_ptr;
extern std::uint32_t math_sync_tile_dst_index;
```

Every test file that includes this header must define these globals, and every LLK function that touches them operates on shared mutable state. This pattern:

- Makes tests non-reentrant and order-dependent.
- Prevents parallel execution of test cases within a single process.
- Mirrors the actual hardware programming model (the real Tensix core has global register state), but leaks that hardware detail into the test harness where it is unnecessary.

This appears across at least 52 test source files in `[tt-llk] tests/sources/`.

---

**Next:** [`developer_ergonomics.md`](./developer_ergonomics.md)
