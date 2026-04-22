# Build System Complexity

## The Include Path Explosion

A single kernel compilation in Metal requires an extraordinary number of include paths, assembled from multiple independent sources that must stay in sync.

### Base includes in `JitBuildEnv::init()`

In `[tt-metal] tt_metal/jit_build/build.cpp` (lines 267-285), the `JitBuildEnv::init()` method hardcodes 10 base include directories. The code itself contains a self-aware comment acknowledging the problem:

```cpp
// TODO(pgk) this list is insane
std::vector<std::string> includeDirs = {
    ".",
    "..",
    root_,
    root_ + "ttnn",
    root_ + "ttnn/cpp",
    root_ + "tt_metal",
    root_ + "tt_metal/hw/inc",
    root_ + "tt_metal/third_party/tt_llk/common",
    root_ + "tt_metal/hostdevcommon/api",
    root_ + "tt_metal/api/"};
```

### HAL adds per-architecture includes

The HAL layer then appends architecture-specific include paths. For Wormhole, `[tt-metal] tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp` (lines 101-120) adds 8 common directories plus 2 more for compute cores:

```cpp
std::vector<std::string> includes(const Params& params) const override {
    std::vector<std::string> includes;
    includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/common");
    includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_io");
    includes.push_back("tt_metal/hw/inc/internal/tt-1xx");
    includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole");
    includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines");
    includes.push_back("tt_metal/hw/inc/internal/tt-1xx/wormhole/noc");
    includes.push_back("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc");
    includes.push_back("tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib");
    // ... plus for COMPUTE cores:
    includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_api");
    includes.push_back("tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu");
    // ... plus for all Tensix cores (COMPUTE and DM):
    includes.push_back("tt_metal/hw/firmware/src/tt-1xx");
```

Blackhole has an identical pattern in `[tt-metal] tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp` (lines 103-114), substituting `blackhole` for `wormhole_b0`.

For a Tensix compute kernel on Wormhole, this totals **10 (base) + 8 (HAL common) + 2 (HAL compute) + 1 (HAL firmware for Tensix) + 1 (SFPI include dir, added at line 321 of build.cpp) = 22 include paths** on a single compiler invocation.

### Consequence

Every `-I` flag increases the compiler's header search cost and, more critically, creates ambiguity: if two directories contain a header with the same name, the include order silently determines which one wins. With 22+ search paths, reasoning about which header gets included becomes a non-trivial exercise.

## Triple Duplication of Include Path Configuration

The include path set is defined in three independent locations, each serving a different purpose:

1. **`[tt-metal] tt_metal/jit_build/build.cpp`** (runtime JIT compilation) -- the authoritative list assembled at lines 269-285 and extended by HAL queries at lines 341-345.

2. **HAL files** (per-architecture) -- `[tt-metal] tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp` and `[tt-metal] tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal.cpp` each independently enumerate architecture-specific include directories.

3. **`[tt-metal] tt_metal/jit_build/fake_kernels_target/CMakeLists.txt`** (IDE support) -- This file recreates the include paths in CMake syntax for a fake build target that exists solely so IDEs can provide code completion and navigation. It lists 23 `target_include_directories` entries (lines 65-87), including LLK-specific paths like:
   ```cmake
   "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/common"
   "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_NAME_LOWER}/common/inc"
   "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${ARCH_NAME_LOWER}/llk_lib"
   ```

When a new include directory is needed, a developer must remember to update all three locations. If the `fake_kernels_target` drifts, IDE users silently lose code navigation for the affected headers. If a HAL file drifts from `build.cpp` expectations, kernels fail to compile only on the affected architecture.

## SFPI Toolchain Management Complexity

The SFPI (Scalar Floating-Point Instruction) toolchain requires a custom RISC-V cross-compiler. Its acquisition logic in `[tt-metal] tt_metal/hw/CMakeLists.txt` (lines 40-170) spans over 130 lines of CMake with three distinct code paths:

1. **System SFPI** (`TT_USE_SYSTEM_SFPI`): Uses `/opt/tenstorrent/sfpi` -- lines 62-64.
2. **Downloaded SFPI**: Uses `FetchContent` to download a tarball by URL and hash -- lines 65-76.
3. **Local build fallback**: When no tarball exists for the platform, builds SFPI from source (estimated at ~10 CPU-hours, per the comment at line 85) -- lines 78-106.

After acquisition, the code performs:
- Version verification by running `riscv-tt-elf-g++ --version` and comparing against `SFPI_version` -- lines 113-133.
- Dependency checking by iterating over all ELF binaries under the SFPI tree, running `ldd -d` on each, and parsing for unmet shared library dependencies -- lines 136-169.

This is duplicated at runtime: `JitBuildEnv::init()` in `[tt-metal] tt_metal/jit_build/build.cpp` (lines 112-130) independently searches for the SFPI compiler in two locations with its own fallback logic:

```cpp
const std::array<std::string, 2> sfpi_roots = {
    this->root_ + "runtime/sfpi",
    "/opt/tenstorrent/sfpi"};
```

The result is two independent SFPI discovery mechanisms -- one at CMake configure time, one at JIT runtime -- that must agree on where the compiler lives.

---

**Next:** [`coupling_and_synchronization.md`](./coupling_and_synchronization.md)
