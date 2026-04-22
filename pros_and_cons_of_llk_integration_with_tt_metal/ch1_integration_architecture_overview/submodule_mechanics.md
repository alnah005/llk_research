# Submodule Mechanics

This section covers how TT-LLK is embedded in TT-Metal as a git submodule, how version synchronization works, and how CMake exposes LLK headers to both the build system and IDE tooling.

## Git Submodule Configuration

The `.gitmodules` file in the TT-Metal repository root declares four submodules. The TT-LLK entry is:

```ini
[submodule "tt_metal/third_party/tt_llk"]
    path = tt_metal/third_party/tt_llk
    url = https://github.com/tenstorrent/tt-llk.git
    branch = main
```

**Source:** [`.gitmodules`](https://github.com/tenstorrent/tt-metal/blob/main/.gitmodules)

Key details:

- **Path:** `tt_metal/third_party/tt_llk` -- LLK is placed alongside other third-party dependencies (`tracy`, `umd`).
- **Branch:** `main` -- The submodule tracks the `main` branch. This means `git submodule update --remote` will pull the latest commit from `main`, but the pinned commit in TT-Metal's tree determines what actually gets checked out in CI and developer builds.
- **URL:** Points to the public `tenstorrent/tt-llk` repository on GitHub.

## Version Synchronization

### Submodule Commit Pinning

Git submodules work by storing a specific commit SHA in the parent repository's tree object. When a developer clones TT-Metal and runs `git submodule update --init`, they get exactly the commit that was pinned at the time of the last submodule-updating commit in TT-Metal.

The typical update workflow is:

1. A developer updates TT-LLK (pushes to `main` in `tt-llk`).
2. In the TT-Metal repo, someone runs `cd tt_metal/third_party/tt_llk && git pull origin main`.
3. They commit the new submodule pointer in TT-Metal: `git add tt_metal/third_party/tt_llk && git commit`.
4. The PR undergoes CI to verify the new LLK commit is compatible.

### Risk of Drift

Because `branch = main` is specified, there is an asymmetry: the `.gitmodules` file suggests tracking `main`, but the actual checked-out version is the pinned commit. This creates several risks:

- **Stale pins:** If TT-Metal does not regularly update the submodule pointer, it falls behind `tt-llk` `main`. New LLK fixes or features are not picked up.
- **Breaking updates:** If TT-Metal updates the pin to a `tt-llk` commit that changed Layer 1 function signatures, Layer 2 wrappers in `tt_metal/hw/ckernels/` will fail to compile. Since Layer 1 and Layer 2 live in different repositories, there is no single PR that atomically updates both.
- **Developer confusion:** A developer working in TT-LLK may run `git submodule update --remote` in TT-Metal and get a newer LLK commit than what CI tested, leading to "works on my machine" issues.

## CMakeLists.txt Integration

The file [`tt_metal/CMakeLists.txt`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/CMakeLists.txt) (lines 157-205) sets up an `INTERFACE` library named `jitapi` that aggregates headers for device-side compilation.

### Header Globbing

LLK headers are collected using `GLOB_RECURSE` at lines 170-176:

```cmake
file(
    GLOB_RECURSE TT_LLK_HEADERS
    third_party/tt_llk/common/*.h
    third_party/tt_llk/tt_llk_wormhole_b0/**/*.h
    third_party/tt_llk/tt_llk_blackhole/**/*.h
    third_party/tt_llk/tt_llk_quasar/**/*.h
)
```

This collects every `.h` file across all architecture-specific LLK directories and the shared `common/` directory. The result is declared as part of an `INTERFACE` file set:

```cmake
add_library(jitapi INTERFACE)
target_sources(
    jitapi
    INTERFACE
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ...
)
```

**Source:** [`tt_metal/CMakeLists.txt:159-183`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/CMakeLists.txt#L159)

### INTERFACE Semantics

The `jitapi` target is `INTERFACE`-only, meaning it carries no compiled object code. It exists purely to:

1. Track which headers belong to the JIT compilation surface for packaging and installation.
2. Provide IDE integration (e.g., clangd can follow these headers for autocompletion).
3. Satisfy CMake's `install(TARGETS jitapi ...)` for distribution.

Notably, `VERIFY_INTERFACE_HEADER_SETS` is set to `FALSE` (line 164), because these headers target the RISC-V device compiler, not the host compiler. The CMake comment at line 160 makes this explicit: "These headers are for the device, not host; will require cross compiling to verify."

### Why GLOB_RECURSE Matters

Using `GLOB_RECURSE` means that CMake will not detect newly added or removed `.h` files until the next CMake re-configure. This is a known CMake limitation. If a developer adds a new header to `tt-llk` and updates the submodule pin, the CMake cache must be regenerated for the new header to appear in the `jitapi` file set. This does not affect JIT compilation at runtime (which uses explicit `-I` paths), but it affects IDE tooling and packaging.

## The Include Path Complexity Problem

The JIT build system constructs include paths from two independent sources:

1. **`JitBuildEnv::init()`** ([`build.cpp:268-279`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L268)) -- a static list of common include directories.
2. **`HalJitBuildQueryInterface::includes()`** ([`wh_hal.cpp:101-134`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp#L101)) -- a per-architecture, per-core-type list.

The developer comment at [`build.cpp:268`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/jit_build/build.cpp#L268) captures the sentiment:

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

This static list is then extended by the HAL query (in `JitBuildState` at lines 340-345), adding architecture-specific paths like `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api`. The final include path for a single Wormhole COMPUTE kernel compilation can easily exceed 15 `-I` flags, spanning three repositories (tt-metal, tt-llk, and the sfpi toolchain).

The `TODO(pgk) this list is insane` comment is not merely humorous -- it signals that even the developers recognize the include path graph as a maintenance burden. Adding a new architecture or restructuring LLK directories requires updating both the static list and the HAL query, with no compile-time check that they are consistent.

---

**Next:** [`jit_compilation_pipeline.md`](./jit_compilation_pipeline.md)
