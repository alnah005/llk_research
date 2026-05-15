# 8.1 Build System Integration

This section analyzes the build system changes required to compile, link, and distribute the `BlazeDeviceOperationAdapter<OpTag>` template and its registrations alongside TT-Metal/TTNN. The current build model is examined, four integration options are defined and scored across five weighted dimensions, CMake integration details are provided for each option, and a build dependency analysis establishes that no circular dependency exists between TT-Blaze and the adapter code. Each build option is illustrated through a concrete scenario -- an engineer named Alex adding a new Blaze MicroOp called `rotary_embed` -- to show what the day-to-day experience looks like under each model.

---

## 8.1.1 Current Build Model

Today, Blaze ops and TTNN ops live in separate build universes:

```text
+-----------------------------+      +------------------------------+
|         TT-Blaze            |      |          TT-Metal            |
|  (pure Python + JIT C++)    |      |  (CMake-compiled C++ + Py)   |
|                             |      |                              |
|  blaze/ops/matmul/op.py     |      |  ttnn/cpp/ttnn/operations/   |
|  blaze/ops/matmul/op.hpp    |      |    eltwise/binary/...        |
|  blaze/compiler.py          |      |    matmul/...                |
|  blaze/kernel_codegen.py    |      |    generic/generic_op.hpp    |
|                             |      |                              |
|  Kernels compiled at        |      |  Ops compiled at build time  |
|  runtime via JIT            |      |  via CMake                   |
+-------------+---------------+      +---------------+--------------+
              |                                      |
              |   ttnn.generic_op(io_tensors, desc)   |
              +------------------------------------->|
```

| Property | TT-Blaze | TT-Metal / TTNN |
|----------|----------|-----------------|
| Language | Python (op logic), C++ (kernels only) | C++ (op + factory + kernels) |
| Build system | `pyproject.toml` / `setuptools` + JIT for kernels | CMake 3.20+ with Ninja generator |
| Registration timing | None (dispatch via `generic_op`) | Compile-time `register_operation<>` template |
| Compilation time | Runtime (JIT for kernels) | Build time (~15--45 min full rebuild) |
| Dependency direction | TT-Blaze depends on TT-Metal (imports `ttnn`) | TT-Metal has no knowledge of TT-Blaze |

The adapter ([Chapter 4](../ch04_adapter_pattern/index.md)) introduces a new C++ component that *must know Blaze op names* but *does not depend on Blaze op logic*. The adapter needs only the op name string (e.g., `"blaze::rotary_embed"`), not the Python `emit()` code, kernel sources, or compilation pipeline. The four build options differ in where that name-to-type mapping is compiled and when.

---

## 8.1.2 Four Build Integration Options

### Option A: In-Tree (Inside TT-Metal)

**Scenario: Alex adds `rotary_embed` under Option A.**

1. Alex writes the Blaze op as usual in `blaze/ops/rotary_embed/op.py`.
2. Alex opens a PR against TT-Metal adding one line to the X-macro:

```cpp
// TT-Metal: ttnn/cpp/ttnn/operations/blaze/blaze_op_list.hpp
#define BLAZE_OP_LIST(X)              \
    X(matmul, Matmul)                 \
    X(copy, Copy)                     \
    X(reduce, Reduce)                 \
    // ... full catalog in Section 5.3
    X(rotary_embed, RotaryEmbed)      // <-- Alex adds this line
```

3. Alex rebuilds TT-Metal. The X-macro expansion automatically generates `RotaryEmbedTag`, `ttnn::blaze::rotary_embed`, and its nanobind binding. Incremental rebuild: ~7--13 seconds.
4. `_resolve_dispatch()` finds `ttnn.blaze.rotary_embed` and routes through it. Tracy shows `[ttnn::blaze::rotary_embed]`.

**Source tree layout:**

```text
ttnn/cpp/ttnn/operations/blaze/
    blaze_device_operation_adapter.hpp    # Template definition (from Section 4.2)
    blaze_op_list.hpp                     # X-macro op list
    blaze_op_registrations.hpp            # Tag structs + register_operation<> calls
    blaze_ops_nanobind.cpp                # Python bindings
    CMakeLists.txt                        # ~15 lines
```

**CMake integration:**

```cmake
# PROPOSED: ttnn/cpp/ttnn/operations/blaze/CMakeLists.txt
target_sources(ttnn_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/blaze_ops_nanobind.cpp
)

target_include_directories(ttnn_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

No new library target needed -- the adapter compiles as part of the existing `ttnn_lib` (which produces `_ttnn.so`). The parent `operations/CMakeLists.txt` adds one line: `add_subdirectory(blaze)`.

**Estimated artifact size:** ~625 lines total (~350 adapter template, ~130 op list, ~50 registrations, ~80 bindings, ~15 CMake).

### Option B: Out-of-Tree Plugin (Separate Shared Library)

**Scenario: Alex adds `rotary_embed` under Option B.**

1. Alex writes the Blaze op as usual.
2. Instead of modifying TT-Metal, Alex opens a PR against the TT-Blaze repository's plugin directory:

```text
tt-blaze/
    blaze_ttnn_plugin/
        plugin_op_list.hpp         # Plugin's own X-macro list
        plugin_nanobind.cpp        # Nanobind bindings
        CMakeLists.txt             # Plugin build rules
```

3. Alex builds the plugin as a separate shared library:

```bash
cd tt-blaze/blaze_ttnn_plugin/build
cmake .. -DTTMETAL_ROOT=/path/to/tt-metal
make -j$(nproc)
# Produces: _ttnn_blaze.so
```

4. At runtime, the plugin is imported before dispatch:

```python
# At import time (e.g., in blaze/__init__.py)
try:
    import blaze_ttnn_plugin  # Registers ttnn.blaze.* ops
except ImportError:
    pass  # Plugin not installed; fall back to generic_op
```

**CMake integration:**

```cmake
# PROPOSED: blaze_ttnn_plugin/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(blaze_ttnn_plugin LANGUAGES CXX)

find_package(ttnn REQUIRED)
find_package(nanobind REQUIRED)

nanobind_add_module(_ttnn_blaze
    plugin_nanobind.cpp
)

target_link_libraries(_ttnn_blaze PRIVATE
    ttnn::ttnn_lib
    nanobind::module
)

install(TARGETS _ttnn_blaze DESTINATION ttnn_blaze)
```

**Estimated artifact size:** ~700 lines (same adapter code + ~75 CMake + ~30 Python glue).

> **Warning:** ABI coupling is the primary risk. If TT-Metal changes `MeshProgramDescriptor` layout, `DeviceOperationConcept` requirements, or `register_operation<>` template mechanics, the plugin must be recompiled. A version check at load time is essential:

```cpp
static_assert(TTMETAL_ABI_VERSION == EXPECTED_ABI_VERSION,
    "Plugin compiled against incompatible TT-Metal version");
```

### Option C: JIT Registration (Runtime, No Pre-Compilation)

**Critical blocker:** TTNN's `register_operation<>` uses a non-type template parameter (NTTP) string literal for the operation name. This string must be a compile-time constant (`tt::stl::fixed_string`). A runtime string from Python cannot serve as an NTTP, making full JIT registration architecturally incompatible with TTNN's current model.

**Estimated effort to enable full JIT:** ~1,500--3,000 lines of C++ changes to TTNN's registration infrastructure -- a framework-level project not justified by the Blaze use case alone.

See [Section 8.4, Open Question 5](04_open_questions.md) for a detailed analysis of dynamic registration approaches and workarounds.

### Option D: Hybrid (Recommended)

**Scenario: Alex adds `rotary_embed` under Option D.**

Alex checks: is `rotary_embed` an official Blaze op?

- **Yes (common case, ~95% of ops):** Alex adds one line to `blaze_op_list.hpp` in TT-Metal (Option A path). Rebuild: ~7--13 seconds.
- **No (experimental/user-defined):** Alex does nothing. The op dispatches through `generic_op` today. Phase 0's `op_name` field provides profiler visibility without adapter code. If persistent, Alex can build an out-of-tree plugin (Option B path).

**Decision tree for engineers:**

```text
                Is this an official Blaze op
                (in the main tt-blaze repo)?
                          |
                   +------+------+
                   |             |
                  Yes           No
                   |             |
                   v             v
              Add 1 line     Does it need
              to X-macro     per-op profiling
              (Option A)     identity?
                   |             |
                   v        +----+----+
              Rebuild        |         |
              (~10s)        Yes       No
                   |         |         |
                   v         v         v
              Done     Use op_name   Use generic_op
                       in Phase 0    (zero effort)
                       descriptor
                       OR
                       Build plugin
                       (Option B)
```

---

## 8.1.3 Quantitative Decision Matrix

Each option is scored on five dimensions using a 1--5 scale (5 = best). Weights reflect engineering priorities for the Blaze use case.

| Dimension | Weight | A: In-Tree | B: Out-of-Tree | C: JIT | D: Hybrid | Notes |
|-----------|:------:|:----------:|:--------------:|:------:|:---------:|-------|
| **Implementation Effort** | 0.25 | **5** | 4 | 1 | 3 | A: ~625 LoC. B: +75 LoC CMake + packaging. C: ~2,000+ LoC new infra. D: A (~625 LoC) + ~100 LoC fallback logic. |
| **Build-Time Impact** | 0.15 | 3 | **5** | 4 | 3 | A: +40--60s incremental. B: zero impact on tt-metal. C: minimal build, import-time cost. D: same as A. |
| **Maintenance Cost** | 0.20 | 4 | 3 | 2 | 3 | A: single repo, standard CMake. B: separate repo + versioning. C: novel infra. D: two code paths. |
| **Registration Fidelity** | 0.25 | **5** | **5** | 2 | 4 | A/B: full `register_operation<>` with NTTP, `type_hash`, profiler, graph tracing. C: no compile-time identity. D: full for known ops, degraded for dynamic. |
| **Extensibility** | 0.15 | 2 | 3 | **5** | **5** | A: new ops require tt-metal rebuild. B: separate rebuild. C/D: dynamic for experimental. |

### Score Derivation

$$\text{Option A:} \quad 0.25(5) + 0.15(3) + 0.20(4) + 0.25(5) + 0.15(2) = 1.25 + 0.45 + 0.80 + 1.25 + 0.30 = 4.05$$

$$\text{Option B:} \quad 0.25(4) + 0.15(5) + 0.20(3) + 0.25(5) + 0.15(3) = 1.00 + 0.75 + 0.60 + 1.25 + 0.45 = 4.05$$

$$\text{Option C:} \quad 0.25(1) + 0.15(4) + 0.20(2) + 0.25(2) + 0.15(5) = 0.25 + 0.60 + 0.40 + 0.50 + 0.75 = 2.50$$

$$\text{Option D:} \quad 0.25(3) + 0.15(3) + 0.20(3) + 0.25(4) + 0.15(5) = 0.75 + 0.45 + 0.60 + 1.00 + 0.75 = 3.55$$

> **Note:** Options A and B tie at 4.05. The tie-breaker is organizational: if TT-Blaze and TT-Metal are maintained by the same team with coordinated releases, Option A is simpler. If they have independent release cycles, Option B provides better decoupling. Option D is recommended because it uses Option A for the common case (~95% of ops) while preserving Option B as an escape hatch and `generic_op` fallback for experimental ops.

---

## 8.1.4 Build Dependency Analysis

A critical concern is whether the adapter creates a circular dependency between TT-Blaze and TT-Metal.

### Dependency Graph (Current)

```text
TT-Blaze (Python)
    |
    | runtime import
    v
ttnn (_ttnn.so)
    |
    | links against
    v
tt_metal (libtt_metal.so)
```

### Dependency Graph (After Option A/D)

```text
TT-Blaze (Python)
    |
    | runtime import
    v
ttnn (_ttnn.so)
    |--- contains: BlazeDeviceOperationAdapter<OpTag> instances
    |--- contains: ttnn.blaze.* nanobind bindings
    |
    | links against
    v
tt_metal (libtt_metal.so)
```

**No circular dependency.** The adapter code inside `_ttnn.so`:
- **Depends on:** tt-metal headers (`operation_concepts.hpp`, `decorators.hpp`, `mesh_program_descriptor.hpp`, `generic_op_device_operation.hpp`)
- **Does NOT depend on:** any TT-Blaze Python code, any TT-Blaze kernel source, any blaze-nn code
- **Does NOT import:** `blaze.*` or `blaze_nn.*` at build time

The op list (`blaze_op_list.hpp`) contains only string literals (`"matmul"`, `"rmsnorm"`, etc.). These are human-maintained or generated by a build-time script that reads from a static YAML/JSON manifest -- not by importing the Blaze Python package.

---

## 8.1.5 Build-Time Cost Analysis

Template instantiation is the primary build-time concern. Each `BlazeDeviceOperationAdapter<Tag>` instantiation triggers:

1. `register_operation<NTTP, Adapter<Tag>>()` -- constexpr friend injection, `operation_name_key_t`, `operation_key_t`
2. `device_operation::launch<Adapter<Tag>>` specialization -- the full dispatch pipeline (~2,000 lines of template code)
3. `MeshDeviceOperationAdapter<Adapter<Tag>>` wrapping (if single-device factory)
4. Nanobind `bind_registered_operation()` call -- generates Python callable

### Measured Per-Instantiation Cost (Analogous)

Using existing TTNN ops as a proxy: the `BinaryDeviceOperation` registration compiles in ~0.4 seconds on a Ryzen 9 7950X. The adapter is structurally simpler (no custom factory, no multi-variant `program_factory_t`), so the expected range is ~0.3--0.5 seconds per instantiation.

### Scaling Projections

| Ops Registered | Estimated Incremental Build Time | Estimated Binary Size Increase |
|:--------------:|:--------------------------------:|:------------------------------:|
| 10 | 3--5 seconds | ~200 KB |
| 50 | 15--25 seconds | ~1.0 MB |
| 112 | 35--50 seconds | ~2.2 MB |
| 200 (future) | 60--100 seconds | ~4.0 MB |

Binary size at 112 ops (~2.2 MB) is < 5% of a typical `_ttnn.so` binary. Instruction cache impact is bounded: at ~500--1,500 bytes per `launch<>` specialization, 112 instantiations produce ~55--165 KB of additional code, which fits in L2 but may create L1 icache pressure during autoregressive decode hot loops (see [Section 8.4, Open Question 7](04_open_questions.md)).

### Mitigation Strategies

1. **Unity build:** Combine all tag instantiations into a single translation unit. Eliminates per-TU overhead but creates a monolithic compile unit (~50 seconds single-threaded).
2. **Explicit instantiation:** Declare `extern template` in the header, define instantiations in batched `.cpp` files (e.g., 10 ops per file). Enables parallel compilation.
3. **Sharded `.cpp` files (recommended):** Split `blaze_ops_nanobind.cpp` into 4--6 shards (e.g., `blaze_ops_nanobind_a_f.cpp`, `blaze_ops_nanobind_g_m.cpp`), each registering ~20 ops. Parallelizes across cores.

Estimated incremental build time with 6 shards on a 16-core machine: **~8--12 seconds** for 112 ops.

---

## 8.1.6 Recommendation

**Primary recommendation: Option D (hybrid).** It combines the determinism and IDE support of in-tree builds (Option A) for the ~95% common case with `generic_op` fallback for experimental ops, and leaves the door open for out-of-tree plugins (Option B) as a power-user escape hatch.

**For teams with coordinated releases:** Use Option A as the primary path. One line of C++ per op, deterministic build, full IDE support.

**For teams with independent release cycles:** Use Option B as the primary path. The +75 LoC overhead is trivial, and build isolation prevents adapter changes from triggering tt-metal CI.

**Option C (JIT-only) is not recommended** as a primary strategy due to the fundamental incompatibility with `register_operation<>`'s compile-time model. It may be revisited if TTNN adds a runtime registration API (see [Section 8.4, Open Question 5](04_open_questions.md)).

---

| **Section 8.1** | [Section 8.2: Migration Plan](02_migration_plan.md) |
|:---|---:|
