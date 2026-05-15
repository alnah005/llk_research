# 5.3 Batch Registration and Scaling

Sections [5.1](01_microop_registration_walkthrough.md) and [5.2](02_fusedop_registration_walkthrough.md) demonstrated that registering a single Blaze op requires one line in the X-macro list and zero per-op C++ logic. This section scales that to the full catalog: 112+ ops (approximately 60 MicroOps and 50+ FusedOps) registered without manual per-op boilerplate. Three approaches are analyzed -- C++ variadic template iteration, Python codegen, and macro-based registration table -- followed by the build system integration, template instantiation cost analysis, runtime discovery mechanism, and the open set problem for ops added after the TT-Metal build.

---

## 5.3.1 Approach 1: C++ Variadic Template

A `TagList<MatmulTag, CopyTag, ...>` type list with a recursive `RegisterAll<TagList<Tags...>>` template could iterate all tags and call `bind_blaze_op()` in a single loop. However, `register_operation<>` requires a compile-time string literal NTTP for the operation name (`"ttnn::blaze::matmul"`), and constructing this string from `Tag::name` at compile time requires `constexpr` string concatenation that is verbose and fragile across compilers. In practice, each `register_operation<>` call must be enumerated individually, defeating the purpose of the variadic approach.

**Verdict:** Impractical for registration due to the string NTTP constraint. However, the variadic template is useful as a *complement* to the X-macro for post-registration bulk validation -- e.g., a `static_assert(DeviceOperationConcept<BlazeDeviceOperationAdapter<Tag>>)` check for all tags at compile time (see [Section 5.3.9](#539-scaling-verification)).

---

## 5.3.2 Approach 2: Python Codegen Script

A build-time script (`tools/generate_blaze_registrations.py`) reads `BlazeOp._class_registry`, derives tag names, and emits four generated files: tag structs + registrations (`.hpp`), nanobind bindings (`.cpp`), Python type stubs (`.pyi`), and an op name list for runtime discovery. A CMake `add_custom_command` triggers the script when Blaze source files change.

**Advantages:** Fully automated (new ops registered at next build without C++ changes), single source of truth (Python registry), emits `.pyi` stubs. **Disadvantages:** Build-time dependency on Blaze Python package, generated files harder to review in PRs, IDE indexing of generated headers may be delayed. See the comparison table in [Section 5.3.4](#534-approach-comparison-and-recommendation) for the full tradeoff analysis.

---

## 5.3.3 Approach 3: Macro-Based Registration Table

The X-macro approach from [Section 4.2.6](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md) maintains a single C++ list expanded in three contexts:

```cpp
// PROPOSED: ttnn/cpp/ttnn/operations/blaze/blaze_op_list.hpp
// SINGLE SOURCE OF TRUTH -- add new ops here only
#define BLAZE_OP_LIST(X)              \
    /* ===== MicroOps ===== */        \
    X(matmul, Matmul)                 \
    X(copy, Copy)                     \
    X(reduce, Reduce)                 \
    X(mcast, Mcast)                   \
    X(gather, Gather)                 \
    X(scatter, Scatter)               \
    X(transpose, Transpose)           \
    X(topk, TopK)                     \
    X(rmsnorm, RMSNorm)              \
    X(residual_add, ResidualAdd)     \
    X(softmax, Softmax)              \
    X(rope, RoPE)                    \
    X(silu, SiLU)                    \
    X(gelu, GeLU)                    \
    X(swiglu, SwiGLU)               \
    X(quantize, Quantize)            \
    X(dequantize, Dequantize)        \
    X(sdpa, SDPA)                    \
    X(sdpa_decode, SDPADecode)       \
    X(noc_write, NocWrite)           \
    X(noc_read, NocRead)             \
    X(barrier, Barrier)              \
    X(allgather, AllGather)          \
    X(reduce_scatter, ReduceScatter) \
    /* ... ~35 more MicroOps ... */   \
    /* ===== FusedOps ===== */        \
    X(gated_reduce, GatedReduce)     \
    X(gated_mlp, GatedMLP)          \
    X(dense_mlp, DenseMLP)          \
    X(dense_swiglu, DenseSwiGLU)    \
    X(shared_expert, SharedExpert)   \
    X(down_proj, DownProj)           \
    X(fused_attention, FusedAttention)  \
    X(fused_decode_attention, FusedDecodeAttention)  \
    X(moe_gate, MoEGate)            \
    X(moe_dispatch, MoEDispatch)     \
    X(rotary_embedding, RotaryEmbedding)  \
    /* ... ~40 more FusedOps ... */
```

Three macro expansions consume this list:

```cpp
// 1. Generate tag structs
#define DEFINE_BLAZE_TAG(op_name, tag_name)         \
    struct tag_name##Tag {                           \
        static constexpr const char* name =          \
            "blaze::" #op_name;                      \
    };

namespace ttnn::operations::blaze {
BLAZE_OP_LIST(DEFINE_BLAZE_TAG)
}

// 2. Generate registrations
#define REGISTER_BLAZE_OP(op_name, tag_name)         \
    constexpr auto op_name =                         \
        ::ttnn::register_operation<                  \
            "ttnn::blaze::" #op_name,                \
            ::ttnn::operations::blaze::              \
                BlazeDeviceOperationAdapter<          \
                    ::ttnn::operations::blaze::       \
                        tag_name##Tag>>();

namespace ttnn::blaze {
BLAZE_OP_LIST(REGISTER_BLAZE_OP)
}

// 3. Generate nanobind bindings
#define BIND_BLAZE_OP(op_name, tag_name)             \
    ::ttnn::operations::blaze::bind_blaze_op(        \
        blaze_mod, ttnn::blaze::op_name,             \
        "Blaze " #tag_name);

void bind_blaze_operations(nb::module_& parent) {
    auto blaze_mod = parent.def_submodule("blaze");
    BLAZE_OP_LIST(BIND_BLAZE_OP)
}
```

**Advantages:**
- No external dependencies; pure C++ preprocessor
- Full IDE support (autocomplete, go-to-definition)
- Easy to review: adding an op is one line in one file
- Deterministic builds: no Python execution at build time

**Disadvantages:**
- Manual maintenance: new Blaze ops require a C++ change
- Risk of silent drift from the Python registry

---

## 5.3.4 Comparison and Recommendation

| Factor | Variadic Template | Python Codegen | X-Macro (Recommended) |
|--------|:-:|:-:|:-:|
| Per-op C++ changes | ~0 (if string NTTP solved) | 0 | 1 line |
| String NTTP constraint | Problematic | N/A (generates string literals) | Natural (preprocessor concatenation) |
| Build dependency on Blaze | No | Yes | No |
| Drift detection | Compile-time (type list) | Inherent (reads registry) | Requires CI check |
| IDE indexing | Full | Partial (generated files) | Full |
| Code review clarity | High | Low (generated code) | High |
| Complexity | High | Medium | Low |

**Recommendation:** Use the X-macro approach (Approach 3) as the primary mechanism, supplemented by CI validation from the hybrid strategy described in [Section 4.3.3](../ch04_adapter_pattern/03_python_dispatch_and_registration.md):

1. **Primary:** `BLAZE_OP_LIST` X-macro in `blaze_op_list.hpp` drives tag structs, registrations, and bindings.
2. **Validation:** A CI script (`validate_blaze_registrations.py`) compares the macro list against `BlazeOp._class_registry` and fails the build if they diverge.
3. **Optional enhancement:** Use the variadic template for compile-time concept validation of all tags.

This combination provides deterministic builds, full IDE support, and automated drift detection with minimal complexity.

---

## 5.3.5 Build System Integration

### Source Tree Layout

```text
ttnn/cpp/ttnn/operations/blaze/
    blaze_device_operation_adapter.hpp   # Adapter template (from Ch4)
    blaze_op_list.hpp                    # X-macro list (single source of truth)
    blaze_op_tags.hpp                    # Generated tag structs (from macro)
    blaze_op_registrations.hpp           # Generated register_operation<> calls
    blaze_ops_nanobind.cpp               # Nanobind binding source
    blaze_batch_validation.hpp           # Optional: variadic concept checks
    CMakeLists.txt                       # Build rules for this directory
```

### CMake Rules

```cmake
# PROPOSED: ttnn/cpp/ttnn/operations/blaze/CMakeLists.txt

# The adapter is header-only; no .cpp compilation needed for it.
# Only the nanobind binding file needs compilation.
target_sources(ttnn_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/blaze_ops_nanobind.cpp
)

target_include_directories(ttnn_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

### Incremental Rebuild Strategy

Adding a new op to `BLAZE_OP_LIST` modifies `blaze_op_list.hpp`. Because this header is `#include`d by `blaze_op_tags.hpp`, `blaze_op_registrations.hpp`, and `blaze_ops_nanobind.cpp`, CMake's dependency tracking triggers recompilation of exactly one translation unit: `blaze_ops_nanobind.cpp`. No other TTNN source files are affected.

The rebuild cost for adding one op:

| Step | Time |
|------|------|
| Preprocess `blaze_ops_nanobind.cpp` with updated macro | ~1 s |
| Instantiate one new `BlazeDeviceOperationAdapter<Tag>` | ~0.5--1 s |
| Instantiate one new `device_operation::launch<>` specialization | ~0.5--1 s |
| Link updated `ttnn_lib` | ~5--10 s |
| **Total** | **~7--13 s** |

This is fast enough for iterative development. The full rebuild (all 112 ops) takes longer due to template instantiation volume, which the next section addresses.

---

## 5.3.6 Template Instantiation Cost

Each `BlazeDeviceOperationAdapter<Tag>` instantiation triggers template expansion of several layers:

| Template | Instantiated Per-Op | Approximate Code Size |
|----------|:---:|---|
| `BlazeDeviceOperationAdapter<Tag>` methods | Yes | ~1--2 KB |
| `device_operation::launch<Adapter<Tag>>` | Yes | ~2--5 KB |
| `MeshDeviceOperationAdapter<Adapter<Tag>>` (if used) | Yes | ~1--3 KB |
| `registered_operation_t<Name, Adapter<Tag>>` | Yes | ~0.5--1 KB |
| `BlazeAdapterMeshProgramFactory` methods | Yes (but shared type) | ~0.5--1 KB |

For 112 instantiations (refining the preliminary ~400 KB -- 1 MB estimate from [Section 4.2.10](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md) to include the `registered_operation_t` overhead identified during concrete instantiation analysis):

$$
\text{Total object code} \approx 112 \times (5\text{--}12\;\text{KB}) \approx 560\;\text{KB}\text{--}1.3\;\text{MB}
$$

### Mitigation Strategy 1: Explicit Template Instantiation

Place all explicit instantiations in a single `.cpp` file to prevent redundant instantiation across translation units:

```cpp
// PROPOSED: blaze_explicit_instantiations.cpp
#include "blaze_device_operation_adapter.hpp"
#include "blaze_op_tags.hpp"

// Explicit instantiation definitions at namespace scope (not in an anonymous namespace,
// which would give internal linkage and defeat cross-TU deduplication)
namespace ttnn::operations::blaze {

#define INSTANTIATE_ADAPTER(op_name, tag_name)                     \
    template struct BlazeDeviceOperationAdapter<tag_name##Tag>;

BLAZE_OP_LIST(INSTANTIATE_ADAPTER)

}  // namespace ttnn::operations::blaze
```

Pair these definitions with `extern template` declarations in the header to suppress implicit instantiation in other translation units:

```cpp
// In blaze_device_operation_adapter.hpp (after the template definition)
namespace ttnn::operations::blaze {

#define EXTERN_ADAPTER(op_name, tag_name)                          \
    extern template struct BlazeDeviceOperationAdapter<tag_name##Tag>;

BLAZE_OP_LIST(EXTERN_ADAPTER)

}  // namespace ttnn::operations::blaze
```

This ensures each adapter is instantiated exactly once in `blaze_explicit_instantiations.cpp`, and the `extern template` declarations prevent other translation units from triggering redundant implicit instantiations.

### Mitigation Strategy 2: Unity Build

Combine all Blaze registration sources into a single compilation unit:

```cmake
# PROPOSED: Unity build for Blaze registrations
set_target_properties(ttnn_lib PROPERTIES
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 0  # all Blaze sources in one unit
)
```

Unity builds eliminate inter-TU duplicate instantiation and enable the compiler to share common subexpression analysis across all 112 instantiations.

### Mitigation Strategy 3: Link-Time Optimization

Modern LTO (LLVM's ThinLTO or GCC's `-flto`) merges identical function bodies across instantiations. Many adapter methods (e.g., `compute_output_specs`, `create_output_tensors`) produce identical machine code regardless of `OpTag`, and LTO can merge them:

$$
\text{After LTO} \approx 560\;\text{KB}\text{--}1.3\;\text{MB} \times 0.4\text{--}0.6 \approx 220\;\text{KB}\text{--}780\;\text{KB}
$$

### Compile Time Impact

| Configuration | Compile Time (full rebuild) | Incremental (1 new op) |
|---------------|:-:|:-:|
| No mitigation | ~30--60 s | ~7--13 s |
| Explicit instantiation | ~20--40 s | ~5--10 s |
| Unity build | ~15--25 s | ~5--10 s |
| Unity + explicit instantiation + LTO | ~15--25 s (compile) + ~10--20 s (link) | ~5--10 s |

These times are design estimates relative to the full TT-Metal build (~10--30 minutes). The Blaze registration adds approximately 0.5--2% to the total build time. See [Section 8.1.5](../ch08_build_migration_alternatives/01_build_system_integration.md) for detailed per-instantiation measurements and sharded compilation projections (~8--12s with 6 shards).

---

## 5.3.7 Runtime Discovery: `ttnn.blaze.list_ops()`

Test frameworks, sweep harnesses, and diagnostic tools need to discover which Blaze ops are registered without hardcoding a list. A `list_ops()` function in the `ttnn.blaze` module provides this, backed by the same `BLAZE_OP_LIST` macro that drives registration:

```cpp
// PROPOSED: In blaze_ops_nanobind.cpp
void bind_blaze_operations(nb::module_& parent_mod) {
    auto blaze_mod = parent_mod.def_submodule("blaze", ...);

    // Register individual ops (as before)
    BLAZE_OP_LIST(BIND_BLAZE_OP)

    // Register discovery function -- driven by BLAZE_OP_LIST
    std::vector<std::string> op_names;
    #define COLLECT_OP_NAME(op_name, tag_name) \
        op_names.push_back(#op_name);
    BLAZE_OP_LIST(COLLECT_OP_NAME)
    #undef COLLECT_OP_NAME

    blaze_mod.def("list_ops", [op_names]() {
        return op_names;
    }, "Return a list of all registered Blaze op names");

    blaze_mod.def("num_ops", [op_names]() {
        return op_names.size();
    }, "Return the number of registered Blaze ops");
}
```

The `list_ops()` implementation is driven by `BLAZE_OP_LIST`, not by Python-side introspection (e.g., `dir()` + `__ttnn_operation__` filtering). This guarantees that the discovery list exactly matches the set of registered operations -- no naming convention dependencies, no risk of including non-op attributes, and deterministic ordering.

Usage from Python:

```python
>>> ttnn.blaze.list_ops()
['matmul', 'copy', 'reduce', 'mcast', 'gather', 'transpose', 'topk',
 'rmsnorm', 'sdpa', 'gated_mlp', 'gated_reduce', 'shared_expert',
 'dense_mlp', 'dense_swiglu', 'fused_attention', ...]

>>> ttnn.blaze.num_ops()
112

>>> # Sweep framework integration
>>> for op_name in ttnn.blaze.list_ops():
...     op_fn = getattr(ttnn.blaze, op_name)
...     assert op_fn.__ttnn_operation__
...     run_sweep_for_op(op_fn)
```

### Integration with Existing Frameworks

| Framework | Integration Point |
|-----------|------------------|
| `sweep_framework/load_ttnn_ops_data_v2.py` | Add `ttnn.blaze.list_ops()` as an op source alongside native TTNN ops |
| `model_tracer/generic_ops_tracer.py` | Use `list_ops()` to differentiate registered Blaze ops from anonymous `generic_op` calls |
| CI test harness | Iterate `list_ops()` to auto-generate per-op test targets |
| Profiler analysis scripts | Map Tracy zone names to `list_ops()` entries for automated regression detection |

---

## 5.3.8 The Open Set Problem

The 112+ ops in `BLAZE_OP_LIST` represent the *current* Blaze op catalog. But Blaze is actively developed: new MicroOps and FusedOps are added regularly. Three questions arise:

### Question 1: Must TT-Metal be rebuilt when a Blaze op is added?

With the X-macro approach, yes. Adding a new op requires adding a line to `blaze_op_list.hpp` and rebuilding the Blaze registration translation unit (~7--13 seconds incremental). This is acceptable for ops that are part of the main Blaze repository, since TT-Metal and Blaze are typically built together.

### Question 2: Can user-defined ops be registered without rebuilding TT-Metal?

User-defined Blaze ops (written by model developers, not in the main repository) cannot be added to `BLAZE_OP_LIST` without modifying and rebuilding TT-Metal sources. Three options exist:

**Option A: Finite "known ops" list (recommended for now).** The X-macro list covers all ops in the official Blaze repository. User-defined ops dispatch through `ttnn.generic_op` as they do today. This is the simplest approach and has zero risk.

**Option B: Out-of-tree plugin (future work).** A separate shared library (`blaze_custom_ops.so`) compiled against TT-Metal headers includes its own `register_operation<>` calls and tag structs for user-defined ops. Feasible but requires users to maintain a C++ build pipeline.

**Option C: Dynamic registration path (future work).** A runtime API (`ttnn.blaze.register_dynamic("my_custom_op")`) that creates adapter instances without C++ compilation. This requires replacing `type_hash` (compile-time type identity) with a string-based hash scheme — architecturally feasible but outside the current scope. See [Chapter 8](../ch08_build_migration_alternatives/index.md) for discussion.

### Question 3: What happens when the Blaze registry and X-macro list diverge?

The CI validation script from [Section 4.3.3](../ch04_adapter_pattern/03_python_dispatch_and_registration.md) catches divergence:

```python
# CI: tools/validate_blaze_registrations.py
def validate():
    registered_in_cpp = parse_macro_list("blaze_op_list.hpp")
    registered_in_python = set(BlazeOp._class_registry.keys())

    missing = registered_in_python - registered_in_cpp
    extra = registered_in_cpp - registered_in_python

    if missing:
        print(f"ERROR: Blaze ops missing from C++ registration: {missing}")
        print("Add these to BLAZE_OP_LIST in blaze_op_list.hpp")
        sys.exit(1)
    if extra:
        print(f"WARNING: C++ registrations for removed Blaze ops: {extra}")
        print("Consider removing these from BLAZE_OP_LIST")
```

This script runs in CI on every PR that modifies Blaze op definitions or the X-macro list. The failure mode is a clear error message with the exact op names that need to be added or removed.

---

## 5.3.9 Scaling Verification

To verify that the batch registration approach works at scale, the following checks should pass for the full 112+ op catalog:

### Compile-Time Checks

```cpp
// PROPOSED: blaze_batch_validation.hpp
#include "blaze_op_list.hpp"
#include "blaze_device_operation_adapter.hpp"

// Verify every adapter satisfies DeviceOperationConcept
#define ASSERT_CONCEPT(op_name, tag_name)                          \
    static_assert(                                                 \
        ::ttnn::device_operation::DeviceOperationConcept<          \
            ::ttnn::operations::blaze::BlazeDeviceOperationAdapter<\
                ::ttnn::operations::blaze::tag_name##Tag>>,        \
        "Adapter<" #tag_name "Tag> must satisfy DeviceOperationConcept");

BLAZE_OP_LIST(ASSERT_CONCEPT)
#undef ASSERT_CONCEPT

// Count ops at compile time
constexpr int count_ops() {
    int n = 0;
    #define COUNT_ONE(op_name, tag_name) ++n;
    BLAZE_OP_LIST(COUNT_ONE)
    #undef COUNT_ONE
    return n;
}
static_assert(count_ops() >= 112,
    "Expected at least 112 registered Blaze ops");
```

### Runtime Checks

```python
def test_all_ops_registered():
    """Verify that all Blaze ops are accessible via ttnn.blaze."""
    from blaze.blaze_op import BlazeOp

    registered = set(ttnn.blaze.list_ops())
    in_blaze = set(BlazeOp._class_registry.keys())

    missing = in_blaze - registered
    assert not missing, f"Blaze ops not registered: {missing}"

def test_all_ops_callable():
    """Verify that every registered op is a callable TTNN operation."""
    for op_name in ttnn.blaze.list_ops():
        op_fn = getattr(ttnn.blaze, op_name)
        assert callable(op_fn), f"{op_name} is not callable"
        assert op_fn.__ttnn_operation__, \
            f"{op_name} is not a registered TTNN operation"

def test_no_name_collisions():
    """Verify that no registered Blaze op name collides
    with a native TTNN op name."""
    blaze_names = set(ttnn.blaze.list_ops())
    # Native TTNN ops are at ttnn.* (not ttnn.blaze.*)
    # The "blaze::" prefix in the registration string prevents
    # collision, but verify at the Python namespace level
    for name in blaze_names:
        assert not hasattr(ttnn, name) or \
               getattr(ttnn, name) is not getattr(ttnn.blaze, name), \
            f"Name collision: ttnn.{name} vs ttnn.blaze.{name}"
```

---

## 5.3.10 Full Registration Cost Summary

| Component | Lines of Code | Compile Cost | Runtime Cost |
|-----------|:---:|:---:|:---:|
| `BlazeDeviceOperationAdapter<OpTag>` template | ~200 (one-time) | Shared across all instantiations | None |
| `BlazeAdapterMeshProgramFactory` | ~50 (one-time) | Shared | None |
| `blaze_op_list.hpp` (112 entries) | ~120 | Preprocessed once | None |
| Generated tag structs (112) | ~224 | ~1 s | None |
| Generated registrations (112) | ~336 | ~5--15 s (template instantiation) | ~1 KB per-op metadata |
| Nanobind bindings (112) | ~112 | ~2--5 s | ~100 bytes per-op Python object |
| Runtime discovery | ~20 | Negligible | ~0.1 ms for `list_ops()` |
| CI validation script | ~50 (Python) | N/A (CI only) | N/A |
| **Total** | **~1,112** | **~15--25 s full rebuild** | **~100 KB metadata** |

Compared to the naive per-op approach (~53,000 lines, estimated in [Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)), the batch registration strategy achieves a **47x reduction in C++ code** and a **multi-month reduction in engineering effort**.

---

| [Section 5.2: FusedOp Registration](./02_fusedop_registration_walkthrough.md) | **Section 5.3** | [Chapter 6: Caching, Tracing, and Profiling](../ch06_caching_tracing_profiling/index.md) |
|:---|:---:|---:|
