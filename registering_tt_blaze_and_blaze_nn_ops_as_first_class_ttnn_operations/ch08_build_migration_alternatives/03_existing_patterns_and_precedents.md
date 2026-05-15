# 8.3 Existing Patterns and Precedents

This section surveys existing patterns within the TT-Metal / TTNN / TT-Blaze ecosystem that provide precedent or alternatives for registering Blaze ops as first-class TTNN operations. Each precedent is presented as a scenario template -- "if the Blaze registration project followed the same path as precedent X, what would the result look like?" -- followed by a gap analysis identifying what works and what does not. Four alternative strategies are evaluated and compared, and cross-ecosystem patterns from PyTorch and XLA provide broader context.

---

## 8.3.1 Precedent: TTNN `experimental/` Operations

### Pattern Description

TTNN's `ttnn/cpp/ttnn/operations/experimental/` directory contains operations that are not yet promoted to the stable API. These ops follow the **exact same registration pattern** as stable ops:

- They define a struct satisfying `DeviceOperationConcept`
- They use `register_operation<"ttnn::experimental::op_name", OpStruct>()`
- They have nanobind bindings via `bind_registered_operation()`
- They appear in the `ttnn.experimental.*` Python namespace

There is no "lightweight" or "provisional" registration path. `experimental/` ops differ from stable ops only in namespace placement and API stability guarantees.

### Scenario Template

**If Blaze ops followed the `experimental/` pattern:** Each Blaze op would get a dedicated C++ `DeviceOperation` struct in `ttnn/cpp/ttnn/operations/experimental/blaze/`, with custom `operation_attributes_t`, custom validation, a custom program factory, and nanobind bindings. Total per-op cost: ~770 lines. For 112 ops: ~86,000 lines of C++.

### Gap Analysis

| Aspect | `experimental/` Pattern | Blaze Adapter |
|--------|:-:|:-:|
| Per-op C++ lines | ~770 | ~5 |
| Program factory | Custom per op | Reuses `GenericMeshProgramFactory` |
| Kernel source | Static (compiled into binary) | Dynamic (JIT via `kernel_codegen.py`) |
| Scaling to 112 ops | ~86,000 lines | ~1,100 lines |
| Maintenance burden | Dual-maintain Python + C++ | Python only (C++ is template) |

### What This Proves

1. **Namespace-based staging works.** The `ttnn::experimental::` prefix signals instability without requiring a separate registration mechanism. The proposed `ttnn::blaze::` namespace follows the same pattern.
2. **Full registration is the only path.** There is no halfway registration in TTNN today. An op either satisfies `DeviceOperationConcept` and goes through `register_operation<>`, or it does not exist as a named operation. This validates the adapter approach: we must satisfy the concept fully, even if the implementation delegates to `GenericMeshProgramFactory`.

**Applicability: 4/5.** Directly applicable for namespace pattern; impractical for per-op investment model.

---

## 8.3.2 Precedent: Moreh Operations

### Pattern Description

The `ttnn/cpp/ttnn/operations/moreh/` directory contains operations developed by an external team (Moreh) and integrated into TTNN. These ops follow standard `DeviceOperationConcept` registration:

```text
ttnn/cpp/ttnn/operations/moreh/
    moreh_adam/
        device/moreh_adam_device_operation.hpp
        device/moreh_adam_device_operation.cpp
        device/moreh_adam_program_factory.cpp
        moreh_adam.hpp
        moreh_adam.cpp
        moreh_adam_nanobind.cpp
    moreh_arange/
    moreh_bmm/
    ... (30+ operations)
```

### What This Proves

1. **External teams can register ops into TTNN.** The moreh pattern demonstrates that ops developed outside the core TTNN team can follow the full registration contract via the standard PR process.
2. **Per-op C++ investment is high.** Each moreh op is 300--800 lines of C++. With 30+ ops, this represents ~15,000--24,000 lines. This validates the "100+ ops problem" from [Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md).
3. **The pattern works but does not scale to Blaze's op count.**

| Metric | Moreh Pattern | Blaze Adapter Pattern |
|--------|:------------:|:--------------------:|
| Per-op C++ LoC | 300--800 | ~5 |
| Total C++ for 112 ops | 33,600--89,600 | ~625 |
| Per-op maintenance | C++ + Python dual | Python only |
| Registration fidelity | Full (native) | Full (via adapter) |
| Custom validation | Yes | Optional (Tier 3) |

**Applicability: 2/5.** Validates external team registration; per-op cost is impractical for Blaze's scale.

---

## 8.3.3 Precedent: CCL Operations (Collective Communication)

### Pattern Description

Collective Communication Library (CCL) operations in `ttnn/cpp/ttnn/operations/ccl/` are the closest analog to Blaze's multi-device dispatch. They use `MeshWorkloadFactoryConcept` with `create_mesh_workload()` and per-coordinate `create_at()`:

```cpp
// EXISTING: CCL reduce_scatter uses MeshWorkloadFactoryConcept
struct ReduceScatterWidthMeshWorkloadFactory {
    static CachedMeshWorkload<shared_variables_t> create_mesh_workload(
        const operation_attributes_t& attrs,
        const MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        MeshWorkloadFactory& factory);

    static void create_at(
        MeshWorkloadFactory& factory,
        const MeshCoordinate& coord,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        shared_variables_t& shared_variables);
};
```

### What This Proves

1. **Mesh-aware factories are a proven pattern.** CCL ops demonstrate that `MeshWorkloadFactoryConcept` works for operations that need per-coordinate program specialization -- exactly what Blaze's `MeshProgramDescriptor` provides.
2. **`create_at()` per-coordinate dispatch is compatible with Blaze's model.** Blaze's `MeshProgramDescriptor` maps `MeshCoordinateRange` to `ProgramDescriptor`. CCL's `create_at()` receives a `MeshCoordinate` and builds the program for that coordinate. The adapter's `BlazeAdapterMeshProgramFactory` maps between these two interfaces.
3. **`override_runtime_arguments` for multi-device programs is validated.** CCL ops handle runtime argument updates across multiple devices, proving the pattern works at scale.

**Applicability: 5/5.** Directly applicable and architecturally aligned. The closest existing precedent for the adapter's mesh workload factory wrapper.

### Precedent-to-Adapter Design Mapping

| CCL Design Decision | Adapter's Corresponding Decision | Where Referenced |
|---------------------|----------------------------------|:----------------:|
| `MeshWorkloadFactoryConcept` with `create_at()` | `BlazeAdapterMeshProgramFactory::create_at()` | [Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md) |
| Per-coordinate shared variables | Adapter passes `MeshProgramDescriptor` as shared state | [Section 4.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md) |
| `override_runtime_arguments()` iteration | Same iteration pattern, delegated to `GenericMeshProgramFactory` | [Section 5.2](../ch05_op_registration/02_fusedop_registration_walkthrough.md) |
| Per-op `register_operation<>` | Parameterized `register_operation<>` via `OpTag` | [Section 5.1](../ch05_op_registration/01_microop_registration_walkthrough.md) |

---

## 8.3.4 Precedent: `model_tracer/generic_ops_tracer.py`

### Pattern Description

TT-Blaze includes a `model_tracer` module that wraps `generic_op` calls and extracts op identity from the Python call stack:

```python
# EXISTING (conceptual): model_tracer/generic_ops_tracer.py
class GenericOpsTracer:
    def trace_generic_op(self, io_tensors, descriptor):
        frame = inspect.currentframe().f_back.f_back
        op_name = self._extract_op_name(frame)
        self._trace_log.append({"op": op_name, "timestamp": time.time()})
        return ttnn.generic_op(io_tensors, descriptor)
```

### What This Proves

1. **The need for op identity is already recognized.** The tracer exists specifically because `generic_op` loses op names.
2. **Python-side identity extraction works for debugging but not for production.** The tracer adds ~10--50 us overhead per call for frame inspection and cannot propagate identity to Tracy, the program cache, or graph tracing.
3. **The adapter is the architectural solution to the problem this tracer works around.** Where the tracer post-hoc infers identity from the call stack, the adapter pre-registers identity at the type level.

**Applicability: 3/5.** Validates the motivation. Does not provide a reusable implementation pattern.

---

## 8.3.5 Precedent: `sweep_framework/load_ttnn_ops_data_v2.py`

### Pattern Description

The TTNN sweep framework dynamically discovers all registered operations for automated testing:

```python
# EXISTING (conceptual): sweep_framework/load_ttnn_ops_data_v2.py
def load_all_ttnn_ops():
    ops = {}
    for name, obj in inspect.getmembers(ttnn):
        if hasattr(obj, "__ttnn_operation__"):
            ops[name] = {"callable": obj, "name": obj.__ttnn_operation__}
    return ops
```

### What This Proves

1. **Registered ops are automatically discoverable.** Any op registered via `register_operation<>` and bound via `bind_registered_operation()` exposes a `__ttnn_operation__` attribute.
2. **Blaze ops registered via the adapter will be automatically included in sweep tests.** Once `ttnn.blaze.matmul` exists, the sweep framework discovers it without modification.
3. **The `ttnn.blaze.list_ops()` runtime discovery API** ([Section 5.3](../ch05_op_registration/03_batch_registration_and_scaling.md)) is complementary to sweep framework discovery.

**Applicability: 4/5.** Validates the benefit of registration for automated testing infrastructure.

---

## 8.3.6 Alternative Strategies

### Alternative 1: Enhanced `generic_op` Only (Phase 0 Extended)

**Mechanism:** Add `op_name` to `MeshProgramDescriptor`, modify profiler hooks to display it, and add op-name-prefixed hashing -- but never create per-op adapter types. This is Phase 0 treated as the final state.

**Estimated effort:** ~35 lines, 0.5 engineering-weeks.

**Delivers:** Profiler visibility, partial cache isolation.

**Does not deliver:** Per-op `type_hash` isolation, `compute_output_specs`, `registered_operation_t` Python callables, sweep framework discovery, graph tracing node differentiation, structured cache keys.

### Alternative 2: Python-Only Composite Wrappers

**Mechanism:** Register Python composite operations that call `generic_op` internally:

```python
@ttnn.register_python_composite("ttnn.blaze.matmul")
def blaze_matmul(io_tensors, descriptor):
    return ttnn.generic_op(io_tensors, descriptor)
```

**Estimated effort:** ~200 lines, 1 engineering-week.

**Delivers:** Named Python callables, some profiler visibility (Python-level tracing).

**Does not deliver:** C++-level Tracy profiling, program cache integration, `DeviceOperationConcept` conformance, graph tracing, Inspector annotations. The `device_operation::launch<>` path is not exercised.

### Alternative 3: Plugin/Extension System (`ttnn.register_external_op()`)

**Mechanism:** A hypothetical runtime registration API:

```python
ttnn.register_external_op(
    name="ttnn::blaze::matmul",
    factory=lambda attrs, tensors: ttnn.generic_op(tensors, attrs.descriptor),
    compute_output_specs=lambda attrs, tensors: tensors[-1].spec,
    compute_hash=lambda attrs, tensors: hash((type_tag, attrs.custom_hash)),
)
```

**Estimated effort:** ~4,000--5,000 lines for the framework, ~20 lines per op registration.

**Delivers:** Fully dynamic registration, no C++ per-op code, third-party ecosystem potential.

**Does not deliver:** Compile-time type safety, `constexpr` operation identity, friend-injection name uniqueness. Requires fundamental changes to TTNN's `register_operation<>` model.

### Alternative 4: Full C++ Port of `emit()` (Deep Integration)

**Mechanism:** Rewrite each Blaze op's `emit()` logic as a C++ program factory. See [Section 8.2.7](02_migration_plan.md) for the full effort comparison (adapter migration vs deep integration).

**Delivers:** Maximum integration -- native `DeviceOperationConcept` with custom validation, semantic hashing, typed attributes. No adapter indirection.

**Does not deliver:** Python JIT flexibility. Blaze's rapid iteration model is fundamentally incompatible with static C++ factories.

---

## 8.3.7 Comprehensive Comparison Table

Each strategy is rated on seven dimensions (1--5 scale, 5 = best):

| Dimension | Enhanced generic_op (Alt 1) | Python Wrappers (Alt 2) | Plugin System (Alt 3) | Full C++ Port (Alt 4) | **Adapter Pattern** |
|-----------|:--:|:--:|:--:|:--:|:--:|
| **Profiler Visibility** | 3 | 2 | 4 | **5** | **5** |
| **Program Cache Integration** | 2 | 1 | 4 | **5** | 4 |
| **Graph Tracing** | 1 | 1 | 4 | **5** | **5** |
| **Development Effort** | **5** | 4 | 2 | 1 | 4 |
| **Maintenance Burden** | **5** | 4 | 3 | 1 | 4 |
| **Registration Fidelity** | 1 | 2 | 3 | **5** | 4 |
| **Extensibility** | 2 | 3 | **5** | 2 | 3 |
| **Average Score** | 2.71 | 2.43 | 3.57 | 3.43 | **4.14** |

### Scoring Rubric

- **Profiler Visibility:** 5 = distinct zones per op. 1 = all ops appear as one name.
- **Program Cache Integration:** 5 = per-op type_hash + typed attributes. 1 = shared generic_op hash.
- **Graph Tracing:** 5 = per-op named nodes with output spec inference. 1 = all ops collapsed into one node type.
- **Development Effort:** 5 = <1 week total. 1 = >40 weeks total.
- **Maintenance Burden:** 5 = zero per-op maintenance. 1 = dual C++/Python maintenance per op.
- **Registration Fidelity:** 5 = indistinguishable from native TTNN op. 1 = not registered at all.
- **Extensibility:** 5 = fully dynamic. 1 = requires full rebuild + C++ per op.

### Key Observations

1. **The adapter pattern scores highest overall (4.14/5.0).** It delivers near-native registration fidelity at a fraction of the effort of full C++ porting.
2. **Alternative 1 (enhanced generic_op) is the wrong stopping point** but the correct *starting* point (Phase 0).
3. **Alternative 3 (plugin system) scores well but requires ~4,000 LoC of new TTNN infrastructure.** Not justified until third-party demand emerges.
4. **Alternative 4 (full C++ port) has maximum fidelity but minimum practicality.** The 40--60 week effort and ongoing dual-maintenance rule it out.

---

## 8.3.8 Cross-Ecosystem Patterns

For broader context, two patterns from outside the Tenstorrent ecosystem are worth noting:

### PyTorch Custom Op Registration

PyTorch's `torch.library` allows registering custom operators at runtime:

```python
@torch.library.custom_op("mylib::matmul", mutates_args=())
def my_matmul(x: Tensor, y: Tensor) -> Tensor:
    return x @ y

@my_matmul.register_fake
def _(x, y):
    return torch.empty(x.shape[0], y.shape[1])
```

This is analogous to Alternative 3 (plugin system). PyTorch achieves it via a runtime dispatch table rather than compile-time template registration. TTNN's `register_operation<>` uses compile-time NTTPs, making the PyTorch pattern not directly portable.

### XLA Custom Calls

XLA allows registering custom calls that bridge Python-defined computations to compiled executables:

```python
xla_client.register_custom_call_target("blaze_matmul", matmul_impl)
```

This is closer to TTNN's model in that the registration associates a string name with a callable. However, XLA custom calls do not participate in XLA's optimization passes -- analogous to how `generic_op` dispatch does not participate in TTNN's graph tracing optimization.

### Lesson

Both ecosystems confirm that **runtime-registered ops sacrifice optimization integration**. Full participation in the framework's optimization passes (XLA's HLO rewrites, TTNN's graph tracing, PyTorch's `torch.compile`) requires compile-time registration with structured metadata. The adapter pattern achieves this for Blaze without per-op C++.

---

| [Section 8.2: Migration Plan](02_migration_plan.md) | **Section 8.3** | [Section 8.4: Open Questions](04_open_questions.md) |
|:---|:---:|---:|
