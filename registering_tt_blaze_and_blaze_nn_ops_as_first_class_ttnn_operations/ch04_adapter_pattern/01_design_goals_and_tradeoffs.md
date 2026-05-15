# 4.1 Design Goals and Tradeoffs

The seven translation challenges from [Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md) converge on a single design requirement: a C++ template that accepts a pre-built `MeshProgramDescriptor` from Python, satisfies `DeviceOperationConcept` structurally, and injects op-specific identity without requiring per-op factory code. But there is more than one way to satisfy that requirement. At one extreme, a single generic adapter template handles every Blaze op uniformly; at the other, each op gets a hand-written `DeviceOperation` struct with bespoke attributes, validation, and hashing. Between these poles lies a hybrid space. This section analyzes the tradeoffs across six dimensions, recommends a three-tier hybrid strategy, provides a decision matrix for choosing between approaches, quantifies hash cost differences, estimates per-op and per-tier effort, and establishes the backward-compatibility constraints that the adapter must satisfy.

---

## 4.1.1 The Two Extremes

### Approach A: Generic Adapter Template

A single C++ class template, `BlazeDeviceOperationAdapter<OpTag>`, parameterized by a lightweight tag struct:

```cpp
// PROPOSED: Generic adapter -- one template covers all Blaze ops
template <typename OpTag>
struct BlazeDeviceOperationAdapter {
    // operation_attributes_t wraps MeshProgramDescriptor + op identity
    // tensor_args_t wraps io_tensors + output_tensor
    // program_factory_t = std::variant<BlazeAdapterMeshProgramFactory>
    // All methods delegate to GenericOpDeviceOperation logic or passthrough
};

// Per-op cost: one tag struct + one registration
struct MatmulTag { static constexpr const char* name = "blaze::matmul"; };
constexpr auto blaze_matmul =
    register_operation<"ttnn::blaze::matmul",
                        BlazeDeviceOperationAdapter<MatmulTag>>();
```

Each `OpTag` instantiation produces a **distinct C++ type**. The framework's `type_hash<BlazeDeviceOperationAdapter<MatmulTag>>` differs from `type_hash<BlazeDeviceOperationAdapter<RMSNormTag>>`, giving each op its own program cache namespace, profiler identity, and graph-trace node name. But all instantiations share the same structural logic: pass-through output management, descriptor-based hashing, minimal validation, and `GenericMeshProgramFactory` as the sole factory.

The adapter does not call `emit()`, does not construct programs, and does not know anything about the op's semantics. It is a *metadata injector*: it wraps the pre-built `MeshProgramDescriptor` in a typed struct that gives it a name, a hash prefix, a validation hook, and output spec exposure.

### Approach B: Per-Op Dedicated Wrappers

Each Blaze op gets a hand-written C++ struct satisfying `DeviceOperationConcept`:

```cpp
// ALTERNATIVE: Dedicated struct for matmul
struct BlazeMatmulDeviceOperation {
    struct operation_attributes_t {
        MeshProgramDescriptor descriptor;
        uint32_t M, N, K;           // Semantic dimensions
        DataType input_dtype;        // For validation
        bool transpose_b;           // For hash optimization
    };

    struct tensor_args_t {
        const Tensor& input_a;
        const Tensor& input_b;
        const Tensor& output;
    };

    using spec_return_value_t   = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t     = std::variant<GenericMeshProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& attrs, const tensor_args_t& args) {
        // Op-specific validation: shape compatibility, dtype checks, etc.
        TT_FATAL(args.input_a.get_logical_shape()[-1] == attrs.K,
            "Matmul inner dimension mismatch");
    }

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& args) {
        // Semantic hash: skip expensive descriptor walk
        return tt::stl::hash::hash_objects_with_default_seed(
            tt::stl::hash::type_hash<BlazeMatmulDeviceOperation>,
            attrs.M, attrs.N, attrs.K, attrs.input_dtype, attrs.transpose_b);
    }

    // ... remaining methods (~200-400 lines total) ...
};
```

This gives full control over validation, hashing, output spec inference, and profiling metadata. But it requires 250--850 lines per op and creates a C++ maintenance obligation for every `emit()` change in Python.

---

## 4.1.2 Six-Dimension Tradeoff Analysis

### Dimension 1: Development Velocity

| Factor | Generic Adapter | Per-Op Wrapper |
|--------|:-:|:-:|
| Lines of C++ per new op | ~5 (tag + registration) | 250--850 |
| Time to register a new MicroOp | Minutes | Hours to days |
| Time to register a new FusedOp | Minutes | Days to weeks |
| Requires C++ expertise from op author | No | Yes |
| Total for 112 ops | ~600 lines + template | ~53,000 lines |

**Verdict:** The generic adapter wins overwhelmingly on velocity. The difference is approximately two orders of magnitude. For a team registering dozens of ops, this is the difference between a one-week sprint and a multi-quarter project.

### Dimension 2: Type Safety and Validation

| Factor | Generic Adapter | Per-Op Wrapper |
|--------|:-:|:-:|
| Compile-time concept conformance | Yes (via template) | Yes (per struct) |
| Per-op runtime validation | No (delegates to `GenericOpDeviceOperation`) | Yes (custom `validate_on_program_cache_miss`) |
| Input shape/dtype checking | No | Yes |
| Bug detection at dispatch time | Limited to coordinate-range duplicates | Full semantic pre-dispatch checks |

**Verdict:** Per-op wrappers are strictly superior for validation. However, the generic adapter still provides the structural safety that `generic_op` lacks -- distinct type identity prevents cross-op cache collisions. The practical question is whether per-op validation catches bugs that Blaze's own Python-side validation (in `BlazeProgram.validate()`) misses. For most ops, Blaze's validation is already comprehensive; the C++ layer would be redundant defense-in-depth.

### Dimension 3: Program Cache Granularity

Both approaches achieve per-op cache isolation through distinct `type_hash` values. The difference lies in hash *efficiency*:

| Factor | Generic Adapter | Per-Op Wrapper |
|--------|:-:|:-:|
| Cache namespace isolation | Yes (`type_hash` per `OpTag`) | Yes (`type_hash` per struct) |
| Hash cost on cache hit | $O(n)$ descriptor walk (or custom hash) | $O(1)$ semantic attribute hash |
| Over-hashing (runtime args in hash) | Possible (descriptor walk includes RT arg counts) | Avoidable (hash only compile-time attributes) |

**Hash Cost Quantification.** Let $T_{\text{hash}}^{\text{generic}}$ be the time for a full descriptor hash and $T_{\text{hash}}^{\text{semantic}}$ be the time for a semantic hash:

$$T_{\text{hash}}^{\text{generic}} \approx (3 \times T_{\text{kernel}} + 8 \times T_{\text{cb}} + 3 \times T_{\text{sem}}) \approx 2\text{--}5\;\mu\text{s}$$

$$T_{\text{hash}}^{\text{semantic}} \approx 5 \times T_{\text{field}} \approx 50\text{--}100\;\text{ns}$$

The difference is ~20--100x, but this is measured against a kernel dispatch cost of 10--100 $\mu$s (for `enqueue_mesh_workload`). At 2--5% of total dispatch time, the hash overhead is unlikely to be the bottleneck unless ops are dispatched at very high frequency with small kernels.

**Verdict:** Per-op wrappers win on hash efficiency, but the practical impact is small. The generic adapter supports `custom_program_hash` as a middle ground -- Blaze computes a semantic hash in Python and passes it through, avoiding the descriptor walk without any C++ changes per op.

### Dimension 4: Profiling and Observability

| Factor | Generic Adapter | Per-Op Wrapper |
|--------|:-:|:-:|
| Unique name in Tracy timeline | Yes (`"ttnn::blaze::matmul"`) | Yes (`"ttnn::blaze::matmul"`) |
| Unique name in graph trace | Yes | Yes |
| Semantic attributes in profiler JSON | No (only descriptor metadata) | Yes (custom attribute reflection) |
| Performance model (`create_op_performance_model`) | No | Possible (per-op compute/bandwidth model) |

Both approaches solve the primary observability problem: Blaze ops stop appearing as `"ttnn::generic_op"` in every diagnostic tool. The generic adapter provides named identity; per-op wrappers additionally expose semantic attributes. For most ops, the op name plus input tensor shapes (already available via `tensor_args_t`) is sufficient.

**Verdict:** Tie for the common case. Per-op wrappers win for ops where detailed profiling is critical (e.g., matmul, attention), but named identity alone closes 80%+ of the observability gap.

### Dimension 5: Python API Ergonomics

| Factor | Generic Adapter | Per-Op Wrapper |
|--------|:-:|:-:|
| Python call signature | `ttnn.blaze.matmul(io_tensors, descriptor)` | `ttnn.blaze.matmul(a, b, output, *, transpose_b=False)` |
| Tensor arg naming | Flat `io_tensors` vector | Named args (`input_a`, `input_b`, `output`) |
| IDE autocompletion | Limited (generic signature) | Full (typed parameters) |

**Verdict:** Per-op wrappers win at the C++ binding level, but Python-side wrappers can close much of the gap without any C++ complexity. This is limited in practice because Blaze ops are typically called through `BlazeCompiler` or `blaze-nn` rather than directly by users.

### Dimension 6: Maintenance Cost

| Factor | Generic Adapter | Per-Op Wrapper |
|--------|:-:|:-:|
| C++ code to maintain per op | 0 (tag struct only) | 250--850 lines |
| Impact of `emit()` changes | None (descriptor interface is stable) | Must update C++ validation/hash |
| Dual-language synchronization | Not needed | Required (Python emit <-> C++ attrs) |
| Risk of C++ / Python divergence | None | High (semantic attrs can drift from emit logic) |

**Verdict:** The generic adapter wins decisively. Maintenance cost is the single strongest argument for the generic approach. Blaze ops evolve rapidly -- new CT args, changed CB layouts, additional ports -- and each change would require a corresponding C++ update in a per-op wrapper. The generic adapter is insulated because it operates on the *output* of the Blaze pipeline (the `MeshProgramDescriptor`), not on its internal structure.

---

## 4.1.3 Tradeoff Summary

| Dimension | Generic Adapter | Per-Op Wrapper | Impact of Difference |
|-----------|:-:|:-:|---|
| Development velocity | ++ | -- | Dominant for scaling to 112+ ops |
| Type safety | - | ++ | Mitigated by type-distinct adapter instantiations |
| Program cache granularity | + | ++ | Small if `custom_program_hash` is used |
| Profiler resolution | + | ++ | Matters only for high-value profiled ops |
| Python API ergonomics | - | ++ | Limited impact (ops called through compiler) |
| Maintenance cost | ++ | -- | Dominant for long-term sustainability |

The two "++" dimensions for the generic adapter -- development velocity and maintenance cost -- are *scaling* dimensions: they multiply by the number of ops. The "++" dimensions for per-op wrappers -- type safety, profiler resolution, and ergonomics -- are *per-op quality* dimensions: they matter for individual high-value ops but do not compound across 112+ ops.

### Scope of the Analysis

This tradeoff analysis applies to registering Blaze ops *as they exist today*: Python-compiled programs that produce `MeshProgramDescriptor` objects. Three scenarios fall outside this analysis:

1. **Ops not yet compiled by Blaze.** If a future op bypasses `BlazeCompiler` entirely and targets TTNN directly, it does not need the adapter -- it can register as a native `DeviceOperation`.
2. **Ops requiring custom C++ kernels.** If an op needs a new compute kernel (not generated by Blaze's `emit()`), the kernel must be written separately. The adapter can still wrap the resulting descriptor, but kernel authoring is outside its scope.
3. **Multi-device orchestration beyond `MeshProgramDescriptor`.** The adapter assumes single-`MeshProgramDescriptor` dispatch. If future multi-device strategies require coordinating multiple descriptors per logical op invocation, the adapter's `create_program_factory` would need extension.

These boundaries are revisited in [Chapter 8](../ch08_build_migration_alternatives/index.md).

---

## 4.1.4 The Three-Tier Hybrid Strategy

Neither extreme is optimal in isolation. The recommended strategy is a **three-tier hybrid** that matches engineering investment to the value delivered by each tier:

**Tier 1: Generic Adapter (default for all ops)**

All 112+ Blaze MicroOps and FusedOps are registered via `BlazeDeviceOperationAdapter<OpTag>`. This provides:
- Named identity in cache, profiler, and graph tracer
- Per-op cache isolation
- Zero per-op C++ maintenance
- Immediate coverage of the entire catalog

**Tier 2: Enhanced Generic Adapter (for ops needing better hashing)**

For ops where the $O(n)$ descriptor hash is measurably expensive, add `custom_program_hash` support in Python:
- Blaze's `BlazeCompiler` computes a semantic hash from `OpSpec` metadata
- The hash is stored in `ProgramDescriptor.custom_program_hash`
- The adapter's `compute_program_hash` detects and uses the custom hash
- Zero C++ changes per op; only Python-side hash computation

**Tier 3: Per-Op Wrapper (for high-value ops)**

For a small number of ops where all of the following are true:
- The op dominates runtime in production workloads (e.g., matmul, SDPA)
- Detailed profiling attributes are needed (compute vs bandwidth time)
- Custom validation catches bugs not caught by Blaze's Python validation
- A stable `operation_attributes_t` can be defined without frequent changes

Estimated candidates for Tier 3: 3--5 ops out of 112+.

---

## 4.1.5 Decision Matrix

The following matrix helps engineers decide which tier to use for a given op:

```
Start
  |
  +-- Does the op need named identity? ----NO----> Stay on generic_op (no change)
  |
  YES
  |
  +-- Is descriptor hash a measured bottleneck? ----YES----> Tier 2 (custom hash)
  |
  NO
  |
  +-- Does the op need custom C++ validation    ----YES----> Tier 3 (per-op wrapper)
  |   OR semantic profiling attributes
  |   OR a performance model?
  |
  NO
  |
  +-- Tier 1 (generic adapter)
```

**Expanded criteria table:**

| Criterion | Tier 1 (Generic) | Tier 2 (Enhanced Hash) | Tier 3 (Per-Op Wrapper) |
|-----------|:-:|:-:|:-:|
| Named identity in profiler needed? | Yes | Yes | Yes |
| Per-op cache isolation needed? | Yes | Yes | Yes |
| Hash cost is measurable bottleneck? | No | Yes | N/A (semantic hash) |
| Custom C++ validation needed? | No | No | Yes |
| Semantic profiling attributes needed? | No | No | Yes |
| Performance model needed? | No | No | Yes |
| Op dominates production runtime? | No | Maybe | Yes |
| C++ engineer available for maintenance? | N/A | N/A | Yes |

**Rule of thumb:** Start with Tier 1. Promote to Tier 2 when profiling shows hash overhead. Promote to Tier 3 only when profiling data or bug reports demonstrate a concrete need for deeper integration.

---

## 4.1.6 Effort Estimates by Phase

### Phase 0: Infrastructure (~1 week)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Implement `BlazeDeviceOperationAdapter<OpTag>` template | 2--3 days | Single header (~200 lines incl. factory wrapper) |
| Implement `BLAZE_OP_LIST(X)` X-macro and registration macros | 0.5 days | Macro header |
| Implement `ttnn.blaze` nanobind module | 1 day | Module with `bind_blaze_op` calls |
| Modify `_run_program()` for named dispatch | 0.5 days | Python change in `compiler.py` |
| Unit tests for adapter template | 1--2 days | C++ tests + Python integration tests |

### Phase 1: Tier 1 Registration (~1 week)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Write X-macro list for all ops | 0.5 days | `blaze_op_list.hpp` |
| Generate/verify tag structs + registrations | 0.5 days | One `.hpp` file |
| Generate nanobind bindings | 0.5 days | Auto-generated from X-macro |
| Integration tests (profiler shows named ops) | 2 days | End-to-end test suite |

### Phase 2: Tier 2 Enhancement (~3 days per op)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Add `custom_program_hash` computation to Blaze op | 1 day | Python-side hash in `emit()` or `compile()` |
| Verify cache hit rate improvement | 1 day | Benchmark before/after |
| Regression tests | 1 day | Hash stability tests |

### Phase 3: Tier 3 Per-Op Wrappers (as needed, ~1 week per op)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Define `operation_attributes_t` and `tensor_args_t` | 1 day | C++ header |
| Implement validation and hash | 1--2 days | C++ source |
| Implement performance model (optional) | 1--2 days | C++ source |
| Nanobind bindings with named args | 0.5 days | C++ source |
| Tests and benchmarks | 1--2 days | C++ and Python tests |

### Aggregate Comparison

| Approach | Per-Op Cost | Total for 112 Ops | Ongoing Maintenance |
|----------|------------|-------------------|-------------------|
| Generic adapter only (Tier 1) | ~5 lines, ~5 min | ~600 lines, ~1 week | Template-only |
| Per-op wrappers only | ~200--850 lines, ~2--8 days | ~53,000 lines, ~6--12 months | Per-op dual-maintenance |
| Hybrid (Tier 1 + ~8 Tier 2 + ~4 Tier 3) | variable | ~2,700 lines, ~4--5 weeks | Template + 4 per-op |

The hybrid approach achieves a **10--25x reduction** in total C++ code while providing full observability for the entire catalog and enhanced profiling for the most critical ops.

---

## 4.1.7 Backward Compatibility

### Requirements

The adapter must not break any existing code path. Five compatibility constraints apply:

**Constraint 1: `ttnn.generic_op()` must continue to work.** `GenericOp`, `GenericOpDeviceOperation`, and `GenericMeshProgramFactory` remain in the codebase unchanged. The adapter is a new parallel path, not a replacement.

**Constraint 2: Blaze compilation pipeline must be unchanged.** The adapter operates on the *output* of `BlazeCompiler.compile()` -- the `MeshProgramDescriptor`. It does not require changes to `BlazeOp`, `MicroOp`, `FusedOp`, `emit()`, `compose()`, or the engine pipeline.

**Constraint 3: Program cache entries must not collide.** The adapter's hash must be disjoint from `GenericOpDeviceOperation`'s hash for the same descriptor.

**Constraint 4: No framework modifications.** The adapter must satisfy the existing `DeviceOperationConcept` with no changes to the concept definition or the dispatch pipeline.

**Constraint 5: Opt-in adoption.** Until an op is explicitly registered, it continues to use the `ttnn.generic_op()` path.

### Hash Disjointness Proof

The adapter's hash incorporates `type_hash<BlazeDeviceOperationAdapter<OpTag>>`, computed from the mangled type name. Since `BlazeDeviceOperationAdapter<MatmulTag>` and `GenericOpDeviceOperation` are different types, their `type_hash` values are different:

$$
h_{\text{adapter}} = \text{hash}\bigl(\underbrace{\texttt{type\_hash}\langle\texttt{BlazeDeviceOperationAdapter}\langle\texttt{MatmulTag}\rangle\rangle}_{\text{unique per OpTag}},\; \texttt{descriptor}\bigr)
$$

$$
h_{\text{generic}} = \text{hash}\bigl(\underbrace{\texttt{type\_hash}\langle\texttt{GenericOpDeviceOperation}\rangle}_{\text{shared by all ops}},\; \texttt{descriptor}\bigr)
$$

$$
h_{\text{adapter}} \neq h_{\text{generic}} \quad \text{for all descriptors}
$$

The program cache can hold entries from both paths simultaneously. If an op is dispatched through both the adapter and `generic_op` (e.g., during migration), both cache entries coexist without collision. The programs are identical (same descriptor, same factory), so the only overhead is a duplicate cache entry -- not incorrect behavior.

### Migration Strategy

The backward-compatible design enables a phased migration where each phase is independently revertible and no phase requires a flag day:

**Phase A -- Transparent registration.** Register all ops via the adapter. `_run_program()` uses the adapter path for registered ops, falls back to `generic_op` for unregistered ones. Behavior is functionally identical; the only visible change is that Tracy timelines and graph traces show named ops instead of `"ttnn::generic_op"`. This phase can be deployed incrementally -- register 10 ops, validate in CI, register the next batch.

**Phase B -- Enhanced hashing and validation.** Enable `custom_program_hash` for Tier 2 ops and custom validation hooks for Tier 3 ops. Monitor for:
- Cache hit-rate changes (should improve or stay flat; any regression indicates a hash bug)
- Validation false positives (new `TT_FATAL` assertions in production workloads)
- Performance regressions (hash overhead changes, dispatch latency)

Each op's promotion from Tier 1 to Tier 2 or 3 is an independent change that can be reverted by removing the `custom_program_hash` field or the `validate()` hook from the tag struct.

**Phase C -- Deprecation of generic_op fallback.** Once all ops are registered and stable (measured by CI green rate and zero fallback invocations in production), deprecate the `generic_op` fallback path in `_run_program()`. The `generic_op` API itself remains in TTNN for non-Blaze users and other consumers; only Blaze's internal dispatch stops using it.

---

## 4.1.8 Design Goals Summary

| # | Goal | Rationale | Tier(s) |
|---|------|-----------|---------|
| G1 | Satisfy `DeviceOperationConcept` structurally | Required for `register_operation<>`, `launch<>`, cache, tracing, and profiling integration | All |
| G2 | Zero per-op C++ factory code | Scalability to 112+ ops; eliminates dual-language maintenance | 1, 2 |
| G3 | Per-op identity in cache, profiler, and graph trace | Primary motivation for first-class registration; each op has a unique `type_hash`, name in Tracy, and node in the graph tracer | All |
| G4 | Reuse `GenericMeshProgramFactory` as sole factory | Avoids reimplementing program construction; leverages existing, tested code via `BlazeAdapterMeshProgramFactory` wrapper | 1, 2 |
| G5 | Preserve Blaze's pre-allocation model | Output tensors are already allocated by `_run_program()`; the adapter must pass them through, not re-allocate | All |
| G6 | Expose output specs for graph tracing | Enable `NO_DISPATCH` graph capture with meaningful output shapes via `compute_output_specs` | All |
| G7 | Support `custom_program_hash` when available | Enable $O(1)$ semantic hashing for high-frequency ops without C++ changes per op | 2, 3 |
| G8 | Backward compatible with `ttnn.generic_op()` | Existing code paths must continue to work; hash disjointness prevents cache collisions | All |
| G9 | Extensible for per-op validation hooks | Allow the ~5% of ops that need custom validation to add it via SFINAE-detected `OpTag::validate()` | 3 |
| G10 | Minimal template instantiation overhead | Avoid quadratic build-time growth; target < 1 MB binary increase for 112 ops | All |

[Section 4.2](02_blaze_device_operation_adapter.md) demonstrates how every goal is achieved simultaneously in the `BlazeDeviceOperationAdapter<OpTag>` template, and includes a cross-reference table mapping each goal to the specific method or design choice that satisfies it.

---

## Key Takeaways

- The ten design goals (G1--G10) are implemented in the `BlazeDeviceOperationAdapter<OpTag>` template in [Section 4.2](02_blaze_device_operation_adapter.md).
- The three-tier hybrid strategy and effort estimates guide the phased rollout described in [Section 4.3](03_python_dispatch_and_registration.md).
- Backward-compatibility constraints (Section 4.1.7) are verified end-to-end in [Chapter 6](../ch06_caching_tracing_profiling/index.md).

---

**Next:** [`02_blaze_device_operation_adapter.md`](./02_blaze_device_operation_adapter.md)
