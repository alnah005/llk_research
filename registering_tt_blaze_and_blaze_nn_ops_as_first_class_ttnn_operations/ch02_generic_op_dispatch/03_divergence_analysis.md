# 2.3 Divergence Analysis: Native TTNN Ops vs. Blaze via `generic_op`

## Opening Summary

This section provides a systematic, dimension-by-dimension comparison of how `ttnn.generic_op()` (the dispatch mechanism used by Blaze) differs from a native TTNN operation. Each divergence is documented with its structural cause, concrete cost, severity, and whether it is addressable without fundamental architecture changes. The divergences are classified into P0/P1/P2 priorities to guide the first-class registration effort described in later chapters. Understanding these gaps precisely is essential: they define the exact work required to promote Blaze ops from opaque `generic_op` invocations to named, typed, fully-integrated TTNN citizens.

---

## 2.3.1 The Comparison Framework

The TTNN device-operation contract (see [Chapter 1](../ch01_ttnn_registration_contract/index.md)) requires each operation to implement a set of static methods within a `DeviceOperation` struct. For each contract point, we compare:

- **Native TTNN op** (e.g., `ttnn::matmul`, `UnaryDeviceOperation`): Implements the method with full semantic logic.
- **Blaze via `generic_op`**: How `GenericOpDeviceOperation` handles the same point, and what Blaze provides (or does not provide) at the Python level.

Reference operations used throughout:

| Property | `ttnn.generic_op` (Blaze) | `ttnn.matmul` (Native) |
|----------|---------------------------|------------------------|
| Header | `generic_op.hpp` | `matmul.hpp` |
| Device operation | `GenericOpDeviceOperation` | `MatmulDeviceOperation` |
| `operation_attributes_t` | `MeshProgramDescriptor` | `MatmulParams` |
| `tensor_args_t` | `{io_tensors, output_tensor}` | `{input_a, input_b, bias, optional_output}` |
| `program_factory_t` | `variant<GenericMeshProgramFactory>` | `variant<6 specialized factories>` |

---

## 2.3.2 Master Divergence Table

| # | Contract Point | Native TTNN Op | Blaze via `generic_op` | Impact | Priority |
|---|---------------|----------------|----------------------|--------|----------|
| 1 | **Op identity** | Unique name per op: `"ttnn::matmul"` | Single name: `"ttnn::generic_op"` for all | High | P0 |
| 2 | **`operation_attributes_t`** | Semantic struct (shapes, dtypes, config) | Opaque `MeshProgramDescriptor` | High | P0 |
| 3 | **`compute_output_specs()`** | Infers output shape/dtype from inputs + attributes | Returns `output_tensor.tensor_spec()` passthrough | High | P0 |
| 4 | **`create_output_tensors()`** | Allocates output tensor on device | Returns caller's pre-allocated output unchanged | Medium | P1 |
| 5 | **`validate_on_program_cache_miss()`** | Full semantic validation (shapes, dtypes, memory, sharding) | Only checks duplicate `MeshCoordinateRange` | High | P0 |
| 6 | **`validate_on_program_cache_hit()`** | Lightweight re-validation | Same coord-uniqueness check | Medium | P1 |
| 7 | **`validate_inputs()`** | Static pre-dispatch validation | Declared but not implemented | Medium | P1 |
| 8 | **`compute_program_hash()`** | Semantic hash of $O(1)$ attribute fields | Content hash of full `ProgramDescriptor`: $O(\sum\|K_i\| + \sum\|CB_j\| + \sum\|S_k\|)$ | Medium | P1 |
| 9 | **`select_program_factory()`** | Selects among multiple factories (matmul has 6) | Single variant: `GenericMeshProgramFactory` | Low | N/A |
| 10 | **Output tensor count** | `tensor_return_value_t` can be `Tensor`, `vector<Tensor>`, or `optional<Tensor>` | Single `Tensor` return only | Medium | P1 |
| 11 | **Tracy / profiling** | Op name in profiling zones identifies the operation | All Blaze ops show as `"ttnn::generic_op"` | High | P0 |
| 12 | **Graph trace capture** | Each op records named attributes and tensor specs | Records only `num_mesh_programs` via reflection | High | P0 |
| 13 | **Performance model** | `create_op_performance_model()` for roofline analysis | Not implemented | Low | P2 |
| 14 | **`io_tensors` convention** | Typed struct with named fields (`input_a`, `bias`, etc.) | Flat `vector<Tensor>` with last-element-is-output convention | Medium | P1 |
| 15 | **Mesh device adaptation** | `MeshDeviceOperationAdapter` handles decomposition | Blaze manually decomposes via `_split_tensors()` | Low | P2 |

---

## 2.3.3 Detailed Analysis of High-Impact Divergences

### Divergence 1: Op Identity Collapse

The TTNN `register_operation` template stamps the operation name into the compiled binary at `constexpr` time:

```cpp
// Native: distinct identity per op
constexpr auto matmul = register_operation<"ttnn::matmul", ...>();
constexpr auto exp    = register_operation<"ttnn::exp", ...>();

// Generic: one identity for all Blaze ops
constexpr auto generic_op = register_operation<"ttnn::generic_op", GenericOp>();
```

The operation name flows into the Tracy profiler zone name, the TTNN graph capture system, the op log / reflection system, and error messages.

**Cost:** When profiling a Blaze model, all fused operations appear as `generic_op` in the timeline. If a model has 20 distinct Blaze ops, the profiler shows 20 `generic_op` entries with no way to distinguish them.

**Structural cause:** `generic_op` is a single registered operation. Multiple Blaze op types share this one registration because Blaze builds programs in Python and dispatches them through a single C++ entry point.

---

### Divergence 2--3: No Output Inference or Allocation

In a native op:

```cpp
// MatmulDeviceOperation (representative)
spec_return_value_t compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    auto output_shape = compute_matmul_output_shape(attrs, args);
    return {TensorSpec(output_shape, ...)};
}

tensor_return_value_t create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return {create_device_tensor(compute_output_specs(attrs, args)[0], ...)};
}
```

In `GenericOpDeviceOperation`:

```cpp
spec_return_value_t compute_output_specs(const operation_attributes_t&,
                                          const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();  // <-- passthrough
}

tensor_return_value_t create_output_tensors(const operation_attributes_t&,
                                             const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor;  // <-- passthrough
}
```

**Cost:** `generic_op` cannot participate in shape-inference chains. Graph tracing (`GraphTracker`) in `NO_DISPATCH` mode cannot infer output shapes without executing the op. The caller (Blaze) must pre-allocate the output with the correct shape, dtype, and memory config before calling `generic_op`.

**Structural cause:** Blaze needs raw `Buffer*` pointers for CB descriptors at compile time, before `generic_op` is called. Output allocation must happen in the Blaze compiler, not in the TTNN op.

**Consequence for first-class promotion:** A promoted Blaze op must implement `compute_output_specs()` that can infer the output shape from its semantic attributes. This requires extracting shape-inference logic from the Python op definition (which currently exists only as implicit knowledge in the `compose()` / `emit()` functions).

---

### Divergence 5: Absent Semantic Validation

Native ops validate extensively:

```cpp
void MatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    TT_FATAL(input_a.shape()[-1] == input_b.shape()[-2], ...);
    TT_FATAL(supports_dtype(input_a.dtype()), ...);
    // ... 50+ checks on shapes, dtypes, memory configs, sharding
}
```

`GenericOpDeviceOperation` checks only mesh topology:

```cpp
void validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    verify_no_duplicate_mesh_coord_ranges(attrs.mesh_programs);
}
```

**Cost:** Errors in CB sizing, kernel paths, tensor shapes, or compute config are not caught by `generic_op`. They surface as hardware-level failures (hangs, incorrect results, crashes) rather than clean Python exceptions.

**Structural cause:** `generic_op` receives an opaque `ProgramDescriptor`. It has no semantic knowledge of what the program does, so it cannot validate whether the descriptor is correct for the given tensors.

---

### Divergence 11--12: Profiling and Graph Trace Opacity

All Blaze ops appear as `"ttnn::generic_op"` in:

- **Tracy profiling:** The `register_operation` name is `"ttnn::generic_op"`, so all Blaze operations are grouped under one profiling label. A Tracy flame graph shows `generic_op` blocks without indicating whether they are RMSNorm, SDPA, MoE gate, or matmul operations.

- **TTNN graph serialization:** The `attribute_values()` method on `MeshProgramDescriptor` returns only the count:

```cpp
static constexpr auto attribute_names = std::forward_as_tuple("num_mesh_programs");
auto attribute_values() const { return std::make_tuple(mesh_programs.size()); }
```

A captured TTNN graph of a Blaze model is a flat sequence of `generic_op` nodes with no information about what each one computes. The graph cannot be optimized, meaningfully serialized, or compared across model versions.

**Addressable:** Blaze could inject Tracy user zones around `generic_op` calls as a workaround, or a future TTNN extension could allow callers to provide a display name. Full resolution requires registering each Blaze op type as its own TTNN operation.

---

## 2.3.4 Program-Cache Behavior Comparison

| Aspect | Native TTNN Op | Blaze via `generic_op` |
|--------|----------------|----------------------|
| **Hash inputs** | Semantic attributes ($O(1)$ fields like shapes, dtypes, config flags) | Full descriptor contents (kernel source strings, all CT/RT args, CB specs) |
| **Hash cost** | $O(1)$ | $O(\sum |K_i| + \sum |CB_j| + \sum |S_k|)$ where $K_i$ are kernel descriptors |
| **Cache sensitivity** | Changes to non-semantic fields (e.g., tensor address) do not invalidate cache | Any field change invalidates (unless `custom_program_hash` is set) |
| **RT-arg-only changes** | Cache hit; `override_runtime_arguments()` patches values | Cache hit (RT arg *values* are excluded from hash; only *counts* are hashed). If count changes: cache miss. |
| **Factory selection** | Factory variant is part of the program hash; a change in factory selection produces a cache miss, not a cache-hit factory swap | Single factory; no selection |
| **Custom hash opt-in** | Not applicable (semantic hash is the norm) | `ProgramDescriptor.custom_program_hash` field available but **unused by Blaze** |

### Quantifying Cache Pollution

Consider a Blaze fused-RMSNorm op called $N$ times with different tensor addresses but identical computation:

- **Native op hash:** 1 cache entry (addresses are not in the hash).
- **`generic_op` default hash:** Potentially $N$ cache entries if descriptor fields happen to differ (e.g., different `Buffer*` pointers feeding into CB descriptors). In practice, Blaze's hash is stable when only runtime arg *values* change (since values are excluded), but changes to CB `buffer` pointers or `total_size` will cause misses.
- **`generic_op` + `custom_program_hash`:** 1 entry (if Blaze pre-computes a semantic hash).

The `custom_program_hash` field in `ProgramDescriptor` was added specifically to address this gap, but Blaze has not yet adopted it.

---

## 2.3.5 What Blaze Does Right (Despite `generic_op`)

The Blaze compilation pipeline compensates for `generic_op`'s minimal contract in several important ways:

1. **CB validation in `BlazeProgram`.** The builder validates CB ID uniqueness, page size alignment, and `total_size` / `page_size` divisibility before constructing the `ProgramDescriptor`.

2. **CT arg validation.** `BlazeProgram.validate()` cross-references Python-supplied CT args against the kernel's C++ header declarations, catching missing or extra args before dispatch.

3. **Tensor lifetime pinning.** `MeshCompiledProgram._lifetime_tensors` and `_lifetime_semaphores` prevent GC of objects whose raw pointers are embedded in descriptors (see [Section 2.2.8](./02_blaze_compilation_pipeline.md#228-lifetime-management)).

4. **Per-device decomposition.** The compiler correctly handles mesh decomposition, ensuring each device gets a properly specialized `ProgramDescriptor` with device-specific tensor addresses.

5. **Named CT/RT args with generated headers.** The `UnifiedKernelDescriptor` generates C++ headers with `constexpr` accessors (e.g., `ct::matmul.out_w`), eliminating the error-prone positional-index pattern used by raw `generic_op` callers.

These compensations work, but they exist in a parallel universe to the TTNN framework. The framework cannot observe, enforce, or benefit from them.

---

## 2.3.6 Divergences That Are Neutral or Intentional

Not all divergences are problems. Some are intentional design choices:

| Divergence | Why It Is Acceptable |
|------------|---------------------|
| Single program factory | Blaze handles factory selection at the Python level. The `ProgramDescriptor` already encodes the specific program variant. |
| No `select_program_factory()` | Same reason -- the factory is implicitly selected during Blaze compilation. |
| No `skip_launch()` | Blaze graphs do not produce degenerate zero-size invocations. Skipping would save negligible overhead. |
| No backward/autograd support | Blaze targets inference-only workloads. Autograd is not a current requirement. |

---

## 2.3.7 Summary Divergence Matrix

| # | Dimension | Severity | Addressable? | Priority |
|---|-----------|----------|--------------|----------|
| 1 | Op identity | High | Yes -- register ops individually | P0 |
| 2 | `operation_attributes_t` | High | Yes -- add semantic wrapper | P0 |
| 3 | `compute_output_specs` | High | Yes -- move Blaze logic to C++ | P0 |
| 4 | `create_output_tensors` | Medium | Yes -- but requires architecture change | P1 |
| 5 | `validate_on_program_cache_miss` | High | Partially -- Blaze does own validation, framework integration missing | P0 |
| 6 | `validate_on_program_cache_hit` | Medium | Same as above | P1 |
| 7 | `validate_inputs` | Medium | Yes -- add implementation | P1 |
| 8 | `compute_program_hash` | Medium | Yes -- use `custom_program_hash` or semantic hash | P1 |
| 9 | `select_program_factory` | Low | N/A by design | N/A |
| 10 | Output tensor count | Medium | Yes -- define typed return | P1 |
| 11 | Profiling / Inspector | High | Partial -- user zones possible; full fix requires registration | P0 |
| 12 | Graph trace capture | High | Yes -- register ops individually | P0 |
| 13 | Performance model | Low | Could add later | P2 |
| 14 | `io_tensors` convention | Medium | Yes -- define typed `tensor_args_t` | P1 |
| 15 | Mesh device adaptation | Low | Could leverage `MeshDeviceOperationAdapter` | P2 |

---

## 2.3.8 Root Cause Analysis

The divergences stem from a single architectural decision: **Blaze builds programs in Python and hands them to TTNN as opaque descriptors.** This means:

1. TTNN's C++ infrastructure (shape inference, factory selection, validation, graph tracing, profiling) operates on `operation_attributes_t`, which for `generic_op` is a `MeshProgramDescriptor` -- structurally complete but semantically opaque.

2. All semantic information (op names, parameter meanings, tensor relationships) exists only in the Python layer and is lost at the `ttnn.generic_op()` boundary (the metadata discard point documented in [Section 2.2.9](./02_blaze_compilation_pipeline.md#229-the-metadata-discard-point)).

3. The path to first-class citizenship requires one of three approaches:

   **(a) Register each Blaze op as a separate TTNN operation.** Full integration, high effort. Each promoted op gets its own `DeviceOperation` struct with typed attributes, proper validation, output spec inference, and named identity. This resolves all P0 and P1 divergences.

   **(b) Extend `generic_op` to carry optional semantic metadata.** Partial integration, moderate effort. Add fields like `op_name`, `attribute_summary`, and `validation_fn` to `MeshProgramDescriptor` or a wrapper. This addresses identity and profiling (P0) but not output spec inference or typed tensor args.

   **(c) Extend TTNN's infrastructure to accept client-provided metadata alongside opaque descriptors.** New pattern, systemic change. Would allow `generic_op` callers to inject profiling names, validation hooks, and spec-inference callbacks without changing the op registration model.

The divergences in Dimensions 1, 3, 5, 11, and 12 (identity, output inference, validation, profiling, graph tracing) are the highest-severity gaps and the primary motivation for the registration effort described in [Chapter 3](../ch03_compilation_model_gap/index.md).

---

## 2.3.9 What Must Change for First-Class Promotion

To promote a Blaze op to a first-class TTNN operation, the following divergences must be resolved:

| Priority | Divergence | Resolution Path |
|----------|-----------|-----------------|
| **P0** | No output-spec inference (#3) | Implement `compute_output_specs()` with shape logic extracted from the Python op |
| **P0** | No semantic validation (#5) | Port Blaze's per-op validation into C++ `validate_on_program_cache_miss()` |
| **P0** | Profiling opacity (#11) | Register each promoted op with its own name (e.g., `"ttnn::blaze_rmsnorm"`) |
| **P0** | Graph trace opacity (#12) | Same -- per-op registration provides distinct graph nodes |
| **P0** | Opaque attributes (#2) | Define per-op `operation_attributes_t` with semantic fields |
| **P1** | No output allocation (#4) | Implement `create_output_tensors()` that allocates from inferred specs |
| **P1** | Content-based program hash (#8) | Implement semantic `compute_program_hash()` or adopt `custom_program_hash` |
| **P1** | Flat tensor vector (#14) | Define typed `tensor_args_t` with named fields |
| **P1** | Multi-output support (#10) | Define appropriate `tensor_return_value_t` |
| **P2** | No performance model (#13) | Optionally implement `create_op_performance_model()` |
| **P2** | Manual mesh adaptation (#15) | Optionally leverage `MeshDeviceOperationAdapter` |

These requirements are explored further in [Chapter 3: Compilation Model Gap](../ch03_compilation_model_gap/index.md).

---

## Key Takeaways

1. **15 divergence points** exist between the native TTNN op contract and the Blaze-via-`generic_op` path. Of these, **6 are P0** (identity, attributes, output inference, validation, profiling, graph tracing) and **6 are P1** (output allocation, validation on hit, validate_inputs, hash, multi-output, tensor args).

2. **The divergences are by design, not accident.** `generic_op` was built to be a minimal dispatch point, and Blaze intentionally owns all semantic logic. The cost is invisibility to TTNN infrastructure.

3. **Blaze compensates effectively** at the Python level -- CB validation, CT arg checking, lifetime pinning, and named arg headers all work. But these compensations are invisible to TTNN, creating a parallel safety system that the framework cannot enforce or benefit from.

4. **Program-cache pollution is the most actionable short-term gap.** Adopting `custom_program_hash` in Blaze would eliminate redundant cache entries without requiring any C++ changes.

5. **Three resolution paths exist:** (a) per-op registration (full integration), (b) metadata extension on `generic_op` (partial), or (c) TTNN infrastructure extension (systemic). Path (a) resolves all divergences but requires the most engineering effort. Path (b) addresses the highest-severity gaps with moderate effort. The choice between them is the subject of [Chapter 3](../ch03_compilation_model_gap/index.md).

---

| [Section 2.2: Blaze Compilation Pipeline](02_blaze_compilation_pipeline.md) | **Section 2.3** | [Chapter 3: Compilation Model Gap](../ch03_compilation_model_gap/index.md) |
|:---|:---:|---:|
