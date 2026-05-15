# 1.1 The DeviceOperationConcept

Every native TTNN device operation must satisfy a C++20 concept called `DeviceOperationConcept`. This concept is the foundational contract between an operation author and the TTNN runtime: if a struct provides the right type aliases and static methods, the framework knows how to validate, hash, create outputs, select a program factory, build programs, cache them, and dispatch them to hardware. No base class is involved -- conformance is entirely structural. This section dissects the concept line by line, explains every required and optional member, describes the two program factory concepts, introduces the `CachedProgram` type family, and closes with a concrete example drawn from `BinaryDeviceOperation`.

---

## 1.1.1 The Five Type Aliases

A struct `Op` satisfying `DeviceOperationConcept` must expose exactly five nested type aliases. Together they form a data contract that the framework uses to thread information through every stage of dispatch.

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp
struct Op {
    using operation_attributes_t = /* user-defined struct */;
    using tensor_args_t          = /* user-defined struct */;
    using spec_return_value_t    = /* TensorSpec or vector/tuple thereof */;
    using tensor_return_value_t  = /* Tensor or vector/tuple thereof */;
    using program_factory_t      = std::variant<Factory1, Factory2, ...>;
};
```

| Alias | Purpose | Typical Concrete Type |
|-------|---------|----------------------|
| `operation_attributes_t` | All non-tensor configuration (op type enum, memory config, dtype, compute kernel config, scalar parameters). Must be hashable via `tt::stl::hash`. | A plain struct with an optional `to_hash()` member. |
| `tensor_args_t` | References to input (and optionally pre-allocated output) tensors. Passed by `const&` throughout dispatch. | A struct of `const Tensor&` and `std::optional<Tensor>` fields. |
| `spec_return_value_t` | The *shape metadata* of the output, returned by `compute_output_specs`. Used during graph tracing when no actual allocation is needed. | `TensorSpec`, `std::vector<TensorSpec>`, or `std::tuple<TensorSpec, ...>`. |
| `tensor_return_value_t` | The actual output tensor(s), returned by `create_output_tensors`. | `Tensor`, `std::vector<Tensor>`, or `std::tuple<Tensor, ...>`. |
| `program_factory_t` | A `std::variant` of one or more program factory types. Each alternative must satisfy either `ProgramFactoryConcept` or `MeshWorkloadFactoryConcept` (but not both, enforced by `AllFactoriesValid`). | `std::variant<ElementWiseMultiCore, BroadcastHeightMultiCore, ...>`. |

---

## 1.1.2 Required Static Methods

The concept requires three static methods that the framework calls unconditionally:

### `validate_on_program_cache_miss`

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (line 107)
static void validate_on_program_cache_miss(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args);
```

Called on every cache miss (i.e., the first invocation with a given hash) **before** program creation. This is the place for shape checks, dtype compatibility assertions, memory layout constraints, and any other invariant that must hold before a kernel is compiled and programs are built.

### `create_output_tensors`

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (line 110-111)
static tensor_return_value_t create_output_tensors(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args);
```

Allocates the output tensor(s) on device. Called on every invocation, both cache hit and cache miss, because output buffers must always be freshly allocated (they might be different sizes, on different addresses, etc.). The framework calls this *before* entering the program factory.

### `compute_output_specs`

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 56-63)
template <typename device_operation_t>
concept HasComputeOutputSpecs = requires(...) {
    { op.compute_output_specs(operation_attributes, tensor_args) }
        -> std::same_as<typename device_operation_t::spec_return_value_t>;
};
```

Returns the shape/spec metadata for outputs *without* allocating device memory. The `GraphTracker` uses this during `NO_DISPATCH` graph capture to record the dataflow graph without touching hardware.

These three, combined with the five type aliases and the `AllFactoriesValid` constraint on `program_factory_t`, constitute the full `DeviceOperationConcept`:

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 101-114)
template <typename device_operation_t>
concept DeviceOperationConcept = requires {
    typename device_operation_t::program_factory_t;

    [](const typename device_operation_t::operation_attributes_t& operation_attributes,
       const typename device_operation_t::tensor_args_t& tensor_args) {
        device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);

        using tensor_return_value_t = typename device_operation_t::tensor_return_value_t;
        static_assert(std::same_as<
                      decltype(device_operation_t::create_output_tensors(operation_attributes, tensor_args)),
                      tensor_return_value_t>);
    };
} && HasComputeOutputSpecs<device_operation_t>
  && AllFactoriesValid<typename device_operation_t::program_factory_t>;
```

| Requirement | Checked by |
|-------------|-----------|
| `program_factory_t` exists and is a variant | `typename device_operation_t::program_factory_t` |
| `validate_on_program_cache_miss(attrs, tensor_args)` exists | Lambda body |
| `create_output_tensors(attrs, tensor_args)` returns `tensor_return_value_t` | `static_assert` inside lambda |
| `compute_output_specs(attrs, tensor_args)` returns `spec_return_value_t` | `HasComputeOutputSpecs` concept |
| Every factory alternative is valid | `AllFactoriesValid` |

Additionally, the operation must provide a static `invoke(...)` method that returns `std::tuple<operation_attributes_t, tensor_args_t>`. This method is not checked by the concept itself but is checked by `registered_operation_t` at call time via a `static_assert`.

---

## 1.1.3 Optional Extensions

The framework detects additional static methods via supplementary concepts. An operation can opt into zero or more of these:

### `validate_on_program_cache_hit`

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 67-72)
template <typename device_operation_t>
concept HasValidateOnProgramCacheHit = requires(
    const typename device_operation_t::operation_attributes_t& attrs,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    device_operation_t::validate_on_program_cache_hit(attrs, tensor_args);
};
```

If absent, the framework falls back to calling `validate_on_program_cache_miss` on cache hits too -- meaning full validation runs every time. Providing a lightweight `validate_on_program_cache_hit` is a performance optimization: you can skip expensive checks (e.g., kernel compatibility) that were already verified when the program was first created, and only assert things that could change between invocations (e.g., that a tensor is still allocated).

### `compute_program_hash`

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 116-125)
template <typename device_operation_t>
concept DeviceOperationWithCustomProgramCacheConcept =
    DeviceOperationConcept<device_operation_t> &&
    requires(...) {
        { device_operation_t::compute_program_hash(attrs, tensor_args) }
            -> std::convertible_to<std::uint64_t>;
    };
```

If absent, the framework computes a default hash by hashing the *type identity* of the operation together with all fields of `operation_attributes_t` and `tensor_args_t`:

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 67-69)
return tt::stl::hash::hash_objects_with_default_seed(
    tt::stl::hash::type_hash<device_operation_t>, operation_attributes, tensor_args);
```

A custom hash is useful when the default over-hashes (e.g., a scalar runtime argument that does not affect kernel compilation should not invalidate the cache).

### `select_program_factory`

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 76-83)
template <typename device_operation_t>
concept HasSelectProgramFactory = requires(...) {
    { device_operation_t::select_program_factory(attrs, tensor_args) }
        -> std::same_as<typename device_operation_t::program_factory_t>;
};
```

Required if `program_factory_t` is a variant with more than one alternative. For single-alternative variants, the framework returns the sole type automatically via `MeshDeviceOperationAdapter::select_program_factory`:

```cpp
// Source: ttnn/api/ttnn/mesh_device_operation_adapter.hpp  (lines 56-63)
static program_factory_t select_program_factory(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    if constexpr (HasSelectProgramFactory<DeviceOperation>) {
        return DeviceOperation::select_program_factory(attrs, tensor_args);
    } else {
        return program_factory_t{std::variant_alternative_t<0, program_factory_t>{}};
    }
}
```

The `MeshDeviceOperationAdapter` enforces this constraint at compile time:

```cpp
// Source: ttnn/api/ttnn/mesh_device_operation_adapter.hpp  (lines 48-51)
static_assert(
    HasSelectProgramFactory<DeviceOperation> ||
        std::variant_size_v<program_factory_t> == 1,
    "DeviceOperation must implement select_program_factory when "
    "program_factory_t has more than one type.");
```

### `skip_launch`

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 128-136)
template <typename device_operation_t>
concept HasSkipLaunch = requires(...) {
    { device_operation_t::skip_launch(attrs, tensor_args, tensor_return_value) }
        -> std::convertible_to<bool>;
};
```

A short-circuit: if `skip_launch` returns `true`, the dispatch logic skips program creation and enqueue entirely. Used for ops that can detect at runtime that no work is needed (e.g., a binary op where both inputs are the same tensor and the result is trivially known).

---

## 1.1.4 ProgramFactoryConcept vs. MeshWorkloadFactoryConcept

Every alternative inside `program_factory_t` must satisfy exactly one of two concepts. The `AllFactoriesValid` constraint enforces this at compile time via a `consteval` fold:

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 87-99)
namespace detail {
template <typename Variant, std::size_t... Is>
consteval bool all_factories_valid(std::index_sequence<Is...>) {
    return (
        (ProgramFactoryConcept<std::variant_alternative_t<Is, Variant>> !=
         MeshWorkloadFactoryConcept<std::variant_alternative_t<Is, Variant>>) &&
        ...);
}
}  // namespace detail

template <typename Variant>
concept AllFactoriesValid =
    detail::all_factories_valid<Variant>(
        std::make_index_sequence<std::variant_size_v<Variant>>{});
```

The XOR (`!=`) ensures each factory is *exactly one* kind -- not both, not neither.

### ProgramFactoryConcept (single-device)

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 24-33)
template <typename T>
concept ProgramFactoryConcept = requires {
    typename T::cached_program_t;

    [](const auto& operation_attributes, const auto& tensor_args,
       auto& tensor_return_value) {
        auto cached_program = T::create(
            operation_attributes, tensor_args, tensor_return_value);
        T::override_runtime_arguments(
            cached_program, operation_attributes, tensor_args, tensor_return_value);
    };
};
```

A `ProgramFactoryConcept` factory must provide:

| Member | Meaning |
|--------|---------|
| `cached_program_t` | Alias for `CachedProgram<shared_variables_t>` -- bundles the compiled `Program` with mutable shared variables. |
| `create(...)` | Builds the program from scratch: creates CBs, kernels, sets compile-time args. Returns a `cached_program_t`. Called on cache miss. |
| `override_runtime_arguments(...)` | Updates only the runtime arguments (buffer addresses, runtime args) of a cached program. Called on cache hit. |

### MeshWorkloadFactoryConcept (multi-device)

```cpp
// Source: ttnn/api/ttnn/operation_concepts.hpp  (lines 36-53)
template <typename T>
concept MeshWorkloadFactoryConcept =
    HasMeshWorkloadType<T> && (HasCreateMeshWorkload<T> || HasCreateAt<T>);
```

A `MeshWorkloadFactoryConcept` factory must provide:

| Member | Meaning |
|--------|---------|
| `cached_mesh_workload_t` | Alias for `CachedMeshWorkload<shared_variables_t>` -- bundles a `MeshWorkload` (programs across coordinates) with shared variables. |
| `create_mesh_workload(...)` **or** `create_at(...)` | Path A creates the entire mesh workload at once; Path B creates per-coordinate and the framework aggregates. |
| `override_runtime_arguments(...)` | Same role as in `ProgramFactoryConcept`, but operates on the `cached_mesh_workload_t`. |

---

## 1.1.5 The `CachedProgram` Type Family

The `CachedProgram`, `CachedMeshWorkload`, and `AdaptedCachedMeshWorkload` types, defined in `tt_metal/api/tt-metalium/program_cache.hpp`, hold the compiled program alongside operation-specific shared state.

### `CachedProgram<shared_variables_t>`

```cpp
// Source: tt_metal/api/tt-metalium/program_cache.hpp  (lines 17-58)
template <typename shared_variables_t>
struct CachedProgram {
    tt::tt_metal::Program& program;
    shared_variables_t& shared_variables;

    CachedProgram(tt::tt_metal::Program&& program, shared_variables_t&& shared_variables);

    // Creates a non-owning proxy for override_runtime_arguments
    static CachedProgram proxy(
        tt::tt_metal::Program& program, shared_variables_t& shared_variables);
};
```

`shared_variables_t` is a user-defined struct (typically containing kernel handles like `KernelHandle`, CB handles like `CBHandle`, `CoreRangeSet`, tile sizes, etc.) that persists across invocations. On cache hit, `override_runtime_arguments` receives the *same* `shared_variables` that `create` populated, so it can update runtime args on the already-compiled kernels without recompiling.

The **owning constructor** takes rvalue references and stores the program/variables internally. The **`proxy`** static factory creates a non-owning `CachedProgram` that references externally-owned state. This proxy pattern is critical: the mesh adapter uses it when calling a single-device `ProgramFactory::override_runtime_arguments` on programs that physically live inside an `AdaptedCachedMeshWorkload`, avoiding unnecessary copies while maintaining the expected `cached_program_t&` signature.

### `CachedMeshWorkload` and `AdaptedCachedMeshWorkload`

```cpp
// Source: tt_metal/api/tt-metalium/program_cache.hpp  (lines 60-81)
template <typename shared_variables_t>
struct CachedMeshWorkload {
    tt::tt_metal::distributed::MeshWorkload workload;
    shared_variables_t shared_variables;
};

template <typename shared_variables_t>
struct AdaptedCachedMeshWorkload {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<distributed::MeshCoordinateRange, shared_variables_t> shared_variables;
};
```

`CachedMeshWorkload` is for native mesh-aware factories (`MeshWorkloadFactoryConcept`). `AdaptedCachedMeshWorkload` is produced by `MeshDeviceOperationAdapter::MeshWorkloadFactoryAdapter` when adapting a single-device `ProgramFactoryConcept` factory -- it creates one program per coordinate range and stores per-range shared variables in the map.

---

## 1.1.6 Concept Hierarchy Diagram

The relationships between concepts form a layered hierarchy:

```
ProgramFactoryConcept          MeshWorkloadFactoryConcept
       |                          |
       +--- XOR ------------------+
                    |
            AllFactoriesValid<variant>
                    |
           DeviceOperationConcept
          /         |          \
  HasComputeOutput  |   validate/create methods
     Specs          |
                    |
    +----- Extensions (optional) ------+
    |               |                  |
HasValidateOn   HasSelectProgram   HasSkipLaunch
ProgramCacheHit   Factory
    |
DeviceOperationWithCustomProgramCacheConcept
```

Or expressed more formally:

$$
\text{DeviceOperationConcept} \;\Longleftarrow\;
\begin{cases}
\text{5 type aliases} \\
\texttt{validate\_on\_program\_cache\_miss} \\
\texttt{create\_output\_tensors} \\
\text{HasComputeOutputSpecs} \\
\text{AllFactoriesValid}\!\bigl(\texttt{program\_factory\_t}\bigr)
\end{cases}
$$

$$
\text{AllFactoriesValid} \;\Longleftarrow\;
\forall\, T_i \in \texttt{program\_factory\_t} : \;
\text{ProgramFactoryConcept}(T_i) \;\oplus\;
\text{MeshWorkloadFactoryConcept}(T_i)
$$

---

## 1.1.7 Concrete Example: BinaryDeviceOperation

The binary eltwise operation (`ttnn.add`, `ttnn.sub`, `ttnn.mul`, etc.) is a canonical example because it exercises almost every feature of the contract: a multi-alternative `program_factory_t`, custom hash, custom cache-hit validation, `skip_launch`, and performance model.

```cpp
// Source: ttnn/cpp/ttnn/operations/eltwise/binary/device/binary_device_operation.hpp
struct BinaryDeviceOperation {
    // --- 1. Type Aliases ---
    struct operation_attributes_t {
        BinaryOpType binary_op_type{};
        const std::optional<unary::EltwiseFusedActivations> activations;
        const std::optional<unary::EltwiseUnaryWithParam> input_tensor_a_activation;
        const std::optional<float> scalar;
        const MemoryConfig memory_config;
        const DataType dtype{};
        const CoreRangeSet worker_grid;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;

        tt::stl::hash::hash_t to_hash() const {
            // Deliberately excludes `scalar` -- it is a runtime arg,
            // not a compile-time parameter.
            return tt::stl::hash::hash_objects_with_default_seed(
                binary_op_type, activations, input_tensor_a_activation,
                memory_config, dtype, compute_kernel_config);
        }
    };

    struct tensor_args_t {
        const Tensor& input_tensor_a;
        std::optional<Tensor> input_tensor_b;
        std::optional<Tensor> output_tensor;
    };

    using spec_return_value_t    = TensorSpec;
    using tensor_return_value_t  = Tensor;

    // --- 2. Program Factories (7 alternatives!) ---
    struct ElementWiseMultiCore {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle binary_reader_kernel_id{};
            tt::tt_metal::KernelHandle unary_writer_kernel_id{};
            tt::tt_metal::KernelHandle eltwise_binary_kernel_id{};
            tt::tt_metal::CBHandle cb_src0{};
            tt::tt_metal::CBHandle cb_src1{};
            tt::tt_metal::CBHandle cb_output{};
            CoreRangeSet all_device_cores;
            uint32_t src0_single_tile_size{};
            uint32_t src1_single_tile_size{};
            uint32_t dst_single_tile_size{};
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t&, const tensor_args_t&,
            tensor_return_value_t&);
        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&,
            const tensor_args_t&, tensor_return_value_t&);
    };
    // ... ElementWiseMultiCoreSfpu, BroadcastWidthMultiCore,
    //     BroadcastHeightMultiCore, BroadcastHeightAndWidthMultiCore,
    //     BroadcastHeightMultiCoreSharded,
    //     BroadcastHeightMultiCoreShardedOptimized ...

    using program_factory_t = std::variant<
        ElementWiseMultiCore,
        ElementWiseMultiCoreSfpu,
        BroadcastWidthMultiCore,
        BroadcastHeightMultiCore,
        BroadcastHeightAndWidthMultiCore,
        BroadcastHeightMultiCoreSharded,
        BroadcastHeightMultiCoreShardedOptimized>;

    // --- 3. Required Methods ---
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);

    // --- 4. Optional Extensions ---
    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(
        const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
    static bool skip_launch(
        const operation_attributes_t&, const tensor_args_t&,
        const tensor_return_value_t&);

    // --- 5. Performance Model (used by profiler) ---
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>
    create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&,
        tensor_return_value_t&);
};
```

Key observations:

- **Seven program factories**: `select_program_factory` must be provided because the variant has more than one alternative. It inspects the input shapes and layouts to decide whether to use element-wise, broadcast-width, broadcast-height, or broadcast-H+W kernels, plus sharded variants.
- **Custom hash via `to_hash()`**: The `operation_attributes_t::to_hash()` method is called by `compute_program_hash`. It excludes `scalar` because that value only changes runtime arguments, not program structure.
- **Shared variables hold kernel/CB handles**: On cache hit, `override_runtime_arguments` uses these handles to patch buffer addresses into the already-compiled kernels. This avoids recompiling kernels for every call.
- **`skip_launch`**: Returns `true` when the output is trivially known (e.g., multiplying by zero).
- **Performance model**: `create_op_performance_model` produces compute/ideal/bandwidth nanosecond estimates that feed into the Tracy profiler JSON.

### Contrast: GenericOpDeviceOperation

For contrast, `GenericOpDeviceOperation` -- the escape hatch through which TT-Blaze currently dispatches -- satisfies the same concept but with fundamentally different characteristics:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp
struct GenericOpDeviceOperation {
    using operation_attributes_t = tt::tt_metal::experimental::MeshProgramDescriptor;
    using tensor_args_t          = generic::tensor_args_t;
    using spec_return_value_t    = TensorSpec;
    using tensor_return_value_t  = Tensor;
    using program_factory_t      = std::variant<program::GenericMeshProgramFactory>;
    // ...
};
```

Where `BinaryDeviceOperation` has 7 specialized factories and typed attributes, `GenericOpDeviceOperation` has a single generic factory and its `operation_attributes_t` is a raw `MeshProgramDescriptor` -- an opaque bag of kernel sources, compile-time args, and CB configs. This flexibility comes at the cost of identity: the framework sees one type (`GenericOpDeviceOperation`) for every operation dispatched through this path. The full implications of this identity loss are analyzed in [Section 1.3](./03_program_cache_graph_tracing_and_profiling.md).

---

## Key Takeaways

1. **Conformance is structural, not inheritance-based**: if a struct's signatures match, it satisfies `DeviceOperationConcept`. No base class is involved.

2. **Factory alternatives are XOR-constrained**: each variant alternative must satisfy exactly one of `ProgramFactoryConcept` or `MeshWorkloadFactoryConcept` -- never both, never neither -- enforced at compile time by `AllFactoriesValid`.

3. **`CachedProgram::proxy` enables the mesh adapter pattern**: the non-owning proxy lets `MeshWorkloadFactoryAdapter` call single-device `override_runtime_arguments` on programs that physically live inside an `AdaptedCachedMeshWorkload`, avoiding copies while preserving the `cached_program_t&` signature.

---

**Next:** [`02_registration_dispatch_and_bindings.md`](./02_registration_dispatch_and_bindings.md)
