# 1.2 Registration, Dispatch, and Bindings

Once an operation struct satisfies `DeviceOperationConcept`, three more layers connect it to the outside world: a compile-time registration template that produces a callable C++ object, a mesh adapter that generalizes single-device ops to multi-device dispatch, a `launch<>` function template that orchestrates the full execution pipeline, and a nanobind binding function that surfaces the whole thing as a Python callable. This section traces each layer, shows how they interact, and explains why the design prevents duplicate registrations while preserving full type identity at every stage.

---

## 1.2.1 The `register_operation<>` Template

Registration happens at namespace scope via a `constexpr` variable initialized by `ttnn::register_operation`:

```cpp
// Source: ttnn/api/ttnn/decorators.hpp  (lines 140-143)
template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation() {
    return register_operation_impl<cpp_fully_qualified_name, operation_t>();
}
```

The implementation creates a `registered_operation_t` instance and enforces uniqueness:

```cpp
// Source: ttnn/api/ttnn/decorators.hpp  (lines 127-138)
template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
constexpr auto register_operation_impl() {
    constexpr auto operation = registered_operation_t<cpp_fully_qualified_name, operation_t>{};
    static_assert(
        not requires(operation_name_key_t<cpp_fully_qualified_name> key) { get(key); },
        "Operation with this `cpp_fully_qualified_name` was already registered.");
    static_assert(
        not requires(operation_key_t<operation_t> key) { get(key); },
        "Operation with this `operation_t` was already registered.");
    static_assert(set_operation_t<cpp_fully_qualified_name, operation_t, operation>::value);
    return operation;
}
```

This uses a "stateful metaprogramming" technique: `set_operation_t` is a `std::true_type`-derived class template whose `friend consteval auto get(...)` definitions inject compile-time lookup entries. The two `static_assert` checks query those entries:

- **Name uniqueness**: no two operations can share the same `cpp_fully_qualified_name` (e.g., `"ttnn::add"`).
- **Type uniqueness**: no two registrations can use the same `operation_t` struct.

A typical registration site looks like:

```cpp
// Source: ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp  (line 306)
constexpr auto add = ttnn::register_operation<
    "ttnn::add",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::ADD>>();
```

---

## 1.2.2 `registered_operation_t` -- the Callable Wrapper

The result of `register_operation<>` is a `registered_operation_t<name, operation_t>` instance. This is a lightweight, stateless struct that acts as a function object:

```cpp
// Source: ttnn/api/ttnn/decorators.hpp  (lines 64-109)
template <reflect::fixed_string cpp_fully_qualified_name, typename operation_t>
struct registered_operation_t {
    std::string base_name() const;               // "add"
    std::string class_name() const;              // "add_t"
    std::string python_fully_qualified_name() const; // "ttnn.add"

    template <typename... Args>
        requires(HasInvoke<operation_t, Args&&...>)
    auto operator()(Args&&... args) const {
        return traced_invoke(std::forward<Args>(args)...);
    }
    // ...
};
```

When called, `operator()` dispatches through a two-branch `invoke`:

```cpp
// Source: ttnn/api/ttnn/decorators.hpp  (lines 90-108)
// Branch 1: Device operations
template <typename... args_t>
    requires device_operation::DeviceOperationConcept<operation_t>
auto invoke(args_t&&... args) const {
    static_assert(
        requires { operation_t::invoke(std::forward<decltype(args)>(args)...); },
        "Device Operation must implement invoke() method.");

    auto [operation_attributes, tensors_args] =
        operation_t::invoke(std::forward<decltype(args)>(args)...);
    return ttnn::device_operation::detail::invoke<operation_t>(
        operation_attributes, tensors_args);
}

// Branch 2: Composite (non-device) operations
template <typename... args_t>
    requires(!device_operation::DeviceOperationConcept<operation_t>)
auto invoke(args_t&&... args) const {
    return operation_t::invoke(std::forward<decltype(args)>(args)...);
}
```

For device operations, `operation_t::invoke(...)` is a user-provided static method that converts the user-facing arguments (e.g., two tensors, a dtype, a memory config) into the `(operation_attributes_t, tensor_args_t)` pair. This structured decomposition is what the framework needs to compute hashes, run validation, and manage caching.

For composite operations (like `BinaryOperation<ADD>`, which wraps `BinaryDeviceOperation`), `invoke` is called directly and the operation itself is responsible for calling device operations internally.

---

## 1.2.3 `MeshDeviceOperationAdapter`

Before `launch<>` dispatches to hardware, it wraps the user's device operation in `MeshDeviceOperationAdapter<DeviceOperation>`. This adapter is the bridge between the single-device contract (`DeviceOperationConcept`) and the mesh-aware dispatch path.

```cpp
// Source: ttnn/api/ttnn/mesh_device_operation_adapter.hpp  (lines 35-190)
template <typename DeviceOperation>
struct MeshDeviceOperationAdapter {
    using device_operation_t = DeviceOperation;

    // Inherits all type aliases from the base operation
    using operation_attributes_t = typename DeviceOperation::operation_attributes_t;
    using tensor_args_t          = typename DeviceOperation::tensor_args_t;
    // ... etc ...

    // Delegates to base or provides defaults
    static program_factory_t select_program_factory(...);
    static void validate_on_program_cache_hit(...);
    static void validate_on_program_cache_miss(...);
    static spec_return_value_t compute_output_specs(...);
    static tensor_return_value_t create_output_tensors(...);
    static tt::stl::hash::hash_t compute_program_hash(...);
    static tt::stl::hash::hash_t compute_mesh_workload_hash(...);

    // Inner adapter: converts ProgramFactoryConcept to MeshWorkloadFactoryConcept
    template <ProgramFactoryConcept ProgramFactory>
    struct MeshWorkloadFactoryAdapter { ... };
};
```

The key inner class is `MeshWorkloadFactoryAdapter`:

```cpp
// Source: ttnn/api/ttnn/mesh_device_operation_adapter.hpp  (lines 110-153)
template <ProgramFactoryConcept ProgramFactory>
struct MeshWorkloadFactoryAdapter {
    using shared_variables_t = typename ProgramFactory::shared_variables_t;
    using cached_mesh_workload_t = AdaptedCachedMeshWorkload<shared_variables_t>;

    static auto create_mesh_workload(
        const operation_attributes_t& attrs,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        // For each coordinate range, call the single-device
        // ProgramFactory::create() and collect into a MeshWorkload
        tt::tt_metal::distributed::MeshWorkload mesh_workload;
        std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
        for (const auto& range : tensor_coords.ranges()) {
            auto cached_program = ProgramFactory{}.create(
                attrs, tensor_args, tensor_return_value);
            mesh_workload.add_program(range, std::move(cached_program.program));
            shared_variables[range] = std::move(cached_program.shared_variables);
        }
        return AdaptedCachedMeshWorkload<shared_variables_t>{
            std::move(mesh_workload), std::move(shared_variables)};
    }

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        // Per-coordinate-range override using the original ProgramFactory
        for (auto& [coordinate_range, program] :
                 cached_workload.workload.get_programs()) {
            auto& svars = cached_workload.shared_variables.at(coordinate_range);
            // ... calls ProgramFactory::override_runtime_arguments via proxy ...
        }
    }
};
```

This means an op author who only writes a `ProgramFactoryConcept` factory automatically gets mesh support. The framework stamps the single-device program across all mesh coordinates.

The mesh workload hash adds coordinate information to the base program hash:

```cpp
// Source: ttnn/api/ttnn/mesh_device_operation_adapter.hpp  (lines 155-165)
static tt::stl::hash::hash_t compute_mesh_workload_hash(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args) {
    auto hash = compute_program_hash(attrs, tensor_args);
    for (const auto& coord :
         mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
        ttsl::hash::hash_combine(hash, coord);
    }
    return hash;
}
```

---

## 1.2.4 `device_operation::launch<>`

The `launch<>` function template is the central dispatch entry point. It orchestrates the entire execution from output allocation through graph tracking to hardware enqueue:

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 412-491)
template <DeviceOperationConcept device_operation_t>
typename device_operation_t::tensor_return_value_t launch(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    // 1. Collect input tensors for GraphTracker
    std::vector<std::reference_wrapper<const Tensor>> input_tensors;
    tt::stl::reflection::visit_object_of_type<Tensor>(
        [&](const Tensor& t) { input_tensors.push_back(std::cref(t)); }, tensor_args);

    // 2. Notify GraphTracker of function start
    tt::tt_metal::GraphTracker::instance().track_function_start(
        detail::get_operation_name<device_operation_t>(operation_attributes),
        operation_attributes, input_tensors);

    // 3. Assert first tensor is on device
    auto first_tensor = tt::stl::reflection::get_first_object_of_type<Tensor>(tensor_args);
    if (first_tensor.has_value()) [[likely]] {
        TT_FATAL(tt::tt_metal::is_device_tensor(first_tensor.value()), ...);
    }

    // 4. Allocate output tensors
    auto tensor_return_value =
        device_operation_t::create_output_tensors(operation_attributes, tensor_args);

    // 5. Get mesh device from first tensor or attributes
    ttnn::MeshDevice* mesh_device =
        detail::get_mesh_device<device_operation_t>(operation_attributes, tensor_args);

    // 6. Compute output topology (placements and distribution shape)
    // ... imputation from input topologies or custom compute_output_topologies ...

    // 7. Dispatch through MeshDeviceOperationAdapter
    detail::launch_operation_with_adapter<
        MeshDeviceOperationAdapter<device_operation_t>>(
            operation_attributes, tensor_args, tensor_return_value, mesh_device);

    // 8. Notify GraphTracker of function end
    tt::tt_metal::GraphTracker::instance().track_function_end(tensor_return_value);

    return tensor_return_value;
}
```

Step 7 calls into `launch_operation_with_adapter`, which handles the program cache:

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 336-377)
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void launch_operation_with_adapter(...) {
    // Optional skip_launch check
    if constexpr (HasSkipLaunch<mesh_device_operation_t>) {
        if (mesh_device_operation_t::skip_launch(...)) return;
    }

    auto& program_cache = mesh_device->get_program_cache();
    auto program_hash = mesh_device_operation_t::compute_mesh_workload_hash(
        mesh_device, operation_attributes, tensor_args);
    bool program_cache_hit = program_cache.contains(program_hash);

    // Optionally forbid cache misses (for production/inference mode)
    if (!program_cache_hit && !program_cache.cache_misses_allowed()) {
        TT_THROW("program cache miss occurred, but cache misses are forbidden");
    }

    if (program_cache_hit) {
        handle_mesh_adapter_cache_hit<mesh_device_operation_t>(...);
    } else {
        create_and_cache_mesh_workload<mesh_device_operation_t>(...);
    }
}
```

The detailed cache-hit and cache-miss code paths are traced in [Section 1.3.1](./03_program_cache_graph_tracing_and_profiling.md#131-the-program-cache).

### The `dispatch_to_mesh_workload_factory` Visitor

Both the cache-hit and cache-miss paths converge through a `std::visit`-based helper that routes to the correct factory type:

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 205-218)
std::visit(
    tt::stl::overloaded{
        [&]<ProgramFactoryConcept T>(const T&) {
            // Adapt ProgramFactory -> MeshWorkloadFactory
            using AdaptedFactory =
                mesh_device_operation_t
                    ::template MeshWorkloadFactoryAdapter<T>;
            fn.template operator()<AdaptedFactory>();
        },
        [&]<MeshWorkloadFactoryConcept WorkloadFactory>(
            const WorkloadFactory&) {
            fn.template operator()<WorkloadFactory>();
        }},
    program_factory);
```

This is the critical convergence point of the two factory concepts. When the `program_factory_t` variant holds a `ProgramFactoryConcept` alternative, it wraps it in `MeshWorkloadFactoryAdapter` before proceeding. When it holds a `MeshWorkloadFactoryConcept` alternative, it passes it through directly. The caller (either the cache-hit handler or the cache-miss handler) then receives a uniform `MeshWorkloadFactoryConcept`-satisfying type.

---

## 1.2.5 `bind_registered_operation()` -- Nanobind Exposure

The final layer exposes the registered operation to Python via nanobind:

```cpp
// Source: ttnn/cpp/ttnn-nanobind/decorators.hpp  (lines 133-167)
template <reflect::fixed_string cpp_fully_qualified_name,
          typename operation_t, typename... overload_t>
auto bind_registered_operation(
    nb::module_& mod,
    const registered_operation_t<cpp_fully_qualified_name, operation_t>& operation,
    const std::string& doc,
    overload_t&&... overloads)
{
    using registered_operation_t = std::decay_t<decltype(operation)>;

    // Create a nanobind class for the operation
    nb::class_<registered_operation_t> py_operation(
        mod, operation.class_name().c_str());   // e.g., "add_t"

    py_operation.doc() = doc;

    // Read-only properties
    py_operation.def_prop_ro("name", ...);                    // "add"
    py_operation.def_prop_ro("python_fully_qualified_name", ...); // "ttnn.add"
    py_operation.def_prop_ro("__ttnn_operation__", ...);      // sentinel

    // Register __call__ overloads
    (def_call_operator<registered_operation_t, operation_t>(
        py_operation, std::forward<overload_t>(overloads)), ...);

    // Bind instance to module attribute
    mod.attr(operation.base_name().c_str()) = operation;  // ttnn.add = <instance>

    return py_operation;
}
```

Each `overload_t` is either a `nanobind_arguments_t` (simple forwarding to `operation_t::invoke`) or a `nanobind_overload_t` (a custom lambda with explicit nanobind argument annotations). The `def_call_operator` function uses different resolution paths depending on whether the operation is a `DeviceOperationConcept` or a composite:

```cpp
// Source: ttnn/cpp/ttnn-nanobind/decorators.hpp  (lines 102-131)
// For device operations: uses resolve_primitive_operation_call_method
template <...>
    requires device_operation::DeviceOperationConcept<operation_t>
void def_call_operator(py_operation_t& py_operation,
                       const nanobind_overload_t<...>& overload) {
    py_operation.def("__call__",
        resolve_primitive_operation_call_method(overload.function), ...);
}

// For composite operations: uses resolve_composite_operation_call_method
template <...>
    requires(!device_operation::DeviceOperationConcept<operation_t>)
void def_call_operator(py_operation_t& py_operation,
                       const nanobind_overload_t<...>& overload) {
    py_operation.def("__call__",
        resolve_composite_operation_call_method(overload.function), ...);
}
```

### Concrete Binding Example: generic_op

The `generic_op` binding in `generic_op_nanobind.cpp` demonstrates the full pattern:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp  (lines 17-61)
void bind_generic_operation(nb::module_& mod) {
    std::string doc = R"doc(
        Executes a custom operation with user-defined kernels...
    )doc";

    using registered_operation_t =
        std::decay_t<decltype(ttnn::generic_op)>;

    // Lambda for MeshProgramDescriptor overload
    auto mesh_program_invoke =
        [](const registered_operation_t& self,
           const std::vector<Tensor>& io_tensors,
           const tt::tt_metal::experimental::MeshProgramDescriptor& desc) {
            return self(io_tensors, desc);
        };

    // Lambda for ProgramDescriptor overload
    auto program_invoke =
        [](const registered_operation_t& self,
           const std::vector<Tensor>& io_tensors,
           const tt::tt_metal::ProgramDescriptor& desc) {
            return self(io_tensors, desc);
        };

    bind_registered_operation(
        mod,
        ttnn::generic_op,
        doc,
        ttnn::nanobind_overload_t{
            mesh_program_invoke,
            nb::arg("io_tensors"),
            nb::arg("mesh_program_descriptor")},
        ttnn::nanobind_overload_t{
            program_invoke,
            nb::arg("io_tensors"),
            nb::arg("program_descriptor")});
}
```

After this binding, Python users can call either:
```python
ttnn.generic_op(io_tensors, mesh_program_descriptor=desc)
ttnn.generic_op(io_tensors, program_descriptor=desc)
```

The `__ttnn_operation__` property lets framework code (e.g., tracing, graph capture) detect that an object is a registered TTNN operation.

---

## 1.2.6 End-to-End Flow Summary

When Python calls `ttnn.add(a, b)`, the following chain executes. Note that
`BinaryOperation<ADD>` is a **composite** operation (Branch 2 in Section 1.2.2),
so the framework calls its `invoke` directly -- the composite itself is
responsible for constructing the device-operation attributes and calling
`device_operation::launch<>` internally:

```
Python: ttnn.add(a, b)
  -> nanobind __call__
    -> registered_operation_t::operator()
      -> [Branch 2: composite path] BinaryOperation<ADD>::invoke(a, b)
        // Inside the composite's invoke -- NOT the framework:
        -> constructs (operation_attributes_t, tensor_args_t)
        -> calls device_operation::launch<BinaryDeviceOperation>(attrs, tensor_args)
          -> GraphTracker::track_function_start(...)
          -> BinaryDeviceOperation::create_output_tensors(...)
          -> launch_operation_with_adapter<MeshDeviceOperationAdapter<BinaryDeviceOperation>>(...)
            -> compute_mesh_workload_hash(...)
            -> program_cache.contains(hash)?
              HIT:  validate_on_program_cache_hit -> override_runtime_arguments -> EnqueueMeshWorkload
              MISS: validate_on_program_cache_miss -> select_program_factory -> create_mesh_workload
                    -> cache -> EnqueueMeshWorkload
            -> TracyOpMeshWorkload(...)
          -> GraphTracker::track_function_end(...)
          -> return output tensor
      -> return result from invoke
```

### Implications for Blaze Op Registration

The end-to-end flow reveals what a TT-Blaze operation must do to become a first-class TTNN citizen:

1. **Define a `DeviceOperationConcept`-satisfying struct** with typed attributes and tensor args (not a raw `MeshProgramDescriptor`).
2. **Implement program factories** with proper `shared_variables_t` for efficient cache-hit patching.
3. **Register via `register_operation<>`** with a unique name and type.
4. **Bind via `bind_registered_operation()`** with appropriate nanobind overloads.

These requirements are the subject of Chapters 4-6.

---

## Key Takeaways

1. **Compile-time uniqueness is enforced via stateful metaprogramming**: `register_operation<>` uses `friend consteval` injection to prevent duplicate names or types -- a technique that catches registration conflicts at compile time rather than runtime.

2. **`dispatch_to_mesh_workload_factory` is the architectural convergence point**: a single `std::visit` routes both `ProgramFactoryConcept` and `MeshWorkloadFactoryConcept` alternatives into a uniform mesh workload interface, used by both cache-hit and cache-miss paths.

3. **Type identity is preserved end to end**: from the `operation_t` template parameter through the mesh adapter to the program cache hash, the framework always knows which operation struct it is dealing with. This is critical for profiling, debugging, and cache correctness -- and is exactly what `generic_op` loses.

4. **Single-device op authors get mesh support for free**: `MeshDeviceOperationAdapter` stamps programs across coordinates and adds mesh-aware hashing, requiring no mesh-specific code from the operation author.

---

**Next:** [`03_program_cache_graph_tracing_and_profiling.md`](./03_program_cache_graph_tracing_and_profiling.md)
