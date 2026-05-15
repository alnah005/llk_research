# 1.3 Program Cache, Graph Tracing, and Profiling

The three cross-cutting systems described in this section -- the program cache, graph tracing, and Tracy profiling -- all depend on the type identity that `DeviceOperationConcept` provides. When an operation goes through `launch<>`, the framework uses its compile-time type to compute cache hashes, record nodes in the execution graph, and emit named Tracy zones. This section explains how each system works, then shows why `generic_op` -- the type-erased escape hatch -- loses identity in all three, and outlines concrete mitigation strategies.

---

## 1.3.1 The Program Cache

### Data Structures

The program cache lives on each `MeshDevice` and maps `uint64_t` hash keys to `CachedProgramFactory` entries:

```cpp
// Source: tt_metal/api/tt-metalium/program_cache.hpp  (lines 108-134)
struct ProgramCache {
    bool contains(uint64_t program_hash) const;
    CachedProgramFactory& get(uint64_t program_hash);
    void insert(uint64_t program_hash, CachedProgramFactory&& program);

    void enable();
    void disable();
    bool is_enabled() const;

    void set_cache_misses_allowed(bool allowed);
    bool cache_misses_allowed() const;

    void clear();
    std::size_t num_entries() const;

private:
    bool is_enabled_ = true;
    bool allow_cache_misses_ = true;
    std::unordered_map<uint64_t, CachedProgramFactory> cache_;
};
```

Each `CachedProgramFactory` stores the compiled program (or mesh workload) in a type-erased `unique_any` alongside the variant index of the program factory that produced it:

```cpp
// Source: tt_metal/api/tt-metalium/program_cache.hpp  (lines 83-104)
struct CachedProgramFactory {
    static constexpr auto MAX_SIZE = 4096;
    static constexpr auto ALIGNMENT = 32;

    tt::stl::unique_any<MAX_SIZE, ALIGNMENT> cached_program;
    std::size_t program_factory_index = 0;

    // Constructors for CachedProgram, CachedMeshWorkload,
    // and AdaptedCachedMeshWorkload
    template <typename shared_variables_t>
    CachedProgramFactory(CachedProgram<shared_variables_t>&& cached_program,
                         std::size_t program_factory_index);
    template <typename shared_variables_t>
    CachedProgramFactory(CachedMeshWorkload<shared_variables_t>&& cached_workload,
                         std::size_t program_factory_index);
    template <typename shared_variables_t>
    CachedProgramFactory(AdaptedCachedMeshWorkload<shared_variables_t>&& cached_workload,
                         std::size_t program_factory_index);
};
```

The `unique_any` is a fixed-size, alignment-aware type-erased container (max 4096 bytes, 32-byte aligned). It stores the `CachedProgram`, `CachedMeshWorkload`, or `AdaptedCachedMeshWorkload` without heap allocation. On cache hit, the framework casts it back to the correct type via `.template get<cached_mesh_workload_t>()`. The `program_factory_index` allows the cache-hit path to reconstruct the variant alternative without calling `select_program_factory` again.

### Hash Computation

The hash that keys into the program cache is computed in two stages:

**Stage 1: Program hash** (operation + attributes + tensor args)

The framework's `compute_program_hash` either calls a custom hash method (if the operation satisfies `DeviceOperationWithCustomProgramCacheConcept`) or falls back to hashing the type identity together with all attributes and tensor args. See [Section 2.1.5](../ch02_generic_op_dispatch/01_generic_op_internals.md) for the full implementation and hash computation details.

The default hash includes `type_hash<device_operation_t>` -- a compile-time constant derived from the mangled type name. This means two different operation types with identical attributes and tensor args will produce different hashes, preventing cache collisions.

**Stage 2: Mesh workload hash** -- The mesh workload hash is computed by combining the program hash with mesh coordinates (see Section 1.2.3, `compute_mesh_workload_hash`).

### Cache Hit Path

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 220-250)
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void handle_mesh_adapter_cache_hit(...) {
    // 1. Validate (lightweight if HasValidateOnProgramCacheHit is provided)
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(
            operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(
            operation_attributes, tensor_args);
    }

    // 2. Retrieve cached factory and determine the program factory type
    auto& cached_program_factory = program_cache.get(program_hash);
    auto program_factory = map_index_to_variant(
        cached_program_factory.program_factory_index,
        mesh_device_operation_t::select_program_factory(
            operation_attributes, tensor_args));

    // 3. Dispatch through the correct factory variant
    dispatch_to_mesh_workload_factory<mesh_device_operation_t>(
        program_factory, [&]<MeshWorkloadFactoryConcept WorkloadFactory>() {
            auto& cached_mesh_workload =
                cached_program_factory.cached_program
                    .template get<typename WorkloadFactory::cached_mesh_workload_t>();

            // 4. Patch runtime arguments
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes,
                tensor_args, tensor_return_value);

            // 5. Enqueue
            enqueue_mesh_workload<mesh_device_operation_t>(..., cached_mesh_workload.workload);
        });
}
```

### Cache Miss Path

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 270-333)
template <DeviceOperationConcept mesh_device_operation_t>
void create_and_cache_mesh_workload(...) {
    // 1. Full validation
    mesh_device_operation_t::validate_on_program_cache_miss(
        operation_attributes, tensor_args);

    // 2. Select and create
    auto program_factory = mesh_device_operation_t::select_program_factory(
        operation_attributes, tensor_args);
    auto program_factory_index = program_factory.index();

    dispatch_to_mesh_workload_factory<mesh_device_operation_t>(
        program_factory, [&]<MeshWorkloadFactoryConcept WorkloadFactory>() {
            // Determine mesh coordinates (fast path for uniform storage)
            ttnn::MeshCoordinateRangeSet tensor_coords;
            if (mesh_device_operation_utils
                    ::all_tensors_have_uniform_storage(tensor_args)) {
                tensor_coords.merge(
                    ttnn::MeshCoordinateRange(mesh_device->shape()));
            } else {
                // Slow path: per-coordinate iteration
                for (const auto& coord : ...) {
                    tensor_coords.merge(
                        ttnn::MeshCoordinateRange(coord, coord));
                }
            }

            auto cached_workload = create_mesh_workload_from_workload_factory<
                WorkloadFactory, mesh_device_operation_t>(...);

            // 3. Inspector annotation
            emit_mesh_workload_annotation<mesh_device_operation_t>(
                cached_workload.workload, operation_attributes, tensor_args);

            // 4. Cache decision (skip during NO_DISPATCH graph capture)
            bool hook_blocks = false;
            if (auto hook = tt::tt_metal::GraphTracker::instance().get_hook()) {
                auto* processor_hooks =
                    dynamic_cast<ttnn::graph::ProcessorHooks*>(hook.get());
                if (processor_hooks) {
                    hook_blocks = processor_hooks->get_block();
                }
            }
            bool should_cache = program_cache.is_enabled() && !hook_blocks;

            // 5. Insert into cache and enqueue
            if (should_cache) {
                program_cache.insert(program_hash,
                    CachedProgramFactory{std::move(cached_workload),
                                         program_factory_index});
                auto& entry = program_cache.get(program_hash);
                auto& workload = entry.cached_program
                    .template get<typename WorkloadFactory::cached_mesh_workload_t>()
                    .workload;
                enqueue_mesh_workload<mesh_device_operation_t>(..., workload);
            } else {
                enqueue_mesh_workload<mesh_device_operation_t>(
                    ..., cached_workload.workload);
            }
        });
}
```

A critical detail: during `NO_DISPATCH` graph capture mode, the framework skips caching. Buffer addresses are invalid (address=0) in that mode, so caching such programs would corrupt the cache for later real execution.

### Cache Miss Prohibition

In production/inference mode, the cache can be locked:

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 359-362)
if (!program_cache_hit && !program_cache.cache_misses_allowed()) {
    TT_THROW("Device operation \"{}\": program cache miss occurred, "
             "but cache misses are forbidden", op_name);
}
```

This is used to detect unexpected new shapes or configurations at inference time, where every op should have been warmed up during a profiling pass.

---

## 1.3.2 Graph Tracing (GraphTracker)

### Architecture

`GraphTracker` is a singleton that maintains a stack of `IGraphProcessor` instances and an optional `IGraphHooks` hook:

```cpp
// Source: tt_metal/api/tt-metalium/graph_tracking.hpp  (lines 101-196)
class GraphTracker {
public:
    static GraphTracker& instance();
    bool is_enabled() const;

    void push_processor(const std::shared_ptr<IGraphProcessor>& processor);
    void pop_processor();

    bool add_hook(const std::shared_ptr<IGraphHooks>& hook);

    // Tracking methods
    template <class... Args>
    void track_function_start(std::string_view function_name, Args&&... args);

    template <class ReturnType>
    void track_function_end(ReturnType& output_tensors);

    void track_allocate(const Buffer* buffer);
    void track_program(Program* program, const IDevice* device);

    // Hook delegation
    bool hook_allocate(const Buffer* buffer);
    bool hook_deallocate(Buffer* buffer);
    bool hook_program(Program* program);
    // ... etc ...
};
```

### How `launch<>` Integrates

The `launch<>` function brackets every operation with `track_function_start` / `track_function_end`:

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 420-421, 489)
tt::tt_metal::GraphTracker::instance().track_function_start(
    detail::get_operation_name<device_operation_t>(operation_attributes),
    operation_attributes, input_tensors);
// ... dispatch ...
tt::tt_metal::GraphTracker::instance().track_function_end(tensor_return_value);
```

The operation name is derived from the C++ type name via `tt::stl::get_type_name<device_operation_t>()`. For `MeshDeviceOperationAdapter` wrapping, the framework recurses to get the underlying operation name:

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 130-138)
template <typename device_operation_t>
auto get_operation_name(const auto& operation_attributes) {
    if constexpr (is_mesh_device_operation_adapter_v<device_operation_t>) {
        return get_operation_name<typename device_operation_t::device_operation_t>(
            operation_attributes);
    } else {
        return tt::stl::get_type_name<device_operation_t>();
    }
}
```

### GraphProcessor and ProcessorHooks

`GraphProcessor` records a DAG of operations, tensors, and buffers:

```cpp
// Source: ttnn/api/ttnn/graph/graph_processor.hpp  (lines 46-126)
class GraphProcessor : public tt::tt_metal::IGraphProcessor {
public:
    GraphProcessor(RunMode mode);  // NORMAL or NO_DISPATCH

    void track_function_start(std::string_view function_name,
                              std::span<TrackedArgument> input_parameters) override;
    void track_function_end(const std::any& output) override;
    void track_allocate(const Buffer* buffer) override;
    void track_program(Program* program, const IDevice* device) override;

    struct Vertex {
        node_id counter = 0;
        std::string node_type;
        std::unordered_map<std::string, std::string> params;
        std::vector<std::string> arguments;
        std::vector<node_id> connections;
        std::vector<node_id> input_tensors;
        int stacking_level = 0;
    };

    // Capture API
    static void begin_graph_capture(RunMode mode);
    static nlohmann::json end_graph_capture();
};
```

`ProcessorHooks` controls whether operations actually execute during capture:

```cpp
// Source: ttnn/api/ttnn/graph/graph_processor.hpp  (lines 20-45)
class ProcessorHooks : public tt::tt_metal::IGraphHooks {
    bool do_block = false;
public:
    bool hook_allocate(const tt::tt_metal::Buffer* buffer) override;
    bool hook_program(tt::tt_metal::Program* program) override;
    // ...
    void set_block(bool block);
    bool get_block() const;
};
```

In `NO_DISPATCH` mode, `do_block = true`, which causes hooks to return `true` (meaning "I handled this, skip the real execution"). In `NORMAL` mode, `do_block = false`, and the hooks are non-blocking -- the graph is recorded alongside real execution.

### ScopedGraphCapture

A RAII wrapper for safe capture:

```cpp
// Source: ttnn/api/ttnn/graph/graph_processor.hpp  (lines 139-153)
class ScopedGraphCapture {
public:
    ScopedGraphCapture(GraphProcessor::RunMode mode);
    ~ScopedGraphCapture();
    nlohmann::json end_graph_capture();
};
```

Usage pattern:

```cpp
{
    ttnn::graph::ScopedGraphCapture capture(
        GraphProcessor::RunMode::NO_DISPATCH);
    // Operations here are traced but not dispatched
    auto result = ttnn::add(a, b);
    auto graph_json = capture.end_graph_capture();
}
```

### Workload Tracking

Each workload is also tracked via `mesh_device_operation_utils::track_workload`:

```cpp
// Source: ttnn/api/ttnn/mesh_device_operation_utils.hpp  (lines 94-101)
inline bool track_workload(
    tt::tt_metal::distributed::MeshWorkload& workload,
    ttnn::MeshDevice* mesh_device) {
    bool hook_program = false;
    for (auto& [_, program] : workload.get_programs()) {
        tt::tt_metal::GraphTracker::instance()
            .track_program(&program, mesh_device);
        hook_program |=
            tt::tt_metal::GraphTracker::instance()
                .hook_program(&program);
    }
    return hook_program;
}
```

If any program is hooked, the workload is intercepted and not enqueued -- `enqueue_mesh_workload` early-returns when `track_workload` returns `true`.

---

## 1.3.3 Tracy Profiling and Op Profiler

### TracyOpMeshWorkload Macro

After a mesh workload is enqueued, `enqueue_mesh_workload` calls the `TracyOpMeshWorkload` macro, which iterates per-coordinate programs, creates a `ZoneScopedN("TT_DNN_DEVICE_OP")` zone for each, and serializes operation metadata to JSON via `op_meta_data_serialized_json`. The critical parameter is `mesh_device_operation_t{}` — an instance of the operation type, which determines both the `op_code` string and attribute serialization. See [Section 6.3.1](../ch06_caching_tracing_profiling/03_tracy_profiling.md) for the full macro expansion with line-by-line analysis.

### Op Profiler JSON

The `op_meta_data_serialized_json` function (templated on the device operation type) produces a JSON blob containing:

- **`op_code`**: The C++ type name of the operation (e.g., `"ttnn::operations::binary::BinaryDeviceOperation"`), or a custom name if the operation provides `get_type_name`.
- **`attributes`**: All fields of `operation_attributes_t`, serialized via `tt::stl::reflection::get_attributes`.
- **`input_tensors`**: Shape, layout, dtype, and storage info for each input.
- **`output_tensors`**: Same for outputs.
- **`kernel_info`**: Compute and data movement kernel source paths, math fidelity, and binary sizes.
- **`performance_model`**: Compute/ideal/bandwidth nanoseconds from `create_op_performance_model`.
- **`op_hash`**: The program hash, used to correlate identical programs across invocations.

Example JSON output when `TTNN_OP_PROFILER=1`:

```json
{
  "global_call_count": 42,
  "op_code": "BinaryDeviceOperation",
  "op_type": "tt_dnn_device",
  "device_id": 0,
  "op_hash": 12345678,
  "attributes": { "binary_op_type": "ADD", "dtype": "BFloat16" },
  "input_tensors": [ { "shape": [1, 1, 32, 32], "dtype": "BFloat16" } ],
  "output_tensors": [ { "shape": [1, 1, 32, 32], "dtype": "BFloat16" } ],
  "kernel_info": {
    "compute_kernels": [ "eltwise_binary" ],
    "datamovement_kernels": [ "binary_reader", "unary_writer" ]
  },
  "performance_model": {
    "compute_ns": 0,
    "ideal_ns": 0,
    "bandwidth_ns": 0
  }
}
```

The profiler also maintains caches to avoid re-serializing the full JSON for programs that have already been profiled:

```cpp
// Source: ttnn/api/tools/profiler/op_profiler.hpp  (lines 83-84)
inline thread_safe_cached_ops_map cached_ops{};
inline thread_safe_call_stack call_stack;
```

Two lookup tables enable post-processing:

- **`RuntimeIDToOpName`** -- maps `(device_id, runtime_id)` to operation name.
- **`ProgramHashToOpName`** -- maps `(device_id, program_hash)` to operation name.

### Runtime ID Assignment

Before profiling, every program in the workload receives a monotonically increasing runtime ID:

```cpp
// Source: ttnn/api/ttnn/mesh_device_operation_utils.hpp  (lines 85-91)
inline void set_runtime_id(
    tt::tt_metal::distributed::MeshWorkload& workload) {
    auto op_id = ttnn::CoreIDs::instance()
        .fetch_and_increment_device_operation_id();
    tt::tt_metal::experimental::inspector::EmitMeshWorkloadRuntimeId(
        workload, op_id);
    for (auto& [_, program] : workload.get_programs()) {
        program.set_runtime_id(op_id);
    }
}
```

This runtime ID is the base from which `EncodePerDeviceProgramID` derives per-device IDs, ensuring every Tracy zone has a unique identifier.

### Environment Control

Profiling is opt-in via environment variable:

```cpp
// Source: ttnn/api/tools/profiler/op_profiler.hpp  (lines 107-113)
inline bool is_op_profiler_env_var_set() {
    const char* op_profiler_enable_str = std::getenv("TTNN_OP_PROFILER");
    if (op_profiler_enable_str != nullptr && op_profiler_enable_str[0] == '1') {
        op_profiler_is_enabled = true;
    }
    return op_profiler_is_enabled;
}
```

All profiling code is compiled out entirely when `TRACY_ENABLE` is not defined.

---

## 1.3.4 Inspector Annotations

The Inspector is an experimental facility that annotates mesh workloads with operation metadata for external debugging tools:

```cpp
// Source: tt_metal/api/tt-metalium/experimental/inspector.hpp
namespace tt::tt_metal::experimental::inspector {
    bool IsEnabled();
    void EmitMeshWorkloadAnnotation(
        MeshWorkload& workload,
        std::string_view operation_name,
        std::string_view operation_parameters);
    void EmitMeshWorkloadRuntimeId(
        MeshWorkload& workload, uint64_t runtime_id);
}
```

The `emit_mesh_workload_annotation` helper in `device_operation.hpp` extracts tensors from the templated `tensor_args_t` and delegates to the non-template `emit_mesh_workload_annotation_impl`:

```cpp
// Source: ttnn/api/ttnn/device_operation.hpp  (lines 255-267)
template <DeviceOperationConcept mesh_device_operation_t>
void emit_mesh_workload_annotation(
    tt::tt_metal::distributed::MeshWorkload& workload,
    const auto& operation_attributes,
    const auto& tensor_args) {
    if (tt::tt_metal::experimental::inspector::IsEnabled()) {
        auto operation_name =
            get_operation_name<mesh_device_operation_t>(operation_attributes);
        std::vector<std::reference_wrapper<const Tensor>> tensors;
        tt::stl::reflection::visit_object_of_type<Tensor>(
            [&tensors](const Tensor& t) { tensors.push_back(std::cref(t)); },
            tensor_args);
        emit_mesh_workload_annotation_impl(workload, operation_name, tensors);
    }
}
```

The non-template `emit_mesh_workload_annotation_impl` lives in `device_operation_detail.hpp/cpp` to reduce template instantiation cost -- a deliberate design choice documented in the code.

---

## 1.3.5 Why `generic_op` Loses Identity

`GenericOp` is registered as a single operation type:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/generic_op.hpp  (lines 34-36)
namespace ttnn {
constexpr auto generic_op = ttnn::register_operation<
    "ttnn::generic_op", ttnn::operations::generic::GenericOp>();
}
```

Its underlying device operation, `GenericOpDeviceOperation`, has a single program factory variant:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp
struct GenericOpDeviceOperation {
    using operation_attributes_t = generic::operation_attributes_t;
    using program_factory_t = std::variant<program::GenericMeshProgramFactory>;
    // ...
};
```

### The Identity Loss Problem

When TT-Blaze or any other system dispatches through `generic_op`, the operation type seen by every subsystem is `GenericOpDeviceOperation`, not the original semantic operation (e.g., "matmul", "softmax", "attention"):

| Subsystem | What It Sees | What You Want | Consequence |
|-----------|-------------|---------------|-------------|
| **Program cache** | `type_hash<GenericOpDeviceOperation>` | Per-op-type hash | All generic ops share one type hash; uniqueness depends entirely on `MeshProgramDescriptor` content hash |
| **Graph tracer** | `"GenericOpDeviceOperation"` | `"BlazeMatmul"`, `"BlazeSoftmax"` | Graph nodes lose semantic meaning; all generic dispatches appear identical in the DAG |
| **Tracy profiler** | `GenericOpDeviceOperation{}` in zone | Per-op-type zones | Timeline shows one name for every zone, making performance attribution impossible |
| **Inspector** | `"GenericOpDeviceOperation"` | Semantic operation name | Workload annotations lack true identity |
| **Validation** | Generic shape checks only | Op-specific invariants | Cannot check e.g. "matmul requires $M \times K$ and $K \times N$ shapes" |
| **Output specs** | Passthrough from user pre-allocation | Inferred from input shapes | Cannot compute output shapes from input shapes because operation semantics are unknown |

### How generic_op Compensates (Partially)

`GenericOpDeviceOperation` provides `compute_program_hash` to avoid total cache chaos:

`GenericOpDeviceOperation::compute_program_hash` walks the full `MeshProgramDescriptor`, hashing each `(MeshCoordinateRange, ProgramDescriptor)` pair, or short-circuits via `custom_program_hash` if set. See [Section 2.1.5](../ch02_generic_op_dispatch/01_generic_op_internals.md) for the full implementation and cost analysis.

The hash can also be user-supplied via `program_descriptor.custom_program_hash`, giving callers full control over cache behavior. But this is a runtime hash, not a type-level distinction -- the profiler, graph tracer, and inspector still see `GenericOpDeviceOperation`.

The `override_runtime_arguments` for `GenericMeshProgramFactory` is notably more expensive than a typical implementation because it must handle arbitrary kernel counts and CB configurations:

`GenericMeshProgramFactory::override_program_runtime_arguments` iterates every kernel handle and core coordinate to copy updated runtime args, then patches CB configs (total sizes, page sizes, dynamic addresses). This is more expensive than a typical typed operation's override because it handles arbitrary kernel counts and CB configurations. See [Section 2.1.7](../ch02_generic_op_dispatch/01_generic_op_internals.md) for the full implementation.

### Mitigation Strategies

Three concrete approaches can address the identity-loss problem:

**Strategy A: Naming metadata in `MeshProgramDescriptor`.** The `ProgramDescriptor::custom_program_hash` field already allows user control over caching. Extending `MeshProgramDescriptor` with an optional `operation_name` field and having `get_type_name` read it would solve the profiling problem. The op profiler already has a conditional path:

```cpp
if constexpr (requires { device_operation_t::get_type_name(operation_attributes); }) {
    opName = device_operation_t::get_type_name(operation_attributes);
}
```

If `GenericOpDeviceOperation` implemented this method to read an operation name from its attributes, Tracy and Inspector identity would be restored without changing the dispatch architecture.

**Strategy B: Dedicated device operations per logical op.** Instead of routing everything through `generic_op`, TT-Blaze could register each logical operation as its own `DeviceOperationConcept`-conforming type. This is the approach taken by native TTNN operations and gives full identity in all systems -- including type-level cache isolation, op-specific validation, and output spec inference -- but requires more boilerplate per operation.

**Strategy C: Hybrid approach.** Use `generic_op` for dispatch during development and gradually promote high-frequency operations to dedicated device operation types. This balances iteration speed with production-quality observability. Strategy A provides the naming, Strategy B provides the full contract benefits, and a project can mix both.

In short, `generic_op` is an escape hatch that trades all of the framework's type-level guarantees for dispatch flexibility. Promoting a TT-Blaze or blaze-nn operation to a first-class TTNN citizen means defining a proper `DeviceOperationConcept`-satisfying struct that restores identity to every layer. The subsequent chapters address exactly this.

---

## Key Takeaways

1. **`NO_DISPATCH` graph capture must skip caching**: buffer addresses are invalid (address=0) during capture, so caching those programs would poison the cache for later real execution -- a non-obvious invariant that would cause silent data corruption if violated.

2. **All four observability systems (`GraphTracker`, Tracy, Inspector, program cache) derive identity from the same source**: `type_hash<device_operation_t>` and `get_type_name<device_operation_t>`. Any operation that collapses to a single type (like `generic_op`) loses identity in all four simultaneously.

3. **`generic_op` identity loss is recoverable without changing the dispatch architecture**: `GenericOpDeviceOperation::get_type_name` can read a name from `MeshProgramDescriptor` attributes (Strategy A), restoring Tracy/Inspector identity while keeping the existing code path. Full contract benefits (type-level cache isolation, op-specific validation) require promoting to a dedicated `DeviceOperationConcept` struct (Strategy B).

---

**Next:** [Chapter 2 -- How generic_op Works as Blaze's Dispatch Mechanism](../ch02_generic_op_dispatch/index.md)
