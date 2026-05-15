# 2.1 GenericOp / GenericOpDeviceOperation Internals

## Opening Summary

`ttnn::generic_op` occupies a unique position in the TTNN operation taxonomy. Where every other registered operation (e.g., `ttnn::matmul`, `ttnn::softmax`) encodes *what* computation to perform and lets its own `DeviceOperation` subclass decide *how* to build the program, `generic_op` inverts the relationship: it accepts a fully-built program description from the caller and merely dispatches it to hardware. This section traces every C++ component in the `ttnn/cpp/ttnn/operations/generic/` directory, documenting the type aliases, contract implementations, program hash strategy, factory mechanics, and the `ProgramDescriptor` / `MeshProgramDescriptor` data model that `generic_op` consumes.

---

## 2.1.1 The `GenericOp` Struct and Registration

The top-level entry point lives in `generic_op.hpp`:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/generic_op.hpp
namespace ttnn {
constexpr auto generic_op =
    ttnn::register_operation<"ttnn::generic_op",
                              ttnn::operations::generic::GenericOp>();
}
```

This makes `generic_op` a first-class TTNN operation (it participates in the `ttnn::register_operation` decorator system described in [Chapter 1, Section 2](../ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md)), yet its semantics are *pass-through* -- it performs no tensor-shape reasoning, no output allocation, and no program-factory selection.

The string `"ttnn::generic_op"` is the operation's **permanent identity** in the TTNN infrastructure. Every Blaze op -- matmul, mcast, gather, RMSNorm, gated MLP -- appears under this single name in program cache keys, graph traces, Tracy profiler timelines, and Inspector annotations. This identity erasure is the root cause of the observability gap that first-class registration aims to close.

`GenericOp` defines two `invoke()` overloads:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/generic_op.hpp
struct GenericOp {
    // Primary: per-device control via MeshProgramDescriptor
    static Tensor invoke(
        const std::vector<Tensor>& io_tensors,
        const MeshProgramDescriptor& mesh_program_descriptor);

    // Convenience: same ProgramDescriptor on every device (SPMD)
    static Tensor invoke(
        const std::vector<Tensor>& io_tensors,
        const ProgramDescriptor& program_descriptor);
};
```

### SPMD Adapter

The single-`ProgramDescriptor` overload is a convenience wrapper that constructs a `MeshProgramDescriptor` covering the full mesh:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/generic_op.cpp
Tensor GenericOp::invoke(
    const std::vector<Tensor>& io_tensors,
    const ProgramDescriptor& program_descriptor) {
    TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
    auto* mesh_device = io_tensors.front().device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

    MeshProgramDescriptor mesh_pd;
    mesh_pd.mesh_programs.emplace_back(
        MeshCoordinateRange(mesh_device->shape()),
        program_descriptor);
    return invoke(io_tensors, mesh_pd);
}
```

All paths converge to the mesh-program overload, which delegates to `ttnn::prim::generic_op()`. There is no single-device fast path -- even a 1x1 mesh uses `MeshProgramDescriptor`.

**Source file:** `ttnn/cpp/ttnn/operations/generic/generic_op.cpp`

---

## 2.1.2 Type Aliases: `operation_attributes_t`, `tensor_args_t`

The types file establishes the `DeviceOperationConcept` aliases:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation_types.hpp
using operation_attributes_t = tt::tt_metal::experimental::MeshProgramDescriptor;
using tensor_return_value_t  = Tensor;
using spec_return_value_t    = TensorSpec;

struct tensor_args_t {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;  // NOTE: last element of io_tensors
};
```

Key observations:

1. **`operation_attributes_t` IS `MeshProgramDescriptor`.** In a native TTNN op this would be a struct with meaningful fields (e.g., `BinaryOpType op_type`, `std::optional<float> scalar`). Here it is the entire program description -- kernels, circular buffers, semaphores, compile-time args -- flattened into a single opaque blob. This is the fundamental architectural difference from native ops.

2. **`tensor_args_t` references, not owns.** The `io_tensors` vector is passed by const reference; the `output_tensor` is an alias for `io_tensors.back()`. No copies occur.

3. **`MeshProgramDescriptor` itself** is a thin wrapper:

```cpp
// Source: tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp
struct MeshProgramDescriptor {
    using MeshPrograms = std::vector<std::pair<
        distributed::MeshCoordinateRange, ProgramDescriptor>>;
    MeshPrograms mesh_programs;

    static constexpr auto attribute_names = std::forward_as_tuple("num_mesh_programs");
    auto attribute_values() const { return std::make_tuple(mesh_programs.size()); }
};
```

The `attribute_names`/`attribute_values()` reflection support only exposes `num_mesh_programs` -- a count, not the content. This means the standard TTNN reflection-based hashing (used by Inspector and graph tracing) sees only the number of sub-programs, not what they contain.

**Source file:** `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation_types.hpp`

---

## 2.1.3 The `GenericOpDeviceOperation` Contract

`GenericOpDeviceOperation` implements the `DeviceOperation` concept required by `ttnn::device_operation::launch<>()`. The following table summarizes every required method and contrasts it against native op behavior:

| Method | Native Op Behavior | `GenericOpDeviceOperation` Behavior |
|--------|-------------------|--------------------------------------|
| `compute_output_specs()` | Infers output shape/dtype from inputs + attributes | Returns `tensor_args.output_tensor.tensor_spec()` verbatim -- the caller pre-allocated the output |
| `create_output_tensors()` | Allocates output tensor(s) on device | Returns `tensor_args.output_tensor` unchanged -- no allocation |
| `validate_on_program_cache_miss()` | Full semantic validation | Only checks for duplicate `MeshCoordinateRange` entries |
| `validate_on_program_cache_hit()` | Lightweight re-validation | Same duplicate-range check |
| `validate_inputs()` | Static pre-dispatch validation | Declared but **not implemented** |
| `compute_program_hash()` | Hash of semantic attributes | Hash of full descriptor contents (or caller-provided `custom_program_hash`) |

### Single-Variant `program_factory_t`

```cpp
using program_factory_t = std::variant<program::GenericMeshProgramFactory>;
```

Native TTNN ops often use multiple program factory variants (e.g., matmul has six: multicore, reuse-optimized, mcast-1D, mcast-2D, DRAM-sharded, batched). `GenericOpDeviceOperation` has exactly **one variant**: `GenericMeshProgramFactory`. The variant dispatch described in [Chapter 1, Section 1](../ch01_ttnn_registration_contract/01_device_operation_concept.md) always selects index 0. This simplification is possible because the program is fully constructed before reaching `generic_op`; there is no compile-time selection between factory strategies.

### `compute_output_specs()` -- Passthrough

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
spec_return_value_t GenericOpDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // User has to do this. Just referencing last element (preallocated output tensor).
    return tensor_args.output_tensor.tensor_spec();
}
```

In TTNN's `NO_DISPATCH` mode (used for graph tracing and validation without actual device execution), this method is called to determine the output shape and layout. Because `generic_op` merely echoes the existing spec, it cannot participate in spec inference -- an important limitation for graph-traced pipelines (see [Section 2.3](./03_divergence_analysis.md)).

### `create_output_tensors()` -- No Allocation

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
tensor_return_value_t GenericOpDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // Don't create anything, user is passing output tensor.
    return tensor_args.output_tensor;
}
```

This is the strongest divergence from native ops. The TTNN output allocation infrastructure is bypassed entirely. The caller (Blaze's `BlazeCompiler` or `FusedProgram`) is responsible for allocating, sizing, and placing the output tensor before calling `generic_op`.

**Source file:** `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp`

---

## 2.1.4 Validation: Minimal by Design

Both validation methods delegate to the same check:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
void verify_no_duplicate_mesh_coord_ranges(
    const MeshProgramDescriptor::MeshPrograms& mesh_programs) {
    std::unordered_set<MeshCoordinateRange> seen;
    seen.reserve(mesh_programs.size());
    for (const auto& [range, _] : mesh_programs) {
        auto [it, inserted] = seen.insert(range);
        TT_FATAL(inserted,
            "Duplicate MeshCoordinateRange found in MeshProgramDescriptor: {}", range);
    }
}

void GenericOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t&) {
    verify_no_duplicate_mesh_coord_ranges(attributes.mesh_programs);
}

void GenericOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t&) {
    verify_no_duplicate_mesh_coord_ranges(attributes.mesh_programs);
}
```

This is the **only validation** `generic_op` performs. Compare with a native TTNN op like `MatmulDeviceOperation`, which validates input tensor shapes, data type compatibility, memory layouts, shard specs, and output spec computability. `generic_op` performs none of these checks. All validation responsibility falls on the Blaze compiler.

> **Note**: `validate_inputs()` is declared in the header but has no implementation in the `.cpp` file. The `DeviceOperationConcept` makes `validate_inputs` optional (see [Chapter 1, Section 1](../ch01_ttnn_registration_contract/01_device_operation_concept.md)), so this is valid -- it simply means no per-input validation occurs.

---

## 2.1.5 Program Hash Computation

The program hash determines whether a cached `Program` object can be reused. Two paths exist:

### Path 1: Custom Hash (Caller-Provided)

If `ProgramDescriptor.custom_program_hash` is set, it is returned directly:

```cpp
if (program_descriptor.custom_program_hash) {
    return *program_descriptor.custom_program_hash;
}
```

This allows sophisticated callers (including Blaze) to pre-compute a semantic hash. **Blaze does not currently use this.**

### Path 2: Content Hash (Default)

Without a custom hash, every structural element of the `ProgramDescriptor` is hashed:

$$
h = \bigoplus_{k \in \text{kernels}} h_k \oplus \bigoplus_{cb \in \text{cbs}} h_{cb} \oplus \bigoplus_{s \in \text{sems}} h_s
$$

Where $\oplus$ denotes `ttsl::hash::hash_combine`. For each kernel $k$:

$$
h_k = \text{hash}(\texttt{kernel\_source}, \texttt{source\_type}, \texttt{core\_ranges}, \texttt{compile\_time\_args}, \texttt{named\_ct\_args}, \texttt{defines}, |\texttt{common\_rt\_args}|, |\texttt{rt\_args}|, \texttt{config\_index}, \texttt{config})
$$

For each CB:

$$
h_{cb} = \text{hash}(\texttt{total\_size}) \oplus \bigoplus_{r \in \text{core\_ranges}} h_r \oplus |\texttt{format\_descs}| \oplus \bigoplus_{f \in \text{format\_descs}} h_f \oplus \ldots
$$

This means **any change to any descriptor field produces a new program-cache entry.** For a typical Blaze fused kernel with 3 RISC kernels and 8 circular buffers, the hash walks all kernel sources, compile-time args, core ranges, CB descriptors, and semaphore descriptors. The cost is $O(\sum |K_i| + \sum |CB_j| + \sum |S_k|)$, in contrast to native TTNN ops that hash only their compact `operation_attributes_t` struct.

> **Important**: Runtime arg **values** are not hashed -- only their **count** is. This means programs that differ only in runtime args (e.g., different tile counts for different sequence lengths) will hash identically, enabling cache reuse with runtime arg override.

**Source file:** `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp`

---

## 2.1.6 The Primitive Entry Point

The actual dispatch happens through `ttnn::prim::generic_op`:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
namespace ttnn::prim {
tensor_return_value_t generic_op(
    const std::vector<Tensor>& io_tensors,
    const operation_attributes_t& operation_attributes) {
    using OperationType = GenericOpDeviceOperation;
    TT_FATAL(
        io_tensors.size() >= 2,
        "io_tensors must contain at least one input tensor and one output tensor, got {} tensors.",
        io_tensors.size());

    auto tensor_args = OperationType::tensor_args_t{
        .io_tensors = io_tensors,
        .output_tensor = io_tensors.back()
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}
```

Key observations:

1. **The last tensor in `io_tensors` is the output.** This is a convention, not enforced by the type system.
2. **The minimum is 2 tensors** (one input, one output). Multi-input/multi-output ops pass all tensors in a flat vector.
3. **`device_operation::launch<>()`** handles program caching, workload submission, and device synchronization -- identical infrastructure to native ops (see [Chapter 1, Section 2](../ch01_ttnn_registration_contract/02_registration_dispatch_and_bindings.md)).

---

## 2.1.7 `GenericMeshProgramFactory`

The program factory is the bridge between the descriptor-based world and `tt_metal::Program` objects. It is responsible for two tasks: creating the initial `MeshWorkload` (on cache miss) and updating runtime arguments (on cache hit).

### Shared Variables

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.hpp
struct GenericMeshProgramFactory {
    struct shared_variables_t {
        uint32_t num_kernel_handles{};
        std::vector<tt::tt_metal::CBHandle> cb_handles;
    };

    struct mesh_shared_variables_t {
        shared_variables_t program_shared_variables;
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<mesh_shared_variables_t>;
};
```

The `shared_variables_t` captures the kernel handle count and CB handles from the initial program creation. These are stored alongside the cached `Program` object in the program cache and are used during runtime argument override on cache hits.

### `create_mesh_workload()`

On a cache miss, this method iterates over each `(MeshCoordinateRange, ProgramDescriptor)` pair and builds per-device programs:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp
cached_mesh_workload_t GenericMeshProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinateRangeSet& /*tensor_coords*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    MeshWorkload mesh_workload;
    for (const auto& [range, pd] : operation_attributes.mesh_programs) {
        auto cached = create_at(pd, tensor_args, tensor_return_value);
        mesh_workload.add_program(range, std::move(cached.program));
    }
    return cached_mesh_workload_t{std::move(mesh_workload), ...};
}
```

### `create_at()` -- Single-Device Program Construction

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp
cached_program_t GenericMeshProgramFactory::create_at(
    const ProgramDescriptor& program_descriptor, ...) {
    Program program{program_descriptor};   // <-- descriptor-driven constructor
    shared_variables_t shared_vars;
    for (const auto& cb : program.circular_buffers()) {
        shared_vars.cb_handles.push_back(cb->id());
    }
    shared_vars.num_kernel_handles = program_descriptor.kernels.size();
    return {std::move(program), std::move(shared_vars)};
}
```

The key line is `Program{program_descriptor}` -- the `ProgramDescriptor` is consumed by the `Program` constructor (inside TT-Metal), which translates all descriptors to their runtime equivalents: `CreateKernel`, `CreateCircularBuffer`, `CreateSemaphore`. This is the point where Blaze's Python-constructed descriptor becomes a concrete tt-metalium `Program` ready for device dispatch.

### `override_runtime_arguments()` -- Cache Hit Fast Path

On a cache hit, only runtime arguments and CB configurations are updated:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp
void override_program_runtime_arguments(
    Program& program,
    shared_variables_t& shared_vars,
    const ProgramDescriptor& program_descriptor) {
    // 1. Update kernel runtime args
    for (size_t kernel_handle = 0; kernel_handle < shared_vars.num_kernel_handles; ++kernel_handle) {
        const auto& kernel_desc = program_descriptor.kernels[kernel_handle];
        for (const auto& [core_coord, runtime_arg] : kernel_desc.runtime_args) {
            if (!runtime_arg.empty()) {
                auto& cached_runtime_args = GetRuntimeArgs(program, kernel_handle, core_coord);
                std::copy(runtime_arg.begin(), runtime_arg.end(), cached_runtime_args.data());
            }
        }
        if (!kernel_desc.common_runtime_args.empty()) {
            auto& cached = GetCommonRuntimeArgs(program, kernel_handle);
            std::copy(kernel_desc.common_runtime_args.begin(),
                      kernel_desc.common_runtime_args.end(),
                      cached.data());
        }
    }

    // 2. Update circular buffer configurations
    for (size_t cb_idx = 0; cb_idx < program_descriptor.cbs.size(); ++cb_idx) {
        const auto& cb_desc = program_descriptor.cbs[cb_idx];
        auto cb_handle = shared_vars.cb_handles[cb_idx];
        const auto& cb_config = GetCircularBufferConfig(program, cb_handle);

        if (cb_config.total_size() != cb_desc.total_size) {
            UpdateCircularBufferTotalSize(program, cb_handle, cb_desc.total_size);
        }
        // ... page size updates, dynamic buffer address updates ...
    }
}
```

The three update categories are:

1. **Kernel runtime args**: copies per-core and common runtime args from the new descriptor into the cached program, asserting that sizes match.
2. **Circular buffer sizes**: updates `total_size` and per-format `page_size` when they differ from the cached values.
3. **Dynamic CB addresses**: updates `Buffer*` and `GlobalCircularBuffer*` pointers for dynamically-addressed CBs.

This override mechanism is what makes the program cache effective for Blaze: the compile-time structure (kernel source, core ranges, compile-time args) stays fixed across invocations, while runtime-varying parameters (tile counts, buffer addresses) are patched in-place.

---

## 2.1.8 The `ProgramDescriptor` and `MeshProgramDescriptor` Data Model

These are the data structures Blaze builds and `generic_op` consumes. Understanding their fields is essential for following the compilation pipeline in [Section 2.2](./02_blaze_compilation_pipeline.md).

```cpp
// Source: tt_metal/api/tt-metalium/program_descriptors.hpp
struct ProgramDescriptor {
    using KernelDescriptors    = SmallVector<KernelDescriptor, 3>;
    using SemaphoreDescriptors = SmallVector<SemaphoreDescriptor, 3>;
    using CBDescriptors        = SmallVector<CBDescriptor, 5>;

    KernelDescriptors kernels;
    SemaphoreDescriptors semaphores;
    CBDescriptors cbs;
    std::optional<uint64_t> custom_program_hash;
};
```

### `KernelDescriptor` Fields

| Field | Type | Purpose |
|-------|------|---------|
| `kernel_source` | `string` | File path or inline source code |
| `source_type` | `SourceType` | `FILE_PATH` or `SOURCE_CODE` |
| `core_ranges` | `CoreRangeSet` | Which cores run this kernel |
| `compile_time_args` | `vector<uint32_t>` | Positional CT args |
| `named_compile_time_args` | `vector<pair<string, uint32_t>>` | Named CT args |
| `defines` | `vector<pair<string, string>>` | Preprocessor defines |
| `runtime_args` | `vector<pair<CoreCoord, vector<uint32_t>>>` | Per-core RT args |
| `common_runtime_args` | `vector<uint32_t>` | Broadcast RT args |
| `config` | `variant<Reader, Writer, DataMovement, Compute>` | RISC processor config |

This is the complete specification of a kernel. When `Program{program_descriptor}` is constructed, each `KernelDescriptor` becomes a compiled kernel on the device.

### `MeshProgramDescriptor` Wrapping

```cpp
// Source: tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp
struct MeshProgramDescriptor {
    using MeshPrograms = vector<pair<MeshCoordinateRange, ProgramDescriptor>>;
    MeshPrograms mesh_programs;
};
```

This descriptor is the **complete** specification of what runs on hardware. There are no callbacks, no factory selection, no shape inference -- just a static description of kernels, buffers, and synchronization primitives per device.

---

## 2.1.9 Python Bindings

The nanobind layer exposes both overloads to Python:

```cpp
// Source: ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp
bind_registered_operation(
    mod, ttnn::generic_op, doc,
    // MeshProgramDescriptor overload
    nanobind_overload_t{mesh_program_invoke,
        nb::arg("io_tensors"), nb::arg("mesh_program_descriptor")},
    // ProgramDescriptor overload (SPMD)
    nanobind_overload_t{program_invoke,
        nb::arg("io_tensors"), nb::arg("program_descriptor")}
);
```

From Python, the call looks like:

```python
# SPMD (single ProgramDescriptor, broadcast to all devices)
ttnn.generic_op(io_tensors, program_descriptor)

# Mesh (explicit per-device control)
ttnn.generic_op(io_tensors, mesh_program_descriptor)
```

This is the exact call that `_run_program()` in `blaze/compiler.py` makes (see [Section 2.2](./02_blaze_compilation_pipeline.md)).

**Source file:** `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp`

---

## 2.1.10 Architectural Data Flow

```
Python caller (Blaze)
    |
    |  ttnn.generic_op(io_tensors, descriptor)
    v
GenericOp::invoke()                          <-- generic_op.cpp
    |
    |  Wraps PD -> MeshPD if SPMD
    v
ttnn::prim::generic_op()                    <-- generic_op_device_operation.cpp
    |
    |  Builds tensor_args_t {io_tensors, output=last}
    v
ttnn::device_operation::launch<GenericOpDeviceOperation>(attrs, tensor_args)
    |                                        <-- device_operation.hpp
    |-- compute_output_specs()   -> passthrough (reads output tensor spec)
    |-- create_output_tensors()  -> passthrough (returns caller's output)
    |-- compute_program_hash()   -> deep hash of MeshProgramDescriptor
    |
    +-- Cache MISS:
    |   |-- validate_on_program_cache_miss()  -> check coord uniqueness only
    |   |-- GenericMeshProgramFactory::create_mesh_workload()
    |   |       |
    |   |       +-- create_at(ProgramDescriptor)
    |   |               |
    |   |               +-- Program{program_descriptor}   <-- tt-metalium
    |   |               +-- Cache CB handles, kernel count
    |   |
    |   +-- Enqueue program on device
    |
    +-- Cache HIT:
        |-- validate_on_program_cache_hit()   -> check coord uniqueness only
        |-- GenericMeshProgramFactory::override_runtime_arguments()
        |       |
        |       +-- Patch RT args in-place
        |       +-- Update CB sizes/addresses if changed
        |
        +-- Enqueue cached program on device
```

---

## Key Takeaways

1. **`generic_op` is a registered TTNN operation** but implements the `DeviceOperation` concept with pass-through semantics -- the caller supplies everything.

2. **Its `operation_attributes_t` is the program itself** (`MeshProgramDescriptor`), unlike native ops where attributes are semantic parameters that a program factory interprets.

3. **Output tensors must be pre-allocated.** `compute_output_specs()` and `create_output_tensors()` are passthroughs that return the caller's tensor unchanged.

4. **Program hashing is structural, not semantic.** The hash captures everything that affects compilation ($O(\sum |K_i| + \sum |CB_j| + \sum |S_k|)$), while runtime arg values are excluded to enable the `override_runtime_arguments` fast path. The `custom_program_hash` bypass reduces this to $O(1)$ but is not yet used by Blaze.

5. **Validation is minimal.** Only mesh coordinate uniqueness is checked. All semantic validation (tensor shapes, data types, shard compatibility) is the Blaze compiler's responsibility.

6. **There is exactly one program factory** (`GenericMeshProgramFactory`) in the `program_factory_t` variant. Native ops like matmul have six. No `select_program_factory` logic exists because the program is already built.

---

| [Chapter 2 Index](index.md) | **Section 2.1** | [Section 2.2: Blaze Compilation Pipeline](02_blaze_compilation_pipeline.md) |
|:---|:---:|---:|
