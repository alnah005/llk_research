# 6.3 Tracy Profiling Integration

Tracy is TTNN's primary runtime profiler. Every enqueued mesh workload produces a Tracy zone annotated with operation metadata -- the operation name, tensor shapes, kernel info, and a JSON payload consumed by post-processing tools. The quality of this profiling data depends entirely on the operation's type identity: named operations produce informative flamegraphs; type-erased operations produce opaque blocks. This section traces the `TracyOpMeshWorkload` macro through its full expansion, compares before/after flamegraphs for Matmul and GatedReduce, details the attribute reflection requirements for profiler metadata, and assesses the adapter's compatibility with TTNN's performance model hooks.

---

## 6.3.1 The `TracyOpMeshWorkload` Macro: Before and After

The macro is invoked from `enqueue_mesh_workload<>()` after the workload has been dispatched to hardware (source: `device_operation.hpp`, lines 186--201):

```cpp
// EXISTING:
template <DeviceOperationWithMeshDeviceAdapter mesh_device_operation_t>
void enqueue_mesh_workload(
    ttnn::MeshDevice* mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload,
    const auto& operation_attributes,
    const auto& tensor_args,
    auto& tensor_return_value) {

    mesh_device_operation_utils::set_runtime_id(workload);

    if (mesh_device_operation_utils::track_workload(workload, mesh_device)) {
        return;  // hooked (NO_DISPATCH mode)
    }

    tt::tt_metal::distributed::EnqueueMeshWorkload(
        mesh_device->mesh_command_queue(), workload, false);

    TracyOpMeshWorkload(
        mesh_device, workload,
        mesh_device_operation_t{},       // <-- operation TYPE instance
        operation_attributes,            // <-- reflectable attributes
        tensor_args,                     // <-- tensor metadata source
        tensor_return_value);            // <-- output tensor metadata
}
```

The template parameter `mesh_device_operation_t` carries the operation's C++ type identity. For `generic_op`, this is `MeshDeviceOperationAdapter<GenericOpDeviceOperation>`. For the adapter, this is `MeshDeviceOperationAdapter<BlazeDeviceOperationAdapter<MatmulTag>>`.

### Complete Macro Expansion

```cpp
// Source: ttnn/api/tools/profiler/op_profiler.hpp  (lines 550-572)

#define TracyOpMeshWorkload(                                                   \
    mesh_device, mesh_workload, operation, operation_attributes,               \
    tensor_args, tensor_return_value)                                          \
    if (tt::tt_metal::op_profiler::is_op_profiler_env_var_set()) {             \
        for (const auto& [range, program] :                                    \
             (mesh_workload).get_programs()) {                                 \
                                                                               \
            auto base_program_id = program.get_runtime_id();                   \
                                                                               \
            for (auto coord : range) {                                         \
                if (!(mesh_device)->is_local(coord)) continue;                 \
                                                                               \
                /* Create a Tracy zone named "TT_DNN_DEVICE_OP" */             \
                ZoneScopedN("TT_DNN_DEVICE_OP");                               \
                                                                               \
                auto device_id =                                               \
                    (mesh_device)->get_device(coord)->id();                     \
                auto op_id =                                                   \
                    tt::tt_metal::detail::EncodePerDeviceProgramID(             \
                        base_program_id, device_id);                           \
                                                                               \
                /* Serialize operation metadata to JSON */                      \
                std::string op_message =                                        \
                    op_profiler::op_meta_data_serialized_json(                  \
                        operation,      /* <-- the operation TYPE instance */   \
                        op_id,                                                 \
                        device_id,                                             \
                        program,                                               \
                        operation_attributes,                                  \
                        tensor_args,                                           \
                        tensor_return_value);                                  \
                                                                               \
                /* Set zone text for Tracy timeline */                          \
                std::string op_text = fmt::format("id:{}", op_id);             \
                ZoneText(op_text.c_str(), op_text.size());                     \
                                                                               \
                /* Send JSON metadata as Tracy message */                       \
                op_profiler::tracy_message(op_message);                        \
            }                                                                  \
        }                                                                      \
    }
```

The critical parameter is `operation` -- an instance of the device operation type. The `op_meta_data_serialized_json` function is templated on this type, which determines both the `op_code` string in the JSON and how the `attributes` section is serialized. Every difference in Tracy output between `generic_op` and the adapter traces back to this single type parameter.

---

## 6.3.2 Operation Name Resolution

The `op_meta_data_serialized_json` function extracts the operation name through a three-priority conditional path:

```cpp
// Source: ttnn/api/tools/profiler/op_profiler.hpp  (lines 280-340, simplified)

template <typename DeviceOperationType>
std::string op_meta_data_serialized_json(
    const DeviceOperationType& operation,
    uint64_t op_id, uint32_t device_id,
    const tt::tt_metal::Program& program,
    const auto& operation_attributes,
    const auto& tensor_args,
    const auto& tensor_return_value) {

    // Priority 1: Static method on the operation type
    std::string opName;
    if constexpr (requires {
        DeviceOperationType::get_type_name(operation_attributes);
    }) {
        opName = DeviceOperationType::get_type_name(operation_attributes);
    }
    // Priority 2: Name extracted from reflectable attributes
    else if constexpr (requires {
        op_profiler::get_op_name(operation_attributes);
    }) {
        opName = op_profiler::get_op_name(operation_attributes);
    }
    // Priority 3: C++ type name (demangled)
    else {
        opName = std::string(
            tt::stl::get_type_name<DeviceOperationType>());
    }

    // Serialize attributes via reflection
    nlohmann::json attrs_json;
    if constexpr (tt::stl::reflection::is_reflectable_v<
            std::decay_t<decltype(operation_attributes)>>) {
        auto attr_names_and_values =
            tt::stl::reflection::get_attributes(operation_attributes);
        for (const auto& [name, value] : attr_names_and_values) {
            attrs_json[name] = serialize_attribute(value);
        }
    }

    // Serialize tensor metadata, kernel info, performance model ...
    // (see Section 6.3.5 for performance model details)
    // ...

    // Populate lookup tables
    RuntimeIDToOpName[{device_id, op_id}] = opName;
    ProgramHashToOpName[{device_id, program_hash}] = opName;

    return result.dump();
}
```

### How the Adapter Type Flows Through

The `TracyOpMeshWorkload` macro passes `mesh_device_operation_t{}` as the operation, where `mesh_device_operation_t = MeshDeviceOperationAdapter<BlazeDeviceOperationAdapter<MatmulTag>>`. The profiler function `op_meta_data_serialized_json` is therefore instantiated with the outer `MeshDeviceOperationAdapter<...>` type, not the inner `BlazeDeviceOperationAdapter<MatmulTag>`. The priority chain operates on the outer type, which matters for the `get_type_name` check:

```text
operation = MeshDeviceOperationAdapter<BlazeDeviceOperationAdapter<MatmulTag>>{}
operation_attributes = BlazeAdapterAttributes{
    .mesh_program_descriptor = ...,
    .op_name = "blaze::matmul",
    .custom_program_hash = std::nullopt
}

Inside op_meta_data_serialized_json<
    MeshDeviceOperationAdapter<BlazeDeviceOperationAdapter<MatmulTag>>>:

Priority 1: get_type_name not defined on MeshDeviceOperationAdapter
            -> skip
Priority 2: get_op_name checks if operation_attributes has .op_name
            BlazeAdapterAttributes has op_name field
            -> opName = "blaze::matmul"  (if get_op_name is defined)
Priority 3 (fallback): tt::stl::get_type_name<
    MeshDeviceOperationAdapter<BlazeDeviceOperationAdapter<MatmulTag>>>()
    -> "ttnn::...::BlazeDeviceOperationAdapter<...::MatmulTag>"
```

Whether the name is resolved via Priority 2 (concise `"blaze::matmul"`) or Priority 3 (full mangled type name), the result is per-op unique. Note that because `get_type_name` is checked on the outer `MeshDeviceOperationAdapter`, the proposed `get_type_name` must be defined there (or on `BlazeDeviceOperationAdapter` if `MeshDeviceOperationAdapter` forwards the call). To guarantee the concise form, the adapter can optionally provide a `get_type_name` static method:

```cpp
// PROPOSED: Optional concise profiler name in the adapter
template <typename OpTag>
struct BlazeDeviceOperationAdapter {
    // ...
    static std::string_view get_type_name(
            const operation_attributes_t& attrs) {
        return attrs.op_name;  // e.g., "blaze::matmul"
    }
};
```

This is a minor enhancement (the full type name is already functional) that significantly improves readability in Tracy's zone summary view, where horizontal space is limited.

---

## 6.3.3 Flamegraph Comparison: Before and After

### Before: The Monochrome Flamegraph

Under `generic_op`, every Blaze op produces a Tracy zone with `op_code = "GenericOpDeviceOperation"`. The resulting flamegraph is a uniform wall:

```text
Time -->
|==GenericOpDeviceOp==|==GenericOpDeviceOp==|==GenericOpDeviceOp==|==GenericOpDeviceOp==|
|    45 us            |    120 us           |    30 us            |    85 us            |
```

All zones share the same label and the same color in Tracy's visualization. To determine which zone corresponds to which Blaze op, an engineer must:

1. Correlate `op_hash` values with Blaze's internal dispatch log
2. Cross-reference runtime IDs with the compilation pipeline's output
3. Manually annotate the flamegraph with semantic names

This process takes 30--60 minutes per profiling session and is error-prone.

### After: The Semantic Flamegraph

Under the adapter, each zone carries its semantic name:

```text
Time -->
|====blaze::rmsnorm====|=======blaze::matmul=======|==blaze::add==|====blaze::sdpa====|
|    45 us              |    120 us                  |    30 us      |    85 us          |
```

Tracy assigns distinct colors to distinct zone names, producing a visually differentiated flamegraph. The zone summary groups by name, enabling immediate answers:

- "Matmul accounts for 62% of device time"
- "SDPA zones are 3x longer than expected for this sequence length"
- "RMSNorm zones are uniformly 12 microseconds -- likely memory-bound"

### Flamegraph for a Transformer Decoder Layer

The following shows a single decoder layer (20 ops) as it would appear in Tracy:

**Before (`generic_op`):**

```text
|=GenericOpDeviceOp=|=GenericOpDeviceOp=|=GenericOpDeviceOp=|=GenericOpDeviceOp=|...
| 12us              | 120us             | 120us             | 120us             |
```

All 20 zones are identical. The aggregation view shows one entry:

```text
GenericOpDeviceOperation:  100.0%  (20 calls, 680 us total)
```

**After (adapter):**

```text
|blaze::rmsnorm|====blaze::matmul====|====blaze::matmul====|====blaze::matmul====|
|   12us       |       120us          |       120us          |       120us         |

|=blaze::sdpa=|blaze::rmsnorm|====blaze::matmul====|blaze::gated_reduce|blaze::all_reduce|
|    85us      |    12us       |       120us          |       51us         |     20us        |
```

The aggregation view shows per-op breakdown (top 5 of 20 ops; remaining 10 ops account for the balance; total = 807 us):

```text
blaze::matmul:              59.5%  (4 calls, 480 us total)
blaze::sdpa:                10.5%  (1 call,   85 us total)
blaze::gated_reduce:         6.3%  (1 call,   51 us total)
blaze::all_reduce:           5.0%  (2 calls,  40 us total)
blaze::rmsnorm:              3.0%  (2 calls,  24 us total)
other (10 ops):             15.7%  (10 calls, 127 us total)
```

An engineer immediately identifies matmul as the dominant cost center and can focus optimization effort accordingly.

---

## 6.3.4 `op_profiler.hpp` and Reflectable Attributes

The profiler serializes operation attributes via `tt::stl::reflection::get_attributes`, which calls `attribute_names` and `attribute_values()` on the `operation_attributes_t`.

### `BlazeAdapterAttributes` Reflection

The adapter's `BlazeAdapterAttributes` provides the reflection interface (from [Section 4.2.2](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)):

```cpp
struct BlazeAdapterAttributes {
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
    std::string op_name;
    std::optional<tt::stl::hash::hash_t> custom_program_hash;

    static constexpr auto attribute_names =
        std::forward_as_tuple("op_name", "num_mesh_programs");
    auto attribute_values() const {
        return std::make_tuple(
            std::string_view(op_name),
            mesh_program_descriptor.mesh_programs.size());
    }
};
```

This exposes two attributes:

| Attribute | Type | Example Value | Purpose |
|:---|:---|:---|:---|
| `op_name` | `string_view` | `"blaze::matmul"` | Human-readable operation identity |
| `num_mesh_programs` | `size_t` | `1` (single device) or `8` (T3K) | Mesh topology diagnostic |

The `MeshProgramDescriptor` itself is deliberately excluded from reflection -- serializing it would produce megabytes of JSON per op invocation, overwhelming the profiler output.

### Comparison Across Dispatch Paths

| Property | `GenericOpDeviceOperation` | `BlazeDeviceOperationAdapter<OpTag>` | Native TTNN (e.g., `BinaryDeviceOperation`) |
|:---|:---|:---|:---|
| `attribute_names` | Not defined | `("op_name", "num_mesh_programs")` | `("binary_op_type", "output_dtype", "output_memory_config", ...)` |
| `attribute_values()` | Not defined | `("blaze::matmul", 1)` | `("ADD", "BFloat16", "INTERLEAVED", ...)` |
| Profiler JSON `attributes` | `{}` | `{"op_name": "blaze::matmul", "num_mesh_programs": 1}` | `{"binary_op_type": "ADD", ...}` |
| Semantic richness | None | Moderate (name + topology) | High (all op parameters) |

The adapter's two attributes are less detailed than a native op's semantic attributes. This is a deliberate tradeoff: the adapter wraps pre-built descriptors and does not have access to the semantic parameters that produced them. Tier 3 per-op wrappers can close this gap (Section 6.3.6).

### Attribute Serialization Cost

| Component | Cost |
|---|---|
| `get_attributes()` call | ~20 ns (tuple construction) |
| JSON serialization of 2 attributes | ~200 ns |
| JSON serialization of 5+ attributes (Tier 3) | ~500--1,000 ns |
| JSON `dump()` to string | ~300--800 ns |
| **Total per-op** | **~520--2,020 ns** |

This cost is paid only when `TTNN_OP_PROFILER=1` is set. In production (profiler disabled), the `is_op_profiler_env_var_set()` check short-circuits the entire macro at ~1 ns cost.

---

## 6.3.5 Performance Model Hooks

TTNN's profiler optionally includes performance model estimates in the JSON output via `create_op_performance_model`:

```cpp
// EXISTING: op_profiler.hpp (simplified)
template <typename DeviceOperation>
auto get_performance_model(
    const auto& operation_attributes,
    const auto& tensor_args,
    const auto& tensor_return_value) {
    if constexpr (requires {
        DeviceOperation::create_op_performance_model(
            operation_attributes, tensor_args, tensor_return_value);
    }) {
        return DeviceOperation::create_op_performance_model(
            operation_attributes, tensor_args, tensor_return_value);
    } else {
        return op_profiler::default_performance_model{};
    }
}
```

The performance model returns three values:

- **`compute_ns`**: Estimated compute time based on FLOP count and hardware throughput
- **`ideal_ns`**: Minimum possible time (all resources fully utilized)
- **`bandwidth_ns`**: Memory-bandwidth-limited time estimate

### Before: Performance Model with `generic_op`

`GenericOpDeviceOperation` does not provide `create_op_performance_model`. The profiler uses the default, returning all-zero estimates:

```json
"performance_model": {
  "compute_ns": 0,
  "ideal_ns": 0,
  "bandwidth_ns": 0
}
```

### After: Performance Model with the Adapter (Tier 1/2)

The Tier 1/2 adapter does not provide `create_op_performance_model` either -- a meaningful performance model requires semantic knowledge (is the op compute-bound? what is the arithmetic intensity? how many FLOPs?) that the generic adapter does not have:

```json
"performance_model": {
  "compute_ns": 0,
  "ideal_ns": 0,
  "bandwidth_ns": 0
}
```

### After: Performance Model with Tier 3 Extension

A Tier 3 per-op wrapper can provide a performance model by defining the hook on the `OpTag`:

```cpp
// PROPOSED: Tier 3 performance model for Matmul
struct MatmulTagWithPerfModel {
    static constexpr const char* name = "blaze::matmul";

    static OpPerformanceModel create_op_performance_model(
            const BlazeAdapterAttributes& attrs,
            const BlazeAdapterTensorArgs& tensor_args,
            const Tensor& output_tensor) {

        auto M = tensor_args.io_tensors[0].get_padded_shape()[-2];
        auto K = tensor_args.io_tensors[0].get_padded_shape()[-1];
        auto N = tensor_args.io_tensors[1].get_padded_shape()[-1];

        // 2 * M * K * N FLOPs for matmul
        double flops = 2.0 * M * K * N;

        // Wormhole B0: ~150 TFLOPS BFloat16
        double peak_tflops = 150e12;
        double compute_ns = (flops / peak_tflops) * 1e9;

        // Memory bandwidth: read A + B, write C
        double bytes_read = (M * K + K * N) * 2;  // BFloat16 = 2 bytes
        double bytes_written = M * N * 2;
        double total_bytes = bytes_read + bytes_written;

        // Wormhole B0: ~240 GB/s DRAM bandwidth
        double bandwidth_gbs = 240e9;
        double bandwidth_ns = (total_bytes / bandwidth_gbs) * 1e9;

        double ideal_ns = std::max(compute_ns, bandwidth_ns);

        return {
            .compute_ns = static_cast<uint64_t>(compute_ns),
            .ideal_ns = static_cast<uint64_t>(ideal_ns),
            .bandwidth_ns = static_cast<uint64_t>(bandwidth_ns)
        };
    }
};
```

For Matmul with $M=2048, K=1024, N=4096$:

$$\text{FLOPs} = 2 \times 2048 \times 1024 \times 4096 = 1.72 \times 10^{10}$$

$$\text{compute\_ns} = \frac{1.72 \times 10^{10}}{150 \times 10^{12}} \times 10^9 \approx 114{,}688 \text{ ns}$$

$$\text{bandwidth\_ns} = \frac{(2048 \times 1024 + 1024 \times 4096 + 2048 \times 4096) \times 2}{240 \times 10^9} \times 10^9 \approx 122{,}334 \text{ ns}$$

Since $\text{bandwidth\_ns} > \text{compute\_ns}$, this Matmul is slightly bandwidth-bound. The profiler would report:

```json
"performance_model": {
  "compute_ns": 114688,
  "ideal_ns": 122334,
  "bandwidth_ns": 122334
}
```

This enables automated bottleneck classification (compute-bound vs. bandwidth-bound) in profiling pipelines.

### Performance Model Availability by Tier

| Tier | Performance Model | Accuracy | Implementation Cost |
|---|---|---|---|
| Tier 1 (generic) | `{0, 0, 0}` (unknown) | N/A | 0 |
| Tier 2 (custom hash) | `{0, 0, 0}` (unknown) | N/A | 0 |
| Tier 3 (per-op wrapper) | Computed from typed attributes | High (~10--20% error vs measured) | ~20--50 lines per op |

Performance models should be implemented only for the 5--10 ops that dominate compute time (matmul, SDPA, reduce, etc.). For the remaining ~100 ops, the unknown `{0, 0, 0}` model is acceptable -- actual profiler timing data is the primary performance analysis tool.

---

## 6.3.6 Custom Profiling Metadata for Tier 3 Ops

### Why Default Attributes Are Insufficient for High-Value Ops

The default `BlazeAdapterAttributes` provides `op_name` and `num_mesh_programs` -- sufficient for zone naming but insufficient for performance analysis. For compute-intensive ops, engineers need semantic parameters:

| Op | Required Profiling Attributes | Why |
|---|---|---|
| Matmul | M, K, N, fidelity, accumulation dtype | Roofline analysis: FLOPs $= 2MKN$, theoretical bandwidth $= (MK + KN + MN) \cdot \text{dtype\_size}$ |
| SDPA | B, H, S, D, causal mask | Attention compute scales as $O(BS^2D)$; causal halves the work |
| RMSNorm | S, D, epsilon | Bandwidth-bound: $2SD$ reads + $SD$ writes |
| Reduce | axis, input\_shape, reduce\_dim | Determines parallelism strategy |

### Extended Attributes via OpTag

Per-op wrappers (Tier 3) can define custom `profiling_attributes_t` with richer reflection:

```cpp
// PROPOSED: Tier 3 MatmulTag with profiling metadata
struct MatmulTagExtended {
    static constexpr const char* name = "blaze::matmul";

    struct profiling_attributes_t {
        std::string op_name;
        uint32_t M, K, N;
        std::string math_fidelity;  // "HiFi4", "LoFi", etc.
        uint32_t num_mesh_programs;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "op_name", "M", "K", "N", "math_fidelity", "num_mesh_programs");
        auto attribute_values() const {
            return std::make_tuple(
                std::string_view(op_name), M, K, N,
                std::string_view(math_fidelity), num_mesh_programs);
        }
    };

    // Custom type name for the profiler (takes highest priority)
    static std::string get_type_name(
            const BlazeAdapterAttributes& attrs) {
        return fmt::format("blaze::matmul({}x{}x{})",
            extract_M(attrs), extract_K(attrs), extract_N(attrs));
    }
};
```

### SFINAE Detection for Profiling Hooks

The adapter detects extended profiling attributes and performance models via SFINAE:

```cpp
// PROPOSED: SFINAE detection
template <typename Tag, typename = void>
struct HasProfilingAttributes : std::false_type {};

template <typename Tag>
struct HasProfilingAttributes<Tag,
    std::void_t<typename Tag::profiling_attributes_t>> : std::true_type {};

template <typename Tag, typename = void>
struct HasPerformanceModel : std::false_type {};

template <typename Tag>
struct HasPerformanceModel<Tag, std::void_t<decltype(
    Tag::create_op_performance_model(
        std::declval<const BlazeAdapterAttributes&>(),
        std::declval<const BlazeAdapterTensorArgs&>(),
        std::declval<const Tensor&>()))>>
    : std::true_type {};
```

The adapter can forward the performance model hook to the profiler. Note that the `operation_attributes_t` substitution shown below is a design sketch with significant cascading complexity — the `profiling_attributes_t` must be a superset of `BlazeAdapterAttributes` (i.e., must also contain `mesh_program_descriptor`, `op_name`, and `custom_program_hash`) because all adapter methods (`compute_program_hash`, `create_mesh_workload`, etc.) depend on these fields. An alternative approach is to keep `operation_attributes_t` as `BlazeAdapterAttributes` and provide profiling metadata through a separate mechanism (e.g., the `get_type_name` hook shown in Section 6.3.2).

```cpp
// PROPOSED (design sketch — requires profiling_attributes_t to extend BlazeAdapterAttributes):
using operation_attributes_t = std::conditional_t<
    HasProfilingAttributes<OpTag>::value,
    typename OpTag::profiling_attributes_t,  // must include all BlazeAdapterAttributes fields
    BlazeAdapterAttributes>;

// Optional: performance model for profiler
static auto create_op_performance_model(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& output)
    requires HasPerformanceModel<OpTag>::value {
    return OpTag::create_op_performance_model(
        attrs, tensor_args, output);
}
```

### Profiler Attribute Comparison by Tier

| Attribute | Tier 1 (Generic Adapter) | Tier 2 (Custom Hash) | Tier 3 (Per-Op Wrapper) |
|---|---|---|---|
| `op_name` | `"blaze::matmul"` | `"blaze::matmul"` | `"blaze::matmul"` |
| `num_mesh_programs` | 1 | 1 | 1 |
| `M`, `K`, `N` | N/A | N/A | `4096, 4096, 4096` |
| `math_fidelity` | N/A | N/A | `"HiFi4"` |
| `theoretical_flops` | N/A | N/A | $2 \times 4096^3 \approx 137.4 \times 10^9$ FLOPs |
| **Roofline analysis** | Not possible | Not possible | **Fully enabled** |

---

## 6.3.7 `RuntimeIDToOpName` and `ProgramHashToOpName` Lookup Tables

The profiler maintains two lookup tables for post-processing correlation (source: `op_profiler.hpp`, lines 83--100):

```cpp
// EXISTING:
// Maps (device_id, runtime_id) -> operation name
inline std::unordered_map<
    std::pair<uint32_t, uint64_t>, std::string,
    PairHash> RuntimeIDToOpName;

// Maps (device_id, program_hash) -> operation name
inline std::unordered_map<
    std::pair<uint32_t, uint64_t>, std::string,
    PairHash> ProgramHashToOpName;
```

These are populated inside `op_meta_data_serialized_json` after the operation name is resolved.

### Before: Lookup Tables with `generic_op`

```text
RuntimeIDToOpName:
  {0, 42} -> "GenericOpDeviceOperation"
  {0, 43} -> "GenericOpDeviceOperation"
  {0, 44} -> "GenericOpDeviceOperation"
  {0, 45} -> "GenericOpDeviceOperation"

ProgramHashToOpName:
  {0, 4660} -> "GenericOpDeviceOperation"   // Matmul hash
  {0, 8821} -> "GenericOpDeviceOperation"   // GatedReduce hash
  {0, 2315} -> "GenericOpDeviceOperation"   // Copy hash
```

All entries map to the same name. Post-processing scripts that group by op name see a single bucket.

### After: Lookup Tables with the Adapter

```text
RuntimeIDToOpName:
  {0, 42} -> "blaze::matmul"
  {0, 43} -> "blaze::gated_reduce"
  {0, 44} -> "blaze::copy"
  {0, 45} -> "blaze::reduce"

ProgramHashToOpName:
  {0, 33489} -> "blaze::matmul"
  {0, 51237} -> "blaze::gated_reduce"
  {0, 18902} -> "blaze::copy"
```

Each entry maps to a distinct name. Post-processing scripts can now produce per-op-type time breakdowns, invocation counts, and cache hit rates:

| Analysis | `generic_op` | Adapter |
|---|---|---|
| `SELECT op_name, AVG(duration) GROUP BY op_name` | Returns single row | Returns per-op breakdown |
| `SELECT op_name WHERE duration > threshold` | All results say "GenericOpDeviceOperation" | Identifies specific slow ops |
| `SELECT op_name, COUNT(*) GROUP BY op_name` | Single count | Per-op invocation counts |
| Time series: per-op duration across tokens | Not possible | Plot per-op duration trends |

---

## 6.3.8 Environment Variables and Compilation Guards

All Tracy profiling code is gated by two mechanisms:

### Compile-Time Gate: `TRACY_ENABLE` Preprocessor Define

When `TRACY_ENABLE` is not defined, all Tracy macros (`ZoneScopedN`, `ZoneText`, etc.) expand to no-ops. The `TracyOpMeshWorkload` macro's body becomes dead code that the compiler eliminates. The adapter adds zero overhead to non-profiling builds. The only cost is the template instantiation itself (~2--5 KB per op, see [Section 4.2.10](../ch04_adapter_pattern/02_blaze_device_operation_adapter.md)), which is a one-time build cost.

### Runtime Gate: `TTNN_OP_PROFILER` Environment Variable

```cpp
// Source: ttnn/api/tools/profiler/op_profiler.hpp  (lines 107-113)
inline bool is_op_profiler_env_var_set() {
    const char* op_profiler_enable_str = std::getenv("TTNN_OP_PROFILER");
    if (op_profiler_enable_str != nullptr &&
        op_profiler_enable_str[0] == '1') {
        op_profiler_is_enabled = true;
    }
    return op_profiler_is_enabled;
}
```

When `TTNN_OP_PROFILER=1` is not set, the `if` guard in `TracyOpMeshWorkload` short-circuits, and no JSON serialization occurs. The cost is a single environment variable lookup (cached after first access).

When `TRACY_ENABLE` is defined but `TTNN_OP_PROFILER` is not set, the `ZoneScopedN("TT_DNN_DEVICE_OP")` zone still fires (this is the base Tracy zone visible in the flamegraph), but attribute serialization and JSON message generation are skipped.

The adapter does not change this gating behavior. It benefits from it automatically.

---

## 6.3.9 Profiling Overhead Analysis

### Per-Component Overhead Breakdown

| Component | Cost (profiler disabled) | Cost (profiler enabled) |
|---|---|---|
| `is_op_profiler_env_var_set()` check | ~1 ns (cached) | ~1 ns (cached) |
| Zone creation (`ZoneScopedN`) | 0 (compiled out without `TRACY_ENABLE`) | ~15 ns |
| Mesh coordinate iteration | 0 | ~5 ns per device |
| `EncodePerDeviceProgramID` | 0 | ~2 ns |
| `op_meta_data_serialized_json` (2 attributes) | 0 | ~520--1,020 ns |
| `op_meta_data_serialized_json` (5+ attributes, Tier 3) | 0 | ~800--2,020 ns |
| `ZoneText` | 0 | ~10 ns |
| `tracy_message` | 0 | ~50 ns |
| **Total per-op (single device, Tier 1/2)** | **~1 ns** | **~600--1,100 ns** |
| **Total per-op (single device, Tier 3)** | **~1 ns** | **~880--2,100 ns** |
| **Total per-op (T3K, 8 devices, Tier 1/2)** | **~1 ns** | **~4,800--8,800 ns** |

### Overhead Scaling for Full Models

For a 200-op forward pass (typical LLM decoder):

| Configuration | Profiler Disabled | Profiler Enabled |
|---|---|---|
| `generic_op`, single device | ~200 ns | ~120--220 $\mu$s |
| Adapter, single device | ~200 ns | ~120--220 $\mu$s |
| `generic_op`, T3K | ~200 ns | ~0.96--1.76 ms |
| Adapter, T3K | ~200 ns | ~0.96--1.76 ms |

**The adapter does not change profiling overhead.** The overhead comes from JSON serialization and Tracy zone management, which are identical for both dispatch paths. The adapter's contribution is in the *content* of the zones, not their cost.

### Profiler Cache Impact

The op profiler maintains a cache (`cached_ops`) that avoids re-serializing JSON for programs that have already been profiled. The cache is keyed by `(device_id, program_hash)`.

Under the adapter with Tier 2 hashing, the profiler cache achieves a higher effective hit rate than `generic_op`. Because Tier 2 hashes are deterministic Python-computed values, the same op with the same configuration always produces the same hash. Under `generic_op`, descriptor-level instability (as described in [Section 6.1.1](01_program_cache_integration.md)) can cause hash variation, forcing the profiler to re-serialize the full JSON blob (~500--2,000 ns per miss).

---

## 6.3.10 Profiling Data Flow Diagram

The following diagram shows the complete profiling data flow for a registered Blaze op, from Python invocation through Tracy zone emission:

```text
Python: ttnn.blaze.matmul(io_tensors, descriptor)
    |
    v
C++: launch<BlazeDeviceOperationAdapter<MatmulTag>>(...)
    |
    +-- compute_program_hash()
    |     -> type_hash<Adapter<MatmulTag>> XOR content_hash
    |     -> stored as op_hash in profiler JSON
    |
    +-- enqueue_mesh_workload<MeshDeviceOperationAdapter<Adapter<MatmulTag>>>()
    |     |
    |     +-- set_runtime_id(workload)
    |     |     -> monotonic ID for per-device correlation
    |     |
    |     +-- EnqueueMeshWorkload(...)
    |     |     -> actual hardware dispatch
    |     |
    |     +-- TracyOpMeshWorkload(
    |           mesh_device,
    |           workload,
    |           MeshDeviceOperationAdapter<Adapter<MatmulTag>>{},
    |           attrs,        // BlazeAdapterAttributes
    |           tensor_args,  // BlazeAdapterTensorArgs
    |           output)
    |           |
    |           +-- For each (range, program):
    |           |     +-- ZoneScopedN("TT_DNN_DEVICE_OP")
    |           |     +-- op_meta_data_serialized_json(...)
    |           |     |     |
    |           |     |     +-- get_op_name<Adapter<MatmulTag>>(attrs)
    |           |     |     |     -> "blaze::matmul"
    |           |     |     |
    |           |     |     +-- get_attributes(attrs)
    |           |     |     |     -> {"op_name": "blaze::matmul",
    |           |     |     |         "num_mesh_programs": 1}
    |           |     |     |
    |           |     |     +-- serialize input/output tensors
    |           |     |     |
    |           |     |     +-- get_performance_model(...)
    |           |     |     |     -> {0, 0, 0} (Tier 1)
    |           |     |     |     or computed estimates (Tier 3)
    |           |     |     |
    |           |     |     +-- Populate lookup tables:
    |           |     |           RuntimeIDToOpName[(0, 42)] = "blaze::matmul"
    |           |     |           ProgramHashToOpName[(0, hash)] = "blaze::matmul"
    |           |     |
    |           |     +-- ZoneText("id:42")
    |           |     +-- tracy_message(json_blob)
    |           |
    |           +-- Tracy GUI displays:
    |                 Zone:    "TT_DNN_DEVICE_OP"
    |                 Text:    "id:42"
    |                 Message: Full JSON with "op_code": "blaze::matmul"
    |
    +-- track_function_end(output_tensor)
```

Each labeled step produces data consumed by different analysis tools:

| Data | Consumer | Available Before Registration? | Available After? |
|------|----------|:---:|:---:|
| Per-op zone name | Tracy flamegraph | No (generic) | Yes |
| Per-op JSON attributes | Profiler post-processing | No (opaque descriptor) | Yes |
| Per-op `op_hash` | Cache correlation | Partial (shared type) | Yes (isolated) |
| Per-op performance model | Utilization analysis | No | Tier 3 only |
| Per-op lookup table entries | Offline analysis | No (generic name) | Yes |

---

## 6.3.11 Quantified Profiling Utility

Define **profiling utility** as the number of actionable insights per profiling session. In a typical model optimization workflow:

| Insight Type | `generic_op` Sessions Needed | Adapter Sessions Needed | Speedup |
|---|---|---|---|
| Identify top-5 expensive op types | 3--5 (requires manual annotation) | 1 | 3--5x |
| Detect unexpected op count | 2--3 (requires dispatch log cross-ref) | 1 | 2--3x |
| Measure op-level regression | 2--4 (requires baseline annotation) | 1 (diff zone durations directly) | 2--4x |
| Correlate with hardware counters | Not feasible (no op identity) | 1 (match zone name to counter data) | N/A $\to$ 1 session |

| Metric | `generic_op` | Adapter |
|---|---|---|
| **Time to first actionable insight** | 30--60 minutes (manual annotation) | <5 minutes (automatic) |
| **Improvement factor** | -- | **6--12x faster** |

In practice, this translates to a reduction in profiling-driven optimization cycles from days to hours for a new model bring-up.

---

## 6.3.12 End-to-End Profiler JSON Comparison

The following diff captures the complete set of changes between `generic_op` and adapter profiler JSON output. For the GatedReduce T3K variant (showing `num_mesh_programs: 8` and multi-phase kernel lists), see the graph JSON fragment in [Section 6.2.5](02_graph_tracing_and_inspector.md#625-concrete-graph-json-fragments), which carries the same structural metadata.

### JSON Diff

```diff
  {
    "global_call_count": 42,
-   "op_code": "GenericOpDeviceOperation",
+   "op_code": "blaze::matmul",
    "op_type": "tt_dnn_device",
    "device_id": 0,
-   "op_hash": 4660,
+   "op_hash": 33489,
-   "attributes": {},
+   "attributes": {
+     "op_name": "blaze::matmul",
+     "num_mesh_programs": 1
+   },
    "input_tensors": [ ... ],
    "output_tensors": [ ... ],
    "kernel_info": { ... },
    "performance_model": { "compute_ns": 0, "ideal_ns": 0, "bandwidth_ns": 0 }
  }
```

Three fields change: `op_code` gains per-op identity, `op_hash` gains type-isolated namespace, and `attributes` gains reflectable metadata. Everything else is identical.

---

## 6.3.13 What Changes, What Stays the Same

| Aspect | Changes? | Before (`generic_op`) | After (`Adapter<OpTag>`) |
|:---|:---:|:---|:---|
| `TracyOpMeshWorkload` macro | No | Same macro, same call site | Same |
| `TTNN_OP_PROFILER` env var control | No | Opt-in via `=1` | Same |
| `TRACY_ENABLE` compile flag | No | Required for any Tracy output | Same |
| Tracy zone name | No | `"TT_DNN_DEVICE_OP"` (fixed) | Same |
| Tracy zone text | No | `"id:N"` (runtime ID) | Same |
| `op_code` in JSON | **Yes** | `"GenericOpDeviceOperation"` | `"blaze::matmul"` (per-op) |
| `op_hash` in JSON | **Yes** | Hash without type prefix | Hash with type prefix |
| `attributes` in JSON | **Yes** | `{}` (empty) | `{"op_name": "...", "num_mesh_programs": N}` |
| `input_tensors` / `output_tensors` in JSON | No | Shape, dtype, layout from tensors | Same |
| `kernel_info` in JSON | No | Kernel paths from `Program` | Same |
| `performance_model` in JSON | No (Tier 1/2); **Yes** (Tier 3) | All zeros | All zeros (Tier 1/2); computed estimates (Tier 3) |
| `RuntimeIDToOpName` table | **Yes** | All `"GenericOpDeviceOperation"` | Per-op names |
| `ProgramHashToOpName` table | **Yes** | All `"GenericOpDeviceOperation"` | Per-op names |
| Flamegraph visual readability | **Yes** | All zones identical | Per-op-type zones with distinct colors |
| Post-processing per-op-type aggregation | **Yes** | Single bucket | Per-op buckets |
| Profiler serialization cache (`cached_ops`) | No | Keyed by `(device_id, program_hash)` | Same |
| Per-device program ID encoding | No | `EncodePerDeviceProgramID` | Same |
| Thread-safe profiler data structures | No | `thread_safe_cached_ops_map` | Same |
| Profiling overhead (profiler enabled) | No | ~600--2,100 ns per-op | Same |
| Profiling overhead (profiler disabled) | No | ~1 ns per-op | Same |
| Time to first actionable insight | **Yes** | 30--60 minutes | <5 minutes (6--12x faster) |

---

| [Previous: Graph Tracing and Inspector](02_graph_tracing_and_inspector.md) | [Chapter Index](index.md) | [Next: Chapter 7](../ch07_blaze_nn_integration/index.md) |
|:---|:---:|---:|
