# 6.2 Graph Tracing and Inspector

TTNN's `GraphTracker` singleton records a directed acyclic graph (DAG) of operations and tensor data flows. Each node carries the operation's C++ type name, its input tensors, and its output tensors. This graph serves two purposes: (1) shape propagation in `NO_DISPATCH` mode (operations are traced but not executed, enabling output shape computation without hardware), and (2) graph-level analysis for optimization, memory scheduling, and debugging. The Inspector is a complementary facility that annotates each `MeshWorkload` with operation metadata for external tooling. Both systems derive their operation names from the same source: `get_operation_name<device_operation_t>()`. This section traces exactly what each system sees before and after registration.

---

## 6.2.1 `GraphTracker::track_function_start/end` Mechanics

The `launch<>` function brackets every device operation with graph tracking calls (source: `device_operation.hpp`, lines 420--421, 489):

```cpp
// EXISTING: inside launch<device_operation_t>(...)
tt::tt_metal::GraphTracker::instance().track_function_start(
    detail::get_operation_name<device_operation_t>(operation_attributes),
    operation_attributes, input_tensors);
// ... dispatch (cache lookup, create/override, enqueue) ...
tt::tt_metal::GraphTracker::instance().track_function_end(tensor_return_value);
```

The operation name is resolved by `get_operation_name<>()` (see [Section 1.3](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md) for the full implementation), which recurses through `MeshDeviceOperationAdapter` wrapping to extract the C++ type name of the innermost `device_operation_t`.

### Before: What `GraphTracker` Sees with `generic_op`

When Blaze dispatches Matmul through `generic_op`:

```text
track_function_start(
    name = "GenericOpDeviceOperation",    <-- type name of the device op
    attrs = MeshProgramDescriptor{...},   <-- opaque descriptor
    inputs = [tensor_A, tensor_B, tensor_out]
)
```

When Blaze dispatches GatedReduce through `generic_op`:

```text
track_function_start(
    name = "GenericOpDeviceOperation",    <-- SAME type name
    attrs = MeshProgramDescriptor{...},   <-- different descriptor, same type
    inputs = [tensor_gate, tensor_input, tensor_out]
)
```

Both operations appear as `"GenericOpDeviceOperation"` in the graph. The graph DAG looks like:

```text
BEFORE: Graph DAG for a simple Matmul -> GatedReduce pipeline

  [tensor_A] ----+
                  |
  [tensor_B] ----+--> [ GenericOpDeviceOperation ] --> [tensor_matmul_out]
                                                              |
  [tensor_gate] --+                                           |
                  +--> [ GenericOpDeviceOperation ] --> [tensor_reduce_out]
```

Every node has the same label. An engineer examining this graph cannot distinguish Matmul from GatedReduce without inspecting the raw `MeshProgramDescriptor` attributes -- which are opaque binary blobs in the graph JSON.

### After: What `GraphTracker` Sees with the Adapter

When Blaze dispatches Matmul through `BlazeDeviceOperationAdapter<MatmulTag>`:

```text
track_function_start(
    name = "BlazeDeviceOperationAdapter<MatmulTag>",
    attrs = BlazeAdapterAttributes{
        .op_name = "blaze::matmul",
        .mesh_program_descriptor = {...},
        .custom_program_hash = nullopt
    },
    inputs = [tensor_A, tensor_B, tensor_out]
)
```

The graph DAG now has distinct labels:

```text
AFTER: Graph DAG for a simple Matmul -> GatedReduce pipeline

  [tensor_A] ----+
                  |
  [tensor_B] ----+--> [ BlazeDeviceOperationAdapter<MatmulTag> ] --> [tensor_matmul_out]
                                                                           |
  [tensor_gate] --+                                                        |
                  +--> [ BlazeDeviceOperationAdapter<GatedReduceTag> ] --> [tensor_reduce_out]
```

### `get_operation_name<>()` Resolution Path

```text
BEFORE (generic_op):
  get_operation_name<MeshDeviceOperationAdapter<GenericOpDeviceOperation>>(attrs)
    |
    +-- is_mesh_device_operation_adapter_v? YES
    |
    +-- recurse: get_operation_name<GenericOpDeviceOperation>(attrs)
         |
         +-- is_mesh_device_operation_adapter_v? NO
         |
         +-- return get_type_name<GenericOpDeviceOperation>()
              = "GenericOpDeviceOperation"


AFTER (Adapter<MatmulTag>):
  get_operation_name<MeshDeviceOperationAdapter<BlazeDeviceOperationAdapter<MatmulTag>>>(attrs)
    |
    +-- is_mesh_device_operation_adapter_v? YES
    |
    +-- recurse: get_operation_name<BlazeDeviceOperationAdapter<MatmulTag>>(attrs)
         |
         +-- is_mesh_device_operation_adapter_v? NO
         |
         +-- return get_type_name<BlazeDeviceOperationAdapter<MatmulTag>>()
              = "BlazeDeviceOperationAdapter<MatmulTag>"
```

The name resolution is compile-time (the `if constexpr` branches are resolved at template instantiation). There is no runtime cost difference.

### Side-by-Side: `track_function_start` Arguments

| Argument | Before (`generic_op`) | After (`Adapter<MatmulTag>`) |
|:---|:---|:---|
| `function_name` | `"GenericOpDeviceOperation"` | `"BlazeDeviceOperationAdapter<MatmulTag>"` |
| `operation_attributes` type | `MeshProgramDescriptor` | `BlazeAdapterAttributes` |
| `operation_attributes` reflectable fields | None (opaque descriptor) | `op_name`, `num_mesh_programs` |
| `input_tensors` | Same tensor references | Same tensor references |

---

## 6.2.2 `NO_DISPATCH` Mode Handling

`NO_DISPATCH` mode is used for graph capture without hardware execution. `ProcessorHooks::set_block(true)` causes all allocation and program hooks to return `true`, meaning "I handled this, skip real execution." This mode is critical for shape propagation, memory estimation, and graph optimization.

### The `compute_output_specs` Role

In `NO_DISPATCH` mode, `launch<>` calls `compute_output_specs` to determine output tensor shapes without actually running the program:

```cpp
// EXISTING: inside launch<device_operation_t>(...), NO_DISPATCH path
auto output_specs = device_operation_t::compute_output_specs(
    operation_attributes, tensor_args);
// Use output_specs to create placeholder tensors for graph tracking
```

### Before: `GenericOpDeviceOperation::compute_output_specs`

```cpp
// EXISTING: generic_op_device_operation.cpp
TensorSpec GenericOpDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}
```

This reads the spec from the pre-allocated output tensor. In `NO_DISPATCH` mode, the output tensor must already exist with the correct spec -- Blaze pre-allocates it. This works, but it means the graph capture caller must provide pre-allocated outputs even when no hardware execution occurs.

### After: `BlazeDeviceOperationAdapter<OpTag>::compute_output_specs`

```cpp
// PROPOSED: BlazeDeviceOperationAdapter<OpTag>::compute_output_specs
static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}
```

The implementation is identical. The adapter preserves Blaze's pre-allocation model (Challenge 3 from [Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)) -- the output tensor is always pre-allocated by the Blaze compiler before reaching C++. The adapter does not attempt to infer output shapes from input shapes because that would require embedding Blaze's shape logic in C++.

### `NO_DISPATCH` Behavior Summary

In `NO_DISPATCH` mode, the adapter follows the same seven-step path as normal dispatch (track_function_start, compute_output_specs, compute_program_hash, cache lookup, create_mesh_workload, track_workload, track_function_end), with the operation name changed per [Section 6.2.1](#621-graphtrackertrack_function_startend-mechanics) and the hash prefixed per [Section 6.1.1](01_program_cache_integration.md#611-hash-computation-before-and-after). The key difference is that cache insertion is blocked (see below) and enqueue is skipped. The graph node recorded carries the adapter's per-op name (`"BlazeDeviceOperationAdapter<MatmulTag>"`) instead of the generic name.

### `NO_DISPATCH` Cache Interaction Detail

A subtle invariant deserves emphasis: during `NO_DISPATCH` capture, buffer addresses in tensors are `0` (no real allocation occurred). If the program were cached with these invalid addresses, a subsequent real dispatch could retrieve the corrupted entry. TTNN prevents this by checking `ProcessorHooks::get_block()` before cache insertion (see [Section 1.3](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md) for the `hook_blocks` guard code). When `get_block()` returns `true`, the `should_cache` flag is set to `false`, preventing contamination. This guard applies identically to `generic_op` and the adapter -- the adapter does not need any `NO_DISPATCH`-specific logic.

---

## 6.2.3 Inspector Integration

The Inspector annotates each `MeshWorkload` with operation metadata for external debugging and analysis tools. The annotation is emitted on cache miss via `emit_mesh_workload_annotation` (see [Section 1.3](../ch01_ttnn_registration_contract/03_program_cache_graph_tracing_and_profiling.md) for the full implementation), which calls `get_operation_name<>()` to extract the type name and `visit_object_of_type<Tensor>` to collect tensor references.

### Inspector Side-by-Side

| Inspector Field | Before (`generic_op`) | After (`Adapter<MatmulTag>`) |
|:---|:---|:---|
| `operation_name` | `"GenericOpDeviceOperation"` | `"BlazeDeviceOperationAdapter<MatmulTag>"` |
| Reflectable attributes | None (opaque `MeshProgramDescriptor`) | `op_name = "blaze::matmul"`, `num_mesh_programs = 1` |
| Tensor metadata | Shape, dtype, layout from `io_tensors` | Same (identical tensors) |
| Runtime ID | Monotonic counter (same mechanism) | Same (identical mechanism) |

---

## 6.2.4 Graph-Level Optimization Implications

Semantic node names unlock several graph-level analyses that are impossible with `generic_op`:

### 1. Op-Type Frequency Analysis

```text
BEFORE: op frequency from graph JSON
  GenericOpDeviceOperation: 47 nodes

AFTER: op frequency from graph JSON
  BlazeDeviceOperationAdapter<MatmulTag>:       12 nodes
  BlazeDeviceOperationAdapter<GatedReduceTag>:   8 nodes
  BlazeDeviceOperationAdapter<CopyTag>:         15 nodes
  BlazeDeviceOperationAdapter<ReduceTag>:        6 nodes
  BlazeDeviceOperationAdapter<McastTag>:         6 nodes
```

### 2. Data-Dependency Pattern Detection

Named nodes allow pattern matching on the graph structure. For example, detecting a "Matmul -> GatedReduce" pattern for potential fusion:

```text
BEFORE: pattern detection
  GenericOpDeviceOperation -> GenericOpDeviceOperation
  // Cannot distinguish: is this Matmul->GatedReduce or Copy->Reduce?

AFTER: pattern detection
  BlazeDeviceOperationAdapter<MatmulTag> -> BlazeDeviceOperationAdapter<GatedReduceTag>
  // Unambiguous: Matmul feeds into GatedReduce
```

In a 32-layer decoder model, RMSNorm-Matmul pairs appear 64 times (2 per layer). If a future graph optimization pass were implemented to fuse such pairs, each fusion would save one dispatch round-trip (~10 $\mu$s) and one intermediate tensor allocation (~1--4 $\mu$s), for an estimated total savings of ~700--900 $\mu$s per forward pass. Registration provides the named nodes that make this pattern detection possible; the optimization pass itself is future work.

### 3. Memory Scheduling

Graph-level memory schedulers need to know operation types to estimate compute duration and overlap data movement with computation. With `generic_op`, every node has the same estimated cost. With the adapter, a scheduler can assign op-type-specific cost estimates.

### 4. Subgraph Extraction

Named nodes enable extracting semantically meaningful subgraphs:

```python
# Extract all attention-related ops
attention_subgraph = graph.subgraph(
    lambda n: n.name in [
        "blaze::sdpa", "blaze::rotary_embedding",
        "blaze::matmul"  # Q/K/V projections
    ])
```

### 5. Regression Detection

Graph diff under `generic_op` shows "480 GenericOpDeviceOperation" in both versions -- any structural change requires manual inspection. With named nodes, a graph diff can show per-op changes: "MatmulTag count changed from 64 to 96."

### Limitations

The adapter's graph integration has two limitations compared to native TTNN ops:

1. **No true shape inference.** `compute_output_specs` returns the pre-allocated output tensor's spec, not a computed spec from input shapes. A graph-level shape propagation pass cannot derive output shapes from input shapes alone -- it needs the pre-allocated tensor.

2. **No attribute-level graph parameters.** Native TTNN ops expose semantic attributes (e.g., `binary_op_type: ADD`, `math_fidelity: HiFi4`) as graph node parameters. The adapter exposes `op_name` and `num_mesh_programs`, which are less detailed. Tier 3 per-op wrappers can add richer attributes.

---

## 6.2.5 Concrete Graph JSON Fragments

The following JSON fragments illustrate the expected graph node structure, assuming `GraphProcessor::track_function_start` serializes reflectable attributes from `operation_attributes_t` into the `params` map (the "Before" structure is verified; the "After" params content depends on the graph processor's reflection behavior with `BlazeAdapterAttributes`).

### Matmul Node (Before)

```json
{
  "counter": 42,
  "node_type": "function_start",
  "params": {
    "name": "GenericOpDeviceOperation"
  },
  "connections": [39, 40, 41],
  "input_tensors": [39, 40, 41]
}
```

### Matmul Node (After)

```json
{
  "counter": 42,
  "node_type": "function_start",
  "params": {
    "name": "BlazeDeviceOperationAdapter<MatmulTag>",
    "op_name": "blaze::matmul",
    "num_mesh_programs": 1
  },
  "connections": [39, 40, 41],
  "input_tensors": [39, 40, 41]
}
```

### GatedReduce Node (After, on T3K)

```json
{
  "counter": 45,
  "node_type": "function_start",
  "params": {
    "name": "BlazeDeviceOperationAdapter<GatedReduceTag>",
    "op_name": "blaze::gated_reduce",
    "num_mesh_programs": 8
  },
  "connections": [43, 44],
  "input_tensors": [43, 44]
}
```

The `num_mesh_programs: 8` for GatedReduce on T3K reflects one `ProgramDescriptor` per device in the 8-device mesh -- a useful diagnostic that was previously invisible.

---

## 6.2.6 Quantitative Impact on Graph Analysis

| Analysis Task | `generic_op` (Before) | Adapter (After) | Improvement |
|---|---|---|---|
| **Op count by type** | `count("GenericOpDeviceOperation") = 480` | `count("MatmulTag") = 64, count("RMSNormTag") = 32, ...` | Per-type counts available |
| **Fusion candidate detection** | Manual only (all nodes look identical) | Automated pattern matching on named nodes | New capability |
| **Redundant op detection** | Cannot distinguish redundant copies from necessary ones | Identical ops with identical inputs flagged automatically | New capability |
| **Regression diff quality** | Structural only (all names identical) | Semantic: "MatmulTag count changed from 64 to 96" | Qualitative improvement |
| **Memory estimation accuracy** | Shape-heuristic only (op type unknown) | Op-aware models possible (matmul output = $M \times N \times \text{dtype\_size}$) | New capability |
| **Per-op trace overhead** | ~301 $\mu$s (NO_DISPATCH, dominated by `create_mesh_workload`) | ~301 $\mu$s (identical) | No change |
| **Graph JSON size (70B model)** | ~1.3 MB | ~1.3 MB (+/- 50 KB for longer names) | Negligible |

The adapter's contribution to graph tracing is not performance (the overhead is unchanged) but **information quality**: every graph node carries the semantic identity needed for automated analysis.

---

## 6.2.7 What Changes, What Stays the Same

| Aspect | Changes? | Before (`generic_op`) | After (`Adapter<OpTag>`) |
|:---|:---:|:---|:---|
| `GraphTracker` singleton | No | Same instance | Same instance |
| `track_function_start` call site | No | Inside `launch<>` | Same location inside `launch<>` |
| `track_function_end` call site | No | Inside `launch<>` | Same location inside `launch<>` |
| Operation name in graph node | **Yes** | `"GenericOpDeviceOperation"` | `"BlazeDeviceOperationAdapter<MatmulTag>"` |
| Graph node parameters (reflectable) | **Yes** | None (opaque descriptor) | `op_name`, `num_mesh_programs` |
| `NO_DISPATCH` mode behavior | No | Skip enqueue, record graph node | Same |
| `NO_DISPATCH` cache interaction | No | `ProcessorHooks::get_block()` prevents caching | Same |
| `compute_output_specs` logic | No | Returns pre-allocated tensor spec | Same |
| `compute_output_specs` source of truth | No | Pre-allocated output tensor | Same |
| Inspector annotation name | **Yes** | `"GenericOpDeviceOperation"` | `"BlazeDeviceOperationAdapter<MatmulTag>"` |
| Inspector reflectable attributes | **Yes** | None | `op_name`, `num_mesh_programs` |
| `ProcessorHooks` / `IGraphHooks` | No | Same interface | Same interface |
| `ScopedGraphCapture` usage | No | Same RAII pattern | Same RAII pattern |
| Workload tracking (`track_workload`) | No | Same per-program iteration | Same |
| Graph-level pattern detection | **Yes** | Impossible (all nodes identical) | Enabled (named nodes) |
| True output shape inference | No | Not available (pre-allocation model) | Not available (same model) |

---

| [Previous: Program Cache Integration](01_program_cache_integration.md) | [Chapter Index](index.md) | [Next: Tracy Profiling](03_tracy_profiling.md) |
|:---|:---:|---:|
