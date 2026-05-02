# 02 -- Device Operation Anatomy

## Context

The previous file cataloged the nine CCL operations and their public API signatures. This file takes one operation -- all-gather -- and traces its complete execution path from the `invoke` call at the TTNN API layer down through the device operation framework, mesh workload creation, per-device program factory, global semaphore allocation, worker core selection, and runtime argument override. The all-gather is the canonical CCL operation in TT-Metal: it is the most mature, the most optimized, and its internal architecture is representative of how all fabric-based CCL operations are structured. The reader should understand the structural pattern that all CCL device operations follow, enabling them to evaluate what a "transparent" alternative would need to replace at each layer.

---

## The All-Gather Execution Pipeline

The all-gather operation traverses five distinct layers before device kernels run:

```
 User Code
    |
    v
 ExecuteAllGather::invoke()        [API entry point]
    |
    v
 ttnn::prim::all_gather()          [Primitive dispatch]
    |
    v
 AllGatherDeviceOperation          [Device operation framework]
    |   - validate
    |   - compute_output_specs
    |   - create_output_tensors
    |   - compute_program_hash
    |
    v
 AllGatherProgram::create_mesh_workload()   [Mesh-level program creation]
    |
    v
 AllGatherProgram::create_at()     [Per-device program creation]
    |
    v
 Device kernels run on Tensix cores + Fabric routers
```

---

## Layer 1: `ExecuteAllGather::invoke`

The entry point (`all_gather.cpp`) performs three tasks before delegating to the primitive:

### 1a. Multi-axis decomposition

When `cluster_axis` is `std::nullopt` and the mesh is not a line topology (i.e., mesh shape is not $1 \times M$ or $M \times 1$), invoke recursively calls itself for each mesh dimension in reverse order:

```cpp
// ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp
if (cluster_axis == std::nullopt) {
    auto mesh_shape = input_tensor.device()->get_view().shape();
    if (!mesh_shape.is_line_topology()) {
        Tensor tensor = input_tensor;
        auto mesh_view = mesh_shape.view();
        for (auto it = mesh_view.rbegin(); it != mesh_view.rend(); ++it) {
            auto axis = std::distance(mesh_view.begin(), it.base()) - 1;
            tensor = ttnn::all_gather(tensor, dim, axis, ...);
        }
        return tensor;
    }
}
```

This means a 2D mesh all-gather decomposes into two 1D all-gathers -- first along the last axis, then along the first. This is a key design decision: the CCL system reduces 2D problems to sequences of 1D problems rather than implementing native 2D collectives.

### 1b. Topology resolution

```cpp
tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);
```

The 2D-to-1D conversion maps `Mesh` to `Linear` and `Torus` to `Ring`. The usable topology check applies boundary mode logic: if the device coordinates along the cluster axis span from 0 to $\text{mesh shape}[\text{axis}] - 1$ and the shape is at least 3, `WRAP` mode enables ring/torus; otherwise, it falls back to `NONE` (linear). See File 04 for the full `get_boundary_mode()` logic.

### 1c. Composite operation check

For certain tensor configurations (e.g., specific sharding patterns), a composite implementation is used instead:

```cpp
if (composite_common::use_composite_all_gather(input_tensor, dim, memory_config)) {
    return composite_common::composite_all_gather(...);
}
```

Otherwise, dispatch proceeds to the primitive.

---

## Layer 2: `ttnn::prim::all_gather`

The primitive constructs the `operation_attributes_t` and `tensor_args_t` structs, then calls `ttnn::device_operation::launch<AllGatherDeviceOperation>`:

```cpp
// ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp
return ttnn::device_operation::launch<OperationType>(
    OperationType::operation_attributes_t{
        .memory_config = memory_config,
        .dim = dim,
        .cluster_axis = cluster_axis,
        .subdevice_id = subdevice_id,
        .topology = topology,
        .num_links = num_links,
        .chunks_per_sync = chunks_per_sync,
        .num_workers_per_link = num_workers_per_link,
        .num_buffers_per_channel = num_buffers_per_channel,
        .sub_core_grid = sub_core_grid},
    OperationType::tensor_args_t{
        .input_tensor = input_tensor,
        .optional_output_tensor = optional_output_tensor});
```

The `device_operation::launch` framework handles program cache lookup (using `compute_program_hash`), validation (cache miss: full validation; cache hit: lightweight validation), output tensor creation, and mesh workload creation or runtime argument override.

---

## Layer 3: The `AllGatherDeviceOperation` Framework

### 3a. `operation_attributes_t`

This struct captures all configuration that is not tensor data:

```cpp
struct operation_attributes_t {
    const MemoryConfig memory_config;
    uint32_t dim;
    const std::optional<uint32_t> cluster_axis;
    const std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    const tt::tt_fabric::Topology topology;
    const uint32_t num_links;
    const std::optional<uint32_t> chunks_per_sync;
    const std::optional<uint32_t> num_workers_per_link;
    const std::optional<uint32_t> num_buffers_per_channel;
    const std::optional<CoreRangeSet> sub_core_grid;
};
```

### 3b. `tensor_args_t`

```cpp
struct tensor_args_t {
    const Tensor input_tensor;
    std::optional<Tensor> optional_output_tensor;
};
```

### 3c. Framework Methods

The device operation framework requires these static methods:

| Method | Purpose |
|--------|---------|
| `validate_on_program_cache_miss` | Full validation when creating a new program (checks storage type, shape rank, memory layout, page alignment, num_links constraints) |
| `validate_on_program_cache_hit` | Lightweight validation on cached program reuse (checks output tensor compatibility) |
| `compute_output_specs` | Computes the output tensor specification (shape along `dim` is multiplied by ring size) |
| `compute_output_topologies` | Computes the output tensor distribution topology (sharded dims become replicated after gather) |
| `create_output_tensors` | Allocates the output tensor (or validates the provided optional output) |
| `compute_program_hash` | Produces a hash key for program caching based on attributes, subdevice cores, and tensor args |

### 3d. Validation Details

On cache miss, `validate_on_program_cache_miss` checks:

1. Input tensor is on device and allocated in a buffer
2. Tensor rank is at least 2
3. Ring size (topological dimension) is greater than 1
4. `num_links > 0` and does not exceed the compute grid's Y dimension
5. Page size is aligned to buffer alignment
6. Memory layout is one of `{INTERLEAVED, WIDTH_SHARDED, BLOCK_SHARDED, HEIGHT_SHARDED}`
7. Block-sharded tensors must be in L1 (not DRAM)

> **Key Insight:** The constraint that worker cores for links are parallelized over rows of the compute grid reveals a fundamental architectural choice -- each link gets its own row of Tensix cores, limiting the maximum number of concurrent links to the grid height.

### 3e. Output Spec Computation

The output tensor's shape at the gather dimension is multiplied by the ring size:

$$\text{output shape}[\text{dim}] = \text{input shape}[\text{dim}] \times \text{ring size}$$

where `ring_size = get_topological_dimension(input_tensor, cluster_axis)`.

### 3f. Output Topology Computation

`compute_output_topologies` updates the tensor distribution placements: if the input tensor was `Shard`ed along the gather dimension, the output becomes `Replicate`d along that dimension (since all-gather makes every device hold the full data).

---

## Layer 4: `create_mesh_workload` -- Mesh-Level Program Factory

This is where multi-device coordination happens.

### 4a. Creates global semaphores

```cpp
std::vector<tt::tt_metal::GlobalSemaphore> multidevice_semaphores = {
    ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
    ttnn::global_semaphore::create_global_semaphore(mesh_device, subdevice_core_range_set, 0),
};
auto barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(
    mesh_device, subdevice_core_range_set, 0);
```

Three `GlobalSemaphore` objects are created:
1. **Two multi-device semaphores** for forward and backward link synchronization within the operation
2. **One barrier semaphore** to ensure all output buffers are allocated across all devices before data movement begins

### 4b. Synchronizes the mesh

```cpp
tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevice_ids);
```

This is a host-side barrier ensuring all devices have completed their allocations. Without it, a fast device could begin sending data to a slow device that has not yet allocated its receive buffer.

### 4c. Creates per-device programs

```cpp
for (const auto& coord : tensor_coords.coords()) {
    auto cached_program = create_at(
        operation_attributes, coord, tensor_args, tensor_return_value,
        tensor_coords, multidevice_semaphores, barrier_semaphore);
    workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
    shared_variables.emplace(coord, std::move(cached_program.shared_variables));
}
```

The factory iterates over every mesh coordinate that holds the tensor and calls `create_at` to build a per-device program. The resulting programs and shared variables are collected into a `MeshWorkload` for coordinated dispatch.

---

## Layer 5: `create_at` -- Per-Device Program Creation

The `create_at` method is where the bulk of program construction happens for a single device.

### 5a. Neighbor Discovery

```cpp
const std::optional<MeshCoordinate> forward_coordinate =
    ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1,
        operation_attributes.topology, operation_attributes.cluster_axis);
const std::optional<MeshCoordinate> backward_coordinate =
    ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, -1,
        operation_attributes.topology, operation_attributes.cluster_axis);
```

For a `Linear` topology, endpoints have `std::nullopt` for the neighbor beyond the end. For a `Ring` topology, the last device wraps to the first.

### 5b. Device Index Linearization

```cpp
uint32_t device_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
    tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);
```

The 2D mesh coordinate is linearized to a 1D index within the cluster axis. This index determines which slice of the output tensor this device is responsible for and how many hops it is from other devices.

### 5c. Worker Core Selection

The `choose_worker_cores` function selects which Tensix cores will execute the CCL worker kernels:

```cpp
std::tuple<CoreRangeSet, std::vector<CoreCoord>> choose_worker_cores(
    size_t num_links,
    size_t num_workers_per_link,
    IDevice* device,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord core_grid_offset = CoreCoord(0, 0),
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    CoreAllocationStrategy strategy = CoreAllocationStrategy::ROW_MAJOR);
```

Two allocation strategies are supported:

| Strategy | Behavior |
|----------|----------|
| `ROW_MAJOR` | Assigns worker cores row-by-row across the compute grid. Each link gets a contiguous set of cores. |
| `COL_MAJOR` | Assigns worker cores column-by-column. Useful when row-based allocation conflicts with other operations. |

The total number of worker cores allocated is $\text{num links} \times \text{num workers per link}$.

> **Key Insight:** Worker cores for CCL operations are *not* Ethernet cores -- they are standard Tensix compute cores temporarily dedicated to communication. This means CCL operations directly compete with compute workloads for core resources.

### 5d. Program Artifact Building

The actual kernel creation is delegated to `build_all_gather_async_minimal_default_program_artifacts`, which:

1. Creates circular buffers on each worker core for staging data
2. Creates reader kernels that read tensor slices from L1/DRAM
3. Creates writer kernels that send data through the fabric to remote devices
4. Encodes the command stream as runtime arguments (see File 03)
5. Sets up fabric connections via `append_fabric_connection_rt_args` or `append_routing_plane_connection_manager_rt_args`

---

## Runtime Argument Override and Program Caching

The `AllGatherProgram::shared_variables_t` struct holds state that persists across cached program reuses:

```cpp
struct shared_variables_t {
    std::vector<tt::tt_metal::GlobalSemaphore> multidevice_semaphores;
    tt::tt_metal::GlobalSemaphore barrier_semaphore;
    AllGatherProgramArtifacts program_artifacts;
};
```

When a cached program is reused with different tensors, the `override_runtime_arguments` method updates only the dynamic portions (buffer addresses, semaphore values) without recreating the program. This is a performance-critical optimization: recompiling kernels is expensive, but updating buffer addresses in the runtime arguments is cheap. The program hash (`compute_program_hash`) determines whether a cached program can be reused.

---

## The Reduce-Scatter Device Operation (Structural Parallel)

To demonstrate that the all-gather pattern is representative, the reduce-scatter device operation (`ReduceScatterDeviceOperation`) follows an identical structure:

- `operation_attributes_t` adds `optional_intermediate_mem_config` and `compute_kernel_config`
- `tensor_args_t` has the same shape
- `ReduceScatterProgram::create_mesh_workload` creates 2 global semaphores + 1 barrier semaphore
- `create_at` discovers forward/backward neighbors identically
- The program factory creates both data movement kernels **and compute kernels** (for the reduction math)

The key architectural insight is that **all fabric-based CCL operations share the same multi-device orchestration pattern**: semaphore allocation, mesh synchronization, per-device neighbor discovery, and program creation with runtime argument override.

---

## The Explicit Orchestration Cost

To summarize the explicit work done by the host for each CCL invocation:

| Step | Host Computation | Per-Device? |
|------|-----------------|-------------|
| Topology resolution | Boundary mode check, downgrade logic | No (global) |
| Semaphore allocation | 2-3 global semaphores across mesh | No (global) |
| Mesh synchronization | Host barrier | No (global) |
| Neighbor discovery | Forward/backward coordinate lookup | Yes |
| Device index computation | Linearize mesh coordinate | Yes |
| Core selection | Intersect subdevice grid, allocate workers | Yes |
| Program creation | Kernel compilation, CB allocation, RT args | Yes |
| Runtime arg override | Update tensor addresses, semaphore values | Yes |

For a 32-device mesh, steps marked "Yes" execute 32 times. This is significant host-side overhead that a transparent runtime could amortize or eliminate.

---

## Key Takeaways

- **Five layers of indirection** separate the user API from device kernel execution: invoke, primitive dispatch, device operation framework, mesh workload factory, and per-device program creation. Understanding one operation's anatomy is sufficient to understand the skeleton of all nine.
- **Global semaphores are the synchronization mechanism:** Two semaphores handle forward/backward link coordination within the operation, and a barrier semaphore ensures all devices have allocated buffers before data movement begins. This explicit synchronization would need to be replicated or replaced by any transparent alternative.
- **Program caching is critical for performance:** The `compute_program_hash` / `override_runtime_arguments` pattern avoids recompiling kernels when only tensor addresses change. A transparent communication model must maintain equivalent reuse efficiency.
- **Neighbor discovery is per-device and topology-aware:** Each device independently determines its forward and backward communication partners based on its mesh coordinate and the requested topology. This topology awareness is pervasive and deeply embedded in the program creation logic.
- **All fabric-based CCL operations share the same orchestration pattern**, suggesting that a unified runtime layer could replace the per-operation host orchestration with a single, generic multi-device communication manager.

## Source Code References

- `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp` -- `ExecuteAllGather::invoke` implementation
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.hpp` -- Device operation structure and `AllGatherProgram`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp` -- Validation, output spec computation, program hash, `ttnn::prim::all_gather`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_program_factory.cpp` -- `create_mesh_workload` and `create_at` implementations
- `ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp` -- `choose_worker_cores`, `get_physical_neighbor_from_physical_coord`, `SenderReceiverConfig`, `RingTopology`, `LineTopology`, topology utility functions
- `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp` -- `get_usable_topology`, `get_topological_dimension`, `get_physical_neighbor_from_physical_coord`
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_device_operation.hpp` -- Parallel structure for reduce-scatter
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_device_operation.hpp` -- Async variant and `AllGatherProgramArtifacts`

---

**Next:** [`03_command_stream_architecture.md`](./03_command_stream_architecture.md)
