# 01 -- Collective Operations Catalog

## Context

Before proposing to eliminate explicit CCL kernels, we must understand precisely what they do. This file catalogs every collective communication operation currently implemented in the `ttnn/cpp/ttnn/operations/ccl/` directory of TT-Metal. For each operation, we document the public API signature (the `invoke` method on the registered operation struct), the supported fabric topologies, supported memory configurations, data types, and the semantic behavior. The reader should come away with a complete inventory of the CCL surface area that any replacement abstraction must be able to replicate -- or consciously choose not to replicate.

---

## Topology Types

All CCL operations reference a shared `Topology` enum defined in `tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp`:

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp
enum class Topology {
    NeighborExchange = 0,  // Direct neighbor-to-neighbor communication
    Linear = 1,            // Open-ended line of devices
    Ring = 2,              // Closed ring of devices (wraps around)
    Mesh = 3,              // 2D mesh (open boundaries)
    Torus = 4              // 2D torus (wrapped boundaries)
};
```

Helper predicates classify these into families:

- **1D topologies:** `NeighborExchange`, `Linear`, `Ring`
- **2D topologies:** `Mesh`, `Torus` (identified by `is_2D_topology()`)
- **Wrapping topologies:** `Ring`, `Torus` (identified by `is_ring_or_torus()`)

The `ttnn::ccl::Topology` alias in `ccl_host_types.hpp` re-exports this enum into the CCL namespace via `using tt::tt_fabric::Topology`.

When a user does not specify a topology, the runtime function `get_usable_topology()` in `ccl_common.hpp` infers one from the tensor's mesh shape and the active fabric configuration. For 2D fabric configurations, `convert_2d_to_1d_topology()` maps `Mesh` to `Linear` and `Torus` to `Ring` for operations that only support 1D communication patterns.

---

## Overview of CCL Operations

The TT-Metal CCL library implements nine distinct collective operations, each registered as a TTNN operation via the `ttnn::register_operation` decorator pattern:

| Operation | Communication Pattern | Return Type | Primary Use Case | Complexity |
|-----------|----------------------|-------------|------------------|------------|
| **all-gather** | Gather + replicate | Single tensor | Data-parallel weight synchronization, tensor-parallel column gather | High (full pipeline) |
| **reduce-scatter** | Reduce + partition | Single tensor | Gradient reduction with sharding | High (reduce + scatter) |
| **all-reduce** | Reduce + replicate | Single tensor | Gradient synchronization (experimental, composed) | Medium (composition) |
| **all-broadcast** | Replicate from each source | Vector of tensors | Multi-source replication patterns | Medium |
| **broadcast** | Single-source replicate | Single tensor | Parameter broadcasting from root | Medium |
| **reduce-to-root** | Reduce to single device | Vector of tensors | Centralized aggregation with scaling | High (3 input streams) |
| **all-to-all-dispatch** | Sparse data routing | Array of 2 tensors | MoE expert token dispatch | Very high (data-dependent) |
| **all-to-all-combine** | Sparse data collection | Single tensor | MoE expert output combination | Very high (data-dependent) |
| **mesh-partition** | Local slicing by position | Single tensor | Splitting replicated tensors along mesh axis | Low (local only) |

---

## 1. All-Gather (`ttnn::all_gather`)

**Semantics:** Each device in the topology holds a shard of a tensor along dimension `dim`. After the operation, every device holds the full concatenated tensor. The output shape along `dim` is $N \times \text{input shape}[\text{dim}]$ where $N$ is the number of devices participating (the "ring size").

**Registered operation:** `ExecuteAllGather` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp
struct ExecuteAllGather {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
        std::optional<uint32_t> num_links = std::nullopt,
        std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);
};
```

**Key parameters:**

| Parameter | Purpose |
|-----------|---------|
| `dim` | The tensor dimension along which shards are concatenated |
| `cluster_axis` | Which mesh dimension to gather along (0 = rows, 1 = columns). When `std::nullopt` and the mesh is not a line topology, the implementation recurses through all mesh dimensions in reverse order. |
| `topology` | Fabric topology to use (`Ring`, `Linear`, `Mesh`, `Torus`, `NeighborExchange`) |
| `num_links` | Number of Ethernet links to use per direction; defaults to the available link count |
| `num_workers_per_link` | Number of Tensix worker cores assigned per link |
| `num_buffers_per_channel` | Number of double-buffering slots per EDM channel |
| `chunks_per_sync` | Synchronization granularity -- how many chunks to send before synchronizing |
| `sub_core_grid` | Optional constraint on which worker cores may be used |

**Supported topologies:** `Ring`, `Linear`, `Mesh`, `Torus` (Mesh/Torus are converted to Linear/Ring via `convert_2d_to_1d_topology`)

**Supported memory configurations:**
- `INTERLEAVED` (L1 or DRAM)
- `WIDTH_SHARDED` (L1)
- `HEIGHT_SHARDED` (L1)
- `BLOCK_SHARDED` (L1 only -- DRAM block sharding is not supported for input)

**Data types:** All standard TT-Metal data types (BFloat16, Float32, BFloat8_b, BFloat4_b, UInt16, UInt32). The operation is agnostic to data type; it operates at page granularity.

**Special behavior:** When `cluster_axis` is `std::nullopt` and the mesh has a non-line topology (e.g., 2x4), the implementation automatically decomposes the all-gather into sequential per-axis all-gathers, iterating from the last mesh dimension to the first.

See Section "Experimental and Async Variants" for async/fused variants of this operation.

---

## 2. Reduce-Scatter (`ttnn::reduce_scatter`)

**Semantics:** Each device holds an input tensor. The operation reduces (sums) corresponding elements across all devices and then scatters the result so that each device holds a non-overlapping shard of the reduced output. The output shape along `dim` is $\text{input shape}[\text{dim}] / N$.

**Registered operation:** `ExecuteReduceScatter` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp
struct ExecuteReduceScatter {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& intermediate_memory_config = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
        std::optional<uint32_t> num_links = std::nullopt,
        std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
        std::optional<uint32_t> chunks_per_sync = std::nullopt,
        std::optional<uint32_t> num_workers_per_link = std::nullopt,
        std::optional<uint32_t> num_buffers_per_channel = std::nullopt,
        const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};
```

**Additional parameters vs. all-gather:**

| Parameter | Purpose |
|-----------|---------|
| `intermediate_memory_config` | Optional memory config for intermediate reduction buffers |
| `compute_kernel_config` | Compute kernel configuration (math fidelity, fp32 accumulation, etc.) for the reduction compute kernel |

**Supported topologies:** Same as all-gather (`Ring`, `Linear`, `Mesh`, `Torus`), with the same boundary-mode downgrade logic.

**Supported memory configurations:** `INTERLEAVED` (L1 or DRAM), `WIDTH_SHARDED`, `HEIGHT_SHARDED`, `BLOCK_SHARDED` (L1 only)

See Section "Experimental and Async Variants" for async/fused variants of this operation.

---

## 3. All-Reduce (`ttnn::all_reduce`)

**Semantics:** Each device holds an input tensor of the same shape. After the operation, every device holds the element-wise reduction (sum) of all inputs. Semantically equivalent to a reduce-scatter followed by an all-gather, though the implementation may be optimized differently. Marked as **experimental** in the codebase.

**Registered operation:** `ExecuteAllReduce` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.hpp
struct ExecuteAllReduce {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<uint32_t> num_links = std::nullopt,
        std::optional<tt::tt_fabric::Topology> topology = std::nullopt);
};
```

**Notable differences:** No `dim` parameter (the reduction is element-wise across the full tensor). Fewer tuning knobs compared to all-gather and reduce-scatter (no `chunks_per_sync`, `num_workers_per_link`, or `num_buffers_per_channel`). This reflects the composed implementation: reduce-scatter followed by all-gather, with the component operations handling tuning internally. The experimental `all_reduce_async` provides a more optimized single-pass implementation.

**Supported topologies:** `Ring`, `Linear`

---

## 4. All-Broadcast (`ttnn::all_broadcast`)

**Semantics:** Every device broadcasts its local tensor to all other devices. Each device receives $N$ tensors (one from each source device, including itself), returned as a vector. This is a symmetric operation where all devices are both senders and receivers simultaneously.

**Registered operation:** `ExecuteAllBroadcast` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/all_broadcast/all_broadcast.hpp
struct ExecuteAllBroadcast {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        std::optional<uint32_t> num_links = 1,
        std::optional<ttnn::ccl::Topology> topology = ttnn::ccl::Topology::Linear);
};
```

**Notable details:**
- Returns `std::vector<ttnn::Tensor>` rather than a single tensor -- one output tensor per device in the topology, where each output tensor at index $i$ is a copy of device $i$'s input
- Currently defaults to `num_links = 1` and `topology = Linear` with explicit TODO comments noting that multi-link and non-linear topology support needs testing (issue #30798)
- Uses `ttnn::ccl::Topology` (an alias for `tt::tt_fabric::Topology`) rather than `tt::tt_fabric::Topology` directly

**Supported topologies:** `Linear` (currently; `Ring` support pending)

---

## 5. Broadcast (`ttnn::broadcast`)

**Semantics:** Broadcasts the input tensor from a single specified sender device to all other devices in the topology. Unlike all-broadcast where every device sends, here only one device (identified by `sender_coord`) is the source.

**Registered operation:** `ExecuteBroadcast` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/broadcast/broadcast.hpp
struct ExecuteBroadcast {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const MeshCoordinate& sender_coord,
        uint32_t num_links = 1,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Linear,
        std::optional<uint32_t> cluster_axis = std::nullopt,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);
};
```

**Key difference from all-broadcast:** The `sender_coord` parameter (a `MeshCoordinate`) explicitly identifies which device in the mesh is the source of the broadcast. All other devices receive a copy. Has dedicated device kernels for both row-major and tile-layout data: `broadcast_rm_reader.cpp`, `broadcast_rm_writer.cpp`, `broadcast_tile_reader.cpp`, `broadcast_tile_writer.cpp`.

**Supported topologies:** `Linear` (default)

---

## 6. Reduce-to-Root (`ttnn::reduce_to_root`)

**Semantics:** Reduces input tensors from all devices onto a single root device. The root device receives the reduced result; other devices' outputs are undefined (or may receive passthrough copies depending on the implementation).

**Registered operation:** `ExecuteReduceToRoot` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/reduce_to_root/reduce_to_root.hpp
struct ExecuteReduceToRoot {
    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor_l,
        const ttnn::Tensor& input_tensor_s,
        const ttnn::Tensor& input_tensor_m,
        const MeshCoordinate& root_coord,
        float scale_fp32,
        tt::tt_fabric::Topology topology,
        const std::optional<ttnn::Tensor>& optional_output_tensor_l = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor_s = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor_m = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_intermediate_tensor = std::nullopt,
        const std::optional<std::vector<ttnn::CoreCoord>>& input_mux_cores = std::nullopt);
};
```

**Notable characteristics:**
- Takes **three** input tensors (`input_tensor_l`, `input_tensor_s`, `input_tensor_m`) -- designed for multi-component reduction (e.g., different precision components in block floating-point formats)
- Returns `std::vector<ttnn::Tensor>` (one per output tensor type)
- Requires a `root_coord` (`MeshCoordinate`) specifying the target device
- Includes a `scale_fp32` parameter for scaling the reduction result
- Supports optional MUX cores (`input_mux_cores`) for traffic aggregation via the Tensix MUX path
- `topology` is a required (non-optional) parameter
- Has dedicated kernel implementations: sender reader/writer, root receive reader/writer, and a compute kernel

**Supported topologies:** `Linear`, `Ring`

---

## 7. All-to-All Dispatch (`ttnn::all_to_all_dispatch`)

**Semantics:** The "dispatch" half of a two-phase all-to-all exchange, designed for **Mixture-of-Experts (MoE)** workloads. Each device sends different data to different destinations based on expert assignment indices.

**Registered operation:** `ExecuteAllToAllDispatch` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch.hpp
struct ExecuteAllToAllDispatch {
    static std::array<ttnn::Tensor, 2> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_indices_tensor,
        const ttnn::Tensor& expert_mapping_tensor,
        std::optional<uint32_t> axis = std::nullopt,
        const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors = std::nullopt,
        std::optional<uint32_t> num_links = std::nullopt,
        std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<uint32_t>& output_concat_dim = std::nullopt);
};
```

**Notable characteristics:**
- Returns `std::array<ttnn::Tensor, 2>` (the dispatched data and associated metadata)
- Takes two additional metadata tensors: `expert_indices_tensor` (which expert each token maps to) and `expert_mapping_tensor` (mapping from experts to devices)
- Supports two transfer modes: `FullPacket` (all pages buffered in intermediate storage) and `PageByPage` (pages sent directly to conserve L1 space)
- This operation represents data-dependent, irregular communication -- the pattern depends on runtime expert assignment rather than being fixed at compile time

**Supported topologies:** `Linear`, `Ring`

---

## 8. All-to-All Combine (`ttnn::all_to_all_combine`)

**Semantics:** The "combine" half of the MoE all-to-all exchange. After experts have processed their assigned tokens, this operation routes the results back to the originating devices.

**Registered operation:** `ExecuteAllToAllCombine` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/all_to_all_combine.hpp
struct ExecuteAllToAllCombine {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_mapping_tensor,
        const ttnn::Tensor& expert_metadata_tensor,
        bool locally_reduced = true,
        std::optional<uint32_t> num_links = std::nullopt,
        std::optional<tt::tt_fabric::Topology> topology = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<uint32_t>& axis = std::nullopt,
        const std::optional<uint32_t>& output_shard_dim = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt);
};
```

**Notable characteristics:**
- The `locally_reduced` flag indicates whether expert outputs have already been locally reduced before combining
- Takes `expert_mapping_tensor` and `expert_metadata_tensor` to know how to route results back
- Returns a single `ttnn::Tensor` (the recombined output)
- Together with all-to-all-dispatch, this pair supports the irregular, data-dependent communication patterns characteristic of MoE models

**Supported topologies:** `Linear`, `Ring`

---

## 9. Mesh Partition (`ttnn::mesh_partition`)

**Semantics:** Partitions a replicated tensor across devices along a specified dimension. This is conceptually the inverse of all-gather: given a tensor replicated on all devices, each device retains only its assigned shard.

**Registered operation:** `ExecuteMeshPartition` in `ttnn::operations::ccl`

**API Signature:**

```cpp
// ttnn/cpp/ttnn/operations/ccl/mesh_partition/mesh_partition.hpp
struct ExecuteMeshPartition {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        std::optional<uint32_t> cluster_axis,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};
```

**Notable characteristics:**
- The simplest CCL operation in terms of API surface -- only `dim` and `cluster_axis` are required beyond the input
- No topology parameter (partitioning is a local operation that does not require inter-device communication -- it simply slices the local tensor based on the device's position in the mesh)
- Internally delegates to the `SliceDeviceOperation` (from `data_movement/slice/`) with per-device slice offsets computed from the mesh position

---

## Topology Support Matrix

| Operation | `Linear` | `Ring` | `Mesh` | `Torus` | `NeighborExchange` |
|-----------|:--------:|:------:|:------:|:-------:|:-----------------:|
| All-Gather | Yes | Yes | Yes* | Yes* | -- |
| Reduce-Scatter | Yes | Yes | Yes* | Yes* | -- |
| All-Reduce | Yes | Yes | -- | -- | -- |
| All-Broadcast | Yes | Planned | -- | -- | -- |
| Broadcast | Yes | -- | -- | -- | -- |
| Reduce-to-Root | Yes | Yes | -- | -- | -- |
| All-to-All Dispatch | Yes | Yes | -- | -- | -- |
| All-to-All Combine | Yes | Yes | -- | -- | -- |
| Mesh Partition | N/A | N/A | N/A | N/A | N/A |

*\* Mesh and Torus topologies are converted to Linear and Ring respectively via `convert_2d_to_1d_topology()` when running on 1D fabric configurations.*

---

## Experimental and Async Variants

Beyond the nine stable operations, TT-Metal maintains several experimental CCL variants in `ttnn/cpp/ttnn/operations/experimental/ccl/`:

| Variant | Description |
|---------|-------------|
| `all_gather_async` | Asynchronous all-gather that can overlap with compute |
| `strided_all_gather_async` | All-gather with strided access patterns for optimized memory bandwidth |
| `ring_attention_all_gather_async` | Specialized all-gather for ring-attention inference workloads |
| `all_gather_concat_heads_fused` | Fused all-gather with attention head concatenation |
| `all_gather_matmul_async` / `llama_all_gather_matmul_async` | Fused all-gather + matrix multiplication for LLM inference |
| `reduce_scatter_minimal_async` | Lightweight asynchronous reduce-scatter with minimal overhead |
| `llama_reduce_scatter` | Model-specific reduce-scatter optimized for LLaMA architecture |
| `matmul_reduce_scatter_async` | Fused matmul + reduce-scatter |
| `deepseek_moe_reduce_scatter` | MoE-specific reduce-scatter for DeepSeek |
| `all_reduce_async` | Async all-reduce variant |
| `rms_allgather` | Fused RMS normalization with all-gather |
| `all_to_all_async` | Async variant of all-to-all |

These experimental operations demonstrate a key trend: the CCL layer is increasingly specialized per-workload, with fused operations that combine communication with computation. This specialization is part of the problem that motivates the GSRP proposal -- each new workload pattern currently requires a new hand-crafted CCL kernel.

---

## Common Memory Configuration Support

Most CCL operations support the following `TensorMemoryLayout` values:

| Memory Layout | Buffer Type | Support Status |
|--------------|-------------|----------------|
| `INTERLEAVED` | `DRAM` | Supported by all-gather, reduce-scatter, and most operations |
| `INTERLEAVED` | `L1` | Supported by all-gather, reduce-scatter, and most operations |
| `WIDTH_SHARDED` | `L1` | Supported by all-gather, reduce-scatter |
| `HEIGHT_SHARDED` | `L1` | Supported by all-gather, reduce-scatter |
| `BLOCK_SHARDED` | `L1` | Supported by all-gather (L1 only), reduce-scatter |
| `BLOCK_SHARDED` | `DRAM` | Not supported for all-gather input |

See File 03 Section 8 for the address generator details for each layout type.

---

## Key Takeaways

- **Nine distinct collective operations** are currently implemented as explicit CCL kernels, each with its own device operation, program factory, and kernel set. Any replacement abstraction must handle or decompose into the communication patterns embodied by all nine.
- **Topology support is primarily 1D.** Even when 2D topologies (Mesh, Torus) are specified, they are decomposed to 1D (Linear, Ring) for execution. True 2D collective algorithms are not yet implemented in the stable CCL path. This asymmetry reflects the maturity gradient -- all-gather and reduce-scatter have received the most optimization attention.
- **The API surface is large and growing:** operations range from simple data movement (mesh-partition is purely local) to complex data-dependent routing (all-to-all-dispatch/combine for MoE). The irregular MoE operations are particularly challenging for any "transparent" communication model because their patterns are only known at runtime.
- **Experimental fused operations** (all-gather-matmul, llama-reduce-scatter, rms-allgather) demonstrate that the explicit CCL model is being extended toward compute-communication overlap, not simplified -- the complexity trajectory is upward. Each new workload pattern currently requires a new hand-crafted CCL kernel.
- **Tuning knobs are pervasive:** `num_links`, `num_workers_per_link`, `num_buffers_per_channel`, and `chunks_per_sync` give the programmer fine-grained control over communication performance. A transparent model would need to either replicate this tunability or demonstrate that runtime decisions can match hand-tuned configurations.

## Source Code References

- `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp` -- All-gather API
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.hpp` -- Reduce-scatter API
- `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.hpp` -- All-reduce API
- `ttnn/cpp/ttnn/operations/ccl/all_broadcast/all_broadcast.hpp` -- All-broadcast API
- `ttnn/cpp/ttnn/operations/ccl/broadcast/broadcast.hpp` -- Broadcast API
- `ttnn/cpp/ttnn/operations/ccl/reduce_to_root/reduce_to_root.hpp` -- Reduce-to-root API
- `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/all_to_all_dispatch.hpp` -- All-to-all dispatch API
- `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/all_to_all_combine.hpp` -- All-to-all combine API
- `ttnn/cpp/ttnn/operations/ccl/mesh_partition/mesh_partition.hpp` -- Mesh partition API
- `ttnn/cpp/ttnn/operations/ccl/ccl_host_types.hpp` -- Topology type alias
- `tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp` -- `Topology` enum definition
- `ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp` -- Sharded address generators
- `ttnn/cpp/ttnn/operations/experimental/ccl/` -- Experimental/async CCL variants

---

**Next:** [`02_device_operation_anatomy.md`](./02_device_operation_anatomy.md)
