# 04 -- Host Orchestration and Topology

## Context

The previous three files covered what CCL operations exist (File 01), how they are structured as device operations (File 02), and how their data movement instructions are encoded as command streams (File 03). This final file in Chapter 1 covers the host-side orchestration layer that ties everything together: how CCL operations discover their communication partners, how topology types map to concrete device relationships, how boundary modes determine ring-wrap versus line-endpoint behavior, how CCL op fusion enables overlapping communication with computation, and how the codebase is transitioning from the legacy ERISC async datamover kernel path to the newer fabric-based CCL path. The reader should understand the full orchestration surface area that makes explicit CCL programming possible -- and complex.

---

## 1. Communication Partner Discovery

### 1.1 `SenderReceiverConfig`

The `SenderReceiverConfig` structure is the fundamental answer to the question "who does this device talk to?":

```cpp
// ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp
struct SenderReceiverConfig {
    uint32_t device_index = 0;                       // This device's position in the ring/line
    std::optional<tt::ChipId> sender_device_id;      // Device that sends TO this device (upstream)
    std::optional<tt::ChipId> receiver_device_id;     // Device that receives FROM this device (downstream)
};
```

The optional fields model endpoint behavior: in a `Linear` topology, the first device has no `sender_device_id` and the last device has no `receiver_device_id`. In a `Ring` topology, both are always populated.

### 1.2 Discovery Functions

Two functions produce `SenderReceiverConfig` for different use cases:

**For flat device lists (legacy path):**

```cpp
SenderReceiverConfig get_device_sender_receiver_config(
    const IDevice* target_device,
    const std::vector<IDevice*>& devices,
    ttnn::ccl::Topology topology);
```

This function iterates through the device list, finds the target device, and assigns sender/receiver based on the topology. For `Ring` topology, indices wrap via modular arithmetic:

```cpp
config.receiver_device_id = is_last_chip_in_clockwise_direction
    ? std::nullopt
    : std::optional<tt::ChipId>(devices.at((i + 1) % num_devices)->id());
config.sender_device_id = is_last_chip_in_counter_clockwise_direction
    ? std::nullopt
    : std::optional<tt::ChipId>(devices.at((i + num_devices - 1) % num_devices)->id());
```

**For mesh-coordinate-based discovery (newer path):**

```cpp
SenderReceiverConfig get_device_sender_receiver_config_in_ring(
    const MeshCoordinate& mesh_coord,
    const distributed::MeshDevice* mesh_device,
    uint32_t cluster_axis,
    int ring_size);
```

This function operates on `MeshCoordinate` values and uses the mesh view to look up device IDs. It restricts to a specific `cluster_axis` (row or column of the mesh).

### 1.3 Physical Neighbor Discovery

The most general neighbor discovery function operates directly on physical mesh coordinates:

```cpp
std::optional<MeshCoordinate> get_physical_neighbor_from_physical_coord(
    const Tensor& tensor,
    const MeshCoordinate& physical_coord,
    int offset,                              // +1 for forward, -1 for backward
    ttnn::ccl::Topology topology,
    const std::optional<uint32_t>& cluster_axis);
```

This function handles two cases:

1. **With `cluster_axis`:** Uses `physical_coord.get_neighbor(mesh_shape, offset, axis, boundary_mode)` to compute the neighbor along the specified axis. The neighbor must exist in the tensor's `device_coords` set to be valid.

2. **Without `cluster_axis`:** Linearizes the device coordinates and applies the offset. For `WRAP` boundary mode, the index wraps via modular arithmetic. For `NONE` boundary mode, out-of-range indices return `std::nullopt`.

The boundary behavior by topology:

| Topology | Boundary Behavior (offset at boundary) |
|----------|---------------------------------------|
| `Linear` / `Mesh` | Returns `std::nullopt` -- no wrapping |
| `Ring` / `Torus` | Returns the wrapped coordinate -- device $N-1$ at offset $+1$ yields device $0$ |
| `NeighborExchange` | Only offset $\pm 1$ is valid |

---

## 2. Boundary Modes and Topology Degradation

The boundary mode determines whether the device ring wraps around or has endpoints:

```cpp
tt::tt_metal::distributed::MeshCoordinate::BoundaryMode get_boundary_mode(
    const Tensor& tensor, tt::tt_fabric::Topology topology, std::optional<uint32_t> cluster_axis);
```

The boundary mode is `WRAP` when all of the following conditions hold:
- The topology is `Ring` or `Torus`, AND
- The tensor's device coordinates span the full range along the cluster axis (from 0 to `mesh_shape[axis] - 1`), AND
- The dimension size along the axis is greater than 2 (at dimension size 2, a ring degenerates to a line because both directions reach the same device)

Otherwise, the boundary mode is `NONE`, which triggers topology degradation:
- `Ring` -> `Linear`
- `Torus` -> `Mesh`

This degradation is handled by `get_usable_topology`:

```cpp
tt::tt_fabric::Topology get_usable_topology(
    const Tensor& tensor,
    const std::optional<tt::tt_fabric::Topology>& topology,
    const std::optional<uint32_t>& cluster_axis) {
    tt::tt_fabric::Topology topology_ = topology.value_or(tt::tt_fabric::get_fabric_topology());
    if (topology_ == tt::tt_fabric::Topology::Ring || topology_ == tt::tt_fabric::Topology::Torus) {
        auto boundary_mode = get_boundary_mode(tensor, topology_, cluster_axis);
        if (boundary_mode == MeshCoordinate::BoundaryMode::WRAP) {
            return topology_;       // Ring stays Ring, Torus stays Torus
        }
        if (topology_ == tt::tt_fabric::Topology::Torus) {
            return Topology::Mesh;  // Torus degrades to Mesh
        }
        return Topology::Linear;   // Ring degrades to Linear
    }
    return topology_;
}
```

The `convert_2d_to_1d_topology` function (described in File 02) collapses 2D topologies to their 1D equivalents for operations that only support 1D communication.

> **Key Insight:** The automatic topology degradation means that the same CCL operation code handles both ring and line topologies transparently. The only difference is whether endpoints have neighbors or not, which flows through the `std::optional` sender/receiver fields. This also means the system gracefully handles submeshes where not all devices along an axis are present.

---

## 3. Topology Descriptor Classes

### 3.1 `RingTopology`

The `RingTopology` class encapsulates the physical Ethernet connection information for a device's position in a ring or line:

```cpp
struct RingTopology {
    RingTopology(
        const IDevice* device, Topology topology,
        std::optional<uint32_t> sender_device_id,
        std::optional<uint32_t> receiver_device_id,
        uint32_t num_links, uint32_t ring_size, uint32_t ring_index);

    bool is_first_device_in_line(bool in_clockwise_direction) const;
    bool is_last_device_in_line(bool in_clockwise_direction) const;

    const IDevice* device;
    std::vector<CoreCoord> eth_sender_cores;    // Ethernet cores for sending
    std::vector<CoreCoord> eth_receiver_cores;  // Ethernet cores for receiving
    uint32_t num_links;
    uint32_t ring_size;
    uint32_t ring_index;
    bool is_linear;   // true for Linear/Mesh, false for Ring/Torus
};
```

The constructor discovers the actual Ethernet cores on the device that connect to the sender and receiver devices, populating `eth_sender_cores` and `eth_receiver_cores`. In a Ring topology, no device is first or last (all devices have both neighbors). In a Linear topology, the device at `ring_index == 0` is first in the clockwise direction and last in the counter-clockwise direction.

### 3.2 `LineTopology`

The `LineTopology` class models a simpler linear arrangement:

```cpp
class LineTopology {
public:
    LineTopology(size_t line_size, size_t line_index);

    bool is_first_device_in_line(LineDirection direction) const;
    bool is_last_device_in_line(LineDirection direction) const;
    bool is_at_end_of_line() const;
    size_t line_size() const;
    size_t line_index() const;
    size_t get_distance_to_end_of_line(LineDirection direction) const;
    Topology topology() const;  // Always returns Topology::Linear
};
```

The `get_distance_to_end_of_line` method is used to compute hop counts for unicast command destinations:
- Forward direction: $d = (L - i) - 1$ where $L$ is the line size and $i$ is the line index
- Backward direction: $d = i$ where $i$ is the line index

```cpp
enum class LineDirection : uint8_t {
    FORWARD,   // Increasing index (clockwise in ring)
    BACKWARD,  // Decreasing index (counter-clockwise in ring)
};
```

---

## 4. Topological Dimension and Index Computation

Two utility functions compute the ring size and per-device index:

**`get_topological_dimension`:** When `cluster_axis` is specified, returns the number of unique coordinate values along that axis in the tensor's device coordinates (the ring size for that axis). When absent, returns the total number of devices the tensor is distributed across.

**`get_linearized_index_from_physical_coord`:** When `cluster_axis` is specified, returns the device's position along that axis (subtracting the minimum value to handle non-zero-based axes). When absent, returns the position in the linear device coordinate list.

---

## 5. Forward/Backward Configuration

For bidirectional communication (used by all-gather and reduce-scatter), devices need to send/receive in both forward and backward directions simultaneously:

```cpp
// Unicast configuration
std::tuple<std::array<uint32_t, 2>, std::array<uint32_t, 2>>
get_forward_backward_line_unicast_configuration(
    const MeshCoordinate& src_device_coord,
    const std::optional<MeshCoordinate>& forward_device_coord,
    const std::optional<MeshCoordinate>& backward_device_coord,
    MeshDevice* mesh_device);

// Multicast configuration (6-element arrays encode full route specification)
std::tuple<std::array<uint32_t, 6>, std::array<uint32_t, 6>>
get_forward_backward_line_mcast_configuration(
    const MeshCoordinate& src_device_coord,
    const std::optional<MeshCoordinate>& forward_device_coord,
    const std::optional<MeshCoordinate>& backward_device_coord,
    uint32_t num_targets_forward,
    uint32_t num_targets_backward,
    MeshDevice* mesh_device);

// Distance computation
std::tuple<uint32_t, uint32_t> get_forward_backward_line_mcast_distance(
    size_t ring_size, size_t ring_index, Topology topology, bool static_alternate);
```

These functions translate the abstract topology position into concrete routing parameters consumed by the command stream encoding.

---

## 6. CCL Op Fusion

CCL op fusion enables overlapping communication (all-gather or reduce-scatter) with computation (typically matmul). This is critical for performance because it hides communication latency behind compute.

### 6.1 `AllGatherFusedOpSignaler`

```cpp
struct AllGatherFusedOpSignaler {
    uint32_t num_fused_op_cores_to_signal = 0;
    std::vector<CoreCoord> fused_op_receiver_cores_noc;
    std::vector<uint32_t> fused_op_receiver_signal_semaphores;
    FusedOpSignalerMode fused_op_signaler_mode = FusedOpSignalerMode::MULTI;

    std::vector<CoreCoord> all_gather_worker_cores_noc;
    uint32_t all_gather_worker_sync_semaphore = 0;

    void init_fused_op(
        const std::vector<CoreCoord>& fused_op_receiver_cores_noc,
        const std::vector<uint32_t>& fused_op_receiver_signal_semaphores,
        FusedOpSignalerMode mode = FusedOpSignalerMode::MULTI);
    void init_all_gather(
        Program& program, const IDevice* device,
        const CoreRangeSet& all_gather_workers,
        std::vector<CoreCoord>& all_gather_worker_cores);
    void push_all_gather_fused_op_rt_args(
        std::vector<uint32_t>& out_rt_args,
        uint32_t num_workers_to_sync,
        uint32_t curr_worker_index,
        uint32_t all_gather_direction);
};
```

The signaling mode determines granularity:

```cpp
enum class FusedOpSignalerMode {
    SINGLE,  // Signal one fused op core per slice
    MULTI    // Signal all fused op cores per slice
};
```

At runtime, as each all-gather worker completes a slice, it signals the fused operation's receiver cores via semaphore increment, enabling the downstream matmul to begin processing that slice before the entire all-gather is complete.

### 6.2 `ReduceScatterFusedOpSignaler`

```cpp
struct ReduceScatterFusedOpSignaler {
    uint32_t num_fused_op_cores_to_signal = 1;
    std::vector<CoreCoord> fused_op_receiver_cores_noc;
    std::vector<uint32_t> fused_op_receiver_signal_semaphores;
    FusedOpSignalerMode fused_op_signaler_mode = FusedOpSignalerMode::SINGLE;

    void init_reduce_scatter(Program& program, const IDevice* device,
        const std::variant<CoreRange, CoreRangeSet>& core_range_to_signal);
    void push_reduce_scatter_fused_op_rt_args(std::vector<uint32_t>& out_rt_args);
};
```

### 6.3 `MatmulFusedOpSignaler`

The most complex signaler manages bidirectional fusion between matmul and either all-gather or reduce-scatter:

```cpp
struct MatmulFusedOpSignaler {
    MatmulFusedOpSignalerType fused_op_type;

    // All-gather fusion parameters
    uint32_t num_transfers, ring_size, start_ring_index;
    uint32_t tensor_slice_shape_width, output_page_offset;
    bool is_clockwise_dir;

    // Reduce-scatter fusion parameters
    std::vector<CoreCoord> matmul_worker_cores_noc;
    uint32_t matmul_worker_sync_semaphore;

    // Llama-specific fusion parameters
    uint32_t matmul_privilaged_semaphore, rs_semaphore;
    CoreRangeSet rs_cores;
    CoreCoord privilaged_core;

    void push_matmul_fused_op_rt_args(
        std::vector<uint32_t>& out_rt_args,
        uint32_t curr_worker_in0_idx,
        uint32_t curr_worker_in1_idx);
};
```

The `MatmulFusedOpSignaler` supports five different fusion patterns:

| Type | Description |
|------|-------------|
| `ALL_GATHER` | Standard all-gather fused with matmul |
| `REDUCE_SCATTER` | Standard reduce-scatter fused with matmul |
| `LLAMA_REDUCE_SCATTER` | Optimized reduce-scatter for LLaMA's all-reduce pattern |
| `LLAMA_ALL_GATHER` | Optimized all-gather for LLaMA |
| `EMPTY` | No fusion (standalone matmul) |

The `LLAMA_REDUCE_SCATTER` and `LLAMA_ALL_GATHER` variants add a "privileged core" concept where one designated matmul core acts as a coordinator, signaling the reduce-scatter workers via a dedicated semaphore once it has accumulated sufficient results.

### 6.4 `MinimalMatmulFusedOpSignaler`

A simplified fusion signaler for the newer async all-gather path, carrying topology information (`ring_size`, `start_ring_index`, `topology`) alongside the standard signaler fields. This reflects the trend toward simpler fusion interfaces, though the underlying complexity remains.

> **Key Insight:** Op fusion is one of the most compelling arguments for retaining explicit CCL operations. The fine-grained signaling protocol -- where individual tensor slices trigger downstream computation as they arrive -- would be extremely difficult to replicate with a transparent runtime that lacks visibility into the application's computation graph.

---

## 7. The Legacy ERISC Async Datamover Path

The original CCL implementation used dedicated ERISC (Ethernet RISC) kernels running the `erisc_async_datamover` firmware for inter-device data transfer.

### 7.1 `EriscDatamoverBuilder`

The `EriscDatamoverBuilder` (defined in `ccl_host_datastructures.hpp`) constructs the kernel arguments for the legacy EDM path:

```cpp
class EriscDatamoverBuilder {
    std::vector<ChannelBufferSpec> active_channels;
    std::vector<uint32_t> local_semaphore_addresses;
    std::vector<uint32_t> local_buffer_addresses;
    uint32_t eth_buffer_size_bytes;
    uint32_t handshake_addr;
    uint32_t num_senders;
    uint32_t num_receivers;

    ChannelBufferInterface add_sender_channel(
        uint32_t worker_semaphore_id,
        uint32_t num_eth_messages_to_forward,
        const std::vector<WorkerXY>& worker_coords, ...);

    ChannelBufferInterface add_receiver_channel(
        uint32_t worker_semaphore_id,
        uint32_t num_eth_messages_to_forward,
        const std::vector<WorkerXY>& worker_coords, ...);

    std::vector<uint32_t> get_compile_time_args(uint32_t risc_id) const;
    std::vector<uint32_t> get_runtime_args() const;
};
```

Each ERISC core runs a kernel with:
- **Sender channels:** Accept data from local Tensix worker cores and forward it over Ethernet
- **Receiver channels:** Accept data from Ethernet and deliver it to local Tensix worker cores
- **Buffer management:** L1 buffers on the ERISC core, with configurable size and sharing mode

### 7.2 `EriscDatamoverConfig`

The `EriscDatamoverConfig` computes the L1 layout for the EDM kernel:

```cpp
struct EriscDatamoverConfig {
    std::size_t total_l1_buffer_space;     // From hal::get_erisc_l1_unreserved_size()
    std::size_t usable_l1_base_address;    // From hal::get_erisc_l1_unreserved_base()
    static constexpr std::size_t semaphore_size = 32;
    static constexpr std::size_t handshake_location_size = 16;  // Ethernet word
    static constexpr std::size_t eth_word_size_bytes = 16;

    uint32_t get_edm_handshake_address() const;
    uint32_t get_semaphores_base_address(size_t num_channels) const;
    uint32_t get_buffers_base_address(size_t num_channels) const;
    uint32_t compute_buffer_size(size_t num_channels, ...) const;
};
```

### 7.3 `ChannelBuffer` State Machine

On the device side, the legacy `ChannelBuffer` class (in `erisc_async_datamover.hpp`) manages the state machine for each EDM channel:

```
Sender States:
  SENDER_SIGNALING_WORKER -> SENDER_WAITING_FOR_WORKER
    -> SENDER_READY_FOR_ETH_TRANSFER -> SENDER_WAITING_FOR_ETH -> DONE

Receiver States:
  RECEIVER_WAITING_FOR_ETH -> RECEIVER_SIGNALING_WORKER
    -> RECEIVER_WAITING_FOR_WORKER -> DONE
```

### 7.4 Buffer Sharing and Termination Modes

The legacy EDM supports configurable buffer sharing between channels:

```cpp
enum EriscDataMoverBufferSharingMode : uint32_t {
    NOT_SHARED = 0,        // Each channel has its own buffer
    ROUND_ROBIN = 1,       // Channels share buffers in round-robin
    SHARED = 2,            // All channels share a single buffer pool
    ROUND_ROBIN_AND_SHARED = 3
};

enum EriscDataMoverTerminationMode : uint32_t {
    MESSAGE_COUNT_REACHED = 0,  // Terminate after N messages
    WORKER_INITIATED = 1        // Terminate when worker signals
};
```

### 7.5 Worker-to-EDM Device Adapters

On the device side, workers communicate with ERISC channels through adapter classes:

**`WorkerToEdmSender`**: Writes data from a circular buffer to the ERISC sender channel. Uses semaphore-based flow control: the worker waits for the ERISC to signal a free buffer slot, writes data via NOC write, then signals the ERISC that data is available.

**`WorkerToEdmReader`**: Reads data from the ERISC receiver channel into a circular buffer. Uses semaphore-based flow control: the worker waits for the ERISC to signal that data has arrived, reads via NOC read, then signals the ERISC that the buffer slot is free.

Both adapters support double-buffering (`num_buffers_per_channel`) for overlapping transfers.

---

## 8. The Newer Fabric-Based CCL Path

The codebase is transitioning from the legacy per-CCL-operation ERISC kernels to using the **TT-Fabric** infrastructure, where persistent fabric router kernels run on ERISC cores and CCL worker kernels connect to them through a standardized API.

### 8.1 `append_fabric_connection_rt_args`

```cpp
tt::tt_fabric::append_fabric_connection_rt_args(
    device_fabric_node_id, target_fabric_node_id, link,
    program, {worker_core}, rt_args);
```

This function appends the runtime arguments needed for a worker core to establish a connection to the fabric router. The fabric router is a persistent kernel running on ERISC cores that handles packet routing -- the worker kernel simply sends packets to its local fabric router, which handles multi-hop forwarding.

### 8.2 `RoutingPlaneConnectionManager`

The `RoutingPlaneConnectionManager` provides a higher-level abstraction for managing up to 4 simultaneous routing-plane connections per worker core. It supports per-connection destination tracking with device ID and mesh ID for 2D fabric routing.

### 8.3 Key Differences from Legacy Path

| Aspect | Legacy ERISC Async Path | Fabric-Based Path |
|--------|------------------------|-------------------|
| **ERISC kernel** | Per-CCL-operation EDM kernel launched/torn down with each op | Persistent fabric router kernel always running |
| **Connection setup** | Handshake between sender/receiver ERISC pairs | Worker connects to local fabric router via L1 tables |
| **Routing** | Direct Ethernet link between adjacent devices | Multi-hop source routing via route buffer in packet header |
| **Buffer management** | CCL code manages ERISC L1 buffers directly | Fabric manages buffers; worker uses flow-control credits |
| **Multi-hop** | Chained EDM kernels (each hop has separate sender/receiver) | Single packet with route buffer, forwarded by intermediate routers |
| **Topology support** | Ring and Linear (1D only) | 1D and 2D (Mesh, Torus) |
| **MUX support** | Not available | Tensix-based MUX aggregates traffic from multiple workers |
| **Used by** | Older all-gather/reduce-scatter implementations | Newer operations (all-broadcast, broadcast, all-to-all) |

### 8.4 MUX Connection Support

For operations that require aggregating traffic from multiple worker cores through a single fabric router channel, the CCL code supports Tensix MUX connections:

```cpp
void fabric_mux_connection_ct_args(
    uint32_t num_workers_per_direction,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& worker_ct_args);

void fabric_mux_connection_rt_args(
    bool mux_connection_valid,
    bool is_termination_master,
    tt::tt_fabric::FabricMuxChannelType channel_type,
    const CoreCoord& mux_virtual_core,
    uint32_t worker_id,
    const CoreCoord& worker_logical_core,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    Program& program,
    CoreCoord termination_master_virtual_core,
    std::vector<uint32_t>& worker_rt_args, ...);
```

The MUX kernel (a Tensix core running `tt_fabric_mux.cpp`) aggregates packets from multiple worker cores into a single stream for the fabric router, reducing the number of worker-to-router connections needed and improving Ethernet link utilization.

---

## 9. The `CCLOpConfig` Helper

The `CCLOpConfig` class provides a unified configuration interface used across CCL operations:

```cpp
class CCLOpConfig {
public:
    CCLOpConfig(std::vector<Tensor>& input_tensors,
                const std::vector<Tensor>& output_tensors,
                Topology topology);

    uint32_t get_page_size() const;
    Tile get_tile() const;
    Topology get_topology() const;
    bool is_input_sharded() const;
    bool is_output_sharded() const;
    const Tensor& get_input_tensor(size_t i) const;
    const Tensor& get_output_tensor(size_t i) const;
    std::map<std::string, std::string> emit_worker_defines() const;
};
```

The `emit_worker_defines` method generates `#define` strings that are injected into worker kernel compilation, allowing compile-time specialization based on tensor properties (sharded vs. interleaved, tile vs. row-major).

---

## 10. The Complete Orchestration Stack

Combining all the components, here is the full host-side orchestration flow for a single CCL invocation:

```
1. TOPOLOGY RESOLUTION
   - Query fabric configuration
   - Determine boundary mode (WRAP vs NONE) via get_boundary_mode()
   - Downgrade topology if needed (Ring->Linear, Torus->Mesh)
   - Convert 2D to 1D if needed

2. MULTI-AXIS DECOMPOSITION (if needed)
   - Split 2D mesh operations into sequential 1D operations

3. DEVICE OPERATION SETUP
   - Compute output tensor specs
   - Allocate output tensors on mesh device
   - Compute program hash

4. MESH WORKLOAD CREATION
   - Allocate global semaphores (2 for sync + 1 for barrier)
   - Synchronize mesh (host barrier)
   - For each device in the mesh:
     a. Discover forward/backward neighbors
     b. Compute device index in the ring/line
     c. Select worker cores
     d. Create kernel program:
        - Allocate circular buffers
        - Set up fabric connections (legacy EDM or fabric API)
        - Generate command streams (tensor slices, semaphore operations)
        - Encode everything into runtime arguments
     e. Optionally: configure op fusion signalers

5. PROGRAM DISPATCH
   - Submit MeshWorkload to command queue
   - Device kernels execute command streams
   - Fabric routes packets between chips
   - Semaphores synchronize producer/consumer relationships

6. ON CACHE HIT (subsequent invocations)
   - Override runtime arguments (tensor addresses, semaphore values)
   - Skip program creation entirely
```

Each layer adds complexity and topology awareness. A transparent GSRP model would need to either eliminate or encapsulate layers 1 through 5, replacing the explicit topology discovery, neighbor computation, command stream encoding, and fabric connection management with runtime-managed alternatives.

---

## Key Takeaways

- **Topology is pervasive and deeply embedded:** Every aspect of CCL orchestration -- neighbor discovery, boundary mode, distance computation, direction assignment -- depends on the topology type. The boundary mode logic is surprisingly subtle: whether a ring wraps depends on the tensor's actual device coordinate distribution, not just the requested topology. A transparent model must either internalize this topology reasoning or expose it through a different abstraction.
- **The boundary mode logic handles partial meshes gracefully:** The system always downgrades to the most conservative topology that the actual mesh geometry supports (Ring to Linear when fewer than 3 devices span the axis, Torus to Mesh when coordinates do not cover the full range). This means the same operation code handles submeshes correctly without special-casing.
- **CCL op fusion is a critical performance feature:** The ability to overlap all-gather with matmul (or reduce-scatter with matmul) via fine-grained semaphore signaling is essential for hiding communication latency. The fusion signalers are deeply coupled to both the CCL and compute operation implementations, with the `MatmulFusedOpSignaler` alone carrying 20+ fields. A transparent model must provide equivalent overlap capability.
- **The transition from legacy ERISC to fabric-based paths is ongoing:** Newer operations (all-broadcast, broadcast, all-to-all) use the fabric API directly, while the core all-gather and reduce-scatter still use the legacy `EriscDatamoverBuilder` path. The fabric-based path with its `RoutingPlaneConnectionManager`, persistent router kernels, and MUX support represents the direction of the codebase -- and is itself a step toward the more transparent communication model explored in later chapters.
- **The host-side orchestration represents the largest surface area of complexity** that a transparent model would eliminate. The device-side command stream (File 03) and kernel implementation are relatively compact; it is the host-side topology resolution, neighbor discovery, core allocation, semaphore management, and program construction that constitute the bulk of the explicit programming burden.

## Source Code References

- `ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp` -- `SenderReceiverConfig`, `RingTopology`, `LineTopology`, `choose_worker_cores`, `get_physical_neighbor_from_physical_coord`, `get_usable_topology`, `get_topological_dimension`, `get_linearized_index_from_physical_coord`, forward/backward configuration functions, `fabric_mux_connection_ct_args`, `fabric_mux_connection_rt_args`
- `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp` -- Implementation of topology resolution, boundary mode computation, neighbor discovery, worker core selection, `SenderReceiverConfig` construction
- `ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.hpp` -- `AllGatherFusedOpSignaler`, `StridedAllGatherFusedOpSignaler`, `ReduceScatterFusedOpSignaler`, `MatmulFusedOpSignaler`, `MinimalMatmulFusedOpSignaler`, `FusedOpSignalerMode`
- `ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp` -- Fusion signaler initialization and runtime argument push implementations
- `ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp` -- `EriscDatamoverConfig`, `EriscDatamoverBuilder`, `CCLOpConfig`, `CclOpTensorConfig`
- `ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_async_datamover.hpp` -- Legacy ERISC datamover `ChannelBuffer` template and state machine
- `ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp` -- Legacy EDM kernel entry point
- `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp` -- `WorkerToEdmSender`, `WorkerToEdmReader` (device-side legacy EDM adapters)
- `ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp` -- `EriscDataMoverBufferSharingMode`, `EriscDataMoverTerminationMode`, `EriscDataMoverWorkerSignal`
- `ttnn/cpp/ttnn/operations/ccl/all_broadcast/device/all_broadcast_program_factory.cpp` -- Example of newer `append_routing_plane_connection_manager_rt_args` usage

---

**Next:** [Chapter 2 -- Per-Chip Memory Architecture and the NOC](../ch02_memory_and_noc/index.md)
