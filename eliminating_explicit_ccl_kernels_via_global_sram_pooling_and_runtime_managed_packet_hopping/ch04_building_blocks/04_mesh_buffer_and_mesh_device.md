# Chapter 4.4 -- MeshBuffer and MeshDevice

## Context

The previous sections in this chapter examined device-side primitives (mesh API, connection managers, sockets) that move data between individual cores and chips. This final section sits at the top of the transparency spectrum, shifting to the **host-side allocation and orchestration layer**: `MeshBuffer`, which allocates a single logical buffer across an entire mesh of devices, and `MeshDevice`, which provides the unified interface to a 2D grid of Tenstorrent chips. Together with `MeshWorkload` (per-coordinate program mapping), `MeshCommandQueue` (dispatch infrastructure), and `DistributedContext` (multi-host coordination), these abstractions form the host-side foundation that a GSRP would build upon for global address space management and coordinated multi-device execution.

Understanding these abstractions is essential because they define the **host-side programming model** that a GSRP runtime would either extend or replace. They also reveal a critical implementation detail: `MeshBuffer::create()` already performs lock-step allocation at a consistent address across all mesh devices, which is a primitive building block for a global address space.

---

## 4.4.1 `MeshBuffer`: Mesh-Wide Buffer Allocation

The `MeshBuffer` class in `tt_metal/api/tt-metalium/mesh_buffer.hpp` represents a buffer that spans multiple devices with a consistent address. It supports two layouts:

```cpp
enum class MeshBufferLayout : uint8_t {
    REPLICATED,
    SHARDED
};

using MeshBufferConfig = std::variant<ReplicatedBufferConfig, ShardedBufferConfig>;
```

### Replicated Layout

Every device in the mesh gets a full copy of the buffer:

```cpp
struct ReplicatedBufferConfig {
    DeviceAddr size = 0;  // Each device gets a buffer of this size
};
```

Write APIs for replicated buffers broadcast the same data to all devices. This is the natural layout for weights or lookup tables that every chip needs.

### Sharded Layout

The buffer is distributed across devices, each receiving a fraction:

```cpp
struct ShardedBufferConfig {
    DeviceAddr global_size = 0;
    Shape2D global_buffer_shape = {0, 0};
    Shape2D shard_shape = {0, 0};
    ShardOrientation shard_orientation = ShardOrientation::ROW_MAJOR;

    uint32_t compute_datum_size_bytes() const;
    std::pair<bool, bool> replicated_dims() const;
    Shape2D physical_shard_shape() const;
};
```

The `global_buffer_shape` and `shard_shape` work together to define a 2D decomposition. For a mesh of shape $R \times C$ with shard shape $h \times w$:

$$
\text{shard per device} = h \times w \times \text{datumSize}
$$

The `shard_orientation` determines whether shards tile across rows first (`ROW_MAJOR`) or columns first (`COL_MAJOR`). The `replicated_dims()` method returns which mesh dimensions see replicated data (e.g., if the shard shape equals the global shape along one axis).

### `DeviceLocalBufferConfig`

Each device's local buffer is further configured with:

```cpp
struct DeviceLocalBufferConfig {
    DeviceAddr page_size = 0;
    BufferType buffer_type = BufferType::DRAM;  // DRAM, L1, SYSTEM_MEMORY, L1_SMALL, TRACE
    BufferShardingArgs sharding_args;
    std::optional<bool> bottom_up;
    std::optional<SubDeviceId> sub_device_id;
};
```

This separates the **inter-device** layout (replicated vs. sharded) from the **intra-device** layout (which memory banks, page size, sharding across banks). A `MeshBuffer` thus represents a two-level allocation hierarchy. For GSRP, L1 buffers would provide the lowest latency for cross-chip access.

### Lock-Step Allocation

The creation API is the most significant design choice for the GSRP vision:

```cpp
class MeshBuffer {
public:
    static std::shared_ptr<MeshBuffer> create(
        const MeshBufferConfig& mesh_buffer_config,
        const DeviceLocalBufferConfig& device_local_config,
        MeshDevice* mesh_device,
        std::optional<DeviceAddr> address = std::nullopt);
};
```

When `address` is not specified, `MeshBuffer::create()` performs **lock-step allocation across all mesh devices at a consistent address**. This means every device in the mesh allocates the buffer at the **same L1 or DRAM address**:

$$
\forall\, d \in \text{mesh devices}: \text{address}(d) = \text{address}_{\text{global}}
$$

This is achieved by allocating on a single "backing" device and then creating non-owning `Buffer` objects at the same address on all other devices. When `address` is provided, the MeshBuffer becomes a non-owning view.

### Internal State Management

The `MeshBuffer` tracks its state through a variant:

```cpp
struct OwnedBufferState {
    std::shared_ptr<Buffer> backing_buffer;  // Single device buffer providing the address
};
struct ExternallyOwnedState {};  // View over an existing address
struct DeallocatedState {};
using MeshBufferState = std::variant<OwnedBufferState, ExternallyOwnedState, DeallocatedState>;
```

- **Owned**: A backing `Buffer` on one device holds the allocation. Deallocation releases that buffer, freeing the address on all devices.
- **Externally owned**: Created as a view over a pre-existing address. No deallocation responsibility.
- **Deallocated**: Terminal state after `deallocate()` or destructor.

### Key `MeshBuffer` Methods

| Method | Purpose |
|--------|---------|
| `address()` | Returns the consistent L1/DRAM address across all devices. |
| `device_local_size()` | Size of the buffer on each individual device. |
| `size()` | Global size (for sharded) or per-device size (for replicated). |
| `global_layout()` | Returns `REPLICATED` or `SHARDED`. |
| `get_device_buffer(coord)` | Returns the `Buffer*` for a specific mesh coordinate. |
| `get_reference_buffer()` | Returns one representative `Buffer*` for querying single-device attributes. |
| `page_size()` | Page size from the device-local config. |
| `num_pages()` | `device_local_size / page_size`. |
| `deallocate()` | Explicitly frees the buffer across all devices. |

## 4.4.2 `MeshDevice`: The Unified Mesh Interface

`MeshDevice` in `tt_metal/api/tt-metalium/mesh_device.hpp` represents a collection of devices arranged in a 2D grid. It implements the `IDevice` interface, meaning it can be used anywhere a single device is expected -- but operations are dispatched across the mesh.

### Creation

```cpp
class MeshDevice : public IDevice, public std::enable_shared_from_this<MeshDevice> {
public:
    static std::shared_ptr<MeshDevice> create(
        const MeshDeviceConfig& config,
        size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
        size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
        size_t num_command_queues = 1,
        const DispatchCoreConfig& dispatch_core_config = DispatchCoreConfig{},
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
        size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

    static std::shared_ptr<MeshDevice> create_unit_mesh(int device_id, ...);
    static std::map<int, std::shared_ptr<MeshDevice>> create_unit_meshes(
        const std::vector<int>& device_ids, ...);
};
```

The `create_unit_mesh` factory wraps a single device as a 1x1 mesh, enabling uniform APIs across single-device and multi-device configurations.

### Topology and Shape

```cpp
const MeshShape& shape() const;
size_t num_devices() const;
size_t num_rows() const;
size_t num_cols() const;
```

The mesh shape is an N-dimensional coordinate space (though most current usage is 2D). The `get_view()` method returns a `MeshDeviceView` that provides physical-to-logical coordinate mapping.

### System Mesh ID

```cpp
uint32_t get_system_mesh_id() const;
```

For multi-mesh (distributed) topologies, this returns which logical mesh this `MeshDevice` belongs to. It defaults to 0 for single-process workloads. For distributed multi-mesh workloads, it corresponds to the Mesh Graph Descriptor's mesh ID. This ID maps directly to the `dst_mesh_id` field used throughout the fabric APIs.

### Submesh Creation

```cpp
std::shared_ptr<MeshDevice> create_submesh(
    const MeshShape& submesh_shape,
    const std::optional<MeshCoordinate>& offset = std::nullopt);

std::vector<std::shared_ptr<MeshDevice>> create_submeshes(const MeshShape& submesh_shape);
```

Submeshes allow partitioning a physical mesh into smaller logical meshes. This is essential for pipeline parallelism and tensor parallelism patterns where different stages use different subsets of the mesh. The `create_submeshes` method tiles the parent mesh with non-overlapping submeshes of the specified shape.

The parent-child relationship is tracked:

```cpp
bool is_parent_mesh() const;
const std::shared_ptr<MeshDevice>& get_parent_mesh() const;
std::vector<std::shared_ptr<MeshDevice>> get_submeshes() const;
```

The `quiesce_devices()` method provides a barrier across all submeshes derived from a parent, ensuring safe handoff between overlapping submesh usage patterns.

### Reshape

```cpp
void reshape(const MeshShape& new_shape);
```

Reshaping changes the logical-to-physical mapping without changing which devices are in the mesh. Rules:
- Volume must be preserved (same number of devices).
- Line-to-line reshaping (1xN to Nx1) is always possible.
- Grid-to-grid reshaping requires physical connectivity in the new layout.

### Device Access and Fabric Mapping

```cpp
std::vector<IDevice*> get_devices() const;  // Row-major order
IDevice* get_device(size_t row_idx, size_t col_idx) const;
tt_fabric::FabricNodeId get_fabric_node_id(const MeshCoordinate& coord) const;
```

The `get_fabric_node_id` method maps a mesh coordinate to a `FabricNodeId`, which is used by all fabric connection APIs. This is the bridge between the logical mesh coordinate system and the physical fabric routing system.

## 4.4.3 `MeshWorkload`: Per-Coordinate Program Mapping

`MeshWorkload` in `tt_metal/api/tt-metalium/mesh_workload.hpp` maps programs to regions of the mesh:

```cpp
class MeshWorkload {
public:
    void add_program(const MeshCoordinateRange& device_range, Program&& program);
    std::unordered_map<MeshCoordinateRange, Program>& get_programs();
};
```

A `MeshWorkload` supports two paradigms:

1. **Single Program Multi Device (SPMD):** One program mapped to the entire mesh coordinate range.
2. **Multi Program Multi Device (MPMD):** Different programs mapped to different coordinate ranges.

Design constraints:

- **Non-overlapping ranges:** Device ranges for different programs must not share devices. Each device coordinate maps to at most one program within a workload.
- **Partial coverage allowed:** Not all devices need a program -- some may be idle during a workload.

This enables heterogeneous computation across the mesh (e.g., different pipeline stages on different devices). Execution is dispatched through the mesh command queue:

```cpp
void EnqueueMeshWorkload(MeshCommandQueue& mesh_cq, MeshWorkload& mesh_workload, bool blocking);
```

## 4.4.4 `MeshCommandQueue`: Dispatch Infrastructure

`MeshCommandQueue` in `tt_metal/api/tt-metalium/mesh_command_queue.hpp` is the main dispatch interface for mesh-level operations:

```cpp
class MeshCommandQueue {
public:
    // Workload dispatch
    virtual void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) = 0;

    // MeshBuffer Write APIs
    virtual void enqueue_write_mesh_buffer(
        const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) = 0;
    virtual void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer, const void* host_data,
        const MeshCoordinateRange& device_range, bool blocking,
        std::optional<BufferRegion> region = std::nullopt) = 0;
    virtual void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<ShardDataTransfer>& shard_data_transfers, bool blocking) = 0;

    // MeshBuffer Read APIs
    virtual void enqueue_read_mesh_buffer(
        void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) = 0;
    virtual void enqueue_read_shards(
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer, bool blocking) = 0;

    // Synchronization
    virtual MeshEvent enqueue_record_event(...) = 0;
    virtual void enqueue_wait_for_event(const MeshEvent& sync_event) = 0;
    virtual void finish(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;

    // Tracing
    virtual void record_begin(const MeshTraceId& trace_id, ...) = 0;
    virtual void record_end() = 0;
    virtual void enqueue_trace(const MeshTraceId& trace_id, bool blocking) = 0;
};
```

Key features:

- **Shard-aware writes:** `enqueue_write_shards` takes a vector of `ShardDataTransfer` objects, each specifying a mesh coordinate and host data pointer. This allows writing different data to different shards in a single API call.
- **Region-based partial writes:** The `BufferRegion` parameter enables writing to a subset of a buffer's pages.
- **Event-based synchronization:** `MeshEvent` objects enable fine-grained inter-queue and host-device synchronization.
- **Trace capture and replay:** The tracing API records a sequence of operations that can be replayed without host intervention, reducing dispatch overhead for repeated workloads.

The `ShardDataTransfer` class provides a builder-style API:

```cpp
class ShardDataTransfer {
public:
    explicit ShardDataTransfer(const MeshCoordinate& shard_coord);
    ShardDataTransfer& host_data(void* host_data);
    ShardDataTransfer& region(std::optional<BufferRegion> region);
};
```

The command queue internally dispatches to per-device command queues, maintaining ordering guarantees across the mesh.

## 4.4.5 `DistributedContext`: Multi-Host Coordination

The `DistributedContext` class in `tt_metal/api/tt-metalium/distributed_context.hpp` provides an MPI-like interface for multi-host TT-Metal deployments:

```cpp
class DistributedContext {
public:
    static void create(int argc, char** argv);
    static const ContextPtr& get_current_world();

    virtual Rank rank() const = 0;
    virtual Size size() const = 0;

    // Point-to-point
    virtual void send(Span<std::byte> buffer, Rank dest, Tag tag) const = 0;
    virtual void recv(Span<std::byte> buffer, Rank source, Tag tag) const = 0;
    virtual RequestPtr isend(Span<std::byte> buffer, Rank dest, Tag tag) const = 0;
    virtual RequestPtr irecv(Span<std::byte> buffer, Rank source, Tag tag) const = 0;

    // Collectives
    virtual void barrier() const = 0;
    virtual void broadcast(Span<std::byte> buffer, Rank root) const = 0;
    virtual void all_gather(Span<std::byte> send, Span<std::byte> recv) const = 0;
    virtual void all_reduce(Span<std::byte> send, Span<std::byte> recv,
                           ReduceOp op, DType dtype) const = 0;
    virtual void reduce_scatter(Span<std::byte> send, Span<std::byte> recv,
                               ReduceOp op, DType dtype) const = 0;

    // Communicator management
    virtual ContextPtr duplicate() const = 0;
    virtual ContextPtr split(Color color, Key key) const = 0;
    virtual ContextPtr create_sub_context(Span<int> ranks) const = 0;
};
```

Strong types are used throughout:

```cpp
using Rank = StrongType<int, struct RankTag>;
using Tag = StrongType<int, struct TagTag>;
using Size = StrongType<int, struct SizeTag>;
```

The role of `DistributedContext` in the mesh/socket hierarchy:

1. **Rank-to-mesh mapping:** Each MPI rank owns one or more `MeshDevice` instances. The `Rank` type maps to mesh ownership.
2. **Socket handshaking:** `MeshSocket` uses `DistributedContext` for initial cross-host socket negotiation (exchanging fabric node IDs, validating configs).
3. **Collective operations:** While the device-side fabric handles data movement, `DistributedContext` handles host-side coordination for collective setup and synchronization.
4. **Fault tolerance:** The `revoke_and_shrink()` and `is_revoked()` methods support resilient distributed execution, mirroring MPI fault-tolerance patterns. The `supports_fault_tolerance()` query allows callers to check availability.
5. **Communicator splitting:** `split(Color, Key)` mirrors `MPI_Comm_split` semantics, enabling sub-group collective operations.

## 4.4.6 `AnyBuffer`: Unified Buffer Handle

```cpp
class AnyBuffer {
public:
    static AnyBuffer create(const ShardedBufferConfig& config,
                           std::optional<uint64_t> address = std::nullopt);
    static AnyBuffer create(const InterleavedBufferConfig& config,
                           std::optional<uint64_t> address = std::nullopt);

    Buffer* get_buffer() const;
    bool is_mesh_buffer() const;
    std::shared_ptr<MeshBuffer> get_mesh_buffer() const;
};
```

`AnyBuffer` is a type-erased container that can hold either a single-device `Buffer` or a `MeshBuffer`, allowing APIs to be written generically without knowing the allocation scope at compile time. This is a practical building block for a GSRP, where a buffer handle must transparently reference either local or distributed memory.

## 4.4.7 The Global Address Space Building Block

The combination of `MeshBuffer`, `MeshDevice`, and the fabric's addressing model provides the key ingredients for a global address space.

### Consistent Address Allocation

`MeshBuffer::create()` guarantees that the buffer lives at address `A` on every device in the mesh. Therefore, a global address can be decomposed as:

$$
\text{globalAddr} = (\text{meshId}, \text{deviceCoord}, A + \text{offset})
$$

where `A` is the consistent base address and `offset` is the byte offset within the buffer.

### Shard-Based Address Resolution

For sharded `MeshBuffer` with a 1D contiguous page assignment (pages assigned sequentially across devices), the host can compute which device owns a particular page:

$$
\text{deviceIndex}(\text{pageId}) = \left\lfloor \frac{\text{pageId}}{\text{pagesPerShard}} \right\rfloor
$$

The local offset within that device is:

$$
\text{localOffset} = (\text{pageId} \bmod \text{pagesPerShard}) \times \text{pageSize}
$$

For 2D sharded layouts (with `shard_orientation` and 2D `shard_shape`), the page-to-device mapping involves tiling arithmetic that depends on the orientation, the global buffer shape, and the shard shape — the runtime handles this via `ShardDataTransfer`'s address computation rather than the simplified formula above.

These formulas, combined with the consistent base address and `MeshDevice::get_fabric_node_id(coord)`, provide a complete page-to-location mapping across the mesh.

### What is Missing

Despite these building blocks, today's system lacks:

1. **Device-side address resolution.** A Tensix kernel cannot currently look up which device owns page N of a sharded `MeshBuffer`. This lookup is host-side only.

2. **Automatic routing from address.** Even if a kernel knew the destination device, it would still need to manually specify `dst_dev_id` and `dst_mesh_id` in the mesh API calls.

3. **Transparent remote access.** There is no mechanism for a Tensix kernel to issue a load/store to a global address and have the hardware or runtime transparently fetch the data from a remote device.

A GSRP runtime would close these gaps by:
- Pre-loading a device-side lookup table mapping global page IDs to `(mesh_id, dev_id, local_addr)`.
- Integrating this table with the `addrgen` mechanism (Section 4.1) so that the mesh API's send functions receive the correct destination automatically.
- Optionally providing a blocking "global read" primitive that sends a request and waits for the response.

## 4.4.8 The Full Stack at a Glance

Combining all four sections of this chapter, the existing building blocks form a layered stack:

```
Application Layer:    FabricSocket.send(tensor)  /  MeshCommandQueue.enqueue_write
                           |                               |
Socket Layer:         MeshSocket (config, data buffers)
                           |
Device Socket API:    SocketSenderInterface / SocketReceiverInterface
                           |
Mesh API Layer:       fabric_unicast_noc_unicast_write / fabric_multicast_...
                           |
Connection Layer:     RoutingPlaneConnectionManager / WorkerToFabricEdmSender
                           |
EDM / Fabric:         [Covered in Chapter 3]
```

A GSRP would insert a new layer between the Mesh API and the Application layers: a **global address translation and routing layer** that converts a virtual address into the appropriate mesh API call, automatically selecting the connection, computing the route, and managing flow control.

## 4.4.9 Boilerplate Accounting: Mesh-Level Operations

The mesh abstractions significantly reduce per-kernel boilerplate for uniform operations:

| Operation | Per-device approach | Mesh approach |
|-----------|-------------------|---------------|
| Buffer allocation | N calls to `CreateBuffer` | 1 call to `MeshBuffer::create()` |
| Program dispatch | N calls to `EnqueueProgram` | 1 call to `EnqueueMeshWorkload` |
| Buffer write | N calls per device | 1 `MeshCommandQueue` call (shards or broadcasts) |
| Synchronization | N barriers | `quiesce_devices()` |

However, the mesh abstractions do not eliminate fabric-level boilerplate within kernels. A kernel running on a mesh device still needs the full connection setup from Section 4.2 if it wants to communicate with a peer on a different device.

**Estimated host-side lines saved by mesh abstractions:** 10-30 lines per operation, compared with manually iterating over devices. But device-side cross-chip code remains unchanged.

## GSRP Implications

`MeshBuffer` with consistent addressing already provides approximately 80% of a global address space at the host level: every device holds the same buffer at the same L1 address, shard metadata maps pages to devices, and `get_fabric_node_id(coord)` bridges logical coordinates to physical routing. The remaining 20% is making this address space **visible and usable from device-side kernels** without explicit connection management. The gaps are: (1) implicit route computation from global virtual addresses, (2) a read path (currently all push-based), and (3) device-side dynamic allocation (currently host-only via `MeshBuffer::create()`).

## Key Takeaways

- `MeshBuffer::create()` performs lock-step allocation at a consistent address across all mesh devices — the most critical building block for a global address space.
- `MeshWorkload` maps programs to coordinate ranges (SPMD or MPMD), and `MeshCommandQueue` provides shard-aware dispatch with `ShardDataTransfer` for per-shard write control.
- `DistributedContext` extends the mesh model across host boundaries via MPI-like collectives; the shard-based address resolution formulas provide the mathematical foundation for a GSRP page table.

## Source Code References

| File | Key Contents |
|------|-------------|
| `tt_metal/api/tt-metalium/mesh_buffer.hpp` | `MeshBuffer`, `DeviceLocalBufferConfig`, `ReplicatedBufferConfig`, `ShardedBufferConfig`, `MeshBufferLayout`, `AnyBuffer`. |
| `tt_metal/api/tt-metalium/mesh_device.hpp` | `MeshDevice`: creation, shape, submesh, reshape, `get_fabric_node_id`, traces. |
| `tt_metal/api/tt-metalium/mesh_workload.hpp` | `MeshWorkload`: per-coordinate program mapping. |
| `tt_metal/api/tt-metalium/mesh_command_queue.hpp` | `MeshCommandQueue`: shard-aware write/read, `ShardDataTransfer`, event sync, tracing. |
| `tt_metal/api/tt-metalium/distributed_context.hpp` | `DistributedContext`: MPI-like interface for multi-host coordination, rank management, collectives, fault tolerance. |
| `tt_metal/api/tt-metalium/mesh_device_view.hpp` | `MeshDeviceView` for logical-to-physical coordinate mapping. |

---

**Next:** [Chapter 5 -- Comparative Survey: How Other Architectures Handle This Problem](../ch05_comparative_survey/index.md)
