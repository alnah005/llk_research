# 05 -- Fabric Topology and Mesh Graph

## Context

File 04 covered the control plane and routing table generation -- the host-side infrastructure that computes routes and distributes them. This file examines the topology abstractions that the control plane operates on: the `MeshGraph` (the authoritative representation of the physical topology), the `TopologyMapper` and `TopologySolver` (mapping logical fabric nodes to physical chips), the `FabricContext` (immutable topology queries and packet specifications), and the Tensix-based fabric extensions (MUX and UDM modes). The reader should understand how the system represents multi-chip and multi-mesh topologies, how the Mesh Graph Descriptor (MGD) protobuf schema defines connectivity, how Tensix cores extend the fabric router capacity, and what these topology abstractions imply for GSRP's global address space geometry.

---

## 1. `MeshGraph` -- Topology Representation

The `MeshGraph` is the canonical representation of the fabric topology. It models how chips are connected within meshes (intra-mesh) and how meshes are connected to each other (inter-mesh).

### 1.1 Core Data Structures

```cpp
// tt_metal/fabric/mesh_graph.cpp
class MeshGraph {
    IntraMeshConnectivity intra_mesh_connectivity_;
    InterMeshConnectivity inter_mesh_connectivity_;
    // ...
};
```

Where:
- `IntraMeshConnectivity = vector<vector<unordered_map<ChipId, RouterEdge>>>` -- indexed as `[mesh_id][src_chip_id][dst_chip_id] -> RouterEdge`
- `InterMeshConnectivity = vector<vector<unordered_map<MeshId, RouterEdge>>>` -- indexed as `[mesh_id][src_chip_id][dst_mesh_id] -> RouterEdge`

Each `RouterEdge` carries:

```cpp
struct RouterEdge {
    RoutingDirection port_direction;       // N, S, E, W, Z
    std::vector<ChipId> connected_chip_ids; // Remote chip(s) at the other end
    uint32_t weight;                        // For load balancing (currently unused)
};
```

| Structure | Description |
|-----------|-------------|
| `IntraMeshConnectivity` | `[mesh_id][chip_id] -> map<dest_chip_id, RouterEdge>` -- adjacency within a mesh |
| `InterMeshConnectivity` | `[mesh_id][chip_id] -> map<dest_mesh_id, RouterEdge>` -- adjacency between meshes |
| `FabricNodeId` | `(MeshId mesh_id, uint32_t chip_id)` -- unique identifier for any chip |
| `MeshId` | `StrongType<uint32_t>` -- typed mesh identifier |
| `MeshCoordinate` | 2D coordinate within a mesh grid |

### 1.2 Coordinate System

Chips within a mesh are arranged in a 2D grid and can be addressed by coordinate or by linear chip ID:

```cpp
MeshCoordinate coord = mesh_graph.chip_to_coordinate(mesh_id, chip_id);
// coord[0] = row (North-South axis), coord[1] = column (East-West axis)
ChipId chip_id = mesh_graph.coordinate_to_chip(mesh_id, coord);
```

This coordinate system is used by the routing table generator for dimension-ordered routing (File 04).

### 1.3 Connectivity Direction Model

Each edge in the connectivity graph has a `RoutingDirection`:

```
N (North)   -- mesh_coord[0] decreasing
S (South)   -- mesh_coord[0] increasing
E (East)    -- mesh_coord[1] increasing
W (West)    -- mesh_coord[1] decreasing
C (Center)  -- self (no movement)
Z            -- inter-shelf / inter-tray
NONE        -- invalid
```

### 1.4 Connectivity Construction

Connections are added to the MeshGraph via `add_to_connectivity`:

```cpp
// tt_metal/fabric/mesh_graph.cpp
void MeshGraph::add_to_connectivity(
    MeshId src_mesh_id, ChipId src_chip_id,
    MeshId dest_mesh_id, ChipId dest_chip_id,
    RoutingDirection port_direction) {
    if (src_mesh_id != dest_mesh_id) {
        // Inter-mesh: add to inter_mesh_connectivity_
        inter_mesh_connectivity_[src_mesh][src_chip][dest_mesh].port_direction = direction;
        inter_mesh_connectivity_[src_mesh][src_chip][dest_mesh]
            .connected_chip_ids.push_back(dest_chip);
    } else {
        // Intra-mesh: add to intra_mesh_connectivity_
        intra_mesh_connectivity_[src_mesh][src_chip][dest_chip].port_direction = direction;
    }
}
```

---

## 2. Mesh Graph Descriptor (MGD) and Cluster Types

### 2.1 MGD Format

Topologies are defined by **Mesh Graph Descriptor** files in Protocol Buffers (textproto) format. The MGD defines:

- **Meshes:** Each mesh has a shape (rows x columns) and a list of chips
- **Intra-mesh connections:** Which chips are neighbors within a mesh, and via which direction
- **Inter-mesh connections:** Which chips bridge between meshes
- **Per-chip Ethernet port assignments:** Which physical Ethernet channels correspond to which logical directions
- **Host rank mappings:** Which MPI rank owns which mesh

The textproto format replaced an older YAML format (MGD 1.0). The MeshGraph constructor enforces the `.textproto` extension and refuses YAML files.

### 2.2 Supported Cluster Types

The `MeshGraph` selects MGD files based on cluster type and fabric type:

```cpp
// tt_metal/fabric/mesh_graph.cpp (excerpt)
{FabricType::MESH, {
    {ClusterType::N150,   "n150_mesh_graph_descriptor.textproto"},
    {ClusterType::N300,   "n300_mesh_graph_descriptor.textproto"},
    {ClusterType::T3K,    "t3k_mesh_graph_descriptor.textproto"},
    {ClusterType::GALAXY, "single_galaxy_mesh_graph_descriptor.textproto"},
    {ClusterType::TG,     "tg_mesh_graph_descriptor.textproto"},
    {ClusterType::P100,   "p100_mesh_graph_descriptor.textproto"},
    {ClusterType::P150,   "p150_mesh_graph_descriptor.textproto"},
    {ClusterType::P150_X2, "p150_x2_mesh_graph_descriptor.textproto"},
    {ClusterType::P150_X4, "p150_x4_mesh_graph_descriptor.textproto"},
    {ClusterType::P150_X8, "p150_x8_mesh_graph_descriptor.textproto"},
    {ClusterType::P300,   "p300_mesh_graph_descriptor.textproto"},
    {ClusterType::BLACKHOLE_GALAXY, "single_bh_galaxy_mesh_graph_descriptor.textproto"},
    // ... simulators, multi-board configs ...
}},
{FabricType::TORUS_X, {
    {ClusterType::GALAXY, "single_galaxy_torus_x_graph_descriptor.textproto"},
    {ClusterType::BLACKHOLE_GALAXY, "single_bh_galaxy_torus_x_graph_descriptor.textproto"},
}},
// TORUS_Y, TORUS_XY variants also defined
```

### 2.3 Constants and Scale Limits

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
static constexpr uint32_t MAX_MESH_SIZE = 256;   // Max chips per mesh
static constexpr uint32_t MAX_NUM_MESHES = 1024;  // Max meshes in system
static constexpr uint32_t PACKET_WORD_SIZE_BYTES = 16;
static constexpr uint32_t CLIENT_INTERFACE_SIZE = 3280;
```

These constants define the theoretical scale of the fabric: a system can have up to $256 \times 1024 = 262{,}144$ chips. In practice, current deployments are far smaller (32 chips for a Galaxy, 8 chips for a T3K).

---

## 3. `TopologyMapper` and `TopologySolver`

### 3.1 TopologyMapper

The `TopologyMapper` bridges the logical topology (MeshGraph) and the physical hardware (cluster descriptor). It:

1. Discovers which physical chips are available
2. Maps logical `FabricNodeId` to physical `ChipId`
3. Validates that physical Ethernet connections match the expected topology
4. Handles multi-host scenarios via `DistributedContext`

### 3.2 Identity Encoding for Multi-Host

In MPI-based multi-host deployments, the topology mapper exchanges identity information between hosts:

```cpp
// tt_metal/fabric/topology_mapper.cpp
// 64-bit encoding: mpi_rank(16) | mesh_id(16) | host_rank(32)
std::uint64_t encode_mpi_rank_mesh_id_and_rank(
    int mpi_rank, MeshId mesh_id, MeshHostRankId host_rank) {
    return (static_cast<uint64_t>(mpi_rank & 0xFFFF) << 48)
         | (static_cast<uint64_t>(mesh_id.get() & 0xFFFF) << 32)
         | static_cast<uint64_t>(host_rank.get());
}

// FabricNodeId encoding: mesh_id(32) | chip_id(32)
std::uint64_t encode_fabric_node_id(const FabricNodeId& fabric_node_id) {
    return (static_cast<uint64_t>(fabric_node_id.mesh_id.get()) << 32)
         | static_cast<uint64_t>(fabric_node_id.chip_id);
}
```

Each host binds to one or more meshes via `LocalMeshBinding`:

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp
struct LocalMeshBinding {
    std::vector<MeshId> mesh_ids;  // Meshes owned by this host
    MeshHostRankId host_rank;
};
```

### 3.3 TopologySolver

The `TopologySolver` resolves the constraint satisfaction problem of mapping logical nodes to physical chips while respecting:

- **Physical Ethernet link availability:** The expected links from the MGD must exist in hardware
- **Corner pinning constraints:** For Galaxy systems (see Section 3.4)
- **Multi-host rank assignments:** Which MPI rank controls which physical chips
- **Inter-mesh bridge chip requirements:** Bridge chips must have Ethernet links to the target mesh
- **UBB (Universal Backplane Board) identification:** Uses bus IDs to determine physical tray placement

```cpp
// tt_metal/fabric/control_plane.cpp
UbbId get_ubb_id(tt::umd::Cluster& cluster, ChipId chip_id) {
    auto* cluster_desc = cluster.get_cluster_description();
    const auto& tray_bus_ids = ubb_bus_ids.at(cluster_desc->get_arch());
    const auto bus_id = get_bus_id(cluster, chip_id);
    // Map bus_id to (tray_id, asic_id) pair
}
```

### 3.4 Galaxy-Specific Corner Pinning

For Galaxy systems (Wormhole-based 32-chip clusters), the control plane applies special ASIC position pinning to ensure QSFP links align with mesh corner nodes:

```
* o o *    <- Corners pinned
o o o o
o o o o
o o o o
o o o o
o o o o
o o o o
* o o *    <- Corners pinned
```

This optimization in `get_galaxy_fixed_asic_position_pinnings()` prevents the topology solver from placing corner nodes in the middle of the physical layout, which would bisect devices and reduce Ethernet link utilization.

---

## 4. `FabricContext` -- Immutable Topology Query Interface

The `FabricContext` is the primary query interface for fabric topology and configuration. It is constructed from a `ControlPlane` and remains immutable after initialization.

### 4.1 Design Split

```
FabricContext (immutable after init):
  - Topology queries (is_wrap_around_mesh, is_2D_routing_enabled)
  - Packet specifications (header size, max payload, channel buffer size)
  - Deadlock avoidance flags
  - Dynamic header sizing

FabricBuilderContext (mutable, lazy-initialized):
  - EDM configs
  - Per-device state
  - Tensix config
```

The split design ensures that topology queries (FabricContext) are immutable and thread-safe, while build-time mutations (FabricBuilderContext) are isolated:

```cpp
// tt_metal/fabric/fabric_context.hpp
FabricBuilderContext& get_builder_context();
const FabricBuilderContext& get_builder_context() const;
```

### 4.2 Key Query Methods

```cpp
// tt_metal/fabric/fabric_context.hpp
class FabricContext {
public:
    // Topology queries
    bool is_wrap_around_mesh(MeshId mesh_id) const;
    tt::tt_fabric::Topology get_fabric_topology() const;
    bool is_2D_routing_enabled() const;
    bool is_bubble_flow_control_enabled() const;
    bool need_deadlock_avoidance_support(eth_chan_directions direction) const;
    bool is_ubb_galaxy() const;

    // Packet specifications
    size_t get_fabric_packet_header_size_bytes() const;
    size_t get_fabric_max_payload_size_bytes() const;
    size_t get_fabric_channel_buffer_size_bytes() const;

    // Dynamic header sizing
    uint32_t get_1d_pkt_hdr_extension_words() const;  // 1D mode only
    uint32_t get_2d_pkt_hdr_route_buffer_size() const; // 2D mode only

    // Fabric kernel defines for compilation
    std::map<std::string, std::string> get_fabric_kernel_defines(
        const ControlPlane& control_plane) const;
};
```

### 4.3 Maximum Hop Limits

The `FabricContext` enforces hard limits on the number of hops based on L1 memory map constraints:

```cpp
// tt_metal/fabric/fabric_context.hpp
struct Limits {
    // 1D: ROUTING_PATH_SIZE_1D = 1024 bytes / 16 bytes per entry = 64 chips max
    static constexpr uint32_t MAX_1D_HOPS = 63;

    // 2D: Route buffer capped at 67 bytes (fits in 128B header)
    static constexpr uint32_t MAX_2D_ROUTE_BUFFER_SIZE = 67;
    static constexpr uint32_t MAX_2D_HOPS = MAX_2D_ROUTE_BUFFER_SIZE;  // 67
};
```

These are hard constraints: exceeding them requires memory map updates, which would reduce available user L1 space.

### 4.4 Dynamic Header Sizing

Rather than using a fixed header size, the `FabricContext` selects the smallest header that accommodates the actual topology:

**1D routing:** Extension words are computed from the maximum hop count:

```
Hops per word: 16  (each 32-bit word holds 16 two-bit hop fields)
  0-16 hops: 0 extension words -> 48B header
 17-32 hops: 1 extension word  -> 64B header
 33-48 hops: 2 extension words -> 64B header
 49-64 hops: 3 extension words -> 64B header
```

**2D routing:** Route buffer tiers are selected to minimize header bloat:

```cpp
// tt_metal/fabric/fabric_context.hpp
static constexpr Routing2DBufferTier ROUTING_2D_BUFFER_TIERS[] = {
    {35, 35},   // 96B header  (61B base + 35B route buffer)
    {51, 51},   // 112B header (61B base + 51B route buffer)
    {67, 67}    // 128B header (61B base + 67B route buffer)
};
```

The 80B tier (19-byte buffer) was disabled because it destabilized benchmarks for 8x4 mesh configurations. The `FabricContext` automatically selects the smallest tier that accommodates the system's actual hop count.

---

## 5. Fabric Type and Topology

The `FabricType` enum describes the physical connectivity pattern of the mesh:

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp
enum class FabricType {
    MESH     = 1 << 0,   // Standard 2D mesh (no wrap-around)
    TORUS_X  = 1 << 1,   // Wrap-around along columns (mesh_coord[1])
    TORUS_Y  = 1 << 2,   // Wrap-around along rows (mesh_coord[0])
    TORUS_XY = TORUS_X | TORUS_Y,  // Full torus
};
```

The `FabricConfig` enum (File 04) maps to `FabricType` as follows:

| FabricConfig | FabricType | Deadlock Avoidance |
|-------------|------------|-------------------|
| `FABRIC_2D` | `MESH` | None needed (acyclic) |
| `FABRIC_2D_TORUS_X` | `TORUS_X` | X-axis dateline |
| `FABRIC_2D_TORUS_Y` | `TORUS_Y` | Y-axis dateline |
| `FABRIC_2D_TORUS_XY` | `TORUS_XY` | Both axes |

The `FabricContext::is_wrap_around_mesh()` method checks whether a specific mesh actually has torus connections by examining the physical Ethernet link assignments.

---

## 6. Physical System Descriptor and Physical Grouping

For large-scale deployments, the system uses a **Physical System Descriptor** that describes the physical hardware:

- **ASIC positions:** Which physical tray and location each chip occupies
- **Inter-ASIC connections:** Which physical Ethernet ports connect which ASICs
- **Host bindings:** Which MPI rank controls which physical chips

The `PhysicalSystemDescriptor` works with the `TopologyMapper` to resolve the mapping between the logical topology (from the MGD) and the physical hardware layout. This is particularly important for Galaxy systems with UBB (Universal Base Board) configurations, where the physical wiring follows specific patterns (e.g., corner pinning, QSFP port alignment).

The **Physical Grouping Descriptor** extends this for hierarchical resource specification -- grouping chips into trays, trays into racks, and racks into systems. This hierarchy enables the control plane to reason about physical locality when making routing decisions in large deployments.

---

## 7. Tensix-Based Fabric Extensions

The fabric can be extended beyond Ethernet cores to include Tensix cores running special kernels. Two modes are supported:

### 7.1 MUX Mode

In MUX mode, a Tensix core runs a multiplexer kernel (`tt_fabric_mux.cpp`) that aggregates traffic from multiple worker cores into a single stream destined for an ERISC fabric router:

```
Worker 0 --+
Worker 1 --+--> Tensix MUX Core --> ERISC Router --> Ethernet Link
Worker 2 --+
Worker 3 --+
```

Without MUX, each worker needs a direct connection to an ERISC sender channel 0, and only one worker can connect at a time (see File 02). With MUX, the Tensix core handles the multiplexing, allowing multiple workers to share a single ERISC channel.

### 7.2 UDM Mode (Unified Data Movement)

UDM extends MUX by adding relay kernels on additional Tensix cores:

```cpp
// tt_metal/fabric/fabric_tensix_builder.hpp
enum class FabricTensixCoreType : uint32_t {
    MUX   = 0,   // BRISC - runs MUX kernel
    RELAY = 1    // NCRISC - runs Relay kernel (UDM mode only)
};
```

In UDM mode, both MUX and RELAY kernels run as Tensix extensions:

- **MUX kernel (BRISC):** Aggregates worker traffic into the fabric
- **RELAY kernel (NCRISC):** Relays packets between fabric routers, extending the routing graph through Tensix cores and enabling cross-chip reads via the `UDMHybridMeshPacketHeader` (File 03)

In UDM mode, Tensix cores effectively become fabric nodes, enabling directions that lack physical Ethernet links. For example, a chip with no NORTH Ethernet link can route NORTH traffic through a relay Tensix core connected to an adjacent router:

```cpp
// tt_metal/fabric/fabric_tensix_builder.hpp
// Check if any directions are missing (no physical Ethernet channel)
bool has_missing_directions(ChipId device_id) const;
std::set<std::pair<routing_plane_id_t, eth_chan_directions>>
    get_missing_directions(ChipId device_id) const;
```

### 7.3 `FabricTensixDatamoverConfig`

The `FabricTensixDatamoverConfig` class manages the configuration of Tensix fabric extensions:

```cpp
// tt_metal/fabric/fabric_tensix_builder.hpp
class FabricTensixDatamoverConfig {
public:
    size_t get_num_configs_per_core() const;
    size_t get_num_riscs_per_core() const;
    size_t get_num_buffers_per_channel() const;
    size_t get_buffer_size_bytes_full_size_channel() const;
    size_t get_base_l1_address(FabricTensixCoreType core_id) const;

    // Get the MUX core assigned to a specific Ethernet channel
    CoreCoord get_core_for_channel(ChipId device_id, uint32_t eth_chan_id) const;
    FabricTensixCoreType get_core_id_for_channel(ChipId device_id, uint32_t eth_chan_id) const;

    // UDM mode: Get assigned tensix core for any worker
    struct WorkerTensixInfo {
        CoreCoord tensix_core;
        uint32_t channel_index;
    };
    WorkerTensixInfo get_worker_tensix_info(ChipId device_id, const CoreCoord& worker) const;
};
```

### 7.4 Tensix Configuration Selection

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp
enum class FabricTensixConfig : uint32_t {
    DISABLED = 0,  // No tensix extension
    MUX      = 1,  // MUX kernel only (traffic aggregation)
    UDM      = 2,  // MUX + Relay kernels
};
```

MUX mode is the default for most production workloads. UDM mode is required for cross-chip read operations and is currently experimental. Each MUX core consumes one Tensix core per direction per routing plane; for a 4-direction 2D mesh with 2 routing planes, that is 8 Tensix cores per chip dedicated to fabric -- a significant resource cost.

---

## 8. GSRP Implications

| Aspect | Current State | GSRP Relevance |
|--------|--------------|----------------|
| MeshGraph as topology model | Complete connectivity information for all cluster types | Provides the graph GSRP needs to determine reachability and hop distance for address space design |
| FabricContext immutable queries | Thread-safe topology properties and packet specs | Enables concurrent GSRP address resolution without locking |
| Dynamic header sizing | Automatic smallest-header selection per topology | Minimizes per-packet overhead for short-distance GSRP accesses |
| MAX MESH SIZE / MAX NUM MESHES | 256 chips per mesh, 1024 meshes | Defines the ceiling of the global address space |
| MUX mode | Solves single-worker-per-channel bottleneck | Critical for GSRP: multiple workers issuing remote accesses simultaneously |
| UDM relay extension | Tensix cores participate in packet forwarding | Demonstrates the mechanism a GSRP firmware shim would use to inject packets transparently |
| Static topology after init | MeshGraph and FabricContext are immutable post-construction | GSRP in a dynamically reconfigurable fabric would require notification mechanisms |

---

## Key Takeaways

- **The `MeshGraph` is the single source of truth for topology.** It is constructed from a Mesh Graph Descriptor (protobuf) and provides both intra-mesh and inter-mesh connectivity maps via `RouterEdge` structures. All routing table generation, topology solving, and fabric node mapping ultimately derive from this graph.
- **The `TopologyMapper` and `TopologySolver` handle the logical-to-physical mapping problem.** The solver respects Galaxy corner pinning, UBB identification via bus IDs, and multi-host MPI coordination, mapping logical `FabricNodeId` values to physical `ChipId` values.
- **`FabricContext` provides immutable, thread-safe topology queries** including hop limits (63 for 1D, 67 for 2D), packet specifications, and deadlock avoidance requirements. Dynamic header sizing selects the smallest header for the actual topology, reducing per-packet overhead on smaller deployments.
- **Tensix MUX kernels solve the one-worker-per-channel limitation.** By running a multiplexer on a Tensix core (`tt_fabric_mux.cpp`), multiple workers can share a single ERISC channel. UDM mode further extends this with relay kernels that enable cross-chip reads, transforming the fabric from an Ethernet-core-only network into a hybrid Ethernet+Tensix network.
- **The MGD protobuf schema supports multi-mesh, multi-host topologies** from single-chip (N150) through multi-Galaxy deployments (hundreds of chips across multiple hosts). Combined with the `TopologyMapper` and `TopologySolver`, the system handles all configurations through the same topology abstraction.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `tt_metal/fabric/mesh_graph.cpp` | `MeshGraph`, `RouterEdge`, `add_to_connectivity()`, `get_valid_connections()`, cluster-type-to-descriptor mapping |
| `tt_metal/fabric/topology_mapper.cpp` | `TopologyMapper`, `encode_fabric_node_id()`, `decode_fabric_node_id()`, `encode_mpi_rank_mesh_id_and_rank()`, `LocalMeshBinding` |
| `tt_metal/fabric/fabric_context.hpp` | `FabricContext`, `Limits`, `Routing2DBufferTier`, `get_1d_pkt_hdr_extension_words()`, `get_2d_pkt_hdr_route_buffer_size()`, `need_deadlock_avoidance_support()` |
| `tt_metal/fabric/fabric_tensix_builder.hpp` | `FabricTensixCoreType` (MUX/RELAY), `FabricTensixDatamoverConfig`, `WorkerTensixInfo`, `get_missing_directions()` |
| `tt_metal/fabric/impl/kernels/edm_fabric/tt_fabric_mux.cpp` | MUX kernel: Tensix-based traffic aggregation from multiple workers to a single ERISC channel |
| `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp` | `FabricType` (MESH/TORUS), `FabricTensixConfig` (DISABLED/MUX/UDM), `FabricUDMMode`, `FabricManagerMode` |
| `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp` | `FabricNodeId`, `LocalMeshBinding`, `MeshScope`, `PortDescriptor` |
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | `MAX_MESH_SIZE`, `MAX_NUM_MESHES`, `PACKET_WORD_SIZE_BYTES`, `CLIENT_INTERFACE_SIZE` |

---

[**Previous:** `04_control_plane_and_routing_tables.md`](./04_control_plane_and_routing_tables.md) | [**Next:** `06_flow_control_and_deadlock_avoidance.md`](./06_flow_control_and_deadlock_avoidance.md)
