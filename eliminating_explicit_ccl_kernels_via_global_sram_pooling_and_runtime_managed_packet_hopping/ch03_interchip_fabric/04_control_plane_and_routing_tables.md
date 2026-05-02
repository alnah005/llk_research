# 04 -- Control Plane and Routing Tables

## Context

File 03 described the packet header formats and per-hop routing encodings. Those encodings must be populated with correct routes -- but who computes them, and how are they distributed to every core in the system? This file covers the host-side **Control Plane** (`ControlPlane` class) that performs topology discovery, constructs routing tables, and writes them to every chip's L1. It also covers the `RoutingTableGenerator` that implements BFS-based shortest-path computation for intra-mesh routing and exit-node lookup tables for inter-mesh routing. The reader should understand how the `routing_l1_info_t` struct is laid out, how compressed direction tables work, the `FabricConfig` enum that selects routing topology, and how routing tables are written to `MEM_TENSIX_ROUTING_TABLE_BASE` and `MEM_AERISC_ROUTING_TABLE_BASE` on every core. This infrastructure is directly relevant to GSRP because any global address translation layer needs the same routing information that the control plane already computes and distributes.

---

## 1. The `ControlPlane` Class

The `ControlPlane` is the central host-side authority for fabric configuration. It is instantiated during Metal runtime initialization and persists for the lifetime of the application.

### 1.1 Construction

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp
class ControlPlane {
public:
    // Auto-discovery constructor
    explicit ControlPlane(
        const ::tt::Cluster& cluster,
        const ::tt::llrt::RunTimeOptions& rtoptions,
        const ::tt::tt_metal::Hal& hal,
        const tt_metal::distributed::multihost::DistributedContext& distributed_context,
        FabricConfig fabric_config = FabricConfig::DISABLED,
        FabricReliabilityMode fabric_reliability_mode = FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        FabricTensixConfig fabric_tensix_config = FabricTensixConfig::DISABLED,
        FabricUDMMode fabric_udm_mode = FabricUDMMode::DISABLED,
        FabricRouterConfig fabric_router_config = FabricRouterConfig{},
        FabricManagerMode fabric_manager = FabricManagerMode::DEFAULT);

    // Custom mesh graph descriptor constructor
    explicit ControlPlane(
        const ::tt::Cluster& cluster, ...,
        const std::string& mesh_graph_desc_file, ...);

    // Custom MGD + explicit logical-to-physical chip mapping
    explicit ControlPlane(
        const ::tt::Cluster& cluster, ...,
        const std::string& mesh_graph_desc_file,
        const std::map<FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping, ...);
};
```

Three constructors support three deployment scenarios:
1. **Auto-discovery:** The control plane discovers topology from the hardware cluster descriptor.
2. **Custom MGD:** A Mesh Graph Descriptor (protobuf) defines the topology explicitly.
3. **Custom MGD + explicit mapping:** Both the topology and the logical-to-physical chip mapping are provided.

### 1.2 Key Control Plane Operations

| Method | Description |
|--------|-------------|
| `configure_routing_tables_for_fabric_ethernet_channels()` | Converts chip-level routing tables to per-Ethernet-channel tables |
| `write_routing_tables_to_all_chips()` | Writes `routing_l1_info_t` to every Tensix and ERISC core on every chip |
| `write_fabric_telemetry_to_all_chips(fabric_node_id)` | Writes telemetry configuration |
| `get_fabric_node_id_from_physical_chip_id(chip_id)` | Maps physical chip -> `(mesh_id, chip_id)` |
| `get_physical_chip_id_from_fabric_node_id(node)` | Maps `(mesh_id, chip_id)` -> physical chip |

---

## 2. `FabricConfig` -- Routing Topology Selection

The `FabricConfig` enum selects the fabric routing topology:

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp
enum class FabricConfig : uint32_t {
    DISABLED                 = 0,
    FABRIC_1D_NEIGHBOR_EXCHANGE = 1,  // 1D, no forwarding beyond neighbors
    FABRIC_1D                = 2,      // 1D routing, no deadlock avoidance
    FABRIC_1D_RING           = 3,      // 1D routing with dateline deadlock avoidance
    FABRIC_2D                = 4,      // 2D mesh routing
    FABRIC_2D_TORUS_X        = 5,      // 2D + X-axis torus wrap
    FABRIC_2D_TORUS_Y        = 6,      // 2D + Y-axis torus wrap
    FABRIC_2D_TORUS_XY       = 7,      // 2D + both axes torus wrap
    CUSTOM                   = 8,      // Custom topology from MGD
};
```

The choice of `FabricConfig` determines:
- Whether 1D or 2D routing tables are generated
- Whether deadlock avoidance (dateline, bubble flow control) is enabled
- Which packet header type is used (via `ROUTING_MODE` kernel defines)
- How many routing planes are allocated per link direction

### 2.1 Reliability Modes

```cpp
enum class FabricReliabilityMode : uint32_t {
    STRICT_SYSTEM_HEALTH_SETUP_MODE     = 0,  // All links must match MGD
    RELAXED_SYSTEM_HEALTH_SETUP_MODE    = 1,  // Fewer planes OK if links are down
    DYNAMIC_RECONFIGURATION_SETUP_MODE  = 2,  // Runtime reconfiguration (placeholder)
};
```

In `STRICT` mode, the control plane fails initialization if any expected Ethernet link is down. In `RELAXED` mode, it gracefully degrades by reducing the number of routing planes. `DYNAMIC_RECONFIGURATION` is the basis for live link recovery via the `RouterCommand::RETRAIN` sequence documented in File 01.

---

## 3. `RoutingTableGenerator`

The `RoutingTableGenerator` computes all routing tables from the mesh graph topology.

### 3.1 Intra-Mesh Routing: Dimension-Ordered Shortest Path

The intra-mesh routing algorithm uses **row-first (NS then EW) dimension-ordered routing**:

```cpp
// tt_metal/fabric/routing_table_generator.cpp
void RoutingTableGenerator::generate_intramesh_routing_table(
    const IntraMeshConnectivity& intra_mesh_connectivity) {
    for (mesh_id in all meshes) {
        for (src_chip in all chips in mesh) {
            for (dst_chip in all chips in mesh) {
                auto src_coord = mesh_graph.chip_to_coordinate(mesh_id, src_chip);
                auto dst_coord = mesh_graph.chip_to_coordinate(mesh_id, dst_chip);

                if (src_coord[0] != dst_coord[0]) {
                    // Different rows: move N or S first
                    direction = get_shorter_direction_on_row_or_col(
                        mesh_id, src_chip, target_on_column, N, S);
                } else if (src_coord[1] != dst_coord[1]) {
                    // Same row, different column: move E or W
                    direction = get_shorter_direction_on_row_or_col(
                        mesh_id, src_chip, dst_chip, E, W);
                } else {
                    // Same chip: RoutingDirection::C (center/self)
                    direction = RoutingDirection::C;
                }
                intra_mesh_table_[mesh_id][src_chip][dst_chip] = direction;
            }
        }
    }
}
```

The `get_shorter_direction_on_row_or_col` function resolves ties by traversing the connectivity graph in both candidate directions (e.g., both North and South) and returning whichever reaches the destination first. This handles torus wrap-around: in a torus topology, South might be shorter than North even though the destination has a lower row coordinate.

### 3.2 Inter-Mesh Routing: BFS-Based First-Hop Computation

For multi-mesh topologies (e.g., TG -- Tenstorrent Galaxy with multiple interconnected mesh groups), the inter-mesh routing uses BFS:

```cpp
// tt_metal/fabric/routing_table_generator.cpp
std::vector<std::vector<std::pair<ChipId, MeshId>>>
RoutingTableGenerator::get_first_hops_to_all_meshes(
    MeshId src, const InterMeshConnectivity& inter_mesh_connectivity) const {
    // BFS from src mesh to all other meshes
    // first_hops[dst_mesh] = vector of (exit_chip_in_src_mesh, next_mesh) pairs
    std::queue<MeshId> q;
    // ... standard BFS ...
}
```

The result is an **exit node lookup table**: for each destination mesh, which local chip should be used as the exit point, and which neighboring mesh is the first hop. The BFS stores only first-hop information for each destination mesh -- more memory-efficient than storing full paths.

---

## 4. Routing Table Data Structures in L1

### 4.1 `routing_l1_info_t` -- The Per-Core Routing Table

Every Tensix core and every ERISC core stores a `routing_l1_info_t` in its L1:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
struct routing_l1_info_t {
    RouterStateManager state_manager{};                       // 32 bytes
    uint16_t my_mesh_id = 0;                                  // 2 bytes
    uint16_t my_device_id = 0;                                // 2 bytes
    direction_table_t<MAX_MESH_SIZE> intra_mesh_direction_table{};    // 96 bytes
    direction_table_t<MAX_NUM_MESHES> inter_mesh_direction_table{};   // 384 bytes

    union __attribute__((packed)) {
        intra_mesh_routing_path_t<1, false> routing_path_table_1d;  // 1024 bytes
        intra_mesh_routing_path_t<2, true> routing_path_table_2d;   // 1024 bytes
    };

    std::uint8_t exit_node_table[MAX_NUM_MESHES] = {};                // 1024 bytes
    uint8_t padding[12] = {};                                         // 12 bytes
};
static_assert(sizeof(routing_l1_info_t) == 2576);
```

Total: **2,576 bytes per core.** This is replicated on every Tensix and every Ethernet core in the system.

Layout breakdown:

| Component | Offset | Size | Description |
|-----------|:------:|:----:|-------------|
| `state_manager` | 0 | 32 B | Router state + host command interface |
| `my_mesh_id` | 32 | 2 B | This core's mesh ID |
| `my_device_id` | 34 | 2 B | This core's chip ID within mesh |
| `intra_mesh_direction_table` | 36 | 96 B | Compressed 3-bit per entry, 256 entries |
| `inter_mesh_direction_table` | 132 | 384 B | Compressed 3-bit per entry, 1024 entries |
| `routing_path_table` (union) | 516 | 1024 B | 1D uncompressed or 2D compressed routes |
| `exit_node_table` | 1540 | 1024 B | Exit chip ID per destination mesh |
| `padding` | 2564 | 12 B | 16-byte alignment |

### 4.2 `direction_table_t` -- Compressed Direction Tables

The direction tables use 3-bit compressed entries for space efficiency:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
template <std::uint32_t ArraySize>
struct __attribute__((packed)) direction_table_t {
    static constexpr std::uint32_t BITS_PER_COMPRESSED_ENTRY = 3;
    static constexpr std::uint8_t COMPRESSED_ENTRY_MASK = 0x7;

    // 3 bits per entry -> 8 entries per 3 bytes (24 bits)
    // For 256 entries: 256 * 3 / 8 = 96 bytes
    // For 1024 entries: 1024 * 3 / 8 = 384 bytes
    std::uint8_t packed_directions[ArraySize * BITS_PER_COMPRESSED_ENTRY / BITS_PER_BYTE];
};
```

The compressed values map to full direction values:

```cpp
enum class compressed_routing_values : std::uint8_t {
    COMPRESSED_EAST                        = 0,
    COMPRESSED_WEST                        = 1,
    COMPRESSED_NORTH                       = 2,
    COMPRESSED_SOUTH                       = 3,
    COMPRESSED_Z                           = 4,
    COMPRESSED_INVALID_DIRECTION           = 5,  // Maps to 0xDD sentinel
    COMPRESSED_INVALID_ROUTING_TABLE_ENTRY = 6,  // Maps to 0xFF sentinel
};
```

The 3-bit encoding supports 7 values (0-6), which covers 5 directions plus 2 special values. Uncompressed direction tables would require 1 byte per entry (256 B for intra-mesh, 1024 B for inter-mesh), so compression saves $256 - 96 = 160$ bytes intra-mesh and $1024 - 384 = 640$ bytes inter-mesh.

> **Key Insight:** The direction tables tell a core which physical port direction to use for reaching any destination. The routing path tables then provide the detailed per-hop route encoding to embed in the packet header. These are separate because the direction table is queried once (to select the outgoing link) while the path table populates the packet's route buffer.

### 4.3 Routing Path Tables

The routing path table occupies a union at offset 516 in `routing_l1_info_t`:

**1D Uncompressed** (`intra_mesh_routing_path_t<1, false>`):
- 64 entries (MAX_CHIPS_LOWLAT_1D)
- 16 bytes per entry (SINGLE_ROUTE_SIZE_1D = 4 words for 64 hops)
- Total: $64 \times 16 = 1024$ bytes
- Contains pre-computed `LowLatencyRoutingFieldsT` values ready to copy into packet headers

**2D Compressed** (`intra_mesh_routing_path_t<2, true>`):
- 256 entries (MAX_CHIPS_LOWLAT_2D)
- 4 bytes per entry (`compressed_route_2d_t`)
- Total: $256 \times 4 = 1024$ bytes
- Must be decoded via `decode_route_to_buffer()` to populate the packet header's route buffer

### 4.4 Exit Node Table

The exit node table is a simple 1024-byte array indexed by destination mesh ID:

```
exit_node_table[destination_mesh_id] = local_chip_id_to_route_through
```

When a core needs to send to a different mesh, it first looks up the exit chip, then uses the intra-mesh routing table to route to that exit chip, which has a direct Ethernet connection to the target mesh.

---

## 5. Worker Routing L1 Info

Tensix cores carry additional fabric connection information beyond the routing table:

```cpp
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h
struct worker_routing_l1_info_t {
    routing_l1_info_t routing_info{};                              // 2576 bytes
    tensix_fabric_connections_l1_info_t fabric_connections{};       // 656 bytes
};
// Total per Tensix: ~3232 bytes of fabric metadata
```

The `tensix_fabric_connections_l1_info_t` (656 bytes, documented in File 02) provides the EDM connection addresses. Combined with the routing table, each Tensix core carries approximately **3.2 KB** of fabric infrastructure overhead in its L1.

> **Key Insight:** The 3.2 KB per-core overhead for fabric metadata is a significant fraction of the available L1. The routing tables are replicated identically on every core -- a design that enables independent route computation without cross-core communication but at a significant memory cost. For Blackhole with 140 Tensix + 14 active ERISC cores, the aggregate routing table consumption per chip is approximately $(140 + 14) \times 2576 \approx 397$ KB.

---

## 6. Routing Table Addresses in L1

The routing table is placed at fixed addresses on each core type:

```cpp
// From dev_mem_map.h (via fabric_common.h macros)
#if defined(COMPILE_FOR_ERISC)
    #define ROUTING_TABLE_BASE MEM_AERISC_ROUTING_TABLE_BASE
#elif defined(COMPILE_FOR_IDLE_ERISC)
    #define ROUTING_TABLE_BASE MEM_IERISC_ROUTING_TABLE_BASE
#else
    #define ROUTING_TABLE_BASE MEM_TENSIX_ROUTING_TABLE_BASE
#endif
```

The control plane writes the routing table to these addresses on every core via NOC writes during fabric initialization.

---

## 7. Routing Table Generation Pipeline

The complete pipeline from topology discovery to routing-enabled hardware:

```
1. TOPOLOGY DISCOVERY
   - Read cluster descriptor (physical Ethernet links)
   - Load Mesh Graph Descriptor (protobuf or auto-discovered)
   - Build IntraMeshConnectivity and InterMeshConnectivity maps

2. TOPOLOGY MAPPING
   - TopologyMapper: map logical fabric nodes to physical chips
   - TopologySolver: resolve mapping constraints

3. ROUTING TABLE GENERATION
   - RoutingTableGenerator: dimension-ordered intramesh shortest paths
   - RoutingTableGenerator: BFS-based intermesh first-hop computation
   - Compress direction tables to 3-bit entries
   - Compute routing paths (1D uncompressed or 2D compressed)
   - Build exit node lookup tables

4. ROUTING TABLE DISTRIBUTION
   - ControlPlane::configure_routing_tables_for_fabric_ethernet_channels()
   - ControlPlane::write_routing_tables_to_all_chips()
   - Host writes routing_l1_info_t to every core's L1 via NOC

5. EDM INITIALIZATION
   - EDM kernels read routing tables from local L1
   - EDM kernels perform Ethernet handshake (File 01)
   - Fabric enters READY_FOR_TRAFFIC state
```

---

## Key Takeaways

- **The `ControlPlane` is the single source of truth for fabric routing.** It constructs routing tables from the mesh graph, distributes them to all chips, and manages the mapping between logical `FabricNodeId` and physical `ChipId`. All fabric routing decisions ultimately trace back to tables computed by this host-side component.
- **Intra-mesh routing uses dimension-ordered shortest paths (N/S first, then E/W).** The `RoutingTableGenerator` traverses the `IntraMeshConnectivity` graph to find the shortest direction for each source-destination pair, with torus wrap-around handled naturally by bidirectional graph traversal.
- **The `routing_l1_info_t` struct is 2,576 bytes and replicated on every core.** It contains the router state manager (32 B), identity (4 B), compressed direction tables (480 B), routing paths (1024 B), exit node table (1024 B), and padding (12 B). Combined with the 656 B `tensix_fabric_connections_l1_info_t`, each Tensix core carries ~3.2 KB of fabric overhead.
- **Compressed direction tables use 3-bit entries** to map destination IDs to routing directions, saving 800+ bytes compared to uncompressed 1-byte-per-entry tables. The trade-off is a ~2x slower lookup on the device side due to bit manipulation.
- **The `FabricConfig` enum drives the entire routing mode.** It selects 1D vs 2D routing, enables or disables deadlock avoidance, determines the packet header type, and controls whether torus wrap-around links are used.

## Source Code References

- `tt_metal/fabric/control_plane.cpp` -- `ControlPlane` implementation: routing table construction, distribution, topology discovery, Galaxy corner pinning
- `tt_metal/fabric/routing_table_generator.cpp` -- `RoutingTableGenerator`: `generate_intramesh_routing_table`, `generate_intermesh_routing_table`, `get_first_hops_to_all_meshes` (BFS)
- `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp` -- `ControlPlane` public interface: constructors, `write_routing_tables_to_all_chips`, `get_fabric_node_id_from_physical_chip_id`
- `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp` -- `FabricConfig`, `FabricReliabilityMode`, `FabricTensixConfig`, `FabricUDMMode`, `FabricNodeId`, `MeshId`
- `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` -- `routing_l1_info_t`, `direction_table_t`, `compressed_routing_values`, `intra_mesh_routing_path_t`, `worker_routing_l1_info_t`

---

[**Previous:** `03_packet_header_formats_and_routing.md`](./03_packet_header_formats_and_routing.md) | [**Next:** `05_fabric_topology_and_mesh_graph.md`](./05_fabric_topology_and_mesh_graph.md)
