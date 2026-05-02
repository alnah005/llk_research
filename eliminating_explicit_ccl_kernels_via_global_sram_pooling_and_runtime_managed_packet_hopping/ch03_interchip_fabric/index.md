# Chapter 3: Inter-Chip Communication -- Ethernet, Fabric, and Routing

This chapter documents how data crosses chip boundaries today: the Ethernet link layer, the ERISC Data Mover (EDM) firmware that performs store-and-forward routing, the packet header formats that encode multi-hop routes, the control plane that generates routing tables, the mesh graph abstraction that models multi-chip topologies, and the flow control and deadlock avoidance mechanisms that keep the fabric stable. Together, these six sections establish the inter-chip communication substrate that the GSRP proposal (Chapter 6) would extend with transparent address translation and runtime-managed routing.

## Files

| # | File | Description |
|---|------|-------------|
| 1 | [`01_ethernet_link_layer.md`](./01_ethernet_link_layer.md) | Ethernet core specifications, link characteristics (~12.5 GB/s per WH link, ~16 GB/s per BH link), `eth_send_packet_bytes_unsafe()` primitive, TXQ/RXQ interfaces, the EDMStatus handshake state machine, and physical connectivity (WH: 16 links, BH: 14 active ERISC cores). |
| 2 | [`02_erisc_datamover_and_channel_architecture.md`](./02_erisc_datamover_and_channel_architecture.md) | EDM kernel architecture: sender channel 0 (local workers) and channel 1 (upstream forwarding), receiver channel logic, `StaticSizedSenderEthChannel` buffer slot management, credit-based flow control, worker-to-router connection interface, and the speedy-path optimization. |
| 3 | [`03_packet_header_formats_and_routing.md`](./03_packet_header_formats_and_routing.md) | `LowLatencyPacketHeader` (1D, 2-bit per-hop routing), `HybridMeshPacketHeader` (2D, 4-bit per-hop), `NocSendType` and `ChipSendType` enums, compressed routing paths, branch offsets for 2D multicast, and how the EDM router firmware reads per-hop forwarding decisions. |
| 4 | [`04_control_plane_and_routing_tables.md`](./04_control_plane_and_routing_tables.md) | The `ControlPlane` class, `RoutingTableGenerator` with BFS shortest-path computation, routing table data structures (`routing_table_t`, `direction_table_t`, `compressed_route_2d_t`), `FabricConfig` enum variants, and how tables are written to L1 at `MEM_TENSIX_ROUTING_TABLE_BASE`. |
| 5 | [`05_fabric_topology_and_mesh_graph.md`](./05_fabric_topology_and_mesh_graph.md) | `FabricContext` (immutable topology queries, max hops: 63 for 1D / 67 for 2D), `FabricBuilderContext`, the `MeshGraph` abstraction with `IntraMeshConnectivity` and `InterMeshConnectivity`, the Mesh Graph Descriptor protobuf schema, `TopologyMapper`, and Tensix-based fabric extensions (MUX and UDM modes). |
| 6 | [`06_flow_control_and_deadlock_avoidance.md`](./06_flow_control_and_deadlock_avoidance.md) | Three-level flow control (worker-to-EDM, EDM-to-EDM, EDM-to-local), buffer slot management, dateline-based VC separation for ring/torus topologies, bubble flow control, DOR routing, `FabricReliabilityMode` (STRICT, RELAXED, DYNAMIC_RECONFIGURATION), and deadlock avoidance configuration. |

## Reading Guide

Read sequentially: file 01 introduces the physical link, file 02 explains the firmware that runs on top of it, file 03 describes what the packets look like, file 04 covers how routes are computed, file 05 explains the topology model, and file 06 addresses flow control and deadlock avoidance. Readers focused on the GSRP proposal can prioritize files 04 and 06, as these are most heavily referenced by Chapters 6 and 7.

## Key Source Directories

- `tt_metal/fabric/` -- Control plane, mesh graph, topology mapper, fabric context
- `tt_metal/fabric/impl/kernels/edm_fabric/` -- EDM router kernel
- `tt_metal/fabric/hw/inc/edm_fabric/` -- Channel architecture, flow control, worker adapters
- `tt_metal/fabric/fabric_edm_packet_header.hpp` -- Packet header definitions
- `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` -- Shared fabric constants
