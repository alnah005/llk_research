# Research Guide Plan: Eliminating Explicit CCL Kernels via Global SRAM Pooling and Runtime-Managed Packet Hopping

## Final Synthesized Plan

---

## Audience

**Primary readers:** Tenstorrent systems engineers, runtime/compiler architects, and senior firmware developers working on multi-chip infrastructure who are evaluating whether the explicit CCL kernel model can be replaced or augmented with a transparent, runtime-managed communication substrate.

**Secondary readers:** Technical leadership evaluating the architectural feasibility of this model shift, and external researchers studying dataflow architectures and multi-chip communication paradigms.

**Assumed prerequisite knowledge:**
- Working knowledge of the TT-Metal runtime and its programming model (programs, kernels, circular buffers, command queues)
- Familiarity with the Tensix core architecture (BRISC, NCRISC, TRISC, L1 SRAM) and the dual-NOC interconnect
- Basic understanding of how multi-chip Tenstorrent systems are physically connected (Ethernet links between Wormhole/Blackhole chips)
- Exposure to collective communication concepts (all-gather, reduce-scatter, all-reduce) at a conceptual level
- Comfortable reading C++ systems code at the level found in `tt_metal/fabric/` and `ttnn/operations/ccl/`

**Not assumed:**
- Deep knowledge of the fabric subsystem internals, EDM protocol, or control plane routing table generation
- Familiarity with competing architectures (Cerebras, Graphcore, GPU NVLink) -- these are covered comparatively within the guide

**Two reading paths:**
- *Full path (newcomers to fabric):* Read chapters sequentially, 1 through 8.
- *Fast path (engineers familiar with TT-Metal internals):* Start at Chapter 5, referencing earlier chapters as needed.

---

## Chapter List

### Chapter 1: The Current CCL Programming Model -- What We Are Proposing to Eliminate

**Description:** Establishes the baseline by documenting how explicit CCL kernels work today in TT-Metal, covering every operation type, the command-stream programming model, and the host-side orchestration that binds it all together -- this is the "problem statement" that motivates the entire guide.

**Directory:** `ch01_current_ccl_model/`

| File | Content |
|------|---------|
| `01_collective_operations_catalog.md` | Catalogs every CCL operation currently implemented: all-gather, reduce-scatter, all-reduce (experimental), all-broadcast, broadcast, reduce-to-root, all-to-all-dispatch, all-to-all-combine, and mesh-partition. For each: API signature (e.g., `ExecuteAllGather::invoke` with `dim`, `cluster_axis`, `topology`, `num_links` parameters), supported topologies (`NeighborExchange`, `Linear`, `Ring`, `Mesh`, `Torus`), supported memory configurations (interleaved, sharded, DRAM, L1), and data types. References: `ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.hpp`, `reduce_scatter/reduce_scatter.hpp`, `all_reduce/all_reduce.hpp`, `all_broadcast/`, `broadcast/`, `reduce_to_root/`, `all_to_all_dispatch/`, `all_to_all_combine/`, `mesh_partition/`. |
| `02_device_operation_anatomy.md` | End-to-end walkthrough of a single CCL operation using all-gather as the canonical example. Traces the path from `ExecuteAllGather::invoke` through `AllGatherDeviceOperation`, its `operation_attributes_t` and `tensor_args_t`, the `create_mesh_workload` factory, per-device `create_at` program creation, global semaphore allocation, worker core selection (`choose_worker_cores()` with row-major/col-major strategies), and runtime argument override. Covers the experimental async variants (all-gather-async, reduce-scatter-minimal-async, llama-reduce-scatter). References: `all_gather_device_operation.hpp/.cpp`, `all_gather_program_factory.cpp`, `ccl_common.hpp`. |
| `03_command_stream_architecture.md` | Explains the CCL command-stream abstraction: `CclCommandArgCode` opcodes, the `CclCommandArgs` variant, `ccl_command.hpp`, `ccl_host_commands.hpp`, and the command-lowering pipeline (`command_lowering.hpp`). Describes how host-side builders (`ccl_command_stream_builders.hpp`, `ccl_worker_builder.hpp`) encode slice descriptors into runtime arguments consumed by device kernels. Covers sharding address generation (`sharding_addrgen.hpp`). References: `ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp`, `common/host/ccl_command_stream_builders.hpp`, `common/kernels/command_processor.hpp`. |
| `04_host_orchestration_and_topology.md` | How CCL operations discover their communication partners and manage execution: `SenderReceiverConfig`, `RingTopology`/`LineTopology`, `get_device_sender_receiver_config`, `get_physical_neighbor_from_physical_coord`, boundary-mode logic (WRAP vs NONE). Covers CCL op fusion (`ccl_op_fusion.hpp`, `AllGatherFusedOpSignaler`) for overlapping communication with computation. Covers the legacy ERISC async datamover kernel path and the newer fabric-based CCL path (`append_fabric_connection_rt_args()`, `RoutingPlaneConnectionManager`). References: `ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp`, `ccl_common.cpp`, `ccl_op_fusion.hpp`. |

---

### Chapter 2: Per-Chip Memory Architecture and the NOC

**Description:** Documents the L1 SRAM layout per core, the dual-NOC system, address encoding, buffer types, and how data moves within a single chip -- the hardware foundation that any "global SRAM pool" must extend across chip boundaries.

**Directory:** `ch02_memory_and_noc/`

| File | Content |
|------|---------|
| `01_tensix_l1_layout.md` | Detailed L1 SRAM memory map for Wormhole (1464 KB per Tensix core, 80 Tensix cores) and Blackhole (1536 KB per Tensix core, 140 Tensix cores). Breakdown of reserved regions: boot code, mailbox (12,896 B), zeros region, firmware bases (BRISC 6 KB, NCRISC 2 KB, TRISC0/1/2 1.5 KB each), NOC counters, fabric counters, fabric connection lock, routing table (2,576 B), fabric connections metadata (656 B), packet header pool (2,304 B). Calculation of `MEM_MAP_END` and how `L1_UNRESERVED_BASE` is derived for user-available space. Covers the `AllocatorConfig` model (`worker_l1_size`, `l1_unreserved_base`, `l1_small_size`, `trace_region_size`), buffer types (`L1`, `L1_SMALL`, `DRAM`, `SYSTEM_MEMORY`, `TRACE`), and the `L1BankingAllocator`. References: `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`, `tt_metal/impl/allocator/l1_banking_allocator.hpp`. |
| `02_ethernet_l1_layout.md` | Ethernet core L1 layout: 256 KB on WH (minus 32 B barrier), 512 KB on BH. Active ERISC vs Idle ERISC (AERISC vs IERISC) memory map differences. Routing firmware reserved regions, kernel config space, fabric router reserved regions, routing table placement, telemetry addresses. How `ERISC_L1_UNRESERVED_BASE` shifts depending on whether routing firmware is enabled. This directly constrains feasibility of address translation tables on Ethernet cores. References: `tt_metal/hw/inc/internal/tt-1xx/wormhole/eth_l1_address_map.h`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/eth_l1_address_map.h`. |
| `03_noc_architecture_and_addressing.md` | Dual-NOC (NOC0, NOC1) system: register-space layout (`NOC_REGS_START_ADDR` at `0xFFB20000`), coordinate virtualization (WH virtual Tensix starts at (18,18); BH at (1,2)), 64-bit NOC address encoding (x, y coords + local address). NOC transaction types: unicast write, unicast read, multicast write, atomic increment, scatter write, fused atomic inc. Transaction IDs, posted vs non-posted writes, `safe_get_noc_addr()`. The `experimental::Noc` class with `AddressType`, `TxnIdMode`, `ResponseMode`, `BarrierMode`. Key constraint: the NOC is chip-local and does not natively cross chip boundaries. References: `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`, `tt_metal/hw/inc/experimental/noc.h`. |
| `04_buffer_types_and_circular_buffers.md` | Circular buffers, `RemoteCircularBuffer` (sender/receiver CB with pages-sent/pages-acked counters and remote NOC atomic increments for flow control), `GlobalCircularBuffer` (multi-sender/multi-receiver shared L1 buffer with sender-receiver core mapping), `GlobalSemaphore`, the dispatch system (`DispatchMemMap`, command queue addresses in L1), and how the command queue enqueues reads/writes to L1 and DRAM. References: `tt_metal/hw/inc/api/remote_circular_buffer.h`, `tt_metal/api/tt-metalium/global_circular_buffer.hpp`, `tt_metal/impl/dispatch/dispatch_mem_map.hpp`. |

---

### Chapter 3: Inter-Chip Communication -- Ethernet, Fabric, and Routing

**Description:** How data crosses chip boundaries today -- the Ethernet link layer, the ERISC Data Mover (EDM) firmware, the fabric control plane, routing tables, and the packet header formats that enable multi-hop data movement.

**Directory:** `ch03_interchip_fabric/`

| File | Content |
|------|---------|
| `01_ethernet_link_layer.md` | Ethernet core specifications (dedicated RISC-V, 256 KB L1 on WH / 512 KB on BH, no Tensix compute), link characteristics (bidirectional, 16B minimum / 1500B maximum packet, ~91% utilization for small payloads), `eth_send_packet_bytes_unsafe()` primitive, Ethernet TXQ/RXQ interfaces, and the handshake protocol (`EDMStatus` state machine: STARTED -> REMOTE_HANDSHAKE_COMPLETE -> READY_FOR_TRAFFIC -> TERMINATED). Physical connectivity: WH has up to 16 Ethernet cores per chip. Ethernet channel direction model: EAST, WEST, NORTH, SOUTH, Z. References: `tt_metal/hw/inc/internal/ethernet/dataflow_api.h`, `tt_metal/fabric/hw/inc/edm_fabric/edm_handshake.hpp`. |
| `02_erisc_datamover_and_channel_architecture.md` | Deep dive into the EDM kernel architecture: two sender channels + one receiver channel per Ethernet link. Sender channel 0 (accepts packets from local workers) vs sender channel 1 (accepts forwarded packets from upstream EDM). Receiver channel logic: write-to-local-chip and/or forward-to-next-hop. `StaticSizedSenderEthChannel` template with buffer slot management, wrap-increment logic, and credit-based flow control. Worker-to-router connection interface: `open_connection_value`, `close_connection_request_value`, stream-register-based credit notification. The speedy-path optimization. References: `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`, `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp`, `tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp`. |
| `03_packet_header_formats_and_routing.md` | `LowLatencyPacketHeader` (1D): 2-bit per-hop routing fields (NOOP, WRITE_ONLY, FORWARD_ONLY, WRITE_AND_FORWARD), supporting up to 64 hops with extension words. `HybridMeshPacketHeader` (2D): per-hop route buffer (35 bytes default, tiers up to 128B for 67 max hops), 4-bit per-hop encoding for EAST/WEST/NORTH/SOUTH forwarding, branch offsets for 2D multicast. `NocSendType` (unicast write, multicast write, atomic inc, scatter write, read), `ChipSendType` (unicast, multicast). `NocUnicastCommandHeader`, `NocMulticastCommandHeader`, `SparseMulticastRoutingCommandHeader`. Compressed routing paths (`compressed_route_2d_t`) with NS/EW hops, direction bits, and turn-point encoding. How the EDM router firmware reads the next hop from the route buffer and makes forwarding decisions. References: `tt_metal/fabric/fabric_edm_packet_header.hpp`, `tt_metal/fabric/hw/inc/tt_fabric_api.h`, `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h`. |
| `04_control_plane_and_routing_tables.md` | The `ControlPlane` class: constructs routing tables from mesh graph descriptors, manages fabric initialization/teardown. `RoutingTableGenerator`: BFS-based intra-mesh shortest-path computation, inter-mesh exit-node lookup tables. Routing table data structures: `routing_table_t`, `direction_table_t` (compressed 3-bit per entry for up to 1024 destinations), `compressed_route_2d_t`. How routing tables are written to every chip's L1 (`MEM_TENSIX_ROUTING_TABLE_BASE`, `MEM_AERISC_ROUTING_TABLE_BASE`). `FabricConfig` enum variants: DISABLED, FABRIC_1D_NEIGHBOR_EXCHANGE, FABRIC_1D, FABRIC_1D_RING, FABRIC_2D, FABRIC_2D_TORUS_X/Y/XY, CUSTOM. References: `tt_metal/fabric/control_plane.cpp`, `tt_metal/fabric/routing_table_generator.cpp`, `tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp`. |
| `05_fabric_topology_and_mesh_graph.md` | `FabricContext` (immutable topology queries, packet specs, max hops: 63 for 1D / 67 for 2D, deadlock avoidance flags) and `FabricBuilderContext` (mutable build-time state, EDM configs). The `MeshGraph` abstraction: `IntraMeshConnectivity`, `InterMeshConnectivity`, `FabricNodeId` (mesh_id + chip_id), `MeshId`. The Mesh Graph Descriptor (MGD) protobuf schema for multi-host topologies. Physical Grouping Descriptor for hierarchical resource specification. `TopologyMapper` and `TopologySolver`. Tensix-based fabric extensions: MUX mode and UDM (Unified Data Movement) mode, `FabricTensixConfig`, `tt_fabric_mux.cpp` kernel. References: `tt_metal/fabric/fabric_context.hpp`, `tt_metal/fabric/mesh_graph.cpp`, `tt_metal/fabric/topology_mapper.cpp`, `tt_metal/fabric/fabric_tensix_builder.hpp`. |
| `06_flow_control_and_deadlock_avoidance.md` | Credit-based flow control between sender and receiver channels. Worker-to-EDM: stream register-based free slot tracking. EDM-to-EDM (across Ethernet): completion counters, ack counters (`ReceiverChannelResponseCreditSender`). Buffer slot management: `wrap_increment()`, power-of-2 optimizations. Deadlock avoidance: dateline concept in ring/torus topologies, bubble flow control, virtual channel separation, `need_deadlock_avoidance_support(direction)` mechanism in `FabricContext`. `FabricReliabilityMode` (STRICT, RELAXED, DYNAMIC_RECONFIGURATION). References: `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp`, `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp`. |

---

### Chapter 4: Existing Building Blocks for Cross-Chip Communication

**Description:** Surveys the infrastructure components already present in TT-Metal that bridge compute kernels and the fabric -- socket APIs, MeshBuffer, mesh device-side primitives, and worker-to-fabric connection management -- establishing how far the system already is from the global SRAM pool vision.

**Directory:** `ch04_building_blocks/`

| File | Content |
|------|---------|
| `01_mesh_api_device_primitives.md` | The 2D mesh fabric API already shipping in `tt_metal/fabric/hw/inc/mesh/api.h`: `fabric_unicast_noc_unicast_write`, `fabric_set_unicast_route`, multicast support via `MeshMcastRange`, `PacketHeaderPool`, and addrgen integration. Shows that these primitives already abstract away much of the packet construction -- the gap to "transparent" is primarily in route computation and address translation. References: `tt_metal/fabric/hw/inc/mesh/api.h`, `tt_metal/fabric/hw/inc/mesh/addrgen_api.h`. |
| `02_worker_to_fabric_connection_management.md` | The `WorkerToFabricEdmSenderImpl` adapter: construction from runtime args, the open/close connection handshake, write-pointer/read-pointer flow control. How Tensix cores discover fabric connections via `MEM_TENSIX_FABRIC_CONNECTIONS_BASE` L1 tables. The `FabricConnectionManager` (forward/backward connections, `BUILD_ONLY` vs `BUILD_AND_OPEN_CONNECTION` modes). `RoutingPlaneConnectionManager` managing up to 4 simultaneous routing-plane connections per worker, with per-connection destination tracking (device ID, mesh ID) in 2D mode. The `append_fabric_connection_rt_args()` and `append_routing_plane_connection_manager_rt_args()` host APIs. References: `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`, `tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp`, `tt_metal/api/tt-metalium/experimental/fabric/fabric.hpp`. |
| `03_socket_abstractions.md` | The device-side socket API: `sender_socket_md`, `receiver_socket_md`, `SocketSenderInterface`, `SocketReceiverInterface`. The sender flow: write_ptr advancement, bytes_sent/bytes_acked tracking, downstream FIFO management, and per-downstream encoding (`downstream_mesh_id`, `downstream_chip_id`, `downstream_noc_x/y`) -- essentially a global address. The socket API's `MeshCoreCoord` = {device_coord, core_coord} cross-device core addressing. `FabricSocket` (implementing `ISocket`), `BidirectionalFabricSocket`, `MpiSocket`, and `MeshSocket` for tensor-level send/recv. `SocketConfig` for specifying sender/receiver mesh IDs and ranks. D2D, H2D, and D2H socket variants. References: `tt_metal/hw/inc/api/socket_api.h`, `tt_metal/api/tt-metalium/experimental/sockets/mesh_socket.hpp`, `ttnn/api/ttnn/distributed/fabric_socket.hpp`. |
| `04_mesh_buffer_and_mesh_device.md` | `MeshBuffer` abstraction: allocating a buffer across a mesh of devices with either `REPLICATED` or `SHARDED` layout. `DeviceLocalBufferConfig` (per-device page size, buffer type, sharding), `ShardedBufferConfig` (global size, shard shape, orientation). How `MeshBuffer::create()` performs lock-step allocation across all mesh devices at a consistent address -- a primitive building block for a global address space. `MeshDeviceImpl` (collection of devices in 2D grid), `MeshDeviceView`, submesh creation, `MeshWorkload` (per-coordinate program mapping), `MeshCommandQueue`. The role of `DistributedContext` (MPI-based or single-host) in mapping ranks to mesh ownership. References: `tt_metal/api/tt-metalium/mesh_buffer.hpp`, `tt_metal/api/tt-metalium/mesh_device.hpp`. |

---

### Chapter 5: Comparative Survey -- How Other Architectures Handle This Problem

**Description:** Reviews how other multi-chip and wafer-scale architectures handle the tension between explicit and implicit inter-chip communication, providing industry context that informs the subsequent design exploration and feasibility assessment.

**Directory:** `ch05_comparative_survey/`

| File | Content |
|------|---------|
| `01_cerebras_wafer_scale.md` | Cerebras WSE: 850,000+ cores on a single wafer with hardware-routed inter-core communication, ~40 GB of on-wafer SRAM, no chip boundaries. Flat 2D mesh routing, per-core SRAM (48 KB), compiler maps dataflow graphs to physical cores with communication implicitly handled by the fabric. Lessons: (a) the ideal of zero-overhead global SRAM is achievable only when there are no chip boundaries, (b) the compiler bears significant complexity, (c) the key advantage is uniform latency rather than transparency per se. |
| `02_nvidia_nvlink_nccl.md` | NVLink/NVSwitch architecture: hardware provides high-bandwidth GPU-to-GPU links (NVLink 4.0 at 900 GB/s bidirectional), NVSwitch enables all-to-all connectivity. NCCL (the collective library) remains explicit software with ring/tree algorithms. NVSwitch in-network reduction complements rather than replaces software orchestration. CUDA Unified Memory provides transparent page migration but with significant performance caveats. Lessons: (a) even with massive hardware investment, explicit collective libraries persist for workload-specific optimization, (b) hardware-assisted collectives complement software orchestration, (c) transparent remote memory access carries significant overhead. |
| `03_graphcore_ipu_exchange.md` | Graphcore IPU: Bulk Synchronous Parallel (BSP) model with compute phase using local SRAM, then explicit exchange phase where all-to-all communication happens simultaneously. IPU-Links provide all-to-all connectivity within a rack. No external memory -- SRAM-only execution. The exchange is compiler-scheduled, not runtime-transparent. Lessons: (a) BSP provides a natural synchronization boundary that avoids deadlocks, (b) compiler-driven communication insertion can hide complexity from the programmer, (c) epoch-based synchronization is powerful but constraining for overlapping compute and communication. |
| `04_lessons_and_applicability.md` | Synthesizes insights across all three architectures and maps them to Tenstorrent's constraints (Tensix core model, explicit kernel programming, Ethernet-based inter-chip links). Key finding: no production multi-chip AI accelerator has achieved fully transparent global SRAM at scale with Ethernet-class interconnects. The practical path is either (a) compiler-assisted communication insertion (Graphcore-like) or (b) increasingly powerful library primitives with hardware assist (NVIDIA-like). For Tenstorrent, the existing TT-Fabric 2D mesh routing + mesh API provide a strong foundation for path (b). The BSP synchronization model suggests epoch-based barriers as a deadlock avoidance strategy worth evaluating. |

---

### Chapter 6: The Global SRAM Pool Abstraction -- Design Space Exploration

**Description:** Proposes and analyzes candidate architectures for presenting all L1 SRAM across a multi-chip mesh as a unified addressable memory, informed by the comparative lessons from Chapter 5 and the existing building blocks cataloged in Chapter 4.

**Directory:** `ch06_global_sram_design/`

| File | Content |
|------|---------|
| `01_problem_statement_and_transparency_levels.md` | Precisely defines the goal: eliminate the need for application-level kernels to manage cross-chip data movement. Distinguishes three levels of transparency: (a) fully hardware-transparent (like cache-coherent NUMA), (b) runtime-assisted with explicit address translation but implicit routing, (c) library-mediated with high-level APIs hiding packet construction. Maps each against current TT hardware capabilities. Uses total addressable SRAM calculation as motivation: 32-chip BH system = 32 chips x ~140 Tensix cores x 1536 KB = ~6.7 GB aggregate (with significant reserved overhead reducing usable space). |
| `02_address_space_design.md` | Proposes global address encoding: {mesh_id, chip_id, core_xy, local_l1_offset} packed into a 64-bit or wider global address. Evaluates the existing `FabricNodeId` (mesh_id + chip_id) + NOC coordinate + L1 offset as a natural basis. Analyzes encoding strategies: (a) extending the NOC address format with reserved bits, (b) software-managed virtual-to-physical translation table in L1, (c) segment-register approach mapping "remote windows" into local address space. Discusses alignment constraints (L1 alignment, NOC word alignment, Ethernet word size of 16 bytes). Considers how `MeshBuffer`'s consistent-address property is already a stepping stone. |
| `03_coherence_and_consistency_models.md` | Argues that full cache coherence is neither necessary nor desirable (no hardware caches on L1 -- the problem simplifies to ordering and visibility guarantees). Proposes relaxed/release consistency with explicit fence operations using existing atomic increment and semaphore primitives (`NocUnicastAtomicIncCommandHeader`, `GlobalSemaphore`). Memory ownership models: single-writer-multiple-reader (SWMR), epoch-based ownership transfer, producer-consumer ring patterns. The BSP epoch model (from Ch 5's Graphcore analysis) as a candidate synchronization framework. |
| `04_candidate_architectures.md` | Presents three concrete candidate architectures: (1) **Fat Fabric API** -- extend the mesh API (`api.h`) with higher-level primitives (`remote_read`, `remote_write`, `remote_atomic`) that hide packet construction and routing but remain explicit calls in kernel code; (2) **Address-Translation Shim** -- a thin firmware layer on each Ethernet core that intercepts NOC transactions targeting out-of-range addresses and converts them to fabric packets; (3) **Compiler-Inserted Communication** -- a TTNN graph compiler pass that detects cross-chip data dependencies and automatically inserts fabric send/receive operations. Evaluates complexity, performance, and hardware requirements for each. |
| `05_address_translation_layer.md` | Designs the address translation mechanism: how a core issuing a NOC write to a "global address" would be intercepted and translated to a fabric packet. Options: (A) host-computed translation tables written to L1, (B) firmware-level translation on ERISC cores, (C) hardware-assisted translation (requires next-gen silicon). The existing `routing_l1_info_t` and compressed routing tables as a foundation. Extending `fabric_set_unicast_route` with an address-based variant. TLB-like page table structures. Overhead analysis: lookup cost per packet vs. amortized cost for bulk transfers. |

---

### Chapter 7: Runtime Infrastructure for Transparent Cross-Chip Data Movement

**Description:** Specifies the runtime components needed to make packet hopping transparent to application kernels, covering transparent injection, flow control, deadlock avoidance, dispatch integration, and resource management.

**Directory:** `ch07_runtime_infrastructure/`

| File | Content |
|------|---------|
| `01_transparent_packet_injection.md` | Designs the mechanism by which a Tensix core's NOC write to a remote global address gets converted into a fabric packet without explicit kernel involvement. Evaluates three approaches: (a) compiler-inserted shim code that intercepts remote writes, (b) firmware trap on out-of-range NOC addresses, (c) hardware NOC extension that detects remote-chip targets. Covers how existing `EriscDatamoverConfig` would need to be adapted. Discusses the write/read path asymmetry: writes can be fire-and-forget (posted), reads require response (round-trip latency implications). The existing `NOC_UNICAST_READ` in `NocSendType` enum suggests hardware awareness of read operations, but cross-chip reads require a fabric-level request/response protocol. References: `tt_metal/fabric/fabric_edm_packet_header.hpp`. |
| `02_flow_control_and_congestion.md` | Analyzes existing flow control mechanisms (static buffer slot allocation, credit-based backpressure via stream registers) and proposes extensions for implicit traffic: dynamic credit allocation, per-destination flow control, priority queuing for latency-sensitive vs. bulk traffic. Discusses congestion at Ethernet links when many cores simultaneously target remote chips. Mitigation strategies: local aggregation via Tensix MUX, rate limiting, traffic shaping. Evaluates `FabricReliabilityMode` as a model for resilience. End-to-end flow control requirements beyond hop-by-hop. |
| `03_deadlock_avoidance_strategies.md` | Explains current deadlock avoidance: dateline-based virtual channel separation for ring topologies, separate routing planes for different traffic classes. Analyzes new deadlock risks introduced by transparent bidirectional traffic (circular dependencies when chip A reads from chip B while chip B reads from chip A). Proposes solutions: virtual channel partitioning by traffic direction, epoch-based barrier synchronization (drawing on Graphcore BSP model from Ch 5), request-reply channel separation. How `need_deadlock_avoidance_support(direction)` could be extended. Buffer reservation strategies: per-destination credit pools vs shared pools with admission control. |
| `04_dispatch_integration_and_resource_management.md` | How the existing dispatch system (hardware command queue, prefetcher, dispatch cores) would interact with transparent cross-chip data movement. Integration with `MeshDevice`, `MeshCommandQueue`, and `MeshSocket` infrastructure. Resource management: L1 is scarce -- adding fabric metadata already consumes ~5.5 KB per Tensix core (routing tables ~2.6 KB + connections ~656 B + packet headers ~2.3 KB). Impact of GSRP on local L1 availability: address translation tables, connection state, buffering for in-transit packets. Partitioning strategies: "globally visible" vs "locally private" L1 regions. Telemetry and debugging: how `FabricTelemetry`, packet recording, watcher, and dprint could be extended to trace implicit cross-chip transfers. |

---

### Chapter 8: Performance Analysis, Hardware Gap Assessment, and Roadmap

**Description:** Quantitative performance comparison grounded in actual benchmark data, per-architecture hardware gap analysis for Wormhole and Blackhole, workload suitability assessment, and a phased adoption roadmap from the current explicit model toward increasingly transparent communication.

**Directory:** `ch08_performance_and_roadmap/`

| File | Content |
|------|---------|
| `01_measured_fabric_bandwidth.md` | Presents actual bandwidth measurements from the TT-Metal benchmark golden data. Blackhole unicast: ~12.8 B/cycle for 4096B packets over a 2-chip line (single link). Wormhole unicast (half-ring, 8 chips): ~6.4 B/cycle for 4096B packets; full-ring: ~5.6 B/cycle. Wormhole multicast (full-ring, 8 chips): ~5.6 B/cycle for 4096B packets. How bandwidth degrades with hop count and bidirectional traffic. Per-link Ethernet bandwidth: WH ~12.5 GB/s per link, 4 links per neighbor direction. References: `tests/tt_metal/microbenchmarks/ethernet/golden/` benchmark CSVs. |
| `02_latency_model.md` | Builds a latency model for cross-chip data movement: per-hop store-and-forward delay, Ethernet link transfer time, NOC write latency at destination, the `PACKET_WORD_SIZE_BYTES = 16` granularity overhead. Compares single-hop vs multi-hop for different mesh sizes (T3K 2x4, Galaxy 8x4, multi-Galaxy). Explicit CCL: latency is deterministic with pre-computed routes and pipelined transfers. GSRP: additional overhead per access from address translation lookup, connection setup, and routing decision. Estimates overhead at 5-15% for small messages, converging to <2% for large bulk transfers. The speedy-path optimization (~9 cycles saved per send) as a reference point for router processing cost. |
| `03_explicit_vs_implicit_tradeoff_matrix.md` | Comprehensive comparison across dimensions: **Throughput** (explicit pipelining vs per-packet routing overhead), **Latency** (deterministic scheduling vs potential congestion), **Resource utilization** (worker cores for communication vs Ethernet/fabric core overhead), **Programmability** (topology-aware programming vs topology-hidden API), **Debuggability** (deterministic execution traces vs non-determinism; current debug tools: watcher, dprint, `fabric_telemetry_converter.hpp`, `fabric_packet_recorder.hpp`), **Scalability** (behavior as chip count grows). Identifies break-even point where runtime overhead becomes negligible. Per-collective analysis: how all-gather's ring-based pipeline, reduce-scatter's chunked reduction, and all-reduce's fused pattern would map onto a global SRAM pool model. |
| `04_wormhole_gap_analysis.md` | What WH hardware already supports vs what is missing for GSRP. Existing: Ethernet cores with programmable firmware, NOC with unicast/multicast/atomic, stream registers for flow control, 1D routing with extension words. Missing: hardware address translation (no MMU/TLB on Tensix/ERISC), no cross-chip cache coherence, no hardware routing beyond on-chip NOC. L1 = 1464 KB per Tensix; ERISC L1 constrained. 16 Ethernet links per chip. The 1D routing model can be extended but 2D mesh routing has limited native support. `NOC_UNICAST_READ` send type exists but cross-chip reads require software protocol. Minimum firmware changes: extend fabric router to support "remote write to arbitrary L1" packet type. |
| `05_blackhole_gap_analysis.md` | BH-specific assessment. L1 = 1536 KB per Tensix; coordinate virtualization enabled; dual RISC-V on Ethernet cores provides more processing headroom for address translation. 2D routing mode (`FABRIC_2D`, `FABRIC_2D_TORUS_*`) already supported, making BH a more natural fit for the global pool model. Blackhole's `noc_address_translation_table` registers and `NOC_SEC_FENCE_RANGE` as potential extension points for remote-chip address interception. Higher Ethernet bandwidth per link. Identification of what Quasar architecture (`tt-2xx`) may provide. |
| `06_workload_suitability.md` | **Data parallelism** (all-reduce dominated): large bulk transfers favor explicit CCL pipelining; GSRP adds overhead without clear benefit. **Tensor/model parallelism** (all-gather/reduce-scatter at potentially small granularity): mixed; GSRP could simplify programming but may sacrifice pipeline overlap. **Pipeline parallelism** (point-to-point): already well-served by `MeshSocket`/`FabricSocket`; GSRP adds marginal value. **Expert parallelism / MoE** (all-to-all with irregular patterns): potentially the best fit for GSRP since explicit coordination is complex and patterns are data-dependent. **Inference with KV-cache sharing**: transparent read-heavy cross-chip access with relaxed consistency is attractive for GSRP. **Activation checkpointing / dynamic routing**: data-dependent communication patterns benefit most from runtime-managed model. |
| `07_phased_adoption_roadmap.md` | **Phase 0 (Today):** Explicit CCL kernels + fabric-based routing. Workers explicitly construct packet headers and manage connections. **Phase 1 (Near-term, software-only, current hardware):** "Named Remote Memory" -- extend `MeshSocket` API to provide named, pre-mapped remote L1 regions. Workers write to local L1 address that is transparently forwarded. Runtime-managed point-to-point remote L1 access using existing fabric infrastructure with host-side global address resolution. Build a TTNN graph pass that auto-inserts fabric communication ops. **Phase 2 (Firmware-assisted, WH/BH):** "Transparent Global Write" -- runtime API (`global_noc_async_write(global_addr, local_src, size)`) that accepts a global address, performs address translation via pre-loaded routing tables, and issues appropriate fabric packets. ERISC firmware extensions for automatic route resolution. Dynamic route caching. Extend ControlPlane with global address registration API. Prototype cross-chip remote read via fabric. **Phase 3 (Next-generation silicon):** "Hardware-Assisted GSRP" -- Tensix cores with global address recognition hardware that intercepts NOC transactions to non-local addresses. Evaluate hardware TLB integration, extended NOC address space, potential in-network reduction support. **Phase 4 (Full integration):** Compiler support for automatic CCL elimination where implicit routing is beneficial, with fallback to explicit CCL for performance-critical collectives. Hybrid model: retain explicit CCL for bulk collectives, use transparent GSRP for point-to-point, irregular, and latency-sensitive communication. Each phase delivers incremental value. Decision gates specified between phases. |

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **L1 / SRAM** | The per-core scratchpad memory on each Tensix or Ethernet core. Used interchangeably. Not a cache -- no coherence protocol. "L1" is the official Tenstorrent term. |
| **NOC** | Network-on-Chip. The on-die interconnect connecting all Tensix cores, DRAM controllers, and Ethernet cores within a single chip. Each chip has two independent NOCs (NOC0, NOC1). |
| **ERISC** | Ethernet RISC. The dedicated RISC-V processor on each Ethernet core that runs fabric router firmware. |
| **EDM** | Erisc Data Mover. The firmware/kernel running on ERISC cores that performs store-and-forward packet routing between chips. Also referred to as the "fabric router." |
| **Fabric** | The inter-chip communication infrastructure: hardware Ethernet links + EDM firmware + host-side control plane + routing tables. |
| **CCL** | Collective Communication Library. The set of high-level multi-chip data movement operations (all-gather, reduce-scatter, etc.) implemented as explicit kernel programs in `ttnn/operations/ccl/`. |
| **Control Plane** | The host-side software (`ControlPlane` class) responsible for topology discovery, routing table generation, and fabric initialization. |
| **Routing Plane** | A set of Ethernet channels forming an independent routing path through the fabric; multiple planes enable parallel traffic flows. WH has 4 planes, BH has 2. |
| **Packet Hopping** | The process of a data packet traversing multiple chips via store-and-forward through intermediate ERISC routers, each forwarding based on route buffer entries. |
| **Global SRAM Pool (GSRP)** | The proposed abstraction where all L1 SRAM across all chips in a mesh appears as a single flat or hierarchically addressed memory space, with the runtime transparently handling cross-chip data movement. |
| **Fabric Node / FabricNodeId** | A logical identifier for a chip within the fabric topology, composed of `(mesh_id, chip_id)`. |
| **Wormhole (WH)** | Tenstorrent's current-generation chip architecture (80 Tensix cores, 1464 KB L1 per core, 16 Ethernet cores, `tt-1xx`). |
| **Blackhole (BH)** | Tenstorrent's next-generation chip architecture (140 Tensix cores, 1536 KB L1 per core, larger Ethernet L1, dual RISC-V per Ethernet core, `tt-1xx`). |
| **MUX** | A Tensix core running a multiplexer kernel that aggregates traffic from multiple worker cores into a single fabric router channel. |
| **UDM** | Unified Data Movement. A mode where Tensix cores run both MUX and relay kernels as extensions to the fabric router. |
| **Source Routing** | TT-Fabric's routing model where the packet header carries the complete route (sequence of per-hop forwarding decisions) rather than relying on per-hop routing table lookups at intermediate routers. |

### Notation

- **Code references** use the format `path/relative/to/tt-metal` with line references where critical (e.g., `tt_metal/fabric/control_plane.cpp:ControlPlane::configure_routing_tables`).
- **Fabric addresses** in examples use the notation `M<mesh_id>:D<chip_id>:C(x,y):0x<offset>` for clarity.
- **Chip generations** are explicitly called out as **Wormhole (WH)** or **Blackhole (BH)** where behavior differs.
- **Memory sizes** use binary units: KB = 1024 bytes, MB = 1024 KB.
- **Bandwidth** is reported in bytes per cycle (B/c) when referencing benchmark golden files, or in GB/s where clock frequency is specified.
- **NOC coordinates** are written as `(x, y)` in virtual coordinate space unless marked as "physical."
- **Addresses** are written in hexadecimal with the `0x` prefix.
- **Latency** values are expressed in microseconds (us) for inter-chip and nanoseconds (ns) for intra-chip unless otherwise stated.

### Formatting Rules

- Each chapter directory contains numbered markdown files that should be read in order.
- Each chapter file begins with a one-paragraph "Context" section stating what the reader should already know and what they will learn.
- Each chapter ends with a "Key Takeaways" section (3-5 bullet points) and a "Source Code References" section listing the exact files examined.
- Code snippets from the TT-Metal codebase are presented in fenced code blocks with C++ syntax highlighting and include the source file path as a comment on the first line.
- Architecture diagrams use ASCII art for maximum portability, consistent with existing TT-Fabric documentation style.
- Key findings and recommendations are highlighted in blockquote callouts prefixed with "Key Insight:" or "Recommendation:".
- Performance numbers are always accompanied by the assumptions and configuration under which they were measured or estimated.
- When contrasting explicit CCL vs. implicit approaches, a consistent two-column format is used for side-by-side comparison.
- Trade-off analyses use structured tables with explicit criteria columns.

---

## Cross-Chapter Dependencies

| Consuming Chapter | Depends On | Concepts Referenced |
|---|---|---|
| Chapter 2 (Memory & NOC) | None | Foundational -- introduces L1 layout and NOC addressing used everywhere. |
| Chapter 1 (Current CCL Model) | Chapter 2 | References L1 buffer types, memory configs, NOC address encoding, and circular buffer mechanisms. |
| Chapter 3 (Inter-Chip Fabric) | Chapter 2 | Builds on NOC addressing to explain how packets are formed and routed across chips. Extends L1 layout to include routing table memory regions on Ethernet cores. |
| Chapter 4 (Building Blocks) | Chapters 2, 3 | Uses NOC address concepts (Ch 2) and fabric routing infrastructure (Ch 3) to explain how existing abstractions bridge the worker-to-fabric gap. |
| Chapter 5 (Comparative Survey) | None directly | Self-contained external survey, but most valuable after reading Chapters 1-4 for Tenstorrent-specific context. |
| Chapter 6 (GSRP Design) | Chapters 2, 3, 4, 5 | Requires per-chip memory layout (Ch 2), routing mechanisms (Ch 3), existing abstractions (Ch 4) to evaluate feasibility. Uses industry lessons (Ch 5) to frame design choices. |
| Chapter 7 (Runtime Infrastructure) | Chapters 3, 4, 6 | Builds on fabric layer (Ch 3), existing building blocks (Ch 4), and proposed design (Ch 6) to specify transparent injection, flow control, and deadlock avoidance. |
| Chapter 8 (Performance & Roadmap) | All prior chapters | Synthesizes hardware facts (Ch 2), fabric capabilities (Ch 3), existing building blocks (Ch 4), external lessons (Ch 5), design proposals (Ch 6), runtime infrastructure (Ch 7) into quantitative analysis, gap assessment, and actionable roadmap. |

**Note on Chapter 1 and Chapter 2 ordering:** Chapter 1 (CCL) is numbered first because it establishes the "problem" -- what we are proposing to eliminate. It references L1 and NOC concepts from Chapter 2. Readers who are not already familiar with Tensix L1 layout and NOC addressing should read Chapter 2 first, then Chapter 1. Readers who already know the hardware basics should read Chapter 1 first for motivation. The "Full path" and "Fast path" reading orders in the Audience section accommodate both approaches.

---

## Selection Rationale

This final plan is a hybrid that incorporates the strongest elements from each of the five candidate plans and addresses the cross-evaluation findings identified by all five evaluators.

### Structural Decisions

**Chapter ordering -- CCL motivation first, hardware second (from Plans 2, 3, 4, 5; addressing evaluator feedback on Plan 1):** Multiple evaluators flagged that Plan 1's strict bottom-up ordering (memory maps before CCL) delays the core question. This plan leads with the CCL programming model (Chapter 1) to establish the problem statement, consistent with Plans 2-5, while providing explicit guidance for readers who prefer to start with hardware foundations (read Chapter 2 first). The dependency table and dual reading paths make both approaches work.

**Comparative survey moved earlier (addressing unanimous evaluator feedback):** All five evaluators noted that burying the comparative architecture survey at the end (as Plans 1, 4, and 5 do) prevents industry lessons from informing the design exploration. This plan places the survey as Chapter 5, before the GSRP design space (Chapter 6), so that Cerebras's wafer-scale lessons, NVIDIA's persistent-explicit-CCL observation, and Graphcore's BSP model can directly inform the design candidates and deadlock avoidance strategies. This follows the spirit of Plans 2 and 3's placement.

**Dedicated Building Blocks chapter (from Plans 3 and 5):** Evaluators for Plans 1, 2, and 4 all recommended adding or expanding coverage of existing building blocks. Plan 3's Chapter 5 and Plan 5's Chapter 4 were cited as the best treatments. This plan includes a dedicated Chapter 4 that surveys the mesh API device primitives, worker-to-fabric connection management, socket abstractions, and MeshBuffer -- establishing the gap between current state and proposed future before the design exploration begins.

**Dedicated Runtime Infrastructure chapter (from Plans 1 and 2):** Evaluators for Plans 3, 4, and 5 all identified the lack of deep runtime infrastructure coverage as a significant gap. Plan 1's Chapter 6 (4 files on address translation, transparent read/write paths, flow control, resource management) and Plan 2's Chapter 5 (4 files on transparent injection, flow control, deadlock avoidance, dispatch integration) were cited as the strongest treatments. This plan includes Chapter 7 with 4 files covering all of these topics, drawing on Plan 1's write/read path asymmetry analysis and Plan 2's dispatch integration treatment.

### Content Incorporated from Each Plan

**From Plan 1 (most technically specific):**
- Exact memory map constants, byte counts for reserved regions, and struct names throughout Chapters 2 and 3
- The write/read path asymmetry analysis (writes are posted, reads require response) in Chapter 7
- Resource management and L1 overhead analysis (~5.5 KB per Tensix core for fabric metadata) in Chapter 7
- The phased roadmap structure (Phase 0-3) with concrete per-phase deliverables in Chapter 8
- The "Key Takeaways" and "Source Code References" formatting convention

**From Plan 2 (best runtime infrastructure and audience definition):**
- Two-tier audience definition (primary engineers, secondary leadership/researchers)
- The `M<mesh_id>:D<chip_id>:C(x,y):0x<offset>` fabric address notation
- Dispatch integration coverage (MeshCommandQueue, hardware command queue interaction) in Chapter 7
- The four-phase roadmap extension including Phase 4 (compiler support for automatic CCL elimination)
- The non-standard reading order acknowledgment, adapted into the "Full path" vs "Fast path" model

**From Plan 3 (best building blocks treatment and candidate architecture framework):**
- The dedicated Building Blocks chapter structure (mesh API primitives, socket abstraction, MeshBuffer, RoutingPlaneConnectionManager) adapted into Chapter 4
- The three-candidate-architecture framework (Fat Fabric API, Address-Translation Shim, Compiler-Inserted Communication) in Chapter 6
- The three transparency levels (hardware-transparent, runtime-assisted, library-mediated) in the problem statement
- The Key Source Files Referenced appendix concept, adapted into per-chapter "Source Code References" sections

**From Plan 4 (best worker-to-fabric bridging and per-architecture analysis):**
- The worker-to-fabric connection management treatment (WorkerToFabricEdmSender, FabricConnectionManager, connection lifecycle) integrated into Chapter 4
- The per-chip-generation hardware gap analysis split (separate Wormhole and Blackhole files) in Chapter 8
- The dual reading path approach (newcomers vs. experienced engineers)
- Coverage of `FabricSocket`, `BidirectionalFabricSocket`, and `MpiSocket` in the socket abstractions file
- The mesh graph descriptor and physical system descriptor coverage in Chapter 3

**From Plan 5 (best benchmark data and hybrid proposal):**
- Actual measured bandwidth data from TT-Metal benchmark golden files (BH unicast ~12.8 B/cycle, WH half-ring ~6.4 B/cycle) in Chapter 8
- The per-architecture hardware requirement breakdown for Wormhole and Blackhole as separate files
- The hybrid architecture proposal (retain explicit CCL for bulk, add GSRP for irregular/P2P) integrated into the roadmap
- Blackhole's `noc_address_translation_table` registers as potential extension points
- The "Context" section at the start of each chapter formatting rule
- Coverage of Quasar architecture features as forward-looking content in the BH gap analysis

### What Was Deliberately Excluded

- **Plan 1's Chapter 4 (Packet Routing Mechanics) as a standalone chapter:** Multiple evaluators noted this overlapped with Chapter 2's fabric coverage. The content is consolidated into Chapter 3 (files 03 and 06) rather than having a separate chapter.
- **Plan 3's standalone Control Plane chapter:** Evaluators noted this was thin at 3 files and could be folded into the fabric chapter. Control plane coverage is integrated into Chapter 3, files 04-05.
- **Intel Gaudi from Plan 1's comparative survey:** Dropped in favor of deeper treatment of the three architectures (Cerebras, NVIDIA, Graphcore) that provide the most distinctive lessons. Intel Gaudi's RDMA-based model is similar enough to the current TT-Metal approach that a brief mention in the synthesis file suffices.
- **Plan 2's recommended reading order that differs from chapter numbering:** Instead of numbering chapters one way and recommending reading another way (which evaluators flagged as confusing), this plan uses the "Full path" / "Fast path" model with a clear note about Chapter 1 and 2 ordering.
