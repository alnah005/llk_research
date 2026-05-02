# Chapter 4: Existing Building Blocks for Cross-Chip Communication

This chapter surveys the infrastructure components already present in TT-Metal that bridge compute kernels and the fabric: the 2D mesh API device primitives, worker-to-fabric connection management, socket abstractions for device-to-device data movement, and the MeshBuffer/MeshDevice abstractions for multi-chip resource management. Together, these four sections establish how far the system already is from the global SRAM pool vision and identify the remaining gaps that Chapters 6 and 7 address.

## Files

| # | File | Description |
|---|------|-------------|
| 1 | [`01_mesh_api_device_primitives.md`](./01_mesh_api_device_primitives.md) | The 2D mesh fabric API in `api.h`: `fabric_unicast_noc_unicast_write`, `fabric_set_unicast_route`, multicast via `MeshMcastRange`, `PacketHeaderPool`, and addrgen integration. Shows that these primitives already abstract much of the packet construction -- the gap to "transparent" is primarily in route computation and address translation. |
| 2 | [`02_worker_to_fabric_connection_management.md`](./02_worker_to_fabric_connection_management.md) | `WorkerToFabricEdmSenderImpl` adapter, open/close connection handshake, fabric connection discovery via `MEM_TENSIX_FABRIC_CONNECTIONS_BASE`, `FabricConnectionManager`, `RoutingPlaneConnectionManager` (up to 4 simultaneous routing-plane connections per worker), and the `append_fabric_connection_rt_args()` host API. |
| 3 | [`03_socket_abstractions.md`](./03_socket_abstractions.md) | Device-side socket API (`sender_socket_md`, `receiver_socket_md`, `SocketSenderInterface`, `SocketReceiverInterface`), `FabricSocket`, `BidirectionalFabricSocket`, `MpiSocket`, `MeshSocket` for tensor-level send/recv, `SocketConfig`, and D2D/H2D/D2H socket variants. The socket API's `MeshCoreCoord` is essentially a global address. |
| 4 | [`04_mesh_buffer_and_mesh_device.md`](./04_mesh_buffer_and_mesh_device.md) | `MeshBuffer` with `REPLICATED` and `SHARDED` layouts, `DeviceLocalBufferConfig`, `ShardedBufferConfig`, consistent-address allocation. `MeshDeviceImpl`, `MeshDeviceView`, submesh creation, `MeshWorkload`, `MeshCommandQueue`, and `DistributedContext` for MPI-based multi-host coordination. |

## Reading Guide

Read sequentially: file 01 shows the low-level device primitives, file 02 explains how workers connect to the fabric, file 03 introduces the socket abstraction that builds on top, and file 04 covers the mesh-level resource management. Together they paint a picture of the existing "almost-transparent" infrastructure. Readers already familiar with the mesh API can skip to file 04 for the MeshBuffer primitives most relevant to GSRP.

## Key Source Directories

- `tt_metal/fabric/hw/inc/mesh/` -- Mesh API device primitives
- `tt_metal/fabric/hw/inc/edm_fabric/` -- Worker-to-fabric adapters, connection management
- `tt_metal/hw/inc/api/` -- Device-side socket API
- `tt_metal/api/tt-metalium/` -- MeshBuffer, MeshDevice, GlobalSemaphore, FabricSocket
- `ttnn/api/ttnn/distributed/` -- TTNN-level socket and distributed abstractions
