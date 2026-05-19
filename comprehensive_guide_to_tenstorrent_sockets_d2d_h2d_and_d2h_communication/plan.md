# Final Synthesized Plan: Tenstorrent Socket System -- Inter-Mesh and Host-Device Communication for Multi-Mesh Workloads

---

## Selection Rationale

This plan is a hybrid drawing the best structural decisions from all four candidates, guided by the consensus patterns identified across evaluators.

**From Plan V1:**
- The detailed, field-level outline depth for per-file content descriptions. V1's outlines are the most thorough and reduce authoring ambiguity.
- The comprehensive 30+ term glossary (expanded here with V4's additional entries for 40+ total).
- The dedicated troubleshooting/debugging file (V1 Ch8 File 4), which V2, V3, and V4 either lacked or buried.
- The thorough DeepSeek V3 pipeline treatment (4 files), adapted here as 4 files in Chapter 7.
- The explicit cross-chapter dependency graph with file-level cross-references.

**From Plan V2:**
- The formalized reading-order recommendations (5 paths for different audiences). This is unique to V2 and universally praised by evaluators.
- The clean separation of D2D configuration (data model) from D2D operations (runtime behavior). This makes Chapter 2 a pure configuration reference and Chapter 3 a runtime/SPMD/TT-Fabric chapter.
- The per-chapter "Addresses questions" annotations and file-level question coverage matrix.
- The "comparison and selection guide" concept in the overview chapter.

**From Plan V3:**
- The dedicated TT-Fabric chapter concept. All evaluators noted that TT-Fabric needs coherent standalone treatment rather than being scattered. V3's dedicated chapter is adopted here but repositioned earlier (Chapter 3) rather than V3's Ch6 or V4's Ch8, so that D2D operations have fabric context available.
- The `index.md` navigation files per chapter for filesystem usability.
- The clean "Chapters 1-4 are independently readable" property for reference use.

**From Plan V4:**
- The consolidated H2D + D2H chapter (V4 Ch3). Evaluators noted this highlights the symmetry between the two PCIe-based socket types and reduces redundancy. Adopted here as Chapter 5.
- The security and operational considerations file (V4 Ch5 File 3), which is unique content no other plan provides.
- The end-to-end DeepSeek V3 inference walkthrough file (V4 Ch7 File 4), which traces initialization through prefill, decode, and result extraction.
- The 45-entry terminology table (expanded and merged with V1's glossary).
- SPMD coverage placed alongside D2D operations (V4 Ch2 File 3), ensuring rank-aware programming is understood before cross-process sharing is introduced.

**Key structural decisions driven by evaluator consensus:**
1. **D2D comes first** after the overview (most complex, foundational for multi-mesh).
2. **TT-Fabric appears early** (Chapter 3, right after D2D config) so D2D operations have transport context.
3. **Intra-process sockets are folded into the D2D chapter** (not a standalone chapter -- V4's Ch6 was unanimously identified as too thin).
4. **Flow control is introduced briefly per-type, then unified once** in a dedicated chapter -- avoiding the triple-coverage problem flagged in V1 and V2 evaluations.
5. **H2D and D2H are combined** into one chapter, following V4's approach, with a shared flow control file highlighting symmetry.
6. **Cross-process sharing gets its own chapter** after per-type introductions, following consensus.
7. **DeepSeek V3 is a comprehensive 4-file case study**, drawing from V1's depth and V4's end-to-end walkthrough.
8. **A dedicated troubleshooting/debugging section** is included, following V1.
9. **7 chapters** (within the 6-8 range), achieving the right balance between depth and navigability.

---

## Audience

**Primary:** Tenstorrent systems engineers and distributed inference developers building multi-host, multi-mesh inference pipelines on Tenstorrent hardware. They are writing or integrating pipeline stages that communicate tensors between mesh devices (D2D), stream data from host to device (H2D), and extract results from device to host (D2H). They may be extending DeepSeek V3's pipeline infrastructure or building new multi-stage inference services.

**Secondary:** TT-Metal runtime developers working on the socket implementation itself (TT-Fabric integration, flow control, cross-process sharing), kernel developers who need to understand how device-side socket endpoints interact with their compute kernels, ML infrastructure engineers integrating socket-based communication into higher-level abstractions, and performance/operations engineers responsible for deploying and debugging multi-process Tenstorrent workloads in production.

**Assumed knowledge:**
- Familiarity with Tenstorrent hardware concepts: mesh devices, Tensix cores, L1 SRAM, DRAM, NOC (Network on Chip), ethernet links between meshes
- Basic understanding of TT-Metal's programming model: Programs, kernels, circular buffers, MeshDevice, MeshCoordinate
- Experience with TTNN at the Python API level: tensor creation, device placement, `ttnn.experimental` operations
- Understanding of PCIe host-device communication fundamentals (DMA, TLB, pinned memory) at a conceptual level
- Python fluency and basic C++ header-reading ability (templates, smart pointers, enums)
- Familiarity with multi-process programming concepts (shared memory, process isolation) and FIFO circular buffer concepts

**Not assumed:**
- Detailed knowledge of TT-Fabric's packet routing, ethernet link management, FABRIC_2D topology, or MGD connections
- Understanding of the socket C++ implementation (MeshSocket, H2DSocket, D2HSocket APIs, SocketHandle, SocketEndpoint)
- Knowledge of SocketConnection, SocketMemoryConfig, SocketConfig, or ExternalConfigBuffer
- Experience with cross-process shared memory (`/dev/shm`), flatbuffer serialization, or the export/connect pattern
- Knowledge of vIOMMU configuration, PCIe TLB management, or HOST_PUSH/DEVICE_PULL modes
- Understanding of circular FIFO flow control protocols (bytes_sent/bytes_acked)
- Familiarity with DeepSeek V3's pipeline architecture, PipelineBlock, SocketInterface, or HostInterface abstractions
- Understanding of SPMD (Single Program, Multiple Data) programming patterns for distributed workloads

**After reading, the reader should be able to:** select the correct socket type for a given communication pattern; configure D2D sockets with proper SocketConfig/SocketConnection/SocketMemoryConfig; implement H2D and D2H data streaming with correct flow control; share sockets across processes using the export/connect pattern; reason about FIFO sizing, L1/DRAM trade-offs, and PCIe alignment; understand SPMD rank-aware programming for multi-process socket deployments; understand how DeepSeek V3 composes all three socket types into a production pipeline; and debug common socket issues related to flow control, alignment, fabric configuration, and cross-process lifecycle.

---

## Chapter List

### Chapter 1: Socket System Overview and Architecture

**Description:** Introduces the three socket types as a unified communication system, establishes the taxonomy of data flow directions and transport mechanisms, provides a decision guide for socket selection, and previews the common abstractions (FIFO model, non-blocking semantics, SPMD) used throughout the guide.

**Directory:** `ch01_socket_overview/`

**Files:**

- **`index.md`**
  - Chapter navigation and reading-order list with clickable links to all files in this chapter

- **`01_socket_taxonomy_and_purpose.md`**
  - The problem sockets solve: multi-mesh workloads require structured, flow-controlled data movement between meshes (D2D), from host into devices (H2D), and from devices back to host (D2H)
  - Historical context: before sockets, multi-mesh communication relied on manual TT-Fabric programming or host-mediated tensor copies -- error-prone and non-composable
  - Taxonomy of the three socket types:
    - **D2D (MeshSocket):** tensor transfer between cores on different mesh devices via TT-Fabric ethernet links. Sender and receiver are both device cores. Transport: TT-Fabric packet routing through ethernet.
    - **H2D (H2DSocket):** data streaming from host CPU memory to device L1. Two sub-modes: HOST_PUSH (host writes directly to device L1 via PCIe TLB) and DEVICE_PULL (host writes to pinned memory, device reads via PCIe NOC). Transport: PCIe.
    - **D2H (D2HSocket):** data streaming from device cores to host CPU memory. Device NOC writes to pinned host memory; host reads from the FIFO. Transport: PCIe via device-initiated NOC writes.
  - Comparison table: socket type vs direction vs transport vs sender location vs receiver location vs API surface vs typical use case vs latency characteristics
  - Where sockets sit in the stack: application code (Python) -> TTNN experimental API / C++ socket classes -> TT-Metal runtime -> TT-Fabric (for D2D) / PCIe subsystem (for H2D/D2H) -> hardware
  - When to use sockets vs other communication mechanisms: sockets vs CCL collective operations (all_gather, reduce_scatter), sockets vs direct DRAM reads/writes, sockets vs program-level data movement
  - Decision tree: given a communication requirement, which socket type to use

- **`02_common_abstractions_preview.md`**
  - The FIFO model at a glance: all socket types use a circular buffer (FIFO) with a producer writing pages and a consumer reading pages. The FIFO has a fixed size (`fifo_size`) and page-oriented access. Full details in Chapter 5 (flow control).
  - Flow control primitives (brief): `bytes_sent` (producer counter), `bytes_acked` (consumer counter). Producer can write when space is available. Consumer signals consumption by advancing `bytes_acked`. (Deep treatment deferred to Chapter 5.)
  - Non-blocking semantics: socket operations (`send_async`, `recv_async`, `write`, `read`) are non-blocking -- they enqueue work and return immediately. `barrier()` blocks until all pending operations complete.
  - The SPMD execution model preview: all processes run the same code (same Python script), but branch on their rank to determine whether they act as sender or receiver. `sender_rank` and `receiver_rank` fields in socket configuration drive this branching. (Full treatment in Chapter 3.)
  - Socket lifecycle phases: creation/configuration -> connection establishment -> data transfer -> teardown
  - Cross-process sharing preview: `export_descriptor` serializes socket state to shared memory; `connect` attaches a remote process to an existing socket. (Full treatment in Chapter 6.)

**Addresses questions:** Q1 (three socket types and differences)

---

### Chapter 2: D2D MeshSocket Configuration

**Description:** Dissects the configuration data model for Device-to-Device MeshSockets -- MeshCoreCoord, SocketConnection, SocketMemoryConfig, and SocketConfig -- showing how these four structs compose to fully specify a cross-mesh communication channel, and covers the send/recv async operations, intra-process socket pairs, and SPMD rank-aware programming.

**Directory:** `ch02_d2d_mesh_socket/`

**Files:**

- **`index.md`**
  - Chapter navigation and reading-order list with clickable links to all files in this chapter

- **`01_configuration_hierarchy.md`**
  - `MeshCoreCoord`: a composite coordinate specifying both the device within the mesh (`device_coord` as `MeshCoordinate`) and the core on that device (`core_coord` as `CoreCoord`). This identifies a unique physical core endpoint across the entire multi-mesh system. The distinction between logical and physical coordinates.
  - `SocketConnection`: a 1:1 pairing of `sender_core: MeshCoreCoord` to `receiver_core: MeshCoreCoord`. Each connection defines one unidirectional data channel. Multiple connections can be specified to create multi-core parallel transfers.
  - Connection topology patterns: 1-to-1 (single core to single core), fan-out, fan-in, and all-to-all between core groups for sharded tensor transfer
  - `SocketMemoryConfig`: controls where the socket's data buffer and config buffer are allocated.
    - `buffer_type`: `BufferType::L1` (fast, limited capacity, ~1.5MB per core on Blackhole) vs `BufferType::DRAM` (slower, larger capacity, 12GB shared DRAM per device)
    - `fifo_size`: the circular buffer capacity in bytes. Larger FIFO allows more in-flight data and hides latency; smaller FIFO conserves memory.
    - Optional `sender_sub_device_id` and `receiver_sub_device_id`: bind the socket to a specific sub-device within a mesh
  - `SocketConfig`: the top-level configuration struct that composes connections, memory config, and endpoint identity.
    - `connections: vector<SocketConnection>` -- the set of core-to-core channels
    - `memory_config: SocketMemoryConfig` -- shared memory configuration for all connections
    - Endpoint identity (two patterns):
      - `sender_rank / receiver_rank` + `distributed_context`: for multi-process SPMD where rank determines endpoint type
      - `sender_mesh_id / receiver_mesh_id`: for intra-process multi-mesh where mesh identity determines endpoint type
    - Optional `distributed_context`: the inter-process communication context; required for cross-process D2D sockets
  - Complete configuration assembly walkthrough: starting from "I want to send a tensor from core (1,0) on mesh 0 to core (2,0) on mesh 1" -- constructing each struct in order, then assembling the final SocketConfig
  - Configuration validation: what happens when connections reference invalid coordinates, when fifo_size is not page-aligned, when ranks do not match the distributed context
  - Source references: `mesh_socket.hpp`

- **`02_send_recv_async_operations.md`**
  - `send_async(tensor, socket)`: non-blocking dispatch that enqueues a tensor for transmission from the sender's cores to the receiver's cores. The tensor must be device-resident and already allocated. Semantics: returns immediately; actual data transfer proceeds asynchronously via TT-Fabric.
  - `recv_async(tensor, socket)`: non-blocking receive into a pre-allocated output tensor. The receiver tensor must be allocated on the destination device with compatible shape and layout. The operation configures the receiver-side fabric endpoint to accept incoming data.
  - Pre-allocation requirement: both send and receive tensors must be allocated before the async operations are issued. Why: the socket maps physical buffer addresses at setup time; dynamic allocation would break the zero-copy transfer model.
  - `get_socket_endpoint_type()`: returns `SocketEndpointType::SENDER` or `SocketEndpointType::RECEIVER` -- the caller uses this to branch logic in SPMD code
  - `get_data_buffer()` and `get_config_buffer()`: access to underlying buffers for inspection or custom operations
  - Synchronization: `ttnn.experimental.barrier(socket)` or `mesh_device.synchronize()` to wait for pending transfers. After barrier, the receiver's tensor contains valid data.
  - Ordering guarantees: sends from the same socket connection are delivered in FIFO order; sends from different connections have no ordering guarantees
  - Error semantics: what happens when send is called on a RECEIVER endpoint, when recv is called on a SENDER endpoint, when the tensor shape does not match the connection topology
  - Source references: `mesh_socket.hpp`, `test_multi_mesh.py`

- **`03_spmd_and_intra_process.md`**
  - **SPMD rank-aware programming:**
    - The SPMD programming model for sockets: all processes run the same code, but branch on their rank to determine whether they act as sender or receiver
    - `tt::distributed::open_distributed_context(rank, world_size)`: establishes the process's identity in the distributed system
    - How rank determines endpoint type: in a SocketConfig with `sender_rank=0, receiver_rank=1`, the process with rank 0 creates a SENDER endpoint and rank 1 creates a RECEIVER endpoint. Both processes use the identical SocketConfig.
    - The canonical SPMD code pattern from `test_multi_mesh.py`: (1) set_fabric_config(FABRIC_2D), open_mesh_device, (2) construct identical SocketConfig, (3) create MeshSocket, (4) branch on get_socket_endpoint_type() for send_async vs recv_async, (5) barrier
    - `FABRIC_2D` setup: the fabric topology must be initialized before socket creation. `set_fabric_config(FABRIC_2D)` establishes ethernet routing tables across the mesh. (Deep TT-Fabric treatment in Chapter 3.)
    - Multi-rank configurations: extending beyond 2 ranks with multiple sockets
  - **Intra-process socket pairs:**
    - `MeshSocket::create_socket_pair(mesh_device_a, mesh_device_b, socket_config)`: static factory for creating a matched sender/receiver pair within a single process that owns two mesh devices
    - When to use: single-process multi-mesh (e.g., one process controlling two meshes on the same host), testing and development, performance benchmarking
    - How it differs from cross-process sockets: no flatbuffer serialization, no `/dev/shm` descriptors, no export/connect lifecycle, no distributed context needed
    - Limitations: both meshes must be accessible from the same process; not suitable for production multi-host deployments
    - Testing pattern from `test_multi_mesh.py`: create socket pair, send tensor from mesh 0, receive on mesh 1, validate tensor equality
    - Transitioning from intra-process to multi-process: the socket operations (send_async, recv_async) are identical; only the creation path changes
  - Source references: `mesh_socket.hpp`, `test_multi_mesh.py`

**Addresses questions:** Q2 (D2D configuration), Q3 (send_async/recv_async), Q8 (intra-process sockets), Q12 (SPMD model)

---

### Chapter 3: TT-Fabric and Multi-Mesh Topology

**Description:** Explains how D2D sockets interact with TT-Fabric's routing layer, covering FABRIC_2D topology configuration, MGD routing for multi-Galaxy systems, ethernet channel management, bandwidth and latency characteristics, and how socket connections map to physical fabric links.

**Directory:** `ch03_tt_fabric/`

**Files:**

- **`index.md`**
  - Chapter navigation and reading-order list with clickable links to all files in this chapter

- **`01_fabric_topology_and_configuration.md`**
  - TT-Fabric as the inter-chip communication network: ethernet links connecting chips within and across meshes
  - Mesh device topology: a single mesh (e.g., 4x4 grid of Blackhole chips) vs multi-mesh (e.g., two 4x4 meshes in a Galaxy system) vs multi-host (meshes connected across physical machines)
  - `FABRIC_2D` topology: the standard 2D mesh fabric configuration for Galaxy systems. Chips are arranged in a 2D grid with dimension-ordered routing (X-then-Y). `set_fabric_config(FABRIC_2D)` sets up routing tables on all ethernet cores, establishes virtual channels, and configures credit-based flow control. Must be called before any MeshSocket creation.
  - `MeshShape` configuration: specifying the dimensions of a mesh (e.g., `MeshShape(4,4)` for a 4x4 grid)
  - MGD (Mesh Gateway Device) connections: designated gateway devices that bridge between meshes; how D2D socket connections route through MGD nodes when crossing mesh boundaries; how the runtime discovers the topology; what `distributed_context` provides for cross-host communication
  - Fabric channels: each ethernet link supports multiple virtual channels for concurrent transfers; channel assignment affects bandwidth sharing when multiple sockets compete for the same link
  - Fabric initialization sequence and its relationship to MeshDevice creation
  - Source reference: `test_multi_mesh.py`

- **`02_socket_to_fabric_mapping.md`**
  - The path of a D2D packet: sender core -> local NOC -> ethernet sender core -> ethernet link -> ethernet receiver core -> remote NOC -> receiver core. Each hop involves buffering and flow control at the fabric level.
  - How `SocketConnection` endpoints map to physical fabric routes: the sender `MeshCoreCoord` and receiver `MeshCoreCoord` determine which ethernet links carry the data
  - Routing within a mesh vs across meshes: intra-mesh connections use direct chip-to-chip ethernet; inter-mesh connections route through gateway devices
  - Multi-connection topologies: a single `SocketConfig` can specify multiple `SocketConnection` objects, enabling parallel data streams across different fabric links
  - Flow control at the fabric level vs socket level: TT-Fabric provides link-level credit-based flow control (preventing link buffer overflow). Socket-level flow control (bytes_sent/bytes_acked FIFO) is an additional layer ensuring the receiver's buffer is not overwritten before consumption.
  - Bandwidth characteristics: per-link ethernet bandwidth (~100 Gbps / ~12.5 GB/s per direction on Blackhole), aggregate bisection bandwidth in a 4x4 mesh, how multi-connection sockets can utilize parallel paths
  - Latency components: NOC traversal within source mesh + ethernet link latency + NOC traversal within destination mesh + software overhead
  - PCIe topology for H2D/D2H (brief): host CPU connected to chips via PCIe lanes, TLB windows for host-initiated writes, NOC paths for device-initiated PCIe reads/writes. vIOMMU requirement for H2D and D2H sockets.
  - Failure modes: what happens when a fabric link is down or congested

**Addresses questions:** Q11 (TT-Fabric: FABRIC_2D, MGD, ethernet routing)

---

### Chapter 4: H2D and D2H Sockets -- Host-Device Streaming

**Description:** Covers both host-device socket types in a single chapter, highlighting their symmetry and differences: the H2DSocket for streaming data from host to device (with HOST_PUSH and DEVICE_PULL modes), the D2HSocket for streaming data from device to host (with ExternalConfigBuffer), and their shared FIFO flow control model.

**Directory:** `ch04_host_device_sockets/`

**Files:**

- **`index.md`**
  - Chapter navigation and reading-order list with clickable links to all files in this chapter

- **`01_h2d_socket_modes_and_api.md`**
  - **HOST_PUSH mode**: the host CPU initiates writes to device L1 memory via PCIe TLB (Translation Lookaside Buffer) windows.
    - Mechanism: the host maps a region of device L1 into its virtual address space via a TLB entry. Writes to this virtual address are translated by the PCIe controller into writes to device L1.
    - Advantages: low-latency for small transfers (host controls timing), simpler programming model
    - Disadvantages: TLB window size limits per-transfer size, host CPU is busy during the write
  - **DEVICE_PULL mode**: the host writes data to pinned host memory, then the device pulls the data via PCIe NOC reads.
    - Mechanism: the host writes to a pinned (page-locked) memory buffer. The device kernel issues NOC read commands that traverse the NOC to the PCIe controller, which performs DMA reads from the host's pinned memory.
    - Advantages: higher throughput for large transfers, host CPU is freed after writing to the pinned buffer
    - Disadvantages: requires pinned memory allocation, higher latency for small transfers
  - Selecting between modes: HOST_PUSH for low-latency token injection (small pages, interactive latency), DEVICE_PULL for bulk data loading (large tensors, throughput-oriented)
  - **vIOMMU requirement**: both modes require the virtual IOMMU for safe device-initiated PCIe transactions. How to verify vIOMMU status. Error messages when vIOMMU is missing.
  - **API surface**:
    - `set_page_size(page_size_bytes)`: configures the page size for subsequent writes. Must be called before the first `write()`.
    - `write(data_ptr, num_pages)`: writes pages from host buffer into the device FIFO. Non-blocking if FIFO space is available; blocks if FIFO is full.
    - `barrier()`: blocks the host until all previously written data has been consumed by the device kernel
  - Socket destruction: RAII cleanup frees the device-side FIFO buffer and config buffer
  - Source references: `h2d_socket.hpp`

- **`02_d2h_socket_api_and_external_config_buffer.md`**
  - **Transport mechanism**: the device kernel writes to pinned host memory via PCIe NOC writes. The kernel issues NOC write commands that traverse the NOC to the PCIe controller, which performs DMA writes to the host's pinned memory buffer.
  - Why the device writes to host (not host reads from device): device-initiated writes are more efficient for streaming because the device controls timing and NOC writes to PCIe are pipelined
  - Asymmetry with H2D: D2H is always device-push (no host-pull mode)
  - **Host-side API**:
    - `set_page_size(page_size_bytes)`: configures read granularity; must match device kernel's page size
    - `has_data()`: non-blocking check returning true if at least one page is available
    - `pages_available()`: returns the number of complete pages available for reading
    - `read(data_ptr, num_pages, notify_sender)`: reads pages from pinned host buffer into caller's buffer. If `notify_sender` is true, advances `bytes_acked` immediately. If false, defers notification (useful for read-then-process-then-ack patterns).
    - `barrier()`: blocks until all expected data has arrived from the device
    - `discard_pending_pages()`: discards all unread pages, advancing `bytes_acked` to match `bytes_sent`. Useful for error recovery or pipeline flush.
  - **ExternalConfigBuffer**:
    - The problem: by default, D2HSocket allocates its config buffer in L1 using the standard allocator. But some use cases require the config buffer at a specific L1 address -- for example, when the device kernel runs on off-allocator cores.
    - `ExternalConfigBuffer`: a struct that allows the caller to provide a pre-allocated L1 address for the config buffer instead of letting D2HSocket allocate it.
    - When to use: off-allocator cores, shared L1 regions for locality, fine-grained L1 layout control
    - How it works: caller allocates L1 space, creates ExternalConfigBuffer pointing to that address, passes it to the D2HSocket constructor
    - Relationship to sub-device binding
  - Source references: `d2h_socket.hpp`

- **`03_host_device_flow_control.md`**
  - **The shared FIFO model for H2D and D2H**: both use a circular FIFO with `bytes_sent` / `bytes_acked` counters. This file covers both types side-by-side, highlighting the role reversal.
  - **H2D flow control**:
    - Host-side tracking: the host maintains `bytes_sent` -- total bytes written to the FIFO. Each `write()` advances `bytes_sent`.
    - Device-side tracking: the device kernel maintains `bytes_acked` -- total bytes consumed from the FIFO.
    - Blocking condition: host blocks when `bytes_sent - bytes_acked >= fifo_size`
    - How `bytes_acked` is communicated back to the host: the device kernel writes the value to a known location (config buffer) that the host polls
    - HOST_PUSH specifics: host reads `bytes_acked` via PCIe TLB read
    - DEVICE_PULL specifics: device reads `bytes_sent` to detect new data, pulls pages from host memory, advances `bytes_acked`
  - **D2H flow control**:
    - Device-side `bytes_sent`: the device kernel tracks how many bytes it has written. After writing a page, updates `bytes_sent` in the config buffer.
    - Host-side `bytes_acked`: the host tracks how many bytes it has consumed. After reading a page, advances `bytes_acked`. The device kernel reads `bytes_acked` to determine FIFO space availability.
    - The `notify_sender` parameter in `read()`: when true, immediately signals the device; when false, the caller must explicitly advance later
  - **Symmetry table**: who writes bytes_sent, who writes bytes_acked, transport for each counter, transport for data, blocking entity when FIFO is full, blocking entity when FIFO is empty
  - Forward reference to Chapter 5 for the unified flow control deep dive covering all three socket types, FIFO sizing, and performance implications

**Addresses questions:** Q4 (H2D modes, flow control, vIOMMU), Q5 (D2H data path, ExternalConfigBuffer, host read API), Q7 (H2D and D2H flow control)

---

### Chapter 5: Flow Control, Memory Configuration, and Performance

**Description:** Provides a unified treatment of the circular FIFO flow control mechanism across all three socket types, then covers memory placement decisions (L1 vs DRAM), FIFO sizing strategies, PCIe alignment requirements, and socket reuse patterns for production performance.

**Directory:** `ch05_flow_control_and_memory/`

**Files:**

- **`index.md`**
  - Chapter navigation and reading-order list with clickable links to all files in this chapter

- **`01_unified_circular_fifo_model.md`**
  - The circular FIFO as the universal flow control primitive across all three socket types:
    - Capacity: `fifo_size` bytes, divided into pages of `page_size` bytes each
    - Write pointer: `bytes_sent % fifo_size`; Read pointer: `bytes_acked % fifo_size`
    - Available space: `fifo_size - (bytes_sent - bytes_acked)`; Available data: `bytes_sent - bytes_acked`
    - Invariants: `bytes_acked <= bytes_sent` always; `bytes_sent - bytes_acked <= fifo_size` always (enforced by blocking)
  - Why counters are 64-bit and do not wrap: monotonically increasing integers; modulo applied only for buffer offsets
  - Buffer layout: FIFO occupies contiguous region in L1/DRAM (D2D, H2D) or pinned host memory (D2H). Config buffer is a separate, smaller allocation.
  - **Per-socket-type flow control specifics:**
    - **D2D**: sender device core writes to receiver's L1/DRAM FIFO. TT-Fabric provides link-level credit-based flow control; socket-level FIFO is an additional layer. The sender reads `bytes_acked` from the receiver's config buffer via TT-Fabric.
    - **H2D (HOST_PUSH)**: host writes to device L1 FIFO via PCIe TLB. Host reads `bytes_acked` via PCIe TLB read. Device kernel advances `bytes_acked` after consumption.
    - **H2D (DEVICE_PULL)**: host writes to pinned host memory, updates host-side indicator. Device reads, pulls pages from host via PCIe NOC read, advances `bytes_acked`.
    - **D2H**: device writes pages to pinned host memory via PCIe NOC write, updates `bytes_sent`. Host reads pages, advances `bytes_acked`.
  - Blocking behavior: producer blocks when FIFO full; consumer blocks when FIFO empty. Wait mechanisms differ per type (host busy-poll, device NOC read poll, fabric credit return).
  - Backpressure propagation: in a multi-stage pipeline, if a downstream stage's FIFO fills up, the blocked kernel cannot consume from its upstream socket, causing upstream sender to block. This creates end-to-end backpressure.
  - Deadlock avoidance: scenarios where circular dependencies between sockets can cause deadlock. Mitigation: asymmetric FIFO sizing, pipeline-stage ordering.

- **`02_memory_placement_and_fifo_sizing.md`**
  - **L1 vs DRAM buffer trade-offs:**
    - L1: lowest latency access, no DRAM bank contention, direct core-local access. Limited capacity (~1.5MB per core on Blackhole, shared with compute CBs). Best for: small FIFOs, latency-sensitive paths (token injection, loopback), cores with low compute memory pressure.
    - DRAM: much larger capacity (12GB shared DRAM per device), does not consume L1. Higher access latency, potential bank contention. Best for: large FIFOs, bulk data transfer, throughput-oriented paths.
    - Mixed strategies: L1 for control metadata, DRAM for data buffers
    - Impact on compute: L1 used by socket FIFOs is L1 not available for compute circular buffers
  - **Sub-device binding**: using `sender_sub_device_id` / `receiver_sub_device_id` in SocketMemoryConfig
  - **FIFO sizing methodology:**
    - Determine page size based on data being transferred (tile row, full tile, tensor slice)
    - Determine pipeline depth: how many pages need to be in-flight to hide transfer latency
    - FIFO size = page_size * (pipeline_depth + 1), rounded up to alignment requirements
    - Rule of thumb for D2D: 2-4 pages for low-latency, 8-16 pages for high throughput
    - Rule of thumb for H2D/D2H: 4-8 pages to hide PCIe round-trip latency
    - Double-buffering pattern: FIFO with exactly 2 pages allows overlap
    - Optimal sizing guideline: 2x-4x the product of `page_size * ceil(round_trip_latency / transfer_time_per_page)`
  - **PCIe alignment requirements for H2D and D2H:**
    - Page size should be a multiple of 16 bytes (NOC alignment) and ideally a multiple of PCIe TLB page size (4KB or 16KB)
    - FIFO size must be a multiple of page_size
    - D2D alignment: NOC transactions require 16-byte alignment; tile-sized pages naturally satisfy this
    - Common pitfalls: page_size of 1 byte, fifo_size smaller than page_size, fifo_size equal to page_size
  - **Socket reuse patterns:**
    - A socket created once can be used for multiple send/recv cycles. After `barrier()`, the FIFO is empty and ready.
    - Benefits: avoids allocation/deallocation overhead, fabric route re-establishment, memory re-pinning, descriptor re-export
    - When recreation is necessary: changed configuration, device reset, topology change
    - Socket pool pattern: create all sockets during initialization and reuse across iterations
    - Warm-up: first transfer may be slower due to one-time setup

**Addresses questions:** Q7 (unified flow control deep dive), Q10 (memory and performance: L1 vs DRAM, FIFO sizing, PCIe alignment, socket reuse)

---

### Chapter 6: Cross-Process Socket Sharing and Distributed Context

**Description:** Provides a unified treatment of the cross-process socket descriptor exchange mechanism spanning all three socket types, covering the export/connect protocol, flatbuffer serialization, `/dev/shm` exchange medium, owner/connector lifecycle management, security considerations, and failure modes.

**Directory:** `ch06_cross_process/`

**Files:**

- **`index.md`**
  - Chapter navigation and reading-order list with clickable links to all files in this chapter

- **`01_export_connect_protocol.md`**
  - The cross-process problem: production Tenstorrent deployments run multiple processes -- one per mesh or per device group. These processes need to communicate via sockets without sharing a MetalContext or device handle.
  - `export_descriptor(socket_id: str)`:
    - Serializes the socket's internal state (FIFO addresses, buffer layout, control metadata, device identity) into a flatbuffer binary
    - Writes the flatbuffer to `/dev/shm/socket_id` (POSIX shared memory)
    - The socket_id is a string identifier chosen by the caller -- typically encodes pipeline stage, socket role, and instance number
  - `H2DSocket::connect(socket_id: str)` / `D2HSocket::connect(socket_id: str)`:
    - Reads the flatbuffer from `/dev/shm/socket_id`
    - Reconstructs a socket handle from the serialized state
    - The resulting handle can perform write/read operations without owning the underlying device or MetalContext
    - For H2D, the connector uses `PCIeCoreWriter` to write directly to device memory without MetalContext ownership
    - For D2H, the connector maps the pinned host memory region
  - Flatbuffer descriptor contents per socket type:
    - D2D: sender/receiver MeshCoreCoords, TT-Fabric route identifiers, FIFO addresses in L1/DRAM
    - H2D: host buffer address, device L1 FIFO address (HOST_PUSH) or pinned host memory address (DEVICE_PULL), vIOMMU mapping info
    - D2H: pinned host buffer address, device-side ExternalConfigBuffer address, FIFO layout
  - Size and overhead: descriptors are typically hundreds of bytes; serialization/deserialization is microseconds
  - The export/connect handshake: the owner must export before the connector connects. Synchronization is the caller's responsibility (e.g., distributed context barrier or file-based readiness signal).
  - Source references: `h2d_socket.hpp`, `d2h_socket.hpp`

- **`02_owner_connector_lifecycle.md`**
  - **Owner responsibilities:** creates the socket (allocates device-side buffers, initializes flow control), exports the descriptor, maintains the socket for the duration of usage, cleans up on destruction
  - **Connector responsibilities:** connects to the exported descriptor, performs read/write operations, does not manage device resources, disconnects (drops the handle) when done
  - Lifecycle ordering constraints:
    1. Owner creates socket and exports descriptor BEFORE connector attempts to connect
    2. Connector connects and performs operations WHILE owner keeps the socket alive
    3. Connector finishes and drops its handle BEFORE owner destroys the socket
  - What the connector cannot do: allocate new buffers, resize the FIFO, change the page size, modify device-side control structures
  - **Failure modes:**
    - Connector connects before owner exports: file not found in `/dev/shm` -> error
    - Owner destroys socket while connector is active: connector operations fail with stale handle errors (dangling memory references)
    - Multiple connectors to the same H2D socket: race to write into same FIFO -- undefined behavior unless externally coordinated
    - Multi-consumer D2H: race conditions on `bytes_acked` unless externally coordinated
    - Process crash without cleanup: stale `/dev/shm` files remain; subsequent runs must handle or clean up stale descriptors

- **`03_security_and_operational_considerations.md`**
  - **Shared memory permissions:** `/dev/shm` is accessible to all processes on the same host with matching UID/GID. Socket descriptors contain raw physical memory addresses -- there is no authentication beyond filesystem permissions. In multi-tenant environments, socket descriptors may leak device access to unauthorized processes.
  - Mitigation strategies: restrict `/dev/shm/` permissions, use per-user directories, employ abstract Unix domain sockets
  - **vIOMMU and cross-process:** device-initiated PCIe transactions require that IOMMU mapping is valid for the physical pages. Cross-process sharing of pinned memory may require shared IOMMU contexts or per-process IOMMU mappings.
  - **`/dev/shm/` lifecycle:** tmpfs (RAM-backed filesystem); files persist until explicitly deleted or system reboot. Best practice: use supervisor processes or signal handlers for cleanup.
  - **Multi-host considerations:** `/dev/shm/` is local to each host. For multi-host deployments, D2D MeshSockets with the distributed context handle inter-host networking via TT-Fabric. H2D/D2H export/connect only works between processes on the same physical host.
  - **Debugging cross-process sockets:** how to inspect `/dev/shm/` files, verify flatbuffer contents, diagnose connection failures, trace data flow across process boundaries

**Addresses questions:** Q6 (cross-process export/connect, flatbuffer, owner/connector lifecycle)

---

### Chapter 7: Production Usage -- DeepSeek V3 Multi-Host Pipeline

**Description:** Examines how all three socket types are composed in DeepSeek V3's production multi-host inference pipeline, covering the PipelineBlock architecture, the five stage configurations, SocketInterface/HostInterface abstractions, loopback configurations, and an end-to-end inference walkthrough with troubleshooting guidance.

**Directory:** `ch07_deepseek_pipeline/`

**Files:**

- **`index.md`**
  - Chapter navigation and reading-order list with clickable links to all files in this chapter

- **`01_pipeline_architecture_and_stages.md`**
  - DeepSeek V3's multi-host pipeline: a distributed inference pipeline spanning multiple Galaxy systems (each Galaxy = 2 mesh devices of 4x4 Blackhole chips). Tokens flow from host -> first stage -> middle stages -> last stage -> host.
  - The PipelineBlock abstraction: a composable unit that encapsulates one pipeline stage's compute, input sockets, and output sockets.
  - `StageMetadata`: per-stage configuration describing rank, mesh_id, and which operations run
  - `PipelineConfigEntry`: maps stage metadata to specific mesh/rank assignments with entry/exit coordinates
  - The five stage configurations:
    1. **First stage (with downstream D2D)**: H2DSocket for token input, D2D exit socket, optional loopback entry, optional D2H for debugging
    2. **Middle stage**: D2D entry socket, D2D exit socket, no host interface
    3. **Last stage with loopback**: D2D entry, D2D loopback to first stage, D2H output
    4. **Last stage without loopback**: D2D entry, D2H output
    5. **Combined H2D + D2H stage**: single-mesh, H2D input, D2H output, no D2D
  - Data flow through the pipeline: host injects tokens (H2D) -> first stage computes and forwards (D2D) -> middle stages compute and forward (D2D) -> last stage computes and returns (D2H or D2D loopback)
  - The autoregressive loop: last stage output (sampled token) loops back to first stage via D2D socket for the next decode step
  - Source reference: `pipeline_block/op.py`

- **`02_socket_interface_and_host_interface.md`**
  - `SocketInterface`: the abstraction managing D2D MeshSocket lifecycle within a pipeline stage.
    - Encapsulates socket configuration, creation, teardown
    - Provides validated send/recv with tensor shape/dtype checking
    - Integration with the pipeline's execution loop
  - `HostInterface`: the abstraction managing H2D and D2H socket lifecycles.
    - H2DSocket for token injection (host writes tokenized input)
    - D2HSocket for result extraction (host reads logits or generated tokens)
    - Export/connect for cross-process usage
    - Page size management and barrier synchronization
  - `HostIoPlacement`: specifies which cores handle H2D, D2H, forward D2D, and loopback D2D operations. Separates socket I/O from compute cores to prevent resource contention.
  - How PipelineBlock composes these: a PipelineBlock has optional entry_socket, exit_socket, and host_interface. The combination determines stage type.
  - The `op.py` dispatch: PipelineBlock's `__call__` orchestrates recv -> H2D read -> compute -> send -> D2H write
  - `LoopbackConfig`: three modes -- `fabric` (D2D loopback through TT-Fabric, lowest latency), `host` (D2H then H2D through host memory, allows host post-processing), `none` (single-pass)
  - Source reference: `pipeline_block/op.py`

- **`03_end_to_end_inference_walkthrough.md`**
  - **Complete DeepSeek V3 inference walkthrough using sockets:**
    1. **Initialization**: create MeshDevices for each mesh, initialize FABRIC_2D, create DistributedContext
    2. **Socket setup**: create H2DSocket for input, D2D MeshSockets for inter-mesh stages, D2HSocket for output, loopback socket for decode iterations
    3. **Pipeline construction**: PipelineBlock configured with 5 stages, StageMetadata per stage, HostIoPlacement per mesh
    4. **Prefill phase**: host pushes prompt tokens via H2D -> pipeline processes through all stages -> KV cache populated
    5. **Decode phase (iterative)**: loopback socket carries previous token -> embedding -> transformer layers via D2D -> logit output -> sampling -> loopback for next iteration
    6. **Result extraction**: final tokens read from D2HSocket by host process
  - Socket wiring for a complete pipeline: example with 4-mesh pipeline showing all D2D, H2D, D2H connections and their SocketConfig consistency requirements
  - Multi-host wiring: when stages span multiple hosts, the distributed_context handles inter-host communication. Each host's process creates socket endpoints based on rank.
  - Multi-process coordination: processes synchronize at barriers before starting inference. Socket export/connect handles inter-process endpoint setup.
  - The initialization sequence: (1) all processes initialize fabric, (2) open mesh devices, (3) create sockets with matching configs, (4) owners export H2D/D2H descriptors, (5) connectors connect, (6) pipeline begins
  - Performance characteristics: where time is spent -- compute dominates; socket transfer adds overhead proportional to tensor size and bandwidth. Pipeline parallelism hides inter-mesh transfer latency.
  - Source reference: `pipeline_block/op.py`

- **`04_troubleshooting_and_debugging.md`**
  - **Hang / Deadlock**: most common cause is FIFO full with no consumer, or FIFO empty with no producer. Debug by checking bytes_sent/bytes_acked counters. If `bytes_sent - bytes_acked == fifo_size`, producer blocked. If `bytes_sent == bytes_acked`, consumer blocked.
  - **Data corruption**: page_size mismatch between sender and receiver, FIFO size not a multiple of page_size, PCIe alignment violated, multiple writers racing on same FIFO
  - **Connection refused / descriptor not found**: connector attempts to connect before owner has exported. Fix: add synchronization between export and connect.
  - **vIOMMU errors**: H2D and D2H sockets fail if vIOMMU is not enabled. Error messages reference IOMMU mapping failures. Fix: enable vIOMMU in BIOS and verify with `dmesg | grep -i iommu`.
  - **Fabric not initialized**: D2D MeshSocket creation fails if `set_fabric_config()` was not called. Fix: call `set_fabric_config(FABRIC_2D)` before creating any MeshSocket.
  - **Rank mismatch**: in SPMD, if sender_rank/receiver_rank do not match actual process ranks, socket endpoint type may be wrong. Debug by logging `get_socket_endpoint_type()` at both processes.
  - **Mismatched SocketConfig**: different FIFO sizes, wrong core coordinates between sender and receiver processes. Verify config objects are identical.
  - **Tensor shape mismatch**: send and receive buffers have incompatible shapes or layouts. Verify pre-allocation matches expected dimensions.
  - **Performance diagnosis**: if throughput is lower than expected, check FIFO utilization (frequently full or empty?), page size (too small?), L1 vs DRAM choice, fabric link utilization. Use profiler to identify socket-related stalls.
  - **Socket creation order**: create all sockets before starting any computation. Socket creation involves TT-Fabric route setup and memory allocation that should not interleave with kernel dispatch.
  - **Teardown ordering**: stop computation -> drain sockets (barrier) -> destroy sockets -> close mesh devices. Incorrect ordering leads to hung processes or data corruption.
  - **Stale `/dev/shm` descriptors**: from previous runs or crashed processes. Clean up before new runs.
  - **Comparison with CCL**: sockets are point-to-point with explicit flow control; CCL operations (all_gather, reduce_scatter) are collective with implicit synchronization. Sockets for pipeline-parallel communication; CCL for data-parallel or tensor-parallel.

**Addresses questions:** Q9 (DeepSeek V3 PipelineBlock, five stage configurations, H2D/D2D/D2H integration)

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| **MeshSocket / D2D socket** | A `MeshSocket` instance for transferring tensors between cores on different mesh devices via TT-Fabric. Both endpoints are device cores. |
| **H2DSocket** | A socket for streaming data from host CPU memory to device L1. The host is the sender; the device kernel is the receiver. Supports HOST_PUSH and DEVICE_PULL modes. |
| **D2HSocket** | A socket for streaming data from device cores to host CPU memory. The device kernel is the sender; the host is the receiver. Always operates in device-push mode. |
| **MeshCoreCoord** | A composite coordinate identifying a specific core: `device_coord` (which device in the mesh, as a 2D MeshCoordinate) + `core_coord` (which Tensix core on that device, as CoreCoord). Uniquely identifies a core in a multi-mesh system. |
| **SocketConnection** | A 1:1 sender-to-receiver core pairing within a socket. Each connection defines one unidirectional data channel from a sender MeshCoreCoord to a receiver MeshCoreCoord. |
| **SocketMemoryConfig** | Configuration for a socket's memory allocation: `buffer_type` (L1 or DRAM), `fifo_size`, and optional `sub_device_id` for sub-device binding. |
| **SocketConfig** | The top-level configuration composing `connections` (list of SocketConnection), `memory_config` (SocketMemoryConfig), endpoint identity (`sender_rank`/`receiver_rank` or `sender_mesh_id`/`receiver_mesh_id`), and optional `distributed_context`. |
| **FIFO** | The circular buffer used for flow-controlled data transfer. Has a fixed `fifo_size` and transfers data in `page_size` units. |
| **bytes_sent** | Monotonically increasing 64-bit counter tracking total bytes written by the producer since socket creation. |
| **bytes_acked** | Monotonically increasing 64-bit counter tracking total bytes consumed by the consumer since socket creation. |
| **page_size** | The granularity of data transfer. All reads and writes operate on integer numbers of pages. Set via `set_page_size()`. |
| **send_async** | Non-blocking D2D socket operation that enqueues a tensor for transmission via TT-Fabric. Returns immediately; data transfer proceeds asynchronously. |
| **recv_async** | Non-blocking D2D socket operation that registers a pre-allocated tensor as the receive target for incoming data. |
| **TT-Fabric** | The packet-switched ethernet network connecting Tenstorrent chips within and between meshes. Handles routing, link-level flow control, and packet delivery. |
| **FABRIC_2D** | A fabric configuration mode using 2D mesh routing with dimension-ordered routing (X-then-Y). Must be set via `set_fabric_config(FABRIC_2D)` before creating D2D sockets. |
| **MGD** | Mesh Gateway Device: a routing bridge for multi-Galaxy systems that connects separate fabric networks across physical boundaries. |
| **HOST_PUSH** | H2D transfer mode where the host CPU writes to device L1 via PCIe TLB windows. |
| **DEVICE_PULL** | H2D transfer mode where the device pulls data from pinned host memory via PCIe NOC reads. Requires vIOMMU. |
| **vIOMMU** | Virtual I/O Memory Management Unit: hardware feature required for H2D/D2H sockets that translates device DMA addresses to physical host memory addresses. |
| **pinned memory** | Host system memory that is locked (non-pageable) so that device-initiated PCIe transactions always find valid physical addresses. Required for D2H and DEVICE_PULL H2D. |
| **export_descriptor** | Method that serializes a socket's state to `/dev/shm` as a flatbuffer, enabling another process to connect to the socket without owning the device. Called by the socket owner. |
| **connect** | Static method that reads an exported descriptor from `/dev/shm` and creates a socket handle for read/write operations. Called by the connector process. |
| **ExternalConfigBuffer** | A D2HSocket mechanism allowing the caller to provide a pre-allocated L1 address for the config buffer, bypassing the standard allocator. Used for off-allocator cores. |
| **discard_pending_pages** | D2HSocket method that drops all unread data and resets FIFO state by advancing `bytes_acked` to match `bytes_sent`. Used for error recovery or pipeline flush. |
| **create_socket_pair** | Static factory method that creates a matched sender/receiver MeshSocket pair within a single process, bypassing distributed context and cross-process machinery. |
| **DistributedContext** | The multi-process coordination layer (`tt::distributed::open_distributed_context()`) providing rank assignment, barriers, and endpoint matching. Required for multi-process socket usage. |
| **SPMD** | Single Program, Multiple Data: a programming model where all processes run identical code but branch on rank for endpoint-specific behavior. |
| **rank** | Process identity in an SPMD deployment; used with `sender_rank`/`receiver_rank` in SocketConfig for role determination. |
| **PipelineBlock** | DeepSeek V3's abstraction for a pipeline stage: encapsulates compute + optional entry/exit D2D sockets + optional H2D/D2H host interface. |
| **SocketInterface** | DeepSeek V3's abstraction managing the lifecycle of D2D MeshSockets within a pipeline stage. Provides validated send/recv. |
| **HostInterface** | DeepSeek V3's abstraction managing H2D and D2H socket lifecycles for host-device data exchange. Provides push_input/pull_output. |
| **StageMetadata** | Per-stage configuration in PipelineBlock specifying rank and mesh_id for SPMD execution. |
| **PipelineConfigEntry** | Per-stage entry/exit coordinate specification in PipelineBlock, defining where data enters and leaves each pipeline stage. |
| **HostIoPlacement** | Configuration specifying which device cores handle H2D, D2H, forward D2D, and loopback D2D operations within a PipelineBlock stage. |
| **LoopbackConfig** | PipelineBlock configuration for iterative workloads: `fabric` (D2D loopback), `host` (D2H-then-H2D loopback), or `none` (single pass). |
| **PCIeCoreWriter** | A low-level interface for writing to device L1 via PCIe without a full MetalContext. Used by cross-process H2D socket connectors. |
| **barrier** | A synchronization operation that blocks until all pending socket transfers complete. |
| **backpressure** | The effect where a slow consumer causes the FIFO to fill, blocking the producer, which cascades upstream through the pipeline. |
| **mesh device** | A connected grid of Tenstorrent chips (e.g., 4x4 Blackhole chips) managed as a single MeshDevice. |
| **sub-device** | A partition of a mesh device used for independent scheduling. Socket memory can be bound to specific sub-devices. |
| **Galaxy** | Tenstorrent's multi-chip system with 32 Blackhole chips arranged in a 4x8 mesh topology. |

### Research Questions Referenced

| # | Question | Primary Chapter(s) | Primary File(s) |
|---|----------|--------------------|-----------------|
| Q1 | What are the three socket types and how do they differ in purpose, direction, transport, and API? | Ch 1 | 01_socket_taxonomy_and_purpose |
| Q2 | How does D2D socket configuration work (MeshCoreCoord, SocketConnection, SocketMemoryConfig, SocketConfig)? | Ch 2 | 01_configuration_hierarchy |
| Q3 | How do send_async and recv_async operations work (non-blocking dispatch, pre-allocation, synchronization)? | Ch 2 | 02_send_recv_async_operations |
| Q4 | How does the H2DSocket work (HOST_PUSH vs DEVICE_PULL, flow control, vIOMMU)? | Ch 4 | 01_h2d_socket_modes_and_api, 03_host_device_flow_control |
| Q5 | How does the D2HSocket work (device writes, host reads, ExternalConfigBuffer, discard_pending_pages)? | Ch 4 | 02_d2h_socket_api_and_external_config_buffer, 03_host_device_flow_control |
| Q6 | How does cross-process socket sharing work (export/connect, flatbuffer, owner/connector lifecycle)? | Ch 6 | All 3 files |
| Q7 | How does flow control work in each socket type (circular FIFO, bytes_sent/bytes_acked, blocking)? | Ch 4 (per-type), Ch 5 (unified) | Ch4: 03_host_device_flow_control; Ch5: 01_unified_circular_fifo_model |
| Q8 | How do intra-process sockets work (create_socket_pair, bypass distributed context)? | Ch 2 | 03_spmd_and_intra_process |
| Q9 | How are sockets used in production (DeepSeek V3 PipelineBlock, five stage configs)? | Ch 7 | All 4 content files |
| Q10 | What are memory and performance considerations (L1 vs DRAM, FIFO sizing, PCIe alignment, reuse)? | Ch 5 | 02_memory_placement_and_fifo_sizing |
| Q11 | How do sockets interact with TT-Fabric (FABRIC_2D, MGD, ethernet routing)? | Ch 3 | Both content files |
| Q12 | What is the SPMD programming model for sockets (rank-aware branching, sender_rank/receiver_rank)? | Ch 2 | 03_spmd_and_intra_process |

### File Path Notation

- TT-Metal/TTNN paths are relative to the tt-metal repository root: `ttnn/cpp/ttnn/operations/experimental/socket/`
- Socket header paths: `mesh_socket.hpp`, `h2d_socket.hpp`, `d2h_socket.hpp`
- Test paths: `tests/ttnn/distributed/test_multi_mesh.py`
- DeepSeek V3 paths are relative to the model repository root: `pipeline_block/op.py`
- All file paths in the guide body are relative to the relevant repository root

### Code Formatting

- C++ code blocks use `cpp` syntax highlighting; Python code blocks use `python`
- C++ class and struct names use `PascalCase`: `MeshSocket`, `SocketConfig`, `SocketConnection`
- C++ enum values use `UPPER_SNAKE_CASE`: `FABRIC_2D`, `HOST_PUSH`, `DEVICE_PULL`
- TTNN Python API calls use dotted notation: `ttnn.experimental.send_async`, `ttnn.experimental.recv_async`
- Configuration fields use `snake_case`: `fifo_size`, `sender_rank`, `receiver_rank`, `bytes_sent`, `bytes_acked`
- System paths use their full form: `/dev/shm/socket_descriptor_name`
- Byte sizes are written with units: `4KB`, `16KB`, `1.5MB`, `12GB`

### Formatting Rules

- Each file begins with a one-paragraph summary stating its purpose and what the reader will learn
- Each file ends with a "Key Takeaways" bulleted list (3-5 items)
- Each chapter includes an `index.md` navigation file with clickable links to all files in the chapter
- Code listings that reference real files include a header comment with the file path
- Diagrams use ASCII art for data flow, FIFO state, pipeline topology, and fabric routing illustrations
- Flow control state is illustrated with FIFO diagrams showing bytes_sent, bytes_acked, and buffer occupancy
- Comparison tables are used for socket type comparisons, mode comparisons, and configuration option comparisons; always include a header row
- When describing a concept that applies differently across socket types, use explicit per-type sub-headings (D2D / H2D / D2H) rather than inline parentheticals
- Warning/caution callouts use blockquote format with `**Warning:**` prefix
- Cross-references use the format `[Chapter N, File M](../chNN_dir/MM_file.md)`
- Performance numbers are annotated as approximate with hardware/configuration context

---

## Cross-Chapter Dependencies

```
Chapter 1 (Socket Overview)
  -- foundational, no dependencies
  |
  +---> Chapter 2 (D2D MeshSocket Config + Operations + SPMD)
  |         -- depends on Ch 1 for socket taxonomy and common abstractions
  |
  +---> Chapter 3 (TT-Fabric and Topology)
  |         -- depends on Ch 1 for transport overview
  |         -- depends on Ch 2 for SocketConnection, MeshCoreCoord context
  |
  +---> Chapter 4 (H2D and D2H Sockets)
  |         -- depends on Ch 1 for socket taxonomy and FIFO model preview
  |         -- independent of Ch 2 and Ch 3 (different socket types)
  |
  +---> Chapter 5 (Flow Control and Memory)
            -- depends on Ch 2 for D2D flow control specifics
            -- depends on Ch 4 for H2D/D2H flow control specifics
            -- depends on Ch 3 for fabric-level flow control context
            |
            +---> Chapter 6 (Cross-Process Sharing)
            |         -- depends on Ch 4 for H2D/D2H socket APIs
            |         -- depends on Ch 2 for D2D distributed context
            |
            +---> Chapter 7 (DeepSeek V3 Pipeline)
                      -- depends on Ch 2 for D2D operations
                      -- depends on Ch 4 for H2D/D2H operations
                      -- depends on Ch 5 for FIFO sizing decisions
                      -- depends on Ch 6 for cross-process deployment
```

**Detailed dependency edges:**

| Chapter | Depends On | Nature of Dependency |
|---------|-----------|---------------------|
| **Ch 1** | (none) | Foundational. Introduces the three socket types, their purposes, transport mechanisms, and common abstractions (FIFO, SPMD preview, lifecycle). All subsequent chapters reference the taxonomy and terminology defined here. |
| **Ch 2** | Ch 1 | Uses Ch 1's socket taxonomy and common abstractions. The D2D overview from Ch 1 motivates the configuration model. SPMD and intra-process concepts build on Ch 1's preview. |
| **Ch 3** | Ch 1, Ch 2 | Extends Ch 1's transport overview with deep fabric architecture. References SocketConnection and MeshCoreCoord from Ch 2 to explain fabric routing for specific socket configurations. |
| **Ch 4** | Ch 1 | Uses Ch 1's socket taxonomy and FIFO model preview. Independent of Ch 2 and Ch 3 (different socket types). The shared flow control file references but does not require Ch 5. |
| **Ch 5** | Ch 2, Ch 3, Ch 4 | Unifies flow control from all socket types (D2D from Ch 2, H2D/D2H from Ch 4, fabric-level from Ch 3). Memory placement references SocketMemoryConfig from Ch 2 and pinned memory from Ch 4. |
| **Ch 6** | Ch 2, Ch 4 | Cross-process export/connect applies to all socket types. Flatbuffer serializes structures from Ch 2 (SocketConfig) and Ch 4 (H2D/D2H config). Distributed context from Ch 2 provides the multi-process coordination layer. |
| **Ch 7** | Ch 2, Ch 4, Ch 5, Ch 6 | PipelineBlock composes D2D sockets (Ch 2), H2D/D2H sockets (Ch 4), applies flow control and memory sizing (Ch 5), uses cross-process sharing for multi-process deployment (Ch 6). Troubleshooting section synthesizes failure modes from all chapters. |

**Key cross-file references:**

- Ch 1 File 02 previews the FIFO model and SPMD -> Ch 5 File 01 provides the deep unified treatment -> Ch 4 File 03 provides per-type detail for H2D/D2H
- Ch 1 File 01 introduces TT-Fabric as D2D transport -> Ch 3 File 02 details fabric routing and packet path -> Ch 7 File 03 shows fabric in production pipeline context
- Ch 2 File 01 defines SocketConfig and SocketConnection -> Ch 3 File 02 explains how connections map to fabric routes -> Ch 7 File 03 shows how SocketConfigs are wired across pipeline stages
- Ch 4 Files 01-02 introduce H2D/D2H socket APIs with brief flow control -> Ch 5 File 01 unifies and deepens the model -> Ch 5 File 02 provides sizing guidelines
- Ch 2 File 03 covers SPMD + intra-process -> Ch 6 File 01 covers cross-process as the production alternative -> Ch 7 File 03 shows both in DeepSeek V3 context

### Reading Order Recommendations

1. **Sequential (recommended for first read):** Chapters 1 through 7 in order. Each chapter builds on the previous, with cross-cutting chapters (5, 6) placed after all individual socket types are covered.

2. **D2D-focused (inter-mesh pipeline developers):** Ch 1 -> Ch 2 -> Ch 3 -> Ch 5 (File 01 only) -> Ch 7. Covers MeshSocket configuration, operations, TT-Fabric routing, and production pipeline usage.

3. **Host-device I/O focused (application integration developers):** Ch 1 -> Ch 4 -> Ch 5 -> Ch 6 -> Ch 7. Covers H2D/D2H sockets, flow control, cross-process sharing, and pipeline integration.

4. **Production deployment focus (operations engineers):** Ch 1 (skim) -> Ch 6 -> Ch 7 (Files 03-04). For engineers deploying multi-process SPMD systems who need cross-process sharing and pipeline integration.

5. **Performance engineering focus:** Ch 1 -> Ch 5 -> Ch 7 (File 04). For engineers tuning FIFO sizes, memory placement, and pipeline throughput. Troubleshooting section provides diagnostic guidance.

**Independently readable sections:**
- **Chapters 1-4** can be read sequentially as a complete API reference for all three socket types and TT-Fabric without requiring cross-cutting chapters 5-7.
- **Chapter 7** can be read after Chapters 1-4 for a production case study, skipping the flow control deep dive in Chapter 5.
- **Chapter 7 File 04** (troubleshooting) can be referenced independently as a debugging guide.
