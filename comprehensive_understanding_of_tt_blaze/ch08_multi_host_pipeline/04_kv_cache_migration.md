# 08 -- Multi-Host Pipeline Infrastructure: KV Cache Migration

[<< Previous: Scheduling Algorithm](03_scheduling_algorithm.md)

---

## 8.29 Overview

The `disaggregation/migration/` directory contains a C++ library for moving KV-cache data between disaggregated pipeline endpoints. In a disaggregated deployment, one set of hosts runs prefill and a separate set runs decode. After prefill completes, the populated KV cache must be migrated from the prefill host's device DRAM to the decode host's device DRAM before decode can begin.

The library lives in `namespace tt::tt_metal::experimental::disaggregation` and is structured in layers:

```
disaggregation/migration/
  include/
    migration_layer.hpp          -- MigrationLayerEndpoint (top-level per-rank)
    migration_frontend.hpp       -- MigrationFrontend (protocol coordinator)
    migration_types.hpp          -- EndpointId, MigrationUUID, MpiRank, etc.
    migration_message.hpp        -- MigrationMessageHeader, BatchMessageHeader
    endpoint_transport.hpp       -- EndpointTransport (control plane abstraction)
    sender_backend.hpp           -- SenderBackend (abstract send interface)
    receiver_backend.hpp         -- ReceiverBackend (abstract receive interface)
    dcn_sender_backend.hpp       -- DcnSenderBackend (MPI-based sender)
    dcn_receiver_backend.hpp     -- DcnReceiverBackend (MPI-based receiver)
    device_reader.hpp            -- DeviceReader (abstract device-to-host read)
    device_writer.hpp            -- DeviceWriter (abstract host-to-device write)
    device_io.hpp                -- UmdDeviceReader/Writer, SocketDeviceReader/Writer
    multi_device_reader.hpp      -- MultiDeviceReader (routes by FabricNodeId)
    multi_device_writer.hpp      -- MultiDeviceWriter (routes by FabricNodeId)
    table_splitter.hpp           -- TableSplitter (abstract)
    dcn_table_splitter.hpp       -- DcnTableSplitter (split by host)
    table_preprocessor.hpp       -- TablePreprocessFn (dedup replicas)
    noc_addr.hpp                 -- NOC address encoding utilities
    spsc_queue.hpp               -- Lock-free SPSC queue
    shared_spsc_queue.hpp        -- Shared-memory SPSC queue
    same_producer_consumer_circular_buffer.hpp
  src/
    migration_layer.cpp          -- MigrationLayerEndpoint implementation
    migration_frontend.cpp       -- MigrationFrontend implementation
    dcn_sender_backend.cpp       -- DcnSenderBackend implementation
    dcn_receiver_backend.cpp     -- DcnReceiverBackend implementation
    dcn_table_splitter.cpp       -- DcnTableSplitter implementation
    table_preprocessor.cpp       -- dedup_first_device implementation
```

---

## 8.30 Three-Tier KV Model

The pipeline manager architecture document defines a three-tier KV cache residency model that frames the migration library's role:

| Tier | Description | Medium | Latency |
|---|---|---|---|
| Tier-0 (Hot) | KV resident on accelerator DRAM, directly used by decode | Device DRAM / HBM | Lowest |
| Tier-1 (Warm) | KV in host memory, must migrate before decode resumes | Host DRAM | Medium |
| Tier-2 (Cold) | KV persisted for long-idle sessions | SSD / object storage | Highest |

The migration library implements the Tier-0-to-Tier-0 path: moving KV data from one endpoint's device DRAM to another endpoint's device DRAM across the data center network. This is the path used in disaggregated prefill-decode separation, where the prefill endpoint's devices hold the freshly computed KV cache that must land on the decode endpoint's devices.

---

## 8.31 Architecture

The library is layered into four levels:

```
MigrationLayerEndpoint    (per-rank orchestrator)
        |
MigrationFrontend         (protocol coordinator: start/migrate/finish/ACK)
        |
   +----+----+
   |         |
SenderBackend   ReceiverBackend    (abstract transport backends)
   |         |
DcnSenderBackend  DcnReceiverBackend  (MPI-based implementations)
   |         |
MultiDeviceReader  MultiDeviceWriter  (device I/O routing by FabricNodeId)
   |         |
DeviceReader    DeviceWriter          (single-device I/O abstraction)
```

### MigrationLayerEndpoint

The top-level class owned by each MPI rank. An endpoint groups multiple ranks (one master plus zero or more subordinates) that collectively own a set of TT devices. The master coordinates with remote endpoints and fans out migration requests to subordinates.

```cpp
class MigrationLayerEndpoint {
public:
    MigrationLayerEndpoint(
        MpiRank rank,
        EndpointId my_endpoint_id,
        const EndpointGrouping& endpoint_grouping,
        std::unique_ptr<SenderBackend> sender,
        std::unique_ptr<ReceiverBackend> receiver,
        std::unique_ptr<EndpointTransport> transport);

    void initialize_my_kv_chunk_mapping_table(
        std::shared_ptr<KvChunkAddressTable> local_table);
    void connect_to_remote_endpoint(
        const EndpointGrouping& remote_endpoint_grouping);

    MigrationUUID migrate_layer(EndpointId remote, uint32_t layer,
        uint32_t pos_start, uint32_t pos_end,
        uint32_t src_slot, uint32_t dst_slot);
    MigrationUUID migrate_slot(EndpointId remote,
        uint32_t pos_start, uint32_t pos_end,
        uint32_t src_slot, uint32_t dst_slot);

    void block_until_migration_received();
    void listen_for_migration();
};
```

### MigrationFrontend

The protocol coordinator that drives the start/migrate/finish sequence and manages completion callbacks:

```cpp
class MigrationFrontend {
public:
    void register_backend(
        std::unique_ptr<SenderBackend> sender,
        std::unique_ptr<ReceiverBackend> receiver);
    void init_sender(
        std::shared_ptr<const KvChunkAddressTable> local_table,
        const std::unordered_map<EndpointId,
            std::shared_ptr<const KvChunkAddressTable>>& remote_tables,
        const EndpointHostnameToRankMap& endpoint_hostname_to_rank);

    void start(EndpointId destination, MigrationUUID uuid);
    void migrate_slot(EndpointId destination,
        uint32_t layer, uint32_t position_start, uint32_t position_end,
        uint32_t src_slot, uint32_t dst_slot, MigrationUUID uuid);
    void finish(MigrationUUID uuid);
    void on_completion(MigrationUUID uuid, CompletionCallback cb);
};
```

### EndpointGrouping

Describes the set of MPI ranks that form one migration endpoint:

```cpp
struct EndpointGrouping {
    std::vector<MpiRank> ranks;
    MpiRank master;
    EndpointId endpoint_id;
    std::vector<SubordinateInfo> subordinate_infos;
};
```

Each `SubordinateInfo` carries the subordinate's hostname and the `FabricNodeId`s of the TT devices physically present on that host. This information is used by the table splitter to assign KV chunks to the correct subordinate for reading/writing.

---

## 8.32 Transport Layer

### EndpointTransport

The abstract control-plane transport. Implementations handle serialization and delivery of:

- `EndpointGrouping` (topology exchange)
- `KvChunkAddressTable` (chunk ownership tables)
- `RankToDevicesMap` (rank-to-FabricNodeId ownership)
- `MigrationRequest` (master-to-subordinate fan-out)
- `MigrationUUID` (completion signals)

```cpp
class EndpointTransport {
public:
    virtual void send_grouping(int dest_rank, const EndpointGrouping&) = 0;
    virtual EndpointGrouping recv_grouping(int src_rank) = 0;
    virtual void send_table(int dest_rank, const KvChunkAddressTable&) = 0;
    virtual KvChunkAddressTable recv_table(int src_rank) = 0;
    virtual void send_rank_device_map(int dest_rank, const RankToDevicesMap&) = 0;
    virtual RankToDevicesMap recv_rank_device_map(int src_rank) = 0;
    virtual void send_migration_request(int dest_rank, const MigrationRequest&) = 0;
    virtual MigrationRequest recv_migration_request(int src_rank) = 0;
    virtual void send_migration_complete(int dest_rank, MigrationUUID) = 0;
    virtual MigrationUUID recv_migration_complete(int src_rank) = 0;
};
```

MPI is the production transport. A loopback implementation exists for unit testing without MPI.

### MpiSender (Data Plane)

The data-plane transport for chunk payloads uses a separate abstraction from the control-plane `EndpointTransport`:

```cpp
class MpiSender {
public:
    virtual void isend(const void* data, uint32_t size, int dest_rank) = 0;
    virtual uint8_t* acquire(uint32_t size) = 0;    // zero-copy
    virtual void publish(uint32_t size, int dest_rank) = 0;
    virtual bool supports_zero_copy() const = 0;
    virtual uint32_t num_sends_in_flight() const = 0;
    virtual bool try_pop_send_completed() = 0;
    virtual bool try_poll_ack(std::function<void(uint64_t)> deliver) { return false; }
};
```

Two usage patterns are supported:

1. **Copy-based**: the sender writes into its own buffer and calls `isend(data, size, rank)`.
2. **Zero-copy**: the sender calls `acquire(size)` to get a pointer into the transport buffer, writes directly into it, then calls `publish(size, rank)`.

The zero-copy path eliminates the extra memcpy per chunk when the transport owns its own send buffers (such as a shared-memory SPSC queue).

### MpiRecvTransport (Data Plane)

The receive-side transport:

```cpp
class MpiRecvTransport {
public:
    virtual bool try_read(const uint8_t*& data, uint32_t& size,
                          int& source_rank) = 0;
    virtual void consume() = 0;
    virtual void shutdown() = 0;
};
```

`try_read()` performs a non-blocking poll; on success, `data` points into the transport's internal buffer (valid until `consume()` is called). This zero-copy receive interface avoids copying inbound data before processing.

---

## 8.33 Wire Format

### MigrationMessageHeader

Every migration message (whether sent individually or as part of a batch) carries a 32-byte header:

```cpp
struct MigrationMessageHeader {
    uint64_t dest_noc_addr;     // Destination NOC address on the target device
    uint64_t uuid;              // Migration UUID
    uint32_t size_bytes;        // Payload size (0 for done marker)
    uint32_t dest_mesh_id;      // Destination FabricNodeId.mesh_id
    uint32_t dest_chip_id;      // Destination FabricNodeId.chip_id
    uint8_t is_done;            // 1 = end-of-migration marker
    uint8_t reserved[3];
};
static_assert(sizeof(MigrationMessageHeader) == 32);
```

Data messages carry `size_bytes` of chunk payload after the header. Done markers have `size_bytes = 0` and `is_done = 1`.

### BatchMessageHeader

To amortize MPI overhead, the sender batches multiple chunks into a single MPI send:

```cpp
struct BatchMessageHeader {
    static constexpr uint32_t MAGIC = 0xBA7C0001;
    uint32_t magic;           // Identifies batch format
    uint32_t num_messages;    // Sub-messages in this batch
};
static_assert(sizeof(BatchMessageHeader) == 8);
```

The receiver detects batch format by checking the first 4 bytes for `MAGIC`. Non-batched (legacy) messages start with `dest_noc_addr` which can never equal the magic value. Each sub-message within a batch is a `[MigrationMessageHeader + payload]` pair.

The default batch size is 32 chunks (`kDefaultMaxBatchCount`). The maximum MPI message size for a batch is:

```cpp
static uint32_t batch_message_size(uint32_t chunk_size_bytes,
    uint32_t max_batch_count = kDefaultMaxBatchCount) {
    uint32_t per_chunk = MigrationMessageHeader::SERIALIZED_SIZE + chunk_size_bytes;
    return BatchMessageHeader::SERIALIZED_SIZE + max_batch_count * per_chunk;
}
```

---

## 8.34 TablePreprocessor and Table Splitting

### KvChunkAddressTable

The `KvChunkAddressTable` (from tt-metalium) describes the physical layout of KV-cache chunks across devices. It maps `(layer, position, slot)` tuples to device addresses. Each chunk belongs to a `DeviceGroup` -- a set of `FabricNodeId`s that host replicas of that chunk (for tensor-parallel models, the same KV position may be present on multiple devices).

### TablePreprocessor

Before migration, the source-side table is preprocessed to eliminate duplicate sends when a chunk is replicated across multiple devices:

```cpp
using TablePreprocessFn = std::function<KvChunkAddressTable(
    const KvChunkAddressTable&)>;

// Default: keep only the first FabricNodeId per device group
KvChunkAddressTable dedup_first_device(const KvChunkAddressTable& table);

// Identity: no dedup (for single-device groups)
KvChunkAddressTable identity_preprocess(const KvChunkAddressTable& table);
```

The `dedup_first_device` strategy picks the first `FabricNodeId` from each device group. A future version could select based on proximity to intermesh ethernet links.

### TableSplitter

When a migration endpoint spans multiple hosts (master + subordinates), the global `KvChunkAddressTable` must be partitioned so each subordinate only reads/writes the chunks physically present on its local devices:

```cpp
class TableSplitter {
public:
    virtual std::vector<KvChunkAddressTable> split(
        const KvChunkAddressTable& table,
        const std::vector<SubordinateInfo>& subordinates) = 0;
};
```

### DcnTableSplitter

The DCN implementation partitions by host: each subordinate's split table contains only the entries whose `DeviceGroup` includes at least one device on that subordinate's host. Duplicate chunks (present on multiple hosts) are assigned to exactly one subordinate to avoid duplicate sends.

```cpp
class DcnTableSplitter : public TableSplitter {
public:
    std::vector<KvChunkAddressTable> split(
        const KvChunkAddressTable& table,
        const std::vector<SubordinateInfo>& subordinates) override;
};
```

---

## 8.35 DeviceReader and DeviceWriter

### Abstract Interface

```cpp
class DeviceReader {
public:
    virtual void read(uint64_t noc_addr, uint32_t size_bytes, void* host_buffer) = 0;
    virtual bool read_async(uint64_t noc_addr, uint32_t size_bytes, void* host_buffer);
    virtual uint32_t num_reads_in_flight() const;
    virtual bool try_pop_completed();
};

class DeviceWriter {
public:
    virtual void write(uint64_t noc_addr, uint32_t size_bytes,
                       const void* host_buffer) = 0;
    virtual bool write_async(uint64_t noc_addr, uint32_t size_bytes,
                             const void* host_buffer);
    virtual uint32_t num_writes_in_flight() const;
    virtual bool try_pop_completed();
};
```

Both support synchronous and asynchronous operations. The async interface defaults to calling the sync method and recording a completion. Implementations with real async DMA override all three async methods.

The `noc_addr` encoding is `(channel << 32 | local_addr)`:

```cpp
inline uint32_t addr_channel(uint64_t noc_addr) { return (uint32_t)(noc_addr >> 32); }
inline uint32_t addr_local(uint64_t noc_addr) { return (uint32_t)(noc_addr & 0xFFFFFFFF); }
inline uint64_t make_noc_addr(uint32_t channel, uint32_t local_addr) {
    return ((uint64_t)channel << 32) | local_addr;
}
```

### UmdDeviceReader/Writer

Direct PCIe DRAM access via UMD (tt-metal's `CreateDevice`):

```cpp
class UmdDeviceReader : public DeviceReader {
    void read(uint64_t noc_addr, uint32_t size, void* buf) override {
        device_->read_dram(addr_channel(noc_addr), addr_local(noc_addr), size, buf);
    }
};
```

The `UmdDevice` wrapper opens the device, verifies DRAM I/O with a write-readback sanity check, and exposes `read_dram` / `write_dram` methods that translate to `ReadFromDeviceDRAMChannel` / `WriteToDeviceDRAMChannel`.

### SocketDeviceReader/Writer

Remote DRAM access via H2D/D2H sockets, used when the migration kernel runs on the device and proxies read/write commands:

```cpp
class SocketDeviceReader : public DeviceReader {
    void read(uint64_t noc_addr, uint32_t size_bytes, void* host_buf) override {
        H2D_CMD cmd = { CMD_TYPE_READ, addr_channel(noc_addr),
                        addr_local(noc_addr), 0, size_bytes };
        h2d_cmd_->write(cmd.data(), 1);
        d2h_data_->read(host_buf, size_bytes / data_page_size_, true);
    }
};
```

The socket reader sends a 64-byte command page over the H2D command socket, then reads the response data from the D2H data socket. The async path (`read_async`) posts the command and tracks pending reads in a FIFO; `try_pop_completed()` checks `d2h_data_->has_data()` and reads when data is available.

The socket writer sends a command page then streams data pages over a separate H2D data socket. The async path tracks watermarks of `bytes_sent` and checks `h2d_data_->acked_past(watermark)` for completion.

### MultiDeviceReader/Writer

Routes operations to the correct single-device reader/writer based on `FabricNodeId`:

```cpp
class MultiDeviceReader {
    std::unordered_map<FabricNodeId, std::unique_ptr<DeviceReader>> devices_;
    std::queue<FabricNodeId> pending_;

    void read(const FabricNodeId& node_id, uint64_t noc_addr,
              uint32_t size, void* buf) {
        devices_.at(node_id)->read(noc_addr, size, buf);
    }

    bool read_async(const FabricNodeId& node_id, uint64_t noc_addr,
                    uint32_t size, void* buf) {
        pending_.push(node_id);
        return devices_.at(node_id)->read_async(noc_addr, size, buf);
    }

    bool try_pop_completed() {
        if (pending_.empty()) return false;
        if (devices_.at(pending_.front())->try_pop_completed()) {
            pending_.pop();
            return true;
        }
        return false;
    }
};
```

Completions are tracked in FIFO order across all devices. This simplifies the caller's drain loop at the cost of head-of-line blocking (a slow device can delay completions for faster ones). This is acceptable because reads within a single migration are typically sized uniformly.

---

## 8.36 DCN Sender Backend

`DcnSenderBackend` implements the full sender pipeline: read chunks from local device DRAM, batch them, and send via MPI.

### Initialization

```cpp
void DcnSenderBackend::init(
    std::shared_ptr<const KvChunkAddressTable> local_table,
    const std::unordered_map<EndpointId,
        std::shared_ptr<const KvChunkAddressTable>>& remote_tables,
    const EndpointHostnameToRankMap& endpoint_hostname_to_rank)
```

Initialization pre-resolves `FabricNodeId` to MPI rank for every device in all remote tables, building `resolved_ranks_`. It also initializes the `send_buffer_` (a `SameProducerConsumerCircularBuffer`) and the per-rank batch buffers.

### migrate_slot Pipeline

The `migrate_slot` method iterates over all chunks in the position range for one `(layer, src_slot -> dst_slot)` pair. For each chunk, it:

1. **Reads from local device DRAM** using the local table's source address.
2. **Translates the destination** using the remote table's entry for the same `(layer, position, dst_slot)`.
3. **Constructs a `MigrationMessageHeader`** with the destination NOC address and FabricNodeId.
4. **Appends to a per-rank batch buffer.** When the batch reaches `max_batch_count`, it is flushed via `MpiSender::isend()` or the zero-copy `acquire/publish` path.

The send pipeline has four logical stages tracked by ring-buffer pointers (`send_info_head_`, `send_info_send_`, `send_info_tail_`):

1. **Issue read** -- post async read to `MultiDeviceReader`.
2. **Wait for read completion** -- `try_pop_completed()`.
3. **MPI send** -- batch and send via `MpiSender`.
4. **Retire** -- for zero-copy mode, wait for transport completion before reusing the slot.

### Done Marker Fan-Out

When `finish()` is called, the sender sends a done marker to every receiver rank (not just the master). This is needed in multi-rank endpoints where each subordinate receiver must know when to stop polling:

```cpp
void DcnSenderBackend::finish(MigrationUUID uuid) {
    flush_all_batches();  // send any remaining partial batch
    // Send done marker to all receiver ranks
    for (auto r : receiver_ranks_) {
        MigrationMessageHeader done{};
        done.uuid = uuid.get();
        done.is_done = 1;
        mpi_sender_->isend(&done, sizeof(done), *r);
    }
}
```

---

## 8.37 DCN Receiver Backend

`DcnReceiverBackend` processes inbound MPI messages and writes chunks to local device DRAM.

### Message Processing

```cpp
void DcnReceiverBackend::handle_message(const void* data,
    uint32_t size, int source_rank)
```

This method handles both batched and non-batched messages:

1. Check for `BatchMessageHeader::MAGIC` at the start. If present, unpack `num_messages` sub-messages.
2. For each sub-message, parse the `MigrationMessageHeader`.
3. **Data chunks** (`is_done == 0`): translate the destination NOC address using the local table, then issue an async write via `MultiDeviceWriter::write_async()`.
4. **Done markers** (`is_done == 1`): increment the per-UUID done counter. When done markers from all expected senders have arrived, drain all outstanding writes, fire the completion callback, and send ACKs back to each sender.

### Multi-Sender Done Protocol

The receiver tracks how many sender ranks are expected per migration via `set_expected_sender_count()`:

```cpp
void DcnReceiverBackend::set_expected_sender_count(uint32_t count) {
    expected_sender_count_ = count;
}
```

When `expected_sender_count_ > 1`, the receiver accumulates done markers per UUID:

```cpp
std::unordered_map<MigrationUUID, uint32_t> done_markers_received_;
std::unordered_map<MigrationUUID, std::vector<int>> done_marker_sources_;
```

Completion fires only when `done_markers_received_[uuid] == expected_sender_count_`. ACKs are sent back to every source rank that contributed a done marker.

### Write Draining

Between processing messages, the caller should periodically call `drain_writes()` to retire completed async writes:

```cpp
uint32_t DcnReceiverBackend::drain_writes() {
    uint32_t retired = 0;
    while (writer_->try_pop_completed()) {
        writes_completed_++;
        retired++;
    }
    return retired;
}
```

When a done marker triggers completion, `drain_all_writes()` blocks until all outstanding writes have finished:

```cpp
void DcnReceiverBackend::drain_all_writes() {
    while (writes_completed_ < writes_issued_) {
        writer_->try_pop_completed();
        writes_completed_++;
    }
}
```

---

## 8.38 Migration Lifecycle

A complete migration proceeds through these phases:

### Phase 1: Table Initialization

The master rank receives the full `KvChunkAddressTable` from the model setup code. It uses `DcnTableSplitter` to partition the table across subordinates and distributes the split portions via `EndpointTransport::send_table`:

```cpp
void MigrationLayerEndpoint::register_local_table(
    std::shared_ptr<KvChunkAddressTable> local_table) {
    local_table_ = local_table;
    // ...
    DcnTableSplitter splitter;
    auto split_tables = splitter.split(*local_table, subordinate_infos_);
    for (auto r : my_group_ranks_) {
        if (r == master_rank_) continue;
        transport_->send_table(*r, *local_table);       // full table
        transport_->send_table(*r, split_tables[i]);     // split portion
    }
    full_local_table_ = local_table_;
    local_table_ = std::make_shared<KvChunkAddressTable>(
        std::move(split_tables[master_idx]));
}
```

Subordinates receive both the full table (for exchange with remote endpoints) and their split portion (for local reads/writes). The master retains `full_local_table_` separately from its own split `local_table_`.

### Phase 2: Endpoint Connection

The master exchanges topology information with the remote endpoint's master:

```cpp
void MigrationLayerEndpoint::master_connect_to_remote_endpoint(
    const EndpointGrouping& remote_endpoint_grouping) {
    // Bidirectional exchange with remote master:
    transport_->send_grouping(remote_master, my_grouping);
    auto remote_grouping = transport_->recv_grouping(remote_master);
    transport_->send_table(remote_master, table_to_send);
    auto remote_table = transport_->recv_table(remote_master);
    transport_->send_rank_device_map(remote_master, my_rank_devices);
    remote_rank_devices_ = transport_->recv_rank_device_map(remote_master);
    // Distribute to subordinates
    for (auto r : my_group_ranks_) {
        if (r == master_rank_) continue;
        transport_->send_grouping(*r, remote_grouping);
        transport_->send_table(*r, *remote_table_ptr);
        transport_->send_rank_device_map(*r, remote_rank_devices_);
    }
}
```

After exchanging tables and rank-device maps, the frontend's sender is initialized with `FabricNodeId`-to-MPI-rank routing. The receiver's expected sender count is set to the number of ranks in the remote endpoint.

### Phase 3: Migration Execution

**Sender side** (on the prefill endpoint):

```cpp
MigrationUUID MigrationLayerEndpoint::migrate_slot(
    EndpointId remote_endpoint_id,
    uint32_t pos_start, uint32_t pos_end,
    uint32_t src_slot, uint32_t dst_slot) {

    auto uuid = allocate_uuid();
    active_migrations_.insert(uuid);
    frontend_->on_completion(uuid,
        [this](MigrationUUID u) { active_migrations_.erase(u); });

    uint32_t num_layers = local_table_->config().num_layers;
    frontend_->start(remote_endpoint_id, uuid);

    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        bool is_last = (layer == num_layers - 1);
        execute_migration(remote_endpoint_id, layer,
            pos_start, pos_end, src_slot, dst_slot, uuid, is_last);
    }

    frontend_->finish(uuid);
    // Wait for subordinate completions
    for (auto r : my_group_ranks_) {
        if (r == master_rank_) continue;
        auto completed = transport_->recv_migration_complete(*r);
    }
    return uuid;
}
```

The `execute_migration` method fans out each layer's migration request to all subordinates and drives the local frontend:

```cpp
void MigrationLayerEndpoint::execute_migration(...) {
    MigrationRequest req{remote_endpoint_id, layer, pos_start, pos_end,
                         src_slot, dst_slot, uuid, emit_finish};
    for (auto r : my_group_ranks_) {
        if (r == master_rank_) continue;
        transport_->send_migration_request(*r, req);
    }
    frontend_->migrate_slot(remote_endpoint_id, layer,
        pos_start, pos_end, src_slot, dst_slot, uuid);
}
```

**Subordinate sender side:**

```cpp
void MigrationLayerEndpoint::listen_for_migration() {
    auto req = transport_->recv_migration_request(*master_rank_);
    frontend_->start(req.remote_endpoint_id, req.uuid);
    frontend_->migrate_slot(req.remote_endpoint_id, req.layer,
        req.pos_start, req.pos_end, req.src_slot, req.dst_slot, req.uuid);
    while (!req.emit_finish) {
        req = transport_->recv_migration_request(*master_rank_);
        frontend_->migrate_slot(...);
    }
    frontend_->finish(req.uuid);
    transport_->send_migration_complete(*master_rank_, req.uuid);
}
```

**Receiver side** (on the decode endpoint):

```cpp
void MigrationLayerEndpoint::block_until_migration_received() {
    bool received = false;
    frontend_->receiver()->on_local_completion(
        [&received](MigrationUUID u) { received = true; });
    while (!received) {
        frontend_->receiver()->try_drain_one();
        frontend_->receiver()->drain_writes();
    }
    // Master: wait for all subordinate receivers
    if (is_master()) {
        for (auto r : my_group_ranks_) {
            if (r == master_rank_) continue;
            transport_->recv_migration_complete(*r);
        }
    } else {
        transport_->send_migration_complete(*master_rank_, received_uuid);
    }
}
```

The receiver polls `try_drain_one()` in a busy loop, processing inbound MPI messages and issuing async device writes. When all done markers arrive and all writes complete, the completion callback fires.

### Phase 4: Completion and ACK

The receiver sends an ACK back to the sender. On the sender side, `wait_migration_send_completion` polls for the ACK:

```cpp
void MigrationLayerEndpoint::wait_migration_send_completion(
    MigrationUUID migration_id) {
    auto* dcn = dynamic_cast<DcnSenderBackend*>(frontend_->sender());
    while (!is_migration_complete(migration_id)) {
        if (dcn) dcn->try_poll_completion_ack();
    }
}
```

The ACK triggers the `on_completion` callback which removes the UUID from `active_migrations_`.

---

## 8.39 Lock-Free Data Structures

The migration library uses two lock-free data structures for inter-thread chunk handoff:

### SpscQueue

A lock-free single-producer single-consumer queue using atomic `wrptr` and `rdptr`:

```cpp
class SpscQueue {
    alignas(64) std::atomic<uint32_t> wrptr_{0};
    alignas(64) std::atomic<uint32_t> rdptr_{0};
};
```

The `alignas(64)` places each pointer on a separate cache line to avoid false sharing between the producer and consumer threads. `num_slots` must be a power of 2; wrap-around uses `2 * num_slots` with bitmask arithmetic to disambiguate full from empty.

### SharedSpscQueueView

A shared-memory variant of the SPSC queue for inter-process use (e.g., between migration processes on the same host). The header and slot data live in a contiguous mmap'd region:

```
[SharedSpscQueueHeader] [slot 0] [slot 1] ... [slot N-1]
```

The view provides the same `try_acquire_write/publish/try_acquire_read/consume` interface. The producer side maintains a local reservation pointer (`producer_wrptr_`) that advances on each `try_acquire_write()`, staying ahead of the consumer-visible `wrptr` until `publish()` closes the gap. This allows consecutive calls to return distinct slots before any are committed.

### SameProducerConsumerCircularBuffer

A single-threaded circular buffer for same-producer-consumer patterns (e.g., the sender's staging buffer where the same thread writes data and then sends it):

```cpp
class SameProducerConsumerCircularBuffer {
    uint8_t* acquire();    // get next slot
    void release();        // free oldest slot
};
```

No atomics needed since both operations run on the same thread. `acquire()` asserts if the buffer is full -- the caller must `release()` before acquiring.

---

## 8.40 Integration with Pipeline Manager

The migration library connects to the pipeline manager through the disaggregated CONTINUE request path. After KV migration completes on the decode endpoint:

1. The inference server sends `ISRequest{type=CONTINUE, gen.disaggregated_decode=true}` with the migrated token.
2. The pipeline manager's `handle_disaggregated_continue` sets the user directly to `DECODE` state at the migrated position and stages the token for immediate decode injection.
3. The user begins generating output tokens without any prefill phase -- the KV cache is already populated from the remote prefill endpoint.

This path bypasses the normal prefill queue entirely, making the migration transparent to the decode scheduling loop.

---

[<< Previous: Scheduling Algorithm](03_scheduling_algorithm.md)
