# 04 -- Buffer Types and Circular Buffers

## Context

The previous three files covered the physical L1 memory maps (files 01 and 02) and the NOC system that moves data between cores (file 03). This file covers the software abstractions built on top of that hardware: circular buffers that stage data for compute kernels, `RemoteCircularBuffer` that extends circular buffer semantics across cores on the same chip, `GlobalCircularBuffer` that provides multi-sender/multi-receiver shared L1, `GlobalSemaphore` that enables cross-core synchronization, and the dispatch system that enqueues commands from the host to device L1. These abstractions are the programming interface that CCL operations (Chapter 1) use to move data, and they are the building blocks that any GSRP would need to extend or replace for cross-chip data movement.

---

## 1. Local Circular Buffers

Circular buffers (CBs) are the fundamental data staging mechanism on Tensix cores. A CB is a region of L1 organized as a ring buffer with page-level producer-consumer semantics:

```
Circular Buffer in L1
+--------------------------------------------------+
| fifo_start_addr                                  |
|  Page 0 | Page 1 | Page 2 | ... | Page N-1      |
+--------------------------------------------------+
    ^                     ^
    fifo_rd_ptr           fifo_wr_ptr
```

### 1.1 `LocalCBInterface`

Each CB is identified by a `cb_id` (0-31) and configured via `CircularBufferConfig`. The device-side state is held in a `LocalCBInterface` struct:

```cpp
// Key fields in the local CB interface (device-side struct)
struct LocalCBInterface {
    uint32_t fifo_start_addr;        // Start of the CB region in L1 (shifted by cb_addr_shift)
    uint32_t fifo_wr_ptr;            // Current write pointer
    uint32_t fifo_rd_ptr;            // Current read pointer
    uint32_t fifo_size;              // Total CB size in bytes
    uint32_t fifo_limit;             // fifo_start + fifo_size (wrap point)
    uint32_t fifo_page_size;         // Size of each page
    uint32_t fifo_num_pages;         // fifo_size / fifo_page_size
};
```

### 1.2 Producer-Consumer Protocol

The standard CB protocol between RISC-V processors on the same core:

**Producer (typically BRISC/NCRISC data mover):**
1. `cb_reserve_back(cb_id, num_pages)` -- block until `num_pages` slots are free
2. Write data to `cb_interface.fifo_wr_ptr`
3. `cb_push_back(cb_id, num_pages)` -- advance write pointer, signal consumer

**Consumer (typically TRISC compute or NCRISC writer):**
1. `cb_wait_front(cb_id, num_pages)` -- block until `num_pages` are available
2. Read data from `cb_interface.fifo_rd_ptr`
3. `cb_pop_front(cb_id, num_pages)` -- advance read pointer, signal producer

This protocol is entirely local to one core -- read and write pointers are in L1, and the blocking is done by spin-waiting on pointer comparisons. No NOC transactions are involved.

### 1.3 CB Configuration from Host

On the host side, `CircularBufferConfig` specifies per-CB parameters:

```cpp
class CircularBufferConfig {
    std::array<std::optional<std::pair<uint32_t, DataFormat>>, NUM_CIRCULAR_BUFFERS> buffer_indices_;
    uint32_t total_size_;
    std::optional<bool> globally_allocated_;
    // ...
};
```

Multiple `cb_id` values can share the same physical L1 region (aliasing), which is commonly used when a producer and consumer need different page sizes or data formats on the same underlying data.

### 1.4 Multi-RISC Coordination

On a single Tensix core, multiple RISC-V processors can coordinate through CBs. A typical pattern:
- BRISC (data mover) reads from DRAM/NOC and pushes to CB 0
- TRISC0 (compute) waits on CB 0, processes tiles, pushes results to CB 1
- NCRISC (data mover) waits on CB 1 and writes results to DRAM or another core via NOC

The CB pointers in L1 serve as the synchronization mechanism between processors.

---

## 2. `RemoteCircularBuffer`

The `RemoteCircularBuffer` (defined in `tt_metal/hw/inc/api/remote_circular_buffer.h`) extends circular buffer semantics so the producer and consumer can be on different Tensix cores on the same chip. Where a local CB uses L1 pointer reads for flow control, a `RemoteCircularBuffer` uses NOC atomic increments to communicate pointer updates between cores.

### 2.1 Architecture

```
Sender Core                                      Receiver Core
+------------------+                             +------------------+
| RemoteSenderCB   |                             | RemoteReceiverCB |
|  fifo_wr_ptr     |                             |  fifo_rd_ptr     |
|  pages_sent[]    | --- NOC atomic inc --------> |  pages_sent[]    |
|  pages_acked[]   | <--- NOC atomic inc -------- |  pages_acked[]   |
|  receiver_noc_xy |                             |  sender_noc_x/y  |
+------------------+                             +------------------+
```

The sender writes data directly to the receiver's L1 via NOC write, then atomically increments the `pages_sent` counter on the receiver. The receiver reads the data from its own L1 (already there), then atomically increments the `pages_acked` counter on the sender to free buffer space.

### 2.2 Flow Control Mechanism

The `RemoteCircularBuffer` uses two counters per sender-receiver pair:

| Counter | Location | Updated By | Purpose |
|---------|----------|:----------:|---------|
| `pages_sent` | Sender L1, mirrored to receiver | Sender (NOC atomic inc) | Tracks how many aligned pages have been sent |
| `pages_acked` | Receiver L1, mirrored to sender | Receiver (NOC atomic inc) | Tracks how many aligned pages have been consumed |

The page counts use `REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE` granularity rather than individual pages, allowing the counters to work with variable page sizes through a resize mechanism.

**Sender flow:**
1. `reserve_back(num_pages)` -- spin-wait until `pages_sent - pages_acked < fifo_capacity` for all receivers
2. Write data to `fifo_wr_ptr` via NOC write to receiver's L1
3. `push_back(...)` -- advance `fifo_wr_ptr`, atomically increment `pages_sent` on the receiver via NOC

**Receiver flow:**
1. `wait_front(num_pages)` -- spin-wait until `pages_sent - pages_acked >= num_pages_needed`
2. Read data from `fifo_rd_ptr` (already in local L1, written by sender)
3. `pop_front(num_pages)` -- advance `fifo_rd_ptr`, atomically increment `pages_acked` on the sender via NOC

### 2.3 Sender and Receiver Interface Structs

The low-level interface structures hold the per-connection state:

**`RemoteSenderCBInterface`:** Maintains arrays of receiver NOC coordinates (`receiver_noc_xy_ptr`), per-receiver aligned page sent counters (`aligned_pages_sent_ptr`), and the local write pointer. The counters are separated by `2 * L1_ALIGNMENT` to avoid false sharing when multiple receivers' counters are updated independently.

**`RemoteReceiverCBInterface`:** Maintains the sender's NOC coordinates (`sender_noc_x`, `sender_noc_y`), the local read pointer, aligned page acked counter, and page sent counter. The receiver reads `pages_sent` from its own L1 (atomically updated by the sender) and writes `pages_acked` to the sender's L1 via NOC.

### 2.4 Multi-Receiver Support

A single sender can write to multiple receivers. The sender iterates over all receivers, checking credit availability for each before writing and updating all sent counters. This fan-out pattern is used in CCL operations where a single worker core distributes data to multiple downstream consumers.

### 2.5 Page Size Resize

The `RemoteCircularBuffer` supports dynamic page size changes via `set_sender_page_size` and `set_receiver_page_size`. When resizing:
1. The current write/read pointer may not be aligned to the new page size
2. The resize function computes the alignment adjustment
3. If `update_remote_over_noc` is true, the adjustment is communicated to the remote side via NOC atomic increment
4. The `fifo_limit_page_aligned` is recalculated to avoid wasting space at the CB boundary

### 2.6 The `RemoteCircularBuffer` Class API

```cpp
// tt_metal/hw/inc/api/remote_circular_buffer.h
class RemoteCircularBuffer {
public:
    enum class RemotePointerUpdate { SKIP, UPDATE_OVER_NOC };

    explicit RemoteCircularBuffer(uint32_t remote_cb_index);

    // Sender operations
    void reserve_back(uint32_t num_pages);
    template <typename Src, RemotePointerUpdate update = RemotePointerUpdate::UPDATE_OVER_NOC>
    void push_back(Noc& noc, const Src& src, uint32_t num_pages,
                   uint32_t num_rows, uint32_t coalesced_num_pages_per_row,
                   uint32_t coalesced_page_size, ...);
    template <RemotePointerUpdate update = RemotePointerUpdate::UPDATE_OVER_NOC>
    void set_sender_page_size(Noc& noc, uint32_t page_size, ...);

    // Receiver operations
    void wait_front(uint32_t num_pages);
    void pop_front(Noc& noc, uint32_t num_pages);
    template <RemotePointerUpdate update = RemotePointerUpdate::UPDATE_OVER_NOC>
    void set_receiver_page_size(Noc& noc, uint32_t page_size, ...);

    // Synchronization
    void barrier();   // Wait for all pages to be consumed by receivers
    void commit();    // Write cached pointers back to L1
};
```

> **Key Insight:** The `RemoteCircularBuffer` is conceptually very close to what a cross-chip circular buffer would look like. The difference is that it uses NOC atomic increments for flow control (which are chip-local) -- extending it to cross-chip would require replacing these with fabric-mediated atomic increments. This is precisely what the `GlobalCircularBuffer` begins to address. The latency gap between intra-chip NOC atomics (nanoseconds) and inter-chip fabric atomics (microseconds) is the primary challenge.

---

## 3. `GlobalCircularBuffer`

The `GlobalCircularBuffer` (defined in `tt_metal/api/tt-metalium/global_circular_buffer.hpp`) provides a multi-sender/multi-receiver shared L1 buffer where different sender cores can write to different receiver cores:

```cpp
// tt_metal/api/tt-metalium/global_circular_buffer.hpp
class GlobalCircularBuffer {
public:
    GlobalCircularBuffer(
        IDevice* device,
        const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
        uint32_t size,
        BufferType buffer_type = BufferType::L1);

    const Buffer& cb_buffer() const;
    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    DeviceAddr buffer_address() const;
    DeviceAddr config_address() const;
    uint32_t size() const;
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping() const;
};
```

### 3.1 Sender-Receiver Mapping

The `sender_receiver_core_mapping` parameter defines which sender cores communicate with which receiver cores:

```cpp
// Example: two senders, each writing to a set of receivers
std::vector<std::pair<CoreCoord, CoreRangeSet>> mapping = {
    {CoreCoord{0, 0}, CoreRangeSet{CoreRange{CoreCoord{1, 0}, CoreCoord{3, 0}}}},
    {CoreCoord{0, 1}, CoreRangeSet{CoreRange{CoreCoord{1, 1}, CoreCoord{3, 1}}}},
};
auto gcb = CreateGlobalCircularBuffer(device, mapping, size);
```

### 3.2 Implementation

Internally, `GlobalCircularBuffer` is implemented as a wrapper around a sharded buffer:
- The data buffer (`cb_buffer_`) is a height-sharded buffer allocated across all participating cores
- A separate config buffer (`cb_config_buffer_`) stores the sender-receiver mapping metadata
- The `CreateCircularBuffer` function binds a program-level CB to the global circular buffer's address space

This implementation means the `GlobalCircularBuffer` leverages the existing buffer allocation and dispatch infrastructure rather than introducing a new allocation mechanism. The comment in the source notes this could be "updated in the future to be its own container with optimized dispatch functions."

### 3.3 Usage Pattern

```cpp
// Host side: create global CB and bind to program
auto gcb = experimental::CreateGlobalCircularBuffer(device, mapping, buffer_size);
CBHandle cb_handle = experimental::CreateCircularBuffer(program, core_spec, cb_config, gcb);

// Device side: use standard CB API with the assigned cb_id
// The underlying L1 addresses are managed by the global CB infrastructure
```

### 3.4 Dynamic Address Updates

When the global circular buffer is allocated, the L1 addresses on each participating core may be determined at allocation time rather than compile time. The config buffer contains the address mapping, and the device-side code reads this mapping at kernel startup to initialize its local CB interface. This late-binding of addresses is a pattern that extends naturally to cross-chip scenarios where addresses on remote devices are not known until the mesh is initialized.

---

## 4. `GlobalSemaphore`

The `GlobalSemaphore` (defined in `tt_metal/api/tt-metalium/global_semaphore.hpp`) allocates a semaphore that is accessible from all cores in a specified `CoreRangeSet`:

```cpp
// tt_metal/api/tt-metalium/global_semaphore.hpp
class GlobalSemaphore {
public:
    GlobalSemaphore(
        IDevice* device,
        const CoreRangeSet& cores,
        uint32_t initial_value,
        BufferType buffer_type = BufferType::L1);

    IDevice* device() const;
    DeviceAddr address() const;
    void reset_semaphore_value(uint32_t reset_value) const;
};
```

### 4.1 Implementation

Like `GlobalCircularBuffer`, `GlobalSemaphore` is implemented as a wrapper around a sharded buffer. The semaphore value is stored at a consistent L1 address across all cores in the `CoreRangeSet`, enabling any core to atomically increment the semaphore on any other participating core via NOC atomic increment.

### 4.2 Usage in CCL Operations

As documented in Chapter 1, File 02, CCL operations use `GlobalSemaphore` for three purposes:

1. **Forward/backward synchronization:** Two global semaphores coordinate ring-step completion between devices
2. **Barrier synchronization:** One global semaphore ensures all devices have allocated buffers before data movement begins
3. **Op fusion signaling:** Semaphores signal downstream compute kernels when data slices arrive

### 4.3 Consistent-Address Property

The key property is that `address()` returns the same L1 address on every participating core, enabling cross-core atomic operations without per-core address resolution. This consistent-address property is shared with `MeshBuffer` (Chapter 4) and is a building block for global address space designs. `MeshBuffer::create()` performs lock-step allocation across all mesh devices to achieve the same address on every chip -- the cross-device extension of this same principle.

---

## 5. The Dispatch System

The dispatch system bridges host-side command submission and device-side kernel execution. It is relevant to the GSRP proposal because any transparent cross-chip communication must integrate with the existing dispatch pipeline.

### 5.1 `DispatchMemMap`

The `DispatchMemMap` class (defined in `tt_metal/impl/dispatch/dispatch_mem_map.hpp`) manages the L1 memory layout for dispatch cores -- the Tensix cores dedicated to processing host commands rather than running user compute kernels:

```cpp
// tt_metal/impl/dispatch/dispatch_mem_map.hpp
class DispatchMemMap {
public:
    DispatchMemMap(const CoreType& core_type, uint32_t num_hw_cqs,
                   const Hal& hal, bool is_galaxy_cluster);

    uint32_t prefetch_q_entries() const;
    uint32_t prefetch_q_size() const;
    uint32_t max_prefetch_command_size() const;
    uint32_t cmddat_q_base() const;
    uint32_t cmddat_q_size() const;
    uint32_t scratch_db_base() const;
    uint32_t dispatch_buffer_base() const;
    uint32_t dispatch_buffer_pages() const;
    uint32_t get_device_command_queue_addr(const CommandQueueDeviceAddrType& type) const;
    uint32_t get_host_command_queue_addr(const CommandQueueHostAddrType& host_addr) const;
    // ...
};
```

### 5.2 Command Queue Architecture

The dispatch pipeline consists of several stages, each consuming a region of L1 on dedicated dispatch cores:

```
Host                  Dispatch Core L1
+------+             +-----------------------------+
| CQ   | ---------> | Prefetch Q (command headers) |
| Write|             +-----------------------------+
+------+             | CmdDat Q (command data)      |
                     +-----------------------------+
                     | Scratch Double Buffer        |
                     +-----------------------------+
                     | Dispatch Buffer (pages)      |
                     +-----------------------------+
```

**Prefetch Q:** A ring buffer of command headers submitted by the host. The prefetcher reads from this queue and fetches the corresponding command data.

**CmdDat Q:** Stores the full command payloads, including embedded data for small writes or program binaries.

**Scratch Double Buffer:** A staging area for the prefetcher to prepare data before dispatching.

**Dispatch Buffer:** The main output buffer from which the dispatcher sends data to worker cores.

### 5.3 Dispatch Synchronization

The dispatch system uses stream registers for synchronization between the prefetcher and dispatcher. The `dispatch_stream_index` and `dispatch_message_update_offset` map to NOC stream registers that the dispatcher writes to signal completion. This register-based signaling avoids the overhead of L1 semaphore polling for the critical dispatch path.

### 5.4 Galaxy Cluster Awareness

The `DispatchMemMap` constructor takes an `is_galaxy_cluster` parameter that affects dispatch buffer sizing. In a Galaxy cluster (multi-chip system), dispatch buffers may need to be larger to accommodate cross-chip program binaries and data transfers. This is one of the points where the dispatch system already accounts for multi-chip configurations.

---

## 6. How Data Reaches L1 and DRAM

The complete data path from host to device involves multiple stages:

### 6.1 Host-to-L1 Writes

```
Host Process
    |
    v
Command Queue (host side, in system memory)
    |
    v (PCIe DMA or MMIO write)
Prefetch Q on Dispatch Core L1
    |
    v (dispatch core reads command, writes to target)
Target Core L1 (via NOC write from dispatch core)
```

For large tensor writes, the host writes data to system memory (`BufferType::SYSTEM_MEMORY`), and the dispatch core DMA-transfers it through PCIe to DRAM or L1.

### 6.2 L1-to-L1 Transfers (intra-chip)

```
Source Core L1
    |
    v (NOC unicast/multicast write)
Destination Core L1
```

Direct NOC writes between L1 regions. This is the mechanism used by circular buffers, remote circular buffers, and CCL worker kernels for intra-chip data movement.

### 6.3 L1-to-DRAM and DRAM-to-L1

```
Core L1 <---> DRAM Controller <---> DRAM Banks
               (via NOC read/write to DRAM controller coordinates)
```

DRAM controllers appear as NOC endpoints at specific coordinates. A core accesses DRAM by issuing NOC reads/writes to the DRAM controller's coordinates, with the DRAM bank offset encoded in the local address portion of the NOC address.

### 6.4 Cross-Chip Transfers (for context)

```
Source Core L1
    |
    v (NOC write to local Ethernet core L1)
Local Ethernet Core L1
    |
    v (Ethernet link transfer)
Remote Ethernet Core L1
    |
    v (NOC write from remote Ethernet core to target)
Target Core L1
```

This is the path documented in Chapter 3. The key observation is that from the source core's perspective, the first step is an ordinary intra-chip NOC write. The Ethernet core firmware handles everything from that point forward. A transparent GSRP would make this entire chain invisible to the source core.

---

## 7. Summary: The Software Stack on Top of L1

```
+------------------------------------------------------------------+
|  CCL Operations (all-gather, reduce-scatter, ...)                |
|  [Chapter 1]                                                      |
+------------------------------------------------------------------+
|  GlobalCircularBuffer | GlobalSemaphore | MeshBuffer             |
|  RemoteCircularBuffer                                             |
+------------------------------------------------------------------+
|  CircularBufferConfig | Local CB API                              |
|  (cb_reserve_back, cb_push_back, cb_wait_front, cb_pop_front)    |
+------------------------------------------------------------------+
|  experimental::Noc                                                |
|  (async_read, async_write, async_write_multicast, barriers)      |
+------------------------------------------------------------------+
|  NOC Hardware (unicast, multicast, atomic, posted/non-posted)    |
|  [File 03]                                                        |
+------------------------------------------------------------------+
|  L1 SRAM (Tensix 1464/1536 KB | Ethernet 256/512 KB)            |
|  [Files 01, 02]                                                   |
+------------------------------------------------------------------+
```

Each layer adds abstraction:
- **L1 SRAM** provides raw byte-addressable storage
- **NOC hardware** provides core-to-core data movement primitives
- **`experimental::Noc`** provides a type-safe C++ interface for NOC operations
- **Circular buffers** provide page-level producer-consumer staging
- **`RemoteCircularBuffer`** extends CB semantics across cores via NOC atomics
- **`GlobalCircularBuffer`/`GlobalSemaphore`** provide multi-core shared state with consistent addressing
- **CCL operations** compose these primitives into collective communication patterns

A GSRP would insert a new layer between the `experimental::Noc` and the fabric, making cross-chip NOC transactions transparent. Alternatively, it could extend the `GlobalCircularBuffer` and `GlobalSemaphore` abstractions to work across chip boundaries, leveraging their existing consistent-address properties.

### Intra-Chip to Inter-Chip Abstraction Mapping

The following table summarizes how existing intra-chip primitives relate to their potential cross-chip extensions:

| Intra-Chip Primitive | Scope | Cross-Chip Extension | Gap to Bridge |
|---------------------|-------|---------------------|---------------|
| NOC unicast write | Core-to-core, same chip | Fabric unicast write | Address translation + packet construction |
| NOC multicast write | Rectangle of cores, same chip | Fabric multicast | 2D route computation + multi-hop forwarding |
| NOC atomic increment | Single address, same chip | Fabric-mediated atomic | Latency: ns to us |
| `RemoteCircularBuffer` | Core-to-core, same chip | Cross-chip remote CB | Replace NOC atomics with fabric atomics |
| `GlobalCircularBuffer` | Multi-core, same device | Cross-device global CB | Extend sharded buffer across mesh |
| `GlobalSemaphore` | Multi-core, same device | Cross-device semaphore | Already partly addressed by `MeshBuffer` |
| `CircularBufferConfig` | Single core | Unchanged | N/A (purely local) |

---

## Key Takeaways

- **Circular buffers are the universal data staging mechanism.** Every data movement on Tensix -- compute input/output, NOC transfers, fabric sends -- flows through circular buffers. A GSRP must either preserve CB semantics for cross-chip transfers or provide an equally efficient alternative.
- **`RemoteCircularBuffer` already solves the cross-core flow control problem** using NOC atomic increments for pages-sent/pages-acked tracking. Extending this to cross-chip requires replacing NOC atomics with fabric-mediated atomics -- a conceptually small step but one with significant latency implications (microseconds instead of nanoseconds).
- **`GlobalCircularBuffer` and `GlobalSemaphore` use consistent L1 addresses across cores.** This consistent-address property -- where the same `address()` value is valid on every participating core -- is the foundation for cross-core coordination. `MeshBuffer` (Chapter 4) extends this property across devices, allocating at the same address on every chip in a mesh.
- **The dispatch system owns specific L1 regions on dedicated cores.** Dispatch cores are excluded from the L1 banking allocator and have their own memory layout managed by `DispatchMemMap`. Any GSRP integration must avoid conflicting with dispatch memory and should ideally integrate with the command queue for host-initiated cross-chip transfers.
- **The software stack is layered and composable.** Each abstraction builds cleanly on the one below. This layering suggests that the most practical GSRP approach is to extend the existing abstractions (adding a cross-chip `RemoteCircularBuffer`, extending `GlobalSemaphore` across chips) rather than replacing the entire stack.

## Source Code References

- `tt_metal/hw/inc/api/remote_circular_buffer.h` -- `RemoteCircularBuffer` class, `RemoteSenderCBInterface`, `RemoteReceiverCBInterface`, flow control functions
- `tt_metal/api/tt-metalium/global_circular_buffer.hpp` -- `GlobalCircularBuffer` class, `CreateGlobalCircularBuffer`, `CreateCircularBuffer` with global CB
- `tt_metal/api/tt-metalium/global_semaphore.hpp` -- `GlobalSemaphore` class
- `tt_metal/api/tt-metalium/circular_buffer.hpp` -- `CircularBufferConfig`, CB host-side API
- `tt_metal/impl/dispatch/dispatch_mem_map.hpp` -- `DispatchMemMap` class, command queue address types
- `tt_metal/api/tt-metalium/buffer_types.hpp` -- `BufferType`, `TensorMemoryLayout`

---

**Next:** [Chapter 3 -- Inter-Chip Communication](../ch03_interchip_fabric/index.md)
