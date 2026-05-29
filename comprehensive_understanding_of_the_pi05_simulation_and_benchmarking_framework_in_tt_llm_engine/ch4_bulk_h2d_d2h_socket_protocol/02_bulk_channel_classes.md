# 4.2 Bulk Channel Classes

> **Source file:** `examples/pi05_pipeline_runner.cpp`, lines 496--627

The raw `H2DSocket` and `D2HSocket` from tt-metal provide page-granular read/write operations with flow control. The Pi0.5 runner wraps these in two higher-level RAII channel classes---`BulkH2DChannel` for sending pixel data to the device, and `BulkD2HChannel` for receiving action output from the device---that implement the framing protocol described in [Section 4.1](01_wire_protocol.md), manage staging buffers, and enforce zero-fill security. Both are defined inside the `#ifdef PI05_HAS_SOCKETS` guard in the anonymous namespace of `pi05_pipeline_runner.cpp` and are only compiled when the build targets real hardware.

---

## 4.2.1 BulkH2DChannel

### Construction and Socket Binding

```cpp
// examples/pi05_pipeline_runner.cpp:501-508
class BulkH2DChannel {
public:
    BulkH2DChannel(const std::string& socket_id, uint32_t connect_timeout_ms)
        : socket_(tt::tt_metal::distributed::H2DSocket::connect(
              socket_id, connect_timeout_ms)) {
        socket_->set_page_size(BULK_PAGE_SIZE);
    }
```

The constructor calls `H2DSocket::connect()`, which is the cross-process attachment path: it waits for a flatbuffer descriptor file at `/dev/shm/tt_h2d_{socket_id}.bin` exported by the device launcher process, opens the corresponding named shared memory, and establishes PCIe write access without requiring a MetalContext. The page size is set to `BULK_PAGE_SIZE` (4096 bytes) immediately after connection. This page size must match the value the device kernel receives as a compile-time argument at `get_compile_time_arg_val(2)` in `pi05_bulk_passthrough.cpp` line 37.

### The `send()` Method

The `send()` method is the core of H2D transfer. It assembles a page-aligned staging buffer and writes it to the socket in a single call:

```cpp
// examples/pi05_pipeline_runner.cpp:512-543
double send(uint32_t slot_id,
            const uint8_t* pixel_data, uint32_t pixel_bytes,
            const uint8_t* action_data, uint32_t action_bytes) {
    uint32_t payload_length = pixel_bytes + action_bytes;
    uint32_t total = sizeof(BulkH2DHeader) + payload_length;
    uint32_t num_pages = (total + BULK_PAGE_SIZE - 1) / BULK_PAGE_SIZE;
    uint32_t required = num_pages * BULK_PAGE_SIZE;

    if (staging_.size() < required) {
        staging_.resize(required);
    }
    // Zero-fill to avoid leaking stale data in padding
    std::memset(staging_.data(), 0, required);

    // Write header
    BulkH2DHeader header{slot_id, payload_length};
    std::memcpy(staging_.data(), &header, sizeof(header));

    // Write pixel data after header
    std::memcpy(staging_.data() + sizeof(BulkH2DHeader), pixel_data, pixel_bytes);

    // Write action data after pixels (if present)
    if (action_data && action_bytes > 0) {
        std::memcpy(staging_.data() + sizeof(BulkH2DHeader) + pixel_bytes,
                    action_data, action_bytes);
    }

    auto start = Clock::now();
    socket_->write(staging_.data(), num_pages);
    auto end = Clock::now();

    return std::chrono::duration<double, std::micro>(end - start).count();
}
```

**Key design decisions:**

1. **Single atomic write:** The entire frame---header, pixel data, action history, and padding---is assembled in the staging buffer before a single `socket_->write(staging_.data(), num_pages)` call. This is critical for atomicity: the device kernel's main loop expects to read a complete header from the first page and then consume `total_pages - 1` additional pages in a tight loop. Splitting the write across multiple calls would create a window where the kernel could read a partial header. See [Section 4.4](04_race_conditions.md) for the full atomicity analysis.

2. **Page count as length signal:** The `socket_->write()` call takes a page count, not a byte count. The device kernel recalculates the expected page count from the header fields:

   ```cpp
   // kernels/pi05_bulk_passthrough.cpp:129-130
   uint32_t total_bytes = 8 + payload_length;  // header + payload
   uint32_t total_pages = (total_bytes + page_size - 1) / page_size;
   ```

   Both sides must agree on the page-alignment formula, which they do by using identical arithmetic.

3. **Return value:** The method returns the wall-clock transfer duration in microseconds, which is collected into `BulkTransferMetrics::h2d_transfer_us` for benchmarking.

#### Assembly Sequence

The staging buffer is populated in four steps:

1. **Grow staging buffer** -- `staging_` is resized upward if needed but never shrunk (see Section 4.2.3).
2. **Zero-fill** -- The entire padded region is zeroed before any data is written (see Section 4.2.4).
3. **Copy header** -- The 8-byte `BulkH2DHeader` is placed at offset 0.
4. **Copy payload** -- Pixel data is placed immediately after the header (offset 8), and action history (if present) is placed after the pixel data.

### The `send_sentinel()` Method

```cpp
// examples/pi05_pipeline_runner.cpp:546-552
void send_sentinel() {
    std::vector<uint8_t> page(BULK_PAGE_SIZE, 0);
    BulkH2DHeader sentinel{SENTINEL_SLOT_ID, 0};
    std::memcpy(page.data(), &sentinel, sizeof(sentinel));
    socket_->write(page.data(), 1);
}
```

This sends a single zero-padded page with `slot_id = 0xFFFFFFFF` and `payload_length = 0`. Unlike `send()`, it allocates a fresh temporary buffer rather than reusing the staging buffer---this is safe because `send_sentinel()` is only called once during shutdown.

### The `barrier()` Method

```cpp
// examples/pi05_pipeline_runner.cpp:554
void barrier() { socket_->barrier(); }
```

Delegates to `H2DSocket::barrier()`, which blocks until the device has acknowledged consumption of all bytes sent through the H2D socket. This is used in the shutdown sequence (see [Section 4.3](03_shutdown_protocol.md)) to ensure the sentinel has been consumed before waiting for the D2H echo.

---

## 4.2.2 BulkD2HChannel

### Construction

```cpp
// examples/pi05_pipeline_runner.cpp:563-569
class BulkD2HChannel {
public:
    BulkD2HChannel(const std::string& socket_id, uint32_t connect_timeout_ms)
        : socket_(tt::tt_metal::distributed::D2HSocket::connect(
              socket_id, connect_timeout_ms)) {
        socket_->set_page_size(BULK_PAGE_SIZE);
    }
```

Mirrors the H2D channel: `D2HSocket::connect()` connects via the descriptor at `/dev/shm/tt_d2h_{socket_id}.bin`, blocks until the device-side descriptor is available, then sets the page size.

### The `recv()` Method

```cpp
// examples/pi05_pipeline_runner.cpp:573-606
double recv(uint32_t& out_slot_id, void* output_data, uint32_t max_output_bytes) {
    // 8B header + 3200B payload = 3208B < 4096B = 1 page
    uint32_t num_pages = 1;
    recv_buf_.resize(num_pages * BULK_PAGE_SIZE, 0);

    // Poll has_data() before read() (matching SocketPipeline pattern)
    auto spin_start = Clock::now();
    uint64_t spin_iters = 0;
    while (!socket_->has_data()) {
        if (++spin_iters % 10000000 == 0) {
            auto elapsed = std::chrono::duration<double>(Clock::now() - spin_start).count();
            if (elapsed > 5.0) {
                std::cerr << "WARNING: D2H recv spin-wait exceeded "
                          << elapsed << "s (" << spin_iters
                          << " iters). Device kernel may be stuck." << std::endl;
                spin_start = Clock::now();
            }
        }
    }

    auto start = Clock::now();
    socket_->read(recv_buf_.data(), num_pages);
    auto end = Clock::now();

    // Parse header
    BulkD2HHeader header;
    std::memcpy(&header, recv_buf_.data(), sizeof(header));
    out_slot_id = header.slot_id;

    uint32_t copy_bytes = std::min(header.payload_length, max_output_bytes);
    std::memcpy(output_data, recv_buf_.data() + sizeof(BulkD2HHeader), copy_bytes);

    return std::chrono::duration<double, std::micro>(end - start).count();
}
```

**Key design decisions:**

1. **Hardcoded single page:** Unlike H2D, the D2H direction always reads exactly one page. This is enforced by the knowledge that the action output (3200 bytes) plus header (8 bytes) fits within a single 4096-byte page. If the action output size ever exceeds 4088 bytes, this code will silently truncate.

2. **Spin-wait with health monitoring:** The `has_data()` poll loop is a busy-wait that yields to no other work. Every 10 million iterations (roughly every few seconds depending on CPU speed), it checks whether 5 seconds have elapsed and emits a warning to stderr. This pattern matches the `SocketPipeline` token path, providing consistent diagnostics across both data paths. Example output:

   ```
   WARNING: D2H recv spin-wait exceeded 5.12s (50000000 iters). Device kernel may be stuck.
   ```

3. **No timeout:** The spin-wait has no hard timeout. If the device kernel never produces data, the host will spin indefinitely, emitting warnings every 5 seconds. This is a deliberate design choice documented as a robustness concern in [Section 4.3](03_shutdown_protocol.md).

4. **Payload copy with bounds check:** `std::min(header.payload_length, max_output_bytes)` prevents buffer overflow if the device sends a larger payload than expected, though in practice the device always sends exactly `ACTION_OUTPUT_BYTES`.

### The `recv_sentinel_echo()` Method

```cpp
// examples/pi05_pipeline_runner.cpp:610-619
bool recv_sentinel_echo() {
    recv_buf_.resize(BULK_PAGE_SIZE, 0);
    while (!socket_->has_data()) {
        // Spin-wait for device kernel to echo sentinel
    }
    socket_->read(recv_buf_.data(), 1);
    uint32_t word0;
    std::memcpy(&word0, recv_buf_.data(), sizeof(word0));
    return word0 == SENTINEL_SLOT_ID;
}
```

This is a stripped-down receiver used only during shutdown. It spins on `has_data()` without any warning timer---the assumption being that the sentinel echo arrives quickly after the H2D sentinel is sent and the barrier completes. The return value indicates whether the received page actually contained the sentinel marker. If it returns `false`, the shutdown sequence logs a warning but continues anyway (line 1139).

**Note the unbounded spin-wait:** Unlike `recv()`, there is no 5-second warning or timeout. If the device kernel crashes between receiving the sentinel and writing the echo, `recv_sentinel_echo()` will block forever with zero diagnostics. See [Section 4.3](03_shutdown_protocol.md) for robustness discussion.

### Auxiliary Methods

```cpp
// examples/pi05_pipeline_runner.cpp:621-622
bool has_data() { return socket_->has_data(); }
void barrier() { socket_->barrier(); }
```

`has_data()` exposes the non-blocking data-availability check. `barrier()` blocks until all data has been acknowledged, which is symmetric with the H2D barrier.

---

## 4.2.3 Staging Buffer Reuse (Grows, Never Shrinks)

`BulkH2DChannel` maintains a persistent staging buffer as a private member:

```cpp
// examples/pi05_pipeline_runner.cpp:555-558
private:
    std::unique_ptr<tt::tt_metal::distributed::H2DSocket> socket_;
    std::vector<uint8_t> staging_;
};
```

The buffer is grown on demand but never shrunk:

```cpp
// examples/pi05_pipeline_runner.cpp:519-521
if (staging_.size() < required) {
    staging_.resize(required);
}
```

This means:

1. After the first `send()` call, the staging buffer is allocated to the padded size of the largest payload seen so far.
2. Subsequent calls with equal or smaller payloads reuse the same allocation with zero additional heap operations.
3. If a larger payload arrives (e.g., higher-resolution images), the buffer grows to accommodate it and stays at that size.

This grow-never-shrink strategy eliminates per-frame allocation overhead in the steady state. Since all users share the same `PixelPayloadConfig`, the staging buffer reaches its steady-state size after the first `send()` and remains there for the lifetime of the channel. The trade-off is that the buffer holds memory for the lifetime of the channel, but this is insignificant compared to the pixel data itself.

Similarly, `BulkD2HChannel` uses a persistent `recv_buf_`:

```cpp
// examples/pi05_pipeline_runner.cpp:625-627
private:
    std::unique_ptr<tt::tt_metal::distributed::D2HSocket> socket_;
    std::vector<uint8_t> recv_buf_;
};
```

Since D2H always reads exactly one page, this buffer is allocated to 4096 bytes on the first `recv()` call and stays there.

---

## 4.2.4 Zero-Fill Security

Every `send()` call zeroes the entire padded region before copying any data:

```cpp
// examples/pi05_pipeline_runner.cpp:523
std::memset(staging_.data(), 0, required);
```

This serves two purposes:

1. **Cross-user data isolation.** Without the zero-fill, the padding region (between the end of the payload and the end of the last page) would contain stale data from a previous transfer. On a multi-user system, this could leak pixel data from one user's frame into another user's padding region if the staging buffer was previously used for a different user with a slightly different payload size.

2. **Predictable kernel behavior.** The device kernel reads pages in their entirety (the NOC transfer is page-granular). Zero-filled padding ensures the kernel never sees garbage data in trailing bytes, even if it reads beyond the `payload_length`.

The D2H kernel side mirrors this pattern:

```cpp
// kernels/pi05_bulk_passthrough.cpp:169-171
// Zero the output page
for (uint32_t w = 0; w < page_size_words; w++) {
    output_cb_addr[w] = 0;
}
```

Both directions thus guarantee that any bytes beyond the declared `payload_length` are zero.

---

## 4.2.5 Channel Lifecycle

The channels are created after the device launcher has started and its socket descriptors are available:

```cpp
// examples/pi05_pipeline_runner.cpp:923-928
auto bulk_h2d = std::make_unique<BulkH2DChannel>(
    pipeline_config.socket.h2d_bulk_socket_id,
    pipeline_config.socket.connect_timeout_ms);
auto bulk_d2h = std::make_unique<BulkD2HChannel>(
    pipeline_config.socket.d2h_bulk_socket_id,
    pipeline_config.socket.connect_timeout_ms);
```

They are held as `std::unique_ptr` and explicitly destroyed during shutdown (lines 1148-1149):

```cpp
// examples/pi05_pipeline_runner.cpp:1149-1150
bulk_h2d.reset();
bulk_d2h.reset();
```

The destruction order matters: the H2D channel is destroyed before the D2H channel, though in practice both are reset sequentially and the socket destructors handle deregistration with the tt-metal socket layer.

### IOMMU Initialization Race Workaround

A 2-second sleep is inserted between `DeviceLauncher::start()` and channel creation (line 920) to allow the device kernel to complete IOMMU page mapping before the host attempts to connect:

```cpp
// examples/pi05_pipeline_runner.cpp:920-921
// Wait for device kernel init to avoid IOMMU mapping race.
std::this_thread::sleep_for(std::chrono::milliseconds(2000));
```

This is a workaround for a race condition where the host `H2DSocket::connect()` succeeds (the descriptor file exists) but the device kernel has not yet mapped the FIFO pages in IOMMU, leading to DMA faults on the first write. The descriptor file is created by the device launcher's socket setup, which runs before the kernel's IOMMU mapping completes. A more robust solution would involve an explicit readiness signal from the device kernel, but the current codebase relies on the fixed delay.

---

**Previous:** [Wire Protocol and Frame Layout](01_wire_protocol.md)

**Next:** [Shutdown Protocol](03_shutdown_protocol.md)
