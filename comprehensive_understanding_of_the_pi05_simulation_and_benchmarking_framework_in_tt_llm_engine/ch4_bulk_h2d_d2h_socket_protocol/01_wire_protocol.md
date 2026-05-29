# 4.1 Wire Protocol and Frame Layout

> **Source files:**
> - `examples/pi05_pipeline_runner.cpp` (lines 200--258, header structs and constants)
> - `kernels/pi05_bulk_passthrough.cpp` (lines 26--33, kernel-side constants)

The bulk socket path carries two fundamentally different data flows over the same page-aligned transport: large multi-page pixel frames from host to device (H2D), and compact single-page action tensors from device to host (D2H). Both directions share an identical 8-byte header format but diverge sharply in payload size, creating an asymmetric protocol that the FIFO sizing must accommodate.

---

## 4.1.1 Common Constants

Three values govern the wire format. Two (`SENTINEL_SLOT_ID` and `ACTION_OUTPUT_BYTES`) are defined as named constants on both the host and device sides. The third (`BULK_PAGE_SIZE`) is a named constant on the host but is delivered to the device kernel as a compile-time argument via `get_compile_time_arg_val(2)`:

```cpp
// examples/pi05_pipeline_runner.cpp:201-203 (host side)
static constexpr uint32_t BULK_PAGE_SIZE       = 4096;
static constexpr uint32_t ACTION_OUTPUT_BYTES   = 3200;   // 50x32 bfloat16
static constexpr uint32_t SENTINEL_SLOT_ID      = 0xFFFFFFFF;
```

```cpp
// kernels/pi05_bulk_passthrough.cpp:31-32 (device side — named constants)
static constexpr uint32_t SENTINEL_SLOT_ID      = 0xFFFFFFFF;
static constexpr uint32_t ACTION_OUTPUT_BYTES   = 3200;   // 50x32 bfloat16

// kernels/pi05_bulk_passthrough.cpp:37 (device side — compile-time arg)
constexpr uint32_t page_size = get_compile_time_arg_val(2);  // = BULK_PAGE_SIZE
```

`BULK_PAGE_SIZE` is the DMA page granularity imposed by the tt-metal socket layer. Every transfer---regardless of actual payload size---must be an integer multiple of 4096 bytes. `ACTION_OUTPUT_BYTES` is the fixed size of the bfloat16 action tensor produced by the model: 50 action dimensions times 32 bfloat16 values each, at 2 bytes per bfloat16: $50 \times 32 \times 2 = 3{,}200$ bytes. `SENTINEL_SLOT_ID` is the magic value that signals shutdown.

---

## 4.1.2 Header Formats

Both directions use an 8-byte header consisting of two `uint32_t` fields:

```cpp
// examples/pi05_pipeline_runner.cpp:205-215
struct BulkH2DHeader {
    uint32_t slot_id;
    uint32_t payload_length;
};
static_assert(sizeof(BulkH2DHeader) == 8);

struct BulkD2HHeader {
    uint32_t slot_id;
    uint32_t payload_length;
};
static_assert(sizeof(BulkD2HHeader) == 8);
```

| Field            | Type       | Bytes | Meaning                                            |
|------------------|------------|-------|-----------------------------------------------------|
| `slot_id`        | `uint32_t` | 4     | User session slot assigned by `DecodeScheduler`     |
| `payload_length` | `uint32_t` | 4     | Byte count of data following the header (unpadded)  |

The `static_assert` guards ensure the structs are exactly 8 bytes with no padding, which is critical because the header is written into the first 8 bytes of the staging buffer via `std::memcpy` and the device kernel reads it at word offsets `[0]` and `[1]`:

```cpp
// kernels/pi05_bulk_passthrough.cpp:88-89
uint32_t slot_id        = first_page[0];
uint32_t payload_length = first_page[1];
```

### Why Separate Structs?

The headers are identical in bit layout. The distinction is semantic: type safety. A function accepting `BulkH2DHeader*` cannot accidentally receive a `BulkD2HHeader*` even though the bit layout is the same, preventing accidental cross-wiring of the two socket directions.

The device kernel echoes the `slot_id` from the received H2D header back in the D2H header (line 175), enabling the host to detect ordering violations.

---

## 4.1.3 Page Alignment and Padding

Every bulk transfer must be padded to a whole number of 4096-byte pages. The host computes this as:

```cpp
// examples/pi05_pipeline_runner.cpp:516-517
uint32_t total    = sizeof(BulkH2DHeader) + payload_length;
uint32_t num_pages = (total + BULK_PAGE_SIZE - 1) / BULK_PAGE_SIZE;
uint32_t required  = num_pages * BULK_PAGE_SIZE;
```

In LaTeX notation, for a payload of $P$ bytes:

$$
\text{num\_pages} = \left\lceil \frac{8 + P}{4096} \right\rceil
$$

$$
\text{wire\_bytes} = \text{num\_pages} \times 4096
$$

The page size of 4096 bytes matches the underlying PCIe alignment requirements enforced by the tt-metal `H2DSocket` and `D2HSocket` implementations. The `set_page_size()` call in both channel constructors configures this at the socket level.

The padding bytes between the end of the actual payload and the page boundary are zero-filled by the host (line 523: `std::memset(staging_.data(), 0, required)`) to prevent information leakage from stale buffer contents.

**Transfer modes:** The page-aligned transport has two physical transfer modes controlled by `SocketConfigParams::h2d_mode`:

- **HOST_PUSH** (`H2DMode::HOST_PUSH`): The host pushes pages into device L1 via PCIe TLB writes. Data is available in L1 when `socket_wait_for_pages()` returns on the kernel side.
- **DEVICE_PULL** (`H2DMode::DEVICE_PULL`): The host writes to pinned memory only; the kernel must explicitly issue `noc_read_page_chunked()` calls to pull each page from host RAM into L1 over the NOC/PCIe path. This adds per-page ordering requirements (see [Section 4.4](./04_race_conditions.md) for atomicity implications).

The wire format is identical in both modes — only the physical transfer mechanism differs.

---

## 4.1.4 H2D Payload: Pixel Frames

The H2D payload consists of pixel data followed by optional action history. The pixel data size is governed by `PixelPayloadConfig`:

```cpp
// examples/pi05_pipeline_runner.cpp:231-239
struct PixelPayloadConfig {
    uint32_t width  = 224;
    uint32_t height = 224;
    uint32_t channels = 3;
    uint32_t frames   = 3;
    uint32_t action_history_bytes = 4096;

    uint32_t pixel_bytes() const { return width * height * channels * frames; }
    uint32_t total_bytes() const { return pixel_bytes() + action_history_bytes; }
};
```

With the default configuration:

$$
\text{pixel\_bytes} = 224 \times 224 \times 3 \times 3 = 451{,}584 \text{ bytes}
$$

$$
\text{total\_payload} = 451{,}584 + 4{,}096 = 455{,}680 \text{ bytes}
$$

$$
\text{wire\_total} = 8 + 455{,}680 = 455{,}688 \text{ bytes}
$$

$$
\text{num\_pages} = \left\lceil \frac{455{,}688}{4096} \right\rceil = 112 \text{ pages}
$$

$$
\text{padded\_size} = 112 \times 4096 = 458{,}752 \text{ bytes}
$$

The wire layout for a single H2D frame is:

```
Byte offset    Content
-----------------------------------------------
0x0000         slot_id (4 bytes)
0x0004         payload_length (4 bytes)
0x0008         pixel_data[0..pixel_bytes-1]
0x0008+P       action_history[0..action_history_bytes-1]
...            zero-pad to next 4096 boundary
-----------------------------------------------
Total: num_pages * 4096 bytes
```

This is a significant transfer: 112 pages per user per turn. The config-validation code at line 848-853 verifies that the padded size does not exceed the FIFO:

```cpp
// examples/pi05_pipeline_runner.cpp:848-853
uint32_t max_payload = sizeof(BulkH2DHeader) + pipeline_config.pixel.total_bytes();
uint32_t max_padded  = ((max_payload + BULK_PAGE_SIZE - 1) / BULK_PAGE_SIZE) * BULK_PAGE_SIZE;
if (max_padded > pipeline_config.socket.fifo_size) {
    throw std::runtime_error("Padded H2D payload (" + std::to_string(max_padded) +
        " bytes) exceeds FIFO size (" + std::to_string(pipeline_config.socket.fifo_size) + ")");
}
```

The default FIFO size is 524,288 bytes (512 KB), which accommodates the default padded payload of 458,752 bytes with 65,536 bytes to spare.

### First Turn vs. Subsequent Turns

On the first turn, no action history is available, so `action_data` is `nullptr` and `action_bytes` is 0:

```cpp
// examples/pi05_pipeline_runner.cpp:991-993
double h2d_us = bulk_h2d->send(users[u].slot_id,
    pixel_data.data(), static_cast<uint32_t>(pixel_data.size()),
    nullptr, 0);
```

On subsequent turns, the action output from the previous turn is recycled as action history:

```cpp
// examples/pi05_pipeline_runner.cpp:1098-1104
std::memcpy(action_history.data(), action_output.data(),
    std::min(static_cast<uint32_t>(action_output.size()),
             pipeline_config.pixel.action_history_bytes));

double h2d_us = bulk_h2d->send(u.slot_id,
    pixel_data.data(), static_cast<uint32_t>(pixel_data.size()),
    action_history.data(), pipeline_config.pixel.action_history_bytes);
```

This means H2D payload size varies between turns:

- **Turn 1:** 451,584 bytes (pixels only). Wire total: $8 + 451{,}584 = 451{,}592$ bytes. Pages: $\lceil 451{,}592 / 4096 \rceil = 111$ pages. Padded: $111 \times 4096 = 454{,}656$ bytes.
- **Turn 2+:** 455,680 bytes (pixels + action history). Wire total: $8 + 455{,}680 = 455{,}688$ bytes. Pages: $\lceil 455{,}688 / 4096 \rceil = 112$ pages. Padded: $112 \times 4096 = 458{,}752$ bytes.

---

## 4.1.5 D2H Payload: Action Output

The D2H direction is much simpler. The device always sends exactly one page:

```
Byte offset    Content
-----------------------------------------------
0x0000         slot_id (4 bytes)
0x0004         payload_length = 3200 (4 bytes)
0x0008         action_output[0..3199] (bfloat16)
0x0C88         zero-pad (888 bytes to fill page)
-----------------------------------------------
Total: 4096 bytes (1 page)
```

$$
8 + 3200 = 3208 \text{ bytes} < 4096 \text{ bytes} \implies \text{num\_pages} = 1
$$

The host receiver hardcodes this:

```cpp
// examples/pi05_pipeline_runner.cpp:575-576
// 8B header + 3200B payload = 3208B < 4096B = 1 page
uint32_t num_pages = 1;
```

The device kernel writes the action output as an incrementing pattern in the passthrough implementation:

```cpp
// kernels/pi05_bulk_passthrough.cpp:179-183
uint32_t action_start_word = 2;  // after 8-byte header
uint32_t action_words = ACTION_OUTPUT_BYTES / sizeof(uint32_t);
for (uint32_t w = 0; w < action_words; w++) {
    output_cb_addr[action_start_word + w] = w + slot_id;
}
```

---

## 4.1.6 H2D/D2H Asymmetry

The protocol is strongly asymmetric:

| Property          | H2D (Pixel Ingestion)    | D2H (Action Output)    |
|-------------------|--------------------------|------------------------|
| Payload size      | 451,584--455,680 bytes   | 3,200 bytes (fixed)    |
| Pages per transfer| 111--112                 | 1                      |
| Padded wire bytes | 454,656--458,752         | 4,096                  |
| Direction         | Host -> Device           | Device -> Host         |
| Bandwidth demand  | High (multi-page)        | Low (single-page)      |
| First-turn special| No action history        | N/A                    |

This asymmetry is fundamental to the Pi0.5 workload: the model ingests high-resolution multi-frame video but produces only a compact action vector. The H2D direction dominates transfer time, while D2H is effectively instantaneous relative to a single PCIe transaction. It also means the H2D FIFO must be sized for at least one full padded payload, while the D2H FIFO only needs to hold a single page per in-flight user.

---

## 4.1.7 Sentinel: The Shutdown Marker

A sentinel is a special H2D message that signals the device kernel to terminate:

```cpp
// Sentinel wire format: exactly one page
// Byte 0..3: slot_id = 0xFFFFFFFF
// Byte 4..7: payload_length = 0
// Byte 8..4095: zeros
```

The sentinel is also defined in the device kernel:

```cpp
// kernels/pi05_bulk_passthrough.cpp:26
static constexpr uint32_t SENTINEL_SLOT_ID = 0xFFFFFFFF;
```

The device kernel checks for the sentinel on every iteration at line 92 of `pi05_bulk_passthrough.cpp`:

```cpp
// kernels/pi05_bulk_passthrough.cpp:92
if (slot_id == SENTINEL_SLOT_ID) {
```

Upon receiving the sentinel, the kernel echoes it back via D2H (a single page with `slot_id = 0xFFFFFFFF`) and exits the main loop. The host reads this echo to confirm the kernel has shut down. The sentinel design relies on the fact that valid `slot_id` values are assigned by the DecodeScheduler and will never equal `0xFFFFFFFF`. The full shutdown protocol is detailed in [Section 4.3](03_shutdown_protocol.md).

---

**Next:** [Bulk Channel Classes](02_bulk_channel_classes.md)
