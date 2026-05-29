# 5.3 Bulk Passthrough Kernel

> **Source file:**
> - `kernels/pi05_bulk_passthrough.cpp` (lines 1--211)

The `pi05_bulk_passthrough` kernel runs on a Tensix core and implements the device side of the bulk H2D/D2H protocol. It receives multi-page pixel frames from the host over PCIe, discards the pixel data (passthrough behavior), constructs a synthetic action-output tensor, and writes it back to the host via D2H. This section traces the kernel from initialization through its main loop to cleanup, with particular attention to the ordering constraints between socket operations and NOC transfers.

---

## 5.3.1 Kernel Entry and Compile-Time Arguments

The kernel retrieves its configuration via `get_compile_time_arg_val()`, which reads values baked into the kernel binary at compile time by the host-side `CreateKernel()` call:

```cpp
// kernels/pi05_bulk_passthrough.cpp:35-39
constexpr uint32_t recv_socket_config_addr = get_compile_time_arg_val(0);
constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t output_cb_index = get_compile_time_arg_val(3);
constexpr bool pull_from_host = get_compile_time_arg_val(4);
```

The `constexpr` keyword is critical for device kernels: it enables the compiler to resolve conditionals like `if constexpr (pull_from_host)` at compile time, eliminating dead code from the final binary and saving both instruction memory and branch overhead on the resource-constrained RISC-V core.

| Argument | Index | Typical value | Purpose |
|----------|-------|--------------|---------|
| `recv_socket_config_addr` | 0 | (L1 address) | Address of H2D socket config struct in L1 |
| `sender_socket_config_addr` | 1 | (L1 address) | Address of D2H socket config struct in L1 |
| `page_size` | 2 | 4096 | Page size for both H2D and D2H transfers |
| `output_cb_index` | 3 | 0 | Circular buffer index for staging D2H output |
| `pull_from_host` | 4 | 0 or 1 | Whether kernel must perform NOC reads (DEVICE_PULL) |

The `pull_from_host` argument is declared as `bool` but receives a `uint32_t` from the compile args. The compiler performs the implicit conversion; any non-zero value is treated as `true`.

---

## 5.3.2 Socket Interface Setup

The kernel creates receiver (H2D) and sender (D2H) socket interfaces from the config addresses:

```cpp
// kernels/pi05_bulk_passthrough.cpp:41-52
volatile tt_l1_ptr uint32_t* output_cb_addr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_write_ptr(output_cb_index));

SocketReceiverInterface receiver =
    create_receiver_socket_interface(recv_socket_config_addr);
SocketSenderInterface sender =
    create_sender_socket_interface(sender_socket_config_addr);

set_receiver_socket_page_size(receiver, page_size);
set_sender_socket_page_size(sender, page_size);
```

The `output_cb_addr` pointer is the L1 address of the circular buffer, used as a staging area for constructing D2H output pages before NOC-writing them to the host. The `volatile tt_l1_ptr` qualifier ensures the compiler does not optimize away writes to this address -- the NOC hardware reads from this physical L1 location, and the compiler must not reorder or elide stores.

`create_receiver_socket_interface()` and `create_sender_socket_interface()` read the socket configuration structures from the L1 addresses that were set up by the host during socket creation. These structures contain FIFO base addresses, read/write pointers, page sizes, and PCIe address mappings.

### Pre-Computed NOC Addresses

```cpp
// kernels/pi05_bulk_passthrough.cpp:56-65
uint32_t write_addr_hi = sender.d2h.data_addr_hi;
uint32_t d2h_pcie_xy_enc = sender.d2h.pcie_xy_enc;

[[maybe_unused]] uint32_t h2d_pcie_xy_enc = receiver.h2d.pcie_xy_enc;
[[maybe_unused]] uint64_t pcie_data_addr = 0;
if constexpr (pull_from_host) {
    pcie_data_addr = (static_cast<uint64_t>(receiver.h2d.data_addr_hi) << 32) |
                     (static_cast<uint64_t>(receiver.h2d.data_addr_lo));
}
```

The kernel pre-computes PCIe NOC coordinates outside the main loop to avoid repeated struct field accesses in the hot path:

- `write_addr_hi`: Upper 32 bits of the D2H PCIe target address (used to form the 64-bit NOC write destination).
- `d2h_pcie_xy_enc`: Encoded NOC X/Y coordinates of the PCIe endpoint for D2H writes.
- `h2d_pcie_xy_enc`: Encoded NOC coordinates of the PCIe endpoint for H2D reads (only used in DEVICE_PULL mode).
- `pcie_data_addr`: Full 64-bit base address of the host's pinned H2D buffer in PCIe address space.

The `[[maybe_unused]]` attributes suppress warnings when `pull_from_host` is `false` (HOST_PUSH mode), in which case the H2D-related variables are never referenced.

### NOC Write State Initialization

```cpp
// kernels/pi05_bulk_passthrough.cpp:67
noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
```

This initializes the NOC write command buffer with unicast write settings. It is called once before the main loop because the D2H write parameters (virtual channel, command buffer) do not change between iterations.

---

## 5.3.3 Main Loop: Step-by-Step

The main loop processes one H2D transfer per iteration. Each iteration follows this sequence:

```
wait -> [pull] -> parse header -> sentinel check -> pop first page ->
consume remaining pages -> construct D2H -> NOC write -> push D2H
```

### Step 1: Wait for First H2D Page

```cpp
// kernels/pi05_bulk_passthrough.cpp:71
socket_wait_for_pages(receiver, 1);
```

This spins until the H2D socket's flow-control counter indicates at least one page is available. In HOST_PUSH mode, this means the host has written the page into L1. In DEVICE_PULL mode, it means the host has written the page into pinned host RAM and updated the flow-control pointer, but the data is **not yet in L1**.

### Step 2: Pull Page from Host (DEVICE_PULL Only)

```cpp
// kernels/pi05_bulk_passthrough.cpp:75-82
if constexpr (pull_from_host) {
    noc_read_page_chunked(
        h2d_pcie_xy_enc,
        pcie_data_addr + receiver.read_ptr - receiver.fifo_addr,
        receiver.read_ptr,
        page_size);
    noc_async_read_barrier();
}
```

In DEVICE_PULL mode, the kernel must explicitly fetch the page from host pinned RAM into L1 via NOC reads over PCIe. The source address is computed as an offset into the host-side circular buffer:

$$\text{src\_pcie} = \texttt{pcie\_data\_addr} + (\texttt{read\_ptr} - \texttt{fifo\_addr})$$

where `read_ptr` is the current read position in the L1 FIFO and `fifo_addr` is the FIFO base. The offset `read_ptr - fifo_addr` maps the L1 FIFO position to the corresponding host-side buffer position.

The `noc_async_read_barrier()` blocks until the NOC read completes, ensuring the page data is fully in L1 before the kernel reads the header fields. The `noc_read_page_chunked()` function (Section 5.4) handles the transfer in `NOC_MAX_BURST_SIZE` chunks.

In HOST_PUSH mode, this block is compiled away entirely by `if constexpr`, producing zero code.

### Step 3: Parse Header

```cpp
// kernels/pi05_bulk_passthrough.cpp:84-89
volatile tt_l1_ptr uint32_t* first_page =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        receiver.read_ptr);

uint32_t slot_id = first_page[0];
uint32_t payload_length = first_page[1];
```

The first two 32-bit words of the first page are the `BulkH2DHeader`:

| Word | Field | Size |
|------|-------|------|
| `first_page[0]` | `slot_id` | 4 bytes |
| `first_page[1]` | `payload_length` | 4 bytes |

Total header: 8 bytes. The remaining $4096 - 8 = 4088$ bytes of the first page contain the beginning of the pixel payload.

### Step 4: Sentinel Check

```cpp
// kernels/pi05_bulk_passthrough.cpp:92
if (slot_id == SENTINEL_SLOT_ID) {
```

If the `slot_id` is `0xFFFFFFFF`, the kernel enters the sentinel-handling path (covered in Section 5.3.4) and breaks out of the main loop.

### Step 5: Pop First Page and Consume Remaining

```cpp
// kernels/pi05_bulk_passthrough.cpp:129-131
uint32_t total_bytes = 8 + payload_length;  // header + payload
uint32_t total_pages = (total_bytes + page_size - 1) / page_size;
```

The total page count is computed with ceiling division:

$$N_{\text{pages}} = \left\lceil \frac{8 + \text{payload\_length}}{\text{page\_size}} \right\rceil$$

For the default pixel configuration ($8 + 451{,}584 + 4{,}096 = 455{,}688$ bytes), this is $\lceil 455{,}688 / 4096 \rceil = 112$ pages.

The first page is popped immediately:

```cpp
// kernels/pi05_bulk_passthrough.cpp:143-146
socket_pop_pages(receiver, 1);
noc_async_writes_flushed();
socket_notify_sender(receiver);
```

The `socket_pop_pages()` advances the receiver's read pointer. The `noc_async_writes_flushed()` ensures any pending NOC writes (from a previous iteration's D2H) are flushed before the pop notification is sent. The `socket_notify_sender()` signals the host that the page slot is free for reuse.

> **PRODUCTION KERNEL NOTE**: The source code contains an explicit inline comment (lines 133-141) warning that this pop-before-D2H ordering is only safe in the passthrough kernel because it does not need the H2D data to construct its response. A real vision-encoder kernel must defer pops until after copying out the pixel data and flushing the D2H write.

Remaining pages are consumed in a loop:

```cpp
// kernels/pi05_bulk_passthrough.cpp:149-164
for (uint32_t p = 1; p < total_pages; p++) {
    socket_wait_for_pages(receiver, 1);
    if constexpr (pull_from_host) {
        noc_read_page_chunked(
            h2d_pcie_xy_enc,
            pcie_data_addr + receiver.read_ptr - receiver.fifo_addr,
            receiver.read_ptr,
            page_size);
        noc_async_read_barrier();
    }
    socket_pop_pages(receiver, 1);
    noc_async_writes_flushed();
    socket_notify_sender(receiver);
}
```

Each remaining page goes through wait -> pull (if DEVICE_PULL) -> pop -> notify. In the passthrough kernel, the pixel data is discarded -- the pages are consumed only to maintain correct flow control. Note that in DEVICE_PULL mode, the kernel must pull every page even though the passthrough discards the data -- the protocol requires the device to read before pop for correct flow control.

### Step 6: Construct D2H Action Output

```cpp
// kernels/pi05_bulk_passthrough.cpp:167-183
socket_reserve_pages(sender, 1);

// Zero the output page
for (uint32_t w = 0; w < page_size_words; w++) {
    output_cb_addr[w] = 0;
}

// Write D2H header
output_cb_addr[0] = slot_id;                // echo slot_id from H2D
output_cb_addr[1] = ACTION_OUTPUT_BYTES;     // payload_length

// Write synthetic action output (passthrough: incrementing pattern)
uint32_t action_start_word = 2;  // after 8-byte header
uint32_t action_words = ACTION_OUTPUT_BYTES / sizeof(uint32_t);
for (uint32_t w = 0; w < action_words; w++) {
    output_cb_addr[action_start_word + w] = w + slot_id;
}
```

The `socket_reserve_pages(sender, 1)` blocks until the D2H socket has a free page slot. The output CB is then zeroed (to prevent leaking stale data in padding bytes), the header is written, and a synthetic action payload is generated with an incrementing pattern seeded by `slot_id`.

### D2H Page Layout

| Offset (bytes) | Field | Size | Value |
|----------------|-------|------|-------|
| 0 | `slot_id` | 4 B | Echoed from H2D header |
| 4 | `payload_length` | 4 B | `ACTION_OUTPUT_BYTES` = 3200 |
| 8 | Action output | 3200 B | Synthetic incrementing pattern |
| 3208 | Padding | 888 B | Zero-filled |

Total: $8 + 3200 + 888 = 4096$ bytes = 1 page. The `ACTION_OUTPUT_BYTES` constant is 3200 bytes, representing a $50 \times 32$ bfloat16 tensor (50 action dimensions, 32 values each, 2 bytes per bfloat16).

The synthetic payload pattern (`output_cb_addr[action_start_word + w] = w + slot_id`) provides a deterministic, slot-specific pattern that the host can validate: if the host receives a D2H page with `slot_id = 3`, word 5 of the action data should be `5 + 3 = 8`. This is purely for testing; a production kernel would write actual model output here.

### Step 7: NOC Write to D2H and Push

```cpp
// kernels/pi05_bulk_passthrough.cpp:186-201
noc_wwrite_with_state<noc_mode, write_cmd_buf,
    CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
    NOC_INDEX,
    get_write_ptr(output_cb_index),
    d2h_pcie_xy_enc,
    ((static_cast<uint64_t>(write_addr_hi) << 32)
        | sender.downstream_fifo_addr) + sender.write_ptr,
    page_size,
    1);

// Barrier BEFORE push
noc_async_write_barrier();

socket_push_pages(sender, 1);
socket_notify_receiver(sender);
```

The NOC write transfers the entire output page from the CB in L1 to the D2H socket's FIFO in host pinned RAM. The destination address is constructed from:

- `write_addr_hi` (upper 32 bits of PCIe BAR)
- `sender.downstream_fifo_addr` (base of D2H FIFO in host memory)
- `sender.write_ptr` (current write offset in the FIFO)

The `noc_async_write_barrier()` is placed **before** `socket_push_pages()`. This ordering is critical: the barrier ensures the NOC write has completed (data is in host RAM) before the sender's write pointer is advanced. Without this barrier, the host could read stale/partial data from the FIFO slot.

After the barrier, `socket_push_pages()` advances the sender's write pointer, and `socket_notify_receiver()` signals the host that a new page is available for reading.

---

## 5.3.4 Sentinel Handling

When the kernel receives a sentinel (`slot_id == 0xFFFFFFFF`), it echoes the sentinel back to the host before exiting:

```cpp
// kernels/pi05_bulk_passthrough.cpp:96-125
// ---- Sentinel echo ----
socket_reserve_pages(sender, 1);

// Write sentinel marker into D2H output page
for (uint32_t w = 0; w < page_size_words; w++) {
    output_cb_addr[w] = 0;
}
output_cb_addr[0] = SENTINEL_SLOT_ID;  // sentinel slot_id echo

// NOC write to D2H socket
noc_wwrite_with_state<noc_mode, write_cmd_buf,
    CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
    NOC_INDEX,
    get_write_ptr(output_cb_index),
    d2h_pcie_xy_enc,
    ((static_cast<uint64_t>(write_addr_hi) << 32)
        | sender.downstream_fifo_addr) + sender.write_ptr,
    page_size,
    1);

// Barrier BEFORE push -- data must be committed
noc_async_write_barrier();
socket_push_pages(sender, 1);
socket_notify_receiver(sender);

// Pop the sentinel H2D page AFTER D2H echo is flushed
socket_pop_pages(receiver, 1);
noc_async_writes_flushed();
socket_notify_sender(receiver);

break;
```

The sentinel handling follows a precise ordering protocol:

1. **Reserve D2H space**: `socket_reserve_pages(sender, 1)` ensures the D2H FIFO has room for the echo page.
2. **Zero-fill and write sentinel marker**: The entire output page is zeroed, then `SENTINEL_SLOT_ID` (0xFFFFFFFF) is written to word 0. The `payload_length` field (word 1) and remaining words are zero.
3. **NOC write to D2H**: The output CB contents are written to the D2H FIFO in host pinned RAM.
4. **Write barrier**: `noc_async_write_barrier()` blocks until the write completes. Data must be fully committed before the push.
5. **Push and notify D2H**: `socket_push_pages` advances the write pointer; `socket_notify_receiver` signals the host.
6. **Pop H2D sentinel page**: Released **after** the D2H echo is flushed, not before.
7. **Flush and notify H2D**: `noc_async_writes_flushed()` ensures the pop notification propagates; `socket_notify_sender` signals the host.
8. **Break**: Exit the main loop.

The sentinel echo differs from normal operation in two important ways:

- **Pop timing**: The sentinel H2D page is popped **after** the D2H echo is flushed, not before as in the normal path. This ensures the echo is fully committed to host RAM before the sentinel page is released. If the kernel popped first, the host might see the freed H2D space and attempt to write a new page before the echo arrives.
- **D2H echo payload**: Only word 0 is set to `SENTINEL_SLOT_ID`. The host checks only `word0 == SENTINEL_SLOT_ID` to confirm the echo (see `recv_sentinel_echo()` in the runner).

---

## 5.3.5 Cleanup

After the main loop exits (sentinel received), the kernel performs final cleanup:

```cpp
// kernels/pi05_bulk_passthrough.cpp:206-211
update_socket_config(receiver);
update_socket_config(sender);
socket_barrier(sender);
noc_async_write_barrier();
noc_async_read_barrier();
```

| Call | Purpose |
|------|---------|
| `update_socket_config(receiver)` | Writes the receiver's final FIFO state (read pointer) back to the config struct in L1 |
| `update_socket_config(sender)` | Writes the sender's final FIFO state (write pointer) back to the config struct in L1 |
| `socket_barrier(sender)` | Ensures all D2H pages have been fully delivered to the host (higher-level than NOC barrier -- waits for host acknowledgment) |
| `noc_async_write_barrier()` | Ensures all NOC writes (including config updates) are complete |
| `noc_async_read_barrier()` | Ensures all NOC reads are complete (relevant for DEVICE_PULL mode) |

The `update_socket_config()` calls are necessary so that the host can query the final socket state after the kernel exits, which is used for correctness validation in test scenarios. This cleanup sequence is identical to the one used by the `pipeline_loopback` kernel (lines 101-106 in `pipeline_loopback.cpp`), forming a standard pattern for all socket-based Tensix kernels.

---

## 5.3.6 Comparison with pipeline_loopback Kernel

Both kernels share the same structural pattern but differ in protocol:

| Aspect | `pipeline_loopback` | `pi05_bulk_passthrough` |
|--------|-------------------|----------------------|
| Core | `(0, 0)` | `(1, 0)` (full) or `(0, 0)` (bulk-only) |
| Page size | 256 bytes | 4096 bytes |
| H2D mode | HOST_PUSH only | DEVICE_PULL or HOST_PUSH |
| Pages per transfer | 1 (fixed) | 1..N (variable, based on payload) |
| D2H payload | 64 B (ResultPage) | 3208 B (header + action output) |
| Response construction | Mimics MockPipeline token logic | Synthetic incrementing pattern |
| Sentinel word | `slot_id == 0xFFFFFFFF` at `[0]` | `slot_id == 0xFFFFFFFF` at `[0]` |
| Pop timing | Pop after D2H write (push+notify) | Pop before D2H (normal) / after D2H (sentinel) |
| NOC read chunking | Not needed (HOST_PUSH) | `noc_read_page_chunked()` for DEVICE_PULL |
| Multi-page H2D | No | Yes (loop consumes remaining pages) |

The most significant difference is the pop timing. The token loopback kernel pops the H2D page **after** writing the D2H result:

```cpp
// kernels/pipeline_loopback.cpp, lines 89-92
socket_push_pages(sender_socket, 1);
socket_notify_receiver(sender_socket);
socket_pop_pages(receiver_socket, 1);       // pop AFTER push
noc_async_writes_flushed();
socket_notify_sender(receiver_socket);
```

The bulk passthrough kernel pops **before** writing D2H in normal operation. This is safe only because the passthrough kernel does not use the H2D data to construct the D2H response.

---

**Next:** [PCIe NOC Utilities](04_pcie_noc_utils.md)
