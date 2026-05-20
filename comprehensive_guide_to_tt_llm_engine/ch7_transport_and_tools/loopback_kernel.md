# The On-Device Loopback Kernel

[Prev: Chapter 7 Index](index.md) | [Next: Loopback Microbenchmark](loopback_microbenchmark.md)

---

The loopback kernel (`kernels/pipeline_loopback.cpp`) is a minimal Tensix data-movement program that reads inject pages from an H2D socket, applies the same deterministic token transformation as MockPipeline, and writes result pages to a D2H socket. It provides the device-side half of the loopback benchmark, enabling end-to-end testing of the DecodeScheduler + SocketPipeline stack against real Tenstorrent hardware without a production model.

**Source file:** `kernels/pipeline_loopback.cpp`

---

## Wire Format Compatibility

The kernel implements the **non-DeepSeek (default) wire format** from [Ch5 wire_format.md](../ch5_pipeline_and_wire_format/wire_format.md). The word indices are documented in a comment at lines 17-21:

### InjectPage (H2D) -- 16 words (64 bytes)

| Word Index | Field | Used by Kernel |
|---|---|---|
| `[0]` | `slot_id` | Yes -- echoed to result |
| `[1]` | `token_id` | Yes -- used to compute `actual_token` |
| `[2]` | `position` | Yes -- used to compute `actual_token_pos` |
| `[3]` | `prefill_token_id` | Yes -- controls prefill vs decode branch |
| `[4]` | `spec_flag` | No |
| `[5-7]` | Sampling params | No |
| `[8-15]` | Reserved | No |

### ResultPage (D2H) -- 16 words (64 bytes)

| Word Index | Field | Set by Kernel |
|---|---|---|
| `[0]` | `slot_id` | Yes -- copied from inject |
| `[1]` | `actual_token` | Yes -- computed |
| `[2]` | `predicted_token` | Yes -- always EMPTY_TOKEN |
| `[3-5]` | Reserved | Zeroed |
| `[6]` | `actual_token_pos` | Yes -- `position + 1` |
| `[7-15]` | Reserved | Zeroed |

**Important**: The loopback kernel uses `PAGE_SIZE_WORDS = 16` (64 bytes), not the DeepSeek MD format's larger page size. The launcher allocates circular buffers and sockets matching this smaller page size (`PAGE_SIZE_BYTES = 64` in `dummy_pipeline_launcher.cpp` line 48). The host-side SocketPipeline must be configured with matching page sizes for the non-DeepSeek format.

---

## Constants and Sentinels

```cpp
// kernels/pipeline_loopback.cpp, lines 23-25
static constexpr uint32_t SENTINEL_USER_ID = 0xFFFFFFFF;  // -1 as uint32
static constexpr uint32_t EMPTY_TOKEN = 0xFFFFFFFF;       // -1 as uint32
static constexpr uint32_t PAGE_SIZE_WORDS = 16;
```

| Constant | Value | Matches Host-Side |
|----------|-------|-------------------|
| `SENTINEL_USER_ID` | `0xFFFFFFFF` | `INVALID_SLOT` in `pipeline_types.hpp` |
| `EMPTY_TOKEN` | `0xFFFFFFFF` | `EMPTY_TOKEN` in `decode_types.hpp` |
| `PAGE_SIZE_WORDS` | 16 | 64 bytes / 4 bytes per word |

---

## Kernel Entry Point and Socket Setup

```cpp
// kernels/pipeline_loopback.cpp, lines 27-45
void kernel_main() {
    constexpr uint32_t recv_socket_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t sender_socket_config_addr = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(3);

    volatile tt_l1_ptr uint32_t* output_cb_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(output_cb_index));

    SocketReceiverInterface receiver_socket =
        create_receiver_socket_interface(recv_socket_config_addr);
    SocketSenderInterface sender_socket =
        create_sender_socket_interface(sender_socket_config_addr);

    set_receiver_socket_page_size(receiver_socket, page_size);
    set_sender_socket_page_size(sender_socket, page_size);

    uint32_t write_addr_hi = sender_socket.d2h.data_addr_hi;
    uint32_t pcie_xy_enc = receiver_socket.h2d.pcie_xy_enc;

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
```

The four compile-time arguments are supplied by the launcher (see [loopback_microbenchmark.md](loopback_microbenchmark.md)):

| Index | Parameter | Typical Value |
|-------|-----------|---------------|
| 0 | H2D socket config buffer address | From `h2d_socket.get_config_buffer_address()` |
| 1 | D2H socket config buffer address | From `d2h_socket.get_config_buffer_address()` |
| 2 | Page size in bytes | 64 (PAGE_SIZE_BYTES from launcher) |
| 3 | Output circular buffer index | `tt::CBIndex::c_0` |

The `output_cb_addr` pointer points to a circular buffer in L1 used as a scratch area for constructing result pages before writing them over PCIe to the D2H socket. The `write_addr_hi` and `pcie_xy_enc` values are cached before the main loop to avoid re-reading them from the socket struct on every iteration.

---

## Main Processing Loop

The kernel runs a tight loop that processes one inject page per iteration:

```
+---> socket_wait_for_pages(receiver, 1)     // Block until H2D has a page
|     Read inject fields from L1
|     socket_reserve_pages(sender, 1)         // Ensure D2H has space
|     Zero output page
|     Compute result fields
|     NOC write output page to D2H socket
|     socket_push_pages(sender, 1)            // Advance D2H write pointer
|     socket_notify_receiver(sender)           // Signal host D2H has data
|     socket_pop_pages(receiver, 1)            // Advance H2D read pointer
|     noc_async_writes_flushed()               // Ensure PCIe write completed
|     socket_notify_sender(receiver)           // Signal host H2D slot is free
|     if (slot_id == SENTINEL) break
+-----/
```

### Reading the Inject Page

```cpp
// Lines 48-58
socket_wait_for_pages(receiver_socket, 1);

volatile tt_l1_ptr uint32_t* in_page =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.read_ptr);

uint32_t slot_id = in_page[0];
uint32_t token_id = in_page[1];
uint32_t position = in_page[2];
uint32_t prefill_token_id = in_page[3];
```

`socket_wait_for_pages` is a blocking call that spins until the H2D socket's FIFO contains at least one unread page. The page data is accessible at `receiver_socket.read_ptr` in L1 memory -- no copy is needed; the kernel reads directly from the FIFO. The `volatile tt_l1_ptr` qualifier prevents the compiler from optimizing away reads from device-mapped memory.

### Computing the Result

```cpp
// Lines 61-77
socket_reserve_pages(sender_socket, 1);

for (uint32_t w = 0; w < PAGE_SIZE_WORDS; w++) {
    output_cb_addr[w] = 0;
}

output_cb_addr[0] = slot_id;
output_cb_addr[6] = position + 1;  // actual_token_pos

if (prefill_token_id == EMPTY_TOKEN) {
    output_cb_addr[1] = token_id + 1;  // actual_token = token_id + 1
    output_cb_addr[2] = EMPTY_TOKEN;   // predicted_token = -1
} else {
    output_cb_addr[1] = EMPTY_TOKEN;   // actual_token = -1
    output_cb_addr[2] = EMPTY_TOKEN;   // predicted_token = -1
}
```

### Token Transformation Model

The kernel implements two cases, matching MockPipeline's behavior exactly:

| Input Mode | Condition | actual_token [1] | predicted_token [2] | actual_token_pos [6] |
|-----------|-----------|-------------------|--------------------|-----------------------|
| **Prefill** | `prefill_token_id != EMPTY_TOKEN` | `EMPTY_TOKEN` (-1) | `EMPTY_TOKEN` (-1) | `position + 1` |
| **Decode** | `prefill_token_id == EMPTY_TOKEN` | `token_id + 1` | `EMPTY_TOKEN` (-1) | `position + 1` |

This model ensures:
- **Prefill tokens** produce no sampled output (actual_token = -1), which the Reader thread on the host skips via the `prefill_in_flight` counter (see [Ch3 chunked_prefill.md](../ch3_scheduling_algorithm/chunked_prefill.md))
- **Decode tokens** produce a deterministic output (`token_id + 1`), enabling the scheduler's completion detection (EOS match, `max_new_tokens`) to work correctly
- **Position advances by 1** on every token, maintaining the monotonic KV position invariant
- **`predicted_token` is always `EMPTY_TOKEN`** -- the loopback kernel does not implement speculative prediction, so the host-side spec decode logic (see [Ch4](../ch4_speculative_decode/index.md)) treats every decode step as a non-speculative result

### Writing the Result via NOC

The result page is written from L1 to the host via NoC-initiated PCIe write (lines 80-87):

```cpp
noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND,
                      CQ_NOC_WAIT, true, false>(
    NOC_INDEX,
    get_write_ptr(output_cb_index),
    pcie_xy_enc,
    ((static_cast<uint64_t>(write_addr_hi) << 32) | sender_socket.downstream_fifo_addr) +
        sender_socket.write_ptr,
    page_size,
    1);
```

The target address is constructed from:
- `write_addr_hi` (upper 32 bits) from the D2H socket's data address
- `downstream_fifo_addr + write_ptr` (lower 32 bits) from the sender socket state
- These combine to form the 64-bit PCIe address of the D2H FIFO on the host

### Post-Write Synchronization

After the PCIe write, the kernel performs a careful synchronization sequence (lines 89-93):

```cpp
socket_push_pages(sender_socket, 1);      // Advance D2H write pointer
socket_notify_receiver(sender_socket);     // Notify host that data is available
socket_pop_pages(receiver_socket, 1);      // Consume H2D page (advance read pointer)
noc_async_writes_flushed();                // Ensure PCIe write is complete
socket_notify_sender(receiver_socket);     // Notify host that H2D space is free
```

The ordering matters:
1. `push_pages` makes the result visible to the host
2. `notify_receiver` wakes the host's D2H poll loop
3. `pop_pages` frees the H2D page for the next inject
4. `writes_flushed` ensures the PCIe write landed before we proceed
5. `notify_sender` tells the host it can write the next inject page

This handshake ensures that the host never reads incomplete data from D2H and never overwrites unread data in H2D.

---

## Sentinel Protocol

The kernel terminates when it receives a page with `slot_id == SENTINEL_USER_ID` (0xFFFFFFFF). The sentinel check happens **after** the result page has been written back (lines 96-98):

```cpp
if (slot_id == SENTINEL_USER_ID) {
    break;
}
```

This echo is critical because the host-side `SocketPipeline::shutdown()` (non-DeepSeek mode) sends the sentinel and drains the D2H socket until the sentinel is echoed back (see [Ch5 socket_pipeline.md](../ch5_pipeline_and_wire_format/socket_pipeline.md)). If the kernel exited before echoing, the host would hang waiting for a response that never arrives.

### Shutdown Sequence

After breaking from the main loop, the kernel performs final cleanup (lines 101-106):

```cpp
update_socket_config(receiver_socket);
update_socket_config(sender_socket);
socket_barrier(sender_socket);

noc_async_write_barrier();
noc_async_read_barrier();
```

- `update_socket_config` writes the final socket state (read/write pointers) back to the config buffers in L1, ensuring the host sees consistent state when it inspects the sockets after the kernel exits
- `socket_barrier` ensures all outstanding D2H socket operations have completed
- The NoC barriers ensure no stale writes or reads are in flight when the kernel exits

---

## End-to-End Data Flow Trace

Here is the concrete data flow for a single decode token:

```
Host (Writer thread)                    Device (Kernel)                    Host (Reader thread)

  serialize_inject()
  [0]=slot_id=3
  [1]=token_id=42
  [2]=position=100
  [3]=prefill_token_id=EMPTY
       |
       | H2D socket write
       v
                              socket_wait_for_pages(1)
                              Read: slot=3, token=42,
                                    pos=100, prefill=EMPTY

                              Decode mode: token+1 = 43

                              Write output_cb:
                              [0]=3 (slot_id)
                              [1]=43 (actual_token)
                              [2]=EMPTY (predicted)
                              [6]=101 (actual_token_pos)
                                    |
                                    | D2H PCIe write
                                    v
                                                           deserialize_result()
                                                           slot_id=3
                                                           actual_token=43
                                                           actual_token_pos=101
```

---

## Device API Summary

| API Call | Purpose | Source |
|---|---|---|
| `get_compile_time_arg_val(N)` | Read Nth compile-time argument | tt-metalium kernel API |
| `get_write_ptr(cb_index)` | Get circular buffer write pointer | dataflow_api.h |
| `create_receiver_socket_interface(addr)` | Create H2D receiver from config address | socket_api.h |
| `create_sender_socket_interface(addr)` | Create D2H sender from config address | socket_api.h |
| `set_receiver_socket_page_size(s, sz)` | Configure receiver page size | socket_api.h |
| `set_sender_socket_page_size(s, sz)` | Configure sender page size | socket_api.h |
| `socket_wait_for_pages(s, n)` | Block until n pages available to read | socket_api.h |
| `socket_reserve_pages(s, n)` | Block until n pages available to write | socket_api.h |
| `socket_push_pages(s, n)` | Advance sender write pointer by n pages | socket_api.h |
| `socket_pop_pages(s, n)` | Advance receiver read pointer by n pages | socket_api.h |
| `socket_notify_receiver(s)` | Signal downstream that data is ready | socket_api.h |
| `socket_notify_sender(s)` | Signal upstream that slot is freed | socket_api.h |
| `noc_write_init_state<...>(...)` | Initialize NoC write command buffer state | dataflow_api.h |
| `noc_wwrite_with_state<...>(...)` | NOC write (PCIe for D2H) | dataflow_api.h |
| `noc_async_writes_flushed()` | Wait for pending NOC writes to complete | dataflow_api.h |
| `update_socket_config(s)` | Write socket state back to config buffer | socket_api.h |
| `socket_barrier(s)` | Wait for all socket operations to complete | socket_api.h |
| `noc_async_write_barrier()` | Global NOC write barrier | dataflow_api.h |
| `noc_async_read_barrier()` | Global NOC read barrier | dataflow_api.h |

---

## Relationship to MockPipeline

The loopback kernel is the on-device equivalent of `MockPipeline` (see [Ch5](../ch5_pipeline_and_wire_format/mock_pipeline.md)). Both implement the same token transformation rules:

| Aspect | MockPipeline | Loopback Kernel |
|--------|-------------|-----------------|
| Environment | Host process, in-memory | Tensix core, via sockets |
| Prefill output | `actual_token = -1` | `actual_token = EMPTY_TOKEN` |
| Decode output | `actual_token = token_id + 1` | `actual_token = token_id + 1` |
| Position | `actual_token_pos = position + 1` | `actual_token_pos = position + 1` |
| Predicted token | `EMPTY_TOKEN` | `EMPTY_TOKEN` |
| Latency | Configurable (simulated) | Real H2D/D2H socket + NOC latency |
| Sentinel | `INVALID_SLOT` detection | `SENTINEL_USER_ID` (0xFFFFFFFF) detection |

The loopback kernel adds real PCIe/NOC latency and socket handshake overhead that MockPipeline does not model, making it valuable for measuring the control-plane's actual end-to-end performance on hardware.

## Limitations

1. **No speculative decode support** -- `predicted_token` is always `EMPTY_TOKEN`. The loopback kernel cannot exercise the ACCEPT/REJECT/CONTINUE/STALE classification from [Ch4](../ch4_speculative_decode/result_classification.md).
2. **No DeepSeek wire format** -- The kernel uses the default (non-DeepSeek) wire layout. The DeepSeek inference runner connects to a real DeepSeek model pipeline, not this kernel.
3. **Single-core only** -- The kernel runs on core (0,0) with a single receiver/sender pair. There is no multi-core or multi-device pipeline staging.
4. **No sampling** -- The kernel's token transformation is purely deterministic (`token_id + 1`). There is no temperature, top-p, or top-k influence on the output.
5. **No p_indices/p_scores** -- The result page does not populate the probability distribution fields used by relaxed acceptance in speculative decode.

These limitations are intentional -- the loopback kernel exists to measure scheduling and transport overhead, not model behavior.

---

[Prev: Chapter 7 Index](index.md) | [Next: Loopback Microbenchmark](loopback_microbenchmark.md)
