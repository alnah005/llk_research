# 4.4 Race Conditions and Concurrency Hazards

> **Source files:**
> - `examples/pi05_pipeline_runner.cpp` (lines 496--627, 1023--1150)
> - `kernels/pi05_bulk_passthrough.cpp` (lines 69--203)

The bulk socket protocol operates across three independent execution domains---the host CPU, the device kernel, and the PCIe DMA engine---with no shared-memory synchronization primitives between them. Coordination is achieved entirely through FIFO flow control (`socket_wait_for_pages`, `socket_pop_pages`, `socket_push_pages`) and NOC barriers. This section catalogues the concurrency hazards that exist in the current implementation, distinguishing between those that are safe by construction, those that are safe only for the passthrough kernel, and those that could cause data corruption or deadlock in production.

---

## 4.4.1 Header/Payload Atomicity

**Hazard:** Could the device kernel read a partial header or a header without its payload?

### The Guarantee

The host writes the entire frame (header + pixel data + action history + padding) as a single `socket_->write(staging_.data(), num_pages)` call:

```cpp
// examples/pi05_pipeline_runner.cpp:539
socket_->write(staging_.data(), num_pages);
```

The underlying `H2DSocket::write()` increments `bytes_sent` only after all pages have been written (in HOST_PUSH mode) or after the host buffer has been populated and the pointer has advanced (in DEVICE_PULL mode). The device kernel's `socket_wait_for_pages(receiver, 1)` checks a monotonically-increasing `bytes_sent` counter against `bytes_acked`:

```cpp
// kernels/pi05_bulk_passthrough.cpp:71
socket_wait_for_pages(receiver, 1);
```

Since the host writes all `num_pages` pages before advancing `bytes_sent`, and the kernel only proceeds after `bytes_sent` indicates at least 1 page is available, the kernel is guaranteed to see a complete first page (with valid header) by the time it reads `first_page[0]` and `first_page[1]`.

### The Subtlety

The kernel processes remaining pages one at a time. After reading the header from page 0, it calculates `total_pages` and then waits for each subsequent page individually:

```cpp
// kernels/pi05_bulk_passthrough.cpp:129-131
uint32_t total_bytes = 8 + payload_length;  // header + payload
uint32_t total_pages = (total_bytes + page_size - 1) / page_size;
```

```cpp
// kernels/pi05_bulk_passthrough.cpp:149-164
for (uint32_t p = 1; p < total_pages; p++) {
    socket_wait_for_pages(receiver, 1);
    // ... pull page if DEVICE_PULL ...
    socket_pop_pages(receiver, 1);
    noc_async_writes_flushed();
    socket_notify_sender(receiver);
}
```

This is safe because the host writes all `num_pages` before signaling. The kernel will find all pages available when it checks, though it processes and pops them one at a time for FIFO pressure management.

### What Could Go Wrong

If a future refactor split the host `write()` into per-page calls (e.g., for streaming large payloads), the kernel could see a partial set of pages. Since the passthrough kernel reads the header in the same iteration as the first `socket_wait_for_pages`, the header itself would still be safe (it fits entirely within page 0, since `sizeof(BulkH2DHeader) == 8 < 4096`). But the subsequent pages might not be available yet, causing the kernel to block in the consumption loop. There is no timeout on the device side for this wait.

**Verdict:** Safe. The first page always contains a complete header, and the remaining pages are available by the time the kernel loops to consume them.

---

## 4.4.2 Pop-Before-D2H Ordering

**Hazard:** The passthrough kernel pops H2D pages before writing the D2H response. Is this safe?

The kernel comments explicitly flag this:

```cpp
// kernels/pi05_bulk_passthrough.cpp:133-141
// ---- PASSTHROUGH-ONLY NOTE ----
// In this passthrough kernel, we pop H2D pages before writing D2H.
// This is safe because the passthrough kernel does not need H2D data
// to construct the D2H response.
//
// PRODUCTION KERNEL MUST DIFFER: A real kernel that reads pixel data
// from the H2D FIFO (e.g., to feed a vision encoder) MUST defer all
// pops until after the data has been copied out AND the D2H write
// has been flushed.
// ---- END NOTE ----
```

In the passthrough kernel, the sequence is:

1. Wait for first H2D page (line 71)
2. Read `slot_id` and `payload_length` from L1 (lines 88-89)
3. Pop the first page (line 144)
4. Consume and pop remaining pages (lines 149-164)
5. Construct D2H response (lines 167-202)

### Why This Is Safe for Passthrough

The passthrough kernel generates synthetic action output (an incrementing pattern seeded by `slot_id`). It does not read or use the pixel data at all. It only needs `slot_id` from the header (already read into a register at step 2). Popping the H2D pages early frees FIFO slots for the next user's transfer, improving pipeline throughput.

### Why This Is Dangerous for Production

A production vision-language kernel must:

1. Read pixel data from the H2D FIFO pages.
2. Feed it to the vision encoder.
3. Run the denoise/action-head inference.
4. Write the action output to the D2H socket.

If the kernel pops H2D pages before copying out the pixel data, the host may immediately overwrite the FIFO region with the next user's pixel data (since the `bytes_acked` counter advances on pop, freeing FIFO space). The kernel would then read corrupted pixel data.

The correct production sequence would be:

1. `socket_wait_for_pages(receiver, total_pages)` -- wait for all pages at once.
2. Copy pixel data from FIFO to a working buffer or directly to the encoder.
3. Process inference.
4. Write D2H output and `noc_async_write_barrier()`.
5. `socket_push_pages(sender, 1)` + `socket_notify_receiver(sender)`.
6. **Only then:** `socket_pop_pages(receiver, total_pages)` + `socket_notify_sender(receiver)`.

**Verdict:** Safe for the passthrough kernel; a critical correctness hazard for any production kernel that reads pixel data.

---

## 4.4.3 D2H Slot ID Mismatch

**Hazard:** The D2H `slot_id` may not match the expected value.

The host checks for mismatches but does not take corrective action:

```cpp
// examples/pi05_pipeline_runner.cpp:1075-1078
if (result_slot_id != out.slot_id) {
    std::cerr << "WARNING: D2H slot_id mismatch: expected "
              << out.slot_id << " got " << result_slot_id << std::endl;
}
```

### How a Mismatch Can Occur

The token pipeline (`DecodeScheduler` + `SocketPipeline` or `PipelineSimulator`) and the bulk pipeline (H2D/D2H sockets) are completely independent data paths. The token pipeline signals completion via `OutputMessage::is_complete`, while the bulk pipeline delivers action output via D2H pages. The host's main loop pops a completion message from the `DecodeScheduler`, then reads the D2H bulk channel:

```cpp
// examples/pi05_pipeline_runner.cpp:1068-1071
uint32_t result_slot_id;
double d2h_us = bulk_d2h->recv(result_slot_id,
                               action_output.data(),
                               static_cast<uint32_t>(action_output.size()));
```

This assumes a 1:1 correspondence between token-level completions and bulk D2H responses, and that they arrive in the same order. The ordering is only guaranteed if:

1. The device processes users in FIFO order (true for the passthrough kernel's H2D receive loop).
2. The token path completion order matches the H2D submission order.

In the current passthrough design, the device kernel processes H2D frames strictly in FIFO order and produces D2H outputs in the same order. But if the token pipeline (in simulation mode) reorders completions depending on the scheduler's batching decisions---e.g., if user B's decode finishes before user A's due to fewer output tokens---the host will read A's D2H output while processing B's completion, triggering the mismatch warning.

### Consequences

After logging the mismatch warning, the host copies the (wrong-user) action output into `action_output` and uses it as `action_history` for the next turn of the *expected* user. This creates a data integrity violation that propagates through subsequent turns.

For the passthrough kernel, the action output is synthetic, so the mismatch is benign. In a production system, a protocol fix would require either:

- A D2H queue per slot (eliminates ordering dependency).
- Strict ordering guarantees between the two paths.
- A correlation mechanism (e.g., sequence numbers).

**Verdict:** Safe for the passthrough kernel; a latent correctness bug for any production kernel that may reorder or multiplex responses.

---

## 4.4.4 D2H FIFO Sizing and Deadlock

**Hazard:** If multiple users' D2H responses back up in the FIFO, the device kernel can deadlock.

### The Constraint

The D2H FIFO size is configured by `SocketConfigParams::fifo_size` (default 524,288 bytes = 512 KB):

```cpp
// examples/pi05_pipeline_runner.cpp:226
uint32_t fifo_size = 524288;  // 512 KB
```

Each D2H response is one page (4096 bytes). The maximum number of outstanding D2H responses before the FIFO fills is:

$$
\text{max\_buffered\_outputs} = \left\lfloor \frac{\text{fifo\_size}}{\text{BULK\_PAGE\_SIZE}} \right\rfloor = \left\lfloor \frac{524{,}288}{4096} \right\rfloor = 128 \text{ pages}
$$

### Deadlock Scenario

The kernel's main loop is single-threaded:

```
while (true) {
    wait for H2D pages
    pop H2D pages
    reserve D2H page     <-- blocks if D2H FIFO is full
    write D2H response
    push D2H page
}
```

If the host is slow to read D2H responses (e.g., because it is blocked sending the next H2D frame for a different user), the D2H FIFO fills up. The kernel blocks at `socket_reserve_pages(sender, 1)` (line 167). Meanwhile, the host may be blocked trying to send H2D data, which requires the kernel to pop H2D pages to free FIFO space. This creates a circular dependency:

- **Kernel** is blocked waiting for D2H FIFO space (needs host to read D2H).
- **Host** is blocked waiting for H2D FIFO space (needs kernel to pop H2D).

**Deadlock condition:** This deadlock is possible when the D2H FIFO capacity is exhausted:

$$
\text{num\_users} \times \text{BULK\_PAGE\_SIZE} > \text{D2H FIFO size}
$$

For the default FIFO size of 512 KB, this means `num_users > 128`. But the actual threshold depends on the host's ability to interleave H2D sends and D2H reads, which in the current single-threaded main loop is sequential per user.

The minimum safe D2H FIFO size is:

$$
\text{min\_d2h\_fifo} = \text{BULK\_PAGE\_SIZE} \times \text{num\_users}
$$

With `num_users = 8` and `BULK_PAGE_SIZE = 4096$:

$$
\text{min\_d2h\_fifo} = 4{,}096 \times 8 = 32{,}768 \text{ bytes}
$$

The default FIFO size of 512 KB can buffer 128 outputs, which far exceeds the typical `num_users` range (1-32). But a configuration with `num_users > 128` would require either a larger FIFO or guaranteed interleaved reads.

### Status in Current Code

In the current passthrough design, this deadlock cannot occur because:

1. The token path (`PipelineSimulator` or `SocketPipeline`) operates independently of the bulk kernel.
2. The host reads D2H output only after receiving `is_complete` from the token path, which comes from a different processing pipeline.
3. The host processes completions in arrival order, reading D2H immediately after each token-level completion, preventing D2H backlog in the common case.

The config validation only checks the H2D FIFO size against the padded payload (line 848), not the D2H FIFO capacity against `num_users`. This is a gap in validation.

### H2D FIFO Serialization

The H2D FIFO has a different constraint. Since H2D payloads are ~458 KB each and the FIFO is 512 KB, the FIFO can hold only one full payload at a time. The host must wait for the device to consume the previous payload before sending the next one. This is handled transparently by the `H2DSocket::write()` flow control (the host blocks on `reserve_bytes()` if the FIFO is full).

This means H2D transfers are strictly serialized---there is no pipelining of multiple users' pixel data through the H2D path. Each user's frame must be fully consumed before the next user's frame can be sent. With 8 users and a 200us H2D transfer time, this adds ~1.6ms of serialization overhead to the batch.

---

## 4.4.5 Echo Hang on Device Crash

**Hazard:** If the device kernel crashes after receiving the sentinel but before echoing it, `recv_sentinel_echo()` blocks forever.

The sentinel echo path has no timeout and no diagnostic output:

```cpp
// examples/pi05_pipeline_runner.cpp:612-614
while (!socket_->has_data()) {
    // Spin-wait for device kernel to echo sentinel
}
```

In contrast, the regular `recv()` method emits warnings every 5 seconds. The echo path is assumed to be quick (the device echoes immediately upon sentinel receipt), but if the device kernel crashes, the host process hangs with no indication of why.

The `_exit(EXIT_SUCCESS)` at line 1196 is the only backstop: if the host process is killed externally, it exits cleanly. But there is no self-recovery mechanism.

**Verdict:** Rare in the passthrough kernel (the kernel reliably exits its loop), but a production concern if the device kernel has more complex shutdown logic.

---

## 4.4.6 Missing Error Propagation

Several error conditions in the bulk path are logged but not propagated:

| Error Condition | Location | Current Handling | Risk |
|----------------|----------|------------------|------|
| D2H slot mismatch | Line 1075 | `WARNING` to stderr | Silent data corruption |
| Sentinel echo failure | Line 1138 | `WARNING` to stderr | Incomplete shutdown |
| D2H spin-wait > 5s | Line 584 | `WARNING` to stderr | Infinite hang |
| `recv_sentinel_echo()` no timeout | Line 612 | Spin forever | Process hang |
| H2D payload > FIFO | Line 848 | `throw` (caught at main) | Clean error (handled well) |

None of the runtime conditions set an error flag, throw an exception, or update the `DecodeScheduler`'s error state. The host continues operating under the assumption that everything is fine.

**Contrast with simulation mode:** In simulation mode, the `DecodeScheduler` propagates errors through `SchedulerResponse::error_code`, and the host checks this value after every `ALLOCATE` (line 951) and `SUBMIT` operation. The bulk channel path has no equivalent error propagation mechanism.

### Specific Risks

1. **D2H slot mismatch with continue:** After logging the mismatch warning, the host copies the (wrong-user) action output into `action_output` and uses it as `action_history` for the next turn of the *expected* user. This creates a data integrity violation that propagates through subsequent turns.

2. **No error on H2D write failure:** `BulkH2DChannel::send()` does not check the return value of `socket_->write()` (the tt-metal API may not return errors; PCIe faults would likely manifest as IOMMU exceptions caught at a different layer). If a write silently fails, the kernel will hang waiting for pages that never arrive.

3. **Process-level escape hatch:** The code at line 1196 uses `_exit(EXIT_SUCCESS)` to bypass atexit handlers:

   ```cpp
   // examples/pi05_pipeline_runner.cpp:1194-1196
   // Use _exit() to skip atexit handlers -- tt-metal's ShmResourceTracker
   // cleanup races with socket teardown causing double-free.
   _exit(EXIT_SUCCESS);
   ```

   This is a workaround for a resource cleanup race in the tt-metal library's `ShmResourceTracker`. The tracker registers shared-memory segments for cleanup via `atexit()`. When the socket destructors (called explicitly in step 5) have already cleaned up the same resources, the `atexit` handler attempts a double-free. Using `_exit()` avoids this, but it also means the process always reports success regardless of whether earlier steps failed.

### Production Recommendations

A production system should:
1. Treat `slot_id` mismatch as a fatal error (or at minimum, discard the mismatched output and re-request).
2. Add a timeout to `recv_sentinel_echo()` and treat timeout as a device crash.
3. Expose D2H spin-wait duration as a metric for health monitoring.
4. Propagate errors from `H2DSocket::write()` and `D2HSocket::read()` (currently these are assumed infallible).

---

## 4.4.7 Summary of Hazards

| Hazard                    | Severity  | Status in Passthrough    | Production Risk            |
|---------------------------|-----------|--------------------------|----------------------------|
| Header/payload atomicity  | High      | Safe (single write)      | Safe if maintained         |
| Pop-before-D2H ordering   | Critical  | Safe (no data dependency)| Data corruption            |
| D2H slot ID mismatch      | Medium    | Benign (synthetic output)| Wrong action output        |
| D2H FIFO deadlock         | High      | Impossible (separate paths)| Possible with unified kernel|
| H2D serialization         | Low       | Functional (slow)        | Throughput bottleneck      |
| Echo hang on crash        | High      | Rare (kernel reliable)   | Unrecoverable hang         |
| Soft error propagation    | Medium    | Acceptable for dev       | Silent corruption          |

The passthrough kernel is deliberately simple and avoids most of these hazards by not reading pixel data and by operating independently of the token pipeline. A production kernel that integrates vision encoding, denoise inference, and action output within a single device program will need to address each of these hazards explicitly.

---

**Previous:** [Shutdown Protocol](03_shutdown_protocol.md)

---

**Next:** [Chapter 5 -- Device Launcher and Kernel Architecture](../ch5_device_launcher_and_kernel_architecture/index.md)
