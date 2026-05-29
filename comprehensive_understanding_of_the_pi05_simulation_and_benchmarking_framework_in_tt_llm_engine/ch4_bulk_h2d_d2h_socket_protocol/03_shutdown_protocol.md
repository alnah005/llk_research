# 4.3 Shutdown Protocol

> **Source files:**
> - `examples/pi05_pipeline_runner.cpp`, lines 1131--1150 (host-side shutdown)
> - `kernels/pi05_bulk_passthrough.cpp`, lines 92--125 (device-side sentinel handling)

The bulk socket channels require an explicit, ordered shutdown handshake between the host and device kernel. Unlike the token-level `DecodeScheduler`, which has its own `stop()` method that drains internal queues, the bulk channels operate at the raw DMA level and have no built-in teardown mechanism. The shutdown protocol is hand-coded in the main loop of `pi05_pipeline_runner.cpp`.

---

## 4.3.1 The Ordered Shutdown Sequence

The host executes five steps in strict order (the source code uses a 4-step comment numbering where steps 1 and 2 below are combined as "step 1"; the guide splits them to highlight the barrier ordering dependency):

```
Host                                    Device Kernel
 |                                         |
 |  1. send_sentinel()  -----------------> |  (receives sentinel H2D page)
 |  2. bulk_h2d->barrier()                 |  (echoes sentinel via D2H)
 |  3. recv_sentinel_echo() <------------- |  (exits main loop)
 |  4. mgr->stop()                         |
 |  5. bulk_h2d.reset(); bulk_d2h.reset()  |
 |                                         |
```

### Step 1: Send Sentinel

```cpp
// examples/pi05_pipeline_runner.cpp:1133
bulk_h2d->send_sentinel();
```

The `send_sentinel()` method writes a single page containing the sentinel header (`slot_id = 0xFFFFFFFF`, `payload_length = 0`):

```cpp
// examples/pi05_pipeline_runner.cpp:546-552
void send_sentinel() {
    std::vector<uint8_t> page(BULK_PAGE_SIZE, 0);
    BulkH2DHeader sentinel{SENTINEL_SLOT_ID, 0};
    std::memcpy(page.data(), &sentinel, sizeof(sentinel));
    socket_->write(page.data(), 1);
}
```

Note that the sentinel is exactly one page, regardless of the normal H2D payload size. The device kernel detects it by checking `slot_id` before computing `total_pages`:

```cpp
// kernels/pi05_bulk_passthrough.cpp:88-92
uint32_t slot_id        = first_page[0];
uint32_t payload_length = first_page[1];

// Sentinel check
if (slot_id == SENTINEL_SLOT_ID) {
```

### Step 2: H2D Barrier

```cpp
// examples/pi05_pipeline_runner.cpp:1134
bulk_h2d->barrier();
```

The barrier blocks until the device kernel has consumed (popped) the sentinel page from the H2D FIFO. The underlying `H2DSocket::barrier()` spins on a `bytes_acked` counter in pinned host memory that the device updates after popping pages. This ensures the sentinel has been received before the host proceeds to wait for the echo. Without the barrier, the host could race ahead and call `recv_sentinel_echo()` before the device has even read the sentinel.

### Step 3: Wait for D2H Echo

```cpp
// examples/pi05_pipeline_runner.cpp:1137-1141
bool echo_ok = bulk_d2h->recv_sentinel_echo();
if (!echo_ok) {
    std::cerr << "WARNING: Expected sentinel echo from bulk kernel" << std::endl;
} else {
    std::cout << "Bulk kernel sentinel echo confirmed." << std::endl;
}
```

The device kernel, upon detecting the sentinel, constructs a D2H echo page:

```cpp
// kernels/pi05_bulk_passthrough.cpp:96-118
// Echo the sentinel back via D2H so the host can confirm
// the kernel received the shutdown command.
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
```

After writing the echo, the device kernel pops the sentinel from the H2D FIFO and breaks out of the main loop:

```cpp
// kernels/pi05_bulk_passthrough.cpp:120-125
// Pop the sentinel H2D page AFTER D2H echo is flushed
socket_pop_pages(receiver, 1);
noc_async_writes_flushed();
socket_notify_sender(receiver);

break;
```

The ordering here is important: the H2D pop happens after the D2H echo write and its barrier. This ensures the echo data is committed to PCIe before the H2D FIFO slot is released, preventing a race where the host could reuse the H2D slot before the echo reaches host memory.

### Step 4: Stop DecodeScheduler

```cpp
// examples/pi05_pipeline_runner.cpp:1145
mgr->stop();
```

The `DecodeScheduler` is stopped after the bulk channels are quiesced. This ordering prevents the scheduler from attempting to push new work into a pipeline whose device-side kernel has already exited.

### Step 5: Destroy Channels

```cpp
// examples/pi05_pipeline_runner.cpp:1148-1150
bulk_h2d.reset();
bulk_d2h.reset();
```

The channels are explicitly destroyed, which triggers `H2DSocket` and `D2HSocket` destructors. These deregister the shared-memory descriptors and release pinned memory. Destroying channels before the device process is terminated could cause the device to write into unmapped host memory, but in practice the device kernel has already exited at step 3.

---

## 4.3.2 Why Ordering Matters

The five steps form a dependency chain:

| Step | Depends on | Reason |
|------|------------|--------|
| 2 (barrier) | 1 (sentinel) | Cannot barrier on a write that hasn't happened |
| 3 (echo) | 2 (barrier) | Echo can only arrive after device has consumed sentinel |
| 4 (stop scheduler) | 3 (echo) | Scheduler must not push work after kernel exits |
| 5 (destroy channels) | 4 (stop) | Socket destructors must not race with scheduler I/O |

Violating any of these orderings can lead to:

| Misordering                 | Failure Mode                                                   |
|-----------------------------|----------------------------------------------------------------|
| Skip sentinel, go to stop   | Device kernel loops forever; `_exit()` required to kill it.    |
| Echo before barrier         | Echo spin could see stale data if sentinel not yet consumed.   |
| Stop before echo            | Token pipeline shutdown may close sockets before echo arrives. |
| Destroy before echo         | `recv_sentinel_echo()` reads from destroyed socket -- UB.      |
| Destroy before stop         | DecodeScheduler may access destroyed socket references.        |

### The `_exit()` Workaround

Even with correct ordering of the explicit shutdown steps, the C++ runtime's `atexit` handlers (registered by tt-metal's `ShmResourceTracker`) can race with the socket destructors that have already run in step 5. The code works around this at line 1196:

```cpp
// examples/pi05_pipeline_runner.cpp:1194-1196
// Use _exit() to skip atexit handlers -- tt-metal's ShmResourceTracker
// cleanup races with socket teardown causing double-free.
_exit(EXIT_SUCCESS);
```

`_exit()` bypasses all `atexit` handlers, avoiding the double-free. This means that even if earlier steps failed, the process reports success. Any monitoring system checking exit codes will not detect the failure. It also means the process does not run C++ static object destructors, so any state held in global objects (e.g., logging flush) is abandoned.

---

## 4.3.3 Sentinel Robustness Concerns

The current sentinel protocol has several gaps:

### No Retry on Sentinel Send Failure

If the H2D `write()` for the sentinel fails (e.g., due to a transient PCIe error or IOMMU remapping), the host has no fallback. The sentinel is sent exactly once. A production system would want at least a retry loop with exponential backoff.

### No Timeout on Echo Wait

`recv_sentinel_echo()` contains an unbounded spin-wait:

```cpp
// examples/pi05_pipeline_runner.cpp:612-614
while (!socket_->has_data()) {
    // Spin-wait for device kernel to echo sentinel
}
```

If the device kernel crashes, hangs, or never receives the sentinel, this spin-wait runs forever with zero diagnostics. Compare this with the regular `recv()` method, which at least prints a warning after 5 seconds. The sentinel echo path has no health monitoring at all.

A robust implementation would:
1. Add a timeout (e.g., the same `connect_timeout_ms` used during socket setup).
2. Check `device_launcher.is_alive()` periodically to detect kernel crashes.
3. Return a failure status rather than hanging.

### No Validation of Echo Content

`recv_sentinel_echo()` only checks `word0 == SENTINEL_SLOT_ID`. It does not verify:
- That `payload_length` (word1) is 0.
- That the remaining page bytes are zero.
- That this is the "right" echo (vs. a stale sentinel from a previous run, if the D2H FIFO had leftover data).

In practice, the D2H FIFO should be empty at shutdown (all inference turns have been read), so stale data is unlikely. But the protocol provides no formal guarantee.

### Non-Fatal Echo Failure

If the echo arrives but contains an unexpected value (i.e., not `SENTINEL_SLOT_ID`), the host logs a warning but proceeds with shutdown:

```cpp
// examples/pi05_pipeline_runner.cpp:1138-1139
if (!echo_ok) {
    std::cerr << "WARNING: Expected sentinel echo from bulk kernel" << std::endl;
```

This means the host may destroy channels and stop the scheduler even if the device kernel is not actually shut down. The `_exit()` call at line 1196 terminates the process without running atexit handlers, which mitigates some cleanup issues but does not prevent device-side corruption.

### Single-Sentinel Protocol

Only one sentinel is sent, regardless of the number of users. The protocol assumes a single bulk kernel handles all users sequentially, which is correct for the current passthrough kernel but may not generalize to multi-kernel architectures.

---

## 4.3.4 Simulation Mode Teardown Contrast

In simulation mode (no sockets), the teardown is entirely different:

```cpp
// examples/pi05_pipeline_runner.cpp:1362-1396
for (auto& u : users) {
    pm::ISRequest req{};
    req.type = pm::RequestType::EVICT;
    req.request_id = req_id++;
    req.slot_id = u.slot_id;
    push_request(mgr, req);
}
auto teardown_deadline = Clock::now() + std::chrono::seconds(timeout_s);
bool teardown_clean = false;
while (Clock::now() < teardown_deadline) {
    pm::OutputMessage out;
    while (mgr.try_pop_output(out)) {}
    bool all_inactive = true;
    for (auto& u : users) {
        if (mgr.get_user_state(u.slot_id) != pm::UserState::INACTIVE) {
            all_inactive = false;
            break;
        }
    }
    if (all_inactive) {
        teardown_clean = true;
        break;
    }
    std::this_thread::yield();
}
```

Key differences:

| Aspect            | Socket Mode                     | Simulation Mode                           |
|--------------------|---------------------------------|-------------------------------------------|
| Signal mechanism   | Sentinel via H2D socket          | EVICT request via DecodeScheduler          |
| Confirmation       | D2H echo page                   | `UserState::INACTIVE` poll                 |
| Timeout            | None (hangs on echo)            | Yes (`timeout_s` deadline)                 |
| Cleanup order      | Sentinel -> barrier -> echo -> stop -> destroy | EVICT -> poll -> stop (normal return) |
| Process exit       | `_exit()` (skip atexit)         | Normal `return EXIT_SUCCESS`               |
| Error handling     | WARNING on echo mismatch        | WARNING on timeout + count of stuck users  |

The simulation mode is notably more robust: it has a bounded timeout, explicit per-user state checks, and a clean exit path. The socket mode's reliance on an unbounded echo wait is a gap that should be addressed before production use.

Note also that socket mode does not verify that all user slots reach `INACTIVE` before destroying the scheduler---the sentinel handles shutdown at the DMA level, and `DecodeScheduler::stop()` handles scheduler-internal cleanup. This means socket mode could leave resource leaks in the scheduler's internal data structures if users were still active when the sentinel was sent.

---

**Previous:** [Bulk Channel Classes](02_bulk_channel_classes.md)

**Next:** [Race Conditions and Concurrency Hazards](04_race_conditions.md)
