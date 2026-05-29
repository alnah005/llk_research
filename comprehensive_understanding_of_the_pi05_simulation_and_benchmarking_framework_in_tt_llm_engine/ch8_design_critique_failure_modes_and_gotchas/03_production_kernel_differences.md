# 8.3 -- Production Kernel Differences

The `pi05_bulk_passthrough.cpp` kernel is a *testing scaffold*, not a production kernel. It implements the H2D/D2H socket protocol correctly but skips all compute. This section documents the four critical behavioral differences between the passthrough and a production kernel, so that anyone writing the real kernel or interpreting benchmark results knows exactly where the passthrough's shortcuts are.

---

## 8.3.1 H2D Pop Timing: Passthrough Pops Immediately

### Passthrough Behavior

The passthrough kernel pops H2D pages **immediately** after reading the header, before constructing the D2H response:

```cpp
// pi05_bulk_passthrough.cpp:143-147
// Read complete, pop the first page
socket_pop_pages(receiver, 1);
noc_async_writes_flushed();
socket_notify_sender(receiver);
```

Remaining pages are popped one-by-one in the loop:

```cpp
// pi05_bulk_passthrough.cpp:149-163
for (uint32_t p = 1; p < total_pages; p++) {
    socket_wait_for_pages(receiver, 1);
    // DEVICE_PULL: must pull page even though passthrough discards it.
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

The kernel source itself documents why this is safe for passthrough but not for production:

```cpp
// pi05_bulk_passthrough.cpp:132-141
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

### Production Requirement

A production kernel must:

1. **Wait** for all H2D pages to arrive (`socket_wait_for_pages(receiver, total_pages)`)
2. **Copy** pixel data from the FIFO into a compute buffer (L1 or DRAM)
3. **Process** the data through the vision encoder, text backbone, and denoise loop
4. **Write** the D2H result
5. **Barrier** (`noc_async_write_barrier()`) to ensure D2H data is committed
6. **Only then** pop all H2D pages (`socket_pop_pages(receiver, total_pages)`)

The reason: popping advances the FIFO read pointer, which signals the host that the FIFO space is available for reuse. If the host immediately writes the next user's pixel data into the same FIFO region, and the kernel has not yet finished reading the previous user's data, the pixel data is corrupted mid-read.

In the passthrough kernel, this race cannot occur because the passthrough discards pixel data entirely -- it reads the header (slot_id + payload_length) and ignores the rest.

### Impact on Benchmarks

The passthrough pop-immediately pattern makes H2D throughput appear **better** than production because:

- FIFO pages are freed immediately, so the host never blocks on FIFO full
- There is no FIFO pressure from retained pages during compute
- The host can begin sending the next user's data while the kernel is still "processing" (echoing) the current one

The deferred-pop approach also means the H2D FIFO must hold a complete transfer for the entire duration of processing (vision + text + denoise). For multi-user scenarios, the FIFO would need to hold multiple complete transfers simultaneously, or the host must implement flow control to avoid sending the next transfer until the kernel signals readiness.

---

## 8.3.2 Signal-Wait for Denoise: Passthrough Writes D2H Immediately

### Passthrough Behavior

After consuming H2D pages, the passthrough kernel immediately constructs and sends the D2H response with no compute step:

```cpp
// pi05_bulk_passthrough.cpp:167-202
// ---- Construct and send D2H action output ----
socket_reserve_pages(sender, 1);

// Zero the output page
for (uint32_t w = 0; w < page_size_words; w++) {
    output_cb_addr[w] = 0;
}

// Write D2H header
output_cb_addr[0] = slot_id;
output_cb_addr[1] = ACTION_OUTPUT_BYTES;

// Write synthetic action output (passthrough: incrementing pattern)
uint32_t action_start_word = 2;
uint32_t action_words = ACTION_OUTPUT_BYTES / sizeof(uint32_t);
for (uint32_t w = 0; w < action_words; w++) {
    output_cb_addr[action_start_word + w] = w + slot_id;
}

// NOC write to D2H socket
noc_wwrite_with_state<...>(...);
noc_async_write_barrier();
socket_push_pages(sender, 1);
socket_notify_receiver(sender);
```

Total time from H2D receipt to D2H response: microseconds (just memset + memcpy + one NOC write).

### Production Requirement

In the real Pi0.5 pipeline, the D2H action output is produced by the **denoise phase**, which runs 5 iterations of a 6-stage flow-matching loop (30 total stages, each ~757 us = ~22.7 ms total). The bulk kernel must **wait** for the denoise phase to complete before writing the D2H response.

The production kernel needs a **signal-wait mechanism** between the compute pipeline and the bulk I/O kernel:

1. The bulk kernel receives H2D pixel data and forwards it to the vision encoder cores
2. The compute pipeline processes vision -> text -> denoise (5 iterations)
3. The denoise output is written to a shared L1 buffer
4. A **semaphore** or **NOC signal** notifies the bulk kernel that the output is ready
5. The bulk kernel reads the output from the shared buffer and writes it to D2H

Possible signal-wait mechanisms:

- A **semaphore** in L1 memory: the last denoise stage writes a value to a known L1 address, and the bulk kernel polls that address.
- A **circular buffer signal**: the denoise output is written to a circular buffer that the bulk kernel monitors via `cb_wait_front()`.
- A **socket-based notification**: an internal device-to-device socket carries a completion signal.

The passthrough kernel has no signal-wait because there is no compute pipeline to wait for.

### Impact on Benchmarks

The passthrough kernel's D2H latency is:

$$t_{\text{D2H, passthrough}} \approx \frac{4096 \text{ bytes (1 page)}}{16 \text{ GB/s}} \approx 0.25 \text{ us}$$

The production kernel's D2H latency includes the full compute pipeline:

$$t_{\text{D2H, production}} \approx 22{,}710 \text{ us (denoise)} + 0.25 \text{ us (transfer)} \approx 22.7 \text{ ms}$$

The framework compensates by using the token pipeline's completion message (`is_complete`) as the timing reference, and the D2H read occurs *after* that completion (line 1068-1073 of the runner). This means the measured D2H transfer time is accurate (it is just the PCIe transfer), but the end-to-end latency from H2D send to D2H receive would be much longer in production.

---

## 8.3.3 Synthetic vs. Real Action Output

### Passthrough Behavior

The passthrough kernel generates a deterministic incrementing pattern as "action output":

```cpp
// pi05_bulk_passthrough.cpp:179-183
uint32_t action_start_word = 2;  // after 8-byte header
uint32_t action_words = ACTION_OUTPUT_BYTES / sizeof(uint32_t);
for (uint32_t w = 0; w < action_words; w++) {
    output_cb_addr[action_start_word + w] = w + slot_id;
}
```

This produces 3200 bytes (800 uint32_t words) where word $w$ of slot $s$ has value $w + s$. The pattern is deterministic and slot-dependent, which allows the host to verify data integrity by checking the pattern, though the current framework does not perform this validation.

### Production Requirement

The real action output is a 50x32 bfloat16 tensor produced by the denoise phase. This tensor encodes robot joint positions, velocities, or end-effector commands. Its values are:

- **Non-deterministic** across runs (due to floating-point non-associativity in parallel reductions).
- **Bounded** but not in the `[0, 255]` range -- bfloat16 can represent values up to ~3.4e38.
- **Sensitive to input** -- small pixel variations can produce significantly different actions, especially in low-confidence regions of the policy network.

The host-side `action_output` buffer (3200 bytes) and `action_history` buffer (4096 bytes, configured via `pixel_payload.action_history_bytes`) are sized for this tensor. The action history bytes (4096) is larger than the action output (3200) to accommodate additional metadata that a production system might append (timestamps, confidence scores, policy version). In production, a downstream component would deserialize the bfloat16 tensor and feed it to the robot control loop.

### Impact on Benchmarks

The synthetic output is useful for validating the H2D-to-D2H data path (correct slot_id routing, correct byte count, no corruption) but says nothing about the **numerical quality** of the inference output. The benchmark's throughput and latency numbers are valid for the I/O and scheduling layers but do not validate the compute pipeline.

---

## 8.3.4 Core Assignment: One Core Passthrough vs. Multi-Core Production

### Passthrough Behavior

The passthrough kernel runs on a **single Tensix core**:

```cpp
// pi05_device_launcher.cpp:133
const CoreCoord bulk_core(bulk_only ? CoreCoord(0, 0) : CoreCoord(1, 0));
```

In full mode, core (0,0) runs the token loopback kernel and core (1,0) runs the bulk passthrough. In bulk-only mode, both functions collapse onto core (0,0).

The entire kernel -- H2D receive, H2D pop, D2H construct, D2H write -- executes on RISCV_0 of a single core:

```cpp
// pi05_device_launcher.cpp:198-199
.processor = DataMovementProcessor::RISCV_0,
.noc = NOC::RISCV_0_default,
```

The kernel is a simple sequential loop (`while (true) { ... }`) with no parallelism.

### Production Requirement

The production Pi0.5 kernel distributes work across **multiple Tensix cores**:

| Component | Core Assignment | Notes |
|-----------|----------------|-------|
| H2D bulk I/O | 1 core (e.g., (1,0)) | Receives pixel data, forwards to vision cores |
| Vision encoder (SigLIP) | Multiple cores | Matrix multiply, attention, layer norm; data-parallel across a grid |
| Text backbone (Gemma) | Multiple cores | Sequence-parallel for attention, tensor-parallel for feedforward |
| Denoise (flow matching) | Multiple cores | 5 iterations, each using 6 pipeline stages on a smaller core grid |
| D2H bulk I/O | 1 core (e.g., (1,0)) | Reads denoise output, sends to host |

The H2D and D2H I/O may share a core (as the passthrough does), or they may be split across two cores for full-duplex operation. The production architecture requires:

1. **NOC multicast** from the I/O core to the compute grid for distributing pixel data
2. **Circular buffer (CB) allocation** in L1 for intermediate tensors between compute phases
3. **Cross-core synchronization** via L1 semaphores for pipeline handoffs
4. **Multi-core scheduling** to overlap compute with I/O

### Impact on Benchmarks

The passthrough kernel's single-core design means:

- **No contention** between I/O and compute for L1 bandwidth, NOC bandwidth, or compute resources
- **No pipeline fill/drain overhead** -- the passthrough responds in microseconds, whereas the production pipeline takes milliseconds to fill
- **No memory pressure** -- the passthrough kernel uses only the output CB and the H2D FIFO, whereas the production kernel must allocate L1 for activations, weights, and intermediate buffers across many cores. On a Tensix core with 1.5 MB of L1, a full pipeline may partition L1 across many cores with careful budget allocation
- **No compute unit utilization** -- FPU, SFPU, and matrix engine are idle in the passthrough

The benchmark measures the **I/O and scheduling envelope** accurately but does not capture the compute-induced latency, contention, or memory pressure of the production system.

---

**Previous:** [Failure Modes](02_failure_modes.md)

**Next:** [Operational Gotchas](04_operational_gotchas.md)
