# 4.3 -- Host-Device Flow Control

Flow control prevents a fast producer from overwriting data that a slow consumer has not yet read. H2D and D2H sockets both use the same conceptual model -- a circular FIFO tracked by `bytes_sent` and `bytes_acked` counters -- but the physical placement of those counters, the transport used to update them, and the polling mechanisms differ across the three transfer paths (HOST_PUSH, DEVICE_PULL, D2H). This section provides the precise counter placement map that resolves these differences, walks through the flow control logic for each path, and covers production techniques including batched acknowledgement and error recovery.

---

## 4.3.1 Shared FIFO Model

Every host-device socket wraps a circular byte buffer divided into fixed-size pages. Two monotonically increasing counters track progress:

```
  bytes_acked                              bytes_sent
      |                                       |
      v                                       v
 +----+--------+--------+--------+-----------+----+
 |free| page A | page B | page C | free ...  |    |
 +----+--------+--------+--------+-----------+----+
      <-- consumed but    <-- written but
          not yet acked       not yet consumed -->
```

- **`bytes_sent`**: Total bytes the producer has written into the FIFO. Incremented by the producer after each write.
- **`bytes_acked`**: Total bytes the consumer has finished reading. Incremented by the consumer after each read.

### The invariant

```
0 <= bytes_acked <= bytes_sent <= bytes_acked + fifo_size
```

- `bytes_sent - bytes_acked` is the number of outstanding (unacknowledged) bytes in the FIFO.
- The producer must not write when `bytes_sent - bytes_acked == fifo_size` (FIFO full).
- The consumer must not read when `bytes_sent == bytes_acked` (FIFO empty).

Both sides enforce this invariant by spinning on the relevant counter before proceeding. The counters are monotonically increasing 64-bit values; wrapping is handled by the FIFO size being a power-of-two or by modular arithmetic on the buffer offset.

---

## 4.3.2 H2D Flow Control

### HOST_PUSH Path

```
  HOST (producer)                          DEVICE (consumer)
 +-------------------+                    +-------------------+
 | 1. Check space:   |                    |                   |
 |    read bytes_acked|<-- local read ----|                   |
 |    from pinned mem |   (mirror copy)    |                   |
 | 2. Write page data |--- TLB write ---->| 4. Read bytes_sent|
 |    to device L1    |                    |    from L1 (local)|
 | 3. Write bytes_sent|--- TLB write ---->| 5. Process page   |
 |    to device L1    |                    |                   |
 |                    |<-- NOC write ------| 6. Write          |
 |    polls pinned    |    through PCIe    |    bytes_acked    |
 |    memory for ack  |    to host pinned  |    to host pinned |
 +-------------------+                    +-------------------+
```

The host checks space by reading `bytes_acked` from the host-pinned mirror (local, no PCIe). The device checks for data by reading `bytes_sent` from L1 (local). The device first updates `bytes_acked` in device L1 (canonical location, same as DEVICE_PULL), then mirrors the value to host-pinned memory via a NOC write through PCIe (step 6). This mirror write is the only cross-bus counter update in the steady-state path.

### DEVICE_PULL Path

```
  HOST (producer)                          DEVICE (consumer)
 +-------------------+                    +-------------------+
 | 1. Check space:   |--- TLB read ------>| (bytes_acked in   |
 |    read bytes_acked|   (PCIe RT)       |  device L1)       |
 | 2. Write data to  |                    |                   |
 |    host pinned mem |                    |                   |
 | 3. Write bytes_sent|--- TLB write ---->| 4. Read bytes_sent|
 |    to device L1    |                    |    from L1 (local)|
 |                    |<-- NOC read -------| 5. Pull page data |
 |                    |    through PCIe    |    from host mem  |
 |                    |                    | 6. Update         |
 |                    |                    |    bytes_acked    |
 |                    |                    |    in device L1   |
 +-------------------+                    +-------------------+
```

In DEVICE_PULL, both counters live in device L1. The device reads both locally. The host must perform a PCIe TLB read to check `bytes_acked` (step 1), incurring a round trip -- acceptable because the host's check frequency is low relative to the device's consumption rate in bulk transfer scenarios.

---

## 4.3.3 D2H Flow Control

```
  DEVICE (producer)                        HOST (consumer)
 +-------------------+                    +-------------------+
 | 1. Check space:   |                    |                   |
 |    read bytes_acked|                    |                   |
 |    from L1 (cache  |                    |                   |
 |    invalidation)   |                    |                   |
 | 2. Write page data |--- NOC write ---->| 4. Read bytes_sent|
 |    to host pinned  |    through PCIe   |    from pinned mem|
 | 3. Write bytes_sent|--- NOC write ---->|    (local read)   |
 |    to host pinned  |    through PCIe   | 5. Read page data |
 |                    |<-- TLB write ------| 6. Write          |
 |    reads updated   |    to device L1   |    bytes_acked    |
 |    bytes_acked     |                    |    to device L1   |
 +-------------------+                    +-------------------+
```

The host checks for data by reading `bytes_sent` from pinned memory (local, fast). The device checks for free space by reading `bytes_acked` from L1, using cache invalidation to see the value written by the host via TLB write (step 6).

---

## 4.3.4 Counter Location Map

This table resolves the precise physical location and transport mechanism for every counter in every mode. It is the single source of truth for counter placement.

| Mode | Counter | Writer | Written To | Via | Reader | Read From | Via |
|------|---------|--------|-----------|-----|--------|-----------|-----|
| H2D HOST_PUSH | `bytes_sent` | Host | Device L1 | TLB write (PCIe) | Device | Device L1 | Local L1 read |
| H2D HOST_PUSH | `bytes_acked` | Device | Device L1 (canonical) + host pinned memory (mirror) | Local L1 write + NOC write through PCIe (mirror) | Host | Host pinned memory (mirror) | Local memory read |
| H2D DEVICE_PULL | `bytes_sent` | Host | Device L1 | TLB write (PCIe) | Device | Device L1 | Local L1 read |
| H2D DEVICE_PULL | `bytes_acked` | Device | Device L1 | Local L1 write | Host | Device L1 | TLB read (PCIe round trip) |
| D2H | `bytes_sent` | Device | Host pinned memory | NOC write through PCIe | Host | Host pinned memory | Local memory read |
| D2H | `bytes_acked` | Host | Device L1 | TLB write (PCIe) | Device | Device L1 | L1 read (cache invalidation) |

**Note on H2D HOST_PUSH bytes_acked:** Both H2D modes canonically keep both counters in device L1. HOST_PUSH adds a mirror of `bytes_acked` to host-pinned memory so the host can poll locally without a PCIe round trip. The "Written To" column above reflects both locations for this counter.

### Counter placement principle

The design places each counter where the **frequent reader** lives:

- A counter that the device reads frequently is placed in device L1, so the device can poll it with a local read (nanosecond latency, zero PCIe traffic).
- A counter that the host reads frequently is placed in host-pinned memory, so the host can poll it with a local DRAM read (nanosecond latency, zero PCIe traffic).
- The infrequent writer pays the cost of a cross-bus write to update the remote copy.

Applying this principle to each row: H2D `bytes_sent` lives in device L1 because the device spins on it in `socket_wait_for_pages`; HOST_PUSH `bytes_acked` canonically lives in device L1 but is mirrored to host-pinned memory because the host polls it before each `write()`; DEVICE_PULL `bytes_acked` stays in device L1 because the device also reads it for FIFO space checks and the host's infrequent TLB read is acceptable; D2H `bytes_sent` lives in host-pinned memory because the host polls it in `has_data()`; D2H `bytes_acked` lives in device L1 because the device spins on it in `socket_reserve_pages`.

---

## 4.3.5 H2D vs D2H Symmetry

H2D and D2H are approximate mirrors of each other, with the roles of producer and consumer swapped. The following table highlights the symmetry and the asymmetries:

| Aspect | H2D (HOST_PUSH) | D2H |
|--------|-----------------|-----|
| Producer | Host CPU | Device kernel |
| Consumer | Device kernel | Host CPU |
| Data transport | Host TLB write to device L1 | Device NOC write to host pinned memory |
| `bytes_sent` location | Device L1 (written by host) | Host pinned memory (written by device) |
| `bytes_acked` location | Host pinned memory (written by device) | Device L1 (written by host) |
| Producer checks space by | Reading ack from host pinned memory (local) | Reading ack from device L1 (local, cache invalidation) |
| Consumer checks data by | Reading sent from device L1 (local) | Reading sent from host pinned memory (local) |
| Device-initiated PCIe transaction | `bytes_acked` write to host | Data + `bytes_sent` write to host |
| Host-initiated PCIe transaction | Data + `bytes_sent` write to device | `bytes_acked` write to device |

The symmetry is clean: in both cases, the frequent reader polls locally, and the infrequent writer pays for the cross-bus update. The asymmetry is in volume -- D2H pushes all data through device-initiated PCIe writes, while H2D HOST_PUSH pushes all data through host-initiated PCIe writes.

---

## 4.3.6 Batched Acknowledgement with notify_sender

By default, every `d2h.read()` call writes `bytes_acked` back to device L1 immediately. When the host reads many small pages in rapid succession, these per-read TLB writes can become a bottleneck -- each one is a PCIe posted write with non-trivial overhead relative to the small data size.

### Batching pattern

```python
# Source: application code pattern
d2h.set_page_size(page_size_bytes)

# Read 10 pages without acknowledging:
for i in range(10):
    d2h.read(buffers[i], 1, notify_sender=False)

# Single acknowledgement covers all 10 pages:
d2h.read(buffers[10], 1, notify_sender=True)
```

The final `read()` with `notify_sender=True` writes `bytes_acked` reflecting all 11 pages consumed. The device sees a single large jump in free space instead of 11 small increments.

### When to batch

- High-frequency small reads (e.g., reading individual output tokens from a decode kernel).
- When the FIFO is large relative to per-read size, so the device is unlikely to stall during the un-acked window.

### When NOT to batch

- When the FIFO is small and the device produces at a high rate. Withholding acknowledgement means the device cannot reuse FIFO slots, and it will stall at `socket_reserve_pages()` once the FIFO fills.

**Warning:** If you use `notify_sender=False`, you MUST eventually call `read()` with `notify_sender=True` or call `barrier()`. Failing to acknowledge will deadlock the device kernel permanently -- `socket_reserve_pages()` will spin forever waiting for space that never appears.

---

## 4.3.7 Error Recovery with discard_pending_pages

`discard_pending_pages()` is an error-recovery mechanism on D2H sockets. It rebases `bytes_acked` to the current `bytes_sent` value, effectively telling the device: "I have consumed everything you sent, even though I actually discarded it."

### When to use

- **Request cancellation.** An LLM serving system cancels a decode request mid-stream. The device kernel continues producing tokens briefly before it receives the cancellation signal. The host calls `discard_pending_pages()` to drain the FIFO without processing the stale tokens.
- **Speculative decode rollback.** A speculative decoding branch is rejected. The host discards the speculative output pages and resets the FIFO to a clean state.
- **Timeout recovery.** The host times out waiting for a result and decides to abandon the current batch.

### What it does and does not do

```python
# Source: d2h_socket.hpp
d2h.discard_pending_pages()
```

- **Does:** Sets `bytes_acked = bytes_sent` and writes the updated `bytes_acked` to device L1.
- **Does not:** Zero out or overwrite the data region in the pinned FIFO. The data remains in memory but is logically "consumed."
- **Does not:** Signal the device kernel to stop producing. The device must receive a separate application-level signal to stop sending.

After `discard_pending_pages()`, the FIFO appears empty to both sides and normal production/consumption can resume immediately.

---

## 4.3.8 PCIe Topology and Bandwidth Asymmetry

The physical PCIe connection between the host and device constrains all H2D and D2H throughput:

| Link type | Count per device | Bandwidth per link | Typical use |
|-----------|------------------|--------------------|-------------|
| High-bandwidth Gen4 x8 (ASIC 6) | 4 | ~16 GB/s each | Bulk data transfer (tensors, activations) |
| Low-bandwidth Gen4 x1 | 28 | ~2 GB/s each | Control, counters, small metadata |

Data pages flow over the high-BW links; a single socket can saturate one link (~16 GB/s) with sufficient page size and FIFO depth. Counter updates (8 bytes each) are negligible. DEVICE_PULL achieves peak bandwidth more easily for large transfers via pipelined NOC reads. Multiple sockets on different cores distribute load across links, approaching aggregate PCIe bandwidth. Minimum page size is 64 bytes (NOC word width = 512 bits); SRAM write alignment is 16 bytes.

---

## 4.3.9 Forward Reference: Chapter 5

This section provided the host-device-specific flow control mechanisms. Chapter 5 ([`../ch05_flow_control_and_memory/index.md`](../ch05_flow_control_and_memory/index.md)) extends this discussion to cover:

- The unified flow control model across all three socket types (H2D, D2D, D2H).
- Memory configuration strategies: L1 allocation budgets, FIFO sizing formulas, and page size tuning.
- Performance measurement and bottleneck identification for PCIe-bound workloads.
- Advanced patterns: double-buffering, credit-based flow control, and multi-socket pipelines.

---

## Key Takeaways

- All host-device sockets use the same circular FIFO model with `bytes_sent` and `bytes_acked` counters enforcing the invariant `0 <= bytes_acked <= bytes_sent <= bytes_acked + fifo_size`.
- Counter placement follows the principle "place the counter where the frequent reader lives" -- the reader polls locally, the writer pays for the cross-bus update.
- HOST_PUSH and D2H are approximate mirrors: the data transport direction reverses, and so does every counter's writer and reader.
- DEVICE_PULL is unique in placing both counters in device L1, which makes the host's space-check slower (TLB read) but enables the device to pipeline pulls efficiently.
- Batched acknowledgement (`notify_sender=False`) reduces per-read PCIe overhead but must eventually be followed by an acknowledgement to avoid deadlock.
- `discard_pending_pages()` is a D2H-only error-recovery mechanism that fast-forwards the ack counter without processing data.
- PCIe Gen4 x8 links provide ~16 GB/s per link; proper page sizing and FIFO depth are essential to saturate this bandwidth.

---

**Previous:** [`02_d2h_socket_api_and_external_config_buffer.md`](./02_d2h_socket_api_and_external_config_buffer.md) | **Next:** [`../ch05_flow_control_and_memory/index.md`](../ch05_flow_control_and_memory/index.md) | **Up:** [`index.md`](./index.md)
