# 5.1 The Unified Circular FIFO Model

Every socket in the Tenstorrent system -- regardless of type, direction, or transport -- uses the same flow control primitive: a circular buffer with two 64-bit monotonically increasing counters, `bytes_sent` and `bytes_acked`. Chapters 2-4 introduced this model per socket type. This section unifies the treatment into a single parameterized model and provides decision frameworks for reasoning about blocking, backpressure, and deadlock across multi-stage pipelines.

---

## 5.1.1 The Canonical FIFO

Every socket maintains a contiguous region of memory (the FIFO buffer) and two 64-bit unsigned integers (`bytes_sent`, `bytes_acked`). These three values are the complete state of flow control.

```
  Circular FIFO (fifo_size bytes, page-aligned)

  +----+----+----+----+----+----+----+----+----+----+
  |    |    | D  | D  | D  | D  |    |    |    |    |
  +----+----+----+----+----+----+----+----+----+----+
             ^                   ^
             |                   |
        read_offset          write_offset
        = bytes_acked        = bytes_sent
          % fifo_size          % fifo_size

  D = unread data pages     (blank) = free space
```

| Quantity | Formula | Meaning |
|----------|---------|---------|
| Available space | `fifo_size - (bytes_sent - bytes_acked)` | Bytes the producer may write before blocking |
| Available data | `bytes_sent - bytes_acked` | Bytes the consumer may read before blocking |
| Write offset | `bytes_sent % fifo_size` | Next write position in the circular buffer |
| Read offset | `bytes_acked % fifo_size` | Next read position in the circular buffer |

**Invariants:** (1) `bytes_acked <= bytes_sent`. (2) `bytes_sent - bytes_acked <= fifo_size` (enforced by blocking). Data moves in pages, not bytes. Minimum page size is 64 bytes (NOC word width, 512 bits). SRAM write alignment is 16 bytes. Both `fifo_size` and `page_size` are fixed at socket creation; `fifo_size` must be an exact multiple of `page_size`.

### Why 64-Bit Counters That Never Wrap

Both counters are 64-bit unsigned integers that increase monotonically from zero and are never reset. The modulo operation (`% fifo_size`) is applied only for buffer offsets, never to the counters. At 100 GB/s sustained throughput, overflow would take over 5,800 years. This eliminates wrap-around bugs inherent in 32-bit ring buffer implementations.

---

## 5.1.2 The Parameterized Model

The canonical FIFO becomes a concrete socket when three parameters are bound:

| Parameter | Description | Values by socket type |
|-----------|-------------|----------------------|
| **FIFO location** | Where the data buffer resides | D2D: receiver L1/DRAM. HOST_PUSH: device L1. DEVICE_PULL: host pinned memory (data) + device L1 (config). D2H: host pinned memory. |
| **Data transport** | How payload bytes move | D2D: TT-Fabric ethernet. HOST_PUSH: PCIe TLB posted write. DEVICE_PULL: device NOC read. D2H: device NOC write. |
| **Counter transport** | How each counter crosses the boundary | See the complete counter map in Section 5.1.3. |

You do not need four mental models. You need one model with four parameter sets.

```
  Type            | Producer     | Consumer     | FIFO location     | Data transport
  ----------------+--------------+--------------+-------------------+-------------------
  D2D             | Sender core  | Receiver core| Receiver L1/DRAM  | TT-Fabric ethernet
  H2D HOST_PUSH   | Host CPU     | Device core  | Device L1         | PCIe TLB posted write
  H2D DEVICE_PULL | Host CPU     | Device core  | Host pinned mem   | Device NOC reads
  D2H             | Device core  | Host CPU     | Host pinned mem   | Device NOC writes
```

---

## 5.1.3 Complete Counter Placement Map

This is the single authoritative reference for counter placement. See [Chapter 4](../ch04_host_device_sockets/index.md) for detailed data-path diagrams per mode.

| Mode | Counter | Writer | Written To | Via | Reader | Read From | Read Via |
|------|---------|--------|-----------|-----|--------|-----------|----------|
| D2D | `bytes_sent` | Sender device | Receiver device L1 | Fabric ethernet | Receiver device | Receiver device L1 | Local L1 read |
| D2D | `bytes_acked` | Receiver device | Sender device L1 | Fabric ethernet | Sender device | Sender device L1 | Local L1 read |
| H2D HOST_PUSH | `bytes_sent` | Host | Device L1 | TLB write (PCIe) | Device | Device L1 | Local L1 read |
| H2D HOST_PUSH | `bytes_acked` | Device | Device L1 (canonical) + host pinned memory (mirror) | Local L1 write + NOC write through PCIe | Host | Host pinned memory (mirror) | Local memory read |
| H2D DEVICE_PULL | `bytes_sent` | Host | Device L1 | TLB write (PCIe) | Device | Device L1 | Local L1 read |
| H2D DEVICE_PULL | `bytes_acked` | Device | Device L1 | Local L1 write | Host | Device L1 | TLB read (PCIe round trip) |
| D2H | `bytes_sent` | Device | Host pinned memory | NOC write through PCIe | Host | Host pinned memory | Local memory read |
| D2H | `bytes_acked` | Host | Device L1 | TLB write (PCIe) | Device | Device L1 | L1 read (cache invalidation) |

**Key distinction -- HOST_PUSH vs DEVICE_PULL:** In HOST_PUSH, the device mirrors `bytes_acked` to host pinned memory via a NOC write across PCIe, so the host polls locally. In DEVICE_PULL, there is no such mirror -- both counters reside only in device L1, and the host reads `bytes_acked` via a PCIe TLB read (a non-posted round-trip). This is why HOST_PUSH detects FIFO space availability faster on the host side.

**Design principle:** Each counter is placed where its frequent reader resides. Device-consumer modes place counters in device L1; host-consumer modes place them in host pinned memory.

### "What Breaks" -- Counter-Related Failure Scenarios

**HOST_PUSH -- host ignores bytes_acked:** Writing data without checking `bytes_acked` from pinned memory means the host cannot detect a full FIFO. It may overwrite unconsumed data. The API prevents this by polling before each write.

**DEVICE_PULL -- host updates bytes_sent before data write completes:** If the host writes `bytes_sent` to device L1 before the data write to pinned host memory finishes, the device NOC-reads stale content. The API orders data-write before counter-update.

**D2H -- host omits acknowledgement:** Calling `read()` with `notify_sender=false` repeatedly without acknowledging causes the device to see stale `bytes_acked` and stall in `socket_reserve_pages()`.

---

## 5.1.4 D2D: Dual-Layer Flow Control

D2D sockets have two independent flow control layers:

```
+----------------------------------------------+
|  Layer 2: Socket FIFO (bytes_sent/bytes_acked)|
|  Scope: end-to-end (sender to receiver core)  |
+----------------------------------------------+
            |  rides on top of
+----------------------------------------------+
|  Layer 1: TT-Fabric Link Credits              |
|  Scope: single hop (one Ethernet link)        |
+----------------------------------------------+
```

**Layer 1 (Fabric credits)** prevents link buffer overflow. Invisible to socket code -- managed by TT-Fabric hardware. **Layer 2 (Socket FIFO)** prevents overwriting unconsumed application data. The two layers are independent: a sender may have fabric credits but still block on a full FIFO, or have FIFO space but stall on a congested link.

H2D and D2H sockets do not have this dual-layer property. PCIe handles its own flow control transparently.

**"What breaks" -- oversized socket FIFO:** A burst can exhaust fabric credits while the socket reports ample space. Throughput appears low with bursty progress. Size the socket FIFO to match the bandwidth-delay product.

---

## 5.1.5 Blocking Behavior

The blocking rules are identical across all socket types. Only the wait mechanism differs.

**Producer blocks** when `bytes_sent - bytes_acked == fifo_size`:

| Mode | Wait mechanism | Unblock latency |
|------|----------------|-----------------|
| D2D | Polls `bytes_acked` via TT-Fabric from receiver | Fabric round-trip (microseconds) |
| H2D HOST_PUSH | Polls `bytes_acked` from host pinned memory (mirror) | PCIe posted write propagation |
| H2D DEVICE_PULL | Polls `bytes_acked` via TLB read from device L1 | PCIe non-posted round-trip |
| D2H | Polls `bytes_acked` in L1 via cache invalidation | L1 invalidation + PCIe read |

**Consumer blocks** when `bytes_sent == bytes_acked`:

| Mode | Wait mechanism | Unblock latency |
|------|----------------|-----------------|
| D2D | Reads `bytes_sent` from local L1 | Local L1 load (no PCIe) |
| H2D (both) | Reads `bytes_sent` in L1 via `socket_wait_for_pages()` | Local L1 load (no PCIe) |
| D2H | Polls `bytes_sent` in pinned memory | Local load (no PCIe) |

**Stall diagnosis:** Read the two counters. `difference == fifo_size` = producer blocked. `difference == 0` = consumer blocked. `0 < difference < fifo_size` = not a flow-control stall.

---

## 5.1.6 Backpressure Propagation

In a multi-stage pipeline, backpressure propagates naturally through FIFO blocking without explicit signaling.

```
  Host --> [H2D] --> Stage A --> [D2D] --> Stage B --> [D2D] --> Stage C --> [D2H] --> Host

  1. Host D2H reader slows -> D2H FIFO fills -> Stage C blocks
  2. Stage C stops consuming upstream D2D FIFO -> that fills -> Stage B blocks
  3. Cascade continues to H2D FIFO -> Host write() blocks
  --> End-to-end backpressure from a single slow consumer
```

This works identically across D2D, H2D, and D2H hops. FIFO sizes determine pipeline slack -- larger FIFOs absorb more per-stage variation before backpressure triggers. In the DeepSeek V3 pipeline ([Chapter 7](../ch07_production_usage/index.md)), FIFO and page sizes are constructor parameters tuned per-deployment.

**"What breaks" with undersized FIFOs:** Lock-step with zero stage overlap. **With oversized FIFOs:** Wasted L1 without throughput gain past the bandwidth-delay product.

---

## 5.1.7 Deadlock Avoidance

Deadlock occurs when socket connections form a cycle and all FIFOs fill simultaneously. This is a real risk in loopback pipelines (e.g., DeepSeek V3 autoregressive decode).

### Decision Tree: "Am I At Risk of Deadlock?"

```
Does my pipeline have a cycle in socket edges?
  |
  +-- No --> No deadlock risk.
  |
  +-- Yes
        |
        Does every stage consume before producing each iteration?
          +-- Yes --> No deadlock (consume-first ordering breaks the cycle).
          +-- No
                |
                Are FIFOs sized asymmetrically so one direction always drains?
                  +-- Yes --> Deadlock mitigated.
                  +-- No --> DEADLOCK RISK. Resize or restructure.
```

**Mitigations:** (1) Asymmetric FIFO sizing -- one direction large enough to always drain. (2) Consume-first ordering in kernel loops. (3) Host-mediated loopback (D2H + H2D breaks the device cycle). (4) Cycle-free topology.

**Warning:** Deadlock is silent -- no error is raised. Both sides spin indefinitely. Recovery requires socket teardown or device reset. Use the decision tree during design, not after a hang.

---

## Key Takeaways

- All four socket modes share one FIFO model with three parameters: FIFO location, data transport, and counter transport. The invariants, blocking rules, and backpressure behavior are identical everywhere.
- Counter placement follows the map in Section 5.1.3. HOST_PUSH mirrors `bytes_acked` to host pinned memory; DEVICE_PULL keeps both counters only in device L1 (host reads via slower TLB read).
- D2D has dual-layer flow control (per-hop fabric credits + end-to-end socket FIFO). Size the socket FIFO for application needs, not fabric credit pools.
- Backpressure propagates upstream automatically through FIFO blocking across all transport types.
- Deadlock arises only from cycles. Mitigate with asymmetric sizing, consume-first ordering, or cycle-free topology.
- All H2D and D2H modes require vIOMMU, including HOST_PUSH for control-path signaling.

---

*Previous: [Chapter 5 Index](index.md) | Next: [5.2 Memory Placement and FIFO Sizing](02_memory_placement_and_fifo_sizing.md)*
