# 8.2 Latency Model for Cross-Chip Data Movement

## Context

Section 8.1 established the throughput ceiling for fabric data movement using measured
benchmark data. Throughput, however, tells only half the story -- **latency** determines
responsiveness for fine-grained communication, synchronization operations, and interactive
workloads. This section builds a first-principles latency model for cross-chip data movement
on Wormhole and Blackhole, decomposes the cost of each pipeline stage, compares explicit CCL
and GSRP latency profiles, and estimates the overhead of the GSRP's address translation and
routing layers. The reader should be familiar with the EDM store-and-forward architecture
(Chapter 3, Section 2), the `PACKET_WORD_SIZE_BYTES = 16` granularity (Chapter 3, Section 1),
the address translation pipeline (Chapter 6, Section 5), and the speedy-path optimization
(Chapter 3, Section 2).

---

## 8.2.1 Latency Decomposition

A cross-chip data transfer traverses multiple pipeline stages. Each stage contributes a
fixed or size-dependent latency component:

```
Tensix Core                EDM Sender            Ethernet Link          EDM Receiver         Dest Core
+----------+    NOC     +------------+   Link   +------------+  NOC   +-----------+  NOC   +----------+
| Payload  |----------->| Buffer     |--------->| Wire       |------->| Buffer    |------->| L1 Write |
| Prepare  |   write    | + Header   |  xfer    | Transfer   | write  | + Forward |  write | Complete |
+----------+            +------------+          +------------+        +-----------+        +----------+
   T_prep      T_noc       T_edm_tx     T_eth      T_wire      T_rx     T_edm_rx   T_noc    T_dest
```

### Stage-by-Stage Cost Model

| Stage | Symbol | Formula | Typical Value (4096 B) |
|-------|--------|---------|:----------------------:|
| Payload preparation | $T_{\text{prep}}$ | Header fill + address setup | ~20--30 ns |
| NOC write to EDM | $T_{\text{noc}}$ | $\lceil \frac{size}{32} \rceil \times T_{\text{noc-beat}}$ | ~130 ns |
| EDM buffer management | $T_{\text{edm-tx}}$ | Slot allocation + credit check | ~15--25 ns |
| Ethernet link transfer | $T_{\text{eth}}$ | $\frac{size + hdr}{BW_{\text{link}}}$ | ~330 ns |
| Wire propagation | $T_{\text{wire}}$ | Physical propagation (PCB trace) | ~5--10 ns |
| EDM receive + forward | $T_{\text{edm-rx}}$ | Buffer write + forwarding decision | ~20--40 ns |
| NOC write at destination | $T_{\text{dest}}$ | $\lceil \frac{size}{32} \rceil \times T_{\text{noc-beat}}$ | ~130 ns |

### Single-Hop Latency (Estimated)

$$T_{\text{single-hop}} = T_{\text{prep}} + T_{\text{noc}} + T_{\text{edm-tx}} + T_{\text{eth}} + T_{\text{wire}} + T_{\text{edm-rx}} + T_{\text{dest}}$$

For a 4096 B payload on a single BH link at ~12.5 GB/s:

$$T_{\text{single-hop}} \approx 25 + 130 + 20 + 328 + 8 + 30 + 130 \approx 671\ \text{ns}$$

### Best/Typical/Worst Case Estimates

| Packet Size | Best Case (ns) | Typical (ns) | Worst Case (ns) |
|:-----------:|:--------------:|:------------:|:----------------:|
| 16 B | 240 | 370 | 580 |
| 256 B | 260 | 390 | 600 |
| 1,024 B | 310 | 440 | 660 |
| 4,096 B | 560 | 671 | 910 |
| 16,384 B | 1,530 | 1,660 | 1,880 |

The range reflects variation in NOC distance (Tensix-to-ERISC hops on-chip), EDM queue
occupancy (contention from other traffic), and destination NOC congestion.

---

## 8.2.2 Multi-Hop Latency: Store-and-Forward Model

TT-Fabric uses store-and-forward routing (Chapter 3, Section 2): each intermediate EDM
node receives the complete packet before forwarding it to the next hop. For $N$ hops:

$$T_{N\text{-hop}} = T_{\text{prep}} + T_{\text{noc}} + T_{\text{edm-tx}} + N \times (T_{\text{eth}} + T_{\text{wire}} + T_{\text{edm-rx}}) + T_{\text{dest}}$$

The per-hop increment is approximately:

$$\Delta T_{\text{hop}} = T_{\text{eth}} + T_{\text{wire}} + T_{\text{edm-rx}} \approx 328 + 8 + 30 = 366\ \text{ns per hop}$$

### Latency Estimates by Mesh Size

| Mesh Configuration | Max Hops | Est. Latency (4096 B) | Est. Latency (16 B) |
|-------------------|:--------:|:---------------------:|:-------------------:|
| 2-chip BH | 1 | ~0.7 us | ~0.2 us |
| T3K (2x4, WH) | 5 | ~2.2 us | ~0.6 us |
| Galaxy (8x4, WH/BH) | 11 | ~4.4 us | ~1.2 us |
| 32-chip BH (4x8, 2D) | 10 | ~4.0 us | ~1.1 us |
| Multi-Galaxy (64 chips) | 23 | ~8.8 us | ~2.4 us |

> **Key Insight:** Latency scales linearly with hop count in the store-and-forward model.
> For a 64-chip multi-Galaxy system, worst-case latency for a single 4096 B packet
> approaches 9 us. This is 4--5 orders of magnitude slower than local L1 access (~1 ns),
> reinforcing the NUMA analogy from Chapter 5.

---

## 8.2.3 Pipeline Saturation: Throughput vs. Latency

When sending a stream of packets, the pipeline fills and the effective throughput becomes
limited by the slowest stage rather than the sum of all stages:

```
Packet 1:  [prep][noc][edm][=====eth=====][rx][dest]
Packet 2:       [prep][noc][edm][=====eth=====][rx][dest]
Packet 3:            [prep][noc][edm][=====eth=====][rx][dest]
                                 ^
                                 |
                           Pipeline bottleneck:
                           Ethernet link transfer
```

The steady-state throughput for a saturated pipeline equals the bandwidth of the bottleneck
stage -- typically the Ethernet link transfer. This explains why the measured bandwidth in
Section 8.1 (e.g., 12.8 B/c for BH) reflects the link bandwidth rather than the sum of
all overheads: the benchmark measures sustained throughput after pipeline warmup.

**Implication for GSRP:** The latency model matters most for:
1. Single-message round trips (e.g., remote read, atomic operations)
2. Pipeline startup cost (first packet in a burst)
3. Synchronization operations (barrier, fence)

For sustained bulk transfers, the throughput model from Section 8.1 is the correct
performance predictor.

---

## 8.2.4 The Speedy-Path Optimization

The EDM router firmware includes a "speedy path" (Chapter 3, Section 2) that reduces
per-packet processing overhead at each hop. From the source:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp
// This header provides amortized credit-passing "speedy" step functions for the
// ERISC router's sender and receiver channels.
```

The speedy path batches credit updates and eliminates per-packet credit round-trips, saving
approximately **9 cycles per send** (from the router step function hot path). At ~1 GHz:

$$\Delta T_{\text{speedy}} \approx 9\ \text{ns per hop saved}$$

For an 11-hop Galaxy path, this saves $11 \times 9 = 99$ ns, reducing total latency by
~2.3%. The `super_speedy_mode` compile-time constant (`fabric_erisc_router_ct_args.hpp`)
enables this path when the channel configuration permits it.

> **Key Insight:** The 9-cycle speedy-path saving per hop provides a reference for the
> **minimum cost of per-hop processing** in the EDM firmware. Any GSRP routing-decision
> logic added to the firmware path must be benchmarked against this baseline -- adding more
> than ~9 cycles of per-hop overhead would negate the optimization entirely.

---

## 8.2.5 Explicit CCL Latency Profile

Explicit CCL operations (Chapter 1) achieve near-optimal latency through:

1. **Pre-computed routes:** All packet headers are constructed before the transfer begins,
   eliminating per-packet routing decisions.
2. **Pipelined transfers:** Chunks are overlapped across hops using ring/pipeline algorithms
   (Chapter 1, Section 2).
3. **Deterministic scheduling:** The host pre-determines the exact sequence of sends, so
   there are no runtime routing conflicts.

For an all-gather on a T3K (2x4, 8 chips), the explicit CCL achieves:

```
Explicit CCL all-gather latency (8 chips, ring topology):
  Pipeline stages = 7 (one per ring hop)
  Per-stage latency = T_eth + T_edm ~ 366 ns
  Total pipeline fill = 7 x 366 ns = 2.56 us
  Steady-state: one chunk delivered every 366 ns after fill
  Total for N chunks: 2.56 us + (N-1) x 366 ns
```

The key property: explicit CCL latency is **predictable** and **deterministic**. There are
no runtime routing decisions, no address translation lookups, and no congestion-induced
queuing delays.

---

## 8.2.6 GSRP Latency Overhead

The GSRP adds per-access overhead components not present in explicit CCL:

The per-access overhead components (address decomposition, route table lookup, connection
pool lookup, header population) total **~13 cycles** on the cached path and **~18--27 cycles**
uncached (see Chapter 6, Section 5.5 for the full component breakdown).

### Overhead as Percentage of Transfer Time

$$\text{Overhead \%} = \frac{T_{\text{translation}}}{T_{\text{transfer}}} \times 100$$

| Payload Size | Transfer Time (single-hop) | Cached Overhead | Overhead % |
|:------------:|:--------------------------:|:--------------:|:----------:|
| 16 B | ~13 ns | ~13 ns | **~100%** |
| 64 B | ~20 ns | ~13 ns | **~65%** |
| 256 B | ~50 ns | ~13 ns | **~26%** |
| 1,024 B | ~110 ns | ~13 ns | **~12%** |
| 2,048 B | ~195 ns | ~13 ns | **~6.7%** |
| 4,096 B | ~370 ns | ~13 ns | **~3.5%** |
| 8,192 B | ~700 ns | ~13 ns | **~1.9%** |
| 16,384 B | ~1,400 ns | ~13 ns | **~0.9%** |

### Break-Even Points

Two useful break-even thresholds:

- **<5% overhead threshold (~3 KB):** Derived from $13 \times 12.78 / 0.05 \approx 3,323$ B.
  At this size, GSRP overhead drops below the 5--15% range cited for small messages.

- **<2% overhead threshold (~8 KB):** Derived from $13 \times 12.78 / 0.02 \approx 8,307$ B.
  At this size, GSRP overhead is below 2%, matching the bulk-transfer convergence target.

> **Recommendation:** For workloads that primarily move tensors (64 KB--several MB), GSRP
> overhead is negligible (<0.2%). For small-message workloads (<1 KB), the runtime should
> employ message aggregation or connection caching to amortize the translation cost across
> multiple logical writes.

---

## 8.2.7 Connection Setup and Amortization

Beyond per-packet translation, the GSRP incurs a one-time **connection setup** cost when
a Tensix core first communicates with a new remote destination:

| Connection Phase | Cost | Amortization |
|:----------------|:----:|:------------:|
| Open connection handshake | ~500--1,000 ns | Once per destination per program |
| Route cache warm-up | ~50 ns (first lookup) | Cached after first access |
| Credit initialization | ~200 ns | Once per connection |
| **Total first-access** | **~750--1,250 ns** | |

The persistent connection pool (Chapter 7, Section 1) maintains open connections across
multiple `gsrp_write` calls, amortizing the setup cost. Quantifying the amortization
benefit:

```
Connection Management Cost Comparison:

  Explicit CCL (per operation):
    100 connection ops x ~3 us each = ~300 us total
    No amortization across operations

  GSRP Persistent Pool:
    First access: ~1 us setup
    Subsequent accesses: ~0.005 us (cached lookup)
    100 operations: ~1 us + 99 x 0.005 us = ~1.5 us

  Improvement: ~300 us / ~1.5 us = ~200x reduction in connection management
```

For workloads with stable communication patterns (e.g., model parallelism with fixed tensor
placements), this amortization is transformative. For dynamic patterns (e.g., MoE with
data-dependent routing), connection thrashing could add 1--2 us per new destination, but
the persistent pool's LRU eviction policy (Chapter 7, Section 1.7) limits the worst case.

---

## 8.2.8 Round-Trip Latency for Remote Reads

Remote reads (Chapter 6, Section 5; Chapter 7, Section 1) require a fabric-level
request-response protocol:

```
gsrp_read() Flow:

  Requester            Fabric              Remote Chip
  +--------+         +--------+          +-----------+
  | Build  |-------->| Forward|--------->| ERISC     |
  | request|  N hops | request|          | processes |
  | packet |         |        |          | read req  |
  +--------+         +--------+          +-----+-----+
                                               |
  +--------+         +--------+          +-----+-----+
  | Receive|<--------| Forward|<---------| Build     |
  | data   |  N hops | response          | response  |
  +--------+         +--------+          +-----------+
```

The round-trip latency:

$$T_{\text{read}} = T_{\text{request}}(N\ \text{hops}) + T_{\text{ERISC processing}} + T_{\text{response}}(N\ \text{hops})$$

For a 4096 B read, the request packet is small (~24 B) and the response carries the full
data. The ERISC processing time at the remote chip includes NOC read from target L1, buffer
copy, and response header construction:

$$T_{\text{ERISC}} \approx 200\text{--}500\ \text{ns}$$

### Read Latency Estimates

| Mesh | Hops | Write Latency | Read Latency (round-trip) | Read/Write Ratio |
|------|:----:|:-------------:|:-------------------------:|:----------------:|
| 2-chip BH | 1 | ~0.7 us | ~1.4--1.6 us | ~2.0--2.3x |
| T3K (2x4) | 5 | ~2.2 us | ~4.4--4.7 us | ~2.0--2.1x |
| Galaxy (8x4) | 11 | ~4.4 us | ~8.8--9.3 us | ~2.0--2.1x |
| 64-chip | 23 | ~8.8 us | ~17.6--18.4 us | ~2.0--2.1x |

The read/write ratio of **~2.0--2.3x** is physically grounded: a read requires one
round-trip (request + response) compared to a write's one-way trip. The ratio converges
toward 2.0x for longer paths because the fixed ERISC processing time becomes a smaller
fraction of the total. For multi-hop reads, each hop adds latency to **both** the request
and response paths, amplifying the absolute penalty but keeping the ratio stable.

> **Key Insight:** The write-read asymmetry means GSRP workloads should favor **push-based**
> (write) patterns over **pull-based** (read) patterns. This is consistent with the existing
> TT-Metal programming model, which is built around posted NOC writes rather than reads.

---

## 8.2.9 Synchronization and Barrier Latency

The GSRP epoch barrier (Chapter 6, Section 3) introduces synchronization latency:

$$T_{\text{barrier}} = T_{\text{drain}} + T_{\text{all-reduce-ack}} + T_{\text{resume}}$$

| Component | Estimate | Notes |
|-----------|:--------:|-------|
| Drain outstanding writes | 0--10 us | Depends on in-flight traffic volume |
| All-reduce acknowledgment | ~2--5 us (T3K) | Atomic increment propagation around ring |
| Resume notification | ~1--2 us | Broadcast to all chips |
| **Total barrier** | **~3--17 us** | |

For comparison, explicit CCL uses per-collective semaphore barriers that cost approximately
2--5 us. The GSRP epoch barrier is 1.5--3x more expensive because it must drain **all**
outstanding traffic, not just a single collective's traffic.

> **Recommendation:** Epoch barriers should be used sparingly -- ideally once per training
> iteration or inference batch, not per-layer. The release-consistency model (Chapter 6,
> Section 3) allows most inter-chip communication to proceed without barriers, using
> acquire-release semantics on individual memory locations instead.

---

## 8.2.10 Summary: Small Messages vs. Large Bulk Transfers

| Metric | Small (64 B) | Medium (4 KB) | Large (64 KB) |
|--------|:------------:|:-------------:|:-------------:|
| Explicit CCL overhead | 0% (no translation) | 0% | 0% |
| GSRP translation overhead | ~65% | ~3.5% | ~0.2% |
| GSRP connection setup (amortized) | ~15% | ~1% | ~0.1% |
| **Total GSRP overhead** | **~80%** | **~4.5%** | **~0.3%** |

The GSRP overhead profile matches the Chapter 6 design targets: 5--15% for small messages
(< 2 KB) and converging to <2% for bulk transfers (> 8 KB).

---

## Key Takeaways

- **Single-hop latency is approximately 671 ns** for a 4096 B packet on BH, scaling linearly
  with hop count at ~366 ns per additional hop.

- **Multi-hop latency reaches 4--9 us** for Galaxy-scale (32--64 chip) deployments, placing
  cross-chip remote access firmly in the microsecond regime.

- **The speedy-path optimization saves ~9 cycles per hop**, establishing a baseline for
  acceptable per-hop firmware overhead. GSRP routing logic must stay within this budget.

- **GSRP translation overhead is ~13 cycles (cached path)**, yielding <5% overhead for
  transfers above ~3 KB and <2% for transfers above ~8 KB.

- **Remote reads incur a ~2.0--2.3x latency multiplier** relative to writes. Push-based
  patterns should be preferred for GSRP workloads.

## GSRP Implications

The latency model reveals that GSRP is not latency-competitive with explicit CCL for
fine-grained, synchronous access patterns. The translation overhead of ~13 cycles per
access (cached) is acceptable for medium and large payloads but prohibitive for 16 B
granularity operations. The GSRP runtime must therefore provide batching and aggregation
primitives that allow applications to accumulate small writes into larger fabric packets.
The connection pool and route cache (Chapter 7, Sections 1 and 2) are essential for
keeping per-access overhead below the 5% target. For remote reads, the ~2x round-trip
penalty makes a producer-push model strongly preferable to a consumer-pull model.

## Source Code References

| File | Content |
|------|---------|
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp` | Speedy-path per-hop optimization (~9 cycles saved) |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` | `super_speedy_mode` compile-time constant |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | EDM router main loop with speedy/non-speedy dispatch |
| `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h` | `PACKET_WORD_SIZE_BYTES = 16` granularity |
| `tt_metal/fabric/fabric_edm_packet_header.hpp` | Packet header structure and size constants |

---

**Previous:** [`01_measured_fabric_bandwidth.md`](./01_measured_fabric_bandwidth.md) | **Next:** [`03_explicit_vs_implicit_tradeoff_matrix.md`](./03_explicit_vs_implicit_tradeoff_matrix.md)
