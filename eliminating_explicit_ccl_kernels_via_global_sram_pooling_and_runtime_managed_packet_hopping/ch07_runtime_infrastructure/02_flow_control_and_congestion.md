# 7.2 Flow Control and Congestion Management

## Context

Section 7.1 designed the transparent injection mechanism that converts a `gsrp_write()` call
into a fabric packet with ~13 cycles of CPU overhead. Once packets are injected, they must
traverse the multi-hop fabric without overflowing intermediate buffers, starving co-located
traffic, or degrading aggregate throughput. Chapter 3, Section 6 documented the existing
credit-based flow control: hop-by-hop credits between EDM sender and receiver channels,
stream-register-based and counter-based credit implementations selected at compile time
via `ReceiverChannelResponseCreditSender`, and the `BufferPtr`/`BufferIndex` strong-type
pointer management with power-of-2 optimized `wrap_increment`. This section analyzes how
GSRP's implicit, potentially bursty traffic stresses these mechanisms and proposes
extensions -- dynamic credit allocation, per-destination flow control, priority queuing via
virtual channels, congestion mitigation including MUX aggregation and token-bucket rate
limiting, and end-to-end credit protocols -- to maintain fabric health under transparent
cross-chip data movement.

---

## 7.2.1 The Flow Control Challenge Under Implicit Traffic

In the explicit CCL model, flow control is manageable because: (1) the kernel knows exactly
how many packets it will send and to which destinations, (2) connection setup pre-allocates
buffer slots (Chapter 3, Section 2), and (3) the `wait_for_empty_write_slot()` backpressure
loop is explicitly coded in the kernel. Under GSRP, these assumptions break down:

| Explicit CCL Property | GSRP Reality |
|:---------------------:|:------------:|
| Known packet count per kernel | Unknown -- data-dependent access patterns |
| Single destination per connection | Multiple destinations via connection pool |
| Static credit allocation | Dynamic credit demands across destinations |
| Coordinated timing (host-orchestrated) | Uncoordinated -- any core, any time |
| Write-only, unidirectional | Bidirectional (writes + read requests/responses) |
| Large, pipelined messages (KB--MB) | Mixed: small (64 B semaphores) to large (KB data) |

The fundamental challenge is **congestion amplification**: when $N$ Tensix cores
transparently target the same remote chip, the aggregate demand on the Ethernet links to
that chip can exceed link capacity by $3\text{--}11\times$.

### Worst-Case Congestion Scenario

Consider a 32-chip Blackhole Galaxy where 140 Tensix cores on chip 0 simultaneously issue
4 KB GSRP writes to chip 15 (8 hops away in a 2D mesh):

$$\text{Aggregate burst} = 140\ \text{cores} \times 4\ \text{KB} = 560\ \text{KB}$$

Each Ethernet link provides ~12.5 GB/s. With 4 links in the direction of chip 15:

$$\text{Link capacity} = 4 \times 12.5\ \text{GB/s} = 50\ \text{GB/s}$$

The 560 KB burst completes in $\frac{560\ \text{KB}}{50\ \text{GB/s}} \approx 11\ \mu$s --
manageable for a single burst. But if all 140 cores sustain 1 GB/s each:

$$\text{Sustained demand} = 140\ \text{GB/s} \gg 50\ \text{GB/s (link capacity)}$$

$$\text{Oversubscription ratio} = \frac{140}{50} = 2.8\times$$

Under explicit CCL, this is managed by the programmer (only designated cores communicate).
Under GSRP, any core can generate remote traffic at any time without coordination, making
congestion management a **runtime responsibility**.

---

## 7.2.2 Existing Flow Control Mechanisms and Their Adequacy

The TT-Fabric flow control operates at three levels -- worker-to-EDM (stream register
credits), EDM-to-EDM (completion + ack credits), and EDM-to-local (posted NOC writes) --
documented in Chapter 3, Section 6. The key invariant: **no packet is sent unless the
receiver has advertised available buffer space**, propagating backpressure from congested
links back to traffic sources.

### Current Credit Sizing

Credit capacity is determined by `NUM_BUFFERS` in `StaticSizedSenderEthChannel`:

| Configuration | Buffer slots | Slot size | Channel capacity |
|:-------------:|:------------:|:---------:|:----------------:|
| Typical WH | 4 | 4 KB | 16 KB |
| Typical BH | 4--8 | 4--16 KB | 16--128 KB |
| BH with MUX | 8 | 16 KB | 128 KB |

When all buffer slots are occupied, the sender blocks. Backpressure propagates upstream:
if the intermediate EDM's sender channel fills, the upstream receiver cannot forward, so
the upstream sender's credits deplete, and the original worker blocks.

### Level-by-Level GSRP Gap Analysis

**Level 1 (Worker-to-EDM):** The `edm_has_space_for_packet()` check (Chapter 4,
Section 2.1) naturally provides backpressure. When credits are exhausted, the GSRP shim
spins in `wait_for_empty_write_slot()`, stalling the kernel until the EDM frees a buffer
slot. This is correct but may cause **head-of-line blocking** -- the stalled core cannot
issue writes to other (non-congested) destinations via the same connection. With MUX
aggregation, the MUX core is the single writer to the EDM channel. The 8--16 Tensix
workers behind the MUX need their own flow control, creating a **two-level credit
hierarchy** that does not exist today.

**Level 2 (EDM-to-EDM):** The hop-by-hop credit mechanism is fundamentally sound -- it
prevents buffer overflow at every hop. However, both credit variants provide a single
credit counter per sender channel. There is no per-destination or per-traffic-class
differentiation. A high-priority GSRP semaphore write (64 B) competes for the same buffer
slots as a low-priority bulk activation transfer (64 KB).

**Level 3 (EDM-to-Local):** Sufficient for writes (NOC writes are posted). For read
responses, an additional completion signaling mechanism is needed (the
`completion_counter_addr` from Section 7.1.5).

---

## 7.2.3 Proposed Extension: Dynamic Credit Allocation

### Motivation

The existing `StaticSizedSenderEthChannel` allocates a fixed number of buffer slots per
channel at initialization. Under GSRP, different routing planes experience different load
levels depending on which cores are active and what destinations they target. Static
allocation either wastes buffers on lightly loaded channels or under-provisions heavily
loaded ones.

### Elastic Buffer Pool

The `ElasticSenderEthChannel` stubs in the codebase (Chapter 3, Section 2.6, tracked as
Issue #26311) would enable runtime buffer reallocation:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
template <typename HEADER_TYPE>
class ElasticSenderEthChannel : public SenderEthChannelInterface<...> {
    // Dynamic buffer allocation from shared pool
    uint32_t* shared_pool_base;
    uint32_t  shared_pool_size;
    uint32_t  allocated_slots;
    uint32_t  max_slots;
};

constexpr bool USE_STATIC_SIZED_CHANNEL_BUFFERS = true;  // Currently hardcoded
```

Implementing elastic channels would allow GSRP traffic to borrow unused buffer capacity
from lightly loaded channels when heavily loaded channels need more slots. The minimum
guarantee (2 reserved slots per channel) prevents starvation.

### Credit Redistribution Protocol

```
ERISC Router Main Loop (extended):
  1. Monitor per-channel occupancy: credits_used / credits_total
  2. If channel[i].occupancy > HIGH_WATERMARK (80%):
     - Check if channel[j].occupancy < LOW_WATERMARK (20%)
     - Transfer 1 buffer slot from j to i
     - Update credit advertisements to both upstream senders
  3. Redistribution frequency: every 128 router iterations
     (matching default ctx_switch interval)
```

### Overhead Analysis

| Operation | Cycles | Frequency |
|-----------|:------:|:---------:|
| Occupancy check (2 channels) | 4 | Every 128 iterations |
| Credit transfer (1 slot) | ~20 | When triggered (~1% of checks) |
| Amortized per-packet cost | $\frac{4}{128} + \frac{20 \times 0.01}{128} \approx 0.03$ | Negligible |

> **Key Insight:** Dynamic credit allocation adds less than 0.1 cycle amortized overhead per
> packet while enabling the router to adapt to GSRP's non-uniform traffic patterns. The
> mechanism should be guarded by a compile-time flag (`GSRP_DYNAMIC_CREDITS_ENABLED`) to
> avoid any impact on non-GSRP workloads.

---

## 7.2.4 Per-Destination Flow Control

### Problem

The current flow control is per-channel, not per-destination. When multiple Tensix cores
share a single EDM channel via MUX, credits are consumed on a first-come-first-served
basis. A single high-bandwidth destination can monopolize the channel, starving traffic
to other destinations.

### Per-Destination Credit Partitioning

Divide the channel's buffer slots into reserved-per-destination and shared partitions:

```
Channel Buffer (8 slots total):
+--------+--------+--------+--------+--------+--------+--------+--------+
| Dest 0 | Dest 0 | Dest 1 | Dest 1 | Dest 2 | Dest 2 | Shared | Shared |
+--------+--------+--------+--------+--------+--------+--------+--------+
   Reserved (2 per destination)       |     Shared pool (first-come)
```

Implementation in the MUX kernel:

```cpp
// Proposed extension to tt_fabric_mux.cpp
struct PerDestCreditTracker {
    uint8_t reserved_slots_per_dest;  // Guaranteed minimum
    uint8_t shared_pool_slots;        // Available to any destination
    uint8_t dest_usage[MAX_DESTINATIONS];  // Current per-dest occupancy

    FORCE_INLINE bool has_credits(uint8_t dest_id) const {
        if (dest_usage[dest_id] < reserved_slots_per_dest) return true;
        uint8_t total_shared_used = 0;
        for (uint8_t i = 0; i < MAX_DESTINATIONS; i++) {
            if (dest_usage[i] > reserved_slots_per_dest)
                total_shared_used += dest_usage[i] - reserved_slots_per_dest;
        }
        return total_shared_used < shared_pool_slots;
    }
};
```

### Design Parameters

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| Max destinations per MUX | 8 | Matches `MAX_DOWNSTREAM_EDM_COUNT` (4) x 2 directions |
| Reserved slots per destination | 1--2 | Guarantees progress for every destination |
| Shared pool size | $N_{\text{total}} - N_{\text{dest}} \times N_{\text{reserved}}$ | Remaining slots |
| Tracker L1 cost | 16 bytes | 8 dest IDs + 8 usage counters |

### Policy Comparison

| Policy | Fairness | Throughput | Tail latency | Complexity |
|--------|:--------:|:----------:|:------------:|:----------:|
| Shared-only (current) | Poor | Best (no fragmentation) | Unbounded | Minimal |
| Strict partitioning | Perfect | Poor (fragmentation) | Bounded | Low |
| Reserved + shared pool | Good | Good | Bounded | Moderate |

> **Recommendation:** The reserved-plus-shared-pool policy provides the best balance. Reserve
> 1 slot per active destination, share the rest. This bounds worst-case starvation to
> $\frac{N_{\text{reserved}} \times \text{slot size}}{\text{link BW}}$ while allowing bursty
> traffic to use shared capacity.

---

## 7.2.5 Priority Queuing via Virtual Channels

### Motivation

GSRP traffic includes both latency-sensitive small messages (semaphore increments, control
words) and bandwidth-sensitive bulk transfers (activation shards, weight blocks). Without
priority differentiation, a large bulk transfer can block a time-critical semaphore for
hundreds of microseconds.

### Two-Priority Scheme via VC Separation

Map GSRP traffic to two priority levels using the existing virtual channel (VC) mechanism.
The `FabricEriscDatamoverConfig` already supports per-VC channel counts:

```cpp
// tt_metal/fabric/erisc_datamover_builder.hpp
std::array<std::size_t, builder_config::MAX_NUM_VCS>
    num_used_sender_channels_per_vc = {0, 0};
std::array<std::size_t, builder_config::MAX_NUM_VCS>
    num_used_receiver_channels_per_vc = {0, 0};
```

Setting `num_used_sender_channels_per_vc = {1, 1}` allocates one sender channel per VC:

| Priority | VC | Traffic type | Max message size | Buffer allocation |
|:--------:|:--:|-------------|:----------------:|:-----------------:|
| High | VC 0 | Semaphores, flags, metadata | 256 B | 2 dedicated slots |
| Normal | VC 1 | Data transfers, shards | 64 KB | Remaining slots |

> **Why not S0/S1 channel split?** In the current EDM architecture (Chapter 3, Section 2),
> S1 is the **forwarding channel** that accepts traffic from upstream receivers -- it is not
> a worker-facing channel. Assigning bulk GSRP worker injection to S1 would mix worker
> traffic with multi-hop forwarding, causing head-of-line blocking in multi-hop scenarios.
> The VC-based approach uses the existing `num_used_sender_channels_per_vc` infrastructure,
> which separates traffic within the EDM scheduler without conflating injection and forwarding.

### Priority Classification

The GSRP shim classifies traffic by size at injection time:

```cpp
FORCE_INLINE uint8_t gsrp_priority_classify(uint32_t size_bytes) {
    return (size_bytes <= GSRP_HIGH_PRIORITY_THRESHOLD) ? 0 : 1;
}

// Default threshold: 256 bytes
static constexpr uint32_t GSRP_HIGH_PRIORITY_THRESHOLD = 256;
```

### Latency Impact

| Message type | Without priority | With priority |
|-------------|:----------------:|:-------------:|
| 64-byte semaphore (behind 64 KB bulk) | ~5.2 $\mu$s wait | ~0.1 $\mu$s |
| 4 KB metadata (behind 64 KB bulk) | ~5.2 $\mu$s wait | ~0.33 $\mu$s |
| 64 KB bulk transfer | ~5.2 $\mu$s | ~5.4 $\mu$s (+3.8% for VC overhead) |

Priority queuing reduces worst-case latency for small messages by approximately 50x while
adding only ~3--4% overhead to bulk transfers.

> **Key Insight:** Priority queuing via existing per-VC channel separation provides the
> latency isolation that GSRP needs without requiring firmware modifications to the EDM
> forwarding path. The VC 0 / VC 1 split also creates the foundation for the deadlock
> avoidance strategy in Section 7.3 (request/response channel separation).

---

## 7.2.6 Congestion Mitigation Strategies

### Strategy 1: Local Aggregation via Tensix MUX

The MUX kernel (Chapter 3, Section 5) aggregates traffic from multiple worker cores into a
single EDM sender channel. For GSRP, the MUX provides a natural congestion mitigation point:

```
Tensix Workers (16 cores)           MUX Core            EDM Channel
+-----+  +-----+  +-----+         +--------+          +----------+
| W0  |  | W1  |  | ... |         |        |          |          |
| gsrp|->| gsrp|->| gsrp|-------->| MUX    |--------->| Sender   |
| write  | write  | write         | kernel |          | Channel 0|
+-----+  +-----+  +-----+         +--------+          +----------+
                                   Aggregates up to
                                   16 sources into 1
                                   channel
```

MUX aggregation provides three benefits for GSRP:

1. **Traffic shaping.** The MUX serializes bursty per-core traffic into a steady stream,
   preventing instantaneous incast at the ERISC.
2. **Write coalescing.** Multiple small GSRP writes to the same destination chip can be
   merged into a single larger packet, improving Ethernet link utilization.
3. **Credit management.** The MUX holds the single EDM connection and distributes virtual
   credits to upstream workers, providing implicit rate limiting.

**Coalescing efficiency analysis:**

| Batch strategy | Overhead per byte | Effective BW |
|---------------|:-----------------:|:------------:|
| Individual 64 B writes | Header (96 B) per 64 B = 60% | ~5.0 GB/s |
| Coalesced 4x64 B = 256 B | Header (96 B) per 256 B = 27% | ~9.1 GB/s |
| Coalesced 16x64 B = 1 KB | Header (96 B) per 1 KB = 9% | ~11.4 GB/s |
| Single 4 KB write | Header (96 B) per 4 KB = 2% | ~12.3 GB/s |

The scatter write capability (`NocUnicastScatterCommandHeader`, Chapter 3, Section 2)
enables coalescing multiple small writes to different local destinations on the same remote
chip into a single fabric packet without requiring new packet types.

**Coalescing rules:**

| Condition | Action |
|-----------|--------|
| Multiple writes to same destination chip within timeout | Merge into single packet |
| Writes to different chips | Send separately (no coalescing) |
| Timeout (e.g., 100 ns) without coalescing opportunity | Flush current batch |
| Coalesced size reaches max packet size | Flush immediately |

### Strategy 2: Token-Bucket Rate Limiting

For workloads with known bandwidth requirements, the GSRP runtime can enforce per-core
rate limits to prevent any single Tensix core from monopolizing the fabric:

```cpp
// Per-core rate limiter (token bucket)
struct GsrpRateLimiter {
    uint32_t tokens;           // Available bandwidth tokens (bytes)
    uint32_t max_tokens;       // Bucket capacity
    uint32_t refill_rate;      // Tokens per refill interval
    uint32_t last_refill_cycle;

    FORCE_INLINE bool try_consume(uint32_t size_bytes) {
        refill();
        if (tokens >= size_bytes) {
            tokens -= size_bytes;
            return true;
        }
        return false;  // Caller must retry
    }
};
```

Rate limiting parameters are set by the host at program launch based on the number of
active GSRP cores and the target per-core bandwidth:

$$R_{\text{per-core}} = \frac{BW_{\text{link}} \times N_{\text{links}}}{N_{\text{active GSRP cores}}}$$

For 140 active BH cores with 4 links at 12.5 GB/s:

$$R_{\text{per-core}} = \frac{50\ \text{GB/s}}{140} \approx 357\ \text{MB/s}$$

**L1 cost:** 16 bytes per core for the rate limiter struct.

### Strategy 3: Destination-Aware Traffic Shaping

Not all destinations are equally congested. A destination-aware traffic shaper tracks
per-destination congestion state using the credit counters from Section 7.2.4 and
preferentially delays traffic to congested destinations:

```
Per-destination congestion state:
  dst_chip_0:  credits=6/8   -> GREEN  (send immediately)
  dst_chip_1:  credits=2/8   -> YELLOW (delay non-critical traffic)
  dst_chip_2:  credits=0/8   -> RED    (block all traffic)
```

This adaptive approach prevents head-of-line blocking: traffic to uncongested destinations
flows freely even when one destination is overwhelmed.

| Strategy | Mechanism | Overhead | Burst tolerance |
|----------|-----------|:--------:|:---------------:|
| Token bucket | Per-core credit counter | ~3 cycles/check | Configurable burst size |
| Leaky bucket | Fixed drain rate | ~3 cycles/check | Zero burst (strict) |
| Windowed admission | Count writes per epoch | ~5 cycles/check | Epoch-level burst |

> **Recommendation:** Token bucket rate limiting with a burst allowance of 2x the average
> rate. This permits short bursts (common in ML workloads with alternating compute and
> communication phases) while preventing sustained overload. MUX-based aggregation should be
> the default for GSRP-enabled systems; rate limiting is a complementary defense for MUX-free
> configurations.

---

## 7.2.7 Congestion Detection and Signaling

### Congestion Score Metric

The simplest congestion signal is the rate at which credits are consumed versus replenished:

```math
\texttt{congestion\_score}(d) = \frac{\texttt{credits\_consumed}(d, \Delta t)}{\texttt{credits\_replenished}(d, \Delta t)}
```

When the congestion score exceeds 1.0, the destination is receiving traffic faster than
it can process -- indicating congestion.

### ECN-Style Congestion Notification

Intermediate EDM routers can detect congestion (buffer occupancy exceeding a threshold)
and set a congestion bit in forwarded packet headers. The destination includes this bit in
credit-update responses. The source reduces its injection rate upon receiving a
congested-credit update:

```
Congestion Detection at Intermediate EDM:

if (buffer_occupancy > HIGH_WATERMARK) {
    packet_header->flags |= GSRP_CONGESTION_BIT;
}
```

The source-side response:

```
On receiving ack with GSRP_CONGESTION_BIT set:
  1. Halve the token bucket max_tokens for that destination
  2. Set backoff timer (exponential backoff: 2^attempt * 100 ns)
  3. Gradually restore max_tokens when congestion clears
```

This is analogous to TCP's ECN mechanism and provides earlier congestion notification than
waiting for credit depletion.

### Congestion Signal Hierarchy

```
Level 0 (ERISC local):  Buffer occupancy > 75%   -> slow-start injection
Level 1 (Per-link):     Credit depletion rate     -> reduce per-dest rate limit
Level 2 (Per-chip):     Multiple links congested  -> activate global throttle
Level 3 (Cross-chip):   Fabric telemetry signal   -> host-side route rebalancing
```

---

## 7.2.8 End-to-End Flow Control

### When Hop-by-Hop Suffices

For most GSRP traffic, hop-by-hop flow control is adequate because:

1. **Backpressure propagation is fast.** Credit depletion at any hop blocks the upstream
   sender within one round-trip Ethernet latency (~1--2 $\mu$s). For a 4-hop path, full
   backpressure reaches the source in ~4--8 $\mu$s.
2. **MUX aggregation limits injection rate.** The MUX serializes traffic from multiple
   cores, preventing instantaneous incast.
3. **ML traffic is phase-structured.** Even without explicit BSP, ML workloads alternate
   between compute and communication phases, providing natural traffic drains.

### When End-to-End Is Needed

End-to-end flow control becomes necessary when:

- **Multi-hop paths share intermediate links** with different endpoints, creating head-of-line
  blocking. Traffic from chip 0 to chip 7 blocks traffic from chip 3 to chip 5 if they share
  an intermediate link.
- **Read response traffic** competes with write traffic on the return path, creating
  potential livelock (Section 7.3).
- **Sustained many-to-one incast** at a single destination chip overwhelms the local
  receiver capacity.

### Proposed End-to-End Credit Protocol

```
Source Core                    Destination ERISC
    |                                |
    |-- gsrp_write (with E2E tag) -->|
    |                                |
    |                                |-- Delivers to target core
    |                                |
    |<-- E2E_ACK (atomic inc) -------|
    |                                |
    Source tracks:
      e2e_outstanding_writes++  (on send)
      e2e_outstanding_writes--  (on ACK)
      Block if e2e_outstanding_writes >= E2E_WINDOW_SIZE
```

| Parameter | Value | Rationale |
|-----------|:-----:|-----------|
| `E2E_WINDOW_SIZE` | 4--8 | Allows pipelining without unbounded queue growth |
| ACK mechanism | `NocUnicastAtomicIncCommandHeader` | Reuses existing atomic inc packet type |
| ACK overhead per write | ~1 additional fabric packet (16 B) | ~0.4% BW overhead for 4 KB writes |

### Buffer Sizing Derivation

To avoid stalling on credit round-trips, the receive buffer must be large enough to absorb
a full round-trip worth of data:

```math
N_{\text{slots}} \geq \frac{BW_{\text{injection}} \times T_{\text{e2e credit}}}{\texttt{slot\_size}}
```

For 1.25 GB/s injection rate, 20 $\mu$s round-trip, and 4 KB slots:

$$N_{\text{slots}} \geq \frac{1.25 \times 10^9 \times 20 \times 10^{-6}}{4096} \approx 6\ \text{slots}$$

Eight slots (32 KB) provide sufficient buffering with margin. This is structurally
identical to the `RemoteCircularBuffer` protocol (Chapter 2, Section 4), extended across
chip boundaries. The fabric's `NocUnicastAtomicIncCommandHeader` provides the primitive for
cross-chip credit signaling.

> **Recommendation:** Implement end-to-end flow control only for the read path (Phase 2)
> and for writes to bounded receive buffers. The write path's fire-and-forget semantics make
> hop-by-hop flow control sufficient for most cases. For reads, the end-to-end credit
> protocol prevents response traffic from causing unbounded backpressure propagation.

---

## 7.2.9 `FabricReliabilityMode` as a Resilience Model

The `FabricReliabilityMode` enum (Chapter 3, Section 6) provides three modes:

```cpp
// tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp
enum class FabricReliabilityMode : uint32_t {
    STRICT_SYSTEM_HEALTH_SETUP_MODE = 0,
    RELAXED_SYSTEM_HEALTH_SETUP_MODE = 1,
    DYNAMIC_RECONFIGURATION_SETUP_MODE = 2,  // Placeholder
};
```

| Mode | Behavior | GSRP Relevance |
|------|----------|:--------------:|
| `STRICT` | All links must be healthy at init | Default for GSRP |
| `RELAXED` | Missing links accepted; reduced routing | Graceful degradation |
| `DYNAMIC_RECONFIGURATION` | Hot-swap failed links at runtime | Production GSRP target |

For GSRP, `DYNAMIC_RECONFIGURATION` is the long-term target because it enables:

1. **Route cache invalidation** when a link fails: the runtime re-computes routes and
   pushes updated translation tables to affected cores.
2. **Transparent failover**: the GSRP shim retries failed writes on an alternate routing
   plane without kernel awareness.
3. **Global address space preservation**: link failure removes capacity but does not
   invalidate the address space.

The route cache invalidation protocol:

```
1. Link failure detected by ERISC (Ethernet timeout > 100 us)
2. ERISC sends RouterCommand::RETRAIN to peer
3. If retrain fails, ERISC notifies ControlPlane via host interrupt
4. ControlPlane recomputes routing tables
5. Host pushes updated gsrp_chip_route tables to all Tensix L1
6. Host increments gsrp_route_cache_epoch counter
7. Tensix cores detect epoch change, invalidate route cache
```

> **Key Insight:** GSRP inherits TT-Fabric's reliability model. For Phase 1, treating
> writes as best-effort with epoch-barrier-based revalidation is sufficient. Full reliable
> delivery (with retransmission) should be deferred to Phase 2, building on the
> `DYNAMIC_RECONFIGURATION` infrastructure.

---

## 7.2.10 GSRP Congestion Telemetry

### Existing Telemetry Infrastructure

The `FabricTelemetry` infrastructure (Chapter 3, Sections 4--5) provides per-ERISC
bandwidth counters:

```cpp
// tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp
struct LowResolutionBandwidthTelemetry {
    RiscTimestamp timestamp_start;
    RiscTimestamp timestamp_end;
    uint64_t num_words_sent;
    uint64_t num_packets_sent;
};
```

### GSRP-Specific Counters

Extend telemetry with GSRP-specific metrics at the per-Tensix level:

```cpp
// Proposed GSRP telemetry extension (per-core)
struct GsrpTelemetryCounters {
    uint32_t gsrp_writes_injected;     // Total GSRP writes from this core
    uint32_t gsrp_writes_local;        // Writes resolved as local (fast path)
    uint32_t gsrp_writes_remote;       // Writes requiring fabric transit
    uint32_t gsrp_credit_stalls;       // Times blocked waiting for credits
    uint32_t gsrp_route_cache_hits;    // Route cache hits
    uint32_t gsrp_route_cache_misses;  // Route cache misses (table lookup)
    uint32_t gsrp_e2e_window_stalls;   // Times blocked on E2E window
};
// Size: 28 bytes per core. For 140 BH cores: 3,920 bytes total per chip.
```

These counters enable the host to detect congestion hotspots, tune rate limiters, identify
workloads where GSRP adds excessive overhead, and feed back into the destination-aware
traffic shaper (Section 7.2.6 Strategy 3). They integrate with the existing
`FabricTelemetrySnapshot` infrastructure and can be read by the host via the
`FabricTelemetryReader`.

---

## 7.2.11 The Three-Layer Flow Control Model

The complete flow control architecture for GSRP operates at three layers:

```
+-----------------------------------------------+
| Layer 3: End-to-End Admission Control          |
| - Per-destination credit table (64 B/core)     |
| - Credit update via inline-write packets       |
| - Congestion score tracking                    |
| - E2E window for bounded destinations          |
+-----------------------------------------------+
        |
+-----------------------------------------------+
| Layer 2: Hop-by-Hop Credit Flow               |
| - Existing EDM credit protocol (Ch 3, Sec 6)  |
| - Stream register or counter-based credits     |
| - Buffer pointer management (wrap_increment)   |
| - Dynamic credit allocation (elastic pool)     |
+-----------------------------------------------+
        |
+-----------------------------------------------+
| Layer 1: Worker-to-EDM Injection Control       |
| - wait_for_empty_write_slot() backpressure     |
| - Per-core rate limiter (token bucket)         |
| - VC selection (VC0 high-priority / VC1 bulk)  |
| - MUX aggregation with coalescing              |
+-----------------------------------------------+
```

| Layer | Scope | Latency | Accuracy | L1 cost |
|-------|-------|:-------:|:--------:|:-------:|
| Layer 1 | Single hop | Immediate | High (exact slot count) | 16 B (rate limiter) |
| Layer 2 | Per-hop chain | ~1 hop latency | High (credit-based) | 0 B (existing) |
| Layer 3 | End-to-end | ~2x path latency | Approximate (sampled) | 64 B (credit table) |

Layer 1 and Layer 2 are **exact** (based on credit counts). Layer 3 is **approximate**
(based on periodic credit updates). The approximation is acceptable because Layer 2
guarantees no buffer overflow regardless of Layer 3's accuracy -- Layer 3 merely improves
utilization by preventing unnecessary injection.

### Bandwidth Allocation Under Mixed Traffic

When GSRP and explicit CCL traffic coexist, bandwidth must be allocated fairly:

| Strategy | Description | GSRP Throughput | CCL Throughput | Recommendation |
|----------|-------------|:---------------:|:--------------:|:--------------:|
| Static partition | Reserve 25% for GSRP | Guaranteed floor | 75% guaranteed | **Phase 1** |
| Weighted fair queue | Weight 1:3 (GSRP:CCL) | Proportional | Proportional | Phase 2 |
| CCL-preemptive | CCL has priority | Unpredictable | Maximum | Not recommended |

> **Recommendation:** Static partition for Phase 1 (simplest implementation via VC
> separation). Weighted fair queuing for Phase 2 (requires firmware modification to the
> EDM scheduling loop).

---

## GSRP Implications

The flow control analysis reveals that the existing hop-by-hop credit system is sound for
GSRP's write path but must be extended in three areas: (1) dynamic credit allocation to
handle non-uniform traffic loads across routing planes, leveraging the elastic channel stubs
at Issue #26311, (2) per-destination credit partitioning at MUX cores to prevent
single-destination starvation, and (3) priority queuing via VC separation to protect
latency-sensitive synchronization messages from head-of-line blocking by bulk transfers.
End-to-end flow control via a sliding window adds bounded congestion guarantees at the cost
of ~0.4% bandwidth overhead per write. Token-bucket rate limiting at the source provides a
first line of defense against incast, with the per-core rate derived from the aggregate link
bandwidth divided by the number of active GSRP cores (~357 MB/s on 140-core BH). MUX-based
aggregation with write coalescing is the highest-impact congestion mitigation strategy,
improving small-packet efficiency from ~47% to ~73% and reducing per-ERISC fan-in from
10:1 to manageable levels. The `FabricReliabilityMode::DYNAMIC_RECONFIGURATION` mode,
combined with route cache epoch tracking, provides the resilience foundation for production
GSRP deployment. The total flow control L1 overhead is ~108 bytes per core (64 B credit
table + 16 B rate limiter + 28 B telemetry counters), well within the 16 KB budget.

---

## Key Takeaways

- **Existing hop-by-hop credit flow control is sufficient for GSRP writes in most cases.**
  Backpressure propagates within ~4--8 $\mu$s across a 4-hop path, and MUX aggregation
  naturally limits injection rate. End-to-end credits are needed only for sustained
  many-to-one incast or multi-hop head-of-line blocking scenarios.

- **Priority queuing using existing per-VC channel separation reduces worst-case
  small-message latency by ~50x** (from ~5.2 $\mu$s to ~0.1 $\mu$s behind a 64 KB bulk
  transfer) at a cost of ~3--4% throughput overhead for bulk traffic. The VC-based approach
  avoids the architectural conflict of mixing GSRP injection with S1 forwarding traffic.

- **MUX aggregation with write coalescing is the highest-impact congestion mitigation
  strategy**, improving small-packet efficiency from ~47% to ~73% and serializing bursty
  per-core traffic into a steady stream at the EDM sender channel.

- **Token-bucket rate limiting at the source provides first-line congestion defense**, with
  per-core rates of ~357 MB/s on a 140-core BH chip with 4 links per direction.

- **GSRP-specific telemetry counters (28 bytes per core) enable congestion diagnosis** by
  tracking credit stalls, route cache performance, and end-to-end window utilization.

## Source Code References

| File | Key Symbols |
|------|-------------|
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_router_flow_control.hpp` | `ReceiverChannelResponseCreditSender` (stream-register and counter variants), `SenderChannelFromReceiverCredits`, compile-time selection via `std::conditional_t` |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp` | `BufferIndex`, `BufferPtr`, `wrap_increment`, `distance_behind`, `normalize_ptr` |
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp` | `StaticSizedSenderEthChannel`, `ElasticSenderEthChannel` (stub, Issue #26311), `NUM_BUFFERS` template parameter |
| `tt_metal/fabric/erisc_datamover_builder.hpp` | `FabricEriscDatamoverConfig`, `num_used_sender_channels_per_vc`, `channel_buffer_size_bytes` |
| `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp` | `WorkerToFabricEdmSenderImpl::edm_has_space_for_packet()`, `wait_for_empty_write_slot()` |
| `tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp` | `LowResolutionBandwidthTelemetry`, `record_packet_send`, `open_perf_recording_window` |
| `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp` | `FabricReliabilityMode` (STRICT, RELAXED, DYNAMIC_RECONFIGURATION) |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | Router main loop, context switch interval, `MAX_DOWNSTREAM_EDM_COUNT` |
| `tt_metal/fabric/fabric_tensix_builder.hpp` | `FabricTensixConfig::MUX`, MUX core configuration |
| `tt_metal/fabric/fabric_edm_packet_header.hpp` | `NocUnicastScatterCommandHeader`, `NocUnicastAtomicIncCommandHeader`, `NocSendType::NOC_UNICAST_INLINE_WRITE` |

---

**Previous:** [`01_transparent_packet_injection.md`](./01_transparent_packet_injection.md) | **Next:** [`03_deadlock_avoidance_strategies.md`](./03_deadlock_avoidance_strategies.md)
