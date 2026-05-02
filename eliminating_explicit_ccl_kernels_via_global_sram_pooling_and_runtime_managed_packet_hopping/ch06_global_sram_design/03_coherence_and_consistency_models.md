# 6.3 Coherence and Consistency Models

## Context

Section 6.2 defined a 64-bit `GlobalAddress` encoding that uniquely names every byte of L1
SRAM across a multi-chip mesh. But addressability alone is insufficient -- the system must
also specify what happens when multiple cores access the same global address concurrently.
The key simplifying property of Tenstorrent hardware is that **L1 is scratchpad memory, not
a cache.** There are no hardware caches on the data path, no cache lines to invalidate, and
no snooping or directory protocols. This eliminates the need for coherence entirely and
reduces the problem to **memory consistency**: the order in which writes by one core become
visible to reads on other cores.

The reader should be familiar with `GlobalSemaphore` and `NocUnicastAtomicIncCommandHeader`
from Chapter 2 (Section 4), the BSP epoch model from Chapter 5 (Section 3), and the
credit-based flow control model from Chapter 3 (Section 6).

---

## 6.3.1 Why Coherence Is Not Needed

Cache coherence protocols solve a specific problem: when multiple processors cache copies of
the same memory location, writes by one processor must be made visible to all others. On
Tenstorrent hardware, this problem does not arise:

| Cache coherence precondition | Present on TT hardware? | Explanation |
|-----------------------------|-----------------------|-------------|
| Multiple copies in different caches | **No** | L1 is scratchpad; each byte exists in exactly one physical location |
| Hardware-managed replication | **No** | All data movement is explicit (NOC writes, fabric packets) |
| Transparent cache fills | **No** | No hardware prefetcher fills L1 automatically |
| Write-back / write-through policy | **N/A** | No caches to write back from |
| Directory or snooping infrastructure | **No** | NOC is point-to-point, not a coherence bus |

Since each byte of L1 exists in exactly one physical location (the core that owns it), there
are no stale copies to invalidate. When a remote core reads that byte via GSRP, it receives
a **private snapshot** -- subsequent writes to the original do not automatically propagate.
This is a feature, not a limitation: the GSRP operates without coherence traffic, avoiding
the 20--40% bandwidth tax that coherence protocols impose in NUMA systems.

> **Key Insight:** The absence of hardware caches on Tensix cores is the single most
> important architectural property for GSRP feasibility. It eliminates the scalability wall
> that limits coherent systems to ~8--16 nodes.

---

## 6.3.2 The Problem That Remains: Ordering and Visibility

While coherence is unnecessary, **consistency** must still be defined. Consider:

```
Core A (Chip 0):                    Core B (Chip 1):
  gsrp_write(addr_X, data_1)         val_X = gsrp_read(addr_X)
  gsrp_write(addr_flag, 1)           if (val_flag == 1):
                                        // Is val_X guaranteed to be data_1?
```

Without ordering guarantees, Core B might observe the flag write before the data write.
The consistency model must specify when writes become visible and in what order.

### Ordering Sources in TT Hardware

| Mechanism | Scope | Ordering guarantee | Reference |
|-----------|-------|-------------------|-----------|
| NOC posted writes | Intra-chip | FIFO per source-destination pair | Ch 2, Sec 3 |
| `noc_async_write_barrier()` | Intra-chip | All prior writes complete before return | `noc.h` |
| `NocUnicastAtomicIncCommandHeader` | Cross-chip | Atomic read-modify-write at target | Ch 3, Sec 3 |
| `GlobalSemaphore` | Cross-chip | Atomic increment visible to all readers | Ch 2, Sec 4 |
| EDM packet ordering | Cross-chip | FIFO per sender-channel pair | Ch 3, Sec 2 |

The critical observation: the fabric provides **per-channel FIFO ordering** but no
**total ordering** across channels or across different source-destination pairs.

---

## 6.3.3 Proposed Consistency Model: Release Consistency

The GSRP adopts **release consistency (RC)**, a well-studied model used in the Stanford
DASH multiprocessor and forming the basis for C++ `memory_order_acquire/release`.

### Definitions

| Term | GSRP definition |
|------|----------------|
| **Ordinary access** | `gsrp_write()` or `gsrp_read()` without ordering flags |
| **Release** | `gsrp_write_release()` or `gsrp_fence_release()` -- ensures all prior ordinary writes are visible before the release |
| **Acquire** | `gsrp_read_acquire()` or `gsrp_fence_acquire()` -- ensures the acquire completes before subsequent ordinary reads |
| **Fence** | `gsrp_fence()` -- full barrier; all prior operations complete before any subsequent operations |

### Implementation on TT Hardware

The RC primitives map directly to existing TT-Metal mechanisms:

| RC operation | TT-Metal implementation |
|-------------|------------------------|
| `gsrp_write()` (ordinary) | `fabric_unicast_noc_unicast_write()` via mesh API |
| `gsrp_write_release()` | Ordinary write + `NocUnicastAtomicIncCommandHeader` to flag |
| `gsrp_read_acquire()` | Spin on `GlobalSemaphore` + ordinary read |
| `gsrp_fence_release()` | `noc_async_write_barrier()` + atomic increment to remote semaphore |
| `gsrp_fence_acquire()` | Spin-wait on local semaphore + `noc_async_read_barrier()` |

The release-acquire pattern:

```cpp
// Producer (core A, chip 0)
gsrp_write(global_addr, local_src, size);       // Posted write -- returns immediately
gsrp_fence_release();                            // Drain all in-flight writes
noc_semaphore_inc(consumer_sem, 1, remote_noc);  // Signal consumer

// Consumer (core B, chip 3)
noc_semaphore_wait(local_sem, expected_value);   // Acquire: wait for signal
// Data is now guaranteed visible in local L1
process(local_target_addr);
```

---

## 6.3.4 Formal Ordering Rules

For completeness, the GSRP RC model is defined by six rules. Let $W_A(x, v)$ denote
core $A$ writing value $v$ to address $x$, and $R_B(x) \to v$ denote core $B$ reading
value $v$ from address $x$.

**Rule 1 (Program Order).** Operations from the same core execute in program order with
respect to local L1.

**Rule 2 (Release Ordering).** If core $A$ performs $W_A(x, v_1)$ followed by
$\text{Release}_A(f)$, then any core $B$ that performs $\text{Acquire}_B(f)$ and subsequently
$R_B(x)$ will observe $v_1$ (or a later value).

**Rule 3 (Per-Channel FIFO).** Two writes from the same core to the same remote chip through
the same fabric routing plane arrive in issue order.

**Rule 4 (No Cross-Channel Ordering).** Writes through different routing planes or to
different chips have no relative ordering guarantee without an explicit fence.

**Rule 5 (Epoch Barrier).** An epoch barrier is equivalent to a release by all cores followed
by an acquire by all cores. After a barrier, every core sees all writes from the preceding
epoch.

**Rule 6 (Atomics).** `NocUnicastAtomicIncCommandHeader` operations are sequentially
consistent with respect to the target address. An atomic increment is both a release and an
acquire at the target location.

These six rules are sufficient to prove correctness of SWMR, producer-consumer, and
epoch-based patterns. They are intentionally weaker than sequential consistency to allow
maximum pipelining of fabric transfers.

### Why Not Sequential Consistency?

SC requires that all cores observe all writes in the same total order. On a multi-chip
system with Ethernet latencies, enforcing SC would require every write to be acknowledged
before the next operation proceeds:

$$\text{SC overhead per write} \geq 2 \times h \times t_{\text{hop}} \approx 2 \times 4 \times 10\ \mu\text{s} = 80\ \mu\text{s}$$

This doubles latency and halves effective bandwidth -- unacceptable for bulk transfers.

### Why Not Eventual Consistency?

Eventual consistency provides no ordering guarantees at all. This is insufficient for ML
workloads where correctness depends on seeing all activations from the previous layer before
computing the next. The acquire-release pairs in RC provide the minimum necessary guarantee.

---

## 6.3.5 Memory Ownership Models

The consistency model defines ordering; ownership models define who may write. Without an
ownership model, multiple cores could simultaneously write to the same `GlobalAddress`,
creating data races that no consistency model can resolve.

### Pattern 1: Single-Writer Multiple-Reader (SWMR)

**Rule:** At any point in time, a given global address region is either being written by
exactly one core (the owner) or being read by any number of cores -- never both
simultaneously.

```
Timeline:
  Phase 1: Core A writes region R      (A is owner, no readers)
  Barrier: A issues gsrp_fence_release()
  Phase 2: Cores B, C, D read region R (A is not writing, B/C/D read freely)
  Barrier: B, C, D complete ... then A resumes writing
```

**Enforcement:** Translation table entries (Section 6.2.5) encode permissions. Writer's
entry has R/W; readers' entries have R-only. Debug builds validate at the GSRP API level.

**Use case:** Tensor-parallel all-gather (one chip owns its shard during computation;
others read at gather boundary), weight distribution, activation sharing.

### Pattern 2: Epoch-Based Ownership Transfer

**Rule:** Time is divided into epochs. Within an epoch, each global address region has a
fixed owner. At epoch boundaries, ownership can transfer. All remote writes from the
previous epoch are guaranteed visible after the barrier.

```
+----------------+     +------------------+     +------------------+
| Epoch N        |     | Barrier          |     | Epoch N+1        |
| Core A owns R1 | --> | Flush all writes | --> | Core B owns R1   |
| Core B owns R2 |     | Transfer R1: A->B|     | Core C owns R2   |
| Core C owns R3 |     | Transfer R2: B->C|     | Core A owns R3   |
+----------------+     +------------------+     +------------------+
```

This pattern subsumes SWMR and producer-consumer as special cases, and provides the
strongest deadlock-freedom guarantees (Section 6.3.7).

### Pattern 3: Producer-Consumer Ring

**Rule:** A fixed set of cores forms a ring. Each core writes to the next core's L1 and
reads from the previous core's L1. Writes and reads are pipelined via semaphore pairs.

```cpp
// Producer side (Core i writing to Core i+1)
gsrp_write(next_core_addr, local_data, chunk_size);
gsrp_fence_release();
noc_semaphore_inc(next_core_produce_sem, 1, next_core_noc);

// Wait for credit from next core (backpressure)
noc_semaphore_wait(my_consume_sem, expected_credits);

// Consumer side (Core i+1 reading from predecessor)
noc_semaphore_wait(my_produce_sem, expected_chunks);
// Data is now available; process it
process(local_target_addr, chunk_size);
noc_semaphore_inc(prev_core_consume_sem, 1, prev_core_noc);
```

This pattern maps directly to ring-based all-reduce and reduce-scatter. The existing CCL
ring implementation in TT-Metal uses this exact pattern with explicit packet construction.
Under the GSRP, the same algorithm uses `gsrp_write` instead, eliminating ~25 lines of
boilerplate per core.

---

## 6.3.6 BSP Epoch Barriers

Chapter 5 identified Graphcore's Bulk Synchronous Parallel model as a powerful
synchronization strategy. The GSRP adapts this concept.

### Epoch Structure

```
Time -->
|<--- Epoch k --->|<- Barrier ->|<--- Epoch k+1 --->|
| Local compute   | Drain all   | Local compute      |
| + GSRP writes   | in-flight   | + GSRP writes      |
| + GSRP reads    | transfers   | + GSRP reads       |
|                 | Reassign    |                    |
|                 | ownership   |                    |
```

### Barrier Implementation

The epoch barrier uses a hierarchical three-phase protocol:

```
Phase 1 (Intra-chip): Each core completes local writes via noc_async_write_barrier().
  Per-chip coordinator collects completion via local semaphore.

Phase 2 (Inter-chip): Coordinator cores signal root via NocUnicastAtomicIncCommandHeader
  on a GlobalSemaphore. Root waits until semaphore == num_chips.

Phase 3 (Broadcast): Root broadcasts "epoch complete" via fabric multicast.
  Each chip's coordinator broadcasts locally via NOC multicast.
```

### Barrier Overhead

| System size | Phase 1 | Phase 2 | Phase 3 | Total |
|------------|:-------:|:-------:|:-------:|:-----:|
| 8-chip BH | ~1 $\mu$s | ~3 hops $\times$ 5 $\mu$s = 15 $\mu$s | ~15 $\mu$s | ~31 $\mu$s |
| 32-chip BH | ~1 $\mu$s | ~7 hops $\times$ 5 $\mu$s = 35 $\mu$s | ~35 $\mu$s | ~71 $\mu$s |

For a typical transformer layer of ~2 ms, a 71 $\mu$s barrier is ~3.5% overhead -- within
the $< 5\%$ throughput target from Section 6.1.5.

### Minimum Epoch Duration

To keep barrier overhead below 5%:

$$t_{\text{epoch}} > 20 \times t_{\text{barrier}}$$

For the worst-case 32-chip estimate of ~280 $\mu$s (including write drain, from V3's
$4 h_{\max} \times t_{\text{hop}}$ formula):

$$t_{\text{epoch}} > 20 \times 280 = 5{,}600\ \mu\text{s} \approx 5.6\ \text{ms}$$

This corresponds to ~5.6 million Tensix cycles at 1 GHz -- sufficient for a full transformer
layer's forward pass on a large model.

### When to Use Epoch Barriers vs. Point-to-Point Sync

| Scenario | Recommended sync | Rationale |
|----------|:----------------:|-----------|
| All-reduce / all-gather | Epoch barrier | All chips participate; global consistency |
| Pipeline parallelism | Point-to-point release-acquire | Only 2 chips; barrier is overkill |
| MoE expert dispatch | Epoch barrier | Irregular all-to-all; barrier simplifies reasoning |
| KV-cache read | Point-to-point acquire | Single producer already completed |
| Activation checkpointing | Epoch barrier | Ownership transfer across many chips |

---

## 6.3.7 Deadlock Avoidance

The consistency model must not introduce new deadlock vectors beyond those already managed
by TT-Fabric (Chapter 3, Section 5).

**Risk 1: Circular dependency.** Core A writes to Core B's L1 while Core B writes to Core
A's L1. If both channels' buffer slots are full, neither write can complete.

**Mitigation:** EDM credit-based flow control (Chapter 3, Section 6) prevents this. Each
sender reserves a buffer slot before transmitting. If slots are exhausted, the sender blocks
at `wait_for_empty_write_slot()` without holding any resource the receiver needs, breaking
the circular dependency.

**Risk 2: Acquire spin-lock starvation.** Core B spins waiting for a semaphore that Core A
will increment, but Core A is blocked on a fabric send to Core B.

**Mitigation:** Separate the semaphore-signaling path from the bulk-data path. Semaphore
increments use `NocUnicastAtomicIncCommandHeader` (header-only, no payload, dedicated buffer
slot class). Bulk data uses payload slots. Since they draw from different resource pools, one
cannot starve the other.

**Epoch barriers provide structural deadlock freedom:** within an epoch, all communication
is unidirectional. Bidirectional exchange happens only across epoch boundaries, where the
barrier drains all in-flight traffic. This is the same insight that makes Graphcore's BSP
model deadlock-free (Chapter 5, Section 3).

---

## 6.3.8 Metadata Cost of Consistency Infrastructure

| Component | Per-Core Cost | Per-Chip Cost (140 cores) |
|-----------|:------------:|:------------------------:|
| Semaphore storage (8 semaphores x 4 B) | 32 B | 4.4 KB |
| Barrier counter (1 per epoch) | 4 B | 560 B |
| Epoch state (current epoch, phase) | 8 B | 1.1 KB |
| Permission bits (in translation table) | Included in TTE | (see Section 6.2) |
| **Total consistency overhead** | **~44 B** | **~6 KB** |

This is negligible compared to the translation table overhead (~4 KB per core from
Section 6.2) and well within the 16 KB per-core metadata budget.

---

## 6.3.9 Comparison with Alternative Consistency Models

| Consistency model | Ordering strength | GSRP suitability | Overhead |
|------------------|:-----------------:|:-----------------:|:--------:|
| Sequential consistency (SC) | Strongest | Poor -- serializes all remote accesses | Very high |
| Total store order (TSO) | Strong | Marginal -- too costly for Ethernet links | High |
| Release consistency (RC) | Moderate | **Best fit** -- matches TT hardware | Low |
| BSP epoch | Barrier-separated | Natural fit for Level C | Moderate |
| Eventual consistency | Weakest | Insufficient -- no ordering | Lowest |

---

## Key Takeaways

- **No cache coherence protocol is needed for the GSRP** because L1 is scratchpad memory
  with no hardware caches -- each byte exists in exactly one physical location, eliminating
  the stale-copy problem that coherence protocols solve.

- **Release consistency with six formal ordering rules provides the minimum guarantees**
  needed for ML communication patterns (SWMR, producer-consumer, reductions) while allowing
  maximum pipelining -- implemented entirely using existing `NocUnicastAtomicIncCommandHeader`
  and `GlobalSemaphore` primitives.

- **BSP-style epoch barriers provide structural deadlock freedom** at ~31 $\mu$s for 8-chip
  or ~71 $\mu$s (~3.5% overhead) for 32-chip systems, subsuming SWMR and producer-consumer
  as special cases and aligning naturally with transformer layer boundaries.

## GSRP Implications

The consistency model imposes two hard constraints on any GSRP implementation: (1) the
API must expose `gsrp_write_release()` and `gsrp_read_acquire()` as first-class operations
to implement RC semantics, and (2) any address translation or routing layer must preserve
per-channel FIFO ordering so that two writes to the same destination maintain their relative
order. The epoch-barrier approach (Graphcore-inspired, from Chapter 5) provides the simplest
correctness path: all cross-chip transfers within a layer are ordinary writes that become
visible atomically at the barrier, reducing fine-grained synchronization to a single
barrier per layer boundary.

---

**Previous:** [`02_address_space_design.md`](./02_address_space_design.md) | **Next:** [`04_candidate_architectures.md`](./04_candidate_architectures.md)
