# Chapter 7, File 3 -- Multi-Core Replay and NOC Coordination

> **Scope.** This file addresses the most complex replay scenario:
> debugging kernels that span multiple Tensix cores communicating via
> the Network-on-Chip (NOC).  It covers when multi-core replay is needed,
> three multi-core replay strategies with scored alternatives, NOC
> transaction modeling, intra-Tensix 5-RISC coordination, dispatch core
> emulation, practical scoping guidance, and implementation priorities.
> All designs are **PROPOSED** unless explicitly noted as existing
> infrastructure.
>
> All source citations reference commit `621b949`.

**Cross-References:**
- Ch 2 (state inventory: L1 memory, DRAM tile data, semaphores, CB state)
- Ch 4 (5-RISC execution model, dispatch protocol, CB synchronization, inter-core NOC)
- Ch 5 File 1 (NOC logging: `NocWriteEvent`/`NocReadEvent` from `noc_debugging.hpp`)
- Ch 5 File 3 (E-taxonomy: NOC as E-Opaque, three-tier strategy)
- Ch 6 File 2 (capture modes: single-core trigger, multi-core burst)
- Ch 6 File 3 (`.ttksnap` FlatBuffer schema, per-core `TensixCoreSnapshot`)
- Ch 8 (implementation phases -- forward reference)

---

## 1. When Multi-Core Replay Is Needed

Most LLK kernel bugs can be diagnosed with single-core replay (File 01
and File 02).  Multi-core replay becomes necessary only for inter-core
communication bugs:

| Bug Class | Example | Multi-Core? |
|-----------|---------|:-----------:|
| Wrong matmul result in TRISC1 | Tile addressing error | No |
| CB underflow on single core | Pack reads before unpack writes | No (intra-Tensix) |
| Multicast SRCB data mismatch | BRISC on (0,0) multicasts, wrong data at (0,7) | **Yes** |
| All-reduce hang | `noc_semaphore_inc` never reaches expected count | **Yes** |
| CCL ring deadlock | Ethernet + Tensix circular dependency | **Yes** (+ Ethernet) |
| DRAM read returns stale data | NCRISC reads before prior write completes | **Yes** |

The fundamental architectural property that simplifies most debugging:
**TRISCs never issue NOC transactions** (Ch 4, File 3).  All data
consumed by unpack/math/pack is in L1 CBs at kernel launch time, and the
snapshot captures this data.  Single-TRISC or intra-Tensix replay with
pre-captured L1 contents is functionally complete for compute debugging.

---

## 2. Multi-Core Replay Strategy Alternatives

### 2.1 Candidates

| Strategy | Description | Capture Req. | Replay Complexity |
|----------|-------------|-------------|-------------------|
| **A: Synchronized lockstep** | All cores in lockstep; NOC transactions at cycle boundaries | Full L1 dump of all cores | Very high |
| **B: Trace-driven replay** | NOC transactions recorded during capture; replayed deterministically | NOC event log + L1 dumps | High |
| **C: Independent replay** | Each core replayed with pre-captured input data; NOC stubbed | Per-core L1 dumps | Low |

### 2.2 Strategy A: Synchronized Lockstep

All participating Tensix cores are instantiated as emulator instances
sharing a unified memory model.  A global scheduler advances all cores
by one instruction at a time, delivering NOC transactions at
deterministic points.

**Pros:** Deterministic; all inter-core interactions faithfully modeled;
can catch race conditions and ordering bugs.

**Cons:** For a 64-core matmul, this means 320 Spike instances
synchronized per instruction -- approximately 0.01-0.1 MIPS aggregate
throughput.  NOC model must handle multicast, completion semantics,
and atomic semaphore operations.

### 2.3 Strategy B: Trace-Driven Replay

During capture, the NOC logging infrastructure records all NOC
transactions.  The existing event structures from
`tt_metal/impl/debug/noc_debugging.hpp` (lines 23-94) provide the
event format.  The actual struct layouts are:

```cpp
// noc_debugging.hpp -- Actual NOC event structures (NOT fabricated)
struct NocWriteEvent {
    uint32_t src_addr;         // Source L1 address
    uint32_t dst_addr;         // Destination L1 address (uint32_t, not uint64_t)
    uint32_t num_bytes;
    uint32_t counter_snapshot; // nonposted/posted write_reqs_sent
    int8_t src_x, src_y;      // Source core (int8_t, not uint32_t)
    int8_t dst_x, dst_y;      // Destination core
    bool posted;               // Posted vs non-posted write
    uint8_t noc;               // NOC_0 or NOC_1
    bool is_semaphore;         // Whether this is a semaphore operation
    bool is_mcast;             // Whether this is a multicast write
    int8_t mcast_end_dst_x, mcast_end_dst_y;  // Multicast endpoint
};

struct NocReadEvent {
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t num_bytes;
    uint32_t counter_snapshot; // read_resps_recv
    int8_t src_x, src_y;
    int8_t dst_x, dst_y;
    uint8_t noc;
};
```

Additionally, barrier and flush events are recorded:

```cpp
struct NocReadBarrierEvent  { int8_t src_x, src_y; uint8_t noc; };
struct NocWriteBarrierEvent { int8_t src_x, src_y; bool posted; uint8_t noc; };
struct NocWriteFlushEvent   { int8_t src_x, src_y; bool posted; uint8_t noc; };
```

These are wrapped in a `NOCDebugEvent` variant type (lines 87-94):
```cpp
using NOCDebugEvent = std::variant<
    NocWriteEvent, NocReadEvent,
    NocReadBarrierEvent, NocWriteBarrierEvent, NocWriteFlushEvent,
    ScopedLockEvent, UnknownNocEvent>;
```

**Note:** These structs have no `timestamp` field.  Timing information
is not part of the event structure; sequence ordering is implicit in
the log order.  The `counter_snapshot` field captures the hardware
NOC completion counter at the time of the event.

During replay, each core executes independently.  When a core attempts
a NOC read, the replay engine consults the transaction log and returns
the captured data.

**Pros:** Deterministic from log; each core runs at full emulator speed.

**Cons:** Requires NOC logging enabled during original execution (off
by default, has overhead); log must be stored in `.ttksnap`.

### 2.4 Strategy C: Independent Replay

Each core is replayed independently with its own L1 snapshot.  NOC
reads return data from the local L1 snapshot.  NOC writes are logged
but not delivered.

This works because the `.ttksnap` snapshot captures L1 state at kernel
launch time.  For compute kernels (TRISC0/1/2), all input data is
already in L1 CBs at this point.

**Pros:** Simplest; each core replays independently at full speed; no
NOC model needed; sufficient for the vast majority of LLK scenarios.

**Cons:** Cannot detect inter-core data corruption or NOC ordering bugs.

### 2.5 Scoring Matrix

| Criterion | Weight | A: Lockstep | B: Trace-Driven | C: Independent |
|-----------|--------|-------------|----------------|----------------|
| Implementation effort | 4 | 1 (NOC model) | 2 (log replay) | 5 (trivial) |
| Execution speed | 3 | 1 (0.01 MIPS) | 4 (per-core speed) | 5 (per-core speed) |
| Inter-core bug detection | 3 | 5 (full fidelity) | 4 (from log) | 1 (none) |
| Capture overhead | 3 | 3 (L1 dumps only) | 2 (L1 + NOC log) | 5 (L1 dumps only) |
| Single-core debugging | 5 | 3 (must run all cores) | 4 (can isolate) | 5 (natural isolation) |
| GDB integration | 4 | 2 (multi-core GDB) | 3 (per-core GDB) | 5 (standard GDB) |
| **Weighted total** | | **59** | **70** | **98** |

### 2.6 Analysis

**Independent replay (C)** dominates for the primary use case.  TRISCs
never issue NOC transactions, so all input data is in L1 CBs at kernel
launch time.  Each core can be debugged independently with its own
snapshot and a standard single-core GDB session.

**Trace-driven replay (B)** is the right tool for the rare inter-core
debugging case (BRISC/NCRISC data movement kernels).  It requires NOC
logging during the original execution.

**Synchronized lockstep (A)** is deferred indefinitely -- on-hardware
replay (File 02) provides better fidelity for timing bugs with lower
implementation cost.

> **PROPOSED: Phased approach.** Phase 3: Independent replay (C) only.
> Phase 4: Trace-driven (B) for BRISC/NCRISC debugging when NOC logging
> is available.  Lockstep (A) deferred indefinitely.

---

## 3. Three NOC Emulation Modes

When multi-core emulation is needed, the `--noc-mode` flag selects
the fidelity level:

**Mode 1 -- Instant delivery (default):** NOC transactions complete
immediately.  Writes applied atomically to destination L1.  Catches
data correctness and address routing bugs.  No timing or ordering.

**Mode 2 -- Logged replay:** Requires `NocTransactionLog` in the
snapshot.  When a RISC issues a NOC operation, the model verifies
address/size against the next log entry and delivers the recorded
payload.  Catches instruction-sequence bugs.

**Mode 3 -- Synchronized emulation:** All RISCs advance in lock-step
quanta (configurable: 100-10000 instructions).  NOC writes queued and
delivered at quantum boundaries.  10-100x slower but can detect race
conditions and ordering bugs.

| Aspect | Instant | Logged | Synchronized |
|--------|:-------:|:------:|:------------:|
| Data correctness bugs | Yes | Partial | Yes |
| Address/routing bugs | Yes | Yes | Yes |
| Ordering/race bugs | No | No | Partial |
| Requires NOC log | No | **Yes** | No |
| Speed | ~10 MIPS | ~5 MIPS | ~0.1-1 MIPS |

---

## 4. Intra-Tensix 5-RISC Coordination During Replay

### 4.1 The Coordination State

Within a single Tensix core, the 5 RISCs coordinate via shared L1
semaphores and the `subordinate_sync` mailbox protocol (Ch 4, File 3):

| State | Location | Purpose |
|-------|---------|---------|
| `subordinate_sync.map[]` | `MEM_MAILBOX_BASE` | RUN_SYNC_MSG per RISC |
| `tiles_received[cb]` | CB sync registers | Tiles available in CB |
| `tiles_acked[cb]` | CB sync registers | Tiles consumed from CB |
| Hardware semaphores | `PC_BUF_BASE + offset` | Unpack/math/pack gates |

Sync constants (from `dev_msgs.h`):
- `RUN_SYNC_MSG_INIT = 0x40`
- `RUN_SYNC_MSG_GO = 0x80`
- `RUN_SYNC_MSG_DONE = 0x00`

### 4.2 Single-TRISC Simplification

For single-TRISC debugging (the most common scenario), the replay
engine bypasses all coordination:

1. Set `subordinate_sync.map[target_trisc]` to `RUN_SYNC_MSG_GO` (0x80)
2. Pre-populate `tiles_received[cb]` to indicate all expected tiles
   are available (unblock unpack waits)
3. Pre-populate hardware semaphores at values that allow the target
   TRISC to proceed without blocking
4. Stub mailbox reads to return immediately (see File 02, host stubs)

This is valid because the snapshot captures L1 state after BRISC has
written GO and NCRISC/BRISC have loaded CB data.

### 4.3 Full 5-RISC Replay: CB Synchronization Model

For debugging inter-RISC bugs, all 5 RISCs must be emulated.  The
Spike extension plugin intercepts writes to the CB sync register
addresses (`get_cb_tiles_received_ptr()` and `get_cb_tiles_acked_ptr()`
from `trisc.cc:96-105`) and routes them through a shared model:

```cpp
// PROPOSED: CB synchronization model for 5-RISC replay
struct CBSyncModel {
    std::atomic<uint32_t> tiles_received[NUM_CIRCULAR_BUFFERS];
    std::atomic<uint32_t> tiles_acked[NUM_CIRCULAR_BUFFERS];

    void push_tiles(uint32_t cb_id, uint32_t count) {
        tiles_received[cb_id].fetch_add(count);
    }

    void pop_tiles(uint32_t cb_id, uint32_t count) {
        tiles_acked[cb_id].fetch_add(count);
    }

    bool wait_for_tiles(uint32_t cb_id, uint32_t count) {
        return (tiles_received[cb_id].load() -
                tiles_acked[cb_id].load()) >= count;
    }
};
```

### 4.4 CB Sync Modeling Alternatives

| Strategy | Description | Fidelity | Effort |
|----------|-------------|----------|--------|
| **Pre-populated** | All CB counters set to max; no blocking | Low | 0 pw |
| **Shared atomic** | Atomic counters shared across emulator threads | Medium | 1 pw |
| **Full hardware model** | Model UNPACK/PACK_TILE_STATUS registers | High | 3 pw |

For Phase 3 (single-TRISC), pre-populated counters are sufficient.
For Phase 4 (5-RISC), shared atomic counters provide correctness.

### 4.5 Multi-Hart Spike for 5-RISC Replay

A full single-core replay maps each RISC to a Spike hart:

| Hart ID | RISC | Role |
|---------|------|------|
| 0 | BRISC | Master orchestrator, NOC access |
| 1 | NCRISC | Data movement, NOC access |
| 2 | TRISC0 | Unpack thread |
| 3 | TRISC1 | Math thread |
| 4 | TRISC2 | Pack thread |

```bash
# PROPOSED: Spike invocation for full 5-RISC single-core replay
spike \
  --isa=rv32imc \
  --priv=m \
  --hartids=0,1,2,3,4 \
  -m0x0:0x180000 \
  -m0xFFB00000:0x100000 \
  -m0xFFE00000:0x200000 \
  --rbb-port=9824 \
  replay_5risc.elf
```

In Spike multi-hart mode, the `subordinate_sync` mailbox is in shared
memory.  The synchronization protocol works naturally because all harts
share the same physical memory model.  However, timing differs: Spike
executes harts round-robin, while hardware runs them truly in parallel.

---

## 5. Dispatch Core Emulation During Replay

On hardware, the dispatch core writes binaries, launch messages, and go
signals to workers via NOC.  Rather than emulating the dispatch firmware,
the replay engine generates a scripted sequence from the snapshot:

1. For each core: write binaries, CB configs, runtime args, semaphores,
   `launch_msg_t` to L1
2. Barrier: all cores initialized
3. For each core: write `go_msg_t` with `signal = RUN_MSG_GO`

This avoids modeling the command queue, hugepage ring buffer, and
prefetcher/dispatcher pipeline (Ch 4, File 1).

**Completion detection:** Poll each core's `subordinate_sync` mailbox
until all bytes equal `RUN_SYNC_MSG_DONE` (0x00) or instruction count
exceeds threshold.  On timeout:

```
WARNING: Core (0,3) did not complete within 10M instructions.
  BRISC: PC=0x1234 (stuck in noc_async_read_barrier)
  TRISC1: PC=0xB3C0 (stuck in semaphore_wait at sem[2])
  Semaphore 2 value: 7 (expected: 8)
  Core (0,2) did not increment semaphore 2 on core (0,3).
```

---

## 6. Multi-Core CLI and GDB Integration

### 6.1 Multi-Core CLI Options

```
tt-kernel-replay emulate <snapshot.ttksnap> [multi-core options]
    --cores <spec>      "all", "0,0-0,7" (range), "0,0;0,1" (list)
    --noc-mode <mode>   instant | logged | sync
    --quantum <N>       Instructions per sync step (sync mode, default: 1000)
    --focus-core <x,y>  Primary core for GDB attach
    --focus-risc <name> Primary RISC on focus core (default: trisc1)
```

### 6.2 Port Assignment and GDB Multi-Inferior

Port assignment formula: `base + (y * grid_width + x) * 5 + risc_index`.

```
(gdb) add-inferior
(gdb) inferior 1
(gdb) target remote :1234      # Core (0,0) TRISC1
(gdb) inferior 2
(gdb) target remote :1239      # Core (0,1) TRISC1
(gdb) set scheduler-locking step  # Step one core at a time
```

---

## 7. PROPOSED: Session -- Debugging a Multi-Core Hang

```
$ tt-kernel-replay emulate allreduce_4core.ttksnap \
    --cores all --noc-mode sync --quantum 500 \
    --focus-core 0,3 --focus-risc brisc --gdb-port 2000

tt-kernel-replay: 4-core synchronized emulation (20 Spike instances)
tt-kernel-replay: WARNING: Core (0,3) BRISC stuck at PC=0x14A8 after 5M insns
  Polling sem 0 at 0x1FFF0, value=2, waiting for >=3
  Core (0,2) completed but did NOT increment sem 0 on core (0,3)
tt-kernel-replay: GDB port 2004 ready

$ riscv-tt-elf-gdb core_0_3_brisc.elf -ex "target remote :2004"
(gdb) bt
#0  noc_semaphore_wait (sem_addr=0x1fff0, target=3) at noc_api.h:245
#1  reduce_scatter_worker (args=0x20000) at reduce_scatter.cpp:87
(gdb) print target
$1 = 3
# Sem expects 3 increments from cores 0,1,2 but only got 2.
```

---

## 8. Practical Scoping Guidance

### 8.1 The 90/10 Rule

Based on the analysis across Files 1-3, approximately 90% of LLK
kernel debugging scenarios can be handled by single-core replay:

| Scenario | Frequency | Replay Target |
|----------|-----------|---------------|
| Math kernel bug (wrong result) | ~40% | Host model or emulator, TRISC1 |
| Unpack configuration error | ~20% | Emulator, TRISC0 |
| Pack output format bug | ~15% | Emulator, TRISC2 |
| CB pointer/size mismatch | ~10% | Emulator, any TRISC |
| Inter-RISC sync bug | ~5% | Emulator, 5-RISC |
| Inter-core NOC bug | ~5% | On-hardware or trace-driven |
| Timing/race condition | ~5% | On-hardware only |

### 8.2 Escalation Path

1. **Inspect** NOC transaction log to identify which cores communicated
   with the failing core
2. **Replay** with only the failing core + its data sources (2-4 cores)
3. **Use logged mode** to verify transaction sequences
4. **Escalate** to synchronized mode only if logged mode shows correct
   sequences but the bug persists

### 8.3 What Cannot Be Faithfully Reproduced

Even with synchronized multi-core emulation, the following remain
E-Opaque (Ch 5, File 3):

- Cycle-accurate NOC latency
- L1 bank contention from simultaneous multi-RISC access
- Ethernet core behavior
- DRAM controller arbitration
- Quasar NEO cluster synchronization (4-engine x 4-processor model)

For these, use on-hardware replay (File 02) or the RTL simulator
(Ch 5, File 3, Section 6).

---

## 9. Snapshot Consistency Across Cores

Multi-core snapshots must be captured at a consistent point.  If core
(0,0) is captured mid-NOC-transfer and core (0,1) is captured after the
transfer completes, the snapshot contains inconsistent state.

**Mitigation:** The interceptor (Ch 6, File 2) captures state after
BRISC completes Phase 4 (CB loading) but before Phase 5 (TRISC launch)
of the execution cycle (Ch 4, File 3).  At this point:

1. All CBs are configured and `tiles_received`/`tiles_acked` are zeroed
2. Config registers are fully programmed
3. Runtime arguments are written
4. Input tiles are DMA'd into L1

This is a consistent snapshot point: no TRISC has begun execution.
For multi-core consistency, all cores must be captured at this same
phase boundary, which requires the dispatch barrier between configuration
and execution to serve as the capture synchronization point.

---

## 10. Scaling Considerations

### 10.1 Snapshot Size vs. Core Count

| Configuration | Cores | Snapshot Size (LZ4) |
|--------------|-------|-------------------|
| Single TRISC | 1 | ~150-200 KB |
| Full Tensix (5 RISC) | 1 | ~200-300 KB |
| 8x8 matmul | 64 | ~10-15 MB |
| Full WH grid | 72 | ~12-17 MB |

### 10.2 Emulator Instance Scaling

| Strategy | Instances | Memory per | Total Memory |
|----------|-----------|-----------|--------------|
| Single TRISC | 1 | ~10 MB | ~10 MB |
| Full Tensix | 5 | ~10 MB | ~50 MB |
| 8x8 independent | 320 | ~10 MB | ~3.2 GB |

For large scenarios, scope narrowing to 2-4 interacting cores
is critical before attempting multi-core replay.

---

## 11. Implementation Priorities

| Priority | Capability | Effort | Value |
|----------|-----------|--------|-------|
| P0 | Single-TRISC emulator replay (File 01) | ~5 pw | Covers 70%+ of LLK debugging |
| P1 | Host functional model replay (File 02, Part B) | ~4 pw | Fast iteration, no hardware |
| P2 | On-hardware single-RISC replay (File 02, Part A) | ~4 pw | Exact fidelity when needed |
| P3 | Intra-Tensix multi-RISC emulation | ~2 pw | CB deadlock debugging |
| P4 | NOC transaction recording in interceptor | ~2 pw | Enables P5 |
| P5 | Multi-core emulation with NOC log replay | ~4 pw | Cross-core debugging |
| P6 | Multi-core emulation with synchronized NOC model | ~6 pw | Full multi-core debug |

Total: ~27 person-weeks for the complete replay debugger stack.  P0-P2
(~13 weeks) cover the vast majority of practical debugging scenarios.
This aligns with Ch 8 Phase 3-4 estimates.

---

## Key Source Files

| File | Path (relative to TT-Metal root) | Lines | Relevance |
|------|----------------------------------|-------|-----------|
| `noc_debugging.hpp` | `tt_metal/impl/debug/noc_debugging.hpp` | 23-94 | `NocWriteEvent`, `NocReadEvent`, `NOCDebugEvent` variant |
| `noc_logging.hpp` | `tt_metal/impl/debug/noc_logging.hpp` | -- | NOC event capture infrastructure |
| `dataflow_api.h` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | 546, 822 | `noc_async_read`, `noc_async_write` |
| `dev_msgs.h` | `tt_metal/hw/inc/hostdev/dev_msgs.h` | 97, 179, 387 | `RUN_SYNC_MSG_DONE`, `go_msg_t`, `subordinate_sync` |
| `brisc.cc` | `tt_metal/hw/firmware/src/tt-1xx/brisc.cc` | 48-55 | Subordinate sync protocol, instrn_buf/pc_buf/mailbox arrays |
| `trisc.cc` | `tt_metal/hw/firmware/src/tt-1xx/trisc.cc` | 96-105 | `get_cb_tiles_received_ptr`, `get_cb_tiles_acked_ptr` |
| `tensix_functions.h` | `tt_metal/hw/inc/internal/tensix_functions.h` | 398-434, 606-687 | Semaphore init, CAS, breakpoint API |
| `dispatch.cpp` | `tt_metal/impl/program/dispatch.cpp` | -- | Dispatch command assembly |
| `ckernel_debug.h` | various | 102-155 | `dbg_thread_halt`/`dbg_thread_unhalt` inter-TRISC sync |

---

## Key Takeaways

- **Independent replay (Strategy C) is the clear winner** for multi-core
  scenarios, scoring 98/130 versus 70 for trace-driven and 59 for
  lockstep.  Its simplicity derives from the fundamental architectural
  property that TRISCs never issue NOC transactions -- all input data is
  in L1 CBs at kernel launch, and the snapshot captures this data.

- **The 90/10 rule guides phasing**: ~90% of LLK debugging scenarios
  require only single-TRISC replay with pre-captured L1 data (Phase 3,
  ~5 person-weeks).  Multi-core and 5-RISC replay (Phase 4-5) addresses
  the remaining ~10%.

- **NOC transaction recording is already partially implemented**: the
  `NocWriteEvent` / `NocReadEvent` types in `noc_debugging.hpp` provide
  the event structure.  These use `int8_t` for core coordinates,
  `uint32_t` for addresses, and include fields like `posted`, `noc`,
  `is_semaphore`, `is_mcast`, and `counter_snapshot`.  The interceptor
  needs to serialize these into the `.ttksnap` format.

- **Intra-Tensix 5-RISC coordination uses a progressive fidelity model**:
  Phase 3 pre-populates CB counters and semaphores to unblock single-TRISC
  replay; Phase 4 adds shared atomic CB counters for 5-RISC replay;
  full hardware synchronization modeling is deferred unless validated
  as necessary.

- **Snapshot consistency across cores requires capture at the Phase 4/5
  boundary** of the execution cycle, where dispatch has configured all
  cores but no TRISC has begun execution.

- **Three NOC emulation modes** provide an escalation path: instant
  (data/address bugs), logged (sequence verification), and synchronized
  (ordering/race bugs).  Start with instant, escalate only when needed.

- **Multi-core emulation at scale (64 cores, 320 Spike instances) is
  tractable but expensive** (~3.2 GB memory).  Developers should narrow
  to the minimum interacting core set (2-4 cores) using the NOC
  transaction log before attempting multi-core replay.
