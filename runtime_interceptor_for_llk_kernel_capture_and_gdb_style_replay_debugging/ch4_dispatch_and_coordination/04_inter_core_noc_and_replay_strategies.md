# 4.4 Inter-Core NOC Communication and Replay Strategies

## Overview

Tenstorrent chips use a Network-on-Chip (NOC) for all inter-core and core-to-DRAM
communication. This section analyzes NOC transaction patterns, proposes three replay
isolation levels with data volume estimates, specifies a capture file format, and
provides a bug-type-to-strategy mapping for the interceptor system.

> **Cross-reference:** Section 4.1 covers the dispatch pipeline that uses NOC
> multicast for launch messages. Section 4.3 details per-core coordination and the
> CB handshake. Ch2 (State Inventory) Section 2 catalogs runtime config state
> delivered by NOC. Ch3 (LightMetal Foundation) Section 2 identifies gaps that the
> replay strategies here address.

---

## 4.4.1 NOC Architecture and Transaction Types

Each Tensix core has two independent NOC instances (NOC0, NOC1). The dispatch system
uses a specific NOC assignment, and kernels use configurable NOC modes:

```cpp
// dev_msgs.h:119-123
enum noc_mode : uint8_t {
    DM_DEDICATED_NOC = 0,  // BRISC uses NOC_X, NCRISC uses NOC_Y
    DM_DYNAMIC_NOC = 1,    // Both processors share both NOCs
};
```

| Transaction Type | Usage | Typical Size |
|-----------------|-------|-------------|
| Unicast write | RTAs to individual cores | 64--512 B/core |
| Multicast write | Launch msgs, CB configs, go signal | 128--4096 B/mcast |
| Paged DRAM read | Binary fetch by prefetcher | 2 KB pages |
| Atomic increment | Completion notification to dispatch | 4 B |

---

## 4.4.2 NOC Operations by RISC

### BRISC (DM0)

| Operation | Purpose | Replay Strategy |
|-----------|---------|-----------------|
| `noc_async_read` | Read tiles from DRAM/remote L1 | **Record/replay** data |
| `noc_async_read_barrier` | Wait for reads | **No-op** (data present) |
| `noc_async_write` | Write tiles to DRAM/remote L1 | **Record** for verification |
| `noc_semaphore_inc/wait` | Inter-core sync | **Stub** or pre-populate |
| Remote CB setup | Initialize remote pointers | **Pre-configure** |

### NCRISC (DM1)

Same operation types as BRISC but typically on the opposite NOC. Usually the data
writer role.

### TRISC0/1/2

**No NOC access.** Communicate exclusively through stream scratch registers (CB
sync), Tensix config registers, and L1 memory. This makes compute-only replay
straightforward: no NOC stubbing needed.

### Dispatch Core

NOC multicast/unicast writes to deliver configs, binaries, go signals. These are
NOT user kernel operations and do not need replay. The interceptor captures their
*output* (data landing on workers), not the transactions themselves.

---

## 4.4.3 Dispatch NOC Transactions: Multicast vs. Unicast

For Tensix cores: `CQDispatchWritePackedMulticastSubCmd` delivers launch messages,
CB configs, and semaphores to entire core ranges simultaneously.

For Ethernet cores and per-core RTAs: unicast writes to each core individually.

**Data flow for a typical 8x8 matmul:**

| Command | Delivery | Total Data |
|---------|----------|------------|
| Launch messages | 1--4 mcast regions | 512 B -- 2 KB |
| CB configs | 1--4 mcast regions | 1 -- 4 KB |
| Unique RTAs | 64 unicast writes | 4 -- 16 KB |
| Kernel binaries | Paged from DRAM | 16 -- 64 KB |
| Go signal | 1 mcast | 64 B |

---

## 4.4.4 Multi-Device Dispatch and Worker Completion

`FDMeshCommandQueue::enqueue_mesh_workload()` iterates programs mapped to
`MeshCoordinateRange` regions. Devices not running a program still receive go
signals to maintain dispatch state (`write_go_signal_to_unused_sub_grids`).

Worker completion uses a 32-bit counter incremented by each completing worker.
The interceptor must record `expected_num_workers_completed` and `num_workers` for
replay synchronization.

---

## 4.4.5 NOC Dependencies in Kernel Execution

### Data Flow

- **Reader (BRISC):** NOC read DRAM -> L1 CB
- **Writer (NCRISC):** NOC write L1 CB -> DRAM
- **Compute (TRISC0-2):** No NOC; operate on L1 data only

### Remote Circular Buffers

One core writes directly to another's L1 via `noc_async_write`. Requires capture
of transferred data at sender and pre-loading at receiver.

### DRAM Banking

NOC addresses use bank-to-NOC-coordinate tables (`brisc.cc:83-86`):
`dram_bank_to_noc_xy`, `l1_bank_to_noc_xy`, `bank_to_dram_offset`,
`bank_to_l1_offset`. Must be captured for address translation during replay.

---

## 4.4.6 NOC Debugging Infrastructure

TT-Metal's `noc_debugging.hpp` records NOC events:

```cpp
struct NocWriteEvent {
    uint32_t src_addr, dst_addr, num_bytes, counter_snapshot;
    int8_t src_x, src_y, dst_x, dst_y;
    bool posted, is_semaphore, is_mcast;
    uint8_t noc;
    int8_t mcast_end_dst_x, mcast_end_dst_y;
};
```

**PROPOSED:** Extend with data payload capture for full replay:
```cpp
struct NocReadEventWithData : NocReadEvent {
    std::vector<uint8_t> data;  // actual bytes read
};
```

---

## 4.4.7 Replay Strategy: Three Isolation Levels

### Level 1: Single-Core Compute Isolation

**Captures:** Kernel binaries + launch message + CB configs + RTAs + input tiles.
**Replays:** TRISC0-2 with pre-loaded input data.
**NOC requirement:** None.
**Use case:** LLK compute correctness (unpack/math/pack pipeline).

| Data | Size |
|------|------|
| TRISC binaries (3 x ~8 KB) | ~24 KB |
| CB configs + RTAs + semaphores | ~1 KB |
| Input tile data | ~8 KB |
| **Total** | **~33 KB** |

**Implementation:**
1. Load binaries, CB configs, RTAs to L1
2. Pre-load tile data into input CBs
3. Pre-set sync registers: `tiles_received[input] = total`, `tiles_acked[output] = 0`
4. Launch via slow dispatch (MMIO)
5. Read output CBs for verification

### Level 2: Single-Core Full Isolation (NOC Stubs)

**Captures:** Level 1 + BRISC/NCRISC binaries + DRAM data + NOC transaction log.
**Replays:** All 5 RISCs with stubbed NOC operations.

**PROPOSED: NocReplayStub (queue-based):**

The stub uses a **queue** (not a map) for NOC read data, because the same address
may be read multiple times with different data at different execution points:

```cpp
class NocReplayStub {
    std::deque<NocReadRecord> read_queue_;       // temporal order
    std::vector<NocWriteRecord> write_log_;      // for verification
    std::unordered_map<uint64_t, std::atomic<uint32_t>> semaphores_;

public:
    void noc_async_read(uint64_t src_noc, uint32_t dst_local,
                        uint32_t size, uint8_t* l1) {
        auto& rec = read_queue_.front();
        if (rec.noc_addr != src_noc || rec.size != size)
            throw ReplayError("NOC read mismatch");
        memcpy(l1 + dst_local, rec.data.data(), size);
        read_queue_.pop_front();
    }
    void noc_async_write(uint32_t src_local, uint64_t dst_noc,
                         uint32_t size, const uint8_t* l1) {
        write_log_.push_back({dst_noc, size, {l1+src_local, l1+src_local+size}});
    }
    void noc_semaphore_inc(uint64_t addr, uint32_t val) {
        semaphores_[addr].fetch_add(val, std::memory_order_release);
    }
};
```

A map-based stub would fail for streaming reads from DRAM in a loop with
intervening writes from other cores changing the data.

**5-Thread Host Replay Model:**

Each RISC runs as a separate host thread with shared access to simulated L1:

```
Thread 0 (BRISC):   subordinate_sync master, NOC stubs, reader kernel
Thread 1 (NCRISC):  waits for BRISC signal, NOC stubs, writer kernel
Thread 2 (TRISC0):  waits for BRISC signal, unpack kernel
Thread 3 (TRISC1):  waits for BRISC signal, math kernel
Thread 4 (TRISC2):  waits for BRISC signal, pack kernel

Shared state:
  simulated_l1[L1_SIZE]            -- byte array
  subordinate_sync                 -- atomic uint32_t
  cb_tiles_received[NUM_CBS]       -- atomic uint16_t
  cb_tiles_acked[NUM_CBS]          -- atomic uint16_t
  cb_interface[NUM_CBS]            -- per-thread copies
```

**Additional capture requirements:**
- DRAM buffer addresses and contents referenced by the program
- Buffer allocation metadata (address, size, page size, bank mapping)
- For sharded buffers: shard-to-core mapping

**Data volume:** 1 MB -- 100 MB depending on tensor sizes.

### Level 3: Multi-Core Coordinated Replay

**Captures:** Level 2 across all cores + inter-core NOC log + semaphore states.
**Replays:** Multiple cores with full NOC connectivity.

**Implementation:**
1. Reserve contiguous core block on device
2. Load all per-core state (binaries, configs, data)
3. Set up inter-core semaphore initial values
4. Launch all cores simultaneously via multicast go signal
5. Monitor completion via worker completion counter
6. Verify all output buffers against captured outputs

**Additional capture requirements:**
- `CoreRangeSet` for each kernel group
- Multicast NOC encodings for inter-core ranges
- Semaphore IDs and initial values per core
- Go signal dispatch core coordinates

---

## 4.4.8 Bug-Type-to-Strategy Mapping

| Bug Type | Strategy | Hooks |
|----------|----------|-------|
| LLK math correctness | Level 1 | H1 + H4 |
| Unpack/pack data format | Level 1 | H1 + H4 |
| Data movement logic | Level 2 | H1 + H4 + DRAM |
| CB producer/consumer deadlock | Level 2 | H1 + H4 |
| Inter-core synchronization | Level 3 | H1 + H4 + NOC trace |
| Multi-core collective bugs | Level 3 | H1 + H4 + NOC trace + DRAM |
| Dispatch system bugs | Raw command capture | H5 (hugepage tap) |

---

## 4.4.9 Config Buffer and Launch Message Ring Buffers

**Config buffer:** For single-dispatch replay, use fixed slot 0
(`hal.get_dev_addr(TENSIX, KERNEL_CONFIG)`). For trace replay, use captured
`ProgramDispatchMetadata::kernel_config_addrs`.

**Launch message ring:** For single replay, use slot 0 with `launch_msg_rd_ptr = 0`.
For trace replay, track write pointer including `RUN_MSG_RESET_READ_PTR` wraps.

The prefetcher cache (`query_prefetcher_cache()`) can be ignored by the replay
engine -- always use the "not cached" path, loading binaries directly to L1.

---

## 4.4.10 Capture Format: Dispatch-Level Data for Ch3's FlatBuffer Schema

Ch3, Section 3 proposes the `KernelSnapshotCommand` FlatBuffer schema as the authoritative serialization format for kernel captures, extending LightMetal's existing FlatBuffer infrastructure. The dispatch-level data captured at H4 maps directly into that schema:

| Dispatch Data (from H4) | Ch3 FlatBuffer Table | Notes |
|--------------------------|---------------------|-------|
| `ProgramCommandSequence` bytes | `KernelSnapshotCommand.command_sequence` | Raw dispatch commands |
| `KernelGroup.launch_msg` | `KernelSnapshotCommand.launch_msg` | Per-group enables, offsets |
| `ProgramTransferInfo` binaries | `KernelBinary` (per processor) | Deduplicated by FNV1a hash |
| `ProgramConfig` offsets | `KernelSnapshotCommand.program_config` | L1 memory map |
| DRAM buffer contents | `DRAMBufferSnapshot` | Optional, Level 2+ |
| NOC trace events | `NocTraceSnapshot` | Optional, Level 3 |

The dispatch interceptor at H4 populates the in-memory representations of these tables. Background serialization writes them into the `LightMetalBinary` container as additional `CommandType` union members, following the versioning and schema evolution strategy defined in Ch3, Section 3, Item 5.

| Content Level | Approximate Size |
|--------------|------|
| Level 1 (compute only) | ~77 KB first, ~13 KB re-dispatch |
| Level 2 (single-core full) | 1--100 MB |
| Level 3 (multi-core) | 10 MB -- 1 GB |

---

## 4.4.11 Performance Overhead Summary

| Operation | Time | Notes |
|-----------|------|-------|
| Capture at H4 (memcpy) | 1.5 us | 13 KB at ~10 GB/s |
| Metadata assembly | 0.3 us | Small struct copies |
| Ring buffer enqueue | 0.2 us | Lock-free SPSC |
| Background serialization | 5 us (amortized) | Async I/O |
| **Total inline overhead** | **~2 us** | Added to dispatch latency |

At 10K dispatches/s: 20 ms/s inline (2% wall time), ~130 MB/s background write
(within NVMe bandwidth).

**PROPOSED: Tiered Capture:**

| Tier | Overhead | Data | Use Case |
|------|----------|------|----------|
| 1 (always on) | <0.5 us | ~1 KB | Post-mortem identification |
| 2 (on-demand) | ~2 us | ~13 KB | Compute replay |
| 3 (debug) | variable | 1-100 MB | Full-fidelity replay |

Mode: `TT_METAL_CAPTURE_MODE=off|metadata|full|full_dram`

---

## 4.4.13 Replay Engine Architecture

```
                    +-------------------+
                    | Kernel Snapshot   |
                    +--------+----------+
                             |
              +--------------+---------------+
              |              |               |
     +--------v-------+ +---v---------+ +---v-------------+
     | Level 1:        | | Level 2:    | | Level 3:         |
     | Compute Only    | | Full Core   | | Multi-Core       |
     +--------+--------+ +---+---------+ +---+-------------+
              |              |               |
     +--------v-------+ +---v---------+ +---v-------------+
     | Slow Dispatch  | | 5-Thread    | | Fast Dispatch    |
     | (MMIO writes)  | | Host Replay | | (CQ submission)  |
     +--------+--------+ +---+---------+ +---+-------------+
              |              |               |
     +--------v-------+ +---v---------+ +---v-------------+
     | GDB Attach     | | GDB Multi-  | | Batch Verify     |
     | (single step)  | | inferior    | | (output compare) |
     +----------------+ +-------------+ +-----------------+
```

### GDB Integration via Symbol Interposition

For host-side replay, hardware primitives are replaced:

```cpp
// PROPOSED: Host-side stubs
volatile uint32_t simulated_sync_regs[NUM_CIRCULAR_BUFFERS * 2];
volatile uint32_t* get_cb_tiles_received_ptr(int op) {
    return &simulated_sync_regs[op * 2]; }
volatile uint32_t* get_cb_tiles_acked_ptr(int op) {
    return &simulated_sync_regs[op * 2 + 1]; }
#define tt_l1_ptr
#define tt_reg_ptr volatile
```

The tt-llk test framework (`trisc.cpp`) provides a host-side TRISC model as a
foundation (see Section 4.3.10).

---

## 4.4.14 Replay Fidelity Levels

| Level | What Replayed | NOC | Hardware | Debug Capability |
|-------|--------------|-----|----------|-----------------|
| L0 | Single LLK op (e.g., matmul tile) | N/A | None (host) | Full GDB |
| L1 | TRISC0+1+2 with CB handshake | N/A | None (host) | GDB each TRISC |
| L2 | All 5 RISCs on one core | Stubbed | None (host) | GDB all 5 threads |
| L3 | All RISCs on all program cores | Emulated | None (host) | Limited (scale) |
| L4 | Actual hardware execution | Real NOC | Tensix device | Watcher + profiler |

**Recommendation:** L1 (compute pipeline replay) provides the best cost/benefit
ratio for LLK debugging. It captures exact compute behavior without NOC complexity.

---

## 4.4.15 Wormhole vs. Blackhole NOC Differences

| Feature | Wormhole | Blackhole |
|---------|----------|-----------|
| Remote CB barrier | Not needed | Required (NOC atomic flush) |
| Dynamic NOC mode | Supported | Supported (dual-NOC assertions) |
| Posted write ordering | In-order per NOC | In-order per NOC |

On Blackhole in dynamic mode, kernel-end assertions verify both NOCs
(`brisc.cc:538-553`). The NOC stub must track transactions on both NOCs.

---

## 4.4.16 Validation Strategy

For the replay to be useful, its output must match what real hardware produces.
The validation strategy has three levels:

1. **Bit-exact tile comparison** -- After replay, compare output CB tile data
   byte-for-byte against device output (read back via `ReadBuffer`). This is the
   primary correctness check.

2. **Sync register trace comparison** -- Record the sequence of `tiles_received` /
   `tiles_acked` updates on device (via profiler or watcher) and compare against
   the replay's update sequence. This validates the CB handshake ordering.

3. **Watcher waypoint comparison** -- The watcher debug infrastructure provides
   per-RISC waypoint tracking:
   - BRISC: `I -> GW -> GD -> R -> D -> NTW -> NTD`
   - NCRISC: `I -> W -> R -> D`
   - TRISC: `I -> W -> R -> D`

   These serve as a coarse execution trace for validating replay phase ordering.

---

## Key Takeaways

1. **TRISCs have no NOC access**, making Level 1 (compute-only, ~33 KB) the simplest
   and most immediately useful replay strategy, requiring no NOC stubbing.

2. **The `NocReplayStub` must use a queue, not a map.** The same address may be read
   multiple times with different data. A queue preserves temporal ordering.

3. **The FlatBuffer capture format (Ch3, Section 3) is extensible across levels.** Optional fields allow
   Level 1 captures (~77 KB) and Level 3 captures (~1 GB) within the same `KernelSnapshotCommand` schema.

4. **Multi-device capture requires correlation metadata** (mesh workload ID,
   coordinate range, dispatch sequence number) for global ordering reconstruction.

5. **Inline overhead of ~2 us per dispatch (2% at 10K/s) is acceptable.** Background
   serialization at ~130 MB/s is within NVMe bandwidth. The tiered system allows
   production-safe tracing.

6. **The bug-type-to-strategy mapping guides tool selection.** LLK math bugs need
   only Level 1 (~33 KB); inter-core sync bugs require Level 3 with NOC connectivity.
