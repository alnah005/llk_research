# 4.3 The 5-RISC Coordination Model and KernelGroup Dispatch

## Overview

Each Tensix core on Tenstorrent hardware contains **five RISC-V processors**
(BRISC, NCRISC, TRISC0, TRISC1, TRISC2) that must be coordinated through a
shared-memory mailbox protocol. This section details the on-chip coordination
mechanism, the circular buffer hardware synchronization mechanism, and the
complete state needed for single-core replay.

> **Cross-reference:** Ch1 (Hardware Debugger Reference) Section 3 establishes
> requirements for a Tenstorrent kernel debugger. Ch2 (State Inventory) Section 4
> covers the implicit coordination state described here at the firmware level.
> Section 4.1 describes the launch message that triggers execution.

---

## 4.3.1 The Five RISCs and Their Roles

| RISC | HAL Enum | Firmware | Role | NOC Access |
|------|----------|----------|------|-----------|
| BRISC | `TensixProcessorTypes::DM0` | `brisc.cc` | Master orchestrator + DM0 | Yes |
| NCRISC | `TensixProcessorTypes::DM1` | `ncrisc.cc` | Data movement 1 (subordinate) | Yes |
| TRISC0 | `TensixProcessorTypes::MATH0` | `trisc.cc` (TRISC=0) | Unpack | Compute-only |
| TRISC1 | (same core type) | `trisc.cc` (TRISC=1) | Math | Compute-only |
| TRISC2 | (same core type) | `trisc.cc` (TRISC=2) | Pack | Compute-only |

TRISCs access Tensix configuration registers and the math engine but do not issue
NOC transactions. They communicate with dataflow RISCs exclusively through circular
buffers and hardware sync registers.

On **Quasar** (source: `kernel.hpp:376-507`), the RISC count expands to 24:
8 DM cores (DM0--DM7) + 16 TRISCs across 4 Neo engines.

---

## 4.3.2 The Subordinate Synchronization Protocol

BRISC coordinates all other RISCs via a shared-memory `subordinate_sync` mailbox (`brisc.cc:48-49`). The `subordinate_map_t` union (defined in `core_config.h`) packs four sync bytes (dm1, trisc0, trisc1, trisc2) into a single `uint32_t all`, enabling bulk completion checks (`all == 0` means all subordinates done). For the full struct definition and sync constant table (`RUN_SYNC_MSG_INIT=0x40`, `RUN_SYNC_MSG_GO=0x80`, `RUN_SYNC_MSG_DONE=0x00`, etc.), see Ch2, Section 4, Item 2.

The dispatch-specific protocol below shows how these constants drive the per-kernel execution cycle:

### 4.3.2.1 The 8-Phase Per-Kernel Execution Cycle

```
Phase 1: INIT -- BRISC writes all = 0x40404040; subordinates spin
Phase 2: SYNC REG INIT -- TRISC0 zeroes tiles_received/tiles_acked for all CBs
Phase 3: GO RECEIVED -- BRISC detects go_messages[].signal == RUN_MSG_GO
Phase 4: CB LOADING -- BRISC sets dm1=LOAD; NCRISC loads CB configs; BRISC inits
Phase 5: TRISC LAUNCH -- BRISC sets trisc0/1/2 = GO; TRISCs begin execution
Phase 6: NCRISC LAUNCH -- Architecture-dependent (see Section 4.3.8)
Phase 7: KERNEL EXECUTION -- All 5 RISCs run concurrently
Phase 8: COMPLETION -- Subordinates set DONE; BRISC polls all==0; notifies dispatch
```

Source references: `brisc.cc` lines 266-329 (coordination functions), 384-581
(main loop); `trisc.cc` lines 96-223; `ncrisc.cc` lines 76-211.

All three TRISCs are launched together when MATH0 is enabled (`brisc.cc:274-285`).
There is no mechanism to launch individual TRISCs independently.

---

## 4.3.3 The `enables` Bitmask and KernelGroup Mapping

The `enables` field in `kernel_config_msg_t` (`dev_msgs.h:158`):

| Processor | Enum Value | Bit Position |
|-----------|-----------|-------------|
| DM0 (BRISC) | 0 | bit 0 (`0x01`) |
| DM1 (NCRISC) | 1 | bit 1 (`0x02`) |
| MATH0 (TRISC0/1/2) | 2 | bit 2 (`0x04`) |

The `KernelGroup` (`program_impl.hpp:69-94`) bundles cores with identical kernel
signatures:

```cpp
struct KernelGroup {
    CoreRangeSet core_ranges;
    std::vector<KernelHandle> kernel_ids;   // ordered by processor index
    dev_msgs::launch_msg_t launch_msg;      // template for this group
    dev_msgs::go_msg_t go_msg;
    // ... rta_sizes, crta_offsets, kernel_text_offsets
};
```

| Scenario | Enables |
|----------|---------|
| Matmul: reader + writer + compute | `0x07` (DM0 + DM1 + MATH0) |
| Data movement only | `0x03` (DM0 + DM1) |
| Compute only | `0x04` (MATH0) |

---

## 4.3.4 The Launch Message Life Cycle (5 Phases)

1. **Template Construction** -- `enables`, `brisc_noc_id/mode`, `local_cb_mask`
2. **Offset Finalization** -- `rta_offset[]`, `kernel_text_offset[]`, `sem_offset[]`
3. **Assembly** -- Copied into packed write command; `LaunchMsgData` preserves
   `original_msg` (phase-2 snapshot) and `msg_ptr` (view into command buffer)
4. **Update** -- `kernel_config_base` patched with config buffer slot;
   `host_assigned_id` set to program runtime ID
5. **Hardware Delivery** -- Multicast to all cores in range via launch message
   ring buffer

At H4, the launch message reflects phases 1--4. Both `original_msg` and the
command buffer view should be captured.

---

## 4.3.5 The Go Signal Protocol

```cpp
// go_msg_t in dev_msgs.h:179-189
struct go_msg_t {
    union { uint32_t all; struct {
        uint8_t dispatch_message_offset;
        uint8_t master_x, master_y;
        uint8_t signal;  // RUN_MSG_GO (0x80)
    }; };
};
```

BRISC polls `go_messages[go_message_index].signal`. The preload flag
(`DISPATCH_ENABLE_FLAG_PRELOAD`) allows early config loading before GO arrives.

---

## 4.3.6 The Circular Buffer Handshake: Stream Scratch Registers

### CB Interface Data Structure

```cpp
// circular_buffer_interface.h (lines 64-84 on WH/BH, 64-90 on Quasar)
struct LocalCBInterface {
    uint32_t fifo_size, fifo_limit, fifo_page_size, fifo_num_pages;
    uint32_t fifo_rd_ptr, fifo_wr_ptr;
    union { uint32_t tiles_acked_received_init;
        struct { uint16_t tiles_acked; uint16_t tiles_received; }; };
    uint32_t fifo_wr_tile_ptr;
};
```

### Hardware Sync via Stream Scratch Registers (NOT L1)

The critical insight: `tiles_received` and `tiles_acked` are stored in **stream
scratch registers**, not L1 memory:

```cpp
// stream_io_map.h (Blackhole, lines 24-31)
inline volatile uint32_t* get_cb_tiles_received_ptr(int operand) {
    return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(
        get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
}
inline volatile uint32_t* get_cb_tiles_acked_ptr(int operand) {
    return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(
        get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}
```

These registers provide **atomic cross-RISC visibility** without memory barriers.
A write by one RISC is immediately visible to reads by other RISCs.

### The Four CB Operations

**`cb_push_back(cb_id, N)`** -- Called by the producer after writing tile data.
Increments `tiles_received` stream register and advances `fifo_wr_ptr` with wrap
check (`dataflow_api.h:200-215`).

**`cb_wait_front(cb_id, N)`** -- Called by the consumer to wait for tiles. Spins
on `tiles_received - tiles_acked >= N` via stream register reads
(`dataflow_api.h:468-479`).

**`cb_pop_front(cb_id, N)`** -- Called by the consumer to free space. Increments
`tiles_acked` stream register and advances `fifo_rd_ptr` with wrap check
(`dataflow_api.h:242-257`).

**`cb_reserve_back(cb_id, N)`** -- Called by the producer to wait for space. Spins
on `fifo_num_pages - (received - acked) >= N` (`dataflow_api.h:388-408`).

### Typical Data Flow: Matmul Example

```
BRISC (reader):
  for each block:
    cb_reserve_back(CB_IN0, N)     // wait for space
    noc_async_read(...)            // DMA tiles from DRAM
    noc_async_read_barrier()       // wait for DMA
    cb_push_back(CB_IN0, N)        // signal tiles available

TRISC0 (unpack):
  for each block:
    cb_wait_front(CB_IN0, N)       // wait for tiles from reader
    // ... unpack tiles to SRC registers ...
    cb_pop_front(CB_IN0, N)        // free space for reader
    cb_push_back(CB_MATH, M)       // signal to math

TRISC1 (math):
  for each block:
    cb_wait_front(CB_MATH, M)      // wait for unpacked tiles
    // ... execute math in DEST register ...
    cb_pop_front(CB_MATH, M)       // free space
    cb_push_back(CB_OUT, M)        // signal to pack

TRISC2 (pack):
  for each block:
    cb_wait_front(CB_OUT, M)       // wait for computed tiles
    // ... pack from DEST to L1 ...
    cb_pop_front(CB_OUT, M)        // free space

NCRISC (writer):
  for each block:
    cb_wait_front(CB_WRITE, P)     // wait for packed tiles
    noc_async_write(...)           // DMA to DRAM
    noc_async_write_barrier()
    cb_pop_front(CB_WRITE, P)      // free space
```

### Sync Register Initialization

Before each kernel, TRISC0 zeroes all sync registers (`trisc.cc:96-105`):

```cpp
void init_sync_registers() {
    for (uint32_t operand = 0; operand < NUM_CIRCULAR_BUFFERS; operand++) {
        get_cb_tiles_received_ptr(operand)[0] = 0;
        get_cb_tiles_acked_ptr(operand)[0] = 0;
    }
}
```

Triggered asynchronously by BRISC via `trigger_sync_register_init()` (line 331).

---

## 4.3.7 CB Interface Initialization per RISC

| RISC | CB read | CB write | init_wr_tile_ptr | Notes |
|------|---------|----------|------------------|-------|
| BRISC | yes | yes | no | Full read/write for dataflow |
| NCRISC | yes | yes | no | Full read/write for dataflow |
| TRISC0 | yes | no | no | Read-only (consumes input) |
| TRISC1 | **none** | **none** | **none** | No CB interface at all |
| TRISC2 | no | yes | yes | Write-only (produces output) |

**TRISC1 (Math) has no CB interface.** The math RISC operates only on hardware
registers (SRC, DEST). The `UCK_CHLKC_MATH` define skips CB includes entirely
(`trisc.cc` lines 20-23). This makes TRISC1 the simplest target for isolated replay.

---

## 4.3.8 Architecture Differences in Coordination

| Aspect | Wormhole | Blackhole | Quasar |
|--------|----------|-----------|--------|
| NCRISC launch | Halt + reset PC (IRAM trampoline) | Direct GO signal | Extended protocol |
| CB mask width | 32-bit | 64-bit (full) | Extended |
| Remote CB barrier | Not needed | Required (NOC atomic flush) | TBD |
| Subordinate count | 4 (dm1, trisc0/1/2) | 4 | 23 (7 DMs + 16 TRISCs) |

**Wormhole NCRISC IRAM trampoline** (`brisc.cc:298-315`): NCRISC copies kernel
from L1 to IRAM via TDMA mover, writes `resume_addr`, signals
`WAITING_FOR_RESET`. BRISC detects this, resets NCRISC's PC, deasserts reset.

**Quasar `subordinate_map_t`** expands to: `uint64_t allDMs` (8 bytes) +
4 x `uint32_t allNeoN` (4 TRISCs each per Neo engine).

**PROPOSED:** The interceptor must parameterize its synchronization model by
architecture, not hard-code the 5-RISC assumption.

---

## 4.3.9 Capture Requirements for Single-Core Replay

| State | Source | Size |
|-------|--------|------|
| `enables` bitmask | `launch_msg_t` | 4 B |
| `brisc_noc_id/mode` | `launch_msg_t` | 2 B |
| `kernel_text_offset[]` | `launch_msg_t` | 20 B |
| `rta_offset[]` | `launch_msg_t` | 20 B |
| `sem_offset[]` | `launch_msg_t` | 8 B |
| `local_cb_offset/mask` | `launch_msg_t` | 10 B |
| Kernel binaries (per RISC) | `ProgramTransferInfo` | 4--48 KB each |
| Runtime args | RTA command | 64--512 B |
| CB configurations | Config buffer cmd | 512 B |
| Semaphore initial values | Config buffer cmd | 32--128 B |

**Total:** ~50--100 KB first dispatch, ~1--2 KB re-dispatch (with deduplication).

---

## 4.3.10 PROPOSED: Single-Core Replay Without Full Dispatch

```cpp
void replay_single_core(const CapturedDispatch& capture,
                        IDevice* device, CoreCoord core) {
    // 1. Write kernel binaries to L1
    for (auto& [processor, binary] : capture.binaries)
        device->write_to_device(binary.data, core, binary.offset, binary.size);
    // 2. Write runtime args
    device->write_to_device(capture.rtas.data(), core, capture.rta_offset, ...);
    // 3. Write CB configs + semaphores
    device->write_to_device(capture.cb_config.data(), core, capture.cb_offset, ...);
    device->write_to_device(capture.sem_values.data(), core, capture.sem_offset, ...);
    // 4. Write launch message and trigger via slow dispatch
    device->write_to_device(&capture.launch_msg, core,
                            hal.get_dev_addr(TENSIX, LAUNCH), sizeof(launch_msg_t));
    device->write_to_device(&capture.go_msg, core,
                            hal.get_dev_addr(TENSIX, GO_MSG), sizeof(go_msg_t));
}
```

This bypasses the entire fast dispatch pipeline.

For compute-only replay (TRISC0/1/2 isolation), the CB sync registers must be
pre-set to avoid deadlocks:

```
For each input CB (consumed by TRISC0/unpack):
  tiles_received = total_tiles_that_would_be_produced
  tiles_acked = 0
  fifo_rd_ptr = fifo_start

For each intermediate CB (between TRISCs):
  tiles_received = 0  (incremented by producing TRISC)
  tiles_acked = 0     (incremented by consuming TRISC)
  fifo_num_pages = actual_capacity

For each output CB (produced by TRISC2/pack):
  tiles_acked = 0
  fifo_num_pages = large_value  (ensure pack never blocks)
```

The tt-llk test infrastructure (`tt-llk/tests/helpers/src/trisc.cpp`) provides a
host-side TRISC execution model that can serve as a foundation for host-side replay
(see Section 4.4).

---

## 4.3.11 Sync Primitives Summary for Replay

| Primitive | Hardware Mechanism | Host Replay Strategy |
|-----------|-------------------|---------------------|
| `subordinate_sync` | L1 mailbox bytes | Shared atomic bytes |
| `tiles_received/acked` | **Stream scratch registers** | Shared atomic uint16 |
| `cb_push_back/pop_front` | Register write + local ptr | Atomic inc + local update |
| `cb_wait_front/reserve_back` | Spin on register read | Spin on atomic or pre-populate |
| `tensix_sync()` | Hardware pipeline barrier | No-op (host) |
| `invalidate_l1_cache()` | Cache control register | No-op (host) |
| NCRISC IRAM copy | TDMA mover DMA | Direct function call (host) |
| `noc_semaphore_inc` | NOC atomic increment | Atomic variable increment |

---

## Key Takeaways

1. **BRISC is the master orchestrator.** All subordinate RISCs are launched and
   synchronized through the `subordinate_sync` mailbox protocol.

2. **CB synchronization uses stream scratch registers, not L1 memory.**
   `tiles_received` and `tiles_acked` are in hardware stream registers, providing
   atomic cross-RISC visibility without barriers. Host replay must replace these
   with shared atomics.

3. **The 8-phase execution cycle provides natural checkpoint boundaries.** The
   transition between Phase 4 (CB loading) and Phase 7 (kernel execution) is the
   ideal on-device capture point.

4. **The `KernelGroup` is the fundamental capture unit.** Each group fully
   describes which processors are active on which cores, with a complete
   `launch_msg_t` carrying all per-processor configuration.

5. **TRISC1 (Math) has no CB interface**, making it the simplest replay target --
   only register file state and runtime args are needed.

6. **Architecture differences are significant.** Wormhole IRAM trampoline, Blackhole
   remote CB barriers, and Quasar's 24-RISC count all require parameterized replay.
