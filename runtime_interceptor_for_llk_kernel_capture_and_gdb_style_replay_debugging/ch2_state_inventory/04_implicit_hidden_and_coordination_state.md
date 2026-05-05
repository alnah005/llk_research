# Implicit, Hidden, and Coordination State

This is the most critical file in the state inventory. Files 01--03 covered state that is explicitly created by the programmer or visible in host-side data structures. This file covers state that is **easy to miss** during capture design -- state that exists only in firmware internals, hardware configuration registers, volatile inter-processor synchronization structures, or environmental context that the programmer never touches directly but that the kernel depends on.

Missing any item in this file will not cause a build error or an obvious API failure; it will cause a silent replay divergence where the captured kernel produces different behavior (or deadlocks) compared to the original execution. These are the state categories that distinguish a "demo" replay from a "faithful" replay.

---

## 1. Firmware Version and Initialization State

### 1.1 The Firmware Binary Mismatch Risk

TT-Metal firmware (BRISC firmware, NCRISC firmware, TRISC firmware) is compiled separately from user kernels and loaded to fixed L1 addresses during device initialization. The firmware version determines:

- How the launch message ring buffer is processed (`launch_msg_rd_ptr`, `launch_msg_buffer_num_entries = 8`)
- How the subordinate synchronization protocol works
- How mailbox addresses are computed (via `GET_MAILBOX_ADDRESS_DEV`)
- How `ebreak` and assert traps are handled

If a snapshot is captured on firmware version N and replayed on firmware version N+1, the firmware initialization of mailbox fields, synchronization structures, and memory layout may differ, causing the kernel to behave differently even with identical binary and configuration state.

**What must match for replay:**

| Item | Why It Matters | Where to Find It |
|---|---|---|
| Firmware ELF binaries | Firmware initializes L1 regions, mailbox layout, CB pointers, and semaphore addresses | `JitBuildEnv::get_firmware_binary_root()` |
| Firmware version / build hash | Identifies the firmware binary uniquely | Build artifacts in `out_firmware_root_` |
| Mailbox initialization | Firmware writes initial values to `mailboxes_t` before any program runs | `tt_metal/hw/firmware/` sources |

> For emulator-based replay, the firmware does not run (the replay system directly sets up the L1 memory map from the snapshot). But the snapshot must record the firmware version so that the replay environment can validate compatibility. For on-hardware replay, the firmware version must match exactly.

### 1.2 `core_info_msg_t` -- Per-Core Environmental Data

The `core_info_msg_t` structure within `mailboxes_t` is populated once during device initialization and never modified during kernel execution:

```cpp
struct core_info_msg_t {
    volatile uint64_t noc_pcie_addr_base;
    volatile uint64_t noc_pcie_addr_end;
    volatile uint64_t noc_dram_addr_base;
    volatile uint64_t noc_dram_addr_end;
    addressable_core_t non_worker_cores[MAX_PHYSICAL_NON_WORKER_CORES];
    addressable_core_t virtual_non_worker_cores[MAX_VIRTUAL_NON_WORKER_CORES];
    volatile uint8_t harvested_coords[MAX_HARVESTED_ON_AXIS];
    volatile uint8_t virtual_harvested_coords[MAX_HARVESTED_ON_AXIS];
    volatile uint8_t noc_size_x;
    volatile uint8_t noc_size_y;
    volatile uint8_t worker_grid_size_x;
    volatile uint8_t worker_grid_size_y;
    volatile uint8_t absolute_logical_x;
    volatile uint8_t absolute_logical_y;
    volatile uint32_t l1_unreserved_start;
    volatile CoreMagicNumber core_magic_number;
};
```

`CoreMagicNumber` values identify the core's role:
- `WORKER = 0x574F524B` ("WORK")
- `IDLE_ETH = 0x49455448` ("IETH")
- `ACTIVE_ETH = 0x41455448` ("AETH")

Data movement kernels use `noc_pcie_addr_base`, `noc_dram_addr_base`, `noc_size_x/y`, and the core mapping tables to compute NOC transaction addresses. If these values are wrong in the replay environment, NOC addresses will be computed incorrectly.

### 1.3 Dispatch Core Configuration

The dispatch system uses dedicated cores (not part of the worker grid) to relay commands from the host to worker cores. The dispatch core's identity is embedded in the `go_msg_t`:

```cpp
struct go_msg_t {
    uint8_t dispatch_message_offset;
    uint8_t master_x;   // Dispatch core physical X
    uint8_t master_y;   // Dispatch core physical Y
    uint8_t signal;
};
```

Worker cores use `master_x` and `master_y` to send completion notifications back to the dispatch core. If the replay environment does not have a dispatch core at those coordinates, the completion signal will be sent to the wrong address. For single-core emulator replay, the completion signal can be intercepted and ignored. For on-hardware replay, a dispatch core must be present or the completion path must be stubbed.

---

## 2. The RUN_SYNC Protocol (Inter-RISC Synchronization)

### 2.1 The Launch Sequence

The five-RISC launch sequence on Blackhole (and analogous 24-processor sequence on Quasar) uses the subordinate sync map:

1. Dispatch core writes `launch_msg_t` to the core's mailbox ring buffer via NOC.
2. Dispatch core writes `go_msg_t` with `signal = RUN_MSG_GO`.
3. BRISC (the master) reads the launch message, sets up the program config, and writes `RUN_SYNC_MSG_GO = 0x80` to each subordinate's byte in the sync map.
4. On Blackhole, the combined pattern `RUN_SYNC_MSG_ALL_GO = 0x80808080` triggers all four subordinates simultaneously (micro-optimization: single 32-bit write).
5. Each subordinate polls its sync byte, sees GO, and begins executing its kernel binary.
6. When each subordinate completes, it writes `RUN_SYNC_MSG_DONE = 0` to its sync byte.
7. BRISC polls all subordinates' bytes until `RUN_SYNC_MSG_ALL_SUBORDINATES_DONE = 0` (all bytes zero), then signals completion.

On Quasar, the pattern is more complex:
- `RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_DONE = 0` (64-bit for 8 DMs)
- `RUN_SYNC_MSG_ALL_SUBORDINATES_DMS_INIT = 0x40404040404040` (64-bit)
- Separate 32-bit words for each NEO engine's TRISCs

### 2.2 Sync Constants

| Constant | Value | Meaning |
|---|---|---|
| `RUN_SYNC_MSG_INIT` | 0x40 | Subordinate initialization |
| `RUN_SYNC_MSG_GO` | 0x80 | Start execution |
| `RUN_SYNC_MSG_LOAD` | 0x01 | Trigger CB/IRAM preloading |
| `RUN_SYNC_MSG_WAITING_FOR_RESET` | 0x02 | Waiting for pointer reset |
| `RUN_SYNC_MSG_INIT_SYNC_REGISTERS` | 0x03 | Initialize sync registers |
| `RUN_SYNC_MSG_DONE` | 0x00 | Execution complete |
| `RUN_SYNC_MSG_ALL_GO` | 0x80808080 | All subordinates should start |
| `RUN_SYNC_MSG_ALL_INIT` | 0x40404040 | All subordinates initializing |
| `RUN_SYNC_MSG_ALL_SUBORDINATES_DONE` | 0x00000000 | All subordinates finished |

### 2.3 Implications for Single-TRISC Replay

When replaying a single TRISC (e.g., TRISC1 for math debugging), the replay environment must simulate the synchronization protocol. The simplest approach:

- Pre-set the TRISC1 sync byte to GO.
- Pre-set all semaphores to values that allow TRISC1 to proceed (simulating that TRISC0 has already completed unpacking).
- Ignore the DONE write from TRISC1 (or capture it to verify completion).

This requires knowing the initial semaphore values for the "unblocked" state, which depends on the specific kernel's synchronization protocol.

---

## 3. Tensix Configuration Registers (CFG State)

### 3.1 The Hardware Configuration Register File

Each Tensix engine has a configuration register file accessible via `TENSIX_CFG_BASE`. This register file controls the entire compute datapath: unpacker data format interpretation, ALU operation modes, packer output format, rounding behavior, and destination register addressing.

In the Quasar LLK codebase (`tt_llk_quasar/common/inc/ckernel_trisc_common.h`):

```cpp
std::uint32_t volatile* const cfg = (std::uint32_t volatile*)TENSIX_CFG_BASE;
```

Key configuration categories:

| Category | Representative Register Fields | Effect on Computation |
|---|---|---|
| ALU format | `ALU_FORMAT_SPEC_REG_SrcA_val`, `ALU_FORMAT_SPEC_REG_SrcB_val` | Input data format interpretation |
| ALU accumulator | `ALU_ACC_CTRL_Fp32_enabled`, `ALU_ACC_CTRL_SFPU_Fp32_enabled`, `ALU_ACC_CTRL_INT8_math_enabled` | FP32 vs FP16 accumulation, INT8 mode |
| Rounding mode | `ALU_ROUNDING_MODE_Fpu_srnd_en` | Stochastic rounding enable |
| Unpacker | `THCON_UNPACKER0_REG0_OUT_DATA_FORMAT` | Unpacker output format |
| Destination | `DEST_TARGET_REG_CFG_MATH_Offset` | Destination register double-buffering |

### 3.2 The Config State ID Problem

On Wormhole/Blackhole, the LLK uses a double-buffered config state mechanism (`cfg_state_id`), where `CFG_STATE_SIZE` determines the offset between the two configuration banks. This allows one bank to be active while the other is being written, enabling pipelining.

The commented-out code in `ckernel.h` reveals the relationship:

```cpp
// const uint32_t cfg_addr0 = (cfg_state_id == 0)
//     ? ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32
//     : (CFG_STATE_SIZE * 4) + ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32;
```

When a kernel starts, it typically calls `reset_cfg_state_id()` and `reset_dest_offset_id()` to initialize the config state machine. But if these reset functions are not called (or if the kernel is a continuation of a multi-phase operation), the config state ID from the previous kernel execution persists.

From `trisc.cpp` (the LLK test harness):

```cpp
#ifndef ARCH_QUASAR
    ckernel::reset_cfg_state_id();
    ckernel::reset_dest_offset_id();
#endif
```

Note: on Quasar, these resets are not called in the test harness, suggesting Quasar handles config state differently.

### 3.3 HW Config Register Dump Size

The hardware config registers accessible through the debug interface span multiple banks:

- `THREAD_0_CFG`, `THREAD_1_CFG`, `THREAD_2_CFG` -- per-thread configuration
- `HW_CFG_0`, `HW_CFG_1` -- hardware configuration with `HW_CFG_SIZE = 187` registers

Total readable config state: approximately $3 \times 187 + 2 \times 187 = 935$ 32-bit registers = ~3.7 KB.

> PROPOSED: For post-mortem captures (where the config registers may have been modified by the running kernel), the interceptor should optionally dump the 187 config registers via the debug array interface. For pre-launch captures, this is unnecessary -- the binary's initialization code will set them.

---

## 4. The SFPU Register File and LREG Preloading

### 4.1 The `regfile` Array

Each TRISC processor has access to a 64-entry register file for SFPU (Special Function Processing Unit) operations:

```cpp
// tt_llk_quasar/common/inc/ckernel.h
volatile std::uint32_t *const regfile = (volatile std::uint32_t *)REGFILE_BASE;
```

In the test harness (`trisc.cpp`), the regfile is explicitly zeroed before kernel execution:

```cpp
std::fill(ckernel::regfile, ckernel::regfile + 64, 0);
```

LLK operations use local registers (`LREG0`-`LREG7`) within the SFPU for intermediate values. These are typically initialized by the LLK functions themselves, but if a kernel is captured mid-execution (post-mortem), the LREG values are part of the execution state.

### 4.2 DEST Register File

The DEST register file holds intermediate computation results between the math and pack phases. Its size is architecture-dependent:

```cpp
// tt_llk_quasar/common/inc/ckernel.h
static int const DEST_MAX_ADDR_16B = 1024;    // Quasar: 1024 rows of 16 bytes
static int const DEST_MAX_ADDR_32B = 512;     // Quasar: 512 rows of 32 bytes (FP32 mode)
```

DEST is not directly readable from the host. On Wormhole/Blackhole, the debug array (`dbg_get_array_row` with `array_id = 0x2`) can read DEST contents. On Quasar, `rvdbg_cmd::RD_VECREG` may provide access, pending hardware validation.

> The `regfile` array (64 x 32-bit = 256 bytes) and DEST register file state are needed only for post-mortem debugging. For pre-launch capture, these are zeroed/uninitialized and will be re-initialized by the kernel.

---

## 5. FPU Rounding Mode and Math Configuration

### 5.1 Stochastic Rounding

The ALU rounding mode (`ALU_ROUNDING_MODE_Fpu_srnd_en`) controls whether stochastic rounding is enabled. This single bit can change computation results:

- When enabled, low-precision operations (BF16, BF8) use random bits to round, making outputs non-deterministic across runs.
- When disabled, deterministic rounding is used.

The rounding mode is set through the config register file:

```cpp
cfg[(cfg_state_id == 0)
    ? ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32
    : (CFG_STATE_SIZE * 4) + ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32] = mode;
```

### 5.2 Math Fidelity Configuration

`MathFidelity` (HiFi4, HiFi3, HiFi2, LoFi) affects the precision of matrix multiply operations by controlling how many input mantissa bits are used. This is a compile-time setting baked into the kernel binary via defines, but the hardware registers that implement it are set at runtime by LLK initialization code.

### 5.3 FP32 Accumulation Mode

The `ALU_ACC_CTRL_Fp32_enabled` and `ALU_ACC_CTRL_SFPU_Fp32_enabled` bits in the config register control whether the accumulator operates in FP32 mode. This is set by `ComputeConfig::fp32_dest_acc_en` and propagated via defines and LLK initialization.

> Rounding mode, math fidelity, and accumulation mode are all derived from `ComputeConfig` fields that are already captured in files 01 and 02. For pre-launch capture, the explicit capture of these config fields is sufficient because the kernel's LLK initialization code will set the hardware registers accordingly. For post-mortem capture, the actual hardware register values should be read via the debug interface.

---

## 6. L1 Data Cache State (Quasar and Blackhole)

### 6.1 The Cached/Uncached Duality (Quasar)

Quasar introduces an L1 data cache that is absent on Wormhole. The memory map on Quasar defines:

```cpp
// tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h
#define MEM_L1_BASE 0x0
#define MEM_L1_SIZE (4 * 1024 * 1024)
#define MEM_L1_UNCACHED_BASE (MEM_L1_BASE + MEM_L1_SIZE)  // upper 4MBs bypass cache
```

The same physical L1 address `A` can be accessed at:
- `A` (cached) -- goes through the L1 data cache
- `A + MEM_L1_UNCACHED_BASE` (i.e., `A + 0x400000`) -- bypasses the cache

Firmware and mailbox accesses use the uncached path:

```cpp
#define GET_MAILBOX_ADDRESS_DEV(x) \
    (&(((mailboxes_t tt_l1_ptr*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE))->x))
```

Kernel code that reads shared data (like semaphore values or circular buffer pointers modified by other processors) must use the uncached path or explicitly invalidate the cache. The `invalidate_l1_cache()` function appears throughout the codebase at synchronization points:

- In `pause.h`: after waiting for host to clear a pause flag
- In `dprint.h`: after writing to the DPRINT buffer
- In `remote_circular_buffer.h`: after checking remote CB state
- In `dataflow_api.h`: after NOC operations complete

### 6.2 Blackhole L1 Data Cache

Blackhole also has an L1 data cache for the RISC-V processors. Cache state is invisible to the programmer under normal operation (the cache is coherent with L1 SRAM), but it introduces a subtlety for capture: if the interceptor reads L1 via NOC while a kernel is running, it may see stale data if the cache has not been flushed.

### 6.3 The Cache Coherence Problem for Capture

When reading L1 from the host (via PCIe or NOC), the host reads the **physical L1**, which may not reflect the most recent writes from a RISC-V processor if those writes are still in the cache and have not been flushed:

1. **Pre-launch capture:** Generally safe because the dispatch system writes to L1 via NOC (which bypasses the cache), and kernel code has not yet started executing.
2. **Post-mortem capture:** Potentially unsafe. If the kernel crashed while dirty cache lines existed, the L1 contents read by the host may not reflect the most recent state.
3. **Mid-execution capture (if halted via debug registers):** The `rvdbg_cmd::FLUSH` command exists precisely for this purpose -- it flushes the processor pipeline (and presumably the cache) before memory is read.

> **Quasar-specific:** When capturing L1 from a halted Quasar core, issue `rvdbg_cmd::FLUSH` before reading L1 memory. For pre-launch capture, this is not needed. This is a **Quasar and Blackhole concern** -- Wormhole does not have an L1 data cache.

---

## 7. Inter-Core Semaphore State

### 7.1 Why Initial Values Are Not Enough

As noted in `02_runtime_configuration_state.md`, semaphore initial values are set at program initialization. But between initialization and the target kernel dispatch, prior kernels in the same program may have modified semaphore values through:

- `noc_semaphore_set()` -- write a new value
- `noc_semaphore_inc()` -- atomic increment (via NOC)
- `noc_semaphore_wait()` -- spin-wait until value reaches threshold
- `semaphore_init()` -- reset to a specific value

At the moment the interceptor captures state for a specific kernel dispatch, the semaphore values in L1 reflect the cumulative effect of all prior operations -- not the initial values recorded in `Semaphore::initial_value_`.

The **actual L1 value** of each semaphore must be captured:

$$\text{sem addr}_{i} = \text{kernel config base} + \text{sem offset} + (i \times 4)$$

for $i \in \{0, 1, \ldots, 15\}$ (since `NUM_SEMAPHORES = 16`).

### 7.2 The Distributed Semaphore Problem

A single program may use semaphores across many cores for inter-core synchronization. For example, a matmul operation might use semaphore 0 on all 64 cores to synchronize data distribution. The `Semaphore` object records `initial_value`, but the **actual volatile value** at any given moment depends on which cores have incremented or decremented it.

For single-core replay, inter-core semaphore values are relevant only if the target core's kernel waits on a semaphore that is incremented by a different core. In that case, the replay must either:

1. **Pre-set to "unblocked" values:** For each semaphore that the target kernel waits on, set the value to the threshold that would allow it to proceed immediately. This enables the kernel to execute without blocking, at the cost of not accurately representing timing.

2. **Capture all participating cores' semaphore state:** For multi-core replay, capture semaphore values across all cores and replay the inter-core coordination faithfully. This requires multi-core snapshots.

**Size estimate:** 16 semaphores x 4 bytes = 64 bytes per core. For 64 cores: 4 KB.

### 7.3 Hardware Semaphores vs Software Semaphores

TT-Metal uses both:
- **Hardware semaphores** (per the `Semaphore` class with `NUM_SEMAPHORES = 16`): stored at known L1 offsets, accessed via NOC atomic operations.
- **Software (pipeline) semaphores** within the compute pipeline: the `pc_buf_base[PC_BUF_SEMAPHORE_BASE + index]` array used for intra-Tensix synchronization (between TRISC0/1/2 and the unpack/math/pack coprocessor threads).

The Quasar codebase (`ckernel.h`) defines multiple named semaphores for intra-engine synchronization:

```cpp
constexpr std::uint8_t TENSIX_MATH_SEMAPHORE = p_stall::SEMAPHORE_1;
constexpr std::uint8_t TENSIX_PERF_SEMAPHORE = p_stall::SEMAPHORE_2;
constexpr std::uint8_t STREAM_SEMAPHORE = 5;
constexpr std::uint8_t TENSIX_STREAM_SEMAPHORE = p_stall::SEMAPHORE_5;
constexpr std::uint8_t UNPACK_TO_DEST_UNPACK_SEMAPHORE = 4;
constexpr std::uint8_t PACK_STREAM_SEMAPHORE = 6;
constexpr std::uint8_t UNPACK_TO_DEST_PACK_SEMAPHORE = 7;
```

These pipeline semaphores coordinate data flow between the unpack coprocessor, math engine, and pack coprocessor. They are internal to the Tensix engine and are reset at kernel initialization. They do not need explicit capture for pre-launch snapshots.

---

## 8. NOC Transaction Ordering and In-Flight State

### 8.1 Asynchronous NOC Operations

NOC reads and writes are asynchronous. When a BRISC issues a NOC read, the data may not arrive in L1 until some time later. A NOC write may be buffered. The kernel tracks completion via counter registers.

The `NocWriteEvent` and `NocReadEvent` structures in `tt_metal/impl/debug/noc_debugging.hpp` record individual transactions when NOC debugging is enabled:

```cpp
struct NocWriteEvent {
    uint32_t src_addr, dst_addr, num_bytes;
    uint32_t counter_snapshot;        // write_reqs_sent
    int8_t src_x, src_y, dst_x, dst_y;
    bool posted;
    uint8_t noc;
    bool is_semaphore, is_mcast;
    int8_t mcast_end_dst_x, mcast_end_dst_y;
};

struct NocReadEvent {
    uint32_t src_addr, dst_addr, num_bytes;
    uint32_t counter_snapshot;        // read_resps_recv
    int8_t src_x, src_y, dst_x, dst_y;
    uint8_t noc;
};
```

The `NOCDebugState` class maintains per-core, per-NOC counter snapshots and pending transaction sets.

### 8.2 Why NOC State is Hard to Capture

NOC transaction state is:
- **Volatile**: transactions complete asynchronously; the state changes every few cycles.
- **Distributed**: each core's NOC engine maintains independent state.
- **Not host-accessible in real time**: the NOC counters are device-side registers, not mapped to L1 (except on Quasar, where `MEM_NOC_COUNTER_L1_SIZE = 80` bytes of NOC counters are stored in L1).
- **Timing-dependent**: the exact sequence of transactions depends on execution timing, which is non-deterministic.

For pre-launch capture, NOC state is clean (no in-flight transactions for this kernel). For post-mortem capture, the NOC state is informative but non-reproducible.

### 8.3 NOC Counters in L1 (Quasar)

On Quasar, NOC counters are stored in L1:

```cpp
// dev_mem_map.h (Quasar)
#define MEM_NOC_COUNTER_SIZE 4
#define MEM_NOC_COUNTER_L1_SIZE (5 * 2 * 2 * MEM_NOC_COUNTER_SIZE)  // 80 bytes
#define MEM_NOC_COUNTER_BASE (MEM_TRISC3_KERNEL_BASE + MEM_TRISC_KERNEL_SIZE)
```

These counters track five transaction types across two NOCs and two directions (sent/received), providing a snapshot of NOC activity.

> NOC transaction state is not capturable at a consistent point for a running kernel. For pre-launch capture, NOC state is clean and does not need recording. For post-mortem capture, the NOC counters and the `NocWriteEvent`/`NocReadEvent` log (if NOC debugging was enabled) should be included as diagnostic data. Each `NocWriteEvent` is approximately 28 bytes; `NocReadEvent` is approximately 20 bytes. A full NOC log for a matmul kernel: ~20--100 KB per core.

---

## 9. Environmental State

### 9.1 Core Coordinates

The target core's physical and logical coordinates determine NOC address computation:
- Physical NOC coordinates $(x, y)$: used for direct NOC transactions
- Logical coordinates within the worker grid: used for runtime arg indexing
- Virtual coordinates: used by the HAL for coordinate translation

These are embedded in `core_info_msg_t::absolute_logical_x/y` and the coordinate translation tables.

### 9.2 Device Architecture and Harvesting Mask

| Item | Description | Size |
|---|---|---|
| `tt::ARCH` | `WORMHOLE_B0`, `BLACKHOLE`, or `QUASAR` | 1 byte |
| SoC descriptor | Core grid dimensions, DRAM channels, NOC topology | ~1--2 KB |
| Harvesting mask | Which cores are disabled (harvested) | ~16 bytes |
| Device clock frequency | Affects timing (not computation correctness) | 4 bytes |

The harvesting mask identifies which cores are disabled (defective). `core_info_msg_t::harvested_coords` records which positions on each axis are harvested. The replay environment must know this to correctly resolve logical-to-physical coordinate mappings.

### 9.3 SoC Descriptor

The SoC descriptor (from UMD's `soc_descriptor.hpp`) describes the full chip topology: DRAM bank locations, PCIe core locations, Ethernet core locations, and the worker grid. It is used by the HAL and coordinate translation layers.

### 9.4 `kernel_config_msg_t::sub_device_origin_x/y`

These fields record the logical origin of the sub-device that this core belongs to. Kernels may use these to compute their position within a sub-device grid.

---

## 10. Quasar-Specific Capture Implications

Quasar's NEO cluster architecture introduces several unique capture challenges that deserve dedicated analysis.

### 10.1 The NEO Cluster Model

A Quasar "core" (addressable by a single NOC coordinate) is actually a **cluster** containing:

- **8 data movement processors** (DM0-DM7), each a full RISC-V core
- **4 NEO engines** (NEO0-NEO3), each containing **4 TRISC processors** (TRISC0-TRISC3)
- **Shared 4 MB L1 SRAM** accessible by all 24 processors

From the Quasar core config:

```cpp
enum class TensixProcessorTypes : uint8_t {
    DM0 = 0, DM1 = 1, DM2 = 2, DM3 = 3,
    DM4 = 4, DM5 = 5, DM6 = 6, DM7 = 7,
    E0_MATH0 = 8,  E0_MATH1 = 9,  E0_MATH2 = 10, E0_MATH3 = 11,
    E1_MATH0 = 12, E1_MATH1 = 13, E1_MATH2 = 14, E1_MATH3 = 15,
    E2_MATH0 = 16, E2_MATH1 = 17, E2_MATH2 = 18, E2_MATH3 = 19,
    E3_MATH0 = 20, E3_MATH1 = 21, E3_MATH2 = 22, E3_MATH3 = 23,
    COUNT = 24
};
```

### 10.2 Capture Size Scaling

Every per-processor data structure in the snapshot scales with the processor count:

| Data Structure | Blackhole (5 procs) | Quasar (24 procs) | Scale Factor |
|---|---|---|---|
| `rta_offset` array | 20 bytes | 96 bytes | 4.8x |
| `kernel_text_offset` array | 20 bytes | 96 bytes | 4.8x |
| `watcher_kernel_ids` array | 10 bytes | 48 bytes | 4.8x |
| Kernel ELF binaries (compute) | 3 ELFs | 16 ELFs | 5.3x |
| Kernel ELF binaries (DM) | 2 ELFs | 8 ELFs | 4x |
| Subordinate sync map | 4 bytes | 36 bytes | 9x |
| Waypoint debug data | 20 bytes | 96 bytes | 4.8x |
| Pause flags | 5 bytes | 24 bytes | 4.8x |
| **Runtime config overhead** | **~128 bytes** | **~500 bytes** | **~4x** |

For a single Quasar cluster snapshot with full binary capture: approximately 500 KB -- 2 MB, compared to 120 KB -- 500 KB for a single Blackhole core.

### 10.3 Shared L1 Contention

On Blackhole, L1 is private to each core -- a capture of one core's L1 is independent of other cores. On Quasar, all 24 processors within a cluster **share the same 4 MB L1**. This means:

1. **CB data is shared**: TRISC0 on NEO0 and TRISC0 on NEO1 may read from different CBs in the same L1 space.
2. **Firmware regions overlap**: DM0's firmware and DM7's firmware coexist in the same L1.
3. **Concurrent writes**: Multiple DM processors may be writing to L1 simultaneously. An L1 snapshot that reads addresses sequentially may see an inconsistent view if processors are actively modifying L1.

For pre-launch capture, this is not a problem (no kernel code is running yet). For post-mortem capture on Quasar, the capture must either:
- Halt all 24 processors before reading L1 (via `rvdbg_cmd::PAUSE` for each, or via the Debug Module's `haltreq` which can target all harts)
- Accept that the L1 snapshot may be inconsistent

> Quasar's shared-L1 model actually simplifies capture in one respect: there is no need to capture separate L1 regions for each processor within a cluster. A single L1 snapshot captures all the data that all 24 processors can see. The complexity is in the larger binary set and the more intricate coordination between 24 processors.

### 10.4 The `QuasarComputeProcessor` Enumeration

The `QuasarComputeProcessor` enum (`tt_metal/impl/kernels/kernel.hpp`) maps user-level processor selection to hardware:

```cpp
enum class QuasarComputeProcessor : uint8_t {
    NEO_0_COMPUTE_0 = 0,  NEO_0_COMPUTE_1 = 1,
    NEO_0_COMPUTE_2 = 2,  NEO_0_COMPUTE_3 = 3,
    NEO_1_COMPUTE_0 = 4,  NEO_1_COMPUTE_1 = 5,
    // ... through NEO_3_COMPUTE_3 = 15
};
```

A `QuasarComputeKernel` specifies which subset of the 16 compute processors to use. The snapshot must record:
- Which `QuasarComputeProcessor` entries are active
- The mapping from processor index to NEO engine and TRISC within that engine
- The `num_threads_per_cluster` from `QuasarComputeConfig` (determines how many NEO engines are used, each contributing 4 TRISCs)

The invariant enforced in the constructor:

```cpp
TT_FATAL(
    config.num_threads_per_cluster * QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE ==
        compute_processors.size(),
    "Number of Tensix engines per cluster * 4 must match compute processors reserved");
```

### 10.5 Debug Register Targeting on Quasar

The `rvdbg_risc_sel` enum on Quasar targets only compute processors (TRISC0-3), not DMs:

```cpp
enum rvdbg_risc_sel {
    TRISC0 = 0x00000000,  // risc_sel = 0
    TRISC1 = 0x00020000,  // risc_sel = 1
    TRISC2 = 0x00040000,  // risc_sel = 2
    TRISC3 = 0x00060000,  // risc_sel = 3
};
```

This means that on Quasar, the debug register interface can target individual TRISCs within a single NEO engine, but:
- **Targeting a specific NEO engine** (NEO0 vs NEO1 vs NEO2 vs NEO3) likely requires addressing via the NEO engine's register space.
- **Targeting DM processors** (DM0-DM7) is not covered by `rvdbg_risc_sel` and requires a different access mechanism (potentially the Debug Module's `dmcontrol.hartsello`).

### 10.6 Quasar Fabric and Routing State

Quasar clusters have dedicated L1 regions for fabric networking that do not exist on Blackhole:

| Region | Size | Purpose |
|---|---|---|
| Fabric counters | 112 bytes | Per-DM barrier counters |
| Fabric connection lock | 144 bytes | Cross-RISC synchronization |
| Routing table | 2,576 bytes | Fabric network routing |
| Fabric connections | 656 bytes | Connection metadata |
| Packet header pool | 3,456 bytes ($144 \times 24$) | Pre-allocated packet headers |

Total: approximately 7 KB of fabric-related state. This state is relevant for data movement kernels that use fabric networking. For compute-only debugging, it can be ignored.

---

## 11. Dataflow Buffers (DFBs)

TT-Metal is introducing Dataflow Buffers (DFBs) as successors to circular buffers. `ProgramImpl` already tracks them:

```cpp
std::vector<std::shared_ptr<DataflowBufferImpl>> dataflow_buffers_;
std::unordered_map<uint32_t, std::shared_ptr<DataflowBufferImpl>> dataflow_buffer_by_id_;
```

DFBs introduce tile counter allocators and remapper configurations that CBs lack. As DFBs mature, the capture inventory will need to include:

- DFB configurations (analogous to CB configs)
- Tile counter state
- Remapper index state
- DFB L1 regions (`ProgramConfig::dfb_offset` + `dfb_size`)

> PROPOSED: The snapshot format should include a version field and extensible schema to accommodate DFB state as the feature stabilizes. The `ProgramConfig` already has `dfb_offset` and `dfb_size` fields (documented in `02_runtime_configuration_state.md`), indicating that the infrastructure is being laid.

---

## 12. State Capture Difficulty Classification

```
Easy to Capture                        Hard to Capture
(host-accessible, stable)              (device-side, volatile, timing-dependent)

  Compiled binaries                      SFPI register file contents
  Compile-time args                      FPU rounding mode (HW register)
  Runtime args                           L1 data cache dirty lines (BH/QS)
  CB configurations                      In-flight NOC transactions
  Semaphore declarations                 Inter-core semaphore live values
  Launch message                         NOC transaction ordering
  Go message                             Subordinate sync state (during launch)
  ProgramConfig                          L1 bank contention patterns
  Architecture / SoC descriptor          Dispatch core ring buffer state
  Core coordinates
  CB tile data (pre-dispatch, at rest)
  Full L1 dump (pre-dispatch)
```

The left column constitutes approximately 95% of what is needed for compute kernel replay. The right column matters primarily for: (a) data movement kernel replay with NOC fidelity, (b) post-mortem diagnosis of timing-dependent failures, and (c) multi-core coordinated replay.

---

## 13. Comprehensive Capture Checklist

This checklist consolidates all items from files 01--04. Items marked with (Q) are Quasar-specific.

### Must Capture (replay will fail without these)

- [ ] Kernel ELF binaries for all active processors
- [ ] Runtime arguments (per-core and common)
- [ ] Circular buffer configurations (data format, page size, total size, address)
- [ ] Semaphore declarations (id, initial value, core ranges)
- [ ] Launch message (`kernel_config_msg_t`) including `enables` bitmask
- [ ] Go message (`go_msg_t`)
- [ ] L1 memory contents for CB data regions (for compute replay)
- [ ] Device architecture identifier (Wormhole B0, Blackhole, Quasar)
- [ ] Core coordinates (physical and logical)
- [ ] Harvesting mask
- [ ] (Q) Active `QuasarComputeProcessor` / DM processor set
- [ ] (Q) `num_threads_per_cluster` for compute and DM configs

### Should Capture (replay may produce subtly different results without these)

- [ ] Compile-time arguments and named compile-time arguments
- [ ] Preprocessor defines (full effective set)
- [ ] HLK descriptor (`hlk_desc` with all per-CB arrays)
- [ ] Build key / hash
- [ ] Optimization level
- [ ] `core_info_msg_t` (NOC address bases, grid dimensions)
- [ ] `subordinate_sync_msg_t` state
- [ ] `ProgramConfig` (L1 layout offsets)
- [ ] Firmware version / TT-Metal git hash
- [ ] Live semaphore values (from L1, not just initial values)
- [ ] DRAM buffer metadata (for DM kernel replay)
- [ ] (Q) Fabric routing tables (for DM kernel debugging)

### Nice to Have (diagnostic value, not required for basic replay)

- [ ] Config register dump (post-mortem only)
- [ ] SFPU regfile / DEST contents (post-mortem only)
- [ ] NOC counter values
- [ ] NOC event log (`NocWriteEvent` / `NocReadEvent` if debugging enabled)
- [ ] Watcher state (waypoints, assert status, pause flags)
- [ ] Profiler data
- [ ] Debug ring buffer contents
- [ ] CB read/write pointer state (post-mortem only)
- [ ] (Q) L1 cache dirty line information (if accessible)

---

## Capture Feasibility Summary

| State Element | Host Accessible? | Requires Halting? | Volatile? | Omission Impact |
|---|---|---|---|---|
| Firmware mailbox region | Yes -- fixed L1 address | No for pre-dispatch | Partially -- launch ptr changes | On-hardware replay needs it; emulator does not |
| `subordinate_sync_msg_t` | Yes -- within mailboxes | Timing-sensitive | Yes -- changes during launch | Full 5-RISC replay needs it; single-TRISC does not |
| Config register state (cfg[]) | No -- hardware registers | Yes -- need debug interface | Yes -- changes during execution | Pre-launch: derived from config. Post-mortem: high value |
| `regfile` / DEST / SFPU LREGs | No on WH/BH; Yes on Quasar (via `RD_VECREG`, halted) | Yes (Quasar) | Yes -- residual from previous kernel | Only matters for post-mortem or buggy kernels |
| FPU rounding mode | Indirectly -- derived from `math_fidelity` | No | No -- set from config | Numerical results may differ if wrong |
| L1 data cache (BH/Quasar) | No -- invisible to host | Need flush before readback | Yes -- dirty lines | Post-mortem L1 readback may be stale |
| Inter-core semaphore values | Yes -- 64 bytes per core via NOC | Ideally halt first | Yes -- modified by other cores | Multi-core replay deadlocks without it |
| NOC transaction log | Only if NOC debugging enabled | No (recorded to L1 buffer) | N/A (historical log) | DM replay loses NOC fidelity |
| NOC in-flight state | No -- device-side registers | Yes -- must halt NOC engine | Yes -- changes every cycle | Cannot reproduce exact NOC behavior |
| Architecture / SoC descriptor | Yes -- host-side | No | No | Snapshot not portable across devices |
| Core coordinates / harvesting | Yes -- SoC descriptor or core_info | No | No | NOC address mapping incorrect |
| Quasar processor mapping | Yes -- Quasar API objects | No | No -- static per kernel | Replay cannot target correct NEO engine |
| Quasar fabric/routing tables | Yes -- in L1 (firmware-initialized) | No | No -- static per session | Only needed for DM kernel debug |
| Dataflow buffer state | Yes -- `ProgramImpl::dataflow_buffers_` | No | No | Future: required when DFBs replace CBs |
| Dispatch core identity | Yes -- in `go_msg_t` | No | No | Not needed for kernel replay |

---

## Key Takeaways

- **The subordinate synchronization map (`subordinate_sync_msg_t`) has a fundamentally different layout on Quasar** (36 bytes covering 23 subordinates) versus Blackhole (4 bytes covering 4 subordinates). Architecture-aware serialization is mandatory; treating it as a fixed-size blob will silently produce corrupt snapshots on the wrong architecture.

- **Quasar's shared 4 MB L1 across 24 processors creates a capture consistency problem** that does not exist on Blackhole (where L1 is per-core). For post-mortem capture, all 24 processors must be halted simultaneously before reading L1, or the snapshot may contain an inconsistent view of shared state. The `rvdbg_cmd::FLUSH` command is the only documented mechanism to force cache writeback.

- **Most "hidden" state (config registers, SFPU registers, FPU rounding mode) is re-initialized by LLK startup code** and does not need explicit capture for pre-launch snapshots. However, for post-mortem debugging -- the highest-value use case -- these hardware register values at the point of failure are essential diagnostic information and can only be captured via the debug register interface.

- **The comprehensive capture checklist (Section 13) with Must/Should/Nice-to-Have tiers** provides a direct requirements spec for the interceptor implementation (Chapter 6). Every "Must Capture" item must be verified present in the snapshot format; "Should Capture" items should be included unless explicitly excluded with justification.

---

**Previous:** [Memory Contents and Tile Data](./03_memory_contents_and_tile_data.md) | **Next:** [Chapter 3 -- LightMetal Capture: The Existing Foundation and Its Gaps](../ch3_lightmetal_foundation/index.md)
