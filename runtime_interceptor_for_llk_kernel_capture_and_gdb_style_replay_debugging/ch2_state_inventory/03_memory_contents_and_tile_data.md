# Memory Contents and Tile Data

This file addresses the largest and most complex component of the capture envelope: the actual data in L1 memory at kernel launch time. It provides a detailed L1 memory map for each architecture with concrete hex addresses, derives tile data sizes across all data formats, calculates total snapshot sizes for representative scenarios, and analyzes the core tension between **completeness** (capture everything) and **feasibility** (capturing a full 1.5 MB L1 takes ~75 ms via PCIe, which is intrusive during normal operation).

The memory layout and size estimates developed here are the foundation for the snapshot schema designed in Chapter 6 and the storage/compression analysis in Chapter 6, `03_snapshot_schema_and_serialization.md`.

---

## 1. L1 Memory Layout by Architecture

### 1.1 Wormhole B0

Wormhole B0 has `MEM_L1_SIZE` $= 1464 \times 1024 = 1{,}499{,}136$ bytes (approximately 1.43 MB) of L1 SRAM per Tensix core, mapped starting at address `0x0`.

The memory map is defined in `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h`. The reserved system area occupies the bottom of L1:

```
Address         Region                                     Size
----------------------------------------------------------------------
0x00000         Boot code / NOC atomic / L1 barrier         16 B
0x00010         Mailbox (mailboxes_t)                     12,896 B
                  +-- ncrisc_halt_msg_t
                  +-- subordinate_sync_msg_t
                  +-- launch_msg_rd_ptr
                  +-- launch_msg_t[8]      (ring buffer)
                  +-- go_msg_t[9]
                  +-- watcher_msg_t
                  +-- dprint_buf_msg_t
                  +-- core_info_msg_t
                  +-- profiler_msg_t
~0x032A0        Zeros region                                512 B
~0x034A0        LLK debug region                          1,024 B
~0x038A0        BRISC firmware                            6,144 B
~0x050A0        NCRISC firmware                           2,048 B
~0x058A0        TRISC0 firmware                           1,536 B
~0x05EA0        TRISC1 firmware                           1,536 B
~0x064A0        TRISC2 firmware                           1,536 B
~0x06AA0        NOC counters                                 80 B
~0x06AF0        Fabric counters                              32 B
~0x06B10        Fabric connection lock                      144 B
~0x06BA0        Routing table                             2,576 B
~0x075B0        Fabric connections                          656 B
~0x07840        Packet header pool                        2,304 B
~0x08140        -- MEM_MAP_END --
                (scratch area during init only)
~0x08140        BRISC init local scratch                  4,096 B
~0x09140        NCRISC init local scratch                 4,096 B
~0x0A140        TRISC0 init local scratch                 2,048 B
~0x0A940        TRISC1 init local scratch                 2,048 B
~0x0B140        TRISC2 init local scratch                 2,048 B
~0x0B940        NCRISC init IRAM scratch                  4,096 B
~0x0C940        Bank-to-NOC scratch                       2,048 B
~0x0D140        Logical-to-virtual scratch                    24 B
                -- end of scratch area --
                ...
                -- l1_unreserved_start (from core_info_msg_t) --
                Program kernel config buffer starts here:
                  +-- Semaphores
                  +-- CB configuration words
                  +-- Runtime arguments
                  +-- Kernel text (ELF binaries)
                  +-- Circular buffer DATA (tile contents)
                ...
0x16E000        -- MEM_L1_SIZE (end of L1) --
```

The `MEM_MAP_END` value for Wormhole is approximately `0x08140` (~33 KB). The scratch area above `MEM_MAP_END` is used only during firmware initialization and is reclaimed for kernel use afterward. The `l1_unreserved_start` value reported in `core_info_msg_t` marks where the program's data region begins; this is the effective start of capturable kernel state.

**Available L1 for kernel use (approximate):**

$$L1_{\text{available,WH}} = 1{,}499{,}136 - 33{,}088 \approx 1{,}466{,}048 \text{ B} \approx 1.40 \text{ MB}$$

### 1.2 Blackhole

Blackhole has `MEM_L1_SIZE` $= 1536 \times 1024 = 1{,}572{,}864$ bytes (1.5 MB) per Tensix core.

The layout follows the same pattern as Wormhole but with some differences:

- **Firmware sizes are larger**: BRISC firmware is $6 \times 1024 + 2560 = 8{,}704$ B (vs. 6,144 on WH). NCRISC firmware is 2,560 B (vs. 2,048 on WH).
- **L1 inline write region**: 64 bytes at address 32 for inline write emulation (Blackhole-specific workaround).
- **Mailbox starts at 96** (vs. 16 on Wormhole) due to the inline write region.
- **No NCRISC IRAM**: Blackhole does not have NCRISC IRAM, so NCRISC kernels run from L1 like all other processors. This removes the 16 KB IRAM size constraint.
- **Local memory**: BRISC and NCRISC have 8 KB each (vs. 4 KB on WH); TRISCs have 4 KB (vs. 2 KB on WH).
- **Packet header pool**: 3,456 B (6 directions x 2 conventions x 2 DMs x 144 bytes, vs. 2,304 on WH with 4 directions).

**Available L1 for kernel use (approximate):**

$$L1_{\text{available,BH}} \approx 1{,}572{,}864 - 40{,}000 \approx 1{,}532{,}864 \text{ B} \approx 1.46 \text{ MB}$$

### 1.3 Quasar

Quasar has `MEM_L1_SIZE` $= 4 \times 1024 \times 1024 = 4{,}194{,}304$ bytes (4 MB) per core, with an additional 4 MB of uncached L1 at `MEM_L1_UNCACHED_BASE = MEM_L1_BASE + MEM_L1_SIZE`.

Quasar's memory layout is substantially different:

- **8 DM processors + 4 NEO engines x 4 TRISCs = 24 processors** per core.
- **`MaxProcessorsPerCoreType = 24`**: all arrays indexed by processor count are 4.8x larger.
- **Mailbox is much larger**: `MEM_MAILBOX_SIZE = 57,424` B (vs. ~12,896 on WH/BH), reflecting the larger `mailboxes_t` structure with arrays sized for 24 processors.
- **DM firmware**: 12 KB (`MEM_DM_FIRMWARE_SIZE`).
- **TRISC firmware**: 4 KB each (`MEM_TRISC_FIRMWARE_SIZE`).
- **DM kernel regions**: 48 KB per DM core x 8 cores = 384 KB total.
- **TRISC kernel regions**: 24 KB per TRISC x 4 engines x 4 TRISCs = 384 KB total.
- **Global memory regions**: 2 KB per DM, 3 KB per TRISC.
- **Local memory**: 8 KB per DM, 4 KB per TRISC.

A rough calculation of `MEM_MAP_END` for Quasar:

```
MEM_MAILBOX_SIZE                      57,424 B
MEM_ZEROS_SIZE                           512 B
MEM_LLK_DEBUG_SIZE                     1,024 B
MEM_DM_FIRMWARE_SIZE                  12,288 B
4 x MEM_TRISC_FIRMWARE_SIZE           16,384 B
MEM_DM_GLOBAL_SIZE                     2,048 B
4 x MEM_TRISC_GLOBAL_SIZE             12,288 B
8 x MEM_DM_LOCAL_SIZE                 65,536 B
8 x MEM_DM_KERNEL_SIZE               393,216 B
4 x MEM_TRISC_KERNEL_SIZE             98,304 B
NOC + fabric + routing + headers      ~10,000 B
-----------------------------------------------
Approximate MEM_MAP_END              ~669,024 B (~653 KB)
```

**Available L1 for kernel use (approximate):**

$$L1_{\text{available,QS}} \approx 4{,}194{,}304 - 669{,}024 \approx 3{,}525{,}280 \text{ B} \approx 3.36 \text{ MB}$$

### 1.4 Comparison Table

| Property | Wormhole B0 | Blackhole | Quasar |
|---|---|---|---|
| Total L1 | 1,464 KB (1.43 MB) | 1,536 KB (1.5 MB) | 4,096 KB (4 MB) |
| System reserved (`MEM_MAP_END`) | ~33 KB | ~39 KB | ~653 KB |
| Available for kernel use | ~1,400 KB | ~1,460 KB | ~3,360 KB |
| Processors per core | 5 | 5 | 24 |
| `NUM_CIRCULAR_BUFFERS` | 32 | 64 | 64 |
| NCRISC kernel location | IRAM (16 KB max) | L1 | L1 |
| Mailbox size | 12,896 B | 12,896 B | 57,424 B |
| L1 alignment | 16 B | 16 B | 16 B |
| L1 data cache | No | Yes | Yes (4 MB cached + 4 MB uncached) |

---

## 2. Tile Data Sizes

### 2.1 Standard Tile Dimensions

The default tile is $32 \times 32$ elements organized as 4 faces of $16 \times 16$ each:

```cpp
// tt_metal/api/tt-metalium/constants.hpp
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HW = TILE_WIDTH * TILE_HEIGHT;  // 1024 elements
constexpr uint32_t FACE_HEIGHT = 16;
constexpr uint32_t FACE_WIDTH = 16;
constexpr uint32_t FACE_HW = FACE_WIDTH * FACE_HEIGHT;  // 256 elements
```

Custom tile shapes are supported via the `Tile` struct, but the standard $32 \times 32$ tile is by far the most common in LLK operations.

### 2.2 Tile Size by Data Format

The `tile_size()` function in `tt_metal/api/tt-metalium/tt_backend_api_types.hpp` defines the exact byte size of a standard $32 \times 32$ tile for each data format:

| Data Format | Datum Size | Tile Body | Exponent Header | Total Tile Size |
|---|---|---|---|---|
| BFP2 / BFP2_b | 2 bits | $64 \times 4 = 256$ B | $16 \times 4 = 64$ B | **320 B** |
| BFP4 / BFP4_b | 4 bits | $128 \times 4 = 512$ B | $16 \times 4 = 64$ B | **576 B** |
| BFP8 / BFP8_b | 8 bits | $256 \times 4 = 1{,}024$ B | $16 \times 4 = 64$ B | **1,088 B** |
| Float16 / Float16_b | 2 bytes | $1{,}024 \times 2 = 2{,}048$ B | 0 | **2,048 B** |
| Int8 / UInt8 / Lf8 / RawUInt8 | 1 byte | $1{,}024 \times 1 = 1{,}024$ B | 0 | **1,024 B** |
| UInt16 / RawUInt16 | 2 bytes | $1{,}024 \times 2 = 2{,}048$ B | 0 | **2,048 B** |
| Float32 / UInt32 / Int32 / RawUInt32 | 4 bytes | $1{,}024 \times 4 = 4{,}096$ B | 0 | **4,096 B** |

The block floating-point formats (BFP2, BFP4, BFP8) include a per-face exponent header of 16 bytes (one exponent byte per row within each of the 4 faces: $4 \text{ faces} \times 16 \text{ rows} = 64$ bytes total).

### 2.3 Circular Buffer Data Sizing

A circular buffer holds $N_{\text{pages}}$ tiles, where each tile is one "page." The total data size of a CB is:

$$S_{\text{CB data}} = N_{\text{pages}} \times \text{tilesize}(\text{format})$$

Common configurations in practice:

| Use Case | Data Format | Pages (Tiles) | CB Data Size |
|---|---|---|---|
| Matmul input A (single tile buffer) | BFloat16 | 2 | 4,096 B |
| Matmul input A (double buffer) | BFloat16 | 4 | 8,192 B |
| Matmul input B (multi-tile) | BFloat16 | 8 | 16,384 B |
| Matmul accumulator | Float32 | 4 | 16,384 B |
| Eltwise input | BFloat16 | 2 | 4,096 B |
| Eltwise output | BFloat16 | 2 | 4,096 B |
| Large CB (64 tiles) | BFloat16 | 64 | 131,072 B |
| Large CB (64 tiles) | Float32 | 64 | 262,144 B |
| Minimal CB (1 tile) | BFP8 | 1 | 1,088 B |
| Minimal CB (1 tile) | BFP2 | 1 | 320 B |

### 2.4 Aggregate CB Data Per Core

A typical compute program uses 4--8 circular buffers. The total tile data across all CBs:

$$S_{\text{all CBs}} = \sum_{i=0}^{N_{\text{CBs}}-1} N_{\text{pages},i} \times \text{tilesize}(\text{format}_i)$$

Representative scenarios:

| Scenario | CBs | Typical Total CB Data |
|---|---|---|
| Simple eltwise (2 CBs, 2 tiles each, BF16) | 2 | 8 KB |
| Matmul (4 CBs: 2 inputs + accum + output) | 4 | 32--64 KB |
| Complex op (8 CBs, mixed sizes) | 8 | 64--256 KB |
| Large matmul (many tiles buffered) | 6 | 128--512 KB |
| Extreme (fills most of L1) | 8+ | 512 KB--1.2 MB |

---

## 3. Capture Timing for CB Data

The critical question: **when** should CB data be captured?

### 3.1 Pre-Dispatch Capture (Ideal for Compute Replay)

At the moment `EnqueueProgram` is called, the data movement kernels from a previous dispatch (or `EnqueueWriteBuffer` calls) have already written tile data into the CBs. The CB data is "at rest" in L1 -- the new kernel has not yet started executing.

This is the ideal capture moment for compute kernels because:
- TRISC0 (unpack) reads tiles from CBs that were populated by prior data movement
- The data is stable (no concurrent writers during the capture window between dispatch and `go_msg`)
- The capture can be triggered from the host interception hook

### 3.2 Post-Mortem Capture (Failure Diagnosis)

When a kernel crashes (detected via Watcher assert, timeout, or `ebreak`), the L1 contents reflect the state at the time of failure. Some CBs may be partially consumed, CB pointers may be mid-update, and data may be corrupted. Post-mortem capture is valuable for diagnosing the failure but does not provide clean inputs for replay.

### 3.3 Mid-Execution Capture (Unreliable)

Reading L1 while the kernel is running introduces race conditions. While the NOC can read L1 from the host side, concurrent writes by the kernel may produce torn reads. This makes mid-execution capture unreliable without halting the core first.

### 3.4 L1 Readback Latency

Reading L1 from the host is supported via PCIe TLB windows or NOC-based reads (e.g., `tt::umd::Chip::read_from_device()`). The latency is approximately:
- ~10 $\mu$s per 4 KB page via PCIe TLB
- ~50 ms for a full 1 MB L1 dump (Wormhole B0)
- ~75 ms for a full 1.5 MB L1 dump (Blackhole)

For pre-dispatch capture, this latency is incurred between `finalize_offsets()` and the actual hardware submission, which is acceptable for a debug-mode capture but not for production use.

---

## 4. L1 Memory Map Diagram (Wormhole B0, Typical Matmul Program)

The following diagram shows a representative L1 layout for a matmul kernel on Wormhole B0 with the program config buffer starting at `l1_unreserved_start`:

```
0x000000 +------------------------------------------+
         |  Boot code + NOC atomic + barrier (16 B) |
0x000010 +------------------------------------------+
         |  Mailboxes (12,896 B)                    |
         |    launch_msg_t[8], go_msg_t[9],         |
         |    watcher, dprint, core_info, profiler   |
0x0032A0 +------------------------------------------+
         |  Zeros (512 B) + LLK debug (1,024 B)    |
0x0038A0 +------------------------------------------+
         |  Firmware: BRISC(6K) NCRISC(2K)          |
         |            TRISC0/1/2 (1.5K each)        |
0x006AA0 +------------------------------------------+
         |  NOC + Fabric + Routing + Packet headers |
0x008140 +------------------------------------------+  <-- MEM_MAP_END
         |  (Scratch area, reusable after init)     |
         |  ...                                     |
~0x21000 +------------------------------------------+  <-- l1_unreserved_start (typical)
         |  Kernel config buffer:                   |
         |  +----------------------------------+    |
         |  | Semaphores (64 B)                |    |
         |  | CB config (128 B, 8 CBs)         |    |
         |  | Dataflow buffers (if any)         |    |
         |  | RTA: BRISC (200 B)               |    |
         |  | RTA: NCRISC (200 B)              |    |
         |  | RTA: TRISC0 (80 B)               |    |
         |  | RTA: TRISC1 (80 B)               |    |
         |  | RTA: TRISC2 (80 B)               |    |
         |  | Kernel text: BRISC ELF (16 KB)   |    |
         |  | Kernel text: NCRISC ELF (12 KB)  |    |
         |  | Kernel text: TRISC0 ELF (8 KB)   |    |
         |  | Kernel text: TRISC1 ELF (12 KB)  |    |
         |  | Kernel text: TRISC2 ELF (8 KB)   |    |
         |  +----------------------------------+    |
~0x30000 +------------------------------------------+
         |  Circular buffer data regions:           |
         |  +----------------------------------+    |
         |  | CB0: in0 (8 KB, 4 tiles BF16)   |    |
         |  | CB1: in1 (8 KB, 4 tiles BF16)   |    |
         |  | CB2: accum (16 KB, 4 tiles FP32) |    |
         |  | CB3: out0 (4 KB, 2 tiles BF16)  |    |
         |  | CB24: intermed (4 KB)            |    |
         |  +----------------------------------+    |
~0x3A000 +------------------------------------------+
         |  Remaining L1 (free / allocatable)       |
         |                                          |
0x16E000 +------------------------------------------+  <-- MEM_L1_SIZE
```

The "capturable kernel state" region spans from `l1_unreserved_start` through the end of the last allocated CB data. In this example, that is approximately $\text{0x3A000} - \text{0x21000} = \text{0x19000} = 102{,}400$ bytes (~100 KB) of meaningful kernel state.

---

## 5. DRAM Buffer State

### 5.1 When DRAM State Matters

For compute-only debugging (TRISC replay), DRAM state is typically not needed: the input data is already in L1 circular buffers at the point where the compute kernel begins. However, data movement kernels (BRISC/NCRISC) read from and write to DRAM buffers, so their replay requires DRAM backing data.

The `Buffer` class (`tt_metal/api/tt-metalium/buffer.hpp`) stores:
- `buffer_type_`: `DRAM`, `L1`, or `SYSTEM_MEMORY`
- `size_`: total buffer size in bytes
- `page_size_`: size of each page
- `address_`: device address
- Shard configuration (for sharded buffers)

### 5.2 DRAM Buffer Size Ranges

| Workload | Typical DRAM Buffer Sizes | Notes |
|---|---|---|
| Small tensor (e.g., bias) | 1--64 KB | Single DRAM bank |
| Medium tensor (e.g., activation) | 64 KB--4 MB | Distributed across banks |
| Large tensor (e.g., weight matrix) | 4--256 MB | Sharded across many banks |
| Full model weights | 256 MB--4 GB | Not practical to snapshot |

### 5.3 Partial DRAM Capture Strategy

For data movement debugging, the snapshot should capture only the DRAM pages that the kernel will actually access during the target invocation. This can be determined from:

1. The kernel's runtime arguments (which often encode DRAM addresses and sizes).
2. The `EnqueueWriteBuffer` calls that preceded the kernel dispatch (captured by LightMetal).

For compute-only replay, DRAM capture can be skipped entirely. The snapshot should record DRAM buffer metadata (addresses, sizes, page configuration) but defer data capture to the DM replay use case.

---

## 6. Capture Strategies

Three capture strategies yield different snapshot sizes, each suited to different use cases.

### Strategy A: Minimal (Compute Kernel Only, No Firmware)

Capture only the kernel binaries, configuration, and CB data. Sufficient for single-TRISC replay.

$$S_A = S_{\text{binaries}} + S_{\text{config}} + S_{\text{CB data}}$$

| Scenario | Optimized Binaries | Debug Binaries |
|---|---|---|
| Simple eltwise | 40 KB | 115 KB |
| Typical matmul | 120 KB | 350 KB |
| Complex op | 330 KB | 560 KB |

### Strategy B: Full Kernel State (All L1 Regions Used by the Program)

Capture everything in the kernel config buffer region: semaphores, CB config, RTAs, kernel text, and CB data. Omit the system-reserved firmware region.

$$S_B = S_A + S_{\text{semaphores}} + S_{\text{extra L1 regions}}$$

| Scenario | Optimized Binaries | Debug Binaries |
|---|---|---|
| Simple eltwise | 42 KB | 117 KB |
| Typical matmul | 130 KB | 360 KB |
| Complex op | 340 KB | 575 KB |

### Strategy C: Full L1 Dump

Capture the entire L1 SRAM for maximum fidelity. This is the simplest approach and captures all state including firmware, mailboxes, and any non-obvious memory-mapped state.

| Architecture | Full L1 Dump Size |
|---|---|
| Wormhole B0 | 1,464 KB (1.43 MB) |
| Blackhole | 1,536 KB (1.5 MB) |
| Quasar | 4,096 KB (4 MB) |

### Sparse L1 Representation (Hybrid)

A hybrid approach captures only non-zero or non-firmware L1 regions as a set of `(address, length, data)` tuples. This avoids capturing uninitialized memory while maintaining full fidelity for initialized regions:

$$S_{\text{sparse}} = \sum_{\text{regions}} (8 + \text{length}_i) \text{ bytes}$$

where the 8-byte overhead per region accounts for address (4 bytes) and length (4 bytes). For a typical program that uses 200--500 KB of L1, the sparse representation saves 60--70% compared to a full dump while being more complete than Strategy A or B.

The `CircularBufferAllocator` in `ProgramImpl` tracks the `l1_regions` vector (start, end pairs) for CB allocations. Combined with the `ProgramConfig` offsets, the interceptor can identify exactly which L1 ranges need to be read.

---

## 7. Multi-Core Snapshot Estimates

For programs spanning multiple cores, the total snapshot scales linearly:

$$S_{\text{multi}} = \sum_{c=0}^{N_{\text{cores}}-1} S_c$$

However, significant deduplication is possible:
- **ELF binaries**: all cores in a kernel group run the same binaries. Store once, reference by hash.
- **Common runtime arguments**: identical across all cores in the range.
- **CB configuration**: identical across all cores (only the data contents differ).

With deduplication:

$$S_{\text{multi,dedup}} = S_{\text{shared}} + \sum_{c=0}^{N_{\text{cores}}-1} S_{\text{per-core-unique},c}$$

where $S_{\text{shared}}$ includes binaries, common RTAs, and CB configs, and $S_{\text{per-core-unique}}$ includes per-core RTAs and CB tile data.

| Multi-Core Scenario | Cores | Shared | Per-Core Unique | Total (dedup) | LZ4 Compressed |
|---|---|---|---|---|---|
| 8x8 matmul (BF16, typical CBs) | 64 | 200 KB | 64 x 32 KB = 2 MB | ~2.2 MB | ~600 KB |
| 8x8 matmul (BF16, large CBs) | 64 | 300 KB | 64 x 128 KB = 8 MB | ~8.3 MB | ~2.5 MB |
| 8x8 matmul (FP32, large CBs) | 64 | 300 KB | 64 x 256 KB = 16 MB | ~16.3 MB | ~5 MB |
| 8x8 full L1 dump (WH) | 64 | -- | 64 x 1,464 KB = 91.5 MB | 91.5 MB | ~25 MB |
| 16x16 large op (BF16) | 256 | 300 KB | 256 x 64 KB = 16 MB | ~16.3 MB | ~5 MB |

### Comparison with LightMetal

For reference, a typical LightMetal capture (which records host API calls but not L1 data or binaries) for a matmul test is approximately 50--200 KB. The kernel snapshot is 5--50x larger because it includes:
- Compiled ELF binaries (not captured by LightMetal)
- L1 tile data contents (not captured by LightMetal)
- Per-core memory state (LightMetal captures per-program, not per-core)

---

## 8. CB Pointer State

Circular buffer read and write pointers are maintained in L1 at addresses determined by the CB configuration. The CB hardware/firmware protocol uses these pointers for producer-consumer synchronization. At the moment of capture:

- **Pre-dispatch**: CB pointers are at their initial positions (typically 0/0 for read/write). Not critical to capture since they reset.
- **Post-mortem**: CB pointers indicate how far unpack progressed (read pointer) and how far data movement or pack progressed (write pointer). Valuable for diagnosing where execution stalled.

The CB pointer addresses are part of the CB config written to L1 (`local_cb_offset` and `remote_cb_offset` in `kernel_config_msg_t`). Each CB occupies `UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4` words (16 bytes) in the config region for local CBs and `UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2` words (8 bytes) for remote CBs (from `tt_metal/api/tt-metalium/circular_buffer_constants.h`).

**Size:** $64 \times 16 = 1{,}024$ bytes for all local CB configs on BH (where `NUM_CIRCULAR_BUFFERS = 64`).

---

## 9. Capturing Live Semaphore Values from L1

As noted in `02_runtime_configuration_state.md`, the semaphore **initial values** are stored in the `Semaphore` objects on the host. But by the time a capture occurs (especially post-mortem), the live values in L1 may differ. Each semaphore slot is a 32-bit word at a known L1 offset:

$$\text{semaphore addr}_{i} = \text{kernel config base} + \text{sem offset} + (i \times 4)$$

for $i \in \{0, 1, \ldots, 15\}$.

Reading these 64 bytes from L1 is fast (~1 $\mu$s per 4-byte word via NOC) and provides the actual semaphore state at capture time. For pre-dispatch capture, the values should equal the initial values (freshly initialized by firmware). For post-mortem capture, they reflect the state at the moment of failure.

---

## Capture Feasibility Summary

| State Element | Host Accessible? | Requires Halting? | Volatile? | Omission Impact |
|---|---|---|---|---|
| CB tile data in L1 | Yes -- via PCIe/NOC read | No for pre-dispatch; Yes for mid-execution | Stable pre-dispatch; volatile during execution | **Fatal** for compute replay -- no inputs |
| Full L1 memory dump | Yes -- via PCIe/NOC read | No for pre-dispatch; Yes for consistent mid-exec read | Stable pre-dispatch | Most thorough but largest capture |
| DRAM buffer contents | Yes -- via PCIe read | No | Stable if not being concurrently written | Required for DM replay; not needed for compute-only |
| Semaphore live values | Yes -- 64 bytes via NOC read | Ideally halt first for consistency | Volatile during execution | Post-mortem: needed to diagnose stalls |
| CB read/write pointers | Yes -- via NOC read of CB config region | No for pre-dispatch (just initialized) | Volatile during execution | Post-mortem: helps diagnose producer-consumer progress |
| L1 firmware regions | Yes -- via NOC read | No | Written once at firmware init | Usually not needed; helpful for full fidelity |
| Stack regions | Yes -- via NOC read | No for pre-dispatch | Volatile during execution | Post-mortem: enables stack trace reconstruction |

---

## Key Takeaways

- **Tile data in circular buffers dominates the capture envelope**, ranging from 8 KB (simple eltwise) to over 1 MB (large matmul with FP32 accumulators). The data format is the primary determinant: a BFloat16 tile is 2 KB, while a Float32 tile is 4 KB, and block floating-point formats (BFP8, BFP4, BFP2) include per-face exponent headers of 64 bytes.

- **The total L1 available for kernel use is approximately 1.4 MB on Wormhole, 1.46 MB on Blackhole, and 3.36 MB on Quasar**, after subtracting the system-reserved firmware, mailbox, and infrastructure regions. A full L1 dump is the simplest capture strategy but is rarely needed; a sparse representation capturing only allocated regions typically saves 60--70%.

- **Multi-core snapshots scale linearly with core count but benefit substantially from deduplication**: ELF binaries and common RTAs are shared across all cores in a kernel group, reducing a 64-core matmul snapshot from ~91 MB (full dump) to ~2--8 MB (deduplicated, selective capture) before compression.

- **Pre-dispatch capture is the cleanest moment for memory state**: the data is at rest, no concurrent writers exist, and semaphores are at their initial values. Post-mortem capture is messier (partially consumed CBs, modified semaphore values) but more valuable for failure diagnosis. The interceptor should support both modes.

---

**Previous:** [Runtime Configuration State](./02_runtime_configuration_state.md) | **Next:** [Implicit, Hidden, and Coordination State](./04_implicit_hidden_and_coordination_state.md)
