# 01 -- Tensix L1 Layout

## Context

Every Tensix core on a Tenstorrent chip has a private L1 SRAM that serves as its primary working memory -- there are no hardware caches, no coherence protocol, and no virtual memory translation. Data lives in L1 or DRAM; the NOC moves it between cores. Understanding the exact memory map of this L1 -- which regions are reserved for firmware, which are available for user buffers, and how much space fabric metadata consumes -- is essential for evaluating the feasibility of a global SRAM pool. This file documents the complete L1 SRAM layout for both Wormhole and Blackhole architectures, the `AllocatorConfig` model that governs runtime allocation, the buffer types available, and the `L1BankingAllocator` that manages the unreserved address space.

---

## 1. Physical L1 Parameters

| Parameter | Wormhole (WH) | Blackhole (BH) |
|-----------|:-------------:|:--------------:|
| L1 size per Tensix core | 1464 KB (1,499,136 B) | 1536 KB (1,572,864 B) |
| `MEM_L1_SIZE` hex | `0x16E000` | `0x180000` |
| Number of Tensix cores | 80 | 140 |
| Aggregate L1 across all Tensix cores | ~114.4 MB | ~210 MB |
| L1 base address | `0x0` | `0x0` |
| NCRISC IRAM | 16 KB at `0xFFC00000` (WH only) | No IRAM constraint |
| Local memory base (BRISC/NCRISC/TRISC) | `0xFFB00000` | `0xFFB00000` |
| BRISC local memory size | 4 KB | 8 KB |
| NCRISC local memory size | 4 KB | 8 KB |
| TRISC local memory size | 2 KB | 4 KB |

Both architectures define the L1 base at address `0x0` (`MEM_L1_BASE`). The entire L1 is byte-addressable from the RISC-V processors on the core and from the NOC. On Wormhole, NCRISC has an IRAM constraint: its kernel runs from a dedicated 16 KB instruction RAM at `0xFFC00000`, limiting NCRISC kernel size to 16 KB. Blackhole removes this constraint -- all five RISC-V processors (BRISC, NCRISC, TRISC0/1/2) can use the full L1 for kernel code.

---

## 2. Reserved Memory Map

The L1 address space from `0x0` through `MEM_MAP_END` is reserved for system use. Each region's base address is derived from the previous region's end address, creating a strict chain of dependencies. The canonical source is `dev_mem_map.h`, which uses `#define` macros so the same constants are available to firmware, host software, and linker scripts.

### 2.1 Memory Map Diagram

```
Address 0x0
 +---------------------------+
 | Boot Code Base (0x0)      |  4 bytes (NOC atomic return value at offset 4)
 +---------------------------+
 | L1 Barrier (0xC)          |  4 bytes
 +---------------------------+
 | [BH only: ARC FW Scratch] |  16 bytes (power throttling state)
 | [BH only: Inline Write]   |  64 bytes (NOC inline write emulation)
 +---------------------------+
 | Mailbox (variable base)   |  12,896 bytes (dev_msgs_t)
 +---------------------------+
 | Zeros Region              |  512 bytes (32-byte aligned)
 +---------------------------+
 | LLK Debug                 |  1,024 bytes
 +---------------------------+
 | BRISC Firmware             |  WH: 6 KB / BH: 8,704 B
 +---------------------------+
 | NCRISC Firmware            |  WH: 2,048 B / BH: 2,560 B
 +---------------------------+
 | TRISC0 Firmware            |  WH: 1,536 B / BH: 2,560 B
 +---------------------------+
 | TRISC1 Firmware            |  WH: 1,536 B / BH: 2,560 B
 +---------------------------+
 | TRISC2 Firmware            |  WH: 1,536 B / BH: 2,560 B
 +---------------------------+
 | NOC Counters               |  80 bytes (5 x 2 x 2 x 4)
 +---------------------------+
 | Fabric Counters            |  32 bytes (3 x 2 x 4 + 8 padding)
 +---------------------------+
 | Fabric Connection Lock     |  144 bytes
 +---------------------------+
 | Routing Table              |  2,576 bytes
 +---------------------------+
 | Fabric Connections         |  656 bytes
 +---------------------------+
 | Packet Header Pool         |  WH: 2,304 B / BH: 3,456 B
 +---------------------------+
 = MEM_MAP_END               =  WH: 0x8120 / BH: 0x9DE0
 +---------------------------+
 | (Scratch/Init area)        |  Reusable after FW init
 +---------------------------+
 | ... User-available L1 ... |
 +---------------------------+
 Address MEM_L1_SIZE
```

### 2.2 Computed Address Table -- Wormhole

| Region | Base Address | Size (Bytes) | Description |
|--------|:----------:|:--------:|-------------|
| Boot code | `0x00000` | 4 | `MEM_BOOT_CODE_BASE`, NOC atomic return value at offset 4, L1 barrier at offset 12 |
| Mailbox | `0x00010` | 12,896 | `MEM_MAILBOX_BASE` -- holds `dev_msgs_t` structure for host-device communication |
| Zeros region | `0x03280` | 512 | 32-byte-aligned region initialized to zero, used by firmware |
| LLK debug | `0x03480` | 1,024 | Low-level kernel debug space |
| BRISC firmware | `0x03880` | 6,144 | BRISC firmware code |
| NCRISC firmware | `0x05080` | 2,048 | NCRISC firmware code (kernel runs from IRAM on WH) |
| TRISC0 firmware | `0x05880` | 1,536 | TRISC0 firmware code |
| TRISC1 firmware | `0x05E80` | 1,536 | TRISC1 firmware code |
| TRISC2 firmware | `0x06480` | 1,536 | TRISC2 firmware code |
| NOC counters | `0x06A80` | 80 | 5 counter types x 2 NOCs x 2 DMs x 4 bytes |
| Fabric counters | `0x06AD0` | 32 | 3 barrier types x 2 DMs x 4 bytes + 8 bytes padding |
| Fabric connection lock | `0x06AF0` | 144 | Lock (4) + initialized (4) + connection object (128) + padding (8) |
| Routing table | `0x06B80` | 2,576 | `MEM_TENSIX_ROUTING_TABLE_BASE` -- base(516) + routing paths(1024) + exit node table(1024) + padding(12) |
| Fabric connections | `0x07590` | 656 | `MEM_TENSIX_FABRIC_CONNECTIONS_BASE` -- `tensix_fabric_connections_l1_info_t` |
| Packet header pool | `0x07820` | 2,304 | 144 B x 4 directions x 2 conventions x 2 DMs (EAST, WEST, NORTH, SOUTH) |
| **MEM_MAP_END** | **`0x08120`** | -- | **33,056 bytes (~32.3 KB) total reserved** |

### 2.3 Computed Address Table -- Blackhole

| Region | Base Address | Size (Bytes) | Description |
|--------|:----------:|:--------:|-------------|
| ARC FW scratch | `0x00010` | 16 | Power throttling state used by ARC firmware |
| Inline write base | `0x00020` | 64 | Workaround for BH inline write hang: 16 B per NOC x 2 NOCs x 2 processors |
| Mailbox | `0x00060` | 12,896 | `MEM_MAILBOX_BASE` -- same `dev_msgs_t` structure |
| Zeros region | `0x032C0` | 512 | 32-byte-aligned zeros |
| LLK debug | `0x034C0` | 1,024 | Low-level kernel debug space |
| BRISC firmware | `0x038C0` | 8,704 | 6 KB + 2560 B (larger than WH) |
| NCRISC firmware | `0x05AC0` | 2,560 | Larger than WH (no IRAM; firmware runs from L1) |
| TRISC0 firmware | `0x064C0` | 2,560 | Larger than WH |
| TRISC1 firmware | `0x06EC0` | 2,560 | Larger than WH |
| TRISC2 firmware | `0x078C0` | 2,560 | Larger than WH |
| NOC counters | `0x082C0` | 80 | Same structure as WH |
| Fabric counters | `0x08310` | 32 | Same structure as WH |
| Fabric connection lock | `0x08330` | 144 | Same structure as WH |
| Routing table | `0x083C0` | 2,576 | Same layout: base + routing paths + exit nodes + padding |
| Fabric connections | `0x08DD0` | 656 | Same `tensix_fabric_connections_l1_info_t` |
| Packet header pool | `0x09060` | 3,456 | 144 B x 6 directions x 2 conventions x 2 DMs (adds UP, DOWN vs WH's 4 directions) |
| **MEM_MAP_END** | **`0x09DE0`** | -- | **40,416 bytes (~39.5 KB) total reserved** |

> **Key Insight:** Blackhole has a larger packet header pool (3,456 B vs 2,304 B on WH) because it supports six routing directions (EAST, WEST, NORTH, SOUTH, UP, DOWN) compared to WH's four directions. This reflects BH's native 2D mesh routing capability -- a structural advantage for the global SRAM pool proposal.

### 2.4 Blackhole Inline Write Workaround

On Blackhole, issuing inline writes and atomics requires all four memory ports to accept the transaction simultaneously. If one port has no back-pressure, the transaction hangs because there is no mechanism for one port to proceed independently. The workaround is to emulate inline writes by first writing the value to a dedicated L1 scratch location (`MEM_L1_INLINE_BASE` at offset `0x20`), then issuing a NOC async write from that location. Each NOC gets 16 bytes of scratch space per RISC processor pair, consuming 64 bytes total. This hardware limitation is relevant when evaluating NOC inline write transaction performance on BH (see file 03).

---

## 3. Firmware Code Regions

Each Tensix core has five RISC-V processors: BRISC, NCRISC, TRISC0, TRISC1, and TRISC2. The firmware for each processor is loaded into a dedicated region of L1:

```cpp
// tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h
#define MEM_BRISC_FIRMWARE_SIZE  (6 * 1024)      // 6 KB
#define MEM_NCRISC_FIRMWARE_SIZE 2048             // 2 KB
#define MEM_TRISC0_FIRMWARE_SIZE 1536             // 1.5 KB
#define MEM_TRISC1_FIRMWARE_SIZE 1536             // 1.5 KB
#define MEM_TRISC2_FIRMWARE_SIZE 1536             // 1.5 KB
```

On Wormhole, the NCRISC has a separate 16 KB IRAM (`MEM_NCRISC_IRAM_BASE = 0xFFC00000`) for kernel code, while the firmware region in L1 is only 2 KB. On Blackhole, there is no IRAM constraint -- all processors can use L1 for kernel code up to the `MEM_MAX_KERNEL_SIZE` limit.

Maximum kernel sizes:

| Architecture | `MEM_MAX_KERNEL_SIZE` | Derivation |
|:------------:|:--------------------:|:----------:|
| WH | 1,466,368 B (1432 KB) | Hard-coded define: `1432 * 1024 = 1,466,368` |
| BH | 1,532,928 B (1497 KB) | Hard-coded define: `1497 * 1024 = 1,532,928` |

These values are hard-coded round-number defines in the source and do not equal `MEM_L1_SIZE - MEM_MAP_END`. They are enforced by the ELF loader (`tt_elffile.cpp`). The practical constraint is the `kernel_config_buffer` aggregate limit, configurable via `worker_l1_size` in `MeshDevice::create_unit_meshes`.

### Usable Space After Reserved Regions

The actual usable (unreserved) L1 is computed from `MEM_L1_SIZE - MEM_MAP_END`, which differs slightly from the hard-coded `MEM_MAX_KERNEL_SIZE` values above:

```math
\text{WH: L1 unreserved} = \texttt{MEM\_L1\_SIZE} - \texttt{MEM\_MAP\_END} = 1{,}499{,}136 - 33{,}056 = 1{,}466{,}080 \text{ bytes} \approx 1431.7 \text{ KB}
```

```math
\text{BH: L1 unreserved} = \texttt{MEM\_L1\_SIZE} - \texttt{MEM\_MAP\_END} = 1{,}572{,}864 - 40{,}416 = 1{,}532{,}448 \text{ bytes} \approx 1496.5 \text{ KB}
```

Note: `MEM_MAX_KERNEL_SIZE` (WH: 1,466,368 B, BH: 1,532,928 B) is slightly different from the unreserved space (WH: 1,466,080 B, BH: 1,532,448 B) because the former is a hard-coded round-number define rather than a computed value.

---

## 4. Fabric Metadata Overhead

A significant portion of the reserved space is dedicated to fabric routing infrastructure. These regions are relatively recent additions to the memory map, reflecting the transition to the persistent TT-Fabric infrastructure described in Chapter 1, File 04. They consume a non-trivial amount of per-core L1:

### 4.1 Per-Component Breakdown

| Component | Size (Bytes) | Purpose |
|-----------|:----------:|---------|
| Fabric counters | 32 | 3 barrier types x 2 DMs x 4 bytes + 8 padding. Track fabric-level transaction barriers. |
| Fabric connection lock | 144 | Lock (4) + initialized (4) + connection object (128) + padding (8). Synchronizes BRISC/NCRISC connection setup. |
| Routing table | 2,576 | Base header (516 B) + routing paths union (1,024 B) + exit node table (1,024 B) + padding (12 B) |
| Fabric connections | 656 | `tensix_fabric_connections_l1_info_t` -- pre-computed connection info for worker-to-fabric communication |
| Packet header pool | 2,304 (WH) / 3,456 (BH) | Pre-allocated packet headers: 144 B x directions x 2 conventions x 2 DMs |

### 4.2 Routing Table Internal Structure

The routing table has three components:
- **Base metadata** (516 bytes): Contains the `routing_table_t` header with routing mode, chip coordinates, and configuration
- **Routing paths** (1,024 bytes, union): Either 1D routing paths (`64 chips x 16 bytes`) or 2D compressed routing paths. The 1D and 2D tables share the same memory offset as a union.
- **Exit node table** (1,024 bytes): Lookup table for finding exit nodes when routing from one mesh to another (`exit_node_table_t`)

### 4.3 Per-Core Totals and Aggregate Impact

**Total fabric metadata per Tensix core:**
- **Wormhole:** 5,712 bytes (~5.6 KB)
- **Blackhole:** 6,864 bytes (~6.7 KB)

This overhead is unavoidable for any core that participates in fabric communication. For a GSRP that requires every Tensix core to have routing capability, this metadata footprint scales linearly with core count:

| Metric | Wormhole (80 cores) | Blackhole (140 cores) |
|--------|:------------------:|:--------------------:|
| Per-chip fabric overhead | 456,960 B (~446 KB) | 960,960 B (~938 KB) |
| As % of aggregate L1 | ~0.38% | ~0.44% |

> **Key Insight:** The total fabric metadata overhead per Tensix core is a fixed cost of enabling fabric communication from any Tensix core. A GSRP design that requires additional per-core metadata (e.g., address translation tables, remote connection state) would further reduce the available space. On a 32-chip BH system, even a modest 1 KB per core for translation tables would consume 32 x 140 x 1,024 = 4.4 MB of aggregate L1 -- a small fraction of the 6.4 GB total but a non-trivial reduction in per-core working memory for computationally intensive kernels.

---

## 5. `MEM_MAP_END` and `L1_UNRESERVED_BASE`

### 5.1 `MEM_MAP_END`

`MEM_MAP_END` marks the boundary between system-reserved L1 and the region available for user kernels and data:

```cpp
// Both WH and BH
#define MEM_MAP_END (MEM_PACKET_HEADER_POOL_BASE + MEM_PACKET_HEADER_POOL_SIZE)
```

The computed `MEM_MAP_END` values and derived usable L1 space are shown in the address tables in Sections 2.2 and 2.3, with the full derivation in Section 3.

### 5.2 Scratch Regions (Transient)

Immediately after `MEM_MAP_END` lies a scratch region used during firmware initialization. This region holds temporary copies of processor local memory, bank-to-NOC mapping arrays, and logical-to-virtual coordinate mappings. Once firmware initialization completes, this space becomes available as part of the unreserved L1:

```
MEM_MAP_END
 +---> BRISC init local (WH: 4 KB, BH: 8 KB)
 +---> NCRISC init local (WH: 4 KB, BH: 8 KB)
 +---> TRISC0/1/2 init local (WH: 2 KB each, BH: 4 KB each)
 +---> NCRISC IRAM scratch (WH: 4 KB, BH: 8 KB)
 +---> Bank-to-NOC mapping (2 KB)
 +---> Logical-to-virtual coord mapping (WH: 24 B, BH: 32 B)
```

### 5.3 `L1_UNRESERVED_BASE`

The actual start of user-available L1 is `L1_UNRESERVED_BASE`, which is set by the runtime in the `AllocatorConfig`. At minimum, it equals `MEM_MAP_END` (the scratch region is reusable after initialization), but the runtime may increase it to account for command queue buffers on dispatch cores or other per-device overhead:

```cpp
// tt_metal/impl/allocator/allocator_types.hpp
struct AllocatorConfig {
    uint32_t l1_unreserved_base = 0;  // Set by runtime
    size_t worker_l1_size = 0;        // Effective L1 capacity for allocation
    // ...
};
```

In `L1BankingAllocator::generate_config`, the `l1_unreserved_base` is set to the `worker_l1_unreserved_start` parameter aligned up to DRAM alignment. This value incorporates `MEM_MAP_END` plus any additional dispatch or configuration overhead.

---

## 6. The `AllocatorConfig` Model

The `AllocatorConfig` structure (defined in `tt_metal/impl/allocator/allocator_types.hpp`) decouples the allocator from the physical SOC descriptor. It encapsulates all the parameters needed for the L1 and DRAM allocators:

```cpp
// tt_metal/impl/allocator/allocator_types.hpp
struct AllocatorConfig {
    // DRAM configuration
    size_t num_dram_channels = 0;
    size_t dram_bank_size = 0;
    std::vector<size_t> dram_bank_offsets;
    uint32_t dram_unreserved_base = 0;
    uint32_t dram_alignment = 0;

    // Worker L1 configuration
    uint32_t l1_unreserved_base = 0;       // First usable address (>= MEM_MAP_END)
    CoreRangeSet worker_grid;               // Set of worker cores with L1 banks
    size_t worker_l1_size = 0;              // Usable L1 per core (may be < MEM_L1_SIZE)
    size_t l1_small_size = 0;              // Size reserved for L1_SMALL buffer type
    size_t trace_region_size = 0;          // Size reserved for trace buffers
    BankMapping l1_bank_remap;             // Remapping table for L1 bank assignment
    CoreRangeSet compute_grid;             // Compute-only cores (no L1 banking)
    uint32_t l1_alignment = 0;            // Alignment for L1 allocations (16 B)
    bool disable_interleaved = false;     // Disable interleaved buffer allocations

    // Core type classification
    std::unordered_map<CoreCoord, AllocCoreType> core_type_from_noc_coord_table;

    // Coordinate mapping
    std::unordered_map<int, int> worker_log_to_virtual_routing_x;
    std::unordered_map<int, int> worker_log_to_virtual_routing_y;
};
```

Key fields for understanding L1 space management:

| Field | Meaning |
|-------|---------|
| `l1_unreserved_base` | The first byte available for user allocation. Set to `MEM_MAP_END` or later. |
| `worker_l1_size` | Total L1 that the allocator manages per core. Can be less than `MEM_L1_SIZE` if the device reserves additional space (e.g., for dispatch). Configurable via `MeshDevice::create_unit_meshes`. |
| `l1_small_size` | A dedicated allocation pool for small buffers (`BufferType::L1_SMALL`), carved out at the top of the L1 address space. Reduces fragmentation from many small allocations mixed with large ones. |
| `trace_region_size` | Space reserved at the top of L1 for trace capture buffers (`BufferType::TRACE`). |
| `l1_alignment` | All L1 allocations are aligned to this boundary (16 bytes, matching the NOC word size). |
| `l1_bank_remap` | An optional remapping table that reassigns which physical core corresponds to which logical L1 bank. Used when the default row-major bank assignment is suboptimal for a particular workload. |
| `core_type_from_noc_coord_table` | Maps NOC coordinates to `AllocCoreType`. Only `ComputeAndStore` cores contribute L1 banks. |

### Core Type Classification

```cpp
enum class AllocCoreType {
    Dispatch,         // Reserved for command dispatch -- no user banking
    ComputeOnly,      // Can run compute kernels but not used for L1 banking
    ComputeAndStore,  // Full participation in L1 banking and compute
    Invalid,          // Not a valid core (harvested or non-existent)
};
```

Dispatch cores are excluded from the L1 banking pool because their L1 is used for the command queue. The number of dispatch cores depends on `num_hw_cqs` and the dispatch architecture. `ComputeOnly` cores can execute kernels but their L1 is not included in the interleaved buffer pool -- this distinction allows certain cores to be reserved for specific purposes without fragmenting the banking scheme.

---

## 7. Buffer Types

The `BufferType` enum (defined in `tt_metal/api/tt-metalium/buffer_types.hpp`) defines five buffer storage types:

```cpp
// tt_metal/api/tt-metalium/buffer_types.hpp
enum class BufferType {
    DRAM,           // Allocated in off-chip DRAM
    L1,             // Allocated in Tensix L1 SRAM (main pool)
    SYSTEM_MEMORY,  // Allocated in host system memory (accessed via PCIe)
    L1_SMALL,       // Allocated in the dedicated small-buffer L1 pool
    TRACE,          // Allocated in the trace region at top of L1
};
```

### L1 vs L1_SMALL

The main `L1` region and the `L1_SMALL` region are both physically in the same SRAM, but managed by separate `BankManager` instances with different base addresses and sizes:

```
+-------------------+ 0x0
|  System Reserved  |
|  (MEM_MAP_END)    |
+-------------------+ l1_unreserved_base
|                   |
|   L1 (main)       |  allocatable_l1_size = worker_l1_size - l1_unreserved_base - l1_small_size
|                   |
+-------------------+ worker_l1_size - l1_small_size
|   L1_SMALL        |  l1_small_size (configurable, default 0)
+-------------------+ worker_l1_size
```

The `L1_SMALL` buffer type addresses a practical problem: when large tensors and small metadata (semaphores, runtime arguments) compete for the same allocator, fragmentation wastes significant space. The `l1_small_size` region is carved from the top of the L1 address range and managed by a separate allocator. Small allocations go to `L1_SMALL`; large tensor buffers go to `L1`. When `l1_small_size = 0`, the entire unreserved space is available for `L1` allocations.

### TRACE Buffers

Trace buffers capture sequences of commands for replay. They occupy a dedicated region at the top of L1 (above the `L1_SMALL` region), and their size is specified by `trace_region_size` in the allocator config. During trace capture, dispatch commands are recorded to this region; during trace replay, the dispatch hardware reads from it.

### `TensorMemoryLayout`

The `TensorMemoryLayout` enum controls how multi-page tensors are distributed across L1 banks:

```cpp
enum class TensorMemoryLayout {
    INTERLEAVED = 0,     // Pages striped round-robin across all banks
    HEIGHT_SHARDED = 2,  // Rows assigned to cores
    WIDTH_SHARDED = 3,   // Columns assigned to cores
    BLOCK_SHARDED = 4,   // 2D blocks assigned to cores
};
```

For **interleaved** buffers, pages are distributed round-robin across all L1 banks. The address of page $p$ on bank $b$ is:

```math
\text{addr}(p, b) = \texttt{l1\_unreserved\_base} + \texttt{bank\_offset}(b) + \lfloor p / \texttt{num\_banks} \rfloor \times \texttt{page\_size}
```

where $b = p \mod \texttt{num\_banks}$.

For **sharded** buffers, pages are assigned to specific cores based on the sharding strategy. `HEIGHT_SHARDED` assigns tensor rows to cores, `WIDTH_SHARDED` assigns columns, and `BLOCK_SHARDED` assigns 2D blocks. The sharding assignment is determined by the `ShardOrientation` (row-major or column-major) and `ShardDistributionStrategy` (round-robin or nearest-neighbor).

---

## 8. The `L1BankingAllocator`

The `L1BankingAllocator` (defined in `tt_metal/impl/allocator/l1_banking_allocator.hpp`) manages the unreserved L1 space across all worker cores:

```cpp
// tt_metal/impl/allocator/l1_banking_allocator.hpp
class L1BankingAllocator : public AllocatorImpl {
public:
    explicit L1BankingAllocator(const AllocatorConfig& alloc_config);
    static AllocatorConfig generate_config(
        ChipId device_id,
        uint8_t num_hw_cqs,
        size_t l1_small_size,
        size_t trace_region_size,
        size_t worker_l1_unreserved_start,
        BankMapping l1_bank_remap);
};
```

### Bank Assignment and Shuffling

Each `ComputeAndStore` core contributes one L1 bank. Bank IDs are assigned to cores in a configurable order. The default behavior shuffles bank assignment with a fixed seed to diversify NOC traffic patterns:

```cpp
// l1_banking_allocator.cpp
std::vector<uint32_t> shuffled_bank_id = {};
if (not config_->l1_bank_remap.empty()) {
    // Use explicit remap from configuration
    std::copy(config_->l1_bank_remap.begin(), config_->l1_bank_remap.end(),
              std::back_inserter(shuffled_bank_id));
} else {
    // Randomize with fixed seed for deterministic behavior
    for (uint32_t id = 0; id < num_l1_banks; id++) {
        shuffled_bank_id.push_back(id);
    }
    auto rng = std::default_random_engine(0);
    std::shuffle(std::begin(shuffled_bank_id), std::end(shuffled_bank_id), rng);
}
```

The shuffled bank assignment diversifies the physical core locations that service consecutive bank IDs, improving NOC traffic distribution for interleaved tensors. When a user reads page 0 followed by page 1 of an interleaved tensor, the reads go to different physical cores rather than adjacent ones, reducing hot spots on the NOC mesh.

### Bank Managers

Two `BankManager` instances are created:

1. **L1 Manager** -- manages the main allocatable region:

```cpp
uint64_t allocatable_l1_size = worker_l1_size - l1_unreserved_base - l1_small_size;
l1_manager_ = std::make_unique<BankManager>(
    BufferType::L1,
    bank_id_to_bank_offset,
    allocatable_l1_size,
    interleaved_address_limit,
    config_->l1_alignment,
    config_->l1_unreserved_base,
    config_->disable_interleaved);
```

2. **L1 Small Manager** -- manages the L1_SMALL region:

```cpp
l1_small_manager_ = std::make_unique<BankManager>(
    BufferType::L1_SMALL,
    small_bank_id_to_bank_offset,
    config_->l1_small_size,
    small_interleaved_address_limit,
    config_->l1_alignment,
    small_alloc_offset,
    config_->disable_interleaved);
```

### The `generate_config` Pipeline

The `generate_config` static method constructs an `AllocatorConfig` from device-specific parameters:

1. Queries the device's SOC descriptor for core grid dimensions and DRAM channel count
2. Classifies each core as `Dispatch`, `ComputeOnly`, or `ComputeAndStore` based on dispatch core assignment
3. Computes `l1_unreserved_base` accounting for command queue L1 overhead on dispatch cores
4. Sets `worker_l1_size` based on the effective per-core L1 capacity
5. Applies the bank remapping table (shuffled or explicit)

### L1 Address Space Partitioning

For a typical worker core, the L1 address space is divided as follows:

```
0x0                         l1_unreserved_base              worker_l1_size
 |--- System Reserved ---|-------- Main L1 Pool ----------|
                          |                                |
                          |  Circular buffer space (bottom)|
                          |  Tensor data (interleaved)     |
                          |  Sharded buffer regions        |
                          |  Runtime arguments             |
                          |  Kernel config buffers         |
                          |  ---- L1_SMALL (top) ------   |
                          |  ---- TRACE (top-most) ----   |
```

---

## 9. Aggregate L1 Capacity and GSRP Implications

For a multi-chip system, the aggregate usable L1 determines the total "global SRAM pool" capacity:

| Configuration | Chips | Cores/Chip | Usable L1/Core | Aggregate Usable |
|:-----------|:-----:|:----------:|:---------:|:----------:|
| Single WH | 1 | 80 | ~1,432 KB | ~111.9 MB |
| WH T3000 (8 chips) | 8 | 80 | ~1,432 KB | ~895.0 MB |
| Single BH | 1 | 140 | ~1,497 KB | ~204.6 MB |
| BH 32-chip mesh | 32 | 140 | ~1,497 KB | ~6.4 GB |

These are theoretical upper bounds. Actual usable space is lower because:

1. **Kernel code:** BRISC/TRISC kernel binaries loaded into L1 (variable, can be large)
2. **Circular buffers:** Staging buffers for data movement, typically 2-8 pages x page size
3. **Dispatch overhead:** Command queue buffers on dispatch cores (2-4 cores reserved per chip)
4. **Runtime arguments:** Packed into L1 alongside kernel code
5. **Local variables:** Stack and global data for each RISC-V processor

> **Key Insight:** A 32-chip Blackhole system has approximately 6.4 GB of aggregate L1 across 4,480 Tensix cores. However, each additional cross-chip communication mechanism (address translation tables, connection state, buffering for in-transit packets) would reduce the per-core usable L1. With the existing fabric metadata already consuming ~6.7 KB per BH core, the budget for additional GSRP overhead is constrained -- perhaps 1-2 KB per core for translation tables before meaningfully impacting workload capacity.

---

## Key Takeaways

- **L1 is scratchpad, not cache.** There is no coherence protocol, no virtual memory, and no hardware-managed data movement. All data placement and movement is software-managed. This simplifies the consistency model for a GSRP (no coherence to maintain) but means all cross-chip transfers must be explicitly initiated.
- **System-reserved space is substantial and growing.** Wormhole reserves ~32.3 KB per Tensix core (`MEM_MAP_END` = 0x8120); Blackhole reserves ~39.5 KB (`MEM_MAP_END` = 0x9DE0). The growth is driven by fabric metadata (routing tables, connection info, packet headers) that enables 2D mesh routing. Every region's address is derived from the previous region's end, making the memory map a strict linear chain -- adding or expanding any region shifts all subsequent addresses.
- **Fabric metadata consumes ~5.6 KB per Tensix core on WH and ~6.7 KB on BH.** This includes the routing table (2,576 B), fabric connections metadata (656 B), and packet header pool (2,304 B on WH / 3,456 B on BH). Any additional GSRP metadata would add to this fixed overhead.
- **The allocator model is bank-based with shuffled assignment.** `L1BankingAllocator` treats each core's unreserved L1 as an independent bank, with interleaved buffer pages striped across banks in a deterministically-shuffled order for NOC traffic distribution. A global address space would need to either extend this banking model across chips or replace it with a new allocation scheme.
- **Five buffer types and four tensor memory layouts partition the L1 address space.** The `TensorMemoryLayout` enum (INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED) determines page-to-core mapping. A GSRP would introduce the concept of "remote L1" -- data that physically resides on another chip's L1 but is logically addressable from the local core.

## Source Code References

- `tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h` -- Wormhole L1 memory map constants
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` -- Blackhole L1 memory map constants
- `tt_metal/impl/allocator/allocator_types.hpp` -- `AllocatorConfig`, `AllocCoreType`, `BankMapping`
- `tt_metal/impl/allocator/l1_banking_allocator.hpp` -- `L1BankingAllocator` class
- `tt_metal/impl/allocator/l1_banking_allocator.cpp` -- Bank shuffling implementation, `BankManager` construction
- `tt_metal/api/tt-metalium/buffer_types.hpp` -- `BufferType`, `TensorMemoryLayout`, `ShardOrientation`

---

**Next:** [`02_ethernet_l1_layout.md`](./02_ethernet_l1_layout.md)
