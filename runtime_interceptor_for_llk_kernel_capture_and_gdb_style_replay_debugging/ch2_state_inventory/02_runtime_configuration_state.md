# Runtime Configuration State

This file inventories the second category of kernel capture state: runtime configuration that parameterizes and launches the kernel. Unlike the compiled binary state (which is fixed at JIT-compilation time), runtime configuration changes on every dispatch -- runtime arguments may differ per core and per invocation, circular buffer addresses shift as programs are allocated, and the launch message carries per-invocation metadata.

All runtime configuration state is assembled progressively during program finalization and dispatch. Some fields are set early (at `CreateKernel` time), others computed during `ProgramImpl::finalize_offsets()`, and the final launch and go messages are assembled only during `program_dispatch::assemble_device_commands()`. The capture moment matters: the interceptor must hook **after** `finalize_offsets()` to see fully resolved configuration.

---

## 1. Runtime Arguments

Runtime arguments are the primary mechanism for passing per-invocation, per-core parameters to device kernels. They are the most dynamic piece of configuration state.

### 1.1 Per-Core Runtime Arguments

Stored in the `Kernel` object:

```cpp
// Kernel class (tt_metal/impl/kernels/kernel.hpp), protected members:
std::vector<std::vector<std::vector<uint32_t>>> core_to_runtime_args_;
std::vector<std::vector<RuntimeArgsData>> core_to_runtime_args_data_;
```

The outer two dimensions index by core coordinate (y, x). Each core receives up to `max_runtime_args = 341` uint32 values (defined in `tt_metal/api/tt-metalium/kernel_types.hpp`). This limit derives from the dispatch packet size:

$$\text{max runtime args} = \left\lfloor \frac{4096}{3 \times 4} \right\rfloor = 341$$

where 4096 is the dispatch packet size and 3 is the number of kernel processors per Tensix core.

The `RuntimeArgsData` wrapper (`tt_metal/api/tt-metalium/runtime_args_data.hpp`) provides an indirection layer:

```cpp
struct RuntimeArgsData {
    std::uint32_t* rt_args_data;   // 8 B (pointer)
    std::size_t rt_args_count;     // 8 B
};
```

Initially `rt_args_data` points into `core_to_runtime_args_`; after command queue generation, it is redirected to point into the CQ command buffer for zero-copy updates. For capture, the underlying uint32 array (not the pointer) is what matters.

### 1.2 Common Runtime Arguments

Shared across all cores running this kernel:

```cpp
std::vector<uint32_t> common_runtime_args_;
RuntimeArgsData common_runtime_args_data_{};
uint32_t common_runtime_args_count_{0};
```

Common runtime args are written once and broadcast to all cores in the kernel's `CoreRangeSet`, reducing dispatch bandwidth for parameters that are uniform across the grid.

### 1.3 The RTA Sentinel Values

When watcher asserts are enabled, uninitialized RTA slots are filled with sentinel values:

```cpp
// tt_metal/hw/inc/hostdev/rta_constants.h
constexpr uint16_t RTA_CRTA_NO_ARGS_SENTINEL = 0xFFFF;
constexpr uint32_t WATCHER_RTA_UNSET_PATTERN = 0xBEEF0000;
```

If `RTA_CRTA_NO_ARGS_SENTINEL` is found in `rta_offset_t::rta_offset` or `rta_offset_t::crta_offset`, no runtime arguments are present for that processor.

### 1.4 Validation Tracking

The `Kernel` class tracks validation metadata:

```cpp
std::size_t max_runtime_args_per_core_{0};
CoreCoord core_with_max_runtime_args_;
std::set<CoreCoord> core_with_runtime_args_;
```

### 1.5 Capture Requirements and Size Estimate

- Per-core args: for each core in `core_with_runtime_args_`, capture the full `uint32_t` vector. A typical kernel uses 10--50 runtime args per core, but complex data movement kernels can use up to the maximum 341.
- Common args: capture `common_runtime_args_` and `common_runtime_args_count_`.
- The `RuntimeArgsData` wrapper itself does not need to be captured -- it is a pointer indirection into the vectors above.

**Size estimate per core:**

$$\text{Per-core RTA size} = \text{num args} \times 4 \text{ bytes} = 40\text{--}200 \text{ bytes typical, } 1364 \text{ bytes max}$$

For a kernel spanning 64 cores with 50 args each: $64 \times 200 = 12{,}800$ bytes (~12.5 KB). For the worst case with 341 args: $64 \times 1364 = 87{,}296$ bytes (~85 KB).

---

## 2. Circular Buffer Configuration

Circular buffers are the tile-granular communication channels between processors within a Tensix core. Their configuration is critical for replay because it determines how the unpack, math, and pack stages find and interpret tile data in L1.

### 2.1 `CircularBufferConfig`

Source: `tt_metal/api/tt-metalium/circular_buffer_config.hpp`.

| Field | C++ Type | Size | Role |
|---|---|---|---|
| `total_size_` | `uint32_t` | 4 B | Total size of the circular buffer region in bytes |
| `globally_allocated_address_` | `std::optional<uint32_t>` | 8 B | If set, CB is backed by a pre-allocated `Buffer` object |
| `data_formats_` | `std::array<std::optional<DataFormat>, NUM_CIRCULAR_BUFFERS>` | $64 \times 2$ B = 128 B (BH) | Data format per CB index (DataFormat is a uint8 enum) |
| `page_sizes_` | `std::array<std::optional<uint32_t>, NUM_CIRCULAR_BUFFERS>` | $64 \times 8$ B = 512 B (BH) | Page size in bytes per CB index |
| `tiles_` | `std::array<std::optional<Tile>, NUM_CIRCULAR_BUFFERS>` | variable | Tile geometry (num faces, face dims) per CB index |
| `buffer_indices_` | `std::unordered_set<uint8_t>` | variable | Set of active CB indices |
| `local_buffer_indices_` | `std::unordered_set<uint8_t>` | variable | CB indices that are local (on-core) |
| `remote_buffer_indices_` | `std::unordered_set<uint8_t>` | variable | CB indices that are remote (cross-core) |
| `dynamic_cb_` | `bool` | 1 B | Whether this is a dynamically allocated CB |
| `max_size_` | `uint32_t` | 4 B | Maximum allowable size (for dynamic CBs) |
| `buffer_size_` | `uint32_t` | 4 B | Backing buffer size |
| `address_offset_` | `uint32_t` | 4 B | Offset within the backing buffer |

**`NUM_CIRCULAR_BUFFERS`** is architecture-dependent:

| Architecture | `NUM_CIRCULAR_BUFFERS` | Notes |
|---|---|---|
| Wormhole | 32 | Limited by 2 KB TRISC memory |
| Blackhole | 64 | Full CB count |
| Host (compile-time) | 64 | Maximum across all architectures; used to size arrays uniformly |

Source: `tt_metal/api/tt-metalium/circular_buffer_constants.h`.

Constants for CB config serialization:

```cpp
constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;
constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;
```

### 2.2 `CircularBufferImpl`

Source: `tt_metal/impl/buffers/circular_buffer.hpp`.

| Field | C++ Type | Size | Role |
|---|---|---|---|
| `id_` | `uintptr_t` (via `CBHandle`) | 8 B | Unique identifier for this CB |
| `core_ranges_` | `CoreRangeSet` | variable | Cores where this CB is allocated |
| `config_` | `CircularBufferConfig` | ~700 B | Full configuration (see above) |
| `locally_allocated_address_` | `std::optional<uint32_t>` | 8 B | Address assigned by program allocator |
| `globally_allocated_address_` | `uint32_t` | 4 B | Address from global buffer |
| `global_circular_buffer_config_address_` | `DeviceAddr` | 8 B | Config address for global CBs |

The `locally_allocated_address_` is computed during `ProgramImpl::allocate_circular_buffers()` and represents the actual L1 address where the CB data begins. This address is **critical for replay** -- the captured memory contents must be mapped to the correct L1 address.

### 2.3 Per-Core CB Tracking in `ProgramImpl`

Source: `tt_metal/impl/program/program_impl.hpp`.

```cpp
std::vector<std::shared_ptr<CircularBufferImpl>> circular_buffers_;
std::unordered_map<CBHandle, std::shared_ptr<CircularBufferImpl>> circular_buffer_by_id_;
std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>> per_core_cb_indices_;
std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>> per_core_local_cb_indices_;
std::unordered_map<CoreCoord, std::bitset<NUM_CIRCULAR_BUFFERS>> per_core_remote_cb_indices_;
```

These bitsets track which CB indices are in use on each core. `NUM_CIRCULAR_BUFFERS = 64` on host, so each bitset is 8 bytes.

The `local_cb_mask` field in `kernel_config_msg_t` (see Section 5) is derived from `per_core_local_cb_indices_` and tells the firmware which CBs are local vs. remote.

**Capture requirements:**

- For each CB in the program: `CircularBufferConfig` (all fields), `core_ranges`, allocated address, buffer indices.
- The per-core bitset of active CB indices (from `per_core_cb_indices_`) determines which CB config entries are meaningful.
- The `locally_allocated_address_` is critical -- it is the L1 address where the CB resides and where tile data must be placed for replay.

**Size estimate per CB config:** Each `CircularBufferConfig` with all optional arrays populated:

$$\text{CB config size} = 4 + 4 + (64 \times 1) + (64 \times 4) + (64 \times \text{sizeof(Tile)}) + \text{set overhead} \approx 400\text{--}800 \text{ bytes}$$

For a typical program with 8--16 CBs: ~3--12 KB of CB configuration.

---

## 3. `ProgramConfig`: The L1 Memory Map

Source: `tt_metal/impl/program/program_impl.hpp`.

`ProgramConfig` describes the memory layout within a core's L1 for one program invocation. It is computed by `ProgramImpl::finalize_offsets()`:

| Field | C++ Type | Size | Role |
|---|---|---|---|
| `rta_offset` | `uint32_t` | 4 B | Offset of runtime args region within the kernel config buffer |
| `sem_offset` | `uint32_t` | 4 B | Offset of semaphore region |
| `sem_size` | `uint32_t` | 4 B | Size of semaphore region in bytes |
| `cb_offset` | `uint32_t` | 4 B | Offset of circular buffer configuration region |
| `cb_size` | `uint32_t` | 4 B | Size of CB configuration region |
| `dfb_offset` | `uint32_t` | 4 B | Offset of dataflow buffer region |
| `dfb_size` | `uint32_t` | 4 B | Size of dataflow buffer region |
| `local_cb_size` | `uint32_t` | 4 B | Size of local CB configuration subset |
| `kernel_text_offset` | `uint32_t` | 4 B | Offset of first kernel binary within the config buffer |
| `kernel_text_size` | `uint32_t` | 4 B | Maximum size of all kernel binaries across all kernel groups |

**Total: 40 bytes per programmable core type.**

`ProgramImpl` stores one `ProgramConfig` per programmable core type (TENSIX, ACTIVE_ETH, IDLE_ETH):

```cpp
std::vector<ProgramConfig> program_configs_;  // indexed by ProgrammableCoreType
```

There is also `ProgramOffsetsState` (internal to `ProgramImpl`) that tracks the incremental computation of these offsets during `finalize_offsets()`:

```cpp
struct ProgramOffsetsState {
    uint32_t config_base_offset = 0;
    uint32_t offset = 0;
    uint32_t rta_offset = 0;
    uint32_t sem_offset = 0;
    uint32_t sem_size = 0;
    uint32_t cb_offset = 0;
    uint32_t cb_size = 0;
    uint32_t local_cb_size = 0;
    uint32_t dfb_offset = 0;
    uint32_t dfb_size = 0;
    uint32_t kernel_text_offset = 0;
    uint32_t kernel_text_size = 0;
};
```

This defines the L1 layout:

```
L1 Address Space (per core)
+---------------------------+  kernel_config_base
|  Runtime Args (rta)       |  <- rta_offset
+---------------------------+
|  Semaphores               |  <- sem_offset, sem_size
+---------------------------+
|  CB Config Region         |  <- cb_offset, cb_size
+---------------------------+
|  Dataflow Buffers (DFB)   |  <- dfb_offset, dfb_size
+---------------------------+
|  Kernel Text (binaries)   |  <- kernel_text_offset, kernel_text_size
+---------------------------+
|  CB Data (tiles)          |  (allocated separately by CB allocator)
+---------------------------+
|  ... other L1 regions ... |
+---------------------------+
```

**Capture requirements:** The finalized `ProgramConfig` for each programmable core type must be captured. The replay environment uses it to reconstruct the L1 memory layout.

**Size estimate:** 40 bytes per `ProgramConfig` (10 uint32 fields). With 2--3 programmable core types: ~80--120 bytes.

---

## 4. Semaphore State

Semaphores coordinate execution between RISC-V processors and between cores.

### 4.1 The `Semaphore` Class

Source: `tt_metal/impl/buffers/semaphore.hpp`.

| Field | C++ Type | Size | Role |
|---|---|---|---|
| `core_range_set_` | `CoreRangeSet` | variable | Cores where this semaphore is initialized |
| `id_` | `uint32_t` | 4 B | Semaphore index (0--15) |
| `initial_value_` | `uint32_t` | 4 B | Initial value written to L1 at program start |
| `core_type_` | `CoreType` | 1 B | Core type (WORKER, ETH) |

**Limits**: `NUM_SEMAPHORES = 16` per core (source: `tt_metal/impl/buffers/semaphore.hpp`). Each semaphore occupies 4 bytes in L1 (one uint32). The semaphore region size in L1 is thus at most $16 \times 4 = 64$ bytes per core.

### 4.2 Program-Level Semaphore Tracking

```cpp
// ProgramImpl:
std::vector<Semaphore> semaphores_;
// ...
std::vector<std::reference_wrapper<const Semaphore>> semaphores_on_core(
    const CoreCoord& core, CoreType core_type) const;
```

The `ProgramImpl::init_semaphores()` method writes initial values to L1 at the semaphore offset for each core. The semaphore address in L1 is computed from `ProgramConfig::sem_offset` plus the semaphore's `offset()` (derived from its `id_`).

### 4.3 Capture Subtlety

> The `Semaphore` object captures the *initial* value. At the moment of kernel dispatch, semaphores may already have been modified by previous kernel invocations within the same program. The runtime (volatile) semaphore value must be read from L1 via NOC for a faithful snapshot. See `04_implicit_hidden_and_coordination_state.md` for the volatile semaphore problem and strategies for handling it.

**Size estimate:** 16 semaphores x (4 bytes id + 4 bytes value + core range metadata) = ~200 bytes per core.

---

## 5. `kernel_config_msg_t`: The Device-Side Launch Configuration

Source: `tt_metal/hw/inc/hostdev/dev_msgs.h`. This is the most important device-side struct for capture because it contains the complete runtime configuration as seen by the firmware.

The struct is `__attribute__((packed))` and the field sizes depend on architecture constants.

### 5.1 Full Field Inventory

| Field | C++ Type | Size (WH/BH) | Size (Quasar) | Role |
|---|---|---|---|---|
| `kernel_config_base[ProgrammableCoreType::COUNT]` | `volatile uint32_t[3]` | 12 B | 12 B | Base addresses of kernel config data per programmable core type |
| `sem_offset[ProgrammableCoreType::COUNT]` | `volatile uint16_t[3]` | 6 B | 6 B | Semaphore region offset per core type |
| `local_cb_offset` | `volatile uint16_t` | 2 B | 2 B | Local circular buffer config offset |
| `remote_cb_offset` | `volatile uint16_t` | 2 B | 2 B | Remote circular buffer config offset |
| `rta_offset[MaxProcessorsPerCoreType]` | `rta_offset_t[5]` / `rta_offset_t[24]` | 20 B | 96 B | Runtime args offset per processor |
| `mode` | `volatile uint8_t` | 1 B | 1 B | Dispatch mode (DEV or HOST) |
| `pad2[1]` | `volatile uint8_t[1]` | 1 B | 1 B | Padding |
| `kernel_text_offset[MaxProcessorsPerCoreType]` | `volatile uint32_t[5]` / `volatile uint32_t[24]` | 20 B | 96 B | Kernel binary text offset per processor |
| `local_cb_mask` | `volatile uint64_t` | 8 B | 8 B | Bitmask of which CB indices are local |
| `brisc_noc_id` | `volatile uint8_t` | 1 B | 1 B | NOC ID for BRISC (0 or 1) |
| `brisc_noc_mode` | `volatile uint8_t` | 1 B | 1 B | NOC mode: `DM_DEDICATED_NOC` (0), `DM_DYNAMIC_NOC` (1), `DM_INVALID_NOC` (2) |
| `min_remote_cb_start_index` | `volatile uint8_t` | 1 B | 1 B | Minimum CB index for remote CBs |
| `exit_erisc_kernel` | `volatile uint8_t` | 1 B | 1 B | Flag for Ethernet kernel exit |
| `host_assigned_id` | `volatile uint32_t` | 4 B | 4 B | Encoded ID: bits [9:0] = physical device ID, bits [30:10] = program ID, bit [31] = 0 |
| `enables` | `volatile uint32_t` | 4 B | 4 B | Bitmask: bit $i$ set means processor $i$ is enabled |
| `watcher_kernel_ids[MaxProcessorsPerCoreType]` | `volatile uint16_t[5]` / `volatile uint16_t[24]` | 10 B | 48 B | Watcher kernel ID per processor |
| `ncrisc_kernel_size16` | `volatile uint16_t` | 2 B | 2 B | NCRISC kernel size in 16-byte units |
| `sub_device_origin_x` | `volatile uint8_t` | 1 B | 1 B | Logical X coordinate of sub-device origin |
| `sub_device_origin_y` | `volatile uint8_t` | 1 B | 1 B | Logical Y coordinate of sub-device origin |
| `pad3[...]` | `volatile uint8_t[...]` | ~15 B | ~15 B | Padding for alignment |
| `preload` | `volatile uint8_t` | 1 B | 1 B | Preload flag; must be at end so it is only written when all other data is written |

### 5.2 The `rta_offset_t` Sub-Struct

```cpp
struct rta_offset_t {
    volatile uint16_t rta_offset;   // 2 B: offset to unique runtime args
    volatile uint16_t crta_offset;  // 2 B: offset to common runtime args
};
```

Total: 4 bytes per processor.

### 5.3 Architecture-Dependent Sizes

| Architecture | `MaxProcessorsPerCoreType` | `ProgrammableCoreType::COUNT` | Approximate `kernel_config_msg_t` Size |
|---|---|---|---|
| Wormhole B0 | 5 | 3 | ~112 B |
| Blackhole | 5 | 3 | ~112 B |
| Quasar | 24 | 3 | ~304 B |

The size variation comes from the arrays indexed by `MaxProcessorsPerCoreType`: `rta_offset` (4 B each), `kernel_text_offset` (4 B each), and `watcher_kernel_ids` (2 B each) -- a total of 10 bytes per processor slot.

### 5.4 Alignment Constraints

The source enforces alignment via `static_assert`:

```cpp
static_assert(offsetof(kernel_config_msg_t, kernel_config_base) % sizeof(uint32_t) == 0);
static_assert(offsetof(kernel_config_msg_t, sem_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, local_cb_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, remote_cb_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, rta_offset) % sizeof(uint16_t) == 0);
static_assert(offsetof(kernel_config_msg_t, kernel_text_offset) % sizeof(uint32_t) == 0);
static_assert(offsetof(kernel_config_msg_t, local_cb_mask) % sizeof(uint64_t) == 0);
static_assert(offsetof(kernel_config_msg_t, host_assigned_id) % sizeof(uint32_t) == 0);
```

These alignment constraints matter for binary-level capture: the packed struct must be captured with exact byte layout, not field-by-field extraction that might reorder or pad differently.

### 5.5 The `enables` Bitmask

The `enables` field is critical for replay: it specifies which processors are active for this kernel group. A compute-only kernel on Blackhole has `enables = 0b11100` (TRISC0, TRISC1, TRISC2), while a DM-only kernel has `enables = 0b00011` (BRISC, NCRISC).

On Quasar with `MaxProcessorsPerCoreType = 24`, bits 0--23 are meaningful. The replay system uses this mask to determine which processor ELFs to load and which to skip.

---

## 6. `launch_msg_t` and `go_msg_t`

### 6.1 `launch_msg_t`

Source: `tt_metal/hw/inc/hostdev/dev_msgs.h`.

```cpp
struct launch_msg_t {  // must be cacheline aligned
    kernel_config_msg_t kernel_config;
} __attribute__((packed));
```

`launch_msg_t` is a thin wrapper around `kernel_config_msg_t`. Its size equals `kernel_config_msg_t`. It is written to the mailbox ring buffer on the device.

The mailbox stores `launch_msg_buffer_num_entries = 8` launch messages in a ring:

```cpp
struct mailboxes_t {
    // ...
    volatile uint32_t launch_msg_rd_ptr;
    struct launch_msg_t launch[launch_msg_buffer_num_entries];  // 8 entries
    volatile struct go_msg_t go_messages[go_message_num_entries]; // 9 entries
    // ...
};
```

### 6.2 `go_msg_t`

Source: `tt_metal/hw/inc/hostdev/dev_msgs.h`.

```cpp
struct go_msg_t {
    union {
        uint32_t all;          // 4 B: entire message as a single uint32
        struct {
            uint8_t dispatch_message_offset;  // 1 B: offset into launch msg buffer
            uint8_t master_x;                 // 1 B: X coordinate of master core
            uint8_t master_y;                 // 1 B: Y coordinate of master core
            uint8_t signal;                   // 1 B: signal value
        };
    };
} __attribute__((packed));
```

**Total: 4 bytes.** The `signal` field is the most important: `RUN_MSG_GO = 0x80` triggers kernel execution. There are `go_message_num_entries = 9` slots (equal to max sub-devices + 1).

### 6.3 Signal Constants

| Constant | Value | Meaning |
|---|---|---|
| `RUN_MSG_INIT` | `0x40` | Initialize (firmware setup) |
| `RUN_MSG_GO` | `0x80` | Execute kernel |
| `RUN_MSG_RESET_READ_PTR` | `0xc0` | Reset launch msg read pointer |
| `RUN_MSG_RESET_READ_PTR_FROM_HOST` | `0xe0` | Host-initiated read pointer reset |
| `RUN_MSG_REPLAY_TRACE` | `0xf0` | Replay a recorded trace |
| `RUN_MSG_DONE` | `0` | Kernel execution complete |

---

## 7. The `KernelGroup` as a Capture Unit

Source: `tt_metal/impl/program/program_impl.hpp`.

`KernelGroup` aggregates kernels that share a core range and are launched together. It holds the fully resolved `launch_msg_t` and `go_msg_t`:

| Field | C++ Type | Size | Role |
|---|---|---|---|
| `programmable_core_type_index` | `uint32_t` | 4 B | Index into `ProgrammableCoreType` enum |
| `core_ranges` | `CoreRangeSet` | variable | Cores this kernel group runs on |
| `kernel_ids` | `std::vector<KernelHandle>` | variable | Handles to kernels, ordered by processor index |
| `rta_sizes` | `std::vector<uint32_t>` | variable | Runtime arg sizes per kernel |
| `crta_offsets` | `std::vector<uint32_t>` | variable | Common RTA offsets per kernel |
| `crta_sizes` | `std::vector<uint32_t>` | variable | Common RTA sizes per kernel |
| `total_rta_size` | `uint32_t` | 4 B | Total runtime args size across all kernels in group |
| `kernel_text_offsets` | `std::vector<uint32_t>` | variable | Per-processor binary text offsets |
| `launch_msg` | `dev_msgs::launch_msg_t` | see Section 5 | Fully populated launch message |
| `go_msg` | `dev_msgs::go_msg_t` | 4 B | Go signal |

Each `KernelGroup` represents an indivisible unit of dispatch. The interceptor should capture the full group structure, including which kernel IDs map to which processors and the core ranges the group spans. The `launch_msg` and `go_msg` are populated during `KernelGroup` construction (called from `ProgramImpl::update_kernel_groups()`), which happens during `ProgramImpl::finalize_offsets()`. They are host-side data structures, fully accessible after finalization but **before** they are serialized into the dispatch command stream.

This is why the plan identifies the hook *inside* `program_dispatch::assemble_device_commands()` (after `finalize_offsets()`) as the ideal capture point: the launch messages are fully resolved but still in structured form.

---

## 8. NOC Configuration

NOC configuration determines how each data movement processor accesses the on-chip network:

```cpp
enum noc_index { NOC_0 = 0, NOC_1 = 1 };

enum noc_mode : uint8_t {
    DM_DEDICATED_NOC = 0,   // Each DM processor gets its own NOC
    DM_DYNAMIC_NOC = 1,     // DM processors share NOCs dynamically
    DM_INVALID_NOC = 2,
};
```

In dedicated NOC mode (the default), BRISC exclusively uses one NOC and NCRISC exclusively uses the other. In dynamic NOC mode, both DM processors can use either NOC. The default assignment from `DataMovementConfig`:

```cpp
struct DataMovementConfig {
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    NOC noc = NOC::RISCV_0_default;
    NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;
};
```

These are stored in `kernel_config_msg_t::brisc_noc_id` and `kernel_config_msg_t::brisc_noc_mode`. For compute-only replay (TRISC debugging), NOC configuration is not directly needed since TRISCs do not issue NOC transactions. For data movement replay, incorrect NOC configuration causes the kernel to issue transactions on the wrong NOC.

---

## 9. Dispatch Enable Flags

```cpp
enum dispatch_enable_flags : uint8_t {
    DISPATCH_ENABLE_FLAG_PRELOAD = 1 << 7,
};
```

The `kernel_config_msg_t::enables` field (uint32) is a bitmask where bit $i$ indicates processor $i$ is active for this kernel group. On WH/BH with `MaxProcessorsPerCoreType = 5`, only bits 0--4 are meaningful. On Quasar with `MaxProcessorsPerCoreType = 24`, bits 0--23 are meaningful.

Additional `kernel_config_msg_t` fields that affect kernel execution:

| Field | Type | Values | Description |
|---|---|---|---|
| `mode` | `uint8_t` | `DISPATCH_MODE_DEV` (0), `DISPATCH_MODE_HOST` (1) | Whether dispatch is device-driven or host-driven |
| `min_remote_cb_start_index` | `uint8_t` | 0--64 | First CB index used for remote (cross-core) CBs |
| `exit_erisc_kernel` | `uint8_t` | 0 or 1 | Whether to exit ethernet kernel mode |
| `sub_device_origin_x/y` | `uint8_t` | coordinates | Logical coordinates of the sub-device origin |

---

## 10. The `subordinate_sync_msg_t` (Inter-RISC Synchronization Map)

While the full inter-RISC synchronization protocol is covered in `04_implicit_hidden_and_coordination_state.md`, the `subordinate_sync_msg_t` structure is part of the runtime dispatch state and is included here for completeness.

### 10.1 Wormhole/Blackhole Layout

Source: `tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h`.

```cpp
union subordinate_map_t {
    volatile uint32_t all;          // 4 B: entire map as one word
    struct {
        volatile uint8_t dm1;       // 1 B: NCRISC status
        volatile uint8_t trisc0;    // 1 B: TRISC0 status
        volatile uint8_t trisc1;    // 1 B: TRISC1 status
        volatile uint8_t trisc2;    // 1 B: TRISC2 status
    };
};
```

**Total: 4 bytes.** The master (BRISC) uses this to synchronize GO/DONE signals with the four subordinate processors. The combined pattern `RUN_SYNC_MSG_ALL_GO = 0x80808080` simultaneously arms all four subordinates.

### 10.2 Quasar Layout

Source: `tt_metal/hw/inc/internal/tt-2xx/quasar/core_config.h`.

```cpp
union subordinate_map_t {
    union {
        struct {
            volatile uint64_t allDMs;       // 8 B: all DM subordinate status
            volatile uint32_t allNeo0;      // 4 B: all NEO engine 0 compute status
            volatile uint32_t allNeo1;      // 4 B: all NEO engine 1 compute status
            volatile uint32_t allNeo2;      // 4 B: all NEO engine 2 compute status
            volatile uint32_t allNeo3;      // 4 B: all NEO engine 3 compute status
        };
        struct {
            volatile uint8_t dm1;           // DM core 1 status
            volatile uint8_t dm2;           // DM core 2 status
            volatile uint8_t dm3;           // DM core 3 status
            volatile uint8_t dm4;           // DM core 4 status
            volatile uint8_t dm5;           // DM core 5 status
            volatile uint8_t dm6;           // DM core 6 status
            volatile uint8_t dm7;           // DM core 7 status
            volatile uint8_t padding;
            volatile uint8_t neo0_trisc0;   // NEO engine 0, TRISC0
            volatile uint8_t neo0_trisc1;
            volatile uint8_t neo0_trisc2;
            volatile uint8_t neo0_trisc3;   // Quasar adds TRISC3
            volatile uint8_t neo1_trisc0;
            volatile uint8_t neo1_trisc1;
            volatile uint8_t neo1_trisc2;
            volatile uint8_t neo1_trisc3;
            volatile uint8_t neo2_trisc0;
            volatile uint8_t neo2_trisc1;
            volatile uint8_t neo2_trisc2;
            volatile uint8_t neo2_trisc3;
            volatile uint8_t neo3_trisc0;
            volatile uint8_t neo3_trisc1;
            volatile uint8_t neo3_trisc2;
            volatile uint8_t neo3_trisc3;
            uint8_t pad[12];                // Padding to 36 bytes
        };
    } __attribute__((packed));
} __attribute__((packed));
```

**Total: 36 bytes** (`subordinate_map_size = sizeof(subordinate_map_t)`). On Quasar, 23 subordinates (7 DMs + 16 compute processors) are tracked instead of 4.

---

## 11. Complete Configuration Size Estimate

| Component | Size per Core (WH/BH) | Size per Core (Quasar) |
|---|---|---|
| `kernel_config_msg_t` | ~112 B | ~304 B |
| `go_msg_t` | 4 B | 4 B |
| Runtime args (typical 20 args) | 80 B | 80 B |
| Common runtime args (typical 10 args) | 40 B | 40 B |
| CB configs (8 active CBs, 16 B each) | 128 B | 128 B |
| Semaphore initial values (4 active) | 16 B | 16 B |
| `ProgramConfig` | 40 B | 40 B |
| `subordinate_sync_msg_t` | 4 B | 36 B |
| **Total** | **~424 B** | **~648 B** |

For 64 cores: approximately 27 KB on WH/BH, 41 KB on Quasar. This is a small fraction of the total capture envelope; the compiled binaries and memory contents dominate.

---

## Capture Feasibility Summary

| State Element | Host Accessible? | Requires Halting? | Volatile? | Omission Impact |
|---|---|---|---|---|
| Per-core runtime args | Yes -- `Kernel` object | No | No -- set before dispatch | **Fatal** -- kernel reads garbage args |
| Common runtime args | Yes -- `Kernel` object | No | No | Same as above |
| CB config (formats, sizes, addresses) | Yes -- `CircularBufferImpl` objects | No | No -- set at program construction | **Fatal** -- L1 map incorrect |
| CB allocated L1 addresses | Yes -- after `allocate_circular_buffers()` | No | No | **Fatal** -- tile data at wrong location |
| Semaphore declarations (id, initial value) | Yes -- `ProgramImpl::semaphores_` | No | No | Replay deadlocks or races |
| Semaphore L1 offsets | Yes -- after `finalize_offsets()` | No | No | Cannot initialize semaphores at correct address |
| `kernel_config_msg_t` (launch message) | Yes -- in `KernelGroup::launch_msg` | No | No -- populated during finalization | **Fatal** -- defines entire L1 layout |
| `go_msg_t` | Yes -- in `KernelGroup::go_msg` | No | No | Needed for dispatch core identification |
| `ProgramConfig` (L1 map summary) | Yes -- `ProgramImpl::program_configs_` | No | No | Redundant if launch message captured, but useful for validation |
| NOC id/mode | Yes -- in kernel config and launch msg | No | No | Incorrect NOC routing in DM replay |
| `subordinate_sync_msg_t` | Yes -- within mailboxes (L1) | Timing-sensitive | Yes -- changes during launch | Full 5-RISC replay needs it; single-TRISC does not |

---

## Key Takeaways

- **Runtime configuration state is entirely host-accessible and non-volatile** (with the exception of `subordinate_sync_msg_t`), making it the easiest category to capture. The capture must occur **after** `ProgramImpl::finalize_offsets()` to ensure all offsets, launch messages, and kernel group assignments are fully resolved.

- **The `kernel_config_msg_t` is the single most information-dense capture target** in this category. It contains the L1 memory map (offsets for binaries, CBs, semaphores, runtime args), the processor enable mask, NOC configuration, and the dispatch mode. Its size varies significantly by architecture (approximately 112 bytes on WH/BH vs. 304 bytes on Quasar) due to the `MaxProcessorsPerCoreType`-indexed arrays.

- **Quasar's `subordinate_sync_msg_t` scales from 4 bytes (WH/BH) to 36 bytes** to track 23 subordinate processors (7 DMs + 16 compute), reflecting the dramatically expanded parallelism within a single Quasar cluster compared to a WH/BH Tensix core.

- **Semaphore initial values are necessary but insufficient for replay.** The program sets initial values, but by the time a specific kernel dispatch occurs, semaphores may hold different values. The distinction between declared initial values (captured here) and live runtime values (captured in `03_memory_contents_and_tile_data.md` via L1 memory reads) is critical.

---

**Previous:** [Compiled Binary and Build State](./01_compiled_binary_and_build_state.md) | **Next:** [Memory Contents and Tile Data](./03_memory_contents_and_tile_data.md)
