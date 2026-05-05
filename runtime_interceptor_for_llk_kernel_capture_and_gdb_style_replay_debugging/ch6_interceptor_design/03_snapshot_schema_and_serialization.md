# Snapshot Schema and Serialization

This file defines the FlatBuffer schema for kernel invocation snapshots (`.ttksnap` files), the serialization and deserialization code, binary deduplication strategy, compression pipeline, storage layout, and concrete size estimates with worked examples. The schema extends the existing LightMetal FlatBuffer infrastructure (Ch 3, File 1) and references the complete state inventory (Ch 2).

> **Cross-references:** Ch 2 (8 state categories, full field lists), Ch 3 Sec 1 (LightMetal FlatBuffer schema: `command.fbs`, `light_metal_binary.fbs`), Ch 3 Sec 2 (gaps: no binaries, no L1 data, no per-core RTAs), Ch 4 Sec 2 (H4 data: ~13 KB command sequences + ~60 KB binaries), File 01 (primary hook, `KernelSnapshotCaptureContext`), File 02 (capture modes, ring buffer).

All source citations reference commit `621b949`.

---

## 1. Schema Design Principles

The snapshot schema satisfies four requirements:

1. **Self-contained**: A `.ttksnap` file contains everything required to replay a kernel invocation on an emulator or single-core hardware target, without external dependencies on the TT-Metal runtime or build artifacts.
2. **Extensible**: New state categories (e.g., Quasar-specific processor types) can be added without breaking backward compatibility, using FlatBuffer's additive field model.
3. **Efficient**: The schema supports zero-copy deserialization for fast loading in the replay tool.
4. **Compatible**: It reuses existing FlatBuffer type definitions from `base_types.fbs` and `program_types.fbs` in `tt_metal/impl/flatbuffer/`.

---

## 2. Existing Schema Inventory

The snapshot schema reuses types from the existing TT-Metal FlatBuffer infrastructure:

| Existing Schema | File | Reused Types |
|----------------|------|-------------|
| `base_types.fbs` | `tt_metal/impl/flatbuffer/base_types.fbs` | `Arch`, `DataFormat` (lines 45-67), `MathFidelity` (lines 37-43), `DefineEntry` (lines 74-77), `Tile` (lines 80-83), `BoolOptional`, `Uint8Optional`, `Uint32Optional` (lines 86-88) |
| `program_types.fbs` | `tt_metal/impl/flatbuffer/program_types.fbs` | `CoreCoord`, `CoreRange`, `CoreRangeSet`, `CoreSpec` (lines 5-23), `DataMovementConfig`, `ComputeConfig`, `EthernetConfig` (lines 25-50), `KernelConfig` union (lines 53-57), `RuntimeArg`, `RuntimeArgValue` (lines 59-78), `UInt32Vector` |
| `buffer_types.fbs` | `tt_metal/impl/flatbuffer/buffer_types.fbs` | `CircularBufferConfig` (lines 68-81) |
| `command.fbs` | `tt_metal/impl/flatbuffer/command.fbs` | `CommandType` union (lines 118-136) with 17 members |
| `light_metal_binary.fbs` | `tt_metal/impl/flatbuffer/light_metal_binary.fbs` | Architecture pattern for top-level binary format (lines 32-36) |

The key difference from `light_metal_binary.fbs` is scope: LightMetal captures a sequence of host API commands for replaying an entire workload, while the snapshot captures the resolved state at a single point in time for a single program dispatch. LightMetal's `Command` union records API calls like `CreateKernel`, `SetRuntimeArgs`, `EnqueueWriteBuffer`; the snapshot instead records the final computed values that result from those calls.

---

## 3. **PROPOSED:** FlatBuffer Schema (`kernel_snapshot.fbs`)

```flatbuffers
// PROPOSED: tt_metal/impl/flatbuffer/kernel_snapshot.fbs

include "base_types.fbs";
include "program_types.fbs";

namespace tt.tt_metal.flatbuffer.snapshot;

// ============================================================
// Schema Version and Capture Metadata
// ============================================================

table CaptureMetadata {
    // Schema version for forward/backward compatibility
    schema_version_major: uint16 = 1;
    schema_version_minor: uint16 = 0;

    // Build provenance
    tt_metal_git_hash: string;          // e.g. "621b949"
    compiler_version: string;           // RISC-V GCC version string
    compiler_flags: string;             // Optimization level + defines

    // Hardware context
    architecture: tt.tt_metal.flatbuffer.Arch;
    device_id: uint32;
    grid_size_x: uint32;               // Logical grid dimensions (harvesting-aware)
    grid_size_y: uint32;
    l1_size_bytes: uint32;             // 1048576 for WH, 1572864 for BH

    // Capture context
    capture_mode: CaptureMode;
    capture_timestamp_ns: uint64;       // steady_clock nanoseconds
    host_hostname: string;
    trigger_reason: string;             // "selective", "failure:assert", "breakpoint:0", etc.
    invocation_index: uint64;           // Monotonic counter across session
}

enum CaptureMode : ubyte {
    MetadataOnly = 0,
    Selective = 1,
    FailureTriggered = 2,
    Breakpoint = 3,
}

// ============================================================
// Kernel Binary Blob
// ============================================================

// Represents a single compiled RISC-V binary for one processor.
// Corresponds to one ll_api::memory object (tt_metal/llrt/tt_memory.h).
table KernelBinaryBlob {
    // Which processor this binary targets
    processor_type: uint32;             // Kernel::get_kernel_processor_type(index)
    processor_class: ubyte;             // HalProcessorClassType: 0=DM, 1=COMPUTE
    processor_name: string;             // "BRISC", "NCRISC", "TRISC0", etc.
    programmable_core_type: ubyte;      // HalProgrammableCoreType: TENSIX, ACTIVE_ETH

    // Binary data -- the raw words from ll_api::memory::data()
    elf_data: [uint32];                 // Complete binary image

    // Memory layout from ll_api::memory accessors
    text_addr: uint32;                  // ll_api::memory::get_text_addr()
    text_size: uint32;                  // ll_api::memory::get_text_size()
    packed_size: uint32;                // ll_api::memory::get_packed_size()
    num_spans: uint32;                  // ll_api::memory::num_spans()

    // Span details for relocatable binaries
    spans: [BinarySpan];

    // Deduplication: FNV1a hash of compile inputs
    build_hash: uint64;                 // From Kernel::compute_hash()

    // Optional: disk path for GDB (if debug info available)
    elf_file_path: string;
}

table BinarySpan {
    device_addr: uint64;                // Byte address in device L1
    length: uint32;                     // Length in uint32 words
    data_offset: uint32;                // Offset into elf_data where this span begins
}

// ============================================================
// Circular Buffer Snapshot
// ============================================================

// Captures the configuration and optionally the data contents of a circular buffer.
// Source: CircularBufferConfig (tt_metal/api/tt-metalium/circular_buffer_config.hpp)
// and CircularBufferImpl internals.
table CircularBufferSnapshot {
    cb_index: uint8;                    // CB index (0-31 for Tensix)
    data_format: tt.tt_metal.flatbuffer.DataFormat;
    tile_shape: [uint32];               // Tile dimensions [height, width]
    transpose_tile: bool;

    total_size: uint32;                 // Total buffer size in bytes
    page_size: uint32;                  // Page size in bytes
    l1_address: uint32;                 // Resolved L1 address (from ProgramConfig::cb_offset)

    is_remote: bool;                    // Local or remote CB
    is_dynamic: bool;                   // Dynamic CB flag

    // Optional: actual data contents at capture time (deep capture only)
    data_contents: [uint8];             // Raw bytes; empty for lightweight captures
}

// ============================================================
// Semaphore Snapshot
// ============================================================

// Captures semaphore configuration and value at the time of capture.
// Source: Semaphore class (tt_metal/impl/buffers/semaphore.hpp lines 17-48).
table SemaphoreSnapshot {
    id: uint32;                         // Semaphore ID (0-15)
    l1_address: uint32;                 // Resolved L1 address
    initial_value: uint32;              // Value configured by the program
    value_at_capture: uint32;           // Value read from L1 at capture time
    value_at_capture_valid: bool;       // True if read from device (not just initial)
    core_type: ubyte;                   // CoreType enum
    core_range_set: tt.tt_metal.flatbuffer.CoreRangeSet;
}

// ============================================================
// L1 Region Snapshot
// ============================================================

// A raw dump of an L1 memory region. Used for firmware mailboxes,
// stack regions, CB data, and other state not captured in structured form.
table L1RegionSnapshot {
    region_name: string;                // "firmware_mailbox", "stack", "cb_data", "full_l1"
    start_address: uint32;              // Byte address in L1
    size: uint32;                       // Size in bytes
    data: [uint8];                      // Raw L1 contents
}

// ============================================================
// Runtime Args Snapshot
// ============================================================

table RuntimeArgsPerCore {
    core_x: uint32;
    core_y: uint32;
    args: [uint32];
}

table RuntimeArgsSnapshot {
    kernel_handle: uint32;              // KernelHandle in the program
    per_core_args: [RuntimeArgsPerCore];
    common_args: [uint32];              // From Kernel::common_runtime_args()
    common_args_count: uint32;
}

// ============================================================
// Launch Message Snapshot
// ============================================================

// Serialized launch_msg_t from a KernelGroup.
// Source: dev_msgs::launch_msg_t (dev_msgs.h:191-193) wrapping
// kernel_config_msg_t (dev_msgs.h:136-167).
table LaunchMessageSnapshot {
    // kernel_config_msg_t fields
    kernel_config_base: [uint32];       // Per programmable core type
    sem_offsets: [uint16];              // Per programmable core type
    local_cb_offset: uint16;
    remote_cb_offset: uint16;
    rta_offsets: [RtaOffsetEntry];      // Per processor
    mode: uint8;                        // dispatch_mode
    kernel_text_offsets: [uint32];      // Per processor
    local_cb_mask: uint64;
    brisc_noc_id: uint8;
    brisc_noc_mode: uint8;
    min_remote_cb_start_index: uint8;
    host_assigned_id: uint32;
    enables: uint32;                    // Processor enable bitmask
    watcher_kernel_ids: [uint16];       // Per processor

    // go_msg_t fields
    dispatch_message_offset: uint8;
    master_x: uint8;
    master_y: uint8;
    go_signal: uint8;

    // Core ranges this launch message applies to
    core_ranges: tt.tt_metal.flatbuffer.CoreRangeSet;
    is_multicast: bool;
}

struct RtaOffsetEntry {
    rta_offset: uint16;                 // rta_offset_t::rta_offset (dev_msgs.h:128)
    crta_offset: uint16;                // rta_offset_t::crta_offset (dev_msgs.h:129)
}

// ============================================================
// Kernel Descriptor
// ============================================================

// Complete description of a kernel within the program.
// Source: Kernel class (tt_metal/impl/kernels/kernel.hpp).
table KernelDescriptor {
    kernel_handle: uint32;
    kernel_name: string;                // From Kernel::name()
    kernel_full_name: string;           // From Kernel::get_full_kernel_name()
    source_path: string;                // Original source file path

    // Configuration variant (matches Kernel::Config)
    config: tt.tt_metal.flatbuffer.KernelConfig;

    // Compile-time arguments
    compile_args: [uint32];             // Kernel::compile_time_args_
    defines: [tt.tt_metal.flatbuffer.DefineEntry]; // Kernel::defines_

    // Compiled binaries -- indices into binary_pool
    binary_indices: [int32];            // One per processor; -1 = not assigned

    // Runtime arguments
    runtime_args: RuntimeArgsSnapshot;

    // Core placement
    core_range_set: tt.tt_metal.flatbuffer.CoreRangeSet;

    // Processor identity
    programmable_core_type: ubyte;
    processor_class: ubyte;

    // Compile hash for deduplication
    compile_hash: uint64;               // From Kernel::compute_hash()
    watcher_kernel_id: int32;
}

// ============================================================
// Per-Core Snapshot (Tensix Core)
// ============================================================

// Complete state of a single Tensix core at the time of capture.
// This is the per-core organization specified by the plan's
// TensixCoreSnapshot design.
table TensixCoreSnapshot {
    // Core identification
    logical_core: tt.tt_metal.flatbuffer.CoreCoord;
    virtual_core: tt.tt_metal.flatbuffer.CoreCoord;
    physical_core: tt.tt_metal.flatbuffer.CoreCoord;
    core_type: ubyte;                   // CoreType enum

    // Binaries assigned to this core (indices into binary_pool)
    brisc_binary_index: int32 = -1;     // -1 = not assigned
    ncrisc_binary_index: int32 = -1;
    trisc0_binary_index: int32 = -1;
    trisc1_binary_index: int32 = -1;
    trisc2_binary_index: int32 = -1;

    // Per-core state
    cb_snapshots: [CircularBufferSnapshot];
    semaphore_snapshots: [SemaphoreSnapshot];
    l1_regions: [L1RegionSnapshot];

    // Runtime args for this specific core (one vector per RISC)
    per_risc_runtime_args: [tt.tt_metal.flatbuffer.UInt32Vector];

    // Launch message that was dispatched to this core
    launch_msg: LaunchMessageSnapshot;

    // Program config (L1 memory layout)
    program_config: ProgramConfigSnapshot;
}

// Maps ProgramConfig (program_impl.hpp lines 97-108)
// Includes all 10 fields of the actual ProgramConfig struct
table ProgramConfigSnapshot {
    rta_offset: uint32;
    sem_offset: uint32;
    sem_size: uint32;
    cb_offset: uint32;
    cb_size: uint32;
    dfb_offset: uint32;                 // Device-side fast buffer offset
    dfb_size: uint32;                   // Device-side fast buffer size
    local_cb_size: uint32;
    kernel_text_offset: uint32;
    kernel_text_size: uint32;
}

// ============================================================
// Kernel Group Snapshot
// ============================================================

table KernelGroupSnapshot {
    programmable_core_type_index: uint32;
    core_ranges: tt.tt_metal.flatbuffer.CoreRangeSet;
    kernel_ids: [uint32];               // Indices into KernelInvocationSnapshot.kernels
    rta_sizes: [uint32];
    crta_offsets: [uint32];
    crta_sizes: [uint32];
    total_rta_size: uint32;
    launch_msg: LaunchMessageSnapshot;
}

// ============================================================
// Buffer Data (for input/output tile captures)
// ============================================================

table BufferDataSnapshot {
    buffer_address: uint64;             // DRAM or L1 address
    page_size: uint32;
    total_size: uint32;
    data: [uint8];                      // Raw buffer contents
    associated_cb_index: int8 = -1;     // CB this buffer feeds, or -1
}

// ============================================================
// Root Type: KernelInvocationSnapshot
// ============================================================

// The top-level container for a .ttksnap file.
// Represents a complete snapshot of one kernel invocation (program dispatch).
table KernelInvocationSnapshot {
    // Capture metadata and provenance
    metadata: CaptureMetadata;

    // Program identification
    program_id: uint64;                 // ProgramImpl::get_id()
    program_runtime_id: uint64;         // ProgramImpl::get_runtime_id()

    // All kernels in this program
    kernels: [KernelDescriptor];

    // Per-core snapshots (may be filtered to a single core)
    core_snapshots: [TensixCoreSnapshot];

    // Kernel group structure (preserves dispatch grouping)
    kernel_groups: [KernelGroupSnapshot];

    // Program-level semaphores
    semaphores: [SemaphoreSnapshot];
    sub_device_id: uint8;

    // Binary deduplication: shared binary pool referenced by index
    binary_pool: [KernelBinaryBlob];    // Unique binaries; cores reference by index

    // Optional: input/output tile data
    input_buffers: [BufferDataSnapshot];
    output_buffers: [BufferDataSnapshot];
}

root_type KernelInvocationSnapshot;
file_identifier "TKSN";
file_extension "ttksnap";
```

---

## 4. **PROPOSED:** Schema Extension to `command.fbs`

To enable snapshots to be embedded in LightMetal binaries, **PROPOSED:** add `KernelSnapshotCommand` to the `CommandType` union in `command.fbs`:

```flatbuffers
// PROPOSED modification to command.fbs:118-137
union CommandType {
  ReplayTraceCommand,
  EnqueueTraceCommand,
  LoadTraceCommand,
  ReleaseTraceCommand,
  BufferCreateCommand,
  BufferDeallocateCommand,
  BufferDeleteCommand,
  EnqueueWriteBufferCommand,
  EnqueueReadBufferCommand,
  FinishCommand,
  ProgramConstructorCommand,
  CreateKernelCommand,
  SetRuntimeArgsUint32Command,
  SetRuntimeArgsUint32VecPerCoreCommand,
  SetRuntimeArgsCommand,
  CreateCircularBufferCommand,
  LightMetalCompareCommand,
  KernelSnapshotCommand,              // PROPOSED: New entry
}
```

This addition is backward-compatible because FlatBuffer unions are extended by appending new members. Existing binaries that do not contain `KernelSnapshotCommand` entries continue to deserialize correctly. When embedded in a LightMetal binary, snapshots appear as entries in the `commands` array, interleaved with existing host-API capture commands.

The standalone `.ttksnap` format (using `KernelInvocationSnapshot` as root type) remains the primary format for the interceptor output. The `KernelSnapshotCommand` union member is for optional LightMetal integration.

---

## 5. **PROPOSED:** Serialization Implementation

### 5.1 Top-Level Serialization Function

Following the pattern of `CaptureCreateKernel` in `host_api_capture_helpers.hpp`:

```cpp
// PROPOSED: tt_metal/impl/debug/snapshot/kernel_snapshot_serialize.cpp

#include "kernel_snapshot_generated.h"  // FlatBuffer-generated header
#include "program_impl.hpp"
#include "program_command_sequence.hpp"

namespace tt::tt_metal::kernel_capture {

flatbuffers::Offset<snapshot::KernelInvocationSnapshot> serialize_snapshot(
    flatbuffers::FlatBufferBuilder& builder,
    const detail::ProgramImpl& program,
    IDevice* device,
    SubDeviceId sub_device_id,
    const KernelCaptureConfig& config) {

    uint64_t build_key = BuildEnvManager::get_instance()
        .get_device_build_env(device->build_id()).build_key();

    // 1. Build binary pool with deduplication
    std::unordered_map<uint64_t, uint32_t> binary_hash_to_index;
    std::vector<flatbuffers::Offset<snapshot::KernelBinaryBlob>> binary_pool;
    // (see Section 7 for deduplication logic)

    // 2. Serialize kernel descriptors
    std::vector<flatbuffers::Offset<snapshot::KernelDescriptor>> kernel_offsets;
    for (const auto& kernels_by_type : program.kernels_) {
        for (const auto& [handle, kernel] : kernels_by_type) {
            kernel_offsets.push_back(
                serialize_kernel(builder, handle, *kernel, device,
                                 build_key, binary_hash_to_index, binary_pool));
        }
    }

    // 3. Serialize circular buffers
    std::vector<flatbuffers::Offset<snapshot::CircularBufferSnapshot>> cb_offsets;
    for (const auto& cb : program.circular_buffers_) {
        cb_offsets.push_back(serialize_circular_buffer(builder, *cb));
    }

    // 4. Serialize semaphores
    std::vector<flatbuffers::Offset<snapshot::SemaphoreSnapshot>> sem_offsets;
    for (const auto& sem : program.semaphores_) {
        sem_offsets.push_back(serialize_semaphore(builder, sem));
    }

    // 5. Serialize kernel groups
    std::vector<flatbuffers::Offset<snapshot::KernelGroupSnapshot>> kg_offsets;
    for (const auto& kg_vec : program.kernel_groups_) {
        for (const auto& kg : kg_vec) {
            kg_offsets.push_back(serialize_kernel_group(builder, *kg));
        }
    }

    // 6. Build metadata
    auto metadata = snapshot::CreateCaptureMetadata(builder,
        /* schema_version_major */ 1,
        /* schema_version_minor */ 0,
        builder.CreateString(/* git_hash */),
        builder.CreateString(/* compiler_version */),
        builder.CreateString(/* compiler_flags */),
        static_cast<tt::tt_metal::flatbuffer::Arch>(device->arch()),
        device->id(),
        /* grid_size_x */ 0, /* grid_size_y */ 0,
        device->l1_size_per_core(),
        snapshot::CaptureMode_Selective,
        /* timestamp */ std::chrono::steady_clock::now().time_since_epoch().count(),
        /* hostname */ 0,
        builder.CreateString("selective"),
        /* invocation_index */ 0);

    // 7. Assemble root
    return snapshot::CreateKernelInvocationSnapshot(builder,
        metadata,
        program.get_id(),
        program.get_runtime_id(),
        builder.CreateVector(kernel_offsets),
        /* core_snapshots */ 0,   // Populated separately if single-core capture
        builder.CreateVector(kg_offsets),
        builder.CreateVector(sem_offsets),
        *sub_device_id,
        builder.CreateVector(binary_pool));
}

}  // namespace tt::tt_metal::kernel_capture
```

### 5.2 Kernel Serialization

```cpp
flatbuffers::Offset<snapshot::KernelDescriptor> serialize_kernel(
    flatbuffers::FlatBufferBuilder& builder,
    KernelHandle handle,
    const Kernel& kernel,
    IDevice* device,
    uint64_t build_key,
    std::unordered_map<uint64_t, uint32_t>& hash_to_index,
    std::vector<flatbuffers::Offset<snapshot::KernelBinaryBlob>>& binary_pool) {

    auto name_offset = builder.CreateString(kernel.kernel_full_name_);
    auto source_offset = builder.CreateString(kernel.kernel_src_.source_);
    auto crs_offset = serialize_core_range_set(builder, kernel.core_range_set_);
    auto cta_offset = builder.CreateVector(kernel.compile_time_args_);

    // Defines
    std::vector<flatbuffers::Offset<flatbuffer::DefineEntry>> define_offsets;
    for (const auto& [key, value] : kernel.defines_) {
        define_offsets.push_back(
            flatbuffer::CreateDefineEntry(builder,
                builder.CreateString(key),
                builder.CreateString(value)));
    }

    // Binary indices via deduplication pool
    std::vector<int32_t> bin_indices;
    if (kernel.binaries_.count(build_key)) {
        const auto& bins = kernel.binaries_.at(build_key);
        for (size_t i = 0; i < bins.size(); ++i) {
            uint64_t hash = kernel.compute_hash();
            uint64_t proc_hash = hash ^ (i * 0x9E3779B97F4A7C15ULL);

            if (hash_to_index.find(proc_hash) == hash_to_index.end()) {
                hash_to_index[proc_hash] = binary_pool.size();
                binary_pool.push_back(
                    serialize_binary_blob(builder, kernel, *bins[i], i));
            }
            bin_indices.push_back(static_cast<int32_t>(hash_to_index[proc_hash]));
        }
    }

    return snapshot::CreateKernelDescriptor(builder,
        handle,
        name_offset,
        /* kernel_full_name */ name_offset,
        source_offset,
        /* config */ 0,
        cta_offset,
        builder.CreateVector(define_offsets),
        builder.CreateVector(bin_indices),
        /* runtime_args */ 0,
        crs_offset,
        static_cast<uint8_t>(kernel.programmable_core_type_),
        static_cast<uint8_t>(kernel.processor_class_),
        kernel.compute_hash(),
        /* watcher_kernel_id */ 0);
}
```

### 5.3 Launch Message Serialization

The `launch_msg_t` struct (`dev_msgs.h:191-193`) wraps `kernel_config_msg_t` (`dev_msgs.h:136-167`). The interceptor serializes it field by field:

```cpp
flatbuffers::Offset<snapshot::LaunchMessageSnapshot> serialize_launch_msg(
    flatbuffers::FlatBufferBuilder& builder,
    const dev_msgs::launch_msg_t& msg) {

    auto view = msg.view();
    auto kc = view.kernel_config();

    std::vector<uint32_t> config_bases;
    for (size_t i = 0; i < kc.kernel_config_base().size(); i++)
        config_bases.push_back(kc.kernel_config_base()[i]);

    std::vector<uint16_t> sem_offsets;
    for (size_t i = 0; i < kc.sem_offset().size(); i++)
        sem_offsets.push_back(kc.sem_offset()[i]);

    std::vector<snapshot::RtaOffsetEntry> rta_entries;
    for (size_t i = 0; i < kc.rta_offset().size(); i++) {
        rta_entries.push_back(snapshot::RtaOffsetEntry(
            kc.rta_offset()[i].rta_offset(),
            kc.rta_offset()[i].crta_offset()));
    }

    std::vector<uint32_t> text_offsets;
    for (size_t i = 0; i < kc.kernel_text_offset().size(); i++)
        text_offsets.push_back(kc.kernel_text_offset()[i]);

    std::vector<uint16_t> watcher_ids;
    for (size_t i = 0; i < kc.watcher_kernel_ids().size(); i++)
        watcher_ids.push_back(kc.watcher_kernel_ids()[i]);

    return snapshot::CreateLaunchMessageSnapshot(builder,
        builder.CreateVector(config_bases),
        builder.CreateVector(sem_offsets),
        kc.local_cb_offset(),
        kc.remote_cb_offset(),
        builder.CreateVectorOfStructs(rta_entries),
        kc.mode(),
        builder.CreateVector(text_offsets),
        kc.local_cb_mask(),
        kc.brisc_noc_id(),
        kc.brisc_noc_mode(),
        kc.min_remote_cb_start_index(),
        kc.host_assigned_id(),
        kc.enables(),
        builder.CreateVector(watcher_ids),
        /* go_msg fields */
        0, 0, 0, 0,
        /* core_ranges */ 0,
        /* is_multicast */ false);
}
```

---

## 6. **PROPOSED:** Deserialization for Replay

The replay tool loads a `.ttksnap` file and reconstructs in-memory state:

```cpp
// PROPOSED: tt_metal/impl/debug/snapshot/kernel_snapshot_load.cpp

namespace tt::tt_metal::kernel_capture {

struct LoadedSnapshot {
    uint64_t program_id;
    std::string architecture;
    std::string git_hash;
    bool has_debug_symbols;

    struct LoadedCore {
        uint32_t logical_x, logical_y;
        uint32_t physical_x, physical_y;

        struct LoadedBinary {
            uint32_t processor_type;
            std::vector<uint32_t> elf_words;
            uint32_t text_size;
            uint32_t text_addr;
        };
        std::vector<LoadedBinary> binaries;

        struct LoadedCB {
            uint32_t index;
            uint32_t address;
            uint32_t total_size;
            uint32_t page_size;
            uint8_t data_format;
            std::vector<uint8_t> data;
        };
        std::vector<LoadedCB> circular_buffers;

        struct LoadedSemaphore {
            uint32_t id;
            uint32_t address;
            uint32_t value;
        };
        std::vector<LoadedSemaphore> semaphores;

        // Full L1 image (if captured)
        std::vector<uint8_t> l1_image;
    };
    std::vector<LoadedCore> cores;

    // ProgramConfig offsets
    uint32_t rta_offset, sem_offset, cb_offset;
    uint32_t kernel_text_offset, kernel_text_size;
};

LoadedSnapshot load_snapshot(const std::string& path) {
    // 1. Read envelope
    auto raw_data = read_file(path);

    // 2. Check for LZ4 frame magic (0x04224D18) and decompress if needed
    std::vector<uint8_t> fb_data;
    if (raw_data.size() >= 4 &&
        raw_data[0] == 0x04 && raw_data[1] == 0x22 &&
        raw_data[2] == 0x4D && raw_data[3] == 0x18) {
        fb_data = decompress_lz4_frame(raw_data);
    } else {
        fb_data = std::move(raw_data);
    }

    // 3. Verify FlatBuffer
    auto verifier = flatbuffers::Verifier(fb_data.data(), fb_data.size());
    TT_FATAL(snapshot::VerifyKernelInvocationSnapshotBuffer(verifier),
             "Corrupted snapshot file: {}", path);

    // 4. Access (zero-copy)
    auto* root = snapshot::GetKernelInvocationSnapshot(fb_data.data());

    // 5. Validate schema version
    auto* meta = root->metadata();
    TT_FATAL(meta->schema_version_major() == 1,
             "Incompatible snapshot schema version: {}.{}",
             meta->schema_version_major(), meta->schema_version_minor());

    // 6. Reconstruct LoadedSnapshot from FlatBuffer fields
    LoadedSnapshot result;
    result.program_id = root->program_id();
    // ... (populate from core_snapshots, binary_pool, etc.)
    return result;
}

}  // namespace tt::tt_metal::kernel_capture
```

FlatBuffers provides zero-copy access to the deserialized data, meaning the binary blobs in `KernelBinaryBlob.elf_data` can be directly passed to the emulator's memory loader without an additional copy.

---

## 7. **PROPOSED:** Binary Deduplication Strategy

Within a single snapshot, binaries are stored in a shared `binary_pool` vector. Each `TensixCoreSnapshot` references binaries by index into this pool rather than embedding its own copy:

```
KernelInvocationSnapshot
    |
    +-- binary_pool: [KernelBinaryBlob]     <-- 5 unique entries for matmul
    |       [0] BRISC binary (8 KB)
    |       [1] NCRISC binary (12 KB)
    |       [2] TRISC0 binary (32 KB)
    |       [3] TRISC1 binary (24 KB)
    |       [4] TRISC2 binary (8 KB)
    |
    +-- core_snapshots: [TensixCoreSnapshot]
            [0] core (0,0): brisc_binary_index=0, ncrisc=1, trisc0=2, ...
            [1] core (0,1): brisc_binary_index=0, ncrisc=1, trisc0=2, ...
            ...
            [63] core (7,7): brisc_binary_index=0, ncrisc=1, trisc0=2, ...
```

The deduplication key is `Kernel::compute_hash()` combined with the processor type index. For a typical matmul, all 64 cores share the same 5 binaries, reducing binary storage from 64 x 84 KB = 5.25 MB to just 84 KB -- a **64x reduction**.

### 7.1 Cross-Snapshot Deduplication (Content-Addressable Store)

For repeated captures of the same program (e.g., iterating on runtime args with the same kernel binaries), a content-addressable binary store provides additional savings:

**PROPOSED:** A `BinaryStore` class computes a 64-bit FNV1a hash of each ELF binary and stores it at `{store_dir}/{hash:016x}.elf` if not already present. The snapshot manifest references binaries by hash. For 1000 dispatches of the same 3-kernel program: without dedup 60 MB, with dedup 60 KB (1000x reduction).

This is an optional optimization; the initial implementation stores self-contained `.ttksnap` files.

---

## 8. **PROPOSED:** Compression Pipeline

### 8.1 Two-Tier Compression

| Tier | Algorithm | Use Case | Ratio | Speed |
|------|-----------|----------|-------|-------|
| Capture-time | LZ4 (frame format) | Real-time capture; minimize dispatch-path stall | ~2.5x | ~2 GB/s compress |
| Archival | zstd (level 3) | Post-capture; `.ttksnap` file on disk | ~4-6x | ~400 MB/s compress |

LZ4 is applied immediately after FlatBuffer serialization in the capture path. Zstd is applied as a background post-processing step when flushing from the ring buffer to disk.

### 8.2 LZ4 Frame Compression

```cpp
// PROPOSED: LZ4 frame compression for .ttksnap files

std::vector<uint8_t> compress_lz4_frame(const std::vector<uint8_t>& input) {
    LZ4F_preferences_t prefs = LZ4F_INIT_PREFERENCES;
    prefs.compressionLevel = 1;  // Fastest compression
    prefs.frameInfo.contentSize = input.size();

    size_t max_size = LZ4F_compressFrameBound(input.size(), &prefs);
    std::vector<uint8_t> output(max_size);

    size_t compressed_size = LZ4F_compressFrame(
        output.data(), output.size(),
        input.data(), input.size(),
        &prefs);

    if (LZ4F_isError(compressed_size)) {
        throw std::runtime_error(
            std::string("LZ4 compression failed: ") +
            LZ4F_getErrorName(compressed_size));
    }

    output.resize(compressed_size);
    return output;
}
```

### 8.3 Streaming Compression for Large Snapshots

For 64-core snapshots exceeding 50 MB uncompressed, **PROPOSED:** a streaming LZ4 writer avoids allocating the full uncompressed buffer:

```cpp
// PROPOSED: Streaming LZ4 frame writer
class StreamingSnapshotWriter {
public:
    explicit StreamingSnapshotWriter(const std::string& path);

    void write(const uint8_t* data, size_t size) {
        size_t bound = LZ4F_compressBound(size, nullptr);
        if (bound > compress_buf_.size()) compress_buf_.resize(bound);

        size_t compressed = LZ4F_compressUpdate(
            ctx_, compress_buf_.data(), compress_buf_.size(),
            data, size, nullptr);
        if (compressed > 0)
            file_.write(reinterpret_cast<char*>(compress_buf_.data()), compressed);
        uncompressed_total_ += size;
    }

    void finish();
    uint64_t uncompressed_size() const { return uncompressed_total_; }

private:
    std::ofstream file_;
    LZ4F_cctx* ctx_ = nullptr;
    std::vector<uint8_t> compress_buf_{64 * 1024};
    uint64_t uncompressed_total_ = 0;
};
```

### 8.4 Sparse L1 Representation

When capturing full L1 dumps, large regions may be zero-filled (unused L1 space). A sparse representation stores only non-zero regions as separate `L1RegionSnapshot` entries:

```cpp
// PROPOSED: Sparse L1 representation
std::vector<std::pair<uint32_t, uint32_t>> find_nonzero_regions(
    const uint8_t* data, uint32_t size, uint32_t granularity = 256) {

    std::vector<std::pair<uint32_t, uint32_t>> regions;
    uint32_t region_start = 0;
    bool in_region = false;

    for (uint32_t i = 0; i < size; i += granularity) {
        uint32_t chunk_size = std::min(granularity, size - i);
        bool all_zero = true;
        for (uint32_t j = 0; j < chunk_size; j += sizeof(uint64_t)) {
            if (*reinterpret_cast<const uint64_t*>(data + i + j) != 0) {
                all_zero = false;
                break;
            }
        }
        if (!all_zero && !in_region) { region_start = i; in_region = true; }
        else if (all_zero && in_region) {
            regions.push_back({region_start, i - region_start});
            in_region = false;
        }
    }
    if (in_region) regions.push_back({region_start, size - region_start});
    return regions;
}
```

For a typical Wormhole B0 core with 1 MB L1, active regions (firmware + kernel text + CB data + RTAs + semaphores) typically occupy 200-500 KB. The sparse representation reduces the L1 dump from 1 MB to 200-500 KB before compression, and to 80-200 KB after LZ4.

### 8.5 Compression Ratio Summary

| Algorithm | Compression Speed | Ratio on Tile Data | Ratio on ELF | Use Case |
|-----------|------------------|-------------------|-------------|----------|
| None | N/A | 1.0x | 1.0x | Development (fastest capture) |
| LZ4 (level 1) | ~2 GB/s | 2.0-2.5x | 2.0-3.0x | Default capture |
| zstd (level 3) | ~400 MB/s | 3.0-4.0x | 3.0-4.0x | CI artifacts |
| zstd (level 19) | ~10 MB/s | 4.0-6.0x | 4.0-5.5x | Long-term archival |

---

## 9. **PROPOSED:** Storage Layout and File Format

### 9.1 Directory Structure

```
$TT_METAL_HOME/generated/kernel_captures/
  |
  +-- session_<timestamp>/
  |     |
  |     +-- manifest.json                  # Session-level metadata
  |     +-- 000001_prog42_core3_4.ttksnap
  |     +-- 000002_prog42_core5_6.ttksnap
  |     +-- ...
  |     +-- postmortem_core3_4.ttksnap     # Failure-triggered captures
```

### 9.2 Manifest JSON

```json
{
    "session_id": "20260505_143022_pid12345",
    "tt_metal_version": "0.58.0",
    "git_hash": "621b949",
    "architecture": "WORMHOLE_B0",
    "capture_mode": "selective",
    "captures": [
        {
            "file": "000001_prog42_core3_4.ttksnap",
            "program_id": 42,
            "kernel_names": ["reader_unary", "eltwise_unary_sfpu", "writer_unary"],
            "cores": ["(3,4)"],
            "uncompressed_bytes": 55000,
            "compressed_bytes": 22000,
            "timestamp_ns": 1746460222000000000
        }
    ]
}
```

### 9.3 `flush_to_disk()` Implementation

```cpp
std::string KernelSnapshotCaptureContext::flush_to_disk(
    const CapturedInvocation& invocation) {

    std::filesystem::create_directories(config_.output_dir);

    uint64_t idx = invocation_counter_.fetch_add(1, std::memory_order_relaxed);
    std::string filename = fmt::format(
        "{:06d}_prog{}_core{}_{}.ttksnap",
        idx,
        invocation.program_id,
        invocation.core_snapshots.empty() ? 0 :
            invocation.core_snapshots[0].logical_core.x,
        invocation.core_snapshots.empty() ? 0 :
            invocation.core_snapshots[0].logical_core.y);

    std::string path = config_.output_dir + "/" + filename;

    // Serialize to FlatBuffer
    flatbuffers::FlatBufferBuilder builder(1 << 20);  // 1 MB initial
    auto root_offset = serialize_snapshot(builder, /* ... */);
    builder.Finish(root_offset);

    // Compress with LZ4
    std::vector<uint8_t> fb_data(
        builder.GetBufferPointer(),
        builder.GetBufferPointer() + builder.GetSize());
    std::vector<uint8_t> compressed = compress_lz4_frame(fb_data);

    // Write to disk
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    file.close();

    log_info(tt::LogMetal,
        "Kernel snapshot: {} ({} bytes uncompressed, {} bytes compressed)",
        path, fb_data.size(), compressed.size());

    return path;
}
```

---

## 10. **PROPOSED:** Delta Serialization for H4-update Path

When the secondary hook (H4-update at dispatch.cpp:2127) fires on subsequent dispatches of the same program, only runtime args and CB addresses change. A full snapshot is wasteful.

**PROPOSED:** A `KernelSnapshotDelta` table for differential capture:

```flatbuffers
// PROPOSED: Addition to kernel_snapshot.fbs

table KernelSnapshotDelta {
    base_snapshot_id: uint64;
    dispatch_count: uint64;
    timestamp_ns: uint64;

    // Changed runtime args (sparse: only cores with modified args)
    rta_updates: [RuntimeArgsPerCore];

    // Changed CB addresses (only if CBs were re-allocated)
    cb_address_updates: [CBAddressUpdate];
}

table CBAddressUpdate {
    cb_id: uint64;
    new_address: uint32;
}
```

Delta snapshots are typically 1-10 KB, representing a 10-100x reduction compared to full snapshots. This makes `EveryDispatch` mode practical for iterative debugging.

---

## 11. Concrete Size Estimates

### 11.1 Worked Example: BFloat16 Matmul on 8x8 Grid

A matrix multiplication kernel deployed on an 8x8 grid (64 Tensix cores) using BFloat16 tiles.

**Tile dimensions**: 32x32 elements at 2 bytes each (BFloat16) = 2048 bytes per tile + 16 bytes header = 2064 bytes per tile.

**Per-core state (shared binaries -- stored once in binary_pool):**

| Component | Count | Size per Item | Total |
|-----------|-------|--------------|-------|
| BRISC binary | 1 | ~16 KB | 16 KB |
| NCRISC binary | 1 | ~14 KB | 14 KB |
| TRISC0 binary (unpack) | 1 | ~12 KB | 12 KB |
| TRISC1 binary (math) | 1 | ~20 KB | 20 KB |
| TRISC2 binary (pack) | 1 | ~10 KB | 10 KB |
| **Binary pool total** | **5** | | **72 KB** |

**Per-core state (captured per core):**

| Component | Count | Size | Per 64 Cores |
|-----------|-------|------|-------------|
| Runtime args (BRISC) | 1 | ~160 B | 10 KB |
| Runtime args (NCRISC) | 1 | ~160 B | 10 KB |
| CB configs (3 CBs) | 3 | ~64 B each | 12 KB |
| Semaphores (4 active) | 4 | 16 B each | 4 KB |
| Launch msg | 1 | ~128 B | 8 KB |
| L1 regions (firmware, stack) | 2 | ~2 KB each | 256 KB |

**64-core totals (structured state only, no tile data):**

| Capture Scope | Size | Compressed (LZ4, ~2.5x) |
|---------------|------|-------------------------|
| Binary pool (deduplicated) | 72 KB | 29 KB |
| Per-core state (64 cores) | ~300 KB | 120 KB |
| Metadata + kernel descriptors | ~12 KB | 5 KB |
| **Total (no L1 data)** | **~384 KB** | **~154 KB** |

**With CB tile data (8 tiles per CB, 3 CBs per core):**

| Component | Size |
|-----------|------|
| Base snapshot | 384 KB |
| CB tile data (64 cores x 3 CBs x 8 tiles x 2064 B) | 3.1 MB |
| **Total** | **~3.5 MB** |
| **LZ4 compressed** | **~1.4 MB** |

**With full L1 dump (all cores):**

| Component | Size |
|-----------|------|
| Base snapshot | 384 KB |
| Full L1 (64 x 1.5 MB for BH) | 96 MB |
| **Total uncompressed** | **~96.4 MB** |
| **LZ4 compressed** | **~38 MB** |

### 11.2 Worked Example: Single-Core Eltwise Op

A simple element-wise operation on a single core:

| Component | Size |
|-----------|------|
| TRISC0 binary | ~8 KB |
| TRISC1 binary | ~6 KB |
| TRISC2 binary | ~6 KB |
| CB 0 config + CB 16 config | ~128 B |
| Semaphores (2) | 32 B |
| Runtime args | ~80 B |
| Launch msg | 128 B |
| Metadata | ~1.5 KB |
| **Total (no CB data)** | **~22 KB** |
| **Total (with 4-tile CB data per CB, BF16)** | **~38 KB** |
| **LZ4 compressed** | **~15 KB** |

### 11.3 Lightweight Metadata Record Size

The metadata-only record (File 02, MetadataOnly mode):

| Field | Size |
|-------|------|
| Program ID + runtime ID | 16 B |
| Timestamp | 8 B |
| Dispatch sequence number | 4 B |
| Kernel names (3 x string ref) | ~200 B |
| Core assignments | ~64 B |
| Compile hash | 8 B |
| **Total per record** | **~300 B** |
| **Ring buffer (16 records)** | **~5 KB** |
| **Ring buffer (1024 records)** | **~300 KB** |

### 11.4 Comparison with LightMetal

| LightMetal Binary | Kernel Snapshot |
|-------------------|-----------------|
| ~100-200 KB | ~22 KB - 38 MB (depending on scope) |
| Contains: API commands, buffer create/write, kernel create, set args | Contains: compiled ELFs, CB data, L1 dumps, full config |
| No compiled binaries | All compiled binaries included |
| Requires source tree + recompilation for replay | Self-contained for replay |
| Captures entire program lifetime | Captures single invocation point-in-time |

The snapshot is 0.1-200x larger than a LightMetal binary depending on scope. The smallest snapshots (single-core eltwise, no L1) are actually smaller than typical LightMetal binaries because they capture only the resolved state rather than the full API call sequence. With L1 data, snapshots are substantially larger, but this is an acceptable trade-off for a debugging tool.

---

## 12. Schema Versioning Strategy

**PROPOSED:** Two-level versioning:

### 12.1 Schema Version

`CaptureMetadata.schema_version_major` and `schema_version_minor`:

- **Major version bump**: Breaking changes (removed fields, changed semantics). The replay tool rejects mismatched major versions.
- **Minor version bump**: Additive changes (new optional fields, new table types). The replay tool accepts the same major version with any minor version.

FlatBuffers' wire format inherently supports forward compatibility (unknown fields are ignored), but the explicit version fields enable informative error messages.

### 12.2 Build Provenance

```
tt_metal_git_hash:  "621b949"              -- Exact commit
compiler_version:   "riscv-tt-elf-g++ 12.2.0"  -- RISC-V toolchain version
compiler_flags:     "-Os -DARCH_WORMHOLE"  -- Flags affecting binary content
```

These fields are critical for reproducing the exact build environment when sharing snapshots across teams or systems.

### 12.3 Compatibility Matrix

| Writer Version | Reader Version 1 | Reader Version 2 |
|:-:|:-:|:-:|
| 1 | Full support | Full support (superset) |
| 2 | Reads v1 fields, ignores v2 | Full support |

---

## 13. Snapshot Content Coverage vs. Ch2 State Inventory

| Ch2 Category | Schema Location | Captured? |
|-------------|-----------------|:---------:|
| Kernel ELF binaries | `binary_pool[].elf_data` | **Yes** |
| Compile-time args | `KernelDescriptor.compile_args` | **Yes** |
| Kernel defines | `KernelDescriptor.defines` | **Yes** |
| Runtime args (per-core) | `RuntimeArgsPerCore.args` | **Yes** |
| Common runtime args | `RuntimeArgsSnapshot.common_args` | **Yes** |
| CB configuration | `CircularBufferSnapshot.*` | **Yes** |
| CB tile data | `CircularBufferSnapshot.data_contents` | **Opt-in** |
| Semaphore values | `SemaphoreSnapshot.*` | **Yes** |
| Launch messages | `LaunchMessageSnapshot.*` | **Yes** |
| Go signal state | `LaunchMessageSnapshot.go_signal` | **Yes** |
| Program config (L1 layout) | `ProgramConfigSnapshot.*` | **Yes** |
| Architecture descriptor | `CaptureMetadata.architecture` | **Yes** |
| NOC transient state | Not captured | **No** (physically uncapturable from host) |
| Full L1 memory | `L1RegionSnapshot.data` | **Opt-in** |

**Coverage:** 13 of 14 state categories are captured (93%). NOC transient state is excluded because it cannot be read atomically from the host. With L1 capture enabled, the interceptor achieves full state coverage for deterministic replay.

---

## Key Takeaways

- **The FlatBuffer schema defines the complete `.ttksnap` format** with `KernelInvocationSnapshot` as the root type (file identifier `"TKSN"`, file extension `.ttksnap`), extending `base_types.fbs` and `program_types.fbs` for type reuse while remaining independently parseable. The `ProgramConfigSnapshot` includes all 10 fields from the actual `ProgramConfig` struct (program_impl.hpp lines 97-108), including `dfb_offset` and `dfb_size`.

- **Binary deduplication reduces 64-core matmul snapshot size by 64x for binary data**, bringing a structured-state-only snapshot to ~384 KB uncompressed (154 KB LZ4 compressed) without tile data. Cross-snapshot deduplication via content-addressable FNV1a hashing provides up to 1000x reduction for iterative workloads.

- **Two-tier compression (LZ4 at capture, zstd for archival) balances speed against ratio**: LZ4 adds only ~5-20 us to the capture path at ~2 GB/s, while zstd achieves 4-6x compression for stored files. Sparse L1 representation further reduces full L1 dumps by 50-80% before compression.

- **Per-core organization via `TensixCoreSnapshot`** groups all state for a single core (binaries by index, CBs, semaphores, L1 regions, runtime args, launch message, program config), supporting the single-core isolation workflow described in File 01.

- **Serialization code maps 1:1 to named C++ struct members**: `Kernel` class members (kernel.hpp), `CircularBufferImpl`, `Semaphore` (semaphore.hpp lines 17-48), `KernelGroup` (program_impl.hpp lines 69-94), `ProgramConfig` (program_impl.hpp lines 97-108), and `kernel_config_msg_t`/`go_msg_t` (dev_msgs.h lines 136-193).

- **The `.ttksnap` file format is self-contained** with schema versioning (major/minor + git hash), compression metadata, and a manifest JSON for session-level indexing. Storage is organized at `$TT_METAL_HOME/generated/kernel_captures/` with per-session directories. Delta snapshots (1-10 KB) make frequent capture practical on the H4-update path.
