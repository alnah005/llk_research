# Extending LightMetal for Kernel Capture

This section proposes concrete extensions to LightMetal's FlatBuffer schema and C++ capture infrastructure that would close the gaps identified in Section 2. All proposals are marked with **PROPOSED:** and represent new designs that do not exist in the current codebase. Where possible, the proposals build on existing LightMetal patterns (FlatBuffer union extension, capture helpers, object-to-global-id mapping, data_collection hooks) to minimize integration friction and maximize reuse.

All source references cite files in `/localdev/salnahari/testing_dir/tt-metal/`.

---

## 1. FlatBuffer Design Constraints

Before presenting the proposed schema, it is important to understand the FlatBuffer constraints that shape all design decisions. These constraints are not hypothetical -- they are enforced by the FlatBuffer compiler and runtime library used in TT-Metal.

### 1.1 Union Member Limit

The `CommandType` union has 17 members out of a maximum 256 (Section 1, 4.2). The proposal adds 5 new members, bringing the total to 22 -- well within the limit. The design groups related data into fewer, broader tables to conserve union slots for long-term growth.

### 1.2 No Union Inheritance

FlatBuffers does not support union inheritance or polymorphic unions. A `KernelSnapshotCommand` cannot "extend" `CreateKernelCommand`. If a snapshot shares fields with the existing kernel command, those fields must either be duplicated or referenced via global ID. The proposed `KernelSnapshotCommand` takes the reference approach: it points to the kernel by `kernel_global_id` rather than re-embedding configuration.

### 1.3 No Native Optional Scalars (in current usage)

The TT-Metal codebase uses FlatBuffers without the `optional` keyword for scalar fields. Instead, it uses wrapper tables: `BoolOptional`, `Uint8Optional`, `Uint32Optional` (defined in `base_types.fbs`, lines 86-89). New schema extensions follow this established pattern for consistency, even though FlatBuffers 23.5+ supports native optional scalars.

### 1.4 Vector Nesting Workaround

FlatBuffers does not support vectors of vectors (`[[uint32]]`). The existing schema works around this with wrapper tables: `UInt32Vector` in `program_types.fbs` (lines 76-78) wraps a single `[uint32]` vector, and `SetRuntimeArgsUint32VecPerCoreCommand` uses `args: [UInt32Vector]` for a vector-of-vectors. L1 memory snapshots across multiple cores require the same wrapper table pattern.

### 1.5 File Identifier

The `file_identifier` directive declares a 4-byte magic number in the binary header. The current schema does not use it. Adding one (proposed below) enables runtime detection of file format.

### 1.6 Field Ordering for Compatibility

**Critical rule**: Union members must only be **appended**, never reordered. Union member indices are encoded as uint8 tags in the binary. If member ordering changes, all existing binaries become unreadable. Similarly, new fields in existing tables must be appended to the end -- FlatBuffer tables use vtable indirection, so old readers skip unknown trailing fields and new readers provide defaults for missing fields.

---

## 2. PROPOSED: `KernelSnapshotCommand` FlatBuffer Schema

The central extension is a new FlatBuffer command type that captures everything needed to replay a single kernel invocation in isolation. Rather than adding many independent commands (one for binary capture, one for L1, one for semaphores), the primary proposal is a self-contained `KernelSnapshotCommand` that bundles all per-kernel capture data into one table. This follows the principle of atomic capture: either you have everything needed to replay the kernel, or you have nothing.

Standalone snapshot commands (L1, semaphore, register) are also proposed for diagnostic use cases that do not require the full kernel context.

### 2.1 PROPOSED: Schema Definition

The following FlatBuffer IDL would be placed in a new file `kernel_snapshot_types.fbs`, included by `command.fbs`:

```flatbuffers
// PROPOSED: kernel_snapshot_types.fbs
// Extends command.fbs with kernel-level capture commands.

include "base_types.fbs";
include "program_types.fbs";
include "buffer_types.fbs";

namespace tt.tt_metal.flatbuffer;

// ============================================================
// Per-RISC Kernel Binary
// ============================================================

// PROPOSED: Stores compiled binary for a single RISC-V processor.
// One kernel may produce multiple binaries (BRISC, NCRISC, TRISC0/1/2).
// The binary data comes from Kernel::binaries(build_key), which returns
// a vector of const ll_api::memory* (kernel.hpp, line 197).
table KernelBinary {
  processor_type: uint8;           // HalProcessorType index (0=BRISC, 1=NCRISC, 2=TRISC0, etc.)
  processor_class: uint8;          // HalProcessorClassType (DM=0, Compute=1, Ethernet=2)
  binary_data: [uint8];            // Raw ELF binary bytes
  text_size: uint32;               // Size of .text section
  packed_size: uint32;             // Total packed size for dispatch
  binary_hash: uint64;             // Hash for identity/diff (e.g., xxhash64)
}

// ============================================================
// L1 Memory Region Snapshot
// ============================================================

// PROPOSED: Captures a contiguous region of L1 memory at a specific core.
// Uses the sparse capture strategy: only active regions (CBs, RT args,
// semaphores) are captured, not the full 1.5 MB L1 space.
table L1MemoryRegion {
  l1_address: uint32;              // Start address in L1 address space
  data: [uint8];                   // Raw memory contents
  label: string;                   // Human-readable label (e.g., "CB0", "semaphores", "stack")
}

// PROPOSED: Full L1 snapshot for a single core.
table CoreL1Snapshot {
  core_coord: CoreCoord;           // Logical core coordinate
  regions: [L1MemoryRegion];       // Selective L1 regions captured
  timestamp_ns: uint64;            // Capture timestamp for ordering
}

// ============================================================
// Semaphore State
// ============================================================

// PROPOSED: Captures all semaphore values for a single core.
table CoreSemaphoreState {
  core_coord: CoreCoord;           // Logical core coordinate
  semaphore_values: [uint32];      // Values of all semaphores (8 on Wormhole, may differ per arch)
  semaphore_base_addr: uint32;     // L1 base address of semaphore region
}

// ============================================================
// Circular Buffer Runtime State
// ============================================================

// PROPOSED: Captures the runtime state of a circular buffer beyond its config.
// The config is already captured by CreateCircularBufferCommand; this table
// adds the runtime data (actual contents, pointer positions, allocated address).
table CBRuntimeState {
  cb_global_id: uint32;            // Reference to CB via global ID
  cb_index: uint8;                 // CB index (0-31)
  l1_address: uint32;              // Actual L1 address after allocation (locally_allocated_address_)
  write_ptr: uint32;               // Current write pointer (tiles written)
  read_ptr: uint32;                // Current read pointer (tiles read)
  num_pages_available: uint32;     // Pages available for read
  data_snapshot: [uint8];          // Optional: actual data in the CB at capture time
}

// ============================================================
// Named Compile Args (closing Gap 7 from Section 2)
// ============================================================

// PROPOSED: Key-value pair for named_compile_args, which are defined as
// std::unordered_map<std::string, uint32_t> in the C++ config structs
// (kernel_types.hpp for DM/Compute, kernel.hpp for Ethernet).
table NamedCompileArgEntry {
  key: string;
  value: uint32;
}

// ============================================================
// KernelSnapshotCommand -- The Core Addition
// ============================================================

// PROPOSED: Captures the complete state needed to replay a single kernel
// invocation in isolation. This is the primary new command type for the
// interceptor. It references the kernel and program by global_id (from
// the existing object map system) and embeds all device-side state that
// LightMetal currently omits.
table KernelSnapshotCommand {
  // --- Identity ---
  snapshot_id: uint32;             // Unique snapshot ID within the binary
  kernel_global_id: uint32;        // Reference to kernel (must have prior CreateKernelCommand)
  program_global_id: uint32;       // Reference to owning program
  dispatch_ordinal: uint32;        // Nth invocation of this program (for repeated dispatch)

  // --- Kernel Binaries (closing Gap 1) ---
  binaries: [KernelBinary];        // Compiled binary per RISC processor

  // --- Core Assignment ---
  core_spec: CoreSpec;             // Which cores this kernel runs on
  arch: Arch;                      // Target architecture

  // --- Extended Compile State (closing Gap 7) ---
  // named_compile_args and opt_level close the config field gaps.
  // hlk_desc (Ch2, Section 1, Item 6) is not serialized separately because
  // the captured ELF binaries already encode the HLK descriptor's effects;
  // for recompilation scenarios, hlk_desc would need its own table.
  named_compile_args: [NamedCompileArgEntry];  // The missing named args
  opt_level: uint8;                // KernelBuildOptLevel enum value (0=O0..6=Oz)
  source_code: string;             // Optional: actual source code text (for inline kernels)
  source_hash: uint64;             // Hash of kernel source for consistency check

  // --- Runtime Arguments (closing Gap 8 for common args) ---
  runtime_args_per_core: [UInt32Vector]; // Per-core runtime args (indexed by core list order)
  common_runtime_args: [uint32];   // Common runtime args shared across cores

  // --- Input Memory State (closing Gaps 2, 3) ---
  l1_snapshots: [CoreL1Snapshot];  // L1 state at each core before kernel launch
  semaphore_states: [CoreSemaphoreState]; // Semaphore values before kernel launch
  cb_states: [CBRuntimeState];     // CB runtime states (config + data + pointers)

  // --- DRAM Input Buffers ---
  dram_input_snapshots: [DRAMBufferSnapshot]; // DRAM regions read by this kernel

  // --- Metadata ---
  capture_mode: CaptureMode;       // Whether this is a full or selective snapshot
  capture_timestamp_ns: uint64;    // Wall-clock time of capture
}

// PROPOSED: Captures a region of DRAM relevant to a kernel.
table DRAMBufferSnapshot {
  buffer_global_id: uint32;        // Reference to Buffer via global ID
  bank_id: uint16;                 // DRAM bank ID
  address: uint64;                 // Device address
  data: [uint8];                   // Raw buffer contents
  data_format: DataFormat;         // Interpretation hint
}

// PROPOSED: Controls snapshot depth.
enum CaptureMode : uint8 {
  Full = 0,           // Capture everything: binaries, L1, semaphores, DRAM
  Selective = 1,      // Capture only explicitly requested regions
  ConfigOnly = 2,     // Capture config/metadata but no data (minimal overhead)
}

// ============================================================
// Standalone Snapshot Commands (for diagnostic use)
// ============================================================

// PROPOSED: Standalone command to snapshot L1 at arbitrary points.
table L1SnapshotCommand {
  snapshot_id: uint32;
  core_snapshots: [CoreL1Snapshot];
  capture_trigger: string;         // Description of what triggered this snapshot
}

// PROPOSED: Standalone command to snapshot semaphore state.
table SemaphoreSnapshotCommand {
  snapshot_id: uint32;
  semaphore_states: [CoreSemaphoreState];
}

// PROPOSED: Captures hardware register configuration for a core.
// Only used in DEEP_FULL capture mode (mid-execution debugging).
table CoreRegisterState {
  core_coord: CoreCoord;
  register_bank: string;           // "risc_gprs", "tensix_cfg", "sfpu", "dest", etc.
  register_data: [uint8];          // Raw register block contents
  register_base_addr: uint32;      // Base address of register block
}

table RegisterStateCommand {
  snapshot_id: uint32;
  core_registers: [CoreRegisterState];
}

// PROPOSED: Captures state after kernel execution for comparison/validation.
// Follows the LightMetalCompareCommand pattern (Section 1, Item 10).
table KernelOutputSnapshotCommand {
  snapshot_id: uint32;
  kernel_global_id: uint32;
  l1_snapshots: [CoreL1Snapshot];  // L1 state after kernel execution
  semaphore_states: [CoreSemaphoreState]; // Semaphore state after execution
  cb_states: [CBRuntimeState];     // CB state after execution
  execution_cycles: uint64;        // Cycle count for the kernel execution
}
```

### 2.2 PROPOSED: Extended `CommandType` Union

The `CommandType` union in `command.fbs` would be extended by appending new members:

```flatbuffers
// PROPOSED: Extended union (appended to existing command.fbs lines 118-136)
union CommandType {
  // --- Existing (17 members, unchanged) ---
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

  // --- PROPOSED additions (5 new members) ---
  KernelSnapshotCommand,           // Primary: full kernel state capture
  L1SnapshotCommand,               // Standalone L1 memory dump
  SemaphoreSnapshotCommand,        // Standalone semaphore state dump
  RegisterStateCommand,            // Hardware register state capture
  KernelOutputSnapshotCommand,     // Post-execution state capture
}
```

This brings the union to 22 members (239 remaining slots within the uint8 limit).

### 2.3 Size Estimates

| New Table | Estimated Per-Instance Size | Frequency |
|-----------|---------------------------|-----------|
| `KernelSnapshotCommand` (Full mode, 64 cores) | 5-33 MB (dominated by L1) | Once per kernel per invocation |
| `KernelSnapshotCommand` (Selective, 1 core) | 65-200 KB | Per target kernel |
| `KernelSnapshotCommand` (ConfigOnly) | 1-5 KB | Once per kernel |
| `L1SnapshotCommand` | 50 KB - 1.5 MB per core | On demand |
| `SemaphoreSnapshotCommand` | 32 bytes per core | On demand |
| `RegisterStateCommand` | ~1-4 KB per core | On demand |
| `KernelOutputSnapshotCommand` | Same as KernelSnapshot (L1 portion) | Once per kernel per invocation |

For a typical single-kernel debugging session (one compute kernel, one core, 4 input CBs at 16 KB each, 3 TRISC ELFs at 16 KB each):

$$\text{Snapshot} \approx 48\text{ KB (binaries)} + 64\text{ KB (CB data)} + 5\text{ KB (config+args)} + 32\text{ B (semaphores)} \approx 117\text{ KB}$$

For a 64-core matmul in Full capture mode:

$$\text{Snapshot} \approx 48\text{ KB (binaries, shared)} + 64 \times 128\text{ KB (L1 regions)} + 64 \times 5\text{ KB (args)} + 64 \times 32\text{ B (sems)} \approx 8.5\text{ MB}$$

These sizes are consistent with the capture budget analysis in Section 2, Item 14.

---

## 3. PROPOSED: Deep Capture Mode Architecture

### 3.1 Strategy: Parallel Capture Path

Rather than modifying LightMetal's existing capture behavior, the proposal adds a parallel "deep capture" mode that can be enabled independently. The existing shallow capture (host API calls only) continues to work unchanged. This separation serves three purposes:

1. **Leverage existing infrastructure**: The global ID maps, FlatBuffer schema, `TraceScope` guard, and capture helper pattern represent significant working infrastructure. Rewriting from scratch would be wasteful and risky.

2. **Backward compatibility**: Standard LightMetal capture remains valuable for its original use cases (offline replay, regression testing). Deep capture is an opt-in layer, not a replacement.

3. **Incremental adoption**: Teams can enable deep capture for specific debugging sessions without affecting production workloads or CI pipelines.

```
                    +----------------------------+
                    | LightMetalCaptureContext   |
                    | is_tracing_ : bool         |
                    | deep_capture_enabled_: bool |  // PROPOSED
                    | capture_mode_ : CaptureMode |  // PROPOSED
                    | builder_                   |
                    | cmds_vec_                  |
                    | object maps (4 existing)   |
                    +----------------------------+
                              |
                    +---------+---------+
                    |                   |
              Shallow Capture     Deep Capture
              (existing, depth 1) (PROPOSED, dispatch boundary)
                    |                   |
           CreateKernelCommand   KernelSnapshotCommand
           SetRuntimeArgs...     (binaries + L1 + sems)
           CreateCBCommand       KernelOutputSnapshotCommand
```

### 3.2 PROPOSED: The `DeepTraceScope` Re-entrancy Guard

The key architectural decision is where the deep capture hooks are inserted. As analyzed in Section 1, LightMetal's `TraceScope` fires at depth == 1 (the public API boundary). Deep capture must hook at a different level -- specifically at the `EnqueueProgram` dispatch boundary, which is inside the API implementation (depth >= 2 in the existing model).

The recommended approach is a separate `DeepTraceScope` with its own `thread_local` depth counter, independent of `TraceScope`. This cleanly separates the two capture modes.

**Architecture note**: Two alternative approaches were considered: (A) reusing `TraceScope` with a configurable depth range (`depth >= 2 && depth <= N`), and (B) hooking entirely outside the `TraceScope` model. Option A couples the two systems and is fragile if the call depth changes across TT-Metal versions. Option B loses the re-entrancy protection. The separate counter approach (below) preserves re-entrancy safety without coupling.

```cpp
// PROPOSED: Separate re-entrancy guard for deep capture
// Added to host_api_capture_helpers.hpp

struct DeepTraceScope {
    static inline thread_local int depth = 0;
    DeepTraceScope() { ++depth; }
    ~DeepTraceScope() { --depth; }
};

#if defined(TT_ENABLE_DEEP_CAPTURE) && (TT_ENABLE_DEEP_CAPTURE == 1)

#define DEEP_CAPTURE_ENTRY() tt::tt_metal::DeepTraceScope __deepTraceScopeGuard

#define DEEP_CAPTURE_CALL(capture_func, ...)                                       \
    do {                                                                           \
        log_trace(                                                                 \
            tt::LogMetalTrace,                                                     \
            "DEEP_CAPTURE_CALL: {} via {} depth: {}",                              \
            #capture_func,                                                         \
            __FUNCTION__,                                                           \
            tt::tt_metal::DeepTraceScope::depth);                                  \
        if (LightMetalCaptureContext::get().deep_capture_enabled()                  \
            && tt::tt_metal::DeepTraceScope::depth == 1) {                         \
            capture_func(__VA_ARGS__);                                              \
        }                                                                           \
    } while (0)

#else  // TT_ENABLE_DEEP_CAPTURE not defined

#define DEEP_CAPTURE_ENTRY() (void)0
#define DEEP_CAPTURE_CALL(capture_func, ...) (void)0

#endif
```

This follows the exact pattern of the existing `LIGHT_METAL_TRACE_FUNCTION_ENTRY()` and `LIGHT_METAL_TRACE_FUNCTION_CALL()` macros (Section 1, Item 3.3), including the `log_trace()` diagnostic and the compile-time gate. The `depth == 1` check ensures that if a deep capture hook internally calls another deep-capture-instrumented function, only the outermost fires.

### 3.3 PROPOSED: Capture Trigger Point

The capture trigger is at the `EnqueueProgram` boundary -- the point where all kernel state is finalized and the program is about to be dispatched. Two options are available:

**Option A: Hook inside `EnqueueProgram` implementation**

Insert a capture call inside the `EnqueueProgram` codepath, after program compilation but before dispatch to hardware:

```cpp
// PROPOSED: In the EnqueueProgram implementation path
void EnqueueProgramImpl(CommandQueue& cq, Program& program, bool blocking) {
    // ... existing compilation and setup ...

    // PROPOSED: Deep capture hook (fires only if deep_capture_enabled_ is true)
    DEEP_CAPTURE_ENTRY();
    DEEP_CAPTURE_CALL(CaptureKernelSnapshots, program, &device, dispatch_ordinal);

    // ... existing dispatch logic ...
}
```

**Option B: External interceptor wrapping `EnqueueProgram`**

The interceptor (designed in Chapter 4) wraps the `EnqueueProgram` call:

```cpp
// PROPOSED: Interceptor layer (Chapter 4)
void InterceptedEnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    auto snapshots = CapturePreDispatchState(program);
    EnqueueProgram(cq, program, blocking);  // Original call
    auto post_state = CapturePostDispatchState(program);
    StoreKernelSnapshots(snapshots, post_state);
}
```

**Recommendation**: Option B is preferred for the full interceptor design because it does not modify the core TT-Metal dispatch path and aligns with the interceptor architecture described in Chapter 4. Option A is suitable for an initial prototype within TT-Metal's own codebase. Both options use the same `DeepTraceScope` mechanism.

### 3.4 PROPOSED: Activation Controls

Deep capture would be activated via environment variables and/or programmatic API, following the pattern established by `TT_LIGHT_METAL_SHOW_READS` and `TT_LIGHT_METAL_DISABLE_CHECKING` in `lightmetal_replay_impl.cpp` (lines 81-82):

```bash
# Enable deep capture for all kernels
export TT_DEEP_CAPTURE=1

# Enable deep capture only for compute kernels
export TT_DEEP_CAPTURE=compute

# Target specific kernel file patterns
export TT_DEEP_CAPTURE_FILTER="*matmul*compute*"

# Target specific dispatch ordinals
export TT_DEEP_CAPTURE_DISPATCH_RANGE="45-50"

# Set maximum snapshot count (to bound capture size)
export TT_DEEP_CAPTURE_MAX_SNAPSHOTS=100
```

```cpp
// PROPOSED: Programmatic API in LightMetalCaptureContext
void set_deep_capture(bool enabled);
void set_capture_mode(CaptureMode mode);  // Full, Selective, ConfigOnly
void set_capture_filter(const std::string& kernel_name_pattern);
void set_capture_dispatch_range(uint32_t first, uint32_t last);
bool deep_capture_enabled() const;
```

### 3.5 PROPOSED: Capture Mode Behavior

The `CaptureMode` enum controls what data is collected per snapshot:

| Mode | Binaries | L1 Regions | Semaphores | CB Data | Registers | DRAM | Use Case |
|------|----------|-----------|------------|---------|-----------|------|----------|
| **ConfigOnly** | Optional | No | No | No | No | No | Workload structure analysis |
| **Selective** | Yes | User-specified CBs | Yes | No | No | No | Narrowing a bug to specific cores/CBs |
| **Full** | Yes | All active CBs | Yes | Yes | Yes | Referenced buffers | Debugging a specific kernel failure |

Mode selection is per-snapshot, not global. This allows the interceptor to start with ConfigOnly during initial workload scan, switch to Selective when a suspicious kernel is identified, and switch to Full for the exact invocation that triggers a failure. This progression is analogous to GDB's approach: `info breakpoints` is cheap, `print *large_array` is expensive, and the developer controls when to pay the cost.

---

## 4. PROPOSED: Object Map Extensions

### 4.1 Current Maps

From `lightmetal_capture.hpp` (lines 80-86), the current maps cover four types: Buffer, Program, Kernel, CBHandle, sharing a single monotonic `next_global_id_` counter.

### 4.2 PROPOSED: Additional Maps for Deep Capture

| New Map | Key Type | Value Type | Purpose |
|---------|----------|------------|---------|
| `snapshot_metadata_map_` | `uint32_t snapshot_id` | `SnapshotMetadata` | Cross-reference between pre/post snapshots |
| `cq_to_global_id_map_` | `const HWCommandQueue*` | `uint32_t` | Formally track command queues (per existing TODO at line 87 of `lightmetal_capture.hpp`) |
| `program_dispatch_ordinal_map_` | `uint32_t program_global_id` | `uint32_t counter` | Track dispatch ordinals per program |

### 4.3 PROPOSED: Extended `LightMetalCaptureContext`

```cpp
// PROPOSED: Extensions to LightMetalCaptureContext
// Added to lightmetal_capture.hpp

class LightMetalCaptureContext {
    // ... existing members (unchanged) ...

    // PROPOSED: Snapshot management
    uint32_t next_snapshot_id_ = 0;
    std::unordered_map<uint32_t, SnapshotMetadata> snapshot_metadata_map_;

    // PROPOSED: Dispatch ordinal tracking (per-program invocation counter)
    std::unordered_map<uint32_t, uint32_t> program_dispatch_ordinal_map_;

    // PROPOSED: Command queue tracking (closing TODO at line 87)
    std::unordered_map<const HWCommandQueue*, uint32_t> cq_to_global_id_map_;

    // PROPOSED: Deep capture state
    bool deep_capture_enabled_ = false;
    CaptureMode current_capture_mode_ = CaptureMode::ConfigOnly;
    std::string capture_filter_pattern_;

public:
    // PROPOSED: Snapshot ID allocation (separate from next_global_id_)
    uint32_t allocate_snapshot_id() { return next_snapshot_id_++; }

    // PROPOSED: Dispatch ordinal tracking
    uint32_t increment_dispatch_ordinal(uint32_t program_global_id) {
        return program_dispatch_ordinal_map_[program_global_id]++;
    }

    // PROPOSED: Deep capture accessors
    void set_deep_capture(bool enabled);
    bool deep_capture_enabled() const { return deep_capture_enabled_; }
    void set_capture_mode(CaptureMode mode);
    CaptureMode get_capture_mode() const { return current_capture_mode_; }
};
```

Note that `next_snapshot_id_` is **separate** from `next_global_id_`. Snapshot IDs create a temporal sequence (snapshot 0 was captured before snapshot 1) that must be preserved during replay. Global IDs, by contrast, identify objects. Keeping them separate avoids polluting the object ID space with snapshot ordinals.

### 4.4 Interaction with Existing Maps

The proposed extensions do not modify any existing map. They only add new maps and counters. The existing `next_global_id_` continues to serve Buffer, Program, Kernel, and CBHandle objects. This means:

- Old capture code continues to work unchanged.
- Old binaries (without snapshot commands) replay normally.
- New binaries with snapshot commands encounter the `default` case in the replay switch (`lightmetal_replay_impl.cpp`, line 359) and throw `TT_THROW` on old replay engines -- the correct failure mode for incompatible binaries.

### 4.5 PROPOSED: Replay-Side Inverse Maps

The replay side (`LightMetalReplayImpl`) needs corresponding inverse maps:

```cpp
// PROPOSED: Additional maps in LightMetalReplayImpl
std::unordered_map<uint32_t, KernelSnapshot> snapshot_map_;   // snapshot_id -> deserialized snapshot
std::unordered_map<uint32_t, HWCommandQueue*> cq_map_;        // global_id -> reconstructed CQ
```

The `KernelSnapshot` type would be a C++ struct mirroring the FlatBuffer `KernelSnapshotCommand` but holding owned data (`std::vector`, `std::shared_ptr`) rather than FlatBuffer references.

---

## 5. PROPOSED: Versioning and `EnvironmentMetadata` Schema

### 5.1 Current State

The versioning gap identified in Section 1, 5.1 and Section 2, Item 10 (no git hash, architecture ID, or system descriptor in the binary) becomes even more critical for kernel-level snapshots, where captured state includes low-level details tightly coupled to specific hardware and firmware versions.

### 5.2 PROPOSED: Extended `LightMetalBinary` Root

```flatbuffers
// PROPOSED: Extended root table (appended fields only -- backward compatible)
table LightMetalBinary {
  // --- Existing fields (unchanged, vtable positions 0-1) ---
  commands: [tt.tt_metal.flatbuffer.Command];
  trace_descriptors: [TraceDescriptorByTraceId];

  // --- PROPOSED versioning fields (positions 2-8) ---
  schema_version: uint32;          // Monotonically increasing schema version
  min_replay_version: uint32;      // Minimum replay engine version that can handle this binary
  git_hash: string;                // Git commit hash of capture-time TT-Metal
  arch: Arch;                      // Target hardware architecture
  capture_timestamp: uint64;       // Unix timestamp of capture (nanoseconds)
  system_desc: SystemDescriptor;   // Hardware configuration at capture time
  file_features: uint32;           // Feature bitmask (see below)
}

file_identifier "LMTL";
root_type LightMetalBinary;

// PROPOSED: Minimal system descriptor for replay compatibility checking.
table SystemDescriptor {
  arch: Arch;
  num_devices: uint8;
  cores_per_device: uint32;        // Total Tensix cores
  l1_size_bytes: uint32;           // L1 memory size per core
  dram_size_bytes: uint64;         // Total DRAM per device
  noc_count: uint8;                // Number of NOC networks (2 for Wormhole/Blackhole)
  num_ethernet_cores: uint16;      // Ethernet core count (varies by arch)
  firmware_version: string;        // Firmware version string
  hal_version: string;             // HAL version identifier
  compiler_version: string;        // RISC-V toolchain version
}
```

### 5.3 Forward Compatibility Analysis

Adding new fields to the end of an existing table is forward compatible in FlatBuffers:

1. **Old replay engine reads new binary**: New fields (`schema_version`, `git_hash`, etc.) have default values (0 for integers, null for strings/tables). The old engine ignores them and proceeds with existing commands.

2. **New replay engine reads old binary**: New fields return their defaults. The replay engine detects `schema_version == 0` and knows it is dealing with a pre-versioning binary.

3. **New command types in union**: An old replay engine encountering a `KernelSnapshotCommand` in the command stream hits the `default` case in the switch and throws. This is the correct behavior: the binary is incompatible.

### 5.4 PROPOSED: Version Number Semantics

| `schema_version` | `min_replay_version` | Meaning |
|-------------------|----------------------|---------|
| 0 | 0 | Pre-versioning binary (current state) |
| 1 | 1 | First versioned binary; adds version fields to root |
| 2 | 1 | Adds `KernelSnapshotCommand` (old engines skip; compatible for non-snapshot commands) |
| 3 | 3 | Breaking change (hypothetical): requires new replay engine |

The distinction between `schema_version` and `min_replay_version` allows non-breaking additions (new optional commands) to increment `schema_version` without requiring engine upgrades, while breaking changes (modified existing command semantics) increment `min_replay_version`.

### 5.5 PROPOSED: Feature Bitmask

An alternative versioning approach uses a `uint32` bitmask (`file_features` field) where each bit indicates a capability. This is more flexible than version numbers for independently-evolving features:

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `HAS_KERNEL_BINARIES` | Binary contains `KernelSnapshotCommand` entries with ELF data |
| 1 | `HAS_L1_SNAPSHOTS` | Binary contains L1 memory snapshots |
| 2 | `HAS_SEMAPHORE_STATE` | Binary contains semaphore state |
| 3 | `HAS_REGISTER_STATE` | Binary contains hardware register state |
| 4 | `HAS_SYSTEM_DESC` | Binary contains system descriptor |
| 5 | `HAS_OUTPUT_SNAPSHOTS` | Binary contains `KernelOutputSnapshotCommand` entries |

Replay implementations check flags before accessing optional fields. Both version numbers and feature flags are included in the proposed schema: version numbers for schema evolution tracking, feature flags for efficient capability detection.

### 5.6 PROPOSED: File Identifier Validation

The `file_identifier "LMTL"` directive causes the FlatBuffer builder to embed a 4-byte sequence at offset 4 in the binary:

```cpp
// PROPOSED: Validation in replay engine
bool LightMetalReplayImpl::validate_binary() {
    if (!flatbuffers::BufferHasIdentifier(binary_.get_data().data(), "LMTL")) {
        log_fatal(LogMetalTrace, "Not a LightMetal binary (missing LMTL identifier)");
        return false;
    }
    return true;
}
```

### 5.7 PROPOSED: Architecture Mismatch Handling

```cpp
// PROPOSED: Environment validation at replay start
bool LightMetalReplayImpl::validate_environment() {
    const auto* desc = fb_binary_->system_desc();
    if (!desc) {
        log_warning(LogMetalTrace, "No system descriptor in binary -- skipping validation");
        return true;  // Backward compatible: old binaries have no descriptor
    }

    auto current_arch = MetalContext::instance().get_cluster().arch();
    if (from_flatbuffer(desc->arch()) != current_arch) {
        log_fatal(LogMetalTrace, "Architecture mismatch: binary={} current={}",
                  EnumNameArch(desc->arch()), current_arch);
        return false;
    }

    // Additional checks (warnings, not fatal)
    if (desc->hal_version() && hal_version() != desc->hal_version()->str()) {
        log_warning(LogMetalTrace, "HAL version mismatch: binary={} current={}",
                    desc->hal_version()->c_str(), hal_version());
    }
    return true;
}
```

| Check | Action on Mismatch |
|-------|-------------------|
| `arch` | Fatal error -- cannot replay across architectures |
| `hal_version` | Warning -- register layouts may differ |
| `firmware_version` | Warning -- dispatch protocol may differ |
| `compiler_version` | Warning if binaries captured; error if recompilation needed |

---

## 6. PROPOSED: Integration with `data_collection.hpp`

### 6.1 Existing Infrastructure

The `data_collection.hpp` integration point (Section 1, 9) already executes at the dispatch boundary and tracks kernel binary sizes via `DISPATCH_DATA_BINARY`. The three recording functions (`RecordDispatchData`, `RecordKernelGroup`, `RecordProgramRun`) are all gated by `MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled()`. The dispatch layer has access to the binary data at the point where `RecordDispatchData` is called; currently only the size is recorded.

### 6.2 PROPOSED: Parallel Capture Hook at Dispatch

The most natural insertion point for deep capture is alongside the existing `RecordDispatchData` call site. The interceptor adds a parallel function that captures the actual data, not just the size:

```cpp
// PROPOSED: Parallel to RecordDispatchData, captures actual content
void InterceptDispatchData(
    uint64_t program_id,
    data_collector_t type,
    const void* data,             // Raw dispatch data (not just size)
    uint32_t transaction_size,
    std::optional<HalProcessorIdentifier> processor) {

    if (!LightMetalCaptureContext::get().deep_capture_enabled()) return;

    switch (type) {
        case DISPATCH_DATA_BINARY:
            // Capture the actual kernel binary bytes
            CaptureKernelBinary(program_id, data, transaction_size, processor);
            break;
        case DISPATCH_DATA_SEMAPHORE:
            // Capture semaphore write values
            CaptureSemaphoreWrite(program_id, data, transaction_size);
            break;
        case DISPATCH_DATA_CB_CONFIG:
            // Capture CB config writes (L1 addresses, allocation)
            CaptureCBConfigWrite(program_id, data, transaction_size);
            break;
        case DISPATCH_DATA_RTARGS:
            // Already captured by LightMetal SetRuntimeArgs commands; cross-validate
            break;
    }
}
```

The key difference from the existing `RecordDispatchData`: the interceptor captures the actual **data** being dispatched, not just the **size**. This requires modifying the call sites to pass the data pointer, which is available at the dispatch level but currently discarded after the size is recorded.

### 6.3 PROPOSED: Binary Extraction via `Kernel::binaries()`

The `Kernel` class exposes `binaries(uint64_t build_key)` at `kernel.hpp`, line 197, returning `const std::vector<const ll_api::memory*>&`. The `ll_api::ElfFile` class (`tt_metal/llrt/tt_elffile.hpp`) provides `GetSegments()` for accessing ELF segments with their contents and load addresses. This is the cleanest binary extraction point:

```cpp
// PROPOSED: Binary capture helper
void CaptureKernelBinaries(const Kernel& kernel, IDevice* device) {
    auto& ctx = LightMetalCaptureContext::get();
    auto& fbb = ctx.get_builder();

    uint64_t build_key = device->build_key();
    const auto& bins = kernel.binaries(build_key);

    std::vector<flatbuffers::Offset<KernelBinary>> binary_offsets;
    for (size_t i = 0; i < bins.size(); ++i) {
        const ll_api::memory* bin = bins[i];
        // Extract binary data from ll_api::memory
        // (actual extraction depends on ll_api::memory's serialization API)
        auto binary_data = /* serialize bin to byte vector */;
        auto data_offset = fbb.CreateVector(binary_data);

        binary_offsets.push_back(CreateKernelBinary(
            fbb,
            kernel.get_kernel_processor_type(i),
            static_cast<uint8_t>(kernel.get_kernel_processor_class()),
            data_offset,
            /* text_size */ 0,  // Extracted from ELF headers
            /* packed_size */ 0,
            /* binary_hash */ xxhash64(binary_data)));
    }
    // Store binary_offsets for later assembly into KernelSnapshotCommand
}
```

This approach captures the binary before it is fragmented into dispatch transactions, providing a complete, self-contained ELF per processor.

### 6.4 PROPOSED: `RecordProgramRun` as Dispatch Ordinal Tracker

The `RecordProgramRun(program_id)` function is called once per `EnqueueProgram`. It is the natural place to increment the dispatch ordinal counter and trigger snapshot capture:

```cpp
// PROPOSED: Extended RecordProgramRun
void RecordProgramRun(uint64_t program_id) {
    if (!MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled())
        return;
    MetalContext::instance().data_collector()->RecordProgramRun(program_id);

    // PROPOSED: Trigger kernel snapshot if deep capture is active
    if (LightMetalCaptureContext::get().deep_capture_enabled()) {
        auto& ctx = LightMetalCaptureContext::get();
        uint32_t program_gid = ctx.get_global_id(/* program by id */);
        uint32_t ordinal = ctx.increment_dispatch_ordinal(program_gid);
        if (ctx.should_capture_dispatch(program_gid, ordinal)) {
            CaptureKernelSnapshotsForProgram(program_id, ordinal);
        }
    }
}
```

### 6.5 PROPOSED: `RecordKernelGroup` for Core-to-Kernel Mapping

The `RecordKernelGroup` function receives the full `KernelGroup` structure and `ProgramImpl` reference. A `KernelGroup` bundles the kernels sharing a core type (Tensix, Ethernet) within a program. This provides the core-to-kernel mapping needed to construct per-core snapshots:

```cpp
// PROPOSED: Capture kernel group metadata for snapshot assembly
void CaptureKernelGroupMetadata(
    ProgramImpl& program,
    HalProgrammableCoreType core_type,
    const KernelGroup& kernel_group) {

    for (auto kernel_id : kernel_group.kernel_ids) {
        auto kernel = program.get_kernel(kernel_id);
        auto build_key = /* device-specific build key */;
        const auto& bins = kernel->binaries(build_key);
        // Map kernel -> cores -> binaries for snapshot assembly
    }
}
```

---

## 7. PROPOSED: Capture and Replay Helpers

### 7.1 PROPOSED: `CaptureKernelSnapshot` Helper

Following the pattern of existing helpers in `host_api_capture_helpers.hpp`:

```cpp
// PROPOSED: New capture helper declaration in host_api_capture_helpers.hpp
void CaptureKernelSnapshot(
    Program& program,
    KernelHandle kernel_id,
    uint32_t dispatch_ordinal,
    IDevice* device,
    CaptureMode mode);
```

### 7.2 PROPOSED: Capture Implementation Sketch

```cpp
// PROPOSED: Implementation in host_api_capture_helpers.cpp
void CaptureKernelSnapshot(
    Program& program,
    KernelHandle kernel_id,
    uint32_t dispatch_ordinal,
    IDevice* device,
    CaptureMode mode) {

    auto& ctx = LightMetalCaptureContext::get();
    auto& fbb = ctx.get_builder();
    auto kernel = program.impl().get_kernel(kernel_id);

    uint32_t snapshot_id = ctx.allocate_snapshot_id();
    uint32_t program_gid = ctx.get_global_id(&program);
    uint32_t kernel_gid = ctx.get_global_id(kernel.get());

    // 1. Capture binaries (always, unless ConfigOnly without binary sub-flag)
    std::vector<flatbuffers::Offset<KernelBinary>> binary_offsets;
    if (mode != CaptureMode::ConfigOnly) {
        CaptureKernelBinaries(*kernel, device);
        // ... populate binary_offsets ...
    }

    // 2. Capture named compile args (closing Gap 7)
    auto named_args = kernel->named_compile_time_args();
    std::vector<flatbuffers::Offset<NamedCompileArgEntry>> named_arg_offsets;
    for (const auto& [key, value] : named_args) {
        named_arg_offsets.push_back(
            CreateNamedCompileArgEntry(fbb, fbb.CreateString(key), value));
    }

    // 3. Capture common runtime args (closing Gap 8)
    auto& common_args = kernel->common_runtime_args();
    auto common_args_offset = fbb.CreateVector(common_args);

    // 4. Per-core: runtime args + L1 + semaphores
    const auto& cores = kernel->logical_cores();
    std::vector<flatbuffers::Offset<CoreL1Snapshot>> l1_offsets;
    std::vector<flatbuffers::Offset<CoreSemaphoreState>> sem_offsets;
    std::vector<flatbuffers::Offset<UInt32Vector>> rt_args_offsets;

    for (const auto& core : cores) {
        // Per-core runtime args
        auto& args = kernel->runtime_args(core);
        rt_args_offsets.push_back(
            CreateUInt32Vector(fbb, fbb.CreateVector(args)));

        // L1 snapshot (Selective or Full mode only)
        if (mode == CaptureMode::Full || mode == CaptureMode::Selective) {
            // Read L1 regions for this core's active CBs
            // device->read_core(data, l1_addr, size, core_coord)
            // ... populate l1_offsets ...
        }

        // Semaphore state (Selective or Full mode)
        if (mode != CaptureMode::ConfigOnly) {
            // Read semaphore values from well-known L1 addresses
            // ... populate sem_offsets ...
        }
    }

    // 5. Build the KernelSnapshotCommand
    auto cmd = CreateKernelSnapshotCommand(
        fbb, snapshot_id, kernel_gid, program_gid, dispatch_ordinal,
        fbb.CreateVector(binary_offsets),
        /* core_spec */ ..., /* arch */ ...,
        fbb.CreateVector(named_arg_offsets),
        /* opt_level */ ..., /* source_code */ 0, /* source_hash */ ...,
        fbb.CreateVector(rt_args_offsets), common_args_offset,
        fbb.CreateVector(l1_offsets), fbb.CreateVector(sem_offsets),
        /* cb_states */ ..., /* dram_input_snapshots */ 0,
        static_cast<uint8_t>(mode), /* timestamp */ ...);

    CaptureCommand(CommandType::KernelSnapshotCommand, cmd.Union());
}
```

### 7.3 PROPOSED: Replay Handler

The replay handler for `KernelSnapshotCommand` operates in two modes:

**Mode 1: Full Replay** -- Restore device state and execute the kernel.

**Mode 2: Analysis Only** -- Deserialize the snapshot for offline inspection without executing on hardware.

```cpp
// PROPOSED: In lightmetal_replay_impl.cpp
void LightMetalReplayImpl::execute(
    const flatbuffer::KernelSnapshotCommand* cmd) {

    // 1. Validate architecture
    if (cmd->arch() != to_flatbuffer(current_arch_)) {
        TT_THROW("Architecture mismatch in kernel snapshot: binary={} device={}",
                 EnumNameArch(cmd->arch()), current_arch_);
    }

    // 2. Restore L1 state on each core
    if (cmd->l1_snapshots()) {
        for (const auto* l1_snap : *cmd->l1_snapshots()) {
            auto core = from_flatbuffer(l1_snap->core_coord());
            for (const auto* region : *l1_snap->regions()) {
                device_->write_core(
                    stl::Span<const uint8_t>(region->data()->data(),
                                              region->data()->size()),
                    region->l1_address(), core);
            }
        }
    }

    // 3. Restore semaphore values
    if (cmd->semaphore_states()) {
        for (const auto* sem_snap : *cmd->semaphore_states()) {
            auto core = from_flatbuffer(sem_snap->core_coord());
            for (size_t i = 0; i < sem_snap->semaphore_values()->size(); i++) {
                uint32_t addr = sem_snap->semaphore_base_addr() + i * sizeof(uint32_t);
                uint32_t val = sem_snap->semaphore_values()->Get(i);
                device_->write_core(stl::Span<const uint8_t>(
                    reinterpret_cast<const uint8_t*>(&val), sizeof(val)),
                    addr, core);
            }
        }
    }

    // 4. Load kernel binaries directly (skip JIT compilation)
    //    PROPOSED: New API path: CreateKernelFromBinary
    for (const auto* bin : *cmd->binaries()) {
        auto elf_span = stl::Span<const uint8_t>(
            bin->binary_data()->data(), bin->binary_data()->size());
        LoadKernelBinary(core_spec, elf_span,
                         bin->processor_type(), bin->processor_class());
    }

    // 5. Set runtime args (per-core and common)
    // ... standard SetRuntimeArgs calls ...

    // 6. Dispatch the kernel and wait for completion
    // ... EnqueueProgram equivalent for single-kernel dispatch ...
}
```

### 7.4 PROPOSED: Binary Loading Without Recompilation

The most significant new capability is loading a pre-compiled ELF into a program without triggering JIT compilation. The `Kernel::set_binaries(uint64_t build_key, std::vector<const ll_api::memory*>&& binaries)` method (`kernel.hpp`, line 198) already provides the setter. The challenge is constructing `ll_api::memory` (or `ll_api::ElfFile`) objects from raw ELF bytes captured in the FlatBuffer.

**Architecture note**: The `ll_api::ElfFile` class (`tt_metal/llrt/tt_elffile.hpp`) parses ELF files into segments with `address`, `lma`, `membytes`, and `contents` fields. The replay path would need to either (a) reconstruct `ElfFile` objects from the captured bytes, or (b) bypass the `ElfFile` layer entirely and write binary data directly to L1 addresses stored in the snapshot. Option (b) is simpler but loses the relocation support that `ElfFile` provides.

---

## 8. PROPOSED: Sparse L1 Capture Strategy

Full L1 readback (1.5 MB per core on Wormhole) is expensive. The interceptor uses a sparse capture strategy that reads only the L1 regions actively used by the target kernel:

1. **CB data regions**: Use `CircularBufferImpl::locally_allocated_address_()` and `total_size()` to read only the occupied portion of each CB. For a typical compute kernel with 4 input CBs of 16 KB each, this is 64 KB per core.

2. **Runtime args region**: Read the runtime args area from the `ProgramConfig` offsets (available from `kernel_config_msg_t` after `finalize_offsets()`).

3. **Semaphore region**: Read the 8-16 semaphore values (32-64 bytes) from their well-known L1 addresses.

4. **Skip firmware and unused regions**: Firmware scratch space, unused L1, and regions belonging to other programs are not captured.

| Capture Strategy | Data Per Core (Wormhole) | Readback Time (est.) |
|------------------|------------------------|---------------------|
| Full L1 dump | 1,464 KB | ~75 ms |
| Active CBs + RT args + semaphores | 50-200 KB | 2-10 ms |
| Semaphores only | 32-64 B | < 0.1 ms |

For the common case of single-core compute debugging (1 core, targeted CBs), readback adds 2-10 ms to dispatch latency -- acceptable for a debugging tool.

Cross-reference: Ch2, Section 3 defines the L1 memory layout and capture timing strategies for each architecture.

---

## 9. PROPOSED: Implementation Ordering

The gap dependency ordering established in Section 2, Item 12 suggests the following phased implementation:

| Phase | What | Commands Added | Estimated Effort | Dependencies |
|-------|------|---------------|------------------|--------------|
| Phase 1 | Versioning + file identifier | None (root table extension only) | 1-2 days | None |
| Phase 2 | Missing config fields (`named_compile_args`, `opt_level`, `noc_mode` for Ethernet) | None (existing table extensions) | 1-2 days | Phase 1 |
| Phase 3 | Kernel binary capture | `KernelSnapshotCommand` (ConfigOnly mode) | 3-5 days | Phase 2, Issue #24955 progress |
| Phase 4 | L1 snapshot + semaphore capture | `L1SnapshotCommand`, `SemaphoreSnapshotCommand`, Selective mode | 5-7 days | Phase 3, NOC readback infrastructure |
| Phase 5 | Register capture + full mode | `RegisterStateCommand`, Full mode in `KernelSnapshotCommand` | 3-5 days | Phase 4 |
| Phase 6 | Post-execution comparison | `KernelOutputSnapshotCommand` | 2-3 days | Phase 5 |
| Phase 7 | Replay engine for snapshots | Replay-side `execute()` handlers + `CreateKernelFromBinary` | 7-10 days | Phase 6 |
| Phase 8 | Test fixture + validation suite | -- | 3-5 days | Phase 7 |

**Total estimated effort: 25-39 person-days** for a single developer familiar with LightMetal and the dispatch path.

Phase 1 is independently valuable even without the interceptor: it protects the existing LightMetal system from silent version incompatibilities. Phase 2 improves existing LightMetal fidelity by closing Gap 7 (Section 2). These two phases can be submitted as standard TT-Metal pull requests without depending on the interceptor architecture.

Phase 3 delivers the kernel binary capture that the existing `CreateKernelCommand` TODO comment (`command.fbs`, line 79: `// Later replace with src, then binary`) already calls for.

---

## 10. PROPOSED: Backward Compatibility and Testing

### 10.1 FlatBuffer Schema Evolution Safety

The proposed changes follow FlatBuffer best practices for non-breaking evolution:

- New fields are added to the **end** of existing tables (backward compatible via vtable indirection)
- New types are added to the **end** of existing unions (backward compatible via uint8 tag encoding)
- No existing fields are modified or removed
- Optional fields default to null/0 (old binaries remain parseable by new code)

### 10.2 Build System Gating

The new capture logic is gated behind a compile-time flag (analogous to `TT_ENABLE_LIGHT_METAL_TRACE`) and a runtime flag:

```cpp
// PROPOSED: Compile-time gate (set via CMake)
#if defined(TT_ENABLE_DEEP_CAPTURE) && (TT_ENABLE_DEEP_CAPTURE == 1)
    // Deep capture code compiled in
#endif

// PROPOSED: Runtime gate
if (parse_env("TT_DEEP_CAPTURE", 0) > 0) {
    LightMetalCaptureContext::get().set_deep_capture(true);
}
```

This ensures zero runtime overhead in non-debug builds, matching the existing `TT_ENABLE_LIGHT_METAL_TRACE` pattern.

### 10.3 PROPOSED: Testing Strategy

| Test Category | Test Name | What It Validates |
|---------------|-----------|-------------------|
| Round-trip | `OldBinaryNewReplay` | Capture with unmodified system, replay with new engine -- identical behavior |
| Forward compat | `NewBinaryOldReplay` | New binary with snapshots, old engine -- clean error on unknown command |
| Schema | `SchemaPrefix` | Old binary schema is a prefix of new schema (via `flatc --schema`) |
| File identifier | `LMTLIdentifier` | Binaries have `"LMTL"` magic; non-LMTL files are rejected |
| Snapshot capture | `SingleComputeKernelSnapshot` | 1 compute kernel, 1 core -- binary capture, L1 snapshot, semaphore capture |
| Snapshot capture | `MultiCoreSnapshot` | 1 compute kernel, 8 cores -- multi-core L1 and semaphore capture |
| Snapshot replay | `SnapshotBinaryRoundTrip` | Capture snapshot, save to file, load, replay, compare output |
| Architecture | `ArchTagMismatch` | Reject replay on wrong architecture |
| Size budget | `CaptureBudgetVerification` | Binary size falls within expected range from capture budget table |

### 10.4 Performance Overhead Estimates

| Operation | Estimated Overhead | Notes |
|-----------|-------------------|-------|
| Binary serialization (once per unique kernel) | 0.1-1 ms | Dominated by memcpy of ELF data |
| L1 readback per core (targeted, 50-200 KB) | 2-10 ms | Via PCIe MMIO or NOC read |
| L1 readback per core (full 1.5 MB) | ~75 ms | Only for Full capture mode |
| Semaphore read per core | < 0.1 ms | 32-64 bytes per core |
| FlatBuffer construction | 0.01-0.1 ms | FlatBufferBuilder serialization |
| **Total: ConfigOnly mode** | **< 1 ms** | Metadata only |
| **Total: Selective mode (1 core)** | **~5-15 ms** | Binaries + targeted L1 + semaphores |
| **Total: Full mode (64 cores)** | **~150-700 ms** | All L1 regions across all cores |

This overhead is acceptable for debugging sessions but not for production workloads. The compile-time and runtime gating mechanisms ensure zero overhead when deep capture is disabled.

### 10.5 PROPOSED: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Issue #24955 rearchitecture invalidates extension design | High | High | Design all extensions as additive; coordinate with LightMetal maintainers |
| L1 readback disturbs device state (cache coherence) | Medium | High | Insert `Finish()` barrier before snapshot; validate with hardware team |
| `FlatBufferBuilder` OOM on large multi-core captures | Medium | Medium | Implement incremental flush or LZ4 compression for L1 data |
| Thread-local `DeepTraceScope::depth` insufficient for async dispatch | Low | Low | Kernel snapshots are synchronous (require `Finish` barrier before capture) |
| Architecture-specific register layouts break portability | High | Low | Include `Arch` tag; each replay backend handles its own register mapping |

---

## 11. Summary: From LightMetal to Kernel-Level Capture

| What Exists | What Must Be Added | Effort | Phase |
|-------------|-------------------|--------|-------|
| `LightMetalBinary` root (2 fields, no versioning) | `schema_version`, `arch`, `git_hash`, `system_desc`, `file_identifier` | Low | 1 |
| `KernelConfig` union (3 types, missing fields) | `named_compile_args`, `opt_level`, `noc_mode` for Ethernet | Low | 2 |
| `CreateKernelCommand` (file path only) | `KernelSnapshotCommand` (binary + L1 + semaphores + args) | High | 3-6 |
| `TraceScope` (depth == 1, API boundary) | `DeepTraceScope` (dispatch boundary, separate counter) | Medium | 3 |
| Object maps (4 types: Buffer, Program, Kernel, CB) | Snapshot map + dispatch ordinal tracking + CQ map | Low | 3 |
| `RecordDispatchData` / `RecordProgramRun` (sizes only) | Parallel hooks that capture actual data | Medium | 3 |
| `CreateKernelCommand` replay (JIT recompile) | `CreateKernelFromBinary` (direct ELF loading) | High | 7 |
| No semaphore/L1 capture | `CoreL1Snapshot`, `CoreSemaphoreState` tables | Medium | 4 |
| `LightMetalCompareCommand` (golden data pattern) | `KernelOutputSnapshotCommand` (post-execution comparison) | Medium | 6 |

LightMetal provides the foundation: the FlatBuffer serialization framework, the object identity system, the sequential replay dispatch loop, and the capture helper pattern. The interceptor extends this foundation with deep state capture at the dispatch boundary. The extensions are designed to be additive, backward compatible, and incrementally deployable -- each phase delivers value independently while building toward the full kernel-level capture capability.

Cross-reference: Chapter 4 will use the `KernelSnapshotCommand` schema and `DeepTraceScope` mechanism defined here as the data format and capture trigger for the runtime interceptor. Ch1, Section 3 (Comparative Analysis) identifies capture-and-replay as the preferred debugging strategy for Tenstorrent hardware; this section provides the concrete data model for that strategy.
