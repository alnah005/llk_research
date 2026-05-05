# LightMetal Architecture and Capture Mechanics

This section provides a comprehensive analysis of TT-Metal's LightMetal capture/replay system -- the closest existing infrastructure to the runtime interceptor proposed in this document. LightMetal records host-side API calls into a FlatBuffer binary for offline replay on real hardware. Understanding its architecture in detail -- its singleton context, its object-to-global-id mapping system, its FlatBuffer command vocabulary, and its replay dispatch loop -- is essential for determining how much of this infrastructure can be reused and where the interceptor must diverge.

All references below cite actual source files from the TT-Metal codebase at `/localdev/salnahari/testing_dir/tt-metal/`.

---

## 1. Purpose and Design Philosophy

LightMetal exists to solve a specific problem: recording a sequence of TT-Metal host API calls so that the exact same workload can be replayed later on physical hardware, without the original host application. The binary it produces is self-contained (modulo kernel source file paths and the device itself) and can be shipped, stored, and replayed by any process with access to a compatible Tenstorrent device.

The design is deliberately **host-centric**: it captures the *calls* the host makes, not the *results* those calls produce on the device. It records "create a buffer of size 4096 in DRAM" rather than the actual DRAM contents. It records "create a kernel from file `reader_matmul.cpp`" rather than the compiled ELF binary that results from JIT compilation. This distinction is fundamental to understanding the gaps identified in Section 2.

The system was built to enable:

1. **Offline replay** -- shipping a LightMetal Binary file to a machine with hardware for execution without the original host code.
2. **Regression testing** -- comparing readback values against golden data embedded in the binary via `LightMetalCompareCommand`.
3. **Workload portability** -- decoupling the host program from the device execution environment.

The implementation lives primarily in four source trees:

| Component | Key Files |
|-----------|-----------|
| Capture context (singleton) | `tt_metal/impl/lightmetal/lightmetal_capture.hpp`, `.cpp` |
| Capture helpers (per-API serializers) | `tt_metal/impl/lightmetal/host_api_capture_helpers.hpp`, `.cpp` |
| FlatBuffer schemas | `tt_metal/impl/flatbuffer/command.fbs`, `light_metal_binary.fbs`, `program_types.fbs`, `buffer_types.fbs`, `base_types.fbs` |
| Replay engine | `tt_metal/impl/lightmetal/lightmetal_replay_impl.hpp`, `.cpp`, `lightmetal_replay.cpp` |

---

## 2. The `LightMetalCaptureContext` Singleton

The capture system is governed by a global singleton defined in `tt_metal/impl/lightmetal/lightmetal_capture.hpp` (lines 38-88).

### 2.1 Class Declaration

```cpp
class LightMetalCaptureContext {
public:
    static LightMetalCaptureContext& get();                        // line 40
    bool is_tracing() const;                                       // line 45
    void set_tracing(bool tracing);                                // line 46
    flatbuffers::FlatBufferBuilder& get_builder();                 // line 48
    std::vector<flatbuffers::Offset<Command>>& get_cmds_vector(); // line 49
    void capture_trace_descriptor(const TraceDescriptor&, uint32_t tid); // line 50
    LightMetalBinary create_light_metal_binary();                  // line 51
    void reset();                                                  // line 52

    // Object map accessors -- 4 overload sets for Buffer, Program, Kernel, CBHandle
    bool is_in_map(const Buffer* obj);
    uint32_t add_to_map(const Buffer* obj);
    void remove_from_map(const Buffer* obj);
    uint32_t get_global_id(const Buffer* obj);
    // ... same pattern for Program*, Kernel*, CBHandle

private:
    LightMetalCaptureContext();  // Private constructor (line 73)

    bool is_tracing_ = false;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<flatbuffers::Offset<Command>> cmds_vec_;
    std::vector<TraceDescriptorByTraceIdOffset> trace_descs_vec_;

    uint32_t next_global_id_ = 0;                                 // line 82
    std::unordered_map<uint64_t, uint32_t> buffer_id_to_global_id_map_;
    std::unordered_map<uint64_t, uint32_t> program_id_to_global_id_map_;
    std::unordered_map<const Kernel*, uint32_t> kernel_to_global_id_map_;
    std::unordered_map<CBHandle, uint32_t> cb_handle_to_global_id_map_;
};
```

Source: `tt_metal/impl/lightmetal/lightmetal_capture.hpp`, lines 38-88.

### 2.2 Key Architectural Properties

| Property | Detail |
|----------|--------|
| **Lifetime** | Process-global singleton; `get()` returns a `static` local instance (Meyer's singleton, lines 24-25 of `lightmetal_capture.cpp`) |
| **Thread safety** | None visible -- no mutex on `builder_`, `cmds_vec_`, or object maps. The `TraceScope::depth` counter is `thread_local` (line 34 of `host_api_capture_helpers.hpp`), making per-thread depth tracking safe, but the shared state is unsynchronized |
| **Builder capacity** | Default `FlatBufferBuilder` starts at 1 KB, grows geometrically (doubles). No pre-allocation observed |
| **Global ID space** | Single monotonic `uint32_t` counter shared across all four object types (line 82). This means Buffer global_id 0, Program global_id 1, Kernel global_id 2, etc. -- IDs are unique across types |
| **Reset semantics** | `reset()` (lines 60-70 of `lightmetal_capture.cpp`) clears builder, all maps, all vectors, resets `next_global_id_` to 0. Asserts tracing is not active |

### 2.3 The Object-to-Global-ID Mapping System

The mapping system is the backbone of LightMetal's ability to serialize object relationships. Every host-side object that a later command might reference is assigned a `uint32_t` global_id at creation time, and subsequent commands reference it by that ID rather than by pointer.

**Four tracked object types:**

| Object Type | Map Key Type | Map Key Source | Source Location |
|-------------|-------------|----------------|-----------------|
| `Buffer*` | `uint64_t` | `obj->unique_id()` | `lightmetal_capture.cpp`, lines 76-102 |
| `Program*` | `uint64_t` | `obj->impl().get_id()` | `lightmetal_capture.cpp`, lines 104-130 |
| `Kernel*` | `const Kernel*` (raw pointer) | The pointer itself | `lightmetal_capture.cpp`, lines 132-156 |
| `CBHandle` | `CBHandle` (`uintptr_t`) | The handle value itself | `lightmetal_capture.cpp`, lines 158-182 |

Critical observations for the interceptor design:

1. **Buffer and Program use semantic IDs** (`unique_id()` and `get_id()` respectively), making them stable across captures. Kernel uses a raw pointer, which is only valid within a single capture session.

2. **The `next_global_id_` counter is shared** across all types. This is a deliberate design choice: any global_id unambiguously identifies both the object type and instance. During replay, the global_id is used to look up the reconstructed object in a type-specific map (see Section 6).

3. **No CommandQueue mapping exists yet.** The TODO at line 87 of `lightmetal_capture.hpp` notes: "consider adding map for HWCommandQueue object." Currently, CQ references are stored by `cq.id()` directly (see `CaptureEnqueueWriteBuffer` at line 211 of `host_api_capture_helpers.cpp`).

4. **The `uint32_t` type for global IDs is acknowledged as potentially insufficient.** The TODO at line 81 of `lightmetal_capture.hpp` states: "upgrade all global_id to be uint64_t for capture + replay."

---

## 3. The `TraceScope` Guard and `LIGHT_METAL_TRACE_FUNCTION_CALL` Macro

The capture system uses a re-entrancy guard to prevent recursive capture. When a captured API call internally invokes other captured API calls, only the outermost call should be recorded.

### 3.1 `TraceScope` Definition

From `tt_metal/impl/lightmetal/host_api_capture_helpers.hpp`, lines 32-38:

```cpp
struct TraceScope {
    // Provide an inline definition in the header
    static inline thread_local int depth = 0;
    // Increment depth on entering scope, decrement on exiting
    TraceScope() { ++depth; }
    ~TraceScope() { --depth; }
};
```

The `depth` counter is `thread_local`, meaning each thread has its own re-entrancy depth tracker. This is important: it means LightMetal capture is inherently per-thread, and multi-threaded workloads would interleave commands in the `cmds_vec_` without synchronization.

### 3.2 The Compile-Time Gate

The entire macro system is conditionally compiled via `TT_ENABLE_LIGHT_METAL_TRACE` (`host_api_capture_helpers.hpp`, line 42):

```cpp
#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
```

When this flag is not defined (or not equal to 1), both macros expand to no-ops (lines 60-64), ensuring zero runtime overhead in non-tracing builds. The flag is set via CMake (`cmake/project_options.cmake`).

### 3.3 The Full Macro

From `host_api_capture_helpers.hpp`, lines 44-58:

```cpp
#define LIGHT_METAL_TRACE_FUNCTION_ENTRY() tt::tt_metal::TraceScope __traceScopeGuard

#define LIGHT_METAL_TRACE_FUNCTION_CALL(capture_func, ...)                                          \
    do {                                                                                            \
        log_trace(                                                                                  \
            tt::LogMetalTrace,                                                                      \
            "LIGHT_METAL_TRACE_FUNCTION_CALL: {} via {} istracing: {} depth: {}",                   \
            #capture_func,                                                                          \
            __FUNCTION__,                                                                           \
            LightMetalCaptureContext::get().is_tracing(),                                           \
            tt::tt_metal::TraceScope::depth);                                                       \
        if (LightMetalCaptureContext::get().is_tracing() && tt::tt_metal::TraceScope::depth == 1) { \
            capture_func(__VA_ARGS__);                                                              \
        }                                                                                           \
    } while (0)
```

The two-part guard works as follows:

1. `LIGHT_METAL_TRACE_FUNCTION_ENTRY()` creates a stack-scoped `TraceScope` object that increments `depth` on entry and decrements on exit.
2. `LIGHT_METAL_TRACE_FUNCTION_CALL(...)` first emits a `log_trace()` diagnostic (showing the capture function name, calling function name, tracing state, and depth), then checks two conditions: (a) tracing is globally enabled via `is_tracing()`, and (b) `depth == 1` (we are at the outermost API boundary). Only when both conditions hold is the capture function invoked.

### 3.4 Instrumented Call Sites in the Codebase

Searching the TT-Metal source for all `LIGHT_METAL_TRACE_FUNCTION_CALL` invocations yields the following capture points:

| Call Site File | API Function | Capture Helper Called |
|---|---|---|
| `tt_metal.cpp:1270` | `CreateKernel()` | `CaptureCreateKernel` |
| `tt_metal.cpp:1324` | `CreateCircularBuffer()` | `CaptureCreateCircularBuffer` |
| `tt_metal.cpp:1469` | `SetRuntimeArgs()` (uint32 span) | `CaptureSetRuntimeArgsUint32` |
| `tt_metal.cpp:1479` | `SetRuntimeArgs()` (uint32 span, overload) | `CaptureSetRuntimeArgsUint32` |
| `tt_metal.cpp:1491` | `SetRuntimeArgs()` (per-core vec) | `CaptureSetRuntimeArgsUint32VecPerCore` |
| `program.cpp:224` | `Program::Program()` constructor | `CaptureProgramConstructor` |
| `program.cpp:229` | `CreateProgram()` factory | `CaptureProgramConstructor` |
| `buffer.cpp:326` | `Buffer::create()` (without address) | `CaptureBufferCreate` |
| `buffer.cpp:365` | `Buffer::create()` (with address) | `CaptureBufferCreate` |
| `buffer.cpp:471` | `DeallocateBuffer()` | `CaptureBufferDeallocate` |
| `buffer.cpp:479` | `Buffer` destructor | `CaptureBufferDelete` |

**Notable absences**: `EnqueueProgram()` is not captured. The replay implementation has a forward declaration for `EnqueueProgramCommand` (`lightmetal_replay_impl.hpp`, line 37) and an `execute()` overload (line 80), but no corresponding `CaptureEnqueueProgram` function exists in `host_api_capture_helpers.hpp`, and no `EnqueueProgramCommand` appears in the `CommandType` union in `command.fbs`. The capture functions for `EnqueueWriteBuffer`, `EnqueueReadBuffer`, and `Finish` exist in `host_api_capture_helpers.hpp` (lines 93-105) but their API-side macro call sites may be in the command-queue path rather than the public API path, suggesting partial rearchitecture per Issue #24955.

### 3.5 Implications for the Interceptor

The `depth == 1` check means LightMetal only captures at the API boundary. For the interceptor's kernel-level capture, we need to capture at a *different* depth -- specifically at the point of `EnqueueProgram` dispatch, which is itself an API call. This means the interceptor either needs its own independent depth tracking (a `DeepTraceScope` with a separate `thread_local` counter) or needs to hook at a layer that is not guarded by the existing `TraceScope`. The `DeepTraceScope` approach (separate counter, separate macro) cleanly separates the two capture modes and is the recommended design, as analyzed in Section 3.

---

## 4. The FlatBuffer Command Vocabulary -- Complete Inventory

The FlatBuffer schema in `tt_metal/impl/flatbuffer/command.fbs` defines the full command vocabulary. This is the exhaustive list of every command type in the `CommandType` union (lines 118-136):

### 4.1 Complete Command Type Enumeration

| # | Command Type | FBS Lines | Purpose | Replay Status |
|---|-------------|-----------|---------|---------------|
| 1 | `ReplayTraceCommand` | 8-13 | Replays a previously captured Metal trace by `tid` on a command queue | Deprecated; throws (Issue #24955) |
| 2 | `EnqueueTraceCommand` | 15-20 | Enqueues a Metal trace for execution | Deprecated; throws (Issue #24955) |
| 3 | `LoadTraceCommand` | 22-25 | Loads trace data into device memory by `tid` | Deprecated; throws (Issue #24955) |
| 4 | `ReleaseTraceCommand` | 27-30 | Releases a trace and its device-side resources | Deprecated; throws (Issue #24955) |
| 5 | `BufferCreateCommand` | 32-44 | Allocates a device buffer with full configuration | Active |
| 6 | `BufferDeallocateCommand` | 46-48 | Deallocates a buffer's device memory without destroying the object | Active |
| 7 | `BufferDeleteCommand` | 50-52 | Destroys a buffer object entirely | Active |
| 8 | `EnqueueWriteBufferCommand` | 54-59 | Writes host data into a device buffer; **includes inline data payload** | Replay disabled (Issue #24955) |
| 9 | `EnqueueReadBufferCommand` | 61-65 | Reads device buffer back to host; no data payload stored | Replay disabled (Issue #24955) |
| 10 | `FinishCommand` | 67-70 | Blocks until all commands on a queue complete | Replay disabled (Issue #24955) |
| 11 | `ProgramConstructorCommand` | 72-74 | Creates a new empty Program object | Active |
| 12 | `CreateKernelCommand` | 76-82 | Creates a kernel from a **file path** with core spec and config | Active |
| 13 | `SetRuntimeArgsUint32Command` | 84-89 | Sets uint32 runtime args for a kernel on a core spec | Active |
| 14 | `SetRuntimeArgsUint32VecPerCoreCommand` | 91-96 | Sets per-core uint32 runtime args (vector of vectors) | Active |
| 15 | `SetRuntimeArgsCommand` | 98-102 | Sets runtime args that may include Buffer references | Active (no `execute()` handler) |
| 16 | `CreateCircularBufferCommand` | 104-109 | Creates a circular buffer with full CB configuration | Active |
| 17 | `LightMetalCompareCommand` | 111-116 | Verification: compares readback data against golden values | Active |

Commands 1-4 are trace-related and have been deprecated per Issue #24955. Their `execute()` handlers in `lightmetal_replay_impl.cpp` (lines 365-402) now throw `TT_THROW("Light Metal Trace is no longer supported.")`. Commands 8-10 have capture functions defined but their replay handlers are commented out pending rearchitecture. This leaves **13 active command types** in the schema, of which **10 have fully functional replay handlers**.

### 4.2 The `Command` Wrapper Table

Every command is wrapped in a `Command` table (lines 138-140):

```flatbuffers
table Command {
    cmd: CommandType;
}
```

The `CommandType` union discriminates the payload. FlatBuffers encodes union type tags as `uint8`, limiting a single union to **256 members**. With 17 used, 239 slots remain -- ample room for the interceptor's extensions but worth tracking as the system grows.

### 4.3 Deep Analysis of Key Command Types

#### 4.3.1 `CreateKernelCommand` -- Critical for Gap Analysis

```flatbuffers
table CreateKernelCommand {
  global_id: uint32;
  program_global_id: uint32;
  file_name: string;          // Later replace with src, then binary
  core_spec: CoreSpec;
  kernel_config: KernelConfig;
}
```

The comment on line 79 is revealing: `// Later replace with src, then binary`. This confirms that the LightMetal designers recognized the limitation of storing a file path and planned a progression: first store source code, then store compiled binary. Neither has been implemented.

The `KernelConfig` union (defined in `program_types.fbs`, lines 53-57) carries the configuration variant:

```flatbuffers
union KernelConfig {
  DataMovementConfig,
  ComputeConfig,
  EthernetConfig
}
```

**FBS coverage gaps** (for full C++ field inventories, see Ch2, Section 1, Item 3):

All three config types in the FlatBuffer schema capture their core functional fields (processor, NOC, compile_args, defines, math fidelity, etc.) but share the same two omissions: `named_compile_args` and `opt_level`. Additionally, `EthernetConfig` omits `noc_mode` (present in the C++ struct at `kernel.hpp`, line 54), and the `KernelConfig` union entirely lacks `QuasarDataMovementConfig` and `QuasarComputeConfig` variants.

| Config Type | FBS Location | Fields Captured | Fields Missing |
|-------------|-------------|----------------|----------------|
| `DataMovementConfig` | `program_types.fbs`, lines 25-31 | processor, noc, noc_mode, compile_args, defines | named_compile_args, opt_level |
| `ComputeConfig` | `program_types.fbs`, lines 33-42 | math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, bfp8_pack_precise, math_approx_mode, compile_args, defines | named_compile_args, opt_level |
| `EthernetConfig` | `program_types.fbs`, lines 44-50 | eth_mode, noc, processor, compile_args, defines | named_compile_args, opt_level, noc_mode |
| `QuasarDataMovementConfig` | -- | -- | **Entirely absent from FBS** |
| `QuasarComputeConfig` | -- | -- | **Entirely absent from FBS** |

The missing `named_compile_args` (`std::unordered_map<std::string, uint32_t>`) are baked into generated defines just like positional `compile_args`, so their omission means the FlatBuffer schema cannot fully reconstruct the compilation environment. The missing `opt_level` (`KernelBuildOptLevel` enum) means replay uses the default (which differs by kernel type: DM=`O2`, Compute=`O3`, Ethernet=`Os`). The missing `hlk_desc` (Ch2, Section 1, Item 6) affects how the generated `chlkc_unpack.cpp`/`chlkc_math.cpp`/`chlkc_pack.cpp` are produced during JIT build.

#### 4.3.2 `EnqueueWriteBufferCommand` -- The Data-Bearing Command

```flatbuffers
table EnqueueWriteBufferCommand {
  cq_global_id: uint32;
  buffer_global_id: uint32;
  src: [uint32];              // Actual data payload
  blocking: bool;
}
```

This is the **only command that stores actual tensor data inline** in the FlatBuffer. The `src` field is a vector of `uint32` values representing the data to be written. The capture helper `CaptureEnqueueWriteBuffer` (lines 199-239 of `host_api_capture_helpers.cpp`) handles multiple `HostDataType` variants (`uint32`, `uint16`, `void*`) by converting them all to `uint32` vectors. Note that `bfloat16`, `float`, `int32_t`, and `uint8_t` data types are **not handled** and would throw. The comment at line 218 acknowledges this: "Currently support limited data formats."

**Size impact**: For a 64 KB tensor, this single command adds 64 KB to the binary. For a typical matmul workload with two input tensors of 1 MB each, the write commands alone contribute approximately 2 MB.

#### 4.3.3 `CreateCircularBufferCommand`

```flatbuffers
table CreateCircularBufferCommand {
  global_id: uint32;
  program_global_id: uint32;
  core_spec: CoreSpec;
  config: CircularBufferConfig;
}
```

The `CircularBufferConfig` table (defined in `buffer_types.fbs`, lines 68-81) captures CB *configuration* comprehensively:

| Field | FBS Type | Purpose |
|-------|----------|---------|
| `total_size` | `uint32` | Total CB size in bytes |
| `globally_allocated_address` | `Uint32Optional` | Pre-allocated L1 address |
| `data_formats` | `[CBConfigDataFormat]` | Per-index data format |
| `page_sizes` | `[CBConfigPageSize]` | Per-index page size |
| `tiles` | `[CBConfigTile]` | Per-index tile shape |
| `shadow_buf_global_id` | `Uint32Optional` | Reference to a shadow buffer |
| `buffer_indices` | `[uint8]` | CB buffer index set |
| `local_buffer_indices` | `[uint8]` | Local buffer index subset |
| `remote_buffer_indices` | `[uint8]` | Remote buffer index subset |
| `dynamic_cb` | `bool` | Whether this is a dynamically-sized CB |
| `max_size` | `uint32` | Maximum size for dynamic CBs |
| `buffer_size` | `uint32` | Current buffer size |

This captures CB *configuration* comprehensively but captures **zero CB data contents**. The interceptor needs both.

#### 4.3.4 Runtime Args Commands -- Three Variants

1. **`SetRuntimeArgsUint32Command`** (lines 84-89): Sets a flat `[uint32]` array for a single core spec.
2. **`SetRuntimeArgsUint32VecPerCoreCommand`** (lines 91-96): Sets a vector of vectors via `[UInt32Vector]`.
3. **`SetRuntimeArgsCommand`** (lines 98-102): Uses `[RuntimeArg]` where each element contains a `RuntimeArgValue` union that can be either `UInt32Value` or `BufferGlobalId`, allowing runtime args that reference buffers.

**Missing from all three variants**: there is no `SetCommonRuntimeArgs` capture. Common runtime args (`Kernel::common_runtime_args_` in `kernel.hpp`) are shared across all cores and are distinct from per-core args. No `CaptureSetCommonRuntimeArgs` helper exists in `host_api_capture_helpers.hpp`.

---

## 5. The LightMetal Binary Format

### 5.1 Root Schema

From `tt_metal/impl/flatbuffer/light_metal_binary.fbs`:

```flatbuffers
table LightMetalBinary {
  // TODO (kmabee) - Git Hash, Versioning, SystemDesc, etc.
  commands: [tt.tt_metal.flatbuffer.Command];
  trace_descriptors: [TraceDescriptorByTraceId];
}
root_type LightMetalBinary;
```

The binary has exactly two top-level fields:

1. **`commands`**: An ordered vector of `Command` tables. Commands are replayed sequentially in the order they were captured.
2. **`trace_descriptors`**: A sorted table of `TraceDescriptorByTraceId` entries, keyed by `trace_id` for $O(\log n)$ lookup via `LookupByKey()`. Each descriptor contains `trace_data: [uint32]` (the raw trace command stream) and sub-device descriptor mappings.

The TODO at line 33 explicitly flags the absence of: git hash, versioning, and system descriptor (`SystemDesc`) information. This means the current binary format has no way to verify that the replay environment matches the capture environment -- a gap that becomes critical for kernel-level snapshots where compiled binaries and L1 memory layouts are architecture-dependent.

### 5.2 Binary Construction

The `create_light_metal_binary()` method (`lightmetal_capture.cpp`, lines 45-57) finalizes the builder:

```cpp
LightMetalBinary LightMetalCaptureContext::create_light_metal_binary() {
    auto cmds_vec_fb = builder_.CreateVector(cmds_vec_);
    auto sorted_trace_descs = builder_.CreateVectorOfSortedTables(&trace_descs_vec_);
    auto light_metal_binary = CreateLightMetalBinary(builder_, cmds_vec_fb, sorted_trace_descs);
    builder_.Finish(light_metal_binary);
    const uint8_t* buffer_ptr = builder_.GetBufferPointer();
    size_t buffer_size = builder_.GetSize();
    std::vector<uint8_t> binary_data(buffer_ptr, buffer_ptr + buffer_size);
    return LightMetalBinary(std::move(binary_data));
}
```

The use of `CreateVectorOfSortedTables` for trace descriptors enables binary-search lookup by `trace_id` during replay -- a FlatBuffer pattern that uses the `(key)` attribute and provides a template for future kernel snapshot indexing.

### 5.3 Size Characteristics for Representative Workloads

| Workload | Dominant Component | Estimated Total |
|----------|-------------------|-----------------|
| Single matmul (1024x1024 bf16, 2 input buffers) | `EnqueueWriteBufferCommand` data (~4 MB) | ~4.5 MB |
| Multi-op pipeline (5 ops, 1024x1024) | Buffer write data (~24 MB) | ~24.5 MB |
| Single LLK unit test (32x32 tile) | Command metadata (~1 KB) | ~5 KB |

The binary is dominated by data payloads in `EnqueueWriteBufferCommand`. For the interceptor's kernel-level capture, compiled ELF binaries (typically 4-32 KB per RISC-V processor, times 3-5 processors per kernel) and L1 memory snapshots (up to 1.5 MB per core on Wormhole) would add substantial additional size -- see the capture budget analysis in Section 3.

---

## 6. Replay Flow

### 6.1 Entry Point

The replay flow begins in `LightMetalReplayImpl::run()` (`lightmetal_replay_impl.cpp`, lines 691-742):

```cpp
bool LightMetalReplayImpl::run() {
    if (!fb_binary_) { return false; }
    const bool replay_manages_device = device_ == nullptr;
    // ...
    if (replay_manages_device) { setup_devices(); }
    for (const auto* cmd : *commands) {
        execute(cmd);
    }
    clear_object_maps();
    if (replay_manages_device) { close_devices(); }
    return true;
}
```

The loop is deliberately simple: iterate commands in order, execute each one. No parallelism, no dependency analysis, no out-of-order execution.

### 6.2 Replay Object Maps

The replay side maintains inverse maps that map `global_id -> reconstructed object`:

| Map | Type | Source |
|-----|------|--------|
| `buffer_map_` | `unordered_map<uint32_t, shared_ptr<Buffer>>` | `lightmetal_replay_impl.hpp`, line 138 |
| `program_map_` | `unordered_map<uint32_t, shared_ptr<Program>>` | line 139 |
| `kernel_handle_map_` | `unordered_map<uint32_t, KernelHandle>` | line 140 |
| `kernel_map_` | `unordered_map<uint32_t, shared_ptr<Kernel>>` | line 141 |
| `cb_handle_map_` | `unordered_map<uint32_t, CBHandle>` | line 142 |

Note that the replay side maintains **two** maps for kernels: `kernel_handle_map_` (for `KernelHandle`, used by `SetRuntimeArgs`) and `kernel_map_` (for `shared_ptr<Kernel>`, used internally). This dualism reflects the API's split between handle and object access patterns.

### 6.3 Replay of `CreateKernelCommand`

The `CreateKernelCommand` replay handler (`lightmetal_replay_impl.cpp`, lines 532-555) demonstrates the recompilation dependency:

```cpp
void LightMetalReplayImpl::execute(const flatbuffer::CreateKernelCommand* cmd) {
    auto program = get_program_from_map(cmd->program_global_id());
    auto core_spec = core_spec_from_flatbuffer(cmd);
    auto kernel_config = kernel_config_from_flatbuffer(cmd);
    KernelHandle kernel_id = std::visit(
        [&](const auto& cfg) -> KernelHandle {
            return CreateKernel(*program, cmd->file_name()->c_str(), core_spec, cfg);
        }, kernel_config);
    add_kernel_handle_to_map(cmd->global_id(), kernel_id);
    std::shared_ptr<Kernel> kernel = program->impl().get_kernel(kernel_id);
    add_kernel_to_map(cmd->global_id(), kernel);
}
```

This calls the actual `CreateKernel()` API with `cmd->file_name()->c_str()` -- the original source file path -- triggering a full JIT compilation. The replaying machine must have the same source tree available and produce compatible binaries. The kernel is not replayed from a captured binary.

### 6.4 Currently Disabled Replay Paths

Multiple replay `execute()` methods are disabled, as noted in comments referencing Issue #24955:

- `execute(EnqueueWriteBufferCommand*)` (line 485): `EnqueueWriteBuffer` call commented out
- `execute(EnqueueReadBufferCommand*)` (line 508): `EnqueueReadBuffer` call commented out
- `execute(FinishCommand*)` (line 524): `Finish` call commented out
- All trace commands (lines 365-402): throw `TT_THROW("Light Metal Trace is no longer supported.")`

The data *is* present in the binary for write commands (`cmd->src()->data()` would return the serialized payload), but the actual device operations are not executed. This means the current LightMetal replay system captures commands but cannot fully replay most data-transfer operations. The rearchitecture work in Issue #24955 is a prerequisite for a functioning end-to-end replay pipeline.

---

## 7. The Capture Helper Layer

### 7.1 `CaptureCommand` -- The Universal Wrapper

All capture helpers funnel through a private helper (`host_api_capture_helpers.cpp`, lines 61-64):

```cpp
void CaptureCommand(tt::tt_metal::flatbuffer::CommandType cmd_type,
                    ::flatbuffers::Offset<void> fb_offset) {
    auto& ctx = LightMetalCaptureContext::get();
    ctx.get_cmds_vector().push_back(
        tt::tt_metal::flatbuffer::CreateCommand(ctx.get_builder(), cmd_type, fb_offset));
}
```

### 7.2 Complete Capture Helper Inventory

| Helper Function | Declared Line | Produces | Notes |
|----------------|---------------|----------|-------|
| `CaptureReplayTrace` | 71 | `ReplayTraceCommand` | Deprecated |
| `CaptureEnqueueTrace` | 73 | `EnqueueTraceCommand` | Deprecated |
| `CaptureLoadTrace` | 75 | `LoadTraceCommand` | Deprecated |
| `CaptureReleaseTrace` | 77 | `ReleaseTraceCommand` | Deprecated |
| `CaptureBufferCreate` | 79 | `BufferCreateCommand` | Handles both `create()` overloads |
| `CaptureBufferDeallocate` | 90 | `BufferDeallocateCommand` | Skips if buffer not in map |
| `CaptureBufferDelete` | 91 | `BufferDeleteCommand` | Same workaround as Deallocate |
| `CaptureEnqueueWriteBuffer` | 93 | `EnqueueWriteBufferCommand` | Converts `HostDataType` to `[uint32]` |
| `CaptureEnqueueReadBuffer` | 99 | `EnqueueReadBufferCommand` | Does NOT capture readback data |
| `CaptureFinish` | 105 | `FinishCommand` | Includes sub-device ID conversion |
| `CaptureProgramConstructor` | 106 | `ProgramConstructorCommand` | Registers program in map |
| `CaptureCreateKernel` | 108 | `CreateKernelCommand` | Stores file_name string, not binary |
| `CaptureSetRuntimeArgsUint32` | 115 | `SetRuntimeArgsUint32Command` | Flat uint32 span |
| `CaptureSetRuntimeArgsUint32VecPerCore` | 121 | `SetRuntimeArgsUint32VecPerCoreCommand` | Vector of vectors |
| `CaptureCreateCircularBuffer` | 127 | `CreateCircularBufferCommand` | Full CB config capture |
| `CaptureLightMetalCompare` | 133 | `LightMetalCompareCommand` | Golden data inline |

All declarations in `host_api_capture_helpers.hpp` (lines 71-138) and implementations in `host_api_capture_helpers.cpp`.

**Observation**: `CaptureEnqueueWriteBuffer`, `CaptureEnqueueReadBuffer`, and `CaptureFinish` are defined but their invocation from the host API call sites is not fully wired -- the capture functions exist but the corresponding `LIGHT_METAL_TRACE_FUNCTION_CALL` macro invocations at the API boundary are part of the ongoing Issue #24955 rearchitecture.

---

## 8. Supporting Schema Files

### 8.1 `base_types.fbs` -- Enumerations and Primitives

`tt_metal/impl/flatbuffer/base_types.fbs` defines the foundational enum types:

| Enum | Values | Lines |
|------|--------|-------|
| `Arch` | `Grayskull=0`, `Wormhole_b0=1`, `Blackhole=2` | 4-8 |
| `DataMovementProcessor` | `RISCV_0` through `RISCV_7` | 10-19 |
| `NOC` | `NOC_0`, `NOC_1` | 21-24 |
| `NOC_MODE` | `DM_DEDICATED_NOC`, `DM_DYNAMIC_NOC` | 26-29 |
| `EthMode` | `SENDER=0`, `RECEIVER=1`, `IDLE=2` | 31-35 |
| `MathFidelity` | `LoFi=0`, `HiFi2=2`, `HiFi3=3`, `HiFi4=4`, `Invalid=255` | 37-43 |
| `DataFormat` | 21 variants from `Float32` to `Invalid` | 45-67 |
| `UnpackToDestMode` | `Default`, `UnpackToDestFp32` | 69-72 |

Notable: the `Arch` enum does **not** include `Quasar`, despite the `Kernel` class in `kernel.hpp` (lines 114-119) already supporting `QuasarDataMovementConfig` and `QuasarComputeConfig` variants in its `Config` typedef. The `KernelConfig` FlatBuffer union also lacks Quasar variants, making Quasar kernels entirely unrepresentable in the current schema.

### 8.2 `buffer_types.fbs` and `program_types.fbs`

`buffer_types.fbs` provides buffer type enums, shard specifications, `BufferDistributionSpec`, and the `CircularBufferConfig` table (11 fields). `program_types.fbs` provides core coordinate types, the `CoreSpec` union (`CoreCoord | CoreRange | CoreRangeSet`), kernel config types, the `KernelConfig` union, runtime arg types with the `RuntimeArgValue` union, and the `UInt32Vector` wrapper table for nested vectors.

---

## 9. The `data_collection.hpp` Integration Point

Separate from LightMetal, `tt_metal/impl/dispatch/data_collection.hpp` provides dispatch-level instrumentation:

```cpp
enum data_collector_t {
    DISPATCH_DATA_CB_CONFIG,
    DISPATCH_DATA_SEMAPHORE,
    DISPATCH_DATA_RTARGS,
    DISPATCH_DATA_BINARY,
};
```

Three recording functions are called during the dispatch path:

| Function | Purpose | Called When |
|----------|---------|------------|
| `RecordDispatchData(program_id, type, size, processor)` | Records per-program dispatch write statistics | Per-transaction during dispatch |
| `RecordKernelGroup(program, core_type, kernel_group)` | Records kernel group structure | Per program creation |
| `RecordProgramRun(program_id)` | Counts program executions | Per enqueue |

All three are gated by `MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled()`. The `DISPATCH_DATA_BINARY` category already tracks kernel binary sizes per processor -- exactly the kind of metadata the interceptor needs but that LightMetal does not capture. This system runs at the dispatch boundary (deeper than the public API boundary where LightMetal hooks), making it a natural integration point for the interceptor's deep capture hooks. See Section 3 for the proposed integration strategy.

---

## 10. The `LightMetalCompare` Verification Pattern

The `LightMetalCompareCommand` / `LightMetalCompareToCapture` / `LightMetalCompareToGolden` system demonstrates an important pattern: LightMetal can embed expected-output data in the binary and validate it at replay time. During capture, `LightMetalCompareToCapture()` reads back a buffer's contents and stores them as `golden_data` in a `LightMetalCompareCommand`. During replay, the same buffer is read back and compared element-wise against the stored golden data (lines 639-688 of `lightmetal_replay_impl.cpp`).

This pattern is conceptually adjacent to embedding snapshot data for replay. The capture-compare-verify cycle (`capture golden -> serialize -> deserialize -> replay -> compare`) provides an exact template for kernel snapshot verification.

**Note**: The capture-time utility functions (`LightMetalCompareToCapture`, `LightMetalCompareToGolden`) are currently disabled pending Issue #24955, but the schema and replay-side comparison logic remain defined.

---

## 11. Test Infrastructure

### C++ Tests

The C++ test file (`tests/tt_metal/tt_metal/lightmetal/test_lightmetal.cpp`) is entirely commented out, gated on Issue #24955. Before being disabled, it used a `SingleDeviceLightMetalFixture` GTest fixture that called `LightMetalBeginCapture()` in `SetUp`, captured a workload, called `LightMetalEndCapture()` in `TearDown`, then replayed via `RunLightMetalBinary()`.

### Python Tests

The Python tests (`tests/ttnn/unit_tests/light_metal/test_single_device_light_metal_trace.py`) are active and represent the current LightMetal test surface. They capture single-op and chain-op workloads using `ttnn.light_metal_begin_capture()` / `ttnn.light_metal_end_capture()` but do not include replay verification (noted as a TODO).

### Standalone Runner

The `lightmetal_runner` tool (`tt_metal/tools/lightmetal_runner/lightmetal_runner.cpp`) loads a binary from a file path, constructs a `LightMetalReplay` with no device (replay manages its own), calls `run()`, and returns 0 (pass) or 1 (fail). This demonstrates that LightMetal binaries are designed to be self-contained and executable without the original build environment -- a property that kernel snapshot binaries must also have.

---

## 12. Summary

LightMetal captures: buffer allocation, host-to-device data (partial -- uint32/uint16/void* only), kernel source file paths (not binaries), kernel core placement, kernel config (missing `named_compile_args`, `opt_level`, `hlk_desc`), runtime args (3 variants, missing common args), CB configuration (no CB data), and command ordering.

LightMetal does **not** capture: compiled binaries, L1/CB memory contents, semaphore runtime state, NOC transaction logs, hardware register state, device/architecture metadata, launch/go messages, or single-kernel dispatch events.

For the quantitative coverage assessment (33.8% weighted score), see Section 2, Item 11.
