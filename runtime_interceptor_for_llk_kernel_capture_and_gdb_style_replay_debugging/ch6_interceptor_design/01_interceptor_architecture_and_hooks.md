# Interceptor Architecture and Hooks

This file presents the **PROPOSED** architecture for a runtime interceptor that captures all state required to replay a single kernel invocation outside TT-Metal. It builds on the state inventory (Ch 2), the LightMetal capture foundation (Ch 3, File 1), the dispatch path analysis (Ch 4, File 1), the existing debug/hardware infrastructure (Ch 5), and is grounded in the TT-Metal codebase at commit `621b949`.

---

## 1. Design Goals and User Story

> *"As a kernel developer, I hit a matmul assert on core (3,4) during a 64-core workload. I want to capture the exact state of that kernel invocation -- binaries, runtime args, circular buffer contents, tile data -- and replay it locally under GDB so I can single-step through the LLK math function and inspect registers."*

This user story drives four design goals:

1. **Self-contained snapshots** -- a `.ttksnap` file must contain everything needed to replay a kernel on a host emulator or single-core hardware target without any TT-Metal runtime dependency (Ch 2, File 1).
2. **Minimal dispatch-path perturbation** -- the interceptor must not alter the command sequence or timing of normal execution beyond what is necessary for serialization.
3. **Single-core isolation** -- capture can target a specific `(program_id, core_coord)` pair rather than always snapshotting the entire program.
4. **Integration with existing infrastructure** -- reuse the existing Inspector compilation hooks, Watcher L1 read mechanisms, and data_collection infrastructure.

The end-to-end workflow:

```
1. Developer sets: TT_METAL_KERNEL_CAPTURE=on-assert
2. Runs workload.  Matmul asserts on core (3,4).
3. Interceptor automatically captures state -> matmul_core3_4_00042.ttksnap
4. Developer inspects:  tt-kernel-capture inspect matmul_core3_4_00042.ttksnap
5. Developer replays:   tt-kernel-capture replay matmul_core3_4_00042.ttksnap --gdb
6. GDB opens, stopped at kernel entry.  Developer debugs.
```

---

## 2. Dispatch Pipeline and Hook Point Selection

### 2.1 Source-Level Call Chain

The path from host API to hardware proceeds through the following verified call chain (Ch 4, File 1):

```
EnqueueProgram (host API)
  |
  v
ProgramImpl::generate_dispatch_commands()
    // tt_metal/impl/program/program.cpp:1356
    // Checks cached_program_command_sequences_ for the command_hash.
    // On cache miss:
    |
    +-> program_dispatch::insert_empty_program_dispatch_preamble_cmd()
    |   // tt_metal/impl/program/dispatch.hpp:109
    |
    +-> program_dispatch::insert_stall_cmds()
    |   // tt_metal/impl/program/dispatch.hpp:111
    |
    +-> program_dispatch::assemble_device_commands()    <-- PRIMARY HOOK (Site B)
    |   // tt_metal/impl/program/dispatch.cpp:1890
    |
    v  (cached path)
program_dispatch::update_program_dispatch_commands()    <-- SECONDARY HOOK (H4-update)
    // tt_metal/impl/program/dispatch.cpp:2127
    |
    v
program_dispatch::write_program_command_sequence()
    // tt_metal/impl/program/dispatch.cpp:2481
    // Writes into SystemMemoryManager issue queue.
```

### 2.2 Component Diagram

```
                  User Code
                      |
                      v
           +--------------------+
           |  CreateKernel /    |
           |  SetRuntimeArgs    |  <-- LightMetal capture (existing, Ch 3)
           +--------------------+
                      |
                      v
           +--------------------+
           |  ProgramImpl::     |
           |  compile()         |  <-- AUXILIARY HOOK A: Inspector callback
           +--------------------+       (inspector.hpp line 39)
                      |                 captures ELF binaries via Kernel::binaries()
                      v
           +--------------------+
           |  ProgramImpl::     |
           |  finalize_offsets()|  <-- All L1 offsets resolved (program.cpp:1702)
           +--------------------+       rta, sem, cb, kernel_text
                      |
                      v
           +--------------------+
           |  assemble_device_  |
           |  commands()        |  <-- PRIMARY HOOK (Site B): state fully resolved,
           +--------------------+       still in structured C++ form
                      |                 dispatch.cpp:1890
                      v
           +--------------------+
           |  update_program_   |
           |  dispatch_commands |  <-- SECONDARY HOOK (H4-update): delta capture
           +--------------------+       dispatch.cpp:2127
                      |
                      v
           +--------------------+
           |  write_program_    |
           |  command_sequence  |  <-- Too late: bytes committed to prefetcher
           +--------------------+
```

### 2.3 Why Site B (`assemble_device_commands`) is the Primary Hook

The function `program_dispatch::assemble_device_commands()` at `tt_metal/impl/program/dispatch.cpp` (line 1890) is the point where all structured state converges. At this location:

1. **`finalize_offsets()` has completed** (called from `ProgramImpl::finalize_offsets()` at `tt_metal/impl/program/program.cpp` line 1702). All L1 memory offsets are resolved: `rta_offset`, `sem_offset`, `cb_offset`, `kernel_text_offset`, and their sizes are finalized in `ProgramConfig` (defined in `tt_metal/impl/program/program_impl.hpp` lines 97-108).

2. **Binaries are available in structured form.** The `Kernel::binaries(build_key)` method (`tt_metal/impl/kernels/kernel.hpp` line 197) returns `std::vector<const ll_api::memory*>` -- the compiled ELFs as `ll_api::memory` objects with typed span information, text sizes, and base addresses. The binary data is stored in `binaries_` (`kernel.hpp` line 241).

3. **Kernel groups carry complete launch messages.** Each `KernelGroup` (defined in `tt_metal/impl/program/program_impl.hpp` lines 69-94) contains the `launch_msg_t` with finalized RTA offsets, CB masks, and kernel text offsets. The `go_msg_t` is also populated.

4. **Runtime args are still in structured form.** `Kernel::runtime_args(core_coord)` returns `std::vector<uint32_t>&` per core. After assembly, these are serialized into command buffers and lose their per-core association.

5. **Circular buffer configs and semaphores are accessible.** `ProgramImpl::circular_buffers()` and `ProgramImpl::semaphores()` return the structured objects before they become write commands.

By contrast, hooking at `update_program_dispatch_commands` (H4, dispatch.cpp:2127) provides access to the final patched state (config buffer addresses, stall counts, go signal dispatch core coordinates) but occurs only on the cached re-dispatch path and requires the initial `ProgramCommandSequence` to have already been assembled. The primary hook at Site B fires on the initial assembly and captures the complete structured state.

**PROPOSED:** Use Site B as the primary hook for first-dispatch capture. For subsequent dispatches (cache-hit path), a secondary hook at H4 captures the delta (updated RTAs and CB addresses). This dual-hook architecture separates the complete initial capture from lightweight per-dispatch deltas.

### 2.4 Hook Priority Summary

| Hook | Location | Captures | Use Case |
|------|----------|----------|----------|
| Site B (primary) | `dispatch.cpp:1890`, inside `assemble_device_commands` | Full structured program state | First dispatch, LLK replay |
| H4-update (secondary) | `dispatch.cpp:2127`, end of `update_program_dispatch_commands` | Delta: RTAs, CB addresses | Subsequent dispatches |
| H1 (auxiliary) | `program.cpp` at Inspector callback (line 1501) | ELF binaries at compile time | Binary caching, GDB symbols |

---

## 3. Data Available at the Primary Hook

### 3.1 Dual Data Sources

When the primary hook fires inside `assemble_device_commands`, the interceptor has simultaneous access to two complementary data sources:

```
+---------------------------+     +--------------------------------+
| ProgramImpl& program      |     | ProgramCommandSequence& cmd_seq|
+---------------------------+     +--------------------------------+
| kernels_ (map<Handle,     |     | runtime_args_command_sequences |
|   shared_ptr<Kernel>>)    |     | program_config_buffer_cmd_seq  |
| circular_buffers_          |     | program_binary_command_sequence|
|   (vector<shared_ptr<      |     | launch_msg_command_sequence    |
|    CircularBufferImpl>>)  |     | go_msg_command_sequence        |
| semaphores_               |     | cb_configs_payloads            |
| kernel_groups_            |     | rta_updates                    |
| program_configs_          |     +--------------------------------+
+---------------------------+
```

The `ProgramImpl` gives the structured, typed view (kernel names, core ranges, config types). The `ProgramCommandSequence` gives the serialized, hardware-ready view. The interceptor reads from `ProgramImpl` for structured capture and falls back to `ProgramCommandSequence` for anything not directly accessible through the program object.

### 3.2 `ProgramImpl` Fields for Capture

The `ProgramImpl` class (`tt_metal/impl/program/program_impl.hpp`) holds the complete kernel-level state. Mapping from Ch 2 capture categories:

```cpp
// tt_metal/impl/program/program_impl.hpp (class ProgramImpl, private section)

// --- Kernel binaries (Ch 2, File 1: "Compiled Binaries") ---
std::vector<std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>> kernels_;
// Each Kernel holds: binaries_ (build_key -> vector<const ll_api::memory*>)
//                    core_to_runtime_args_
//                    common_runtime_args_

// --- Circular buffers (Ch 2, File 1: "CB Configs") ---
std::vector<std::shared_ptr<CircularBufferImpl>> circular_buffers_;

// --- Semaphores (Ch 2, File 1: "Semaphores") ---
std::vector<Semaphore> semaphores_;

// --- Kernel groups / launch messages (Ch 2, File 1: "Launch Messages") ---
std::vector<std::vector<std::shared_ptr<KernelGroup>>> kernel_groups_;

// --- Program config (memory layout, lines 97-108) ---
std::vector<ProgramConfig> program_configs_;
// ProgramConfig: rta_offset, sem_offset, sem_size, cb_offset, cb_size,
//                dfb_offset, dfb_size, local_cb_size,
//                kernel_text_offset, kernel_text_size
```

### 3.3 Kernel Class Fields

The `Kernel` base class (`tt_metal/impl/kernels/kernel.hpp`) exposes the fields the interceptor captures per-kernel:

```cpp
class Kernel : public JitBuildSettings {
protected:
    KernelSource kernel_src_;                          // Source file path or code string
    std::string kernel_full_name_;                     // Name + hash (unique ID)
    CoreRangeSet core_range_set_;                      // Target cores
    std::vector<uint32_t> compile_time_args_;          // Compile-time arguments
    std::vector<std::vector<std::vector<uint32_t>>> core_to_runtime_args_;
    std::vector<uint32_t> common_runtime_args_;        // Common RTAs
    std::map<std::string, std::string> defines_;       // Preprocessor defines
    std::unordered_map<uint64_t, std::vector<const ll_api::memory*>> binaries_;
    HalProgrammableCoreType programmable_core_type_;   // TENSIX, ACTIVE_ETH, IDLE_ETH
    HalProcessorClassType processor_class_;            // DM or COMPUTE
};
```

Three concrete subclasses: `DataMovementKernel` (line 251), `EthernetKernel` (line 293), `ComputeKernel` (line 331).

---

## 4. **PROPOSED:** `KernelSnapshotCaptureContext` Class Design

**PROPOSED:** A new singleton class `KernelSnapshotCaptureContext` following the pattern of `LightMetalCaptureContext` (`tt_metal/impl/lightmetal/lightmetal_capture.hpp`). Unlike extending `LightMetalCaptureContext` via inheritance (which is problematic because the base is a singleton with deleted copy/move constructors), the proposed class is a standalone context with its own mutex and per-filter capture logic.

```cpp
// PROPOSED: tt_metal/impl/debug/snapshot/kernel_snapshot_capture.hpp

#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>

namespace tt::tt_metal {

// Forward declarations
class IDevice;
class Kernel;
struct KernelGroup;
struct ProgramConfig;
namespace detail { class ProgramImpl; }

// Configuration parsed from environment variables and/or programmatic API
struct KernelCaptureConfig {
    enum class Mode : uint8_t {
        Disabled = 0,
        MetadataOnly = 1,     // ~300 B per invocation, <1% overhead
        Selective = 2,        // Full capture for matching programs/kernels
        FailureTriggered = 3, // Capture on Watcher assert or timeout
    };

    Mode mode = Mode::Disabled;

    // Selective mode filters
    std::string kernel_name_pattern;         // glob: "matmul*", "*eltwise*"
    std::optional<uint64_t> program_id;      // Capture specific program ID
    std::optional<CoreCoord> target_core;    // Capture specific logical core
    uint32_t max_captures = 100;             // Limit total captures per session

    // Ring buffer settings for failure-triggered mode
    uint32_t ring_buffer_capacity = 16;
    bool include_l1_dump = false;            // Read L1 memory via NOC
    bool include_dram_buffers = false;       // Read DRAM backing buffers

    // Output directory (defaults to $TT_METAL_HOME/generated/kernel_captures/)
    std::string output_dir;
};

// Intermediate representation of a captured invocation (see File 03 for schema)
struct CapturedInvocation;  // Defined in kernel_snapshot_types.hpp

class KernelSnapshotCaptureContext {
public:
    // Singleton access (mirrors LightMetalCaptureContext::get())
    static KernelSnapshotCaptureContext& get();

    KernelSnapshotCaptureContext(const KernelSnapshotCaptureContext&) = delete;
    KernelSnapshotCaptureContext& operator=(const KernelSnapshotCaptureContext&) = delete;

    // --- Configuration ---
    void configure(const KernelCaptureConfig& config);
    const KernelCaptureConfig& config() const { return config_; }
    bool is_active() const {
        return config_.mode != KernelCaptureConfig::Mode::Disabled;
    }

    // --- Primary Capture Entry Point (Site B) ---
    bool maybe_capture(
        detail::ProgramImpl& program,
        IDevice* device,
        SubDeviceId sub_device_id);

    // --- Secondary Hook (H4-update) ---
    void maybe_capture_delta(
        detail::ProgramImpl& program,
        const ProgramCommandSequence& cmd_seq,
        const ProgramDispatchMetadata& dispatch_md);

    // --- Failure-Triggered Capture ---
    void capture_on_failure(
        IDevice* device,
        const CoreCoord& failing_core,
        const std::string& failure_reason);

    // --- Ring Buffer Access ---
    std::vector<const CapturedInvocation*> recent_captures(uint32_t count) const;

    // --- Flush to Disk ---
    std::string flush_to_disk(const CapturedInvocation& invocation);

    // --- Statistics ---
    uint32_t total_captures() const {
        return total_captures_.load(std::memory_order_relaxed);
    }

private:
    KernelSnapshotCaptureContext();

    bool should_capture(
        const detail::ProgramImpl& program,
        const std::vector<std::shared_ptr<KernelGroup>>& kernel_groups) const;

    CapturedInvocation capture_program(
        detail::ProgramImpl& program,
        IDevice* device,
        SubDeviceId sub_device_id);

    void push_to_ring_buffer(CapturedInvocation&& invocation);

    KernelCaptureConfig config_;
    mutable std::mutex mutex_;
    std::deque<CapturedInvocation> ring_buffer_;
    std::atomic<uint32_t> total_captures_{0};
    std::atomic<uint64_t> total_bytes_{0};
    std::atomic<uint64_t> invocation_counter_{0};
};

}  // namespace tt::tt_metal
```

### 4.1 Design Rationale: Standalone vs. Extending LightMetal

The `LightMetalCaptureContext` singleton has a global toggle and is not thread-safe for per-CQ capture. A standalone `KernelSnapshotCaptureContext` is preferred because:

- **Thread safety by design** -- per-CQ mutex vs. LightMetal's unsynchronized singleton.
- **Schema independence** -- interceptor schema can evolve without forcing LightMetal version bumps.
- **Selective capture** -- per-filter capture logic vs. LightMetal's all-or-nothing toggle.
- **Different lifecycles** -- LightMetal records an entire session; snapshots are per-invocation.

A read-only bridge to LightMetal's object ID maps (`program_id_to_global_id_map_`) provides cross-reference capability without coupling the two systems (Ch 3, File 1).

---

## 5. **PROPOSED:** Primary Hook Insertion Point

**PROPOSED:** The hook is inserted inside `assemble_device_commands()` in `tt_metal/impl/program/dispatch.cpp`, after the LOG_TRACE_LAZY calls but before command assembly begins. At this point all program state is fully resolved: kernel groups populated, ProgramConfig offsets finalized, binaries compiled and cached, CB allocations complete.

```cpp
// PROPOSED insertion in dispatch.cpp (around line 1896, after logging)
void assemble_device_commands(
    ProgramCommandSequence& program_command_sequence,
    ProgramImpl& program,
    IDevice* device,
    SubDeviceId sub_device_id,
    bool use_prefetcher_cache) {
    LOG_TRACE_LAZY(tt::LogDispatch, "");
    LOG_TRACE_LAZY(
        tt::LogDispatch,
        "========== Assembling Device Commands for Program ID {} ==========",
        program.get_id());

    // ===== PROPOSED: Kernel Snapshot Capture Hook (Site B) =====
    #if defined(TT_ENABLE_KERNEL_CAPTURE)
    {
        auto& snapshot_ctx = KernelSnapshotCaptureContext::get();
        if (snapshot_ctx.is_active()) {
            snapshot_ctx.maybe_capture(program, device, sub_device_id);
        }
    }
    #endif
    // ===== End Kernel Snapshot Capture Hook =====

    CommandConstants constants{};
    // ... rest of existing function unchanged ...
```

This placement guarantees:
1. All state is finalized (offsets, binaries, CB allocations, kernel groups).
2. The `ProgramImpl&` reference provides friend-level access to private members.
3. The `IDevice*` pointer enables L1 readback if configured.
4. No command bytes have been assembled yet, so the capture has zero interaction with the command stream.
5. The overhead of the `is_active()` check in the non-capture path is a single atomic load (~1 ns).

### 5.1 Friend Access Requirement

`ProgramImpl` already grants friend access to `assemble_device_commands` (program_impl.hpp lines 430-441). **PROPOSED:** Add a friend declaration for the capture context:

```cpp
    friend class KernelSnapshotCaptureContext;
```

This grants read access to `kernels_`, `circular_buffers_`, `semaphores_`, `kernel_groups_`, and `program_configs_` -- all private members needed for complete capture, matching the existing pattern where `HWCommandQueue` and `assemble_device_commands` are already friends.

---

## 6. **PROPOSED:** Capture Implementation

### 6.1 `maybe_capture()` -- Entry Point

```cpp
bool KernelSnapshotCaptureContext::maybe_capture(
    detail::ProgramImpl& program,
    IDevice* device,
    SubDeviceId sub_device_id) {

    // Fast path: check capture limit
    if (total_captures_.load(std::memory_order_relaxed) >= config_.max_captures) {
        return false;
    }

    // Evaluate trigger conditions against all kernel groups
    uint32_t pct_count = program.get_program_config_sizes().size();
    bool should = false;
    for (uint32_t pct_idx = 0; pct_idx < pct_count; ++pct_idx) {
        auto& kernel_groups = program.get_kernel_groups(pct_idx);
        if (should_capture(program, kernel_groups)) {
            should = true;
            break;
        }
    }
    if (!should) return false;

    // Perform the capture
    CapturedInvocation invocation = capture_program(program, device, sub_device_id);
    total_captures_.fetch_add(1, std::memory_order_relaxed);

    // Route to ring buffer or disk depending on mode
    if (config_.mode == KernelCaptureConfig::Mode::FailureTriggered) {
        push_to_ring_buffer(std::move(invocation));
    } else {
        flush_to_disk(invocation);
    }
    return true;
}
```

### 6.2 `capture_program()` -- Full Program Capture

The core capture logic iterates kernel groups and collects state. It runs on the dispatch thread (the thread calling `EnqueueProgram`), so it is inherently single-threaded with respect to the program being captured.

```cpp
CapturedInvocation KernelSnapshotCaptureContext::capture_program(
    detail::ProgramImpl& program,
    IDevice* device,
    SubDeviceId sub_device_id) {

    CapturedInvocation inv;
    inv.program_id = program.get_id();
    inv.capture_timestamp_ns = /* steady_clock nanoseconds */;
    inv.device_id = device->id();
    inv.git_hash = "621b949";  // In production: from build metadata
    inv.architecture = MetalContext::instance().get_cluster().arch_name();

    uint64_t build_key = BuildEnvManager::get_instance()
        .get_device_build_env(device->build_id()).build_key();
    inv.build_key = build_key;

    uint32_t pct_count = program.get_program_config_sizes().size();
    for (uint32_t pct_idx = 0; pct_idx < pct_count; ++pct_idx) {
        // Capture ProgramConfig (L1 memory layout)
        const ProgramConfig& pc = program.get_program_config(pct_idx);
        inv.program_config = {
            .rta_offset = pc.rta_offset,
            .sem_offset = pc.sem_offset,
            .sem_size = pc.sem_size,
            .cb_offset = pc.cb_offset,
            .cb_size = pc.cb_size,
            .kernel_text_offset = pc.kernel_text_offset,
            .kernel_text_size = pc.kernel_text_size,
        };

        auto& kernel_groups = program.get_kernel_groups(pct_idx);
        auto& kernels_map = program.get_kernels(pct_idx);

        for (const auto& kg : kernel_groups) {
            // Capture kernel metadata and binaries
            for (KernelHandle kh : kg->kernel_ids) {
                auto kernel_ptr = kernels_map.at(kh);
                capture_kernel_metadata(kernel_ptr, inv);
                // Binary data: kernel_ptr->binaries(build_key)
                // Returns vector<const ll_api::memory*>
                // Each ll_api::memory has data(), get_text_size(), get_text_addr()
            }

            // Per-core capture (filtered by target_core if set)
            for (const auto& core_range : kg->core_ranges.ranges()) {
                for (auto x = core_range.start_coord.x;
                     x <= core_range.end_coord.x; ++x) {
                    for (auto y = core_range.start_coord.y;
                         y <= core_range.end_coord.y; ++y) {
                        CoreCoord logical_core{x, y};

                        // Single-core filter
                        if (config_.target_core.has_value() &&
                            logical_core != config_.target_core.value()) {
                            continue;
                        }

                        CoreCoord physical_core =
                            device->worker_core_from_logical_core(logical_core);
                        // Capture: binaries, runtime args, CB configs,
                        //          semaphores, optional L1 readback
                    }
                }
            }

            // Capture launch message from kernel group
            serialize_launch_msg(kg->launch_msg, kg->core_ranges, inv);
        }
    }

    // Capture semaphores
    for (const auto& sem : program.semaphores()) {
        serialize_semaphore(sem, inv);
    }

    // Capture circular buffer configs
    for (const auto& cb : program.circular_buffers()) {
        serialize_circular_buffer(cb, inv);
    }

    return inv;
}
```

---

## 7. Auxiliary Hooks

### 7.1 Hook A: Post-Compilation ELF Capture

The existing `Inspector::program_kernel_compile_finished()` callback (declared in `tt_metal/impl/debug/inspector/inspector.hpp` line 39) fires at `tt_metal/impl/program/program.cpp` line 1501, after each kernel's JIT build completes. This hook receives the `Kernel` shared pointer and `JitBuildOptions`.

**PROPOSED:** Register a parallel callback to cache ELF binary metadata before potential JIT cache eviction:

```cpp
// PROPOSED: Extension of Inspector callback chain
void KernelSnapshotCaptureContext::on_kernel_compiled(
    const detail::ProgramImpl* program,
    const IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const JitBuildOptions& build_options) {

    if (config_.mode == KernelCaptureConfig::Mode::Disabled) return;

    std::lock_guard<std::mutex> lock(mutex_);
    CompileRecord record;
    record.kernel_name = kernel->name();
    record.build_key = /* from device build env */;
    record.has_debug_info =
        MetalContext::instance().rtoptions().get_riscv_debug_info_enabled();
    record.elf_paths = kernel->file_paths(*device);
    compile_records_[record.build_key] = std::move(record);
}
```

This data enriches snapshots with whether debug symbols are present and disk paths for GDB attachment.

### 7.2 Hook B: Buffer Write Interception

Tile data reaches the device via `EnqueueWriteBuffer`. The existing LightMetal capture already records these writes (`CaptureEnqueueWriteBuffer` in `tt_metal/impl/lightmetal/host_api_capture_helpers.hpp`).

**PROPOSED:** In `Selective` mode, intercept the same `EnqueueWriteBuffer` path to capture raw tile data for input CBs:

```cpp
void KernelSnapshotCaptureContext::on_enqueue_write_buffer(
    const Buffer& buffer,
    const void* src_data,
    uint32_t size_bytes) {

    if (config_.mode != KernelCaptureConfig::Mode::Selective) return;
    if (is_tracked_input_buffer(buffer)) {
        snapshot_buffer_data(buffer.address(), buffer.page_size(),
                             src_data, size_bytes);
    }
}
```

### 7.3 Hook C: Post-Execution Output Capture

After the program runs, outputs can be captured by reading L1 or DRAM via the UMD `Cluster::read_from_device()` API (declared in `build_Release/include/umd/device/cluster.hpp` line 360):

```cpp
void Cluster::read_from_device(
    void* mem_ptr, ChipId chip, CoreCoord core, uint64_t addr, uint32_t size);
```

**PROPOSED:** An optional post-execution hook reads output CB regions, enabling differential debugging (compare expected vs. actual).

---

## 8. Single-Core Isolation

A full program may span hundreds of cores. For debugging a specific LLK kernel (e.g., a matmul compute kernel on core (1,2)), the `CaptureFilter` structure enables targeted capture:

```cpp
CaptureFilter filter;
filter.program_id = program.get_id();
filter.target_core = CoreCoord(1, 2);
filter.kernel_name_substr = "matmul";
```

When `target_core` is set, the per-core capture loop skips all cores not matching the filter. For a kernel group where all cores share the same binary, only one binary copy is stored; per-core data (runtime args, semaphore values) is captured only for the target core.

**Size impact:** Single-core capture reduces snapshot size from ~395 KB (64-core matmul, structured state only) to ~95 KB, or from ~38 MB (with L1 data) to ~600 KB. This makes iterative debugging practical.

---

## 9. L1 Memory Readback

For failure-triggered captures and deep snapshots, the interceptor reads raw L1 content beyond what the structured state provides. This uses the same UMD path as Watcher's device reader (`tt_metal/impl/debug/watcher_device_reader.hpp`).

Key L1 regions to capture (referencing ProgramConfig from `tt_metal/impl/program/program_impl.hpp` lines 97-108):

| Region | Address Source | Typical Size |
|--------|---------------|-------------|
| Runtime args | `ProgramConfig::rta_offset` | 0-1364 B (up to 341 uint32 per core) |
| Semaphores | `ProgramConfig::sem_offset` | 0-256 B (up to 16 semaphores x 16 B) |
| CB configs | `ProgramConfig::cb_offset` | 0-512 B (local + remote CB config words) |
| Kernel text | `ProgramConfig::kernel_text_offset` | 4-64 KB per RISC |
| Firmware mailbox | HAL-defined addresses | 128 B |
| Stack region | Below kernel text | 1-4 KB |

**PROPOSED:** L1 readback is performed in 4 KB pages to avoid overwhelming the NOC:

```cpp
// Read L1 in pages via UMD
static constexpr uint32_t kPageSize = 4096;
for (uint32_t offset = 0; offset < l1_size; offset += kPageSize) {
    uint32_t chunk_size = std::min(kPageSize, l1_size - offset);
    cluster.read_from_device(
        data + offset, device->id(), physical_core, offset, chunk_size);
}
```

L1 readback timing (empirical on Wormhole B0): ~10 us per 4 KB page. Full L1 (1 MB WH): ~2.5 ms. Full L1 (1.5 MB BH): ~3.8 ms. 64-core full dump: ~240 ms. These times are acceptable for post-mortem but not for per-dispatch capture.

---

## 10. Performance Overhead Model

**PROPOSED:** The overhead varies by capture mode and is modeled against a baseline `EnqueueProgram` dispatch of ~45 us for a single-op matmul:

| Mode | Primary Cost | Per-Dispatch Overhead | Notes |
|------|-------------|----------------------|-------|
| Disabled | `is_active()` atomic load | ~1 ns | Branch prediction (always-not-taken) |
| MetadataOnly | Metadata copy to ring buffer | ~1 us (<1%) | ~300 B kernel names + IDs |
| Selective (single core, no L1) | Binary serialization + FlatBuffer build | ~15-20 us (~35%) | ~95 KB snapshot |
| Selective (64 cores, no L1) | Full program serialization | ~50-70 us (~120%) | ~395 KB snapshot |
| Selective (single core + L1) | Above + L1 readback | ~2.5-4 ms | Dominated by PCIe reads |
| FailureTriggered (no fault) | Same as MetadataOnly | ~1 us (<1%) | Ring buffer push only |
| FailureTriggered (on fault) | L1 readback after halt | ~2.5-75 ms | Off critical path (post-mortem) |

**Mitigation strategies:**

1. **Binary deduplication**: All cores in a `KernelGroup` share the same binary. Store one copy and reference it by hash (64x reduction for 64-core matmul). See File 03 for details.
2. **Shadow dispatch**: Copy the ~13 KB `ProgramCommandSequence` inside the mutex (~2 us), defer serialization to a background thread via an SPSC queue. The `shared_ptr<Kernel>` refcount prevents ELF lifetime issues.
3. **Lazy L1 readback**: Defer L1 reads until flush time; in many cases the structured state from the hook is sufficient.

---

## 11. Compile-Time Gating

**PROPOSED:** Following the LightMetal pattern (`LIGHT_METAL_TRACE_FUNCTION_CALL` in `host_api_capture_helpers.hpp` line 46) and the Watcher `WATCHER_ENABLED` pattern:

```cpp
// Compile-time gate for production builds (zero cost when disabled)
#if defined(TT_ENABLE_KERNEL_CAPTURE) && (TT_ENABLE_KERNEL_CAPTURE == 1)

#define KERNEL_CAPTURE_MAYBE_SNAPSHOT(program, device, sub_device_id) \
    do { \
        auto& ctx = KernelSnapshotCaptureContext::get(); \
        if (ctx.is_active()) { \
            ctx.maybe_capture(program, device, sub_device_id); \
        } \
    } while (0)

#else
#define KERNEL_CAPTURE_MAYBE_SNAPSHOT(program, device, sub_device_id) \
    do {} while (0)
#endif
```

CMake option:

```cmake
# PROPOSED: CMakeLists.txt addition
option(TT_ENABLE_KERNEL_CAPTURE "Enable kernel capture interceptor" ON)
if(TT_ENABLE_KERNEL_CAPTURE)
    target_compile_definitions(tt_metal PRIVATE TT_ENABLE_KERNEL_CAPTURE=1)
endif()
```

When `TT_ENABLE_KERNEL_CAPTURE` is not set, all hooks compile to nothing -- zero overhead in production builds. When compiled in but disabled at runtime (`TT_METAL_KERNEL_CAPTURE=off`), the overhead is a single branch per dispatch (~1 ns).

---

## 12. Thread Safety Model

The interceptor's thread safety is derived from architectural alignment with the dispatch path:

1. **Capture path** (`maybe_capture`, `capture_program`): runs on the dispatch thread. No synchronization needed for reading `ProgramImpl` state because the program object is not concurrently modified during command assembly (it is under the CQ mutex in `FDMeshCommandQueue::enqueue_mesh_workload`, `tt_metal/distributed/fd_mesh_command_queue.cpp`).

2. **Ring buffer access**: protected by `mutex_` because `push_to_ring_buffer()` runs on the dispatch thread while `recent_captures()` and `capture_on_failure()` may be called from the Watcher polling thread.

3. **Atomic counters**: `total_captures_` and `total_bytes_` use relaxed ordering -- advisory statistics only.

4. **Disk I/O**: `flush_to_disk()` performs file I/O on the calling thread. For production use, a background flusher thread could be added.

---

## 13. Integration with Existing Infrastructure

### 13.1 Inspector Callbacks

The Inspector system (`tt_metal/impl/debug/inspector/inspector.hpp`) provides compile-event callbacks. The interceptor registers as an additional listener:

| Inspector Hook | Source Location | Interceptor Use |
|----------------|----------------|-----------------|
| `program_created` | `program.cpp` | Register program for tracking |
| `program_kernel_compile_finished` | `program.cpp` line 1501 | Cache ELF binaries (Hook A) |
| `program_compile_finished` | `program.cpp` | Close ELF capture window |

### 13.2 `data_collection.hpp`

The existing `data_collection.hpp` (`tt_metal/impl/dispatch/data_collection.hpp`) provides `RecordDispatchData`, `RecordKernelGroup`, and `RecordProgramRun`. The `MetadataOnly` mode piggybacks on the `RecordProgramRun` call path to record lightweight metadata per dispatch without additional hooks.

### 13.3 Watcher Integration

The Watcher system (`tt_metal/impl/debug/watcher_server.hpp`) provides `killed_due_to_error()` (line 34) and `exception_message()` (line 36). When Watcher detects a device-side assert, the interceptor flushes its ring buffer and performs a post-mortem L1 capture. See File 02 for the detailed failure-triggered capture flow.

### 13.4 CLI Tool: `tt-kernel-capture`

**PROPOSED:** A standalone CLI tool providing capture management and replay initiation:

```bash
$ tt-kernel-capture list                     # List captured snapshots
$ tt-kernel-capture inspect snap.ttksnap     # Show snapshot contents
$ tt-kernel-capture replay snap.ttksnap --gdb # Replay under GDB
$ tt-kernel-capture diff a.ttksnap b.ttksnap # Compare two snapshots
```

The `inspect` command extracts individual fields without full deserialization using FlatBuffer's random-access properties. The `replay` command supports three targets: emulator (Spike/QEMU), hardware (single-core via NOC writes), and host functional model (recompile with LLK stubs).

---

## 14. Architecture Portability

The interceptor operates at the host-side dispatch level, above the Hardware Abstraction Layer (`tt_metal/llrt/hal/`). Architecture-specific concerns:

| Concern | Wormhole B0 | Blackhole | Quasar |
|---------|-------------|-----------|--------|
| L1 size | 1 MB | 1.5 MB | 1.5 MB |
| Compute processors per core | 3 (TRISC0/1/2) | 3 (TRISC0/1/2) | Up to 16 (NEO cluster) |
| Snapshot size (single core) | ~400 KB | ~600 KB | ~2.4 MB (16 procs) |
| Hardware debug registers | **No** RISC_DBG_CNTL (breakpoint APIs only) | `rvdbg_cmd` + Debug Module | `rvdbg_cmd` + Debug Module |
| L1 readback via PCIe | TLB | TLB | TLB |

Note: Wormhole B0 does **not** have RISC_DBG_CNTL registers (only Blackhole and Quasar do). The `CAPTURE_BREAKPOINT()` macro (File 02) uses L1 mailbox polling rather than hardware debug registers, ensuring WH compatibility.

For Quasar, the interceptor must additionally capture NEO cluster assignment from `QuasarComputeKernel::compute_processors_` and per-compute-processor binary and runtime args.

---

## Key Takeaways

- **The primary hook sits inside `assemble_device_commands()` (dispatch.cpp line 1890), after `finalize_offsets()` resolves all L1 memory layout but before state is flattened into byte-level command sequences.** This is the unique point where binaries, runtime args, CB configs, semaphores, and launch messages are simultaneously available in their structured C++ representations. A secondary hook at `update_program_dispatch_commands()` (dispatch.cpp:2127) captures per-dispatch deltas on the cached path.

- **The `KernelSnapshotCaptureContext` is a standalone singleton** rather than an extension of `LightMetalCaptureContext`, providing per-CQ thread safety, independent schema evolution, and selective per-filter capture. It follows the compile-time gating pattern established by `LIGHT_METAL_TRACE_FUNCTION_CALL` and `WATCHER_ENABLED`.

- **Three auxiliary hooks supplement the primary hook**: post-compilation ELF caching via Inspector, buffer write interception for tile data, and optional post-execution L1 readback using the UMD `read_from_device` API (cluster.hpp line 360).

- **Single-core isolation reduces snapshot size by 4-60x** (from ~395 KB to ~95 KB for structured state, or from ~38 MB to ~600 KB with L1 data), making iterative debugging practical. The `target_core` filter in `KernelCaptureConfig` drives per-core filtering in the capture loop.
