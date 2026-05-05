# Ch6 Verification Report

Verified against TT-Metal source tree at `/localdev/salnahari/testing_dir/tt-metal/` (commit 621b949).

---

## File 01: Interceptor Architecture and Hooks

### Claim: `assemble_device_commands()` at `dispatch.cpp:1890`
**CORRECT.** `grep -n` confirms function definition at line 1890 of `tt_metal/impl/program/dispatch.cpp`.

### Claim: `generate_dispatch_commands()` at `program.cpp:1356`
**CORRECT.** `ProgramImpl::generate_dispatch_commands` is at line 1356 of `tt_metal/impl/program/program.cpp`.

### Claim: `finalize_offsets()` at `program.cpp:1702`
**CORRECT.** `detail::ProgramImpl::finalize_offsets` is at line 1702 of `tt_metal/impl/program/program.cpp`.

### Claim: `ProgramConfig` at `program_impl.hpp:97-108`
**CORRECT.** `struct ProgramConfig` starts at line 97 and its closing brace is at line 108 of `tt_metal/impl/program/program_impl.hpp`. All 10 fields match exactly: `rta_offset`, `sem_offset`, `sem_size`, `cb_offset`, `cb_size`, `dfb_offset`, `dfb_size`, `local_cb_size`, `kernel_text_offset`, `kernel_text_size`.

### Claim: `KernelGroup` at `program_impl.hpp:69-94`
**CORRECT.** `struct KernelGroup` starts at line 69 and its closing brace is at line 94.

### Claim: `Kernel::binaries()` at `kernel.hpp:197`
**CORRECT.** `const std::vector<const ll_api::memory*>& binaries(uint64_t build_key) const;` is at line 197.

### Claim: `binaries_` at `kernel.hpp:241`
**CORRECT.** `std::unordered_map<uint64_t, std::vector<const ll_api::memory*>> binaries_;` is at line 241.

### Claim: Inspector hooks at `inspector.hpp:39`
**CORRECT.** `static void program_kernel_compile_finished(...)` starts at line 39 of `tt_metal/impl/debug/inspector/inspector.hpp`.

### Claim: `Cluster::read_from_device` at `cluster.hpp:360`
**CORRECT.** `void read_from_device(void* mem_ptr, ChipId chip, CoreCoord core, uint64_t addr, uint32_t size);` is at line 360 of `build_Release/include/umd/device/cluster.hpp`.

### Claim: `LightMetalCaptureContext` singleton pattern
**CORRECT.** The class at `tt_metal/impl/lightmetal/lightmetal_capture.hpp` has `static LightMetalCaptureContext& get();` (line 40), deleted copy/move constructors (lines 42-43), and private constructor (line 73) -- confirming singleton pattern.

### Claim: Friend declarations at `program_impl.hpp:430-438`
**INCORRECT (minor).** The friend declarations span lines 430-441, not 430-438. Lines 430-435 declare `friend void program_dispatch::assemble_device_commands(...)` (correct). Line 437 is `friend HWCommandQueue;` and line 438 is `friend EnqueueProgramCommand;` (correct). However, lines 439-441 add `friend Program;`, `friend distributed::MeshWorkload;`, and `friend distributed::MeshWorkloadImpl;` which are not included in the claimed range. The doc text at Section 5.1 says "lines 430-435" for just the `assemble_device_commands` friend, which is correct. But the headline claim of "430-438" is incomplete.

### Claim: `update_program_dispatch_commands` at `dispatch.cpp:2127`
**CORRECT.** Function definition at line 2127 of `tt_metal/impl/program/dispatch.cpp`.

### Claim: `write_program_command_sequence` at `dispatch.cpp:2481`
**CORRECT.** Function definition at line 2481.

### Claim: `insert_empty_program_dispatch_preamble_cmd` at `dispatch.hpp:109`
**CORRECT.** Declaration at line 109 of `tt_metal/impl/program/dispatch.hpp`.

### Claim: `insert_stall_cmds` at `dispatch.hpp:111`
**CORRECT.** Declaration at line 111.

### Claim: Inspector callback at `program.cpp:1501`
**CORRECT.** `Inspector::program_kernel_compile_finished(this, device, kernel, build_options);` is at line 1501 of `tt_metal/impl/program/program.cpp`.

### Claim: `DataMovementKernel` (line 251), `EthernetKernel` (line 293), `ComputeKernel` (line 331)
**CORRECT.** All three subclass declarations match exactly at those line numbers in `kernel.hpp`.

### Claim: `BuildEnvManager` path for `build_key` retrieval
**CORRECT.** `BuildEnvManager::get_instance().get_device_build_env(device->build_id()).build_key()` matches the actual usage pattern in `kernel.cpp` (e.g., line 587). `build_key()` is at line 25 of `build_env_manager.hpp`.

### Claim: `ll_api::memory` has `data()`, `get_text_size()`, `get_text_addr()`, `get_packed_size()`, `num_spans()`
**CORRECT.** All five methods exist in `tt_metal/llrt/tt_memory.h` at lines 47, 52, 54, 53, and 59 respectively. `data()` returns `const std::vector<word_t>&` where `word_t = std::uint32_t`.

---

## File 02: Capture Triggers and Modes

### Claim: `TT_METAL_WATCHER` env var pattern in `rtoptions.hpp`/`rtoptions.cpp`
**CORRECT.** `TT_METAL_WATCHER` and related env vars are enumerated at lines 134-152 of `rtoptions.cpp`. `ParseWatcherEnv()` is declared at line 744 of `rtoptions.hpp` and defined at line 1344 of `rtoptions.cpp`.

### Claim: Watcher env var entries at lines 134-152 of `rtoptions.cpp`
**CORRECT.** The `TT_METAL_WATCHER` family spans lines 134-152 exactly.

### Claim: `WatcherServer::killed_due_to_error()` existence and location
**CORRECT.** Declared at line 34 of `tt_metal/impl/debug/watcher_server.hpp`. `exception_message()` at line 36, also correct.

### Claim: `debug_ring_buf_msg_t` structure (32 uint32 entries)
**CORRECT.** Defined at line 273 of `tt_metal/hw/inc/hostdev/dev_msgs.h`. Contains `uint32_t data[DEBUG_RING_BUFFER_ELEMENTS]` where `DEBUG_RING_BUFFER_ELEMENTS = 32` (line 271). Also has `int16_t current_ptr` and `uint16_t wrapped` metadata fields.

### Claim: `RecordDispatchData`/`RecordKernelGroup`/`RecordProgramRun` in `data_collection.hpp`
**CORRECT.** `RecordDispatchData` at line 35, `RecordKernelGroup` at line 43, `RecordProgramRun` at line 49 of `tt_metal/impl/dispatch/data_collection.hpp`.

### Claim: `data_collector_t` enum at lines 19-24
**CORRECT (for line range)** but the doc in Section 8 **misrepresents the enum values**. The doc proposes adding to the enum with values like `DISPATCH_DATA = 0, KERNEL_GROUP = 1, PROGRAM_RUN = 2`, but the actual enum values are `DISPATCH_DATA_CB_CONFIG`, `DISPATCH_DATA_SEMAPHORE`, `DISPATCH_DATA_RTARGS`, `DISPATCH_DATA_BINARY` (lines 19-24). The existing enum values categorize dispatch data by type (CB config, semaphore, rtargs, binary), not by action. The proposed addition should use a compatible naming scheme.

### Claim: `PAUSE()` macro in `pause.h`
**CORRECT.** `PAUSE()` is defined at line 31 of `tt_metal/hw/inc/api/debug/pause.h`. Guarded by `WATCHER_ENABLED && !WATCHER_DISABLE_PAUSE && !FORCE_WATCHER_OFF`. Uses `debug_pause_msg_t` mailbox polling.

### Claim: `LIGHT_METAL_TRACE_FUNCTION_CALL` at `host_api_capture_helpers.hpp` line 46
**CORRECT.** Macro defined at line 46.

---

## File 03: Snapshot Schema and Serialization

### Claim: Existing FlatBuffer schemas: `command.fbs`, `base_types.fbs`, `program_types.fbs`
**CORRECT.** All exist at `tt_metal/impl/flatbuffer/`. Also confirmed: `buffer_types.fbs` and `light_metal_binary.fbs` at the same path.

### Claim: `CommandType` union member count (17)
**CORRECT.** The `CommandType` union in `command.fbs` (lines 118-136) contains exactly 17 members: `ReplayTraceCommand` through `LightMetalCompareCommand`.

### Claim: `ProgramConfig` fields (all 10 including `dfb_offset`, `dfb_size`)
**CORRECT.** All 10 fields verified at `program_impl.hpp` lines 97-108: `rta_offset`, `sem_offset`, `sem_size`, `cb_offset`, `cb_size`, `dfb_offset`, `dfb_size`, `local_cb_size`, `kernel_text_offset`, `kernel_text_size`.

### Claim: `ll_api::memory::data()` method
**CORRECT.** Returns `const std::vector<word_t>&` at line 47 of `tt_metal/llrt/tt_memory.h`. `word_t = std::uint32_t`.

### Claim: `BuildEnvManager` path for `build_key` retrieval
**CORRECT.** `BuildEnvManager::get_instance()` (line 44 of `build_env_manager.hpp`), `.get_device_build_env(ChipId)` (line 51), `.build_key()` (line 25 via `DeviceBuildEnv`).

### Claim: `Semaphore` class at `semaphore.hpp`
**CORRECT.** `class Semaphore` spans lines 17-48 of `tt_metal/impl/buffers/semaphore.hpp`.

### Claim: `base_types.fbs` type locations (`DataFormat` lines 45-67, `MathFidelity` lines 37-43, `DefineEntry` lines 74-77, `Tile` lines 80-83, `BoolOptional`/`Uint8Optional`/`Uint32Optional` lines 86-88)
**CORRECT.** All line ranges verified exactly.

### Claim: `program_types.fbs` type locations (lines 4-22, 24-49, 52-56, 58-77)
**INCORRECT (minor off-by-one).** Actual locations:
- `CoreCoord` through `CoreSpec`: lines 5-23 (not 4-22)
- `DataMovementConfig` through `EthernetConfig`: lines 25-50 (not 24-49)
- `KernelConfig` union: lines 53-57 (not 52-56)
- `RuntimeArgValue` through `UInt32Vector`: lines 59-78 (not 58-77)
All ranges are off by +1 on both start and end.

### Claim: `buffer_types.fbs` `CircularBufferConfig` at lines 68-81
**CORRECT.** Table starts at line 68, closing brace at line 81.

### Claim: `light_metal_binary.fbs` architecture pattern at lines 32-36
**CORRECT.** `table LightMetalBinary` at lines 32-36.

### Claim: `command.fbs` CommandType at lines 118-136
**CORRECT.** Union starts at line 118, closing brace at line 136.

---

## Required Fixes

1. **File 01, Section 5.1**: Change "program_impl.hpp lines 430-438" to "program_impl.hpp lines 430-441" to include the full friend declaration block (adds `friend Program;`, `friend distributed::MeshWorkload;`, `friend distributed::MeshWorkloadImpl;` at lines 439-441).

2. **File 02, Section 8**: The existing `data_collector_t` enum values are misrepresented. The doc shows proposed values as `DISPATCH_DATA = 0, KERNEL_GROUP = 1, PROGRAM_RUN = 2`, but the actual existing values are `DISPATCH_DATA_CB_CONFIG`, `DISPATCH_DATA_SEMAPHORE`, `DISPATCH_DATA_RTARGS`, `DISPATCH_DATA_BINARY`. The proposed `KERNEL_CAPTURE = 3` addition should follow the existing naming convention (e.g., `DISPATCH_DATA_KERNEL_CAPTURE`), and the existing values should be shown correctly.

3. **File 03, Section 2 table**: The `program_types.fbs` line ranges are all off by +1. Update:
   - `CoreCoord`, `CoreRange`, `CoreRangeSet`, `CoreSpec` from "(lines 4-22)" to "(lines 5-23)"
   - `DataMovementConfig`, `ComputeConfig`, `EthernetConfig` from "(lines 24-49)" to "(lines 25-50)"
   - `KernelConfig` union from "(lines 52-56)" to "(lines 53-57)"
   - `RuntimeArg`, `RuntimeArgValue` from "(lines 58-77)" to "(lines 59-78)"
