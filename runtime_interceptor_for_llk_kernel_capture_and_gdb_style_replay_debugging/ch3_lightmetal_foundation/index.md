# Chapter 3: LightMetal Capture -- The Existing Foundation and Its Gaps

This chapter analyzes TT-Metal's existing LightMetal capture/replay system as the closest existing prototype for kernel-level capture. It inventories what LightMetal captures today, identifies what it misses for single-kernel replay debugging, and proposes concrete schema and infrastructure extensions to bridge the gap.

**Key quantitative findings**: LightMetal's 17-command FlatBuffer vocabulary and 4-type object mapping system provide approximately 34% of the capture infrastructure needed for single-kernel replay debugging (weighted by importance to the replay use case). The largest gaps are compiled binary capture (contributing 1.25 of a possible 25 weighted points) and L1 memory state (contributing 0 of a possible 15 points). The proposed extensions add 5 new command types and a versioning layer to the existing schema, with an estimated implementation effort of 25-39 person-days across 8 phases.

---

## Sections

1. **[LightMetal Architecture and Capture Mechanics](01_lightmetal_architecture_and_capture.md)**
   The `LightMetalCaptureContext` singleton, the `TraceScope` re-entrancy guard with the full `LIGHT_METAL_TRACE_FUNCTION_CALL` macro (including `log_trace()` diagnostic and the `TT_ENABLE_LIGHT_METAL_TRACE` compile-time gate), the complete FlatBuffer command vocabulary (all 17 command types with replay status), the object-to-global-id mapping system (4 types sharing a single monotonic counter), the `LightMetalBinary` format, the sequential replay dispatch loop with disabled handlers (Issue #24955), the capture helper layer, supporting schemas (`base_types.fbs`, `buffer_types.fbs`, `program_types.fbs`), the `data_collection.hpp` dispatch instrumentation, the `LightMetalCompare` verification pattern, and the test infrastructure (C++ tests commented out, Python tests active, standalone `lightmetal_runner`).

2. **[Gaps for Kernel-Level Debugging](02_gaps_for_kernel_level_debugging.md)**
   Ten specific gaps quantified against the Ch2 state inventory: no compiled binary capture (Gap 1), no L1/CB data snapshots (Gap 2), no semaphore state (Gap 3), no NOC transaction log (Gap 4), no single-kernel dispatch granularity (Gap 5), no hardware register state (Gap 6), missing configuration fields including `named_compile_args`, `opt_level`, `noc_mode` for EthernetConfig, and `hlk_desc` (Gap 7), no common runtime args (Gap 8), no launch message capture (Gap 9), and no versioning metadata (Gap 10). Weighted quantitative assessment: LightMetal covers approximately 34% of the state needed for single-kernel replay. Gap interaction analysis and dependency ordering for resolution. Capture budget tables for three representative workloads.

3. **[Extending LightMetal for Kernel Capture](03_extending_lightmetal_for_kernel_capture.md)**
   FlatBuffer design constraints (256 union limit, no inheritance, vector nesting workaround, optional scalar pattern). PROPOSED: `KernelSnapshotCommand` FlatBuffer schema with sub-tables for `KernelBinary`, `CoreL1Snapshot`, `CoreSemaphoreState`, `CBRuntimeState`, `NamedCompileArgEntry`, and `DRAMBufferSnapshot`. Five standalone commands (`KernelSnapshotCommand`, `L1SnapshotCommand`, `SemaphoreSnapshotCommand`, `RegisterStateCommand`, `KernelOutputSnapshotCommand`). Deep capture mode architecture with `DeepTraceScope` (separate `thread_local` counter) and three capture modes (Full, Selective, ConfigOnly). Object map extensions. `EnvironmentMetadata` / `SystemDescriptor` versioning with `file_identifier "LMTL"` and feature bitmask. Integration with `data_collection.hpp` (`RecordDispatchData`, `RecordProgramRun`, `RecordKernelGroup`). Sparse L1 capture strategy. Capture and replay helper implementations including `CreateKernelFromBinary`. Eight-phase incremental adoption path (25-39 person-days). Backward compatibility analysis, testing strategy, performance overhead estimates, and risk assessment.

---

## Cross-References

- **Chapter 1** provides the prior art from GPU debugger architectures (RenderDoc, NVIDIA Nsight, AMD RGP) that motivates the capture-and-replay approach. Section 3 (Comparative Analysis) identifies capture-and-replay as the preferred strategy for Tenstorrent hardware. Section 2 (Focus Model) describes hierarchical debug identifiers that inform the kernel snapshot identity fields.

- **Chapter 2** provides the complete state inventory for kernel replay -- the target that this chapter's gap analysis (Section 2) measures LightMetal against. Section 1 defines compiled binary and build state. Section 2 defines runtime configuration. Section 3 defines memory contents and capture timing. Section 4 defines implicit and hidden state.

- **Chapter 4** (forthcoming) will design the runtime interceptor that uses the `KernelSnapshotCommand` schema and `DeepTraceScope` mechanism proposed in Section 3 to perform actual kernel capture at the dispatch boundary.

---

## Key Dependencies and Risks

- **Issue #24955** (LightMetal rearchitecture): Multiple replay handlers are currently disabled or throwing. The proposed extensions are designed as additive to avoid conflict with this ongoing work, but coordination with the LightMetal maintainers is essential.

- **`EnqueueProgram` dispatch path**: The deep capture hooks proposed in Section 3 must integrate with the dispatch path where `data_collection.hpp` functions are already called. Changes to this path affect all program execution.

- **`Kernel::binaries()` / `ll_api::ElfFile`**: Binary extraction depends on the `Kernel` class exposing compiled ELFs via the `binaries(build_key)` method (line 197 of `kernel.hpp`). The replay side requires constructing `ll_api::memory` objects from raw bytes, which is not currently supported.

---

## Key Source Files

| File | Role |
|------|------|
| `tt_metal/impl/lightmetal/lightmetal_capture.hpp` | `LightMetalCaptureContext` singleton definition (lines 38-88) |
| `tt_metal/impl/lightmetal/lightmetal_capture.cpp` | Singleton implementation, object maps, binary creation |
| `tt_metal/impl/lightmetal/lightmetal_replay_impl.hpp` | `LightMetalReplayImpl` class with replay object maps |
| `tt_metal/impl/lightmetal/lightmetal_replay_impl.cpp` | Replay dispatch loop and per-command `execute()` handlers |
| `tt_metal/impl/lightmetal/host_api_capture_helpers.hpp` | `TraceScope`, `LIGHT_METAL_TRACE_FUNCTION_CALL` macro, `TT_ENABLE_LIGHT_METAL_TRACE` gate |
| `tt_metal/impl/lightmetal/host_api_capture_helpers.cpp` | Capture helper implementations for all 16 active command types |
| `tt_metal/impl/flatbuffer/command.fbs` | FlatBuffer command vocabulary (17 types in `CommandType` union) |
| `tt_metal/impl/flatbuffer/light_metal_binary.fbs` | `LightMetalBinary` root schema (with versioning TODO) |
| `tt_metal/impl/flatbuffer/program_types.fbs` | `KernelConfig` union, core specs, runtime arg types |
| `tt_metal/impl/flatbuffer/buffer_types.fbs` | Buffer types, shard specs, `CircularBufferConfig` |
| `tt_metal/impl/flatbuffer/base_types.fbs` | Arch, processor, NOC, data format enums, optional wrappers |
| `tt_metal/impl/dispatch/data_collection.hpp` | `RecordDispatchData`, `RecordKernelGroup`, `RecordProgramRun` |
| `tt_metal/impl/kernels/kernel.hpp` | `Kernel` class, `EthernetConfig` (line 35), `binaries()` (line 197), `set_binaries()` (line 198) |
| `tt_metal/api/tt-metalium/kernel_types.hpp` | `DataMovementConfig`, `ComputeConfig`, `KernelBuildOptLevel` |
| `tt_metal/llrt/tt_elffile.hpp` | `ll_api::ElfFile` class for ELF parsing and segment access |
