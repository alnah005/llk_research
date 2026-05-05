# Chapter 6: Interceptor Design -- Where to Hook and What to Serialize

## Purpose

This chapter specifies the precise interceptor architecture, capture API,
selective trigger mechanisms, and the FlatBuffer-based snapshot format for
kernel invocation captures.  It builds on the state inventory (Chapter 2),
LightMetal foundation (Chapter 3), dispatch path hooks (Chapter 4), and
feasibility constraints from debug infrastructure and hardware capabilities
(Chapter 5).

## Key Findings

1. **Primary hook at Site B (`assemble_device_commands`, dispatch.cpp:1890)
   captures state in its most structured form.**  At this point all L1 offsets
   are resolved (`finalize_offsets` at program.cpp:1702), kernel binaries are
   available via `Kernel::binaries()`, runtime args are per-core, CB configs
   and semaphores are accessible, and launch messages are populated -- but
   nothing has been serialized into raw command bytes yet.

2. **A standalone `KernelSnapshotCaptureContext` singleton is preferred over
   extending `LightMetalCaptureContext`.**  The existing capture context is
   a singleton with deleted copy constructors and a different lifecycle
   (whole-program vs per-invocation), making inheritance impractical.  The
   standalone design follows the same compile-time gating and macro patterns.

3. **Three capture modes span the overhead spectrum.**  Always-on lightweight
   (~300 bytes/dispatch, <1% overhead), selective deep capture (5-80 ms per
   snapshot depending on L1 scope), and failure-triggered post-mortem capture
   (via WatcherServer integration).  A ring buffer of recent lightweight
   snapshots enables time-travel debugging with bounded memory (512 MB cap).

4. **The `.ttksnap` FlatBuffer format achieves 93% coverage of the Ch2 state
   inventory.**  A single-core compute kernel snapshot is ~22 KB compressed
   without tile data; a 64-core matmul with binary deduplication is ~384 KB
   without L1 dumps or ~10-15 MB compressed with full L1.  Binary
   deduplication provides up to 64x storage reduction for shared-binary
   kernel groups.

5. **The `CAPTURE_BREAKPOINT()` device-side macro provides interactive
   capture on demand.**  It reuses the `debug_ring_buf_msg_t` mailbox
   (data[31]) with a magic sentinel, detected by Watcher polling, enabling
   the kernel to pause while the host performs a deep L1 snapshot.

## Chapter Contents

### [6.1 Interceptor Architecture and Hooks](01_interceptor_architecture_and_hooks.md)

Dispatch call chain (EnqueueProgram -> generate_dispatch_commands ->
assemble_device_commands -> update_program_dispatch_commands ->
write_program_command_sequence), primary hook at Site B with rationale,
`KernelSnapshotCaptureContext` singleton design, three auxiliary hooks
(post-compilation via Inspector, buffer write interception, post-execution
L1 readback), single-core isolation via `CaptureFilter`, thread-safety model,
compile-time gating, `tt-kernel-capture` CLI tool design, performance
overhead model.

### [6.2 Capture Triggers and Modes](02_capture_triggers_and_modes.md)

Environment variable interface (`TT_METAL_KERNEL_CAPTURE` family following
`TT_METAL_WATCHER` pattern), `ParseKernelCaptureEnv()` implementation,
trigger evaluation logic (kernel name glob, program ID, core coordinate,
dispatch count), `CAPTURE_BREAKPOINT()` macro with L1 mailbox protocol,
ring buffer with FIFO eviction and adaptive sizing, WatcherServer
failure-triggered capture, `data_collection.hpp` integration, three
scenario walkthroughs (on-assert, selective-by-name, iteration-based),
overhead analysis per mode.

### [6.3 Snapshot Schema and Serialization](03_snapshot_schema_and_serialization.md)

Complete FlatBuffer `.fbs` schema (`kernel_snapshot.fbs`) with
`KernelBinaryBlob`, `CircularBufferSnapshot`, `SemaphoreSnapshot`,
`L1RegionSnapshot`, `RuntimeArgsSnapshot`, `ProgramConfigSnapshot`
(all 10 fields including dfb_offset/dfb_size), `TensixCoreSnapshot`,
`KernelInvocationSnapshot` (root type, file_identifier "TKSN"),
serialization and deserialization C++ code, binary deduplication via
`compute_hash()`, two-tier compression (LZ4 capture-time at 2 GB/s,
zstd archival), sparse L1 representation, delta serialization for
cache-hit dispatches, storage layout at
`$TT_METAL_HOME/generated/kernel_captures/`, worked size estimates
(single-core eltwise ~22 KB, 8x8 matmul ~384 KB deduplicated),
schema versioning, 93% Ch2 state coverage mapping.

## Cross-References

- **Ch1 (Hardware Debugger Reference):** The interceptor implements the
  "capture-and-replay" pattern identified in Ch1's decision matrix as the
  alternative to live hardware attach.
- **Ch2 (State Inventory):** The `.ttksnap` schema is the serialized form
  of Ch2's 8 state categories; the 93% coverage mapping identifies what
  is captured and what requires runtime reconstruction.
- **Ch3 (LightMetal Foundation):** The interceptor reuses LightMetal's
  FlatBuffer infrastructure (`base_types.fbs`, `program_types.fbs`) and
  macro gating pattern, but uses a standalone context rather than extending
  `LightMetalCaptureContext`.
- **Ch4 (Dispatch and Coordination):** Hook point selection directly uses
  the interception point analysis from Ch4 File 2; the ~13 KB command
  sequence size informs overhead estimates.
- **Ch5 (Debug and Hardware):** Failure-triggered capture integrates with
  Watcher's `killed_due_to_error()` and the `PAUSE()` macro pattern;
  `CAPTURE_BREAKPOINT()` reuses the `debug_ring_buf_msg_t` mailbox.
  WH's lack of `RISC_DBG_CNTL` registers means on-hardware replay (Ch7)
  must use ebreak+mailbox on WH vs debug registers on BH/Quasar.
- **Ch7 (proposed):** The `.ttksnap` format is consumed by all three
  replay targets (emulator, on-hardware, host functional model).
- **Ch8 (proposed):** Effort estimates for interceptor implementation
  feed into the phased roadmap.

## Key Source Files

| File | Path | Relevance |
|------|------|-----------|
| `dispatch.cpp` | `tt_metal/impl/program/dispatch.cpp` | Primary hook at line 1890 |
| `dispatch.hpp` | `tt_metal/impl/program/dispatch.hpp` | Dispatch API declarations |
| `program.cpp` | `tt_metal/impl/program/program.cpp` | `generate_dispatch_commands` (line 1356), `finalize_offsets` (line 1702) |
| `program_impl.hpp` | `tt_metal/impl/program/program_impl.hpp` | `ProgramConfig` (lines 97-108), `KernelGroup` (lines 69-94) |
| `kernel.hpp` | `tt_metal/impl/kernels/kernel.hpp` | `Kernel::binaries()` (line 197), `binaries_` (line 241) |
| `lightmetal_capture.hpp` | `tt_metal/impl/lightmetal/lightmetal_capture.hpp` | Capture context pattern |
| `command.fbs` | `tt_metal/impl/flatbuffer/command.fbs` | Existing FlatBuffer schema |
| `base_types.fbs` | `tt_metal/impl/flatbuffer/base_types.fbs` | Shared FlatBuffer types |
| `program_types.fbs` | `tt_metal/impl/flatbuffer/program_types.fbs` | Program FlatBuffer types |
| `inspector.hpp` | `tt_metal/impl/debug/inspector/inspector.hpp` | Compile event hooks |
| `watcher_server.hpp` | `tt_metal/impl/debug/watcher_server.hpp` | Failure detection |
| `data_collection.hpp` | `tt_metal/impl/dispatch/data_collection.hpp` | Dispatch statistics |
| `rtoptions.hpp` | `tt_metal/llrt/rtoptions.hpp` | Env var pattern |
| `dev_msgs.h` | `tt_metal/hw/inc/hostdev/dev_msgs.h` | `launch_msg_t`, `go_msg_t` |

## Notation

- **PROPOSED:** Designs not yet implemented in TT-Metal.
- File paths relative to TT-Metal repository root.
- All citations reference commit `621b949`.
