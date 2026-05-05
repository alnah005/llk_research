# 4.1 The TT-Metal Dispatch Path: End-to-End Pipeline

## Overview

This section traces the complete dispatch pipeline from `EnqueueMeshWorkload` through
JIT compilation, command assembly, hugepage submission, device-side prefetcher/dispatcher
processing, and finally BRISC/NCRISC/TRISC launch on a Tensix core. Every stage is
cited to its source location in the `tt-metal` tree, with quantitative sizing for
capture feasibility.

> **Cross-reference:** Ch1 (Hardware Debugger Reference) Section 2 establishes the
> capture-and-replay paradigm that motivates identifying interception opportunities here.
> Ch2 (State Inventory) Section 1 defines the compiled binary state flowing through this
> pipeline. Ch3 (LightMetal Foundation) Section 1 describes the existing LightMetal
> capture system that this pipeline's interception extends.

---

## 4.1.1 The Big Picture: ASCII Data Flow

```
  HOST PROCESS                                           DEVICE (per chip)
  ============                                           =================

  CreateKernel() -------> Program::add_kernel()
  CreateCircularBuffer()-> Program::add_circular_buffer()
  SetRuntimeArgs() ------> Kernel::set_runtime_args()

  EnqueueMeshWorkload(mesh_cq, mesh_workload, blocking)
       |
       v
  mesh_workload.impl().compile(mesh_device)
       |  ProgramImpl::compile() per device
       |    |-- JitBuildOptions + set_build_options()
       |    |-- jit_build_genfiles_descriptors()  --> chlkc_unpack.cpp, chlkc_math.cpp, chlkc_pack.cpp
       |    |-- KernelCompileHash() via FNV1a
       |    |-- jit_build_once(hash, GenerateBinaries)
       |    |     |-- riscv-tt-elf-g++ invocations
       |    |-- kernel->read_binaries()
       |    +-- Inspector::program_compile_finished()
       |
  mesh_workload.impl().load_binaries(mesh_cq)
       |  allocate_kernel_bin_buf_on_device()
       |
  mesh_workload.impl().generate_dispatch_commands(mesh_cq)
       |  ProgramImpl::finalize_offsets(device)
       |    |-- finalize_rt_args() -> configure_rta_offsets/crta_offsets
       |    |-- finalize_sems()
       |    |-- finalize_cbs()
       |    |-- finalize_kernel_bins()
       |    +-- set_program_attrs_across_core_types()
       |         |-- populate_dispatch_data()
       |
       |  ProgramImpl::generate_dispatch_commands(device, use_prefetcher_cache)
       |    |-- insert_empty_program_dispatch_preamble_cmd()
       |    |-- insert_stall_cmds()
       |    +-- program_dispatch::assemble_device_commands()
       |          |-- assemble_runtime_args_commands()
       |          |-- SemaphoreCommandGenerator
       |          |-- CircularBufferCommandGenerator
       |          |-- ProgramBinaryCommandGenerator
       |          |-- LaunchMessageGenerator
       |          +-- GoSignalGenerator
       |
  mesh_cq.enqueue_mesh_workload(mesh_workload, blocking)
       |
       |  program_dispatch::update_program_dispatch_commands()
       |    |-- patch stall counts, config buffer addrs, RTA copies
       |    |-- update launch message write pointers
       |    +-- update go signal dispatch core coords
       |
       |  program_dispatch::write_program_command_sequence()
       |    |-- preamble -> hugepage ring buffer
       |    |-- stall (if needed)
       |    |-- runtime_args commands
       |    |-- program_config_buffer commands
       |    |-- program_binary commands
       |    |-- launch_msg commands
       |    +-- go_signal commands -> hugepage ring buffer
       |
       v                            HUGEPAGE (PCIe BAR)
  +---------------------------+
  | issue_queue_reserve()     |
  | cq_write() -> hugepage    |---->  cq_prefetch.cpp (on dispatch core)
  | fetch_queue_write()       |         reads from hugepage
  +---------------------------+         forwards to cq_dispatch.cpp
                                           |
                                           v
                              cq_dispatch.cpp (on dispatch core)
                                |-- interprets CQ commands
                                |-- NOC multicast: kernel config -> worker L1
                                |-- NOC multicast: launch_msg -> worker mailbox
                                |-- NOC multicast: go_msg -> worker go_messages[]
                                           |
                                           v
                              WORKER TENSIX CORE (each target core)
                                BRISC: polls go_messages[].signal == RUN_MSG_GO
                                  |-- firmware_config_init()
                                  |-- setup CB interfaces
                                  |-- run_triscs(enables) -> subordinate_sync
                                  |-- noc_init(), call kernel_main()
                                NCRISC: wait subordinate_sync->dm1 == RUN_SYNC_MSG_GO
                                  |-- kernel_main()
                                TRISC0/1/2: wait subordinate_sync == RUN_SYNC_MSG_GO
                                  |-- run_kernel() (chlkc unpack/math/pack)
                                  :
                                BRISC: wait_ncrisc_trisc()
                                  |-- notify_dispatch_core_done()
                                  |-- advance launch_msg_rd_ptr
```

---

## 4.1.2 Pipeline Stages with Quantitative Sizing

The dispatch pipeline has **seven logical stages**, each producing a distinct byte
stream that an interceptor may need to capture:

| # | Stage | Key Function | Output | Typical Size |
|---|-------|-------------|--------|-------------|
| 1 | Finalize offsets | `ProgramImpl::finalize_offsets()` | `ProgramConfig` per core type | ~40 B per core type |
| 2 | Populate dispatch data | `ProgramImpl::populate_dispatch_data()` | `ProgramTransferInfo` (binary pages) | 4--256 KB |
| 3 | Assemble device commands | `assemble_device_commands()` | `ProgramCommandSequence` | 0.5--64 KB |
| 4 | Reserve config buffer slot | `reserve_space_in_kernel_config_buffer()` | `ProgramDispatchMetadata` | ~32 B |
| 5 | Update dispatch commands | `update_program_dispatch_commands()` | In-place mutations of stage 3 | 0 (mutates) |
| 6 | Write to issue queue | `write_program_command_sequence()` | Byte stream to hugepage | = stage 3 total |
| 7 | Fetch queue doorbell | `fetch_queue_write()` | 4 B doorbell write | 4 B |

**Source:** `tt_metal/impl/program/dispatch.cpp` (stages 3--6),
`tt_metal/impl/program/program.cpp:1105-1262` (stage 2),
`tt_metal/impl/program/program.cpp:1356-1396` (stage 3 entry),
`tt_metal/distributed/fd_mesh_command_queue.cpp:258-453` (orchestration).

---

## 4.1.3 The ProgramImpl Class

The core data structure is `detail::ProgramImpl`, declared in
`tt_metal/impl/program/program_impl.hpp` (line 175). It holds:

| Member | Type | Purpose |
|--------|------|---------|
| `kernels_` | `vector<unordered_map<KernelHandle, shared_ptr<Kernel>>>` | Per-core-type kernel map (line 360) |
| `circular_buffers_` | `vector<shared_ptr<CircularBufferImpl>>` | All CBs in the program (line 363) |
| `semaphores_` | `vector<Semaphore>` | Software semaphores (line 381) |
| `kernel_groups_` | `vector<vector<shared_ptr<KernelGroup>>>` | Per-core-type kernel groups (line 388) |
| `program_configs_` | `vector<ProgramConfig>` | Per-core-type L1 layout (line 391) |
| `program_config_sizes_` | `vector<uint32_t>` | Config ring buffer space per core type (line 393) |
| `cached_program_command_sequences_` | `unordered_map<uint64_t, ProgramCommandSequence>` | Cached dispatch commands keyed by sub-device manager ID (line 398) |
| `program_transfer_info` | `ProgramTransferInfo` | Binary transfer metadata (line 319) |
| `kernels_buffer_` | `unordered_map<ChipId, shared_ptr<Buffer>>` | DRAM buffer for kernel binaries (line 318) |
| `compiled_` | `unordered_set<uint64_t>` | Set of build keys already compiled (line 383) |

The `KernelGroup` struct (line 69, same file) bundles kernels that share the same core
ranges. Each group carries its own `launch_msg_t` and `go_msg_t`, plus per-processor
`kernel_text_offsets`.

The `ProgramConfig` struct (`program_impl.hpp:97-108`, 10 uint32 fields = 40 B) stores the L1 memory map for one core type. For the full field inventory, see Ch2, Section 2, Item 3. The dispatch pipeline uses `ProgramConfig` to compute base addresses for all `finalize_*` operations below.

---

## 4.1.4 JIT Compilation: ProgramImpl::compile()

Defined at `tt_metal/impl/program/program.cpp:1438`.

### 4.1.4.1 Early Exit Check

If `compiled_` already contains the current `build_key`, the function returns
immediately (line 1442). The `Inspector::program_compile_already_exists()` hook fires
in this case (line 1443).

### 4.1.4.2 Parallel Kernel Build

For every kernel across all core types, the compile launches a build step via
`launch_build_step()`:

```cpp
// program.cpp:1465-1502
launch_build_step([kernel, device, this, &build_env] {
    JitBuildOptions build_options(build_env.build_env);
    kernel->set_build_options(build_options);
    this->set_cb_data_fmt(kernel->logical_coreranges(), build_options);
    this->set_cb_tile_dims(kernel->logical_coreranges(), build_options);
    ...
    auto kernel_hash = KernelCompileHash(kernel, build_options, build_env.build_key());
    ...
    jit_build_once(kernel_hash, [&] {
        GenerateBinaries(device, build_options, kernel);
    });
    Inspector::program_kernel_compile_finished(this, device, kernel, build_options);
}, events);
```

### 4.1.4.3 The FNV1a Hash and Build Cache

`KernelCompileHash` (program.cpp:163-184) produces an FNV1a hash combining the
`build_key`, the HLK descriptor hash, and the kernel's own `compute_hash()`. The
FNV1a implementation is in `tt_metal/common/stable_hash.hpp`:

```cpp
class FNV1a {
    static constexpr uint64_t FNV_PRIME = 0x100000001b3;
    static constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325;
    // XOR-then-multiply per element
};
```

The `jit_build_once` function (`build.cpp:785`) uses a singleton `JitBuildCache` with
a `std::unordered_map<size_t, State>`. For each hash:
- If not present, the calling thread executes the build lambda and marks it complete.
- If present and in-progress, the calling thread blocks on a condition variable.
- If present and complete, the call returns immediately.

This ensures that even across multiple programs in a mesh workload, each unique kernel
configuration is compiled exactly once.

The `build_state_hash_` in `JitBuildState` further captures the GCC compiler path
(`env_.gpp_`), all cflags, defines, includes, lflags, linker script path, source file
list, and default optimization levels. This hash is written to a `.build_state` file;
on the next build, if the hash matches, object files are reused.

### 4.1.4.4 The Generated Files Pipeline (genfiles.cpp)

`GenerateBinaries` (program.cpp:146-157) first calls `jit_build_genfiles_descriptors()`
(`tt_metal/jit_build/genfiles.hpp:26`, implemented in `genfiles.cpp`) to generate:

| Generated File | TRISC | Purpose |
|---|---|---|
| `chlkc_unpack.cpp` | TRISC0 (Unpacker) | `#define TRISC_UNPACK` + transformed source |
| `chlkc_math.cpp` | TRISC1 (Math) | `#define TRISC_MATH` + transformed source |
| `chlkc_pack.cpp` | TRISC2 (Packer) | `#define TRISC_PACK` + transformed source |
| `chlkc_isolate_sfpu.cpp` | Quasar SFPU | `#define TRISC_ISOLATE_SFPU` + transformed source |
| `chlkc_descriptors.h` | All TRISCs | Data format arrays, tile dimension arrays |

The descriptor header (`chlkc_descriptors.h`) contains `unpack_src_format[]`,
`unpack_dst_format[]`, `pack_src_format[]`, `pack_dst_format[]`, and tile dimension
arrays guarded by `#if defined(UCK_CHLKC_MATH)` / `UCK_CHLKC_PACK` /
`UCK_CHLKC_UNPACK` to include only the relevant arrays for each TRISC.

The generator also detects whether the kernel uses the modern `void kernel_main()`
syntax or the legacy `namespace NAMESPACE { void MAIN() { } }` syntax, and transforms
accordingly.

Then `kernel->generate_binaries()` invokes `JitBuildState::build()` (`build.hpp:165`),
which orchestrates: `compile()` (riscv-tt-elf-g++), `link()` (ELF output), and
`weaken()` (firmware symbol weakening).

### 4.1.4.5 Read Binaries

After all compilations complete (`sync_build_steps(events)`), each kernel's
`read_binaries()` is called in parallel (program.cpp:1512-1518). The `read_binaries`
call parses the ELF, extracts text sections, and stores them as `ll_api::memory`
objects. These are later packed into a contiguous buffer by `populate_dispatch_data()`.

---

## 4.1.5 Finalize Offsets: ProgramImpl::finalize_offsets()

Defined at `program.cpp:1702`. This function computes the L1 memory layout for each
programmable core type by calling a sequence of finalization functions from
`tt_metal/impl/program/dispatch.cpp`:

```
offset = 0 (start of kernel config ring buffer slot)
  |
  +-- finalize_rt_args()        (dispatch.cpp:262)  -> rta_offset, crta_offset
  +-- finalize_sems()           (dispatch.cpp:280)  -> sem_offset, sem_size
  +-- finalize_cbs()            (dispatch.cpp:299)  -> cb_offset, cb_size, local_cb_size
  +-- finalize_dfbs()           (dataflow_buffer)   -> dfb_offset, dfb_size
  +-- finalize_kernel_bins()    (dispatch.cpp:335)  -> kernel_text_offset, kernel_text_size
```

Each function advances the offset, and all offsets are L1-alignment padded. The L1
layout within a kernel config ring buffer slot:

```
  +---------------------------+  <-- kernel_config_base (from HAL)
  | Runtime Args (unique+common)  rta_offset
  +---------------------------+
  | Semaphores                    sem_offset
  +---------------------------+
  | Circular Buffer configs       cb_offset (local then remote)
  +---------------------------+
  | Dataflow Buffer configs       dfb_offset
  +---------------------------+
  | Kernel text (BRISC,NCRISC,    kernel_text_offset
  |   TRISC0, TRISC1, TRISC2)
  +---------------------------+  <-- offset = total slot size
```

The config ring buffer has `launch_msg_buffer_num_entries = 8` slots (`dev_msgs.h`,
line 380), enabling pipelining of consecutive programs.

After per-core-type layout, `set_program_attrs_across_core_types()` (program.cpp:1692)
calls `set_launch_msg_sem_offsets()` to write semaphore offsets into each kernel
group's `launch_msg_t`, then calls `populate_dispatch_data()` to prepare binary
transfer metadata.

---

## 4.1.6 Populate Dispatch Data

Defined at `program.cpp:1105`. This function iterates over all kernels, extracts binary
spans from each `ll_api::memory` object, and packs them into `program_transfer_info`:

- `binary_data` -- flat `vector<uint32_t>` of all kernel binary words, page-aligned
- `kernel_bins` -- per-kernel-group transfer descriptors with NOC multicast/unicast
  encoding of destination cores
- `num_active_cores` -- total number of cores this program targets

Each `kernel_bins_transfer_info` (source: `program_device_map.hpp:35-42`) contains
`dst_base_addrs`, `page_offsets`, `lengths`, and `processor_ids` vectors.

**Capture feasibility:** `ProgramTransferInfo` is already in serializable form. The
`binary_data` vector is contiguous and can be written directly to disk. Total overhead
to serialize: one length prefix + bulk memcpy per vector.

---

## 4.1.7 Generate Dispatch Commands: assemble_device_commands()

Defined at `dispatch.cpp:1890`. This is the central function that builds the
`ProgramCommandSequence` -- a self-contained packet of device commands.

### The ProgramCommandSequence Structure

```cpp
// File: tt_metal/impl/program/program_command_sequence.hpp:23-81
struct ProgramCommandSequence {
    // 1. Write offset relocation preamble
    HostMemDeviceCommand preamble_command_sequence;                          // ~48 B

    // 2. Stall/wait commands (two variants: uncached with prefetch stall, cached without)
    uint32_t current_stall_seq_idx = 0;
    HostMemDeviceCommand stall_command_sequences[2];                        // ~64 B

    // 3. Per-core-group runtime args
    std::vector<HostMemDeviceCommand> runtime_args_command_sequences;       // 1-64 KB

    // 4. CB configs + semaphores + CRTAs (batched multicast)
    HostMemDeviceCommand program_config_buffer_command_sequence;            // 256 B - 4 KB

    // 5. Prefetcher cache setup (if applicable)
    HostMemDeviceCommand program_binary_setup_prefetcher_cache_command;     // 0-32 B

    // 6. Kernel binary transfer commands
    HostMemDeviceCommand program_binary_command_sequence;                   // 64-256 B

    // 7. Launch messages (per kernel group)
    HostMemDeviceCommand launch_msg_command_sequence;                       // 256 B - 2 KB

    // 8. Go signal (multicast to all cores)
    HostMemDeviceCommand go_msg_command_sequence;                           // ~64 B

    // Mutable state for updates:
    std::vector<uint32_t*> cb_configs_payloads;       // pointers into config buffer cmd
    std::vector<RtaUpdate> rta_updates;               // memcpy descriptors
    std::vector<LaunchMsgData> launch_messages;       // msg copies + views
    std::vector<CQDispatchWritePackedCmd*> launch_msg_write_packed_cmd_ptrs;
    CQDispatchGoSignalMcastCmd* mcast_go_signal_cmd_ptr;
};
```

### Assembly Phases

The assembly proceeds in five phases within `assemble_device_commands()`:

1. **Runtime args** (line 1914): `assemble_runtime_args_commands()` generates
   `CQ_DISPATCH_CMD_WRITE_PACKED` commands for multicast and unicast RTA delivery.
   Size: $\text{sizeof(CQDispatchCmd)} + \text{align}(N_{\text{cores}} \times \text{sizeof(subcmd)}, 16) + N_{\text{cores}} \times \text{align}(N_{\text{args}} \times 4, 16)$.

2. **Config buffer** (lines 1917-1938): Semaphore init commands
   (`SemaphoreCommandGenerator`) and CB config commands (`CircularBufferCommandGenerator`)
   are sized, then assembled. The `BatchedTransferGenerator` merges adjacent transfers
   separated by at most `l1_alignment` bytes.

3. **Binary** (lines 1941-1965): `ProgramBinaryCommandGenerator` computes write commands
   to transfer kernel ELFs from DRAM (or inline from hugepage) to worker L1.

4. **Launch message** (lines 1968-1977): `LaunchMessageGenerator` builds packed-write
   commands that multicast `launch_msg_t` structures to each kernel group's core range.

5. **Go signal** (lines 1979-1995): `GoSignalGenerator` builds the multicast go signal
   that writes `go_msg_t` with `signal = RUN_MSG_GO` to each worker's
   `mailboxes->go_messages[]`.

### Total ProgramCommandSequence Size

| Component | Min | Typical | Max |
|-----------|-----|---------|-----|
| Preamble | 48 B | 48 B | 48 B |
| Stall | 64 B | 64 B | 64 B |
| Runtime Args | 128 B | 8 KB | 64 KB |
| Config Buffer | 256 B | 2 KB | 8 KB |
| Binary Refs | 64 B | 128 B | 512 B |
| Launch Messages | 256 B | 1 KB | 4 KB |
| Go Signal | 64 B | 64 B | 64 B |
| **Total** | **880 B** | **~12 KB** | **~77 KB** |

---

## 4.1.8 The Caching Layer

A critical optimization: once assembled, the `ProgramCommandSequence` is cached in
`ProgramImpl::cached_program_command_sequences_` keyed by the active
`SubDeviceManagerId` value (program.cpp:1373-1388). On the first dispatch, the full
assembly path runs. On subsequent dispatches, only `update_program_dispatch_commands()`
(stage 5) runs, mutating the cached sequence in-place.

| Aspect | First Dispatch (Uncached) | Subsequent Dispatches (Cached) |
|--------|---------------------------|-------------------------------|
| Command assembly | Full `assemble_device_commands()` | Skipped |
| Binary sending | Always sent | Sent only if `ProgramBinaryStatus != Committed` |
| RTA updates | Assembled from scratch | `rta_updates` copies from modified `RuntimeArgsData` |
| CB config updates | Assembled from scratch | `cb_configs_payloads` pointers updated in-place |
| Stall sequence | `UncachedStallSequenceIdx` | `CachedStallSequenceIdx` (wait only) |

The `ProgramBinaryStatus` enum tracks binary lifecycle: `NotSent`, `InFlight`, and
`Committed`.

---

## 4.1.9 Update Dispatch Commands

`update_program_dispatch_commands()` (`dispatch.cpp:2127-2299`) performs in-place
mutations on the cached `ProgramCommandSequence`:

1. **Stall count update:** Writes `dispatch_md.sync_count` into the wait command.
2. **Preamble update:** Writes kernel config base addresses and program host ID.
3. **CB config update:** Iterates `cb_configs_payloads` and overwrites addresses/sizes.
4. **RTA memcpy:** Executes `rta_updates` vector to copy cross-referenced RTAs.
5. **Launch message update:** Writes `kernel_config_base` addresses and `host_assigned_id`.
6. **Go signal update:** Sets dispatch core coordinates and wait count.

**Thread safety:** This function is called under the `lock_api_function_()` mutex
in `FDMeshCommandQueue::enqueue_mesh_workload()` (`fd_mesh_command_queue.cpp:258-259`).
All program data, command sequences, and dispatch metadata are protected by this lock.

---

## 4.1.10 Writing to the Hugepage: write_program_command_sequence()

Defined at `dispatch.cpp:2481`. The command sequence is written to the host-side
hugepage ring buffer managed by `SystemMemoryManager`. The function supports two modes:

- **One-shot mode** (line 2502): If the total command size fits within
  `max_prefetch_command_size`, all commands are written contiguously.
- **Multi-entry mode**: Each command block gets its own reserve/write/push_back cycle.

The write order to the hugepage:
1. Preamble (`CQ_DISPATCH_CMD_SET_WRITE_OFFSET`)
2. Stall (optional, if `stall_first`)
3. Runtime args (`CQ_DISPATCH_CMD_WRITE_PACKED` per group)
4. Program config buffer (CBs, semaphores, CRTAs -- batched)
5. Stall (optional, if `stall_before_program`)
6. Program binary (with optional prefetcher cache setup)
7. Launch message (`CQ_DISPATCH_CMD_WRITE_PACKED`)
8. Go signal (`CQ_DISPATCH_CMD_SEND_GO_SIGNAL`)

---

## 4.1.11 Device-Side Dispatch: Prefetcher and Dispatcher Kernels

**cq_prefetch.cpp** (`tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`):
- Runs on BRISC of a dedicated dispatch core
- Reads commands from the host hugepage via PCIe BAR
- Forwards data to `cq_dispatch` through a shared L1 ring buffer
- Manages flow control via semaphores (`page_ready`, `page_done`)
- Supports DRAM caching of program binaries (prefetcher cache)

**cq_dispatch.cpp** (`tt_metal/impl/dispatch/kernels/cq_dispatch.cpp`):
- Receives pages from the prefetcher
- Interprets `CQDispatchCmd` structures
- Executes NOC writes: multicast kernel configs, binaries, launch messages, and go
  signals to worker cores

The key command types processed by the dispatcher:
- `CQ_DISPATCH_CMD_WRITE_PACKED` -- packed multicast/unicast writes
- `CQ_DISPATCH_CMD_WRITE_PACKED_LARGE` -- varying-length writes (CB configs, binaries)
- `CQ_DISPATCH_CMD_WAIT` -- stall until workers complete
- `CQ_DISPATCH_CMD_SEND_GO_SIGNAL` -- go signal to workers
- `CQ_DISPATCH_SET_WRITE_OFFSETS` -- configure base addresses

---

## 4.1.12 Worker Firmware: BRISC Main Loop (13-Step Breakdown)

The BRISC firmware (`tt_metal/hw/firmware/src/tt-1xx/brisc.cc:351-584`) runs an
infinite loop that performs the following sequence for each kernel dispatch:

**Step 1: Poll for go signal** (line 392). Waits until `go_messages[go_message_index].signal`
equals `RUN_MSG_GO`, or the preload flag (`DISPATCH_ENABLE_FLAG_PRELOAD`) is set on
the launch message.

**Step 2: Read launch message** (line 429). Reads `launch_msg_rd_ptr` to index into the
ring buffer of `launch_msg_t` structures at `mailboxes->launch[]`.

**Step 3: Trigger NCRISC CB load** (line 438). Sets `subordinate_sync->dm1 = RUN_SYNC_MSG_LOAD`
to tell NCRISC to start loading CBs.

**Step 4: Initialize firmware config** (line 442). Calls `firmware_config_init()` which
copies kernel text from L1 to IRAM (on architectures with IRAM), and sets up CB base
addresses, RTA pointers, and semaphore pointers.

**Step 5: Invalidate instruction cache** (line 445). All five RISC cores get their icache
invalidated via `cfg_regs[RISCV_IC_INVALIDATE]`.

**Step 6: Launch TRISCs** (line 448). `run_triscs(enables)` first waits for TRISC0's
previous `init_sync_registers` to complete, then sets
`subordinate_sync->trisc0/1/2 = RUN_SYNC_MSG_GO`.

**Step 7: Configure NOC** (line 450). Sets `noc_index` and `noc_mode` from the launch
message.

**Step 8: Set up CB interfaces** (lines 493-504). Initializes local and remote circular
buffer read/write interfaces from the config data in L1.

**Step 9: Launch NCRISC** (line 506). On non-Wormhole architectures:
`subordinate_sync->dm1 = RUN_SYNC_MSG_GO`. On Wormhole, NCRISC must be halted and
reset to its IRAM address (lines 298-315).

**Step 10: Run BRISC kernel** (line 509). Calls the kernel as a function pointer:
`reinterpret_cast<uint32_t (*)()>(kernel_lma)()`.

**Step 11: Wait for subordinates** (line 534). `wait_ncrisc_trisc()` polls
`subordinate_sync->all == RUN_SYNC_MSG_ALL_SUBORDINATES_DONE`.

**Step 12: Notify dispatcher** (line 577). `notify_dispatch_core_done()` sends a
NOC write to the dispatch core's stream register, incrementing its worker-done counter.

**Step 13: Advance read pointer** (line 578).
`launch_msg_rd_ptr = (launch_msg_rd_ptr + 1) & (launch_msg_buffer_num_entries - 1)`.

---

## 4.1.13 Fast Dispatch vs. Slow Dispatch

The `kernel_config_msg_t.mode` field (`dev_msgs.h:143`) distinguishes:
- `DISPATCH_MODE_DEV` (fast dispatch): Commands flow through prefetcher/dispatcher.
  Worker completion is signaled via NOC writes to dispatch core stream registers.
- `DISPATCH_MODE_HOST` (slow dispatch): The host directly writes to worker L1 and
  polls for completion. No dispatch kernels are used.

Fast dispatch is the default and production path. Slow dispatch is used for debugging,
testing, and single-core replay scenarios where direct MMIO control is preferred.

---

## 4.1.14 End-to-End Timing Model

| Stage | Typical Duration | Capturable? |
|-------|-----------------|-------------|
| Finalize offsets | <1 us | Yes, immutable after |
| Populate dispatch data | 10--100 us (first run) | Yes, immutable after |
| Assemble device commands | 5--50 us (first run) | Yes, before update |
| Reserve config buffer | <1 us | Yes, output is small |
| Update dispatch commands | 1--10 us | **Primary capture point** |
| Write to issue queue | 1--5 us | Too late (data in hugepage) |
| Fetch queue doorbell | <1 us | Not relevant |

Total dispatch latency: 20--170 us for a first run, 3--15 us for a cached re-dispatch.
An interceptor adding 1--5 us of memcpy overhead at stage 5 represents a 7--33%
overhead on re-dispatch, measurable but acceptable for a debug tool.

---

## 4.1.15 Data Structure Summary for Capture

| Structure | Location | Size | Serialization |
|-----------|---------|------|--------------|
| `ProgramConfig` | `program_impl.hpp:97-108` | 40 B | Trivial (POD) |
| `ProgramTransferInfo` | `program_device_map.hpp:44-49` | 16--256 KB | Easy (vectors of primitives) |
| `ProgramCommandSequence` | `program_command_sequence.hpp:23-81` | 1--77 KB | Medium (pointer fixups) |
| `KernelGroup` | `program_impl.hpp:69-94` | ~512 B per group | Medium (`launch_msg_t` views) |
| `launch_msg_t` | `dev_msgs.h:191-193` | ~112 B (BH) | Easy (POD, packed) |
| `go_msg_t` | `dev_msgs.h:179-189` | 4 B | Trivial (POD) |
| `ProgramDispatchMetadata` | `dispatch.hpp:44-55` | ~32 B | Trivial (POD) |

**Estimated total capture size per dispatch:** For a typical matmul on Wormhole:
- Binaries: ~64 KB (first dispatch only; can be deduplicated via FNV1a hash)
- Command sequence: ~12 KB
- Metadata: ~1 KB
- Total: ~77 KB first dispatch, ~13 KB on re-dispatch

At 10,000 dispatches/second, this produces ~130 MB/s of capture data on re-dispatch,
well within NVMe write bandwidth.

---

## 4.1.16 Key Source File Index

| File | Path | Primary Role |
|------|------|-------------|
| program_impl.hpp | `tt_metal/impl/program/program_impl.hpp` | ProgramImpl class definition |
| program.cpp | `tt_metal/impl/program/program.cpp` | compile(), finalize_offsets(), populate_dispatch_data() |
| dispatch.hpp | `tt_metal/impl/program/dispatch.hpp` | assemble_device_commands() declaration, finalize helpers |
| dispatch.cpp | `tt_metal/impl/program/dispatch.cpp` | assemble_device_commands() implementation, write_program_command_sequence() |
| program_command_sequence.hpp | `tt_metal/impl/program/program_command_sequence.hpp` | ProgramCommandSequence struct |
| dev_msgs.h | `tt_metal/hw/inc/hostdev/dev_msgs.h` | launch_msg_t, go_msg_t, kernel_config_msg_t, RUN_MSG/SYNC constants |
| build.hpp | `tt_metal/jit_build/build.hpp` | JitBuildEnv, JitBuildState |
| build.cpp | `tt_metal/jit_build/build.cpp` | jit_build_once(), FNV1a build key |
| genfiles.cpp | `tt_metal/jit_build/genfiles.cpp` | jit_build_genfiles_descriptors() |
| stable_hash.hpp | `tt_metal/common/stable_hash.hpp` | FNV1a hash class |
| brisc.cc | `tt_metal/hw/firmware/src/tt-1xx/brisc.cc` | BRISC firmware main loop |
| ncrisck.cc | `tt_metal/hw/firmware/src/tt-1xx/ncrisck.cc` | NCRISC kernel entry point |
| trisck.cc | `tt_metal/hw/firmware/src/tt-1xx/trisck.cc` | TRISC kernel entry point |
| cq_prefetch.cpp | `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp` | Prefetcher dispatch kernel |
| cq_dispatch.cpp | `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` | Dispatcher dispatch kernel |
| fd_mesh_command_queue.cpp | `tt_metal/distributed/fd_mesh_command_queue.cpp` | enqueue_mesh_workload() orchestration |
| inspector.hpp | `tt_metal/impl/debug/inspector/inspector.hpp` | Inspector hooks for compile/dispatch events |

---

## 4.1.17 Cross-References

- **Ch1 (Hardware Debugger Reference)** Section 2: The capture-and-replay paradigm
  motivates identifying the optimal interception point within this pipeline.
- **Ch2 (State Inventory)** Section 1: The compiled binary state (ELFs, `ll_api::memory`
  objects) flowing through `populate_dispatch_data()` is detailed in Ch2's binary
  inventory.
- **Ch3 (LightMetal Foundation)** Section 1: The `LightMetalCaptureContext` singleton
  and its FlatBuffer command vocabulary operate at a higher abstraction level than this
  dispatch pipeline; Section 4.2 identifies where a deeper hook is needed.
- **Section 4.2**: Identifies the optimal interception points within this pipeline.
- **Section 4.3**: The `enables` bitmask and launch message from Step 6 above trigger
  the 5-RISC coordination protocol.

---

## Key Takeaways

1. **The `ProgramCommandSequence` is the single most important data structure for
   capture.** It contains all dispatch commands in their final structured form and is
   the natural serialization boundary between command assembly and hugepage write.

2. **FNV1a hashing provides natural binary deduplication.** The build cache keyed by
   `KernelCompileHash` means the interceptor only needs to capture each unique kernel
   configuration once, regardless of how many times it is dispatched.

3. **The caching layer bifurcates the capture strategy.** On first dispatch, the full
   `ProgramCommandSequence` must be captured. On subsequent dispatches, only the
   mutations applied by `update_program_dispatch_commands()` (RTA values, config buffer
   addresses, launch message pointers) need recording.

4. **The generated files pipeline (`genfiles.cpp`) is the bridge from user kernels to
   LLK.** The `chlkc_descriptors.h` header encodes data format arrays and tile
   dimensions that control LLK behavior; capturing these is essential for replay
   fidelity.

5. **Estimated capture overhead of ~2 us per dispatch is acceptable.** At ~13 KB per
   re-dispatch capture, the memcpy cost is dominated by the existing dispatch latency
   and well within storage bandwidth limits.
