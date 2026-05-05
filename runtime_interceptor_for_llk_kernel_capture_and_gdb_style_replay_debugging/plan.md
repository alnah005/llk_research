# Final Plan: Runtime Interceptor for LLK Kernel Capture and GDB-Style Replay Debugging

---

## Selection Rationale

This hybrid plan draws from all five candidate plans, guided by the evaluations' consensus findings:

**From Plan v5 (state-first approach):** The state inventory is placed early (Chapter 2) because every subsequent chapter -- LightMetal gap analysis, interceptor hook design, serialization format, replay architecture -- is an exercise in "how do we handle this known set of state." All five evaluators noted that Plans placing the state inventory before the debug tool survey produced more rigorous gap analyses. v5's codebase specificity (exact struct hierarchies, field limits like `max_runtime_args = 341`, line number ranges) is adopted as the standard throughout.

**From Plan v1 (breadth and roadmap):** The phased implementation roadmap with person-week effort estimates (Chapter 8) comes from v1, which all evaluators praised as the most actionable for engineering planning. v1's comprehensive coverage of Tensix hardware debug capabilities (breakpoint registers, debug array access for SRCA/SRCB/DEST, Quasar debug module) is incorporated into Chapter 5, though trimmed to avoid the 30-file bloat that evaluators criticized.

**From Plan v2 (hardware debug register detail):** The `ckernel_riscv_debug.h` analysis (rvdbg_cmd enumeration, hardware watchpoints HW_WCHPT0-7, RISC_DBG_CNTL_0/1 protocol, risc_sel targeting) and the coprocessor debug bus coverage (`dbg_bus_cntl_t`, per-thread config register dumps, HW_CFG_SIZE=187) are adopted from v2. Multiple evaluators identified this as uniquely valuable content that only v2 surfaced at sufficient depth. The concrete GDB RSP-to-rvdbg_cmd mapping is also included.

**From Plan v4 (ordering principle):** Hardware feasibility assessment (Chapter 5) and emulation evaluation (Chapter 5) are placed BEFORE the interceptor design (Chapter 6), following v4's rationale that the interceptor's capture scope should be informed by what the replay targets can actually use. v4's evaluator noted this prevents designing capture for state that no replay environment can consume. v4's "PROPOSED:" prefix convention for distinguishing proposed designs from existing infrastructure is adopted.

**From Plan v3 (LightMetal depth and host functional model):** The dedicated LightMetal analysis (Chapter 3) draws from v3's thorough treatment including FlatBuffer schema details, object-to-global-id mapping, and the concrete `extending_lightmetal_for_kernel_capture.md` approach. v3's unique "host functional model" replay target (compiling kernels with a host compiler against stubs, referencing `tt-llk/tests/helpers/src/trisc.cpp`) is included in Chapter 7 as a third replay strategy that other plans overlooked.

**Structural decisions informed by evaluator consensus:**
- Chapter 1 opens with cuda-gdb/GPU debugger fundamentals (trimmed to 2 files per v1 evaluator's suggestion), establishing the vocabulary of GDB stubs, debug APIs, and cooperative vs. preemptive halt before the reader encounters these concepts later.
- Multi-core coordination (Chapter 4) is placed BEFORE the debug infrastructure survey (Chapter 5) and interceptor design (Chapter 6), per v3's evaluator suggestion that the interceptor should be designed with multi-core requirements already understood.
- Total file count is kept to 24 files (down from v1's 30), addressing the evaluators' concern about navigation overhead and cross-file inconsistency.

---

## Audience

This guide targets **TT-Metal runtime engineers, LLK kernel developers, and debug-tooling architects** who need to understand the feasibility and design of a capture-and-replay debugging system for LLK kernels running on Tenstorrent Tensix cores.

**Prerequisites the reader should have:**

- Working knowledge of C++ (C++17), sufficient to read TT-Metal and TT-LLK source code
- Familiarity with the Tensix core architecture: the five RISC-V processors per Tensix (BRISC, NCRISC, TRISC0/1/2), L1 SRAM, circular buffers, the NOC, and hardware semaphores
- Experience with the TT-Metal host API: creating Programs, Kernels, Circular Buffers, setting runtime args, enqueuing programs
- Understanding of the distinction between host-side dispatch (command queue, device commands) and device-side execution (firmware, kernel mailboxes, NOC transactions)
- Basic familiarity with GDB concepts (breakpoints, watchpoints, single-stepping) and the RISC-V ISA at an introductory level (registers, privilege modes, the `ebreak` instruction)
- Access to a TT-Metal development environment with the full source tree

**No prior experience is assumed with:** cuda-gdb internals, RISC-V Debug Specification (v0.13), JTAG debugging protocols, FlatBuffers serialization, the LightMetal capture system, or RISC-V emulator internals. The guide teaches these where needed.

---

## Chapter List

### Chapter 1: How Hardware Debuggers Work -- cuda-gdb and the Reference Architecture

**Description:** Establishes the conceptual vocabulary and reference architecture for hardware kernel debuggers by analyzing how cuda-gdb, rocgdb, and peers attach to accelerator cores, identifying the common patterns that a Tenstorrent debugger must replicate or replace.

**Directory:** `ch1_hardware_debugger_reference/`

**Files:**

- `01_gpu_debugger_architectures.md`
  - The layered architecture of cuda-gdb: GDB frontend -> CUDA debug API (libcudbg) -> CUDA driver debug interface -> SM-level hardware debug unit
  - Core cuda-gdb operations: `cudbgAttach`, `cudbgSetBreakpoint`, `cudbgSingleStep`, `cudbgReadRegister`, `cudbgReadMemory` and how they map to SM debug hardware (warp-level halt, trap handler, debug registers)
  - The "focus" model: selecting a specific GPU, SM, warp, and lane for inspection; warp-level breakpoints and SIMT execution interaction
  - Comparative analysis: rocgdb (AQLPROFILE trap handlers, wavefront granularity, CWSR context save/restore), Intel GT Debugger (EU debug mode, System Routine injection), Arm DS (CoreSight debug + JTAG halt-mode debugging)
  - Common patterns extracted: (1) a hardware debug unit with halt/resume/step, (2) a driver-level debug API, (3) a host-side GDB integration layer, (4) cooperative or preemptive halt semantics
  - Which patterns Tenstorrent hardware provides today and which are missing

- `02_lessons_and_requirements_for_tenstorrent.md`
  - Translation of GPU debugger capabilities into Tenstorrent-specific requirements: what "attach to BRISC on core (3,4)" means, what "inspect SRCA register bank for TRISC1" requires
  - The key insight from all hardware debuggers: all require either hardware debug circuitry or an instrumented software trap mechanism
  - Definition of the "capture and replay" alternative: instead of live-attaching to running cores, serialize all state needed to reproduce a kernel invocation and replay it in a controllable environment
  - Decision matrix: hardware JTAG debugging vs. firmware-cooperative debugging vs. capture-and-replay-in-emulator, with trade-offs in fidelity, intrusiveness, and implementation effort
  - Gap analysis summary table: cuda-gdb feature vs. Tenstorrent hardware capability vs. current TT-Metal debug infrastructure

---

### Chapter 2: Complete State Inventory for Kernel Capture

**Description:** Systematically maps every piece of state that constitutes a reproducible LLK kernel invocation, organized by category with exact struct names, field inventories, and byte-size estimates, providing the requirements checklist that all subsequent chapters build against.

**Directory:** `ch2_state_inventory/`

**Files:**

- `01_compiled_binary_and_build_state.md`
  - The `Kernel` class hierarchy (`DataMovementKernel`, `ComputeKernel`, `EthernetKernel`) and the `Config` variant each carries: `DataMovementConfig` (processor, NOC, compile_args, defines), `ComputeConfig` (math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, compile_args, defines), `EthernetConfig`
  - `KernelSource` struct: FILE_PATH vs SOURCE_CODE variants; binary artifacts via `Kernel::binaries_` mapping build_key to per-processor ELF images (5 for compute, 2 for data movement)
  - Preprocessor defines that affect compilation: `ARCH_BLACKHOLE`/`ARCH_WORMHOLE_B0`, `MATH_FIDELITY`, `is_fp32_dest_acc_en`, data format defines, user-provided defines from `Kernel::defines_` (a `std::map<std::string, std::string>`)
  - Compile-time arguments: `compile_time_args_` (indexed vector) and `named_compile_time_args_` (named map), baked into the binary via `hlk_defines_generated.h`
  - JIT compilation parameters: `hlk_desc` (data formats, tile dims, HLK args), architecture defines, optimization level
  - Size estimate: typical compute kernel produces 5 ELF binaries of 4-32 KB each; all binaries + source + defines + compile args ~ 200-500 KB per kernel
  - Key source files: `tt_metal/impl/kernels/kernel.hpp`, `tt_metal/jit_build/build.hpp`, `tt_metal/jit_build/jit_build_options.hpp`

- `02_runtime_configuration_state.md`
  - Runtime arguments: `Kernel::core_to_runtime_args_` (3D vector [core_y][core_x][args], up to `max_runtime_args = 341` uint32 values per core), common runtime args (`Kernel::common_runtime_args_`), `RuntimeArgsData` wrapper with lazy update semantics
  - Circular buffer configuration: `CircularBufferConfig` with buffer indices, total_size, page sizes, data formats, tile specs; `ProgramImpl` stores all CBs and tracks per-core index usage via `per_core_cb_indices_` (bitset of NUM_CIRCULAR_BUFFERS=32)
  - The `ProgramConfig` struct: rta_offset, sem_offset, sem_size, cb_offset, cb_size, kernel_text_offset, kernel_text_size -- the L1 memory map for a single core
  - Semaphore state: `Semaphore` class stores id (0-15), initial_value, core_range_set; `NUM_SEMAPHORES = 16` per core
  - The `launch_msg_t` / `kernel_config_msg_t` structures: kernel_config_base offsets, sem_offsets, CB offsets, RTA offsets, kernel_text_offsets, enables bitmask (which processors are active), noc_id, noc_mode, host_assigned_id
  - The `go_msg_t`: dispatch_message_offset, master_x, master_y, signal (RUN_MSG_GO)
  - NOC configuration: noc_index (NOC_0 or NOC_1), noc_mode (DM_DEDICATED_NOC vs DM_DYNAMIC_NOC)
  - Size estimate: ~6 KB per core for all configuration state
  - Key source files: `tt_metal/api/tt-metalium/runtime_args_data.hpp`, `tt_metal/impl/buffers/circular_buffer.hpp`, `tt_metal/hw/inc/hostdev/dev_msgs.h`

- `03_memory_contents_and_tile_data.md`
  - Tile data in circular buffers: the actual tensor data present in CBs at kernel launch; sizes vary from 2 KB (single tile in BFP8) to 2 MB+ (large CB with 64 tiles in FP32)
  - L1 memory layout at launch: firmware region, kernel binary region, circular buffers, semaphores, runtime args, mailboxes, scratch regions
  - DRAM buffer state: `Buffer` objects with BufferType (DRAM/L1/SYSTEM_MEMORY), page_size, size, shard configuration; for data movement kernels, the relevant DRAM regions must be captured
  - Strategy for partial capture: capture only L1 contents at kernel launch for compute-only debugging; capture DRAM backing buffers for data movement debugging
  - Total per-core snapshot size estimate: 200 KB - 2 MB for single-core compute (binaries + args + CB configs + tile data); full L1 dump is up to 1.5 MB per core
  - For multi-core coordinated ops: 10-50 MB compressed for a 64-core matmul

- `04_implicit_hidden_and_coordination_state.md`
  - State that is easy to miss: firmware version and mailbox initialization, dispatch core configuration, `subordinate_sync_msg_t` map, SFPI register file preloading, FPU rounding mode, L1 data cache state (Blackhole), NOC transaction ordering state
  - Inter-RISC synchronization: `RUN_SYNC_MSG_ALL_GO = 0x80808080` pattern triggers all five RISCs simultaneously; `RUN_SYNC_MSG_ALL_SUBORDINATES_DONE` completion barrier
  - Inter-core semaphores: semaphore values across all cores participating in the program, not just the target core
  - NOC transaction dependencies: in-flight reads/writes at moment of capture (from `NocWriteEvent`/`NocReadEvent` if NOC debugging enabled)
  - Environmental state: core coordinates, device architecture, harvesting mask, device clock frequency, SoC descriptor
  - Quasar implications: `QuasarComputeKernel` with up to 16 compute processors per Tensix engine (`QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE = 4`); snapshot sizes and replay complexity scale accordingly
  - Why semaphore and NOC state is hard to capture: volatile during execution, timing-dependent, asynchronous

---

### Chapter 3: LightMetal Capture -- The Existing Foundation and Its Gaps

**Description:** Analyzes TT-Metal's existing LightMetal capture/replay system as the closest existing prototype for kernel-level capture, identifying what it captures, what it misses, and how the interceptor can extend its infrastructure.

**Directory:** `ch3_lightmetal_foundation/`

**Files:**

- `01_lightmetal_architecture_and_capture.md`
  - LightMetal's purpose: recording host API calls into a FlatBuffer binary for offline replay on real hardware
  - The singleton `LightMetalCaptureContext` (`tt_metal/impl/lightmetal/lightmetal_capture.hpp`): tracing enable/disable, FlatBufferBuilder accumulation, object-to-global-id maps for Buffer/Program/Kernel/CBHandle
  - The `LIGHT_METAL_TRACE_FUNCTION_CALL` macro and `TraceScope` re-entrancy guard: prevents recursive capture during replay; only captures at depth==1
  - The FlatBuffer command vocabulary (from `command.fbs`): `BufferCreateCommand`, `EnqueueWriteBufferCommand`, `ProgramConstructorCommand`, `CreateKernelCommand` (with `file_name`, `core_spec`, `KernelConfig`), `SetRuntimeArgsUint32Command`, `CreateCircularBufferCommand`, `EnqueueProgramCommand`, `FinishCommand`
  - Buffer data capture: `EnqueueWriteBufferCommand` stores `src: [uint32]` -- actual data payload serialized inline
  - The `LightMetalBinary` FlatBuffer root: commands vector + trace_descriptors sorted table
  - Replay flow: `LightMetalReplayImpl::run()` iterates commands, dispatches to per-command `execute()` overloads reconstructing objects from FlatBuffer and calling corresponding Metal APIs
  - Key source files: `tt_metal/impl/lightmetal/lightmetal_capture.hpp`, `tt_metal/impl/lightmetal/lightmetal_replay_impl.hpp`, `tt_metal/impl/flatbuffer/command.fbs`, `tt_metal/impl/flatbuffer/light_metal_binary.fbs`

- `02_gaps_for_kernel_level_debugging.md`
  - No binary capture: `CreateKernelCommand` stores `file_name` (a path), not the compiled ELF binaries; replay requires same source tree and recompilation
  - No L1/CB memory snapshot: circular buffer data contents (tiles in L1 at invocation time) not captured; only CB configuration (sizes, formats)
  - No semaphore runtime state: only initial values captured, not volatile state at moment of a specific kernel dispatch
  - No NOC transaction log: exact sequence of NOC reads/writes by BRISC/NCRISC not recorded
  - No single-kernel granularity: capture is per-Program, not per-individual-kernel-invocation; cannot isolate one TRISC2 invocation within a multi-kernel program
  - No hardware register state: Tensix config registers, SFPU state, destination register file contents not captured
  - Missing details: `named_compile_args` and `hlk_desc` tile metadata not fully represented in `KernelConfig`
  - Assessment: LightMetal provides approximately 40% of the capture infrastructure needed; the host API interception pattern, FlatBuffer tooling, and object ID tracking are directly reusable
  - Key source files: `tt_metal/impl/lightmetal/host_api_capture_helpers.hpp`, `tt_metal/impl/flatbuffer/program_types.fbs`

- `03_extending_lightmetal_for_kernel_capture.md`
  - PROPOSED: add a `KernelSnapshotCommand` to the FlatBuffer schema including compiled binary blobs, L1 memory region dumps, semaphore state vector, and full HLK descriptor
  - Strategy: use LightMetal as the backbone for API-level capture, add a parallel "deep capture" mode that also snapshots memory and binary state
  - Object map extensions: add maps for device L1 memory regions and semaphore state snapshots
  - Versioning: the LightMetalBinary schema already has a TODO for git hash and versioning; kernel snapshots would need architecture identifier, HAL version, and build environment hash
  - Integration with `data_collection.hpp`: the existing `RecordDispatchData`/`RecordProgramRun` infrastructure already tracks per-program dispatch statistics and could trigger captures

---

### Chapter 4: The TT-Metal Dispatch Path and Multi-Core Coordination

**Description:** Traces the complete kernel dispatch pipeline from host API to Tensix execution, identifying every interception opportunity, then explains the five-RISC-per-core coordination model and inter-core NOC dependencies that the interceptor and replay system must account for.

**Directory:** `ch4_dispatch_and_coordination/`

**Files:**

- `01_dispatch_path_end_to_end.md`
  - End-to-end overview: `CreateKernel()` -> `Program::compile()` -> `EnqueueProgram()` -> dispatch command assembly -> prefetcher/dispatcher pipeline -> BRISC/NCRISC/TRISC launch on core
  - The `ProgramImpl` class (`tt_metal/impl/program/program_impl.hpp`): holds `kernels_`, `circular_buffers_`, `semaphores_`, `program_configs_`, `cached_program_command_sequences_`
  - JIT compilation: `JitBuildEnv` setup with architecture-specific flags, `JitBuildState` compilation, compiler invocation with `riscv-tt-elf-g++`, generated files pipeline (`genfiles.cpp` producing `chlkc_unpack.cpp`, `chlkc_math.cpp`, `chlkc_pack.cpp`), build cache with FNV1a hash deduplication
  - `ProgramImpl::finalize_offsets()`: computes `ProgramConfig` for each programmable core type
  - `ProgramImpl::populate_dispatch_data()`: fills `ProgramTransferInfo` with binary data, destination base addresses, page offsets, lengths, processor IDs
  - `program_dispatch::assemble_device_commands()`: builds `ProgramCommandSequence` containing preamble, stall, runtime_args, program_config_buffer, binary, launch_msg, go_msg command sequences
  - `HWCommandQueue` submission: writes assembled commands into SystemMemoryManager hugepage ring buffer for device-side prefetcher to consume
  - Dispatch kernel pipeline on device: `cq_prefetch.cpp` reads from host hugepage, `cq_dispatch.cpp` interprets commands and writes to worker L1/DRAM
  - Fast dispatch (device-side relay) vs. slow dispatch (host writes directly to mailboxes)
  - Key source files: `tt_metal/impl/program/dispatch.hpp`, `tt_metal/impl/dispatch/hardware_command_queue.hpp`, `tt_metal/hw/inc/hostdev/dev_msgs.h`

- `02_interception_point_analysis.md`
  - **Hook A -- EnqueueProgram entry**: has access to `Program` object with all kernels, CBs, semaphores, runtime args in structured form; CB allocation may not be finalized; best for capturing "intent"
  - **Hook B -- Inside `program_dispatch::assemble_device_commands()`**: called after `ProgramImpl::finalize_offsets()`, all kernel groups finalized, `launch_msg` and `go_msg` populated; the ideal interception point because all configuration is resolved but not yet serialized to raw command stream
  - **Hook C -- After `write_program_command_sequence()`**: command sequence in hugepage; requires parsing raw CQ command format; too low-level for structured capture
  - **Auxiliary hook at `EnqueueWriteBuffer`**: captures tile data flowing to device DRAM/L1
  - **Post-compilation hook**: after `JitBuildState::build()`, capture ELF binaries, full defines map, compile args; extend `Inspector::program_kernel_compile_finished`
  - Recommendation: primary hook at site B (between `assemble_device_commands()` and hardware submission), supplemented by post-compilation and buffer write hooks
  - Evaluation of each hook: what information is available, what is already lost, ordering constraints, estimated overhead

- `03_five_risc_coordination_model.md`
  - The Tensix core model: BRISC (data movement, NOC0), NCRISC (data movement, NOC1), TRISC0 (unpack), TRISC1 (math/SFPU), TRISC2 (pack) -- five independent RISC-V cores sharing L1 SRAM and circular buffer pointers
  - The circular buffer producer/consumer protocol: BRISC writes tiles into CB, advances write pointer; TRISC0 reads from CB after unpacking
  - Semaphore-based gating: unpack signals math that tiles are ready, math signals pack that results are ready
  - The `launch_msg_t.enables` bitmask: which processors are active for this kernel group
  - The `RUN_SYNC_MSG` protocol: BRISC as master writing GO to subordinates, subordinates signaling DONE back
  - Why debugging a compute kernel (TRISC1) in isolation is often sufficient: it depends on TRISC0 having unpacked data, but if CB data is captured at the right moment, TRISC1 can be replayed with pre-populated inputs

- `04_inter_core_noc_and_replay_strategies.md`
  - NOC transactions between Tensix cores: multicast writes for data distribution, unicast reads for DRAM fetch; `noc_mode` determines which NOC each processor uses
  - Dispatch core interaction: dispatch cores write kernel config, binaries, and go signals to worker cores via NOC
  - Replay strategy 1 -- Single-core isolation: replay one Tensix with all five RISCs; NOC reads return captured data from snapshot buffer; NOC writes logged but not sent. Simplest; sufficient for most LLK debugging
  - Replay strategy 2 -- Trace-driven replay: capture NOC transaction log during execution (using `NocWriteEvent`/`NocReadEvent` from `noc_debugging.hpp`), replay exact transaction sequence. Highest capture overhead but most faithful
  - Replay strategy 3 -- Multi-Tensix emulation: replay multiple Tensix cores simultaneously, model NOC as message passing. Requires snapshots from all participating cores
  - For single-TRISC debugging: pre-populate semaphores at values allowing target TRISC to proceed without blocking (simulate other TRISCs having completed)
  - Practical scoping: single-core-single-program replay is feasible; multi-core with NOC fidelity requires a cycle-accurate simulator

---

### Chapter 5: Debug Infrastructure, Hardware Capabilities, and Emulation Feasibility

**Description:** Surveys all existing TT-Metal debug tools, investigates the hardware-level debug features on Tensix RISC-V cores (including the Quasar debug register interface and coprocessor debug bus), and evaluates RISC-V emulators as replay targets -- establishing what is already available and what is physically possible before the interceptor is designed.

**Directory:** `ch5_debug_and_hardware/`

**Files:**

- `01_existing_debug_tools.md`
  - Watcher system: `WatcherServer` (`tt_metal/impl/debug/watcher_server.hpp`) polls device mailboxes; monitors waypoints (4-byte ASCII tags per RISC), NOC sanitization, assert status, pause flags, ring buffer debug data, stack usage; `watcher_device_reader.cpp` reads and decodes
  - DPRINT system: `DPrintServer` (`tt_metal/impl/debug/dprint_server.hpp`) provides printf-like output from device; per-RISC circular print buffers; `DPRINT_MATH`, `DPRINT_PACK`, `DPRINT_UNPACK` macros; tile printing via `dprint_tile.h`
  - Assert and pause mechanisms: `ASSERT` macro writes line_num + type to mailbox then hangs; `LIGHTWEIGHT_KERNEL_ASSERTS` uses `ebreak`; `LLK_ASSERT` (`tt-llk/common/llk_assert.h`) delegates to ebreak or Metal ASSERT; `PAUSE()` macro sets pause flag and spins until host clears it
  - Inspector system: RPC-based (Cap'n Proto) program lifecycle tracking, compile events, dispatch core logging; `Inspector::program_compile_finished` callback; the `riscv_debug_info_enabled` option compiles kernels with `-g` for DWARF debug info
  - Profiler: Tracy integration for performance tracing, per-RISC timing data via `profiler_msg_t`
  - NOC logging: `noc_debugging.hpp` records `NocWriteEvent`/`NocReadEvent` for post-mortem analysis
  - Data collection: `RecordDispatchData()` tracks per-program dispatch transaction sizes
  - Debug ring buffer: `debug_ring_buf_msg_t` with 32 uint32 entries per core for post-mortem trace
  - Gap summary: all tools are polling-based (Watcher), one-way (DPRINT), post-mortem (NOC logging), or fatal (ASSERT/ebreak) -- none support interactive stepping, conditional breakpoints, or full state capture at arbitrary points
  - Key source files: `tt_metal/impl/debug/watcher_server.hpp`, `tt_metal/impl/debug/dprint_server.hpp`, `tt_metal/hw/inc/api/debug/dprint.h`, `tt_metal/hw/inc/api/debug/pause.h`, `tt-llk/common/llk_assert.h`

- `02_tensix_hardware_debug_capabilities.md`
  - Wormhole/Blackhole breakpoint infrastructure: `breakpoint_set(thread, bkpt_index, pc_valid, pc)`, `breakpoint_clear`, `breakpoint_resume_execution`, `breakpoint_status`, `breakpoint_data` from `tensix_functions.h`; command protocol via `RISCV_DEBUG_REG_BREAKPOINT_CTRL` encoding thread ID, command type, breakpoint index, and data
  - Conditional breakpoints: `breakpoint_set_condition_op` (break on specific opcode), `breakpoint_set_condition_loop` (break on loop iteration), `breakpoint_set_condition_other_thread` (break when another thread hits a breakpoint)
  - Note: `#ifndef ARCH_QUASAR` guard -- these breakpoint APIs exist for Wormhole/Blackhole but not Quasar
  - Quasar RISC-V debug register interface (`ckernel_riscv_debug.h`): `rvdbg_cmd` enumeration (PAUSE, STEP, CONTINUE, RD_REG, WR_REG, RD_MEM, WR_MEM, FLUSH, RD_CSR, WR_CSR, RD_FPREG, WR_FPREG, RD_VECREG, WR_VECREG, RD_UNIT); `rvdbg_status` union (paused, brkpt_hit, wchpt_hit, ebrk_hit flags); hardware watchpoints HW_WCHPT0 through HW_WCHPT7 (8 per RISC)
  - The RISC_DBG_CNTL_0 / RISC_DBG_CNTL_1 / RISC_DBG_STATUS_0 / RISC_DBG_STATUS_1 register protocol for accessing debug registers from a neighboring core; `risc_sel` field targets BRISC (0), TRISC0-2 (1-3), NCRISC (4)
  - Quasar Debug Module registers (`overlay_reg.h`): `dmcontrol` (haltreq, resumereq, setresethaltreq), `dmstatus` (anyhalted, allhalted, anyrunning, allrunning, impebreak)
  - Implication: Tensix RISC-V cores DO support hardware halt, single-step, register read/write, memory read/write, CSR access, and hardware watchpoints -- the primitives needed for GDB-style debugging
  - Debug array read: `dbg_dump_array_enable()`, `dbg_dump_read(thread, array_id, addr)` for SRCA (0x0), SRCB (0x1), DEST (0x2), MAX_EXP (0x3); provides read-only access to compute register file
  - Coprocessor debug bus (`ckernel_debug.h`): `dbg_bus_cntl_t` control structure, `dbg_daisy_id` for instruction issue monitoring; per-thread config register dumps (THREAD_0_CFG through THREAD_2_CFG, HW_CFG_0, HW_CFG_1 with HW_CFG_SIZE=187 registers)
  - The `ebreak` exception flow: triggers breakpoint exception (cause=3) jumping to trap handler; default handler hangs; could be replaced with debug stub saving state to L1 and waiting for host commands
  - **Risk caveat**: whether the Quasar debug register interface is accessible from the host via PCIe (versus only from on-chip) is an open question requiring hardware validation
  - Key source files: `tt_metal/hw/inc/internal/debug/tensix_functions.h`, `ckernel_riscv_debug.h`, `ckernel_debug.h`, `overlay_reg.h`

- `03_emulation_and_replay_target_assessment.md`
  - JTAG infrastructure: `JtagDevice` class (`umd/device/jtag/jtag_device.hpp`) provides J-Link-based `read32`/`write32` to arbitrary NOC addresses, `dbus_memdump`; primarily for manufacturing test, not interactive debugging; JTAG provides side-channel to read/write L1 without going through dispatch
  - RISC-V emulator options: Spike (functional ISA simulator, GDB stub built-in, supports RV32IMC), QEMU (system-mode, GDB stub built-in, faster execution, custom board definition for L1/CB address space), riscvOVPsim (Imperas reference model)
  - What works in emulation: RISC-V core execution, L1 memory access patterns, basic control flow, breakpoints, variable inspection, stack traces
  - What is lost: SFPU behavior (custom Tenstorrent vector unit), NOC transactions, circular buffer hardware semantics, multi-RISC timing, FPU data format handling (BFloat16/BFloat8, block float formats), hardware semaphore atomics
  - Pragmatic approach: emulate for control-flow debugging of math/pack/unpack logic; mock NOC and CB operations with stubs returning pre-captured data; accept "logical correctness debugging" not "timing/hardware-interaction debugging"
  - Host functional model option (from TT-LLK test infrastructure): `tt-llk/tests/helpers/src/trisc.cpp` already runs LLK kernels in a test harness with `ckernel::regfile` as host-side array; compile kernel with host compiler + stub headers + "virtual L1" library loading captured snapshot; full GDB support, fast iteration, no hardware needed
  - Concrete emulator workflow: capture -> extract TRISC ELF + L1 memory image -> launch Spike with `--isa=rv32im` and memory preloaded -> connect `riscv-tt-elf-gdb` (bundled in TT-Metal toolchain) to GDB port -> set breakpoints in LLK source, single-step, inspect tile data
  - Feasibility of GDB attach on real hardware: four options assessed -- (A) OpenOCD + JTAG with custom Tensix target config, (B) custom GDB remote stub using UMD JTAG/NOC reads, (C) software-only debug using ebreak + mailbox protocol, (D) PROPOSED debug firmware on BRISC as debug agent controlling TRISCs (inspired by AMD's on-GPU debug kernel)
  - The `TT_METAL_RISCV_DEBUG_INFO` flag and `riscv-tt-elf-gdb` binary: existing infrastructure for debug symbols; `KernelBuildOptLevel::O0` with `-g` supported
  - Key source files: `umd/device/jtag/jtag_device.hpp`, `tt-llk/tests/helpers/src/trisc.cpp`

---

### Chapter 6: Interceptor Design -- Where to Hook and What to Serialize

**Description:** Specifies the precise interceptor architecture, capture API, selective trigger mechanisms, and the FlatBuffer-based snapshot format for kernel invocation captures, building on the state inventory (Chapter 2), LightMetal foundation (Chapter 3), dispatch path hooks (Chapter 4), and feasibility constraints (Chapter 5).

**Directory:** `ch6_interceptor_design/`

**Files:**

- `01_interceptor_architecture_and_hooks.md`
  - PROPOSED: `KernelSnapshotCaptureContext` extending `LightMetalCaptureContext`, activating after `ProgramImpl::populate_dispatch_data()` and before `HWCommandQueue::enqueue_command()`
  - Primary hook point (site B from Chapter 4): inside `program_dispatch::assemble_device_commands()` after `finalize_offsets()`; all kernel groups finalized, launch_msg and go_msg populated; state is fully resolved but still in structured host-side form
  - What the interceptor captures at this hook: (1) iterates all kernel groups, (2) reads compiled binaries from `Kernel::binaries_`, (3) serializes `ProgramConfig`, CB configurations, semaphore initial values, runtime args, (4) optionally reads L1 memory regions via UMD `read_from_device()` for current tile data, (5) serializes to FlatBuffer
  - Auxiliary hooks: post-compilation (capture ELFs, defines map, compile args via Inspector integration), buffer write (intercept `EnqueueWriteBuffer` for tile data), post-execution (optional output state for validation)
  - The single-core isolation requirement: interceptor targets a specific `(program, core_coord)` pair
  - Interaction with existing Inspector hooks: `program_compile_started`, `program_kernel_compile_finished` callbacks provide natural extension points
  - Key source files: `tt_metal/impl/program/program_impl.hpp`, `tt_metal/impl/program/dispatch.cpp`, `tt_metal/impl/lightmetal/host_api_capture_helpers.hpp`

- `02_capture_triggers_and_modes.md`
  - PROPOSED: Three capture modes:
    - **Always-on lightweight**: capture only metadata (program IDs, kernel names, core assignments, compile hashes) with <1% runtime impact, ~10 KB per program execution
    - **Selective deep capture**: triggered by environment variable (`TT_METAL_KERNEL_CAPTURE=selective`, `TT_METAL_CAPTURE_PATTERN="matmul*"`), programmatic API (`program.enable_capture()`), or per-core targeting
    - **Failure-triggered capture**: automatic on Watcher assert detection (via `WatcherServer::killed_due_to_error()`), timeout, or `ebreak` trap; uses post-mortem L1 state via NOC reads
  - PROPOSED: `CAPTURE_BREAKPOINT()` macro: writes sentinel to known L1 mailbox location; Watcher polling detects sentinel and initiates capture; kernel spins on mailbox awaiting host resume signal
  - Ring buffer approach: maintain rolling buffer of last N invocation snapshots in memory; flush to disk on trigger; provides "time-travel" debugging with bounded overhead
  - Integration with `data_collection.hpp`: existing `RecordDispatchData` / `RecordKernelGroup` / `RecordProgramRun` infrastructure extended to trigger captures
  - Performance overhead analysis: metadata-only ~10 KB per execution (<1% impact); full snapshot with L1 readback ~10 us per 4 KB page, ~75 ms for full 1.5 MB L1, ~800 ms for 64-core snapshot; acceptable for debugging, not production

- `03_snapshot_schema_and_serialization.md`
  - PROPOSED FlatBuffer schema additions extending `light_metal_binary.fbs`:
    - `KernelBinaryBlob`: processor_type, elf_data (byte vector), base_address, text_size
    - `CircularBufferSnapshot`: cb_index, data_format, tile_dims, total_size, page_size, address, data_contents (byte vector)
    - `SemaphoreSnapshot`: id, address, value_at_capture
    - `L1RegionSnapshot`: start_address, size, data (byte vector)
    - `TensixCoreSnapshot`: core_coord, brisc_binary, ncrisc_binary, trisc_binaries (array of 3), cb_snapshots, semaphore_snapshots, l1_regions, runtime_args_per_processor, launch_msg
    - `KernelInvocationSnapshot` (root type for `.ttksnap` file): program_id, kernel_id, architecture (WORMHOLE_B0/BLACKHOLE/QUASAR), device_id, core_snapshots, compile_defines, compile_args, hlk_descriptor, capture_timestamp, git_hash, build_key
  - Versioning: schema version field, TT-Metal git commit hash, compiler flags, HAL architecture descriptor, SoC descriptor
  - Compression: LZ4 for fast capture (2-4x on tile data), zstd for archival; sparse L1 representation for efficient storage
  - Storage: write to `$TT_METAL_HOME/generated/kernel_captures/<timestamp>_<program_id>_<core_xy>/` with manifest JSON
  - Size estimates: single-core compute ~600 KB uncompressed (~150-200 KB compressed); 8x8 matmul (64 cores) ~38 MB uncompressed (~10-15 MB compressed)
  - Comparison to LightMetal: typical LightMetal binary for matmul test is ~100 KB; kernel snapshots 5-50x larger because of binary blobs and L1 data

---

### Chapter 7: Replay Debugger Architecture

**Description:** Designs the replay system that loads captured snapshots and enables GDB-style interactive debugging across three targets: RISC-V emulator, on-hardware via debug registers, and host functional model.

**Directory:** `ch7_replay_debugger/`

**Files:**

- `01_emulator_based_replay.md`
  - Architecture: snapshot loader (deserializes `.ttksnap` into memory model) -> emulator (Spike or QEMU, one instance per RISC-V core) -> GDB stub (one per core, each on its own port) -> GDB frontend (standard GDB or VS Code debug adapter)
  - Snapshot loading: load ELF into Spike/QEMU, map L1 snapshot to emulator memory at correct base address, initialize CB pointers and semaphores, set PC to kernel entry point
  - Intra-Tensix replay: run all 5 RISC-V cores in separate emulator threads with captured L1 snapshot as shared memory, using same semaphore synchronization protocol
  - Single-RISC replay: for TRISC1 math debugging, pre-populate semaphores at values that allow TRISC1 to proceed; input data already in CBs from snapshot
  - User workflow: (1) capture failing kernel, (2) `tt-kernel-replay --snapshot path/to/snap.ttksnap --core trisc1 --gdb-port 1234`, (3) `riscv-tt-elf-gdb kernel.elf -ex "target remote :1234"`, (4) set breakpoints in LLK functions, inspect variables, single-step through unpack/math/pack
  - ELF/DWARF symbol availability: JIT-compiled kernels with `-g` flag (via `TT_METAL_RISCV_DEBUG_INFO`); snapshot records path to debug ELF
  - What is lost: SFPU/FPU exact behavior, NOC transaction timing, compute pipeline latency, L1 bank contention, hardware-specific data format handling

- `02_on_hardware_and_host_model_replay.md`
  - On-hardware replay via debug registers:
    - PROPOSED: host-side "debug agent" writes snapshot L1 contents to target core via NOC/TLB, loads captured binary, sets breakpoint at kernel entry, triggers launch via synthetic `go_msg_t`
    - GDB RSP implementation: translate g/G (register read/write), m/M (memory read/write), c (continue), s (single-step), Z (breakpoint) packets to `rvdbg_cmd` operations (PAUSE, STEP, CONTINUE, RD_REG, WR_REG, RD_MEM, WR_MEM)
    - Advantages: perfect hardware fidelity, no emulation gaps
    - Disadvantages: requires hardware access, can only debug one core at a time, risk of device state corruption, exclusive device access needed
    - Quasar advantage: full RISC-V Debug Module (dmcontrol with haltreq/resumereq) could enable OpenOCD-based GDB attach
  - Host functional model replay:
    - Compile kernel source with host C++ compiler (`g++` or `clang++`) instead of `riscv-tt-elf-g++`; replace hardware intrinsics with software stubs
    - Precedent: `tt-llk/tests/helpers/src/trisc.cpp` already runs LLK kernels in test harness with `ckernel::regfile` as host-side array and `mailbox_t` as host pointers; `run_kernel(temp_args)` executes kernel function directly
    - PROPOSED workflow: capture -> extract kernel source + defines + args -> compile with host compiler and stub headers -> link against "virtual L1" library that loads captured L1 snapshot -> run under host GDB
    - Advantages: full GDB support, fast iteration, no hardware or emulator needed, can instrument every memory access
    - Disadvantages: SFPI/FPU behavior differs (host float vs. Tensix FPU), NOC absent, timing meaningless, any hardware register access must be stubbed
  - PROPOSED debug firmware concept (inspired by AMD's on-GPU debug kernel): dedicate BRISC as debug agent controlling TRISCs via semaphore manipulation and L1 memory inspection; analogous to Arm DS-5's CoreSight non-invasive trace

- `03_multi_core_replay_and_noc_coordination.md`
  - Multi-Tensix emulation: replay multiple Tensix cores simultaneously; model NOC transactions as message passing between emulator instances; requires capturing snapshots from all participating cores
  - NOC transaction recording: if `NocWriteEvent`/`NocReadEvent` logging was enabled during capture, events can be replayed as deterministic script feeding data at correct time
  - Without NOC logging: inter-core data movement cannot be faithfully replayed; capture must include DRAM/L1 contents at all NOC endpoints
  - Dispatch core emulation: during replay, dispatch core behavior (sending go signals, polling completion) must be emulated or scripted
  - Practical scoping: for LLK compute kernel debugging, NOC operations are typically only in BRISC/NCRISC; TRISCs operate on data already in L1 CBs, so TRISC replay does not require NOC modeling

---

### Chapter 8: Implementation Roadmap, Gap Analysis, and Open Questions

**Description:** Synthesizes all findings into a phased implementation plan with effort estimates, consolidates the gap analysis, identifies critical open questions requiring hardware validation, and proposes concrete validation experiments.

**Directory:** `ch8_roadmap/`

**Files:**

- `01_gap_analysis.md`
  - Structured gap analysis across five dimensions:
    - (1) State capture completeness: what LightMetal captures vs. what full replay needs (cross-reference Chapter 2 inventory against Chapter 3 LightMetal coverage)
    - (2) Debug hardware accessibility: what breakpoint/debug array/rvdbg APIs exist vs. what a GDB stub needs (from Chapter 5)
    - (3) Emulator availability: what off-the-shelf RISC-V emulators provide vs. what Tensix-specific behavior requires (from Chapter 5)
    - (4) Toolchain support: DWARF info quality for JIT-compiled kernels, `riscv-tt-elf-gdb` bundled in toolchain
    - (5) Multi-core coordination: what single-core replay provides vs. full-program replay requirements (from Chapter 4)
  - Top 7 gaps: (a) no L1 snapshot capability in LightMetal, (b) no GDB stub for Tensix RISC-V cores, (c) no Tensix functional emulator modeling CB/semaphore/NOC, (d) no NOC transaction history capture, (e) JIT-compiled kernel ELFs not preserved with debug symbols by default, (f) no single-kernel-invocation isolation in LightMetal, (g) DWARF info generated but no on-device tool consumes it

- `02_phased_implementation_plan.md`
  - **Phase 0 -- Foundation (2-3 person-weeks):** Enable `-g` flag by default when Inspector is active (partially done via `riscv_debug_info_enabled`); ensure all kernel ELF binaries retained in build cache with debug symbols; extend Inspector to record full compile-time state (defines, args, binary hash)
  - **Phase 1 -- Minimal Capture (4-6 person-weeks):** Extend LightMetal capture to include compiled ELF binaries, CB configs, runtime args, and semaphore values; implement FlatBuffer schema extensions for `KernelInvocationSnapshot`; add selective capture triggers (env var, programmatic API); deliverable: `.ttksnap` files describing a kernel invocation
  - **Phase 2 -- L1 Snapshot and Failure-Triggered Capture (4-6 person-weeks):** Add L1 memory dump capability (read L1 via NOC before kernel launch); implement Watcher-triggered post-mortem capture; implement ring buffer approach for low-overhead rolling capture; implement sparse L1 representation for storage efficiency
  - **Phase 3 -- Replay Engine (6-8 person-weeks):** Build minimal Tensix functional model (L1 memory, CB semantics, semaphore operations) around Spike; implement snapshot loading and GDB integration; port host functional model from TT-LLK test infrastructure; target: single-TRISC kernel replay with GDB step/inspect; deliverable: `tt-kernel-replay` CLI tool
  - **Phase 4 -- Multi-Core and Hardware Replay (4-6 person-weeks):** Extend emulator to model intra-Tensix 5-core coordination; implement hardware replay agent using debug register APIs (rvdbg_cmd for Quasar, breakpoint APIs for Wormhole/Blackhole); add NOC transaction recording for inter-Tensix debugging; deliverable: multi-RISC replay, hardware GDB stub prototype
  - **Phase 5 -- Integration and UX (ongoing):** VS Code debug adapter; integration with CI/CD for automatic failure capture and artifact upload; integration with tt-triage and Inspector; `tt-kernel-capture` CLI tool for listing/inspecting captures; documentation and developer training
  - Total estimated effort: 20-29 person-weeks for Phases 0-4; Phase 5 ongoing

- `03_open_questions_and_validation_experiments.md`
  - Can the Quasar RISC-V debug register interface (`rvdbg_cmd`) be accessed from the host via PCIe L1 writes, or only from on-chip (another RISC-V core on same Tensix)?
  - Does `ebreak` put the core into a state the debug register interface can interact with, or does it require the Debug Module's `impebreak` feature?
  - What is the latency of debug register read/write operations through PCIe, and is it fast enough for interactive single-stepping?
  - Can L1 memory be read back reliably while a kernel is running (dual-ported?), or does it require halting the core first?
  - Can the coprocessor be halted independently of the RISC-V cores, or does halting a core also stall the coprocessor pipeline?
  - How does the debug register interface interact with the kernel dispatch system -- will halting a core cause dispatch infrastructure to time out?
  - Does the JIT build cache need modification to retain debug symbols without bloating the cache?
  - How to handle non-deterministic behavior (NOC arbitration, L1 bank contention) in replay?
  - Licensing implications of embedding Spike in a commercial tool
  - How the SFPU (custom Tenstorrent vector unit) can be functionally modeled for emulation -- SFPI intrinsics may need a host-side functional model
  - The Quasar NEO cluster model (4 engines x 4 compute processors = 16 per cluster): how this scales snapshot sizes and replay complexity
  - PROPOSED validation experiments: (a) write minimal test using `riscv_dbg_wr`/`riscv_dbg_rd` from BRISC to halt and inspect TRISC0, (b) measure L1 read latency from host for various sizes via PCIe and JTAG, (c) prototype GDB stub connecting to a single halted RISC-V core, (d) capture a matmul kernel snapshot and replay TRISC1 in Spike with pre-populated CB data

---

## Conventions

### Terminology

| Term | Definition |
|------|-----------|
| Tensix core | A single processing tile on a Tenstorrent chip containing five RISC-V processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2), shared L1 SRAM, and a tile math engine (FPU, SFPU, unpackers, packers) |
| BRISC | Baby RISC -- data movement processor using NOC0 |
| NCRISC | Network-connected RISC -- data movement processor using NOC1 |
| TRISC0/1/2 | Tensor RISC processors for unpack (0), math/SFPU (1), and pack (2) |
| LLK | Low-Level Kernel library (`tt-llk` repository) providing tile-level compute primitives as C++ inline functions targeting Tensix RISC-V cores |
| CB | Circular Buffer -- a region of L1 SRAM used for tile-granular producer/consumer communication between processors |
| NOC | Network-on-Chip -- the on-die interconnect connecting Tensix cores, DRAM banks, and Ethernet ports |
| L1 | Local SRAM within each Tensix core (1 MB on Wormhole B0, 1.5 MB on Blackhole) |
| Snapshot | A self-contained serialized record of all state needed to reproduce a kernel invocation (stored as `.ttksnap` FlatBuffer file) |
| Capture envelope | The minimal complete set of data that enables bit-exact replay of a single kernel invocation |
| Interceptor | The host-side runtime hooks that capture kernel state at dispatch time without modifying kernel behavior |
| Replay target | The execution environment where a captured kernel is re-executed: emulator, hardware via debug registers, or host functional model |
| LightMetal | TT-Metal's existing capture-replay subsystem recording host API calls as FlatBuffer commands |
| Inspector | TT-Metal's RPC-based instrumentation layer logging program lifecycle events |
| Watcher | TT-Metal's polling-based device health monitor checking mailboxes for asserts, waypoints, and NOC violations |
| Launch message | The `launch_msg_t` struct written to a core's mailbox to trigger kernel execution |
| Go signal | The `go_msg_t` written by the dispatch core to initiate launch message processing |
| RSP | GDB Remote Serial Protocol -- the wire protocol by which GDB communicates with debug stubs |
| DM | RISC-V Debug Module -- the hardware block enabling external tools to halt, step, and inspect RISC-V harts |
| SFPI | Tenstorrent's custom RISC-V ISA extension for special-function unit operations |
| CQ | Command Queue -- the host-to-device command submission mechanism managed by `HWCommandQueue` |
| HAL | Hardware Abstraction Layer (`tt_metal/llrt/hal/`) providing architecture-independent device access |

### Notation

- File paths within TT-Metal are relative to the repository root: `tt_metal/impl/program/program_impl.hpp` means `<tt-metal-root>/tt_metal/impl/program/program_impl.hpp`
- File paths within TT-LLK are relative to the TT-LLK repository root: `common/llk_assert.h` means `<tt-llk-root>/common/llk_assert.h`
- C++ identifiers use their fully qualified names on first reference (e.g., `tt::tt_metal::detail::ProgramImpl`), shortened thereafter (e.g., `ProgramImpl`)
- Struct fields referenced as `StructName::field_name` (e.g., `kernel_config_msg_t::enables`)
- FlatBuffer tables referenced with schema file and table name: `command.fbs::CreateKernelCommand`
- Register names in `SCREAMING_SNAKE_CASE` (e.g., `RISCV_DEBUG_REG_BREAKPOINT_CTRL`)
- L1 addresses in hex with `0x` prefix (e.g., `0x21000`)
- Size estimates use binary units: KB = 1024 bytes, MB = 1024 KB
- Source file line references in format `filename.cpp:123`
- Environment variables in `MONOSPACE_CAPS` (e.g., `TT_METAL_WATCHER`, `TT_METAL_KERNEL_CAPTURE`)
- Estimated effort in person-weeks (pw)

### Formatting Rules

- Each `.md` file begins with a level-1 heading matching the file's topic
- Code blocks use triple backticks with language specifiers: ` ```cpp `, ` ```python `, ` ```fbs `
- Descriptions of proposed (not yet existing) infrastructure are prefixed with **PROPOSED:** to distinguish from descriptions of existing systems
- Cross-references use the format: "See Chapter N, `filename.md`"
- Diagrams use ASCII art or Mermaid syntax (` ```mermaid `)
- Tables use GitHub-Flavored Markdown pipe syntax with header separators
- Key findings, risks, and recommendations in blockquotes prefixed with `>`
- Comparison tables for multi-option analysis use columns for Criteria, Option A, Option B, etc.
- Each file ends with a "Key Takeaways" section of 2-4 bullet points

---

## Cross-Chapter Dependencies

| Dependent Chapter | References From | Nature of Dependency |
|---|---|---|
| Chapter 2 (State Inventory) | Chapter 1 (Hardware Debuggers) | Chapter 2's state inventory is scoped by the requirements established in Chapter 1 -- the reader knows what a debugger needs before seeing what state exists |
| Chapter 3 (LightMetal Foundation) | Chapter 2 (State Inventory) | Evaluating LightMetal's capture completeness requires the state inventory from Chapter 2 as a checklist of what must be captured |
| Chapter 4 (Dispatch and Coordination) | Chapter 2 (State Inventory) | The dispatch path walkthrough references the data structures inventoried in Chapter 2; interception point analysis evaluates each hook against the state checklist |
| Chapter 4 (Dispatch and Coordination) | Chapter 3 (LightMetal Foundation) | Understanding where LightMetal hooks into the dispatch path and how the interceptor extends those hooks requires both chapters |
| Chapter 5 (Debug and Hardware) | Chapter 1 (Hardware Debuggers) | Hardware debug capabilities on Tensix are evaluated against the reference architecture and requirements from Chapter 1's cuda-gdb analysis |
| Chapter 5 (Debug and Hardware) | Chapter 2 (State Inventory) | Assessing existing debug tool coverage requires the state inventory as a baseline checklist |
| Chapter 5 (Debug and Hardware) | Chapter 4 (Dispatch and Coordination) | Understanding how Watcher/DPRINT/Inspector hook into the runtime requires dispatch path knowledge from Chapter 4 |
| Chapter 6 (Interceptor Design) | Chapter 2 (State Inventory) | The capture data model is the serialized form of the state inventory from Chapter 2 |
| Chapter 6 (Interceptor Design) | Chapter 3 (LightMetal Foundation) | The interceptor extends LightMetal's FlatBuffer infrastructure and macro patterns from Chapter 3 |
| Chapter 6 (Interceptor Design) | Chapter 4 (Dispatch and Coordination) | Hook point selection directly uses the interception point analysis from Chapter 4; multi-core coordination informs capture scope |
| Chapter 6 (Interceptor Design) | Chapter 5 (Debug and Hardware) | Replay target feasibility from Chapter 5 determines what state is worth capturing (e.g., if emulation cannot model SFPU, capturing SFPU state has less value) |
| Chapter 7 (Replay Debugger) | Chapter 5 (Debug and Hardware) | Hardware replay uses debug register APIs from Chapter 5; emulator replay uses the feasibility assessment |
| Chapter 7 (Replay Debugger) | Chapter 6 (Interceptor Design) | The replay system consumes the `.ttksnap` snapshot format defined in Chapter 6 |
| Chapter 7 (Replay Debugger) | Chapter 4 (Dispatch and Coordination) | Multi-core replay strategies build on the coordination model and NOC dependency analysis from Chapter 4 |
| Chapter 8 (Roadmap) | All prior chapters | The gap analysis, phased plan, and open questions synthesize findings from every preceding chapter |
