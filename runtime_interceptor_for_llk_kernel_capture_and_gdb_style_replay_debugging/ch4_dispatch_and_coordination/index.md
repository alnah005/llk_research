# Chapter 4: The TT-Metal Dispatch Path and Multi-Core Coordination

## Purpose

This chapter analyzes how TT-Metal dispatches programs to Tenstorrent hardware,
from host-side JIT compilation and command assembly through NOC multicast delivery
to the 5-RISC coordination protocol on each Tensix core. The analysis identifies
optimal interception points for a runtime capture tool that records kernel dispatch
state and replays it in a GDB-style debugging environment.

## Key Findings

1. **Primary hook point:** Post-`update_program_dispatch_commands()` (H4) provides
   full data availability under the command queue mutex. Weighted score: 4.85/5.00.

2. **Dual-hook H1+H4 architecture recommended.** H1 captures immutable binaries
   once per compilation (~100 KB amortized); H4 captures per-dispatch deltas
   (~13 KB steady-state, ~1.5 us overhead).

3. **CB sync uses stream scratch registers, not L1.** `tiles_received`/`tiles_acked`
   are in hardware stream registers, providing atomic cross-RISC visibility.
   Host replay must replace these with shared atomics.

4. **Single-core compute replay requires ~33 KB.** Kernel binaries + CB configs +
   RTAs + input tiles, loadable via slow dispatch MMIO. No NOC stubbing needed
   because TRISCs lack NOC access.

5. **~2 us inline capture overhead is acceptable.** At 10K dispatches/s, background
   serialization produces ~130 MB/s, within NVMe bandwidth.

## Chapter Contents

### [4.1 The Dispatch Path End-to-End](01_dispatch_path_end_to_end.md)

Seven-stage pipeline from JIT compilation through hardware doorbell. ASCII data
flow diagram, FNV1a build cache, `ProgramCommandSequence` anatomy, caching layer,
BRISC 13-step main loop breakdown.

### [4.2 Interception Point Analysis](02_interception_point_analysis.md)

Six hook points (H1--H6) evaluated via 8-criterion weighted scoring matrix. H4
recommended (4.85/5.00). `RuntimeArgsData` aliasing resolution, multi-CQ/multi-device
considerations, H1+H4 dual-hook architecture.

### [4.3 The 5-RISC Coordination Model](03_five_risc_coordination_model.md)

Subordinate sync protocol, 8-phase execution cycle, CB handshake via stream scratch
registers, `KernelGroup` mapping, launch message lifecycle, architecture differences
(WH/BH/Quasar), single-core replay specification.

### [4.4 Inter-Core NOC and Replay Strategies](04_inter_core_noc_and_replay_strategies.md)

NOC transaction patterns, three replay isolation levels (compute-only / full-core /
multi-core), queue-based `NocReplayStub`, bug-type-to-strategy mapping, `.ttlk`
capture format, tiered capture system, GDB integration.

## Cross-References

- **Ch1 (Hardware Debugger Reference):** Replay strategies address Ch1's identified
  gaps in GPU debugger architectures.
- **Ch2 (State Inventory):** Section 4.1 traces compiled binary state from Ch2
  through the dispatch pipeline.
- **Ch3 (LightMetal Foundation):** Section 4.2 identifies where deeper hooks are
  needed beyond Ch3's API-level capture. The `.ttlk` format fills Ch3 Section 2's
  identified gaps.
- **Ch5 (proposed):** Sections 4.4.10/4.4.12 provide the foundation for an
  implementation guide.

## Key Source Files

| File | Path | Relevance |
|------|------|-----------|
| `program_impl.hpp` | `tt_metal/impl/program/program_impl.hpp` | `ProgramImpl`, `KernelGroup`, `ProgramConfig` |
| `dispatch.cpp` | `tt_metal/impl/program/dispatch.cpp` | `assemble_device_commands`, `update_program_dispatch_commands` |
| `program_command_sequence.hpp` | `tt_metal/impl/program/program_command_sequence.hpp` | `ProgramCommandSequence`, `LaunchMsgData` |
| `dev_msgs.h` | `tt_metal/hw/inc/hostdev/dev_msgs.h` | `launch_msg_t`, `go_msg_t`, sync constants |
| `brisc.cc` | `tt_metal/hw/firmware/src/tt-1xx/brisc.cc` | BRISC firmware, subordinate coordination |
| `stream_io_map.h` | `tt_metal/hw/inc/internal/tt-1xx/blackhole/stream_io_map.h` | CB sync register mappings |
| `fd_mesh_command_queue.cpp` | `tt_metal/distributed/fd_mesh_command_queue.cpp` | `enqueue_mesh_workload` orchestration |
| `genfiles.cpp` | `tt_metal/jit_build/genfiles.cpp` | Generated TRISC source and descriptors |

## Notation

- **PROPOSED:** Designs not yet implemented in TT-Metal.
- File paths relative to TT-Metal repository root.
- Size estimates for Wormhole/Blackhole unless noted for Quasar.
- "Typical" values assume matmul-like workload on 8x8 Tensix grid.
