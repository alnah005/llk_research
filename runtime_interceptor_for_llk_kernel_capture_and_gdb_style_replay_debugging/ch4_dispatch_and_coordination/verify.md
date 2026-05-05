# Chapter 4 Verification Report

## Verification Methodology

Every factual claim in sections 01--04 and the index was checked against the actual
source code in `/localdev/salnahari/testing_dir/tt-metal/` and
`/localdev/salnahari/testing_dir/tt-llk/`. Verification included: file path existence,
line number spot-checks (50+ citations), struct/function/macro name verification,
constant values, behavioral claims, and cross-references to Ch1--Ch3.

---

## Issues Found

```
ISSUE 1: CRITICAL
File: 03_five_risc_coordination_model.md, Section: 4.3.3
Claim: "| MATH0 (TRISC0/1/2) | 1 | bit 1 (0x02) |"
       "| DM1 (NCRISC) | 4 | bit 4 (0x10) |"
Actual: In core_config.h (both Wormhole and Blackhole), the TensixProcessorTypes
  enum is: DM0=0, DM1=1, MATH0=2, MATH1=3, MATH2=4.
  Therefore:
    DM1 (NCRISC) = enum value 1, bit position 1 (0x02)
    MATH0 (TRISC0/1/2) = enum value 2, bit position 2 (0x04)
  The chapter has DM1 and MATH0 swapped in their enum values and bit positions.
Fix: Correct the table to:
  | DM0 (BRISC)       | 0 | bit 0 (0x01) |
  | DM1 (NCRISC)      | 1 | bit 1 (0x02) |
  | MATH0 (TRISC0/1/2)| 2 | bit 2 (0x04) |
```

```
ISSUE 2: CRITICAL
File: 03_five_risc_coordination_model.md, Section: 4.3.3
Claim: "| Matmul: reader + writer + compute | 0x13 |"
       "| Data movement only | 0x11 |"
       "| Compute only | 0x02 |"
Actual: With the correct enum values (DM0=0, DM1=1, MATH0=2):
  Matmul (DM0+DM1+MATH0) = (1<<0)|(1<<1)|(1<<2) = 0x07
  Data movement only (DM0+DM1) = (1<<0)|(1<<1) = 0x03
  Compute only (MATH0) = (1<<2) = 0x04
  All three enables values in the table are wrong.
Fix: Correct the table to:
  | Matmul: reader + writer + compute | 0x07 |
  | Data movement only                | 0x03 |
  | Compute only                      | 0x04 |
```

```
ISSUE 3: MINOR
File: 03_five_risc_coordination_model.md, Section: 4.3.6
Claim: "trisc.cc:96-105" for init_sync_registers()
Actual: The function is at trisc.cc:95-104 (lines 95-104, 0-indexed file starts
  at line 0 with the copyright header). The function signature is at line 95,
  closing brace at line 104.
Fix: Change "trisc.cc:96-105" to "trisc.cc:95-104"
```

```
ISSUE 4: MINOR
File: 01_dispatch_path_end_to_end.md, Section: 4.1.15
Claim: "launch_msg_t ... ~128 B"
Actual: The kernel_config_msg_t struct (which is the sole member of launch_msg_t)
  totals approximately 112 bytes on Wormhole/Blackhole (MaxProcessorsPerCoreType=5,
  ProgrammableCoreType::COUNT=3). On Quasar it would be larger due to
  MaxProcessorsPerCoreType=24.
Fix: Change "~128 B" to "~112 B" or "112--128 B depending on architecture"
```

```
ISSUE 5: MINOR
File: 01_dispatch_path_end_to_end.md, Section: 4.1.7
Claim: ProgramCommandSequence code snippet omits several fields present in the
  actual struct: circular_buffers_on_core_ranges, unicast_launch_msg_write_packed_cmd_ptrs,
  prefetcher_cache_used, kernel_bins_sizeB, kernel_bins_base_addr.
Actual: The struct at program_command_sequence.hpp:23-81 has these additional
  members that the chapter's quoted snippet does not show.
Fix: Add a note "(simplified; omits bookkeeping fields)" to the code block,
  or include the missing fields.
```

```
ISSUE 6: MINOR
File: 03_five_risc_coordination_model.md, Section: 4.3.1
Claim: "| TRISC0 | TensixProcessorTypes::MATH0 | trisc.cc (TRISC=0) | Unpack |"
Actual: The HAL enum for TRISC0 is technically MATH0 (value 2), TRISC1 is MATH1
  (value 3), and TRISC2 is MATH2 (value 4). The table correctly maps the name
  but should note that MATH0/1/2 correspond to TRISC0/1/2. Additionally, the table
  shows "TRISC1: (same core type)" without listing its MATH1 enum, which is fine
  but slightly imprecise.
Fix: Minor -- consider listing MATH1 and MATH2 explicitly for TRISC1 and TRISC2.
```

```
ISSUE 7: MINOR
File: 01_dispatch_path_end_to_end.md, Section: 4.1.6
Claim: "kernel_bins_transfer_info (source: program_device_map.hpp:36-42)"
Actual: The struct begins at line 35 (struct keyword) and ends at line 42.
Fix: Change "36-42" to "35-42"
```

---

## Verified Claims (Selected Highlights)

### File Paths (All Verified)
All 16+ cited source file paths exist in the tt-metal tree:
- tt_metal/impl/program/program_impl.hpp -- EXISTS
- tt_metal/impl/program/dispatch.cpp -- EXISTS
- tt_metal/impl/program/program_command_sequence.hpp -- EXISTS
- tt_metal/hw/inc/hostdev/dev_msgs.h -- EXISTS
- tt_metal/hw/firmware/src/tt-1xx/brisc.cc -- EXISTS
- tt_metal/hw/inc/internal/tt-1xx/blackhole/stream_io_map.h -- EXISTS
- tt_metal/distributed/fd_mesh_command_queue.cpp -- EXISTS
- tt_metal/jit_build/genfiles.cpp -- EXISTS
- tt_metal/impl/program/program.cpp -- EXISTS
- tt_metal/common/stable_hash.hpp -- EXISTS
- tt_metal/hw/firmware/src/tt-1xx/trisc.cc -- EXISTS
- tt_metal/hw/firmware/src/tt-1xx/ncrisc.cc -- EXISTS
- tt_metal/impl/dispatch/kernels/cq_prefetch.cpp -- EXISTS
- tt_metal/impl/dispatch/kernels/cq_dispatch.cpp -- EXISTS
- tt_metal/impl/debug/noc_debugging.hpp -- EXISTS
- tt-llk/tests/helpers/src/trisc.cpp -- EXISTS

### Line Number Spot-Checks (50+ Verified)

| Claim | Actual | Status |
|-------|--------|--------|
| ProgramImpl at program_impl.hpp:175 | Line 175 | PASS |
| KernelGroup at program_impl.hpp:69 | Line 69 | PASS |
| ProgramConfig at program_impl.hpp:97 | Lines 97-108 | PASS |
| compile() at program.cpp:1438 | Line 1438 | PASS |
| compiled_ check at program.cpp:1442 | Line 1442 | PASS |
| Inspector::program_compile_already_exists at 1443 | Line 1443 | PASS |
| KernelCompileHash at program.cpp:163 | Line 163 | PASS |
| GenerateBinaries at program.cpp:146-157 | Lines 146-157 | PASS |
| launch_build_step at program.cpp:1465-1502 | Lines 1465-1502 | PASS |
| read_binaries at program.cpp:1512-1518 | Lines 1512-1518 | PASS |
| compile() return at program.cpp:1527 | Line 1527 | PASS |
| populate_dispatch_data at program.cpp:1105-1262 | Lines 1105-1262 | PASS |
| finalize_offsets at program.cpp:1702 | Line 1702 | PASS |
| set_program_attrs_across_core_types at program.cpp:1692 | Line 1692 | PASS |
| cached sequences at program.cpp:1373-1388 | Lines 1373+ | PASS |
| assemble_device_commands at dispatch.cpp:1890 | Line 1890 | PASS |
| assemble_runtime_args_commands at dispatch.cpp:1914 | Line 1914 | PASS |
| config buffer at dispatch.cpp:1917-1938 | Lines 1917-1938 | PASS |
| binary at dispatch.cpp:1941-1965 | Lines 1941+ | PASS |
| launch message at dispatch.cpp:1968-1977 | Lines 1968-1977 | PASS |
| go signal at dispatch.cpp:1979-1995 | Lines 1979-1995 | PASS |
| assemble_device_commands end at dispatch.cpp:2037 | Line 2037 | PASS |
| update_program_dispatch_commands at dispatch.cpp:2127-2299 | Lines 2127-2299 | PASS |
| CB config update at dispatch.cpp:2190-2213 | Lines 2190-2213 | PASS |
| RTA aliasing at dispatch.cpp:556-559 | Lines 556-560 | PASS |
| write_program_command_sequence at dispatch.cpp:2481 | Line 2481 | PASS |
| one-shot mode at dispatch.cpp:2502 | Line 2502 | PASS |
| finalize_rt_args at dispatch.cpp:262 | Line 262 | PASS |
| finalize_sems at dispatch.cpp:280 | Line 280 | PASS |
| finalize_cbs at dispatch.cpp:299 | Line 299 | PASS |
| finalize_kernel_bins at dispatch.cpp:335 | Line 335 | PASS |
| enqueue_mesh_workload at fd_mesh_command_queue.cpp:258 | Line 258 | PASS |
| lock_api_function_() at fd_mesh_command_queue.cpp:259 | Line 259 | PASS |
| subordinate_sync at brisc.cc:48-49 | Lines 48-49 | PASS |
| run_triscs at brisc.cc:274-285 | Lines 274-285 | PASS |
| NCRISC IRAM trampoline at brisc.cc:298-315 | Lines 298-315 | PASS |
| trigger_sync_register_init at brisc.cc:331 | Line 331 | PASS |
| BRISC main loop at brisc.cc:384+ | Line 384 (while(1)) | PASS |
| Step 1 poll at brisc.cc:392 | Line 392 | PASS |
| Step 2 read at brisc.cc:429 | Line 429 | PASS |
| Step 3 DM1 LOAD at brisc.cc:438 | Line 438 | PASS |
| Step 4 firmware_config_init at brisc.cc:442 | Line 442 | PASS |
| Step 5 icache invalidate at brisc.cc:445 | Line 445 | PASS |
| Step 6 run_triscs at brisc.cc:448 | Line 448 | PASS |
| Step 7 noc_index at brisc.cc:450 | Line 450 | PASS |
| Step 10 kernel call at brisc.cc:509 | Line 509 | PASS |
| Step 11 wait_ncrisc_trisc at brisc.cc:534 | Line 534 | PASS |
| Step 12 notify_dispatch_core_done at brisc.cc:577 | Line 577 | PASS |
| Step 13 advance rd_ptr at brisc.cc:578 | Line 578 | PASS |
| ProgramCommandSequence at program_command_sequence.hpp:23-81 | Lines 23-81 | PASS |
| jit_build_once at build.cpp:785 | Line 785 | PASS |
| JitBuildState::build at build.hpp:165 | Line 165 | PASS |
| jit_build_genfiles_descriptors at genfiles.hpp:26 | Line 26 | PASS |
| Tracy include at dispatch.cpp:13 | Line 13 | PASS |
| ProgramDispatchMetadata at dispatch.hpp:44-55 | Lines 44-55 | PASS |
| ProgramTransferInfo at program_device_map.hpp:44-49 | Lines 44-49 | PASS |

### Struct/Function Names (All Verified)
- ProgramImpl, KernelGroup, ProgramConfig, ProgramBinaryStatus -- all exist at claimed locations
- ProgramCommandSequence with all major fields -- exists
- launch_msg_t, go_msg_t, kernel_config_msg_t -- all exist
- subordinate_map_t (Blackhole, Wormhole, Quasar variants) -- all exist
- FNV1a class with FNV_PRIME=0x100000001b3, FNV_OFFSET=0xcbf29ce484222325 -- confirmed
- NocWriteEvent fields match exactly
- LightMetal ID maps (buffer_id_to_global_id_map_, program_id_to_global_id_map_,
  kernel_to_global_id_map_, cb_handle_to_global_id_map_) -- all confirmed
- Generated file names: chlkc_unpack.cpp, chlkc_math.cpp, chlkc_pack.cpp,
  chlkc_isolate_sfpu.cpp, chlkc_descriptors.h -- all confirmed
- QuasarComputeKernel with compute_processors_ -- confirmed at kernel.hpp:452/502

### Constants (All Verified)
- RUN_SYNC_MSG_INIT=0x40, RUN_SYNC_MSG_GO=0x80, RUN_SYNC_MSG_LOAD=0x01,
  RUN_SYNC_MSG_WAITING_FOR_RESET=0x02, RUN_SYNC_MSG_INIT_SYNC_REGISTERS=0x03,
  RUN_SYNC_MSG_DONE=0x00 -- all at dev_msgs.h:91-101
- launch_msg_buffer_num_entries=8 -- at dev_msgs.h:380
- DM_DEDICATED_NOC=0, DM_DYNAMIC_NOC=1 -- at dev_msgs.h:119-123
- DISPATCH_MODE_DEV, DISPATCH_MODE_HOST -- at dev_msgs.h:109-112
- DISPATCH_ENABLE_FLAG_PRELOAD = 1 << 7 -- at dev_msgs.h:133

### Key Technical Finding: CB Sync Uses Stream Scratch Registers (VERIFIED)
The chapter's central claim that `tiles_received` and `tiles_acked` use **stream
scratch registers**, not L1 memory, is CONFIRMED by:
- stream_io_map.h (Blackhole, lines 24-31):
  `get_cb_tiles_received_ptr()` returns `STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX)`
  `get_cb_tiles_acked_ptr()` returns `STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX)`
- Comment at line 21-22: "Pointers to stream scratch registers (implemented using
  don't-care functional registers) that are used for CB synchronization"
- trisc.cc init_sync_registers() (line 95) zeroes these register-mapped pointers
- The LocalCBInterface struct in circular_buffer_interface.h has tiles_acked/tiles_received
  fields for local bookkeeping, but the actual cross-RISC synchronization uses the
  hardware stream registers, not L1 copies.

### Behavioral Claims (Verified)
- BRISC is the master orchestrator that launches subordinates -- confirmed in brisc.cc
- The 8-phase execution cycle description matches the firmware code flow
- CB operations (push_back, wait_front, pop_front, reserve_back) exist at claimed locations
- UCK_CHLKC_MATH exclusion of CB interface for TRISC1 -- confirmed at trisc.cc:18-22, 77-80
- Wormhole NCRISC IRAM trampoline with halt+reset -- confirmed at brisc.cc:298-315
- Quasar 24 RISCs (8 DM + 16 TRISC) -- confirmed from core_config.h enum values
- RTA retargeting into command buffer -- confirmed at dispatch.cpp:556-560

### Cross-References (All Verified)
All referenced chapters and sections exist in the output directory:
- Ch1 Section 2 -> ch1_hardware_debugger_reference/02_lessons_and_requirements_for_tenstorrent.md
- Ch2 Section 1 -> ch2_state_inventory/01_compiled_binary_and_build_state.md
- Ch2 Section 2 -> ch2_state_inventory/02_runtime_configuration_state.md
- Ch2 Section 4 -> ch2_state_inventory/04_implicit_hidden_and_coordination_state.md
- Ch3 Section 1 -> ch3_lightmetal_foundation/01_lightmetal_architecture_and_capture.md
- Ch3 Section 2 -> ch3_lightmetal_foundation/02_gaps_for_kernel_level_debugging.md

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 (both related: enables enum values and derived bitmask examples) |
| MODERATE | 0 |
| MINOR    | 5 |

## Verdict: **FAIL**

Two CRITICAL issues were found. Both involve the `TensixProcessorTypes` enum values
and the `enables` bitmask in Section 4.3.3. The chapter incorrectly states MATH0=1
and DM1=4, when the actual values are DM1=1 and MATH0=2 (on both Wormhole and
Blackhole). This propagates to all three enables example values being wrong (0x13
should be 0x07, 0x11 should be 0x03, 0x02 should be 0x04).

All other claims (50+ line number citations, struct/function names, constants,
behavioral descriptions, the key CB stream-register finding, and cross-references)
were verified as accurate.

### Required Fixes for PASS

1. **Section 4.3.3 enables table**: Fix enum values and bit positions:
   - MATH0 (TRISC0/1/2): enum value 2, bit 2 (0x04)
   - DM1 (NCRISC): enum value 1, bit 1 (0x02)

2. **Section 4.3.3 scenario table**: Fix enables values:
   - Matmul: 0x07 (not 0x13)
   - Data movement only: 0x03 (not 0x11)
   - Compute only: 0x04 (not 0x02)
