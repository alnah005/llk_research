# Chapter 8 Technical Verification Report

**Verified against:** tt-metal codebase at `/localdev/salnahari/testing_dir/tt-metal/` (commit 621b949)
**Date:** 2026-05-05
**Files verified:**
- `ch8_implementation_roadmap/01_gap_analysis.md`
- `ch8_implementation_roadmap/02_phased_implementation_plan.md`
- `ch8_implementation_roadmap/03_open_questions_and_validation_experiments.md`

---

## Summary

| Metric | Count |
|--------|-------|
| Total claims verified | 20 |
| Confirmed correct | 20 |
| Errors found | 0 |
| Minor precision notes | 2 |
| Recommended fixes | 0 |

---

## Detailed Verification Results

### Claim 1: WH lacks RISC_DBG_CNTL registers entirely

**Status: CORRECT**

Searched `tt_metal/hw/inc/internal/tt-1xx/wormhole/tensix.h` for `RISC_DBG_CNTL`. No matches found. Wormhole tensix.h does not define any `RISC_DBG_CNTL` registers, neither active nor commented out.

**Referenced in:** 01_gap_analysis.md (GAP-DI-4), 03_open_questions.md (OQ-1 context)

---

### Claim 2: BH has RISC_DBG_CNTL at tensix.h:162-163 but NOT the rvdbg_cmd software API

**Status: CORRECT**

- `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h`:
  - Line 162: `#define RISCV_DEBUG_REG_RISC_DBG_CNTL_0 (RISCV_DEBUG_REGS_START_ADDR | 0x80)`
  - Line 163: `#define RISCV_DEBUG_REG_RISC_DBG_CNTL_1 (RISCV_DEBUG_REGS_START_ADDR | 0x84)`
  - Lines 164-165 additionally define STATUS_0 and STATUS_1.
- `ckernel_riscv_debug.h` (which contains `rvdbg_cmd`) exists ONLY under `tt_llk_quasar/common/inc/`, not under any blackhole path.

**Note:** The task description says "162-165" but the documents themselves correctly specify "162-163" for CNTL registers. The STATUS registers at 164-165 are separate defines. The documents are accurate.

**Referenced in:** 01_gap_analysis.md (GAP-DI-2), 03_open_questions.md (OQ-2)

---

### Claim 3: ckernel_riscv_debug.h exists ONLY under tt_llk_quasar/common/inc/

**Status: CORRECT**

`find` across the entire tt-metal tree (excluding build_Release) returns exactly one result:
`tt_metal/third_party/tt_llk/tt_llk_quasar/common/inc/ckernel_riscv_debug.h`

**Referenced in:** 01_gap_analysis.md (GAP-DI-2), 02_phased_plan.md (Phase 4)

---

### Claim 4: Quasar tensix.h has RISC_DBG_CNTL COMMENTED OUT (lines 175-178)

**Status: CORRECT**

`tt_metal/hw/inc/internal/tt-2xx/quasar/tensix.h`:
- Line 175: `// #define RISCV_DEBUG_REG_RISC_DBG_CNTL_0 (RISCV_DEBUG_REGS_START_ADDR | 0x80)`
- Line 176: `// #define RISCV_DEBUG_REG_RISC_DBG_CNTL_1 (RISCV_DEBUG_REGS_START_ADDR | 0x84)`
- Line 177: `// #define RISCV_DEBUG_REG_RISC_DBG_STATUS_0 (RISCV_DEBUG_REGS_START_ADDR | 0x88)`
- Line 178: `// #define RISCV_DEBUG_REG_RISC_DBG_STATUS_1 (RISCV_DEBUG_REGS_START_ADDR | 0x8C)`

All four lines (175-178) are commented out. The document correctly notes that Quasar accesses the debug module through `RISCV_DEBUG_REGS` memory-mapped struct instead.

**Referenced in:** 01_gap_analysis.md (GAP-DI-2, Section 8)

---

### Claim 5: Quasar HAS BREAKPOINT_CTRL at tensix.h:240

**Status: CORRECT**

`tt_metal/hw/inc/internal/tt-2xx/quasar/tensix.h` line 240:
`#define RISCV_DEBUG_REG_BREAKPOINT_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x1C0)`

**Referenced in:** 01_gap_analysis.md (GAP-DI-4, Section 8)

---

### Claim 6: dbg_dump_array_rd_cmd() is the actual function (NOT dbg_dump_read)

**Status: CORRECT**

`tt_metal/hw/inc/internal/tensix_functions.h`:
- Line 697: `dbg_dump_array_enable()`
- Line 705: `dbg_dump_array_rd_cmd(uint thread, uint array_id, uint addr)`
- Line 712: `dbg_dump_array_to_l1(uint thread, uint addr)`

No function named `dbg_dump_read` exists anywhere in the file. The document correctly names all three functions and notes the correct starting line (697).

**Referenced in:** 01_gap_analysis.md (GAP-SC-6)

---

### Claim 7: trisc.cc: kernel_lma from kernel_text_offset, function pointer call, rta_l1_base global

**Status: CORRECT**

`tt_metal/hw/firmware/src/tt-1xx/trisc.cc`:
- Line 36: `uint32_t tt_l1_ptr* rta_l1_base __attribute__((used));` -- global variable, NOT a function argument
- Line 214: `uint32_t kernel_lma = (kernel_config_base + launch_msg->kernel_config.kernel_text_offset[index]);`
- Line 215: `auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();` -- function pointer call

All three sub-claims verified exactly as documented.

**Referenced in:** 01_gap_analysis.md (GAP-SC-8)

---

### Claim 8: NocWriteEvent/NocReadEvent: int8_t coords, uint32_t addrs, NO timestamp

**Status: CORRECT**

`tt_metal/impl/debug/noc_debugging.hpp`:
- `NocWriteEvent` (lines 23-38): `int8_t src_x/src_y/dst_x/dst_y`, `uint32_t src_addr/dst_addr`, NO timestamp field
- `NocReadEvent` (lines 40-50): `int8_t` coords, `uint32_t` addresses, NO timestamp field
- Timestamps are passed as separate `uint64_t` parameters to `handle_write_event()` (line 232) and `handle_read_event()` (line 233)

The document correctly identifies that timestamps are external to the structs.

**Referenced in:** 01_gap_analysis.md (GAP-SC-5), 02_phased_plan.md (P4-T5)

---

### Claim 9: data_collector_t: CB_CONFIG=0, SEMAPHORE=1, RTARGS=2, BINARY=3

**Status: CORRECT**

`tt_metal/impl/dispatch/data_collection.hpp` lines 19-24:
```
enum data_collector_t {
    DISPATCH_DATA_CB_CONFIG,    // = 0
    DISPATCH_DATA_SEMAPHORE,    // = 1
    DISPATCH_DATA_RTARGS,       // = 2
    DISPATCH_DATA_BINARY,       // = 3
};
```

The document correctly references `data_collection.hpp:21` for SEMAPHORE (enum value 1).

**Referenced in:** 01_gap_analysis.md (GAP-SC-4)

---

### Claim 10: dispatch.cpp:1890 = assemble_device_commands

**Status: CORRECT**

`tt_metal/impl/program/dispatch.cpp` line 1890:
```cpp
void assemble_device_commands(
```

Function signature begins at exactly line 1890.

**Referenced in:** 01_gap_analysis.md (GAP-SC-1, GAP-SC-8), 02_phased_plan.md (P1-T2), 03_open_questions.md (OQ-9)

---

### Claim 11: rtoptions.hpp:204 for riscv_debug_info_enabled, default false

**Status: CORRECT**

`tt_metal/llrt/rtoptions.hpp` line 204:
`bool riscv_debug_info_enabled = false;`

**Referenced in:** 01_gap_analysis.md (GAP-DI-6), 02_phased_plan.md (P0-T1)

---

### Claim 12: Inspector::program_kernel_compile_finished at inspector.hpp:39-43

**Status: CORRECT**

`tt_metal/impl/debug/inspector/inspector.hpp` lines 39-43:
```cpp
static void program_kernel_compile_finished(
    const detail::ProgramImpl* program,
    const IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const tt::tt_metal::JitBuildOptions& build_options) noexcept;
```

**Referenced in:** 01_gap_analysis.md (GAP-SC-2), 02_phased_plan.md (P0-T3)

---

### Claim 13: WatcherServer::killed_due_to_error() at watcher_server.hpp:34

**Status: CORRECT**

`tt_metal/impl/debug/watcher_server.hpp` line 34:
`bool killed_due_to_error();`

**Referenced in:** 01_gap_analysis.md (GAP-RF-4), 02_phased_plan.md (P2-T4)

---

### Claim 14: Kernel::binaries(build_key) at kernel.hpp:197

**Status: CORRECT**

`tt_metal/impl/kernels/kernel.hpp` line 197:
`const std::vector<const ll_api::memory*>& binaries(uint64_t build_key) const;`

**Referenced in:** 01_gap_analysis.md (GAP-SC-2), 02_phased_plan.md (P0 Go/No-Go)

---

### Claim 15: trisc.cpp host model at tt-llk/tests/helpers/src/trisc.cpp

**Status: CORRECT**

File exists at: `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp`

Specific line claims verified:
- Line 58: `std::fill(ckernel::regfile, ckernel::regfile + 64, 0);` (regfile as host-side register array)
- Line 72: `run_kernel(__runtime_args_start);` (kernel execution)
- Line 76: `*mailbox = ckernel::KERNEL_COMPLETE;` (completion signaling)

**Referenced in:** 01_gap_analysis.md (GAP-DI-3), 02_phased_plan.md (P3-T2), 03_open_questions.md (OQ-5)

---

### Claim 16: tensix_functions.h dbg_dump functions guarded by #ifndef ARCH_QUASAR

**Status: CORRECT**

`tt_metal/hw/inc/internal/tensix_functions.h` line 696:
`#ifndef ARCH_QUASAR`

The `dbg_dump_array_enable()`, `dbg_dump_array_rd_cmd()`, and `dbg_dump_array_to_l1()` functions are all inside this guard, meaning they are excluded for the Quasar architecture.

**Referenced in:** 01_gap_analysis.md (GAP-DI-4, Section 8)

---

### Claim 17: t6_debug_map.h lines 220-223 for Quasar RISC_DBG_CNTL_0

**Status: CORRECT**

`tt_metal/hw/inc/internal/tt-2xx/quasar/t6_debug_map.h` lines 220-223:
```
#define T6_DEBUG_REGS__RISC_DBG_CNTL_0__RISC_DBG_CNTL_0_bm 0xffffffff
#define T6_DEBUG_REGS__RISC_DBG_CNTL_0__RISC_DBG_CNTL_0_bp 0
#define T6_DEBUG_REGS__RISC_DBG_CNTL_0__RISC_DBG_CNTL_0_bw 32
#define T6_DEBUG_REGS__RISC_DBG_CNTL_0__RISC_DBG_CNTL_0_reset 0x0
```

**Referenced in:** 01_gap_analysis.md (GAP-DI-2), 03_open_questions.md (OQ-1)

---

### Claim 18: ckernel_riscv_debug.h rvdbg_cmd enum and riscv_dbg_wr at lines 113-115

**Status: CORRECT**

`tt_llk_quasar/common/inc/ckernel_riscv_debug.h`:
- `rvdbg_cmd` enum at lines 11-31 with PAUSE, STEP, CONTINUE, RD_REG, WR_REG, RD_MEM, WR_MEM, RD_FPREG, WR_FPREG, RD_VECREG, WR_VECREG
- `rvdbg_reg` enum at lines 41-58 with HW_WCHPT0-7 (8 hardware watchpoints)
- `riscv_dbg_wr` function starting at line 109, with the actual writes at lines 113-115:
  ```
  RISCV_DEBUG_REGS->RISC_DBG_CNTL_1 = val;
  RISCV_DEBUG_REGS->RISC_DBG_CNTL_0 = 0;
  RISCV_DEBUG_REGS->RISC_DBG_CNTL_0 = cntl0_wr_cmd;
  ```

**Referenced in:** 01_gap_analysis.md (GAP-DI-1, GAP-DI-2), 03_open_questions.md (OQ-1)

---

### Claim 19: command.fbs:79 stores file_name (kernel source path, not compiled binary)

**Status: CORRECT**

`tt_metal/impl/flatbuffer/command.fbs` lines 76-82:
```
table CreateKernelCommand {
  global_id: uint32;
  program_global_id: uint32;
  file_name: string;          // Later replace with src, then binary
  core_spec: CoreSpec;
  kernel_config: KernelConfig;
}
```

Line 79 is `file_name: string;` with the comment "Later replace with src, then binary" -- confirming that LightMetal currently stores the source path, not the compiled binary.

**Referenced in:** 01_gap_analysis.md (GAP-SC-2)

---

### Claim 20: LLK_ASSERT at tt_llk_wormhole_b0/llk_lib/ and tt_llk_blackhole/llk_lib/, NOT at tt-llk/common/

**Status: CORRECT**

`find` results show `llk_assert.h` exists at:
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_assert.h`
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_assert.h`

No `llk_assert.h` exists under any `common/` directory.

**Referenced in:** 01_gap_analysis.md (GAP-DI-6)

---

## Minor Precision Notes

These are not errors but clarifications for maximum accuracy:

1. **BH tensix.h line range:** The task description key fact says "162-165" which includes both CNTL and STATUS registers. The documents themselves correctly distinguish between CNTL (162-163) and STATUS (164-165) in most references. No correction needed.

2. **data_collector_t naming:** The enum values are `DISPATCH_DATA_CB_CONFIG`, `DISPATCH_DATA_SEMAPHORE`, `DISPATCH_DATA_RTARGS`, `DISPATCH_DATA_BINARY` (with the `DISPATCH_DATA_` prefix). The documents use the shorthand "CB_CONFIG=0, SEMAPHORE=1, RTARGS=2, BINARY=3" which is clear in context. No correction needed.

---

## Conclusion

All 20 verified claims are technically accurate against the tt-metal codebase at commit 621b949. The Chapter 8 documents demonstrate thorough and precise sourcing of code references including file paths, line numbers, function names, enum values, and architectural differences across Wormhole/Blackhole/Quasar. No corrections are recommended.
