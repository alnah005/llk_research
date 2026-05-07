# Appendix A.3 -- Developer Tools and Debugging

This appendix covers the developer tooling ecosystem around TT-Blaze: L1 profiling, debug kernel instrumentation, Tracy integration, barrier utilities, Claude Code skills, and the environment variable reference.

## A.3.1 L1 Profiling

**File**: `blaze/l1_profile.py`

The L1 profiler reports per-CB metadata (type, L1 address, data format, page size, tile geometry) and per-kernel metadata (RISC role, source file, core ranges) from compiled programs.

### Activation

Set `BLAZE_L1_PROFILE` to any truthy value. Three execution entry points check this variable:

```python
# blaze/compiler.py — CompiledProgram.run() (same pattern in MeshCompiledProgram.run() and BlazeProgram.run())
if os.environ.get("BLAZE_L1_PROFILE"):
    from blaze.l1_profile import print_cb_stats
    print_cb_stats(self)
```

Profiling runs at every `.run()` call, before `ttnn.generic_op()` dispatch.

### `print_cb_stats(program)`

Accepts any of: `ttnn.ProgramDescriptor`, `ttnn.MeshProgramDescriptor`, Blaze compiled programs (`CompiledProgram`, `MeshCompiledProgram`), or `BlazeProgram` (which exposes `_cb_descriptors` directly).

For each CB descriptor, reports:

| Field | Description |
|---|---|
| `type` | `tensor_backed` (has backing buffer), `aliased` (multiple format descriptors sharing one buffer), or `scratch` (no backing tensor) |
| `buffer_index` | CB slot ID(s) |
| `total_size_B` | Total bytes |
| `l1_addr` | L1 address -- hex, with base+offset decomposition for tensor-backed CBs |
| `dtype` | Data format per format descriptor |
| `page_size_B` | Page size per format descriptor |
| `tile` | Tile geometry (e.g., `32x32`, `32x32T` for transposed) |
| `core_ranges` | Compressed core range set |

Example output:
```
cb_stats: 12 CB descriptor(s)
  cb[0] type=tensor_backed buffer_index=0 total_size_B=65536 l1_addr=0x1e000
    core_ranges={[(x=0,y=0) - (x=7,y=7)]}
  cb[1] type=scratch buffer_index=3 total_size_B=2048 l1_addr=metal-managed
    core_ranges={[(x=0,y=0) - (x=7,y=7)]}
```

### `print_kernel_stats(program)`

Reports per-kernel: RISC role (`brisc`, `ncrisc`, `trisc`), source file stem, and core ranges.

### Logging

All output goes to both `stdout` and a timestamped log file at:
```
$TT_METAL_CACHE/logs/blaze_l1_profiler_log_YYYY-mm-dd_HHMMSS.log
```

Falls back to `~/.cache/tt-metal-logs/` if `TT_METAL_CACHE` is unset. The log file is opened lazily on first output and closed at process exit via `atexit.register()`.

### CB Address Capture

The docstring in `l1_profile.py` documents two distinct address-capture mechanisms:

- **Blaze path**: CB addresses come from Metal's L1 allocator at runtime. Tensor-backed CBs use `buffer->address() + address_offset`. Scratch CBs get addresses assigned by Metal during dispatch, captured via the `allocated_address` field on `CBDescriptor`.

- **Blitz path**: Addresses come from the CbReconfig tensor -- a pre-allocated `[n_cores, 264]` UINT32 L1 tensor encoding 64 CB slots x 4 words per core. The blitz compiler pre-computes all CB addresses before dispatch. The captured state is a composite of all phases: a CB configured in phase 1 and never touched again retains its phase-1 address.

### Overlap Detection

`_build_overlap_groups()` finds CBs that share L1 bytes using union-find over (address, size) intervals. This detects temporal CbReconfig overlaps where different phases reuse the same L1 region.

## A.3.2 Debug Kernels

The `BLAZE_DEBUG_KERNELS` environment variable controls DPRINT trace statement injection into generated kernels. The full mechanism -- env var parsing, per-RISC preprocessor guards (`_RISC_GUARDS`), phase markers, and debug filename suffixing -- is documented in [Ch. 3.05 -- Kernel Codegen, Section 9](../ch03_compilation_pipeline/05_kernel_codegen.md). The one additional detail: debug-enabled codegen also inserts `#include "api/debug/dprint.h"` at the top of the generated file, guarded by the same RISC filter if one is specified.

## A.3.3 Tracy Profiling Zones

Every generated phase is wrapped in a `DeviceZoneScopedN` macro for Tracy timeline visualization. The wrapping mechanism and noop-suffix convention are documented in [Ch. 3.05 -- Kernel Codegen, Sections 5 and 8](../ch03_compilation_pipeline/05_kernel_codegen.md). Tracy must be enabled separately via `TT_METAL_DEVICE_PROFILER=1`.

## A.3.4 Barrier Utilities

### `barrier.py`

**File**: `blaze/barrier.py`

The `Barrier` class provides multi-sender, multi-receiver synchronization using global semaphores:

```python
class Barrier:
    def __init__(self, f, prefix, receiver_cores, receive_on_ncrisc=False, receive_on_brisc=False):

    def signal(self, cores: list[CoreCoord], risc: Risc) -> None:
    def wait(self, cores: list[CoreCoord], risc: Risc) -> None:
```

**Usage pattern**: multiple groups of cores signal a barrier, and another group waits for all signals to arrive.

- `signal()` emits a `BarrierSender` micro-op that increments global semaphores on all receiver cores.
- `wait()` emits a `BarrierReceiver` micro-op that spins on the semaphore until `wait_value` is reached.
- `wait_value` accumulates automatically: each `signal()` call adds `len(sender_cores)` to the expected count.

Constraints:
- TRISC cannot participate (assertion: `risc != Risc.TRISC`)
- No duplicate cores within a single signal/wait call
- Once `wait()` is called, no further `signal()` calls are allowed
- A receiver core+risc pair can only be registered once

### `injected_fabric_barrier_builder.py`

**File**: `blaze/injected_fabric_barrier_builder.py`

This module handles inter-CCL fabric barrier injection. When a FusedProgram contains multiple CCL operations, fabric barriers are automatically inserted between them to ensure cross-device synchronization.

`prepare_for_build(fp)` walks the shadow graph looking for injected barrier sender/receiver nodes, patches their CT args with the correct fabric core coordinates and semaphore addresses, and appends a final barrier receiver at the end. The key function `_compute_new_unified_ct_arg_values()` determines per-CCL fabric cores, semaphore allocation, and wait values based on the CCL ordering.

## A.3.5 Claude Code Skills

**Directory**: `.claude/skills/`

Seven skills are defined for the Claude Code AI assistant, each as a `SKILL.md` file with structured procedures:

| Skill | Purpose |
|---|---|
| `bench-kernel` | Benchmark device kernel execution time using Tracy profiler. Writes a wrapper script that sets `TT_METAL_DEVICE_PROFILER=1`, runs warmup + measured iterations, and extracts kernel duration from `ttnn.get_latest_programs_perf_data()`. Outputs markdown comparison tables with before/after speedup. |
| `check-kernel-sync` | Paper-walk audit of kernel synchronization correctness. Checks CB push/pop balance, semaphore signal/wait matching, deadlock potential (cycle detection in dependency graph), NOC address verification, and disjoint-grid CB sharing correctness. Includes a 9-step multi-chip fabric audit covering packet routing, cross-device semaphore matching, and link direction verification. Documents 21 common hang patterns. |
| `dprint` | Guide for adding DPRINT debug statements to device kernels. Covers header inclusion per RISC (`dprint.h` requires different base headers for BRISC/NCRISC vs TRISC), environment variable configuration (`TT_METAL_DPRINT_CORES`, `TT_METAL_DPRINT_RISCVS`), formatting helpers (`ENDL()`, `HEX()`, `SETW()`), and BF16 raw value inspection. |
| `fuse-l1-tensors` | Convert multiple `from_torch` calls to the fused tensor + `OverlappedView` pattern. Three patterns: Pattern A (disjoint-core weight fusion with tile-reshape), Pattern B (same-core output tensor fusion with cumulative byte offsets), Pattern C (automatic scratch CB compaction via `CbReconfig` and `scratch_mapping`). |
| `memory-audit` | Generate comprehensive L1 memory utilization reports. Traces all `from_torch` and `cb_scratch` allocations, maps core grids and overlaps, computes per-core L1 budgets against usable L1 (1,248 KB on Blackhole), and analyzes temporal sharing opportunities. |
| `port-micro-op` | Port a micro-op from tt-metal's old infrastructure to Blaze. Covers C++ kernel header translation (template params to struct member access), Python op creation, unit testing with binary hash parity, and a 5-table translation guide for CT arg types (`CB`, `Semaphore`, `PerCore`, `Flag`). |
| `triage-hang` | Triage device hangs using `tt-triage`. Critical ordering: triage first (while hung), then kill processes, then reset device. Covers `dump_callstacks`, `dump_running_operations`, `dump_lightweight_asserts`, and interpretation of stuck functions (`cb_wait_front`, `noc_semaphore_wait`, `tile_regs_acquire`). |

## A.3.6 Environment Variable Reference

### Blaze-Specific Variables

| Variable | Values | Default | See |
|---|---|---|---|
| `BLAZE_EXPORT` | `1` / unset | unset | [A.1.5](#a15-auto-export), [Ch. 3.06 S10](../ch03_compilation_pipeline/06_blaze_compiler.md) |
| `BLAZE_EXPORT_PATH` | directory path | `generated/viz` | [A.1.5](#a15-auto-export) |
| `BLAZE_L1_PROFILE` | any truthy | unset | [A.3.1](#a31-l1-profiling) |
| `BLAZE_DEBUG_KERNELS` | `0`/`1`/`all`/risc names | `0` | [A.3.2](#a32-debug-kernels), [Ch. 3.05 S9](../ch03_compilation_pipeline/05_kernel_codegen.md) |
| `BLAZE_WEIGHT_SOURCE` | `synthetic`/`state_dict:<path>`/`blitz:<path>` | `synthetic` | [A.2.4](02_weight_provider.md) |
| `BLAZE_DISABLE_TEMPORAL_REUSE` | `1` / unset | unset | [Ch. 4.04](../ch04_data_flow/04_cb_reconfig.md) |
| `BLAZE_DISABLE_TENSOR_CB_SHARE` | `1` / unset | unset | [Ch. 4.04](../ch04_data_flow/04_cb_reconfig.md) |

### tt-metal Variables (Referenced by Blaze)

| Variable | Values | Default | Description |
|---|---|---|---|
| `TT_METAL_CACHE` | directory path | unset | Root for kernel cache and L1 profiler logs |
| `TT_METAL_DEVICE_PROFILER` | `1` / unset | unset | Enable device profiling (Tracy) |
| `TT_METAL_PROFILER_MID_RUN_DUMP` | `1` / unset | unset | Enable mid-run profiler data dumps |
| `TT_METAL_PROFILER_CPP_POST_PROCESS` | `1` / unset | unset | Enable C++ post-processing of profiler data |
| `TT_METAL_DPRINT_CORES` | `"(x,y),(x,y)"` | unset | Select cores for DPRINT output (logical coordinates) |
| `TT_METAL_DPRINT_RISCVS` | `"BR,NC,TR0,TR1,TR2"` | unset | Select RISC processors for DPRINT |
| `TT_METAL_DPRINT_FILE` | file path | unset | Redirect DPRINT output to file |
| `TT_METAL_DPRINT_CHIPS` | `"0"` / `"0,1,2"` | all | Select devices for DPRINT (multi-chip) |

### RISC Processor Reference

For the full RISC model see [Ch. 5.01 -- Per-RISC Model](../ch05_risc_compilation/01_per_risc_model.md). The additional ERISC0 processor is used in Blaze for fabric operations (CCL) but is not covered in Ch. 5.

| ID | Processor |
|---|---|
| `ER0` / ERISC0 | Ethernet RISC 0 -- Fabric operations (CCL) |

## A.3.7 Debugging Workflow

A typical debugging session for a hanging kernel:

1. **Run with DPRINT** (if the hang is reproducible):
   ```bash
   rm -rf ~/.cache/tt-metal-cache/
   TT_METAL_DPRINT_CORES="(0,9),(11,9)" \
   TT_METAL_DPRINT_RISCVS="BR,NC,TR0,TR1,TR2" \
   pytest tests/blaze/my_test.py -v -s
   ```

2. **Triage the hang** (while still hung):
   ```bash
   python tt-metal/tools/triage/triage.py \
     --run=dump_callstacks --run=dump_running_operations
   ```

3. **Inspect L1 layout** (before the hang):
   ```bash
   BLAZE_L1_PROFILE=1 pytest tests/blaze/my_test.py -v -s
   ```

4. **Export visualizer data** for offline analysis:
   ```bash
   BLAZE_EXPORT=1 BLAZE_EXPORT_PATH=/tmp/viz pytest tests/blaze/my_test.py
   ```

5. **Recover the device**:
   ```bash
   pkill -9 -f pytest
   tt-smi -r
   rm -rf ~/.cache/tt-metal-cache/
   ```

6. **Run with debug kernels** to add phase markers:
   ```bash
   BLAZE_DEBUG_KERNELS=ncrisc pytest tests/blaze/my_test.py -v -s
   ```

For systematic auditing before hardware execution, run the `check-kernel-sync` skill to perform a paper-walk of CB push/pop balance, semaphore matching, and deadlock analysis. For L1 pressure issues, run the `memory-audit` skill to compute per-core budgets.
