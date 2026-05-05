# Chapter 2: Complete State Inventory for Kernel Capture

This chapter systematically maps every piece of state that constitutes a reproducible LLK kernel invocation on a Tenstorrent Tensix core. The inventory is organized into four categories -- compiled binaries and build state, runtime configuration, memory contents and tile data, and implicit/hidden coordination state -- with exact struct names, field inventories, and byte-size estimates drawn from the TT-Metal and TT-LLK codebases.

Chapter 1 established that the capture-and-replay approach requires serializing "all state needed to reproduce a kernel invocation." This chapter answers the question: *what exactly is that state?* Every subsequent chapter -- LightMetal gap analysis (Chapter 3), dispatch path interception (Chapter 4), interceptor design (Chapter 6), and replay architecture (Chapter 7) -- builds directly against this inventory as its requirements checklist.

---

## Chapter Files

1. [`01_compiled_binary_and_build_state.md`](./01_compiled_binary_and_build_state.md) -- Kernel class hierarchy, `KernelSource`, ELF binaries, preprocessor defines, compile-time args, JIT parameters, size estimates

2. [`02_runtime_configuration_state.md`](./02_runtime_configuration_state.md) -- Runtime args, CB config, `ProgramConfig`, semaphores, `launch_msg_t`/`go_msg_t`, NOC config, size estimates

3. [`03_memory_contents_and_tile_data.md`](./03_memory_contents_and_tile_data.md) -- L1 memory maps (all architectures), tile data in CBs, DRAM buffer state, capture strategies, multi-core deduplication, size estimates

4. [`04_implicit_hidden_and_coordination_state.md`](./04_implicit_hidden_and_coordination_state.md) -- Firmware state, inter-RISC sync, inter-core semaphores, NOC dependencies, environmental state, Quasar NEO cluster analysis, comprehensive capture checklist

---

## State Categories at a Glance

| Category | Primary Concern | Typical Size Per Core | Key Source Files |
|---|---|---|---|
| Compiled binaries and build state | What code runs on each RISC-V processor | 30--530 KB | `kernel.hpp`, `build.hpp`, `hlk_desc.hpp` |
| Runtime configuration | How the kernel is parameterized and launched | ~4--6 KB | `runtime_args_data.hpp`, `dev_msgs.h`, `circular_buffer_config.hpp` |
| Memory contents and tile data | What data is present in L1 and DRAM at launch | 8 KB -- 1.2 MB | `buffer.hpp`, `circular_buffer.hpp` |
| Implicit and coordination state | Hidden dependencies that affect reproducibility | 1--10 KB (metadata) | `dev_msgs.h`, `noc_debugging.hpp`, `semaphore.hpp` |
| **Total per-core snapshot** | | **~50 KB -- 1.5 MB** | |

> The dominant cost is tile data in circular buffers (Category 3). Binary blobs (Category 1) are the second largest contributor. Configuration metadata (Categories 2 and 4) is comparatively small but critical for correctness.

---

## Capture Envelope Size Summary

| Scenario | Uncompressed | LZ4 Compressed | Notes |
|---|---|---|---|
| Single-TRISC compute (minimal) | ~20 KB | ~6 KB | 1 TRISC ELF + config + 2 CB tiles |
| Single-TRISC compute (typical) | ~50 KB | ~15 KB | 1 TRISC ELF + config + 8 CB tiles |
| All 5 RISCs (debug ELFs, large CBs) | ~560 KB | ~150 KB | Full single-core, BF16 |
| Full L1 dump (WH) | 1,464 KB | ~400 KB | Brute-force post-mortem |
| Full L1 dump (BH) | 1,536 KB | ~420 KB | Brute-force post-mortem |
| Full L1 dump (Quasar) | 4,096 KB | ~1,100 KB | Brute-force post-mortem |
| 64-core matmul (deduplicated, BF16) | ~2--8 MB | ~600 KB -- 2.5 MB | Shared ELFs + per-core CB data |
| 64-core matmul (full L1 dump) | ~91 MB | ~25 MB | Maximum fidelity |

For reference, a typical LightMetal capture for a matmul test is 50--200 KB. Kernel snapshots are 5--50x larger because they include compiled ELF binaries and L1 tile data contents that LightMetal does not capture.

---

## How to Use This Inventory

- **As a capture checklist**: when implementing the interceptor (Chapter 6), verify that every row in the inventory tables is captured or explicitly excluded with justification.
- **As a gap analysis baseline**: when evaluating LightMetal's existing capture (Chapter 3), compare each state item against what the `CreateKernelCommand` and related FlatBuffer types record.
- **As a replay requirements spec**: when designing the replay environment (Chapter 7), verify that each state item can be loaded into the target (emulator, host model, or hardware).
- **As a size budget**: the byte-size estimates inform compression strategy, storage format, and capture overhead analysis.

---

**Next:** [`01_compiled_binary_and_build_state.md`](./01_compiled_binary_and_build_state.md)
