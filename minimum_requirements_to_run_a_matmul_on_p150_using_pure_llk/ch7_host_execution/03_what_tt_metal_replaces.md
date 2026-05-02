# 7.3 What TT-Metal Replaces

This section catalogs every manual operation that the LLK test infrastructure performs which TT-Metal would normally handle. The purpose is twofold: to understand the full scope of what "pure LLK" means in practice, and to clearly delineate the boundary between kernel logic (which LLK provides) and everything else (which the test harness recreates from scratch).

---

## 7.3.1 The Division of Labor

When running LLK kernels on hardware without TT-Metal, the test infrastructure must manually handle every layer between "I have kernel source code" and "I got a result tensor." Here is the complete responsibility map:

| Responsibility | TT-Metal Runtime | Pure LLK Test Infrastructure |
|----------------|-----------------|------------------------------|
| Device discovery and initialization | `tt::tt_metal::CreateDevice()` | `tt_exalens_init.init_ttexalens()` + `check_context()` |
| Power state management | Automatic | Manual `arc_msg(GO_BUSY)` / `arc_msg(GO_IDLE)` |
| Kernel compilation | `CreateKernel()` + internal build system | Custom Python build with `riscv-tt-elf-g++` (`build_elfs()`) |
| Memory allocation | `tt::tt_metal::CreateBuffer()` | Manual address arithmetic starting at `0x21000` |
| Data tilization | `tt::tt_metal::tilize()` | Custom `pack_*` functions per data format |
| Data transfer to device | `tt::tt_metal::EnqueueWriteBuffer()` | `ttexalens.write_to_device()` |
| RISC core boot and lifecycle | Dispatch firmware | Custom BRISC firmware (`brisc.cpp`) |
| Hardware register initialization | Dispatch firmware | `device_setup()` in `boot.h` |
| Semaphore initialization | Dispatch firmware | `t6_semaphore_init()` in `boot.h` |
| Runtime argument passing | `SetRuntimeArgs()` API | Manual `struct.pack()` + `write_to_device()` at `0x20000` |
| Inter-core synchronization | NOC-based message passing | Shared-memory semaphores only |
| Kernel launch | `tt::tt_metal::EnqueueProgram()` | `commit_brisc_command(START_TRISCS)` |
| Completion detection | Command queue fence | Manual mailbox polling |
| Data transfer from device | `tt::tt_metal::EnqueueReadBuffer()` | `ttexalens.read_from_device()` |
| Data untilization | `tt::tt_metal::untilize()` | Custom `untilize_block()` in Python |
| Multi-core dispatch | Automatic grid mapping | Not supported (single core only) |
| NOC data movement | DataMovement kernels | Not available |
| Circular buffer management | CB API | Not available |

---

## 7.3.2 The Session Lifecycle: Manual vs. Automatic

The following table maps each phase of the test session lifecycle to its implementation in both approaches:

| Phase | Pure LLK Manual Step | TT-Metal Equivalent |
|-------|---------------------|---------------------|
| **Init** | `tt_exalens_init.init_ttexalens()` | `tt_metal.CreateDevice()` |
| **Power** | `_send_arc_message("GO_BUSY")` | Implicit in device creation |
| **Compile** | `build_elfs()` with cross-compiler | `CreateProgram()` + JIT cache |
| **Format** | `data_formats()` inference + struct pack | `DataType` + automatic inference |
| **Tilize** | `pack_fp16()` etc. in Python | `tilize()` on device or optimized host |
| **Write data** | `write_to_device()` at manual addresses | `WriteToBuffer()` with allocator |
| **Write args** | `struct.pack()` + `write_to_device()` | `SetRuntimeArgs()` |
| **Boot BRISC** | `load_elf("brisc.elf")` + soft reset | Firmware pre-loaded at device init |
| **Boot TRISCs** | `BriscCommand` protocol + polling | `EnqueueProgram()` |
| **Poll** | `read_word_from_device()` loop on mailboxes | `Finish()` with event system |
| **Readback** | `read_from_device()` + `untilize_block()` | `ReadFromBuffer()` + auto untilize |
| **Cleanup** | `_send_arc_message("GO_IDLE")` | `CloseDevice()` |

---

## 7.3.3 Memory Management: Manual vs. Automatic

TT-Metal provides a buffer allocation API that tracks which L1 addresses are in use, manages alignment requirements, and handles the mapping between logical buffers and physical addresses. The pure LLK infrastructure replaces this with simple sequential allocation:

The `write_pipeline_operands_to_l1()` function (see Section 7.2.6 for full code) uses simple sequential allocation starting at `0x21000`, with no free list, no alignment enforcement beyond what the data format naturally provides, and no bounds checking. The starting address is chosen to be safely above the runtime parameters region (`0x20000`). For coverage builds, it shifts to `0x70000`.

The address layout for a typical Blackhole non-coverage build:

```
0x00000 - 0x03FFF   TRISC0 code (unpack.elf)
0x04000 - 0x07FFF   TRISC1 code (math.elf)
0x08000 - 0x0BFFF   TRISC2 code (pack.elf)
0x0C000 - 0x0FFFF   BRISC code (brisc.elf)
0x10000 - 0x1FFF7   TRISC data, stack, loader_init
0x1FFB8 - 0x1FFCF   Mailboxes
0x20000 - 0x200FF   Runtime arguments (RuntimeParams struct)
0x21000 - ...        Input tensors (src_a, src_b) and output buffer
```

> **Cross-reference:** [Chapter 6, Section 3](../ch6_compilation/03_linker_scripts_and_memory_regions.md) documents the L1 memory map defined by the linker scripts, showing exactly which address ranges are available for operand data.

---

## 7.3.4 The BRISC Firmware: A Minimal Dispatch Engine

The most significant piece of infrastructure that TT-Metal replaces is the entire dispatch firmware stack. TT-Metal's dispatch firmware handles command queues, program scheduling, NOC data movement, circular buffer management, and multi-core coordination. The LLK test infrastructure replaces all of this with approximately 130 lines of C++ in `brisc.cpp`.

The BRISC firmware understands exactly three commands:

| Command | What It Does |
|---------|-------------|
| `START_TRISCS` | Reset mailboxes, call `device_setup()`, release TRISC soft-reset |
| `RESET_TRISCS` | Assert TRISC soft-reset (halt all TRISCs) |
| `UPDATE_START_ADDR_CACHE_AND_START` | (Wormhole only) Cache `_start` addresses, then do `START_TRISCS` |

Compare this to TT-Metal's dispatch firmware, which implements:

- Command queue parsing (ring buffer with read/write pointers)
- Program descriptor interpretation (kernel binaries, CB configs, semaphore configs)
- NOC multicast for multi-core kernel distribution
- Circular buffer setup across the Tensix tile grid
- Runtime argument passing through CB infrastructure
- Fast dispatch with pre-fetching and pipelining
- Device-side completion signaling through command queue fences

The LLK BRISC firmware deliberately omits all of this because the test infrastructure only needs to run one kernel on one core at a time. The host handles everything else over PCIe via tt-exalens.

---

## 7.3.5 Hardware Initialization: What device_setup() Does

In TT-Metal, hardware initialization of a Tensix tile happens as part of the dispatch firmware's program launch sequence. In the LLK infrastructure, BRISC's `device_setup()` function performs five critical operations before every kernel launch: disabling destination register clock gating, clearing the accumulator (ZEROACC), enabling the SFPU condition code stack (SFPENCC), loading constant -1.0 into SFPU LREG11 (SFPCONFIG), and initializing inter-thread semaphores. See Section 7.2.10 for the full code listing and per-operation explanation, or Chapter 3, Section 3.2 for the firmware perspective.

TT-Metal performs equivalent initialization, but as part of a larger setup sequence that also configures NOC routing, circular buffers, and per-core runtime parameters.

> **Cross-reference:** [Chapter 5](../ch5_compile_time_config/) documents how the semaphores coordinate the three-phase unpack/math/pack pipeline during matmul execution.

---

## 7.3.6 Completion Detection and Error Handling

### Mailbox Polling vs. Command Queue Fences

TT-Metal uses a command queue architecture where the host enqueues programs and the device processes them in order. Completion is detected through a **fence mechanism**: the host writes a fence value after the program command, and when the device's read pointer advances past it, the host knows execution is complete. This requires only a single read of the device's read pointer.

The LLK infrastructure uses a simpler but less efficient mechanism: individual mailbox polling. Each TRISC writes `0xFF` to its dedicated mailbox word in L1, and the host polls all three words in a loop. This requires three PCIe reads per polling iteration plus the overhead of Python's polling loop. For a kernel that completes in microseconds, the polling overhead dominates execution time.

### Error Handling: EBREAK Detection and Callstack Extraction

When a kernel hits an `LLK_ASSERT`, the RISC-V core executes an `EBREAK` instruction and halts. The pure LLK path detects this by polling:

```python
def is_assert_hit(risc_name, core_loc="0,0", device_id=0):
    risc_debug = block.get_risc_debug(risc_name, neo_id=...)
    return risc_debug.is_ebreak_hit()
```

If an assertion is detected, the full callstack is extracted from the device using DWARF debug info from the ELF files:

```python
def handle_if_assert_hit(elfs, core_loc="0,0", device_id=0):
    for core in TRISC_CORES:
        if is_assert_hit(str(core.name), core_loc=core_loc):
            stack_trace = callstack(core_loc, elfs, risc_name=str(core.name))
            # Format with file/line/column info from DWARF and raise LLKAssertException
```

This is a post-mortem diagnostic -- the host discovers the failure only when the mailbox polling times out (2 seconds by default). There is no mechanism for a kernel to signal a non-fatal warning or report progress. TT-Metal provides richer error reporting through its watcher infrastructure, which continuously monitors core health.

---

## 7.3.7 The Single-Core Limitation

The most significant limitation of pure-LLK execution is the restriction to a single Tensix tile. The LLK test infrastructure operates exclusively on one NOC coordinate (typically `"0,0"`). This means:

**What works:**
- Any operation that can be expressed as a single tile or a sequence of tiles processed on one core
- Matmul of tiles that fit in L1 (one tile A, one tile B, one tile output minimum)
- Elementwise operations, reductions, SFPU operations
- Multi-tile processing via L1-to-L1 iteration

**What does not work:**
- Multi-core matmul (distributing a large matmul across a grid of Tensix tiles)
- NOC data movement (streaming tiles between cores)
- Circular buffer pipelines (producer-consumer patterns across cores)
- Any operation that exceeds single-core L1 capacity

On Blackhole, each Tensix core has 1.5 MB of L1 SRAM. After accounting for instruction memory, runtime parameters, mailboxes, and stack space, the usable data region for input and output tiles is roughly 1 MB. For Float16_b tiles (2048 bytes each), this allows approximately 500 tiles of data.

TT-Metal's runtime handles multi-core dispatch through grid mapping (assigning work to a 2D grid of Tensix tiles), NOC multicasting (broadcasting kernel binaries and data to multiple tiles), circular buffers (managing producer-consumer data flow), and DataMovement kernels (BRISC/NCRISC kernels that move data between L1 and DRAM or between tiles). None of this infrastructure exists in the LLK test harness.

---

## 7.3.8 What the LLK Infrastructure Provides That TT-Metal Does Not

While the pure-LLK approach lacks TT-Metal's runtime capabilities, it provides some unique advantages for kernel development:

| Capability | How |
|-----------|-----|
| **Direct register observation** | `get_register_store().read_register()` reads any Tensix config register |
| **Instruction injection** | `TensixDebug.inject_instruction()` executes arbitrary Tensix ops from the host |
| **Callstack extraction** | `callstack()` provides source-level debug info using DWARF data from ELFs |
| **Assert detection** | `is_assert_hit()` detects `ebreak` on any RISC core |
| **Coverage analysis** | Coverage builds compile with `-fprofile-arcs -ftest-coverage`; the host reads coverage data from L1 via `__coverage_start` symbol and processes it through `gcov-tool merge-stream` and `lcov` for standard coverage reports |
| **Per-instruction profiling** | Zone-scoped profiler markers with hardware timestamp counters |
| **Rapid iteration** | No TT-Metal build dependency; compile one kernel file and run |
| **Format sweep coverage** | The test harness can sweep every combination of input format, output format, math fidelity, and dest accumulation mode via `@parametrize` decorators |
| **Deterministic execution** | No asynchronous dispatch, no DMA, no multi-core coordination -- same inputs always produce same outputs |
| **Architecture-portable testing** | Same Python test code runs on Wormhole, Blackhole, and Quasar by switching `CHIP_ARCH` |

These capabilities make the LLK infrastructure ideal for kernel development and debugging, even though it cannot run production workloads.

---

## 7.3.9 Boot Mode Comparison

The pure-LLK harness supports three boot modes, each with distinct characteristics:

| Aspect | BRISC Mode (WH/BH default) | TRISC Mode (Quasar default) | EXALENS Mode (debugging) |
|--------|---------------------------|---------------------------|------------------------|
| Supervisor core | BRISC runs persistent loop | TRISC0 coordinates | None (host injects) |
| Kernel re-launch | BRISC command protocol | Full soft-reset cycle | Full re-injection |
| HW init by | BRISC (`device_setup()`) | TRISC0 (`device_setup()`) | Host (`inject_instruction()`) |
| TRISC release | BRISC clears reset bits | TRISC0 clears reset bits | Host clears reset bits |
| Advantage | Fast re-launch, no BRISC reload | Simpler, no BRISC ELF | Lowest-level control |

TT-Metal abstracts all three boot modes behind its dispatch infrastructure. The developer never chooses or manages boot sequencing.

---

## 7.3.10 Alternative Approaches: The Spectrum of Control

Between pure LLK and full TT-Metal, there are intermediate approaches that offer different tradeoffs:

```
More Control                                             Less Control
Pure LLK  <-->  TT-Metal Custom Kernel  <-->  TT-Metal Op  <-->  TTNN
   |                    |                         |                |
   |  Manual everything |  Manual compute         |  Config only   |  API call
   |  Single core       |  Multi-core possible    |  Multi-core    |  Multi-core
   |  No data movement  |  Framework data movement|  Optimized DM  |  Optimized DM
   |  Direct L1 access  |  Buffer API             |  Tensor API    |  Tensor API
```

- **TT-Metal with Custom Compute Kernels** -- TT-Metal allows users to write custom compute kernels that call LLK functions directly, while using TT-Metal's runtime for everything else. This provides the LLK's compute flexibility with TT-Metal's infrastructure, including multi-core execution and NOC-based data movement.

- **TT-Metal with Built-in Matmul Operations** -- Optimized matmul implementations handle tiling, blocking, data format selection, multi-core distribution, and DRAM streaming automatically.

- **TTNN (TT Neural Network)** -- The highest-level API, where a matmul is a single call: `output = ttnn.matmul(tensor_a, tensor_b)`.

Each step up the stack trades visibility and control for productivity and performance. Pure LLK sits at the bottom -- maximum visibility, minimum abstraction. The choice depends on the task: validating a new LLK function calls for pure LLK; integrating into a production operator calls for TT-Metal custom kernels; building an end-to-end model calls for TTNN.

---

## 7.3.11 Summary: The Minimum Host-Side Requirements

To run a matmul on P150 using pure LLK, the host must provide every function in the following table. This is the irreducible minimum -- skip any one of these and the kernel will not execute correctly:

| # | Requirement | Implementation |
|---|------------|----------------|
| 1 | PCI device initialization | `tt_exalens_init.init_ttexalens()` |
| 2 | Power state activation | `arc_msg(GO_BUSY)` |
| 3 | Architecture detection | `get_chip_architecture()` |
| 4 | Cross-compilation of 4 ELFs | `riscv-tt-elf-g++` with correct flags and linker scripts |
| 5 | Runtime parameter serialization to L1 | `struct.pack()` + `write_to_device(0x20000, ...)` |
| 6 | Input data tilization and write to L1 | `pack_*()` + `write_to_device(0x21000+, ...)` |
| 7 | BRISC firmware load and release | `load_elf(brisc.elf)` + `set_tensix_soft_reset(0, [BRISC])` |
| 8 | TRISC ELF loading | `load_elf(unpack.elf)`, `load_elf(math.elf)`, `load_elf(pack.elf)` |
| 9 | BRISC command to start TRISCs | `commit_brisc_command(START_TRISCS)` |
| 10 | Mailbox polling for completion | `read_word_from_device(mailbox_addr) == 0xFF` |
| 11 | Output readback and untilization | `read_from_device()` + `unpack_res_tiles()` + `untilize_block()` |
| 12 | Power state deactivation | `arc_msg(GO_IDLE)` |

This represents the complete host-side contract for pure-LLK kernel execution. TT-Metal wraps all twelve steps into a handful of API calls (`CreateDevice`, `CreateKernel`, `CreateBuffer`, `EnqueueWriteBuffer`, `EnqueueProgram`, `EnqueueReadBuffer`, `CloseDevice`), but underneath, these same operations must occur.

> **Cross-reference:** [Chapter 3, Section 1](../ch3_kernel_architecture/01_single_source_three_kernels.md) provides the architectural context for why four separate ELFs are needed (BRISC + three TRISCs). [Chapter 5](../ch5_compile_time_config/) documents the LLK kernel APIs (unpack, math, pack) that make up the `run_kernel()` call. [Chapter 6](../ch6_compilation/) covers Step 4 (compilation) in exhaustive detail. [Chapter 8](../ch8_test_infrastructure/01_matmul_test_survey.md) examines the test infrastructure that wraps this workflow.

---

**Previous:** [Step-by-Step Execution](./02_step_by_step_execution.md) | **Next:** [Chapter 8 -- Test Infrastructure](../ch8_test_infrastructure/01_matmul_test_survey.md)
