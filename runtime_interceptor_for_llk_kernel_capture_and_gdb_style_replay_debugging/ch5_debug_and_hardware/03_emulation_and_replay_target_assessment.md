# Chapter 5, File 3 -- Emulation and Replay Target Assessment

> **Scope.** This file evaluates the feasibility of replaying captured LLK
> kernel state on a host-side emulator or attaching a GDB session to a
> running Tensix core.  It covers JTAG access, RISC-V emulators
> (Spike, QEMU, riscvOVPsim), the existing host functional model
> (`trisc.cpp`), concrete emulator workflows, four GDB-attach options
> ranked by feasibility, and the `TT_METAL_RISCV_DEBUG_INFO` flag.
> All citations reference commit `621b949`.  Cross-references use
> **(Ch N, File M)** for inter-chapter and **(File M, Section N)** for
> intra-chapter links.

---

## 1. JTAG Access Layer

### 1.1 JtagDevice API

The UMD (User Mode Driver) layer provides a JTAG abstraction via the
`JtagDevice` class:

| File | Path |
|------|------|
| API header | `tt_metal/third_party/umd/device/api/umd/device/jtag/jtag_device.hpp` |
| Low-level wrapper | `tt_metal/third_party/umd/device/api/umd/device/jtag/jtag.hpp` |
| Implementation | `tt_metal/third_party/umd/device/jtag/jtag_device.cpp`, `jtag.cpp` |

Key operations (v1 File 03, Section 1.1):

```cpp
class JtagDevice {
    static std::shared_ptr<JtagDevice> create(binary_directory, target_devices);

    // NOC-based read/write (JTAG -> TDR -> NOC -> target core L1)
    void write32(chip_id, noc_x, noc_y, address, data, noc_id);
    optional<uint32_t> read32(chip_id, noc_x, noc_y, address, noc_id);
    void write(chip_id, mem_ptr, noc_x, noc_y, addr, size, noc_id);
    void read(chip_id, mem_ptr, noc_x, noc_y, addr, size, noc_id);

    // AXI-based read/write (direct PCIe BAR access via JTAG)
    void write32_axi(chip_id, address, data);
    optional<uint32_t> read32_axi(chip_id, address);

    // Debug bus / TAP Data Register access
    optional<uint32_t> read_tdr(chip_id, client, reg_offset);
    optional<int> write_tdr(chip_id, client, reg_offset, data);
    optional<uint32_t> readmon_tdr(chip_id, client, id, reg_offset);
    optional<int> writemon_tdr(chip_id, client, id, reg_offset, data);

    // Signal / memory dump
    void dbus_memdump(chip_id, client, mem, thread_id, start_addr, end_addr);
    optional<int> dbus_sigdump(chip_id, client, dbg_client_id, sig_sel_start, sig_sel_end);

    // Device identification
    optional<uint32_t> read_id(chip_id);
    bool is_hardware_hung(chip_id);
};
```

The underlying `Jtag` class (`jtag.hpp`) dynamically loads a shared library
(`DlHandle`) and exposes J-Link compatible operations.

### 1.2 JTAG -> Debug Module Path (Quasar)

On Quasar, the JTAG TDR (TAP Data Register) can reach the Debug Module
APB registers documented in **(File 2, Section 3)**.  The path is:

```
Host -> J-Link USB -> JTAG TAP -> TDR client "tensix" ->
    Debug Module APB (0x0300A000) -> dmcontrol/dmstatus/command/progbuf/data
```

This provides the same halt/step/register-read capabilities as the
on-device `rvdbg_cmd` interface, but driven from the host.

### 1.3 JTAG -> NOC -> L1 Path (WH/BH/Quasar)

For all architectures, JTAG can reach any core's L1 via NOC routing:

```
Host -> J-Link -> JTAG -> NOC write/read -> target core L1
```

This enables reading watcher mailboxes, DPRINT buffers, kernel ELF images
from L1, and writing to `RISCV_DEBUG_REG_BREAKPOINT_CTRL` (WH/BH) via
the NOC-translated L1 address.

**Bandwidth limitation:** JTAG transfers are serial, ~1-10 KB/s typical.
Reading a full DEST array (e.g. 16 rows x 8 x 32 bits = 512 bytes per
face) takes ~50-500 ms depending on J-Link speed.  Not suitable for
real-time tracing but acceptable for breakpoint-and-inspect workflows.

---

## 2. RISC-V Emulators

### 2.1 Spike

Spike is the official RISC-V ISA simulator from the RISC-V Foundation.
It executes RISC-V ELF binaries instruction-by-instruction and supports
the GDB Remote Serial Protocol (RSP) via `--rbb-port` or `-d`.

**Architecture match** (v3 File 03, Section 3.2):

| Feature | Tensix RISC-V | Spike Support |
|---------|--------------|---------------|
| RV32I base | Yes | Yes |
| M extension (multiply) | Yes | Yes |
| F extension (float) | Yes (WH/BH: RV32IMF) | Yes |
| Custom `.ttinsn` extension | Yes | **No** -- requires custom extension |
| Memory-mapped coprocessor | Yes | Partially -- custom MMIO handlers |
| Local memory at `0xFFB00000` | Yes | Configurable memory map |
| L1 at `0x0` | Yes | Configurable memory map |

**What works in Spike emulation:**
- RISC-V scalar code execution (control flow, memory access to L1 model)
- GDB breakpoints, stepping, register inspection on RISC-V instructions
- Memory-mapped I/O callbacks for the instruction buffer at `0xFFE40000`

**What is lost in Spike emulation:**
- Tensix coprocessor state (SRCA/SRCB/DEST arrays, FPU pipeline)
- Hardware semaphores and synchronization primitives
- NOC transactions (real network fabric behavior)
- TDMA (tile DMA) and unpacker/packer hardware behavior
- Cycle-accurate timing (Spike is a functional ISA simulator, **not**
  cycle-accurate -- contra v4's erroneous claim of "cycle-accurate CSR
  emulation")
- Multi-core interactions

**Custom extension approaches** (v3 File 03, Section 3.3):

*Approach A -- Custom instruction extension:* Spike's extension API
allows registering custom opcodes.  Each `.ttinsn` opcode would be
registered as a custom instruction that calls a C++ functional model.
This handles `TTI_*` (inline) instructions.

*Approach B -- Instruction buffer trap:* Trap writes to the instruction
buffer address (`0xFFE40000`) and interpret the written value as a Tensix
opcode.  This handles `TT_*` (trigger) instructions but not `TTI_*`.

**PROPOSED:** Both approaches should converge on a single `execute_instrn()`
dispatch that decodes the major opcode (bits [31:24] from `ckernel_ops.h`)
and calls the appropriate functional model.  The `TT_*` path invokes via
MMIO handler; the `TTI_*` path invokes via Spike's custom instruction
handler.  This unified path ensures identical results for both instruction
delivery mechanisms.

### 2.2 QEMU

QEMU provides system-level and user-mode RISC-V emulation.  Its JIT
compilation offers better raw emulation performance than Spike (~10-50x
slower than silicon vs Spike's ~100x estimate).

**Key mismatches with Tensix** (v3 File 03, Section 4):
- QEMU assumes M+S+U privilege modes (Tensix is M-only)
- Expects Sv39/Sv48 MMU (Tensix has none)
- Expects PLIC/CLINT interrupt controllers (Tensix has none)
- Uses a standard SiFive memory map (Tensix uses custom layout)

A custom machine type (`qemu/hw/riscv/tensix.c`) would be needed to map
L1 at `0x0`, local memory at `0xFFB00000`, and the instruction buffer at
`0xFFE40000`.  Implementing Tensix instructions requires TCG (Tiny Code
Generator) helper functions -- higher effort than Spike's extension API.

**Verdict:** QEMU is viable but not recommended over Spike.  The custom
machine type and TCG implementation overhead exceed the benefit for this
use case.  The JIT performance advantage is irrelevant for 16K kernels.

### 2.3 riscvOVPsim (Imperas Reference Model)

The plan requires assessment of riscvOVPsim, the Imperas RISC-V reference
simulator.  riscvOVPsim is a commercially-supported ISA simulator that
supports custom instruction extensions via the OVP API and includes
GDB RSP integration.

**Applicability to Tensix:**

| Feature | riscvOVPsim | Assessment |
|---------|-------------|------------|
| RV32IMF support | Yes | Matches Tensix ISA base |
| Custom instructions | OVP instruction decode API | Medium effort |
| MMIO device models | OVP platform modeling | Medium effort |
| GDB RSP | Built-in | Works out of the box |
| Licensing | Commercial (free ref model available) | Cost consideration |
| Multi-hart | Yes | Could model 3-4 TRISCs |

riscvOVPsim's primary advantage over Spike is its commercially-supported
custom instruction framework, which may reduce the effort of implementing
Tensix instruction stubs.  Its primary disadvantage is the commercial
license requirement for production use.  The free reference model
(riscvOVPsimPlus) may be sufficient for debug-only workflows.

**Verdict:** riscvOVPsim is a reasonable alternative to Spike if
commercial support and a more structured extension API are valued.
For an open-source-first project, Spike remains preferred.

### 2.4 Emulator Comparison Summary

| Aspect | Spike | QEMU | riscvOVPsim | Verdict |
|--------|-------|------|-------------|---------|
| RISC-V scalar code | Full | Full | Full | All work |
| GDB RSP | Built-in | Built-in | Built-in | All work |
| Tensix coprocessor | Not modeled | Not modeled | Not modeled | **Gap** |
| Custom instructions | Plugin API | TCG frontend | OVP decode API | Spike easiest |
| Memory-mapped I/O | Callbacks | Device model | OVP platform | All feasible |
| Integration effort | Low (standalone) | Medium (system) | Medium (API) | Spike preferred |
| License | Open source | Open source | Commercial | Spike/QEMU for OSS |
| Performance | ~100x slower | ~10-50x slower | ~50-100x slower | QEMU fastest |

No references to Spike, QEMU, or riscvOVPsim were found in the tt-metal
or tt-llk codebases, indicating none is currently integrated.

---

## 3. Host Functional Model: trisc.cpp

### 3.1 What trisc.cpp Actually Does

The file `tt-llk/tests/helpers/src/trisc.cpp` is the entry point for LLK
test kernels compiled to run on Tensix TRISC processors.  It is **not** a
host-side functional model -- it is a bare-metal RISC-V program that runs
on actual hardware.  However, its structure reveals exactly what a host
functional model would need to provide.

The execution flow (`trisc.cpp`, lines 36-77) (v3 File 03, Section 1.1):

```
_start()                           // CRT0: set GP, SP, init BSS, copy .data
  -> main()
       -> fill(regfile, 0, 64)     // Zero the 64-entry register file
       -> reset_cfg_state_id()     // Reset config state flip-flop (WH/BH only)
       -> reset_dest_offset_id()   // Reset dest offset flip-flop (WH/BH only)
       -> [profiler reset/sync]    // If LLK_PROFILER enabled
       -> run_kernel(__runtime_args_start)  // Execute the LLK kernel
       -> tensix_sync()            // Wait for all coprocessor ops to complete
       -> *mailbox = KERNEL_COMPLETE  // Signal completion
```

Runtime args are passed directly via the `__runtime_args_start` linker symbol
to `run_kernel()` — there is no intermediate copy step.

The `reset_cfg_state_id()` and `reset_dest_offset_id()` calls are guarded
by `#ifndef ARCH_QUASAR` (`trisc.cpp` lines 80-83), confirming they are
WH/BH-specific.

### 3.2 Hardware Dependencies in trisc.cpp

The function depends on these hardware-specific elements
(v3 File 03, Section 1.2):

| Dependency | Source | Emulation Difficulty |
|-----------|--------|---------------------|
| Mailbox memory-mapped I/O | `mailboxes_start = 0x1FFC0` | Easy: memory-mapped region |
| `RuntimeParams` from L1 | `__runtime_args_start[]` linker symbol | Easy: populate from capture |
| `regfile` (config register base) | `REGFILE_BASE` memory-mapped address | Medium: functional model |
| `pc_buf_base` (PC buffer) | `PC_BUF_BASE` memory-mapped address | Medium: used for `tensix_sync()` |
| `instrn_buffer` (instruction FIFO) | `0xFFE40000` memory-mapped address | Hard: Tensix instruction interface |
| Tensix coprocessor instructions | `TTI_*` macros (`.ttinsn`) | Hard: custom ISA extension |
| L1 SRAM (data tiles, CBs) | Memory range `0x0`-`0x100000` | Medium: flat memory model |
| Semaphores | `t6_semaphore_init()` | Easy: software counter model |
| Wall clock | `read_wall_clock()` | Easy: host timestamp |

### 3.3 The Instruction Buffer Interface

The critical bottleneck for emulation is the Tensix instruction interface.
LLK kernels issue Tensix coprocessor instructions through two mechanisms
(v3 File 03, Section 1.3):

**TTI (Tensix Target Instruction):** Inline assembly that swizzles a
32-bit opcode into the RISC-V instruction stream via the `.ttinsn`
pseudo-op (`ckernel_ops.h`, line 12):

```cpp
#define INSTRUCTION_WORD(x) __asm__ __volatile__(".ttinsn %0" : : "i"((x)))
```

This is a custom RISC-V ISA extension.  Standard emulators do not support
`.ttinsn`.

**TT (Tensix Trigger):** Register writes to the instruction buffer at
`instrn_buffer[0]`, which triggers the coprocessor to execute the written
instruction:

```cpp
#define TT_MATMUL(...) ckernel::instrn_buffer[0] = TT_OP_MATMUL(__VA_ARGS__)
```

This mechanism is memory-mapped and can be intercepted by a memory write
handler in an emulator.

### 3.4 Build System Context

LLK test kernels are built using a RISC-V cross-compiler (`riscv-tt-elf`)
with custom linker scripts.  The Blackhole memory layout
(`tests/helpers/ld/memory.blackhole.ld`) (v3 File 03, Section 1.4):

```
LOCAL_MEM_BASE = 0xFFB00000       // Private local memory
L1_BASE = 0x0                     // Shared L1 SRAM
RUNTIME_ARGS_START = 0x20000      // Runtime arguments region

BRISC_CODE          : ORIGIN = 0x0,      LENGTH = 16K
TRISC0_CODE         : ORIGIN = 0x6000,   LENGTH = 16K
TRISC1_CODE         : ORIGIN = 0xB000,   LENGTH = 16K
TRISC2_CODE         : ORIGIN = 0x10000,  LENGTH = 16K
RUNTIME_ARGS        : ORIGIN = 0x20000,  LENGTH = 1K
```

### 3.5 SFPU Stub Pattern

The `sfpu_stub.h` (`tests/helpers/include/sfpu_stub.h`, path at time of writing;
may have been relocated) shows
the pattern for stubbing out a TRISC:

```cpp
#ifdef LLK_TRISC_ISOLATE_SFPU
void run_kernel(RUNTIME_PARAMETERS params) {
    (void)params;
    // Stub: SFPU TRISC not used in this test
}
#endif
```

This demonstrates that the LLK test infrastructure already supports
stub-based compilation for TRISCs not exercised by a particular test.
The same pattern extends to stubbing out entire coprocessor subsystems
for host replay.

---

## 4. Emulability Classification: The E-Taxonomy

### 4.1 Four Categories

Every LLK operation falls into one of four emulation categories
(v3 File 03, Section 5.1):

| Category | Description | Stub Strategy |
|----------|-------------|---------------|
| **E-Pure** | Pure computation, no hardware dependency | Direct execution (host or emulator) |
| **E-Model** | Hardware-dependent but functionally modelable | Software functional model |
| **E-Stub** | Hardware-dependent, approximate model only | Behavioral stub with configurable fidelity |
| **E-Opaque** | Hardware-dependent, not modelable without RTL | Capture actual results, replay from recording |

### 4.2 Detailed Operation Map

| LLK Operation | Category | Rationale | Stub Complexity |
|---------------|----------|-----------|-----------------|
| `llk_math_matmul_init` | E-Model | Config register writes for address modes/fidelity | Medium |
| `llk_math_matmul` (inner loop) | E-Model | Issues `TTI_MVMUL` -- 16x16 face multiply | High -- need FP16/BFP8/FP32 matmul with correct rounding |
| `llk_unpack_A_init` | E-Model | Unpacker format, tile dims, address gen config | Medium |
| `llk_unpack_A` (tile load) | E-Model | L1 read, format convert, write to SRCA | High -- BFP8/FP16/FP32 format conversion, tile addressing |
| `llk_pack_init` | E-Model | Packer output format, dest offset config | Medium |
| `llk_pack` (tile store) | E-Model | DEST read, format convert, write to L1 | High -- reverse format conversion |
| `llk_math_eltwise_binary` | E-Model | Element-wise add/sub/mul SRCA x SRCB -> DEST | Medium |
| `llk_math_reduce` | E-Model | Row/column/scalar reduction | Medium |
| `llk_math_eltwise_unary_sfpu` | E-Pure to E-Model | SFPU ops (exp, log, sqrt, etc.) | Low-Medium -- standard math library |
| `llk_acquire_dst` / `llk_release_dst` | E-Stub | Semaphore-based dest double buffering | Low -- flip `dest_offset_id` |
| `cb_push_back` / `cb_pop_front` | E-Stub | Circular buffer pointer management | Low -- update read/write pointers |
| `tensix_sync()` | E-Stub | Wait for coprocessor pipeline to drain | Trivial -- no-op in functional model |
| Config register writes | E-Model | Sets hardware behavior for subsequent ops | Low -- store to array, interpret on operation |
| Semaphore operations | E-Stub | `semaphore_post`, `semaphore_wait` | Low -- atomic increment/decrement |
| NOC transactions | E-Opaque | Network-on-chip reads/writes between cores | Cannot model without multi-core simulation |
| Clock gating / branch prediction | E-Stub | `RISCV_DEBUG_REG_DEST_CG_CTRL`, etc. | Trivial -- no-op |

### 4.3 Format Conversion Fidelity

The hardest part of functional modeling is matching the hardware's numeric
behavior.  Tensix supports these data formats (v3 File 03, Section 5.3):

| Format | Bits | Exponent | Mantissa | Rounding | Emulation Difficulty |
|--------|------|----------|----------|----------|---------------------|
| FP32 | 32 | 8 | 23 | IEEE 754 | Trivial -- native float |
| FP16 | 16 | 5 | 10 | Truncation | Low -- standard half-float |
| BFP16 | 16 | 8 | 7 | Truncation | Low -- bfloat16 |
| BFP8 | Shared exp + 8b mantissa | Shared | 7 | Truncation | Medium -- block format |
| BFP4 | Shared exp + 4b mantissa | Shared | 3 | Truncation | Medium -- block format |
| BFP2 | Shared exp + 2b mantissa | Shared | 1 | Truncation | Medium -- block format |
| TF32 | 19 | 8 | 10 | Truncation | Low |

The block floating-point formats (BFP8, BFP4, BFP2) share an exponent
across a block of values and are the primary source of divergence between
a software model and hardware.  TT-Metal already has host-side BFP
conversion functions in `tt_metal/impl/data_format/blockfloat_common.hpp`
that can be reused.  The existing `fp16a`/`fp16b` types in
`ckernel_dest.h` (lines 314-525) also provide correct format conversion
routines.

---

## 5. Emulator Workflow for Replay Debugging

### 5.1 Overview

**PROPOSED:** The recommended replay architecture uses Spike with a custom
Tensix extension, loading state from a captured trace file.  The workflow
has four phases:

```
[1. Capture] -> [2. Prepare] -> [3. Load] -> [4. Debug]
```

### 5.2 Phase 1: Capture (On Device)

Using the interceptor described in **(Ch 2, Ch 3)**:

1. Intercept LLK API calls (`llk_unpack`, `llk_math`, `llk_pack`, etc.)
2. Capture: function name, arguments, L1 tile data (input tiles)
3. Capture: Tensix config register state at entry (via `dbg_read_cfgreg`)
4. Optionally capture: SRCA/SRCB/DEST snapshots (via `dbg_get_array_row`)
5. Write capture log to host via DPRINT buffer or dedicated DMA channel

The capture produces a trace bundle (v3 File 03, Section 6.2):

```
trace_bundle/
  metadata.json          # Kernel name, arch, core coords, timestamp
  trisc0_kernel.elf      # Unpack kernel ELF binary
  trisc1_kernel.elf      # Math kernel ELF binary
  trisc2_kernel.elf      # Pack kernel ELF binary
  l1_snapshot.bin         # L1 SRAM contents at capture point (1MB)
  local_mem_trisc0.bin   # TRISC0 local memory (4K)
  local_mem_trisc1.bin   # TRISC1 local memory (4K)
  local_mem_trisc2.bin   # TRISC2 local memory (4K)
  config_regs.bin        # 187 HW config registers + 3 thread configs
  runtime_args.bin       # Runtime arguments
  cb_config.json         # Circular buffer addresses, sizes, formats
  semaphore_state.bin    # Initial semaphore values
```

**PROPOSED:** The capture header should include an architecture tag for
cross-architecture validation (v5 File 03, Section 7.2):

```cpp
struct CaptureHeader {
    uint32_t magic;           // "LLKC"
    uint32_t arch;            // ARCH_WORMHOLE_B0=0, ARCH_BLACKHOLE=1, ARCH_QUASAR=2
    uint32_t capture_level;   // TIER1=1, TIER2=2, TIER3=3
    uint32_t num_entries;
    uint64_t wall_clock_start;
};
```

### 5.3 Phase 2: Prepare

A preparation script transforms the trace bundle into Spike-loadable
format (v3 File 03, Section 6.3):

```bash
#!/bin/bash
# prepare_replay.sh <trace_dir> <output_dir>
TRACE=$1; OUT=$2; mkdir -p "$OUT"

# 1. Generate a combined memory image for Spike
python3 merge_memory.py \
    --l1 "$TRACE/l1_snapshot.bin" \
    --local0 "$TRACE/local_mem_trisc0.bin" \
    --local1 "$TRACE/local_mem_trisc1.bin" \
    --local2 "$TRACE/local_mem_trisc2.bin" \
    --runtime-args "$TRACE/runtime_args.bin" \
    --config-regs "$TRACE/config_regs.bin" \
    --semaphores "$TRACE/semaphore_state.bin" \
    --output "$OUT/memory.bin"

# 2. Copy ELFs
cp "$TRACE/trisc1_kernel.elf" "$OUT/kernel.elf"
```

### 5.4 Phase 3: Load

Launch Spike with the prepared state and GDB stub
(v3 File 03, Section 6.4):

```bash
# Start Spike with Tensix extension and captured memory state
spike --isa=rv32imf \
      --extension=tensix_coproc \
      --load-memory="$OUT/memory.bin" \
      --pc=0x6000 \
      -d --rbb-port=9824 \
      "$OUT/kernel.elf"
```

Where:
- `--extension=tensix_coproc` loads the custom Tensix coprocessor extension
- `--load-memory` pre-populates emulated memory with the captured L1 snapshot
- `--pc=0x6000` sets the initial PC to TRISC0's code base
- `--rbb-port=9824` enables the GDB remote debug port

### 5.5 Phase 4: Debug

Connect GDB for interactive debugging (v3 File 03, Section 6.5):

```bash
# Connect with Tenstorrent RISC-V GDB
riscv-tt-elf-gdb "$OUT/kernel.elf" \
    -ex "set architecture riscv:rv32" \
    -ex "target remote :9824" \
    -ex "break run_kernel" \
    -ex "continue"

# Once at breakpoint:
(gdb) info registers          # View RISC-V GPRs
(gdb) x/16xw 0x1FFC0         # View mailbox contents
(gdb) x/32xw 0x20000         # View runtime args
(gdb) break llk_math_matmul
(gdb) continue
(gdb) step                    # Step into matmul implementation
(gdb) watch *(uint32_t*)0xFFE40000   # Break on instruction buffer write
```

### 5.6 Fidelity Levels

| Level | Models | Accuracy | Effort |
|-------|--------|----------|--------|
| **L0: RISC-V only** | Scalar code, control flow | Correct for branches/loads/stores | Low |
| **L1: + Tensix instruction decode** | Instruction stream recording | Can verify instruction sequence | Medium |
| **L2: + Register array model** | SRCA/SRCB/DEST contents | Approximate (no pipeline timing) | High |
| **L3: + Full pipeline** | Cycle-accurate FPU/SFPU | Exact match to hardware | Very high |

For debugging LLK correctness bugs, **L1 or L2** is typically sufficient.
L3 is only needed for timing-related bugs.

---

## 6. RTL Simulation Infrastructure

### 6.1 RtlSimulationChip

The UMD layer provides a full RTL simulation backend
(v2 File 03, Section 5.3.1):

Source files:
- `tt_metal/third_party/umd/device/api/umd/device/simulation/rtl_simulation_chip.hpp`
- `tt_metal/third_party/umd/device/simulation/rtl_simulation_chip.cpp`
- `tt_metal/third_party/umd/device/simulation/simulation_host.cpp`
- `tt_metal/third_party/umd/device/simulation/simulation_chip.cpp`

```cpp
class RtlSimulationChip : public SimulationChip {
    void write_to_device(CoreCoord core, const void* src,
                         uint64_t l1_dest, uint32_t size) override;
    void read_from_device(CoreCoord core, void* dest,
                          uint64_t l1_src, uint32_t size) override;
    void send_tensix_risc_reset(tt_xy_pair translated_core,
                                const TensixSoftResetOptions& soft_resets) override;
    void assert_risc_reset(CoreCoord core, const RiscType selected_riscs) override;
    void deassert_risc_reset(CoreCoord core, const RiscType selected_riscs,
                             bool staggered_start) override;
};
```

This interface maps directly to the operations a replay engine would
need: write captured tile data to L1, configure RISC reset state,
deassert reset to begin execution, and read results back.

### 6.2 Versim and Simulation Core Descriptors

Multiple simulation-targeted core descriptors exist (v2 File 03, Section 5.3.1):

| File | Architecture | Grid Size | Purpose |
|------|-------------|-----------|---------|
| `wormhole_b0_versim_1x1_arch.yaml` | WH | 1x1 | Single-core Versim |
| `blackhole_versim_1x1_arch.yaml` | BH | 1x1 | Single-core Versim |
| `blackhole_simulation_1x2_arch.yaml` | BH | 1x2 | Two-core simulation |
| `quasar_simulation_1x3_arch.yaml` | Quasar | 1x3 | Three-core simulation |
| `quasar_simulation_8x4_arch.yaml` | Quasar | 8x4 | Full-grid simulation |

These confirm the infrastructure already supports reduced-grid simulation,
aligning with the interceptor's need to replay single-core operations.

### 6.3 Emulation Container

An `emulation_rocky8.def` Singularity definition file exists in the UMD
directory, indicating that containerized emulation environments are a
supported deployment model (v2 File 03, Section 5.3.1).

### 6.4 tt-exalens Simulation Bridge

The existing test infrastructure integrates with a cycle-accurate simulator
via tt-exalens (v5 File 03, Section 2.6):

```python
# conftest.py
if test_target.run_simulator:
    simulator_path = os.environ.get("TT_UMD_SIMULATOR_PATH")
    _exalens_server = ExalensServer(
        simulator_path=simulator_path,
        port=test_target.simulator_port,
    )
    _exalens_server.start()
```

<!-- Citation: tests/python_tests/conftest.py -->
<!-- Citation: ExalensServer READY_TIMEOUT_S = 600 (exalens_server.py, path at time of writing; may have been relocated) -->

The simulator provides full Tensix RTL-level emulation with the same
debug access as silicon (via tt-exalens' unified interface).  Startup
time is 60-600 seconds per session.

---

## 7. Four GDB-Attach Options

### 7.1 Option A: GDB via Quasar Debug Module + JTAG

**Path:** Host GDB -> GDB RSP -> custom stub -> JTAG -> Debug Module APB
-> RISC-V core

**Feasibility:** HIGH for Quasar.  The Debug Module provides all GDB RSP
requirements: halt, step, register read/write, memory access, breakpoints.
The JTAG `read_tdr`/`write_tdr` functions can reach the Debug Module.

The standard flow: write `haltreq=1` to `DMCONTROL` to halt, issue
"Access Register" abstract commands for GPR reads, inject `lw`/`sw` into
the program buffer for memory access, set the `step` bit for single-step.
`HALTSUMMARY0/1` provides instant halt-status across all harts.

**Latency:** ~100 ms per register read via JTAG.  Single-stepping takes
~500 ms per step.

**Advantages:**
- Live debugging on real hardware
- Full RISC-V state visibility
- Standard GDB UI
- No firmware modifications needed

**Disadvantages:**
- Quasar only (WH/BH lack a compliant Debug Module)
- No Tensix coprocessor state in GDB (requires custom GDB commands)
- JTAG bandwidth limits usability
- Requires J-Link hardware (no SSH-forwarded debugging)
- Multi-hart complexity (up to 24 harts per Tensix on Quasar)

**Estimated effort:** 2-4 engineer-weeks (primarily OpenOCD configuration).

### 7.2 Option B: GDB via Custom Host Stub + UMD PCIe MMIO

**Path:** Host GDB -> GDB RSP -> host stub -> PCIe MMIO -> debug registers
+ L1 memory

**Feasibility:** LOW on WH/BH.  **Critical limitation:** on WH/BH, Baby
RISC-V GPRs are not exposed via any memory-mapped register (v4 File 03,
Section 5.3.3).  `RISC_DBG_STATUS_0` reports halt status but not register
contents or PC values.  Without a Debug Module's "Access Register" command,
there is no hardware path to read `x0`-`x31` or CSRs.  The only workaround
(saving GPRs to L1 via a trap handler) makes this equivalent to Option C.

L1 memory is fully accessible via PCIe MMIO (~200 ns/word), so memory
inspection works.  But the halt mechanism is limited.

**Verdict:** Not viable as a standalone approach on WH/BH.  On Quasar,
dominated by Option A (OpenOCD).  Not recommended.

### 7.3 Option C: GDB via ebreak + Mailbox Protocol (Recommended for WH/BH)

**Path:** Host GDB -> GDB RSP -> host relay -> L1 mailbox -> on-device
trap handler -> RISC-V debug interface

**Feasibility:** HIGH.  This is the most practical approach for WH/BH.

#### 7.3.1 The ebreak Mechanism

The `ebreak` instruction (encoding `0x00100073`) is standard RISC-V.
On Tenstorrent's Baby RISC-V cores (v4 File 03, Section 5.3.4):

1. The `ebreak` opcode is fetched and decoded
2. The processor traps to the exception handler vector at `mtvec`
3. `mcause` is set to `3` (breakpoint)
4. `mepc` saves the PC of the `ebreak` instruction
5. Execution transfers to the trap handler

Without a trap handler installed, the core halts.  `RISC_DBG_STATUS_0`
(offset `0x88`) reflects the halt.  On Quasar, the Debug Module detects
`ebreak` and sets `dmi_dmstatus_impebreak` in the status register.

#### 7.3.2 PROPOSED: L1 Mailbox Layout

```
Offset  Size   Field
------  -----  -----
0x000   4      cmd        : Host -> Device command
0x004   4      status     : Device -> Host status
0x008   4      arg0       : Command argument 0
0x00C   4      arg1       : Command argument 1
0x010   4      result     : Command result
0x014   4      pc         : Saved PC (from mepc CSR)
0x018   128    gpr[32]    : Saved GPRs x0-x31
0x098   4      mcause     : Saved mcause
0x09C   4      mstatus    : Saved mstatus
0x0A0   256    data_buf   : Data buffer for memory reads
------
Total: 0x1A0 = 416 bytes per RISC
```

For 5 RISCs, total mailbox overhead: 5 x 416 = 2080 bytes.

#### 7.3.3 PROPOSED: Command Codes

```c
enum DebugCmd {
    DBG_CMD_NONE     = 0x00,  // No command pending
    DBG_CMD_HALT     = 0x01,  // Request halt at next safe point
    DBG_CMD_RESUME   = 0x02,  // Resume execution
    DBG_CMD_READ_REG = 0x03,  // Read GPR (arg0 = reg index)
    DBG_CMD_WRITE_REG= 0x04,  // Write GPR (arg0 = index, arg1 = value)
    DBG_CMD_READ_MEM = 0x05,  // Read memory (arg0 = addr, arg1 = len)
    DBG_CMD_WRITE_MEM= 0x06,  // Write memory (arg0 = addr, arg1 = value)
    DBG_CMD_STEP     = 0x07,  // Single step
    DBG_CMD_SET_BP   = 0x08,  // Set software breakpoint (arg0 = addr)
    DBG_CMD_CLR_BP   = 0x09,  // Clear software breakpoint (arg0 = addr)
};
```

#### 7.3.4 PROPOSED: Trap Handler (Device-Side Pseudocode)

```c
// Installed at mtvec, runs on ebreak or DBG_CMD_HALT flag
__attribute__((naked, aligned(4)))
void debug_trap_handler(void) {
    // Save all GPRs to mailbox
    asm volatile(
        "sw x1,  0x01C(%0)\n"
        "sw x2,  0x020(%0)\n"
        // ... save x3-x31 ...
        "sw x31, 0x098(%0)\n"
        "csrr t0, mepc\n"
        "sw t0, 0x014(%0)\n"     // Save PC
        "csrr t0, mcause\n"
        "sw t0, 0x098(%0)\n"     // Save mcause
        "csrr t0, mstatus\n"
        "sw t0, 0x09C(%0)\n"     // Save mstatus
        : : "r"(debug_mailbox_base)
    );

    debug_mailbox->status = DBG_STATUS_HALTED;

    // Command loop: poll L1 for host commands
    while (1) {
        while (debug_mailbox->cmd == DBG_CMD_NONE)
            invalidate_l1_cache();

        switch (debug_mailbox->cmd) {
        case DBG_CMD_READ_REG:
            debug_mailbox->result = debug_mailbox->gpr[debug_mailbox->arg0];
            debug_mailbox->status = DBG_STATUS_ACK;
            break;
        case DBG_CMD_READ_MEM: {
            uint32_t addr = debug_mailbox->arg0;
            uint32_t len = min(debug_mailbox->arg1, 256);
            memcpy(debug_mailbox->data_buf, (void*)addr, len);
            debug_mailbox->status = DBG_STATUS_ACK;
            break;
        }
        case DBG_CMD_SET_BP: {
            uint32_t* bp_addr = (uint32_t*)debug_mailbox->arg0;
            debug_mailbox->result = *bp_addr;  // Save original insn
            *bp_addr = 0x00100073;             // Write ebreak
            debug_mailbox->status = DBG_STATUS_ACK;
            break;
        }
        case DBG_CMD_STEP:
            // Set ebreak at mepc+4, then resume
            // Will immediately re-trap at next instruction
            goto restore_and_return;
        case DBG_CMD_RESUME:
            debug_mailbox->status = DBG_STATUS_RUNNING;
            goto restore_and_return;
        }
        debug_mailbox->cmd = DBG_CMD_NONE;
    }

restore_and_return:
    // Restore all GPRs from mailbox and mret
    asm volatile("lw x1, 0x01C(%0)\n" /* ... */ "mret\n"
                 : : "r"(debug_mailbox_base));
}
```

#### 7.3.5 PROPOSED: Host-Side GDB RSP Mapping

| GDB RSP Command | Mailbox Operation | Latency |
|---|---|---|
| `g` (read all regs) | Read `gpr[0..31]` from mailbox (already saved) | ~0.2 us |
| `G` (write all regs) | Write `gpr[0..31]` to mailbox | ~0.2 us |
| `m addr,len` (read mem) | `DBG_CMD_READ_MEM` in 256-byte chunks | ~0.2 us/word |
| `M addr,len:data` (write mem) | `DBG_CMD_WRITE_MEM` | ~0.2 us/word |
| `s` (single step) | `DBG_CMD_STEP` + wait for re-trap | ~1 us + kernel |
| `c` (continue) | `DBG_CMD_RESUME` | async |
| `Z0,addr` (set SW breakpoint) | `DBG_CMD_SET_BP` | ~0.6 us |
| `z0,addr` (clear SW breakpoint) | `DBG_CMD_CLR_BP` | ~0.6 us |

The stub also provides a target description XML declaring the Baby RISC-V
as an RV32IM target with 32 GPRs, PC, and relevant CSRs.

**Note:** For RISC-V compressed instructions (RVC), single-step must
decode instruction length (2 or 4 bytes) to set the next breakpoint
correctly.

#### 7.3.6 Feasibility Assessment

**Advantages:**
- Works on all architectures (WH/BH/Quasar)
- PCIe-speed communication (~200 ns per L1 read/write, no J-Link needed)
- Full register visibility via trap handler GPR save
- Software breakpoints via `ebreak` injection
- Single-step via next-instruction `ebreak` insertion

**Disadvantages:**
- Requires trap handler in the kernel binary (build flag `TT_METAL_DEBUG_TRAP_ENABLED`)
- Trap handler consumes 416 bytes of L1 per RISC (~2 KB total)
- Cannot debug the trap handler itself (chicken-and-egg problem)
- Does not provide hardware watchpoints (data breakpoints)
- RVC instruction length decoding adds complexity

**Estimated effort:** 4-6 engineer-weeks (prototype), 8-12 weeks
(production with GDB thread model support for multi-RISC).

### 7.4 Option D: BRISC Debug Agent

**Path:** GDB -> RSP -> Host Stub -> PCIe/MMIO -> BRISC (Debug Agent)
-> L1 shared memory -> TRISC0/1/2/NCRISC (targets)

**Concept:** Dedicate BRISC to run a permanent debug agent that manages
breakpoints on the other RISCs, reads their state, and communicates
with the host (v4 File 03, Section 5.3.5).

Because all RISCs share L1, BRISC can read/write trap handler mailboxes,
inject `ebreak` into TRISC code, and read saved GPRs -- all without
NOC transactions.  BRISC also has access to Tensix debug registers
(breakpoint controller, debug arrays, instruction buffer override).

**Advantages:**
- Separates debug infrastructure from the debugged kernel
- BRISC has the most privileged position (controls TRISC reset, manages
  Tensix pipeline)
- Does not modify the target kernel binary

**Disadvantages:**
- **Sacrifices BRISC:** Normal BRISC kernel (data movement) cannot run
  simultaneously -- may fundamentally change debugged program behavior
- Cannot debug BRISC itself
- Firmware complexity: essentially a small RTOS
- Reset PC override is BH-only for TRISCs (limited on WH)

**Estimated effort:** 12-16 engineer-weeks.

### 7.5 Comparative Scoring

Weighted evaluation (v4 File 03, Section 5.3.6):

| Criterion (weight) | A: OpenOCD+JTAG | B: Host Stub | C: ebreak+Mailbox | D: BRISC Agent |
|---|---|---|---|---|
| Arch coverage (0.20) | 2 (Quasar only) | 3 (all, limited) | 5 (all) | 4 (BH preferred) |
| Register visibility (0.25) | 5 (full via DM) | 1 (no GPR access) | 4 (via trap save) | 4 (via trap save) |
| Implementation effort (0.20) | 4 (config only) | 2 (incomplete) | 3 (medium) | 1 (high) |
| Performance (0.15) | 2 (JTAG slow) | 4 (PCIe fast) | 4 (PCIe fast) | 3 (relay overhead) |
| Invasiveness (0.10) | 5 (no FW change) | 3 (minimal) | 3 (trap handler) | 1 (loses BRISC) |
| Single-step (0.10) | 5 (hardware) | 1 (not possible) | 3 (SW breakpoint) | 3 (SW breakpoint) |
| **Weighted total** | **3.35** | **2.15** | **3.90** | **2.70** |

**Recommendation:** Option C (ebreak + mailbox) for WH/BH today.
Option A (OpenOCD + JTAG) for Quasar when available.

---

## 8. Seven-Dimensional Decision Matrix

Expanding beyond the four GDB-attach options, the full replay target space
includes six targets evaluated across seven dimensions
(v5 File 03, Sections 3.1-3.7):

### 8.1 Implementation Effort

| Target | New code (est.) | Dependencies | Timeline |
|--------|----------------|-------------|----------|
| Spike emulation | 15,000-25,000 lines | Spike source, custom ISA ext | 6-12 months |
| QEMU | 20,000-30,000 lines | QEMU source, TCG extensions | 9-15 months |
| Host functional model | 8,000-15,000 lines | None (pure C++) | 4-8 months |
| Real HW via ebreak | 500-1,500 lines | tt-exalens, LLK_ASSERT infra | 2-4 weeks |
| Real HW via Quasar debug regs | 2,000-5,000 lines | Quasar hardware, tt-exalens | 1-3 months |
| Cycle-accurate simulator | 500-2,000 lines | Simulator build, license | 2-6 weeks |

### 8.2 Fidelity

| Target | Scalar RISC-V | Tensix FPU | SFPU | TDMA/Pack/Unpack | NOC | Multi-core |
|--------|--------------|-----------|------|-----------------|-----|------------|
| Spike | Exact | Requires model | Requires model | Requires model | None | No |
| Host functional model | N/A (x86) | Approximate | Approximate | Software model | None | No |
| Real HW via ebreak | Exact | Exact | Exact | Exact | Exact | Yes |
| Real HW via Quasar debug | Exact | Exact | Exact | Exact | Exact | Yes |
| Cycle-accurate simulator | Exact | Exact | Exact | Exact | Exact | Yes |

### 8.3 Bug-Type Diagnosability

| Bug type | Spike | Host model | HW ebreak | Quasar dbg | Simulator |
|----------|-------|-----------|-----------|------------|-----------|
| Incorrect math (FPU/SFPU) | If modeled | Approximate | **Yes** | **Yes** | **Yes** |
| Data format conversion | If modeled | **Yes** | **Yes** | **Yes** | **Yes** |
| L1 buffer overflow | No | No | **Yes** | **Yes** | **Yes** |
| Config register misconfiguration | No | No | **Yes** | **Yes** | **Yes** |
| Pack/unpack addressing | No | Limited | **Yes** | **Yes** | **Yes** |
| Multi-thread race condition | No | No | Partial | **Yes** | **Yes** |
| Pipeline stall/deadlock | No | No | Partial | Partial | **Yes** |
| Timing-dependent behavior | No | No | No | No | **Yes** |
| Semaphore protocol violation | No | No | **Yes** | **Yes** | **Yes** |

---

## 9. Recommended Tiered Debug Strategy

**PROPOSED:** A three-tier debug strategy (v5 File 03, Section 5):

```
 Bug reported
      |
      v
 +--------------------+     Reproduced?     +--------------------------+
 | Tier 1: Host model | ------- Yes ------> | Debug with x86 GDB      |
 | (x86 replay)       |                     | Full source-level debug  |
 +--------------------+                     +--------------------------+
      | No
      v
 +--------------------+     Reproduced?     +--------------------------+
 | Tier 2: HW ebreak  | ------- Yes ------> | Inspect via tt-exalens   |
 | + Quasar debug regs |                    | Single-step, watchpoints |
 +--------------------+                     +--------------------------+
      | No (timing/multi-core)
      v
 +--------------------+     Reproduced?     +--------------------------+
 | Tier 3: Cycle-acc  | ------- Yes ------> | Full RTL waveform debug  |
 | simulator replay   |                     | Every signal visible     |
 +--------------------+                     +--------------------------+
```

### 9.1 Tier Transition Criteria

| From | To | Trigger |
|------|----|---------|
| Tier 1 | Tier 2 | Host model output matches golden; hardware diverges |
| Tier 1 | Tier 3 | Suspected timing or multi-core race condition |
| Tier 2 | Tier 3 | Bug only reproduces intermittently on hardware |
| Any | Tier 1 (retry) | After narrowing input data via higher tiers |

### 9.2 Capture Format Requirements per Tier

| Tier | Required capture fields |
|------|------------------------|
| 1 | LLK function name, arguments, input tile data (L1 buffer contents) |
| 2 | Tier 1 + kernel ELF path, L1 full snapshot, config register values |
| 3 | Tier 2 + complete `t6_debug_regs_t` dump, DEST/SRCA/SRCB contents |

### 9.3 Implementation Priority

| Priority | Target | Rationale |
|----------|--------|-----------|
| P0 | Host functional model | Highest ROI: fast, no HW needed, catches most math bugs |
| P1 | HW ebreak integration | Minimal code, leverages existing tt-exalens |
| P2 | Quasar GDB RSP bridge (Option C + Option A) | Enables true source-level on-device debug |
| P3 | Simulator test generator | Converts captures to RTL simulator test cases |
| P4 | Spike/QEMU/riscvOVPsim | Only if hardware-independent CI is a strategic goal |

---

## 10. TT_METAL_RISCV_DEBUG_INFO and riscv-tt-elf-gdb

### 10.1 Environment Variable

The `TT_METAL_RISCV_DEBUG_INFO` environment variable controls whether
kernel ELFs include DWARF debug information
(v1 File 03, Section 6):

```cpp
// rtoptions.hpp line 204
bool riscv_debug_info_enabled = false;

// rtoptions.cpp lines 1110-1122
case EnvVarID::TT_METAL_RISCV_DEBUG_INFO: {
    bool enable = this->get_inspector_enabled();  // default from inspector
    if (env_value)
        enable = true;
        if (strcmp(env_value, "0") == 0)
            enable = false;
    this->set_riscv_debug_info_enabled(enable);
}

// rtoptions.cpp lines 1332-1334
// Inherits from inspector if not explicitly set
if (std::getenv("TT_METAL_RISCV_DEBUG_INFO") == nullptr) {
    HandleEnvVar(EnvVarID::TT_METAL_RISCV_DEBUG_INFO, nullptr);
```

When enabled:
- Kernel compilation adds `-g` (or equivalent DWARF flags)
- ELF files contain source file paths, line numbers, variable names
- `riscv-tt-elf-gdb` can load these for symbol resolution
- Watcher dumps kernel ELF paths to `kernel_elf_paths.txt`
  (`watcher_server.cpp` lines 220-231)

### 10.2 riscv-tt-elf-gdb

Tenstorrent provides a custom GDB binary (`riscv-tt-elf-gdb`) targeting
the Baby RISC-V architecture.  This binary understands the RV32IM(F)
instruction set and Tenstorrent-specific target descriptions.  It is the
intended GDB frontend for any of the four attach options.

### 10.3 Interaction with Interceptor

**PROPOSED:** The interceptor **(Ch 2, Ch 3)** should ensure
`TT_METAL_RISCV_DEBUG_INFO=1` is set when capturing kernel state, so
that the replay debugger can provide source-level navigation.  The
capture bundle should always include the ELF with debug info enabled.

---

## 11. What Cannot Be Emulated

### 11.1 NOC Transactions

NOC reads and writes involve the on-chip network connecting to DRAM,
other Tensix cores, and external PCIe (v3 File 03, Section 7.1):
- `noc_async_read()` -- reads from remote L1 or DRAM
- `noc_async_write()` -- writes to remote address
- `noc_semaphore_inc()` -- atomically increments a remote semaphore
- `eth_send_packet()` -- sends data through Ethernet core

**Mitigation:** For single-core replay, pre-populate L1 with all data
the kernel will read (captured by the interceptor).  NOC writes are
logged but not executed.

### 11.2 Cycle-Accurate Timing

The Tensix FPU has pipeline stages with data-dependent latencies.  No
software emulator can reproduce pipeline stalls, memory latency
variations, or the exact cycle count for `read_wall_clock()`.

### 11.3 Multi-Core Interactions

Any kernel depending on runtime coordination with other cores (matmul
with multicast SRCB, all-reduce, CCL kernels) cannot be fully replayed
in a single-core emulator.

**Mitigation:** Capture and replay each core independently, using
the captured NOC transaction log to verify inter-core consistency.

---

## 12. Effort Estimation

| Component | Effort | Depends On |
|-----------|--------|-----------|
| Spike memory map configuration | 1 person-day | Spike documentation |
| Spike custom instruction framework | 2 person-days | Spike extension API |
| Tensix FPU functional model (matmul, eltwise, reduce) | 2 person-weeks | LLK math headers, ISA docs |
| Unpack functional model (format conversion) | 1.5 person-weeks | LLK unpack headers, BFP spec |
| Pack functional model (reverse conversion) | 1 person-week | LLK pack headers |
| SFPU functional model | 1 person-week | SFPI documentation |
| Trace bundle format and preparation scripts | 3 person-days | Ch 6 capture schema |
| GDB RSP server (host-side, Option C) | 2 person-weeks | GDB RSP spec |
| ebreak trap handler (device-side, Option C) | 1 person-week | RV32 trap convention |
| Host functional model (x86 LLK stubs) | 2 person-weeks | All LLK headers |
| **Total for Spike path** | **~7 person-weeks** | |
| **Total for Option C GDB stub** | **~3 person-weeks** (prototype) | |
| **Total for host functional model** | **~5 person-weeks** | |

---

## 13. PROPOSED: Implementation Roadmap

### Phase 1: Trap Handler Foundation (Weeks 1-3)

1. Write `debug_trap_handler` assembly stub for Baby RISC-V (RV32IM)
2. Add build flag `TT_METAL_DEBUG_TRAP_ENABLED` to include trap handler
   and set `mtvec`
3. Implement L1 mailbox layout at fixed address in watcher debug region
4. Test: inject `ebreak` into simple kernel, verify GPR save to mailbox,
   verify host can read saved state via MMIO

### Phase 2: Host GDB Stub (Weeks 4-6)

1. Implement GDB RSP server (TCP socket, packet parsing)
2. Map RSP commands to mailbox operations
3. Implement target description XML for Baby RISC-V
4. Test: `riscv-tt-elf-gdb -ex "target remote :1234"` with a halted TRISC

### Phase 3: Multi-RISC and Integration (Weeks 7-10)

1. Extend GDB stub for multiple threads (one per RISC)
2. Implement `Hg` (set thread) and `qfThreadInfo`/`qsThreadInfo`
3. Add "halt all RISCs on breakpoint" mode
4. Integrate with runtime interceptor: capture snapshot at breakpoint
5. Add DWARF debug info support via `TT_METAL_RISCV_DEBUG_INFO`

### Phase 4: Quasar OpenOCD (Weeks 11-12)

1. Write OpenOCD target configuration for Quasar Debug Module
2. Test standard GDB/OpenOCD workflow
3. Evaluate using OpenOCD as alternate transport for host GDB stub

---

## Key Takeaways

1. **JTAG** provides a host-accessible path to all core memory and (on
   Quasar) the Debug Module, but bandwidth is severely limited (~KB/s).
   The `JtagDevice` API is fully implemented in UMD.

2. **No RISC-V emulator** (Spike, QEMU, or riscvOVPsim) is integrated
   with tt-metal today.  Building a replay target requires a Tensix
   coprocessor plugin.  Spike is preferred over QEMU and riscvOVPsim
   for its simpler extension API and open-source license.

3. **The `trisc.cpp` entry point** reveals the minimum state needed to
   replay a kernel: L1 image, register file, config state, and the
   instruction stream.  It is not itself a host model but provides the
   exact template for a replay harness.

4. **The E-Pure/E-Model/E-Stub/E-Opaque taxonomy** classifies every LLK
   operation by emulation feasibility.  Block floating-point format
   conversion (BFP8/BFP4/BFP2) is the primary source of divergence
   between a software model and hardware.

5. **Option C (ebreak + mailbox protocol)** is the recommended approach
   for live GDB debugging on WH/BH (weighted score 3.90/5.00).
   Option A (OpenOCD + JTAG) is the long-term solution for Quasar.

6. **A three-tier debug strategy** (host model -> hardware ebreak ->
   RTL simulator) provides the best coverage-to-effort ratio.  No single
   replay target is optimal for all bug types.

7. **`TT_METAL_RISCV_DEBUG_INFO`** must be enabled during capture to
   provide source-level debugging in any replay scenario.  The capture
   bundle should always include debug-enabled ELFs.

8. **The interceptor design** from **(Ch 2, Ch 3)** aligns with the
   "capture on device, replay on host" model.  Captured LLK API traces
   plus L1/register snapshots provide the inputs needed for any tier
   of the debug strategy.

---

*Cross-references: Ch 1 (LLK API call semantics defining the replay trace
format), Ch 2 (data flow model determining what state must be captured),
Ch 3 (memory map for L1 capture buffer placement), Ch 4 (pipeline stage
model for breakpoint placement), File 1 (existing debug tools the
interceptor extends), File 2 (hardware debug primitives the replay engine
uses).*

---

**Key source files for this section:**

| File | Relevance |
|------|-----------|
| `tt_metal/third_party/umd/device/api/umd/device/jtag/jtag_device.hpp` | JtagDevice API |
| `tt_metal/third_party/umd/device/jtag/jtag_device.cpp` | JtagDevice implementation |
| `tt-llk/tests/helpers/src/trisc.cpp` | LLK test entry point / emulation template |
| `tt-llk/tests/helpers/ld/memory.blackhole.ld` | Tensix memory map |
| `tt-llk/tests/helpers/include/sfpu_stub.h` | SFPU stub pattern |
| `tt-llk/common/inc/ckernel_ops.h` | Tensix instruction opcodes |
| `tt_metal/impl/data_format/blockfloat_common.hpp` | Host-side BFP conversion |
| `tt_metal/third_party/umd/device/simulation/rtl_simulation_chip.cpp` | RTL simulation backend |
| `tt_metal/llrt/rtoptions.hpp` (line 204) | `riscv_debug_info_enabled` |
| `tt_metal/llrt/rtoptions.cpp` (lines 1110-1122, 1332-1334) | `TT_METAL_RISCV_DEBUG_INFO` handling |
| `tt_metal/hw/inc/api/debug/assert.h` (line 60) | `ebreak` in assert path |
