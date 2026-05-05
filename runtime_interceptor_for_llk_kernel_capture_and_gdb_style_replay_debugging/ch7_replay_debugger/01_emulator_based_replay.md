# Chapter 7, File 1 -- Emulator-Based Replay Architecture

> **Scope.** This file presents the architecture for replaying captured
> `.ttksnap` kernel snapshots on a host-side RISC-V emulator (Spike)
> connected to GDB via the Remote Serial Protocol (RSP).  It covers
> emulator engine selection with scored alternatives, snapshot loading
> into emulator memory, GDB integration strategies, SFPU instruction
> handling, single-RISC and intra-Tensix replay modes, the
> `tt-kernel-replay` CLI tool design, fidelity boundaries, and detailed
> terminal session examples.  All designs are **PROPOSED** unless
> explicitly noted as existing infrastructure.
>
> All source citations reference commit `621b949`.

**Cross-References:**
- Ch 2 (state inventory: regfile[64], config registers, L1, CBs, semaphores)
- Ch 4 (5-RISC execution model, dispatch protocol, CB synchronization)
- Ch 5 File 3 (Spike/QEMU assessment, E-taxonomy: E-Pure/E-Model/E-Stub/E-Opaque, three-tier debug strategy)
- Ch 6 File 3 (`.ttksnap` FlatBuffer schema, `KernelInvocationSnapshot` root type, `file_identifier "TKSN"`)
- Ch 8 (implementation phases -- forward reference)

---

## 1. Replay Pipeline Overview

The emulator-based replay pipeline consists of four stages:

```
.ttksnap file  -->  Snapshot Loader  -->  Emulator Engine  -->  GDB Stub  -->  Debugger Frontend
  (Ch 6, F3)         (this chapter)       (this chapter)       (this chapter)    (GDB / VS Code)
```

The snapshot loader deserializes the `.ttksnap` FlatBuffer (Ch 6, File 3)
into an in-memory representation: ELF binaries, L1 memory image, CB
configurations, semaphore values, runtime arguments, and metadata.  The
emulator engine executes the RISC-V binary against a memory model
initialized from the snapshot.  The GDB stub translates GDB Remote Serial
Protocol (RSP) packets into emulator control operations.  The debugger
frontend provides the developer-facing breakpoint/step/inspect interface.

Each stage admits multiple implementation strategies.  The sections below
present scoring matrices for the key decisions.

---

## 2. Emulator Engine Selection

### 2.1 Candidates

Three emulator engines are viable for RISC-V RV32IMC execution:

| Property | Spike | QEMU (system mode) | Custom Functional Model |
|----------|-------|---------------------|------------------------|
| Origin | UC Berkeley reference ISA sim | FOSS full-system emulator | Purpose-built for Tensix |
| ISA coverage | RV32/64IMACFDV + extensions | RV32/64IMACFDV + extensions | RV32IMC + SFPI stubs |
| GDB stub | Via `--rbb-port` + OpenOCD | Built-in (`-gdb tcp::PORT`) | Must implement from scratch |
| Speed (MIPS) | ~10-50 MIPS | ~100-500 MIPS (TCG) | ~5-20 MIPS (interpreted) |
| Custom CSR support | Plugin API (`extension_t`) | TCG helper functions | Native |
| MMIO device model | htif + custom device tree | Full MMIO framework (QDev) | Native address decode |
| Build complexity | Moderate (autotools) | High (Meson, ~2M LOC) | Low (self-contained) |
| License | BSD-3-Clause | GPL-2.0 (linking concern) | Proprietary-compatible |

### 2.2 Scoring Matrix

Criteria are weighted by importance to the replay use case (1 = low, 5 = critical):

| Criterion | Weight | Spike | QEMU | Custom Model |
|-----------|--------|-------|------|--------------|
| GDB integration maturity | 5 | 4 (via OpenOCD bridge) | 5 (mature, multi-arch) | 1 (must build) |
| Memory model flexibility | 4 | 3 (htif + plugins) | 4 (QDev MMIO) | 5 (full control) |
| SFPI extension potential | 5 | 3 (extension_t API) | 2 (TCG codegen complex) | 5 (native) |
| Build/integration effort | 3 | 4 (small codebase) | 2 (massive codebase) | 3 (moderate effort) |
| Execution speed | 2 | 3 | 5 | 2 |
| License compatibility | 3 | 5 (BSD) | 2 (GPL-2.0) | 5 (proprietary OK) |
| **Weighted total** | | **87** | **80** | **85** |

### 2.3 Analysis

**Spike** scores highest because it provides a clean extension API for
custom instructions, a BSD-3-Clause license that avoids GPL linking
concerns, and sufficient performance for single-TRISC debugging.  The
`extension_t` interface allows registering custom CSRs and instructions,
which maps directly to modeling Tensix configuration registers at
`0xFFB10000` (see `ckernel::reg_base` in
`tt_metal/hw/firmware/src/tt-1xx/trisc.cc:58`).

**Critical note on Spike's GDB interface:** Spike's `--rbb-port` flag
exposes a Remote BitBang JTAG interface, NOT a direct GDB RSP stub.
Connecting GDB requires an intermediate OpenOCD bridge:

```bash
# Terminal 1: Spike with JTAG bitbang interface
spike --rbb-port=9824 --isa=rv32imc --priv=m \
  -m0x0:0x180000 -m0xFFB00000:0x100000 -m0xFFE00000:0x200000 \
  replay_trisc1.elf

# Terminal 2: OpenOCD bridge (JTAG -> GDB RSP)
openocd -c "adapter driver remote_bitbang; transport select jtag; \
  remote_bitbang host localhost; remote_bitbang port 9824" \
  -f target/riscv.cfg

# Terminal 3: GDB connects to OpenOCD's RSP port
riscv-tt-elf-gdb replay_trisc1.elf \
  -ex "target remote localhost:3333"
```

Alternatively, the `tt-kernel-replay` tool can implement a direct RSP
server that wraps Spike's `sim_t` API, eliminating the OpenOCD dependency
at the cost of building against Spike's internal (unstable) API.

**QEMU** wins on raw execution speed (5-10x faster than Spike) but its
GPL-2.0 license creates distribution concerns and adding custom RISC-V
instructions requires modifying the TCG translation layer.

**Custom functional model** provides full control but requires building a
GDB RSP stub from scratch (~2-3 person-weeks).  This option becomes
attractive only if SFPI modeling fidelity is a hard requirement.

> **PROPOSED: Recommendation.** Use Spike as the primary emulator for
> Phase 3 (Ch 8), with a custom `extension_t` plugin modeling Tensix MMIO
> registers and CB semaphore semantics.  Reserve the custom functional model
> as a Phase 5 option if SFPI fidelity becomes a requirement.

### 2.4 Spike Extension Architecture

The Spike extension API provides two hooks relevant to Tensix modeling:

```cpp
// PROPOSED: tensix_extension.h
class tensix_extension_t : public extension_t {
public:
    // Called for loads/stores to the Tensix register space
    bool load(reg_t addr, size_t len, uint8_t* bytes) override;
    bool store(reg_t addr, size_t len, const uint8_t* bytes) override;

    // Called for unrecognized instructions (SFPI opcodes)
    std::vector<insn_desc_t> get_instructions() override;

private:
    // Models ckernel::reg_base (0xFFB10000)
    uint32_t tensix_cfg_regs_[256];
    // Models ckernel::regfile (REGFILE_BASE)
    uint32_t regfile_[64];
    // Models ckernel::pc_buf_base (PC_BUF_BASE)
    uint32_t pc_buf_[256];
    // CB semaphore counters
    uint32_t tiles_received_[NUM_CIRCULAR_BUFFERS];
    uint32_t tiles_acked_[NUM_CIRCULAR_BUFFERS];
};
```

This mirrors the hardware register layout documented in **(Ch 2, File 4)**
and the `ckernel::regfile[64]` initialization loop in
`tt_metal/hw/firmware/src/tt-1xx/trisc.cc:114-117`:

```cpp
// trisc.cc:114-117 -- GPR initialization at boot
#pragma GCC unroll 0
for (int i = 0; i < 64; i++) {
    regfile[i] = 0;
}
```

### 2.5 Spike Invocation for Tensix Kernel Replay

Tensix TRISCs are RV32IMC cores (Wormhole/Blackhole) or RV32IMCF cores
(Quasar with additional float support).  The Spike invocation must match
the ISA configuration used by the TT-Metal RISC-V GCC toolchain:

```bash
# PROPOSED: Spike invocation for a single TRISC (e.g., TRISC1 = math thread)
spike \
  --isa=rv32imc \
  --priv=m \
  --pc=0x0 \
  --hartids=0 \
  -m0x0:0x180000 \
  -m0xFFB00000:0x100000 \
  -m0xFFE00000:0x200000 \
  --rbb-port=9824 \
  replay_trisc1.elf
```

| Flag | Purpose |
|------|---------|
| `--isa=rv32imc` | Matches TRISC ISA: integer, multiply, compressed |
| `--priv=m` | Machine-mode only (TRISCs run bare-metal) |
| `-m0x0:0x180000` | L1 SRAM region: 1.5 MB for BH (`MEM_L1_SIZE` = `1536 * 1024` in `dev_mem_map.h:33`) |
| `-m0xFFB00000:0x100000` | MMIO region covering local memory, debug regs (`MEM_LOCAL_BASE` = `0xFFB00000`) |
| `-m0xFFE00000:0x200000` | Tensix regfile, instruction buffer, PC buffer, config registers |
| `--rbb-port=9824` | Remote BitBang JTAG interface (requires OpenOCD bridge for GDB) |

For Quasar targets with 4 MB L1 (`MEM_L1_SIZE` = `4 * 1024 * 1024` in
`tt-2xx/quasar/dev_mem_map.h:33`), the L1 region becomes `-m0x0:0x400000`.

---

## 3. GDB Integration Architecture

### 3.1 Candidates

Three strategies for connecting GDB to the emulated Tensix core(s):

| Strategy | Description | Multi-RISC support |
|----------|-------------|-------------------|
| **A: Separate ports** | One GDB stub per RISC-V core, each on its own TCP port | User runs 5 separate GDB sessions |
| **B: GDB multi-inferior** | Single GDB session with 5 inferiors sharing one connection | `inferior 1` = BRISC, `inferior 3` = TRISC1, etc. |
| **C: Integrated frontend** | Custom debug adapter (DAP) translating between VS Code and multiple stubs | Single unified UI |

### 3.2 Scoring Matrix

| Criterion | Weight | A: Separate Ports | B: Multi-Inferior | C: Integrated DAP |
|-----------|--------|-------------------|-------------------|--------------------|
| Implementation effort | 4 | 5 (trivial) | 3 (multi-inferior protocol) | 1 (full DAP server) |
| Single-TRISC UX | 5 | 5 (one port, one GDB) | 4 (must select inferior) | 5 (transparent) |
| Multi-RISC UX | 3 | 2 (5 terminals) | 5 (switch inferiors) | 5 (tabbed views) |
| Existing tooling reuse | 4 | 5 (standard GDB) | 4 (GDB built-in) | 2 (custom code) |
| Spike compatibility | 4 | 5 (native) | 3 (needs mux layer) | 2 (needs adapter) |
| **Weighted total** | | **92** | **77** | **63** |

### 3.3 Analysis

**Separate ports (A)** wins decisively for initial implementation.  For
the most common debugging scenario -- debugging a single TRISC (typically
TRISC1 for math bugs) -- the developer connects a single GDB session to
a single port.  For multi-RISC scenarios, GDB's multi-inferior protocol
(Strategy B) is the Phase 4 follow-up.  VS Code DAP (Strategy C) is
Phase 5 UX polish.

> **PROPOSED: Phased approach.** Phase 3: separate ports (A), one RISC at
> a time.  Phase 4: multi-inferior (B) with a thin mux layer.  Phase 5:
> VS Code DAP adapter (C).

### 3.4 GDB RSP Protocol Mapping

The RSP protocol uses ASCII packets over TCP.  Key packets for replay:

| RSP Packet | Meaning | Replay Implementation |
|-----------|---------|----------------------|
| `g` | Read all registers | Return RISC-V x0-x31 + PC from emulator state |
| `G XX...` | Write all registers | Set emulator register state |
| `m addr,length` | Read memory | Read from emulator memory model |
| `M addr,length:XX` | Write memory | Write to emulator memory model |
| `s` | Single step | Execute one RISC-V instruction |
| `c` | Continue | Run until breakpoint or completion |
| `Z0,addr,kind` | Set SW breakpoint | Insert `ebreak` at addr in emulator |
| `z0,addr,kind` | Remove SW breakpoint | Restore original instruction |
| `?` | Halt reason | Return signal (SIGTRAP for breakpoint) |
| `qSupported` | Feature query | Advertise `PacketSize`, memory map |

The `riscv-tt-elf-gdb` binary shipped with TT-Metal's RISC-V toolchain
(at `runtime/sfpi/compiler/bin/riscv-tt-elf-gdb`) is the intended GDB
frontend.  It already understands the RV32IMC ISA and can load DWARF
debug info from JIT-compiled kernel ELFs.

---

## 4. Memory Model Alternatives

### 4.1 The Problem

A Tensix core's address space includes several distinct regions that must
be modeled during replay (see **(Ch 2, File 3)** for full layout):

| Region | Address Range (BH) | Size | Access Pattern |
|--------|-------------------|------|---------------|
| L1 SRAM | `0x00000000`-`0x0017FFFF` | 1.5 MB | Load/store by all 5 RISCs |
| Tensix config regs | `0xFFB10000`+ | Variable | MMIO by TRISCs |
| Regfile | `REGFILE_BASE` (`0xFFE00000`) | 256 B (64 x 32b) | MMIO by TRISCs |
| PC buffer | `PC_BUF_BASE` (`0xFFE80000`) | Variable | Semaphore + PC tracking |
| Instruction buffer | `INSTRN_BUF_BASE` (`0xFFE40000`) | Variable | Tensix coprocessor queue |
| NOC registers | `0xFFB20000`+ | ~4 KB | MMIO by BRISC/NCRISC |

### 4.2 Scoring Matrix

| Criterion | Weight | Flat Array | Segmented+Traps | Full Device Model |
|-----------|--------|-----------|-----------------|-------------------|
| Implementation effort | 4 | 5 | 3 | 1 |
| Compute kernel fidelity | 5 | 2 (config writes lost) | 4 (config captured) | 5 (bit-accurate) |
| Data movement fidelity | 3 | 1 (no NOC) | 2 (NOC stubbed) | 4 (NOC modeled) |
| GDB inspect accuracy | 5 | 3 (L1 OK, regs wrong) | 5 (all readable) | 5 (all readable) |
| Emulator integration | 3 | 5 (trivial) | 3 (Spike plugin) | 1 (custom emulator) |
| **Weighted total** | | **65** | **74** | **66** |

**Segmented with traps** is the clear winner.  L1 is modeled as a flat
byte array initialized from the `.ttksnap` snapshot.  Config registers at
`0xFFB10000` and the regfile at `REGFILE_BASE` are trapped in the Spike
extension plugin and stored in shadow arrays.  NOC operations are stubbed
because TRISCs never issue NOC transactions (Ch 4, File 3).

---

## 5. SFPU Instruction Handling

### 5.1 Scoring Matrix

| Criterion | Weight | Trap+Skip | Functional Stubs | Bit-Exact Model |
|-----------|--------|-----------|-----------------|-----------------|
| Control-flow debugging | 5 | 3 (flow OK, values wrong) | 5 (values close) | 5 (values exact) |
| Numerical debugging | 4 | 1 (meaningless) | 3 (IEEE vs BF16) | 5 (bit-accurate) |
| Implementation effort | 4 | 5 | 3 | 1 |
| Maintenance burden | 3 | 5 (none) | 3 (track SFPI changes) | 1 (RTL sync) |
| **Weighted total** | | **56** | **65** | **52** |

**Functional stubs** provide the best balance.  For the majority of
LLK debugging scenarios, approximate SFPU values are sufficient.

> **PROPOSED:** Phase 3 implements trap-and-skip with warnings.  Phase 4
> adds functional stubs for the 15 most common SFPI operations
> (`loadi`, `mad`, `add`, `mul`, `exp`, `recip`, `sqrt`).

---

## 6. Snapshot Loading: From `.ttksnap` to Emulator State

### 6.1 Loading Sequence

```
1. Detect compression (LZ4 frame magic 0x04224D18, zstd magic 0xFD2FB528)
2. Decompress if needed
3. Verify FlatBuffer + file identifier "TKSN"
4. Validate schema_version_major == 1
5. For each RISC to replay:
   a. Load ELF binary into emulator instruction memory
   b. Initialize L1 memory from L1RegionSnapshot entries
   c. Initialize CB pointers from CircularBufferSnapshot entries
   d. Set semaphore values from SemaphoreSnapshot entries
   e. Write runtime args to correct L1 offsets
   f. Initialize regfile[0..63] to zero (matching trisc.cc:114-117)
   g. Set PC to kernel entry point
   h. Configure Spike memory regions and extension plugin
6. Start GDB stub on specified port
7. Wait for GDB connection
```

### 6.2 L1 Population Order

L1 is populated in layers, with later writes overlaying earlier data:

1. Full L1 dump (if present in `L1RegionSnapshot`) -- base layer
2. Kernel binaries at `text_addr` from `KernelBinaryBlob`
3. CB data contents at `l1_address` from `CircularBufferSnapshot`
4. Semaphore values at resolved addresses from `SemaphoreSnapshot`
5. Runtime args at `rta_offset` from `ProgramConfigSnapshot`

### 6.3 ELF Loading and Kernel Entry

TT-Metal JIT-compiled kernels use a custom linker script placing `.text`
at the `kernel_text_offset` from `launch_msg_t`.  The actual firmware
does NOT call `run_kernel(__runtime_args_start)` directly.  Instead, it
reads `launch_msg`, computes `kernel_lma` from `kernel_text_offset`, and
calls via function pointer:

```cpp
// trisc.cc:214-215 -- actual kernel invocation
uint32_t kernel_lma = (kernel_config_base +
    launch_msg->kernel_config.kernel_text_offset[index]);
auto stack_free = reinterpret_cast<uint32_t (*)()>(kernel_lma)();
```

Runtime arguments are accessed via `rta_l1_base` derived from the launch
message, not passed as a function argument.  The regfile initialization
(`for (int i = 0; i < 64; i++) regfile[i] = 0;`) happens once before
the firmware's `while(1)` loop, not per-kernel-invocation.

The snapshot records the `kernel_text_offset` (see **(Ch 6, File 3)**,
`KernelBinaryBlob.base_address`), and the loader must instruct Spike
to map the ELF at the correct virtual address for GDB source-level
debugging to work with the DWARF info.

### 6.4 DWARF Symbol Availability

The `TT_METAL_RISCV_DEBUG_INFO` environment variable controls whether
kernel ELFs are compiled with the `-g` flag.  It is processed in
`tt_metal/llrt/rtoptions.cpp` lines 1110-1122.  When enabled, the
`riscv-tt-elf-g++` compiler (see `tt_metal/hw/CMakeLists.txt` line 108)
produces ELFs with full DWARF sections enabling source-level display,
variable inspection, and stack traces.

**PROPOSED:** For replay debugging, the snapshot capture should always
set `TT_METAL_RISCV_DEBUG_INFO=1`.  If the captured binary lacks debug
info, the replay tool warns the user.

### 6.5 Single-RISC vs. Multi-RISC Loading

For **single-TRISC replay** (the most common case), the loader:
- Loads only the target TRISC's ELF binary
- Initializes L1 with full snapshot (all CBs populated)
- Pre-sets all semaphores to values that allow the target TRISC to
  proceed without blocking (see Section 7)
- Does NOT load BRISC/NCRISC -- their data movement has already been
  captured as L1 contents

For **intra-Tensix 5-RISC replay**, the loader spawns 5 Spike instances
sharing a common L1 memory model.  See **(File 3)** for multi-core
coordination alternatives.

---

## 7. Single-RISC Replay with Pre-Populated Sync State

### 7.1 Rationale

The most valuable replay target is TRISC1 (the math processor).  Math
kernel bugs -- incorrect accumulation, wrong data format handling,
fidelity mismatches -- account for the majority of LLK debugging sessions.

### 7.2 Pre-Populating Synchronization State

**PROPOSED:** The snapshot loader computes "fast-forward" semaphore and CB
values:

```
For each semaphore S that TRISC1 waits on:
    Set S.value = S.threshold  (barrier already passed)

For each input CB that TRISC1 reads:
    Set CB.tiles_received = total expected tiles
    (tiles are already available for consumption)

For each output CB that TRISC1 writes:
    Set CB.tiles_acked = 0  (output buffer is empty, room available)
```

This eliminates all synchronization waits without modifying the kernel
binary.  The kernel executes its math operations on the tile data present
in L1, producing results into the output CB region.

### 7.3 Capabilities of Single-RISC Replay

| Capability | Available? |
|-----------|-----------|
| Source-level stepping through math kernel | Yes (with DWARF) |
| Inspecting local variables and call stack | Yes |
| Setting breakpoints on LLK API calls | Yes |
| Inspecting tile data in L1 (input CBs) | Yes |
| Inspecting computed results (output CBs) | Yes |
| Verifying control flow (which LLK path taken) | Yes |
| Comparing register values at specific points | Yes |

---

## 8. PROPOSED: `tt-kernel-replay` CLI Tool

### 8.1 Subcommand Overview

```
tt-kernel-replay <subcommand> [options]

SUBCOMMANDS:
    inspect     Display snapshot metadata, kernel list, core map, CB layout
    emulate     Launch RISC-V emulator with captured state and GDB port
    host-model  Compile and run kernel in host functional model (see File 02)
    hw-replay   Replay on real hardware via debug registers (see File 02)
    validate    Compare replay output against captured golden data
    export      Extract snapshot contents (ELFs, L1 dumps, CB data) to files

GLOBAL OPTIONS:
    --snapshot <path>      Path to .ttksnap file (required for all)
    --verbose              Enable detailed diagnostic output
    --color <auto|on|off>  Terminal color output (default: auto)
    --version              Print tool version and schema compatibility
```

### 8.2 `emulate` Subcommand

```
tt-kernel-replay emulate <snapshot.ttksnap> [options]

OPTIONS:
    --core <risc>       brisc, ncrisc, trisc0, trisc1, trisc2, all
                        Default: trisc1 (most common for LLK math debugging)
    --gdb-port <port>   GDB RSP port (default: 1234; with --core all,
                        ports 1234-1238 are used)
    --core-coord <x,y>  Core coordinate if snapshot contains multiple cores
                        (default: first core in snapshot)
    --emulator <name>   Emulator backend: spike|qemu
                        (default: spike)
    --fidelity <level>  L0 (RISC-V only), L1 (+instruction decode), L2 (+reg arrays)
    --sfpu-mode <mode>  SFPI handling: skip|stub|exact (default: stub)
    --break-at-entry    Set initial breakpoint at kernel entry (default: on)
    --timeout <sec>     Kill emulator after N seconds (default: 300)
    --spike-path <path> Custom Spike binary location
    --log-noc           Log all NOC-mapped memory accesses to stderr
```

### 8.3 Internal Architecture

```
                     +-----------------------+
                     |  tt-kernel-replay     |
                     |  (host process)       |
                     +-----------+-----------+
                                 |
                +----------------+----------------+
                |                                  |
      +---------v---------+            +----------v----------+
      | Snapshot Loader   |            | Emulator Manager    |
      | - FlatBuffer      |            | - Spike process(es) |
      |   deserializer    |            | - Memory model init |
      | - ELF extractor   |            | - Extension plugin  |
      | - L1 image builder|            | - GDB stub config   |
      +-------------------+            +---------------------+
```

---

## 9. PROPOSED: Terminal Session -- Debugging a Matmul Assert

A matmul assert on core (3,4) was captured via `on-assert` mode (Ch 6,
File 2):

```
$ tt-kernel-replay inspect 000042_prog17_core3_4.ttksnap

=== Kernel Invocation Snapshot ===
  Architecture:   WORMHOLE_B0    TT-Metal: 621b949
  Trigger:        failure:assert (Watcher detected)
  Program ID:     17             Invocation: 42

  Kernels:
    [0] reader_bmm_8bank (BRISC)        hash=0x1A3F...
    [1] bmm_large_block  (TRISC0/1/2)   hash=0x7C02...
    [2] writer_bmm_8bank (NCRISC)       hash=0x55E1...

  Core (3,4):
    Binaries: BRISC=16KB, NCRISC=14KB, TRISC0=32KB, TRISC1=24KB, TRISC2=8KB
    CBs: 3 active (CB0: BF16 8 tiles, CB1: BF16 8 tiles, CB16: BF16 8 tiles)
    Semaphores: 4 active (IDs 0-3)
    L1 captured: Yes (sparse, 487 KB)
  Snapshot size: 612 KB (compressed from 1.4 MB)

$ tt-kernel-replay emulate 000042_prog17_core3_4.ttksnap \
    --core trisc1 --gdb-port 9824

tt-kernel-replay: Populating L1 from sparse dump (487 KB)
tt-kernel-replay: Writing TRISC1 binary (24576 bytes) at 0x0000B000
tt-kernel-replay: Pre-satisfying semaphores for single-RISC isolation
tt-kernel-replay:   sem[0] at 0x1FFF0 = 1 (TRISC0 unpack complete)
tt-kernel-replay:   sem[1] at 0x1FFF4 = 0 (dest available)
tt-kernel-replay: Launching Spike (rv32imc, Tensix extension, fidelity=L1)
tt-kernel-replay: GDB RSP listening on port 9824
```

In a second terminal:

```
$ riscv-tt-elf-gdb trisc1.elf \
    -ex "set architecture riscv:rv32" \
    -ex "target remote :9824"
0x0000b000 in _start ()

(gdb) break run_kernel
Breakpoint 1 at 0xb1a0: file bmm_large_block.cpp, line 42.
(gdb) continue
Breakpoint 1, run_kernel (args=0x20000) at bmm_large_block.cpp:42

(gdb) break llk_math_matmul
Breakpoint 2 at 0xb3c8: file llk_math_matmul.h, line 127.
(gdb) continue
Breakpoint 2, llk_math_matmul<...>() at llk_math_matmul.h:127

(gdb) info locals
dst_index = 0
num_faces = 4
face_r_dim = 16

(gdb) x/8xw 0x20000
0x20000: 0x00000003  0x00000004  0x00000008  0x00000008

(gdb) step
128     for (uint32_t face = 0; face < num_faces; face++) {
```

---

## 10. Error Messages and Diagnostics

| Condition | Message |
|-----------|---------|
| Schema mismatch | `ERROR: Incompatible snapshot schema version. Snapshot: 2.0, Tool: 1.3 (supports 1.x). Action: Update tt-kernel-replay or re-capture.` |
| Architecture warning | `WARNING: Snapshot captured on BLACKHOLE (L1=1572864). Spike configured for BLACKHOLE layout.` |
| Missing debug symbols | `WARNING: No debug symbols in TRISC1 binary. Source-level debug unavailable. Set TT_METAL_RISCV_DEBUG_INFO=1 before workload. Assembly-level debug still works.` |
| Corrupt snapshot | `ERROR: FlatBuffer verification failed at offset 12044: invalid table offset. Re-capture the kernel invocation.` |
| Emulator timeout | `ERROR: Emulator timed out after 10s without GDB connection. Connect with: riscv-tt-elf-gdb <elf> -ex "target remote :1234"` |
| Tensix instruction trap | `ERROR: Illegal instruction at PC 0x000103b8. This is likely a Tensix custom instruction (.ttinsn). Ensure the Tensix extension plugin is loaded.` |

---

## 11. Replay Fidelity Across Emulator Levels

Building on the E-taxonomy from Ch 5, File 3, Section 4:

| Bug Type | L0 (RISC-V only) | L1 (+instr decode) | L2 (+reg arrays) | HW Replay |
|----------|:-:|:-:|:-:|:-:|
| Wrong control flow / branch | Yes | Yes | Yes | Yes |
| Incorrect runtime arg usage | Yes | Yes | Yes | Yes |
| Off-by-one in loop bounds | Yes | Yes | Yes | Yes |
| Wrong CB index in LLK call | No | Yes (logged) | Yes | Yes |
| FPU rounding divergence | No | No | Approximate | Yes |
| BFP format conversion error | No | No | If modeled | Yes |
| Pack/unpack data corruption | No | No | Yes | Yes |
| Semaphore protocol violation | No | No | Yes | Yes |
| Timing-dependent hang | No | No | No | No (need simulator) |

**L1 is the recommended default** -- it catches most control flow and
instruction sequence bugs without requiring a full Tensix FPU model.

### 11.1 Fidelity Summary (E-Taxonomy Mapping)

| Aspect | Fidelity | E-Class | Impact on Debugging |
|--------|----------|---------|-------------------|
| Scalar RISC-V code | Exact | E-Pure | Full source-level debug |
| LLK API control flow | Exact | E-Pure | Correct branching/looping |
| L1 memory contents | Exact (from snapshot) | E-Pure | Correct data inspection |
| Semaphore logic | Functional | E-Stub | Correct sync decisions |
| CB pointer math | Functional | E-Stub | Correct flow control |
| SFPU math | Near-exact | E-Model | 1-2 ULP potential difference |
| FPU format conversion | Functional | E-Model | BFP formats need custom model |
| NOC data movement | Absent | E-Opaque | Cannot debug NOC issues |
| Cycle timing | Absent | E-Opaque | Cannot debug perf issues |
| Multi-core interaction | Absent (single) | E-Opaque | Need multi-core replay |

What is lost in all emulation levels (E-Opaque from Ch 5, File 3):
NOC transaction timing, L1 bank contention, Tensix pipeline stages,
hardware semaphore atomics, L1 data cache (BH only).

---

## 12. Integration with the Three-Tier Debug Strategy

Emulator-based replay corresponds to **Tier 1** in the three-tier debug
strategy defined in Ch 5, File 3:

```
Tier 1: Host / Emulator Replay
  - Fastest iteration (~seconds to load and attach GDB)
  - No hardware required
  - Covers: logic errors, control flow bugs, data format issues
  - Escalate to Tier 2 when: emulator output matches expected,
    but hardware diverges

Tier 2: On-Hardware Debug (ebreak + mailbox on WH; debug registers on BH/Quasar)
  - Requires hardware
  - Covers: hardware-specific behavior, timing-sensitive bugs
  - See File 02 of this chapter

Tier 3: Cycle-Accurate Simulation / RTL Waveform
  - Slowest, highest fidelity
  - Covers: pipeline stalls, NOC congestion, race conditions
```

---

## 13. PROPOSED: VS Code Debug Adapter Integration

```
VS Code UI -> DAP over stdio -> tt-kernel-replay-dap -> GDB RSP -> Spike
```

The adapter spawns `tt-kernel-replay emulate` as a child process,
connects to Spike's GDB port, and translates DAP requests
(setBreakpoints, continue, stepIn, evaluate) to GDB RSP commands.

Custom Tensix views:

| DAP Extension | Purpose |
|--------------|---------|
| `tt/cbViewer` | Render CB contents as tile grid with data format decoding |
| `tt/l1Map` | Visual L1 memory map with region annotations |
| `tt/instrLog` | Tensix instruction log (L1+ fidelity) |
| `tt/semaphores` | Semaphore state panel with current values |

This is a Phase 5 deliverable following CLI stabilization.

---

## Key Source Files

| File | Path (relative to TT-Metal root) | Lines | Relevance |
|------|----------------------------------|-------|-----------|
| `tensix.h` (BH) | `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix.h` | 47, 56, 62, 73, 132 | REGFILE_BASE, INSTRN_BUF_BASE, PC_BUF_BASE, CFG_BASE, DEBUG_BASE |
| `dev_mem_map.h` (BH) | `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h` | 33, 42, 45 | MEM_L1_SIZE, MEM_LOCAL_BASE, MEM_TRISC_LOCAL_SIZE |
| `dev_mem_map.h` (Quasar) | `tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h` | 33, 41 | Quasar L1 size, MEM_LOCAL_BASE |
| `trisc.cc` | `tt_metal/hw/firmware/src/tt-1xx/trisc.cc` | 58, 60, 114-117, 214-215 | reg_base, regfile init, kernel_lma invocation |
| `trisc.cpp` | `tt_metal/third_party/tt_llk/tests/helpers/src/trisc.cpp` | 36-77 | Host-side TRISC entry point, regfile zeroing, run_kernel |
| `rtoptions.cpp` | `tt_metal/llrt/rtoptions.cpp` | 1110-1122 | TT_METAL_RISCV_DEBUG_INFO flag processing |
| `CMakeLists.txt` (hw) | `tt_metal/hw/CMakeLists.txt` | 108 | riscv-tt-elf-g++ compiler path |
| `tensix_functions.h` | `tt_metal/hw/inc/internal/tensix_functions.h` | 26, 398, 606-687 | ex_push_insn, semaphore init, breakpoint APIs |
| `ckernel.h` | `tt_metal/third_party/tt_llk/.../ckernel.h` | 64, 152-155 | KERNEL_COMPLETE, tensix_sync() |

---

## Key Takeaways

- **Spike is the recommended emulator engine** for Phase 3: it provides a
  clean extension API for Tensix-specific MMIO modeling, a BSD-3-Clause
  license, and sufficient performance for single-TRISC debugging.  Note
  that Spike's `--rbb-port` requires an OpenOCD bridge for GDB; a direct
  RSP server wrapping Spike's `sim_t` API is the preferred alternative.

- **Separate GDB ports per RISC** is the simplest and highest-scoring GDB
  integration strategy for initial implementation.

- **Segmented memory with MMIO traps** correctly models the three dominant
  TRISC access patterns (L1 loads/stores, config register writes, regfile
  operations) while keeping the implementation tractable.

- **SFPU handling follows a progressive strategy**: trap-and-skip first,
  functional stubs for common operations next, bit-exact model only if
  numerical precision debugging is a validated requirement.

- **Single-TRISC replay with pre-populated CB/semaphore state** is the
  highest-value, lowest-effort debug target, covering ~70% of practical
  LLK debugging scenarios with ~5 person-weeks of implementation effort.

- **Three fidelity levels (L0/L1/L2)** provide an escalation path from
  scalar-only debugging through instruction sequence verification to
  full register array inspection.  L1 is the recommended default.
