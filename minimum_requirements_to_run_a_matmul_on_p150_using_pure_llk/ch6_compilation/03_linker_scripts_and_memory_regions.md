# 6.3 Linker Scripts and Memory Regions

The linker scripts define how compiled kernel code maps into the physical memory of a Blackhole Tensix core. They are the bridge between the compiler's abstract sections (`.text`, `.data`, `.bss`) and the concrete L1 SRAM addresses where `tt-exalens` will load the ELF. Getting this wrong produces an ELF that loads silently but executes garbage -- one of the hardest classes of bugs to diagnose on bare metal.

This section documents the complete linker script chain, the memory map it defines, and the startup assembly that runs before `main()`.

---

### 6.3.1 The Three-Script Chain

Every compilation passes three `-T` linker scripts in a specific order:

```
-T memory.blackhole.ld    # Step 1: Define MEMORY{} blocks with absolute addresses
-T <thread>.ld            # Step 2: Create REGION_* aliases pointing to this thread's regions
-T sections.ld            # Step 3: Place sections into REGION_* aliases
```

| Layer | File | Role |
|:------|:-----|:-----|
| 1. Memory layout | `memory.blackhole.ld` | Defines all physical memory regions (L1 code/data, L0 private memory, stack) for BRISC and all three TRISCs. Architecture-specific. |
| 2. Thread alias | `unpack.ld`, `math.ld`, `pack.ld`, `brisc.ld` | Maps generic region names (`REGION_CODE`, `REGION_DATA`, etc.) to the specific core's regions from layer 1. Thread-specific. |
| 3. Section rules | `sections.ld` | Defines how compiler-generated sections (`.text`, `.rodata`, `.bss`) are placed into the generic region names. Shared across all threads and architectures. |

This layering achieves a key design goal: `sections.ld` is architecture-agnostic and thread-agnostic. It always writes to `REGION_CODE`, `REGION_DATA`, `REGION_STACK`, etc. The per-thread alias script is the only file that changes between the unpack, math, pack, and BRISC compilations. The memory layout script is the only file that changes between architectures (Blackhole vs. Wormhole). Adding a new architecture requires only a new `memory.*.ld` file -- the alias scripts and section placement rules remain unchanged.

All linker scripts are located at `tests/helpers/ld/`.

---

### 6.3.2 Layer 1: `memory.blackhole.ld` -- Physical Memory Map

This script defines the complete address space layout for all cores on a Blackhole Tensix tile. A key design feature is that L1 regions are defined using **relational expressions** (`ORIGIN(prev) + LENGTH(prev)`), so changing one region's size automatically propagates to all subsequent regions. The constants and one representative TRISC region:

```ld
LOCAL_MEM_BASE = 0xFFB00000;
BRISC_LOCAL_MEM_LENGTH = 8K;
TRISC_LOCAL_MEM_LENGTH = 4K;
NCRISC_LOCAL_MEM_LENGTH = 8K;
L1_BASE = 0x0;
RUNTIME_ARGS_START = 0x20000;
UNDEFINED = 0;

MEMORY {
    /* L0 private memory (same addresses, different physical SRAM per core) */
    TRISC0_LOCAL_DATA_MEM : ORIGIN = LOCAL_MEM_BASE, LENGTH = 3584
    TRISC0_STACK_MEM      : ORIGIN = ORIGIN(TRISC0_LOCAL_DATA_MEM)
                            + LENGTH(TRISC0_LOCAL_DATA_MEM),
                            LENGTH = TRISC_LOCAL_MEM_LENGTH
                            - LENGTH(TRISC0_LOCAL_DATA_MEM)
    /* ... TRISC1, TRISC2, BRISC similarly ... */

    /* L1 shared memory */
    BRISC_CODE            : ORIGIN = L1_BASE, LENGTH = 16K
    BRISC_LOADER_INIT_MEM : ORIGIN = ORIGIN(BRISC_CODE) + LENGTH(BRISC_CODE),
                            LENGTH = BRISC_LOCAL_MEM_LENGTH
    TRISC0_CODE           : ORIGIN = ORIGIN(BRISC_LOADER_INIT_MEM)
                            + LENGTH(BRISC_LOADER_INIT_MEM), LENGTH = 16K
    /* ... TRISC0_LOADER_INIT, TRISC1_CODE, etc. chain similarly ... */

    RUNTIME_ARGS          : ORIGIN = RUNTIME_ARGS_START, LENGTH = 1K

    /* Zero-length (unused on Blackhole): *_LOADER_CODE_MEM, *_GCOV_MEM */
}
PROVIDE(__instrn_buffer = 0xFFE40000);
```

#### Address Constants

| Symbol | Value | Meaning |
|:-------|:------|:--------|
| `LOCAL_MEM_BASE` | `0xFFB00000` | Start of each core's private L0 memory. This is a core-local address: each RISC-V core sees its own private SRAM at this address. Not accessible via NOC. |
| `L1_BASE` | `0x0` | Start of shared L1 memory. Visible to all cores and accessible via NOC from the host (for ELF loading). |
| `RUNTIME_ARGS_START` | `0x20000` | L1 address where the host writes the `RuntimeParams` struct data. |
| `NCRISC_LOCAL_MEM_LENGTH` | `8K` | NCRISC gets 8 KiB (unused by LLK). |
| `UNDEFINED` | `0` | Placeholder for regions that do not exist on Blackhole (IRAM loader code, GCOV). |
| `__instrn_buffer` | `0xFFE40000` | Replay buffer base address (see Section 6.3.5). |

#### L1 Memory Map (Computed Addresses)

Computing the absolute addresses from the relational expressions:

```
L1 Address    Size    Region
----------    ----    ------
0x00000       16 KB   BRISC_CODE              (.init, .text for BRISC)
0x04000        8 KB   BRISC_LOADER_INIT_MEM   (.loader_init for BRISC data)
0x06000       16 KB   TRISC0_CODE             (.init, .text for unpack)
0x0A000        4 KB   TRISC0_LOADER_INIT_MEM  (.loader_init for unpack data)
0x0B000       16 KB   TRISC1_CODE             (.init, .text for math)
0x0F000        4 KB   TRISC1_LOADER_INIT_MEM  (.loader_init for math data)
0x10000       16 KB   TRISC2_CODE             (.init, .text for pack)
0x14000        4 KB   TRISC2_LOADER_INIT_MEM  (.loader_init for pack data)
  ...
0x20000        1 KB   RUNTIME_ARGS            (host-written parameters)
```

#### L0 Private Memory Layout

Each RISC-V core has a private L0 memory at `0xFFB00000`. Although all cores see the same base address, the hardware maps each core's access to its own physical SRAM.

| Core | Data Region | Data Size | Stack Region | Stack Size | Total L0 |
|:-----|:------------|:----------|:-------------|:-----------|:---------|
| BRISC | `0xFFB00000` | 6144 bytes (6 KiB) | `0xFFB01800` | 2048 bytes (2 KiB) | 8 KiB |
| TRISC0 | `0xFFB00000` | 3584 bytes | `0xFFB00E00` | 512 bytes | 4 KiB |
| TRISC1 | `0xFFB00000` | 3584 bytes | `0xFFB00E00` | 512 bytes | 4 KiB |
| TRISC2 | `0xFFB00000` | 3584 bytes | `0xFFB00E00` | 512 bytes | 4 KiB |

> **Critical constraint:** Each TRISC thread has only **3584 bytes** for read-only data, initialized data, BSS, and global variables, plus **512 bytes** for the call stack. This tight memory constraint is why the LLK API uses global variables (e.g., `unp_cfg_context`, `pack_sync_tile_dst_ptr`) instead of stack-allocated state, and why `-O3` inlining is critical -- inlined functions do not consume stack frames.

#### The Loader Init Mechanism

Private L0 memory (`0xFFB00000`) is not accessible via NOC -- the host cannot write to it directly. But initialized global variables need to live in L0 for fast core-local access. The solution is a **two-copy architecture**:

1. The linker places initialized data in the `.loader_init` section, which resides in L1 (NOC-accessible).
2. The host's ELF loader (`ttexalens.load_elf()`) writes the ELF to L1, including the `.loader_init` content.
3. At boot, the CRT startup code (`do_crt0()` or `tmu-crt0.S`) copies `.loader_init` from L1 into `.ldm_data` in L0 private memory before `main()` runs.

Each core's `LOADER_INIT_MEM` region is sized to match its total L0 data allocation (BRISC: 8 KB, TRISC: 4 KB), ensuring there is room in L1 for the full data image.

#### Discarded Regions

Several regions are defined with `ORIGIN = UNDEFINED (0), LENGTH = UNDEFINED (0)`:

- **`*_LOADER_CODE_MEM`** -- On architectures with IRAM (instruction RAM), this would hold code that gets copied to IRAM at boot. Blackhole does not have IRAM on these cores, so these are zero-length.
- **`*_GCOV_MEM`** -- In the non-coverage build (`memory.blackhole.ld`), GCOV regions are zero-length. A separate `memory.blackhole.debug.ld` provides actual GCOV regions for coverage builds.

---

### 6.3.3 Layer 2: Per-Thread Alias Scripts

Each thread has a small linker script that maps seven generic `REGION_*` names to thread-specific physical regions. For example, `math.ld` (TRISC1):

```ld
REGION_ALIAS("REGION_CODE",         TRISC1_CODE)
REGION_ALIAS("REGION_DATA",         TRISC1_LOCAL_DATA_MEM)
REGION_ALIAS("REGION_STACK",        TRISC1_STACK_MEM)
REGION_ALIAS("REGION_LOADER_INIT",  TRISC1_LOADER_INIT_MEM)
REGION_ALIAS("REGION_LOADER_CODE",  TRISC1_LOADER_CODE_MEM)
REGION_ALIAS("REGION_GCOV",         TRISC1_GCOV_MEM)
REGION_ALIAS("REGION_RUNTIME_ARGS", RUNTIME_ARGS)
```

The four scripts are identical except for the core prefix:

| Script | Core Prefix | `REGION_CODE` Maps To | Code Start |
|:-------|:------------|:----------------------|:-----------|
| `brisc.ld` | `BRISC_` | `BRISC_CODE` | `0x00000` |
| `unpack.ld` | `TRISC0_` | `TRISC0_CODE` | `0x06000` |
| `math.ld` | `TRISC1_` | `TRISC1_CODE` | `0x0B000` |
| `pack.ld` | `TRISC2_` | `TRISC2_CODE` | `0x10000` |

`RUNTIME_ARGS` is not aliased per-thread -- all threads share the same 1 KB region at `0x20000`. This works because all three TRISCs read the same runtime parameter struct from L1 during `copy_runtimes_from_L1()` in `trisc.cpp`.

As described in Chapter 3, Section 1, this alias mechanism is the linker-level parallel to the `#ifdef LLK_TRISC_*` pattern at the preprocessor level: the same template (`sections.ld`) is instantiated differently for each compilation target, producing three distinct ELF files targeting three non-overlapping L1 code regions.

---

### 6.3.4 Layer 3: `sections.ld` -- Section Placement Rules

This is the most complex linker script (`tests/helpers/ld/sections.ld`). It defines the ELF output format, the entry point, and the placement of every section into the abstract `REGION_*` names.

#### Preamble

```ld
__firmware_stack_size = LENGTH(REGION_STACK);
__firmware_start = DEFINED(__fw_export_end_text) ? __fw_export_end_text
                                                 : ORIGIN(REGION_CODE);
__ldm_start = DEFINED(__fw_export_ldm_end) ? __fw_export_ldm_end
                                           : ORIGIN(REGION_DATA);

OUTPUT_FORMAT("elf32-littleriscv", "elf32-littleriscv", "elf32-littleriscv")
OUTPUT_ARCH(riscv)
ENTRY(_start)
```

- `__firmware_stack_size` captures the stack region length (512 bytes for TRISC, 2 KB for BRISC).
- `__firmware_start` defaults to `ORIGIN(REGION_CODE)` unless a previous firmware stage exported end-of-text markers. In the LLK test infrastructure these symbols are not defined, so code starts at the beginning of `REGION_CODE`.
- The output is a 32-bit little-endian RISC-V ELF with `_start` as the entry point.

#### `.loader_init` -- L0 Data Staging Area

```ld
.loader_init :
{
    . = ALIGN(4);
    __loader_init_start = .;
    . += LENGTH(REGION_LOADER_INIT);
    __loader_init_end = .;
} > REGION_LOADER_INIT
```

This reserves space in L1 equal to the LOADER_INIT region length (4 KB for TRISC, 8 KB for BRISC). The linker populates this section with the initialized data content. Symbols `__loader_init_start` and `__loader_init_end` are used by the CRT startup to know how many bytes to copy from L1 to L0.

#### `.init` and `.text` -- Code Sections

```ld
PROVIDE (__executable_start = __firmware_start);
.init __firmware_start :
{
    KEEP (*(SORT_NONE(.init)))
} > REGION_CODE

.text :
{
    *(.text.unlikely .text.*_unlikely .text.unlikely.*)
    *(.text.exit .text.exit.*)
    *(.text.startup .text.startup.*)
    *(.text.hot .text.hot.*)
    *(.text .stub .text.* .gnu.linkonce.t.*)
    *(.gnu.warning)
} > REGION_CODE

.fini :
{
    KEEP (*(SORT_NONE(.fini)))
} > REGION_CODE
```

The `.init` section is placed first, at the start of the code region. This is where `_start` lives (marked with `__attribute__((section(".init")))` in both `tmu-crt0.S` and `trisc.cpp`/`brisc.cpp`). The hardware starts execution at the base of the code region, so `_start` must be the first instruction. The ordering within `.text` (unlikely, exit, startup, hot, default) allows the compiler to separate cold and hot paths.

#### `l1_data` and `l1_data_noinit`

```ld
l1_data :
{
    *(l1_data)
} > REGION_CODE

l1_data_noinit (NOLOAD) :
{
    *(l1_data_noinit)
} > REGION_CODE
```

Custom sections that allow code to explicitly place data in L1 (within the code region) rather than L0. Code can use `__attribute__((section("l1_data"))) const uint32_t lookup_table[] = { ... };` to avoid consuming scarce L0 private memory for large read-only tables.

#### `.ldm_data` -- Local Data Memory (L0)

```ld
PROVIDE(__global_pointer$ = ORIGIN(REGION_DATA) + 0x800);

.ldm_data __ldm_start :
{
    . = ALIGN(4);
    __ldm_data_start = .;
    *(.rodata .rodata.* .gnu.linkonce.r.*)  *(.rodata1)  *(.srodata .srodata.*)
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) ...))
    PROVIDE_HIDDEN(__init_array_end = .);
    /* ... fini_array similarly ... */
    *(.sdata .sdata.* .gnu.linkonce.s.*)  *(.data .data.* .gnu.linkonce.d.*)
    SORT(CONSTRUCTORS)  *(.data1)
    *(.got.plt) *(.igot.plt) *(.got) *(.igot)
    . = ALIGN(4);
    __ldm_data_end = .;
    __ldm_bss_start = .;
    *(.sbss .sbss.* .gnu.linkonce.sb.*)  *(.bss .bss.* .gnu.linkonce.b.*)  *(COMMON)
    . = ALIGN(4);
    __ldm_bss_end = .;
} > REGION_DATA
```

This section consolidates all data into L0 private memory (`REGION_DATA` at `0xFFB00000`). It contains, in order: read-only data (`.rodata`), constructor/destructor arrays (`.init_array`, `.fini_array`), small read-only data (`.srodata`), initialized data (`.sdata`, `.data`), GOT entries, and BSS (`.sbss`, `.bss`, `COMMON`). Note that the condensed snippet above is reorganized for pedagogical clarity; the actual `sections.ld` places `.srodata` after the constructor arrays, not adjacent to `.rodata`. This consolidation is unusual compared to hosted environments but necessary because L0 is only 3584 bytes and the linker must lay everything out contiguously.

The `__global_pointer$` symbol is set to `ORIGIN(REGION_DATA) + 0x800`. RISC-V uses the `gp` register for single-instruction access to data within a 4 KB window centered on this pointer.

Key exported symbols: `__ldm_data_start`/`__ldm_data_end` (initialized data bounds), `__ldm_bss_start`/`__ldm_bss_end` (BSS bounds), `__init_array_start`/`__init_array_end` (constructor table).

#### `.runtime_args` -- Runtime Parameters

```ld
.runtime_args (NOLOAD):
{
    . = ALIGN(16);
    __runtime_args_start = .;
    . += LENGTH(REGION_RUNTIME_ARGS);
    __runtime_args_end = .;
} > REGION_RUNTIME_ARGS
```

This `NOLOAD` section reserves the address range at `0x20000` (1 KiB) where the host writes the `RuntimeParams` struct before kernel launch. The `(NOLOAD)` attribute means the ELF does not contain data for this section -- the host fills it at runtime via `write_to_device()`. The `__runtime_args_start` symbol is referenced by `trisc.cpp`:

```cpp
extern const volatile struct RuntimeParams __runtime_args_start[];
ckernel::memcpy_blocking(temp_args, __runtime_args_start, sizeof(struct RuntimeParams));
```

#### `.stack` -- Stack Region

```ld
.stack :
{
    __stack_bottom = .;
    . += __firmware_stack_size;
    __stack_top = .;
} > REGION_STACK
```

The stack grows downward from `__stack_top`. For TRISC threads, this is 512 bytes starting at `0xFFB00E00` (top at `0xFFB01000`).

#### Debug Sections

Debug sections (`.debug_info`, `.debug_line`, `.debug_abbrev`, etc.) are placed at address 0 using the `0 :` syntax in the linker script. They are **retained in the ELF file** for host-side debugging via `riscv-tt-elf-objdump`, but they are not loaded to device memory. This is distinct from the `/DISCARD/` treatment -- discarded sections are removed from the ELF entirely.

#### `/DISCARD/` -- Stripped Sections

An extensive list of sections is discarded:

- ELF interpreter (`.interp`), build ID (`.note.gnu.build-id`)
- Symbol hash tables (`.hash`, `.gnu.hash`)
- Dynamic linking tables (`.dynsym`, `.dynstr`, `.dynamic`)
- All relocation sections (`.rela.*`)
- Exception handling (`.eh_frame`, `.gcc_except_table`)
- Thread-local storage (`.tdata`, `.tbss`)
- Residual PLT/GOT entries (`.got.plt`, `.igot.plt`, `.got`, `.igot` are first claimed by `.ldm_data`; only unmatched leftovers reach `/DISCARD/`)

This is safe because the bare-metal environment has no dynamic linker, no exception handling, and no TLS. Discarding these sections is essential for keeping the ELF small enough to fit in the 16 KB code regions.

#### `_Z11kernel_initv` Symbol

At the very end of `sections.ld`:

```ld
_Z11kernel_initv = _etext;
```

This defines the mangled C++ symbol for `void kernel_init()` as an alias to `_etext` (end of code). This provides a default no-op implementation -- calling it would jump to the end of text, which is harmless. This accommodates LLK library code that optionally calls `kernel_init()` without requiring every kernel to define it.

---

### 6.3.5 The `__instrn_buffer` Symbol

```ld
PROVIDE(__instrn_buffer = 0xFFE40000);
```

This symbol defines the base address of the **replay buffer** -- a special memory region where the MOP (Matrix Operation Pipeline) engine reads pre-programmed instruction sequences. The matmul math kernel writes 16 `TTI_MVMUL` instructions to consecutive addresses starting at `__instrn_buffer`, then the MOP replays them autonomously during execution.

The address `0xFFE40000` is in the Tensix core's private address space (above `0xFFB00000`), not in L1. It is a hardware-mapped register region specific to the Tensix compute engine.

> **Cross-reference:** Chapter 5, Section 3 documents how the `ckernel_template` class programs the replay buffer and MOP configuration registers.

---

### 6.3.6 Startup Assembly: `tmu-crt0.S`

The file `tests/helpers/tmu-crt0.S` provides the assembly-language CRT (C Runtime) startup routine. While the current codebase primarily uses the C++ inline assembly version in `boot.h` (via the `do_crt0()` function), the assembly version documents the startup sequence in its most explicit form and serves as a reference implementation.

#### Full Assembly Source

```asm
.section .init
.global _start
.type   _start, @function

_start:
CRT_START:
  .option push
  .option norelax
  la gp, __global_pointer$       # Step 1: Init gp (norelax prevents circular relaxation)
  .option pop
  la sp, __stack_top              # Step 2: Set stack pointer

  la      t0, __ldm_bss_start    # Step 3: Clear BSS (byte-by-byte)
  la      t1, __ldm_bss_end
1:beq     t0, t1, 2f
  sw      zero, 0(t0)
  addi    t0, t0, 1
  j       1b

2:la      t0, __loader_init_start # Step 4: Copy .loader_init (L1) to .ldm_data (L0)
  la      t1, __loader_init_end
  beq     t0, t1, 4f
  la      t1, __ldm_data_end
  la      t2, __ldm_data_start
3:beq     t2, t1, 4f
  lw      t3, 0(t0)
  sw      t3, 0(t2)
  addi    t0, t0, 4
  addi    t2, t2, 4
  j       3b

4:la      s2, __init_array_start  # Step 5: Execute global constructors
  la      s3, __init_array_end
  j       6f
5:lw      a0, 0(s2)
  jalr    a0
  addi    s2, s2, 4
6:bne     s2, s3, 5b

  call    main
  #ifdef COVERAGE
  call gcov_dump
  #endif
  10: j 10b                       # Loop forever (never returns)
```

#### Startup Sequence Analysis

The five steps mirror the C++ `do_crt0()` function below:

1. **Initialize `gp`** -- `.option norelax` prevents the linker from relaxing this `la` into a `gp`-relative access (circular dependency). `__global_pointer$` is set to `ORIGIN(REGION_DATA) + 0x800` in `sections.ld`, centering the +/- 2 KB addressing window.
2. **Initialize `sp`** -- `__stack_top` from `sections.ld`. For TRISC: `0xFFB01000` (top of 4 KB L0).
3. **Clear BSS** -- The loop stores a 32-bit zero via `sw` but advances the pointer by only 1 byte per iteration (`addi t0, t0, 1`), resulting in overlapping writes. This is slower than word-aligned clearing but handles arbitrary BSS boundaries without assuming alignment. The C++ version in `do_crt0()` uses word-granularity (4-byte `uint32_t*` iteration), which is faster but relies on `ALIGN(4)` in `sections.ld`.
4. **Copy `.loader_init` to `.ldm_data`** -- The critical step that makes initialized globals work. The ELF loader writes the data image to L1, and this loop copies it word-by-word into L0 private memory.
5. **Execute global constructors** -- Iterates function pointers from `__init_array_start` to `__init_array_end`. Typically empty for LLK kernels.

After these steps, control falls through to `call main`.

#### The `boot.h` C++ Equivalent: `do_crt0()`

In practice, `brisc.cpp` and `trisc.cpp` use an inline C++ version of the same five steps, defined as `do_crt0()` in `boot.h`. It performs the same operations using word-granularity (4-byte `uint32_t*` iteration) for BSS clearing and `.loader_init` copying. See Chapter 3, Section 3.2 for the full `do_crt0()` source and analysis.

Both BRISC and TRISC entry points call `do_crt0()` from a `_start()` function marked with `section(".init")`, `naked`, and `noreturn` attributes. The `section(".init")` attribute places the entry point at the very start of the code region, making it the first instruction executed when the core is released from reset. The TRISC version additionally includes `no_profile_instrument_function` to avoid profiler hooks before the stack is initialized.

Both the assembly and C++ versions define `_init` and `_fini` as empty functions to satisfy the toolchain's expectations.

---

### 6.3.7 Coverage Build Memory Layout

When coverage is enabled (`coverage_build=CoverageBuild.Yes`), the build uses `memory.blackhole.debug.ld` instead. The key change: TRISC code regions expand to 64 KiB and data moves from L0 to L1 (32 KiB), because coverage instrumentation generates counters that far exceed L0's 3584-byte limit. Since data is already L1-resident, `LOADER_INIT` regions become `UNDEFINED` (no copy-on-boot needed). GCOV regions get 8 KiB per core. Stack expands to the full 4 KiB L0. Runtime args shift to `0x6E000` and mailboxes to `0x6DFB8` to accommodate the larger regions.

---

### 6.3.8 Memory Layout Visualization

The complete memory view for a TRISC1 (math) kernel, highlighting which ELF maps where:

```
  L1 SRAM (NOC-accessible)                  Per-Core Private L0
  ========================                   ====================

  0x00000 +-----------------------+
          | BRISC_CODE (16K)      |          0xFFB00000 +------------------+
  0x04000 +-----------------------+                     | .ldm_data        |
          | BRISC_INIT (8K)       |                     | (3584B for TRISC)|
  0x06000 +-----------------------+                     | rodata, data,    |
          | TRISC0_CODE (16K)     |                     | bss              |
  0x0A000 +-----------------------+          0xFFB00E00 +------------------+
          | TRISC0_INIT (4K)      |                     | .stack (512B)    |
  0x0B000 +======================+           0xFFB01000 +------------------+
          | TRISC1_CODE  <--HERE  |
          |   .init, .text,       |          Instruction RAM (Hardware)
          |   l1_data             |          0xFFE40000: __instrn_buffer
  0x0F000 +======================+            (replay buffer for MOP)
          | TRISC1_INIT (4K)      |
  0x10000 +-----------------------+
          | TRISC2_CODE (16K)     |
  0x14000 +-----------------------+
          | TRISC2_INIT (4K)      |
          |       ... gap ...     |
  0x1FFB8 | Mailboxes (24B)       |
  0x20000 +-----------------------+
          | RUNTIME_ARGS (1K)     |
          +-----------------------+
```

---

### 6.3.9 Practical Implications for Kernel Development

Understanding the linker scripts reveals several constraints that affect how matmul kernels must be written:

1. **16 KiB code limit per thread.** Each TRISC thread has exactly 16 KiB for compiled code. Complex kernels may need `-O3` and careful function inlining to fit. The `SPEED_OF_LIGHT` mode can reduce code size by making all parameters compile-time constants.

2. **3584 bytes for data, 512 bytes for stack.** Static arrays, lookup tables, and deep recursion are essentially prohibited. Most LLK functions are designed to operate on data in register files and L1 directly, avoiding L0 data usage. Large read-only tables should use the `l1_data` section attribute to place them in L1 code space.

3. **All three threads share L1.** A kernel that overflows its 16 KiB code region will silently overwrite the next thread's loader_init or code region. The linker will warn about this (`-Wl,--trace` helps diagnose), but it is not always caught at compile time.

4. **Runtime parameters at a fixed address.** The `RuntimeParams` struct must be written to `0x20000` before kernel launch. This address is hardcoded in both the linker script and `test_config.py`.

5. **Mailbox addresses are convention, not linker-enforced.** The mailbox addresses (`0x1FFB8` for non-coverage) are defined as constants in `brisc.cpp` and `trisc.cpp`, not in the linker script. Changing the L1 layout without updating these constants will break BRISC/TRISC communication.

---

### 6.3.10 Verifying Memory Placement

After building, verify that each ELF's sections land in the correct regions:

```bash
for elf in unpack.elf math.elf pack.elf; do
    echo "=== $elf ==="
    riscv-tt-elf-objdump -h "$elf" | \
        grep -E 'Idx|\.init|\.text|l1_data|\.ldm_data|\.stack|\.loader_init'
done
```

Expected `.init` VMA addresses: `unpack.elf` at `0x06000` (TRISC0), `math.elf` at `0x0B000` (TRISC1), `pack.elf` at `0x10000` (TRISC2), `brisc.elf` at `0x00000` (BRISC). All `.ldm_data` sections should show VMA `0xFFB00000` regardless of thread. See Section 6.2.12 for code-size verification commands.

---

### 6.3.11 End-to-End Flow

The complete path from ELF to running kernel connects compilation (Section 6.2), linker scripts (this section), and the CRT startup:

1. **Host compiles** four ELFs with the three-script linker chain, placing code and data at the correct L1 addresses.
2. **Host writes `RuntimeParams`** to L1 address `0x20000` via `ttexalens.write_to_device()`.
3. **Host loads ELFs** -- `brisc.elf` to `0x00000`, `unpack.elf` to `0x06000`, `math.elf` to `0x0B000`, `pack.elf` to `0x10000`. Load addresses are encoded in each ELF's program headers.
4. **Host releases BRISC from soft reset.** BRISC executes `_start`, runs `do_crt0()`, then enters the boot-and-poll loop.
5. **Host sends START_TRISCS command via mailbox.** BRISC initializes hardware, clears TRISC soft reset bits.
6. **Each TRISC executes `_start`** -- CRT copies `.loader_init` from L1 to L0, zeroes BSS, runs constructors. `main()` reads `RuntimeParams`, calls `run_kernel()`.
7. **Host polls mailboxes** until all three TRISC threads write `KERNEL_COMPLETE`, then reads results.

> **Cross-reference:** Chapter 3, Section 2 details the BRISC boot sequence and mailbox protocol. Chapter 7 covers host-side ELF loading and device execution.

---

*Previous: [6.2 -- Compilation Commands and Flags](02_compilation_commands_and_flags.md)*
| *Next: [Chapter 7 -- ELF Loading and Kernel Execution](../ch7_final/01_elf_loading.md)*
