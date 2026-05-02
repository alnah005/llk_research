# 7.2 Step-by-Step Execution Workflow

This section traces the complete workflow for executing an LLK kernel on hardware, from architecture detection through result readback and golden comparison. The central focus is the BRISC command protocol -- the mailbox-based communication system through which the host orchestrates TRISC kernel launches without directly manipulating soft-reset registers for every test.

---

## 7.2.1 The Execution Workflow Overview

The entry point for every test execution is `TestConfig.run()`:

```python
def run(self, location="0,0"):
    self.generate_variant_hash()
    if TestConfig.MODE in [TestMode.PRODUCE, TestMode.DEFAULT]:
        self.build_elfs()
    self.write_runtimes_to_L1(location)
    if self.variant_stimuli:
        self.variant_stimuli.write(location)
    self.run_elf_files(location)
    self.wait_for_tensix_operations_finished(location)
    return TestOutcome(result=(...))
```

Here is the complete sequence that `TestConfig.run()` and its surrounding test infrastructure execute for every test variant:

| Step | Action | Where |
|------|--------|-------|
| 1 | Architecture detection and build setup | `conftest.py` / `TestConfig.setup_build()` |
| 2 | ARC GO_BUSY message | `conftest.py` / `pytest_sessionstart()` |
| 3 | Kernel compilation (four ELFs) | `TestConfig.build_elfs()` |
| 4 | Runtime parameters written to L1 | `TestConfig.write_runtimes_to_L1()` |
| 5 | Stimuli data written to L1 (pre-tilized) | `StimuliConfig.write()` or `write_pipeline_operands_to_l1()` |
| 6 | BRISC ELF loaded and released from reset (once) | `TestConfig.run_elf_files()` |
| 7 | TRISC ELFs loaded to device memory | `TestConfig.run_elf_files()` |
| 8 | BRISC command to launch TRISCs | `commit_brisc_command()` |
| 9 | Host polls TRISC mailboxes for completion | `TestConfig.wait_for_tensix_operations_finished()` |
| 10 | Results read back from L1 and compared against golden | `collect_results()` + `passed_test()` |

Steps 1-2 happen once per session. Step 3 happens once per variant (with caching). Steps 4-10 happen for every test execution. The following subsections detail each step.

---

## 7.2.2 Step 1: Architecture Detection and Build Setup

**Python call:** `TestConfig.setup_build(sources_path)` from `conftest.py :: pytest_configure()`

At pytest startup, `conftest.py` calls `TestConfig.setup_build()` which triggers `TestConfig.setup_arch()` to detect the connected hardware. For P150 (Blackhole), the architecture dictates:

- Boot mode: `BootMode.BRISC` (BRISC manages TRISC lifecycle)
- Compiler target: `-mcpu=tt-bh-tensix` (compute), `-mcpu=tt-bh` (non-compute)
- Architecture define: `-DARCH_BLACKHOLE`
- Data format enum mapping: `BLACKHOLE_DATA_FORMAT_ENUM_VALUES`

The mailbox enum is also selected here:

```python
device_module.Mailboxes = (
    (MailboxesCoverageQuasar if with_coverage else MailboxesQuasar)
    if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR
    else (MailboxesCoverage if with_coverage else Mailboxes)
)
```

For a standard (non-coverage) Blackhole build, the mailbox layout is:

| Mailbox | Address | Purpose |
|---------|---------|---------|
| `Unpacker` | `0x1FFB8` | TRISC0 (unpack) completion flag |
| `Math` | `0x1FFBC` | TRISC1 (math) completion flag |
| `Packer` | `0x1FFC0` | TRISC2 (pack) completion flag |
| `BriscCommand0` | `0x1FFC4` | Command slot 0 (host -> BRISC) |
| `BriscCommand1` | `0x1FFC8` | Command slot 1 (host -> BRISC) |
| `BriscCounter` | `0x1FFCC` | Acknowledgment counter (BRISC -> host) |

These six 32-bit words in L1 constitute the entire inter-processor communication (IPC) protocol between host, BRISC, and the three TRISC cores. For coverage builds, the entire mailbox region shifts to base address `0x6DFB8` to avoid conflicts with the larger coverage data sections.

> **Cross-reference:** [Section 7.1.3](./01_host_device_communication.md) documents the architecture detection mechanism. [Chapter 6](../ch6_compilation/) provides exhaustive coverage of the compilation flags.

---

## 7.2.3 Step 2: ARC GO_BUSY

Before any device memory access, the host sends `GO_BUSY` (`0xAA52`) to the ARC processor. This ensures the chip is in its active power state with all clocks enabled.

**Hardware-level effect:** ARC responds by disabling clock gating and power-saving modes for the Tensix cores. Without this step, ARC may autonomously power-gate the core array during kernel execution, causing unpredictable stalls. The host blocks until ARC acknowledges (up to 10 seconds).

> **Cross-reference:** [Section 7.1.4](./01_host_device_communication.md) documents the ARC message protocol in detail.

---

## 7.2.4 Step 3: Kernel Compilation

**Python call:** `TestConfig.build_elfs()`

This step compiles the kernel source into four ELF files: `brisc.elf`, `unpack.elf`, `math.elf`, and `pack.elf`. BRISC is compiled once as a shared artifact; the three TRISC ELFs are compiled per test variant using `ThreadPoolExecutor` for parallel compilation.

Each variant gets a SHA-256 hash of all compilation-affecting parameters (test name, data formats, template parameters, boot mode). A cache check (`.build_complete` marker file) avoids redundant builds, and `FileLock` ensures thread safety when pytest-xdist runs parallel workers.

> **Cross-reference:** [Chapter 6](../ch6_compilation/) provides exhaustive coverage of the compilation pipeline, toolchain flags, and linker script architecture.

---

## 7.2.5 Step 4: Runtime Parameters Written to L1

**Python call:** `TestConfig.write_runtimes_to_L1(location)`

Before loading any ELFs, the host writes the `RuntimeParams` struct to L1 so that the kernels can read their configuration at startup:

```python
def write_runtimes_to_L1(self, location="0,0"):
    argument_data = [
        self.pack_size,       # uint32_t TILE_SIZE_PACK
        self.unpack_size_a,   # uint32_t TILE_SIZE_UNPACK_A
        self.unpack_size_b,   # uint32_t TILE_SIZE_UNPACK_B
    ]

    if not self.compile_time_formats:
        for format_tuple in self.formats_config:
            argument_data.extend([
                TestConfig.DATA_FORMAT_ENUM[format_tuple.unpack_A_src],
                TestConfig.DATA_FORMAT_ENUM[format_tuple.unpack_B_src],
                # ... 11 format enum values per FormatConfig ...
                TestConfig.DATA_FORMAT_ENUM[format_tuple.pack_S_dst],
            ])

    # Plus variant-specific stimuli addresses and runtime parameters

    serialised_data = struct.pack(self.runtime_format, *argument_data)
    write_to_device(location, TestConfig.RUNTIME_ADDRESS_NON_COVERAGE, serialised_data)
```

**Hardware-level effect:** The runtime parameters are written to L1 address `0x20000` (or `0x6E000` for coverage builds). The linker script defines a `NOLOAD` section at this address with the symbol `__runtime_args_start`, so the TRISC kernels know where to find the data. The `trisc.cpp` startup code copies this into a local struct:

```cpp
void copy_runtimes_from_L1(struct RuntimeParams* temp_args)
{
    extern const volatile struct RuntimeParams __runtime_args_start[];
    ckernel::memcpy_blocking(temp_args, __runtime_args_start, sizeof(struct RuntimeParams));
}
```

The `struct.pack` format string starts with `"@III"` for the three tile sizes, then appends eleven `I`s per format configuration. For a matmul with runtime formats, the total payload is typically 14 words = 56 bytes.

> **Cross-reference:** [Chapter 6, Section 3](../ch6_compilation/03_linker_scripts_and_memory_regions.md) documents the `REGION_RUNTIME_ARGS` linker section and the `__runtime_args_start` symbol. [Chapter 5, Section 2](../ch5_compile_time_config/02_build_h_generation_and_constexpr.md) covers the RuntimeParams struct layout.

---

## 7.2.6 Step 5: Stimuli Data Written to L1 (Pre-Tilized Format)

**Python call:** `self.variant_stimuli.write(location)` or `write_pipeline_operands_to_l1(pipeline, location)`

Input tensors must be written in **pre-tilized** format -- the tile-based layout that the Tensix unpacker hardware expects. The host performs tilization in Python before writing to L1.

For the fused-operation pipeline path, the address allocation is sequential starting at `0x21000`:

```python
def write_pipeline_operands_to_l1(pipeline, location="0,0"):
    current_address = 0x21000

    for operation in pipeline:
        src_a = operation.src_a
        if src_a.is_input() and src_a.l1_address is None:
            src_a.l1_address = current_address
            current_address += calculate_size(src_a.data_format, src_a.tile_count)
            write_operand_data(src_a, location)
        # ... same for src_b, then reserve space for output ...
```

Each tile is packed into its target data format before writing:

| Data Format | Packer Function | Bytes Per Tile (1024 elements) |
|------------|----------------|-------------------------------|
| `Float16_b` | `pack_bfp16()` | 2048 |
| `Float16` | `pack_fp16()` | 2048 |
| `Float32` | `pack_fp32()` | 4096 |
| `Bfp8_b` | `pack_bfp8_b()` | 1088 (1024 data + 64 exponent) |

**Hardware-level effect:** A 32x32 tile in Float16_b is stored as four 16x16 faces in a specific memory order:

```
Face 0 (rows 0-15, cols 0-15):  256 values x 2 bytes = 512 bytes
Face 1 (rows 0-15, cols 16-31): 256 values x 2 bytes = 512 bytes
Face 2 (rows 16-31, cols 0-15): 256 values x 2 bytes = 512 bytes
Face 3 (rows 16-31, cols 16-31):256 values x 2 bytes = 512 bytes
                                              Total = 2048 bytes
```

The data must arrive in L1 in exact tile format because the Tensix unpacker reads directly from these addresses using DMA, interpreting the bytes according to the data format registers configured by the unpack LLK.

For a single-tile matmul (one tile of A, one tile of B), the L1 layout after this step is:

```
0x20000 - 0x20FFF: RuntimeParams (padded to fill region)
0x21000 - 0x217FF: src_a tile (2048 bytes in Float16_b)
0x21800 - 0x21FFF: src_b tile (2048 bytes in Float16_b)
0x22000 - 0x227FF: output tile (reserved, 2048 bytes)
```

> **Cross-reference:** [Chapter 2, Section 1](../ch2_data_formats_and_memory/01_tile_geometry_and_face_layout.md) explains tile geometry and face layout. [Chapter 2, Section 2](../ch2_data_formats_and_memory/02_data_format_encoding.md) covers data format encoding.

---

## 7.2.7 Step 6: BRISC ELF Loaded and Released from Reset

BRISC is loaded exactly once per test session. On subsequent tests, BRISC remains running and only receives new commands:

```python
def run_elf_files(self, location="0,0"):
    boot_mode = CHIP_DEFAULT_BOOT_MODES[TestConfig.CHIP_ARCH]

    if boot_mode == BootMode.BRISC:
        if not TestConfig.BRISC_ELF_LOADED:
            set_tensix_soft_reset(1, location=location)         # Hold ALL cores in reset
            TestConfig.BRISC_ELF_LOADED = True
            load_elf(
                elf_file=str((TestConfig.SHARED_ELF_DIR / "brisc.elf").absolute()),
                location=location,
                risc_name="brisc",
                verify_write=False,
            )
            set_tensix_soft_reset(0, [RiscCore.BRISC], location) # Release BRISC only
```

**Hardware-level effect:**

1. `set_tensix_soft_reset(1)` -- Sets bits 11-14 in `RISCV_DEBUG_REG_SOFT_RESET_0`, holding all cores in reset.
2. `load_elf("brisc.elf", ...)` -- Parses the ELF, extracts loadable segments, writes them to L1 at addresses specified in the program headers.
3. `set_tensix_soft_reset(0, [RiscCore.BRISC])` -- Clears bit 11 only. BRISC begins executing `_start()` -> `do_crt0()` -> `main()`.

After this sequence, BRISC is running its `main()` loop while all three TRISCs remain in reset. BRISC's first action is to initialize its mailbox state:

```cpp
// brisc.cpp -- main()
std::uint32_t counter = 0;

ckernel::store_blocking(brisc_command_buffer, 0);      // Clear command slot 0
ckernel::store_blocking(brisc_command_buffer + 1, 0);   // Clear command slot 1
ckernel::store_blocking(brisc_counter, 0);              // Reset ack counter
ckernel::store_blocking(brisc_bread0, 0);               // Clear debug bread0
ckernel::store_blocking(brisc_bread1, 0);               // Clear debug bread1
```

BRISC then enters its infinite command polling loop, which is the core of the BRISC command protocol (detailed in Section 7.2.9).

---

## 7.2.8 Step 7: TRISC ELFs Loaded to Device Memory

With BRISC running, the host sends `RESET_TRISCS` to ensure all three TRISCs are in reset, then loads the kernel ELFs:

```python
if get_chip_architecture() != ChipArchitecture.QUASAR:
    commit_brisc_command(location, BriscCmd.RESET_TRISCS)

for i, elf_file_path in enumerate(self.temp_elfs):
    if TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE:
        start_address = load_elf(
            elf_file=elf_file_path,
            location=location,
            risc_name=f"trisc{i}",
            return_start_address=True,
            verify_write=False,
        )
        write_words_to_device(location, TestConfig.TRISC_START_ADDRS[i], [start_address])
    else:  # Blackhole
        load_elf(
            elf_file=elf_file_path,
            location=location,
            risc_name=f"trisc{i}",
            verify_write=False,
        )
```

Note the Wormhole-specific extra step: after loading each ELF, the host writes the `_start` address to a known L1 location (`0x16DFF0`, `0x16DFF4`, `0x16DFF8`). This is necessary because on Wormhole, the ELF loader cannot directly set the TRISC program counter -- BRISC must read these addresses and program the `TRISC_RESET_PC` configuration registers before releasing TRISCs from reset. On Blackhole, this indirection is not needed because the hardware reads the program counter directly from the ELF load address.

After loading new ELFs on Wormhole, the host sends a combined command:

```python
if boot_mode == BootMode.BRISC and TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE:
    commit_brisc_command(location, BriscCmd.UPDATE_START_ADDR_CACHE_AND_START)
    return
```

On subsequent runs with the same ELFs (same variant), loading is skipped entirely and only the start command is sent. This is tracked by `TestConfig.LAST_LOADED_ELFS`.

---

## 7.2.9 The BRISC Command Protocol (Deep Dive)

The BRISC command protocol is the heart of the host-device coordination for BRISC-mode boot. It uses a double-buffered mailbox scheme with a monotonic counter for handshaking.

The mailbox layout (see Section 7.2.2 for addresses) includes three TRISC completion words, two command slots, a counter, and two debug snapshots.

### The BriscCommandState Enum

```cpp
// brisc.cpp
enum class BriscCommandState : std::uint32_t
{
    IDLE_STATE                        = 0,
    START_TRISCS                      = 1,
    RESET_TRISCS                      = 2,
    UPDATE_START_ADDR_CACHE_AND_START = 3,
};
```

The Python side mirrors this:

```python
class BriscCmd(Enum):
    IDLE_STATE = 0
    START_TRISCS = 1
    RESET_TRISCS = 2
    UPDATE_START_ADDR_CACHE_AND_START = 3
```

### Double-Buffered Command Slots

The protocol uses **two command slots** and an **alternating counter** to avoid race conditions between the host writing a command and BRISC reading the previous one. The counter modulo 2 selects which slot is active.

**Host side** (`commit_brisc_command` in `device.py`):

```python
common_counter = 0

def commit_brisc_command(location="0,0", command=BriscCmd.IDLE_STATE, timeout=0.1):
    global common_counter

    # Write command to the slot selected by counter parity
    if common_counter & 1:
        write_words_to_device(location, Mailboxes.BriscCommand1.value, [command.value])
    else:
        write_words_to_device(location, Mailboxes.BriscCommand0.value, [command.value])

    common_counter += 1

    # Poll until BRISC acknowledges by matching our counter
    end_time = time.time() + timeout
    while time.time() < end_time:
        temp_value = read_word_from_device(location, Mailboxes.BriscCounter.value, 0)
        if temp_value == common_counter:
            return

    raise TimeoutError("Polling brisc command timed out")
```

**BRISC side** (`brisc.cpp`):

```cpp
while (true)
{
    ckernel::invalidate_data_cache();

    switch (static_cast<BriscCommandState>(
        ckernel::load_blocking(brisc_command_buffer + (counter & 1))))
    {
        case BriscCommandState::START_TRISCS:
            // ... execute start sequence ...
            reset_state(counter);
            commit_store(brisc_bread0, counter);
            break;

        case BriscCommandState::RESET_TRISCS:
            set_triscs_soft_reset();
            reset_state(counter);
            commit_store(brisc_bread1, counter);
            break;

        default:  // IDLE_STATE
            break;
    }

    // Wait for 1us before polling again (1350 cycles on Blackhole)
    ckernel::wait(ARCH_CYCLE_MICRO_SECOND);
}
```

### The Handshake Sequence

Here is the full lifecycle of a single command:

```
Time    Host                           BRISC (on-chip)
----    ----                           ---------------
 T0     common_counter = 0             counter = 0
        (both slots read as IDLE)      (polling slot 0, sees IDLE)

 T1     Write START_TRISCS to slot 0   ...polling slot 0...
        common_counter++ => 1
        Begin polling BriscCounter

 T2     ...polling...                  invalidate_data_cache()
                                       load_blocking(slot 0) => START_TRISCS
                                       Execute start sequence:
                                         - Reset all TRISC mailboxes to RESET_VAL
                                         - Call device_setup()
                                         - Call clear_trisc_soft_reset()

 T3     ...polling...                  reset_state(counter):
                                         counter++ => 1
                                         Write IDLE to slot 1 (next slot)
                                         Write counter=1 to brisc_counter

 T4     Read BriscCounter => 1
        1 == common_counter(1) => ACK!
        Return successfully

 T5     Write RESET_TRISCS to slot 1   ...polling slot 1, sees IDLE (just cleared)
        common_counter++ => 2
        Begin polling BriscCounter     Wait... sees RESET_TRISCS
                                       set_triscs_soft_reset()
                                       reset_state(counter):
                                         counter++ => 2
                                         Write IDLE to slot 0
                                         Write counter=2 to brisc_counter

 T6     Read BriscCounter => 2
        2 == common_counter(2) => ACK!
```

### The reset_state() Function

The `reset_state()` function is the critical acknowledgment mechanism:

```cpp
void reset_state(std::uint32_t& counter)
{
    counter++;
    // Pre-clear the NEXT slot to IDLE so BRISC won't re-execute
    ckernel::store_blocking(brisc_command_buffer + (counter & 1),
                            static_cast<std::uint32_t>(BriscCommandState::IDLE_STATE));
    // Signal completion by writing the new counter value
    commit_store(brisc_counter, counter);
}
```

The `commit_store()` helper in `boot.h` uses a store-then-readback-verify loop to ensure the write has propagated before BRISC continues:

```cpp
template <typename T, typename U>
inline void commit_store(volatile T* ptr, U&& val)
{
    ckernel::store_blocking(ptr, val);
    do { asm volatile("nop"); } while (ckernel::load_blocking(ptr) != val);
}
```

This design has three important properties:

1. **No lost commands** -- the host writes to one slot while BRISC reads from the other, avoiding write-read races.
2. **Atomic acknowledgment** -- the counter increment is the last operation, so the host knows the full command has been executed when it sees the counter match.
3. **Idempotent polling** -- BRISC polls the current slot every microsecond. If it sees `IDLE_STATE`, it does nothing. The `default` case in the switch handles this.

### Why Data Cache Invalidation Is Required

At the top of each polling iteration, BRISC calls `ckernel::invalidate_data_cache()`. This is essential because L1 memory is shared between the host (writing via PCIe/NOC) and BRISC (reading via its data cache). Without invalidation, BRISC could read a stale cached value and never see the new command. The invalidation forces BRISC to re-read from L1 on every poll cycle, ensuring it picks up host-written commands within one microsecond of the write.

---

## 7.2.10 Step 8: BRISC Executes the START_TRISCS Command

When BRISC sees `START_TRISCS` (or `UPDATE_START_ADDR_CACHE_AND_START`), it performs the kernel launch sequence:

```cpp
case BriscCommandState::UPDATE_START_ADDR_CACHE_AND_START:
#ifdef ARCH_WORMHOLE
    // Cache _start addresses from L1 (Wormhole only)
    for (int i = 0; i < 3; i++)
    {
        TRISC_ADDR_CACHE[i] = ckernel::load_blocking(trisc_start_addresses + i);
    }
#endif
    [[fallthrough]];

case BriscCommandState::START_TRISCS:
#ifdef ARCH_WORMHOLE
    // Write cached addresses to TRISC PC config registers
    for (int i = 0; i < 3; i++)
    {
        commit_store(trisc_start_addresses + i, TRISC_ADDR_CACHE[i]);
    }
#endif

    // Reset all TRISC mailboxes to RESET_VAL (not 0, not 0xFF)
    commit_store(mailbox_math, ckernel::RESET_VAL);
    commit_store(mailbox_unpack, ckernel::RESET_VAL);
    commit_store(mailbox_pack, ckernel::RESET_VAL);

    // Reset profiler barriers
    commit_store(profiler_barrier, 0U);
    commit_store(profiler_barrier + 1, 0U);
    commit_store(profiler_barrier + 2, 0U);

    // Hardware initialization
    device_setup();

    // Release all TRISCs from soft reset
    clear_trisc_soft_reset();

    // Acknowledge to host
    reset_state(counter);
    commit_store(brisc_bread0, counter);
    break;
```

### The device_setup() Function

The `device_setup()` function (from `boot.h`) performs critical hardware initialization before every kernel launch. The snippet below shows the Blackhole path; on Wormhole, `device_setup()` first programs the TRISC_RESET_PC config registers from the L1 start-address cache (see Section 7.2.8), and on Quasar the SFPENCC and semaphore instructions differ slightly:

```cpp
TT_ALWAYS_INLINE void device_setup()
{
    // 1. Blackhole: disable destination register clock gating
    ckernel::reg_write(RISCV_DEBUG_REG_DEST_CG_CTRL, 0);

    // 2. Clear the entire accumulator (all rows, all columns)
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0, 1, 0);

    // 3. Enable SFPU condition code stack (10 entries deep)
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_NOP;

    // 4. Load constant -1.0 into SFPU LREG11 (expected by SFPI compiler)
    TTI_SFPCONFIG(0, 11, 1);

    // 5. Initialize inter-thread semaphores to correct initial values
    ckernel::t6_semaphore_init(ckernel::semaphore::UNPACK_TO_DEST, 0, 1);
    ckernel::t6_semaphore_init(ckernel::semaphore::MATH_DONE, 0, 1);
    ckernel::t6_semaphore_init(ckernel::semaphore::PACK_DONE, 0, 1);
}
```

Each of these operations is essential:

1. **Dest CG disable** -- without this, the destination register file may be clock-gated, causing reads to return garbage.
2. **ZEROACC** -- clears any residual values from a previous kernel's accumulator state.
3. **SFPENCC** -- enables the condition code stack used by SFPU branching operations; the depth of 10 allows nested conditionals.
4. **SFPCONFIG** -- the SFPI compiler assumes LREG11 contains -1.0 for certain negation patterns.
5. **Semaphore init** -- the unpack/math/pack pipeline uses hardware semaphores for synchronization. They must be initialized to the correct values (max count 1, current count 0) before any kernel begins.

BRISC then calls `clear_trisc_soft_reset()`, which clears bits 12-14 of `RISCV_DEBUG_REG_SOFT_RESET_0` via a read-modify-write with a readback-verify loop, releasing all three TRISCs simultaneously. See Chapter 3, Section 3.2 for the full function listing.

> **Cross-reference:** [Chapter 5](../ch5_compile_time_config/) documents the semaphore protocol (UNPACK_TO_DEST, MATH_DONE, PACK_DONE) that coordinates the three TRISC threads during kernel execution.

---

## 7.2.11 TRISC Startup and Kernel Execution

Once released from reset, each TRISC executes `trisc.cpp`:

```cpp
int main(void)
{
    mailbox_t mailbox = reinterpret_cast<volatile std::uint32_t*>(
        mailboxes_start + mailbox_offset);

    // In TRISC boot mode only: TRISC0 performs device_setup + releases other TRISCs
    // (Not executed in BRISC boot mode -- BRISC already did this)

    struct RuntimeParams temp_args;
    copy_runtimes_from_L1(&temp_args);     // Read runtime config from L1

    std::fill(ckernel::regfile, ckernel::regfile + 64, 0);  // Clear register file

    ckernel::reset_cfg_state_id();
    ckernel::reset_dest_offset_id();

    {
        ZONE_SCOPED("KERNEL")
        run_kernel(temp_args);             // Execute the actual LLK kernel
        ckernel::tensix_sync();            // Wait for all pipeline operations to drain
    }

    *mailbox = ckernel::KERNEL_COMPLETE;   // Signal completion (writes 0xFF)
}
```

The mailbox offsets are determined at compile time via preprocessor defines:

| Define | Offset | Mailbox Address (standard) |
|--------|--------|---------------------------|
| `LLK_TRISC_UNPACK` | 0 | `0x1FFB8` |
| `LLK_TRISC_MATH` | `sizeof(uint32_t)` | `0x1FFBC` |
| `LLK_TRISC_PACK` | `2 * sizeof(uint32_t)` | `0x1FFC0` |

After writing `KERNEL_COMPLETE` (`0xFF`) to the mailbox, the TRISC enters an infinite loop (via the `_start` epilogue). On coverage builds, `gcov_dump()` is called before the infinite loop to flush coverage data to L1. The TRISC will be put back into reset by the next `RESET_TRISCS` command.

---

## 7.2.12 Step 9: Host Polls Mailboxes for Completion

The host polls all three TRISC mailboxes until each reads `0xFF`:

```python
KERNEL_COMPLETE = 0xFF

def wait_for_tensix_operations_finished(self, core_loc="0,0", timeout=2):
    mailboxes = {core for core in device_module.Mailboxes}
    if self.CHIP_ARCH != ChipArchitecture.QUASAR:
        mailboxes -= {
            device_module.Mailboxes.BriscCommand0,
            device_module.Mailboxes.BriscCommand1,
            device_module.Mailboxes.BriscCounter,
        }

    timeout = 600 if test_target.run_simulator else timeout

    time.sleep(0.001)  # Brief initial delay

    completed = set()
    end_time = time.time() + timeout
    while time.time() < end_time:
        for mailbox in mailboxes - completed:
            if read_word_from_device(core_loc, mailbox.value) == KERNEL_COMPLETE:
                completed.add(mailbox)
        if completed == mailboxes:
            return

    # Timeout -- check for assertions before raising
    handle_if_assert_hit(self.temp_elfs, core_loc=core_loc)

    trisc_hangs = [mailbox.name for mailbox in (mailboxes - completed)]
    raise TimeoutError(
        f"Timeout reached: waited {timeout} seconds for {', '.join(trisc_hangs)}"
    )
```

The polling loop:

1. Starts with a 1ms sleep to allow TRISCs time to begin execution.
2. Iterates over the set of uncompleted mailboxes, reading each one.
3. When a mailbox reads `0xFF`, it is moved to the completed set.
4. When all mailboxes are complete, the function returns successfully.
5. On timeout (default 2 seconds for hardware, 600 seconds for simulator), it first checks whether any TRISC hit an assertion (ebreak instruction) via `handle_if_assert_hit()`, and if so, extracts and raises callstacks using DWARF debug info. Otherwise, it raises a `TimeoutError` listing which TRISCs did not complete.

The mailbox values follow a lifecycle that depends on the boot mode. In **BRISC mode** (Blackhole/P150), the lifecycle is two-state:

| Value | Meaning | Set By |
|-------|---------|--------|
| `ckernel::RESET_VAL` | BRISC reset (pre-launch) | BRISC `START_TRISCS` handler |
| `0xFF` (`KERNEL_COMPLETE`) | Execution complete | Each TRISC in `trisc.cpp` |

In **TRISC/EXALENS boot modes**, an additional initial state exists: Python calls `reset_mailboxes()` to write `0xA3` before starting cores, creating a three-state lifecycle (`0xA3` → `RESET_VAL` → `0xFF`). The `0xA3` sentinel helps triage hangs -- if the host reads `0xA3`, the BRISC/TRISC firmware never reached the mailbox reset, indicating a boot failure.

---

## 7.2.13 Step 10: Result Readback and Golden Comparison

Once all TRISCs complete, the host reads the output tensor from L1:

```python
def collect_pipeline_results(pipeline, location="0,0"):
    for operation in pipeline:
        output_operand = operation.output

        read_bytes_cnt = (
            output_operand.data_format.num_bytes_per_tile(TILE_ELEMENTS)
            * operation.output.tile_count
        )
        read_data = read_from_device(
            location, output_operand.l1_address, num_bytes=read_bytes_cnt
        )

        res_from_L1 = unpack_res_tiles(
            read_data, output_format, tile_count=tile_cnt,
            sfpu=False, num_faces=operation.num_faces,
            face_r_dim=operation.face_r_dim,
        )

        torch_format = format_dict[output_format]
        tilized_tensor = torch.tensor(res_from_L1, dtype=torch_format)

        if output_format != DataFormat.Bfp8_b and output_dimensions is not None:
            raw_tensor = untilize_block(
                tilized_tensor, stimuli_format=output_format, dimensions=output_dimensions,
            )
```

The readback performs these operations:

1. **Read raw bytes** from the output L1 address (computed during Step 5 address allocation).
2. **Unpack tiles** from the device data format back to floating-point values (`unpack_res_tiles`).
3. **Convert to PyTorch tensor** in the appropriate dtype.
4. **Untilize** -- convert from tile layout back to row-major layout for comparison.

The golden reference is computed entirely on the host using PyTorch. For matmul, the golden generator optionally models the precision loss of the specified `MathFidelity` mode (LoFi, HiFi2, HiFi3, HiFi4):

```python
generate_golden = get_golden_generator(MatmulGolden)
golden_tensor = generate_golden(
    src_A, src_B, formats.output_format, math_fidelity,
    input_A_dimensions=input_A_dimensions,
    input_B_dimensions=input_B_dimensions,
    tilize=True,
)
```

The final assertion applies format-appropriate tolerances:

```python
res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
assert passed_test(golden_tensor, res_tensor, formats.output_format)
```

> **Cross-reference:** [Chapter 2, Section 1](../ch2_data_formats_and_memory/01_tile_geometry_and_face_layout.md) explains the tilize/untilize transformation. [Chapter 2, Section 2](../ch2_data_formats_and_memory/02_data_format_encoding.md) covers data format encoding and byte sizes.

---

## 7.2.14 The Complete Execution Timeline

Combining all steps, here is the temporal ordering for a typical test session executing multiple test variants:

```
Session Start:
  [Host]  init_ttexalens()                    -- PCI init, create Context
  [Host]  get_chip_architecture()             -- Detect Blackhole
  [Host]  arc_msg(GO_BUSY)                    -- Activate chip
  [Host]  setup_build()                       -- Configure paths, flags, mailboxes

First Variant:
  [Host]  build_elfs()                        -- Compile brisc.elf + 3 TRISC ELFs
  [Host]  write_runtimes_to_L1(0x20000)       -- RuntimeParams struct
  [Host]  stimuli.write(0x21000)              -- Pre-tilized input tensors
  [Host]  set_tensix_soft_reset(1)            -- Hold all cores in reset
  [Host]  load_elf(brisc.elf)                 -- Load BRISC firmware
  [Host]  set_tensix_soft_reset(0, [BRISC])   -- Release BRISC only
  [BRISC] _start -> do_crt0() -> main()       -- Enter command loop
  [Host]  commit_brisc_command(RESET_TRISCS)  -- Ensure TRISCs in reset
  [BRISC] set_triscs_soft_reset()             -- Assert TRISC reset
  [Host]  load_elf(unpack.elf, math.elf, pack.elf) -- Load kernel ELFs
  [Host]  commit_brisc_command(START_TRISCS)  -- Launch!
  [BRISC] device_setup()                      -- HW init (zeroacc, semaphores)
  [BRISC] clear_trisc_soft_reset()            -- Release T0, T1, T2
  [T0-T2] _start -> do_crt0() -> main()       -- Run kernel
  [T0-T2] *mailbox = 0xFF                     -- Signal completion
  [Host]  wait_for_tensix_operations_finished()-- Poll mailboxes
  [Host]  collect_results(L1)                 -- Read output, untilize, compare

Second Variant (same test, different params):
  [Host]  build_elfs()                        -- Compile 3 new TRISC ELFs
  [Host]  write_runtimes_to_L1(0x20000)       -- New RuntimeParams
  [Host]  stimuli.write(0x21000)              -- New input data
  [Host]  commit_brisc_command(RESET_TRISCS)  -- Put TRISCs back in reset
  [Host]  load_elf(unpack.elf, math.elf, pack.elf) -- Load new kernel ELFs
  [Host]  commit_brisc_command(START_TRISCS)  -- Launch!
  ... (same execution flow as above) ...

Session End:
  [Host]  arc_msg(GO_IDLE)                    -- Deactivate chip
```

Note that BRISC is loaded and released exactly once. For all subsequent variants, only `RESET_TRISCS` / `START_TRISCS` commands cycle the TRISCs. If two consecutive variants have the same ELFs (same compilation hash), the ELF loading is skipped entirely -- only the command is sent.

The entire execution for a single matmul variant (steps 4-10, assuming compiled ELFs) takes approximately 5-50 milliseconds on silicon, dominated by the L1 read/write latency over PCIe.

---

**Previous:** [Host-Device Communication](./01_host_device_communication.md) | **Next:** [What TT-Metal Replaces](./03_what_tt_metal_replaces.md)
