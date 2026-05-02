# Boot Sequence and Firmware

## Overview of the Boot Path

Before `run_kernel()` executes, a precise boot sequence must complete on each RISC-V core. The boot path differs between the BRISC core (which manages the Tensix lifecycle) and the three TRISC cores (which run the LLK kernel). On Blackhole, the default boot mode is **BRISC boot mode** (`BootMode.BRISC`), in which BRISC starts first, performs hardware initialization, and then releases the TRISC cores from reset.

The full boot sequence for all cores follows the same initial pattern:

```
_start()  -->  do_crt0()  -->  main()
```

But the `main()` function is radically different for BRISC vs. TRISC.

## The `_start()` Entry Point

Every ELF binary -- whether BRISC or TRISC -- begins execution at `_start()`. This function is placed in the `.init` section of the ELF, marked `naked` (no compiler-generated prologue/epilogue) and `noreturn` (execution never returns from it). The two definitions are structurally similar but not identical. In `brisc.cpp`:

```cpp
extern "C" __attribute__((section(".init"), naked, noreturn))
std::uint32_t _start()
{
    do_crt0();
    main();
    for (;;) { }  // Loop forever
}
```

In `trisc.cpp`, `_start()` carries an additional `no_profile_instrument_function` attribute (ensuring profiling hooks are not injected before the C runtime is initialized) and includes a conditional `gcov_dump()` call for code-coverage builds:

```cpp
extern "C" __attribute__((section(".init"), naked, noreturn, no_profile_instrument_function))
std::uint32_t _start()
{
    do_crt0();
    main();
#ifdef COVERAGE
    gcov_dump();
#endif
    for (;;) { }  // Loop forever
}
```

The attributes are significant:

- `section(".init")` -- places this function at the very beginning of the ELF's executable region, at the address the processor jumps to on reset.
- `naked` -- tells the compiler not to generate a function prologue or epilogue (no stack frame setup), since the stack pointer is not yet valid.
- `noreturn` -- signals that this function never returns; the infinite loop is the halting state.
- `no_profile_instrument_function` (TRISC only) -- prevents the compiler from inserting profiling instrumentation into this function, which would crash because the C runtime is not yet initialized.

The `extern "C"` linkage ensures the symbol name is not mangled, matching the ELF entry point specified in the linker script.

## The `do_crt0()` Function: C Runtime Initialization

The `do_crt0()` function in `boot.h` performs the C runtime initialization that must happen before any C++ code can safely execute. It is marked `always_inline` and `no_profile_instrument_function` to ensure it runs before any instrumentation or profiling infrastructure.

The five steps are:

### 1. Set the Global Pointer

```cpp
asm volatile(
    ".option push\n"
    ".option norelax\n"
    "la gp, __global_pointer$\n"
    ".option pop" ::: "memory");
```

The global pointer (`gp`) is set to the linker-provided `__global_pointer$` symbol, enabling GP-relative addressing for globals. The `.option norelax` directive prevents the assembler from optimizing this instruction into a relaxed form that would fail if `gp` is not yet set.

### 2. Set the Stack Pointer

```cpp
asm volatile("la sp, %0" : : "i"(__stack_top) : "memory");
```

The stack pointer (`sp`) is loaded from the linker symbol `__stack_top`, which points to the top of the stack region for the current core. Each RISC-V processor (BRISC, TRISC0, TRISC1, TRISC2) has its own separate stack region in L1, defined by the linker script.

### 3. Zero the BSS Segment

```cpp
for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start;
     p < (volatile std::uint32_t*)__ldm_bss_end; p++)
{
    *p = 0;
}
```

All uninitialized global and static variables reside in the `.bss` section. The C standard requires them to be zero-initialized before program execution begins.

### 4. Copy Initialized Data

```cpp
if ((std::uint32_t)__loader_init_start != (std::uint32_t)__loader_init_end)
{
    volatile std::uint32_t* src = (volatile std::uint32_t*)__loader_init_start;
    volatile std::uint32_t* dst = (volatile std::uint32_t*)__ldm_data_start;
    volatile std::uint32_t* end = (volatile std::uint32_t*)__ldm_data_end;
    while (dst < end) { *dst++ = *src++; }
}
```

The initialized data section (`.data`) may be stored in a different memory region than where it needs to reside at runtime. The `__loader_init_start` / `__ldm_data_start` symbols mark the source and destination of this copy.

### 5. Execute Global Constructors

```cpp
for (void (**temp_constructor)(void) = __init_array_start;
     temp_constructor < __init_array_end; temp_constructor++)
{
    (*temp_constructor)();
}
```

Global C++ constructors (the `.init_array` section) are called in order.

After `do_crt0()` completes, the C++ runtime environment is fully operational: the stack is live, global variables are initialized, and constructors have run.

## BRISC `main()`: The Lifecycle Manager

After `do_crt0()` completes on the BRISC core, `main()` in `brisc.cpp` begins. Its role is to act as a permanent command loop, responding to instructions from the host (written to L1 mailbox addresses via `ttexalens`).

### Mailbox Setup

BRISC defines a set of mailbox addresses in L1 for communication:

```cpp
mailbox_t mailboxes_arr = (mailbox_t)0x1FFB8U;  // Performance mode base

mailbox_t mailbox_unpack = mailboxes_arr;        // +0: TRISC0 completion
mailbox_t mailbox_math   = mailboxes_arr + 1;    // +4: TRISC1 completion
mailbox_t mailbox_pack   = mailboxes_arr + 2;    // +8: TRISC2 completion

mailbox_t brisc_command_buffer = mailboxes_arr + 3;  // +12: Command buffer (2 entries)
mailbox_t brisc_counter        = mailboxes_arr + 5;  // +20: Command counter
```

The mailbox addresses are fixed L1 locations. The host writes commands to `brisc_command_buffer` and reads completion status from `mailbox_unpack`, `mailbox_math`, and `mailbox_pack`.

In coverage builds (`#ifdef COVERAGE`), the mailbox base address shifts to `0x6DFB8U` to accommodate the larger L1 layout used for coverage data collection.

### The Command Loop

BRISC enters an infinite polling loop, checking for commands:

```cpp
int main()
{
    disable_branch_prediction();

    std::uint32_t counter = 0;

    // Initialize mailboxes and command buffer
    ckernel::store_blocking(brisc_command_buffer, 0);
    ckernel::store_blocking(brisc_command_buffer + 1, 0);
    ckernel::store_blocking(brisc_counter, 0);
    ckernel::store_blocking(brisc_bread0, 0);
    ckernel::store_blocking(brisc_bread1, 0);

    while (true)
    {
        ckernel::invalidate_data_cache();

        switch (static_cast<BriscCommandState>(
            ckernel::load_blocking(brisc_command_buffer + (counter & 1))))
        {
            case BriscCommandState::START_TRISCS:
                // Reset completion mailboxes
                commit_store(mailbox_math, ckernel::RESET_VAL);
                commit_store(mailbox_unpack, ckernel::RESET_VAL);
                commit_store(mailbox_pack, ckernel::RESET_VAL);

                // Initialize hardware
                device_setup();

                // Release TRISC cores from soft reset
                clear_trisc_soft_reset();

                reset_state(counter);
                commit_store(brisc_bread0, counter);
                break;

            case BriscCommandState::RESET_TRISCS:
                set_triscs_soft_reset();
                reset_state(counter);
                commit_store(brisc_bread1, counter);
                break;

            default:
                break;
        }

        ckernel::wait(ARCH_CYCLE_MICRO_SECOND);  // 1350 cycles = ~1us on Blackhole
    }
}
```

The command buffer uses a double-buffering scheme indexed by `counter & 1`, allowing the host to queue a new command while BRISC processes the current one.

The `commit_store()` function ensures mailbox writes are fully committed to L1 by performing a store, then a load from the same address, then spinning until the loaded value matches:

```cpp
template <typename T, typename U, ...>
inline void commit_store(volatile T* ptr, U&& val)
{
    ckernel::store_blocking(ptr, val);
    do { asm volatile("nop"); }
    while (ckernel::load_blocking(ptr) != val);
}
```

This is necessary because L1 writes are not immediately visible to other processors on Blackhole without explicit ordering.

The `BriscCommandState` enum defines the available commands:

| Command | Value | Action |
|:--------|:------|:-------|
| `IDLE_STATE` | 0 | No action; BRISC continues polling |
| `START_TRISCS` | 1 | Initialize device hardware, clear TRISC soft reset |
| `RESET_TRISCS` | 2 | Assert TRISC soft reset (halt all TRISC cores) |
| `UPDATE_START_ADDR_CACHE_AND_START` | 3 | Wormhole-specific; on Blackhole, falls through to `START_TRISCS` |

## The `device_setup()` Function

Before releasing the TRISC cores, BRISC calls `device_setup()` to initialize the Tensix Engine hardware to a known state. On Blackhole, this performs the following operations:

### 1. Disable Destination Clock Gating

```cpp
#if defined(ARCH_BLACKHOLE) && !defined(ARCH_QUASAR)
    ckernel::reg_write(RISCV_DEBUG_REG_DEST_CG_CTRL, 0);
#endif
```

This ensures the Destination register is always clocked during computation, preventing potential clock-gating-related timing issues. Blackhole has a hardware register that can power-gate unused Destination register partitions; writing zero disables all gating.

### 2. Clear the Destination Register

```cpp
#if defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0, 1, 0);
#else
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0);
#endif
```

The `ZEROACC` instruction clears all entries in the Destination register. On Blackhole, this instruction takes five parameters (the additional `use_32_bit_mode` and `clear_zero_flags` fields), while Wormhole's version takes three.

### 3. Enable the CC Stack (Condition Code Stack)

```cpp
TTI_SFPENCC(3, 0, 0, 10);
TTI_NOP;
```

The `SFPENCC` instruction initializes the SFPU's condition code stack. Even though a basic matmul does not use the SFPU, this initialization is part of the standard device setup.

### 4. Set SFPU Constants

```cpp
TTI_SFPCONFIG(0, 11, 1);  // Load -1 to LREG11 where sfpi expects it
```

This configures the constant -1 in the SFPU's local register 11 (`LREG11`). The SFPI compiler runtime assumes this register contains -1 for negation operations.

### 5. Initialize Hardware Semaphores

```cpp
ckernel::t6_semaphore_init(ckernel::semaphore::UNPACK_TO_DEST, 0, 1);
ckernel::t6_semaphore_init(ckernel::semaphore::MATH_DONE, 0, 1);
ckernel::t6_semaphore_init(ckernel::semaphore::PACK_DONE, 0, 1);
```

Each call to `t6_semaphore_init()` emits a `SEMINIT` instruction that sets the minimum value to 0 and the maximum value to 1 for the specified hardware semaphore, establishing them as binary semaphores. These semaphores are used for inter-thread synchronization during kernel execution (covered in the next section). Note that other semaphores (like `MATH_PACK`) are initialized separately by the math-pack sync protocol rather than in `device_setup()`.

## Releasing TRISC Cores: `clear_trisc_soft_reset()`

After `device_setup()`, BRISC releases the TRISC cores by clearing their soft reset bits:

```cpp
TT_ALWAYS_INLINE void clear_trisc_soft_reset()
{
    std::uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset &= ~TRISC_SOFT_RESET_MASK;  // TRISC_SOFT_RESET_MASK = 0x7000
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);

    do {
        asm volatile("nop");
    } while (ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0) != soft_reset);
}
```

The `TRISC_SOFT_RESET_MASK` is `0x7000`, which covers bits 12, 13, and 14 of the soft reset register -- one bit per TRISC core. Clearing these bits simultaneously releases all three TRISC cores, which then begin executing from their respective `_start()` entry points.

The spin loop after the write ensures the register update has propagated through the memory subsystem before proceeding. This is necessary because on Blackhole, register writes are not immediately visible to subsequent reads due to the memory ordering model.

The complementary function `set_triscs_soft_reset()` reasserts the bits to halt the TRISCs:

```cpp
TT_ALWAYS_INLINE void set_triscs_soft_reset()
{
    std::uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset |= TRISC_SOFT_RESET_MASK;   // Set bits 12-14
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);
    do { asm volatile("nop"); }
    while (ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0) != soft_reset);
}
```

## TRISC `main()`: The Kernel Runner

After `do_crt0()` completes on each TRISC core, `main()` in `trisc.cpp` begins. Unlike BRISC's infinite loop, TRISC's `main()` runs exactly once: it initializes state, executes the kernel, signals completion, and halts.

### Step 1: Determine the Mailbox Address

Each TRISC core computes its mailbox address based on which thread it is:

```cpp
#if defined(LLK_TRISC_UNPACK)
constexpr std::uint32_t mailbox_offset = 0;
#elif defined(LLK_TRISC_MATH)
constexpr std::uint32_t mailbox_offset = sizeof(std::uint32_t);    // 4
#elif defined(LLK_TRISC_PACK)
constexpr std::uint32_t mailbox_offset = 2 * sizeof(std::uint32_t); // 8
#else
#error "No TRISC define set"
#endif

mailbox_t mailbox = reinterpret_cast<volatile std::uint32_t*>(
    mailboxes_start + mailbox_offset);
```

This gives each TRISC core its own completion mailbox: TRISC0 at `0x1FFB8`, TRISC1 at `0x1FFBC`, TRISC2 at `0x1FFC0`. These correspond to the `mailbox_unpack`, `mailbox_math`, and `mailbox_pack` addresses monitored by BRISC.

### Step 2: Copy Runtime Parameters from L1

```cpp
struct RuntimeParams temp_args;
copy_runtimes_from_L1(&temp_args);
```

The `copy_runtimes_from_L1()` function copies the `RuntimeParams` struct from a known L1 address (where the host placed it) into a local variable on the stack:

```cpp
void copy_runtimes_from_L1(struct RuntimeParams* temp_args)
{
    extern const volatile struct RuntimeParams __runtime_args_start[];
    ckernel::memcpy_blocking(temp_args, __runtime_args_start,
                             sizeof(struct RuntimeParams));
}
```

The `__runtime_args_start` symbol is defined in the linker script and points to the L1 region reserved for runtime parameters (at 0x20000). The `memcpy_blocking()` function is a blocking copy that ensures all data is visible before returning -- it uses a sequence of loads, stores, readback loads, and NOPs to guarantee memory ordering on the RISC-V core's weak memory model.

### Step 3: Zero the Register File

```cpp
std::fill(ckernel::regfile, ckernel::regfile + 64, 0);
```

The 64-entry software register file (`regfile`) is zeroed. This file is used as scratch space by the Tensix instruction macros (e.g., `TT_SETDMAREG` writes values into GPRs in this register file, which are then referenced by Tensix instructions).

### Step 4: Reset Configuration State

```cpp
ckernel::reset_cfg_state_id();
ckernel::reset_dest_offset_id();
```

These reset the configuration context ping-pong counter (`cfg_state_id = 0`) and the destination double-buffer offset counter (`dest_offset_id = 0`) to known initial states.

### Step 5: Execute the Kernel

```cpp
run_kernel(temp_args);
ckernel::tensix_sync();
```

This is the call to the user's kernel function -- the `run_kernel()` defined in the `#ifdef` block of the kernel source file. After `run_kernel()` returns, `tensix_sync()` is called to ensure all in-flight Tensix instructions have completed before signaling the host:

```cpp
inline void tensix_sync()
{
    store_blocking(&pc_buf_base[1], 0);
}
```

The `tensix_sync()` function writes zero to `pc_buf_base[1]` using `store_blocking()`. The PC buffer is a memory-mapped interface to the Tensix Engine; writing to offset 1 acts as a **pipeline drain barrier** -- it stalls the RISC-V core until all previously issued Tensix instructions have retired. This ensures the kernel's results are fully committed to the Destination register and L1 before the completion signal is sent.

The `store_blocking()` function itself uses a store-then-readback-then-dependent-instruction sequence to force serialization:

```cpp
asm volatile(
    "sw %[raw], (%[ptr])\n\t"  // Store 0 to pc_buf_base[1]
    "lw %[raw], (%[ptr])\n\t"  // Read back from same address
    "and x0, x0, %[raw]\n\t"
    : [raw] "+r"(raw) : [ptr] "r"(ptr) : "memory");
```

The readback load is ordered after the store (same address), and the `and` instruction creates a data dependency that blocks the pipeline until the load completes. The memory clobber prevents the compiler from reordering any subsequent memory accesses before this barrier.

Without `tensix_sync()` before the mailbox write, a TRISC core could signal completion while pack or unpack instructions are still in flight, leading to corrupted results.

### Step 6: Signal Completion

```cpp
*mailbox = ckernel::KERNEL_COMPLETE;
```

The TRISC core writes the value `KERNEL_COMPLETE` (0xFF) to its designated mailbox address in L1. The host polls this address; when all three mailboxes read 0xFF, the kernel execution is considered complete and the host can read the results from L1.

After writing the completion signal, execution falls through to the infinite loop at the end of `_start()`:

```cpp
for (;;) { }  // Loop forever
```

The TRISC core idles here until BRISC reasserts its soft reset bit (via the `RESET_TRISCS` command), which halts the core.

## TRISC Boot Mode (Alternative)

There is an alternative boot mode (`BootMode.TRISC`) where BRISC is not involved and TRISC0 takes on the initialization role. In this mode, TRISC0 runs `device_setup()` and `clear_trisc_soft_reset()` -- the same functions that BRISC calls in BRISC mode:

```cpp
#if defined(LLK_TRISC_UNPACK) && defined(LLK_BOOT_MODE_TRISC)
    mailbox_t mailbox_base = reinterpret_cast<volatile std::uint32_t*>(mailboxes_start);
    *(mailbox_base)        = ckernel::RESET_VAL;
    *(mailbox_base + 1)    = ckernel::RESET_VAL;
    *(mailbox_base + 2)    = ckernel::RESET_VAL;
    device_setup();
    clear_trisc_soft_reset();
#endif
```

TRISC0 resets the mailboxes, initializes the hardware, and then releases TRISC1 and TRISC2. This mode is the default for Quasar but not for Blackhole. On P150 (Blackhole), BRISC mode is always used.

## The Complete Boot Timeline

Putting it all together, the boot sequence for a single kernel execution on Blackhole is:

```
Host                          BRISC                       TRISC0/1/2
  |                             |                            |
  |-- Load ELFs to L1 -------->|                            |
  |-- Write RuntimeParams ---->|                            |
  |-- Write stimuli tiles ---->|                            |
  |                             |                            |
  |                     _start()                             |
  |                     do_crt0()                            |
  |                     main() [enters command loop]         |
  |                             |                            |
  |-- Write START_TRISCS ----->|                            |
  |                     Read START_TRISCS                    |
  |                     Reset mailboxes to 0                 |
  |                     device_setup()                       |
  |                       - ZEROACC (clear Dest)             |
  |                       - SFPENCC (init CC stack)          |
  |                       - SFPCONFIG (load constants)       |
  |                       - SEMINIT x3 (init semaphores)     |
  |                     clear_trisc_soft_reset()             |
  |                             |                            |
  |                             |                     _start() x3
  |                             |                     do_crt0() x3
  |                             |                     main() x3
  |                             |                       copy_runtimes_from_L1()
  |                             |                       zero regfile
  |                             |                       reset_cfg_state_id()
  |                             |                       reset_dest_offset_id()
  |                             |                       run_kernel(params)
  |                             |                         [kernel execution]
  |                             |                       tensix_sync()
  |                             |                       *mailbox = 0xFF
  |                             |                       [infinite loop]
  |                             |                            |
  |<---- Poll mailboxes --------|                            |
  |  (wait for all 3 = 0xFF)   |                            |
  |                             |                            |
  |-- Read results from L1 --->|                            |
  |-- Write RESET_TRISCS ----->|                            |
  |                     set_triscs_soft_reset()              |
  |                             |                     [halted]
```

All three TRISC cores are released simultaneously by `clear_trisc_soft_reset()` and execute concurrently. The total firmware overhead -- from TRISC release to `run_kernel()` entry -- is dominated by the `copy_runtimes_from_L1()` call (which performs a blocking byte-by-byte copy of the `RuntimeParams` struct) and the register file zeroing. The `device_setup()` cost is paid once by BRISC before the TRISCs even start.

The three TRISC cores share the same Tensix Engine backend, with hardware semaphores mediating access to shared resources. The inter-thread synchronization protocol that governs this shared access is the subject of the next section.

---

**Next:** [`03_inter_thread_synchronization.md`](./03_inter_thread_synchronization.md)
