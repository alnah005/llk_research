# Inter-Thread Synchronization

## The Hardware Semaphore System

The Tensix Engine provides a bank of 8 hardware semaphores, shared across all three TRISC threads. Each semaphore is a 4-bit counter (values 0 through 15) that supports atomic increment (SEMPOST), decrement (SEMGET), and conditional stall (SEMWAIT) operations. These semaphores are the sole mechanism by which the unpack, math, and pack threads coordinate their access to shared hardware resources.

The semaphore definitions are declared in `ckernel_structs.h`:

```cpp
struct semaphore
{
    constexpr static std::uint32_t FPU_SFPU            = 0;
    constexpr static std::uint32_t MATH_PACK           = 1;
    constexpr static std::uint32_t UNPACK_TO_DEST      = 2;
    constexpr static std::uint32_t UNPACK_OPERAND_SYNC = 3;
    constexpr static std::uint32_t PACK_DONE           = 4;
    constexpr static std::uint32_t UNPACK_SYNC         = 5;
    constexpr static std::uint32_t UNPACK_MATH_DONE    = 6;
    constexpr static std::uint32_t MATH_DONE           = 7;

    constexpr static std::uint8_t NUM_SEMAPHORES      = 8;
    constexpr static std::uint8_t SEMAPHORE_BIT_COUNT  = 4;
    constexpr static std::uint8_t SEMAPHORE_MAX_VALUE  = 15;  // (1 << 4) - 1

    constexpr static std::uint16_t t6_sem(const std::uint8_t sem_index)
    {
        return (1 << sem_index);
    }
};
```

The `t6_sem()` function converts a semaphore index (0-7) to a bitmask (`1 << index`) for use in the Tensix `SEMINIT`, `SEMPOST`, `SEMGET`, and `SEMWAIT` instructions, which address semaphores by bitmask rather than by index. For example, `MATH_PACK` (index 1) becomes `0x02`, and `UNPACK_SYNC` (index 5) becomes `0x20`.

For a standard matmul kernel, two semaphores are central to the data synchronization:

| Semaphore | Index | Purpose |
|:----------|:------|:--------|
| `MATH_PACK` | 1 | Coordinates math and pack threads on the Destination register |
| `UNPACK_SYNC` | 5 | Synchronizes the unpack thread (TRISC0) with the unpacker hardware engine |

Additionally, three semaphores are initialized during `device_setup()` before kernel execution begins (see Section 02):

| Semaphore | Index | Initialized To | Used In |
|:----------|:------|:---------------|:--------|
| `UNPACK_TO_DEST` | 2 | min=0, max=1 | Unpack-to-dest mode (not standard matmul) |
| `MATH_DONE` | 7 | min=0, max=1 | Unpack-to-dest mode (not standard matmul) |
| `PACK_DONE` | 4 | min=0, max=1 | Profiling barriers and synchronization fencing |

The `MATH_PACK` semaphore is initialized separately by `_llk_math_pack_sync_init_()` during the kernel setup phase, not in `device_setup()`.

## Semaphore Operations

The ckernel infrastructure provides two layers of semaphore access: software-level functions (called from the RISC-V core) and Tensix instruction-level operations (issued to the Tensix Engine pipeline). Understanding the distinction between these two layers is critical for reading LLK synchronization code.

### RISC-V-Side Semaphore Access (via PC Buffer)

These functions operate from the RISC-V core by reading and writing the semaphore memory-mapped registers through the PC buffer:

```cpp
inline std::uint8_t semaphore_read(const std::uint8_t index)
{
    return pc_buf_base[PC_BUF_SEMAPHORE_BASE + index];
}

inline void semaphore_post(const std::uint8_t index)
{
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 0;  // LSB clear -> increment
}

inline void semaphore_get(const std::uint8_t index)
{
    pc_buf_base[PC_BUF_SEMAPHORE_BASE + index] = 1;  // LSB set -> decrement
}
```

The semaphore hardware has exotic read/write semantics: reading returns the current value, but writing performs an atomic operation determined by the LSB of the written value -- writing 0 increments (SEMPOST), writing 1 decrements (SEMGET). These software-level functions are used for polling loops on the RISC-V side (e.g., `wait_for_next_context()` in the unpack loop). The RISC-V core spins explicitly, consuming cycles.

### Tensix Instruction-Level Semaphore Operations

These functions emit Tensix ISA instructions that execute within the Tensix Engine pipeline. They can optionally stall the pipeline until preceding operations of a specified type have completed:

```cpp
template <std::uint32_t WaitRes = p_stall::NONE>
inline void t6_semaphore_post(const std::uint8_t index)
{
    if constexpr (WaitRes != p_stall::NONE)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);
    }
    TTI_SEMPOST(semaphore::t6_sem(index));
}

template <std::uint32_t WaitRes = p_stall::NONE>
inline void t6_semaphore_get(const std::uint8_t index)
{
    if constexpr (WaitRes != p_stall::NONE)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes);
    }
    TTI_SEMGET(semaphore::t6_sem(index));
}
```

The `WaitRes` template parameter specifies which pipeline stage must complete before the semaphore operation executes. Common values include `p_stall::MATH` (wait for math to finish), `p_stall::PACK` (wait for packer to finish), and `p_stall::WAIT_SFPU` (wait for SFPU).

The key distinction: `TTI_SEMPOST` / `TTI_SEMGET` / `TTI_SEMWAIT` are **Tensix instructions** that execute in the Tensix pipeline and can be synchronized with other Tensix resources via optional stall conditions. `semaphore_post()` / `semaphore_get()` / `semaphore_read()` are **RISC-V operations** that read/write through the PC buffer memory-mapped interface and execute immediately from the core's perspective. The unpack thread uses RISC-V-side polling (because it needs to spin the RISC-V core while the unpacker hardware runs), while the math-pack synchronization uses Tensix-side stalls (because it can block the hardware pipeline directly).

## The Unpack Synchronization Protocol

The unpack thread uses a double-buffered configuration context to overlap configuration programming with active data movement. This is the most complex synchronization mechanism in a matmul kernel because it involves both software-level polling and Tensix-level semaphore operations.

### Context Double Buffering

The unpacker has two independent configuration contexts (context 0 and context 1). While one context is being used by the unpacker hardware to actively unpack data from L1 into Source registers, the TRISC0 software can program the other context with the configuration for the next tile.

The `unp_cfg_context` global variable tracks which context is currently being programmed. After each unpack operation, `switch_config_context()` toggles it:

```cpp
inline void switch_config_context(std::uint32_t &unp_cfg_context)
{
    unp_cfg_context = 1 - unp_cfg_context;
    if (unp_cfg_context == 0)
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    }
    else
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
    }
}
```

The `SETC16` instruction programs the context offset register, telling the Tensix configuration unit which set of registers subsequent configuration writes target. The value `0x0101` sets the offset for both unpacker 0 and unpacker 1 to context 1.

### The Unpack Sync Flow in `_llk_unpack_AB_matmul_()`

Within the matmul unpack function, the synchronization pattern for each tile iteration follows these steps:

**1. Wait for a free context:**

```cpp
wait_for_next_context(2);
```

This is a RISC-V-side spin loop:

```cpp
inline void wait_for_next_context(const std::uint32_t num_contexts)
{
    while (semaphore_read(semaphore::UNPACK_SYNC) >= num_contexts)
    {
    }
}
```

The `UNPACK_SYNC` semaphore (index 5) counts the number of contexts currently in use. If both contexts are occupied (semaphore value >= 2), the TRISC0 core spins until one becomes available.

**2. Configure addresses and post the semaphore:**

```cpp
_llk_unpack_configure_addresses_(address_b, address_a, cfg);
semaphore_post(semaphore::UNPACK_SYNC);
```

After writing the source addresses to the configuration registers for the current context, the unpack thread increments `UNPACK_SYNC` via `semaphore_post()`. This signals that a new context has been filled and is ready for the unpacker hardware to consume.

**3. Stall the unpacker until configuration writes are visible:**

```cpp
TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
```

This Tensix instruction stalls the unpacker until all pending configuration register writes from TRISC have completed. Without this stall, the unpacker might begin using a context before its configuration is fully written.

**4. Issue unpack instructions and run the MOP:**

The `TTI_UNPACR` instructions trigger the actual data movement from L1 to Source registers. These execute within the Tensix pipeline, overlapping with the TRISC0 core's next loop iteration. The MOP (Macro Operation) replays a pre-programmed sequence of instructions:

```cpp
TT_MOP(0, (reuse_a ? ct_dim : rt_dim) - 1, unp_cfg_context == 0 ? 0 : 0xff);
```

The third argument (the zmask) selects which context's replay buffer entries the MOP uses: `0x00` for context 0, `0xff` for context 1.

**5. Release the context and switch:**

```cpp
t6_semaphore_get(semaphore::UNPACK_SYNC);
switch_config_context(unp_cfg_context);
```

After the MOP completes, the unpack thread decrements `UNPACK_SYNC` (releasing the context) and switches to programming the other context for the next iteration.

### Unpack-Math Data Handoff

The unpack-to-math data handoff is implicit: it is mediated by hardware "data valid" flags rather than software semaphores. When the unpacker finishes writing data into the Source A and Source B register banks, it sets hardware-level validity flags (`SRCA_VLD` / `SRCB_VLD`). The math hardware waits on these flags (via `STALLWAIT` on `SRCA_VLD` / `SRCB_VLD`) before reading the data. The MOP programmed by `_llk_unpack_AB_matmul_mop_config_` records a sequence of `UNPACR` instructions that sets the `Dvalid` flag, which triggers the FPU to begin its matrix multiplication. This hardware-level handoff requires no explicit software synchronization between the unpack and math threads.

### Timeline Visualization

For a two-tile sequence, the unpack synchronization looks like:

```
TRISC0 (software):  [Config ctx0] [Config ctx1] [Config ctx0] ...
                         |              |              |
                    SEM_POST       SEM_POST       SEM_POST
                         |              |              |
Unpacker (hardware): [Execute ctx0] [Execute ctx1] [Execute ctx0] ...
                                   |              |
                              SEM_GET         SEM_GET
```

The key insight is that the TRISC0 core and the unpacker hardware operate concurrently: while the unpacker executes from one context, TRISC0 programs the other.

## The Math-Pack Synchronization Protocol

The math and pack threads share the Destination register. The synchronization between them ensures that:

1. The math thread does not write to a Destination half that the packer is reading.
2. The pack thread does not read from a Destination half that the math thread is writing.
3. The math thread does not start a new computation until the packer has cleared the previous result.

This is achieved through the `MATH_PACK` semaphore (index 1) and `DstSync::SyncHalf` double buffering.

### Math-Pack Sync Init

Before the main kernel loop, the math thread initializes the synchronization:

```cpp
_llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

This function:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) { };

    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);  // max=1, min=0
    }
    else  // SyncHalf
    {
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);  // max=2, min=0
    }
    reset_dest_offset_id();
    set_dest_section_base<StartZero>();
}
```

The function:

1. Calls `tensix_sync()` to drain the pipeline.
2. Waits for any previous pack operations to finish by spinning on the `MATH_PACK` semaphore.
3. Initializes the `MATH_PACK` semaphore. For `SyncHalf` (used in the matmul test), the max is set to 2, allowing the math thread to fill both Destination halves before stalling. For `SyncFull`, the max is 1 (no double buffering of the Destination).
4. Resets `dest_offset_id` to 0 and sets the destination section base to the first half.

### The `SEMWAIT` Stall Modes

The `TTI_SEMWAIT` instruction stalls specified pipeline stages until a semaphore meets a condition. It is important to understand the two stall modes:

| Mode | Meaning | Stall releases when... |
|:-----|:--------|:-----------------------|
| `STALL_ON_ZERO` | Stall while the semaphore value **is** zero | The semaphore becomes non-zero (someone posted to it) |
| `STALL_ON_MAX` | Stall while the semaphore value **is** at its programmed max | The semaphore drops below its max (someone consumed from it) |

### Math Side: Wait, Compute, Signal

The math thread follows a three-step pattern for each tile:

**1. Wait for a free Destination half:**

```cpp
_llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
```

Internally, this calls `math_dest_wait()`:

```cpp
inline void math_dest_wait()
{
    TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_MAX);
}
```

This stalls the math pipeline, SFPU, and sync unit until the `MATH_PACK` semaphore is **below its maximum value**. If the semaphore is at max (meaning both Destination halves contain unread results), the math pipeline stalls until the packer consumes one and decrements the semaphore.

At kernel startup, the `MATH_PACK` semaphore is at 0 (its minimum value, set by `SEMINIT`). Since 0 is not at the maximum value of 2, the `STALL_ON_MAX` condition is not triggered, and the math thread proceeds immediately on its first iteration. This is the correct initial behavior: both Destination halves are free at startup.

**2. Execute the matmul:**

```cpp
_llk_math_matmul_<MATH_FIDELITY>(0, params.CT_DIM, params.RT_DIM);
```

The math thread writes results into the current Destination half (determined by `dest_offset_id`).

**3. Signal completion and flip:**

```cpp
_llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

This function performs two operations:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    set_math_semaphores();
    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        dest_section_flip();
    }
}
```

`set_math_semaphores()` increments the `MATH_PACK` semaphore to tell the packer that a result is available:

```cpp
inline void set_math_semaphores()
{
    t6_semaphore_post<p_stall::MATH | p_stall::WAIT_SFPU>(semaphore::MATH_PACK);
}
```

The `p_stall::MATH | p_stall::WAIT_SFPU` template parameter ensures the SEMPOST instruction only executes after all preceding math and SFPU instructions have completed -- guaranteeing that the result is fully written to the Destination register before the packer is told to read it.

`dest_section_flip()` switches the math thread's view to the other Destination half:

```cpp
inline void dest_section_flip()
{
    update_dest_offset_id();          // dest_offset_id = 1 - dest_offset_id
    std::uint32_t base_addr = get_dest_buffer_base();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);
}
```

`get_dest_buffer_base()` returns `DEST_REGISTER_HALF_SIZE` when `dest_offset_id == 1`, or `0` when `dest_offset_id == 0`. The `TT_SETC16` instruction programs the math thread's destination offset register, directing subsequent math instructions to the other half.

### Pack Side: Wait, Pack, Clear, Signal

The pack thread follows a complementary pattern:

**1. Wait for math to produce a result:**

```cpp
_llk_packer_wait_for_math_done_();
```

```cpp
inline void _llk_packer_wait_for_math_done_()
{
    TTI_SEMWAIT(p_stall::STALL_TDMA,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_ZERO);
}
```

This stalls the packer's TDMA (Transfer DMA) stage until `MATH_PACK` is **non-zero** -- meaning the math thread has posted at least one result. The `STALL_ON_ZERO` condition means "stall while the semaphore is zero."

**2. Pack tiles from Destination to L1:**

```cpp
for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
{
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(
        i, L1_ADDRESS(params.buffer_Res[i]));
}
```

Each `_llk_pack_()` call reads tile data from the current Destination half and writes it to the specified L1 address, performing format conversion (e.g., from 32-bit accumulator to Float16_b) in the process.

**3. Clear the Destination half and signal math:**

```cpp
_llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

This function (from `tt_llk_blackhole/llk_lib/llk_pack_common.h`):

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_section_done_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);  // Wait for pack to finish

    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_ZEROACC(p_zeroacc::CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0);
    }
    else
    {
        static_assert(Dst == DstSync::SyncHalf);
        TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
    }

    // Tell math that it can write again
    _llk_packer_set_math_semaphore_<p_stall::NONE>();

    if constexpr (Dst == DstSync::SyncHalf)
    {
        flip_packer_dest_offset_id();
        select_packer_dest_registers<Dst>();
    }
}
```

The sequence is:

1. `TTI_STALLWAIT` -- Wait for the pack hardware to finish writing all tile data to L1.
2. `TT_ZEROACC` / `TTI_ZEROACC` -- Clear the Destination region that was just packed, so math can write new results there. On Blackhole, both instructions take five arguments: the clear mode (`CLR_HALF` or `CLR_ALL`), the `is_fp32_dest_acc_en` flag (passed as a separate parameter rather than folded into the clear mode), a zero literal, the address modifier `ADDR_MOD_1`, and the half selector (`dest_offset_id % 2` for `SyncHalf`, `0` for `SyncFull`). When `Dst == DstSync::SyncFull`, the `TTI_ZEROACC` variant clears the entire Destination register; when `Dst == DstSync::SyncHalf`, the `TT_ZEROACC` variant clears only the half identified by `dest_offset_id`.
3. `_llk_packer_set_math_semaphore_()` -- Decrement the `MATH_PACK` semaphore, signaling to the math thread that one Destination half is now free:
   ```cpp
   template <std::uint32_t WaitRes = p_stall::NONE>
   inline void _llk_packer_set_math_semaphore_()
   {
       t6_semaphore_get<WaitRes>(semaphore::MATH_PACK);
   }
   ```
4. `flip_packer_dest_offset_id()` -- Toggle the packer's dest offset to point at the other half.
5. `select_packer_dest_registers<Dst>()` -- Program the packer's destination register offset to read from the correct half on the next iteration:
   ```cpp
   template <DstSync Dst>
   inline void select_packer_dest_registers()
   {
       if constexpr (Dst == DstSync::SyncFull)
       {
           TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO,
                     p_cfg::WRCFG_128b,
                     DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
       }
       else
       {
           TT_WRCFG(get_packer_dest_offset_index(),
                     p_cfg::WRCFG_128b,
                     DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
       }
       TTI_DMANOP;
       TTI_DMANOP;
   }
   ```
   In `SyncFull` mode, the packer always reads from the same fixed offset (`DEST_OFFSET_LO`), since the entire Destination register is used as a single buffer. In `SyncHalf` mode (the `else` branch), the packer reads from whichever half `get_packer_dest_offset_index()` returns -- this tracks the ping-pong state toggled by `flip_packer_dest_offset_id()`. The two `TTI_DMANOP` instructions after the config write give the hardware time to process the register update before the next pack operation begins.

### Math-Pack Synchronization Timeline

For `DstSync::SyncHalf` with two tiles:

```
                    MATH_PACK semaphore: 0
                              |
Math thread:        [wait_for_dest_available]
                    (sem < max? yes, 0 < 2)
                    [compute tile 0 -> Dest half 0]
                    [set_math_semaphores: SEMPOST]
                              |
                    MATH_PACK semaphore: 1
                              |
Math thread:        [dest_section_flip: now points to half 1]
                    [wait_for_dest_available]
                    (sem < max? yes, 1 < 2)
                    [compute tile 1 -> Dest half 1]
                    [set_math_semaphores: SEMPOST]
                              |
                    MATH_PACK semaphore: 2 (MAX)
                              |
Math thread:        [wait_for_dest_available]
                    (sem < max? NO, stall...)
                              |
Pack thread:        [wait_for_math_done]               |
                    (sem > 0? yes, 2 > 0)              |
                    [pack tile 0 from Dest half 0]     |
                    [ZEROACC half 0]                    |
                    [SEMGET: decrement]                 |
                              |                        |
                    MATH_PACK semaphore: 1             |
                              |                        |
Math thread:        (...unstalled, resumes)            |
```

The semaphore acts as a bounded counter: math increments it (up to max=2) to indicate a half is ready; pack decrements it to indicate a half has been freed. The `STALL_ON_MAX` and `STALL_ON_ZERO` conditions prevent the producer from overrunning the consumer and vice versa.

## The `tensix_sync()` Barrier

The `tensix_sync()` function provides a **full pipeline drain barrier**:

```cpp
inline void tensix_sync()
{
    store_blocking(&pc_buf_base[1], 0);
}
```

The `pc_buf_base` is a memory-mapped register array connected to the Tensix Engine's instruction interface. Writing to offset 1 via `store_blocking()` forces the RISC-V core to wait until **all** previously issued Tensix instructions -- across all execution units (unpacker, math, SFPU, packer, configuration) -- have fully retired.

The `store_blocking()` function uses a store-then-readback-then-dependent-instruction sequence:

```cpp
asm volatile(
    "sw %[raw], (%[ptr])\n\t"  // Store 0 to pc_buf_base[1]
    "lw %[raw], (%[ptr])\n\t"  // Read back from same address
    "and x0, x0, %[raw]\n\t"
    : [raw] "+r"(raw) : [ptr] "r"(ptr) : "memory");
```

The readback load is ordered after the store (same address), and the `and` instruction creates a data dependency that blocks the pipeline until the load completes. The memory clobber prevents the compiler from reordering any subsequent memory accesses before this barrier.

The `tensix_sync()` barrier is used at three critical points:

1. **After `run_kernel()` returns** -- In TRISC `main()`, to ensure all compute and pack instructions have completed before writing the completion mailbox.
2. **Before math-pack sync initialization** -- In `_llk_math_pack_sync_init_()`, to ensure any prior pipeline state is flushed before resetting semaphores.
3. **During `_llk_pack_dest_init_()`** -- To ensure a clean starting state for the packer's destination tracking.

## The `TTI_STALLWAIT` Instruction

The `TTI_STALLWAIT` instruction is the primary pipeline flow-control mechanism in the Tensix ISA. Unlike `tensix_sync()` which drains everything, `STALLWAIT` can selectively stall specific pipeline stages until specific conditions are met. Its general form is:

```
TTI_STALLWAIT(stall_what, wait_for_what)
```

Where:
- **`stall_what`** specifies which pipeline stages to stall.
- **`wait_for_what`** specifies what condition to wait for.

**Stall targets:**

| Constant | Block |
|:---------|:------|
| `p_stall::STALL_MATH` | FPU / math pipeline |
| `p_stall::STALL_SFPU` | SFPU / vector pipeline |
| `p_stall::STALL_SYNC` | Sync unit (semaphore operations) |
| `p_stall::STALL_TDMA` | TDMA / packer DMA |
| `p_stall::STALL_THCON` | ThCon (scalar) unit |
| `p_stall::STALL_UNPACK` | Unpackers |
| `p_stall::STALL_CFG` | Configuration register writes |

**Wait conditions:**

| Constant | Meaning |
|:---------|:--------|
| `p_stall::MATH` | Wait for FPU to finish |
| `p_stall::PACK` | Wait for packer to finish |
| `p_stall::UNPACK0` / `UNPACK1` | Wait for unpacker 0 / 1 |
| `p_stall::SRCA_VLD` | Wait for Source A data valid |
| `p_stall::SRCB_VLD` | Wait for Source B data valid |
| `p_stall::TRISC_CFG` | Wait for TRISC config writes to commit |
| `p_stall::WAIT_SFPU` / `SFPU1` | Wait for SFPU to finish |
| `p_stall::NONE` | No wait -- execute immediately |

Multiple targets and conditions can be combined with bitwise OR. Common combinations seen in the matmul kernel:

| Code | Meaning |
|:-----|:--------|
| `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD)` | Stall math until Source A data is valid (unpacker finished loading it) |
| `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD)` | Stall math until Source B data is valid |
| `TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG)` | Stall unpacker until TRISC configuration writes are committed |
| `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK)` | Stall math until packer finishes (used before clearing Dest) |
| `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH \| p_stall::WAIT_SFPU)` | Stall config writes until math and SFPU complete |
| `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK)` | Stall config writes until packer completes |
| `TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::PACK)` | Stall TDMA (packer's DMA) until packer finishes |
| `TTI_STALLWAIT(p_stall::STALL_TDMA \| p_stall::STALL_THCON, p_stall::PACK)` | Stall TDMA and ThCon until packer finishes |

These stalls are zero-cost when the waited-upon condition is already met (the instruction is a NOP in that case). They only introduce latency when there is an actual data hazard, making them suitable for fine-grained pipeline synchronization.

## Putting It All Together: Matmul Synchronization Flow

For a minimal single-tile matmul with `DstSync::SyncHalf`, the synchronization across all three threads follows this sequence:

```
TRISC0 (Unpack)              TRISC1 (Math)                TRISC2 (Pack)
      |                            |                            |
[hw_configure]              [matmul_init]                [pack_hw_configure]
[matmul_init]               [pack_sync_init]             [pack_init]
      |                     [hw_configure]               [pack_dest_init]
      |                            |                            |
      |                     [wait_for_dest_available]           |
      |                       SEMWAIT(MATH_PACK < max)          |
      |                       (sem=0 < 2, proceed)              |
      |                            |                            |
[wait_for_next_context]            |                            |
  (UNPACK_SYNC < 2, proceed)       |                            |
[configure addresses]              |                            |
[SEMPOST(UNPACK_SYNC)]             |                            |
[STALLWAIT(UNPACK, TRISC_CFG)]     |                            |
[UNPACR -> SrcA, SrcB]            |                            |
[MOP(unpack)]                      |                            |
      |                      [math_matmul]                      |
[SEMGET(UNPACK_SYNC)]         (waits on SrcA/B valid            |
[switch_config_context]        via hardware Dvalid flags)       |
      |                       [MVMUL x N]                       |
      |                            |                            |
      |                     [dest_section_done]                 |
      |                       set_math_semaphores()             |
      |                       SEMPOST(MATH_PACK)                |
      |                       (sem: 0 -> 1)                     |
      |                       dest_section_flip()               |
      |                            |                     [wait_for_math_done]
      |                            |                       SEMWAIT(MATH_PACK > 0)
      |                            |                       (sem=1, proceed)
      |                            |                     [pack tiles from Dest]
      |                            |                     [pack_dest_section_done]
      |                            |                       STALLWAIT(MATH, PACK)
      |                            |                       ZEROACC(clear half)
      |                            |                       SEMGET(MATH_PACK)
      |                            |                       (sem: 1 -> 0)
      |                            |                       flip_packer_dest_offset
      |                            |                       select_packer_dest_regs
      |                            |                            |
[tensix_sync]               [tensix_sync]                [tensix_sync]
[mailbox = 0xFF]            [mailbox = 0xFF]             [mailbox = 0xFF]
```

The critical invariants maintained by this protocol:

1. **Unpack never overwrites an in-use configuration context.** The `UNPACK_SYNC` semaphore and `wait_for_next_context()` prevent TRISC0 from writing to a context that the Tensix Unpack hardware is still reading.

2. **Math never writes to a Destination half being packed.** The `MATH_PACK` semaphore with `STALL_ON_MAX` prevents math from proceeding when both halves are occupied.

3. **Pack never reads from a Destination half being written.** The `MATH_PACK` semaphore with `STALL_ON_ZERO` prevents the packer from reading until math signals completion.

4. **Pack clears the Destination half before releasing it to math.** The `ZEROACC` instruction in `_llk_pack_dest_section_done_()` ensures math starts with a clean accumulator.

5. **The unpack-math handoff is hardware-mediated.** The Source register validity flags (`SRCA_VLD` / `SRCB_VLD`) provide implicit synchronization between unpack and math without software semaphores.

6. **The pipeline drains completely before completion signaling.** `tensix_sync()` after `run_kernel()` ensures no instructions remain in flight when the mailbox completion value is written.

---

**Next:** [Chapter 4 -- Minimal LLK API Call Sequences for MatMul](../ch4_matmul_api_calls/index.md)
