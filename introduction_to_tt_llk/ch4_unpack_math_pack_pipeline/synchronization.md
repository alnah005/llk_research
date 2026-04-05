# Synchronization

## Overview

The three pipeline stages (unpack, math, pack) run on separate RISC-V processors (TRISC0, TRISC1, TRISC2) and execute concurrently. Synchronization is needed at two levels:

1. **Between TRISC threads and the Tensix hardware engines** they control (unpacker, FPU/SFPU, packer).
2. **Between the three TRISC threads** to ensure data dependencies are respected: math must not read Source registers until unpack has filled them, and pack must not read the Destination register until math has written it.

## The `DstSync` Enum

The `DstSync` enum (defined in `tt_llk_wormhole_b0/llk_lib/llk_defs.h`) controls how the Destination register is shared between math and pack:

```cpp
enum DstSync
{
    SyncHalf = 0,  // Double-buffered: math and pack operate on opposite halves
    SyncFull = 1,  // Single-buffered: math and pack take turns on the full register
};
```

### `SyncHalf` -- Double-Buffered Mode

In `SyncHalf` mode, the Destination register is split into two halves. While math writes to one half, the packer reads from the other half. After each tile, both sides flip:

```
Time -->
Math:  [ Write Half 0 ] [ Write Half 1 ] [ Write Half 0 ] ...
Pack:       idle         [ Read Half 0 ]  [ Read Half 1 ]  ...
```

The semaphore `MATH_PACK` is initialized to 2 (the number of available buffer halves):

```cpp
// In _llk_math_pack_sync_init_<DstSync::SyncHalf>():
TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
```

This allows math to write up to 2 tiles ahead of pack before stalling.

### `SyncFull` -- Single-Buffered Mode

In `SyncFull` mode, math uses the entire Destination register for one tile, then pack reads the entire register. There is no overlap between math and pack on the Destination register:

```
Time -->
Math:  [ Write Full ] [    stall    ] [ Write Full ] ...
Pack:      idle        [ Read Full ]      idle        ...
```

The semaphore is initialized to 1:

```cpp
// In _llk_math_pack_sync_init_<DstSync::SyncFull>():
TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
```

`SyncFull` is used when the operation needs the entire Destination register (e.g., large reductions or operations that accumulate across the full register).

## Destination Register Double Buffering

Two functions in `cmath_common.h` manage the destination buffer flip in `SyncHalf` mode:

### `dest_section_flip()`

Called by TRISC1 after completing math on a tile. It updates the internal destination offset ID and reprograms the math destination base address register:

```cpp
inline void dest_section_flip()
{
    update_dest_offset_id();
    std::uint32_t base_addr = get_dest_buffer_base();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);
}
```

The `update_dest_offset_id()` toggles a counter between 0 and 1. When the offset is 0, math writes to the first half (starting at address 0); when 1, math writes to the second half (starting at `DEST_REGISTER_HALF_SIZE`).

### `math_dest_wait()`

Called by TRISC1 before starting math on a new tile. It stalls until the packer has finished with the destination half that math wants to write to:

```cpp
inline void math_dest_wait()
{
    TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
                semaphore::t6_sem(semaphore::MATH_PACK),
                p_stall::STALL_ON_MAX);
}
```

This waits until the `MATH_PACK` semaphore is below its maximum value, meaning at least one destination half is available for writing.

## Semaphore-Based Sync Between Unpack/Math/Pack

The pipeline uses two primary semaphores:

### `UNPACK_SYNC` -- Unpack Context Coordination

This semaphore coordinates between TRISC0 and the unpacker hardware engine:

| Action | Who | When |
|:-------|:----|:-----|
| `semaphore_post(semaphore::UNPACK_SYNC)` | TRISC0 | After programming unpacker config registers |
| `t6_semaphore_get(semaphore::UNPACK_SYNC)` | Tensix engine | After the MOP completes, releasing the context |

This enables double-buffered context switching: TRISC0 programs context N+1 while the unpacker hardware executes context N.

### `MATH_PACK` -- Math/Pack Destination Coordination

This semaphore coordinates between TRISC1 (math) and TRISC2 (pack) over access to the Destination register:

| Action | Who | When |
|:-------|:----|:-----|
| `math_dest_wait()` | TRISC1 | Before starting math -- waits for a free destination section |
| `set_math_semaphores()` | TRISC1 | After math completes -- signals pack that data is ready |
| `_llk_packer_wait_for_math_done_()` | TRISC2 | Before packing -- waits for math to produce data |
| `_llk_packer_set_math_semaphore_()` | TRISC2 | After packing -- releases destination section for math |

The math-side functions from `llk_math_common.h`:

```cpp
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_()
{
    math_dest_wait();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_()
{
    set_math_semaphores();       // post MATH_PACK semaphore
    if constexpr (Dst == DstSync::SyncHalf)
    {
        math_sync_tile_dst_index = 0;
        dest_section_flip();     // switch to other half
    }
}
```

The pack-side functions from `llk_pack_common.h`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_section_done_()
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);  // wait for pack HW
    // Clear the packed destination section
    TTI_ZEROACC(CLEAR_MODE, ADDR_MOD_1, ...);
    // Release semaphore so math can write again
    _llk_packer_set_math_semaphore_<p_stall::NONE>();
    if constexpr (Dst == DstSync::SyncHalf)
    {
        flip_packer_dest_offset_id();
        select_packer_dest_registers<Dst>();
    }
}
```

### `_llk_math_pack_sync_init_`

This function initializes the synchronization state at the beginning of a compute kernel. It must be called before any math or pack operations:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_()
{
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {};  // drain previous packs
    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_1);
    }
    else  // SyncHalf
    {
        TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
    }
    reset_dest_offset_id();
    set_dest_section_base<StartZero>();
}
```

## Stall Instructions

The `TTI_STALLWAIT` instruction is the fundamental synchronization primitive at the hardware level. It stalls one or more engines until a specified condition is met:

```cpp
TTI_STALLWAIT(stall_target, stall_condition);
```

Common stall targets and conditions used in the pipeline:

| Stall Pattern | Meaning |
|:-------------|:--------|
| `TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG)` | Stall unpacker until TRISC config writes complete |
| `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD)` | Stall math until Source A has valid data |
| `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD)` | Stall math until Source B has valid data |
| `TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH)` | Stall SFPU until FPU is done |
| `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)` | Stall math until SFPU is done |
| `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH \| p_stall::SFPU1)` | Stall config writes until math and SFPU complete |
| `TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::PACK)` | Stall TDMA until pack completes |
| `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK)` | Stall math engine until pack completes |

The stall target selects which engine to stall (UNPACK, MATH, SFPU, TDMA, CFG, etc.), and the stall condition selects what event to wait for.

## The `ckernel_template` Class

The `ckernel_template` class (defined in `tt_llk_wormhole_b0/common/inc/ckernel_template.h`) encapsulates a hardware MOP (Micro-Op Program) -- a small instruction loop that executes on the Tensix engine without RISC-V intervention.

### Structure

```
LOOP_OUTER: <m_outer_loop_len>
  m_start_op0
  LOOP_INNER: <m_inner_loop_len>
    m_loop_op0
    m_loop_op1
  END_LOOP_INNER
  m_end_op0
  m_end_op1
END_LOOP_OUTER
```

### Last-Iteration Replacement

The MOP supports replacing the loop body instruction on the last iteration:

```
if (last_inner_loop_iter && last_outer_loop_iter):
    m_loop_op1 = m_loop0_last_instr    // "last outer loop instruction"
else if (last_inner_loop_iter):
    m_loop_op1 = m_loop1_last_instr    // "last inner loop instruction"
else:
    m_loop_op1 = m_loop_op1            // normal instruction
```

This is used to change address modifier behavior on the last iteration (e.g., resetting counters, clearing fidelity phase).

### Usage

```cpp
// 1. Construct with loop counts and body instructions
ckernel_template tmp(outer_loop_count, inner_loop_count, loop_op0, loop_op1);

// 2. Optionally set start/end/last instructions
tmp.set_start_op(start_instruction);
tmp.set_end_op(end_instruction);
tmp.set_last_inner_loop_instr(last_inner_instr);
tmp.set_last_outer_loop_instr(last_outer_instr);

// 3. Program the MOP registers (done once in _init_)
tmp.program();

// 4. Run the MOP (done per-tile in the execute function)
ckernel_template::run();
```

### The `ckernel_unpack_template` Variant

For unpack operations, a more flexible `ckernel_unpack_template` class is available. It supports conditional instruction selection based on a Z-mask, allowing different instructions to execute on different loop iterations:

```
LOOP:
  if (!zmask[iteration]):
    UNPACR_A0
    UNPACR_A1  (if halo enabled)
    UNPACR_A2  (if halo enabled)
    UNPACR_A3  (if halo enabled)
    UNPACR_B   (if B enabled)
  else:
    SKIP_A
    SKIP_B     (if B enabled)
```

This is used in matmul unpacking to reuse one source while cycling through new tiles on the other source, and in SDPA (Scaled Dot-Product Attention) unpacking where Source A broadcasts across multiple Source B tiles.

---

**Next:** [Chapter 5 -- SFPU Operations](../ch5_sfpu_operations/index.md)
