# The I/O Layer: `llk_io/` Headers

## Location and files

```
tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/
    llk_io.h           -- Minimal base (includes circular_buffer_interface.h)
    llk_io_pack.h      -- Pack-side (TRISC2): wait_for_free_tiles, push_tiles, push_to_brisc
    llk_io_unpack.h    -- Unpack-side (TRISC0): wait_tiles, pop_tiles, matmul stagger
    llk_operands.h     -- Input operand metadata accessors
    llk_outputs.h      -- Output operand metadata accessors
```

This layer manages the **circular buffer interactions** that coordinate data flow between the three TRISC processors and the data movement engines (BRISC/NCRISC). Unlike the `llk_api/` wrappers that handle compute operations, the `llk_io/` layer handles the producer/consumer protocol for tiles moving through L1 memory.

## The circular buffer interface

All I/O functions interact with circular buffers through `get_local_cb_interface()`, which returns a struct containing:

| Field | Description |
|-------|-------------|
| `fifo_rd_ptr` | Read pointer (consumer side) |
| `fifo_wr_ptr` | Write pointer (producer side) |
| `fifo_wr_tile_ptr` | Offset within current write batch |
| `fifo_page_size` | Size of one tile in 16-byte words |
| `fifo_num_pages` | Total capacity in tiles |
| `fifo_limit` | Wrap-around boundary address |
| `fifo_size` | Total buffer size in words |
| `tiles_received` | Local count of tiles pushed (TRISC2-maintained) |
| `tiles_acked` | Local count of tiles consumed (TRISC0-maintained) |

The `llk_io.h` base file is minimal -- it simply includes the circular buffer interface:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io.h, lines 1-8

#pragma once
#include <cstdint>
#include "internal/circular_buffer_interface.h"
```

## Unpack-side I/O (`llk_io_unpack.h`)

TRISC0 (the unpacker processor) uses `llk_io_unpack.h` to wait for tiles from producers and release consumed tiles.

### `llk_wait_tiles()` -- blocking wait for input tiles

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_unpack.h, lines 66-81

inline void llk_wait_tiles(int operand, std::int32_t num_tiles) {
    DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT");
    std::uint32_t input = operand;
    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;
    uint16_t num_tiles_recv;
    do {
        tiles_received = (std::uint16_t)reg_read((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - get_local_cb_interface(input).tiles_acked;
    } while (num_tiles_recv < num_tiles_u);

    apply_mm_stagger(operand);
}
```

This function spins in a polling loop, reading the hardware `tiles_received_ptr` register and comparing against the locally-tracked `tiles_acked` count. The difference gives the number of available tiles. Once enough tiles are available, it optionally applies a matmul stagger delay (see below).

All counter arithmetic uses 16-bit unsigned types, which correctly handles wraparound when counters overflow past 65535.

### `llk_pop_tiles()` -- release consumed tiles

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_unpack.h, lines 84-100

inline void llk_pop_tiles(const std::int32_t operand, const std::int32_t num_tiles,
                           const std::int32_t block_c_dim = 0) {
    std::uint32_t input = operand;
    volatile tt_reg_ptr std::uint32_t* tiles_acked_ptr =
        (volatile std::uint32_t*)((((volatile std::uint32_t)get_cb_tiles_acked_ptr(operand)) >> 2)
                                   & 0x3ffff);
    std::uint32_t num_words = num_tiles * get_local_cb_interface(operand).fifo_page_size;

    get_local_cb_interface(input).tiles_acked += num_tiles;
    TT_SETDMAREG(0, get_local_cb_interface(input).tiles_acked, 0, LO_16(4));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::UNPACK);
    TT_STOREREG(4, (std::uint32_t)&tiles_acked_ptr[0]);
    get_local_cb_interface(input).fifo_rd_ptr += num_words;

    if (get_local_cb_interface(input).fifo_rd_ptr >= get_local_cb_interface(input).fifo_limit) {
        get_local_cb_interface(input).fifo_rd_ptr -= get_local_cb_interface(input).fifo_size;
    }
}
```

Key implementation details:

1. **Tiles-acked update**: The local counter is incremented, then the new value is written to a hardware-visible location using Tensix DMA register instructions (`TT_SETDMAREG` + `TT_STOREREG`). The `TTI_STALLWAIT` ensures the unpacker has finished before the register write takes effect.

2. **Read pointer advancement**: The `fifo_rd_ptr` advances by `num_tiles * fifo_page_size` words. If it passes the FIFO limit, it wraps around by subtracting the total FIFO size.

3. **Address masking**: The `tiles_acked_ptr` computation (`>> 2 & 0x3ffff`) converts a byte address to a Tensix 4-byte word address within the L1 address space.

### Matmul stagger

The `apply_mm_stagger()` function is a power-management mechanism that prevents all Tensix cores from starting matmul operations simultaneously, which could cause di/dt (current rate-of-change) voltage drops:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_unpack.h, lines 19-63

inline __attribute__((__always_inline__)) void apply_mm_stagger(int operand) {
#if MM_STAGGER_TYPE == 1  // first block stagger
    static bool stagger_applied = false;
    constexpr int stagger_operand = 1;
    constexpr int stagger_delay_in_cycles = MM_STAGGER_VALUE;
    if (stagger_applied == false && operand == stagger_operand) {
        stagger_applied = true;
        // ...
        uint32_t my_logical_y = (noc_id_logical_reg >> NOC_ADDR_NODE_ID_BITS)
                                & NOC_NODE_ID_MASK;
        if (my_logical_y & 0x1) {
            wait(stagger_delay_in_cycles);
        }
    }
    // ... types 2 (every block) and 3 (hybrid, every 20th block) ...
#endif
}
```

Three stagger modes are supported (compile-time selected via `MM_STAGGER_TYPE`):
- **Type 1**: Stagger the first matmul block only (one-shot).
- **Type 2**: Stagger every matmul block.
- **Type 3**: Hybrid -- stagger every 20th block as a compromise between latency and power safety.

Only cores on odd rows of the Tensix grid are delayed, creating a checkerboard activation pattern.

## Pack-side I/O (`llk_io_pack.h`)

TRISC2 (the packer processor) uses `llk_io_pack.h` to wait for free output buffer space and signal tile completion.

### `llk_wait_for_free_tiles()` -- blocking wait for output space

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h, lines 18-41

template <bool skip_sync, bool wait_for_blocks, bool brisc_pack>
inline void llk_wait_for_free_tiles(const std::int32_t operand,
                                     const std::int32_t num_tiles) {
    DeviceZoneScopedSumN2("CB-COMPUTE-RESERVE-BACK");
    std::uint32_t output = operand;

    volatile tt_reg_ptr std::uint32_t* tiles_acked_ptr = get_cb_tiles_acked_ptr(operand);
    volatile tt_reg_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);

    // Use the local tiles_received, not the hardware register, to avoid a
    // data race with the packer (which updates tiles_received_ptr)
    uint16_t tiles_received = get_local_cb_interface(output).tiles_received;

    std::int32_t free_tiles;
    do {
        std::uint16_t tiles_acked = (std::uint16_t)reg_read((std::uint32_t)tiles_acked_ptr);
        std::uint16_t free_tiles_wrap =
            get_local_cb_interface(output).fifo_num_pages - (tiles_received - tiles_acked);
        free_tiles = (std::int32_t)free_tiles_wrap;
    } while (free_tiles < num_tiles);
}
```

This is the inverse of `llk_wait_tiles()`: it computes free space as `fifo_num_pages - (tiles_received - tiles_acked)`. The function deliberately reads the local `tiles_received` copy rather than the hardware register to avoid a data race -- the packer asynchronously updates `tiles_received_ptr`, so reading it could produce a stale value that underestimates available space.

### `llk_push_to_brisc()` -- signal tile completion

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h, lines 43-68

inline void llk_push_to_brisc(const std::int32_t operand, const std::int32_t num_tiles,
                               const std::int32_t num_words) {
    std::uint32_t output = operand;

    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr_tensix =
        (volatile tt_l1_ptr std::uint32_t*)
            ((((volatile std::uint32_t)get_cb_tiles_received_ptr(operand)) >> 2) & 0x3ffff);

    get_local_cb_interface(output).tiles_received += num_tiles;
    uint16_t tiles_received_new = get_local_cb_interface(output).tiles_received;

    TT_SETDMAREG(0, tiles_received_new, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);  // wait for pack to finish
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, (uint32_t)&tiles_received_ptr_tensix[0]);
}
```

This function:
1. Increments the local `tiles_received` counter.
2. Uses `TT_SETDMAREG` to load the new counter value into a Tensix GPR.
3. Stalls until the packer hardware has finished (`STALLWAIT`).
4. Writes the counter to the hardware-visible `tiles_received_ptr` using `TT_STOREREG`.

The consumer side (TRISC0 or BRISC) polls this register, so using a Tensix instruction for the write ensures the value is only visible after the packer has actually committed the tile data to L1.

### `llk_push_tiles()` -- advance write pointer and signal

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h, lines 71-84

template <bool push_blocks, bool brisc_pack>
inline void llk_push_tiles(const std::int32_t operand, const std::int32_t num_tiles) {
    std::uint32_t output = operand;
    std::uint32_t num_words = num_tiles * get_local_cb_interface(operand).fifo_page_size;

    get_local_cb_interface(output).fifo_wr_ptr += num_words;
    get_local_cb_interface(output).fifo_wr_tile_ptr = 0;

    if (get_local_cb_interface(output).fifo_wr_ptr >= get_local_cb_interface(output).fifo_limit) {
        get_local_cb_interface(output).fifo_wr_ptr -= get_local_cb_interface(output).fifo_size;
    }

    llk_push_to_brisc(operand, num_tiles, num_words);
}
```

After advancing the write pointer (with wraparound), it resets `fifo_wr_tile_ptr` to 0 (since the batch of tiles being written is now complete) and calls `llk_push_to_brisc()` to make the tiles visible to consumers.

## TRISC synchronization model

The I/O layer implements a **lock-free producer/consumer** protocol between the three TRISC processors using two counters per circular buffer:

```
                    tiles_received (counter)
                    ┌─────────────┐
    TRISC2 (pack) ──┤  writes     │──► TRISC0 (unpack) reads via reg_read()
                    └─────────────┘

                    tiles_acked (counter)
                    ┌─────────────┐
    TRISC0 (unpack)─┤  writes     │──► TRISC2 (pack) reads via reg_read()
                    └─────────────┘
```

- **TRISC2** increments `tiles_received` after packing tiles, using Tensix store instructions to ensure visibility ordering.
- **TRISC0** increments `tiles_acked` after consuming tiles, similarly using Tensix store instructions.
- **TRISC1** (math) does not participate in I/O directly -- it synchronizes with TRISC2 via the dest register semaphore (`llk_packer_wait_for_math_done()` / `llk_packer_set_math_semaphore()`).
- Each side reads the other's counter via `reg_read()` in a polling loop.

This design avoids locks or memory barriers beyond the Tensix `STALLWAIT` instructions, which are necessary only to order the packer/unpacker hardware operations relative to the counter writes.

## Block-oriented aliases

For convenience, the layer provides block-granularity aliases:

```cpp
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_unpack.h, line 102
inline void llk_wait_blocks(int operand, std::int32_t num_blocks) {
    llk_wait_tiles(operand, num_blocks);
}

// tt_metal/hw/ckernels/wormhole_b0/metal/llk_io/llk_io_pack.h, lines 86-92
inline void llk_wait_for_free_blocks(const std::int32_t operand,
                                      const std::int32_t num_blocks) {
    llk_wait_for_free_tiles<false, true>(operand, num_blocks);
}

inline void llk_push_blocks(const std::int32_t operand, const std::int32_t num_blocks) {
    llk_push_tiles<true>(operand, num_blocks);
}
```

These are functionally identical to their tile counterparts -- blocks and tiles use the same counter mechanism.

## Integration with `llk_pack_common.h`

`llk_io_pack.h` includes `llk_pack_common.h` from tt-llk (line 12), which provides the underlying pack hardware control (`_llk_packer_wait_for_math_done_()`, `_llk_packer_set_math_semaphore_()`, etc.). The I/O layer builds the circular-buffer management on top of this hardware abstraction.

---

**Next:** [`sfpu_extensions.md`](./sfpu_extensions.md)
