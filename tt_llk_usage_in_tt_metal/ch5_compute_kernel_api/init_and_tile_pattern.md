# Init and Tile Pattern

Every compute kernel in TT-Metal follows a canonical control-flow pattern. This section documents that pattern, the role of `compute_kernel_hw_startup()`, the `state_configure()` optimization, and the synchronization protocol between the three TRISC threads.

## The Standard Compute Kernel Pattern

```
1. Hardware startup          (one-time, beginning of kernel)
2. Operation init            (per-operation configuration)
3. Loop over blocks:
   a. cb_wait_front          (wait for input tiles from data-movement kernels)
   b. tile_regs_acquire      (MATH acquires DST register)
   c. Compute loop           (unpack + math per tile)
   d. tile_regs_commit       (MATH releases DST to PACK)
   e. tile_regs_wait         (PACK waits for MATH to finish)
   f. Pack loop              (pack tiles from DST to output CB)
   g. tile_regs_release      (PACK releases DST back to MATH)
   h. cb_pop_front           (free input CB space)
   i. cb_push_back           (make output tiles visible to consumer)
```

## A Complete Example

The following is from `tt_metal/kernels/compute/eltwise_binary.cpp`, which implements element-wise binary operations (add, sub, mul):

```cpp
// tt_metal/kernels/compute/eltwise_binary.cpp
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    // Step 1+2: Hardware + operation init
    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    binary_tiles_init<true, ELTWISE_OP_TYPE>(cb_in0, cb_in1);

    // Step 3: Main loop
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        // Step 3a: Wait for input tiles
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        // Step 3b: MATH acquires DST
        tile_regs_acquire();

        // Step 3c: Compute loop (unpack + math)
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            ELTWISE_OP(cb_in0, cb_in1, i, i, i);  // e.g. add_tiles
        }

        // Step 3d: MATH signals done
        tile_regs_commit();

        // Step 3e: PACK waits for MATH
        tile_regs_wait();

        // Step 3f: Pack loop
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);
        }

        // Step 3g: PACK releases DST
        tile_regs_release();

        // Step 3h-3i: Free inputs, publish outputs
        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

## Phase 1: Hardware Startup

### `compute_kernel_hw_startup()`

Source: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`, line 41

This function performs the one-time hardware initialization that must happen before any operation. It configures all three TRISC threads via MMIO writes:

```cpp
ALWI void compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init<false, false, false>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>(ocb)));

    ComputeKernelSentinel::instance().set_srca(icb0).set_srcb(icb1).set_pack(ocb);
}
```

What each LLK call does at the hardware level:

| LLK Call | Thread | What It Configures |
|----------|--------|--------------------|
| `llk_unpack_hw_configure` | TRISC0 | Unpacker source/dest data formats, tile dimensions, face counts via config register writes |
| `llk_math_pack_sync_init` | TRISC1 | Math-to-pack semaphore initialization (DST double-buffer section control) |
| `llk_math_hw_configure` | TRISC1 | ALU source data formats, accumulation mode settings |
| `llk_pack_hw_configure` | TRISC2 | Packer output data format, tile dimensions, L1 destination configuration |
| `llk_pack_init` | TRISC2 | Pack address generator counters and stride registers |
| `llk_pack_dest_init` | TRISC2 | DST register read pointer, section offsets for double-buffering |

**Critical constraint:** This function performs slow MMIO writes and requires execution units to be idle. It must be called only once, at the very beginning of the kernel, before any other Compute API calls. Calling it mid-kernel can cause race conditions because the execution units may still be processing tiles.

### Relationship to Full `_init()` Functions

In practice, many kernels skip an explicit `compute_kernel_hw_startup()` call because the full `_init()` functions (like `binary_op_init_common()`, `mm_init()`, `reduce_init()`) already contain all the same LLK calls. The full init functions are self-contained -- they configure all three TRISC threads and do not require a preceding `compute_kernel_hw_startup()`.

The `compute_kernel_hw_startup()` function is useful when:
- A kernel performs multiple different operations and wants to do HW setup once with a known-good CB configuration.
- Subsequent operations use only "short init" variants (`mm_init_short()`, `copy_tile_to_dst_init_short()`) that skip pack/HW configuration.

## Phase 2: Operation Init

After hardware startup, the kernel calls an operation-specific init function. These come in two flavors:

### Full Init

Full inits configure all three threads. Examples:
- `binary_op_init_common()` + `binary_tiles_init<true, ...>()`
- `mm_init()`
- `mm_block_init()`
- `reduce_init()`
- `init_bcast<op, dim>()`

### Short Init

Short inits only reconfigure the unpack and math threads, skipping pack/HW configuration. They are used when switching between operations within a kernel (e.g., switching from a copy to a matmul):
- `mm_init_short()` -- reconfigures unpack + math for matmul
- `copy_tile_to_dst_init_short()` -- reconfigures unpack + math for datacopy
- `add_bcast_rows_init_short()` -- reconfigures unpack + math for row broadcast add
- `binary_tiles_init<false, ...>()` -- math-only reconfiguration

### `_with_dt` Variants

When switching between operations that use different data formats, the `_with_dt` (data type) variants first reconfigure the source register data formats before doing the short init:

```cpp
// tt_metal/hw/inc/api/compute/matmul.h, line 206
ALWI void mm_init_short_with_dt(uint32_t in0_cb_id, uint32_t in1_cb_id,
                                 uint32_t c_in_old_srca,
                                 const uint32_t transpose = 0) {
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(
            c_in_old_srca, in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(
          c_in_old_srca, in1_cb_id)));
    mm_init_short(in0_cb_id, in1_cb_id, transpose);
}
```

## Phase 3: The Compute Loop

### Step 3a: Wait for Input Tiles

```cpp
cb_wait_front(cb_in0, per_core_block_size);  // TRISC0: llk_wait_tiles
cb_wait_front(cb_in1, per_core_block_size);  // TRISC0: llk_wait_tiles
cb_reserve_back(cb_out0, per_core_block_size);  // TRISC2: llk_wait_for_free_tiles
```

`cb_wait_front` runs on TRISC0 (unpack) and blocks until the specified number of tiles are available in the input CB. The data-movement reader kernel pushes tiles into this CB.

`cb_reserve_back` runs on TRISC2 (pack) and blocks until there is space in the output CB for the specified number of tiles. The data-movement writer kernel pops tiles from this CB.

### Steps 3b-3d: Math Phase (DST Owned by MATH)

```cpp
tile_regs_acquire();     // TRISC1: llk_math_wait_for_dest_available()

for (...) {
    // Each op call contains UNPACK() + MATH() calls
    add_tiles(cb_in0, cb_in1, i, i, i);
}

tile_regs_commit();      // TRISC1: llk_math_dest_section_done<DST_ACCUM_MODE>()
```

`tile_regs_acquire()` runs on TRISC1 and blocks until the DST register is available (not currently being packed). The math thread then "owns" DST and can write computed results into it.

Each tile operation (e.g., `add_tiles`) contains both `UNPACK()` and `MATH()` calls. Remember that these are compiled into separate TRISC binaries. The unpack and math threads synchronize implicitly via the SRC register semaphores -- the math engine waits for the unpacker to fill the SRC registers before consuming them.

`tile_regs_commit()` runs on TRISC1 and signals that the DST contents are ready for packing.

### Steps 3e-3g: Pack Phase (DST Owned by PACK)

```cpp
tile_regs_wait();        // TRISC2: llk_packer_wait_for_math_done()

for (...) {
    pack_tile(i, cb_out0);  // TRISC2: llk_pack(...)
}

tile_regs_release();     // TRISC2: llk_pack_dest_section_done<DST_ACCUM_MODE>()
```

`tile_regs_wait()` runs on TRISC2 and blocks until TRISC1 has committed the DST. The pack thread then reads tiles from DST and writes them into the output CB in L1.

`tile_regs_release()` runs on TRISC2 and signals that the pack thread is done reading DST, making it available for the math thread again.

### Steps 3h-3i: CB Bookkeeping

```cpp
cb_pop_front(cb_in0, per_core_block_size);   // TRISC0: llk_pop_tiles
cb_pop_front(cb_in1, per_core_block_size);   // TRISC0: llk_pop_tiles
cb_push_back(cb_out0, per_core_block_size);  // TRISC2: llk_push_tiles
```

`cb_pop_front` advances the read pointer of the input CB, freeing space for the reader data-movement kernel. `cb_push_back` advances the write pointer of the output CB, making packed tiles visible to the writer data-movement kernel.

## The DST Double-Buffer Protocol

The DST register supports double-buffering: while the math thread writes to one section, the pack thread can read from the other. The synchronization primitives manage this:

```
MATH thread:                    PACK thread:
  tile_regs_acquire()  <------+
  compute tiles to DST[0..N]  |   (idle, waiting)
  tile_regs_commit()   ------>+
                              |   tile_regs_wait()
  (can start next batch)      |   pack tiles from DST[0..N]
  tile_regs_acquire()  <------+   tile_regs_release()
  ...                             ...
```

The deprecated `acquire_dst()` / `release_dst()` API combined both math and pack synchronization into single calls, which prevented overlapping math and pack work. The newer four-function API (`tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release`) enables true pipelining between TRISC1 and TRISC2.

## `state_configure()` -- Avoiding Redundant Reconfiguration

Source: `tt_metal/hw/inc/api/compute/sentinel/compute_kernel_sentinel.h`

Every init function begins with a call to `state_configure()`:

```cpp
// sentinel/compute_kernel_sentinel.h, lines 315-326
template <Operand operand = Operand::SRCA>
ALWI void state_configure(uint32_t cb, uint32_t call_line) {
    ComputeKernelSentinel::instance()
        .reconfigure_single_operand<operand>(cb, call_line);
}

template <Operand operand_a = Operand::SRCA, Operand operand_b = Operand::SRCB>
ALWI void state_configure(uint32_t cb_a, uint32_t cb_b, uint32_t call_line) {
    ComputeKernelSentinel::instance()
        .reconfigure_dual_operand<operand_a, operand_b>(cb_a, cb_b, call_line);
}

ALWI void state_configure(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out,
                            uint32_t call_line) {
    ComputeKernelSentinel::instance()
        .reconfig_all_operands(cb_a, cb_b, cb_out, call_line);
}
```

The `ComputeKernelSentinel` singleton tracks which CB IDs are currently configured for each operand (SRCA, SRCB, PACK). The `__builtin_LINE()` default argument captures the call site for debugging.

When an init function is called with the same CB IDs that are already configured, `state_configure()` can skip redundant reconfiguration. This optimization is important for kernels that perform the same operation in a loop -- the first iteration does the full configuration, and subsequent iterations detect that the state is already correct.

The sentinel also records the configured state after `compute_kernel_hw_startup()` completes:

```cpp
// compute_kernel_hw_startup.h, line 52
ComputeKernelSentinel::instance()
    .set_srca(icb0).set_srcb(icb1).set_pack(ocb);
```

## Additional HW Startup Utilities

### `math_enable_fp32_dest_acc()` / `math_disable_fp32_dest_acc()`

Source: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`, lines 91-116

Lightweight mid-kernel reconfiguration that toggles FP32 accumulation in the ALU and SFPU without touching unpacker or packer state:

```cpp
ALWI void math_enable_fp32_dest_acc() {
    MATH((llk_math_set_fp32_dest_acc(true)));
}

ALWI void math_disable_fp32_dest_acc() {
    MATH((llk_math_set_fp32_dest_acc(false)));
}
```

These are safe to call mid-kernel because they only modify the `ALU_ACC_CTRL` register bits.

## Summary of the Full Stack

To trace one tile through the entire stack, consider `add_tiles(cb0, cb1, 0, 0, 0)`:

```
Kernel code:         add_tiles(cb0, cb1, 0, 0, 0)
                          |
Compute API layer:   UNPACK((llk_unpack_AB(cb0, cb1, 0, 0)))
(eltwise_binary.h)   MATH((llk_math_eltwise_binary<ELWADD, ...>(cb0, cb1, 0, true)))
                          |
LLK API layer:       llk_unpack_AB() -> _llk_unpack_AB_()
(llk_unpack_AB_api.h)    configures SRC register addresses, triggers unpacker
                     llk_math_eltwise_binary() -> _llk_math_eltwise_binary_()
(llk_math_binary_api.h)  executes MOP (ckernel_template), writes result to DST
                          |
Hardware:            Unpacker reads tiles from L1 into SRC A/B
                     FPU/SFPU executes element-wise add
                     Result accumulates in DST register
```

The Compute API layer adds zero overhead (all `ALWI`-inlined), provides a clean separation of concerns (kernel authors never call LLK functions directly), and handles the TRISC dispatch transparently via the `MATH()` / `UNPACK()` / `PACK()` macros.

---

**Next:** [Chapter 6 -- Op to Kernel Mapping](../ch6_op_to_kernel_mapping/index.md)
