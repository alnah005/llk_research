# Key Concepts

This document covers the three foundational concepts you need to understand before diving into TT-LLK code: the Tensix core's three-processor architecture, the tile-based computation model, and the macro system that compiles a single kernel source into three separate binaries.

## 1. Tensix Core Architecture: Three TRISC Processors

Each Tensix core on a Tenstorrent chip contains three RISC-V processors dedicated to compute operations, called **TRISC0**, **TRISC1**, and **TRISC2**. Each processor is responsible for one stage of the compute pipeline:

| Processor | Guard Macro | Role | Entry Point |
|-----------|------------|------|-------------|
| **TRISC0** | `TRISC_UNPACK` | **Unpack** -- reads tiles from circular buffers in L1 and unpacks them into source registers (SrcA, SrcB) | `unpack_main()` |
| **TRISC1** | `TRISC_MATH` | **Math** -- executes arithmetic operations (FPU or SFPU) on data in source registers, writing results to the DST register | `math_main()` |
| **TRISC2** | `TRISC_PACK` | **Pack** -- reads computed results from the DST register, packs them into the target data format, and writes them to output circular buffers in L1 | `pack_main()` |

All three processors execute **concurrently** on the same Tensix core, forming a three-stage pipeline. They synchronize through hardware semaphores on the shared DST (destination) register.

### Why Three Compilations of the Same Source

A single compute kernel source file (e.g., `eltwise_binary_kernel.cpp`) is compiled **three times** -- once for each TRISC processor. The JIT build system accomplishes this by generating four separate `.cpp` files from the same kernel source (three for the TRISC processors, plus a fourth for SFPU isolation), each with a different preprocessor define.

From `tt_metal/jit_build/genfiles.cpp`, line 137:

```cpp
// Generates TRISC prolog: #define + #include for defines_generated.h
string build_trisc_prolog(const char* trisc_define) {
    ostringstream prolog;
    prolog << "#define " << trisc_define << "\n";
    prolog << "#include \"defines_generated.h\"\n";
    return prolog.str();
}
```

And at lines 196-199, four prologs are generated:

```cpp
const string unpack_prolog = build_trisc_prolog("TRISC_UNPACK");
const string math_prolog = build_trisc_prolog("TRISC_MATH");
const string pack_prolog = build_trisc_prolog("TRISC_PACK");
const string isolate_sfpu_prolog = build_trisc_prolog("TRISC_ISOLATE_SFPU");
```

The kernel source is then wrapped into four separate compilation units (`chlkc_unpack.cpp`, `chlkc_math.cpp`, `chlkc_pack.cpp`, and `chlkc_isolate_sfpu.cpp`). For the simplified `kernel_main()` syntax (line 216):

```cpp
unpack_src = line_directive + simple_kernel_syntax::transform_to_legacy_syntax(
    kernel_content, "chlkc_unpack", "unpack_main");
math_src = line_directive + simple_kernel_syntax::transform_to_legacy_syntax(
    kernel_content, "chlkc_math", "math_main");
pack_src = line_directive + simple_kernel_syntax::transform_to_legacy_syntax(
    kernel_content, "chlkc_pack", "pack_main");
```

Each generated file defines one of `TRISC_UNPACK`, `TRISC_MATH`, or `TRISC_PACK`, causing the `MATH()`, `PACK()`, and `UNPACK()` macros to selectively include or exclude code for that processor.

## 2. The `MATH()`, `PACK()`, and `UNPACK()` Macros

These macros are defined in `tt_metal/hw/inc/api/compute/compute_kernel_api.h` and are the central mechanism for splitting a single kernel into three TRISC-specific binaries.

### Macro Definitions

From `tt_metal/hw/inc/api/compute/compute_kernel_api.h`, lines 19-58:

```cpp
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_matmul_api.h"
#include "llk_math_unary_datacopy_api.h"
#ifndef ARCH_QUASAR
#include "llk_math_binary_api.h"
#include "llk_math_unary_sfpu_api.h"
#include "llk_math_binary_sfpu_api.h"
#include "llk_math_reduce_api.h"
#endif
#define MATH(x) x
#define MAIN math_main()
#else
#define MATH(x)
#endif

#ifdef TRISC_PACK
#include "llk_pack_api.h"
#include "llk_io_pack.h"
#define PACK(x) x
#define MAIN pack_main()
#else
#define PACK(x)
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_matmul_api.h"
#include "llk_unpack_A_api.h"
#ifndef ARCH_QUASAR
#include "llk_unpack_AB_api.h"
#include "llk_unpack_reduce_api.h"
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_untilize_api.h"
#endif
#include "llk_io_unpack.h"
#define UNPACK(x) x
#define MAIN unpack_main()
#else
#define UNPACK(x)
#endif
```

### How the Macros Work

When `TRISC_MATH` is defined (i.e., compiling for TRISC1):
- `MATH(x)` expands to `x` -- the code inside executes.
- `PACK(x)` expands to nothing -- pack code is stripped out.
- `UNPACK(x)` expands to nothing -- unpack code is stripped out.

The same applies symmetrically for the other two processors. For example, `copy_tile` (`tt_metal/hw/inc/api/compute/tile_move_copy.h`, line 95):

```cpp
ALWI void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE,
                         UnpackToDestEn>(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE,
                                          UnpackToDestEn>(dst_tile_index, in_cb_id)));
}
```

When compiled for TRISC0, only the `UNPACK()`-wrapped call survives; for TRISC1, only the `MATH()`-wrapped call survives; for TRISC2, the body is empty. See [`codebase_layout.md`](./codebase_layout.md) for how a call like `matmul_tiles` traverses all four layers.

### Why the Double Parentheses?

You will always see `MATH((expression))` with double parentheses. The outer parentheses are the macro invocation; the inner parentheses wrap the entire expression as a single macro argument. Without them, any commas in template arguments (e.g., `llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)`) would be interpreted as macro argument separators, causing a compilation error.

## 3. Tile-Based Computation Model

All computation in the Tensix pipeline operates on **tiles** -- fixed-size 2D blocks of data, typically 32x32 elements. Data flows through three stages, mediated by hardware registers and circular buffers.

### Data Flow Overview

```
 L1 Memory                  Hardware Registers              L1 Memory
┌──────────┐    TRISC0     ┌────────────────┐    TRISC2    ┌──────────┐
│  Input    │──(unpack)──▶ │   SrcA / SrcB  │              │  Output  │
│ Circular  │              │   Registers    │              │ Circular │
│ Buffers   │              └───────┬────────┘              │ Buffers  │
└──────────┘                       │                        └──────────┘
                            TRISC1 │ (math)                      ▲
                                   ▼                             │
                           ┌────────────────┐                    │
                           │ DST Register   │───(pack)───────────┘
                           │ (16 tiles of   │    TRISC2
                           │  32x32 each)   │
                           └────────────────┘
```

### Step-by-Step Tile Processing

**Step 1: Wait for input tiles (TRISC0 -- Unpack)**

The kernel waits for input tiles to be available in circular buffers:

```cpp
// From tt_metal/hw/inc/api/compute/cb_api.h, line 44
ALWI void cb_wait_front(uint32_t cbid, uint32_t ntiles) {
    UNPACK((llk_wait_tiles(cbid, ntiles)));
}
```

`cb_wait_front` is gated by `UNPACK()`, so only TRISC0 actually blocks waiting for tiles.

**Step 2: Acquire DST register (TRISC1 -- Math)**

Before the math engine can write results, it acquires exclusive access to the DST register:

```cpp
// From tt_metal/hw/inc/api/compute/reg_api.h, line 42
ALWI void tile_regs_acquire() {
    MATH((llk_math_wait_for_dest_available()));
}
```

**Step 3: Unpack and compute**

Operations like `copy_tile` or `matmul_tiles` combine unpack and math in a single call. Using `copy_tile` as an example (`tt_metal/hw/inc/api/compute/tile_move_copy.h`, line 95):

```cpp
ALWI void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE,
                         UnpackToDestEn>(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE,
                                          UnpackToDestEn>(dst_tile_index, in_cb_id)));
}
```

TRISC0 unpacks from the circular buffer into SrcA. TRISC1 copies from SrcA to DST. These execute concurrently, synchronized by hardware.

**Step 4: Commit DST (TRISC1 -- Math) and wait for math (TRISC2 -- Pack)**

After TRISC1 finishes writing to DST, it signals completion:

```cpp
// From tt_metal/hw/inc/api/compute/reg_api.h, line 83
ALWI void tile_regs_commit() {
    MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
}
```

TRISC2 waits for this signal:

```cpp
// From tt_metal/hw/inc/api/compute/reg_api.h, line 51
ALWI void tile_regs_wait() {
    PACK((llk_packer_wait_for_math_done()));
}
```

**Step 5: Pack tiles to output (TRISC2 -- Pack)**

The packer reads from DST and writes to an output circular buffer:

```cpp
// From tt_metal/hw/inc/api/compute/pack.h, line 60
template <bool out_of_order_output = false>
ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb, std::uint32_t output_tile_index = 0) {
    PACK((llk_pack<DST_ACCUM_MODE, out_of_order_output, false>(ifrom_dst, icb, output_tile_index)));
}
```

**Step 6: Release DST and manage buffers**

After packing, TRISC2 releases the DST register so TRISC1 can use it again:

```cpp
// From tt_metal/hw/inc/api/compute/reg_api.h, line 94
ALWI void tile_regs_release() {
    PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
}
```

Input tiles are freed and output tiles are published:

```cpp
cb_pop_front(cb_in0, num_tiles);    // TRISC0: free input buffer space
cb_push_back(cb_out0, num_tiles);   // TRISC2: make output tiles visible
```

### Complete Example: Binary Element-Wise Kernel

Here is a simplified view of how these concepts come together in an actual kernel. From `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp`, line 13:

```cpp
void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_2;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);   // HW init: all 3 TRISCs
    binary_tiles_init<false, ELTWISE_OP_TYPE>(cb_in0, cb_in1);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_in0, per_core_block_size);    // TRISC0: wait for inputs
        cb_wait_front(cb_in1, per_core_block_size);    // TRISC0: wait for inputs
        cb_reserve_back(cb_out0, per_core_block_size); // TRISC2: reserve output space

        tile_regs_acquire();                            // TRISC1: acquire DST

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            // TRISC0: unpack tiles, TRISC1: compute binary op
            binary_tile<...>(cb_in0, cb_in1, i, i, i);
        }

        tile_regs_commit();                             // TRISC1: signal done

        tile_regs_wait();                               // TRISC2: wait for math
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);                      // TRISC2: pack to output
        }
        tile_regs_release();                            // TRISC2: release DST

        cb_pop_front(cb_in0, per_core_block_size);      // TRISC0: free input
        cb_pop_front(cb_in1, per_core_block_size);      // TRISC0: free input
        cb_push_back(cb_out0, per_core_block_size);     // TRISC2: publish output
    }
}
```

Despite appearing as a single sequential function, after compilation:
- **TRISC0** only executes the `UNPACK()`-gated code: `cb_wait_front`, unpack portions of `binary_tile`, and `cb_pop_front`.
- **TRISC1** only executes the `MATH()`-gated code: `tile_regs_acquire`, math portions of `binary_tile`, and `tile_regs_commit`.
- **TRISC2** only executes the `PACK()`-gated code: `cb_reserve_back`, `tile_regs_wait`, `pack_tile`, `tile_regs_release`, and `cb_push_back`.

The three processors run these instruction streams concurrently, synchronizing through the DST register semaphore and circular buffer pointers.

### Hardware Initialization

Before any tile operations, the hardware must be configured for the specific data formats and operation types. This is done via `compute_kernel_hw_startup()` or operation-specific init functions like `mm_init()`. From `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`, line 41:

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

Each TRISC configures its respective hardware unit (unpacker, math engine, or packer) with the data formats derived from the circular buffer identifiers.

---

**Next:** [Chapter 2 -- Build System Integration](../ch2_build_system_integration/index.md)
