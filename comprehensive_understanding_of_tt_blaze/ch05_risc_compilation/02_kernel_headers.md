# 02 -- Kernel Headers and LLK Extensions

<- [Per-RISC Compilation Model](01_per_risc_model.md) | [Index](index.md) | [Build System and JIT](03_build_system_and_jit.md) ->

---

**Source files:** `blaze/kernels/ops.hpp`, `blaze/kernels/kernel_op_api.hpp`, `blaze/kernels/kernel_utils.hpp`, `blaze/kernels/ct_types.h`, `blaze/kernels/rt_str.hpp`, `blaze/kernels/sfpu/`, `blaze/kernels/kernel_includes/`

## The kernels/ Directory Layout

The `blaze/kernels/` directory provides the infrastructure layer between TT-Metal's hardware APIs and the per-op kernel headers. Every generated fused kernel includes these shared headers.

```
blaze/kernels/
    ops.hpp              # Aggregator -- includes every per-op kernel header
    kernel_op_api.hpp    # COMPILE_FOR_* constexpr bools + SelectByRISCV
    kernel_utils.hpp     # Grid utilities, sharded buffer setup, CB reconfig, sync
    ct_types.h           # Typed aliases: CB, Semaphore, PerCore, Flag
    rt_str.hpp           # Runtime arg stub (actual generation is JIT)
    sfpu/
        clamped_silu_sfpu.hpp   # Clamped SiLU/SwiGLU SFPU activation
        zero_pad_sfpu.hpp       # SFPU-based zero-padding
    kernel_includes/
        tt_metal/        # Patched TT-Metal compute APIs (custom_mm, rmsnorm, etc.)
        tt_llk/          # Patched LLK library headers (Blackhole + Wormhole)
```

---

## ct_types.h: Semantic Type Aliases

**Source file:** `blaze/kernels/ct_types.h`

This header defines type aliases that carry semantic meaning for both the C++ kernel and the Python CT arg parser:

```cpp
using CB = uint32_t;           // circular buffer ID (port name)
using Semaphore = uint32_t;    // global semaphore L1 address
using PerCore = uint32_t;      // per-core varying value
using Flag = bool;             // boolean flag
```

At the C++ level these are zero-cost aliases. The design rationale is entirely about tooling and documentation -- the same principle as newtype wrappers in languages like Haskell or Rust, except here it crosses the Python/C++ boundary through a parser rather than a compiler. When the Python-side CT arg parser (referenced in Ch2 S03) sees `static constexpr CB src = A::src`, it knows `src` is a circular buffer ID and can validate that the Python side passes a CB allocation result. When it sees `static constexpr PerCore sender_idx = A::sender_idx`, it knows this value varies per core and must be supplied through `PerCoreCompileTimeDescriptor` rather than as a uniform scalar.

---

## kernel_utils.hpp: Shared Utilities

**Source file:** `blaze/kernels/kernel_utils.hpp`

This is the largest shared header, providing dataflow and compute utilities guarded by RISC defines. It conditionally includes different APIs per RISC:

```cpp
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif
#if defined(COMPILE_FOR_TRISC)
#include "kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#endif
```

### Core Coordinate Globals and Grid Utilities

Firmware-set variables provide each RISC access to its core's logical grid coordinates:

```cpp
extern uint8_t my_logical_x_;
extern uint8_t my_logical_y_;
```

`linear_id_in_grid<RowMajor>` computes a core's linear index within a rectangular sub-grid. The `RowMajor` template parameter controls iteration order and is resolved at compile time:

```cpp
template <bool RowMajor>
uint32_t linear_id_in_grid(uint32_t grid_start_x, uint32_t grid_start_y,
                           uint32_t grid_end_x, uint32_t grid_end_y);
```

`SplitHalfCoreInfo` and `get_split_half_core_info<RowMajor>` divide a grid into two halves -- used when a single grid serves two different roles (e.g., first half processes gate weights, second half processes up weights).

### Sharded Buffer Setup (NCRISC/BRISC Only)

```cpp
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
FORCE_INLINE void setup_sharded_buffer(uint32_t cb_id, uint32_t num_tiles) {
    cb_reserve_back(cb_id, num_tiles);
    cb_push_back(cb_id, num_tiles);
}
#endif
```

This is the fundamental operation for making a tensor-backed sharded CB available to compute. The tensor data already resides in L1; this call sets up the CB's internal bookkeeping (write pointer, tile count) so TRISC can `cb_wait_front` on it. Nearly every op's `init()` on NCRISC calls this.

Other NCRISC/BRISC utilities:

- `semaphore_dec` -- Atomic decrement for global semaphore reset across iterations.
- `zero_pad_cb<dst_cb, src_cb, padded_tiles, src_bytes, padded_bytes>` -- NOC-DMA copy from source CB into destination CB. The padding tail is NOT zeroed here -- TRISC handles it via SFPU after squaring (see `sfpu/zero_pad_sfpu.hpp`).

### Cross-RISC Synchronization

**Source file:** `blaze/kernels/kernel_utils.hpp` (lines 98-132)

The `sync_riscs_enter`/`sync_riscs_exit` pair implements a two-phase barrier using atomic L1 semaphores. The design rationale is that CB reconfig (changing buffer base addresses and sizes between phases of a fused kernel) requires all RISC processors to have finished their prior work before any RISC begins modifying CB state.

**Phase 1 (enter):** BRISC, TRISC0/unpack (`UCK_CHLKC_UNPACK`), and TRISC2/pack (`UCK_CHLKC_PACK`) each atomically increment `sem[0]`. NCRISC spins until `sem[0]` reaches 3, then resets it to 0 and proceeds. TRISC sub-processors call `tensix_sync()` before signaling to ensure their pipeline is flushed.

**Phase 2 (exit):** NCRISC atomically adds 3 to `sem[1]`. BRISC, TRISC0, and TRISC2 each wait until `sem[1]` is nonzero, then atomically decrement it.

Why NCRISC is the coordinator: only NCRISC can reset the hardware stream registers (`get_cb_tiles_received_ptr`, `get_cb_tiles_acked_ptr`) that track CB read/write positions. This is a hardware constraint, not an arbitrary assignment. Note that TRISC1 (math) does not participate in the barrier.

```cpp
// Phase 1: BR/TR0/TR2 signal done, NC waits for all 3
FORCE_INLINE void sync_riscs_enter(volatile uint32_t tt_l1_ptr* sem_addr) {
#if defined(COMPILE_FOR_BRISC) || defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
#if defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
    tensix_sync();
#endif
    __atomic_fetch_add(&sem_addr[0], 1, __ATOMIC_RELAXED);
#elif defined(COMPILE_FOR_NCRISC)
    while (__atomic_load_n(&sem_addr[0], __ATOMIC_RELAXED) < 3) { }
    sem_addr[0] = 0;
#endif
}
```

### CB Read-Pointer Utilities (TRISC Only)

```cpp
#if defined(COMPILE_FOR_TRISC)
FORCE_INLINE void override_cb_rd_ptr(uint32_t cb_id, uint32_t byte_address);
FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id);
FORCE_INLINE uint32_t get_local_cb_page_size(uint32_t cb_id);
FORCE_INLINE void update_local_cb_rd_ptr(uint32_t cb_id, uint32_t val);
#endif
```

These manipulate the local CB interface directly, bypassing the normal CB producer-consumer protocol. Used when TRISC needs to override the read pointer for techniques like weight streaming where the same CB buffer is reinterpreted at different offsets.

### CB Reconfiguration

The `reconfig_cb_interfaces` function handles the most delicate operation in fused kernels: changing circular buffer configurations between phases. A fused kernel might run RMSNorm in phase 1 and Matmul in phase 2, each needing different CB layouts.

```cpp
template <bool do_read, bool do_write, bool do_write_tile_ptr, bool do_reset_stream_regs>
FORCE_INLINE void reconfig_cbs_for_mask(uint32_t tt_l1_ptr* cb_config,
                                         uint32_t mask, uint32_t start_cb);
```

The config tensor layout per core (260+ uint32 words):
- Words 0-255: 64 CB configs, 4 words each `[addr, size, num_pages, page_size]` (in bytes)
- Word 256: `cb_mask_low` (bits 0-31, which CBs 0-31 to reconfigure)
- Word 257: `cb_mask_high` (bits 32-63, which CBs 32-63 to reconfigure)
- Words 258-259: semaphore words for `sync_riscs_enter`/`sync_riscs_exit`

Each RISC reconfigures with different read/write permissions following firmware conventions:

| RISC         | do_read | do_write | do_write_tile_ptr | do_reset_stream_regs |
|--------------|---------|----------|-------------------|----------------------|
| NCRISC       | true    | true     | false             | true                 |
| BRISC        | true    | true     | false             | false                |
| TRISC0/unpack (`UCK_CHLKC_UNPACK`) | true | false | false | false |
| TRISC2/pack (`UCK_CHLKC_PACK`) | false | true | true | false |

The full `reconfig_cb_interfaces(cb_config)` function calls `sync_riscs_enter`, applies both masks, and calls `sync_riscs_exit`.

---

## Per-Op Kernel Headers: The Op Pattern

Every op in `blaze/ops/<name>/kernels/op.hpp` follows a consistent three-layer structure.

### Layer 1: CT Arg Interface Structs

Each op defines up to four CT arg struct templates. The naming convention maps to RISC roles:

| Struct | RISC Affinity | Alias in Op |
|--------|---------------|-------------|
| `CoreCTArgs<A>` | ALL (shared flags and CBs) | `core_cta` |
| `ReaderCTArgs<A>` | NCRISC | `dm0_cta` |
| `WriterCTArgs<A>` | BRISC | `dm1_cta` |
| `ComputeCTArgs<A>` | TRISC | `compute_cta` |

The template parameter `A` is the generated `ct_args::prefix` struct from the JIT system. Because `A` is a compile-time struct with `static constexpr` members, every field access like `A::in0` resolves to a compile-time constant with zero runtime cost.

### Layer 2: The Op Class Template

```cpp
template <typename Args>
class Op {
    using core_cta = CoreCTArgs<Args>;
    using dm0_cta = ReaderCTArgs<Args>;
    using compute_cta = ComputeCTArgs<Args>;
public:
    void init();
    void operator()();
    void teardown();
};
```

### Layer 3: RISC-Guarded Method Bodies

Each lifecycle method contains `#if defined(COMPILE_FOR_*)` blocks:

```cpp
void init() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_active) {
        if constexpr (dm0_cta::in0_is_tensor_backed) {
            unified_kernels::setup_sharded_buffer(core_cta::in0, dm0_cta::k_num_tiles);
        }
    }
#endif
}

void operator()() {
#if defined(COMPILE_FOR_TRISC)
    if constexpr (core_cta::is_active) {
        custom_mm_block_init_short<...>(...);
        tile_regs_acquire();
        custom_mm_block<...>(...);
        // ... pack results ...
    }
#endif
}
```

The two-level guard system (`#if` + `if constexpr`) is the key to dead-code elimination. `#if` removes code that would not even parse on the wrong RISC (e.g., `noc_async_write` does not exist on TRISC). `if constexpr` removes code that parses on the correct RISC but is not needed for the current core's role.

### Mcast: Multi-RISC Data Movement

The Mcast op demonstrates the most complex cross-RISC coordination:

- **NCRISC `init()`**: `setup_sharded_buffer` for the source CB on the sender.
- **BRISC `init()`**: Compute multicast NOC address, initialize persistent sender state.
- **BRISC `operator()()`**: `cb_wait_front` on source, `send_data_mcast`, signal semaphore.
- **NCRISC `operator()()`**: `cb_reserve_back` on dest, `noc_semaphore_wait`, `cb_push_back`.
- **BRISC `teardown()`**: `teardown_persistent_sender`.
- **TRISC**: No code for Mcast -- pure data movement op.

The `run_as<A>()` method is unique to Mcast and enables the persistent sender optimization: a single init/teardown pair can send multiple multicast payloads with different source/destination CBs, avoiding the overhead of re-initializing NOC state for each subsequent multicast through the same grid topology.

### Gather: The PerCore Pattern

The Gather op demonstrates how `PerCore` CT args eliminate runtime index computation:

```cpp
uint32_t core_index;
if constexpr (core_cta::use_per_core) {
    core_index = dm0_cta::sender_idx;  // compile-time constant
} else {
    core_index = unified_kernels::linear_id_in_grid<true>(...);  // runtime calc
}
uint32_t offset = core_index * dm0_cta::data_size_bytes;
```

With `use_per_core`, the compiler folds `core_index * data_size_bytes` into a single constant, eliminating the multiply at runtime. Gather also uses NCRISC for sending (NOC writes) and BRISC for receiving (semaphore waits) -- the opposite of Mcast, where BRISC sends and NCRISC receives.

---

## ops.hpp: The Aggregator Header

**Source file:** `blaze/kernels/ops.hpp`

A generated kernel needs exactly one include to access all op types:

```cpp
#include "blaze/kernels/ops.hpp"
```

This file aggregates all 54 registered op headers:

```cpp
#include "../ops/all_reduce/kernels/op.hpp"
#include "../ops/matmul/kernels/op.hpp"
#include "../ops/mcast/kernels/op.hpp"
// ... 54 op includes total
```

The compiler only instantiates templates that are actually used, so including all headers does not bloat the binary. An unused `blaze::RMSNorm::Op<...>` template is never instantiated and produces zero code.

### Op Inventory by Category

| Category | Ops |
|----------|-----|
| **Data movement** | `mcast`, `gather`, `scatter`, `scatter_raw`, `copy`, `shard2cb`, `tile_copy` |
| **Compute** | `matmul`, `matmul_cb`, `matmul_fused_act`, `kn_sliced_matmul`, `dram_streaming_matmul` |
| **Normalization** | `rmsnorm`, `padded_rmsnorm`, `residual_add` |
| **Activation** | `clamped_silu`, `eltwise_mul` |
| **Attention** | `sdpa`, `sdpa_reduce`, `flash_mla`, `sparse_flash_mla`, `rope`, `create_q_heads` |
| **MoE** | `deepseek_moe_gate`, `glm_moe_gate`, `glm_moe_gate_merge` |
| **Reduction** | `reduce_to_one`, `gated_reduce`, `gated_local_reduce`, `gather_reduce`, `argmax`, `distributed_topk` |
| **CB management** | `cb_flush`, `cb_reconfig`, `cb_scratch_reset` |
| **CCL** | `all_reduce`, `ccl_broadcast` |
| **Synchronization** | `barrier_sender`, `barrier_receiver`, `dm_risc_handshake`, `pipeline_stage_sync` |
| **DSA (attention)** | `dsa_cache_update`, `dsa_indexer`, `dsa_kv_prep`, `dsa_sparse_flash_decode`, `dsa_sparse_gather`, `dsa_topk` |
| **Format** | `retilize`, `untilize`, `tile_row_convert`, `q_idx_tilize` |
| **Other** | `embedding`, `migration` |

---

## sfpu/: Shared SFPU Activation Functions

**Source file:** `blaze/kernels/sfpu/`

The `sfpu/` subdirectory contains shared activation functions used by multiple ops:

### clamped_silu_sfpu.hpp

Defines SFPU activation functions for clamped SiLU (GPT-OSS SwiGLU), used by both the standalone `ClampedSilu` micro-op and the fused activation path inside `DRAMStreamingMatmul`.

Fused activation mode constants:

```cpp
static constexpr uint32_t FUSED_ACT_NONE = 0;
static constexpr uint32_t FUSED_ACT_SILU = 1;
static constexpr uint32_t FUSED_ACT_CLAMPED_GATE = 2;
static constexpr uint32_t FUSED_ACT_CLAMPED_UP = 3;
```

Functions (guarded by `#ifdef TRISC_PACK` -- pack-side SFPU only):

- `calculate_clamped_silu_gate<is_fp32_dest_acc_en, ITERATIONS>`: Computes `clamp(x, max=limit) * sigmoid(alpha * clamp(x, max=limit))`
- `calculate_clamped_up<is_fp32_dest_acc_en, ITERATIONS>`: Computes `clamp(x, -limit, limit) + 1.0`

### zero_pad_sfpu.hpp

Provides SFPU-based zero-padding for dest register tile rows, used after the NOC DMA copy in `zero_pad_cb` to handle the padding tail.

---

## kernel_includes/: LLK Extensions

**Source file:** `blaze/kernels/kernel_includes/`

This directory contains TT-Blaze's custom extensions to the Low-Level Kernel (LLK) library. They are organized to mirror the tt-metal source tree so the compiler's include path resolves them correctly. Op headers include them via explicit relative paths:

```cpp
// In blaze/ops/matmul/kernels/op.hpp
#include "../../../kernels/kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
```

### How LLK Extensions Work: Three Layers

The LLK extension mechanism follows a consistent layering:

**Layer 1 -- SFPU ckernel** (lowest level): Files like `ckernel_sfpu_add_rsqrt.h` implement the raw SFPU instruction sequence using `TTI_SFPLOAD`, `TTI_SFPMAD`, etc.

**Layer 2 -- LLK math wrapper**: Files like `llk_math_eltwise_unary_sfpu_add_rsqrt.h` wrap the ckernel into the LLK math dispatch framework.

**Layer 3 -- Compute kernel API** (user-facing): Files like `add_rsqrt.h` provide the API that op headers call:

```cpp
ALWI void add_rsqrt_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_add_rsqrt_init<APPROX>()));
}

template <bool fast_and_approx = false, int vec_mode = VectorMode::RC, int ITERATIONS = 8>
ALWI void add_rsqrt_tile(uint32_t idst, uint32_t addend) {
    MATH((llk_math_eltwise_unary_sfpu_add_rsqrt<
        APPROX, DST_ACCUM_MODE, fast_and_approx, ITERATIONS>(
        idst, addend, vec_mode)));
}
```

The `MATH((...))` macro ensures the code only compiles on TRISC1 (the math sub-RISC). The double-parentheses are required because the macro wraps a comma-separated template instantiation. Similarly, `UNPACK((...))` targets TRISC0 and `PACK((...))` targets TRISC2.

### Compute Kernel API Extensions

**Source:** `kernel_includes/tt_metal/include/compute_kernel_api/`

| File | Purpose |
|------|---------|
| `add_rsqrt.h` | Fused add-epsilon-then-rsqrt for RMSNorm |
| `custom_mm.h` | Custom matrix multiply with dense packing and split accumulation |
| `deepseek_compute_kernel_hw_startup.h` | Compute kernel hardware init (`deepseek_compute_kernel_init()`) |
| `deepseek_moe_gate.h` | MoE gate scoring (DeepSeek-specific) |
| `eltwise_mul_scalar.h` | Element-wise multiply by scalar |
| `glm_moe_gate.h` | GLM-style MoE gate |
| `rmsnorm.h` | Fused RMSNorm multiply-broadcast with dest register reuse |
| `sdpa.h` | Scaled dot-product attention math extensions |
| `sdpa_custom_mm.h` | Custom matrix multiply for SDPA |
| `sdpa_custom_mm_reuse_dest_srcb.h` | SDPA custom MM with dest/srcB reuse |
| `topk_xl.h` | Extended top-k for large vocabularies |

### LLK Library Extensions

**Source:** `kernel_includes/tt_llk/`

Custom LLK library files organized by architecture:

**Blackhole** (`tt_llk_blackhole/`):
- `llk_lib/`: Math and unpack routines -- `llk_math_deepseek_moe_gate_eltwise_binary.h`, `llk_math_rmsnorm_bcast_scalar_dest_reuse.h`, `llk_math_sdpa_bcast_col_srca_srcb_reuse.h`, `llk_unpack_A_rmsnorm.h`, `llk_unpack_A_sdpa.h`, `llk_unpack_A_topk_xl_copy.h`, etc.
- `common/inc/sfpu/`: SFPU ckernel implementations -- `ckernel_sfpu_deepseek_moe_gate_topk_single_face.h`, `ckernel_sfpu_sdpa_reduce_row.h`, `ckernel_sfpu_topk_xl.h`

**Wormhole B0** (`tt_llk_wormhole_b0/`):
- Fewer files -- only the DeepSeek MoE gate extensions (`llk_math_deepseek_moe_gate_eltwise_binary.h`, `ckernel_sfpu_deepseek_moe_gate_topk_single_face.h`), suggesting limited custom LLK needs on this architecture.

**Third-party LLK extensions** (`kernel_includes/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/`):
- `llk_math_custom_mm.h`, `llk_math_sdpa_custom_mm.h`, `llk_math_sdpa_custom_mm_reuse_dest_srcb.h` -- custom matmul variants at the LLK level.

---

## PhaseInfo: Connecting Python Codegen to C++ Ops

Each op registers a `PhaseInfo` object that tells the kernel codegen how to emit C++ for that op. `PhaseInfo` carries the C++ type (`cpp_type`), alias suffix, lifecycle flags (`has_init_teardown`, `init_is_empty`, `teardown_is_empty`), and an optional `setup_method` for mcast follower phases. See [Ch3 S05 -- Kernel Codegen](../ch03_compilation_pipeline/05_kernel_codegen.md) for the full `PhaseInfo` dataclass definition, registry, and the codegen algorithm that transforms these into generated kernel source.

The codegen uses `PhaseInfo` to generate type aliases (`using ActMcast = blaze::Mcast::Op<ct_args::act_mcast>`) and lifecycle calls (`var.init()`, `var()`, `var.teardown()`) in the generated `kernel_main()`. When `init_is_empty=True`, the `init()` call is omitted entirely.

---

## How ct_args:: Structs Bridge Python and C++

The `ct_args::act_mcast` struct that the Op template reads is generated by the JIT build from the named compile-time arg list. For the sender-group NCRISC descriptor with args like `[("act_mcast.src", 0), ("act_mcast.src_num_pages", 4), ...]`, the JIT generates:

```cpp
// Auto-generated by JIT
namespace ct_args {
struct act_mcast {
    static constexpr uint32_t src = 0;          // CB ID
    static constexpr uint32_t src_num_pages = 4;
    static constexpr bool src_is_tensor_backed = true;
    static constexpr uint32_t receiver_semaphore = 0x1000;
    static constexpr uint32_t dst = 3;
    static constexpr uint32_t dst_num_pages = 4;
    static constexpr bool is_sender = 1;     // per-core specialized
    static constexpr bool is_receiver = 0;
    // ...
};
}
```

When the Op template reads `A::src`, the compiler resolves it to `ct_args::act_mcast::src = 0`. This is a compile-time constant -- the value is baked into the instruction stream. No runtime lookup, no register read, no L1 access. Every named CT arg becomes a constant-folded literal in the compiled firmware.

This is why per-core CT arg splits produce separate binaries: changing `is_sender` from 0 to 1 changes which code paths survive `if constexpr` elimination. The sender binary has multicast send code; the receiver binary has semaphore wait code. They are fundamentally different programs compiled from the same source.

---

## The Full Include Chain

When the codegen produces a `.cpp` file for our mcast -> matmul -> gather example, the include chain is:

```
generated/kernels/fused_abc123.cpp
    |
    +-- blaze/kernels/ops.hpp
    |       +-- blaze/ops/mcast/kernels/op.hpp
    |       |       +-- blaze/kernels/ct_types.h
    |       |       +-- blaze/kernels/kernel_utils.hpp
    |       |       +-- api/dataflow/dataflow_api.h  (NCRISC/BRISC only)
    |       +-- blaze/ops/matmul/kernels/op.hpp
    |       |       +-- kernel_includes/.../custom_mm.h  (TRISC only)
    |       +-- blaze/ops/gather/kernels/op.hpp
    |       +-- ... (55 total, but only used ones instantiated)
    |
    +-- blaze/kernels/kernel_op_api.hpp
    |       +-- COMPILE_FOR_* constexpr bools
    |       +-- SelectByRISCV
    |
    +-- blaze/kernels/kernel_utils.hpp
            +-- grid utilities, sync primitives, CB reconfig
```

Each RISC compilation pass prunes a different subtree. The TRISC pass never sees `api/dataflow/dataflow_api.h`; the NCRISC pass never sees `api/compute/matmul.h`.

---

## rt_str.hpp: Runtime Arg Stub

**Source file:** `blaze/kernels/rt_str.hpp`

This file is intentionally minimal. The original string-based runtime arg lookup (`rt_arg<>`, `rt_common<>`) has been fully replaced by JIT-generated `constexpr` index constants. Kernels access runtime args via:

```cpp
uint32_t n = get_common_arg_val<uint32_t>(rt::my_op::num_tiles);
```

The generated `runtime_args_generated.h` is auto-included via `-include` during compilation. The `rt::` namespace (not `ct_args::`) holds the index constants. A typo in the arg name produces an immediate compile error.

---

<- [Per-RISC Compilation Model](01_per_risc_model.md) | [Index](index.md) | [Build System and JIT](03_build_system_and_jit.md) ->
