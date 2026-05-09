# 2.2 The Kernel Op API and Unified Headers

The Python-side `UnifiedKernelDescriptor` generates multiple `KernelDescriptor` objects from one source path, but the mechanism that makes a single `.cpp` file compile correctly for three different RISC processors lives in the C++ header library under `unified_kernels/`. This section examines the three layers of that library: the core detection and type dispatch in `kernel_op_api.hpp`, the canonical struct pattern that every micro-op header follows, the coordination and reconfiguration primitives in `kernel_utils.hpp`, and the complete catalog of 27 reusable micro-op headers organized into five structural categories.

**Source files:** `unified_kernels/kernel_op_api.hpp` (38 lines), `unified_kernels/kernel_utils.hpp` (197 lines), `unified_kernels/*.hpp` (25 op headers)

---

## 2.2.1 kernel_op_api.hpp: RISC Detection and Type Dispatch

This 38-line header is `#include`d by every unified kernel. It provides two mechanisms:

### Constexpr Bool Detection

Lines 9-25 define three `inline constexpr bool` variables based on which preprocessor macro the compiler defines:

```cpp
#if defined(COMPILE_FOR_NCRISC)
inline constexpr bool is_ncrisc = true;
inline constexpr bool is_brisc = false;
inline constexpr bool is_trisc = false;
#elif defined(COMPILE_FOR_BRISC)
inline constexpr bool is_ncrisc = false;
inline constexpr bool is_brisc = true;
inline constexpr bool is_trisc = false;
#elif defined(COMPILE_FOR_TRISC)
inline constexpr bool is_ncrisc = false;
inline constexpr bool is_brisc = false;
inline constexpr bool is_trisc = true;
#endif
```

Because these are `inline constexpr`, they are compile-time constants. Any `if constexpr (is_ncrisc)` branch is resolved at compile time; the dead branches are eliminated entirely from the binary. This is the mechanism that allows a single `.cpp` file to produce three completely different executables.

Each guard also pulls in the appropriate profiler header:
- NCRISC and BRISC include `tt_metal/tools/profiler/kernel_profiler.hpp`
- TRISC includes `tools/profiler/kernel_profiler.hpp` (different path) plus `api/compute/blank.h` for compute baseline initialization

### The SelectByRISCV Type Alias

Lines 32-37 define the primary type dispatch mechanism:

```cpp
namespace unified_kernels {

template <typename Reader, typename Writer, typename Compute>
using SelectByRISCV =
    std::conditional_t<is_ncrisc, Reader,
        std::conditional_t<is_brisc, Writer, Compute>>;

}  // namespace unified_kernels
```

This nested `std::conditional_t` resolves at compile time to one of three types:

| Compilation target | `SelectByRISCV<R, W, C>` resolves to |
|---|---|
| `COMPILE_FOR_NCRISC` | `R` (Reader type) |
| `COMPILE_FOR_BRISC` | `W` (Writer type) |
| `COMPILE_FOR_TRISC` | `C` (Compute type) |

Every micro-op uses it to define a unified `RTArgs` type alias:

```cpp
using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;
```

When compiled for NCRISC, `RTArgs` resolves to `ReaderArgs`. The `kernel_main()` function constructs a single `RTArgs` value, and the compiler ensures only the matching struct layout is instantiated.

**Important naming convention:** The `SelectByRISCV` template parameters are ordered `<Reader, Writer, Compute>`, mapping to `<NCRISC, BRISC, TRISC>`. This matches the hardware convention where NCRISC is the reader and BRISC is the writer, though some ops invert the semantic roles. For example:

```cpp
// gather.hpp -- NCRISC=Sender, BRISC=Receiver:
using RTArgs = unified_kernels::SelectByRISCV<SenderArgs, ReceiverArgs, ComputeArgs>;

// mcast.hpp -- BRISC=Sender, NCRISC=Receiver:
using RTArgs = unified_kernels::SelectByRISCV<ReceiverArgs, SenderArgs, ComputeArgs>;
```

The swap reflects hardware optimization: Gather senders use NOC_0 (NCRISC's dedicated NOC) for unicast writes, while Mcast senders use NOC_1 (BRISC's dedicated NOC) for multicast writes.

---

## 2.2.2 The Canonical Unified Kernel Struct Pattern

All 25 unified kernel op headers in `unified_kernels/` follow a remarkably consistent structure. The pattern decomposes into four components:

```
struct OpName {
    // --- Component 1: Compile-time args (one per RISC) ---
    struct ReaderCTArgs { ... };        // NCRISC
    struct WriterCTArgs { ... };        // BRISC
    template <params...>
    struct ComputeCTArgs {              // TRISC
        static constexpr ... ;
    };

    // --- Component 2: Runtime args (one per RISC) ---
    struct ReaderArgs { field; field; ... };
    struct WriterArgs { field; field; ... };
    struct ComputeArgs { field; field; ... };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // --- Component 3: Op class ---
    template <typename CTArgs, bool IsActiveCore, ...extra template params...>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) { impl(args); }
        }
    private:
        void impl(const RTArgs& args) {
            #if defined(COMPILE_FOR_NCRISC)
                // reader logic
            #elif defined(COMPILE_FOR_BRISC)
                // writer logic
            #elif defined(COMPILE_FOR_TRISC)
                // compute logic
            #endif
        }
    };
};
```

### Key Design Elements

**1. CTArgs as template parameters with `static constexpr` members.** Compile-time arguments are encoded as non-type template parameters on nested structs. For example, `Matmul::ComputeCTArgs`:

```cpp
template <uint32_t out_w_, bool transpose_ = false, uint32_t fused_activation_ = 0>
struct ComputeCTArgs {
    static constexpr uint32_t out_w = out_w_;
    static constexpr bool transpose = transpose_;
    static constexpr FusedActivation fused_activation =
        static_cast<FusedActivation>(fused_activation_);
    static constexpr bool fuse_sigmoid = fused_activation == FusedActivation::SIGMOID;
    static constexpr bool fuse_silu = fused_activation == FusedActivation::SILU;
};
```

Because `out_w` is `constexpr`, loop bounds like `for (uint32_t w = 0; w < out_w; w++)` are known at compile time, enabling full unrolling.

**2. The `IsActiveCore` guard.** Every `Op::operator()` wraps its logic in `if constexpr (IsActiveCore)`. The `IsActiveCore` boolean comes from a named compile-time arg:

```cpp
struct Core {
    static constexpr bool is_active_core =
        get_named_compile_time_arg_val("is_active_core") == 1;
};
```

This allows the same kernel binary to be placed on cores that do not participate in a given operation. Inactive cores execute the `operator()` call but it compiles to a no-op.

**3. Dual dispatch: `#if defined` + `if constexpr`.** The `#if defined(COMPILE_FOR_*)` guards select which *APIs* are available (dataflow vs. compute). The `if constexpr` checks select which *code paths* within those APIs are emitted. Both layers contribute to dead-code elimination.

**4. Empty structs for inactive RISCs.** When a RISC has no work (e.g., `Matmul::ReaderCTArgs`, `Matmul::ReaderArgs`), the struct is defined but empty. This ensures the type system is always satisfied -- `SelectByRISCV` always resolves to *something* -- without generating any runtime overhead.

### How Python Compile-Time Args Become C++ constexpr

The chain from Python integer to hardware-optimized binary is:

**Python side:**
```python
trisc_named_compile_time_args = [("matmul_out_w", 4), ("matmul_transpose", 0)]
```

**C++ kernel side:**
```cpp
constexpr uint32_t out_w = get_named_compile_time_arg_val("matmul_out_w");  // == 4
using MatmulCT = deepseek_b1_ops::Matmul::ComputeCTArgs<out_w, false, 0>;
deepseek_b1_ops::Matmul::Op<MatmulCT, true, true, true> matmul_op;
matmul_op(args);
```

Because `out_w` is `constexpr`, it can be used as a non-type template parameter. The `ComputeCTArgs<4, false, 0>` instantiation produces a unique type with `static constexpr uint32_t out_w = 4`, and all loops bounded by `out_w` are fully unrolled. This chain -- Python `int` to named arg to `constexpr` to template parameter to `static constexpr` member -- is the foundation of the entire unified kernel performance story.

### How a Kernel .cpp File Uses the Pattern

A complete standalone micro-op kernel (`matmul_kernel.cpp`) demonstrates composition:

```cpp
#include "../../../unified_kernels/kernel_op_api.hpp"
#include "../../../unified_kernels/kernel_utils.hpp"
#include "../../../unified_kernels/matmul.hpp"

struct Core {
    static constexpr bool is_active_core =
        get_named_compile_time_arg_val("is_active_core") == 1;
};

void kernel_main() {
    #if defined(COMPILE_FOR_NCRISC)
        using MatmulCTArgs = deepseek_b1_ops::Matmul::ReaderCTArgs;
        // ... setup sharded buffers ...
        deepseek_b1_ops::Matmul::ReaderArgs matmul_args{};

    #elif defined(COMPILE_FOR_BRISC)
        using MatmulCTArgs = deepseek_b1_ops::Matmul::WriterCTArgs;
        deepseek_b1_ops::Matmul::WriterArgs matmul_args{};

    #elif defined(COMPILE_FOR_TRISC)
        using MatmulCTArgs = deepseek_b1_ops::Matmul::ComputeCTArgs<...>;
        deepseek_b1_ops::Matmul::ComputeArgs matmul_args{
            .in0 = in0_cb, .in1 = in1_cb, .out = out_cb,
            .k_num_tiles = num_tiles_k,
        };
        deepseek_compute_kernel_init();
    #endif

    // Unified dispatch -- same line for all RISCs
    deepseek_b1_ops::Matmul::Op<MatmulCTArgs, Core::is_active_core, true, true> matmul;
    matmul(matmul_args);
}
```

The `#if defined(COMPILE_FOR_...)` sections handle RISC-specific setup, while the final two lines are identical across all three compilations.

---

## 2.2.3 Taxonomy of Unified Kernel Headers

The `unified_kernels/` directory contains 27 `.hpp` files: 25 operation headers and 2 infrastructure headers (`kernel_op_api.hpp`, `kernel_utils.hpp`). The 25 operation headers fall into five structural categories:

### Category 1: Compute-Only Ops

These ops do all meaningful work on TRISC; NCRISC and BRISC are no-ops (or at most signal sharded buffer readiness).

| Header | Operation | TRISC Work |
|--------|-----------|-----------|
| `matmul.hpp` | $\mathbf{C} = \mathbf{A} \times \mathbf{B}$ with optional fused sigmoid/SiLU | `custom_mm_block` |
| `rmsnorm.hpp` | $\text{RMS}(\mathbf{x}) = \mathbf{x} / \sqrt{\text{mean}(\mathbf{x}^2) + \epsilon} \cdot \gamma$ | mul\_reduce\_scalar + rsqrt + bcast |
| `eltwise_add.hpp` | $\mathbf{y} = \mathbf{a} + \mathbf{b}$ with CB pointer indexing | `add_tiles` |
| `eltwise_mul.hpp` | Element-wise multiply | `mul_tiles` |
| `residual_add.hpp` | Matmul output + sharded residual | `add_tiles` with offset indexing |
| `local_reduce.hpp` | $\sum_i \mathrm{tile}_i$ with optional SiLU | `add_tiles` (acc\_to\_dest) |
| `gated_reduce.hpp` | $\text{SiLU}(\sum g_1) \cdot \sum g_2$ | add + silu + mul |
| `argmax.hpp` | Argmax over tiles | comparison |
| `create_q_heads.hpp` | Reshape Q heads from latent space | `tile_move_copy` |

**Structural pattern:** Empty `ReaderCTArgs`/`WriterCTArgs`/`ReaderArgs`/`WriterArgs`; all logic inside `#if defined(COMPILE_FOR_TRISC)`.

### Category 2: Reader-Driven Ops (NCRISC Active)

These ops require NCRISC to fetch data from DRAM or remote L1 before TRISC can compute.

| Header | Operation | NCRISC Work | TRISC Work |
|--------|-----------|-------------|------------|
| `rope.hpp` | Rotary Position Embedding | DRAM interleaved reads of cos/sin | matmul + bcast\_mul + add |
| `dram_streaming_matmul.hpp` | Matmul with DRAM-streamed B | Triple-buffered DRAM pipelining | `custom_mm_block` + optional SiLU |
| `kv_cache_update.hpp` | KV cache write-back | Sharded buffer reads | tile copy |

**Structural pattern:** Non-empty `ReaderCTArgs` with DRAM addressing parameters (`bank_id`, `vc`, `page_size`); `WriterCTArgs` typically empty.

### Category 3: Sender/Receiver Pairs (Dataflow Ops)

These ops move data between cores via NOC writes, with TRISC either idle or performing a trivial operation.

| Header | Operation | NCRISC Role | BRISC Role |
|--------|-----------|-------------|-----------|
| `gather.hpp` | N:1 gather | Sender (`noc_async_write`) | Receiver (semaphore wait + `cb_push`) |
| `mcast.hpp` | 1:N multicast | Receiver (semaphore wait) | Sender (persistent mcast) |
| `gather_reduce.hpp` | Gather + reduction | Sender | Receiver + TRISC reduction |
| `moe_gather.hpp` | MoE-specific gather variant | Receiver | Sender |

**Structural pattern:** `RTArgs` uses semantic names (`SenderArgs`/`ReceiverArgs` instead of `ReaderArgs`/`WriterArgs`). The `Op` template includes `IsSenderCore`/`IsReceiverCore` booleans. The `SelectByRISCV` mapping swaps the conventional order:

```cpp
// gather.hpp: NCRISC=Sender, BRISC=Receiver
using RTArgs = unified_kernels::SelectByRISCV<SenderArgs, ReceiverArgs, ComputeArgs>;

// mcast.hpp and moe_gather.hpp: NCRISC=Receiver, BRISC=Sender (reversed!)
using RTArgs = unified_kernels::SelectByRISCV<ReceiverArgs, SenderArgs, ComputeArgs>;
```

### Category 4: Multi-Phase Fused Ops

These are the most complex kernels, with non-trivial logic on *all three* RISCs and lifecycle methods beyond a single `operator()`.

| Header | Operation | NCRISC | BRISC | TRISC |
|--------|-----------|--------|-------|-------|
| `flash_mla.hpp` | Flash attention decode | Pipelined DRAM reads + mcast receiver | Q distribution + K multicast + tree reduction | SDPA compute + tree reduce |
| `broadcast.hpp` | CCL broadcast | Fabric write (sender/secondary/receiver) | Socket/CB setup | no-op |
| `all_reduce_sender.hpp` | CCL all-reduce (send) | Local tensor read into CB | Fabric packet send | no-op |
| `all_reduce_receiver.hpp` | CCL all-reduce (recv) | Wait for fabric data + push | no-op | Reduction add |
| `sdpa_reduce_worker.hpp` | SDPA partial reduce | Semaphore coordination | Data routing | Attention reduce |
| `sdpa_reduce_forwarder.hpp` | SDPA output forwarding | Data forward | Semaphore management | no-op |
| `reduce_to_one_b1.hpp` | Cross-core reduction | Sender/receiver | Coordination | Add reduction |
| `deepseek_moe_gate.hpp` | MoE gating logic | Gate data movement | Expert selection | Softmax + top-K |
| `kn_sliced_matmul.hpp` | K/N-sliced matmul | K-dim streaming | N-dim coordination | Blocked matmul |

**Structural pattern:** Multiple template parameters on `Op` (e.g., `IsKVCacheUpdateCore`, `IsMcastGridCore`); lifecycle methods (`init()`, `operator()`, `teardown()` in `Mcast`); dynamic NOC mode assertion in `FlashMLADecode`.

### Category 5: Paired Kernel Sets

These are operations split across two separate headers that together form one logical operation across different cores.

| Pair | Operation |
|------|-----------|
| `all_reduce_sender.hpp` + `all_reduce_receiver.hpp` | Cross-device all-reduce |
| `sdpa_reduce_worker.hpp` + `sdpa_reduce_forwarder.hpp` | Multi-core SDPA reduction |

Each header in the pair is compiled for a different `CoreRangeSet`, but both are included in the same `ProgramDescriptor`.

---

## 2.2.4 The Mcast Lifecycle Pattern

The `Mcast` struct in `mcast.hpp` demonstrates a lifecycle pattern unique among the unified kernels. Unlike most ops whose `Op::operator()` is a fire-and-forget call, `Mcast::Op` exposes three methods:

```cpp
template <typename CTArgsT, bool IsSenderCore, bool IsMcastGridCore,
          bool IsReceiverCore, bool pop_src>
class Op {
public:
    template <bool init_noc = true>
    void init(const RTArgs& args);      // Phase 1: configure NOC multicast state

    void operator()(const RTArgs& args); // Phase 2: send/receive data

    void teardown();                     // Phase 3: flush + hardware bug workaround
};
```

This three-phase design exists because multicast NOC state is expensive to set up: `init()` configures the `write_cmd_buf` and `write_reg_cmd_buf` registers, sets the multicast destination rectangle, and initializes the sender semaphore. If a fused kernel multicasts multiple buffers in sequence, it calls `init()` once, `operator()` $N$ times, then `teardown()` once -- amortizing the setup cost.

The `teardown()` includes a notable hardware workaround:

```cpp
riscv_wait(1000);  // guarantee safety due to posted mcast hw bug
```

---

## 2.2.5 Structural Variations Across Headers

While the canonical pattern is consistent, several headers introduce variations:

**No runtime args** (`operator()()` with no parameters): `EltwiseAdd` and `DRAMStreamingMatmul` capture all parameters at compile time or read them directly from hardware state (e.g., CB read pointers). The `RTArgs` type still exists but is not passed.

**Dual CTArgs templates on `Op`:** `AllReduceSender` and `AllReduceReceiver` template `Op` on *two* CTArgs types (one per active RISC) rather than a single unified CTArgs:

```cpp
template <typename ReaderCT, typename WriterCT>
class Op { ... };
```

**Overloaded `operator()` with fabric arg index:** `AllReduceSender` and `AllReduceReceiver` provide a second overload that passes a mutable reference to a fabric argument index, allowing chained fabric operations.

**Flash MLA three-active-RISC architecture:** `flash_mla.hpp` is the most complex single op, with all three RISCs doing substantial work: NCRISC reads K/V cache pages from DRAM and receives multicast data, BRISC distributes Q data and multicasts K data with tree reduction, and TRISC performs full SDPA compute. It requires `DM_DYNAMIC_NOC` mode and uses `get_named_compile_time_arg_val()` directly in TRISC rather than a CTArgs struct.

---

## 2.2.6 kernel_utils.hpp: Coordination Primitives

The file `unified_kernels/kernel_utils.hpp` (197 lines) provides five families of utility functions.

### Grid Coordinate Utilities

**`linear_id_in_grid<RowMajor>`** (lines 28-39): Computes a core's linear index within a rectangular grid using the firmware-provided `my_logical_x_` and `my_logical_y_` globals:

$$
\text{RowMajor: } \mathrm{idx} = (\mathrm{rel{-}y}) \times W + \mathrm{rel{-}x}
$$
$$
\text{ColMajor: } \mathrm{idx} = (\mathrm{rel{-}x}) \times H + \mathrm{rel{-}y}
$$

**`get_split_half_core_info<RowMajor>`** (lines 46-53): Splits a grid into two halves and returns which half the current core belongs to plus its local index within that half. Used by `KNSlicedMatmul` to assign gate vs. up projection roles.

### Sharded Buffer Setup

**`setup_sharded_buffer`** (lines 64-67): A one-liner that marks a sharded L1 buffer as ready for consumption:

```cpp
FORCE_INLINE void setup_sharded_buffer(uint32_t cb_id, uint32_t num_tiles) {
    cb_reserve_back(cb_id, num_tiles);
    cb_push_back(cb_id, num_tiles);
}
```

By reserving and immediately pushing, the firmware's FIFO pointers are set so that TRISC can `cb_wait_front()` on the buffer. Only callable from NCRISC or BRISC.

### Atomic Semaphore Decrement

**`semaphore_dec`** (lines 70-72): Atomic decrement for global semaphore reset across iterations, used by the Broadcast micro-op.

### Cross-RISC Synchronization Barriers

Lines 80-111 implement a two-phase barrier protocol using atomic L1 semaphores. This is critical for the CB reconfig path in fused ops.

```
Phase 1 (enter):
  BR/TR0/TR2: atomic_fetch_add(&sem[0], 1)   // signal done
  NC:         spin while sem[0] < 3           // wait for all 3
              sem[0] = 0                       // reset

Phase 2 (exit):
  NC:         atomic_fetch_add(&sem[1], 3)    // signal done
  BR/TR0/TR2: spin while sem[1] == 0          // wait
              atomic_fetch_sub(&sem[1], 1)     // consume
```

The protocol is safe for repeated use because the full MoE body executes between `exit` and the next `enter`, guaranteeing `sem[0]` is always 0 when the next iteration begins. The `sem[0]` reset by NCRISC is non-atomic but safe because the other RISCs are blocked on `sem[1]` at that point.

Note the TRISC-specific defines: `UCK_CHLKC_UNPACK` (TRISC0/unpack thread) and `UCK_CHLKC_PACK` (TRISC2/pack thread) participate in the barrier. TRISC1 (math) does not access CB interfaces and is excluded.

### CB Reconfig Utilities

**`reconfig_cbs_for_mask`** (lines 131-162): The inner loop that reconfigures CB interfaces from a config tensor. See Section 2.3 for the full treatment.

**`reconfig_cb_interfaces`** (lines 164-194): The top-level reconfig function that determines per-RISC read/write/reset flags, executes the cross-RISC barrier, and calls `reconfig_cbs_for_mask` twice (CBs 0-31, CBs 32-63). Detailed in Section 2.3.5.

---

## 2.2.7 The DRAMStreamingMatmul: A Deep Example

To illustrate the full depth of the unified kernel pattern, consider `DRAMStreamingMatmul` (378 lines in `dram_streaming_matmul.hpp`). This op streams weight tiles from DRAM through a triple-buffered pipeline while TRISC computes partial matmul results.

The `ReaderCTArgs` template captures 16 compile-time parameters including the DRAM bank ID, virtual channel, page dimensions, and expert indexing configuration. Two of these -- `bank_id` and `vc` -- are per-core values provided through `PerCoreCompileTimeDescriptor` on the Python side, producing a separate kernel binary for each DRAM bank.

The NCRISC implementation demonstrates advanced NOC programming: it uses transaction IDs (`noc_async_read_set_trid`) for out-of-order completion tracking, triple-buffers the L1 write region, and wraps addresses at CB boundaries for multi-iteration reuse. The TRISC implementation performs blocked matmul with optional fused SiLU activation, operating on subblocks to hide DRAM latency.

The BRISC implementation is empty -- `WriterCTArgs` is an empty struct. For `DRAMStreamingMatmul`, all data movement is handled by NCRISC, and BRISC is free to handle other data movement tasks if the op is part of a fused kernel.

---

## 2.2.8 Fused Kernels: Composing Multiple Ops

The real power of the unified pattern is composition. The fused MoE kernel (`fused_ops/moe/moe_kernel.cpp`) includes 12 micro-op headers (plus 2 infrastructure headers) and composes them into a multi-step pipeline:

```cpp
#include "../../unified_kernels/kernel_op_api.hpp"
#include "../../unified_kernels/mcast.hpp"
#include "../../unified_kernels/matmul.hpp"
#include "../../unified_kernels/rmsnorm.hpp"
#include "../../unified_kernels/dram_streaming_matmul.hpp"
#include "../../unified_kernels/eltwise_mul.hpp"
#include "../../unified_kernels/eltwise_add.hpp"
// ... 6 more headers ...

struct Core {
    static constexpr bool is_sender_core = ...;
    struct Routed {
        static constexpr bool is_gate_mm_core = ...;
        static constexpr bool is_gate_proj_core = ...;
    };
    struct Shared {
        static constexpr bool is_compute_core = ...;
    };
};

void kernel_main() {
    // CB reconfig at startup (if needed)
    #if defined(RECONFIG_MOE_CBS)
        unified_kernels::reconfig_cb_interfaces(cb_config);
    #endif

    // Step 0: Residual Mcast
    // Step 1: RMSNorm + Mcast
    // Step 2: Gate Matmul+Sigmoid
    // Step 3: Gate Gather
    // ... steps 4-12 ...
}
```

Each step instantiates the relevant `Op` template with the core's role flags as `IsActiveCore`. For example, step 2 (gate matmul) only runs on cores where `Core::Routed::is_gate_mm_core` is true. The compiler eliminates all other code paths for those cores, producing a lean binary despite the source file including 12 different operation implementations.

### Dead-Code Elimination in Practice

Consider what happens when the fused MoE kernel is compiled for NCRISC on a sender core (`is_sender_core=1`):

- All `#if defined(COMPILE_FOR_TRISC)` blocks are removed by the preprocessor.
- All `#if defined(COMPILE_FOR_BRISC)` blocks are removed.
- Of the NCRISC code, only the `Mcast::Op` receiver logic and any `setup_sharded_buffer` calls for the sender role survive.
- All `if constexpr (Core::Routed::is_gate_mm_core)` blocks (which are `false` for the sender) are eliminated.

The result is a minimal binary that contains only the sender's NCRISC work, despite the source file containing the full 12-op pipeline.

---

## 2.2.9 Summary: The Three Layers of Abstraction

```
+---------------------------------------------------------------+
|  Layer 3: kernel_main() in .cpp                               |
|    Reads named compile-time args, constructs CTArgs/RTArgs,   |
|    instantiates Op<CTArgs, ...>, calls operator()             |
+---------------------------------------------------------------+
|  Layer 2: unified_kernels/*.hpp (Op structs)                  |
|    Defines CTArgs, RTArgs, Op::operator() with #if guards     |
|    and constexpr logic. Reusable across multiple .cpp files.  |
+---------------------------------------------------------------+
|  Layer 1: kernel_op_api.hpp + kernel_utils.hpp                |
|    RISC detection (is_ncrisc/is_brisc/is_trisc),              |
|    SelectByRISCV, grid math, sync primitives, CB reconfig     |
+---------------------------------------------------------------+
```

This layering means that adding a new unified kernel requires only:
1. Writing a new `*.hpp` struct following the canonical pattern (Layer 2).
2. Writing a thin `kernel_main()` that wires up args and calls the op (Layer 3).
3. Writing a Python class that creates the `UnifiedKernelDescriptor` (see Section 2.1).

The foundational infrastructure (Layer 1) is stable and shared across all 25 operations.

---

**Next:** [`03_circular_buffer_management.md`](./03_circular_buffer_management.md)
