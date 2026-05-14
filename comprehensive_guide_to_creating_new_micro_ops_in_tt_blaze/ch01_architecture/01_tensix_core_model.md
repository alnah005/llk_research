# 1.1 The Tensix Core Model

## What you will learn

- The physical structure of a Tensix core: five RISC-V processors grouped into three logical roles.
- How DM0 (NCRISC), DM1 (BRISC), and TRISC divide the work of data movement and compute, including the per-processor CT arg struct interfaces.
- L1 SRAM as the shared workspace, circular buffers as the intra-core synchronization primitive, and semaphores for inter-core coordination.
- Why all three processors run `kernel_main()` in parallel and what synchronization between them looks like in practice.
- How `COMPILE_FOR_*` preprocessor guards enable per-processor code selection in a single `.hpp` file.
- Why "empty processors enable overlap" -- the performance principle that makes fused kernels fast.

---

## 1.1.1 Five RISC-V cores, three logical processors

Every Tensix core on a Tenstorrent Blackhole chip contains **five RISC-V processor cores**. These five cores are organized into **three logical processors**, each with a distinct role in the data-movement and compute pipeline:

| Logical processor | RISC-V core(s) | TT-Blaze enum | Primary role | Preprocessor guard |
|---|---|---|---|---|
| **DM0 (NCRISC)** | 1 core | `Risc.NCRISC` | NOC reads: fetching data from DRAM or remote L1 into local L1 | `COMPILE_FOR_NCRISC` |
| **DM1 (BRISC)** | 1 core | `Risc.BRISC` | NOC writes: sending data from local L1 to remote cores (multicast, scatter) | `COMPILE_FOR_BRISC` |
| **TRISC** | 3 sub-cores (TRISC0, TRISC1, TRISC2) | `Risc.TRISC` | Compute pipeline: unpack, math, and pack | `COMPILE_FOR_TRISC` |

The three TRISC sub-cores form a hardware-pipelined compute architecture:

- **TRISC0 (Unpack)**: Reads tiles from circular buffers in L1, decompresses them (e.g., from bfloat8_b to float), and places them into source registers for the math engine.
- **TRISC1 (Math)**: Executes the actual arithmetic -- matrix multiplies, elementwise operations, reductions, and SFPU (Special Function Processing Unit) operations.
- **TRISC2 (Pack)**: Takes computed results from destination registers, recompresses them to the target data format, and writes them back to circular buffers in L1.

This hardware pipeline is why the TT-Metal compute APIs use macros like `UNPACK((...))`, `MATH((...))`, and `PACK((...))` -- each macro ensures the enclosed code runs only on the correct TRISC sub-core. For example, in `blaze/ops/rmsnorm/kernels/op.hpp`:

```cpp
// Square: element-wise multiply input * input
for (uint32_t i = 0; i < num_tiles; i++) {
    UNPACK((llk_unpack_AB(compute_cta::input, compute_cta::input, i, i)));
    MATH((llk_math_eltwise_mul_reduce_scalar<DST_ACCUM_MODE, MATH_FIDELITY>(
        i, compute_cta::input)));
}
```

The `UNPACK` call feeds tiles from L1 into source registers; the `MATH` call performs the multiply-reduce on those tiles. These execute on TRISC0 and TRISC1 respectively, overlapping in a pipeline. TRISC never touches the NOC directly -- all of its data arrives and departs through circular buffers in L1.

The `Risc` enum in `blaze/blaze_op.py` captures the processor grouping:

```python
# blaze/blaze_op.py
class Risc(Flag):
    """RISC processors that read compile-time arguments."""
    NCRISC = auto()
    BRISC = auto()
    TRISC = auto()

ALL_RISCS: tuple[Risc, ...] = (Risc.NCRISC, Risc.TRISC, Risc.BRISC)
RISC_NAMES = {Risc.NCRISC: "ncrisc", Risc.TRISC: "trisc", Risc.BRISC: "brisc"}
```

Because `Risc` is a `Flag` enum, a single compile-time argument can target multiple processors with a bitwise-OR expression such as `Risc.NCRISC | Risc.TRISC`.

> **Naming note:** In TT-Metal's kernel infrastructure, NCRISC corresponds to "Reader" kernels and BRISC corresponds to "Writer" kernels. TT-Blaze's CT arg struct names reflect this heritage: `ReaderCTArgs` maps to NCRISC, `WriterCTArgs` maps to BRISC, and `ComputeCTArgs` maps to TRISC. You will see this in every op's `op.hpp` kernel header.

## 1.1.2 Division of labor: who does what

Understanding which processor owns which responsibility is the single most important mental model for writing TT-Blaze ops.

### DM0 / NCRISC: the data fetcher

DM0 handles **NOC read operations**: pulling data from DRAM banks or from other cores' L1 into the local core's L1 SRAM. In Blaze kernels, DM0 typically:

- Sets up sharded tensor buffers with `cb_reserve_back()` / `cb_push_back()` to make tensor-backed data available to TRISC.
- Waits for inter-core semaphores (e.g., multicast receiver waiting for data from the sender).
- Issues `noc_async_read` calls to pull remote data.

In a kernel header's CT arg interface, DM0's arguments live in the `ReaderCTArgs` struct. From `blaze/ops/mcast/kernels/op.hpp`:

```cpp
template <typename A>
struct ReaderCTArgs {
    static constexpr Semaphore receiver_semaphore = A::receiver_semaphore;
    static constexpr CB dst = A::dst;
    static constexpr uint32_t dst_num_pages = A::dst_num_pages;
    static constexpr CB src = A::src;
    static constexpr uint32_t src_num_pages = A::src_num_pages;
    static constexpr bool src_is_tensor_backed = A::src_is_tensor_backed;
};
```

The NCRISC side of the Mcast op handles the **receiver** logic:

```cpp
// blaze/ops/mcast/kernels/op.hpp -- receiver logic (DM0/NCRISC)
#elif defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_receiver) {
        volatile tt_l1_ptr uint32_t* recv_sem_ptr =
            (volatile tt_l1_ptr uint32_t*)dm0_cta::receiver_semaphore;
        cb_reserve_back(dm0_cta::dst, dm0_cta::dst_num_pages);
        noc_semaphore_wait(recv_sem_ptr, VALID);
        noc_semaphore_set(recv_sem_ptr, INVALID);
        cb_push_back(dm0_cta::dst, dm0_cta::dst_num_pages);
    }
```

### DM1 / BRISC: the data sender

DM1 handles **NOC write operations**: sending data from local L1 to other cores or to DRAM. This includes the critical **multicast** operation, where a single sender pushes data to all receiver cores simultaneously. In Blaze kernels, DM1 typically:

- Initializes persistent multicast NOC command buffers (in `init()`).
- Waits for compute output via `cb_wait_front()`, then fires multicast sends.
- Tears down multicast state and issues NOC write barriers.

DM1's CT args live in the `WriterCTArgs` struct. From the Mcast op:

```cpp
template <typename A>
struct WriterCTArgs {
    static constexpr uint32_t dest_noc_start_x = A::dest_noc_start_x;
    static constexpr uint32_t dest_noc_start_y = A::dest_noc_start_y;
    static constexpr uint32_t dest_noc_end_x = A::dest_noc_end_x;
    static constexpr uint32_t dest_noc_end_y = A::dest_noc_end_y;
    static constexpr Semaphore sender_semaphore = A::sender_semaphore;
    static constexpr Semaphore receiver_semaphore = A::receiver_semaphore;
    static constexpr uint32_t data_size_bytes = A::data_size_bytes;
    // ...
};
```

The BRISC side of the Mcast op handles the **sender** logic:

```cpp
// blaze/ops/mcast/kernels/op.hpp -- sender logic (DM1/BRISC)
#if defined(COMPILE_FOR_BRISC)
    if constexpr (core_cta::is_sender) {
        cb_wait_front(dm1_cta::src, dm1_cta::src_num_pages);
        mcast_detail::send_data_mcast<...>(
            get_read_ptr(dm1_cta::src), get_write_ptr(dm1_cta::dst));
        // ... signal receivers via semaphore ...
    }
```

This separation is architecturally significant: DM0 and DM1 can run concurrently. While DM0 is fetching the next batch of data, DM1 can be sending the previous batch's results.

### TRISC: the compute engine

TRISC owns the **unpack-math-pack pipeline**. It reads input data from circular buffers, performs compute, and writes results back to an output circular buffer. TRISC's CT args live in the `ComputeCTArgs` struct. From `blaze/ops/matmul/kernels/op.hpp`:

```cpp
template <typename A>
struct ComputeCTArgs {
    static constexpr CB in1 = A::in1;
    static constexpr CB out = A::out;
    static constexpr uint32_t k_num_tiles = A::k_num_tiles;
    static constexpr uint32_t out_w_per_core = A::out_w_per_core;
};
```

The full TRISC compute pattern, from the Matmul op:

```cpp
// blaze/ops/matmul/kernels/op.hpp -- compute logic (TRISC)
#if defined(COMPILE_FOR_TRISC)
    if constexpr (core_cta::is_active) {
        cb_wait_front(core_cta::in0, compute_cta::k_num_tiles);
        cb_reserve_back(compute_cta::out, out_w);

        // Unpack + Math
        tile_regs_acquire();
        custom_mm_block<finalize, false>(
            core_cta::in0, compute_cta::in1,
            0, 0, 0, compute_cta::k_num_tiles, out_w);
        tile_regs_commit();

        // Pack
        tile_regs_wait();
        for (uint32_t dst_idx = 0; dst_idx < out_w; dst_idx++) {
            pack_tile(dst_idx, compute_cta::out, dst_idx);
        }
        tile_regs_release();

        cb_pop_front(core_cta::in0, compute_cta::k_num_tiles);
        cb_push_back(compute_cta::out, out_w);
    }
#endif
```

The three-phase pipeline within TRISC is visible here: `tile_regs_acquire` / `custom_mm_block` (unpack + math), `tile_regs_wait` / `pack_tile` (pack), and `tile_regs_release` (release registers for the next round).

## 1.1.3 L1 SRAM: the shared workspace

Every Tensix core has a private **L1 SRAM** (approximately 1.5 MB on Blackhole). This SRAM is the shared workspace where all three processors read and write data. There is no cache hierarchy -- L1 is the memory, and all processors on a core see the same addresses.

Data flows through L1 as follows:

```
DRAM / Remote L1
     |
     | (NOC read -- DM0/NCRISC)
     v
   +-----------------------------+
   |         L1 SRAM             |
   |                             |
   |  +--------+   +--------+   |
   |  | CB 0   |   | CB 1   |   |  Circular Buffers
   |  |(input) |   |(output)|   |
   |  +--------+   +--------+   |
   |                             |
   +-----------------------------+
     |                       |
     | (unpack/              | (NOC write -- DM1/BRISC)
     |  math/pack            |
     |  -- TRISC)            v
     |                 Remote L1 / DRAM
     v
  Register File
  (internal to TRISC)
```

Key properties of L1:

- **Directly addressable by all five RISC-V cores** on the same Tensix core.
- **Accessible via NOC** by other Tensix cores (for multicast, scatter, gather).
- **Holds circular buffer storage**, sharded tensor data, semaphore state, and configuration data.
- **Finite and precious**: all buffers for all fused phases share this space. Buffer reuse and careful sizing are essential.

## 1.1.4 Circular buffers: the intra-core synchronization primitive

**Circular buffers (CBs)** are the fundamental mechanism by which the three processors on a Tensix core communicate. A CB is a region of L1 SRAM with producer/consumer synchronization semantics:

| Operation | Called by | Effect |
|---|---|---|
| `cb_reserve_back(cb_id, n)` | Producer | Blocks until $n$ pages are free in the buffer |
| `cb_push_back(cb_id, n)` | Producer | Marks $n$ pages as ready for the consumer |
| `cb_wait_front(cb_id, n)` | Consumer | Blocks until $n$ pages are available to read |
| `cb_pop_front(cb_id, n)` | Consumer | Frees $n$ pages for the producer to reuse |

These four calls are the entire synchronization API. There are no mutexes, no condition variables, no atomics. The CB protocol is inherently deadlock-free as long as the producer/consumer relationship is acyclic.

A typical data flow through CBs on a single core:

1. **DM0 (NCRISC)** reads tiles from DRAM into CB 0 via `cb_reserve_back` / `cb_push_back`.
2. **TRISC** waits for tiles in CB 0 via `cb_wait_front`, processes them, writes results to CB 1 via `cb_reserve_back` / `cb_push_back`.
3. **DM1 (BRISC)** waits for results in CB 1 via `cb_wait_front`, sends them to remote cores.

The common shorthand for making sharded tensor data available is the `setup_sharded_buffer` utility (from `blaze/kernels/kernel_utils.hpp`), which performs the reserve-and-push sequence:

```cpp
// blaze/kernels/kernel_utils.hpp
FORCE_INLINE void setup_sharded_buffer(uint32_t cb_id, uint32_t num_tiles) {
    cb_reserve_back(cb_id, num_tiles);
    cb_push_back(cb_id, num_tiles);
}
```

You will see `setup_sharded_buffer` calls in nearly every op's NCRISC `init()` path, as in the Matmul op:

```cpp
// blaze/ops/matmul/kernels/op.hpp -- NCRISC init
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_active) {
        if constexpr (dm0_cta::in0_is_tensor_backed) {
            unified_kernels::setup_sharded_buffer(core_cta::in0, dm0_cta::k_num_tiles);
        }
    }
#endif
```

CBs are identified by integer IDs (0 through 63 on Blackhole). In TT-Blaze, CB IDs are auto-assigned by the `CBEngine` (`blaze/cb_engine.py`). The C++ kernel headers use the typed alias `CB` (defined in `blaze/kernels/ct_types.h`) to make the intent clear:

```cpp
// blaze/kernels/ct_types.h
using CB = uint32_t;           // circular buffer ID (port name)
using Semaphore = uint32_t;    // global semaphore L1 address
using PerCore = uint32_t;      // per-core varying value
using Flag = bool;             // boolean flag (is_active, pop_src, etc.)
```

These type aliases carry no runtime cost -- they are all `uint32_t` or `bool` at the C++ level. Their purpose is entirely for the Python parser (`blaze/cpp_parser.py`), which reads the kernel header and uses the type names to mechanically classify each compile-time argument as a CB reference, semaphore address, per-core value, or flag.

## 1.1.5 Semaphores: inter-core coordination

While circular buffers synchronize processors **within** a core, **semaphores** synchronize processors **across** different Tensix cores. A global semaphore is a single `uint32_t` word in L1, accessed atomically over the NOC.

The canonical inter-core synchronization pattern:

1. **Sender core's BRISC** multicasts data to $N$ receiver cores, then multicasts a semaphore increment to signal completion.
2. **Receiver core's NCRISC** calls `noc_semaphore_wait(ptr, VALID)` to block until the signal arrives.
3. **Receiver** resets the semaphore with `noc_semaphore_set(ptr, INVALID)` and pushes the received data into a local CB for TRISC to consume.

The sender/receiver pattern was shown in full in Section 1.1.2. The semaphore-specific addition is the `mcast_detail::send_with_state` call on the sender side, which multicasts the semaphore signal to all receivers:

```cpp
// DM1 (BRISC) -- after send_data_mcast (see Section 1.1.2)
mcast_detail::send_with_state<...>(
    dm1_cta::sender_semaphore, dm1_cta::receiver_semaphore, 4);
```

The `SemEngine` (`blaze/sem_engine.py`) automatically identifies which ops need semaphores by inspecting CT arg schemas for `Sem`-kind entries, and assigns sequential global semaphore indices during compilation. In the Python `emit()` function, you request semaphores through the `FusedProgram` API:

```python
# From blaze/ops/mcast/op.py
sender_sem = f.semaphore(f"{prefix}.sender")
receiver_sem = f.semaphore(f"{prefix}.receiver")
```

## 1.1.6 Parallel execution and `kernel_main()`

A crucial fact about TT-Metal's execution model: **all three logical processors execute `kernel_main()` simultaneously on every core**. There is a single `kernel_main()` function, and it is compiled three times -- once for each processor -- with different preprocessor defines active.

The generated kernel (produced by `blaze/kernel_codegen.py`) follows a uniform structure -- each fused phase is instantiated as an `Op<ct_args::...>` struct, then run through the `init()` / `operator()()` / `teardown()` lifecycle:

```cpp
// Auto-generated kernel (sketch -- see Section 1.2.6 for the full codegen
// output including PhaseInfo, DPRINT markers, and role guards)
void kernel_main() {
    // Phase 1
    OpA op_a;  op_a.init();  op_a();  op_a.teardown();
    // Phase 2
    OpB op_b;  op_b.init();  op_b();  op_b.teardown();
    // ...additional phases in topological order...
}
```

This single source file is compiled three times by the JIT -- once with `COMPILE_FOR_NCRISC` defined (for DM0), once with `COMPILE_FOR_BRISC` defined (for DM1), and once with `COMPILE_FOR_TRISC` defined (for the three TRISC sub-cores). The `#if defined(...)` guards cause each compilation to include only the code relevant to that processor. The compiler dead-code-eliminates everything else, producing three lean, specialized binaries.

The `kernel_op_api.hpp` header (`blaze/kernels/kernel_op_api.hpp`) provides constexpr booleans for use with `constexpr if` as an alternative to preprocessor guards:

```cpp
// blaze/kernels/kernel_op_api.hpp
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

And for selecting types at compile time:

```cpp
template <typename Reader, typename Writer, typename Compute>
using SelectByRISCV = std::conditional_t<
    is_ncrisc, Reader,
    std::conditional_t<is_brisc, Writer, Compute>>;
```

The kernel codegen module (`blaze/kernel_codegen.py`) tracks which processor guards are used to generate the correct `#if` structure:

```python
# blaze/kernel_codegen.py
_RISC_GUARDS = {
    "brisc": "defined(COMPILE_FOR_BRISC)",
    "ncrisc": "defined(COMPILE_FOR_NCRISC)",
    "trisc": "defined(COMPILE_FOR_TRISC)",
    "trisc0": "defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 0",
    "trisc1": "defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 1",
    "trisc2": "defined(COMPILE_FOR_TRISC) && COMPILE_FOR_TRISC == 2",
}
```

## 1.1.7 Synchronization in practice: a walkthrough

Consider RMSNorm running on a single Tensix core. All three processors start `kernel_main()` at the same time:

| Time | DM0 (NCRISC) | DM1 (BRISC) | TRISC |
|---|---|---|---|
| $t_0$ | `cb_reserve_back(input, N)` | *idle -- no BRISC logic in RMSNorm* | `cb_wait_front(input, N)` -- **blocks** |
| $t_1$ | `cb_push_back(input, N)` -- signals data ready | | `cb_wait_front` **unblocks** |
| $t_2$ | `cb_reserve_back(gamma, N)` | | Starts computing: square, reduce... |
| $t_3$ | `cb_push_back(gamma, N)` | | `cb_wait_front(gamma, N)` -- unblocks |
| $t_4$ | *done* | *done* | multiply by gamma, `cb_push_back(output, N)` |

DM1 (BRISC) has no work for RMSNorm -- there are no NOC writes. Its `kernel_main()` compiles to an empty function body and returns immediately. This is by design, and it enables the overlap behavior described in the next section.

## 1.1.8 Why empty processors enable overlap

This is a key performance insight in TT-Blaze's fused kernel model. When multiple ops are fused into a single kernel, each op only occupies the processors it needs. Consider a fused kernel with three phases:

| Phase | DM0 (NCRISC) | DM1 (BRISC) | TRISC |
|---|---|---|---|
| Mcast | Wait for semaphore, push dst CB | Multicast data | **Empty** -- falls through immediately |
| Matmul | Push activations CB | **Empty** -- falls through immediately | Wait for CB, compute matmul |
| RMSNorm | Push gamma CB | **Empty** -- falls through immediately | Wait for CB, compute RMSNorm |

Because TRISC has no work in the Mcast phase, it enters the Matmul phase while DM0 and DM1 are still finishing the multicast. TRISC blocks at `cb_wait_front()` in Matmul, but it is already "queued up" and will proceed the instant the data arrives. Similarly, DM1 has no work in Matmul or RMSNorm, so it races ahead through both phases while TRISC is still computing.

This creates two distinct performance benefits:

**Benefit 1: Eliminated dispatch overhead.** In separate kernels, each kernel launch incurs host-side dispatch latency ($T_{\text{dispatch}}$) and per-kernel initialization ($T_{\text{init}}$). In a fused kernel, there is a single dispatch and processors transition between phases with zero overhead.

**Benefit 2: True pipeline overlap across phases.** Because idle processors advance immediately, data movement and compute from *different phases* overlap temporally. In the example above, DM0 can begin setting up Matmul's input CBs (Phase 2) while TRISC is still blocked waiting for Mcast data (Phase 1). Once Mcast completes and TRISC starts computing Matmul, DM1 has already finished its Phase 1 work and races ahead. In a steady-state pipeline with repeated iterations, this gives:

$$T_{\text{fused\_per\_iteration}} \approx \max(T_{\text{data\_movement}}, T_{\text{compute}})$$

rather than the fully serialized separate-kernel cost:

$$T_{\text{separate\_per\_iteration}} = T_{\text{dispatch}} + T_{\text{data\_movement}} + T_{\text{init}} + T_{\text{compute}}$$

For a single pass through the fused phases (not steady state), the benefit is the elimination of $T_{\text{dispatch}}$ and $T_{\text{init}}$ between phases, since processors are pre-positioned and begin the instant their input data arrives.

This overlap is "free" -- no explicit scheduling is needed. The hardware's concurrent execution of `kernel_main()` across all three processors, combined with the fact that empty function bodies return instantly, gives fused kernels an inherent latency advantage over dispatching separate programs.

The key design principle: **do not add unnecessary work to a processor just for symmetry.** If your op does not need DM1, leave its code path empty. DM1 will advance to the next fused phase, enabling it to start its work (if any) earlier.

In kernel code, this pattern appears as `constexpr if` blocks that compile to no-ops on certain processors:

```cpp
// blaze/ops/copy/kernels/op.hpp -- processor selection
void operator()() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (cta::use_ncrisc) { copy_impl(); }
#elif defined(COMPILE_FOR_BRISC)
    if constexpr (!cta::use_ncrisc) { copy_impl(); }
#endif
    // TRISC: no code path at all -- immediately advances to next phase
}
```

The code generator detects empty `init()` and `teardown()` bodies and omits them (see Section 1.2.6).

## Key takeaways

1. **Five processors, three roles:** Every Tensix core has DM0 (NCRISC, NOC reads), DM1 (BRISC, NOC writes), and TRISC (3 sub-cores: unpack/math/pack compute). All three run `kernel_main()` in parallel.

2. **Circular buffers synchronize within a core:** The producer/consumer protocol (`cb_reserve_back`/`cb_push_back` vs. `cb_wait_front`/`cb_pop_front`) is how data flows between processors on the same core. No locks needed.

3. **Semaphores synchronize across cores:** Global semaphores (`noc_semaphore_wait`/`noc_semaphore_set`) coordinate data transfers between different Tensix cores (e.g., multicast sender and receivers).

4. **One source file, three compilations:** Every kernel `.hpp` file uses `COMPILE_FOR_*` preprocessor guards to place per-processor code in a single file. The JIT compiler compiles the same source three times, once per processor.

5. **Empty phases overlap for free:** When a processor has no work for a phase, it advances immediately. This means fused kernels get free overlap at phase boundaries -- a core architectural advantage of TT-Blaze's fusion model.

6. **Type aliases drive automation:** The `CB`, `Semaphore`, `PerCore`, and `Flag` type aliases in `ct_types.h` carry zero runtime cost but enable the Python parser to mechanically classify compile-time arguments.

---

**Next:** [`02_blaze_compilation_pipeline.md`](./02_blaze_compilation_pipeline.md)
