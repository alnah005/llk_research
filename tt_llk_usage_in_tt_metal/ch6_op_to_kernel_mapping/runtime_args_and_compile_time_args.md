# Runtime Args and Compile-Time Args

Device kernels in TT-Metal receive parameters through two distinct mechanisms: **compile-time arguments** (baked into the compiled binary) and **runtime arguments** (written to L1 memory before each kernel launch). Understanding when to use each is essential for writing efficient operations.

## Compile-Time Arguments

Compile-time arguments are integer values known when the kernel is compiled. They are embedded directly into the binary, so the compiler can optimize based on their values (loop unrolling, dead code elimination, etc.).

### Setting from Host

Compile-time args are provided via `ComputeConfig.compile_args` (positional) and `ComputeConfig.named_compile_args` (named):

```cpp
// From matmul_multicore_reuse_mcast_2d_program_factory.cpp (line 848)
tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
    "bmm_large_block_zm_fused_bias_activation.cpp",
    all_cores_with_work,
    tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_kernel_args,   // positional
        .defines = mm_kernel_defines,
        .named_compile_args = {                // named
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in1", tt::CBIndex::c_1},
            {"cb_bias", tt::CBIndex::c_3},
            {"cb_out", tt::CBIndex::c_4},
            {"cb_intermed0", tt::CBIndex::c_5},
            // ...
        }});
```

### Reading on Device

Positional args are accessed by index:

```cpp
// From bmm_large_block_zm.cpp (line 11)
uint32_t in0_block_w = get_compile_time_arg_val(0);
uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
uint32_t in1_per_core_w = get_compile_time_arg_val(6);
uint32_t num_blocks = get_compile_time_arg_val(7);
uint32_t out_subblock_h = get_compile_time_arg_val(8);
uint32_t out_subblock_w = get_compile_time_arg_val(9);
uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);
uint32_t batch = get_compile_time_arg_val(11);
```

Named args are accessed by string key and can be used with `constexpr`:

```cpp
// From bmm_large_block_zm.cpp (line 24)
constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
constexpr uint32_t cb_intermed0 = get_named_compile_time_arg_val("cb_intermed0");
```

The `constexpr` qualifier is significant: because named compile-time args are known at compile time, they can be used in template parameters, array sizes, and other contexts that require constant expressions. This is especially useful for circular buffer indices, which are used as template arguments to LLK functions.

### What Compile-Time Args Control

Compile-time args typically define the **structure** of the computation:

- **Tile counts and block sizes**: `in0_block_w`, `out_subblock_h`, `out_subblock_w` in matmul.
- **Loop bounds**: `num_blocks`, `batch` -- the compiler can unroll inner loops when these are known.
- **CB indices**: Which circular buffers to read from and write to.
- **Dimensional parameters**: `Ht`, `Wt`, `NC` in reduction kernels.

For reduction (`reduce_h.cpp`, line 11):

```cpp
uint32_t Ht = get_compile_time_arg_val(0);
uint32_t Wt = get_compile_time_arg_val(1);
uint32_t NC = get_compile_time_arg_val(2);
uint32_t row_chunk = get_compile_time_arg_val(3);
```

### Trade-off: Recompilation

Each unique set of compile-time args produces a distinct compiled binary. TT-Metal caches compiled kernels, so repeated invocations with the same args do not trigger recompilation. However, if args change frequently (e.g., varying tensor shapes across iterations), compile-time args cause cache misses and recompilation overhead. Use runtime args for values that change between invocations.

## Runtime Arguments

Runtime arguments are `uint32_t` values written to L1 memory before each kernel dispatch. They do not affect compilation -- the same binary can run with different runtime args.

### Setting from Host

Runtime args are set via `tt_metal::SetRuntimeArgs()`:

```cpp
// From eltwise_multi_core_program_factory_common.hpp (line 303)
SetRuntimeArgs(program, binary_reader_kernel_id, cores, binary_reader_args);
SetRuntimeArgs(program, eltwise_binary_kernel_id, cores, eltwise_binary_args);
SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);
```

The third argument (`cores`) specifies that different cores can receive different runtime arg values -- essential for data-parallel ops where each core processes a different slice of the tensor.

### Reading on Device

Runtime args are read via `get_arg_val<T>(idx)`:

```cpp
// From eltwise_binary_kernel.cpp (line 14)
uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
```

Under the hood, `get_arg_val()` reads from a fixed location in L1 memory where the host wrote the args before kernel launch. There is a maximum of 341 runtime args per kernel (defined as `max_runtime_args` in `kernel_types.hpp`, line 24), calculated from the 4096-byte dispatch packet size divided by 3 kernels per Tensix core.

### What Runtime Args Control

Runtime args typically define the **data** of the computation:

- **Tensor addresses**: DRAM or L1 buffer addresses that vary per invocation.
- **Per-core tile counts**: When work is split unevenly across cores.
- **Offsets**: Start tile IDs for each core's data slice.
- **Addresses of output buffers**: Different outputs per invocation.

## Compile-Time vs. Runtime: Decision Guide

| Consideration | Use Compile-Time | Use Runtime |
|--------------|-----------------|-------------|
| Value affects kernel structure (loop nesting, CB choice) | Yes | No |
| Value changes between invocations | No | Yes |
| Value differs per core | Possible but wasteful | Yes |
| Want compiler optimization (unrolling, DCE) | Yes | N/A |
| Need `constexpr` (template args, array sizes) | Yes | No |
| Avoiding recompilation overhead | No | Yes |

## Interaction with Defines

Compile-time args and defines serve complementary roles:

- **Defines** select which code paths exist (via `#ifdef`) and which LLK function variants to call (via macro expansion). They control the **qualitative** behavior of the kernel.
- **Compile-time args** parameterize the selected code paths with numeric values (tile counts, block dimensions). They control the **quantitative** behavior.
- **Runtime args** provide per-invocation data that neither defines nor compile-time args can handle.

For example, in the eltwise binary kernel:
- The `ELTWISE_OP` define selects `add_tiles` (qualitative).
- There are no compile-time args for this particular kernel (all parameterization is via defines and runtime args).
- Runtime args `per_core_block_cnt` and `per_core_block_size` specify how many tiles this core processes (quantitative, per-invocation).

In the matmul kernel:
- The `mm_kernel_defines` map might set `FUSE_BIAS` or `SFPU_OP_CHAIN_0` (qualitative).
- Compile-time args set `in0_block_w`, `out_subblock_h`, `num_blocks`, etc. (quantitative, structural).
- Named compile-time args set `cb_in0`, `cb_out`, etc. (structural).
- Runtime args (in the reader/writer kernels) set tensor addresses and offsets (per-invocation).

---

**Next:** [`concrete_examples.md`](./concrete_examples.md)
