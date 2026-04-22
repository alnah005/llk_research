# Disadvantages of Preprocessor Defines as Configuration

## 1. Readability: Kernel Files Become `#ifdef` Forests

Heavy use of conditional compilation makes kernel source difficult to follow. The reader must mentally evaluate multiple define combinations to understand what any given code path actually does.

Measured `#ifdef` / `#ifndef` / `#if defined` counts in key kernel files:

| Kernel file | Directive count |
|---|---|
| `layernorm.cpp` (non-sharded) | 17 |
| `compute_common.hpp` (SDPA) | 15 |
| `layernorm_sharded.cpp` | 14 |
| `bmm_large_block_zm_fused_bias_activation.cpp` | 25 |
| `eltwise_binary_kernel.cpp` | 9 |

The layernorm kernel is a concrete example. Within a single 303-line file, the reader encounters interleaved `#ifdef FUSE_PRE_ADD`, `#ifdef RMSNORM`, `#ifndef RMSNORM`, `#ifdef SFPU_OP_INIT_ACTIVATION`, and `#if defined RMSNORM and not defined FUSE_PRE_ADD` blocks. Some of these are nested:

```cpp
// ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp (lines 66-84)

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t cb_x = cb_xmm;
#else
    constexpr uint32_t cb_x = tt::CBIndex::c_23;
#endif
#else
    constexpr uint32_t cb_x = cb_in;
#endif

#ifdef FUSE_PRE_ADD
    binary_op_init_common(cb_in, cb_inb, cb_x);
#else
#ifdef RMSNORM
    binary_op_init_common(cb_xmm, cb_xmm, cb_xmm2);
#else
    binary_op_init_common(cb_x, cb_scaler, cb_ex);
#endif
#endif
```

This produces a 2x2 matrix of behaviors (FUSE_PRE_ADD x RMSNORM) encoded as nested preprocessor blocks. A developer modifying the variance computation must trace through all four combinations to verify correctness.

The eltwise binary kernel similarly nests `#ifdef SFPU_OP_INIT_PRE_IN0_0` inside `#ifndef SFPU_OP_INIT_PRE_IN1_0`, requiring the reader to reason about 4 possible pre-scaling configurations.

## 2. Implicit Contract: No Schema or Validation for Expected Defines

There is no formal specification of which defines a kernel expects. The "API" between a program factory and its kernel is an implicit agreement encoded in string literals scattered across two separate files in different directories.

The host side constructs arbitrary string keys:

```cpp
// ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp (line 165)

defines["ELTWISE_OP"] = op_name;
defines["ELTWISE_OP_TYPE"] = op_binary_type;
```

The kernel side checks for those strings using `#ifdef`:

```cpp
// eltwise_binary_kernel.cpp (line 119)

#ifdef SFPU_OP_INIT_0
    SFPU_OP_INIT_0
    SFPU_OP_FUNC_0
#endif
```

Nothing enforces that these names match. A typo in the factory (`SFPU_OP_INIT0` instead of `SFPU_OP_INIT_0`) produces no compile error and no runtime error -- the kernel simply takes the `#else` path (or omits the block entirely).

There is no equivalent of a header file declaring "this kernel requires `ELTWISE_OP`, `ELTWISE_OP_TYPE`, and optionally `PACK_RELU`." The documentation exists only in comments (if at all) and in the convention of reading the kernel source to discover which defines it checks.

## 3. Combinatorial Explosion: Each Combination Produces a Distinct JIT Binary

Because defines are part of the kernel hash, every unique define combination compiles a separate binary. This creates exponential growth in compilation work.

Consider the matmul 2D multicast program factory (`matmul_multicore_reuse_mcast_2d_program_factory.cpp`), which can set any combination of:

- `FUSE_BIAS`
- `PACK_RELU`
- `PACKER_L1_ACC`
- `FP32_DEST_ACC_EN`
- `IN1_TRANSPOSE_TILE`
- `IN0_SHARDED`
- `SKIP_MCAST`
- `IN1_SHARDED` / `IN1_DRAM_WIDTH_SHARDED` / `IN1_DRAM_HEIGHT_SHARDED`
- `OUT_SHARDED`
- `INTERMEDIATE_CB_READ`

Even with only the boolean flags, this is over 1,000 theoretical combinations for the compute kernel alone. In practice, many combinations are unreachable, but the build system has no way to know this -- each new combination encountered at runtime triggers a fresh JIT compilation.

The SDPA program factory adds granularity parameters as defines with numeric values:

```cpp
defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
// ... 4 more granularity defines ...
```

Each distinct sequence length or head dimension can produce different granularity values, which means different hashes and different compiled binaries. A sweep over 10 sequence lengths can trigger 10 separate compilations of the same kernel with only minor constant differences.

## 4. Error Opacity: Missing Defines Silently Take the Wrong Branch

When a define is absent, the preprocessor silently takes the `#else` path (or skips the `#ifdef` block entirely). This can produce functional but incorrect kernels with no diagnostic.

For example, if the layernorm factory fails to set `RMSNORM` for an RMSNorm operation, the kernel silently computes LayerNorm (which includes mean subtraction). The output will be numerically wrong, but the kernel compiles and runs without error. The bug manifests as a subtle accuracy regression rather than a clear failure.

Similarly, the eltwise binary kernel's `PACK_RELU` define:

```cpp
// eltwise_binary_kernel.cpp (line 39)

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif
```

If the factory forgets to set `PACK_RELU` when it should, the kernel runs without ReLU clamping. The output has negative values where there should be none, but nothing crashes.

This is fundamentally different from a missing function argument or a missing template parameter, both of which produce compile errors. The define system's failure mode is silent correctness bugs.

## 5. Refactoring Hazard: Changes Require Coordinating Multiple Layers

Adding, removing, or renaming a define requires changes across at least three locations that have no compile-time coupling:

1. **The kernel source** (e.g., `layernorm.cpp`) -- where the `#ifdef` is read.
2. **The program factory** (e.g., `layernorm_op_multi_core.cpp`) -- where the define is set.
3. **The operation utility** (e.g., `binary_op_utils.cpp`) -- where the define map is constructed from operation parameters.

In some cases, there is a fourth layer: the Python binding that passes the operation type from user code to the C++ factory.

The layernorm operation distributes defines across `layernorm_op_multi_core.cpp` (non-sharded factory), `layernorm_op_multi_core_sharded.cpp` (sharded factory), and `sharded_layernorm_factory_helpers.cpp` (shared helper). The helper constructs reader, writer, and compute defines separately:

```cpp
// sharded_layernorm_factory_helpers.cpp (lines 482-522)

KernelDefines defines;

// Reader defines
if (has_b)           defines.reader.emplace_back("FUSE_PRE_ADD", "1");
if (has_gamma)       defines.reader.emplace_back("FUSE_GAMMA", "1");
if (has_beta)        defines.reader.emplace_back("FUSE_BETA", "1");

// Writer defines
if (rms_norm)        defines.writer.emplace_back("RMSNORM", "1");
if (skip_write_back) defines.writer.emplace_back("SKIP_WRITE_BACK", "1");

// Compute defines
if (has_b)           defines.compute.emplace_back("FUSE_PRE_ADD", "1");
if (rms_norm && !use_welford) defines.compute.emplace_back("RMSNORM", "1");
```

If a developer adds a new define to the compute kernel (say, `FUSE_DROPOUT`), they must update this helper, ensure both sharded and non-sharded factories pass it through, and verify that the reader and writer kernels do not need corresponding changes. No type system or interface definition enforces consistency. The only safety net is tests.

The matmul family is even more fragmented. The define `FUSE_BIAS` must be set on *both* the compute kernel and the writer kernel:

```cpp
// matmul_multicore_reuse_mcast_2d_program_factory.cpp (lines 555-558)

mm_kernel_defines["FUSE_BIAS"] = "1";
mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
mm_kernel_in1_receiver_writer_defines["FUSE_BIAS"] = "1";
mm_kernel_in1_receiver_writer_other_noc_setup_defines["FUSE_BIAS"] = "1";
```

Forgetting to set it on one of the four define maps produces a kernel mismatch: the compute kernel writes bias-corrected tiles, but the writer kernel does not expect the extra circular buffer traffic. This kind of bug is difficult to diagnose because it manifests as a hang or data corruption rather than a compile error.

---

**Next:** [Chapter 4 -- Tile Processing Pattern](../ch4_tile_processing_pattern/index.md)
