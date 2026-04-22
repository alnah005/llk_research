# Advantages of Preprocessor Defines as Configuration

## 1. Zero-Cost Abstraction: All Configuration Resolved at Compile Time

Preprocessor defines are eliminated entirely before code generation. Unlike runtime branches that consume cycles on the RISC-V cores inside Tensix, `#ifdef` branches produce no conditional instructions in the final binary. The compiler sees only the active code path and can optimize it fully -- inlining constants, eliminating dead code, and scheduling instructions without penalty.

This is particularly valuable for the math TRISC, where every cycle in the inner tile loop matters. Consider the layernorm kernel, which uses defines to choose between LayerNorm and RMSNorm:

```cpp
// ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp

#ifndef RMSNORM
    // E[x] -- row-wise mean computation
    numeric::row_wise_mean<FLOAT32_REDUCTION, policies::FullBlockWithoutPopPolicy>(
        cb_x, cb_scaler, cb_ex, W, Wt, blk);

    // x - E[x] -- subtract mean
    sub_bcast_cols_init_short(cb_x, cb_ex);
    for (auto block : generic::blocks(Wt, blk)) {
        // ...
        sub_tiles_bcast_cols(cb_x, cb_ex, i, 0, i);
        // ...
    }
#endif
```

When `RMSNORM` is defined, the entire mean-subtraction loop is removed from the binary. No branch prediction, no dead register allocation -- it simply does not exist. A runtime `if (is_rmsnorm)` check would still compile both paths and pay the code-size cost even if only one runs.

## 2. Single Kernel Source Serves Many Operation Variants

The define system allows one kernel source file to express a family of related operations. The eltwise binary kernel (`eltwise_binary_kernel.cpp`) is a clear example. This single 140-line file handles:

- Basic binary ops (add, sub, mul) via `ELTWISE_OP` and `ELTWISE_OP_TYPE`
- Fused post-op activations via `SFPU_OP_INIT_0` / `SFPU_OP_FUNC_0` / `SFPU_OP_CHAIN_0`
- Pre-scaling of input 0 via `SFPU_OP_INIT_PRE_IN0_0` / `SFPU_OP_FUNC_PRE_IN0_0`
- Pre-scaling of input 1 via `SFPU_OP_INIT_PRE_IN1_0` / `SFPU_OP_FUNC_PRE_IN1_0`
- ReLU packing via `PACK_RELU`

The host-side `get_defines()` function in `binary_op_utils.cpp` maps each `BinaryOpType` to the correct set of defines. For instance, `BinaryOpType::LOGADDEXP` produces:

```cpp
// ttnn/cpp/ttnn/operations/eltwise/binary/common/binary_op_utils.cpp (lines 95-101)

case BinaryOpType::LOGADDEXP:
    defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN0_0"));
    defines.merge(get_defines(UnaryOpType::EXP, std::vector<float>{0}, "PRE_IN1_0"));
    op_name = "add_tiles";
    op_binary_type = "EltwiseBinaryType::ELWADD";
    break;
```

This generates defines like `SFPU_OP_INIT_PRE_IN0_0`, `SFPU_OP_FUNC_PRE_IN0_0`, etc. that inject `exp()` calls before each input reaches the binary add. Without this mechanism, `logaddexp` would require its own dedicated kernel file, duplicating the surrounding tile loop boilerplate.

The matmul family similarly uses defines to express variants across a single compute kernel. The `bmm_large_block_zm_fused_bias_activation.cpp` kernel (25 `#ifdef` directives) handles bias fusion (`FUSE_BIAS`), ReLU packing (`PACK_RELU`), L1 accumulation (`PACKER_L1_ACC`), FP32 destination accumulation (`FP32_DEST_ACC_EN`), input transpose (`IN1_TRANSPOSE_TILE`), and DRAM-sharded operation (`MATMUL_DRAM_SHARDED`).

## 3. Build Cache Deduplication via Define Hashing

The JIT build cache uses the defines map as part of the kernel hash (via `Kernel::compute_hash()` in `tt_metal/impl/kernels/kernel.cpp`). This means:

- **Identical configurations share binaries.** If two different operations happen to produce the same defines and compile-time arguments, they reuse the same compiled kernel. For example, a `torch.add` and a bias-add fused into another op that both compile `eltwise_binary_kernel.cpp` with `ELTWISE_OP=add_tiles` will share the same binary.

- **Only genuinely different variants incur compilation cost.** The hash is computed from define keys and values, compile-time args, and kernel source. The `JitBuildCache::build_once()` method uses this hash to gate compilation:

```cpp
// tt_metal/jit_build/jit_build_cache.cpp (lines 9-27)

void JitBuildCache::build_once(size_t hash, const std::function<void()>& build_fn) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        auto it = entries_.find(hash);
        if (it != entries_.end()) {
            if (it->second == State::Built) {
                return;  // Already built, reuse cached binary
            }
            cv_.wait(lock);  // Another thread is building this hash
            continue;
        }
        entries_.emplace(hash, State::Building);
        break;
    }
    // ... perform build ...
}
```

This deduplication is transparent -- program factories do not need to manually check for cache hits. They simply declare their defines, and the build system handles the rest.

## 4. Host-Device Decoupling: Declarative Behavior Specification

The defines mechanism creates a clean boundary between host-side operation logic and device-side kernel code. The host factory expresses *what* the kernel should do (e.g., "fuse a pre-add, use RMSNorm, apply GELU activation") without knowing *how* the kernel implements it. The kernel source expresses *how* each feature works without knowing which operation triggered it.

This is visible in the layernorm program factory, where the host builds up defines declaratively:

```cpp
// ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp

if (fuse_pre_add) {
    reader_defines.emplace_back("FUSE_PRE_ADD", "1");
    if (!use_welford) {
        compute_defines.emplace_back("FUSE_PRE_ADD", "1");
    }
}
if (gamma.has_value()) {
    reader_defines.emplace_back("FUSE_GAMMA", "1");
}
if (beta.has_value()) {
    reader_defines.emplace_back("FUSE_BETA", "1");
}
if (rms_norm) {
    reader_defines.emplace_back("RMSNORM", "1");
    compute_defines.emplace_back("RMSNORM", "1");
}
```

The host does not need to understand the internal kernel control flow. It simply toggles feature flags. Similarly, the kernel does not need to decode runtime arguments to figure out which features are active -- the preprocessor has already resolved the configuration.

The SDPA program factory demonstrates the same pattern with numerical tuning parameters passed as defines:

```cpp
// ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp (lines 502-507)

defines["STATS_GRANULARITY"] = std::to_string(stats_granularity);
defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
defines["MUL_BCAST_GRANULARITY"] = std::to_string(mul_bcast_granularity);
defines["DHT_GRANULARITY"] = std::to_string(dht_granularity);
defines["REDUCE_GRANULARITY"] = std::to_string(reduce_granularity);
defines["EXP_APPROX_MODE"] = std::to_string(exp_approx_mode);
```

These granularity values control loop unrolling factors inside the SDPA compute kernel. By passing them as defines rather than runtime arguments, the compiler can fully unroll loops and schedule instructions with known trip counts.

---

**Next:** [`disadvantages_of_defines.md`](./disadvantages_of_defines.md)
