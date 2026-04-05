# Concrete Examples

This section walks through three real TTNN operations end-to-end, showing how each host-side factory configures, compiles, and dispatches a compute kernel that uses LLK.

## Example 1: Eltwise Binary (Add / Sub / Mul)

**Host factory:** `ttnn/cpp/ttnn/operations/eltwise/binary/device/element_wise_multi_core_program_factory.cpp`
**Device kernel:** `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp`

### Step 1: Build the Defines Map

The factory calls `utils::get_defines()` (line 99) to construct a defines map based on the requested operation type:

```cpp
std::map<std::string, std::string> eltwise_defines = utils::get_defines(
    op_type, a.dtype(), output.dtype(), fused_activations,
    operation_attributes.input_tensor_a_activation);
```

For `BinaryOpType::ADD`, this produces:

```
ELTWISE_OP        = "add_tiles"
ELTWISE_OP_TYPE   = "EltwiseBinaryType::ELWADD"
```

For `BinaryOpType::ADD` with a fused RELU activation:

```
ELTWISE_OP        = "add_tiles"
ELTWISE_OP_TYPE   = "EltwiseBinaryType::ELWADD"
PACK_RELU         = "1"
```

For `BinaryOpType::DIV` (decomposed as `a * recip(b)`):

```
ELTWISE_OP                = "mul_tiles"
ELTWISE_OP_TYPE           = "EltwiseBinaryType::ELWMUL"
SFPU_OP_INIT_PRE_IN1_0   = "recip_tile_init();"
SFPU_OP_FUNC_PRE_IN1_0   = "recip_tile(i);"
```

### Step 2: Create the Compute Kernel

The factory passes the defines to `CreateKernel()` (line 175):

```cpp
auto eltwise_binary_kernel_id = tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/"
    "eltwise_binary_kernel.cpp",
    all_device_cores,
    tt_metal::ComputeConfig{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .defines = eltwise_defines
    });
```

Note that this kernel uses no compile-time args -- all parameterization is through defines and runtime args.

### Step 3: Set Runtime Args

The factory calls `set_eltwise_binary_runtime_args()` (line 181), which distributes work across cores and calls `SetRuntimeArgs()` for each kernel:

```cpp
SetRuntimeArgs(program, binary_reader_kernel_id, cores, binary_reader_args);
SetRuntimeArgs(program, eltwise_binary_kernel_id, cores, eltwise_binary_args);
SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);
```

The compute kernel receives two runtime args per core:
- `per_core_block_cnt` -- number of tile blocks this core processes.
- `per_core_block_size` -- number of tiles per block.

### Step 4: Device Execution

The compiled kernel runs on each assigned core. The full `kernel_main()` for the common case (no pre-scaling):

```cpp
void kernel_main() {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_2;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    binary_tiles_init<false, ELTWISE_OP_TYPE>(cb_in0, cb_in1);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            ELTWISE_OP(cb_in0, cb_in1, i, i, i);  // e.g., add_tiles(...)
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);
        }
        tile_regs_release();

        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
```

The LLK functions invoked are:
- `binary_op_init_common()` -- initializes unpack/pack for binary operation.
- `binary_tiles_init<false, EltwiseBinaryType::ELWADD>()` -- initializes the FPU for add.
- `add_tiles()` -- the actual FPU tile-add (expanded from `ELTWISE_OP`).
- `pack_tile()` -- packs results from DST to the output CB.

The acquire/commit/wait/release cycle (covered in Chapter 3) governs DST register access throughout.

---

## Example 2: Matmul (Block Matrix Multiply)

**Host factory:** `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`
**Device kernel:** `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp`

### Step 1: Compile-Time Args

The matmul factory assembles a vector of 12+ compile-time args that define the block structure:

```cpp
std::vector<uint32_t> compute_kernel_args = {
    in0_block_w,              // [0]  inner block size in tiles
    in0_num_subblocks,        // [1]  outer row block count
    in0_block_num_tiles,      // [2]  total tiles in in0 block
    in0_subblock_num_tiles,   // [3]  tiles per in0 subblock
    in1_num_subblocks,        // [4]  outer column block count
    in1_block_num_tiles,      // [5]  total tiles in in1 block
    in1_per_core_w,           // [6]  in1 width per core
    num_blocks,               // [7]  outer inner dimension blocks
    out_subblock_h,           // [8]  output subblock height
    out_subblock_w,           // [9]  output subblock width
    out_subblock_num_tiles,   // [10] tiles per output subblock
    batch,                    // [11] batch dimension
    // ...
};
```

Named compile-time args map CB indices to meaningful names:

```cpp
.named_compile_args = {
    {"cb_in0", tt::CBIndex::c_0},
    {"cb_in1", tt::CBIndex::c_1},
    {"cb_bias", tt::CBIndex::c_3},
    {"cb_out", tt::CBIndex::c_4},
    {"cb_intermed0", tt::CBIndex::c_5},
    // ...
}
```

### Step 2: Create with Full ComputeConfig

```cpp
tt_metal::CreateKernel(
    program,
    "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/"
    "bmm_large_block_zm_fused_bias_activation.cpp",
    all_cores_with_work,
    tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_kernel_args,
        .defines = mm_kernel_defines,
        .named_compile_args = { /* ... */ }
    });
```

Here, `math_fidelity` directly controls LLK's FPU precision for the matmul. `fp32_dest_acc_en` determines whether DST uses FP32 accumulation (critical for large matmul accuracy).

### Step 3: Device Execution

The kernel (`bmm_large_block_zm.cpp`) implements a blocked matrix multiply with the classic three-level loop nest:

```cpp
void kernel_main() {
    // Read all compile-time args
    uint32_t in0_block_w = get_compile_time_arg_val(0);
    // ... (11 more positional args)

    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t cb_intermed0 = get_named_compile_time_arg_val("cb_intermed0");

    mm_init(cb_in0, cb_in1, cb_intermed0);  // LLK matmul init

    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_wait_front(cb_in0, in0_block_num_tiles);
            cb_wait_front(cb_in1, in1_block_num_tiles);

            for (uint32_t in0_subblock = 0; ...) {
                for (uint32_t in1_subblock = 0; ...) {
                    acquire_dst();

                    // Inner product: accumulate across in0_block_w tiles
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                matmul_tiles(cb_in0, cb_in1,
                                    in0_index, in1_index, dst_index);
                            }
                        }
                    }

                    // Pack results
                    if (last_out) {
                        cb_reserve_back(cb_out, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, cb_out);
                        }
                        cb_push_back(cb_out, out_subblock_num_tiles);
                    } else {
                        // Spill partial results to intermediate CB
                        cb_reserve_back(cb_intermed0, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            pack_tile(i, cb_intermed0);
                        }
                        cb_push_back(cb_intermed0, out_subblock_num_tiles);
                    }

                    release_dst();
                }
            }
            cb_pop_front(cb_in0, in0_block_num_tiles);
            cb_pop_front(cb_in1, in1_block_num_tiles);
        }
    }
}
```

Key LLK functions:
- `mm_init()` -- initializes unpack, math, and pack for matmul mode.
- `matmul_tiles()` -- the core FPU multiply-accumulate on two tiles, accumulating into DST.
- `copy_tile_to_dst_init_short_with_dt()` and `copy_tile()` -- reload partial sums from the intermediate CB when spilling.
- `mm_init_short_with_dt()` -- re-initialize matmul after a reload (data format switch).
- `pack_tile()` -- pack results from DST to output or intermediate CB.

The blocked structure exists because DST has limited registers. When the inner dimension (`num_blocks`) is large, the kernel must spill partial sums to `cb_intermed0` and reload them for the next block.

---

## Example 3: Reduction (Reduce Along H)

**Host factory:** `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp`
**Device kernel:** `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h.cpp`

### Step 1: Defines for Reduction Type

The factory builds defines via `reduce_op_utils::get_defines()` (line 189):

```cpp
std::map<std::string, std::string> reduce_defines =
    reduce_op_utils::get_defines(
        operation_attributes.math_op,
        tt::tt_metal::ReduceOpDim::H);
```

For a sum-reduction along H, this produces:

```
REDUCE_OP  = "PoolType::SUM"
REDUCE_DIM = "ReduceDim::REDUCE_COL"
```

For a max-reduction along H:

```
REDUCE_OP  = "PoolType::MAX"
REDUCE_DIM = "ReduceDim::REDUCE_COL"
```

### Step 2: Compile-Time Args and Kernel Creation

```cpp
std::vector<uint32_t> compute_kernel_args_group_1 = {
    Ht,                         // [0] Height in tiles
    num_cols_per_core_group_1,  // [1] Width tiles per core
    1,                          // [2] NC (batch count)
    chunk_size,                 // [3] Column chunk size
};

tt_metal::CreateKernel(
    program,
    compute_kernel,  // "reduce_h.cpp" or "reduce_h_neg.cpp"
    core_group_1,
    tt_metal::ComputeConfig{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .compile_args = compute_kernel_args_group_1,
        .defines = reduce_defines
    });
```

Note that multi-core reduction creates separate kernel instances for core groups with different `num_cols_per_core` values (when tiles do not divide evenly across cores).

### Step 3: Device Execution

The kernel (`reduce_h.cpp`) processes tiles in a chunked column-major order:

```cpp
void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    uint32_t row_chunk = get_compile_time_arg_val(3);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb2(tt::CBIndex::c_2);
    experimental::CircularBuffer cb3(tt::CBIndex::c_3);

    reduce_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

    cb2.wait_front(1);  // scaler tile from reader

    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt = 0; wt < Wt; wt += row_chunk) {
            uint32_t chunk_end = std::min(wt + row_chunk, Wt);
            int reduce_dst_idx = 0;

            acquire_dst();
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                reduce_dst_idx = 0;
                for (uint32_t i = wt; i < chunk_end; ++i) {
                    cb0.wait_front(1);
                    reduce_tile(tt::CB::c_in0, tt::CB::c_in2,
                                0, 0, reduce_dst_idx);
                    cb0.pop_front(1);
                    ++reduce_dst_idx;
                }
            }
            // Pack reduced results
            for (uint32_t i = wt; i < chunk_end; ++i) {
                cb3.reserve_back(1);
                pack_tile((i - wt), tt::CBIndex::c_3);
                cb3.push_back(1);
            }
            release_dst();
        }
    }
}
```

Key LLK functions:
- `reduce_init()` -- initializes unpack, math, and pack for reduction mode (parameterized by `REDUCE_OP` and `REDUCE_DIM` defines).
- `reduce_tile()` -- accumulates one input tile into a DST register using the configured reduction op (sum or max) and dimension (column reduction for reduce-H).
- `pack_tile()` -- packs the accumulated result from DST.

The chunking strategy (`row_chunk`) exists because DST has limited registers. Each chunk processes `row_chunk` columns simultaneously, accumulating `Ht` tiles per column into separate DST slots. The reader kernel delivers tiles in the order the compute kernel expects (column chunks, then rows within each chunk).

### The Scaler Tile

The `cb2.wait_front(1)` at line 23 reads a "scaler tile" provided by the reader kernel. For sum-reduction, this is typically a tile of ones (or `1/Ht` for mean). The scaler is passed as the second input to `reduce_tile()`, which multiplies each element by the scaler during accumulation. This avoids a separate scaling pass.

---

## Summary: The Configuration Chain

All three examples follow the same pattern:

```
TTNN Op (Python) 
  -> C++ Operation (device_operation.cpp)
    -> Program Factory (select variant based on layout/sharding)
      -> get_defines() builds defines map (selects LLK functions)
      -> Assemble compile_args vector (parameterizes block structure)
      -> CreateKernel() with ComputeConfig (compiles kernel binary)
      -> SetRuntimeArgs() with per-core data (tensor addresses, offsets)
        -> Device kernel runs kernel_main()
          -> LLK init functions (configured by defines + ComputeConfig fields)
          -> LLK compute functions (selected by defines, parameterized by args)
          -> LLK pack functions (output to CB)
```

The defines mechanism is what makes a single kernel source file (like `eltwise_binary_kernel.cpp`) serve dozens of different operations. The compile-time args define the tiling structure. The runtime args handle the per-invocation data. Together, they form a flexible yet efficient configuration system that maps high-level tensor operations down to LLK hardware primitives.

---

**Next:** [Chapter 7 -- Hardware Target Selection](../ch7_hardware_target_selection/index.md)
