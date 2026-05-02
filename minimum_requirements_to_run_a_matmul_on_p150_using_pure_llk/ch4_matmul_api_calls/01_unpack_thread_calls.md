# Chapter 4 -- Minimal LLK API Call Sequences for MatMul

## 4.1 Unpack Thread (TRISC0) API Sequence

The unpack thread runs on TRISC0 and is responsible for transferring tile data from L1 memory into the SrcA and SrcB source registers, where the FPU can consume it. This section presents the complete, ordered sequence of LLK API calls required on TRISC0 for a tiled matmul, with parameter-by-parameter explanation drawn from the Blackhole source.

> **Operand mapping reminder** (see Chapter 2): In LLK matmul, operands are cross-mapped:
> - **in0 / inA** (the left matrix) loads into **SrcB** (SEC1)
> - **in1 / inB** (the right matrix) loads into **SrcA** (SEC0)
>
> This reversal exists because the FPU's MVMUL instruction computes $D[8,16] = B[8,16] \times A[16,16]$, where SrcB provides the 8-row slice and SrcA provides the full 16x16 face. The "A matrix" in mathematical terms maps to the register that provides 16x16 faces, which is SrcB in hardware terms.

### 4.1.1 The Complete TRISC0 Kernel

The unpack thread's `run_kernel` as compiled under `#ifdef LLK_TRISC_UNPACK`:

```cpp
void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src,
        formats.unpack_A_dst, formats.unpack_B_dst,
        FACE_R_DIM, FACE_R_DIM,
        params.num_faces_A, params.num_faces_B,
        params.TILE_SIZE_UNPACK_A, params.TILE_SIZE_UNPACK_B);

    _llk_unpack_AB_matmul_init_<>(
        0, params.CT_DIM, params.RT_DIM, params.KT_DIM,
        FACE_R_DIM, FACE_R_DIM, 4, 4, false, false);

    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(params.buffer_A[0]),
            L1_ADDRESS(params.buffer_B[0]),
            j, j * params.CT_DIM,
            params.TILE_SIZE_UNPACK_A, params.TILE_SIZE_UNPACK_B,
            false, false,
            params.CT_DIM, params.RT_DIM, params.KT_DIM);
    }
}
```

The unpack thread makes exactly **three distinct API calls**: one hardware configuration, one matmul-specific init, and one per-K-iteration unpack loop. The following sections dissect each call.

---

### 4.1.2 Call 1: `_llk_unpack_hw_configure_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_unpack_common.h`:

```cpp
template <bool is_fp32_dest_acc_en, bool disable_src_zero_flag = false>
inline void _llk_unpack_hw_configure_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim,
    const std::uint32_t unpB_face_r_dim,
    const std::uint32_t unpA_num_faces,
    const std::uint32_t unpB_num_faces,
    const std::uint32_t unpA_tile_size = 0,
    const std::uint32_t unpB_tile_size = 0);
```

#### What the Test Passes

```cpp
_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
    formats.unpack_A_src,       // unpA_src_format
    formats.unpack_B_src,       // unpB_src_format
    formats.unpack_A_dst,       // unpA_dst_format
    formats.unpack_B_dst,       // unpB_dst_format
    FACE_R_DIM,                 // unpA_face_r_dim
    FACE_R_DIM,                 // unpB_face_r_dim
    params.num_faces_A,         // unpA_num_faces
    params.num_faces_B,         // unpB_num_faces
    params.TILE_SIZE_UNPACK_A,  // unpA_tile_size
    params.TILE_SIZE_UNPACK_B); // unpB_tile_size
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `is_fp32_dest_acc_en` | `bool` (template) | `is_fp32_dest_acc_en` (compile-time define) | When `true`, the DEST accumulator uses FP32 precision; unpackers may output TF32 to SrcA/SrcB. When `false`, DEST uses 16-bit accumulation. Sets `ALU_ACC_CTRL_Fp32_enabled` and `ALU_ACC_CTRL_SFPU_Fp32_enabled`. |
| `disable_src_zero_flag` | `bool` (template) | `false` (default) | When `true`, disables zero-flag detection on source registers. Required for UInt16 destination format; otherwise leave at default. |
| `unpA_src_format` | `uint32_t` | `formats.unpack_A_src` | Data format of inA (left matrix) tiles as stored in L1. Encodes the `InDataFormat` field in the THCON_SEC0 tile descriptor. |
| `unpB_src_format` | `uint32_t` | `formats.unpack_B_src` | Data format of inB (right matrix) tiles as stored in L1. Encodes the `InDataFormat` field in the THCON_SEC1 tile descriptor. |
| `unpA_dst_format` | `uint32_t` | `formats.unpack_A_dst` | Target register format for unpacker A output (written to SrcA/SrcB registers). Encodes `OutDataFormat` in THCON_SEC0 config. Must satisfy the format-conversion rules (see Chapter 2). |
| `unpB_dst_format` | `uint32_t` | `formats.unpack_B_dst` | Target register format for unpacker B output. Encodes `OutDataFormat` in THCON_SEC1 config. |
| `unpA_face_r_dim` | `uint32_t` | `FACE_R_DIM` (= 16) | Number of rows per face for unpacker A. Programs the per-context `Tile_x_dim` field and address counter endpoint. Standard 32x32 tiles use 16. |
| `unpB_face_r_dim` | `uint32_t` | `FACE_R_DIM` (= 16) | Number of rows per face for unpacker B. Programs the `x_dim` field in the THCON_SEC1 tile descriptor. |
| `unpA_num_faces` | `uint32_t` | `params.num_faces_A` | Number of faces in an inA tile (1, 2, or 4). Programs the `z_dim` field in the THCON_SEC0 tile descriptor. Standard 32x32 tiles have 4 faces. |
| `unpB_num_faces` | `uint32_t` | `params.num_faces_B` | Number of faces in an inB tile (1, 2, or 4). Programs the `z_dim` field in the THCON_SEC1 tile descriptor. |
| `unpA_tile_size` | `uint32_t` | `params.TILE_SIZE_UNPACK_A` | Byte size of one inA tile in L1 (including any header). Stored in `p_gpr_unpack::TILE_SIZE_A` GPR for runtime address arithmetic. |
| `unpB_tile_size` | `uint32_t` | `params.TILE_SIZE_UNPACK_B` | Byte size of one inB tile in L1. Stored in `p_gpr_unpack::TILE_SIZE_B` GPR. |

Note: The parameter order groups by role -- all source formats first (`unpA_src_format`, `unpB_src_format`), then all destination formats (`unpA_dst_format`, `unpB_dst_format`), then face geometry, then tile sizes. This grouping-by-role matches the actual Blackhole source header.

#### What It Does Internally

This function delegates to `configure_unpack_AB<is_fp32_dest_acc_en, ...>()` in `cunpack_common.h`, then stores tile sizes into GPRs. The hardware configuration proceeds through the following stages:

**Stage 1 -- Wait for idle and reset address counters.**
The function calls `wait_for_idle()` to spin until the `UNPACK_SYNC` semaphore reaches 0, ensuring no in-flight unpack operations. Then `unpacker_addr_counter_init()` zeroes all XY and ZW address counters for both unpackers:

```cpp
TTI_SETADCXY(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1011);
TTI_SETADCZW(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);
```

**Stage 2 -- Program address stride registers.**
The function computes the channel-1 (destination-side) X stride based on the output data format width:
- Float32: stride = 4
- Float16: stride = 2
- All others: stride = 1

The Z stride (face stride in the destination register) is computed as $\text{FACE\_C\_DIM} \times \text{FACE\_R\_DIM} \times \text{x\_stride}$. These strides are written to `UNP0_ADDR_CTRL_ZW_REG_1` and `UNP1_ADDR_CTRL_ZW_REG_1` configuration registers.

**Stage 3 -- Program ALU format registers.**
The function acquires the `REG_RMW` mutex (since these registers are shared between threads), then writes into `ALU_FORMAT_SPEC_REG`. It sets the `SrcAUnsigned` and `SrcBUnsigned` flags if the respective input format is `UInt8`. It also configures stochastic rounding mode and the zero-flag disable bit. Then releases the mutex.

**Stage 4 -- Program tile descriptors.**
For each unpacker section, an `unpack_tile_descriptor_t` struct is constructed and written to `THCON_SEC0_REG0` and `THCON_SEC1_REG0`. The tile descriptor contains:
- `in_data_format`: the L1 source format (masked to low nibble)
- `uncompressed`: set to 1 (tiles are raw, not compressed)
- `x_dim`: set to `face_r_dim * FACE_C_DIM` for unpacker B; overridden per-context for unpacker A
- `y_dim`: set to 1
- `z_dim`: set to `num_faces` (number of 16x16 faces in the tile)

**Stage 5 -- Program unpack configuration registers.**
An `unpack_config_t` struct is written to `THCON_SEC0_REG2` and `THCON_SEC1_REG2` containing:
- `out_data_format`: the register output format
- `throttle_mode`: set to 2
- `context_count`: set to 0
- `haloize_mode`: controls XY transpose in the unpacker (set to 0 for matmul initially)
- `uncompress_cntx0_3` and `uncompress_cntx4_7`: set to 0xF (all contexts uncompressed)

**Stage 6 -- Program address counter endpoints and face dimensions.**
The X-end values for both unpackers are set via `TT_SETADCXX`. For unpacker A, the per-context face dimension is programmed into `THCON_SEC0_REG5_Tile_x_dim_cntx0`. Several precomputed face dimension values are stored into GPRs (`FACE_DIM_16x16`, `FACE_DIM_8x16`, etc.) for potential runtime use.

**Stage 7 -- Store tile sizes in GPRs.**
After `configure_unpack_AB` returns, the tile sizes are stored in dedicated GPRs for use during address arithmetic in the inner loop:

```cpp
TT_SETDMAREG(0, LOWER_HALFWORD(unpA_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
TT_SETDMAREG(0, LOWER_HALFWORD(unpB_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
```

**Stage 8 -- Reset config context.**
The function ends by calling `reset_config_context()`, which sets `unp_cfg_context = 0` and writes `0x0000` to `UNPACK_MISC_CFG_CfgContextOffset_0`, establishing context 0 as the starting configuration context.

#### Classification

**One-time call.** Programs format registers, tile descriptors, ALU format registers, strides, and GPR tile sizes. Unchanged across iterations.

---

### 4.1.3 Call 2: `_llk_unpack_AB_matmul_init_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`:

```cpp
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_init_(
    const std::uint32_t transpose       = 0,
    const std::uint32_t ct_dim          = 1,
    const std::uint32_t rt_dim          = 1,
    const std::uint32_t kt_dim          = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4,
    const bool unpA_partial_face        = false,
    const bool unpB_partial_face        = false);
```

#### What the Test Passes

```cpp
_llk_unpack_AB_matmul_init_<>(
    0,                // transpose
    params.CT_DIM,    // ct_dim
    params.RT_DIM,    // rt_dim
    params.KT_DIM,    // kt_dim
    FACE_R_DIM,       // unpA_face_r_dim
    FACE_R_DIM,       // unpB_face_r_dim
    4,                // unpA_num_faces
    4,                // unpB_num_faces
    false,            // unpA_partial_face
    false);           // unpB_partial_face
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `kernel_broadcast_a` | `uint32_t` (template) | `0` (default) | If non-zero, operand A addresses wrap modulo this value, enabling broadcast of fewer tiles across the K-dimension. 0 = no broadcast. |
| `kernel_broadcast_b` | `uint32_t` (template) | `0` (default) | If non-zero, operand B addresses wrap modulo this value. 0 = no broadcast. |
| `transpose` | `uint32_t` | `0` | Controls within-face XY transpose on the unpacker (`THCON_SEC0_REG2_Haloize_mode`). 0 = no transpose; 1 = enable face-level transpose. |
| `ct_dim` | `uint32_t` | `params.CT_DIM` | Column tile dimension of the output block. Together with `rt_dim`, determines `reuse_a` (if $ct\_dim \geq rt\_dim$). Controls MOP replay structure. |
| `rt_dim` | `uint32_t` | `params.RT_DIM` | Row tile dimension of the output block. |
| `kt_dim` | `uint32_t` | `params.KT_DIM` | Inner (reduction) tile dimension. Stored in `p_gpr_unpack::KT_DIM` GPR. Used by the unpack loop to compute tile address offsets. |
| `unpA_face_r_dim` | `uint32_t` | `FACE_R_DIM` (= 16) | Rows per face for unpacker A. Programs address counter endpoints when `unpA_partial_face` is `true`. |
| `unpB_face_r_dim` | `uint32_t` | `FACE_R_DIM` (= 16) | Rows per face for unpacker B. |
| `unpA_num_faces` | `uint32_t` | `4` | Number of faces per tile for inA. Programs the `x_end` address counter as $\text{num\_faces} \times \text{face\_r\_dim} \times \text{FACE\_C\_DIM} - 1$. Must be 1, 2, or 4. |
| `unpB_num_faces` | `uint32_t` | `4` | Number of faces per tile for inB. |
| `unpA_partial_face` | `bool` | `false` | When `true`, the unpacker processes faces one at a time with reduced `x_end`, enabling sub-face tile shapes. |
| `unpB_partial_face` | `bool` | `false` | Same as above but for unpacker B. |

#### What It Does Internally

This function serves two purposes: (1) configure the unpacker's operating mode for matmul, and (2) program the replay buffer (MOP) that will be executed on each unpack iteration.

**Stage 1 -- Configure haloize mode.**
The `haloize_mode` bit in `THCON_SEC0_REG2` controls within-face transpose in the unpacker:

```cpp
cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose);
```

**Stage 2 -- Reset Z and W address counters.**
Both unpackers' Z and W counters are zeroed to start from the first face:

```cpp
TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
```

**Stage 3 -- Program X endpoint for full-tile unpacking.**
For standard (non-partial-face) matmul with 4 faces and `FACE_R_DIM=16`:

$$\text{unpA\_x\_end} = 4 \times 16 \times 16 - 1 = 1023$$

This value is written via `TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0)`. Similarly for unpacker B.

**Stage 4 -- Store kt_dim in GPR.**
The K-dimension tile count is stored in a GPR for use during address scaling:

```cpp
TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(p_gpr_unpack::KT_DIM));
```

**Stage 5 -- Program the replay buffer (MOP).**
The function calls `_llk_unpack_AB_matmul_mop_config_()`, which programs a hardware replay buffer that the unpacker will execute on each iteration. The replay buffer content depends on the `reuse_a` strategy.

##### The reuse_a vs. reuse_b Strategy

The decision is:

```cpp
const bool reuse_a = ct_dim >= rt_dim;
```

- **reuse_a = true** ($ct\_dim \geq rt\_dim$): The A-matrix tile (in SrcB) is held while multiple B-matrix tiles (in SrcA) are streamed through. The replay buffer programs SrcA address advancement.
- **reuse_a = false** ($ct\_dim < rt\_dim$): The roles are reversed: operand B is held, and the MOP replays operand A unpack `rt_dim` times.

##### Replay Buffer Structure (reuse_a case, no partial face)

The replay buffer length is 12 instructions (6 per context). For each double-buffered context, the buffer contains:

1. `TTI_UNPACR(SrcA, ...)` -- Trigger unpack of the non-reused operand (SrcA for reuse_a) with `OvrdThreadId=1` and `Dvalid=1`
2. `TTI_RDCFG(TMP0, THCON_SEC0_REG3_Base_address_ADDR32)` -- Read the current SrcA base address from the config register
3. `TTI_ADDDMAREG(0, TMP0, TMP0, TILE_SIZE_A)` -- Add one tile size to advance to the next tile
4. `TTI_STALLWAIT(STALL_CFG, THCON)` -- Wait for config pipeline
5. `TTI_WRCFG(TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32)` -- Write the advanced address back
6. `TTI_NOP` -- Ensure WRCFG completes (2-cycle latency)

The second half (instructions 7-12) is identical but operates on the context-1 address register (`THCON_SEC0_REG3_Base_cntx1_address`).

**Stage 6 -- Program the unpack MOP template.**
After the replay buffer is loaded, a `ckernel_unpack_template` is constructed that maps context 0 to the first half and context 1 to the second half:

```cpp
ckernel_unpack_template tmp = ckernel_unpack_template(
    false,                                           // src B
    false,                                           // halo
    lltt::replay_insn(0, replay_buf_run_len),        // context 0 replay
    0, 0, 0,
    lltt::replay_insn(replay_buf_run_len, replay_buf_run_len), // context 1 replay
    0, 0);
tmp.program();
```

#### Classification

**One-time call.** Programs the MOP replay buffer, address counters, and kt_dim GPR. Unchanged across iterations.

---

### 4.1.4 Call 3 (per-K-iteration): `_llk_unpack_AB_matmul_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`:

```cpp
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
inline void _llk_unpack_AB_matmul_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const bool unpA_partial_face = false,
    const bool unpB_partial_face = false,
    std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim   = 1,
    const std::uint32_t kt_dim   = 1);
```

#### What the Test Passes

```cpp
for (std::uint32_t j = 0; j < params.KT_DIM; j++)
{
    _llk_unpack_AB_matmul_<>(
        L1_ADDRESS(params.buffer_A[0]),   // base_address_a
        L1_ADDRESS(params.buffer_B[0]),   // base_address_b
        j,                                // tile_index_a
        j * params.CT_DIM,                // tile_index_b
        params.TILE_SIZE_UNPACK_A,        // tile_size_a
        params.TILE_SIZE_UNPACK_B,        // tile_size_b
        false,                            // unpA_partial_face
        false,                            // unpB_partial_face
        params.CT_DIM,                    // ct_dim
        params.RT_DIM,                    // rt_dim
        params.KT_DIM);                   // kt_dim
}
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `kernel_broadcast_a` | `uint32_t` (template) | `0` (default) | Broadcast modulus for operand A. When > 0, tile address wraps: `(tile_index_a + offset) % kernel_broadcast_a`. |
| `kernel_broadcast_b` | `uint32_t` (template) | `0` (default) | Broadcast modulus for operand B. |
| `base_address_a` | `uint32_t` | `L1_ADDRESS(params.buffer_A[0])` | L1 byte address of the start of the A (left matrix) tile buffer. `L1_ADDRESS()` converts a buffer pointer to a 16-byte-aligned L1 address (see Chapter 2). |
| `base_address_b` | `uint32_t` | `L1_ADDRESS(params.buffer_B[0])` | L1 byte address of the start of the B (right matrix) tile buffer. |
| `tile_index_a` | `uint32_t` | `j` (loop variable, 0 to KT_DIM-1) | Index into the A tile array for the current K-step. Used to compute byte offset: $\text{offset}\_a = \text{tile\_size}\_a \times \text{tile\_index}\_a$. With `reuse_a`, factors in the t-loop: $\text{tile\_size}\_a \times (\text{tile\_index}\_a + t \times kt\_dim)$. |
| `tile_index_b` | `uint32_t` | `j * params.CT_DIM` | Index into the B tile array. B is laid out with CT_DIM tiles per K-row, so at K-step `j`, the first B tile is at index $j \times CT\_DIM$. |
| `tile_size_a` | `uint32_t` | `params.TILE_SIZE_UNPACK_A` | Byte size of one A tile in L1. Used for address arithmetic. |
| `tile_size_b` | `uint32_t` | `params.TILE_SIZE_UNPACK_B` | Byte size of one B tile in L1. |
| `unpA_partial_face` | `bool` | `false` | When `true`, uses face-by-face unpack instructions instead of full-tile unpack. |
| `unpB_partial_face` | `bool` | `false` | Same for operand B. |
| `ct_dim` | `uint32_t` | `params.CT_DIM` | Column tile dimension. Determines `reuse_a` and the outer t-loop bound. |
| `rt_dim` | `uint32_t` | `params.RT_DIM` | Row tile dimension. |
| `kt_dim` | `uint32_t` | `params.KT_DIM` | Inner dimension. Used for address stride computation when `reuse_a` is true: each A row-block is `kt_dim` tiles apart. |

#### What It Does Internally

This function executes the actual data movement from L1 to source registers. It iterates over `t_dim` sub-iterations (where `t_dim = rt_dim` when reusing A, or `t_dim = ct_dim` when reusing B). Within each sub-iteration, the following double-buffered context protocol executes:

##### Double-Buffered Context Protocol

The unpack thread maintains a double-buffered context via the global variable `unp_cfg_context`, which alternates between 0 and 1. This allows the unpacker hardware to process one tile pair while the RISC-V core configures the next.

```
for t in 0..t_dim-1:
    1. wait_for_next_context(2)
    2. _llk_unpack_configure_addresses_(address_b, address_a, cfg)
    3. semaphore_post(semaphore::UNPACK_SYNC)
    4. TTI_STALLWAIT -- stall unpacker until CFG writes complete
    5. UNPACR -- trigger unpack of the reused operand
    6. TT_MOP(0, rut_dim-1, context_select) -- run the replay buffer
    7. t6_semaphore_get(semaphore::UNPACK_SYNC)
    8. switch_config_context(unp_cfg_context)
```

**Step-by-step:**

1. **`wait_for_next_context(2)`**: Spins until the `UNPACK_SYNC` semaphore count drops below 2, meaning at least one context slot is free.

2. **`_llk_unpack_configure_addresses_`**: Writes L1 base addresses into THCON configuration registers. Note the cross-mapping swap: `address_b` (operand B / in1) goes to SEC0 (SrcA), and `address_a` (operand A / in0) goes to SEC1 (SrcB). The register chosen depends on `unp_cfg_context`:
   - Context 0: writes to `THCON_SEC0_REG3_Base_address` and `THCON_SEC1_REG3_Base_address`
   - Context 1: writes to `THCON_SEC0_REG3_Base_cntx1_address` and `THCON_SEC1_REG3_Base_cntx1_address`

3. **`semaphore_post(UNPACK_SYNC)`**: Increments the semaphore to signal that a context has been claimed.

4. **`TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG)`**: The unpacker hardware stalls until the TRISC configuration writes have propagated.

5. **UNPACR**: Issues an unpack command for the reused operand (SrcB if `reuse_a`, SrcA if `reuse_b`). Key fields:
   - `OvrdThreadId = 1`: use thread-overridden face dimension
   - `Dvalid = 1`: set data-valid flag to signal the math thread
   - Last field `= 1`: flush the search cache

   ```cpp
   TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
   ```

6. **`TT_MOP(0, rut_dim-1, context_select)`**: Runs the MOP with replay buffer 0 for `rut_dim` iterations. The third argument selects which replay buffer half: `0x00` for context 0, `0xFF` for context 1. For `reuse_a` with `ct_dim = 4`, the MOP executes 4 times, each time triggering `UNPACR` on SrcA to load one B-tile and advancing the base address.

7. **`t6_semaphore_get(UNPACK_SYNC)`**: Decrements the semaphore, freeing the context slot.

8. **`switch_config_context(unp_cfg_context)`**: Flips `unp_cfg_context` between 0 and 1, and writes the corresponding context offset to `UNPACK_MISC_CFG_CfgContextOffset_0` (0x0000 for context 0, 0x0101 for context 1).

##### Address Computation

The L1 address for each tile is computed as:

$$\text{address} = \text{base\_address} + \text{tile\_size} \times (\text{tile\_index} + \text{offset})$$

Where the offset depends on the reuse strategy and the current iteration indices. With `reuse_a`, operand A advances by `t * kt_dim` rows per sub-iteration while operand B's address is incremented by the MOP replay buffer.

##### Tile Indexing in the K-Loop

The caller passes tile indices that advance through the K dimension:

- `tile_index_a = j` -- A advances one tile per K step along A's columns.
- `tile_index_b = j * CT_DIM` -- B advances by CT_DIM tiles per K step, because B's tile layout has CT_DIM columns per K-row.

For a matmul $C[R,C] = A[R,K] \times B[K,C]$:
- A is laid out as RT_DIM rows of KT_DIM tiles each.
- B is laid out as KT_DIM rows of CT_DIM tiles each.
- C is laid out as RT_DIM rows of CT_DIM tiles each.

#### Classification

**Per-K-iteration call.** Invoked $KT\_DIM$ times. Each call internally iterates $t\_dim = \min(RT\_DIM, CT\_DIM)$ times, performing one double-buffered context-switched unpack per iteration.

---

### 4.1.5 Effective Iteration Count

For a matmul with $RT\_DIM \times CT\_DIM \times KT\_DIM$, the total hardware-level operations on the unpack thread are:

- **Top-level API calls:** $2 + KT\_DIM$ (2 init + $KT\_DIM$ loop)
- **Context-switch cycles per call:** $t\_dim = \min(RT\_DIM, CT\_DIM)$
- **MOP replay iterations per cycle:** $rut\_dim = \max(RT\_DIM, CT\_DIM)$
- **Total unpack operations:** $KT\_DIM \times \min(RT\_DIM, CT\_DIM) \times \max(RT\_DIM, CT\_DIM) = KT\_DIM \times RT\_DIM \times CT\_DIM$

This is exactly the total number of tile-pair unpacks needed for a full matmul.

**Source files:**
- `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`
- `tt_llk_blackhole/llk_lib/llk_unpack_common.h`
- `tt_llk_blackhole/common/inc/cunpack_common.h`
- `tests/sources/matmul_test.cpp` (UNPACK block)

---

**Next:** [`02_math_thread_calls.md`](./02_math_thread_calls.md)
