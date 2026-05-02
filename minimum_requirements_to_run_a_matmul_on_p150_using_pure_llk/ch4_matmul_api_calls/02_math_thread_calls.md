## 4.2 Math Thread (TRISC1) API Sequence

The math thread runs on TRISC1 and drives the FPU to execute the actual matrix-multiply-accumulate operations. It reads from SrcA and SrcB source registers (filled by the unpack thread) and writes results to the DEST accumulator register. This section documents the complete, ordered sequence of six LLK API calls required on TRISC1.

### 4.2.1 The Complete TRISC1 Kernel

```cpp
void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;

    _llk_math_matmul_init_<MATH_FIDELITY>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM,
        false, 0, params.CT_DIM, params.RT_DIM);

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    LLK_ASSERT(
        (get_dest_max_matmul_tiles(0, params.CT_DIM, params.RT_DIM) <
         get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "Block tile index exceeds maximum destination tiles for matmul");

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_math_matmul_<MATH_FIDELITY>(0, params.CT_DIM, params.RT_DIM);
    }

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}
```

Note the `LLK_ASSERT` between the init block and the compute block. This asserts that the output tile block ($RT\_DIM \times CT\_DIM$ tiles) fits in one DEST half. If the block exceeds capacity, the computation must be split into multiple passes (not shown in the minimal kernel).

---

### 4.2.2 Call 1: `_llk_math_matmul_init_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_math_matmul.h`:

```cpp
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_init_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t transpose      = 0,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t rt_dim         = 1);
```

#### What the Test Passes

```cpp
_llk_math_matmul_init_<MATH_FIDELITY>(
    TILE_R_DIM, TILE_C_DIM,   // in0 dimensions (32x32)
    TILE_R_DIM, TILE_C_DIM,   // in1 dimensions (32x32)
    false,                      // partial_face
    0,                          // transpose
    params.CT_DIM,
    params.RT_DIM);
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `math_fidelity` | `MathFidelity` (template) | `MATH_FIDELITY` (compile-time define) | Controls FPU fidelity phases per tile multiply. `LoFi` = 1 phase (fastest). `HiFi2` = 2 phases. `HiFi3` = 3 phases. `HiFi4` = 4 phases (highest precision). Determines whether `is_high_fidelity()` returns true and sets the MOP inner loop count. |
| `THROTTLE_LEVEL` | `int` (template) | `0` (default) | When non-zero (1--5), inserts NOP instructions between MVMULs to reduce compute throughput. Level 1 = 73% of max; Level 5 = 33% of max. 0 = no throttling. |
| `in0_tile_r_dim` | `uint32_t` | `TILE_R_DIM` (= 32) | Row dimension of in0 (left matrix) tiles. Used to detect sub-tile shapes (16x32, 32x16, etc.). Standard matmul uses 32. |
| `in0_tile_c_dim` | `uint32_t` | `TILE_C_DIM` (= 32) | Column dimension of in0 tiles. |
| `in1_tile_r_dim` | `uint32_t` | `TILE_R_DIM` (= 32) | Row dimension of in1 (right matrix) tiles. |
| `in1_tile_c_dim` | `uint32_t` | `TILE_C_DIM` (= 32) | Column dimension of in1 tiles. |
| `partial_face` | `bool` | `false` | Enables partial-face math mode for sub-face tile dimensions. Changes the MVMUL instruction sequence and address modifier selection. |
| `transpose` | `uint32_t` | `0` | When non-zero, selects the transposed address-modifier configuration where SrcA faces are read with doubled increments. |
| `ct_dim` | `uint32_t` | `params.CT_DIM` | Column tile dimension. Together with `rt_dim`, determines `reuse_a` and the MOP end-operation. |
| `rt_dim` | `uint32_t` | `params.RT_DIM` | Row tile dimension. |

#### What It Does Internally

This function performs three major tasks: configure address modifier registers, program the math MOP (replay buffer), and reset the FPU counters.

##### Stage 1: Address Modifier Configuration

`matmul_configure_addrmod<math_fidelity, THROTTLE_LEVEL>(...)` sets up address modifier registers that control how the FPU traverses SrcA, SrcB, and Dest during each MVMUL instruction. The fundamental computation unit is a face-level multiply:

$$D[8,16] = B[8,16] \times A[16,16]$$

Each MVMUL instruction multiplies 8 rows of SrcB against the full 16x16 face of SrcA, producing 8 rows of output in Dest. To complete a full 16x16 face output, 2 MVMUL instructions are needed ($16 / 8 = 2$). A full 32x32 tile requires traversing all 4 face pairs.

For standard 32x32 tiles without transpose, the address modifiers are:

**ADDR_MOD_0** -- Inner-loop increment. After each MVMUL, SrcB and Dest advance by 8 rows; SrcA stays at the same face:

```cpp
addr_mod_t {
    .srca = {.incr = 0, .clr = 0, .cr = 0},    // SrcA stays at same face
    .srcb = {.incr = 8, .clr = 0, .cr = 0},     // SrcB advances 8 rows
    .dest = {.incr = 8, .clr = 0, .cr = 0},     // Dest advances 8 rows
}.set(ADDR_MOD_0);
```

**ADDR_MOD_1** -- Face transition within half-tile. After processing one SrcA face (two MVMUL iterations covering 16 SrcB rows), SrcA advances to the next face and SrcB resets:

```cpp
addr_mod_t {
    .srca = {.incr = 16, .clr = 0, .cr = 0},   // SrcA to next face (+16 rows)
    .srcb = {.incr = 0, .clr = 0, .cr = 1},     // SrcB counter-reset
    .dest = {.incr = 8, .clr = 0, .cr = 0},     // Dest continues advancing
}.set(ADDR_MOD_1);
```

**ADDR_MOD_2** -- Half-tile boundary. Crossing from face pair (f0,f1) to face pair (f2,f3):

```cpp
addr_mod_t {
    .srca = {.incr = 0, .clr = 0, .cr = 1},     // SrcA counter-reset to face 0
    .srcb = {.incr = 32, .clr = 0, .cr = 1},     // SrcB jump to second half (+32)
    .dest = {.incr = 8, .clr = 0, .cr = 0},      // Dest continues
}.set(ADDR_MOD_2);
```

**ADDR_MOD_4** -- Tile completion. After all four faces, prepare for the next tile:

```cpp
addr_mod_t {
    .srca = {.incr = 32, .clr = 0, .cr = 1},    // SrcA wrap-around
    .srcb = {.incr = 48, .clr = 0, .cr = 1},     // SrcB wrap
    .dest = {.incr = 0, .clr = 0, .cr = 1},      // Dest counter-reset
}.set(ADDR_MOD_4);
```

**ADDR_MOD_5** -- Phase/tile end. Resets all counters and, for high-fidelity modes, increments the fidelity phase counter:

```cpp
addr_mod_t {
    .srca     = {.incr = 0, .clr = 1, .cr = 1},
    .srcb     = {.incr = 0, .clr = 1, .cr = 1},
    .dest     = {.incr = 0, .clr = 1, .cr = 1},
    .fidelity = {.incr = fidelity_increment, .clr = 0},
}.set(ADDR_MOD_5);
```

Where `fidelity_increment = 1` for HiFi modes and `0` for LoFi.

**Address modifier field legend:**
- `incr`: the number of rows to add to the pointer after the MVMUL instruction completes.
- `clr`: when 1, clear (zero) the counter before applying the increment.
- `cr`: counter-reset. When 1, reset the counter to the value set by the last clear operation, creating "zigzag" traversal patterns needed to correctly pair faces between SrcA and SrcB.

##### Stage 2: MOP Programming

`matmul_configure_mop<math_fidelity>(ct_dim, rt_dim, ...)` programs the replay buffer starting at offset `ckernel::math::replay_buf_offset` (= 16, since positions 0-15 are reserved for SFPU). For standard 32x32 tiles, the replay buffer contains 16 MVMUL instructions:

```
Instruction 0:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_0, 0)  // B0*A0, srcb+=8, dest+=8
Instruction 1:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_1, 0)  // B0*A0, srca+=16, srcb cr, dest+=8
Instruction 2:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_0, 0)  // B0*A1, srcb+=8, dest+=8
Instruction 3:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_2, 0)  // B0*A1, srca cr, srcb=32, dest+=8
Instruction 4:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_0, 0)  // B2*A0, srcb+=8, dest+=8
Instruction 5:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_1, 0)  // B2*A0, srca+=16, srcb cr, dest+=8
Instruction 6:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_0, 0)  // B2*A1, srcb+=8, dest+=8
Instruction 7:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_4, 0)  // B2*A1, wrap for next half
Instruction 8:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_0, 0)  // B1*A2, srcb+=8, dest+=8
Instruction 9:  TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_1, 0)  // B1*A2, srca+=16, dest+=8
Instruction 10: TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_0, 0)  // B1*A3, srcb+=8, dest+=8
Instruction 11: TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_2, 0)  // B1*A3, srca cr, srcb=48
Instruction 12: TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_0, 0)  // B3*A2, srcb+=8, dest+=8
Instruction 13: TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_1, 0)  // B3*A2, srca+=16, dest+=8
Instruction 14: TTI_MVMUL(CLR_NONE, 0, ADDR_MOD_0, 0)  // B3*A3, srcb+=8, dest+=8
Instruction 15: TTI_MVMUL(CLR_A/CLR_B, 0, ADDR_MOD_5, 0) // reset all, clear reused src
```

The face multiplication pattern maps to a full 32x32 output tile through four quadrants. Each MVMUL computes $D[8,16] = B[8,16] \times A[16,16]$, accumulating 8 output rows. The 16 instructions cover all 4 output faces (4 MVMUL per face: 2 per SrcA face, 2 SrcA faces per output face).

The final instruction's `CLR_A` or `CLR_B` flag clears the cycled source register bank:
- `reuse_a` ($ct\_dim \geq rt\_dim$): `CLR_A` clears SrcA (which held the cycling B-tiles)
- `reuse_b` ($ct\_dim < rt\_dim$): `CLR_B` clears SrcB (which held the cycling A-tiles)

The MOP is then programmed as a `ckernel_template` with:

```cpp
constexpr std::uint32_t inner_loops = high_fidelity ? to_underlying(math_fidelity) : 1;
ckernel_template tmp(1, inner_loops, lltt::replay_insn(replay_buf_offset, replay_buf_len));
if constexpr (high_fidelity)
{
    tmp.set_end_op(TT_OP_SETRWC(reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B,
                                  0, 0, 0, 0, p_setrwc::SET_ABD_F));
}
tmp.program();
```

For high-fidelity modes, the MOP outer loop runs `math_fidelity` phases (2/3/4). ADDR_MOD_5's fidelity increment field advances the phase counter between passes. A `set_end_op` clears the appropriate source register after all phases.

##### MathFidelity Phases

| MathFidelity | Phases | Inner Loops | Description |
|---|---|---|---|
| `LoFi` | 1 | 1 | Single pass; fastest, lowest precision |
| `HiFi2` | 2 | 2 | Two accumulation passes |
| `HiFi3` | 3 | 3 | Three accumulation passes |
| `HiFi4` | 4 | 4 | Four accumulation passes; highest precision |

For LoFi, the source register clear is embedded in the replay buffer's final MVMUL instruction. For HiFi modes, the clear is deferred to the `ckernel_template`'s `set_end_op`.

##### Stage 3: Reset Counters

After MOP programming, all FPU address counters are reset:

```cpp
math::reset_counters(p_setrwc::SET_ABD_F);
// Emits: TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
```

This clears SrcA, SrcB, Dest, and Fidelity counters.

#### Classification

**One-time call.** Programs address modifiers and the FPU MOP. Unchanged across iterations.

---

### 4.2.3 Call 2: `_llk_math_pack_sync_init_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_math_common.h`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_pack_sync_init_();
```

#### What the Test Passes

```cpp
_llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `Dst` | `DstSync` (template) | `DstSync::SyncHalf` | Synchronization mode. `SyncHalf` = double-buffered, math and pack alternate between two halves of DEST. `SyncFull` = single-buffered. SyncHalf is the standard choice for matmul. |
| `is_fp32_dest_acc_en` | `bool` (template) | `is_fp32_dest_acc_en` | FP32 accumulation mode. Affects DEST section base address and available tile capacity. |

#### What It Does Internally

1. **Full synchronization barrier.** `tensix_sync()` issues a `TTI_STALLWAIT` that blocks until all pending Tensix operations complete.

2. **Drain any pending pack operations.** `while (semaphore_read(semaphore::MATH_PACK) > 0) {}` spins until the `MATH_PACK` semaphore reaches zero.

3. **Initialize the semaphore.** For `SyncHalf`: `TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1)` -- max value 2, initial value 0. This allows double-buffered operation where math can be one half ahead of pack.

4. **Reset dest offset and set section base.**
   ```cpp
   reset_dest_offset_id();
   set_dest_section_base<StartZero>();
   // Emits: TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
   ```
   This sets the math thread's Dest write pointer to the beginning of the first half (offset 0).

#### Classification

**One-time call.** Initializes the MATH_PACK semaphore and DEST section base.

---

### 4.2.4 Call 3: `_llk_math_hw_configure_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_math_common.h`:

```cpp
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(
    const std::uint32_t srca_data_format,
    const std::uint32_t srcb_data_format);
```

#### What the Test Passes

```cpp
_llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
```

Note: the same format code is passed for both SrcA and SrcB. On Blackhole, the ALU format is inferred from the source register contents, so these values primarily affect INT8 detection and FP32 accumulation control.

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `is_fp32_dest_acc_en` | `bool` (template) | `is_fp32_dest_acc_en` | When `true`, sets `ALU_ACC_CTRL_Fp32_enabled` and `ALU_ACC_CTRL_SFPU_Fp32_enabled`, enabling 32-bit accumulation. When `false`, DEST uses 16-bit accumulation. |
| `srca_data_format` | `uint32_t` | `formats.math` | Data format code for SrcA. Used to detect Int8/Int32 formats that enable `ALU_ACC_CTRL_INT8_math_enabled`. |
| `srcb_data_format` | `uint32_t` | `formats.math` | Data format code for SrcB. Same INT8 detection logic. |

#### What It Does Internally

1. **Configure ZEROACC mode.** `cfg_reg_rmw_tensix<DEST_ACCESS_CFG_zeroacc_absolute_tile_mode_RMW>(0)` -- non-legacy mode where ZEROACC auto-detects which Dest bank to clear.

2. **Stall until math pipeline is idle.** `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH)`.

3. **Configure INT8 math mode.** Checks if either format is `Int8` or `Int32` and sets `ALU_ACC_CTRL_INT8_math_enabled` accordingly:
   ```cpp
   std::uint32_t int8_math_enabled =
       (masked_data_format(srca_data_format) == to_underlying(DataFormat::Int8)) ||
       (masked_data_format(srcb_data_format) == to_underlying(DataFormat::Int8)) ||
       (srca_data_format == to_underlying(DataFormat::Int32)) ||
       (srcb_data_format == to_underlying(DataFormat::Int32));
   cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(int8_math_enabled);
   ```

4. **Configure FP32 accumulation.**
   ```cpp
   cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
   cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(is_fp32_dest_acc_en);
   ```

5. **Apply hardware bug workarounds.** If INT8 math is enabled or UInt16 is used with FP32 dest, disables a debug feature:
   ```cpp
   _llk_math_dbg_feature_disable_();
   // Writes: reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1 << 11);
   ```

#### Classification

**One-time call.** Configures FP32 accumulation, INT8 math mode, and ZEROACC addressing.

---

### 4.2.5 Call 4: `_llk_math_wait_for_dest_available_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_math_common.h`:

```cpp
template <DstSync Dst>
inline void _llk_math_wait_for_dest_available_();
```

#### What the Test Passes

```cpp
_llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `Dst` | `DstSync` (template) | `DstSync::SyncHalf` | Must match the sync mode used in `_llk_math_pack_sync_init_`. Controls the semaphore stall condition. |

#### What It Does Internally

Calls `math_dest_wait()`, which issues:

```cpp
TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_MAX);
```

This stalls the math and SFPU pipelines until the `MATH_PACK` semaphore drops below its maximum value. For `SyncHalf` (max=2), this blocks when both halves of the Dest register are occupied. When the packer finishes one half and decrements the semaphore, the math thread unblocks.

#### Classification

**Per-dest-section call.** Called once before the K-loop in the single-block case. In a multi-block pipeline, called once per output block.

---

### 4.2.6 Call 5 (per-K-iteration): `_llk_math_matmul_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_math_matmul.h`:

```cpp
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_(
    std::uint32_t dst_index,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1);
```

#### What the Test Passes

```cpp
for (std::uint32_t j = 0; j < params.KT_DIM; j++)
{
    _llk_math_matmul_<MATH_FIDELITY>(0, params.CT_DIM, params.RT_DIM);
}
```

Note: `dst_index` is 0 for every K-step because matmul accumulates in-place. Each K-iteration adds to the existing DEST contents.

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `math_fidelity` | `MathFidelity` (template) | `MATH_FIDELITY` | Must match the fidelity used in `_llk_math_matmul_init_`. Determines `high_fidelity` flag. |
| `THROTTLE_LEVEL` | `int` (template) | `0` (default) | Must match `_llk_math_matmul_init_`. Controls the MOP variant used. |
| `dst_index` | `uint32_t` | `0` | Starting tile index in DEST. Each tile occupies $2^6 = 64$ rows (for `Tile32x32`). Actual DEST address: $\text{dst\_index} \ll 6 + \text{dest\_buffer\_base}$. |
| `ct_dim` | `uint32_t` | `params.CT_DIM` | Column tile dimension. |
| `rt_dim` | `uint32_t` | `params.RT_DIM` | Row tile dimension. |

#### What It Does Internally

The function contains a nested double loop covering all output tiles in the $rt\_dim \times ct\_dim$ block. The reuse strategy (see Section 4.1.3) determines which source register is held constant:

```
reuse_a = (ct_dim >= rt_dim)
t_dim   = reuse_a ? rt_dim : ct_dim
rut_dim = reuse_a ? ct_dim : rt_dim

for t in 0..t_dim-1:
    for rut in 0..rut_dim-1:
        1. set_dst_write_addr(dst_index + tile_offset)
        2. ckernel_template::run()
        3. if (rut == rut_dim - 1): clear opposite source register
```

**Step 1 -- Set DEST write address.** `math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(index)` computes the destination register address for this output tile. The tile indexing maps the 2D output grid into linear DEST space:
- `reuse_a`: row-major order `ct_dim * t + rut`
- `reuse_b`: column-major order `t + rut * ct_dim`

The tile index is shifted left by `DstTileSizeLog2[Tile32x32]` = 6 (each 32x32 tile occupies $2^6 = 64$ rows) and added to the current dest buffer base.

**Step 2 -- Execute the MOP.** `ckernel_template::run()` triggers the pre-programmed multiply sequence (16 MVMUL instructions for 32x32 tiles). For LoFi, one pass through the replay buffer. For HiFi, the MOP outer loop runs multiple passes with fidelity phase increments.

**Step 3 -- Clear source register at reuse boundary.** At the end of each reuse sweep (when `rut == rut_dim - 1`):

```cpp
if (rut == (rut_dim - 1))
{
    if (reuse_a)
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    else
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}
```

For `reuse_a`: SrcB (which holds the A-matrix tile) is cleared, allowing the unpacker to load a new A-tile for the next outer-loop iteration. SrcA (holding B-matrix tiles) was already cleared by the MOP's final MVMUL instruction on each `rut` iteration.

##### Accumulation Semantics

Across the `kt_dim` loop, each call to `_llk_math_matmul_` **accumulates** into the same DEST tiles. The DEST is only cleared (zeroed) after the pack thread reads it. After all $KT\_DIM$ iterations:

$$\text{Dest}[r][c] = \sum_{k=0}^{KT\_DIM-1} A[r][k] \times B[k][c]$$

#### Classification

**Per-K-iteration call.** Invoked $KT\_DIM$ times. Each call internally iterates over all $RT\_DIM \times CT\_DIM$ output tiles via the nested t/rut loops.

---

### 4.2.7 Call 6: `_llk_math_dest_section_done_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_math_common.h`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_math_dest_section_done_();
```

#### What the Test Passes

```cpp
_llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `Dst` | `DstSync` (template) | `DstSync::SyncHalf` | Must match the sync mode used throughout. |
| `is_fp32_dest_acc_en` | `bool` (template) | `is_fp32_dest_acc_en` | Passed through to DEST section flip logic. |

#### What It Does Internally

1. **Post to MATH_PACK semaphore.** `set_math_semaphores()` calls `t6_semaphore_post<p_stall::MATH | p_stall::WAIT_SFPU>(semaphore::MATH_PACK)`, incrementing the semaphore to tell the packer that data is available.

2. **Reset tile index.** `math_sync_tile_dst_index = 0`.

3. **Flip the DEST section.** `dest_section_flip()`:
   - Toggles `dest_offset_id` between 0 and 1.
   - Computes the new DEST buffer base address: 0 for the first half, `DEST_REGISTER_HALF_SIZE` for the second.
   - Stalls until the math pipeline is idle, then writes the new base:
     ```cpp
     TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
     TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);
     ```

After this call, math is pointing at the other DEST half and can begin computing the next batch while the packer reads from the just-completed half.

#### Classification

**Per-dest-section call.** Signals the packer and flips the DEST section.

---

### 4.2.8 The MVMUL Face Multiplication Pattern

Each MVMUL instruction computes:

$$D[8,16] = B[8,16] \times A[16,16]$$

A full 32x32 output tile requires 16 MVMUL instructions organized as 4 output faces, each requiring 4 MVMUL calls (2 per SrcA face, 2 SrcA faces per output face):

```
Output Face 0 (rows 0-15, cols 0-15):
    MVMUL: D[0:7, 0:15]   += B_face0[0:7, 0:15]  * A_face0[0:15, 0:15]
    MVMUL: D[8:15, 0:15]  += B_face0[8:15, 0:15]  * A_face0[0:15, 0:15]
    MVMUL: D[0:7, 0:15]   += B_face0[0:7, 0:15]  * A_face1[0:15, 0:15]
    MVMUL: D[8:15, 0:15]  += B_face0[8:15, 0:15]  * A_face1[0:15, 0:15]

Output Face 1 (rows 0-15, cols 16-31):
    ... same pattern with B_face2, A_face0, A_face1 ...

Output Face 2 (rows 16-31, cols 0-15):
    ... same pattern with B_face1, A_face2, A_face3 ...

Output Face 3 (rows 16-31, cols 16-31):
    ... same pattern with B_face3, A_face2, A_face3 ...
```

The address modifiers orchestrate this traversal automatically. The `cr` (counter-reset) fields create the "zigzag" pattern needed to correctly pair faces between the two source registers.

**Source files:**
- `tt_llk_blackhole/llk_lib/llk_math_matmul.h`
- `tt_llk_blackhole/llk_lib/llk_math_common.h`
- `tt_llk_blackhole/common/inc/cmath_common.h`
- `tests/sources/matmul_test.cpp` (MATH block)

---

**Next:** [`03_pack_thread_calls.md`](./03_pack_thread_calls.md)
