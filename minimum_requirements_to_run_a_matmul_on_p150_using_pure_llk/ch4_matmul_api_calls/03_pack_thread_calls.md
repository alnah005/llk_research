## 4.3 Pack Thread (TRISC2) API Sequence

The pack thread runs on TRISC2 and is responsible for reading computed results from the DEST accumulator register and writing them to L1 memory as output tiles. It coordinates with the math thread via the `MATH_PACK` semaphore and the double-buffered DEST mechanism described in Chapter 3. This section documents the complete, ordered sequence of six LLK API calls required on TRISC2.

Note: On Blackhole, several pack functions use a **three-template-parameter variant** -- `<is_fp32_dest_acc_en, untilize, tilize>` or `<untilize, zero_output, tilize>` -- that differs from earlier architectures. All mode booleans are `false` for standard matmul output.

### 4.3.1 The Complete TRISC2 Kernel

The pack thread's `run_kernel` as compiled under `#ifdef LLK_TRISC_PACK` with `#ifdef ARCH_BLACKHOLE`:

```cpp
void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;

    // (1) Hardware configure
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);

    // (2) Pack init -- address modifiers and MOP
    _llk_pack_init_<false, false, false>(formats.pack_dst);

    // (3) Dest sync init
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // (4) Wait for math to produce results
    _llk_packer_wait_for_math_done_();

    // (5) Pack loop over output tiles
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(
            i, L1_ADDRESS(params.buffer_Res[i]));
    }

    // (6) Section done -- clear dest half, release to math
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}
```

There are **six API calls** (three one-time setup, one sync wait, one per-tile pack in a loop, and one finalization call).

---

### 4.3.2 Call 1: `_llk_pack_hw_configure_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_pack.h`:

```cpp
template <bool is_fp32_dest_acc_en, bool untilize = false, bool tilize = false>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t tile_c_dim  = TILE_C_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const bool narrow_tile          = false,
    const std::uint32_t relu_config = 0);
```

#### What the Test Passes

```cpp
_llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
    formats.pack_src,       // Source format (DEST register format)
    formats.pack_dst,       // Destination format (L1 output format)
    params.TILE_SIZE_PACK); // Byte size of one output tile in L1
```

The Blackhole variant takes **three template booleans**: `is_fp32_dest_acc_en`, `untilize` (false), and `tilize` (false). The remaining parameters default to standard 32x32 tile values.

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `is_fp32_dest_acc_en` | `bool` (template) | `is_fp32_dest_acc_en` | Controls DEST read width. When `true`, sets `PCK_DEST_RD_CTRL_Read_32b_data = 1`. When `false`, reads 16-bit values. Also controls exponent thresholding for BFP-A formats. |
| `untilize` | `bool` (template) | `false` | Enables untilize mode for row-major L1 output layout. Standard matmul packs in tile order. |
| `tilize` | `bool` (template) | `false` | Enables tilize mode for Blackhole-specific row reordering. Not used for matmul. |
| `pack_src_format` | `uint32_t` | `formats.pack_src` | Data format of the DEST register data (the packer's input). Programs `in_data_format` in the pack config register and sets `ALU_FORMAT_SPEC_REG2_Dstacc`. |
| `pack_dst_format` | `uint32_t` | `formats.pack_dst` | Target data format for packed output in L1. Programs `out_data_format` in pack config. Controls format conversion in the packer pipeline. |
| `tile_size` | `uint32_t` | `params.TILE_SIZE_PACK` | Byte size of one output tile in L1. Stored in `p_gpr_pack::TILE_HEADER` GPR. |
| `face_r_dim` | `uint32_t` | `FACE_R_DIM` (= 16, default) | Rows per face. Programs `pack_reads_per_xy_plane` in pack counters. |
| `tile_c_dim` | `uint32_t` | `TILE_C_DIM` (= 32, default) | Columns per tile. Used to compute packer strides. |
| `num_faces` | `uint32_t` | `4` (default) | Faces per tile. Programs `exp_section_size` for BFP formats. |
| `partial_face` | `bool` | `false` (default) | Unused on Blackhole (asserted false). |
| `narrow_tile` | `bool` | `false` (default) | Unused on Blackhole (asserted false). |
| `relu_config` | `uint32_t` | `0` (default) | ReLU activation configuration. 0 = disabled. Programs `STACC_RELU_ApplyRelu` and `STACC_RELU_ReluThreshold`. |

#### What It Does Internally

Delegates to `configure_pack<is_fp32_dest_acc_en, untilize, tilize>(...)` in `cpack_common.h`:

1. **Set packer strides** (`set_packer_strides<>()`):
   - Computes `x_stride` from format width (4 for FP32, 2 for FP16, 1 for 8-bit).
   - `y_stride = FACE_C_DIM * x_stride` (stride between rows).
   - `z_stride = FACE_R_DIM * y_stride` (stride between faces).
   - `w_stride = 4 * z_stride` (stride between tiles).
   - Writes to `PCK0_ADDR_CTRL_XY_REG_0` and `PCK0_ADDR_CTRL_ZW_REG_0`.

2. **Program ALU format registers** (mutex-protected):
   - Sets FP8 E4M3 mode bit (`THCON_SEC0_REG1_Pac_LF8_4b_exp`).
   - Sets `ALU_FORMAT_SPEC_REG2_Dstacc` to the masked source format.
   - Configures ReLU (zero config = disabled for matmul).

3. **Set packer config** (`set_packer_config<>()`):
   - Programs THCON_SEC0_REG1: `exp_section_size`, `uncompress = 1`, `out_data_format`, `in_data_format`.
   - Configures DEST read control: `PCK_DEST_RD_CTRL_Read_32b_data`, `Read_int8`, `Read_unsigned`, `Round_10b_mant`.
   - Resets L1 accumulation mode.
   - Configures exponent thresholding for FP32-dest-to-BFP-A conversions.

4. **Program pack counters** (`PACK_COUNTERS_SEC0_pack_reads_per_xy_plane = face_r_dim`).

5. **Set edge mask** (`PCK_EDGE_OFFSET_SEC0_mask = 0xFFFF`, all datums pass).

6. **Store tile header** (`regfile[p_gpr_pack::TILE_HEADER] = tile_size`).

7. **Program X address counter** (`TTI_SETADCXX(PAC, FACE_C_DIM - 1, 0x0)` -- bounding X to one face column, 0..15).

#### Classification

**One-time call.** Full packer hardware configuration for the output data format and tile geometry.

---

### 4.3.3 Call 2: `_llk_pack_init_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_pack.h`:

```cpp
template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false,
    const std::uint32_t num_tiles  = 1);
```

#### What the Test Passes

```cpp
_llk_pack_init_<false, false, false>(formats.pack_dst);
```

Only `pack_dst_format` is provided; all others default to standard 32x32 tile values.

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `untilize` | `bool` (template) | `false` | Selects untilize MOP variant with strided access pattern. |
| `zero_output` | `bool` (template) | `false` | When `true`, sets `P_ZERO_OUTPUT_ENABLED` in PACR instructions, writing zeros regardless of DEST content. |
| `tilize` | `bool` (template) | `false` | Selects tilize MOP variant for Blackhole-specific row reordering. |
| `pack_dst_format` | `uint32_t` | `formats.pack_dst` | Output format code. Forwarded to MOP config. |
| `face_r_dim` | `uint32_t` | `FACE_R_DIM` (= 16, default) | Rows per face. MOP inner loop count: `face_r_dim >> 2` = 4. |
| `tile_c_dim` | `uint32_t` | `TILE_C_DIM` (= 32, default) | Used only in untilize mode. |
| `num_faces` | `uint32_t` | `4` (default) | MOP outer loop count: `num_faces * num_tiles`. |
| `partial_face` | `bool` | `false` (default) | Unused on Blackhole. |
| `narrow_tile` | `bool` | `false` (default) | Unused on Blackhole. |
| `num_tiles` | `uint32_t` | `1` (default) | Tiles per MOP execution. Multiplied with `num_faces` for outer loop. |

#### What It Does Internally

Two substeps:

**Step 1: Address modifier configuration** (`_llk_pack_configure_addrmod_<false, false>()`)

Programs three address modifiers for the standard (non-untilize, non-tilize) pack path:

```cpp
addr_mod_pack_t { .y_src = {.incr = 4}, .y_dst = {.incr = 4} }.set(ADDR_MOD_0);
```
- `ADDR_MOD_0`: Increments both source (DEST register) and destination (output buffer) Y-pointers by 4 rows per PACR instruction. Each PACR processes 4 rows when all 4 packer interfaces are active.

```cpp
addr_mod_pack_t {
    .y_src = {.incr = 0, .clr = 1, .cr = 0},
    .y_dst = {.incr = 0, .clr = 1, .cr = 0},
    .z_src = {.incr = 0, .clr = 1},
    .z_dst = {.incr = 0, .clr = 0},
}.set(ADDR_MOD_1);
```
- `ADDR_MOD_1`: Last outer loop -- clears Y source/dest and Z source, marking tile completion.

```cpp
addr_mod_pack_t {
    .y_src = {.incr = 0, .clr = 1, .cr = 0},
    .y_dst = {.incr = 4, .clr = 0, .cr = 0},
    .z_src = {.incr = 1, .clr = 0},
}.set(ADDR_MOD_2);
```
- `ADDR_MOD_2`: Last inner loop -- resets Y source, increments Z source (next face), increments Y dest by 4.

**Step 2: MOP configuration** (`_llk_pack_mop_config_<false, false, false>(...)`)

For standard 32x32 tiles with `FACE_R_DIM=16`, `num_faces=4`:
- `PACK_INTF_SEL = ALL_INTF_ACTIVE` (all 4 packer interfaces active, processing 4 rows simultaneously).
- `MOP_INNER_LOOP = face_r_dim >> 2` = 4 (packs 16 rows per face in 4 steps of 4 rows each).
- `MOP_OUTER_LOOP = num_faces * num_tiles` = 4 (4 faces per tile).

The MOP is a `ckernel_template` with:
- **Primary instruction** (ADDR_MOD_0): Standard PACR that reads from DEST and writes to L1 with `ALL_INTF_ACTIVE`.
- **Last inner loop instruction** (ADDR_MOD_2): Face boundary modifier.
- **Last outer loop instruction** (ADDR_MOD_1): Tile completion modifier with tile-close flag set to 1.

This produces $4 \times 4 = 16$ PACR operations per tile, reading 4 rows at a time from the 64-row DEST tile (4 faces of 16 rows each).

#### Classification

**One-time call.** Programs packer address modifiers and MOP for standard tiled output.

---

### 4.3.4 Call 3: `_llk_pack_dest_init_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_pack_common.h`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const bool narrow_tile = false);
```

#### What the Test Passes

```cpp
_llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `Dst` | `DstSync` (template) | `DstSync::SyncHalf` | Synchronization mode. Determines DEST offset register programming. |
| `is_fp32_dest_acc_en` | `bool` (template) | `is_fp32_dest_acc_en` | FP32 accumulation mode flag. Passed through to register selection logic. |
| `face_r_dim` | `uint32_t` | `FACE_R_DIM` (= 16, default) | Unused on Blackhole (asserted to equal FACE_R_DIM). |
| `narrow_tile` | `bool` | `false` (default) | Unused on Blackhole (asserted false). |

#### What It Does Internally

```cpp
tensix_sync();
reset_dest_offset_id();
_llk_init_packer_dest_offset_registers_<Dst>(face_r_dim, narrow_tile);
packer_addr_counter_init();
pack_sync_tile_dst_ptr = 0;
```

1. **`tensix_sync()`**: Full pipeline barrier.
2. **`reset_dest_offset_id()`**: Sets `dest_offset_id = 0`.
3. **`_llk_init_packer_dest_offset_registers_`**: Programs the packer's DEST offset GPRs:
   ```cpp
   TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
   TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
   ```
   Then calls `select_packer_dest_registers<Dst>()` which writes the current DEST offset into `DEST_TARGET_REG_CFG_PACK_SEC0_Offset` via a 128-bit config write, followed by two `TTI_DMANOP` instructions to ensure the write completes.

4. **`packer_addr_counter_init()`**: Resets packer X/Y/Z/W address counters.
5. **`pack_sync_tile_dst_ptr = 0`**: Resets the pack-side tile pointer.

#### Classification

**One-time call.** Initializes packer DEST offset registers and address counters for double-buffered DEST access.

---

### 4.3.5 Call 4: `_llk_packer_wait_for_math_done_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_pack_common.h`:

```cpp
inline void _llk_packer_wait_for_math_done_();
```

#### What the Test Passes

```cpp
_llk_packer_wait_for_math_done_();
```

#### What It Does Internally

```cpp
TTI_SEMWAIT(p_stall::STALL_TDMA,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_ZERO);
```

Stalls the packer's TDMA engine until the `MATH_PACK` semaphore is non-zero. The semaphore is posted by `_llk_math_dest_section_done_` on TRISC1. `STALL_ON_ZERO` = block while the semaphore equals zero; proceed once math posts it to a positive value.

#### Classification

**Per-dest-section call.** Synchronizes the packer with the math engine.

---

### 4.3.6 Call 5 (per-tile): `_llk_pack_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_pack.h`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_(
    const std::uint32_t tile_index,
    const std::uint32_t address);
```

#### What the Test Passes

```cpp
for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
{
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(
        i,                              // tile_index within DEST register
        L1_ADDRESS(params.buffer_Res[i]) // L1 output address
    );
}
```

Where `TILE_CNT = RT_DIM * CT_DIM` -- the total number of output tiles.

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `Dst` | `DstSync` (template) | `DstSync::SyncHalf` | Sync mode -- must match all other sync configurations. |
| `is_fp32_dest_acc_en` | `bool` (template) | `is_fp32_dest_acc_en` | FP32 accumulation flag. |
| `untilize` | `bool` (template) | `false` | Untilize mode. |
| `tile_index` | `uint32_t` | `i` (0 to TILE_CNT-1) | Tile index within the current DEST section. Programs the packer's W address counter: `TT_SETADC(PAC, CH_0, SET_W, tile_index)`. |
| `address` | `uint32_t` | `L1_ADDRESS(params.buffer_Res[i])` | L1 destination address. Written (with bit 31 set) to `THCON_SEC0_REG1_L1_Dest_addr` via GPR + WRCFG. |

#### What It Does Internally

Four operations per tile:

**Step 1 -- Set DEST read address.**

```cpp
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, tile_index);
}
```

The W counter, combined with the W stride programmed during `hw_configure`, positions the packer to read from the correct tile within the DEST half.

**Step 2 -- Program L1 destination address.**

```cpp
inline void program_packer_destination(std::uint32_t addr)
{
    std::uint32_t new_l1_addr = (1 << 31) | addr;
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_DMANOP;
}
```

Bit 31 is set to signal the packer hardware that the address is valid, then immediately cleared by restoring the original upper halfword.

**Step 3 -- Execute the MOP.**

```cpp
ckernel::ckernel_template::run();
```

Runs the pre-programmed pack MOP (4 outer loops x 4 inner loops = 16 PACR instructions). Each PACR reads 4 rows from DEST (using all 4 packer interfaces in parallel) and writes them to L1 through the pack gasket.

**Step 4 -- Reset Z counters.**

```cpp
TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
```

Resets the Z counters (face counters) for the next tile.

#### Classification

**Per-tile call.** Invoked $TILE\_CNT = RT\_DIM \times CT\_DIM$ times.

---

### 4.3.7 Call 6: `_llk_pack_dest_section_done_`

#### Signature

From `tt_llk_blackhole/llk_lib/llk_pack_common.h`:

```cpp
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_section_done_();
```

#### What the Test Passes

```cpp
_llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

#### Parameter Table

| Parameter | Type | Value in `matmul_test.cpp` | What It Controls |
|---|---|---|---|
| `Dst` | `DstSync` (template) | `DstSync::SyncHalf` | Sync mode. |
| `is_fp32_dest_acc_en` | `bool` (template) | `is_fp32_dest_acc_en` | Controls ZEROACC instruction parameter (32-bit vs 16-bit clear). |

#### What It Does Internally

1. **Wait for pack to finish.** `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK)` -- stalls until all PACR operations are complete and data has been written to L1.

2. **Clear the DEST section.** For `SyncHalf`:
   ```cpp
   TT_ZEROACC(p_zeroacc::CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id % 2);
   ```
   Zeros out only the half that was just read, using `dest_offset_id` to select which half (0 or 1).

3. **Release math semaphore.** `_llk_packer_set_math_semaphore_<p_stall::NONE>()` calls `t6_semaphore_get(semaphore::MATH_PACK)`, decrementing the semaphore to signal the math thread that a DEST section is now free.

4. **Flip DEST offset.**
   - `flip_packer_dest_offset_id()`: Toggles `dest_offset_id` between 0 and 1.
   - `select_packer_dest_registers<Dst>()`: Writes the new DEST offset to `DEST_TARGET_REG_CFG_PACK_SEC0_Offset`, pointing the packer at the other half for the next pack operation.

#### Classification

**Per-dest-section call.** Completes the pack cycle, clears the consumed DEST half, and releases it to the math thread.

---

### 4.3.8 Blackhole vs. Wormhole Pack API Differences

The following table summarizes the template parameter differences between Blackhole and Wormhole for the pack APIs used in matmul:

| Function | Blackhole Template Params | Wormhole Template Params |
|----------|--------------------------|--------------------------|
| `_llk_pack_hw_configure_` | `<is_fp32, untilize, tilize>` | `<is_fp32, untilize>` |
| `_llk_pack_init_` | `<untilize, zero_output, tilize>` | `<untilize, zero_output>` |
| `_llk_pack_dest_init_` | `<DstSync, is_fp32>` | `<DstSync, is_fp32, untilize>` |
| `_llk_pack_` | `<DstSync, is_fp32, untilize>` | `<DstSync, is_fp32, untilize>` (same) |
| `_llk_pack_dest_section_done_` | `<DstSync, is_fp32>` | `<DstSync, is_fp32>` (same) |

Key differences:

- **Blackhole adds a `tilize` template boolean** to `_llk_pack_hw_configure_` and `_llk_pack_init_`. This parameter controls row reordering to handle a Blackhole-specific row-swizzling behavior. When `tilize = true`, the packer applies row unswizzling. For standard matmul, `tilize = false` and the difference is moot.

- **Wormhole `_llk_pack_dest_init_` takes a third `untilize` template boolean** that Blackhole omits. This parameter is `false` for standard matmul on both architectures.

- **Single packer on Blackhole.** Blackhole has `NUM_PACKERS = 1` (down from 4 on Grayskull/Wormhole). The pack config and counters are programmed for a single packer instance, though the `PACK_INTF_SEL` field still selects how many packer interfaces (1, 2, or 4) are active within that single packer.

- **Format inference.** On Blackhole, the ALU infers SrcA/SrcB data formats from register contents, so `_llk_math_hw_configure_` does not need to explicitly program format fields. The function primarily controls INT8 math mode and FP32 accumulation.

- **PACR megarow.** Standard pack mode uses `MEGAROW = 0` (tile-ordered output). Untilize mode uses `MEGAROW = 1`. The `_llk_pack_mop_config_` function handles this automatically.

**Source files:**
- `tt_llk_blackhole/llk_lib/llk_pack.h`
- `tt_llk_blackhole/llk_lib/llk_pack_common.h`
- `tt_llk_blackhole/common/inc/cpack_common.h`
- `tests/sources/matmul_test.cpp` (PACK block)

---

**Next:** [`04_minimum_call_summary.md`](./04_minimum_call_summary.md)
