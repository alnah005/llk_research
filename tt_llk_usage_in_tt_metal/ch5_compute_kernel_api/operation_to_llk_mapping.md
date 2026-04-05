# Operation-to-LLK Mapping

This section provides a detailed mapping of every major Compute API function to the specific LLK calls it dispatches on each TRISC thread. Each subsection covers one per-operation header file.

Throughout this document:
- **TRISC0** = Unpack thread (`TRISC_UNPACK`)
- **TRISC1** = Math thread (`TRISC_MATH`)
- **TRISC2** = Pack thread (`TRISC_PACK`)

## 1. Matrix Multiply -- `matmul.h`

Source: `tt_metal/hw/inc/api/compute/matmul.h`

### `mm_init()` (line 88)

Full initialization for tile-level matmul. Configures all three TRISC threads.

```cpp
ALWI void mm_init(uint32_t in0_cb_id, uint32_t in1_cb_id,
                   uint32_t out_cb_id, const uint32_t transpose = 0,
                   uint32_t call_line = __builtin_LINE()) {
    state_configure(in1_cb_id, in0_cb_id, out_cb_id, call_line);
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_matmul_init(in0_cb_id, in1_cb_id, transpose)));

    MATH((llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, transpose)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(in0_cb_id, in1_cb_id)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(out_cb_id)));
    PACK((llk_pack_init(out_cb_id)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}
```

| TRISC | LLK Call | Purpose |
|-------|----------|---------|
| TRISC0 | `llk_unpack_hw_configure<DST_ACCUM_MODE>(in1, in0)` | Configure unpacker hardware for the given data formats |
| TRISC0 | `llk_unpack_AB_matmul_init(in0, in1, transpose)` | Initialize dual-operand unpack for matmul mode |
| TRISC1 | `llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0, in1, transpose)` | Configure math engine MOP for matmul |
| TRISC1 | `llk_math_pack_sync_init<DST_ACCUM_MODE>()` | Initialize math-pack synchronization |
| TRISC1 | `llk_math_hw_configure<DST_ACCUM_MODE>(in0, in1)` | Configure ALU data formats |
| TRISC2 | `llk_pack_hw_configure<DST_ACCUM_MODE>(out)` | Configure packer hardware |
| TRISC2 | `llk_pack_init(out)` | Initialize pack address generators |
| TRISC2 | `llk_pack_dest_init<DST_ACCUM_MODE, false>()` | Initialize DST read offset for pack |

### `matmul_tiles()` (line 137)

Per-tile matmul execution: unpacks A and B, then performs the math.

```cpp
ALWI void matmul_tiles(uint32_t in0_cb_id, uint32_t in1_cb_id,
                        uint32_t in0_tile_index, uint32_t in1_tile_index,
                        uint32_t idst) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id,
                                  in0_tile_index, in1_tile_index)));
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)));
}
```

| TRISC | LLK Call | Purpose |
|-------|----------|---------|
| TRISC0 | `llk_unpack_AB_matmul(in0, in1, tile_a, tile_b)` | Unpack tiles from both input CBs into SRC registers |
| TRISC1 | `llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)` | Execute matmul MOP, accumulate into DST[idst] |

### `mm_init_short()` (line 182)

Lightweight re-init that skips pack/HW configuration -- used when switching back to matmul after another op.

| TRISC | LLK Call |
|-------|----------|
| TRISC0 | `llk_unpack_AB_matmul_init(in0, in1, transpose)` |
| TRISC1 | `llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0, in1, transpose)` |

### `mm_block_init()` (line 231) / `matmul_block()` (line 282)

Block-level matmul variants that process multiple tiles at once. Same LLK functions but with additional `ct_dim`, `rt_dim`, `kt_dim` parameters:

| TRISC | Init LLK Call | Execute LLK Call |
|-------|---------------|------------------|
| TRISC0 | `llk_unpack_AB_matmul_init(in0, in1, transpose, ct_dim, rt_dim, kt_dim)` | `llk_unpack_AB_matmul(in0, in1, tile_a, tile_b, ct_dim, rt_dim, kt_dim)` |
| TRISC1 | `llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0, in1, transpose, ct_dim, rt_dim)` | `llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)` |
| TRISC2 | `llk_pack_hw_configure`, `llk_pack_init<false, false>`, `llk_pack_dest_init` | (via `pack_tile` separately) |

On Blackhole architecture, `matmul_block()` uses firmware-controlled dynamic throttling via `matmul_block_math_dynamic_throttle()`, which can switch between `MM_THROTTLE` (default, typically 0) and `MM_THROTTLE_MAX` (5) based on a firmware flag at `MEM_L1_ARC_FW_SCRATCH`.

---

## 2. Element-wise Binary -- `eltwise_binary.h`

Source: `tt_metal/hw/inc/api/compute/eltwise_binary.h`

### `binary_op_init_common()` (line 31)

Full hardware initialization for any binary operation. This is the heaviest init, touching all three TRISC threads.

```cpp
ALWI void binary_op_init_common(uint32_t icb0, uint32_t icb1,
                                 uint32_t ocb,
                                 uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, ocb, call_line);

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}
```

| TRISC | LLK Call | Purpose |
|-------|----------|---------|
| TRISC0 | `llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)` | Configure unpacker data formats |
| TRISC0 | `llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)` | Initialize dual-input unpack (no broadcast) |
| TRISC1 | `llk_math_pack_sync_init<DST_ACCUM_MODE>()` | Math-pack sync setup |
| TRISC1 | `llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)` | ALU format configuration |
| TRISC2 | `llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)` | Packer HW config |
| TRISC2 | `llk_pack_init(ocb)` | Pack address generator init |
| TRISC2 | `llk_pack_dest_init<DST_ACCUM_MODE, false>()` | DST read offset for pack |

### `add_tiles()` / `sub_tiles()` / `mul_tiles()` (lines 170, 194, 137)

All three follow the identical pattern, differing only in the `EltwiseBinaryType` template parameter:

```cpp
// add_tiles (line 170)
ALWI void add_tiles(uint32_t icb0, uint32_t icb1,
                     uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE,
          MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
          icb0, icb1, idst, true)));
}
```

| TRISC | LLK Call | `add_tiles` | `sub_tiles` | `mul_tiles` |
|-------|----------|-------------|-------------|-------------|
| TRISC0 | `llk_unpack_AB(icb0, icb1, itile0, itile1)` | Same | Same | Same |
| TRISC1 | `llk_math_eltwise_binary<OP, ...>(icb0, icb1, idst, true)` | `OP = ELWADD` | `OP = ELWSUB` | `OP = ELWMUL` |

### `binary_tiles_init<full_init, type>()` (line 60)

Short init for switching between binary op types without full HW reconfiguration:

| TRISC | LLK Call |
|-------|----------|
| TRISC1 | `llk_math_eltwise_binary_init_with_operands<type, NONE, MATH_FIDELITY>(icb0, icb1, acc_to_dest)` |
| TRISC0 (if `full_init`) | `llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, 0)` |

### `binary_dest_reuse_tiles<type, reuse>()` (line 239)

Variant that reads one operand from DST (for fused ops):

| TRISC | LLK Call |
|-------|----------|
| TRISC0 | `llk_unpack_A<BroadcastType::NONE, true, reuse>(in_cb_id, in_tile_index)` |
| TRISC1 | `llk_math_eltwise_binary<type, NONE, DST_ACCUM_MODE, MATH_FIDELITY, reuse>(in_cb_id, in_cb_id, dst_tile_index, true)` |

---

## 3. Reduce -- `reduce.h`

Source: `tt_metal/hw/inc/api/compute/reduce.h`

### `reduce_init<reduce_type, reduce_dim>()` (line 50)

```cpp
template <PoolType reduce_type, ReduceDim reduce_dim,
          bool enforce_fp32_accumulation = false>
ALWI void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb,
                       uint32_t call_line = __builtin_LINE()) {
    state_configure(icb, icb_scaler, ocb, call_line);
    UNPACK((llk_unpack_AB_reduce_init<reduce_type, reduce_dim,
            enforce_fp32_accumulation>(icb, icb_scaler)));
    MATH((llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE,
          MATH_FIDELITY, enforce_fp32_accumulation>()));
    if constexpr (enforce_fp32_accumulation) {
        MATH((tensix_sync()));
        MATH((reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1 << 11)));
    }
    PACK((llk_pack_reduce_mask_config<false, reduce_dim>()));
}
```

| TRISC | LLK Call | Purpose |
|-------|----------|---------|
| TRISC0 | `llk_unpack_AB_reduce_init<reduce_type, reduce_dim, enforce_fp32>(icb, icb_scaler)` | Configure unpacker for reduce pattern |
| TRISC1 | `llk_math_reduce_init<reduce_type, reduce_dim, DST_ACCUM_MODE, MATH_FIDELITY, enforce_fp32>()` | Configure math engine reduce mode |
| TRISC2 | `llk_pack_reduce_mask_config<false, reduce_dim>()` | Set packer edge masks for reduced output shape |

Note: When `enforce_fp32_accumulation` is true, additional register writes disable certain hardware optimizations to preserve FP32 precision.

### `reduce_tile<reduce_type, reduce_dim>()` (line 126)

```cpp
ALWI void reduce_tile(uint32_t icb, uint32_t icb_scaler,
                       uint32_t itile, uint32_t itile_scaler,
                       uint32_t idst) {
    MATH((llk_math_reduce<reduce_type, reduce_dim, DST_ACCUM_MODE,
          MATH_FIDELITY, false, enforce_fp32_accumulation>(
          icb, icb_scaler, idst)));
    UNPACK((llk_unpack_AB_reduce<reduce_type, reduce_dim>(
            icb, icb_scaler, itile, itile_scaler)));
}
```

| TRISC | LLK Call | Purpose |
|-------|----------|---------|
| TRISC1 | `llk_math_reduce<reduce_type, reduce_dim, ...>(icb, icb_scaler, idst)` | Execute reduce math |
| TRISC0 | `llk_unpack_AB_reduce<reduce_type, reduce_dim>(icb, icb_scaler, itile, itile_scaler)` | Unpack input tile + scaler |

### `reduce_uninit()` (line 78)

Cleanup after reduce -- resets packer masks and math state:

| TRISC | LLK Call |
|-------|----------|
| TRISC1 | `llk_math_reduce_uninit<enforce_fp32>(icb)` |
| TRISC2 | `llk_pack_reduce_mask_clear()` |

---

## 4. Tile Copy -- `tile_move_copy.h`

Source: `tt_metal/hw/inc/api/compute/tile_move_copy.h`

### `copy_tile_to_dst_init_short()` (line 31)

```cpp
ALWI void copy_tile_to_dst_init_short(uint32_t cbid,
                                       uint32_t transpose = 0,
                                       uint32_t transpose_within_16x16_face = false,
                                       uint32_t call_line = __builtin_LINE()) {
    state_configure(cbid, call_line);
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false,
            EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            transpose, transpose_within_16x16_face, cbid)));
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE,
          BroadcastType::NONE>(cbid)));
}
```

| TRISC | LLK Call | Purpose |
|-------|----------|---------|
| TRISC0 | `llk_unpack_A_init<NONE, false, NONE, UnpackToDestEn>(transpose, transpose_within_face, cbid)` | Configure single-operand unpack |
| TRISC1 | `llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, NONE>(cbid)` | Configure datacopy MOP (A-to-DST) |

### `copy_tile()` (line 95)

```cpp
ALWI void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index,
                     uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false,
            EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE,
          BroadcastType::NONE, UnpackToDestEn>(
          dst_tile_index, in_cb_id)));
}
```

| TRISC | LLK Call | Purpose |
|-------|----------|---------|
| TRISC0 | `llk_unpack_A<NONE, false, NONE, UnpackToDestEn>(cb, tile_idx)` | Unpack single tile from CB to SRC |
| TRISC1 | `llk_math_eltwise_unary_datacopy<A2D, ...>(dst_idx, cb)` | Move tile from SRC to DST |

---

## 5. Pack -- `pack.h`

Source: `tt_metal/hw/inc/api/compute/pack.h`

### `pack_tile()` (line 61)

```cpp
template <bool out_of_order_output = false>
ALWI void pack_tile(uint32_t ifrom_dst, uint32_t icb,
                     std::uint32_t output_tile_index = 0) {
    PACK((llk_pack<DST_ACCUM_MODE, out_of_order_output, false>(
          ifrom_dst, icb, output_tile_index)));
}
```

| TRISC | LLK Call | Purpose |
|-------|----------|---------|
| TRISC2 | `llk_pack<DST_ACCUM_MODE, out_of_order, false>(ifrom_dst, icb, output_tile_index)` | Read tile from DST register and write to output CB |

### `pack_tile_block()` (line 101)

| TRISC | LLK Call |
|-------|----------|
| TRISC2 | `llk_matmul_pack<DST_ACCUM_MODE, false, false>(ifrom_dst, icb, ntiles)` |

### `pack_reconfig_data_format()` (line 126)

| TRISC | LLK Call |
|-------|----------|
| TRISC2 | `llk_pack_reconfig_data_format<DST_ACCUM_MODE, is_tile_dim_reconfig_en>(new_cb_id)` |

---

## 6. Broadcast -- `bcast.h`

Source: `tt_metal/hw/inc/api/compute/bcast.h`

### `init_bcast<tBcastOp, tBcastDim>()` (line 240)

Full init for broadcast binary operations:

| TRISC | LLK Call |
|-------|----------|
| TRISC0 | `llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1)` |
| TRISC0 | `llk_unpack_AB_init<tBcastDim>(icb0, icb1)` |
| TRISC1 | `llk_math_eltwise_binary_init<tBcastOp, tBcastDim, fidelity>()` |
| TRISC1 | `llk_math_pack_sync_init<DST_ACCUM_MODE>()` |
| TRISC1 | `llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1)` |
| TRISC2 | `llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)` |
| TRISC2 | `llk_pack_init(ocb)` |
| TRISC2 | `llk_pack_dest_init<DST_ACCUM_MODE, false>()` |

Note: For `ELWMUL` broadcast, `MATH_FIDELITY` is used. For `ELWADD` and `ELWSUB` broadcast, the full `init_bcast` uses `MathFidelity::LoFi`. Among the short-inits, only `add_bcast_cols_init_short` hardcodes `MathFidelity::LoFi`. Other short-inits -- including `sub_bcast_cols_init_short`, row-broadcast short-inits (e.g., `add_bcast_rows_init_short`), and scalar-broadcast short-inits -- use the configured `MATH_FIDELITY`.

### `add_tiles_bcast<tBcastDim>()` / `sub_tiles_bcast<tBcastDim>()` / `mul_tiles_bcast<tBcastDim>()` (lines 312-333)

All delegate to `any_tiles_bcast<tBcastOp, tBcastDim>()`:

| TRISC | LLK Call |
|-------|----------|
| TRISC1 | `llk_math_eltwise_binary<tBcastOp, tBcastDim, DST_ACCUM_MODE, MATH_FIDELITY, NONE>(icb0, icb1, idst, true)` |
| TRISC0 | `llk_unpack_AB<tBcastDim>(icb0, icb1, itile0, itile1, bcast_row_idx)` |

---

## 7. Circular Buffer Operations -- `cb_api.h`

Source: `tt_metal/hw/inc/api/compute/cb_api.h`

| Compute API | TRISC | LLK Call |
|-------------|-------|----------|
| `cb_wait_front(cbid, ntiles)` | TRISC0 | `llk_wait_tiles(cbid, ntiles)` |
| `cb_pop_front(cbid, ntiles)` | TRISC0 | `llk_pop_tiles(cbid, ntiles)` |
| `cb_reserve_back(cbid, ntiles)` | TRISC2 | `llk_wait_for_free_tiles<false, false, false>(cbid, ntiles)` |
| `cb_push_back(cbid, ntiles)` | TRISC2 | `llk_push_tiles<false, false>(cbid, ntiles)` |

Consumer-side operations (wait/pop) run on TRISC0 because the unpacker needs to read input data. Producer-side operations (reserve/push) run on TRISC2 because the packer writes output data.

---

## 8. DST Register Synchronization -- `reg_api.h`

Source: `tt_metal/hw/inc/api/compute/reg_api.h`

| Compute API | TRISC | LLK Call | Purpose |
|-------------|-------|----------|---------|
| `tile_regs_acquire()` | TRISC1 | `llk_math_wait_for_dest_available()` | Math thread waits for DST to be available |
| `tile_regs_commit()` | TRISC1 | `llk_math_dest_section_done<DST_ACCUM_MODE>()` | Math thread signals DST is ready for packing |
| `tile_regs_wait()` | TRISC2 | `llk_packer_wait_for_math_done()` | Pack thread waits for math to finish |
| `tile_regs_release()` | TRISC2 | `llk_pack_dest_section_done<DST_ACCUM_MODE>()` | Pack thread signals DST is free |

The deprecated `acquire_dst()` combines `tile_regs_acquire` + `tile_regs_wait`, and `release_dst()` combines `tile_regs_commit` + `tile_regs_release`.

---

## Comprehensive Mapping Table

| Compute API Function | Header | TRISC0 (Unpack) LLK | TRISC1 (Math) LLK | TRISC2 (Pack) LLK |
|---------------------|--------|---------------------|-------------------|-------------------|
| `mm_init()` | `matmul.h` | `llk_unpack_hw_configure`, `llk_unpack_AB_matmul_init` | `llk_math_matmul_init`, `llk_math_pack_sync_init`, `llk_math_hw_configure` | `llk_pack_hw_configure`, `llk_pack_init`, `llk_pack_dest_init` |
| `matmul_tiles()` | `matmul.h` | `llk_unpack_AB_matmul` | `llk_math_matmul` | -- |
| `mm_init_short()` | `matmul.h` | `llk_unpack_AB_matmul_init` | `llk_math_matmul_init` | -- |
| `binary_op_init_common()` | `eltwise_binary.h` | `llk_unpack_hw_configure`, `llk_unpack_AB_init` | `llk_math_pack_sync_init`, `llk_math_hw_configure` | `llk_pack_hw_configure`, `llk_pack_init`, `llk_pack_dest_init` |
| `add_tiles()` | `eltwise_binary.h` | `llk_unpack_AB` | `llk_math_eltwise_binary<ELWADD>` | -- |
| `sub_tiles()` | `eltwise_binary.h` | `llk_unpack_AB` | `llk_math_eltwise_binary<ELWSUB>` | -- |
| `mul_tiles()` | `eltwise_binary.h` | `llk_unpack_AB` | `llk_math_eltwise_binary<ELWMUL>` | -- |
| `reduce_init()` | `reduce.h` | `llk_unpack_AB_reduce_init` | `llk_math_reduce_init` | `llk_pack_reduce_mask_config` |
| `reduce_tile()` | `reduce.h` | `llk_unpack_AB_reduce` | `llk_math_reduce` | -- |
| `reduce_uninit()` | `reduce.h` | -- | `llk_math_reduce_uninit` | `llk_pack_reduce_mask_clear` |
| `copy_tile_to_dst_init_short()` | `tile_move_copy.h` | `llk_unpack_A_init` | `llk_math_eltwise_unary_datacopy_init` | -- |
| `copy_tile()` | `tile_move_copy.h` | `llk_unpack_A` | `llk_math_eltwise_unary_datacopy` | -- |
| `pack_tile()` | `pack.h` | -- | -- | `llk_pack` |
| `init_bcast()` | `bcast.h` | `llk_unpack_hw_configure`, `llk_unpack_AB_init` | `llk_math_eltwise_binary_init`, `llk_math_pack_sync_init`, `llk_math_hw_configure` | `llk_pack_hw_configure`, `llk_pack_init`, `llk_pack_dest_init` |
| `add_tiles_bcast<dim>()` | `bcast.h` | `llk_unpack_AB<dim>` | `llk_math_eltwise_binary<ELWADD, dim>` | -- |
| `cb_wait_front()` | `cb_api.h` | `llk_wait_tiles` | -- | -- |
| `cb_pop_front()` | `cb_api.h` | `llk_pop_tiles` | -- | -- |
| `cb_reserve_back()` | `cb_api.h` | -- | -- | `llk_wait_for_free_tiles` |
| `cb_push_back()` | `cb_api.h` | -- | -- | `llk_push_tiles` |
| `tile_regs_acquire()` | `reg_api.h` | -- | `llk_math_wait_for_dest_available` | -- |
| `tile_regs_commit()` | `reg_api.h` | -- | `llk_math_dest_section_done` | -- |
| `tile_regs_wait()` | `reg_api.h` | -- | -- | `llk_packer_wait_for_math_done` |
| `tile_regs_release()` | `reg_api.h` | -- | -- | `llk_pack_dest_section_done` |

---

**Next:** [`init_and_tile_pattern.md`](./init_and_tile_pattern.md)
