# Matrix Multiply Walkthrough

This section walks through `tests/sources/matmul_test.cpp` and its supporting LLK headers. Matrix multiply is the most complex kernel pattern in TT-LLK, involving LLTT replay buffers, multi-phase fidelity, register reuse strategies, and throttle-level tuning. Understanding this kernel provides insight into how the hardware is driven at maximum efficiency for the most performance-critical operation in deep learning.

## Kernel Structure Overview

The matmul kernel multiplies a block of Source A tiles by a block of Source B tiles, accumulating results in the destination register. The blocking is controlled by three dimensions:

- **`RT_DIM`** -- number of tile rows (from Source A)
- **`CT_DIM`** -- number of tile columns (from Source B)
- **`KT_DIM`** -- the reduction (inner) dimension, iterated in the main loop

The output is an `RT_DIM x CT_DIM` block of result tiles in the destination register.

## UNPACK Section

### Hardware Configuration

```cpp
_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
    formats.unpack_A_src, formats.unpack_B_src,
    formats.unpack_A_dst, formats.unpack_B_dst,
    FACE_R_DIM, FACE_R_DIM,
    params.num_faces_A, params.num_faces_B,
    params.TILE_SIZE_UNPACK_A, params.TILE_SIZE_UNPACK_B);
```

Unlike the eltwise binary kernel which uses `tensor_shape.face_r_dim` (supporting partial faces), the matmul unpack always uses the full `FACE_R_DIM` (16 rows).

### Matmul-Specific Initialization

```cpp
_llk_unpack_AB_matmul_init_<>(0, params.CT_DIM, params.RT_DIM, params.KT_DIM,
    FACE_R_DIM, FACE_R_DIM, 4, 4, false, false);
```

The function signature (from `tt_llk_wormhole_b0/llk_lib/llk_unpack_AB_matmul.h`):

```cpp
void _llk_unpack_AB_matmul_init_(
    const std::uint32_t transpose       = 0,
    const std::uint32_t ct_dim          = 1,
    const std::uint32_t rt_dim          = 1,
    const std::uint32_t kt_dim          = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4,
    const bool unpA_partial_face        = false,
    const bool unpB_partial_face        = false)
```

This function sets up LLTT replay buffers for the unpack MOP. It records UNPACR instruction sequences into the replay buffer -- one sequence for context 0 and one for context 1 (double-buffered). The MOP body then replays these sequences:

```cpp
// From the init function internals:
lltt::record(0, replay_buf_prog_len);
// ... sequence of UNPACR instructions recorded here ...

// MOP uses replay as its loop body:
ckernel_unpack_template tmp = ckernel_unpack_template::lBA(
    lltt::replay_insn(0, replay_buf_run_len),                     // Context 0
    ...,
    lltt::replay_insn(replay_buf_run_len, replay_buf_run_len),    // Context 1
    ...);
```

This double-buffered approach means the unpacker can prepare the next tile's data in one register bank while the math engine reads from the other.

### Inner Loop over K Dimension

```cpp
for (std::uint32_t j = 0; j < params.KT_DIM; j++)
{
    _llk_unpack_AB_matmul_<>(
        L1_ADDRESS(params.buffer_A[0]),
        L1_ADDRESS(params.buffer_B[0]),
        j,                           // K index
        j * params.CT_DIM,           // B tile offset
        params.TILE_SIZE_UNPACK_A,
        params.TILE_SIZE_UNPACK_B,
        false, false,
        params.CT_DIM, params.RT_DIM, params.KT_DIM);
}
```

The K-dimension loop pairs Source A tiles with Source B tiles. For each K step, the unpacker loads the appropriate A and B tiles based on the K index and the blocking dimensions.

## MATH Section

### Initialization

The MATH section requires three initialization calls. First, the matmul-specific init configures address modifiers, register reuse, and the MOP. Then two common init calls set up the math-pack synchronization protocol and the math engine's format registers:

```cpp
_llk_math_matmul_init_<MATH_FIDELITY>(
    TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM,
    false, 0, params.CT_DIM, params.RT_DIM);
_llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
_llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
```

`_llk_math_pack_sync_init_()` sets up the half/full synchronization protocol between math and pack engines (here, `SyncHalf`). `_llk_math_hw_configure_()` programs the math engine's format registers based on the configured data format.

`_llk_math_matmul_init_()` is the most complex initialization in the LLK library. It performs three major steps:

#### 1. Address Modifier Configuration

`matmul_configure_addrmod<MathFidelity, THROTTLE_LEVEL>()` sets up `ADDR_MOD_0` through `ADDR_MOD_6`. These control how the Source A, Source B, and destination pointers advance through the four faces of a tile during matrix multiplication.

The core computation is `D[8,16] = B[8,16] * A[16,16]` -- the FPU multiplies 8 rows of B by the full 16x16 A face, producing 8 rows of the destination. Four such operations cover one full 32x16 column of the output.

Key address modifiers for the standard 32x32 case:

| Slot | Purpose | SrcA | SrcB | Dest |
|------|---------|------|------|------|
| `ADDR_MOD_0` | Normal inner step | incr=0 | incr=8 | incr=8 |
| `ADDR_MOD_1` | Face transition (A advance) | incr=16 | cr=1 (reset to 0+16) | incr=8 |
| `ADDR_MOD_2` | Cross-face (B jump) | cr=1 | incr=32, cr=1 | incr=8 |
| `ADDR_MOD_3` | Like MOD_0 but with bias increment | incr=0 | incr=8 | incr=8, bias=1 |
| `ADDR_MOD_4` | Half-tile transition (A+B advance, dest reset via cr, bias incr) | incr=32, cr=1 | incr=48, cr=1 | incr=0, cr=1, bias=1 |
| `ADDR_MOD_5` | Reset all, increment fidelity | clr=1, cr=1 | clr=1, cr=1 | clr=1, cr=1, fidelity.incr=1 |
| `ADDR_MOD_6` | Reset all including fidelity (throttle) | clr=1, cr=1 | clr=1, cr=1 | clr=1, cr=1, fidelity.clr=1 |

The address modifiers cycle the FPU through all four face pairs (B0*A0, B0*A1, B2*A0, B2*A1 for the left half; B1*A2, B1*A3, B3*A2, B3*A3 for the right half) without any explicit pointer arithmetic in the instruction stream.

#### 2. Register Reuse Configuration

The init function determines a register reuse strategy based on the blocking dimensions:

```cpp
const bool reuse_a = ct_dim >= rt_dim;
```

- **Reuse A** (`ct_dim >= rt_dim`): Source A stays in the register while multiple Source B tiles are cycled through. This avoids reloading A for each column of the output.
- **Reuse B** (`ct_dim < rt_dim`): Source B stays while multiple Source A tiles are cycled.

The reuse strategy is communicated to the hardware by disabling the auto-clear of the reused source's valid flag:

```cpp
if (reuse_a)
    TTI_SETC16(CLR_DVALID_SrcB_Disable_ADDR32, CLR_DVALID_SrcB_Disable_MASK);
else
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, CLR_DVALID_SrcA_Disable_MASK);
```

#### 3. MOP Configuration via LLTT

`matmul_configure_mop<math_fidelity>()` records the MVMUL instruction sequence into an LLTT replay buffer and programs a `ckernel_template` whose loop body replays that sequence:

```cpp
lltt::record(ckernel::math::replay_buf_offset, replay_buf_len);

// Record 16 MVMUL instructions for 32x32 tiles (fewer for smaller tiles):
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0*A0, rows 0-7
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B0*A0, rows 8-15; advance A
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B0*A1, rows 0-7
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);  // B0*A1, rows 8-15; jump to B2
// ... 12 more instructions covering all face combinations ...

// The MOP replays this entire sequence as a single "instruction":
ckernel_template tmp(1, inner_loops, lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buf_len));
```

For high-fidelity math, `inner_loops` equals the fidelity level (2, 3, or 4), and `ADDR_MOD_5` at the end of each replay increments the fidelity phase counter while resetting all source/dest pointers. This means the same source data is processed multiple times with increasing precision.

### Main Compute Loop

```cpp
_llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
for (std::uint32_t j = 0; j < params.KT_DIM; j++)
{
    _llk_math_matmul_<MATH_FIDELITY>(0, params.CT_DIM, params.RT_DIM);
}
_llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

The math loop is simpler than the eltwise binary case because the output tile blocking is fixed. The K-dimension loop accumulates partial products into the destination register. Each call to `_llk_math_matmul_()` internally iterates over the `CT_DIM x RT_DIM` output tiles:

```cpp
// Inside _llk_math_matmul_:
for (std::uint32_t t = 0; t < t_dim; t++)
{
    for (std::uint32_t rut = 0; rut < rut_dim; rut++)
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(
            dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));
        ckernel_template::run();  // Fires the LLTT-backed MOP
    }
}
```

The inner loop order depends on the reuse strategy. When reusing A, the inner loop sweeps across columns (varying B), so Source A stays loaded. When reusing B, the inner loop sweeps across rows (varying A).

## Fidelity Phases

Matrix multiplication precision is controlled by the `MathFidelity` template parameter. The Tensix FPU can perform multiply-accumulate operations in multiple phases, where each phase processes the same input data but captures additional precision bits.

| Fidelity | Phases | Use Case |
|----------|--------|----------|
| `LoFi`   | 1      | Fast, sufficient for BF16/BFP8 |
| `HiFi2`  | 2      | Better precision for FP16 |
| `HiFi3`  | 3      | High precision for FP32 |
| `HiFi4`  | 4      | Maximum precision |

In LoFi mode, the MOP's inner loop count is 1 -- the replay buffer executes once. In HiFi modes, the inner loop count equals the fidelity level. After each phase, `ADDR_MOD_5` resets source/dest pointers and increments the fidelity counter, so the hardware re-reads the same data with the next phase's precision.

## Throttle Level Concept

Throttle levels (0-5) insert NOP instructions between MVMUL operations to reduce power consumption and thermal stress at the cost of throughput. The throttle level is a compile-time template parameter:

```cpp
template <MathFidelity math_fidelity, int THROTTLE_LEVEL = 0>
inline void _llk_math_matmul_init_(...);
```

At throttle level 0 (default), MVMULs execute back-to-back. Higher levels insert progressively more NOPs:

- **Level 1**: One NOP before every other MVMUL pair.
- **Level 2**: One NOP before each MVMUL pair.
- **Level 3**: One NOP before every MVMUL.
- **Levels 4-5**: More aggressive throttling with restructured replay buffer segments.

When `THROTTLE_LEVEL > 0`, the init function calls `matmul_configure_mop_throttled` instead of `matmul_configure_mop`. The throttled version records a different instruction sequence into the LLTT replay buffer with NOPs interleaved:

```cpp
// Throttle level 1 example:
TTI_NOP;
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
TTI_NOP;
TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
// ...
```

For levels 4 and 5, the replay buffer is split into segments, and the MOP loop body references different sub-ranges:

```cpp
constexpr std::uint32_t loop_instruction_0 = (THROTTLE_LEVEL == 5)
    ? lltt::replay_insn(ckernel::math::replay_buf_offset + 1, 8)
    : (THROTTLE_LEVEL == 4)
    ? lltt::replay_insn(ckernel::math::replay_buf_offset + 2, 6)
    : lltt::replay_insn(ckernel::math::replay_buf_offset, replay_buff_len_throttle);
```

## PACK Section

The matmul pack section follows the same pattern as eltwise binary but is simpler because matmul always uses `DstSync::SyncHalf` and full 32x32 tiles:

```cpp
_llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
    formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
_llk_pack_init_<false, false>(formats.pack_dst);
_llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>();

_llk_packer_wait_for_math_done_();
for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
{
    LLK_ASSERT((i < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "i exceeds max dest tiles");
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(i, L1_ADDRESS(params.buffer_Res[i]));
}
_llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
```

The `TILE_CNT` runtime parameter equals `CT_DIM * RT_DIM` -- the total number of output tiles in the result block.

## Summary: Eltwise Binary vs. Matmul

| Aspect | Eltwise Binary | Matmul |
|--------|---------------|--------|
| MOP body | Direct FPU instruction (`TT_OP_ELWADD`, etc.) | LLTT replay of recorded MVMUL sequence |
| Address modifiers | 4 slots (ADDR_MOD_0-3) | 7 slots (ADDR_MOD_0-6) |
| Fidelity phases | Only for ELWMUL | Always available |
| Register reuse | None (or dest reuse) | Source A or Source B reuse |
| Throttle levels | Not applicable | 0-5 |
| Tile dimensions | Variable (16x16 to 32x32) | Full 32x32 (16x16 not supported) |
| Unpack init | `_llk_unpack_AB_init_` | `_llk_unpack_AB_matmul_init_` with LLTT |
| K-dimension loop | Single tile loop | Accumulating over KT_DIM |

---

**End of guide.** Return to [Guide Index](../index.md)
