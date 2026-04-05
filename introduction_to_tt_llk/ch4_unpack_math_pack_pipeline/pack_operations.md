# Pack Operations

## What Packing Does

Packing is the third and final stage of the compute pipeline. It performs a DMA transfer of tile data from the **Destination register** to **L1 memory**, applying format conversion along the way. The pack stage is controlled by **TRISC2**.

Unlike the unpack stage which has two independent unpackers, there is a **single packer** that reads from the Destination register and writes to L1. The packer can convert from the internal register format (typically Float16 or Float32) to any supported L1 data format, including block floating point formats.

## LLK Pack API Headers

The pack API is defined in three header files in `tt_llk_wormhole_b0/llk_lib/`:

### `llk_pack.h` -- Standard Tile Packing

The primary packing interface. Transfers a tile from the Destination register to a specified L1 address.

```cpp
template <bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const bool narrow_tile          = false,
    const std::uint32_t relu_config = 0);

template <bool untilize = false, bool zero_output = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false);

template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address);
```

The `_llk_pack_` function sets the destination read address from `tile_index`, programs the packer's L1 write address from `address`, and runs the MOP with `mop_run(1, 1)`.

### `llk_pack_untilize.h` -- Pack with Untilization

Converts tile-ordered data to row-major layout during pack. This is the inverse of tilize-during-unpack: the packer reads faces from the Destination register and writes them in row-major order to L1. The MOP uses different outer loop counts based on `face_r_dim` and uses `ADDR_MOD_1` with a carry-reset pattern to handle the row-interleaved face layout.

### `llk_pack_rows.h` -- Row-Level Packing

Provides finer-grained packing at the row level rather than full tiles.

## Common API Pattern

Pack API headers follow the same four-function convention as unpack and math:

| Function | Purpose |
|:---------|:--------|
| `_llk_pack_hw_configure_<>()` | One-time hardware configuration: set source/destination formats, tile size, ReLU config. |
| `_llk_pack_init_<>()` | Configure address modifiers and MOP, set packer L1 offset and address counter limits. |
| `_llk_pack_<>()` | Execute one pack operation: set destination read address, program L1 write address, run MOP. |
| `_llk_pack_uninit_()` | Restore address counter defaults (`SETADCXX`). |

## MOP Configuration

The packer MOP is configured by `_llk_pack_mop_config_`:

```cpp
template <bool untilize = false, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false);
```

For standard (non-untilize) packing, the MOP is simple: a single `PACR` instruction that packs all faces in one shot using `PACK_SEL(num_faces)`. The `MEGAROW` flag is always set to 1. For BFP formats with partial faces, the MOP uses a start op to pack without closing the tile, then a final instruction to close it.

For untilize packing, the MOP outer loop runs `face_r_dim / 2` iterations, with `INCADCXY` instructions to advance the source Y pointer between rows.

## Address Modifiers

The packer uses its own address modifier type, `addr_mod_pack_t`, which controls source (Destination register) and destination (L1) Y, Z, and carry-reset behavior:

```cpp
template <bool untilize = false>
inline void _llk_pack_configure_addrmod_()
{
    addr_mod_pack_t {
        .y_src = {.incr = 15},  // 4-bit value, max 15; INCADCXY adds 1 more
        .y_dst = {.incr = 1},
    }.set(ADDR_MOD_0);

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 1, .cr = 0},
    }.set(ADDR_MOD_1);
}
```

`ADDR_MOD_0` advances the source pointer by 16 rows (15 from the modifier + 1 from `INCADCXY`), which is the face height. `ADDR_MOD_1` clears both source and destination pointers, used at the end of a tile pack to reset for the next tile.

## The `zero_output` Template Parameter

When `zero_output` is `true`, the `PACR` instruction is emitted with `P_ZERO_OUTPUT_ENABLED`, which causes the packer to write zeros to L1 regardless of the Destination register contents. This is used for operations that need to produce a zero-filled output tile without performing any computation.

```cpp
constexpr std::uint32_t ZERO_OUTPUT_FLAG = zero_output
    ? p_pacr::P_ZERO_OUTPUT_ENABLED
    : p_pacr::P_ZERO_OUTPUT_DISABLED;
```

## ReLU Integration

The packer hardware has built-in support for applying ReLU (Rectified Linear Unit) activation during the pack operation. This avoids a separate SFPU pass for ReLU. The `relu_config` parameter in `_llk_pack_hw_configure_` encodes both the ReLU mode and threshold.

The `ReluType` enum (defined in `tt_llk_wormhole_b0/llk_lib/llk_defs.h`) specifies the mode:

```cpp
enum ReluType
{
    NO_RELU,             // No ReLU applied
    ZERO_RELU,           // Standard ReLU: max(0, x)
    MIN_THRESHOLD_RELU,  // ReLU with minimum threshold
    MAX_THRESHOLD_RELU,  // ReLU with maximum threshold
};
```

The `relu_config` parameter is a 32-bit value decoded in `_llk_pack_hw_configure_` (in `cpack_common.h`):
- Bits [15:0] select the `ReluType` (only the low 4 bits are meaningful, since `STACC_RELU_ApplyRelu` is a 4-bit field in the hardware register).
- Bits [31:16] encode the ReLU threshold value (mapped to the 16-bit `STACC_RELU_ReluThreshold` field).
- Bits [15:4] are effectively unused (they are masked off when written to the 4-bit hardware field).

This allows fusing ReLU with the pack operation at zero additional latency cost, since the packer applies the activation function as it reads each datum from the Destination register.

## Packer Synchronization with Math

The packer coordinates with the math stage through the `MATH_PACK` semaphore. For the full semaphore protocol, including `_llk_packer_wait_for_math_done_`, `_llk_packer_set_math_semaphore_`, and `_llk_pack_dest_section_done_`, see [synchronization.md -- MATH_PACK Semaphore](./synchronization.md#math_pack----mathpack-destination-coordination).

---

**Next:** [`synchronization.md`](./synchronization.md)
