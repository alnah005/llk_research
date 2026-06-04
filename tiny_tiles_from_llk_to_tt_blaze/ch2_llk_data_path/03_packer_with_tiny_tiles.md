# 2.3 The Packer with Tiny Tiles

The packer is the third pipeline stage in the LLK data path. After the math engine writes results to the destination register, the packer reads from Dest and writes back to L1 in tile-major format. For tiny tiles, the packer must (a) write only the populated rows of each face, (b) program L1 strides that reflect the shrunk tile footprint, and (c) skip face-row segments that do not exist. This section covers how `_llk_pack_init_<>()` is parameterized, the role of the [BH] third template parameter overload, the MOP/address-modifier programming for partial-row writes, and how `tile.get_tile_size(dtype)` produces the L1 byte count used everywhere downstream.

**Prerequisites:** Chapter 1 (tile geometry: `num_faces`, `face_r_dim`, `face_c_dim`), Chapter 2, Section 1 (unpacker partial-face mechanics), Chapter 2, Section 2 (math engine and Dest occupancy).

## 2.3.1 The two `_llk_pack_init_<>()` overloads

There are two flavors of `_llk_pack_init_<>()` in `tt_llk_blackhole/llk_lib/llk_pack.h`. The first overload covers the legacy case where tile geometry is fixed at compile time. The second is the [BH]-specific entry point that accepts `pack_src_format` and `num_tiles` explicitly and is the one tiny-tile call sites use.

```cpp
// tt_llk_blackhole/llk_lib/llk_pack.h:359
template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_init_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const std::uint32_t num_tiles  = 1)
{
    _llk_pack_configure_addrmod_<untilize, tilize>();
    _llk_pack_mop_config_<untilize, zero_output, tilize>(
        face_r_dim, tile_c_dim, num_faces, num_tiles);
}
```

```cpp
// tt_llk_blackhole/llk_lib/llk_pack.h:372  [BH]
template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_src_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    const bool          skip_bh_tilize_workaround = false)
{
    // ... format-dependent setup ...
    _llk_pack_mop_config_<untilize, zero_output, tilize>(
        face_r_dim, tile_c_dim, num_faces, num_tiles);            // :402
    set_packer_strides<untilize, tilize>(pack_src_format, tile_c_dim);  // :403
}
```

The three template parameters `<untilize, zero_output, tilize>` select which address-modifier and MOP layouts the packer programs into hardware:

- `untilize=false, tilize=false` (default): standard tiled pack. Dest is read in row-major within each face; faces are emitted F0, F1, F2, F3 in tile order.
- `untilize=true`: pack with untilization. Reads from Dest are strided so that the L1 output is row-major across the full tile (rows 0, 16, 1, 17, ..., interleaving the two halves of a 32-row tile).
- `tilize=true`: pack with tilization ([BH] workaround). Used to recover tile order from row-major Dest contents; takes a different replay-buffer path with `y_src` stride 4 and `y_dst` stride 2 (see `_llk_pack_configure_addrmod_` at `llk_pack.h:38–46`).

These three booleans are independent of `face_r_dim` and `num_faces` — they are orthogonal axes. Tiny tiles compose with all three, with one important restriction noted in §2.3.4.

## 2.3.2 MOP configuration for partial-row writes

`_llk_pack_mop_config_<>()` is the workhorse. For a standard tiled pack (the most common tiny-tile case), the relevant excerpt is:

```cpp
// tt_llk_blackhole/llk_lib/llk_pack.h:250
const uint PACK_INTF_SEL =
    face_r_dim == 1 ? p_pacr::SINGLE_INTF_ACTIVE
  : face_r_dim == 2 ? p_pacr::TWO_INTFS_ACTIVE
                    : p_pacr::ALL_INTF_ACTIVE;                    // :252-253

const uint MOP_INNER_LOOP = (face_r_dim < 4) ? 1 : (face_r_dim >> 2);  // :255
const uint MOP_OUTER_LOOP = num_faces * num_tiles;                     // :256
```

Three parameters change with tile geometry:

1. **`PACK_INTF_SEL`** — selects which packer interfaces (PACR engines) are active. With `face_r_dim == 1`, only a single PACR interface is enabled; the others are gated off so they do not consume Dest read bandwidth. `face_r_dim == 2` enables two; `face_r_dim ≥ 4` enables all four (one per face row group).

2. **`MOP_INNER_LOOP`** — for `face_r_dim < 4`, only one inner iteration runs per outer step (one PACK_RUN per face). For `face_r_dim ≥ 4`, the loop count is `face_r_dim >> 2` (one iteration per four rows). This is why `face_r_dim` is constrained to powers of two ∈ {1, 2, 4, 8, 16}: the inner-loop formula must produce a whole number, and the PACR hardware processes rows in groups of four.

3. **`MOP_OUTER_LOOP`** — `num_faces * num_tiles`. The packer steps once per face per tile, so an 8x32 tile (two faces) takes half as many outer iterations as a 32x32 tile (four faces).

The address-modifier configuration is set up by `_llk_pack_configure_addrmod_<untilize, tilize>()` at `llk_pack.h:20–69`. In standard mode (lines 47–69) the modifiers do this:

- `ADDR_MOD_0`: increment `y_src` and `y_dst` by 4 between inner-loop iterations (advance by 4 rows within a face).
- `ADDR_MOD_1`: clear `y_src`/`y_dst` and advance to the next face at outer-loop boundary.
- `ADDR_MOD_2`: cleanup at end of tile.

Because the inner-loop count is gated by `face_r_dim >> 2` and the per-iteration row stride is fixed at 4, the total rows packed per face is exactly `face_r_dim`. Nothing is written beyond the populated face rows, which is the whole point: for an 8x32 tile, the packer emits 8 rows × 32 cols, not 32 × 32 with zeros.

## 2.3.3 Stride programming and L1 byte count

`set_packer_strides<untilize, tilize>(pack_src_format, tile_c_dim)` (called at `llk_pack.h:403`) writes the PACR row and face strides for L1. These strides are how the packer knows the per-tile address footprint — i.e. how far to jump when finishing one tile and starting the next.

The stride is derived from the tile's L1 byte count, which is in turn the value returned by `tile.get_tile_size(dtype)` on the host side. For tiny tiles this shrinks proportionally with `num_faces × face_r_dim`. Concrete sizes:

| Tile shape | Format    | Datums | Raw bytes | L1 size (16B aligned) |
|------------|-----------|--------|-----------|------------------------|
| 32x32      | float32   | 1024   | 4096      | 4096                   |
| 32x32      | bfloat16  | 1024   | 2048      | 2048                   |
| 32x32      | bfp8      | 1024   | 1024 + 64 (exp) | 1088              |
| 16x32      | float32   | 512    | 2048      | 2048                   |
| 16x32      | bfloat16  | 512    | 1024      | 1024                   |
| 8x32       | float32   | 256    | 1024      | 1024                   |
| 8x32       | bfloat16  | 256    | 512       | 512                    |
| 8x32       | bfp8      | 256    | 256 + 32  | 288                    |
| 1x32       | bfloat16  | 32     | 64        | 64                     |
| 1x32       | bfp8      | 32     | 32 + 4    | 48 (next 16B mult)     |

Two notes on the table:

- **Block-float overhead.** BFP8 carries one 8-bit shared exponent per 16 datums (one per face row in a 16-wide face). For a full 32x32 BFP8 tile that yields 64 exponent bytes for 1024 mantissa bytes (= 1088 total, exactly `(1024 >> 4) + (64 >> 4) = 64 + 4 = 68` units of 16 bytes, matching the constant in `ckernel_defs.h:146`). For tiny tiles the exponent count shrinks with the row count: 8x32 BFP8 = 256 mantissa + 16 exponents per face × 2 faces = ~280 B, rounded to 288 B for 16B alignment.

- **16-byte L1 granularity.** All L1 allocations are 16-byte aligned. Any tile size that is not a multiple of 16 is rounded up by `get_tile_size()`. The 1x32 BFP8 case above (36 raw bytes → 48 aligned) is the only common case where this is actually visible. For float32 and bfloat16, every legal tiny-tile geometry is already a multiple of 16.

This byte count is what shows up everywhere downstream — CB page size, DRAM tile stride, `get_noc_addr()` offsets. Flash MLA threads it through explicitly:

```python
# micro_ops/flash_mla/op.py:536
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))  # Q_TILE_HEIGHT = 8
full_tile   = ttnn.Tile((K_TILE_HEIGHT, TILE_WIDTH))  # K_TILE_HEIGHT = 32
# ...
q_tile_size = q_tiny_tile.get_tile_size(q_df)         # :552
```

`q_tile_size` here is the actual L1 byte count for an 8x32 tile in the chosen data format. Q's CB is allocated as `num_pages × q_tile_size`; K and V CBs use `full_tile.get_tile_size(...)`.

## 2.3.4 Tilize mode and tiny tiles

Pack-tilize (the [BH] reverse-swizzle path) is gated by an assertion at `llk_pack.h:143`:

```cpp
// tt_llk_blackhole/llk_lib/llk_pack.h:143
ASSERT(face_r_dim == 2 || face_r_dim == 4 ||
       face_r_dim == 8 || face_r_dim == 16);
```

`face_r_dim == 1` is not supported in tilize mode. The reason is the replay-buffer construction at line 144:

```cpp
const uint replay_buf_len = face_r_dim - 1;                       // :144
// ...
const uint num_instrs_per_face = (face_r_dim >> 1) - 1;           // :154
```

The tilize MOP emits `face_r_dim − 1` instructions per face and then closes the face with a separate cleanup step. With `face_r_dim == 1` the replay buffer length would be zero and `(1 >> 1) - 1 = -1`, which is malformed. In practice, no tiny-tile op chain in tt-blaze uses pack-tilize with `face_r_dim < 2`; the lowest legal tilize geometry is 2x32 per face, e.g. 4x32 with two faces stacked.

Pack-untilize, by contrast, is supported for all legal `face_r_dim` and `num_faces` combinations. The PACK_INTF_SEL formula above (line 86 for the untilize path) adapts to the face row count the same way.

## 2.3.5 End-to-end configuration sequence

The matmul test harness calls the packer setup like this:

```cpp
// tests/sources/unpack_matmul_test.cpp:115
_llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
    formats.pack_src, formats.pack_dst,
    params.TILE_SIZE_PACK,
    params.in0_tile_r_dim < FACE_R_DIM
        ? params.in0_tile_r_dim
        : FACE_R_DIM,                                              // :120
    TILE_C_DIM,
    params.num_faces,
    params.PARTIAL_FACE_PACK);                                     // :122

_llk_pack_init_<false, false, false>(
    params.in0_tile_r_dim < FACE_R_DIM
        ? params.in0_tile_r_dim
        : FACE_R_DIM,                                              // :123
    TILE_C_DIM,
    params.num_faces);
```

Two points worth highlighting:

1. **`face_r_dim` clamp.** The expression `in0_tile_r_dim < FACE_R_DIM ? in0_tile_r_dim : FACE_R_DIM` collapses to the actual sub-face row count for tiny tiles (1, 2, 4, 8) and to `FACE_R_DIM = 16` for any tile with full 16-row faces. The packer always operates on a per-face row count, never a per-tile row count — when a 32x32 tile has `face_r_dim == 16`, two stacked 16-row faces make up its 32 rows.

2. **`PARTIAL_FACE_PACK`.** This flag (a different one than the unpacker's `partial_face`) tells `_llk_pack_hw_configure_` to program PACR for sub-16-row faces by clamping the per-face row counter. It is `true` whenever `face_r_dim < 16`.

The `TILE_SIZE_PACK` field passed to `_llk_pack_hw_configure_` is computed exactly the same way as the table in §2.3.3 — it is the per-tile L1 byte count of the output. The matmul test source computes it from the output tile geometry; production ops get it from `tile.get_tile_size(dtype)`. From the packer's perspective, this is the value used to compute inter-tile L1 strides; it is consumed by `set_packer_strides` and lives in PACR registers for the duration of the kernel.

## Summary

The packer's tiny-tile contract is symmetric with the unpacker's. `_llk_pack_init_<>()` takes `face_r_dim`, `tile_c_dim`, `num_faces` (and `num_tiles` on the [BH] overload), and from those derives PACK_INTF_SEL, MOP_INNER_LOOP, and MOP_OUTER_LOOP — the three knobs that change per geometry. Address modifiers stride row counters in groups of four within each face; only the populated rows are emitted to L1. The L1 byte count shrinks linearly with `num_faces × face_r_dim`, with BFP8 carrying proportionally fewer exponents and a 16-byte alignment round-up for the smallest geometries. Pack-untilize composes freely with tiny tiles; pack-tilize requires `face_r_dim ∈ {2, 4, 8, 16}`.

With unpacker, math, and packer all parameterized by the same `(face_r_dim, num_faces, face_c_dim)` triple, the next concerns are how Dest is laid out per tile (covered in Chapter 2, Section 4) and how throttle/DstSync interact with sub-32-row geometries (Chapter 2, Section 5).
