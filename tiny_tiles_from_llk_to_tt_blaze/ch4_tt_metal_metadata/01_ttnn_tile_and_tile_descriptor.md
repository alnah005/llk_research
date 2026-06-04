# 4.1 `ttnn.Tile` and `TileDescriptor`: the host-side tile object

`ttnn.Tile` is the single place where a tile's logical geometry — its rows,
columns, face partitioning, narrowness, and partial-face status — is declared
on the host. Every downstream piece of TT-Metal metadata (CB page size,
shard-shape arithmetic, `TileDescriptor` for CB format descriptors, the
`(face_r_dim, num_faces)` pair surfaced to the LLK unpacker) is computed from
this one object. For tiny-tile work this is the API a kernel author sees
first: the moment you write `ttnn.Tile((8, 32))` you have committed the entire
downstream chain to an 8x32 tile.

**Prerequisites**

- Chapter 1, Sections 1–2 — canonical 32x32 tile, 16x16 faces, legal tiny-tile
  geometries (`num_faces ∈ {1, 2, 4}`, `face_r_dim ∈ {1, 2, 4, 8, 16}`).
- Familiarity with TT-Metal data formats (BFP8/BFP4 shared-exponent layout,
  Float16/Float32 dense layout) at the level of [[introduction_to_tt_llk]]
  Chapter 3.

---

## 4.1.1 The `Tile` constructor and the legal-shape lookup

The Python `ttnn.Tile` symbol binds to the C++ `tt::tt_metal::Tile` struct
declared in `tt_metal/api/tt-metalium/tile.hpp`. The constructor takes a
2-tuple `(H, W)` defaulted to `(32, 32)` and an optional `transpose_tile`
flag:

```cpp
// tt_metal/api/tt-metalium/tile.hpp:23-26
struct Tile {
    Tile(
        std::array<uint32_t, 2> tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH},
        bool transpose_tile = false);
    ...
};
```

Validation happens in `tile.cpp` against a fixed lookup table of 12 legal
`(tile_shape, face_shape)` pairs. The table is the authoritative list of
shapes the Tile object will accept:

```cpp
// tt_metal/impl/data_format/tile.cpp:19-34
constexpr std::array<std::array<std::array<uint32_t, 2>, 2>, 12> TILE_FACE_HW_CHOICES = {
    {// TODO: add other tile shapes once llk supported it
     {{{32, 32}, {16, 16}}},
     {{{16, 32}, {16, 16}}},
     {{{32, 16}, {16, 16}}},
     {{{16, 16}, {16, 16}}},
     // these shapes are not supported yet on llk, just for host loopback
     {{{8, 32}, {8, 16}}},
     {{{4, 32}, {4, 16}}},
     {{{2, 32}, {2, 16}}},
     {{{1, 32}, {1, 16}}},
     // these shapes are not supported yet on llk, just for host loopback
     {{{8, 16}, {8, 16}}},
     {{{4, 16}, {4, 16}}},
     {{{2, 16}, {2, 16}}},
     {{{1, 16}, {1, 16}}}}};
```

Two observations matter for tiny-tile authors:

1. The first four entries (32x32, 16x32, 32x16, 16x16) are the shapes the LLK
   data path can natively unpack, math, and pack on all three architectures
   today. The remaining eight entries — the `face_r_dim < 16` rows — are
   accepted by the host-side `Tile` object but the comment is explicit:
   "*not supported yet on llk, just for host loopback*". In practice this
   "host loopback" caveat is out of date for the row-direction tiny tiles
   used in production (1x32 / 2x32 / 4x32 / 8x32 are exercised by `unpack`
   and `math` matmul test suites — see Chapter 3). The narrow-width variants
   (`*x16`) remain less-traveled territory; if you are writing a new tiny-tile
   op, confirm coverage in the LLK test matrix before relying on them.
2. The lookup is `find_if` over the table, so any `(H, W)` not in the list
   throws `"Tile size is not valid for our hardware"`. There is no implicit
   rounding or padding — `ttnn.Tile((24, 32))` is a hard error, not a tile
   rounded up to 32x32.

The constructor body then derives the per-tile bookkeeping fields:

```cpp
// tt_metal/impl/data_format/tile.cpp:63-67
tile_hw = this->tile_shape[0] * this->tile_shape[1];
face_hw = face_shape[0] * face_shape[1];
num_faces = tile_hw / face_hw;
partial_face = static_cast<uint32_t>(this->tile_shape[0] < constants::TILE_HEIGHT);
narrow_tile = static_cast<uint32_t>(this->tile_shape[1] < constants::TILE_WIDTH);
```

The two flags `partial_face` and `narrow_tile` are the host-visible signal
that this tile is sub-canonical and that downstream LLK configuration must
take the partial-face path. `partial_face = 1` when `tile_shape[0] < 32`
(the height-direction tiny tiles); `narrow_tile = 1` when `tile_shape[1] < 32`
(the rare width-direction tiles).

### Worked table: derived fields for each legal shape

| `(H, W)` | `face_shape` | `tile_hw` | `face_hw` | `num_faces` | `partial_face` | `narrow_tile` | LLK-native? |
|---|---|---|---|---|---|---|---|
| (32, 32) | (16, 16) | 1024 | 256 | 4 | 0 | 0 | yes |
| (16, 32) | (16, 16) | 512  | 256 | 2 | 0 | 0 | yes |
| (32, 16) | (16, 16) | 512  | 256 | 2 | 0 | 1 | yes |
| (16, 16) | (16, 16) | 256  | 256 | 1 | 0 | 1 | yes |
| (8, 32)  | (8, 16)  | 256  | 128 | 2 | 1 | 0 | yes (tiny) |
| (4, 32)  | (4, 16)  | 128  | 64  | 2 | 1 | 0 | yes (tiny) |
| (2, 32)  | (2, 16)  | 64   | 32  | 2 | 1 | 0 | yes (tiny) |
| (1, 32)  | (1, 16)  | 32   | 16  | 2 | 1 | 0 | yes (tiny) |
| (8, 16)  | (8, 16)  | 128  | 128 | 1 | 1 | 1 | host-only |
| (4, 16)  | (4, 16)  | 64   | 64  | 1 | 1 | 1 | host-only |
| (2, 16)  | (2, 16)  | 32   | 32  | 1 | 1 | 1 | host-only |
| (1, 16)  | (1, 16)  | 16   | 16  | 1 | 1 | 1 | host-only |

The key invariant: for every tiny-tile shape in the second block,
`face_c_dim` stays pinned at 16 and `face_r_dim` shrinks. This matches the
LLK validator's rule (`face_c_dim == 16`, `face_r_dim ∈ {1, 2, 4, 8, 16}`,
`num_faces ∈ {1, 2, 4}`) documented in Chapter 1, Section 2.

The transpose path is also worth flagging:

```cpp
// tt_metal/impl/data_format/tile.cpp:57-61
if (transpose_tile) {
    TT_FATAL(
        (this->tile_shape[0] == constants::FACE_HEIGHT || this->tile_shape[0] == constants::TILE_HEIGHT),
        "Tile height must equal 16 or 32 in transpose mode");
}
```

`transpose_tile=true` is incompatible with `face_r_dim < 16` — tiny-tile
transposes (e.g., transposing an 8x32 Q in place) are not supported by the
host-side Tile object. Ops that need a tiny-tile transpose must transpose
via the kernel data path, not via the tile descriptor.

---

## 4.1.2 `get_tile_size(dtype)`: bytes per tile, BFP8 included

Once a `Tile` is constructed, its byte footprint in L1 for a given data
format is computed by `get_tile_size()`:

```cpp
// tt_metal/impl/data_format/tile.cpp:70-105
uint32_t Tile::get_tile_size(const DataFormat& format) const {
    uint32_t l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    uint32_t aligned_exp_size = tt::round_up(face_shape[0] * num_faces, l1_alignment);
    switch (format) {
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b: return (tile_hw / 4) + aligned_exp_size;
        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b: return (tile_hw / 2) + aligned_exp_size;
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b: return tile_hw + aligned_exp_size;
        case DataFormat::Float16:
        case DataFormat::Float16_b: return (tile_hw * 2);
        case DataFormat::Float32:   return (tile_hw * 4);
        ...
    }
}
```

The structure is the same for every dtype: a *datum* contribution plus, for
the block-floating-point formats, an aligned *shared-exponent* contribution.
The dense formats (Float16, Float32, Int*, UInt*) need only the datum
contribution.

### Dense formats: linear in `tile_hw`

For Float16 / Float32 the formula collapses to `tile_hw * bytes_per_datum`.
An 8x32 Float16 tile is `256 * 2 = 512 bytes` versus 32x32 at
`1024 * 2 = 2048 bytes` — a 4x reduction that maps directly to the row
ratio. This is the simplest case and is what the Flash MLA `cb_mask`
(8x32 bfloat16) and `cb_ms_in` (8x32 bfloat16 stats) CBs land on.

### BFP8 and friends: the shared-exponent overhead

For block-floating-point formats the byte count has two parts:

- **Datum bytes:** `tile_hw` (BFP8), `tile_hw / 2` (BFP4), `tile_hw / 4` (BFP2).
  Each datum is stored in the per-block mantissa table.
- **Shared-exponent bytes:** one 8-bit exponent per 16-element row in each
  face, totalling `face_shape[0] * num_faces` exponent bytes, **rounded up
  to L1 alignment (16 bytes on BH and WH-B0)**.

Concretely:

```
aligned_exp_size = round_up(face_shape[0] * num_faces, l1_alignment)
                 = round_up(face_r_dim * num_faces, 16)
```

This is where tiny tiles diverge from a naive "shrink everything by 4x"
expectation. Consider the canonical comparison:

| Shape | dtype | `tile_hw` | `face_shape[0] * num_faces` | `aligned_exp_size` | Total bytes |
|---|---|---|---|---|---|
| 32x32 | BFP8 | 1024 | 32 * 4 = 128 | round_up(128, 16) = 128 | 1024 + 128 = **1152** |
| 16x32 | BFP8 | 512  | 16 * 2 = 32  | round_up(32, 16)  = 32  | 512 + 32 = **544** |
| 8x32  | BFP8 | 256  | 8 * 2 = 16   | round_up(16, 16)  = 16  | 256 + 16 = **272** |
| 4x32  | BFP8 | 128  | 4 * 2 = 8    | round_up(8, 16)   = 16  | 128 + 16 = **144** |
| 2x32  | BFP8 | 64   | 2 * 2 = 4    | round_up(4, 16)   = 16  | 64  + 16 = **80** |
| 1x32  | BFP8 | 32   | 1 * 2 = 2    | round_up(2, 16)   = 16  | 32  + 16 = **48** |

(Note: `constants::BFLOAT8_B_TILE_HW = TILE_HW + 64 = 1088` in
`tt_metal/api/tt-metalium/constants.hpp:19`; the canonical 32x32 BFP8 tile
size assumes a 64-byte exponent payload, which matches the BH/WH path that
rounds 32x4 = 128 down to the canonical 64-byte exponent block on some
codepaths. The expression in `tile.cpp:72` is the authoritative formula
used by the host runtime — read your build's resolved value if there is a
discrepancy.)

Three things to notice:

1. **The 8x32 BFP8 tile is 272 bytes — a 4.24x reduction from 1152.** This
   is the load-bearing number for Flash MLA's L1 budget: every Q-side CB
   shrinks by ~4x, multiplied across 6+ tiny-tile CBs in the program.
2. **Below 8 rows, the exponent overhead stops shrinking.** The L1 alignment
   `round_up` clamps `aligned_exp_size` to a 16-byte minimum, so a 4x32 BFP8
   tile is 144 bytes (not the 136 a naive halving would predict), and a 1x32
   BFP8 tile is 48 bytes (not 18). For 1x32 BFP8 the exponent now costs 33%
   of the tile — by this point the format is being stretched.
3. **BFP4 and BFP2 follow the same alignment pattern.** The datum cost
   halves/quarters, but the exponent cost is the same `face_r_dim * num_faces`
   rounded up to 16 — so the alignment floor matters proportionally more for
   the lower-precision formats.

For the Flash MLA worked example below, the relevant computation is
`q_tiny_tile.get_tile_size(q_df)` where `q_df = bfloat16`: an 8x32 bfloat16
tile is `256 * 2 = 512` bytes. The mask CB and the stats CBs (also bfloat16)
share this size.

---

## 4.1.3 `TileDescriptor`: the CB-facing wrapper

A `Tile` is the full host-side object — it carries the face shape,
num-faces, transpose flags, and the dtype-dependent size method. A
`TileDescriptor` is the strip-down version that travels with a circular
buffer's format descriptor:

```cpp
// tt_metal/api/tt-metalium/program_descriptors.hpp:40-53
struct TileDescriptor {
    TileDescriptor() = default;
    TileDescriptor(const Tile& tile);
    TileDescriptor(uint32_t height, uint32_t width, bool transpose) :
        height(height), width(width), transpose(transpose) {}

    uint32_t height = constants::TILE_HEIGHT;
    uint32_t width = constants::TILE_WIDTH;
    bool transpose = false;

    bool operator==(const TileDescriptor& other) const {
        return height == other.height && width == other.width && transpose == other.transpose;
    }
};
```

Three fields: `height`, `width`, `transpose`. That is the entire payload that
flows into the CB descriptor surface. The face shape, `num_faces`,
`partial_face`, and `narrow_tile` are *not* carried on `TileDescriptor` —
they are re-derived on the consumer side (either by reconstructing a `Tile`
from `(height, width)` or by looking up the same `TILE_FACE_HW_CHOICES`
table).

The conversion constructor `TileDescriptor(const Tile& tile)` reads
`tile.get_height()` and `tile.get_width()` from the host `Tile` and copies
them into the descriptor's `height` and `width` fields. The `transpose`
field reflects whether the original `Tile` was constructed with
`transpose_tile=true`.

`TileDescriptor` then appears inside `CBFormatDescriptor`:

```cpp
// tt_metal/api/tt-metalium/program_descriptors.hpp:55-60
struct CBFormatDescriptor {
    uint8_t buffer_index = 0;
    tt::DataFormat data_format = tt::DataFormat::Float32;
    uint32_t page_size = 0;
    std::optional<TileDescriptor> tile;
};
```

The `tile` field is `std::optional` because raw row-major (non-tilized) CBs
can elide it; tilized CBs always set it. The pairing of
`(data_format, page_size, tile)` is what `cb_descriptor_from_sharded_tensor`
populates automatically — see Chapter 4, Section 2.

### Equality and propagation

The `operator==` on `TileDescriptor` is a structural compare across all three
fields. This matters for the propagation rule (Chapter 4, Section 3): two CBs
on either side of a producer-consumer handoff must have descriptors that
compare equal, or the kernel will see geometry it was not configured for.
The descriptor's narrow footprint makes this comparison cheap, but it also
means **the consumer side never sees `face_r_dim` or `num_faces` directly** —
those are reconstructed from `(height, width)` via the same legal-shape
table, so a producer that writes 8x32 and a consumer that expects 8x32 will
always agree on face shape because the lookup is deterministic.

---

## 4.1.4 Shard-shape arithmetic against `tile_shape[0]` and `tile_shape[1]`

The two raw fields kernel-host code reaches for most are
`tile.tile_shape[0]` (height) and `tile.tile_shape[1]` (width). These appear
in shard-shape math throughout production op-emit code. Two patterns
dominate:

**Pattern 1: derive tile dimension count from a shard.** Given a tensor
sharded as `(shard_h, shard_w)` rows × columns of physical elements, the
tile dimensions are:

```python
num_tile_rows_per_shard = shard_h // tile.tile_shape[0]
num_tile_cols_per_shard = shard_w // tile.tile_shape[1]
```

Flash MLA does this for the K-side standard tile (`St = S // K_TILE_HEIGHT`,
`DHt = DH // TILE_WIDTH`) and for the Q-side tiny tile
(`PNHt = num_q_heads_per_core // Q_TILE_HEIGHT`). When `Q_TILE_HEIGHT=8` and
`num_q_heads_per_core=8`, this yields `PNHt=1` — exactly one tile-row of Q
per core.

**Pattern 2: derive tile shape from a shard.** The inverse — given the shard
shape, construct the tile to match. RoPE does this:

```python
# models/demos/deepseek_v3_b1/micro_ops/rope/op.py (~line 121-122)
num_q_heads_per_core = shard_shape[0]
tile = ttnn.Tile((num_q_heads_per_core, ttnn.TILE_SIZE))
```

The shard's row count becomes the tile height directly. If the shard has 8
rows, the tile is 8x32; if it has 4 rows, the tile is 4x32. The kernel then
sees a tile geometry that exactly matches the data it will consume — no
padding, no masking — because the host-side declaration is parametric in the
shard.

Both patterns rely on `tile_shape[0]` being a small power-of-two that
divides the shard height. The constructor's validation against
`TILE_FACE_HW_CHOICES` enforces this implicitly: only `{1, 2, 4, 8, 16, 32}`
are legal heights, and shards are typically sized to one of these.

---

## 4.1.5 Worked example: Flash MLA's `q_tiny_tile`

Flash MLA's op-emit code is the canonical production example of `ttnn.Tile`
used for tiny tiles. The relevant block, abbreviated:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:409-413
q_tile = input_tensor_q.get_tile()
k_tile = input_tensor_k.get_tile()
Q_TILE_HEIGHT = q_tile.tile_shape[0]
K_TILE_HEIGHT = k_tile.tile_shape[0]
TILE_WIDTH = 32  # Width is always 32

# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:535-552
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))
full_tile = ttnn.Tile((K_TILE_HEIGHT, TILE_WIDTH))

# All intermediate/stats tiles use tiny tile dimensions (same as Q)
im_tile = q_tiny_tile
stats_tile = q_tiny_tile
# K uses full tiles (V read from K directly)
k_tile_obj = full_tile

# Create tile descriptors for CB setup
q_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)
stats_tile_descriptor = ttnn.TileDescriptor(stats_tile)

# Tile sizes - use tile.get_tile_size(dtype) for proper sizing
q_tile_size = q_tiny_tile.get_tile_size(q_df)
k_tile_size = k_tile_obj.get_tile_size(k_df)
mask_tile_size = q_tile_size
stats_tile_size = stats_tile.get_tile_size(stats_df)
```

Walking through this for the canonical DeepSeek V3 B1 configuration
(`num_q_heads_per_core = 8`, Q in bfloat16, K in BFP8):

1. **`Q_TILE_HEIGHT = 8`, `K_TILE_HEIGHT = 32`.** The Q tensor was sharded so
   that each core's shard has 8 attention-head rows; the K tensor was sharded
   in the standard 32-row alignment. The host pulls these out of the
   incoming tensors' tile descriptors — the tile geometry was committed
   upstream (by `create_q_heads` for Q, by the KV-cache reader for K) and
   Flash MLA only reads it back.
2. **`q_tiny_tile = ttnn.Tile((8, 32))`.** The constructor validates against
   the lookup, finds the `{{{8, 32}, {8, 16}}}` entry, sets `face_shape =
   (8, 16)`, computes `tile_hw = 256`, `face_hw = 128`, `num_faces = 2`,
   `partial_face = 1`, `narrow_tile = 0`. Q's tile is now a 2-face tile with
   8-row partial faces.
3. **`full_tile = ttnn.Tile((32, 32))`.** Standard 4-face canonical tile;
   `partial_face = 0`. K uses this directly.
4. **`q_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)`.** Strips Q down
   to `{height: 8, width: 32, transpose: false}` for the CB format
   descriptors. This is what every Q-consuming CB on the kernel side will
   compare against.
5. **`q_tile_size = q_tiny_tile.get_tile_size(bfloat16) = 256 * 2 = 512` bytes.**
   Versus `full_tile.get_tile_size(bfloat16) = 1024 * 2 = 2048` bytes — the
   4x reduction the tiny-tile path was introduced to capture.
6. **`stats_tile_size = stats_tile.get_tile_size(bfloat16) = 512` bytes**, same
   8x32 bfloat16 geometry as Q. The intermediate online-softmax statistics
   (m, s, output O accumulator) all ride the same tile shape so they can
   share CBs and aliasing patterns with Q — see Chapter 5, Section 2 for
   the full CB chain.

The result: a single line declaring `Q_TILE_HEIGHT=8` propagates through (a)
the Tile object, (b) the TileDescriptor for every Q/intermediate/stats CB,
(c) the BFP8/bfloat16 byte-size computation for each CB's page budget, and
(d) the `PNHt=1` tile-row count surfaced as a compile-time arg to the
kernel. The host-side `Tile` is the single source of truth; everything else
is derived.

This is the contract Chapter 4 builds on. The next section walks through
how `cb_descriptor_from_sharded_tensor` packages the `Tile` (plus the
tensor's buffer and shard spec) into a `CBDescriptor` ready for program
construction.
