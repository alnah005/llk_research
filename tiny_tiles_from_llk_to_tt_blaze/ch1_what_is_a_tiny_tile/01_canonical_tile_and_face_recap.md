# Canonical Tile and Face Recap

The canonical Tensix tile is a 32x32 block of elements subdivided into four 16x16 **faces** (F0–F3). Every piece of LLK infrastructure — unpacker descriptors, dest-register addressing, packer stride logic, the matrix-unit MVMUL — is built around this two-level structure. Before we can talk about *tiny* tiles, we need a precise picture of what the *canonical* tile looks like and why 16x16 is the hardware granule.

**Prerequisites:** familiarity with matrix-multiply terminology. No prior Tensix knowledge required; readers wanting deeper background should consult [[introduction_to_tt_llk]] (ch3_data_organization/tiles_and_faces.md), which this section recaps.

## The 32x32 tile

A tile is the smallest unit the Tensix engines (unpacker, math unit, packer) address as a whole. Its dimensions are fixed at compile time by the constants in `ckernel_defs.h`:

```cpp
// tt_llk_blackhole/common/inc/ckernel_defs.h:86-97
constexpr uint32_t FACE_HEIGHT      = 16;
constexpr uint32_t FACE_WIDTH       = 16;
constexpr uint32_t TILE_HEIGHT      = 32;
constexpr uint32_t TILE_WIDTH       = 32;
constexpr uint32_t FACE_R_DIM       = 16;
constexpr uint32_t FACE_C_DIM       = 16;
constexpr uint32_t TILE_R_DIM       = 32;
constexpr uint32_t TILE_C_DIM       = 32;
constexpr uint32_t TILE_NUM_FACES   = (TILE_R_DIM * TILE_C_DIM) / (FACE_R_DIM * FACE_C_DIM); // = 4
```

The full set of "canonical" dimension constants used throughout the LLK is summarized below.

| Constant            | Value | Meaning                                            |
|---------------------|-------|----------------------------------------------------|
| `TILE_R_DIM`        | 32    | Rows per tile                                      |
| `TILE_C_DIM`        | 32    | Columns per tile                                   |
| `FACE_R_DIM`        | 16    | Rows per face                                      |
| `FACE_C_DIM`        | 16    | Columns per face                                   |
| `TILE_NUM_FACES`    | 4     | Faces per canonical tile (2 row-faces x 2 col-faces) |
| `MAX_FACE_R_DIM`    | 16    | Upper bound on `face_r_dim` in `TensorShape`       |
| `MAX_FACE_C_DIM`    | 16    | Upper bound on `face_c_dim` (always 16 in practice)|
| `MAX_NUM_FACES_R_DIM` | 2   | Upper bound on row-face count                      |
| `MAX_NUM_FACES_C_DIM` | 2   | Upper bound on column-face count                   |

## Face arrangement (F0–F3)

The four faces are laid out in row-major order within the 32x32 tile:

```
+--------+--------+
|   F0   |   F1   |
| (16x16)| (16x16)|
+--------+--------+
|   F2   |   F3   |
| (16x16)| (16x16)|
+--------+--------+
```

F0 occupies rows 0–15, columns 0–15; F1 occupies rows 0–15, columns 16–31; F2 occupies rows 16–31, columns 0–15; F3 occupies rows 16–31, columns 16–31. **Within each face, elements are stored row-major**: all 16 elements of face-row 0, followed by all 16 elements of face-row 1, and so on for 16 rows. The 16-row x 16-col face is therefore a contiguous 256-element block in L1.

This two-level structure (tile composed of faces, face composed of rows) is reflected in the `TensorShape` struct that parameterizes every tile-dependent LLK call:

```cpp
// tt_metal/tt_metal/tt-llk/common/tensor_shape.h:44-73
struct TensorShape {
    std::uint8_t face_r_dim;
    std::uint8_t face_c_dim;
    std::uint8_t num_faces_r_dim;
    std::uint8_t num_faces_c_dim;

    constexpr std::uint16_t total_row_dim()    const { return face_r_dim     * num_faces_r_dim; }
    constexpr std::uint16_t total_col_dim()    const { return face_c_dim     * num_faces_c_dim; }
    constexpr std::uint16_t total_tensor_size() const { return total_row_dim() * total_col_dim(); }
    constexpr std::uint8_t  total_num_faces() const { return num_faces_r_dim * num_faces_c_dim; }
};
```

For the canonical 32x32 tile, the shape is `{face_r_dim=16, face_c_dim=16, num_faces_r_dim=2, num_faces_c_dim=2}` — which the header exposes as the default:

```cpp
// tt_metal/tt_metal/tt-llk/common/tensor_shape.h:78
constexpr TensorShape DEFAULT_TENSOR_SHAPE = {16, 16, 2, 2};
```

Plugging into the helpers: `total_row_dim() = 16 * 2 = 32`, `total_col_dim() = 16 * 2 = 32`, `total_num_faces() = 2 * 2 = 4`.

## Why 16x16 is the hardware granule

The 16x16 face size is not an arbitrary software choice — it is the granularity of the Tensix matrix-vector multiply unit (MVMUL). A single MVMUL issue consumes one 16-row face from each of two source operands and produces one 16-row face of result. This dictates three concrete properties of tile geometries:

1. **`face_c_dim` is always 16.** The column dimension of a face matches the width of the MVMUL datapath; reducing it would leave hardware lanes idle and is not supported by the source-register layout. Throughout the LLK, `face_c_dim` is treated as a constant 16.
2. **`face_r_dim` can shrink but not grow.** The 16 in `MAX_FACE_R_DIM` reflects the maximum row count the unpacker fills into a single face slot in srcA/srcB. Smaller `face_r_dim` (e.g., 8, 4, 2, 1) means the unpacker simply writes fewer rows into the same face slot — the math unit then processes a shorter face.
3. **Face count is capped at 4.** With `MAX_NUM_FACES_R_DIM = 2` and `MAX_NUM_FACES_C_DIM = 2`, no tile geometry can exceed the canonical 4-face layout. Tiny tiles only ever *reduce* the face count (to 1 or 2), never increase it.

The packer's stride logic and the dest-register's banked layout are similarly organized around 16-row faces: the packer walks four face slots per tile, and a single dest tile occupies eight rows of the dest register because each row holds two face-rows (the face row is striped across columns of the bank). Anything that changes within a tile — fewer rows, fewer faces — must be expressible as "use a subset of the 16x16 slots that already exist," not "redesign the slot."

With this canonical layout fixed, the question for the rest of Chapter 1 becomes: which subsets are *legal* (see Chapter 1, "Tiny Tile Definition and Legal Geometries"), and what motivates allowing them at all (see Chapter 1, "Motivation: Batch-1 Decode Activations")?
