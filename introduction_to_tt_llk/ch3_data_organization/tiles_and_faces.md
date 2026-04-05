# Tiles and Faces

## The Tile: Fundamental Data Unit

Every LLK operation works on **tiles**. A standard tile is a 32x32 block of datums (1024 datums total). The Tensix Engine's unpackers, FPU, and packers are all designed around this tile granularity.

The tile dimensions are defined as constants in `common/tensor_shape.h`:

```cpp
constexpr std::uint8_t MAX_TILE_R_DIM = 32;  // tile row dimension
constexpr std::uint8_t MAX_TILE_C_DIM = 32;  // tile column dimension
```

## Faces

Each tile is subdivided into four **faces**. A face is a 16x16 block of datums (256 datums). The four faces are labeled F0 through F3 and are arranged in row-major order within the tile:

```
+--------+--------+
|   F0   |   F1   |
| (16x16)| (16x16)|
+--------+--------+
|   F2   |   F3   |
| (16x16)| (16x16)|
+--------+--------+
```

Face dimension constants from `common/tensor_shape.h`:

```cpp
constexpr std::uint8_t MAX_FACE_R_DIM      = 16;  // face row dimension
constexpr std::uint8_t MAX_FACE_C_DIM      = 16;  // face column dimension
constexpr std::uint8_t MAX_NUM_FACES_R_DIM = 2;   // faces per tile in row direction
constexpr std::uint8_t MAX_NUM_FACES_C_DIM = 2;   // faces per tile in column direction
constexpr std::uint8_t MAX_NUM_FACES       = MAX_NUM_FACES_R_DIM * MAX_NUM_FACES_C_DIM;  // = 4
```

Within each face, datums are stored in **row-major order**: all 16 elements of row 0, then all 16 elements of row 1, and so on.

## The `TensorShape` Struct

The `TensorShape` struct (defined in `common/tensor_shape.h`) provides a unified description of tile geometry. It replaces older, inconsistent parameters such as `num_faces`, `face_r_dim`, `narrow_tile`, `partial_face`, and `VectorMode`.

```cpp
struct __attribute__((packed)) TensorShape
{
    std::uint8_t face_r_dim;      // Row dimension of each face (typically 16)
    std::uint8_t face_c_dim;      // Column dimension of each face (always 16 for HW)
    std::uint8_t num_faces_r_dim; // Number of faces in row dimension
    std::uint8_t num_faces_c_dim; // Number of faces in column dimension
};
```

The struct is exactly 4 bytes (`static_assert(sizeof(TensorShape) == 4)`). It provides helper methods:

| Method | Computation | Example (32x32 tile) |
|:-------|:------------|:---------------------|
| `total_row_dim()` | `face_r_dim * num_faces_r_dim` | 16 * 2 = 32 |
| `total_col_dim()` | `face_c_dim * num_faces_c_dim` | 16 * 2 = 32 |
| `total_tensor_size()` | `total_row_dim() * total_col_dim()` | 32 * 32 = 1024 |
| `total_num_faces()` | `num_faces_r_dim * num_faces_c_dim` | 2 * 2 = 4 |

The default tile shape is:

```cpp
constexpr TensorShape DEFAULT_TENSOR_SHAPE = {
    MAX_FACE_R_DIM, MAX_FACE_C_DIM, MAX_NUM_FACES_R_DIM, MAX_NUM_FACES_C_DIM
};
// Equivalent to: {16, 16, 2, 2} -> 32x32 tile with 4 faces
```

## Tile Dimension Examples

`TensorShape` can describe tiles smaller than 32x32. Some examples:

| Tile Shape | `face_r_dim` | `face_c_dim` | `num_faces_r_dim` | `num_faces_c_dim` | Total |
|:-----------|:-------------|:-------------|:-------------------|:-------------------|:------|
| 32x32 | 16 | 16 | 2 | 2 | 1024 |
| 32x16 | 16 | 16 | 2 | 1 | 512 |
| 16x16 | 16 | 16 | 1 | 1 | 256 |

## Tiny Tiles (Non-32x32)

Not all operations use the full 32x32 tile. The `num_faces` parameter (the total face count, which can be 1, 2, or 4) and `face_r_dim` (which can be 1, 2, 4, 8, or 16) allow describing smaller tiles. A validation function enforces these constraints:

```cpp
bool validate_tensor_shape_tile_dependent_ops_(const TensorShape &tensor_shape)
{
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 ||
            face_r_dim == 8 || face_r_dim == 16) &&
           (face_c_dim == 16);
}
```

Key constraints:
- `face_c_dim` is always 16 (a hardware requirement).
- `face_r_dim` must be a power of two from 1 to 16.
- `num_faces` (total) must be 1, 2, or 4.

A "partial face" is one where `face_r_dim` is less than 16. This is used for operations that process fewer rows, such as certain reduction or broadcast patterns.

## Row-Major Order vs. Tile Order in L1

Input tensors can be stored in L1 memory in two layouts:

**Row-major order** stores datums sequentially row by row across the full tensor width. This is the standard linear layout. For a 32x64 tensor, the first 64 values in memory are row 0, the next 64 are row 1, and so on.

**Tile order** partitions the tensor into 32x32 tiles, and within each tile the data is stored face by face (F0, F1, F2, F3). Within each face, data is in row-major order. For a 32x64 tensor, L1 would contain: tile(0,0) [F0, F1, F2, F3], then tile(0,1) [F0, F1, F2, F3].

Tile order is the native format for the Tensix Engine. The LLK unpack operations `llk_unpack_tilize` and `llk_unpack_untilize` convert between these two layouts during the unpack step. Most compute operations expect input data already in tile order.

---

**Next:** [`data_formats.md`](./data_formats.md)
