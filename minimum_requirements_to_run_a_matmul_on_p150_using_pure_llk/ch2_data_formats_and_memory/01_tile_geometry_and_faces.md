# Tile Geometry and Faces

## The Canonical 32x32 Tile

All data flowing through the Tensix Engine is organized into **tiles** -- fixed-size, two-dimensional blocks of elements. A tile is the fundamental quantum of data for unpack, math, and pack operations: every LLK API call operates on one or more tiles, never on individual scalars or arbitrary-length vectors. The canonical tile is 32 rows by 32 columns, containing 1024 elements.

The constants that define this geometry are declared in `tt_llk_blackhole/common/inc/ckernel_defs.h`:

```cpp
constexpr std::uint32_t FACE_HEIGHT = 16;
constexpr std::uint32_t FACE_WIDTH  = 16;
constexpr std::uint32_t TILE_HEIGHT = 32;
constexpr std::uint32_t TILE_WIDTH  = 32;

constexpr std::uint32_t FACE_R_DIM = FACE_HEIGHT;  // 16
constexpr std::uint32_t FACE_C_DIM = FACE_WIDTH;   // 16

constexpr std::uint32_t TILE_R_DIM = TILE_HEIGHT;  // 32
constexpr std::uint32_t TILE_C_DIM = TILE_WIDTH;   // 32
```

The `R_DIM` / `C_DIM` aliases are the names used most frequently throughout the LLK codebase -- as default parameters in init, configure, and runtime functions.

## Faces: The 16x16 Building Blocks

A tile is not stored as a flat 32x32 array. It is subdivided into **faces** -- 16x16 blocks of 256 elements each. The face is the hardware's fundamental processing unit: the FPU, the unpackers, and the Source registers all operate on face-sized chunks. The canonical 32x32 tile contains exactly four faces, arranged in a 2x2 grid:

```
         Col 0..15     Col 16..31
       +------------+------------+
Row    |            |            |
 0..15 |     F0     |     F1     |
       |            |            |
       +------------+------------+
Row    |            |            |
16..31 |     F2     |     F3     |
       |            |            |
       +------------+------------+
```

The number of faces is computed from the tile and face dimensions:

```cpp
constexpr std::uint32_t TILE_NUM_FACES = ((TILE_R_DIM * TILE_C_DIM) / (FACE_R_DIM * FACE_C_DIM));
// = (32 * 32) / (16 * 16) = 4
```

Within each face, elements are stored in **row-major order**: all 16 elements of row 0, then row 1, and so on through row 15. The four faces are stored sequentially in memory:

```
Memory offset:  [0..255]   [256..511]  [512..767]  [768..1023]
                   F0          F1          F2          F3
```

This is not the same as a conventional 32x32 row-major layout. In a standard row-major matrix, element (0, 16) would be at offset 16. In the tile/face layout, element (0, 16) belongs to F1 and is at offset 256. Understanding that a tile is four faces -- not a flat 32x32 matrix -- is essential for interpreting how the matmul kernel iterates over data.

The Python test infrastructure mirrors these constants in `tests/python_tests/helpers/tile_constants.py`:

```python
MAX_FACE_R_DIM = 16         # Maximum row dimension of a single face
FACE_C_DIM = 16             # Column dimension of a single face (always 16)

DEFAULT_TILE_R_DIM = 32     # Standard tile rows
DEFAULT_TILE_C_DIM = 32     # Standard tile columns

MAX_NUM_FACES_R_DIM = 2     # Maximum faces in row dimension
MAX_NUM_FACES_C_DIM = 2     # Maximum faces in column dimension
MAX_NUM_FACES = MAX_NUM_FACES_R_DIM * MAX_NUM_FACES_C_DIM  # = 4
```

The face column dimension is always 16 -- this is a hardware constraint. The face row dimension can vary (1, 2, 4, 8, or 16), enabling non-standard tile shapes.

## Non-Standard Tile Dimensions for MatMul

While the default tile is 32x32 with four 16x16 faces, the LLK matmul supports several non-standard tile geometries. These are expressed through two parameters: `num_faces` (how many faces the tile contains) and `face_r_dim` (how many rows each face has, when `partial_face` mode is enabled).

The supported tile sizes, as defined in `tile_constants.py`, are:

| Tile Shape | face_r_dim | num_faces | Face Layout | Faces Used |
|:-----------|:-----------|:----------|:------------|:-----------|
| 32x32 | 16 | 4 | 2 rows x 2 cols of 16x16 | F0, F1, F2, F3 |
| 16x32 | 16 | 2 | 1 row x 2 cols of 16x16 | F0, F1 |
| 32x16 | 16 | 2 | 2 rows x 1 col of 16x16 | F0, F2 |
| 16x16 | 16 | 1 | Single 16x16 face | F0 only |
| 8x32  | 8  | 2 | 1 row x 2 cols of 8x16 | partial F0, F1 |
| 4x32  | 4  | 2 | 1 row x 2 cols of 4x16 | partial F0, F1 |
| 2x32  | 2  | 2 | 1 row x 2 cols of 2x16 | partial F0, F1 |
| 1x32  | 1  | 2 | 1 row x 2 cols of 1x16 | partial F0, F1 |

In the unpack init function (`_llk_unpack_AB_matmul_init_` in `llk_unpack_AB_matmul.h`), these parameters appear explicitly:

```cpp
template <std::uint32_t kernel_broadcast_a = 0, std::uint32_t kernel_broadcast_b = 0>
__attribute__((always_inline)) inline void _llk_unpack_AB_matmul_init_(
    const std::uint32_t transpose       = 0,
    const std::uint32_t ct_dim          = 1,
    const std::uint32_t rt_dim          = 1,
    const std::uint32_t kt_dim          = 1,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,   // default: 16
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,   // default: 16
    const std::uint32_t unpA_num_faces  = 4,             // default: 4 (full 32x32)
    const std::uint32_t unpB_num_faces  = 4,             // default: 4
    const bool unpA_partial_face        = false,
    const bool unpB_partial_face        = false)
```

The `num_faces` parameter is constrained by an assertion to be 1, 2, or 4:

```cpp
LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4,
           "unpA_num_faces must be 1, 2, or 4");
```

When `partial_face = true`, the unpacker performs face-by-face unpacking with a custom `face_r_dim` (the row dimension of each face), rather than unpacking the entire tile at once. The `get_tile_params()` function in `tile_constants.py` computes the corresponding face parameters for each supported shape:

```python
# Examples from get_tile_params():
# [8, 32]  -> face_r_dim=8,  num_faces=2, num_faces_r_dim=1, num_faces_c_dim=2
# [32, 16] -> face_r_dim=16, num_faces=2, num_faces_r_dim=2, num_faces_c_dim=1
# [32, 32] -> face_r_dim=16, num_faces=4, num_faces_r_dim=2, num_faces_c_dim=2
```

## The 16x16 by 16x16 MatMul Restriction

There is one tile dimension combination that is explicitly forbidden: using 16x16 tiles (single-face) for **both** operands simultaneously. The assertion appears in both the math init and unpack init functions:

In `_llk_math_matmul_init_` (`llk_math_matmul.h`):

```cpp
// 16x16 inputs not supported - no dedicated math path;
// falls to 32x32 default which is incorrect for < 4 faces
LLK_ASSERT(
    !((in0_tile_r_dim == FACE_R_DIM) && (in0_tile_c_dim == FACE_C_DIM) &&
      (in1_tile_r_dim == FACE_R_DIM) && (in1_tile_c_dim == FACE_C_DIM)),
    "16x16 by 16x16 matmul is not supported");
```

In `_llk_unpack_AB_matmul_init_` (`llk_unpack_AB_matmul.h`):

```cpp
// 16x16 matmul not supported - no dedicated math path;
// falls to 32x32 default which is incorrect for < 4 faces
LLK_ASSERT(!(unpA_num_faces == 1 && unpB_num_faces == 1),
           "16x16 by 16x16 matmul is not supported");
```

The reason is architectural: the matmul MOP (Macro Operation) sequences -- the pre-recorded replay buffer instruction sequences and address modifier patterns -- are designed for tiles with at least two faces. A single-face-by-single-face matmul would require different address modifier configurations and replay buffer sequences that do not exist in the codebase. If attempted, the 32x32 default iteration pattern would read past the single face's data and produce incorrect results.

A 16x16 tile can still be used as **one** of the two operands. For example, multiplying a 16x32 input by a 32x16 input is valid and produces a 16x16 output.

## How MatMul Operates at the Face Level

The FPU's `MVMUL` instruction operates at a granularity smaller than a full face. As documented in the address modifier setup in `llk_math_matmul.h`:

```
// MVMUL does D = B*A

// Inner Loop --> 32/8 = 4 times for the full 32x16 face
// DEST -- 8 rows are calculated each time
// SRCB -- 8 rows are needed
// SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
// D[8,16] = B[8,16] * A[16,16]
```

Each `MVMUL` instruction computes:

$$D[8, 16] = B[8, 16] \times A[16, 16]$$

This means 8 rows from Source B are multiplied against a full 16x16 face from Source A, producing 8 rows of 16-wide output in the Destination register. The hardware internally pairs columns of A with rows of B for the dot product.

To compute a full 32x16 output face (32 output rows, 16 output columns), the inner loop must execute 4 times ($32 / 8 = 4$), each time advancing by 8 rows in Source B and 8 rows in the Destination register. Source A stays at the same 16x16 face throughout these 4 iterations.

For a complete 32x32 output tile, the kernel must process all four face pairs:

| Step | SrcB Face | SrcA Face | Dest Region |
|:-----|:----------|:----------|:------------|
| 1 | B-F0 (rows 0-15) | A-F0 (16x16) | D rows 0-15, cols 0-15 (output F0) |
| 2 | B-F0 (rows 0-15) | A-F1 (16x16) | D rows 0-15, cols 16-31 (output F1) |
| 3 | B-F2 (rows 16-31) | A-F0 (16x16) | D rows 16-31, cols 0-15 (output F2) |
| 4 | B-F2 (rows 16-31) | A-F1 (16x16) | D rows 16-31, cols 16-31 (output F3) |

The complete sequence of `MVMUL` instructions is recorded into the **replay buffer** -- a hardware instruction cache that can re-execute a recorded instruction sequence without the RISC-V core re-issuing each instruction. For a standard 32x32 tile with 4 face pairs and 4 inner iterations per face pair, the replay buffer contains 16 `MVMUL` instructions. The address modifiers (ADDR_MOD_0 through ADDR_MOD_5) control how the Source A, Source B, and Destination register pointers advance between successive `MVMUL` instructions. The full MOP construction and address modifier programming are covered in Chapter 4.

## Block Dimensions: ct_dim, rt_dim, kt_dim

When a matmul spans multiple tiles (as it typically does in any real workload), the operation is decomposed into a **tile block** described by three dimensions:

| Parameter | Name | Meaning |
|:----------|:-----|:--------|
| `ct_dim`  | Column-tile dimension | Number of output tiles in the column (horizontal) direction |
| `rt_dim`  | Row-tile dimension    | Number of output tiles in the row (vertical) direction |
| `kt_dim`  | K-tile dimension      | Number of tiles along the inner (accumulation) dimension |

For a matrix multiplication $C = A \times B$ where $A$ is $(M \times K)$ and $B$ is $(K \times N)$, with all dimensions in tiles of 32:

- $rt\_dim = M / 32$ -- how many tile-rows in the output
- $ct\_dim = N / 32$ -- how many tile-columns in the output
- $kt\_dim = K / 32$ -- how many tiles to accumulate along the inner dimension

For a simple single-tile matmul, all three default to 1.

The `_llk_math_matmul_` function uses these dimensions to control **operand reuse**:

```cpp
const bool reuse_a = ct_dim >= rt_dim;
```

When `ct_dim >= rt_dim`, the kernel holds one tile in Source A (the right-hand matrix in $D = B \times A$) and streams tiles through Source B -- this is "reuse A" mode. When `ct_dim < rt_dim`, the roles reverse. The inner loop iterates over the reuse dimension, and at the end of the inner loop sequence, the reused register is cleared and refilled with the next tile:

```cpp
const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim;
const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim;  // reuse dimension

for (std::uint32_t t = 0; t < t_dim; t++)
{
    for (std::uint32_t rut = 0; rut < rut_dim; rut++)
    {
        math::set_dst_write_addr<DstTileShape::Tile32x32>(
            dst_index + (reuse_a ? ct_dim * t + rut : t + rut * ct_dim));
        ckernel_template::run();
        if (rut == (rut_dim - 1))
        {
            TTI_SETRWC(p_setrwc::CLR_B, ...);  // or CLR_A
        }
    }
}
```

The `kt_dim` parameter controls the accumulation depth in the unpack kernel. It is stored in a general-purpose register for efficient address computation:

```cpp
TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(p_gpr_unpack::KT_DIM));
```

All three dimensions must be consistent between the unpack, math, and pack threads, and they must accurately reflect the actual tensor dimensions in L1 memory.

---

**Next:** [`02_data_format_pipeline.md`](./02_data_format_pipeline.md)
