# Tiny-Tile Definition and Legal Geometries

A **tiny tile** is any Tensix tile whose total row extent is less than the canonical 32 — i.e. a tile that uses fewer than four 16x16 faces, or whose faces have fewer than 16 rows of valid data, or both. The set of geometries the Tensix data path can actually consume is narrow, and it is defined by a single C++ validator function in the LLK common headers. This section quotes that validator, enumerates the legal `(num_faces, face_r_dim, face_c_dim)` combinations, lists the shapes seen in production, and explains the one important hardware restriction the validator does *not* catch: the "16x16 by 16x16 matmul is not supported" assertion in `_llk_unpack_AB_matmul_init_<>()`.

**Prerequisites:** Chapter 1, Section "Canonical Tile and Face Recap" (canonical 32x32 tile, four 16x16 faces F0-F3, row-major datum order). Background equivalent to [[introduction_to_tt_llk]] `ch3_data_organization/tiles_and_faces.md` is sufficient.

---

## 1. The Definition: What the Hardware Accepts

A `TensorShape` is the LLK common-header descriptor that replaces the older menagerie of `num_faces`, `face_r_dim`, `narrow_tile`, `partial_face`, and `VectorMode` parameters with a single 4-byte packed struct. It is declared in `tt_metal/tt-llk/common/tensor_shape.h`:

```cpp
// tt_metal/tt-llk/common/tensor_shape.h:44-74
struct __attribute__((packed)) TensorShape
{
    std::uint8_t face_r_dim;      ///< Row dimension of each face (typically 16)
    std::uint8_t face_c_dim;      ///< Column dimension of each face (always 16 for HW)
    std::uint8_t num_faces_r_dim; ///< Number of faces in row dimension
    std::uint8_t num_faces_c_dim; ///< Number of faces in column dimension

    constexpr std::uint16_t total_row_dim()   const { return face_r_dim   * num_faces_r_dim; }
    constexpr std::uint16_t total_col_dim()   const { return face_c_dim   * num_faces_c_dim; }
    constexpr std::uint16_t total_tensor_size() const { return total_row_dim() * total_col_dim(); }
    constexpr std::uint8_t  total_num_faces() const { return num_faces_r_dim * num_faces_c_dim; }
};
```

The default tile - the canonical 32x32 - is exactly the maximum of every field:

```cpp
// tt_metal/tt-llk/common/tensor_shape.h:78
constexpr TensorShape DEFAULT_TENSOR_SHAPE = {MAX_FACE_R_DIM, MAX_FACE_C_DIM, MAX_NUM_FACES_R_DIM, MAX_NUM_FACES_C_DIM};
// → {16, 16, 2, 2} → 32 rows x 32 cols x 4 faces
```

The validator is the **definitive specification of which tile geometries the tile-dependent op path will accept**:

```cpp
// tt_metal/tt-llk/common/tensor_shape.h:87-94
__attribute__((noinline)) bool validate_tensor_shape_tile_dependent_ops_(const TensorShape &tensor_shape)
{
    const std::uint8_t num_faces  = tensor_shape.total_num_faces();
    const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
    const std::uint8_t face_c_dim = tensor_shape.face_c_dim;
    return (num_faces == 1 || num_faces == 2 || num_faces == 4) &&
           (face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == 16) &&
           (face_c_dim == 16);
}
```

Three constraints, all simultaneously enforced:

- **`face_c_dim == 16`.** The column dimension of an individual face is *always* 16. This is hard-baked into the matrix unit, the source-register layout, and the packer's stride logic - there is no Tensix microarchitectural state that supports 8-wide or 32-wide faces.
- **`face_r_dim ∈ {1, 2, 4, 8, 16}`.** The row count of each face must be a power of two no larger than 16. This is the dimension the unpacker can shrink in `partial_face` mode. The MOP inner-loop counters are programmed in terms of `face_r_dim`, and the unpacker supports a small set of pre-validated counter values; intermediate values like 3, 5, 6, 7, 12, etc. are rejected.
- **`num_faces ∈ {1, 2, 4}`.** A tile contains 1, 2, or 4 faces total. With `num_faces_r_dim` and `num_faces_c_dim` each in `{1, 2}`, the legal `(rows, cols)` face-grid arrangements are `(1,1)`, `(1,2)`, `(2,1)`, and `(2,2)`.

Notably absent from the validator: an explicit ban on `face_r_dim` values like 3, 6, 12 - the validator simply doesn't enumerate them. There is also no joint check that `num_faces_r_dim` and `num_faces_c_dim` are themselves powers of two (only that their product is one of `{1, 2, 4}`). The function is intentionally conservative: it accepts a hardware-tested subset and rejects everything else, even shapes that might *seem* representable.

The comment above `MAX_NUM_FACES_R_DIM` in `tensor_shape.h:16-17` makes this explicit:

> "The current max constraints are set for large default size of 32x32, but that is until tensorShape is piped to all ops. Once it is piped to all ops, we can relax max number of faces, to be closer to description of a tensorshape."

In other words: the validator is gated on what has been *tested*, not what is theoretically expressible.

---

## 2. The Three Axes of Variation

There are three independent knobs in `TensorShape` once `face_c_dim=16` is fixed:

| Axis | Legal values | Controls |
|---|---|---|
| `face_r_dim` | 1, 2, 4, 8, 16 | Rows of valid data inside each face (the rest of the 16-row physical face is left undefined / partial-face mode). |
| `num_faces_r_dim` | 1, 2 | Whether the tile stacks one or two face-rows vertically. |
| `num_faces_c_dim` | 1, 2 | Whether the tile stacks one or two face-columns horizontally. |

The **effective tile height** is `face_r_dim * num_faces_r_dim` and the **effective tile width** is `face_c_dim * num_faces_c_dim = 16 * num_faces_c_dim`. With `num_faces = num_faces_r_dim * num_faces_c_dim ∈ {1, 2, 4}`, the cross-product of legal geometries (subject to the validator's `num_faces ∈ {1,2,4}` rule) is:

| `face_r_dim` | `num_faces_r_dim=1, num_faces_c_dim=1` | `num_faces=2` (1x2 or 2x1) | `num_faces_r_dim=2, num_faces_c_dim=2` |
|---|---|---|---|
| 1  | 1x16  | 1x32 *or* 2x16  | 2x32 |
| 2  | 2x16  | 2x32 *or* 4x16  | 4x32 |
| 4  | 4x16  | 4x32 *or* 8x16  | 8x32 |
| 8  | 8x16  | 8x32 *or* 16x16 | 16x32 |
| 16 | 16x16 | 16x32 *or* 32x16 | **32x32** (canonical) |

Every cell of this table passes `validate_tensor_shape_tile_dependent_ops_()`. The bold cell is the canonical 32x32 tile, the default. Every other cell is a tiny tile.

Three useful sanity checks fall out of this table:

1. **A 16x32 tile can be expressed two ways.** Either `(face_r_dim=16, num_faces_r_dim=1, num_faces_c_dim=2)` (1 face-row of 2 faces) or `(face_r_dim=8, num_faces_r_dim=2, num_faces_c_dim=2)` (2 face-rows of 2 partial-faces each). These are **not equivalent at the LLK level** - the unpacker MOP and the SFPU face-iteration count differ. The producer op must declare the geometry that matches the physical face count it intends to write.
2. **The maximum effective height is 32** (`face_r_dim=16` × `num_faces_r_dim=2`). The hardware cannot represent a 64-row tile - that has to be expressed as two separate tiles in L1.
3. **The maximum effective width is 32** (`face_c_dim=16` × `num_faces_c_dim=2`). There is no 64-column tile; wide tensors are tiled along the column axis.

---

## 3. Practically Observed Geometries

The set above is much larger than what production code actually uses. In real TT-Metal / TT-Blaze programs we typically see only a handful of shapes:

| Shape | TensorShape | num_faces | Where it appears | Notes |
|---|---|---|---|---|
| **32x32** | `{16,16,2,2}` | 4 | Everywhere by default | Canonical tile, full data path. |
| **8x32** | `{8,16,1,2}` | 2 | Flash MLA Q/mask/stats CBs (DeepSeek V3 B1) | Primary tiny-tile production case - 8 attention heads per core in batch-1 decode. |
| **16x32** | `{16,16,1,2}` | 2 | Half-tile intermediates in fused chains | Common when the row dim halves between stages. |
| **4x32** | `{4,16,1,2}` | 2 | Narrow batch / partial-prefill | Used when sequence chunking lands on row counts < 8. |
| **2x32** | `{2,16,1,2}` | 2 | Narrow batch | Same family as 4x32. |
| **1x32** | `{1,16,1,2}` | 2 | Sampling outputs, argmax-of-row, scalar broadcast | A single row of valid data; SFPU still iterates over 2 column faces. |
| **32x16** | `{16,16,2,1}` | 2 | Column-reduced intermediate | Rare; appears when an op reduces along the column axis. |
| **16x16** | `{16,16,1,1}` | 1 | Column-reduced single-face | **Rare and special-cased** - illegal as a matmul operand (see Section 5). |

The 8x32 case is the load-bearing one. From the DeepSeek V3 B1 Flash MLA micro-op:

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:412-459
Q_TILE_HEIGHT = q_tile.tile_shape[0]                # 8
K_TILE_HEIGHT = k_tile.tile_shape[0]                # 32
...
PNHt = num_q_heads_per_core // Q_TILE_HEIGHT        # 8 // 8 = 1
```

```python
# models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py:538
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))  # (8, 32)
```

At LLK declaration time this becomes `TensorShape{face_r_dim=8, face_c_dim=16, num_faces_r_dim=1, num_faces_c_dim=2}`: one face-row of two 8x16 partial-faces. The Q activation, the mask CB, and the online-softmax stats CBs (m, l) all use this shape; the K / V tiles remain at 32x32 because they are weight-side and large-side along the rt dimension. The L1 saving is substantial - each 8x32 BF16 tile is ~512 bytes versus ~2,048 bytes for a padded 32x32 - and the same 4x ratio applies to the dest-budget computation on the producer side.

The 1x32 shape shows up in fused activation tails and in sampling output. The SFPU still iterates over `N=2` faces (two column-faces of 16 cols each, with only the top row carrying live data); a `<approx_mode, false, 2>` template argument is correct for it, *not* `N=1`. This is covered in detail in Chapter 2, Section "SFPU Iteration Count Rule".

---

## 4. Hardware vs API Legality

Passing `validate_tensor_shape_tile_dependent_ops_()` is a *necessary* condition for a tile to flow through the LLK data path. It is not always *sufficient*. A geometry can be API-legal at the TensorShape level but trap inside the unpacker, math, or packer because that particular `(num_faces, face_r_dim)` combination has no dedicated hardware path for the operation in question. Two cases:

- **Op-specific assertions inside `_llk_unpack_*_init_<>()`** can reject shapes that the generic validator allows. The matmul case (next section) is the canonical example.
- **Datatype/face-c-dim coupling.** Some packer paths key off `tile_size_bytes`, and certain `(face_r_dim, face_c_dim, dtype)` combinations land on byte counts that aren't a multiple of the 16-byte L1 alignment quantum. These are not always caught at compile time; they manifest as malformed L1 writes at runtime.

Per-architecture status:

- **[BH]** Full support for the validator's geometry set in `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h`, `llk_math_matmul.h`, and `llk_pack.h`. All shapes in the table in Section 3 are exercised by `tests/sources/unpack_matmul_test.cpp` and `math_matmul_test.cpp` (with the 16x16-by-16x16 matmul exception).
- **[WH]** Functionally equivalent support in `tt_llk_wormhole_b0/llk_lib/`. Minor API differences in how `partial_face` is wired through `_llk_unpack_AB_init_<>()` versus Blackhole - the validator and assertion set are the same.
- **[Q]** Status unclear for tiny-tile matmul. The Quasar test harness `tests/sources/matmul_quasar_test.cpp` does not exercise sub-32-row tiles, and the Quasar LLK lib uses semantic file naming (`llk_unpack_unary_operand.h`, `llk_unpack_binary_operands.h`) that has diverged from the WH/BH letter-based layout. The validator constants are present, but the unpack/math/pack triple has not been verified end-to-end for `face_r_dim < 16`.

The general principle: **the validator describes the API contract; the per-op assertions describe what is actually wired up**. The two diverge in exactly one well-known place.

---

## 5. The 16x16-by-16x16 Matmul Assertion

The single most important hardware-vs-API gotcha lives in `_llk_unpack_AB_matmul_init_<>()`:

```cpp
// tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h:200-203
LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");
// 16x16 matmul not supported - no dedicated math path; falls to 32x32 default which is incorrect for < 4 faces
LLK_ASSERT(!(unpA_num_faces == 1 && unpB_num_faces == 1), "16x16 by 16x16 matmul is not supported");
```

The first two assertions restate the validator: matmul accepts 1-, 2-, or 4-face operands on each side. The third is the new constraint: **both operands cannot simultaneously be 1-face tiles**. The in-source comment is precise about why: a 1-face × 1-face matmul has no dedicated math-engine code path; control flow falls through to the default 32x32 inner-loop structure, which assumes four faces and produces incorrect output if either operand has fewer.

Concretely, the math-engine MOP for a default 2x2-face × 2x2-face matmul issues an MVMUL sequence shaped for 4-face output accumulation:

```
F0_out = A_F0 * B_F0 + A_F1 * B_F2
F1_out = A_F0 * B_F1 + A_F1 * B_F3
F2_out = A_F2 * B_F0 + A_F3 * B_F2
F3_out = A_F2 * B_F1 + A_F3 * B_F3
```

If `unpA_num_faces == 1` and `unpB_num_faces == 1`, both inputs hold only F0; A_F1, A_F2, A_F3, B_F1, B_F2, B_F3 are all garbage. The math engine still issues the full 4-output-face inner loop, accumulating garbage × garbage into F1_out, F2_out, F3_out. There is no `partial_face` path on the math side equivalent to the unpacker's - the math MOP cannot conditionally shrink its loop count for 1-face inputs. Hence the assert.

The combinations that **are** legal for matmul:

| A | B | Result | Notes |
|---|---|---|---|
| 4-face (32x32) | 4-face (32x32) | 32x32 | Canonical case. |
| 4-face (32x32) | 2-face (32x16 or 16x32) | matches B | Common in column-reduced output. |
| 4-face (32x32) | 1-face (16x16) | 16x16 partial | Legal because A has > 1 face. |
| 2-face (16x32) | 4-face (32x32) | 16x32 | Primary tiny-tile case (Q × Kᵀ in Flash MLA). |
| 2-face (16x32) | 2-face (32x16) | 16x16 | Legal: at least one input > 1 face. |
| 1-face | 4-face | matches B | Legal. |
| **1-face** | **1-face** | **-** | **Asserts. Not supported.** |

The Flash MLA Q × Kᵀ matmul is precisely the "A is 2-face, B is 4-face" row of this table: Q is 8x32 (declared as `face_r_dim=8, num_faces_c_dim=2` - 2 column-faces of partial-row data, counted by the assertion as `num_faces = 2`), and K is the canonical 32x32. This lands inside the legal region with clearance to spare.

The pragmatic rule: **at least one matmul operand must have `num_faces ≥ 2`**. When designing a new tiny-tile micro-op, the easiest guarantee is to keep weight-side tiles at 32x32 and only shrink the activation-side tile. If both sides genuinely shrink, the operand with the larger reduction axis (K) should keep its full face count; if even that's impossible, the op must be expressed as a pair of dot-products on the SFPU rather than a matmul.

---

## Summary

The legal tiny-tile geometry space is defined by exactly three rules - `face_c_dim = 16`, `face_r_dim ∈ {1,2,4,8,16}`, `num_faces ∈ {1,2,4}` - and is encoded once, in `validate_tensor_shape_tile_dependent_ops_()`. The practical set of geometries seen in production is far narrower than the validator's full enumeration: 8x32 dominates (Flash MLA Q-side), 32x32 is the default, and 1x32 / 2x32 / 4x32 / 16x32 appear as fused-chain intermediates. One important hardware-side restriction sits outside the validator: matmul forbids 1-face × 1-face operand pairs, because the math MOP has no shrunk inner loop for that case. Chapter 2 picks up from here and walks through what changes inside each LLK thread - unpack, math, pack, SFPU - when one of these tiny shapes flows through the data path.
