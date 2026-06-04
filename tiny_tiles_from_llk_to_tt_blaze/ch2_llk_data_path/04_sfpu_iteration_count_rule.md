# 2.4 The SFPU Iteration Count Rule

SFPU (Special Function Processing Unit) operations process a tile **one 16x16 face at a time**. When you invoke an eltwise SFPU LLK such as `_llk_math_eltwise_unary_sfpu_sigmoid_<...>()`, the third template argument is an iteration count `N`, and it must equal the **number of 16-row faces** in the tile that the operation is consuming from the destination register. This file derives that rule, gives a lookup table for every legal tiny-tile geometry, and walks through the load-bearing example from the fused matmul kernel.

**Prerequisites:** Chapter 1 (tile / face / partial-face geometry), Chapter 2 Section 1 (the unpacker face-iteration model), Chapter 2 Section 3 (destination register occupancy).

## Why N exists at all

The SFPU is a 32-lane vector unit. One SFPU "call" processes one 16x16 face — sixteen rows of sixteen lanes each — that lives in the destination register. A tile with more than one face cannot be processed by a single SFPU call; the kernel has to iterate the SFPU operation `N` times, advancing the destination address by one face between iterations.

That address advance is exactly what `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` does. From `tt_llk/llk_lib/llk_math_eltwise_unary_sfpu.h:73–77`:

```cpp
// tt_llk/llk_lib/llk_math_eltwise_unary_sfpu.h:73
inline void _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_() {
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}
```

Two 8-byte increments per call: that's 16 bytes of destination address, which is the byte distance between consecutive face slots inside a tile's destination footprint. The SFPU params handler (in `llk_math_eltwise_unary_sfpu_params.h`) loops `N` times, executing the per-face SFPU instruction and then calling `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` between iterations.

So `N` is **not** "rows divided by 16" and it is **not** "number of tiles." It is the count of 16x16 faces the SFPU needs to walk across to cover the tile.

## The face-count formula

Recall from Chapter 1 (and `tensor_shape.h:70–72`) that a tile's total face count is

```
num_faces = num_faces_r_dim * num_faces_c_dim
```

with `face_c_dim` fixed at 16 in all legal geometries. For a tile of shape RxC:

- `num_faces_c_dim = C / 16` (so 32-wide tiles have 2 column-faces, 16-wide tiles have 1)
- `num_faces_r_dim = max(1, R / 16)` (sub-16-row tiles still have one row of faces — those faces are just shorter; `face_r_dim` becomes R)

The SFPU iteration count is

```
N = num_faces = num_faces_r_dim * num_faces_c_dim
```

A "tall and skinny" tile with R<16 still has the full column dimension's worth of faces — the sub-16-row geometries are partial-face tiles, but they remain *two* faces across when C=32. This is the single most counterintuitive part of the rule, and it's responsible for the worked example below.

## Lookup table

Every legal tiny-tile geometry (`num_faces ∈ {1, 2, 4}`, `face_r_dim ∈ {1, 2, 4, 8, 16}`, `face_c_dim = 16`) maps to a fixed `N`:

| Tile (RxC) | face_r_dim | num_faces_r_dim | num_faces_c_dim | num_faces | **N** |
| ---------- | ---------- | --------------- | --------------- | --------- | ----- |
| 1x32       | 1          | 1               | 2               | 2         | **2** |
| 2x32       | 2          | 1               | 2               | 2         | **2** |
| 4x32       | 4          | 1               | 2               | 2         | **2** |
| 8x32       | 8          | 1               | 2               | 2         | **2** |
| 16x32      | 16         | 1               | 2               | 2         | **2** |
| 32x32      | 16         | 2               | 2               | 4         | **4** |
| 16x16      | 16         | 1               | 1               | 1         | **1** |

Two observations worth internalizing:

1. **Every 32-wide tile, no matter how short, uses N=2.** A 1x32 tile is one row of 16 lanes wide twice over — F0 covers columns 0..15, F1 covers columns 16..31. Both faces exist; both must be visited. The SFPU does not care that 15 of the 16 rows inside each face are unused.
2. **N=4 is reserved for the full 32x32 tile**, and **N=1 only occurs for the 16x16 single-face geometry**. There is no tiny-tile path that lands on N=3.

The first observation is exactly why the fused-activation matmul kernel hardcodes `2` even when the output is 1x32.

## Worked example: fused sigmoid on a 1x32 output

The unified matmul kernel in `tt-blaze/unified_kernels/matmul.hpp` fuses an optional activation (sigmoid or SiLU) into the matmul tail. The relevant snippet (lines 170–178):

```cpp
// tt-blaze/unified_kernels/matmul.hpp:170
if constexpr (CTArgs::fuse_sigmoid) {
    PACK((ckernel::llk_math_eltwise_unary_sfpu_sigmoid<
              CTArgs::fused_activation_approx_mode, false, 2>(
              0, (int)VectorMode::R)));
}
```

The template parameters are `<approx_mode, is_fp32_dst_acc_en, N>`. The third one — `2` — is the iteration count.

Why `2` and not `1`? The output tile here is 1x32: a single row of 32 elements. From the lookup table that's `num_faces_r_dim = 1`, `num_faces_c_dim = 2`, **two faces**. The SFPU must process F0 (columns 0..15) and then F1 (columns 16..31) of that single-row tile. Hardcoding `1` here would silently leave the right half of the output (columns 16..31) un-sigmoid'd — the SFPU would advance through F0, never increment the dst address to F1, and the packer would later commit the raw matmul accumulator values for the right half.

The number `2` is a property of the **tile geometry**, not the output volume. It is independent of how many output rows the matmul produces and independent of the inner-product reduction depth.

## The PACK macro and why tiny tiles cost nothing here

Note the `PACK(...)` wrapper around the SFPU call. On Tensix, the math engine is split across threads: UNPACK (T0), MATH (T1), and PACK (T2). SFPU instructions can be issued from either MATH or PACK; for fused tail activations in a matmul kernel, the convention is to let the PACK thread drive the SFPU so that the MATH thread can move on to the next tile's MVMULs. The `PACK` macro guards the call so it compiles into the PACK thread only.

The important property is that **none of this changes when we switch to tiny tiles**. The macro, the function name, the template-argument order, the `VectorMode::R` argument — all of it is identical for a 32x32 output and a 1x32 output. The single thing that changes is the value of the third template argument, and that value is derived from compile-time tile geometry (`CTArgs::output_tile_shape` or equivalent), not from a runtime variable. There is no new SFPU dispatch, no new face-iteration loop, no per-tile fixup. The pattern composes.

## Common mistakes and how to spot them

- **Using N=1 for tall-skinny outputs.** If you see the right half of a 1x32, 2x32, 4x32, or 8x32 SFPU result come out wrong (PCC drops sharply, max-abs-diff localized to columns 16..31), check the SFPU template argument. The kernel almost certainly says `<..., false, 1>` where it should say `<..., false, 2>`.
- **Using N = R/16.** This integer-divides to 0 for tiny tiles, which most compilers will either reject or silently treat as a no-op. Either way, the SFPU stage drops out entirely.
- **Passing N as a runtime argument.** It is a template argument by design — the SFPU MOP and the dst-address-advance unrolling depend on `N` at compile time. Make sure your CTArgs / compile-time-args plumbing surfaces the face count, not a runtime tile pointer.

The safe pattern is to derive `N` once, in the host op-emit layer, from `output_tile.get_num_faces()` (or equivalently from `num_faces_r_dim * num_faces_c_dim`) and propagate it as a compile-time constant into the kernel's `CTArgs` struct. That way every SFPU call site in the kernel can simply spell `CTArgs::output_num_faces` and the geometry-to-N mapping is enforced in one place.
