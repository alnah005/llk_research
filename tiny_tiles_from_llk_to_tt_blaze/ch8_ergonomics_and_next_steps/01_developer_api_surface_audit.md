# Developer API Surface Audit

Adding a tiny-tile op to a production pipeline today requires the author to declare the tile geometry in three independent places — Python host code, the C++ compile-time arg struct, and the SFPU iteration count inside the kernel body — and to keep all three in sync by hand. None of the three places knows about the others at compile time, and the canonical legal-geometry specification lives in an LLK header that no developer-facing doc references. This file audits the surface, names the keep-in-sync hazard, and proposes three concrete changes that would reduce friction without changing the hardware path or the LLK.

**Prerequisites:** Chapter 1 (tiny-tile geometry, `face_r_dim` / `num_faces` model), Chapter 4 (TT-Metal CB descriptors and `TileDescriptor` plumbing), Chapter 5 (Flash MLA / RoPE production usage of 8x32 Q tiles), and Chapter 7 (trade-offs and operational symptoms of geometry mismatch).

The cost of getting any of the three declarations wrong is asymmetric: page-size mismatches surface immediately as `cb_wait_front` hangs, but a wrong SFPU iteration count silently leaves un-processed rows in the destination register, producing PCC divergence that looks like a numerics bug rather than a geometry bug. Chapter 7 catalogued these failure modes; this chapter asks what the API would have to look like for them to be unrepresentable.

---

## The Three-Place Declaration

A single tiny-tile op currently requires the author to touch three layers, in roughly this order:

**Layer 1 — Python tile construction (host).** The op emit code constructs a `ttnn.Tile` with the chosen geometry. From Flash MLA, `models/.../flash_mla/op.py:538`:

```python
# models/.../flash_mla/op.py:538
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))   # Q_TILE_HEIGHT = 8
```

**Layer 2 — CT-arg / CB descriptor (host → kernel boundary).** The Python tile is wrapped in a `TileDescriptor` so the CB descriptor knows the per-page layout. `op.py:548`:

```python
# models/.../flash_mla/op.py:548
q_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)
```

This object eventually becomes a field on the CB descriptor and influences `page_size` and `num_pages`. The CT-arg struct that the kernel reads is generated from the same op emit, but the kernel author is responsible for ensuring its `TILE_HEIGHT` field matches `q_tiny_tile.tile_shape[0]`.

**Layer 3 — Kernel-body SFPU iteration count (device).** Inside the kernel, every SFPU op processes the destination register face-by-face. The number of 16-row faces is a template argument on the LLK call. From `unified_kernels/matmul.hpp:173–176`:

```cpp
// unified_kernels/matmul.hpp:173-176
// 8x32 Q tile = 2 faces of 16x8 each → N = 2
PACK((ckernel::llk_math_eltwise_unary_sfpu_sigmoid<
        CTArgs::fused_activation_approx_mode, false, 2>(
        0, (int)VectorMode::R)));
```

The `2` is hand-derived from the fact that an 8x32 tile decomposes into two faces along the column axis. Nothing in the type system enforces it; nothing in the kernel knows that the bound CB's `TileDescriptor` is `(8, 32)`. If the author later changes `Q_TILE_HEIGHT` to 16, Layer 1 and Layer 2 update via the named constant, but Layer 3 still says `2` and the upper face is now silently skipped.

The keep-in-sync cost grows linearly with the number of SFPU calls in the kernel — Flash MLA has the sigmoid above plus several other fused-activation paths, each carrying its own hardcoded `N`. Chapter 5 noted that the RoPE op (`rope/op.py:113–127`) derives tile geometry from the input shard shape, which is the closest the codebase comes to a pattern, but the derivation is open-coded per op and not exposed as a reusable helper.

---

## Proposal 1: Auto-Derive SFPU Iteration Count From `face_r_dim`

The SFPU iteration count `N` is not free information — it is mechanically `total_num_faces()` for the tile in the destination register. The `TensorShape` struct already exposes this as a `constexpr` method (`tensor_shape.h:44–74`):

```cpp
// tt_llk_blackhole/common/inc/tensor_shape.h (paraphrased; see :44-74)
struct TensorShape {
    std::uint8_t face_r_dim;
    std::uint8_t face_c_dim;     // always 16
    std::uint8_t num_faces_r;
    std::uint8_t num_faces_c;
    constexpr std::uint8_t total_num_faces() const { return num_faces_r * num_faces_c; }
    constexpr std::uint32_t total_row_dim() const  { return num_faces_r * face_r_dim; }
    constexpr std::uint32_t total_col_dim() const  { return num_faces_c * face_c_dim; }
    constexpr std::uint32_t total_tensor_size() const { return total_row_dim() * total_col_dim(); }
};
```

What is missing is the bridge from a CT-arg-visible `TileDescriptor` to the SFPU template argument. A minimal helper would look like this (proposed; not in tree):

```cpp
// proposed: tt_llk_*/common/inc/tile_descriptor_helpers.h
template <typename TileDesc>
constexpr std::uint8_t sfpu_face_count_v =
    TileDesc::num_faces_r * TileDesc::num_faces_c;

// usage at call site (replaces the hand-written `2`):
PACK((ckernel::llk_math_eltwise_unary_sfpu_sigmoid<
        CTArgs::fused_activation_approx_mode,
        /*is_fp32_dest_acc_en=*/false,
        sfpu_face_count_v<typename CTArgs::q_tile>>(
        0, (int)VectorMode::R)));
```

With this in place, the `Q_TILE_HEIGHT = 8 → N = 2` step disappears: the kernel author writes the SFPU call once, and changing the Python tile to `(16, 32)` propagates through the CT arg struct to the template parameter automatically. The legal `(num_faces, face_r_dim)` pairs are already enforced by `validate_tensor_shape_tile_dependent_ops_` (see Proposal 3), so `sfpu_face_count_v` is guaranteed to evaluate to one of `{1, 2, 4}` for any tile that passes validation.

The change is non-invasive at the LLK boundary — the existing `llk_math_eltwise_unary_sfpu_*` signatures already take the face count as a template argument, and every call site that today writes a literal `2` or `4` would write `sfpu_face_count_v<...>` instead.

---

## Proposal 2: Promote `tile_desc` to a First-Class Field on `CBHandle`

CB handoffs between ops are the natural place to detect geometry mismatch — a producer kernel writes pages of a certain tile shape, and a consumer kernel reads them. Today the producer's `TileDescriptor` does ride along on the CB:

```python
# tt-blaze/.../cb_handle.py:35-96 (relevant excerpt)
@dataclass(eq=False)
class CBHandle:
    ...
    tile_desc: object
    ...
```

But `tile_desc` is typed `object`, has no getter, and is treated as opaque pass-through metadata for the L1 allocator. A consumer op that wants to know "what tile shape will I receive on this CB?" must either (a) re-declare the geometry from scratch and trust the producer to match, or (b) reach into private state and hope the field stays where it is.

Two changes would unlock CB-level validation without altering the allocator:

1. **Type the field.** Replace `tile_desc: object` with `tile_desc: TileDescriptor`, and add a typed accessor `CBHandle.tile_descriptor() -> TileDescriptor`. Consumers can then read producer geometry as documented API.
2. **Add an equality / compatibility check at bind time.** When a consumer op binds a CB to one of its CT-arg slots, the framework compares the slot's declared `TileDescriptor` against `cb.tile_descriptor()` and raises at op-emit time if they disagree. This catches the "producer is 8x32, consumer expects 32x32" class of bug before any kernel is built.

The proposed consumer-side pattern then becomes:

```python
# proposed op-emit pattern
q_cb = ...  # received from upstream
expected = ttnn.TileDescriptor(ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH)))
assert q_cb.tile_descriptor() == expected, \
    f"Q CB tile {q_cb.tile_descriptor()} != expected {expected}"
```

The lint check described next would lift this assertion out of hand-written op code and into the toolchain.

---

## Proposal 3: A Compile-Time Lint Pass Backed by the Existing Validator

The legal-geometry specification already exists as code. From `tensor_shape.h:87–94`:

```cpp
// tt_llk_*/common/inc/tensor_shape.h:87-94
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

The enumerated legal combinations are:

| `num_faces` | `face_r_dim` | `face_c_dim` | Tile shape (RxC) | Use site |
|---|---|---|---|---|
| 1 | 1 | 16 | 1x16 | (theoretical; not exercised in tree) |
| 1 | 2 | 16 | 2x16 | (not exercised) |
| 1 | 4 | 16 | 4x16 | (not exercised) |
| 1 | 8 | 16 | 8x16 | (not exercised) |
| 1 | 16 | 16 | 16x16 | small SFPU paths |
| 2 | 1 | 16 | 1x32 | (not exercised) |
| 2 | 2 | 16 | 2x32 | (not exercised) |
| 2 | 4 | 16 | 4x32 | (not exercised) |
| 2 | 8 | 16 | 8x32 | **Flash MLA Q tile** |
| 2 | 16 | 16 | 16x32 | partial-face matmul |
| 4 | 16 | 16 | 32x32 | full-tile default |

Note that `face_c_dim` is always 16 — see Chapter 8, Section "Open Questions and Future Work" for why this asymmetry exists.

This validator is the authoritative spec, but it is hidden inside an LLK header used at runtime by a handful of `LLK_ASSERT` sites; no developer-facing guide references it, and the op author has no way to ask "is this tile legal?" at the Python layer. Two changes would close the loop:

1. **Mirror the validator in `ttnn.TileDescriptor`.** Construction of a `TileDescriptor` should fail (Python `ValueError`) for any `(num_faces, face_r_dim, face_c_dim)` combination not in the table above. This moves the error from a device-side `LLK_ASSERT` at kernel launch to the first line of host code that names an illegal geometry.
2. **Add a CB-binding lint pass.** Once `CBHandle.tile_desc` is typed (Proposal 2), the compiler / op-emit layer can statically check that every consumer's CT-arg `TILE_HEIGHT` (and width) matches the bound CB's `tile_descriptor().tile_shape`. Mismatch is a build-time error with a message naming both the producer op and the consumer op, rather than a runtime `cb_wait_front` hang.

The lint pass is cheap because the information is already available: the CT-arg struct's `TILE_HEIGHT` is visible in `named_args_generated.h` (`viz_export.py:756, 771`), and the CB descriptor's tile is visible in the L1 profile dump (`l1_profile.py`, gated by `BLAZE_L1_PROFILE`). Today these two views can only be reconciled by a human reading both files; making the reconciliation a pass in the toolchain is mechanical work.

---

## Documentation Gap: The Validator Has No Audience

`validate_tensor_shape_tile_dependent_ops_` is referenced from roughly ten LLK header sites across the unpack / pack / math paths and is the de-facto source of truth for "what tile shapes can this hardware actually handle?" It is not surfaced in:

- any README under `tt-llk/` or `tt-metal/`,
- the LLK API reference (where SFPU and matmul init functions are documented),
- the TT-Metal CB descriptor docs (which describe `tile_shape` as if any `(R, C)` with `R, C` divisible by 16 were legal),
- the tt-blaze op-author guide.

A newcomer porting an op to a tiny tile has to either find this function by grep or reverse-engineer the rule from working examples (Flash MLA, RoPE). The proposed fix is mechanical: copy the table above plus a short prose explanation of the face-decomposition rule into the op-author guide, with the validator function as the cited source. Chapter 1 of this guide already describes the geometry model; promoting that description into the official docs closes the loop.

---

## Impact Estimate

The three proposals are independent and can land in any order, but they compound:

- **Proposal 1 (constexpr SFPU face count)** alone eliminates one of the three places the geometry is declared. It is the smallest change — a single header, plus mechanical replacement of literal `2`/`4` template arguments at the ~dozen SFPU call sites in `unified_kernels/`.
- **Proposal 2 (typed `CBHandle.tile_desc`)** is the precondition for any inter-op static checking. It is also the smallest user-visible API change: existing code that doesn't touch `tile_desc` is unaffected.
- **Proposal 3 (mirrored validator + CB-binding lint)** turns the three remaining failure modes from Chapter 7 — `cb_wait_front` hangs, SFPU under-iteration, and silent corruption from illegal `(num_faces, face_r_dim)` — into build-time errors with named call sites.

Taken together, the authoring burden for a new tiny-tile op drops from "declare in three places and pray" to "declare the tile once in Python; the CT-arg struct and the SFPU iteration count are derived; the toolchain rejects illegal combinations and producer/consumer mismatches at build time." The hardware path, the LLK, and the existing op surface are untouched.

See Chapter 8, Section "Debugging Tooling Landscape" for the complementary tooling-side proposals (visualizer geometry column, L1-profile explicit tile labels) that surface the same information at runtime when a build-time check is not enough.
