# Cross-Architecture Support for Tiny Tiles

Tiny-tile geometry — `num_faces ∈ {1,2,4}` with `face_r_dim ∈ {1,2,4,8,16}` and
`face_c_dim = 16` — is a Tensix-wide concept, but the three architectures the LLK
library targets (Blackhole, Wormhole B0, Quasar) reach it through very different
hardware. This section audits each architecture's LLK implementation against the
shared legality spec, calls out where parameterization actually exists in the
data path, and explains why Quasar's TDMA-based descriptor model currently
locks the geometry to canonical 32x32.

**Prerequisites:** Chapter 1 (tiny-tile definition and the legal-geometry
vocabulary `face_r_dim` / `face_c_dim` / `num_faces`) and Chapter 2 (unpack /
math / pack data-path walk). See Chapter 4 for the TT-Metal metadata layer that
exposes these knobs to host code.

---

## 1. The Validator as Cross-Architecture Spec

The single source of truth for tiny-tile legality lives in
`tt_llk/common/tensor_shape.h`. It is shared across all three architecture
trees — Blackhole, Wormhole B0, and Quasar all consume the same header — so
the legal set of `(num_faces, face_r_dim, face_c_dim)` tuples is, by
construction, identical at the C++ type level:

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

The validator returns true for **15 combinations** (3 `num_faces` × 5
`face_r_dim` values, with `face_c_dim` always 16). Whether the underlying
hardware can actually execute a given tuple is a separate question — and the
answer differs sharply between Blackhole/WH and Quasar. The validator
specifies what the LLK API will *accept*; it does not certify hardware
support.

The `TensorShape` struct itself is a packed 4-byte descriptor (asserted at
`tensor_shape.h:76`), with `face_r_dim`, `face_c_dim`, `num_faces_r_dim`, and
`num_faces_c_dim` as `std::uint8_t` fields. This descriptor is what gets
plumbed through unpack/math/pack init calls; nothing in the descriptor itself
distinguishes one architecture from another.

---

## 2. [BH] Blackhole — Full Support

Blackhole's LLK layer parameterizes every stage of the data path on the
runtime `face_r_dim` / `num_faces` pair. The pack MOP is the canonical example:

```cpp
// tt_llk_blackhole/llk_lib/llk_pack.h:72-79
template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim = TILE_C_DIM,
    const std::uint32_t num_faces  = 4,
    const std::uint32_t num_tiles  = 1)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
```

The MOP inner/outer loop counts and address-modifier strides are computed from
these runtime arguments (`llk_pack.h:84-133` for the untilize path,
`:135-149` for the tilize path). When `face_r_dim < 16`, `MOP_INNER_LOOP`
shrinks proportionally, and `PACK_INTF_SEL` switches to `SINGLE_INTF_ACTIVE`
when `tile_c_dim < TILE_C_DIM` (i.e., column-reduced tiles like 16x16).

The corresponding upstream knobs:

- **Unpack:** `_llk_unpack_AB_matmul_init_<>()` in
  `tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h` takes `face_r_dim_A`,
  `face_r_dim_B`, `num_faces_A`, `num_faces_B`, and `partial_face_A` /
  `partial_face_B` template arguments. Tiny tiles activate `partial_face` mode,
  which short-circuits the second 16-row half of each face.
- **Math:** `_llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>()` in
  `llk_math_matmul.h` accepts `tile_r_dim` / `tile_c_dim` parameters that
  drive the MVMUL inner-loop count. `reuse_a` / `reuse_b` strategies operate
  unchanged on tiny geometries — the FPU sees a smaller `KT_DIM`-style count,
  nothing else.

Every `(num_faces, face_r_dim)` combination that passes the validator is
validated on Blackhole silicon (P150 / P300) by the test suite under
`tests/sources/` (see Chapter 3).

---

## 3. [WH] Wormhole B0 — Full Support

Wormhole B0's LLK layer mirrors Blackhole's. The pack MOP configuration in
`tt_llk_wormhole_b0/llk_lib/llk_pack.h` uses the same `(face_r_dim,
tile_c_dim, num_faces, num_tiles)` runtime signature and the same
`num_faces ∈ {1,2,4}` assertion. There is **no third template parameter
difference** between the two architectures in the current codebase — the
historical "WH-era" pack variants that used a different signature have been
removed; modern WH packs share the Blackhole interface.

Practically:

- WH and BH share the validator (it is in `common/`, not in an arch-specific
  directory).
- WH and BH share the `TensorShape` descriptor layout.
- WH packs and unpacks accept the same `face_r_dim` / `num_faces` runtime
  parameters and assert the same legality.

A tiny-tile kernel that compiles for Blackhole will compile for WH B0 with no
source change at the LLK-API level. Differences live above the LLK layer
(metal compute API surface) and below it (SFPU instruction mix, dest-register
banking), neither of which affects tile geometry semantics.

---

## 4. [Q] Quasar — Status Unclear; TDMA Forces 16x16 Faces

Quasar replaces the programmable face-striding unpack/pack engine of BH/WH
with the **TDMA (Tenstorrent Data Movement Accelerator)**: a descriptor-driven
data-movement unit where tile shape is fixed in the buffer descriptor table
at configure time. The current matmul test wires this descriptor with
hardcoded face dimensions and explicitly comments out the possibility of
tiny tiles:

```cpp
// tt_metal/tt-llk/tests/sources/quasar/matmul_quasar_test.cpp:43-45
tdma_desc_src_a.buf_desc.f.x_dim        = FACE_C_DIM;   // Default face dimension is 16, tiny tiles not supported for quasar
tdma_desc_src_a.buf_desc.f.y_dim        = FACE_R_DIM;   // Default face dimension is 16, tiny tiles not supported for quasar
tdma_desc_src_a.buf_desc.f.z_dim        = num_faces_A;  // Number of faces = 4, tiny tiles not supported for quasar
```

The identical pattern repeats for source B at lines 54-56. `CT_DIM`, `RT_DIM`,
and `KT_DIM` are tile-count parameters (how many 32x32 tiles fit in the
matmul shape), **not** face-shape parameters — the test does not sweep
tiny-tile geometries.

The Quasar pack header has a structurally different signature that omits
`face_r_dim` entirely:

```cpp
// tt_llk_quasar/llk_lib/llk_pack.h:23
inline void _llk_pack_mop_config_(const std::uint8_t buf_desc_id, const std::uint32_t num_tiles)
```

There is no place in this signature for the host to communicate sub-16-row
face geometry. The pack MOP is built from `MOP_OUTER_LOOP = 1` and
`MOP_INNER_LOOP = num_tiles`, with a single `TT_OP_PACR0_TILE_INC` instruction
(`llk_pack.h:25-29`). The face layout that this MOP packs is whatever the
TDMA buffer descriptor said it was at configure time, and that descriptor
hardcodes `x_dim = y_dim = 16`.

**Why this matters:** TDMA's contract with the host is the buffer descriptor.
Once a `tdma_descriptor_t` is bound to a `buf_desc_id` via
`_configure_buf_desc_table_()`, the face shape is fixed for the lifetime of
that descriptor. Supporting tiny tiles on Quasar is not a matter of
parameterizing an existing MOP — it requires extending the descriptor
architecture (or maintaining multiple `buf_desc_id` bindings) so the unpack
and pack TDMA engines agree on a non-default `y_dim`. None of that machinery
exists in the LLK layer today.

The shared validator does not save Quasar here: validation happens in
`common/tensor_shape.h` against the `TensorShape` *struct*, but Quasar's
data-movement path never consumes a `TensorShape` at runtime — it consumes a
`tdma_descriptor_t.buf_desc` instead. Compile-time legality and run-time
support are decoupled.

---

## 5. Support Matrix

The following table summarizes the state of each stage. "Parameterized" means
the LLK function signature exposes `face_r_dim` / `num_faces` as a
runtime/template argument and the implementation honors it. "Fixed 16x16"
means the face shape is hardcoded into the data-movement descriptor or MOP.

| Stage    | [BH] Blackhole                                | [WH] Wormhole B0                              | [Q] Quasar                                |
|----------|-----------------------------------------------|-----------------------------------------------|-------------------------------------------|
| Unpack   | Parameterized (`llk_unpack_AB_matmul.h`)      | Parameterized (`llk_unpack_AB_matmul.h`)      | Fixed 16x16 via TDMA `buf_desc.x/y_dim`   |
| Math     | Parameterized (`llk_math_matmul.h`)           | Parameterized (`llk_math_matmul.h`)           | `llk_math_matmul.h` present; not exercised|
| Pack     | Parameterized (`llk_pack.h:72-77`)            | Parameterized (same signature)                | Fixed (`llk_pack.h:23` — no face arg)     |
| Validator| Shared `common/tensor_shape.h`                | Shared                                        | Shared (struct only; not consumed by TDMA)|
| Support  | **Full** — all 15 validator tuples            | **Full** — all 15 validator tuples            | **Canonical 32x32 only** in current tests |

---

## 6. Consequences for the Layers Above

- **TT-Metal metadata (Chapter 4):** the `TileDescriptor` / `Tile` host-side
  classes carry the geometry uniformly across all three arches, but emitting a
  non-canonical tile descriptor on Quasar will silently produce a kernel that
  ignores the height/face count below 32x32, because the LLK pack path has no
  parameter to receive it.
- **TT-Blaze CBHandle propagation (see Chapter 7 Section 2):** the
  `tile_desc` field of `CBHandle` carries tiny-tile shape downstream
  regardless of arch; the failure mode on Quasar is silent geometry
  mismatch between the consumer's declared `tile_desc` and the unpack TDMA
  descriptor it actually uses, not a validator rejection.
- **Multi-arch fused programs:** any micro-op authored to depend on tiny-tile
  geometry must either (a) gate its emit on the target arch and fall back to
  32x32 on Quasar, or (b) accept that the Quasar code path will pad. There is
  no LLK-level abstraction today that makes the choice transparent.

The asymmetry between the validator (shared, lenient) and the data-movement
descriptor (Quasar-specific, strict) is the practical center of the
cross-architecture tiny-tile story: a tile shape that compiles cleanly on all
three architectures still only runs correctly on two of them.
