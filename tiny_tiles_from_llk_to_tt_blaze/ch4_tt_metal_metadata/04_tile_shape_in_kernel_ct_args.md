# Tile Shape in Kernel CT Args

Tile geometry crosses the host/device boundary in two distinct ways: **explicitly**, as
compile-time integer fields named `TILE_HEIGHT`, `FACE_R_DIM`, or `NUM_FACES` in a
`ComputeCTArgs` / `ReaderCTArgs` / `WriterCTArgs` struct; and **implicitly**, encoded in the
CB page size and `TileDescriptor` that the kernel later reads via `cb_get_tile()`. Both
patterns coexist in tt-blaze, and the TT-Blaze parser (`cpp_parser.py`) is the bridge that
makes either form visible to the Python op-emit layer. This file shows how the two patterns
look in production code (Flash MLA), how the parser walks a kernel header, and when an op
author should prefer one form over the other.

**Prerequisites:** Chapter 4 Sections 1–3 (the `ttnn.Tile` constructor, `TileDescriptor`,
`cb_descriptor_from_sharded_tensor`, and the producer/consumer propagation rule);
[[introduction_to_tt_blaze]] for the CT-args / runtime-args split and the per-RISC struct
convention.

---

## 1. Two ways tile geometry reaches the kernel

A compute kernel ultimately needs to know two things about each circular buffer it touches:

1. **How big is one tile in L1?** — needed for `cb_reserve_back` / `cb_push_back` accounting
   and for any tile-pointer arithmetic.
2. **What is the tile's internal layout?** — face count, face row dim, partial-face flag —
   needed by `_llk_unpack_*_init_<>()` calls and by any compute math that iterates rows
   inside a face.

TT-Metal answers question (1) at the **CB descriptor** level: `page_size` is set when the
host builds the CBFormatDescriptor (see Chapter 4, Section 2), and the kernel reads it
through the CB index alone. Answer (2) is split: the geometry lives inside the
`TileDescriptor` attached to the CBFormatDescriptor, but the LLK init calls that the
compute kernel issues also need the same numbers as **template parameters** so they can
specialise at compile time.

In practice, op authors choose between two patterns:

- **Pattern A — Geometry implicit in the CB.** No `TILE_HEIGHT` field appears in
  `ComputeCTArgs`. The kernel calls `cb_get_tile_size(cb_q_in)` and the LLK init helpers
  pick up face geometry from the unpacker descriptor that the runtime populated from the
  CBFormatDescriptor. This is Flash MLA's approach.
- **Pattern B — Geometry explicit in CT args.** A `static constexpr uint32_t q_tile_height`
  (and possibly `face_r_dim`, `num_faces`) lives in `ComputeCTArgs`. Kernel code references
  it as `ComputeCTArgs::q_tile_height` and uses it as a template argument and as a loop
  bound. This is preferred when the kernel's control flow (loop unroll counts, register
  allocation) depends on the row count.

The decision is per-CB and per-op: it is perfectly legal — and Flash MLA does this — to
mix a tiny-Q (8x32) with a standard-K (32x32) in a single compute kernel without ever
naming `Q_TILE_HEIGHT` or `K_TILE_HEIGHT` in the CT-args struct, because both numbers are
already encoded in the CB page sizes and TileDescriptors that the host attached to
`cb_q_in` and `cb_k_in`.

---

## 2. The `ComputeCTArgs` struct in Flash MLA

Flash MLA exposes its compile-time surface through three structs in
`micro_ops/flash_mla/flash_mla.hpp:96–128`, one per RISC class:

```cpp
// tt-blaze/micro_ops/flash_mla/flash_mla.hpp:96
template <uint32_t k_page_size_, uint32_t vDHt_, uint32_t cb_out_o_>
struct WriterCTArgs {
    static constexpr uint32_t k_page_size = k_page_size_;
    static constexpr uint32_t vDHt        = vDHt_;
    static constexpr CB       cb_out_o    = CB(cb_out_o_);
};

struct ReaderCTArgs {};  // reader takes only runtime args

template <
    uint32_t cb_q_in_, uint32_t cb_k_in_, uint32_t cb_mask_,
    uint32_t cb_interm_out_, uint32_t cb_interm_ms_,
    uint32_t cb_out_in_, uint32_t cb_ms_in_,
    uint32_t cb_out_o_, uint32_t cb_out_ms_, uint32_t cb_out_final_>
struct ComputeCTArgs {
    static constexpr CB cb_q_in       = CB(cb_q_in_);
    static constexpr CB cb_k_in       = CB(cb_k_in_);
    static constexpr CB cb_mask       = CB(cb_mask_);
    static constexpr CB cb_interm_out = CB(cb_interm_out_);
    static constexpr CB cb_interm_ms  = CB(cb_interm_ms_);
    static constexpr CB cb_out_in     = CB(cb_out_in_);
    static constexpr CB cb_ms_in      = CB(cb_ms_in_);
    static constexpr CB cb_out_o      = CB(cb_out_o_);
    static constexpr CB cb_out_ms     = CB(cb_out_ms_);
    static constexpr CB cb_out_final  = CB(cb_out_final_);
};
```

Two things worth noting:

1. **Every field is a CB index.** There is no `q_tile_height`, no `face_r_dim`. The kernel
   reaches tile geometry by calling LLK init routines on each CB; those routines read the
   unpacker descriptor that the runtime built from the CBFormatDescriptor host-side. The
   8-vs-32 row distinction between Q and K never surfaces as a named integer at the kernel
   API.
2. **The `CB` wrapper is load-bearing for the parser.** The non-trivial type signals to
   `cpp_parser.py` that this field is a CB binding (not a plain uint32 like `k_page_size`
   or `vDHt`). The parser uses this distinction to classify the field's *source*.

---

## 3. `cpp_parser.py`: from C++ struct to ParsedCTArg

The TT-Blaze parser at `tt-blaze/tt_blaze/cpp_parser.py:1–144` is a small regex-driven
scanner that consumes a kernel header and yields a list of `ParsedCTArg` records the
op-emit layer can iterate. There are three moving parts.

### 3.1 The struct-to-RISC map

```python
# tt-blaze/tt_blaze/cpp_parser.py:102
_STRUCT_TO_RISC = {
    "ComputeCTArgs": Risc.TRISC,
    "ReaderCTArgs":  Risc.NCRISC,
    "WriterCTArgs":  Risc.BRISC,
}
```

The struct's *name* alone tells the parser which RISC the field belongs to. There is no
attribute or pragma — convention is enforced lexically.

| CT arg struct     | RISC   | Role                                  |
| ----------------- | ------ | ------------------------------------- |
| `ComputeCTArgs`   | TRISC  | Math kernel (unpack/math/pack)        |
| `ReaderCTArgs`    | NCRISC | DRAM/NoC -> input CB                  |
| `WriterCTArgs`    | BRISC  | Output CB -> DRAM/NoC                 |

A field can also live on multiple RISCs if it appears in more than one struct (e.g.
`cb_out_o` appears in both `WriterCTArgs` and `ComputeCTArgs` above, so the parser tags it
with both `BRISC` and `TRISC` in the same `ParsedCTArg.riscs` bitmask).

### 3.2 The two regex patterns

```python
# tt-blaze/tt_blaze/cpp_parser.py:56
_CT_TYPED_FIELD_RE = re.compile(
    r"static\s+constexpr\s+(CB|Semaphore|PerCore|Flag)\s+(\w+)\s*="
)
_PLAIN_FIELD_RE = re.compile(
    r"static\s+constexpr\s+(uint32_t|int32_t|bool)\s+(\w+)\s*="
)
```

`_CT_TYPED_FIELD_RE` catches fields whose C++ type carries semantic meaning — `CB`,
`Semaphore`, `PerCore`, `Flag` — and stamps the captured type as the `source` on the
emitted `ParsedCTArg`. `_PLAIN_FIELD_RE` catches everything else (the plain integers and
bools used for sizes, strides, and loop bounds) and marks them `source = "derived"`.

This is the mechanism by which an explicit tile-dim CT arg, if you add one, gets surfaced:
declaring `static constexpr uint32_t q_tile_height = q_tile_height_;` makes the parser
emit a `ParsedCTArg(name="q_tile_height", source="derived", riscs=TRISC)`. The op-emit
layer can then look this up by name and pass `q_tile_height_=8` when instantiating the
template.

### 3.3 The `ParsedCTArg` record

```python
# tt-blaze/tt_blaze/cpp_parser.py:30
@dataclass
class ParsedCTArg:
    name:   str
    source: str           # "cb" | "sem" | "per_core" | "flag" | "derived"
    riscs:  int           # bitmask: BRISC | NCRISC | TRISC
```

`source` is the join key against the op's host-side state. For `source == "cb"`, the emit
layer looks up the CB index in the op's CB table. For `source == "derived"`, the field
must be supplied by the op author when emitting the kernel — either as a literal or as a
function of the input tensor shape. **This is where an explicit `q_tile_height` would be
resolved**: the emit code computes `q_tile_height = input_tensor_q.tensor_spec().tile().get_height()`
and passes it through.

---

## 4. Worked example: Flash MLA's tiny-Q, standard-K geometry

In Flash MLA, Q has shape `(batch * num_heads, Dq)` with `num_heads` small enough that
each core gets `Q_TILE_HEIGHT = 8` rows; K has the full `K_TILE_HEIGHT = 32`. The op-emit
code creates two `Tile` objects with different heights and lets `get_tile_size()` compute
the L1 byte cost for each.

```python
# tt-metal/models/.../deepseek_v3_b1/micro_ops/flash_mla/op.py:538
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))      # (8, 32)
k_tile      = ttnn.Tile((K_TILE_HEIGHT, TILE_WIDTH))      # (32, 32)

q_tile_size = q_tiny_tile.get_tile_size(q_df)             # 272 B in BFP8
k_tile_size = k_tile.get_tile_size(k_df)                  # 1088 B in BFP8
```

The two CBs are then declared with their own page sizes and `TileDescriptor`s:

```python
# tt-metal/models/.../deepseek_v3_b1/micro_ops/flash_mla/op.py:713
cb_descriptors.append(
    ttnn.cb_descriptor_from_sharded_tensor(cb_q_in, input_tensor_q)
)   # picks up q_tiny_tile (8x32) via tensor_spec().tile()

cb_descriptors.append(
    ttnn.cb_descriptor_from_sharded_tensor(cb_k_in, input_tensor_k)
)   # picks up k_tile (32x32) via tensor_spec().tile()
```

The compute kernel template is instantiated with **only** the CB indices — the geometry
is already encoded in each CB's `TileDescriptor`:

```cpp
// instantiation (paraphrased from op.py emit)
using MyComputeArgs = ComputeCTArgs<
    /* cb_q_in_       = */ 0,
    /* cb_k_in_       = */ 1,
    /* cb_mask_       = */ 2,
    /* cb_interm_out_ = */ 3,
    /* cb_interm_ms_  = */ 4,
    /* cb_out_in_     = */ 5,
    /* cb_ms_in_      = */ 6,
    /* cb_out_o_      = */ 7,
    /* cb_out_ms_     = */ 8,
    /* cb_out_final_  = */ 9>;
```

Inside the kernel, the LLK init for the Q-times-K matmul reads geometry from the two CBs'
unpacker descriptors:

```cpp
_llk_unpack_AB_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
    ComputeCTArgs::cb_q_in,    // unpacker A: 8x32 face geometry
    ComputeCTArgs::cb_k_in,    // unpacker B: 32x32 face geometry
    /* transpose=*/ 0);
```

That single init call sees `num_faces=2`, `face_r_dim=8`, `partial_face=1` on the A side
and `num_faces=4`, `face_r_dim=16`, `partial_face=0` on the B side, configures the
unpacker descriptors per-RISC, and proceeds. No `Q_TILE_HEIGHT` constant was ever named in
the CT-args header — the parser saw only CB indices, and that was enough.

---

## 5. When to switch to explicit tile-dim CT args

Pattern A (implicit) is enough for Flash MLA because every loop in the compute kernel
iterates over **tile counts**, not over rows inside a tile. The face-internal iteration is
hidden inside the LLK math primitives. The CB index alone gives the kernel everything it
needs.

Pattern B becomes attractive when:

1. **Loop unrolling depends on row count.** If the kernel has `for (uint32_t r = 0; r <
   tile_height; ++r) { ... }` and you want the compiler to unroll it, `tile_height` must
   be a template parameter, which means it must be a `static constexpr` in
   `ComputeCTArgs`. The implicit form (a CB-derived runtime read) cannot unroll.
2. **`reconfig_data_format` switching between two tile geometries on the same CB.** The
   reconfig API takes the new geometry as a template parameter; carrying it as a CT-arg
   keeps the choice visible in the call site.
3. **Static shape assertions.** A `static_assert(q_tile_height <= 32, ...)` in the kernel
   only works if `q_tile_height` is a compile-time constant. The implicit form gives the
   kernel no name to assert on.

The mechanical change is small. Add the field to the C++ struct:

```cpp
template <
    uint32_t cb_q_in_, uint32_t cb_k_in_, ...,
    uint32_t q_tile_height_,
    uint32_t k_tile_height_>
struct ComputeCTArgs {
    static constexpr CB       cb_q_in        = CB(cb_q_in_);
    /* ... */
    static constexpr uint32_t q_tile_height  = q_tile_height_;
    static constexpr uint32_t k_tile_height  = k_tile_height_;
};
```

`cpp_parser.py` picks this up via `_PLAIN_FIELD_RE` and emits two new `ParsedCTArg`
records with `source="derived"`, `riscs=TRISC`. The op-emit Python then resolves them at
template-instantiation time:

```python
q_tile_height = input_tensor_q.tensor_spec().tile().get_height()
k_tile_height = input_tensor_k.tensor_spec().tile().get_height()
```

and passes them through the same CT-arg dictionary that already carries the CB indices.
The kernel can now write `for (uint32_t r = 0; r < ComputeCTArgs::q_tile_height; ++r)`
and the compiler will unroll it.

---

## 6. Summary

- Tile geometry reaches the kernel **either** as compile-time integers in a
  `ComputeCTArgs` struct (pattern B) **or** as page-size + TileDescriptor metadata
  attached to each CB index (pattern A). Flash MLA uses pattern A exclusively.
- `cpp_parser.py` discovers CT args by regex-scanning the kernel header, classifies each
  field as a typed CB/semaphore/flag binding or a plain derived integer, and tags it with
  the owning RISC via the struct-name lookup table (`_STRUCT_TO_RISC`).
- The pattern-A example in Flash MLA carries a 4x size gap between Q (272 B) and K
  (1088 B) tiles through the CBFormatDescriptors without ever naming the row counts at
  the kernel API. That is sufficient because the LLK init helpers re-read the geometry
  from the unpacker descriptors at runtime.
- Switch to pattern B when the kernel's *own* control flow depends on the tile row count —
  static asserts, loop unrolling, or `reconfig_data_format` calls that need the geometry
  as a template parameter rather than as a runtime CB lookup.

This closes the metadata layer: by the end of Chapter 4 the reader knows how a tile shape
is declared at the tensor (`ttnn.Tile`), bound to a CB (`TileDescriptor`), validated by
the producer/consumer rule, and surfaced — or deliberately *not* surfaced — at the kernel
CT-args boundary. Chapter 5 picks up the thread on the device side and follows the same
8x32 Q tile all the way through Flash MLA's compute loop.
