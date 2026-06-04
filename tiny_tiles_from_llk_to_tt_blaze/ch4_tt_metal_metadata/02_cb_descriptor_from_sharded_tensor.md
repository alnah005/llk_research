# 4.2 `cb_descriptor_from_sharded_tensor`: from tensor to CB layout

`cb_descriptor_from_sharded_tensor()` is the bridge between TT-Metal's tensor
abstraction and the circular-buffer (CB) descriptor surface a kernel program
consumes. It takes a sharded `Tensor` plus a CB index and produces a fully
populated `CBDescriptor` whose page size, page count budget, core range set,
and tile geometry are all derived from the tensor itself. For tiny-tile work
this function is load-bearing: the tile attached to the tensor (via its
`TensorSpec`) flows straight through into the CB's `TileDescriptor`, and the
buffer's `aligned_page_size()` determines the per-page L1 byte cost.

**Prerequisites**

- Chapter 3 (Tile-Aware Tensor Layout) — `TensorSpec`, shard specs, and how
  `ttnn.Tile` is attached to a tensor.
- Chapter 4, Section 1 (`ttnn.Tile` and `get_tile_size`) — how the tile object
  computes its byte footprint and how `TileDescriptor` wraps it.
- Familiarity with TT-Metal circular buffers (CB index, page size, page count,
  double-buffering). See [[introduction_to_tt_llk]] for the kernel-side view.

---

## 4.2.1 Function signature and auto-derivation

The host-side declaration lives in `ttnn/api/ttnn/tensor/tensor_utils.hpp`:

```cpp
// ttnn/api/ttnn/tensor/tensor_utils.hpp:62
CBDescriptor cb_descriptor_from_sharded_tensor(
    uint8_t cb_index,
    const Tensor& tensor,
    uint32_t address_offset = 0,
    uint32_t total_size = 0,
    const std::optional<CoreRangeSet>& core_ranges = std::nullopt);
```

Only the first two arguments are required. The remaining three are escape
hatches the caller uses when they need to override the tensor-derived defaults
(typically for CB aliasing or sub-grid placement).

The implementation in `ttnn/core/tensor/tensor_utils.cpp` is short enough to
read in full:

```cpp
// ttnn/core/tensor/tensor_utils.cpp:19
CBDescriptor cb_descriptor_from_sharded_tensor(
    uint8_t cb_index,
    const Tensor& tensor,
    uint32_t address_offset,
    uint32_t total_size,
    const std::optional<CoreRangeSet>& core_ranges) {
    TT_FATAL(tensor.is_sharded(),
             "cb_descriptor_from_sharded_tensor requires a sharded tensor");

    const auto effective_total_size =
        (total_size != 0) ? total_size
                          : tensor.buffer()->aligned_size_per_bank();

    return CBDescriptor{
        .total_size       = effective_total_size,
        .core_ranges      = core_ranges.value_or(tensor.shard_spec()->grid),
        .format_descriptors = {
            CBFormatDescriptor{
                .buffer_index = cb_index,
                .data_format  = datatype_to_dataformat_converter(tensor.dtype()),
                .page_size    = tensor.buffer()->aligned_page_size(),
                .tile         = TileDescriptor(tensor.tensor_spec().tile()),
            },
        },
        .buffer         = tensor.buffer(),
        .address_offset = address_offset,
    };
}
```

Five distinct fields of the returned `CBDescriptor` are auto-derived from
`tensor`:

| Field | Source on tensor | Override |
|---|---|---|
| `total_size` | `tensor.buffer()->aligned_size_per_bank()` | `total_size` arg |
| `core_ranges` | `tensor.shard_spec()->grid` | `core_ranges` arg |
| `data_format` | `tensor.dtype()` → `DataFormat` | none |
| `page_size` | `tensor.buffer()->aligned_page_size()` | none (must rebuild) |
| `tile` | `tensor.tensor_spec().tile()` → `TileDescriptor` | none (must rebuild) |

The two fields with **no override** — `page_size` and `tile` — are exactly the
two that encode the tile's L1 footprint. If the caller wants a CB with a
different page size or tile geometry than the tensor exposes, they cannot use
this helper at all; they construct a `CBDescriptor` by hand (see the K-input
branch in Flash MLA below).

---

## 4.2.2 Where the byte budget comes from

The per-page byte cost flows from the tile attached to the tensor:

```
                  TensorSpec.tile()
                         │
                         ▼
                    ttnn.Tile{H, W}
                         │
                         │ get_tile_size(dtype)
                         ▼
                  ┌──────────────┐
                  │  page_size   │  ← exposed via buffer()->aligned_page_size()
                  └──────────────┘
                         │
                         ▼
            total_size = num_pages × page_size
            (rounded to L1 alignment per bank)
```

Two things to keep separate:

1. **Page size** is set by the tile and the dtype. It shrinks proportionally
   when the tile shrinks. For BFP8, an 8x32 page is 272 B versus 1088 B for a
   32x32 page (Section 4.1).
2. **Number of pages** is a CB *capacity* decision, not a tile decision. It is
   determined by what the consumer kernel needs to keep resident — typically a
   double-buffering factor times the inner-loop accumulation depth — and it
   does not change when the tile gets smaller.

`cb_descriptor_from_sharded_tensor` only sees one of those two knobs directly:
the page size. The page count is implicit in
`buffer()->aligned_size_per_bank()`, which already reflects how many tiles per
core the tensor's shard spec carved out. If the kernel needs more pages than
the input tensor's shard supplies (for example, a producer that double-buffers
two tile-rows ahead), the caller passes an explicit `total_size` override.

The practical consequence for tiny-tile pipelines: **shrinking the tile shrinks
the CB's L1 footprint by exactly the same factor**, because the page-count
budget is unchanged. An 8x32 Q chain that needs 18 resident pages costs
`18 * 272 = 4,896 B`. The same chain at 32x32 would cost
`18 * 1088 = 19,584 B` — a 4x inflation for no kernel-side reason other than
"the tile got bigger."

---

## 4.2.3 The TileDescriptor handoff

The line that closes the loop between tensor-side tile geometry and CB-side
tile geometry is `tensor_utils.cpp:39`:

```cpp
.tile = TileDescriptor(tensor.tensor_spec().tile()),
```

`TileDescriptor` is the lightweight `{height, width, transpose}` struct from
`program_descriptors.hpp:39`. Its constructor from a `Tile` simply copies the
two dimensions out:

```cpp
// tt_metal/api/tt-metalium/program_descriptors.hpp:46
TileDescriptor(const Tile& tile)
    : height(tile.get_height()),
      width(tile.get_width()),
      transpose(false) {}
```

That means the CB's tile-descriptor is *exactly* `(H, W)` of the tensor's
tile. There is no validation here, no padding, no rounding — whatever shape
was attached to the tensor at construction time is what the CB will advertise
to the kernel via CT-args (see Chapter 4, Section 4). This is why the
TensorShape validator (`tt_llk/common/tensor_shape.h:87`) has to run earlier:
once the geometry reaches `cb_descriptor_from_sharded_tensor`, it is taken on
faith.

---

## 4.2.4 Flash MLA: tiny Q in, full K beside it

Flash MLA's CB setup is a clean illustration of when to use the helper and
when to hand-build. The Q chain is tiny (8x32), the K input is full (32x32),
and both live in the same program.

```python
# tt-metal/.../deepseek_v3_b1/micro_ops/flash_mla/op.py:538
q_tiny_tile = ttnn.Tile((Q_TILE_HEIGHT, TILE_WIDTH))  # Q_TILE_HEIGHT = 8
...
# op.py:552
q_tile_size = q_tiny_tile.get_tile_size(q_df)         # 272 B for BFP8
...
# op.py:712
q_input_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
    cb_q_in, input_tensor_q
)
q_input_cb_descriptor.core_ranges = core_grid         # override grid only
```

The Q input tensor was built with the 8x32 tile attached. The helper picks
that up unchanged: `page_size = 272 B`, `tile = (8, 32)`, `data_format = BFP8`,
and `total_size` reflects whatever the shard spec already laid out per bank.
The only post-hoc adjustment is `core_ranges`, because the program wants Q on
a specific compute grid rather than the tensor's storage grid.

The K input cannot use the helper, because the K tensor is laid out at the
full 32x32 tile and the CB has been sized for a specific number of resident
tiles independent of the shard:

```python
# op.py:718
k_input_cb_descriptor = ttnn.CBDescriptor(
    total_size=k_tiles * k_tile_size,                 # 144 * 1088 = 156,672 B
    core_ranges=core_grid,
    format_descriptors=[
        ttnn.CBFormatDescriptor(cb_k_in, k_df, k_tile_size)
    ],
)
```

Two CBs, two construction paths, both correct: the helper handles the case
where "the tensor already says what I want", and the explicit constructor
handles the case where the CB capacity is decoupled from the tensor's shard.

Further down, the intermediate stats CB is also explicit, because no tensor
backs it:

```python
# op.py:768
stats_cb_descriptor = ttnn.CBFormatDescriptor(
    cb_out_o, stats_df, stats_tile_size, stats_tile_descriptor
)
```

`stats_tile_descriptor` here wraps the same 8x32 geometry as the Q chain, which
is what allows the aliasing pattern at op.py:763–772 (cb_out_o and
cb_interm_out sharing L1) to be safe. If the two CBs disagreed on tile shape,
the aliasing would corrupt the format-descriptor view of the shared region.

---

## 4.2.5 RoPE: deriving the tile from the shard

The RoPE micro-op shows a slightly different idiom: rather than fixing the
tile height as a constant, it derives it from the input tensor's shard shape
at op-emit time.

```python
# tt-metal/.../deepseek_v3_b1/micro_ops/rope/op.py:121
num_q_heads_per_core = shard_shape[0]
tile = ttnn.Tile((num_q_heads_per_core, ttnn.TILE_SIZE))
```

That tile is then used both implicitly (via `cb_descriptor_from_sharded_tensor`
on the input and output tensors, which already carry it) and explicitly (for
the intermediate `cos_sin` CB, which has no tensor backing):

```python
# op.py:143
input_cb_descriptor  = ttnn.cb_descriptor_from_sharded_tensor(input_cb,  input_tensor)
# op.py:145
cos_sin_cb_descriptor = ttnn.CBDescriptor(
    total_size=2 * head_dim_per_core_t * tile.get_tile_size(cos_sin_df),
    core_ranges=core_grid,
    format_descriptors=[
        ttnn.CBFormatDescriptor(cos_sin_cb, cos_sin_df,
                                tile.get_tile_size(cos_sin_df),
                                ttnn.TileDescriptor(tile))
    ],
)
# op.py:161
output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)
```

The pattern to internalize is: any CB whose page size and tile must match the
tensor going in or out of it should go through `cb_descriptor_from_sharded_tensor`;
any CB that is purely internal (intermediates, scratch, fused-output buffers)
must be built by hand using the *same* tile object so that the geometry
stays coherent across the chain. Chapter 4 Section 3 covers why the
producer-consumer tile match is a hard invariant.

---

## 4.2.6 L1 budget impact, end to end

Pulling the per-CB numbers together for the Flash MLA Q-vs-K example, with
BFP8 pages and the page counts from the live program:

| CB | Tile | Pages | Page size | L1 size |
|---|---|---:|---:|---:|
| Q input (tiny)    | 8x32  |  18 |   272 B |   4,896 B |
| Q chain intermed. | 8x32  | ~40 |   272 B |  ~10,880 B |
| K input (full)    | 32x32 | 144 | 1,088 B | 156,672 B |
| Stats / output    | 8x32  |  ~8 |   272 B |   ~2,176 B |

Total Flash MLA CB footprint at the tiny Q geometry is roughly **217 KB per
core**. If Q were padded to 32x32 — keeping the same page counts because the
kernel still needs the same number of resident tiles — the Q-side CBs (input,
all Q chain intermediates, and the stats/output buffers that share Q geometry)
inflate by 4x, pushing the total to **~280 KB**. That 63 KB saving is the L1
budget that buys the rest of the Flash MLA fused chain: scratch CBs for the
softmax-online accumulators, the K-projection double-buffer, and the output
aliasing slot.

The mechanical reason the saving falls out so cleanly is exactly the structure
of this section: `page_size` tracks the tile, `num_pages` tracks the kernel's
resident-tile requirement, and the two are multiplied to get the bank-aligned
total. `cb_descriptor_from_sharded_tensor` does nothing more than read those
two numbers off the tensor and the buffer and bind them, together with the
tile descriptor, into the CB. Get the tile right on the tensor and the whole
CB chain inherits it.
