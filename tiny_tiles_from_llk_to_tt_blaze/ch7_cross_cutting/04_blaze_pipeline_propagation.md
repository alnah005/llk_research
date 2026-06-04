# Pipeline Propagation of Tile Geometry

Tile geometry is not a single global parameter; it is a per-edge property that
must remain consistent at every stage boundary in a multi-stage pipeline. In
TT-Blaze, the carrier of this property is the `CBHandle.tile_desc` field, and
the place it ultimately matters is the D2D socket page size that connects one
stage's output to the next stage's input. This file traces how a tiny-tile
geometry produced by one stage propagates to downstream stages, how it is (and
isn't) captured in the PipelineGraph, what happens under multi-host
serialization, and the most common bug pattern at tile-geometry boundaries.

**Prerequisites**: Chapter 1 (tiny-tile definition, face geometry), Chapter 4
(TT-Metal `TileDescriptor` metadata), Chapter 7 Section 1 (TensorShape
validator), and Chapter 7 Section 3 (host-side weight invariants). External
cross-references: [[comprehensive_guide_to_tenstorrent_sockets_d2d_h2d_and_d2h_communication]]
for socket mechanics and [[comprehensive_understanding_of_tt_blaze]] Chapter 8
for the PipelineGraph builder.

## 1. Socket Page Size and Tile Geometry

A D2D socket carries one "frame" per scheduling cycle, and a frame is sized
in tiles, not in bytes-from-thin-air. The rule is simple and load-bearing:

```
socket.page_size_bytes = tile.get_tile_size(dtype) * pages_per_shard
```

`SocketConfig` (blaze/socket_config.py:17-24) holds this as a plain field:

```python
# blaze/socket_config.py:17
@dataclass
class SocketConfig:
    sender_core_ranges: list
    receiver_core_ranges: list
    page_size_bytes: int          # tile bytes * pages_per_shard
    ...
```

`tile.get_tile_size(dtype)` is the canonical query (see Chapter 4). For a
BFP8 tile, the byte count tracks the face geometry directly:

| Tile shape | `num_faces` | `face_r_dim` | BFP8 bytes |
|------------|-------------|--------------|------------|
| 32x32      | 4           | 16           | ~2112      |
| 16x32      | 2           | 16           | ~1088      |
| 8x32       | 2           |  8           | ~576       |
| 4x32       | 2           |  4           | ~320       |
| 1x32       | 2           |  1           | ~96        |

(Exact byte counts include exponent overhead; the load-bearing point is the
ratio.) An 8x32 BFP8 tile is roughly a quarter the size of a 32x32 BFP8 tile,
so a stage that emits 8x32 outputs and a stage that emits 32x32 outputs need
different `page_size_bytes` on their downstream sockets even when
`pages_per_shard` is identical.

The page size is baked into per-core fabric runtime args via
`ttnn.compute_fabric_connection_rt_args()` (see
[[comprehensive_guide_to_tenstorrent_sockets_d2d_h2d_and_d2h_communication]]).
Once written, the receiver's NCRISC reads frames of exactly that size from the
incoming fabric channel; a mismatch is not negotiated, it is silently broken.

## 2. CBHandle TileDescriptor as the Metadata Carrier

The propagation mechanism inside a single stage is the `CBHandle`. From
blaze/cb_handle.py:35-95:

```python
# blaze/cb_handle.py:35
@dataclass
class CBHandle:
    cb_id: int
    num_pages: int
    page_size: int            # bytes per page = tile.get_tile_size(dtype)
    tile_desc: object         # TileDescriptor or compatible
    backing_tensor: object | None
    access_mode: AccessMode
    ...
```

Two fields matter for propagation:

- `page_size` — the byte count for a single tile-page in this CB.
- `tile_desc` — the `ttnn.TileDescriptor` (or compatible) that names the tile
  shape, e.g. `TileDescriptor(Tile((8, 32)))`.

When a producer micro-op calls `f.output(...)` it returns a `CBHandle` whose
`tile_desc` reflects the tile shape it actually emitted. When a downstream
micro-op binds that handle as an input, it can read both `handle.page_size`
(for CB allocation) and `handle.tile_desc.tile_shape` (for kernel CT args and
for socket configuration on the egress side). Activation tile geometry
therefore flows through the FusedProgram as a chain of CBHandle references,
with no separate side-channel.

Weights are deliberately excluded from this propagation: per Chapter 7
Section 3, weights stay in canonical 32x32 DRAM layout regardless of
downstream activation geometry. `OverlappedTensor` and the weight-prep code
in tt-metal/models/demos/deepseek_v3_b1/prepare_weights.py:43-60 do not
inspect the activation `tile_desc` — they preserve the canonical fusion-group
layout and let the consuming kernel do any in-tile reshaping it needs.

## 3. PipelineGraph and Implicit Geometry

Across stage boundaries the carrier changes from `CBHandle` (a runtime,
in-process Python object) to the PipelineGraph and its `Edge` objects (see
[[comprehensive_understanding_of_tt_blaze]] Chapter 8). The PipelineGraph
captures, per stage:

- A `Node` with a shape spec for the stage's submesh.
- `Edge` objects describing data flow between stages.

Edges do **not** carry an explicit tile-geometry field. The geometry is
*implicit* in the producing stage's micro-op `emit()` declarations: the
producer declares which `CBHandle` it writes to, and the receiver's `emit()`
declares which `CBHandle` it reads from. When the `PipelineLayout` is
materialized, each stage's `FusedProgram` is instantiated and its micro-ops
re-run `emit()`, recreating the local `CBHandle` chain. The socket on either
end of an `Edge` then reads `CBHandle.page_size` to set
`SocketConfig.page_size_bytes`.

This design is deliberate: tile geometry is a property of the *kernel*, not
of the topology. The PipelineGraph captures topology (which stages talk to
which); the kernels capture geometry. As long as both ends of an edge use
matching `emit()` logic — and they do, because they are the same micro-op
declarations — geometry stays consistent across the boundary without
explicit graph-level encoding.

## 4. Multi-Host Serialization: Geometry Re-Derived Per-Host

`pipeline_builder.serialize_layout()` (see
[[comprehensive_understanding_of_tt_blaze]] Chapter 8) emits a
`PipelineLayout` containing `PipelineConfigEntry` tuples and mesh-coordinate
mappings, but **not** explicit tile-geometry metadata. Each host loads its
assigned stage(s) and runs the same FusedProgram `emit()` code that the
original builder ran, recreating the same `CBHandle` chain and the same
`page_size` values locally.

Socket connection args (including `page_size_bytes`) are computed at the
host-local stage-instantiation step from the local `CBHandle.page_size`, not
read from a serialized field. This is safe because:

- `emit()` is deterministic given the same micro-op inputs.
- Tile shape is a function of dtype and the micro-op's compile-time
  parameters, both of which are recoverable from the serialized
  `PipelineConfigEntry`.
- Both ends of a cross-host edge run the same micro-op declarations, so they
  derive matching page sizes independently.

The implication for tiny tiles: there is no extra serialization step you need
to do to "tell the other host that Q is 8x32." As long as both hosts run the
same micro-op code path, both will compute `page_size = ~576 * pages_per_shard`
for the Q edge and the fabric frame size will match on both ends.

## 5. Worked Example: Flash MLA (8x32 Q) into a Gather Stage

Consider a two-stage pipeline:

- **Stage A**: Flash MLA. Emits an 8x32 Q tile per head-slot into output CB
  `cb_q_out` with `tile_desc = TileDescriptor(Tile((8, 32)))`,
  `page_size = tile.get_tile_size(DataFormat.Bfp8_b)` for an 8x32 tile.
- **Stage B**: A downstream gather/projection stage that consumes Q tiles
  from across the producer submesh.

The edge between A and B is a D2D socket. On Stage A's side, the socket
sender is configured from the output CBHandle:

```python
# Conceptual emit-time wiring inside Stage A
q_handle = f.output(cb_q_out)                       # CBHandle, tile_desc=(8,32)
socket_cfg = SocketConfig(
    sender_core_ranges=...,
    receiver_core_ranges=...,
    page_size_bytes=q_handle.page_size * pages_per_shard,
    ...
)
```

On Stage B's side, the receiver creates a matching `CBHandle` for the
ingress CB. Because the page size was derived from the same `tile_desc`,
the receiver's CB allocation and the sender's frame size agree.

The Flash MLA output uses `padded_tile.get_tile_size(out_ti.data_format)`
(see blaze/ops/create_q_heads/op.py:101 for an analogous pattern) so that
the byte count is always taken from the *actual* tile descriptor, never
hardcoded.

## 6. Common Bug Pattern: Socket Page Size Mismatch

The frequent failure mode at a tiny-tile stage boundary is hardcoding the
socket page size — usually because the receiver was authored against a
32x32 assumption and never re-derived:

```python
# WRONG: assumes canonical 32x32 BFP8 tile (~2112 bytes)
socket_cfg = SocketConfig(..., page_size_bytes=2112 * pages_per_shard)

# CORRECT: derive from the producer's CBHandle
socket_cfg = SocketConfig(
    ...,
    page_size_bytes=q_handle.page_size * pages_per_shard,
)
```

If the producer emits an 8x32 tile (~576 bytes) but the socket is
configured for 2112 bytes, one of two things happens depending on whether
sender or receiver is mismatched:

- Sender frame too large: receiver discards extra bytes, CBs on the
  receiving side end up with garbage in pages 2+ of each tile, downstream
  pack/unpack reads misaligned face data.
- Sender frame too small: receiver waits for a frame that never finishes
  arriving, the receiving CB never advances, the entire pipeline stalls.

Both symptoms look like "the device hung" or "PCC is zero" — the underlying
cause is a single integer mismatch in `SocketConfig.page_size_bytes`.

Defensive practice when authoring a stage that may live behind a tiny-tile
producer:

- Never write a literal integer into `page_size_bytes`.
- Always derive from the upstream `CBHandle.page_size`.
- If the stage exists in isolation for testing, parameterize the test
  harness with the expected `tile_desc` rather than baking in 32x32.
- For multi-host edges, confirm both ends compute page size from the same
  `emit()` path; if either end has a divergent branch (e.g. an old code
  path that defaults to canonical tiles), the multi-host run will fail in
  ways the single-host run does not.

The root cause of these mismatches is almost always cultural: developers
internalize "tiles are 32x32" and write code that assumes it. The
`TensorShape` validator (Chapter 7 Section 1) catches illegal geometries,
but it cannot catch a *legal* 32x32 page-size hardcode that happens to be
wrong for this edge. The only reliable defense is to read `CBHandle`
metadata at every boundary and never assume.

## 7. Summary

- Socket page size = `tile_bytes * pages_per_shard`, computed per-edge.
- `CBHandle.tile_desc` and `CBHandle.page_size` are the in-stage carriers
  of tile geometry.
- The PipelineGraph does not serialize tile geometry; it serializes
  topology and lets each host re-derive geometry from `emit()`.
- Multi-host deployment is safe as long as both ends of every edge run the
  same micro-op `emit()` code path.
- The dominant bug at tiny-tile stage boundaries is a hardcoded
  `page_size_bytes` on one side of a socket; always derive from the
  upstream `CBHandle`.
