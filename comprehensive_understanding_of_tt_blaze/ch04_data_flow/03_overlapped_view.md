# 03 -- OverlappedView and Shared L1 Views

**Source file:** `blaze/fused_program.py`

OverlappedView is the data structure that enables multiple logical sub-tensors to share one physical L1 allocation. A fused weight buffer on the device might pack Q, K, and V projections contiguously in each core's shard. OverlappedView carries the byte offset, shard shape, and format for each sub-tensor so that `cb_from_view()` and `cb_from_shared_l1_views()` can allocate CBs that point at the correct sub-regions without copying data.

On Blackhole silicon, L1 is scarce. Allocating separate sharded tensors for gate weights, up-projection weights, and bias vectors wastes memory on alignment padding and shard-spec overhead. OverlappedView lets you pack them into a single fused tensor and hand each op a view with its own byte offset, shard shape, and data format.

## OverlappedView Dataclass

```python
@dataclass(frozen=True)
class OverlappedView:
    tensor: ttnn.Tensor
    shard_shape: tuple[int, int]
    core_ranges: ttnn.CoreRangeSet
    dtype: ttnn.DataType
    tile: object
    byte_offset: int
    total_size: int
```

The dataclass is frozen -- all fields are immutable after construction. This is correct because a view describes a fixed region of a fixed tensor; there is no reason to mutate it. (In contrast, CBHandle is deliberately not frozen -- the build pipeline mutates `cb_id`, `_node_id`, and `_port_name` in place.)

### Every field

| Field | Type | Purpose |
|---|---|---|
| `tensor` | `ttnn.Tensor` | The fused device tensor (mesh or per-device). Multiple views share this same object -- they differ only in offset and shape metadata. |
| `shard_shape` | `tuple[int, int]` | Per-core shard shape `(H, W)` of this sub-tensor. This is NOT the fused container's shard shape. For example, if the fused buffer packs Q (128 rows) and K (128 rows) contiguously, each view's `shard_shape` is `(128, W)` while the container shard is `(256, W)`. |
| `core_ranges` | `ttnn.CoreRangeSet` | Which cores own this sub-tensor. May differ from other views in the same fused buffer, enabling heterogeneous core assignments (e.g. gate weights on A-cores, up-proj weights on B-cores). |
| `dtype` | `ttnn.DataType` | Data format of this sub-tensor. May differ from the fused container's dtype (e.g. `bfp8_b` data in a `uint32` bag). |
| `tile` | `object` | Tile descriptor (`ttnn.Tile` or `ttnn.TileDescriptor`) for this sub-tensor. Used by `page_size` to compute byte count per tile. |
| `byte_offset` | `int` | Byte offset within each core's fused shard where this sub-tensor starts. The first view in a fused buffer typically has `byte_offset=0`, the second view has `byte_offset=first_view.total_size`, and so on. |
| `total_size` | `int` | Bytes per shard for this sub-tensor. Must be a multiple of `page_size`. Together with `byte_offset`, defines the exact L1 address range: `[buffer_address() + byte_offset, buffer_address() + byte_offset + total_size)`. |

### page_size property

```python
@property
def page_size(self) -> int:
    tile = self.tile
    if hasattr(tile, "get_tile_size"):
        return tile.get_tile_size(self.dtype)
    if isinstance(tile, ttnn.TileDescriptor):
        h, w = tile.height, tile.width
    else:
        h, w = tile
    bytes_per_element = UNPACKED_DTYPE_BYTES.get(self.dtype)
    if bytes_per_element is not None:
        return h * w * bytes_per_element
    return ttnn.Tile([h, w]).get_tile_size(self.dtype)
```

The property derives page size from tile geometry and dtype. It has three paths:

1. **`ttnn.Tile` with `get_tile_size`**: The fast path. Calls the C++ binding directly.
2. **`ttnn.TileDescriptor` or raw tuple**: Extracts `(h, w)`, looks up `UNPACKED_DTYPE_BYTES` for the dtype. For unpacked dtypes (bfloat16, float32, etc.), computes `h * w * bytes_per_element` without constructing a `ttnn.Tile` -- this avoids initializing `MetalContext`, which would fail on CPU-only test runners.
3. **Packed dtypes**: Falls back to constructing `ttnn.Tile([h, w])` and calling `get_tile_size()`.

### memory_config() method

```python
def memory_config(self):
    return _OverlappedMemoryConfig(
        _OverlappedShardSpec(shape=self.shard_shape, grid=self.core_ranges)
    )
```

Returns a duck-typed stand-in for `ttnn.MemoryConfig`. The only property exposed is `shard_spec`, which itself exposes `shape` and `grid`. This enables ops to call `view.memory_config().shard_spec.shape` exactly as they would on a `ttnn.Tensor`, with no `isinstance` checks needed.

The helper dataclasses are frozen and minimal:

```python
@dataclass(frozen=True)
class _OverlappedShardSpec:
    shape: tuple[int, int]
    grid: ttnn.CoreRangeSet

@dataclass(frozen=True)
class _OverlappedMemoryConfig:
    shard_spec: _OverlappedShardSpec
```

### from_overlapped_tensor() classmethod

```python
@classmethod
def from_overlapped_tensor(cls, ot) -> "OverlappedView":
    return cls(
        tensor=ot.fused_tensor,
        shard_shape=ot.shard_shape,
        core_ranges=ot.core_range_set,
        dtype=ot.dtype,
        tile=ttnn.Tile(ot.tile_shape),
        byte_offset=ot.byte_offset,
        total_size=ot.total_size,
    )
```

Bridges the old infra's `OverlappedTensor` into the Blaze OverlappedView. The old infra (e.g. `BlitzDecodeWeights.get_tt_*()` methods) returns `OverlappedTensor` objects with slightly different field names (`fused_tensor` instead of `tensor`, `core_range_set` instead of `core_ranges`, `tile_shape` as a tuple instead of a tile object). This classmethod normalizes them. The tile is wrapped in `ttnn.Tile()` to enable `get_tile_size()` on the fast path.

## Worked Example: Gate+Up Weight Packing

Consider a gated MLP where gate and up projections share a fused weight tensor. The weight-packing infrastructure creates a single sharded tensor with gate weights in the first region and up weights in the second.

### Step 1: Create Views from Packed Weights

Suppose the fused tensor has shard shape (2048, 128) per core, with gate weights occupying the first 1024 rows and up weights occupying the next 1024 rows. Both are bfp8_b with 32x32 tiles.

```python
tile = ttnn.Tile([32, 32])
page_size = tile.get_tile_size(ttnn.bfp8_b)  # e.g., 1088 bytes

gate_total = (1024 // 32) * (128 // 32) * page_size   # 32 * 4 * 1088 = 139264
up_total   = (1024 // 32) * (128 // 32) * page_size   # same

gate_view = OverlappedView(
    tensor=fused_weight_tensor,
    shard_shape=(1024, 128),
    core_ranges=all_matmul_cores,
    dtype=ttnn.bfp8_b,
    tile=tile,
    byte_offset=0,
    total_size=gate_total,
)

up_view = OverlappedView(
    tensor=fused_weight_tensor,    # same physical tensor
    shard_shape=(1024, 128),
    core_ranges=all_matmul_cores,
    dtype=ttnn.bfp8_b,
    tile=tile,
    byte_offset=gate_total,         # starts after gate region
    total_size=up_total,
)
```

### Step 2: Allocate CBs in FusedProgram

```python
gate_cb = f.cb_from_view(gate_view)
up_cb   = f.cb_from_view(up_view)
```

### Step 3: Inside cb_from_view -- First Call (gate_view)

The first call enters `cb_from_view`. It resolves effective dtype, tile, page_size, and core_ranges from the view. Then:

```python
key = id(view.tensor)       # id of fused_weight_tensor
existing = self._view_tensor_cbs.get(key)   # None on first call
```

No existing entry. Falls through to format-key reuse (`_try_reuse_cb_id`). Assuming no prior CB with matching format, allocates fresh:

```python
cb_id = self.program.cb_from_tensor_overlapped(
    view.tensor, view.byte_offset, view.total_size, eff_page_size,
    core_ranges=eff_core_ranges, data_format=eff_dtype, tile=eff_tile,
)
# cb_id = 5 (next available)
```

Registers the view for disjoint-core sharing: `self._view_tensor_cbs[key] = (5, all_matmul_cores)`.

Returns a CBHandle: `handle.cb_id = 5, handle.byte_offset = 0, handle.buffer_address() = fused_weight_tensor.buffer_address() + 0`.

### Step 4: Inside cb_from_view -- Second Call (up_view)

The second call finds the existing entry:

```python
key = id(view.tensor)           # same fused_weight_tensor
existing = self._view_tensor_cbs.get(key)  # (5, all_matmul_cores)
```

Checks disjointness: both views use `all_matmul_cores` -- they are NOT disjoint. The `_are_disjoint` check fails. Falls through to format-key reuse, which also fails (same cores overlap). A new CB is allocated:

```python
cb_id = 6  # fresh allocation
```

This descriptor is backed by the same `fused_weight_tensor` but at `address_offset=139264` (the up_view's byte_offset).

The returned handle: `handle.cb_id = 6, handle.byte_offset = 139264, handle.buffer_address() = fused_weight_tensor.buffer_address() + 139264`.

### What If the Views Were on Disjoint Cores?

If gate weights live on A-cores and up weights on B-cores (an A/B split MLP), the second call's `_are_disjoint` check succeeds. In that case, both views share `cb_id=5`. The CB has two `CBDescriptor` entries: one for A-cores at offset 0, one for B-cores at offset 139264. The kernel on A-cores reads gate weights; the kernel on B-cores reads up weights. Same CB ID, different L1 addresses per core range.

### Summary Diagram

```
  fused_weight_tensor (one L1 allocation per core)
  |<----------- shard_shape (2048, 128) ----------->|
  |                                                  |
  |  gate_view                    up_view            |
  |  byte_offset=0                byte_offset=139264 |
  |  shard_shape=(1024,128)       shard_shape=(1024,128)
  |  total_size=139264            total_size=139264  |
  |                                                  |
  +--------------------------------------------------+

  Disjoint cores (A/B split):
    gate_view on A-cores -> cb_id=5, descriptor at offset 0
    up_view   on B-cores -> cb_id=5, descriptor at offset 139264
    (one cb_id, two descriptors)

  Same cores:
    gate_view -> cb_id=5, offset 0
    up_view   -> cb_id=6, offset 139264
    (two cb_ids)

  cb_from_shared_l1_views (same cores, direct address):
    gate_view -> cb_id=5, DIRECT_ADDRESS, byte_offset=0
    up_view   -> cb_id=5, DIRECT_ADDRESS, byte_offset=139264
    (one cb_id, kernels read by address)
```

## cb_from_view() -- the FIFO Allocator

`FusedProgram.cb_from_view(view, **kwargs)` allocates a FIFO CB from a view, with three sharing paths:

### Path 1: Same-tensor disjoint-cores sharing

```python
key = id(view.tensor)
existing = self._view_tensor_cbs.get(key)
if existing is not None:
    ex_cb_id, ex_core_ranges = existing
    if _are_disjoint(ex_core_ranges, eff_core_ranges):
        # Share cb_id, append separate descriptor
        ...
```

When a previous `cb_from_view()` call used the same backing tensor and the cores are disjoint, the same CB ID is reused with a second `CBDescriptor` scoped to the new cores. The merged grid is stored back in `_view_tensor_cbs` and the `_format_key_to_cb` entry is updated to prevent a later overlapping-grid caller from false-positive reuse.

### Path 2: Format-key cross-tensor sharing

If no same-tensor match, falls through to the general disjoint-core sharing mechanism via `_try_reuse_cb_id()`. This uses the same format-key matching and disjoint-grid check described in [Section 02 -- FusedProgram, Disjoint-Core Sharing](./02_fused_program.md).

### Path 3: Fresh allocation

Falls through to `BlazeProgram.cb_from_tensor_overlapped()` with the view's `byte_offset` as `address_offset` and `total_size` as the region size.

All three paths build a `CBHandle` via `_make_view_handle()` and update `_tensor_handle_cache`.

## cb_from_shared_l1_views() -- the Direct-Address Allocator

```python
def cb_from_shared_l1_views(self, views: list[OverlappedView]) -> list[CBHandle]:
```

This API is for same-core or overlapping-core consumers that all read different logical slices from a shared backing tensor. Unlike `cb_from_view()` (which gives each view its own FIFO CB), `cb_from_shared_l1_views()` allocates one CB ID shared by all views, with each returned handle carrying its own `byte_offset`. The handles have `access_mode=DIRECT_ADDRESS` -- kernels must read via `handle.buffer_address()`.

### Why this exists alongside cb_from_view()

When multiple views must be consumed on the same (or overlapping) cores, FIFO semantics cannot serve them through a single CB slot -- there is only one set of read/write pointers per slot per core. `cb_from_view()` would allocate separate CB IDs, consuming precious slots. `cb_from_shared_l1_views()` uses a single slot plus direct addressing, trading FIFO convenience for slot efficiency. The kernel must explicitly manage read addresses rather than using `cb_wait_front`.

### Validation

The method is thorough about validation, checking every edge case:

1. At least one view is required.
2. All views must share the same backing tensor (by identity), dtype, and tile shape. Checked via the tuple key `(id(tensor), dtype, _tile_shape(tile))`.
3. All views must have the same `page_size`.
4. Every view's `core_ranges` must be a subset of the backing tensor's shard grid (verified via `_core_ranges_subset`).
5. Every `byte_offset` must be non-negative and page-aligned.
6. Every `total_size` must be page-aligned.
7. Every `total_size` must match `_view_total_size_from_shape(view, page_size, tile_desc)` -- consistency between `shard_shape` and declared `total_size`.
8. Every `byte_offset + total_size` must not exceed the backing tensor's shard bytes (if computable).

### Allocation

```python
cb_id = self.program.cb_from_tensor_overlapped(
    backing, 0, 0, page_size,
    core_ranges=union_cores,
    data_format=first.dtype,
    tile=tile_desc,
)
self._track_cb_allocation(cb_id, tensor_backed=True)
self._direct_address_cbs.add(cb_id)
```

The descriptor is bound at `address_offset=0` with `total_size=0` (which asks tt-metal for the full backing shard). The union of all view cores is used. Kernels override the read pointer per-view using the absolute L1 address from each handle.

The CB ID is added to `_direct_address_cbs`, which excludes it from FIFO reuse caches, `cb_for_tensor_or_none` scans, and output wiring.

## Helper Functions

### _format_key() -- the sharing bucket key

The `_format_key()` function computes the 3-tuple `(data_format, (h, w), page_size)` used as the sharing bucket key for disjoint-core CB reuse. See [Section 02 -- FusedProgram, Disjoint-Core Sharing](./02_fused_program.md) for the full implementation and rationale.

### _view_total_size_from_shape()

```python
def _view_total_size_from_shape(view, page_size, tile_desc) -> int:
    shard_h, shard_w = view.shard_shape
    tile_h, tile_w = tile_desc.height, tile_desc.width
    if shard_h % tile_h or shard_w % tile_w:
        raise ValueError(...)
    return (shard_h // tile_h) * (shard_w // tile_w) * page_size
```

Computes the expected total size from the view's shard shape and tile geometry. Used by `cb_from_shared_l1_views()` to validate that each view's declared `total_size` matches what the shard shape implies.

### _tensor_shard_bytes()

Best-effort computation of one shard's byte size for a tensor-backed L1 buffer. Handles both row-major and tiled layouts, with multiple fallback paths for tile size computation. Returns `None` if the tensor lacks a shard spec or the tile geometry is indeterminate. Used by `cb_from_shared_l1_views()` to check that `byte_offset + total_size` does not exceed the backing shard.

### _are_disjoint() and _core_ranges_subset()

```python
def _are_disjoint(a, b) -> bool:
    if a is None or b is None:
        return False
    return a.subtract(b).num_cores() == a.num_cores()

def _core_ranges_subset(inner, outer) -> bool:
    if inner is None or outer is None:
        return True
    return inner.subtract(outer).num_cores() == 0
```

`_are_disjoint` returns `True` when two core range sets share no cores. It conservatively returns `False` when either is `None`. `_core_ranges_subset` returns `True` when `inner` is fully covered by `outer`, defaulting to `True` when either is `None` (broad-match historical behavior). Note the asymmetry: `_are_disjoint` defaults to "not disjoint" (conservative, prevents sharing), while `_core_ranges_subset` defaults to "is subset" (permissive, allows the operation to proceed).

## MultiOutput

While not an overlapped view, `MultiOutput` is defined alongside `OverlappedView` in `fused_program.py` and serves the multi-output case:

```python
class MultiOutput:
    def __init__(self, outputs: dict[str, CBHandle]):
        self._outputs = outputs
```

Supports three access patterns:
- **Positional unpacking**: `scores, indices = DeepseekMoeGate.emit(f, ...)`
- **Attribute access**: `result.output`
- **Dict-style access**: `result["output_indices"]`

The `__iter__` yields values (not keys), matching `dict.values()` order. `__len__` returns the number of output ports. `__contains__` checks for key membership.

## Phantom Cores and Harvested Devices

When `DeviceContext.from_device(device)` constructs the grid configuration, it accounts for harvested (non-functional) cores on the device. The `full_device_grid` attribute on FusedProgram is the grid after harvesting -- it may have gaps. OverlappedViews inherit `core_ranges` from either the backing tensor's shard spec or an explicit override, both of which already reflect the device's physical topology.

The `_core_ranges_subset` check in `cb_from_shared_l1_views()` validates that every view's cores are within the backing tensor's shard grid. On a harvested device, a view constructed with the full logical grid (before harvesting) would fail this check if the backing tensor was sharded only on functional cores -- catching a topology mismatch before it becomes a silent L1 corruption.

The key invariant: OverlappedView metadata is device-independent. `byte_offset`, `total_size`, `shard_shape` are the same on every device in a mesh. Only the backing tensor's L1 address differs per device, and that is resolved by `buffer_address()` at runtime.

---

<- [02 -- FusedProgram](02_fused_program.md) | [Index](index.md) | [04 -- CB Reconfig](04_cb_reconfig.md) ->
