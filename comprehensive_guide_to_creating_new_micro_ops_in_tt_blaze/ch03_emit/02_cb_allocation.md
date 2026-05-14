# CB Allocation Methods

## Overview

Circular Buffers (CBs) are the fundamental data-passing mechanism in TT-Blaze. Every piece of data that flows through a fused kernel -- input tensors, intermediate results, scratch workspace, output tensors -- lives in a CB. The FusedProgram class provides several CB allocation methods, each suited to a different data source.

This section covers every allocation method, their constraints, and when to use each one.

## `f.cb_from_tensor(tensor)` -- CB Backed by a Sharded Tensor

The most common allocation method. It creates a CB backed by the L1 memory of a sharded tensor. The tensor must already be allocated on the device with a shard spec.

**Signature** (from `blaze/fused_program.py`):

```python
def cb_from_tensor(self, tensor, **kwargs) -> CBHandle:
```

**What it does:**

1. If `tensor` is already a `CBHandle`, returns it (with a FIFO access mode check via `require_fifo_handle()`).
2. If `tensor` is an `OverlappedView`, delegates to `cb_from_view()`.
3. Attempts disjoint-cores sharing: if another tensor-backed CB with matching `(data_format, tile_desc, page_size)` exists on disjoint cores, reuses that CB ID.
4. Otherwise, allocates a fresh CB ID via `self.program.cb_from_tensor()`.
5. Returns a `CBHandle` with all metadata derived from the tensor (page count from shard shape, page size from tile geometry, core ranges from shard spec grid).

**Simplified implementation** (from `blaze/fused_program.py`):

```python
def cb_from_tensor(self, tensor, **kwargs) -> CBHandle:
    if isinstance(tensor, CBHandle):
        return require_fifo_handle(tensor, "FusedProgram.cb_from_tensor")
    if isinstance(tensor, OverlappedView):
        return self.cb_from_view(tensor, **kwargs)
    reused = self._try_reuse_tensor_cb(tensor, **kwargs)
    if reused is not None:
        return reused
    cb_id = self.program.cb_from_tensor(tensor, **kwargs)
    self._track_cb_allocation(cb_id, tensor_backed=True)
    handle = self._make_tensor_handle(cb_id, tensor, **kwargs)
    self._record_cb_format(cb_id, handle.data_format, handle.tile_desc,
                           handle.core_ranges, page_size=handle.page_size)
    return handle
```

**Page size derivation:**

For tiled tensors, page size comes from `tile.get_tile_size(dtype)`. For row-major tensors, page size is `num_row_elements * element_size_bytes`. The method supports optional overrides via kwargs:

- `tile=`: Override tile descriptor (e.g., reinterpret 1x32 row tiles as 32x32)
- `page_size=`: Override page size in bytes
- `core_ranges=`: Override which cores see this CB
- `data_format=`: Override the data format
- `total_size=`: Sub-region size for overlapped allocations
- `address_offset=`: Byte offset within the tensor's L1 buffer

**Example from RMSNorm.emit():**

```python
# Standard: page size and tile derived from tensor metadata
inp = f.cb_from_tensor(input_handle)

# With overrides: reinterpreted tile geometry for row-tile inputs
inp = f.cb_from_tensor(input_handle, tile=itile, page_size=page_sz)
```

## `f.cb_scratch(name, ...)` -- Internal Scratch CB in L1

Allocates a CB in L1 that is NOT backed by any tensor. This is workspace memory -- it exists only during kernel execution and is not readable by the host after the kernel completes.

**Signature** (from `blaze/fused_program.py`):

```python
def cb_scratch(
    self,
    name: str,
    *,
    num_pages,
    core_ranges,
    data_format,
    tile,
    page_size,
    balanced: bool = True,
) -> CBHandle:
```

**All parameters are required** (except `balanced`). Unlike `cb_from_tensor()`, the caller must specify everything -- there is no tensor to derive metadata from.

**The `name` parameter and scratch_mapping:**

The `name` is critical. It is NOT just a label -- it is the key used by the `scratch_mapping` system. When a fused op is compiled with `all_scratch_mapped=True`, every scratch CB name must appear in the `scratch_mapping` dict. The mapping redirects the scratch allocation to a sub-region of a pre-allocated backing tensor (via `cb_from_tensor_overlapped()` internally), converting what would be a transient scratch buffer into a tensor-backed CB.

**Scratch CB naming convention:**

Scratch names use the triple-underscore delimiter from `BlazeOp.cb_name()`:

```python
f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "out_cb"),  # -> "rmsnorm___out_cb"
    ...
)
```

The pattern is `f"{prefix}___<suffix>"`. The `select_prefix` on FusedProgram is prepended to form the full lookup key: `f"{select_prefix}{name}"`.

Duplicate scratch names within the scratch mapping path (`_cb_scratch_from_mapping`) raise a `ValueError` -- the `_cb_scratch_names_seen` set enforces uniqueness for mapped scratch CBs:

```python
if mapped_key in cb_scratch_names_seen:
    raise ValueError(f"Duplicate scratch name encountered: {mapped_key!r}")
```

Note: This deduplication applies only within the mapping path. The non-mapping fallback allocates a fresh CB ID regardless of name duplication.

**Simplified implementation** (from `blaze/fused_program.py`):

```python
def cb_scratch(self, name, *, num_pages, core_ranges, data_format,
               tile, page_size, balanced=True) -> CBHandle:
    # First try the scratch_mapping system
    mapped_handle = self._cb_scratch_from_mapping(
        name=name, num_pages=num_pages, core_ranges=core_ranges,
        data_format=data_format, tile=tile, page_size=page_size,
    )
    if mapped_handle is not None:
        mapped_handle.scratch_name = name
        self._scratch_cbs.add(mapped_handle.cb_id)
        if not balanced:
            self._unbalanced_cbs.add(mapped_handle.cb_id)
        return mapped_handle

    # Fall through to fresh allocation
    cb_id = self.program.cb_scratch(
        name=name, num_pages=num_pages, core_ranges=core_ranges,
        data_format=data_format, tile=tile, page_size=page_size,
    )
    self._track_cb_allocation(cb_id, tensor_backed=False)
    self._scratch_cbs.add(cb_id)
    if not balanced:
        self._unbalanced_cbs.add(cb_id)
    handle = CBHandle(
        cb_id=cb_id, num_pages=num_pages, page_size=page_size,
        core_ranges=core_ranges, data_format=data_format,
        tile_desc=BlazeProgram._normalize_tile(tile) if tile else None,
        scratch_name=name,
    )
    self._cb_metadata[cb_id] = handle
    return handle
```

**The `balanced` flag:**

`balanced=True` (default) means every core that pushes to this CB also pops from it, keeping the FIFO state consistent. The temporal compaction pass can reuse the CB ID across phases.

`balanced=False` tells the compactor that some cores push without popping (or vice versa). This is common for multicast destinations -- every receiver core gets a push, but only a subset of cores (those running the consuming op) pop. Setting `balanced=False` excludes the CB from temporal reuse, preventing FIFO state leaks. From Mcast.emit():

```python
dst = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "dst"),
    num_pages=_dst_num_pages,
    core_ranges=f.all_cores,
    data_format=_dst_data_format,
    tile=_dst_tile_desc,
    page_size=_dst_page_size,
    # Conservative: we don't know at emit time whether the consumer
    # covers the full receiver grid. Any receivers that aren't
    # consumers push but don't pop, so flag unbalanced.
    balanced=False,
)
```

**Example from RMSNorm.emit():**

```python
out_cb = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "out_cb"),
    num_pages=compute_n,
    core_ranges=cores,
    data_format=out_fmt,
    tile=out_tile,
    page_size=out_page_sz,
)
```

## `f.cb_alias(source_cb)` -- Sharing L1 with Different Tile Format

Creates a second CB ID that reads the same L1 bytes as the source CB but interprets them with a different tile format, data type, or page size. This is useful when the same data needs to be consumed by two ops that expect different tile geometries.

**Signature** (from `blaze/fused_program.py`):

```python
def cb_alias(
    self,
    source: CBHandle,
    *,
    tile=None,
    page_size: int | None = None,
    total_size: int | None = None,
    dtype=None,
) -> CBHandle:
```

**What it does:**

1. Extends the source CB's lifetime to the current op index (prevents compaction from discarding the source while the alias is live).
2. Computes effective format: explicit kwargs override source's format.
3. Attempts disjoint-cores sharing: if another CB with matching format exists on disjoint cores, reuses that CB ID.
4. Calls `self.program.cb_alias()` to append a new `CBFormatDescriptor` to every `CBDescriptor` hosting the source CB ID.
5. Returns a new `CBHandle` with the alias CB ID, inheriting the source's backing tensor, byte offset, and access mode. The handle's `_node_id` and `_port_name` are copied from the source so shadow-graph edges attribute the data correctly.

**Key constraints:**
- Does not work with direct-address CBHandles (raises `ValueError`).
- The alias and source share the same L1 memory. They have different CB IDs, so the kernel can `cb_wait_front` on one while `cb_pop_front` on the other. This is two views of the same data, not a copy.

## `f.cb_from_view(view)` -- From OverlappedView for Fused Weight Buffers

Allocates a CB from an `OverlappedView`, which represents a logical sub-region of a larger fused L1 buffer. This is the primary mechanism for consuming pre-packed weight tensors where multiple logical tensors (e.g., Q/K/V projections) share a single contiguous L1 allocation.

**Signature** (from `blaze/fused_program.py`):

```python
def cb_from_view(self, view: "OverlappedView", **kwargs) -> CBHandle:
```

The `OverlappedView` dataclass (from `blaze/fused_program.py`):

```python
@dataclass(frozen=True)
class OverlappedView:
    tensor: ttnn.Tensor       # Fused device tensor (shared across views)
    shard_shape: tuple[int, int]  # Per-core shard shape of THIS sub-tensor
    core_ranges: ttnn.CoreRangeSet
    dtype: ttnn.DataType
    tile: object               # ttnn.Tile or TileDescriptor
    byte_offset: int           # Byte offset within the fused shard
    total_size: int            # Bytes per shard for this sub-tensor
```

The `OverlappedView` also provides a `page_size` property (computed from `tile.get_tile_size(dtype)`) and a `from_overlapped_tensor()` classmethod for constructing views from pre-allocated overlapped tensors.

**What it does:**

1. Checks if another view of the same backing tensor already has a CB ID with disjoint cores -- if so, reuses that CB ID (appending a per-grid CBDescriptor).
2. Falls through to format-key reuse: shares a CB ID with any CB (not just same-tensor) whose `(data_format, tile, page_size)` matches and whose grid is disjoint.
3. Otherwise allocates a fresh CB via `cb_from_tensor_overlapped()`.
4. Returns a `CBHandle` with the view's `byte_offset` so `handle.buffer_address()` resolves to the correct L1 sub-region.

**Optional overrides via kwargs:** `tile`, `page_size`, `core_ranges`, `data_format` -- any omitted values fall back to the view's attributes.

## `f.cb_from_shared_l1_views([...])` -- Multiple Direct-Address Handles from One Packed Tensor

Allocates a single CB ID for multiple views that all read from the same backing tensor but at different offsets. Returns a list of `CBHandle` objects, each with `access_mode=DIRECT_ADDRESS`.

**Signature** (from `blaze/fused_program.py`):

```python
def cb_from_shared_l1_views(self, views: list["OverlappedView"]) -> list[CBHandle]:
```

**What it does:**

1. Validates all views share the same backing tensor, dtype, tile shape, and page size.
2. Validates each view's core ranges are a subset of the backing tensor's shard grid.
3. Validates byte offsets are non-negative, page-aligned, and within the shard bounds.
4. Allocates one CB via `cb_from_tensor_overlapped()` at offset 0 on the union of all view cores.
5. Returns one `CBHandle` per view, each with `access_mode=CBAccessMode.DIRECT_ADDRESS` and the view's `byte_offset`.

**Key constraint:** The returned handles are direct-address -- consumers must read via `handle.buffer_address()` rather than CB FIFO front state. They cannot be passed to ops that call `require_fifo_handle()`.

**Validation is strict:** The method checks shard bounds, page alignment, and total-size consistency against `shard_shape / tile_shape * page_size`. Failing any check raises `ValueError` with a detailed message.

## `f.cb_output(tensor)` -- Marking an Output Tensor

Allocates a CB backed by an output tensor and marks it as a program output. Structurally similar to `cb_from_tensor()`, with the addition that the tensor is appended to `program._output_tensors`.

**Signature** (from `blaze/fused_program.py`):

```python
def cb_output(self, tensor, **kwargs) -> CBHandle:
```

**Why it should NOT be used directly in `emit()`:**

In the composition model, whether a scratch CB becomes an output is a decision made by the composition layer, not by the individual op. An op should allocate its output as a scratch CB (via `cb_scratch()`), and the calling code or compiler should wire it to an output tensor via `f.wire_output()` or the auto-wiring in `f.run()`.

Using `cb_output()` inside `emit()` hard-codes the assumption that this op is the terminal op in the pipeline. If someone later composes this op as an intermediate step, the output tensor registration creates a conflict. The scratch CB approach is universally safe: if the op is terminal, the composition layer binds its scratch output to a tensor; if the op is intermediate, the scratch CB simply passes to the next op.

The only valid use of `cb_output()` in `emit()` is when the op signature explicitly receives an output tensor (like `Copy.emit()`'s `output_tensor` parameter), and the caller's intent is clear:

```python
# From Copy.emit() -- the caller passes the output tensor explicitly
dst_cb = f.cb_from_tensor(output_tensor)
```

Note that Copy actually uses `cb_from_tensor()` here, not `cb_output()`. The output wiring happens at a higher level.

## Comparison Table

| Method | L1 Source | Backed by tensor? | FIFO/Direct | When to use |
|--------|-----------|-------------------|-------------|-------------|
| `cb_from_tensor()` | Sharded tensor | Yes | FIFO | Input data, weight tensors |
| `cb_scratch()` | Fresh L1 allocation | No (unless mapped) | FIFO | Intermediate compute buffers |
| `cb_alias()` | Same L1 as source | Inherited | Inherited | Reinterpret tile format |
| `cb_from_view()` | Sub-region of fused tensor | Yes | FIFO | Packed weight sub-tensors |
| `cb_from_shared_l1_views()` | Shared backing tensor | Yes | Direct-address | Multiple reads from one packed tensor |
| `cb_output()` | Output tensor | Yes | FIFO | Terminal output (use at composition layer) |

## Decision Guide

| Scenario | Method |
|----------|--------|
| Consuming an input tensor | `cb_from_tensor()` |
| Allocating working memory | `cb_scratch()` |
| Same data, different tile format | `cb_alias()` |
| Sub-region of a fused weight buffer | `cb_from_view()` |
| Multiple sub-regions sharing one CB ID | `cb_from_shared_l1_views()` |
| Terminal output binding | `cb_output()` at composition layer, not in emit() |

## CB ID Limits

From `blaze/cb_engine.py`:

```python
# Blackhole hardware limit
MAX_CB_ID = 64
```

Each CB allocation consumes one CB ID (0-63). Disjoint-core sharing (`_try_reuse_cb_id()`) and `cb_alias()` reuse existing IDs when the format matches and core grids do not overlap, but the 64-ID ceiling is hard. Complex fused ops with many intermediate buffers can approach this limit.

The CBEngine warns when usage exceeds 56 (64 - 8):

```python
if next_cb_id > self.max_cb_id - 8:
    warnings.warn(
        f"CB usage is high: {next_cb_id}/{self.max_cb_id} CBs assigned. "
        f"Approaching Blackhole hardware limit.",
        stacklevel=2,
    )
```

**Mitigation strategies when approaching the limit:**
- Enable disjoint-core sharing (the default) to reuse CB IDs across non-overlapping core grids.
- Use `cb_alias()` to share L1 between format views rather than allocating separate CBs.
- Consolidate scratch CBs where possible (e.g., reuse an intermediate buffer if its data is no longer needed).
- Let the temporal compaction pass reclaim CB IDs from CBs whose lifetimes do not overlap.

## Disjoint-Core CB Sharing

When two tensor-backed CBs have the same `(data_format, tile_desc, page_size)` and their core grids do not overlap, they can share a single CB ID. Each grid gets its own `CBFormatDescriptor` within the same `CBDescriptor`, so the kernel runtime binds the right buffer per core.

This optimization is managed by `_try_reuse_cb_id()` and `_record_cb_format()` on FusedProgram. The sharing index is a dict mapping `(data_format, (tile_h, tile_w), page_size)` to a list of `(cb_id, merged_core_ranges)` entries. Lookup is first-fit in insertion order.

The format key construction:

```python
def _format_key(data_format, tile_desc, page_size):
    """Construct the sharing lookup key for a CB format."""
    if tile_desc is not None:
        h, w = tile_desc.height, tile_desc.width
    else:
        h, w = None, None
    return (data_format, (h, w), page_size)
```

The disjointness check:

```python
def _are_disjoint(a, b) -> bool:
    """Check if two CoreRangeSets have no overlapping cores."""
    if a is None or b is None:
        return False
    return a.subtract(b).num_cores() == a.num_cores()
```

The reuse algorithm:

```python
def _try_reuse_cb_id(self, data_format, tile_desc, core_ranges, page_size):
    key = _format_key(data_format, tile_desc, page_size)
    entries = self._cb_format_index.get(key, [])
    for i, (ex_cb_id, ex_cr) in enumerate(entries):
        if _are_disjoint(ex_cr, core_ranges):
            entries[i] = (ex_cb_id, ex_cr.merge(core_ranges))
            return ex_cb_id
    return None
```

When reuse succeeds, the existing entry's core range is expanded via `.merge()`, preventing a third CB from sharing the same ID if it overlaps either of the first two grids.

**Debugging tip:** An environment variable `BLAZE_DISABLE_TENSOR_CB_SHARE=1` disables this optimization for bisection debugging when a shared CB ID appears to cause issues.

## Practical Guidance for New Op Authors

When writing `emit()` for a new MicroOp:

1. **Start with `cb_from_tensor()` for inputs.** The `isinstance(input, CBHandle)` passthrough pattern handles composition automatically.
2. **Use `cb_scratch()` for all intermediate and output buffers.** Name them with `BlazeOp.cb_name(prefix, "...")` for scratch_mapping compatibility.
3. **Set `balanced=False` only when you know some cores push without popping** (e.g., multicast destination CBs).
4. **Do not use `cb_output()` inside `emit()`.** Let the composition layer handle output wiring.
5. **Do not worry about CB ID exhaustion** unless your op allocates more than 5-6 CBs. Disjoint-core sharing and temporal compaction handle most cases automatically.

## Reference

- `blaze/fused_program.py` -- All allocation methods, disjoint-core sharing logic, OverlappedView
- `blaze/cb_handle.py` -- CBHandle dataclass definition
- `blaze/cb_engine.py` -- CB ID assignment engine, MAX_CB_ID constant
- `blaze/blaze_op.py` -- `BlazeOp.cb_name()`, `CB_NAME_DELIMITER`
