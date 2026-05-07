# 07.02 -- Advanced Composition Patterns

[<-- Composition Basics](01_composition_basics.md) | [Next: Worked Example -- SharedExpert -->](03_worked_example_shared_expert.md)

This section covers the advanced FusedProgram APIs that go beyond the basic `emit()` chain: CB aliasing, overlapped views, scratch arenas, multi-output ops, noop kernel prefixes, mesh-aware composition, and CB reconfiguration.

---

## CB Aliasing: f.cb_alias()

CB aliasing lets two CB IDs point at the same L1 bytes through different format interpretations. The common use case: an activation buffer was allocated with one tile geometry (e.g., 1x32 face-view tiles), but a downstream op needs to read those same bytes as a different geometry (e.g., 32x32 full tiles for RMSNorm).

### API

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

The method appends a new `CBFormatDescriptor` to every `CBDescriptor` that hosts the source CB ID. The result: one backing buffer, two CB IDs with different `(dtype, tile, page_size)` interpretations.

### How It Works

From `blaze/fused_program.py`:

```python
def cb_alias(self, source, *, tile=None, page_size=None, total_size=None, dtype=None):
    source_id = int(source)
    self._mark_cb_use(source_id)  # Extend source's lifetime

    eff_dtype = dtype if dtype is not None else source.data_format
    eff_tile = normalized_tile if tile is not None else source.tile_desc

    # Disjoint-cores sharing: aliases share _format_key_to_cb with tensor-backed CBs
    reused = self._try_reuse_cb_id(eff_dtype, eff_tile, source_alias_cores, page_size=eff_page_size)

    new_id = self.program.cb_alias(
        source_id,
        dtype=eff_dtype, tile=tile, page_size=page_size,
        total_size=total_size, reuse_cb_id=reused,
        core_ranges=source_alias_cores,
    )

    # Track: alias inherits tensor_backed and scratch status from source
    self._track_cb_allocation(new_id, tensor_backed=source_id in self._tensor_backed_cbs)
    if source_id in self._scratch_cbs:
        self._scratch_cbs.add(new_id)

    # Build new handle with the alias format
    handle = CBHandle(cb_id=new_id, num_pages=eff_num_pages, ...)
    handle._node_id = source._node_id      # inherit shadow-graph identity
    handle._port_name = source._port_name
    return handle
```

### Real Usage: MoE RMSNorm Input

In the MoE fused op (`blaze/ops/moe/op.py`), the activation tensor arrives with face-view (1xW) tiles. RMSNorm needs full-width row tiles. `cb_alias` provides a second view:

```python
# blaze/ops/moe/op.py -- inside MoE.emit()

if act.tile_desc.height == 1:
    itile, n_tiles = interpret_tile(width)
    page_sz = itile.get_tile_size(ttnn.bfloat16)
    rmsnorm_input = f.cb_alias(
        act, tile=itile, page_size=page_sz,
        total_size=page_sz * n_tiles,
    )
```

No data copy occurs. The kernel reads the same L1 address through the alias CB ID using the wider tile format.

### Key Constraints

- The alias must have a compatible total byte size (the source's L1 allocation is not resized).
- Direct-address handles cannot be aliased (`ValueError` if attempted).
- The alias inherits `_unbalanced_cbs` membership from the source.

---

## OverlappedView Pattern: f.cb_from_shared_l1_views()

When multiple weight matrices are packed into a single L1 allocation (a "fused weight tensor"), each matrix is a sub-region at a different byte offset. The `OverlappedView` dataclass and the `cb_from_shared_l1_views()` / `cb_from_view()` APIs are covered in detail in [Ch. 4.03 -- OverlappedView and Shared L1 Views](../ch04_data_flow/03_overlapped_view.md).

In composition context, the key usage patterns are:

**Direct-address for same-core views** (`cb_from_shared_l1_views`): Returns `CBHandle` objects with `access_mode=DIRECT_ADDRESS`. Kernels read via `handle.buffer_address()` rather than CB FIFO front state.

**FIFO for disjoint-core views** (`cb_from_view`): Returns standard FIFO handles with disjoint-core sharing. Internally checks `_view_tensor_cbs` for same-tensor reuse before allocating a new CB.

---

## Scratch Arena: f.scratch_tensors and f.scratch_mapping

The scratch arena maps named scratch CBs to pre-allocated backing tensors (see [Ch. 4.02 -- FusedProgram, Scratch Mapping](../ch04_data_flow/02_fused_program.md) for the constructor parameters and `cb_scratch()` API). The `select_prefix` parameter (set at `FusedProgram.__init__`) is prepended to every scratch mapping lookup key, enabling the same op to be instantiated multiple times with different mapping namespaces.

In practice, the pipeline builder pre-plans L1 layout and passes `scratch_mapping` at compilation time:

```python
scratch_mapping = {
    "moe__shared__gu___gate_out": {
        "tensor_name": "arena_0",
        "offset_address": 0,
    },
    "moe__shared__gu___up_out": {
        "tensor_name": "arena_0",
        "offset_address": 16384,
    },
}
```

When `f.cb_scratch(name=...)` is called, it checks `scratch_mapping` first. If a mapping exists, the scratch CB becomes tensor-backed via `cb_from_tensor_overlapped()`. If `all_scratch_mapped=True` is set at compile time, unmapped scratch names raise `ValueError` -- this catches layout planning bugs early.

---

## Multi-Output Ops

Some ops produce multiple named outputs. The `f.multi_output()` method handles this:

```python
# blaze/fused_program.py

def multi_output(
    self, op_name: str, outputs: list[tuple[str, CBHandle]],
    *, grid=None, prefix: str = "", **port_sources,
) -> MultiOutput:
```

It returns a `MultiOutput` object that supports positional unpacking, attribute access, and dict-style access:

```python
# Usage in an op's emit():
scores, indices = f.multi_output("moe_gate", [
    ("output", scores_handle),
    ("output_indices", indices_handle),
], prefix=prefix, input=act)

# Or:
result = f.multi_output(...)
result.output          # by name
result["output_indices"]  # dict-style
```

The `MultiOutput` wrapper (defined in `blaze/fused_program.py`, see [Ch. 4.03](../ch04_data_flow/03_overlapped_view.md)) supports positional unpacking, attribute access, and dict-style access. The first entry in the `outputs` list becomes the primary output for auto-wiring purposes (the compiler reads `f._last_output_cb_ids[0]`).

---

## Noop Kernel Prefixes

The `noop_prefixes` parameter selectively disables phases in a fused kernel. When passed to `BlazeCompiler.compile()`, any shadow-graph node whose prefix matches a noop entry is tagged with `noop_kernel=True`:

```python
# blaze/compiler.py

def _tag_noop_nodes(graph, noop_prefixes):
    for node in graph.nodes:
        if any(_graph_prefix_matches(_graph_node_prefix(node), p) for p in noop_prefixes):
            node.kwargs["noop_kernel"] = True
```

The kernel codegen then generates an empty body for tagged phases -- the hardware skips them entirely. This is useful for debugging (disable one branch of MoE), profiling (measure shared-expert cost by nooping the routed branch), or conditional compilation (disable shared expert on certain devices).

Usage at the compiler level:

```python
result = compiler.compile(
    graph=ctx.graph,
    tensors=tensors,
    output_tensor=output,
    noop_prefixes=["moe__routed"],  # Disable routed expert branch
)
```

---

## Mesh-Aware Composition

Three attributes on `FusedProgram` expose the multi-device topology to composition logic:

| Attribute       | Type              | Source |
|----------------|-------------------|--------|
| `f.mesh_coord` | `tuple[int, int]` | `(row, col)` of this device in the mesh |
| `f.mesh_shape` | `tuple[int, int]` | `(num_rows, num_cols)` of the mesh |
| `f.mesh_device`| `ttnn.MeshDevice`  | The mesh device object (for semaphore/tensor allocation) |

These are injected by `BlazeCompiler._compile_fused_op()` during the per-device compilation loop:

```python
# blaze/compiler.py -- per-device specialization

if "mesh_coord" in merged_args:
    f.mesh_coord = merged_args["mesh_coord"]
    f.mesh_shape = merged_args["mesh_shape"]
    f.mesh_device = merged_args["mesh_device"]
```

### Usage: MoE Index Offset

The MoE fused op computes a per-device expert index offset from mesh coordinates:

```python
# blaze/ops/moe/op.py -- MoE.emit()

row, col = f.mesh_coord
rows, cols = f.mesh_shape
index_offset = row * cols + col + routed_index_offset
```

### Usage: ReduceToOne Root Detection

The ReduceToOne op needs to know whether the current device is the reduction root:

```python
is_reduce_root = (row, col) == tuple(red2one_root_mesh_coord)
```

### Named Semaphores Across Devices

`f.semaphore(name)` allocates mesh-global semaphores deduped by name. The backing `_sem_dict` is shared across per-device `FusedProgram` iterations within one `BlazeCompiler.compile()` call:

```python
# blaze/fused_program.py -- semaphore()

def semaphore(self, name=None, *, program_semaphore=False, initial_value=0, ...):
    if program_semaphore:
        return self.program.program_semaphore(...)
    dev = self.mesh_device or self.device
    sem = _alloc_mesh_semaphore(dev, name, self._sem_dict, initial_value=initial_value)
    self.program._global_semaphores.append(sem)
    return int(ttnn.get_global_semaphore_address(sem))
```

The first device iteration allocates; subsequent device iterations return the cached semaphore from `_sem_dict`. The same pattern applies to `f.named_tensor()` for data-bearing allocations.

---

## CB Reconfiguration

When a fused kernel has more logical CBs than the hardware's 64-slot limit, or when phases need incompatible CB formats at the same slot, multi-phase CB reconfiguration is used. The `CircularBufferIdManager`, the six-step build pass (`_build_multi_phase`), scratch CB compaction, and the `CBContext` manual-control API are all covered in [Ch. 4.04 -- CB Reconfig](../ch04_data_flow/04_cb_reconfig.md).

In composition, the key interaction points are:

- **Marking boundaries**: An op calls `f.reconfig()` or emits a `CbReconfig` MicroOp to mark a phase boundary. This clears the format-key and view-tensor sharing caches so Phase N+1 allocations do not accidentally reuse Phase N CB IDs.
- **Automatic handling**: The build pass runs automatically inside `f.build()` via `_prepare_for_build()`. No composition-level code beyond calling `f.reconfig()` is needed.

For the routed expert's CB reconfiguration between gate-up and down phases, see [Section 04](04_worked_example_moe.md).

---

## Summary Table

| Pattern | API | When to Use |
|---------|-----|-------------|
| **CB alias** | `f.cb_alias(source, tile=..., page_size=...)` | Reinterpret same L1 bytes with different tile geometry |
| **Shared L1 views** | `f.cb_from_shared_l1_views([view1, view2])` | Multiple weight sub-regions in one fused tensor |
| **Single view** | `f.cb_from_view(view)` | One sub-region of a fused tensor |
| **Scratch mapping** | `f.cb_scratch(name=...) + scratch_mapping` | Pre-planned L1 layout from pipeline builder |
| **Multi-output** | `f.multi_output(op_name, [(name, handle), ...])` | Ops producing multiple named outputs |
| **Noop prefixes** | `noop_prefixes=["prefix"]` at compile time | Disable phases for debug/profiling |
| **Mesh context** | `f.mesh_coord`, `f.mesh_shape`, `f.mesh_device` | Device-aware routing, index offsets |
| **CB reconfig** | `f.reconfig()` or `CbReconfig.emit(f, prefix=...)` | More than 64 logical CBs, or incompatible formats across phases |

[<-- Composition Basics](01_composition_basics.md) | [Next: Worked Example -- SharedExpert -->](03_worked_example_shared_expert.md)
