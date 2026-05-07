# 02 -- FusedProgram

**Source file:** `blaze/fused_program.py`

FusedProgram is the composition context: it holds the device, grid info, and underlying BlazeProgram, and exposes the CB allocation, CT-arg, RT-arg, semaphore, per-core flag, and graph-recording APIs that `emit()` functions call. This section covers every public method and constructor parameter, the internal tracking state that enables multi-phase reconfiguration and scratch compaction, and the build/run pipeline that turns a sequence of `emit()` calls into an executable program.

## Constructor -- Every Parameter

```python
class FusedProgram:
    def __init__(
        self,
        kernel: str | None,
        device,
        *,
        math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
        math_approx_mode: bool = False,
        fp32_dest_acc_en: bool = False,
        dst_full_sync_en: bool = False,
        name: str | None = None,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        _sem_dict: dict | None = None,
        _tensor_dict: dict | None = None,
        _internal_tensor_dict: dict | None = None,
        scratch_tensors: dict | None = None,
        scratch_mapping: dict[str, dict] | None = None,
        select_prefix: str = "",
        all_scratch_mapped: bool = False,
        compact_scratch_cbs: bool = True,
    ):
```

### Required parameters

| Parameter | Purpose |
|---|---|
| `kernel` | Path to a `.cpp` kernel file, or `None` for auto-codegen from the shadow graph. When `None`, the kernel is generated at `build()` time from the recorded graph via `set_kernel_from_graph()`. |
| `device` | A `ttnn.Device` or `ttnn.MeshDevice`. Used to query the compute grid, NOC coordinates, and as the target for tensor allocations. |

### Compute config

| Parameter | Default | Purpose |
|---|---|---|
| `math_fidelity` | `HiFi4` | Fidelity mode for SFPU math operations. Passed through to `ComputeConfigDescriptor`. |
| `math_approx_mode` | `False` | Enable approximate math for transcendentals (e.g. `exp`, `recip`). |
| `fp32_dest_acc_en` | `False` | Use FP32 accumulation in the destination register for higher precision. |
| `dst_full_sync_en` | `False` | Force full synchronization of the DST register between math ops. |
| `noc_mode` | `DM_DYNAMIC_NOC` | NOC routing mode for data movement. |

### Naming and identity

| Parameter | Default | Purpose |
|---|---|---|
| `name` | `None` | Human-readable name for the program. Used as the filename stem for generated kernel code. Passed to `set_kernel_from_graph(graph, name=name)`. |

### Cross-iteration dedup dicts

These three dicts are injected by `BlazeCompiler` or `MeshFusedProgram` to share allocations across per-device compile iterations. When `None` (default for standalone use), a fresh dict is created per FusedProgram instance.

| Parameter | Default | Contents |
|---|---|---|
| `_sem_dict` | `None` | `{name: ttnn.GlobalSemaphore}` -- deduplicates `f.semaphore(name)` calls across iterations. |
| `_tensor_dict` | `None` | `{name: ttnn.Tensor}` -- deduplicates `f.named_tensor(name)` calls across iterations. |
| `_internal_tensor_dict` | `None` | `{cache_key: ttnn.Tensor}` -- deduplicates compiler-internal allocations (reconfig tensors, scratch arenas) keyed by content signature. |

### Scratch mapping

| Parameter | Default | Purpose |
|---|---|---|
| `scratch_tensors` | `None` | `{tensor_name: ttnn.Tensor}` -- pre-allocated backing tensors for scratch-mapped CBs. |
| `scratch_mapping` | `None` | `{scratch_key: {"tensor_name": str, "offset_address": int}}` -- maps each scratch CB name to a region in a backing tensor. |
| `select_prefix` | `""` | Prepended to scratch CB names before looking up `scratch_mapping`. Enables selecting different mapping tables for different kernel variants (e.g. prefill vs decode). |
| `all_scratch_mapped` | `False` | When `True`, every `cb_scratch()` call must find a mapping entry. Unmapped scratch raises `ValueError`. Safety net for configurations that promise all scratch is pre-allocated. |
| `compact_scratch_cbs` | `True` | Enable scratch CB compaction (spatial + temporal reuse). Set to `False` to disable for debugging. |

## Precomputed Grid Attributes

The constructor derives several grid attributes from `DeviceContext`:

```python
self._ctx = DeviceContext.from_device(device)
self.grid = self._ctx.grid_config           # GridConfig dataclass
self.sender = ttnn.CoreCoord(*self.grid.sender_core)
self.sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(self.sender, self.sender)])
self.mcast_range = ttnn.CoreRange(
    ttnn.CoreCoord(0, 0),
    ttnn.CoreCoord(self.grid.grid_cols - 1, self.grid.grid_rows - 1),
)
self.all_cores = ttnn.CoreRangeSet([self.mcast_range])
self.num_mcast_cores = self.grid.total_cores
self.full_device_grid = self._ctx.full_device_grid
```

`full_device_grid` is the entire device grid (all physical cores), not the compute grid. It is used when allocating the reconfig tensor and scratch arena, which must cover every core for the `HEIGHT_SHARDED` shard spec.

The constructor also computes `matmul_cores`, `matmul_cores_cc`, `mcast_receiver_grid`, and NOC coordinates (`noc_sender`, `noc_start`, `noc_end`) derived from `DeviceContext.worker_core_from_logical_core()`. These precomputed grids eliminate repeated CoreRangeSet construction inside op emit functions -- ops reference `f.all_cores`, `f.matmul_cores`, `f.sender_grid` directly.

## CB Allocation APIs

FusedProgram wraps every BlazeProgram CB allocator, adding CBHandle metadata tracking, lifetime recording, and disjoint-cores sharing. All allocators return `CBHandle` objects (not bare `int` IDs like BlazeProgram does).

### cb_from_tensor(tensor, **kwargs)

The primary allocator. Accepts a `ttnn.Tensor` (sharded, with L1 storage), an `OverlappedView`, or a pre-existing `CBHandle`:

- If `tensor` is a `CBHandle`: calls `require_fifo_handle()` and returns it unchanged (passthrough for already-allocated inputs).
- If `tensor` is an `OverlappedView`: delegates to `cb_from_view()`.
- Otherwise: attempts disjoint-cores sharing via `_try_reuse_tensor_cb()`. On miss, calls `BlazeProgram.cb_from_tensor()`, tracks the allocation, builds a `CBHandle` via `_make_tensor_handle()`, and records the format for future sharing.

Keyword overrides: `tile`, `page_size`, `core_ranges`, `data_format`, `address_offset`, `total_size`.

### cb_from_tensor_overlapped(tensor, address_offset, total_size, page_size, ...)

Allocates a CB for a sub-region of a sharded tensor's L1 buffer. The `address_offset` positions the CB within the tensor's shard, and `total_size` limits the region. Follows the same sharing/tracking pattern as `cb_from_tensor`.

### cb_output(tensor, **kwargs) / cb_output_overlapped(...)

Same as `cb_from_tensor` / `cb_from_tensor_overlapped`, but additionally marks the tensor as a program output (appends to `program._output_tensors`).

### cb_from_view(view, **kwargs)

Allocates a CB from an `OverlappedView`. Three paths:

1. **Same-tensor, disjoint cores**: If `_view_tensor_cbs[id(tensor)]` exists and the existing grid is disjoint from the new view's cores, appends a per-grid `CBDescriptor` to the existing CB ID.
2. **Format-key reuse**: If no same-tensor match but `_try_reuse_cb_id()` finds a matching `(data_format, tile, page_size)` on a disjoint grid, reuses that CB ID.
3. **New allocation**: Falls through to `BlazeProgram.cb_from_tensor_overlapped()`.

Accepts optional overrides for `tile`, `page_size`, `core_ranges`, `data_format`; omitted values fall back to the view's attributes. Covered in detail in section 03.

### cb_from_shared_l1_views(views)

Covered in detail in section 03. Returns a list of `CBHandle` objects with `access_mode=DIRECT_ADDRESS`.

### cb_scratch(name, *, num_pages, core_ranges, data_format, tile, page_size, balanced=True)

Allocates a scratch CB -- L1 storage without a backing tensor. The flow:

1. Checks `_cb_scratch_from_mapping()` -- if `scratch_mapping` contains an entry for `{select_prefix}{name}`, allocates via `cb_from_tensor_overlapped()` against the mapped backing tensor.
2. If no mapping (or `scratch_mapping` is empty), calls `BlazeProgram.cb_scratch()` which allocates a bare CB descriptor with no L1 backing.

The `balanced` parameter (default `True`) marks whether push/pop operations are balanced per core. Set `balanced=False` for CBs like multicast destinations where the sender pushes but only a subset of receivers pops. Unbalanced CBs are excluded from temporal compaction because residual FIFO state could leak across reuse.

### cb_alias(source, *, tile, page_size, total_size, dtype)

Creates a second CB ID that reads the same L1 bytes as `source` through a different format view. Appends a `CBFormatDescriptor` to every `CBDescriptor` hosting the source CB ID. The implementation:

1. Extends the source's lifetime to the alias-creation op (prevents compaction from folding the source away).
2. Attempts disjoint-cores sharing: `_try_reuse_cb_id()` with the effective `(dtype, tile, page_size)`. Aliases share `_format_key_to_cb` with tensor-backed CBs, so cross-layer folds are possible.
3. Calls `BlazeProgram.cb_alias()` which attaches the format descriptor and optionally resizes the backing descriptor.
4. Inherits `_node_id`, `_port_name`, `tensor_backed` status, `scratch_cbs` membership, and `unbalanced_cbs` membership from the source handle.

### set_cb_core_ranges(handle, core_ranges)

Updates `core_ranges` on both the CBHandle and the underlying `CBDescriptor`. Finds the descriptor by matching `fmt.buffer_index == handle.cb_id`.

### wire_output(handle, output_tensor) / wire_output_from_graph(output_tensor)

Converts an intermediate scratch CB into a tensor-backed output CB. `wire_output` replaces the CB descriptor with one backed by the output tensor, preserving tile and page_size from the original (the kernel was compiled against those formats). `wire_output_from_graph` auto-detects the output CB from `_last_output_cb_ids` -- called by BlazeCompiler when `compose()` does not explicitly wire an output. Both reject direct-address handles.

## CT-Arg and RT-Arg Declaration

FusedProgram wraps every BlazeProgram arg declaration method, adding shadow-graph capture and CB position tracking. See [Ch5 S01 -- Per-RISC Model](../ch05_risc_compilation/01_per_risc_model.md) for the complete BlazeProgram method reference and RISC target mappings.

### Compile-time args (CT)

```python
f.unified_ct_args(args, **kwargs)      # all three RISCs
f.ncrisc_ct_args(args, **kwargs)       # NCRISC only
f.brisc_ct_args(args, **kwargs)        # BRISC only
f.trisc_ct_args(args, **kwargs)        # TRISC only
```

Each wrapper does three things:
1. `_track_cb_arg_positions(args, riscs)` -- records the position of any CBHandle-typed value so multi-phase remap can patch it later.
2. `_capture_ct_values(args)` -- coerces all values to `int` and accumulates them in `_pending_ct_values` for the next shadow graph node.
3. Forwards to `self.program.<method>(args, **kwargs)`.

The `_track_cb_arg_positions` method is the critical link between CBHandle and the build pipeline:

```python
def _track_cb_arg_positions(self, args, riscs: list[str]):
    cb_args = [(i, v) for i, (_, v) in enumerate(args) if isinstance(v, CBHandle)]
    if not cb_args:
        return
    for i, v in cb_args:
        for risc in riscs:
            arg_list = self._get_risc_args(risc)
            idx = len(arg_list) + i
            self._cb_arg_positions.append((risc, idx, v.cb_id))
        self._mark_cb_use(v.cb_id)
```

When multi-phase reconfig later remaps CB IDs, it rewrites these tracked positions directly. If an op passed `int(handle)` instead of the CBHandle object, the position is not tracked, and the remap falls back to a slower spec-based heuristic that scans for CB-typed field names. The `_warn_untracked_remaps` function logs warnings for these fallback cases.

### Positional CT args

```python
f.ncrisc_positional_ct_args(args)
f.brisc_positional_ct_args(args)
```

For APIs that read CT args by index (`get_compile_time_arg_val(index)`) rather than by name. No shadow-graph capture -- these are infrastructure-level (e.g. `TensorAccessorArgs`).

### Runtime args (RT)

```python
f.ncrisc_rt_args(args)        # named common RT args
f.brisc_rt_args(args)
f.trisc_rt_args(args)
f.ncrisc_rt_arg_arrays(args)  # named RT arg arrays
f.brisc_rt_arg_arrays(args)
f.trisc_rt_arg_arrays(args)
f.ncrisc_per_core_rt_arg_arrays(args)   # per-core RT arg arrays
f.brisc_per_core_rt_arg_arrays(args)
f.trisc_per_core_rt_arg_arrays(args)
```

These are thin pass-throughs to BlazeProgram (no CB tracking needed for runtime args).

### Per-core CT args

```python
f.per_core_unified_ct_args(args, **kwargs)
f.per_core_ncrisc_ct_args(args, **kwargs)
f.per_core_trisc_ct_args(args, **kwargs)
f.per_core_brisc_ct_args(args, **kwargs)
```

Each call validates that the arg name has not been seen before via `_per_core_ct_args_check`, then forwards to `BlazeProgram`. Duplicate names raise `ValueError` -- a safety net against two ops accidentally writing the same per-core arg.

## Per-Core Flags

```python
def flag(self, name, cores, enabled=True):
    return (name, {cores: int(enabled)})
```

A convenience that builds a `(name, mapping)` tuple for `per_core_unified_ct_args`. The flag is 1 on the specified cores when `enabled=True`, 0 when `enabled=False`. Cores not in the range always get 0 (via the `other_value` default in BlazeProgram). Typical usage:

```python
f.per_core_unified_ct_args([
    f.flag("is_sender", f.sender_grid),
    f.flag("is_matmul_core", f.matmul_cores),
])
```

## Semaphores

```python
def semaphore(
    self,
    name: str | None = None,
    *,
    program_semaphore: bool = False,
    initial_value: int = 0,
    core_ranges: ttnn.CoreRangeSet | None = None,
) -> int:
```

Two allocation models:

1. **Named mesh-global** (`name` set): Allocates a `ttnn.GlobalSemaphore` on the compute-with-storage grid (`mesh_device.compute_with_storage_grid_size()`), keyed by name in `_sem_dict`. Returns the L1 address as an integer. Same-name calls across FusedPrograms sharing a `_sem_dict` return the same semaphore (dedup). Calling the same name twice within one op raises `RuntimeError`. `core_ranges` is not valid here.

2. **Program-local slot** (`program_semaphore=True`): Allocates a per-device slot-based semaphore via `BlazeProgram.program_semaphore()`. Not address-tied; safe for local synchronization in per-device compile mode. `core_ranges` restricts the semaphore to specific cores, allowing tt-metal to reuse slot IDs across disjoint cores.

The unnamed path (both `name=None` and `program_semaphore=False`) raises `RuntimeError` with a message directing the caller to choose one of the two models.

## named_tensor()

```python
def named_tensor(
    self,
    name: str,
    *,
    torch_tensor,
    dtype,
    layout,
    memory_config,
    tile,
) -> ttnn.Tensor:
```

Allocates a mesh-global tensor keyed by name, deduped across FusedPrograms sharing a `_tensor_dict`. First call allocates via `ttnn.from_torch()`; later calls return the cached tensor. Same-name calls within one op raise `RuntimeError`. Returns a `ttnn.Tensor` (not a CBHandle) -- callers typically follow up with `f.cb_from_tensor()`. The tensor is pinned for the lifetime of the resulting `MeshCompiledProgram`.

## output() and multi_output()

### output(op_name, handle, *, prefix, **port_sources) -> CBHandle

The primary recording method. Creates an `OpNode`, wires input edges from `port_sources`, stamps `_node_id` and `_port_name` on the returned handle, advances `_op_index`, and stores the handle in `_last_output_cb_ids`. Also triggers injected fabric barrier logic for CCL ops.

### multi_output(op_name, outputs, *, prefix, **port_sources) -> MultiOutput

For ops that produce multiple named outputs. `outputs` is a `list[tuple[str, CBHandle]]` where the string is the port name. Returns a `MultiOutput` object supporting positional unpacking, attribute access, and dict-style access:

```python
scores, indices = DeepseekMoeGate.emit(f, ...)   # positional
result = DeepseekMoeGate.emit(f, ...)
result.output                                     # by name
result["output_indices"]                          # dict-style
```

## Internal Tracking State

FusedProgram maintains extensive internal state for the build pipeline:

| Attribute | Type | Purpose |
|---|---|---|
| `_cb_metadata` | `dict[int, CBHandle]` | CB ID to handle metadata. The FIRST allocator's geometry on shared IDs. |
| `_tensor_handle_cache` | `dict[int, CBHandle]` | `id(tensor)` to handle. Each caller's own handle for `cb_for_tensor()` lookups. |
| `_cb_arg_positions` | `list[tuple[str, int, int]]` | `(risc, arg_index, cb_id)` -- positions of CBHandle-typed CT args for build-time remap. |
| `_op_index` | `int` | Monotonic counter advanced by `output()` / `multi_output()`. |
| `_cb_lifetimes` | `dict[int, tuple[int, int]]` | `cb_id -> (first_use_op, last_use_op)`. Used by compaction and cross-boundary validation. |
| `_cb_phase` | `dict[int, int]` | `cb_id -> phase_index`. Determined by how many `reconfig()` boundaries preceded the allocation. |
| `_reconfig_boundaries` | `list[int]` | Op indices where `reconfig()` was called. |
| `_tensor_backed_cbs` | `set[int]` | CB IDs backed by tensors (not scratch). |
| `_direct_address_cbs` | `set[int]` | CB IDs using direct-address mode. |
| `_scratch_cbs` | `set[int]` | CB IDs allocated via `cb_scratch()`. |
| `_unbalanced_cbs` | `set[int]` | Scratch CBs with unbalanced push/pop (excluded from temporal reuse). |
| `_view_tensor_cbs` | `dict[int, tuple[int, object]]` | `id(tensor) -> (cb_id, core_ranges)` for same-tensor view sharing. |
| `_format_key_to_cb` | `dict[tuple, list[tuple[int, object]]]` | Format-key sharing index. Key: `(data_format, (tile_h, tile_w), page_size)`. Value: `[(cb_id, merged_core_ranges), ...]`. Multi-entry, first-fit lookup over disjoint-cores entries. |
| `_reconfig_placeholders` | `list[tuple[int, object, dict]]` | `(phase_idx, OpNode, {risc: arg_idx})` for build-time address patching. |
| `_compaction_applied` | `bool` | Guard against double-applying compaction/reconfig. |

## Worked Example: 3-Op Pipeline (Mcast -> Matmul -> Reduce)

Consider a decode attention pipeline: multicast distributes activations to all cores, matmul computes Q*K^T, and reduce sums partial results back to the sender core.

### Phase 0: Mcast Activation

```python
# Inside Mcast.emit(f, act_tensor, prefix="act_mcast"):
act_cb = f.cb_from_tensor(act_tensor)
```

Inside `cb_from_tensor`:
1. `_try_reuse_tensor_cb(act_tensor)` -- no existing CB, returns None.
2. `self.program.cb_from_tensor(act_tensor)` allocates `cb_id=0`.
3. `_track_cb_allocation(0, tensor_backed=True)` records `_cb_phase[0] = 0`, `_cb_lifetimes[0] = (0, 0)`.
4. `_make_tensor_handle(0, act_tensor)` builds a CBHandle with geometry from the tensor's shard spec.
5. `_record_cb_format(0, bfloat16, TileDesc(32,32), sender_grid, page_size=2048)` registers for future disjoint-core sharing.

The mcast op also allocates a scratch CB for the multicast destination:

```python
mcast_dst = f.cb_scratch(
    name="act_mcast___dst",
    num_pages=2,
    core_ranges=f.all_cores,
    data_format=ttnn.bfloat16,
    tile=tile_desc,
    page_size=2048,
    balanced=False,     # sender pushes, not all cores pop
)
```

This creates `cb_id=1`, marked as `_scratch_cbs` and `_unbalanced_cbs`.

CT args are declared:

```python
f.unified_ct_args([
    (f"{prefix}.src_cb", act_cb),       # int(act_cb) -> 0
    (f"{prefix}.dst_cb", mcast_dst),    # int(mcast_dst) -> 1
    (f"{prefix}.num_tiles", 32),
])
```

`_track_cb_arg_positions` records that `act_cb` appears at RISC arg index 0 and `mcast_dst` at index 1 for all three RISCs.

Finally, `f.output("mcast", mcast_dst, input=act_cb, prefix="act_mcast")` stamps the shadow graph node and edges, advances `_op_index` to 1.

**State after mcast:**
- `cb_id=0`: act_tensor (tensor-backed), lifetime (0, 0)
- `cb_id=1`: scratch dst (unbalanced), lifetime (0, 1)
- `_op_index = 1`

### Phase 1: Matmul

```python
# Inside Matmul.emit(f, mcast_dst, weight_tensor, prefix="matmul"):
weight_cb = f.cb_from_tensor(weight_tensor, core_ranges=f.matmul_cores)
```

`cb_id=2` is allocated (or `cb_id=0` is reused if act_tensor and weight_tensor have matching `_format_key` and `f.sender_grid` is disjoint from `f.matmul_cores`).

The matmul scratch output:

```python
matmul_out = f.cb_scratch(
    name="matmul___dst",
    num_pages=4,
    core_ranges=f.matmul_cores,
    data_format=ttnn.bfloat16,
    tile=tile_desc,
    page_size=2048,
)
```

CT args are declared, `_mark_cb_use(1)` extends mcast_dst's lifetime, and `f.output("matmul", matmul_out, ...)` advances `_op_index` to 2.

### Phase 2: Reduce

```python
# Inside Reduce.emit(f, matmul_out, output_tensor, prefix="reduce"):
out_cb = f.cb_from_tensor(output_tensor)
```

CT args are declared and `f.output("reduce", out_cb, input=matmul_out)` records the final op.

**Final state before build:**
- `cb_id=0`: act_tensor (+ possibly weight_tensor shared, tensor-backed)
- `cb_id=1`: mcast dst (scratch, unbalanced), lifetime (0, 2)
- `cb_id=2`: matmul dst (scratch, balanced), lifetime (1, 2)
- `cb_id=3`: output tensor (tensor-backed)
- Shadow graph: `mcast_1 -> matmul_1 -> reduce_1`

### The Complete Pipeline Diagram

```
  FusedProgram(device, name="decode_attn")
  |
  |-- Mcast.emit(f, act_tensor, prefix="act_mcast")
  |   |-- f.cb_from_tensor(act_tensor)                -> cb_id=0 (tensor-backed)
  |   |-- f.cb_scratch("act_mcast___dst", ...)         -> cb_id=1 (scratch)
  |   |-- f.unified_ct_args([src_cb=0, dst_cb=1, ...])
  |   |-- f.output("mcast", handle_1, input=handle_0)  -> _op_index=1
  |
  |-- Matmul.emit(f, handle_1, weight_tensor, prefix="matmul")
  |   |-- f.cb_from_tensor(weight_tensor)              -> cb_id=0 or 2
  |   |-- f.cb_scratch("matmul___dst", ...)            -> cb_id=2 or 3
  |   |-- f.unified_ct_args([in_act=1, weight=0, out=2, ...])
  |   |-- f.output("matmul", handle_2, ...)            -> _op_index=2
  |
  |-- Reduce.emit(f, handle_2, out_tensor, prefix="reduce")
  |   |-- f.cb_from_tensor(out_tensor)                 -> cb_id=3
  |   |-- f.unified_ct_args([in=2, out=3, ...])
  |   |-- f.output("reduce", handle_3, ...)            -> _op_index=3
  |
  |-- f.build()
      |-- _prepare_for_build()
      |   |-- _compact_scratch_cbs()    # merge cb_id=1 and cb_id=2 if possible
      |   |-- _build_multi_phase()      # if reconfig boundaries exist
      |-- set_kernel_from_graph()       # codegen from shadow graph
      |-- program.build()               # ProgramDescriptor
      |-- CompiledProgram(pd, [act_tensor, weight_tensor, out_tensor])
```

## Disjoint-Core Sharing

The `_format_key` function creates a dedup key from `(data_format, (tile_h, tile_w), page_size)`:

```python
def _format_key(data_format, tile_desc, page_size=None) -> tuple | None:
    if data_format is None or tile_desc is None:
        return None
    h = getattr(tile_desc, "height", None)
    w = getattr(tile_desc, "width", None)
    if h is None or w is None:
        return None
    return (data_format, (h, w), page_size)
```

When two ops allocate CBs with matching format keys on disjoint core ranges, they share one `cb_id`. The second allocation appends a per-grid `CBFormatDescriptor` to the existing `CBDescriptor` rather than creating a new one. This reduces the CB ID count (the hardware limit is 64).

The `page_size` is included in the key because mismatched page sizes on the same `cb_id` would leave the descriptor inconsistent across grids -- the kernel runtime cannot represent one CB ID with two different page sizes on overlapping cores.

Returns `None` (skip sharing) when `data_format`, `tile_desc`, or tile dimensions are missing. This prevents sharing for row-major CBs or CBs constructed without tile metadata. Sharing is also disabled when the `BLAZE_DISABLE_TENSOR_CB_SHARE=1` environment variable is set -- a bisection escape hatch for debugging.

## build() and run()

### build() -> CompiledProgram

```python
def build(self, noc_mode=None) -> CompiledProgram:
    self._prepare_for_build()
    if self.program._kernel is None:
        self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)
    pd = self.program.build()
    io_tensors = [t for _, t in sorted(self.program._io_tensors, key=lambda x: x[0])]
    return CompiledProgram(pd, io_tensors, ...)
```

`_prepare_for_build()` runs two passes:
1. `cb_reconfig_prepare_to_build(self)` -- scratch compaction and multi-phase reconfig (see section 04).
2. `injected_fabric_barrier_prepare_to_build(self)` -- inserts fabric barrier ops for CCL workloads.

Then generates the kernel from the shadow graph if not provided, builds the `ProgramDescriptor` via BlazeProgram, and returns a `CompiledProgram`.

### run(output_tensor=None, noc_mode=None) -> ttnn.Tensor

Calls `_prepare_for_build()`, auto-wires the output tensor if not explicitly set (using `_last_output_cb_ids` or the last `_io_tensors` entry), and executes the program. Returns the output tensor.

## Context Manager

```python
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    for t in self._cleanup_tensors:
        ttnn.deallocate(t)
    self._cleanup_tensors.clear()
    return False
```

`register_cleanup(*tensors)` adds tensors to `_cleanup_tensors`. The context manager deallocates them on exit. This is the mechanism for scratch arena tensors and other intermediate allocations.

## MeshFusedProgram

`MeshFusedProgram` wraps per-device `FusedProgram` instances for mesh execution. It creates one FP per `(row, col)` in the mesh, all sharing `_sem_dict`, `_tensor_dict`, and `_internal_tensor_dict` so allocations dedup across devices.

```python
mfp = MeshFusedProgram(mesh_device=mesh_device, kernel=None, name="my_op")
mfp.compose(compose_fn, tensors, output_tensor, per_device_args=lambda r, c: {...})
result = mfp.build(mesh_io_tensors, mesh_output_tensor)
```

`compose()` calls `compose_fn(fp, *args, **{**kwargs, **extra})` for each device. `per_device_args` is an optional callable `(row, col) -> dict` for per-device specialization.

`build()` calls `fp.build()` on each device's FP, assembles a `ttnn.MeshProgramDescriptor` mapping each `MeshCoordinateRange` to its `ProgramDescriptor`, and returns a `MeshCompiledProgram`. The `lifetime_tensors` list pins shared named tensors and internal tensors for the compiled program's lifetime.

## Multi-Device and Mesh Attributes

```python
self.mesh_coord: tuple[int, int] | None = None
self.mesh_shape: tuple[int, int] | None = None
self.mesh_device = None
```

Set by `MeshFusedProgram` or `BlazeCompiler` when compiling for a multi-device mesh. `mesh_coord` identifies this FusedProgram's device position. `mesh_device` is the `MeshDevice` object used for mesh-global allocations (semaphores, named tensors).

## Grid Helpers

### build_ab_grids(ab_coords=None)

Builds an `ABGrid` for KN-sliced matmul with gate/up split. If `ab_coords` is not provided, derives from `self.grid.build_ab_grids()`. Returns an `ABGrid` dataclass with `a_cores`, `b_cores`, `a_grid`, `b_grid`, `compute_grid`, `num_per_branch`.

### set_core_ranges(core_ranges)

Overrides which cores the kernel runs on. Default is `self.all_cores` (full compute grid). CCL ops override to a single worker core since the kernel runs on one core per device.

### add_define(name, value="1")

Adds a preprocessor `#define` for kernel compilation.

---

<- [01 -- CBHandle Abstraction](01_cbhandle_abstraction.md) | [Index](index.md) | [03 -- OverlappedView](03_overlapped_view.md) ->
