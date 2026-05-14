# 9.2 Performance Optimization and L1 Management

This section covers the performance-critical aspects of micro-op development: circular buffer allocation strategies, L1 memory management, multi-phase CB reconfiguration, scratch compaction, and profiling tools.

---

## CB Allocation Strategies

TT-Blaze offers four distinct CB allocation strategies, each with different L1 memory semantics. Choosing the right strategy is the single most impactful performance decision in micro-op design.

### Strategy 1: Tensor-Backed CBs (`cb_from_tensor`)

**When to use**: For CBs that read from or write to external tensors.

```python
input_handle = f.cb_from_tensor(input_tensor)
```

`FusedProgram.cb_from_tensor()` (source: `blaze/fused_program.py`, line 1050) creates a CB descriptor that shares the tensor's L1 buffer. The CB ID, page size, tile format, and core ranges are derived from the tensor's shard spec.

Key behaviors:
- **Disjoint-cores sharing**: Before allocating a new CB ID, `_try_reuse_tensor_cb()` (source: lines 997-1048) probes the `_format_key_to_cb` index for an existing CB ID with matching `(data_format, tile_shape, page_size)` whose grid is disjoint from the caller's. On hit, a second `CBFormatDescriptor` is appended to the existing descriptor. This lets two ops on non-overlapping core grids share one CB ID for tensors of the same format.
- **Format key composition**: The format key is computed by `_format_key()` (source: lines 59-78):
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
  The `page_size` is included because callers with the same dtype and tile but different page sizes would create inconsistent CB descriptors.
- **Env-gate escape hatch**: Set `BLAZE_DISABLE_TENSOR_CB_SHARE=1` to disable format-key sharing (useful for bisecting CB-related bugs). Controlled by `_tensor_cb_share_disabled()` (source: lines 142-148).
- **CB limit**: Blackhole has a hardware limit of 64 CB IDs (`MAX_CB_ID = 64` in `blaze/cb_engine.py`, line 65). Disjoint-cores sharing is essential for staying under this limit in large fused ops.

The `CBHandle` returned by `cb_from_tensor` has `backing_tensor` set to the tensor object, and `buffer_address()` returns the tensor's L1 address plus any byte offset.

### Strategy 2: Scratch CBs (`cb_scratch`)

**When to use**: For intermediate buffers that exist only within a fused kernel. The data is produced by one op and consumed by the next; it never needs to be read as a tensor.

```python
out_handle = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "out"),
    num_pages=num_tiles,
    core_ranges=cores,
    data_format=input_handle.data_format,
    tile=input_handle.tile_desc,
    page_size=input_handle.page_size,
)
```

`FusedProgram.cb_scratch()` (source: `blaze/fused_program.py`, line 1540) works as follows:

1. **Check scratch mapping**: If `scratch_mapping` contains an entry for this name, the CB is backed by a pre-allocated tensor at a specific byte offset. This is the `all_scratch_mapped` path (covered below).
2. **Allocate**: Calls `self.program.cb_scratch()` to allocate a new CB ID with the specified format.
3. **Track**: Adds to `_scratch_cbs` set. The `balanced` parameter (default `True`) controls temporal compaction eligibility:
   ```python
   self._scratch_cbs.add(cb_id)
   if not balanced:
       self._unbalanced_cbs.add(cb_id)
   ```

At build time, scratch CBs are materialized into a shared L1 arena by `_materialize_scratch_cbs()` (source: `blaze/cb_reconfig_builder.py`, lines 372-475). This function:
1. Collects all scratch CBs per phase
2. Assigns running byte offsets within each phase (16-byte aligned, per Blackhole/Wormhole L1 alignment)
3. Allocates a single `uint32` arena tensor on `full_device_grid` sized to `max(phase_totals)`
4. Replaces each scratch CB's descriptor with one pointing at the arena using `cb_descriptor_from_sharded_tensor()` with an `address_offset`

The arena is a zero-filled `[num_cores, shard_uint32]` tensor. It is deduplicated across per-device compile iterations via `fp._internal_tensor_dict` with a cache key of `("arena", num_cores, shard_uint32)`.

**L1 cost**: `num_pages * page_size` bytes per core in the `core_ranges` grid. However, scratch CBs are eligible for both spatial and temporal compaction (see below), which can dramatically reduce effective L1 usage.

### Strategy 3: Aliased CBs (`cb_alias`)

**When to use**: When one L1 buffer must be interpreted with two different tile formats or data types. A common case is reading the same data as both bfloat16 (for compute) and bfloat8_b (for lower-precision matmul).

```python
alias_handle = f.cb_alias(
    source_handle,
    tile=new_tile,
    page_size=new_page_size,
    dtype=new_dtype,
)
```

`FusedProgram.cb_alias()` (source: `blaze/fused_program.py`, lines 1370-1468) creates a second CB ID that reads the same L1 bytes as the source through a different format view. Implementation details:

1. **Appends a `CBFormatDescriptor`**: To every `CBDescriptor` hosting the source CB ID. This means one backing buffer with two CB IDs, each with different `(dtype, tile, page_size)`.
2. **Inherits tensor-backed status**: The alias is marked as tensor-backed if the source is, preventing compaction from overwriting the shared buffer.
3. **Inherits scratch membership**: If the source is a scratch CB, the alias is added to `_scratch_cbs`. The alias also inherits membership in `_unbalanced_cbs`.
4. **Disjoint-cores alias sharing**: Two `cb_alias` calls over distinct sources on disjoint grids with matching format fold to one alias CB ID.
5. **Shadow graph identity**: The alias inherits `_node_id` and `_port_name` from the source, so graph edges from the alias attribute back to the source's producer.

There is zero L1 memory overhead -- the alias and source share the same backing buffer.

### Strategy 4: Direct-Address Mode (`cb_from_shared_l1_views`)

**When to use**: When multiple consumers need to read different logical slices from the same backing tensor at overlapping addresses. Used for overlapped weight views in MLA/MoE patterns.

```python
handles = f.cb_from_shared_l1_views([view1, view2, view3])
```

`FusedProgram.cb_from_shared_l1_views()` (source: `blaze/fused_program.py`, lines 1257-1368) allocates a single CB ID for all views:

1. **Validates**: All views must share the same backing tensor, dtype, tile shape, and page size. View core ranges must be subsets of the backing tensor's shard grid.
2. **Allocates one descriptor**: Bound at byte offset 0 on the union of all view core ranges with `total_size=0` (full backing shard).
3. **Returns per-view handles**: Each handle carries its own `byte_offset` and `access_mode=CBAccessMode.DIRECT_ADDRESS`.
4. **Marks as direct-address**: The CB ID is added to `_direct_address_cbs`, excluding it from FIFO reuse caches. Consumers must read via `handle.buffer_address()` instead of CB front-pointer semantics.

The `CBAccessMode` enum (source: `blaze/cb_handle.py`, lines 13-17):

```python
class CBAccessMode(str, Enum):
    FIFO = "fifo"
    DIRECT_ADDRESS = "direct_address"
```

The `CBHandle.buffer_address()` method (source: `blaze/cb_handle.py`, lines 85-95) returns `backing_tensor.buffer_address() + byte_offset`, giving each view a distinct L1 read address:

```python
def buffer_address(self) -> int | None:
    if self.backing_tensor is not None:
        return self.backing_tensor.buffer_address() + self.byte_offset
    return None
```

The `require_fifo_handle()` guard (source: `blaze/cb_handle.py`, lines 23-31) is called by `cb_from_tensor()` and other FIFO-expecting APIs to reject direct-address handles with a clear error message:

```python
def require_fifo_handle(handle: "CBHandle", op_name: str) -> "CBHandle":
    if handle.is_direct_address:
        raise ValueError(
            f"{op_name} requires a FIFO CBHandle, but received a "
            "direct-address handle. Use an op that explicitly reads "
            "handle.buffer_address()."
        )
    return handle
```

### CB Strategy Decision Matrix

| Scenario | API | L1 Cost | Cross-Phase Safe |
|----------|-----|---------|------------------|
| External tensor (input/output/weight) | `cb_from_tensor()` | Zero | Yes |
| Intermediate between ops | `cb_scratch()` | `num_pages * page_size` | No |
| Sub-region of fused buffer | `cb_from_tensor_overlapped()` | Zero | Yes |
| Same L1 bytes, different format | `cb_alias()` | Zero | Inherits |
| Multi-consumer overlapping cores | `cb_from_shared_l1_views()` | Zero | Yes |
| All scratch pre-mapped | `all_scratch_mapped=True` | Zero | Yes |

---

## Multi-Phase CB Reconfiguration (`CbReconfig`)

When a fused kernel exceeds the 64 CB limit (or when different phases need different CB layouts), TT-Blaze uses multi-phase CB reconfiguration to reprogram CB addresses at runtime.

### How It Works

1. **Phase boundaries**: Ops call `f.reconfig()` (or emit `CbReconfig`) to mark phase boundaries. The `CbReconfig.emit()` method (source: `blaze/ops/cb_reconfig/op.py`):
   - Calls `f.reconfig()` which appends `self._op_index` to `_reconfig_boundaries` and clears `_format_key_to_cb` and `_view_tensor_cbs` (source: `blaze/fused_program.py`, lines 1876-1884):
     ```python
     def reconfig(self):
         self._reconfig_boundaries.append(self._op_index)
         self._format_key_to_cb.clear()
         self._view_tensor_cbs.clear()
     ```
   - Emits a placeholder CT arg `(f"{prefix}.cb_config_l1_addr", 0)` -- the real address is patched at build time
   - Records a `_reconfig_placeholder` for the build pass to find

2. **Phase tracking**: Every CB allocation records its phase via `_track_cb_allocation()`: `self._cb_phase[cb_id] = len(self._reconfig_boundaries)`. CBs in phase 0 are before the first boundary, phase 1 between the first and second, etc.

3. **Build pass**: `_build_multi_phase()` in `blaze/cb_reconfig_builder.py` (lines 245-353) runs during `_prepare_for_build()`:

   a. **Partition CBs into phases**: Groups CB IDs by `_cb_phase` value.

   b. **Validate no scratch crossing**: `_validate_no_scratch_crossing()` (lines 355-369) checks that no scratch CB's lifetime `[first_use, last_use]` spans a reconfig boundary. Scratch CBs change L1 addresses at boundaries, so cross-boundary references would read stale addresses. The actual code:

   ```python
   def _validate_no_scratch_crossing(fp):
       for cb_id in fp._cb_lifetimes:
           if cb_id in fp._tensor_backed_cbs:
               continue
           first, last = fp._cb_lifetimes[cb_id]
           for boundary_op_idx in fp._reconfig_boundaries:
               if first < boundary_op_idx <= last:
                   meta = fp._cb_metadata.get(cb_id)
                   name = f"{meta._node_id}.{meta._port_name}" if meta and meta._node_id else f"CB {cb_id}"
                   raise ValueError(
                       f"CB '{name}' is scratch-allocated but referenced across "
                       f"reconfig boundary at op {boundary_op_idx}. "
                       f"Use cb_from_tensor() for cross-phase data."
                   )
   ```

   Note: the error message includes the CB's node ID and port name from the shadow graph metadata (when available), providing clear identification of the offending CB. Tensor-backed CBs are explicitly skipped since their L1 address is fixed by the backing tensor and survives across phases.

   c. **Materialize scratch CBs**: `_materialize_scratch_cbs()` allocates the shared L1 arena, sized to `max(phase_totals)` across all phases:

   ```python
   for cb_ids in phase_scratch_cbs.values():
       offset = 0
       for cb_id in sorted(cb_ids):
           meta = fp._cb_metadata[cb_id]
           total_bytes = _cb_total_bytes(meta)
           aligned_offset = round_up(offset, L1_ALIGN)
           arena_offsets[cb_id] = aligned_offset
           offset = aligned_offset + total_bytes
       phase_total = round_up(offset, L1_ALIGN)
       arena_size = max(arena_size, phase_total)
   ```

   This sizing means Phase 1's scratch CBs can reuse the same L1 bytes as Phase 0's, as long as Phase 0 is done with them. This is the fundamental L1 memory benefit of multi-phase programs.

   d. **Assign CB IDs across phases**: Each phase gets a fresh context from `CircularBufferIdManager.create_context()`. Matching `(data_format, tile)` pairs reuse the same CB ID across phases, minimizing total CB ID consumption.

   e. **Build per-phase reconfig tensors**: For each phase, `record_cb_metadata()` captures the CB descriptor state, and `build_cb_reconfig_tensor()` creates a pre-allocated L1 tensor encoding the CB addresses, sizes, and page counts. Reconfig tensors are deduplicated across per-device compile iterations via content signature (lines 67-83).

   f. **Patch placeholder addresses**: `_patch_placeholder_addresses()` (lines 501-519) rewrites the `cb_config_l1_addr=0` placeholders with the actual tensor buffer addresses:

   ```python
   for phase_idx, node, risc_positions in fp._reconfig_placeholders:
       tensor = reconfig_tensors[phase_idx] if phase_idx < len(reconfig_tensors) else None
       if tensor is None:
           continue
       addr = tensor.buffer_address()
       for risc, idx in risc_positions.items():
           args = fp._get_risc_args(risc)
           name, _ = args[idx]
           args[idx] = (name, addr)
       for key in list(node.ct_values):
           if key.endswith(".cb_config_l1_addr"):
               node.ct_values[key] = addr
   ```

   g. **Insert Phase 0 init**: `_insert_phase0_reconfig()` (lines 478-498) adds a CBReconfig node at position 0 in the shadow graph -- the user does not need to emit Phase 0 explicitly.

   h. **Deduplicate descriptors**: `_dedup_cb_descriptors()` (lines 760-783) collapses overlapping-core duplicates while preserving disjoint-core descriptors.

### Validation: Empty Phases

Phases 1..N must each have at least one CB (lines 299-306). An empty phase means no ops were emitted between two consecutive reconfig boundaries. The kernel would read from L1 address 0, causing an indefinite hardware hang.

### Usage Pattern

```python
from blaze.ops.cb_reconfig.op import CbReconfig

# Phase 0: ops that use one set of CBs
Mcast.emit(f, ...)
RmsNorm.emit(f, ...)

# Phase boundary
CbReconfig.emit(f, prefix="reconfig_1")

# Phase 1: ops that use a different set of CBs
Matmul.emit(f, ...)
ResidualAdd.emit(f, ...)
```

---

## Scratch CB Compaction

Scratch compaction reduces CB ID consumption by merging non-conflicting scratch CBs onto shared slots. The compaction pass `_compact_scratch_cbs()` (source: `blaze/cb_reconfig_builder.py`, lines 104-191) runs automatically during `_prepare_for_build()`.

### Two Reuse Modes

1. **Spatial reuse**: Two scratch CBs with matching format `(data_format, tile_shape, page_size, num_pages)` on **disjoint core grids** share one CB ID. Each keeps its own descriptor. This is always safe.

2. **Temporal reuse**: Two scratch CBs with matching format on the **same core grid** but **non-overlapping lifetimes** share one CB ID. This requires:
   - Both CBs are `balanced` (push/pop balanced across cores)
   - Neither CB is tensor-backed (tensor-backed CBs have distinct address offsets that dedup would collapse)
   - No tensor-backed CB already occupies the target slot
   - The environment variable `BLAZE_DISABLE_TEMPORAL_REUSE` is not set

### Temporal Reuse Details

The `_slot_accepts()` function (source: lines 199-212) decides whether a new CB can fold onto an existing slot:

```python
def _slot_accepts(slot: _Slot, lifetime, core_ranges, *, allow_temporal: bool):
    decision = "spatial"
    for u_cr, u_lt in slot.users:
        if _are_disjoint(u_cr, core_ranges):
            continue
        if not allow_temporal:
            return None
        if _lifetimes_overlap(u_lt, lifetime):
            return None
        if not _cores_equal(u_cr, core_ranges):
            return None
        decision = "temporal"
    return decision
```

- Returns `"spatial"` if all existing users have disjoint core ranges
- Returns `"temporal"` if core ranges overlap but lifetimes don't (and temporal reuse is allowed). The `_cores_equal()` check (line 194) enforces strict equality -- partial overlap is rejected to prevent dedup from dropping coverage on a subset of cores.
- Returns `None` if the merge is unsafe

When temporal reuse occurs, `_insert_scratch_reset()` (lines 215-242) inserts a `CbScratchReset` node at the new user's first-use position in the shadow graph. This node resets the CB's read/write pointers so the new user starts from a clean state:

```python
def _insert_scratch_reset(fp, op_idx, cb_id, core_ranges):
    count = fp._op_counters.get("cb_scratch_reset", 0) + 1
    fp._op_counters["cb_scratch_reset"] = count
    prefix = f"cb_scratch_reset_{count}"

    risc_indices = {risc: len(fp._get_risc_args(risc)) for risc in ("ncrisc", "brisc", "trisc")}
    fp.program.unified_ct_args([(f"{prefix}.cb_id", cb_id)], prefix=False)
    for risc, idx in risc_indices.items():
        fp._cb_arg_positions.append((risc, idx, cb_id))

    spec = get_op_spec("cb_scratch_reset")
    node = OpNode(
        id=f"cb_scratch_reset_{count}",
        spec=spec,
        grid=core_ranges,
        kwargs={"ct_prefix": prefix},
        ct_values={f"{prefix}.cb_id": cb_id},
    )
    fp._shadow_graph.nodes.insert(op_idx, node)
```

### Eligibility

Not all scratch CBs are eligible for compaction. The function skips:
- CBs spanning multiple descriptors (already shared across disjoint grids)
- CBs with no lifetime data or metadata
- Unbalanced CBs (marked via `balanced=False` in `cb_scratch()`; line 149: `is_unbalanced = cb_id in fp._unbalanced_cbs`)
- Tensor-backed scratch CBs for temporal reuse (distinct `address_offset` descriptors would collide; line 150: `is_tensor_backed = cb_id in fp._tensor_backed_cbs`)

### Bisection

Set `BLAZE_DISABLE_TEMPORAL_REUSE=1` to disable temporal reuse while keeping spatial reuse active. This is useful when debugging issues that might stem from incorrect CB pointer reset behavior.

---

## The `all_scratch_mapped` Mode

The `all_scratch_mapped` flag (passed to `BlazeCompiler.compile()` or `FusedProgram.__init__()`) enforces that every `cb_scratch()` call must have a corresponding entry in `scratch_mapping`. This is used in production pipelines where L1 layout is pre-planned by a higher-level allocator.

### How scratch_mapping Works

```python
compiler.compile(
    ...,
    scratch_tensors={"weights": pre_allocated_tensor},
    scratch_mapping={
        "mcast___dst": {"tensor_name": "weights", "offset_address": 0},
        "rmsnorm___scratch": {"tensor_name": "weights", "offset_address": 4096},
    },
    all_scratch_mapped=True,
)
```

When `cb_scratch()` is called with a name that has a `scratch_mapping` entry, `_cb_scratch_from_mapping()` (source: `blaze/fused_program.py`, lines 1481-1537):

1. Looks up the mapping entry: `{"tensor_name": "weights", "offset_address": 0}`
2. Finds the backing tensor in `scratch_tensors`
3. Calls `f.cb_from_tensor_overlapped()` with the backing tensor and offset
4. Returns a tensor-backed `CBHandle` instead of a scratch CB

When `all_scratch_mapped=True` and a scratch name is NOT in the mapping, `_cb_scratch_from_mapping()` raises a `ValueError` (lines 1505-1508):

```python
if all_scratch_mapped:
    raise ValueError(
        f"cb_scratch('{name}') is not mapped to a backing tensor while "
        f"all_scratch_mapped=True. "
        f"Expected scratch_mapping key {mapped_key!r}."
    )
```

This catches layout mismatches early. This mode eliminates all scratch CBs, giving the developer full control over L1 layout. It is the preferred mode for production models where L1 budget is tight and predictable layout is required.

The `select_prefix` parameter allows scoping scratch_mapping keys to a specific pipeline stage:

```python
compiler.compile(
    ...,
    select_prefix="stage0.",
    scratch_mapping={
        "stage0.mcast___dst": {"tensor_name": "arena", "offset_address": 0},
    },
)
```

`_cb_scratch_from_mapping()` prepends `select_prefix` to the scratch name before looking up the mapping (line 1485).

### Duplicate Detection

The method tracks seen scratch names in `_cb_scratch_names_seen` (line 1497). If a name is encountered twice, it raises a `ValueError`. This catches bugs where two ops use the same scratch name but expect different backing regions.

---

## OverlappedView for Fused Weight Buffers

`OverlappedView` (defined at line 220 in `blaze/fused_program.py`) represents a logical sub-region of a fused L1 buffer. Multiple views can share one physical backing tensor, each with different `shard_shape`, `core_ranges`, `dtype`, `tile`, `byte_offset`, and `total_size`.

```python
@dataclass(frozen=True)
class OverlappedView:
    tensor: ttnn.Tensor        # shared backing tensor
    shard_shape: tuple[int, int]  # per-core shard shape of THIS view
    core_ranges: ttnn.CoreRangeSet
    dtype: ttnn.DataType
    tile: object
    byte_offset: int           # offset within each core's fused shard
    total_size: int            # bytes per shard for this view
```

`cb_from_view()` (source: lines 1164-1255) handles three sharing scenarios:

1. **Same-tensor, disjoint grids**: Reuses the existing CB ID from a previous view of the same backing tensor. Appends a per-grid `CBDescriptor` with the correct `address_offset`.
2. **Different tensor, disjoint grids, matching format**: Falls through to format-key reuse via `_try_reuse_cb_id()`.
3. **Overlapping grids or incompatible format**: Allocates a fresh CB ID.

---

## L1 Profiling with `BLAZE_L1_PROFILE`

TT-Blaze includes built-in L1 profiling triggered by the environment variable `BLAZE_L1_PROFILE`.

### Activation

```bash
export BLAZE_L1_PROFILE=1
```

When set, `CompiledProgram.run()` and `MeshCompiledProgram.run()` call `print_cb_stats(self)` before executing:

```python
def run(self):
    if os.environ.get("BLAZE_L1_PROFILE"):
        from blaze.l1_profile import print_cb_stats
        print_cb_stats(self)
    return _run_program(...)
```

### Output

`print_cb_stats()` reports for each CB descriptor:
- **CB type**: `tensor_backed`, `aliased`, or `scratch`
- **L1 address**: Reported for tensor-backed CBs; scratch CB addresses are Metal-managed at dispatch time
- **Buffer indices**: All `buffer_index` values from format descriptors (multiple for aliased CBs)
- **Data format**: The dtype of the CB
- **Page size**: Bytes per tile
- **Tile shape**: (height, width)
- **Core ranges**: Which cores the CB is bound to

Example output:
```
cb_stats: 5 CB descriptor(s)
  cb[0] type=tensor_backed buffer_index=0 total_size_B=32768 l1_addr=0x11a000
    core_ranges={[(x=0,y=0) - (x=0,y=0)]}
  cb[1] type=tensor_backed buffer_index=1 total_size_B=32768 l1_addr=0x122000
    core_ranges={[(x=0,y=0) - (x=0,y=0)]}
  cb[2] type=scratch buffer_index=2 total_size_B=32768 l1_addr=metal-managed
    core_ranges={[(x=0,y=0) - (x=0,y=0)]}
  cb[3] type=aliased buffer_index=3,4 total_size_B=8192 l1_addr=0x14340
    [dtype=DataType.bfloat16 page_size_B=2048 tile=32x32;
     dtype=DataType.bfloat8_b page_size_B=1088 tile=32x32]
    core_ranges={[(x=0,y=0) - (x=7,y=7)]}
```

### Log File

Output is printed to stdout AND written to a log file under `$TT_METAL_CACHE/logs/` (or `~/.cache/tt-metal-logs/` if `TT_METAL_CACHE` is unset). The file is named `blaze_l1_profiler_log_<timestamp>.log`.

The `_lp()` function (source: `blaze/l1_profile.py`, lines 106-117) writes to both stdout and the log file:

```python
def _lp(*args, sep=" ", end="\n", flush=True, file=None):
    if file is None:
        file = sys.stdout
    text = sep.join(str(a) for a in args) + end
    try:
        fp = _ensure_l1_profile_log_fp()
        fp.write(text)
        fp.flush()
    except OSError:
        pass
    builtins.print(*args, sep=sep, end=end, flush=flush, file=file)
```

The log directory resolution (source: lines 70-74):

```python
def _l1_profile_out_dir() -> Path:
    cache = os.environ.get("TT_METAL_CACHE", "").strip()
    if cache:
        return Path(cache).expanduser().resolve() / "logs"
    return Path.home() / ".cache" / "tt-metal-logs"
```

### CB Address Capture

For Blaze programs, CB addresses come from Metal's L1 allocator at runtime. Tensor-backed CBs get `buffer->address() + address_offset` from the backing tensor buffer. Scratch CBs are assigned addresses by Metal during program dispatch, captured via the `allocated_address` field on `CBDescriptor`.

For blitz (CbReconfig-based) programs, CB addresses come from the CbReconfig tensor -- a pre-allocated `[n_cores, 264]` UINT32 L1 tensor. The profiler reads this tensor to CPU before dispatch and decodes each core's row. The captured state is a composite of all phases: CbReconfig only reprograms CB slots that change between phases, so a CB configured in phase 0 and never touched again retains its phase 0 address in the final tensor.

### Kernel Stats

`print_kernel_stats()` reports per-kernel:
- **Role**: `trisc`, `brisc`, or `ncrisc`
- **Source stem**: The kernel source file
- **Core ranges**: Which cores execute the kernel

### CB Blueprint and Overlap Detection

The `_build_overlap_groups()` function detects L1 address overlaps between CBs using union-find to group connected overlapping CBs. This is essential for debugging CbReconfig: two CBs at the same L1 address are legitimate if they belong to different phases (temporal CbReconfig overlap), but indicate a bug if they are in the same phase.

---

## CBEngine: Graph-Based CB Assignment

The `CBEngine` (source: `blaze/cb_engine.py`) is the graph-path counterpart to `FusedProgram`'s manual CB allocation. It walks a `BlazeGraph` in topological order, assigning sequential CB IDs to all data paths.

### Assignment Categories

| Key pattern | Category | `is_tensor_backed` | Description |
|-------------|----------|---------------------|-------------|
| `ext_in_{node}_{port}` | External input | True | Tensor-backed input port |
| `internal_{node}_{port}` | Internal | False | Internal CB from `Internal()` descriptor |
| `intermed_{node}_{port}` | Intermediate | False | Inter-op edge (producer output -> consumer input) |
| `ext_out_{node}_{port}` | External output | True | Terminal output port |

### Fan-out Handling

When one output feeds multiple consumers, the same CB ID is shared across all edges. The `core_ranges` is the union of the producer grid and all consumer grids:

```python
all_grids = [node.grid] + [e.consumer.grid for e in port_edges]
core_ranges = _union_grids(all_grids)
```

### Compact Mode (Interval Coloring)

The engine supports `compact=True` mode that applies interval-graph coloring to minimize CB ID usage:

```python
def compact_cb_ids(self, assignments):
    sorted_cbs = sorted(assignments.values(), key=lambda a: a.first_use)
    color_end: dict[int, int] = {}  # color -> last_use
    for cb in sorted_cbs:
        color = 0
        while color in color_end and color_end[color] >= cb.first_use:
            color += 1
        cb.cb_id = color
        color_end[color] = cb.last_use
    return assignments
```

This is interval-graph coloring: two CBs with non-overlapping lifetimes share the same ID.

### CB Limit Warning

The engine warns when approaching the hardware limit (lines 377-382):

```python
if next_cb_id > self.max_cb_id - 8:
    warnings.warn(
        f"CB usage is high: {next_cb_id}/{self.max_cb_id} CBs assigned. "
        f"Approaching Blackhole hardware limit.",
        stacklevel=2,
    )
```

### Export to CircularBufferIdManager

`to_cb_id_manager()` (lines 389-420) exports engine assignments to a `CircularBufferIdManager` for interop. This is used when downstream code (e.g., MoE ops) needs to allocate additional CBs without ID collisions.

---

## CB Descriptor Deduplication

When multiple passes (compaction, multi-phase remap, disjoint-cores sharing) produce duplicate CB descriptors at the same CB ID, `_dedup_cb_descriptors()` (source: `blaze/cb_reconfig_builder.py`, lines 760-783) reconciles them:

1. **Group by buffer index**: Build a map from CB ID to list of descriptors.
2. **Competition**: Sort by `(total_size, num_cores)` descending. The largest descriptor wins overlapping cores.
3. **Residuals**: When a larger descriptor overlaps only part of a wider descriptor, the uncovered portion is split into a residual descriptor. This keeps every core statically bound.
4. **Aliased descriptors**: A descriptor with `format_descriptors` at multiple buffer indexes may lose some indexes (pruned) while retaining others. The descriptor is dropped only when every format descriptor has been pruned.

The result is a set of descriptors where:
- Same CB ID on overlapping cores: collapsed to one (largest total_size wins)
- Same CB ID on disjoint cores: preserved as separate descriptors (spatial reuse)
- Aliased descriptors: pruned per-index, surviving only where they're the largest

---

## Performance Optimization Patterns

### Pattern 1: Minimize Scratch CBs

Every scratch CB consumes L1 space in the materialized arena. Prefer:
- **In-place operations**: Write output to the same CB as input when semantically correct.
- **Wire output**: Use `output_tensor` parameter to make the final op's output tensor-backed instead of scratch.
- **cb_alias**: When you need to reinterpret the same buffer with a different format, use `cb_alias` instead of allocating a separate scratch CB.

### Pattern 2: Control Subblocking

The TRISC destination register file has a limited number of registers (typically 8). Operations that process more tiles than available registers must use subblocking:

```python
from blaze.utils import compute_subblock_w

subblock_w = compute_subblock_w(
    num_tiles,
    fp32_dest_acc_en=fp32_dest_acc_en,
    dst_full_sync_en=dst_full_sync_en,
)
```

The kernel loops over subblocks, processing `subblock_w` tiles per iteration with `tile_regs_acquire/commit/wait/release` synchronization.

### Pattern 3: Guard init() with is_active

The NCRISC `init()` method must be guarded by `is_active` to avoid calling `setup_sharded_buffer()` on cores that don't participate:

```cpp
void init() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (!core_cta::is_active) return;
    if constexpr (reader_cta::input_is_tensor_backed) {
        unified_kernels::setup_sharded_buffer(
            reader_cta::input, reader_cta::num_tiles);
    }
#endif
}
```

Without the guard, NCRISC would try to setup sharded buffers for CBs that don't exist on inactive cores, causing L1 corruption.

### Pattern 4: Use `tensor_backed` Flags

When a CB might be either tensor-backed (from `cb_from_tensor`) or scratch-allocated (from an upstream op's scratch output), the kernel uses a compile-time flag to conditionally call `setup_sharded_buffer()`:

```python
f.ncrisc_ct_args([
    (f"{prefix}.input_is_tensor_backed",
     int(isinstance(input, ttnn.Tensor))),
])
```

This pattern is used extensively in `EltwiseMul` and other ops that accept both CBHandle and Tensor inputs.

### Pattern 5: Wait/Pop Control

Operations downstream of a multicast or fan-out may need to wait on a different CB than they read from (due to aliasing). The `in0_wait` / `in0_wait_tiles` pattern in EltwiseMul separates the wait CB from the read CB:

```python
i0w = in0_wait_cb if in0_wait_cb is not None else in0_handle
i0wt = in0_wait_tiles if in0_wait_tiles is not None else num_tiles
```

The kernel then uses:
```cpp
cb_wait_front(compute_cta::in0_wait, compute_cta::in0_wait_tiles);
```

while reading from `compute_cta::in0`.

### Pattern 6: Balanced vs Unbalanced Scratch

Set `balanced=False` when allocating scratch CBs for mcast destinations or other patterns where push/pop is not symmetric across cores:

```python
dst_cb = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "dst"),
    num_pages=num_tiles,
    core_ranges=all_cores,
    data_format=data_format,
    tile=tile_desc,
    page_size=page_size,
    balanced=False,  # mcast pushes on all receivers, consumer pops on subset
)
```

Unbalanced CBs are excluded from temporal compaction (`_compact_scratch_cbs` checks `is_unbalanced` at line 149) to prevent FIFO state from leaking across reuse boundaries.

### Pattern 7: Empty Processor Bodies Enable Pipelining

A micro-op kernel has three RISC processors executing concurrently: BRISC (writer), NCRISC (reader), and TRISC (compute). When one processor has no work for a particular op, its `operator()()` body should be empty or contain only `(void)0;`. This allows RISC-level pipelining: while TRISC computes on op A's data, NCRISC can already be setting up CB buffers for op B.

The `init_is_empty` and `teardown_is_empty` fields from `ParsedKernel` (populated by `_method_body_is_empty()` in `cpp_parser.py`, lines 177-199) enable codegen to skip init/teardown calls entirely when the methods contain only whitespace and comments.

### Pattern 8: Visualization with BLAZE_EXPORT

Set `BLAZE_EXPORT=1` to auto-export a JSON visualization of the compiled graph:

```bash
export BLAZE_EXPORT=1
export BLAZE_EXPORT_PATH=generated/viz  # optional
```

The auto-export runs after `compile()` and produces a `<name>_viz.json` file that can be loaded in the Blaze visualizer web tool.

---

## Minimizing CB Count: 64 CB ID Limit Strategies

The Blackhole hardware supports a maximum of 64 CB IDs. Large fused ops can hit this limit. Strategies ordered from automatic to manual:

1. **Automatic Scratch Compaction** (default): `_compact_scratch_cbs()` runs automatically during `_prepare_for_build()`. It merges non-conflicting scratch CBs via spatial and temporal reuse.

2. **Disjoint-Core Format Sharing** (default): `_try_reuse_cb_id()` automatically shares CB IDs between tensor-backed CBs with matching formats on disjoint core grids.

3. **CbReconfig Multi-Phase** (explicit): For ops that genuinely need more than 64 unique CB configurations, use `f.reconfig()` to create phase boundaries. Each phase gets its own fresh set of CB IDs.

4. **`cb_alias()`** (explicit): When two ops need to read the same L1 data with different tile formats, `cb_alias()` creates a second CB ID without duplicating L1 memory.

5. **Internal Ports** (semi-explicit): The `Internal()` port descriptor declares a scratch buffer at the OpSpec level. Internal CBs participate in the engine's lifetime-based compaction.

6. **Monitoring**: The `CBEngine` warns when approaching the limit and raises a hard error when exceeded.

---

## CB Lifetime Tracking

Every CB allocation is tracked with `[first_use, last_use]` op indices. Lifetimes are updated on every CB allocation, CT arg tracking, and `f.output()` call. They drive:

1. **Compaction**: Two scratch CBs can share an ID only if their lifetimes are disjoint
2. **Scratch-boundary validation**: Scratch CBs cannot span reconfig boundaries
3. **Phase partitioning**: Each CB's phase is recorded in `_cb_phase[cb_id]`

---

## Practical L1 Budget Example

Consider a fused op that chains RMSNorm, ScaledAdd, and Matmul on a 4x8 grid (32 cores) with bfloat16 data and (1, 32) tiles:

| CB | Type | num_pages | page_size | Per-Core | Total (32 cores) |
|----|------|-----------|-----------|----------|-------------------|
| act input | tensor-backed | 16 | 64 B | 1024 B | 32 KB |
| gamma input | tensor-backed | 16 | 64 B | 1024 B | 32 KB |
| norm intermediate | scratch | 16 | 64 B | 1024 B | 32 KB |
| norm output | scratch | 16 | 64 B | 1024 B | 32 KB |
| bias input | tensor-backed | 16 | 64 B | 1024 B | 32 KB |
| scaled_input | scratch | 16 | 64 B | 1024 B | 32 KB |
| sa_output | scratch | 16 | 64 B | 1024 B | 32 KB |
| weight input | tensor-backed | 16 | 64 B | 1024 B | 32 KB |
| matmul output | tensor-backed | 16 | 64 B | 1024 B | 32 KB |

**Without compaction**: 9 CB IDs, 4 scratch CBs consuming 4 KB per core = 128 KB total scratch.

**With temporal compaction**: If norm_intermediate and scaled_input have non-overlapping lifetimes, they can share a CB slot. Similarly for norm_output and sa_output. This reduces scratch CBs from 4 to 2, saving 64 KB total.

**With spatial sharing**: If the gamma and bias tensors use the same format on disjoint core grids, they can share a CB ID, reducing total CB count from 9 to 8.

The `BLAZE_L1_PROFILE=1` output confirms the actual allocations and addresses, letting you validate that compaction achieved the expected savings.

### Tile Page Size Calculation

The fundamental formula for tile page size (source: `blaze/cb_engine.py`, lines 73-79):

```python
def _tile_page_size(dtype, tile_shape) -> int:
    dt = dtype or DEFAULT_DTYPE
    ts = tile_shape or DEFAULT_TILE_SHAPE   # (32, 32)
    return DTYPE_BYTES[dt] * ts[0] * ts[1]
```

Examples:
- `bfloat16` with `(32, 32)` tile: `2 * 32 * 32 = 2048 bytes`
- `bfloat16` with `(1, 32)` tile: `2 * 1 * 32 = 64 bytes`
- `float32` with `(32, 32)` tile: `4 * 32 * 32 = 4096 bytes`

The total L1 cost of a scratch CB is `num_pages * page_size`. For a 16-tile scratch CB in bfloat16 with standard tiles: `16 * 2048 = 32,768 bytes = 32 KB` per core.

---

## Summary of Key Environment Variables

| Variable | Effect |
|----------|--------|
| `BLAZE_L1_PROFILE=1` | Print CB stats on every `run()` |
| `BLAZE_EXPORT=1` | Auto-export graph visualization JSON |
| `BLAZE_EXPORT_PATH=<dir>` | Override export directory (default: `generated/viz`) |
| `BLAZE_DISABLE_TENSOR_CB_SHARE=1` | Disable disjoint-cores CB sharing for tensor-backed CBs |
| `BLAZE_DISABLE_TEMPORAL_REUSE=1` | Disable temporal scratch CB compaction |

---

## Summary: L1 Management API Reference

| API | L1 Cost | CB ID Cost | Cross-Phase Safe | Compaction Eligible |
|-----|---------|------------|-------------------|---------------------|
| `cb_from_tensor()` | Zero | 1 (shared if disjoint) | Yes | No (tensor-backed) |
| `cb_scratch()` | `N * page_size` | 1 (compactable) | No | Yes (spatial + temporal) |
| `cb_scratch(balanced=False)` | `N * page_size` | 1 | No | Yes (spatial only) |
| `cb_alias()` | Zero | 1 (shared if disjoint) | Inherits source | Inherits source |
| `cb_from_view()` | Zero | 1 (shared per tensor) | Yes | No (tensor-backed) |
| `cb_from_shared_l1_views()` | Zero | 1 | Yes | No (direct-address) |
| `cb_from_tensor_overlapped()` | Zero | 1 | Yes | No (tensor-backed) |

---

## Compilation Pipeline Summary

The full compilation pipeline, from `BlazeCompiler.compile()` to execution:

```
compile()
  |
  +-- CBEngine().assign(graph)        # CB ID assignment (graph path)
  +-- SemEngine().assign(graph)       # Semaphore assignment
  |
  +-- For each device:
  |     +-- _compile_single()
  |           |
  |           +-- FusedOp path:       # Single-node graph with FusedOpConfig
  |           |     +-- FusedProgram() construction
  |           |     +-- config.compose_fn(f, tensors, output, args)
  |           |     |     +-- Op.emit() calls: cb_from_tensor, cb_scratch,
  |           |     |     |   unified_ct_args, per_core_unified_ct_args,
  |           |     |     |   output()
  |           |     +-- _prepare_for_build()
  |           |     |     +-- _compact_scratch_cbs()     # Spatial + temporal
  |           |     |     +-- _build_multi_phase()       # If reconfig boundaries
  |           |     |           +-- _validate_no_scratch_crossing()
  |           |     |           +-- _materialize_scratch_cbs()
  |           |     |           +-- Per-phase CB ID assignment
  |           |     |           +-- Per-phase reconfig tensor allocation
  |           |     |           +-- _remap_cb_ids()
  |           |     |           +-- _patch_placeholder_addresses()
  |           |     |           +-- _dedup_cb_descriptors()
  |           |     +-- build()
  |           |           +-- Kernel codegen (if kernel=None)
  |           |           +-- ProgramDescriptor construction
  |           |
  |           +-- Engine path:        # Multi-node graph
  |                 +-- CB descriptors from engine assignments
  |                 +-- CT args from CTArgEngine
  |                 +-- UnifiedKernelDescriptor
  |                 +-- ProgramDescriptor
  |
  +-- MeshProgramDescriptor assembly
  +-- MeshCompiledProgram
  |
  v
run()
  +-- BLAZE_L1_PROFILE? print_cb_stats()
  +-- ttnn.generic_op(io_tensors, descriptor)
  +-- Deallocate cleanup tensors
```

This pipeline ensures that CB IDs are globally unique within each phase, scratch CBs are packed into minimal L1 space, reconfig tensors enable phase switching at kernel runtime, and the final program descriptor is ready for Metal's dispatch infrastructure.
