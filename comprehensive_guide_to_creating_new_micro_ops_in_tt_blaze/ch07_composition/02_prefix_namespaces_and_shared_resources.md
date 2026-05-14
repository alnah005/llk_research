# 7.2 Prefix Namespaces and Shared Resources

## The Namespace Problem

A fused kernel is a single compilation unit. Every compile-time argument, every circular buffer, and every semaphore lives in a flat namespace -- the underlying `BlazeProgram` does not have scoping. When `SharedExpert` chains `KNMatmul`, `GatedReduce`, `Mcast`, and `DownProj`, and `DownProj` itself chains `matmul`, `Mcast`, `ResidualAdd`, and `Gather`, the total count of named compile-time arguments can reach into the hundreds. If two micro-ops both declare a CT arg called `"out"`, or two stages both allocate a scratch CB called `"dst"`, the kernel will not compile or -- worse -- will compile with silent corruption.

TT-Blaze solves this with two delimiter-based namespace mechanisms: `child_prefix()` for compile-time arguments and semaphores, and `cb_name()` for circular buffer names. Together with the `FusedProgram`'s duplicate-name checks, these mechanisms catch composition bugs at build time rather than at runtime.

## child_prefix() and the __ Delimiter

The `child_prefix()` classmethod is defined on `BlazeOp` (in `blaze/blaze_op.py`):

```python
class BlazeOp:
    CHILD_PREFIX_DELIMITER: str = "__"

    @classmethod
    def child_prefix(cls, prefix: str, name: str) -> str:
        """Build a nested-op prefix under this op's prefix scope."""
        return f"{prefix}{cls.CHILD_PREFIX_DELIMITER}{name}" if prefix else name
```

When a FusedOp calls a child micro-op's `emit()`, it passes a scoped prefix:

```python
# From SharedExpert.emit()
gu = KNMatmul.emit(
    f,
    activation,
    gate_up_weights,
    prefix=BlazeOp.child_prefix(prefix, "gu"),  # "shared_expert__gu"
    ...
)
```

If `prefix` is `"shared_expert"`, then `child_prefix(prefix, "gu")` produces `"shared_expert__gu"`. Inside `KNMatmul.emit()`, CT args are registered with this scoped prefix:

```python
# From KNMatmul.emit()
f.unified_ct_args(
    [
        (f"{prefix}.act", act_handle),
        (f"{prefix}.act_total_tiles", K_gate_tiles),
    ]
)
```

With `prefix = "shared_expert__gu"`, these become:

- `"shared_expert__gu.act"`
- `"shared_expert__gu.act_total_tiles"`

The subsequent stages get their own prefixes:

- `BlazeOp.child_prefix(prefix, "gated_reduce")` -> `"shared_expert__gated_reduce"`
- `BlazeOp.child_prefix(prefix, "mcast")` -> `"shared_expert__mcast"`
- `BlazeOp.child_prefix(prefix, "down_proj")` -> `"shared_expert__down_proj"`

And when `DownProj.emit()` calls its own children, the nesting goes deeper:

```python
# From DownProj.emit()
mm = DownProj.matmul(
    f, act_handle, weights_tensor,
    prefix=BlazeOp.child_prefix(prefix, "matmul"),
    # prefix = "shared_expert__down_proj"
    # result = "shared_expert__down_proj__matmul"
    cores=matmul_cores,
)
```

The full prefix tree for `SharedExpert` looks like:

```
shared_expert
  +-- shared_expert__gu                              (KNMatmul)
  +-- shared_expert__gated_reduce                    (GatedReduce)
  +-- shared_expert__mcast                           (Mcast)
  +-- shared_expert__down_proj                       (DownProj)
        +-- shared_expert__down_proj__matmul          (DownProj.matmul)
        +-- shared_expert__down_proj__mcast2          (Mcast for bias)
        +-- shared_expert__down_proj__residual_add    (ResidualAdd)
        +-- shared_expert__down_proj__gather          (Gather)
```

Every CT arg, flag, and semaphore name inherits this prefix chain. Two `Mcast` instances can both internally use `f"{prefix}.src"` because their prefixes are different (`"shared_expert__mcast"` vs. `"shared_expert__down_proj__mcast2"`).

### Why __ Instead of .

The delimiter is `__` (double underscore), not `.` (dot). This matters because dots appear within a single micro-op's CT arg names (e.g., `"matmul.in0"`, `"ag.dest_noc_x"`). The `__` delimiter separates hierarchy levels, while `.` separates the op-internal argument name from its namespace. Reading a fully-qualified CT arg name:

```
shared_expert__down_proj__matmul.in0
|______________|_________|______|__|
   level 1       level 2  level3  arg name
   (SharedExpert)(DownProj)(matmul method)
```

The double underscore is also easy to split programmatically for debugging and visualization tools. The `_prefix_matches` function in `blaze/graph.py` handles both delimiters when performing prefix-based graph queries:

```python
def _prefix_matches(node_prefix: str, target: str) -> bool:
    """True if node_prefix equals target or is nested under it."""
    return (
        node_prefix == target
        or node_prefix.startswith(target + "__")
        or node_prefix.startswith(target + ".")
    )
```

### The Empty Prefix Edge Case

When `prefix` is the empty string, `child_prefix()` returns just the name without a delimiter:

```python
return f"{prefix}{cls.CHILD_PREFIX_DELIMITER}{name}" if prefix else name
```

This handles the top-level case. When `DownProj.compose()` is called directly (not from another FusedOp), it passes empty-string or bare prefixes:

```python
# From DownProj.compose()
act = Mcast.emit(f, tensors["input"], prefix="mcast")
cls.emit(
    f, act, tensors["weights"], tensors["add_input"], output,
    mcast_sender_sem=0,
    core_coords=user_args.get("core_coords"),
)
```

And inside `DownProj.emit()`, when `prefix=""`:

```python
mm = DownProj.matmul(
    f, act_handle, weights_tensor,
    prefix=BlazeOp.child_prefix(prefix, "matmul"),
    # child_prefix("", "matmul") -> "matmul"
    cores=matmul_cores,
)
```

The result is clean: `"matmul.in0"` instead of `"__matmul.in0"`.

## cb_name() and the ___ Delimiter

Circular buffer names use a different delimiter to avoid conflicts with CT arg names:

```python
class BlazeOp:
    CB_NAME_DELIMITER: str = "___"

    @classmethod
    def cb_name(cls, prefix: str, name: str) -> str:
        """Build a CB name under this op's prefix scope."""
        return f"{prefix}{cls.CB_NAME_DELIMITER}{name}"
```

Three underscores separate the prefix from the CB name. This appears throughout micro-op `emit()` functions:

```python
# From GatedReduce.emit()
group1 = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "group1"),
    # e.g., "shared_expert__gated_reduce___group1"
    num_pages=group_num_pages,
    core_ranges=gather_dst_ranges,
    ...
)
```

```python
# From DownProj.matmul()
out = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "out"),
    # e.g., "shared_expert__down_proj__matmul___out"
    num_pages=out_w_per_core,
    core_ranges=cores,
    ...
)
```

```python
# From Mcast.emit()
dst = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "dst"),
    # e.g., "shared_expert__mcast___dst"
    num_pages=_dst_num_pages,
    core_ranges=f.all_cores,
    ...
)
```

The triple-underscore delimiter ensures that CB names cannot collide with CT arg names (which use `.`) or with the hierarchy delimiter (which uses `__`). This three-level disambiguation -- `__` for hierarchy, `.` for arg names, `___` for CB names -- eliminates an entire class of composition bugs.

### CB Names and scratch_mapping

The `cb_name()` output is the key used in `FusedProgram.scratch_mapping`, a dictionary that maps CB names to pre-allocated tensor-backed CB descriptors. When the compiler pre-plans memory layout, it populates `scratch_mapping` with entries like:

```python
scratch_mapping = {
    "shared_expert__gated_reduce___group1": { ... },
    "shared_expert__gated_reduce___group2": { ... },
    "shared_expert__mcast___dst": { ... },
    ...
}
```

Inside `cb_scratch()`, the name is used to look up a pre-allocated tensor:

```python
def cb_scratch(self, name, *, num_pages, core_ranges, data_format, tile, page_size, balanced=True):
    mapped_handle = self._cb_scratch_from_mapping(
        name=name,
        num_pages=num_pages,
        ...
    )
    if mapped_handle is not None:
        mapped_handle.scratch_name = name
        ...
        return mapped_handle
    # Fall back to dynamic allocation
    cb_id = self.program.cb_scratch(name=name, ...)
```

The unique, hierarchy-scoped CB names make this mapping deterministic and collision-free. The `_cb_scratch_from_mapping()` method also enforces uniqueness -- it raises `ValueError` if the same scratch name is encountered twice:

```python
if mapped_key in cb_scratch_names_seen:
    raise ValueError(f"Duplicate scratch name encountered: {mapped_key!r}")
```

## CT Arg Scoping: Per-Core Flags and Per-RISC Args

Within each micro-op's `emit()`, CT args are registered using the prefix as a namespace. Several scoping levels are available:

**Unified CT args** (all RISC processors see the same value):

```python
f.unified_ct_args([
    (f"{prefix}.act", act_handle),
    (f"{prefix}.act_total_tiles", K_gate_tiles),
])
```

**Per-RISC CT args** (only one RISC processor sees the value):

```python
f.ncrisc_ct_args([(f"{prefix}.act_is_tensor_backed", int(act_is_tensor_backed))])
f.trisc_ct_args([(f"{prefix}.weights", gu_weights_cb), ...])
```

**Per-core flags** (different cores get different values):

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", ab.compute_grid),
    f.flag(f"{prefix}.pop_act", ab.compute_grid, pop_act),
])
```

The common CT arg registration methods are:

| Method | Scope |
|--------|-------|
| `f.per_core_unified_ct_args()` | All RISC processors (NCRISC, BRISC, TRISC) |
| `f.per_core_ncrisc_ct_args()` | NCRISC only |
| `f.per_core_trisc_ct_args()` | TRISC only |
| `f.per_core_brisc_ct_args()` | BRISC only |

The per-core methods enforce uniqueness via `_per_core_ct_args_check()`:

```python
def _per_core_ct_args_check(self, args):
    for name, mapping in args:
        if name in self._ct_arg_names_seen:
            raise ValueError(f"Duplicate per-core CT arg: {name!r}")
        self._ct_arg_names_seen.add(name)
```

This is the first line of defense against prefix collisions. If two micro-ops accidentally use the same prefix, the duplicate check raises immediately at composition time, not at kernel compile time or -- worse -- at runtime.

The non-per-core methods (`f.unified_ct_args()`, `f.ncrisc_ct_args()`, etc.) do not enforce uniqueness at the Python level -- they delegate directly to `BlazeProgram`, which appends to flat lists. Uniqueness there relies on the prefix convention, but collisions would manifest as kernel compilation errors rather than Python exceptions.

## The flag() Helper

The `flag()` method on `FusedProgram` (line 1595 in `fused_program.py`) is a convenience for declaring per-core boolean CT args:

```python
def flag(self, name, cores, enabled=True):
    """Build a per-core CT arg tuple for a boolean flag.

    Returns a (name, mapping) tuple suitable for passing to
    per_core_unified_ct_args() and variants.
    The flag is 1 on the given cores when enabled, 0 when disabled.
    Cores not in the range always get 0 (via other_value default).
    """
    return (name, {cores: int(enabled)})
```

This is used extensively by micro-ops to declare activation flags:

```python
# From KNMatmul.emit()
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", ab.compute_grid),
    f.flag(f"{prefix}.pop_act", ab.compute_grid, pop_act),
])
```

```python
# From Mcast.emit()
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
    f.flag(f"{prefix}.pop_src", f.all_cores),
    f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
])
```

### Unprefixed Flags: A Special Case

Some flags are deliberately unprefixed because they represent a global core-role assignment referenced by multiple ops. In `KNMatmul.emit()`, when not in `k_half_split` mode:

```python
if not k_half_split:
    f.per_core_unified_ct_args([
        f.flag("is_gate_compute_core", ab.a_grid),
        f.flag("is_up_compute_core", ab.b_grid),
    ])
```

The `is_gate_compute_core` and `is_up_compute_core` flags are unprefixed because they are a cross-cutting concern: `KNMatmul.emit()` sets them, and `GatedReduce.emit()` references the same A/B grid partition via the `a_grid` and `b_grid` parameters. This is an intentional design choice, not a naming oversight. The same pattern applies to `ag.sender_idx` and `bg.sender_idx`, which are set in `KNMatmul.emit()` and consumed by `GatedReduce.emit()`.

## Shared Semaphores

Semaphores are allocated via `f.semaphore(name)` (line 560 in `fused_program.py`). The name is a string, and like CT args, it should be scoped with the op's prefix:

```python
# From Mcast.emit()
if sender_sem is None:
    sender_sem = f.semaphore(f"{prefix}.sender")
receiver_sem = f.semaphore(f"{prefix}.receiver")
```

With `prefix = "shared_expert__mcast"`, these become:

- `"shared_expert__mcast.sender"`
- `"shared_expert__mcast.receiver"`

### Two Semaphore Modes

`FusedProgram.semaphore()` supports two distinct modes:

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

1. **Named mesh-global semaphores** (`name` provided, `program_semaphore=False`): Returns an L1 address. Deduped by name across `FusedProgram` instances sharing a `_sem_dict`. Used for cross-device rendezvous in multi-device configurations.

2. **Program-local semaphores** (`program_semaphore=True`): Per-device slot-based sync. Cheaper than mesh-global and appropriate for local synchronization within a single device.

Semaphore name deduplication is handled by `_alloc_mesh_semaphore()`:

```python
def _alloc_mesh_semaphore(mesh_device, name, sem_dict, initial_value=0):
    if name not in sem_dict:
        gs = mesh_device.compute_with_storage_grid_size()
        avail = ttnn.num_cores_to_corerangeset(gs.x * gs.y, gs, row_wise=True)
        sem_dict[name] = ttnn.create_global_semaphore(mesh_device, avail, initial_value)
    return sem_dict[name]
```

First call allocates; subsequent calls with the same name return the cached semaphore. The `FusedProgram` also guards against calling `f.semaphore()` with the same name twice within a single op:

```python
if any(s is sem for s in self.program._global_semaphores):
    raise RuntimeError(
        f"f.semaphore({name!r}) called twice within one op -- "
        "give distinct sync points distinct names."
    )
```

### Semaphore Sharing Across Stages

Sometimes two stages need to share a semaphore for cross-stage synchronization. The pattern is to allocate the semaphore in the parent `emit()` and pass it to the child:

```python
# From Mcast.emit() signature
def emit(f, src, *, prefix, sender_sem=None, ...):
    if sender_sem is None:
        sender_sem = f.semaphore(f"{prefix}.sender")
    ...
```

When `SharedExpert.emit()` calls `Mcast.emit()`, it passes `sender_sem=None`, letting `Mcast` allocate its own. But when `DownProj.emit()` calls `Mcast.emit()` for the bias multicast, it can pass a pre-allocated semaphore:

```python
# From DownProj.emit()
res = Mcast.emit(
    f, bias_tensor,
    prefix=BlazeOp.child_prefix(prefix, "mcast2"),
    sender_sem=mcast_sender_sem,  # can be passed from parent
    ...
)
```

This allows the parent FusedOp to coordinate synchronization across stages that the individual micro-ops cannot see. In `DownProj.compose()`, `mcast_sender_sem=0` is passed, meaning the compose path reuses semaphore slot 0 (the first program semaphore). In the embedded path, `None` is passed, and each mcast allocates its own.

## Scratch CBs: The Data-Flow Backbone

### Scratch CB Allocation

Scratch CBs are allocated via `f.cb_scratch()` and are the primary mechanism for passing data between micro-ops within a pipeline. Unlike tensor-backed CBs, scratch CBs are pure L1 allocations with no DRAM backing.

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

The `name` parameter is the CB's identity, constructed using `BlazeOp.cb_name()`. The returned `CBHandle` carries everything a downstream op needs:

```python
@dataclass(eq=False)
class CBHandle:
    cb_id: int
    num_pages: int
    page_size: int
    core_ranges: ttnn.CoreRangeSet
    data_format: object
    tile_desc: object
    byte_offset: int = 0
    backing_tensor: object = None
    access_mode: CBAccessMode = CBAccessMode.FIFO
```

The `__int__` and `__index__` methods allow `CBHandle` to be used directly where an integer CB ID is expected:

```python
def __int__(self) -> int:
    return self.cb_id

def __index__(self) -> int:
    return self.cb_id
```

### Tensor-Backed CBs vs Scratch CBs

The framework supports two kinds of circular buffers:

- **Scratch CBs** (`f.cb_scratch()`): Allocated by the framework, no backing tensor. Used for intermediate data within the fusion. Their L1 addresses are determined at program build time. Can be compacted (temporally reused) across non-overlapping pipeline stages.

- **Tensor-backed CBs** (`f.cb_from_tensor()`): Backed by an existing `ttnn.Tensor` in L1. Used for inputs (activations, weights) and outputs. Their L1 addresses are fixed by the tensor allocation.

In a typical FusedOp pipeline:
- The first stage receives tensor-backed CBs for its inputs.
- Intermediate stages use scratch CBs for inter-stage data.
- The final stage writes to a tensor-backed CB for the output.

SharedExpert illustrates this pattern: `activation` and `gate_up_weights` arrive as tensors (converted to tensor-backed CBs inside KNMatmul and Mcast), intermediate results flow through scratch CBs, and `output` is a tensor-backed CB in the final Gather.

### FIFO vs Direct-Address Access

The `access_mode` field on `CBHandle` distinguishes two consumption patterns:

- `CBAccessMode.FIFO` (default): Standard CB wait/pop semantics. The kernel calls `cb_wait_front()` and `cb_pop_front()`.
- `CBAccessMode.DIRECT_ADDRESS`: The kernel reads via `handle.buffer_address()` and must not treat the CB front as positioned. Used for weight tensors and shared L1 views.

The `require_fifo_handle()` function guards against incorrect usage:

```python
def require_fifo_handle(handle: CBHandle, op_name: str) -> CBHandle:
    """Return handle if it supports ordinary FIFO consumption."""
    if handle.is_direct_address:
        raise ValueError(
            f"{op_name} requires a FIFO CBHandle, but received a "
            "direct-address handle."
        )
    return handle
```

Ops that require FIFO semantics (like `KNMatmul` and `DownProj.matmul`) call this at the top of their `emit()`:

```python
# From KNMatmul.emit():
if isinstance(act, CBHandle):
    act_handle = require_fifo_handle(act, "KNMatmul.act")
```

### CBHandle Propagation Between Stages

The core composition pattern is returning and forwarding `CBHandle` objects. When one op's `emit()` returns a `CBHandle`, the next op receives it and can:

- Use it as a CT arg value (it coerces to `int` via `__int__()`, returning the `cb_id`).
- Read its `num_pages`, `page_size`, `data_format`, and `tile_desc` to derive downstream buffer sizes.
- Pass it directly to `f.unified_ct_args()`, `f.trisc_ct_args()`, etc.

This is the zero-copy handoff: the CB exists in L1, and the downstream op reads from the same physical buffer that the upstream op wrote to. No host-side data movement occurs.

### Tensor-to-CBHandle Conversion

When an `emit()` receives a raw `ttnn.Tensor` instead of a `CBHandle`, it converts it using `f.cb_from_tensor()`:

```python
# From Mcast.emit():
if isinstance(src, CBHandle):
    src_handle = src
else:
    src_handle = f.cb_from_tensor(src)
```

This pattern allows `emit()` functions to accept both `ttnn.Tensor` (when called from `compose()` with external tensors) and `CBHandle` (when called from another `emit()` with an intermediate result).

## Disjoint-Core CB Sharing

Scratch CBs allocated via `f.cb_scratch()` can share CB IDs when they live on disjoint core grids. The `FusedProgram` maintains an internal index (`_format_key_to_cb`) that allows CB ID reuse when:

1. Two CBs have the same `(data_format, tile_shape, page_size)` format key.
2. Their core ranges are disjoint (no overlapping cores).

The sharing is implemented via `_try_reuse_cb_id()`:

```python
def _try_reuse_cb_id(self, data_format, tile_desc, core_ranges, page_size=None):
    """Return the first cb_id in the bucket whose covered grid is
    disjoint from core_ranges; else None."""
    if _tensor_cb_share_disabled() or core_ranges is None:
        return None
    key = _format_key(data_format, tile_desc, page_size)
    if key is None:
        return None
    entries = self._format_key_to_cb.get(key)
    if not entries:
        return None
    for i, (ex_cb_id, ex_cr) in enumerate(entries):
        if _are_disjoint(ex_cr, core_ranges):
            entries[i] = (ex_cb_id, ex_cr.merge(core_ranges))
            return ex_cb_id
    return None
```

Where `_format_key()` produces a tuple `(data_format, (h, w), page_size)`:

```python
def _format_key(data_format, tile_desc, page_size=None):
    if data_format is None or tile_desc is None:
        return None
    h = getattr(tile_desc, "height", None)
    w = getattr(tile_desc, "width", None)
    if h is None or w is None:
        return None
    return (data_format, (h, w), page_size)
```

This optimization is transparent to the `emit()` author. Two independent micro-ops that each call `f.cb_scratch()` with compatible formats on disjoint cores will automatically share a CB ID, reducing the total CB count and enabling more aggressive L1 memory utilization.

### The balanced Parameter

The `balanced` parameter on `cb_scratch()` opts specific CBs out of temporal reuse. Mcast destinations are the canonical example: the sender pushes to all receiver cores, but only the downstream consumer pops. Cores that push but never pop have residual FIFO state. Reusing such a CB for a later phase would require a reset, which the compaction system does not inject. So `balanced=False` opts the CB out of temporal reuse:

```python
# From Mcast.emit()
dst = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "dst"),
    ...
    balanced=False,  # receivers push but not all pop
)
```

## reconfig() and Temporal Compaction

### Phase Boundaries

For multi-phase kernels that need to reconfigure circular buffers between phases, `FusedProgram` provides the `reconfig()` method (line 1876 in `fused_program.py`):

```python
def reconfig(self):
    """Mark a phase boundary. Called by CbReconfig.emit()."""
    self._reconfig_boundaries.append(self._op_index)
    self._format_key_to_cb.clear()
    self._view_tensor_cbs.clear()
```

When `reconfig()` is called:

1. The current op index is recorded as a phase boundary.
2. The CB reuse caches (`_format_key_to_cb`) are cleared, so phase-1 allocations cannot accidentally reuse phase-0 CB IDs.
3. The view tensor cache is cleared for the same reason.

This enables **temporal compaction**: scratch CBs that do not overlap in time can be assigned the same CB ID. The `_cb_lifetimes` tracking in `FusedProgram` records the first and last `_op_index` at which each CB is referenced. The `_build_multi_phase()` method uses these lifetimes along with the `_reconfig_boundaries` to reassign CB IDs across phases.

### cb_context(): Higher-Level Phase API

The `cb_context()` method (line 1903 in `fused_program.py`) provides a higher-level API for managing phases:

```python
def cb_context(self, name: str = "") -> "CBContext":
    """Create a named CB phase context. Marks a boundary but does NOT emit a graph node."""
    from .cb_reconfig import CBContext

    self.reconfig()
    ctx = CBContext(name=name, manager=self.cb_manager)
    self._cb_contexts.append(ctx)
    return ctx
```

This is used in advanced multi-phase ops where different stages require different CB configurations on the same physical buffers.

### CB Lifetime Tracking

Every `cb_scratch()`, `cb_from_tensor()`, `unified_ct_args()` (when a value is a `CBHandle`), and `output()` call updates the CB lifetime:

```python
def _mark_cb_use(self, cb_id: int):
    """Update CB lifetime to include the current op index."""
    if cb_id in self._cb_lifetimes:
        first, _ = self._cb_lifetimes[cb_id]
        self._cb_lifetimes[cb_id] = (first, self._op_index)
    else:
        self._cb_lifetimes[cb_id] = (self._op_index, self._op_index)
```

The `_op_index` counter increments with each `output()` call, providing a total ordering of micro-op phases within the fused kernel.

## ABGrid Partitioning

The `ABGrid` dataclass (line 362 in `fused_program.py`) partitions the compute grid into two branches for KN-sliced matrix multiply patterns:

```python
@dataclass
class ABGrid:
    """A/B (gate/up) core grids for KN-sliced matmul."""

    a_cores: list         # list[CoreCoord]
    b_cores: list         # list[CoreCoord]
    a_grid: ttnn.CoreRangeSet
    b_grid: ttnn.CoreRangeSet
    compute_grid: ttnn.CoreRangeSet
    num_per_branch: int

    @staticmethod
    def from_coords(a_coords, b_coords) -> "ABGrid":
        a_cores = [ttnn.CoreCoord(c, r) for c, r in a_coords]
        b_cores = [ttnn.CoreCoord(c, r) for c, r in b_coords]
        all_cores = a_cores + b_cores
        return ABGrid(
            a_cores=a_cores,
            b_cores=b_cores,
            a_grid=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in a_cores]),
            b_grid=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in b_cores]),
            compute_grid=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_cores]),
            num_per_branch=len(a_cores),
        )
```

The A branch computes the "gate" weights and the B branch computes the "up" weights. Both branches process the same activation tiles but different weight matrices. After the matmul phase, `GatedReduce` gathers the results from both branches, applies `SiLU(gate) * up`, and produces a reduced output on the sender core.

`ABGrid` is constructed either from explicit coordinates (passed by the caller) or from the device grid configuration:

```python
def build_ab_grids(self, ab_coords=None) -> ABGrid:
    """Build A/B core grids. Uses device grid if ab_coords not provided."""
    if ab_coords is not None:
        a_coords, b_coords = ab_coords
    else:
        a_coords, b_coords = self.grid.build_ab_grids()
    return ABGrid.from_coords(a_coords, b_coords)
```

The `ab_coords` flow from the top-level model configuration through `user_args` in `compose()`:

```python
# From SharedExpert.compose()
cls.emit(
    f,
    ...
    ab_coords=user_args.get("ab_grids"),
    ...
)
```

This allows the model layer to control core assignment based on the physical chip topology.

The `ABGrid` is not a framework-level concept -- it is a convention used by `KNMatmul` and related ops. Other ops partition cores differently. For example, `DownProj` uses `matmul_cores` (a subset of the grid derived from the device configuration) for its matmul stage, and `sender_grid` for its Mcast stage.

## FusedProgram Grid Properties

When composing micro-ops, you frequently need to reference the device grid. `FusedProgram` pre-computes several grid properties that are shared by all ops in the pipeline:

| Property | Type | Description |
|----------|------|-------------|
| `f.sender` | `CoreCoord` | The sender/master core |
| `f.sender_grid` | `CoreRangeSet` | Single-core set for the sender |
| `f.mcast_range` | `CoreRange` | Full rectangular grid |
| `f.all_cores` | `CoreRangeSet` | All compute cores (same as mcast_range) |
| `f.num_mcast_cores` | `int` | Total core count |
| `f.mcast_receiver_grid` | `CoreRangeSet` | All cores except sender |
| `f.matmul_cores` | `CoreRangeSet` | Cores available for matmul |
| `f.matmul_cores_cc` | `list[CoreCoord]` | Same as above, as a list |
| `f.num_matmul_cores` | `int` | Count of matmul cores |
| `f.noc_sender` | `CoreCoord` | NOC coordinates of sender |
| `f.noc_start` | `CoreCoord` | NOC coordinates of grid start |
| `f.noc_end` | `CoreCoord` | NOC coordinates of grid end |
| `f.full_device_grid` | `CoreRangeSet` | Complete device grid |

These are immutable properties of the device configuration. The exact core counts depend on the device grid configuration (e.g., a default 13x10 Blackhole grid yields approximately 130 Tensix cores, but the precise number of matmul-eligible, DRAM-worker, and phantom cores varies by chip variant).

For example, `Mcast.emit()` uses `f.sender_grid`, `f.mcast_receiver_grid`, `f.all_cores`, `f.noc_start`, `f.noc_end`, and `f.num_mcast_cores`:

```python
# From blaze/ops/mcast/op.py:
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
    ...
])

f.brisc_ct_args([
    (f"{prefix}.dest_noc_start_x", f.noc_start.x),
    (f"{prefix}.dest_noc_start_y", f.noc_start.y),
    (f"{prefix}.dest_noc_end_x", f.noc_end.x),
    (f"{prefix}.dest_noc_end_y", f.noc_end.y),
    (f"{prefix}.num_cores", f.num_mcast_cores),
    ...
])
```

## TileInfo: Tile Metadata for Cross-Op Propagation

When tile format needs to propagate between ops that cannot directly inspect each other's tensors, the `TileInfo` dataclass (line 32 in `fused_program.py`) carries the metadata:

```python
@dataclass
class TileInfo:
    tile: object      # ttnn.Tile
    data_format: object  # ttnn dtype
    size: int         # page size in bytes
    desc: object      # ttnn.TileDescriptor

    @staticmethod
    def from_tensor(tensor) -> "TileInfo":
        tile = tensor.get_tile()
        data_format = tensor.dtype
        return TileInfo(
            tile=tile,
            data_format=data_format,
            size=tile.get_tile_size(data_format),
            desc=ttnn.TileDescriptor(tile),
        )
```

`SharedExpert.emit()` constructs a `TileInfo` from the activation tensor and passes it to `GatedReduce.emit()` and `Mcast.emit()`:

```python
# From blaze/ops/shared_expert/op.py:
if isinstance(activation, ttnn.Tensor):
    ti = TileInfo.from_tensor(activation)
else:
    tile = ttnn.Tile([activation.tile_desc.height, activation.tile_desc.width])
    ti = TileInfo(
        tile=tile,
        data_format=activation.data_format,
        size=activation.page_size,
        desc=activation.tile_desc,
    )

# Passed to GatedReduce:
reduced = GatedReduce.emit(f, gu.handle, ..., tile_info=ti, ...)

# Passed to Mcast:
down_act = Mcast.emit(f, reduced, ..., dst_tile_info=ti, ...)
```

This ensures that the downstream ops use the correct tile geometry even when the intermediate data passes through scratch CBs that might have different native formats (e.g., face-view tiles in the reduce path).

## Sub-FusedOp Embedding

A FusedOp can embed another FusedOp by calling its `emit()` method. This is structurally identical to embedding a MicroOp -- `emit()` is `emit()` regardless of whether it calls `f.trisc_ct_args()` directly or chains other `emit()` calls. The difference is purely in the prefix depth.

`SharedExpert` embeds `DownProj`:

```python
# From SharedExpert.emit()
return DownProj.emit(
    f,
    down_act,
    down_weights,
    bias,
    output,
    prefix=BlazeOp.child_prefix(prefix, "down_proj"),
    mcast_sender_sem=None,
    core_coords=down_coords,
    ...
)
```

`DownProj.emit()` then chains its own children:

```python
# From DownProj.emit()
mm = DownProj.matmul(
    f, act_handle, weights_tensor,
    prefix=BlazeOp.child_prefix(prefix, "matmul"),
    cores=matmul_cores,
)
...
res = Mcast.emit(
    f, bias_tensor,
    prefix=BlazeOp.child_prefix(prefix, "mcast2"),
    sender_sem=mcast_sender_sem,
    ...
)
added = ResidualAdd.emit(
    f, mm, res,
    prefix=BlazeOp.child_prefix(prefix, "residual_add"),
    cores=matmul_cores,
    ...
)
return Gather.emit(
    f, added, output_tensor=output_tensor,
    prefix=BlazeOp.child_prefix(prefix, "gather"),
    dst_cb=gather_dst
)
```

### compose() vs emit() in Embedding

When embedding a sub-FusedOp, you call `emit()` -- **not** `compose()`. The differences are critical:

| | `compose()` | `emit()` |
|---|---|---|
| Called by | `BlazeCompiler` (top-level) | Other ops' `compose()` or `emit()` |
| Input format | `tensors: dict[str, Tensor]` | Explicit positional args, `CBHandle` or `Tensor` |
| Output handling | Wired to pre-allocated `output` tensor | Returns `CBHandle` |
| Prefix | From `user_args` with default | Explicit `prefix=` keyword argument |
| Standalone setup | May add extra stages (e.g., Mcast) | Core pipeline logic only |

`DownProj.compose()` adds an initial `Mcast` step:

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    act = Mcast.emit(f, tensors["input"], prefix="mcast")
    cls.emit(f, act, tensors["weights"], tensors["add_input"], output, ...)
```

When DownProj is embedded in SharedExpert, the Mcast has already been done by SharedExpert's stage 3, so `DownProj.emit()` is called directly, skipping the duplicate Mcast.

This separation -- `compose()` for the standalone entry point, `emit()` for the composable building block -- is a core design principle. A well-designed FusedOp makes both available.

### Arbitrary Nesting Depth

The nesting is arbitrary. `DenseMLP` embeds `SharedExpert`, which embeds `DownProj`, which embeds `Mcast`, `ResidualAdd`, and `Gather`. Larger compositions like `MoE` create even deeper prefix trees:

```
moe
  +-- moe__shared_expert
        +-- moe__shared_expert__gu
        +-- moe__shared_expert__gated_reduce
        +-- moe__shared_expert__mcast
        +-- moe__shared_expert__down_proj
              +-- moe__shared_expert__down_proj__matmul
              +-- moe__shared_expert__down_proj__mcast2
              +-- moe__shared_expert__down_proj__residual_add
              +-- moe__shared_expert__down_proj__gather
```

There is no technical limit on nesting depth. The `__` delimiter ensures uniqueness at every level, as long as each `emit()` caller passes a unique child name (which `child_prefix()` guarantees automatically).

### Delegation of Grid Configuration

When a sub-FusedOp is embedded, the parent controls which cores the sub-FusedOp uses. `SharedExpert.emit()` passes `down_coords` to `DownProj.emit()`:

```python
return DownProj.emit(
    f,
    down_act,
    down_weights,
    bias,
    output,
    prefix=BlazeOp.child_prefix(prefix, "down_proj"),
    core_coords=down_coords,
    ...
)
```

Inside `DownProj.emit()`:

```python
if core_coords is None:
    matmul_cores = f.matmul_cores
elif isinstance(core_coords, tuple):
    matmul_cores = coords_to_core_range_set(list(core_coords))
else:
    matmul_cores = coords_to_core_range_set(core_coords)
```

The sub-FusedOp does not assume ownership of any particular core set. It accepts `core_coords` from its parent, falls back to `f.matmul_cores` if not provided, and passes the resolved grid to its children.

This delegation pattern allows the same sub-FusedOp to be reused with different core assignments in different contexts. When `DownProj` runs standalone via `compose()`, it uses the full matmul grid. When embedded in `SharedExpert`, it uses whatever subset the parent assigns.

### Helper Methods on FusedOps

Some fused ops provide additional static methods as building blocks beyond the main `emit()`. For example, `DownProj` exposes a `matmul()` static method:

```python
@staticmethod
def matmul(f, in0_handle, weights_tensor, *, prefix="matmul", cores):
    """WIDTH_SHARDED matrix multiply building block."""
    ...
```

This is called by `DownProj.emit()` but could also be called directly by other ops that need the same width-sharded matmul pattern without the full DownProj pipeline.

## The output() Method: Recording Graph Nodes

Every micro-op's `emit()` ends with a call to `f.output()`, which serves two purposes:

1. **Creates a CBHandle** with all the metadata needed by downstream ops (including `_node_id` and `_port_name` for graph tracking).
2. **Records a node in the shadow graph** for visualization and codegen.

```python
# From Mcast.emit()
return f.output("mcast", dst, prefix=prefix, src=src)
```

```python
# From Gather.emit()
return f.output(
    "gather", dst_cb,
    num_pages=dst_num_pages,
    core_ranges=recv_grid,
    grid=src_handle.core_ranges,
    prefix=prefix,
    src=src_handle,
    receiver=recv_logical,
)
```

The `port_sources` keyword arguments (like `src=src_handle`) create edges in the graph. When `src` is a CBHandle with a `_node_id`, the graph records a data-flow edge from the source op's output to this op's input. This is how the framework tracks the full data-dependency graph across the composition.

The shadow graph is used for:

1. **Kernel auto-generation**: The codegen pipeline walks the graph to determine which phases to include in the generated kernel source.
2. **CB lifetime tracking**: `f._mark_cb_use(cb_id)` is called in `output()`, updating `_cb_lifetimes` for temporal compaction.
3. **Visualization and debugging**: The graph can be serialized for inspection.

## wire_output(): Binding CBs to Output Tensors

At the boundary between a fused op's internal pipeline and its external output tensor, the `wire_output()` method (line 1626 in `fused_program.py`) binds a scratch CB to an output tensor:

```python
def wire_output(self, handle: CBHandle, output_tensor) -> None:
    """Bind a scratch CB to an output tensor."""
```

Typically, the compiler calls `wire_output_from_graph()` automatically when `compose()` does not explicitly wire an output. This auto-wiring uses the last `CBHandle` recorded by `f.output()`:

```python
def wire_output_from_graph(self, output_tensor) -> None:
    """Wire the primary output CB to an output tensor."""
    if not self._last_output_cb_ids:
        raise RuntimeError("no output CB recorded")
    _, handle = self._last_output_cb_ids[0]
    self.wire_output(handle, output_tensor)
```

## Core Range Management Across Stages

Different stages of a fused pipeline run on different subsets of the Tensix grid. The framework manages this through `CoreRangeSet` objects and per-core flags.

Each micro-op declares which cores it runs on through its `is_active` flag:

```python
# KNMatmul: runs on the AB compute grid
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", ab.compute_grid),
])

# GatedReduce: runs only on the sender core (for the reduce phase)
f.per_core_unified_ct_args([
    f.flag(f"{gr}.is_active", f.sender_grid),
])

# Mcast: sender and receiver flags on different core sets
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
])
```

The kernel reads these flags at compile time and skips phases that are not active on a given core. This is how a single kernel binary can encode different behavior per core: core (0,0) might run Mcast sender + GatedReduce + Gather receiver, while core (3,5) might run Mcast receiver + KNMatmul gate + Gather sender.

Scratch CBs must be allocated on all cores that will access them. For example, Mcast's destination CB is allocated on `f.all_cores` because the sender writes to it (multicast) and every receiver reads from it:

```python
dst = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "dst"),
    num_pages=_dst_num_pages,
    core_ranges=f.all_cores,  # All cores need this CB
    ...
)
```

But GatedReduce's intermediate CB is only needed on the sender core:

```python
intermed = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "intermed"),
    num_pages=2,
    core_ranges=f.sender_grid,  # Only sender needs this
    ...
)
```

## Common Pitfalls and How to Avoid Them

### 1. Forgetting to Pass the Prefix

If you call a child `emit()` without passing a prefix (or passing an empty string), its CT args will be registered at the root namespace. This works for a single op but collides when two ops of the same type are composed.

**Always pass `prefix=BlazeOp.child_prefix(parent_prefix, "child_name")`.**

### 2. Duplicate Per-Core Flag Names

`FusedProgram._per_core_ct_args_check()` raises `ValueError` if a per-core CT arg name is registered twice. This catches naming collisions early -- at composition time, not at runtime.

### 3. Reusing a Semaphore Name Within One Op

`FusedProgram.semaphore()` raises `RuntimeError` if the same named semaphore is allocated twice within one op call. Each sync point needs a distinct name.

### 4. Scratch CB Name Reuse

`FusedProgram._cb_scratch_from_mapping()` tracks scratch CB names and raises `ValueError` on duplicates. Always use `BlazeOp.cb_name()` to construct unique CB names.

### 5. Using compose() Instead of emit() for Embedding

Calling `compose()` when embedding a sub-FusedOp will bypass prefix control and may add duplicate setup stages. Always use `emit()` for embedding.

### 6. Inconsistent Core Ranges

If a matmul allocates its output on `matmul_cores` but the downstream ResidualAdd expects it on `all_cores`, the kernel will read uninitialized memory on cores outside the matmul set. Always verify that the `core_ranges` in each `CBHandle` match the expectations of the consuming op.

## Summary of Namespace Mechanisms

| Mechanism | Delimiter | Example | Purpose |
|-----------|-----------|---------|---------|
| `child_prefix()` | `__` | `"shared_expert__gu"` | Scopes CT args, flags, semaphores |
| `cb_name()` | `___` | `"shared_expert__gu___gu_out"` | Scopes scratch CB names |
| CT arg `.` | `.` | `"shared_expert__gu.act"` | Separates op prefix from arg name |
| Semaphore name | `.` | `"shared_expert__mcast.sender"` | Follows CT arg convention |

## Summary of Resource Sharing Patterns

| Pattern | Mechanism | Example |
|---------|-----------|---------|
| **Prefix scoping** | `BlazeOp.child_prefix(prefix, name)` with `__` delimiter | Every fused op composition |
| **CB name scoping** | `BlazeOp.cb_name(prefix, name)` with `___` delimiter | Scratch CB naming |
| **Shared semaphores** | Allocate once, pass to multiple `emit()` calls | `Mcast.emit(sender_sem=shared_sem)` |
| **Disjoint-core CB sharing** | `_try_reuse_cb_id()` with format-key matching | Automatic for matching (dtype, tile, page_size) |
| **Temporal CB compaction** | `reconfig()` boundaries + `_cb_lifetimes` tracking | Multi-phase pipelines |
| **ABGrid partitioning** | `ABGrid.from_coords()` or `f.build_ab_grids()` | KN-sliced matmul gate/up split |
| **Sub-FusedOp embedding** | Call `FusedOp.emit()` with child prefix | `SharedExpert` embedding `DownProj` |
| **TileInfo propagation** | `TileInfo.from_tensor()` + explicit passing | Activation tile format across ops |
| **CBHandle as currency** | Return from `emit()`, accept in next `emit()` | Every producer-consumer boundary |
| **Grid delegation** | Pass `core_coords` to sub-FusedOp `emit()` | `SharedExpert` -> `DownProj` |
