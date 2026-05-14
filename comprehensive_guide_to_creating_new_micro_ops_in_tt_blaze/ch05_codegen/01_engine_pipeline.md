# 5.1 The Engine Pipeline, Shadow Graph, and BlazeCompiler

Every TT-Blaze kernel starts as a sequence of Python `emit()` calls. Those calls allocate circular buffers, declare compile-time arguments, assign semaphores, and set role flags. But between the Python declarations and the final JIT-compiled kernel binary, four engines transform those declarations into concrete values, and a shadow graph captures the execution order so the codegen can reconstruct it. This section covers that entire pipeline.

---

## 5.1.1 Overview: from emit() to device dispatch

The pipeline has three major phases:

```
Phase 1: Graph Construction
  emit() calls on FusedProgram
     |
     v
  Shadow graph (BlazeGraph) records OpNodes, edges, CT arg values
     |
Phase 2: Engine Pipeline
     v
  CBEngine   -> CB IDs, buffer sizes, lifetimes
  SemEngine  -> semaphore indices, global semaphore objects
  RoleEngine -> (GridConfig already available on FusedProgram)
  CTArgEngine -> resolved (name, value) tuples per RISC
     |
Phase 3: Codegen + JIT
     v
  kernel_codegen.py -> generated .cpp file (or handwritten kernel)
  named_args_generated.h -> per-RISC CT arg structs
  JIT compiler -> 3 binaries (NCRISC, BRISC, TRISC)
     |
     v
  ProgramDescriptor -> ttnn.generic_op() -> device execution
```

In the composition-driven path (the most common path in production), `FusedProgram` manages the shadow graph internally. The engines are invoked either by `BlazeCompiler.compile()` (graph-API path) or implicitly during `FusedProgram.build()` (imperative path). The two paths converge at JIT compilation.

---

## 5.1.2 The shadow graph

The shadow graph is the bridge between `emit()` calls and codegen. Every `FusedProgram` maintains an internal `BlazeGraph` instance at `self._shadow_graph`. Each time an op calls `f.output()` (the final step of the three-step `emit()` pattern), the FusedProgram records:

1. **An OpNode** with the op's name, spec, grid placement, kwargs, and accumulated CT arg values.
2. **Input edges** from upstream CBHandles (each CBHandle carries `_node_id` and `_port_name` set by the producing op's `f.output()` call).
3. **External input ports** for tensors that enter from outside the fusion.

The recording mechanism lives in two methods:

```python
# blaze/fused_program.py

def _create_op_node(self, op_name, grid, **kwargs):
    """Create an OpNode, add it to the shadow graph, and flush pending state."""
    spec = get_op_spec(op_name)
    count = self._op_counters.get(op_name, 0) + 1
    self._op_counters[op_name] = count
    node_id = f"{op_name}_{count}"

    node = OpNode(
        id=node_id,
        spec=spec,
        grid=grid,
        kwargs=kwargs,
        ct_values=dict(self._pending_ct_values),
    )
    self._pending_ct_values.clear()
    self._shadow_graph.add_node(node)
    return node, node_id

def _wire_input_edges(self, node, node_id, inputs):
    """Wire input edges from source handles/tensors to the given node."""
    for port_name, source in inputs.items():
        if isinstance(source, CBHandle) and source._node_id:
            producer = self._shadow_graph.node_map[source._node_id]
            edge = Edge(producer=producer, producer_port=source._port_name,
                        consumer=node, consumer_port=port_name)
            self._shadow_graph.add_edge(edge)
        elif isinstance(source, ExternalTensor):
            self._shadow_graph.external_input_ports.add((node_id, port_name))
```

Key design points:

- **Node IDs are auto-incremented**: `_create_op_node` uses `_op_counters` to generate IDs like `mcast_1`, `matmul_2`. Node insertion order follows `emit()` call order, which defines the execution sequence.
- **CT values are flushed per-op**: `_pending_ct_values` accumulates all `f.unified_ct_args(...)` / `f.ncrisc_ct_args(...)` calls made by an op's `emit()`, then `_create_op_node` snapshots and clears them.
- **CBHandle carries graph identity**: after `_record_op`, the output CBHandle's `_node_id` and `_port_name` fields let downstream ops' `emit()` calls trace edges back to the producer. This is how the shadow graph captures dataflow.

**Key insight: node insertion order equals phase execution order.** The shadow graph preserves the order in which `emit()` calls happen. This insertion order directly becomes the phase sequence in the generated kernel's `kernel_main()`. Getting this order wrong (e.g., emitting a consumer before its producer) causes device hangs because the consumer waits on an empty CB that the producer has not yet filled.

### Deferred codegen

When `FusedProgram` is created without an explicit `kernel` argument, the kernel source is generated from the shadow graph at build time:

```python
# blaze/fused_program.py

def build(self, noc_mode=None):
    self._prepare_for_build()

    # Deferred codegen: generate kernel from shadow graph if not provided
    if self.program._kernel is None:
        self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)

    pd = self.program.build()
    ...
```

This deferral is essential because:

1. **CB compaction happens first**: `_prepare_for_build()` runs multi-phase CB reconfiguration, which may remap CB IDs. The kernel must see final CB IDs.
2. **Noop tagging**: `_tag_noop_nodes()` marks shadow graph nodes matching noop prefixes. The kernel codegen emits empty `DeviceZoneScopedN` blocks for these nodes instead of actual op code.
3. **The complete graph is needed**: all `emit()` calls must have completed before the graph is valid for codegen.

### The output() method and graph recording

The `output()` method on FusedProgram is the canonical way for an `emit()` function to declare its output:

```python
def output(self, op_name, handle=None, *, cb_id=None, prefix="", **port_sources):
```

It combines CBHandle construction with `_record_op()` into a single call. Port inputs are passed as keyword arguments matching the op's `Input()` descriptor names. This ensures the shadow graph faithfully records the dataflow for both codegen and visualization.

---

## 5.1.3 CBEngine: circular buffer assignment

**Source:** `blaze/cb_engine.py`

CBEngine walks the graph in topological order and assigns a unique CB ID to every data path. There are four categories of CB allocations:

| Category | Key prefix | `is_tensor_backed` | Description |
|----------|-----------|---------------------|-------------|
| `ext_in` | `ext_in_{node}_{port}` | True | External tensor inputs (sharded L1 buffers) |
| `internal` | `internal_{node}_{port}` | False | Internal scratch buffers declared in OpSpec |
| `intermed` | `intermed_{node}_{port}` | False | Inter-op edges (output of one op feeding another) |
| `ext_out` | `ext_out_{node}_{port}` | True | Terminal outputs written to a tensor |

### The CBAssignment dataclass

Each allocation produces a `CBAssignment`:

```python
@dataclass
class CBAssignment:
    cb_id: int
    total_size: int       # bytes
    page_size: int
    num_pages: int
    core_ranges: Any      # which cores need this CB
    is_tensor_backed: bool
    data_format: str
    tile_shape: tuple[int, int]
    node_id: str = ""
    port_name: str = ""
    category: str = ""    # "ext_in" | "intermed" | "ext_out" | "internal"
    first_use: int = 0    # topo position of first node that reads/writes
    last_use: int = 0     # topo position of last node that reads/writes
```

### Assignment algorithm

1. **Topological walk:** Nodes are visited in topological order. For each node:
   - External input ports not already connected via an incoming edge get a new CB ID.
   - Internal CBs declared in the `OpSpec.internal_cbs` list get a new CB ID.
   - Output ports: if the output has downstream consumers (edges), it becomes an `intermed` CB; otherwise, it becomes an `ext_out` CB.

2. **Fan-out sharing:** When one output port feeds multiple consumers, all edges share a single CB ID. The `core_ranges` is the **union** of the producer grid and all consumer grids, computed via `_union_grids()`:

```python
# From CBEngine.assign() -- fan-out handling
if not port_edges:
    key = f"ext_out_{node.id}_{port_spec.name}"
    ...
else:
    all_grids = [node.grid] + [e.consumer.grid for e in port_edges]
    core_ranges = _union_grids(all_grids)
    cb_id = _alloc(dtype, tile_shape)
    for e in port_edges:
        e.cb_id = cb_id  # Same CB ID for all consumers
```

3. **Shared tensor deduplication:** If multiple ports on different nodes reference the same `ExternalTensor` name (via `graph.tensor_to_ports`), they reuse the same CB ID. This is tracked via the `shared_tensor_cb` dict.

4. **Lifetime tracking:** After assignment, the engine computes `first_use` and `last_use` for each CB based on topological positions. This feeds into the compact mode.

### MAX_CB_ID and the hardware limit

```python
MAX_CB_ID = 64  # Blackhole hardware limit
```

The Blackhole architecture supports at most 64 circular buffers per core. The engine warns when usage exceeds `MAX_CB_ID - 8` (56 CBs), and raises `ValueError` if the limit is exceeded. Two mitigation strategies exist:

- **Compact mode** (`compact=True`): Applies interval graph coloring to reuse CB IDs for buffers with non-overlapping lifetimes. Two CBs whose `[first_use, last_use]` intervals do not overlap can share an ID:

```python
def compact_cb_ids(self, assignments):
    """Recolor CB IDs via interval coloring."""
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

- **CircularBufferIdManager integration:** When a `cb_id_manager` is provided, CB IDs are allocated through tt-metal's `CircularBufferIdManager`, which handles data-format-aware ID reuse natively.

### Disjoint-core sharing (FusedProgram path)

In the FusedProgram path (rather than the graph engine path), disjoint-core sharing is handled directly during `emit()`. The `_try_reuse_cb_id()` method on `FusedProgram` checks whether an existing CB with matching `(data_format, tile_shape, page_size)` has a core grid that is **disjoint** from the new allocation's grid. If so, the same CB ID is reused with a second `CBFormatDescriptor` appended. This is critical for fitting complex fused ops within the hardware CB limit.

### Exporting to CircularBufferIdManager

The `to_cb_id_manager()` method exports engine assignments into a `CircularBufferIdManager`, pre-seeded with all allocated IDs. This allows downstream code (e.g., MoE ops) to allocate additional CBs without ID collisions:

```python
manager = cb_engine.to_cb_id_manager(assignments)
# Downstream code can now call manager.get_cb_id(dtype, tile_desc) safely
```

---

## 5.1.4 SemEngine: semaphore assignment

**Source:** `blaze/sem_engine.py`

The SemEngine identifies inter-core synchronization points in the graph and assigns sequential **global semaphore indices**. Each index maps to a `ttnn.GlobalSemaphore` object created at compile time.

### Sync point identification

The engine walks the graph in topological order and calls `_get_sem_source_keys()` for each node. This method looks up the node's `CTArgSchema` and extracts entries with `source == "sem"`:

```python
def _get_sem_source_keys(self, node: OpNode) -> list[str]:
    schema = get_ct_schema(node.spec.name)
    if schema is None:
        return []
    source_keys = list(dict.fromkeys(
        spec.source_key for spec in schema.args if spec.source == "sem"
    ))
    # Conditional noc1 for gather/moe_gather
    if node.spec.name in ("gather", "moe_gather") and not node.kwargs.get("dual_noc", True):
        source_keys = [k for k in source_keys if "noc1" not in k]
    return source_keys
```

Each `source_key` (e.g., `"noc0_receiver_semaphore"`, `"sender_semaphore"`) identifies a specific semaphore purpose. The same `source_key` across multiple RISCs produces only one semaphore (deduplication via `dict.fromkeys`).

### The SemAssignment dataclass

```python
@dataclass
class SemAssignment:
    sem_id: int          # sequential index into global semaphore list
    global_sem: bool     # always True
    initial_value: int
    core_ranges: Any
    protocol: str        # e.g. "noc0_receiver_semaphore", "sender_semaphore"
    node_id: str
```

### Mcast sender grouping

Mcast ops that share the same explicit `sender` kwarg use **persistent NOC command buffers**. Their sender semaphores must use the **same** semaphore ID because the kernel design requires sequential operations (init, run_as, teardown) on a single mcast object.

The `_McastSenderGroups` class tracks this:

```python
class _McastSenderGroups:
    def __init__(self, sync_points: list[SyncPoint]):
        self._groups: dict[str, int] = {}
        counter = 0
        for sp in sync_points:
            key = self._sender_key(sp)
            if key is not None and key not in self._groups:
                self._groups[key] = counter
                counter += 1

    @staticmethod
    def _sender_key(sp: SyncPoint) -> str | None:
        if sp.node.spec.name == "mcast" and sp.protocol == "sender_semaphore":
            sender = sp.node.kwargs.get("sender")
            if sender is not None:
                return str(sender)
        return None
```

When two mcast sender sync points have the same `sender_key`, they are assigned the same `sem_id`. All other sync points get unique sequential indices.

### Global semaphore creation

The `BlazeCompiler._create_global_semaphores()` method converts engine-assigned indices into actual `ttnn.GlobalSemaphore` objects:

```python
def _create_global_semaphores(self, sem_assignments):
    num_sems = max(sa.sem_id for sa in sem_assignments) + 1
    global_sems = [ttnn.create_global_semaphore(self._mesh_device, avail, 0)
                   for _ in range(num_sems)]
    sem_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in global_sems]
```

The resulting L1 addresses are fed to the CTArgEngine, which maps them to the appropriate `source_key` CT args so the kernel can call `get_semaphore(addr)`.

> **Warning:** GlobalSemaphore objects must be kept alive for the lifetime of the compiled program. The compiler pins them in `MeshCompiledProgram._lifetime_semaphores`. If they are garbage collected, the program reads freed L1 (use-after-free).

---

## 5.1.5 RoleEngine: per-core boolean flags

**Source:** `blaze/role_engine.py` (GridConfig) and `blaze/fused_program.py` (flag composition)

The RoleEngine is not a standalone class like CBEngine or SemEngine. Instead, per-core role flags are composed directly in the FusedProgram via the `flag()` method and the `per_core_unified_ct_args()` family of calls.

### The flag() method

```python
def flag(self, name, cores, enabled=True):
    """Build a per-core CT arg tuple for a boolean flag.
    Returns a (name, mapping) tuple suitable for passing to
    per_core_unified_ct_args() and variants.
    """
    return (name, {cores: int(enabled)})
```

The `flag()` method returns a `(name, {CoreRangeSet: value})` tuple. When passed to `per_core_unified_ct_args()`, the underlying `UnifiedCompileTimeCoreDescriptor` system splits the kernel compilation into groups: cores in the specified range get value `1`, all others get `0`.

### CoreRole composition

In practice, `emit()` functions compose multiple flags to define the **role** of each core:

```python
f.per_core_unified_ct_args([
    f.flag("is_gate_compute_core", ab.a_grid),
    f.flag("is_up_compute_core", ab.b_grid),
    f.flag("is_gated_reduce_core", f.sender_grid),
    f.flag("is_mcast_sender_core", f.sender_grid),
    f.flag("is_mcast_receiver_core", f.mcast_receiver_grid),
    f.flag("is_matmul_core", f.matmul_cores),
    f.flag("is_gather_receiver_core", f.sender_grid),
])
```

Each flag becomes a `static constexpr bool` in the kernel via `get_named_compile_time_arg_val()`. The C++ compiler uses these for dead code elimination -- on a core where `is_mcast_sender_core == false`, all sender code paths are compiled away entirely.

### GridConfig

The `GridConfig` dataclass provides the device-specific grid layout used to derive core roles:

```python
@dataclass(frozen=True)
class GridConfig:
    grid_cols: int        # e.g. 13 for unharvested Blackhole
    grid_rows: int        # e.g. 10
    dram_worker_positions: frozenset[tuple[int, int]]
```

Key properties:
- `sender_core`: `(grid_cols - 1, grid_rows - 1)` -- the designated multicast sender core
- `get_matmul_cores()`: all cores excluding DRAM workers, phantom columns, and sender
- `build_ab_grids()`: balanced A/B core grids for gate/up matmul (ensures `len(a_cores) == len(b_cores)`)
- `get_gate_mm_cores(num_cores)`: phantom column cores for gate weight placement
- `build_matmul_core_grid()` and `build_mcast_receiver_grid()`: construct `CoreRangeSet` objects for matmul cores and mcast receivers

GridConfig adapts to harvested Blackhole devices:

```python
@classmethod
def from_device(cls, device):
    grid_size = device.compute_with_storage_grid_size()
    dram_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    return cls(
        grid_cols=grid_size.x,
        grid_rows=grid_size.y,
        dram_worker_positions=frozenset((c.x, c.y) for c in dram_workers),
    )
```

### Per-core vs unified CT args

There are two mechanisms for per-core specialization:

1. **`UnifiedCompileTimeCoreDescriptor`**: A single value for all cores in a `CoreRangeSet`, and a different value for all other cores. Best for binary flags (is/isn't a sender).

2. **`PerCoreCompileTimeDescriptor`**: A unique value for each individual `CoreCoord`. Used when each core needs a distinct compile-time value (e.g., `sender_idx` in gather ops where each sender core gets its own sequential index):

```python
descriptors.append(
    PerCoreCompileTimeDescriptor(
        f"{prefix}.sender_idx",
        core_values=[(core, idx) for idx, core in enumerate(sender_cores)],
        other_value=0,
    )
)
```

---

## 5.1.6 CTArgEngine: compile-time argument resolution

**Source:** `blaze/ct_args.py`

The CTArgEngine is the final engine in the pipeline. It walks the graph, looks up each op's `CTArgSchema`, and resolves concrete `(name, value)` tuples for each RISC processor.

### CTArgSpec and CTArgSchema

Each micro-op registers a `CTArgSchema` describing its compile-time args:

```python
@dataclass
class CTArgSpec:
    name: str          # base arg name (e.g. "out_w")
    risc: str          # "ncrisc", "brisc", or "trisc"
    type: str          # C++ type ("uint32_t", "bool", etc.)
    source: str        # "cb" | "sem" | "derived" | "per_core"
    source_key: str    # key into the relevant assignment dict
    description: str
```

The `source` field determines how the value is resolved:

| Source | Resolution | Example |
|--------|-----------|---------|
| `"cb"` | CB ID from CBEngine assignments | `source_key="in0"` resolves to the CB ID assigned to port `in0` |
| `"sem"` | Semaphore L1 address from SemEngine | `source_key="noc0_receiver_semaphore"` resolves to the global sem address |
| `"derived"` | User kwargs, grid context, or computed values | `source_key="k_num_tiles"` from user args |
| `"per_core"` | Skipped here; resolved by PerCoreCompileTimeDescriptor at assembly time | `source_key="sender_idx"` |

### Resolution logic

```python
def _resolve_value(self, spec, cb, sem, user):
    if spec.source == "cb":
        return (cb[spec.source_key], True) if spec.source_key in cb else (0, False)
    elif spec.source == "sem":
        return (sem[spec.source_key], True) if spec.source_key in sem else (0, False)
    elif spec.source == "per_core":
        return (int(user.get(spec.source_key, 0)), True) if spec.source_key in user else (0, False)
    else:
        # "derived", "user", "grid" -- all from user dict
        val = user.get(spec.source_key, 0)
        if isinstance(val, float):
            return (_float_to_bits(val), True)
        return (int(val), True)
```

Note the float-to-bits encoding: when a CT arg value is a Python `float`, the engine encodes it as a `uint32_t` bit pattern via `_float_to_bits()`. The C++ side reinterprets the bits back to a float. This avoids the overhead of runtime float argument passing while preserving full precision.

### Auto-prefixing

To avoid name collisions when multiple instances of the same op type appear in a fused kernel, the engine **auto-prefixes** each arg name with the op's CT arg prefix:

```python
def _compute_prefix(self, node, graph, merged_user=None):
    source = merged_user if merged_user is not None else node.kwargs
    prefix = source.get("ct_prefix", node.spec.name)
    if not prefix:
        return ""
    return f"{prefix}."
```

For example, two matmul ops with `ct_prefix="gate"` and `ct_prefix="up"` produce args like `gate.out_w` and `up.out_w` instead of colliding on `matmul.out_w`. Setting `ct_prefix=""` disables prefixing, used for fused ops where arg names are already fully qualified.

### Shared prefix deduplication

When multiple graph nodes share the same **explicit** `ct_prefix` (e.g., gate and up matmul both using `"gu"`), the engine generates args from the first node and reuses them for subsequent nodes. This avoids duplicate entries in the generated header:

```python
if explicit_prefix and prefix in shared_prefix_args:
    result[node.id] = shared_prefix_args[prefix]
    continue
```

Implicit prefixes (defaulting to `spec.name`) are NOT shared -- collisions between those are caught by validation.

### Collision detection

After resolving all args, the engine validates that no two args share the same prefixed name within a RISC. Shared-prefix nodes reference the same list object (Python identity), so they are automatically excluded from collision checks:

```python
def _validate_no_collisions(self, all_args):
    per_risc = {"ncrisc": {}, "brisc": {}, "trisc": {}}
    seen_ids: set[int] = set()  # list identity dedup
    for node_id, args in all_args.items():
        if id(args) in seen_ids:
            continue
        seen_ids.add(id(args))
        for arg in args:
            existing = per_risc[arg.risc].get(arg.name)
            if existing is not None:
                raise ValueError(
                    f"CT arg name collision on {arg.risc}: '{arg.name}' "
                    f"defined by both '{existing}' and '{node_id}'"
                )
            per_risc[arg.risc][arg.name] = node_id
```

### Strict mode

When `strict=True` (used by the compiler path), CTArgEngine raises `ValueError` for any unresolved args (args whose source key was not found in the lookup dict). In non-strict mode, it emits a warning and defaults unresolved values to 0.

### Output format

The `generate_tuples()` method returns the final output consumed by `UnifiedKernelDescriptor`:

```python
def generate_tuples(self, graph, cb_assignments, sem_assignments, ...
) -> dict[str, list[tuple[str, int]]]:
    # Returns: {"ncrisc": [(name, value), ...],
    #           "brisc":  [(name, value), ...],
    #           "trisc":  [(name, value), ...]}
```

Per-core args (`source="per_core"`) are excluded from `generate_tuples()` because they use a separate descriptor system (`PerCoreCompileTimeDescriptor`), not the flat `named_compile_time_args` list.

---

## 5.1.7 BlazeCompiler as the orchestrator

**Source:** `blaze/compiler.py`

`BlazeCompiler` ties the engines together. Its `compile()` method follows this sequence:

```
graph + tensors + user_args
    |
    v
[1] Pre-compute engine results (device-agnostic)
    |-- CBEngine().assign(graph)
    |-- SemEngine().assign(graph)
    v
[2] Per-device loop (for each device in the mesh):
    |-- Merge user_args with per-device overrides + mesh context
    |-- Dispatch: fused-op path vs. engine path
    |   |
    |   +-- Fused-op path (single-node graph with compose()):
    |   |       FusedProgram -> compose() -> emit() chains -> build()
    |   |       Shadow graph -> kernel_codegen (if no handwritten kernel)
    |   |
    |   +-- Engine path (multi-node graph):
    |       |-- Build CB descriptors from assignments + tensors
    |       |-- Create global semaphores, map to L1 addresses
    |       |-- CTArgEngine().generate_tuples() with CB/sem/grid context
    |       |-- Build UnifiedKernelDescriptor
    |       |-- Assemble ProgramDescriptor
    |       v
    |   CompiledProgram
    v
[3] Assemble MeshProgramDescriptor from per-device programs
    |
    v
MeshCompiledProgram.run() -> ttnn.generic_op()
```

### Engine result caching

Engine results are device-agnostic -- CB assignment logic and semaphore index assignment depend only on graph topology, not on specific tensor addresses. BlazeCompiler computes them once before the per-device loop:

```python
engine_results = _EngineResults(
    cb_assignments=CBEngine().assign(graph),
    sem_assignments=SemEngine().assign(graph),
)
```

### Per-device compilation dispatch

For each device in the mesh, BlazeCompiler calls `_compile_single()`, which dispatches based on graph size:

```python
def _compile_single(self, ctx, graph, tensors, output_tensor, ...):
    if len(graph.nodes) == 1:
        config = get_fused_op_config(graph.nodes[0].spec.name)
        if config is not None:
            return self._compile_fused_op(graph, tensors, output_tensor, config, ...)
    return self._compile_for_device(ctx, graph, tensors, output_tensor, ...)
```

The **fused-op path** (`_compile_fused_op`) creates a `FusedProgram`, calls the op's `compose()` function which chains `emit()` calls, then calls `f.build()`. The FusedProgram's shadow graph drives kernel codegen when no handwritten kernel is provided. This is the path used by most composed ops (SharedExpert, RoutedExpert, MoE, etc.).

The **engine path** (`_compile_for_device`) is for multi-node graphs built with `blaze.fuse()`. It uses the pre-computed engine results and builds everything explicitly.

### Mesh assembly

Per-device `ProgramDescriptor` objects are keyed by `MeshCoordinate` and assembled into a `MeshProgramDescriptor`:

```python
mesh_pd = ttnn.MeshProgramDescriptor()
for idx in range(num_devices):
    row, col = divmod(idx, self._num_cols)
    compiled = self._compile_single(...)
    coord = ttnn.MeshCoordinate(row, col)
    mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = compiled.program_descriptor
```

### Lifetime pinning: preventing use-after-free

A subtle but critical concern is lifetime management. The `ProgramDescriptor` holds raw C++ pointers (e.g., `CBDescriptor.buffer` is a `Buffer*` pointing into a tensor's L1 allocation). If the backing Python tensor is garbage collected between `compile()` and `run()`, those pointers become dangling.

BlazeCompiler addresses this with explicit GC-pinning lists:

```python
program = MeshCompiledProgram(
    mesh_program_descriptor=mesh_pd,
    io_tensors=mesh_io_tensors,
    output_tensor=output_tensor,
    lifetime_tensors=all_per_device_io_tensors
        + list(self._tensor_dict.values())
        + list(self._internal_tensor_dict.values()),
    lifetime_semaphores=list(self._sem_dict.values()) + list(self._graph_sem_pool),
)
```

The `_lifetime_tensors` list keeps Python references alive for every per-device tensor whose `Buffer*` appears in a `CBDescriptor`. Similarly, `_lifetime_semaphores` pins `GlobalSemaphore` objects whose L1 addresses were baked into CT args.

### Auto-export for visualization

When `BLAZE_EXPORT=1` is set, BlazeCompiler automatically exports a JSON file containing the shadow graph, tensor metadata, and runtime args. This file feeds an interactive visualizer that renders the dataflow graph with CB IDs, semaphore assignments, and per-core CT arg values. This is invaluable for debugging new micro-ops: set `BLAZE_EXPORT=1`, run a test, and open the visualizer to see exactly how your op's CBs, semaphores, and CT args were assigned.

---

## 5.1.8 Engine pipeline execution order

The engines have a dependency order:

```
CBEngine (assigns CB IDs)
    |
    v
SemEngine (assigns semaphore IDs; uses CTArgSchema to find sem fields)
    |
    v
CTArgEngine (resolves all args; needs CB IDs and sem IDs as inputs)
    |
    v
kernel_codegen (reads shadow graph; produces .cpp from PhaseInfo registry)
    |
    v
JIT (compiles .cpp with named_args_generated.h)
```

Note that `kernel_codegen` does not depend on the engine outputs directly. It reads the shadow graph (which was populated during `emit()` calls) and the `PhaseInfo` registry (which was populated at module import time when ops were registered). The engines and codegen operate on independent data paths that converge at the JIT stage, where the generated `.cpp` file and the `named_args_generated.h` (produced from engine CT arg tuples) are compiled together.

---

## Key takeaways

- The **shadow graph** records every `emit()` call as a `BlazeGraph`, serving both kernel codegen and engine validation. Node insertion order directly determines phase execution order in the kernel.
- **CBEngine** assigns sequential CB IDs, handles fan-out via grid union, deduplicates shared tensors, tracks lifetimes for optional compaction, and respects the 64 CB ID hardware limit (`MAX_CB_ID = 64`).
- **SemEngine** identifies inter-core sync points from CTArgSchema `source="sem"` entries and groups mcast sender semaphores that share persistent state.
- **CTArgEngine** resolves final values from four sources (`"cb"`, `"sem"`, `"derived"`, `"per_core"`), auto-prefixes names, validates no collisions, and outputs per-RISC tuples.
- **BlazeCompiler** orchestrates everything: runs engines, builds descriptors, creates global semaphores, handles mesh compilation, and uses lifetime pinning to prevent use-after-free.
- Kernel codegen is deferred to build time, after CB compaction and noop tagging have finalized the graph.
