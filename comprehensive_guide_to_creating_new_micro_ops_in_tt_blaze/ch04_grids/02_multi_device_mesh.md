# 02 -- Multi-Device Mesh Compilation and CCL Ops

Source: `blaze/compiler.py`, `blaze/fused_program.py`, `blaze/ccl.py`, `blaze/ops/ccl_broadcast/op.py`, `blaze/ops/all_reduce/op.py`

TT-Blaze supports multi-device execution on Tenstorrent mesh topologies. This section explains how the compiler extends single-device compilation to meshes, and how CCL (collective communication library) ops use the fabric interconnect to coordinate across devices.

---

## MeshDevice fundamentals

A `ttnn.MeshDevice` represents a rectangular grid of Tenstorrent devices. Its shape is accessed as a two-element tuple:

```python
mesh_shape = mesh_device.shape  # e.g. (2, 4) for a 2x4 T3K
num_rows = mesh_shape[0]
num_cols = mesh_shape[1]
```

Common configurations:

| System | Mesh Shape | Total Devices |
|---|---|---|
| Single device | 1x1 | 1 |
| T3K | 1x8 or 2x4 | 8 |
| Galaxy | 4x8 or 8x4 | 32 |

A MeshDevice exposes the same device-level APIs as a single device -- `compute_with_storage_grid_size()`, `worker_core_from_logical_core()`, etc. -- because Tenstorrent meshes are homogeneous: every device in the mesh has the same grid dimensions and DRAM worker layout. This means a single `DeviceContext` can be shared across all devices in a mesh.

Tensors on a mesh are "mesh tensors": a single `ttnn.Tensor` object that internally wraps one shard per device. The function `ttnn.get_device_tensors(mesh_tensor)` decomposes a mesh tensor into a list of per-device tensors, indexed in row-major order.

---

## BlazeCompiler: mesh compilation

`BlazeCompiler` (defined in `blaze/compiler.py`) is the primary entry point for compiling a `BlazeGraph` into an executable program. It always operates at the mesh level, even for single-device execution (a 1x1 mesh).

### Construction

```python
compiler = BlazeCompiler(mesh_device)
```

The constructor extracts the mesh shape and creates a single `DeviceContext`:

```python
class BlazeCompiler:
    def __init__(self, mesh_device):
        self._mesh_device = mesh_device
        mesh_shape = mesh_device.shape
        self._num_rows = mesh_shape[0]
        self._num_cols = mesh_shape[1]
        self._ctx = DeviceContext.from_device(mesh_device)
        self._sem_dict: dict = {}
        self._graph_sem_pool: list = []
        self._tensor_dict: dict = {}
        self._internal_tensor_dict: dict = {}
```

The `_sem_dict`, `_tensor_dict`, and `_internal_tensor_dict` are shared across all per-device compilation iterations within a single `compile()` call, enabling cross-device deduplication of semaphores and tensors.

### The compile() method

`compile()` takes a `BlazeGraph`, a dictionary of mesh tensors, and a pre-allocated output mesh tensor, then returns a `MeshCompiledProgram`. The compilation follows this flow:

1. **Pre-compute device-agnostic results.** `CBEngine` and `SemEngine` run once on the graph to produce CB assignments and semaphore assignments. These are reused across all devices:

```python
engine_results = _EngineResults(
    cb_assignments=CBEngine().assign(graph),
    sem_assignments=SemEngine().assign(graph),
)
```

2. **Split mesh tensors.** `_split_tensors()` decomposes mesh-level tensors into per-device tensors. Output tensors are split via `ttnn.get_device_tensors(output_tensor)`.

3. **Auto-create global semaphores.** If the op class declares `num_global_semaphores`, the compiler creates them before the per-device loop (see the "Auto-created global semaphores" section below).

4. **Per-device compilation loop.** For each device `(row, col)` in the mesh:

```python
for idx in range(num_devices):
    row, col = divmod(idx, self._num_cols)
    dev_tensors = {name: per_dev[idx]
                   for name, per_dev in per_device_inputs.items()}
    merged_args = {
        "mesh_coord": (row, col),
        "mesh_shape": (self._num_rows, self._num_cols),
        "mesh_device": self._mesh_device,
        **(user_args or {}),
        **dev_overrides,
    }
    compiled = self._compile_single(self._ctx, graph, dev_tensors,
                                     per_device_outputs[idx],
                                     user_args=merged_args, ...)
```

Three keys are automatically injected into each device's args:
- `mesh_coord`: `(row, col)` tuple identifying this device in the mesh
- `mesh_shape`: `(num_rows, num_cols)` of the full mesh
- `mesh_device`: the `MeshDevice` object (needed for fabric node ID lookups)

5. **Assemble MeshProgramDescriptor.** Each device's `ProgramDescriptor` is assigned to its mesh coordinate:

```python
coord = ttnn.MeshCoordinate(row, col)
mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = compiled.program_descriptor
```

6. **Return MeshCompiledProgram.** The final result wraps the `MeshProgramDescriptor` with mesh-level I/O tensors.

### Tensor splitting: _split_tensors()

The module-level `_split_tensors()` function decomposes a dict of mesh tensors into per-device lists:

```python
def _split_tensors(tensors):
    """Split mesh tensors/views, preserving shared per-device tensor identity."""
    per_mesh = {}

    def split_mesh(t):
        if id(t) not in per_mesh:
            per_mesh[id(t)] = ttnn.get_device_tensors(t)
        return per_mesh[id(t)]

    out = {}
    for name, t in tensors.items():
        if t is None:
            continue
        if isinstance(t, _OverlappedView):
            out[name] = [dataclasses.replace(t, tensor=dt)
                         for dt in split_mesh(t.tensor)]
        else:
            out[name] = split_mesh(t)
    return out
```

Key design choices:
- **Identity-preserving cache**: `per_mesh` is keyed by `id(t)`, so if multiple `OverlappedView` objects share the same backing mesh tensor, `get_device_tensors()` is called only once and the per-device views share the same per-device tensor objects. This is critical for CB deduplication downstream (grouping by `id()`).
- **OverlappedView handling**: For views, the backing tensor is split but the view metadata (shard_shape, core_ranges, byte_offset, etc.) is preserved identically for each device.

### Per-device overrides

The `per_device_overrides` parameter lets callers customize compile-time arguments per device:

```python
result = compiler.compile(
    graph=ctx.graph,
    tensors=tensors,
    output_tensor=output,
    user_args=base_args,
    per_device_overrides=lambda row, col: {
        "is_sender": int(row == 0 and col == 0),
        "ring_index": row,
    },
)
```

Device-specific overrides are merged on top of the base `user_args`, so the override wins on conflicts.

---

## MeshCompiledProgram

`MeshCompiledProgram` (defined in `blaze/compiler.py`) is the mesh counterpart of `CompiledProgram`. It holds:

```python
class MeshCompiledProgram:
    def __init__(self, mesh_program_descriptor, io_tensors,
                 output_tensor=None, deallocate_tensors=None,
                 shadow_graph=None, lifetime_tensors=None,
                 lifetime_semaphores=None):
        self.mesh_program_descriptor = mesh_program_descriptor
        self.io_tensors = io_tensors
        self.output_tensors = _normalize_output_tensors(output_tensor)
        self._lifetime_tensors = lifetime_tensors or []
        self._lifetime_semaphores = lifetime_semaphores or []
```

Key fields:

| Field | Purpose |
|---|---|
| `mesh_program_descriptor` | `ttnn.MeshProgramDescriptor` mapping each device coordinate to its `ProgramDescriptor` |
| `io_tensors` | Mesh-level tensors (not per-device) passed to `ttnn.generic_op()` |
| `_lifetime_tensors` | Per-device internal tensors whose `Buffer*` pointers are baked into CB descriptors -- kept alive to prevent use-after-free |
| `_lifetime_semaphores` | `GlobalSemaphore` objects whose L1 addresses are baked into compile-time args |

### Lifetime pinning

The `_lifetime_tensors` and `_lifetime_semaphores` lists are critical for correctness. `ProgramDescriptor` holds raw L1 buffer pointers and semaphore addresses as integers. If the backing Python objects are garbage-collected between `compile()` and `run()`, those pointers become dangling. These lists keep the objects alive.

Both lists are assembled during compilation:

```python
program = MeshCompiledProgram(
    # ...
    lifetime_tensors=all_per_device_io_tensors
        + list(self._tensor_dict.values())
        + list(self._internal_tensor_dict.values()),
    lifetime_semaphores=list(self._sem_dict.values()) + list(self._graph_sem_pool),
)
```

### Execution

Execution is identical to single-device:

```python
def run(self):
    return _run_program(self.mesh_program_descriptor,
                        self.io_tensors,
                        self.output_tensors,
                        self._deallocate_tensors)
```

`_run_program()` calls `ttnn.generic_op(io_tensors, descriptor)`, which handles the mesh dispatch internally, sending each device's ProgramDescriptor to the correct chip. Both `ProgramDescriptor` (single device) and `MeshProgramDescriptor` (multi-device) are accepted transparently.

---

## MeshFusedProgram

For composition-driven ops that do not use `BlazeCompiler.compile()`, `MeshFusedProgram` (defined in `blaze/fused_program.py`) provides the same per-device iteration pattern:

### Construction

The constructor creates a `FusedProgram` per mesh coordinate, all sharing the same dedup dictionaries for semaphores and tensors:

```python
class MeshFusedProgram:
    def __init__(self, mesh_device, kernel, ...):
        self._sem_dict: dict = {}
        self._tensor_dict: dict = {}
        self._internal_tensor_dict: dict = {}

        mesh_shape = mesh_device.shape
        num_rows, num_cols = mesh_shape[0], mesh_shape[1]

        self._programs: dict[tuple[int, int], FusedProgram] = {}
        for row in range(num_rows):
            for col in range(num_cols):
                fp = FusedProgram(
                    kernel=kernel, device=mesh_device,
                    _sem_dict=self._sem_dict,
                    _tensor_dict=self._tensor_dict,
                    _internal_tensor_dict=self._internal_tensor_dict,
                    ...
                )
                fp.mesh_coord = (row, col)
                fp.mesh_shape = (num_rows, num_cols)
                fp.mesh_device = mesh_device
                self._programs[(row, col)] = fp
```

Key design points:

- **Shared dedup dicts**: `_sem_dict`, `_tensor_dict`, and `_internal_tensor_dict` are shared across all per-coordinate `FusedProgram` instances. When `fp.semaphore("all_reduce.sem0")` is called for device (0,0) and then again for device (0,1), both get the same `GlobalSemaphore` object. This is essential for cross-device synchronization.

- Each FusedProgram receives the `mesh_device` directly as its device (not a sub-device). MeshDevice exposes device-level APIs (grid size, NOC translation) directly.

### Composition

The `compose()` method calls the composition function once per device, optionally providing per-device extra arguments:

```python
def compose(self, compose_fn, *args, per_device_args=None, **kwargs):
    for (row, col), fp in self._programs.items():
        extra = per_device_args(row, col) if per_device_args else {}
        compose_fn(fp, *args, **{**kwargs, **extra})
```

### Building

`build()` assembles the per-device programs into a `MeshCompiledProgram`:

```python
def build(self, mesh_io_tensors, mesh_output_tensor=None, noc_mode=None):
    mesh_pd = ttnn.MeshProgramDescriptor()
    for (row, col), fp in self._programs.items():
        compiled = fp.build(noc_mode=noc_mode)
        coord = ttnn.MeshCoordinate(row, col)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = compiled.program_descriptor
    return MeshCompiledProgram(
        mesh_program_descriptor=mesh_pd,
        io_tensors=first_io_tensors if first_io_tensors is not None else mesh_io_tensors,
        output_tensor=mesh_output_tensor,
        lifetime_tensors=list(self._tensor_dict.values()) + ...,
        lifetime_semaphores=list(self._sem_dict.values()),
    )
```

### BlazeCompiler vs MeshFusedProgram

- **`BlazeCompiler`**: Takes a `BlazeGraph` (from `blaze.fuse()` context) and compiles it. Suitable for graph-API workflows.
- **`MeshFusedProgram`**: Takes a compose function directly. Suitable for direct `FusedProgram` composition without a graph.

Both share the same dedup patterns (`_sem_dict`, `_tensor_dict`, `_internal_tensor_dict`) across per-device iterations so that `f.semaphore(name)` and `f.named_tensor(name)` allocate once and reuse across devices.

---

## Mesh context on FusedProgram

Whether created by `BlazeCompiler` or `MeshFusedProgram`, every per-device `FusedProgram` has three mesh attributes set before the composition function runs:

```python
f.mesh_coord: tuple[int, int] | None  # (row, col) in the mesh
f.mesh_shape: tuple[int, int] | None  # (num_rows, num_cols)
f.mesh_device                          # the MeshDevice object
```

For single-device execution these are `None`. Ops should check for `None` or use them conditionally:

```python
if f.mesh_coord is not None:
    row, col = f.mesh_coord
    # Multi-device logic
```

---

## Global semaphore lifecycle

Mesh-global semaphores require careful lifetime management because `ttnn.generic_op()` only receives integer L1 addresses, not the `GlobalSemaphore` objects. If the Python object is garbage-collected between `compile()` and `run()`, the L1 address becomes a dangling pointer.

Both `BlazeCompiler` and `MeshFusedProgram` solve this by pinning semaphore objects on the compiled program:

```python
# In BlazeCompiler.compile()
program = MeshCompiledProgram(
    ...,
    lifetime_semaphores=list(self._sem_dict.values())
                      + list(self._graph_sem_pool),
)

# In MeshFusedProgram.build()
return MeshCompiledProgram(
    ...,
    lifetime_semaphores=list(self._sem_dict.values()),
)
```

The `FusedProgram.semaphore()` method handles allocation and deduplication:

- **Named mesh semaphores** (`f.semaphore("op.barrier")`) are deduped by name across all per-device FusedPrograms via the shared `_sem_dict`. Each call within a single op registers the semaphore on that op's program and returns its L1 address. The L1 address is the same on all devices -- this is what enables cross-device synchronization.
- **Program semaphores** (`f.semaphore(program_semaphore=True, core_ranges=...)`) are per-device, slot-based, and not address-tied. They are used for local synchronization (e.g. fabric teardown and buffer indexing).

### Auto-created global semaphores

If a `BlazeOp` class declares `num_global_semaphores`, the compiler auto-creates them before the per-device loop:

```python
if "semaphores" not in (user_args or {}):
    op_cls = BlazeOp._class_registry.get(op_name)
    num_sems = getattr(op_cls, "num_global_semaphores", 0)
    if num_sems > 0:
        gs = self._mesh_device.compute_with_storage_grid_size()
        avail = ttnn.num_cores_to_corerangeset(gs.x * gs.y, gs, row_wise=True)
        sems = [ttnn.create_global_semaphore(self._mesh_device, avail, 0)
                for _ in range(num_sems)]
        user_args = {**(user_args or {}), "semaphores": sems}
```

This lets ops declare their semaphore requirements declaratively without manual allocation.

---

## CCL ops and fabric setup

Collective communication ops (AllReduce, CclBroadcast, etc.) are the primary consumers of mesh state. They use the Tenstorrent fabric interconnect to move data between devices.

### FabricNodeId and mesh coordinate mapping

Each device in a mesh has a unique `ttnn.FabricNodeId`. Ops obtain it by calling:

```python
coord = ttnn.MeshCoordinate(row, col)
node_id = mesh_device.get_fabric_node_id(coord)
```

This ID is passed to fabric routing APIs to specify the destination of a data transfer.

### The setup_fabric() utility

`blaze/ccl.py` provides a shared `setup_fabric()` function that handles the boilerplate of fabric connection setup:

```python
def setup_fabric(
    fp: FusedProgram,
    prefix: str,
    worker_core: ttnn.CoreCoord,
    risc: Risc,
    dst_nodes: list[ttnn.FabricNodeId],
    link_indices: list[int] | None = None
):
```

What `setup_fabric()` does:

1. **For receivers** (empty `dst_nodes`): Creates an empty per-core RT args slot and returns.

2. **Allocates semaphores** -- Creates teardown and buffer-index program semaphores per connection, scoped to the worker core:

```python
worker_core_ranges = ttnn.CoreRangeSet(
    [ttnn.CoreRange(worker_core, worker_core)])
num_connections = len(dst_nodes)
teardown_sems = [
    fp.semaphore(program_semaphore=True, core_ranges=worker_core_ranges)
    for _ in range(num_connections)
]
buf_idx_sems = [
    fp.semaphore(program_semaphore=True, core_ranges=worker_core_ranges)
    for _ in range(num_connections)
]
```

3. **Computes fabric runtime args** -- Uses `ttnn.compute_fabric_connection_rt_args()` with the source node ID, destination nodes, link indices, and semaphores:

```python
coord = ttnn.MeshCoordinate(row, col)
rt_args = ttnn.compute_fabric_connection_rt_args(
    mesh_device.get_fabric_node_id(coord),
    dst_nodes, link_indices or [0], teardown_sems, buf_idx_sems,
)
```

4. **Sets per-core runtime args** -- Injects the fabric RT args onto the designated RISC (NCRISC or BRISC) for the worker core:

```python
assert risc != Risc.TRISC, "Can only connect to fabric using data movement RISCs"
ncrisc_prev_len, _, brisc_prev_len = fp._set_per_core_runtime_args(
    worker_core, **{RISC_NAMES[risc]: rt_args})
```

5. **Adds fabric kernel defines** -- Enables the fabric kernel code path:

```python
for name, val in ttnn.get_fabric_kernel_defines("Linear"):
    fp.add_define(name, val)
```

6. **Returns the RT arg base index** -- so the op can emit a `fabric_arg_idx` compile-time argument telling the kernel where to find its fabric connection data:

```python
return {Risc.NCRISC: ncrisc_prev_len, Risc.BRISC: brisc_prev_len}[risc]
```

The `risc` parameter must be `Risc.NCRISC` or `Risc.BRISC` -- fabric connections cannot be managed from TRISC (the compute RISC). Program semaphores (not global semaphores) are used because they are per-core slot-based -- scoping them to the single fabric worker core lets tt-metal reuse slots on other cores.

---

## CclBroadcast: anatomy of a CCL op

`CclBroadcast` (in `blaze/ops/ccl_broadcast/op.py`) broadcasts data from a sender device to all devices in the mesh. It demonstrates the full pattern for a CCL micro-op.

### Routing computation

The `compute_routing()` function determines each device's role from its mesh coordinate relative to the sender:

```python
@dataclass
class BroadcastRouting:
    is_sender: bool
    is_secondary_sender: bool
    is_receiver: bool
    num_fwd: int            # targets in forward direction
    num_bwd: int            # targets in backward direction
    has_secondary: int      # 0 or 1 (for 2D mesh column broadcast)
    enable_torus: bool
    ring_index: int
    num_connections: int

def compute_routing(
    row, col, sender_coord, mesh_shape, secondary_cluster_axis, is_torus,
) -> BroadcastRouting:
    sender_row, sender_col = sender_coord
    mesh_rows, mesh_cols = mesh_shape
    is_sender = (row == sender_row) and (col == sender_col)
    is_secondary_sender = (
        secondary_cluster_axis is not None
        and row == sender_row
        and col != sender_col
    )
    # ... compute forward/backward hops, torus wrapping, etc.
```

The routing supports:
- **Forward direction**: devices with higher row index than the sender.
- **Backward direction**: devices with lower row index.
- **Secondary axis**: for 2D meshes, the sender can forward along both axes (e.g., row 0 sends down column 0, then device (0,1) becomes a secondary sender and sends down column 1).
- **Torus mode**: wraps around the mesh edges for optimal hop count. When enabled, `num_fwd = (mesh_rows - 1) // 2`, `num_bwd = mesh_rows - 1 - num_fwd`.

### Destination node computation

`compute_dst_nodes()` translates the routing decisions into fabric destination node IDs:

```python
def compute_dst_nodes(row, col, routing, sender_coord,
                      mesh_shape, mesh_device) -> list:
    dst_nodes = []
    if routing.num_fwd > 0:
        fc = ttnn.MeshCoordinate((row + 1) % mesh_rows, col)
        dst_nodes.append(mesh_device.get_fabric_node_id(fc))
    if routing.num_bwd > 0:
        bc = ttnn.MeshCoordinate((row - 1 + mesh_rows) % mesh_rows, col)
        dst_nodes.append(mesh_device.get_fabric_node_id(bc))
    if routing.has_secondary:
        sc = ttnn.MeshCoordinate(row, 1 if sender_col == 0 else 0)
        dst_nodes.append(mesh_device.get_fabric_node_id(sc))
    return dst_nodes
```

> **Note:** The actual source includes explicit torus boundary overrides beyond the modular arithmetic shown above. For example, when `routing.enable_torus` and the sender is at the last row, the forward coordinate is explicitly set to row 0 rather than relying solely on `(row + 1) % mesh_rows`. The modular arithmetic produces the same result in this case, but the source uses the explicit check for clarity.

### Emit flow

`CclBroadcast.emit()` orchestrates the full broadcast setup on a single `FusedProgram`:

1. Reads `f.mesh_coord`, `f.mesh_shape`, `f.mesh_device` from the FusedProgram.
2. Computes routing via `compute_routing()`.
3. Creates or reuses the source CB from input tensor or `CBHandle`.
4. Registers the output tensor in `io_tensors`.
5. Allocates three named mesh-global semaphores (`out_sem`, `bar_sem`, `sec_sem`) for cross-device synchronization.
6. Translates the worker core to physical NOC coordinates via `mesh_device.worker_core_from_logical_core()`.
7. Emits unified CT args (page counts, hop distances, NOC coordinates).
8. Emits per-core CT args for role flags -- these are 1 only on the worker core, 0 on all other cores:

```python
f.per_core_unified_ct_args([
    (f"{prefix}.is_sender", {worker_crs: int(routing.is_sender)}),
    (f"{prefix}.is_secondary_sender", {worker_crs: int(routing.is_secondary_sender)}),
    (f"{prefix}.is_receiver", {worker_crs: int(routing.is_receiver)}),
])
```

9. Emits NCRISC runtime args (tensor addresses, semaphore addresses, ring index).
10. Sets up fabric connections via `setup_fabric()` for senders, using **NCRISC** for the fabric RISC.

---

## AllReduce: bidirectional exchange pattern

`AllReduce` (in `blaze/ops/all_reduce/op.py`) performs a neighbor-exchange reduction: each device sends its data to a neighbor and accumulates the received data.

### Core layout

Unlike `CclBroadcast` (single worker core), `AllReduce` uses two cores per device:

- **Sender core**: Reads local data and sends it over the fabric. Placed adjacent to the data core (either `data_core.x - 1` or `data_core.x + 1` if at column 0).
- **Receiver/data core**: Receives remote data and performs the reduction.

```python
data_core = ttnn.CoreCoord(input_shard_grid_start.x, input_shard_grid_start.y)
if data_core.x > 0:
    sender_core = ttnn.CoreCoord(data_core.x - 1, data_core.y)
else:
    sender_core = ttnn.CoreCoord(data_core.x + 1, data_core.y)
```

### Ring topology

The `cluster_axis` parameter (0 or 1) determines which mesh dimension the ring runs along:

```python
ring_index = row if cluster_axis == 0 else col
is_first_chip = ring_index == 0
```

Each device exchanges data with exactly one neighbor. The first chip in the ring sends forward; all others send backward:

```python
if is_first_chip:
    neighbor_row = row + 1 if cluster_axis == 0 else row
    neighbor_col = col if cluster_axis == 0 else col + 1
else:
    neighbor_row = row - 1 if cluster_axis == 0 else row
    neighbor_col = col if cluster_axis == 0 else col - 1

neighbor_mesh_coord = ttnn.MeshCoordinate(neighbor_row, neighbor_col)
```

### Semaphore assignment for deadlock avoidance

To avoid deadlock, device 0 and device 1 use semaphores in opposite order:

```python
receiver_semaphore_address = global_semaphore1_address if is_first_chip \
                             else global_semaphore0_address
sender_semaphore_address = global_semaphore0_address if is_first_chip \
                           else global_semaphore1_address
```

### Fabric link selection

AllReduce uses different link indices based on direction to avoid contention on the physical fabric links:

```python
fabric_arg_idx = setup_fabric(
    f, prefix=prefix, worker_core=fabric_core,
    risc=Risc.BRISC,
    dst_nodes=[mesh_device.get_fabric_node_id(neighbor_mesh_coord)],
    link_indices=[0 if is_first_chip else 1],
)
```

Link 0 is used for forward direction; link 1 for backward.

### Per-core role flags

Role flags gate kernel execution on the correct cores:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender_core", sender_core_range_set),
    f.flag(f"{prefix}.is_receiver_core", receiver_core_range_set),
])
```

### RISC-specific CT args

AllReduce distributes work across all three RISC processors:

- **NCRISC** (reader): Sender tile counts, page sizes, CB handles for sender and receiver, semaphore addresses, NOC coordinates for the data core, and the tensor buffer address for data readback.
- **TRISC** (compute): Receiver tile counts, CB handles for input/intermediate/output, residual flags.
- **BRISC** (writer): Sender tile counts, page sizes, CB handles, NOC coordinates for data transfer and semaphore signaling, fabric barrier addresses.

This separation reflects the Blackhole RISC architecture: NCRISC handles data movement reads, BRISC handles data movement writes, and TRISC handles compute. Each RISC gets only the CT args it needs.

### Key differences from CclBroadcast

| Aspect | CclBroadcast | AllReduce |
|---|---|---|
| Cores per device | 1 (worker core) | 2 (sender + receiver/data) |
| Fabric RISC | NCRISC | BRISC |
| Direction | Unidirectional (sender to all) | Bidirectional (neighbor exchange) |
| Link indices | Default `[0]` | Direction-dependent (`[0]` or `[1]`) |
| Semaphore ordering | Same order on all devices | Reversed between neighbors (deadlock avoidance) |

---

## Summary of multi-device compilation flow

```
MeshDevice
    |
    v
BlazeCompiler(mesh_device)
    |
    |-- DeviceContext.from_device()  (shared across all devices)
    |
    v
compile(graph, tensors, output_tensor)
    |
    |-- CBEngine + SemEngine       (device-agnostic, computed once)
    |
    |-- _split_tensors()           (mesh tensors -> per-device lists)
    |
    |-- for each (row, col):
    |       |
    |       |-- inject mesh_coord, mesh_shape, mesh_device
    |       |-- merge per_device_overrides
    |       |-- _compile_single()
    |       |       |
    |       |       |-- fused-op path: FusedProgram.compose() + build()
    |       |       |-- engine path:   CBEngine + SemEngine + CTArgEngine
    |       |       |
    |       |       v
    |       |   CompiledProgram (per-device ProgramDescriptor)
    |       |
    |       v
    |   mesh_pd[MeshCoordinate(r,c)] = program_descriptor
    |
    v
MeshCompiledProgram
    |
    v
.run()  -->  ttnn.generic_op(mesh_io_tensors, mesh_program_descriptor)
```

---

## Writing a new CCL op: checklist

When writing a multi-device collective communication op, follow this checklist:

1. **Read mesh context from `FusedProgram`**: Use `f.mesh_coord`, `f.mesh_shape`, `f.mesh_device`. These are set by `BlazeCompiler` or `MeshFusedProgram` before your `emit()` runs.

2. **Compute per-device routing from mesh coordinates**: Determine this device's role (sender, receiver, forwarder) and its destination node IDs. All routing should be relative to `f.mesh_coord` and `f.mesh_shape` -- an op that works on T3K should also work on Galaxy without code changes.

3. **Translate logical cores to NOC coordinates** via `mesh_device.worker_core_from_logical_core(core)`. Never hardcode physical coordinates.

4. **Get fabric node IDs** via `mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col))`.

5. **Allocate semaphores**:
   - Use `f.semaphore(f"{prefix}.name")` for mesh-global cross-device synchronization (deduped by name in `_sem_dict`).
   - Use `f.semaphore(program_semaphore=True, core_ranges=...)` for per-device local sync (fabric teardown, buffer indexing).

6. **Call `setup_fabric()`** with the worker core, RISC assignment (NCRISC or BRISC, never TRISC), and destination nodes. Store the returned arg index as a per-core CT arg so the kernel knows where fabric RT args start.

7. **Set per-core role flags** using `f.per_core_unified_ct_args()` and `f.flag()` so the unified kernel only runs CCL logic on the designated worker cores. Cores not covered by the flag dict entries default to 0.

8. **Distribute CT args to the correct RISC** using `f.ncrisc_ct_args()`, `f.brisc_ct_args()`, and `f.trisc_ct_args()`. NCRISC handles reads, BRISC handles writes, TRISC handles compute.

9. **Never hardcode mesh positions or device counts**: All routing should be derived from `f.mesh_coord` and `f.mesh_shape`. An op that works on a 2-device pair should also work on an 8-device T3K or 32-device Galaxy without code changes.

---

## Key takeaways

1. **One DeviceContext, many devices.** Tenstorrent meshes are homogeneous, so the compiler creates a single `DeviceContext` and reuses it for all coordinates.

2. **Per-device compilation with mesh-level execution.** Each device gets its own `ProgramDescriptor` (allowing different CT args, fabric routing, role flags), but `ttnn.generic_op()` receives mesh-level tensors and dispatches internally.

3. **mesh_coord, mesh_shape, mesh_device are auto-injected.** Op authors access them via `f.mesh_coord`, `f.mesh_shape`, and `f.mesh_device` -- no manual plumbing needed.

4. **Semaphore and tensor dedup is cross-device.** The shared `_sem_dict` and `_tensor_dict` ensure that `f.semaphore("name")` returns the same object across all per-device iterations, and lifetime is managed by `MeshCompiledProgram`.

5. **setup_fabric() is the standard CCL entry point.** It handles semaphore allocation, RT arg computation, per-core injection, and kernel define emission. CCL ops only need to compute their routing (destination nodes, link indices) and call `setup_fabric()`.

6. **Fabric connections use data-movement RISCs only.** NCRISC or BRISC, never TRISC. The choice depends on which RISC the op uses for its data transfer path (reader vs. writer).

| Concept | Source | Role |
|---|---|---|
| `BlazeCompiler` | `blaze/compiler.py` | Compiles a `BlazeGraph` for mesh execution |
| `BlazeCompiler.compile()` | Same | Per-device loop, produces `MeshCompiledProgram` |
| `MeshCompiledProgram` | Same | Wraps `MeshProgramDescriptor` + IO tensors + lifetime pins |
| `MeshFusedProgram` | `blaze/fused_program.py` | Composition-driven mesh compilation |
| `_split_tensors()` | `blaze/compiler.py` | Decomposes mesh tensors to per-device with identity caching |
| `_EngineResults` | Same | Pre-computed CB/semaphore assignments (device-agnostic) |
| `setup_fabric()` | `blaze/ccl.py` | Allocates semaphores, computes RT args, adds fabric defines |
| `FabricNodeId` | `ttnn` | Unique fabric address per device |
| `mesh_coord` / `mesh_shape` / `mesh_device` | Set on `FusedProgram` | Per-device mesh context |
| `CclBroadcast` | `blaze/ops/ccl_broadcast/op.py` | Reference CCL op: unidirectional broadcast |
| `AllReduce` | `blaze/ops/all_reduce/op.py` | Reference CCL op: bidirectional exchange + reduction |
| `BroadcastRouting` | `blaze/ops/ccl_broadcast/op.py` | Routing state dataclass for broadcast |
| `compute_routing()` | Same | Computes per-device routing from mesh coordinates |
| `compute_dst_nodes()` | Same | Converts routing to `FabricNodeId` list |
