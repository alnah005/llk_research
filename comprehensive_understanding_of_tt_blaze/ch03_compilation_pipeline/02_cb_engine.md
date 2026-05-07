# 02 -- CB Engine

The CBEngine is the first engine in the compilation pipeline. It walks a `BlazeGraph` in topological order and assigns sequential circular buffer IDs to every data path in the graph -- external tensor inputs, internal scratch buffers, inter-op edges, and terminal outputs. It also computes buffer sizes, tracks lifetimes, and optionally compacts CB IDs via interval coloring to stay within the Blackhole hardware limit of 64 CBs.

**Source file:** `blaze/cb_engine.py`

---

## 1. Overview

On Blackhole, each Tensix core has a fixed pool of 64 circular buffer slots (indices 0--63). Every data path through a fused kernel -- each input tensor, each intermediate result passed between ops, each output tensor, and each scratch buffer -- requires its own CB slot. For long pipelines with many ops (10+ ops in a gated MLP, for example), this 64-CB budget becomes a real constraint. The CBEngine's job is to assign these slots automatically from the graph structure, eliminating the manual `cb0`, `cb1`, ... wiring that dominated the composition API's predecessor.

### 1.1 Constructor

```python
class CBEngine:
    def __init__(self, max_cb_id=64, compact=False, cb_id_manager=None):
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `max_cb_id` | `64` | Hardware CB limit (`MAX_CB_ID`). The engine warns when approaching this. |
| `compact` | `False` | When `True`, applies interval-coloring compaction after assignment. |
| `cb_id_manager` | `None` | When provided, delegates ID allocation to a `CircularBufferIdManager` for format-aware reuse. |

---

## 2. The Four CB Categories

The engine produces a `dict[str, CBAssignment]` where each key follows a naming convention that encodes the assignment's category and origin:

| Key pattern | Category | `is_tensor_backed` | Description |
|-------------|----------|---------------------|-------------|
| `ext_in_{node_id}_{port}` | `ext_in` | `True` | External tensor input. The CB wraps a shard of a tensor supplied at compile time. |
| `internal_{node_id}_{port}` | `internal` | `False` | Op-internal scratch buffer declared via `InternalCB` in the `OpSpec`. Not a graph edge. |
| `intermed_{node_id}_{port}` | `intermed` | `False` | Inter-op edge. Connects a producer's output to one or more consumers. Lives entirely in L1. |
| `ext_out_{node_id}_{port}` | `ext_out` | `True` | Terminal output. The CB wraps a shard of the pre-allocated output tensor. |

---

## 3. The Assignment Algorithm

### 3.1 Topological Walk

The engine processes nodes in topological order (Kahn's algorithm, see [Section 01](./01_blaze_graph_and_fusion_context.md)). For each node, it handles three phases in sequence:

**Phase 1 -- Input ports**: For each input port on the node, check if it's fed by an edge (skip -- the producer already assigned its CB) or by an external tensor (assign `ext_in`). Ports in `graph.external_input_ports` that are not already covered by an incoming edge receive a new `ext_in` allocation.

**Phase 2 -- Internal CBs**: For each `InternalCB` in `node.spec.internal_cbs`, assign an `internal` CB unless the port name is already covered. Each `InternalCB` specifies:

```python
@dataclass
class InternalCB:
    name: str                    # port-like name, e.g. "intermed"
    num_pages: int = 1           # number of buffer pages
    page_size_source: str = "input"  # "input" or explicit shape like "1x32"
```

**Phase 3 -- Output ports**: For each output port, check if it has outgoing edges (assign `intermed`) or not (assign `ext_out`).

### 3.2 ID Allocation

CB IDs are allocated sequentially starting from 0:

```python
def _alloc(dtype, tile_shape) -> int:
    nonlocal next_cb_id
    if mgr_context is not None:
        # Delegate to CircularBufferIdManager for format-aware reuse
        ttnn_dtype = _get_ttnn_dtype(dtype)
        tile_desc = _get_cached_tile(tile_cache, tile_shape[0], tile_shape[1])
        cb_id = mgr_context.get_cb_id(ttnn_dtype, tile_desc)
        next_cb_id = max(next_cb_id, cb_id + 1)
        return cb_id
    cb_id = next_cb_id
    next_cb_id += 1
    return cb_id
```

When a `CircularBufferIdManager` is provided (via the constructor), the engine delegates allocation to the manager, which can reuse CB IDs across different data formats and tiles. Without a manager, IDs are strictly sequential.

### 3.3 Page Size Computation

Each CB's page size is computed from the port's dtype and tile shape:

$$\text{page\_size} = \text{DTYPE\_BYTES}[\text{dtype}] \times \text{tile\_h} \times \text{tile\_w}$$

Where `DTYPE_BYTES` maps data types to byte sizes:

| dtype | bytes |
|-------|-------|
| `bfloat16` | 2 |
| `float16` | 2 |
| `float32` | 4 |
| `uint8` | 1 |
| `uint16` | 2 |
| `uint32` | 4 |

Defaults are `bfloat16` and `(32, 32)`, yielding a page size of $2 \times 32 \times 32 = 2048$ bytes.

The number of pages comes from `node.kwargs.get(f"num_tiles_{port_name}", 1)`, allowing user control over double-buffering.

For interop with `ttnn`, an expanded dtype set is supported via lazy-loaded conversion: `bfloat8_b` and `bfloat4_b` are available for `_get_ttnn_dtype()` (used by `_alloc()` with a manager and by `to_cb_id_manager()`).

---

## 4. Fan-Out Handling

When a single output port feeds multiple consumers (fan-out), the engine assigns **one** CB ID shared across all edges from that port:

```python
if not port_edges:
    # Terminal output -- ext_out
    ...
else:
    # Inter-op edge(s) -- possibly fan-out
    all_grids = [node.grid] + [e.consumer.grid for e in port_edges]
    core_ranges = _union_grids(all_grids)
    cb_id = _alloc(dtype, tile_shape)
    # Set cb_id on ALL edges from this port
    for e in port_edges:
        e.cb_id = cb_id
```

The `core_ranges` for the shared CB is the union of the producer's grid and all consumer grids:

$$\text{core\_ranges} = \text{grid}_\text{producer} \cup \text{grid}_{\text{consumer}_1} \cup \text{grid}_{\text{consumer}_2} \cup \cdots$$

The `_union_grids()` helper handles various grid representations:

- `CoreRangeSet` objects: merged via the `|` operator.
- List-of-tuple grids: set union of coordinates.
- Identical grids: returned as-is.
- Mixed types: returned as a `frozenset`.

---

## 5. Shared Tensor Optimization

When multiple nodes consume the same `ExternalTensor` through different ports, they share a single CB ID rather than each getting its own allocation. This is tracked via `shared_tensor_cb`:

```python
port_to_tensor: dict[tuple[str, str], str] = {}
shared_tensor_cb: dict[str, int] = {}

# During ext_in assignment:
tensor_name = port_to_tensor.get((node.id, port_spec.name))
if tensor_name and tensor_name in shared_tensor_cb:
    cb_id = shared_tensor_cb[tensor_name]  # reuse existing
else:
    cb_id = _alloc(dtype, tile_shape)
    if tensor_name:
        shared_tensor_cb[tensor_name] = cb_id
```

The `shared_tensor_cb` dict is keyed by `ExternalTensor.name`, which is node-agnostic -- the optimization works across different nodes, not just within a single op. This is critical for ops like `gated_reduce` where two ports may share the same activation tensor, and for pipeline structures where the same weight tensor feeds multiple stages.

---

## 6. CBAssignment

Each assignment carries comprehensive metadata:

```python
@dataclass
class CBAssignment:
    cb_id: int              # 0--63
    total_size: int         # bytes (num_pages * page_size)
    page_size: int          # bytes per page
    num_pages: int          # number of buffer pages
    core_ranges: Any        # which cores need this CB
    is_tensor_backed: bool  # external tensor vs intermediate
    data_format: str        # e.g. "bfloat16"
    tile_shape: tuple[int, int]  # e.g. (32, 32)
    node_id: str            # owning graph node
    port_name: str          # port on that node
    category: str           # "ext_in" | "internal" | "intermed" | "ext_out"
    first_use: int          # topological position of first reader/writer
    last_use: int           # topological position of last reader/writer
```

The `first_use` and `last_use` fields track the CB's lifetime in topological space. These are computed after the main assignment loop:

```python
topo_pos = {n.id: i for i, n in enumerate(topo_order)}
for key, cb in assignments.items():
    if cb.node_id:
        cb.first_use = topo_pos.get(cb.node_id, 0)
        cb.last_use = topo_pos.get(cb.node_id, 0)

# Extend last_use for inter-op edges
for edge in graph.edges:
    if edge.cb_id is not None:
        for key, cb in assignments.items():
            if cb.cb_id == edge.cb_id and cb.category == "intermed":
                consumer_pos = topo_pos.get(edge.consumer.id, 0)
                cb.last_use = max(cb.last_use, consumer_pos)
```

---

## 7. Lifetime Tracking and Interval Coloring Compaction

### 7.1 The 64-CB Limit

For long pipelines with many ops, sequential allocation can exhaust the 64 CB slots. The engine issues a warning when usage approaches the limit:

```python
if next_cb_id > self.max_cb_id - 8:
    warnings.warn(f"CB usage is high: {next_cb_id}/{self.max_cb_id} CBs assigned.")
```

### 7.2 `compact_cb_ids()` -- Interval Coloring

When `compact=True` is passed to the CBEngine constructor, the `compact_cb_ids()` method recolors CB IDs using interval graph coloring. Two CBs with non-overlapping lifetimes $[\text{first\_use}_i, \text{last\_use}_i]$ can share the same physical CB slot:

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

The algorithm is a greedy interval coloring: for each CB (sorted by `first_use`), find the lowest-numbered color whose previous occupant has already completed. This is optimal for interval graphs -- the chromatic number equals the maximum clique size, which corresponds to the maximum number of simultaneously live CBs.

For a pipeline of $n$ ops where each intermediate is consumed by the next op, the maximum simultaneous liveness is typically 2--3 (the current op's input, output, and possibly scratch), so compaction can reduce a 20-CB pipeline to under 5 physical slots.

---

## 8. `to_cb_id_manager()` Export

For interop with downstream code that needs to allocate additional CBs without ID collisions (e.g., `MoeOp`), the engine can export its assignments to a `CircularBufferIdManager`:

```python
def to_cb_id_manager(self, assignments) -> CircularBufferIdManager:
    manager = CircularBufferIdManager()
    for key, a in sorted(assignments.items(), key=lambda x: x[1].cb_id):
        ttnn_dtype = DTYPE_TO_TTNN[a.data_format]
        tile_desc = ttnn.TileDescriptor(a.tile_shape[0], a.tile_shape[1], False)
        manager.seed_reserved_cb(a.cb_id, ttnn_dtype, tile_desc)
    return manager
```

The `seed_reserved_cb()` call registers each engine-assigned CB ID with its format, so subsequent `manager.create_context().get_cb_id()` calls will avoid collisions. The `CircularBufferIdManager` (from `blaze/cb_reconfig.py`) can also reuse IDs across contexts when the data format and tile match -- a key optimization for multi-phase fused programs.

---

## 9. Summary: CBEngine Data Flow

```
BlazeGraph
    |
    v
topological_order()
    |
    v
For each node in topo order:
    ext_in ports -> CBAssignment (is_tensor_backed=True)
    internal CBs -> CBAssignment (is_tensor_backed=False)
    output ports:
        no edges -> ext_out CBAssignment (is_tensor_backed=True)
        has edges -> intermed CBAssignment (is_tensor_backed=False)
                     edge.cb_id set for all fan-out edges
    |
    v
Compute lifetimes (first_use, last_use)
    |
    v
Optional: compact_cb_ids() via interval coloring
    |
    v
dict[str, CBAssignment]  -->  CTArgEngine (source="cb")
                          -->  BlazeCompiler (CBDescriptor construction)
```

---

<- [01 -- BlazeGraph and Fusion Context](./01_blaze_graph_and_fusion_context.md) | [Index](index.md) | [03 -- Sem Engine](./03_sem_engine.md) ->
