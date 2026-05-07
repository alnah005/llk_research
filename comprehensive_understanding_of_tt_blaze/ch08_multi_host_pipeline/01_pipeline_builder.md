# 08 -- Pipeline Builder: Graph-Based Multi-Host Pipeline Layout

[Next: Pipeline Manager Architecture >>](02_pipeline_manager_architecture.md)

---

## Overview

The `pipeline_builder/` Python package replaces hardcoded physical topology tables with a declarative graph that describes pipeline stages, their submesh shapes, and their directed connections. At runtime, the builder partitions a shared parent `MeshDevice` into submeshes, resolves physical connectivity, and emits a `PipelineLayout` that the downstream `PipelineBlock` infrastructure (covered in earlier chapters) consumes without modification.

The package exports five public symbols:

```python
# pipeline_builder/__init__.py
from pipeline_builder.graph import Edge, Node, PipelineGraph
from pipeline_builder.submesh_partition import PipelineLayout, SubmeshPartition
from pipeline_builder.visualize import serialize_layout, write_json
```

---

## Node and Edge Primitives

### Node

A `Node` represents one pipeline stage. It carries the submesh shape required for that stage, an optional factory callable that creates the stage instance from a submesh, and a display name used by the visualizer.

```python
@dataclass
class Node:
    shape: Any                          # ttnn.MeshShape, e.g. MeshShape(4, 2)
    factory: Optional[Callable] = None  # factory(submesh) -> stage instance
    name: str = ""                      # display label for visualization
```

The `factory` field binds model-level stage construction directly to the graph definition. When all nodes carry factories, `PipelineLayout.build_pipeline()` can construct the per-rank `Pipeline` object without external wiring.

### Edge

An `Edge` is a directed connection between two nodes identified by name. Edges come in two flavors:

1. **Regular edges** -- forward connections from one stage to the next.
2. **Loopback edges** -- the return path from the last stage back to the first, carrying the decode token for re-injection.

```python
@dataclass
class Edge:
    src: str
    dst: str
    src_exit_chip: Optional[int] = None   # explicit path only
    dst_entry_chip: Optional[int] = None  # explicit path only
    is_loopback: bool = False
```

For the auto-discovery build path (`build_topology`), the chip ID fields are left as `None` -- physical coordinates are resolved at runtime from the fabric routing tables. For the legacy explicit path (`build`), the caller supplies global chip IDs at each ethernet boundary.

---

## PipelineGraph

`PipelineGraph` is the central class. It holds an ordered dict of nodes and a list of edges, validates that all edge endpoints reference valid nodes, and provides two build paths to produce a `PipelineLayout`.

```python
class PipelineGraph:
    def __init__(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
        h2d_entry_chip: Optional[int] = None,  # explicit path only
        d2h_exit_chip: Optional[int] = None,    # explicit path only
    ) -> None:
```

The `h2d_entry_chip` and `d2h_exit_chip` parameters are only used by the legacy build path to identify the host-to-device and device-to-host socket endpoints in the first stage's submesh.

### Validation

On construction, `_validate()` asserts that every edge's `src` and `dst` exist in the `nodes` dict. This catches typos and dangling references before any hardware interaction.

### Edge Classification

Internal helpers partition edges:

- `_regular_edges()` -- all edges with `is_loopback=False`.
- `_loopback_edges()` -- all edges with `is_loopback=True`.
- `_outgoing(node)` / `_incoming(node)` -- regular edges filtered by source or destination.

---

## Stage Ordering via Kahn's Algorithm

The `stage_order()` method computes the pipeline execution order by running a topological sort (Kahn's algorithm) on the non-loopback edges. Loopback edges are excluded because they form a cycle (last stage to first stage) that would make topological sorting impossible.

```python
def stage_order(self) -> list[str]:
    in_degree: dict[str, int] = {name: 0 for name in self.nodes}
    for edge in self._regular_edges():
        in_degree[edge.dst] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for edge in self._outgoing(node):
            in_degree[edge.dst] -= 1
            if in_degree[edge.dst] == 0:
                queue.append(edge.dst)

    assert len(order) == len(self.nodes), (
        "Cycle detected in non-loopback edges; pipeline graph must be a DAG"
    )
    return order
```

The algorithm proceeds as follows:

1. Compute in-degree for each node considering only regular edges.
2. Seed the queue with nodes that have zero in-degree (no incoming regular edges).
3. Repeatedly dequeue a node, append it to the order, and decrement in-degrees of its outgoing neighbors. When a neighbor reaches zero in-degree, enqueue it.
4. Assert that the resulting order covers all nodes -- if not, a cycle exists in the regular edges, which is a graph definition error.

For a typical four-stage linear pipeline (`f0 -> f1 -> f2 -> f3` with `f3 -> f0` loopback), the result is `["f0", "f1", "f2", "f3"]`.

---

## PipelineConfigEntry

The output of both build paths is a list of `PipelineConfigEntry` values imported from the `PipelineBlock` op module:

```python
from models.demos.deepseek_v3_b1.micro_ops.pipeline_block.op import PipelineConfigEntry, StageMetadata
```

Each entry carries two `MeshCoordinate` values:

| Field | Meaning |
|-------|---------|
| `entry_node_coord` | Chip in this stage's submesh that receives data from the previous stage (or H2D for stage 0) |
| `exit_node_coord` | Chip in this stage's submesh that sends data to the next stage |

The list has `num_stages + 1` entries. Entries `[0..N-1]` describe the forward path. Entry `[N]` describes the loopback:

- `pipeline_config[N].entry_node_coord` -- chip in stage 0's submesh that receives the loopback return.
- `pipeline_config[N].exit_node_coord` -- chip in stage 0's submesh used by the D2H socket.

---

## Build Path 1: Explicit Chip IDs (Legacy)

The `build()` method is the legacy path where the caller supplies:

- A `SubmeshPartition` (or any object providing `chip_to_coord_map()` and `build_layout()`).
- An `ordering` list mapping `ordering[stage_idx] = submesh_idx`.
- A `mesh_id` (0 for single-galaxy).

Edges must have `src_exit_chip` and `dst_entry_chip` set to global chip IDs. The graph must also have `h2d_entry_chip` and `d2h_exit_chip` set.

```python
def build(self, partition, ordering, mesh_id=0):
    chip_to_coord = partition.chip_to_coord_map()
    stage_names = self.stage_order()
    regular_edges = self._regular_edges()
    loopback_edges = self._loopback_edges()

    pipeline_config = []
    for i, name in enumerate(stage_names):
        if i == 0:
            entry_chip = self.h2d_entry_chip
        else:
            in_edge = next(e for e in regular_edges if e.dst == name)
            entry_chip = in_edge.dst_entry_chip

        out_edges = [e for e in regular_edges if e.src == name]
        if out_edges:
            exit_chip = out_edges[0].src_exit_chip
        else:
            lb = next(e for e in loopback_edges if e.src == name)
            exit_chip = lb.src_exit_chip

        pipeline_config.append(PipelineConfigEntry(
            chip_to_coord[entry_chip],
            chip_to_coord[exit_chip],
        ))

    # Loopback entry
    for lb in loopback_edges:
        pipeline_config.append(PipelineConfigEntry(
            chip_to_coord[lb.dst_entry_chip],
            chip_to_coord[self.d2h_exit_chip],
        ))

    return partition.build_layout(
        ordering=ordering,
        pipeline_config=pipeline_config,
        mesh_id=mesh_id,
    )
```

Example usage with a 4-stage pipeline on an 8x4 galaxy (four 4x2 submeshes):

```python
graph = PipelineGraph(
    nodes={f"f{i}": Node(shape=ttnn.MeshShape(4, 2)) for i in range(4)},
    edges=[
        Edge("f0", "f1", src_exit_chip=5,  dst_entry_chip=6),
        Edge("f1", "f2", src_exit_chip=15, dst_entry_chip=19),
        Edge("f2", "f3", src_exit_chip=22, dst_entry_chip=21),
        Edge("f3", "f0", src_exit_chip=16, dst_entry_chip=12, is_loopback=True),
    ],
    h2d_entry_chip=4,
    d2h_exit_chip=8,
)
partition = SubmeshPartition(mesh_device, ttnn.MeshShape(4, 2))
layout = graph.build(partition, ordering=[0, 1, 3, 2])
```

---

## Build Path 2: Auto-Discovery via Topology (build_topology)

The `build_topology()` method eliminates hardcoded chip IDs entirely. It takes only a `MeshDevice` and a `mesh_id`, then:

1. **Validates uniform shape** -- all nodes must have the same `MeshShape` (a current limitation; heterogeneous shapes are a design goal but not yet implemented in the auto-discovery path).

2. **Creates submeshes** -- calls `mesh_device.create_submeshes(submesh_shape)` and asserts the count matches the number of nodes.

3. **Detects the local submesh** -- iterates submeshes to find which one is physically present on this MPI rank by checking `sub.get_view().is_local(MeshCoordinate(0, 0))`.

4. **Collects per-submesh chip info** -- for each submesh, enumerates all `(row, col)` positions, extracts `FabricNodeId` (mesh_id, chip_id), and builds the `submesh_chips` structure needed by the C++ resolver.

5. **Calls the C++ topology resolver**:

```python
result = ttnn._ttnn.multi_device.experimental.resolve_graph_layout(
    edges=edge_tuples,
    submesh_chips=submesh_chips,
)
```

This C++ function (`resolve_graph_layout`) uses the control-plane APIs (`get_forwarding_direction`, `get_chip_neighbors`) to discover which physical ethernet links connect each pair of submeshes. It performs backtracking to assign submeshes to nodes such that every edge in the graph corresponds to a real physical connection.

6. **Builds pipeline_config** -- extracts entry/exit coordinates from the resolved edges.

7. **Derives stage ordering and metadata** -- maps node names to submesh indices via `result.node_to_submesh`, uses `distributed_context_allgather_int` to discover which MPI rank owns each stage, and builds `StageMetadata` with rank and mesh_id.

8. **Collects factories and labels** -- extracts `Node.factory` and `Node.name` for each stage.

9. **Returns `PipelineLayout`** -- the fully resolved per-rank layout.

Example usage:

```python
graph = PipelineGraph(
    nodes={
        "f0": Node(shape=ttnn.MeshShape(4, 2)),
        "f1": Node(shape=ttnn.MeshShape(4, 2)),
        "f2": Node(shape=ttnn.MeshShape(4, 2)),
        "f3": Node(shape=ttnn.MeshShape(4, 2)),
    },
    edges=[
        Edge("f0", "f1"),
        Edge("f1", "f2"),
        Edge("f2", "f3"),
        Edge("f3", "f0", is_loopback=True),
    ],
)
layout = graph.build_topology(mesh_device)
```

No chip IDs, no ordering list, no partition object. The topology resolver handles everything.

---

## SubmeshPartition

`SubmeshPartition` is a standalone class that partitions a parent `MeshDevice` into uniform submeshes and resolves which one is local to the current MPI rank. It is used by the legacy `build()` path but is not needed for `build_topology()` (which creates submeshes internally).

```python
class SubmeshPartition:
    def __init__(self, mesh_device, submesh_shape):
        self.mesh_device = mesh_device
        self.submesh_shape = submesh_shape
        self.submeshes = mesh_device.create_submeshes(submesh_shape)
        self._local_submesh_idx = None
        for i, sub in enumerate(self.submeshes):
            if sub.get_view().is_local(ttnn.MeshCoordinate(0, 0)):
                self._local_submesh_idx = i
                break
        assert self._local_submesh_idx is not None
```

### chip_to_coord_map

Builds a `{chip_id -> MeshCoordinate}` mapping across every device in every submesh. Each submesh uses 0-based local coordinates; `chip_id` (globally unique within a fabric mesh) is the key:

```python
def chip_to_coord_map(self) -> dict[int, Any]:
    mapping = {}
    for sub in self.submeshes:
        for row in range(sub.shape[0]):
            for col in range(sub.shape[1]):
                coord = ttnn.MeshCoordinate(row, col)
                fid = sub.get_fabric_node_id(coord)
                mapping[fid.chip_id] = coord
    return mapping
```

### build_layout

Derives a `PipelineLayout` from a caller-supplied `ordering` and `pipeline_config`:

```python
def build_layout(self, *, ordering, pipeline_config, mesh_id=0):
    submesh_to_stage = {sub_idx: stage_idx
                        for stage_idx, sub_idx in enumerate(ordering)}
    my_stage_idx = submesh_to_stage[self._local_submesh_idx]

    all_stage_idxs = ttnn.distributed_context_allgather_int(my_stage_idx)
    stage_to_rank = {stage: rank for rank, stage in enumerate(all_stage_idxs)}
    stages_metadata = {
        stage_idx: StageMetadata(rank=stage_to_rank[stage_idx], mesh_id=mesh_id)
        for stage_idx in range(len(ordering))
    }
    return PipelineLayout(
        my_stage_idx=my_stage_idx,
        my_submesh=self.submeshes[self._local_submesh_idx],
        my_submesh_idx=self._local_submesh_idx,
        submesh_to_stage=submesh_to_stage,
        stages_metadata=stages_metadata,
        pipeline_config=pipeline_config,
        submeshes=self.submeshes,
    )
```

The `ordering` parameter is a list where `ordering[stage_idx] = submesh_idx`. For example, `[0, 1, 3, 2]` means stage 0 runs on submesh 0, stage 1 on submesh 1, stage 2 on submesh 3, stage 3 on submesh 2. The allgather call resolves the mapping between stage indices and MPI ranks, which is not necessarily identity when rank binding permutes device assignments.

---

## PipelineLayout

`PipelineLayout` is the result dataclass returned by both build paths. It bundles everything a rank needs to construct and run its pipeline stage:

```python
@dataclass
class PipelineLayout:
    my_stage_idx: int                          # which stage this rank runs
    my_submesh: Any                            # ttnn.MeshDevice (submesh)
    my_submesh_idx: int                        # index into submeshes[]
    submesh_to_stage: dict                     # submesh_idx -> stage_idx
    stages_metadata: dict                      # stage_idx -> StageMetadata
    pipeline_config: list                      # list[PipelineConfigEntry]
    submeshes: list                            # all submeshes
    stage_factories: Optional[dict] = None     # stage_idx -> factory
    mesh_shape: Optional[tuple] = None         # parent (rows, cols)
    stage_labels: dict = field(default_factory=dict)  # stage_idx -> name
```

### build_pipeline

When all graph nodes carry `factory` callables, `PipelineLayout.build_pipeline()` constructs the per-rank `Pipeline` object:

```python
def build_pipeline(self):
    factory = self.stage_factories[self.my_stage_idx]
    stage = factory(self.my_submesh)
    return Pipeline(
        self.my_submesh,
        stage,
        self.my_stage_idx,
        stages_metadata=self.stages_metadata,
        pipeline_config=self.pipeline_config,
    )
```

### Diagnostic Methods

- `print_my_submesh()` -- prints this rank's MPI rank, submesh index, stage index, and a grid of `FabricNodeId` values.
- `print_submeshes()` -- prints all submeshes with stage assignments (rank 0 only).

---

## Visualization

The `visualize.py` module serializes a `PipelineLayout` into JSON for the blaze observability viewer (`visualizer/index.html`).

### serialize_layout

Extracts all visualization data from a layout:

1. Iterates all submeshes and builds `chip_records` with chip_id, mesh_id, submesh_idx, stage_idx, local coordinates, and an empty roles list.
2. Annotates entry/exit roles from `pipeline_config`: `h2d_entry`, `d2d_entry`, `d2d_exit` for forward stages, plus `lb_d2d_entry` and `d2h_exit` for the loopback path.
3. Builds `stages` records with entry/exit chip IDs, rank, mesh_id, and display name.
4. Returns a dict with `num_stages`, `has_loopback`, `mesh_rows`, `mesh_cols`, `chips`, and `stages`.

### write_json

Calls `serialize_layout` and writes the result as indented JSON:

```python
def write_json(layout, path):
    data = serialize_layout(layout)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
```

### BLAZE_EXPORT Integration

When `BLAZE_EXPORT=1` is set, `PipelineLayout.write_json()` also writes `pipeline_config.json` to the blaze export directory (default `generated/viz`), allowing the compiler to embed pipeline topology data in the op visualization JSON.

---

## Loopback Edges

Loopback edges are central to the pipeline design. In a circular pipeline, the last stage sends the sampled decode token back to the first stage for re-injection. The loopback edge:

- Is excluded from Kahn's algorithm to avoid cycle detection failures.
- Contributes an extra entry at `pipeline_config[num_stages]` with the loopback entry coordinate and the D2H socket coordinate in stage 0's submesh.
- In auto-discovery mode, is resolved like any regular edge but with special handling in the resolver to ensure the physical return path exists.

Without a loopback edge, the pipeline would be open-ended -- tokens enter at stage 0 and exit at the last stage with no return path. This is the expected topology for a complete pipeline: tokens flow forward through model layers, and the decode output returns for the next generation step.

---

## Integration with blaze/stages/

The `pipeline_builder` package produces `PipelineLayout` values that feed directly into the stage infrastructure in `blaze/stages/stage_io.py`. The key integration points:

### StageKind and StageContext

`StageKind` is the abstract base class for pipeline stages (covered in earlier chapters). The `StageContext` bundles the arguments that `StageKind` methods receive:

```python
@dataclass
class StageContext:
    mesh_device: ttnn.MeshDevice
    pipeline_config: list          # from PipelineLayout.pipeline_config
    my_mesh_id: int
```

### Concrete Stage Kinds

- `EmbeddingStage` -- stage 0: receives tokens via H2D, performs embedding lookup, forwards activations. The `PipelineBlock` it creates uses `TOKEN_PAGE_SIZE_BYTES` (64) for H2D/D2H and `ACTIVATION_PAGE_SIZE_BYTES` (7168 * 2 = 14336) for the downstream D2D socket.
- `LMHeadStage` -- the last compute stage: receives activations, runs LM head + sampling, sends the sampled token back via the loopback path.
- `PassthroughStage` -- a no-compute stage that simply forwards data through the pipeline. Used for stages that do not own compute but must participate in the pipeline topology.

### Constants

The stage module defines constants that the `PipelineBlock` and pipeline manager both depend on:

```python
TOKEN_PAGE_SIZE_BYTES = 64
TOKEN_FIFO_SIZE = 1024
ACTIVATION_DIM = 7168
ACTIVATION_PAGE_SIZE_BYTES = ACTIVATION_DIM * 2  # 14336
ACTIVATION_FIFO_SIZE = ACTIVATION_PAGE_SIZE_BYTES * 4
PIPELINE_CORE_COORD = ttnn.CoreCoord(11, 0)
```

---

## Complete Example: Topology-Based Four-Stage Pipeline

```python
from pipeline_builder import PipelineGraph, Node, Edge

graph = PipelineGraph(
    nodes={
        "embed":  Node(shape=ttnn.MeshShape(4, 2),
                       factory=lambda m: EmbeddingStage(load_embedding(m)),
                       name="Embedding"),
        "dec0":   Node(shape=ttnn.MeshShape(4, 2),
                       factory=lambda m: DecoderStage(load_decoder(0, m)),
                       name="Decoder 0"),
        "dec1":   Node(shape=ttnn.MeshShape(4, 2),
                       factory=lambda m: DecoderStage(load_decoder(1, m)),
                       name="Decoder 1"),
        "lmhead": Node(shape=ttnn.MeshShape(4, 2),
                       factory=lambda m: LMHeadStage(load_lm_head(m)),
                       name="LM Head"),
    },
    edges=[
        Edge("embed", "dec0"),
        Edge("dec0", "dec1"),
        Edge("dec1", "lmhead"),
        Edge("lmhead", "embed", is_loopback=True),
    ],
)

layout = graph.build_topology(mesh_device)

# Rank 0 only: dump layout for debugging and export
if ttnn.distributed_context_get_rank() == 0:
    layout.print_submeshes()
    layout.write_json("pipeline_config.json")

# Build and run the pipeline
pipeline = layout.build_pipeline()
pipeline.setup_and_run()
```

---

## Design Rationale: Why Two Build Paths

The legacy `build()` path exists for backward compatibility with deployments that have known physical topology and need deterministic chip-to-stage mappings. It also serves as a fallback when the C++ topology resolver (`resolve_graph_layout`) is not yet available or when debugging topology issues requires explicit control.

The auto-discovery `build_topology()` path is the intended production path. It eliminates:

- Hardcoded `(tray_id, asic_location)` tables that break on board revision changes or cable swaps.
- The `rev_c` flag that swapped tray IDs depending on Galaxy board revision.
- The `generate_blitz_decode_pipeline_configs.py` script that required hostname sorting.
- The triple-coupling of `pipeline_stage_index == mesh_id == mpi_rank`.

With auto-discovery, adding a new pipeline shape (different submesh count, different mesh topology) requires only changing the graph definition in Python -- no C++ changes, no recompilation.

---

[Next: Pipeline Manager Architecture >>](02_pipeline_manager_architecture.md)
