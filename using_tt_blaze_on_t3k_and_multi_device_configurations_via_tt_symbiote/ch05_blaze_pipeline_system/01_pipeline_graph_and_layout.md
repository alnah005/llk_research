# 5.1 PipelineGraph and Layout

This section covers the `PipelineGraph` class -- the logical description of a multi-stage pipeline as a directed graph of nodes and edges. It explains how stages are ordered via topological sort, how the two build paths (legacy explicit vs. topology auto-discovery) produce a `PipelineLayout`, and how a token physically enters and exits the pipeline ring. This is the top layer of a three-layer architecture that separates concerns across the pipeline subsystem.

**Prerequisites:** Ch1 (mesh topologies, fabric routing), Ch4 (BlazeOp hierarchy, FusedProgram).

---

## Three-Layer Architecture Overview

Every TT-Blaze pipeline separates concerns into three distinct layers. This section covers Layer 1; subsequent sections cover Layers 2 and 3.

```
 +============================================================+
 |  LAYER 1 -- LOGICAL GRAPH  (this section)                  |
 |  PipelineGraph, Node, Edge, stage_order()                  |
 |  "What stages exist and how are they connected?"           |
 +========================+===================================+
                          |  build() or build_topology()
 +========================v===================================+
 |  LAYER 2 -- PHYSICAL PARTITION  (Section 5.2)              |
 |  SubmeshPartition, chip_to_coord_map, resolve_graph_layout |
 |  "Which physical chips run each stage?"                    |
 +========================+===================================+
                          |  PipelineLayout
 +========================v===================================+
 |  LAYER 3 -- RUNTIME EXECUTION  (Sections 5.3, 5.4, 5.5)   |
 |  StageKind, Pipeline orchestrator, PipelineManager (C++)   |
 |  "How does each stage execute and how are tokens routed?"  |
 +============================================================+
```

---

## Core Data Structures

`PipelineGraph` is defined in `pipeline_builder/graph.py` (line 110). It holds three pieces of state:

```
PipelineGraph
  .nodes   : dict[str, Node]       # ordered dict -- insertion order matters
  .edges   : list[Edge]            # directed connections + one loopback
  .h2d_entry_chip : Optional[int]  # legacy path only
  .d2h_exit_chip  : Optional[int]  # legacy path only
```

### Node (line 80)

```python
@dataclass
class Node:
    shape: Any                          # ttnn.MeshShape (e.g. MeshShape(4,2))
    factory: Optional[Callable] = None  # factory(submesh) -> StageKind instance
    name: str = ""                      # display label for visualization
```

Each `Node` represents one pipeline stage. The `shape` field determines how many chips that stage occupies (e.g. `MeshShape(4,2)` = 8 chips). The `factory` is a callable that, given a submesh device, creates the concrete `StageKind` (see Section 5.3). For topology-based builds, all nodes must have the same shape.

### Edge (line 89)

```python
@dataclass
class Edge:
    src: str                             # source node name
    dst: str                             # destination node name
    src_exit_chip: Optional[int] = None  # legacy explicit path only
    dst_entry_chip: Optional[int] = None # legacy explicit path only
    is_loopback: bool = False            # True for the return edge
```

Regular edges form the forward path. Exactly one edge should have `is_loopback=True` -- this is the return path from the last stage back to the first, closing the pipeline ring. The loopback edge is excluded from topological sort. The chip ID fields are only populated for the legacy `build()` path; `build_topology()` discovers them from the fabric routing tables.

---

## Stage Ordering: Kahn's Algorithm

`PipelineGraph.stage_order()` (line 176) computes the execution order via Kahn's algorithm on **non-loopback** edges only:

```
stage_order():
    1. Compute in-degree for each node (ignoring loopback edges)
    2. Seed queue with zero-in-degree nodes
    3. BFS: pop node, append to order, decrement successors' in-degrees
    4. Assert len(order) == len(nodes)  -- cycle check
```

For a linear 4-stage pipeline, this produces `["f0", "f1", "f2", "f3"]`. The loopback edge `f3 -> f0` is intentionally excluded; including it would create a cycle and fail the DAG assertion.

> **Warning:** If two nodes have identical in-degree (both zero), `queue.pop(0)` uses dict insertion order as the tiebreaker. If you rely on a specific stage ordering for performance (e.g., placing Embedding first), ensure the `nodes` dict has that entry first.

---

## Token Data Flow Through the Pipeline Ring

```
 Host CPU                                        Device Pipeline
 =========                                       ================

 write_token()                                 +---> [Stage 0: Embed]
      |                                        |        |  activation
      v                                        |        v
  H2D Socket ----(token)----> Entry Chip  -----+     [Stage 1: Decoder]
                                                        |  activation
                                                        v
                                                     [Stage 2: Decoder]
                                                        |  activation
  D2H Socket <---(token)---- Exit Chip  <---+           v
      |                                     |        [Stage 3: LMHead]
      v                                     |           |  token
 read_output()                              +-----------+
                                              (loopback)
```

**Forward path:** A token enters via the H2D socket at `h2d_entry_chip` (stage 0's entry), gets embedded into an activation vector, and flows through D2D sockets between consecutive stages. Each stage's `exit_node_coord` connects to the next stage's `entry_node_coord` via ethernet.

**Loopback path:** The last stage (e.g. LMHead) produces a token that travels back to stage 0's submesh via the loopback D2D socket. Stage 0 then delivers the token to the host through the D2H socket at `d2h_exit_chip`.

---

## Build Path 1: Legacy Explicit (`build()`, line 204)

The legacy path requires the caller to supply hardcoded chip IDs on every edge:

```python
graph = PipelineGraph(
    nodes={...},
    edges=[
        Edge("f0", "f1", src_exit_chip=5,  dst_entry_chip=6),
        Edge("f1", "f2", src_exit_chip=15, dst_entry_chip=19),
        Edge("f2", "f3", src_exit_chip=22, dst_entry_chip=21),
        Edge("f3", "f0", src_exit_chip=16, dst_entry_chip=12, is_loopback=True),
    ],
    h2d_entry_chip=4,
    d2h_exit_chip=8,
)
layout = graph.build(partition, ordering=[0, 1, 3, 2])
```

The `ordering` list maps `ordering[stage_idx] = submesh_idx` -- which physical submesh each logical stage runs on. The `build()` method (line 204) converts global chip IDs to local `MeshCoordinate` via `partition.chip_to_coord_map()`, then assembles a `PipelineConfigEntry` list:

- `pipeline_config[i]` for `i in 0..N-1`: entry/exit coords for stage `i`.
- `pipeline_config[N]`: loopback entry coord + D2H exit coord.

> **Warning:** The legacy path requires knowing exact chip IDs from the physical topology. These IDs change between hardware configurations and firmware versions. Use `build_topology()` for portable pipelines.

---

## Build Path 2: Topology Auto-Discovery (`build_topology()`, line 269)

The preferred path requires **no hardcoded chip IDs**. The method:

1. **Validates** all nodes have the same submesh shape (line 298).
2. **Creates submeshes** via `mesh_device.create_submeshes(submesh_shape)` (line 302).
3. **Detects local submesh** by checking `sub.get_view().is_local(MeshCoordinate(0,0))` (line 313).
4. **Collects chip info** (fabric node IDs + local coordinates) for each submesh (line 325).
5. **Calls C++ resolver**: `ttnn._ttnn.multi_device.experimental.resolve_graph_layout()` (line 338) -- this uses `get_forwarding_direction` / `get_chip_neighbors` to discover ethernet links.
6. **Builds pipeline_config** from the resolved edge coordinates (line 350).
7. **Derives ordering** and per-rank metadata via `distributed_context_allgather_int()` (line 386).

The result is a `PipelineLayout` that maps each MPI rank to its stage, submesh, and config entries.

> **Warning:** `build_topology()` requires that all nodes have the **same** submesh shape (line 298). Mixed shapes (e.g. 4x2 for decoders and 2x2 for LMHead) are not supported by the topology resolver. Use `build()` with explicit chip IDs for heterogeneous stage shapes.

> **Warning (failure mode):** All ranks must call `distributed_context_allgather_int()` collectively. If one rank crashes between `resolve_graph_layout()` and the allgather, the remaining ranks hang indefinitely. There is no timeout on the allgather.

---

## PipelineConfigEntry Layout

Both build paths produce the same layout:

```
  Index    entry_node_coord                    exit_node_coord
  -------  ----------------------------------  ----------------------------------
  [0]      H2D entry chip in stage 0's mesh    chip connecting to stage 1
  [1]      chip receiving from stage 0         chip connecting to stage 2
  ...
  [N-1]    chip receiving from stage N-2       chip sending loopback to stage 0
  [N]      chip in stage 0 receiving loopback  D2H exit chip in stage 0's mesh
```

Each entry has two `MeshCoordinate` fields:
- `entry_node_coord` -- the chip coordinate in this stage's submesh that receives data from the previous stage (or H2D for stage 0).
- `exit_node_coord` -- the chip coordinate that sends data to the next stage.

---

## Example: 4-Stage Linear Pipeline with Loopback

```python
graph = PipelineGraph(
    nodes={
        "embed":   Node(shape=ttnn.MeshShape(4,2), factory=make_embed),
        "decoder": Node(shape=ttnn.MeshShape(4,2), factory=make_decoder),
        "lmhead":  Node(shape=ttnn.MeshShape(4,2), factory=make_lmhead),
        "fwd":     Node(shape=ttnn.MeshShape(4,2), factory=make_passthrough),
    },
    edges=[
        Edge("embed",   "decoder"),
        Edge("decoder", "lmhead"),
        Edge("lmhead",  "fwd"),
        Edge("fwd",     "embed", is_loopback=True),
    ],
)
layout = graph.build_topology(mesh_device)
pipeline = layout.build_pipeline()  # creates Pipeline for this rank's stage
pipeline.setup_and_run()            # 4-phase lifecycle (see Section 5.3)
```

Data flow with loopback:

```
  Host
    |  H2D (token)
    v
  [embed]---activation--->[decoder]---activation--->[lmhead]---token--->[fwd]
    ^                                                                    |
    |                         token (loopback)                           |
    +--------------------------------------------------------------------+
    |
    v  D2H (token)
  Host
```

> **Warning:** `build_topology()` requires that the number of nodes equals the number of submeshes created from the mesh device. A 32-chip galaxy with `MeshShape(4,2)` yields 4 submeshes -- the graph must have exactly 4 nodes.

---

## Key Invariants

1. All nodes must have the **same submesh shape** when using `build_topology()`.
2. The number of nodes must **equal** the number of submeshes yielded by `create_submeshes()`.
3. Exactly **one loopback edge** must exist, connecting the last stage to the first.
4. The `pipeline_config` list has **N+1 entries**: indices 0..N-1 for forward stages, index N for the loopback/D2H metadata.

---

## Visualization Export

`PipelineLayout.write_json(path)` [pipeline_builder/submesh_partition.py, lines 98-115] serializes the layout as JSON for the Blaze observability viewer. When `BLAZE_EXPORT=1` is set, it additionally writes to the blaze export directory. The JSON includes per-chip roles (`h2d_entry`, `d2d_entry`, `d2d_exit`, `lb_d2d_entry`, `d2h_exit`), stage-to-submesh mappings, and mesh grid dimensions.

---

| Previous | Up | Next |
|----------|-----|------|
| [Ch4: Blaze Kernel Composition](../ch4/) | [Table of Contents](../README.md) | [5.2 SubmeshPartition and Topology](02_submesh_partition_and_topology.md) |
