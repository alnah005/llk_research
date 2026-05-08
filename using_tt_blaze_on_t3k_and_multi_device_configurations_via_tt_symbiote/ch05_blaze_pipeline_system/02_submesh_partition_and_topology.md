# 5.2 SubmeshPartition and Topology

This section covers the physical partitioning layer that carves a parent `MeshDevice` into uniform submeshes, the C++ topology resolver that discovers inter-submesh ethernet links, the MPI rank-to-stage binding, and the `PipelineLayout` dataclass that bundles all per-rank artifacts needed to launch a pipeline stage. This is Layer 2 of the three-layer architecture introduced in Section 5.1.

**Prerequisites:** Section 5.1 (PipelineGraph, build paths), Ch1 (MeshDevice, fabric node IDs).

---

## Layer Mapping

```
  Layer 1 (Logical)                    Layer 2 (Physical)
  +-----------------------------+      +------------------------------+
  | PipelineGraph               |      | SubmeshPartition             |
  |   nodes: {name -> Node}     |----->|   submeshes[]                |
  |   edges: [Edge]             |      |   local_submesh_idx          |
  |   stage_order() -> [names]  |      |   chip_to_coord_map()        |
  +-----------------------------+      +------------------------------+
              |                                     |
              |  build() or build_topology()        |  build_layout()
              |                                     |
              v                                     v
  +------------------------------------------------------------+
  | PipelineLayout                                              |
  |   my_stage_idx, my_submesh, pipeline_config,                |
  |   submesh_to_stage, stages_metadata, stage_factories        |
  +------------------------------------------------------------+
              |
              v
  Layer 3 (Runtime) -- StageKind.setup(), Pipeline.setup_and_run()
```

---

## SubmeshPartition

Defined in `pipeline_builder/submesh_partition.py` (line 143), `SubmeshPartition` wraps the `MeshDevice.create_submeshes()` call and resolves which submesh belongs to the current MPI rank.

### Construction

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
```

The `create_submeshes()` call tiles the parent mesh into non-overlapping regions of the given shape. For example, an 8x4 parent mesh with `MeshShape(4,2)` yields 4 submeshes:

```
  Parent MeshDevice (8x4)         Submeshes (4x2 each)
  +-----------+-----------+       Submesh 0: rows 0-3, cols 0-1
  |           |           |       Submesh 1: rows 0-3, cols 2-3
  | Submesh 0 | Submesh 1 |       Submesh 2: rows 4-7, cols 0-1
  |           |           |       Submesh 3: rows 4-7, cols 2-3
  +-----------+-----------+
  |           |           |
  | Submesh 2 | Submesh 3 |
  |           |           |
  +-----------+-----------+
```

> **Warning:** The `is_local()` test checks only coordinate (0,0). If the assertion on line 164 fires ("No local submesh found"), this almost always means the rank binding YAML does not cover all chips in the parent mesh, or the submesh shape does not evenly divide the mesh dimensions.

### chip_to_coord_map() (line 189)

Builds a `{global_chip_id -> local MeshCoordinate}` dictionary across all submeshes:

```python
def chip_to_coord_map(self) -> dict[int, MeshCoordinate]:
    mapping = {}
    for sub in self.submeshes:
        for row in range(sub.shape[0]):
            for col in range(sub.shape[1]):
                coord = ttnn.MeshCoordinate(row, col)
                fid = sub.get_fabric_node_id(coord)
                mapping[fid.chip_id] = coord
    return mapping
```

This is used only by the legacy `build()` path. The topology path uses fabric node IDs directly, never referencing global chip IDs.

> **Warning (failure mode):** In multi-galaxy setups, chip IDs must be unique across all galaxies. If two galaxies share a chip_id namespace (incorrect mesh graph descriptor), the mapping silently overwrites, and one submesh gets the wrong coordinates.

---

## Concrete Example: 4-Galaxy Partitioning

For DeepSeek V3 with `ModelPipelineSpec(num_galaxies=4, stage_rows=4, stage_cols=2)`:

```
Parent MeshDevice: 32 rows x 4 cols = 128 chips
Submesh shape:     MeshShape(4, 2)  = 8 chips each
Result:            128 / 8 = 16 submeshes
```

The parent mesh is tiled in row-major order:

```
Parent 32x4 grid (4 galaxies stacked vertically):

Galaxy 0  | sub0 (rows 0-3, cols 0-1)  | sub1 (rows 0-3, cols 2-3)  |
(rows 0-7)| sub2 (rows 4-7, cols 0-1)  | sub3 (rows 4-7, cols 2-3)  |
----------|-------------------------------------------------------|
Galaxy 1  | sub4 (rows 8-11, cols 0-1)  | sub5 (rows 8-11, cols 2-3) |
(rows 8-15)| sub6 (rows 12-15, cols 0-1) | sub7 (rows 12-15, cols 2-3)|
----------|-------------------------------------------------------|
Galaxy 2  | sub8 (rows 16-19, cols 0-1) | sub9 (rows 16-19, cols 2-3)|
(rows 16-23)| sub10(rows 20-23, cols 0-1) | sub11(rows 20-23, cols 2-3)|
----------|-------------------------------------------------------|
Galaxy 3  | sub12(rows 24-27, cols 0-1) | sub13(rows 24-27, cols 2-3)|
(rows 24-31)| sub14(rows 28-31, cols 0-1) | sub15(rows 28-31, cols 2-3)|
```

Each MPI rank owns exactly one submesh. With 16 processes, rank `r` initially maps to `sub_r`, but the topology resolver may permute this (submesh 5 might become stage 12).

---

## C++ Topology Resolver

The auto-discovery path calls into C++ via `resolve_graph_layout()` [graph.py, line 338], which performs adjacency discovery, backtracking assignment of logical nodes to physical submeshes, and entry/exit coordinate resolution. See [Chapter 1, File 3](../ch01_device_topologies/03_fabric_and_routing.md) for the full algorithm, call signature, and return fields.

> **Warning (failure mode):** If no valid assignment exists (e.g., the mesh graph descriptor has disconnected components), the resolver exhausts all permutations and raises a RuntimeError. On a 64-stage quad-galaxy config, this can take several seconds.

---

## MPI Rank Binding

After `build_topology()` determines the submesh-to-stage mapping, it uses an MPI allgather to determine which rank owns which stage (line 386):

```
 Rank 0         Rank 1         Rank 2         Rank 3
 my_stage=2     my_stage=0     my_stage=3     my_stage=1

 allgather([2, 0, 3, 1])

 stage_to_rank = {0: 1, 1: 3, 2: 0, 3: 2}
```

This is stored in `StageMetadata` objects (one per stage) inside the `PipelineLayout`. The rank-to-stage mapping is **not** identity in general -- it depends on how the rank binding YAML assigns hosts to physical chips.

> **Warning:** If rank binding is misconfigured (e.g. two ranks mapped to the same submesh, or a submesh with no rank), `build_topology()` will fail at the local submesh detection step. Verify rank binding YAML matches the mesh graph descriptor before debugging topology failures.

---

## PipelineLayout Dataclass

Defined at line 54, this is the central output of the graph build process:

```python
@dataclass
class PipelineLayout:
    my_stage_idx: int                  # this rank's stage index
    my_submesh: MeshDevice             # this rank's submesh device
    my_submesh_idx: int                # index into submeshes[]
    submesh_to_stage: dict             # submesh_idx -> stage_idx
    stages_metadata: dict              # stage_idx -> StageMetadata(rank, mesh_id)
    pipeline_config: list              # list[PipelineConfigEntry]
    submeshes: list                    # all submeshes
    stage_factories: Optional[dict]    # stage_idx -> factory callable
    mesh_shape: Optional[tuple]        # parent mesh (rows, cols)
    stage_labels: dict                 # stage_idx -> display name
```

### build_pipeline() (line 117)

This convenience method creates a `Pipeline` for the current rank:

```python
def build_pipeline(self):
    factory = self.stage_factories[self.my_stage_idx]
    stage = factory(self.my_submesh)
    return Pipeline(self.my_submesh, stage, self.my_stage_idx,
                    stages_metadata=self.stages_metadata,
                    pipeline_config=self.pipeline_config)
```

> **Warning (failure mode):** If `Node.factory` is None for any stage, `build_pipeline()` raises RuntimeError (lines 125-132). If the factory callable raises on one rank (e.g., OOM during weight loading), other ranks proceed but the pipeline stalls when the failing stage never opens its D2D sockets. Diagnosis: check all ranks' logs; use `mpirun --output-filename` to capture per-rank stderr.

### build_layout() (line 211)

The `SubmeshPartition.build_layout()` method is the legacy counterpart of `build_topology()`. It takes an explicit `ordering` list and pre-computed `pipeline_config`, derives the rank binding via allgather, and returns a `PipelineLayout`.

> **Warning:** `build_topology()` and `SubmeshPartition.build_layout()` both call `create_submeshes()`. If you mix the two paths, you get two sets of submeshes which may conflict. Pick one path.

---

## Diagnostic Methods

`PipelineLayout` provides methods useful for debugging multi-rank setups:

- `print_my_submesh()` (line 68): Prints this rank's submesh grid with fabric node IDs.
- `print_submeshes()` (line 82): Rank-0-only dump of all submeshes with stage assignments.
- `write_json(path)` (line 98): Exports layout for the web visualizer. When `BLAZE_EXPORT=1`, also writes to the blaze export directory.

> **Warning:** `print_submeshes()` silently returns on non-zero ranks. `write_json()` should only be called from rank 0 -- concurrent writes from multiple ranks cause file corruption.

---

---

| Previous | Up | Next |
|----------|-----|------|
| [5.1 PipelineGraph and Layout](01_pipeline_graph_and_layout.md) | [Table of Contents](../README.md) | [5.3 Stage Kinds and Execution](03_stage_kinds_and_execution.md) |
