# 03 -- Fabric and Routing

This file explains the Tenstorrent ethernet fabric -- the inter-chip communication layer that enables collective operations across multi-chip topologies. It covers fabric configuration modes, the distinction between linear and ring topologies for CCL operations, the performance implications of link counts, the all-reduce strategy selection based on mesh dimensionality, and the C++ control-plane APIs that TT-Blaze's pipeline builder uses for automatic topology discovery. Prerequisites: familiarity with the physical topologies from [`01_physical_topologies.md`](./01_physical_topologies.md) and the device initialization flow from [`02_mesh_device_initialization.md`](./02_mesh_device_initialization.md).

## What the Ethernet Fabric Is

The ethernet fabric is the software/hardware layer that manages inter-chip ethernet links (described in [`01_physical_topologies.md`](./01_physical_topologies.md)) for CCL operations, pipeline-parallel forwarding, and semaphore-based synchronization. It must be configured **before** the mesh device is opened via `set_fabric()` in `conftest.py`, and reset to `DISABLED` after device closure.

## Fabric Configuration Modes

The `ttnn.FabricConfig` enum controls how the fabric is initialized. This is set via the `fabric_config` key in `device_params` and applied by the `set_fabric()` function in [tt-metal/conftest.py:set_fabric] before `ttnn.open_mesh_device()` is called.

### FabricConfig.FABRIC_1D_RING

Configures the ethernet fabric as a one-dimensional ring topology. This is the appropriate choice for 1D mesh shapes (e.g., T3K with shape (1, 8), P150x8, N300, P300, P150x4) where chips are arranged in a line.

In ring mode, the fabric enables wraparound routing: a packet from Chip 0 can reach Chip 7 by traversing either direction, halving the maximum hop distance compared to a pure linear arrangement.

```
  Ring topology (T3K example):

  +---+   +---+   +---+   +---+   +---+   +---+   +---+   +---+
  | 0 |---| 1 |---| 2 |---| 3 |---| 4 |---| 5 |---| 6 |---| 7 |
  +---+   +---+   +---+   +---+   +---+   +---+   +---+   +---+
    |                                                         |
    +---------------------------------------------------------+
                       wraparound link
```

Maximum hops in a ring of $N$ chips: $\lfloor N/2 \rfloor$. For T3K ($N=8$): 4 hops maximum.

### FabricConfig.FABRIC_2D

Configures the fabric as a two-dimensional mesh, appropriate for 2D topologies like TG (8, 4) or BHGLX (8, 4). In 2D mode, routing can proceed along both axes independently, and CCL operations can specify a `cluster_axis` to restrict communication to a single row or column of the mesh.

```
  2D fabric (Galaxy example, showing axis directions):

       axis 1 (horizontal, East-West)
       -------------------------------->
  a    +---+   +---+   +---+   +---+
  x  | | 0 |---| 1 |---| 2 |---| 3 |
  i  | +---+   +---+   +---+   +---+
  s  |   |       |       |       |
     | +---+   +---+   +---+   +---+
  0  | | 4 |---| 5 |---| 6 |---| 7 |
     | +---+   +---+   +---+   +---+
  (  |   |       |       |       |
  v  |  ...     ...     ...     ...
  e  v
  r)
```

### FabricConfig.DISABLED

No fabric. Used on single-chip configurations (N150, P150) or during teardown. Setting fabric to `DISABLED` after test completion ensures a clean state for subsequent tests:

```python
def reset_fabric(fabric_config):
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
```

> **Warning:** Fabric configuration must be set **before** calling `ttnn.open_mesh_device()`. Setting it after device creation has no effect on the already-opened mesh. Setting `FABRIC_2D` on a system that only has 1D connectivity (e.g., T3K) or `FABRIC_1D_RING` on a system that requires 2D routing (e.g., TG) will result in incorrect behavior or runtime errors.

### Additional Fabric Parameters

The `set_fabric()` function accepts several parameters beyond the primary `FabricConfig`:

```python
# [tt-metal/conftest.py:set_fabric] (simplified)
def set_fabric(fabric_config, reliability_mode=None, fabric_tensix_config=None,
               fabric_manager=None, fabric_router_config=None):
    if fabric_config:
        if reliability_mode is None:
            reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
        ttnn.set_fabric_config(fabric_config, reliability_mode, ...)
```

| Parameter               | Default                                  | Purpose                                                    |
|-------------------------|------------------------------------------|------------------------------------------------------------|
| `reliability_mode`      | `FabricReliabilityMode.STRICT_INIT`      | Controls error-checking guarantees during fabric init      |
| `fabric_tensix_config`  | `FabricTensixConfig.DISABLED`            | Enables Tensix extensions for fabric router (mux kernels)  |
| `fabric_manager`        | `FabricManagerMode.DEFAULT`              | Fabric manager operating mode                              |
| `fabric_router_config`  | `None` (optional)                        | Fine-grained router tuning; passed when explicitly needed  |

### Fabric Lifecycle

```
  set_fabric(config)              <-- Configures fabric mode globally
       |
  ttnn.open_mesh_device(...)      <-- Opens mesh; fabric links come up
       |
  [test execution]                <-- CCL ops use the active fabric
       |
  ttnn.close_mesh_device(...)     <-- Closes mesh; fabric links go down
       |
  reset_fabric(config)            <-- Resets to DISABLED for cleanup
```

## CCL Topology: Linear vs Ring

Independent of the fabric configuration, individual CCL operations specify a **topology** parameter via the `ttnn.Topology` enum that controls the communication pattern:

### ttnn.Topology.Linear

Data flows in one direction through a chain of chips. The first and last chips are endpoints -- no wrap-around. This is the default for most CCL operations:

```
  Chip 0 --> Chip 1 --> Chip 2 --> ... --> Chip N-1
```

```python
# [ccl.py:tt_all_reduce] -- default topology
topology=ttnn.Topology.Linear
```

### ttnn.Topology.Ring

Data flows in both directions around a ring, allowing each chip to send and receive simultaneously. This can improve latency for symmetric operations like all-reduce:

```
  Chip 0 --> Chip 1 --> Chip 2 --> ... --> Chip N-1
    ^                                        |
    |________________________________________|
```

> **Warning:** Ring topology is a logical construct on top of the physical links. On T3K (where chips are in a physical line with 1 link each), a ring topology means the "return" path from chip 7 to chip 0 must traverse all intermediate chips. True hardware rings exist only when `FABRIC_1D_RING` establishes the wraparound link.

### Topology Choice and cluster_axis Interaction

The topology parameter interacts with `cluster_axis` to determine the actual communication pattern:

- **1D mesh (e.g., T3K, shape (1, 8)):** `cluster_axis` is irrelevant (axis 0 has extent 1). The 8 chips communicate along axis 1. With `Topology.Ring`, reduce-scatter and all-gather operations wrap around.
- **2D mesh (e.g., TG, shape (8, 4)):** `cluster_axis=0` restricts communication to a column (8 chips), `cluster_axis=1` restricts it to a row (4 chips). The topology parameter determines whether the row/column operates as a line or a ring.

In practice, most production CCL calls in the codebase use `ttnn.Topology.Linear` as the default.

## CCL Performance and Link Counts

Each ethernet link provides an independent data channel, so the link counts from [`01_physical_topologies.md`](./01_physical_topologies.md) directly determine CCL bandwidth (e.g., P300's 2 links yield 2x the per-hop bandwidth of N300's 1 link).

### Auto-Detection in CCL Operations

The `TT_CCL` class in [ccl.py:TT_CCL] manages semaphores for collective communication and provides `get_num_links()` to auto-detect the link count. CCL functions auto-detect the number of links when explicit values are not provided:

```python
# [ccl.py:tt_all_reduce] -- auto-detect links
if num_reduce_scatter_links is None:
    num_reduce_scatter_links = tt_ccl.get_num_links(cluster_axis)
if num_all_gather_links is None:
    num_all_gather_links = tt_ccl.get_num_links(cluster_axis)
```

This means CCL operations automatically use the maximum available bandwidth for the target axis without manual configuration.

### TT_CCL Semaphore Management

The `TT_CCL` class in [ccl.py:TT_CCL] maintains three families of double-buffered global semaphores, indexed by cluster axis:

- **Barrier semaphores** -- for synchronization barriers.
- **All-gather semaphores** -- for all-gather operations (2 semaphores per buffer).
- **Reduce-scatter semaphores** -- for reduce-scatter operations (3 semaphores per buffer).

Each family has 3 slots indexed [0, 1, 2]. Due to Python truthiness, `semaphore_index = 2 if not cluster_axis else cluster_axis` maps both `cluster_axis=None` and `cluster_axis=0` to index 2 (since `not 0` is `True`), while `cluster_axis=1` maps to index 1. Index 0 is effectively unused by the current logic. The `get_and_cycle_*` methods implement double-buffering by alternating between two semaphore sets on successive calls, avoiding WAW (write-after-write) hazards between back-to-back CCL operations:

```python
def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
    semaphore_index = 2 if not cluster_axis else cluster_axis
    current_idx = self.barrier_semaphore_idx[semaphore_index]
    self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % 2
    return self.barrier_semaphore_handles[semaphore_index][current_idx]
```

## The All-Reduce Decision Tree

The `tt_all_reduce` function in [ccl.py:tt_all_reduce] makes different choices based on the mesh shape:

**1. Skip condition** — specifically when `cluster_axis == 1` and the mesh shape contains a dimension of size 1 (e.g., `(1,8)` with `cluster_axis=1`): Return the input tensor unchanged. Note that `cluster_axis=0` on a `(1,N)` mesh is *not* caught by this guard — only `cluster_axis == 1` triggers the skip.

**2. 1D mesh** (`1 in list(mesh_device.shape)`) — the N300/T3K/P150x4/P150x8 path: Uses `reduce_scatter_minimal_async` alone, since the mesh has only one meaningful dimension. There is no need for a separate all-gather step.

> **Warning:** On 1D meshes, `tt_all_reduce` returns a *reduced-and-scattered* (sharded) result, not a fully replicated all-reduce. Each device receives only its $1/N$ slice of the reduced output. Callers that need the full tensor on every device must follow with an explicit `all_gather`.

```python
if 1 in list(mesh_device.shape):
    reduced = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor, ..., topology=topology, num_links=num_reduce_scatter_links, ...
    )
    return reduced  # sharded: each device holds 1/N of the result
```

**3. 2D mesh** (no dimension is 1) -- the TG/BHGLX path: Two strategies are available, controlled by the `use_composite` flag:

- **Non-composite (default):** `all_gather_async` along `cluster_axis`, then `fast_reduce_nc` locally to sum across the gathered dimension.
- **Composite:** `reduce_scatter_minimal_async` along `cluster_axis`, then `all_gather_async` along the same axis. This can be more memory-efficient because the intermediate tensor after reduce-scatter is smaller.

```python
# 2D non-composite path
if not use_composite:
    gathered_tensor = ttnn.experimental.all_gather_async(...)
    reduced_tensor = ttnn.experimental.fast_reduce_nc(gathered_tensor, ...)
# 2D composite path
else:
    reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(...)
    reduced_tensor = ttnn.experimental.all_gather_async(reduced_tensor, ...)
```

## C++ Control-Plane APIs for Pipeline Topology Discovery

TT-Blaze's `PipelineGraph.build_topology()` method in [graph.py:PipelineGraph.build_topology] uses C++ control-plane APIs to automatically discover the physical layout of a pipeline without hardcoded chip IDs.

### resolve_graph_layout

The central topology resolver, invoked as:

```python
result = ttnn._ttnn.multi_device.experimental.resolve_graph_layout(
    edges=edge_tuples,           # [(src_name, dst_name, is_loopback), ...]
    submesh_chips=submesh_chips,  # [[(mesh_id, chip_id, row, col), ...], ...]
)
```

This C++ function:
1. Examines the physical ethernet connectivity between all submeshes.
2. Performs a topological sort of the pipeline graph (Kahn's algorithm on non-loopback edges).
3. Uses backtracking search to assign logical pipeline nodes to physical submeshes based on which pairs have direct ethernet connectivity.
4. Returns the resolved layout including:
   - `stage_order` -- node names in pipeline execution order.
   - `node_to_submesh` -- mapping from node name to submesh index.
   - `resolved_edges` -- per-edge entry/exit chip coordinates (`entry_row`, `entry_col`, `exit_row`, `exit_col`).
   - `h2d_entry_row/col` -- host-to-device entry point coordinates.
   - `d2h_exit_row/col` -- device-to-host exit point coordinates.

### Topological Sort (Stage Ordering)

The `PipelineGraph.stage_order()` method performs Kahn's algorithm on non-loopback edges to determine the pipeline execution order. Loopback edges (last stage back to first) are excluded to avoid cycles:

```python
# [graph.py:PipelineGraph.stage_order]
def stage_order(self) -> list[str]:
    in_degree = {name: 0 for name in self.nodes}
    for edge in self._regular_edges():
        in_degree[edge.dst] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for edge in self._outgoing(node):
            in_degree[edge.dst] -= 1
            if in_degree[edge.dst] == 0:
                queue.append(edge.dst)
    return order
```

### get_forwarding_direction / get_chip_neighbors

Lower-level C++ APIs used internally by `resolve_graph_layout`:

- **`get_forwarding_direction`**: Given two chip IDs, returns the direction (North, South, East, West) that data should be forwarded to reach the destination.
- **`get_chip_neighbors`**: Returns the set of chips directly reachable from a given chip via ethernet links.

These are not called directly from Python -- they form the foundation of the auto-discovery algorithm within the C++ `resolve_graph_layout` implementation.

### Submesh Information Collection

Before calling the C++ resolver, `build_topology()` collects per-submesh chip information:

```python
submesh_chips = []
for sub in submeshes:
    chips = []
    for row in range(sub.shape[0]):
        for col in range(sub.shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            fid = sub.get_fabric_node_id(coord)
            chips.append((int(fid.mesh_id), int(fid.chip_id), row, col))
    submesh_chips.append(chips)
```

## Topology Auto-Discovery vs. Explicit Chip IDs

TT-Blaze's `PipelineGraph` [graph.py:PipelineGraph] supports two build paths: `build()` (legacy explicit with hardcoded chip IDs) and `build_topology()` (auto-discovery with no hardcoded IDs). The auto-discovery path is preferred for production use because it is hardware-agnostic. See [Chapter 5, File 1](../ch05_blaze_pipeline_system/01_pipeline_graph_and_layout.md) for the full comparison, code examples, and detailed explanation of both paths.

### Pipeline Example: 4-Stage on TG (8x4)

```
  Parent mesh 8x4, submesh shape 4x2 -> 4 submeshes:

  +----------+----------+
  | submesh 0| submesh 1|
  |  (4x2)   |  (4x2)   |
  +----------+----------+
  | submesh 2| submesh 3|
  |  (4x2)   |  (4x2)   |
  +----------+----------+

  Pipeline graph:
    s0 --> s1 --> s2 --> s3 --+
    ^                         |
    +---- (loopback) ---------+

  resolve_graph_layout() determines:
    - Which submesh hosts which stage
    - Which chip in each submesh is the entry/exit point for inter-stage data
    - H2D entry and D2H exit coordinates
```

### PipelineLayout

The `PipelineLayout` dataclass [submesh_partition.py:PipelineLayout] bundles everything a rank needs to participate in a pipeline -- stage index, submesh reference, stage-to-submesh mapping, pipeline config entries, and stage factories. See [Chapter 5, File 2](../ch05_blaze_pipeline_system/02_submesh_partition_and_topology.md) for the full dataclass definition and the end-to-end topology auto-discovery flow.

---

**Next:** [Chapter 2 — TT-Symbiote Core Architecture](../ch02_symbiote_core/index.md)
