# 03 -- Sem Engine

The SemEngine is the second engine in the compilation pipeline. It identifies inter-core synchronization points in the graph and assigns sequential global semaphore indices. Each index maps to a `ttnn.GlobalSemaphore` object that the `BlazeCompiler` creates at compile time. The engine derives its sync points directly from `Sem`-kind entries in the CT arg schema, and handles mcast sender grouping for persistent NOC command state sharing.

**Source file:** `blaze/sem_engine.py`

---

## 1. Background: Global Semaphores on Blackhole

On Blackhole, inter-core synchronization uses global semaphores rather than program-level semaphores. There is no hardware slot limit for global semaphores (the old 16-slot constraint applied to program-level semaphores, which are no longer used). Each global semaphore is a `ttnn.GlobalSemaphore` object backed by L1 memory, and its address is resolved to an integer at compile time.

Not every op in a Blaze graph requires semaphores. Semaphores are only needed for **inter-core** operations -- ops that move data between different Tensix cores via the NOC:

- **Gather**: Collects tiles from multiple sender cores to a single receiver core. Requires semaphores on both NOC0 and NOC1 (dual-NOC mode) or NOC0 only (single-NOC mode).
- **Mcast** (multicast): Broadcasts tiles from a sender core to a grid of receiver cores. Requires a sender semaphore and a receiver semaphore.
- **moe_gather**: A variant of gather specialized for Mixture of Experts routing.
- **CCL ops**: Cross-chip collective operations that synchronize across devices.

Local compute ops (matmul, elementwise, reduce) do not need semaphores -- the RISC processors on each core coordinate via L1-local CBs, not semaphores.

### Design Note

The semaphore engine is deliberately simple: sequential allocation, no lifetime-based reuse, and the protocol string IS the source_key. This simplicity is intentional -- semaphore count is never a bottleneck (no hardware slot limit), so the engine optimizes for correctness and debuggability rather than minimal allocation.

---

## 2. Sync Point Identification

### 2.1 Protocol Source Keys from CT Arg Schemas

Rather than hardcoding sync patterns per op type, the SemEngine discovers semaphore requirements by inspecting each op's CT arg schema. Any `CTArgSpec` with `source="sem"` indicates a semaphore dependency, and its `source_key` (e.g., `"noc0_receiver_semaphore"`, `"sender_semaphore"`) becomes the protocol identifier:

```python
def _get_sem_source_keys(self, node: OpNode) -> list[str]:
    schema = get_ct_schema(node.spec.name)
    if schema is None:
        return []
    # Deduplicate: same source_key across multiple RISCs = one semaphore
    source_keys = list(dict.fromkeys(
        spec.source_key for spec in schema.args if spec.source == "sem"
    ))
    if not source_keys:
        return []
    # Conditional: gather/moe_gather with dual_noc=False skip noc1 sems
    if node.spec.name in ("gather", "moe_gather") and not node.kwargs.get("dual_noc", True):
        source_keys = [k for k in source_keys if "noc1" not in k]
    return source_keys
```

Key details:

1. **Schema lookup**: `get_ct_schema()` retrieves the registered `CTArgSchema` for the op type. If no schema exists (e.g., a simple elementwise op), no semaphores are needed.
2. **Deduplication across RISCs**: The same `source_key` may appear in CT arg specs targeting different RISCs (e.g., NCRISC and BRISC both reference `"noc0_receiver_semaphore"`). These are deduplicated via `dict.fromkeys()` -- one semaphore suffices.
3. **Conditional NOC1**: Gather and `moe_gather` ops default to dual-NOC mode (`dual_noc=True`), which requires both `noc0_*` and `noc1_*` semaphores. When `dual_noc=False`, the engine filters out `noc1` keys, reducing the semaphore count from 2 to 1.

### 2.2 SyncPoint Data Structure

Each identified sync requirement becomes a `SyncPoint`:

```python
@dataclass
class SyncPoint:
    node: OpNode
    protocol: str       # e.g. "noc0_receiver_semaphore", "sender_semaphore"
    topo_pos: int        # position in topological ordering
    initial_value: int = 0
    core_ranges: Any = None
```

The `protocol` field is the canonical CT arg `source_key`. This is deliberately the same key that the CTArgEngine will use to look up the resolved semaphore address -- creating a clean handoff between engines.

---

## 3. The Assignment Algorithm

The `assign()` method proceeds in two phases:

### Phase 1: Identify Sync Points

Walk the graph in topological order. For each node with inter-core semaphore requirements, create one `SyncPoint` per protocol key:

```python
def _identify_sync_points(self, topo_order, pos_map) -> list[SyncPoint]:
    sync_points = []
    for node in topo_order:
        source_keys = self._get_sem_source_keys(node)
        if not source_keys:
            continue
        for key in source_keys:
            sync_points.append(SyncPoint(
                node=node, protocol=key,
                topo_pos=pos_map[node.id],
                initial_value=0, core_ranges=node.grid,
            ))
    return sync_points
```

### Phase 2: Assign Indices

Walk the sync points in order and assign unique sequential indices, with special handling for mcast sender groups:

```python
def _assign_indices(self, sync_points):
    groups = _McastSenderGroups(sync_points)
    group_sem: dict[int, int] = {}  # group_id -> assigned sem_id
    next_id = 0
    assignments = []

    for sp in sync_points:
        gid = groups.group_id(sp)
        if gid is not None and gid in group_sem:
            sem_id = group_sem[gid]  # reuse group's sem_id
        else:
            sem_id = next_id
            next_id += 1
            if gid is not None:
                group_sem[gid] = sem_id

        assignments.append(SemAssignment(
            sem_id=sem_id, global_sem=True,
            initial_value=sp.initial_value,
            core_ranges=sp.core_ranges,
            protocol=sp.protocol,
            node_id=sp.node.id,
        ))
    return assignments
```

Each sync point gets a unique sequential index, **except** mcast senders that share a persistent state group -- they reuse the same index.

---

## 4. Mcast Sender Grouping

### 4.1 The Persistent State Problem

Mcast ops with the same explicit `sender` kwarg share a single persistent NOC command buffer. The kernel design requires that `init()`, `run_as()`, and `teardown()` execute on the same persistent sender state. Consequently, their sender semaphores must use the **same** semaphore ID.

### 4.2 `_McastSenderGroups`

The `_McastSenderGroups` helper identifies and groups mcast sender sync points:

```python
class _McastSenderGroups:
    def __init__(self, sync_points):
        self._groups: dict[str, int] = {}  # sender_key -> group_id
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

**Grouping logic**:
- Only mcast nodes with `protocol == "sender_semaphore"` are candidates.
- The grouping key is the stringified `sender` kwarg value (typically a core coordinate tuple like `"(0, 0)"`).
- Nodes with the same `sender` value share a group and reuse the same semaphore ID.
- Nodes without an explicit `sender` kwarg are not grouped (each gets its own ID).

This is essential for the kernel codegen's mcast leader/follower pattern (see [Section 05](./05_kernel_codegen.md)), where the leader `init()`s the persistent state and followers `run_as<>()` against it.

---

## 5. SemAssignment

```python
@dataclass
class SemAssignment:
    sem_id: int         # sequential index (0, 1, 2, ...)
    global_sem: bool    # always True (kept for backward compat / viz)
    initial_value: int
    core_ranges: Any
    protocol: str       # CT arg source_key, e.g. "noc0_receiver_semaphore"
    node_id: str        # ID of the graph node
```

The `global_sem` field is always `True` in current Blaze -- program-level semaphores are no longer used. The field is retained for backward compatibility with visualization tools.

The `protocol` field serves double duty: it is both the identifier for the sync point and the lookup key used by the CT arg engine to resolve the semaphore's L1 address.

---

## 6. From Index to L1 Address

The semaphore engine only assigns logical indices (0, 1, 2, ...). The mapping from index to physical L1 address happens in `BlazeCompiler._create_global_semaphores()`:

```python
def _create_global_semaphores(self, sem_assignments):
    num_sems = max(sa.sem_id for sa in sem_assignments) + 1
    while len(self._graph_sem_pool) < num_sems:
        self._graph_sem_pool.append(
            ttnn.create_global_semaphore(self._mesh_device, avail, 0)
        )
    global_sems = self._graph_sem_pool[:num_sems]
    sem_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in global_sems]
    return global_sems, sem_addrs
```

$$\text{sem\_id} \rightarrow \text{GlobalSemaphore}[i] \rightarrow \text{L1 address}$$

The compiler then builds a `sem_for_ct` mapping (`{node_id: {protocol: L1_address}}`) that feeds into the CT arg engine, closing the loop between semaphore index assignment and compile-time argument resolution. At runtime, the kernel accesses the semaphore via its L1 address directly -- there is no indirection through a semaphore table.

The `GlobalSemaphore` objects are pinned to the `MeshCompiledProgram._lifetime_semaphores` list to prevent garbage collection. Without this, the semaphore L1 addresses baked into `ProgramDescriptor` CT args would become dangling pointers, causing use-after-free on dispatch.

---

## 7. Worked Example: Gather + Mcast Pipeline

Consider a graph with a gather op (dual NOC) followed by two mcast ops sharing a sender:

```python
with blaze.fuse() as ctx:
    gathered = blaze.gather({"input": ext}, grid=grid, dual_noc=True)
    mc1 = blaze.mcast({"input": gathered}, grid=grid, sender=(0, 0))
    mc2 = blaze.mcast({"input": mc1}, grid=grid, sender=(0, 0))
```

The semaphore engine produces:

| sem_id | node | protocol | Notes |
|--------|------|----------|-------|
| 0 | gather_1 | `noc0_receiver_semaphore` | Gather NOC0 |
| 1 | gather_1 | `noc1_receiver_semaphore` | Gather NOC1 (`dual_noc=True`) |
| 2 | mcast_1 | `sender_semaphore` | Mcast sender group for `sender=(0,0)` |
| 3 | mcast_1 | `receiver_semaphore` | Mcast receiver (unique per node) |
| **2** | mcast_2 | `sender_semaphore` | **Same** group -- shares sem_id with mcast_1 |
| 4 | mcast_2 | `receiver_semaphore` | Mcast receiver (unique per node) |

The shared `sem_id=2` for the two mcast sender sync points reflects the persistent NOC command buffer sharing. Total semaphores allocated: 5 (not 6), because the two mcast sender semaphores share an index.

---

## 8. Integration with the Pipeline

The SemEngine is the second engine in the `compile_engines()` pipeline (see [Section 01](./01_blaze_graph_and_fusion_context.md) for the full orchestrator). It has no dependency on CB assignments — it only reads the graph topology and CT arg schemas. The CTArgEngine (next in the pipeline) depends on semaphore assignments to resolve `source="sem"` CT args. In the `BlazeCompiler`, semaphore assignments are consumed by `_create_global_semaphores()` and `_build_sem_for_ct()` to map logical indices to L1 addresses.

---

<- [02 -- CB Engine](./02_cb_engine.md) | [Index](index.md) | [04 -- CT Arg Engine](./04_ct_arg_engine.md) ->
