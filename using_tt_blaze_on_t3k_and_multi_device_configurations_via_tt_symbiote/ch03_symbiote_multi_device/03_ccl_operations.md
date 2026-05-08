# Collective Communication Operations

Every multi-device matmul in TT-Symbiote produces partial results that must be combined via Collective Communication Library (CCL) operations. This file covers the `TT_CCL` class that manages semaphores for double-buffered CCL, the `get_num_links()` function that maps topologies to ethernet link counts, the core CCL primitives (`reduce_scatter`, `all_gather`, and the composite `tt_all_reduce`), topology adaptation for different mesh shapes, the trace-compatible all-reduce decomposition, Ring vs Linear topology selection, and a back-of-envelope cost analysis for T3K. Prerequisites: [Chapter 1](../ch01_device_topologies/index.md) (topologies, link counts, cluster axes) and [File 2](./02_weight_sharding_strategies.md) of this chapter (which CCL ops follow which matmul patterns).

## TT_CCL: The Semaphore Manager

`TT_CCL` is instantiated once per mesh device via `DistributedConfig.__post_init__()` and provides semaphore handles for all CCL operations [ccl.py:TT_CCL].

### Initialization

```python
class TT_CCL:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.sub_device_crs = ttnn.CoreRangeSet({
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(
                    mesh_device.compute_with_storage_grid_size().x - 1,
                    mesh_device.compute_with_storage_grid_size().y - 1,
                ),
            )
        })

        # Three pools: [cluster_axis_0, cluster_axis_1, no_cluster_axis]
        self.barrier_semaphore_handles = [[], [], []]
        self.ag_semaphore_handles = [[], [], []]
        self.rs_semaphore_handles = [[], [], []]

        # Double-buffered semaphores for each pool
        for i in range(3):
            for _ in range(2):  # double buffer
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(mesh_device, self.sub_device_crs, 0)
                )
                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(...) for _ in range(2)]
                )
                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(...) for _ in range(3)]
                )
```

The `sub_device_crs` (core range set) spans the entire compute grid, making semaphores accessible from all compute cores. The structure is organized along three axes:

- Index 0: cluster axis 0 (vertical, North-South communication)
- Index 1: cluster axis 1 (horizontal, East-West communication)
- Index 2: no explicit cluster axis (1D mesh, e.g., N300/T3K)

For each axis, there are two sets of semaphores (double-buffered):

```
Per axis (0, 1, or None):
  barrier:        2 semaphores (double-buffered)
  all_gather:     2 x [2 semaphores] = 4 semaphores
  reduce_scatter: 2 x [3 semaphores] = 6 semaphores
                                        ----
Total per axis:                         12 semaphores
Total across 3 axes:                    36 semaphores
```

### Semaphore Cycling

Each `get_and_cycle_*` method returns the current semaphore handle(s) and advances the index to the next buffer [ccl.py:TT_CCL.get_and_cycle_barrier_semaphore_handle]:

```python
def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
    semaphore_index = 2 if not cluster_axis else cluster_axis
    current_idx = self.barrier_semaphore_idx[semaphore_index]
    self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % 2
    return self.barrier_semaphore_handles[semaphore_index][current_idx]
```

The modular cycling (`% 2`) ensures consecutive CCL operations on the same axis alternate between buffer 0 and buffer 1, preventing a semaphore from being reused before the previous operation completes:

```
Layer N:   RS uses semaphore_set[0]  ---->  completes
Layer N+1: RS uses semaphore_set[1]  ---->  completes
Layer N+2: RS uses semaphore_set[0]  ---->  (set[0] is free, safe to reuse)
```

> **Warning:** The cycling logic uses `not cluster_axis` which evaluates to `True` for both `None` and `0`. This means cluster axis 0 and no-cluster-axis share the same semaphore pool (index 2). This is intentional for N300/T3K topologies where there is only one meaningful communication axis, but could be surprising if you attempt to use axis 0 and axis-less operations simultaneously on a 2D mesh.

## get_num_links: Per-Topology Link Count

`get_num_links()` returns the number of ethernet links available for CCL operations per axis. For T3K, this is 1 link per hop on both axes -- a significant bandwidth constraint that directly affects CCL throughput. TG/Galaxy has asymmetric links (4 vertical, 3 horizontal). See [Chapter 1, File 1](../ch01_device_topologies/01_physical_topologies.md) for the full per-topology link table and query semantics. The `TT_CCL` instance exposes this as a method: `self.get_num_links(cluster_axis)`.

## Core CCL Primitives

### ttnn.reduce_scatter

Reduces (sums) partial results across devices along a dimension and scatters the result so each device holds a different slice:

```python
tt_output = ttnn.reduce_scatter(
    tt_output,
    dim=3,                              # Last dimension (hidden)
    num_links=1,                        # T3K has 1 link
    cluster_axis=1,                     # Horizontal axis
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Ring,        # Ring for trace compatibility
)
```

**Parameters:**
- `dim=3`: The tensor dimension along which reduction and scattering occur (last dim in the 4D reshaped tensor `[B, 1, S, H]`).
- `cluster_axis=1`: Communicate along the horizontal axis (columns) of the mesh. For T3K `(1, 8)`, all 8 devices participate.
- `topology=ttnn.Topology.Ring`: Ring communication pattern for trace compatibility.
- `num_links=1`: Ethernet links to use per hop.

**Input/Output shapes** (dim=3, 8 devices):
- Input: each device has `[1, 1, S, H]` (full hidden dim, partial sum)
- Output: each device has `[1, 1, S, H/8]` (reduced and scattered)

### ttnn.all_gather

Gathers tensor slices from all devices so every device has the full tensor:

```python
tt_output = ttnn.all_gather(
    tt_output,
    dim=3,
    num_links=1,
    cluster_axis=1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Ring,
)
```

**Input/Output shapes** (dim=3, 8 devices):
- Input: each device has `[1, 1, S, H/8]` (one slice)
- Output: each device has `[1, 1, S, H]` (full tensor, replicated)

### Combined: reduce_scatter + all_gather = Decomposed All-Reduce

```
                    Partial sums on each device
                    [1, 1, S, H] per device
                              |
                    reduce_scatter(dim=3)
                              |
                    [1, 1, S, H/8] per device
                    (reduced, each device has a unique slice)
                              |
                    all_gather(dim=3)
                              |
                    [1, 1, S, H] per device
                    (full result, replicated on all devices)
```

## tt_all_reduce: Topology-Adaptive Reduction

`tt_all_reduce()` is a high-level function that adapts the reduction strategy to the mesh topology [ccl.py:tt_all_reduce].

### Early-Return Guard

Before any reduction logic, `tt_all_reduce` checks whether the operation is a no-op:

```python
if mesh_shape == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
    return input_tensor
```

This guard returns the input unchanged in two cases: (1) the mesh is a single device `[1, 1]`, or (2) the caller requests reduction along `cluster_axis=1` and `1` appears anywhere in the mesh shape. On T3K with shape `(1, 8)`, `1 in list(mesh_device.shape)` is true (the row dimension is 1), so calling `tt_all_reduce` with `cluster_axis=1` hits this guard and returns immediately -- making it a no-op. This is significant because TT-Symbiote's distributed linear layers use `cluster_axis=1` for all CCL operations on T3K. In practice, Symbiote modules call `ttnn.reduce_scatter` and `ttnn.all_gather` directly rather than going through `tt_all_reduce`, so this guard does not affect Symbiote execution, but it would affect any code path that routes through `tt_all_reduce` on T3K with `cluster_axis=1`.

It then follows two code paths based on mesh shape:

### Path 1: N300 and T3K (1D meshes)

When `1 in list(mesh_device.shape)`, the mesh is 1D. The function uses `reduce_scatter_minimal_async` with semaphore management:

```python
if 1 in list(mesh_device.shape):
    reduced = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor,
        dim=dim,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_reduce_scatter_links,
        topology=topology,
    )
    return reduced
```

On 1D meshes, `reduce_scatter` alone suffices because the downstream layer expects sharded input.

### Path 2: TG/Galaxy (2D meshes)

For 2D meshes, two sub-paths exist:

**Non-composite (default):** `all_gather_async` + `fast_reduce_nc`
```python
gathered_tensor = ttnn.experimental.all_gather_async(...)
reduced_tensor = ttnn.experimental.fast_reduce_nc(gathered_tensor, dims=[dim], ...)
```

**Composite:** `reduce_scatter_minimal_async` + `all_gather_async`
```python
reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(...)
reduced_tensor = ttnn.experimental.all_gather_async(...)
```

The composite path mirrors the decomposition used by `TTNNLinearIColShardedWAllReduced` and is preferred for trace compatibility.

> **Warning:** `tt_all_reduce` for 1D meshes returns a **sharded** result (only `reduce_scatter`, no `all_gather`), which differs from the mathematical definition of all-reduce that returns the full reduced result to every device. The Symbiote distributed linear layers handle this by explicitly calling both `reduce_scatter` and `all_gather` when they need a replicated result.

> **Warning:** `tt_all_reduce` from `models/tt_transformers/tt/ccl.py` is designed for tt-transformers (non-Symbiote) model implementations. TT-Symbiote modules call `ttnn.reduce_scatter` and `ttnn.all_gather` directly rather than going through this wrapper.

## Trace-Compatible CCL Decomposition

The trace compatibility requirement affects every distributed module.

**The problem:** `ttnn.all_reduce` allocates an intermediate buffer dynamically during execution. TTNN trace capture records buffer addresses. On replay, the dynamic allocation may produce a different address, causing the trace to write to the wrong memory.

**The solution:** Decompose `all_reduce` into `reduce_scatter` + `all_gather`. Both operations use pre-allocated output buffers whose addresses are stable across trace replays.

This pattern appears in three places:
1. `TTNNLinearIColShardedWAllReduced.forward()` -- for linear layers needing replicated output
2. `TTNNDistributedRMSNorm.forward()` -- for gathering normalization statistics (uses `all_gather` alone)
3. `TTNNGemma4Attention._maybe_all_gather()` -- for gathering KV heads across devices

All three use `ttnn.Topology.Ring` rather than `ttnn.Topology.Linear`.

## cluster_axis Parameter

The `cluster_axis` parameter selects the communication direction within the 2D mesh:

```
            cluster_axis = 1 (horizontal, East-West)
            =========================>

    +------+------+------+------+------+------+------+------+
    | D0   | D1   | D2   | D3   | D4   | D5   | D6   | D7   |  Row 0
    +------+------+------+------+------+------+------+------+

    ||                                                      ||
    ||  cluster_axis = 0 (vertical, North-South)            ||
    \/                                                      \/
```

- **cluster_axis=0:** Communication between devices in the same column (vertical). Each column participates as a group.
- **cluster_axis=1:** Communication between devices in the same row (horizontal). Each row participates.

For T3K with shape `(1, 8)`, only `cluster_axis=1` is meaningful (there is only one row). In TT-Symbiote's distributed linear layers, `cluster_axis=1` is hardcoded because the hidden dimension is sharded along mesh columns (the default `(0, -1)` sharding from `DistributedConfig`).

For TG/Galaxy `(8, 4)`, both axes are useful: axis 0 spans 8 rows with 4 links, axis 1 spans 4 columns with 3 links.

## Topology: Ring vs Linear

Two communication topologies are supported:

**ttnn.Topology.Ring:** Devices form a circular chain. Data flows around the ring, each device sending to and receiving from its neighbors. Requires N-1 steps for N devices. Uses deterministic buffer allocation -- compatible with TTNN trace capture.

**ttnn.Topology.Linear:** Devices form a linear chain. Bidirectional communication possible. May allocate dynamic intermediate buffers, making it **incompatible with trace capture**.

```
Ring topology (T3K, 8 devices):

    +---+    +---+    +---+    +---+
    | 0 |--->| 1 |--->| 2 |--->| 3 |
    +---+    +---+    +---+    +---+
      ^                          |
      |                          v
    +---+    +---+    +---+    +---+
    | 7 |<---| 6 |<---| 5 |<---| 4 |
    +---+    +---+    +---+    +---+

Linear topology:

    +---+    +---+    +---+    +---+    +---+    +---+    +---+    +---+
    | 0 |<-->| 1 |<-->| 2 |<-->| 3 |<-->| 4 |<-->| 5 |<-->| 6 |<-->| 7 |
    +---+    +---+    +---+    +---+    +---+    +---+    +---+    +---+
```

The practical rule for TT-Symbiote module authors: **always use Ring topology for CCL operations in modules that may run under TracedRun** (modules decorated with `@trace_enabled` or called by trace-enabled parents).

> **Warning:** Using `ttnn.Topology.Linear` inside any module that will run under `TracedRun` will cause trace capture to succeed but replay to silently produce incorrect results (or crash) because buffer addresses shift between capture and replay. Always use `ttnn.Topology.Ring` in traced code paths.

## Distributed RMSNorm CCL Pattern

`tt_distributed_rmsnorm()` in `ccl.py` demonstrates the standard pattern for cross-device normalization [ccl.py:tt_distributed_rmsnorm]:

```python
def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, tt_ccl, ...):
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, ...)      # per-device partial stats
    tt_stats = tt_all_gather(tt_stats, mesh_device, tt_ccl,
                             dim=3, cluster_axis=1, ...)    # gather from all devices
    tt_out = ttnn.rms_norm_post_all_gather(inp, tt_stats,
                                           epsilon=epsilon,
                                           weight=gamma, ...)  # apply with global stats
    return tt_out
```

This three-phase pattern (compute local statistics, gather globally, apply normalization) is reused by `TTNNDistributedRMSNorm` in [File 4](./04_distributed_modules.md), though the Symbiote implementation calls `ttnn.all_gather` directly rather than going through the `tt_all_gather` wrapper.

A sharded variant `tt_sharded_distributed_rmsnorm()` exists for decode-mode performance on L1-resident tensors, using `ttnn.experimental.all_gather_async` with explicit memory configs. This variant uses `ttnn.Topology.Linear` and is designed for non-traced execution paths in tt-transformers models.

## CCL Operation Costs on T3K

On T3K (1 link per hop, 8 devices, Ring topology):

| Operation | Data Moved Per Device | Total Hops |
|-----------|----------------------|------------|
| `reduce_scatter(dim=3)` | `tensor_size * 7/8` (sends 7 chunks) | 7 |
| `all_gather(dim=3)` | `tensor_size * 7/8` (receives 7 chunks) | 7 |
| Decomposed all-reduce | Both of the above | 14 |

With a single ethernet link at ~12.5 GB/s per direction (Wormhole), a 4096-dimensional bfloat16 tensor (8 KB per token-position) traversing `reduce_scatter` + `all_gather` incurs approximately `2 * 7 * 8KB / 12.5 GB/s ~ 9 us` of CCL latency per layer per token. For a 60-layer model, this accumulates to ~0.5 ms of pure CCL overhead per token, which is typically small relative to the matmul compute time on T3K but can dominate for short-sequence decode.

For a per-module breakdown of CCL operations in a single transformer block, see [File 4, CCL Operations Per Transformer Block](./04_distributed_modules.md). The total is **6 CCL operations per Gemma4 block** (2x decomposed all-reduce + 2x all_gather for RMSNorm).

## Summary: Which CCL Operation Where

| Context | Operation | Topology | Trace-Safe |
|---------|-----------|----------|------------|
| Row-parallel linear output | `reduce_scatter` | Ring | Yes |
| All-reduce variant linear | `reduce_scatter` + `all_gather` | Ring | Yes |
| Distributed RMSNorm stats | `all_gather` | Ring | Yes |
| Gemma4 attention KV gather | `all_gather` | Ring | Yes |
| MoE forward (Qwen3) | `all_gather` + `reduce_scatter` | Linear/Ring | Varies |
| tt_all_reduce (1D mesh) | `reduce_scatter_minimal_async` | Linear | No |
| tt_all_reduce (2D mesh) | `all_gather_async` + `fast_reduce_nc` | Linear | No |

The Qwen3 MoE module is notable for using `ttnn.Topology.Linear` for the initial `all_gather` (to replicate input before routing) and `ttnn.Topology.Ring` for the final `reduce_scatter`. The all-to-all operations used by `TTNNQwenExperts` are distinct from the standard gather/scatter primitives and are covered in [File 4](./04_distributed_modules.md).

---

**Next:** [`04_distributed_modules.md`](./04_distributed_modules.md)
