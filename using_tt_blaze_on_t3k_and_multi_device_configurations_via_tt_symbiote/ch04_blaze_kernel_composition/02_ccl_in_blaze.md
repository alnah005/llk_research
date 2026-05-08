# 02 -- CCL Operations at the Blaze Kernel Level

This file covers how collective communication library (CCL) operations are implemented at the kernel level in TT-Blaze. Where [Chapter 3, File 03](../ch03_symbiote_multi_device/03_ccl_operations.md) described CCL operations from the ttnn API perspective (all-gather, reduce-scatter, all-reduce as black-box TTNN calls), this file descends to the Blaze kernel layer: `setup_fabric()` in `blaze/ccl.py` and its role in establishing inter-device fabric connections, fabric semaphore allocation, fabric runtime argument computation, the RISC processor model for CCL data movement, the four CCL MicroOps (AllReduce, CclBroadcast, ReduceToOne, Scatter), and the `injected_fabric_barrier_builder.py` that automatically inserts inter-CCL synchronization into multi-CCL fused programs.

---

## 1. Fabric Setup -- setup_fabric()

The `setup_fabric()` function [blaze/ccl.py, line 18] is the single entry point for connecting a Blaze kernel to the inter-device ethernet fabric. Every CCL op calls it during `emit()` to set up its fabric connections.

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

### 1.1 What setup_fabric() Does

The function performs five operations in sequence:

```
  setup_fabric(fp, prefix="all_reduce", worker_core, risc=Risc.NCRISC,
               dst_nodes=[node_fwd, node_bwd])

  Step 1: Allocate program semaphores (per-connection)
  +------------------------------------------------------+
  | teardown_sems = [fp.semaphore(program_semaphore=True, |
  |                   core_ranges=worker_core_ranges)     |
  |                  for _ in range(num_connections)]     |
  | buf_idx_sems  = [same pattern]                        |
  +------------------------------------------------------+

  Step 2: Compute fabric RT args
  +------------------------------------------------------+
  | coord = ttnn.MeshCoordinate(row, col)                 |
  | rt_args = ttnn.compute_fabric_connection_rt_args(     |
  |     mesh_device.get_fabric_node_id(coord),            |
  |     dst_nodes, link_indices, teardown_sems,           |
  |     buf_idx_sems,                                     |
  | )                                                     |
  +------------------------------------------------------+

  Step 3: Set per-core runtime args on the target RISC
  +------------------------------------------------------+
  | ncrisc_prev_len, _, brisc_prev_len =                  |
  |   fp._set_per_core_runtime_args(worker_core,          |
  |       **{RISC_NAMES[risc]: rt_args})                  |
  +------------------------------------------------------+

  Step 4: Add fabric kernel defines
  +------------------------------------------------------+
  | for name, val in ttnn.get_fabric_kernel_defines(      |
  |     "Linear"):                                        |
  |     fp.add_define(name, val)                           |
  +------------------------------------------------------+

  Step 5: Track fabric cores for barrier injection
  +------------------------------------------------------+
  | fp._fabric_cores_per_risc_per_ccl[prefix][risc]       |
  |     .append(worker_core)                              |
  +------------------------------------------------------+

  Returns: previous length of the RISC's per-core RT args
           (used as fabric_arg_idx by callers)
```

### 1.2 Semaphore Allocation Pattern

Each fabric connection requires two semaphores scoped to the worker core:

| Semaphore     | Purpose                                                       |
|--------------|---------------------------------------------------------------|
| teardown_sem | Signals the fabric connection teardown sequence               |
| buf_idx_sem  | Tracks which double-buffer slot is currently active           |

These are **program semaphores** (slot-based, per-device) rather than global mesh semaphores, because they only need local synchronization on the single worker core that owns the fabric connection. The `core_ranges` parameter restricts each semaphore to the single worker core, letting tt-metal reuse slot IDs across disjoint cores -- important when multiple CCL ops in a fused kernel each have their own fabric worker core.

```python
worker_core_ranges = ttnn.CoreRangeSet([
    ttnn.CoreRange(worker_core, worker_core)
])
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

[blaze/ccl.py:setup_fabric, line 49]

### 1.3 ttnn.compute_fabric_connection_rt_args()

This TTNN API computes the runtime arguments needed by the fabric kernel:

```python
coord = ttnn.MeshCoordinate(row, col)
rt_args = ttnn.compute_fabric_connection_rt_args(
    mesh_device.get_fabric_node_id(coord),   # this device's fabric ID
    dst_nodes,                                 # destination fabric IDs
    link_indices or [0],                       # which ethernet links to use
    teardown_sems,                             # teardown sync semaphores
    buf_idx_sems,                              # buffer index semaphores
)
```

The returned `rt_args` is an opaque list of integers that the fabric kernel reads at runtime to configure its DMA operations: NOC addresses of remote fabric endpoints, link configuration, and semaphore addresses.

### 1.4 Per-Core RT Args and RISC Targeting

The runtime args are injected on a specific RISC processor:

```python
assert risc != Risc.TRISC, "Can only connect to fabric using data movement RISCs"
ncrisc_prev_len, _, brisc_prev_len = fp._set_per_core_runtime_args(
    worker_core,
    **{RISC_NAMES[risc]: rt_args}
)
```

[blaze/ccl.py:setup_fabric, line 67]

The returned `prev_len` is the starting index of the fabric args within the RISC's per-core RT args array. CCL ops pass this as `fabric_arg_idx` in their compile-time args so the kernel knows where to find the connection data.

> **Warning:** Fabric connections can only be established on data movement RISCs (NCRISC or BRISC). TRISC handles compute only and has no NOC access for fabric DMA. Attempting to set `risc=Risc.TRISC` raises an assertion error.

### 1.5 Fabric Kernel Defines

```python
for name, val in ttnn.get_fabric_kernel_defines("Linear"):
    fp.add_define(name, val)
```

The `"Linear"` topology identifier selects defines for linear (non-ring) fabric communication. These preprocessor defines are injected into the kernel compilation and control the fabric communication code paths in the C++ kernel. The topology mode matches the fabric configuration described in [Chapter 1, File 03](../ch01_device_topologies/03_fabric_and_routing.md).

### 1.6 Barrier Tracking

```python
fp._fabric_cores_per_risc_per_ccl.setdefault(prefix, {}).setdefault(risc, []).append(worker_core)
```

This records which cores are used by which CCL op on which RISC. The `_fabric_cores_per_risc_per_ccl` dict is the input to the injected fabric barrier builder (Section 5), which needs to know the exact core-to-RISC mapping to insert synchronization between consecutive CCL operations.

### 1.7 Receiver Path

When `dst_nodes` is empty (the device is a receiver, not a sender), `setup_fabric()` creates an empty per-core RT args slot but does not allocate semaphores or compute fabric args:

```python
if len(dst_nodes) == 0:
    _, _, _ = fp._set_per_core_runtime_args(worker_core)
    return
```

The receiver's connection is established by the sender's fabric DMA -- the receiver passively accepts data at the target L1 address.

---

## 2. RISC Processor Model for CCL

Tenstorrent Blackhole cores have three RISC processors. CCL ops assign specific roles to each:

```
  Per-Core RISC Layout for CCL Operations
  +-----------------------------------------------------------+
  |  NCRISC  (Data Movement - NOC Reader)                      |
  |  - Reads data from local L1 or remote L1 via NOC          |
  |  - Fabric sender: writes to ethernet fabric                |
  |  - Fabric receiver: reads from ethernet fabric             |
  +-----------------------------------------------------------+
  |  BRISC   (Data Movement - NOC Writer / Multicast)          |
  |  - Multicast: broadcasts data to multiple cores            |
  |  - Can also be used as fabric sender/receiver              |
  |  - Handles NOC writes for scatter/gather patterns          |
  +-----------------------------------------------------------+
  |  TRISC   (Compute)                                         |
  |  - Matrix multiply, element-wise, reductions               |
  |  - NEVER used for fabric connections                       |
  |  - Performs local reduction after data arrives              |
  +-----------------------------------------------------------+
```

The split is enforced by `setup_fabric()` which only accepts `Risc.NCRISC` or `Risc.BRISC` for fabric connections. The kernel code is structured as a single `.cpp` file with `#ifdef` guards that compile different logic per RISC:

- **NCRISC/BRISC sections:** fabric connection setup, DMA send/receive, semaphore signaling
- **TRISC sections:** local elementwise operations, CB push/pop for compute pipeline

Per-core role flags (`is_sender`, `is_receiver`, `execute_on_ncrisc`, `execute_on_brisc`) ensure that each RISC on each core only executes the code path relevant to its role in the CCL operation.

For example, in AllReduce:
- **NCRISC** on the sender core: reads local data and sends it via fabric to the neighbor.
- **NCRISC/BRISC** on the receiver core: receives fabric data into the intermediate buffer.
- **TRISC** on all cores: performs the local sum reduction after data arrives.

---

## 3. CCL Ops in Blaze

Blaze implements four CCL MicroOps, each building on `setup_fabric()`:

### 3.1 AllReduce

[blaze/ops/all_reduce/op.py:AllReduce]

AllReduce performs a ring-based sum reduction across devices along a specified cluster axis. Each device sends its data to its neighbor and accumulates received data:

```
  AllReduce: ring-based reduction across mesh devices

  Device 0          Device 1          Device 2          Device 3
  +--------+        +--------+        +--------+        +--------+
  | data_0 |--fwd-->| data_1 |--fwd-->| data_2 |--fwd-->| data_3 |
  +--------+        +--------+        +--------+        +--------+
       |                 |                 |                 |
       v                 v                 v                 v
  local_sum(         local_sum(       local_sum(       local_sum(
    data_0,            data_1,           data_2,          data_3,
    recv_from_3)       recv_from_0)      recv_from_1)     recv_from_2)
```

**Port structure:**

```python
class AllReduce(MicroOp):
    name: str = "all_reduce"

    sender_cb_in0: Internal = Internal()    # scratch for outgoing data
    receiver_cb_in0: Input = Input()        # intermediate accumulation buffer
    receiver_cb_in1: Input = Input()        # input activation
    receiver_cb_out0: Output = Output()     # reduced output
    receiver_cb_residual: Input = Input()   # optional residual connection
```

AllReduce's `emit()`:

1. Determines `ring_index` and `is_first_chip` from `mesh_coord` and `cluster_axis`.
2. Allocates input, intermediate, and output CBs.
3. Calls `setup_fabric()` for the forward fabric connection (NCRISC).
4. Calls `setup_fabric()` for the receiver path (empty `dst_nodes`).
5. Sets unified CT args: CB IDs, tile counts, page sizes, ring position, residual add flag.
6. Optionally fuses a residual addition into the reduction.

The `cluster_axis` parameter determines the ring direction (0 = along rows, 1 = along columns). The `is_first_chip` flag (derived from `mesh_coord`) controls whether the device initializes the ring or waits for incoming data.

### 3.2 CclBroadcast

[blaze/ops/ccl_broadcast/op.py:CclBroadcast]

CclBroadcast is a multi-device fabric broadcast from a single sender to all devices in the mesh. It supports dual-axis broadcast on 2D meshes (e.g., 4x2).

**Routing computation** [blaze/ops/ccl_broadcast/op.py:compute_routing]:

The `compute_routing()` function determines each device's role (sender, secondary_sender, or receiver) and computes forward/backward hop counts:

```python
@dataclass
class BroadcastRouting:
    is_sender: bool              # this device originates the broadcast
    is_secondary_sender: bool    # relay sender on secondary axis
    is_receiver: bool            # pure receiver
    num_fwd: int                 # forward hops
    num_bwd: int                 # backward hops
    has_secondary: int           # 0 or 1 (secondary axis relay)
    enable_torus: bool           # torus wrap-around optimization
    ring_index: int              # position in the ring
    num_connections: int         # total fabric connections
```

For torus topologies, forward and backward chains are balanced: `(mesh_rows - 1) // 2` hops forward, the remainder backward.

**Data flow on a 4x2 mesh (sender at (1,0), secondary_cluster_axis=1):**

```
  +-------+-------+
  | (0,0) | (0,1) |
  | recv  | recv  |
  +-------+-------+
  | (1,0) | (1,1) |
  | SEND  | 2nd   |  <- sender_coord=(1,0), secondary at (1,1)
  +-------+-------+
  | (2,0) | (2,1) |
  | recv  | recv  |
  +-------+-------+
  | (3,0) | (3,1) |
  | recv  | recv  |
  +-------+-------+

  Data flow:
  sender (1,0) --fwd--> (2,0) --> (3,0)   (forward chain, column 0)
  sender (1,0) --bwd--> (0,0)             (backward chain, column 0)
  sender (1,0) --sec--> (1,1)             (secondary: cross-column)
  secondary (1,1) --fwd/bwd--> column 1   (secondary fan-out)
```

**Emit flow** [blaze/ops/ccl_broadcast/op.py:CclBroadcast.emit]:

```python
@staticmethod
def emit(f, src, output_tensor, *, prefix, sender_coord, ...):
    # 1. Compute per-device routing
    routing = compute_routing(row, col, sender_coord, mesh_shape, ...)

    # 2. Allocate CBs
    src_handle = f.cb_from_tensor(src)
    f.cb_from_tensor(output_tensor)  # register for lifetime

    # 3. Allocate semaphores (mesh-global, deduped by name)
    out_sem = f.semaphore(f"{prefix}.out_sem")
    bar_sem = f.semaphore(f"{prefix}.bar_sem")
    sec_sem = f.semaphore(f"{prefix}.sec_sem")

    # 4. Set compile-time args (routing config)
    f.unified_ct_args([
        (f"{prefix}.num_targets_forward_direction", routing.num_fwd),
        (f"{prefix}.num_targets_backward_direction", routing.num_bwd),
        (f"{prefix}.has_secondary_target", routing.has_secondary),
        ...
    ])

    # 5. Per-core role flags (only broadcast core is active)
    f.per_core_unified_ct_args([
        (f"{prefix}.is_sender", {worker_crs: int(routing.is_sender)}),
        (f"{prefix}.is_receiver", {worker_crs: int(routing.is_receiver)}),
    ])

    # 6. Runtime args (tensor addresses, semaphore L1 addrs)
    f.ncrisc_rt_args([
        (f"{prefix}.tensor_address0", int(output_tensor.buffer_address())),
        (f"{prefix}.out_ready_sem_bank_addr", out_sem),
        ...
    ])

    # 7. Setup fabric connections (sender/secondary only)
    if routing.is_sender or routing.is_secondary_sender:
        dst_nodes = compute_dst_nodes(row, col, routing, ...)
        fabric_arg_idx = setup_fabric(f, prefix=prefix,
                                       worker_core=fabric_core,
                                       risc=Risc.NCRISC,
                                       dst_nodes=dst_nodes)
```

Key points:
- The broadcast runs on a **single worker core per device** (the core at the tensor's shard start position).
- Fabric connections use **NCRISC** for DMA.
- Role flags (`is_sender`, `is_receiver`) are per-core CT args so only the broadcast core executes CCL logic; all other cores in the fused kernel see 0 and skip.
- Semaphores are **mesh-global** (deduped by name) so all devices see the same L1 addresses.

### 3.3 ReduceToOne

[blaze/ops/reduce_to_one/op.py:ReduceToOne]

ReduceToOne implements a tree reduction across the mesh, collapsing partial results from all devices onto a single root device. It supports TP=2, 4, and 8 configurations.

The topology is computed by `get_reduction_topology()` which determines `(num_receiving_stages, is_root)` for each device:

```
  ReduceToOne Tree Reduction (TP=8, 4x2 mesh)

  Stage 1: Leaf devices send to their column partner
  +-----------+         +-----------+
  | (0,0)     |-------->| (3,0)     |   Row 0 sends to Row 3
  +-----------+         +-----------+
  | (1,0)     |-------->| (2,0)     |   Row 1 sends to Row 2
  +-----------+         +-----------+
  | (0,1)     |-------->| (3,1)     |   Row 0 sends to Row 3 (col 1)
  +-----------+         +-----------+
  | (1,1)     |-------->| (2,1)     |   Row 1 sends to Row 2 (col 1)
  +-----------+         +-----------+

  Stage 2: Stage-1 receivers reduce across columns
  +-----------+         +-----------+
  | (3,0)     |         | (3,1)     |
  |  sum(0,3) |-------->|  not used |
  +-----------+         +-----------+
  | (2,0)     |         | (2,1)     |
  |  sum(1,2) |         |  sum(1,2) |
  +-----------+         +-----------+

  Stage 3: Final reduction to root
  +-----------+
  | (2,0)     |  <- ROOT: receives from (3,0), has sum of all 8 devices
  +-----------+
```

The multi-stage structure avoids linear chain latency by parallelizing reductions across independent branches. For TP=2, it reduces to a single send; for TP=4, two stages.

### 3.4 Scatter

[blaze/ops/scatter/op.py:Scatter]

Scatter distributes data from source core(s) to multiple destination cores within a single device. It is the intra-device inverse of Gather and uses explicit `dest_mapping`:

```python
Scatter.emit(f, src, dest_tensor,
    prefix="scatter",
    dest_mapping={
        CoreCoord(12, 9): [CoreCoord(1, 0), CoreCoord(2, 0), ...],
    },
    tiles_per_dest=4,
)
```

Each source core sends `tiles_per_dest` tiles to each of its destination cores via NOC writes. The mapping is explicit because scatter patterns depend on the specific weight sharding and core grid layout of the enclosing FusedOp.

---

## 4. Fabric Semaphore Management

CCL operations use three categories of semaphores:

### 4.1 Mesh-Global Semaphores (Named)

Created via `f.semaphore("name")`. Backed by `ttnn.GlobalSemaphore` objects allocated on the mesh device. The `_alloc_mesh_semaphore()` function [blaze/fused_program.py:_alloc_mesh_semaphore] deduplicates by name across per-device compilation iterations:

```python
def _alloc_mesh_semaphore(mesh_device, name, sem_dict, initial_value=0):
    if name not in sem_dict:
        gs = mesh_device.compute_with_storage_grid_size()
        avail = ttnn.num_cores_to_corerangeset(gs.x * gs.y, gs, row_wise=True)
        sem_dict[name] = ttnn.create_global_semaphore(mesh_device, avail, initial_value)
    return sem_dict[name]
```

The `sem_dict` is shared between the `BlazeCompiler` and all `FusedProgram` instances within one `compile()` call, ensuring that the same semaphore name resolves to the same L1 address on every device. The L1 address is obtained via `ttnn.get_global_semaphore_address(sem)` and passed as a CT arg or RT arg to the kernel.

Typical semaphore names in CCL ops:

| Name | Purpose |
|------|---------|
| `{prefix}.out_sem` | Output-ready signaling (sender to receiver) |
| `{prefix}.bar_sem` | Barrier synchronization between phases |
| `{prefix}.sec_sem` | Secondary sender synchronization (cross-column) |

### 4.2 Program Semaphores (Slot-Based)

Created via `f.semaphore(program_semaphore=True, core_ranges=...)`. These are per-device, slot-based semaphores managed by tt-metal. Used by `setup_fabric()` for fabric teardown and buffer index tracking where only local (same-device) synchronization is needed.

### 4.3 Injected Barrier Semaphores

Created by the injected fabric barrier builder (Section 5). Named with the pattern `injected_fabric_barrier.ncrisc_semaphore_{idx}` and `injected_fabric_barrier.brisc_semaphore_{idx}`. These synchronize the handoff between consecutive CCL operations within a fused kernel.

---

## 5. The Injected Fabric Barrier Builder

When a fused kernel contains multiple CCL operations (e.g., AllReduce followed by CclBroadcast), the fabric hardware on each device must not be reconfigured while an operation is in progress. The `injected_fabric_barrier_builder.py` [blaze/injected_fabric_barrier_builder.py] automatically inserts synchronization barriers between consecutive CCL operations.

### 5.1 Problem Statement

```
  Without barriers:
  CCL_A (AllReduce) starts fabric DMA on NCRISC
  CCL_B (CclBroadcast) starts fabric DMA on NCRISC  <-- RACE!
  Both try to use the same fabric link simultaneously.

  With barriers:
  CCL_A (AllReduce) starts fabric DMA on NCRISC
  BarrierSender: signals "CCL_A done" to CCL_B's fabric cores
  BarrierReceiver: waits for signal before starting CCL_B
  CCL_B (CclBroadcast) starts fabric DMA on NCRISC  <-- SAFE
```

### 5.2 The prepare_for_build() Entry Point

```python
def prepare_for_build(fp: FusedProgram) -> None:
    if fp._injected_fabric_barrier_applied:
        return
    elif len(fp._fabric_cores_per_risc_per_ccl) == 0:
        return
    fp._injected_fabric_barrier_applied = True

    role_counts = {BarrierSender.name: 0, BarrierReceiver.name: 0}
    semaphore_addrs = []

    _patch_all_instances(fp, role_counts, semaphore_addrs)
    _append_final_barrier_receiver(fp, role_counts[BarrierReceiver.name],
                                    semaphore_addrs)
```

[blaze/injected_fabric_barrier_builder.py:prepare_for_build, line 21]

This is called automatically during `FusedProgram._prepare_for_build()`, before the program is finalized for hardware. It is idempotent (the `_applied` flag prevents re-entry) and skips entirely if no CCL ops were registered.

### 5.3 Barrier Tracking via _fabric_cores_per_risc_per_ccl

The key data structure enabling barrier injection is `FusedProgram._fabric_cores_per_risc_per_ccl`:

```python
# dict[prefix, dict[Risc, list[CoreCoord]]]
# Example after two CCL ops:
{
    "all_reduce": {
        Risc.NCRISC: [CoreCoord(12, 9)],
    },
    "ccl_broadcast": {
        Risc.NCRISC: [CoreCoord(12, 9)],
        Risc.BRISC: [CoreCoord(12, 9)],
    },
}
```

`setup_fabric()` populates this dict (line 72 of ccl.py):

```python
fp._fabric_cores_per_risc_per_ccl.setdefault(prefix, {}) \
    .setdefault(risc, []).append(worker_core)
```

The ordered insertion (Python 3.7+ dict ordering) preserves the CCL execution order, which the barrier builder relies on to determine which CCL precedes which.

### 5.4 Barrier Injection Pattern

For N CCL operations, the builder inserts N-1 sender-receiver pairs plus one final receiver:

```
  CCL_0 (AllReduce)
     |
  BarrierSender_0   -- signals CCL_1's fabric cores
  BarrierReceiver_0  -- CCL_1's cores wait
     |
  CCL_1 (CclBroadcast)
     |
  BarrierSender_1   -- signals CCL_0's fabric cores (wrap-around)
  BarrierReceiver_final  -- CCL_0's cores wait (for next iteration)
```

### 5.5 Sender/Receiver CT Arg Patching

The `_patch_all_instances()` function walks the shadow graph and patches BarrierSender/BarrierReceiver nodes with their CT args:

**For BarrierSender** [blaze/injected_fabric_barrier_builder.py:_compute_new_unified_ct_arg_values]:

```python
def _compute_new_unified_ct_arg_values(fp, op_name, role_idx, semaphore_addrs):
    if op_name == BarrierSender.name:
        next_ccl_idx = (role_idx + 1) % len(fp._fabric_cores_per_risc_per_ccl)
        next_ccl_fabric_cores_per_risc = list(
            fp._fabric_cores_per_risc_per_ccl.values()
        )[next_ccl_idx]

        send_to_ncrisc = bool(next_ccl_fabric_cores_per_risc.get(Risc.NCRISC))
        send_to_brisc = bool(next_ccl_fabric_cores_per_risc.get(Risc.BRISC))

        new_semaphore_addrs = (
            fp.semaphore(f"injected_fabric_barrier.ncrisc_semaphore_{role_idx}")
                if send_to_ncrisc else 0,
            fp.semaphore(f"injected_fabric_barrier.brisc_semaphore_{role_idx}")
                if send_to_brisc else 0
        )
```

| CT Arg                     | Value                                           |
|---------------------------|------------------------------------------------|
| `num_receiver_cores`       | Number of cores in the next CCL's fabric set   |
| `send_to_ncrisc`          | Whether the next CCL uses NCRISC fabric        |
| `send_to_brisc`           | Whether the next CCL uses BRISC fabric         |
| `ncrisc_semaphore_l1_addr` | L1 address of the NCRISC barrier semaphore     |
| `brisc_semaphore_l1_addr`  | L1 address of the BRISC barrier semaphore      |

**For BarrierReceiver:**

```python
elif op_name == BarrierReceiver.name:
    prev_ccl_fabric_cores_per_risc = list(
        fp._fabric_cores_per_risc_per_ccl.values()
    )[prev_ccl_idx]

    return {
        "ncrisc_semaphore_l1_addr": semaphore_addrs[role_idx][0],
        "brisc_semaphore_l1_addr": semaphore_addrs[role_idx][1],
        "wait_value": sum(len(cores) for cores in
                         prev_ccl_fabric_cores_per_risc.values()),
    }
```

The `wait_value` equals the total number of sender cores from the previous CCL -- the receiver spins until its semaphore reaches this count.

### 5.6 Runtime Arg Patching for Sender NOC Coordinates

The barrier builder also patches per-core runtime arg arrays so the sender kernel knows the physical NOC coordinates of each receiver core:

```python
def _compute_new_rt_arg_array_values(fp, op_name, role_idx):
    if op_name == BarrierSender.name:
        receiver_cores = list({c for cores in
                               next_ccl_fabric_cores_per_risc.values()
                               for c in cores})
        receiver_core_noc_coordinates = []
        for receiver_core in receiver_cores:
            phys = fp.device.worker_core_from_logical_core(receiver_core)
            receiver_core_noc_coordinates.append(phys.x)
            receiver_core_noc_coordinates.append(phys.y)

        return {"receiver_core_noc_coordinates": receiver_core_noc_coordinates}
```

[blaze/injected_fabric_barrier_builder.py:_compute_new_rt_arg_array_values, line 95]

This logical-to-physical translation is essential because the barrier sender writes directly to the receiver's L1 via NOC, which requires physical coordinates.

### 5.7 The Final Barrier Receiver

After all barrier sender/receiver pairs are patched, a final `BarrierReceiver` is appended. It runs on the first CCL's fabric cores and waits for the last CCL's sender barrier:

```python
def _append_final_barrier_receiver(fp, role_idx, semaphore_addrs):
    prefix = f"{fp._injected_fabric_barrier_prefix}_receiver_final_barrier_receiver"

    ccl_idx = 0  # waits before the first CCL (for loop iteration)
    fabric_cores_per_risc = list(
        fp._fabric_cores_per_risc_per_ccl.values()
    )[ccl_idx]

    execute_on_ncrisc = bool(fabric_cores_per_risc.get(Risc.NCRISC))
    execute_on_brisc = bool(fabric_cores_per_risc.get(Risc.BRISC))
    receiver_cores = list({c for cores in fabric_cores_per_risc.values()
                           for c in cores})

    fp.program.per_core_unified_ct_args([
        fp.flag(f"{prefix}.execute_on_ncrisc", ..., enabled=execute_on_ncrisc),
        fp.flag(f"{prefix}.execute_on_brisc",  ..., enabled=execute_on_brisc),
    ])
```

[blaze/injected_fabric_barrier_builder.py:_append_final_barrier_receiver, line 148]

This ensures that if the fused kernel is executed in a loop (common for decoder iterations), the fabric is fully quiesced before the next iteration starts.

### 5.8 Barrier Lifecycle in a Decode Loop

```
  Iteration 0:
    [Final BarrierReceiver] <-- skipped (no previous iteration signal)
    [CCL_0: AllReduce]
    [BarrierSender_0] --> [BarrierReceiver_0]
    [CCL_1: CclBroadcast]
    [BarrierSender_1] --> signal stored for next iteration

  Iteration 1:
    [Final BarrierReceiver] <-- waits for BarrierSender_1 from iter 0
    [CCL_0: AllReduce]
    [BarrierSender_0] --> [BarrierReceiver_0]
    [CCL_1: CclBroadcast]
    [BarrierSender_1] --> signal stored for next iteration

  Iteration N:
    [Final BarrierReceiver] <-- waits for BarrierSender_1 from iter N-1
    ... (same pattern) ...
```

The skip-on-first-iteration behavior is implicit: the final barrier receiver's semaphore starts at 0 and `wait_value` equals the number of senders from the last CCL. On the first iteration, no sender has incremented the semaphore, so the wait is skipped (the kernel checks a flag that tracks whether this is the first execution).

---

## 6. CCL Operation Integration Within FusedOps

CCL ops are composed into larger FusedOps just like any other MicroOp. Here are two representative integration patterns:

### 6.1 BroadcastRMSNorm: CclBroadcast + RMSNorm

[blaze/ops/broadcast_rmsnorm/op.py:BroadcastRMSNorm]

```
  BroadcastRMSNorm compose():
      |
      +-> CclBroadcast.emit(f, src, output_tensor,
      |       prefix="ccl_broadcast",
      |       sender_coord=sender_coord)
      |   # Broadcasts activation from sender to all devices
      |   # Returns: CBHandle pointing to received data
      |
      +-> RMSNorm.emit(f, received_handle, gamma,
              prefix="rmsnorm")
          # Applies layer normalization on each device
          # Returns: normalized CBHandle
```

The CclBroadcast's output CBHandle is passed directly as input to RMSNorm -- no intermediate tensor allocation needed. Both ops run in a single kernel dispatch.

### 6.2 AllReduceMoE: AllReduce + MoE

[blaze/ops/allreduce_moe/op.py:AllReduceMoE]

```
  AllReduceMoE compose():
      |
      +-> AllReduce.emit(f, input, intermediate, output,
      |       prefix="all_reduce",
      |       cluster_axis=cluster_axis)
      |   # Ring-reduce across devices, writes to output tensor
      |   # Includes optional residual addition
      |
      +-> MoE.emit(f, output_as_activation,
              rmsnorm_gamma, gate_weight, gate_bias,
              expert_weights, shared_weights, ...)
          # Full MoE pipeline (RMSNorm -> Gate -> Experts -> Reduce)
```

The AllReduce writes its result to a shared tensor that MoE reads as its activation input. The fabric barrier builder automatically inserts synchronization between AllReduce and any subsequent CCL operations within MoE.

### 6.3 Multi-Device Context in compose()

CCL ops access mesh information through FusedProgram attributes set by the compiler:

```python
# Available inside any compose() or emit():
f.mesh_coord   # (row, col) -- this device's position
f.mesh_shape   # (num_rows, num_cols) -- mesh dimensions
f.mesh_device  # the MeshDevice handle (for fabric node lookups)
```

These are injected by `BlazeCompiler._compile_fused_op()` [blaze/compiler.py:BlazeCompiler._compile_fused_op, line 905] (mesh context injection at line 957):

```python
if "mesh_coord" in merged_args:
    f.mesh_coord = merged_args["mesh_coord"]
    f.mesh_shape = merged_args["mesh_shape"]
    f.mesh_device = merged_args["mesh_device"]
```

On a 1x1 mesh, `mesh_coord` is `(0, 0)` and CCL broadcast routing computes zero forward/backward hops, effectively making the broadcast a no-op.

---

## 7. Relationship to Symbiote-Level CCL

The Blaze CCL layer and the Symbiote CCL layer ([Chapter 3, File 03](../ch03_symbiote_multi_device/03_ccl_operations.md)) serve different purposes:

```
  Symbiote CCL (Chapter 3)              Blaze CCL (this file)
  +-------------------------------+     +-------------------------------+
  | ttnn.all_gather()             |     | AllReduce.emit(f, ...)        |
  | ttnn.reduce_scatter()         |     | CclBroadcast.emit(f, ...)     |
  | tt_all_reduce()               |     | ReduceToOne.emit(f, ...)      |
  |                               |     | setup_fabric()                |
  |                               |     |                               |
  | Black-box TTNN API calls      |     | Kernel-level composition      |
  | Each is a separate dispatch   |     | Multiple CCLs in one dispatch |
  | Inter-op gaps between CCLs    |     | Zero inter-op overhead        |
  | Topology handled by TTNN      |     | Explicit routing in Python    |
  | Used by Symbiote TTNNModules  |     | Used by Blaze FusedOps        |
  +-------------------------------+     +-------------------------------+
```

The Blaze approach trades simplicity for performance: by fusing CCL operations with compute ops in a single kernel, it eliminates dispatch overhead and enables the fabric barrier builder to overlap synchronization with computation. The Symbiote approach handles topology transparently but incurs a dispatch boundary between every CCL call and the surrounding compute.

---

## 8. Key Source Files Reference

| File | Purpose |
|------|---------|
| `blaze/ccl.py` | `setup_fabric()`: shared fabric connection setup for all CCL ops |
| `blaze/injected_fabric_barrier_builder.py` | Automatic inter-CCL barrier insertion |
| `blaze/ops/all_reduce/op.py` | `AllReduce` MicroOp: ring-based sum reduction |
| `blaze/ops/ccl_broadcast/op.py` | `CclBroadcast` MicroOp: multi-device fabric broadcast |
| `blaze/ops/reduce_to_one/op.py` | `ReduceToOne` MicroOp: tree reduction to single device |
| `blaze/ops/scatter/op.py` | `Scatter` MicroOp: intra-device data distribution |
| `blaze/ops/barrier_sender/op.py` | `BarrierSender` MicroOp: fabric barrier send |
| `blaze/ops/barrier_receiver/op.py` | `BarrierReceiver` MicroOp: fabric barrier wait |
| `blaze/ops/broadcast_rmsnorm/op.py` | `BroadcastRMSNorm` FusedOp: broadcast + normalize |
| `blaze/ops/allreduce_moe/op.py` | `AllReduceMoE` FusedOp: reduce + mixture of experts |

---

[**Previous:** `01_blaze_op_architecture.md`](./01_blaze_op_architecture.md) | [**Next:** `03_ops_catalog_overview.md`](./03_ops_catalog_overview.md)
