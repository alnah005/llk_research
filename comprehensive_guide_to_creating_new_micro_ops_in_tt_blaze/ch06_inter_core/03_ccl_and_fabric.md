# Section 3: CCL and Fabric Operations

## 3.1 Overview: From Intra-Device to Inter-Device

The operations in Section 2 (Mcast, Gather, Scatter, Copy) move data between
cores within a single Tenstorrent device via the on-chip NOC. For multi-device
configurations (T3K, Galaxy), data must also move between devices over the
Tenstorrent **fabric interconnect** -- dedicated inter-device Ethernet links
with their own routing and flow control, independent of the on-chip NOC.
TT-Blaze provides Collective Communication Library (CCL) ops that abstract
this inter-device communication.

CCL ops share the same MicroOp structure as intra-device ops -- they have
`compose()`, `emit()`, Python CT arg registration, and C++ kernel structs. The
key difference is the addition of **fabric connections**: routed links between
devices managed by the fabric connection manager.

This section covers the CCL ops, barrier synchronization primitives, and the
shared `setup_fabric()` utility:

| Op | Pattern | Purpose |
|----|---------|---------|
| `CclBroadcast` | 1-to-all devices | Broadcast activations from one device to all devices in a mesh |
| `AllReduce` | pairwise exchange + reduce | Exchange data between neighboring devices and sum |
| `BarrierSender` / `BarrierReceiver` | synchronization | Mesh synchronization (no data payload) |
| `DMRiscHandshake` | intra-core sync | Synchronize BRISC and NCRISC on the same core |
| `setup_fabric()` | utility | Shared fabric connection setup for all CCL ops |

---

## 3.2 The Fabric Interconnect Model

### 3.2.1 Mesh Topology

Tenstorrent multi-device systems are organized as a 2D mesh. A T3K system might
be a 4x2 mesh (4 rows, 2 columns). Each device has a **mesh coordinate**
`(row, col)` and a globally unique **FabricNodeId** assigned by the runtime.

The FusedProgram exposes the current device's position through mesh-level
fields:

| Field | Type | Purpose |
|-------|------|---------|
| `f.mesh_coord` | `tuple[int, int]` | `(row, col)` of this device in the mesh |
| `f.mesh_shape` | `tuple[int, int]` | `(rows, cols)` of the mesh |
| `f.mesh_device` | `ttnn.MeshDevice` | Device handle with fabric APIs |

These are set by the `MeshFusedProgram` wrapper or the compiler. They enable
the same `emit()` code to produce different behavior per device (sender vs
receiver) based on position:

```python
row, col = f.mesh_coord       # (int, int) position in mesh
mesh_shape = f.mesh_shape     # (rows, cols) mesh dimensions
mesh_device = f.mesh_device   # the MeshDevice handle
```

### 3.2.2 FabricNodeId

Each device in the mesh has a `FabricNodeId` that uniquely identifies it in the
fabric routing tables. The ID encodes both the physical device location and its
fabric address. The Python emit code resolves FabricNodeIds from mesh
coordinates:

```python
# Get the fabric node ID for a device at mesh coordinate (row, col)
coord = ttnn.MeshCoordinate(row, col)
node_id = mesh_device.get_fabric_node_id(coord)
```

CCL ops use `FabricNodeId` values as destination addresses when setting up
fabric connections. For example, CclBroadcast computes destination node IDs
for forward and backward hops:

```python
# From CclBroadcast: resolve next-hop device
fc = ttnn.MeshCoordinate((row + 1) % mesh_rows, col)
dst_node = mesh_device.get_fabric_node_id(fc)
```

And AllReduce resolves its single neighbor device:

```python
# From AllReduce: resolve neighbor device
neighbor_mesh_coord = ttnn.MeshCoordinate(neighbor_row, neighbor_col)
dst_node = mesh_device.get_fabric_node_id(neighbor_mesh_coord)
```

### 3.2.3 Link Indices

Each device-to-device connection has a numbered link. In a simple 1D ring,
link 0 connects forward and link 1 connects backward. Most ops use link 0
by default, but AllReduce uses directional link selection:

```python
# From blaze/ops/all_reduce/op.py
link_indices=[0 if is_first_chip else 1]
```

The first device in a pair uses link 0 (forward direction) while the second
uses link 1 (backward direction). This ensures both devices can communicate
simultaneously without contention on the same physical fabric link.

### 3.2.4 Fabric Links and Connection Manager

Fabric connections are established via the `RoutingPlaneConnectionManager` in
the C++ kernel. Each connection represents a link to one destination device.
The connection is opened with `open_connections()` and closed with
`close_connections()`:

```cpp
// From CclBroadcast kernel:
tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
size_t fabric_arg_idx = dm0::fabric_arg_idx;
open_connections(fabric_connection, num_connections, fabric_arg_idx);
// ... send data ...
close_connections(fabric_connection);
```

The `fabric_arg_idx` tells the connection manager where in the per-core runtime
args to find the pre-computed routing information (teardown semaphores, buffer
index semaphores, destination FabricNodeIds).

---

## 3.3 The `setup_fabric()` Utility

**Source**: `blaze/ccl.py`

All CCL ops share the `setup_fabric()` function for fabric connection setup.
This function consolidates semaphore allocation, runtime arg computation, kernel
define injection, and fabric core tracking.

### 3.3.1 Signature

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

**Parameters:**

| Parameter | Type | Purpose |
|-----------|------|---------|
| `fp` | `FusedProgram` | The fused program context with `mesh_coord` and `mesh_device` set |
| `prefix` | `str` | Op prefix (for tracking fabric cores per CCL op) |
| `worker_core` | `ttnn.CoreCoord` | The single logical worker core that owns the fabric connection |
| `risc` | `Risc` | Which RISC manages the fabric connection (must be `Risc.NCRISC` or `Risc.BRISC`, not `Risc.TRISC`) |
| `dst_nodes` | `list[ttnn.FabricNodeId]` | Fabric node IDs of destination devices |
| `link_indices` | `list[int]` or `None` | Fabric link indices (default `[0]`) |

**Returns:** The starting index of the fabric runtime args for this core's
specified RISC. This index is stored as a CT arg so the kernel knows where
to find the fabric args in the per-core runtime arg array.

### 3.3.2 Implementation Walkthrough

**Step 1: Empty destination guard.** For receiver devices that do not send
fabric data, `setup_fabric` is called with an empty `dst_nodes` list. In this
case, it only creates an empty per-core runtime args slot (no semaphores, no
routing) and returns early:

```python
if len(dst_nodes) == 0:
    _, _, _ = fp._set_per_core_runtime_args(worker_core)
    return
```

**Step 2: Semaphore allocation.** For each connection (destination), two
program semaphores are allocated -- one for teardown signaling and one for
buffer index tracking:

```python
worker_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])
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

These are `program_semaphore=True` semaphores -- per-device, slot-based, scoped
to the single worker core. They are not mesh-global because fabric connections
are per-device. The `core_ranges` parameter restricts the semaphore to the
single worker core, allowing tt-metal to reuse slot IDs on other cores.

**Step 3: Compute fabric runtime args.** The runtime args encode the routing
information for the fabric connection manager:

```python
coord = ttnn.MeshCoordinate(row, col)
rt_args = ttnn.compute_fabric_connection_rt_args(
    mesh_device.get_fabric_node_id(coord),  # source device
    dst_nodes,                               # destination devices
    link_indices or [0],                     # which fabric link
    teardown_sems,                           # teardown semaphore ids
    buf_idx_sems,                            # buffer index semaphore ids
)
```

`compute_fabric_connection_rt_args` is a ttnn host-side API that generates the
full list of runtime arg values that `open_connections()` will consume in the
kernel.

**Step 4: RISC constraint and runtime arg injection.** Fabric connections can
only be established on data-movement RISCs. The computed runtime args are
injected into the per-core runtime args for the specified RISC:

```python
assert risc != Risc.TRISC, "Can only connect to fabric using data movement RISCs"
ncrisc_prev_len, _, brisc_prev_len = fp._set_per_core_runtime_args(
    worker_core, **{RISC_NAMES[risc]: rt_args}
)
```

The function returns the previous length of the runtime arg array for each
RISC. This is the `fabric_arg_idx` -- the offset where the fabric args start.
The kernel uses this index to find the fabric routing data at runtime.

**Step 5: Add fabric kernel defines.** Fabric operations require specific
preprocessor defines to enable the fabric API:

```python
for name, val in ttnn.get_fabric_kernel_defines("Linear"):
    fp.add_define(name, val)
```

`get_fabric_kernel_defines("Linear")` returns defines for the linear fabric
topology.

**Step 6: Track fabric cores.** The function registers which cores use fabric
connections, per-RISC, per-CCL-op. This tracking enables later barrier
injection (see Section 3.7) to find all fabric-connected cores:

```python
fp._fabric_cores_per_risc_per_ccl.setdefault(prefix, {}).setdefault(risc, []).append(worker_core)
```

**Step 7: Return the arg index.** The return value is the runtime arg index
where the fabric args start, which the calling op stores as a CT arg:

```python
return {Risc.NCRISC: ncrisc_prev_len, Risc.BRISC: brisc_prev_len}[risc]
```

---

## 3.4 CclBroadcast -- Multi-Device Fabric Broadcast

**Source**: `blaze/ops/ccl_broadcast/op.py`, `blaze/ops/ccl_broadcast/kernels/op.hpp`

CclBroadcast multicasts data from a single sender device to all other devices
in the mesh. It supports dual-axis broadcast on 2D meshes: the primary sender
first unicasts to a secondary sender on another column, then both senders
multicast along their respective columns simultaneously.

### 3.4.1 Routing Computation

The `compute_routing()` helper determines each device's role based on its
mesh position:

```python
def compute_routing(
    row: int,
    col: int,
    sender_coord: tuple[int, int],
    mesh_shape: tuple[int, int],
    secondary_cluster_axis: int | None,
    is_torus: bool,
) -> BroadcastRouting:
```

The `BroadcastRouting` dataclass captures the per-device routing state:

```python
@dataclass
class BroadcastRouting:
    is_sender: bool            # primary sender
    is_secondary_sender: bool  # secondary sender (dual-axis only)
    is_receiver: bool          # pure receiver
    num_fwd: int               # forward hops (down the column)
    num_bwd: int               # backward hops (up the column)
    has_secondary: int         # 0 or 1 -- does primary send to secondary?
    enable_torus: bool         # wrap-around routing enabled
    ring_index: int            # position in the ring
    num_connections: int       # total fabric connections needed
```

**Role determination.** Each device's role depends on its mesh position
relative to the sender:

```python
sender_row, sender_col = sender_coord
mesh_rows, mesh_cols = mesh_shape

is_sender = (row == sender_row) and (col == sender_col)
is_secondary_sender = (
    secondary_cluster_axis is not None
    and row == sender_row
    and col != sender_col
)
```

**Forward and backward hops.** In a non-torus mesh, the sender at row R in a
mesh with M rows has `num_fwd = M - R - 1` forward hops and `num_bwd = R`
backward hops. In torus mode (when the sender is at an edge), the hops are
balanced: `num_fwd = (M-1)//2`, `num_bwd = M - 1 - num_fwd`.

```python
if enable_torus:
    num_fwd = (mesh_rows - 1) // 2
    num_bwd = mesh_rows - 1 - num_fwd
else:
    num_fwd = mesh_rows - row - 1
    num_bwd = row
```

**Example.** For a non-torus 4-device column with sender at row 1:

```
Row 0: is_receiver=True,  num_fwd=3, num_bwd=0, num_connections=0
Row 1: is_sender=True,    num_fwd=2, num_bwd=1, num_connections=2
Row 2: is_receiver=True,  num_fwd=1, num_bwd=2, num_connections=0
Row 3: is_receiver=True,  num_fwd=0, num_bwd=3, num_connections=0
```

The `num_fwd` and `num_bwd` fields are computed for all devices (not just the
sender), but only the sender opens fabric connections (`num_connections > 0`).
The sender multicasts forward 2 hops (to rows 2 and 3) and backward 1 hop
(to row 0).

**Dual-axis broadcast.** When `secondary_cluster_axis` is set and the mesh
has multiple columns, the primary sender first unicasts to a secondary sender
on the other column. Both then multicast along their respective columns.
This halves the broadcast latency for 2D meshes.

**Destination node computation.** The `compute_dst_nodes()` function resolves
the fabric destination FabricNodeIds for a sender device:

```python
def compute_dst_nodes(row, col, routing, sender_coord, mesh_shape, mesh_device):
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

### 3.4.2 Python Emit

```python
@staticmethod
def emit(
    f: FusedProgram,
    src,
    output_tensor,
    *,
    prefix: str = "ccl_broadcast",
    sender_coord: tuple[int, int],
    secondary_cluster_axis: int | None = None,
    num_links: int = 1,
    num_iterations: int = 1,
    is_torus: bool = False,
) -> CBHandle:
```

**Parameters:**

| Parameter | Purpose |
|-----------|---------|
| `sender_coord` | `(row, col)` of the sender device in the mesh |
| `secondary_cluster_axis` | Enable dual-axis broadcast. `None` = single axis. |
| `num_links` | Number of fabric links per connection |
| `num_iterations` | Loop count for multi-iteration broadcast |
| `is_torus` | Enable wrap-around routing (shorter paths on ring topologies) |

**Single-core execution.** CCL broadcast runs on a single worker core per
device (the first core of the input tensor's shard grid):

```python
shard_start = src.memory_config().shard_spec.grid.bounding_box().start
worker_core = ttnn.CoreCoord(shard_start.x, shard_start.y)
worker_crs = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])
phys = mesh_device.worker_core_from_logical_core(worker_core)
core_noc_x, core_noc_y = phys.x, phys.y
```

**Three semaphores.** Three mesh-global semaphores coordinate the broadcast:

```python
out_sem = f.semaphore(f"{prefix}.out_sem")       # output ready signal
bar_sem = f.semaphore(f"{prefix}.bar_sem")       # barrier sync
sec_sem = f.semaphore(f"{prefix}.sec_sem")       # secondary sender sync
```

**CT args.** Uniform args go to `f.unified_ct_args()`:

```python
f.unified_ct_args([
    (f"{prefix}.src", src_handle),
    (f"{prefix}.num_pages_to_read", num_pages),
    (f"{prefix}.tensor0_page_size", page_size_bytes),
    (f"{prefix}.num_targets_forward_direction", routing.num_fwd),
    (f"{prefix}.num_targets_backward_direction", routing.num_bwd),
    (f"{prefix}.core_noc_x", core_noc_x),
    (f"{prefix}.core_noc_y", core_noc_y),
    (f"{prefix}.has_secondary_target", routing.has_secondary),
    (f"{prefix}.start_distance_in_hops_forward", 1 if routing.num_fwd > 0 else 0),
    (f"{prefix}.range_hops_forward", routing.num_fwd),
    (f"{prefix}.start_distance_in_hops_backward", 1 if routing.num_bwd > 0 else 0),
    (f"{prefix}.range_hops_backward", routing.num_bwd),
    (f"{prefix}.num_iterations", num_iterations),
])
```

**Per-device role flags.** Unlike intra-device ops where flags are per-core
within a single device, CclBroadcast has per-device roles. But since the op
runs on a single worker core per device, the flags are per-core within that
device. Only the broadcast worker core gets non-zero values; all other cores
in the fused kernel see 0:

```python
f.per_core_unified_ct_args([
    (f"{prefix}.is_sender", {worker_crs: int(routing.is_sender)}),
    (f"{prefix}.is_secondary_sender", {worker_crs: int(routing.is_secondary_sender)}),
    (f"{prefix}.is_receiver", {worker_crs: int(routing.is_receiver)}),
])
```

Note that these are per-core *value maps* (not `f.flag()`), because the
role depends on which device is being compiled.

**Runtime args.** Unlike intra-device ops that use only CT args, CclBroadcast
uses NCRISC runtime args for values that may change between dispatches --
tensor buffer addresses and semaphore L1 addresses:

```python
f.ncrisc_rt_args([
    (f"{prefix}.tensor_address0", tensor_address0),
    (f"{prefix}.out_ready_sem_bank_addr", out_sem),
    (f"{prefix}.wait_output_semaphore", wait_sem),
    (f"{prefix}.reset_global_semaphore", wait_sem),
    (f"{prefix}.out_ready_sem_noc0_x", core_noc_x),
    (f"{prefix}.out_ready_sem_noc0_y", core_noc_y),
    (f"{prefix}.out_ready_sem_wait_value", num_links),
    (f"{prefix}.barrier_sem", bar_sem),
    (f"{prefix}.barrier_sem_noc0_x", core_noc_x),
    (f"{prefix}.barrier_sem_noc0_y", core_noc_y),
    (f"{prefix}.ring_index", routing.ring_index),
    (f"{prefix}.secondary_sync_sem", sec_sem),
    (f"{prefix}.num_connections", routing.num_connections),
])
```

**Fabric setup.** Sender and secondary sender devices open fabric connections.
CclBroadcast uses `Risc.NCRISC` for fabric sends:

```python
if routing.is_sender or routing.is_secondary_sender:
    dst_nodes = compute_dst_nodes(row, col, routing, sender_coord, mesh_shape, mesh_device)
    fabric_arg_idx = setup_fabric(
        f,
        prefix=prefix,
        worker_core=fabric_core,
        risc=Risc.NCRISC,
        dst_nodes=dst_nodes
    )
```

The `fabric_arg_idx` is stored as a per-core NCRISC CT arg:

```python
f.per_core_ncrisc_ct_args([
    (f"{prefix}.fabric_arg_idx", {fabric_core: fabric_arg_idx})
])
```

### 3.4.3 C++ Kernel

The kernel has three distinct code paths based on the role flags, split
between NCRISC (fabric communication) and BRISC (local CB management).

**CT Arg Structs:**

```cpp
template <typename A>
struct CoreCTArgs {
    static constexpr Flag is_sender = A::is_sender;
    static constexpr Flag is_secondary_sender = A::is_secondary_sender;
    static constexpr Flag is_receiver = A::is_receiver;
    static constexpr uint32_t num_iterations = A::num_iterations;
};

template <typename A>
struct ReaderCTArgs {  // NCRISC
    static constexpr CB src = A::src;
    static constexpr uint32_t num_pages_to_read = A::num_pages_to_read;
    static constexpr uint32_t tensor0_page_size = A::tensor0_page_size;
    static constexpr uint32_t num_targets_forward_direction = A::num_targets_forward_direction;
    static constexpr uint32_t num_targets_backward_direction = A::num_targets_backward_direction;
    static constexpr uint32_t core_noc_x = A::core_noc_x;
    static constexpr uint32_t core_noc_y = A::core_noc_y;
    static constexpr uint32_t has_secondary_target = A::has_secondary_target;
    static constexpr uint32_t start_distance_in_hops_forward = A::start_distance_in_hops_forward;
    static constexpr uint32_t range_hops_forward = A::range_hops_forward;
    static constexpr uint32_t start_distance_in_hops_backward = A::start_distance_in_hops_backward;
    static constexpr uint32_t range_hops_backward = A::range_hops_backward;
    static constexpr uint32_t fabric_arg_idx = A::fabric_arg_idx;
};

template <typename A>
struct WriterCTArgs {  // BRISC
    static constexpr CB src = A::src;
    static constexpr uint32_t num_pages_to_read = A::num_pages_to_read;
};
```

**NCRISC includes** -- the fabric API headers are guarded behind
`COMPILE_FOR_NCRISC` because only NCRISC runs the fabric code in
CclBroadcast:

```cpp
#if defined(COMPILE_FOR_NCRISC)
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
```

**BRISC path.** BRISC only manages the source CB on the sender device. This
CB reserve/push pattern makes the source data available for NCRISC's
`cb_wait_front()` in the fabric sender path:

```cpp
#if defined(COMPILE_FOR_BRISC)
for (uint32_t iter = 0; iter < core::num_iterations; ++iter) {
    if constexpr (core::is_sender) {
        cb_reserve_back(dm1::src, dm1::num_pages_to_read);
        cb_push_back(dm1::src, dm1::num_pages_to_read);
    }
}
```

**NCRISC primary sender path.** The sender opens fabric connections, configures
multicast routes, and sends the payload:

```cpp
// Open fabric connections
size_t fabric_arg_idx = dm0::fabric_arg_idx;
open_connections(fabric_connection, num_connections, fabric_arg_idx);

// Configure fused route (payload + semaphore increment)
fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<...>(
    fabric_connection, fused_route_id, starts, ranges,
    fused_header, dm0::tensor0_page_size);

// Wait for source CB data
cb_wait_front(dm0::src, dm0::num_pages_to_read);
size_t l1_read_addr = get_read_ptr(dm0::src);

// Dual-axis: unicast to secondary sender first (if applicable)
if constexpr (dm0::has_secondary_target) {
    fabric_unicast_noc_fused_unicast_with_atomic_inc(
        &secondary_slot.sender, secondary_header,
        l1_read_addr,
        dm0::tensor0_page_size * dm0::num_pages_to_read,
        NocUnicastAtomicIncFusedCommandHeader{
            dst_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, true},
        1);  // 1 hop
}

// Multicast along primary axis
fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<...>(
    fabric_connection, fused_route_id, l1_read_addr,
    NocUnicastAtomicIncFusedCommandHeader{
        dst_noc_addr, out_ready_sem_noc_addr_in_pkt, 1, true},
    dm0::tensor0_page_size * dm0::num_pages_to_read);

// Local copy to own output buffer
noc_async_write(l1_read_addr, dst_noc_addr,
                dm0::tensor0_page_size * dm0::num_pages_to_read);
noc_semaphore_inc(out_ready_sem_noc_addr, 1);
```

The "fused" fabric operations combine data payload writes with semaphore
increments into a single packet, reducing fabric round-trips.

**NCRISC secondary sender path.** The secondary sender waits for data from the
primary sender, then multicasts along its own column:

```cpp
} else if constexpr (core::is_secondary_sender) {
    // Wait for data from primary
    if (wait_output_semaphore) {
        noc_semaphore_wait_min(sem_ptr, out_ready_sem_wait_value);
    }
    if (reset_global_semaphore) {
        unified_kernels::semaphore_dec(sem_ptr);
    }

    // Multicast received data along own column
    fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<...>(
        fabric_connection, fused_route_id,
        static_cast<size_t>(tensor_address0), ...);
    noc_async_writes_flushed();
```

**NCRISC receiver path.** Receivers simply wait for the output semaphore and
reset it:

```cpp
} else if constexpr (core::is_receiver) {
    if (wait_output_semaphore) {
        noc_semaphore_wait_min(sem_ptr, out_ready_sem_wait_value);
    }
    if (reset_global_semaphore) {
        unified_kernels::semaphore_dec(sem_ptr);
    }
}
```

**Cleanup.** Senders close fabric connections at the end of each iteration:

```cpp
if constexpr (core::is_secondary_sender || core::is_sender) {
    close_connections(fabric_connection);
}
noc_async_write_barrier();
```

### 3.4.4 Multi-Iteration Support

CclBroadcast supports `num_iterations > 1` for use cases where the same
broadcast pattern repeats (e.g., across transformer layers). The outer loop
in the kernel resets the `PacketHeaderPool` and re-opens fabric connections
each iteration. The connection lifecycle (open-send-close) repeats per
iteration:

```cpp
for (uint32_t iter = 0; iter < core::num_iterations; ++iter) {
    if constexpr (core::is_secondary_sender || core::is_sender) {
        size_t fabric_arg_idx = dm0::fabric_arg_idx;
        open_connections(fabric_connection, num_connections, fabric_arg_idx);
    }
    // ... data transfer ...
    if constexpr (core::is_secondary_sender || core::is_sender) {
        close_connections(fabric_connection);
    }
    noc_async_write_barrier();
}
```

---

## 3.5 AllReduce -- Pairwise Exchange and Reduction

**Source**: `blaze/ops/all_reduce/op.py`, `blaze/ops/all_reduce/kernels/op.hpp`

AllReduce exchanges data between neighboring devices and performs an element-wise
sum. Unlike CclBroadcast which is pure dataflow, AllReduce involves all three
RISC processors: NCRISC and BRISC for data movement, TRISC for the addition.

### 3.5.1 Two-Core Architecture

AllReduce uses a **two-core** design per device, in contrast to CclBroadcast's
single worker core:

- **Sender core**: Adjacent to the data core. NCRISC reads input data from the
  data core via NOC. BRISC sends it to the neighbor device via fabric.
- **Receiver core** (= data core): NCRISC receives remote data and pushes it
  to TRISC. TRISC adds local and remote data. BRISC signals the sender core
  that local data is ready.

The sender and receiver cores are different logical cores on the same device.
The sender core is placed adjacent to the data core to minimize NOC distance
for the initial data read:

```python
data_core = ttnn.CoreCoord(input_shard_grid_start.x, input_shard_grid_start.y)
if data_core.x > 0:
    sender_core = ttnn.CoreCoord(data_core.x - 1, data_core.y)
else:
    sender_core = ttnn.CoreCoord(data_core.x + 1, data_core.y)
receiver_core = data_core
```

### 3.5.2 Python Emit

```python
@staticmethod
def emit(
    f: FusedProgram,
    input: ttnn.Tensor | CBHandle,
    intermediate: ttnn.Tensor | CBHandle,
    output: ttnn.Tensor,
    residual: ttnn.Tensor | None = None,
    *,
    prefix: str = "all_reduce",
    cluster_axis: int,
) -> CBHandle:
```

**Tensor handling.** AllReduce accepts both tensors and CBHandles for input
and intermediate. It resolves tile formats and page sizes from either source:

```python
if isinstance(intermediate, CBHandle):
    receiver_cb_in0 = intermediate
    intermediate_dtype = intermediate.data_format
    intermediate_tile = ttnn.Tile((intermediate.tile_desc.height, intermediate.tile_desc.width))
    receiver_num_tiles = intermediate.num_pages
    receiver_page_size = intermediate.page_size
else:
    receiver_cb_in0 = f.cb_from_tensor(intermediate)
    intermediate_dtype = intermediate.dtype
    intermediate_tile = intermediate.get_tile()
    receiver_num_tiles = get_num_tiles(intermediate.shape, ...)
    receiver_page_size = intermediate_tile_size
```

**CB allocation.** AllReduce allocates a scratch CB on the sender core for
buffering the data before fabric transmission:

```python
sender_cb_in0 = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "sender_cb_in0"),
    num_pages=sender_num_tiny_tiles,
    core_ranges=sender_core_range_set,
    data_format=input_dtype,
    tile=input_tiny_tile,
    page_size=sender_page_size,
)
```

**Semaphores and reversed ordering.** Three semaphores coordinate the exchange:

```python
global_semaphore0_address = f.semaphore(f"{prefix}.semaphore0")
global_semaphore1_address = f.semaphore(f"{prefix}.semaphore1")
barrier_semaphore_address = f.semaphore(f"{prefix}.barrier_semaphore")
```

The first two semaphores are used asymmetrically depending on chip position.
The two devices in a pair **swap** which semaphore they use for sending vs
receiving, ensuring no deadlock:

```python
receiver_semaphore_address = global_semaphore1_address if is_first_chip else global_semaphore0_address
sender_semaphore_address = global_semaphore0_address if is_first_chip else global_semaphore1_address
```

This means device 0 sends on semaphore 0 and receives on semaphore 1, while
device 1 does the reverse. The barrier semaphore synchronizes the sender
core's data read from the data core (a local intra-device signal).

**Four-RISC CT arg distribution.** AllReduce is the most complex inter-core op
in terms of CT arg distribution, because it involves all three RISC types plus
per-core unified args:

```python
# Per-core role flags (unified -- all RISCs)
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender_core", sender_core_range_set),
    f.flag(f"{prefix}.is_receiver_core", receiver_core_range_set),
])

# NCRISC CT args -- sender read + receiver semaphore wait
f.ncrisc_ct_args([
    (f"{prefix}.sender_num_tiny_tiles", sender_num_tiny_tiles),
    (f"{prefix}.sender_page_size", sender_page_size),
    (f"{prefix}.sender_cb_in0", sender_cb_in0),
    (f"{prefix}.sender_input_data_core_noc_x", sender_input_data_core_noc_x),
    (f"{prefix}.sender_input_data_core_noc_y", sender_input_data_core_noc_y),
    (f"{prefix}.sender_input_tensor_address", input.buffer_address()),
    (f"{prefix}.sender_barrier_semaphore_address", barrier_semaphore_address),
    (f"{prefix}.receiver_num_tiles", receiver_num_tiles),
    (f"{prefix}.receiver_cb_in0", receiver_cb_in0),
    (f"{prefix}.receiver_cb_in1", receiver_cb_in1),
    (f"{prefix}.receiver_has_residual", receiver_has_residual),
    (f"{prefix}.receiver_cb_residual", receiver_cb_residual),
    (f"{prefix}.receiver_semaphore_address", receiver_semaphore_address),
    (f"{prefix}.receiver_setup_receiver_cb_in1", receiver_setup_receiver_cb_in1),
])

# TRISC CT args -- element-wise addition
f.trisc_ct_args([
    (f"{prefix}.receiver_num_tiles", receiver_num_tiles),
    (f"{prefix}.receiver_cb_in0", receiver_cb_in0),
    (f"{prefix}.receiver_cb_in1", receiver_cb_in1),
    (f"{prefix}.receiver_cb_out0", receiver_cb_out0),
    (f"{prefix}.receiver_has_residual", receiver_has_residual),
    (f"{prefix}.receiver_cb_residual", receiver_cb_residual),
])

# BRISC CT args -- fabric send + barrier signal
f.brisc_ct_args([
    (f"{prefix}.sender_num_tiny_tiles", sender_num_tiny_tiles),
    (f"{prefix}.sender_page_size", sender_page_size),
    (f"{prefix}.sender_cb_in0", sender_cb_in0),
    (f"{prefix}.sender_data_core_noc_x", sender_data_core_noc_x),
    (f"{prefix}.sender_data_core_noc_y", sender_data_core_noc_y),
    (f"{prefix}.sender_semaphore_core_noc_x", sender_semaphore_core_noc_x),
    (f"{prefix}.sender_semaphore_core_noc_y", sender_semaphore_core_noc_y),
    (f"{prefix}.sender_data_address", intermediate.buffer_address()),
    (f"{prefix}.sender_semaphore_address", sender_semaphore_address),
    (f"{prefix}.receiver_barrier_semaphore_noc_x", sender_core_noc_x),
    (f"{prefix}.receiver_barrier_semaphore_noc_y", sender_core_noc_y),
    (f"{prefix}.receiver_barrier_semaphore_address", barrier_semaphore_address),
])
```

**Fabric setup.** AllReduce uses `Risc.BRISC` for fabric sends (unlike
CclBroadcast which uses NCRISC). The reason is that NCRISC is already busy
reading input data from the data core via NOC reads, so BRISC handles the
fabric send since it would otherwise be idle on the sender core:

```python
fabric_arg_idx = setup_fabric(
    f,
    prefix=prefix,
    worker_core=fabric_core,
    risc=Risc.BRISC,
    dst_nodes=[mesh_device.get_fabric_node_id(neighbor_mesh_coord)],
    link_indices=[0 if is_first_chip else 1]
)
```

The `link_indices` selection ensures both devices send toward each other
simultaneously without link contention: device 0 uses link 0 (forward
direction) while device 1 uses link 1 (backward direction).

The fabric arg index is stored as a per-core unified CT arg (not per-core
BRISC, despite BRISC being the fabric RISC):

```python
f.per_core_unified_ct_args([
    (f"{prefix}.sender_fabric_arg_idx", {fabric_core: fabric_arg_idx})
])
```

### 3.5.3 C++ Kernel -- Three-RISC Coordination

AllReduce is unique among TT-Blaze inter-core ops because it uses **all three
RISC processors**. Each RISC has a distinct role on each core.

**NCRISC -- sender core:** Waits on a barrier semaphore (signaled by BRISC on
the receiver core to indicate local data is ready), then reads the input data
from the data core via NOC and pushes it into the sender CB:

```cpp
if constexpr (core_cta::is_sender_core) {
    const uint64_t base_src_address = get_noc_addr(
        dm0_cta::sender_input_data_core_noc_x,
        dm0_cta::sender_input_data_core_noc_y,
        dm0_cta::sender_input_tensor_address);

    cb_reserve_back(dm0_cta::sender_cb_in0, dm0_cta::sender_num_tiny_tiles);
    const uint32_t l1_write_address = get_write_ptr(dm0_cta::sender_cb_in0);

    // Wait for data to arrive on the data core
    auto barrier_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        dm0_cta::sender_barrier_semaphore_address);
    noc_semaphore_wait(barrier_sem_ptr, 1);
    noc_semaphore_set(barrier_sem_ptr, 0);

    constexpr uint32_t bytes_to_read =
        dm0_cta::sender_num_tiny_tiles * dm0_cta::sender_page_size;
    noc_async_read(base_src_address, l1_write_address, bytes_to_read);
    noc_async_read_barrier();
    cb_push_back(dm0_cta::sender_cb_in0, dm0_cta::sender_num_tiny_tiles);
}
```

**NCRISC -- receiver core:** Pushes residual data (if present), waits for the
remote device's data via the receiver semaphore, then pushes intermediate and
local data to TRISC via CBs:

```cpp
} else if constexpr (core_cta::is_receiver_core) {
    if constexpr (dm0_cta::receiver_has_residual) {
        cb_reserve_back(dm0_cta::receiver_cb_residual, dm0_cta::receiver_num_tiles);
        cb_push_back(dm0_cta::receiver_cb_residual, dm0_cta::receiver_num_tiles);
    }

    // Wait for remote data via fabric
    auto semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        dm0_cta::receiver_semaphore_address);
    noc_semaphore_wait(semaphore_ptr, 1);
    noc_semaphore_set(semaphore_ptr, 0);

    cb_reserve_back(dm0_cta::receiver_cb_in0, dm0_cta::receiver_num_tiles);
    cb_push_back(dm0_cta::receiver_cb_in0, dm0_cta::receiver_num_tiles);
}
```

**TRISC -- receiver core:** Performs the element-wise addition of local and
remote data, with optional residual accumulation:

```cpp
#elif defined(COMPILE_FOR_TRISC)
if constexpr (core_cta::is_receiver_core) {
    reconfig_data_format<false, true>(
        compute_cta::receiver_cb_in0, compute_cta::receiver_cb_in1);
    pack_reconfig_data_format<true>(compute_cta::receiver_cb_out0);

    // Optional: pre-load residual into DST registers
    if constexpr (compute_cta::receiver_has_residual) {
        copy_tile_to_dst_init_short(compute_cta::receiver_cb_residual);
        cb_wait_front(compute_cta::receiver_cb_residual,
                      compute_cta::receiver_num_tiles);
        tile_regs_acquire();
        for (uint32_t i = 0; i < compute_cta::receiver_num_tiles; i++) {
            copy_tile(compute_cta::receiver_cb_residual, i, i);
        }
        cb_pop_front(compute_cta::receiver_cb_residual,
                     compute_cta::receiver_num_tiles);
    }

    // Add local (cb_in1) and remote (cb_in0) tiles
    add_tiles_init(compute_cta::receiver_cb_in0,
                   compute_cta::receiver_cb_in1,
                   compute_cta::receiver_has_residual);

    cb_wait_front(compute_cta::receiver_cb_in0, compute_cta::receiver_num_tiles);
    cb_wait_front(compute_cta::receiver_cb_in1, compute_cta::receiver_num_tiles);
    cb_reserve_back(compute_cta::receiver_cb_out0, compute_cta::receiver_num_tiles);

    if constexpr (!compute_cta::receiver_has_residual) {
        tile_regs_acquire();
    }
    for (uint32_t i = 0; i < compute_cta::receiver_num_tiles; i++) {
        add_tiles(compute_cta::receiver_cb_in0,
                  compute_cta::receiver_cb_in1, i, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < compute_cta::receiver_num_tiles; i++) {
        pack_tile(i, compute_cta::receiver_cb_out0, i);
    }
    tile_regs_release();

    cb_pop_front(compute_cta::receiver_cb_in0, compute_cta::receiver_num_tiles);
    cb_pop_front(compute_cta::receiver_cb_in1, compute_cta::receiver_num_tiles);
    cb_push_back(compute_cta::receiver_cb_out0, compute_cta::receiver_num_tiles);
}
```

The constraint `static_assert(compute_cta::receiver_num_tiles <= 8)` limits
the tile count to the DST register file capacity.

**BRISC -- sender core:** Opens the fabric connection, sends the data to the
neighbor device, and pops the sender CB:

```cpp
#elif defined(COMPILE_FOR_BRISC)
if constexpr (core_cta::is_sender_core) {
    // Open fabric connection
    size_t fabric_arg_idx = dm1_cta::sender_fabric_arg_idx;
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, 1, fabric_arg_idx);

    // Configure unicast packet header
    PacketHeaderPool::reset();
    auto* packet_header_ptr = PacketHeaderPool::allocate_header(1);
    fabric_set_unicast_route(fabric_connection, packet_header_ptr, 0);
    auto& sender = fabric_connection.get(0).sender;

    constexpr uint32_t bytes_to_send =
        dm1_cta::sender_num_tiny_tiles * dm1_cta::sender_page_size;
    packet_header_ptr->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{
            data_noc_address, semaphore_noc_address, 1, true},
        align(bytes_to_send, L1_ALIGNMENT));

    cb_wait_front(dm1_cta::sender_cb_in0, dm1_cta::sender_num_tiny_tiles);
    uint32_t packet_base_address = get_read_ptr(dm1_cta::sender_cb_in0);

    sender.wait_for_empty_write_slot();
    sender.send_payload_without_header_non_blocking_from_address(
        packet_base_address, bytes_to_send);
    sender.send_payload_flush_blocking_from_address(
        (uint32_t)packet_header_ptr, sizeof(PACKET_HEADER_TYPE));
    close_connections(fabric_connection);

    cb_pop_front(dm1_cta::sender_cb_in0, dm1_cta::sender_num_tiny_tiles);
}
```

**BRISC -- receiver core:** Signals the sender core that local data is ready
via the barrier semaphore. This is the trigger that allows NCRISC on the
sender core to begin its NOC read:

```cpp
} else if constexpr (core_cta::is_receiver_core) {
    constexpr uint8_t BARRIER_NOC = 0;
    const uint64_t barrier_semaphore_noc_address = get_noc_addr(
        dm1_cta::receiver_barrier_semaphore_noc_x,
        dm1_cta::receiver_barrier_semaphore_noc_y,
        dm1_cta::receiver_barrier_semaphore_address, BARRIER_NOC);
    noc_semaphore_inc(barrier_semaphore_noc_address, 1, BARRIER_NOC);
    noc_async_atomic_barrier(BARRIER_NOC);
}
```

### 3.5.4 AllReduce Data Flow Summary

The complete data flow for a two-device AllReduce:

```
Device 0                                 Device 1
========                                 ========

1. BRISC(receiver) signals               1. BRISC(receiver) signals
   barrier_sem to sender core               barrier_sem to sender core

2. NCRISC(sender) reads input from       2. NCRISC(sender) reads input from
   data core via NOC read                   data core via NOC read

3. BRISC(sender) sends data to           3. BRISC(sender) sends data to
   Device 1 via fabric (link 0)             Device 0 via fabric (link 1)

4. NCRISC(receiver) waits for            4. NCRISC(receiver) waits for
   remote data semaphore (sem1)             remote data semaphore (sem0)

5. TRISC(receiver) adds local +          5. TRISC(receiver) adds local +
   remote data, packs output                remote data, packs output
```

Note the reversed semaphore ordering: device 0 receives on semaphore 1 and
sends on semaphore 0, while device 1 does the reverse. This prevents
deadlock in the bidirectional exchange.

---

## 3.6 Barriers and Synchronization Primitives

Multi-device operations require careful synchronization to avoid data races.
TT-Blaze provides dedicated barrier ops and synchronization patterns.

### 3.6.1 BarrierSender

**Source**: `blaze/ops/barrier_sender/op.py`, `blaze/ops/barrier_sender/kernels/op.hpp`

BarrierSender signals one or more receiver cores by incrementing their
semaphores. It is a pure synchronization op -- no data payload.

```python
@staticmethod
def emit(
    f: FusedProgram,
    *,
    prefix: str = "barrier_sender",
    sender_cores: list[ttnn.CoreCoord],
    execute_on_ncrisc: bool = False,
    execute_on_brisc: bool = False,
    send_to_ncrisc: bool = False,
    send_to_brisc: bool = False,
    ncrisc_semaphore_l1_addr: int | None = None,
    brisc_semaphore_l1_addr: int | None = None,
    receiver_cores: list[ttnn.CoreCoord] = [],
) -> None:
```

Key design points:

- **RISC flexibility**: Can execute on either NCRISC or BRISC (or both),
  selected via `execute_on_ncrisc` / `execute_on_brisc` flags.
- **Target RISC selection**: Can signal either NCRISC or BRISC semaphores
  on receiver cores (or both), via `send_to_ncrisc` / `send_to_brisc`.
- **Runtime arg arrays**: Receiver NOC coordinates are passed as runtime
  arg arrays to both NCRISC and BRISC (since either may execute):

```python
receiver_noc_coordinates = []
for receiver_core in receiver_cores:
    receiver_core_phys = f.device.worker_core_from_logical_core(receiver_core)
    receiver_noc_coordinates.append(receiver_core_phys.x)
    receiver_noc_coordinates.append(receiver_core_phys.y)
f.ncrisc_rt_arg_arrays([
    (f"{prefix}.receiver_core_noc_coordinates", receiver_noc_coordinates),
])
f.brisc_rt_arg_arrays([
    (f"{prefix}.receiver_core_noc_coordinates", receiver_noc_coordinates),
])
```

**CT arg registration:**

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.execute_on_ncrisc", sender_core_range_set, enabled=execute_on_ncrisc),
    f.flag(f"{prefix}.execute_on_brisc", sender_core_range_set, enabled=execute_on_brisc),
])

f.unified_ct_args([
    (f"{prefix}.num_receiver_cores", len(receiver_cores)),
    (f"{prefix}.send_to_ncrisc", send_to_ncrisc),
    (f"{prefix}.send_to_brisc", send_to_brisc),
    (f"{prefix}.ncrisc_semaphore_l1_addr",
     ncrisc_semaphore_l1_addr if ncrisc_semaphore_l1_addr is not None else 0),
    (f"{prefix}.brisc_semaphore_l1_addr",
     brisc_semaphore_l1_addr if brisc_semaphore_l1_addr is not None else 0),
])
```

**No-output pattern.** BarrierSender does not produce data. Its `f.output()`
call uses dummy zeros to create a graph node for codegen ordering without
requiring a valid CB:

```python
f.output(
    "barrier_sender",
    cb_id=0,
    num_pages=0,
    page_size=0,
    core_ranges=sender_core_range_set,
    grid=sender_core_range_set,
    does_produce_output=False,
    prefix=prefix,
)
```

**C++ kernel.** The kernel iterates over receiver cores and increments their
semaphores:

```cpp
static FORCE_INLINE void sender_impl(
    uint32_t num_receiver_cores,
    bool send_to_ncrisc,
    bool send_to_brisc,
    uint32_t ncrisc_semaphore_l1_addr,
    uint32_t brisc_semaphore_l1_addr) {
    for (uint32_t i = 0; i < num_receiver_cores * 2; i += 2) {
        uint32_t receiver_core_noc_x =
            rt_args::get<Args::receiver_core_noc_coordinates>(i);
        uint32_t receiver_core_noc_y =
            rt_args::get<Args::receiver_core_noc_coordinates>(i + 1);

        if (send_to_ncrisc) {
            uint64_t ncrisc_semaphore_noc_addr = get_noc_addr(
                receiver_core_noc_x, receiver_core_noc_y,
                ncrisc_semaphore_l1_addr);
            noc_semaphore_inc(ncrisc_semaphore_noc_addr, 1);
        }
        if (send_to_brisc) {
            uint64_t brisc_semaphore_noc_addr = get_noc_addr(
                receiver_core_noc_x, receiver_core_noc_y,
                brisc_semaphore_l1_addr);
            noc_semaphore_inc(brisc_semaphore_noc_addr, 1);
        }
    }
}
```

**Fabric barrier injection.** BarrierSender provides a factory method for
injecting fabric teardown barriers. The compiler uses this to ensure fabric
connections are properly closed before the program completes:

```python
@staticmethod
def emit_injected_fabric_barrier_sender(
    f: FusedProgram,
    *,
    prefix: str,
    fabric_cores_per_risc: dict[Risc, list[ttnn.CoreCoord]],
) -> None:
    BarrierSender.emit(
        f,
        prefix=prefix,
        sender_cores=list({c for cores in fabric_cores_per_risc.values()
                          for c in cores}),
        execute_on_ncrisc=bool(fabric_cores_per_risc.get(Risc.NCRISC)),
        execute_on_brisc=bool(fabric_cores_per_risc.get(Risc.BRISC)),
        # ...
    )
```

This method auto-derives the sender core list and RISC selection from the
fabric cores tracked by `setup_fabric()` via
`fp._fabric_cores_per_risc_per_ccl`.

### 3.6.2 BarrierReceiver

**Source**: `blaze/ops/barrier_receiver/op.py`, `blaze/ops/barrier_receiver/kernels/op.hpp`

BarrierReceiver is the complement to BarrierSender. It waits on a semaphore
until it reaches a specified value, then resets it:

```python
@staticmethod
def emit(
    f: FusedProgram,
    *,
    prefix: str = "barrier_receiver",
    receiver_cores: list[ttnn.CoreCoord],
    execute_on_ncrisc: bool = False,
    execute_on_brisc: bool = False,
    ncrisc_semaphore_l1_addr: int | None = None,
    brisc_semaphore_l1_addr: int | None = None,
    wait_value: int = 0,
) -> None:
```

The `wait_value` parameter specifies how many sender signals to wait for
before proceeding:

```python
f.unified_ct_args([
    (f"{prefix}.ncrisc_semaphore_l1_addr",
     ncrisc_semaphore_l1_addr if ncrisc_semaphore_l1_addr is not None else 0),
    (f"{prefix}.brisc_semaphore_l1_addr",
     brisc_semaphore_l1_addr if brisc_semaphore_l1_addr is not None else 0),
    (f"{prefix}.wait_value", wait_value),
])
```

The kernel is a simple semaphore wait-and-reset:

```cpp
static FORCE_INLINE void receiver_impl(
    uint32_t semaphore_l1_addr, uint32_t wait_value) {
    auto semaphore_l1_ptr = reinterpret_cast<
        volatile tt_l1_ptr uint32_t*>(semaphore_l1_addr);
    noc_semaphore_wait(semaphore_l1_ptr, wait_value);
    noc_semaphore_set(semaphore_l1_ptr, 0);
}
```

BarrierReceiver also uses the no-output pattern (`does_produce_output=False`).

### 3.6.3 DMRiscHandshake -- Intra-Core RISC Synchronization

**Source**: `blaze/ops/dm_risc_handshake/op.py`

DMRiscHandshake synchronizes BRISC and NCRISC on the *same* core. One RISC
increments a semaphore while the other waits on it. This is useful when one
RISC produces data that the other RISC needs to consume within the same core,
and CB-based synchronization is insufficient:

```python
@staticmethod
def emit(
    f: FusedProgram,
    *,
    prefix: str = "dm_risc_handshake",
    cores: list,
    signalling_risc: Risc,
) -> None:
    semaphore = f.semaphore(f"{prefix}.semaphore")

    f.per_core_unified_ct_args([
        f.flag(f"{prefix}.is_active_core", core_range_set),
    ])

    f.unified_ct_args([
        (f"{prefix}.execute_signalling_logic_on_ncrisc", execute_signalling_logic_on_ncrisc),
        (f"{prefix}.semaphore_l1_addr", semaphore),
    ])
```

The `signalling_risc` parameter determines which RISC performs the increment
and which performs the wait.

### 3.6.4 Synchronization Patterns in CCL

CCL ops use several synchronization mechanisms:

**Semaphore wait/reset.** The standard pattern for all CCL ops:

```cpp
volatile tt_l1_ptr uint32_t* sem_ptr =
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
noc_semaphore_wait(sem_ptr, expected_value);
noc_semaphore_set(sem_ptr, 0);  // or unified_kernels::semaphore_dec
```

**Fused atomic increment.** Fabric operations use fused unicast-with-atomic-
increment to combine data writes with semaphore signals in a single packet,
reducing fabric round-trips and latency:

```cpp
packet_header_ptr->to_noc_fused_unicast_write_atomic_inc(
    NocUnicastAtomicIncFusedCommandHeader{
        data_noc_address,        // where to write data
        semaphore_noc_address,   // which semaphore to increment
        1,                       // increment value
        true                     // flush
    },
    payload_size);
```

This eliminates the need for a separate semaphore signal after the data write.

**Barrier semaphore (AllReduce).** AllReduce uses a barrier semaphore to
synchronize between the receiver core's BRISC (which knows local data is
ready) and the sender core's NCRISC (which needs to read it):

```cpp
// BRISC on receiver core signals sender core
noc_semaphore_inc(barrier_semaphore_noc_address, 1, BARRIER_NOC);

// NCRISC on sender core waits for signal
noc_semaphore_wait(barrier_sem_ptr, 1);
noc_semaphore_set(barrier_sem_ptr, 0);
```

---

## 3.7 Fabric Connection Lifecycle

All CCL ops follow the same fabric connection lifecycle, split between
Python setup and C++ kernel execution.

### 3.7.1 Python Side

```python
# 1. Resolve destination FabricNodeIds
dst_node = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(row, col))

# 2. Call setup_fabric() to allocate semaphores and compute RT args
fabric_arg_idx = setup_fabric(
    f, prefix=prefix, worker_core=core,
    risc=Risc.NCRISC,  # or Risc.BRISC
    dst_nodes=[dst_node],
    link_indices=[0],
)

# 3. Store fabric_arg_idx as a CT arg
f.per_core_ncrisc_ct_args([
    (f"{prefix}.fabric_arg_idx", {core: fabric_arg_idx})
])
```

### 3.7.2 C++ Side

```cpp
// 1. Create connection manager
tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;

// 2. Open connections (reads RT args starting at fabric_arg_idx)
size_t fab_idx = dm_cta::fabric_arg_idx;
open_connections(fabric_connection, num_connections, fab_idx);

// 3. Reset packet header pool and allocate headers
PacketHeaderPool::reset();
auto route_id = PacketHeaderPool::allocate_header_n(num_connections);

// 4. Configure routes and send data via fabric API
fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<...>(...);
// -- or --
fabric_unicast_noc_fused_unicast_with_atomic_inc(...);

// 5. Close connections
close_connections(fabric_connection);
```

### 3.7.3 Barrier Teardown (Optional)

After the main CCL op completes, `BarrierSender` / `BarrierReceiver` ops may
be injected by the compiler to synchronize across the mesh before the next
program. The `emit_injected_fabric_barrier_sender` factory method (see
Section 3.6.1) uses the fabric core tracking from `setup_fabric()` to
determine which cores need barrier signals.

### 3.7.4 RISC Selection for Fabric

CclBroadcast uses `Risc.NCRISC` for fabric sends, while AllReduce uses
`Risc.BRISC`. The choice depends on which RISC has the available bandwidth:

- **CclBroadcast**: BRISC only manages the local CB (`cb_reserve_back` /
  `cb_push_back`). NCRISC handles all fabric communication because it has
  the more complex workload and BRISC's role is trivial.
- **AllReduce**: NCRISC is busy reading input data from the data core via NOC
  reads. BRISC handles the fabric send because it would otherwise be idle on
  the sender core.

---

## 3.8 Design Patterns for New CCL Ops

When implementing a new CCL op, follow these patterns from the existing
implementations:

1. **Single worker core per device.** CCL ops run on one core per device (not
   the full grid). Use the input tensor's shard grid start as the worker core.
   AllReduce is an exception -- it uses two cores per device.

2. **Use `setup_fabric()`.** Never manually allocate fabric semaphores or
   compute routing RT args. Call `setup_fabric()` with the appropriate RISC
   and destination nodes.

3. **Store `fabric_arg_idx` as a CT arg.** The kernel needs this index to
   find the fabric routing data in the per-core runtime args.

4. **Handle all device roles.** Every device in the mesh will execute the same
   kernel. Use role flags (`is_sender`, `is_receiver`) to branch into the
   correct code path. Guard fabric-only code with role checks.

5. **Close connections.** Always call `close_connections()` after fabric sends.
   Failing to close leaks fabric resources.

6. **Guard RISC-specific code.** Use `#if defined(COMPILE_FOR_NCRISC)` guards
   around fabric API includes and connection manager code. TRISC and the
   non-fabric RISC should never compile fabric headers.

7. **Choose the right RISC for fabric.** Assign fabric sends to the RISC that
   is least busy with other work (NOC reads, CB management, compute). TRISC
   cannot run fabric operations.

8. **Use fused atomic increments.** Combine data writes with semaphore signals
   via `NocUnicastAtomicIncFusedCommandHeader` to minimize fabric round-trips.

---

## 3.9 Summary

### Choosing the Right Inter-Core/Inter-Device Op

| Need | Op | Scope |
|------|----|-------|
| Broadcast same data to all cores | **Mcast** | Intra-device |
| Collect data from all cores to one | **Gather** | Intra-device |
| Distribute different data to specific cores | **Scatter** | Intra-device |
| Generic local/remote L1 copy | **Copy** | Intra-device |
| Broadcast data to all devices | **CclBroadcast** | Inter-device |
| Exchange and sum across device pairs | **AllReduce** | Inter-device |
| Synchronize cores without data | **BarrierSender/Receiver** | Intra-device |
| Synchronize RISCs on same core | **DMRiscHandshake** | Intra-core |

### Intra-Device vs Inter-Device Comparison

| Aspect | Intra-device (Mcast/Gather) | Inter-device (CCL) |
|--------|-----------------------------|--------------------|
| Transport | NOC multicast/unicast | Fabric API |
| Coordinate system | NOC (x, y) | FabricNodeId + mesh coordinates |
| Connection setup | None (NOC is always-on) | `setup_fabric()` + open/close |
| Semaphore scope | L1-local | Mesh-global (named) or program-local |
| Sender identification | `f.sender_grid` | `f.mesh_coord` vs `sender_coord` |
| Packet format | Raw NOC writes | Fused unicast + atomic increment |

### Key API Reference

| Concept | Key API | Location |
|---------|---------|----------|
| Fabric setup | `setup_fabric(fp, prefix, worker_core, risc, dst_nodes)` | `blaze/ccl.py` |
| Device node ID | `mesh_device.get_fabric_node_id(MeshCoordinate(r, c))` | `ttnn` API |
| Broadcast routing | `compute_routing()`, `compute_dst_nodes()` | `blaze/ops/ccl_broadcast/op.py` |
| AllReduce neighbor | `MeshCoordinate(neighbor_row, neighbor_col)` | `blaze/ops/all_reduce/op.py` |
| Barrier sync | `BarrierSender.emit()`, `BarrierReceiver.emit()` | `blaze/ops/barrier_sender/op.py` |
| RISC handshake | `DMRiscHandshake.emit()` | `blaze/ops/dm_risc_handshake/op.py` |
| Per-core fabric args | `f.per_core_ncrisc_ct_args([(...fabric_arg_idx, {core: idx})])` | CCL ops |
| Program semaphores | `f.semaphore(program_semaphore=True, core_ranges=...)` | `blaze/fused_program.py` |
| Mesh context | `f.mesh_coord`, `f.mesh_shape`, `f.mesh_device` | `FusedProgram` fields |
| Fabric kernel defines | `ttnn.get_fabric_kernel_defines("Linear")` | `blaze/ccl.py` |
| Connection lifecycle | `open_connections()` ... `close_connections()` | Fabric kernel API |
