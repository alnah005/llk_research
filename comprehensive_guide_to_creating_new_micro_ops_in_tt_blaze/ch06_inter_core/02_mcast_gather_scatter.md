# Section 2: Mcast, Gather, Scatter, and Copy

This section dissects the four primitive inter-core micro-ops in TT-Blaze.
Each follows the sender/receiver pattern from Section 1, but with distinct
data-movement topologies, RISC assignments, and synchronization strategies.

| Op | Topology | Sender RISC | Receiver RISC | Use Case |
|----|----------|-------------|---------------|----------|
| **Mcast** | One-to-many (multicast) | DM1 (BRISC) | DM0 (NCRISC) | Broadcast weights, activations to all cores |
| **Gather** | Many-to-one | DM0 (NCRISC) | DM1 (BRISC) | Collect partial results from all cores to one |
| **Scatter** | One-to-many (unicast per dest) | DM1 (BRISC) | DM0 (NCRISC) | Distribute shards to specific cores |
| **Copy** | One-to-one | Configurable | N/A (single core) | Local or remote L1 copy |

---

## 2.1 Mcast -- One-to-All Multicast

**Source**: `blaze/ops/mcast/op.py`, `blaze/ops/mcast/kernels/op.hpp`

Mcast is the most heavily used inter-core op. It multicasts data from a single
sender core to every core in the device grid. Typical uses include broadcasting
embedding lookups, RMSNorm weights, and shared activations before column-
parallel matmuls.

### 2.1.1 Python Emit

The `Mcast.emit()` static method accepts either a `ttnn.Tensor` or a `CBHandle`
as source, along with configuration options:

```python
@staticmethod
def emit(
    f: FusedProgram,
    src,
    *,
    prefix: str,
    sender_sem: int | None = None,
    dst_num_pages: int | None = None,
    dst_tile_info: TileInfo | None = None,
) -> CBHandle:
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `src` | `CBHandle` or `ttnn.Tensor` | required | Source data. Tensor allocates a new input CB; CBHandle reuses an existing CB (e.g., a TRISC-filled scratch buffer). |
| `prefix` | `str` | required | Namespace for all CT args. Must be unique within the fused program. |
| `sender_sem` | `int` or `None` | `None` | Reuse an existing sender semaphore. `None` allocates a new one. |
| `dst_num_pages` | `int` or `None` | `None` | Override destination page count. Defaults to source page count. |
| `dst_tile_info` | `TileInfo` or `None` | `None` | Override destination tile format (e.g., face-view to full tiles). |

**Returns:** `CBHandle` for the destination CB on all cores.

**Source resolution.** If `src` is a tensor, `f.cb_from_tensor(src)` allocates
a CB for it. If it is already a CBHandle (from an upstream op), it is used
directly:

```python
if isinstance(src, CBHandle):
    src_handle = src
else:
    src_handle = f.cb_from_tensor(src)
```

**Data size computation.** The transfer size is derived from the source CB:

```python
num_pages = src_handle.num_pages
tile_size = src_handle.page_size
data_size_bytes = num_pages * tile_size
```

**Destination CB.** Mcast allocates a scratch CB on `f.all_cores` (the full
grid) for the destination. The `balanced=False` flag acknowledges that not
every receiver may pop all pages:

```python
dst = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "dst"),
    num_pages=_dst_num_pages,
    core_ranges=f.all_cores,
    data_format=_dst_data_format,
    tile=_dst_tile_desc,
    page_size=_dst_page_size,
    balanced=False,
)
```

The optional `dst_tile_info` parameter allows the destination to use a
different tile format than the source (e.g., converting face-view tiles to
full tiles during the multicast).

**Semaphore allocation.** Two semaphores coordinate the transfer:

```python
if sender_sem is None:
    sender_sem = f.semaphore(f"{prefix}.sender")
receiver_sem = f.semaphore(f"{prefix}.receiver")
```

The caller can pass an existing `sender_sem` to share it across multiple
Mcast instances in a fused op.

**Role flags.** Four per-core flags control kernel behavior:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
    f.flag(f"{prefix}.pop_src", f.all_cores),
    f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
])
```

- `is_sender`: 1 only on the sender core.
- `is_receiver`: 1 on all cores except the sender.
- `pop_src`: 1 on all cores (both sender and receivers pop their respective CBs).
- `init_src`: 1 on the sender core only if `src` is a tensor (needs
  `setup_sharded_buffer`). When src is a CBHandle from an upstream op, init
  is skipped because data was already pushed.

**NCRISC CT args** (receiver-side dataflow):

```python
f.ncrisc_ct_args([
    (f"{prefix}.src", src_handle),
    (f"{prefix}.src_num_pages", num_pages),
    (f"{prefix}.src_is_tensor_backed", src_is_tensor_backed),
    (f"{prefix}.receiver_semaphore", receiver_sem),
    (f"{prefix}.dst", dst),
    (f"{prefix}.dst_num_pages", dst.num_pages),
])
```

**BRISC CT args** (sender-side dataflow):

```python
f.brisc_ct_args([
    (f"{prefix}.dest_noc_start_x", f.noc_start.x),
    (f"{prefix}.dest_noc_start_y", f.noc_start.y),
    (f"{prefix}.dest_noc_end_x", f.noc_end.x),
    (f"{prefix}.dest_noc_end_y", f.noc_end.y),
    (f"{prefix}.num_cores", f.num_mcast_cores),
    (f"{prefix}.sender_semaphore", sender_sem),
    (f"{prefix}.receiver_semaphore", receiver_sem),
    (f"{prefix}.data_size_bytes", data_size_bytes),
    (f"{prefix}.src", src_handle),
    (f"{prefix}.src_num_pages", num_pages),
    (f"{prefix}.dst", dst),
    (f"{prefix}.is_part_of_receiver_grid",
     int(f.mcast_range.contains(f.sender))),
    (f"{prefix}.linked", 1),
    (f"{prefix}.posted", 1),
    (f"{prefix}.loopback", 0),
])
```

Key BRISC args:
- `dest_noc_start_x/y` and `dest_noc_end_x/y`: The NOC bounding box for the
  multicast rectangle, precomputed from `f.noc_start` and `f.noc_end`.
- `is_part_of_receiver_grid`: Whether the sender is inside the multicast
  rectangle. This affects the destination count for non-posted ACKs.
- `linked`, `posted`, `loopback`: NOC transfer mode flags (see 2.1.3).

### 2.1.2 C++ Kernel -- Persistent Sender State

The Mcast kernel uses a **persistent sender** pattern. The `init()` method
configures the NOC command buffers once, and subsequent `operator()` calls
(or `run_as<A>()` calls) reuse this state without re-programming the NOC.

**CT Arg Structs:**

```cpp
template <typename A>
struct CoreCTArgs {
    static constexpr Flag is_sender = A::is_sender;
    static constexpr Flag is_receiver = A::is_receiver;
    static constexpr Flag pop_src = A::pop_src;
    static constexpr Flag init_src = A::init_src;
};

template <typename A>
struct ReaderCTArgs {  // NCRISC
    static constexpr Semaphore receiver_semaphore = A::receiver_semaphore;
    static constexpr CB dst = A::dst;
    static constexpr uint32_t dst_num_pages = A::dst_num_pages;
    static constexpr CB src = A::src;
    static constexpr uint32_t src_num_pages = A::src_num_pages;
    static constexpr bool src_is_tensor_backed = A::src_is_tensor_backed;
};

template <typename A>
struct WriterCTArgs {  // BRISC
    static constexpr uint32_t dest_noc_start_x = A::dest_noc_start_x;
    static constexpr uint32_t dest_noc_start_y = A::dest_noc_start_y;
    static constexpr uint32_t dest_noc_end_x = A::dest_noc_end_x;
    static constexpr uint32_t dest_noc_end_y = A::dest_noc_end_y;
    static constexpr Semaphore sender_semaphore = A::sender_semaphore;
    static constexpr Semaphore receiver_semaphore = A::receiver_semaphore;
    static constexpr uint32_t data_size_bytes = A::data_size_bytes;
    static constexpr CB src = A::src;
    static constexpr uint32_t src_num_pages = A::src_num_pages;
    static constexpr CB dst = A::dst;
    static constexpr uint32_t num_cores = A::num_cores;
    static constexpr bool is_part_of_receiver_grid = A::is_part_of_receiver_grid;
    static constexpr bool linked = A::linked;
    static constexpr bool posted = A::posted;
    static constexpr bool loopback = A::loopback;
};
```

**init() -- persistent sender setup (BRISC only):**

```cpp
void init() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_sender && dm0_cta::src_is_tensor_backed) {
        unified_kernels::setup_sharded_buffer(dm0_cta::src, dm0_cta::src_num_pages);
    }
#elif defined(COMPILE_FOR_BRISC)
    if constexpr (core_cta::is_sender) {
        uint64_t mcast_flag_noc_addr = mcast_detail::get_noc_multicast_addr<noc_index>(
            dm1_cta::dest_noc_start_x, dm1_cta::dest_noc_start_y,
            dm1_cta::dest_noc_end_x, dm1_cta::dest_noc_end_y,
            (uint64_t)dm1_cta::receiver_semaphore);
        volatile tt_l1_ptr uint32_t* sender_sem_ptr =
            (volatile tt_l1_ptr uint32_t*)dm1_cta::sender_semaphore;
        noc_semaphore_set(sender_sem_ptr, INVALID);
        mcast_detail::init_persistent_sender<
            dm1_cta::num_cores, dm1_cta::loopback,
            dm1_cta::is_part_of_receiver_grid,
            dm1_cta::linked, dm1_cta::posted>(
            mcast_flag_noc_addr, dm1_cta::sender_semaphore);
        noc_semaphore_set(sender_sem_ptr, VALID);
    }
#endif
}
```

`init_persistent_sender` programs the NOC command buffer registers (CTRL,
coordinates, address fields) so that subsequent sends only need to update
the source address and fire. This eliminates per-send register setup
overhead, which is significant for back-to-back multicasts.

**operator() -- the actual multicast:**

```cpp
void operator()() {
#if defined(COMPILE_FOR_BRISC)
    if constexpr (core_cta::is_sender) {
        cb_wait_front(dm1_cta::src, dm1_cta::src_num_pages);
        mcast_detail::send_data_mcast<
            dm1_cta::num_cores, dm1_cta::loopback,
            dm1_cta::is_part_of_receiver_grid,
            dm1_cta::linked, dm1_cta::posted,
            dm1_cta::data_size_bytes>(
            get_read_ptr(dm1_cta::src), get_write_ptr(dm1_cta::dst));
        // Signal receivers
        mcast_detail::send_with_state<...>(
            dm1_cta::sender_semaphore, dm1_cta::receiver_semaphore, 4);
        noc_async_posted_writes_flushed();
        if constexpr (core_cta::pop_src) {
            cb_pop_front(dm1_cta::src, dm1_cta::src_num_pages);
        }
    }
#elif defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_receiver) {
        volatile tt_l1_ptr uint32_t* recv_sem_ptr =
            (volatile tt_l1_ptr uint32_t*)dm0_cta::receiver_semaphore;
        cb_reserve_back(dm0_cta::dst, dm0_cta::dst_num_pages);
        noc_semaphore_wait(recv_sem_ptr, VALID);
        noc_semaphore_set(recv_sem_ptr, INVALID);
        cb_push_back(dm0_cta::dst, dm0_cta::dst_num_pages);
    }
#endif
}
```

The sender flow: wait for source data, multicast it, signal the semaphore,
flush, pop source. The receiver flow: reserve destination space, wait for
the semaphore, reset it, push destination.

**run_as<A>() -- reusing persistent state with different args:**

```cpp
template <typename A>
void run_as() {
#if defined(COMPILE_FOR_BRISC)
    if constexpr (core_cta::is_sender) {
        using w = WriterCTArgs<A>;
        cb_wait_front(w::src, w::src_num_pages);
        mcast_detail::send_data_mcast<
            dm1_cta::num_cores, dm1_cta::loopback,
            dm1_cta::is_part_of_receiver_grid,
            dm1_cta::linked, dm1_cta::posted,
            w::data_size_bytes>(
            get_read_ptr(w::src), get_write_ptr(w::dst));
        // ...same signal pattern...
    }
#elif defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_receiver) {
        using r = ReaderCTArgs<A>;
        // ...wait/push with r's args...
    }
#endif
}
```

Note how `run_as` uses `dm1_cta` (the *original* init args) for NOC
configuration (num_cores, loopback, etc.) but `w` / `r` (the *new* args)
for CB IDs, semaphores, and data sizes. This lets a fused kernel invoke the
Mcast with a different set of CT args (different source CB, different data
size, different receiver semaphore) while reusing the persistent NOC command
buffer state (same destination coordinates, same link/posted settings).

**setup_src<A>() -- reconfiguring the source for a follower phase:**

```cpp
template <typename A>
void setup_src() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_sender) {
        using r = ReaderCTArgs<A>;
        unified_kernels::setup_sharded_buffer(r::src, r::src_num_pages);
    }
#endif
}
```

**Typical usage pattern in a fused kernel:**

```cpp
Op<ct_args::act_mcast> m;
m.init();                           // Set up persistent NOC state
m();                                // First mcast (uses ct_args::act_mcast)
m.setup_src<ct_args::weight_mcast>();  // Re-init source for weights
m.run_as<ct_args::weight_mcast>();     // Second mcast with different data
m.teardown();                       // Clean up persistent state
```

**teardown() -- cleaning up persistent state:**

```cpp
void teardown() {
#if defined(COMPILE_FOR_BRISC)
    if constexpr (core_cta::is_sender) {
        mcast_detail::teardown_persistent_sender<
            dm1_cta::num_cores, dm1_cta::loopback,
            dm1_cta::is_part_of_receiver_grid>(
            dm1_cta::sender_semaphore);
    }
#endif
}
```

Teardown issues a non-posted write to flush the pipeline and adds a safety
delay (`riscv_wait(1000)`) to work around a posted-mcast hardware bug. The
teardown redirects the final NOC write to `sender_semaphore` instead of
`receiver_semaphore` to avoid racing with NCRISC's semaphore wait-and-reset
cycle.

### 2.1.3 Mcast Configuration Flags

Three boolean flags control the NOC transfer mode:

| Flag | Default | Effect |
|------|---------|--------|
| `linked` | 1 | VC-linked mode: back-to-back multicasts share a virtual channel, reducing inter-send latency. The NOC command includes `NOC_CMD_VC_LINKED`. |
| `posted` | 1 | Fire-and-forget writes: sender does not wait for per-write ACKs. Faster but requires explicit flush via `noc_async_posted_writes_flushed()`. |
| `loopback` | 0 | Whether the sender includes itself in the multicast destination count (via `NOC_CMD_BRCST_SRC_INCLUDE`). |

These defaults are set in `Mcast.emit()` and can be overridden by callers
through the BRISC CT args.

---

## 2.2 Gather -- Many-to-One

**Source**: `blaze/ops/gather/op.py`, `blaze/ops/gather/kernels/op.hpp`

Gather is the inverse of Mcast: multiple sender cores each write their local
data to a single receiver core. The receiver concatenates the data in
sender-index order.

### 2.2.1 Python Emit

```python
@staticmethod
def emit(
    f: FusedProgram,
    src_handle: CBHandle,
    output_tensor: ttnn.Tensor | None,
    *,
    prefix: str,
    dst_cb: int | CBHandle | None = None,
    dst_num_pages: int | None = None,
    row_major: bool = True,
    dst_tile=None,
    dst_page_size: int | None = None,
    dst_data_format=None,
    receiver_core: "ttnn.CoreCoord | None" = None,
) -> CBHandle:
```

**Key parameters:**

| Parameter | Purpose |
|-----------|---------|
| `src_handle` | `CBHandle` on scattered sender cores |
| `output_tensor` | Destination tensor (can be `None` if `dst_cb` provided) |
| `receiver_core` | Override gather destination. Defaults to `f.sender_grid`. |
| `row_major` | Core ordering for sender enumeration. `True` = row-wise, `False` = column-wise. Determines order of data concatenation at receiver. |
| `dst_cb` | Reuse an existing destination CB instead of allocating one. |

**Receiver core selection.** By default, the gather destination is
`f.sender_grid` (the device's sender core). An optional `receiver_core`
parameter allows overriding this:

```python
if receiver_core is not None:
    recv_grid = ttnn.CoreRangeSet([ttnn.CoreRange(receiver_core, receiver_core)])
    recv_noc = f._ctx.worker_core_from_logical_core(receiver_core)
    recv_logical = (receiver_core.x, receiver_core.y)
else:
    recv_grid = f.sender_grid
    recv_noc = f.noc_sender
    recv_logical = f.grid.sender_core
```

**Sender enumeration.** The sender cores are extracted from the source
CBHandle's core ranges and enumerated:

```python
sender_cores = ttnn.corerange_to_cores(src_handle.core_ranges, row_wise=row_major)
num_senders = len(sender_cores)
```

**Destination CB.** The total number of destination pages is
`num_senders * src_num_pages` by default -- each sender contributes
`src_num_pages` pages:

```python
if dst_num_pages is None:
    dst_num_pages = num_senders * src_num_pages
```

When no output tensor is provided, a scratch CB is allocated on a core range
that includes both sender and receiver cores:

```python
if recv_already_covered:
    scratch_ranges = ttnn.CoreRangeSet(src_ranges)
else:
    scratch_ranges = ttnn.CoreRangeSet(src_ranges + recv_ranges)
dst_cb = f.cb_scratch(
    name=BlazeOp.cb_name(prefix, "dst_cb"),
    num_pages=dst_num_pages,
    core_ranges=scratch_ranges,
    data_format=dst_data_format,
    tile=dst_tile,
    page_size=dst_page_size,
)
```

**Dual-NOC semaphores.** Gather allocates two semaphores, one for each NOC
network:

```python
noc0_sem = f.semaphore(f"{prefix}.noc0")
noc1_sem = f.semaphore(f"{prefix}.noc1")
```

Currently `noc1_num_senders` defaults to 0 (all senders use NOC0). The
infrastructure supports splitting senders across both NOCs for congestion
avoidance.

**Per-core sender index.** Each sender gets a unique index so it writes to
the correct offset in the receiver's buffer:

```python
f.per_core_unified_ct_args([
    (f"{prefix}.sender_idx",
     {core: idx for idx, core in enumerate(sender_cores)}),
])
```

**Receiver data address resolution.** The receiver's destination address is
resolved differently depending on whether the destination is a CBHandle or
a tensor:

```python
cb_addr = dst_cb.buffer_address() if isinstance(dst_cb, CBHandle) else None
if cb_addr is not None:
    receiver_addr_from_dst_cb = False
    receiver_data_addr = cb_addr
elif output_tensor is not None:
    receiver_addr_from_dst_cb = False
    receiver_data_addr = output_tensor.buffer_address()
else:
    receiver_addr_from_dst_cb = True
    receiver_data_addr = 0
```

When `receiver_addr_from_dst_cb` is True, the kernel reads the destination
address from the CB's write pointer at runtime rather than using a
compile-time constant.

### 2.2.2 C++ Kernel

**RISC assignment reversal.** Unlike Mcast, Gather runs sender logic on
**NCRISC** and receiver logic on **BRISC**:

- NCRISC (sender): reads local CB, issues `noc_async_write_one_packet` to
  the receiver, increments the receiver's semaphore.
- BRISC (receiver): waits on semaphore count, pushes destination CB.

**Sender (NCRISC):**

```cpp
void operator()() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_sender) {
        uint32_t core_index;
        if constexpr (core_cta::use_per_core) {
            core_index = dm0_cta::sender_idx;
        } else {
            core_index = unified_kernels::linear_id_in_grid<true>(...);
        }
        uint32_t offset = core_index * dm0_cta::data_size_bytes;

        const uint64_t dst_noc_coord =
            get_noc_addr(dm0_cta::dest_noc_x, dm0_cta::dest_noc_y, 0);
        uint32_t dst_base;
        if constexpr (dm0_cta::receiver_addr_from_dst_cb) {
            dst_base = get_write_ptr(dm0_cta::dst);
        } else {
            dst_base = dm0_cta::receiver_data_addr;
        }
        uint64_t dst_data_noc_addr = dst_noc_coord | (uint64_t)(dst_base + offset);
        uint64_t dst_semaphore_noc_addr =
            dst_noc_coord | (uint64_t)dm0_cta::noc0_receiver_semaphore;

        cb_wait_front(dm0_cta::src, dm0_cta::src_num_pages);
        uint32_t input_data_addr = get_read_ptr(dm0_cta::src);
        noc_async_write_one_packet<true, true>(
            input_data_addr, dst_data_noc_addr, dm0_cta::data_size_bytes);
        noc_semaphore_inc(dst_semaphore_noc_addr, 1);
        noc_async_posted_writes_flushed();

        if constexpr (core_cta::pop_src) {
            cb_pop_front(dm0_cta::src, dm0_cta::src_num_pages);
        }
        noc_async_atomic_barrier();
    }
```

Each sender writes to `dst_base + (sender_idx * data_size_bytes)`, ensuring
non-overlapping regions. The `noc_async_write_one_packet` template parameters
`<true, true>` enable linked mode and static VC. The `noc_async_atomic_barrier()`
call after the sender completes ensures all posted writes are globally visible
before the sender proceeds to any subsequent op.

**Receiver (BRISC):**

```cpp
#elif defined(COMPILE_FOR_BRISC)
    if constexpr (core_cta::is_receiver) {
        volatile tt_l1_ptr uint32_t* noc0_sem_ptr =
            (volatile tt_l1_ptr uint32_t*)dm1_cta::noc0_receiver_semaphore;

        cb_reserve_back(dm1_cta::dst, dm1_cta::dst_num_pages);
        noc_semaphore_wait(noc0_sem_ptr, dm1_cta::noc0_num_senders);
        noc_semaphore_set(noc0_sem_ptr, 0);

        if constexpr (dm1_cta::noc1_num_senders > 0) {
            volatile tt_l1_ptr uint32_t* noc1_sem_ptr =
                (volatile tt_l1_ptr uint32_t*)dm1_cta::noc1_receiver_semaphore;
            noc_semaphore_wait(noc1_sem_ptr, dm1_cta::noc1_num_senders);
            noc_semaphore_set(noc1_sem_ptr, 0);
        }

        cb_push_back(dm1_cta::dst, dm1_cta::dst_num_pages);
    }
#endif
```

The receiver waits until the semaphore count reaches `noc0_num_senders`
(i.e., all senders have signaled). If senders use both NOC networks, it
also waits on the NOC1 semaphore.

### 2.2.3 Gather vs. Mcast: Why the RISC Assignment Differs

Mcast uses BRISC for sending because multicast requires the write command
buffer (`write_cmd_buf`) which is BRISC's domain. Gather uses NCRISC for
sending because it performs unicast `noc_async_write` operations, which are
available on both RISCs, and keeping senders on NCRISC frees BRISC for the
receiver's semaphore wait -- which can overlap with other BRISC work in a
fused program. When composed together (mcast -> compute -> gather), BRISC
handles Mcast send then Gather receive, while NCRISC handles Mcast receive
then Gather send, keeping both RISCs busy.

---

## 2.3 Scatter -- One-to-Many Distribution

**Source**: `blaze/ops/scatter/op.py`, `blaze/ops/scatter/kernels/op.hpp`

Scatter distributes data from one or more source cores to multiple destination
cores. Unlike Mcast (which broadcasts the same data to all cores), Scatter
sends different data to each destination. Each source core writes to its
assigned list of destinations via explicit NOC unicasts.

### 2.3.1 Python Emit

```python
@staticmethod
def emit(
    f: FusedProgram,
    src,
    dest_tensor: ttnn.Tensor,
    *,
    prefix: str = "scatter",
    dest_mapping: dict,
    tiles_per_dest: int,
) -> CBHandle:
```

**Explicit destination mapping.** Unlike Mcast (which targets all cores) or
Gather (which targets a single receiver), Scatter takes an explicit
`dest_mapping` dict:

```python
dest_mapping: {CoreCoord: [CoreCoord, ...]}
```

Each source core maps to its ordered list of destination cores. All source
cores must have the same number of destinations.

**Source layout assumption.** Source data is contiguous per destination: for N
destinations with T tiles each, the first T tiles go to dest 0, the next T
to dest 1, etc. If the producing op stores data in a different layout (e.g.,
interleaved rows across tiles), it must rearrange to contiguous-per-dest
before Scatter consumes it.

**BRISC CT args (sender):**

```python
f.brisc_ct_args([
    (f"{prefix}.src", src_handle),
    (f"{prefix}.num_dests", num_dests_per_src),
    (f"{prefix}.tiles_per_dest", tiles_per_dest),
    (f"{prefix}.payload_bytes", payload_bytes),
    (f"{prefix}.dest_l1_addr", dest_l1_addr),
    (f"{prefix}.arrival_sem_l1_addr", arrival_sem),
])
```

**Per-core runtime arg arrays.** Destination NOC coordinates are passed as
per-core runtime args (not CT args) because each source core has a different
list of destinations:

```python
dest_noc_coordinates = {}
for src_core in src_cores:
    coords = []
    for d in range(num_dests_per_src):
        dc = dest_mapping[src_core][d]
        phys = f.device.worker_core_from_logical_core(dc)
        coords.extend([phys.x, phys.y])
    dest_noc_coordinates[src_core] = coords

f.brisc_per_core_rt_arg_arrays([
    (f"{prefix}.dest_noc_coordinates", dest_noc_coordinates),
])
```

In the kernel, these are accessed via `rt_args::get<>()`:

```cpp
static constexpr rt_args::ArrayArg dest_noc_coordinates = A::dest_noc_coordinates;
```

**skip_src_init flag.** When `src` is a `CBHandle` (from an upstream op),
Scatter sets `skip_src_init` on the source cores to prevent
`setup_sharded_buffer` from pre-pushing pages that would conflict with the
upstream producer:

```python
skip_src = src_core_ranges if src_is_cb else dest_core_ranges
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", src_core_ranges),
    f.flag(f"{prefix}.is_receiver", dest_core_ranges),
    f.flag(f"{prefix}.skip_src_init", skip_src) if src_is_cb else ...,
])
```

### 2.3.2 C++ Kernel

**Sender (BRISC):**

```cpp
void operator()() {
#if defined(COMPILE_FOR_BRISC)
    if constexpr (core_cta::is_sender) {
        constexpr uint32_t src_tiles = num_dests * tiles_per_dest;
        cb_wait_front(src_cb, src_tiles);
        uint32_t src_addr = get_read_ptr(src_cb);

        for (uint32_t d = 0; d < num_dests; d++) {
            uint32_t dest_noc_x = rt_args::get<dm1_cta::dest_noc_coordinates>(d * 2);
            uint32_t dest_noc_y = rt_args::get<dm1_cta::dest_noc_coordinates>(d * 2 + 1);

            uint64_t dest_noc_addr = get_noc_addr(dest_noc_x, dest_noc_y, dest_l1_addr);
            noc_async_write(src_addr + d * payload_bytes, dest_noc_addr, payload_bytes);

            uint64_t sem_noc_addr = get_noc_addr(dest_noc_x, dest_noc_y, arrival_sem_addr);
            noc_semaphore_inc(sem_noc_addr, 1);
        }
        noc_async_posted_writes_flushed();
        noc_async_atomic_barrier();
        cb_pop_front(src_cb, src_tiles);
    }
```

The sender iterates over destinations, writing `payload_bytes` to each and
incrementing the arrival semaphore. The source data is contiguous, so
destination d reads from offset `d * payload_bytes`.

**Receiver (NCRISC):**

```cpp
#elif defined(COMPILE_FOR_NCRISC)
    if constexpr (core_cta::is_receiver) {
        volatile tt_l1_ptr uint32_t* sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_sem_addr);
        noc_semaphore_wait(sem_ptr, 1);
        noc_semaphore_set(sem_ptr, 0);
        unified_kernels::setup_sharded_buffer(dst_cb, tiles_per_dest);
    }
#endif
```

The receiver waits for a single semaphore increment (each dest has exactly
one sender), then calls `setup_sharded_buffer` to push the received data
into the destination CB. Note the receiver calls `setup_sharded_buffer`
after the semaphore wait -- this configures the CB metadata for downstream
consumption.

### 2.3.3 ScatterRaw -- A Specialized Variant

TT-Blaze also includes `ScatterRaw` (`blaze/ops/scatter_raw/op.py`), which
extends the Scatter pattern with:

- **Per-head replication** via a `cph` (cores per head) parameter: each head's
  row is replicated to `cph` destination cores.
- **Broadcast mode** where every destination gets the full source payload
  (all pages) instead of per-destination slicing.
- **Row-oriented** data layout for untilized output, with data size based on
  `num_tile_cols * 32 * 2` (bf16 row bytes) rather than tile-level pages.

ScatterRaw uses the same BRISC-sender / NCRISC-receiver structure and the
same `brisc_per_core_rt_arg_arrays` mechanism for destination coordinates.

---

## 2.4 Copy -- Generic L1 Data Movement

**Source**: `blaze/ops/copy/op.py`, `blaze/ops/copy/kernels/op.hpp`

Copy is the most flexible inter-core op: it handles local copies (same core),
remote copies (different core), CB-to-CB copies, and raw address copies.
It generalizes the older `CbFlush` and `TileCopy` ops.

### 2.4.1 Python Emit

```python
@staticmethod
def emit(
    f: FusedProgram,
    src: CBHandle | ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    prefix: str = "copy",
    wait_src: bool = False,
    pop_src: bool = True,
    push_dst: bool = True,
    processor: Risc = Risc.NCRISC,
    dest_core: ttnn.CoreCoord | None = None,
    src_addr: int | None = None,
    dst_addr: int | None = None,
    num_bytes: int | None = None,
) -> CBHandle:
```

**Configuration matrix:**

| Flag | Default | Purpose |
|------|---------|---------|
| `wait_src` | False | Call `cb_wait_front` before reading source |
| `pop_src` | True | Call `cb_pop_front` after copy |
| `push_dst` | True | Call `cb_reserve_back`/`cb_push_back` per page on dest |
| `processor` | Risc.NCRISC | Which RISC runs the copy |
| `dest_core` | None | Remote destination core (None = local copy) |
| `src_addr` | None | Raw L1 source address (bypasses CB) |
| `dst_addr` | None | Raw L1 destination address (bypasses CB) |

**Common usage patterns:**

```python
# CbFlush equivalent: wait for TRISC-written scratch CB, flush to output
Copy.emit(f, scratch_cb, output, wait_src=True, prefix="flush")

# TileCopy equivalent: copy sharded tensor to output (no wait)
Copy.emit(f, tensor, output, prefix="tile_copy")

# Remote copy to another core
Copy.emit(f, src_cb, output, dest_core=CoreCoord(1, 0), prefix="remote")

# Run on BRISC instead of NCRISC
Copy.emit(f, src_cb, output, processor=Risc.BRISC, prefix="copy")
```

**CT arg registration.** Copy uses `f.unified_ct_args()` (not RISC-specific
methods) because both BRISC and NCRISC compile the same `copy_impl()` code,
and only one executes it based on `use_ncrisc`:

```python
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_active", src_cb.core_ranges),
])

f.unified_ct_args([
    (f"{prefix}.src", src_cb),
    (f"{prefix}.dst", dst_cb),
    (f"{prefix}.num_pages", src_cb.num_pages),
    (f"{prefix}.num_bytes", num_bytes),
    (f"{prefix}.src_addr", src_addr if src_addr is not None else 0),
    (f"{prefix}.dst_addr", dst_addr if dst_addr is not None else 0),
    (f"{prefix}.dest_noc_x", dest_noc_x),
    (f"{prefix}.dest_noc_y", dest_noc_y),
    (f"{prefix}.wait_src", int(wait_src)),
    (f"{prefix}.pop_src", int(pop_src)),
    (f"{prefix}.push_dst", int(push_dst)),
    (f"{prefix}.src_is_cb", int(src_is_cb)),
    (f"{prefix}.dst_is_cb", int(dst_is_cb)),
    (f"{prefix}.is_remote", int(is_remote)),
    (f"{prefix}.use_ncrisc", int(processor == Risc.NCRISC)),
])
```

### 2.4.2 C++ Kernel

Copy uses a single `CoreCTArgs` struct (no separate Reader/Writer structs):

```cpp
template <typename A>
struct CoreCTArgs {
    static constexpr Flag is_active = A::is_active;
    static constexpr CB src = A::src;
    static constexpr CB dst = A::dst;
    static constexpr uint32_t num_pages = A::num_pages;
    static constexpr uint32_t num_bytes = A::num_bytes;
    static constexpr uint32_t src_addr = A::src_addr;
    static constexpr uint32_t dst_addr = A::dst_addr;
    static constexpr uint32_t dest_noc_x = A::dest_noc_x;
    static constexpr uint32_t dest_noc_y = A::dest_noc_y;
    static constexpr bool wait_src = A::wait_src;
    static constexpr bool pop_src = A::pop_src;
    static constexpr bool push_dst = A::push_dst;
    static constexpr bool src_is_cb = A::src_is_cb;
    static constexpr bool dst_is_cb = A::dst_is_cb;
    static constexpr bool is_remote = A::is_remote;
    static constexpr bool use_ncrisc = A::use_ncrisc;
};
```

**RISC selection:**

```cpp
void operator()() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (cta::use_ncrisc) { copy_impl(); }
#elif defined(COMPILE_FOR_BRISC)
    if constexpr (!cta::use_ncrisc) { copy_impl(); }
#endif
}
```

**copy_impl() -- the unified copy logic (four modes via if constexpr):**

1. **Local CB-to-CB with push**: Page-by-page `cb_reserve_back` / memcpy /
   `cb_push_back`. This is the CbFlush replacement pattern.

2. **Local raw copy**: Direct `uint32_t*` word-copy loop. No CB sync.

3. **Remote CB-to-CB with push**: Per-page `cb_reserve_back`, then
   `noc_async_write` to the remote core's L1 address with
   `noc_async_write_barrier`.

4. **Remote bulk copy**: Single `noc_async_write` for the full payload.

The compile-time flag resolution means only one of these code paths exists
in the final binary for any given Copy instance.

---

## 2.5 Composing Inter-Core Ops with Compute Ops

Inter-core ops are designed to compose with compute ops in fused kernels.
The key mechanism is CBHandle chaining: one op's output CBHandle becomes the
next op's input. The CB acts as the synchronization boundary.

### 2.5.1 EmbedMcast: Embedding + Mcast

The simplest composition example (`blaze/ops/embed_mcast/op.py`):

```python
class EmbedMcast(FusedOp):
    name: str = "embed_mcast"

    @staticmethod
    def emit(f, token_ids, *, prefix="embed_mcast",
             weight_buffer_address, weight_page_size) -> CBHandle:
        emb = Embedding.emit(
            f, token_ids,
            prefix=BlazeOp.child_prefix(prefix, "embedding"),
            cores=f.sender_grid,
            weight_buffer_address=weight_buffer_address,
            weight_page_size=weight_page_size,
        )
        return Mcast.emit(f, emb, prefix=BlazeOp.child_prefix(prefix, "mcast"))
```

The Embedding op runs on `f.sender_grid` (the sender core) and returns a
CBHandle. This CBHandle is passed directly to `Mcast.emit()` as the source.
Since `src` is a CBHandle, Mcast skips `setup_sharded_buffer` initialization
and the `init_src` flag is set to False.

The `BlazeOp.child_prefix()` helper creates namespaced CT arg prefixes to
prevent name collisions between the two sub-ops.

### 2.5.2 CB-Based Synchronization Flow

A typical fused pipeline:

```
[Mcast] -> [dst CB on all cores] -> [Matmul reads from dst CB]
                                        |
                                    [result CB]
                                        |
[Gather] <- [result CB on all cores] <-+
```

No explicit synchronization code is needed between these ops -- the CB
producer/consumer protocol handles it automatically:

- BRISC (Mcast sender) writes data and signals the receiver semaphore.
- NCRISC (Mcast receiver) waits for the semaphore, pushes the destination CB.
- TRISC (Matmul compute) calls `cb_wait_front()` on the Mcast destination CB,
  which blocks until NCRISC's `cb_push_back()` completes.
- TRISC writes results to the output CB via `cb_push_back()`.
- NCRISC (Gather sender) calls `cb_wait_front()` on the output CB, then NOC
  writes to the receiver.
- BRISC (Gather receiver) waits for the semaphore, pushes the gathered CB.

### 2.5.3 Composition Guidelines

When composing inter-core ops with compute ops:

1. **CBHandle chaining**: Pass the output CBHandle of one op as the input to
   the next. No intermediate tensor allocation is needed.

2. **Prefix namespacing**: Use `BlazeOp.child_prefix(parent, child)` to
   create non-colliding CT arg names.

3. **Core range awareness**: Upstream ops may produce data on a subset of cores.
   The downstream op must account for this (e.g., Mcast's sender is always
   `f.sender_grid`, so the upstream op should produce data there).

4. **Init skipping**: When an upstream op pushes data into a CB, the downstream
   op must not re-initialize it with `setup_sharded_buffer`. The `isinstance(src, CBHandle)`
   check in Mcast and Gather handles this automatically.

---

## 2.6 Summary Table

| Feature | Mcast | Gather | Scatter | ScatterRaw | Copy |
|---------|-------|--------|---------|------------|------|
| **Topology** | 1-to-all | many-to-1 | 1-to-many | 1-to-many (row) | 1-to-1 |
| **Sender RISC** | BRISC | NCRISC | BRISC | BRISC | configurable |
| **Receiver RISC** | NCRISC | BRISC | NCRISC | NCRISC | N/A |
| **NOC operation** | multicast write | unicast write | unicast write | unicast write | unicast write or local memcpy |
| **Semaphore type** | binary (VALID/INVALID) | counting (N senders) | binary (arrival) | binary (arrival) | none |
| **Persistent state** | yes (init/teardown) | no | no | no | no |
| **Supports run_as<>()** | yes | no | no | no | no |
| **Dest coords source** | f.noc_start/end (CT) | f.noc_sender (CT) | RT arg arrays | RT arg arrays | CT args |
| **Tile format conversion** | via dst_tile_info | via dst_tile/page_size | implicit | implicit | implicit |
| **TRISC involvement** | none | none | none | none | none |
| **Linked/posted** | yes | N/A | N/A | N/A | N/A |
| **CT arg method** | brisc + ncrisc | brisc + ncrisc | brisc + ncrisc | brisc + ncrisc | unified |
| **Typical use** | broadcast activations | reduce partial sums | distribute weight rows | attention head scatter | flush scratch CBs |
