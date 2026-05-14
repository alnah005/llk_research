# Section 1: The Sender/Receiver Pattern and NOC Fundamentals

## 1.1 Why Inter-Core Data Movement Matters

A Tenstorrent Wormhole device contains a grid of Tensix cores (typically 8x8 or
8x4 usable workers). Each core has its own L1 SRAM, compute units (TRISC), and
two data-movement processors (BRISC and NCRISC). There is no shared memory
between cores. When an operation like RMSNorm needs its input replicated across
every core, or a matmul scatter needs to distribute weight rows, data must move
through the Network-on-Chip (NOC).

Every inter-core data movement in TT-Blaze follows a single architectural
pattern:

1. A **sender** core reads data from its local L1 (usually from a circular
   buffer) and writes it to one or more remote cores over the NOC.
2. The sender increments a **semaphore** on the remote core to signal that the
   data has landed.
3. The **receiver** core waits on that semaphore, then consumes the data from
   its own local L1 circular buffer.

TT-Blaze abstracts this movement behind a small set of micro-ops (Mcast,
Gather, Scatter, Copy) that share this common sender/receiver pattern.
Understanding it is essential before diving into any specific inter-core op.

---

## 1.2 The Dual-NOC Architecture: NOC0 and NOC1

Tenstorrent devices provide **two independent NOC networks**: NOC0 and NOC1.
Each core has a port on both networks, and each network supports unicast reads,
unicast writes, and multicast writes. They are completely independent: traffic
on NOC0 cannot interfere with traffic on NOC1, and vice versa.

### NOC Index and Processor Affinity

In TT-Blaze's unified kernel model, BRISC (the "writer" data-movement
processor) typically uses one NOC and NCRISC (the "reader") uses the other.
The conventional mapping is:

| Network | Index | Default owner | Typical role |
|---------|-------|---------------|--------------|
| NOC0    | 0     | NCRISC (Reader/DM0) | Reads, unicast writes, semaphore signals |
| NOC1    | 1     | BRISC (Writer/DM1)  | Multicast writes, semaphore signals |

This dual-network design allows sender and receiver traffic to flow
simultaneously without contention. The Mcast op leverages this: BRISC sends
data over NOC while NCRISC waits on the semaphore, with no cross-network
dependency.

The key reason for having two NOCs is **congestion avoidance**. A typical
technique is to split read traffic onto one NOC and write traffic onto the
other. The Gather op demonstrates this by providing separate semaphore pairs
for each network:

```python
# From blaze/ops/gather/op.py -- two semaphores, one per NOC
noc0_sem = f.semaphore(f"{prefix}.noc0")
noc1_sem = f.semaphore(f"{prefix}.noc1")
```

### NOC Coordinate Translation

Each core has **logical coordinates** (column, row in the software grid) and
**physical NOC coordinates** (x, y on the hardware mesh, which may differ due
to harvesting and routing). Python emit code must translate between them.

The translation is done by `worker_core_from_logical_core()`, accessed
through the device or mesh_device object:

```python
# FusedProgram precomputes common translations at init time:
self.noc_sender = self._ctx.worker_core_from_logical_core(self.sender)
self.noc_start = self._ctx.worker_core_from_logical_core(self.mcast_range.start)
self.noc_end = self._ctx.worker_core_from_logical_core(self.mcast_range.end)
```

The `noc_start` and `noc_end` properties give the bounding box of the multicast
rectangle in NOC coordinates. These are passed to the kernel as CT args so the
hardware multicast engine knows which cores to target.

For arbitrary core-to-core transfers (Gather, Scatter, Copy), the emit code
calls `worker_core_from_logical_core` directly:

```python
# From Gather.emit():
recv_noc = f._ctx.worker_core_from_logical_core(receiver_core)

# From Copy.emit():
phys = f.device.worker_core_from_logical_core(dest_core)
dest_noc_x = phys.x
dest_noc_y = phys.y

# From AllReduce.emit():
data_core_physical = mesh_device.worker_core_from_logical_core(data_core)
sender_core_physical = mesh_device.worker_core_from_logical_core(sender_core)
```

**Key rule**: Never pass logical coordinates to a kernel as NOC addresses.
Always use `worker_core_from_logical_core()` to convert, and pass the `.x`
and `.y` fields of the resulting physical coordinate.

### NOC Multicast Address Construction (C++)

In the kernel, the NOC multicast address encodes source and destination
coordinate ranges plus the target L1 address. The Mcast kernel provides a
helper that handles the coordinate reversal required by NOC1:

```cpp
// From blaze/ops/mcast/kernels/op.hpp -- mcast_detail namespace
template <uint8_t noc>
FORCE_INLINE uint64_t get_noc_multicast_addr(
    uint32_t noc_x_start, uint32_t noc_y_start,
    uint32_t noc_x_end, uint32_t noc_y_end, uint32_t addr) {
    if constexpr (noc == 0) {
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(noc, noc_x_start), DYNAMIC_NOC_Y(noc, noc_y_start),
            DYNAMIC_NOC_X(noc, noc_x_end), DYNAMIC_NOC_Y(noc, noc_y_end), addr);
    } else {
        // NOC1: swap start/end coordinates
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(noc, noc_x_end), DYNAMIC_NOC_Y(noc, noc_y_end),
            DYNAMIC_NOC_X(noc, noc_x_start), DYNAMIC_NOC_Y(noc, noc_y_start), addr);
    }
}
```

The `DYNAMIC_NOC_X` / `DYNAMIC_NOC_Y` macros apply the physical-to-NOC
translation at the hardware level. The key insight: NOC0 and NOC1 use
**opposite coordinate orderings** for multicast ranges, so any code that
supports both NOCs must swap start and end. This is a hardware requirement --
the two NOCs route in opposite directions.

---

## 1.3 Semaphore Protocols

Semaphores are the only synchronization primitive available between cores.
They are 32-bit L1 memory locations with hardware-accelerated atomic
operations. TT-Blaze defines four semaphore protocol identifiers in the
`SemProtocol` enum (`blaze/blaze_op.py`):

```python
class SemProtocol(str, Enum):
    """Valid semaphore protocol identifiers."""
    SENDER = "sender_semaphore"
    RECEIVER = "receiver_semaphore"
    NOC0_RECEIVER = "noc0_receiver_semaphore"
    NOC1_RECEIVER = "noc1_receiver_semaphore"
```

These protocols label the *role* that a semaphore plays in the CT arg structs.
They help the code generator map semaphore CT args to the correct `Semaphore`
type in C++:

| Protocol | Owner | Purpose |
|----------|-------|---------|
| `SENDER` | Sender core | Sender sets this to `VALID` after `init()` to signal readiness. Mcast uses this to gate persistent sender state. |
| `RECEIVER` | Receiver core(s) | Sender writes (increments) this on the remote receiver to signal data arrival. Receiver waits for it, then resets. |
| `NOC0_RECEIVER` | Receiver core | Same as `RECEIVER`, but specifically for NOC0-routed traffic. Gather uses this when senders write via NOC0. |
| `NOC1_RECEIVER` | Receiver core | Companion to `NOC0_RECEIVER` for NOC1-routed traffic. |

### The Wait/Set/Inc Protocol

Three hardware operations underpin all semaphore-based sync:

| Operation | C++ API | Semantics |
|-----------|---------|-----------|
| **wait**  | `noc_semaphore_wait(ptr, val)` | Spin until `*ptr >= val`, then continue |
| **set**   | `noc_semaphore_set(ptr, val)` | Atomically write `val` to `*ptr` (local L1) |
| **inc**   | `noc_semaphore_inc(noc_addr, val)` | Atomically add `val` to the semaphore at `noc_addr` (remote or local) |

The standard protocol for inter-core data movement uses a pair of semaphores:

**Receiver side (NCRISC):**
```cpp
// Wait for sender to signal data arrival
volatile tt_l1_ptr uint32_t* recv_sem_ptr =
    (volatile tt_l1_ptr uint32_t*)dm0_cta::receiver_semaphore;
cb_reserve_back(dm0_cta::dst, dm0_cta::dst_num_pages);
noc_semaphore_wait(recv_sem_ptr, VALID);    // block until VALID
noc_semaphore_set(recv_sem_ptr, INVALID);   // reset for next iteration
cb_push_back(dm0_cta::dst, dm0_cta::dst_num_pages);
```

**Sender side (BRISC):**
```cpp
// After writing data via NOC multicast...
// Signal all receivers by multicasting the sender semaphore value
mcast_detail::send_with_state<...>(
    dm1_cta::sender_semaphore,
    dm1_cta::receiver_semaphore, 4);
noc_async_posted_writes_flushed();
```

The critical ordering is:
1. Receiver calls `cb_reserve_back` to ensure the destination CB has space.
2. Receiver calls `noc_semaphore_wait` with the expected value.
3. Sender writes data to the receiver's destination CB L1 address.
4. Sender multicasts the semaphore increment (or writes VALID) to all receivers.
5. Receiver wakes up, resets the semaphore, and calls `cb_push_back`.

### Gather's Counting Semaphore Variant

Gather uses a different protocol because *multiple* senders signal a *single*
receiver. Instead of waiting for `VALID` (1), the receiver waits for a count
equal to the number of senders:

```cpp
// Gather receiver (BRISC):
noc_semaphore_wait(noc0_sem_ptr, dm1_cta::noc0_num_senders);
noc_semaphore_set(noc0_sem_ptr, 0);
```

Each sender increments the semaphore by 1 after writing its data:

```cpp
// Gather sender (NCRISC):
noc_semaphore_inc(dst_semaphore_noc_addr, 1);
```

This counting protocol naturally handles N:1 gather without the receiver
needing to know which specific senders have completed.

Gather also reserves a second semaphore (`noc1_receiver_semaphore`) for cases
where senders are split across both NOC networks:

```cpp
if constexpr (dm1_cta::noc1_num_senders > 0) {
    volatile tt_l1_ptr uint32_t* noc1_sem_ptr =
        (volatile tt_l1_ptr uint32_t*)dm1_cta::noc1_receiver_semaphore;
    noc_semaphore_wait(noc1_sem_ptr, dm1_cta::noc1_num_senders);
    noc_semaphore_set(noc1_sem_ptr, 0);
}
```

In the current Gather implementation, `noc1_num_senders` defaults to 0, so all
senders use NOC0. The split is a compile-time decision -- the infrastructure
is ready for ops that distribute senders across both NOCs.

### Semaphore Allocation in Python

The `f.semaphore()` method provides two allocation modes:

**Named (mesh-global) semaphores.** The default mode. Pass a name string, and
the semaphore is deduplicated across all `FusedProgram` instances sharing a
`_sem_dict`. They return an L1 address that is the same across all devices in
a mesh -- critical for cross-device rendezvous (CCL ops):

```python
sender_sem = f.semaphore(f"{prefix}.sender")
receiver_sem = f.semaphore(f"{prefix}.receiver")
```

**Program (slot-based) semaphores.** For per-device-local synchronization,
pass `program_semaphore=True`. These use tt-metal's program semaphore slots,
which are scoped per-core and per-program. Passing `core_ranges` restricts the
semaphore to specific cores, letting tt-metal reuse slot IDs across disjoint
cores:

```python
# From blaze/ccl.py -- fabric connection setup
teardown_sems = [
    fp.semaphore(program_semaphore=True, core_ranges=worker_core_ranges)
    for _ in range(num_connections)
]
```

The naming convention `f"{prefix}.purpose"` scopes semaphore names to their
op instance, preventing collisions when multiple ops are fused into a single
kernel. Each call to `f.semaphore(name)` within a single op must use a unique
name; duplicate names raise a `RuntimeError`.

---

## 1.4 The Sender/Receiver Pattern

Every inter-core TT-Blaze op follows this three-phase pattern:

```
Phase 1: ROLE ASSIGNMENT (compile time)
    Per-core flags determine which branch each RISC executes.

Phase 2: DATA MOVEMENT (runtime)
    Sender: wait for source CB -> NOC write/multicast -> signal semaphore
    Receiver: wait on semaphore -> push destination CB

Phase 3: CLEANUP
    Pop source CBs, reset semaphores, flush NOC writes.
```

### Role Assignment via Flags

Every inter-core op needs each core to know its role at compile time. This
is implemented via per-core boolean flags set in the unified CT args. The
`f.flag()` helper builds the mapping:

```python
# From FusedProgram.flag():
def flag(self, name, cores, enabled=True):
    """Build a per-core CT arg tuple for a boolean flag.

    Returns a (name, mapping) tuple suitable for passing to
    per_core_unified_ct_args() and variants. The flag is 1 on the
    given cores when enabled, 0 when disabled. Cores not in the range
    always get 0 (via other_value default).
    """
    return (name, {cores: int(enabled)})
```

Mcast uses four flags to partition the core grid:

```python
# From Mcast.emit():
f.per_core_unified_ct_args([
    f.flag(f"{prefix}.is_sender", f.sender_grid),
    f.flag(f"{prefix}.is_receiver", f.mcast_receiver_grid),
    f.flag(f"{prefix}.pop_src", f.all_cores),
    f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
])
```

Cores in the specified range get value 1 (or the `enabled` value); all other
cores get 0. In the C++ kernel, these appear as `static constexpr Flag`
fields:

```cpp
// From Mcast kernel, CoreCTArgs:
template <typename A>
struct CoreCTArgs {
    static constexpr Flag is_sender = A::is_sender;
    static constexpr Flag is_receiver = A::is_receiver;
    static constexpr Flag pop_src = A::pop_src;
    static constexpr Flag init_src = A::init_src;
};
```

Because each core is compiled with its own set of CT arg values, the
`if constexpr` branches are resolved at compile time. A core that is not a
sender will have `is_sender == 0`, and the entire sender code path will be
dead-code-eliminated. This means **zero runtime overhead** for role checking,
and eliminates both the binary size of unused code paths and any risk of
a receiver core executing sender NOC writes (which would corrupt memory) or
a sender waiting on a receiver semaphore (which would deadlock).

Different ops use different flag names depending on their semantics:

| Op | Flags | Meaning |
|----|-------|---------|
| Mcast | `is_sender`, `is_receiver`, `pop_src`, `init_src` | Sender multicasts; receiver waits; pop controls CB lifecycle; init controls sharded buffer setup |
| Gather | `is_sender`, `is_receiver`, `pop_src`, `use_per_core` | Sender writes to receiver; use_per_core selects per-core sender_idx for offset computation |
| Scatter | `is_sender`, `is_receiver`, `skip_src_init` | Sender distributes; skip_src_init avoids pre-pushing when src is a CBHandle |
| Copy | `is_active` | Single flag -- active cores execute, others skip |
| AllReduce | `is_sender_core`, `is_receiver_core` | Two distinct core roles; sender reads from neighbor, receiver reduces |

### Sender Grid vs. Receiver Grid

The FusedProgram precomputes several grid definitions:

```python
# From FusedProgram.__init__():
self.sender = ttnn.CoreCoord(*self.grid.sender_core)
self.sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(self.sender, self.sender)])

self.mcast_range = ttnn.CoreRange(
    ttnn.CoreCoord(0, 0),
    ttnn.CoreCoord(self.grid.grid_cols - 1, self.grid.grid_rows - 1),
)
self.all_cores = ttnn.CoreRangeSet([self.mcast_range])
self.num_mcast_cores = self.grid.total_cores

self.mcast_receiver_grid = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(c, r), ttnn.CoreCoord(c, r))
    for r in range(self.grid.grid_rows)
    for c in range(self.grid.grid_cols)
    if (c, r) != self.grid.sender_core
])
```

| Field | Type | Description |
|-------|------|-------------|
| `f.sender` | `ttnn.CoreCoord` | Logical coordinate of the sender core |
| `f.sender_grid` | `ttnn.CoreRangeSet` | Single-core range set for the sender |
| `f.mcast_range` | `ttnn.CoreRange` | Bounding rectangle of all logical cores |
| `f.all_cores` | `ttnn.CoreRangeSet` | All cores in the grid |
| `f.num_mcast_cores` | `int` | Total core count |
| `f.mcast_receiver_grid` | `ttnn.CoreRangeSet` | All cores except sender |
| `f.noc_sender` | Physical coord | NOC address of sender core |
| `f.noc_start` | Physical coord | NOC address of grid corner (0,0) |
| `f.noc_end` | Physical coord | NOC address of grid corner (cols-1, rows-1) |

The sender core may or may not also be a receiver. The Mcast op tracks this
with `is_part_of_receiver_grid`:

```python
# From Mcast.emit():
(f"{prefix}.is_part_of_receiver_grid",
 int(f.mcast_range.contains(f.sender))),
```

When the sender is within the multicast rectangle, the hardware multicast
includes it automatically. The kernel uses this flag to adjust the destination
count for non-posted write acknowledgments.

---

## 1.5 Data Size Computation from CBHandle

Every inter-core op must know how many bytes to transfer. TT-Blaze computes
this from the CBHandle's properties, following a consistent pattern:

```
data_size_bytes = num_pages * page_size
```

### Mcast

```python
# From Mcast.emit():
num_pages = src_handle.num_pages
tile_size = src_handle.page_size
data_size_bytes = num_pages * tile_size
```

These values are passed as CT args and become compile-time constants in the
kernel. For Mcast, the `data_size_bytes` drives the chunked multicast
template:

```cpp
// Compile-time unrolled multicast in chunks of NOC_MAX_BURST_SIZE:
template <..., uint32_t num_bytes, uint32_t offset = 0>
FORCE_INLINE void send_data_mcast(uint32_t src_local_addr, uint32_t dst_local_addr) {
    if constexpr (num_bytes > NOC_MAX_BURST_SIZE) {
        send_with_state<...>(
            src_local_addr + offset, dst_local_addr + offset, NOC_MAX_BURST_SIZE);
        send_data_mcast<..., num_bytes - NOC_MAX_BURST_SIZE,
                         offset + NOC_MAX_BURST_SIZE>(src_local_addr, dst_local_addr);
    } else {
        send_with_state<...>(
            src_local_addr + offset, dst_local_addr + offset, num_bytes);
    }
}
```

Because `num_bytes` is a template parameter, the compiler fully unrolls the
loop. For a 2048-byte transfer with a 1024-byte burst limit, this generates
exactly two `send_with_state` calls with zero runtime branching.

### Gather

```python
# From Gather.emit():
src_num_pages = src_handle.num_pages
data_size_bytes = src_num_pages * src_handle.page_size
```

Each sender writes `data_size_bytes` at an offset of
`sender_idx * data_size_bytes` into the receiver's destination buffer.

### Copy

```python
# From Copy.emit():
if src_is_cb:
    num_bytes = src_cb.num_pages * src_cb.page_size
elif num_bytes is None:
    raise ValueError("num_bytes is required when src_addr is provided")
```

When the source is a raw L1 address (not a CB), `num_bytes` must be provided
explicitly by the caller.

### CBHandle as Source

The source can be either a `ttnn.Tensor` (which gets converted to a CBHandle
via `f.cb_from_tensor()`) or an already-existing CBHandle from an upstream op:

```python
# From Mcast.emit():
if isinstance(src, CBHandle):
    src_handle = src
else:
    src_handle = f.cb_from_tensor(src)
```

When the source is a CBHandle (e.g., a scratch buffer filled by TRISC), the
Mcast op skips the NCRISC `setup_sharded_buffer` initialization because the
data was already pushed into the CB by the upstream op:

```python
needs_init = not isinstance(src, CBHandle)
f.per_core_unified_ct_args([
    ...
    f.flag(f"{prefix}.init_src", f.sender_grid, needs_init),
])
```

---

## 1.6 RISC Processor Roles in Inter-Core Ops

The assignment of RISC processors to sender/receiver roles follows hardware
constraints and NOC ownership:

| Op | Sender RISC | Receiver RISC | Rationale |
|----|-------------|---------------|-----------|
| Mcast | BRISC (DM1) | NCRISC (DM0) | BRISC owns the multicast write command buffer |
| Gather | NCRISC (DM0) | BRISC (DM1) | Frees BRISC for receiver semaphore wait; overlaps with other BRISC work |
| Scatter | BRISC (DM1) | NCRISC (DM0) | BRISC sends via NOC writes; NCRISC waits and calls setup_sharded_buffer |
| Copy | Configurable | -- | `processor` parameter selects NCRISC or BRISC |
| AllReduce | NCRISC reads, BRISC sends via fabric | NCRISC waits | Complex: three RISCs involved including TRISC for compute |

### BRISC (DM1 / Writer)

- **Mcast sender**: BRISC owns the multicast send path. It waits for the
  source CB, issues the multicast via NOC write commands, and signals receivers.
- **Gather receiver**: BRISC waits on the semaphore(s) and pushes the
  destination CB.
- **Scatter sender**: BRISC reads source data and issues per-destination
  unicast NOC writes.

BRISC CT args are registered via `f.brisc_ct_args()` and appear in the
kernel as `WriterCTArgs<A>` (aliased to `dm1_cta`).

### NCRISC (DM0 / Reader)

- **Mcast receiver**: NCRISC waits on the receiver semaphore and pushes the
  destination CB.
- **Gather sender**: NCRISC reads from its local CB and NOC-writes to the
  receiver core.
- **Scatter receiver**: NCRISC waits on the arrival semaphore and calls
  `setup_sharded_buffer`.

NCRISC CT args are registered via `f.ncrisc_ct_args()` and appear in the
kernel as `ReaderCTArgs<A>` (aliased to `dm0_cta`).

### TRISC (Compute)

Pure dataflow ops (Mcast, Gather, Scatter, Copy) do not use TRISC. The
AllReduce op is an exception -- it uses TRISC for the element-wise addition
of local and remote data. TRISC CT args use `f.trisc_ct_args()` and appear
as `ComputeCTArgs<A>`.

### The Copy Exception

The Copy op is unique in allowing the caller to choose which RISC runs the
data movement:

```python
# From Copy.emit():
processor: Risc = Risc.NCRISC,  # default
```

In the kernel, both BRISC and NCRISC compile the copy logic, but only one
executes it:

```cpp
void operator()() {
#if defined(COMPILE_FOR_NCRISC)
    if constexpr (cta::use_ncrisc) { copy_impl(); }
#elif defined(COMPILE_FOR_BRISC)
    if constexpr (!cta::use_ncrisc) { copy_impl(); }
#endif
}
```

### RISC Assignment Rationale: Mcast + Gather Composition

When Mcast and Gather are composed in the same fused kernel (e.g., mcast
weights to all cores, compute, gather results), the sender and receiver duties
are naturally split across the two data movement RISCs. BRISC handles the Mcast
send, and then BRISC handles the Gather receive -- while NCRISC handles the
Mcast receive and then the Gather send. Neither RISC is idle.

---

## 1.7 Unified vs. Per-RISC CT Args

Inter-core ops use a mix of CT arg registration methods:

| Method | Scope | Typical use |
|--------|-------|-------------|
| `f.unified_ct_args()` | All RISCs, all cores | Constants shared everywhere |
| `f.per_core_unified_ct_args()` | All RISCs, varies per core | Role flags (is_sender, is_receiver) |
| `f.brisc_ct_args()` | BRISC only, all cores | Sender NOC coordinates, semaphore addresses |
| `f.ncrisc_ct_args()` | NCRISC only, all cores | Source CB, receiver semaphore, page counts |
| `f.trisc_ct_args()` | TRISC only, all cores | Compute parameters (AllReduce only) |
| `f.per_core_ncrisc_ct_args()` | NCRISC only, varies per core | Per-core fabric arg index |

The choice depends on two axes:

1. **Which RISC needs the value?** Sender-only args go to the sender RISC;
   receiver-only args go to the receiver RISC. Shared values (role flags)
   go to `unified`.

2. **Is the value the same on every core?** Per-core values (role flags,
   sender indices) use the `per_core_*` variants with a
   `{CoreRangeSet: value}` mapping. Uniform values use the non-per-core
   variants.

### Per-Core Sender Index Example (Gather)

Gather assigns each sender core a unique index so they write to non-overlapping
offsets in the receiver's destination buffer:

```python
# From Gather.emit():
sender_cores = ttnn.corerange_to_cores(src_handle.core_ranges, row_wise=row_major)

f.per_core_unified_ct_args([
    (f"{prefix}.sender_idx",
     {core: idx for idx, core in enumerate(sender_cores)}),
])
```

In the kernel, the sender computes its offset:

```cpp
uint32_t core_index;
if constexpr (core_cta::use_per_core) {
    core_index = dm0_cta::sender_idx;
} else {
    core_index = unified_kernels::linear_id_in_grid<true>(...);
}
uint32_t offset = core_index * dm0_cta::data_size_bytes;
```

Each core sees its own value at compile time. The sender computes its offset
into the receiver's buffer as `core_index * data_size_bytes`.

---

## 1.8 The `f.output()` Registration Step

Every inter-core op concludes its emit by calling `f.output()` to register the
result with the FusedProgram's dataflow graph:

```python
# From Mcast.emit():
return f.output("mcast", dst, prefix=prefix, src=src)

# From Gather.emit():
return f.output(
    "gather", dst_cb,
    num_pages=dst_num_pages,
    core_ranges=recv_grid,
    grid=src_handle.core_ranges,
    prefix=prefix,
    src=src_handle,
    receiver=recv_logical,
)
```

The first positional argument is the op type name. The second is the CBHandle
that downstream ops will consume. Additional keyword arguments provide metadata
that the compiler uses for scheduling, CB reuse, and error checking.

---

## 1.9 Putting It All Together: Anatomy of an Inter-Core Transfer

Consider a simplified Mcast transfer. The complete lifecycle is:

1. **Python emit**: `Mcast.emit()` is called during FusedProgram construction.
   It allocates source and destination CBs, allocates semaphores, computes
   NOC coordinates, sets role flags, and registers CT args.

2. **Code generation**: The CT args become compile-time constants in the C++
   kernel. Each core gets its own set of values.

3. **BRISC init()**: The sender core initializes the persistent multicast
   state -- configures NOC command buffer registers with the multicast
   bounding box, sets the sender semaphore to VALID.

4. **BRISC operator()()**: The sender waits for data in the source CB
   (`cb_wait_front`), multicasts it to all receivers (`send_data_mcast`),
   multicasts the sender semaphore value to all receivers' receiver semaphore
   addresses, flushes writes, and pops the source CB.

5. **NCRISC operator()()**: Each receiver reserves space in its destination CB
   (`cb_reserve_back`), waits for the receiver semaphore to become VALID
   (`noc_semaphore_wait`), resets the semaphore to INVALID, and pushes the
   destination CB (`cb_push_back`).

6. **Downstream compute**: A subsequent compute op can now `cb_wait_front` on
   the receiver's destination CB and process the data.

7. **BRISC teardown()**: The sender tears down the persistent multicast state,
   cleaning up NOC command buffer registers.

This lifecycle is the blueprint for all inter-core ops. Gather reverses the
direction (many senders, one receiver). Copy reduces to a single core.
Scatter extends to explicit per-destination mapping. CCL ops add fabric
connections on top.

---

## 1.10 Common Pitfalls

### Forgetting NOC Coordinate Translation

Logical core coordinates and NOC coordinates are **not** the same. Passing a
logical coordinate as a NOC address will write to the wrong core (or an
unmapped address). Always translate via `worker_core_from_logical_core` in
Python or use the `DYNAMIC_NOC_X`/`DYNAMIC_NOC_Y` macros in C++.

### Semaphore Name Reuse

Each call to `f.semaphore(name)` within a single op must use a unique name.
Duplicate names raise a `RuntimeError`. Use the prefix convention:

```python
sender_sem = f.semaphore(f"{prefix}.sender")
receiver_sem = f.semaphore(f"{prefix}.receiver")
```

### Forgetting to Reset Semaphores

After `noc_semaphore_wait`, always `noc_semaphore_set` back to 0 (or
INVALID). Without the reset, the next iteration sees a stale value and
skips the wait, reading incomplete data.

### Source CB Initialization

When chaining ops (e.g., EmbedMcast = Embedding + Mcast), the downstream op
receives a CBHandle, not a tensor. The downstream op must **not** call
`setup_sharded_buffer` on this handle, because the upstream op already pushed
data into it. The `init_src` / `skip_src_init` flags control this:

```python
needs_init = not isinstance(src, CBHandle)
```

### Destination CB Balance

Mcast sets `balanced=False` on the destination scratch CB because not all
receiver cores may consume all pages. If the receiver grid does not match
the consumer grid, some cores push but never pop, which would violate
temporal reuse assumptions. The `balanced=False` flag opts the CB out of
temporal reuse.

### Missing CB Reserve/Push on Receiver

The receiver must call `cb_reserve_back` before the data arrives and
`cb_push_back` after the semaphore confirms arrival. Without these calls,
downstream TRISC compute ops will never see the data as available.

### Sender-Receiver RISC Mismatch

If both sender and receiver logic accidentally run on the same RISC, one
side's `cb_wait_front` and the other's `noc_semaphore_wait` may deadlock.
Always check that sender and receiver use different RISCs (BRISC vs NCRISC)
or that the op is designed for single-RISC operation (like Copy with its
`use_ncrisc` flag).

### data_size_bytes Exceeding NOC_MAX_BURST_SIZE

The Mcast kernel handles this automatically via template recursion in
`send_data_mcast`. Other ops using `noc_async_write` must either ensure
the transfer fits in one packet or split it manually. The
`noc_async_write_one_packet` variant used in Gather requires the data to
fit in one packet (assertion failure otherwise).

---

## 1.11 Summary

The sender/receiver pattern is the universal building block for inter-core
communication in TT-Blaze:

| Concept | Key API | Location |
|---------|---------|----------|
| Two NOC networks | `noc_index` template parameter, `DYNAMIC_NOC_X/Y` | Hardware / kernel API |
| Semaphore protocols | `SemProtocol.SENDER`, `.RECEIVER`, `.NOC0_RECEIVER`, `.NOC1_RECEIVER` | `blaze/blaze_op.py` |
| Logical-to-physical translation | `worker_core_from_logical_core()` | Device context / `FusedProgram` |
| Role flags | `f.flag(name, cores, enabled)` | `blaze/fused_program.py` |
| Per-core CT args | `f.per_core_unified_ct_args()` | `blaze/fused_program.py` |
| Data size computation | `num_pages * page_size` from `CBHandle` | All inter-core ops |
| Grid precomputation | `f.sender_grid`, `f.mcast_receiver_grid`, `f.noc_start`, `f.noc_end` | `blaze/fused_program.py` |
| Semaphore allocation | `f.semaphore(name)` or `f.semaphore(program_semaphore=True)` | `blaze/fused_program.py` |

The next section applies this pattern to the four concrete inter-core
micro-ops: Mcast, Gather, Scatter, and Copy.
