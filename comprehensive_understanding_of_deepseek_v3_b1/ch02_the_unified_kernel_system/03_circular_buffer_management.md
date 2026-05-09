# 2.3 Circular Buffer Management

Circular buffers (CBs) are the primary mechanism for passing data between RISC processors within a Tensix core and between successive operations in a fused pipeline. Each Tensix core supports up to 64 CB slots (IDs 0-63), and each slot is configured with a data format, tile shape, page size, and L1 address. In simple micro-ops, CBs are hand-numbered and statically allocated. In fused operations with dozens of CBs, this manual approach would be fragile and error-prone. DeepSeek V3 B1's `circular_buffer_utils.py` (224 lines) provides the infrastructure for allocating CB IDs across complex fused operations, constructing CB descriptors, and building the runtime reconfiguration tensor that allows a fused kernel to override the CB configuration left by a preceding operation.

**Source files:** `circular_buffer_utils.py`, `unified_kernels/kernel_utils.hpp` (lines 117-196)

---

## 2.3.1 CircularBufferIdManager: Format-Aware ID Allocation

The `CircularBufferIdManager` class (lines 15-70 of `circular_buffer_utils.py`) solves a combinatorial problem: a fused MoE operation uses dozens of CBs across multiple logical contexts (routed expert, shared expert, intermediate stages), but only 64 physical CB slots exist per core. The manager allows **cross-context reuse** of CB IDs when the `data_format` and `tile` descriptor match, while guaranteeing that within any single context, every CB ID is unique.

### Internal State

```python
class CircularBufferIdManager:
    NUM_CIRCULAR_BUFFERS = 64

    def __init__(self):
        self._id_to_format: dict[int, tuple] = {}  # cb_id -> (data_format, tile)
        self._next_id = 0
```

The `_id_to_format` dictionary tracks the `(data_format, TileDescriptor)` pair assigned to each allocated ID. The `_next_id` counter is a simple bump allocator for new IDs.

### The Allocation Algorithm

The `_allocate_id` method (lines 31-51) implements a two-phase strategy:

**Phase 1 -- Reuse.** Scan all existing IDs. If one has the same `(data_format, tile)` key and is not in the caller's `exclude` set, return it:

```python
for cb_id, fmt_key in self._id_to_format.items():
    if fmt_key == key and cb_id not in exclude:
        return cb_id
```

**Phase 2 -- Allocate.** If no reusable ID exists, allocate the next sequential ID:

```python
cb_id = self._next_id
if cb_id >= self.NUM_CIRCULAR_BUFFERS:
    raise RuntimeError(f"All {self.NUM_CIRCULAR_BUFFERS} circular buffer IDs are exhausted")
self._next_id += 1
self._id_to_format[cb_id] = (data_format, ttnn.TileDescriptor(tile))
return cb_id
```

The tile descriptor is copied (`ttnn.TileDescriptor(tile)`) to avoid external mutation affecting the manager's internal state. The type is enforced: a `TypeError` is raised if `tile` is not a `ttnn.TileDescriptor`.

### The Context Pattern

The `Context` inner class (lines 53-67) provides a scoped view:

```python
class Context:
    def __init__(self, manager: CircularBufferIdManager):
        self._manager = manager
        self._used_ids: set[int] = set()

    def get_cb_id(self, data_format, tile: ttnn.TileDescriptor) -> int:
        cb_id = self._manager._allocate_id(data_format, tile, self._used_ids)
        self._used_ids.add(cb_id)
        return cb_id
```

Each call to `get_cb_id` allocates (or reuses) a CB ID and marks it as used within this context. The context's `_used_ids` set grows monotonically -- there is no "release" operation.

### Why Format-Based Reuse Matters

Consider a fused MoE operation that needs 40+ CBs. Without reuse, a naive allocator would exhaust the 64 slots. With format-aware reuse, if the routed expert path already allocated a `bfloat16, 32x32` CB and the shared expert path needs one with the same format, the same ID is returned (as long as it is not in the current context's exclude set). In practice this keeps the MoE's total ID consumption under 64 despite the large number of logical buffers.

### Complexity Ladder: When Is the Manager Needed?

| Level | Example | CB Management | Manager Used? | IDs Allocated |
|-------|---------|--------------|---------------|---------------|
| 1 | Matmul | 3 hand-numbered CBs (0, 1, 2) | No | 3 |
| 2 | Mcast | 2 hand-numbered CBs (0, 1) | No | 2 |
| 3 | SharedExpertOp | 15 hand-numbered CBs (0-14) | No | 15 |
| 4 | MoeOp | 40+ CBs, routed + shared | **Yes** | ~40 (reuse keeps under 64) |

At Levels 1-3, the CB count is small enough that manual numbering is safe. At Level 4, the `CircularBufferIdManager` becomes essential.

---

## 2.3.2 CB Descriptor Construction: Three Patterns

Once CB IDs are allocated, they must be materialized as `CBDescriptor` objects that tell the runtime where in L1 the CB lives, how large it is, and what data format it uses. Three construction patterns cover the DeepSeek V3 B1 use cases.

### Pattern 1: cb_descriptor_from_sharded_tensor

The most common pattern for input/output tensors. This is a TT-Metal built-in function that creates a CB descriptor backed by an existing sharded tensor's L1 allocation:

```python
cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
    cb_index,
    device_tensor,
    address_offset=0,          # Optional byte offset into the shard
    total_size=tensor_size,    # Total CB size in bytes
    core_ranges=core_ranges,   # Which cores have this CB
)
```

The resulting descriptor inherits the tensor's L1 address, data format, and tile shape. The CB is "persistent" -- it points directly at the tensor's memory rather than owning separate storage. This pattern is used at every level of the complexity ladder.

### Pattern 2: cb_descriptor_from_overlapped_tensor

Defined in `circular_buffer_utils.py` (lines 99-129), this function handles the case where a CB must reference a sub-region of a larger "fused" tensor. An `OverlappedTensor` describes a view into a fused tensor with its own byte offset, total size, tile shape, and dtype:

```python
def cb_descriptor_from_overlapped_tensor(
    cb_index: int,
    overlapped: OverlappedTensor,
    fused_tensor_device: ttnn.Tensor,
) -> ttnn.CBDescriptor:
    # Start with the fused tensor's buffer/address plumbing
    cb_desc = ttnn.cb_descriptor_from_sharded_tensor(
        cb_index,
        fused_tensor_device,
        address_offset=overlapped.byte_offset,
        total_size=overlapped.total_size,
        core_ranges=overlapped.core_range_set,
    )
    # Override format to match the sub-tensor's properties
    tile = ttnn.Tile(overlapped.tile_shape)
    cb_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=cb_index,
            data_format=overlapped.dtype,
            page_size=tile.get_tile_size(overlapped.dtype),
            tile=ttnn.TileDescriptor(tile),
        )
    ]
    return cb_desc
```

This is used extensively in the MoE shared expert path, where gate and up-projection weight tensors are stored as sub-regions of a single fused L1 allocation to minimize memory fragmentation. The `address_offset` field positions the CB at the correct byte offset within the larger allocation.

### Pattern 3: Manual CBDescriptor

For intermediate CBs not backed by an existing tensor (e.g., temporary buffers between matmul and activation):

```python
fmt = ttnn.CBFormatDescriptor(
    buffer_index=cb_id,
    data_format=data_format,
    page_size=page_size,
    tile=tile_desc,
)
desc = ttnn.CBDescriptor()
desc.total_size = num_pages * page_size
desc.core_ranges = core_ranges
desc.format_descriptors = [fmt]
```

### The SharedExpertOp CB Layout (Level 3)

To illustrate how these patterns compose, consider the SharedExpertOp's 15 CBs:

```
CB 0:  A_gather_dst / reduce_g1  (sender, tensor-backed via dummy tensor)
CB 1:  B_gather_dst / reduce_g2  (sender, tensor-backed via dummy tensor)
CB 2:  intermediate              (sender, 2 tiles, manual)
CB 3:  mcast1 source             (sender, manual, TRISC-filled)
CB 4:  mcast1 dest / down_in0    (all 130 cores, manual)
CB 5:  down_weights              (112 matmul cores, tensor-backed)
CB 6:  down_matmul_out           (112 matmul cores, manual)
CB 7:  output_gather_dst         (sender, tensor-backed)
CB 8:  act_mcast_src             (sender, tensor-backed)
CB 9:  act_mcast_recv            (all 130 cores, manual)
CB 10: residual_mcast_src        (sender, tensor-backed)
CB 11: residual_mcast_dst        (all 130 cores, manual)
CB 12: residual_add_out          (112 matmul cores, manual)
CB 13: gate_up_weights           (128 compute cores, tensor-backed)
CB 14: gate_up_matmul_out        (128 compute cores, 1 tile, manual)
```

Key observations:
- **Tensor-backed CBs** (5, 7, 8, 10, 13) are created from weight/output tensors already sharded in L1.
- **Manual CBs** (2, 3, 4, 6, 9, 11, 12, 14) are scratch buffers allocated by the framework.
- **Dummy-tensor-backed CBs** (0, 1) use a dummy tensor purely to reserve L1 space.
- **Different core ranges**: CBs 4, 9, 11 span all 130 cores. CBs 5, 6, 12 are on 112 matmul cores. CBs 13, 14 are on 128 compute cores. CBs 0-3, 7, 8, 10 are on the single sender core.

Despite this complexity, `_build_cb_descriptors` returns a flat list of 15 `CBDescriptor` objects passed directly to `ProgramDescriptor`.

---

## 2.3.3 build_dummy_cb_descriptors: Placeholder CBs for Fused Ops

When a fused kernel uses CB reconfig at startup (see Section 2.3.5), the `ProgramDescriptor` still needs CB descriptors for all allocated IDs -- otherwise the runtime does not know which CB slots the kernel expects to use. The `build_dummy_cb_descriptors` method (lines 72-96) generates minimal placeholders:

```python
def build_dummy_cb_descriptors(self, core_ranges) -> list:
    descs = []
    for cb_id, (data_format, tile_desc) in self._id_to_format.items():
        tile = ttnn.Tile([tile_desc.height, tile_desc.width])
        page_size = tile.get_tile_size(data_format)

        fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_id,
            data_format=data_format,
            page_size=page_size,
            tile=tile_desc,
        )
        desc = ttnn.CBDescriptor()
        desc.total_size = page_size       # Minimal: just one page
        desc.core_ranges = core_ranges    # Full device grid
        desc.format_descriptors = [fmt]
        descs.append(desc)
    return descs
```

Key properties of dummy descriptors:
- **Correct format**: `data_format`, `tile`, and `page_size` match the real CB, because the runtime uses these to configure the data format registers that are **not overwritten** by the reconfig tensor. A format mismatch would cause silent data corruption.
- **Minimal sizing**: `total_size = page_size` (one page) rather than the real buffer size.
- **Full device grid**: `core_ranges` covers all cores, not just the cores that use the CB.
- **No real address**: The address is a placeholder; the real address comes from the reconfig tensor.

---

## 2.3.4 record_cb_metadata: Extracting Per-CB Configuration

The `record_cb_metadata` function (lines 132-152) bridges the gap between the real CB descriptors (which have correct addresses and sizes) and the reconfig tensor (which needs that information in a flat format):

```python
def record_cb_metadata(cb_descriptors):
    cb_metadata = {}
    for desc in cb_descriptors:
        for fmt in desc.format_descriptors:
            cb_id = fmt.buffer_index
            addr = ttnn.get_cb_address(desc)
            total_size = desc.total_size
            page_size = fmt.page_size
            num_pages = total_size // page_size
            cb_metadata[cb_id] = (addr, total_size, num_pages, page_size, desc.core_ranges)
    return cb_metadata
```

The returned dictionary maps `cb_id` to a 5-tuple:

| Field | Type | Meaning |
|-------|------|---------|
| `addr` | `uint32_t` | L1 byte address of the CB FIFO |
| `total_size` | `uint32_t` | Total FIFO size in bytes |
| `num_pages` | `uint32_t` | Number of pages (tiles) that fit in the FIFO |
| `page_size` | `uint32_t` | Size of one page in bytes |
| `core_ranges` | `CoreRangeSet` | Which cores have this CB configured |

The `core_ranges` field is critical: not every CB is present on every core. A CB for the sender core's RMSNorm output only has valid addresses on that one core.

---

## 2.3.5 build_cb_reconfig_tensor: The 264-Word L1 Layout

The `build_cb_reconfig_tensor` function (lines 155-224) is the culmination of the CB management system. It constructs an L1-sharded tensor that the fused kernel reads at startup to reconfigure its CB read/write interfaces.

### Why Reconfig Is Needed

When operations are fused into a single kernel dispatch, a preceding operation (e.g., SDPA) may have configured the CB hardware with its own addresses and sizes. The fused kernel needs to override these with its own CB configuration. Rather than issuing a separate program dispatch to reset the CBs (costing tens of microseconds in kernel launch overhead), the reconfig tensor encodes the entire CB configuration and the kernel applies it at startup.

### Memory Layout

The tensor has one shard per core, each shard containing exactly 264 uint32 words (1,056 bytes, 32-byte aligned):

```
Word Offset  Byte Offset  Contents
-----------  -----------  --------
  0 -   3       0 -  15   CB 0:  [addr_bytes, size_bytes, num_pages, page_size_bytes]
  4 -   7      16 -  31   CB 1:  [addr_bytes, size_bytes, num_pages, page_size_bytes]
  8 -  11      32 -  47   CB 2:  [addr_bytes, size_bytes, num_pages, page_size_bytes]
     ...          ...      ...
252 - 255    1008 - 1023   CB 63: [addr_bytes, size_bytes, num_pages, page_size_bytes]
-----------  -----------  --------
    256          1024      cb_mask_low:  bitmask of active CBs 0-31
    257          1028      cb_mask_high: bitmask of active CBs 32-63
-----------  -----------  --------
    258          1032      sync_semaphore[0]: cross-RISC enter barrier
    259          1036      sync_semaphore[1]: cross-RISC exit barrier
-----------  -----------  --------
260 - 263    1040 - 1055   reserved (zeros, for 32B alignment)
```

**Total: 264 uint32 words = 1,056 bytes per core.**

The bitmask words (256-257) tell the kernel which CB IDs need reconfiguration, avoiding unnecessary work for unused slots. The sync semaphores at words 258-259 are the two-phase barrier semaphores used by `reconfig_cb_interfaces` (Section 2.2.6). They are initialized to 0 and live inside the reconfig tensor itself, so no separate semaphore allocation is needed.

### Construction Code

```python
def build_cb_reconfig_tensor(cb_metadata, full_device_grid, mesh_device):
    all_cores = ttnn.corerange_to_cores(full_device_grid, row_wise=True)
    num_cores = len(all_cores)
    core_to_idx = {(c.x, c.y): idx for idx, c in enumerate(all_cores)}

    WORDS_PER_CORE = 264
    config = torch.zeros((num_cores, WORDS_PER_CORE), dtype=torch.uint32)

    for cb_id, (addr, total_size, num_pages, page_size, core_ranges) in cb_metadata.items():
        cb_cores = ttnn.corerange_to_cores(core_ranges, row_wise=True)
        for core in cb_cores:
            key = (core.x, core.y)
            if key not in core_to_idx:
                continue
            core_idx = core_to_idx[key]
            base = cb_id * 4
            config[core_idx, base + 0] = addr
            config[core_idx, base + 1] = total_size
            config[core_idx, base + 2] = num_pages
            config[core_idx, base + 3] = page_size
            if cb_id < 32:
                config[core_idx, 256] |= 1 << cb_id
            else:
                config[core_idx, 257] |= 1 << (cb_id - 32)
```

The per-core configuration is not uniform -- each core only gets entries for the CBs that have that core in their `core_ranges`. The tensor is placed in L1 as a `HEIGHT_SHARDED` tensor:

```python
shard_spec = ttnn.ShardSpec(full_device_grid, (1, WORDS_PER_CORE),
                            ttnn.ShardOrientation.ROW_MAJOR)
mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    shard_spec)
```

Multi-device support: if `mesh_device` has multiple devices, the tensor is replicated via `ReplicateTensorToMesh`.

---

## 2.3.6 The C++ Reconfig Machinery

### reconfig_cbs_for_mask()

The inner loop that reconfigures CB interfaces (lines 131-162 of `kernel_utils.hpp`):

```cpp
template <bool do_read, bool do_write, bool do_reset_stream_regs>
FORCE_INLINE void reconfig_cbs_for_mask(
    uint32_t tt_l1_ptr* cb_config, uint32_t mask, uint32_t start_cb)
{
    uint32_t cb = start_cb;
    while (mask) {
        if (mask & 1) {
            uint32_t base = cb * 4;
            uint32_t fifo_addr      = cb_config[base + 0] >> cb_addr_shift;
            uint32_t fifo_size      = cb_config[base + 1] >> cb_addr_shift;
            uint32_t fifo_num_pages = cb_config[base + 2];
            uint32_t fifo_page_size = cb_config[base + 3] >> cb_addr_shift;

            LocalCBInterface& iface = get_local_cb_interface(cb);
            if constexpr (do_read) {
                iface.fifo_rd_ptr = fifo_addr;
            }
            if constexpr (do_write) {
                iface.fifo_wr_ptr = fifo_addr;
                iface.fifo_num_pages = fifo_num_pages;
            }
            iface.fifo_size = fifo_size;
            iface.fifo_limit = fifo_addr + fifo_size;
            iface.fifo_page_size = fifo_page_size;
            iface.tiles_acked_received_init = 0;

            if constexpr (do_reset_stream_regs) {
                *get_cb_tiles_received_ptr(cb) = 0;
                *get_cb_tiles_acked_ptr(cb) = 0;
            }
        }
        mask >>= 1;
        cb++;
    }
}
```

**The `cb_addr_shift` conversion:** The config tensor stores addresses and sizes in *bytes*, but the right-shift `>> cb_addr_shift` applied during reconfig is **RISC-dependent**. The shift value is set at compile time in `circular_buffer_interface.h`:

```cpp
#if defined(COMPILE_FOR_TRISC)
constexpr uint32_t cb_addr_shift = CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT;  // == 4
#else
constexpr uint32_t cb_addr_shift = 0;
#endif
```

- **TRISC** (`cb_addr_shift = 4`): Addresses and sizes are converted from bytes to 16-byte alignment units (right-shift by 4). The `LocalCBInterface` fields on TRISC use these alignment units.
- **NCRISC and BRISC** (`cb_addr_shift = 0`): The right-shift is a no-op; addresses and sizes remain in bytes. The `LocalCBInterface` fields on these RISCs use raw byte values.

Because `reconfig_cbs_for_mask` is compiled separately per RISC (via the unified kernel's per-RISC compilation model from Section 2.2.1), the same template instantiation produces different shift behavior on each processor.

**The bitmask walk:** `mask & 1` tests the lowest bit. If set, that CB is active and its four config words are read. `mask >>= 1` advances to the next CB. This makes the reconfig cost proportional to the number of active CBs, not the total 64.

**Per-field semantics:**

| Field | What it controls |
|-------|-----------------|
| `fifo_rd_ptr` | Where the consumer (reader) starts reading. Reset to FIFO base. |
| `fifo_wr_ptr` | Where the producer (writer) starts writing. Reset to FIFO base. |
| `fifo_size` | Total FIFO capacity in alignment units. |
| `fifo_limit` | `fifo_addr + fifo_size`. The wrap-around boundary. |
| `fifo_page_size` | Size of one tile/page in alignment units. |
| `fifo_num_pages` | How many pages fit in the FIFO. |
| `tiles_acked_received_init` | Reset to 0 (no tiles produced or consumed). |
| Stream regs (`tiles_received`, `tiles_acked`) | Only reset on NCRISC. Hardware registers tracking tile flow. |

### reconfig_cb_interfaces()

The top-level reconfig function (lines 164-194):

```cpp
FORCE_INLINE void reconfig_cb_interfaces(uint32_t tt_l1_ptr* cb_config) {
    // Per-RISC flag selection
    #if defined(COMPILE_FOR_NCRISC)
        constexpr bool do_read = true, do_write = true, do_reset_stream_regs = true;
    #elif defined(COMPILE_FOR_BRISC)
        constexpr bool do_read = true, do_write = true, do_reset_stream_regs = false;
    #elif defined(UCK_CHLKC_UNPACK)
        constexpr bool do_read = true, do_write = false, do_reset_stream_regs = false;
    #elif defined(UCK_CHLKC_PACK)
        constexpr bool do_read = false, do_write = true, do_reset_stream_regs = false;
    #else
        constexpr bool do_read = false, do_write = false, do_reset_stream_regs = false;
    #endif

    volatile uint32_t tt_l1_ptr* reconfig_sem =
        reinterpret_cast<volatile uint32_t tt_l1_ptr*>(&cb_config[258]);
    sync_riscs_enter(reconfig_sem);

    reconfig_cbs_for_mask<do_read, do_write, do_reset_stream_regs>(
        cb_config, cb_config[256], 0);   // CBs 0-31
    reconfig_cbs_for_mask<do_read, do_write, do_reset_stream_regs>(
        cb_config, cb_config[257], 32);  // CBs 32-63

    sync_riscs_exit(reconfig_sem);
}
```

Per-RISC template parameter table:

| RISC | `do_read` | `do_write` | `do_reset_stream_regs` |
|------|-----------|-----------|----------------------|
| NCRISC | `true` | `true` | `true` |
| BRISC | `true` | `true` | `false` |
| TRISC0 (unpack) | `true` | `false` | `false` |
| TRISC2 (pack) | `false` | `true` | `false` |
| TRISC1 (math) | `false` | `false` | `false` |

NCRISC is the only RISC that resets stream registers because it owns the hardware counters that track tile flow between data-movement and compute pipelines. TRISC1 (the `#else` fallback) does not directly interact with CB state.

---

## 2.3.7 The Four-RISC Synchronization Barrier

CB reconfiguration is a coordinated operation. All four active RISCs (NCRISC, BRISC, TRISC0, TRISC2) must agree on the transition point. If NCRISC resets stream registers while TRISC2 is still packing tiles from the previous phase, data corruption occurs.

The barrier uses two semaphores embedded in the reconfig tensor at words 258 and 259:

### Phase 1: Enter (sync_riscs_enter)

```cpp
FORCE_INLINE void sync_riscs_enter(volatile uint32_t tt_l1_ptr* sem_addr) {
#if defined(COMPILE_FOR_BRISC) || defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
    __atomic_fetch_add(&sem_addr[0], 1, __ATOMIC_RELAXED);
#elif defined(COMPILE_FOR_NCRISC)
    while (__atomic_load_n(&sem_addr[0], __ATOMIC_RELAXED) < 3) { }
    sem_addr[0] = 0;
#endif
}
```

**BRISC, TRISC0, TRISC2:** Each atomically increments `sem[0]` by 1, signaling "I have completed all prior work."

**NCRISC:** Spins until `sem[0]` reaches 3 (all three others have signaled), then resets `sem[0]` to 0. NCRISC is the coordinator because it is the only RISC that resets stream registers.

### Phase 2: Exit (sync_riscs_exit)

```cpp
FORCE_INLINE void sync_riscs_exit(volatile uint32_t tt_l1_ptr* sem_addr) {
#if defined(COMPILE_FOR_NCRISC)
    __atomic_fetch_add(&sem_addr[1], 3, __ATOMIC_RELAXED);
#elif defined(COMPILE_FOR_BRISC) || defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_PACK)
    while (__atomic_load_n(&sem_addr[1], __ATOMIC_RELAXED) == 0) { }
    __atomic_fetch_sub(&sem_addr[1], 1, __ATOMIC_RELAXED);
#endif
}
```

**NCRISC:** After applying the reconfig, atomically adds 3 to `sem[1]`.

**BRISC, TRISC0, TRISC2:** Each spins until `sem[1]` is non-zero, then atomically decrements by 1. When the last RISC decrements, `sem[1]` returns to 0.

### Safety Across Iterations

The barrier is safe for repeated use within a loop because:

1. After `sync_riscs_exit`, all four RISCs proceed to execute the next phase's micro-ops.
2. The full phase body executes between `sync_riscs_exit` and the next iteration's `sync_riscs_enter`.
3. By the time the next `sync_riscs_enter` begins, `sem[1]` is guaranteed to be 0 (the last of BRISC/TRISC0/TRISC2 decremented it).
4. `sem[0]` was reset to 0 by NCRISC during the previous `sync_riscs_enter`.

The non-atomic write `sem_addr[0] = 0` in `sync_riscs_enter` is safe because at that point, BR/TR0/TR2 are blocked on `sem[1]` and cannot observe or modify `sem[0]`.

---

## 2.3.8 The Full MoeOp Reconfig Flow

The MoE fused operation is the most complex consumer of the CB reconfig system. When `reconfig_moe_cbs=True`, the following 7-step sequence occurs:

### Step 1: Setup Phase (Python, at op construction time)

```python
self.cb_id_manager = CircularBufferIdManager()
cb_id_context = self.cb_id_manager.create_context()

# Both routed and shared expert allocate CB IDs from the same context
routed_ctx = MoeRoutedExpertOp._setup_dimensions(..., cb_id_context=cb_id_context)
shared_ctx = MoeSharedExpertOp._setup_dimensions(..., cb_id_context=cb_id_context)
```

### Step 2: Build Real CB Descriptors

```python
cb_descriptors = []
cb_descriptors += MoeRoutedExpertOp._build_cb_descriptors(self.ctx.routed_ctx)
cb_descriptors += MoeSharedExpertOp._build_cb_descriptors(self.ctx.shared_ctx)
self.cb_metadata = record_cb_metadata(cb_descriptors)
```

### Step 3: Build the Reconfig Tensor

```python
self.reconfig_tensor = build_cb_reconfig_tensor(
    self.cb_metadata, self.ctx.full_device_grid, self.ctx.mesh_device
)
```

### Step 4: Build Dummy CB Descriptors

```python
self.dummy_cb_descs = self.cb_id_manager.build_dummy_cb_descriptors(self.ctx.full_device_grid)
```

### Step 5: Pass the Reconfig Tensor Address to the Kernel

```python
addr = self.reconfig_tensor.buffer_address()
self.ncrisc_args.append(("reconfig_cb_config_l1_addr", addr))
self.brisc_args.append(("reconfig_cb_config_l1_addr", addr))
self.trisc_args.append(("reconfig_cb_config_l1_addr", addr))
defines += [("RECONFIG_MOE_CBS", "1")]
```

The address becomes a named compile-time arg, and `RECONFIG_MOE_CBS` is set as a preprocessor define. The reconfig tensor is added to the IO tensors list for lifetime tracking.

### Step 6: ProgramDescriptor with Dummy CBs

```python
program = ttnn.ProgramDescriptor(
    kernels=kernel_result.kernels,
    cbs=self.dummy_cb_descs if ctx.reconfig_moe_cbs else self.device_cb_descs,
    semaphores=self.device_sem_descs,
)
```

The dummy descriptors ensure the firmware initializes CB interface structures with correct data formats and tile shapes, while the actual FIFO addresses come from the reconfig tensor.

### Step 7: Kernel Startup (C++, at runtime)

```cpp
#if defined(RECONFIG_MOE_CBS) && !defined(UCK_CHLKC_MATH)
{
    constexpr uint32_t cb_config_l1_addr =
        get_named_compile_time_arg_val("reconfig_cb_config_l1_addr");
    uint32_t tt_l1_ptr* cb_config =
        reinterpret_cast<uint32_t tt_l1_ptr*>(cb_config_l1_addr);
    unified_kernels::reconfig_cb_interfaces(cb_config);
}
#endif
```

The `UCK_CHLKC_MATH` exclusion ensures TRISC1 (the math thread) does not execute the reconfig, since it does not access CB interfaces.

### Why Is This Necessary?

When MoeOp is fused with a preceding operation (e.g., an MLA attention kernel), the preceding op leaves its own CB configuration in the firmware's `LocalCBInterface` structures. Without reconfig, the MoE kernel would read from and write to wrong memory addresses. The dummy-descriptor approach avoids a more expensive alternative: tearing down and recreating the program.

---

## 2.3.9 Worked Example: Tracing a Single CB Through the System

To make the system concrete, let us trace CB ID 0 (a typical input buffer) through the entire pipeline.

**Python side:**

1. `cb_id_context.get_cb_id(ttnn.bfloat16, tile_32x32)` returns `0`.
2. `ttnn.cb_descriptor_from_sharded_tensor(0, input_tensor)` creates a CBDescriptor with `addr=0x1A000`, `total_size=7168`, `page_size=1024` (a 32x32 bfloat16 tile).
3. `record_cb_metadata()` extracts: `{0: (0x1A000, 7168, 7, 1024, core_ranges)}`.
4. `build_cb_reconfig_tensor()` writes to word offsets 0-3 for each core in `core_ranges`:
   - `config[core_idx, 0] = 0x1A000` (addr bytes)
   - `config[core_idx, 1] = 7168` (total\_size bytes)
   - `config[core_idx, 2] = 7` (num\_pages)
   - `config[core_idx, 3] = 1024` (page\_size bytes)
   - `config[core_idx, 256] |= 0x1` (bit 0 set in cb\_mask\_low)

**C++ side (runtime):**

5. `reconfig_cbs_for_mask()` reads these four words and applies the RISC-dependent `cb_addr_shift`:

   **On TRISC** (`cb_addr_shift = 4`, converts to 16-byte alignment units):
   - `fifo_addr = 0x1A000 >> 4 = 0x1A00` (6,656 alignment units)
   - `fifo_size = 7168 >> 4 = 448`
   - `fifo_num_pages = 7`
   - `fifo_page_size = 1024 >> 4 = 64`

   **On NCRISC and BRISC** (`cb_addr_shift = 0`, addresses stay in bytes):
   - `fifo_addr = 0x1A000 >> 0 = 0x1A000` (106,496 bytes)
   - `fifo_size = 7168 >> 0 = 7168`
   - `fifo_num_pages = 7`
   - `fifo_page_size = 1024 >> 0 = 1024`

6. Sets `LocalCBInterface` for CB 0 (showing TRISC values; NCRISC/BRISC use byte values above):
   - `fifo_rd_ptr = 0x1A00`
   - `fifo_wr_ptr = 0x1A00`
   - `fifo_size = 448`
   - `fifo_limit = 0x1A00 + 448 = 0x1BC0`
   - `fifo_page_size = 64`
   - `fifo_num_pages = 7`
   - `tiles_acked_received_init = 0`

After reconfig, CB 0 is ready for use: `cb_reserve_back(0, 7)` will succeed immediately, and `cb_push_back(0, 7)` will make all 7 tiles available to the consumer.

---

## 2.3.10 The Complete Reconfig Pipeline

Putting it all together:

```
Python side (op setup):
  1. CircularBufferIdManager allocates CB IDs across all sub-ops
  2. Real CB descriptors are built (with tensor-backed addresses)
  3. record_cb_metadata() extracts (addr, size, num_pages, page_size) per CB
  4. build_cb_reconfig_tensor() encodes metadata into L1-sharded tensor
  5. build_dummy_cb_descriptors() creates minimal placeholders
  6. ProgramDescriptor gets dummy_cb_descs (format-correct, address-dummy)
       + reconfig_tensor as an IO tensor

C++ side (kernel startup):
  7. Runtime configures CB hardware from dummy descriptors (wrong addresses, right formats)
  8. Kernel reads reconfig tensor from L1 at compile-time-known address
  9. sync_riscs_enter: BRISC/TRISC0/TRISC2 signal ready, NCRISC waits
 10. reconfig_cbs_for_mask: Each RISC overwrites its CB interface fields
       with correct addresses, sizes, and page counts from the reconfig tensor
 11. NCRISC resets stream registers (tiles_received, tiles_acked) to 0
 12. sync_riscs_exit: NCRISC signals done, others proceed
 13. Kernel body executes with correctly configured CBs
```

This pipeline adds negligible overhead (approximately 100 cycles for the barrier and register writes in the common case) but eliminates the need for a separate program dispatch to reset CB state between operations, which would cost tens of microseconds in kernel launch overhead.

---

## 2.3.11 Design Constraints and Trade-offs

**64-slot limit**: With `NUM_CIRCULAR_BUFFERS = 64` and a single context potentially allocating 40+ CBs, the limit is a real concern for complex fused ops. The format-aware reuse in `CircularBufferIdManager` mitigates this, but operations with many distinct data formats or tile shapes will consume IDs faster.

**Per-core heterogeneity**: The reconfig tensor supports per-core CB configurations (different cores can have different CB addresses and sizes for the same CB ID). This is essential because sharded tensors place different data at different L1 addresses on each core.

**Synchronization cost**: The two-phase barrier requires all four participating RISCs to reach the barrier before any can proceed. In practice this is rare overhead because reconfig happens at kernel startup before any per-core work begins.

**Dummy descriptor format correctness**: The dummy descriptors must have the correct `data_format` and `tile` because the runtime uses these to configure format registers that are **not overwritten** by the reconfig tensor. The reconfig tensor only updates FIFO pointer state (addresses, sizes, page counts), not data format configuration. This is why `build_dummy_cb_descriptors` copies the exact format from the manager's allocation table.

**All configs known at op time**: Dynamic CB allocation at runtime is not supported. All CB configurations must be determined at Python `op()` time. The reconfig tensor is a one-time DMA to L1; the actual reconfig at runtime is entirely local to each core.

---

## 2.3.12 CB Management Across the Complexity Ladder

| Level | Example | CBs | ID Management | Reconfig? | Key Pattern |
|-------|---------|-----|--------------|-----------|-------------|
| 1 | Matmul | 3 | Hand-numbered | No | `cb_descriptor_from_sharded_tensor` |
| 2 | Mcast | 2 | Hand-numbered | No | Tensor-backed src + manual dst |
| 3 | SharedExpertOp | 15 | Hand-numbered | No | Mixed tensor-backed + manual + dummy-tensor |
| 4 | MoeOp | 40+ | `CircularBufferIdManager` | **Yes** | Manager + `record_cb_metadata` + `build_cb_reconfig_tensor` + dummy descs |

The progression is smooth: at each level, new utilities layer on top of existing ones without replacing them. The tensor-backed `cb_descriptor_from_sharded_tensor` pattern from Level 1 is still used at Level 4 -- it is just supplemented by the manager and reconfig machinery. This is the circular buffer counterpart to the fractal insight from Sections 2.1 and 2.2: the same `CBDescriptor` objects, the same `ProgramDescriptor` interface, the same `kernel_main()` entry point. What scales is the tooling that *generates* those descriptors, not the descriptors themselves.

---

**Next:** [Chapter 3 -- The Micro-Op Library](../ch03_the_micro_op_library/index.md)
