# 04 -- CB Reconfig: Multi-Phase Pipeline Reconfiguration

**Source files:** `blaze/cb_reconfig.py`, `blaze/cb_reconfig_builder.py`

Blackhole's Tensix cores have 64 circular buffer slots, and each slot's configuration (L1 address, size, page size, number of pages) is programmed at kernel launch via CB descriptors. For a fused kernel with many ops, 64 slots can run out -- or more commonly, different phases of the pipeline need the same slot to point at different L1 regions with different data formats. CB reconfiguration solves this by reprogramming CB hardware registers mid-kernel, without restarting the kernel or re-dispatching from the host. The reconfig tensor is the mechanism that delivers new CB configurations to each core's L1 at phase boundaries.

The system comprises two modules: `cb_reconfig.py` (ID management and tensor construction) and `cb_reconfig_builder.py` (the multi-phase build pass that transforms a FusedProgram with reconfig boundaries into a multi-phase executable).

## CircularBufferIdManager

```python
class CircularBufferIdManager:
    NUM_CIRCULAR_BUFFERS = 64

    def __init__(self):
        self._id_to_format: dict[int, tuple] = {}
        self._next_id = 0
```

The `NUM_CIRCULAR_BUFFERS = 64` constant reflects the hardware limit -- Tensix cores have exactly 64 CB slot register banks. The manager owns a global CB ID namespace and allocates IDs that can be reused across phases. Two CBs in different phases that share the same `(data_format, tile)` get the same CB ID -- the kernel accesses both through the same hardware register, and the reconfig tensor reprograms the register at the phase boundary.

### _normalize_tile(tile)

```python
@staticmethod
def _normalize_tile(tile: ttnn.TileDescriptor | None) -> ttnn.TileDescriptor | None:
    if tile is None:
        return None
    return ttnn.TileDescriptor(tile)
```

Ensures tiles are always stored as `TileDescriptor` instances, normalizing any compatible input. Returns `None` for non-tiled buffers.

### seed_reserved_cb(cb_id, data_format, tile)

```python
def seed_reserved_cb(self, cb_id, data_format, tile) -> None:
    if cb_id not in self._id_to_format:
        self._id_to_format[cb_id] = (data_format, self._normalize_tile(tile))
    self._next_id = max(self._next_id, cb_id + 1)
```

Registers a pre-assigned CB ID (e.g., from `CBEngine.assign` in the graph-API path, Ch3 S02). This is the bridge between the graph-API's engine-driven allocation and the composition-API's manager-driven allocation. The method:

1. Validates that `tile` is a `TileDescriptor` and `data_format` is a `DataType` (raises `TypeError` otherwise).
2. Stores `(data_format, normalized_tile)` in `_id_to_format[cb_id]` if not already present.
3. Advances `_next_id` past the seeded ID so future allocations do not collide.

### _allocate_id(data_format, tile, exclude)

```python
def _allocate_id(
    self,
    data_format: ttnn.DataType,
    tile: ttnn.TileDescriptor | None,
    exclude: set[int],
) -> int:
    key = (data_format, tile)

    for cb_id, fmt_key in self._id_to_format.items():
        if fmt_key == key and cb_id not in exclude:
            return cb_id

    cb_id = self._next_id
    if cb_id >= self.NUM_CIRCULAR_BUFFERS:
        raise RuntimeError(f"All {self.NUM_CIRCULAR_BUFFERS} circular buffer IDs are exhausted")
    self._next_id += 1
    self._id_to_format[cb_id] = (data_format, self._normalize_tile(tile))
    return cb_id
```

The core allocation logic implements format-based reuse:

1. **Reuse path**: Scans `_id_to_format` for an existing ID with matching `(data_format, tile)` that is not in the `exclude` set. Returns the first match. This is how cross-phase sharing works -- two phases with the same format get the same ID.
2. **Fresh allocation**: If no reusable ID exists, takes `_next_id`, validates it is less than 64, stores the format, increments `_next_id`, and returns the new ID.

The `exclude` set is provided by each `Context` and contains the IDs already used within that context. Within a single phase, every CB ID must be unique; across phases, matching-format IDs can be shared.

**Why format-based reuse instead of unconditional reuse?** The reconfig tensor format stores address, size, num_pages, and page_size per CB slot, but the CB descriptor also has data_format and tile metadata that the kernel reads at compile time. If two phases reused the same CB ID with different data formats, the kernel's compile-time format assumption would be wrong. By constraining reuse to matching formats, the system guarantees that the kernel's compile-time view of each CB slot remains consistent across phases.

### Context (inner class)

```python
class Context:
    def __init__(self, manager: "CircularBufferIdManager"):
        self._manager = manager
        self._used_ids: set[int] = set()

    def get_cb_id(self, data_format: ttnn.DataType, tile: ttnn.TileDescriptor | None) -> int:
        cb_id = self._manager._allocate_id(data_format, tile, self._used_ids)
        self._used_ids.add(cb_id)
        return cb_id
```

A scoped view of the manager. Each context tracks which IDs it has used (`_used_ids`) and passes that set as the `exclude` argument to `_allocate_id`. This enforces uniqueness within a phase while enabling sharing across phases. Created by `create_context()`.

The invariant: within a Context, every CB ID is unique. Across Contexts backed by the same manager, CB IDs reuse whenever format matches. This is the fundamental mechanism that allows programs with more than 64 logical CBs to fit within 64 hardware slots.

### build_dummy_cb_descriptors(core_ranges)

```python
def build_dummy_cb_descriptors(self, core_ranges) -> list:
    descs = []
    for cb_id, (data_format, tile_desc) in self._id_to_format.items():
        page_size = 1
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_id,
            data_format=data_format,
            page_size=page_size,
            tile=tile_desc,
        )
        desc = ttnn.CBDescriptor()
        desc.total_size = page_size
        desc.core_ranges = core_ranges
        desc.format_descriptors = [fmt]
        descs.append(desc)
    return descs
```

Builds stub `CBDescriptor` objects for program descriptor (PD) validation. The real CB configuration comes from the reconfig tensor at runtime, but tt-metal's validation pass requires at least one descriptor per CB ID to exist. Each stub has `total_size = 1` and `page_size = 1` -- the minimum valid values to pass validation without reserving real L1 space.

## CBContext

```python
class CBContext:
    def __init__(self, name: str, manager: CircularBufferIdManager):
        self.name = name
        self._manager = manager
        self._ctx = manager.create_context()
        self.used_ids: set[int] = self._ctx._used_ids
        self.descriptors: list = []
```

A named phase handle that wraps a `CircularBufferIdManager.Context`. This is the Tier 3 (manual phase control) API -- callers that bypass the automatic `_build_multi_phase` pass can use `CBContext` directly.

### get_cb_id(data_format, tile)

Delegates to the inner context. Returns a CB ID unique within this phase.

### pin(cb_id, data_format, tile)

```python
def pin(self, cb_id, data_format, tile):
    self.used_ids.add(cb_id)
    td = self._manager._normalize_tile(tile)
    self._manager._id_to_format[cb_id] = (data_format, td)
    self._manager._next_id = max(self._manager._next_id, cb_id + 1)
```

Reserves a specific CB ID with the given format, bypassing the `_allocate_id` reuse logic. Used when the caller has a hard constraint on which ID to use -- for example, when a kernel hardcodes `cb_wait_front(7, ...)` rather than reading the CB ID from a compile-time argument.

### build_reconfig_tensor(full_device_grid, device)

```python
def build_reconfig_tensor(self, full_device_grid, device):
    cb_metadata = record_cb_metadata(self.descriptors)
    return build_cb_reconfig_tensor(cb_metadata, full_device_grid, device)
```

Extracts metadata from the context's accumulated `descriptors` and builds a reconfig tensor.

## FusedProgram Integration

The FusedProgram class exposes the reconfig machinery through three integration points:

### cb_manager property

```python
@property
def cb_manager(self) -> CircularBufferIdManager:
    if self._cb_manager_instance is None:
        from .cb_reconfig import CircularBufferIdManager
        self._cb_manager_instance = CircularBufferIdManager()
    return self._cb_manager_instance
```

Lazy-initialized. Shared across all phases of a FusedProgram -- this ensures cross-phase ID reuse works through a single `_id_to_format` dictionary.

### reconfig()

```python
def reconfig(self):
    self._reconfig_boundaries.append(self._op_index)
    self._format_key_to_cb.clear()
    self._view_tensor_cbs.clear()
```

Marks a phase boundary. The clearing of `_format_key_to_cb` and `_view_tensor_cbs` is essential: without it, Phase 1 allocations would accidentally reuse Phase 0 CB IDs through the sharing caches, and the multi-phase builder would see incorrect phase assignments.

### cb_context(name)

```python
def cb_context(self, name: str = "") -> CBContext:
    self.reconfig()
    ctx = CBContext(name=name, manager=self.cb_manager)
    self._cb_contexts.append(ctx)
    return ctx
```

Creates a named phase and returns a `CBContext` for manual control. Calls `reconfig()` internally to mark the boundary.

## record_cb_metadata()

```python
def record_cb_metadata(cb_descriptors, remap: dict[int, int] | None = None):
    cb_metadata = {}
    for desc in cb_descriptors:
        for fmt in desc.format_descriptors:
            cb_id = fmt.buffer_index
            if remap:
                cb_id = remap.get(cb_id, cb_id)
            addr = ttnn.get_cb_address(desc)
            if addr == 0:
                raise RuntimeError(f"CB {cb_id} has address 0, which means it's not backed by a tensor")
            total_size = desc.total_size
            page_size = fmt.page_size
            num_pages = total_size // page_size
            cb_metadata.setdefault(cb_id, []).append(
                (addr, total_size, num_pages, page_size, desc.core_ranges)
            )
    return cb_metadata
```

Extracts per-CB config from descriptors into `{cb_id: [(addr, total_size, num_pages, page_size, core_ranges)]}`.

A single CB ID can map to multiple entries when different core ranges have different L1 addresses -- this happens with disjoint-cores sharing (e.g., Q-rope and K-rope sharing an ID across disjoint core sets). The `remap` parameter applies CB ID remapping on-the-fly (used by `_build_multi_phase` to remap before building reconfig tensors), avoiding a separate pass over the descriptors.

The `addr == 0` check catches a critical bug: attempting to build a reconfig tensor for scratch CBs that have not been materialized to L1 yet. Address zero on Tensix L1 is not a valid CB base address. If the kernel reads from L1[0], it picks up whatever happens to be there (likely configuration registers or random data), causing silent corruption or a hardware hang.

## build_cb_reconfig_tensor()

```python
def build_cb_reconfig_tensor(cb_metadata, full_device_grid, mesh_device):
    all_cores = ttnn.corerange_to_cores(full_device_grid, row_wise=True)
    num_cores = len(all_cores)
    core_to_idx = {(c.x, c.y): idx for idx, c in enumerate(all_cores)}

    WORDS_PER_CORE = 264
    config = torch.zeros((num_cores, WORDS_PER_CORE), dtype=torch.uint32)
```

Builds a `HEIGHT_SHARDED` L1 tensor with per-core CB configuration for runtime reconfiguration. This is the data structure that the kernel's reconfig routine reads at each phase boundary.

### Layout: 264 uint32 per core

The 264-word layout maps directly to what the reconfig kernel reads:

| Word range | Count | Content |
|---|---|---|
| 0..255 | 256 words | 64 CBs x 4 words each: `[addr, size, num_pages, page_size]` |
| 256 | 1 word | Active-CB bitmask, CBs 0..31 (bit N set means CB N is active) |
| 257 | 1 word | Active-CB bitmask, CBs 32..63 |
| 258..259 | 2 words | Sync semaphores (reserved for phase synchronization) |
| 260..263 | 4 words | Reserved |

The four words per CB (`addr`, `total_size`, `num_pages`, `page_size`) are the exact values that the reconfig kernel writes into the Tensix CB configuration registers.

**Why bitmask instead of count?** The active-CB bitmask allows the kernel to skip unconfigured CB slots in O(popcount) rather than iterating all 64 slots. On a core that uses only 8 CBs, the kernel checks 8 bits rather than 64 entries. The split at CB 32 aligns with the uint32 word boundary.

### Construction

For each CB in `cb_metadata`, the function iterates over that CB's core ranges, looks up each core's index in `core_to_idx`, and writes the 4-word config:

```python
for cb_id, entries in cb_metadata.items():
    for addr, total_size, num_pages, page_size, core_ranges in entries:
        cb_cores = _cores_cache.get(id(core_ranges))
        if cb_cores is None:
            cb_cores = ttnn.corerange_to_cores(core_ranges, row_wise=True)
            _cores_cache[id(core_ranges)] = cb_cores
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

A `_cores_cache` keyed by `id(core_ranges)` avoids recomputing `corerange_to_cores` for the same core range set. Cores not in `core_to_idx` (e.g., fabric/ethernet cores not in the grid) are silently skipped.

### HEIGHT_SHARDED placement

```python
shard_spec = ttnn.ShardSpec(
    full_device_grid, (1, WORDS_PER_CORE), ttnn.ShardOrientation.ROW_MAJOR
)
mem_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
)
```

The tensor covers every core on the device (`full_device_grid`), not just the program's compute grid, because the reconfig applies to all cores. Each core's shard is exactly one row of 264 uint32 values placed in its own L1 -- the reconfig kernel reads from local L1, not from remote cores via NOC. For multi-device meshes, the tensor is replicated to all devices via `ttnn.ReplicateTensorToMesh`.

## cb_reconfig_builder.py -- The Build Pass

The builder transforms a FusedProgram with reconfig boundaries into a multi-phase program. It is invoked from `FusedProgram._prepare_for_build()` via `prepare_for_build(fp)`.

### prepare_for_build(fp) -- entry point

```python
def prepare_for_build(fp: FusedProgram) -> None:
    if fp._compaction_applied:
        return
    fp._compaction_applied = True
    if fp._compact_scratch_cbs:
        _compact_scratch_cbs(fp)
    if fp._reconfig_boundaries and _has_multi_phase_cbs(fp):
        fp._build_multi_phase()
    elif fp._reconfig_placeholders:
        # Trailing boundary: strip placeholder nodes to prevent
        # the kernel from reading unpatched cb_config_l1_addr=0.
        placeholder_nodes = {id(node) for _, node, _ in fp._reconfig_placeholders}
        fp._shadow_graph.nodes = [
            n for n in fp._shadow_graph.nodes if id(n) not in placeholder_nodes
        ]
        fp._reconfig_placeholders.clear()
```

Idempotent via the `_compaction_applied` guard. Three paths:

1. **Scratch compaction**: Always runs first (if `compact_scratch_cbs=True`). Merges non-conflicting scratch CBs onto shared slots.
2. **Multi-phase reconfig**: If reconfig boundaries exist and at least one CB is in phase > 0, runs `_build_multi_phase()`.
3. **Trailing boundary cleanup**: If reconfig boundaries exist but no CBs crossed them (all phase-0), strips the placeholder graph nodes to prevent reading from L1[0].

### Synthetic keys

```python
_RECONFIG_TENSOR_KEY_BASE = -1000   # per phase: -1000, -1001, ...
_ARENA_TENSOR_KEY = -2000
```

Internal tensors (reconfig tensors and the scratch arena) are stored in `program._io_tensors` with negative key IDs so they cannot collide with real CB IDs.

## Scratch CB Compaction: _compact_scratch_cbs

Merges non-conflicting scratch CBs onto shared CB-ID slots to reduce CB ID pressure. Operates per-phase and uses two reuse strategies.

### Spatial reuse

Two scratch CBs with matching `(data_format, tile_shape, page_size, num_pages)` on disjoint core ranges share one CB ID. Both descriptors survive -- they cover different cores. No lifetime constraint required because different cores never see both CBs.

### Temporal reuse

Two scratch CBs with matching format on the SAME core ranges share one CB ID if their lifetimes (`_cb_lifetimes`) do not overlap. Requirements:

- Both CBs are balanced (push/pop symmetry per core) -- an unbalanced CB's FIFO state would leak onto cores that pushed but never popped, corrupting the reused CB.
- Neither is tensor-backed -- distinct `address_offset` descriptors would collide after dedup.
- Neither the slot owner nor any folded user is tensor-backed.
- The `BLAZE_DISABLE_TEMPORAL_REUSE=1` env var is not set.
- Their cores must be exactly equal (`_cores_equal`) -- partial overlap is rejected even with disjoint lifetimes because dedup would drop one descriptor, leaving cores without a valid CB binding and causing silent L1 corruption.

When temporal reuse is applied, `_insert_scratch_reset(fp, op_idx, slot_cb_id, core_ranges)` inserts a `CbScratchReset` node at the point where the reused CB starts its second life, ensuring the CB's FIFO pointers are reset before the new user begins pushing.

### _Slot and _slot_accepts

The `_Slot` NamedTuple tracks each sharing slot:

```python
class _Slot(NamedTuple):
    cb_id: int
    users: list[tuple]       # (core_ranges, lifetime) per user
    user_cb_ids: list[int]   # original cb_ids folded onto this slot
```

`_slot_accepts(slot, lifetime, core_ranges, *, allow_temporal)` returns `"spatial"`, `"temporal"`, or `None`:

```python
def _slot_accepts(slot, lifetime, core_ranges, *, allow_temporal):
    decision = "spatial"
    for u_cr, u_lt in slot.users:
        if _are_disjoint(u_cr, core_ranges):
            continue
        if not allow_temporal:
            return None
        if _lifetimes_overlap(u_lt, lifetime):
            return None
        if not _cores_equal(u_cr, core_ranges):
            return None
        decision = "temporal"
    return decision
```

The format key for compaction is a 4-tuple: `(data_format, tile_shape, page_size, num_pages)`. The `num_pages` element is included because differing page counts break wrap-around arithmetic on temporal reuse.

### Eligibility filter

CBs that already span multiple descriptors (from disjoint-grid sharing) are excluded from compaction:

```python
desc_counts: dict[int, int] = {}
for desc in fp.program._cb_descriptors:
    for fmt in desc.format_descriptors:
        desc_counts[fmt.buffer_index] = desc_counts.get(fmt.buffer_index, 0) + 1
```

Only scratch CBs with `desc_counts == 1`, a recorded lifetime, and recorded metadata are eligible. This prevents re-compacting CBs whose address_offsets were already set for disjoint-core coexistence.

### Post-compaction cleanup

After compaction:

1. `_remap_cb_ids(fp, remap)` rewrites all CB IDs across the program.
2. `_dedup_cb_descriptors` removes duplicate descriptors for the same CB ID on overlapping cores (temporal reuse creates these).
3. Lifetime ranges are extended: the slot owner's lifetime is expanded to cover all folded-in users.
4. `CbScratchReset` nodes are inserted in descending op_idx order so earlier insertions do not shift the positions of later insertions.

## Multi-Phase Build: _build_multi_phase

The six-step pipeline transforms a FusedProgram with reconfig boundaries into one with per-phase reconfig tensors and remapped CB IDs.

### Step 1: Partition CBs into phases

Groups CB IDs by `_cb_phase[cb_id]` (set during allocation -- the phase index equals the number of `reconfig()` calls that have occurred). Validates that no scratch CB crosses a reconfig boundary:

```python
def _validate_no_scratch_crossing(fp: FusedProgram) -> None:
    for cb_id in fp._cb_lifetimes:
        if cb_id in fp._tensor_backed_cbs:
            continue
        first, last = fp._cb_lifetimes[cb_id]
        for boundary_op_idx in fp._reconfig_boundaries:
            if first < boundary_op_idx <= last:
                raise ValueError(
                    f"CB '{name}' is scratch-allocated but referenced across "
                    f"reconfig boundary at op {boundary_op_idx}. "
                    f"Use cb_from_tensor() for cross-phase data."
                )
```

**Why scratch CBs cannot cross boundaries:** A scratch CB's L1 address comes from the arena, which is reprogrammed at each boundary. If a scratch CB is live across a boundary, its arena offset may be reassigned to a different scratch CB in the next phase. The data would be silently overwritten. Tensor-backed CBs can cross boundaries because their L1 address is fixed (they point at a real tensor, not the arena).

### Step 2: Materialize scratch CBs to L1

`_materialize_scratch_cbs(fp)` packs all scratch CBs across all phases into a single shared L1 arena tensor.

```python
L1_ALIGN = 16  # Wormhole/Blackhole L1 alignment

# 1. Collect scratch CBs per phase
# 2. Assign running offsets within each phase
arena_offsets: dict[int, int] = {}
arena_size = 0
for cb_ids in phase_scratch_cbs.values():
    offset = 0
    for cb_id in sorted(cb_ids):  # sorted for determinism
        meta = fp._cb_metadata[cb_id]
        total_bytes = _cb_total_bytes(meta)
        aligned_offset = round_up(offset, L1_ALIGN)
        arena_offsets[cb_id] = aligned_offset
        offset = aligned_offset + total_bytes
    phase_total = round_up(offset, L1_ALIGN)
    arena_size = max(arena_size, phase_total)
```

**Why the arena is sized to max(phase_totals):** Cross-phase reuse is automatic -- the reconfig tensor reprograms CB addresses at each boundary. Phase 0's scratch region can be repurposed for Phase 1's scratch CBs because the reconfig kernel writes new addresses. The arena only needs to be large enough for the largest single phase.

Offsets are 16-byte aligned (`L1_ALIGN = 16`), matching Blackhole's L1 alignment requirement. Scratch CBs within each phase are sorted by `cb_id` before offset assignment, ensuring deterministic layout regardless of emit order.

Each scratch CB's descriptor is replaced with one pointing at the arena at its assigned offset. The arena's dtype is `uint32`; each CB descriptor overrides its own `data_format`, `page_size`, and `tile`:

```python
addr_desc = ttnn.cb_descriptor_from_sharded_tensor(
    cb_id, arena_tensor,
    address_offset=byte_offset,
    total_size=total_bytes,
    core_ranges=meta.core_ranges,
)
addr_desc.format_descriptors = [
    ttnn.CBFormatDescriptor(
        buffer_index=cb_id,
        data_format=meta.data_format,
        page_size=meta.page_size,
        tile=meta.tile_desc,
    )
]
```

The arena tensor is deduped across per-device compile iterations via `_internal_tensor_dict` keyed by `("arena", num_cores, shard_uint32)`. The arena is kept alive via `_cleanup_tensors` and stored in `_io_tensors` with key `_ARENA_TENSOR_KEY = -2000`.

### Step 3: Assign CB IDs across phases

Each phase gets a fresh `Context` from the shared `CircularBufferIdManager`:

```python
manager = fp.cb_manager
phase_remaps: list[dict[int, int]] = []

for phase_idx in range(num_phases):
    ctx = manager.create_context()
    remap: dict[int, int] = {}
    for cb_id in sorted(phase_cb_ids[phase_idx]):
        meta = fp._cb_metadata.get(cb_id)
        if meta is None:
            continue
        new_id = ctx.get_cb_id(meta.data_format, meta.tile_desc)
        if new_id != cb_id:
            remap[cb_id] = new_id
    phase_remaps.append(remap)
```

Matching `(data_format, tile)` pairs reuse the same CB ID across phases. This minimizes the total number of distinct CB IDs, staying within the 64-ID hardware limit.

### Step 4: Build per-phase reconfig tensors

Descriptors are partitioned by phase, metadata is extracted (with the phase-specific remap applied on-the-fly), and reconfig tensors are constructed:

```python
phase_descriptors = _partition_descriptors_by_phase(
    fp.program._cb_descriptors, fp._cb_phase, num_phases,
)

reconfig_tensors = []
for phase_idx in range(num_phases):
    descs = phase_descriptors[phase_idx]
    phase_remap = phase_remaps[phase_idx] or None
    cb_metadata = record_cb_metadata(descs, remap=phase_remap)
    sig = _reconfig_content_signature(cb_metadata)
    cache_key = ("reconfig", phase_idx, sig)
    tensor = fp._internal_tensor_dict.get(cache_key)
    if tensor is None:
        tensor = build_cb_reconfig_tensor(
            cb_metadata, fp.full_device_grid, fp.device,
        )
        fp._internal_tensor_dict[cache_key] = tensor
    reconfig_tensors.append(tensor)
    fp.program._io_tensors.append((_RECONFIG_TENSOR_KEY_BASE - phase_idx, tensor))
```

**Empty phase validation:** Phases 1..N must each have at least one CB. An empty phase means no ops were emitted between two consecutive reconfig boundaries -- the kernel would receive `cb_config_l1_addr=0` and read from L1 address 0 (arbitrary content), causing an indefinite hardware hang. This raises `ValueError`.

**Content signature deduplication:** `_reconfig_content_signature(cb_metadata)` produces a hashable tuple of all CB metadata (sorted CB IDs, each with `(addr, total_size, num_pages, page_size, core_ranges_signature)`). Two phases with identical CB configurations share the same tensor, avoiding redundant L1 allocations across per-device compile iterations.

After building reconfig tensors, all phase remaps are merged and applied:

```python
merged_remap: dict[int, int] = {}
for remap in phase_remaps:
    merged_remap.update(remap)
fallback_hit_ids = _remap_cb_ids(fp, merged_remap)
_warn_untracked_remaps(fp, fallback_hit_ids)
```

### Step 5: Patch placeholder addresses, insert Phase 0

During composition, `CbReconfig.emit()` stamps `cb_config_l1_addr=0` as a placeholder because the reconfig tensor does not exist yet. `_patch_placeholder_addresses` rewrites each placeholder with the real L1 address:

```python
def _patch_placeholder_addresses(fp, reconfig_tensors):
    for phase_idx, node, risc_positions in fp._reconfig_placeholders:
        tensor = reconfig_tensors[phase_idx]
        if tensor is None:
            continue
        addr = tensor.buffer_address()
        for risc, idx in risc_positions.items():
            args = fp._get_risc_args(risc)
            name, _ = args[idx]
            args[idx] = (name, addr)
        for key in list(node.ct_values):
            if key.endswith(".cb_config_l1_addr"):
                node.ct_values[key] = addr
```

Both the per-RISC arg lists and the shadow graph node's `ct_values` are patched.

Phase 0's reconfig node is inserted at position 0 in the shadow graph (the user never explicitly emits a Phase 0 boundary):

```python
def _insert_phase0_reconfig(fp, reconfig_tensor):
    prefix = "cb_reconfig_0"
    addr = reconfig_tensor.buffer_address()
    fp.program.unified_ct_args(
        [(f"{prefix}.cb_config_l1_addr", addr)], prefix=False
    )
    spec = get_op_spec("cb_reconfig")
    count = fp._op_counters.get("cb_reconfig", 0) + 1
    fp._op_counters["cb_reconfig"] = count
    node = OpNode(
        id=f"cb_reconfig_{count}",
        spec=spec,
        grid=fp.all_cores,
        kwargs={"ct_prefix": prefix},
        ct_values={f"{prefix}.cb_config_l1_addr": addr},
    )
    fp._shadow_graph.nodes.insert(0, node)
```

### Step 6: Deduplicate PD descriptors

After remapping, multiple descriptors may reference the same CB ID on overlapping cores (from temporal compaction or cross-phase remap). tt-metal rejects duplicate CB IDs on overlapping cores at the descriptor level. `_dedup_cb_descriptors` resolves this:

```python
def _dedup_cb_descriptors(descriptors):
    by_cb = _group_cb_descriptors_by_index(descriptors)
    losers_at, partials_at = _dedup_descriptor_claims(by_cb)
    split_descs = _split_partial_descriptors(descriptors, partials_at)
    result = _filter_deduped_descriptors(descriptors, losers_at)
    result.extend(split_descs)
    return result
```

The dedup operates per CB ID:

1. **Group** descriptors by `buffer_index`.
2. **Sort** each group by `(total_size, num_cores)` descending.
3. **Iterate**: the largest descriptor claims its cores first. Fully covered smaller descriptors become losers (pruned). Partially covered descriptors are split -- the overlapping portion is pruned, and a residual descriptor is created for the uncovered cores via `_clone_desc_with_core_ranges()`.

Aliased descriptors (multiple `format_descriptors` at different `buffer_index` values) are handled correctly: a larger descriptor may claim some buffer indexes while the alias remains the winner at others. The format descriptor at each claimed index is pruned; the descriptor is dropped only when all its format descriptors are pruned.

Disjoint-core descriptors at the same CB ID both survive -- this is the expected outcome of spatial reuse after compaction.

## _remap_cb_ids: The Five-Target Rewrite

```python
def _remap_cb_ids(fp: FusedProgram, remap: dict[int, int]) -> set[int]:
```

CB ID remapping touches every data structure that stores CB IDs. Returns the set of old CB IDs that hit the spec-based fallback (path 1b).

### 1a. Primary path: _cb_arg_positions

Rewrites args tracked via CBHandle in `_cb_arg_positions`:

```python
already_remapped: set[tuple[str, int]] = set()
for risc, idx, old_id in fp._cb_arg_positions:
    if old_id in remap:
        args = fp._get_risc_args(risc)
        name, _ = args[idx]
        args[idx] = (name, remap[old_id])
        already_remapped.add((risc, idx))
```

### 1b. Fallback path: C++ spec field-name matching

For int-valued args not tracked via CBHandle, scans all CT args by name against `_cb_field_names()` (parsed from registered C++ op headers):

```python
cb_fields = _cb_field_names()
for risc in ("ncrisc", "brisc", "trisc"):
    args = fp._get_risc_args(risc)
    for idx, (name, val) in enumerate(args):
        if (risc, idx) in already_remapped:
            continue
        if not isinstance(val, int) or val not in remap:
            continue
        field = name.rsplit(".", 1)[-1] if "." in name else name
        if field in cb_fields:
            args[idx] = (name, remap[val])
            fallback_hit_ids.add(val)
```

This catches cases where an op passed `int(handle)` instead of the CBHandle object to its `*_ct_args()` call, bypassing the tracked-position mechanism. `_warn_untracked_remaps` logs which CB IDs hit this path, signaling that those ops should be fixed.

### 2. CB descriptors

Rewrites `fmt.buffer_index` on every `CBDescriptor`.

### 3. io_tensor keys

Rewrites the CB-ID key in `program._io_tensors`.

### 4. Internal metadata

Remaps `_cb_metadata` keys. For N:1 remaps (compaction), keeps the handle with the largest `num_pages` so downstream L1 sizing covers the largest consumer of the shared slot.

### 5. Membership sets

Propagates remap into `_tensor_backed_cbs`, `_direct_address_cbs`, `_scratch_cbs`, `_unbalanced_cbs`.

## _partition_descriptors_by_phase

```python
def _partition_descriptors_by_phase(descriptors, cb_phase, num_phases) -> list[list]:
```

An aliased descriptor hosts multiple `format_descriptors` with different `cb_ids`, and those IDs can live in different phases. Each phase's bucket gets the descriptor so its own CB ID gets programmed at the boundary. Dedup by `(phase, id(desc))` prevents double-counting when two format_descriptors on the same descriptor belong to the same phase.

## _reconfig_content_signature

```python
def _reconfig_content_signature(cb_metadata) -> tuple:
```

Produces a hashable signature from `record_cb_metadata`'s output. Two calls produce equal signatures if and only if they would write byte-identical bytes to the reconfig tensor. Uses sorted CB IDs and `_core_ranges_signature` (order-independent sorted tuple of `(start_x, start_y, end_x, end_y)` per range). This enables safe deduplication across per-device compile iterations in a single `compile()` call.

## Environment Variable Escape Hatches

| Variable | Default | Effect |
|---|---|---|
| `BLAZE_DISABLE_TENSOR_CB_SHARE` | unset | Disables format-key (cross-tensor) sharing on tensor-backed allocator surfaces. Same-tensor view sharing in `cb_from_view` is unaffected. |
| `BLAZE_DISABLE_TEMPORAL_REUSE` | unset | Disables temporal scratch reuse in compaction. Spatial reuse still applies. |

Both are binary (`=1` to enable). They exist as bisection escape hatches for debugging CB-related issues in production models.

## Worked Example: MoE Expert Format Switch

This traces a Mixture-of-Experts layer end-to-end, where Phase 0 runs the router (gate) in bfloat16, and Phase 1 runs expert FFNs in bfp8_b. The scratch CBs in each phase need different data formats, requiring different CB configurations.

### Step 1: Composition with Reconfig Boundary

```python
f = FusedProgram(kernel=None, device=device, name="moe_layer")

# Phase 0: Gate (bfloat16 activations)
gate_act  = f.cb_from_tensor(act_tensor)                    # cb_id=0
gate_scratch = f.cb_scratch(
    name="gate___scratch", num_pages=2,
    core_ranges=f.all_cores,
    data_format=ttnn.bfloat16, tile=td, page_size=2048,
)                                                            # cb_id=1
f.unified_ct_args([("gate.in_cb", gate_act), ("gate.scratch_cb", gate_scratch)])
gate_out = f.output("moe_gate", gate_scratch, input=gate_act)
# _op_index = 1, _cb_phase = {0: 0, 1: 0}

# Reconfig boundary
CbReconfig.emit(f, prefix="reconfig_1")
# This calls f.reconfig(), which:
#   1. Appends _op_index to _reconfig_boundaries
#   2. Clears _format_key_to_cb and _view_tensor_cbs
#   3. Emits a placeholder CbReconfig node with cb_config_l1_addr=0

# Phase 1: Expert FFN (bfp8_b weights)
expert_weight = f.cb_from_tensor(weight_tensor)              # cb_id=2
expert_scratch = f.cb_scratch(
    name="expert___scratch", num_pages=4,
    core_ranges=f.matmul_cores,
    data_format=ttnn.bfp8_b, tile=td, page_size=1088,
)                                                            # cb_id=3
f.unified_ct_args([("expert.weight_cb", expert_weight), ("expert.scratch_cb", expert_scratch)])
expert_out = f.output("matmul", expert_scratch, weight=expert_weight)
# _op_index = 2, _cb_phase = {0: 0, 1: 0, 2: 1, 3: 1}
```

After composition:
- Phase 0 CBs: {0, 1} -- bfloat16 tensor-backed and scratch
- Phase 1 CBs: {2, 3} -- bfp8_b tensor-backed and scratch
- `_reconfig_boundaries = [1]`

### Step 2: prepare_for_build Triggers Multi-Phase

At `f.build()`, `prepare_for_build(fp)` runs:

1. **_compact_scratch_cbs**: Phase 0 has `cb_id=1` (scratch). Phase 1 has `cb_id=3` (scratch). Since they are in different phases and have different formats, they cannot be folded onto the same slot. Compaction is a no-op.

2. **_build_multi_phase**: `_reconfig_boundaries = [1]` is non-empty and `max(_cb_phase.values()) = 1 > 0`, so the multi-phase pipeline runs.

### Step 3: Inside _build_multi_phase

**Step 3a: Partition and validate.**

```
phase_cb_ids = [{0, 1}, {2, 3}]
```

`_validate_no_scratch_crossing`: cb_id=1 has lifetime `(0, 1)` and cb_id=3 has lifetime `(1, 2)`. The boundary is at op_index=1. For cb_id=1: `first=0 < 1 <= last=1` -- but cb_id=1's last use is AT the boundary, not after it. The check is `first < boundary <= last`, so this passes. For cb_id=3: `first=1` is not less than `boundary=1`, so it also passes. No crossing detected.

**Step 3b: Materialize scratch CBs.**

Phase 0's scratch: cb_id=1, `2 * 2048 = 4096` bytes.
Phase 1's scratch: cb_id=3, `4 * 1088 = 4352` bytes.

Each phase's scratch CBs get offsets independently:
```
Phase 0: cb_id=1 -> offset=0, total=4096
Phase 1: cb_id=3 -> offset=0, total=4352  (different phase, reuses offset 0)
Arena size = max(4096, 4352) = 4352 bytes
```

A single `HEIGHT_SHARDED` arena tensor of 4352 bytes per core is allocated. Both scratch CB descriptors are replaced with descriptors pointing into this arena.

**Step 3c: Assign CB IDs.**

Phase 0 context:
- cb_id=0 (bfloat16, 32x32): `_allocate_id` finds no existing entry (manager is fresh). Allocates ID 0. `_used_ids = {0}`.
- cb_id=1 (bfloat16, 32x32): `_allocate_id` finds ID 0 has matching format, but ID 0 is in `_used_ids`. Allocates fresh ID 1. `_used_ids = {0, 1}`.

Phase 1 context:
- cb_id=2 (bfp8_b, 32x32): `_allocate_id` scans `_id_to_format`. ID 0 has `(bfloat16, 32x32)` -- no match. ID 1 has `(bfloat16, 32x32)` -- no match. Allocates fresh ID 2. `_used_ids = {2}`.
- cb_id=3 (bfp8_b, 32x32): `_allocate_id` finds ID 2 has matching `(bfp8_b, 32x32)`, but ID 2 is in `_used_ids`. Allocates fresh ID 3. `_used_ids = {2, 3}`.

No remaps needed in this case -- all original IDs happen to match the allocator's assignments.

Note: if Phase 1 had included a bfloat16 CB, it would have reused ID 0 from Phase 0 (same format, not in Phase 1's `_used_ids`). This is how cross-phase sharing reduces ID pressure.

**Step 3d: Build reconfig tensors.**

For each phase, `_partition_descriptors_by_phase` groups the descriptors, `record_cb_metadata` extracts the CB config (including the arena addresses from Step 3b), and `build_cb_reconfig_tensor` produces the 264-word-per-core L1 tensor.

Phase 0 reconfig tensor encodes:
- CB 0: act_tensor's L1 address, total_size, num_pages, page_size
- CB 1: arena address + offset 0, 4096 bytes, 2 pages, 2048 page_size
- Bitmask word 256: `0b11` (bits 0 and 1 set)

Phase 1 reconfig tensor encodes:
- CB 2: weight_tensor's L1 address, total_size, num_pages, page_size
- CB 3: arena address + offset 0, 4352 bytes, 4 pages, 1088 page_size
- Bitmask word 256: `0b1100` (bits 2 and 3 set)

Both tensors are stored in `_io_tensors` with keys `-1000` and `-1001`.

**Step 3e: Patch placeholders.**

The `CbReconfig.emit()` placeholder from Step 1 (for Phase 1) is patched with the Phase 1 reconfig tensor's `buffer_address()`. The Phase 0 reconfig node is inserted at position 0 of the shadow graph.

**Step 3f: Dedup descriptors.**

`_dedup_cb_descriptors` runs on the final descriptor list. In this example, no CB IDs were remapped, so no duplicate descriptors exist. Dedup is a no-op.

### Step 4: Runtime Execution

The kernel binary now has:

1. A Phase 0 reconfig node that programs CBs 0-1 with bfloat16 addresses from the arena.
2. The gate computation using CBs 0 (activation) and 1 (scratch).
3. A Phase 1 reconfig node that reprograms CBs 2-3 with bfp8_b addresses from the arena.
4. The expert FFN computation using CBs 2 (weight) and 3 (scratch).

### What happens on silicon at each reconfig boundary

1. The kernel reads `cb_config_l1_addr` from its compile-time arguments -- the L1 address of the reconfig tensor's shard for this core.
2. The reconfig kernel reads the 264-word configuration block from that L1 address.
3. It checks the bitmask at words 256-257 to determine which CB slots to reprogram.
4. For each active CB, it writes the four configuration words (addr, size, num_pages, page_size) into the Tensix CB configuration registers.
5. The kernel uses semaphore synchronization (words 258-259) to ensure all cores complete reconfiguration before the next phase begins.
6. Execution continues with the next phase's ops, which now see reprogrammed CB slots pointing at new L1 regions.

The entire reconfiguration happens on-chip, within the running kernel. There is no host-device round-trip, no kernel re-dispatch, and no program descriptor rebuild.

### The complete data flow

```
  Composition Time                 Build Time                    Runtime
  ----------------                 ----------                    -------
  Phase 0 ops                      _compact_scratch_cbs()        cb_reconfig_0:
    cb_id=0 (tensor-backed)          merge compatible scratches     read 264 words
    cb_id=1 (scratch, bfloat16)                                     program CB table
                                   _build_multi_phase():            set active mask
  CbReconfig.emit()                  1. partition by phase
    placeholder addr=0               2. materialize scratch       Gate computation:
                                        -> arena tensor             CB0 = act, CB1 = scratch
  Phase 1 ops                        3. assign IDs via manager
    cb_id=2 (tensor-backed)          4. build reconfig tensors    cb_reconfig_1:
    cb_id=3 (scratch, bfp8_b)        5. patch placeholders          read 264 words
                                     6. insert phase0 init          reprogram CB table
  f.build()                          7. dedup descriptors
                                                                  Expert FFN:
                                                                    CB2 = weight, CB3 = scratch
```

## What Would Break If CB Reconfig Were Removed

1. **64-CB hard limit becomes binding.** Any fused program needing more than 64 logically distinct CBs would fail at allocation time. Complex models (attention + MLP + gating) routinely exceed this limit.
2. **Scratch CBs have no L1 address.** The arena materialization step gives scratch CBs their L1 addresses. Without it, scratch CBs would need individual tensor allocations, fragmenting L1 and wasting memory.
3. **Cross-phase CB-ID reuse disappears.** CBs in different phases would each consume a unique slot, exacerbating the 64-slot exhaustion problem.
4. **Runtime CB reconfiguration is impossible.** The reconfig tensor is the mechanism by which the kernel learns new CB addresses at phase boundaries. Without it, the kernel would read stale addresses from the previous phase.

---

<- [03 -- OverlappedView](03_overlapped_view.md) | [Index](index.md) ->
