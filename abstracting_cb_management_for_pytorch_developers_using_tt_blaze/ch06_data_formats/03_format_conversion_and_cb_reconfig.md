# Format Conversion and CB Reconfig: The Mechanics of Format Transitions

Once the negotiation algorithm (File 02) identifies where format transitions must occur, the system must implement those transitions using existing TT-Blaze primitives. This section explains the three mechanisms for format conversion -- tile_row_convert ops, retilize ops, and in-CB reconfig -- details the `CircularBufferIdManager` infrastructure for CB ID reuse across format-differing phases, walks through the `cb_reconfig_builder.py` multi-phase transformation pipeline, and identifies the cases where format conversion is effectively free (handled at the SFPU level by the hardware unpacker). The walkthroughs use DeepSeek V3's gated MLP (three matmul phases with different weight formats) and trace CB ID allocation step by step.

> *Design Principle P3 (Validate at Each Lowering Step):* Every format conversion must be validated before execution. The conversion must preserve the logical shape, maintain padding correctness (fill values representable in the target format), and fit within the L1 budget including the conversion CB.

**What you will learn:**

- The three conversion primitives: tile_row_convert, retilize, and in-CB reconfig -- when each applies and what it costs
- How `CircularBufferIdManager` from `cb_reconfig.py` manages CB ID allocation across format-differing phases, including the `_allocate_id()` reuse logic and the `Context` scoping mechanism
- The `cb_reconfig_builder.py` infrastructure: how `prepare_for_build()` transforms a multi-phase FusedProgram, materializes scratch CBs into a shared arena, builds per-phase reconfig tensors, and remaps CB IDs
- When format conversion is free: the complete list of SFPU-level conversions handled by the unpacker/packer hardware
- `cb_alias` for zero-copy format reinterpretation within fused ops
- Walked example: a three-phase gated MLP showing CB ID reuse across phases with different weight formats
- How TensorAdapter integrates with the reconfig infrastructure to insert format transitions automatically

---

## Conversion Primitives

### Primitive 1: Tile Row Convert

Tile row convert changes the storage format of a tile while preserving its logical content. It reads from one CB, converts element-by-element through the SFPU, and writes to another CB.

```text
Input CB:  bfloat8_b tiles (1,088 bytes each)
  |
  v  [Unpack: bfloat8_b -> bfloat16 in SRCA]
  |
  v  [SFPU: identity pass]
  |
  v  [Pack: bfloat16 -> output CB]
  |
Output CB: bfloat16 tiles (2,048 bytes each)
```

**Cost:** One TRISC cycle per tile (unpack + identity + pack). Requires two CB slots (input and output) and the output CB's L1 allocation.

**When used:**
- Converting BFP weights to standard format when the downstream op cannot consume BFP directly (rare -- most ops handle BFP via the unpacker)
- Converting between incompatible tile geometries that also require format change

### Primitive 2: Retilize

Retilize changes the tile geometry of data in L1 without changing the format. It reads row-tiles (1x32) and writes compute-tiles (32x32 or 16x32), or vice versa.

```text
Input CB:  1x32 row tiles (64 bytes each, bfloat16)
  |
  v  [NCRISC: gather 32 row-tiles into one 32x32 tile]
  |
Output CB: 32x32 compute tiles (2,048 bytes each, bfloat16)
```

The `interpret_tile()` function from `blaze/utils.py` determines the target geometry:

```python
# EXISTING -- from blaze/utils.py
def interpret_tile(width: int) -> tuple[ttnn.Tile, int]:
    FULL = ttnn.Tile((32, 32))
    HALF = ttnn.Tile((16, 32))
    use_half = (width // FULL.tile_shape[1]) % FULL.tile_shape[0] != 0
    tile = HALF if use_half else FULL
    th, tw = tile.tile_shape
    return tile, width // (th * tw)
```

**Cost:** NCRISC data movement (gather from row-tile addresses to compute-tile address). No SFPU compute, but L1 bandwidth and a second CB slot.

**When used:**
- Converting from storage layout (1x32 row tiles from DRAM read) to compute layout (32x32 tiles for FPU operations)
- This is the common path for RMSNorm gamma weights and activation vectors stored as row-major data

### Primitive 3: In-CB Reconfig

In-CB reconfig changes the metadata (format, page_size, tile geometry) associated with a CB slot without moving data. The `CircularBufferIdManager` and `cb_reconfig_builder.py` implement this by writing a reconfig tensor that reprograms CB descriptors between program phases.

```text
Phase 0:  CB 3 = {bfloat8_b, 1,088 bytes/page, 32x32 tiles}
  |
  v  [CbReconfig kernel reads reconfig tensor, reprograms CB descriptors]
  |
Phase 1:  CB 3 = {bfloat16, 2,048 bytes/page, 32x32 tiles}
```

**Cost:** One CbReconfig kernel invocation per phase boundary (a few microseconds). No data movement -- the CB slot's L1 address, format, and page_size are reprogrammed via the reconfig tensor.

> **Warning:** In-CB reconfig changes the **interpretation** of the CB's memory, not the **content**. If Phase 0 wrote bfloat8_b data to CB 3 and Phase 1 reconfigures CB 3 to bfloat16, the Phase 1 reader will interpret the bfloat8_b bytes as bfloat16 -- producing garbage. Reconfig is only valid when the Phase 1 producer writes fresh data in the new format before the consumer reads.

### cb_alias: Zero-Copy Format Reinterpretation

In addition to the three conversion primitives, `cb_alias` (from `fused_program.py`) provides zero-copy format reinterpretation for FIFO CBs. As described in File 02, `cb_alias` creates a second CB ID pointing at the same L1 bytes with a different format descriptor. This is useful when one phase needs to read data as bfloat8_b and a subsequent phase needs to read the same data as bfloat16 -- the unpack hardware handles the difference transparently.

The key constraint: `cb_alias` does not support direct-address CBs. Only FIFO-mode CBs can be aliased.

---

## CircularBufferIdManager: Format-Aware CB ID Reuse

The `CircularBufferIdManager` in `cb_reconfig.py` is the core infrastructure for managing CB IDs across multiple format contexts. It enables the 64-slot hardware limit to accommodate more logical CBs than physical slots.

### Architecture

```python
# EXISTING -- from blaze/cb_reconfig.py
class CircularBufferIdManager:
    NUM_CIRCULAR_BUFFERS = 64

    def __init__(self):
        self._id_to_format: dict[int, tuple] = {}
        self._next_id = 0
```

The manager maintains a mapping from CB ID to `(data_format, tile)` pair. When a new CB is requested, the manager first checks whether an existing CB ID has a matching format -- if so, it can be reused.

### The _allocate_id() Method: Reuse Logic

```python
# EXISTING -- from blaze/cb_reconfig.py
def _allocate_id(self, data_format, tile, exclude):
    """Reuse an existing ID with matching (data_format, tile), or allocate a fresh one."""
    if tile is not None and not isinstance(tile, ttnn.TileDescriptor):
        raise TypeError(f"tile must be a ttnn.TileDescriptor, got {type(tile).__name__}")
    if not isinstance(data_format, ttnn.DataType):
        raise TypeError(f"data_format must be a ttnn.DataType, got {type(data_format).__name__}")
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

The reuse logic is keyed on `(data_format, tile)`. Two CBs with the same format and tile geometry share a CB ID as long as they are not in the same context (same program phase). Two CBs with different formats always get different CB IDs.

### The Context Scoping Mechanism

Each program phase gets a `Context` that tracks which CB IDs are in use within that phase:

```python
# EXISTING -- from blaze/cb_reconfig.py
class Context:
    def __init__(self, manager):
        self._manager = manager
        self._used_ids: set[int] = set()

    def get_cb_id(self, data_format, tile) -> int:
        cb_id = self._manager._allocate_id(data_format, tile, self._used_ids)
        self._used_ids.add(cb_id)
        return cb_id
```

Within a context, every CB ID is unique (the `exclude=self._used_ids` parameter prevents reuse). Across contexts, IDs reuse when formats match.

### CBContext: Named Phase Handle

The `CBContext` class wraps the `Context` with a name and additional operations:

```python
# EXISTING -- from blaze/cb_reconfig.py
class CBContext:
    def __init__(self, name, manager):
        self.name = name
        self._manager = manager
        self._ctx = manager.create_context()
        self.used_ids = self._ctx._used_ids
        self.descriptors = []

    def get_cb_id(self, data_format, tile) -> int:
        return self._ctx.get_cb_id(data_format, tile)

    def pin(self, cb_id, data_format, tile):
        """Reserve a specific CB ID with the given format in this context."""
        self.used_ids.add(cb_id)
        td = self._manager._normalize_tile(tile)
        self._manager._id_to_format[cb_id] = (data_format, td)
        self._manager._next_id = max(self._manager._next_id, cb_id + 1)
```

The `pin()` method allows reserving a specific CB ID -- used when tensor-backed CBs have fixed CB IDs from the initial allocation phase and subsequent phases must respect those assignments.

### The CBEngine-to-CircularBufferIdManager Bridge

The `CBEngine` (from `cb_engine.py`) manages CB allocation during the graph-level compilation path, while `CircularBufferIdManager` manages CB IDs during the composition API path. The `to_cb_id_manager()` method bridges these two:

```python
# EXISTING -- from blaze/cb_engine.py (simplified)
def to_cb_id_manager(self) -> CircularBufferIdManager:
    """Export graph-level CB allocations into a CircularBufferIdManager
    for use by the composition API."""
    manager = CircularBufferIdManager()
    for cb_id, meta in self._allocations.items():
        manager.seed_reserved_cb(cb_id, meta.data_format, meta.tile)
    return manager
```

The `seed_reserved_cb` method pre-populates the manager with CB IDs that are already allocated by the graph compiler, ensuring the composition API does not accidentally reuse or collide with graph-level allocations.

---

## cb_reconfig_builder.py: The Multi-Phase Transformation

The `cb_reconfig_builder.py` module implements the complete multi-phase transformation pipeline. It is called from `FusedProgram._prepare_for_build()` after all `emit()` calls are complete.

### The prepare_for_build() Entry Point

```python
# EXISTING -- from blaze/cb_reconfig_builder.py
def prepare_for_build(fp: FusedProgram) -> None:
    """Apply scratch-CB compaction and multi-phase reconfig if needed. Idempotent."""
    if fp._compaction_applied:
        return
    fp._compaction_applied = True
    if fp._compact_scratch_cbs:
        _compact_scratch_cbs(fp)
    if fp._reconfig_boundaries and _has_multi_phase_cbs(fp):
        fp._build_multi_phase()
    elif fp._reconfig_placeholders:
        placeholder_nodes = {id(node) for _, node, _ in fp._reconfig_placeholders}
        fp._shadow_graph.nodes = [
            n for n in fp._shadow_graph.nodes if id(n) not in placeholder_nodes
        ]
        fp._reconfig_placeholders.clear()
```

### Step-by-Step Pipeline

The `_build_multi_phase()` function implements six steps:

**Step 1: Partition CBs into phases**

Each CB is assigned to a phase based on when it was allocated relative to `reconfig()` boundaries. The validator ensures no scratch CB is used across a boundary (scratch L1 addresses change with reconfig).

**Step 2: Materialize scratch CBs to L1**

All scratch CBs across all phases are packed into a single L1 arena tensor. The arena is sized to the maximum total of any single phase. Phases share the arena memory because they execute sequentially -- Phase 1's scratch CBs overwrite Phase 0's. Offsets within the arena are 16-byte aligned.

**Step 3: Assign CB IDs across phases**

```python
# EXISTING -- from cb_reconfig_builder.py (Step 3)
manager = fp.cb_manager
phase_remaps: list[dict[int, int]] = []

for phase_idx in range(num_phases):
    ctx = manager.create_context()
    remap = {}
    for cb_id in sorted(phase_cb_ids[phase_idx]):
        meta = fp._cb_metadata.get(cb_id)
        if meta is None:
            continue
        new_id = ctx.get_cb_id(meta.data_format, meta.tile_desc)
        if new_id != cb_id:
            remap[cb_id] = new_id
    phase_remaps.append(remap)
```

Each phase gets a fresh `Context` from the shared `CircularBufferIdManager`. The context's `get_cb_id()` method reuses IDs when `(data_format, tile)` matches.

**Step 4: Build per-phase reconfig tensors**

Each reconfig tensor is a HEIGHT_SHARDED L1 tensor with 264 uint32 values per core:

```python
# EXISTING -- from cb_reconfig.py
WORDS_PER_CORE = 264  # 64 CBs x 4 words [addr, size, num_pages, page_size]
                       # + 2 mask words + 2 sync semaphores + 4 reserved
```

The reconfig tensor programs each core's CB registers: for each active CB ID, it sets the L1 address, total size, num_pages, and page_size. The CbReconfig kernel reads this tensor at the phase boundary and reprograms the CB hardware.

**Step 5: Patch placeholder addresses and insert Phase 0 init**

`CbReconfig.emit()` stamps `cb_config_l1_addr=0` as a placeholder because the real L1 address is not known until the reconfig tensor is allocated. This step patches the placeholders with actual addresses. Phase 0's reconfig node is inserted at position 0 in the shadow graph.

**Step 6: Deduplicate CB descriptors**

After remapping, multiple descriptors may claim the same CB ID. Deduplication keeps the descriptor with the largest total_size (so the L1 allocation covers all users) and handles disjoint-core cases (where different cores have different descriptors at the same CB ID).

---

## Scratch CB Compaction: Spatial and Temporal Reuse

Before multi-phase processing, `_compact_scratch_cbs()` merges non-conflicting scratch CBs:

### Spatial Reuse

Two scratch CBs with the same `(data_format, tile, page_size, num_pages)` on **disjoint core ranges** can share a CB ID:

```text
CB 5: bfloat16, 32x32, 2048 bytes, cores (0,0)-(3,0)
CB 8: bfloat16, 32x32, 2048 bytes, cores (4,0)-(7,0)
  -> Compacted: both use CB 5 (disjoint cores, same format)
```

### Temporal Reuse

Two scratch CBs with the same format key on the **same cores** but **non-overlapping lifetimes** can share a CB ID with a `CbScratchReset` node inserted between them.

Temporal reuse has restrictions:

```python
# EXISTING -- from cb_reconfig_builder.py
is_unbalanced = cb_id in fp._unbalanced_cbs
is_tensor_backed = cb_id in fp._tensor_backed_cbs
cb_allow_temporal = (allow_temporal
                     and not is_unbalanced
                     and not is_tensor_backed)
```

- **Unbalanced CBs** (different page counts across cores) cannot temporally reuse because the wrap-around arithmetic differs.
- **Tensor-backed CBs** cannot temporally reuse because they have fixed L1 addresses that collide.

---

## When Conversion Is Free: The SFPU-Level Path

### Free Conversions (Zero Extra CBs, Zero Extra Cycles)

| Conversion | Mechanism | Where It Happens | Cost |
|-----------|-----------|-----------------|------|
| bfloat8_b -> bfloat16 | Unpacker expansion | TRISC0 unpack stage | 0 (hardware pipeline) |
| bfloat4_b -> bfloat16 | Unpacker expansion | TRISC0 unpack stage | 0 (hardware pipeline) |
| bfloat16 -> float32 (accum) | `fp32_dest_acc_en=True` | TRISC1 FPU dest register | 0 (register width) |
| float32 -> bfloat16 (pack) | Packer truncation | TRISC2 pack stage | 0 (hardware pipeline) |

The unpacker automatically converts BFP-format tiles to bfloat16 when loading them into the SRCA or SRCB registers. This is part of the normal tile-load pipeline -- no extra instructions, no extra CB slots, no extra time.

### Walkthrough: Free Conversion in LLaMA Matmul

```text
Weight tensor: stored as bfloat8_b in DRAM
  |
  v  [NCRISC: stream bfloat8_b tiles from DRAM to weight CB]
  |     Weight CB: data_format = bfloat8_b, page_size = 1,088
  v
  |  [TRISC0 (unpack): read bfloat8_b tile, expand to bfloat16 in SRCB]
  |     -- This is the free conversion --
  v
  |  [TRISC1 (math): multiply SRCA (bfloat16 activation) x SRCB (bfloat16 from unpack)]
  |     -- Compute in bfloat16, accumulate in dest register
  v
  |  [TRISC2 (pack): write bfloat16 result to output CB]
  |     Output CB: data_format = bfloat16, page_size = 2,048
  v
```

### Near-Free Conversions (Packer-Level)

| Conversion | Mechanism | Cost |
|-----------|-----------|------|
| bfloat16 -> bfloat8_b | Packer compression | ~1 cycle/tile |
| bfloat16 -> bfloat4_b | Packer compression | ~1 cycle/tile |
| float32 -> bfloat8_b | Packer compression | ~1 cycle/tile |
| float32 -> bfloat4_b | Packer compression | ~1 cycle/tile |

### Expensive Conversions (Require Explicit Ops)

| Conversion | Why Expensive | Estimated Cost |
|-----------|---------------|---------------|
| Retilize (1x32 -> 32x32) | Data movement in L1 | 10-50 us per tile batch |
| Cross-format with geometry change | Two-step: retilize + reformat | 20-100 us |
| Any conversion needing a separate kernel | Full kernel launch overhead | 5-10 us + per-tile cost |

---

## Walkthrough: Three-Phase Gated MLP with Format Transitions

Consider a gated MLP with three matmul phases where weight formats differ:

```text
Phase 0: RMSNorm + Mcast
  CB 0: activation input (bfloat16, FIFO, from tensor)
  CB 1: gamma (bfloat16, direct-address)
  CB 2: rmsnorm output (bfloat16, scratch)
  CB 3: mcast output (bfloat16, scratch)

  -- reconfig() boundary --

Phase 1: Gate matmul + Up matmul + SiLU + EltwiseMul
  CB 3: mcast act (bfloat16, reuse from Phase 0 -- same format)
  CB 4: gate weight (bfloat8_b, direct-address)
  CB 5: up weight (bfloat8_b, direct-address)
  CB 6: gate output (bfloat16, scratch)
  CB 7: up output (bfloat16, scratch)
  CB 8: silu*up output (bfloat16, scratch)

  -- reconfig() boundary --

Phase 2: Down matmul
  CB 8: silu*up result (bfloat16, reuse -- same format)
  CB 9: down weight (bfloat4_b, direct-address)  -- DIFFERENT format
  CB 6: down output (bfloat16, scratch, reuse of ID 6)
```

### CB ID Allocation Trace via CircularBufferIdManager

```text
Manager state after Phase 0:
  ID 0: (bfloat16, 32x32)    -- activation
  ID 1: (bfloat16, 32x32)    -- gamma
  ID 2: (bfloat16, 32x32)    -- rmsnorm out
  ID 3: (bfloat16, 32x32)    -- mcast out

Phase 1 context creates:
  ID 3: reused (bfloat16, 32x32)  -- mcast act (match found, not in exclude)
  ID 4: new (bfloat8_b, 32x32)    -- gate weight (no bfloat8_b ID exists)
  ID 5: new (bfloat8_b, 32x32)    -- up weight (ID 4 in exclude set)
  ID 6: new (bfloat16, 32x32)     -- gate out (IDs 0-3 in exclude)
  ID 7: new (bfloat16, 32x32)     -- up out (IDs 0-3,6 in exclude)
  ID 8: new (bfloat16, 32x32)     -- silu*up out

Phase 2 context creates:
  ID 8: reused (bfloat16, 32x32)  -- silu*up in (match found)
  ID 9: new (bfloat4_b, 32x32)    -- down weight (NO match: bfloat4_b != bfloat8_b)
  ID 6: reused (bfloat16, 32x32)  -- down out (match found)

Total unique CB IDs: 10 (out of 64)
```

Note that the down weight (bfloat4_b) cannot reuse CB ID 4 or 5 (bfloat8_b) because the format key `(bfloat4_b, 32x32)` does not match `(bfloat8_b, 32x32)`. The reconfig tensor programs different page_size values for different formats, and aliasing would produce wrong page_size.

### Format Conversions in This Example

All format conversions are free:
- Phase 1 gate matmul: bfloat8_b weights unpacked to bfloat16 by hardware
- Phase 1 up matmul: same
- Phase 2 down matmul: bfloat4_b weights unpacked to bfloat16 by hardware

No explicit conversion ops are needed.

---

## How TensorAdapter Integrates with Reconfig

# PROPOSED

TensorAdapter's format negotiation feeds into the existing reconfig infrastructure through a `NegotiatedCBPlan` that bundles negotiation results with CB reconfig instructions:

```python
# PROPOSED -- NegotiatedCBPlan bridges negotiation to reconfig
@dataclass
class NegotiatedCBPlan:
    """Output of the format negotiation algorithm, input to the reconfig pipeline."""
    format_assignments: dict[str, ttnn.DataType]   # edge_id -> assigned format
    conversions: list[ConversionNode]               # explicit conversions needed
    free_conversions: list[FreeConversion]           # unpacker-handled transitions
    alias_opportunities: list[AliasOp]              # zero-copy cb_alias uses
    cb_id_estimate: int                             # estimated CB ID consumption
```

The `FormatIntegrator` applies the plan to a `FusedProgram`:

```python
# PROPOSED -- FormatIntegrator
class FormatIntegrator:
    """Bridges format negotiation decisions to FusedProgram reconfig."""

    def apply(self, plan: NegotiatedCBPlan, fp: FusedProgram) -> None:
        """Apply negotiated format decisions to a FusedProgram.

        1. For free conversions: set the consumer CB's data_format.
           The unpacker handles the rest.
        2. For alias opportunities: call fp.cb_alias() to create
           second CB IDs on the same buffer.
        3. For explicit conversions: insert conversion micro-ops
           between producer and consumer phases, calling fp.reconfig()
           at each new phase boundary.
        4. The existing cb_reconfig_builder.py handles the rest:
           CB ID remapping, arena allocation, reconfig tensor construction.
        """
        for conv in plan.conversions:
            if conv.cost == 0:
                pass  # Free: unpacker handles it
            elif conv.cost <= CONVERSION_COSTS["pack_convert"]:
                pass  # Pack-level: set output CB format
            else:
                self._insert_conversion_op(fp, conv)
                if self._crosses_phase_boundary(conv):
                    fp.reconfig()

        for alias in plan.alias_opportunities:
            fp.cb_alias(alias.source_handle,
                       dtype=alias.target_format,
                       page_size=alias.target_page_size)
```

### The Integration Flow

```text
Format Negotiation (File 02)
  |
  v  NegotiatedCBPlan
  |
FormatIntegrator.apply()
  |
  +-- Free conversions: set CB data_format
  +-- Alias opportunities: call fp.cb_alias()
  +-- Explicit conversions: insert micro-op + reconfig()
  |
  v
FusedProgram._prepare_for_build()
  |
  v
cb_reconfig_builder.prepare_for_build()
  |
  +-- _compact_scratch_cbs()     -- spatial + temporal reuse
  +-- _build_multi_phase()       -- CB ID assignment, reconfig tensors
  |
  v
Compiled kernel with correct per-phase CB configurations
```

---

## Validation: Format Conversion Safety

# PROPOSED

Before inserting a format conversion, TensorAdapter validates:

```python
# PROPOSED -- conversion validation
def validate_conversion(
    source_format: ttnn.DataType,
    target_format: ttnn.DataType,
    padding_strategy: str,
) -> list[str]:
    """Validate that a format conversion is safe."""
    issues = []

    if padding_strategy == "NEG_INF":
        if target_format == ttnn.bfloat4_b:
            issues.append(
                "NEG_INF padding may not fully suppress exp() in bfloat4_b "
                "(limited dynamic range). Promote to bfloat16 before softmax."
            )

    if source_format == ttnn.float32 and target_format in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        issues.append(
            f"Converting float32 -> {target_format}: significant precision loss. "
            "Verify the downstream op does not require float32."
        )

    return issues
```

---

## Key Takeaways

- Three conversion mechanisms exist: **tile_row_convert** (explicit SFPU conversion between CBs), **retilize** (tile geometry change without format change), and **in-CB reconfig** (CB descriptor reprogramming between phases via reconfig tensors). Reconfig is the primary mechanism for format transitions within fused ops. Additionally, **cb_alias** provides zero-copy format reinterpretation for FIFO CBs.
- `CircularBufferIdManager` manages CB ID allocation across phases by keying reuse on `(data_format, tile)` pairs. Within a phase (Context), every CB ID is unique. Across phases, matching formats reuse the same CB ID, conserving the 64-slot hardware limit. The `seed_reserved_cb` method bridges graph-level CB allocations from `CBEngine` to the composition API.
- The `cb_reconfig_builder.py` pipeline transforms a multi-phase FusedProgram in six steps: partition CBs into phases, materialize scratch CBs into a shared L1 arena, assign CB IDs via the manager, build per-phase reconfig tensors (264 uint32 words per core), patch placeholder addresses, and deduplicate descriptors.
- Most format conversions are **free**: the hardware unpacker converts `bfloat8_b` and `bfloat4_b` to `bfloat16` automatically during the unpack stage, and the packer can compress from `bfloat16`/`float32` to BFP formats during the pack stage. Weight compression (bfloat8_b/bfloat4_b storage with bfloat16 compute) adds zero conversion overhead.
- TensorAdapter integrates with the existing reconfig infrastructure through a `NegotiatedCBPlan` that bundles format assignments, conversion decisions, and alias opportunities. The `FormatIntegrator` translates these into `cb_scratch()` calls, `cb_alias()` calls, and `reconfig()` boundaries, and the existing `cb_reconfig_builder.py` handles the rest.

## Source Files

- `blaze/cb_reconfig.py` -- `CircularBufferIdManager` (format-keyed CB ID reuse), `CBContext` (named phase handle), `record_cb_metadata()`, `build_cb_reconfig_tensor()` (264-word per-core L1 tensor construction)
- `blaze/cb_reconfig_builder.py` -- `prepare_for_build()` (entry point), `_compact_scratch_cbs()` (spatial + temporal CB reuse), `_build_multi_phase()` (6-step pipeline), `_materialize_scratch_cbs()`, `_remap_cb_ids()` (CT arg + descriptor rewriting)
- `blaze/fused_program.py` -- `FusedProgram.reconfig()` (phase boundary), `_format_key()` (sharing key), `cb_from_view()` (tile reinterpretation), `cb_alias()` (zero-copy format alias)
- `blaze/cb_engine.py` -- `CBEngine.to_cb_id_manager()` (graph-to-composition bridge)
- `blaze/utils.py` -- `interpret_tile()`, `interpret_tile_padded()` (tile geometry selection)

---

← Previous: [Chapter 5 — Padding](../ch05_padding/) | Next: [Chapter 7 — Automatic CB Sizing and L1 Memory Budgeting](../ch07_cb_sizing/) →

< Previous: [File 02 -- Format Negotiation Protocol](./02_format_negotiation_protocol.md) | Next: [File 04 -- Precision Profiles and User Overrides](./04_precision_profiles_and_user_overrides.md) >
