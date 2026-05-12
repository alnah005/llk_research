# CB Compaction and Temporal Reuse

A fused kernel with 40+ micro-ops can allocate dozens of circular buffers, each with a distinct ID. But the Blackhole hardware supports only 64 CB IDs per core, and many of these buffers are not alive simultaneously -- an RMSNorm output is consumed and freed before the downstream matmul's output is allocated. This temporal non-overlap creates an opportunity: two CBs whose lifetimes do not overlap can share the same hardware ID, reducing the total CB count without changing any data or compute behavior. TT-Blaze already implements this optimization through `CBEngine.compact_cb_ids()` and `CircularBufferIdManager` in `cb_reconfig.py`. This section explains these existing mechanisms, frames compaction as a graph coloring problem with formal guarantees, shows how the ShapeTracker from [Chapter 3](../ch03_tensor_adapter/) provides lifetime information automatically, and designs the integration with the automatic sizing system from Files 01-03.

> *Design Principle P3 (Validate at Each Lowering Step, Not Just at the End):* The compaction pass validates that the 64 CB ID hardware limit is not exceeded. If uncompacted CB usage would breach this limit, the pass catches it at compile time — before it becomes a silent hardware failure — and applies interval coloring to resolve it.

> *Design Principle P2 (Carry Metadata from Both Worlds Simultaneously):* CBHandle objects carry both semantic meaning ("matmul output," "RMSNorm gamma") and hardware metadata (data_format, page_size, num_pages). The compaction pass maps these rich descriptors to hardware CB IDs, preserving the metadata needed for format-aware reuse while the developer operates at the semantic level.

**What you will learn:**

- The interval coloring algorithm in `CBEngine.compact_cb_ids()`: how non-overlapping lifetimes are detected and recolored
- The `CBAssignment.first_use` and `last_use` fields: how topological position drives lifetime tracking
- The `CircularBufferIdManager` and its `Context` system: format-aware CB ID reuse across multi-phase programs
- How `CBEngine(compact=True)` triggers compaction and what the developer observes
- How the ShapeTracker propagates lifetime information automatically, eliminating manual annotation
- The 64 CB ID wall: concrete scenarios from DeepSeek V3 MoE and GLM-5.1 that approach or exceed the limit
- The proposed integration: `AutoSizer` + `AutoBlocker` + compaction as a unified compile-time pipeline

---

## 1. The Interval Coloring Algorithm

CB compaction is a classic interval coloring problem. Given a set of intervals (each CB's `[first_use, last_use]` range), assign colors (CB IDs) such that no two overlapping intervals share the same color, while minimizing the total number of colors used.

### Graph Coloring Theory

The CB slot assignment problem maps directly to interval graph coloring, a well-studied problem in combinatorial optimization:

- Each CB has a lifetime interval `[f_i, l_i]` where `f_i` is the first program point that uses it and `l_i` is the last.
- Two CBs conflict if their lifetimes overlap: `f_i <= l_j` and `f_j <= l_i`.
- A coloring assigns a color (CB slot ID) to each CB such that no two conflicting CBs share a color.
- The chromatic number (minimum colors needed) equals the clique number (maximum number of pairwise-overlapping intervals at any program point).

For interval graphs, the chromatic number is achievable by a simple greedy algorithm that processes intervals in order of start time. This means `compact_cb_ids()` produces an optimal coloring -- no algorithm can use fewer CB IDs.

### The Implementation

TT-Blaze implements this in `CBEngine.compact_cb_ids()`:

```python
# EXISTING -- from blaze/cb_engine.py (lines 422-437)
def compact_cb_ids(self, assignments: dict[str, "CBAssignment"]) -> dict[str, "CBAssignment"]:
    """Recolor CB IDs via interval coloring.

    Two CBs with non-overlapping lifetimes [first_use, last_use] can
    share the same ID. This reduces total CB count for long pipelines.
    """
    sorted_cbs = sorted(assignments.values(), key=lambda a: a.first_use)
    color_end: dict[int, int] = {}  # color -> last_use
    for cb in sorted_cbs:
        # Find lowest available color whose prior user is done
        color = 0
        while color in color_end and color_end[color] >= cb.first_use:
            color += 1
        cb.cb_id = color
        color_end[color] = cb.last_use
    return assignments
```

### How the Algorithm Works

The algorithm processes CBs in order of their `first_use` (topological position of the producing op). For each CB:

1. Start with color 0
2. Check if color 0 is free (its previous occupant's `last_use` is before the current CB's `first_use`)
3. If not, try color 1, 2, ... until a free color is found
4. Assign the color and record the current CB's `last_use` as the color's new end time

This greedy approach produces an optimal coloring for interval graphs (proven by the interval scheduling theory).

### Worked Example

Consider a fused op with 8 CBs and their lifetimes (topological positions):

```text
CB  first_use  last_use  Purpose
A        0         1     RMSNorm input
B        0         1     RMSNorm gamma
C        1         2     RMSNorm output / Mcast input
D        2         3     Mcast output / Matmul activation
E        2         4     Matmul weights (direct-address, long-lived)
F        3         4     Matmul output / EltwiseMul input
G        4         5     EltwiseMul output / Gather input
H        5         6     Gather output

Timeline:
  0    1    2    3    4    5    6
  A----+
  B----+
       C----+
            D----+
            E---------+
                 F----+
                      G----+
                           H----+
```

**Without compaction:** 8 CB IDs (A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7)

**With compaction (greedy coloring):**

```text
Process A (first_use=0): color 0 is free -> A=0, color_end={0: 1}
Process B (first_use=0): color 0 end=1 >= 0, try 1 -> B=1, color_end={0: 1, 1: 1}
Process C (first_use=1): color 0 end=1 >= 1, color 1 end=1 >= 1, try 2 -> C=2
Process D (first_use=2): color 0 end=1 < 2, free! -> D=0, color_end={0: 3, ...}
Process E (first_use=2): color 0 end=3 >= 2, color 1 end=1 < 2, free! -> E=1
Process F (first_use=3): color 0 end=3 >= 3, color 1 end=4 >= 3, color 2 end=2 < 3 -> F=2
Process G (first_use=4): color 0 end=3 < 4, free! -> G=0
Process H (first_use=5): color 0 end=5 >= 5, color 1 end=4 < 5, free! -> H=1
```

**Result:** 3 CB IDs instead of 8.

```text
Color 0: A (0-1), D (2-3), G (4-5)
Color 1: B (0-1), E (2-4), H (5-6)
Color 2: C (1-2), F (3-4)
```

This is a 62.5% reduction in CB IDs, well below the 64-slot limit.

### Visualization of Compacted Assignment

```text
CB ID timeline (after compaction):
  ID 0: [A  ][  ][D  ][  ][G  ][  ]
  ID 1: [B  ][  ][E--------][H  ]
  ID 2:      [C  ][  ][F  ]

  0    1    2    3    4    5    6   (topological position)
```

---

## 2. Lifetime Tracking via first_use and last_use

The compaction algorithm requires accurate lifetime information. The `CBAssignment` dataclass carries this:

```python
# EXISTING -- from blaze/cb_engine.py (lines 122-138)
@dataclass
class CBAssignment:
    """Result of CB assignment for one data path."""
    cb_id: int
    total_size: int
    page_size: int
    num_pages: int
    core_ranges: Any
    is_tensor_backed: bool
    data_format: str
    tile_shape: tuple[int, int]
    node_id: str = ""
    port_name: str = ""
    category: str = ""
    first_use: int = 0   # topo position of first node that reads/writes
    last_use: int = 0    # topo position of last node that reads/writes
```

### How Lifetimes Are Computed

The CBEngine computes lifetimes from the topological ordering of the BlazeGraph:

```python
# EXISTING -- from blaze/cb_engine.py (lines 358-374)
# Compute CB lifetimes from topological positions
topo_pos = {n.id: i for i, n in enumerate(topo_order)}
for key, cb in assignments.items():
    if cb.node_id:
        producer_pos = topo_pos.get(cb.node_id, 0)
        cb.first_use = producer_pos
        cb.last_use = producer_pos

# Extend last_use for inter-op edges (consumer topo position)
for edge in graph.edges:
    if edge.cb_id is not None:
        for key, cb in assignments.items():
            if cb.cb_id == edge.cb_id and cb.category == "intermed":
                consumer_pos = topo_pos.get(edge.consumer.id, 0)
                cb.last_use = max(cb.last_use, consumer_pos)
                break
```

The logic:
1. Each CB's `first_use` is the topological position of its producing node
2. Each CB's `last_use` starts at the same position (producer)
3. For inter-op edges, `last_use` is extended to the consumer's topological position
4. For fan-out edges (one producer feeding multiple consumers), `last_use` is the maximum consumer position

### Lifetime Categories

| Category | first_use | last_use | Lifetime Pattern |
|----------|-----------|----------|-----------------|
| `ext_in` (external input) | First consumer position | Last consumer position | Read-only, short-lived |
| `internal` (scratch) | Owner node position | Owner node position | Single-phase, shortest |
| `intermed` (inter-op) | Producer position | Last consumer position | Varies by fan-out |
| `ext_out` (terminal output) | Producer position | End of program | Long-lived |

External outputs have the longest lifetimes because they persist until the program completes. This limits their reuse potential -- output CB IDs are rarely shareable.

---

## 3. The CircularBufferIdManager

For multi-phase programs (fused ops with CB reconfiguration), simple interval coloring is insufficient. Different phases may need the same CB ID to carry different data formats. The `CircularBufferIdManager` in `cb_reconfig.py` handles this:

```python
# EXISTING -- from blaze/cb_reconfig.py (lines 18-90)
class CircularBufferIdManager:
    """Manages circular buffer ID allocation across multiple contexts.

    CB IDs can be reused across contexts when the data_format and tile match,
    but within a single context every CB ID must be unique.
    """
    NUM_CIRCULAR_BUFFERS = 64

    def __init__(self):
        self._id_to_format: dict[int, tuple] = {}
        self._next_id = 0
```

### Format-Aware Reuse

The key innovation: CB IDs can be reused across phases only when the `(data_format, tile_descriptor)` pair matches. This is because the hardware CB register set includes format information -- reusing a CB ID with a different format requires a reconfiguration step.

```python
# EXISTING -- from blaze/cb_reconfig.py (lines 53-75)
def _allocate_id(
    self,
    data_format: ttnn.DataType,
    tile: ttnn.TileDescriptor | None,
    exclude: set[int],
) -> int:
    """Reuse an existing ID with matching (data_format, tile), or allocate a fresh one."""
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

### The Context System

Each phase of a multi-phase program creates a `Context` that tracks which CB IDs are in use during that phase:

```python
# EXISTING -- from blaze/cb_reconfig.py (lines 77-90)
class Context:
    """Scoped view: within a context every CB ID is unique, across contexts IDs reuse."""

    def __init__(self, manager: "CircularBufferIdManager"):
        self._manager = manager
        self._used_ids: set[int] = set()

    def get_cb_id(self, data_format: ttnn.DataType, tile: ttnn.TileDescriptor | None) -> int:
        cb_id = self._manager._allocate_id(data_format, tile, self._used_ids)
        self._used_ids.add(cb_id)
        return cb_id
```

Within a single context, every CB ID must be unique (multiple CBs cannot share an ID simultaneously). Across contexts, the same ID can be reused if the format matches.

### Example: Multi-Phase MoE Expert

A routed expert in DeepSeek V3 MoE has two phases:

**Phase 1: Gate + Up projection**
```text
CB 0: activation (bfloat16, 1x32)         -- gate input
CB 1: gate_weights (bfloat8_b, 32x32)     -- DRAM streaming buffer
CB 2: gate_output (bfloat16, 1x32)        -- gate result
CB 3: up_weights (bfloat8_b, 32x32)       -- DRAM streaming buffer
CB 4: up_output (bfloat16, 1x32)          -- up result
CB 5: eltwise_output (bfloat16, 1x32)     -- gate * up
```

**Phase 2: Down projection**
```text
CB 0: activation (bfloat16, 1x32)         -- REUSE: same format as Phase 1 CB 0
CB 1: down_weights (bfloat8_b, 32x32)     -- REUSE: same format as Phase 1 CB 1
CB 2: down_output (bfloat16, 1x32)        -- REUSE: same format as Phase 1 CB 2
```

Without the manager: 9 CB IDs (6 + 3 new).
With the manager: 6 CB IDs (3 reused across phases).

### The CBEngine-to-Manager Bridge

The `CBEngine.to_cb_id_manager()` method exports engine assignments to a `CircularBufferIdManager` so downstream code can allocate additional CBs without collisions:

```python
# EXISTING -- from blaze/cb_engine.py (lines 389-420)
def to_cb_id_manager(self, assignments: dict[str, CBAssignment]) -> "CircularBufferIdManager":
    """Export CB assignments to a CircularBufferIdManager for interop."""
    from .cb_reconfig import CircularBufferIdManager

    manager = CircularBufferIdManager()

    for key, a in sorted(assignments.items(), key=lambda x: x[1].cb_id):
        ttnn_dtype = DTYPE_TO_TTNN.get(a.data_format)
        tile_desc = ttnn.TileDescriptor(a.tile_shape[0], a.tile_shape[1], False)
        manager.seed_reserved_cb(a.cb_id, ttnn_dtype, tile_desc)

    return manager
```

The `seed_reserved_cb()` method registers pre-assigned CB IDs so the manager knows which IDs are already in use:

```python
# EXISTING -- from blaze/cb_reconfig.py (lines 38-51)
def seed_reserved_cb(
    self,
    cb_id: int,
    data_format: ttnn.DataType,
    tile: ttnn.TileDescriptor | None,
) -> None:
    """Register a pre-assigned CB id and format (e.g. after CBEngine.assign)."""
    if cb_id not in self._id_to_format:
        self._id_to_format[cb_id] = (data_format, self._normalize_tile(tile))
    self._next_id = max(self._next_id, cb_id + 1)
```

---

## 4. The CBContext for Phase-Level Control

The `CBContext` class in `cb_reconfig.py` provides a named handle for inspecting and controlling CB ID allocation within a specific phase:

```python
# EXISTING -- from blaze/cb_reconfig.py (lines 111-136)
class CBContext:
    """A named phase handle for inspecting and controlling CB ID allocation."""

    def __init__(self, name: str, manager: CircularBufferIdManager):
        self.name = name
        self._manager = manager
        self._ctx = manager.create_context()
        self.used_ids: set[int] = self._ctx._used_ids
        self.descriptors: list = []

    def get_cb_id(self, data_format, tile) -> int:
        """Allocate a CB ID within this context (bypass emit)."""
        return self._ctx.get_cb_id(data_format, tile)

    def pin(self, cb_id: int, data_format, tile):
        """Reserve a specific CB ID with the given format in this context."""
        self.used_ids.add(cb_id)
        td = self._manager._normalize_tile(tile)
        self._manager._id_to_format[cb_id] = (data_format, td)
        self._manager._next_id = max(self._manager._next_id, cb_id + 1)
```

The `pin()` method is the expert escape hatch: a developer can force a specific CB ID for a specific buffer, overriding the automatic allocation. This is used in performance-critical paths where the CB ID must match a hardcoded value in the kernel.

---

## 5. How the ShapeTracker Provides Lifetime Information

In the current codebase, lifetime information comes from the BlazeGraph's topological structure. The `CBEngine` traverses the graph and computes `first_use`/`last_use` from node positions and edge connectivity.

With the proposed TensorAdapter architecture from [Chapter 3](../ch03_tensor_adapter/), the `ShapeTracker` carries additional metadata that can improve lifetime analysis:

### Automatic Lifetime from Op Chain

When ops are composed via `emit()` calls in a `FusedProgram`, the ShapeTracker records the order of operations and which CBs are consumed by each:

```python
# PROPOSED -- ShapeTracker lifetime recording
class ShapeTracker:
    """Propagates ShapeDescriptor through op chains, tracking CB lifetimes."""

    def __init__(self):
        self._step = 0
        self._cb_first_use: dict[int, int] = {}  # cb_id -> first step
        self._cb_last_use: dict[int, int] = {}   # cb_id -> last step

    def record_use(self, cb_id: int) -> None:
        """Record that a CB is used at the current step."""
        if cb_id not in self._cb_first_use:
            self._cb_first_use[cb_id] = self._step
        self._cb_last_use[cb_id] = self._step

    def advance(self) -> None:
        """Advance to the next step (next emit() call)."""
        self._step += 1

    def get_lifetimes(self) -> dict[int, tuple[int, int]]:
        """Return {cb_id: (first_use, last_use)} for all tracked CBs."""
        return {
            cb_id: (self._cb_first_use[cb_id], self._cb_last_use[cb_id])
            for cb_id in self._cb_first_use
        }
```

### Integration with FusedProgram

The ShapeTracker would be called by `FusedProgram` methods that create or consume CBs:

```python
# PROPOSED -- FusedProgram with lifetime tracking
class FusedProgram:
    def __init__(self, ...):
        # ... existing init ...
        self._shape_tracker = ShapeTracker()

    def cb_scratch(self, name, *, num_pages, ...):
        handle = self._allocate_scratch(...)
        self._shape_tracker.record_use(handle.cb_id)
        return handle

    def cb_from_tensor(self, tensor, ...):
        handle = self._allocate_from_tensor(...)
        self._shape_tracker.record_use(handle.cb_id)
        return handle

    def output(self, op_name, cb_id, ...):
        # Record that this CB is consumed by the current op
        if isinstance(cb_id, CBHandle):
            self._shape_tracker.record_use(cb_id.cb_id)
        self._shape_tracker.advance()  # next op
        return ...
```

This automatic recording eliminates the need for the CBEngine to traverse graph edges -- the ShapeTracker provides lifetime information directly from the `emit()` call sequence. This is especially valuable for the FusedProgram composition path (where no BlazeGraph exists) versus the graph API path (where the graph provides the structure).

---

## 6. When Compaction Matters: Approaching the 64-Slot Limit

### Scenario 1: DeepSeek V3 MoE Layer

A full DeepSeek V3 MoE layer fuses the following ops:

```text
Op                          CBs (before compaction)
──────────────────────────────────────────────────
RMSNorm                     3 (input, gamma, output)
Mcast (activation broadcast) 2 (source, destination)
DeepseekMoeGate             4 (input, scores_out, indices_out, scratch)
Mcast (gate results)        2 (source, destination)
RoutedExpert SwigluOp:
  DRAMStreamingMatmul (up)  4 (act, weights, out, index)
  DRAMStreamingMatmul (gate) 3 (weights, out, [shared act])
  EltwiseMul                2 (scalar, out)
  Gather                    2 (src, dst)
  Mcast                     2 (src, dst)
  DRAMStreamingMatmul (down) 3 (act, weights, out)
SharedExpert:
  KNSlicedMatmul            3 (act, weights, out)
  GatedReduce               3 (gate_in, up_in, out)
  Mcast                     2 (src, dst)
  DRAMStreamingMatmul (down) 3 (act, weights, out)
GatedLocalReduce            3 (routed_in, shared_in, out)
AllReduce                   2 (input, output)
ResidualAdd                 3 (input, residual, output)
──────────────────────────────────────────────────
Raw total:                 ~46 CBs
After shared-tensor dedup: ~38 CBs (shared act, shared index)
After compaction:          ~18-22 CBs
```

Without compaction, 38 CBs is under the 64-slot limit but leaves only 26 slots for additional CBs (e.g., for debug tensors, profiling, or future op additions). With compaction, 18-22 CBs provides 42-46 slots of headroom.

### Scenario 2: GLM-5.1 Large MoE with Dynamic Sparse Attention

GLM-5.1 adds DSA (Dynamic Sparse Attention) ops alongside MoE, significantly increasing CB count:

```text
DSA Pipeline:
  DsaIndexer             3 CBs
  DsaTopk                4 CBs
  DsaSparseGather        3 CBs
  DsaKvPrep              3 CBs
  DsaSparseFlashDecode   6 CBs
  ──────────────────────
  DSA subtotal:          19 CBs

Large MoE (>256 experts, two-face routing):
  GlmMoeGateMerge        5 CBs (two-face gate merge)
  GlmLargeRoutedExpert   16 CBs (phantom grid, two-column)
  GlmMoeRouter           4 CBs
  ──────────────────────
  Large MoE subtotal:    25 CBs

Combined DSA + MoE + RMSNorm + Mcast + ResidualAdd:
  Raw total:             ~55-60 CBs

Without compaction: EXCEEDS 64-slot limit!
With compaction:    ~28-35 CBs (safe)
```

This is the scenario where compaction is not optional -- it is required for correctness. Without it, the CBEngine raises:

```python
# EXISTING -- from blaze/cb_engine.py (lines 216-220)
if mgr_context is None and next_cb_id >= self.max_cb_id:
    raise ValueError(
        f"CB ID limit exceeded ({self.max_cb_id}): "
        f"cannot assign CB for external input '{port_spec.name}' on '{node.id}'"
    )
```

### Enabling Compaction

Compaction is enabled via the `compact` flag on CBEngine:

```python
# EXISTING -- from blaze/cb_engine.py (line 154)
def __init__(self, max_cb_id: int = MAX_CB_ID, compact: bool = False, cb_id_manager=None):
    self.max_cb_id = max_cb_id
    self.compact = compact
```

And triggered in the `assign()` method:

```python
# EXISTING -- from blaze/cb_engine.py (lines 384-386)
if self.compact:
    assignments = self.compact_cb_ids(assignments)
```

The proposed automatic system would enable compaction by default (P4 -- provide defaults):

```python
# PROPOSED -- default-on compaction
class AutoCBEngine(CBEngine):
    """CBEngine with automatic compaction and budget awareness."""

    def __init__(self, l1_budget: L1Budget, **kwargs):
        super().__init__(compact=True, **kwargs)  # compaction on by default
        self._budget = l1_budget
```

---

## 7. Compaction and Format Compatibility

A subtlety: the interval coloring algorithm in `compact_cb_ids()` does not consider data format or tile shape. Two CBs with non-overlapping lifetimes but different formats (e.g., bfloat16 and bfloat8_b) will be assigned the same CB ID. This is fine for the graph API path (where CBEngine manages all assignments), but creates a problem for the composition API path (where downstream code expects specific format-to-ID mappings).

The `CircularBufferIdManager` solves this by making reuse format-aware:

```text
Compaction comparison:

compact_cb_ids() (format-agnostic):
  CB A (bfloat16, lifetime 0-1) -> ID 0
  CB B (bfloat8_b, lifetime 2-3) -> ID 0  (REUSE despite format mismatch)
  -> Requires CB reconfiguration at the phase boundary

CircularBufferIdManager (format-aware):
  CB A (bfloat16, lifetime 0-1) -> ID 0
  CB B (bfloat8_b, lifetime 2-3) -> ID 1  (new ID, different format)
  CB C (bfloat16, lifetime 2-3) -> ID 0  (REUSE, matching format)
  -> No reconfiguration needed for C; B gets its own ID
```

The automatic system should use format-aware compaction by default, falling back to format-agnostic compaction only when approaching the 64-slot limit:

```python
# PROPOSED -- hybrid compaction strategy
def compact_with_format_awareness(
    assignments: dict[str, CBAssignment],
    max_cb_id: int = 64,
) -> dict[str, CBAssignment]:
    """Compact CB IDs with format-aware reuse.

    First pass: format-aware (prefer same-format reuse).
    If result exceeds max_cb_id, second pass: format-agnostic.
    """
    # Pass 1: format-aware
    sorted_cbs = sorted(assignments.values(), key=lambda a: a.first_use)
    color_end: dict[tuple[int, str, tuple], int] = {}
    color_count = 0

    for cb in sorted_cbs:
        fmt_key = (cb.data_format, cb.tile_shape)
        # Try to find a free color with matching format
        assigned = False
        for (color, fmt, tile), end in sorted(color_end.items()):
            if (fmt, tile) == fmt_key and end < cb.first_use:
                cb.cb_id = color
                color_end[(color, fmt, tile)] = cb.last_use
                assigned = True
                break

        if not assigned:
            cb.cb_id = color_count
            color_end[(color_count, cb.data_format, cb.tile_shape)] = cb.last_use
            color_count += 1

    if color_count <= max_cb_id:
        return assignments

    # Pass 2: format-agnostic (original compact_cb_ids)
    return CBEngine.compact_cb_ids(None, assignments)
```

---

## 8. The CB Reconfig Tensor

When format-agnostic compaction assigns the same CB ID to buffers with different formats, the hardware must be reconfigured between phases. This is handled by the CB reconfig tensor:

```python
# EXISTING -- from blaze/cb_reconfig.py (lines 162-218)
def build_cb_reconfig_tensor(cb_metadata, full_device_grid, mesh_device):
    """Build a HEIGHT_SHARDED L1 tensor with per-core CB config for reconfig.

    264 uint32 per core: 64 CBs x 4 words [addr, size, num_pages, page_size],
    2 mask words (which CBs are active), 2 sync semaphores, 4 reserved.
    """
    WORDS_PER_CORE = 264
    config = torch.zeros((num_cores, WORDS_PER_CORE), dtype=torch.uint32)

    for cb_id, entries in cb_metadata.items():
        for addr, total_size, num_pages, page_size, core_ranges in entries:
            # ... write per-core config for each CB ...
            base = cb_id * 4
            config[core_idx, base + 0] = addr
            config[core_idx, base + 1] = total_size
            config[core_idx, base + 2] = num_pages
            config[core_idx, base + 3] = page_size
            # Set active bit
            if cb_id < 32:
                config[core_idx, 256] |= 1 << cb_id
            else:
                config[core_idx, 257] |= 1 << (cb_id - 32)
```

The reconfig tensor is a HEIGHT_SHARDED L1 tensor where each core's shard contains the complete CB configuration for all 64 slots. The kernel reads this tensor at the start of each phase to reconfigure its CB registers.

**L1 cost:** 264 * 4 = 1,056 bytes per core. This is accounted for in the L1 budget (File 01).

**Active mask:** Two uint32 words (positions 256 and 257) form a 64-bit bitmask indicating which CB IDs are active. The kernel only reconfigures CBs whose bits are set.

---

## 9. The Unified Compile-Time Pipeline

Bringing together all four files of this chapter, the complete automatic CB sizing pipeline is:

```text
Unified Pipeline: Tensor Shape -> CB Configuration

Step 1: ShapeDescriptor (Chapter 4)
  PyTorch tensor [1, 7168] -> padded [32, 7168] -> tile_grid [1, 224]

Step 2: Format Selection (Chapter 6)
  Precision profile -> bfloat16 activations, bfloat8_b weights

Step 3: Page Size (File 02)
  tile.get_tile_size(bfloat16) -> 2,048 B
  tile.get_tile_size(bfloat8_b) -> 1,088 B

Step 4: Blocking Strategy (File 03)
  AutoBlocker.select_matmul_blocking() -> RT=1, CT=48, KT=56

Step 5: Page Count (File 02)
  AutoSizer.solve():
    activation_cb: 224 pages (K tiles)
    weight_cb: 56 * 3 pages (subblock * triple-buffer)
    output_cb: 48 pages (N per core)

Step 6: L1 Budget Check (File 01)
  L1Budget.allocate() for each CB
  Total: 14 KB + 183 KB + 3 KB = 200 KB (15% of available)

Step 7: Compaction (File 04)
  compact_cb_ids() reduces ID count
  CircularBufferIdManager provides format-aware reuse

Step 8: Double-Buffering Optimization (File 02)
  Remaining budget distributed to high-priority CBs

Step 9: Validation (File 01)
  Assert total <= available
  Assert CB count <= 64
  Emit actionable error if violated (P6)

Step 10: Feedback (if budget exceeded)
  Adjust format (Ch6) or core grid (Ch4) before failing to developer
```

### Integration Point: FusedProgram.build()

The pipeline executes during `FusedProgram.build()`, after all `emit()` calls have been made but before the `ProgramDescriptor` is assembled:

```python
# PROPOSED -- FusedProgram integration
class FusedProgram:
    def build(self):
        if self._auto_size:
            # Run the unified pipeline
            lifetimes = self._shape_tracker.get_lifetimes()
            blocking = self._auto_blocker.select_blocking(...)
            sizing = self._auto_sizer.solve()
            compacted = compact_with_format_awareness(...)

            # Apply results to CB descriptors
            self._apply_sizing(sizing)
            self._apply_compaction(compacted)

            # Validate
            self._validate_l1_budget()
            self._validate_cb_count()

        # ... existing build logic ...
        return self.program.build()
```

### Error Reporting Example

When the pipeline detects an L1 budget violation:

```text
L1BudgetExceeded: Fused op 'moe_layer_3' exceeds L1 budget.

Budget:
  Total L1:          1,500 KB
  Reserved:            128 KB
  Kernel text:          80 KB
  Available:         1,292 KB

Allocations (by size):
  routed_expert__up_proj__weights_cb:     183 KB (56 * 3 * 1,088 B)
  routed_expert__gate_proj__weights_cb:   183 KB (56 * 3 * 1,088 B)
  routed_expert__down_proj__weights_cb:   183 KB (56 * 3 * 1,088 B)
  shared_expert__gu__weights (DA):          0 KB (DIRECT_ADDRESS)
  activation_cb:                           14 KB (224 * 64 B)
  rmsnorm.input:                           14 KB
  rmsnorm.gamma:                           14 KB
  rmsnorm.out_cb:                          14 KB
  mcast.dst:                               14 KB
  ... (12 more CBs) ...
  ──────────────────────────────────────────
  Total:                                1,384 KB
  Over budget by:                          92 KB

Suggestion: Reduce routed_expert subblock_k from 56 to 28
  Savings: 3 * (56-28) * 3 * 1,088 = 274 KB
  New total: 1,110 KB (86% utilization)

Or: Switch routed expert weights to bfloat4_b (currently bfloat8_b)
  Savings: 3 * 56 * 3 * (1,088 - 576) = 258 KB
  New total: 1,126 KB (87% utilization)
```

This error message (P6) gives the developer:
1. The exact CB that caused the overflow
2. The current allocation breakdown
3. Two concrete suggestions with estimated savings

---

## 10. Compaction Performance Impact

Compaction is a compile-time operation with zero runtime cost. The compacted program produces identical results to the uncompacted version -- the data flows and computations are unchanged. Only the CB ID assignment differs.

However, compaction affects debuggability:

- **Without compaction:** Each CB has a unique ID that maps 1:1 to its logical purpose. `l1_profile.py` output is easy to interpret.
- **With compaction:** Multiple logical CBs share IDs. `l1_profile.py` shows the same ID for different phases, which can be confusing.

The `extract_cb_names()` function in `l1_profile.py` mitigates this by mapping CB IDs to logical names from the shadow graph. With compaction, a CB ID might map to multiple names across phases:

```text
CB 0: "rmsnorm.input" (phase 0-1), "matmul.act" (phase 2-4), "gather.out" (phase 5-6)
```

The proposed system would annotate the profiling output with phase boundaries to maintain clarity:

```text
# PROPOSED -- phase-annotated profiling
cb[0] phase 0-1: rmsnorm.input (bfloat16, 14 KB)
cb[0] phase 2-4: matmul.act (bfloat16, 14 KB)   [reused: same format]
cb[0] phase 5-6: gather.out (bfloat16, 3 KB)     [reused: same format, smaller]
```

---

## Key Takeaways

- **CB compaction via interval coloring** (`compact_cb_ids()`) reduces CB ID usage by assigning the same ID to buffers with non-overlapping lifetimes. The greedy algorithm is provably optimal for interval graphs -- no algorithm can use fewer colors. For a typical MoE layer, this reduces 38 CBs to 18-22 -- a 40-50% reduction that keeps the op well within the 64-slot hardware limit.
- **Lifetime tracking** via `CBAssignment.first_use` and `last_use` (computed from topological position) is the input to the compaction algorithm. The proposed ShapeTracker automates lifetime recording from the `emit()` call sequence, eliminating the need for graph traversal in the composition API path.
- The **`CircularBufferIdManager`** provides format-aware CB ID reuse across multi-phase programs. Its `Context` system ensures within-phase uniqueness while enabling cross-phase reuse when `(data_format, tile_descriptor)` matches.
- **Format-agnostic compaction** (sharing IDs across formats) is necessary when approaching the 64-slot limit but requires CB reconfiguration via the reconfig tensor (1,056 bytes per core of L1 overhead). The automatic system should prefer format-aware compaction and fall back to format-agnostic only when necessary.
- For **GLM-5.1 large MoE with DSA**, the raw CB count (55-60) exceeds the 64-slot limit. Compaction is not optional -- it is required for correctness. The automatic system should enable compaction by default (P4).
- The **unified compile-time pipeline** (shape -> format -> page size -> blocking -> page count -> budget check -> compaction -> validation -> feedback) executes entirely at compile time with zero runtime overhead. When budget is exceeded, the feedback loop adjusts format ([Chapter 6](../ch06_data_formats/)) or core grid ([Chapter 4](../ch04_tile_decomposition/)) before failing to the developer.

## Source Files

- `blaze/cb_engine.py` -- `compact_cb_ids()` (lines 422-437), `CBAssignment.first_use` / `last_use` (lines 137-138), lifetime computation from topo order (lines 358-374), `to_cb_id_manager()` (lines 389-420), `MAX_CB_ID = 64` (line 65), `CBEngine.__init__(compact=)` (line 154)
- `blaze/cb_reconfig.py` -- `CircularBufferIdManager` (lines 18-90), `_allocate_id()` format-aware reuse (lines 53-75), `Context` scoped uniqueness (lines 77-90), `CBContext` named phase handle (lines 111-136), `build_cb_reconfig_tensor()` (lines 162-218), `WORDS_PER_CORE = 264` (line 175)
- `blaze/l1_profile.py` -- `extract_cb_names()` (line 482), `_build_overlap_groups()` (lines 410-479)
- `blaze/fused_program.py` -- `FusedProgram.cb_scratch()` (lines 1540-1593), `_format_key()` disjoint-share optimization (lines 59-78)
- `blaze/ops/swiglu/op.py` -- SwigluOp.emit() 6-stage CB chain (lines 69-225)
- `blaze/ops/dense_mlp/op.py` -- DenseMLP composition with 12 micro-ops (lines 28-80)

---

< Previous: [File 03 -- Blocking Strategy Selection](./03_blocking_strategy_selection.md) | Next: [Chapter 8 -- Broadcasting and Multi-Dimensional Shape Alignment](../ch08_broadcasting/) >

← [Chapter 6 -- Data Format Selection and Automatic Negotiation](../ch06_data_formats/) | [Chapter 8 -- Broadcasting and Multi-Dimensional Shape Alignment](../ch08_broadcasting/) →
