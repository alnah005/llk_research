# 01 -- CBHandle Abstraction

**Source file:** `blaze/cb_handle.py`

CBHandle is the unit of inter-op data flow in TT-Blaze's composition API. Every `emit()` call that produces data returns a CBHandle, and every `emit()` call that consumes data accepts one. The dataclass wraps a circular buffer ID together with enough metadata for downstream ops to derive page counts, byte addresses, and tile geometry without re-inspecting the underlying tensor.

## What a CBHandle Represents on Silicon

A Tensix core on Blackhole has 64 hardware circular buffer slots (indices 0--63). Each slot is configured by registers that specify a base L1 address, total size, page size, number of pages, and data format. CBHandle is the Python-side mirror of this hardware state -- its dataclass fields map to the register values that the runtime programs into each core's CB configuration:

| CBHandle field | Hardware concept |
|---|---|
| `cb_id` | CB slot index (0--63). Selects which hardware register bank is programmed. |
| `page_size` | Bytes per tile in L1. Determined by tile geometry and data format. |
| `num_pages` | Number of pages the CB can hold before wrapping. Controls FIFO depth. |
| `core_ranges` | Which physical Tensix cores have this CB configured. |
| `data_format` | The `ttnn.DataType` that determines how tile bytes are interpreted by the unpacker. |
| `tile_desc` | A `ttnn.TileDescriptor` encoding tile height and width (e.g., 32x32). |
| `byte_offset` | Offset within a backing tensor's L1 allocation for overlapped views. |
| `backing_tensor` | The `ttnn.Tensor` whose L1 buffer backs this CB. `None` for pure scratch. |

The remaining fields (`access_mode`, `_node_id`, `_port_name`, `scratch_name`) are Python-level concerns that do not map to hardware registers -- they control consumption semantics, shadow graph recording, and scratch-mapping lookup.

## CBAccessMode Enum

```python
class CBAccessMode(str, Enum):
    FIFO = "fifo"
    DIRECT_ADDRESS = "direct_address"

    def __str__(self) -> str:
        return self.value
```

CBAccessMode inherits from both `str` and `Enum`. The custom `__str__` ensures that `str(CBAccessMode.FIFO)` returns `"fifo"` rather than `"CBAccessMode.FIFO"`, which matters for debug printing and serialized graph annotations.

**FIFO mode** is the default and the common case. The kernel uses `cb_wait_front()` / `cb_pop_front()` / `cb_push_back()`. The CB hardware tracks read and write pointers. Downstream ops call `int(handle)` to get the CB index and issue standard FIFO operations.

**DIRECT_ADDRESS mode** bypasses FIFO semantics entirely. The kernel reads from an explicit L1 address obtained via `handle.buffer_address()`. This mode is used by `cb_from_shared_l1_views()` (section 03) where multiple logical tensors share one physical CB descriptor and each consumer reads its own sub-region by absolute address. Since FIFO state (read/write pointers) is per-CB-slot and shared across all consumers on a core, multiple consumers cannot independently advance through the same CB -- direct addressing sidesteps this. Direct-address handles are excluded from FIFO-reuse caches (`_format_key_to_cb`, `_view_tensor_cbs`) and from output wiring (`wire_output` and `wire_output_from_graph` both reject them, and `cb_for_tensor_or_none` skips them in the `_io_tensors` scan).

## require_fifo_handle()

The module-level guard enforces that a handle supports standard FIFO consumption:

```python
def require_fifo_handle(handle: "CBHandle", op_name: str) -> "CBHandle":
    if handle.is_direct_address:
        raise ValueError(
            f"{op_name} requires a FIFO CBHandle, but received a "
            "direct-address handle. Use an op that explicitly reads "
            "handle.buffer_address()."
        )
    return handle
```

This is a composition-time firewall. It catches misuse before the kernel compiles, not after the hardware hangs. FusedProgram's own `cb_from_tensor()` uses it when a CBHandle is passed directly instead of a tensor:

```python
def cb_from_tensor(self, tensor, **kwargs) -> CBHandle:
    if isinstance(tensor, CBHandle):
        return require_fifo_handle(tensor, "FusedProgram.cb_from_tensor")
    ...
```

The function returns the validated handle, enabling an inline guard pattern: `handle = require_fifo_handle(handle, "MyOp")`.

## The CBHandle Dataclass -- Every Field

The `@dataclass(eq=False)` decorator deliberately disables value-based equality. Two CBHandles with the same `cb_id` should not compare equal if they represent different views (e.g., after disjoint-cores sharing gives both callers handles with the same `cb_id` but different `core_ranges`). Identity semantics prevent accidental deduplication in sets or dicts keyed by handle value.

```python
@dataclass(eq=False)
class CBHandle:
    cb_id: int
    num_pages: int
    page_size: int
    core_ranges: ttnn.CoreRangeSet
    data_format: object
    tile_desc: object
    byte_offset: int = 0
    backing_tensor: object = None
    access_mode: CBAccessMode = CBAccessMode.FIFO
    _node_id: str = ""
    _port_name: str = "out"
    scratch_name: str = ""
```

### Primary fields

| Field | Type | Purpose |
|---|---|---|
| `cb_id` | `int` | Index into the hardware CB register file (0--63). Allocated by BlazeProgram's monotonic counter or by `CircularBufferIdManager` in multi-phase programs. |
| `num_pages` | `int` | Number of pages (tiles or rows) the CB holds. Derived from the tensor's shard shape divided by tile geometry, or from explicit `total_size / page_size`. Determines double-buffering depth. |
| `page_size` | `int` | Byte size of one page. For tiled tensors, `tile.get_tile_size(dtype)`. For row-major tensors, `num_elements_per_row * element_bytes`. |
| `core_ranges` | `ttnn.CoreRangeSet` | Which cores this CB is bound to. Comes from the tensor's `shard_spec.grid` or an explicit override. Used by disjoint-cores sharing to decide whether two CBs can fold onto one ID. |
| `data_format` | `object` | A `ttnn.DataType` (e.g. `ttnn.bfloat16`). Typed as `object` to avoid import cycles, but always holds a DataType at runtime. |
| `tile_desc` | `object` | A `ttnn.TileDescriptor` or `None` for row-major CBs. Normalized by `BlazeProgram._normalize_tile()` -- raw `ttnn.Tile` objects are converted on entry. |

### Overlapped view / backing fields

| Field | Default | Purpose |
|---|---|---|
| `byte_offset` | `0` | Byte offset within the backing L1 buffer. Non-zero for overlapped views where multiple CBs share one fused tensor but address different sub-regions. |
| `backing_tensor` | `None` | The `ttnn.Tensor` providing L1 storage. Set for tensor-backed and scratch-mapped CBs. Pure scratch CBs leave this `None` until the materialize pass in `cb_reconfig_builder.py` packs them into an arena tensor. |
| `access_mode` | `CBAccessMode.FIFO` | Controls whether the CB uses standard wait/pop semantics or direct-address reads. Validated in `__post_init__`. |

### Shadow graph metadata

| Field | Default | Purpose |
|---|---|---|
| `_node_id` | `""` | ID of the shadow graph `OpNode` that produced this handle. Set by `FusedProgram._record_op()` after `output()` or `multi_output()`. |
| `_port_name` | `"out"` | Output port name on the producing node. Single-output ops get the first declared port name from the spec (falling back to `"out"`); multi-output ops set each handle to its declared port name (e.g. `"output"`, `"output_indices"`). |

These fields are prefixed with `_` because they are internal bookkeeping, not part of the CB's hardware contract. They enable `FusedProgram._wire_input_edges()` to connect the producer node to the consumer node in the shadow graph.

### Scratch metadata

| Field | Default | Purpose |
|---|---|---|
| `scratch_name` | `""` | The symbolic name passed to `cb_scratch(name=...)`. Used by the `scratch_mapping` lookup path to correlate a CB with its pre-allocated backing tensor. When `cb_scratch()` goes through the mapping path (`_cb_scratch_from_mapping`), the mapped handle gets `scratch_name` set explicitly so the name survives even though the underlying CB was allocated by `cb_from_tensor_overlapped()`. |

## __post_init__ Invariants

```python
def __post_init__(self) -> None:
    if not isinstance(self.access_mode, CBAccessMode):
        self.access_mode = CBAccessMode(self.access_mode)
    if self.is_direct_address and self.backing_tensor is None:
        raise ValueError("Direct-address CBHandle requires a backing_tensor.")
```

Two things happen at construction time:

1. **access_mode normalization**: If a raw string like `"fifo"` is passed instead of `CBAccessMode.FIFO`, it is coerced to the enum. This prevents mismatches when constructing handles from deserialized data.

2. **Direct-address requires a backing tensor**: A direct-address CB must have a `backing_tensor` because the kernel reads from `buffer_address()`. Without it, the handle would return `None` from `buffer_address()` and the kernel would read from address 0 -- certain L1 corruption.

## Integer Coercion: __int__ and __index__

```python
def __int__(self) -> int:
    return self.cb_id

def __index__(self) -> int:
    return self.cb_id
```

`__int__` means `int(handle)` returns the CB ID, so handles can be passed directly into CT-arg tuples that expect integer values:

```python
f.unified_ct_args([("in_cb", in_handle), ("out_cb", out_handle)])
```

FusedProgram's `_capture_ct_values` coerces all values via `int(v)`, so the CBHandle's `__int__` turns it into the CB ID transparently.

`__index__` means handles work in any context Python requires an integer index -- list subscripts, bitwise operations, `hex()`, or `bin()`. This prevents `TypeError` failures when handles appear in computed expressions.

**Why both exist**: `__int__` handles explicit `int(handle)` calls. `__index__` handles implicit integer contexts: indexing, slicing, and C-extension APIs that call `PyNumber_Index`. Together they guarantee that any code path that previously accepted an integer CB ID continues to work unmodified when given a CBHandle. This was critical for incremental migration -- the composition API could adopt CBHandle as its return type while underlying BlazeProgram methods (which accept integer CB IDs) required no changes.

**Important subtlety**: passing `int(handle)` instead of the CBHandle object to CT-arg methods bypasses FusedProgram's position tracking, triggering a fallback remap path at build time. Always pass the CBHandle object directly. See [Section 02 -- FusedProgram, CT-Arg and RT-Arg Declaration](./02_fused_program.md) for the full `_track_cb_arg_positions` mechanism.

## Convenience Properties

```python
@property
def is_fifo(self) -> bool:
    return self.access_mode is CBAccessMode.FIFO

@property
def is_direct_address(self) -> bool:
    return self.access_mode is CBAccessMode.DIRECT_ADDRESS
```

These use `is` identity comparison against the enum singleton, not `==`, which is both faster and semantically correct (there is exactly one `CBAccessMode.FIFO` object in the process).

## buffer_address()

```python
def buffer_address(self) -> int | None:
    if self.backing_tensor is not None:
        return self.backing_tensor.buffer_address() + self.byte_offset
    return None
```

This mirrors `ttnn.Tensor.buffer_address()` but adds the handle's `byte_offset`. Three cases arise:

1. **Tensor-backed CB** (`backing_tensor` set, `byte_offset == 0`): Returns the tensor's L1 base address. The common path for `cb_from_tensor()`.
2. **Overlapped view** (`backing_tensor` set, `byte_offset > 0`): Returns the base address plus the sub-region offset. This is the primary use case for direct-address handles, where the kernel needs the exact L1 location rather than a CB index.
3. **Pure scratch CB** (`backing_tensor is None`): Returns `None`. The kernel must use `get_write_ptr` at runtime to discover the address. After the `_materialize_scratch_cbs` pass packs scratch CBs into an arena tensor, this starts returning a real address.

Direct-address consumers rely on this method to compute the L1 address they read from, since they bypass the FIFO `cb_wait_front` mechanism entirely.

## Worked Example: Matmul Output Flowing into Gather Input

Consider a fused decode pipeline where a matmul produces an activation tensor that a subsequent gather op consumes. Here is how the CBHandle chain works, step by step.

### Step 1: Matmul Allocates an Output CB

The matmul op's `emit()` calls `f.cb_scratch()` for an intermediate output buffer:

```python
matmul_out = f.cb_scratch(
    name="matmul___dst",
    num_pages=4,
    core_ranges=f.matmul_cores,
    data_format=ttnn.bfloat16,
    tile=tile_desc,
    page_size=2048,
)
```

Inside `FusedProgram.cb_scratch()`, the framework:
1. Asks `BlazeProgram` for the next `cb_id` (say, `cb_id=3`).
2. Creates a `CBDescriptor` with `total_size = 4 * 2048 = 8192` bytes.
3. Constructs a CBHandle with all the geometry fields filled in.
4. Records it in `_cb_metadata[3]` for later lookup.
5. Calls `_track_cb_allocation(3, tensor_backed=False)` to register the CB's lifetime phase.

The matmul op then calls `f.output("matmul", matmul_out, ...)`, which stamps `_node_id = "matmul_1"` and `_port_name = "out"` onto the handle.

### Step 2: The Handle Flows to Gather

The composition function passes `matmul_out` to the gather op:

```python
gather_result = Gather.emit(f, matmul_out, weight_tensor, prefix="gather")
```

### Step 3: Gather Reads the Handle's Metadata

Inside `Gather.emit()`, the gather op calls:

```python
f.unified_ct_args([
    (f"{prefix}.in_cb", matmul_out),   # integer coercion: int(matmul_out) -> 3
    (f"{prefix}.num_pages", matmul_out.num_pages),
    ...
])
```

When `matmul_out` appears in the CT arg tuple, `_track_cb_arg_positions` detects it is a CBHandle and records its position for later CB ID remapping. The integer value `3` is what the kernel sees at compile time. But the gather op also reads `matmul_out.page_size`, `matmul_out.data_format`, and `matmul_out.core_ranges` to configure its own CB descriptors and tile arithmetic.

### Step 4: Shadow Graph Records the Edge

When the gather op calls `f.output("gather", gather_out, in_act=matmul_out)`, the `_record_op` method:

1. Creates an `OpNode` for `gather_1`.
2. Sees that `matmul_out._node_id` is `"matmul_1"`.
3. Creates an `Edge(producer=matmul_1, producer_port="out", consumer=gather_1, consumer_port="in_act")`.
4. Stamps `gather_out._node_id = "gather_1"` for the next op in the chain.

### Step 5: Build-Time CB ID Remap

At `build()` time, the scratch-CB compaction pass (`_compact_scratch_cbs`) may decide that `matmul_out`'s CB ID 3 can share a slot with another non-overlapping scratch CB. If it remaps `3 -> 1`, the remap propagates to:

- The CT arg tuples (via `_cb_arg_positions` recorded in step 3).
- The CB descriptors.
- The `_cb_metadata` dict.

Because step 3 recorded the arg positions, the remap finds the exact indices in the per-RISC arg lists where `3` appears and rewrites them to `1`. The handle's `cb_id` field is also updated directly.

### The Chain Diagram

```
                    CBHandle (cb_id=3)
                    num_pages=4, page_size=2048
                    data_format=bfloat16
  Matmul.emit() --> _node_id="matmul_1"      --> Gather.emit()
                    _port_name="out"               |
                                                   | int(handle) -> 3
                                                   | handle.num_pages -> 4
                                                   | handle.page_size -> 2048
                                                   v
                                              CT args registered
                                              CB arg position tracked
                                              Shadow graph edge wired
                                                   |
                                                   v  (build time)
                                              _compact_scratch_cbs
                                              remap {3 -> 1}
                                              arg position [ncrisc, idx=7]
                                              rewritten 3 -> 1
```

The key insight is that the CBHandle is never just an integer. It carries the metadata the consumer needs (page geometry, data format, core ranges) and the graph identity the framework needs (node ID, port name) for validation and remapping. The integer coercion is a convenience that makes it compatible with low-level APIs, but the object is the primary interface.

## Shadow Graph Recording

Every CBHandle produced by `output()` or `multi_output()` gets stamped with graph metadata:

```python
# Single output (in _record_op)
output_handles._node_id = node_id
output_handles._port_name = (
    node.spec.output_ports[0].name if node.spec.output_ports else "out"
)

# Multi output (in _record_op)
for port_name, handle in output_handles.items():
    handle._node_id = node_id
    handle._port_name = port_name
```

When a downstream op passes this handle as a keyword argument to `output()`, `_separate_port_sources` classifies it as a graph input (it is a CBHandle with a non-empty `_node_id`), and `_wire_input_edges` creates an `Edge`:

```python
if isinstance(source, CBHandle) and source._node_id:
    producer = nmap[source._node_id]
    edge = Edge(
        producer=producer,
        producer_port=source._port_name,
        consumer=node,
        consumer_port=port_name,
    )
    self._shadow_graph.add_edge(edge)
```

The invariant is: a CBHandle with `_node_id != ""` is graph-connected. A CBHandle with `_node_id == ""` represents an external input (a tensor not produced by any op in the current FusedProgram). This distinction determines whether the shadow graph wires an internal edge or records an external input port.

`cb_alias` inherits the source's graph identity so that alias consumers trace back to the original producer:

```python
handle._node_id = source._node_id
handle._port_name = source._port_name
```

Without this, an alias handle would appear as an orphan input in the shadow graph -- a consumer with no edge from its actual data source.

## Lifetime Tracking

Every time a CBHandle is consumed -- either by passing it as a CT arg or by referencing its `cb_id` in an allocation -- `_mark_cb_use` updates the CB's lifetime:

```python
def _mark_cb_use(self, cb_id: int):
    if cb_id in self._cb_lifetimes:
        first, _ = self._cb_lifetimes[cb_id]
        self._cb_lifetimes[cb_id] = (first, self._op_index)
    else:
        self._cb_lifetimes[cb_id] = (self._op_index, self._op_index)
```

This `(first_use, last_use)` pair drives scratch-CB compaction in `cb_reconfig_builder.py`. Two scratch CBs with non-overlapping lifetimes and matching formats can share a single CB ID, reducing the 64-slot pressure.

## Edge Cases and Less Obvious Behaviors

**Mutable after construction**: CBHandle is not frozen. `set_cb_core_ranges()` mutates `handle.core_ranges` in place. `_record_op` mutates `_node_id` and `_port_name`. The `_remap_cb_ids` pass in `cb_reconfig_builder.py` mutates `handle.cb_id` directly on every handle in `_cb_metadata`. Treating handles as mutable references rather than immutable values is intentional -- it lets the build pipeline patch handles globally without rebuilding the entire graph.

**_tensor_handle_cache vs _cb_metadata**: When disjoint-cores sharing folds two callers onto the same `cb_id`, `_cb_metadata[cb_id]` keeps the first caller's geometry. The second caller's handle is stored in `_tensor_handle_cache[id(tensor)]` so that `cb_for_tensor()` returns the correct per-caller metadata. This separation prevents the second caller from silently inheriting the first caller's `core_ranges` or `num_pages`.

**Direct-address handles and cb_for_tensor**: The `cb_for_tensor_or_none` method explicitly skips CB IDs in `_direct_address_cbs` when scanning `_io_tensors`. This prevents a direct-address CB from being returned when a caller asks "which FIFO CB owns this tensor?" -- the two allocation models live in separate namespaces.

**scratch_name persistence through mapping**: When `cb_scratch()` goes through the scratch-mapping path (`_cb_scratch_from_mapping`), the mapped handle gets `mapped_handle.scratch_name = name` set explicitly. This preserves the name for diagnostic purposes even though the actual CB was allocated by `cb_from_tensor_overlapped()`.

### Why CBHandle exists (design rationale)

If CBHandle were replaced by plain integer CB IDs:

1. **Access-mode safety disappears.** Ops that require FIFO semantics would silently accept direct-address CBs, leading to incorrect kernel reads.
2. **Shadow-graph edges cannot be wired.** The graph recorder would need a separate `cb_id -> (node_id, port_name)` side-table, adding complexity and requiring every op to register its output separately from its return value.
3. **Overlapped views lose their L1 address.** The `byte_offset + backing_tensor` pattern would need to be tracked externally, with every consumer re-deriving the address.
4. **CB-ID remapping becomes fragile.** The multi-phase reconfig builder uses `_cb_arg_positions` to track which CT-arg slots hold CBHandle references. Without CBHandle, the tracker would need to inspect every integer argument against a known set of CB IDs -- the fallback path that already exists in `_remap_cb_ids` and emits a warning when hit.

---

<- [Index](index.md) | [02 -- FusedProgram](02_fused_program.md) ->
