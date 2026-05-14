# CBHandle Data Flow

## Overview

The `CBHandle` is the universal currency of inter-op data flow in TT-Blaze. Every `emit()` returns a `CBHandle`, and every `emit()` that consumes upstream data receives a `CBHandle`. This section covers the CBHandle dataclass in detail: its fields, access modes, the chaining pattern that connects ops, `MultiOutput` for multi-output ops, and metadata propagation through the shadow graph.

## CBHandle Dataclass

The full definition from `blaze/cb_handle.py`:

```python
@dataclass(eq=False)
class CBHandle:
    """A reference to a CB, carrying metadata for downstream ops."""

    cb_id: int
    num_pages: int
    page_size: int
    core_ranges: ttnn.CoreRangeSet
    data_format: object
    tile_desc: object
    byte_offset: int = 0
    backing_tensor: object = None
    access_mode: CBAccessMode = CBAccessMode.FIFO
    # Graph metadata (for shadow graph recording)
    _node_id: str = ""
    _port_name: str = "out"
    # Scratch CB name (set by cb_scratch for scratch_mapping lookup)
    scratch_name: str = ""
```

### Core Fields

**`cb_id: int`** -- The hardware circular buffer index (0-63). This is the value that appears in CT arg tuples when the handle is coerced to int via `CBHandle.__int__()`. Two handles with the same `cb_id` point to the same hardware CB.

**`num_pages: int`** -- Number of tile-sized pages this CB holds. For a sharded tensor, this is derived from `shard_shape / tile_shape`. For a scratch CB, this is the value passed to `cb_scratch()`.

**`page_size: int`** -- Size of one page in bytes. Derived from `tile.get_tile_size(dtype)` for tiled tensors, or `num_row_elements * element_size_bytes` for row-major tensors.

**`core_ranges: ttnn.CoreRangeSet`** -- Which cores this CB exists on. Used for per-core CT arg scoping and disjoint-core sharing decisions.

**`data_format: object`** -- The ttnn data type (e.g., `ttnn.bfloat16`, `ttnn.float32`). Passed through to CB descriptors.

**`tile_desc: object`** -- The tile descriptor (ttnn.TileDescriptor). Captures tile geometry (height, width). `None` for row-major tensors.

### Extended Fields

**`byte_offset: int = 0`** -- Byte offset within the backing L1 buffer. Non-zero for overlapped views, where multiple logical tensors share one contiguous L1 allocation. Combined with `backing_tensor.buffer_address()` to compute the absolute L1 read address.

**`backing_tensor: object = None`** -- The ttnn.Tensor providing L1 memory for this CB. `None` for pure scratch CBs (their L1 address is only known at kernel runtime via `get_write_ptr`). When set, `handle.buffer_address()` returns a valid L1 address.

**`access_mode: CBAccessMode = CBAccessMode.FIFO`** -- How consumers may read this handle. See the FIFO vs. Direct-Address section below.

### Graph Metadata Fields

**`_node_id: str = ""`** -- Set by `f.output()` to the shadow graph node ID of the op that produced this handle (e.g., `"rmsnorm_1"`, `"mcast_2"`). Used by downstream `f.output()` calls to wire graph edges.

**`_port_name: str = "out"`** -- The output port name on the producer node. Set to the first `Output()` descriptor's name from the op's spec (e.g., `"output"` for RMSNorm, `"dst"` for Mcast), or `"out"` if no output port is defined.

**`scratch_name: str = ""`** -- Set by `cb_scratch()` to the scratch CB's name. Used by the `scratch_mapping` system for tensor-backed scratch redirection.

### `eq=False`

The `@dataclass(eq=False)` decorator means CBHandles are compared by identity (`is`), not by value. Two handles with the same `cb_id` are not considered equal unless they are the same object. This prevents accidental conflation of handles that share a CB ID via disjoint-core sharing.

## Integer Coercion: `__int__()` and `__index__()`

CBHandle supports transparent integer coercion via two dunder methods:

```python
def __int__(self) -> int:
    return self.cb_id

def __index__(self) -> int:
    return self.cb_id
```

This means a CBHandle can be used directly anywhere a CB ID integer is expected:

```python
# These are equivalent:
(f"{prefix}.input", inp.cb_id)
(f"{prefix}.input", inp)         # __int__() called automatically
```

The coercion is invoked by `_capture_ct_values()`:

```python
def _capture_ct_values(self, args):
    self._pending_ct_values.update({k: int(v) for k, v in args})
```

The `__index__()` method additionally supports using CBHandle in array indexing and `hex()` calls.

## `buffer_address()`: L1 Address Resolution

```python
def buffer_address(self) -> int | None:
    """Return the L1 address where this CB's data actually lives."""
    if self.backing_tensor is not None:
        return self.backing_tensor.buffer_address() + self.byte_offset
    return None
```

Returns:
- For tensor-backed CBs: the L1 address of the data (`tensor.buffer_address() + byte_offset`).
- For pure scratch CBs: `None`. The kernel must use `get_write_ptr` at runtime.

This method is used by ops that need raw L1 addresses rather than CB IDs -- for example, direct-address consumers that override the CB read pointer.

## FIFO vs. Direct-Address Access Modes

```python
class CBAccessMode(str, Enum):
    FIFO = "fifo"
    DIRECT_ADDRESS = "direct_address"
```

### FIFO Mode (Default)

The standard CB consumption model. The kernel waits for data via `cb_wait_front()`, reads tiles from the CB front, and releases them via `cb_pop_front()`. The CB runtime manages the read/write pointers automatically.

Most CBHandles are FIFO. All of `cb_from_tensor()`, `cb_scratch()`, `cb_alias()`, and `cb_from_view()` produce FIFO handles by default.

```python
@property
def is_fifo(self) -> bool:
    return self.access_mode is CBAccessMode.FIFO
```

### Direct-Address Mode

The kernel reads data from a fixed L1 address rather than the CB front. Used when multiple logical sub-tensors share one physical CB (via `cb_from_shared_l1_views()`). Each handle carries a different `byte_offset`, and the kernel computes the read address from `handle.buffer_address()`.

Direct-address CBHandles require a `backing_tensor` -- the `__post_init__` validates this:

```python
def __post_init__(self) -> None:
    if not isinstance(self.access_mode, CBAccessMode):
        self.access_mode = CBAccessMode(self.access_mode)
    if self.is_direct_address and self.backing_tensor is None:
        raise ValueError("Direct-address CBHandle requires a backing_tensor.")
```

```python
@property
def is_direct_address(self) -> bool:
    return self.access_mode is CBAccessMode.DIRECT_ADDRESS
```

### `require_fifo_handle()`: Guarding Against Accidental Direct-Address Consumption

Many ops assume FIFO semantics -- they call `cb_wait_front()` and `cb_pop_front()` on their input. Passing a direct-address handle to these ops would silently produce wrong behavior (reading from a stale CB front position instead of the intended L1 address).

The guard function (from `blaze/cb_handle.py`):

```python
def require_fifo_handle(handle: "CBHandle", op_name: str) -> "CBHandle":
    """Return *handle* if it supports ordinary FIFO consumption."""
    if handle.is_direct_address:
        raise ValueError(
            f"{op_name} requires a FIFO CBHandle, but received a "
            "direct-address handle. Use an op that explicitly reads "
            "handle.buffer_address()."
        )
    return handle
```

This is called internally by `cb_from_tensor()` when it receives a CBHandle:

```python
def cb_from_tensor(self, tensor, **kwargs) -> CBHandle:
    if isinstance(tensor, CBHandle):
        return require_fifo_handle(tensor, "FusedProgram.cb_from_tensor")
    ...
```

And in `wire_output()`:

```python
def wire_output(self, handle: CBHandle, output_tensor) -> None:
    if handle.is_direct_address:
        raise ValueError("wire_output cannot bind a direct-address CBHandle.")
```

The rule: **ops that use standard CB wait/pop must guard against direct-address handles**. Direct-address handles require explicit L1 address reads.

## Handle Chaining: Zero-Coupling Between Ops

The CBHandle chain is the core composition mechanism. Each op's `emit()` returns a handle, and downstream ops receive it as input. The ops never know about each other -- they only see CBHandles.

### Example: Mcast -> RMSNorm -> Copy

```python
# Step 1: Mcast receives a tensor, returns a handle
act = Mcast.emit(f, input_tensor, prefix="act_mcast")
# act.cb_id = 3 (assigned by Mcast's cb_scratch)
# act._node_id = "mcast_1" (set by f.output)
# act._port_name = "dst"

# Step 2: RMSNorm receives act handle as input
result = RMSNorm.emit(f, act, gamma, prefix="rmsnorm", cores=cores)
# result.cb_id = 7
# RMSNorm sees: inp.cb_id=3, inp.num_pages=4, inp.page_size=2048, ...
# RMSNorm does NOT know that cb_id=3 came from Mcast

# Step 3: Copy receives result handle
output = Copy.emit(f, result, output_tensor, prefix="flush")
# Copy sees: inp.cb_id=7, inp.num_pages=..., ...
# Copy does NOT know that cb_id=7 came from RMSNorm
```

### Zero-Coupling

Each `emit()` only knows about CBHandles, not about the identity of the producer or consumer. This zero-coupling has two important consequences:

1. **Recomposability**: Any op can be inserted, removed, or replaced in the chain. Mcast does not need to know about RMSNorm. RMSNorm does not need to know about Copy. The only interface contract is "I receive a CBHandle, I return a CBHandle."

2. **Metadata propagation**: The CBHandle carries all the metadata downstream ops need -- page count, page size, data format, tile geometry, core ranges. Downstream ops derive their own parameters from this metadata rather than receiving them separately.

This is enforced by the `isinstance(input, CBHandle)` pattern at the start of every `emit()`:

```python
if isinstance(src, CBHandle):
    src_cb = src          # upstream op's output
else:
    src_cb = f.cb_from_tensor(src)  # external tensor
```

The op treats both cases identically after this point. The rest of `emit()` uses `src_cb.cb_id`, `src_cb.num_pages`, `src_cb.page_size`, etc., regardless of origin.

### No Data Copy

When a CBHandle is passed to a downstream op, the downstream op uses the same physical CB. The upstream op's output CB becomes the downstream op's input CB. Data stays in place in L1.

### How Metadata Flows

When RMSNorm receives `act` (a CBHandle from Mcast), it reads:
- `act.num_pages` to know how many tiles to process
- `act.page_size` to compute buffer sizes
- `act.data_format` to set up compute units
- `act.tile_desc` to configure tile geometry
- `act.core_ranges` to know where to place its scratch CBs

This metadata propagation is why CBHandle carries more than just a CB ID. A bare integer would require every op to receive all metadata as separate parameters, coupling it to the upstream op's allocation choices.

## MultiOutput: Structured Results from Multi-Output Ops

Some ops produce multiple outputs. Instead of returning a tuple (which would lose port names), they return a `MultiOutput` wrapper.

**Definition** (from `blaze/fused_program.py`):

```python
class MultiOutput:
    """Structured result from a multi-output op's emit()."""

    def __init__(self, outputs: dict[str, CBHandle]):
        self._outputs = outputs

    def __getattr__(self, name: str) -> CBHandle:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._outputs[name]
        except KeyError:
            raise AttributeError(
                f"MultiOutput has no output port '{name}'. "
                f"Available: {sorted(self._outputs)}"
            )

    def __getitem__(self, key: str) -> CBHandle:
        return self._outputs[key]

    def __contains__(self, key: str) -> bool:
        return key in self._outputs

    def __len__(self) -> int:
        return len(self._outputs)

    def __iter__(self):
        return iter(self._outputs.values())

    def __repr__(self) -> str:
        port_names = ", ".join(self._outputs.keys())
        return f"MultiOutput({port_names})"
```

Note the `_`-prefix guard in `__getattr__`: names starting with `_` raise `AttributeError` immediately, preventing infinite recursion when Python looks up internal attributes like `__class__`.

**Supports four access patterns:**

1. **Positional unpacking** (order matches the `outputs` list passed to `multi_output()`):
   ```python
   scores, indices = DeepseekMoeGate.emit(f, ...)
   ```

2. **Attribute access** (by port name):
   ```python
   result = DeepseekMoeGate.emit(f, ...)
   result.output          # CBHandle for the "output" port
   result.output_indices  # CBHandle for the "output_indices" port
   ```

3. **Dict-style access**:
   ```python
   result["output"]
   result["output_indices"]
   ```

4. **Membership testing** (via `__contains__`):
   ```python
   "output" in result  # True
   ```

**How multi-output ops declare their outputs:**

Instead of calling `f.output()`, they call `f.multi_output()`:

```python
def multi_output(
    self,
    op_name: str,
    outputs: list[tuple[str, CBHandle]],
    *,
    grid=None,
    prefix: str = "",
    **port_sources,
) -> MultiOutput:
```

The `outputs` list is `[(port_name, CBHandle), ...]`. The first entry is the primary output used for auto-wiring. Port names must match `Output()` descriptors on the op class. Each handle gets `_node_id` and `_port_name` set so downstream ops can wire edges to any of the outputs.

## Metadata Propagation Through the Shadow Graph

The shadow graph is a `BlazeGraph` maintained by FusedProgram that records every op and its data dependencies. Each call to `f.output()` (or `f.multi_output()`) creates an `OpNode` and wires edges from input handles. The graph is built purely from CBHandle metadata -- no global state or explicit graph construction calls are needed from the op author.

### Node Creation

When `f.output()` is called, it invokes `_create_op_node()`:

```python
def _create_op_node(self, op_name, grid, **kwargs):
    spec = get_op_spec(op_name)
    count = self._op_counters.get(op_name, 0) + 1
    self._op_counters[op_name] = count
    node_id = f"{op_name}_{count}"  # e.g., "rmsnorm_1", "mcast_2"

    node = OpNode(
        id=node_id,
        spec=spec,
        grid=grid,
        kwargs=kwargs,
        ct_values=dict(self._pending_ct_values),  # CT args accumulated since last output()
    )
    self._pending_ct_values.clear()
    self._shadow_graph.add_node(node)
    return node, node_id
```

CT values emitted between two `f.output()` calls are associated with the second call's node. This is why the order of CT arg calls and the `output()` call matters: CT args must be declared before the `f.output()` that closes the op.

### Edge Wiring

Edges are created for CBHandle inputs that have a `_node_id` (meaning they came from an upstream `f.output()`). External tensors become external input ports:

```python
def _wire_input_edges(self, node, node_id, inputs):
    for port_name, source in inputs.items():
        if isinstance(source, CBHandle) and source._node_id:
            producer = nmap[source._node_id]
            edge = Edge(
                producer=producer,
                producer_port=source._port_name,
                consumer=node,
                consumer_port=port_name,
            )
            self._shadow_graph.add_edge(edge)
        elif isinstance(source, ExternalTensor):
            self._shadow_graph.external_input_ports.add((node_id, port_name))
```

### How `_node_id` and `_port_name` Are Set

After graph recording, the output handle's identity fields are stamped by `_record_op()`:

```python
def _record_op(self, op_name, inputs, output_handles, grid, **kwargs):
    node, node_id = self._create_op_node(op_name, grid, **kwargs)
    self._wire_input_edges(node, node_id, inputs)

    if isinstance(output_handles, dict):
        for port_name, handle in output_handles.items():
            handle._node_id = node_id
            handle._port_name = port_name
    else:
        output_handles._node_id = node_id
        output_handles._port_name = (
            node.spec.output_ports[0].name
            if node.spec.output_ports else "out"
        )
```

For single-output ops, the port name comes from the op's first `Output()` descriptor (e.g., `"output"` for RMSNorm, `"dst"` for Mcast). For multi-output ops, each handle gets its own port name from the `outputs` dict.

### Handle Identity Through the Graph

The CBHandle's `_node_id` and `_port_name` fields create an implicit identity chain:

```
Mcast.emit() -> f.output("mcast", dst, ...) -> handle._node_id = "mcast_1"
                                              -> handle._port_name = "dst"

RMSNorm.emit(f, handle, ...) -> f.output("rmsnorm", ..., input=handle)
                               -> edge: mcast_1:dst -> rmsnorm_1:input

Copy.emit(f, result, ...) -> f.output("copy", ..., src=result)
                            -> edge: rmsnorm_1:output -> copy_1:src
```

This chain is what makes the shadow graph a faithful record of the dataflow. The act of passing handles between emit() calls implicitly builds the graph.

### Alias Handle Identity

When `cb_alias()` creates a new handle, it inherits the source handle's `_node_id` and `_port_name`:

```python
handle._node_id = source._node_id
handle._port_name = source._port_name
```

This ensures that downstream consumers of the alias see the original data producer in the shadow graph, preserving data lineage even when the format view changes.

### Op Index and CB Lifetime Tracking

Each `f.output()` call increments `_op_index`:

```python
self._mark_cb_use(int(cb_id))
self._op_index += 1
```

CB lifetimes are tracked as `(first_use, last_use)` pairs indexed by `_op_index`. This information drives temporal compaction: CBs with non-overlapping `[first_use, last_use]` intervals can share the same CB ID.

### What the Shadow Graph Enables

1. **Kernel codegen**: The graph determines the phase ordering, include list, and init/teardown structure of the auto-generated kernel.
2. **CB lifetime analysis**: `_cb_lifetimes` tracks when each CB ID is first and last used, enabling temporal compaction.
3. **CT arg validation**: `_validate_ct_args_against_schema()` checks that emitted args match the op's declared schema.
4. **Engine validation**: `f.validate()` can run the CB/CT/Sem engines on the shadow graph and diff against the imperative allocations.

## The CBHandle Lifecycle

A summary of the journey every CBHandle takes through the system:

1. **Creation:** A `CBHandle` is created by one of the CB allocation methods (`cb_from_tensor`, `cb_scratch`, `cb_alias`, `cb_from_view`, `cb_from_shared_l1_views`, `cb_output`). At this point it has a `cb_id`, dimensions, and format metadata, but no graph identity.

2. **CT Arg Registration:** The handle's `cb_id` (via `int(handle)`) is passed as a CT arg value. The shadow graph captures this value in `_pending_ct_values`.

3. **Output Stamping:** `f.output()` stamps the handle with `_node_id` and `_port_name`, recording it as the output of a specific graph node. The pending CT values are cleared and associated with this node.

4. **Consumption:** The handle is passed to a downstream `emit()`. The downstream op reads `handle.cb_id` for its own CT args and passes the handle to its own `f.output()` as a `**port_source`, which creates a graph edge from the producer to the consumer.

5. **Terminal Use:** If the handle reaches `wire_output()` or is the last in the chain, it is bound to an output tensor. The CB's data becomes readable by the host after program execution.

This lifecycle -- create, register, stamp, consume, terminate -- is the heartbeat of every TT-Blaze fused program.

## CBHandle Metadata Summary

| Field | What downstream ops learn |
|-------|---------------------------|
| `cb_id` | Which hardware CB to read from |
| `num_pages` | How many tiles to process (loop count) |
| `page_size` | Buffer size for matching scratch allocations |
| `core_ranges` | Where to place downstream scratch CBs |
| `data_format` | Compute unit configuration, output format matching |
| `tile_desc` | Tile geometry for compute LLK calls |
| `byte_offset` | Where within the backing buffer to read (overlapped views) |
| `backing_tensor` | L1 address for direct-address consumers |
| `access_mode` | Whether FIFO or direct-address consumption is valid |
| `_node_id` | Shadow graph edge wiring (automatic) |
| `_port_name` | Shadow graph port identification (automatic) |
| `scratch_name` | Scratch mapping lookup key (internal) |

## Reference

- `blaze/cb_handle.py` -- `CBHandle` dataclass, `CBAccessMode` enum, `require_fifo_handle()`
- `blaze/fused_program.py` -- `MultiOutput` class, `f.output()`, `f.multi_output()`, `_record_op()`, `_create_op_node()`, `_wire_input_edges()`
