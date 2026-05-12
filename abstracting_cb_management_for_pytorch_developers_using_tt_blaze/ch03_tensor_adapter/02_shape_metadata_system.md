# The Shape Metadata System: ShapeDescriptor, ShapeTracker, and TypedHandle

The shape metadata system is the informational backbone of TensorAdapter. It carries PyTorch-world shape information (logical dimensions, dtype) alongside hardware-world metadata (tile shape, padded dimensions, tile grid counts, padding strategy) through every stage of the pipeline -- from initial tensor analysis through op emission to final output extraction. This section defines the three core data structures (`ShapeDescriptor`, `ShapeTracker`, `TypedHandle`), their factory methods, shape inference rules per operation category, and their precise relationship to existing Blaze structures (`TensorPort`, `CBHandle`).

**What you will learn:**

- The `ShapeDescriptor` dataclass: what fields it carries, why each field exists, why it is frozen (immutable), and the invariants between fields -- including `__post_init__` validation
- Static factory methods (`from_torch`, `from_ttnn`, `from_cb_handle`) and why each input path requires different extraction logic
- Functional update methods (`with_tile`, `with_format`, `with_padding`) for safe mutation of frozen descriptors
- The `ShapeTracker`: how it propagates `ShapeDescriptor` through chains of `emit()` calls within a `FusedProgram`
- `TypedHandle`: the composite return type (CBHandle + ShapeDescriptor) that enables downstream ops to configure themselves without manual parameters
- Shape inference rules per op category: matmul, elementwise, reduction, broadcast, reshape
- A worked 11-node gated MLP walkthrough demonstrating shape propagation through a production-scale fused op
- Edge cases: scalar tensors, 1D tensors, sub-tile tensors, and batch dimensions
- How ShapeDescriptor fills the gaps in existing `TensorPort` and `CBHandle`

---

## The ShapeDescriptor Dataclass

### Design Rationale

Chapter 2's Principle P2 (Carry Metadata from Both Worlds Simultaneously) requires a data structure that holds PyTorch-world information and hardware-world information at the same time, without either side losing information. Symbiote's `TorchTTNNTensor` achieves this with `.elem` (torch) and `.ttnn_tensor` (device). TensorAdapter's analog is `ShapeDescriptor`.

The key insight -- drawn from XLA's shape tracking (DP12) and MLIR's type preservation through lowering (DP13) -- is that the system must track **three** shape representations simultaneously:

1. **Logical shape**: what the developer wrote (`[1, 4096]`)
2. **Padded shape**: after tile-boundary alignment (`[32, 4096]`)
3. **Tile grid shape**: tile counts per dimension (`[1, 128]` with `(32, 32)` tiles)

Losing any one of these creates bugs. Losing the logical shape means the output cannot be sliced back to the correct size. Losing the padded shape means downstream ops receive incorrect CB sizing. Losing the tile grid means CT args like `k_num_tiles` cannot be derived.

```python
# PROPOSED (illustrative, not prescriptive) -- ShapeDescriptor dataclass
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class ShapeDescriptor:
    """Immutable metadata tracking a tensor's shape at all three levels.

    Invariants (validated in __post_init__):
      - len(logical_shape) == len(padded_shape)
      - padded_shape[i] >= logical_shape[i] for all i
      - padded_shape[-2] % tile_shape[0] == 0
      - padded_shape[-1] % tile_shape[1] == 0
      - tile_grid_shape == (padded_shape[-2] // tile_shape[0],
                            padded_shape[-1] // tile_shape[1])
      - total_tiles == product(tile_grid_shape) * product(batch_dims)
      - page_size == tile.get_tile_size(data_format)  (for BFP formats)
              or == tile_h * tile_w * DTYPE_BYTES[data_format]  (for unpacked)
    """

    # --- PyTorch world ---
    logical_shape: tuple[int, ...]
    """The tensor shape as written by the developer. E.g., (1, 4096)."""

    # --- Tile world ---
    tile_shape: tuple[int, int] = (32, 32)
    """Tile geometry: (tile_H, tile_W). Default (32, 32)."""

    padded_shape: tuple[int, ...] = ()
    """Logical shape rounded up to tile boundaries on the last two dims."""

    tile_grid_shape: tuple[int, ...] = (1, 1)
    """Number of tiles per dimension. For rank > 2, includes batch dims."""

    total_tiles: int = 1
    """Total tile count across all dimensions (including batch)."""

    # --- Hardware world ---
    data_format: str = "bfloat16"
    """Data format for CB allocation. Matches cb_engine.DTYPE_BYTES keys."""

    page_size: int = 2048
    """Bytes per tile page. For BFP formats, includes shared exponent overhead."""

    # --- Padding ---
    padding_strategy: str = "ZERO"
    """Padding policy name: 'ZERO', 'NEG_INF', 'MASK', 'IDENTITY', 'NONE'."""

    padding_fill: float = 0.0
    """Fill value for padded positions."""

    # --- Provenance ---
    source: str = ""
    """How this descriptor was created: 'from_torch', 'from_ttnn', 'inferred'."""

    op_hint: str = ""
    """The op_hint that was active when this descriptor was created."""
```

### Field Invariants and __post_init__ Validation

The frozen dataclass ensures immutability, but invariants must be checked at construction time. The `__post_init__` method (using `object.__setattr__` for frozen dataclasses) validates (cherry-picked from V4):

```python
# PROPOSED (illustrative, not prescriptive) -- ShapeDescriptor __post_init__
    def __post_init__(self):
        # Rank consistency
        if len(self.logical_shape) != len(self.padded_shape):
            raise ValueError(
                f"logical_shape rank ({len(self.logical_shape)}) != "
                f"padded_shape rank ({len(self.padded_shape)})"
            )

        # Padded >= logical
        for i, (l, p) in enumerate(zip(self.logical_shape, self.padded_shape)):
            if p < l:
                raise ValueError(
                    f"padded_shape[{i}] ({p}) < logical_shape[{i}] ({l})"
                )

        # Tile alignment on last two dims
        th, tw = self.tile_shape
        if len(self.padded_shape) >= 2 and self.padded_shape[-2] % th != 0:
            raise ValueError(
                f"padded_shape[-2] ({self.padded_shape[-2]}) not aligned "
                f"to tile_shape[0] ({th})"
            )
        if len(self.padded_shape) >= 1 and self.padded_shape[-1] % tw != 0:
            raise ValueError(
                f"padded_shape[-1] ({self.padded_shape[-1]}) not aligned "
                f"to tile_shape[1] ({tw})"
            )

        # Total tiles consistency
        expected = 1
        for g in self.tile_grid_shape:
            expected *= g
        if self.total_tiles != expected:
            raise ValueError(
                f"total_tiles ({self.total_tiles}) != product of "
                f"tile_grid_shape {self.tile_grid_shape} ({expected})"
            )
```

These checks catch misconfigured descriptors at construction time rather than allowing incorrect values to propagate silently through the pipeline.

### Why Frozen (Immutable)?

`ShapeDescriptor` is a frozen dataclass. This is a deliberate design choice with three motivations:

1. **Safety in op chains:** When op A's output `ShapeDescriptor` is passed to op B's `adapt()` call, op B must not mutate it -- op A might pass the same descriptor to op C (fan-out). Immutability prevents this class of bug.
2. **Hashability:** Frozen dataclasses are hashable, enabling use as dict keys for caching (e.g., memoizing tile decomposition results for repeated shapes).
3. **Debug clarity:** When a shape mismatch occurs, the developer can inspect the descriptor and know it was not silently modified after creation.

Mutations produce new instances via `with_*()` methods:

```python
# PROPOSED (illustrative, not prescriptive) -- ShapeDescriptor mutation methods
    def with_tile(self, tile_shape: tuple[int, int]) -> "ShapeDescriptor":
        """Return a new ShapeDescriptor with the given tile shape.
        Recomputes padded_shape, tile_grid_shape, total_tiles, page_size.
        """
        padded = _pad_to_tiles(self.logical_shape, tile_shape)
        grid = _compute_tile_grid(padded, tile_shape)
        return ShapeDescriptor(
            logical_shape=self.logical_shape,
            tile_shape=tile_shape,
            padded_shape=padded,
            tile_grid_shape=grid,
            total_tiles=_total_tiles(self.logical_shape, grid),
            data_format=self.data_format,
            page_size=_page_size(tile_shape, self.data_format),
            padding_strategy=self.padding_strategy,
            padding_fill=self.padding_fill,
            source=self.source,
            op_hint=self.op_hint,
        )

    def with_format(self, data_format: str) -> "ShapeDescriptor":
        """Return a new ShapeDescriptor with the given data format.
        Recomputes page_size (tile_shape unchanged).
        """
        return ShapeDescriptor(
            logical_shape=self.logical_shape,
            tile_shape=self.tile_shape,
            padded_shape=self.padded_shape,
            tile_grid_shape=self.tile_grid_shape,
            total_tiles=self.total_tiles,
            data_format=data_format,
            page_size=_page_size(self.tile_shape, data_format),
            padding_strategy=self.padding_strategy,
            padding_fill=self.padding_fill,
            source=self.source,
            op_hint=self.op_hint,
        )

    def with_padding(self, strategy: str, fill: float) -> "ShapeDescriptor":
        """Return a new ShapeDescriptor with the given padding policy."""
        return ShapeDescriptor(
            logical_shape=self.logical_shape,
            tile_shape=self.tile_shape,
            padded_shape=self.padded_shape,
            tile_grid_shape=self.tile_grid_shape,
            total_tiles=self.total_tiles,
            data_format=self.data_format,
            page_size=self.page_size,
            padding_strategy=strategy,
            padding_fill=fill,
            source=self.source,
            op_hint=self.op_hint,
        )
```

---

## Factory Methods

Three factory methods handle the three input types that TensorAdapter encounters. Each requires different extraction logic because the metadata lives in different places.

### ShapeDescriptor.from_torch(tensor)

```python
# PROPOSED (illustrative, not prescriptive) -- Factory: from PyTorch tensor
@staticmethod
def from_torch(
    tensor: torch.Tensor,
    *,
    op_hint: str = "",
    tile_shape: tuple[int, int] = (32, 32),
    data_format: str = "bfloat16",
) -> "ShapeDescriptor":
    """Infer all fields from a PyTorch tensor.

    The PyTorch tensor carries logical_shape and dtype. Tile shape,
    padded shape, and data format must be provided or defaulted.

    Validates:
    - tensor is a torch.Tensor (not None, not scalar without reshaping)
    - rank >= 1 (scalars are promoted to [1, 1])
    - no zero-size dimensions

    This is the primary entry point for new tensors entering the system.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    logical = tuple(tensor.shape)
    if len(logical) == 0:
        logical = (1, 1)  # promote scalar to [1, 1]
    elif len(logical) == 1:
        logical = (1,) + logical  # promote 1D to [1, N]
    if any(d <= 0 for d in logical):
        raise ValueError(f"Zero or negative dimension in shape {logical}")

    padded = _pad_to_tiles(logical, tile_shape)
    grid = _compute_tile_grid(padded, tile_shape)

    return ShapeDescriptor(
        logical_shape=logical,
        tile_shape=tile_shape,
        padded_shape=padded,
        tile_grid_shape=grid,
        total_tiles=_total_tiles(logical, grid),
        data_format=data_format,
        page_size=_page_size(tile_shape, data_format),
        source="from_torch",
        op_hint=op_hint,
    )
```

### ShapeDescriptor.from_ttnn(tensor)

```python
# PROPOSED (illustrative, not prescriptive) -- Factory: from on-device TTNN tensor
@staticmethod
def from_ttnn(tensor) -> "ShapeDescriptor":
    """Extract shape metadata from a TTNN tensor already on device.

    TTNN tensors carry tile information (tensor.get_tile()), dtype,
    and memory config (shard spec). The logical shape must be recovered
    from the unpadded shape or stored metadata.

    This path is used when TensorAdapter receives pre-placed weights
    (e.g., from Symbiote's preprocess_weights() pipeline).

    Note: page_size is obtained via tile.get_tile_size(tensor.dtype),
    which correctly handles BFP formats (bfloat8_b -> ~1088 bytes,
    bfloat4_b -> ~576 bytes).
    """
    tile = tensor.get_tile()
    tile_shape = (tile.tile_shape[0], tile.tile_shape[1])
    padded = tuple(tensor.shape)
    logical = _infer_logical_from_padded(padded, tile_shape)

    return ShapeDescriptor(
        logical_shape=logical,
        tile_shape=tile_shape,
        padded_shape=padded,
        tile_grid_shape=_compute_tile_grid(padded, tile_shape),
        total_tiles=_total_tiles(logical, _compute_tile_grid(padded, tile_shape)),
        data_format=_ttnn_dtype_to_str(tensor.dtype),
        page_size=tile.get_tile_size(tensor.dtype),
        source="from_ttnn",
    )
```

### ShapeDescriptor.from_cb_handle(handle)

```python
# PROPOSED (illustrative, not prescriptive) -- Factory: from existing CBHandle
@staticmethod
def from_cb_handle(handle: CBHandle) -> "ShapeDescriptor":
    """Reconstruct shape metadata from a CBHandle.

    CBHandle carries page_size, num_pages, data_format, and tile_desc,
    but NOT logical_shape. This factory produces a "best-effort"
    descriptor where logical_shape == padded_shape (no padding info).

    This path is used when adapt() receives a CBHandle from a prior
    emit() call that did not go through TensorAdapter (backward
    compatibility with hand-written code).
    """
    tile_h = handle.tile_desc.height
    tile_w = handle.tile_desc.width
    total_elements = handle.num_pages * tile_h * tile_w
    padded_w = tile_w  # minimum: one tile wide
    padded_h = total_elements // padded_w

    return ShapeDescriptor(
        logical_shape=(padded_h, padded_w),  # best effort
        tile_shape=(tile_h, tile_w),
        padded_shape=(padded_h, padded_w),
        tile_grid_shape=(padded_h // tile_h, padded_w // tile_w),
        total_tiles=handle.num_pages,
        data_format=str(handle.data_format),
        page_size=handle.page_size,
        source="from_cb_handle",
    )
```

> **Warning:** `from_cb_handle()` cannot recover the original logical shape because `CBHandle` does not store it. The resulting descriptor has `logical_shape == padded_shape`, which means output unpadding will be a no-op. Additionally, the shape reconstruction assumes single-tile width, which may not match the original layout for multi-column tile grids. This is the correct "best-effort" behavior for backward compatibility -- hand-written emit() calls never tracked logical shape. However, mixing hand-written and TensorAdapter-managed ops in a single fused chain may produce incorrect output shapes at the boundary. The workaround is to use `TypedHandle` consistently within TensorAdapter-managed chains.

---

## Relationship to Existing Structures

`ShapeDescriptor` fills specific gaps in two existing Blaze data structures:

### TensorPort (blaze/graph.py)

```python
# EXISTING -- TensorPort
@dataclass
class TensorPort:
    name: str
    dtype: Optional[str] = None
    tile_shape: Optional[tuple[int, int]] = None
```

`TensorPort` defines a CB endpoint on an op. It carries `dtype` and `tile_shape` (both optional) but **not** logical shape, padded shape, tile grid counts, page size, padding strategy, or total tiles.

| Field | TensorPort | ShapeDescriptor | Relationship |
|-------|-----------|----------------|-------------|
| dtype | `dtype: str` | `data_format: str` | Same information, different field name |
| tile_shape | `tile_shape: tuple` | `tile_shape: tuple` | Identical |
| logical_shape | -- | `logical_shape: tuple` | **New**: not in TensorPort |
| padded_shape | -- | `padded_shape: tuple` | **New**: not in TensorPort |
| tile_grid | -- | `tile_grid_shape: tuple` | **New**: derived from padded + tile |
| page_size | -- | `page_size: int` | **New**: derived from tile + dtype |
| padding | -- | `padding_strategy, padding_fill` | **New**: per-op semantics |

TensorAdapter does not modify TensorPort. Instead, it attaches `ShapeDescriptor` to graph edges (see [File 04](./04_interaction_with_graph_and_compilation_paths.md)), enriching the graph with metadata that TensorPort cannot carry.

### CBHandle (blaze/cb_handle.py)

CBHandle carries the **hardware-side** information (page geometry, format, tile descriptor) but not the **PyTorch-side** information (logical shape, padding metadata). The mapping:

| CBHandle Field | ShapeDescriptor Derivation |
|---------------|--------------------------|
| `num_pages` | Computed from `tile_grid_shape`, blocking strategy, and op requirements |
| `page_size` | From `tile_shape` + `data_format` (using `tile.get_tile_size()` for BFP) |
| `data_format` | `data_format` field (after format negotiation) |
| `tile_desc` | Constructed from `tile_shape` |

`ShapeDescriptor` is the **input** from which `CBHandle` parameters are derived. The CBBackend layer performs this derivation. `TypedHandle` bundles both together so downstream ops have access to both.

---

## The ShapeTracker

Within a `FusedProgram` composition -- where multiple `emit()` calls are chained -- shape metadata must propagate from one op's output to the next op's input. The `ShapeTracker` manages this propagation.

### Why Not Just Pass ShapeDescriptor Manually?

In simple cases, the developer could pass `ShapeDescriptor` objects between adapter calls. But in complex fused ops (like the 11-node gated MLP graph from `blaze/_gated_mlp.py`), manual passing becomes error-prone. The ShapeTracker automates this by recording each op's output descriptor and making it available when the next op's `adapt()` is called.

```python
# PROPOSED (illustrative, not prescriptive) -- ShapeTracker
class ShapeTracker:
    """Propagates ShapeDescriptor through chains of emit() calls.

    Each op's TensorAdapter method returns a TypedHandle. The ShapeTracker
    records the CBHandle -> ShapeDescriptor mapping so that when the next op
    receives a CBHandle as input, adapt() can recover the ShapeDescriptor.

    The tracker is scoped to a single FusedProgram composition and is
    discarded after build().
    """

    def __init__(self):
        self._handle_to_shape: dict[int, ShapeDescriptor] = {}
        # Key is id(CBHandle) -- CBHandle uses eq=False, so identity is stable

    def record(self, handle: CBHandle, shape: ShapeDescriptor) -> None:
        """Record the shape metadata for a CBHandle."""
        self._handle_to_shape[id(handle)] = shape

    def lookup(self, handle: CBHandle) -> ShapeDescriptor | None:
        """Recover the ShapeDescriptor for a CBHandle from a prior emit()."""
        return self._handle_to_shape.get(id(handle))

    def clear(self) -> None:
        """Discard all recorded metadata (called by FusedProgram.build())."""
        self._handle_to_shape.clear()
```

### Design Decision: id()-Based Lookup

The ShapeTracker keys on `id(CBHandle)` rather than `CBHandle.cb_id` or a string name. The reasons:

1. **cb_id is not unique across programs.** Two different FusedProgram compositions might assign the same cb_id to different CBs.
2. **CBHandle uses `eq=False`.** This means Python's default `id()`-based identity is stable for the lifetime of the object.
3. **Name-based lookup would require consistent naming.** Ops use varied prefix conventions (`"matmul"`, `"mm"`, `"gate_matmul"`); relying on names is fragile.

The limitation: `id()` is only valid while the CBHandle object exists. If a CBHandle is garbage-collected, its `id()` can be reused. Within a single `FusedProgram.build()` scope, all handles are alive, so this is safe. The `clear()` method ensures no stale references leak.

---

## TypedHandle: The Composite Return Type

`TypedHandle` wraps a `CBHandle` with its `ShapeDescriptor`. It is the return type of `TensorAdapter.adapt()` and the currency passed between adapter-managed ops:

```python
# PROPOSED (illustrative, not prescriptive) -- TypedHandle
@dataclass
class TypedHandle:
    """A CBHandle enriched with shape metadata.

    The CBHandle provides hardware identity (cb_id, page geometry).
    The ShapeDescriptor provides shape intelligence (logical dims,
    padding, format). Together, they give downstream ops everything
    needed for automatic configuration.

    Attribute access delegates to the underlying CBHandle via __getattr__,
    so TypedHandle can be passed where CBHandle is expected without
    modification to existing emit() code.
    """

    handle: CBHandle
    """The underlying CB reference. Use this for emit() calls."""

    shape: ShapeDescriptor
    """Shape metadata. Use this for shape-dependent decisions."""

    # Delegate attribute access to the underlying CBHandle
    # so TypedHandle can be used wherever CBHandle is expected.
    def __getattr__(self, name):
        return getattr(self.handle, name)

    def __int__(self) -> int:
        """Enable TypedHandle where an int CB ID is expected."""
        return int(self.handle)

    def __index__(self) -> int:
        return int(self.handle)

    @property
    def cb_id(self) -> int:
        return self.handle.cb_id

    @property
    def num_pages(self) -> int:
        return self.handle.num_pages

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return self.shape.logical_shape

    @property
    def tile_grid(self) -> tuple[int, ...]:
        return self.shape.tile_grid_shape
```

### Why __getattr__ Delegation?

Existing `emit()` methods (like `Matmul.emit()`) accept `CBHandle` objects and read `.num_pages`, `.page_size`, `.data_format`, `.tile_desc`. By delegating attribute access, `TypedHandle` can be passed directly to existing `emit()` code without modification:

```python
# EXISTING -- Matmul.emit() reads CBHandle fields (no changes needed)
k_num_tiles = in0_handle.num_pages    # works with TypedHandle: delegates to .handle
page_size = in0_handle.page_size      # works with TypedHandle: delegates to .handle

# PROPOSED -- ShapeTracker reads ShapeDescriptor fields (new capability)
in0_shape = tracker.lookup(in0_handle) # returns ShapeDescriptor
K = in0_shape.logical_shape[-1]        # 4096 -- the contraction dimension
```

### Why a Wrapper Instead of Extending CBHandle?

Adding `ShapeDescriptor` fields directly to `CBHandle` was rejected because:

1. **Lifetime mismatch.** CBHandle lives through program execution; ShapeDescriptor is only needed during graph construction.
2. **Circular dependency.** ShapeDescriptor includes `page_size` and `data_format`, which are also on CBHandle. Embedding one inside the other creates redundancy.
3. **API contract.** Every existing call site that constructs a CBHandle would need to pass the new field, or it would always be `None` in non-TensorAdapter paths.

---

## Shape Inference Rules Per Op Category

When an op's `emit()` produces an output `CBHandle`, the ShapeEngine must infer the output `ShapeDescriptor` from the input descriptors. These rules are registered per op category:

### Matmul: Contract K, Produce [M, N]

```python
# PROPOSED (illustrative, not prescriptive) -- Matmul shape inference
def infer_matmul_output(
    self, in0: ShapeDescriptor, in1: ShapeDescriptor
) -> ShapeDescriptor:
    """Matmul: in0=[..., M, K] x in1=[..., K, N] -> out=[..., M, N].

    The K dimension contracts (inner product). The output inherits:
    - Batch dimensions from in0 (broadcasting applied if needed)
    - M (rows) from in0's second-to-last dimension
    - N (columns) from in1's last dimension
    - Tile shape from in0 (both inputs must share tile geometry)
    - Data format from in0 (output matches activation format)
    """
    M = in0.logical_shape[-2]
    N = in1.logical_shape[-1]
    batch = in0.logical_shape[:-2]
    logical = batch + (M, N)
    return ShapeDescriptor(
        logical_shape=logical,
        tile_shape=in0.tile_shape,
        padded_shape=_pad_to_tiles(logical, in0.tile_shape),
        # ... remaining fields derived
        data_format=in0.data_format,
        source="inferred",
        op_hint="matmul.out",
    )
```

### Elementwise: Preserve Shape

```python
# PROPOSED (illustrative, not prescriptive) -- Elementwise shape inference
def infer_elementwise_output(
    self, in0: ShapeDescriptor, in1: ShapeDescriptor | None = None
) -> ShapeDescriptor:
    """Elementwise ops (add, mul, silu, relu): output shape == input shape.

    For binary ops (add, mul), shapes must match after broadcasting.
    For unary ops (silu, relu), output shape == in0 shape.
    """
    if in1 is not None:
        logical = _broadcast_shapes(in0.logical_shape, in1.logical_shape)
    else:
        logical = in0.logical_shape
    return in0.with_tile(in0.tile_shape)  # preserves shape, recomputes grids
```

### Reduction: Collapse Dimension

```python
# PROPOSED (illustrative, not prescriptive) -- Reduction shape inference
def infer_reduction_output(
    self, input_desc: ShapeDescriptor, reduce_dim: int
) -> ShapeDescriptor:
    """Reduction ops (sum, mean, max along a dimension).
    The reduced dimension becomes size 1 (keepdim=True) or is removed.
    """
    logical = list(input_desc.logical_shape)
    logical[reduce_dim] = 1  # keepdim=True default
    return ShapeDescriptor(
        logical_shape=tuple(logical),
        tile_shape=input_desc.tile_shape,
        padded_shape=_pad_to_tiles(tuple(logical), input_desc.tile_shape),
        data_format=input_desc.data_format,
        source="inferred",
        op_hint=f"reduce.dim{reduce_dim}",
    )
```

### Broadcast: Expand Dimension

Shape inference for broadcasts is detailed in [Chapter 8: Broadcasting](../ch08_broadcasting/01_pytorch_broadcasting_rules.md). The key rule: when a dimension of size 1 is broadcast to size N, the output `ShapeDescriptor`'s `logical_shape` has N in that position, but the CB only holds 1 tile for that dimension (the tile is reused N times by the compute kernel).

### Registered Inference Rules

| Op Category | Op Names | Rule | CT Args Derivable |
|------------|----------|------|-------------------|
| Matmul | matmul, matmul_cb, kn_sliced_matmul | Contract K: `[M,K] x [K,N] -> [M,N]` | `k_num_tiles`, `out_w_per_core` |
| Elementwise binary | add, mul, sub | Broadcast, then preserve shape | (none shape-dependent) |
| Elementwise unary | silu, relu, gelu | Preserve shape | (none shape-dependent) |
| Reduction | reduce, gated_local_reduce | Collapse reduced dim to 1 | `num_tiles` (accumulation) |
| Norm | rmsnorm, layernorm, padded_rmsnorm | Preserve shape, may change tile to (1,32) | `width` (tile count) |
| Data movement | mcast, gather | Preserve shape, change core_ranges | `num_cores`, NOC coords |
| Attention | sdpa | Complex: Q/K/V shapes -> attention output | Multiple blocking params |

---

## Worked Example: Propagating Shapes Through a Gated MLP

Tracing shapes through the 11-node gated MLP graph from `blaze/_gated_mlp.py` (cherry-picked from V3's walkthrough). This demonstrates that ShapeTracker handles production-scale fused ops, not just toy examples:

```
Node 1: mcast(activation [1, 4096])
  input:  ShapeDescriptor(logical=(1, 4096), tile=(1, 32), grid=(1, 128))
  output: ShapeDescriptor(logical=(1, 4096), tile=(1, 32), grid=(1, 128))
  # Shape unchanged; data replicated to all cores.

Node 2: kn_sliced_matmul(act [1, 4096], gate_weights [4096, 11008])
  in0:    ShapeDescriptor(logical=(1, 4096), tile=(1, 32), grid=(1, 128))
  in1:    ShapeDescriptor(logical=(4096, 11008), tile=(32, 32), grid=(128, 344))
  output: ShapeDescriptor(logical=(1, 11008), tile=(1, 32), grid=(1, 344))
  # Contract K=4096, produce N=11008 (gate half of gate_up)

Node 3: kn_sliced_matmul(act, up_weights [4096, 11008])
  output: ShapeDescriptor(logical=(1, 11008), tile=(1, 32), grid=(1, 344))

Node 4-5: gather(gate), gather(up)
  # Shapes unchanged; data moves from compute cores to sender core.

Node 6: gated_reduce(gate [1, 11008], up [1, 11008])
  in0:    ShapeDescriptor(logical=(1, 11008), ...)
  in1:    ShapeDescriptor(logical=(1, 11008), ...)
  output: ShapeDescriptor(logical=(1, 11008), tile=(1, 32), grid=(1, 344))
  # Elementwise SiLU(gate) * up. Shape preserved.

Node 7: mcast(reduced [1, 11008])
  output: ShapeDescriptor(logical=(1, 11008), ...)  # replicate to all cores

Node 8: matmul(reduced [1, 11008], down_weights [11008, 4096])
  in0:    ShapeDescriptor(logical=(1, 11008), tile=(1, 32), grid=(1, 344))
  in1:    ShapeDescriptor(logical=(11008, 4096), tile=(32, 32), grid=(344, 128))
  output: ShapeDescriptor(logical=(1, 4096), tile=(1, 32), grid=(1, 128))
  # Contract K=11008, produce N=4096. Back to hidden dimension.

Node 9: mcast(bias [1, 4096])
  output: ShapeDescriptor(logical=(1, 4096), ...)

Node 10: residual_add(matmul_out [1, 4096], bias [1, 4096])
  output: ShapeDescriptor(logical=(1, 4096), tile=(1, 32), grid=(1, 128))

Node 11: gather(result [1, 4096])
  output: ShapeDescriptor(logical=(1, 4096), ...)  # to sender core
```

The ShapeTracker maintains all 11 descriptors and provides them to each subsequent op. Without the tracker, each op's `emit()` must recompute shape information from raw CBHandle fields -- which is what happens in manual Blaze code today.

---

## Handling Shape-Dependent CT Args

One of TensorAdapter's primary responsibilities is deriving the CT args that each op's kernel expects:

```python
# EXISTING -- manual CT arg computation in Matmul.emit()
k_num_tiles = in0_handle.num_pages                    # K dimension tiles
out_w_per_core = in1_cb.num_pages // k_num_tiles       # N tiles per core

# PROPOSED -- automatic CT arg derivation from ShapeDescriptor
k_num_tiles = in0_desc.tile_grid_shape[-1]             # K tiles from grid
out_w_per_core = in1_desc.total_tiles // k_num_tiles   # N tiles from grid
```

The key difference: in existing Blaze, `k_num_tiles = in0_handle.num_pages` relies on the developer's knowledge that `num_pages` for in0 represents the K dimension. With ShapeDescriptor, the meaning is explicit: `tile_grid_shape[-1]` is the last dimension of the tile grid, which is the contraction dimension.

| CT Arg | Manual Derivation | ShapeDescriptor Derivation |
|--------|------------------|--------------------------|
| `k_num_tiles` | `in0_handle.num_pages` | `in0.tile_grid_shape[-1]` |
| `out_w_per_core` | `in1_cb.num_pages // k_num_tiles` | `in1.total_tiles // k_num_tiles` |
| `num_tiles` (reduction) | Counted from tensor shape | `input.total_tiles` |
| `width` (rmsnorm) | `tensor.shape[-1] // 32` | `input.tile_grid_shape[-1]` |
| `is_broadcast` | Manual shape comparison | `in0.logical_shape != in1.logical_shape` |

> **Warning:** The CT arg derivation must produce **identical** integer values to the manual computation, because the kernel reads these values as compile-time constants. A single off-by-one error in tile count produces silent numerical corruption. The validation strategy is to run both derivation paths (manual and ShapeDescriptor-based) in parallel during development and assert equality.

---

## Edge Cases and Validation

### Scalar Tensors

```python
desc = ShapeDescriptor.from_torch(torch.tensor(3.14), tile_shape=(32, 32))
# logical_shape = (1, 1)    (promoted from scalar)
# padded_shape  = (32, 32)  (padded to one full tile)
# tile_grid_shape = (1, 1)  (one tile)
# total_tiles   = 1
```

### 1D Tensors

```python
desc = ShapeDescriptor.from_torch(torch.randn(768), tile_shape=(1, 32))
# logical_shape = (1, 768)  (promoted: [768] -> [1, 768])
# padded_shape  = (1, 768)  (768 / 32 = 24, already aligned)
# tile_grid_shape = (1, 24)
# total_tiles   = 24
```

### Tensors Smaller Than One Tile

```python
desc = ShapeDescriptor.from_torch(torch.randn(1, 7), tile_shape=(32, 32))
# logical_shape = (1, 7)
# padded_shape  = (32, 32)   (both dims padded to one full tile)
# tile_grid_shape = (1, 1)   (one tile, mostly padding)
# total_tiles   = 1
# padding_fill  = 0.0
#
# WARNING: 7 out of 1024 elements are real data. 99.3% of the tile is padding.
# The adapter should warn the developer when the padding ratio exceeds a threshold.
```

### Batch Dimensions

```python
desc = ShapeDescriptor.from_torch(torch.randn(2, 8, 4096, 4096), tile_shape=(32, 32))
# logical_shape = (2, 8, 4096, 4096)
# padded_shape  = (2, 8, 4096, 4096)  (last two dims already aligned)
# tile_grid_shape = (2, 8, 128, 128)  (batch dims preserved in grid)
# total_tiles   = 2 * 8 * 128 * 128 = 262144
```

### Non-Aligned Shapes (the Common Case for RMSNorm)

```python
desc = ShapeDescriptor.from_torch(torch.randn(1, 37), tile_shape=(1, 32))
# logical_shape = (1, 37)
# padded_shape  = (1, 64)    (ceil(37/32)*32 = 64)
# tile_grid_shape = (1, 2)
# total_tiles   = 2
# The second tile holds only 5 real values + 27 padded zeros.
```

---

## Key Takeaways

- `ShapeDescriptor` is a frozen (immutable) dataclass that carries three shape levels simultaneously: logical (PyTorch), padded (tile-aligned), and tile grid (hardware). Immutability prevents accidental mutation in fan-out scenarios and enables memoization. A `__post_init__` validator checks rank consistency, padding correctness, tile alignment, and total_tiles consistency at construction time.
- Three factory methods (`from_torch`, `from_ttnn`, `from_cb_handle`) handle the three input paths, with `from_cb_handle` providing best-effort backward compatibility. Three `with_*()` methods (`with_tile`, `with_format`, `with_padding`) enable functional updates that safely create new frozen instances.
- `ShapeTracker` propagates descriptors through emit() chains using `id(CBHandle)` as the lookup key, scoped to a single FusedProgram build. This eliminates manual shape threading in complex fused ops like the 11-node gated MLP.
- `TypedHandle` wraps `CBHandle` + `ShapeDescriptor` with `__getattr__` delegation, making it a transparent drop-in for CBHandle in existing emit() code while carrying shape metadata for TensorAdapter-managed paths.
- Shape inference rules are registered per op category (matmul, elementwise, reduction, broadcast, reshape), enabling automatic derivation of CT args like `k_num_tiles` and `out_w_per_core`.

## Source Files

- `blaze/graph.py` -- `TensorPort` (existing: dtype + tile_shape only), `OpSpec`, `Edge`
- `blaze/cb_handle.py` -- `CBHandle` (existing: num_pages, page_size, data_format, tile_desc)
- `blaze/cb_engine.py` -- `DTYPE_BYTES`, `_tile_page_size()`, `DEFAULT_TILE_SHAPE` (existing derivation logic)
- `blaze/ct_args.py` -- `CTArgSpec`, `CTArgSchema` (existing: CT arg type system)
- `blaze/context.py` -- `FusionResult` (existing: node + port_name, extended in File 04)
- `blaze/fused_program.py` -- `TileInfo.from_tensor()` (existing: tile metadata extraction pattern)
- `blaze/_gated_mlp.py` -- 11-node gated MLP graph (worked example for shape propagation)

---

[Previous: 01 -- Architecture Overview](./01_architecture_overview.md) | [Next: 03 -- Wrapping vs. Extending vs. Replacing](./03_wrapping_vs_extending_vs_replacing.md)
