# Shape Propagation Through Op Chains

When operations are chained -- matmul feeding into SiLU feeding into another matmul -- each operation transforms the tensor shape. The shape propagation protocol ensures that every downstream operation receives correct sizing information from its predecessors without manual specification. This section defines the `infer_output_shape()` protocol, the shape rules for every common op category, how TypedHandle carries shape metadata through emit() chains, how shape-dependent CT args are derived automatically, and what happens when shapes mismatch at op boundaries.

> *Design Principle P1 (Separate Computation Intent from Execution Strategy):* Shape propagation separates the developer's computation chain (matmul -> silu -> matmul) from the machinery that determines tile grids, page sizes, and CT args at each boundary. The developer chains ops; the propagation protocol fills in the hardware details.

**What you will learn:**

- The shape propagation protocol: each op declares how it transforms input shapes into output shapes via `infer_output_shape()`
- Shape inference rules for 10 op categories: matmul, elementwise unary, elementwise binary, reduction, gated reduce, normalization, data movement, SDPA, broadcast, and reshape/transpose
- How `TypedHandle` wraps `CBHandle` with `ShapeDescriptor` so each op's output carries shape metadata for the next op
- Handling shape-dependent CT args: deriving `k_num_tiles`, `out_w_per_core`, `num_tiles`, and `width` from ShapeDescriptor
- Shape mismatch handling: tile shape transitions, shape incompatibility, and padding mismatch resolution
- End-to-end walkthroughs: 11-node gated MLP (LLaMA-7B) and DeepSeek V3 MoE expert matmul

---

## The Shape Propagation Protocol

### Design

Each operation in TT-Blaze transforms input tensors into output tensors. The shape propagation protocol formalizes this transformation at the ShapeDescriptor level:

```python
# PROPOSED -- shape propagation protocol
class ShapeInferenceRule:
    """Base class for per-op shape inference rules.

    Each rule takes input ShapeDescriptors and produces an output ShapeDescriptor.
    The rule encodes the operation's mathematical shape transformation.
    """

    @staticmethod
    def infer_output_shape(
        input_descriptors: dict[str, ShapeDescriptor],
        op_params: dict[str, Any] | None = None,
    ) -> ShapeDescriptor:
        raise NotImplementedError
```

### The Inference Registry

Shape inference rules are registered per op type:

```python
# PROPOSED -- shape inference registry
SHAPE_INFERENCE_REGISTRY: dict[str, ShapeInferenceRule] = {
    "matmul":               MatmulShapeRule(),
    "matmul_cb":            MatmulShapeRule(),
    "kn_sliced_matmul":     MatmulShapeRule(),
    "dram_streaming_matmul": MatmulShapeRule(),
    "add":                  ElementwiseBinaryShapeRule(),
    "mul":                  ElementwiseBinaryShapeRule(),
    "sub":                  ElementwiseBinaryShapeRule(),
    "eltwise_mul":          ElementwiseBinaryShapeRule(),
    "residual_add":         ElementwiseBinaryShapeRule(),
    "silu":                 ElementwiseUnaryShapeRule(),
    "relu":                 ElementwiseUnaryShapeRule(),
    "gelu":                 ElementwiseUnaryShapeRule(),
    "rmsnorm":              NormShapeRule(),
    "padded_rmsnorm":       NormShapeRule(),
    "broadcast_rmsnorm":    NormShapeRule(),
    "layernorm":            NormShapeRule(),
    "reduce":               ReductionShapeRule(),
    "gated_reduce":         GatedReduceShapeRule(),
    "gated_local_reduce":   GatedReduceShapeRule(),
    "softmax":              ElementwiseUnaryShapeRule(),
    "sdpa":                 SDPAShapeRule(),
    "mcast":                DataMovementShapeRule(),
    "gather":               DataMovementShapeRule(),
}

def infer_output_shape(
    op_type: str,
    input_descriptors: dict[str, ShapeDescriptor],
    op_params: dict[str, Any] | None = None,
) -> ShapeDescriptor:
    """Look up and apply the shape inference rule for an op.

    Falls back to identity rule (output == first input) for unknown ops.
    """
    rule = SHAPE_INFERENCE_REGISTRY.get(op_type)
    if rule is None:
        warnings.warn(f"No shape inference rule for op '{op_type}'. Assuming identity.")
        return next(iter(input_descriptors.values()))
    return rule.infer_output_shape(input_descriptors, op_params)
```

---

## Shape Inference Rules for Common Ops

### Matmul: Contract K, Produce [M, N]

Matmul is the most important shape rule because it changes the output shape and has strict compatibility requirements on its inputs.

```python
# PROPOSED -- matmul shape inference
class MatmulShapeRule(ShapeInferenceRule):
    """Matmul: in0=[..., M, K] x in1=[..., K, N] -> out=[..., M, N]

    The K dimension (last dim of in0, second-to-last of in1) contracts.
    Batch dimensions are broadcast following NumPy rules.
    """

    @staticmethod
    def infer_output_shape(
        input_descriptors: dict[str, ShapeDescriptor],
        op_params: dict[str, Any] | None = None,
    ) -> ShapeDescriptor:
        in0 = input_descriptors["in0"]
        in1 = input_descriptors["in1"]

        # Validate K dimension compatibility
        K_in0 = in0.logical_shape[-1]
        K_in1 = in1.logical_shape[-2]
        if K_in0 != K_in1:
            raise ShapeMismatchError(
                f"Matmul K dimension mismatch: "
                f"in0 has K={K_in0}, in1 has K={K_in1}."
            )

        M = in0.logical_shape[-2]
        N = in1.logical_shape[-1]
        batch_out = _broadcast_batch_dims(in0.logical_shape[:-2], in1.logical_shape[:-2])
        logical_out = batch_out + (M, N)
        tile = in0.tile_shape
        padded_out = _pad_to_tiles(logical_out, tile)
        grid_out = _compute_tile_grid(padded_out, tile)

        return ShapeDescriptor(
            logical_shape=logical_out,
            tile_shape=tile,
            padded_shape=padded_out,
            tile_grid_shape=grid_out,
            total_tiles=_total_tiles(grid_out),
            data_format=in0.data_format,
            page_size=in0.page_size,
            padding_strategy="ZERO" if logical_out != padded_out else "NONE",
            padding_fill=0.0,
            source="inferred",
            op_hint="matmul.out",
        )
```

#### Matmul Validation Details

| Check | Condition | Error |
|-------|-----------|-------|
| K dimension match | `in0.logical[-1] != in1.logical[-2]` | `ShapeMismatchError` |
| Tile shape compatibility | `in0.tile_shape != in1.tile_shape` | Warning (weights may use different tile) |
| Minimum rank | `len(in0.logical) < 2` | `ValueError("Matmul requires rank >= 2")` |
| Batch broadcast | Dimensions not broadcast-compatible | `ShapeMismatchError` |

> **Warning:** In the existing Blaze codebase, `Matmul.emit()` validates K-dimension compatibility via `if k_num_tiles <= 0 or in1_cb.num_pages % k_num_tiles:`. TensorAdapter validates at both the logical level (element count) and the tile level (tile count divisibility).

### Elementwise Ops: Preserve Shape (Unary and Binary)

```python
# PROPOSED -- elementwise unary shape inference
class ElementwiseUnaryShapeRule(ShapeInferenceRule):
    """Unary elementwise ops (silu, relu, gelu): output shape == input shape."""

    @staticmethod
    def infer_output_shape(input_descriptors, op_params=None):
        desc = next(iter(input_descriptors.values()))
        return ShapeDescriptor(
            logical_shape=desc.logical_shape,
            tile_shape=desc.tile_shape,
            padded_shape=desc.padded_shape,
            tile_grid_shape=desc.tile_grid_shape,
            total_tiles=desc.total_tiles,
            data_format=desc.data_format,
            page_size=desc.page_size,
            padding_strategy=desc.padding_strategy,
            padding_fill=desc.padding_fill,
            source="inferred",
            op_hint="eltwise.out",
        )
```

```python
# PROPOSED -- elementwise binary shape inference
class ElementwiseBinaryShapeRule(ShapeInferenceRule):
    """Binary elementwise ops (add, mul, sub): broadcast inputs, preserve shape."""

    @staticmethod
    def infer_output_shape(input_descriptors, op_params=None):
        descs = list(input_descriptors.values())
        in0, in1 = descs[0], descs[1]
        logical_out = _broadcast_shapes(in0.logical_shape, in1.logical_shape)
        tile = in0.tile_shape
        padded_out = _pad_to_tiles(logical_out, tile)
        grid_out = _compute_tile_grid(padded_out, tile)

        return ShapeDescriptor(
            logical_shape=logical_out, tile_shape=tile,
            padded_shape=padded_out, tile_grid_shape=grid_out,
            total_tiles=_total_tiles(grid_out),
            data_format=in0.data_format, page_size=in0.page_size,
            padding_strategy=in0.padding_strategy, padding_fill=in0.padding_fill,
            source="inferred", op_hint="eltwise.out",
        )
```

### Broadcast Ops: Expand Dimensions

Broadcast operations (including mcast) replicate data along dimensions of size 1. The output shape has the expanded dimension, but the CB holds only one tile for the broadcast source dimension.

```python
# PROPOSED -- Broadcast shape inference
def infer_broadcast(
    inputs: dict[str, ShapeDescriptor],
    target_shape: tuple[int, ...],
) -> ShapeDescriptor:
    """Broadcast: expand dimensions of size 1 to match target.

    The output ShapeDescriptor has the expanded logical shape,
    but the CB configuration for the broadcast source uses
    num_pages=1 for broadcast dimensions (tile reused by kernel).
    """
    desc = inputs["in0"]
    logical = _broadcast_shapes(desc.logical_shape, target_shape)
    tile = desc.tile_shape
    padded = _pad_to_tiles(logical, tile)
    grid = _compute_tile_grid(padded, tile)

    return ShapeDescriptor(
        logical_shape=logical, tile_shape=tile,
        padded_shape=padded, tile_grid_shape=grid,
        total_tiles=_total_tiles(grid),
        page_size=desc.page_size, data_format=desc.data_format,
        source="inferred", op_hint="broadcast.out",
    )
```

> **Warning:** Broadcast inference must distinguish between (a) expanding a size-1 dimension to size N (NumPy broadcasting) and (b) physically multicasting data across cores (mcast). Broadcasting changes the logical shape; multicasting changes the core ranges without changing the shape. The DataMovementShapeRule handles multicasting; the broadcast inference function handles logical dimension expansion.

### Reduction Ops: Collapse Dimension

```python
# PROPOSED -- reduction shape inference
class ReductionShapeRule(ShapeInferenceRule):
    """Reduction ops: collapse the reduced dimension to size 1 (keepdim=True)."""

    @staticmethod
    def infer_output_shape(input_descriptors, op_params=None):
        desc = next(iter(input_descriptors.values()))
        reduce_dim = op_params.get("reduce_dim", -1) if op_params else -1
        rank = len(desc.logical_shape)
        if reduce_dim < 0:
            reduce_dim = rank + reduce_dim

        logical_out = list(desc.logical_shape)
        logical_out[reduce_dim] = 1
        logical_out = tuple(logical_out)

        tile = desc.tile_shape
        padded_out = _pad_to_tiles(logical_out, tile)
        grid_out = _compute_tile_grid(padded_out, tile)

        return ShapeDescriptor(
            logical_shape=logical_out, tile_shape=tile,
            padded_shape=padded_out, tile_grid_shape=grid_out,
            total_tiles=_total_tiles(grid_out),
            data_format=desc.data_format, page_size=desc.page_size,
            padding_strategy="ZERO" if logical_out != padded_out else "NONE",
            padding_fill=0.0, source="inferred", op_hint="reduce.out",
        )
```

### Gated Reduce, Normalization, Data Movement, and SDPA

```python
# PROPOSED -- gated reduce
class GatedReduceShapeRule(ShapeInferenceRule):
    """GatedReduce: SiLU(gate) * up. Both inputs must have matching shapes."""
    @staticmethod
    def infer_output_shape(input_descriptors, op_params=None):
        descs = list(input_descriptors.values())
        in0, in1 = descs[0], descs[1]
        if in0.logical_shape != in1.logical_shape:
            raise ShapeMismatchError(
                f"GatedReduce inputs must match: {in0.logical_shape} != {in1.logical_shape}"
            )
        return ShapeDescriptor(
            logical_shape=in0.logical_shape, tile_shape=in0.tile_shape,
            padded_shape=in0.padded_shape, tile_grid_shape=in0.tile_grid_shape,
            total_tiles=in0.total_tiles,
            data_format=in0.data_format, page_size=in0.page_size,
            padding_strategy=in0.padding_strategy, padding_fill=in0.padding_fill,
            source="inferred", op_hint="gated_reduce.out",
        )

# PROPOSED -- normalization
class NormShapeRule(ShapeInferenceRule):
    """RMSNorm/LayerNorm: preserve shape, may change tile for norm compute.

    The input may arrive with (32, 32) tiles from a preceding matmul,
    but norm ops use (1, 32) or (16, 32) tiles internally.
    """
    @staticmethod
    def infer_output_shape(input_descriptors, op_params=None):
        desc = next(iter(input_descriptors.values()))
        logical = desc.logical_shape
        tile = _select_norm_tile_with_limit(logical[-1])
        padded = _pad_to_tiles(logical, tile)
        grid = _compute_tile_grid(padded, tile)
        return ShapeDescriptor(
            logical_shape=logical, tile_shape=tile,
            padded_shape=padded, tile_grid_shape=grid,
            total_tiles=_total_tiles(grid),
            data_format=desc.data_format,
            page_size=_compute_page_size(tile, desc.data_format),
            padding_strategy=desc.padding_strategy, padding_fill=desc.padding_fill,
            source="inferred", op_hint="rmsnorm.out",
        )

# PROPOSED -- data movement
class DataMovementShapeRule(ShapeInferenceRule):
    """Mcast/Gather: shape preserved, data moves between cores."""
    @staticmethod
    def infer_output_shape(input_descriptors, op_params=None):
        desc = next(iter(input_descriptors.values()))
        return ShapeDescriptor(
            logical_shape=desc.logical_shape, tile_shape=desc.tile_shape,
            padded_shape=desc.padded_shape, tile_grid_shape=desc.tile_grid_shape,
            total_tiles=desc.total_tiles,
            data_format=desc.data_format, page_size=desc.page_size,
            padding_strategy=desc.padding_strategy, padding_fill=desc.padding_fill,
            source="inferred", op_hint="mcast.out",
        )

# PROPOSED -- SDPA
class SDPAShapeRule(ShapeInferenceRule):
    """SDPA: Q=[B,H,S,D], K=[B,H,S,D], V=[B,H,S,D] -> out=[B,H,S,D]."""
    @staticmethod
    def infer_output_shape(input_descriptors, op_params=None):
        q = input_descriptors.get("q") or input_descriptors.get("in0")
        return ShapeDescriptor(
            logical_shape=q.logical_shape, tile_shape=q.tile_shape,
            padded_shape=q.padded_shape, tile_grid_shape=q.tile_grid_shape,
            total_tiles=q.total_tiles,
            data_format=q.data_format, page_size=q.page_size,
            padding_strategy="ZERO", padding_fill=0.0,
            source="inferred", op_hint="sdpa.out",
        )
```

---

## TypedHandle: Carrying Shape Through emit() Chains

### How TypedHandle Enables Automatic Propagation

In the Composition API path, ops are chained via `emit()` calls that pass `CBHandle` objects. `TypedHandle` wraps `CBHandle` with `ShapeDescriptor`, enabling the next op to extract shape information without manual specification:

```python
# PROPOSED -- TypedHandle propagation pattern
# Step 1: Adapt activation
act_typed = adapter.adapt(act_tensor, op_hint="matmul.in0", fused_program=f)

# Step 2: Emit matmul (returns raw CBHandle)
mm_out_handle = Matmul.emit(f, act_typed.handle, wt_typed.handle, prefix="mm")

# Step 3: Infer output shape from input shapes
mm_out_shape = infer_output_shape("matmul", {
    "in0": act_typed.shape,
    "in1": wt_typed.shape,
})

# Step 4: Record and wrap
tracker.record(mm_out_handle, mm_out_shape)
mm_out_typed = TypedHandle(handle=mm_out_handle, shape=mm_out_shape)

# Step 5: Next op reads shape from TypedHandle
silu_typed = adapter.adapt(mm_out_typed, op_hint="silu.input", fused_program=f)
```

### ShapeTracker Integration

The `ShapeTracker` stores the mapping from `CBHandle` identity to `ShapeDescriptor`:

```python
# PROPOSED -- ShapeTracker propagation
class ShapeTracker:
    def __init__(self):
        self._handle_to_shape: dict[int, ShapeDescriptor] = {}

    def record(self, handle: CBHandle, shape: ShapeDescriptor) -> None:
        self._handle_to_shape[id(handle)] = shape

    def lookup(self, handle) -> ShapeDescriptor | None:
        if isinstance(handle, TypedHandle):
            return handle.shape
        return self._handle_to_shape.get(id(handle))

    def propagate_through_op(
        self,
        op_type: str,
        input_handles: dict[str, CBHandle | TypedHandle],
        output_handle: CBHandle,
        op_params: dict | None = None,
    ) -> TypedHandle:
        """Infer output shape and record it for the output handle."""
        input_descs = {}
        for port, handle in input_handles.items():
            desc = self.lookup(handle)
            if desc is None:
                raise ShapePropagationError(
                    f"Input port '{port}' has no ShapeDescriptor."
                )
            input_descs[port] = desc

        output_desc = infer_output_shape(op_type, input_descs, op_params)
        self.record(output_handle, output_desc)
        return TypedHandle(handle=output_handle, shape=output_desc)
```

### Why CBHandle Cannot Carry Shape Alone

`CBHandle` has `num_pages` and `page_size` but not the original logical shape, tile grid breakdown, or padding metadata:

```python
# EXISTING -- CBHandle after matmul
mm_handle.num_pages   # 344 (but: is this K tiles? N tiles? total tiles?)
mm_handle.page_size   # 2048 (but: which tile shape produced this?)
```

Without ShapeDescriptor, a downstream op cannot determine whether `num_pages=344` represents 344 K-dimension tiles (for another matmul's input) or 344 N-dimension tiles (for an elementwise op). The ShapeDescriptor disambiguates:

```python
# PROPOSED -- With ShapeDescriptor
mm_shape.tile_grid_shape  # (1, 344) -- 1 row tile, 344 column tiles
mm_shape.logical_shape    # (1, 11008) -- the actual output dimensions
```

---

## Handling Shape-Dependent CT Args

### The CT Arg Derivation Table

| CT Arg | Op | ShapeDescriptor Derivation |
|--------|----|-----------------------------|
| `k_num_tiles` | Matmul | `in0.tile_grid_shape[-1]` |
| `out_w_per_core` | Matmul | `in1.total_tiles // in0.tile_grid_shape[-1]` |
| `num_tiles` | GatedReduce | `input.total_tiles` |
| `width` | RMSNorm | `input.tile_grid_shape[-1]` |
| `input_bytes` | PaddedRMSNorm | `input.logical_shape[-1] * DTYPE_BYTES[format]` |
| `padded_bytes` | PaddedRMSNorm | `input.padded_shape[-1] * DTYPE_BYTES[format]` |

```python
# PROPOSED -- CT arg derivation from ShapeDescriptor
def derive_ct_args_for_op(
    op_type: str,
    input_descs: dict[str, ShapeDescriptor],
    output_desc: ShapeDescriptor,
) -> dict[str, int]:
    """Derive shape-dependent CT args for a given op."""
    args = {}

    if op_type in ("matmul", "matmul_cb", "kn_sliced_matmul"):
        in0 = input_descs.get("in0")
        in1 = input_descs.get("in1")
        if in0 and in1:
            k_num_tiles = in0.tile_grid_shape[-1]
            args["k_num_tiles"] = k_num_tiles
            if k_num_tiles > 0:
                args["out_w_per_core"] = in1.total_tiles // k_num_tiles

    elif op_type in ("rmsnorm", "padded_rmsnorm", "broadcast_rmsnorm"):
        inp = next(iter(input_descs.values()))
        args["width"] = inp.tile_grid_shape[-1]
        if op_type == "padded_rmsnorm":
            args["input_bytes"] = inp.logical_shape[-1] * 2
            args["padded_bytes"] = inp.padded_shape[-1] * 2

    elif op_type in ("reduce", "gated_local_reduce"):
        args["num_tiles"] = next(iter(input_descs.values())).total_tiles

    elif op_type == "gated_reduce":
        args["num_tiles"] = next(iter(input_descs.values())).tile_grid_shape[-1]

    return args
```

### Cross-Validation With Manual Values

> *Design Principle P3 (Validate at Each Lowering Step):* During the transition period (when both manual and automatic CT arg derivation coexist), the system cross-checks auto-derived values against manual ones:

```python
# PROPOSED -- CT arg cross-validation
def _validate_ct_args(auto_args: dict, manual_args: dict, op_name: str):
    for key in auto_args:
        if key in manual_args and auto_args[key] != manual_args[key]:
            raise CTArgMismatchError(
                f"CT arg mismatch for op '{op_name}', arg '{key}': "
                f"auto-derived={auto_args[key]}, manual={manual_args[key]}."
            )
```

> **Warning:** CT arg values must be exact integers. A k_num_tiles value of 127 instead of 128 means the kernel's inner loop executes one fewer iteration, producing a partial result. There is no tolerance for approximation.

---

## Edge Cases in Shape Propagation

### Tile Shape Transitions at Op Boundaries

When a matmul (32x32 tiles) feeds into an RMSNorm (1x32 or 16x32 tiles), the ShapeDescriptor must be recomputed with the new tile shape:

```
Matmul output:   logical=[1, 4096], tile=(32, 32), padded=[32, 4096], grid=[1, 128]
RMSNorm input:   logical=[1, 4096], tile=(16, 32), padded=[16, 4096], grid=[1, 128]
```

The `with_tile()` method handles this transition. The data in L1 does not move -- the tiles are reinterpreted by the consumer's compute kernel.

> **Warning:** A tile shape transition may change the padded shape. A `[1, 4096]` tensor with (32, 32) tiles has padded height 32, but with (16, 32) tiles has padded height 16, and with (1, 32) tiles has padded height 1. CB sizes change correspondingly.

### Ops That Reshape (View/Permute)

Reshape operations change the logical shape without moving data:

```
Input:   logical=[1, 4096], tile=(32, 32), padded=[32, 4096], grid=[1, 128]
         reshape to [32, 128]
Output:  logical=[32, 128], tile=(32, 32), padded=[32, 128], grid=[1, 4]
```

Total element count must be preserved. The tile grid changes because elements are reinterpreted in different dimensions.

### Ops That Transpose

Transpose swaps the last two dimensions:

```
Input:   logical=[4096, 11008], tile=(32, 32), grid=[128, 344]
Output:  logical=[11008, 4096], tile=(32, 32), grid=[344, 128]
```

---

## Handling Shape Mismatches

When a producer's output shape does not match what the consumer expects, there are three categories:

### Category 1: Tile Shape Mismatch (Recoverable)

Producer used 32x32 tiles but consumer expects 1x32 tiles. Resolution: create a new ShapeDescriptor via `with_tile()`. The data in L1 is reinterpreted, not moved.

### Category 2: Shape Incompatibility (Error)

Producer outputs `[1, 4096]` but consumer expects `[1, 11008]`. This is a genuine shape error -- the tensors are incompatible.

### Category 3: Padding Mismatch (Recoverable)

Producer's zero-padding does not match what the consumer needs (e.g., NEG_INF for softmax). Resolution: insert an implicit re-padding node.

> **Warning:** Implicit re-padding adds a compute step. For performance-critical paths, the developer should arrange ops so that padding strategies are consistent through the chain, or use the PaddingPolicy registry (Chapter 5).

---

## Propagation Walkthrough: 11-Node Gated MLP (LLaMA-7B)

Tracing shape propagation through the production gated MLP from `blaze/_gated_mlp.py`:

```
Hidden dim = 4096, Intermediate dim = 11008
All weights bfloat8_b, activations bfloat16, (32, 32) tiles

Node 1: mcast(activation)
  Input:  SD(logical=(1, 4096), padded=(32, 4096), grid=(1, 128))
  Rule:   DataMovement (preserve)
  Output: SD(logical=(1, 4096), grid=(1, 128))

Node 2: kn_sliced_matmul(activation, gate_up_weights)
  in0: SD(logical=(1, 4096), grid=(1, 128))
  in1: SD(logical=(4096, 11008), grid=(128, 344))
  Rule: Matmul (contract K=4096)
  K check: 4096 == 4096 -> valid
  Output: SD(logical=(1, 11008), padded=(32, 11008), grid=(1, 344))
  CT args: k_num_tiles=128, out_w_per_core=344

Node 3: kn_sliced_matmul(activation, gate_up_weights) [on b_cores]
  Same shapes as Node 2
  Output: SD(logical=(1, 11008), grid=(1, 344))

Nodes 4-5: gather(gate), gather(up)
  Rule: DataMovement (preserve)
  Output: SD(logical=(1, 11008)) each

Node 6: gated_reduce(gate, up)
  in0: SD(logical=(1, 11008))
  in1: SD(logical=(1, 11008))
  Rule: GatedReduce (shapes must match -> OK)
  Output: SD(logical=(1, 11008), grid=(1, 344))

Node 7: mcast(reduced)
  Output: SD(logical=(1, 11008))

Node 8: matmul(reduced, down_weights)
  in0: SD(logical=(1, 11008), grid=(1, 344))
  in1: SD(logical=(11008, 4096), grid=(344, 128))
  Rule: Matmul (contract K=11008)
  Output: SD(logical=(1, 4096), grid=(1, 128))
  CT args: k_num_tiles=344, out_w_per_core=128

Node 9: mcast(bias)
  Output: SD(logical=(1, 4096))

Node 10: residual_add(matmul_out, bias)
  Rule: ElementwiseBinary (broadcast, then preserve)
  Output: SD(logical=(1, 4096))

Node 11: gather(result)
  Output: SD(logical=(1, 4096))
```

All 11 nodes successfully propagate shape metadata, and the CT args for both matmul operations are derived automatically.

---

## Propagation Walkthrough: DeepSeek V3 MoE Expert Matmul

```
Hidden dim = 7168, MoE expert width = 2048

Step 1: Expert gate matmul
  in0: SD(logical=(1, 7168), tile=(32,32), grid=(1, 224))
  in1: SD(logical=(7168, 2048), tile=(32,32), grid=(224, 64))

  K check: 7168 == 7168 -> valid

  Output:
    M = 1, N = 2048
    logical = (1, 2048)
    padded  = (32, 2048)
    grid    = (1, 64)
    total   = 64

  CT args:
    k_num_tiles   = 224
    out_w_per_core = 14336 / 224 = 64

Step 2: SiLU activation
  Input: SD(logical=(1, 2048), grid=(1, 64))
  Rule: ElementwiseUnary (preserve)
  Output: SD(logical=(1, 2048), grid=(1, 64))

Step 3: Expert up matmul
  in0: SD(logical=(1, 7168), grid=(1, 224))
  in1: SD(logical=(7168, 2048), grid=(224, 64))
  Output: SD(logical=(1, 2048), grid=(1, 64))

Step 4: Elementwise multiply (SiLU(gate) * up)
  in0: SD(logical=(1, 2048), grid=(1, 64))
  in1: SD(logical=(1, 2048), grid=(1, 64))
  Rule: ElementwiseBinary (shapes match, preserve)
  Output: SD(logical=(1, 2048), grid=(1, 64))

Step 5: Down projection
  in0: SD(logical=(1, 2048), grid=(1, 64))
  in1: SD(logical=(2048, 7168), grid=(64, 224))
  Rule: Matmul (contract K=2048)
  Output: SD(logical=(1, 7168), grid=(1, 224))
  CT args: k_num_tiles=64, out_w_per_core=224
```

Compared to LLaMA-7B's intermediate dim of 11008 (344 tiles, 43 per 8 cores, prime-number subblock issue), DeepSeek V3's 2048 (64 tiles) is much more amenable to even core distribution.

---

## Shape Rules Reference Table

| Op Category | Op Names | Output Shape Rule | Derivable CT Args |
|------------|----------|-------------------|------------------------|
| Matmul | matmul, matmul_cb, kn_sliced_matmul, dram_streaming_matmul | `[..., M, K] x [..., K, N] -> [..., M, N]` | `k_num_tiles`, `out_w_per_core` |
| Elementwise binary | add, mul, sub, eltwise_mul, residual_add | Broadcast, then preserve | (none) |
| Elementwise unary | silu, relu, gelu, softmax | Preserve | (none) |
| Broadcast | mcast (logical expansion) | Expand size-1 dims | (none) |
| Reduction | reduce, gated_local_reduce | Collapse dim to 1 | `num_tiles` |
| Gated reduce | gated_reduce | Preserve (inputs must match) | `num_tiles` |
| Normalization | rmsnorm, padded_rmsnorm, layernorm | Preserve (may change tile) | `width`, `input_bytes`, `padded_bytes` |
| Data movement | mcast, gather | Preserve | (none shape-dependent) |
| Attention | sdpa | Output matches Q shape | Multiple blocking params |
| Reshape / Transpose | view, reshape, transpose | User-specified / swap last two dims | (recompute all) |

---

## Key Takeaways

- Shape propagation eliminates manual CT arg computation in op chains. Each op declares how it transforms shapes via a registered `ShapeInferenceRule`, and the `ShapeTracker` records the output descriptor for the next op to consume.
- Matmul is the most complex rule: it validates K-dimension compatibility, contracts K, computes the output from M and N, broadcasts batch dimensions, and derives `k_num_tiles` and `out_w_per_core` -- two of the most error-prone manual calculations in existing Blaze code.
- TypedHandle bridges existing `emit()` code (which expects CBHandle) and TensorAdapter's shape system (which expects ShapeDescriptor). Through `__getattr__` delegation, TypedHandle is a drop-in replacement for CBHandle in all existing call sites.
- Three categories of shape mismatch are handled: tile shape transitions (recoverable via `with_tile()`), shape incompatibility (error with diagnostic), and padding strategy mismatch (recoverable via implicit re-padding).
- The broadcast inference rule handles logical dimension expansion (size-1 dims expanding to match a target), while data movement rules handle physical multicasting. These are separate concepts that must not be conflated.
- The 11-node LLaMA-7B gated MLP and DeepSeek V3 MoE expert walkthroughs demonstrate that shape tracking handles production-scale fused ops, deriving all shape-dependent CT args automatically.

## Source Files

- `blaze/context.py` -- `FusionResult` (extended with `shape` field), `FusionContext.add_op()`
- `blaze/ct_args.py` -- `CTArgSpec`, `CTArgSchema` (CT arg type system)
- `blaze/ops/matmul/op.py` -- `Matmul.emit()` (manual K-dimension validation and CT arg computation)
- `blaze/ops/padded_rmsnorm/op.py` -- `PaddedRMSNorm.emit()` (manual padding CT args)
- `blaze/_gated_mlp.py` -- `build_gated_mlp_graph()` (11-node graph for the worked example)
- `blaze/graph.py` -- `Edge` (extended with `shape_descriptor`), `TensorPort`

---

← Previous: [Chapter 3: The TensorAdapter Architecture](../ch03_tensor_adapter/) | Next: [Chapter 5: Transparent Padding Management](../ch05_padding/) →
