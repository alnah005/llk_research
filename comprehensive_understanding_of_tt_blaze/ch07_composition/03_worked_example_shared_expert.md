# 07.03 -- Worked Example: SharedExpert

> **Navigation:** [Previous: Advanced Patterns](./02_advanced_patterns.md) | [Composition Basics](./01_composition_basics.md) | [Next: Worked Example: MoE](./04_worked_example_moe.md)

---

## Overview

`SharedExpert` (in `blaze/ops/shared_expert/op.py`) is a FusedOp that chains four building blocks into a single fused kernel:

```
KNMatmul -> GatedReduce -> Mcast -> DownProj
```

The DownProj itself is a nested FusedOp containing:

```
Matmul -> Mcast(bias) -> ResidualAdd -> Gather
```

So the full pipeline is:

```
KNSlicedMatmul -> GatedReduce -> Mcast -> Matmul -> Mcast(bias) -> ResidualAdd -> Gather
```

This replaces what would be approximately 979 lines of manual TT-Metal wiring with a 67-line FusedOp (115 lines total including class definition and compose()).

---

## The Full compose() and emit()

Here is the complete source of SharedExpert:

```python
class SharedExpert(FusedOp):
    """SharedExpert: composed from KNMatmul + GatedReduce + DownProj building blocks."""

    name: str = "shared_expert"
    math_fidelity: str = "LoFi"
    math_approx_mode: bool = True

    activation: Input = Input()
    gate_up_weights: Input = Input()
    down_weights: Input = Input()
    bias: Input = Input()
    output: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(
            f,
            tensors["activation"],
            tensors["gate_up_weights"],
            tensors["down_weights"],
            tensors["bias"],
            output,
            prefix=user_args.get("prefix", "shared_expert"),
            ab_coords=user_args.get("ab_grids"),
            down_coords=user_args.get("down_coords"),
            pop_shared_act=user_args.get("pop_shared_act", True),
        )

    @staticmethod
    def emit(
        f,
        activation,
        gate_up_weights,
        down_weights,
        bias,
        output,
        *,
        prefix: str = "shared_expert",
        ab_coords=None,
        down_coords=None,
        pop_shared_act=True,
        bias_dst_tile_info=None,
        bias_dst_num_pages=None,
        skip_bias_add: bool = False,
    ):
        """Compose SharedExpert pipeline on a FusedProgram."""
        # ...
```

---

## Line-by-Line Walkthrough of emit()

### Step 0: Resolve Tile Info

```python
if isinstance(activation, ttnn.Tensor):
    ti = TileInfo.from_tensor(activation)
else:
    tile = ttnn.Tile([activation.tile_desc.height, activation.tile_desc.width])
    ti = TileInfo(
        tile=tile,
        data_format=activation.data_format,
        size=activation.page_size,
        desc=activation.tile_desc,
    )
```

The activation can arrive either as a raw `ttnn.Tensor` (when SharedExpert is compiled standalone) or as a `CBHandle` (when embedded inside MoE). `TileInfo` captures the tile shape, dtype, page size, and tile descriptor in a single object. This is passed downstream to MicroOps that need to know the activation geometry without caring about its origin.

**Why this matters:** Every subsequent CB allocation derives its format from `ti`. Getting the tile info wrong here cascades into wrong page sizes in every downstream scratch CB.

### Step 1: KNSlicedMatmul (Gate/Up Projection)

```python
gu = KNMatmul.emit(
    f,
    activation,
    gate_up_weights,
    prefix=BlazeOp.child_prefix(prefix, "gu"),
    ab_coords=ab_coords,
    k_half_split=False,
    pop_act=pop_shared_act,
)
```

This is the heaviest phase. KNMatmul splits the gate and up weight matrices across two disjoint core groups (A-grid and B-grid). Each core performs a K-sliced matmul, producing partial sums that feed into the next phase.

**Prefix:** `shared_expert__gu`. All CT args inside KNMatmul will be prefixed with this, e.g., `shared_expert__gu.in0`, `shared_expert__gu.k_num_tiles`.

**Return value:** `KNMatmulOutput` -- a dataclass carrying not just the output CBHandle but also derived parallel configuration:

```python
@dataclass
class KNMatmulOutput:
    handle: CBHandle       # output CB
    k_parallel: int        # number of K-slices per branch
    n_parallel: int        # number of N-tiles per K-slice
    num_per_branch: int    # cores per A/B branch
    a_grid: object         # CoreRangeSet for gate branch
    b_grid: object         # CoreRangeSet for up branch
    gate_handle: CBHandle  # shadow graph handle for gate
    up_handle: CBHandle    # shadow graph handle for up
```

This is an example of emit() returning structured metadata beyond a bare CBHandle. The downstream GatedReduce needs `k_parallel`, `n_parallel`, `a_grid`, and `b_grid` to configure its gather destinations.

### Step 2: GatedReduce

```python
reduced = GatedReduce.emit(
    f,
    gu.handle,
    prefix=BlazeOp.child_prefix(prefix, "gated_reduce"),
    k_parallel=gu.k_parallel,
    n_parallel=gu.n_parallel,
    tile_info=ti,
    a_grid=gu.a_grid,
    b_grid=gu.b_grid,
    gate_handle=gu.gate_handle,
    up_handle=gu.up_handle,
)
```

GatedReduce gathers partial sums from the A-grid (gate) and B-grid (up) cores, applies `SiLU(gate) * up` element-wise, and writes the result to a single mcast-source CB on the sender core.

**Prefix:** `shared_expert__gated_reduce`.

**CBHandle chain:** `gu.handle` (the KNMatmul output) flows in as `gu_out_handle`. GatedReduce allocates gather destination CBs on the union of sender and matmul cores, configures the gather CT args, and returns a CBHandle pointing at the `mcast_src` scratch CB on the sender core.

**Key metadata flow:** `gu.k_parallel` and `gu.n_parallel` determine the gather tile counts. `gu.a_grid` and `gu.b_grid` determine which cores participate in each gather. This is why KNMatmul returns a structured output -- these values are not derivable from the CBHandle alone.

### Step 3: Mcast (Scatter Reduced Result)

```python
down_act = Mcast.emit(
    f,
    reduced,
    prefix=BlazeOp.child_prefix(prefix, "mcast"),
    sender_sem=None,
    dst_num_pages=gu.n_parallel,
    dst_tile_info=ti,
)
```

Multicasts the reduced activation from the sender core to all cores. The `dst_num_pages` override ensures the destination CB has the right page count for the downstream matmul (which expects `n_parallel` tiles per core, not the full activation width).

**Prefix:** `shared_expert__mcast`.

**CBHandle chain:** `reduced` (GatedReduce output, on sender core only) goes in. `down_act` comes out as a CBHandle for a scratch CB on `f.all_cores`.

**Semaphore allocation:** `sender_sem=None` causes Mcast.emit() to allocate a fresh mesh-global semaphore via `f.semaphore(f"{prefix}.sender")`.

### Step 4: DownProj (Down Projection + Bias + Gather)

```python
return DownProj.emit(
    f,
    down_act,
    down_weights,
    bias,
    output,
    prefix=BlazeOp.child_prefix(prefix, "down_proj"),
    mcast_sender_sem=None,
    core_coords=down_coords,
    bias_dst_tile_info=bias_dst_tile_info,
    bias_dst_num_pages=bias_dst_num_pages,
    skip_bias_add=skip_bias_add,
)
```

DownProj is itself a FusedOp that chains four micro-ops internally:

```python
# Inside DownProj.emit():
mm = DownProj.matmul(f, act_handle, weights_tensor,
                     prefix=BlazeOp.child_prefix(prefix, "matmul"), cores=matmul_cores)
res = Mcast.emit(f, bias_tensor,
                 prefix=BlazeOp.child_prefix(prefix, "mcast2"), sender_sem=mcast_sender_sem,
                 dst_tile_info=bias_dst_tile_info, dst_num_pages=bias_dst_num_pages)
added = ResidualAdd.emit(f, mm, res,
                         prefix=BlazeOp.child_prefix(prefix, "residual_add"), cores=matmul_cores,
                         skip_add=skip_bias_add)
return Gather.emit(f, added, output_tensor=output_tensor,
                   prefix=BlazeOp.child_prefix(prefix, "gather"), dst_cb=gather_dst)
```

**Prefix hierarchy:**
- `shared_expert__down_proj__matmul` -- the down-projection matmul
- `shared_expert__down_proj__mcast2` -- the bias mcast
- `shared_expert__down_proj__residual_add` -- element-wise add
- `shared_expert__down_proj__gather` -- gather to sender/output

**Output handling:** The `output` parameter is either a `ttnn.Tensor` (pre-allocated output) or a `CBHandle` (when SharedExpert is embedded in a larger composition like MoE). If it is a tensor, DownProj calls `f.cb_from_tensor(output_tensor)` to create a tensor-backed CB for the gather destination. If it is a CBHandle, the gather writes directly into the existing CB.

---

## CBHandle Chain Diagram

```
activation (Tensor or CBHandle)
    |
    v
[KNMatmul.emit]  prefix="shared_expert__gu"
    |  Allocates: act CB (from tensor or reuses handle),
    |             weight CBs (direct-address from OverlappedView),
    |             output scratch CB on a_grid + b_grid
    |  Returns: KNMatmulOutput { handle, k_parallel, n_parallel, a_grid, b_grid }
    |
    v
gu.handle (CBHandle, scratch on compute_grid)
    |
    v
[GatedReduce.emit]  prefix="shared_expert__gated_reduce"
    |  Allocates: gather dst CBs (on sender + matmul union),
    |             intermed scratch (2 pages),
    |             mcast_src scratch (on sender core)
    |  Consumes: gu.handle, gu.k_parallel, gu.n_parallel, gu.a_grid, gu.b_grid
    |  Returns: CBHandle for mcast_src (on sender core)
    |
    v
reduced (CBHandle, scratch on sender_grid)
    |
    v
[Mcast.emit]  prefix="shared_expert__mcast"
    |  Allocates: dst scratch (on all_cores, dst_num_pages=n_parallel)
    |  Semaphores: sender, receiver (mesh-global)
    |  Returns: CBHandle for dst (on all_cores)
    |
    v
down_act (CBHandle, scratch on all_cores)
    |
    v
[DownProj.emit]  prefix="shared_expert__down_proj"
    |
    +--[DownProj.matmul]  prefix="shared_expert__down_proj__matmul"
    |    |  Allocates: output scratch (on matmul_cores)
    |    |  Weight CB: direct-address from tensor
    |    |  Returns: CBHandle for mm_out (on matmul_cores)
    |    v
    |  mm (CBHandle, scratch on matmul_cores)
    |
    +--[Mcast.emit]  prefix="shared_expert__down_proj__mcast2"
    |    |  Allocates: dst scratch (on all_cores) for bias
    |    |  Returns: CBHandle for bias_dst (on all_cores)
    |    v
    |  res (CBHandle, scratch on all_cores)
    |
    +--[ResidualAdd.emit]  prefix="shared_expert__down_proj__residual_add"
    |    |  Allocates: output scratch (on matmul_cores)
    |    |  Returns: CBHandle for add_out (on matmul_cores)
    |    v
    |  added (CBHandle, scratch on matmul_cores)
    |
    +--[Gather.emit]  prefix="shared_expert__down_proj__gather"
         |  Allocates: dst CB (tensor-backed from output, or reuses CBHandle)
         |  Returns: CBHandle for gather_dst (on sender_grid)
         v
    final_output (CBHandle, tensor-backed or scratch)
```

---

## Key Observations

### 1. Tensor vs. CBHandle Polymorphism

SharedExpert.emit() accepts `activation` as either a `ttnn.Tensor` or a `CBHandle`. This is what makes it composable both as a standalone op and as a nested building block in MoE. The choice propagates down:

- **Standalone (tensor in):** KNMatmul calls `f.cb_from_tensor(activation)` to allocate an input CB.
- **Nested (CBHandle in):** KNMatmul reuses the existing CB handle from the upstream Mcast.

### 2. Metadata Flows Beyond CBHandle

`KNMatmulOutput` demonstrates that CBHandle alone is not always sufficient. The downstream GatedReduce needs grid topology (`a_grid`, `b_grid`) and parallel configuration (`k_parallel`, `n_parallel`) that are computed inside KNMatmul. The structured return type avoids recomputing these values.

### 3. Nested FusedOp Composition

DownProj is itself a FusedOp containing four MicroOps. When SharedExpert calls `DownProj.emit()`, the prefix hierarchy automatically namespaces all DownProj internals under `shared_expert__down_proj__*`. No special handling is needed -- the same child_prefix + cb_name conventions work at arbitrary nesting depth.

### 4. Output Flexibility

The `output` parameter to SharedExpert.emit() can be:
- A `ttnn.Tensor` -- DownProj calls `f.cb_from_tensor(output)` and the gather writes into that tensor. The compiler auto-wires it as program output.
- A `CBHandle` -- DownProj's gather writes into the existing CB. This is used when SharedExpert is a building block in MoE, where its output feeds into a downstream mcast before the cross-device reduction.

### 5. pop_shared_act Control

The `pop_shared_act` parameter controls whether KNMatmul pops the activation CB after reading it. When SharedExpert runs inside MoE, both the shared and routed expert branches read the same activation mcast. Setting `pop_shared_act=False` keeps the activation CB live for the routed expert's consumption.

---

> **Navigation:** [Previous: Advanced Patterns](./02_advanced_patterns.md) | [Composition Basics](./01_composition_basics.md) | [Next: Worked Example: MoE](./04_worked_example_moe.md)
