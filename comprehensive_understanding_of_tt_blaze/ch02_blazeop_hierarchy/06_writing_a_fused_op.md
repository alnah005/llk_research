# 06 -- Writing a FusedOp

A FusedOp composes multiple MicroOps into a single kernel dispatch. Instead of launching each op separately -- paying kernel launch overhead and transiting data through DRAM between ops -- a FusedOp chains them in L1. Data flows between ops through circular buffers without leaving the core. This section walks through the authoring process step by step, using DownProj and SharedExpert as concrete examples.

## When to Write a FusedOp

Write a `FusedOp` when:

- You need to chain multiple MicroOps into a single fused kernel (no dispatch overhead between stages)
- The same composition pattern appears in multiple places and should be reusable
- You want the graph API to treat the entire pipeline as a single node

Do not write a `FusedOp` when a single `MicroOp` suffices. The FusedOp is a composition tool, not a code organization tool.

## Step 1: Directory Layout

FusedOps follow the same directory convention as MicroOps:

```
blaze/ops/down_proj/
    __init__.py          # empty
    op.py                # Python class
```

The difference: there is **no `kernels/op.hpp`**. A FusedOp either:

- Has no kernel at all (the kernel is auto-generated from the shadow graph by `kernel_codegen.py`), or
- Provides a handwritten kernel via the `kernel` class attribute (uncommon; used when the auto-generated kernel needs manual optimization)

Most FusedOps use the auto-generated path, so no `kernels/` directory is needed.

## Step 2: The Python Class Declaration

```python
from ...blaze_op import BlazeOp, FusedOp, Input, Output

class DownProj(FusedOp):
    name: str = "down_proj"
    math_fidelity: str = "LoFi"

    input: Input = Input()
    weights: Input = Input()
    add_input: Input = Input()
    output: Output = Output()
```

Key differences from a MicroOp declaration:

- Inherits from `FusedOp`, not `MicroOp`
- No `ct_args`, `phase`, `op_class` -- these are not needed because the child MicroOps handle their own CT arg registration
- `kernel` is not set (defaults to `""`, meaning auto-generated)
- Ports describe the FusedOp's external interface, not the internal pipeline wiring

## Step 3: Pipeline Composition (compose and emit)

### The `compose()` Method

`compose()` is required by `FusedOp.__init_subclass__`. It bridges the compiler's calling convention to your pipeline logic:

```python
@classmethod
def compose(cls, f, tensors, output, user_args):
    act = Mcast.emit(f, tensors["input"], prefix="mcast")
    cls.emit(
        f, act,
        tensors["weights"],
        tensors["add_input"],
        output,
        mcast_sender_sem=0,
        core_coords=user_args.get("core_coords"),
    )
```

Keep `compose()` thin -- it should unpack the tensor dict, extract any user args, and delegate to `emit()`. In DownProj's case, `compose()` first emits an Mcast for the input activation, then delegates to `emit()`.

### The `emit()` Method

`emit()` is where the actual pipeline is wired. DownProj composes four stages:

```python
@staticmethod
def emit(f, act_handle, weights_tensor, bias_tensor,
         output, *, prefix="", mcast_sender_sem=None,
         core_coords=None, ...):

    mm = DownProj.matmul(f, act_handle, weights_tensor,
         prefix=BlazeOp.child_prefix(prefix, "matmul"),
         cores=matmul_cores)

    res = Mcast.emit(f, bias_tensor,
         prefix=BlazeOp.child_prefix(prefix, "mcast2"),
         sender_sem=mcast_sender_sem)

    added = ResidualAdd.emit(f, mm, res,
         prefix=BlazeOp.child_prefix(prefix, "residual_add"),
         cores=matmul_cores)

    return Gather.emit(f, added, output_tensor=...,
         prefix=BlazeOp.child_prefix(prefix, "gather"),
         dst_cb=gather_dst)
```

The pipeline reads as a linear dataflow:

$$\text{act} \xrightarrow{\text{matmul}} \text{mm} \xrightarrow[\text{bias}]{\text{residual\_add}} \text{added} \xrightarrow{\text{gather}} \text{output}$$

Each `emit()` call returns a `CBHandle` that becomes the input to the next stage. This is the CBHandle chaining pattern -- data flows through the Python composition layer exactly as it will flow through L1 circular buffers at runtime.

### Key Composition Points

1. **CBHandle chaining.** `mm` (matmul output) feeds into `ResidualAdd.emit()`. `res` (mcast output) also feeds into `ResidualAdd.emit()`. The `CBHandle` carries all the metadata (CB ID, page geometry, data format) that the next stage needs.

2. **`child_prefix()` scoping.** Each child op gets a scoped prefix: `"down_proj__matmul"`, `"down_proj__mcast2"`, etc. This prevents CT arg name collisions when multiple DownProj instances appear in the same fused program.

3. **Mixed building blocks.** `DownProj.matmul()` is a static helper method on the class (not a separate op) -- it uses a custom width-sharded matmul producing output tiles in $[1, 32]$ shape rather than the standard $[32, 32]$. `Mcast.emit()`, `ResidualAdd.emit()`, and `Gather.emit()` are standard MicroOp `emit()` calls. FusedOps can freely mix helper methods and child op calls.

4. **Flexible output handling.** `output` can be either a `ttnn.Tensor` or a `CBHandle`. When it is a tensor, DownProj allocates a CB via `f.cb_from_tensor(output)` and passes it to Gather. When it is a CBHandle (because DownProj is embedded in a larger pipeline), it passes the handle directly. This flexibility is what makes FusedOps composable.

## `child_prefix()` and `cb_name()` in Practice

These two naming methods prevent collisions when the same op type appears multiple times within a fused pipeline.

### `child_prefix(prefix, name)`

Produces: `f"{prefix}__{name}"` (double underscore delimiter, or just `name` if prefix is empty).

When DownProj composes a Matmul with `prefix = "down_proj"`:

```python
BlazeOp.child_prefix("down_proj", "matmul")
# => "down_proj__matmul"
```

All CT args inside that Matmul emit are prefixed with `"down_proj__matmul."`:
- `"down_proj__matmul.k_num_tiles"`
- `"down_proj__matmul.out_w_per_core"`
- `"down_proj__matmul.is_active"`

### `cb_name(prefix, name)`

Produces: `f"{prefix}___{name}"` (triple underscore delimiter).

```python
BlazeOp.cb_name("down_proj__matmul", "out")
# => "down_proj__matmul___out"
```

The different delimiter lengths prevent ambiguity: `"shared_expert__gu___out"` is unambiguously a CB named `"out"` belonging to the `"gu"` child of `"shared_expert"`.

### Nesting Depth

For deeply nested compositions like SharedExpert > DownProj > Matmul, the prefixes compose hierarchically:

```
SharedExpert prefix:         "shared_expert"
  KNMatmul child prefix:     "shared_expert__gu"
  GatedReduce child prefix:  "shared_expert__gated_reduce"
  Mcast child prefix:        "shared_expert__mcast"
  DownProj child prefix:     "shared_expert__down_proj"
    Matmul child prefix:     "shared_expert__down_proj__matmul"
      CB name for output:    "shared_expert__down_proj__matmul___out"
      CT arg name:           "shared_expert__down_proj__matmul.k_num_tiles"
    Mcast2 child prefix:     "shared_expert__down_proj__mcast2"
    ResidualAdd child:       "shared_expert__down_proj__residual_add"
    Gather child prefix:     "shared_expert__down_proj__gather"
```

This hierarchical naming means every CT arg and every CB in a fused kernel has a globally unique name, regardless of composition depth.

## Handling Inputs and Outputs at FusedOp Boundaries

FusedOps must handle the boundary between external tensors and internal CBHandles:

### Input Resolution

The first stage in the pipeline receives either a `ttnn.Tensor` (from compose's tensor dict) or a `CBHandle` (when called from another FusedOp's emit). The MicroOp's emit handles this automatically:

```python
if isinstance(src, CBHandle):
    src_handle = src
else:
    src_handle = f.cb_from_tensor(src)
```

### Output Wiring

The last stage must wire to the output tensor. DownProj handles this explicitly:

```python
if isinstance(output, CBHandle):
    output_tensor = None
    gather_dst = output
else:
    output_tensor = output
    gather_dst = f.cb_from_tensor(output_tensor)
```

When `output` is a `ttnn.Tensor`, the Gather op writes to a CB backed by that tensor. When it is a `CBHandle`, the Gather op writes to an existing CB (useful when the FusedOp is itself a building block inside a larger pipeline).

## Optional: Handwritten Kernel

Most FusedOps rely on auto-generated kernels. But if you need manual control over the C++ kernel, set the `kernel` class attribute:

```python
class MyCustomFused(FusedOp):
    name: str = "my_custom_fused"
    kernel: str = "blaze/ops/my_custom_fused/kernels/kernel.cpp"
```

When `kernel` is set, the `FusedOpConfig` registered during `register()` carries this path, and the compiler uses the handwritten kernel instead of running codegen.

A handwritten kernel is glue code that sequences Op structs:

```cpp
#include "blaze/kernels/ops.hpp"
#include "blaze/kernels/unified_kernels/kernel_op_api.hpp"

using ActMcast = blaze::Mcast::Op<ct_args::mcast>;
using Mm = blaze::Matmul::Op<ct_args::matmul>;
using Gath = blaze::Gather::Op<ct_args::gather>;

void kernel_main() {
    ActMcast mcast; mcast.init();
    mcast(); mcast.teardown();
    Mm mm; mm.init(); mm(); mm.teardown();
    Gath g; g.init(); g(); g.teardown();
}
```

The computation logic lives entirely in the MicroOp structs. The kernel `.cpp` only controls execution order and profiling. Use the handwritten path when you need custom phase ordering, manual profiler markers (`DeviceZoneScopedN`), or overlap optimizations where adjacent phases' processors run concurrently.

FusedOps that use codegen (no `kernel` attribute) can inspect the generated source via `program.generated_kernel` or on disk at `generated/kernels/<name>_<content_hash>.cpp`.

## DenseMLP: A FusedOp Delegating to Another FusedOp

`DenseMLP` (`blaze/ops/dense_mlp/op.py`) demonstrates FusedOp-on-FusedOp composition:

```python
class DenseMLP(FusedOp):
    name: str = "dense_mlp"

    act: Input = Input()
    rmsnorm_gamma: Input = Input()
    gate_up_weight: Input = Input()
    down_weight: Input = Input()
    out: Output = Output()
```

Its `emit()` calls `SharedExpert.emit()` (itself a FusedOp) as a building block:

```python
@staticmethod
def emit(f, act, rmsnorm_gamma, gate_up_weight,
         down_weight, *, prefix, ...):
    act_mcast_handle, act_ti, act_mcast_pages = \
        emit_mlp_preamble(f, act, rmsnorm_gamma,
                          prefix, rmsnorm_epsilon)
    return SharedExpert.emit(
        f, act_mcast_handle, gate_up_weight,
        down_weight, act, shared_output,
        ab_coords=shared_ab_grids,
        down_coords=shared_down_coords,
        pop_shared_act=False,
    )
```

The DenseMLP pipeline is: RMSNorm (via `emit_mlp_preamble`) then SharedExpert (which internally chains KNMatmul, GatedReduce, Mcast, and DownProj). A single `FusedProgram` dispatch runs the entire sequence -- over a dozen MicroOp stages -- as one kernel. The nesting depth is arbitrary; FusedOps compose at any level.

## SharedExpert: Preview (67 Lines Replacing 979 Lines)

`SharedExpert` (`blaze/ops/shared_expert/op.py`) is the canonical example of a production FusedOp. It chains four major building blocks:

$$\text{KNMatmul} \to \text{GatedReduce} \to \text{Mcast} \to \text{DownProj}$$

This pipeline implements the shared expert branch of DeepSeek V3's Mixture-of-Experts layer. The entire `op.py` is 115 lines of Python (including imports and the class declaration). The `emit()` method that wires the four-stage pipeline is approximately 67 lines. This replaces 979 lines of manual TT-Metal wiring in the equivalent hand-coded implementation -- approximately a $14\times$ code reduction for the full class, or $\sim 8.5\times$ for the `emit()` method alone.

SharedExpert demonstrates several advanced patterns:

- **Nested FusedOp embedding**: DownProj is itself a FusedOp (Matmul + Mcast + ResidualAdd + Gather), embedded as a sub-pipeline within SharedExpert.
- **Structured output types**: KNMatmul returns a `KNMatmulOutput` dataclass carrying not just the CBHandle but also derived parallel configuration ($k\_\text{parallel}$, $n\_\text{parallel}$, grid assignments).
- **Grid coordination**: The KN-sliced matmul uses A/B branch grids (gate vs. up projections), the GatedReduce uses the combined grid, and the DownProj uses a potentially different set of core coordinates.
- **Tile format transitions**: The pipeline transitions between face-view tiles ($1 \times 32$, 64 bytes) and full tiles ($32 \times 32$) at the GatedReduce-to-Mcast boundary, using `TileInfo` to propagate format metadata.

The full SharedExpert analysis -- including weight packing strategies, AB grid management, and multi-model generality -- is deferred to Chapter 7. The purpose of mentioning it here is to show that the FusedOp authoring pattern scales: the same `emit()` chaining, prefix nesting, and CBHandle plumbing that works for a three-op fusion works for a production pipeline spanning multiple core grids and over a dozen MicroOp phases.

## Testing

Testing a FusedOp follows the same pattern as testing a MicroOp -- build a symbolic graph, bind real tensors, compile, run, verify:

```python
import blaze
from blaze import ExternalTensor
from blaze.compiler import BlazeCompiler

with blaze.fuse() as ctx:
    act = blaze.mcast(
        ExternalTensor("activation"), grid=all_cores)
    mm = blaze.matmul(
        act, ExternalTensor("weights"), grid=compute_cores)
    out = blaze.gather(mm, grid=compute_cores)

compiler = BlazeCompiler(mesh_device)
program = compiler.compile(
    graph=ctx.graph,
    tensors={"activation": ttnn_act,
             "weights": ttnn_w},
    output_tensor=ttnn_output,
)
program.run()
assert comp_pcc(ttnn.to_torch(ttnn_output), expected, 0.999)
```

### Debugging Tools

The same debugging tools apply as for MicroOps (see [Section 05 -- "Debugging Tools"](./05_writing_a_micro_op.md)): `BLAZE_L1_PROFILE`, `BLAZE_DEBUG_KERNELS`, `program.generated_kernel`, and `compile_engines().ct_tuples`.

## Checklist for Writing a FusedOp

1. Create `blaze/ops/<name>/` with `__init__.py` and `op.py`
2. Define a class inheriting from `FusedOp` with a non-empty `name`
3. Declare all external ports using `Input()` and `Output()` descriptors
4. Implement `compose()` to unpack the tensor dict and call `emit()`
5. Implement `emit()` to chain MicroOp (or FusedOp) `emit()` calls
6. Use `BlazeOp.child_prefix(prefix, child_name)` for every child op
7. Return a `CBHandle` from `emit()` so callers can chain
8. (Optional) Set `kernel` if you need a handwritten kernel instead of codegen
9. Test with `FusedProgram.run()` on silicon and/or unit-test the CB wiring

## Summary: FusedOp vs MicroOp Authoring

For a side-by-side comparison of MicroOp vs FusedOp across all dimensions (kernel headers, CT args, `emit()`, `compose()`, enforcement, typical use), see the summary table in [Section 02 -- "Summary: MicroOp vs FusedOp"](./02_microop_and_fusedop.md). The key additional point for FusedOp authoring: the codegen path produces kernel C++ from the shadow graph unless `kernel` is set to a handwritten source file.

---
<- [05 -- Writing a MicroOp](./05_writing_a_micro_op.md) | [Index](index.md) | [Chapter 3 -- The Compilation Pipeline](../ch03_compilation_pipeline/index.md) ->
