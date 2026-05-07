# 03 -- The Two-API Design

Blaze provides two distinct APIs for building fused kernel programs. Both ultimately produce a `BlazeGraph` -- a validated DAG of operations connected by typed data-flow edges -- but they target different stages of the development lifecycle and different levels of abstraction.

## The Graph API

The graph API is a declarative interface for specifying **what** ops to run and **how** data flows between them, without specifying the mechanical details of CB allocation, CT arg wiring, or semaphore management.

### Entry Point: `blaze.fuse()`

The `fuse()` function returns a `FusionContext`, used as a context manager:

```python
import blaze

act = blaze.ExternalTensor("activation")
wt  = blaze.ExternalTensor("weights")

with blaze.fuse() as ctx:
    mm = blaze.matmul(act, wt, grid=matmul_grid)
    out = blaze.gather(mm, grid=gather_grid)

graph = ctx.graph  # validated BlazeGraph
```

Inside the `with` block, every `blaze.<op_name>(...)` call dispatches to the active `FusionContext` via `_op_call()`. The context records each op as an `OpNode` and each data dependency as an `Edge`.

### Key Types

**`FusionContext`** (`context.py`) -- the context manager object returned by `blaze.fuse()`. It captures every op call made within the `with` block:

- Maintains a list of `OpNode` instances and `Edge` instances
- Tracks `ExternalTensor` inputs and their port bindings
- Auto-generates node IDs using per-op-type counters (e.g., `matmul_1`, `matmul_2`)
- On exit (without exception), calls `_build_graph()` to construct a `BlazeGraph` and `validate()` to check for structural errors (duplicate node IDs, invalid port references, cycles via Kahn's algorithm)

**`ExternalTensor`** -- a named placeholder for an input that will be bound to an actual `ttnn.Tensor` at compile time. When passed as input to an op call, the context records it in the graph's input tracking maps. See [Ch 3 Section 01](../ch03_compilation_pipeline/01_blaze_graph_and_fusion_context.md) for the full dataclass definition and field descriptions.

**`FusionResult`** -- a lightweight handle returned by each op call inside a `fuse()` block. Passing a `FusionResult` as input to another op creates a data-dependency `Edge` in the graph. See [Ch 3 Section 01](../ch03_compilation_pipeline/01_blaze_graph_and_fusion_context.md) for details.

**`BlazeGraph`** -- a validated DAG of `OpNode` instances connected by `Edge` instances, with topological ordering and edge indexing. See [Ch 3 Section 01](../ch03_compilation_pipeline/01_blaze_graph_and_fusion_context.md) for the complete data structure, Kahn's algorithm implementation, and graph validation.

### How Graph API Calls Dispatch

When you call `blaze.matmul(activation, weights)`, the `_OpHandle.__call__` method routes to `Matmul.graph_call()`:

```python
@classmethod
def graph_call(cls, *args, grid=None, **kwargs):
    from .context import _op_call
    input_names = cls._get_input_names()
    inputs_dict = {name: arg for name, arg in zip(input_names, args)}
    return _op_call(cls.name, inputs_dict, grid=grid, **kwargs)
```

`_op_call` checks that a `FusionContext` is active (raises `RuntimeError` otherwise), then delegates to `FusionContext.add_op()`, which creates an `OpNode`, wires `Edge` objects for data dependencies, records external tensor inputs, and returns a `FusionResult`. For the full `add_op()` implementation walkthrough, see [Ch 3 Section 01](../ch03_compilation_pipeline/01_blaze_graph_and_fusion_context.md).

### Engine Compilation

After building the graph, call `blaze.compile_engines()` to run the CB, semaphore, and CT-arg engines:

```python
result = blaze.compile_engines(ctx.graph)
# result.cb_assignments   -- CB ID per edge/port
# result.sem_assignments  -- semaphore IDs
# result.ct_tuples        -- per-RISC (name, value) pairs
```

This produces an `EngineResult` that maps every graph edge to a CB ID, every synchronization point to a semaphore, and every CT arg to its resolved value. The `BlazeCompiler` uses this result to assemble the final `MeshProgramDescriptor`.

## The Composition API

The composition API is an imperative interface for specifying **how** an op wires itself into a fused program. It operates at a lower level than the graph API, giving the op author direct control over CB allocation, CT/RT arg values, and per-core role flags.

### Entry Point: `FusedProgram`

`FusedProgram` wraps a `BlazeProgram` with device context, grid configuration, and high-level allocation helpers:

```python
from blaze.fused_program import FusedProgram

f = FusedProgram(
    kernel="path/to/kernel.cpp",  # or None for codegen
    device=device,
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=True,
)
```

On construction, `FusedProgram` queries the device to populate `GridConfig` (grid dimensions, DRAM worker positions), precomputes common core grids (all cores, matmul cores, sender core, mcast receivers), translates logical-to-physical NOC coordinates, and initializes the underlying `BlazeProgram`.

### `emit()` Methods and CBHandle Chaining

Each op defines a static `emit()` method that takes a `FusedProgram` and returns a `CBHandle`:

```python
class Matmul(MicroOp):
    @staticmethod
    def emit(f, in0, in1, *, prefix="matmul") -> CBHandle:
        in0_handle = f.cb_from_tensor(in0)
        out = f.cb_scratch(name=..., num_pages=..., ...)
        f.unified_ct_args([(f"{prefix}.in0", in0_handle)])
        f.trisc_ct_args([(f"{prefix}.out", out), ...])
        return out
```

The returned `CBHandle` is the key to composition. It carries the CB ID, page geometry, data format, and tile descriptor. When the next op's `emit()` receives it as an input argument, it reads those fields to configure its own CBs and CT args. This is **CBHandle chaining** -- data flows through the Python composition layer exactly as it will flow through L1 circular buffers at runtime.

A complete fused pipeline is a sequence of `emit()` calls:

```python
def emit(f, activation, weights, bias, output):
    act = Mcast.emit(f, activation, prefix="act_mcast")
    mm  = Matmul.emit(f, act, weights, prefix="matmul")
    res = Mcast.emit(f, bias, prefix="mcast2")
    add = ResidualAdd.emit(f, mm, res, prefix="residual_add")
    return Gather.emit(f, add, output, prefix="gather")
```

### How `emit()` Works Internally

Each op's `emit()` is a static method on the op class. Taking `Matmul` as a representative example, the internal steps are:

1. **Resolve inputs**: if the input is a `CBHandle`, use it directly; if it is a `ttnn.Tensor`, allocate a CB via `f.cb_from_tensor()`
2. **Compute dimensions**: derive `k_num_tiles`, `out_w_per_core` from input geometry
3. **Allocate output CB**: `f.cb_scratch(name=..., num_pages=..., core_ranges=..., data_format=..., tile=..., page_size=...)`
4. **Register CT args**: `f.per_core_unified_ct_args(...)` for role flags, `f.unified_ct_args(...)` for shared CB IDs, `f.ncrisc_ct_args(...)` for reader-specific values, `f.trisc_ct_args(...)` for compute-specific values
5. **Register RT args**: `f.trisc_rt_args(...)` for weight L1 addresses
6. **Return CBHandle**: via `f.output("matmul", out, core_ranges=..., prefix=..., in0=..., in1=...)`, which records the node in the shadow graph and returns a handle carrying the output CB's metadata

### The `prefix` Convention

Every `emit()` call takes a `prefix` keyword argument. This prefix namespaces all CT/RT arg names and CB names for that op instance:

- **CT args**: `f"{prefix}.field_name"` (e.g., `"matmul.k_num_tiles"`)
- **CB names**: `f"{prefix}___cb_name"` (triple underscore delimiter, e.g., `"matmul___out"`)
- **Child ops**: `BlazeOp.child_prefix(prefix, child_name)` produces `f"{prefix}__{child_name}"` (double underscore, e.g., `"shared_expert__gu"`)

This namespacing prevents name collisions when the same op type appears multiple times in a fused pipeline, and it enables the CT arg engine to map Python-side values to kernel-side struct fields.

### Key `FusedProgram` Methods

| Method | Purpose |
|---|---|
| `cb_from_tensor(tensor)` | Allocate a CB backed by a sharded tensor. Returns `CBHandle`. |
| `cb_from_view(view)` | Allocate a CB from an `OverlappedView` (sub-region of a fused L1 buffer). |
| `cb_scratch(name, num_pages, ...)` | Allocate a scratch CB (not tensor-backed). |
| `cb_alias(source_cb, dtype, ...)` | Create a CB alias sharing L1 with a different data format. |
| `cb_output(tensor, ...)` | Allocate a CB for the final output tensor. |
| `unified_ct_args(args)` | Add named CT args to all RISCs. |
| `ncrisc_ct_args(args)` | Add named CT args for NCRISC only. |
| `trisc_ct_args(args)` | Add named CT args for TRISC only. |
| `brisc_ct_args(args)` | Add named CT args for BRISC only. |
| `per_core_unified_ct_args(args)` | Add per-core CT args to all RISCs. |
| `flag(name, cores, enabled)` | Create a per-core boolean CT arg (role flag). |
| `semaphore(name)` | Allocate a global semaphore (dedup by name across iterations). |
| `output(name, cb, ...)` | Wire the final output tensor and return a `CBHandle`. Records a node in the shadow graph. |
| `multi_output(outputs)` | Wire multiple output tensors, returning a `MultiOutput`. |
| `reconfig()` | Mark a phase boundary for multi-phase CB reconfiguration. |
| `build()` | Assemble the `ProgramDescriptor` (without running). |
| `run()` | Build and dispatch via `ttnn.generic_op()`. |

### Key Composition Types

**`CBHandle`** -- the currency of the composition API. Returned by `cb_from_tensor()`, `cb_scratch()`, and ultimately by each op's `emit()`. It carries:

- `cb_id` (integer-coercible: `int(handle)` works in CT arg tuples)
- `num_pages`, `page_size`, `data_format`, `tile_desc` -- tile geometry
- `core_ranges` -- which cores own this CB
- `byte_offset` and `backing_tensor` -- for overlapped views into shared L1
- `access_mode` -- FIFO (normal push/pop) or DIRECT_ADDRESS (explicit L1 address reads)

**`OverlappedView`** -- a frozen dataclass representing a logical sub-region of a fused L1 buffer. Multiple views can share one backing tensor with different `byte_offset`, `shard_shape`, and `dtype` values. Used for weight packing where multiple weight matrices are fused into a single L1 allocation. `FusedProgram.cb_from_view()` creates a CB backed by the view's sub-region.

**`MultiOutput`** -- returned by multi-output ops (e.g., `DeepseekMoeGate`). Supports positional unpacking, attribute access, and dict-style access:

```python
scores, indices = DeepseekMoeGate.emit(f, ...)
# or: result.output, result["output_indices"]
```

## How the Two APIs Converge

Both APIs produce a `BlazeGraph`. The convergence paths differ in when and how the graph is constructed:

```
Graph API path:                          Composition API path:

blaze.fuse() context manager             FusedProgram constructor
    |                                        |
blaze.<op>() calls                       Op.emit(f, ...) calls
    |                                        |
FusionContext records OpNodes + Edges    Shadow graph records nodes + edges
    |                                    (+ direct CB/sem/CT allocation)
    |                                        |
ctx.graph -> BlazeGraph                  f._shadow_graph -> BlazeGraph
    |                                        |
compile_engines(graph) -> EngineResult   f.build() -> ProgramDescriptor
    |                                        |
BlazeCompiler.compile() -> program       f.run() -> ttnn.generic_op()
```

**Graph API path**: The `FusionContext` directly builds a `BlazeGraph` from the recorded `OpNode` and `Edge` lists. The resulting graph is a pure data-flow description -- it has no CB IDs, no semaphore assignments, no CT arg values yet. These are assigned later by running `compile_engines(graph)`, which invokes the `CBEngine`, `SemEngine`, and `CTArgEngine` in sequence, returning a unified `EngineResult`.

**Composition API path**: Each `emit()` call on a `FusedProgram` simultaneously:

- Records a node in the `FusedProgram`'s internal **shadow graph** (`_shadow_graph`)
- Allocates actual CB IDs and semaphores on the underlying `BlazeProgram`
- Registers concrete CT/RT arg values

The shadow graph serves two purposes: (1) it is the input to the kernel codegen system (`kernel_codegen.py`), which can auto-generate the C++ kernel source from the graph topology, and (2) it drives the interactive visualizer.

**Compiler path** (`BlazeCompiler`): The compiler bridges the two APIs. It takes a `BlazeGraph` (from the graph API), resolves external tensors to actual `ttnn.Tensor` objects, looks up each op's `FusedOpConfig`, and calls each op's `compose()` classmethod with a `FusedProgram`. The `compose()` method maps the tensor dict to `emit()` calls, effectively translating the declarative graph into imperative composition. This is the path used when calling `BlazeCompiler.compile()`:

```python
compiler = BlazeCompiler(mesh_device)
result = compiler.compile(
    graph=ctx.graph,
    tensors={"activation": ttnn_act, "weights": ttnn_w},
    output_tensor=ttnn_output,
    user_args={"prefix": "my_op"},
)
output = result.run()
```

The convergence point is that **`compose()` bridges the graph API to the composition API**. Every `FusedOp` and `MicroOp` that wants to be usable from the graph API implements a `compose()` classmethod:

```python
class Matmul(MicroOp):
    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(f, tensors["in0"], tensors["in1"],
                 prefix=user_args.get("prefix", "matmul"))
```

The `BlazeCompiler` calls `compose()` for each node in topological order, passing the `FusedProgram`, resolved tensor dict, output tensor, and user-supplied arguments. Inside `compose()`, the op calls `emit()` -- exactly the same function used in the composition API path.

### Convergence Summary

| Dimension | Graph API | Composition API |
|---|---|---|
| **Entry point** | `blaze.fuse()` context manager | `FusedProgram` + `Op.emit()` chains |
| **Op invocation** | `blaze.<op>(...)` inside `with` block | `Op.emit(f, ...)` calls |
| **Data flow** | `FusionResult` handles | `CBHandle` chaining |
| **Graph construction** | Explicit: `FusionContext` builds on exit | Implicit: shadow graph during `emit()` |
| **CB/sem allocation** | Deferred (engines run later) | Immediate (on each `emit()` call) |
| **Compilation** | `compile_engines()` then `BlazeCompiler` | `f.build()` / `f.run()` |
| **Typical user** | Model authors, tooling developers | Op authors, kernel developers |

## When to Use Each API

**Use the graph API when:**

- Building declarative fusion graphs that will be compiled later with different tensor bindings
- Exploring graph structure via `compile_engines()`, the visualizer, or the engine outputs before committing to device execution
- Leveraging the `BlazeCompiler` for mesh-aware, multi-device compilation where the compiler handles per-device tensor splitting and per-device program assembly
- Auto-generating kernels via `kernel_codegen.py` -- the codegen system operates on a `BlazeGraph`, not on imperative composition traces
- Building tooling that reasons about pipeline structure

**Use the composition API when:**

- Writing a new op (`emit()` is the canonical op definition)
- Composing FusedOps from existing MicroOps -- `SharedExpert`, for example, chains four `emit()` calls in its own `emit()` method
- Prototyping on a single device where you want direct control over CB allocation, core grids, and execution
- Needing runtime values that are only available when you have real tensors (buffer addresses, shard shapes, per-core mappings)
- Iterating quickly: write `emit()`, call `f.run()`, check output on silicon

In practice, the two APIs are complementary. Most **op authors** work primarily with the composition API (writing `emit()` methods and testing with `FusedProgram.run()`), while **model authors** work with the graph API (composing ops in `blaze.fuse()` blocks and compiling with `BlazeCompiler`). A typical workflow: define individual ops using the composition API, compose them into `FusedOp` classes, then optionally express higher-level fusion patterns using the graph API for the compiler path. The graph API provides the specification; the composition API provides the implementation.

---

**Next:** [Chapter 2 — The BlazeOp Class Hierarchy and Op Authoring](../ch02_blazeop_hierarchy/index.md)
