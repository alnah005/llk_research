# 7.1 Current blaze-nn Dispatch Architecture

This section maps the complete dispatch path from blaze-nn's functional API through its internal registry, tracing contexts, and composition pipeline to the point where `ttnn.generic_op()` is called. Rather than describing each component in isolation, we trace a **concrete scenario** -- a single transformer attention block expressed in blaze-nn -- through every layer, showing what data flows where and why the current architecture has no awareness of TTNN op registration.

---

## 7.1.1 The Scenario: A Transformer Attention Block

Consider a standard pre-norm transformer attention block implemented with blaze-nn:

```python
# model.py -- a blaze-nn transformer attention block
import blaze_nn
from blaze_nn import Module, Parameter
import blaze_nn.functional as F

class Attention(Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = RMSNorm(dim)
        self.wq = Parameter(shape=(dim, dim))
        self.wk = Parameter(shape=(dim, dim))
        self.wv = Parameter(shape=(dim, dim))
        self.wo = Parameter(shape=(dim, dim))

    def forward(self, x, freqs_cis, mask):
        # Step 1: RMSNorm
        h = F.rmsnorm(x, self.norm.weight)

        # Step 2: QKV projections (three linear ops)
        q = F.linear(h, self.wq)
        k = F.linear(h, self.wk)
        v = F.linear(h, self.wv)

        # Step 3: RoPE (rotary position embedding)
        q = F.rope(q, freqs_cis)
        k = F.rope(k, freqs_cis)

        # Step 4: Scaled dot-product attention
        out = F.sdpa(q, k, v, mask=mask)

        # Step 5: Output projection
        out = F.linear(out, self.wo)
        return out
```

This block dispatches **eight** blaze-nn functional calls: 1 `rmsnorm`, 4 `linear` (QKV + output), 2 `rope`, and 1 `sdpa`. We will trace all eight through the current dispatch stack, focusing on `F.linear(h, self.wq)` as the primary walkthrough and noting where the other ops diverge.

---

## 7.1.2 The `_REGISTRY` and `OpMapping`

blaze-nn maintains a central registry that maps functional API names to Blaze op specifications. The registry is populated at import time and serves as the single source of truth for all dispatch decisions.

### Data Structures

```python
# blaze_nn/_registry.py (simplified from source)
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any

@dataclass
class OpMapping:
    """Maps a blaze-nn functional name to a Blaze op."""
    blaze_op_name: str              # The Blaze op name (e.g., "matmul")
    default_grid: str = "1x1"       # Default core grid specification
    kwargs_transform: Optional[Callable] = None  # Transforms F.* kwargs to
                                                  # Blaze op user_args
    extra_defaults: Dict[str, Any] = field(default_factory=dict)

# The registry: functional name -> OpMapping
_REGISTRY: Dict[str, OpMapping] = {}

def register_op(func_name: str, mapping: OpMapping):
    """Register a functional name to a Blaze op mapping."""
    _REGISTRY[func_name] = mapping
```

### Registry Population

At import time, `blaze_nn` populates the registry with entries for every supported functional operation:

```python
# blaze_nn/_registry.py (population block, simplified)

def _linear_kwargs_transform(weight, bias=None, **kwargs):
    """Transform F.linear kwargs to Blaze matmul user_args."""
    user_args = {"transpose_b": True}
    if bias is not None:
        user_args["bias"] = bias
    return user_args

register_op("linear", OpMapping(
    blaze_op_name="matmul",
    default_grid="4x8",
    kwargs_transform=_linear_kwargs_transform,
))

register_op("rmsnorm", OpMapping(
    blaze_op_name="rmsnorm",
    default_grid="1x32",
))

register_op("sdpa", OpMapping(
    blaze_op_name="sdpa",
    default_grid="8x4",
    kwargs_transform=_sdpa_kwargs_transform,
))

register_op("rope", OpMapping(
    blaze_op_name="rope",
    default_grid="1x32",
))
```

**Key observation:** The `OpMapping` contains no reference to TTNN operations. It maps functional names to *Blaze* op names (`"matmul"`, `"rmsnorm"`, etc.) and specifies grid configurations and argument transformations. The TTNN dispatch path is entirely implicit -- buried inside the Blaze compilation pipeline, terminating at `generic_op`.

### What the Registry Knows and Does Not Know

| Registry Field | Purpose | TTNN Awareness |
|----------------|---------|:-:|
| `blaze_op_name` | Resolves to a `BlazeOp` subclass via `BlazeOp._class_registry` | None |
| `default_grid` | Core grid for compilation | None |
| `kwargs_transform` | Translates `F.*` kwargs to `compose()` user_args | None |
| `extra_defaults` | Default values for optional kwargs | None |
| *(missing)* `ttnn_callable` | Would point to `ttnn.blaze.*` | Does not exist |

The registry is a blaze-nn-to-Blaze bridge. It has no concept of TTNN-level op registration, program caching semantics, or profiler identity. This is the core gap that Section 7.2 addresses.

---

## 7.1.3 The `_dispatch()` Function

Every functional API call converges on `_dispatch()`, the central dispatch choke point:

```python
# blaze_nn/_registry.py (simplified)
def _dispatch(func_name: str, *args, **kwargs):
    """
    Central dispatch: resolves the functional name to a Blaze op
    and delegates to the active tracing context or compose mode.
    """
    mapping = _REGISTRY[func_name]

    # Transform kwargs if a transform function is registered
    if mapping.kwargs_transform is not None:
        user_args = mapping.kwargs_transform(*args, **kwargs)
    else:
        user_args = kwargs

    # Get the active context (tracing or compose)
    ctx = _get_active_context()

    if ctx is not None:
        # Graph mode: record a node in the trace graph
        return ctx.dispatch(
            op_name=mapping.blaze_op_name,
            args=args,
            user_args=user_args,
            grid=mapping.default_grid,
        )
    else:
        # Compose mode: compile and execute immediately
        return _compose_and_run(
            op_name=mapping.blaze_op_name,
            args=args,
            user_args=user_args,
            grid=mapping.default_grid,
        )
```

The branching point is `_get_active_context()`. If a tracing context is active (graph mode), the dispatch records a graph node without executing anything. If no context is active (compose/direct mode), the dispatch compiles and runs the op immediately. Both paths eventually reach `_run_program()`, but at different times: graph mode defers execution to graph compilation; compose mode executes immediately.

---

## 7.1.4 `TensorProxy` and `Parameter`

Before tracing the scenario, we need two supporting types.

### `TensorProxy`: Lazy Tensors for Graph Mode

```python
# blaze_nn/tracing.py (simplified)
class TensorProxy:
    """
    Lazy tensor type used during graph tracing. Records operations
    without executing them. Carries shape, dtype, and layout metadata
    but no device data.
    """
    def __init__(self, shape, dtype, layout, source_node=None):
        self.shape = shape
        self.dtype = dtype
        self.layout = layout
        self.source_node = source_node  # The graph node that produced this proxy
```

During graph tracing, `_dispatch()` receives `TensorProxy` objects instead of real `ttnn.Tensor` objects. The tracing context uses the proxy metadata (shape, dtype) to construct graph nodes with correct edge types, then returns a new `TensorProxy` representing the output. No hardware execution occurs.

### `Parameter`: Weight Tensor Wrapper

```python
# blaze_nn/module.py (simplified)
class Parameter:
    """
    Wraps a weight tensor. During tracing, behaves like a TensorProxy
    with known shape. At execution time, resolved to a device tensor.
    """
    def __init__(self, shape, dtype=None, data=None):
        self.shape = shape
        self.dtype = dtype
        self._data = data       # None until loaded via load_state_dict()
        self._device_tensor = None  # Lazily allocated on first use

    def to_device(self, device):
        """Allocate on device and return the device tensor."""
        if self._device_tensor is None:
            self._device_tensor = ttnn.from_torch(self._data, device=device)
        return self._device_tensor
```

`Parameter` ownership stays entirely within blaze-nn. The adapter and registration machinery never touch parameter management -- they operate on the `MeshProgramDescriptor` that the compilation pipeline produces *after* parameters have been resolved to device tensors. This separation is critical for understanding why module integration is transparent (Section 7.2).

---

## 7.1.5 Tracing Contexts: Graph Mode vs Compose Mode

blaze-nn supports two execution modes, managed by tracing contexts.

### `GraphTracingContext` (Graph Mode)

```python
# blaze_nn/tracing.py (simplified)
class GraphTracingContext:
    """
    Active during model.trace(). Records all dispatch calls as
    nodes in a BlazeNNGraph without executing them.
    """
    def __init__(self):
        self.graph = BlazeNNGraph()

    def dispatch(self, op_name, args, user_args, grid):
        # Create proxy inputs from args
        proxy_inputs = [
            a if isinstance(a, TensorProxy)
            else TensorProxy.from_parameter(a) if isinstance(a, Parameter)
            else TensorProxy.from_tensor(a)
            for a in args
        ]

        # Record a node in the graph
        node = self.graph.add_node(
            op_name=op_name,
            inputs=proxy_inputs,
            user_args=user_args,
            grid=grid,
        )

        # Return a proxy representing the output
        output_shape = _infer_output_shape(op_name, proxy_inputs, user_args)
        return TensorProxy(
            shape=output_shape,
            dtype=proxy_inputs[0].dtype,
            layout=proxy_inputs[0].layout,
            source_node=node,
        )
```

In graph mode, `F.linear(h, self.wq)` does not execute a matmul. It records a graph node with `op_name="matmul"`, input edges from `h` (a `TensorProxy`) and `self.wq` (a `Parameter`), and `user_args={"transpose_b": True}`. The returned `TensorProxy` carries the inferred output shape and a reference to the graph node.

### `ComposeTracingContext` (Compose Mode)

```python
# blaze_nn/tracing.py (simplified)
class ComposeTracingContext:
    """
    Active during direct execution. Compiles each op via BlazeCompiler
    and runs it immediately on hardware.
    """
    def dispatch(self, op_name, args, user_args, grid):
        # Resolve BlazeOp class from name
        op_class = BlazeOp._class_registry[op_name]

        # Resolve tensors (Parameters -> device tensors)
        tensors = _resolve_tensors(args)

        # Compile via BlazeCompiler
        compiled = BlazeCompiler().compile(
            op=op_class,
            tensors=tensors,
            output=_allocate_output(op_name, tensors, user_args),
            user_args=user_args,
        )

        # Execute: this calls _run_program() -> ttnn.generic_op()
        return compiled.run(tensors)
```

In compose mode, every `F.linear()` call triggers a full compilation and execution cycle. The `BlazeCompiler` produces a `MeshCompiledProgram`, which calls `_run_program()`, which calls `ttnn.generic_op()`. This is the hot path for eager-mode inference.

---

## 7.1.6 Scenario Walkthrough: Attention Block Through Current Dispatch

Now we trace the full attention block from Section 7.1.1 through the current dispatch, in **compose mode** (eager execution). Graph mode is discussed separately in Section 7.1.8.

### Call 1: `F.rmsnorm(x, self.norm.weight)`

```text
F.rmsnorm(x, self.norm.weight)
  -> _dispatch("rmsnorm", x, self.norm.weight)
     -> mapping = _REGISTRY["rmsnorm"]
        -> OpMapping(blaze_op_name="rmsnorm", default_grid="1x32",
                     kwargs_transform=None)
     -> user_args = {}  (no transform)
     -> ctx = _get_active_context()  =>  ComposeTracingContext  [compose mode]
     -> ctx.dispatch("rmsnorm", [x, weight], {}, "1x32")
        -> op_class = BlazeOp._class_registry["rmsnorm"]  =>  RMSNorm
        -> tensors = [x_device, weight_device]
        -> compiled = BlazeCompiler().compile(RMSNorm, tensors, output, {})
           -> [CB engine, sem engine, CT-arg engine, kernel codegen]
           -> MeshProgramDescriptor assembled
           -> MeshCompiledProgram created
        -> compiled.run(tensors)
           -> _run_program(io_tensors, mesh_program_descriptor)
              -> ttnn.generic_op(io_tensors, mesh_program_descriptor)
                 -> GenericOpDeviceOperation::invoke(...)
                 -> launch<GenericOpDeviceOperation>(...)
                    -> TracyOpMeshWorkload("ttnn::generic_op")  // OPAQUE
                    -> compute_program_hash(type_hash<GenericOpDeviceOperation>, ...)
                    -> [cache lookup, factory execution, enqueue]
     -> returns: h  (normalized tensor)
```

### Call 2: `F.linear(h, self.wq)` (Q projection)

```text
F.linear(h, self.wq)
  -> _dispatch("linear", h, self.wq)
     -> mapping = _REGISTRY["linear"]
        -> OpMapping(blaze_op_name="matmul", default_grid="4x8",
                     kwargs_transform=_linear_kwargs_transform)
     -> user_args = _linear_kwargs_transform(self.wq)
        -> {"transpose_b": True}
     -> ctx.dispatch("matmul", [h, wq_device], {"transpose_b": True}, "4x8")
        -> op_class = BlazeOp._class_registry["matmul"]  =>  Matmul
        -> compiled = BlazeCompiler().compile(Matmul, tensors, output,
                                              {"transpose_b": True})
        -> compiled.run(tensors)
           -> _run_program(io_tensors, mesh_program_descriptor)
              -> ttnn.generic_op(io_tensors, mesh_program_descriptor)
                 -> TracyOpMeshWorkload("ttnn::generic_op")  // OPAQUE
                 -> compute_program_hash(type_hash<GenericOpDeviceOperation>, ...)
     -> returns: q
```

### Calls 3--8: Remaining Six Calls

The remaining six calls -- K/V projections (`F.linear`), two RoPE applications (`F.rope`), SDPA (`F.sdpa`), and the output projection (`F.linear`) -- follow the same dispatch path demonstrated in Calls 1 and 2, differing only in the Blaze op name (e.g., `"rope"`, `"sdpa"`) and grid configuration (e.g., `"1x32"` for RoPE, `"8x4"` for SDPA). All terminate at `ttnn.generic_op(io_tensors, mesh_program_descriptor)` with an opaque `TracyOpMeshWorkload("ttnn::generic_op")` profiler zone.

---

## 7.1.7 The Observability Problem

After executing the attention block, the Tracy profiler timeline shows:

```text
|--- ttnn::generic_op ---|--- ttnn::generic_op ---|--- ttnn::generic_op ---|
|--- ttnn::generic_op ---|--- ttnn::generic_op ---|--- ttnn::generic_op ---|
|--- ttnn::generic_op ---|--- ttnn::generic_op ---|
```

Eight calls, all labeled identically. The graph trace shows:

```text
GenericOpDeviceOperation -> GenericOpDeviceOperation -> GenericOpDeviceOperation ->
GenericOpDeviceOperation -> GenericOpDeviceOperation -> GenericOpDeviceOperation ->
GenericOpDeviceOperation -> GenericOpDeviceOperation
```

The program cache contains entries keyed with:

```text
hash(type_hash<GenericOpDeviceOperation>, descriptor_rmsnorm)
hash(type_hash<GenericOpDeviceOperation>, descriptor_matmul_q)
hash(type_hash<GenericOpDeviceOperation>, descriptor_matmul_k)
hash(type_hash<GenericOpDeviceOperation>, descriptor_matmul_v)
hash(type_hash<GenericOpDeviceOperation>, descriptor_rope_q)
hash(type_hash<GenericOpDeviceOperation>, descriptor_rope_k)
hash(type_hash<GenericOpDeviceOperation>, descriptor_sdpa)
hash(type_hash<GenericOpDeviceOperation>, descriptor_matmul_out)
```

All eight share the same `type_hash` prefix. There is no way to determine, from profiler output alone, which `generic_op` call corresponds to the RMSNorm, which to the Q projection, which to the SDPA. Correlating profiler entries to model operations requires external timestamp matching against Python-side logs -- a manual, error-prone process that does not scale beyond simple models.

### Quantifying the Impact

For a full transformer model with 32 layers, each layer containing an attention block (8 ops) and an MLP block (~4 ops), the total is approximately $32 \times 12 = 384$ `generic_op` calls per forward pass. All 384 appear as identical entries in every observability system. Performance analysis becomes a forensic exercise rather than a diagnostic one.

---

## 7.1.8 Graph Mode: What Changes, What Stays

In graph mode, the eight functional calls record graph nodes instead of executing immediately:

```python
# Graph mode usage
with blaze_nn.trace() as tracer:
    output = attention(x_proxy, freqs_cis_proxy, mask_proxy)
    # No hardware execution has occurred
    graph = tracer.graph
```

The `BlazeNNGraph` contains eight nodes, each with the Blaze op name (`"rmsnorm"`, `"matmul"`, `"rope"`, `"sdpa"`), input/output edge types, grid specifications, and user_args. When the graph is compiled:

```python
# Graph compilation (simplified)
compiled_graph = blaze_nn.compile(graph)
# This compiles all ops, creating a sequence of MeshCompiledProgram objects
```

Each `MeshCompiledProgram` in the compiled graph still routes through `_run_program()` -> `ttnn.generic_op()` at execution time. Graph mode defers compilation and enables cross-op optimizations (e.g., fusing adjacent ops, reordering for memory efficiency), but the eventual dispatch path is identical: `generic_op` with no TTNN-visible identity.

The graph does retain op names internally (for blaze-nn's own optimization passes), but these names are not propagated to TTNN's `GraphTracker` or the program cache. The information exists in blaze-nn's Python-level graph representation but is lost at the TTNN dispatch boundary.

---

## 7.1.9 The Structural Gap

The current architecture has a clean separation of concerns:

```text
blaze-nn layer:  F.linear() -> _dispatch() -> _REGISTRY -> OpMapping -> blaze_op_name
                                                                             |
Blaze layer:     BlazeOp._class_registry[blaze_op_name] -> BlazeCompiler -> MeshProgramDescriptor
                                                                             |
TTNN layer:      MeshCompiledProgram._run_program() -> ttnn.generic_op() -> GenericOpDeviceOperation
```

The gap is at the TTNN layer boundary. blaze-nn knows the op name (`"matmul"`), the Blaze layer knows the op class (`Matmul`), but the TTNN layer sees only `GenericOpDeviceOperation` with an opaque `MeshProgramDescriptor`. The op identity that exists at the blaze-nn and Blaze layers is discarded at the `generic_op` boundary.

This is not a bug. It reflects the original design intent: `generic_op` is a universal executor that does not need to know what it is executing. But that universality is precisely what prevents TTNN's infrastructure (caching, tracing, profiling) from providing per-op visibility. The registration pathway from Chapters 4--6 creates named dispatch points (`ttnn.blaze.matmul`, `ttnn.blaze.rmsnorm`, etc.) that preserve op identity across the boundary. The next section shows how blaze-nn connects to those dispatch points.

---

## 7.1.10 Summary: Current Dispatch Layer Map

The following table maps every layer in the current dispatch path for `F.linear(h, self.wq)`, showing what information is available at each layer and what is lost:

| Layer | Function/Method | Input | Output | Op Identity Visible? |
|-------|----------------|-------|--------|:---:|
| 1. Functional API | `F.linear(h, self.wq)` | tensor, Parameter | dispatch call | Yes: `"linear"` |
| 2. Registry Lookup | `_REGISTRY["linear"]` | `"linear"` | `OpMapping` | Yes: `"matmul"` (via `blaze_op_name`) |
| 3. Kwargs Transform | `_linear_kwargs_transform(...)` | weight, kwargs | `user_args` | N/A |
| 4. Context Dispatch | `ctx.dispatch("matmul", ...)` | op_name, args, user_args | compiled program | Yes: `"matmul"` |
| 5. BlazeCompiler | `BlazeCompiler().compile(Matmul, ...)` | op class, tensors, user_args | `MeshCompiledProgram` | Yes: `Matmul` class |
| 6. `_run_program()` | `compiled.run(tensors)` | io_tensors, descriptor | device execution | **No**: only `generic_op` |
| 7. TTNN Dispatch | `ttnn.generic_op(...)` | io_tensors, descriptor | `GenericOpDeviceOperation` | **No**: `"ttnn::generic_op"` |
| 8. Program Cache | `compute_program_hash(...)` | `type_hash<GenericOpDeviceOperation>` | hash | **No**: shared prefix |
| 9. Tracy Profiler | `TracyOpMeshWorkload(...)` | `"ttnn::generic_op"` | profiler zone | **No**: opaque name |

Op identity flows from Layer 1 through Layer 5, then is lost at Layer 6 when `_run_program()` calls `generic_op` unconditionally. The registration pathway restores identity at Layers 6--9 by routing through `ttnn.blaze.matmul` instead.

---

| [Chapter 6: Caching, Tracing, and Profiling](../ch06_caching_tracing_profiling/index.md) | [Chapter Index](index.md) | [Rewired Dispatch and Module Integration](02_rewired_dispatch_and_module_integration.md) |
|:---|:---:|---:|
