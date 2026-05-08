# 03 -- Run Modes

This file catalogs all TT-Symbiote run modes, explains how they are selected, and provides detailed coverage of the `TracedRun` mode's three-phase lifecycle, the `TTNNLayerStack` trace wrapper, trace decorators, the Ring topology requirement for traced CCL, the pre/post trace hooks, and practical guidance on when to use each mode. Each run mode defines two methods (`torch_dispatch` and `module_run`) that control how individual tensor operations and module-level forward passes are executed. Prerequisites: understanding of the dispatch and tensor wrapping machinery from [`02_dispatch_and_tensor_wrapping.md`](./02_dispatch_and_tensor_wrapping.md) and the module lifecycle from [`01_module_replacement_engine.md`](./01_module_replacement_engine.md).

## Run Mode Overview

A **run mode** defines how TT-Symbiote executes both module-level calls (`module_run()`) and operation-level dispatch (`torch_dispatch()`). All run modes are registered in `_RUN_MODE_REGISTRY` in [run_config.py]:

```python
_RUN_MODE_REGISTRY = {
    "LIGHTWEIGHT": LightweightRun,
    "NORMAL":      NormalRun,
    "NORMAL_WITH_FALLBACK": NormalRunWithFallback,
    "SEL":         SELRun,
    "DPL":         DPLRun,
    "DPL_NO_ERROR_PROP": DPLRunNoErrorProp,
    "CPU":         CPU,
    "TRACED":      TracedRun,
}
```

Each entry maps a string key to a **class** (not an instance) that provides static methods `new_instance()`, `torch_dispatch()`, `module_run()`, `to_ttnn()`, and `to_torch()`.

## Mode Selection

### Environment Variable

The primary selection mechanism is the `TT_SYMBIOTE_RUN_MODE` environment variable:

```bash
export TT_SYMBIOTE_RUN_MODE=NORMAL        # Default: run ops on TTNN when possible
export TT_SYMBIOTE_RUN_MODE=TRACED         # Enable trace capture and replay
export TT_SYMBIOTE_RUN_MODE=SEL            # Side-by-side comparison
export TT_SYMBIOTE_RUN_MODE=CPU            # Everything on CPU
```

### Programmatic Selection

```python
from models.experimental.tt_symbiote.core.run_config import set_run_mode
set_run_mode("TRACED")  # Must be called before any tensor operations
```

### Resolution Order

In `get_tensor_run_implementation()`:

```
  1. Check TT_SYMBIOTE_RUN_MODE env var
  2. If env var is set, use it (overrides programmatic setting with warning)
  3. If env var not set, use programmatic setting from set_run_mode()
  4. If neither is set, default to "NORMAL"
```

The implementation class is resolved at import time and stored in the module-level `TENSOR_RUN_IMPLEMENTATION` variable. This means the run mode cannot be changed after the first import of TT-Symbiote modules.

> **Warning:** The run mode can only be set **once**. Calling `set_run_mode()` a second time with a different value raises an assertion error. The environment variable must be set *before* any TT-Symbiote import. If `TT_SYMBIOTE_RUN_MODE` is set after `TorchTTNNTensor` or `TTNNModule` has been imported, it has no effect.

The run implementation is also validated at selection time: `LightweightRun` and its subclasses (`TracedRun`) require the CPU dispatcher to be active, enforced by an assertion.

### Custom Run Modes

The registry supports runtime extension via `add_run_mode()`:

```python
from models.experimental.tt_symbiote.core.run_config import add_run_mode

class MyCustomRun(NormalRun):
    @staticmethod
    def torch_dispatch(cls, func, types, args=(), kwargs=None):
        # Custom dispatch logic
        ...

add_run_mode("CUSTOM", MyCustomRun)
# Then: export TT_SYMBIOTE_RUN_MODE=CUSTOM
```

The custom class must provide at minimum `torch_dispatch()` and `module_run()` static methods. Inheriting from `NormalRun` provides sensible defaults for all other methods.

## Run Mode Catalog

### NORMAL (NormalRun)

The default mode. Routes each `aten` operation through the `can_dispatch_to_ttnn` / `dispatch_to_ttnn` decision tree. At the module level, applies the full input transform pipeline, calls `preprocess_weights()` and `move_weights_to_device()`, then `forward()`.

**torch_dispatch behavior**: Check `can_dispatch_to_ttnn()`. If True, dispatch to TTNN handler. If False, unwrap to torch tensors, execute on CPU, re-wrap results.

**module_run behavior**: Transform inputs -> preprocess weights -> move weights to device -> `forward()` -> post-process outputs.

### NORMAL_WITH_FALLBACK (NormalRunWithFallback)

Extends `NormalRun` with exception handling at both levels:

**torch_dispatch behavior**: Tries TTNN dispatch; on **any exception**, catches it, logs the error, and falls back to PyTorch execution.

**module_run behavior**: If `self.device` is set, attempts the normal TTNN path. On exception, falls back to `self.torch_layer(*args, **kwds)` -- the original PyTorch module. If `self.device` is `None`, goes directly to the PyTorch fallback.

```python
# [run_config.py:NormalRunWithFallback.torch_dispatch]
try:
    if can_dispatch_to_ttnn(...):
        rs = DispatchManager.dispatch_to_ttnn_wrapper(...)
    else:
        rs = DispatchManager.dispatch_to_torch_wrapper(...)
except Exception as e:
    print(f"Error {e} in dispatching {func.name()}, falling back to torch")
    rs = DispatchManager.dispatch_to_torch_wrapper(...)
```

> **Warning:** This mode masks errors. Use it during initial porting to identify which operations fail on TTNN, but switch to NORMAL for production to catch regressions. The fallback path in `module_run` requires `self.torch_layer` to be set (i.e., `from_torch` must have been called with the original module).

### SEL (SELRun) -- Side-by-Side Execution and Comparison

**Purpose**: Validates TTNN correctness by running every operation on **both** TTNN and PyTorch, then comparing results.

**torch_dispatch behavior**:
1. Copies all input tensors to create independent PyTorch-only copies.
2. Runs the PyTorch path on the copies.
3. If `can_dispatch_to_ttnn()` is True, also runs the TTNN path.
4. Compares inputs (to verify TTNN did not corrupt them) and outputs (to verify numerical agreement) via `compare_fn_outputs()`.
5. Merges the TTNN tensor result into the PyTorch-tracked result (the PyTorch path provides the "golden" `.elem`, while the TTNN path provides `.ttnn_tensor`). After merging, `.elem` is set to `None` -- subsequent operations continue with the TTNN tensor exclusively.

**module_run behavior**: Same dual-execution pattern -- runs `self.torch_layer()` on copied inputs, then `self.forward()` on TTNN-transformed inputs, and compares both paths.

> **Warning:** SEL mode approximately **doubles** execution time and memory usage because every operation runs twice. Use it for debugging and validation, not production inference.

### DPL (DPLRun) -- Dual Path with Error Propagation

Similar to SEL, but with a critical difference: after comparing outputs, DPL **preserves the PyTorch `.elem`** alongside the TTNN `.ttnn_tensor` on the output wrapper (`assign_ttnn_to_torch=True`). This allows downstream operations to see both representations and track how errors compound through the computation graph.

In SEL, `.elem` is set to `None` after merging, so subsequent operations use the TTNN tensor exclusively. In DPL, `.elem` is preserved, enabling continuous dual-path verification through the entire model.

### DPL_NO_ERROR_PROP (DPLRunNoErrorProp)

A variant of DPL that **does not propagate errors** from one operation to the next. Before each TTNN dispatch, it creates **fresh copies** of the inputs from the original torch tensors via `copy_to_ttnn()`:

```python
# DPL_NO_ERROR_PROP: creates fresh TTNN tensors from torch elem for each op
ttnn_no_error_prop_args = tree_map(copy_to_ttnn(func), args)
```

This isolates each operation's TTNN error from subsequent operations, making it possible to identify which specific operation introduces numerical divergence.

### CPU (CPU)

Forces all execution to CPU. The `torch_dispatch` routes everything through `dispatch_to_torch_wrapper`. The `module_run` calls `self.torch_layer()` directly, ignoring any TTNN forward implementation.

Useful for establishing CPU baselines or debugging without hardware.

### LIGHTWEIGHT (LightweightRun)

Inherits from `NormalRun` but overrides `torch_dispatch` to **always** dispatch to PyTorch (never TTNN) at the operation level, with `wrap=False` -- results are returned as plain `torch.Tensor`, not `TorchTTNNTensor`. This requires the **CPU dispatcher** to be active:

```bash
export TT_SYMBIOTE_DISPATCHER=CPU
export TT_SYMBIOTE_RUN_MODE=LIGHTWEIGHT
```

> **Warning:** `LightweightRun` and `TracedRun` (which inherits from `LightweightRun`) require the CPU dispatcher. Running them without `TT_SYMBIOTE_DISPATCHER=CPU` raises an assertion error.

### TRACED (TracedRun)

The most complex run mode. Captures the forward pass of trace-enabled modules into TTNN's trace system, then replays them with near-zero host dispatch overhead. Detailed in the sections below.

## Run Mode Comparison Summary

| Mode | TTNN Dispatch | Module Forward | Error Handling | Dual-Path | Tracing |
|------|--------------|----------------|----------------|-----------|---------|
| NORMAL | Yes | TTNN | None (crash) | No | No |
| NORMAL_WITH_FALLBACK | Yes (try/catch) | TTNN (try/catch) | Falls back to torch | No | No |
| SEL | Yes | Both + compare | None | Yes (drop elem) | No |
| DPL | Yes | Both + compare | None | Yes (keep elem) | No |
| DPL_NO_ERROR_PROP | Yes (fresh copies) | Both + compare | Isolated errors | Yes (keep elem) | No |
| CPU | No | torch_layer only | None | No | No |
| LIGHTWEIGHT | No | Normal module_run | None | No | No |
| TRACED | No (LightweightRun) | Three-phase trace | None | No | Yes |

## Development Workflow Progression

The typical development workflow progresses through these modes:

1. **CPU** -- verify model loads and produces output without any hardware.
2. **NORMAL_WITH_FALLBACK** -- identify which layers and operations can run on TTNN; fallback catches failures gracefully.
3. **SEL** or **DPL** -- validate numerical accuracy of TTNN implementations against PyTorch ground truth.
4. **NORMAL** -- production single-device inference with full TTNN dispatch.
5. **TRACED** -- maximum performance with trace replay for the hot path.

## Run Mode Inheritance Hierarchy

```
  NormalRun                      <-- Base: TTNN dispatch + module_run
       |
       +-- NormalRunWithFallback <-- Adds try/except fallback
       +-- SELRun                <-- Dual-path: compare TTNN vs PyTorch
       +-- DPLRun                <-- Dual-path: error propagation
       +-- DPLRunNoErrorProp     <-- Dual-path: isolated errors
       +-- CPU                   <-- All ops on CPU
       +-- LightweightRun        <-- torch_dispatch always to CPU, no wrapping
              |
              +-- TracedRun      <-- Adds trace capture/replay
```

All modes except `LightweightRun` and `TracedRun` use the standard `NormalRun.new_instance()` for tensor construction and `NormalRun.to_ttnn()` / `NormalRun.to_torch()` for conversion.

`LightweightRun` overrides only `torch_dispatch` to skip TTNN dispatch. `TracedRun` inherits this (since traced modules call `forward()` directly with raw `ttnn.Tensor` arguments) and adds the three-phase `module_run()`.

## TracedRun -- Three-Phase Lifecycle

`TracedRun` in [run_config.py:TracedRun] implements a three-phase per-module lifecycle that progressively transitions from normal execution to zero-copy trace replay.

### Phase Overview

```
  +-----------+    +-------------+    +-----------+
  | Phase 1:  |    | Phase 2:    |    | Phase 3:  |
  | WARM-UP   |--->| CAPTURE     |--->| REPLAY    |
  | (Run 1)   |    | (Run 2)     |    | (Run 3+)  |
  +-----------+    +-------------+    +-----------+
  Normal forward    begin_trace ->     execute_trace
  No recording      forward() ->       Copy inputs
  Primes JIT/CCL    end_trace          to buffers
                    Allocates          Near-zero
                    persistent         host overhead
                    buffers
```

### TracedRun.configure()

Before using traced execution, the mode must be configured:

```python
TracedRun.configure(
    device=mesh_device,
    cq_id=0,                                      # Command queue ID
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,   # Memory config for persistent buffers
)
```

This initializes the trace cache, warmup key set, and stores references to the base `pre_trace_execute` / `post_trace_execute` methods for identity comparison during replay.

### Cache Key Computation

The cache key is a tuple of `(module_name, input_signatures)`:

```python
# [run_config.py:TracedRun._make_cache_key]
@staticmethod
def _make_cache_key(module_name: str, args) -> Tuple:
    return (module_name, _compute_args_signature(args))
```

The input signature is computed from tensor properties:

```python
def _compute_tensor_signature(tensor) -> Tuple:
    if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None:
        # TorchTTNNTensor with a materialized ttnn_tensor: signature includes layout
        t = tensor.ttnn_tensor
        return (tuple(t.shape), t.dtype, t.layout)
    if isinstance(tensor, ttnn.Tensor):
        return (tuple(tensor.shape), tensor.dtype, tensor.layout)
    if isinstance(tensor, torch.Tensor):
        return (tuple(tensor.shape), tensor.dtype)
    return ()
```

Note that `TorchTTNNTensor` inputs whose underlying `ttnn_tensor` has been materialized are handled by the first branch: their signature includes layout (not just shape and dtype), matching the bare `ttnn.Tensor` branch. This means a single module can have **multiple traces** for different input shapes (e.g., the prefill pass vs. decode pass of a language model). Each unique combination of module name and input tensor shapes/dtypes/layouts gets its own trace.

### Phase 1: Warm-Up (First Encounter)

On the first call to a trace-enabled module (for a given cache key), `TracedRun` performs a normal forward pass without any trace capture:

```python
# [run_config.py:TracedRun.module_run] -- warm-up branch
TracedRun._warmup_keys.add(cache_key)
_TRACE_RUNNING = True
result = self.forward(*func_args, **func_kwargs)
_TRACE_RUNNING = False
```

**Purpose**: The warm-up primes JIT compilation paths, CCL semaphore allocation, and the device memory allocator. Without this warm-up, the trace capture phase would record one-time initialization operations that should not be replayed.

The `_TRACE_RUNNING` flag is set to `True` during warm-up to prevent nested trace-enabled modules from attempting their own trace capture. Any `TTNNModule` that checks `is_trace_enabled(self)` will see `_TRACE_RUNNING == True` and fall back to normal execution.

### Phase 2: Capture (Second Encounter)

On the second call (same cache key), the trace is captured:

```python
# [run_config.py:TracedRun.module_run] -- capture branch
elif cache_key in TracedRun._warmup_keys:
    _TRACE_RUNNING = True
    entry = TracedRun._capture_trace(self, func_args, func_kwargs, cache_key)
    result = entry.trace_output
    _TRACE_RUNNING = False
```

The `_capture_trace()` static method:

1. **Allocates persistent input buffers**: For each tensor argument, creates a new `ttnn.Tensor` on device with `DRAM_MEMORY_CONFIG`. These buffers persist across all future replays.

2. **Allocates persistent keyword argument buffers**: Handles both scalar tensors and list/tuple containers (e.g., `position_embeddings = [cos, sin]`).

3. **Runs a warm-up forward**: Calls `module.forward()` with the original arguments to prime the system one more time. The output is discarded.

4. **Synchronizes the device**: `ttnn.synchronize_device(device)` ensures all in-flight operations (including CCL) complete before capture begins.

5. **Captures the trace**:
   ```python
   trace_id = ttnn.begin_trace_capture(device, cq_id=cq_id)
   trace_output = module.forward(*trace_func_args, **trace_func_kwargs)
   ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)
   ttnn.synchronize_device(device)
   ```

6. **Stores the `TraceEntry`** in the trace cache:
   ```python
   @dataclass(slots=True)
   class TraceEntry:
       trace_id: int              # TTNN trace handle
       trace_inputs: List[Any]    # Persistent input buffers
       trace_kwargs: Dict[...]    # Persistent kwarg buffers
       trace_output: Any          # Output tensor (overwritten each replay)
       device: Any                # Device reference
   ```

Note the **two** forward calls during capture: one warm-up and one actual capture. The warm-up ensures that any lazy initialization within the forward path is complete before the trace recorder begins. The `ttnn.synchronize_device()` call between them is critical -- without it, in-flight CCL operations from the warm-up can cause "Writes/Reads are not supported during trace capture" errors.

> **Warning:** All tensor allocations that happen during trace capture are baked into the trace. If a module allocates temporary buffers inside `forward()`, those buffers will be part of the trace's memory footprint. Design your `forward()` to use pre-allocated buffers when possible.

### Phase 3: Replay (Third Encounter and Beyond)

On all subsequent calls, the trace is replayed:

```python
# [run_config.py:TracedRun.module_run] -- replay branch
if cache_key in TracedRun._trace_cache:
    entry = TracedRun._trace_cache[cache_key]
    TracedRun._copy_inputs_to_trace_buffer(func_args, entry.trace_inputs)
    TracedRun._copy_kwargs_to_trace_buffer(func_kwargs, entry.trace_kwargs)
    if type(self).pre_trace_execute is not TracedRun._base_pre_trace_execute:
        self.pre_trace_execute(func_args, func_kwargs)
    ttnn.execute_trace(entry.device, entry.trace_id, cq_id=TracedRun._cq_id, blocking=False)
    result = entry.trace_output
    if type(self).post_trace_execute is not TracedRun._base_post_trace_execute:
        self.post_trace_execute(func_args, func_kwargs, result)
```

The replay flow:

```
  New inputs arrive
       |
       v
  _copy_inputs_to_trace_buffer()    -- ttnn.copy(new_input, persistent_buffer)
       |
       v
  _copy_kwargs_to_trace_buffer()    -- Same for keyword arguments
       |
       v
  pre_trace_execute()               -- Optional: CCL-sensitive buffer copies
       |
       v
  ttnn.execute_trace(trace_id)      -- Replay recorded op sequence on device
       |                               blocking=False for overlap
       v
  post_trace_execute()              -- Optional: post-processing
       |
       v
  Return entry.trace_output         -- Same tensor object; DRAM contents updated
```

The replay is extremely efficient:
- **No host-side dispatch** -- the entire op sequence is replayed from a DRAM buffer.
- **No memory allocation** -- all buffers (inputs, intermediates, outputs) were allocated during capture.
- **Non-blocking** -- `execute_trace` returns immediately; the device executes asynchronously.
- **O(1) input copy** -- only the top-level inputs and tensor kwargs need to be copied.

> **Warning:** The `trace_output` from the capture phase is the **same tensor object** returned on every replay. Its device memory (DRAM buffer) is overwritten by `execute_trace()`. Do not hold references to previous replay outputs -- they share the same buffer and will contain the most recent execution's data.

### _copy_inputs_to_trace_buffer

Copies new input data into the persistent trace buffers without creating new allocations:

```python
# [run_config.py:TracedRun._copy_inputs_to_trace_buffer]
@staticmethod
def _copy_inputs_to_trace_buffer(new_args, trace_inputs) -> None:
    trace_idx = 0
    for arg in new_args:
        if trace_idx >= len(trace_inputs):
            break
        trace_input = trace_inputs[trace_idx]
        if isinstance(arg, ttnn.Tensor):
            if arg is not trace_input:
                ttnn.copy(arg, trace_input)
            trace_idx += 1
        elif hasattr(arg, "ttnn_tensor") and arg.ttnn_tensor is not None:
            if arg.ttnn_tensor is not trace_input:
                ttnn.copy(arg.ttnn_tensor, trace_input)
            trace_idx += 1
```

The identity check (`arg is not trace_input`) avoids redundant copies when the caller passes the same buffer.

### Keyword Argument Handling

`_copy_kwargs_to_trace_buffer` handles both scalar tensor kwargs and **list/tuple of tensors** (a pattern used by position embeddings in many transformer models):

```python
for key, trace_buf in trace_kwargs.items():
    new_val = new_kwargs.get(key)
    if isinstance(trace_buf, (list, tuple)):
        for tb, nv in zip(trace_buf, new_val):
            if tb is not None:
                TracedRun._copy_one_to_trace_buffer(nv, tb)
    else:
        TracedRun._copy_one_to_trace_buffer(new_val, trace_buf)
```

## The _TRACE_RUNNING Global

The module-level variable `_TRACE_RUNNING` in [run_config.py] acts as a re-entrancy guard:

- Set to `True` during warm-up (Phase 1) and capture (Phase 2).
- Set to `False` during replay (Phase 3) -- this is important because hooks called during replay need to be able to call `ttnn.copy` without triggering trace assertions.
- Checked by `TracedRun.module_run` to prevent nested tracing.
- Checked by `TTNNModule.preprocess_weights()` and `move_weights_to_device()` to enforce that weights are already prepared.

## TTNNLayerStack

`TTNNLayerStack` in [module.py:TTNNLayerStack] is a trace-enabled container that wraps a sequence of `TTNNModule` layers into a single traceable unit. This is critical for transformer models where the same layer is repeated $N$ times.

### Why TTNNLayerStack Exists

Without `TTNNLayerStack`, each transformer layer would be traced individually. With 40 layers, that means 40 traces, 40 input buffer copies, and 40 `execute_trace()` calls per forward pass. With `TTNNLayerStack`, the entire layer stack is captured as **one trace**, reducing host dispatch overhead from $O(N_{\text{layers}})$ to $O(1)$.

```
  Without TTNNLayerStack:                 With TTNNLayerStack:

  copy_inputs(layer_0)                    copy_inputs(stack)
  execute_trace(layer_0)                  execute_trace(stack)   <-- one trace for all layers
  copy_inputs(layer_1)                    return result
  execute_trace(layer_1)
  ...
  copy_inputs(layer_39)
  execute_trace(layer_39)
```

### Implementation

```python
# [module.py:TTNNLayerStack]
@trace_enabled
class TTNNLayerStack(TTNNModule):
    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers)
        assert all(isinstance(layer, TTNNModule) for layer in self.layers)

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, **kwargs)
        return hidden_states
```

Key design points:

- **`@trace_enabled` decorator**: Marks the class so `TracedRun` captures it.
- **Calls `layer.forward()` directly**: Bypasses `layer.__call__()` / `layer.call()` and the per-layer `module_run()`. This prevents each layer from trying to manage its own trace lifecycle. Since the stack is itself a `TTNNModule`, all bookkeeping (input transformation, weight preprocessing) happens once at the stack level.
- **Single hidden_states argument**: The stack takes `hidden_states` as the positional argument and passes all keyword arguments (e.g., `attention_mask`, `position_embeddings`) to every layer.

### Weight Management Delegation

`TTNNLayerStack` delegates weight operations to its layers:

```python
def preprocess_weights_impl(self):
    for layer in self.layers:
        if isinstance(layer, TTNNModule):
            layer.preprocess_weights()

def move_weights_to_device_impl(self):
    for layer in self.layers:
        if isinstance(layer, TTNNModule):
            layer.move_weights_to_device()
```

## @trace_enabled and @trace_disabled Decorators

### @trace_enabled

```python
# [run_config.py:trace_enabled]
_TRACE_ENABLED_CLASSES: Set[Type] = set()

def trace_enabled(cls: Type) -> Type:
    _TRACE_ENABLED_CLASSES.add(cls)
    return cls
```

Marks a `TTNNModule` subclass as eligible for trace capture. During `TracedRun.module_run()`, the check `is_trace_enabled(self)` returns `True` only for instances of classes in `_TRACE_ENABLED_CLASSES`.

**When to use**: Apply to modules that:
- Have a deterministic forward pass (same inputs -> same op sequence).
- Are called repeatedly with the same input shapes.
- Represent a significant portion of compute (e.g., the full transformer layer stack).

### @trace_disabled

```python
# [run_config.py:trace_disabled]
_TRACE_DISABLED_CLASSES: Set[Type] = set()

def trace_disabled(cls: Type) -> Type:
    _TRACE_DISABLED_CLASSES.add(cls)
    return cls
```

Overrides `@trace_enabled` on a subclass. If a parent class is `@trace_enabled` but a specific subclass should not be traced:

```python
@trace_enabled
class BaseModule(TTNNModule):
    ...

@trace_disabled
class SpecialModule(BaseModule):
    # Not traced even though BaseModule is trace-enabled
    ...
```

The check in `is_trace_enabled()` uses `isinstance`, which respects inheritance:

```python
def is_trace_enabled(module) -> bool:
    return isinstance(module, tuple(_TRACE_ENABLED_CLASSES)) \
       and not isinstance(module, tuple(_TRACE_DISABLED_CLASSES))
```

### disable_trace Function Decorator

For individual functions (not classes), `disable_trace` prevents trace capture within the decorated function's scope:

```python
# [run_config.py:disable_trace]
def disable_trace(fn):
    def new_fn(*args, **kwargs):
        global _TRACE_RUNNING
        was_tracing = _TRACE_RUNNING
        _TRACE_RUNNING = True          # Pretend we're inside a trace
        try:
            return fn(*args, **kwargs)
        finally:
            _TRACE_RUNNING = was_tracing
    return new_fn
```

By setting `_TRACE_RUNNING = True`, any trace-enabled module called from within the decorated function will see it and fall back to normal execution.

## Ring Topology Requirement for Traced Execution

When using traced execution with CCL (collective communication) operations on multi-device configurations, the CCL operations must use **Ring** topology (`ttnn.Topology.Ring`) rather than **Linear** topology (`ttnn.Topology.Linear`).

### Why Ring Is Required

During trace capture, TTNN records all memory allocations and the exact sequence of operations. Linear topology CCL operations may perform **dynamic intermediate allocations** that vary between runs (e.g., temporary buffers for partial results at chain endpoints). These dynamic allocations are incompatible with trace replay, which expects the memory layout to be identical on every execution.

Ring topology operations use a fixed, symmetric communication pattern where every chip performs the same sequence of operations. This determinism makes Ring compatible with trace capture and replay.

### CCLManagerConfig Default

The `CCLManagerConfig` dataclass in [run_config.py:CCLManagerConfig] defaults to `ttnn.Topology.Linear`:

```python
@dataclass
class CCLManagerConfig:
    mesh_device: Any
    num_links: Optional[int] = None
    topology: Optional[Any] = None

    def __post_init__(self):
        if self.num_links is None:
            self.num_links = 1
        if self.topology is None:
            self.topology = ttnn.Topology.Linear
```

> **Warning:** The `CCLManagerConfig` defaults to `ttnn.Topology.Linear`, which is incompatible with traced execution. Before enabling traced execution for a multi-device model, you must explicitly override the topology to `ttnn.Topology.Ring`. Using `ttnn.Topology.Linear` inside a trace-enabled module will cause trace replay failures (corrupted results or crashes) because the replayed allocation pattern will not match the captured one. Always use `ttnn.Topology.Ring` for CCL operations inside traced modules.

See [Chapter 1, File 3](../ch01_device_topologies/03_fabric_and_routing.md) for details on fabric configuration and the Linear vs. Ring distinction.

## pre_trace_execute and post_trace_execute Hooks

`TTNNModule` provides two hooks that are called during trace replay (Phase 3 only):

### pre_trace_execute

```python
# [module.py:TTNNModule.pre_trace_execute]
def pre_trace_execute(self, func_args, func_kwargs):
    """Hook called before ttnn.execute_trace during trace replay.
    Override to copy trace-sensitive inputs to module-owned persistent
    buffers (avoiding TTNN trace allocator buffer aliasing).
    """
```

Called **after** inputs are copied to trace buffers but **before** `ttnn.execute_trace()`. The hook is only called if the subclass overrides it (checked via identity comparison with the base class method).

**Use case**: CCL operations that require their inputs to be in specific persistent buffers that are not part of the standard trace input set. For example, if a module uses `ttnn.all_gather` internally and the gather input must be in a module-owned buffer (to avoid aliasing with trace-managed buffers), `pre_trace_execute` copies the data into that module-owned buffer. A common use is for attention modules that maintain a KV cache -- the module must copy the current KV cache state into the trace's persistent buffers before `execute_trace` overwrites them.

### post_trace_execute

```python
# [module.py:TTNNModule.post_trace_execute]
def post_trace_execute(self, func_args, func_kwargs, result):
    """Hook called after ttnn.execute_trace during trace replay.
    Override to perform post-processing on trace outputs or update
    module state based on the replayed result.
    """
```

Called **after** `ttnn.execute_trace()` returns. Receives the trace output (which is the same `trace_output` object from the `TraceEntry`).

**Use case**: Updating module-level state (e.g., KV cache pointers, attention mask state, position counters) that depends on the trace output, or performing non-traceable operations that must happen after the traced computation.

## Trace Cache Management

### Cache Size and Inspection

```python
TracedRun.cache_size()           # Number of cached traces
TracedRun.cached_keys()          # List of all cache keys
```

### Releasing Traces

```python
TracedRun.release_all()          # Release ALL cached traces
TracedRun.release("module_name") # Release traces for a specific module
```

Release calls `ttnn.release_trace(device, trace_id)` for each cached trace, freeing the device DRAM occupied by the recorded operation sequence. This is important for long-running services where input shapes may change over time, leading to cache growth.

## Signpost Mode

An orthogonal profiling feature controlled by `TT_SYMBIOTE_SIGNPOST_MODE`:

```bash
export TT_SYMBIOTE_SIGNPOST_MODE=1
```

When enabled, `NormalRun.module_run()` inserts Tracy signposts before each module's forward call, enabling precise module-level profiling in Tracy traces:

```python
if NormalRun.signpost_mode is not None:
    signpost(f"{self.module_name}", f"{self.__class__.__name__}")
```

This is useful when profiling with `tracy-capture` to correlate device-side kernel execution with host-side module invocations.

---

**Next:** [`04_end_to_end_model_flow.md`](./04_end_to_end_model_flow.md)
