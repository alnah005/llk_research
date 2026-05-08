# 02 -- Dispatch and Tensor Wrapping

This file explains the tensor-level dispatch machinery that enables TT-Symbiote to intercept individual PyTorch operations and route them to TTNN hardware. It covers the `TorchTTNNTensor` subclass and its dual-backing design, the `__torch_dispatch__` mechanism, the pluggable dispatcher system, the `DistributedTensorConfig` for multi-device tensors, the `_bypass_tensor_wrapping` optimization for TTNN-to-TTNN parent-child communication, `fast_unwrap_to_device`, the `compose_transforms` utility, and `DispatchManager` timing infrastructure. Prerequisites: understanding of the module replacement engine from [`01_module_replacement_engine.md`](./01_module_replacement_engine.md) and basic knowledge of PyTorch's `__torch_dispatch__` protocol.

## Architecture Overview

TT-Symbiote intercepts PyTorch operations at two levels:

1. **Module level**: `TTNNModule.__call__()` routes entire module invocations through the run mode's `module_run()` method (covered in [File 1](./01_module_replacement_engine.md)).
2. **Operation level**: `TorchTTNNTensor.__torch_dispatch__()` intercepts individual `aten` operations (e.g., `aten::add.Tensor`, `aten::mm`) and decides whether each goes to TTNN or falls back to PyTorch.

These two levels work together: when a replaced module's `forward()` executes TTNN-native code, intermediate tensors may still trigger `aten` operations through PyTorch's operator dispatch (e.g., `torch.add(a, b)` when both `a` and `b` are `TorchTTNNTensor` instances). The dispatch layer catches these and routes them to TTNN.

```
  User code: model(input)
       |
       v
  TTNNModule.__call__() --> module_run()
       |
       v
  TTNNModule.forward()
       |
       +-- ttnn.linear(...)          # Direct TTNN call (no dispatch needed)
       +-- torch.add(a, b)           # aten op on TorchTTNNTensor
              |
              v
         __torch_dispatch__()
              |
              +-- can_dispatch_to_ttnn?
              |       YES --> dispatch_to_ttnn()  --> TTNN handler
              |       NO  --> dispatch_to_torch() --> PyTorch fallback
              |
              v
         Result: TorchTTNNTensor
```

## TorchTTNNTensor

`TorchTTNNTensor` in [tensor.py:TorchTTNNTensor] is a `torch.Tensor` subclass created via `torch.Tensor._make_wrapper_subclass`. It maintains two backing stores:

| Attribute     | Type                | Description                                           |
|---------------|---------------------|-------------------------------------------------------|
| `.elem`       | `torch.Tensor`      | The PyTorch-side backing tensor (CPU or meta device)  |
| `.ttnn_tensor`| `ttnn.Tensor`       | The TTNN-side backing tensor (on Tenstorrent device)  |

At any given time, one or both may be populated:

```
  State 1: Just created from a PyTorch tensor
    .elem = torch.Tensor(...)     .ttnn_tensor = None

  State 2: Converted to TTNN (via .to_ttnn property)
    .elem = torch.Tensor(...)     .ttnn_tensor = ttnn.Tensor(...)

  State 3: Result from TTNN op (elem freed for memory)
    .elem = None                  .ttnn_tensor = ttnn.Tensor(...)

  State 4: Converted back to PyTorch (via .to_torch property)
    .elem = torch.Tensor(...)     .ttnn_tensor = ttnn.Tensor(...)
```

### Construction via __new__

`TorchTTNNTensor.__new__` delegates to the current run mode's `new_instance()` static method. In `NormalRun.new_instance()`:

```python
# [run_config.py:NormalRun.new_instance] (simplified)
@staticmethod
def new_instance(cls, elem, *args, **kwargs):
    ttnn_tensor = None
    if isinstance(elem, ttnn.Tensor):
        ttnn_tensor = elem
        elem = get_empty_torch_tensor_from_ttnn(ttnn_tensor, ...)  # meta tensor for shape/dtype
        delete_elem = True
    # ...
    r = torch.Tensor._make_wrapper_subclass(
        cls, output_shape, strides=strides, dtype=output_dtype,
        device="cpu", requires_grad=requires_grad, ...
    )
    r.ttnn_tensor = ttnn_tensor
    r.elem = elem if not delete_elem else None
    r.set_distributed_tensor_config(get_default_distributed_tensor_config(...))
    return r
```

Key behaviors:
- When wrapping a `ttnn.Tensor`, the `.elem` is set to `None` (or a meta tensor for shape inference) and `.ttnn_tensor` holds the device tensor.
- When wrapping a `torch.Tensor`, `.elem` holds the CPU tensor and `.ttnn_tensor` is `None` until explicitly converted.
- A `DistributedTensorConfig` is automatically assigned based on the current device state (see below).
- Double-wrapping is prevented: wrapping a `TorchTTNNTensor` inside another raises an assertion error.

### Lazy Conversion Properties

The `.to_ttnn` and `.to_torch` properties perform lazy conversion:

**`.to_ttnn`** -- Convert to TTNN tensor:
```python
# [run_config.py:NormalRun.to_ttnn]
if self.ttnn_tensor is not None:
    return self.ttnn_tensor                    # Already exists
self.ttnn_tensor = ttnn.from_torch(
    self.elem.cpu(),
    dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype),
    mesh_mapper=self.ttnn_distributed_tensor_config.mesh_mapper if ... else None,
    layout=ttnn.TILE_LAYOUT if self.dtype == torch.bool else None,
)
return self.ttnn_tensor
```

**`.to_torch`** -- Convert to PyTorch tensor:
```python
# [run_config.py:NormalRun.to_torch]
if self.elem is not None and self.elem.device.type != "meta" and self.ttnn_tensor is None:
    return self.elem                           # Already a real CPU tensor
# Otherwise, convert from TTNN:
result = ttnn.to_torch(self.ttnn_tensor,
    mesh_composer=self.ttnn_distributed_tensor_config.mesh_composer if is_mesh_device else None
).to(self.device, self.dtype)
self.elem = result
return self.elem
```

### Shape Property

The `.shape` property accounts for distributed tensors by applying the logical shape function:

```python
@property
def shape(self):
    if self.ttnn_distributed_tensor_config is not None and self.ttnn_tensor is not None:
        return self.ttnn_distributed_tensor_config.get_logical_shape(self.ttnn_tensor.shape)
    return self.elem.shape if self.elem is not None else tuple(int(i) for i in self.ttnn_tensor.shape)
```

This ensures that code checking tensor shapes sees the **logical** (unsplit) shape even when the tensor is physically sharded across devices.

### Operator Overloads

`TorchTTNNTensor` overloads common operators (`__add__`, `__mul__`, `__sub__`, `__matmul__`, etc.) to route through `torch.*` functions, which in turn trigger `__torch_dispatch__`:

```python
def __mul__(self, other):
    return torch.mul(self, other)

def __add__(self, other):
    return torch.add(self, other)

def __matmul__(self, other):
    return torch.matmul(self, other)
```

This is how Python-level expressions like `x + y` or `x @ w` enter the dispatch machinery when `x` is a `TorchTTNNTensor`.

## __torch_dispatch__ -- The Operation Router

The `__torch_dispatch__` classmethod is PyTorch's hook for intercepting all `aten` operations on tensor subclasses. `TorchTTNNTensor` delegates it to the current run mode:

```python
# [tensor.py:TorchTTNNTensor.__torch_dispatch__]
@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    return TENSOR_RUN_IMPLEMENTATION.torch_dispatch(cls, func, types, args, kwargs)
```

In `NormalRun.torch_dispatch()`, the decision flow is:

```python
# [run_config.py:NormalRun.torch_dispatch]
@staticmethod
def torch_dispatch(cls, func, types, args=(), kwargs=None):
    can_to_ttnn = can_dispatch_to_ttnn(func.name(), args, kwargs)
    if can_to_ttnn:
        rs = DispatchManager.dispatch_to_ttnn_wrapper(func, args, kwargs)
    else:
        rs = DispatchManager.dispatch_to_torch_wrapper(func, args, kwargs)
    return rs
```

### can_dispatch_to_ttnn()

The `can_dispatch_to_ttnn()` function in [default_dispatcher.py:can_dispatch_to_ttnn] determines whether an operation can run on TTNN. The decision depends on three criteria:

1. **Handler exists**: The operation's `aten` name must be present in the handler registry (`_get_func_to_ttnn_compatible()`). The registry maps approximately 70 `aten` operations to TTNN handlers (e.g., `aten::mul.Tensor` -> `handle_mul`, `aten::mm` -> `handle_bmm`, `aten::_softmax` -> `handle_softmax`).

2. **TTNN tensor present**: At least one argument must be a `TorchTTNNTensor` with a non-None, allocated `.ttnn_tensor` that has a device assigned. This ensures we never try to dispatch a pure-CPU tensor to TTNN.

3. **Operation-specific validation**: Certain operations have additional requirements:
   - `aten::slice.Tensor`: Requires integer arguments for start, end, step.
   - `aten::addmm`: Does not support keyword arguments.
   - `aten::index.Tensor`: Requires exactly one 1D tensor index.
   - `aten::constant_pad_nd`: Only supports rank-4 tensors.
   - `aten::sum`: Only supports float32, bfloat16, bfloat8_b, and uint32 dtypes.
   - Operations with unsupported dtypes (checked against `TORCH_TO_TTNN` mapping) are rejected.

When an operation fails and has no handler, it logs a one-time message suggesting the operation be mapped to TTNN, then returns `False`.

### dispatch_to_ttnn()

When `can_dispatch_to_ttnn()` returns `True`, the dispatch path is:

```
  DispatchManager.dispatch_to_ttnn_wrapper(func, args, kwargs)
       |
       v
  dispatch_to_ttnn(func.name(), ttnn_args, ttnn_kwargs)    # [dispatcher.py]
       |
       v
  get_active_dispatcher().dispatch_to_ttnn(func_name, args, kwargs)
       |
       v
  _get_func_to_ttnn_compatible()[func_name](func_name, args, kwargs)  # handler call
       |
       v
  e.g., handle_mul(func_name, args, kwargs) --> TorchTTNNTensor(ttnn.multiply(...))
```

Each handler:
1. Extracts `TorchTTNNTensor` inputs from `args`.
2. Converts to `ttnn.Tensor` via the `.to_ttnn` property.
3. Ensures inputs are on the same device (transfers if needed).
4. Calls the corresponding `ttnn.*` function.
5. Wraps the result in a new `TorchTTNNTensor`.
6. Cleans up any temporary tensors.

### dispatch_to_torch() -- PyTorch Fallback

When an operation cannot be dispatched to TTNN, it falls back to PyTorch:

```python
# [run_config.py:DispatchManager.dispatch_to_torch_wrapper] (simplified)
with no_dispatch():                                        # Prevent infinite recursion
    func_args = tree_map(unwrap_to_torch(func), torch_args)    # Convert all to torch.Tensor
    func_kwargs = tree_map(unwrap_to_torch(func), torch_kwargs)
    func_res = func(*func_args, **func_kwargs)             # Execute on CPU
    rs = tree_map(wrap_from_torch, func_res)               # Re-wrap results
```

### The no_dispatch() Context Manager

The `no_dispatch()` context manager (`torch._C._DisableTorchDispatch`) is critical -- it prevents the fallback execution from re-triggering `__torch_dispatch__`, which would cause infinite recursion. Without it, calling `func(*func_args)` on a `torch.Tensor` that was unwrapped from a `TorchTTNNTensor` could re-enter the dispatch machinery.

```python
# [run_config.py:no_dispatch]
@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard
```

> **Warning:** `no_dispatch()` uses a private PyTorch API (`torch._C._DisableTorchDispatch`). This is an internal implementation detail that could change between PyTorch versions. TT-Symbiote pins to specific PyTorch versions to avoid surprises.

## The Pluggable Dispatcher System

The dispatch system is pluggable via a registry in [dispatcher_config.py:dispatcher_config]. Dispatchers are selected via the `TT_SYMBIOTE_DISPATCHER` environment variable or `set_dispatcher()`:

| Dispatcher  | Env Value      | Behavior                                                    |
|-------------|----------------|-------------------------------------------------------------|
| DEFAULT     | `"DEFAULT"`    | Full TTNN dispatch with all ~70 operation handlers          |
| CPU         | `"CPU"`        | Always returns `False` from `can_dispatch_to_ttnn()` -- all ops run on CPU |
| DEBUG       | `"DEBUG"`      | Wraps DEFAULT with verbose logging of every dispatch decision |
| TENSOR_OPS  | `"TENSOR_OPS"` | Specialized dispatcher for tensor operations                |

The CPU dispatcher is notable: it has a trivial implementation that always returns `False` for `can_dispatch_to_ttnn()` and raises `NotImplementedError` for `dispatch_to_ttnn()`. It is **required** when using the `LIGHTWEIGHT` or `TRACED` run modes (which inherit from `LightweightRun`):

```python
# [run_config.py:get_tensor_run_implementation]
if issubclass(result, LightweightRun):
    assert get_active_dispatcher() == cpu_dispatcher, \
        "CPU dispatcher needs to be active to run LightweightRun modes."
```

> **Warning:** When no `TT_SYMBIOTE_DISPATCHER` environment variable is set and no programmatic call to `set_dispatcher()` is made, the default dispatcher is `"CPU"`, not `"DEFAULT"`. This is a common source of confusion -- if you see all operations falling back to PyTorch despite having TTNN handlers, check that `TT_SYMBIOTE_DISPATCHER=DEFAULT` is set.

## DistributedTensorConfig

`DistributedTensorConfig` in [run_config.py:DistributedTensorConfig] is a dataclass that controls how a tensor is distributed across a multi-device mesh. It is stored per-tensor on `TorchTTNNTensor` instances.

```python
@dataclass
class DistributedTensorConfig:
    mesh_mapper: Any          # How to distribute tensor across devices
    mesh_composer: Any        # How to reassemble tensor from devices
    logical_shape_fn: Any     # Optional: maps physical shape to logical shape
```

### mesh_mapper

Controls the sharding strategy when converting a tensor to TTNN via `ttnn.from_torch()`:

- **`ttnn.ShardTensor2dMesh(mesh_device, mesh_shape, (0, -1))`**: Shard the tensor across both mesh dimensions -- dim 0 across rows (axis 0) and the last dimension across columns (axis 1).
- **`ttnn.ReplicateTensorToMesh(mesh_device)`**: Replicate the full tensor to every device in the mesh.

### mesh_composer

Controls reassembly when converting back via `ttnn.to_torch()`:

- **`ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, (0, -1))`**: Concatenate shards back along the sharded dimensions.

### logical_shape_fn

An optional callable that maps the physical per-device tensor shape to the logical (unsplit) shape. For batch-channel sharding on an $(R, C)$ mesh:

$$\text{logical\_shape} = (B_{\text{per-device}} \times R, \ldots, C_{\text{hidden,per-device}} \times C)$$

```python
# [run_config.py:logical_shape_for_batch_channel_sharding]
def logical_shape_for_batch_channel_sharding(mesh_shape):
    def _logical_shape(shape):
        shape = list(shape)
        return tuple([shape[0] * mesh_shape[0]] + shape[1:-1] + [shape[-1] * mesh_shape[1]])
    return _logical_shape
```

This means a per-device tensor of shape `(B/R, ..., C_hidden/C)` reports a logical shape of `(B, ..., C_hidden)` -- the shape the model's control flow expects.

### Default Configuration

When a `TorchTTNNTensor` is created and a multi-device mesh is active, `get_default_distributed_tensor_config()` automatically assigns the default config from the `DeviceInit.DEVICE_TO_STATE_DICT`:

```python
# [run_config.py:get_default_distributed_tensor_config]
def get_default_distributed_tensor_config(mesh_device=None, torch_tensor=None, module_name=None):
    state = DeviceInit.DEVICE_TO_STATE_DICT.get(mesh_device) or next(iter(DeviceInit.DEVICE_TO_STATE_DICT.values()))
    if torch_tensor is not None:
        return state.get_tensor_config_for_tensor(module_name, torch_tensor)
    return state.tensor_config
```

If a tensor's dimensions are not evenly divisible by the mesh shape, `get_tensor_config_for_tensor()` falls back to replication with a warning.

> **Warning:** The automatic tensor config assignment uses the **first device** in `DEVICE_TO_STATE_DICT` if no specific mesh device is provided. In multi-mesh setups (e.g., pipeline parallelism with submeshes), ensure the correct device is used by explicitly setting the module's `device_state`. Tensors with shapes not evenly divisible by the mesh dimensions are silently replicated rather than sharded, which can lead to unexpected memory usage.

## The _bypass_tensor_wrapping Optimization

By default, when a `TTNNModule`'s `module_run()` processes inputs, it applies a three-step transform pipeline:

```python
transform = compose_transforms(
    wrap_to_torch_ttnn_tensor,    # Ensure inputs are TorchTTNNTensor
    to_ttnn_wrap,                 # Convert to raw ttnn.Tensor via .to_ttnn
    set_device_wrap(self.device)  # Move to target device
)
```

This pipeline converts every input through `TorchTTNNTensor`, which involves wrapper allocation, shape tracking, and distributed config assignment. When a `TTNNModule` is called by another `TTNNModule` (parent-child relationship), the inputs are already `ttnn.Tensor` on the correct device -- the full wrapping pipeline is unnecessary overhead.

The bypass flag is set automatically by `set_device()` during device initialization -- see [File 01, "set_device() -- Binding Modules to Hardware"](./01_module_replacement_engine.md) for the `_set_device_recursive` logic that marks `TTNNModule` children of `TTNNModule` parents with `_bypass_tensor_wrapping = True`.

### How Bypass Works in module_run()

When `_bypass_tensor_wrapping` is `True`, the run mode uses a lightweight transform instead of the full pipeline:

```
Standard Path (bypass=False):       Bypass Path (bypass=True):

  torch.Tensor input                   TorchTTNNTensor or ttnn.Tensor input
       |                                    |
       v                                    v
  wrap_to_torch_ttnn_tensor           fast_unwrap_to_device(device)
       |                                    |
       v                                    v
  to_ttnn_wrap                         raw ttnn.Tensor on device
       |                                    (no wrapper, no config)
       v
  set_device_wrap(device)
       |
       v
  TorchTTNNTensor on device
```

```python
# [run_config.py:NormalRun.module_run] (simplified)
bypass = getattr(self, "_bypass_tensor_wrapping", False)
if bypass:
    transform = fast_unwrap_to_device(self.device)
    _map = flat_map_bypass          # Optimized mapping function
else:
    transform = compose_transforms(wrap_to_torch_ttnn_tensor, to_ttnn_wrap, set_device_wrap(self.device))
    _map = tree_map

func_args = _map(transform, args)
```

And the result post-processing is also skipped:

```python
if bypass:
    result = self.forward(*func_args, **func_kwargs)     # Raw ttnn.Tensor in/out
else:
    result = post_process_ttnn_module_output(self, self.forward(*func_args, **func_kwargs))
```

## fast_unwrap_to_device

`fast_unwrap_to_device()` in [run_config.py:fast_unwrap_to_device] is the lightweight transform used in bypass mode. It extracts the raw `ttnn.Tensor` from a `TorchTTNNTensor` (or passes through a bare `ttnn.Tensor`) and ensures it is on the correct device:

```python
def fast_unwrap_to_device(device):
    def _transform(e):
        if isinstance(e, TorchTTNNTensor):
            t = e.ttnn_tensor if e.ttnn_tensor is not None else e.to_ttnn
            if device is not None and t.device() != device:
                t = ttnn.to_device(t, device)
            return t                              # Returns raw ttnn.Tensor, NOT TorchTTNNTensor
        elif isinstance(e, ttnn.Tensor):
            if device is not None and e.device() != device:
                e = ttnn.to_device(e, device)
            return e
        return e
    return _transform
```

The key difference from the standard pipeline: `fast_unwrap_to_device` returns **raw `ttnn.Tensor`** objects, not `TorchTTNNTensor` wrappers. This means:
- No wrapper allocation overhead.
- No distributed config tracking.
- No shape property indirection.
- The child module's `forward()` works directly with `ttnn.Tensor` -- operations like `ttnn.matmul(x, w)` are direct TTNN calls with no dispatch interception.

> **Warning:** When `_bypass_tensor_wrapping` is True, the module's `forward()` receives raw `ttnn.Tensor` inputs, not `TorchTTNNTensor`. Using `torch.*` operations on these inputs will fail because they are not `torch.Tensor` subclasses. The `forward()` implementation must use `ttnn.*` operations exclusively.

## The compose_transforms Pipeline

The standard (non-bypass) input pipeline is built using `compose_transforms()`:

```python
# [run_config.py:compose_transforms]
def compose_transforms(*transforms):
    def _composed(e):
        result = e
        for transform in transforms:
            result = transform(result)
        return result
    _composed.__name__ = "_".join([t.__name__ for t in transforms])
    return _composed
```

The three stages of the standard pipeline:

```
  Input (torch.Tensor or TorchTTNNTensor or other)
       |
       v
  [1] wrap_to_torch_ttnn_tensor(e)
       |  - If e is a plain torch.Tensor (not TorchTTNNTensor), wrap it.
       |  - If e is a ttnn.Tensor, wrap it.
       |  - If e is already TorchTTNNTensor, pass through.
       v
  [2] to_ttnn_wrap(e)
       |  - Calls e.to_ttnn to ensure .ttnn_tensor is populated.
       |  - Returns the raw ttnn.Tensor (the value of .ttnn_tensor), NOT the wrapper.
       v
  [3] set_device_wrap(device)(e)
       |  - e is now a bare ttnn.Tensor (from step 2).
       |  - Primary path: isinstance(e, ttnn.Tensor) branch moves it to device.
       |  - Also handles TorchTTNNTensor inputs via a secondary branch.
       v
  Output: ttnn.Tensor on the target device
  (post_process_ttnn_module_output re-wraps the forward() result into TorchTTNNTensor)
```

The `wrap_to_torch_ttnn_tensor` function handles both `torch.Tensor` and bare `ttnn.Tensor` inputs:

```python
# [run_config.py:wrap_to_torch_ttnn_tensor]
def wrap_to_torch_ttnn_tensor(e):
    if isinstance(e, torch.Tensor) and not isinstance(e, TorchTTNNTensor):
        result = TorchTTNNTensor(e)
    elif not isinstance(e, TorchTTNNTensor) and isinstance(e, ttnn.Tensor):
        result = TorchTTNNTensor(e)
    else:
        result = e
    return result
```

The `set_device_wrap` function handles both bare `ttnn.Tensor` (the primary path after `to_ttnn_wrap`) and `TorchTTNNTensor` inputs:

```python
# [run_config.py:set_device_wrap]
def set_device_wrap(device):
    def _set_device_wrap(e):
        if isinstance(e, ttnn.Tensor):                         # Primary path: bare ttnn.Tensor
            if e.device() != device:
                e = ttnn.to_device(e, device)
        elif isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None and e.ttnn_tensor.device() != device:
            e.ttnn_tensor = ttnn.to_device(e.ttnn_tensor, device)
        return e
    return _set_device_wrap
```

The `isinstance(e, ttnn.Tensor)` branch is the one actually exercised in the standard pipeline, since `to_ttnn_wrap` returns a raw `ttnn.Tensor`.

A concrete trace through the pipeline is provided in [File 4](./04_end_to_end_model_flow.md) using the GLM-4 test pattern.

## DispatchManager -- Timing and Context Tracking

`DispatchManager` in [run_config.py:DispatchManager] is a static class that serves two purposes:

### Module Context Stack

`DispatchManager` maintains a stack of module names to track which module is currently executing. This enables accurate attribution of `aten` operations to their parent module in profiling:

```python
DispatchManager.set_current_module_name("model.layers.0.self_attn")  # Push
# ... forward() executes, individual aten ops inherit this module name ...
DispatchManager.set_current_module_name(None)                         # Pop
```

### Timing Records

Every operation (both TTNN and PyTorch fallback) is timed and recorded:

```python
DispatchManager.record_timing(
    backend="TTNN",              # or "Torch", "TorchModules"
    module_name="model.layers.0.self_attn.q_proj.TTNN::aten::mm",
    func_name="TTNN::aten::mm",
    attrs={},
    duration=0.0023,             # seconds
)
```

The timing infrastructure has three levels:

```
Level 1: Module-level timing (via timed_call wrapper set by set_device())
    Records: "TorchModules" backend, module_name, class_name, duration

Level 2: Module lifecycle timing (in module_run)
    Records: "TTNN" backend, module_name, "{ClassName}_preprocess_weights", duration
    Records: "TTNN" backend, module_name, "{ClassName}_forward", duration

Level 3: Per-operation timing (in dispatch wrappers)
    Records: "TTNN" or "Torch" backend, "{module_name}.aten::op_name", op_name, duration
```

These records can be exported via `DispatchManager.save_stats_to_file("timing.csv")` for analysis (covered in [File 4](./04_end_to_end_model_flow.md)).

### Disabling Timing

In production, timing overhead can be eliminated:

```python
DispatchManager.DisableTiming()
```

This sets `DispatchManager.ENABLED = False`, causing `record_timing` to return immediately without recording.

---

**Next:** [`03_run_modes.md`](./03_run_modes.md)
