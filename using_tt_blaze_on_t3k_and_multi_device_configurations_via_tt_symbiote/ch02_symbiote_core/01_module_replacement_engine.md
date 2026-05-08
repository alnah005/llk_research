# 01 -- Module Replacement Engine

TT-Symbiote accelerates unmodified PyTorch models by surgically replacing individual `nn.Module` instances with TTNN-backed `TTNNModule` equivalents. The replacement is recursive, name-preserving, and exclusion-aware: a single call to `register_module_replacement_dict()` walks the model tree, substitutes every module whose class appears in a user-provided mapping, assigns hierarchical names that mirror the original PyTorch namespace, and returns a flat dictionary of all replaced modules so the caller can batch weight preprocessing. This file covers the full API surface, the `TTNNModule` lifecycle (construction through forward execution), the naming system, and the device binding flow that wires modules to hardware. Prerequisites: familiarity with `torch.nn.Module` trees and basic Python class hierarchies.

## Overview

TT-Symbiote's acceleration strategy begins at the **module level**: before any inference occurs, the user provides a mapping from PyTorch layer classes to TTNN replacement classes. The framework walks the model's module tree, swaps matching modules, and then manages their lifecycle (weight preprocessing, device placement, forward execution). This design preserves the original model's control flow and HuggingFace integration while delegating compute-heavy operations to Tenstorrent hardware.

```
  PyTorch Model (nn.Module tree)
       |
       v
  register_module_replacement_dict(model, {nn.Linear: TTNNLinearLLama, ...})
       |
       v
  Walks model._modules recursively
       |
       +-- nn.Linear found? --> TTNNLinearLLama.from_torch(old_linear)
       +-- nn.SiLU found?   --> TTNNSilu.from_torch(old_silu)
       +-- Other modules?   --> Keep as-is (stay on CPU/PyTorch)
       |
       v
  Model tree now contains TTNNModule instances
  where matched PyTorch modules once were.
```

## register_module_replacement_dict()

The top-level entry point for module replacement is `register_module_replacement_dict()` in [module_replacement.py:register_module_replacement_dict]. It takes a model, a class mapping dictionary, an optional model config, and an optional exclusion set.

### Signature

```python
def register_module_replacement_dict(
    model,                           # nn.Module -- the PyTorch model to modify in-place
    old_class_to_new_class_dict,     # Dict[type, type] -- {nn.Linear: TTNNLinearLLama, ...}
    model_config=None,               # Optional dict -- passed to each new module
    exclude_replacement=None,        # Optional Set[str] -- module names to skip
) -> Dict[str, TTNNModule]:
```

### Return Value

Returns a `Dict[str, TTNNModule]` mapping each replaced module's unique name to its new `TTNNModule` instance. This dictionary is used downstream for bulk weight preprocessing and device placement.

### Replacement Dictionary

The `old_class_to_new_class_dict` maps **PyTorch classes** (not instances) to **TTNNModule subclasses**:

```python
nn_to_ttnn = {
    nn.Linear:  TTNNLinearLLama,
    nn.SiLU:    TTNNSilu,
}
```

The matching uses `__class__` identity -- subclass relationships are not considered. If a model uses a custom `MyLinear(nn.Linear)`, it will not be replaced unless `MyLinear` is added to the dictionary explicitly.

### How the Recursion Works

1. **Builds a name map** from the model's `named_modules()` iterator, producing `{module_object: "layer.0.self_attn.q_proj", ...}`.

2. **Delegates** to `register_module_replacement_dict_with_module_names()`, which performs the recursive walk.

3. **Recursive walk** over `model._modules`: for each child module whose class appears as a key in `old_class_to_new_class_dict`, calls `initialize_module()` to create the replacement.

4. **initialize_module()** performs the actual swap:
   - Checks if the module's name is in `exclude_replacement` -- if so, returns `None` (no replacement).
   - Calls `new_class.from_torch(old_module)` to create the TTNN replacement.
   - Assigns the `_unique_name` from the name map.
   - Calls `override_children_module_names()` to propagate names to children.
   - Calls `set_model_config()` with the provided config.

5. **Handles non-standard containers**: Beyond `model._modules`, the walk also inspects public attributes that are `dict`, `list`, or `tuple` containers, replacing any matching `nn.Module` instances found within them. This catches models that store layers in non-standard ways (e.g., a Python `dict` attribute rather than `nn.ModuleDict`).

6. **Handles TTNNModule parents**: If a replacement itself contains children that need replacement (e.g., a custom `TTNNModule` that internally holds `nn.Linear` layers), the recursion continues into `TTNNModule` instances, scanning `dir(model)` and skipping `_`-prefixed and `torch_layer`-prefixed attributes.

```python
# Simplified from [module_replacement.py:register_module_replacement_dict_with_module_names]
if isinstance(model, nn.Module):
    for name, module in model._modules.items():
        if module.__class__ in old_class_to_new_class_dict:
            new_module = initialize_module(module, ...)
            if new_module is not None:
                model._modules[name] = new_module        # In-place swap
        else:
            register_module_replacement_dict_with_module_names(module, ...)  # Recurse
    # Also inspect dict/list/tuple attributes for non-standard containers
    for attr_name in dir(model):
        value = getattr(model, attr_name)
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, nn.Module) and v.__class__ in mapping:
                    value[k] = initialize_module(v, ...)
```

> **Warning:** `register_module_replacement_dict()` modifies the model **in-place**. After the call, the original `nn.Module` instances referenced in the replacement dict are no longer part of the model tree. Their parameters are still accessible through the `_fallback_torch_layer` attribute of the replacement `TTNNModule`.

> **Warning:** The attribute scan via `dir(model)` on `TTNNModule` objects skips names starting with `_`, which means private attributes that hold modules will not be discovered for replacement. If a model stores a replaceable module in `self._hidden_linear`, it will be silently skipped. Additionally, accessing attributes via `dir()` can trigger `@property` accessors, which may have side effects on certain model classes.

## The TTNNModule Base Class

`TTNNModule` in [module.py:TTNNModule] is the base class for all TTNN-accelerated modules. It is **not** a subclass of `torch.nn.Module` -- it is a standalone class with its own lifecycle. This separation is intentional: TTNN modules manage device memory and weight formats that are fundamentally different from PyTorch's parameter system, and avoiding `nn.Module` inheritance sidesteps interference with PyTorch's autograd, parameter registration, and serialization systems.

### Instance Attributes

| Attribute                  | Type                          | Purpose                                                   |
|---------------------------|-------------------------------|-----------------------------------------------------------|
| `_device`                 | `Any` (ttnn device)           | The target device (set by `to_device()` / `set_device()`) |
| `_preprocessed_weight`    | `bool`                        | Guard: `preprocess_weights()` has been called             |
| `_weights_on_device`      | `bool`                        | Guard: `move_weights_to_device()` has been called         |
| `_fallback_torch_layer`   | `nn.Module` or `None`         | Original PyTorch module (for fallback execution)          |
| `_unique_name`            | `str` or `None`               | Unique name in the model tree                             |
| `_device_state`           | `DistributedConfig` or `None` | Multi-device configuration (tensor sharding, CCL)         |
| `_model_config`           | `dict`                        | Arbitrary model config passed from replacement call       |
| `_bypass_tensor_wrapping` | `bool`                        | Optimization flag: skip `TorchTTNNTensor` wrapping        |

### Lifecycle

The `TTNNModule` lifecycle has four phases, each with a guard to prevent redundant execution:

```
  from_torch(old_module)
       |
       v
  [1] preprocess_weights()          -- CPU-side weight transformation
       |                               (tiling, dtype conversion, etc.)
       v
  [2] move_weights_to_device()      -- Copy preprocessed weights to TTNN device
       |
       v
  [3] forward(*args, **kwargs)      -- Actual inference computation on device
       |
       v
  [4] deallocate_weights()          -- Free device memory (optional)
```

### from_torch() -- Construction

The `from_torch()` classmethod is the standard constructor for replacement modules:

```python
# [module.py:TTNNModule.from_torch]
@classmethod
def from_torch(cls, torch_layer, *args, **kwargs):
    new_layer = cls(*args, **kwargs)
    new_layer._fallback_torch_layer = torch_layer
    return new_layer
```

Subclasses typically override this to extract parameters from the original PyTorch layer:

```python
class TTNNLinearLLama(TTNNModule):
    @classmethod
    def from_torch(cls, torch_layer):
        new = cls()
        new._fallback_torch_layer = torch_layer
        new.weight = torch_layer.weight.data   # Extract weight for preprocessing
        new.bias = torch_layer.bias             # May be None
        return new
```

### preprocess_weights() -- Weight Preprocessing

```python
# [module.py:TTNNModule.preprocess_weights]
def preprocess_weights(self):
    if _TRACE_RUNNING:
        assert self._preprocessed_weight, "Weights must be preprocessed before traced execution."
        return
    if not self._preprocessed_weight:
        self._preprocessed_weight = True
    else:
        return
    self.preprocess_weights_impl()
```

Key behaviors:
- **Idempotent**: The `_preprocessed_weight` guard ensures the method runs at most once.
- **Trace safety**: During traced execution (`_TRACE_RUNNING` is True), it asserts that weights are already preprocessed rather than attempting preprocessing -- traced execution requires a fully initialized module.
- **Recursive default**: The base `preprocess_weights_impl()` recurses into child `TTNNModule` attributes.

> **Warning:** The `_preprocessed_weight` flag is set to `True` **before** calling `preprocess_weights_impl()`. If the implementation raises an exception, the flag remains set and subsequent calls will silently skip preprocessing. This means a failed preprocessing cannot be retried without manually resetting `_preprocessed_weight = False`.

### move_weights_to_device() -- Device Placement

```python
# [module.py:TTNNModule.move_weights_to_device]
def move_weights_to_device(self):
    if _TRACE_RUNNING:
        assert self._weights_on_device, "Weights must be on device before traced execution."
        return
    assert self._preprocessed_weight, "Weights must be preprocessed before moving to device."
    assert self.device is not None, "Device must be set before moving weights."
    if not self._weights_on_device:
        self._weights_on_device = True
    else:
        return
    self.move_weights_to_device_impl()
```

Key behaviors:
- **Ordering enforcement**: Asserts that `preprocess_weights()` has been called first.
- **Device requirement**: Asserts that a device has been assigned via `to_device()`.
- **Idempotent**: The `_weights_on_device` guard prevents double-transfers.

### forward() -- Must Be Overridden

The base `forward()` raises `NotImplementedError`. Subclasses implement the actual TTNN computation.

### __call__() -- Dispatch Through Run Mode

The `__call__` method does **not** call `forward()` directly. Instead, it delegates to the current run mode's `module_run()`:

```python
def __call__(self, *args, **kwds):
    return self.call(*args, **kwds)

def call(self, *args, **kwds):
    return TENSOR_RUN_IMPLEMENTATION.module_run(self, *args, **kwds)
```

The `TENSOR_RUN_IMPLEMENTATION` is a class (e.g., `NormalRun`, `TracedRun`) selected by the current run mode (see [Chapter 2, File 3](./03_run_modes.md)). The `module_run()` method handles:
- Input tensor transformation (wrapping, conversion to TTNN, device placement)
- Calling `preprocess_weights()` and `move_weights_to_device()`
- Calling `forward()` with transformed inputs
- Post-processing results
- Timing/profiling

### deallocate_weights() -- Memory Reclamation

```python
def deallocate_weights(self):
    self.deallocate_weights_impl()
    self._weights_on_device = False
```

Frees device memory occupied by weights. The `deallocate_weights_after` decorator can be applied to `forward()` to automatically deallocate after each call, useful for memory-constrained scenarios:

```python
@deallocate_weights_after
def forward(self, x):
    return ttnn.linear(x, self.tt_weight)
```

## Weight Preprocessing Pipeline

The full weight pipeline has two stages, each with an `_impl` method that subclasses override:

```
  preprocess_weights()
       |
       +-- Guard: _preprocessed_weight?  --> return if already done
       +-- Set _preprocessed_weight = True
       |
       v
  preprocess_weights_impl()              <-- OVERRIDE THIS
       |
       +-- (Default: recurse into children)
       +-- (Subclass: tile weights, convert dtypes, transpose, etc.)
       |
       v
  move_weights_to_device()
       |
       +-- Guard: _weights_on_device?    --> return if already done
       +-- Assert: _preprocessed_weight must be True
       +-- Assert: device must be set
       +-- Set _weights_on_device = True
       |
       v
  move_weights_to_device_impl()          <-- OVERRIDE THIS
       |
       +-- (Default: recurse into children)
       +-- (Subclass: ttnn.to_device(weight, self.device))
```

The default implementations in the base class simply recurse into any child `TTNNModule` instances stored as instance attributes:

```python
def preprocess_weights_impl(self):
    for child in self.__dict__.values():
        if isinstance(child, TTNNModule):
            child.preprocess_weights()
    return self

def move_weights_to_device_impl(self):
    for child in self.__dict__.values():
        if isinstance(child, TTNNModule):
            child.move_weights_to_device()
    return self
```

## The exclude_replacement Set

The `exclude_replacement` parameter to `register_module_replacement_dict()` is an `Optional[Set[str]]` containing module names (as they appear in `model.named_modules()`) that should **not** be replaced, even if their class matches the replacement dictionary.

```python
# Keep the final language model head on CPU
persistent_weights = {"lm_head"}
modules = register_module_replacement_dict(
    model, nn_to_ttnn, model_config=None,
    exclude_replacement=persistent_weights
)
```

In `initialize_module()`, the check is:

```python
# [module_replacement.py:initialize_module]
if old_module in module_names and module_names[old_module] in exclude_replacement:
    return None  # Do not replace
```

When `None` is returned, the original module stays in the tree unchanged.

> **Warning:** The exclusion is by **module name string**, not by class. If two `nn.Linear` layers have names `"model.embed_tokens"` and `"lm_head"`, excluding `"lm_head"` keeps only that specific linear layer on CPU while all others get replaced. Be precise with name strings -- partial matches are not supported.

## Module Name Assignment

Each `TTNNModule` has a `_unique_name` that identifies it within the model tree. This name is used for:
- Cache keys in traced execution
- Timing/profiling entries in `DispatchManager`
- Debug logging

### Assignment During Replacement

When `initialize_module()` creates a replacement, it assigns the name from the pre-built name map:

```python
if old_module in module_names:
    new_module._unique_name = module_names[old_module]
    new_module.override_children_module_names()
```

### Fallback Name Generation

If no name is assigned (e.g., a module created outside the replacement flow), the `module_name` property generates one from the class name and object id:

```python
@property
def module_name(self):
    if self._unique_name is None:
        self._unique_name = f"{self.__class__.__name__}_{id(self)}"
    return self._unique_name
```

### Recursive Name Propagation

After assigning a name to a parent module, `override_children_module_names()` propagates hierarchical names to all children via `set_module_name_recursively()`:

```python
# [module.py:set_module_name_recursively]
def set_module_name_recursively(module, prefix=""):
    for name, child in module.__dict__.items():
        if isinstance(child, TTNNModule):
            child._unique_name = f"{prefix}.{name}"
            child.override_children_module_names()  # Recurse
        elif isinstance(child, torch.nn.Module):
            set_module_name_recursively(child, f"{prefix}.{name}")
        elif isinstance(child, dict):
            for k, v in child.items():
                if isinstance(v, TTNNModule):
                    v._unique_name = f"{prefix}.{name}[{k}]"
                    # ...
        elif isinstance(child, (list, tuple)):
            for i, v in enumerate(child):
                if isinstance(v, TTNNModule):
                    v._unique_name = f"{prefix}.{name}[{i}]"
                    # ...
```

This produces names like `model.layers[0].self_attn.q_proj`, `model.layers[0].mlp.gate_proj`, etc., mirroring the hierarchical structure of the original PyTorch model.

### named_modules() and named_children()

`TTNNModule` provides its own `named_modules()` and `named_children()` iterators that mirror `torch.nn.Module`'s interface but operate on `__dict__` attributes rather than `_modules`. The two iterators differ in scope: `named_modules()` only recurses into direct `TTNNModule` and `nn.Module` attributes found in `__dict__`, while `named_children()` additionally handles dict, list, and tuple containers (iterating into them to find child modules). The `named_children()` method explicitly excludes `_fallback_torch_layer` to avoid yielding the original PyTorch module as a child.

## DeviceArch and run_on_devices

While device topologies are covered in detail in [Chapter 1](../ch01_device_topologies/01_physical_topologies.md), two constructs from [module.py] are relevant to module replacement:

- **`DeviceArch`** enum: Enumerates all supported device architectures (N150, N300, T3K, TG, P150, P300, P150x4, P150x8, BHGLX).
- **`run_on_devices(*allowed_archs)`** decorator: Gates a module's `forward()` method to only execute on specific architectures by reading the `MESH_DEVICE` environment variable at call time.

These are used by TTNNModule subclasses to provide topology-specific forward implementations, as described in [Chapter 1, File 2](../ch01_device_topologies/02_mesh_device_initialization.md).

## set_device() -- Binding Modules to Hardware

After replacement, all TTNN modules must be bound to a device. The `set_device()` function in [device_management.py:set_device] recursively walks the model tree and performs several operations:

```python
def set_device(obj, device, device_init=DeviceInit, **kwargs):
    def _set_device_recursive(current_obj, parent_is_ttnn=False):
        if isinstance(current_obj, TTNNModule):
            _initialize_module_on_device(current_obj, device, device_init)
            if not getattr(current_obj, "_bypass_tensor_wrapping", False):
                current_obj._bypass_tensor_wrapping = parent_is_ttnn
            for value in current_obj.__dict__.values():
                _set_device_recursive(value, parent_is_ttnn=True)
```

For each `TTNNModule`:

1. **`to_device(device)`** -- stores the MeshDevice reference.
2. **`set_device_state(DistributedConfig)`** -- if `device.get_num_devices() > 1`, initializes the distributed configuration (tensor sharding strategy, CCL manager) via `DeviceInit.init_state()`.
3. **`_bypass_tensor_wrapping`** -- set to `True` for TTNN children of TTNN parents to avoid redundant `TorchTTNNTensor` wrapping (see [02_dispatch_and_tensor_wrapping.md](./02_dispatch_and_tensor_wrapping.md)). The `getattr` guard ensures that manually-set bypass flags are not overridden.

Additionally, `set_device()` wraps every `nn.Module.forward` and every `TTNNModule.call` with `timed_call()`, which records execution duration in the `DispatchManager` under the `"TorchModules"` backend. After the recursive walk, it calls `draw_model_graph(obj)` to generate a model structure visualization (unless `dump_visualization=False`).

## DeviceInit -- Device State Factory

The `DeviceInit` class in [device_management.py:DeviceInit] maintains a class-level `DEVICE_TO_STATE_DICT` that maps each device to its `DistributedConfig`. The `init_state()` classmethod creates a new `DistributedConfig` on first use and caches it:

```python
class DeviceInit:
    DEVICE_TO_STATE_DICT = {}

    @classmethod
    def init_state(cls, device):
        if device not in cls.DEVICE_TO_STATE_DICT:
            res = cls.init_state_impl(device)
            cls.DEVICE_TO_STATE_DICT[device] = res
        return cls.DEVICE_TO_STATE_DICT[device]

    @classmethod
    def init_state_impl(cls, device) -> DistributedConfig:
        return DistributedConfig(device)
```

Users can subclass `DeviceInit` to customize the `DistributedConfig` (e.g., different tensor sharding strategies or CCL configurations) and pass the subclass to `set_device()` via the `device_init` parameter.

## Complete Replacement and Initialization Flow

```
model = AutoModelForCausalLM.from_pretrained(...)
    |
    v
nn_to_ttnn = {nn.Linear: TTNNLinearLLama, nn.SiLU: TTNNSilu}
    |
    v
modules = register_module_replacement_dict(model, nn_to_ttnn, exclude_replacement={...})
    |                          |
    |  For each matching module:
    |     from_torch(old_module) --> new TTNNModule
    |     _unique_name = "model.layers.0.mlp.gate_proj"
    |     override_children_module_names()
    |     set_model_config(model_config)
    |     model._modules[name] = new_module
    |
    v
set_device(model, mesh_device)
    |                          |
    |  For each TTNNModule:
    |     to_device(mesh_device)
    |     set_device_state(DistributedConfig(mesh_device))   [if multi-device]
    |     _bypass_tensor_wrapping = (parent is TTNNModule)
    |
    |  For each nn.Module:
    |     forward = timed_call(forward, name, class_name)
    |
    v
for k, v in modules.items():
    v.preprocess_weights()       # Phase 1: CPU-side weight transforms
    v.move_weights_to_device()   # Phase 2: host-to-device transfer
    |
    v
model.generate(**inputs)         # All replaced modules run on TTNN
```

---

**Next:** [`02_dispatch_and_tensor_wrapping.md`](./02_dispatch_and_tensor_wrapping.md)
