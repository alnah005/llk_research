# TT-Symbiote: Module Replacement and Dispatch as an Abstraction Pattern

**What you will learn:**

- How TT-Symbiote bridges the PyTorch/hardware gap using module replacement and `__torch_dispatch__`
- The `TorchTTNNTensor` dual-backing design: a four-state lifecycle that makes hardware execution invisible to PyTorch code
- The full `TTNNModule` lifecycle: a four-phase protocol that decouples weight preparation from execution
- The dispatcher system's pluggable routing pattern for operation-level interception
- Why Symbiote's architecture succeeds at the TTNN level but does not reach CB-level control
- Four design patterns (DP1-DP4) that TensorAdapter should adopt or adapt

---

## The Problem Symbiote Solves

TT-Symbiote targets a specific version of the tensor-to-hardware impedance mismatch: running unmodified PyTorch models on Tenstorrent hardware via TTNN. A developer loads a HuggingFace model, provides a class-to-class replacement mapping, and Symbiote handles the rest -- module substitution, weight conversion, device placement, and operation routing.

```python
# EXISTING -- TT-Symbiote usage pattern (3 lines to accelerate a model)
nn_to_ttnn = {nn.Linear: TTNNLinearLLama, nn.SiLU: TTNNSilu}
modules = register_module_replacement_dict(model, nn_to_ttnn)
set_device(model, mesh_device)
for m in modules.values():
    m.preprocess_weights()
    m.move_weights_to_device()
output = model.generate(**inputs)
```

The developer writes zero TTNN code beyond defining the replacement classes. The model's control flow, HuggingFace integration, and Python-level logic remain untouched.

---

## Design Pattern 1 (DP1): Transparent Module Replacement

### EXISTING

Symbiote's module replacement engine (`register_module_replacement_dict()`) walks the model's `nn.Module` tree recursively and substitutes matching modules in-place. The pattern has four key properties:

**Property A -- Class-level matching.** The replacement dictionary maps `nn.Module` subclasses (not instances) to `TTNNModule` subclasses. Matching uses `__class__` identity -- `nn.Linear` matches but `MyLinear(nn.Linear)` does not unless explicitly listed.

**Property B -- Name preservation.** Each replacement inherits the original module's hierarchical name (`model.layers[0].self_attn.q_proj`), propagated via `override_children_module_names()`. This preserves compatibility with profiling, logging, and cache keys.

**Property C -- In-place substitution.** The replacement modifies `model._modules[name]` directly. After the call, the model tree contains `TTNNModule` instances where matched PyTorch modules once were. The original module is retained in `_fallback_torch_layer` for fallback execution.

**Property D -- Exclusion-aware.** The optional `exclude_replacement` set (by module name string) lets the developer keep specific modules on CPU -- for example, keeping `lm_head` on host for post-processing.

The replacement algorithm:

1. **Builds a name map** from `model.named_modules()`, producing `{module_object: "layer.0.self_attn.q_proj", ...}`.
2. **Delegates** to `register_module_replacement_dict_with_module_names()`, which performs the recursive walk over `model._modules`.
3. **For each child module** whose `__class__` appears in the mapping (exact class identity, no subclass matching), calls `initialize_module()`:
   - Checks the exclusion set by module name string
   - Calls `new_class.from_torch(old_module)` to create the replacement
   - Assigns `_unique_name` from the name map
   - Propagates hierarchical names to children via `override_children_module_names()`
4. **Handles non-standard containers**: Beyond `model._modules`, inspects public attributes that are `dict`, `list`, or `tuple` containers for matching `nn.Module` instances.

> **Warning:** The attribute scan via `dir(model)` on TTNNModule objects skips names starting with `_`, meaning modules stored in private attributes (`self._hidden_linear`) are silently skipped for replacement.

### Lesson for TensorAdapter

Module replacement is a **coarse-grained interception point**. Symbiote intercepts at the `nn.Module` boundary -- whole layers are swapped. TensorAdapter needs a finer-grained version of the same idea: intercept at the **tensor operation boundary** (matmul, add, softmax) and inject tile/CB/format decisions at that level. The lesson is not "replace modules" but "transparent interception with preserved identity." Module replacement also provides **structural context** -- it knows that `q_proj` is part of `self_attn` in `layers[0]` -- which is unavailable from per-op dispatch alone.

---

## Design Pattern 2 (DP2): Dual-Backing Tensor Wrapper

### EXISTING

`TorchTTNNTensor` is a `torch.Tensor` subclass (via `_make_wrapper_subclass`) that maintains two backing stores simultaneously:

| Attribute      | Type            | When populated                        |
|----------------|-----------------|---------------------------------------|
| `.elem`        | `torch.Tensor`  | On construction from PyTorch tensor   |
| `.ttnn_tensor` | `ttnn.Tensor`   | After `.to_ttnn` conversion           |

The tensor transitions through four states:

| State | `.elem` | `.ttnn_tensor` | When |
|-------|---------|----------------|------|
| PyTorch-only | `torch.Tensor(...)` | `None` | Just created from a PyTorch tensor |
| Dual-backed | `torch.Tensor(...)` | `ttnn.Tensor(...)` | After `.to_ttnn` conversion |
| TTNN-only | `None` | `ttnn.Tensor(...)` | Result from TTNN op (`.elem` freed) |
| Converted back | `torch.Tensor(...)` | `ttnn.Tensor(...)` | After `.to_torch` readback |

Construction delegates to the active run mode's `new_instance()` method. When wrapping a `ttnn.Tensor`, `.elem` is set to `None` (memory optimization) and a meta tensor captures shape/dtype for PyTorch's bookkeeping. Double-wrapping is prevented by assertion.

Lazy conversion properties (`.to_ttnn` and `.to_torch`) implement on-demand bidirectional conversion:

```python
# EXISTING -- .to_ttnn converts PyTorch -> TTNN on first access
@property
def to_ttnn(self):
    if self.ttnn_tensor is not None:
        return self.ttnn_tensor
    self.ttnn_tensor = ttnn.from_torch(
        self.elem.cpu(), dtype=torch_dtype_to_ttnn_dtype(self.elem.dtype),
        mesh_mapper=self.ttnn_distributed_tensor_config.mesh_mapper if ... else None,
    )
    return self.ttnn_tensor
```

Key design choices:

- **Lazy conversion**: `.to_ttnn` and `.to_torch` convert on first access, avoiding unnecessary host-device transfers.
- **Shape transparency**: The `.shape` property applies `logical_shape_fn` from the `DistributedTensorConfig`, so code checking shapes sees the unsplit logical shape even when the tensor is physically sharded across devices.
- **Double-wrap prevention**: Wrapping a `TorchTTNNTensor` inside another raises an assertion.

### The `_bypass_tensor_wrapping` Optimization

When a `TTNNModule` is called by another `TTNNModule` (parent-child), inputs are already `ttnn.Tensor` on the correct device. Symbiote sets `_bypass_tensor_wrapping = True` on child modules, replacing the three-stage transform pipeline (`compose_transforms` of wrap -> convert -> place) with `fast_unwrap_to_device`:

```
Standard path:     wrap -> to_ttnn -> set_device -> TorchTTNNTensor on device
Bypass path:       fast_unwrap_to_device -> raw ttnn.Tensor on device
```

This eliminates wrapper allocation overhead for inner modules -- a performance-critical optimization for deep model trees.

### Lesson for TensorAdapter

The dual-backing pattern demonstrates that **the abstraction boundary can carry metadata from both worlds simultaneously**. A TensorAdapter `ShapeDescriptor` would serve an analogous role: carrying PyTorch shape information (logical dimensions, dtype) alongside Blaze-specific metadata (tile shape, padded dimensions, CB sizing hints). The bypass optimization teaches that abstraction layers must have a fast path for the common case where both producer and consumer live on the same side of the boundary.

---

## Design Pattern 3 (DP3): Operation-Level Dispatch via `__torch_dispatch__`

### EXISTING

`TorchTTNNTensor.__torch_dispatch__` intercepts every `aten` operation on tensor subclasses. The routing decision in `NormalRun.torch_dispatch()` follows a two-branch pattern:

```
aten op on TorchTTNNTensor
    |
    v
can_dispatch_to_ttnn(func_name, args, kwargs)?
    |                    |
   YES                  NO
    |                    |
    v                    v
dispatch_to_ttnn()    dispatch_to_torch()
    |                    |
    v                    v
TTNN handler          PyTorch fallback (with no_dispatch() guard)
    |                    |
    v                    v
TorchTTNNTensor       TorchTTNNTensor
```

The `can_dispatch_to_ttnn()` function checks three criteria:

1. **Handler exists**: The `aten` op name must appear in the handler registry (~70 mapped operations).
2. **TTNN tensor present**: At least one argument must be a `TorchTTNNTensor` with an allocated `.ttnn_tensor` on a device.
3. **Operation-specific validation**: Some ops have shape, dtype, or rank constraints (e.g., `aten::constant_pad_nd` only supports rank-4).

### The Pluggable Dispatcher System

Dispatchers are selected via the `TT_SYMBIOTE_DISPATCHER` environment variable. Four built-in dispatchers exist:

| Dispatcher   | Behavior                                                |
|-------------|--------------------------------------------------------|
| `DEFAULT`   | Full TTNN dispatch with all ~70 operation handlers      |
| `CPU`       | Always returns False -- all ops run on CPU              |
| `DEBUG`     | Wraps DEFAULT with verbose logging of every decision    |
| `TENSOR_OPS`| Specialized dispatcher for tensor operations            |

> **Warning:** When no `TT_SYMBIOTE_DISPATCHER` is set, the default dispatcher is `"CPU"`, not `"DEFAULT"`. This catches developers who expect TTNN dispatch without explicitly opting in.

### Lesson for TensorAdapter

The `__torch_dispatch__` mechanism is a proven entry point for transparent operation interception. TensorAdapter should **adapt** this pattern -- not reject it -- as a secondary entry point alongside its primary graph/compilation-time path. For eager-mode debugging and incremental adoption, a `__torch_dispatch__`-based integration allows developers to test TensorAdapter's shape analysis on individual operations before committing to full graph compilation. The dispatch pattern maps to a **can_adapt / adapt / fallback** three-phase approach:

1. `can_adapt(op, tensor)` -- Can this tensor's shape/dtype be automatically decomposed into tiles and CBs for this op?
2. `adapt(tensor, op_hint)` -- Compute tile shape, padding, format, CB sizing, and return a `ShapeDescriptor`.
3. Fallback: if `can_adapt` returns False, surface the decision to the developer with diagnostic information.

> **Note:** The interface sketches in this chapter illustrate how surveyed principles might manifest in TensorAdapter. The actual design is defined in [Chapter 3: TensorAdapter Architecture](../ch03_tensor_adapter/index.md).

---

## Design Pattern 4 (DP4): Four-Phase Module Lifecycle

### EXISTING

Every `TTNNModule` follows a strict four-phase lifecycle:

```
from_torch(old_module)     -- Construction: extract weights from PyTorch module
    |
    v
preprocess_weights()       -- CPU-side transforms: tiling, dtype conversion, transpose
    |
    v
move_weights_to_device()   -- Host-to-device transfer of preprocessed weights
    |
    v
forward(*args)             -- Execution on hardware
```

Each phase is guarded by a boolean flag (`_preprocessed_weight`, `_weights_on_device`) that enforces ordering and prevents redundant execution. `TTNNModule` is deliberately **not** a subclass of `torch.nn.Module` -- it stands alone to avoid interference with PyTorch's autograd, parameter registration, and serialization.

The `__call__()` method does not call `forward()` directly -- it delegates to the current run mode's `module_run()`, which handles the full input transformation pipeline via `compose_transforms`:

```python
# EXISTING -- Standard input pipeline (run_config.py)
transform = compose_transforms(
    wrap_to_torch_ttnn_tensor,     # Ensure TorchTTNNTensor wrapper
    to_ttnn_wrap,                  # Extract raw ttnn.Tensor via .to_ttnn
    set_device_wrap(self.device),  # Move to target device
)
```

> **Warning:** The `_preprocessed_weight` flag is set to `True` before calling `preprocess_weights_impl()`. If the implementation raises an exception, the flag stays set and subsequent calls silently skip preprocessing. A failed preprocessing cannot be retried without manually resetting the flag.

### Lesson for TensorAdapter

TensorAdapter should adopt an analogous phased lifecycle for tensor adaptation:

```python
# PROPOSED -- TensorAdapter lifecycle phases (illustrative, not prescriptive)
analyze(tensor, op_hint)    -- Phase 1: compute ShapeDescriptor (tile, padding, format)
validate(descriptors, l1)   -- Phase 2: check L1 budget, format compatibility
materialize(descriptor)     -- Phase 3: create CBHandle with computed parameters
```

The phase guards prevent common errors: you cannot materialize a CB without validation, and you cannot skip the analysis phase. This is the same "idempotent, ordered, guarded" pattern that makes Symbiote's lifecycle robust.

---

## What Symbiote Does Not Solve

Symbiote operates at the **TTNN level** -- it maps PyTorch modules to TTNN operations. It does not descend to the CB level. The gap between TTNN and raw TT-Metal/TT-Blaze includes:

| Symbiote handles          | Symbiote does not handle                |
|--------------------------|-----------------------------------------|
| Module replacement       | Tile shape selection                     |
| Tensor format conversion | CB allocation (num_pages, page_size)     |
| Device placement         | L1 memory budgeting                      |
| Operation routing        | Padding strategy per op                  |
| Multi-device sharding    | CT arg derivation from shapes            |
| Trace capture/replay     | Multi-phase CB reconfiguration           |

The 12 manual decisions per op identified in [Chapter 1: The Manual Plumbing Burden](../ch01_the_gap/03_the_manual_plumbing_burden.md) -- tile shape, padding, format, num_pages, page_size, core placement, CB ID, semaphores, CT args, broadcasting, blocking, and L1 budget -- remain entirely manual when working at the Blaze level. Symbiote's abstractions stop one layer above where TensorAdapter must operate.

Specific limitations relevant to CB management:

1. **No kernel fusion.** Each TTNN op manages its own CBs independently. For single-token decode where each matmul completes in microseconds, dispatch overhead can exceed compute time.
2. **No CB visibility.** The CB lifecycle is entirely internal to each `ttnn.*` call. There is no mechanism to share CBs between operations or apply interval coloring for CB compaction.
3. **No L1 memory control.** Weight layout, activation tiling, and scratch buffer sizing are all delegated to TTNN's internal allocator.
4. **No per-RISC configuration.** There is no mechanism to set compile-time arguments per RISC processor or manage semaphore protocols between operations.
5. **The 64-CB limit is invisible but binding.** Complex fused kernels can approach this limit. Blaze addresses it with interval coloring and multi-phase reconfiguration, but Symbiote never encounters the constraint because each op independently allocates and frees its CBs.

---

## Key Takeaways

- **DP1: Module replacement** provides transparent interception at the `nn.Module` boundary. TensorAdapter needs the same pattern at the tensor-operation boundary, injecting tile/CB decisions without modifying the op's kernel code.
- **DP2: Dual-backing tensor wrapper** demonstrates that abstraction objects can carry metadata from both worlds (PyTorch and hardware). TensorAdapter's `ShapeDescriptor` should carry logical shape alongside tile shape, padding, and format.
- **DP3: `__torch_dispatch__`** provides a proven routing mechanism. TensorAdapter should adapt this as a secondary entry point for eager-mode integration, even though its primary path is graph/compilation-time analysis.
- **DP4: Four-phase lifecycle** with boolean guards prevents ordering violations and redundant work. TensorAdapter should adopt analyze/validate/materialize phases with the same guard pattern.
- **Symbiote does not reach CB level** -- it solves TTNN-level abstraction, leaving the 12-decision-per-op burden intact for Blaze developers.

## Source Files

- `tt_symbiote/core/tensor.py` -- `TorchTTNNTensor`, `__torch_dispatch__`, dual-backing store
- `tt_symbiote/core/module.py` -- `TTNNModule` base class, lifecycle methods, `DeviceArch`
- `tt_symbiote/utils/module_replacement.py` -- `register_module_replacement_dict()`, recursive walk, `initialize_module()`
- `tt_symbiote/core/run_config.py` -- `NormalRun`, `compose_transforms`, `fast_unwrap_to_device`, `DispatchManager`
- `tt_symbiote/core/dispatcher_config.py` -- Pluggable dispatcher registry
- `tt_symbiote/core/default_dispatcher.py` -- `can_dispatch_to_ttnn()`, handler registry

---

**Next:** [02 -- TVM Relay/TIR Scheduling](./02_tvm_relay_tir_scheduling.md)
