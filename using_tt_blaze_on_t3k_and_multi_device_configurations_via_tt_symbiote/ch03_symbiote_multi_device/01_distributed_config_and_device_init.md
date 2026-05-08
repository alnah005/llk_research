# Distributed Configuration and Device Initialization

The transition from single-device to multi-device execution in TT-Symbiote is driven by a single condition: `device.get_num_devices() > 1`. When this condition is true, the `DeviceInit` system creates a `DistributedConfig` that bundles three components -- a mesh mapper for distributing tensors, a mesh composer for reassembling them, and a CCL manager for inter-device communication. This file traces the complete initialization path from `set_device()` through to the point where every module in the model tree has its distributed state configured and is ready for multi-device execution. Prerequisites: [Chapter 2, File 1](../ch02_symbiote_core/01_module_replacement_engine.md) (TTNNModule lifecycle, `set_device()` introduction) and [Chapter 1](../ch01_device_topologies/index.md) (mesh shapes, `get_num_devices()`).

## The Multi-Device Activation Gate

The entire multi-device system pivots on a single conditional in `_initialize_module_on_device` [device_management.py:_initialize_module_on_device]:

```python
def _initialize_module_on_device(module: "TTNNModule", device, device_init=DeviceInit):
    module.to_device(device)
    if device.get_num_devices() > 1:
        module.set_device_state(device_init.init_state(device))
```

When `device` is a single-chip `ttnn.Device` (N150, P150), `get_num_devices()` returns 1 and the module runs without distributed state. When `device` is a `ttnn.MeshDevice` with multiple chips (N300, T3K, TG), each module receives a `DistributedConfig` that governs how its inputs and outputs are sharded across the mesh.

This check appears in three critical locations:

1. `DistributedConfig.__post_init__()` -- creates default tensor config and CCL manager
2. `_initialize_module_on_device()` -- sets distributed state on each module
3. Various `move_weights_to_device_impl()` methods -- select mesh mappers for weight distribution

> **Warning:** If you create a `ttnn.MeshDevice` with a single chip (e.g., mesh shape `(1,1)`), `get_num_devices()` still returns 1 and multi-device codepaths are not activated. The gate is on actual device count, not on the type of the device object.

## DistributedConfig Dataclass

`DistributedConfig` is the per-device container for all multi-device state [run_config.py:DistributedConfig]:

```python
@dataclass
class DistributedConfig:
    mesh_device: Any
    tensor_config: Optional[DistributedTensorConfig] = None
    ccl_manager: Optional[Any] = None
```

| Field | Type | Purpose |
|-------|------|---------|
| `mesh_device` | `ttnn.MeshDevice` | The physical mesh this config describes |
| `tensor_config` | `DistributedTensorConfig` | Default sharding strategy for tensors |
| `ccl_manager` | `TT_CCL` | Semaphore and link management for CCL ops |

### Auto-Initialization via __post_init__

The `__post_init__` method auto-populates both optional fields when the mesh has more than one device:

```python
def __post_init__(self):
    if self.tensor_config is None and self.mesh_device.get_num_devices() > 1:
        self.tensor_config = DistributedTensorConfig(
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, self.mesh_device.shape, (0, -1)
            ),
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, self.mesh_device.shape, (0, -1)
            ),
            logical_shape_fn=logical_shape_for_batch_channel_sharding(
                self.mesh_device.shape
            ),
        )
    if self.ccl_manager is None and self.mesh_device.get_num_devices() > 1:
        self.ccl_manager = TT_CCL(self.mesh_device)
```

The default sharding strategy is **batch-channel sharding**: dimension 0 (batch) is sharded along mesh rows and dimension -1 (hidden/channel) is sharded along mesh columns. For T3K with shape `(1, 8)`, this means batch is not split (1 row) and hidden is divided 8 ways across the 8 columns.

> **Warning:** The `TT_CCL` constructor allocates global semaphores on the mesh device. If you create multiple `DistributedConfig` instances for the same mesh device, you will waste device resources on duplicate semaphore allocations. The `DeviceInit` singleton pattern (described below) prevents this.

## DistributedTensorConfig

`DistributedTensorConfig` describes how a single tensor is distributed across the mesh [run_config.py:DistributedTensorConfig]:

```python
@dataclass
class DistributedTensorConfig:
    mesh_mapper: Any        # How to distribute host tensor to devices
    mesh_composer: Any      # How to reassemble device shards to host
    logical_shape_fn: Optional[Any] = None  # Sharded shape -> logical shape
```

**mesh_mapper** is a TTNN strategy object passed to `ttnn.from_torch()`. Common mappers:

- `ShardTensor2dMesh(mesh_device, mesh_shape, (dim0, dim1))` -- shards along two dimensions simultaneously.
- `ShardTensorToMesh(mesh_device, dim=N)` -- shards along a single dimension.
- `ReplicateTensorToMesh(mesh_device)` -- copies the full tensor to every device.

**mesh_composer** is the inverse, used during `ttnn.to_torch()` to reconstruct a single host tensor:

- `ConcatMesh2dToTensor(mesh_device, mesh_shape, (dim0, dim1))` -- concatenates along two dimensions.
- `ConcatMeshToTensor(mesh_device, dim=N)` -- concatenates along one dimension.

**logical_shape_fn** computes the full tensor shape from the per-device shard shape:

```python
def get_logical_shape(self, sharded_shape):
    if self.logical_shape_fn is not None:
        return self.logical_shape_fn(sharded_shape)
    return sharded_shape
```

This is used by `TorchTTNNTensor` to report shapes to PyTorch that match what the original unsharded model expects.

The mapper distributes host tensors to devices and the composer reassembles them. For T3K `(1, 8)` with `ShardTensor2dMesh(dims=(0, -1))`, a `[B, S, H]` tensor becomes `[B, S, H/8]` per device; `ConcatMesh2dToTensor(dims=(0, -1))` reverses this.

## logical_shape_for_batch_channel_sharding

This factory function creates a closure that computes the pre-sharded logical shape from a per-device physical shape [run_config.py:logical_shape_for_batch_channel_sharding]:

```python
def logical_shape_for_batch_channel_sharding(mesh_shape):
    def _logical_shape(shape):
        shape = list(shape)
        logical_shape = [shape[0] * mesh_shape[0]] + shape[1:-1] + \
                        [shape[-1] * mesh_shape[1]]
        return tuple(logical_shape)
    return _logical_shape
```

For T3K `(1, 8)` with a per-device shard of shape `[1, 32, 512]`:
- `logical_shape[0] = 1 * 1 = 1` (batch unchanged, 1 mesh row)
- `logical_shape[-1] = 512 * 8 = 4096` (hidden reconstructed from 8 columns)
- Result: `[1, 32, 4096]`

```
Physical layout on T3K (1, 8):

  Device 0: [1, 32, 512]  \
  Device 1: [1, 32, 512]   |
  Device 2: [1, 32, 512]   |
  ...                       +---> Logical shape: [1, 32, 4096]
  Device 7: [1, 32, 512]  /

  mesh_mapper shards dim=-1 into 8 slices
  mesh_composer concatenates dim=-1 back to full
```

For a TG with shape `(8, 4)` and host tensor `[batch=8, seq=128, hidden=4096]`: batch is split 8 ways across rows, hidden is split 4 ways across columns, giving each device `[1, 128, 1024]`.

## get_tensor_config_for_tensor: Automatic Replication Fallback

Not every tensor's dimensions divide evenly by the mesh shape. The `get_tensor_config_for_tensor` method handles this [run_config.py:DistributedConfig.get_tensor_config_for_tensor]:

```python
def get_tensor_config_for_tensor(self, module_name, tensor):
    if tensor is not None:
        if (
            len(tensor.shape) < 2
            or tensor.shape[-1] % self.mesh_device.shape[-1] != 0
            or tensor.shape[0] % self.mesh_device.shape[0] != 0
        ):
            print(f"Could not determine tensor config for {module_name} ...")
            return DistributedTensorConfig(
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                mesh_composer=ttnn.create_mesh_composer(...),
            )
    return self.tensor_config
```

The fallback conditions are:
1. Tensor rank less than 2 (scalars, 1D tensors)
2. Last dimension not evenly divisible by mesh columns
3. First dimension not evenly divisible by mesh rows

When any condition is true, the tensor is **replicated** to all devices instead of sharded.

> **Warning:** The replication fallback prints a warning but does NOT raise an error. If a weight that should be sharded gets replicated, each device stores the full weight (wasting DRAM) and the matmul produces incorrect results because the input/output dimensions no longer match the sharding assumptions. Always verify that weight tensors divide evenly by the mesh shape.

## DeviceInit: The Singleton State Manager

`DeviceInit` is a singleton that maps each `MeshDevice` to its `DistributedConfig`, ensuring the config (and its `TT_CCL` semaphores) is created exactly once per device. See [Chapter 2, File 1](../ch02_symbiote_core/01_module_replacement_engine.md) for the base `DeviceInit` class, its `init_state()` caching mechanism, and the `DEVICE_TO_STATE_DICT` pattern.

Model-specific test harnesses can subclass `DeviceInit` and override `init_state_impl()` to provide custom `DistributedConfig` instances with non-default tensor configs or CCL configurations: `set_device(model, mesh_device, device_init=MyCustomDeviceInit)`.

## CCLManagerConfig

`CCLManagerConfig` provides an alternative way to configure the CCL layer [run_config.py:CCLManagerConfig]:

```python
@dataclass
class CCLManagerConfig:
    mesh_device: Any
    num_links: Optional[int] = None    # Default: 1
    topology: Optional[Any] = None     # Default: ttnn.Topology.Linear
```

This is not used by the default `DistributedConfig.__post_init__` path (which directly creates a `TT_CCL`), but is available for custom `DeviceInit` subclasses that need finer-grained CCL configuration.

> **Warning:** The default `CCLManagerConfig` uses `ttnn.Topology.Linear` and 1 link. For traced execution on T3K, modules that call CCL directly (not through `TT_CCL`) must use `ttnn.Topology.Ring` to avoid dynamic buffer allocation that breaks trace capture. See [File 3](./03_ccl_operations.md) for details.

## set_device: Recursive Multi-Device Initialization

`set_device()` is the entry point that propagates device and distributed state to every module in a model tree [device_management.py:set_device]. Its responsibilities in the multi-device context:

1. **Module initialization:** For each `TTNNModule`, calls `_initialize_module_on_device()`, which stores the device reference and (if multi-device) attaches the `DistributedConfig`.

2. **Bypass tensor wrapping:** For `TTNNModule` children of `TTNNModule` parents, sets `_bypass_tensor_wrapping = True`, enabling `fast_unwrap_to_device` which avoids the `TorchTTNNTensor` wrapping overhead for inner modules. See [Chapter 2, File 2](../ch02_symbiote_core/02_dispatch_and_tensor_wrapping.md).

3. **Deep recursive walk:** Processes `nn.Module._modules`, attribute dictionaries, lists, and tuples to find all `TTNNModule` instances, including those stored as dict values or list elements (common in MoE architectures where experts may be stored in containers).

4. **Timing instrumentation:** Wraps each module's `forward` (for `nn.Module`) or `call` (for `TTNNModule`) with `timed_call()` for profiling via `DispatchManager.save_stats_to_file()`.

5. **Visualization:** Optionally calls `draw_model_graph(obj)` to produce a module tree visualization.

```
set_device(model, mesh_device)
    |
    +-- Build module_names dict: {module: name} from named_modules()
    |
    +-- _set_device_recursive(model)
            |
            +-- If nn.Module:
            |       Register forward timing hook
            |       For each child in _modules:
            |           If TTNNModule: _initialize_module_on_device()
            |           Recurse into child
            |       For each non-private attribute:
            |           If TTNNModule: initialize + recurse
            |           If dict: check values for TTNNModule
            |           If list/tuple: check elements for TTNNModule
            |
            +-- If TTNNModule:
                    Set _bypass_tensor_wrapping based on parent
                    _initialize_module_on_device()
                    Register call timing hook
                    Recurse into all attributes
```

## get_default_distributed_tensor_config

`get_default_distributed_tensor_config()` is the bridge between tensor creation during dispatch and the distributed state stored in `DeviceInit` [run_config.py:get_default_distributed_tensor_config]:

```python
def get_default_distributed_tensor_config(mesh_device=None, torch_tensor=None,
                                          module_name=None):
    if mesh_device is not None:
        assert mesh_device in DeviceInit.DEVICE_TO_STATE_DICT, \
            f"mesh_device {mesh_device} not found in DEVICE_TO_STATE_DICT"
        state = DeviceInit.DEVICE_TO_STATE_DICT[mesh_device]
    else:
        state = next(iter(DeviceInit.DEVICE_TO_STATE_DICT.values()))
    if torch_tensor is not None:
        return state.get_tensor_config_for_tensor(module_name, torch_tensor)
    return state.tensor_config
```

This function is called during `NormalRun.new_instance()` when wrapping a PyTorch tensor into `TorchTTNNTensor`. It automatically attaches the correct `DistributedTensorConfig` to every tensor created during inference. The config is stored on the tensor wrapper as `ttnn_distributed_tensor_config` and used in two places:
1. **`to_ttnn` property**: passes `mesh_mapper` to `ttnn.from_torch` so the tensor is sharded across devices
2. **`to_torch` property**: passes `mesh_composer` to `ttnn.to_torch` so the tensor is reassembled

Individual modules can override this per-tensor assignment via `set_output_tensors_config_impl()`. For example, `TTNNBailingRotaryEmbedding` overrides to use `ReplicateTensorToMesh` for its cos/sin outputs since these are identical on every device.

## Complete Initialization Sequence

```
1. mesh_device = ttnn.open_mesh_device(mesh_shape=(1, 8))   # T3K: 8 chips

2. model = load_huggingface_model()                          # Pure PyTorch

3. register_module_replacement_dict(model, {...})            # Replace modules

4. set_device(model, mesh_device)                            # <-- THIS FILE
       |
       +-- DeviceInit.init_state(mesh_device)
       |       |
       |       +-- DistributedConfig(mesh_device)
       |               |
       |               +-- DistributedTensorConfig(
       |               |       ShardTensor2dMesh(..., (0,-1)),
       |               |       ConcatMesh2dToTensor(..., (0,-1)),
       |               |       logical_shape_for_batch_channel_sharding(...)
       |               |   )
       |               |
       |               +-- TT_CCL(mesh_device)
       |                       +-- Allocate 2x barrier semaphores (per axis)
       |                       +-- Allocate 2x2 all-gather semaphores
       |                       +-- Allocate 2x3 reduce-scatter semaphores
       |                       (across 3 axis configurations: 0, 1, None)
       |
       +-- For each TTNNModule in model tree:
               _initialize_module_on_device(module, mesh_device)
                   +-- module.to_device(mesh_device)
                   +-- module.set_device_state(distributed_config)

5. During inference, every new TorchTTNNTensor gets:
   get_default_distributed_tensor_config()
     +-- Looks up cached DistributedConfig
     +-- Checks tensor divisibility
     +-- Returns sharding config or replication fallback
```

---

**Next:** [`02_weight_sharding_strategies.md`](./02_weight_sharding_strategies.md)
