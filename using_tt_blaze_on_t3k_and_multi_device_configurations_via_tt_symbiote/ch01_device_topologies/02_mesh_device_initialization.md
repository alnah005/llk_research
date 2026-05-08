# 02 -- Mesh Device Initialization

This file explains how a multi-chip mesh device is created and configured in the TT-Metal and TT-Blaze test frameworks. It covers the `MESH_DEVICE` environment variable that selects the mesh shape at test time, the pytest fixtures that translate that selection into a live `ttnn.MeshDevice`, the device parameter system, the key TTNN mesh APIs, the `run_on_devices` decorator for gating module execution by topology, and how TT-Blaze bootstraps TT-Metal's conftest infrastructure. Prerequisites: familiarity with the physical topologies from [`01_physical_topologies.md`](./01_physical_topologies.md) and basic pytest fixture mechanics.

## The MESH_DEVICE Environment Variable

`MESH_DEVICE` selects the target topology for a test. Valid values are the string keys in the `MeshShapeToDeviceArch` dictionary (see [`01_physical_topologies.md`](./01_physical_topologies.md)).

The value flows through two paths:

1. **Test parametrization:** The `mesh_device` pytest fixture reads `os.environ.get("MESH_DEVICE")` to look up the mesh shape (a `(rows, cols)` tuple) from a per-test dictionary.
2. **Runtime gating:** The `run_on_devices` decorator reads `os.environ.get("MESH_DEVICE")` to resolve the current `DeviceArch` and check it against the allowed set.

Setting the variable before test invocation:

```bash
# Run on a T3K system (8 Wormhole chips, 1x8 mesh)
MESH_DEVICE=T3K pytest tests/test_glm_4_7.py -v

# Run on a TG Galaxy (32 Wormhole chips, 8x4 mesh)
MESH_DEVICE=TG pytest tests/test_glm_4_7.py -v

# Run on a Blackhole quad (4 chips, 1x4 mesh)
MESH_DEVICE=P150x4 pytest tests/test_my_model.py
```

## Test Parametrization Across Topologies

Both TT-Symbiote test files ([test_glm_4_7.py:test_glm] and [test_qwen3_coder_next.py:test_qwen3_coder_next]) use the same pattern to make a single test function runnable across every supported topology. The `mesh_device` fixture parameter is a dictionary keyed by `MESH_DEVICE` values that maps to `(rows, cols)` tuples:

```python
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_glm(mesh_device):
    ...
```

How this works step by step:

1. The dictionary maps each `MESH_DEVICE` string to a `(rows, cols)` tuple.
2. `.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))` reads the environment variable and falls back to the total device count if `MESH_DEVICE` is not set.
3. `indirect=True` means the resolved value is passed to the `mesh_device` **fixture** (not directly to the test function) as `request.param`.
4. The fixture receives the tuple or integer and creates the actual `ttnn.MeshDevice`.

> **Warning:** If `MESH_DEVICE` is not set and the fallback `len(ttnn.get_device_ids())` does not match any expected shape, the fixture will create a `(1, N)` mesh for integer `N`. This may not be the intended topology for multi-dimensional configurations like TG.

## The `device_params` Fixture

The `device_params` fixture passes hardware configuration parameters to the device creation call. It is defined in the root `conftest.py` of tt-metal:

```python
# [tt-metal/conftest.py:device_params]
@pytest.fixture(scope="function")
def device_params(request):
    return getattr(request, "param", {})
```

Tests inject parameters via `@pytest.mark.parametrize("device_params", [...], indirect=True)`. Common parameters include:

| Parameter               | Type                              | Purpose                                                  |
|-------------------------|-----------------------------------|----------------------------------------------------------|
| `trace_region_size`     | `int` (e.g., `50000000`)         | Size in bytes of the trace region for operation capture   |
| `num_command_queues`    | `int` (e.g., `1`)                | Number of hardware command queues per device              |
| `fabric_config`         | `ttnn.FabricConfig`               | Fabric topology: `FABRIC_1D_RING`, `FABRIC_2D`, etc.     |
| `fabric_router_config`  | `ttnn.FabricRouterConfig`         | Router-level tuning (experimental)                       |
| `fabric_tensix_config`  | `ttnn.FabricTensixConfig`         | Tensix extensions for fabric router (mux kernels)        |
| `reliability_mode`      | `ttnn.FabricReliabilityMode`      | Reliability guarantees for fabric initialization         |
| `fabric_manager`        | `ttnn.FabricManagerMode`          | Fabric manager operating mode                            |

### Example: GLM-4 test (no fabric)

```python
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
```

### Example: Qwen3-Coder-Next test (with fabric)

```python
# [test_qwen3_coder_next.py]
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1,
      "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
```

## The `mesh_device` Fixture

The `mesh_device` fixture is defined in tt-metal's root `conftest.py` at function scope. It handles the full lifecycle of a `ttnn.MeshDevice`:

```python
# [tt-metal/conftest.py:mesh_device] (simplified)
@pytest.fixture(scope="function")
def mesh_device(request, silicon_arch_name, device_params):
    param = request.param  # The (rows, cols) tuple or integer

    if isinstance(param, tuple):
        grid_dims = param
        num_devices_requested = grid_dims[0] * grid_dims[1]
        if not ttnn.using_distributed_env() and num_devices_requested > ttnn.get_num_devices():
            pytest.skip("Requested more devices than available.")
        mesh_shape = ttnn.MeshShape(*grid_dims)
    else:
        if not ttnn.using_distributed_env() and param > ttnn.get_num_devices():
            pytest.skip("Requested more devices than available.")
        mesh_shape = ttnn.MeshShape(1, param)

    # Extract fabric settings from device_params before passing to open_mesh_device
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)
    # ... other fabric params popped similarly ...

    set_fabric(fabric_config, ...)  # Configure fabric BEFORE opening mesh
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)

    yield mesh_device

    # Teardown: close submeshes, then close the parent mesh
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    reset_fabric(fabric_config)
```

Key behaviors:

- The fixture is **function-scoped** -- each test function gets a fresh mesh device.
- If the requested number of devices exceeds what is physically available, the test is **skipped** via `pytest.skip()` -- not failed. When using `ttnn.using_distributed_env()` (MPI-based multi-process), the skip is bypassed because device count checks happen differently.
- Fabric configuration (`set_fabric`) happens **before** `ttnn.open_mesh_device()`.
- Teardown closes all submeshes before closing the parent mesh, then resets fabric to `DISABLED`.

## The `run_on_devices` Decorator

The `run_on_devices` decorator in [module.py:run_on_devices] gates a module's `forward()` method to only execute on specific device architectures. It reads `MESH_DEVICE` at call time (not at decoration time):

```python
@run_on_devices(DeviceArch.N300, DeviceArch.T3K)
def forward(self, input_tensor):
    return ttnn.linear(input_tensor, self.tt_weight)
```

### Implementation

```python
# [core/module.py:run_on_devices]
def run_on_devices(*allowed_archs: DeviceArch):
    allowed_set = frozenset(allowed_archs)

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            mesh_device = MeshShapeToDeviceArch.get(os.environ.get("MESH_DEVICE"))
            if mesh_device not in allowed_set:
                raise RuntimeError(
                    f"{self.__class__.__name__}: Device architecture {mesh_device} "
                    f"not supported. Allowed: {allowed_set}"
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
```

### How It Works

1. Reads `os.environ.get("MESH_DEVICE")` to get the configuration string.
2. Looks it up in `MeshShapeToDeviceArch` to get the `DeviceArch` enum value.
3. Checks that the resolved architecture is in the `allowed_set` (a `frozenset` of the decorator arguments).
4. Raises `RuntimeError` if the check fails or if `MESH_DEVICE` is unset/unrecognized.

This mechanism enables a single model class to have topology-specific `forward()` implementations:

```python
class MyAttention(TTNNModule):
    @run_on_devices(DeviceArch.T3K, DeviceArch.P150x8)
    def forward_8chip(self, x):
        # Optimized path for 8-chip ring topology
        ...

    @run_on_devices(DeviceArch.TG, DeviceArch.BHGLX)
    def forward_galaxy(self, x):
        # Optimized path for 32-chip 2D mesh
        ...
```

> **Warning:** `run_on_devices` reads the `MESH_DEVICE` environment variable, not the actual hardware configuration. If `MESH_DEVICE` is unset, it raises `RuntimeError` -- it does not fall back to auto-detection. Always set `MESH_DEVICE` before running tests that use decorated modules.

## Key TTNN Mesh APIs

### ttnn.MeshDevice

The central object representing a multi-chip device mesh. Created by `ttnn.open_mesh_device()`. Key methods:

| API                                        | Purpose                                                    |
|--------------------------------------------|------------------------------------------------------------|
| `ttnn.open_mesh_device(mesh_shape, ...)`   | Open a mesh device with the given shape and params         |
| `ttnn.close_mesh_device(mesh)`             | Close a mesh device and release resources                  |
| `mesh_device.get_num_devices()`            | Return total chip count in the mesh                        |
| `mesh_device.shape`                        | Access `[rows, cols]` of the mesh                          |
| `mesh_device.create_submeshes(shape)`      | Partition the mesh into uniform submeshes                  |
| `mesh_device.get_submeshes()`              | Return previously created submeshes                        |
| `mesh_device.get_fabric_node_id(coord)`    | Get the fabric node ID for a chip at a coordinate          |
| `mesh_device.compute_with_storage_grid_size()` | Get the per-chip compute core grid size               |
| `mesh_device.dram_grid_size()`             | Get DRAM grid dimensions (used to distinguish P100/P150)   |

### ttnn.MeshShape

Describes the logical grid dimensions of a mesh:

```python
mesh_shape = ttnn.MeshShape(1, 8)    # T3K or P150x8: 1 row, 8 columns
mesh_shape = ttnn.MeshShape(8, 4)    # TG or BHGLX: 8 rows, 4 columns
```

### ttnn.MeshCoordinate

Identifies a specific chip within a mesh by its `(row, col)` position:

```python
coord = ttnn.MeshCoordinate(0, 0)   # Top-left chip
coord = ttnn.MeshCoordinate(3, 1)   # Row 3, Column 1
```

Used by `get_fabric_node_id(coord)` to retrieve the physical fabric address of a chip within a submesh. The returned `FabricNodeId` has `.chip_id` and `.mesh_id` attributes providing the global identity of each chip within the fabric network.

### create_submeshes()

Partitions the parent mesh into uniform sub-meshes for pipeline parallelism. Returns a list of submesh `MeshDevice` objects, each owning `submesh_shape` chips. The `SubmeshPartition` class [submesh_partition.py:SubmeshPartition] wraps this API and adds local-rank detection. See [Chapter 5, File 2](../ch05_blaze_pipeline_system/02_submesh_partition_and_topology.md) for the full partitioning explanation, ASCII diagrams, and `chip_to_coord_map()` details.

## How TT-Blaze Loads TT-Metal's Conftest

TT-Blaze lives in a separate repository (`tt-blaze/`) from TT-Metal (`tt-metal/`), but its tests need access to the `mesh_device`, `device_params`, and other fixtures defined in TT-Metal's root `conftest.py`. Rather than duplicating those fixtures, TT-Blaze's root `conftest.py` dynamically loads TT-Metal's conftest as a pytest plugin:

```python
# [tt-blaze/conftest.py]
_tt_metal_dir = str(Path(__file__).parent / "tt-metal")
if _tt_metal_dir not in sys.path:
    sys.path.insert(0, _tt_metal_dir)

_path = Path(__file__).parent / "tt-metal" / "conftest.py"
_spec = importlib.util.spec_from_file_location("tt_metal_conftest", str(_path))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tt_metal_conftest"] = _mod
_spec.loader.exec_module(_mod)

pytest_plugins = ["tt_metal_conftest"]
```

This achieves three things:

1. **Adds `tt-metal/` to `sys.path`** so that all TT-Metal imports (including `ttnn`) resolve correctly.
2. **Loads and executes `tt-metal/conftest.py`** as a module, making its `pytest_addoption`, `pytest_generate_tests`, and all fixtures available.
3. **Registers it as a pytest plugin** via `pytest_plugins`, so pytest's fixture resolution sees all TT-Metal fixtures as if they were defined locally.

This means any TT-Blaze test can use `mesh_device`, `device_params`, `device`, `silicon_arch_name`, and all other TT-Metal fixtures transparently. The `MESH_DEVICE` environment variable, the device parametrization dictionaries, and the fixture lifecycle all work identically in both codebases. Any bug fix or feature addition to tt-metal's conftest propagates automatically to TT-Blaze.

## Complete Initialization Flow

The full sequence from environment variable to live mesh device:

```
MESH_DEVICE=T3K pytest test_model.py
        |
        v
  @pytest.mark.parametrize("mesh_device", [...], indirect=True)
  dict.get("T3K") -> (1, 8)
        |
        v
  mesh_device fixture receives param=(1, 8)
        |
        v
  mesh_shape = ttnn.MeshShape(1, 8)
        |
        v
  set_fabric(fabric_config)           <-- from device_params
        |
        v
  ttnn.open_mesh_device(mesh_shape, **device_params)
        |
        v
  MeshDevice object yielded to test
        |
        +---> model uses run_on_devices(DeviceArch.T3K) to gate forward()
        +---> CCL ops use get_num_links(mesh_device) to discover 1 link
        +---> Pipeline builder uses create_submeshes() for stage partitioning
        |
        v
  Teardown: close submeshes, close mesh_device, reset_fabric
```

---

**Next:** [`03_fabric_and_routing.md`](./03_fabric_and_routing.md)
