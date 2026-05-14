# 8.1 Writing Tests for Blaze Micro-Ops

## Test File Organization

Blaze tests are organized by scope and hardware dependency:

```
tests/blaze/
    conftest.py                         # Shared fixtures, _SILICON_PATHS
    test_utils.py                       # comp_pcc, verify_mesh_result, helpers
    torch_golden.py                     # Golden reference comp_pcc implementation
    micro-ops/
        common/
            test_mcast.py               # Per micro-op: FusedProgram + compose path
            test_matmul.py
            test_rmsnorm.py
            test_scatter.py
            test_gather.py
            test_kn_sliced_matmul.py
            test_copy.py
            test_cb_flush.py
            ...
        dsa/                            # DSA-specific micro-op tests
            test_dsa_golden.py
            test_dsa_sparse_flash_decode.py
            ...
    fused_ops/
        mlp/
            test_dense_mlp.py           # Fused op tests (compose path)
            test_gated_mlp.py
            ...
        moe/
            test_moe.py
            test_routed_expert.py
            test_shared_expert.py
            ...
        mla/
            test_pre_sdpa.py
            test_post_sdpa.py
            ...
    infra/
        test_cb_engine.py               # Unit tests (no device needed)
        test_ct_args.py
        test_compiler.py
        test_kernel_codegen.py
        test_role_engine.py
        test_sem_engine.py
        test_graph_ir.py
        test_visualizer.py
        test_cb_reconfig.py
        test_fused_program_ct_args.py
        test_silicon.py                 # Silicon-required infra tests
        ...
    generality/
        test_rmsnorm_all_models.py      # Cross-model parametrized tests
        test_mlp_all_models.py
        ...
```

New micro-op tests go in `tests/blaze/micro-ops/common/test_my_op.py`. Fused ops that compose multiple micro-ops go in `tests/blaze/fused_ops/<domain>/test_my_fused_op.py`.

## Silicon vs CPU Test Gating

The `conftest.py` at `tests/blaze/conftest.py` uses `_SILICON_PATHS` and `collect_ignore` to skip silicon-dependent tests on CPU-only CI runners:

```python
# From tests/blaze/conftest.py

_SILICON_PATHS = (
    "backed",
    "fused_ops",
    "generality",
    "glm5_1",
    "micro-ops",
    "migration",
    "infra/test_cb_alias_silicon.py",
    "infra/test_cb_compaction_silicon.py",
    "infra/test_cb_overlap_silicon.py",
    "infra/test_cb_reconfig_silicon.py",
    "infra/test_cb_reconfig_dedup_silicon.py",
    "infra/test_cb_reuse_intra_phase_silicon.py",
    "infra/test_mesh.py",
    "infra/test_named_runtime_args.py",
    "infra/test_overlapped_integration.py",
    "infra/test_silicon.py",
)

collect_ignore = list(_SILICON_PATHS)
```

This means all directories under `micro-ops/` and `fused_ops/` require silicon. The `infra/` tests that do not end in `_silicon.py` run on CPU-only runners (they test graph construction, engine assignment, and codegen without dispatching to hardware).

Silicon workflows invoke pytest against explicit file paths, bypassing `collect_ignore`.

**Running infra-only tests on CPU**: To run the CPU-only infrastructure tests without silicon hardware, simply target the `infra/` directory -- pytest's `collect_ignore` automatically skips the silicon-specific files:

```bash
# Run all CPU-only infra tests (no device required)
pytest tests/blaze/infra/ -v

# These test CB engine, CT args, codegen, semaphores, etc.
# without needing any Tenstorrent hardware
```

## Two Test Paths

Every micro-op should be tested through both paths:

### Path 1: FusedProgram Composition (Direct `emit()`)

This path exercises the micro-op's `emit()` method directly. You create a `FusedProgram`, call `emit()` on each op in the pipeline, then call `f.run()`.

**When to use**: Testing a micro-op in isolation or in a small pipeline. Gives direct control over tensor layout, CB allocation, and pipeline ordering.

**Canonical example** -- `test_mcast.py`:

```python
import pytest
import torch
import ttnn

from blaze.fused_program import FusedProgram
from blaze.ops.mcast import Mcast
from blaze.ops.gather import Gather


@pytest.mark.requires_grid_size((11, 10))
@pytest.mark.parametrize("K", [512, 1024], ids=["512", "1024"])
def test_mcast_fused(device, K):
    """Test Mcast via FusedProgram: Mcast -> Gather composition."""
    torch.manual_seed(0)

    from blaze.role_engine import GridConfig
    grid_config = GridConfig.from_device(device)
    a_tile = ttnn.Tile([1, 32])
    num_tiles = K // 32

    matmul_tuples = grid_config.get_matmul_cores()
    num_matmul_cores = len(matmul_tuples)
    matmul_cores_cc = [ttnn.CoreCoord(c, r) for c, r in matmul_tuples]
    matmul_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(c, c) for c in matmul_cores_cc]
    )

    # Input on sender core
    torch_input = torch.randn((1, K), dtype=torch.bfloat16)
    sender_cc = ttnn.CoreCoord(*grid_config.sender_core)
    sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_cc, sender_cc)])
    input_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_grid, (1, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=input_mem, tile=a_tile,
    )

    # Output tensor for gathered results
    total_tiles = num_tiles * num_matmul_cores
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_grid, (total_tiles, 32),
                       ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_output = ttnn.from_torch(
        torch.zeros((total_tiles, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=out_mem, tile=a_tile,
    )

    # Build FusedProgram: Mcast -> Gather
    f = FusedProgram(kernel=None, device=device, name="test_mcast")
    mcast_out = Mcast.emit(f, ttnn_input, prefix="mcast")

    # Scope the mcast output to matmul cores for Gather
    from blaze.fused_program import CBHandle
    matmul_mcast_out = CBHandle(
        cb_id=mcast_out.cb_id,
        num_pages=mcast_out.num_pages,
        page_size=mcast_out.page_size,
        core_ranges=matmul_core_grid,
        data_format=mcast_out.data_format,
        tile_desc=mcast_out.tile_desc,
    )

    Gather.emit(f, matmul_mcast_out, ttnn_output, prefix="gather")

    result = f.run()
    output_torch = ttnn.to_torch(result)

    # Verify each matmul core received the same input
    for i in range(num_matmul_cores):
        start = i * num_tiles
        end = start + num_tiles
        shard = output_torch[start:end, :]
        expected = torch_input.reshape(num_tiles, 32)
        assert torch.equal(shard, expected), (
            f"Mcast shard {i} mismatch at core {matmul_tuples[i]}"
        )
```

Key steps in the FusedProgram test path:
1. Create input/output device tensors with `ttnn.from_torch()` and appropriate shard specs
2. Create `FusedProgram(kernel=None, device=device, name="...")` -- `kernel=None` triggers auto-codegen from the shadow graph
3. Call `OpClass.emit(f, ...)` for each op, chaining `CBHandle` outputs as inputs to downstream ops
4. Call `f.run()` to build, codegen, and execute
5. Read results back with `ttnn.to_torch()` and compare against golden

**Internal mechanics**: When `kernel=None`, `FusedProgram.run()` calls `self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)` to trigger deferred codegen. The codegen path (`blaze/kernel_codegen.py`) reads the shadow graph built by the `emit()` calls and generates a complete C++ kernel. For separate build/run, use `f.build()` to get a `CompiledProgram`, then call `.run()` on it.

### Path 2: Graph Compilation (BlazeCompiler)

This path exercises the full graph API: `blaze.fuse()` context, `ExternalTensor` placeholders, `BlazeCompiler.compile()`, and `program.run()`.

**When to use**: Testing a micro-op through the graph compiler pipeline, which mirrors how fused ops use it in production. Required for ops with `compose()` methods.

**Canonical example** -- `test_mcast.py`, compose path:

```python
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.requires_grid_size((11, 10))
@pytest.mark.parametrize("K", [512])
def test_mcast_compose(mesh_device, K):
    """Test Mcast via blaze.fuse() + BlazeCompiler compose() path."""
    import blaze
    from blaze import ExternalTensor
    from blaze.compiler import BlazeCompiler
    from blaze.role_engine import GridConfig

    torch.manual_seed(0)
    grid_config = GridConfig.from_device(mesh_device)
    a_tile = ttnn.Tile([1, 32])
    num_cores = grid_config.total_cores
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    torch_input = torch.randn((1, K), dtype=torch.bfloat16)

    # Input tensor (mesh)
    sender_cc = ttnn.CoreCoord(*grid_config.sender_core)
    sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_cc, sender_cc)])
    input_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_grid, (1, K),
                       ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=input_mem, tile=a_tile,
        mesh_mapper=mesh_mapper,
    )

    # Output tensor (mesh)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(grid_config.grid_cols - 1,
                       grid_config.grid_rows - 1),
    )])
    output_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(all_cores, (1, K),
                       ttnn.ShardOrientation.ROW_MAJOR),
    )
    ttnn_output = ttnn.from_torch(
        torch.zeros((num_cores, K), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh_device, memory_config=output_mem, tile=a_tile,
        mesh_mapper=mesh_mapper,
    )

    # Phase 1: Build graph
    with blaze.fuse() as ctx:
        blaze.mcast(ExternalTensor("src"))

    # Phase 2: Compile
    compiler = BlazeCompiler(mesh_device)
    program = compiler.compile(
        graph=ctx.graph,
        tensors={"src": ttnn_input},
        output_tensor=ttnn_output,
    )

    # Phase 3: Execute
    result = program.run()

    output_torch = ttnn.to_torch(result)
    # Verify receiver rows match input
    num_receivers = num_cores - 1
    receiver_rows = output_torch[:num_receivers, :]
    torch_expected = torch_input.expand(num_receivers, K)
    assert torch.equal(receiver_rows, torch_expected)
```

Key steps in the graph compilation path:
1. Build mesh tensors with `ttnn.from_torch(..., device=mesh_device, mesh_mapper=...)`
2. Build graph: `with blaze.fuse() as ctx: blaze.my_op(ExternalTensor("name"))`
3. Compile: `compiler = BlazeCompiler(mesh_device)`, `program = compiler.compile(graph=ctx.graph, tensors={...}, output_tensor=...)`
4. Execute: `result = program.run()`
5. Verify with `comp_pcc()` or exact comparison

### Comparing the Two Paths

| Aspect | FusedProgram (emit) | BlazeCompiler (graph) |
|--------|--------------------|-----------------------|
| Entry point | `Op.emit(f, ...)` | `blaze.op_name(ExternalTensor(...))` |
| CB assignment | Manual via `cb_from_tensor`, `cb_scratch` | Automatic via `CBEngine.assign()` |
| CT arg wiring | Manual via `f.ncrisc_ct_args(...)` | Automatic via `CTArgEngine.generate_tuples()` |
| Device | Single device (`device`) | Mesh device (`mesh_device`) |
| Compose fn | Not called | `Op.compose(f, tensors, output, user_args)` |
| Kernel source | Auto-codegen from shadow graph | Auto-codegen or explicit `kernel_source=` |
| Use case | Testing emit() logic directly | Testing compose() + full pipeline |

## Graph-Only Unit Tests (No Device)

For ops with a `compose()` method (fused ops), write graph-level unit tests that validate topology without any hardware. These tests run on CPU-only CI.

**Example from `test_dense_mlp.py`**:

```python
import blaze
from blaze import ExternalTensor


class TestDenseMLPGraph:
    """Verify the DenseMLP graph topology via blaze.fuse()."""

    @staticmethod
    def _build_graph():
        with blaze.fuse() as ctx:
            blaze.dense_mlp(
                ExternalTensor("act"),
                ExternalTensor("rmsnorm_gamma"),
                ExternalTensor("gate_up_weight"),
                ExternalTensor("down_weight"),
            )
        return ctx.graph

    def test_node_count(self):
        graph = self._build_graph()
        assert len(graph.nodes) == 1

    def test_op_name(self):
        graph = self._build_graph()
        assert graph.nodes[0].spec.name == "dense_mlp"

    def test_external_tensors(self):
        graph = self._build_graph()
        expected = {"act", "rmsnorm_gamma", "gate_up_weight", "down_weight"}
        assert set(graph.input_tensors.keys()) == expected

    def test_input_ports(self):
        graph = self._build_graph()
        node = graph.nodes[0]
        input_names = [p.name for p in node.spec.input_ports]
        assert input_names == ["act", "rmsnorm_gamma",
                               "gate_up_weight", "down_weight"]

    def test_validates(self):
        graph = self._build_graph()
        errors = graph.validate()
        assert errors == [], f"Validation errors: {errors}"
```

Also test that the op is registered and callable:

```python
class TestDenseMLPRegistration:
    def test_op_handle_exists(self):
        assert hasattr(blaze, "dense_mlp")

    def test_op_handle_is_callable(self):
        assert callable(blaze.dense_mlp)

    def test_emit_exists(self):
        assert hasattr(blaze.dense_mlp, "emit")
        assert callable(blaze.dense_mlp.emit)
```

## Engine Unit Tests

The infra tests in `tests/blaze/infra/` test the engine pipeline (CB assignment, CT args, semaphores) in isolation, without dispatching to hardware. These tests are invaluable during development because they run in seconds with no hardware dependency.

**CBEngine tests** (`test_cb_engine.py`):

```python
from blaze import CBEngine, ExternalTensor

class TestSimpleFusion:
    def test_matmul_gather_cb_count(self):
        """matmul(ext, ext) -> gather: 2 ext inputs + 1 intermed
        + 1 ext output = 4 CBs."""
        with blaze.fuse() as ctx:
            result = blaze.matmul(_ext("act"), _ext("weights"),
                                 grid="compute")
            out = blaze.gather(result, grid="sender")

        engine = CBEngine()
        assignments = engine.assign(ctx.graph)
        assert len(assignments) == 4

        # Verify no CB ID collisions
        cb_ids = [a.cb_id for a in assignments.values()]
        assert len(cb_ids) == len(set(cb_ids))
```

**CT arg schema tests** (`test_ct_args.py`):

```python
from blaze.ct_args import get_ct_schema

class TestSchemaRegistry:
    def test_matmul_schema_trisc_args(self):
        schema = get_ct_schema("matmul")
        trisc_args = schema.args_for_risc("trisc")
        names = [a.name for a in trisc_args]
        assert "in0" in names
        assert "in1" in names
        assert "out" in names
        assert "k_num_tiles" in names
        assert "out_w_per_core" in names
        assert len(trisc_args) == 5
```

**Kernel codegen tests** (`test_kernel_codegen.py`):

```python
class TestSimpleGraphs:
    def test_mcast_matmul_gather(self):
        """Generate kernel for simple mcast -> matmul -> gather."""
        graph = build_down_proj_graph()
        kernel = generate_kernel(graph)

        # Basic structure checks
        assert 'ops.hpp"' in kernel
        assert "void kernel_main()" in kernel
        assert "deepseek_compute_kernel_init();" in kernel

        # Should have mcast, matmul, gather op types
        assert "blaze::Mcast::Op" in kernel
        assert "blaze::Matmul::Op" in kernel
        assert "blaze::Gather::Op" in kernel
```

**Additional infra tests**:

| Test File | What It Tests |
|-----------|---------------|
| `test_sem_engine.py` | Semaphore assignment: correct IDs, protocol matching |
| `test_fused_program_ct_args.py` | `f.flag()` API, duplicate CT arg detection |
| `test_role_engine.py` | Grid layouts, core counts, sender core position |
| `test_graph_ir.py` | BlazeGraph node/edge construction |
| `test_cb_compaction.py` | Interval coloring for CB ID reuse |
| `test_cb_reconfig.py` | Multi-phase CB reconfiguration |

## Device Tensor Setup

### Sharding Patterns

The most common sharding patterns used in Blaze tests:

**HEIGHT_SHARDED on sender core** (activations):
```python
sender_cc = ttnn.CoreCoord(*grid_config.sender_core)
sender_grid = ttnn.CoreRangeSet([ttnn.CoreRange(sender_cc, sender_cc)])
mem = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(sender_grid, (1, K),
                   ttnn.ShardOrientation.ROW_MAJOR),
)
ttnn_tensor = ttnn.from_torch(
    torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    device=device, memory_config=mem, tile=a_tile,
)
```

**WIDTH_SHARDED across matmul cores** (weights):
```python
matmul_core_grid = grid_config.build_matmul_core_grid()
b_mem = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(matmul_core_grid, (K, N_per_core),
                   ttnn.ShardOrientation.ROW_MAJOR),
)
ttnn_weights = ttnn.from_torch(
    torch_weights, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    device=device, memory_config=b_mem, tile=b_tile,
)
```

**BLOCK_SHARDED** (2D partitioning):
```python
block_mem = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(core_grid, (shard_h, shard_w),
                   ttnn.ShardOrientation.ROW_MAJOR),
)
```

**Mesh tensors** (multi-device):
```python
mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
ttnn_input = ttnn.from_torch(
    torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    device=mesh_device, memory_config=input_mem, tile=a_tile,
    mesh_mapper=mesh_mapper,
)
```

### Tile Objects

The tile must match what the kernel expects. Common choices:

```python
a_tile = ttnn.Tile([1, 32])    # 1x32 activation tile (decode)
b_tile = ttnn.Tile([32, 32])   # 32x32 weight tile (standard)
```

**Source reference**: `TileInfo.from_tensor()` in `blaze/fused_program.py` extracts tile metadata: `tile.get_tile_size(data_format)` gives page size in bytes.

## Golden Reference and PCC Validation

### Using `comp_pcc()`

The standard approach is to compute a PyTorch golden reference and compare using `comp_pcc()` from `models.common.utility_functions` or from the local `torch_golden` module:

```python
from models.common.utility_functions import comp_pcc

# Compute golden
torch_expected = (torch_a.float() @ torch_b.float()).bfloat16()

# Run on device
result = f.run()
output_torch = ttnn.to_torch(result)

# Compare
passing, pcc_message = comp_pcc(
    torch_expected.reshape(1, -1),
    output_torch.reshape(1, -1),
    0.99,  # PCC threshold
)
assert passing, f"PCC check failed: {pcc_message}"
```

Two import paths exist -- choose based on availability:

```python
# From tt-metal models (when available)
from models.common.utility_functions import comp_pcc

# Pure-PyTorch (no tt-metal models dependency)
from torch_golden import comp_pcc
```

### PCC Thresholds

| Operation | Typical PCC | Notes |
|-----------|-------------|-------|
| Exact copy (mcast, gather, scatter) | 1.0 (exact match) | Use `torch.equal()` instead of PCC |
| Matmul (LoFi) | 0.99 | Lower fidelity allows more error |
| Matmul (HiFi4) | 0.999 | Higher fidelity, tighter threshold |
| RMSNorm (fp32 accum) | 0.999 | fp32 accumulation preserves precision |
| RMSNorm (bf16 accum, wide) | 0.98 | bf16 accumulation loses precision at large widths |

### Torch Golden Functions

Write a separate golden function for your op's math, matching the kernel's algorithm:

```python
def _golden(input_tensor, gamma_tensor, epsilon=1e-6):
    """Same golden as deepseek RMSNormSingleCore.golden."""
    return RMSNormSingleCore.golden(input_tensor, gamma_tensor, epsilon=epsilon)
```

Keep goldens simple: use `float()` for intermediate precision, then convert back to `bfloat16()` to match the kernel's output format.

### Mesh Result Verification

For multi-device tests, use `verify_mesh_result()` from `test_utils.py`:

```python
from tests.blaze.test_utils import verify_mesh_result

verify_mesh_result(result, torch_expected, pcc_threshold=0.99,
                   label="matmul")
```

This splits the mesh tensor into per-device tensors and checks PCC on each:

```python
# From test_utils.py
def verify_mesh_result(result, torch_expected, pcc_threshold, label=""):
    device_tensors = ttnn.get_device_tensors(result)
    for idx, device_tensor in enumerate(device_tensors):
        result_torch = ttnn.to_torch(device_tensor)
        passing, pcc_message = comp_pcc(torch_expected, result_torch,
                                         pcc_threshold)
        assert passing, (
            f"{label} Device {idx} PCC check failed: {pcc_message}"
        )
```

### Binary Hash Comparison

For bit-exact tests, `test_utils.py` provides `tensor_bin_hash()`:

```python
from tests.blaze.test_utils import tensor_bin_hash

hash_a = tensor_bin_hash(output_torch)
hash_b = tensor_bin_hash(expected_torch)
assert hash_a == hash_b, "Bit-exact mismatch"
```

For migration testing (comparing old and new implementations across separate runs), the `assert_binary_match` and `save_reference_hash` / `assert_matches_reference` functions provide richer diagnostics:

```python
from tests.blaze.test_utils import save_reference_hash, assert_matches_reference

# Save from old implementation
save_reference_hash(tensor, op_name="rmsnorm", test_id="w7168_fp32")

# Assert match from new implementation
assert_matches_reference(tensor, op_name="rmsnorm", test_id="w7168_fp32")
```

## Test Fixtures

### `device` Fixture

The `device` fixture provides a single Tenstorrent device. Comes from the pytest-ttnn plugin.

### `mesh_device` Fixture

The `mesh_device` fixture is parametrized indirectly with `(rows, cols)`:

```python
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_my_op(mesh_device):
    compiler = BlazeCompiler(mesh_device)
    ...
```

For multi-device tests:

```python
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_my_multi_device_op(mesh_device):
    ...
```

### `@pytest.mark.requires_grid_size`

Many tests require a minimum grid size (Blackhole has up to 13x10 cores):

```python
@pytest.mark.requires_grid_size((11, 10))
def test_my_op(device, ...):
    ...
```

## Parametrized Tests

### Width and Dimension Parametrization

```python
@pytest.mark.parametrize(
    "K, N_per_core",
    [(512, 64), (256, 32)],
    ids=["512x64", "256x32"],
)
@pytest.mark.parametrize("pcc", [0.99])
def test_matmul_fused(device, K, N_per_core, pcc):
    ...
```

### Data Format Parametrization

```python
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_my_op_formats(device, dtype):
    ...
```

### Tile Shape Parametrization

Different tile shapes should be tested when the op supports them:

```python
@pytest.mark.parametrize("tile_shape", [[1, 32], [32, 32]])
def test_my_op(device, tile_shape):
    tile = ttnn.Tile(tile_shape)
    ...
```

### Cross-Model Parametrization

For ops that must work across model families, use parametrized model lists:

```python
DENSE_MLP_MODELS = [
    "llama3_1_8b", "llama3_1b", "llama3_2_3b",
    "qwen3_32b", "qwen3_8b", "olmo2_7b",
]

class TestDenseMLPModelCoverage:
    @pytest.mark.parametrize("model", DENSE_MLP_MODELS)
    def test_graph_builds(self, model):
        graph = self._build_graph()
        assert len(graph.nodes) == 1
        assert graph.nodes[0].spec.name == "dense_mlp"
```

## Recommended Test Structure for a New Micro-Op

For a new micro-op named `my_op`, create `tests/blaze/micro-ops/common/test_my_op.py` with:

1. **FusedProgram test** (`test_my_op_fused`): Direct `emit()` in a pipeline, verify golden PCC
2. **Compose/graph test** (`test_my_op_compose`): `blaze.fuse()` + `BlazeCompiler.compile()`, verify golden PCC
3. **Edge cases**: Parametrize widths, data formats, tile shapes, number of cores
4. **Graph-only tests** (if fused op): topology validation, port names, registration checks -- these go in `tests/blaze/fused_ops/` or `tests/blaze/infra/`

Note: Some ops cannot be tested standalone via the compose path because they require a prior pipeline stage to fill their input CBs. For example, `Matmul` requires `Mcast` to fill in0 first. In such cases, document this and test the compose path through a fused op that includes the prerequisite stages.

### Complete Test Template

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Blaze MyOp micro-op test -- FusedProgram composition path.

Tests MyOp through its emit() API within a FusedProgram. Uses Mcast
to provide input and Gather to collect output for readback.

PCC target 0.99.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from blaze.fused_program import FusedProgram
from blaze.ops.mcast import Mcast
from blaze.ops.my_op import MyOp
from blaze.ops.gather import Gather


def _golden(input_tensor, ...):
    """Torch reference implementation matching MyOp's kernel math."""
    # Implement the same computation your kernel performs
    return expected_output


@pytest.mark.requires_grid_size((11, 10))
@pytest.mark.parametrize("K", [512, 1024], ids=["512", "1024"])
@pytest.mark.parametrize("pcc", [0.99])
def test_my_op_fused(device, K, pcc):
    """Test MyOp via FusedProgram: Mcast -> MyOp -> Gather composition."""
    torch.manual_seed(0)

    from blaze.role_engine import GridConfig
    grid_config = GridConfig.from_device(device)

    # Create torch golden
    torch_input = torch.randn((1, K), dtype=torch.bfloat16)
    torch_expected = _golden(torch_input)

    # Create device tensors
    # ... (shard specs, memory configs, ttnn.from_torch)

    # Build FusedProgram
    f = FusedProgram(kernel=None, device=device, name="test_my_op")

    mcast_out = Mcast.emit(f, ttnn_input, prefix="act_mcast")
    my_op_out = MyOp.emit(f, mcast_out, prefix="my_op")
    Gather.emit(f, my_op_out, ttnn_output, prefix="gather")

    result = f.run()

    # Verify
    output_torch = ttnn.to_torch(result)
    passing, pcc_message = comp_pcc(
        torch_expected.reshape(1, -1),
        output_torch.reshape(1, -1),
        pcc,
    )
    logger.info(pcc_message)
    assert passing, f"PCC check failed: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.requires_grid_size((11, 10))
@pytest.mark.parametrize("K", [512])
def test_my_op_compose(mesh_device, K):
    """Test MyOp via blaze.fuse() + BlazeCompiler compose() path."""
    import blaze
    from blaze import ExternalTensor
    from blaze.compiler import BlazeCompiler

    torch.manual_seed(0)
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # Create tensors with mesh_mapper
    # ...

    with blaze.fuse() as ctx:
        blaze.my_op(ExternalTensor(name="input"), grid=[(0, 0)])

    compiler = BlazeCompiler(mesh_device)
    program = compiler.compile(
        graph=ctx.graph,
        tensors={"input": ttnn_input},
        output_tensor=ttnn_output,
        user_args={"param1": value1},
    )

    result = program.run()
    output_torch = ttnn.to_torch(result)
    # Verify ...
```

## Test Execution

```bash
# Run a specific micro-op test
pytest tests/blaze/micro-ops/common/test_matmul.py -v -s

# Run all infra tests (no device required)
pytest tests/blaze/infra/ -v

# Run with L1 profiling enabled
BLAZE_L1_PROFILE=1 pytest tests/blaze/micro-ops/common/test_matmul.py -v -s

# Run with debug kernels
BLAZE_DEBUG_KERNELS=1 pytest tests/blaze/micro-ops/common/test_matmul.py -v -s
```
