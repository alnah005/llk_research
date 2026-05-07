# Appendix B.1 -- Test Infrastructure

This appendix documents the test organization, shared fixtures, utility modules,
and CI pipelines that make up TT-Blaze's quality gate.

## B.1.1 Test Directory Layout

All tests live under `tests/` in two top-level trees:

```
tests/
  blaze/                   # Core Blaze tests
    conftest.py            # Silicon-path collection filter
    conftest_capture.py    # CB blueprint capture plugin
    test_utils.py          # Shared PCC / hash / fabric helpers
    torch_golden.py        # Pure-PyTorch PCC and RoPE references
    sdpa_helpers.py        # SDPA reduce forwarder scratch helpers
    weight_cache.py        # Disk-cached torch tensor generation
    device_weight_cache.py # Disk-cached device-format tensors (tensorbin)
    generate_weight_cache.py  # CLI to pre-warm both cache layers
    test_program_semaphore.py
    infra/                 # Engine unit tests (CB, compiler, graph IR)
    micro-ops/             # Per-MicroOp silicon tests
      common/              # matmul, mcast, rmsnorm, scatter, gather, ...
      mla/                 # flash_mla, sdpa, rope, kv_cache_update, ...
      moe/                 # deepseek_moe_gate, glm_moe_gate, eltwise_mul, ...
      dsa/                 # dsa_indexer, dsa_sparse_flash_decode, ...
      migration/           # KV cache migration micro-ops
      others/              # argmax, embedding
    fused_ops/             # Multi-MicroOp fused-program tests
      mla/                 # pre_sdpa, post_sdpa, mla, q_branch, kv_branch, ...
      mlp/                 # dense_mlp, gated_mlp, dense_swiglu, mlp_mixed
      moe/                 # moe, moe_router, routed_expert, shared_expert, swiglu
      gqa/                 # GQA attention (multi-model)
      glm5/                # GLM-5.1 fused attention ops
      others/              # lm_head_sampling
    backed/                # Full-layer backed tests (MLA, MoE, sparse)
    generality/            # Cross-model generality scoreboard tests
      test_shapes/         # model_specs_table_loader.py, YAML specs
      generate_scoreboard.py
    glm5_1/                # GLM-5.1 model-specific ops
      micro-ops/           # glm5_moe_gate
    migration/             # Multi-host KV cache migration tests
  pipeline_builder/        # Pipeline topology tests
    test_pipeline_builder_infra.py
    test_model_pipelines.py
    test_deepseek_stages.py
```

### Category Descriptions

| Directory | Purpose | Hardware |
|-----------|---------|----------|
| `infra/` | CB engine, reconfig, compaction, graph IR, kernel codegen, compiler, visualizer | CPU + silicon |
| `micro-ops/` | One test file per `BlazeOp` MicroOp; validates a single kernel | Silicon only |
| `fused_ops/` | Tests composed of multiple MicroOps via `blaze.fuse()` | Silicon only |
| `backed/` | Full decoder-layer tests with real weight shapes | Silicon only |
| `generality/` | Parametric tests across model architectures (DeepSeek, Llama, GLM, Qwen, ...) | Silicon only |
| `glm5_1/` | GLM-5.1-specific fused op tests | Silicon only |
| `migration/` | Multi-host KV cache migration layer tests | Multi-device |
| `pipeline_builder/` | Pipeline graph topology discovery and submesh partition | Multi-device |

## B.1.2 Root conftest.py

The repository root `conftest.py` bootstraps `tt-metal`'s test fixtures:

```python
# conftest.py (repo root)
_path = Path(__file__).parent / "tt-metal" / "conftest.py"
_spec = importlib.util.spec_from_file_location("tt_metal_conftest", str(_path))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["tt_metal_conftest"] = _mod
_spec.loader.exec_module(_mod)

pytest_plugins = ["tt_metal_conftest"]
```

This makes `tt-metal` fixtures (`device`, `device_params`, `mesh_device`,
`bh_2d_mesh_device`, etc.) available to every test without duplicating their
definitions. The `device_params` fixture is used extensively via
`@pytest.mark.parametrize("device_params", [...], indirect=True)` to configure
fabric mode, router config, and trace region sizes.

## B.1.3 Silicon Path Collection Filter (tests/blaze/conftest.py)

The inner `conftest.py` at `tests/blaze/conftest.py` prevents silicon-only
tests from being collected on CPU-only CI runners:

```python
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

Silicon CI workflows invoke `pytest` against explicit file paths, which
bypasses `collect_ignore`, so these files still run on Loudbox and P150
hardware.

## B.1.4 CB Blueprint Capture Plugin (conftest_capture.py)

The `conftest_capture.py` module is a pytest plugin that intercepts
`ttnn.generic_op` calls to capture `MeshProgramDescriptor` and IO tensors.
After each test it dumps:

- Per-core CB address/size/page layout
- L1 overlap analysis
- Multi-device layout comparison
- CB-to-tensor mapping table
- JSON blueprints to `BLAZE_CAPTURE_DIR` (default `/tmp/blaze_capture`)

Usage:

```bash
PYTHONPATH=tests/blaze:$PYTHONPATH pytest -p conftest_capture <test> -v -s
```

This plugin supports both Blaze (reads CB addresses from `ProgramDescriptor`)
and Blitz (auto-detects `CbReconfig` tensor and decodes post-reconfig
addresses).

## B.1.5 Test Utility Modules

### test_utils.py

Core shared helpers used across all test categories:

| Function | Purpose |
|----------|---------|
| `verify_mesh_result(result, torch_expected, pcc_threshold)` | PCC check on each device in a mesh result |
| `ttnn_to_torch_sync(ttnn_tensor)` | Synchronize device, then read tensor to host |
| `pad_to_dram_banks(num, tile_w, lcm)` | Round up to nearest multiple for DRAM bank alignment |
| `shuffle_tensor_tiles(tensor, tile_size, num_banks)` | Row-major to column-major tile reorder for DRAM streaming |
| `glm_style_top8_golden(input_t, bias_t)` | GLM-style hierarchical sort reference |
| `tensor_bin_hash(tensor)` | SHA-256 of a tensor's raw binary contents |
| `assert_binary_match(actual, expected)` | Bit-exact comparison with detailed diagnostics |
| `save_reference_hash(tensor, op_name, test_id)` | Persist reference hash for cross-run comparison |
| `assert_matches_reference(tensor, op_name, test_id)` | Verify tensor matches a previously saved hash |

### torch_golden.py

Pure PyTorch functions with no `tt-metal` or `ttnn` dependency:

```python
def comp_pcc(golden, calculated, pcc=0.99) -> tuple[bool, float]:
    """Pearson correlation coefficient check."""
    # Handles NaN, Inf, all-zero tensors gracefully
    # Returns (passes, correlation_value)

def get_rot_transformation_mat() -> torch.Tensor:
    """32x32 RoPE swap matrix used by DeepSeek V3 TT RoPE."""
```

The `comp_pcc` function is the standard correctness metric across all Blaze
tests.

### sdpa_helpers.py

Helpers for SDPA reduce forwarder scratch buffer setup:

```python
def compute_forwarder_scratch_size(batch_size, l_width, num_cores, ...) -> int:
    """Return per-forwarder-core L1 byte requirement for SDPA reduce."""

def make_forwarder_scratch_tensor(device, *, fwd_bytes, forwarder_cores, ...):
    """Create a ROW_MAJOR forwarder scratch tensor with correct per-core size."""
```

## B.1.6 Weight Cache Infrastructure

Tests for large fused ops (MLA, MoE) involve generating hundreds of megabytes
of weight tensors. Two caching layers eliminate redundant computation.

### Phase 1: WeightCache (weight_cache.py)

Caches **torch tensors** on disk at `<repo>/.cache/weights/` (or
`BLAZE_WEIGHT_CACHE_DIR`).

```python
class WeightCache:
    def get_or_create(
        self,
        func_name: str,
        params: dict,
        generator: Callable[[], dict[str, torch.Tensor]],
        source_func: Callable | None = None,
    ) -> tuple[dict[str, torch.Tensor], bool]:
```

Cache keys incorporate a SHA-256 of (function name, parameters, source code
hash) to auto-invalidate when the generator logic changes.

### Phase 2: DeviceWeightCache (device_weight_cache.py)

Caches **device-format tensors** (ttnn.Tensor) in Blitz's
`tensorbin + manifest.json` format at `<repo>/.cache/weights/device/` (or
`BLAZE_DEVICE_CACHE_DIR`).

```python
class DeviceWeightCache:
    def save(self, name, mesh_shape, tensors, source_hash=""): ...
    def load(self, name, mesh_shape, device, source_hash=""): ...
    def prepare_and_load(self, name, mesh_shape, device, source_hash, prepare_fn): ...
```

Supports three tensor types:
- `OverlappedView` -- fusion group members sharing a backing ttnn.Tensor
- `ttnn.Tensor` -- standalone device tensors
- `list[ttnn.Tensor]` -- per-expert tensor lists (e.g., 256 routed experts)

### generate_weight_cache.py

CLI script to pre-warm both cache layers:

```bash
python tests/blaze/generate_weight_cache.py           # Both MLA + MoE
python tests/blaze/generate_weight_cache.py --mla      # MLA only
python tests/blaze/generate_weight_cache.py --moe      # MoE only
python tests/blaze/generate_weight_cache.py --clear    # Remove all caches
python tests/blaze/generate_weight_cache.py \
    --source state_dict:/path/to/DeepSeek-R1-0528      # Real weights
```

Env var control: set `BLAZE_WEIGHT_CACHE=0` to disable both caching layers
entirely.

## B.1.7 Generality Scoreboard

The `tests/blaze/generality/` directory contains parametric tests that run
each fused op across all supported model architectures defined in
`model_specs_table.yaml`. The `generate_scoreboard.py` script regenerates a
markdown table from test results:

```python
# generate_scoreboard.py
TP_VALUES = [1, 2, 4, 8]
MAX_GRID_COLS = 13
MAX_GRID_ROWS = 10
SRAM_PER_CORE = 1_572_864  # 1.5 MB

def _compute_qkv_valid(spec: ModelSpec, tp: int) -> bool:
    """Return True if a valid QKV allocation exists that fits L1 SRAM."""
```

Generality test files include:
`test_gqa_proj_all_models.py`, `test_mlp_all_models.py`,
`test_moe_all_models.py`, `test_moe_router_all_models.py`,
`test_rmsnorm_all_models.py`, `test_routed_expert_all_models.py`,
`test_shared_expert_all_models.py`, `test_swiglu_all_models.py`, and others.

## B.1.8 CI Workflows

CI is defined in `.github/workflows/` with the following structure:

### PR Gate (`pr-gate.yaml`)

Fast, lightweight pipeline for every PR push:

```
download-build-artifact  -->  unit-tests  -->  PR Gate Status
```

- Triggers on: `pull_request`, `merge_group`, `push` to `main`
- Skips draft PRs
- Runs **only CPU unit tests** (`python -m pytest tests/blaze/ -v`)
- 30-minute timeout

### Merge Gate CI (`ci.yaml`)

Full pipeline required before merging:

```
download-build-artifact
    |-- unit-tests
    |-- silicon-tests (P150)
    |-- silicon-tests-bh-loudbox
    --> Merge Gate Status
```

All three test tiers must pass.

### Unit Tests (`unit-tests-impl.yaml`)

- **Runner**: `tt-ubuntu-2204-large-stable` (CPU-only)
- **Container**: `ubuntu-22.04-dev-amd64` with `ARCH_NAME=blackhole`
- **Command**: `python -m pytest tests/blaze/ -v`
- Tests: infra tests only (silicon paths excluded by `collect_ignore`)

### Silicon Tests -- Blackhole Loudbox (`silicon-tests-bh-loudbox-impl.yaml`)

- **Runners**: `in-service` + `BH-Loudbox` (non-ephemeral, direct ghcr.io pull)
- **Strategy**: 7 parallel buckets with `fail-fast: false`
- **Device access**: `--device /dev/tenstorrent`, hugepages mount

| Bucket | Contents |
|--------|----------|
| 1 | MOE, MLPMixed, Scatter, PreSDPA, QA Projection, Q Branch, KV Branch, CreateQheads, Embedding, EmbedMcast, ReduceToOne |
| 2 | CB Reconfig Dedup, CB Compaction, PostSDPA, SDPA, CB Overlap, Fused SDPA/PostSDPA, Overlapped Integration, MLA, GLM5 fused/Q ops |
| 3 | Matmul, KN Matmul, DRAM Streaming, Mcast, RMSNorm, BroadcastRMSNormMcast, Gather, Flash MLA, LM Head, SDPA |
| 4 | DSA multi-dispatch, PostDSA, Wo, GLM5 fused proj/kv/q branches, Untilize, Scatter raw, DSA flash decode 32-core, DSA sparse gather |
| 5 | MoE backed tests |
| 6 | GLM5 multi-device fused attention |
| 7 | CB Reconfig silicon |

Each bucket produces JUnit XML artifacts and publishes results via
`dorny/test-reporter`.

### Silicon Tests -- P150 (`silicon-tests-p150-impl.yaml`)

- **Runner**: `tt-ubuntu-2204-p150b-viommu-stable`
- **Timeout**: 35 minutes
- Runs `tests/blaze/infra/test_silicon.py`, `tests/blaze/micro-ops/`,
  selected infra silicon tests
- Ignores tests already covered by Loudbox buckets (extensive `--ignore` list)

### Build Artifact

All workflows depend on `download-build-artifact.yaml`, which fetches
pre-built `tt-metal` artifacts from the `blaze-metal` branch (or a custom
branch specified via `metal-branch-name` input).

## B.1.9 Test Execution Modes

Most silicon tests require `TT_METAL_SLOW_DISPATCH_MODE=1` (set in the CI
bucket commands). The key distinction:

- **Slow dispatch**: Host-driven kernel launch, used for Blaze/Blitz fused
  programs that manage their own kernel scheduling.
- **Fast dispatch**: Runtime-managed kernel queue, used for some DSA and
  GLM-5 micro-ops that do not require host coordination.

The CI buckets mix modes: buckets 1-3, 5-7 use slow dispatch; bucket 4 uses
fast dispatch for DSA ops.
