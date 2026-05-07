# Appendix A.2 -- Weight Provider Abstraction

The weight provider system decouples weight sourcing from the compilation and
execution pipeline. Tests and inference share identical calling code; only
the `WeightProvider` implementation changes. This appendix documents the ABC,
its three concrete implementations, the factory function, the disk cache, and
the weight transform pipeline.

Source files:

| File | Role |
|------|------|
| `blaze/weight_provider.py` | ABC + three implementations + factory |
| `tests/blaze/weight_cache.py` | `WeightCache` -- disk cache for torch tensors |
| `tests/blaze/device_weight_cache.py` | `DeviceWeightCache` -- device tensor cache |
| `tests/blaze/generate_weight_cache.py` | CLI for pre-populating caches |

## A.2.1 The Abstract Base Class

```python
class WeightProvider(ABC):
    """Abstract base for sourcing layer weights as torch tensors."""

    @abstractmethod
    def get_mla_torch_weights(self, layer_idx: int, **params) -> dict[str, torch.Tensor]:
        """Return MLA attention weights for *layer_idx*."""

    @abstractmethod
    def get_moe_torch_weights(self, layer_idx: int, **params) -> dict[str, torch.Tensor]:
        """Return MoE weights (gate, experts, shared) for *layer_idx*."""
```

Both methods return `dict[str, torch.Tensor]` with standardized keys.
The `params` keyword arguments carry model-dimension parameters that the
caller already knows (e.g., `num_tp`, `mesh_rows`, `embedding_dim`,
`hidden_dim`, `num_experts`).

### MLA Weight Keys

Returned by `get_mla_torch_weights`:

| Key | Shape (typical) | Description |
|-----|----------------|-------------|
| `t_gamma` | `[1, 7168]` | Input layernorm weight |
| `t_qa_w` | `[7168, 1536]` | Q_A projection (transposed) |
| `t_rn2_gamma` | `[1, 1536]` | Q_A layernorm weight |
| `t_qb_w` | `[1536, 24576]` | Q_B projection (deinterleaved) |
| `t_kv_a_w` | `[7168, 576]` | KV_A projection (transposed) |
| `t_kv_norm_gamma` | `[1, 512]` | KV_A layernorm weight |
| `t_kv_b1_w` | `[512, K]` | KV_B split part 1 |
| `t_kv_b2_w_full` | varies | KV_B split part 2 (full) |
| `t_kv_b2_mesh` | varies | KV_B part 2, mesh-replicated |
| `t_o_proj_w_full` | varies | Output projection (full) |
| `t_o_proj_mesh` | varies | Output projection, mesh-replicated |
| `cos_base` / `sin_base` | `[max_seq_len, 64]` | RoPE frequency tables |
| `trans_mat_raw` | `[1, 1, 32, 32]` | RoPE transformation matrix |
| `t_qa_w_packed` | `[3584, 3072]` | Q_A packed (half-split concat) |

### MoE Weight Keys

Returned by `get_moe_torch_weights`:

| Key | Shape (typical) | Description |
|-----|----------------|-------------|
| `rmsnorm_gamma_torch` | `[1, 7168]` | Post-attention layernorm |
| `moe_gate_weight_torch` | `[1, 1, 7168, 256]` | Gate weight (transposed) |
| `moe_gate_bias_torch` | `[1, 1, 1, 256]` | Gate bias (e_score_correction) |
| `routed_swiglu_gate_weight_torch` | `[256, 7168, 2048]` | Routed expert gate |
| `routed_swiglu_up_weight_torch` | `[256, 7168, 2048]` | Routed expert up |
| `routed_swiglu_down_weight_torch` | `[256, 2048, 7168]` | Routed expert down |
| `shared_gate_weights_torch` | `[4, 2, 7168, 256]` | Shared expert gate (per mesh) |
| `shared_up_weights_torch` | `[4, 2, 7168, 256]` | Shared expert up (per mesh) |
| `shared_down_weights_torch` | `[4, 2, 256, 7168]` | Shared expert down (per mesh) |
| `shared_gate_up_stacked_torch` | varies | Gate+up shards stacked |
| `routed_up_shuffled` | `[256, ...]` | Tile-shuffled routed up |
| `routed_gate_shuffled` | `[256, ...]` | Tile-shuffled routed gate |
| `routed_down_shuffled` | `[256, ...]` | Tile-shuffled routed down |

## A.2.2 Concrete Implementations

### SyntheticWeightProvider

```python
class SyntheticWeightProvider(WeightProvider):
    def __init__(
        self,
        *,
        mla_generator: Callable[..., dict[str, torch.Tensor]] | None = None,
        moe_generator: Callable[..., dict[str, torch.Tensor]] | None = None,
        seed: int = 42,
        cache_dir: Path | str | None = None,
    ): ...
```

Generates seeded random weights by delegating to caller-supplied generator
callables. The generator functions live in test code (e.g.,
`tests/blaze/backed/weight_helpers.py`) so this class stays independent of
the test tree.

The seed is deterministic: `torch.manual_seed(seed + layer_idx)` before
each generation call. When a `WeightCache` is importable (tests are on
`sys.path`), results are cached to disk via `cache.get_or_create()`.

### StateDictWeightProvider

```python
class StateDictWeightProvider(WeightProvider):
    def __init__(self, model_path: Path | str): ...
```

Loads real weights from HuggingFace safetensors at `model_path`. State dict
loading is lazy -- `_ensure_state_dict()` loads on first access, preferring
`LazyStateDict` (from tt-metal) for memory efficiency, falling back to
`safetensors.torch.load_file`.

Weight transforms applied (via `_state_dict_to_mla_tensors` and
`_state_dict_to_moe_tensors`):
- FP8 to bfloat16 conversion (`.to(torch.bfloat16)`)
- Transpose (`.T.contiguous()` for projections)
- `deinterleave_q_b_proj` for Q_B
- `split_kv_b_proj` to separate KV_B into parts 1 and 2
- TP slicing via `_slice_attention_weights_for_mla_tp`
- Mesh replication (stacking per-device TP slices along dim 0)
- `_shuffle_tensor_tiles` for routed expert DRAM bank layout
- Unsqueeze for norms

Results are cached to disk when `WeightCache` is importable, keyed by
model path, layer index, and TP parameters.

### BlitzCacheWeightProvider

```python
class BlitzCacheWeightProvider(WeightProvider):
    def __init__(self, cache_path: Path | str): ...
```

Loads weights from Blitz's tensorbin cache format -- pre-fused device tensors
stored as `.tensorbin` files with a `manifest.json` per layer.

Two loading paths:

1. **Torch path** (`get_mla_torch_weights`, `get_moe_torch_weights`):
   Reconstructs an HF-format state dict from tensorbin files via
   `_load_layer_state_dict()`, then applies the same transforms as
   `StateDictWeightProvider`.

2. **Device path** (`load_overlapped_views`): Loads tensors directly as
   `OverlappedView` objects, bypassing torch intermediates entirely. This
   is the fast path for production inference.

```python
def load_overlapped_views(self, layer_idx: int, device: Any) -> dict[str, Any]:
    """Load Blitz cache tensors directly as OverlappedViews (device path)."""
```

Returns a dict of `field_name -> OverlappedView` for fields like `q_a_proj`,
`q_b_proj`, `kv_a_proj`, `o_proj`, `gate_mm`, norms, and expert projections.

## A.2.3 Weight Transforms

### Tile Shuffling (`_shuffle_tensor_tiles`)

Routed expert weights must have their tiles reordered from row-major to
column-major within each DRAM bank shard for efficient streaming:

```python
def _shuffle_tensor_tiles(tensor: torch.Tensor, tile_size: int, num_banks: int) -> torch.Tensor:
    """Reorder tiles within each DRAM bank shard from row-major to column-major."""
```

The function pads N to the LCM of (tile_size * num_banks), reshapes into
bank shards, permutes tile indices via `source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)`,
and reshapes back. This ensures sequential DRAM reads fetch tiles in the
order the matmul engine consumes them.

### DRAM Bank Padding (`_pad_to_dram_banks`)

```python
def _pad_to_dram_banks(num: int, tile_w: int, lcm: int) -> int:
    remainder = num % lcm
    return num if remainder == 0 else num + (lcm - remainder)
```

Pads dimension sizes to align with DRAM bank boundaries (LCM of tile width
and bank count), preventing cross-bank tile splits.

### RoPE Transformation Matrix

```python
def _get_rot_transformation_mat() -> torch.Tensor:
```

Returns a `[1, 1, 32, 32]` matrix where even-indexed rows select
odd-indexed columns with weight +1, and odd-indexed rows select
even-indexed columns with weight -1. This implements the interleaved
RoPE rotation pattern.

## A.2.4 Factory Function

```python
def get_weight_provider(source: str | None = None) -> WeightProvider:
    """Create a WeightProvider based on *source* or BLAZE_WEIGHT_SOURCE env var."""
```

Source formats:

| Source string | Provider | Example |
|---------------|----------|---------|
| `synthetic` | `SyntheticWeightProvider()` | Default |
| `state_dict:<path>` | `StateDictWeightProvider(path)` | `state_dict:/proj/deepseek-r1` |
| `blitz:<path>` | `BlitzCacheWeightProvider(path)` | `blitz:/cache/deepseek_v3` |

If `source` is `None`, falls back to the `BLAZE_WEIGHT_SOURCE` environment
variable (default: `"synthetic"`).

Note: `SyntheticWeightProvider` created via the factory has no generators
attached -- the caller must supply them separately. This is by design:
generator functions live in test code and should not be imported by
production modules.

## A.2.5 Disk Cache (`WeightCache`)

```python
class WeightCache:
    """Transparent disk cache for seeded random torch tensors."""

    def __init__(self, cache_dir: Path | str | None = None): ...

    def get_or_create(
        self,
        func_name: str,
        params: dict,
        generator: Callable[[], dict[str, torch.Tensor]],
        source_func: Callable | None = None,
    ) -> tuple[dict[str, torch.Tensor], bool]: ...
```

Located at `tests/blaze/weight_cache.py`. Caches torch tensors (NOT device
tensors) to disk so subsequent test runs skip expensive generation.

Cache location priority:
1. `BLAZE_WEIGHT_CACHE_DIR` environment variable
2. `<repo_root>/.cache/weights/`

Cache invalidation uses `_source_hash(func)` -- a SHA-256 of the generator
function's source code. If the function body changes, cached entries are
regenerated.

Disable caching: set `BLAZE_WEIGHT_CACHE=0`.

## A.2.6 Pre-Generation CLI

```bash
# Generate all caches (MLA + MoE) with synthetic weights:
python tests/blaze/generate_weight_cache.py

# MLA only:
python tests/blaze/generate_weight_cache.py --mla

# MoE only:
python tests/blaze/generate_weight_cache.py --moe

# Use real DeepSeek R1 weights:
python tests/blaze/generate_weight_cache.py --source state_dict:/proj/deepseek-r1

# Clear all caches:
python tests/blaze/generate_weight_cache.py --clear
```

The script opens a 4x2 Blackhole mesh device and populates both the torch
tensor cache (Phase 1 via `WeightCache`) and the device tensorbin cache
(Phase 2 via `DeviceWeightCache`).

## A.2.7 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BLAZE_WEIGHT_SOURCE` | `synthetic` | Provider selection (see A.2.4) |
| `BLAZE_WEIGHT_CACHE` | `1` | Set to `0` to disable disk caching |
| `BLAZE_WEIGHT_CACHE_DIR` | `<repo>/.cache/weights/` | Override cache directory |

## A.2.8 Integration with BlazeCompiler

Weight providers are not directly consumed by `BlazeCompiler`. Instead, the
test or inference harness calls `provider.get_mla_torch_weights(layer_idx, **params)`
to obtain torch tensors, then converts them to device tensors via
`ttnn.from_torch(...)` before passing to `BlazeCompiler.compile(tensors=...)`.

For the BlitzCache device path, `load_overlapped_views()` returns
`OverlappedView` objects that can be passed directly to `FusedProgram`'s
`cb_from_tensor()` dispatch (see Ch 5 for OverlappedView mechanics).
