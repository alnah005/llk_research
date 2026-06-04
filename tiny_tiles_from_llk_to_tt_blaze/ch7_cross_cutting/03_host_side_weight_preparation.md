# 7.3 Host-Side Weight Preparation

Activations and weights are prepared on separate host-side paths. Activation tile geometry — including any tiny-tile shape declared at the micro-op `emit()` boundary — does **not** propagate into weight layout. Weights stay in canonical 32x32 DRAM/L1 layout regardless of the downstream activation pipeline geometry, with one structural exception (covered in Section 4). This file traces why that invariant holds, where it is enforced, and the narrow class of cases where `prepare_weights.py` must be tiny-tile aware.

**Prerequisites:** Chapter 1 (tiny-tile definition and face packing), Chapter 4 (TT-Metal tile metadata: `TileDescriptor`, `tile_shape`), Chapter 6 (CB handle propagation). Cross-references [[comprehensive_guide_to_creating_new_micro_ops_in_tt_blaze]] Chapter 6 for the host-side weight grouping pattern that this file builds on.

## 1. Weight Tile Geometry as Independent Spec

Weight tile geometry is decided when weights are tilized on the host, long before any activation pipeline runs. The decision input is the **weight shard shape** (set by tensor parallel factor and the matmul layout the kernel expects), not the activation tile carried on the corresponding input CB.

This separation is visible in `prepare_weights.py`. The activation tile for a given attention path is decided by the `emit()` for the consuming micro-op (see Chapter 6 on `CBHandle.tile_desc` propagation). The weight tile is decided here:

```python
# tt-metal/models/demos/deepseek_v3_b1/prepare_weights.py:807-808
_LM_HEAD_B_TILE = ttnn.Tile([32, 32])
_LM_HEAD_A_TILE = ttnn.Tile([1, 32])
```

The LM-head case is the loudest example: activation `A` is declared as a 1x32 tile (decode bs=1 per-token row), while weight `B` is declared as a canonical 32x32 tile. The two are independent declarations and the host code never derives one from the other. There is no function in `prepare_weights.py` that takes an activation tile descriptor as input.

For attention/FFN, the weight tiles are not even named explicitly in the dataclasses — they are implicit 32x32 because that is the default `ttnn.Tile` and no constructor overrides it for those projections.

## 2. Fusion Groups and the OverlappedTensor Pattern

Weights in DeepSeek V3 B1 are not allocated one tensor at a time. They are collected into **fusion groups** that share a single underlying DRAM allocation (one `fused_tensor`), with each individual weight exposed as an `OverlappedTensor` view into that allocation:

```python
# tt-metal/models/demos/deepseek_v3_b1/prepare_weights.py:45-60
_FIELD_TO_FUSION_GROUP: dict[str, str] = {
    "q_a_proj":         "q_ab_kv_a",
    "q_b_proj":         "q_ab_kv_a",
    "kv_a_proj":        "q_ab_kv_a",
    "o_proj":           "o_proj_gate_mm_norms",
    "gate_mm":          "o_proj_gate_mm_norms",
    "attn_norm":        "o_proj_gate_mm_norms",
    "q_norm":           "o_proj_gate_mm_norms",
    "kv_norm":          "o_proj_gate_mm_norms",
    "ffn_norm":         "o_proj_gate_mm_norms",
    "kv_b1_proj":       "kv_b12",
    "kv_b2_proj":       "kv_b12",
    "shared_gate_proj": "gate_up",
    "shared_up_proj":   "gate_up",
}
```

These four fusion groups (`q_ab_kv_a`, `o_proj_gate_mm_norms`, `kv_b12`, `gate_up`) are packed into contiguous DRAM regions and reach the kernel as views. The `AttentionWeights` dataclass enumerates the fields without specifying any tile geometry per field:

```python
# tt-metal/models/demos/deepseek_v3_b1/prepare_weights.py:63-78
@dataclass
class AttentionWeights:
    """Attention fusion groups: q_ab_kv_a + kv_b12 + o_proj_gate_mm_norms."""

    q_a_proj:    OverlappedTensor
    q_b_proj:    OverlappedTensor
    kv_a_proj:   OverlappedTensor
    o_proj:      OverlappedTensor
    gate_mm:     OverlappedTensor | None  # None for dense layers
    attn_norm:   OverlappedTensor
    q_norm:      OverlappedTensor
    kv_norm:     OverlappedTensor
    ffn_norm:    OverlappedTensor
    kv_b1_proj:  OverlappedTensor
    kv_b2_proj:  OverlappedTensor
    gate_bias:   ttnn.Tensor | None  # e_score_correction_bias for MoE only
```

The absence of any `tile_desc` / `tile_shape` field in the dataclass is the point. Tile geometry for each weight is implicit in the `ttnn.Tensor` carried inside the `OverlappedTensor` view, set once at tilization and never refreshed against activation pipeline state.

## 3. Non-Propagation from Activation to Weight

The strongest evidence that activation tile geometry does not flow into weight preparation is the signature of `_moe_routed_expert_stride_bytes`, which computes per-expert DRAM stride:

```python
# tt-metal/models/demos/deepseek_v3_b1/prepare_weights.py:103-120
def _moe_routed_expert_stride_bytes(weights_tensor: ttnn.Tensor) -> int:
    """Packed DRAM size of one routed expert tensor (bytes), per DRAMStreamingMatmul indexing."""
    shard_spec = weights_tensor.memory_config().shard_spec
    ...
    tile = weights_tensor.tile
    th, tw = tile.tile_shape[0], tile.tile_shape[1]
    per_core_n = weights_shard_shape[1] // tw
    Kt = K // th
    ...
    weights_tile_size = tile.get_tile_size(weights_tensor.dtype)
    return Kt * per_core_n * weights_tile_size
```

The only inputs are the **weights tensor itself** and its own `tile.tile_shape`. There is no activation parameter, no CBHandle, no consumer-side tile descriptor in the call. The stride is a function of weight shard shape and weight tile size only.

The top-level call chain that builds a decoder layer (`prepare_moe_decoder_layer` near line 703-730 of the same file) calls `prepare_attention_weights()` and `prepare_routed_expert_weights()` without threading any activation pipeline metadata. The pipeline that will *consume* these weights is built separately — it reads weight tensors back through `BlitzDecodeWeights` and pairs them with whatever activation tile the relevant micro-op declared.

Concretely, in MLA decode: Q activation is an 8x32 tile (8 KV heads per query group, packed as a tiny tile face), while K/V projections and o_proj weights remain 32x32. The consuming matmul kernel handles the mismatch through its own face-walking logic; the weight host-side path is unaware that Q is tiny.

| op | activation tile height | weight tile height | coupling |
|---|---|---|---|
| q_a_proj | 32 (rmsnorm out)      | 32 | none |
| q_b_proj | 32                    | 32 | none |
| MLA score (Q×Kᵀ) | 8 (Q is tiny)  | 32 | none — kernel handles asymmetry |
| MLA out (P×V)    | 8              | 32 | none — kernel handles asymmetry |
| o_proj   | 8 (attn out)          | 32 | none |
| gate / up | 32                   | 32 | none |
| lm_head matmul | 1 (decode token row) | 32 | **structural** — see Section 4 |
| MoE gate_bias add | 1            | 16 (bias is 16x16) | **structural** — bias shape |

Across this table only two rows have any coupling, and neither is "activation tile drives weight tile." Both are structural: the weight tensor is small enough on one axis that the natural shard shape falls below 32.

## 4. Rare Exception: Structurally Tiny Tiles on the Weight Side

Two weights in DeepSeek V3 B1 carry non-32x32 tile geometry, and both reasons are structural — they have nothing to do with downstream activation geometry.

**LM-head activation `A` is 1x32; the weight `B` is still 32x32:**

```python
# tt-metal/models/demos/deepseek_v3_b1/prepare_weights.py:795-808
# LM head: HF keeps full vocab (129280, 7168). Prepare shards vocab (N) across the mesh (TP=mesh size)
# and uses the same per-device L1 WIDTH_SHARDED layout as test_lm_head_sampling (101 matmul cores).

_LM_HEAD_K = 7168
_LM_HEAD_VOCAB_SIZE = 129280
_LM_HEAD_NUM_MATMUL_CORES = 101
...
_LM_HEAD_B_TILE = ttnn.Tile([32, 32])
_LM_HEAD_A_TILE = ttnn.Tile([1, 32])
_LM_HEAD_N_PER_CORE = 160
```

The weight tile stays canonical because both axes of the per-core shard remain ≥ 32: K = 7168 (way more than 32), and N_per_core = 160 (also ≥ 32). The activation being a 1x32 row is irrelevant to the weight layout decision — it just means each core multiplies a 1x32 row against a 32x160 weight shard one tile-row at a time.

If TP were extreme enough to drive `N_per_core` below 32 (e.g., a vocab of 129280 split across more than 4040 columns of devices, which is implausible), the weight tile width would become structurally tiny. That would be the trigger for `prepare_weights.py` to start caring — and the trigger would still be the **weight's own shard shape**, not the activation tile.

**MoE gate bias is 16x16:**

```python
# tt-metal/models/demos/deepseek_v3_b1/prepare_weights.py:43
_GATE_BIAS_TILE = ttnn.Tile([16, 16])
```

`e_score_correction_bias` is a 1-D vector of length = number of routed experts, which is < 32, so its natural tile becomes 16x16. This is a genuine structurally-tiny weight tile and the only one currently declared in `prepare_weights.py`. The reason is purely "this tensor is small," not "the consumer wanted a tiny tile."

**Rule of thumb:** `prepare_weights.py` declares a non-32x32 tile only when the weight tensor's own shard shape forces it on one or both axes — typically vocab-sharded heads, per-expert biases, or anything whose minor dimension is < 32 after TP/EP sharding.

## 5. Blitz Cache and Layout Decoupling

`BlitzDecodeWeights` (imported on prepare_weights.py:28) caches weights at their on-device layout:

```python
from models.demos.deepseek_v3_b1.blitz_decode_weights import (
    BlitzDecodeWeights, OverlappedTensor,
)
```

The cache key is a function of (weight identity, dtype, mesh shape, tilization parameters, fusion group). It is *not* a function of any activation pipeline. Consequences:

- Swapping the activation pipeline (e.g., switching the MLA decode kernel from an 8x32 Q tile to a 4x32 Q tile, or rewriting a fused op to fan out differently) does **not** invalidate the blitz cache.
- Re-tilization happens when the mesh shape or TP factor changes, because those change the weight shard shape — which is what actually determines the tile geometry.
- Two pipelines that share weights but use different activation tile geometries can both load from the same cached blob.

The decoupling is enforced by omission: nothing in the blitz cache key, in `OverlappedTensor`, or in the `prepare_*_weights` call signatures references CBHandle metadata, micro-op emit declarations, or activation tile descriptors.

## 6. Practical Checklist

When porting a new model or adding a tiny-tile micro-op:

1. Does the op consume a tiny-tile activation? → Declare the tile in the consumer micro-op's `emit()` (see Chapter 6). Do **not** touch `prepare_weights.py`.
2. Does the op multiply that activation by a standard weight matrix (e.g., a linear projection)? → Weight stays 32x32. The matmul kernel is responsible for handling the asymmetric face walk.
3. Is the weight tensor itself smaller than 32 on either axis after TP/EP sharding? → This is the only case where `prepare_weights.py` declares a non-32x32 `ttnn.Tile`. Compute the per-core shard shape first, then pick the largest legal tile dims that divide it (consult Chapter 1 / `tensor_shape.h` for the legal set).
4. Does adding the op affect the blitz cache? → Only if step 3 changed a weight tile or the fusion group layout. Activation-only changes never invalidate the cache.

Weights stay in 32x32 unless structurally constrained by shard shape. Activation tile geometry is the consumer's concern, not the host weight preparer's.
