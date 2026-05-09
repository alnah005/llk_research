# 7.1 Weight Preparation Pipeline

This section traces how raw HuggingFace DeepSeek V3 checkpoint tensors are transformed into the device-ready format consumed by the Blitz decode engine. The pipeline lives in `prepare_weights.py` (1402 lines) and orchestrates key mapping, transposition, the $W_{kv\_b}$ split, norm reshaping, fusion group assignment, TP slicing, and per-layer serialization -- producing `OverlappedTensor` and `ttnn.Tensor` objects that can be placed directly in device L1 or DRAM.

> **Cross-reference:** The overlap specs and `BlitzDecodeWeights` class that consume these prepared tensors are detailed in [Section 7.2](./02_blitz_decode_weight_overlapping.md). The caching system that serializes and reloads prepared weights is covered in [Section 7.3](./03_cache_serialization_and_special_tensors.md). The MLA and MoE compute kernels that use these weights at runtime are described in Chapters 5 and 6, respectively.

Source: `models/demos/deepseek_v3_b1/prepare_weights.py`

---

## 7.1.1 Lifecycle Overview

The weight preparation pipeline converts a HuggingFace `state_dict` into device-resident tensors through a well-defined sequence of stages:

```
HF checkpoint (safetensors)
        |
        v
  LazyStateDict (memory-mapped access)
        |
        v
  _get_layer_raw_tensors()        <-- key mapping, .T, kv_b split, norm unsqueeze
        |
        v
  _slice_*_for_*_tp()             <-- TP slicing for single-device testing
        |
        v
  BlitzDecodeWeights.get_tt_*()   <-- fusion, shuffle, tile-reshape, device placement
        |
        v
  OverlappedTensor / ttnn.Tensor  <-- device-ready weights
        |
        v
  save_*() / load_*()             <-- cache serialization (Section 7.3)
```

Each stage is deterministic and preserves mathematical equivalence -- the numerical values reaching the compute kernels are identical to those in the original HuggingFace checkpoint, just rearranged for hardware efficiency.

---

## 7.1.2 Key Mapping and Raw Tensor Extraction

The function `_get_layer_raw_tensors()` (lines 289--337) is the single entry point for extracting all attention-related tensors from a HuggingFace state dict. It performs three classes of transformation.

### State Dict Key Construction

All per-layer keys are constructed by `_key()` (line 205):

```python
def _key(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"
```

### Complete HF-to-Internal Transformation Table

The following table shows every tensor extracted by `_get_layer_raw_tensors()`, its HuggingFace key suffix, original shape, transformation, and the shape passed to the Blitz fusion layer:

| Weight | HF Key Suffix | HF Shape | Transform | Blitz Shape |
|--------|---------------|----------|-----------|-------------|
| `q_a_proj` | `self_attn.q_a_proj.weight` | $(1536, 7168)$ | `.T` | $(7168, 1536)$ |
| `q_b_proj` | `self_attn.q_b_proj.weight` | $(24576, 1536)$ | `.T` | $(1536, 24576)$ |
| `kv_a_proj` | `self_attn.kv_a_proj_with_mqa.weight` | $(576, 7168)$ | `.T` | $(7168, 576)$ |
| `kv_b1_proj` | `self_attn.kv_b_proj.weight` | $(32768, 512)$ | split | $(16384, 512)$ |
| `kv_b2_proj` | `self_attn.kv_b_proj.weight` | $(32768, 512)$ | split + `.T` | $(512, 16384)$ |
| `o_proj` | `self_attn.o_proj.weight` | $(7168, 16384)$ | `.T` | $(16384, 7168)$ |
| `attn_norm` | `input_layernorm.weight` | $(7168,)$ | `unsqueeze(0)` | $(1, 7168)$ |
| `q_norm` | `self_attn.q_a_layernorm.weight` | $(1536,)$ | `unsqueeze(0)` | $(1, 1536)$ |
| `kv_norm` | `self_attn.kv_a_layernorm.weight` | $(512,)$ | `unsqueeze(0)` | $(1, 512)$ |
| `ffn_norm` | `post_attention_layernorm.weight` | $(7168,)$ | `unsqueeze(0)` | $(1, 7168)$ |

The transpose convention is significant: HuggingFace stores weights as $(d_\text{out}, d_\text{in})$ for `nn.Linear`, but the Blitz matmul kernels expect $(K, N) = (d_\text{in}, d_\text{out})$ so that a row vector left-multiplies the weight: $\mathbf{y} = \mathbf{x} \cdot W$.

MoE layers additionally require tensors read directly in `prepare_attention_weights()` and `prepare_shared_expert_weights()`:

| Weight | HF Key Suffix | HF Shape | Transform | Blitz Shape |
|--------|---------------|----------|-----------|-------------|
| `gate_mm` | `mlp.gate.weight` | $(256, 7168)$ | `.T` | $(7168, 256)$ |
| `gate_bias` | `mlp.gate.e_score_correction_bias` | $(256,)$ | reshape | $(16, 16)$ |
| `shared_gate` | `mlp.shared_experts.gate_proj.weight` | $(2048, 7168)$ | `.T` | $(7168, 2048)$ |
| `shared_up` | `mlp.shared_experts.up_proj.weight` | $(2048, 7168)$ | `.T` | $(7168, 2048)$ |
| `shared_down` | `mlp.shared_experts.down_proj.weight` | $(7168, 2048)$ | `.T` | $(2048, 7168)$ |

For dense layers (0--2), the MLP keys differ: `mlp.gate_proj.weight` with HF shape $(18432, 7168)$, containing 9 experts fused into one weight. The first 2048 columns serve as the shared expert; the remaining $8 \times 2048$ are distributed as one routed expert per device.

### The $W_{kv\_b}$ Split

The `_split_kv_b_proj()` function (lines 210--224) decomposes the monolithic HuggingFace $W_{kv\_b}$ $(32768, 512)$ into $W_{kv\_b1}$ $(16384, 512)$ and $W_{kv\_b2}$ $(512, 16384)$.

> **Cross-reference → Chapter 5, Section 5.1.4**: The reshape-split algorithm and per-head decomposition ($d_\text{qk\_nope} = 128$ elements to kv\_b1, $d_v = 128$ to kv\_b2).

Critically, only $W_{kv\_b2}$ is transposed (with `.T.contiguous()`) — this matches the decode-time computation where $W_{kv\_b1}$ is the right operand in the absorbed Q-path matmul3 and $W_{kv\_b2}$ is the post-SDPA matmul5 value reconstruction.

> **Cross-reference -> Chapter 5, Section 5.1**: The MLA weight consumption pattern that motivates this asymmetric split.

Key constants governing the split (lines 196--202):

| Constant | Value | Meaning |
|----------|-------|---------|
| `_NUM_HEADS` | 64 | Defined but unused; `_split_kv_b_proj` computes `num_heads` dynamically |
| `_QK_NOPE_HEAD_DIM` | 128 | Nope dimension per head |
| `_V_HEAD_DIM` | 128 | V dimension per head |
| `_KV_LORA_RANK` | 512 | KV latent bottleneck dimension |
| `_KV_B_PROJ_HEAD_DIM` | 256 | $128 + 128$ combined per head |

### Norm Unsqueezing

All four RMSNorm gamma vectors are `unsqueeze(0)` from 1-D $(W,)$ to 2-D $(1, W)$. This matches the tile layout expected by the hardware: even 1-row gammas are stored as tiles (either $1 \times 32$ or $32 \times 32$, depending on the overlap spec).

---

## 7.1.3 Dataclass Hierarchy

`prepare_weights.py` defines a hierarchy of frozen dataclasses that serve as typed containers for prepared weights. Each dataclass maps one-to-one to the weight requirements of a specific model component:

### Intermediate Grouping Dataclasses

```python
@dataclass
class AttentionWeights:                    # Line 63
    q_a_proj, q_b_proj, kv_a_proj:   OverlappedTensor  # q_ab_kv_a group
    o_proj, gate_mm:                  OverlappedTensor | None  # o_proj_gate_mm_norms group
    attn_norm, q_norm, kv_norm, ffn_norm: OverlappedTensor
    kv_b1_proj, kv_b2_proj:          OverlappedTensor  # kv_b12 group
    gate_bias:                        ttnn.Tensor | None  # MoE only

@dataclass
class SharedExpertWeights:                 # Line 82
    shared_gate_proj, shared_up_proj: OverlappedTensor  # gate_up group
    shared_down_proj:                 ttnn.Tensor  # Not overlapped

@dataclass
class DenseRoutedExpertWeights:            # Line 91
    routed_gate_proj, routed_up_proj, routed_down_proj: ttnn.Tensor

@dataclass
class MoERoutedExpertWeights:              # Line 100
    routed_gate_proj, routed_up_proj, routed_down_proj: list[ttnn.Tensor]  # 256 each
```

### Layer-Level Dataclasses

> **Cross-reference → Chapter 1, Section 3.2**: Full definitions of `DeepSeekV3DenseLayerWeights` (line 109, 16 fields), `DeepSeekV3MoELayerWeights` (line 143, 18 fields), `DeepSeekV3EmbeddingLayerWeights` (line 181), and `DeepSeekV3LMHeadWeights` (line 188) with per-field shape and placement.

The intermediate grouping dataclasses above enable incremental cache generation — attention, shared experts, and routed experts can be prepared and saved independently (critical for MoE layers where the 256 routed experts dominate preparation time).

---

## 7.1.4 Fusion Group Mapping

The `_FIELD_TO_FUSION_GROUP` dictionary (lines 46--60) defines which weight fields share a fused device buffer:

```python
_FIELD_TO_FUSION_GROUP: dict[str, str] = {
    "q_a_proj":        "q_ab_kv_a",
    "q_b_proj":        "q_ab_kv_a",
    "kv_a_proj":       "q_ab_kv_a",
    "o_proj":          "o_proj_gate_mm_norms",
    "gate_mm":         "o_proj_gate_mm_norms",
    "attn_norm":       "o_proj_gate_mm_norms",
    "q_norm":          "o_proj_gate_mm_norms",
    "kv_norm":         "o_proj_gate_mm_norms",
    "ffn_norm":        "o_proj_gate_mm_norms",
    "kv_b1_proj":      "kv_b12",
    "kv_b2_proj":      "kv_b12",
    "shared_gate_proj": "gate_up",
    "shared_up_proj":   "gate_up",
}
```

All fields assigned to the same fusion group name share a single `ttnn.Tensor` as their backing store. The four fusion groups (`q_ab_kv_a`, `o_proj_gate_mm_norms`, `kv_b12`, `gate_up`) are detailed with dtypes, sharding strategies, and core placements in Sections 7.2.4--7.2.7. Tensors not in any fusion group are stored as standalone `.tensorbin` files: `shared_down_proj`, `gate_bias` (MoE only), and per-expert routed projections.

---

## 7.1.5 TP Slicing Functions

Two slicing functions handle the case where a state dict contains full-model logical shapes but the runtime targets a single device (TP=1):

### `_slice_attention_weights_for_mla_tp()` (lines 238--262)

When `mla_tp == 1`, slices the first TP-shard from full 2-TP logical shapes:

| Tensor | Full 2-TP Shape | Sliced TP=1 Shape | Slice Dimension | Per-TP Constant |
|--------|-----------------|-------------------|-----------------|-----------------|
| `q_b` | $(1536, 24576)$ | $(1536, 12288)$ | columns `[:12288]` | `_MLA_TP1_Q_B_WIDTH = 12288` |
| `o_proj` | $(16384, 7168)$ | $(8192, 7168)$ | rows `[:8192]` | `_MLA_TP1_O_PROJ_HEIGHT = 8192` |
| `kv_b1` | $(16384, 512)$ | $(8192, 512)$ | rows `[:8192]` | `_MLA_TP1_KV_B1_HEIGHT = 8192` |
| `kv_b2` | $(512, 16384)$ | $(512, 8192)$ | columns `[:8192]` | `_MLA_TP1_KV_B2_WIDTH = 8192` |

Note: `q_a_proj` $(7168, 1536)$ and `kv_a_proj` $(7168, 576)$ are **not** TP-sliced -- they are replicated on every device because the latent dimension is shared across all heads.

### `_slice_shared_expert_weights_for_moe_tp()` (lines 265--286)

When `moe_tp == 1`, slices the first TP-shard from full 8-TP logical shapes:

| Tensor | Full 8-TP Shape | Sliced TP=1 Shape | Slice Dimension |
|--------|-----------------|-------------------|-----------------|
| `shared_gate` | $(7168, 2048)$ | $(7168, 256)$ | columns `[:256]` |
| `shared_up` | $(7168, 2048)$ | $(7168, 256)$ | columns `[:256]` |
| `shared_down` | $(2048, 7168)$ | $(256, 7168)$ | rows `[:256]` |

Both functions are no-ops when the TP factor is greater than 1 (i.e., on the actual multi-device mesh), making the preparation code path-independent of target topology.

---

## 7.1.6 Layer Preparation Entry Points

Three top-level preparation functions orchestrate the full pipeline for each layer type:

### `prepare_attention_weights()` (lines 417--489)

Prepares the three attention fusion groups for one layer:

1. Calls `_get_layer_raw_tensors()` to extract and transform all attention tensors
2. Applies TP slicing via `_slice_attention_weights_for_mla_tp()`
3. Calls `bdw.get_tt_q_ab_proj_and_kv_a_proj_weights()` -- produces the `q_ab_kv_a` fusion group
4. Calls `bdw.get_tt_kv_b12_proj_weights()` -- produces the `kv_b12` fusion group
5. For MoE layers: reads `mlp.gate.weight`, calls `bdw.get_tt_o_proj_and_gate_mm_weights()` with the real gate_mm, and creates the `gate_bias` tensor
6. For dense layers: passes a zero dummy for gate_mm (the gate_mm `OverlappedTensor` is discarded at line 482)

The MoE path also creates the gate bias via `create_gate_bias_tensor()` (lines 345--382), which reshapes the 256-element `e_score_correction_bias` from $(256,)$ to $(16, 16)$, transposes it, and places it as a HEIGHT_SHARDED tensor on the sender core $(10, 9)$ with $16 \times 16$ tiles.

### `prepare_shared_expert_weights()` (lines 492--526)

Handles MoE and dense shared expert paths differently:

```python
if is_moe:
    shared_gate = state_dict[_key(layer_idx, "mlp.shared_experts.gate_proj.weight")].T
    shared_up   = state_dict[_key(layer_idx, "mlp.shared_experts.up_proj.weight")].T
    shared_down = state_dict[_key(layer_idx, "mlp.shared_experts.down_proj.weight")].T
else:
    mlp_gate = state_dict[_key(layer_idx, "mlp.gate_proj.weight")].T
    # get_tt_mlp_shared_expert_weights extracts first 2048 cols/rows
```

### `prepare_routed_expert_weights()` (lines 529--582)

The most expensive step for MoE layers -- iterates over 256 experts:

```python
for e in range(256):
    gate_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.gate_proj.weight")].T)
    up_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.up_proj.weight")].T)
    down_list.append(state_dict[_key(layer_idx, f"mlp.experts.{e}.down_proj.weight")].T)
gate_stacked = torch.stack(gate_list, dim=0)   # (256, 7168, 2048)
```

Each expert is uploaded as a separate WIDTH_SHARDED DRAM tensor, replicated across all 8 devices.

---

## 7.1.7 Expert Weight Handling

The routed expert pipeline deserves special attention due to its scale. For a single MoE layer with 256 experts:

### Per-Expert Dimensions

| Projection | Shape | dtype | Storage |
|---|---|---|---|
| gate_proj | $(7168, 2048)$ | BFP4 | DRAM, WIDTH_SHARDED |
| up_proj | $(7168, 2048)$ | BFP4 | DRAM, WIDTH_SHARDED |
| down_proj | $(2048, 7168)$ | BFP4 | DRAM, WIDTH_SHARDED |

### DRAM Tile Shuffle

Before upload, each expert weight undergoes `_shuffle_dram_tiles()` (lines 1555--1613 of `blitz_decode_weights.py`), which reorders tiles within each DRAM bank shard from row-major to column-major order. This ensures the streaming matmul kernel can linearly read $K$ tiles contiguously per $N$ column.

> **Cross-reference -> Chapter 3**: The DRAM tile shuffle algorithm and the `dram_streaming_matmul` per-core bank_id/vc assignment.

### Dense Layer Routed Experts

Dense layers (0--2) use a different partitioning. The `get_tt_mlp_routed_expert_weights()` method (lines 1458--1549) treats the full MLP weight as containing 9 "experts" of width 2048: the first is the shared expert, and the remaining 8 are assigned one per device across the $4 \times 2$ mesh via `ShardTensor2dMesh`.

---

## 7.1.8 End-to-End Data Flow: `q_a_proj` Through All Stages

To make the pipeline concrete, here is the complete journey of the `q_a_proj` weight for layer 5 (an MoE layer) on the $4 \times 2$ mesh (`mla_tp=2`):

| Stage | Operation | Shape |
|-------|-----------|-------|
| HF checkpoint | `self_attn.q_a_proj.weight` | $(1536, 7168)$ |
| Key mapping | `_key(5, "self_attn.q_a_proj.weight")` | -- |
| Transpose | `.T.contiguous()` | $(7168, 1536)$ |
| TP slice | Not sliced (q_a replicated across TP devices) | $(7168, 1536)$ |
| K-interleave pack | `shuffle_q_a`: `reshape(2, 3584, 1536).permute(1, 0, 2).reshape(3584, 3072)` | $(3584, 3072)$ |
| Stitch with q_b | `_stitch_width_sharded` fuses per-core shards vertically | Part of fused tensor across 96 cores |
| Quantize | `ttnn.from_torch(dtype=bfloat8_b)` | BFP8, $32 \times 32$ tiles |
| Device placement | WIDTH_SHARDED L1 via `ShardTensor2dMesh(dims=(None, 1))` | Per-core shard: $(3584, 32)$ |
| OverlappedTensor | View at byte offset 0 within fused buffer | `tensor_shape=(3584, 3072)`, `shard_shape=(3584, 32)` |
| Serialization | `ttnn.dump_tensor` to `layer_005/q_ab_kv_a.tensorbin` | -- |
| Manifest | Metadata saved under `fusion_groups.q_ab_kv_a.fields.q_a_proj` | -- |
| Runtime load | `_overlapped_tensor_from_dict` reconstructs `OverlappedTensor` | Identical to prepared form |

---

## 7.1.9 End-to-End Data Flow: `kv_b_proj` Split

The `kv_b_proj` transformation produces two sub-tensors in the same fusion group:

| Stage | Operation | kv_b1 Shape | kv_b2 Shape |
|-------|-----------|-------------|-------------|
| HF checkpoint | `self_attn.kv_b_proj.weight` | $(32768, 512)$ | -- |
| Reshape | `.reshape(128, 256, 512)` | $(128, 256, 512)$ | -- |
| Split at dim 128 | `[:, :128, :].reshape(-1, 512)` / `[:, 128:, :].reshape(-1, 512).T` | $(16384, 512)$ | $(512, 16384)$ |
| TP slice (mla_tp=2) | Passed through (full shapes kept) | $(16384, 512)$ | $(512, 16384)$ |
| Per-TP slice inside BDW | First 64 heads | $(8192, 512)$ | $(512, 8192)$ |
| kv_b2 tile-reshape | `shuffle_kv_b2()` | -- | $(128, 512)$ per core |
| Fused buffer | HEIGHT_SHARDED, 128 cores total | shard $(128, 512)$ | shard $(128, 512)$ |
| OverlappedTensor | kv_b1 on 64 Qnope cores, kv_b2 on 64 other cores | byte_offset=0 | byte_offset=0 |

Note that both `kv_b1_proj` and `kv_b2_proj` have `byte_offset=0` because they occupy different core regions within the same fused buffer -- they do not vertically stack like q_a and q_b do.

---

## 7.1.10 Dense vs. MoE Layer Comparison

Understanding the structural differences between dense and MoE layer weights is critical for the cache system:

| Aspect | Dense Layer (0--2) | MoE Layer (3--60) |
|--------|-------------------|-------------------|
| Gate weight | None (zero dummy passed, discarded) | $(7168, 256)$ in `o_proj_gate_mm_norms` |
| Gate bias | None | $(256,)$ standalone on sender core |
| Shared expert source | `mlp.{gate,up,down}_proj.weight` (first 2048 cols/rows) | `mlp.shared_experts.{gate,up,down}_proj.weight` |
| Routed experts | 1 per device (8 total from MLP experts 1--8) | 256, all replicated to every device |
| Routed expert storage | DRAM, 1 tensor per proj per device | DRAM, 256 tensors per proj |
| Total `.tensorbin` files | ~9 | ~775 (4 fusion + 2 standalone + manifest + 3 x 256 experts) |
| Cache generation time | ~3 minutes | ~5--10 minutes (dominated by 256 expert uploads) |

The routed expert count is the primary driver of cache size: each MoE layer stores $256 \times 3 = 768$ expert `.tensorbin` files. Across 58 MoE layers, this amounts to $44{,}544$ expert files -- motivating the split between `moe` mode (fast, fusion groups only) and `experts` mode (slow, expert uploads) in `generate_cache.py`.

---

## 7.1.11 The Full Layer Assembly

The `prepare_dense_layer_weights()` (lines 585--616) and `prepare_moe_layer_weights()` (lines 618--656) functions assemble the complete layer dataclass by calling all three preparation functions in sequence:

```python
def prepare_moe_layer_weights(bdw, state_dict, layer_idx, *, num_routed_experts=256):
    attn   = prepare_attention_weights(bdw, state_dict, layer_idx, is_moe=True)
    shared = prepare_shared_expert_weights(bdw, state_dict, layer_idx, is_moe=True)
    routed = prepare_routed_expert_weights(bdw, state_dict, layer_idx,
                                           is_moe=True, num_routed_experts=num_routed_experts)
    return DeepSeekV3MoELayerWeights(
        q_a_proj=attn.q_a_proj, q_b_proj=attn.q_b_proj, kv_a_proj=attn.kv_a_proj,
        o_proj=attn.o_proj, gate_mm=attn.gate_mm, ...
        routed_gate_proj=routed.routed_gate_proj, ...
    )
```

The returned dataclass contains all weights needed to execute one decoder layer: 4 fused `OverlappedTensor` groups, 2--3 standalone `ttnn.Tensor` objects, and (for MoE) 3 lists of 256 `ttnn.Tensor` objects for the routed experts.

---

## 7.1.12 Numerical Precision Through the Pipeline

Each stage of the preparation pipeline has specific precision implications:

### Transposition, Reshape, and Shuffle

All transposition, reshape, and shuffle operations are exact: `.T.contiguous()`, `.reshape()`, `.unsqueeze()`, the kv_b split, and column/row reorderings produce numerically identical floating-point values. No quantization occurs at these stages.

### Quantization Boundary

The single point of precision loss is the conversion from float32 PyTorch tensors to device dtype:

| Target dtype | Precision | Used By |
|---|---|---|
| BFP8 (`bfloat8_b`) | 7-bit mantissa, shared exponent per 16-element block | q_ab_kv_a, kv_b12, o_proj, LM head |
| BFP4 (`bfloat4_b`) | 3-bit mantissa, shared exponent per 16-element block | gate_up, down_proj, routed experts |
| BFP16 (`bfloat16`) | 7-bit mantissa, per-element exponent | gate_mm, norm gammas |
| UINT32 | Exact (raw bytes) | o_proj_gate_mm_norms fused container |

For the `o_proj_gate_mm_norms` fusion group, quantization happens during the manual `_tilize_and_pack_bfp8()` and `_tilize_and_pack_bfloat16()` calls rather than in `ttnn.from_torch()`, because the fused tensor is stored as raw UINT32 bytes. The BFP8 packing implements round-to-nearest-even (lines 1685--1687 of `blitz_decode_weights.py`):

```python
round_up = (round_value > TIE) | ((round_value == TIE) & (guard_bit == 1))
mantissa7 = np.where(round_up, np.minimum(mantissa7 + 1, 127), mantissa7)
```

This matches the C++ hardware encoder (`pack_as_bfp_tiles<Bfp8_b>`), ensuring that weights prepared on the host are bit-identical to what the hardware would produce from the same float32 input.

---

## 7.1.13 Memory Considerations

The preparation pipeline processes one layer at a time to bound peak memory usage:

- **`LazyStateDict`**: Memory-maps HuggingFace safetensors shards, loading only the requested tensors into RAM. Without lazy loading, the full 671B model would require ~1.3 TB of host memory.

- **Expert stacking**: For MoE layers, all 256 experts' gate/up/down projections are stacked into three tensors of shape $(256, K, N)$ before calling `get_tt_moe_routed_expert_weights()`. Peak host memory per MoE layer during expert preparation: approximately $256 \times 3 \times 7168 \times 2048 \times 2$ bytes (BFP16) $\approx 22$ GB.

- **Contiguous copies**: The `.T.contiguous()` pattern creates a new contiguous tensor after transposition. This is necessary because `ttnn.from_torch()` requires contiguous memory, but it doubles the momentary memory usage for each transposed tensor.

- **Garbage collection**: After each layer is saved to cache, its tensors can be freed before the next layer is prepared. The `generate_cache.py` script processes one layer per invocation, ensuring clean memory boundaries.

---

## 7.1.14 Error Handling and Validation

The preparation pipeline includes extensive shape assertions that catch common mistakes early:

- **Input shape validation**: Every `get_tt_*()` method in `BlitzDecodeWeights` asserts that input tensors match the expected shapes for the current TP configuration (lines 712--721 for q_ab_kv_a, lines 866--882 for o_proj_gate_mm_norms).

- **TP consistency**: The `_slice_*_for_*_tp()` functions check shard widths against expected per-TP dimensions, using module-level constants (`_MLA_TP1_Q_B_WIDTH = 12288`, `_MLA_TP1_O_PROJ_HEIGHT = 8192`, etc.).

- **Shard width alignment**: `_stitch_width_sharded()` asserts that kv_a shard width equals the fused q_ab shard width at line 744.

- **Layer type dispatch**: `prepare_dense_layer_weights()` and `prepare_moe_layer_weights()` are separate functions with distinct type assertions, preventing accidental cross-calling.

These assertions are designed to fail early with clear messages when the state dict does not match the expected DeepSeek V3 architecture, avoiding cryptic failures deeper in the device upload path.

---

## 7.1.15 Summary: From Checkpoint to Device

| Stage | Input | Output | Key Function |
|-------|-------|--------|--------------|
| Key mapping | HF state_dict keys | `model.layers.{i}.{suffix}` | `_key()` |
| Transpose | $(d_\text{out}, d_\text{in})$ | $(K, N) = (d_\text{in}, d_\text{out})$ | `.T.contiguous()` |
| kv_b split | $(32768, 512)$ | kv_b1 $(16384, 512)$ + kv_b2 $(512, 16384)$ | `_split_kv_b_proj()` |
| Norm reshape | $(W,)$ | $(1, W)$ | `.unsqueeze(0)` |
| TP slicing | Full logical shape | Per-device shape | `_slice_*_for_*_tp()` |
| Fusion + shuffle | Raw torch tensors | `OverlappedTensor` groups | `BlitzDecodeWeights.get_tt_*()` |
| Expert upload | $(256, K, N)$ stacked | 256 DRAM tensors | `get_tt_moe_routed_expert_weights()` |

---

| [< Chapter 6: MoE Deep Dive](../ch06_mixture_of_experts_deep_dive/01_moe_gating_and_routing.md) | [7.2 Blitz Decode Weight Overlapping >](./02_blitz_decode_weight_overlapping.md) |
|:---|---:|
