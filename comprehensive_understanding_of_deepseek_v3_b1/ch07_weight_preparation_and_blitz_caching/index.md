# Chapter 7 -- Weight Preparation and Blitz Caching

Chapters 5 and 6 showed how the attention and MoE kernels consume weight tensors at runtime. This chapter traces where those tensors come from: the offline pipeline that transforms HuggingFace DeepSeek V3 checkpoints into the exact device-ready format the compute kernels expect.

The system has three layers:

1. **Weight preparation pipeline** -- key mapping, transposition, the $W_{kv\_b}$ split, norm reshaping, TP slicing, and fusion group assignment in `prepare_weights.py`.
2. **Blitz decode weight overlapping** -- fusing multiple weight tensors into single WIDTH_SHARDED device buffers where sub-weights share an L1 base address, accessed via `OverlappedTensor` byte offsets.
3. **Cache serialization** -- the offline `generate_cache.py` script that persists prepared weights as `.tensorbin` files, reducing model load time from minutes to seconds.

Source root: `models/demos/deepseek_v3_b1/`

---

## Contents

### [7.1 Weight Preparation Pipeline](./01_weight_preparation_pipeline.md)

| Topic | Key Concepts |
|-------|-------------|
| HF-to-device pipeline | state_dict → key mapping → transpose → kv_b split → norm unsqueeze → fusion group → TP slice → serialize |
| Dataclass hierarchy | `AttentionWeights`, `SharedExpertWeights`, `DenseRoutedExpertWeights` → layer-level dataclasses |
| `_split_kv_b_proj()` | Splits $W_{kv\_b}$ [7168, 576] into kv_b1 [8192, 512] and kv_b2 [512, 8192] with `.contiguous()` |
| Fusion groups | `_FIELD_TO_FUSION_GROUP`: q_ab_kv_a, kv_b12, o_proj_gate_mm_norms, gate_up |
| TP slicing | `_slice_attention_weights_for_mla_tp()` for column-parallel MLA distribution |
| Expert weights | 256 experts × {gate, up, down}_proj as individual DRAM-sharded tensors |

### [7.2 Blitz Decode Weight Overlapping](./02_blitz_decode_weight_overlapping.md)

| Topic | Key Concepts |
|-------|-------------|
| Overlapping concept | Multiple weight tensors fused into one WIDTH_SHARDED buffer; sub-weights at known row offsets |
| `OverlappedTensor` | `byte_offset`, `total_size`, `core_range_set`, `tile_shape`, `dtype` |
| `BlitzDecodeWeights` | Groups fields by fusion spec, builds fused device tensors per overlap group |
| 4 overlap specs | `QAB_KVA_PROJ`, `O_PROJ_GATE_MM_RMSNORM_GAMMA`, `KVB12_PROJ`, `GATE_UP_PROJ` |
| Tile reshape | Reconciling mismatched shard widths across fused tensors |
| BFP8 encoding | 1088 bytes/tile (16 exponent uint32 + 256 mantissa uint32), round-to-nearest-even |

### [7.3 Cache Serialization and Special Tensors](./03_cache_serialization_and_special_tensors.md)

| Topic | Key Concepts |
|-------|-------------|
| `generate_cache.py` | Offline script: loads HF checkpoint → prepares weights → serializes to `.tensorbin` per layer |
| Cache structure | `layer_{N}/` directories with manifest JSON + binary tensor files |
| Embedding/LM head | Separate preparation paths; embedding as DRAM-interleaved, LM head as WIDTH_SHARDED |
| Special tensors | `gate_bias` (256 floats, tile-size trimmed), `gate_indices` (static permutation) |
| Load path | `load_dense_decoder_layer()` / `load_moe_decoder_layer()` consume cached `.tensorbin` files |

---

**Previous:** [Chapter 6 -- Mixture-of-Experts Deep Dive](../ch06_mixture_of_experts_deep_dive/index.md)
**Next:** [Chapter 8 -- Multi-Device Scaleout, Demo Runtime, and Full Decode Trace](../ch08_multi_device_scaleout_demo_runtime_and_full_decode_trace/index.md)
