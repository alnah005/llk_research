# 7.3 Cache Serialization and Special Tensors

This section covers the offline weight cache system that eliminates the need to re-run the full weight preparation pipeline at model load time, the `generate_cache.py` script that drives it, the embedding and LM head handling, and the special tensors (gate bias, gate indices) that do not fit the standard fusion group model. The cache system reduces DeepSeek V3 startup time from tens of minutes (online HuggingFace-to-device transformation) to seconds (loading pre-serialized `.tensorbin` files directly to device memory).

> **Cross-references:** Section 7.1 covers the weight preparation pipeline that produces the tensors serialized here. Section 7.2 covers the overlapping system whose fused buffers are the primary cache payload. Chapter 6 Section 6.1 covers the gate bias and gate indices consumption by the MoE gating kernel.

Source: `models/demos/deepseek_v3_b1/scripts/generate_cache.py` (771 lines), `models/demos/deepseek_v3_b1/prepare_weights.py` (serialization functions)

---

## 7.3.1 The generate\_cache.py Script

### Purpose and Design

The `generate_cache.py` script is the offline tool that converts HuggingFace DeepSeek V3 safetensors into the device-ready `.tensorbin` cache format. It runs on a machine with a $4 \times 2$ Blackhole mesh device and HuggingFace model files, and produces a directory tree that can be copied to any deployment machine for instant model loading.

The script is designed around five modes that can be invoked independently:

| Mode | Layers | Requires | Dispatch Mode | Output Directory |
|---|---|---|---|---|
| `dense` | 0--2 | `--layer-num`, `--model-path` | Slow dispatch | `layer_NNN/` |
| `moe` | 3--60 | `--layer-num`, `--model-path` | Slow dispatch | `layer_NNN/` |
| `experts` | 3--60 | `--layer-num`, `--model-path` | Fast dispatch OK | `layer_NNN/experts/` |
| `embedding` | N/A | `--model-path` | Either | `embedding/` |
| `lm_head` | N/A | `--model-path` | Either | `lm_head/` |

### Why Separate `moe` and `experts` Modes?

The split between `moe` (attention + shared experts) and `experts` (256 routed experts) is the most operationally significant design decision in the cache system:

1. **Dispatch mode constraints.** The attention and shared expert weight preparation uses `ttnn.from_torch()` with complex `MemoryConfig` and `MeshMapper` configurations that require slow dispatch mode (`TT_METAL_SLOW_DISPATCH_MODE=1`). Routed expert preparation uses simpler per-expert uploads that work in both slow and fast dispatch. Since dispatch mode can only be set once per program, the split allows operators to prepare routed experts in fast dispatch (faster per-upload) while preparing attention/shared in slow dispatch (required by the framework).

2. **Memory and time budget.** Loading 256 experts for a single MoE layer requires ~22 GB of host memory and takes 30--60 seconds. With 58 MoE layers, this totals ~1300 GB of I/O across all layers. The `experts` mode can be parallelized across multiple workers (one per layer, with `--layer-num` controlling which layer to process), while the `moe` mode is typically run sequentially.

### Main Execution Flow

The `main()` function (lines 621--767) follows a consistent pattern:

```python
with LazyStateDict(model_path) as state_dict:
    with bh_2d_mesh_device_context(device_params) as mesh_device:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(4, 2))
        bdw = BlitzDecodeWeights(submesh)

        if mode == "dense":
            layer = prepare_dense_layer_weights(bdw, state_dict, layer_num)
            save_decoder_layer(layer, output_path, layer_num, ...)
```

The `LazyStateDict` is a custom wrapper around HuggingFace safetensors that avoids loading the entire 1.3 TB model into memory. It reads the `model.safetensors.index.json` to build a key-to-file mapping, then loads individual tensors on demand when accessed via `state_dict[key]`.

### Argument Validation

The `_validate_args()` function (lines 129--256) enforces strict constraints:

- Dense mode only accepts layer indices 0--2 ($< \text{FIRST\_K\_DENSE\_REPLACE} = 3$).
- MoE and experts modes only accept layer indices 3--60 ($\geq 3$).
- `--force` is required to overwrite existing cache files.

### Dispatch Mode Requirements

Lines 244--255 enforce dispatch mode constraints:

| Mode | Required Dispatch | Reason |
|---|---|---|
| `dense`, `moe` | Slow (`TT_METAL_SLOW_DISPATCH_MODE=1`) | Complex `MemoryConfig` and `MeshMapper` require synchronous host-device protocol |
| `experts` | Fast dispatch OK | Per-expert uploads use simple DMA transfers |
| `embedding`, `lm_head` | Either | Simple tensor uploads with standard configs |

### Mesh Device Setup

The script uses `bh_2d_mesh_device_context` to open the $4 \times 2$ mesh device with `fabric_config = FABRIC_2D`. The fabric router sync timeout is set to 30 seconds (line 657--659) because fabric initialization on Blackhole can be slow. A submesh of shape $(4, 2)$ is created from the mesh device -- the same topology used by the model runtime.

---

## 7.3.2 The Manifest System

Each cache directory (layer, embedding, or lm\_head) contains a `manifest.json` that serves as the metadata registry for all tensors in that directory. The manifest enables **load-time reconstruction** of `OverlappedTensor` views without re-running the weight preparation pipeline.

### Manifest Schema

```json
{
  "version": 1,
  "created_time": "ISO-8601 timestamp",
  "hf_model_name": "deepseek-ai/DeepSeek-V3",
  "hf_state_dict_name": "lazy",
  "device_mesh_shape": [4, 2],
  "layer_idx": 4,
  "layer_type": "moe",
  "fusion_groups": {
    "<group_name>": {
      "tensorbin": "<group_name>.tensorbin",
      "fields": {
        "<field_name>": {
          "tensor_shape": [H, W],
          "shard_shape": [sh, sw],
          "core_range_set": [[[sx, sy], [ex, ey]], ...],
          "dtype": "BFLOAT8_B" | "BFLOAT4_B" | "BFLOAT16" | "UINT32",
          "tile_shape": [th, tw],
          "byte_offset": <int>
        }
      }
    }
  },
  "standalone_tensors": {
    "<name>": "<filename>.tensorbin"
  },
  "routed_experts": {
    "num_experts": 256
  }
}
```

### Incremental Manifest Updates

The `_read_or_create_manifest()` function (lines 898--928 in `prepare_weights.py`) supports incremental cache building:

- If `manifest.json` already exists, it is read and returned for merging.
- Otherwise, a new manifest is created with baseline metadata (version, timestamps, model name, mesh shape, layer index, layer type).

Each `save_*` function (attention, shared, routed) merges its entries into the existing manifest and writes it back. This means the `moe` and `experts` modes can be run in any order -- each adds its section to the manifest.

### Dtype and CoreRangeSet Serialization

The `_DTYPE_TO_STR` and `_STR_TO_DTYPE` mappings (lines 32--38) handle dtype enum serialization:

| ttnn dtype | String |
|---|---|
| `ttnn.DataType.BFLOAT4_B` | `"BFLOAT4_B"` |
| `ttnn.DataType.BFLOAT8_B` | `"BFLOAT8_B"` |
| `ttnn.DataType.UINT32` | `"UINT32"` |
| `ttnn.DataType.BFLOAT16` | `"BFLOAT16"` |

Core range sets are serialized as lists of `[[start_x, start_y], [end_x, end_y]]` pairs by `_core_range_set_to_list()` (lines 831--837) and deserialized by `_core_range_set_from_list()` (lines 840--849).

### Version Compatibility

The manifest `version` field (currently `1`) enables forward-compatible cache invalidation. At load time, loaders check:

```python
if manifest.get("version", 0) > _MANIFEST_VERSION:
    raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
```

---

## 7.3.3 Cache Loading (Deserialization)

Loading a cached layer reverses the serialization. The `load_dense_decoder_layer()` (lines 1238--1311) and `load_moe_decoder_layer()` (lines 1314--1402) functions:

1. Read `manifest.json` and validate the version and layer type.
2. For each fusion group in the manifest, call `ttnn.load_tensor()` to load the `.tensorbin` directly to the mesh device.
3. For each field within the fusion group, call `_overlapped_tensor_from_dict()` to reconstruct an `OverlappedTensor` view using the manifest metadata and the loaded fused tensor:

```python
def _overlapped_tensor_from_dict(fused_tensor, d):
    return OverlappedTensor(
        fused_tensor=fused_tensor,
        tensor_shape=tuple(d["tensor_shape"]),
        shard_shape=tuple(d["shard_shape"]),
        core_range_set=_core_range_set_from_list(d["core_range_set"]),
        dtype=_STR_TO_DTYPE[d["dtype"]],
        tile_shape=tuple(d["tile_shape"]),
        byte_offset=d["byte_offset"],
    )
```

4. Load standalone tensors (`shared_down_proj`, `gate_bias`, routed experts).

Multiple `OverlappedTensor` instances within the same fusion group share the same `fused_tensor` reference, exactly as they would if created fresh by the `BlitzDecodeWeights` methods.

### Routed Expert Loading

Routed experts can be loaded separately via `load_moe_routed_experts()` (lines 1180--1235), which iterates `experts/e_NNN/{gate,up,down}_proj.tensorbin` and calls `ttnn.load_tensor()` for each. This function operates under fast dispatch, and the pre-loaded experts can be passed to `load_moe_decoder_layer()` via the `preloaded_routed_experts` parameter to avoid reloading them in slow dispatch.

This two-phase pattern exists because `setup_fast_dispatch` can only be called once per program:

```
Phase 1 (fast dispatch):  load_moe_routed_experts()  -> MoERoutedExpertWeights
Phase 2 (slow dispatch):  load_moe_decoder_layer(preloaded_routed_experts=...)
```

### Tensor Topology Verification

After loading, the verification mode checks tensor topology placements match the expected patterns for the $4 \times 2$ mesh:

| Tensor Category | Expected Placements | Meaning |
|---|---|---|
| q\_ab\_kv\_a, o\_proj\_gate\_mm\_norms | `[Replicate, Shard(1)]` | Replicated across rows, sharded across columns |
| kv\_b12 | `[Replicate, Shard(0)]` | Replicated across columns, sharded across rows |
| gate\_up, shared\_down\_proj, dense routed | `[Shard(0), Shard(1)]` | Sharded across both mesh dimensions |
| MoE routed experts | `[Replicate]` | Replicated across all devices |

---

## 7.3.4 Embedding Handling

The embedding layer is the simplest weight to prepare. The `prepare_embedding_weights()` function (lines 659--675 in `prepare_weights.py`):

1. Extracts `model.embed_tokens.weight` from the state dict: shape $(129280, 7168)$.
2. Converts to `ttnn.bfloat16` with `ROW_MAJOR_LAYOUT` (not tiled -- embedding lookup is a gather, not a matmul).
3. Places in `DRAM_MEMORY_CONFIG` with `ReplicateTensorToMesh` (every device gets a full copy).

The embedding is replicated because every device needs to look up the same token embedding. The DRAM placement is appropriate because embedding lookup is a single random-access read per token, not a bandwidth-intensive streaming operation. Storage: $129280 \times 7168 \times 2 \approx 1.73$ GB per device.

### Cache Structure

```
embedding/
  embedding.tensorbin    (~1.8 GB)
  manifest.json
```

---

## 7.3.5 LM Head Handling

The LM head is the most bandwidth-intensive single weight in the model: shape $(7168, 129280)$ after transposition. The `prepare_lm_head_weights()` function (lines 732--789) applies several specialized transformations.

### Vocabulary Sharding

The vocabulary dimension (129280) is sharded across all 8 devices:

$$N_\text{per\_device} = \frac{129280}{8} = 16160$$

### L1 WIDTH\_SHARDED Layout

Within each device, the weight is WIDTH\_SHARDED across 101 matmul cores with shard shape $(7168, 160)$:

```python
_LM_HEAD_MATMUL_CORE_GRID = ttnn.CoreRangeSet([
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),  # 100 cores
    ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)), # 1 core
])
```

### LM Head Constants

| Constant | Value | Purpose |
|---|---|---|
| `_LM_HEAD_K` | 7168 | Hidden dimension (contraction dimension) |
| `_LM_HEAD_VOCAB_SIZE` | 129280 | Full vocabulary size |
| `_LM_HEAD_NUM_MATMUL_CORES` | 101 | Cores in the LM head matmul grid |
| `_LM_HEAD_N_PER_CORE` | 160 | Vocab elements per core shard |
| `_LM_HEAD_B_TILE` | $32 \times 32$ | Weight tile shape |
| `_LM_HEAD_A_TILE` | $1 \times 32$ | Activation tile shape (batch=1) |
| `_LM_HEAD_MCAST_CORE` | $(10, 9)$ | Core for activation multicast |

The weight uses `bfloat8_b` dtype, which halves per-core memory compared to bfloat16.

### Final RMSNorm

The final `model.norm.weight` gamma vector (shape $(7168,)$) is prepared alongside the LM head. It is unsqueezed to $(1, 7168)$, converted to `bfloat16` with $1 \times 32$ tile layout, and placed as `HEIGHT_SHARDED` on the mcast core $(10, 9)$ with `ReplicateTensorToMesh`. This core multicasts the normalized hidden state to all 101 matmul cores before the LM head matmul.

### Cache Structure

```
lm_head/
  lm_head.tensorbin      (~920 MB)
  final_norm.tensorbin    (~14 KB)
  manifest.json
```

---

## 7.3.6 Gate Bias and Gate Indices Tensors

These two tensors serve the MoE gating kernel and have a unique physical layout that differs from all other weights.

> **Cross-reference -> Chapter 6, Section 6.1**: The MoE gating kernel that consumes these tensors and the 16-group expert-as-tile-face mapping.

### Gate Bias (`create_gate_bias_tensor`, lines 345--382)

The `e_score_correction_bias` from the HuggingFace state dict is a flat vector of shape $(256,)$. The transformation:

1. **Reshape:** $(256,) \to (16, 16)$ mapping the 256 experts into a $16 \times 16$ grid matching the tile face size.
2. **Transpose:** `torch.transpose(reshaped, 0, 1)` to match the kernel's expected group-major ordering.
3. **Cast:** To `torch.bfloat16`.
4. **Place:** As `HEIGHT_SHARDED` on the single sender core $(10, 9)$ with shard shape $(16, 16)$, tile size $16 \times 16$, and `bfloat16` dtype. Replicated across all devices.

### Gate Indices (`create_gate_indices_tensor`, lines 385--414)

A constant tensor containing $0, 1, \ldots, 255$ reshaped $(16, 16)$ and transposed identically to the gate bias. Same `HEIGHT_SHARDED` layout on the sender core, `uint16` dtype. The transpose matches the gate bias so that top-K score selection yields the corresponding flat expert IDs.

**Important:** The gate indices tensor is **not cached** -- it is a runtime constant created during model initialization by `create_gate_indices_tensor()`. The gate bias, however, is cached as a standalone `.tensorbin` in the layer directory. This distinction exists because the indices are a deterministic constant (values 0--255) that can be regenerated with negligible cost, while the gate bias contains learned parameters from the checkpoint.

---

## 7.3.7 Down Projection Spec (DOWN\_PROJ\_SingleDeviceSpec)

> **Cross-reference → Section 7.2.7**: The full layout, core grid construction (112 cores from 130 minus DRAM workers and phantoms), BFP4 dtype rationale, and standalone justification are detailed in Section 7.2.7.

In the cache, the down projection is serialized as a standalone `.tensorbin` file per layer — it is the only shared-expert weight not part of a fusion group.

---

## 7.3.8 Cache Verification

The `generate_cache.py` script supports a `--verify` mode that validates an existing cache without re-generating it. Verification proceeds in three stages:

### Stage 1: Manifest Integrity

- Parse `manifest.json` and check that the version does not exceed `MANIFEST_VERSION = 1`.
- Verify `layer_type` matches the expected type for the mode.

### Stage 2: File Existence

For each mode, check that all expected files exist and have non-zero size:

| Mode | Required Files |
|------|---------------|
| `dense` | 4 fusion group `.tensorbin` + 4 standalone `.tensorbin` + manifest |
| `moe` | 4 fusion group `.tensorbin` + 2 standalone `.tensorbin` + manifest |
| `experts` | `experts/` with 256 subdirectories, each containing 3 `.tensorbin` (768 total) |
| `embedding` | `embedding.tensorbin` + manifest |
| `lm_head` | `lm_head.tensorbin` + `final_norm.tensorbin` + manifest |

### Stage 3: Device Load (Full Layers Only)

If the cache represents a complete layer, the verification mode:

1. Opens a $4 \times 2$ mesh device.
2. Loads the layer using `load_dense_decoder_layer()` or `load_moe_decoder_layer()`.
3. Verifies every tensor is on device (`storage_type() == DEVICE`).
4. Verifies tensor topology placements match expected patterns.
5. Spot-checks one tensor shape against the manifest.

This device-load verification catches corruption, serialization bugs, and firmware/driver incompatibilities that file-existence checks cannot detect.

---

## 7.3.9 Fusion Group File Sizes

Each fusion group `.tensorbin` file contains the raw bytes of the fused `ttnn.Tensor` as produced by `ttnn.dump_tensor()`, including header metadata and the flat byte buffer with all per-core shards and padding.

### File Sizes (Representative, Single MoE Layer)

| File | Approximate Size | Content |
|---|---|---|
| `q_ab_kv_a.tensorbin` | 45 MB | 96-core q\_ab + 18-core kv\_a fused BFP8 |
| `o_proj_gate_mm_norms.tensorbin` | 55 MB | 112-core o\_proj + 8-core gate\_mm + norms, raw UINT32 |
| `kv_b12.tensorbin` | 18 MB | 128-core kv\_b1 + kv\_b2 fused BFP8 |
| `gate_up.tensorbin` | 9 MB | 128-core gate + up fused BFP4 |
| `shared_down_proj.tensorbin` | 4.5 MB | 112-core WIDTH\_SHARDED BFP4 |
| `gate_bias.tensorbin` | < 1 KB | Single-core 16x16 BFP16 |
| Each expert (3 files) | ~43 MB | DRAM WIDTH\_SHARDED BFP4 per projection |
| All 256 experts | ~11 GB | 768 files total |

The total per-MoE-layer cache size is approximately $11.1$ GB, of which $\sim 99\%$ is routed expert weights.

---

## 7.3.10 Complete Cache Directory Structure

For a full 61-layer DeepSeek V3 model on a $4 \times 2$ mesh:

```
cache_root/
  embedding/
    embedding.tensorbin          (~1.8 GB)
    manifest.json
  lm_head/
    lm_head.tensorbin            (~920 MB)
    final_norm.tensorbin         (~14 KB)
    manifest.json
  layer_000/                     (dense)
    manifest.json
    q_ab_kv_a.tensorbin
    o_proj_gate_mm_norms.tensorbin
    kv_b12.tensorbin
    gate_up.tensorbin
    shared_down_proj.tensorbin
    routed_gate_proj.tensorbin
    routed_up_proj.tensorbin
    routed_down_proj.tensorbin
  layer_001/                     (dense, same structure)
  layer_002/                     (dense, same structure)
  layer_003/                     (moe)
    manifest.json
    q_ab_kv_a.tensorbin
    o_proj_gate_mm_norms.tensorbin
    kv_b12.tensorbin
    gate_up.tensorbin
    shared_down_proj.tensorbin
    gate_bias.tensorbin
    experts/
      e_000/
        gate_proj.tensorbin
        up_proj.tensorbin
        down_proj.tensorbin
      e_001/
        ...
      e_255/
        ...
  layer_004/ ... layer_060/      (moe, same structure as layer_003)
```

### Estimated Sizes

| Component | Count | Per-Item Size (approx) | Total (approx) |
|---|---|---|---|
| Embedding | 1 | 1.8 GB | 1.8 GB |
| LM head + norm | 1 | 920 MB | 920 MB |
| Dense layer (all) | 3 | ~600 MB | 1.8 GB |
| MoE layer (attn+shared) | 58 | ~132 MB | 7.6 GB |
| MoE layer (256 experts) | 58 | ~11 GB | 638 GB |
| **Total** | | | **~650 GB** |

The 256 routed experts dominate the cache size, consistent with the model's 671B total parameter count being dominated by expert weights.

### File Count Summary

| Component | Files per Layer | Layers | Total Files |
|-----------|----------------|--------|-------------|
| Dense layer | 9 (4 fusion + 4 standalone + manifest) | 3 | 27 |
| MoE layer (fusion + shared) | 7 (4 fusion + 2 standalone + manifest) | 58 | 406 |
| MoE layer (experts) | 768 ($256 \times 3$) | 58 | 44,544 |
| Embedding | 2 | 1 | 2 |
| LM head | 3 | 1 | 3 |
| **Total** | | | **44,982** |

The dominant cost is routed expert files: 44,544 out of 44,982 total. This is why the `experts` mode exists separately -- it runs under fast dispatch and can be parallelized across multiple processes.

---

## 7.3.11 End-to-End Timing and Operational Workflow

### Cache Generation Workflow

A typical cache generation workflow for the full model:

```bash
# Step 1: Embedding and LM head (~2 minutes total)
python generate_cache.py --model-path $MODEL --output-path $CACHE --type embedding
python generate_cache.py --model-path $MODEL --output-path $CACHE --type lm_head

# Step 2: Dense layers 0-2 (slow dispatch, ~3 minutes per layer)
for i in 0 1 2; do
  TT_METAL_SLOW_DISPATCH_MODE=1 \
    python generate_cache.py --model-path $MODEL --output-path $CACHE \
      --layer-num $i --type dense
done

# Step 3: MoE attention + shared (slow dispatch, ~2 min/layer, parallelizable)
for i in $(seq 3 60); do
  TT_METAL_SLOW_DISPATCH_MODE=1 \
    python generate_cache.py --model-path $MODEL --output-path $CACHE \
      --layer-num $i --type moe &
done

# Step 4: MoE routed experts (fast dispatch, ~5-10 min/layer, parallelizable)
for i in $(seq 3 60); do
  python generate_cache.py --model-path $MODEL --output-path $CACHE \
    --layer-num $i --type experts &
done

# Step 5: Verify
for i in $(seq 0 60); do
  python generate_cache.py --output-path $CACHE --layer-num $i \
    --type $([ $i -lt 3 ] && echo dense || echo moe) --verify
done
```

Total wall-clock time for the full model on a single machine: ~10--12 hours (dominated by 58 layers of expert weight I/O). With parallelization across 8 machines: ~1.5--2 hours.

Each invocation opens its own mesh device context and creates its own `BlitzDecodeWeights` instance. The `--force` flag is needed to overwrite existing files if re-running after a failure.

### Runtime Loading Sequence

At model initialization time, the runtime calls the `load_*` functions:

| Runtime Function | Source | What It Loads |
|---|---|---|
| `load_embedding_weights()` | `prepare_weights.py:703` | `embedding/embedding.tensorbin` |
| `load_dense_decoder_layer()` | `prepare_weights.py:1238` | `layer_NNN/` (all fusion groups + standalone) |
| `load_moe_routed_experts()` | `prepare_weights.py:1180` | `layer_NNN/experts/e_NNN/*.tensorbin` |
| `load_moe_decoder_layer()` | `prepare_weights.py:1314` | `layer_NNN/` (fusion groups + standalone + experts) |
| `load_lm_head_weights()` | `prepare_weights.py:816` | `lm_head/lm_head.tensorbin + final_norm.tensorbin` |

The typical two-phase loading pattern:

```python
# Phase 1: Load MoE routed experts (fast dispatch for better DMA throughput)
for i in range(3, 61):
    routed[i] = load_moe_routed_experts(cache_path, mesh_device, i)

# Phase 2: Load fusion groups (slow dispatch, with pre-loaded experts)
for i in range(3, 61):
    moe_layers[i] = load_moe_decoder_layer(
        cache_path, mesh_device, i,
        preloaded_routed_experts=routed[i],
    )
```

**DMA transfers per MoE layer:** 6 (4 fused groups + 2 standalone) + 768 (experts) = 774. With fast dispatch and NVMe storage, full model loading takes approximately 3--5 minutes.

The load functions return the same dataclass types as the prepare functions, making the runtime agnostic to whether weights came from online preparation or cached files.

---

## 7.3.12 Design Rationale: Why Per-Layer, Per-Mode Caching?

The per-layer, per-mode cache design has several advantages over a single monolithic cache file:

1. **Incremental generation:** If one layer's cache generation fails (e.g., due to OOM or device timeout), only that layer needs to be re-generated.

2. **Parallelism:** Independent layer directories have no write conflicts, enabling embarrassingly parallel cache generation.

3. **Partial loading:** The runtime can load layers on demand rather than reading the entire cache upfront, enabling pipeline-parallel inference.

4. **Debuggability:** Each `.tensorbin` file can be loaded independently for inspection. If layer 47's MoE gate produces incorrect routing, the operator can verify that layer's `gate_bias.tensorbin` in isolation.

5. **Storage management:** The cache directory tree can be stored on distributed filesystems where different layers reside on different storage nodes.

The trade-off is complexity: managing 61 layer directories with ~775 files each requires careful scripting. The `--verify` mode provides automated validation of the entire cache tree.

---

## 7.3.13 The Cache as a Memory Budget Contract

The weight cache is not merely a serialization format -- it is a contract between the offline preparation system and the runtime kernel. Each `.tensorbin` file encodes not just the weight data but the exact L1 layout: which cores hold which bytes at which offsets. The manifest records this layout so the runtime can reconstruct `OverlappedTensor` views that the kernels depend on.

This contract makes the Blitz decode system possible: kernels are compiled against specific byte offsets, core assignments, and tile formats. The cache guarantees that these assumptions hold when weights are loaded from disk, eliminating any runtime layout negotiation and enabling immediate kernel execution after a single `ttnn.load_tensor` call per fusion group.

| Key Metric | Without Cache | With Cache |
|-----------|--------------|-----------|
| Weight prep per MoE layer | Seconds (host fusion) | 0 (pre-computed) |
| L1 buffer allocations per layer | ~13+ individual | 4 fused + 2 standalone |
| DMA transfers (non-expert) | ~13+ | 6 |
| Runtime layout negotiation | Required | None |

---

| [< 7.2 Blitz Decode Weight Overlapping](./02_blitz_decode_weight_overlapping.md) | [Chapter 8 >](../ch08_next_chapter/index.md) |
|:---|---:|
