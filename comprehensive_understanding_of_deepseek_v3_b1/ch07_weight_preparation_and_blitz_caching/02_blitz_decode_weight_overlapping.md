# 7.2 Blitz Decode Weight Overlapping

This section explains the core innovation behind the Blitz decode weight system: fusing multiple weight tensors into a single device buffer so they share the same L1 base address on each core. We cover the `OverlappedTensor` abstraction, the `BlitzDecodeWeights` class, all four overlap specification classes, the tile-reshape mechanism for reconciling mismatched shard widths, the weight shuffles that prepare data for hardware-optimal access patterns, and the BFP8 byte-level encoding contract.

> **Cross-reference:** The preparation pipeline that feeds tensors into this system is covered in [Section 7.1](./01_weight_preparation_pipeline.md). The cache serialization that persists and reloads fused tensors is in [Section 7.3](./03_cache_serialization_and_special_tensors.md). For how kernels consume these fused buffers, see Chapter 5 (MLA) and Chapter 6 (MoE).

Source: `models/demos/deepseek_v3_b1/blitz_decode_weights.py` (1855 lines)

---

## 7.2.1 Why Overlapping Matters: The L1 Budget Arithmetic

Each Tensix core on Blackhole has approximately 1.5 MB of L1 SRAM. A single decode step through one DeepSeek V3 layer touches at least 10 distinct weight tensors across attention, gating, and FFN stages. Without fusion, each tensor requires its own L1 allocation with tile-alignment padding, buffer management overhead, and a separate DMA transfer from DRAM or host. The overhead compounds:

- **Alignment waste:** Each L1 buffer is aligned to tile boundaries. For $32 \times 32$ BFP8 tiles (1088 bytes), alignment can waste up to 1087 bytes per buffer.
- **Buffer count limits:** The number of concurrent L1 buffers is limited by the CBM (Circular Buffer Manager). Fewer buffers means simpler scheduling.
- **Load latency:** Each `ttnn.load_tensor` call has fixed overhead. Fusing $N$ tensors into one buffer reduces load calls by a factor of $N$.

The overlapping system addresses all three: one allocation, one alignment penalty, one load call per fusion group. The trade-off is that the fused buffer must be sized to the maximum shard across all sub-tensors, with smaller shards zero-padded. In practice, this overhead is small because the spec classes are carefully designed to group tensors with similar per-core footprints.

> **Cross-reference -> Chapter 2**: CB descriptor creation from overlapped tensors via `cb_descriptor_from_overlapped_tensor`, and CB reconfiguration mechanics.

---

## 7.2.2 `OverlappedTensor` Dataclass

The `OverlappedTensor` dataclass (lines 600--621) is the fundamental abstraction for a sub-tensor view into a fused buffer.

> **Cross-reference → Chapter 1, Section 3.2**: The full 8-field dataclass definition (`fused_tensor`, `tensor_shape`, `shard_shape`, `core_range_set`, `dtype`, `tile_shape`, `byte_offset`, `total_size`) and general field descriptions.

Ch7-specific semantics:

- **`fused_tensor`**: The Python `id(fused_tensor)` is used during serialization to detect which fields belong to the same fusion group (see Section 7.3) — fields sharing the same `id()` are written as one `.tensorbin`.

- **`byte_offset`**: Accumulates sizes of preceding sub-tensors on the same core. For the first sub-tensor in a region, this is 0.

- **`tile_shape`**: Enables mixed tile shapes within one fused buffer (e.g., $32 \times 32$ for weights and $1 \times 32$ for norms in the `o_proj_gate_mm_norms` group).

---

## 7.2.3 `BlitzDecodeWeights` Class

The `BlitzDecodeWeights` class (lines 623--1855) is the engine that converts raw PyTorch tensors into fused device-resident `OverlappedTensor` groups. It is instantiated once per model with a device reference:

```python
class BlitzDecodeWeights:
    def __init__(self, device) -> None:
        self._device = device
        num_devices = device.get_num_devices()
        if num_devices == 1:
            self.mla_tp = 1
            self.moe_tp = 1
        else:
            self.mla_tp = 2   # MLA weights TP-sharded across 2 columns
            self.moe_tp = 8   # MoE weights TP-sharded across all 8 devices
```

### Public Methods

| Method | Fusion Group | Returns | Sharding |
|--------|-------------|---------|----------|
| `get_tt_q_ab_proj_and_kv_a_proj_weights()` | `q_ab_kv_a` | 3 `OverlappedTensor` | WIDTH_SHARDED |
| `get_tt_o_proj_and_gate_mm_weights()` | `o_proj_gate_mm_norms` | 6 `OverlappedTensor` | WIDTH_SHARDED (raw UINT32) |
| `get_tt_kv_b12_proj_weights()` | `kv_b12` | 2 `OverlappedTensor` | HEIGHT_SHARDED |
| `get_tt_moe_shared_expert_weights()` | `gate_up` + standalone | 2 `OverlappedTensor` + 1 `ttnn.Tensor` | HEIGHT_SHARDED + WIDTH_SHARDED |
| `get_tt_moe_routed_expert_weights()` | (no overlap) | 3 lists of `ttnn.Tensor` | DRAM WIDTH_SHARDED |
| `get_tt_mlp_shared_expert_weights()` | delegates to moe_shared | same as above | same |
| `get_tt_mlp_routed_expert_weights()` | (no overlap) | 3 `ttnn.Tensor` | DRAM WIDTH_SHARDED |

---

## 7.2.4 Overlap Spec 1: `QAB_KVA_PROJ_SingleDeviceOverlapSpec`

**Lines 83--194.** Fuses three MLA input projections into one WIDTH_SHARDED tensor.

### Layout

```
+-- Top region: 96 cores (12x8) --+
| q_a_proj packed (3584, 3072)     |  <- (7168/2, 1536*2), packed H/2 x 2W
| q_b_proj shuffled (1536, 12288)  |  <- Qnope/Qrope interleaved columns
+----------------------------------+
+-- Bottom region: 18 cores (9x2 at rows 8-9) --+
| kv_a_proj reordered (7168, 576)                |  <- shard-reordered, zero-padded
+------------------------------------------------+
```

### Configuration Constants

| Property | Value | Derivation |
|----------|-------|------------|
| `q_a_proj_shape` | $(7168, 1536)$ | $d_\text{model} \times d_\text{q\_latent}$ |
| `q_b_proj_shape` | $(1536, 12288)$ | Per-TP: $1536 \times (64 \times 128 + 64 \times 64)$ |
| `kv_a_proj_shape` | $(7168, 576)$ | $d_\text{model} \times (d_\text{kv\_lora} + d_\text{rope})$ |
| `packed_h` | $3584$ | $7168 / 2$ (q_a packing halves height) |
| `packed_w` | $3072$ | $1536 \times 2$ (q_a packing doubles width) |

### Weight Shuffles

**`shuffle_q_a()`** (line 150): Packs $W_{q\_a}$ from $(H, W)$ to $(H/2, 2W)$ by interleaving the top and bottom halves:

```python
def shuffle_q_a(weights):
    H, W = weights.shape
    return weights.reshape(2, H//2, W).permute(1, 0, 2).reshape(H//2, 2*W)
```

This halves the number of shard rows, making room for $W_{q\_b}$ in the vertical stitch, and enables the packed matmul kernel to process two K-partitions per row without cross-core communication.

**`shuffle_q_b()`** (lines 155--164): Calls `shuffle_weights_for_interleaved_qnope_qrope()` (lines 27--80), which reorders columns so the matmul2 output is distributed in the interleaved layout the downstream grid expects. Each group of 8 heads maps to one grid row:

```
[QNOPE_heads_0:8 (8*128=1024 cols) | QROPE_heads_0:8 (8*64=512 cols)]
[QNOPE_heads_8:16                  | QROPE_heads_8:16                ]
...
[QNOPE_heads_56:63                 | QROPE_heads_56:63               ]
```

Total per row group: $1024 + 512 = 1536$ elements across 12 cores (8 Qnope cores at 128 each + 4 Qrope cores at 128 each with 2 heads per core). With $64/8 = 8$ row groups: $8 \times 1536 = 12288$.

**`shuffle_kv_a()`** (lines 186--191): Reorders the 18 kv_a shards according to `kv_a_proj_shard_order = (0,1,2,3,4,5,6,7,16,8,9,10,11,12,13,14,15,17)`, which places the Knope-rope boundary shard (shard 16) between shards 7 and 8 so the physical core layout matches the KV cache branch kernel's expected split between the 512-dim nope and 64-dim rope components.

### Multi-Device Distribution

For `mla_tp = 2`, the method builds per-TP fused tensors independently (each with its own q_b slice but shared q_a and kv_a), concatenates them along the width dimension, and distributes via `ShardTensor2dMesh(dims=(None, 1))` -- sharding across mesh columns while replicating across rows.

---

## 7.2.5 Overlap Spec 2: `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec`

**Lines 197--339.** The most heterogeneous fusion group, packing 6 sub-tensors of 3 different dtypes and 2 different tile shapes into a single UINT32 raw-byte buffer.

### Layout

| Sub-tensor | Shape | dtype | Tile | Cores | Bytes/Shard |
|------------|-------|-------|------|-------|-------------|
| `o_proj` | $(8192, 7168)$ | BFP8 | $32 \times 32$ | 112 cores | $512 \times 1088 = 557{,}056$ |
| `gate_mm` | $(7168, 256)$ | BFP16 | $32 \times 32$ | 8 cores (col 12, rows 0--7) | $224 \times 2048 = 458{,}752$ |
| `attn_norm` | $(1, 7168)$ | BF16 | $1 \times 32$ | core $(12, 9)$ | $14{,}336$ |
| `q_norm` | $(1, 1536)$ | BF16 | $1 \times 32$ | core $(12, 9)$ | $3{,}072$ |
| `ffn_norm` | $(1, 7168)$ | BF16 | $1 \times 32$ | core $(12, 9)$ | $14{,}336$ |
| `kv_norm` | $(1, 512)$ | BF16 | $1 \times 32$ | core $(0, 8)$ | $1{,}024$ |

The o_proj shard calculation: shape $(8192, 7168)$ WIDTH_SHARDED on 112 cores gives shard $(8192, 64)$. Tiles per shard = $(8192/32) \times (64/32) = 256 \times 2 = 512$ tiles. At 1088 bytes/tile: $512 \times 1088 = 557{,}056$ bytes.

### The Raw-Byte Approach

Because the sub-tensors use different dtypes and tile shapes, the fused tensor cannot be a standard typed tensor. Instead, `get_tt_o_proj_and_gate_mm_weights()` (lines 826--1045):

1. Manually tilizes and packs each sub-tensor into raw bytes using `_tilize_and_pack_bfp8()`, `_tilize_and_pack_bfloat16()`, or `_pack_bfloat16_1x32()`
2. Concatenates the packed bytes per core, zero-padding each shard to `max_shard_bytes`
3. Wraps the result as a UINT32 tensor with ROW_MAJOR_LAYOUT
4. Places it on device as WIDTH_SHARDED

The `max_shard_bytes` property (lines 329--336) ensures every core gets the same shard size:

$$\text{max\_shard\_bytes} = \max(557056, 458752, 31744, 1024) = 557{,}056 \text{ bytes}$$

### Gamma Core Packing

Three RMSNorm gammas are packed back-to-back on core $(12, 9)$:

| Offset | Gamma | Size |
|--------|-------|------|
| $0$ | `attn_norm` $(1, 7168)$ | $7168 \times 2 = 14{,}336$ bytes |
| $14{,}336$ | `q_norm` $(1, 1536)$ | $1536 \times 2 = 3{,}072$ bytes |
| $17{,}408$ | `ffn_norm` $(1, 7168)$ | $7168 \times 2 = 14{,}336$ bytes |

Total: $31{,}744$ bytes, zero-padded to $557{,}056$.

The `kv_norm` gamma occupies a separate dedicated core $(0, 8)$ at offset 0, because it is consumed by the KV cache branch which runs on a disjoint core set from the Q-path.

---

## 7.2.6 Overlap Spec 3: `KVB12_PROJ_SingleDeviceOverlapSpec`

**Lines 342--423.** Fuses the two post-SDPA weight matrices into one HEIGHT_SHARDED tensor.

### Layout

```
+-- kv_b1 region: 64 cores (8x8 Qnope grid) --+
| kv_b1_proj (8192, 512) BFP8, shard (128, 512) |
+------------------------------------------------+
+-- kv_b2 region: 64 cores (5x8 + 12x2) --+
| kv_b2_proj (512, 8192) BFP8              |
| tile-reshaped to shard (128, 512)         |
+-------------------------------------------+
```

Both regions use the same shard shape $(128, 512)$ for a uniform HEIGHT_SHARDED layout across 128 cores.

### The kv_b2 Tile-Reshape

$W_{kv\_b2}$ has logical shape $(512, 8192)$ with per-core column slices of $(512, 128)$ -- a $16 \times 4$ grid of $32 \times 32$ tiles. The `shuffle_kv_b2()` method (lines 400--420) rearranges these into $(128, 512)$ shards -- a $4 \times 16$ tile grid:

```python
def shuffle_kv_b2(self, weights: torch.Tensor) -> torch.Tensor:
    per_core = weights.reshape(kv_dim, n_cores, head_dim).permute(1, 0, 2)
    tiles = per_core.reshape(n_cores, k_tiles, t, n_tiles, t)
    tiles = tiles.permute(0, 1, 3, 2, 4).reshape(n_cores, k_tiles * n_tiles, t, t)
    tiles = tiles.reshape(n_cores, head_dim // t, kv_dim // t, t, t)
    return tiles.permute(0, 1, 3, 2, 4).reshape(n_cores * head_dim, kv_dim)
```

The critical invariant: per-tile element data is preserved exactly, so BFP8 encoding produces byte-identical tiles before and after the tile-level rearrangement.

> **Cross-reference -> Chapter 5, Section 5.4**: How the post-SDPA kernels consume kv_b1 (matmul3) and kv_b2 (matmul5) from this fused buffer.

---

## 7.2.7 Overlap Spec 4: `GATE_UP_PROJ_SingleDeviceOverlapSpec`

**Lines 426--553.** Fuses the shared expert gate and up projections into a HEIGHT_SHARDED tensor using block-sharding.

### Block-Shard Decomposition

Both gate and up projections have logical shape $(7168, 256)$. With `k_parallel = 8` and `n_parallel = 8`, each is decomposed into $64$ shards of $(896, 32)$:

$$\text{shard}_{(i,j)} = W[i \cdot 896 : (i+1) \cdot 896,\; j \cdot 32 : (j+1) \cdot 32]$$

### Core Assignment

Gate shards map to the 64 A compute cores and up shards to the 64 B compute cores of the shared expert dual-matmul layout:

> **Cross-reference -> Chapter 4, Section 4.3**: The shared expert fused-op core grid layout showing A/B core assignment.

### `reshuffle_block_to_height_sharded()` (lines 529--550)

The block decomposition produces shards in $(k\_idx, n\_idx)$ order, but HEIGHT_SHARDED placement assigns shards to cores in CoreRangeSet enumeration order. The `_crs_shard_permutation()` helper (lines 510--527) computes a permutation that maps enumeration order to global row-major core order:

```python
@staticmethod
def _crs_shard_permutation(core_range_set):
    crs_cores = []
    for cr in core_range_set.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                crs_cores.append((x, y))
    sorted_cores = sorted(crs_cores, key=lambda c: (c[1], c[0]))  # Row-major
    core_to_sorted_idx = {c: i for i, c in enumerate(sorted_cores)}
    return tuple(core_to_sorted_idx[c] for c in crs_cores)
```

The reshuffled blocks are concatenated into a single $(57344, 32)$ stacked tensor per sub-weight, then the two are concatenated vertically for the fused HEIGHT_SHARDED buffer of $(114688, 32)$.

### Down Projection (Standalone)

The `DOWN_PROJ_SingleDeviceSpec` (lines 556--597) handles the down projection as a separate WIDTH_SHARDED tensor on 112 matmul cores. Per-device layout: $(256, 7168)$ with shard $(256, 64)$ in BFP4. The 112-core grid is computed by `build_matmul_core_grid()`, which excludes 8 DRAM worker positions at fixed coordinates and column 12 (phantom/mcast cores) from the $13 \times 10$ grid. The down projection cannot be fused with gate/up because it runs on a different core grid (112 matmul cores vs 128 dual-matmul cores) and at a different stage of the computation pipeline.

---

## 7.2.8 Tile-Reshape for Mixed Shard Widths

The `_stitch_width_sharded()` static method (lines 1749--1822) is the core mechanism for combining two WIDTH_SHARDED tensors with different shard widths into a single fused tensor:

### Algorithm

1. **Determine target width**: Use the narrower shard width as the fused shard width
2. **Tile-reshape the wider tensor**: For each core, take the wider shard and rearrange its tiles from a $(H_\text{wide}, W_\text{wide})$ grid to a $(H_\text{reshaped}, W_\text{target})$ grid, where $H_\text{reshaped} = H_\text{wide} \times W_\text{wide} / W_\text{target}$
3. **Vertical concatenation**: Stack the narrow shard above the reshaped wide shard for each core

The tile-reshape is implemented by `_tile_reshape()` (lines 1824--1855):

```python
@staticmethod
def _tile_reshape(tensor, src_shape, dst_shape, tile_h=32, tile_w=32):
    src_tr, src_tc = src_h // tile_h, src_w // tile_w
    dst_tr, dst_tc = dst_h // tile_h, dst_w // tile_w
    tiles = tensor.reshape(src_tr, tile_h, src_tc, tile_w).permute(0, 2, 1, 3)
    tiles = tiles.reshape(-1, tile_h, tile_w).reshape(dst_tr, dst_tc, tile_h, tile_w)
    return tiles.permute(0, 2, 1, 3).reshape(dst_h, dst_w)
```

The critical invariant: **every tile's internal data is unchanged**. Only the grid layout changes. This means the BFP8/BFP4/BFP16 encoding of each tile is identical before and after reshape -- no re-quantization is needed.

---

## 7.2.9 Weight Shuffle Summary

| Shuffle | Method | Purpose |
|---------|--------|---------|
| q_a pack | `shuffle_q_a()` | $(H, W) \to (H/2, 2W)$: halve rows to fit with q_b |
| q_b interleave | `shuffle_q_b()` | Reorder columns for Qnope/Qrope grid layout |
| kv_a shard reorder | `shuffle_kv_a()` | Place Knope-rope boundary shard on correct core |
| kv_b2 tile-reshape | `shuffle_kv_b2()` | $(512, 128) \to (128, 512)$ per core for uniform HEIGHT_SHARDED |
| block-to-height | `reshuffle_block_to_height_sharded()` | Block-sharded $(K, N) \to$ HEIGHT_SHARDED stacked form |
| DRAM tile shuffle | `_shuffle_dram_tiles()` | Row-major $\to$ column-major tile order within DRAM bank shards |

---

## 7.2.10 BFP8 Tile Encoding: The Byte-Level Contract

The overlapping system's correctness depends on the byte-level format of each tile matching what the hardware expects. The `_tilize_and_pack_bfp8` method (lines 1616--1700) implements the BFP8_b encoding specification.

### BFP8_b Tile Format

Each $32 \times 32$ BFP8_b tile occupies exactly 1088 bytes:

| Section | Content | Words | Bytes |
|---------|---------|-------|-------|
| Exponents | 64 shared exponents (one per 16-element block) | 16 `uint32` | 64 |
| Mantissas | 1024 packed mantissa bytes | 256 `uint32` | 1024 |
| **Total** | | **272 `uint32`** | **1088** |

### Face Ordering

The 1024 elements within a tile are stored in face order, not row-major:

```
Face 0: rows 0-15,  cols 0-15   (256 elements)
Face 1: rows 0-15,  cols 16-31  (256 elements)
Face 2: rows 16-31, cols 0-15   (256 elements)
Face 3: rows 16-31, cols 16-31  (256 elements)
```

Within each face, elements are stored row-major. Each row of 16 elements forms one BFP8 block that shares a single exponent.

### Per-Element Encoding

For each block of 16 elements:
1. **Shared exponent:** $e_\text{shared} = \max(e_0, e_1, \ldots, e_{15})$ where $e_i$ is the IEEE-754 float32 biased exponent
2. **Per-element mantissa:** The 24-bit explicit mantissa is right-shifted by $\Delta e = e_\text{shared} - e_i$, then further by 17 bits ($24 - 7$) to yield a 7-bit result, with round-to-nearest-even
3. **Packed byte:** $\text{sign}(1\text{ bit}) \| \text{mantissa}(7\text{ bits})$

### BFP16 and BF16 Variants

| Method | Tile Size | Bytes/Tile | Used By |
|--------|-----------|-----------|---------|
| `_tilize_and_pack_bfloat16` (line 1702) | $32 \times 32$ | 2048 | gate_mm |
| `_pack_bfloat16_1x32` (line 1736) | $1 \times 32$ | 64 | Norm gammas |

Both use bfloat16 encoding (top 16 bits of float32). The $32 \times 32$ variant uses the same face ordering as BFP8.

---

## 7.2.11 Multi-Device Distribution Patterns

Each fusion group uses a specific mesh distribution strategy:

| Fusion Group | Mesh Mapper | `dims` | Rationale |
|---|---|---|---|
| `q_ab_kv_a` | `ShardTensor2dMesh` | `(None, 1)` | q_b head-sharded across columns; q_a/kv_a replicated |
| `o_proj_gate_mm_norms` | `ShardTensor2dMesh` | `(None, 1)` | o_proj head-sharded across columns |
| `kv_b12` | `ShardTensor2dMesh` | `(None, 0)` | kv_b1/b2 head-sharded across columns (HEIGHT dim) |
| `gate_up` | `ShardTensor2dMesh` | `(0, 1)` | Sharded across both mesh dimensions (full 8-way TP) |
| `down_proj` | `ShardTensor2dMesh` | `(0, 1)` | Sharded across both mesh dimensions |
| Routed experts (MoE) | `ReplicateTensorToMesh` | N/A | All 256 experts replicated to all devices |
| Routed experts (dense) | `ShardTensor2dMesh` | `(0, 1)` | One expert per device |

MoE routed experts are replicated because routing decisions determine which expert runs on which device -- every device must have access to all 256 experts. They reside in DRAM, not L1, so the memory cost is acceptable.

---

## 7.2.12 Detailed Walkthrough: q_ab_kv_a Fusion Assembly

To make the overlapping mechanics concrete, this section traces the full `get_tt_q_ab_proj_and_kv_a_proj_weights()` call for a single device (`mla_tp = 1`).

### Input Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `q_a_proj_weights` | $(7168, 1536)$ | Raw Q compression projection |
| `q_b_proj_weights` | $(1536, 12288)$ | Raw Q decompression projection |
| `kv_a_proj_weights` | $(7168, 576)$ | Raw KV compression projection |

### Step 1: Pack q_a

`shuffle_q_a()` interleaves the two halves: $(7168, 1536) \to (3584, 3072)$.

### Step 2: Shuffle q_b

`shuffle_q_b()` reorders the 12288 output columns into row-grouped Qnope/Qrope blocks. Result: $(1536, 12288)$, same shape but reordered columns.

### Step 3: Reorder kv_a shards

`shuffle_kv_a()` applies `kv_a_proj_shard_order` to the 18 shard columns ($576 / 18 = 32$ each). Shard 16 (the Knope-rope boundary) is moved to position 8.

### Step 4: Stitch q_ab

`_stitch_width_sharded(packed, shuffled, 96)` fuses packed q_a and shuffled q_b:

- q_a packed shard width: $3072 / 96 = 32$
- q_b shuffled shard width: $12288 / 96 = 128$
- Target width: $\min(32, 128) = 32$
- q_b tile-reshaped: $(1536, 128) \to (6144, 32)$ per core (same 192 tiles, from $48 \times 4$ to $192 \times 1$ grid)
- Fused shard height: $3584 + 6144 = 9728$

### Step 5: Pad and concatenate kv_a

kv_a has shard width $576 / 18 = 32$ (matches target). Zero-padded from height 7168 to 9728. Result after concatenation: $(9728, 3648)$ -- 114 cores $\times$ 32 columns each.

### Step 6: Device placement

The combined tensor is placed as WIDTH_SHARDED with shard $(9728, 32)$ across the union of `q_ab_core_range_set` (96 cores) and `kv_core_range_set` (18 cores).

### Step 7: Build OverlappedTensor views

Three views are returned, all pointing to the same fused tensor:

| View | `tensor_shape` | `shard_shape` | `core_range_set` | `byte_offset` |
|------|---------------|---------------|-------------------|---------------|
| q_a_proj | $(3584, 3072)$ | $(3584, 32)$ | 96 cores | 0 |
| q_b_proj | $(1536, 12288)$ | $(1536, 128)$ | 96 cores | $112 \times 1088 = 121{,}856$ bytes |
| kv_a_proj | $(7168, 576)$ | $(7168, 32)$ | 18 cores | 0 |

Note: q_b's `shard_shape` records the *logical* shard dimensions $(1536, 128)$, not the tile-reshaped physical dimensions $(6144, 32)$. The kernel uses the `byte_offset` and `tile_shape` to navigate the physical layout.

---

## 7.2.13 L1 Memory Budget Analysis

The following table shows per-core shard sizes for each fusion group:

| Fusion Group | Cores | Shard Bytes | Derivation |
|---|---|---|---|
| `q_ab_kv_a` (q_ab cores) | 96 | $330{,}752$ (~323 KB) | $(3584/32 + 6144/32) \times 1 \times 1088 = 304 \times 1088$ |
| `q_ab_kv_a` (kv cores) | 18 | $330{,}752$ (zero-padded) | $224 \times 1088 = 243{,}712$ active, rest padding |
| `o_proj_gate_mm_norms` (o_proj) | 112 | $557{,}056$ (~544 KB) | $256 \times 2 \times 1088 = 512 \times 1088$ |
| `o_proj_gate_mm_norms` (gate_mm) | 8 | $458{,}752$ (~448 KB) | $224 \times 1 \times 2048$ |
| `o_proj_gate_mm_norms` (gamma) | 1 | $31{,}744$ (~31 KB) | $14336 + 3072 + 14336$ |
| `o_proj_gate_mm_norms` (kv_norm) | 1 | $1{,}024$ (~1 KB) | $512 \times 2$ |
| `kv_b12` | 128 | $69{,}632$ (~68 KB) | $4 \times 16 \times 1088$ |
| `gate_up` | 128 | $16{,}128$ (~16 KB) | $28 \times 1 \times 576$ |

All cores in the `o_proj_gate_mm_norms` group are zero-padded to `max_shard_bytes = 557,056`. The o_proj cores carry the heaviest burden at ~544 KB, which is roughly one-third of the available 1.5 MB L1 per core, leaving room for activation buffers and circular buffers.

### What Overlapping Saves

Without overlapping, the same weights would require 13 separate L1 buffer allocations (not counting down_proj and experts) versus 4 fused allocations. Beyond the alignment savings (up to 1087 bytes wasted per buffer), the reduced buffer count simplifies the circular buffer manager's scheduling and reduces the number of `ttnn.load_tensor` calls at model initialization by $3\times$.

---

## 7.2.14 DRAM Tile Reordering for Routed Experts

Routed expert weights (256 per MoE layer) live in DRAM, not L1. The `_shuffle_dram_tiles` method (lines 1555--1613) reorders tiles within each bank shard from row-major to column-major order.

> **Cross-reference → Chapter 6, Section 6.4.3**: The column-major reordering algorithm, permutation formula, and performance rationale (K tiles contiguous for streaming reads).

Expert weights are $N$-padded to align with DRAM bank count $\times 32$ for uniform bank utilization.

---

## 7.2.15 Design Rationale: Why These Specific Fusion Groups?

The grouping of tensors into fusion groups is not arbitrary -- it follows the execution order of the Blitz decode pipeline:

1. **`q_ab_kv_a`**: All three tensors are consumed by the pre-SDPA fused op's first phase (matmul1 for q_a, matmul2 for q_b, and the kv_a branch). They execute on overlapping core sets within the same op dispatch.

2. **`o_proj_gate_mm_norms`**: The o_proj is consumed by the post-SDPA output projection, the norms by various RMSNorm stages, and gate_mm by the MoE gating. These execute within the same inter-layer transition: attention output projection, layer norms, and MoE gate are tightly sequenced.

3. **`kv_b12`**: Both kv_b matrices are consumed by the post-SDPA fused op (matmul3 for kv_b1 absorption, matmul5 for kv_b2 value reconstruction). They share the same op dispatch context.

4. **`gate_up`**: The gate and up projections of the shared expert run simultaneously on disjoint core sets within the `shared_expert` fused op.

This alignment ensures that all tensors in a fusion group are "live" (needed in L1) at the same time. No fusion group contains tensors from different pipeline stages, which would waste L1 capacity by keeping unused weights resident.

---

## 7.2.16 From Fusion to Kernel: How Kernels Consume Overlapped Tensors

The `OverlappedTensor` is not consumed by ttnn ops directly. Instead, the model's fused ops extract the metadata at graph-build time:

1. **`fused_tensor`**: Passed to the op as the L1 buffer reference
2. **`byte_offset`**: Compiled into the kernel as a constant offset from the buffer base address
3. **`shard_shape`** and **`tile_shape`**: Used to configure the matmul/reduce op's CB parameters
4. **`core_range_set`**: Determines which cores the op dispatches to

> **Cross-reference -> Chapter 2**: How `cb_descriptor_from_overlapped_tensor` creates the CB descriptor from the offset, shard shape, and tile shape.

This means the overhead of overlapping is entirely at weight-preparation time. At runtime, the kernel sees a single contiguous L1 buffer and reads sub-tensors at fixed offsets -- exactly as if they were separately allocated, but with the memory savings of shared allocation.

---

| [< 7.1 Weight Preparation Pipeline](./01_weight_preparation_pipeline.md) | [7.3 Cache Serialization and Special Tensors >](./03_cache_serialization_and_special_tensors.md) |
|:---|---:|
