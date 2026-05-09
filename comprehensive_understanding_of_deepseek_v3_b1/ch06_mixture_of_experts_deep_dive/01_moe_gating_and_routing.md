# 6.1 MoE Gating and Routing

This section traces the path a token takes from raw hidden state to a set of 8 expert indices and normalized scores. Rather than describing the gating mechanism top-down, we follow the data step by step -- from the initial $[1, 7168]$ activation through the gate matmul, the gather to the sender core, and the hierarchical selection logic inside `deepseek_moe_gate`.

> **Cross-reference:** Chapter 1 Section 3.1.3 introduces the dense-vs-MoE layer split. Chapter 3 (Section 3.1) catalogs the `deepseek_moe_gate` micro-op interface. Chapter 4 (Section 4.3) provides the fused-op composition tree for `moe`. This section goes deeper into the mathematical algorithm and its hardware mapping.

---

## 6.1.1 Dense vs. MoE Layers: Where Routing Begins

DeepSeek V3 has 61 decoder layers (indices 0--60). The first three layers use dense MLP (a single "expert" with no routing), while layers 3 through 60 use the full MoE pathway with 256 routed experts plus one shared expert.

This is controlled by the `first_k_dense_replace = 3` parameter (`FIRST_K_DENSE_REPLACE`). In the weight preparation code (`prepare_weights.py`, lines 529--570), MoE layers load 256 expert weight sets from state dict keys of the form `mlp.experts.{e}.gate_proj.weight`, while dense layers load a single set from `mlp.gate_proj.weight`.

| Layer Index | Type | Gate Network | Routed Experts |
|---|---|---|---|
| 0 -- 2 | Dense | None (`enable_routing=False`) | 1 per device (acts as dense MLP) |
| 3 -- 60 | MoE | Active (`enable_routing=True`) | 256 across all devices |

The routing flag propagates into the kernel: `_MoeRoutedExpertContext` carries `enable_routing: bool` (line 206, `moe/op.py`), and when routing is disabled, all gate-related CB indices and compile-time args are set to 0, causing the kernel to skip gate matmul, gate gather, gate computation, and index/scale mcast stages entirely (lines 866--895 of `moe/op.py`).

---

## 6.1.2 The 16x16 Tile Layout: 256 Experts as a Single Face

The 256-expert space is not treated as a flat vector. Instead, it is organized as a $16 \times 16$ matrix -- 16 groups of 16 experts each. This layout maps directly onto a single Tensix "face" (16 rows by 16 columns), enabling the entire gating computation to run on one core with efficient tile-level operations.

```
                     Expert Index within Group
                     0    1    2   ...   15
              +-----+----+----+--------+----+
  Group  0    | e0  | e1 | e2 |  ...   |e15 |
  Group  1    | e16 |e17 |e18 |  ...   |e31 |
  Group  2    | e32 |e33 |e34 |  ...   |e47 |
    ...       |     |    |    |  ...   |    |
  Group 15    |e240 |e241|e242|  ...   |e255|
              +-----+----+----+--------+----+
              Shape: [16, 16]  =  1 tile (face)
```

This mapping is fundamental to the efficiency of the hierarchical selection algorithm: sorting within groups becomes sorting within tile rows, and group-level ranking becomes inter-row operations -- both of which the TRISC compute engine can execute without data movement.

The layout is established during weight preparation. The `e_score_correction_bias` from the state dict (shape $(256,)$) is reshaped to $(16, 16)$ in `prepare_weights.py` (the `create_gate_bias_tensor` function, line 352). The reshape involves:

```python
# (256,) -> (1, 8, 32) -> (16, 16) then transpose to match op layout
reshaped = raw.reshape(1, 8, 32).reshape(16, 16)
transposed = torch.transpose(reshaped, 0, 1).contiguous().to(torch.bfloat16)
```

The transpose ensures that the physical tile layout matches the kernel's expected group-major ordering. A companion indices tensor is also pre-built with shape $(16, 16)$, where each element contains the flat expert index (0--255). When rows are later selected and flattened, these indices travel alongside the scores so that the final top-8 output can report which flat expert indices were chosen.

---

## 6.1.3 Step 1: Gate Matmul -- From Hidden State to 256 Scores

The first step in routing is a standard matrix multiplication that projects the $[1, 7168]$ hidden state to $[1, 256]$ routing logits, followed by a fused sigmoid activation.

**Dimensions:**

$$\text{logits}_{[1, 256]} = \text{input}_{[1, 7168]} \times W_{\text{gate}}^{[7168, 256]}$$

$$\text{scores}_{[1, 256]} = \sigma(\text{logits}_{[1, 256]})$$

**Hardware mapping:** The gate matmul is an SRAM matmul (not a DRAM streaming matmul) because the gate weight matrix $[7168, 256]$ is small enough to reside in L1. It runs on 8 cores -- the gate matmul weight tensor is an `OverlappedTensor` with `WIDTH_SHARDED` layout across these cores. The `setup_sram_matmul` function (`moe/op.py`, line 2930) configures this with `fused_activation=ACTIVATION_SIGMOID` (value 1), so sigmoid is applied immediately after accumulation without a separate pass.

The matmul setup (`moe/op.py`, lines 1014--1021):

```python
gate_mm_params = MoeOp.setup_sram_matmul(
    in0_cb=gate_mm_input_cb,
    in1_cb=gate_mm_weights_cb,
    out_cb=gate_mm_output_cb,
    weights_overlapped=gate_mm_weights_tensor,
    k_num_tiles=num_tiles_k,            # 7168 / 32 = 224 tiles
    fused_activation=MoeRoutedExpertOp.ACTIVATION_SIGMOID,
)
```

**Key design decisions:**

1. **Fused sigmoid**: The `ACTIVATION_SIGMOID` constant (value 1) tells the compute kernel to apply $\sigma$ before writing the output tile. Because sigmoid is applied during the matmul, the downstream gate micro-op receives post-sigmoid scores and sets `enable_sigmoid=False`.

2. **8-core parallelism**: The 256-element output is WIDTH_SHARDED across the gate matmul cores (8 optimal DRAM bank cores). Each core computes $[1, 7168] \times [7168, 32] \rightarrow [1, 32]$, producing 32 expert scores (one 1x32 tile per core).

3. **Input reuse**: The gate matmul input CB (`gate_mm_input_cb`) is the same CB that receives the mcasted, RMSNorm-normalized activation. This CB is shared between the gate matmul and the DRAM-streaming expert matmuls to avoid duplicating the 7168-element input in L1.

After this step, each of the 8 cores holds a width shard of the 256 sigmoid scores (32 scores per core).

---

## 6.1.4 Step 2: Gate Gather -- Collecting 256 Scores to One Core

The 256 scores are distributed across the 8 gate matmul cores. They must be gathered onto the sender core to form the full $[16, 16]$ gate input. This uses the standard `MoeGather` pattern: each compute core sends its output shard to the receiver (sender core), which writes the incoming data into contiguous CB memory.

```
  Core 0         Core 1       ...     Core 7
  [scores        [scores              [scores
   0..31]         32..63]              224..255]
     \              |                   /
      \             |                  /
       ----- NOC0 unicast ------------
                    |
                    v
              Sender Core (12,9)
         +---------------------+
         |  gate_input_cb      |
         |  [16, 16] as 1 tile |
         |  (16x16 tile format)|
         +---------------------+
```

The gather uses column-major grid traversal (`row_major=False` in `setup_gather` at line 1039, `moe_routed_expert/op.py`). This ensures that the 32-score shards from each core are placed into the correct positions of the $16 \times 16$ tile.

The destination CB uses a $16 \times 16$ tile format (not the standard $1 \times 32$), so the 256 scores land in one tile that the gate compute kernel can process as a native face operation. The `dst_num_pages=1` (line 1041 of `moe/op.py`) confirms that all 256 scores fill exactly one face.

### NOC Coordinate Resolution

The gather needs the receiver core's physical NOC coordinates (line 309 of `moe_routed_expert/op.py`):

```python
receiver_core_noc = device.worker_core_from_logical_core(receiver_core)
```

Logical coordinates (used in the Python setup) differ from physical NOC coordinates. The `worker_core_from_logical_core` method performs the translation, accounting for the hardware's core ID mapping.

### Sender Grid Bounding Box

The gather sender's compile-time arguments include a bounding box of the sender grid (lines 311--316):

```python
sender_min_x = min(r.start.x for r in sender_ranges)
sender_min_y = min(r.start.y for r in sender_ranges)
sender_max_x = max(r.end.x for r in sender_ranges)
sender_max_y = max(r.end.y for r in sender_ranges)
```

The kernel uses this bounding box to compute each core's position within the sender grid (its "sender index") when `use_explicit_sender_index=False`. The sender index determines the byte offset at which the core writes into the gather destination buffer on the receiver.

---

## 6.1.5 Step 3: The DeepSeek MoE Gate Algorithm

The `DeepseekMoeGateSingleCore` micro-op (`micro_ops/deepseek_moe_gate/op.py`) implements DeepSeek V3's hierarchical expert selection on a single core. This is the most algorithmically complex micro-op in the system: it performs multi-level sorting and selection entirely within the TRISC compute pipeline on a single 16x16 tile.

### Paper vs. Implementation

DeepSeek V3 organizes 256 routed experts into 16 groups of 16 experts each. The paper specifies:

1. Compute affinity scores via sigmoid: $s_i = \sigma(u_i^T h)$
2. Add a learned bias $b_i$ (the `e_score_correction_bias`) for load balancing: $s_i' = s_i + b_i$
3. Select top-$K$ experts ($K=8$) using a hierarchical group-then-expert strategy
4. Normalize the selected scores and apply a scaling factor

The TT implementation maps this algorithm onto a single 16x16 tile processed by one compute core. The key algorithmic subtlety is the **separation between biased scores and original scores**: biased scores drive all selection decisions (sorting, group ranking, top-8 extraction), but the original unbiased sigmoid scores are used for the final normalization. This ensures the bias influences only which experts are selected (load balancing), not how their contributions are weighted.

### Algorithm Steps

The golden reference implementation (`op.py`, lines 25--59) reveals the precise algorithm:

**Phase 1: Bias-Corrected Scoring**

$$R_{[16, 16]} = S_{[16, 16]} + B_{[16, 16]}$$

The bias $B$ is a learned per-expert correction that shifts the ranking scores to encourage more uniform expert selection. The raw sigmoid scores $S$ are preserved separately for the final normalization. When `enable_sigmoid=True`, sigmoid is applied to the input scores before adding bias. In the fused MoE kernel, sigmoid is already applied in the gate matmul, so `enable_sigmoid=False` is used here (line 1058, `moe/op.py`).

**Phase 2: Intra-Group Sorting**

Within each of the 16 groups (rows of the $16 \times 16$ matrix), experts are sorted by their biased score $R$ in descending order:

$$\text{sorted\_bias}[g, :], \text{sorted\_indices}[g, :] = \text{sort\_descending}(R[g, :])$$

The original scores $S$ are reordered to match via gather:

$$\text{sorted\_scores}[g, :] = \text{gather}(S[g, :], \text{sorted\_indices}[g, :])$$

The flat indices are computed by adding row offsets:

$$\text{sorted\_indices}[g, j] \mathrel{+}= g \times 16$$

This row-offset addition converts intra-group column indices (0--15) to global expert IDs (0--255), ensuring that when indices are flattened during later stages, the result is global expert IDs rather than within-group positions.

**Phase 3: Group Ranking by Top-2 Sum**

Groups are ranked by the sum of their two highest biased scores:

$$\text{top2\_sum}[g] = \text{sorted\_bias}[g, 0] + \text{sorted\_bias}[g, 1]$$

$$\text{group\_order}[:] = \text{argsort\_descending}(\text{top2\_sum}[:])$$

This selects the 4 best groups (the top 4 entries of `group_order`). The intuition: groups whose top experts have the highest combined scores are most likely to contain relevant experts for this token.

**Phase 4: Flatten Top-4 Groups to 64 Candidates**

The top 4 groups contribute all 16 of their sorted experts to a candidate pool:

$$\text{candidates}_{[1, 64]} = \text{flatten}(\text{sorted\_bias}[\text{group\_order}[0:4], :])$$

$$\text{candidate\_scores}_{[1, 64]} = \text{flatten}(\text{sorted\_scores}[\text{group\_order}[0:4], :])$$

$$\text{candidate\_indices}_{[1, 64]} = \text{flatten}(\text{sorted\_indices}[\text{group\_order}[0:4], :])$$

This reduces the search space from 256 to 64, while preserving the guarantee that the globally best 8 experts (by biased score) are included.

**Phase 5: Final Top-8 Selection and Normalization**

From the 64 candidates, select the top 8 by biased score:

$$\text{top8\_idx}[:] = \text{topk}(\text{candidates}, k=8)$$

Map back to unbiased scores and flat indices:

$$\text{top8\_scores}[:] = \text{gather}(\text{candidate\_scores}, \text{top8\_idx})$$

$$\text{top8\_indices}[:] = \text{gather}(\text{candidate\_indices}, \text{top8\_idx})$$

Normalize and scale:

$$\text{normalized}[i] = \frac{\text{top8\_scores}[i]}{\sum_{j=0}^{7} \text{top8\_scores}[j] + \epsilon} \times \alpha$$

where $\epsilon = 10^{-20}$ and $\alpha = 2.5$ (the configurable scaling factor). These values are set in the gate setup (`moe/op.py`, lines 1045--1058) and passed to the kernel as uint32 bit-pattern encodings via `float_to_uint32()`.

The scaling factor of 2.5 compensates for the fact that only 8 of 256 experts contribute. Without it, each expert's effective weight would be roughly $1/8 = 0.125$, which is too small for stable gradients during training. The 2.5x amplification ensures the selected experts have meaningful influence.

### Why Hierarchical Selection?

A naive top-8 from 256 experts would require sorting all 256 values. The hierarchical approach reduces this:

1. **16 parallel row sorts** (16 elements each) -- cheap on a 16x16 tile
2. **1 sort of 16 group sums** -- cheap (only 16 values)
3. **1 top-8 from 64 candidates** -- much cheaper than top-8 from 256

This hierarchy exploits the tile architecture: row operations within a 16x16 tile are natural for the TRISC engine, and the group-level ranking reduces the search space by 4x before the final selection.

| Approach | Groups Examined | Candidates | Final Selection |
|---|---|---|---|
| Flat top-8 | All 16 (256 experts) | 256 | Best 8 globally |
| Hierarchical (paper + impl) | Top-4 by group score | 64 (4 groups x 16) | Best 8 from 64 |

### Golden Reference Code

```python
# Phase 1: Optional sigmoid + bias addition
scores = torch.sigmoid(input_tensor) if enable_sigmoid else input_tensor
bias_scores = scores + bias_tensor

# Phase 2: Intra-group sort
sorted_bias, sorted_indices = torch.sort(bias_scores, dim=-1, descending=True)
sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices)
sorted_indices = sorted_indices + row_offsets.view(1, -1, 1)

# Phase 3: Group ranking
top2_sum = sorted_bias[:, :, 0] + sorted_bias[:, :, 1]
sorted_top2_sum, sorted_top2_indices = torch.sort(top2_sum, dim=-1, descending=True)

# Phase 4: Flatten top-4 groups
top4_values = sorted_bias[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
top4_scores = sorted_scores[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
top4_indices = sorted_indices[batch_idx, sorted_top2_indices[:, :4]].flatten(1)

# Phase 5: Final top-8 + normalize
top8_values, top8_indices = torch.topk(top4_values, 8, dim=-1, sorted=True)
top8_values = torch.gather(top4_scores, dim=-1, index=top8_indices)
top8_indices = torch.gather(top4_indices, dim=-1, index=top8_indices)
denominator = torch.sum(top8_values, dim=-1, keepdim=True) + eps
normalized_scores = top8_values / denominator * scaling_factor
```

### Hardware Implementation

The micro-op's 5-CB layout and compile-time arguments (epsilon, scaling factor, sigmoid flag) are documented in [Chapter 3 Section 3.3.8](../ch03_the_micro_op_library/03_communication_and_infrastructure_micro_ops.md). The output tile format is $1 \times 16$ (not $1 \times 32$), which is the minimum tile width that can hold the 8 useful values — keeping the mcast payload for index and scale distribution as small as possible (32 bytes each for bfloat16/uint16).

---

## 6.1.6 Step 4: Index and Scale Multicast

After the gate produces top-8 indices and scores on the sender core, they must be distributed to all DRAM matmul cores that will execute the expert projections.

### Index Multicast

The top-8 expert indices (stored in `gate_output_indices_tensor`, a 1x16 tile of uint16) are multicast from the sender core to all cores in the mcast grid. The mcast parameters are configured at `moe/op.py` lines 1060--1065:

```python
index_tile_size = TILE_1x16.get_tile_size(ttnn.bfloat16)  # 1 * 16 * 2 = 32 bytes
index_mcast_num_pages = 1
index_mcast_data_size_bytes = index_tile_size              # 32 bytes
```

This is a remarkably small multicast -- only 32 bytes -- but it carries the information that determines which of the 256 expert weight matrices each DRAM core will stream. The receiver semaphore is `INDEX_MCAST_RECEIVER` (ID 10), separate from the data mcast semaphore to allow temporal overlap.

### Expert Scale Multicast

The normalized score for the first expert (the highest-scoring one) is broadcast similarly:

```python
expert_scale_mcast_num_pages = 1
expert_scale_mcast_data_size_bytes = index_tile_size  # 32 bytes (1x16 tile)
```

The expert scale uses a different receiver semaphore (`EXPERT_SCALE_MCAST_RECEIVER`, ID 9) to avoid race conditions with the index mcast. Both mcasts reuse the same sender semaphore (`MCAST_SENDER`, ID 0) because they execute sequentially from the sender's perspective.

### Per-Device Expert Indexing

In the multi-device (4x2 mesh) configuration, each device processes a different expert from the top-8 list. The `gate_proj_index_offset` compile-time argument is set to `mesh_chip_id` when routing is enabled (`moe/op.py`, line 1530):

```python
("gate_proj_index_offset", mesh_chip_id if ctx.enable_routing else 0),
```

This means device 0 uses `top8_indices[0]`, device 1 uses `top8_indices[1]`, and so on. All 8 devices process different experts in parallel, and the results are combined via ReduceToOne. Similarly, `mul_scalar_index_offset` (line 1630) selects the corresponding score for each device.

---

## 6.1.7 The RMSNorm Preceding MoE

Before the gating pipeline begins, the hidden state must be normalized. In the fused MoE kernel, RMSNorm runs on the sender core as the very first operation, producing the normalized input that feeds both the routed expert path and the shared expert path.

**Where in the code:** `_setup_dimensions` in `moe/op.py` (lines 908--974) sets up the RMSNorm configuration. The computation is:

$$\text{normalized}_{[1, K]} = \frac{\text{input}_{[1, K]}}{\text{RMS}(\text{input})} \times \gamma$$

where $\text{RMS}(x) = \sqrt{\frac{1}{K} \sum_{i=1}^{K} x_i^2}$ and $\gamma$ is the per-element learned scale from `ffn_norm`.

**Tile reinterpretation for RMSNorm:** The input arrives as $K / 32 = 224$ tiles of $1 \times 32$. The RMSNorm compute kernel works more efficiently with larger tiles. The code reinterprets these as either $32 \times 32$ tiles or $16 \times 32$ tiles depending on divisibility:

```python
# moe/op.py, lines 912-918
FULL_32x32_TILE = ttnn.Tile((32, 32))
HALF_16x32_TILE = ttnn.Tile((16, 32))
is_16x32_tile = (K // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
rmsnorm_interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
```

For $K = 7168$: $7168 / 32 = 224$ columns, $224 / 32 = 7$ full 32x32 tiles. Since $224 \bmod 32 = 0$, the check `(224) % 32 != 0` evaluates to `False`, so `is_16x32_tile=False` and the full $32 \times 32$ tile is used, giving `rmsnorm_num_tiles = 7`.

The gamma weights use the same reinterpreted tile format. The CB descriptors override the tensor's native $1 \times 32$ tile to the reinterpreted format (lines 928--933). The epsilon and scalar values are packed as uint32 (lines 970--971):

```python
rmsnorm_epsilon_packed = float_to_uint32(epsilon)       # 1e-6
rmsnorm_scalar_packed = float_to_uint32(1.0 / math.sqrt(float(K)))  # 1/sqrt(7168)
```

After RMSNorm, the normalized output is multicast to all 130 cores, kicking off both the gate matmul (routed path) and the activation distribution (shared path). The raw (un-normalized) input is separately multicast as the "residual" for the shared expert's bias-add step.

---

## 6.1.8 Gate Weight Preparation and Tensor Parallelism

### Routing Weights ($W_\text{gate}$)

The gate matmul weights $[7168, 256]$ are part of the fused `o_proj_gate_mm_norms` overlapped tensor group. They are stored in BFP16 (bfloat16) format and WIDTH_SHARDED across the gate matmul cores. The `O_PROJ_GATE_MM_RMSNORM_GAMMA_SingleDeviceOverlapSpec` class (`blitz_decode_weights.py`, lines 198--288) defines the layout: 8 cores, each with shard shape $(7168, 32)$. The gate_mm region follows the o_proj region in the fused buffer at a byte offset computed by `gate_mm_shard_bytes`.

### Single-Device Mode (`moe_tp = 1`)

| Tensor | Shape | Layout | Location |
|--------|-------|--------|----------|
| Gate MM weights | $[7168, 256]$ | `OverlappedTensor`, WIDTH_SHARDED on 8 cores | L1 (SRAM) |
| Gate bias | $[16, 16]$ | HEIGHT_SHARDED on sender core | L1 |
| Gate indices | $[16, 16]$ | HEIGHT_SHARDED on sender core | L1 |
| Gate output scores | $[1, 16]$ | HEIGHT_SHARDED on sender core | L1 |
| Gate output indices | $[1, 16]$ | HEIGHT_SHARDED on sender core | L1 |

### Multi-Device Mode (`moe_tp = 8`, 4x2 mesh)

The gate routing weights are replicated on all 8 devices -- every device independently computes the full 256-expert gating to determine which expert to process. The `e_score_correction_bias` and indices are also replicated. The routing weights are small enough that replication is efficient: $7168 \times 256 \times 2 = 3.67$ MB per device (bfloat16). This is far cheaper than communicating the gating decision across devices.

---

## 6.1.9 SRAM Gate Matmul vs. DRAM Streaming Matmul

It is worth noting why the gate matmul uses SRAM while the expert matmuls use DRAM streaming:

| Property | Gate Matmul | Expert Matmul |
|----------|-------------|---------------|
| Weight shape | $[7168, 256]$ | $[7168, 2048]$ per expert |
| Weight size (bf16) | 3.67 MB | 29.4 MB per expert |
| Number of instances | 1 (shared) | 256 (one per expert) |
| Total weight memory | 3.67 MB | 7.5 GB |
| Fits in L1? | Yes (as OverlappedTensor) | No |
| Implementation | `setup_sram_matmul` | `setup_dram_matmul` |
| Cores used | 8 (gate_mm weight cores) | 8 (DRAM bank-optimal cores) |
| Fused activation | Sigmoid (`ACTIVATION_SIGMOID = 1`) | SiLU for gate_proj, none for up/down |

The gate matmul weight tensor is an `OverlappedTensor` -- a tensor whose L1 memory is shared with other tensors that have non-overlapping lifetimes. This is managed by the CB overlap system described in Chapter 4. The expert weights, being 2000x larger in aggregate, must reside in DRAM and be streamed on demand.

### Activation Fusion Constants

The matmul kernel supports three activation modes, defined as constants in `MoeRoutedExpertOp` (lines 337--339):

```python
ACTIVATION_NONE = 0     # No post-matmul activation
ACTIVATION_SIGMOID = 1  # sigma(x) -- used for gate routing scores
ACTIVATION_SILU = 2     # x * sigma(x) -- used for gate_proj
```

---

## 6.1.10 End-to-End Gating Data Flow

The complete gating and routing path for a single MoE layer:

```
  +----------------------------------------------------------------------+
  |                          SENDER CORE (12,9)                          |
  |                                                                      |
  |  RMSNorm(residual) --> input [1, 7168]                               |
  |         |                                                            |
  |         | (1) Mcast [1, 7168] to 130-core grid                       |
  |         v                                                            |
  |  +---------------------+    +-------------------------------+        |
  |  | 8 Gate MM Cores     |    | 8 DRAM Expert Matmul Cores    |        |
  |  | [1,7168]x[7168,256] |    | (receive mcast, wait for      |        |
  |  | + sigmoid           |    |  index/scale)                  |        |
  |  +--------+------------+    +-------------------------------+        |
  |           |                                                          |
  |           | (2) Gather 256 scores to sender                          |
  |           v                                                          |
  |  +-------------------------------------+                             |
  |  | Gate Input [16, 16] on sender core  |                             |
  |  | + bias [16, 16]                     |                             |
  |  | + indices [16, 16]                  |                             |
  |  +--------+----------------------------+                             |
  |           |                                                          |
  |           | (3) DeepSeek MoE Gate (single-core TRISC)                |
  |           |    sort per group -> rank groups -> flatten top-4         |
  |           |    -> top-8 -> normalize x 2.5                           |
  |           v                                                          |
  |  +--------------------+  +--------------------+                      |
  |  | Top-8 Indices      |  | Top-8 Scores       |                      |
  |  | [1, 16] uint16     |  | [1, 16] bfloat16   |                      |
  |  +--------+-----------+  +--------+-----------+                      |
  |           |                       |                                  |
  |           | (4) Index Mcast       | (5) Scale Mcast                  |
  |           v                       v                                  |
  |  +------------------------------------------+                        |
  |  |  All DRAM Expert Matmul Cores            |                        |
  |  |  Now have: expert_index + expert_scale   |                        |
  |  |  Ready for gate_proj / up_proj / ...     |                        |
  |  +------------------------------------------+                        |
  +----------------------------------------------------------------------+
```

The entire gating pipeline, from RMSNorm to expert selection, is a single fused kernel dispatch. There are no host round-trips between steps -- all synchronization is via on-chip semaphores and multicast operations. This is critical for latency: at decode time with batch size 1, the gate computation must complete in microseconds to avoid becoming a bottleneck before the much more expensive DRAM-streaming expert matmuls begin.

---

[Previous: Chapter 5](../ch05/index.md) | [Next: 6.2 Routed Expert Computation](02_routed_expert_computation.md)
