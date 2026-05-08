# 03 -- Ops Catalog and Composition Patterns

This file surveys the `blaze/ops/` directory (114 op modules comprising 54 MicroOps and 49 FusedOps plus utility modules), catalogs the major op families for DeepSeek V3 and GLM-5.1 model architectures, demonstrates the five principal FusedOp composition patterns with a complete SharedExpert worked example, explains how ops handle multi-device execution via mesh coordinates, documents the OverlappedView system for weight packing, and covers the weight_provider system for sourcing model weights from synthetic, HuggingFace, and Blitz cache sources.

---

## 1. Directory Structure and Auto-Discovery

Each op lives in its own package under `blaze/ops/`:

```
blaze/ops/
    matmul/
        op.py              # Python class (MicroOp or FusedOp)
        kernels/
            op.hpp         # C++ kernel header (MicroOps only)
    mcast/
        op.py
        kernels/
            op.hpp
    moe/
        op.py              # FusedOp -- no kernels/ dir
    ...
```

Auto-discovery at import time registers every `BlazeOp` subclass via `register_all()` (see [file 01, Section 4](./01_blaze_op_architecture.md#phase-1-class-definition-and-registration) for the implementation). The 114 ops break down as follows:
- **54 MicroOps** (C++ kernel-backed building blocks)
- **49 FusedOps** (compositions of MicroOps)
- **~11 other** (utilities, multi-file ops, special-purpose)

### 1.1 Op Family Breakdown by Count

```
  Op Distribution Across Model Families

  +-------------------+--------+----------+-------+
  | Family            | Micro  | Fused    | Total |
  +-------------------+--------+----------+-------+
  | DeepSeek V3       |   15   |    19    |   34  |
  | GLM-5.1           |    3   |    15    |   18  |
  | Infrastructure    |   22   |     2    |   24  |
  | DSA (Sparse)      |    7   |     7    |   14  |
  | CCL/Fabric        |    5   |     4    |    9  |
  | Shared/Generic    |    2   |     2    |    4  |
  +-------------------+--------+----------+-------+
  | Total             |   54   |    49    |  114* |
  +-------------------+--------+----------+-------+
  * includes utility modules (utils, README, __init__)
```

---

## 2. MicroOp Building Blocks

MicroOps are backed by C++ kernel headers and form the atomic computation units. They are composed by FusedOps.

### 2.1 Data Movement

| Op Name             | Description                                              |
|---------------------|----------------------------------------------------------|
| `mcast`             | Multicast from sender core to all cores                  |
| `gather`            | Gather tiles from scattered cores to sender              |
| `gather_reduce`     | Dual-destination gather with pairwise reduction          |
| `scatter`           | Distribute data from source to multiple destinations     |
| `scatter_raw`       | Row-level scatter with head replication                  |
| `copy`              | Generic L1 data movement via NOC                         |
| `shard2cb`          | Fetch pages from sharded tensor into local scratch CB    |
| `tile_copy`         | Copy tiles between CBs on a single core                  |

### 2.2 Compute

| Op Name             | Description                                              |
|---------------------|----------------------------------------------------------|
| `matmul`            | Matrix multiply: activations x weights                   |
| `matmul_cb`         | Matmul writing to caller-provided output CB              |
| `matmul_fused_act`  | Matmul with fused sigmoid or SiLU on output              |
| `kn_sliced_matmul`  | KN-sliced matmul on A/B split core grids                 |
| `dram_streaming_matmul` | DRAM-weight-streaming matmul                         |
| `eltwise_mul`       | Element-wise multiply with optional scalar scale         |
| `gated_local_reduce`| SiLU(sum(group1)) * sum(group2), single core             |
| `gated_reduce`      | Multi-core gated reduction (gate + up branches)          |
| `reduce_to_one`     | Tree-reduce to single core                               |
| `clamped_silu`      | Clamped SiLU activation for GPT-OSS clamped SwiGLU       |

### 2.3 Normalization and Embedding

| Op Name             | Description                                              |
|---------------------|----------------------------------------------------------|
| `rmsnorm`           | RMS normalization (tile-aligned widths)                  |
| `padded_rmsnorm`    | RMS normalization with zero-padding                      |
| `rope`              | Rotary Position Embedding                                |
| `embedding`         | Token ID to DRAM weight row lookup                       |

### 2.4 Attention

| Op Name             | Description                                              |
|---------------------|----------------------------------------------------------|
| `sdpa`              | Scaled dot-product attention for GQA and MLA             |
| `sdpa_reduce`       | Multi-device streaming SDPA tail reduction               |
| `flash_mla`         | Flash MLA decode (single-device)                         |
| `sparse_flash_mla`  | Sparse (DSA) variant of flash MLA                        |
| `create_q_heads`    | Gather from sender cores to receiver cores + tilize      |

### 2.5 MoE Routing

| Op Name             | Description                                              |
|---------------------|----------------------------------------------------------|
| `deepseek_moe_gate` | Top-8 routed scores and indices per token                |
| `glm_moe_gate`      | Top-k routed scores and indices (GLM variant)            |
| `glm_moe_gate_merge`| Two-core on-device merge for 512 experts                 |
| `distributed_topk`  | Distributed top-k across devices                         |

### 2.6 Tile/Format Conversion

| Op Name             | Description                                              |
|---------------------|----------------------------------------------------------|
| `retilize`          | Convert (1,32) row tiles to (N,32) standard tiles        |
| `untilize`          | Convert (N,32) tiles to (1,32) row-major pages           |
| `q_idx_tilize`      | Convert Q_idx from row-major to face-packed tile order   |
| `tile_row_convert`  | Convert TopK position IDs to unique tile-row IDs         |

### 2.7 CCL and Synchronization

| Op Name             | Description                                              |
|---------------------|----------------------------------------------------------|
| `all_reduce`        | AllReduce: exchange + local sum reduction across devices  |
| `ccl_broadcast`     | Multi-device fabric broadcast from sender                |
| `barrier_sender`    | Signal completion to barrier receiver cores              |
| `barrier_receiver`  | Wait for completion signals from barrier sender          |
| `pipeline_stage_sync` | Synchronize pipeline stages                            |
| `dm_risc_handshake` | Data-movement RISC handshake                             |

### 2.8 Housekeeping

| Op Name             | Description                                              |
|---------------------|----------------------------------------------------------|
| `cb_flush`          | Wait for scratch CB, copy to tensor-backed output CB     |
| `cb_reconfig`       | Reconfigure CB interfaces at phase boundary              |
| `cb_scratch_reset`  | Reset scratch CB fifo pointers at temporal reuse boundary|
| `residual_add`      | Element-wise residual addition                           |
| `argmax`            | Argmax for LM head sampling                              |
| `migration`         | State migration utilities                                |

---

## 3. DeepSeek V3 Ops

### 3.1 MLA (Multi-head Latent Attention)

The `MLA` FusedOp [blaze/ops/mla/op.py] is the largest composition in Blaze, integrating the full attention pipeline into a single dispatch:

```
  MLA Composition Pipeline

  PreSDPA (FusedOp)
  +---------------------------------------------------------------+
  | RMSNorm(activation)                                            |
  |    |                                                           |
  |    +--> Matmul(q_a_proj) --> RMSNorm2 --> Matmul(q_b_proj)    |
  |    |         |                    |                            |
  |    |         v                    v                            |
  |    |    CreateQHeads(Q)     RoPE(Q_rope)                       |
  |    |                                                           |
  |    +--> Matmul(kv_a_proj) --> RMSNorm(kv_norm)                |
  |              |                    |                            |
  |              v                    v                            |
  |         KVBranch          Matmul(kv_b1) --> RoPE(K_rope)      |
  +---------------------------------------------------------------+
       |
       v
  DistributedFlashMLA (FusedOp)
  +---------------------------------------------------------------+
  | FlashMLA(Q, KV_cache, cur_pos)                                 |
  | SDPA_Reduce (multi-device reduce for distributed attention)    |
  +---------------------------------------------------------------+
       |
       v
  PostSdpa (FusedOp)
  +---------------------------------------------------------------+
  | Scatter(sdpa_output) --> Matmul(kv_b2) --> Gather              |
  |    |                                                           |
  |    +--> Mcast --> Matmul(o_proj) --> Gather                    |
  |                                                                |
  |    --> AllReduce (cross-device) --> ResidualAdd                 |
  +---------------------------------------------------------------+
```

MLA takes 20+ input tensors and orchestrates PreSDPA, DistributedFlashMLA, and PostSdpa as sub-FusedOps, each of which composes MicroOps internally.

### 3.2 MoE (Mixture of Experts)

The `MoE` FusedOp [blaze/ops/moe/op.py] composes the MoE block:

```
  MoE Composition Pipeline

  Input Activation
       |
       v
  RMSNorm --> Mcast (broadcast normalized activation)
                |
       +--------+---------+
       |                   |
       v                   v
  RoutedExpert         SharedExpert
  (FusedOp)            (FusedOp)
       |                   |
       v                   v
  MoERouter            KNMatmul(gate_up)
  -> SwigluOp          -> GatedReduce
  -> ReduceToOne       -> Mcast
                       -> DownProj
       |                   |
       +--------+---------+
                |
                v
          ReduceToOne (cross-device)
                |
                v
          ResidualAdd
```

Variants by expert count and model:

| Op Name              | Experts | Shared Expert? | Model                    |
|----------------------|---------|----------------|--------------------------|
| `moe`                | <=256   | Yes            | DeepSeek V3/R1           |
| `moe_no_shared`      | <=256   | No             | DeepSeek V3/R1           |
| `large_moe`          | >256    | Yes            | DeepSeek V3/R1           |
| `large_moe_no_shared`| >256    | No             | DeepSeek V3/R1           |
| `allreduce_moe`      | <=256   | Yes            | DeepSeek V3/R1 (TP)     |
| `glm_moe`            | <=256   | Yes            | GLM 5.1                  |
| `glm_moe_no_shared`  | <=256   | No             | GLM 5.1                  |
| `glm_large_moe`      | >256    | Yes            | GLM 5.1                  |
| `glm_large_moe_no_shared` | >256 | No           | GLM 5.1                  |

The `>256 expert` variants need two phantom columns on the 13x10 grid (up to 576 experts) because gate matmul weight sharding requires more cores than a single column provides.

### 3.3 MoE Router

The `MoERouter` FusedOp composes the expert routing gate:

```
  MoERouter: DeepseekMoeGate --> Mcast(topk_index)
```

`DeepseekMoeGate` is a MicroOp that computes softmax-normalized expert scores, applies the `e_score_correction_bias`, selects the top-K experts (K=8 for DeepSeek V3), and outputs expert indices and routing weights. It is a multi-output op:

```python
scores, indices = DeepseekMoeGate.emit(f, ...)   # positional unpacking
result = DeepseekMoeGate.emit(f, ...)
result.output                                      # by name
result["output_indices"]                           # dict-style
```

### 3.4 RoutedExpert

The `RoutedExpert` FusedOp [blaze/ops/routed_expert/op.py] composes:

```
  RoutedExpert = MoERouter --> SwigluOp

  SwigluOp (FusedOp): indexed DRAM-streaming matmul with SiLU-gated activation
    DramStreamingMatmul(gate) --> CopiedSilu
    DramStreamingMatmul(up)   --> EltwiseMul
    DramStreamingMatmul(down) --> accumulate
```

The DramStreamingMatmul reads weights from DRAM at expert-specific offsets, streaming tiles through L1 CBs to minimize memory footprint.

### 3.5 SharedExpert

The `SharedExpert` FusedOp [blaze/ops/shared_expert/op.py] implements the shared expert path:

```
  SharedExpert = KNMatmul(gate_up) --> GatedReduce --> Mcast --> DownProj
```

`KNMatmul` is a K-N sliced matmul that partitions weight tiles across A (gate) and B (up) core grids for parallel execution. `GatedReduce` applies SiLU gating and reduces the gate and up branches. `DownProj` is the output projection. A full worked example of this composition is provided in Section 6.

### 3.6 Large MoE Variants

`LargeMoE`, `LargeMoENoShared`, `LargeRoutedExpert` handle larger expert counts (>256 experts) by using multi-column phantom grids for gate weight placement:

```python
# role_engine.py:GridConfig
def get_gate_mm_cores(self, num_cores: int):
    rows_per_col = self.grid_rows - 1   # 9 rows per phantom col
    ncols = self.num_phantom_cols(num_cores)  # 1 or 2
    for col_offset in range(ncols):
        col = self.grid_cols - 1 - col_offset
        for row in range(rows_per_col):
            ...
```

### 3.7 Sparse Layer (Full Decoder Layer)

The `sparse_layer` op composes MLA + MoE into a complete decoder layer:

```python
class SparseLayer(FusedOp):
    name: str = "sparse_layer"
    # Combines MLA, CB Reconfig, and MoE in sequence
```

---

## 4. GLM-5.1 Ops

### 4.1 GlmFusedProj

[blaze/ops/glm5_fused_proj/op.py]

GLM-5.1's fused projection runs four parallel matmuls from a single Mcast:

```
  GlmFusedProj: Mcast(hidden) --> 4 x parallel Matmul --> 4 x Gather

  All weights packed in one bfloat8_b fused tensor with OverlappedViews
  at byte_offset=0 on disjoint cores (no stacking).

  Outputs:
    k_new   [1, idx_dim]       -- indexer K projection
    Q_idx   [total_tiles, 32]  -- indexer Q projection
    weights [1, n_idx_heads]   -- indexer weight projection
    Q_a     [total_tiles, 32]  -- Q_a projection
```

The key insight is that all four weight matrices are stored in a single fused L1 buffer using `OverlappedView`s at different core ranges. After one Mcast of the hidden state, all four matmuls execute in parallel on disjoint core grids, achieving kernel time of `max(matmul_time)` instead of `sum(matmul_times)`.

### 4.2 GlmQBranch

[blaze/ops/glm5_q_branch/op.py]

Composes the Q attention branch for GLM-5.1 GQA (grouped-query attention):

```
  GlmQBranch = Matmul(q_b_proj) --> RoPE --> CreateQHeads
```

### 4.3 GLM MoE Family

GLM-5.1 has its own MoE variants that differ from DeepSeek V3 in routing:

| Op                     | Difference from DeepSeek V3                           |
|-----------------------|-------------------------------------------------------|
| `GLMMoE`              | Uses `GLMRoutedExpert` with GLM-style gate merge     |
| `GLMMoENoShared`      | No shared expert path                                 |
| `GLMMoERouter`        | Uses `GLMMoeGate` + `GLMMoeGateMerge` for routing    |
| `GLMMoELargeRouter`   | Large expert count routing (>256)                     |
| `GLMLargeRoutedExpert`| Large expert variant                                  |
| `GLMLargeMoE`         | Full large MoE with shared expert                     |

The `GLMMoeGate` MicroOp implements GLM-specific expert selection with normalization (`normalize=True` by default) and configurable `num_selected_experts` (8 by default for GLM). `GLMMoeGateMerge` is a two-core on-device merge for 512 expert routing.

### 4.4 GLM Attention Variants

GLM 5.1 has multiple attention dispatch patterns depending on the execution context:

| Op Name                       | Device | Q Branch? | Description                                    |
|-------------------------------|--------|-----------|------------------------------------------------|
| `glm5_sdpa_post_sdpa`        | Single | No        | SparseMLA + PostDsa in one dispatch            |
| `glm5_q_sdpa_post_sdpa`      | Single | Yes       | Q Branch + SparseMLA + PostDsa                 |
| `glm5_sdpa_post_sdpa_multi`  | Multi  | No        | Multi-device with SDPA reduce                  |
| `glm5_q_sdpa_post_sdpa_multi`| Multi  | Yes       | Multi-device Q + SparseMLA + PostDsa           |

The Q branch variant includes a full Q pipeline:

```
  Q Branch --> Sparse SDPA --> PostDsa Matmul --> Wo Matmul --> output

  Q Branch:
    RMSNorm(Q_a) --> Mcast --> Matmul(W_nope) --> Gather -->
    Barrier --> Shard2CB --> MatmulCB(absorb) --> MatmulCB(rope) -->
    ScatterRaw(head cores --> S1 cores)
```

---

## 5. DSA (Dynamic Sparse Attention) Ops

The DSA family implements sparse attention for long-context inference. GLM 5.1 uses dynamic sparse attention with indexer-based KV cache access:

```
  DSA Pipeline

  Stage 1: Score indexer keys against query
  +---------------------------------------------------+
  | DsaIndexer  -- Per-bank scoring of indexer cache   |
  +---------------------------------------------------+
       |
       v
  Stage 2: Select top-K cache positions
  +---------------------------------------------------+
  | DsaTopk     -- Hierarchical top-k across shards   |
  +---------------------------------------------------+
       |
       v
  Stage 3: Gather selected KV entries
  +---------------------------------------------------+
  | DsaSparseGather -- Index-gather KV rows from cache |
  +---------------------------------------------------+
       |
       v
  Stage 4: Compute attention on sparse KV
  +---------------------------------------------------+
  | DsaSparseFlashDecode -- Sparse flash attention     |
  +---------------------------------------------------+
       |
       v
  Stage 5: Update cache for next iteration
  +---------------------------------------------------+
  | DsaKvPrep       -- Tilize gathered KV for DRAM    |
  | DsaCacheUpdate  -- Write new KV to DRAM cache     |
  +---------------------------------------------------+
```

`DsaPipeline` (FusedOp) fuses stages 1-3 into a single dispatch. `DsaAttention` and `DsaFullAttention` compose larger portions of the pipeline with the attention computation included.

DSA-related ops:

| Op Name                | Type    | Description                                       |
|------------------------|---------|---------------------------------------------------|
| `dsa_indexer`          | Micro   | Per-bank scoring of indexer key cache              |
| `dsa_topk`            | Micro   | Hierarchical top-k across sharded cores            |
| `dsa_sparse_gather`   | Micro   | Index-gather selected KV rows from cache           |
| `dsa_sparse_flash_decode` | Micro | Sparse flash decode on gathered KV               |
| `dsa_cache_update`    | Micro   | Write one row to DRAM ROW_MAJOR cache              |
| `dsa_kv_prep`         | Micro   | Tilize gathered KV and write K/V to DRAM           |
| `dsa_retilize_indexer`| Fused   | QIdxTilize + Mcast + DsaIndexer                   |
| `dsa_pipeline`        | Fused   | Indexer + TopK + SparseGather pipeline             |
| `dsa_attention`       | Fused   | DSA Sparse Attention                               |
| `dsa_full_attention`  | Fused   | Full attention layer                               |

---

## 6. FusedOp Composition -- Worked Example: SharedExpert

The canonical pattern for composing ops follows a consistent structure. Here is a complete walkthrough using `SharedExpert`:

### Step 1: Define the FusedOp Class

```python
class SharedExpert(FusedOp):
    name: str = "shared_expert"
    math_fidelity: str = "LoFi"
    math_approx_mode: bool = True

    activation: Input = Input()
    gate_up_weights: Input = Input()
    down_weights: Input = Input()
    bias: Input = Input()
    output: Output = Output()
```

The class attributes declare the op's interface: input ports for activation and weight tensors, an output port for the result, and metadata (`math_fidelity`, `math_approx_mode`) that propagate as kernel defines.

### Step 2: Implement compose() to Bridge Graph API to emit()

```python
    @classmethod
    def compose(cls, f, tensors, output, user_args):
        cls.emit(
            f,
            tensors["activation"],
            tensors["gate_up_weights"],
            tensors["down_weights"],
            tensors["bias"],
            output,
            prefix=user_args.get("prefix", "shared_expert"),
            ab_coords=user_args.get("ab_grids"),
            down_coords=user_args.get("down_coords"),
            pop_shared_act=user_args.get("pop_shared_act", True),
        )
```

`compose()` is the graph API entry point -- it receives the FusedProgram, named tensors, and user args, then delegates to `emit()` with explicit keyword arguments.

### Step 3: Implement emit() as a Chain of MicroOp.emit() Calls

```python
    @staticmethod
    def emit(f, activation, gate_up_weights, down_weights, bias, output,
             *, prefix, ab_coords=None, down_coords=None, pop_shared_act=True):

        # 1. KN-Sliced Matmul: gate + up projections in parallel
        gu = KNMatmul.emit(
            f, activation, gate_up_weights,
            prefix=BlazeOp.child_prefix(prefix, "gu"),
            ab_coords=ab_coords,
            k_half_split=False,
            pop_in0=pop_shared_act,
        )

        # 2. Gated Reduce: SiLU(gate) * up, then reduce
        reduced = GatedReduce.emit(
            f, gu,
            prefix=BlazeOp.child_prefix(prefix, "gated_reduce"),
        )

        # 3. Mcast: broadcast reduced result to all cores
        mcasted = Mcast.emit(
            f, reduced,
            prefix=BlazeOp.child_prefix(prefix, "mcast_act"),
        )

        # 4. DownProj: output projection with bias add
        DownProj.emit(
            f, mcasted, down_weights, bias, output,
            prefix=BlazeOp.child_prefix(prefix, "down_proj"),
            down_coords=down_coords,
        )
```

### Data Flow Through Composition

```
  activation (ttnn.Tensor)
       |
       | KNMatmul.emit() returns CBHandle
       v
  gu_output (CBHandle, cb_id=5, on A+B cores)
       |
       | GatedReduce.emit() returns CBHandle
       v
  reduced (CBHandle, cb_id=8, on sender core)
       |
       | Mcast.emit() returns CBHandle
       v
  mcasted (CBHandle, cb_id=10, on all cores)
       |
       | DownProj.emit() wires to output tensor
       v
  output (ttnn.Tensor, written via cb_output)
```

Each `emit()` call:

1. Resolves its inputs (tensor via `cb_from_tensor()`, or CBHandle passthrough).
2. Allocates scratch CBs via `f.cb_scratch()`.
3. Sets role flags via `f.per_core_unified_ct_args([f.flag(...)])`.
4. Sets CT args for each RISC via `f.ncrisc_ct_args()`, `f.brisc_ct_args()`, `f.trisc_ct_args()`.
5. Returns a `CBHandle` to its output CB via `f.output()`.

### Prefix Nesting

FusedOps use `BlazeOp.child_prefix(prefix, name)` to build hierarchical prefixes:

```
  SharedExpert prefix = "shared_expert"
    KNMatmul prefix  = "shared_expert__gu"
    GatedReduce      = "shared_expert__gated_reduce"
    Mcast            = "shared_expert__mcast_act"
    DownProj         = "shared_expert__down_proj"
```

The `CHILD_PREFIX_DELIMITER` is `"__"`. This ensures CT arg names are unique across all composed ops (e.g., `shared_expert__gu.gate.in0` vs `shared_expert__down_proj.matmul.in0`).

---

## 7. Five Named Composition Patterns

### 7.1 Pattern: Mcast --> Matmul --> Gather

The most common pattern broadcasts activations to all cores, runs per-core matmul against local weight shards, and gathers results back:

```
  sender core        all cores          sender core
  +---------+       +---------+        +---------+
  | Mcast   | ----> | Matmul  | -----> | Gather  |
  | (act)   |       | (per-   |        | (merge  |
  |         |       |  core   |        |  partial|
  |         |       |  weight)|        |  results|
  +---------+       +---------+        +---------+
```

Used by: `qa_projection`, `gqa_pre_sdpa`, `down_proj`, `post_sdpa`

### 7.2 Pattern: A/B Branch Parallel Matmul

For gated MLPs (SwiGLU), two matmuls run in parallel on disjoint core halves:

```
  A cores (gate)            B cores (up)
  +-----------+             +-----------+
  | Matmul    |             | Matmul    |
  | (gate     |             | (up       |
  |  weights) |             |  weights) |
  +-----------+             +-----------+
       |                         |
       +---- GatedReduce -------+
             (SiLU * elem_mul + reduce)
```

Used by: `kn_sliced_matmul`, `shared_expert`, `swiglu`

### 7.3 Pattern: CCL Sandwiched Between Local Compute

```
  Local compute            CCL              Local compute
  +---------------+    +----------+    +---------------+
  | PreSDPA       | -> | SDPA     | -> | PostSdpa      |
  | (matmul chain)|    | Reduce   |    | (matmul chain)|
  +---------------+    | (cross-  |    +---------------+
                       |  device) |         |
                       +----------+    +----------+
                                       | AllReduce|
                                       | (cross-  |
                                       |  device) |
                                       +----------+
```

Used by: `mla`, `distributed_flash_mla_post_sdpa`, `allreduce_moe`

### 7.4 Pattern: Router --> Expert Selection --> Expert Execution

```
  act --> Gate Matmul --> Top-K --> Expert Indices
                                        |
                                        v
                              DRAM-streamed expert matmuls
                              (only for selected experts)
                                        |
                                        v
                                  ReduceToOne
```

Used by: `routed_expert`, `glm_routed_expert`, `large_routed_expert`

### 7.5 Pattern: BroadcastRMSNorm

Combines a CCL broadcast with RMSNorm in a single dispatch, used as the first stage of multi-device MLA/MoE pipelines:

```python
class BroadcastRMSNorm(FusedOp):
    name: str = "broadcast_rmsnorm"
    # CclBroadcast.emit(f, ...) --> RMSNorm.emit(f, ...)
```

See [Chapter 4, File 02](./02_ccl_in_blaze.md) Section 6.1 for the detailed integration pattern.

---

## 8. The OverlappedView System

`OverlappedView` [blaze/fused_program.py, lines 220-314] packs multiple weight matrices into a single L1 buffer using byte offsets on disjoint core ranges. See [file 01, Section 5.2](./01_blaze_op_architecture.md#52-circular-buffer-management) for the dataclass definition, field semantics, and L1 layout diagram.

FusedProgram allocates CBs from OverlappedViews via `cb_from_view()`:

```python
handle = f.cb_from_view(view, prefix="matmul.in1")
# handle.cb_id = <assigned>
# handle.byte_offset = view.byte_offset
# handle.access_mode = CBAccessMode.DIRECT_ADDRESS
```

The `DIRECT_ADDRESS` access mode (vs. `FIFO` for streaming CBs) tells the kernel to read from a fixed L1 address offset rather than the standard CB FIFO head pointer. This is essential for weight tensors that are pre-loaded into L1 and read multiple times.

OverlappedViews are central to GLM-5.1's fused projection pattern (`GlmFusedProj`, Section 4.1) and any op that pre-packs multiple weight matrices into a single tensor for bandwidth efficiency.

### MultiOutput

Multi-output ops return `MultiOutput` wrappers [blaze/fused_program.py, lines 317-358]; see [file 01, Section 5.6](./01_blaze_op_architecture.md#56-multioutput-for-multi-output-ops) for the three access modes (attribute, dict-style, positional unpacking).

---

## 9. How Ops Handle Multi-Device

### 9.1 Mesh Context Injection

When `BlazeCompiler.compile()` runs the per-device loop, it injects mesh context into each `FusedProgram`:

```python
for idx in range(num_devices):
    row, col = divmod(idx, self._num_cols)
    merged_args = {
        "mesh_coord": (row, col),
        "mesh_shape": (self._num_rows, self._num_cols),
        "mesh_device": self._mesh_device,
        **(user_args or {}),
        **dev_overrides,
    }
```

Ops access this via `f.mesh_coord`, `f.mesh_shape`, and `f.mesh_device`.

### 9.2 Per-Device Specialization

CCL ops use mesh coordinates to determine their role in the communication ring:

```python
# all_reduce/op.py
row, col = f.mesh_coord
ring_index = row if cluster_axis == 0 else col
is_first_chip = ring_index == 0
```

Non-CCL ops are generally device-agnostic -- the same `compose()` runs on every device, and per-device differences arise naturally from the per-device tensor shards passed in `tensors`.

### 9.3 Per-Device Overrides

`BlazeCompiler.compile()` accepts a `per_device_overrides` callable that returns device-specific CT arg overrides:

```python
result = compiler.compile(
    graph=ctx.graph,
    tensors={...},
    output_tensor=output,
    per_device_overrides=lambda row, col: {
        "ring_index": row,
        "is_first_chip": int(row == 0),
    },
)
```

### 9.4 Multi-Device FusedOp Variants

Several ops have explicit single-device and multi-device variants:

```
  Single-device                    Multi-device
  +---------------------------+    +---------------------------+
  | glm5_sdpa_post_sdpa       |    | glm5_sdpa_post_sdpa_multi |
  | (one chip, no CCL)        |    | (mesh, with SDPA reduce)  |
  +---------------------------+    +---------------------------+
  | flash_mla                 |    | distributed_flash_mla     |
  | (single-device decode)    |    | (FlashMLA + SdpaReduce)   |
  +---------------------------+    +---------------------------+
```

The multi-device variants add CCL operations (SDPA reduce, all-reduce) between local compute phases and coordinate fabric connections via `setup_fabric()`.

### 9.5 Mesh Tensor Decomposition

The compiler automatically splits mesh tensors into per-device tensors:

```python
per_device_inputs = _split_tensors(tensors)
per_device_outputs = ttnn.get_device_tensors(output_tensor)

for idx in range(num_devices):
    row, col = divmod(idx, self._num_cols)
    dev_tensors = {name: per_dev[idx]
                   for name, per_dev in per_device_inputs.items()}
    compiled = self._compile_single(ctx, graph, dev_tensors, ...)
```

`OverlappedView` inputs are re-wrapped per-device so downstream ops see the view's `shard_shape`, not the fused container's.

---

## 10. The Weight Provider System

The `weight_provider.py` [blaze/weight_provider.py] abstracts weight sourcing so tests and inference can switch between synthetic and real weights without code changes.

### 10.1 Architecture

```
                    WeightProvider (ABC)
                    +------------------+
                    | get_mla_torch_weights(layer_idx, **params)
                    | get_moe_torch_weights(layer_idx, **params)
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
  +-----------+--+  +-------+------+  +----+----------+
  | Synthetic    |  | StateDict    |  | BlitzCache    |
  | WeightProv.  |  | WeightProv.  |  | WeightProv.   |
  | (random)     |  | (HuggingFace)|  | (tensorbin)   |
  +--------------+  +--------------+  +---------------+
```

### 10.2 Provider Selection

Controlled by the `BLAZE_WEIGHT_SOURCE` environment variable:

| Value                  | Provider                 | Source                              |
|-----------------------|--------------------------|-------------------------------------|
| `synthetic` (default) | `SyntheticWeightProvider` | Seeded random with disk caching     |
| `state_dict:<path>`   | `StateDictWeightProvider` | HuggingFace safetensors from path   |
| `blitz:<path>`        | `BlitzCacheWeightProvider`| Blitz tensorbin cache at path       |

```python
def get_weight_provider(source=None) -> WeightProvider:
    if source is None:
        source = os.environ.get("BLAZE_WEIGHT_SOURCE", "synthetic")
    if source == "synthetic":
        return SyntheticWeightProvider()
    if source.startswith("state_dict:"):
        return StateDictWeightProvider(source[len("state_dict:"):])
    if source.startswith("blitz:"):
        return BlitzCacheWeightProvider(source[len("blitz:"):])
```

### 10.3 SyntheticWeightProvider

Uses caller-supplied generator callables with seeded randomness and optional disk caching:

```python
provider = SyntheticWeightProvider(
    mla_generator=_generate_backed_mla_torch_weights,
    moe_generator=_generate_backed_moe_torch_weights,
    seed=42,
)
mla_weights = provider.get_mla_torch_weights(layer_idx=0, **mla_params)
```

Seeds are deterministic: `torch.manual_seed(seed + layer_idx)`. The `WeightCache` (from test infrastructure) provides disk caching keyed on generator function source + params.

### 10.4 StateDictWeightProvider

Loads real weights from HuggingFace safetensors and applies Blaze-format transforms:

```python
provider = StateDictWeightProvider("/path/to/deepseek-v3")
mla = provider.get_mla_torch_weights(layer_idx=0, num_tp=8, mesh_rows=4)
```

The MLA weight transform pipeline:
1. Extract raw HF tensors by state_dict key
2. Convert FP8 to bfloat16
3. Transpose weight matrices (HF stores `[out, in]`; Blaze needs `[in, out]`)
4. Deinterleave `q_b_proj` (interleaved head dimensions must be separated)
5. Split `kv_b_proj` into `kv_b1` (shared, replicated) and `kv_b2` (TP-sharded)
6. Apply TP slicing for multi-device
7. Generate RoPE constants (cos/sin/trans_mat)

### 10.5 BlitzCacheWeightProvider

Loads pre-fused device tensors from Blitz's tensorbin cache:

```python
provider = BlitzCacheWeightProvider("/path/to/blitz/cache")

# Torch path (reconstructs from tensorbin -> state dict -> Blaze format):
moe = provider.get_moe_torch_weights(layer_idx=0, **params)

# Device path (fast, no torch intermediates):
views = provider.load_overlapped_views(layer_idx=0, device=mesh_device)
q_a_view = views["q_a_proj"]  # OverlappedView, ready for f.cb_from_view()
```

The device path uses Blitz's `load_moe_decoder_layer()` to load pre-fused tensors directly as `OverlappedView`s, skipping all torch intermediates.

### 10.6 MoE Weight Transform Pipeline

The MoE weight transform (`_state_dict_to_moe_tensors()`) is particularly involved:

```
  HuggingFace State Dict
       |
       +--> RMSNorm gamma (pad to embedding_dim_p)
       |
       +--> Gate weight (transpose, unsqueeze, pad)
       |    Gate bias (reshape)
       |
       +--> 256 x Routed experts:
       |       gate_proj.weight.T   -->  stack  -->  shuffle_tiles
       |       up_proj.weight.T     -->  stack  -->  shuffle_tiles
       |       down_proj.weight.T   -->  stack  -->  shuffle_tiles
       |
       +--> Shared experts:
              gate_proj.weight.T  -->  TP-shard across mesh
              up_proj.weight.T    -->  TP-shard across mesh
              down_proj.weight.T  -->  TP-shard across mesh
              gate+up stacked     -->  per-core shard layout
```

The `_shuffle_tensor_tiles()` function reorders tiles within each DRAM bank shard from row-major to column-major order. This is required because the DRAM-streaming matmul reads tiles column-by-column (K-dimension first), and the DRAM bank interleaving distributes columns across banks:

```
  Before shuffle (row-major):       After shuffle (column-major):
  Bank 0: [t00 t01 t02 ...]        Bank 0: [t00 t30 t60 ...]
  Bank 1: [t10 t11 t12 ...]        Bank 1: [t10 t40 t70 ...]
  ...                                ...

  Where tRC = tile at row R, column C within the bank's shard
```

---

## 11. Complete Op Catalog Reference

### 11.1 All FusedOps (49 total)

```
  Model-specific FusedOps:
  --------
  DeepSeek V3/R1:
    mla, pre_sdpa, post_sdpa, distributed_flash_mla,
    distributed_flash_mla_post_sdpa, qa_projection, q_branch,
    q_heads, kv_branch, moe, moe_no_shared, moe_router,
    moe_large_router, routed_expert, shared_expert,
    large_moe, large_moe_no_shared, large_routed_expert,
    allreduce_moe, dense_mlp, dense_mlp_dram, mlp_mixed,
    dense_swiglu, swiglu, down_proj, sparse_layer

  GLM 5.1:
    glm5_fused_proj, glm5_fused_proj_indexer, glm5_q_branch,
    glm5_sdpa_post_sdpa, glm5_q_sdpa_post_sdpa,
    glm5_sdpa_post_sdpa_multi, glm5_q_sdpa_post_sdpa_multi,
    glm_moe, glm_moe_no_shared, glm_moe_router,
    glm_moe_large_router, glm_routed_expert,
    glm_large_moe, glm_large_moe_no_shared, glm_large_routed_expert,
    sparse_gather_flash_mla_scatter_glm

  Cross-model / utility:
    broadcast_rmsnorm, broadcast_rmsnorm_mcast,
    embed_mcast, gqa_pre_sdpa, lm_head_sampling,
    post_dsa, pre_dsa, pre_dsa_attention,
    dsa_attention, dsa_full_attention, dsa_pipeline,
    dsa_retilize_indexer
```

### 11.2 All MicroOps (54 total)

```
  Data movement:
    mcast, gather, gather_reduce, scatter, scatter_raw,
    copy, shard2cb, tile_copy

  Compute:
    matmul, matmul_cb, matmul_fused_act, kn_sliced_matmul,
    dram_streaming_matmul, eltwise_mul, gated_local_reduce,
    gated_reduce, reduce_to_one, clamped_silu

  Normalization / embedding / position:
    rmsnorm, padded_rmsnorm, rope, embedding

  Attention:
    sdpa, sdpa_reduce, flash_mla, sparse_flash_mla, create_q_heads

  MoE gate / routing:
    deepseek_moe_gate, glm_moe_gate, glm_moe_gate_merge,
    distributed_topk

  Tile/format conversion:
    retilize, untilize, q_idx_tilize, tile_row_convert

  CCL / sync:
    all_reduce, ccl_broadcast, barrier_sender, barrier_receiver,
    pipeline_stage_sync, dm_risc_handshake

  DSA:
    dsa_cache_update, dsa_indexer, dsa_kv_prep,
    dsa_sparse_flash_decode, dsa_sparse_gather, dsa_topk

  Housekeeping:
    cb_flush, cb_reconfig, cb_scratch_reset, residual_add,
    argmax, migration
```

---

## Key Takeaways

- **114 ops, 54 MicroOps, 49 FusedOps**: The catalog covers DeepSeek V3, GLM-5.1, and general infrastructure.
- **FusedOps compose via emit() chaining**: Each MicroOp.emit() returns a CBHandle that feeds the next op.
- **Prefixes prevent CT arg collisions**: Hierarchical prefixes (`parent__child.arg`) ensure unique names.
- **Five composition patterns**: Mcast-Matmul-Gather, A/B Branch Parallel Matmul, CCL Sandwiched, Router-Expert, and BroadcastRMSNorm cover the vast majority of FusedOp structures.
- **Multi-device is transparent to most ops**: The compiler injects mesh_coord/mesh_shape; only CCL ops read them.
- **OverlappedViews enable weight packing**: Multiple weight matrices in one L1 buffer, accessed at different byte offsets on different core ranges.
- **Weight providers decouple sources from consumers**: Switch from synthetic to real weights via an env var.

---

## Key Source Paths

| Component              | Path                                              |
|------------------------|---------------------------------------------------|
| Op base classes        | `blaze/blaze_op.py`                               |
| FusedProgram           | `blaze/fused_program.py`                          |
| CT arg system          | `blaze/ct_args.py`                                |
| Compiler               | `blaze/compiler.py`                               |
| CB engine              | `blaze/cb_engine.py`                              |
| Sem engine             | `blaze/sem_engine.py`                             |
| Grid/Role engine       | `blaze/role_engine.py`                            |
| Graph IR               | `blaze/graph.py`                                  |
| Registry               | `blaze/registry.py`                               |
| Weight providers       | `blaze/weight_provider.py`                        |
| CCL setup              | `blaze/ccl.py`                                    |
| Barrier injection      | `blaze/injected_fabric_barrier_builder.py`        |
| Device context         | `blaze/device_context.py`                         |
| Ops directory          | `blaze/ops/` (114 packages)                       |

---

[**Previous:** `02_ccl_in_blaze.md`](./02_ccl_in_blaze.md) | [**Next:** Chapter 5](../ch05_name/index.md)
