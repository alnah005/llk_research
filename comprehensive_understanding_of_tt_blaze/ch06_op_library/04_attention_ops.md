# 06 -- Attention Ops

[<< Compute Ops](03_compute_ops.md) | [Next: MoE Ops >>](05_moe_ops.md)

This section covers every attention-related op: MLA (Multi-Latent Attention) for DeepSeek, DSA (Dynamic Sparse Attention) for GLM-5.1, and the SDPA (Scaled Dot-Product Attention) primitives they build on. The full op trees are documented in the README at `blaze/ops/README.md` and reproduced here with additional detail.

---

## SDPA Primitives

### SDPADecode

**File:** `blaze/ops/sdpa/op.py`
**Class:** `SDPADecode(MicroOp)`
**Name:** `sdpa_decode`

Scaled dot-product attention for GQA and MLA decode. The fundamental attention compute kernel.

| Property | Value |
|----------|-------|
| Math Fidelity | LoFi |
| Supported | Causal, non-paged, HEIGHT_SHARDED Q/output, GQA, MLA (DHt != vDHt), tree reduction |
| Not Supported | Paged attention, half tiles, attention sink, attention mask, K multicast, tilize_q |
| NOC Mode | DM_DYNAMIC_NOC |
| Constants | `TILE_HEIGHT=32`, `TILE_WIDTH=32`, `NUM_DRAM_BANKS=8`, `MAX_TREE_REDUCTION_ROUNDS=6` |

**Execution model:**
- NCRISC: Read K chunks from DRAM
- BRISC: Q input multicast, mask generation, tree reduction
- TRISC: Chunked SDPA (Q x K^T, softmax, x V) with online reduction

---

### FlashMLADecode

**File:** `blaze/ops/flash_mla/op.py`
**Class:** `FlashMLADecode(MicroOp)`
**Name:** `flash_mla`

Flash Multi-Latent Attention decode. K and V share the same tensor (V = first `head_dim_v` elements of KV cache).

| Property | Value |
|----------|-------|
| Grid | 8x8 = 64 cores per device (8 S-blocks x 8 cores) |
| NOC Mode | DM_DYNAMIC_NOC (required) |
| Tree Reduction | 3 rounds |

**S-Block grid:** Hard-coded definitions for SDPA compute grid. Each S-block handles one attention head with K-split parallelism across its 8 cores.

---

### SparseFlashMLADecode

**File:** `blaze/ops/sparse_flash_mla/op.py`
**Class:** `SparseFlashMLADecode(MicroOp)`
**Name:** `sparse_flash_mla`

Reuses the same kernel as FlashMLADecode but with a GLM-specific grid layout. Same C++ kernel (fully CT-arg parameterized).

| Property | FlashMLADecode | SparseFlashMLADecode |
|----------|----------------|---------------------|
| Use Case | DeepSeek V3 MLA full attention | GLM-5.1 sparse attention |
| Grid | 8x8 = 64 cores | GlmSparseGrid (8 cores) or L1SparseGrid (32 cores) |
| KV Source | DRAM KV cache (full sequence) | L1-sharded from DsaSparseGather (topk subset) |
| Q Input | Pre-tilized TILE_LAYOUT | Inline tilize from ROW_MAJOR (`tilize_q=True`) |
| Tree Reduction | 3 rounds | 3 rounds (8 cores) or 5 rounds (32 cores) |

**Inline Q tilize:** When `tilize_q=True`, TRISC on block0 converts ROW_MAJOR Q to face-packed TILE_LAYOUT before SDPA compute, eliminating a separate retilize dispatch.

---

### DsaSparseFlashDecode

**File:** `blaze/ops/dsa_sparse_flash_decode/op.py`
**Class:** `DsaSparseFlashDecode(MicroOp)`
**Name:** `dsa_sparse_flash_decode`

Purpose-built for GLM-5.1 TP=8: 8 heads, k=4096 gathered tokens, 8 DRAM banks. Uses GlmSparseGrid (8 cores, one per DRAM bank).

**GlmSparseGrid:** 8-core grid for GLM-5.1 TP=8 sparse attention. One core per DRAM bank, direct bank-to-core mapping. Each "block" is a single core (no intra-block multicast). Tree reduction: 3 rounds (log2(8)).

---

## MLA (Multi-Latent Attention) -- DeepSeek V3

### MLA (Top-Level FusedOp)

**File:** `blaze/ops/mla/op.py`
**Class:** `MLA(FusedOp)`
**Name:** `mla`

The complete DeepSeek MLA attention layer. Composes three major stages:

```text
MLA
+-- PreSDPA
|   +-- BroadcastRMSNormMcast
|   |   +-- BroadcastRMSNorm
|   |   |   +-- CclBroadcast^
|   |   |   +-- RMSNorm^
|   |   +-- Mcast (mcast1)^
|   +-- KNMatmul (matmul1)^
|   +-- GatherReduce^
|   +-- RMSNorm (rmsnorm2)^
|   +-- Mcast (mcast2)^
|   +-- Matmul (matmul2)^
|   +-- Matmul (matmul3)^
|   +-- Rope (qrope)^
|   +-- CreateQHeads^
|   +-- Matmul (kv_a_proj)^
|   +-- Gather (kv_gather)^
|   +-- RMSNorm (kv_norm)^
|   +-- Rope (krope)^
|   +-- KVCacheUpdate^
+-- DistributedFlashMLA
|   +-- FlashMLADecode^
|   +-- SdpaReduceToAll^
+-- PostSdpa
    +-- Scatter^
    +-- Matmul (kv_b2_matmul)^
    +-- Gather (kv_b2_gather)^
    +-- Mcast (o_proj_mcast)^
    +-- Matmul (o_proj_matmul)^
    +-- Gather (o_proj_gather)^
    +-- AllReduce^
```

---

### PreSDPA

**File:** `blaze/ops/pre_sdpa/op.py`
**Class:** `PreSDPA(FusedOp)`
**Name:** `pre_sdpa`

Pre-Self-Dot-Product Attention. Composes three pipeline stages:

```text
activation -> BroadcastRMSNormMcast -> mcasted -> QBranch -> Q output
                                    |          (QAProjection + QHeads)
                                    +---------> KVBranch -> DRAM KV cache
```

**Sub-op key sets** (for parameter forwarding):
- `_CCL_BCAST_USER_KEYS`: sender_coord, secondary_cluster_axis, num_links, is_torus
- `_QROPE_USER_KEYS`: qrope_cos_addr, qrope_sin_addr, qrope_pos_ids_addr, qrope_Wt, qrope_Ht, qrope_cos_sin_page_size, qrope_total_Wt, qrope_cores
- `_CQH_USER_KEYS`: cqh_receiver_grid, cqh_qnope_data_size_bytes, cqh_qrope_head_size_bytes, cqh_head_stride_bytes, cqh_qnope_cols
- `_KV_USER_KEYS`: kv_a_proj, kv_norm, krope parameters

---

### QBranch

**File:** `blaze/ops/q_branch/op.py`
**Class:** `QBranch(FusedOp)`
**Name:** `q_branch`

Full Q pipeline from mcasted activation to Q heads. Composes QAProjection and QHeads.

```
mcasted -> QAProjection(KNMatmul + GatherReduce + RMSNorm2 + Mcast2) -> QHeads(q_b_proj + kv_b1_proj + QRoPE + CreateQHeads) -> Q
```

---

### QAProjection

**File:** `blaze/ops/qa_projection/op.py`
**Class:** `QAProjection(FusedOp)`
**Name:** `qa_projection`

Q-A projection stage: KNMatmul(q_a_proj, K-split) -> GatherReduce -> RMSNorm2 -> Mcast2.

Input: `mcasted` CBHandle (1x32 tile BF16, full device grid).
Output: `q_a_out` CBHandle (1x32 tile BF16, full device grid).

---

### QHeads

**File:** `blaze/ops/q_heads/op.py`
**Class:** `QHeads(FusedOp)`
**Name:** `q_heads`

Final Q stage. q_b_proj matmul output feeds two parallel sub-stages on disjoint core sets:
- **kv_b1_proj** (QNoPE cores): projects to Q-nope heads
- **QRoPE** (QRoPE cores): applies rotary position embedding

CreateQHeads gathers both streams to receiver cores and tilizes them.

---

### CreateQHeads

**File:** `blaze/ops/create_q_heads/op.py`
**Class:** `CreateQHeads(MicroOp)`
**Name:** `create_q_heads`

Gather QNOPE + QROPE data from 12x8 sender cores to 4x2 receiver cores, then tilize in 3 phases using 3 semaphores for race-free synchronization.

| Property | Value |
|----------|-------|
| Inter-core | True |
| Phases | 3 (noc0_receiver for nope phase1, noc1_receiver for nope phase2, sender for rope) |

---

### KVBranch

**File:** `blaze/ops/kv_branch/op.py`
**Class:** `KVBranch(FusedOp)`
**Name:** `kv_branch`

KV pipeline: kv_a_proj -> Gather -> kv_norm -> KRoPE -> KVCacheUpdate.

Takes the Mcast1 output (not Mcast2) directly from BroadcastRMSNormMcast, projects to KV latent space, normalizes k-nope, applies KRoPE to k-rope, and splices both into the DRAM KV cache. Writes directly to DRAM (no CB output).

---

### PostSdpa

**File:** `blaze/ops/post_sdpa/op.py`
**Class:** `PostSdpa(FusedOp)`
**Name:** `post_sdpa`

MLA output projection after SDPA:

```
kv_b2_matmul -> kv_b2_gather -> o_proj_mcast -> o_proj_matmul -> o_proj_gather -> all_reduce
```

The kv_b2 projection absorbs the latent KV dimension per-head. The o_proj projection maps to the model's embedding dimension. AllReduce across devices sums partial results and fuses a residual add.

---

### DistributedFlashMLA

**File:** `blaze/ops/distributed_flash_mla/op.py`
**Class:** `DistributedFlashMLA(FusedOp)`
**Name:** `distributed_flash_mla`

Fused FlashMLADecode + SdpaReduceToAll. Zero-copy: flash MLA's output CBs (out_o / out_ms) are directly aliased as SDPA reduce's input CBs (local_l / local_ms).

**Constraint:** SDPA reduce worker cores MUST be the same physical cores as flash_mla's S1 block output cores (data stays in L1).

---

### DistributedFlashMLAPostSdpa

**File:** `blaze/ops/distributed_flash_mla_post_sdpa/op.py`
**Class:** `DistributedFlashMLAPostSdpa(FusedOp)`
**Name:** `distributed_flash_mla_post_sdpa`

True single-kernel fusion of the entire MLA decode-to-output pipeline:

```
FlashMLADecode -> SdpaReduceToAll -> Scatter -> kv_b2 matmul -> gather -> mcast -> o_proj matmul -> gather -> AllReduce
```

---

## DSA (Dynamic Sparse Attention) -- GLM-5.1

### DSA Pipeline MicroOps

#### DsaIndexer

**File:** `blaze/ops/dsa_indexer/op.py`
**Class:** `DsaIndexer(MicroOp)`
**Name:** `dsa_indexer`

Per-bank scoring of indexer key cache. Scores all positions using lightweight attention in 128-dim space. Each chunk of 32 positions: QK = Q x K^T -> ReLU -> weighted sum -> scores.

| Mode | Description |
|------|-------------|
| L1 key cache | Bring-up, single core |
| DRAM key cache | Production, single or multi-bank parallel |

Multi-bank mode: N cores score in parallel (one per DRAM bank partition), each reading its slice of K cache via TensorAccessor.

`dsa_compute_grid()` helper: Up to 8 cores with DRAM-optimal placement (one adjacent to each bank). Beyond 8 cores: non-optimal placement for extra compute.

---

#### DsaTopk

**File:** `blaze/ops/dsa_topk/op.py`
**Class:** `DsaTopk(MicroOp)`
**Name:** `dsa_topk`

Hierarchical top-k selection across sharded cores.

| Phase | Description |
|-------|-------------|
| Phase 1 | Each active core packs local (score, index) candidates, NOC-writes to coordinator |
| Phase 2 | Coordinator runs min-heap selection over all candidates, writes global top-k indices |

Candidate size: 8 bytes each (score_f32: 4B + global_index_u32: 4B).

---

#### DsaSparseGather

**File:** `blaze/ops/dsa_sparse_gather/op.py`
**Class:** `DsaSparseGather(MicroOp)`
**Name:** `dsa_sparse_gather`

Index-gather selected KV rows from cache.

| Mode | Description |
|------|-------------|
| L1 path (`use_dram=False`) | KV cache in L1, direct NOC copy |
| DRAM path (`use_dram=True`) | KV cache in DRAM (ROW_MAJOR interleaved), TensorAccessor reads |

Optional tilize_output mode: NCRISC gathers ROW_MAJOR KV rows, TRISC tilizes via tilize_block, BRISC writes tiles to DRAM/L1. Tilize is always fused into the gather kernel.

---

#### DsaKvPrep

**File:** `blaze/ops/dsa_kv_prep/op.py`
**Class:** `DsaKvPrep(MicroOp)`
**Name:** `dsa_kv_prep`

Tilize gathered KV and write **separate** K/V tensors to DRAM:
- K: `[1, 1, k, DH]` TILE_LAYOUT DRAM INTERLEAVED
- V: `[1, 1, k, vDH]` TILE_LAYOUT DRAM INTERLEAVED

Output dtype configurable: bfloat16 (bring-up) or bfloat8_b (production).

---

### DSA FusedOps

#### DsaPipeline

**File:** `blaze/ops/dsa_pipeline/op.py`
**Class:** `DsaPipeline(FusedOp)`
**Name:** `dsa_pipeline`

Fused DSA selection pipeline (single dispatch):

```text
DsaPipeline
+-- DsaIndexer^ (8-bank DRAM-parallel scoring)
+-- DsaTopk^ (min-heap top-k selection)
+-- DsaSparseGather^ (index-gather KV rows, tilize_output=True)
```

Output: `[1,1,k,576]` bfloat8_b ND-sharded DRAM.

---

#### DsaSparseAttention

**File:** `blaze/ops/dsa_attention/op.py`
**Class:** `DsaSparseAttention(FusedOp)`
**Name:** `dsa_sparse_attention`

Full sparse attention in one dispatch:

```
DsaIndexer -> DsaTopk -> DsaSparseGather -> DsaKvPrep -> [CbReconfig] -> SDPADecode -> PostSdpa
```

---

#### DsaFullAttention

**File:** `blaze/ops/dsa_full_attention/op.py`
**Class:** `DsaFullAttention(FusedOp)`
**Name:** `dsa_full_attention`

GLM-5 single-dispatch full attention layer:

```
RMSNorm -> Mcast -> PreDsa(indexer projections) -> DsaIndexer -> DsaTopk -> DsaSparseGather -> DsaKvPrep -> SDPADecode -> PostDsa(V absorption)
```

No CbReconfig needed -- all scratch CBs carved from one pre-allocated SDPA scratchpad tensor via `cb_from_tensor_overlapped()`.

---

#### DsaRetilizeIndexer

**File:** `blaze/ops/dsa_retilize_indexer/op.py`
**Class:** `DsaRetilizeIndexer(FusedOp)`
**Name:** `dsa_retilize_indexer`

Fused QIdxTilize -> Mcast -> DsaIndexer.

QIdxTilize converts Q_idx from (1,32) row-major tile order to (32,32) face-packed on the sender core. Mcast delivers the 4 face-packed (32,32) Q tiles to all indexer cores.

---

### DSA Pre/Post Processing Ops

#### PreDsa

**File:** `blaze/ops/pre_dsa/op.py`
**Class:** `PreDsa(FusedOp)`
**Name:** `pre_dsa`

GLM-5 MLA indexer projections + KV cache update. All projections branch from the same Mcast-broadcast activation:

```
activation -> Matmul(idx.wk) -> k_new -> indexer cache update
activation -> Matmul(idx.wq_b) -> Q_idx
activation -> Matmul(idx.weights_proj) -> weights
activation -> Matmul(wkv_a) -> KV cache update
```

---

#### PreDsaAttention

**File:** `blaze/ops/pre_dsa_attention/op.py`
**Class:** `PreDsaAttention(FusedOp)`
**Name:** `pre_dsa_attention`

Pre-DSA indexer projection matmuls only (without KV cache update):

```
activation -> Matmul(idx.wk) -> k_new
activation -> Matmul(idx.wq_b) -> Q_idx
activation -> Matmul(idx.weights_proj) -> weights
```

---

#### PostDsa

**File:** `blaze/ops/post_dsa/op.py`
**Class:** `PostDsa(FusedOp)`
**Name:** `post_dsa`

GLM-5 MLA V absorption + output projection:

```
sdpa_output [H, 1, c_dim=512]
  -> Matmul(wkv_b_v): per-head [c_dim] -> [v_dim=256]  (V absorption)
  -> Matmul(wo): [H * v_dim] -> [hidden_dim=6144]       (output projection)
```

GLM-5 MLA dimensions: c_dim=512, v_dim=256, num_heads=64 (8 at TP=8), hidden_dim=6144.

---

## GLM-5.1 TP=8 Dispatch-Level Ops

These are dispatch-level wrappers for the GLM-5.1 TP=8 sparse attention block. Unlike FusedOps, each manages its own FusedProgram lifecycle.

### GlmFusedProj (Dispatch 0)

**File:** `blaze/ops/glm5_fused_proj/op.py`
**Class:** `GlmFusedProj(FusedOp)`
**Name:** `glm5_fused_proj`

One FusedProgram with parallel matmuls on disjoint core grids:

```text
GlmFusedProj
+-- Mcast (hidden -> all matmul cores)^
+-- Matmul (W_idx_q, Q_idx projection)^     --+
+-- Matmul (W_q_a, Q_a projection)^           | parallel on
+-- Matmul (W_idx_k, k_new projection)^       | disjoint cores
+-- Matmul (W_idx_w, weights projection)^    --+
+-- Gather (Q_idx -> sender)^
+-- Gather (Q_a -> sender)^
+-- Gather (k_new -> sender)^
+-- Gather (weights -> sender)^
```

All weights packed in one bfloat8_b fused tensor with OverlappedViews (Pattern A). Gather outputs packed in one sender scratch tensor (Pattern B).

When `keep_mcast=True`, the Mcast CB persists for downstream KVBranch consumption.

---

### GlmFusedProjIndexer

**File:** `blaze/ops/glm5_fused_proj_indexer/op.py`
**Class:** `GlmFusedProjIndexer`
**Name:** (not a registered BlazeOp -- utility class)

Full D0 projections + indexer in one dispatch:

```
Mcast(hidden) -> 4x parallel Matmul -> 4x Gather -> QIdxTilize(distribute) -> DsaIndexer -> scores
```

---

### GlmQBranch (Dispatch 1)

**File:** `blaze/ops/glm5_q_branch/op.py`
**Class:** `GlmQBranch(FusedOp)`
**Name:** `glm5_q_branch`

Single FusedProgram chain:

```text
GlmQBranch
+-- RMSNorm (Q_a -> Q_a_normed)^
+-- Mcast (Q_a_normed -> nope + head cores)^
+-- Matmul (W_nope, on nope cores)^
+-- Gather (Q_nope -> sender)^
+-- BarrierSender (sender -> head cores)^
+-- BarrierReceiver (head cores wait)^
+-- Shard2CB (Q_nope per-head from sender to head cores)^
+-- MatmulCB (Q_nope x Wkv -> absorption, on head cores)^
+-- MatmulCB (Q_a_normed x W_rope -> rope, on head cores)^
+-- Gather (Q -> output core)^
```

W_nope on 48 separate cores (6/head for nope_dim=192). Wkv + W_rope on 8 head cores. Shard2CB bridges Gather to per-head consumption.

---

### GLM-5.1 Fused Attention Dispatches

#### SDPAPostSDPA

**File:** `blaze/ops/glm5_sdpa_post_sdpa/op.py`
**Class:** `SDPAPostSDPA`

Single-dispatch: Sparse SDPA -> PostDsa Matmul -> Wo Matmul -> output.

```text
SparseGatherFlashMLAScatterGLM (64 flash cores)
  -> ScatterRaw (S1 -> PostDsa cores)
  -> Matmul(PostDsa, 64 non-flash cores)
  -> Gather -> Mcast -> Matmul(Wo, 96 cores)
  -> Gather -> output
```

Uses DM_DYNAMIC_NOC. PostDsa cores exclude all 64 flash cores to avoid L1 CB overlap.

---

#### QSDPAPostSDPA

**File:** `blaze/ops/glm5_q_sdpa_post_sdpa/op.py`
**Class:** `QSDPAPostSDPA`

Single-dispatch with Q branch: Q Branch -> Sparse SDPA -> PostDsa -> Wo.

---

#### SDPAPostSDPA_Multi

**File:** `blaze/ops/glm5_sdpa_post_sdpa_multi/op.py`
**Class:** `SDPAPostSDPA_Multi`

Multi-device: Sparse SDPA -> PostDsa -> Wo -> Scatter -> ReduceToOne -> CclBroadcast.

---

#### QSDPAPostSDPA_Multi

**File:** `blaze/ops/glm5_q_sdpa_post_sdpa_multi/op.py`
**Class:** `QSDPAPostSDPA_Multi`

Multi-device with Q branch: Q Branch -> Sparse SDPA -> PostDsa -> Wo -> Scatter -> ReduceToOne -> CclBroadcast.

---

### SparseGatherFlashMLAScatterGLM

**File:** `blaze/ops/sparse_gather_flash_mla_scatter_glm/op.py`
**Class:** `SparseGatherFlashMLAScatterGLM(FusedOp)`
**Name:** `sparse_gather_flash_mla_scatter_glm`

Fused SDPA pipeline for GLM-5.1:

```
DsaSparseGather -> Barrier -> FlashMLADecode -> ScatterRaw
```

1. DsaSparseGather: DRAM KV cache -> L1 (sparse position indices)
2. Barrier: gather cores -> FlashMLADecode mcast_sender cores
3. FlashMLADecode: Q x K -> softmax -> x V on 64 cores (8 heads x 8 K-splits)
4. ScatterRaw: S1 output -> PostDsa matmul cores

Exposes `get_post_dsa_grid()` to allocate PostDsa cores that avoid all flash cores.

---

## Composite Layer-Level Ops

### SparseLayer

**File:** `blaze/ops/sparse_layer/op.py`
**Class:** `SparseLayer(FusedOp)`
**Name:** `sparse_layer`

Full sparse transformer layer: MLA + CbReconfig + MoE in one FusedOp.

```
MLA (attention) -> CbReconfig -> MoE (feed-forward)
```

Uses `noc_mode = ttnn.NOC_MODE.DM_DYNAMIC_NOC`.

---

### AllReduceMoE

**File:** `blaze/ops/allreduce_moe/op.py`
**Class:** `AllReduceMoE(FusedOp)`
**Name:** `allreduce_moe`

Fused AllReduce + MoE: TP all-reduce followed by MoE in a single program.

```
AllReduce (ring sum + residual) -> MoE (RMSNorm -> Mcast -> RoutedExpert + SharedExpert -> ResidualAdd -> ReduceToOne)
```

AllReduce writes the reduced activation via a 32x32 CB. MoE derives 1x32 tile info internally for Mcast dst and downstream ops.

---

## Complete Two-Dispatch Architecture (GLM-5.1 TP=8)

```text
Dispatch 1: DsaPipeline (BlazeCompiler)
  DsaIndexer -> DsaTopk -> DsaSparseGather -> DsaSparseGather(tilize_output=True)
  Output: [1,1,k,576] bfloat8_b ND-sharded DRAM

Dispatch 2: DsaSparseFlashDecode (BlazeCompiler)
  8-core flash attention on gathered KV
  Output: [1,1,H,512] bfloat16 L1
```

---

## GLM-5.1 Full Attention Block (TP=8)

```text
hidden [1, hidden_dim]
  |
  v GlmFusedProj          Mcast -> 4xMatmul -> 4xGather (parallel)
  +- k_new (CPU)
  +- Q_idx (device) -----------------------------------------------+
  +- weights (CPU)                                                  | DsaPipeline
  +- Q_a (device) --+                                              | (indexer)
                    |                                              |
  v GlmQBranchFlash |  +- Q branch + FlashDecode ---------------+  |
                    |  |                                         |  |
  Q_a -------------+  |  RMSNorm -> Mcast -> Matmul(nope)       |  |
                      |  -> Gather -> Barrier -> Shard2CB        |  |
                      |  -> MatmulCB(absorb) -> MatmulCB(rope)  |  |
                      |  -> Gather(Q -> block0)                  |  |
                      |     | CbReconfig                         |  |
                      |  DsaSparseFlashDecode(Q, KV -> O)        |  |
                      +------------------------------------------+  |
                   KV [1,1,k,kv_dim] <-----------------------------+
                      |
                SDPA output [H, v_dim]
                      |
  v GlmPostDsaWo      PostDsa -> Gather -> Mcast -> Matmul(Wo) -> Gather
                      |
                output [1, hidden_dim]
```

---

## Complete Attention Op Reference Table

| Op Name | Type | Family | Key Function |
|---------|------|--------|-------------|
| `sdpa` | MicroOp | Core | Scaled dot-product attention decode |
| `flash_mla` | MicroOp | MLA | Flash MLA decode (64-core grid) |
| `sparse_flash_mla` | MicroOp | DSA | Sparse flash MLA (GLM grid) |
| `dsa_sparse_flash_decode` | MicroOp | DSA | 8-core sparse flash decode |
| `sdpa_reduce_to_all` | MicroOp | Multi-device | Ring SDPA reduction |
| `create_q_heads` | MicroOp | MLA | Q head assembly (3-phase gather+tilize) |
| `kv_cache_update` | MicroOp | MLA | KV cache splice (read-modify-write) |
| `dsa_indexer` | MicroOp | DSA | Multi-bank scoring |
| `dsa_topk` | MicroOp | DSA | Hierarchical top-k selection |
| `dsa_sparse_gather` | MicroOp | DSA | Index-gather KV from cache |
| `dsa_kv_prep` | MicroOp | DSA | Tilize + split K/V to DRAM |
| `dsa_cache_update` | MicroOp | DSA | Indexer key cache write |
| `q_idx_tilize` | MicroOp | DSA | Q_idx tile-order transpose |
| `tile_row_convert` | MicroOp | DSA | TopK IDs to tile-row IDs |
| `mla` | FusedOp | MLA | Complete MLA layer |
| `pre_sdpa` | FusedOp | MLA | PreSDPA pipeline |
| `post_sdpa` | FusedOp | MLA | PostSDPA output projection |
| `q_branch` | FusedOp | MLA | Full Q pipeline |
| `qa_projection` | FusedOp | MLA | Q-A projection stage |
| `q_heads` | FusedOp | MLA | Q head assembly stage |
| `kv_branch` | FusedOp | MLA | KV projection + cache update |
| `distributed_flash_mla` | FusedOp | MLA | Flash MLA + multi-device reduce |
| `distributed_flash_mla_post_sdpa` | FusedOp | MLA | Full MLA decode + output proj |
| `dsa_pipeline` | FusedOp | DSA | Indexer -> topk -> gather -> prep |
| `dsa_sparse_attention` | FusedOp | DSA | Full DSA + SDPA + PostSdpa |
| `dsa_full_attention` | FusedOp | DSA | GLM-5 single-dispatch attention |
| `dsa_retilize_indexer` | FusedOp | DSA | QIdxTilize -> Mcast -> Indexer |
| `pre_dsa` | FusedOp | DSA | GLM-5 indexer projections |
| `pre_dsa_attention` | FusedOp | DSA | Indexer projection matmuls |
| `post_dsa` | FusedOp | DSA | V absorption + output projection |
| `glm5_fused_proj` | FusedOp | GLM-5.1 | 4-way parallel projection |
| `glm5_q_branch` | FusedOp | GLM-5.1 | Q branch with absorption |
| `sparse_layer` | FusedOp | Layer | MLA + CbReconfig + MoE |
| `allreduce_moe` | FusedOp | Layer | AllReduce + MoE |
| `sparse_gather_flash_mla_scatter_glm` | FusedOp | GLM-5.1 | Gather + FlashMLA + Scatter |
| `gqa_pre_sdpa` | FusedOp | GQA | GQA Q-projection |
| `distributed_topk` | MicroOp | DSA | Multi-device top-k |

[<< Compute Ops](03_compute_ops.md) | [Next: MoE Ops >>](05_moe_ops.md)
