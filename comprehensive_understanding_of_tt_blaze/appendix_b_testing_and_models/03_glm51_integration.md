# Appendix B.3 -- GLM-5.1 Model Integration

This section documents the GLM-5.1 ops in `blaze/ops/`, the Dense Sparse
Attention (DSA) architecture, differences from DeepSeek V3, and the dedicated
test coverage.

---

## 1. GLM-5.1 Op Catalog

GLM-5.1 ops are spread across 18 op directories under `blaze/ops/`. They fall
into three categories: attention (DSA-based), MoE routing, and MoE execution.

### 1.1 Attention Ops

| Op Directory | Class | Purpose |
|-------------|-------|---------|
| `glm5_fused_proj/` | `GlmFusedProj` | D0: Mcast(hidden) -> 4x parallel Matmul -> 4x Gather |
| `glm5_fused_proj_indexer/` | `GlmFusedProjIndexer` | D0 + indexer: projections + QIdxTilize + DsaIndexer |
| `glm5_q_branch/` | `GlmQBranch` | D1: RMSNorm -> Mcast -> Nope Matmul -> Shard2CB -> Absorption -> Rope -> Gather |
| `glm5_sdpa_post_sdpa/` | `SDPAPostSDPA` | D2: Sparse SDPA + PostDsa + Wo in one dispatch |
| `glm5_sdpa_post_sdpa_multi/` | (multi-device) | D2 variant for multi-device mesh execution |
| `glm5_q_sdpa_post_sdpa/` | (fused Q+SDPA) | Combined D1+D2: Q branch + SDPA + PostDsa + Wo |
| `glm5_q_sdpa_post_sdpa_multi/` | (multi-device) | Multi-device variant of the combined D1+D2 |

### 1.2 MoE Routing Ops

| Op Directory | Class | Purpose |
|-------------|-------|---------|
| `glm_moe_gate/` | `GLMMoeGate` | Top-k gate with sigmoid + normalization (up to 256 experts) |
| `glm_moe_gate_merge/` | `GLMMoeGateMerge` | Two-face gate merge for >256 experts |
| `glm_moe_router/` | `GLMMoERouter` | Full router: gate + indexed expert selection |
| `glm_moe_large_router/` | (large router) | Router variant for >256 experts |

### 1.3 MoE Execution Ops

| Op Directory | Class | Purpose |
|-------------|-------|---------|
| `glm_routed_expert/` | `GLMRoutedExpert` | Router + indexed SwiGLU (up to 256 experts) |
| `glm_large_routed_expert/` | `GLMLargeRoutedExpert` | Two-face routing for >256 experts |
| `glm_moe/` | `GLMMoE` | Full MoE: shared + routed + ReduceToOne |
| `glm_moe_no_shared/` | `GLMMoENoShared` | MoE variant without shared expert |
| `glm_large_moe/` | `GLMLargeMoE` | Full MoE for >256 experts with shared expert |
| `glm_large_moe_no_shared/` | (large no-shared) | >256 experts without shared expert |

---

## 2. Dense Sparse Attention (DSA) Architecture

GLM-5.1 uses a fundamentally different attention mechanism from DeepSeek V3's
MLA. While MLA projects into a compressed latent space, DSA uses a sparse
indexer to select which KV cache pages to attend to.

### 2.1 Three-Dispatch Attention Pipeline

The GLM-5.1 attention is decomposed into three dispatches (D0, D1, D2),
each implemented as a FusedProgram:

**Dispatch 0 (GlmFusedProj / GlmFusedProjIndexer):**
```
hidden -> Mcast -> 4x parallel Matmul -> 4x Gather
                   |-- W_idx_q  -> Q_idx    (indexer Q projection)
                   |-- W_q_a    -> Q_a      (Q_a projection)
                   |-- W_idx_k  -> k_new    (indexer K projection, to CPU)
                   |-- W_idx_w  -> weights  (indexer weight projection, to CPU)
```

All four weight matrices are packed into a single `bfloat8_b` fused tensor
with `OverlappedView`s on disjoint cores. After one Mcast of the hidden state,
all four matmuls execute in parallel. When fused with the indexer
(`GlmFusedProjIndexer`), the dispatch continues with QIdxTilize and DsaIndexer
to produce sparse attention scores.

**Dispatch 1 (GlmQBranch):**
```
Q_a -> RMSNorm -> Mcast -> Nope Matmul (48 cores) -> Gather to sender
    -> Shard2CB (per-head) -> Absorption: Q_nope x Wkv -> q_out[0:c_dim]
    -> Rope: Q_a_normed x W_rope -> q_out[c_dim:kv_dim]
    -> Gather/ScatterRaw -> Q on output core(s)
```

The Q branch uses a barrier-synchronized handoff: BarrierSender signals head
cores once the nope Gather is complete, then BarrierReceiver on head cores
gates the Shard2CB fetch. Weights are fused into a single tensor with stacked
OverlappedViews (rope at offset 0, Wkv stacked after rope on head cores,
nope at offset 0 on disjoint nope cores).

**Dispatch 2 (SDPAPostSDPA):**
```
Q -> Sparse SDPA (64 flash cores, DsaSparseGather) -> ScatterRaw
  -> PostDsa Matmul (64 non-flash cores) -> Wo Matmul (96 cores) -> output
```

Uses `DM_DYNAMIC_NOC` for the SparseFlashMLADecode kernel. PostDsa cores are
explicitly excluded from the flash grid to avoid L1 CB overlap.

### 2.2 Core Allocation Strategy

The GlmFusedProj `run()` method dynamically allocates cores across the four
projections, reserving small-weight projections first:

```python
k_new_nc = pick_matmul_cores(k_new_N, d0_budget)
wts_nc = pick_matmul_cores(wts_N, d0_budget)
big_budget = d0_budget - k_new_nc - wts_nc
# Split remaining cores between Q_idx and Q_a
q_idx_nc = pick_matmul_cores(q_idx_N, big_budget // 2)
q_a_nc = pick_matmul_cores(q_a_N, big_budget - q_idx_nc)
```

When KV branch integration is enabled (W_kv_a provided), a 5th parallel
matmul chain is appended for the kv_a projection, followed by RMSNorm + RoPE
+ KVCacheUpdate, all within the same FusedProgram.

---

## 3. GLM MoE Gate -- Hierarchical Top-K

GLM-5.1 uses a distinct gating mechanism from DeepSeek V3:

| Feature | DeepSeek V3 | GLM-5.1 |
|---------|------------|---------|
| Grouping | 8 groups, top-k within groups | Global top-k (n_group=1) |
| Activation | Softmax | Sigmoid |
| Score normalization | No | Yes (normalize=True) |
| Max experts per gate op | 256 | 256 (>256 via two-face merge) |
| Gate type label | "DeepSeek" / "C2" | "B1" / "C1" / "C0" |

The `GLMMoeGate` micro-op operates on 16x16 tiles (one face). For models
with >256 experts, `GLMMoeGateMerge` runs two gate passes (face A: experts
0-255, face B: 256+) then merges the top-k results.

```python
class GLMMoeGate(MicroOp):
    """GLM MoE gate micro-op for a single 16x16 face per core.
    GLM uses C1 gate type with:
    - Global top-k across all 256 experts (no grouping like DeepSeek)
    - Sigmoid activation on logits
    - Score normalization (normalize=True for C1)
    """
```

The `create_gate_tensors` helper handles the tensor layout for both <=256 and
>256 expert configurations, packing scores into 16x16 faces with appropriate
padding.

---

## 4. GLM MoE Fused Op

`GLMMoE` in `blaze/ops/glm_moe/op.py` assembles the full MoE block:

```
activation -> RMSNorm -> Mcast
    |-- GLMRoutedExpert (gate + indexed SwiGLU) -> routed_out
    |-- SharedExpert (gate_up + down matmuls) -> shared_out
    -> Mcast(shared_out) -> ResidualAdd(routed_out, shared_out)
    -> ReduceToOne (cross-device all-reduce)
```

Key parameters exposed through `compose()`:

```python
eps=ua.get("eps", 1e-20),
routing_scaling_factor=ua.get("routing_scaling_factor", 2.5),
normalize=ua.get("normalize", True),
num_selected_experts=ua.get("num_selected_experts", 8),
num_experts=ua.get("num_experts", 256),
```

The `index_offset` is computed from the mesh coordinate so each device in the
mesh handles a different expert subset:

```python
row, col = f.mesh_coord
rows, cols = f.mesh_shape
index_offset = row * cols + col + routed_index_offset
```

For >256 experts, the `emit()` method calls `_prepare_bias_faces` to split
the gate bias into face A and face B tensors before passing to
`GLMRoutedExpert`.

---

## 5. Differences from DeepSeek V3

### 5.1 Attention Architecture

| Aspect | DeepSeek V3 (MLA) | GLM-5.1 (DSA) |
|--------|-------------------|---------------|
| Mechanism | Multi-head Latent Attention with compressed KV | Dense Sparse Attention with indexer |
| KV cache | Compressed latent vectors in DRAM | Full KV pages, sparsely selected |
| Q branch | Q_a -> RMSNorm -> Q_nope + Q_rope | Same structure, adds absorption step |
| Dispatch count | 2 (pre-SDPA + SDPA+post-SDPA) | 3 (D0 projections, D1 Q branch, D2 SDPA+post) |
| SDPA kernel | FlashMLADecode | SparseGatherFlashMLAScatterGLM |
| Indexer | None | DsaIndexer produces sparse page indices |

### 5.2 MoE Architecture

| Aspect | DeepSeek V3 | GLM-5.1 |
|--------|------------|---------|
| Gate style | Group-based top-k | Global top-k |
| Expert count | 256 | 256 (or >256 with two-face) |
| Selected experts | 8 | 8 (configurable) |
| Shared expert | 1 | 0 or 1 (model-dependent) |
| Routing op | `MoERouter` | `GLMMoERouter` |
| Expert op | `SwigluOp` (indexed) | `SwigluOp` (indexed, same) |

### 5.3 Weight Fusion

Both models use OverlappedView weight fusion, but GLM-5.1 introduces a
more aggressive pattern in D0 where 4-5 weight matrices share one backing
tensor on disjoint cores (Pattern B in `GlmFusedProj`), while D1 uses
stacked OverlappedViews on overlapping cores (rope + Wkv at different byte
offsets on the same head cores).

---

## 6. Test Coverage

### 6.1 Micro-Op Tests (Silicon)

Located in `tests/blaze/micro-ops/`:

- `test_glm5_fused_proj.py` -- D0 projection accuracy
- `test_glm5_q_branch.py` -- D1 Q branch with barrier synchronization
- `test_glm5_kv_branch.py` -- KV branch integration
- `test_post_dsa_glm5.py` -- PostDsa matmul
- `test_wo_glm5.py` -- Wo output projection
- `test_dsa_multi_dispatch_glm5.py` -- Full 3-dispatch sequence
- `test_dsa_flash_decode_32core.py` -- Sparse flash decode at 32 cores
- `dsa/test_dsa_sparse_gather_tilize.py` -- Sparse gather + tilize
- `dsa/test_dsa_sparse_gather_bf8.py` -- Sparse gather in bfloat8

### 6.2 Fused Op Tests (Loudbox)

Located in `tests/blaze/fused_ops/glm5/`:

- `test_glm5_fused.py` -- Full D0+D1 fused sequence
- `test_glm5_fused_multi.py` -- Multi-device D0+D1
- `test_glm5_proj_tilize_fused.py` -- D0 with tilize
- `test_glm5_q_branch.py` -- Q branch as FusedOp
- `test_glm5_q_fused.py` -- Fused Q+SDPA
- `test_glm5_q_fused_multi.py` -- Multi-device fused Q+SDPA

### 6.3 GLM-5.1 MoE Tests

Located in `tests/blaze/glm5_1/`:

- `test_glm5_moe.py` -- Full GLM MoE fused op
- `test_glm5_moe_router.py` -- GLM MoE router
- `test_glm5_routed_expert.py` -- Routed expert (gate + SwiGLU)
- `test_glm5_shared_expert.py` -- Shared expert pathway
- `micro-ops/test_glm5_moe_gate.py` -- GLM gate micro-op

### 6.4 Generality Tests

The `tests/blaze/generality/test_moe_all_models.py` parametrizes GLM models
(gate types B1, C1, C0) through `blaze.glm_moe`, exercising the GLM MoE
pathway across architectures like Arcee Trinity, GLM-4, Hunyuan, and others
from the model_specs_table.yaml catalog.

---

## 7. Cross-References

- The OverlappedView and FusedProgram APIs used in all GLM ops are documented
  in Chapter 3 (FusedProgram Internals).
- The Mcast, Matmul, Gather, and RMSNorm primitives composed by these ops are
  covered in Chapter 6 (Op Catalog).
- DeepSeek V3 pipeline integration is in Appendix B.2.
- The Loudbox CI buckets that run GLM tests are in Appendix B.1 Section 8.3.
