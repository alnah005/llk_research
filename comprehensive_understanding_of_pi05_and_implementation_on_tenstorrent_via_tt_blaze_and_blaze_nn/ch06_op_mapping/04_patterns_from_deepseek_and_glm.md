# 04: Patterns from DeepSeek V3 and GLM-5.1

## Context

TT-Blaze was built primarily for DeepSeek V3 and GLM-5.1 (ChatGLM). pi0.5 shares significant architectural DNA with these models (Gemma-family transformer, GQA, gated FFN) but omits their most complex feature: Mixture of Experts. This file catalogs the DeepSeek V3 and GLM-5.1 patterns that transfer to pi0.5, the patterns that do not apply, and the key differences that drive pi0.5-specific work.

---

## DeepSeek V3 Patterns

### 1. Dense Layer Pipeline (Directly Transferable)

**TT-Blaze implementation:** `DenseMLP` FusedOp (`blaze/ops/dense_mlp/op.py`)

The DenseMLP pipeline is:
```
RMSNorm -> Mcast -> SharedExpert(KNMatmul -> GatedReduce -> Mcast -> DownProj)
```

This is composed from:
- `emit_mlp_preamble()`: RMSNorm + Mcast of activation to compute cores.
- `SharedExpert.emit()`: KNMatmul (gate+up), GatedReduce, Mcast of intermediate, DownProj (with residual_add built in).

**Transfer to pi0.5:** This is the exact pipeline for pi0.5's Gemma FFN, with two changes:
1. Activation: GELU instead of SiLU (handled by `activation="gelu"` parameter -- see ADAPT tier).
2. Normalization: adaRMSNorm for the action expert (requires replacing `emit_mlp_preamble` for expert 1).

For the PaliGemma expert (expert 0), the pipeline transfers directly with only the GELU change.

**Code reference:** The `DenseMLP.emit()` method shows the composition:
```python
# blaze/ops/dense_mlp/op.py
def emit(f, act, rmsnorm_gamma, gate_up_weight, down_weight, ...):
    act_mcast_handle, act_ti, act_mcast_pages = emit_mlp_preamble(
        f, act, rmsnorm_gamma, prefix, rmsnorm_epsilon)
    return SharedExpert.emit(
        f, act_mcast_handle, gate_up_weight, down_weight, act, ...)
```

For pi0.5, the action expert variant would look like:
```python
# Pseudocode for pi0.5 action expert FFN
def emit_pi05_action_ffn(f, act, ada_norm_cond, gate_up_weight, down_weight, ...):
    normed, gate = adaRMSNorm.emit(f, act, ada_norm_cond, ...)
    mcast_handle = Mcast.emit(f, normed, ...)
    ffn_out = SharedExpert.emit(
        f, mcast_handle, gate_up_weight, down_weight,
        act, ..., activation="gelu")
    output = gated_residual.emit(f, act, ffn_out, gate, ...)
    return output
```

---

### 2. Pipeline Stages and Dispatch Organization (Conceptually Transferable)

**DeepSeek V3 pattern:** The model is decomposed into dispatch stages (D0, D1, D2, ...) that map to separate FusedPrograms. Each FusedProgram is a self-contained compute graph with explicit CB allocation and teardown.

**GLM-5.1 pattern:** `GlmFusedProj` (Dispatch 0) demonstrates fusing multiple parallel matmuls into one FusedProgram:
```
Mcast(hidden) -> N x Matmul (parallel on disjoint core grids) -> N x Gather
```

This pattern runs all Q/K/V/index projections simultaneously after a single Mcast, achieving `kernel_time = max(matmul_time)` instead of `sum(matmul_times)`.

**Transfer to pi0.5:** pi0.5's dual-expert architecture naturally decomposes into:

**Dispatch 0: Prefix embedding (runs once)**
- SigLIP image encoding (27 ViT layers)
- Token embedding for language inputs
- Prefix attention fill (all 18 layers, PaliGemma expert only)
- KV cache write

**Dispatch 1: Suffix embedding (runs per denoising step)**
- Action projection (`action_in_proj`)
- Timestep embedding + time MLP
- AdaRMSNorm conditioning preparation

**Dispatch 2: Transformer layers (runs per denoising step, 18 iterations or fused)**
- For each layer:
  - RMSNorm / adaRMSNorm
  - Q/K/V projections (parallel matmuls -- GLM5 pattern)
  - RoPE
  - SDPA with KV cache
  - Output projection
  - Residual / gated residual
  - RMSNorm / adaRMSNorm
  - FFN (DenseMLP with GELU)
  - Residual / gated residual

**Dispatch 3: Action decoding (runs per denoising step)**
- Final norm
- Action output projection (`action_out_proj`)

The GLM5 parallel matmul pattern is directly applicable to pi0.5's Q/K/V projection:
```
Mcast(hidden) -> [Q_matmul, K_matmul, V_matmul] (parallel) -> [Q_gather, KV_gather]
```

Since pi0.5 uses GQA (8 Q heads, 1 KV head), the Q matmul is larger than K/V. This is exactly the asymmetric projection pattern that GlmFusedProj handles.

---

### 3. Weight Providers and DRAM Tensor Management (Directly Transferable)

**DeepSeek V3 pattern:** Weights are stored in DRAM-interleaved tensors with WIDTH_SHARDED memory configs. The FusedProgram system manages:
- `cb_from_tensor()`: Registers a DRAM tensor and creates a CB for reading it.
- `OverlappedView`: Maps multiple logical weight tensors onto disjoint regions of a single physical DRAM tensor.
- DRAM streaming: `DRAMStreamingMatmul` reads weight tiles from DRAM on-the-fly during computation.

**Transfer to pi0.5:** pi0.5 has straightforward weight management:
- **PaliGemma weights (expert 0):** Pre-trained, frozen during fine-tuning. Can be stored in BFP8 for memory efficiency.
- **Action expert weights (expert 1):** Smaller (1024-wide instead of 2048-wide). Potentially fit in L1 for L1-resident matmul.
- **Shared embedding table:** 257,152 x 2048 = ~500M elements. Must be DRAM-interleaved.

The OverlappedView pattern from GlmFusedProj is useful for packing multiple projection weights (Q, K, V, FFN gate, FFN up) into a single fused tensor:

```python
# From glm5_fused_proj/op.py, demonstrating OverlappedView for parallel matmuls:
fused_cpu = torch.zeros(fused_h, target_w * total_cores, dtype=torch.bfloat16)
# Fill each projection's weights into disjoint column ranges
_fill_width(W_idx_q, q_idx_npc, q_idx_start, q_idx_nc)
_fill_width(W_q_a, q_a_npc, q_a_start, q_a_nc)
```

For pi0.5, each layer's projection weights can be fused into a single DRAM tensor with OverlappedViews pointing to Q, K, V sub-regions.

---

### 4. KV Cache Management (Directly Transferable)

**DeepSeek V3 pattern:** KV cache is stored in DRAM with BFP8 format for memory efficiency. The `KVCacheUpdate` MicroOp handles:
- Read existing cache row from DRAM.
- Untilize (BFP8 -> BF16) for splice.
- Write new K/V values at the current position.
- Retilize (BF16 -> BFP8) for memory efficiency.
- Write back to DRAM.

The SDPA decode op reads K/V chunks from DRAM during attention computation, streaming through the sequence dimension.

**Transfer to pi0.5:** The KV cache pattern transfers directly with one simplification: pi0.5's KV cache is **write-once, read-many**. The prefix K/V are written during prefill and then read (not updated) during every denoising step. This means:
- `KVCacheUpdate` runs once (during prefix fill).
- SDPA's DRAM K/V streaming reads run `num_denoising_steps` times.
- No concurrent read/write conflicts.

**Key difference from DeepSeek V3:** DeepSeek V3 updates the KV cache on every token generation step (autoregressive). pi0.5 only writes the KV cache once (during prefix processing) and then reads it repeatedly during denoising. This simplifies cache management but means the cache must hold the entire prefix sequence (image tokens + language tokens, potentially 256 + 200 = 456 tokens per image).

---

### 5. DRAM-Streaming Matmul (Directly Transferable)

**TT-Blaze implementation:** `DRAMStreamingMatmul` MicroOp (`blaze/ops/dram_streaming_matmul/op.py`)

This op streams weight tiles from DRAM during matmul computation, enabling matmuls where the weight matrix exceeds L1 capacity. It supports:
- Expert indexing (for MoE -- not needed for pi0.5).
- Fused activations (SiLU, clamped gate/up -- needs GELU addition for pi0.5).
- Configurable subblock parameters.

**Transfer to pi0.5:** The DRAM-streaming approach is likely needed for the PaliGemma FFN:
- Gate+up weight: `(2048, 16384)` = 32M elements -> ~32MB in BFP8, exceeding L1 capacity.
- Down weight: `(16384, 2048)` = 32M elements -> ~32MB in BFP8.

The `DenseSwiGLU` FusedOp wraps this pattern:
```
Mcast -> DRAM MM (up) + DRAM MM (gate) -> eltwise_mul -> Gather -> Mcast -> DRAM MM (down)
```

For pi0.5, the same pipeline applies with `fused_activation="gelu"` (see ADAPT tier).

The action expert FFN is smaller: gate+up is `(1024, 4096)` = 4M elements -> ~4MB in BFP8. This likely fits in L1, so the L1-resident `DenseMLP` path may be feasible for the action expert.

---

## GLM-5.1 FusedOp Patterns

### 1. GlmFusedProj: Parallel Projection Pattern (Conceptually Transferable)

**TT-Blaze implementation:** `GlmFusedProj` FusedOp (`blaze/ops/glm5_fused_proj/op.py`)

This is a single FusedProgram that runs 4+ parallel matmuls on disjoint core grids after one Mcast:
```
Mcast(hidden) -> [MM_idx_q, MM_q_a, MM_idx_k, MM_idx_w] (parallel) -> [Gather x4]
```

Key techniques:
- **OverlappedViews at byte_offset=0:** All weight tensors packed into one physical DRAM tensor with views on disjoint core ranges.
- **Disjoint core allocation:** `pick_matmul_cores()` distributes cores across projections proportional to output width.
- **Single FusedProgram:** One kernel launch for all projections, amortizing launch overhead.

**Transfer to pi0.5:** The dual-expert Q/K/V projection can use the same pattern:

```
Mcast(hidden_paligemma) -> [Q_pali, K_pali, V_pali] (parallel)
Mcast(hidden_action)    -> [Q_action, K_action, V_action] (parallel)
```

Since the two experts have different hidden widths (2048 vs 1024), they need separate Mcasts. But within each expert, the Q/K/V projections are parallel and benefit from the GlmFusedProj pattern.

For the action expert specifically:
- Q: `(1024, 8*256) = (1024, 2048)` -> 2M elements
- K: `(1024, 1*256) = (1024, 256)` -> 256K elements
- V: `(1024, 1*256) = (1024, 256)` -> 256K elements

The asymmetry (Q is 8x larger than K or V) matches the GLM5 pattern where projections have very different sizes but share the same Mcasted input.

---

### 2. GlmFusedProj with KVBranch Integration (Pattern Transferable)

**From `glm5_fused_proj/op.py`:**
```python
# Optional KVBranch: adds kv_a_proj as 5th parallel matmul
# followed by Gather + RMSNorm + RoPE + KVCacheUpdate
if has_kv_branch:
    KVBranch.emit(f, mcast_cb, kv_a_view,
                  tt_kv_norm_gamma, tt_krope_trans_mat,
                  kv_params, prefix="kv_branch")
```

This demonstrates fusing projection + normalization + RoPE + cache update into a single FusedProgram.

**Transfer to pi0.5:** The same pattern applies to pi0.5's KV processing:
```
Mcast(hidden) -> K_matmul -> RoPE(K) -> KVCacheUpdate
              -> V_matmul -> KVCacheUpdate
```

The GLM5 KVBranch pattern (Gather + RMSNorm + RoPE + CacheUpdate) provides a template for this chain, though pi0.5 skips the RMSNorm step (no latent compression on K/V).

---

### 3. Sender/Receiver Core Architecture (Directly Transferable)

**Pattern:** TT-Blaze uses a sender/receiver architecture where:
- **Sender core:** Holds the activation, runs Mcast to distribute it.
- **Compute cores:** Run matmuls on disjoint weight shards.
- **Gather:** Results from compute cores are collected back to the sender core.

This is configured via `GridConfig` with `sender_core`, `matmul_cores`, and core range allocation.

**Transfer to pi0.5:** The same sender/receiver pattern applies. The sender core holds the 1-token activation (1x2048 for PaliGemma, 1x1024 for action expert), Mcasts it, and collects matmul results.

---

## Key Differences: No MoE

The single most important difference between pi0.5 and DeepSeek V3/GLM-5.1 is the **absence of MoE**. This eliminates a large number of TT-Blaze ops:

| DeepSeek/GLM Op | Purpose | pi0.5 Equivalent |
|-----------------|---------|-------------------|
| `moe` | Full MoE pipeline | Not needed |
| `moe_router` / `glm_moe_router` | Expert routing | Not needed |
| `routed_expert` / `glm_routed_expert` | Per-expert FFN | Not needed |
| `large_moe` / `glm_large_moe` | Large-scale MoE | Not needed |
| `moe_gate` / `glm_moe_gate` | Expert gating | Not needed |
| `distributed_topk` / `dsa_topk` | Top-K expert selection | Not needed |
| `sparse_layer` | Sparse FFN execution | Not needed |
| `reduce_to_one` | MoE output merge | Not needed |
| `all_reduce` / `allreduce_moe` | Cross-device MoE reduction | Not needed |

This removes approximately **15-20 ops** from the pipeline. pi0.5 uses dense FFN throughout, which is simpler but also means every token goes through the full FFN width (no expert specialization).

The dense pipeline also simplifies multi-device deployment. DeepSeek V3's MoE requires sophisticated expert placement and all-to-all communication. pi0.5 only needs standard tensor parallelism (split weight matrices across devices).

---

## Key Differences: Dual-Expert vs. MoE

| Property | DeepSeek V3 MoE | pi0.5 Dual Expert |
|----------|-----------------|---------------------|
| Number of experts | 256 routed + 1 shared | 2 (PaliGemma + action) |
| Expert selection | Top-K routing per token | Fixed (each token goes to its expert) |
| Expert width | Same width, different specialization | Different widths (2048 vs 1024) |
| Shared computation | Shared K/V in MLA | Shared attention (concat Q, K, V across experts) |
| Normalization | Standard RMSNorm everywhere | Standard RMSNorm (expert 0) + adaRMSNorm (expert 1) |
| Activation | SiLU (SwiGLU) | GELU (Gated GELU) |
| Residual | Standard `x + y` | Standard (expert 0), gated `x + y * gate` (expert 1) |

---

## Transferable Patterns Summary

| Pattern | Source | Effort to Transfer | pi0.5 Benefit |
|---------|--------|--------------------|---------------|
| DenseMLP pipeline | DeepSeek V3 | Low (GELU param change) | PaliGemma FFN |
| DRAM-streaming matmul | DeepSeek V3 | Low (GELU activation mode) | Large FFN weights |
| Parallel projection FusedOp | GLM-5.1 | Moderate (adapt core allocation) | Q/K/V projection fusion |
| KVBranch integration | GLM-5.1 | Low (remove MLA compression) | K projection + RoPE + cache update |
| OverlappedView weight packing | GLM-5.1 | Low (same pattern, different shapes) | Multi-weight DRAM management |
| KV cache management | DeepSeek V3 | Low (write-once simplification) | Prefix caching for denoising |
| Sender/receiver core arch | DeepSeek V3 / GLM-5.1 | None (identical pattern) | All matmul pipelines |
| SDPA decode with GQA | DeepSeek V3 | Moderate (add prefix-LM masking) | Shared attention |

## Key Takeaways

1. **pi0.5 is a strict subset of DeepSeek V3's compute patterns.** Every compute primitive in pi0.5 (matmul, RMSNorm, RoPE, SDPA, embedding, residual) exists in DeepSeek V3. The new requirements (adaRMSNorm, GELU, LayerNorm, gated residual) are additions, not replacements.

2. **The MoE absence is the biggest simplification.** Removing 15-20 MoE-related ops eliminates the most complex part of TT-Blaze's pipeline (expert routing, sparse execution, cross-device all-to-all). pi0.5 deployment only needs the dense path.

3. **GLM-5.1's GlmFusedProj is the best template for pi0.5's projection stage.** The pattern of Mcast -> N x parallel matmul -> N x Gather maps directly to pi0.5's Q/K/V projections. The OverlappedView weight packing technique reduces DRAM tensor management overhead.

4. **The KV cache pattern simplifies from read-write to write-once-read-many.** This eliminates cache consistency concerns and enables optimizations like pre-formatting the cache for optimal SDPA read patterns.

5. **Multi-device strategy shifts from MoE-style all-to-all to standard tensor parallelism.** For T3K deployment, pi0.5 can use the simpler `broadcast_rmsnorm` + weight-sharded matmul approach instead of DeepSeek V3's complex MoE routing across devices.

6. **The DenseSwiGLU pipeline is the likely FFN implementation for PaliGemma's large FFN (width 16384), while DenseMLP may work for the action expert's smaller FFN (width 4096).** This means both DRAM-streaming and L1-resident matmul paths are needed, but both already exist.
