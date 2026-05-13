# 01: Reusable Ops Inventory (REUSE Tier)

## Context

These TT-Blaze ops can be used directly for pi0.5 without kernel modifications. The pi0.5 architecture is a dual-expert Gemma transformer with GQA (8 Q heads, 1 KV head), which maps cleanly onto TT-Blaze's existing infrastructure built for DeepSeek V3 and GLM-5.1. This file covers each REUSE-tier op, its TT-Blaze implementation, and exactly where it appears in pi0.5.

---

## 1. matmul (Linear Projections)

**TT-Blaze location:** `blaze/ops/matmul/op.py`
**blaze-nn API:** `blaze_nn.F.linear(input, weight)`

The `Matmul` MicroOp is the workhorse of TT-Blaze. It performs L1-sharded matrix multiplication: `output[1, out_w] = act @ weight` on the compute grid, with automatic tile layout and CB management.

**pi0.5 usage -- PaliGemma expert (width=2048):**
- `qkv_einsum`: fused QKV projection, shape `(3, 8, 2048, 256)` or split Q `(8, 2048, 256)` + KV `(2, 1, 2048, 256)`. Since `num_kv_heads=1 != num_heads=8`, pi0.5 uses the split path with separate `q_einsum` and `kv_einsum`.
- `attn_vec_einsum`: attention output projection, `(8, 256, 2048)`.
- `gating_einsum`: FFN gate+up projection, `(2, 2048, 16384)`. Both branches computed via two matmuls from the same input.
- `linear`: FFN down projection, `(16384, 2048)`.

**pi0.5 usage -- action expert (width=1024):**
- Same structure with `(8, 1024, 256)` Q, `(2, 1, 1024, 256)` KV, `(8, 256, 1024)` output, `(2, 1024, 4096)` gate+up, `(4096, 1024)` down.

**pi0.5 usage -- head projections:**
- `action_in_proj`: Linear `(action_dim, 1024)` -- projects noisy actions into action expert width.
- `action_out_proj`: Linear `(1024, action_dim)` -- projects decoder output back to action space.
- `time_mlp_in`: Linear `(1024, 1024)` -- first layer of timestep MLP.
- `time_mlp_out`: Linear `(1024, 1024)` -- second layer of timestep MLP.

**Why REUSE:** The matmul kernel is shape-agnostic (parameterized by tile counts). All pi0.5 linear layers have standard `(M, K) x (K, N)` shapes that fit the existing CB allocation and tiling logic. The only consideration is that pi0.5's expert widths (2048 and 1024) are both powers of 2 and tile-aligned.

---

## 2. rmsnorm

**TT-Blaze location:** `blaze/ops/rmsnorm/op.py`
**blaze-nn API:** `blaze_nn.F.rmsnorm(input, gamma, epsilon=1e-6)`

The `RMSNorm` MicroOp computes `x * rsqrt(mean(x^2) + eps) * (1 + gamma)` on TRISC with configurable epsilon, scalar, and fp32 accumulation.

**pi0.5 usage:**
- `pre_attention_norm` (layer norm before attention): Each Gemma Block applies `RMSNorm(name="pre_attention_norm")` and `RMSNorm(name="pre_attention_norm_1")` for the two experts respectively.
- `pre_ffw_norm` (layer norm before FFN): Same pattern, `RMSNorm(name="pre_ffw_norm")` and `RMSNorm(name="pre_ffw_norm_1")`.
- `final_norm`: Applied after the last transformer block, one per expert: `RMSNorm(name="final_norm")` and `RMSNorm(name="final_norm_1")`.

**Important:** Only the PaliGemma expert uses standard RMSNorm. The action expert uses adaRMSNorm (see 03_new_ops_required.md). However, when `adarms_cond[i]` is None (PaliGemma side), the code falls back to regular RMSNorm (`self.param("scale", ...)` path in `gemma.py:RMSNorm`). So the standard `rmsnorm` op covers all PaliGemma norms (36 instances across 18 layers + 1 final norm).

---

## 3. rope (Rotary Position Embedding)

**TT-Blaze location:** `blaze/ops/rope/op.py`
**blaze-nn API:** `blaze_nn.F.rope(input, trans_mat=None)`

The `Rope` MicroOp applies RoPE to Q and K tensors. pi0.5's `_apply_rope` function in `gemma.py` implements standard RoPE with `max_wavelength=10000`:

```python
freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2)
timescale = max_wavelength ** freq_exponents
radians = positions[..., None] / timescale
sin, cos = jnp.sin(radians), jnp.cos(radians)
x1, x2 = jnp.split(x, 2, axis=-1)
result = concat([x1*cos - x2*sin, x2*cos + x1*sin])
```

**pi0.5 usage:**
- Applied to Q and K after projection, before attention, in every layer.
- Positions are computed from cumulative input masks, supporting variable-length prefixes.
- Head dimension is 256, split into two 128-dim halves for sin/cos rotation.

**Why REUSE:** TT-Blaze's RoPE implementation handles variable positions (passed as a runtime argument), configurable head dimensions, and the standard interleaved sin/cos pattern. The `head_dim=256` configuration is the same as DeepSeek V3's Q-rope dimension.

---

## 4. embedding

**TT-Blaze location:** `blaze/ops/embedding/op.py`

The `Embedding` MicroOp performs token ID to DRAM weight row lookup. NCRISC reads a token ID, uses it as a page index to fetch the corresponding row from a DRAM-interleaved embedding table.

**pi0.5 usage:**
- `Embedder.encode()`: Maps tokenized language prompts to embedding vectors, shape `(vocab_size=257152, embed_dim=2048)`, with `sqrt(embed_dim)` scaling.
- Called once during prefix embedding: `self.PaliGemma.llm(obs.tokenized_prompt, method="embed")`.

**Why REUSE:** The embedding op is a simple DRAM page lookup. pi0.5's vocab size (257,152) and embedding dimension (2048) are standard. The `sqrt(embed_dim)` scaling can be absorbed into the embedding weights or applied as a post-embedding multiply.

---

## 5. residual_add

**TT-Blaze location:** `blaze/ops/residual_add/op.py`
**blaze-nn API:** `blaze_nn.F.residual_add(input, bias)`

Element-wise addition for residual connections.

**pi0.5 usage:**
- Standard transformer residual: `x = x + attn_out` and `x = x + ffn_out` after every attention and FFN block in the PaliGemma expert.
- The action expert uses a gated variant (`x + y * gate`) when adaRMSNorm is active (see NEW tier), but when gate is None, it falls back to plain `x + y`.

From `gemma.py`:
```python
def _gated_residual(x, y, gate):
    if gate is None:
        return x + y       # <-- standard residual_add
    return x + y * gate    # <-- gated_residual (NEW)
```

**Why REUSE:** For the PaliGemma expert, all residual connections use the plain `x + y` path. This is a direct match for TT-Blaze's `residual_add`.

---

## 6. kv_cache_update

**TT-Blaze location:** `blaze/ops/kv_cache_update/op.py`

The `KVCacheUpdate` MicroOp splices new KV values into a DRAM KV cache at the current position. NCRISC reads from DRAM, TRISC does untilize/tilize format conversion (BFP8 <-> BF16), and BRISC writes back.

**pi0.5 usage:**
- During `sample_actions`, the prefix KV cache is computed once and stored:
  ```python
  _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=..., positions=...)
  ```
- The KV cache is then read (not updated) during each denoising step, since the suffix tokens don't modify it -- they only attend to it.
- The cache structure is `(L, B, T, K, H)` where L=depth, T=max_seq_len, K=num_kv_heads, H=head_dim.

**Why REUSE:** The KV cache write path (prefix fill) matches the standard kv_cache_update pattern. During denoising, only KV cache reads are needed, which is handled by the SDPA op's DRAM streaming reads. The BFP8 format conversion for memory efficiency is already built into the op.

**Note:** pi0.5 concatenates cached K/V with new K/V: `k = concat([cache_k, k], axis=1)`. In TT-Blaze, this is handled by SDPA reading the full cache from DRAM, so no explicit concat is needed.

---

## 7. q_branch / q_heads

**TT-Blaze location:** `blaze/ops/q_branch/op.py`, `blaze/ops/q_heads/`

The `QBranch` FusedOp chains QAProjection (low-rank Q projection + norm + mcast) and QHeads (up-projection + q_nope + q_rope + CreateQHeads). This is designed for DeepSeek V3's multi-latent attention (MLA).

**pi0.5 usage:**
- pi0.5 uses standard GQA, not MLA. The Q projection is a simple `q_einsum("BTD,NDH->BTNH", x)` producing 8 heads of dimension 256.
- The `q_branch` op's matmul + RoPE chain is reusable for this, though the MLA-specific compression/decompression steps (q_a, q_b) are unnecessary.

**Why REUSE (with simplification):** The core pipeline (matmul -> reshape -> RoPE) within q_branch maps directly to pi0.5's Q path. The MLA decomposition layers can be skipped by configuring the op to use a single full-rank matmul.

---

## 8. mcast / gather

**TT-Blaze location:** `blaze/ops/mcast/op.py`, `blaze/ops/gather/op.py`
**blaze-nn API:** `blaze_nn.F.mcast(input)`, `blaze_nn.F.gather(input)`

`Mcast` broadcasts activation tiles from a sender core to all compute cores. `Gather` collects partial results from compute cores back to the sender core (or directly to an output tensor).

**pi0.5 usage:**
- Every matmul pipeline begins with mcast (distribute activation) and ends with gather (collect results).
- In the DenseMLP pipeline: `RMSNorm -> Mcast -> Matmul -> GatedReduce -> Mcast -> DownProj`.
- The SigLIP encoder would also use mcast/gather for its dense layers.

**Why REUSE:** These are infrastructure ops that are model-agnostic. The activation widths (2048, 1024) and tile formats are standard.

---

## 9. dram_streaming_matmul

**TT-Blaze location:** `blaze/ops/dram_streaming_matmul/op.py`

The `DRAMStreamingMatmul` MicroOp streams weight tiles from DRAM during the matmul computation, avoiding the need to fit all weights in L1 simultaneously. Supports fused activations (SiLU, clamped gate/up).

**pi0.5 usage:**
- Potentially useful for the large PaliGemma FFN: gate+up weight is `(2, 2048, 16384)` = 64M parameters per layer. With BFP8, each weight shard may exceed L1 capacity for single-core execution.
- The DRAM streaming approach from DeepSeek V3's `DenseSwiGLU` can be adapted: stream weight columns while keeping the 1-token activation in L1.

**Why REUSE:** The streaming mechanism is shape-agnostic. The activation mode parameter needs to switch from SiLU to GELU (see ADAPT tier for the activation change), but the DRAM streaming logic itself is reusable.

---

## Key Takeaways

1. **The core transformer building blocks (matmul, rmsnorm, rope, embedding, residual_add, mcast, gather) cover approximately 60% of pi0.5's compute graph** and can be used without modification.

2. **KV cache management is simpler in pi0.5 than in DeepSeek V3** because there is no MoE, no latent compression, and the cache is written once (prefix fill) then read-only during denoising.

3. **The GQA attention configuration (8 Q heads, 1 KV head, head_dim=256) is identical to DeepSeek V3's per-head structure**, making the existing SDPA op's GQA support directly applicable (with masking changes -- see ADAPT tier).

4. **The q_branch pipeline is over-engineered for pi0.5's needs.** pi0.5 uses full-rank Q projection, not MLA's two-stage low-rank decomposition. A simplified Q path (single matmul + reshape + RoPE) would be more efficient than the full QBranch chain.

5. **Tile alignment is natural.** All pi0.5 dimensions (2048, 1024, 256, 16384, 4096) are multiples of 32, so no padding is needed for tile layout.
