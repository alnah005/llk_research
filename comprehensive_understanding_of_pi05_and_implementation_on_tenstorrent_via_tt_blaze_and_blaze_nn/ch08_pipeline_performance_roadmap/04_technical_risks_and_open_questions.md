# 04 -- Technical Risks and Open Questions

## Context

This section catalogs eight technical risks identified during the architecture analysis. Each risk is assessed for likelihood (how likely it is to cause delay) and impact (how much delay it would cause), and paired with a mitigation strategy. The risks are ordered by combined severity.

## Risk Summary

```
+-----+----------------------------------------+------------+--------+
| #   | Risk                                   | Likelihood | Impact |
+-----+----------------------------------------+------------+--------+
|  1  | SDPA with prefix-LM masking            |    HIGH    |  HIGH  |
|  2  | Dual-expert width mismatch in SDPA     |    HIGH    |  HIGH  |
|  3  | Conv2d for SigLIP patch extraction      |   MEDIUM   | MEDIUM |
|  4  | SigLIP bidirectional attention at scale |   MEDIUM   | MEDIUM |
|  5  | Variable-length prefix sequences        |   MEDIUM   |  LOW   |
|  6  | Multi-branch tracing in blaze-nn        |    HIGH    | MEDIUM |
|  7  | Numerical precision across denoising    |   MEDIUM   | MEDIUM |
|  8  | Pipeline looping on device              |    LOW     |  HIGH  |
+-----+----------------------------------------+------------+--------+
```

## Risk 1: SDPA with Prefix-LM Masking

**What:** Pi0.5 uses a prefix-LM attention mask where prefix tokens attend bidirectionally to each other, while suffix tokens attend causally to each other and fully to all prefix tokens. This is implemented by the `make_attn_mask` function:

```python
# From pi0.py lines 19-44
cumsum = jnp.cumsum(mask_ar, axis=1)
attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
return jnp.logical_and(attn_mask, valid_mask)
```

The resulting mask looks like:

```
              prefix tokens    suffix tokens
            p0  p1  p2  p3   s0  s1  s2
prefix p0 [  1   1   1   1    0   0   0 ]
prefix p1 [  1   1   1   1    0   0   0 ]
prefix p2 [  1   1   1   1    0   0   0 ]
prefix p3 [  1   1   1   1    0   0   0 ]
suffix s0 [  1   1   1   1    1   0   0 ]
suffix s1 [  1   1   1   1    1   1   0 ]
suffix s2 [  1   1   1   1    1   1   1 ]
```

This is neither a pure causal mask nor a pure bidirectional mask.

**Why it is risky:** TT hardware SDPA kernels are optimized for two common patterns:
1. **Causal (lower-triangular):** Used in autoregressive LLM decoding.
2. **No mask (bidirectional):** Used in BERT-style models and vision transformers.

The prefix-LM pattern is a block-structured mask with a dense upper-left quadrant and a triangular lower-right quadrant. The kernel must either:
- Accept an explicit mask tensor and apply it during softmax computation, or
- Support a "causal with prefix" mode that takes a prefix length parameter.

If the existing SDPA kernel only supports causal or no-mask modes, implementing prefix-LM masking requires kernel modifications.

**Likelihood:** HIGH. The DeepSeek pipeline uses only causal masking.

**Impact:** HIGH. SDPA is on the critical path of every transformer layer.

**Mitigation:**
1. Test existing TTNN/Blaze SDPA with an explicit boolean mask tensor. Some implementations accept `attn_mask` as an additive bias (0 or -inf).
2. If the kernel does not accept arbitrary masks, implement a "causal_from_index" mode: treat positions 0..prefix_len as bidirectional, positions prefix_len..end as causal. This is a simpler kernel change than full arbitrary masking.
3. Worst case: compute attention in two passes -- (a) prefix self-attention (bidirectional), (b) suffix-to-all attention (standard causal from the suffix start position). This doubles the attention compute but avoids kernel changes.

**During denoising (Stage 3):** The mask is simpler because the prefix is in the KV cache. The suffix queries attend to all cached prefix KV (full attention) plus each other (causal). This is equivalent to a standard causal attention where the "prompt" is in the KV cache -- a pattern that autoregressive decoding already handles. **Denoising-stage masking may work out of the box** if the SDPA kernel supports KV-cache cross-attention with causal masking on the query portion.

## Risk 2: Dual-Expert Width Mismatch in SDPA

**What:** The Gemma `Attention` module in `gemma.py` (lines 158-249) concatenates Q/K/V from two experts with different input widths before running joint attention:

```python
# PaliGemma expert (width=2048):
q_pg = q_einsum("BTD,NDH->BTNH", x_paligemma)   # [B, T_p, 8, 256]  (D=2048)
k_pg, v_pg = kv_einsum("BSD,2KDH->2BSKH", x_pg) # [B, T_p, 1, 256]  (D=2048)

# Action expert (width=1024):
q_act = q_einsum("BTD,NDH->BTNH", x_action)      # [B, T_s, 8, 256]  (D=1024)
k_act, v_act = kv_einsum("BSD,2KDH->2BSKH", x_a) # [B, T_s, 1, 256]  (D=1024)

# Concatenate along sequence dimension:
q = concat([q_pg, q_act], axis=1)   # [B, T_p+T_s, 8, 256]
k = concat([k_pg, k_act], axis=1)   # [B, T_p+T_s, 1, 256]
v = concat([v_pg, v_act], axis=1)   # [B, T_p+T_s, 1, 256]
```

After attention, the output is split back and projected through separate output matrices of different sizes:

```python
# Split output at T_p boundary:
encoded_pg = encoded[:, :T_p]   # [B, T_p, 8, 256]
encoded_act = encoded[:, T_p:]  # [B, T_s, 8, 256]

# Project with different output widths:
out_pg = out_einsum_pg("BTNH,NHD->BTD", encoded_pg)   # D=2048
out_act = out_einsum_act("BTNH,NHD->BTD", encoded_act) # D=1024
```

**Why it is risky:** While the Q/K/V tensors have the same head dimension (256) and head count (8), the *input projections* have different weight shapes. The TT implementation must:

1. Project Q/K/V separately for each expert (two matmuls with different shapes per Q, K, V).
2. Concatenate the projected Q/K/V tensors (simple concat).
3. Run SDPA on the concatenated tensors (standard).
4. Split the output back at the correct boundary.
5. Apply different output projections per expert (two matmuls with different shapes).

This is not a standard transformer attention pattern. A naive implementation would require 6 separate matmul dispatches (3 projections x 2 experts) plus 2 output projections -- 8 matmuls per attention layer. A fused implementation would need to handle the width mismatch internally.

**Likelihood:** HIGH. No existing TT model has this pattern.

**Impact:** HIGH. An unoptimized implementation (8 matmuls) would significantly increase per-layer latency. However, *correctness* is straightforward -- it is just multiple matmuls and a concat.

**Mitigation:**
1. **Start unfused:** Implement as separate matmul calls per expert. This is correct but slow. Use for Phase 2 validation.
2. **Fuse Q/K/V per expert:** Create a single Blaze FusedOp per expert that computes all Q/K/V projections in one kernel launch. This is 2 launches instead of 6.
3. **During denoising (Stage 3):** The PaliGemma expert is `None`, so only the action expert's Q/K/V projections run. The width mismatch disappears entirely. The concat is just: `q = q_act`, `k = concat([kv_cache_k, k_act])`, `v = concat([kv_cache_v, v_act])`. This is a standard KV-cache attention pattern.
4. **Conclusion:** The width mismatch is a problem for Stage 2 (prefix fill) but not for Stage 3 (denoising). Since Stage 2 runs once and Stage 3 is the bottleneck, the unfused approach for Stage 2 is acceptable initially.

## Risk 3: Conv2d for SigLIP Patch Extraction

**What:** SigLIP's first operation is a 2D convolution:

```python
nn.Conv(1152, (14, 14), strides=(14, 14), padding="VALID")
# Input: [B, 224, 224, 3]
# Output: [B, 16, 16, 1152]
```

This is not a typical deep-learning convolution (small kernel, stride 1). It is a single-layer "patch extraction" with a large kernel that equals the stride, meaning there is no overlap between patches.

**Why it is risky:** Tenstorrent's conv2d implementation may not be optimized for this unusual configuration. The kernel is large (14x14 = 196 spatial elements x 3 channels = 588 input values per patch), the stride eliminates overlap, and the output channel count is high (1152). Performance may be poor or the op may not be supported in all data formats.

**Likelihood:** MEDIUM. TTNN conv2d exists but may have edge cases with large kernels.

**Impact:** MEDIUM. SigLIP runs once per inference (not on the denoising critical path), so even a slow implementation does not destroy throughput. But a broken implementation blocks Phase 3.

**Mitigation:**
1. **Im2col + matmul:** Reshape the 224x224x3 image into [256, 588] patches (each patch is a flattened 14x14x3 vector) and multiply by the [588, 1152] weight matrix. This is a guaranteed-correct implementation using only matmul.
2. **TTNN conv2d:** Test with the exact configuration. If it works and is fast enough, use it.
3. **Host-side extraction:** As a last resort, do the patchification on the host CPU and send the [256, 588] matrix to the device for the matmul. The data transfer is small (256 x 588 x 4 bytes = ~0.6 MB).

## Risk 4: SigLIP Bidirectional Attention at Scale

**What:** SigLIP uses bidirectional (no-mask) multi-head self-attention across 256 tokens (per image) with 16 heads, width 1152. This runs for 27 layers per image, 3 images total.

**Why it is risky:** This is *not* a causal attention. The attention kernel must support the "no mask" (all-attend-to-all) mode. Additionally:
- The width (1152) is not a power of 2 and not a multiple of 128 (it is 9 x 128). This may cause alignment issues or suboptimal tile utilization on Tensix cores.
- The MLP hidden dim (4304) is also not a power of 2 (4304 = 16 x 269). This is unusual.

**Likelihood:** MEDIUM. Bidirectional SDPA is a standard pattern (BERT uses it), but the non-power-of-2 widths are unusual for TT hardware.

**Impact:** MEDIUM. If the width causes issues, padding is the standard fix (pad 1152 to 1280 or 1536), but this wastes compute and memory.

**Mitigation:**
1. Test SDPA with width=1152 in bidirectional mode. Verify tile utilization.
2. If tile utilization is poor, pad to the nearest efficient width and adjust the projection head to compensate.
3. For the MLP dim 4304, test with the exact value. If alignment is required, pad to 4352 (= 17 x 256) or 4096 + 256.

## Risk 5: Variable-Length Prefix Sequences

**What:** The prefix length varies per inference call depending on:
- Which images are present (controlled by `image_masks`)
- The tokenized prompt length (up to `max_token_len=200`)

The `input_mask` tensor marks valid vs. padding positions. The KV cache stores up to `prefix_len` entries, but the actual number of valid entries varies.

**Why it is risky:** TT-Blaze programs are typically compiled with fixed tensor shapes. Variable-length sequences require either:
- Padding to max length and masking (wastes compute but is simple), or
- Dynamic shapes (more complex, may require recompilation).

**Likelihood:** MEDIUM. Variable lengths are common in LLM inference and there are established patterns (pad + mask) for handling them.

**Impact:** LOW. Padding wastes some compute but the prefix is short (max ~968 tokens) and runs once. The denoising iterations have a fixed suffix length (action_horizon=50), so they are not affected.

**Mitigation:**
1. Pad all prefix sequences to the maximum length (968 tokens) and use the `input_mask` to zero out padding positions in the attention mask.
2. The KV cache can be allocated at max size; invalid entries are never attended to because the mask excludes them.
3. This is the approach the reference implementation already uses (the `input_mask` in `make_attn_mask` handles it).

## Risk 6: Multi-Branch Tracing in blaze-nn

**What:** Pi0.5's forward pass has a branching structure:
- `embed_prefix` produces different token types (image vs. text) via separate code paths.
- `embed_suffix` branches on `self.pi05` to choose between adaRMSNorm (pi0.5) and direct timestep concatenation (pi0).
- The Gemma `Block` iterates over a *list* of experts, skipping `None` entries.
- The full `sample_actions` uses `jax.lax.while_loop` for the denoising loop.

The blaze-nn tracing system (defined in `_tracing.py`) operates by recording ops dispatched through a `TracingContext`. It is unclear how well it handles:
- Conditional execution (`if x is not None: skip`)
- Data-dependent branching (`self.pi05` flag)
- Loop constructs (`while_loop`)
- Operations on lists of varying-length inputs

**Why it is risky:** If blaze-nn tracing cannot represent the pi0.5 control flow, the model cannot be traced as a single graph. It would need to be decomposed into multiple traced subgraphs connected by host logic.

**Likelihood:** HIGH. The tracing system in `_tracing.py` appears to be designed for straight-line computation graphs (single input -> sequence of ops -> single output). Conditionals and loops are likely not supported.

**Impact:** MEDIUM. The host-orchestrated approach (Strategy A) already decomposes the model into separate stages. Each stage can be traced independently as a straight-line graph. The control flow (looping, branching) is handled by the host Python code.

**Mitigation:**
1. **Do not try to trace the full `sample_actions`.** Instead, trace three separate subgraphs:
   - `trace_siglip(images)` -> image tokens
   - `trace_prefix_fill(prefix_tokens, mask)` -> KV cache
   - `trace_denoise_step(x_t, t, kv_cache)` -> v_t
2. Host Python handles the loop and concatenation logic.
3. This aligns perfectly with Strategy A (host-orchestrated) from section 01.
4. Conditional branches within each subgraph (e.g., `if x is not None`) can be resolved at trace time by fixing the configuration (e.g., always pi0.5 mode, always 3 images).

## Risk 7: Numerical Precision Across Denoising Steps

**What:** Flow matching inference is an ODE solve using Euler integration:

```
x_{t-dt} = x_t + dt * v_t(x_t)
```

Over 10 steps, numerical errors in `v_t` compound. The bf16 format has limited mantissa precision (7 bits). If the velocity field `v_t` has small magnitudes relative to `x_t`, the Euler step may lose significant precision.

**Why it is risky:** The flow matching paper uses float32 for inference. The reference openpi code uses bf16 throughout (set via `config.dtype`). If TT hardware introduces additional precision loss (e.g., through BFP8 weights, or through reduced-precision intermediate accumulations), the compounding error across 10 steps could degrade action quality.

**Likelihood:** MEDIUM. The reference already runs at bf16 successfully, so bf16 on TT should be fine. The risk increases with BFP8 quantization.

**Impact:** MEDIUM. Degraded action quality may not be detectable through simple PCC comparison but could cause policy failures in the robot control loop.

**Mitigation:**
1. Validate bf16 inference on TT against bf16 inference on GPU. PCC should be very high (>0.999 per step, >0.99 after 10 steps).
2. For BFP8 quantization, validate action quality on a robotics benchmark (not just PCC). Use the application-level metric (task success rate) as the acceptance criterion.
3. Consider keeping the Euler step in bf16 even if weights are BFP8: `x_t += dt * v_t` should be computed at bf16 precision.
4. If precision is insufficient, the adaRMSNorm conditioning path (which modulates with timestep-derived scale/shift/gate) is the most likely source of precision issues. Test this path first.

## Risk 8: Pipeline Looping on Device

**What:** Strategy B (device-side loop) and Strategy C (unrolled pipeline) both require executing the denoising loop without host intervention.

For Strategy B: The device must maintain a step counter, compare against `num_steps`, and conditionally re-dispatch the denoising body. TT-Blaze's `PipelineGraph` supports loopback edges (see `pipeline_builder/graph.py`), but the control flow (conditional termination) is not demonstrated in the existing codebase.

For Strategy C: The loop must be unrolled at compile time into 10 copies of the denoising body. This requires either manual unrolling or compiler support for `jax.lax.while_loop` unrolling.

**Why it is risky:** Device-side conditional branching on Tenstorrent hardware requires firmware-level support. The `PipelineGraph` loopback edge creates the physical routing, but the actual iteration control (where does the step counter live? how is termination signaled?) is unclear.

**Likelihood:** LOW. This is a Phase 5 optimization. Host-orchestrated looping (Strategy A) works without any of this.

**Impact:** HIGH if it cannot be done. The difference between host-orchestrated and device-side looping is ~20 us per iteration of overhead -- negligible relative to ~3-15 ms compute per iteration. But for future multi-device or production deployment, device-side looping may become necessary.

**Mitigation:**
1. Start with Strategy A (host-orchestrated). This is the recommendation from section 01.
2. If device-side looping is needed later, investigate TT-Metal's `jax.lax.while_loop` equivalent or add a simple step-counter-and-branch primitive to the dispatch path.
3. The `PipelineGraph` loopback infrastructure is already in place for the routing. The missing piece is the control flow, not the data flow.

## Open Questions

Beyond the eight risks above, the following questions remain open and should be resolved during Phase 1:

1. **Does TTNN natively support non-power-of-2 matmul widths?** SigLIP uses 1152 and 4304. If padding is required, what is the overhead?

2. **What is the actual PCIe H2D/D2H latency for small tensors?** The analysis assumes <1 us for a [1, 50, 32] bf16 tensor (6.4 KB). This needs measurement.

3. **Can the KV cache be kept in DRAM between separate program dispatches?** Strategy A launches Stage 3 as a separate program 10 times. Each launch must find the KV cache where Stage 2 left it. This requires persistent DRAM allocation across dispatches.

4. **What is the actual DRAM bandwidth utilization during decode-like workloads?** The analysis uses the theoretical peak (256 GB/s). Measured utilization on DeepSeek decode should provide a realistic number.

5. **Does the embedder table (257K x 2048 = ~1 GB bf16) fit efficiently in DRAM?** This is a large lookup table with sparse access patterns (only a few hundred tokens are looked up per inference). The access pattern is random, not streaming.

## Key Takeaways

- The top three risks (SDPA masking, width mismatch, Conv2d) should all be confronted in Phase 1 of the implementation roadmap. They are the "known unknowns" that could derail the project if discovered late.
- Risks 1 and 2 become significantly simpler during the denoising loop (Stage 3) compared to the prefix fill (Stage 2). The denoising stage uses standard KV-cache attention with a fixed suffix length and a single expert, avoiding both the prefix-LM mask complexity and the width mismatch.
- Multi-branch tracing (Risk 6) is mitigated by the host-orchestrated approach: trace each stage as a separate straight-line graph, handle control flow on the host.
- Numerical precision (Risk 7) is a concern primarily for BFP8 quantization, which is a Phase 5 optimization. The bf16 baseline should match the reference.
- Device-side looping (Risk 8) is the lowest-priority risk because host-orchestrated looping has negligible overhead.

---

**Back to:** [`index.md`](./index.md)
