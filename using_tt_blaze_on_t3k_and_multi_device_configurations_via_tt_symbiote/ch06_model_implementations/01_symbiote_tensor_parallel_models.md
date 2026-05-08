# Chapter 6 -- Part 1: Symbiote Tensor-Parallel Models (GLM-4.7, Gemma4, Ling, Qwen3.5)

This section traces a single token through the complete forward pass of each Symbiote model, showing where sharding, CCL collectives, and device boundaries are crossed. Symbiote replaces PyTorch `nn.Module` subclasses with TTNN equivalents via `register_module_replacement_dict`, then drives inference through HuggingFace `model.generate()` -- the HF control loop stays on host while replaced layers execute on-device. Each model is assessed for production maturity alongside its architectural details.

**Prerequisites:** Chapter 3 (Symbiote module replacement, `register_module_replacement_dict`), Chapter 5 (CCL primitives, `ttnn.reduce_scatter`, `ttnn.all_gather`).

---

## 6.1 Module Replacement Skeleton (All Models)

Every Symbiote test follows the same six-step skeleton (source: `tests/test_glm_4_7.py`, lines 52-89):

1. Load HF model + tokenizer (`AutoModelForCausalLM.from_pretrained`).
2. Define `nn_to_ttnn` replacement dict(s) mapping HF class to TTNN class.
3. Call `register_module_replacement_dict(model, nn_to_ttnn)` -- recursive DFS replaces matching modules in-place.
4. Call `set_device(model, mesh_device)` to propagate the mesh to every TTNN module.
5. Loop `v.preprocess_weights(); v.move_weights_to_device()` -- weights are tiled, sharded, and sent to DRAM.
6. `model.generate()` runs inference; HF calls each replaced module's `forward()`.

All tests parametrize `mesh_device` via a topology dict keyed by `MESH_DEVICE` env var, enabling the same test to run on N150 through BHGLX without code changes (see Pattern section in Part 3).

> **Warning:** `register_module_replacement_dict` does a single DFS. If called twice with overlapping classes, the second pass silently skips already-replaced subtrees because the class no longer matches the dict key. Gemma4 and Ling work around this with explicit multi-pass ordering (Sections 6.3, 6.4).

### @run_on_devices Gating

Distributed modules use the `@run_on_devices` decorator to gate their forward paths by topology. A single TTNN module class can implement both single-device and multi-device forward methods:

```python
@run_on_devices(DeviceArch.T3K)
def forward(self, hidden_states, ...):
    # Distributed forward with CCL ops (reduce_scatter, all_gather)
    ...

def forward(self, hidden_states, ...):
    # Default single-device forward (no CCL)
    ...
```

The decorator checks `MeshShapeToDeviceArch[mesh_device.shape]` at dispatch time. On T3K `(1,8)`, the decorated method runs; on N150 `(1,1)`, the undecorated fallback runs. This enables the same replacement dict to work across all topologies — the model adapts its data path at runtime based on the device count.

---

## 6.2 GLM-4.7: Simplest Baseline (Proof-of-Concept)

**Source:** `tt_symbiote/tests/test_glm_4_7.py` (90 lines). Dense decoder, standard GQA attention.

**Replacement dict** (lines 52-55):
```python
nn_to_ttnn = {nn.Linear: TTNNLinearLLama, nn.SiLU: TTNNSilu}
```

**Token trace through one decoder layer:**

```
Host:  input_ids  -->  HF embedding (CPU)  -->  hidden_states [1, seq, hidden]
       |
Device (per layer):
  1. hidden_states arrives on device as replicated ttnn.Tensor
  2. TTNNLinearLLama.forward (q/k/v projections):
     - ttnn.linear(input, weight) on each device independently
     - weights are bfloat8_b, deallocated after use
  3. HF attention runs SDPA (CPU -- no TTNN attention replacement)
  4. TTNNLinearLLama.forward (o_proj) --> output
  5. Residual add runs on CPU (aten::add)
  6. TTNNLinearLLama.forward (gate_proj, up_proj, down_proj via SiLU)
  7. Residual add on CPU
       |
Host:  logits --> argmax --> next token
```

**CCL crossings: None.** `TTNNLinearLLama` has no mesh sharding -- every device runs the full computation independently. This leaves multi-device bandwidth unused.

| Aspect | Status |
|--------|--------|
| Module coverage | Minimal (Linear + SiLU only) |
| Attention | CPU fallback (no TTNN replacement) |
| KV cache | HF default (`DynamicCache`) |
| Trace | None |
| Fabric config | None needed |

**Production gap:** Proof-of-concept only. Production use requires: TTNN attention, paged KV cache, decoder-layer replacement for on-device residuals, and `@trace_enabled` wrappers.

---

## 6.3 Gemma4-31B: Heterogeneous Attention with 3-Pass Replacement

**Source:** `tt_symbiote/tests/test_gemma4.py` (199 lines). 60 layers (50 sliding-window + 10 global), GeGLU FFN, logit soft-capping.

**Three replacement passes** (lines 126-144):

| Pass | Dict | Purpose |
|------|------|---------|
| 1 | `{DecoderLayer: TTNNGemma4DecoderLayer, RMSNorm: TTNNDistributedRMSNorm, Embedding: TTNNGemma4ScaledEmbedding}` | Structural layers + norms + embed (excluding vision tower) |
| 2 | `{nn.Linear: TTNNLinearIColShardedWRowSharded}` | All remaining linears with TP sharding |
| 3 | `{Gemma4TextModel: TTNNGemma4TextModel}` | Model wrapper for trace-safe iteration |

**Token trace through one decoder layer** (column-sharded across 8 devices):

```
TTNNGemma4DecoderLayer.forward (gemma4_modules.py, line 142):
  1. residual = hs (stays on device)
  2. TTNNDistributedRMSNorm.forward (normalization.py, line 200):
     a. rms_norm_pre_all_gather(inp) --> per-device partial stats
     b. ttnn.all_gather(stats, topology=Ring) --> <-- CCL CROSSING #1
     c. rms_norm_post_all_gather(inp, gathered_stats, weight)
  3. TTNNGemma4Attention.forward:
     a. TTNNLinearIColShardedWRowSharded (q_proj, k_proj, v_proj):
        - Weight sharded on dim=-2 across 8 devices
        - ttnn.linear(input, shard) --> partial result
        - ttnn.reduce_scatter(dim=3, cluster_axis=1, Ring) --> <-- CCL CROSSING #2
     b. Paged attention (on-device SDPA, dual cache: sliding/global)
     c. TTNNLinearIColShardedWRowSharded (o_proj):
        - ttnn.linear + ttnn.reduce_scatter --> <-- CCL CROSSING #3
  4. post_attention_layernorm --> all_gather --> <-- CCL CROSSING #4
  5. hs = ttnn.add(residual, attn_out) [ON DEVICE -- no host trip]
  6. TTNNGemma4TextMLP (GeGLU):
     a. gate_proj linear + reduce_scatter --> <-- CCL CROSSING #5
     b. up_proj linear + reduce_scatter --> <-- CCL CROSSING #6
     c. ttnn.gelu(gate) * up
     d. down_proj linear + reduce_scatter --> <-- CCL CROSSING #7
  7. pre/post_feedforward_layernorm --> all_gather x2 --> <-- CCL #8, #9
  8. hs = ttnn.add(residual, mlp_out) [ON DEVICE]
```

**Total CCL crossings per layer: ~9** (4 reduce_scatter from linears, 3 all_gather from distributed norms, 2 from feedforward norms).

| Aspect | Status |
|--------|--------|
| Module coverage | Complete (decoder, norms, embed, linear, model wrapper) |
| Attention | Fused QKV (1 matmul + 2 CCL vs 3+4), on-device paged SDPA |
| KV cache | Dual paged (sliding: 16 KV heads, dim=256; global: 4 heads, dim=512) |
| Trace | `@trace_enabled` on decoder layer and attention |
| Shared rotary | Class-level cache prevents 60-layer OOM (lines 293-306) |

> **Warning:** `TTNNGemma4TextModel` allocates a persistent `_decode_cache_position` buffer on the first decode step (`gemma4_text.py`, lines 191-216). Without this, the trace allocator aliases the cache_position address with intermediates, corrupting values for layers 1-59. This was a production bug.

**Production gap:** Multimodal input pipeline is text-only; `_to_replicated()` in attention does a host round-trip for decode tokens (tiny, but does not scale to large batches); `model.device` property hack needs upstream fix.

---

## 6.4 Ling-mini-2.0: MoE + Dense Hybrid with Paged Attention

**Source:** `tt_symbiote/tests/test_ling_mini_2_0.py` (153 lines). BailingMoeV2, first-k-dense layers + MoE layers.

**Three replacement passes** (lines 87-99):

| Pass | Dict | Purpose |
|------|------|---------|
| 1 | `{DecoderLayer: TTNNBailingMoEDecoderLayerPadded, RMSNorm: TTNNDistributedRMSNorm, Embedding: TTNNBailingPaddedEmbedding, RotaryEmb: TTNNBailingRotaryEmbedding}` | Structural + infra |
| 2 | `{nn.Linear: TTNNLinearIColShardedWRowSharded, nn.SiLU: TTNNSilu}` | All remaining linears |
| 3 | `{BailingMoeV2Model: TTNNBailingMoeV2Model}` | Model wrapper (trace-safe iteration) |

**Token trace through a MoE layer** (decoder_layer.py + moe.py):

```
TTNNBailingMoEDecoderLayerPadded.forward:
  Pads seq_len to next power-of-2 (line 213-214) for trace reuse
  -> TTNNBailingMoEDecoderLayer.forward (line 66):
    1. residual = hs
    2. input_layernorm (TTNNDistributedRMSNorm):
       pre_all_gather -> all_gather(Ring) -> post  --> <-- CCL #1
    3. Attention (col-sharded linears):
       q/k/v: linear + reduce_scatter x3         --> <-- CCL #2,3,4
       SDPA with paged KV cache (on device)
       o_proj: linear + reduce_scatter            --> <-- CCL #5
    4. ttnn.add(residual, attn_out) [ON DEVICE]
    5. post_attention_layernorm -> all_gather      --> <-- CCL #6
    6. TTNNBailingMoE (subclass of TTNNMoE):
       a. all_gather(x, dim=-1, Linear)           --> <-- CCL #7 (revert TP)
       b. Gate matmul (replicated weight, float32 accum)
       c. TTNNMoERouterDecode: 3-pass centering topk (sigmoid)
       d. TTNNExperts.forward:
          - all_to_all_dispatch (tokens -> expert devices)  --> <-- CCL #8
          - sparse_matmul(gate_proj), sparse_matmul(up_proj)
          - silu(gate) * up, sparse_matmul(down_proj)
          - all_to_all_combine (expert outputs -> positions) --> <-- CCL #9
       e. reduce_scatter(routed_output, Ring)      --> <-- CCL #10
       f. shared_experts: gate+up+down with reduce_scatter x3 --> <-- CCL #11,12,13
       g. ttnn.add(routed, shared)
    7. ttnn.add(residual, mlp_out) [ON DEVICE]
```

**Total CCL crossings per MoE layer: ~13.** The `all_to_all_dispatch/combine` pair is the most expensive -- it redistributes tokens across all devices based on expert routing.

| Aspect | Status |
|--------|--------|
| Module coverage | Complete (decoder, norms, embed, rotary, linear, model wrapper) |
| Decoder layer | Power-of-2 padding reduces unique trace keys from O(N) to O(log N) |
| MoE | `TTNNBailingMoE` with dense/MoE auto-detection per layer |
| KV cache | Paged attention, all layers |
| On-device residuals | Yes (eliminates 2 host round-trips per layer) |

**Production gap:** MTP layers handled but `roll_tensor` is not implemented (would crash if `num_nextn_predict_layers > 0`); `token_type_ids` silently deleted rather than handled.

---

## 6.5 Qwen3.5-35B-A3B: Hybrid Attention + 256-Expert MoE

**Source:** `tt_symbiote/tests/test_qwen3_5_35b_a3b.py`. 40 layers (30 DeltaNet linear-attention + 10 full GQA), 256 experts top-8 per MoE layer.

**Dynamic class discovery** (lines 137-183) -- unlike the other models, Qwen3.5 discovers HF classes at runtime:
```python
for layer in model.model.layers:
    if layer_type == "linear_attention": linear_attn_class = layer.linear_attn.__class__
    elif layer_type == "full_attention": full_attn_class = layer.self_attn.__class__
    if hasattr(layer, "mlp"): moe_class = layer.mlp.__class__
```

**Replacement dict** (lines 189-200):
```python
nn_to_ttnn = {
    linear_attn_class: TTNNQwen3LinearAttention,
    full_attn_class:   TTNNQwen3FullAttention,
    moe_class:         TTNNQwen3MoE,
}
```

**Token trace through a linear-attention + MoE layer:**

```
1. TTNNQwen3LinearAttention (DeltaNet-style, no KV cache):
   - Projection linears (q, k, v, o) with col-sharded TP
   - DeltaNet recurrent state update (no CCL -- state is local)
   --> CCL: reduce_scatter from each projection linear

2. TTNNQwen3MoE.forward (qwen_moe.py, line 789):
   a. all_gather(x, dim=-1, Linear)              --> <-- CCL (revert TP)
   b. Gate matmul (replicated, float32 for precision)
   c. TTNNQwenMoERouterDecode: softmax -> 3-pass centering topk
      (KEY DIFF: softmax, not sigmoid like Ling)
   d. TTNNQwenExperts.forward (fused w1/w3):
      - all_to_all_dispatch                       --> <-- CCL
      - SINGLE sparse_matmul for fused gate+up [H, 2*I]
      - ttnn.slice to split gate/up halves
      - silu(gate) * up, then sparse_matmul(down)
      - all_to_all_combine                        --> <-- CCL
   e. reduce_scatter(routed, Ring)                --> <-- CCL
   f. shared_expert with sigmoid gate:
      gate_values = sigmoid(linear(x, gate_weight))
      shared_out = gate_values * shared_expert(x) --> <-- CCL (shared expert linears)
```

**Paged KV cache** is created only for the 10 full-attention layers. The function inspects `config.layer_types` to identify `"full_attention"` indices, then creates `TTNNQwenPagedAttentionKVCache` with `num_layers=10`.

| Aspect | Status |
|--------|--------|
| Module coverage | High (both attention types + MoE replaced) |
| MoE routing | 3-pass centering topk for BF16 precision |
| Expert compute | Fused `sparse_matmul` (W1/W3 in single read) |
| nn.Linear replacement | Commented out -- small gating layers (output_size=1) break TP |
| CPU fallbacks | 3 env vars: `TT_QWEN_CPU_LINEAR_ATTN`, `TT_QWEN_CPU_EXPERTS`, `TT_SYMBIOTE_RUN_MODE` |

> **Warning:** The Qwen3 MoE router uses softmax (not sigmoid). Using the wrong activation silently produces valid-looking but incorrect routing weights. `TTNNQwenMoERouterDecode` overrides `forward()` to call `ttnn.softmax` instead of `ttnn.sigmoid`.

> **Warning:** When `TT_QWEN_CPU_LINEAR_ATTN=1` forces DeltaNet layers to CPU, paged KV cache must also be disabled because HF's native DeltaNet uses `conv_states`/`recurrent_states` that the TTNN cache does not provide.

**Production gap:** Remove CPU fallback paths, stabilize DeltaNet on TTNN, replace remaining `nn.Linear` with size-aware TP, add model wrapper (Pass 3) for on-device input conversion.

---

## 6.6 Comparative Maturity Summary

| Feature | GLM-4.7 | Gemma4 | Ling | Qwen3.5 |
|---------|---------|--------|------|---------|
| TTNN attention | No | Yes (fused QKV) | Yes | Yes (hybrid) |
| MoE on device | N/A | N/A (dense) | Yes | Yes (sparse_matmul) |
| Paged KV cache | No | Dual (sliding+global) | Yes (all layers) | Partial (full-attn only) |
| Trace support | No | Yes | Yes | Partial |
| Model wrapper | No | Yes | Yes | No |
| On-device residuals | No | Yes | Yes | No |
| CCL crossings/layer | 0 | ~9 | ~13 (MoE) | varies |

---

| Previous | Up | Next |
|----------|-----|------|
| [Ch5: Pipeline System](../ch5_final/) | [Table of Contents](../README.md) | [Part 2: Blaze DeepSeek V3 Pipeline](02_blaze_deepseek_v3_pipeline.md) |
