# 02: Ops Requiring Adaptation (ADAPT Tier)

## Context

These TT-Blaze ops exist and cover the right computational pattern, but need parameter changes, new activation modes, or extended interfaces to support pi0.5. The adaptations range from trivial (passing `activation="gelu"` to an existing flag) to moderate (adding prefix-LM masking logic to SDPA). None require new kernels from scratch -- the hardware primitives and CB management patterns already exist.

---

## 1. gated_reduce: GELU Instead of SiLU

**TT-Blaze location:** `blaze/ops/gated_reduce/op.py`

**Current behavior:** The `GatedReduce` MicroOp computes `activation(sum(group1)) * sum(group2)` where activation defaults to SiLU. The kernel already has a `use_gelu` compile-time flag:

```python
# From gated_reduce/op.py, line 236:
f.trisc_ct_args([
    ...
    (f"{gr}.use_gelu", 1 if activation == "gelu" else 0),
])
```

And the golden function already supports both:
```python
def golden(group1_inputs, group2_inputs, activation="silu"):
    act_fn = F.gelu if activation == "gelu" else F.silu
    group1_sum = sum(group1_inputs)
    return act_fn(group1_sum) * sum(group2_inputs)
```

**What pi0.5 needs:** The Gemma FeedForward module uses gated GELU:
```python
# From gemma.py, FeedForward.__call__:
ff_gate = jnp.dot(x, w_gating[0])
gate_value = nn.gelu(ff_gate)
ff1 = jnp.dot(x, w_gating[1])
activations = gate_value * ff1
```

This is exactly `gelu(gate) * up`, which maps to `GatedReduce(group1=gate, group2=up, activation="gelu")`.

**Adaptation required:**
- **Kernel level:** Already done. The `use_gelu` CT arg is wired through to the TRISC compute kernel.
- **Pipeline level:** The `DenseMLP` and `DenseSwiGLU` FusedOps that compose GatedReduce need to accept and propagate an `activation` parameter. Currently `DenseSwiGLU.emit()` passes `fused_activation="silu"` by default. This needs to be parameterizable.

**Effort:** Low. The kernel work is done; only Python plumbing in the FusedOp wrappers needs updating.

---

## 2. sdpa: Prefix-LM Masking and Variable Q/KV Lengths

**TT-Blaze location:** `blaze/ops/sdpa/op.py`

**Current behavior:** `SDPADecode` supports causal masking and GQA. The kernel applies a causal mask based on position indices. From the op's docstring:

```
Supports: causal, non-paged, Q/output HEIGHT_SHARDED in L1, GQA, MLA
Not supported: attention mask (non-causal)
```

**What pi0.5 needs:** pi0.5 uses a prefix-LM attention pattern implemented via `make_attn_mask`:

```python
# From pi0.py:
def make_attn_mask(input_mask, mask_ar):
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

This produces a hybrid mask where:
- **Prefix tokens (image + language):** `mask_ar=0`, so cumsum is flat -- tokens attend bidirectionally to all prefix tokens.
- **Suffix tokens (action + state):** `mask_ar=1` at the transition, so cumsum jumps -- suffix tokens attend to all prefix tokens but only to preceding suffix tokens (causal within suffix).

There are two distinct execution modes:

**Mode A: Prefill (prefix-only)**
During `sample_actions`, the prefix is processed alone:
```python
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, ...)
```
The prefix is fully bidirectional. This is a standard prefill with a full attention mask (no causal constraint).

**Mode B: Suffix decode (with cached prefix)**
Each denoising step processes suffix tokens attending to cached prefix + self:
```python
full_attn_mask = concat([prefix_attn_mask, suffix_attn_mask], axis=-1)
# shape: (B, suffix_len, prefix_len + suffix_len)
```
The suffix-to-prefix part is fully bidirectional. The suffix-to-suffix part has a causal boundary at the first action token.

**Adaptation required:**

1. **Prefill path:** The current SDPA decode op is optimized for single-token decode. For prefill, TT-Blaze typically uses ttnn's built-in `sdpa` prefill op. The bidirectional prefix is just a standard prefill without causal masking -- this may already work if the causal mask is simply omitted.

2. **Decode path with prefix-to-suffix attention:** The suffix tokens need to attend to the full KV cache (prefix) without causal restriction. The existing SDPA decode kernel reads K/V from DRAM sequentially and applies causal masking based on `cur_pos`. For pi0.5, the mask needs to be:
   - All prefix positions: always attend (no causal mask for prefix-to-suffix).
   - Suffix positions: causal within suffix only.

   This can be approximated by setting the "current position" to the end of the prefix for all suffix tokens, so the causal mask allows attending to all prefix positions. The suffix-internal causal structure would then need additional mask bits.

3. **Variable Q length:** The suffix length varies (action_horizon + optional state token). The SDPA decode op currently handles `PNHt` (padded num heads in tiles) as a fixed parameter. For pi0.5, the "Q length" from the SDPA's perspective is the number of suffix tokens (typically 50 for action_horizon), not the number of heads.

**Effort:** Moderate to High. The core SDPA kernel may need a new mode or an explicit attention mask input instead of position-based causal masking. Alternatively, the prefill can use ttnn's built-in SDPA (which supports arbitrary masks), and the decode iterations can treat each suffix token independently with a modified position offset.

---

## 3. broadcast_rmsnorm: Extending to adaRMSNorm

**TT-Blaze location:** `blaze/ops/broadcast_rmsnorm/op.py`

**Current behavior:** `BroadcastRMSNorm` fuses CCL broadcast + RMSNorm. It broadcasts a tensor from a sender device, then applies standard RMSNorm: `x * rsqrt(mean(x^2) + eps) * (1 + scale)`.

**What pi0.5 needs:** The action expert uses adaRMSNorm, which extends RMSNorm with conditioning:

```python
# From gemma.py, RMSNorm.__call__:
if cond is None:
    # standard RMSNorm
    scale = self.param("scale", ...)
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs, None
else:
    # adaptive RMSNorm
    modulation = Dense(x.shape[-1] * 3)(cond)  # cond -> (scale, shift, gate)
    scale, shift, gate = split(modulation, 3, axis=-1)
    normed_inputs = normed_inputs * (1 + scale) + shift
    return normed_inputs, gate
```

**Adaptation required:**

The `broadcast_rmsnorm` op's RMSNorm sub-op needs an adaRMSNorm variant that:
1. Takes an additional conditioning input (the timestep embedding, shape `(B, width)`).
2. Projects it through a Dense layer to produce `3 * width` modulation values.
3. Splits into scale, shift, and gate.
4. Applies `normed * (1 + scale) + shift` instead of `normed * (1 + gamma)`.
5. Returns the gate as a side output for the downstream `gated_residual`.

The CCL broadcast part is only needed for multi-device configurations (which pi0.5 on T3K would need). For single-device, a standalone `adaRMSNorm` op suffices (see NEW tier).

**Effort:** Moderate. The normalization compute (rsqrt, multiply) can reuse the existing RMSNorm kernel. The new parts are the conditioning Dense layer (a matmul), the 3-way split, and the shift addition.

---

## 4. dense_mlp: GELU Activation Variant

**TT-Blaze location:** `blaze/ops/dense_mlp/op.py`

**Current behavior:** `DenseMLP` composes `RMSNorm -> Mcast -> SharedExpert(KNMatmul -> GatedReduce -> Mcast -> DownProj)` for dense (non-MoE) models. The SharedExpert uses SiLU activation by default.

**What pi0.5 needs:** The same pipeline but with GELU activation in the GatedReduce step.

```
RMSNorm -> Mcast -> Matmul(gate) + Matmul(up) -> GatedReduce(gelu) -> Mcast -> DownProj
```

**Adaptation required:**
- Pass `activation="gelu"` through the DenseMLP -> SharedExpert -> GatedReduce chain.
- For the action expert layers, also replace the RMSNorm step with adaRMSNorm (see NEW tier).

The `emit_mlp_preamble` utility (in `blaze/ops/utils/moe_common.py`) currently hardcodes RMSNorm. It would need a variant that accepts an adaRMSNorm conditioning input.

**Effort:** Low for the GELU change (parameter propagation). Moderate when combined with adaRMSNorm.

---

## 5. dense_swiglu: GELU Variant

**TT-Blaze location:** `blaze/ops/dense_swiglu/op.py`

**Current behavior:** `DenseSwiGLU` delegates to `SwigluOp.emit()` for the DRAM-streaming matmul pipeline:
```
Mcast -> DRAM MM (up) + DRAM MM (gate) -> eltwise_mul -> Gather -> Mcast -> DRAM MM (down)
```

The `fused_activation` parameter defaults to `"silu"` and is passed to the DRAM streaming matmul kernel.

**What pi0.5 needs:** The same pipeline with `fused_activation="gelu"`.

```python
# From dense_swiglu/op.py:
fused_activation = ua.get("fused_activation", "silu")  # change default to "gelu" for pi0.5
```

**Adaptation required:**
- The `DRAMStreamingMatmul` kernel supports a `fused_activation` parameter with mode constants (`_FUSED_ACT_NONE=0`, `_FUSED_ACT_SILU=1`, `_FUSED_ACT_CLAMPED_GATE=2`, `_FUSED_ACT_CLAMPED_UP=3`). A new `_FUSED_ACT_GELU` mode constant needs to be added to the kernel.
- The `eltwise_mul` step (gate * up) remains unchanged.

**Effort:** Low-Moderate. Adding a GELU mode to the DRAM streaming matmul kernel requires adding a GELU compute LLK call to the TRISC pack/compute path, which the Tenstorrent LLK library supports.

---

## 6. kn_sliced_matmul: Gate/Up Branching

**TT-Blaze location:** `blaze/ops/kn_sliced_matmul/`
**blaze-nn API:** `blaze_nn.F.sliced_matmul(input, weight, branch="gate")`

The KN-sliced matmul splits a large weight matrix into gate and up branches, computing them on disjoint core groups.

**pi0.5 usage:** The Gemma FFN uses `gating_einsum` with shape `(2, width, mlp_dim)`, which is a fused gate+up weight. This maps naturally to KN-sliced matmul with two branches.

**Adaptation required:** The branch naming and compute flow can be used as-is. The downstream activation change (SiLU -> GELU) is handled by the GatedReduce adaptation above.

**Effort:** None at the matmul level. The activation change propagates from the GatedReduce adaptation.

---

## Summary Table

| Op | Adaptation | Effort | pi0.5 Component |
|----|-----------|--------|------------------|
| gated_reduce | Pass `activation="gelu"` | Low | Gemma FFN gating |
| sdpa | Prefix-LM mask support, variable Q len | High | Shared attention (prefix+suffix) |
| broadcast_rmsnorm | Add adaRMSNorm conditioning path | Moderate | Action expert layer norms (multi-device) |
| dense_mlp | Propagate GELU activation param | Low | Dense FFN pipeline |
| dense_swiglu | Add `_FUSED_ACT_GELU` to DRAM streaming MM | Low-Moderate | DRAM-streaming FFN variant |
| kn_sliced_matmul | None (downstream GELU handled by gated_reduce) | None | Gate/up projection |

## Key Takeaways

1. **The GELU activation gap is the recurring theme.** Four ops (gated_reduce, dense_mlp, dense_swiglu, dram_streaming_matmul) need GELU support. The gated_reduce kernel already has it; the others need the same pattern added. This is a single-theme adaptation effort, not four independent tasks.

2. **SDPA masking is the hardest adaptation.** The prefix-LM attention pattern breaks the causal-only assumption baked into the SDPA decode kernel. This is not just a parameter change -- it may require an explicit mask input mechanism or a partitioned prefill/decode strategy where:
   - Prefill uses standard bidirectional SDPA (ttnn built-in).
   - Decode uses a modified SDPA that starts the "causal boundary" at the prefix-suffix transition.

3. **The adaRMSNorm extension to broadcast_rmsnorm is only relevant for multi-device (T3K).** On a single Wormhole chip, the CCL broadcast is unnecessary. The standalone adaRMSNorm (NEW tier) is the primary target. The broadcast_rmsnorm adaptation becomes relevant only when scaling to multi-device tensor parallelism.

4. **dense_mlp vs. dense_swiglu: one pipeline, two implementations.** DenseMLP uses L1-resident weights (Mcast -> KNMatmul -> GatedReduce). DenseSwiGLU uses DRAM-streaming weights. For pi0.5, the choice depends on whether the FFN weights fit in L1. PaliGemma's `(2048, 16384)` gate+up weight is 32M elements = ~32MB in BFP8, which likely exceeds single-chip L1. DenseSwiGLU (DRAM streaming) is the more likely path.

5. **All adaptations are backward-compatible.** Adding GELU as an activation option, adding an optional mask input to SDPA, and adding an optional conditioning input to RMSNorm do not break existing model deployments.
