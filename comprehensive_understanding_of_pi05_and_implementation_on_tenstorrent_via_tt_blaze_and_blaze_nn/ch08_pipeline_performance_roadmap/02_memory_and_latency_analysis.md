# 02 -- Memory and Latency Analysis

## Context

This section answers two questions: (1) Does pi0.5 fit on a single Tenstorrent device? (2) Can it run fast enough for real-time robot control? The analysis uses concrete numbers derived from the model source code and Tenstorrent hardware specifications. All parameter counts are computed from the architecture configs in `/tmp/openpi/src/openpi/models/gemma.py` and `/tmp/openpi/src/openpi/models/siglip.py`.

## Parameter Count and Memory Budget

### Component-by-Component Breakdown

**PaliGemma Expert (Gemma-2B):**
- Config: width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256
- Per-layer attention:
  - Q projection: 8 heads x 2048 x 256 = 4,194,304
  - KV projection: 2 x 1 head x 2048 x 256 = 1,048,576
  - Output projection: 8 heads x 256 x 2048 = 4,194,304
  - Subtotal per layer: 9,437,184 (~9.4M)
- Per-layer MLP:
  - Gating einsum: 2 x 2048 x 16384 = 67,108,864
  - Linear: 16384 x 2048 = 33,554,432
  - Subtotal per layer: 100,663,296 (~100.7M)
- Per-layer norms: 2 x 2048 = 4,096
- Per-layer total: ~110.1M
- 18 layers: ~1,982M
- Embedder: 257,152 x 2048 = 526,639,616 (~527M)
- Final norm: 2,048
- **PaliGemma total: ~2,509M parameters**

**Action Expert (Gemma-300M):**
- Config: width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
- Per-layer attention:
  - Q projection: 8 x 1024 x 256 = 2,097,152
  - KV projection: 2 x 1 x 1024 x 256 = 524,288
  - Output projection: 8 x 256 x 1024 = 2,097,152
  - Subtotal: 4,718,592 (~4.7M)
- Per-layer MLP:
  - Gating: 2 x 1024 x 4096 = 8,388,608
  - Linear: 4096 x 1024 = 4,194,304
  - Subtotal: 12,582,912 (~12.6M)
- Per-layer norms: 2 x 1024 = 2,048 (adaRMSNorm has additional Dense: 1024 x 3072 per norm)
  - adaRMS Dense per norm: 1024 x (1024 x 3) = 3,145,728
  - Two norms per layer: 2 x 3,145,728 = 6,291,456
- Per-layer total (with adaRMS): ~23.6M
- 18 layers: ~424.3M
- Final norm: 1,024
- **Action Expert total: ~424M parameters**

**SigLIP So400m/14:**
- Config: width=1152, depth=27, mlp_dim=4304, num_heads=16, patch_size=14
- Per-layer attention: 4 x 1152^2 / 16-heads MHSA = ~5.3M (standard MHSA, not GQA)
  - K,Q,V + out projections: 4 x 1152 x 1152 = 5,308,416
- Per-layer MLP: 1152 x 4304 + 4304 x 1152 + biases = ~9.9M
- Per-layer norms: 2 x 1152 x 2 (weight + bias) = 4,608
- Per-layer total: ~15.2M
- 27 layers: ~410.4M
- Patch embedding Conv: 3 x 14 x 14 x 1152 = 677,376
- Positional embedding: 256 x 1152 (sincos, not learned) = 0 params
- Projection head: 1152 x 2048 = 2,359,296
- **SigLIP total: ~413M parameters**

**Projection Layers (pi0.5 specific):**
- action_in_proj: 32 x 1024 = 32,768
- time_mlp_in: 1024 x 1024 = 1,048,576
- time_mlp_out: 1024 x 1024 = 1,048,576
- action_out_proj: 1024 x 32 = 32,768
- **Projections total: ~2.2M parameters**

### Summary Table

```
+---------------------------+------------+------------+
| Component                 | Params (M) | bf16 (MB)  |
+---------------------------+------------+------------+
| PaliGemma (Gemma-2B)      |   2,509    |   4,779    |
| Action Expert (Gemma-300M)|     424    |     808    |
| SigLIP So400m/14          |     413    |     787    |
| Projection layers         |       2    |       4    |
+---------------------------+------------+------------+
| TOTAL                     |   3,348    |   6,378    |
+---------------------------+------------+------------+
```

Note: The embedder table (527M params, ~1,003 MB) is shared -- it belongs to PaliGemma and is used during both prefix (Stage 2) and suffix text embedding. For pi0.5 with `discrete_state_input=True`, the state is tokenized through this same embedder.

Rounding to practical estimates: **~3.3B parameters, ~6.4 GB at bf16, ~3.2 GB at BFP8.**

### Does It Fit?

```
+---------------------------------------------+
|          Wormhole / Blackhole DRAM           |
|                  12 GB                       |
+---------------------------------------------+
| Model params (bf16):     ~6.4 GB            |
| KV cache (bf16):         ~0.02 GB           |
| Activations + workspace: ~0.5 GB (est.)     |
| TTNN runtime overhead:   ~0.5 GB (est.)     |
+---------------------------------------------+
| TOTAL ESTIMATED:         ~7.4 GB            |
| HEADROOM:                ~4.6 GB            |
+---------------------------------------------+
```

**Yes, pi0.5 fits on a single device at bf16 with comfortable headroom.** At BFP8 weights (~3.2 GB), the headroom doubles to ~8 GB, leaving ample room for larger batch sizes or double-buffered weight streaming.

## KV Cache Analysis

The KV cache is generated once (Stage 2) and read repeatedly (Stage 3, 10 times).

**Cache dimensions:**
- Layers: 18 (shared between PaliGemma and Action Expert in the same block)
- Batch size: B (typically 1 for robot control)
- Prefix length: N_images * 256 + max_token_len
  - 3 images x 256 patches = 768 image tokens
  - max_token_len = 200 (pi0.5 config)
  - Total prefix_len = 968 tokens (upper bound; actual depends on prompt length and image masking)
- KV heads: 1 (GQA with 1 KV head)
- Head dim: 256

**Cache size per K or V tensor:**
```
18 layers x 1 batch x 968 tokens x 1 KV_head x 256 head_dim x 2 bytes (bf16)
= 18 x 968 x 256 x 2
= 8,929,280 bytes
= ~8.5 MB per tensor (K or V)
```

**Total KV cache: ~17 MB for K+V at batch_size=1.**

This is tiny relative to DRAM capacity. Even at batch_size=8, the KV cache would be ~136 MB. The cache fits entirely in DRAM with no pressure on the memory budget.

**L1 residency:** With ~1.5 MB L1 per Tensix core and 64+ cores on a Wormhole device, total L1 capacity is ~96 MB. The full KV cache for a single batch could fit in L1 across the core grid, enabling zero-DRAM-access KV reads during denoising. This is a significant optimization opportunity.

## Compute Breakdown

### Per-Stage FLOP Estimates (batch_size=1)

**Stage 1: SigLIP Encode**

For each image (256 tokens through 27 layers):
- Per-layer attention: 4 x 256 x 256 x 1152 (Q/K/V projections) + 256 x 256 x 72 (MHSA softmax*V, per head) x 16 heads
  - Projections: 4 x 256 x 1152^2 x 2 = ~2.7 GFLOP
  - Attention: 2 x 256^2 x 1152 x 2 = ~0.3 GFLOP
  - Subtotal: ~3.0 GFLOP per layer
- Per-layer MLP: 2 x 256 x 1152 x 4304 x 2 = ~5.1 GFLOP
- Per-layer total: ~8.1 GFLOP
- 27 layers: ~219 GFLOP
- 3 images: **~657 GFLOP**

**Stage 2: Prefix + KV Fill**

968 tokens through 18 layers (PaliGemma expert only, width=2048):
- Per-layer attention (GQA):
  - Q: 968 x 2048 x (8 x 256) x 2 = ~8.1 GFLOP
  - KV: 968 x 2048 x (2 x 256) x 2 = ~2.0 GFLOP
  - Softmax*V: 2 x 968^2 x 2048 x 2 = ~7.7 GFLOP
  - Output: 968 x (8 x 256) x 2048 x 2 = ~8.1 GFLOP
  - Subtotal: ~25.9 GFLOP
- Per-layer MLP: 3 x 968 x 2048 x 16384 x 2 = ~194.0 GFLOP (gating + up + down)
- Per-layer total: ~220 GFLOP
- 18 layers: **~3,960 GFLOP**

**Stage 3: Denoising Iteration (one step)**

Suffix length = 50 (action_horizon) attending to prefix_len + suffix_len = 968 + 50 = 1018 KV positions.

PaliGemma expert is `None` during denoising (only KV cache is used). But the attention still computes Q from suffix against full K/V. The computation involves both experts:

- Action input projection: 50 x 32 x 1024 x 2 = ~3.3 MFLOP
- Time MLP: 2 x 1024 x 1024 x 2 = ~4.2 MFLOP
- Per-layer dual-expert attention:
  - Action expert Q: 50 x 1024 x (8 x 256) x 2 = ~0.42 GFLOP
  - Action expert KV: 50 x 1024 x (2 x 256) x 2 = ~0.10 GFLOP
  - Cross-attention with cached prefix: attention is suffix-to-(prefix+suffix)
    - The suffix queries (50 tokens) attend to prefix KV (968 cached) + suffix KV (50 new)
    - Logits: 2 x 50 x 1018 x (8 heads x 256) x 2 = ~0.42 GFLOP (accounts for GQA expansion)
  - Output: 50 x (8 x 256) x 1024 x 2 = ~0.21 GFLOP
  - Subtotal: ~1.15 GFLOP
- Per-layer MLP (action expert only):
  - Gating + up + down: 3 x 50 x 1024 x 4096 x 2 = ~1.26 GFLOP
- Per-layer adaRMSNorm: 2 x 1024 x 3072 x 2 = ~12.6 MFLOP (Dense for modulation)
- Per-layer total: ~2.42 GFLOP
- 18 layers: ~43.6 GFLOP
- Action output projection: 50 x 1024 x 32 x 2 = ~3.3 MFLOP
- **Per iteration total: ~44 GFLOP**
- **10 iterations: ~440 GFLOP**

**Stage 4: Action Output**
- Negligible (just the final Euler step: 50 x 32 multiply-adds)

### Summary

```
+---------------------------+------------+--------+
| Stage                     | GFLOP      | % Time |
+---------------------------+------------+--------+
| Stage 1: SigLIP (3 imgs)  |     657    |  13.0% |
| Stage 2: Prefix + KV      |   3,960    |  78.3% |
| Stage 3: Denoise (10x)    |     440    |   8.7% |
| Stage 4: Action Output    |      ~0    |   0.0% |
+---------------------------+------------+--------+
| TOTAL                     |   5,057    | 100.0% |
+---------------------------+------------+--------+
```

**Important nuance:** FLOP counts do not equal wall-clock time. Stage 2 has the most FLOPs but runs once. Stage 3 has fewer FLOPs per iteration but is the bottleneck because:
1. It runs 10 times.
2. Each iteration is heavily memory-bandwidth-bound (small suffix, large weights to stream).
3. The dual-expert architecture means two sets of MLP weights per layer.

## Bandwidth and Latency Analysis

### The Denoising Bottleneck: Bandwidth-Bound, Not Compute-Bound

For Stage 3 (denoising), each iteration processes only 50 suffix tokens through 18 layers of dual-expert weights. This is a classic "decode" scenario: small activation tensors, large weight tensors. The compute is limited by how fast weights can be streamed from DRAM.

**Weight data per denoising iteration (action expert only, since PaliGemma expert is not re-applied to suffix):**
- Per-layer attention weights: Q(8 x 1024 x 256) + KV(2 x 1024 x 256) + Out(8 x 256 x 1024) = 5.2M params
- Per-layer MLP weights: gating(2 x 1024 x 4096) + linear(4096 x 1024) = 12.6M params
- Per-layer adaRMS Dense: 2 x 1024 x 3072 = 6.3M params
- Per-layer norms: negligible
- Per-layer total: ~24.1M params = ~48.2 MB at bf16
- 18 layers: ~434M params = **~867 MB per iteration**

**Note on PaliGemma weights during denoising:** Although the PaliGemma expert input is `None` (no new tokens), the KV cache was filled in Stage 2. During Stage 3, only the action expert's Q/K/V projections and MLP run. However, the *attention computation* reads the cached K/V which were produced by the PaliGemma expert. The PaliGemma expert's *weights* are not needed during Stage 3 -- only its cached KV output.

**DRAM bandwidth on Wormhole:** ~256 GB/s aggregate across all channels.

**Time to stream action expert weights per iteration:**
```
867 MB / 256 GB/s = ~3.4 ms per iteration
```

**At BFP8 (half the weight size):**
```
434 MB / 256 GB/s = ~1.7 ms per iteration
```

**Plus KV cache reads per iteration:**
```
17 MB / 256 GB/s = ~0.07 ms  (negligible)
```

**Total bandwidth-limited time for 10 denoising iterations:**
```
bf16:  10 x 3.4 ms = ~34 ms
BFP8:  10 x 1.7 ms = ~17 ms
```

### Full Pipeline Latency Estimate

```
+---------------------------+-----------+-----------+
| Stage                     | bf16 (ms) | BFP8 (ms) |
+---------------------------+-----------+-----------+
| Stage 1: SigLIP           |   5-8     |   3-5     |
| Stage 2: Prefix + KV      |   8-15    |   5-10    |
| Stage 3: Denoise (10x)    |  34-50    |  17-25    |
| Stage 4: Action Output    |   <0.1    |   <0.1    |
| Host overhead             |   1-2     |   1-2     |
+---------------------------+-----------+-----------+
| TOTAL                     |  48-75    |  26-42    |
+---------------------------+-----------+-----------+
| Control frequency         | 13-21 Hz  | 24-38 Hz  |
+---------------------------+-----------+-----------+
```

**Target:** Robot control typically requires 10-50 Hz. The bf16 baseline may achieve the low end (13-21 Hz). BFP8 weights bring us into the mid-range (24-38 Hz). Further optimizations are needed for the high end.

## Optimization Levers

### Lever 1: Reduce Denoising Steps (Highest Impact)

The reference implementation defaults to `num_steps=10` but the `sample_actions` interface accepts any value. Recent flow-matching literature shows that distilled models can achieve comparable quality with 4-5 steps.

```
Impact: 10 steps -> 5 steps halves Stage 3 latency.
         bf16: ~34 ms -> ~17 ms (total: ~31-47 ms, 21-32 Hz)
         BFP8: ~17 ms -> ~8.5 ms (total: ~17-25 ms, 40-59 Hz)
```

This is the single most impactful lever and requires no hardware or kernel changes -- only model-level validation that reduced steps maintain action quality.

### Lever 2: BFP8 Weight Quantization

Convert model weights from bf16 to block floating point 8-bit. Tenstorrent hardware natively supports BFP8 in the matrix engine.

```
Impact: Halves weight streaming time for all stages.
Risk: Potential accuracy degradation, especially in the adaRMSNorm
      conditioning path where the timestep signal is subtle.
Validation: Compare action MSE between bf16 and BFP8 on standard
            robot manipulation benchmarks.
```

### Lever 3: L1 Caching of Action Expert Weights

The action expert is only ~424M params (~808 MB bf16, ~404 MB BFP8). With 96 MB of total L1 across the core grid, a subset of the hottest weights (attention projections: ~94 MB at bf16, or ~47 MB at BFP8) could be pinned in L1 across iterations.

```
Impact: Eliminates DRAM reads for cached weights.
         If attention weights are L1-cached at BFP8:
         ~47 MB saved per iteration x 10 = 470 MB less DRAM traffic
         Time saved: ~1.8 ms total (10 iterations)
Complexity: Requires careful L1 allocation planning via
            BlazeCompiler's CB engine. Weights must remain
            pinned across the 10 loop iterations.
```

### Lever 4: Batching

Processing multiple observations in a single batch amortizes weight streaming across batch elements. Each additional batch element adds only activation memory (small) while reusing the same weight stream.

```
Impact at batch_size=4:
  Weight streaming time unchanged: ~3.4 ms per iteration (bf16)
  Compute per iteration: 4x more FLOPs but still bandwidth-bound
  Throughput: ~4x improvement (4 robots controlled simultaneously)
  KV cache: 4 x 17 MB = 68 MB (still small)
  
Memory cost: 4x activations + 4x KV cache = negligible
             Weights are shared across batch.
```

Batching is particularly effective here because the denoising iterations are bandwidth-bound. The matrix units are underutilized at batch_size=1.

### Lever 5: Op Fusion

Key fusion opportunities in the denoising loop body:

1. **RMSNorm + Dense (adaRMSNorm):** Fuse the normalization with the modulation Dense layer. The `RMSNorm.__call__` in `gemma.py` (lines 113-131) shows the pattern: normalize, then apply Dense to get scale/shift/gate, then modulate. This is three sequential ops that can be a single Blaze FusedOp.

2. **Attention QKV + SDPA:** Fuse Q/K/V projections with the scaled dot-product attention into a single Blaze FusedOp. The DeepSeek pipeline already does this via `_gqa_attention.py`.

3. **Gated MLP (gate + up + GELU + down):** The `FeedForward` module computes `GELU(x @ W_gate) * (x @ W_up)`, then `result @ W_down`. This is exactly the `GatedReduce` + `DownProj` pattern from DeepSeek's `SharedExpert`.

4. **Euler step + action re-projection:** The Euler update `x_t + dt * v_t` followed by `action_in_proj` at the start of the next iteration can be fused.

```
Expected impact of full fusion: 10-30% reduction in per-iteration
latency from eliminated activation reads/writes between ops.
```

### Lever 6: Skip PaliGemma Expert Computation in Stage 3

During denoising, the PaliGemma expert receives `None` input. The reference code at `gemma.py` line 173 skips Q/K/V projection for `None` inputs:

```python
for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
    if x is None:
        continue
```

Ensuring the TT implementation truly skips all PaliGemma weight reads during denoising is critical. A naive implementation that streams both expert weights would double the bandwidth requirement.

### Combined Optimization Scenario

```
Baseline bf16, 10 steps:                    ~48-75 ms (13-21 Hz)

Apply Lever 1 (5 steps):                    ~31-47 ms (21-32 Hz)
Apply Lever 2 (BFP8):                       ~17-25 ms (40-59 Hz)
Apply Lever 3 (L1 cache action attn):       ~15-23 ms (43-67 Hz)
Apply Lever 5 (op fusion, 15% est.):        ~13-20 ms (50-77 Hz)

Target range for aggressive optimization:   50+ Hz
```

This suggests that the 50 Hz target is achievable with the combination of step reduction, quantization, and L1 caching, without requiring multi-device parallelism.

## Key Takeaways

- Pi0.5 fits comfortably on a single device: ~6.4 GB model params (bf16) + ~17 MB KV cache against 12 GB DRAM.
- The denoising loop is bandwidth-bound, not compute-bound. Weight streaming dominates at ~3.4 ms per iteration (bf16) or ~1.7 ms (BFP8).
- Step reduction is the highest-impact optimization: halving steps from 10 to 5 halves Stage 3 latency.
- With aggressive optimization (5 steps + BFP8 + L1 caching + fusion), 50+ Hz control frequency appears achievable on a single device.
- The PaliGemma expert weights must not be read during denoising iterations. This is a correctness requirement that also has major performance implications.

---

**Next:** [`03_implementation_roadmap.md`](./03_implementation_roadmap.md)
