# Agent B Review — Chapter 5

## Pass 1

**1. [01_mla_query_projection_and_weight_layout.md, Section 5.1.3 / 04_post_sdpa_output_projection.md, Section 5.4.4 — kv_b2_proj sharding type and shard shape are wrong]**

The weight shape catalog (Section 5.1.3, line 70) states that `kv_b2_proj` is **WIDTH_SHARDED** with shard shape `[512, 128]`. Section 5.4.4 repeats this: "Sharding: WIDTH_SHARDED on 64 cores. Shard: [512, 128] per core."

The source code (`KVB12_PROJ_SingleDeviceOverlapSpec` in `blitz_decode_weights.py`, lines 344-421) shows the opposite. The logical shape `(512, 8192)` is tile-rearranged by `shuffle_kv_b2()` into `(8192, 512)`, after which it is **HEIGHT_SHARDED** with shard shape `(128, 512)`. The docstring at line 385 explicitly states: "HEIGHT_SHARDED shard: (128, 512) -- same for both sub-tensors." A reader implementing the kv_b2 weight layout from this chapter would use the wrong sharding strategy and the transposed shard dimensions, producing an incorrect on-device layout.

**2. [03_flash_mla_decode.md, Section 5.3.2 — Online softmax pseudocode applies scale inconsistently, leading to double-scaling if implemented literally]**

Step 1 of the online softmax pseudocode (line 97) states: "scores = Q @ K_chunk^T" with the comment "scores = Q @ K^T" and then Step 3 (lines 105-110) applies `exp((m - m_new) * scale)` and Step 5 applies `exp((scores - m_new) * scale)`. Taken together, the pseudocode says scores are raw un-scaled QK^T products and the scale is folded into the exp -- which is how the hardware actually works (see `init_fast_approx_exp_constants` in `sdpa.h`, where `A_scaled = A * scale_fp32` bakes the scale into the exponential instruction).

However, the narrative text at line 66 says: "Compute local scores: scores = Q * K_chunk^T * alpha where alpha = 1/sqrt(576) is the scale factor." This explicitly claims the scores are pre-multiplied by alpha. If scores are already scaled by alpha, then the max `m` tracks scaled values, and the correction should be `exp(m_old - m_new)` WITHOUT the extra `* alpha` shown in Step 3. Someone implementing from this description would apply the scale factor twice -- once when computing scores and again inside the exp -- producing incorrect softmax values.

The fix: either (a) remove `* alpha` from the score computation in Step 1 and the narrative (since the hardware folds it into exp), or (b) remove `* alpha` from the exp terms in Steps 3 and 5.

**3. [01_mla_query_projection_and_weight_layout.md, Section 5.1.1 — Compression ratio baseline is wrong]**

Section 5.1.1 (line 11) states: "Standard multi-head attention stores separate K and V projections for all heads -- for 128 heads at 128+64 dimensions each, that is 24,576 elements per token. MLA instead stores a single compressed latent vector of 576 elements... a 42.7x compression ratio."

The 24,576 figure counts only 128 x (128+64) = 128 x 192, which is the dimension of a single K projection (nope+rope per head). But the text explicitly says "K and V projections" -- standard MHA stores BOTH K and V per token. A standard KV cache for 128 heads with head_dim=128 for both K and V would be 128 x 128 x 2 = 32,768 elements (or 128 x 192 + 128 x 128 = 40,960 if K includes the rope dim). The 42.7x ratio (= 24,576 / 576) is computed from only K, not K+V. A reader trying to verify the compression ratio from the stated premise ("stores separate K and V projections") would get a different number.

**4. [03_flash_mla_decode.md, Section 5.3.3 — DST register layout uses "bytes" as the size unit but these are tile-slot indices]**

Section 5.3.3 (lines 140-144) describes the DST register layout with columns labeled "Size" containing values like "2 bytes", "14 bytes", "16 bytes". These are not byte counts -- they are tile-slot counts (indices into the DST register file). The DST register file is addressed by tile indices, not bytes. For example, `max_dst_offset` is described as "2 bytes" but it actually occupies 2 tile slots (from index `16*vDHt` to `16*vDHt + 2`). Using "bytes" here would mislead someone computing L1 or register usage into off-by-orders-of-magnitude estimates. These should be labeled as "tile slots" or "register indices."

**No further issues flagged for this pass.**

### Agent A Change Log

**Issue 1 fix (kv_b2 sharding) -- files 01 + 04:**
- Changed kv_b2_proj from WIDTH_SHARDED [512,128] to HEIGHT_SHARDED [128,512] in both the weight catalog (file 01 line 70) and the weight layout section (file 04 lines 46, 175-185). Per `blitz_decode_weights.py` KVB12_PROJ_SingleDeviceOverlapSpec.

**Issue 2 fix (online softmax double-scaling) -- file 03:**
- Removed `* alpha` from score computation in Step 1 (line 66). Now reads "raw dot products, no pre-scaling" with scale factor introduced only in the exp operations (Steps 3, 5), matching the hardware implementation.

**Issue 3 fix (compression ratio) -- file 01:**
- Corrected baseline from K-only (24,576) to K+V (40,960 = 128 × (192+128)). Updated compression ratio from 42.7x to 71x element-level, ~134x with BFP8.

**Issue 4 fix (DST register units) -- file 03:**
- Changed "bytes" to "tile slots" in all 5 rows of the DST register layout table (lines 140-144).

---

## Pass 2

### Fix Verification

1. **kv_b2_proj sharding (Issue 1):** PARTIALLY FIXED. File 01 Section 5.1.3 (line 70) correctly reads `HEIGHT` with shard `[128, 512]`. File 04 Section 5.4.4 (line 180) correctly reads `[128, 512] per core`. However, file 04 Section 5.4.11 table (line 484) still reads `(512, 128) = 64 tiles` -- the shard shape was not updated in this table. See new issue 1 below.
2. **Online softmax (Issue 2):** VERIFIED FIXED. Step 1 now reads "raw dot products, no pre-scaling." Scale alpha appears only inside exp in Steps 3 and 5. Matches `sdpa.h` compute flow: `sdpa_custom_mm_block` (unscaled QK^T) -> `sdpa_sub_bcast` (scores - m_new) -> `init_fast_approx_exp_constants<scale_fp32>` + `fast_approx_exp` (exp with scale).
3. **Compression ratio (Issue 3):** VERIFIED FIXED. Baseline is 40,960 elements (K+V); ratio is 71x element-level, ~134x with BFP8.
4. **DST register units (Issue 4):** VERIFIED FIXED. Table column reads "tile slots" throughout.

### New Issues

**1. [04_post_sdpa_output_projection.md, Section 5.4.11 line 484 -- kv_b2_proj shard shape still wrong in summary table]**

The `kv_b12 Fused Buffer` table in Section 5.4.11 lists the `kv_b2_proj` per-core shard as `(512, 128)`. After `shuffle_kv_b2()` rearranges the logical `(512, 8192)` into `(8192, 512)` for HEIGHT_SHARDED placement, each core's physical shard is `(128, 512)` -- matching the `shard_shape` property at `blitz_decode_weights.py` line 385 and the correct description in Section 5.4.4 (line 180) of the same file. The table should read `(128, 512) = 64 tiles` to be consistent with Section 5.4.4, file 01 Section 5.1.3, and the source code. This is a remnant of Pass 1 Issue 1.

**2. [01_mla_query_projection_and_weight_layout.md, Section 5.1.4 lines 123-127 -- kv_b_proj split uses wrong num_heads and output shapes for the full-model HF weight]**

The kv_b_proj split example shows:

```
kv_b_proj:  [32768, 512]   (num_heads=64, head_dim=256, kv_lora_rank=512)
  reshape:  [64, 256, 512]  (per-head view)
  kv_b1:    w[:, :128, :].reshape(-1, 512)  = [8192, 512]
  kv_b2:    w[:, 128:, :].reshape(-1, 512).T = [512, 8192]
```

The input shape `[32768, 512]` is the full-model HF weight (128 heads total across both TP devices). But `num_heads=64` gives `64 * 256 = 16384`, not 32768. The `_split_kv_b_proj` function in `prepare_weights.py` (line 220) computes `num_heads = 32768 // 256 = 128`. The correct full-model split produces `kv_b1 = [16384, 512]` and `kv_b2 = [512, 16384]`; the per-TP=2 slice then yields `[8192, 512]` and `[512, 8192]`. The section should either use the full-model head count (128) with the full-model output shapes (16384), or use the per-device input shape `[16384, 512]` with `num_heads=64`. As written, the input dimension and the head count are mathematically inconsistent (`64 * 256 != 32768`), which would confuse a reader implementing the split.

### Agent A Change Log — Pass 2

**Issue 1 fix (kv_b2 summary table remnant) -- file 04:**
- Changed shard from `(512, 128)` to `(128, 512)` and logical shape from `(512, 8192)` to `(8192, 512)` in the fused buffer summary table (line 484).

**Issue 2 fix (kv_b_proj split num_heads) -- file 01:**
- Changed num_heads from 64 to 128 in the kv_b_proj split example (lines 123-127). Now shows full-model shapes: reshape to [128, 256, 512], kv_b1=[16384, 512], kv_b2=[512, 16384]. Added per-TP=2 shard step producing the per-device [8192, 512] and [512, 8192] shapes.
