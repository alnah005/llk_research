# Agent B Review -- Chapter 4

## Pass 1

**1. [02_attention_fused_ops.md, Section 4.2.2 -- Core Grid Diagram legend contradicts diagram labels for M1/M2]**

The chapter states:
> M1 = Matmul1 core (q_a_proj, 96 cores: cols 0-11 x rows 0-7)
> M2 = After Matmul2, splits into:
>      Qnope (cols 0-7 x rows 0-7, 64 cores): Matmul3 kv_b1_proj
>      Qrope (cols 8-11 x rows 0-7, 32 cores): QRoPE

But the grid diagram labels cells at cols 0-7 as `[M1]` and cells at cols 8-11 as `[M2]`. The actual source code at `pre_sdpa/op.py` lines 357 and 323-329 shows that `matmul_weights_core_grid` (Matmul1) spans all 96 cores across cols 0-11 x rows 0-7 -- not just the 64 cells labeled `[M1]` in the diagram.

This is wrong because a reader sees 64 `[M1]` cells and 32 `[M2]` cells, concluding Matmul1 uses only 64 cores. The legend claims M1 = 96 cores but the diagram only shows 64 `[M1]` labels, creating a direct contradiction. Fix: Use a single label (e.g., `[Q ]`) for all 96 cores in rows 0-7, cols 0-11, with the legend explaining that all 96 participate in Matmul1, Matmul2, and then split into Qnope (cols 0-7) and Qrope (cols 8-11) for the subsequent stage. Alternatively, label all 96 cells as `[M1]` and add a separate annotation for the post-Matmul2 Qnope/Qrope split.

**2. [03_moe_fused_ops.md, Section 4.3.6 -- down_proj role count is 4 but source has 5]**

The chapter states:
> ### Role Flags (4 roles, from source code)

And the table lists only `is_mcast_sender_core`, `is_mcast_receiver_core`, `is_matmul_core`, `is_gather_receiver_core`.

The actual source code at `down_proj/op.py` lines 474-506 defines 5 `UnifiedCompileTimeCoreDescriptor` entries, including `gather_use_per_core_sender_idx` (set to 1 on the 112-core matmul grid). Note that the same chapter's Section 4.1.4 role taxonomy table correctly lists 5 roles for `down_proj` including `gather_use_per_core_sender_idx`.

This is wrong because someone implementing `down_proj` from this section would create only 4 role descriptors, omitting the `gather_use_per_core_sender_idx` flag. Without this flag, non-rectangular gather sender indexing would not be enabled, causing the gather to produce incorrect offsets. Fix: Change "4 roles" to "5 roles" and add a fifth row to the table:

| `gather_use_per_core_sender_idx` | 112 matmul cores | Non-rectangular gather indexing |

**No further correctness issues found.**

---

## Pass 2

**Pass 1 fix verification:**
- Fix 1 (pre_sdpa grid diagram): Confirmed correct. All 96 cores in rows 0-7, cols 0-11 now show `[Q ]` with a legend explaining the 3-stage reuse (Matmul1 all 96, Matmul2 all 96, then split into Qnope cols 0-7 / Qrope cols 8-11). Matches source at `pre_sdpa/op.py` lines 323-393.
- Fix 2 (down_proj role count): Confirmed correct. Table now shows 5 roles including `gather_use_per_core_sender_idx`. Matches source at `down_proj/op.py` lines 475-506 (5 `UnifiedCompileTimeCoreDescriptor` entries).

**1. [01_fusion_principles_and_patterns.md, Section 4.1.1 -- CB overlap count "fourteen" is wrong; actual count is 17]**

The chapter states:

> In `pre_sdpa`, fourteen temporary CBs (matmul input, matmul output, RMSNorm2 buffers, QRoPE intermediates, CreateQHeads staging) are all overlapped with `sdpa_out_interm_buffer` at tracked byte offsets

The actual source at `pre_sdpa/op.py` lines 1384-1866 overlaps **17** CB descriptors with `sdpa_out_interm_buffer_device`: matmul_input (CB 5), matmul_output (CB 4), rmsnorm2_input (CB 7), gather_reduce_half1_scratch (CB 8), rmsnorm2_output (CB 9), matmul2_input (CB 10), matmul2_output (CB 12), matmul3_output (CB 14), qrope_output (CB 15), qrope_rotated_input_interm (CB 20), qrope_cos_interm (CB 21), qrope_sin_interm (CB 22), create_q_heads_interm (CB 31), dkv_matmul_output (CB 24), kv_rmsnorm_input (CB 25), kv_rmsnorm_output (CB 27), and krope_output (CB 28). The parenthetical in the chapter omits the 4 KV-path CBs entirely, and the stated count of 14 is 3 short of the actual 17.

A reader using "fourteen" as a budget figure for buffer overlap planning would underestimate the required tracking, and cross-referencing with the source would create confusion. Fix: change "fourteen" to "seventeen" and add "DKV matmul output, KV RMSNorm buffers, K-RoPE output" to the parenthetical list, or simply state "17+" and keep the parenthetical as an illustrative (non-exhaustive) sample.

**No further issues found.**

### Agent A Change Log

**Issue 1 fix (pre_sdpa grid diagram) -- `02_attention_fused_ops.md`:**
- Replaced `[M1]`/`[M2]` split labels with unified `[Q ]` label for all 96 cores in rows 0-7, cols 0-11. Legend now explains the 3-stage reuse: Matmul1 (all 96) → Matmul2 (all 96) → split into Qnope (cols 0-7, 64) and Qrope (cols 8-11, 32).

**Issue 2 fix (down_proj role count) -- `03_moe_fused_ops.md`:**
- Changed "4 roles" to "5 roles" and added `gather_use_per_core_sender_idx` row (112 matmul cores, non-rectangular gather indexing) per `down_proj/op.py` lines 474-506.

### Agent A Change Log — Pass 2

**Issue 1 fix (pre_sdpa overlapped CB count) -- `01_fusion_principles_and_patterns.md`:**
- Changed "fourteen temporary CBs" to "seventeen temporary CBs" and added "DKV matmul output, KV RMSNorm buffers, K-RoPE output" to the parenthetical list. Per `pre_sdpa/op.py` lines 1384-1866.

## Pass 3

**Pass 2 fix verification:**
- Fix 1 (pre_sdpa overlapped CB count): Confirmed correct. The text now reads "seventeen temporary CBs" and the parenthetical includes "DKV matmul output, KV RMSNorm buffers, K-RoPE output". Enumerated all 17 CBs overlapped with `sdpa_out_interm_buffer_device` from `pre_sdpa/op.py` lines 1444-1866: matmul_input (CB 5), matmul_output (CB 4), rmsnorm2_input (CB 7), gather_reduce_half1_scratch (CB 8), rmsnorm2_output (CB 9), matmul2_input (CB 10), matmul2_output (CB 12), matmul3_output (CB 14), qrope_output (CB 15), qrope_rotated_input_interm (CB 20), qrope_cos_interm (CB 21), qrope_sin_interm (CB 22), create_q_heads_interm (CB 31), dkv_matmul_output (CB 24), kv_rmsnorm_input (CB 25), kv_rmsnorm_output (CB 27), krope_output (CB 28). Count = 17. Correct.

**1. [02_attention_fused_ops.md, Section 4.2.4 -- post_sdpa role count heading says "8 base, + 2 SDPA optional" but actual is 6 base + 2 optional]**

The heading states:

> ### Role Flags (8 base, + 2 SDPA optional)

This reads as 8 base roles plus 2 additional SDPA-only roles = 10 total. The source at `post_sdpa/op.py` lines 1182-1238 creates 6 base `UnifiedCompileTimeCoreDescriptor` entries (`is_matmul4_core`, `is_gather_receiver_core`, `is_matmul5_core`, `is_mcast3_receiver_core`, `is_ccl_sender_core`, `is_ccl_receiver_core`) plus 2 conditionally added SDPA entries (`is_sdpa_worker_core`, `is_sdpa_forwarder_core`) = 8 total. The table directly below the heading correctly lists exactly 8 rows, and the Section 4.1.4 taxonomy table also correctly says `post_sdpa` has 8 roles.

A reader implementing `post_sdpa` from this heading would create 8 base descriptors plus 2 more SDPA descriptors (10 total), producing a kernel that fails or silently mis-assigns roles. Fix: change "8 base, + 2 SDPA optional" to "6 base + 2 SDPA optional" (or "8 total: 6 base + 2 SDPA optional").

**No further issues found.**

### Agent A Change Log — Pass 3

**Issue 1 fix (post_sdpa role heading) -- `02_attention_fused_ops.md`:**
- Changed "Role Flags (8 base, + 2 SDPA optional)" to "Role Flags (8 total: 6 base + 2 SDPA optional)" per `post_sdpa/op.py` lines 1182-1238.
