# Compression Analysis -- Chapter 4: The Fused-Op Library

## Summary

| File | Lines | Estimated Compressible Lines | Notes |
|---|---|---|---|
| `01_fusion_principles_and_patterns.md` | ~309 | ~15 | Minimal bloat; some overlap with Ch2 on CB reconfig and OverlappedTensor |
| `02_attention_fused_ops.md` | ~432 | ~30 | Some verbose restating of info already in 01; kv_cache_branch duplicates pre_sdpa KV-path table |
| `03_moe_fused_ops.md` | ~899 | ~95 | Heaviest redundancy: moe composition tree duplicates routed/shared expert detail; shared_expert and gated_local_reduce_down_proj share near-identical CB/semaphore/role tables; face-view explained 3 times |
| `04_output_fused_ops.md` | ~387 | ~20 | Minor: RMSNorm tile-size autodetection restated; mesh argmax reduction pattern echoes reduce_to_one_b1 |
| **Total** | **~2027** | **~160** | |

**Overall compression potential: ~8%.**

## Load-Bearing Evidence

- `01_fusion_principles_and_patterns.md` line ~99: "This is the mechanism introduced in Chapter 2 -- here we see it deployed in production for the MoE fused op" -- load-bearing because it explicitly bridges Ch2 and Ch4, orienting the reader on where the theory (Ch2) meets practice (Ch4).
- `01_fusion_principles_and_patterns.md` lines ~280-295 (Fused Op Inventory table): load-bearing because it is the only master index of all 11 fused ops with their source paths and section cross-references. This table is unique to this file and is not duplicated elsewhere.
- `02_attention_fused_ops.md` lines ~71-88 (pre_sdpa mathematical operation): load-bearing because it is the only place the full Q-path and KV-path math is collected in a single block, connecting the model-level equations from Ch1 to the hardware stages.
- `02_attention_fused_ops.md` lines ~89-114 (Core Grid Diagram): load-bearing because the ASCII grid map with role annotations is unique to this file and essential for understanding the physical core assignment.
- `03_moe_fused_ops.md` lines ~79-98 (MoeSem table): load-bearing because this is the only authoritative enumeration of all 17 MoE global semaphores with their stage associations.
- `03_moe_fused_ops.md` lines ~266-300 (DRAM Streaming Matmul Details in routed expert): load-bearing because it documents expert-indexed DRAM addressing, triple buffering parameters, and per-core bank/VC assignment specific to the MoE context (distinct from the generic dram_streaming_matmul micro-op in Ch3).
- `04_output_fused_ops.md` lines ~232-280 (Mesh Argmax section): load-bearing because the two-stage hierarchical mesh reduction and scratch buffer slot layout are unique to lm_head_sampling and not documented elsewhere.

## CRUCIAL Suggestions

None identified. The redundancies in this chapter are of the "repeated explanation" and "near-duplicate table" variety, not structural issues that would confuse or mislead a reader.

## MINOR Suggestions

### `01_fusion_principles_and_patterns.md` ~lines 41-99 (Section 4.1.2 CB Reconfig and CircularBufferIdManager)
**Issue:** This section re-explains `CircularBufferIdManager`, the Context pattern, dummy CBs, the reconfig tensor, and the 264-word shard layout. Chapter 2 Section 2.3 already provides a comprehensive treatment of all of these (2.3.1 through 2.3.5), including the exact same code snippets (`_allocate_id`, `create_context`/`get_cb_id`), the same 264-word table layout, and the same `build_cb_reconfig_tensor` call pattern. The Ch4 version adds one sentence of new context ("here we see it deployed in production for the MoE fused op"), but the remaining ~55 lines are restated material.
**Suggestion:** Condense to a 5-line summary referencing Ch2 Section 2.3, keeping only the MoE-specific deployment note and the fact that the MoE orchestrator creates separate contexts for routed and shared paths. Remove the duplicated code snippet and table.

### `01_fusion_principles_and_patterns.md` ~lines 186-224 (Section 4.1.5 OverlappedTensor and Weight Packing)
**Issue:** The `OverlappedTensor` dataclass definition and the `cb_descriptor_from_overlapped_tensor` function are already shown in Ch2 Section 2.3.2 (Pattern 2). The only new information here is the enumeration of which fused ops use it (pre_sdpa, post_sdpa). The dataclass and function code are repeated verbatim.
**Suggestion:** Replace the code blocks with a forward-reference to Ch2 Section 2.3.2, retaining only the list of which fused ops use overlapped tensors and what specific weight buffers are packed.

### `02_attention_fused_ops.md` ~lines 242-302 (Section 4.2.3 kv_cache_branch)
**Issue:** The kv_cache_branch stage pipeline table (lines 290-296) is a near-exact subset of the pre_sdpa KV-path pipeline table (lines 220-227 in the same file). The five stages (DKV Matmul, DKV Gather, KV RMSNorm, K-RoPE, KV Cache Write) match one-to-one. The section even acknowledges this: "This fused op is the standalone version of the KV branch that also appears embedded in pre_sdpa."
**Suggestion:** After the opening sentence that establishes the relationship, add a sentence noting which CB indices differ from pre_sdpa, and collapse the pipeline table into a diff-style comparison or a note saying "Stages are identical to pre_sdpa KV-path (Section 4.2.2) with remapped CB indices" followed by a small CB-index mapping table. This would save ~25 lines.

### `03_moe_fused_ops.md` ~lines 17-59 (Section 4.3.1 moe Composition Tree)
**Issue:** The composition tree on lines 17-59 fully enumerates every stage of both the routed expert path and the shared expert path. But the routed expert path is then re-enumerated in Section 4.3.2 (lines 147-172) with its own composition tree, and the shared expert path is re-enumerated in Section 4.3.3 (lines 402-428). The top-level tree and the per-op trees are almost identical in content.
**Suggestion:** Trim the top-level composition tree to show only the two top-level branches (routed and shared) with a one-line summary of each ("14-stage pipeline, see 4.3.2" and "11-stage pipeline, see 4.3.3"), removing the per-stage sub-bullets. This saves ~30 lines of duplicated stage listings.

### `03_moe_fused_ops.md` ~lines 533-537 and ~lines 746-756 (Face-View in shared_expert and gated_local_reduce_down_proj)
**Issue:** The face-view optimization is explained three times across the chapter: once in Section 4.1.3 (lines 103-138 in file 01), once in the shared_expert section (lines 533-537), and once in gated_local_reduce_down_proj (lines 746-756). Each instance re-derives the same arithmetic ($8 \times 32 = 256 = 16 \times 16$). The Ch3 section on `local_reduce` (3.1.5) also explains face-view.
**Suggestion:** The explanation in 4.1.3 is sufficient. The shared_expert and gated_local_reduce_down_proj sections should cross-reference 4.1.3 rather than re-deriving. A single sentence like "Face-view is active when `can_use_face_view` returns True (see Section 4.1.3)" replaces ~10 lines in each section.

### `03_moe_fused_ops.md` ~lines 720-745 (gated_local_reduce_down_proj Semaphore Map and Role Flags)
**Issue:** The semaphore map (lines 720-732) and role flags table (lines 734-745) for `gated_local_reduce_down_proj` are nearly identical to those of `shared_expert` (lines 504-529). Seven of 8 role flags are the same; 7 of 9 semaphores share the same purpose descriptions. The only differences are the input-gather semaphores (IDs 4-7) replacing A/B gather semaphores.
**Suggestion:** Add a note: "Semaphore and role layout matches `shared_expert` (Section 4.3.3) except that IDs 4-7 serve input gather instead of A/B gather." Then show only the differing entries, not the full tables. Saves ~20 lines.

### `03_moe_fused_ops.md` ~lines 767-878 (Section 4.3.6 down_proj)
**Issue:** The `down_proj` section is thorough but much of its content (mcast+matmul+residual_add+gather pattern, 130-core grid, 112 matmul cores, DRAM worker positions, phantom cores) is restated nearly verbatim when describing the same pattern inside `shared_expert` (lines 418-428, 478-482) and `gated_local_reduce_down_proj` (lines 677-700). The section itself notes: "This is the prototypical mcast-matmul-gather building block."
**Suggestion:** Since the section already identifies itself as the prototype, add a short note at the top of the shared_expert and gated_local_reduce_down_proj down-proj stages saying "The down projection stage follows the canonical `down_proj` pattern (Section 4.3.6)" and omit the re-listing of the same core count breakdown and grid diagram in those sections. This is a minor structural improvement rather than a line-count savings within file 03 itself, since all three patterns live in the same file.

### `04_output_fused_ops.md` ~lines 200-209 (RMSNorm tile-size autodetection)
**Issue:** The tile-size autodetection logic (`is_16x32_tile = (input_shape[1] // 32) % 32 != 0`) and the worked example for 7168 elements are restated here from both `broadcast_rms` (Section 4.2.1, file 02 line 58) and the RMSNorm micro-op itself (Ch3 Section 3.1.2, lines 180-188). The code snippet and the "For the standard [1, 7168] input..." derivation appear three times total across the guide.
**Suggestion:** Replace with a one-line cross-reference: "RMSNorm uses the standard tile-size autodetection (see Section 3.1.2) -- for the standard [1, 7168] input, this yields 7 full 32x32 tiles."

### `04_output_fused_ops.md` ~lines 186-198 (Stage 1: CCL Broadcast)
**Issue:** The CCL broadcast description here (packet size, ring routing, forward/backward hop counts, secondary-axis relay) restates material from both `broadcast_rms` (Section 4.2.1) and the CCL broadcast micro-op (Ch3 Section 3.3.2). The ring routing code snippet on lines 193-195 is the same formula shown in `broadcast_rms`.
**Suggestion:** Condense to: "CCL broadcast follows the same protocol as `broadcast_rms` (Section 4.2.1) with 14 KB packets. See Section 3.3.2 for the underlying micro-op." Remove the ring routing formula restatement.

### `03_moe_fused_ops.md` ~lines 302-322 (Fused Mul Stage CB Aliasing in moe_routed_expert)
**Issue:** The CB aliasing technique ($1 \times 32$ tiles viewed as $16 \times 16$ tiles for element-wise multiply) is explained here and also in Ch3 Section 3.2.5.8 (dram_streaming_matmul fused SiLU + multiply + scalar chain). The formula `mul_num_tiles = ceil(total_elements / 256)` appears in both places.
**Suggestion:** Add a cross-reference to Ch3 Section 3.2.5.8 and remove the formula restatement. Keep the MoE-specific worked example (per_core_n=8 tiles yielding 1 face tile) as it adds context.

## VERDICT
Crucial updates: no
