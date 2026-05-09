# Compression Analysis -- Chapter 5: Multi-head Latent Attention Deep Dive

**Agent:** C (Compressor, Pass 1)
**Date:** 2026-05-08
**Files reviewed:** 01 (514 lines), 02 (416 lines), 03 (771 lines), 04 (641 lines) -- 2342 lines total
**Prior chapters scanned:** ch01 (01, 03), ch04 (02)

---

## Summary Table

| ID | Type | File(s) | Lines Affected | Savings Est. | Description |
|----|------|---------|---------------|-------------|-------------|
| C5-1 | CRUCIAL | 01 vs ch01/01 | 01: 9-13, 118-131 | ~30 lines | MLA architectural intro and kv_b_proj split duplicate ch01 Section 1.1.1 almost verbatim |
| C5-2 | CRUCIAL | 01 vs ch04/02 | 01: 42-54, 137-187 | ~50 lines | Full Q-path shape trace table and fusion group layout tables restated from ch04 4.2.2 pre_sdpa tables |
| C5-3 | CRUCIAL | 03: 173-184 vs 02: 386-397 | Both files | ~15 lines | S-block-to-DRAM-bank table reproduced identically between 02 (Section 5.2.11) and 03 (Section 5.3.4) |
| C5-4 | MINOR | 02: 247-265 vs 03: 248-254 | Both files | ~10 lines | BFP8 tile size / BF16 tile size / 1.88x ratio stated in both 02 and 03 |
| C5-5 | MINOR | 03: 419-428 vs 03: 60-76 | 03 internal | ~8 lines | Online softmax merge formula in Section 5.3.7 (tree reduction) restates the formula from Section 5.3.2 |
| C5-6 | MINOR | 01: 346-353 vs 02: 54-58 | Both files | ~6 lines | RoPE formula and rotate_half definition given in full in 01, then restated in 02 ("same mechanism as QROPE") |
| C5-7 | MINOR | 04: 476-500 vs 01: 62-76, ch01/03: 212-217 | 04, 01, ch01 | ~20 lines | kv_b12 and o_proj fusion group tables repeated across 04 Section 5.4.11, 01 Section 5.1.5, and ch01 Section 3 |
| C5-8 | MINOR | 03: 654-668 vs 04: 9-24 | Both files | ~10 lines | The mathematical chain for post-SDPA (kv_b2 + o_proj + all-reduce) is summarized in 03 Section 5.3.13 (SDPAReduceToAll) and then fully restated in 04 Section 5.4.1 |
| C5-9 | MINOR | Various | Scattered | ~15 lines | Hedging and filler phrases throughout |

**Total estimated savings:** ~164 lines (~7% of chapter)

---

## Load-Bearing Evidence

### C5-1: MLA architectural intro duplicates ch01

**ch01/01 lines 28-48** introduce MLA, the 576-element KV cache, the two-stage Q projection, and the kv_b_proj split with code. **ch05/01 lines 9-13** restate the same motivation, the same dimensional arithmetic ("128 heads x (192+128) = 40,960 elements ... 576 elements ... 71x compression"), and **ch05/01 lines 118-131** reproduce the kv_b_proj split logic with identical reshape/slice operations. Ch05 adds TP=2 shard details but the first ~10 lines of Section 5.1.1 and the "The kv_b_proj Split" subsection overlap substantially with ch01.

**Suggested fix:** Replace Section 5.1.1 with a 2-sentence forward reference to ch01 Section 1.1.1, retaining only the one new sentence about TP=2 consequences. Move the kv_b_proj split to a brief "see ch01 Section 1.1.1 for derivation; TP=2 shard sizes are:" followed by the per-device shape table.

### C5-2: Q-path shape trace and fusion groups overlap ch04

**ch04/02 lines 89-216** contain a complete stage pipeline table for pre_sdpa including a core grid diagram, CB map (45 entries), and semaphore budget. **ch05/01 lines 42-54** provide a nearly identical stage table (same operations, same core grids, same shapes) and **ch05/01 lines 137-187** reproduce the three fusion groups with per-sub-tensor shard shapes, which also appear in **ch01/03 lines 191-217**. The ch05 versions add byte-level shard detail (tiles/shard, dtype) but the high-level grid assignments and shapes are restated.

**Suggested fix:** Replace the ch05/01 shape trace table (Section 5.1.2 table) with a cross-reference to ch04 Table 4.2.2 and annotate only the *new* columns (shard shape, tiles/shard, bytes). For fusion groups (Section 5.1.5), keep only the byte-offset stacking diagrams (those are new) and reference ch01/03 for the sub-tensor roster.

### C5-3: S-block-to-DRAM-bank table duplicated verbatim

**ch05/02 Section 5.2.11** (lines 386-397) contains the `FlashMLAOptimalGridNOC0` S-block table with all 8 S-blocks, their core assignments, and their optimal DRAM banks. **ch05/03 Section 5.3.4** (lines 173-184) reproduces this exact same table with the same data, the same bank order, and the same commentary about NOC locality.

**Suggested fix:** Keep the table in 03 (which is the Flash MLA file where it is architecturally motivated) and replace the copy in 02 with "See Section 5.3.4 for the S-block-to-DRAM-bank mapping that governs physical shard placement."

### C5-4: BFP8 tile arithmetic restated

**ch05/02 lines 247-253** state BFP8 tile = 1088 bytes, BF16 tile = 2048 bytes, ratio ~1.88x. **ch05/03 lines 248-254** restate "If the K buffer used BF16 instead of BFP8, each 32x32 tile would be 2,048 bytes instead of 1,088." Both files derive the savings from the same ratio. The tile sizes also appear in 01 (Section 5.1.3, weight catalog).

**Suggested fix:** Define the BFP8/BF16 tile sizes once (in 02 where the KV cache format is introduced) and use them by name elsewhere. A single sentence in 03 like "Using BFP8 (1088 B/tile vs 2048 B for BF16, see Section 5.2.7)" replaces the redundant derivation.

### C5-5: Online softmax formula restated internally in 03

**ch05/03 lines 60-76** (Section 5.3.2) derive the online softmax update: new max, correction exponents, rescale O and s, accumulate. **ch05/03 lines 419-428** (Section 5.3.7, "Reduction Math: sdpa_tail") restate the identical formula with "This is mathematically identical to the online softmax update but operates on pre-accumulated statistics." The formulas use different variable names but are structurally identical.

**Suggested fix:** In Section 5.3.7, replace the full formula block with a forward reference: "The reduction uses the same online softmax merge from Section 5.3.2, applied to pre-accumulated (O, m, s) triples:" followed by only the 4-line formula without the surrounding explanation paragraph.

### C5-6: RoPE formula repeated

**ch05/01 Section 5.1.10** (lines 346-353) defines the RoPE formula and rotate_half in full. **ch05/02 Section 5.2.5** (lines 54-58) says "Per-core RoPE (same mechanism as QROPE)" and then states `output = x * cos + (x @ trans_mat) * sin` again. The phrase "same mechanism as QROPE" already signals duplication.

**Suggested fix:** In 02, remove the formula line and keep only "Per-core RoPE (identical to QROPE, Section 5.1.10)."

### C5-7: Fusion group tables appear three times

The kv_b12 and o_proj fusion group sub-tensor tables appear in:
- **ch01/03 lines 212-217** (initial overview)
- **ch05/01 Section 5.1.5** (lines 137-187, with byte-offset diagrams)
- **ch05/04 Section 5.4.11** (lines 476-500, with byte-sizes)

The ch05/04 version adds no information beyond what ch05/01 already provides -- same shapes, same core grids, same BFP8 format.

**Suggested fix:** In 04 Section 5.4.11, keep the prose explaining *why* weights participate in fused buffers and the kv_b12 disjoint-core-set insight, but replace the two repeated tables with cross-references to Section 5.1.3 (weight catalog) and Section 5.1.5 (fusion group layout).

### C5-8: Post-SDPA math chain summarized, then restated

**ch05/03 Section 5.3.13** (lines 654-668) summarizes SDPAReduceToAll including the online softmax merge formula *and* mentions scatter to matmul4 input. **ch05/04 Section 5.4.1** (lines 9-24) then fully derives the same mathematical chain (kv_b2 per-head, concatenation, o_proj, all-reduce + residual). Some overlap is natural as a section opener, but the SDPAReduceToAll merge formula in 03 is given in full (3 equations) and then 04 does not re-derive it only because 04 focuses on the CCL variant. The preview in 03 could be shorter.

**Suggested fix:** Trim Section 5.3.13 to state the purpose of SDPAReduceToAll and the scatter destination without deriving the full merge equations (which are the same as Section 5.3.7). A one-line forward reference to Section 5.4 for the post-SDPA pipeline suffices.

### C5-9: Hedging and filler phrases

Scattered instances of verbose or hedging language:

- 01 line 3: "making the dimensional arithmetic explicit at every boundary" -- can be cut; the section's existence makes this obvious.
- 01 line 19: "The Q-path implements the low-rank factorization that is the defining innovation of MLA." -- restates Section 5.1.1; cut "that is the defining innovation of MLA."
- 02 line 15: "The critical insight: the Q-path decompresses the 576-dimensional latent to full 12288 dimensions via W_{q_b_proj} and then recompresses Qnope via W_{kv_b1_proj}. But the KV cache stores only the 576-dimensional compressed form. This asymmetry -- expensive Q computation, cheap KV storage -- is the core of MLA's efficiency." -- This is a restatement of Section 5.1.1 and 5.1.9.
- 03 line 3: "This section is the heart of Chapter 5." -- filler.
- 04 line 163: "but the simplicity of a single rectangular mcast is worth this cost" -- mild hedging that could be tightened to "the rectangular mcast constraint requires this."

---

## CRUCIAL Suggestions

### C5-1: Deduplicate MLA motivation from ch01

**Location:** 01 Section 5.1.1 (lines 9-13) and 01 "The kv_b_proj Split" (lines 118-131)
**Action:** Replace the 13-line architectural motivation with a 2-line cross-reference to ch01 Section 1.1.1. Keep only information unique to ch05 (the TP=2 shard sizes after split). Similarly condense the kv_b_proj split to reference ch01's code listing and add only the per-device shapes.
**Estimated savings:** ~30 lines
**Risk:** Low. Readers of ch05 have already read ch01 (sequential structure).

### C5-2: Deduplicate Q-path stage table and fusion groups from ch04

**Location:** 01 Section 5.1.2 table (lines 42-54) and 01 Section 5.1.5 (lines 137-187)
**Action:** For the stage table, reference ch04 Table 4.2.2 and present only the *additional* columns (shard shape, tiles/shard, byte sizes). For fusion groups, keep the byte-offset stacking diagrams (new content) and replace the sub-tensor roster with a reference to ch01 Section 3 / ch04 Section 4.2.2.
**Estimated savings:** ~50 lines
**Risk:** Low to medium. The ch05 tables do add byte-level detail; a merged approach preserves that while cutting repetition.

### C5-3: Remove duplicate S-block table from 02

**Location:** 02 Section 5.2.11 (lines 383-398) vs 03 Section 5.3.4 (lines 173-184)
**Action:** Delete the table from 02 and replace with a one-line cross-reference to 03 Section 5.3.4.
**Estimated savings:** ~15 lines
**Risk:** None. The table is byte-for-byte identical.

---

## MINOR Suggestions

### C5-4: Consolidate BFP8 tile arithmetic

**Location:** 02 Section 5.2.7 (lines 247-253), 03 Section 5.3.5 (lines 248-254), 01 Section 5.1.3
**Action:** State tile sizes once in 02 where BFP8 is introduced; use shorthand references elsewhere.
**Estimated savings:** ~10 lines

### C5-5: Trim restated online softmax merge in tree reduction

**Location:** 03 Section 5.3.7 (lines 419-428)
**Action:** Replace full derivation with back-reference to Section 5.3.2 plus the 4-line formula only.
**Estimated savings:** ~8 lines

### C5-6: Remove redundant RoPE formula from 02

**Location:** 02 Section 5.2.5 (lines 54-58)
**Action:** Replace formula with "identical to QROPE (Section 5.1.10)."
**Estimated savings:** ~6 lines

### C5-7: Collapse third copy of fusion group tables in 04

**Location:** 04 Section 5.4.11 (lines 476-500)
**Action:** Keep prose rationale, replace tables with cross-references to 01 Sections 5.1.3 and 5.1.5.
**Estimated savings:** ~20 lines

### C5-8: Trim SDPAReduceToAll preview in 03

**Location:** 03 Section 5.3.13 (lines 654-668)
**Action:** Remove the 3-equation merge formula (same as Section 5.3.7); keep scatter description.
**Estimated savings:** ~10 lines

### C5-9: Tighten hedging and filler

**Location:** Scattered (01 lines 3, 19; 02 line 15; 03 line 3; 04 line 163)
**Action:** Remove or shorten identified phrases.
**Estimated savings:** ~15 lines

---

## VERDICT

**Crucial updates: yes**

Three crucial compressions identified: (1) MLA motivation and kv_b_proj split are substantially duplicated from ch01, (2) Q-path stage table and fusion group layouts are restated from ch04, and (3) the S-block-to-DRAM-bank table is copy-pasted between Sections 5.2 and 5.3. Combined with six minor suggestions, total estimated savings are ~164 lines (~7% of chapter). The chapter's technical content is strong and information-dense; the redundancy is cross-chapter and intra-chapter repetition of tables and formulas, not structural bloat.

---

## Pass 2 Verification

**Agent:** C (Compressor, Pass 2)
**Date:** 2026-05-08

### C5-1: MLA architectural intro condensed to cross-reference -- VERIFIED

Section 5.1.1 (file 01, lines 9-11) now reads as a 2-sentence summary plus one TP=2 sentence, with a parenthetical cross-reference "(see [Section 1.1.1](../ch01_.../01_deepseek_v3_architecture_on_tenstorrent.md) for the full derivation)." The target section (ch01/01 line 29, "### 1.1.1 Multi-head Latent Attention (MLA)") exists and contains the full compression derivation. The kv_b_proj split subsection (line 120) likewise uses "see [Section 1.1.1](...) for derivation" before its per-device shape table. No information was lost: TP=2 unique content (64-of-128 heads, per-device shard shapes) is retained. No new duplication introduced.

### C5-2: Q-path stage table and fusion groups cross-referenced to ch04 -- VERIFIED

The stage table in Section 5.1.2 (file 01, line 41) is preceded by "The pipeline stages follow the same flow as [ch04 Section 4.2.2](../ch04_.../02_attention_fused_ops.md); the table below adds per-core shard shapes and tile counts." The target (ch04/02 line 65, "## 4.2.2 `pre_sdpa`") exists and contains the high-level stage pipeline. The ch05 table retains its unique per-core shard/tile detail columns. Section 5.1.5 (file 01, line 131) opens with "see [Section 4.1.5](../ch04_.../01_fusion_principles_and_patterns.md) for the OverlappedTensor mechanism." The target (ch04/01 line 186, "## 4.1.5 `OverlappedTensor` and Weight Packing") exists. The byte-offset stacking diagrams (lines 133-178) are retained as unique content. No information lost; no new duplication introduced.

### C5-3: S-block-to-DRAM-bank table removed from 02, cross-referenced to 03 -- VERIFIED

Section 5.2.11 (file 02, lines 383-385) now contains a 3-line paragraph ending with "See [Section 5.3.4](03_flash_mla_decode.md) for the full S-block-to-DRAM-bank mapping table with core coordinates." The full table remains in file 03 Section 5.3.4 (lines 168-184), including all 8 S-blocks, their core coordinates, and DRAM bank assignments. The optimal bank order `(1, 3, 2, 0, 5, 7, 6, 4)` is still mentioned in both locations (contextually appropriate in 02 for the DRAM layout discussion), but the table itself appears only once. No information lost; no new duplication introduced.

---

**VERDICT: Crucial updates: no**

All three CRUCIAL fixes (C5-1, C5-2, C5-3) are correctly applied. Cross-references point to valid sections in the target files. No information was lost that should have been retained. No new duplication was introduced by the fixes. The chapter is compression-complete for CRUCIAL items. Six MINOR suggestions (C5-4 through C5-9) remain unaddressed but are outside the scope of this pass.
