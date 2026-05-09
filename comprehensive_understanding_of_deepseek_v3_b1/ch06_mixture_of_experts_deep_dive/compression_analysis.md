# Compression Analysis -- Chapter 6: Mixture-of-Experts Deep Dive

**Agent:** C (Compressor, Pass 1)
**Date:** 2026-05-08
**Files reviewed:**
- `01_moe_gating_and_routing.md` (465 lines)
- `02_routed_expert_computation.md` (702 lines)
- `03_shared_expert_and_combination.md` (565 lines)
- `04_dram_streaming_matmul_deep_dive.md` (677 lines)
- Total: 2409 lines

**Prior chapters scanned:**
- ch01: `03_layer_taxonomy_and_decode_overview.md`
- ch03: `01_compute_micro_ops.md`, `02_data_movement_micro_ops.md`, `03_communication_and_infrastructure_micro_ops.md`
- ch04: `03_moe_fused_ops.md`

---

## Summary Table

| ID | Type | File(s) | Lines Affected | Savings Est. | Description |
|----|------|---------|---------------|-------------|-------------|
| C6-1 | CRUCIAL | 04 vs ch03/02 | 04:66-144 vs ch03/02:384-432 | ~40 lines | Column-major tile reorder algorithm reproduced nearly verbatim from ch03 Section 3.2.5.2 |
| C6-2 | CRUCIAL | 04 vs ch03/02 | 04:148-191 vs ch03/02:402-420 | ~25 lines | Triple-buffering scheme and CB1 sizing reproduced from ch03 Section 3.2.5.3 |
| C6-3 | CRUCIAL | 04 vs ch03/02 | 04:260-305 vs ch03/02:422-434 | ~25 lines | Page size optimization with get_max_page_size_and_num_pages reproduced from ch03 Section 3.2.5.4 |
| C6-4 | CRUCIAL | 04 vs ch03/02 | 04:343-379 vs ch03/02:499-519 | ~20 lines | Bank ID and VC conflict resolution algorithm reproduced from ch03 Section 3.2.5.9 |
| C6-5 | CRUCIAL | 02 vs ch04/03 | 02:81-147 vs ch04/03:266-301 | ~30 lines | Expert weight layout, per-expert size calculation, and _shuffle_dram_tiles explanation repeated from ch04 Section 4.3.2 DRAM Streaming details |
| C6-6 | CRUCIAL | 02 vs ch04/03 | 02:475-501 vs ch04/03:285-300 | ~15 lines | Bank ID and VC conflict resolution code repeated from ch04 Section 4.3.2 |
| C6-7 | CRUCIAL | 01 vs ch03/03 | 01:142-267 vs ch03/03:651-763 | ~50 lines | DeepSeek MoE gate algorithm (hierarchical selection) substantially restates ch03 Section 3.3.8 with expanded golden reference |
| C6-8 | CRUCIAL | 02 vs 01 | 02:60-78 vs 01:416-461 | ~15 lines | Intra-chapter: Stages 1-5 summary in 02 restates the end-to-end diagram from 01 |
| C6-9 | CRUCIAL | 03 vs ch04/03 | 03:371-406 vs ch04/03:79-98 | ~25 lines | Semaphore table in 03 is nearly identical to the MoeSem table in ch04 Section 4.3.1 |
| C6-10 | CRUCIAL | 04 vs ch04/03 | 04:195-228 vs ch04/03:270-283 | ~20 lines | subblock_k parameter table and explanation reproduced from ch04 Section 4.3.2 |
| C6-11 | MINOR | 02 vs 04 | 02:152-179 vs 04:195-228 | ~10 lines | Intra-chapter: gate_proj DRAM streaming matmul parameter table in 02 partially duplicates the performance table in 04 |
| C6-12 | MINOR | 01 vs ch01/03 | 01:9-20 vs ch01/03:6-45 | ~8 lines | Dense vs MoE layer split table restates ch01 Section 3.1 |
| C6-13 | MINOR | 01 vs ch04/03 | 01:269-288 vs ch04/03:116-119 | ~8 lines | CB5 table for the gate micro-op partially restates ch03 Section 3.3.8 CB table |
| C6-14 | MINOR | 02 vs ch04/03 | 02:354-371 vs ch04/03:73-77 | ~8 lines | _MoeRoutedExpertContext field group table partially restates ch04 Section 4.3.1 dataclass description |
| C6-15 | MINOR | 04 vs ch04/03 | 04:384-418 vs ch03/02:483-497 | ~10 lines | Optional fused operations (SiLU, mul, scalar) repeated from ch03 Section 3.2.5.8 |
| C6-16 | MINOR | 02 vs 01 | 02:140-147 vs 01:320-328 | ~5 lines | Multi-device expert indexing paragraph restates nearly identical content from 01 lines 320-328 |
| C6-17 | MINOR | 03 vs ch04/03 | 03:24-42 vs ch04/03:435-466 | ~8 lines | A/B core grid diagram in 03 restates the grid diagram from ch04 Section 4.3.3 |

**Total estimated savings:** ~322 lines (~13% of chapter)

---

## Load-Bearing Evidence

### C6-1: Column-Major Tile Reorder Algorithm (ch06/04 vs ch03/02)

Chapter 6 Section 6.4.3 (file 04, lines 66-144) presents the full column-major tile reordering algorithm with diagrams, the `_shuffle_dram_tiles()` step-by-step description, the permutation index formula, and a worked example. Chapter 3 Section 3.2.5.2 (file ch03/02, lines 384-398) already covers the same reordering with the same diagram layout, the same formula, and the same conceptual explanation.

Ch06/04, lines 83-93:
```
DRAM shard for bank i (after _shuffle_dram_tiles):
  [K=0,N=0] [K=1,N=0] [K=2,N=0] ... [K=Kt-1,N=0]
  [K=0,N=1] [K=1,N=1] [K=2,N=1] ... [K=Kt-1,N=1]
```

Ch03/02, lines 389-397:
```
Column-major reordering per shard:
  tile(0,0), tile(1,0), tile(2,0), ...  // column 0 (all K tiles)
  tile(0,1), tile(1,1), tile(2,1), ...  // column 1
```

Both also include the source index formula: `source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)`.

Additionally, the same reorder is also described in ch06/02 at lines 119-137, making it appear three times across prior chapters + this chapter.

**Suggested fix:** In 04 Section 6.4.3, replace the full algorithmic re-derivation with a cross-reference to ch03 Section 3.2.5.2, retaining only the worked example specific to the MoE dimensions (lines 127-140) and the performance impact note (lines 142-144) as new content.

### C6-2: Triple-Buffering Scheme (ch06/04 vs ch03/02)

Ch06/04, lines 148-191 describes triple-buffering with the same CB sizing formula, the same pipeline timing diagram concept, the same "why not double buffering" rationale, and the same assertion on buffer count. Ch03/02, lines 402-420 covers the identical triple-buffering strategy with the same code snippet and same pipeline timing diagram.

Ch06/04, lines 154-158:
```python
num_in1_buffers = 3
in1_CB_tiles = subblock_k * num_in1_buffers
in1_CB_size = in1_CB_tiles * in1_tile_size
```

Ch03/02, lines 404-407:
```python
num_in1_buffers = 3
in1_CB_tiles = subblock_k * num_in1_buffers
in1_CB_size = in1_CB_tiles * in1_tile_size
```

Verbatim identical code snippet.

**Suggested fix:** Cross-reference ch03 Section 3.2.5.3 for the triple-buffering mechanism. Retain only the MoE-specific concrete values (168 tiles for gate_proj, 96 tiles for down_proj) and the "why not double buffering" discussion if it adds insight beyond ch03.

### C6-3: Page Size Optimization (ch06/04 vs ch03/02)

Ch06/04, lines 260-305 reproduces the entire `get_max_page_size_and_num_pages` function from ch03/02, lines 422-434. The function body is identical. Ch06 adds two worked examples (Wormhole and Blackhole) which are new content, but the function itself is already fully documented.

**Suggested fix:** Replace the function reproduction with a cross-reference. Keep the two worked examples as they are new.

### C6-4: Bank ID and VC Conflict Resolution (ch06/04 vs ch03/02)

Ch06/04, lines 343-379 reproduces the bank ID assignment and VC conflict resolution algorithm from ch03/02, lines 499-519. The code snippets are nearly identical, and the explanatory text covers the same logic (start with bank_id & 0x3, check same-row cores, increment if conflict). This algorithm also appears in ch06/02 at lines 475-501 and in ch04/03 at lines 285-300.

Ch06/04, lines 363-369:
```python
vc = bank_id & 0x3  # Default: low 2 bits of bank_id
for j in range(idx):
    prev_core = all_worker_cores[j]
    if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
        vc = (vc + 1) & 0x3
        break
```

Ch03/02, lines 503-509:
```python
for idx, core in enumerate(all_worker_cores):
    bank_id = idx % num_dram_banks
    vc = bank_id & 0x3
    # VC conflict resolution: avoid NOC contention for cores on same row
    for j in range(idx):
        if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
            vc = (vc + 1) & 0x3
            break
```

The same algorithm appears four times total across the guide.

**Suggested fix:** Define this algorithm once (ch03 is the natural home) and cross-reference from all other locations. In ch06/04 and ch06/02, replace with a one-line reference and state only the fact that all three DRAM matmuls share the same bank/vc assignments.

### C6-5: Expert Weight Layout and Size Calculation (ch06/02 vs ch04/03)

Ch06/02, lines 81-98 presents the per-expert weight table (gate_proj, up_proj, down_proj shapes, sizes, storage) and the per-expert size calculation formula. Ch04/03, lines 266-273 covers the same content in the DRAM Streaming Matmul Details subsection.

Ch06/02, line 94:
```
7168/32 x 2048/32 x 576 = 224 x 64 x 576 = 8,257,536 bytes ~ 7.9 MB
```

Ch04/03 implicitly covers the same sizes via the subblock_k and triple-buffering details. The weight upload pipeline in ch06/04, lines 421-468 also repeats the padding, shard spec, and shuffle description from ch04/03 lines 266-274 and ch03/02.

**Suggested fix:** In 02, keep the per-expert weight table as a quick reference but remove the repeated size derivation formula. Cross-reference ch04 Section 4.3.2 for the upload pipeline details.

### C6-6: Bank ID/VC in Routed Expert (ch06/02 vs ch04/03)

Ch06/02, lines 475-501 repeats the same bank ID and VC conflict resolution code and explanation from ch04/03, lines 285-300. This is the same content as C6-4 but in a different ch06 file.

**Suggested fix:** Consolidate all bank ID/VC discussion into one location (either ch06/04 or a cross-reference to ch03).

### C6-7: DeepSeek MoE Gate Algorithm (ch06/01 vs ch03/03)

Ch06/01, lines 142-267 provides an extensive treatment of the hierarchical selection algorithm. Ch03/03, lines 651-763 already documents this algorithm at the micro-op level, including the same three-level hierarchy (intra-group top-2, inter-group top-4, final top-8), the same normalization formula with the same epsilon and scaling_factor values, and the same CB layout table.

The ch06 version adds: (a) the full golden reference code (lines 242-267), (b) the "why hierarchical" comparison table (lines 235-239), (c) detailed mathematical notation for each phase (lines 159-221). Much of (c) restates the ch03 description with more mathematical notation but the same algorithmic content.

**Suggested fix:** In 01 Section 6.1.5, add a cross-reference to ch03 Section 3.3.8 for the base algorithm. Focus on what is new: the mathematical derivation showing the separation of biased vs. unbiased scores (lines 155-156, which ch03 does not cover), the golden reference code, and the "why hierarchical" comparison. Remove the repeated CB table (lines 272-279) and compile-time args (lines 281-287) which are identical to ch03.

### C6-8: Intra-Chapter Stages 1-5 Summary (ch06/02 vs ch06/01)

Ch06/02, lines 60-78 summarizes stages 1-5 (input mcast, gate matmul, gate gather, DeepSeek MoE gate, index/scale mcast). This is a condensed restatement of the entire content of ch06/01. While a brief "previously covered" summary is reasonable, the 19-line summary largely restates ch06/01's end-to-end diagram (lines 416-461) without adding new information.

**Suggested fix:** Reduce to a 3-4 line cross-reference: "Stages 1-5 (RMSNorm through index/scale mcast) are covered in Section 6.1. At the end of stage 5, every DRAM matmul core has: [bullet list of the three items at lines 74-78]."

### C6-9: MoeSem Semaphore Table (ch06/03 vs ch04/03)

Ch06/03, lines 378-395 reproduces the full 17-semaphore allocation map from ch04/03, lines 79-98. The tables are nearly identical, with the same IDs, names, producer/consumer roles, and reuse safety notes.

Ch06/03, lines 378-395 (17-row table):
```
| 0 | MCAST_SENDER | BRISC (sender) | Various receivers |
| 1 | MCAST_DATA_RECEIVER | NCRISC (receivers) | BRISC (sender) |
...
| 16 | REDUCE_SYNC | Various | ReduceToOne sync |
```

Ch04/03, lines 80-98 (17-row table):
```
| 0 | MCAST_SENDER | Input mcast sender sync |
| 1 | MCAST_DATA_RECEIVER | Input mcast receiver ready |
...
| 16 | REDUCE_SYNC | ReduceToOne synchronization |
```

Same content, slightly different column headers. Ch06/03 adds the "Temporal Reuse Safety" analysis (lines 397-406) which is partially new (ch04 had a one-line mention at line 98).

**Suggested fix:** Replace the full table with a cross-reference to ch04 Section 4.3.1. Retain only the "Temporal Reuse Safety" analysis (lines 397-406) as new content.

### C6-10: subblock_k Parameter Table (ch06/04 vs ch04/03)

Ch06/04, lines 214-221 presents a table with projection, Kt, num_subblocks_k, subblock_k, CB1 size values. Ch04/03, lines 270-272 covers the same subblock configuration ("4 for gate/up, 2 for down") with the same derivation.

**Suggested fix:** Retain the table in ch06/04 as it serves as a useful quick-reference for the deep dive, but add a cross-reference note to ch04.

---

## CRUCIAL Suggestions

1. **C6-1: Column-major reorder duplication**
   - Location: `04_dram_streaming_matmul_deep_dive.md`, lines 66-144
   - Action: Replace the full algorithm derivation with a cross-reference to ch03 Section 3.2.5.2. Retain only the MoE-specific worked example (lines 127-140) and performance impact note (lines 142-144).
   - Estimated savings: ~40 lines
   - Risk: Low. The worked example preserves all MoE-specific context.

2. **C6-2: Triple-buffering scheme duplication**
   - Location: `04_dram_streaming_matmul_deep_dive.md`, lines 148-191
   - Action: Cross-reference ch03 Section 3.2.5.3. Keep MoE-specific concrete values and the pipeline diagram only if it differs from ch03.
   - Estimated savings: ~25 lines
   - Risk: Low.

3. **C6-3: Page size optimization function reproduction**
   - Location: `04_dram_streaming_matmul_deep_dive.md`, lines 260-305
   - Action: Remove the reproduced function body. Cross-reference ch03 Section 3.2.5.4. Keep the two worked examples (Wormhole and Blackhole).
   - Estimated savings: ~25 lines
   - Risk: Low.

4. **C6-4 + C6-6: Bank ID/VC algorithm appears 4 times across the guide**
   - Location: `04_dram_streaming_matmul_deep_dive.md` lines 343-379 and `02_routed_expert_computation.md` lines 475-501
   - Action: Define once in ch03. In both ch06 files, replace with a cross-reference and a single sentence noting all three matmuls share the same assignments.
   - Estimated savings: ~35 lines
   - Risk: Low.

5. **C6-5: Expert weight layout repeated**
   - Location: `02_routed_expert_computation.md`, lines 81-147
   - Action: Keep the per-expert weight table but remove the size derivation formula (already in ch04). Cross-reference ch04 Section 4.3.2 for the upload pipeline and shuffle algorithm.
   - Estimated savings: ~30 lines
   - Risk: Low.

6. **C6-7: DeepSeek MoE gate algorithm substantially restated**
   - Location: `01_moe_gating_and_routing.md`, lines 142-267
   - Action: Cross-reference ch03 Section 3.3.8 for the base algorithm and CB table. Retain: (a) the biased-vs-unbiased score separation insight (new), (b) the golden reference code (new), (c) the "why hierarchical" comparison table (new). Remove the CB table (lines 272-279) and compile-time args (lines 281-287).
   - Estimated savings: ~50 lines
   - Risk: Medium. The mathematical notation is more rigorous than ch03 and some readers may prefer the expanded treatment. However, the core algorithm is the same.

7. **C6-8: Intra-chapter stages 1-5 restatement**
   - Location: `02_routed_expert_computation.md`, lines 60-78
   - Action: Reduce to a 3-4 line cross-reference to Section 6.1, keeping only the "at the end of stage 5" bullet list.
   - Estimated savings: ~15 lines
   - Risk: Low.

8. **C6-9: MoeSem semaphore table reproduced from ch04**
   - Location: `03_shared_expert_and_combination.md`, lines 378-406
   - Action: Replace the 17-row table with a cross-reference to ch04 Section 4.3.1. Retain only the "Temporal Reuse Safety" analysis (lines 397-406) as new content.
   - Estimated savings: ~25 lines
   - Risk: Low.

## MINOR Suggestions

1. **C6-10:** `04_dram_streaming_matmul_deep_dive.md` lines 214-221: subblock_k table partially duplicates ch04 Section 4.3.2. Add a cross-reference note. Savings: ~5 lines.

2. **C6-11:** `02_routed_expert_computation.md` lines 152-179 (gate_proj DRAM streaming parameters) partially duplicates `04_dram_streaming_matmul_deep_dive.md` lines 214-228. Consider a forward-reference to Section 6.4 from 02 to avoid duplication. Savings: ~10 lines.

3. **C6-12:** `01_moe_gating_and_routing.md` lines 9-20 (dense vs MoE layer split) restates ch01 Section 3.1. The table at lines 15-18 is a subset of ch01's table. Keep but add cross-reference. Savings: ~8 lines.

4. **C6-13:** `01_moe_gating_and_routing.md` lines 269-287 (gate micro-op CB table and compile-time args) partially restates ch03 Section 3.3.8 CB layout and compile-time args table. Savings: ~8 lines.

5. **C6-14:** `02_routed_expert_computation.md` lines 354-371 (_MoeRoutedExpertContext field group table) partially restates ch04 Section 4.3.1 dataclass description. Both serve valid purposes (catalog vs deep-dive). Savings: ~8 lines.

6. **C6-15:** `04_dram_streaming_matmul_deep_dive.md` lines 384-418 (optional fused operations) repeats SiLU, mul, and scalar chain description from ch03 Section 3.2.5.8. Savings: ~10 lines.

7. **C6-16:** `02_routed_expert_computation.md` lines 140-147 (multi-device expert indexing) restates nearly identical content from `01_moe_gating_and_routing.md` lines 320-328. Savings: ~5 lines.

8. **C6-17:** `03_shared_expert_and_combination.md` lines 24-42 (A/B core grid diagram) restates the grid diagram from ch04 Section 4.3.3 lines 435-466. The ch06 version is slightly simplified. Keep but add cross-reference. Savings: ~8 lines.

---

## VERDICT

**Crucial updates: yes**

Chapter 6 contains substantial cross-chapter duplication, primarily with ch03 (micro-op library) and ch04 (fused-op library). The most significant items are:

1. The DRAM streaming matmul deep dive (Section 6.4) reproduces approximately 40% of its content from ch03 Section 3.2.5, including verbatim code snippets for column-major reordering, triple-buffering, page size optimization, and bank ID/VC assignment. The ch06 version adds MoE-specific worked examples and a more cohesive narrative, but the algorithmic core is duplicated.

2. The DeepSeek MoE gate algorithm (Section 6.1.5) substantially restates ch03 Section 3.3.8 with expanded mathematical notation. The new content (golden reference code, biased-vs-unbiased score insight, hierarchical comparison table) is valuable and should be preserved, but the CB table and compile-time args are copied.

3. The MoeSem semaphore table (Section 6.3.12) and bank ID/VC algorithm appear in multiple chapters (ch03, ch04, and twice within ch06).

The recommended approach is to replace reproduced algorithmic content with cross-references while retaining all MoE-specific analysis, worked examples, and insights that are genuinely new in Chapter 6. Estimated total savings of ~322 lines (~13% of the chapter) with no loss of information.

---

## Pass 2 Verification

**Agent:** C (Compressor, Pass 2)
**Date:** 2026-05-08
**Purpose:** Verify that all 8 CRUCIAL fixes (C6-1 through C6-9, with C6-4 and C6-6 combined) were correctly applied by Agent A.

### C6-1: Column-Major Tile Reorder (04 Section 6.4.3)

**VERIFIED.** Section 6.4.3 (line 67) now opens with a one-sentence description of `_shuffle_dram_tiles()` followed by a cross-reference: "See [Chapter 3 Section 3.2.5.2](../ch03_the_micro_op_library/02_data_movement_micro_ops.md) for the full algorithm derivation and permutation formula." The full algorithm derivation and permutation index code have been removed. The MoE-specific worked example (lines 72--84, concrete gate_proj dimensions) and the performance impact note (lines 86--88, 90% vs 30--40% bandwidth utilization) are retained as new content. The cross-reference target (ch03/02, line 384, heading "### 3.2.5.2 Column-Major Tile Reorder") exists and the relative path resolves correctly.

### C6-2: Triple-Buffering Scheme (04 Section 6.4.4)

**VERIFIED.** Section 6.4.4 (line 92) now consists of a two-sentence summary with a cross-reference: "See [Chapter 3 Section 3.2.5.3](../ch03_the_micro_op_library/02_data_movement_micro_ops.md) for the buffering mechanism and pipeline timing." The verbatim `num_in1_buffers = 3` / `in1_CB_tiles = subblock_k * num_in1_buffers` code block has been removed. The MoE-specific concrete CB 1 sizes are retained (lines 96--98: 168 tiles for gate_proj/up_proj, 96 tiles for down_proj). The cross-reference target (ch03/02, line 400, heading "### 3.2.5.3 Triple Buffering Strategy") exists.

### C6-3: Page Size Function (04 Section 6.4.7)

**VERIFIED.** Section 6.4.7 (line 169) now references the function by name and location (`dram_streaming_matmul/op.py`, lines 45--76) with a cross-reference: "See [Chapter 3 Section 3.2.5.4](../ch03_the_micro_op_library/02_data_movement_micro_ops.md) for the full function and algorithm." The reproduced function body has been removed. Both worked examples are retained: Wormhole (lines 173--181) and Blackhole (lines 183--191). The cross-reference target (ch03/02, line 421, heading "### 3.2.5.4 NOC Burst Optimization: `get_max_page_size_and_num_pages()`") exists.

### C6-4 + C6-6: Bank ID / Virtual Channel Conflict Resolution (04 Section 6.4.9 and 02 Section 6.2.12)

**VERIFIED.** Two locations were fixed:

1. **04 Section 6.4.9** (line 229): The full code block showing `vc = bank_id & 0x3` and the conflict resolution loop has been replaced with a narrative summary and cross-reference: "The algorithm ... is documented in [Chapter 3 Section 3.2.5.9](../ch03_the_micro_op_library/02_data_movement_micro_ops.md)." The MoE-specific context (all three projections share the same bank_id/vc values, passed as `dram_mm_bank_id` and `dram_mm_vc`) is retained (lines 232--233). Cross-reference target (ch03/02, line 498, heading "### 3.2.5.9 Per-Core Bank ID and VC Assignment") exists.

2. **02 Section 6.2.12** (line 463): Reduced to a 3-line section that cross-references Section 6.4.9: "computes these using the optimal-bank-to-worker assignment and VC conflict resolution algorithm described in [Section 6.4.9](04_dram_streaming_matmul_deep_dive.md)." No code is reproduced. The intra-chapter cross-reference target (04, line 229, heading "## 6.4.9 Bank ID and Virtual Channel Conflict Resolution") exists.

### C6-5: Expert Weight Layout (02 Section 6.2.3)

**VERIFIED WITH NOTE.** The per-expert weight table (lines 75--79) is retained as a quick-reference, which matches the suggested fix. The size derivation formula (line 83: `7168/32 x 2048/32 x 576 = 8,257,536 bytes`) remains in file 02. This formula also appears at line 17 of file 04 (Section 6.4.1). However, both occurrences serve load-bearing contextual roles -- 04 uses it to motivate DRAM streaming, and 02 uses it to explain the weight footprint. The one-line formula is not the kind of substantial algorithmic duplication that the CRUCIAL classification targets; the main concern in C6-5 was the repeated upload pipeline and shuffle explanation, which has been addressed: file 02 Section 6.2.3 now contains a forward-reference to Section 6.4 at line 125 ("see Section 6.4 for details") for the DRAM streaming mechanics. The column-major tile reorder subsection at lines 106--125 of file 02 remains, but it is contextually necessary within the weight layout discussion and is more concise than the original -- it does not reproduce the full algorithm derivation (which is in ch03) but rather explains the motivation within the weight preparation context. Accepted as correctly applied.

### C6-7: Gate Algorithm CB Table and Compile-Time Args (01 Section 6.1.5)

**VERIFIED.** The "Hardware Implementation" subsection at line 269 now reads: "The micro-op's 5-CB layout and compile-time arguments (epsilon, scaling factor, sigmoid flag) are documented in [Chapter 3 Section 3.3.8](../ch03_the_micro_op_library/03_communication_and_infrastructure_micro_ops.md)." The previously reproduced CB table and compile-time argument list have been replaced with this cross-reference. The retained content includes: (a) the full golden reference code (lines 242--267), (b) the "Why Hierarchical Selection?" comparison table (lines 227--238), (c) the 5-phase mathematical derivation with biased-vs-unbiased score separation insight (lines 159--221) -- all identified as genuinely new content in Pass 1. The cross-reference target (ch03/03, line 651, heading "## 3.3.8 deepseek_moe_gate -- Deep Dive") exists. The output tile format note ($1 \times 16$, line 271) is retained as new MoE-specific context.

### C6-8: Intra-Chapter Stages 1--5 Summary (02 Section 6.2.2)

**VERIFIED.** Section 6.2.2 (lines 60--65) has been reduced from the original 19-line summary to a concise 3-line cross-reference: "Stages 1--5 (RMSNorm through index/scale mcast) are covered in [Section 6.1](01_moe_gating_and_routing.md). At the end of stage 5, every DRAM matmul core has:" followed by a 3-item bullet list specifying exactly what each core holds (input activation, expert index, expert scale). This matches the suggested fix precisely.

### C6-9: Semaphore Table (03 Section 6.3.12)

**VERIFIED.** Section 6.3.12 (lines 371--401) now opens with the section title and introductory context, then includes an explicit cross-reference (line 377): "The full 17-semaphore allocation map (IDs 0--16) is documented in [Chapter 4 Section 4.3.1](../ch04_the_fused_op_library/03_moe_fused_ops.md)." The 17-row semaphore table has been removed. The "Temporal Reuse Safety" analysis (lines 379--389) is retained as new content, explaining why semaphore 0 can be shared across 6 mcast operations. The semaphore creation code snippet (lines 395--399) is also retained. The cross-reference target (ch04/03, line 9, heading "## 4.3.1 `moe` -- The Top-Level MoE Orchestrator", with the `MoeSem` table at lines 79--98) exists.

---

### Cross-Reference Integrity Check

All 7 cross-reference links introduced by the fixes were verified:

| Source | Link Target | Target Exists |
|--------|-------------|---------------|
| 04 Section 6.4.3 | ch03/02 Section 3.2.5.2 (line 384) | Yes |
| 04 Section 6.4.4 | ch03/02 Section 3.2.5.3 (line 400) | Yes |
| 04 Section 6.4.7 | ch03/02 Section 3.2.5.4 (line 421) | Yes |
| 04 Section 6.4.9 | ch03/02 Section 3.2.5.9 (line 498) | Yes |
| 02 Section 6.2.12 | 04 Section 6.4.9 (line 229) | Yes |
| 01 Section 6.1.5 | ch03/03 Section 3.3.8 (line 651) | Yes |
| 03 Section 6.3.12 | ch04/03 Section 4.3.1 (line 9) | Yes |

All relative markdown link paths (`../ch03_the_micro_op_library/...`, `../ch04_the_fused_op_library/...`, `04_dram_streaming_matmul_deep_dive.md`, `01_moe_gating_and_routing.md`) resolve to existing files.

### Residual Duplication Check

After all fixes, two instances of minor residual duplication remain:

1. The per-expert size formula (`7168/32 x 2048/32 x 576 = 8,257,536`) appears in both 02 (line 83) and 04 (line 17). Both are load-bearing single-line contextual references, not algorithmic reproductions. Acceptable.

2. The column-major tile reorder motivation in 02 Section 6.2.3 (lines 106--125) overlaps with the cross-referenced content in 04 Section 6.4.3, but serves a distinct narrative purpose (weight preparation context vs. DRAM streaming mechanics) and is more concise than the original. Acceptable.

No CRUCIAL-level duplication remains.

---

**VERDICT: Crucial updates: no**
