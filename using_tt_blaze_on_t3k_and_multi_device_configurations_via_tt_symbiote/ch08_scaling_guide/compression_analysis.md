# Compression Analysis: Chapter 8 (Scaling Guide)

## Methodology

Each Chapter 8 file was compared against Chapters 1-7 for content duplication. The threshold for CRUCIAL is >15 lines of substantially duplicated explanation or data; MINOR is 5-15 lines. Code examples showing how to USE APIs documented earlier are not counted as duplication. New practical guidance that BUILDS ON earlier content (e.g., failure-first framing, "For your model" generalization, decision trees) is not duplication.

---

## CRUCIAL Duplications

**None found.**

Chapter 8 consistently cross-references earlier chapters rather than re-explaining their content. Every section that touches on material covered in Ch1-Ch7 does so in one of two acceptable ways:

1. **Brief summary + cross-reference link**: e.g., "Pattern A (Column-Sharded Chain -- Ling/Bailing) ... [Ch7/02, Pattern 2]" (Part 1, Step 3). The description is 2-3 sentences that contextualize the pattern for the scaling decision, not a re-explanation of the sharding mechanics.

2. **Worked-example code showing HOW TO USE**: e.g., the `from_torch()` fused QKV implementation in Part 2 Step 2 mirrors code in Ch3/04 (Gemma4 Attention section), but Part 2 presents it as a step-by-step recipe with inline commentary on what to change for a different model. This is practical application guidance, not reference documentation.

---

## MINOR Duplications

### 1. Scaling Ladder Table (Part 1, lines 13-19) vs Ch1/01

Part 1 includes a 5-row "Scaling Ladder" table mapping hardware rungs to mesh shapes and model sizes. Ch1/01 has a detailed supported-configurations table. However, the Ch8 table adds columns not in Ch1 (CCL requirements, typical model size ranges) and frames them as decision rungs rather than hardware reference. The overlap is the mesh shape column (~5 lines of data). **Acceptable** -- the framing is different enough and the table is short.

### 2. Module Checklist Table (Part 1, lines 185-193) vs Ch3/04 Module Selection Guide

Part 1 Step 4 has an 8-row module checklist table mapping single-device to multi-device modules. Ch3/04 has a nearly identical "Module Selection Guide" table (lines 447-455). The Ch8 version adds a "Notes" column with brief explanations and includes the Decoder Layer and Model Wrapper rows (absent from Ch3/04). The core 6 shared rows are ~8 lines of overlapping data. The Ch8 version explicitly attributes: "Source: Ch3/04, Module Selection Guide." **Minor overlap** -- the attribution and additional rows justify keeping it inline, but a cross-reference alone could suffice.

### 3. Complete Replacement Map Reference Table (Part 2, lines 274-284) vs Ch3/02 Sharding Strategy Selection Summary

Part 2 ends with an 8-row replacement map table. Ch3/02 has a 4-row "Sharding Strategy Selection Summary" table (lines 340-347). The Ch8 table is a superset (adds RMSNorm, Embedding, Small linear rows and a "Pattern" column). The 4 shared linear rows overlap with Ch3/02 (~6 lines). Ch8 attributes: "Source: Ch3/02, Ch3/04." **Minor overlap** -- the superset nature and pattern attribution justify keeping it.

### 4. CCL Cost Model Table (Part 4, lines 42-49) vs Ch3/03

Part 4 Step 1 presents a 5-row CCL cost table for different topologies. Ch3/03 has a "CCL Operation Costs on T3K" section (lines 298-308) that covers the same back-of-envelope calculation, but only for T3K. The Ch8 table is broader (N300, T3K, P150x8, TG horizontal, TG vertical) and provides RS+AG combined latency. Only the T3K row (~2 lines) overlaps directly. **Acceptable** -- Ch8 extends, not duplicates.

### 5. Run-Mode Progression (Part 3, lines 59-92) vs Ch2/03

Part 3 Step 2 presents a 5-step run-mode progression (CPU -> NORMAL_WITH_FALLBACK -> SEL -> NORMAL -> TRACED). Ch2/03 has a "Development Workflow Progression" section (lines 178-187) with the same 5 modes in the same order (~10 lines). However, Ch8 adds failure descriptions and action items per step, and frames each mode as a specific validation gate rather than a general workflow overview. The structural skeleton (mode names + order) is the same. **Minor overlap** -- the value-add (failure modes, specific thresholds) is substantial.

### 6. Hybrid PP+TP Configuration Matrix (Part 4, lines 230-238) vs Ch7/02

Part 4 Step 6 includes a 4-row PP+TP configuration matrix. Ch7/02 has a 5-row "Hybrid PP+TP Configuration Matrix" (lines 178-186). Three rows are identical (1G-4x2, 1G-2x2, 4G-1x2). Ch8 adds a 2G-4x2 row and drops the 1G-1x2 and 4G-4x2 rows. ~5 lines overlap. Ch8 adds a "Max Model (bf16)" column not in Ch7/02. **Minor overlap** -- the added column and different row selection justify keeping it.

### 7. PipelineGraph Code Example (Part 4, lines 200-219) vs Ch5/01

Part 4 Step 5 shows a PipelineGraph construction example (~18 lines of code). Ch5/01 documents PipelineGraph in detail with its own code examples. However, the Ch8 code is a minimal "how to use" recipe for a generic model (not DeepSeek V3), showing the sequential handoff pattern with `make_embed_stage`, `make_decoder_stage`, and `make_lmhead_stage` factories that do not appear in Ch5/01. This is new practical guidance, not duplication of Ch5/01's reference material. **Not duplication.**

---

## Verdict

**Crucial updates: no**

No content blocks exceed the 15-line threshold for mandatory replacement with cross-references. Chapter 8 demonstrates good cross-referencing discipline throughout, with explicit `[ChX/YY]` links accompanying every reference to earlier material.

The minor overlaps identified above (items 2, 3, 5, 6) are each in the 5-10 line range and serve a legitimate purpose: providing self-contained decision context so the reader does not need to interrupt the scaling workflow to consult reference chapters. All overlapping tables include source attributions. No action required.
