# Compression Analysis -- Chapter 6: Model Implementations

## Summary

Chapter 6 consists of three parts totaling ~590 lines. Part 1 (Symbiote TP models, 278 lines) traces forward passes through four models. Part 2 (Blaze DeepSeek V3, 223 lines) traces the pipeline-parallel forward pass. Part 3 (Patterns and Antipatterns, 298 lines) distills cross-cutting patterns and failure modes. The chapter is well-structured and mostly additive, but contains significant overlap with earlier chapters in the pattern/antipattern catalog and some within-chapter redundancy in the model token traces.

---

## CRUCIAL Suggestions (>15 lines duplication)

### C1. Distributed RMSNorm 3-Phase Pattern -- Explained Three Times

**Ch3/03_ccl_operations.md** (lines 279-296) documents the `tt_distributed_rmsnorm()` 3-phase pattern (pre_all_gather, all_gather, post_all_gather) with code. **Ch3/04_distributed_modules.md** (lines 9-68) re-explains the same pattern with the full `TTNNDistributedRMSNorm.forward()` code and weight distribution. **Ch6/03 Pattern 1** (lines 9-23) explains the same 3-phase pattern a third time with a pseudocode block that is a subset of what Ch3 already provides.

Combined redundancy: ~40 lines across the three locations. Ch6 Pattern 1 adds no new information beyond what Ch3/03 and Ch3/04 already cover.

**Recommendation:** Replace Ch6 Pattern 1 with a 2-3 line summary referencing Ch3/03 Section "Distributed RMSNorm CCL Pattern" and Ch3/04 Section "TTNNDistributedRMSNorm". Keep only the "Used by" line listing which models use it.

### C2. Column-Sharded Linear with Reduce-Scatter -- Re-explained in Ch6

**Ch3/02_weight_sharding_strategies.md** (lines 88-153) provides the complete Row-Parallel (`TTNNLinearIColShardedWRowSharded`) pattern with data flow diagrams, forward pass code, and the reduce_scatter call. **Ch3/02** (lines 156-199) also covers the All-Reduce variant. **Ch6/03 Pattern 2** (lines 28-39) re-explains the same pattern, including the `reduce_scatter + all_gather` decomposition and `TTNNLinearIColShardedWAllReduced`.

Combined redundancy: ~20 lines. Ch6 Pattern 2 is a strict subset of Ch3/02.

**Recommendation:** Replace Ch6 Pattern 2 with a cross-reference to Ch3/02 "Row-Parallel" and "All-Reduce Variant" sections. Retain only the "Used by" attribution and the note about chaining sharded ops without intermediate all_gather.

### C3. Multi-Pass Module Replacement -- Covered in Ch2 and Ch6

**Ch2/01_module_replacement_engine.md** (lines 58-96) documents the recursive DFS walk and warns about class matching. **Ch6/01** (lines 9-22) re-explains the 6-step skeleton. **Ch6/03 Pattern 3** (lines 44-62) explains multi-pass ordering with the same pass-order table. The warning about pass order in Ch6/03 line 62 is a restatement of the Ch6/01 line 22 warning and Ch2/01's existing recursion semantics.

Combined redundancy: ~25 lines. The 6-step skeleton in Ch6/01 and the pass-order guidance in Ch6/03 partially duplicate each other and both overlap with Ch2.

**Recommendation:** Keep the 6-step skeleton in Ch6/01 (it adds model-specific context). Remove the mechanistic explanation from Ch6/03 Pattern 3 and replace with a brief reference to Ch6/01 Section 6.1 and Ch2/01. Retain only the Gemma4 vision-exclusion example (lines 55-59) as a novel contribution.

### C4. Pipeline Topology Tables Duplicated Between Ch5/05 and Ch6/02

**Ch5/05_hardware_configurations.md** (lines 126-163) provides the complete stage layout for 4, 16, and 64 processes with factory function details. **Ch6/02 Section 6.7** (lines 9-19) repeats the same 3-row topology table with identical information (stage layout, fabric config) and re-describes `create_pipeline_configuration_from_num_procs`. The SP4 64-stage mapping in Ch6/02 (line 19) replicates the SP4 layout from Ch5/05 (lines 149-158).

Combined redundancy: ~20 lines.

**Recommendation:** Replace Ch6/02 Section 6.7 table with a cross-reference to Ch5/05. Keep only novel Ch6 content: the stage factory closure code block (lines 21-27) and the layer_id_override warning.

### C5. 4-Phase Pipeline Lifecycle Duplicated Between Ch5/03 and Ch6/02

**Ch5/03_stage_kinds_and_execution.md** (lines 188-234) documents the 4-phase lifecycle (`configure_block`, `setup`, `start_pipeline`, `start_compute`) with pseudocode and persistent mode explanation. **Ch6/02 Section 6.9** (lines 148-162) repeats the same 4-phase table with nearly identical descriptions and the same method names.

Combined redundancy: ~18 lines.

**Recommendation:** Replace Ch6/02 Section 6.9 with a cross-reference to Ch5/03 "Pipeline Orchestrator: 4-Phase Lifecycle". Keep only the Ch6-specific content about `_build_decoder_program_context()` arguments.

### C6. PipelineGraph build_topology Duplicated Between Ch5/01 and Ch6/02

**Ch5/01_pipeline_graph_and_layout.md** (lines 148-162) documents `build_topology()` with the 7-step process including C++ resolver, submesh creation, and Kahn's algorithm. **Ch6/02 Section 6.10** (lines 165-177) re-explains the same auto-discovery process including `resolve_graph_layout()`, submesh creation, and topological sort.

Combined redundancy: ~16 lines.

**Recommendation:** Replace Ch6/02 Section 6.10 with a 2-3 line reference to Ch5/01 "Build Path 2: Topology Auto-Discovery". Keep only the experimental API warning (Ch6/02 line 178) if it is not already in Ch5/01.

---

## MINOR Suggestions

### M1. Trace-Compatible CCL Decomposition Re-explained

Ch3/03 (lines 209-221) explains why `all_reduce` must be decomposed into `reduce_scatter + all_gather` for trace safety. Ch6/03 Antipattern 3 (lines 185-193) re-explains the same issue. The warning about Ring vs Linear topology appears in Ch3/03 (line 277), Ch3/02 (line 154), and Ch6/03 Pattern 1 (line 23). This is a ~10-line overlap spread across multiple locations.

### M2. `_bypass_tensor_wrapping` Explained Three Times

Ch2/02 (lines 336-421) provides the full bypass mechanism with code. Ch3/01 (line 197) references it. Ch6/03 Pattern 5 (lines 83-96) re-explains it with a code snippet. ~12 lines of overlap, just under the threshold.

### M3. 3-Pass Centering TopK Explained Twice Within Ch6

Ch6/01 Section 6.5 (lines 228-231) describes the 3-pass centering topk for Qwen3.5 with a pseudocode block. Ch6/03 Pattern 7 (lines 114-128) re-explains the same technique with a nearly identical pseudocode block. Both describe the same `qwen_moe.py` lines 106-141. ~14 lines of within-chapter duplication.

### M4. `model.device` Antipattern Stated Twice

Ch6/01 Section 6.3 (line 137) mentions the `model.device` property hack. Ch6/03 Antipattern 1 (lines 159-169) explains the same issue in detail. Minor within-chapter overlap (~5 lines of unique content in Ch6/01).

### M5. On-Device Residual Add Pattern Repetitive Across Model Traces

The on-device residual add (`ttnn.add(residual, attn_out)`) is shown in Ch6/01 for Gemma4 (lines 115, 122), Ling (lines 166-167, 180), and then explained again in Ch6/03 Pattern 4 (lines 67-79). The per-model traces legitimately show where residuals appear, but the mechanistic explanation in Pattern 4 could reference Ch6/01 instead of re-explaining.

### M6. MoE Expert Dispatch Pattern Repeated Across Ling and Qwen3.5

Ch6/01 Ling (lines 168-178) and Qwen3.5 (lines 226-241) both trace `all_to_all_dispatch` -> `sparse_matmul` -> `all_to_all_combine`. The shared expert path is similar. Ch3/04 (lines 294-373) covers `TTNNQwen3MoE` with the same flow. ~15 lines of structural repetition across the three locations.

### M7. Persistent Mode Explained Twice

Ch5/03 (lines 221-234) explains persistent mode in detail. Ch6/02 Section 6.12 (lines 198-203) re-explains the same concept. ~8 lines of overlap.

### M8. WeightProvider Modes Table Adds Little Over Ch5 Content

Ch6/02 Section 6.11 (lines 182-195) documents `CacheWeightProvider`, `StateDictWeightProvider`, and `SyntheticWeightProvider`. This content is new to Ch6 but the weight transform details partially overlap with Ch3/02's weight preprocessing discussion.

---

## Load-Bearing Evidence

Not applicable (verdict is not "no").

---

## VERDICT

**Compress.** Six CRUCIAL-level duplications totaling approximately 140 lines can be replaced with cross-references. The per-model token traces in Part 1 are the chapter's core contribution and must be preserved. Part 2's Blaze DeepSeek trace is similarly valuable but its framing sections (6.7, 6.9, 6.10) substantially duplicate Ch5. Part 3's pattern catalog is the most bloated section -- Patterns 1, 2, 3, and 5 are re-explanations of material already covered in Chapters 2 and 3. Estimated compressible content: ~140-160 lines (~24% of total), primarily from Part 3 and the pipeline framing sections in Part 2.
