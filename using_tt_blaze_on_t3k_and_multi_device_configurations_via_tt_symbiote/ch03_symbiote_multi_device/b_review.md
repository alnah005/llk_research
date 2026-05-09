# Agent B Review: Chapter 3 — Pass 1

1. **File:** `02_weight_sharding_strategies.md`, lines 56-69.
   **Error:** The column-parallel data flow diagram shows weight shards as `W[0:H/8,:]`, `W[H/8:2H/8,:]`, etc., which depicts slicing along the first dimension (rows). Column-parallel sharding (`dim=-1`) slices along the last dimension (columns), so each device holds a slice of the output features. The correct notation is `W[:,0:H/8]`, `W[:,H/8:2H/8]`, etc. A reader implementing tensor parallelism from this diagram would shard the wrong weight dimension.
   **Fix:** Change the weight shard labels from `W[0:H/8,:]` to `W[:,0:H/8]`, from `W[H/8:2H/8,:]` to `W[:,H/8:2H/8]`, and from `W[7H/8:H,:]` to `W[:,7H/8:H]`.

2. **File:** `01_distributed_config_and_device_init.md`, lines 276-283.
   **Error:** The `get_default_distributed_tensor_config` code snippet shows `state = DeviceInit.DEVICE_TO_STATE_DICT.get(mesh_device) or next(iter(DeviceInit.DEVICE_TO_STATE_DICT.values()))`, which suggests the function silently falls back to any available device config if the requested mesh_device is not found. The actual source code (run_config.py:452-467) performs an `assert` that raises an error when `mesh_device` is provided but not in the dict, and only falls through to `next(iter(...))` when `mesh_device is None`. A reader relying on the chapter's version would believe the function tolerates missing device registrations and silently returns another device's config, when in reality it crashes with an assertion error. This difference is material for anyone writing custom DeviceInit subclasses or debugging initialization failures.
   **Fix:** Replace the simplified code with the actual two-branch logic: assert when `mesh_device is not None` and it is missing from the dict; fall back to `next(iter(...))` only when `mesh_device is None`.

3. **File:** `02_weight_sharding_strategies.md`, lines 268-310.
   **Error:** The general activation flow diagram (lines 268-304) shows the O-projection as "Row-Parallel / IColShardedWRow" with `reduce_scatter`, producing col-sharded output. The Gemma4-specific text immediately below (lines 306-310) contradicts this by correctly stating that Gemma4's O-projection uses `TTNNLinearIReplicatedWColSharded` (column-parallel, no CCL). The diagram is labeled as a "standard transformer layer" but does not indicate which model it represents, so a reader attempting to implement Gemma4 distributed attention from the diagram would select the wrong linear layer class for the O-projection (row-parallel with reduce_scatter instead of column-parallel without CCL).
   **Fix:** Add a label to the general diagram specifying it depicts the Ling/Bailing pattern (not Gemma4). The Gemma4 paragraph already provides the correct classes, but the diagram needs an explicit scope qualifier to prevent misapplication.

4. **File:** `03_ccl_operations.md`, lines 186-215.
   **Error:** The chapter states that `tt_all_reduce` for 1D meshes (path including T3K) checks `if 1 in list(mesh_device.shape)` and then performs `reduce_scatter_minimal_async`. The actual source code (ccl.py:176-178) has an earlier guard: `if mesh_shape == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)): return input_tensor`. For T3K with shape `(1,8)` and the default `cluster_axis=0`, `1 in [1,8]` is True, so the `reduce_scatter` path does execute as described. However, with `cluster_axis=1`, the early guard fires and returns the input unchanged -- the function is a no-op. The chapter omits this early-return guard and would lead a reader to believe `tt_all_reduce` always performs a `reduce_scatter` on T3K regardless of `cluster_axis`. Additionally, the chapter's code snippet on line 193 omits the `cluster_axis` parameter from `reduce_scatter_minimal_async`, but the actual source on 1D meshes also omits it (correct for the 1D path). The primary error is the missing early-return guard.
   **Fix:** Add the early-return guard (`if mesh_shape == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)): return input_tensor`) before the 1D mesh path description. Note that this means `tt_all_reduce` is a no-op on T3K when called with `cluster_axis=1`.

---

# Agent B Review: Chapter 3 — Pass 2

**Pass 1 issues verification:** All 4 issues from Pass 1 are confirmed fixed.

1. Column-parallel weight shard labels (File 02, lines 56-69): Now correctly use `W[:,0:H/8]` column-slicing notation. FIXED.
2. `get_default_distributed_tensor_config` (File 01, lines 276-287): Now shows the `assert` branch for known `mesh_device` and `next(iter(...))` only for `None`. FIXED.
3. Activation flow diagram labeling (File 02, lines 268-269): Now explicitly labeled as "Ling/Bailing activation pattern" with Gemma4 difference noted. FIXED.
4. `tt_all_reduce` early-return guard (File 03, lines 189-196): Now includes the early-return guard and explains the no-op behavior on T3K with `cluster_axis=1`. FIXED.

**New issues found:**

1. **File:** `02_weight_sharding_strategies.md`, lines 269-310.
   **Error:** The Ling/Bailing activation flow diagram (lines 271-304) and the accompanying text at line 310 both claim Q/K/V projections use `TTNNLinearIReplicatedWColSharded` (column-parallel, replicated input, no CCL). The actual Ling test harness (`test_ling_mini_2_0.py`, line 94) maps ALL `nn.Linear` layers to `TTNNLinearIColShardedWRowSharded` via a blanket replacement: `{nn.Linear: TTNNLinearIColShardedWRowSharded}`. This means Q/K/V projections in Ling/Bailing are row-parallel (col-sharded input, reduce_scatter), not column-parallel. Additionally, the diagram shows MLP gate/up as column-parallel with replicated input and MLP down as the all-reduce variant (`reduce_scatter + all_gather`), but the source confirms both use plain `TTNNLinearIColShardedWRowSharded` (reduce_scatter only, col-sharded output). A reader implementing the Ling/Bailing distributed model from this diagram would select the wrong linear class for Q/K/V projections, MLP gate/up, and MLP down.
   **Fix:** Replace the Ling/Bailing diagram and text to show all linear layers as `TTNNLinearIColShardedWRowSharded`. The correct flow is: col-sharded input from distributed RMSNorm feeds into row-parallel Q/K/V (reduce_scatter), col-sharded Q/K/V; row-parallel O-projection (reduce_scatter); col-sharded residual feeds row-parallel MLP gate/up (reduce_scatter); row-parallel MLP down (reduce_scatter). No layer uses column-parallel or all-reduce variant in Ling/Bailing.

---

# Agent B Review: Chapter 3 — Pass 3

**Pass 2 fix verification:** The Ling/Bailing activation flow diagram in `02_weight_sharding_strategies.md` (lines 267-322) was correctly rewritten. Verified against `test_ling_mini_2_0.py:94` which maps `nn.Linear: TTNNLinearIColShardedWRowSharded`.

Specific checks:
- Lines 272-273: Text now correctly states "Ling/Bailing uses a blanket replacement mapping all `nn.Linear` layers to `TTNNLinearIColShardedWRowSharded` (row-parallel)." Matches source.
- Lines 275-322: Diagram now shows all projections (QKV, O, MLP gate/up, MLP down) as "Row-Parallel / IColShardedWRow" with `reduce_scatter`, all outputs as "COL-SHARDED". Matches source.
- Lines 273: Text correctly notes `TTNNDistributedRMSNorm` produces col-sharded output. Verified against `normalization.py:184-230` (three-phase pattern preserves input sharding).

**Full chapter re-review:** Cross-referenced all four chapter files against source code. No new factual errors found. Key verifications:
- File 01: `DistributedConfig`, `DistributedTensorConfig`, `DeviceInit`, `get_default_distributed_tensor_config` all match `run_config.py`.
- File 02: Linear module hierarchy, weight sharding dims, forward pass CCL operations, decorator stacking, Gemma4 activation pattern all match `linear.py` and `gemma4_attention.py`/`gemma4_mlp.py`.
- File 03: `TT_CCL` semaphore structure, `get_num_links` link dictionary, `tt_all_reduce` early-return guard and paths, Ring vs Linear topology descriptions all match `ccl.py`.
- File 04: `TTNNDistributedRMSNorm` three-phase forward and weight distribution, `TTNNEmbedding` sharding, Gemma4 fused QKV and O-projection classes, Gemma4 MLP fusion, `TTNNQwen3MoE` expert sharding and router, KV cache patterns all match respective source files.

No feedback — chapter approved.

---

# Combined B+C Review: Chapter 3 — Pass 4/2

## Agent B (Factual Accuracy) -- Pass 4

**Scope:** Verify that the 6 compression edits (C1, C2, C4, C5, C6) did not introduce factual errors. Cross-reference all remaining claims and cross-links against source code.

### C1: get_num_links section in ch03/03 (lines 84-86)
- **Compressed text:** "For T3K, this is 1 link per hop on both axes -- a significant bandwidth constraint that directly affects CCL throughput."
- **Source verification:** `ccl.py` line 39: `"T3K": (1, 1)`. Confirmed correct.
- **Cross-ref:** Points to `[Chapter 1, File 1](../ch01_device_topologies/01_physical_topologies.md)`. File exists; the full `link_dict` and query semantics are present at lines 211-235. Verified correct.
- **Verdict:** No factual error introduced.

### C2: mesh_mapper/mesh_composer summary in ch03/01 (line 104)
- **Compressed text:** "The mapper distributes host tensors to devices and the composer reassembles them. For T3K `(1, 8)` with `ShardTensor2dMesh(dims=(0, -1))`, a `[B, S, H]` tensor becomes `[B, S, H/8]` per device; `ConcatMesh2dToTensor(dims=(0, -1))` reverses this."
- **Source verification:** `run_config.py` lines 74-76 confirm `ShardTensor2dMesh(self.mesh_device, self.mesh_device.shape, (0, -1))` and `ConcatMesh2dToTensor(self.mesh_device, self.mesh_device.shape, (0, -1))`. For T3K `(1, 8)`, dim 0 has extent 1 (no batch split) and dim -1 is split 8 ways, so `[B, S, H]` becomes `[B, S, H/8]`. Confirmed correct.
- **Verdict:** No factual error introduced.

### C4: Trace-compat warnings consolidated in ch03/02 (lines 153-154, 187)
- **Consolidated note (line 153-154):** States that both `reduce_scatter` and `all_gather` use `topology=ttnn.Topology.Ring` and references ch03/03 for the full explanation.
- **Short reference (line 187):** "The `reduce_scatter` + `all_gather` decomposition is the trace-compatible form of all-reduce (see note in Row-Parallel section above)."
- **Source verification:** `linear.py` lines 143-150 (row-parallel) use `topology=ttnn.Topology.Ring`; lines 183-198 (all-reduce variant) also use `topology=ttnn.Topology.Ring`. The cross-ref to ch03/03's "Trace-Compatible CCL Decomposition" section (lines 209-221) is accurate.
- **Verdict:** No factual error introduced. The consolidated note correctly characterizes the Ring topology requirement, and the cross-reference target exists with the correct content.

### C5: CCL per-block breakdown in ch03/03 (lines 310-311)
- **Compressed text:** "For a per-module breakdown of CCL operations in a single transformer block, see [File 4, CCL Operations Per Transformer Block](./04_distributed_modules.md). The total is **6 CCL operations per Gemma4 block** (2x decomposed all-reduce + 2x all_gather for RMSNorm)."
- **Cross-ref verification:** ch03/04 lines 458-471 contain the section "### CCL Operations Per Transformer Block" with the detailed per-module table. Total is 6: 2x all_gather (RMSNorm stats), 2x reduce_scatter + 2x all_gather (fused projections). Matches summary.
- **Factual check on "2x decomposed all-reduce + 2x all_gather for RMSNorm":** Each decomposed all-reduce = 1 reduce_scatter + 1 all_gather. So 2x = 2 RS + 2 AG. Plus 2x all_gather for RMSNorm = 2 AG. Total = 2 RS + 4 AG = 6. Correct.
- **Verdict:** No factual error introduced.

### C6: DeviceInit section in ch03/01 (lines 169-173)
- **Compressed text:** "DeviceInit is a singleton that maps each `MeshDevice` to its `DistributedConfig`, ensuring the config (and its `TT_CCL` semaphores) is created exactly once per device. See [Chapter 2, File 1](../ch02_symbiote_core/01_module_replacement_engine.md) for the base `DeviceInit` class, its `init_state()` caching mechanism, and the `DEVICE_TO_STATE_DICT` pattern."
- **Cross-ref verification:** ch02/01 lines 411-431 present `DeviceInit` with `DEVICE_TO_STATE_DICT`, `init_state()` caching, and `init_state_impl()`. All named elements exist at the referenced location. The file exists.
- **Factual check:** "created exactly once per device" matches the caching logic in `init_state()` (ch02/01 lines 421-424: `if device not in cls.DEVICE_TO_STATE_DICT: ... cls.DEVICE_TO_STATE_DICT[device] = res`). Correct.
- **Verdict:** No factual error introduced.

### Full Chapter Cross-Check (post-compression)
- All internal chapter cross-references (File 1 -> File 2, File 2 -> File 3, File 3 -> File 4, etc.) resolve to valid section headings.
- All external chapter cross-references (ch01/01, ch02/01, ch02/02) point to existing files with the referenced content.
- No factual claims were altered by the compressions; only detail level changed, with authoritative content preserved at the referenced locations.

**No feedback -- chapter approved.**

---

## Agent C (Compression Verification) -- Pass 2

### Compression Application Verification

| ID | Description | Status | Notes |
|----|-------------|--------|-------|
| C1 | get_num_links in ch03/03 replaced with 3-line summary + cross-ref to ch01/01 | APPLIED | Lines 84-86. Full link_dict removed; key T3K fact retained. Cross-ref to ch01/01 is accurate. |
| C2 | mesh_mapper/mesh_composer ASCII diagram in ch03/01 replaced with 2-line summary | APPLIED | Line 104. The 16-line ASCII diagram (physical layout on T3K) removed; replaced with concise text stating mapper/composer semantics and the T3K example. |
| C4 | Two trace-compat warnings in ch03/02 consolidated into single note + short reference | APPLIED | Lines 153-154 (consolidated note after row-parallel section), line 187 (short back-reference in all-reduce section). Two separate warning blocks replaced with one note + one sentence. |
| C5 | CCL per-block breakdown in ch03/03 replaced with cross-ref to ch03/04 | APPLIED | Lines 310-311. Per-block detail removed; cross-ref to ch03/04's detailed table added with total count preserved. |
| C6 | DeviceInit section in ch03/01 trimmed to brief recap + cross-ref to ch02/01 | APPLIED | Lines 169-173. Full code block and detailed explanation removed; brief 3-line recap with cross-ref to ch02/01. Customization detail retained at lines 173. |

Note: C3 was listed in the Pass 1 analysis as a change to ch01 (not ch03) and was not included in the compression edits applied to ch03. This is expected.

### New Material Duplication Check

Checked whether the summary lines added by compressions introduced new overlapping content:

1. **C1 summary (ch03/03 line 86):** "The `TT_CCL` instance exposes this as a method: `self.get_num_links(cluster_axis)`." This is a single new sentence. No duplication with other chapter content.

2. **C2 summary (ch03/01 line 104):** "The mapper distributes host tensors to devices..." This sentence overlaps mildly with the table at lines 82-87 (which describes mesh_mapper purpose as "How to distribute host tensor to devices"). However, the summary serves as a bridge paragraph after the dataclass definition and adds the concrete T3K example, so this is acceptable restatement rather than problematic duplication.

3. **C4 consolidated note (ch03/02 lines 153-154):** This note cross-references ch03/03 rather than re-explaining the trace compatibility issue. No new duplication.

4. **C5 cross-ref (ch03/03 lines 310-311):** Summary states "6 CCL operations per Gemma4 block" which is also stated in the ch03/04 table header at line 470. This is a minimal one-number echo necessary for the cross-reference to be useful. Acceptable.

5. **C6 recap (ch03/01 lines 169-173):** The sentence "DeviceInit is a singleton that maps each `MeshDevice` to its `DistributedConfig`" echoes the same concept at ch02/01 line 413. This is the minimal restatement needed to make the chapter self-navigable. Acceptable.

**No problematic new duplication was introduced by the compression edits.**

### Verdict

- **Crucial compressions applied:** C1, C2, C4, C5, C6 -- all 5 compressions targeting ch03 were applied correctly.
- **C3 (ch01 change):** Correctly excluded from ch03 edits; it targets ch01/03 and is outside the scope of this pass.
- **New duplication:** None introduced.
- **Crucial updates: yes -- all applied.**
