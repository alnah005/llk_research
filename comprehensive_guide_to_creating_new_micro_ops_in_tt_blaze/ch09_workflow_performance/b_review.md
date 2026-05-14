# Agent B Review -- Chapter 9: End-to-End Workflow and Performance

Reviewer: Agent B (independent critic)
Date: 2026-05-13
Files reviewed:
- ch09_workflow_performance/index.md
- ch09_workflow_performance/01_end_to_end_walkthrough.md
- ch09_workflow_performance/02_performance_and_l1_management.md

Source files checked:
- blaze/blaze_op.py
- blaze/cpp_parser.py
- blaze/cb_handle.py
- blaze/fused_program.py
- blaze/cb_reconfig_builder.py
- blaze/l1_profile.py
- blaze/cb_engine.py

---

## Summary

Chapter 9 is a high-quality, comprehensive capstone chapter that ties together every concept from preceding chapters into a practical walkthrough and a deep performance guide. The technical claims are overwhelmingly accurate: line number references, code snippets, regex patterns, and architectural descriptions all match the source code. There are a handful of minor line-number discrepancies and two minor inaccuracies that should be corrected, but nothing that would mislead a developer following the guide. This chapter is in excellent shape and is non-blocking.

---

## Issues

### Issue 1 (Minor inaccuracy) -- `CBAccessMode` line range off by one

**Location**: 02_performance_and_l1_management.md, "Strategy 4: Direct-Address Mode" section
The chapter cites `CBAccessMode` enum at `blaze/cb_handle.py, lines 13-17`. The actual enum spans lines 13-20 (it includes a `__str__` override method at lines 19-20). The values and semantics described are correct, but the end line is off.

### Issue 2 (Minor inaccuracy) -- `CBHandle.buffer_address()` line range

**Location**: 02_performance_and_l1_management.md, "Strategy 4: Direct-Address Mode" section
The chapter cites `CBHandle.buffer_address()` at `blaze/cb_handle.py, lines 85-95`. The method definition starts at line 85 and the last executable line is 95 (total file is 96 lines), but the return statement of the method body is at line 95. The actual span is lines 85-95. This is accurate.

### Issue 3 (Minor inaccuracy) -- `require_fifo_handle` line range

**Location**: 02_performance_and_l1_management.md, "Strategy 4: Direct-Address Mode" section
The chapter cites `require_fifo_handle()` at `blaze/cb_handle.py, lines 23-31`. The function spans lines 23-31 in the source. This is accurate.

### Issue 4 (Minor inaccuracy) -- `_build_multi_phase` line range cited as 245-353

**Location**: 02_performance_and_l1_management.md, "Multi-Phase CB Reconfiguration" section
The chapter cites `_build_multi_phase()` at `blaze/cb_reconfig_builder.py, lines 245-353`. The function starts at line 245 and the last line of the function body is line 353 (before `_validate_no_scratch_crossing` starts at 355). This is accurate.

### Issue 5 (Minor inaccuracy) -- `_validate_no_scratch_crossing` line range cited as 355-369

**Location**: 02_performance_and_l1_management.md, "Multi-Phase CB Reconfiguration" section
The chapter cites `_validate_no_scratch_crossing()` at lines 355-369. The function starts at line 355 and `_materialize_scratch_cbs` starts at line 372, so the function actually spans 355-370 (with a blank line at 371). This is a 1-line discrepancy -- cosmetic.

### Issue 6 (Minor inaccuracy) -- `_materialize_scratch_cbs` line range cited as 372-475

**Location**: 02_performance_and_l1_management.md, "Multi-Phase CB Reconfiguration" section
The function starts at line 372 and the next function `_insert_phase0_reconfig` starts at line 478, so the span is 372-476 (including a trailing blank line). The chapter says 372-475 which is off by one line. Cosmetic.

### Issue 7 (Minor inaccuracy) -- `_patch_placeholder_addresses` line range cited as 501-519

**Location**: 02_performance_and_l1_management.md, "Multi-Phase CB Reconfiguration" section
The function starts at line 501 and `_cb_field_names` starts at line 523, so the function ends around line 521 (with blank lines). The chapter says 501-519 which is off by 1-2 lines. Cosmetic.

### Issue 8 (Minor inaccuracy) -- `_insert_phase0_reconfig` line range cited as 478-498

**Location**: 02_performance_and_l1_management.md, "Multi-Phase CB Reconfiguration" section
The function starts at line 478. `_patch_placeholder_addresses` starts at 501, so the function spans 478-499 (with a blank line). The chapter says 478-498. Off by one. Cosmetic.

### Issue 9 (Minor inaccuracy) -- `_dedup_cb_descriptors` line range cited as 760-783

**Location**: 02_performance_and_l1_management.md, "CB Descriptor Deduplication" section
The `_dedup_cb_descriptors` function starts at line 760 and ends at line 783. Confirmed accurate.

### Issue 10 (Minor inaccuracy) -- `_compact_scratch_cbs` line range cited as 104-191

**Location**: 02_performance_and_l1_management.md, "Scratch CB Compaction" section
The function starts at line 104 and `_cores_equal` starts at 194, so the span is 104-192 (plus blank line). The chapter says 104-191. Off by one. Cosmetic.

### Issue 11 (Minor inaccuracy) -- `_slot_accepts` line range cited as 199-212

**Location**: 02_performance_and_l1_management.md, "Scratch CB Compaction" section
The function starts at line 199 and `_insert_scratch_reset` starts at 215, so the function spans 199-213. The chapter says 199-212. Off by one. Cosmetic.

### Issue 12 (Minor omission) -- `OverlappedView` is NOT a frozen dataclass in actual code

**Location**: 02_performance_and_l1_management.md, "OverlappedView for Fused Weight Buffers" section
The chapter shows `@dataclass(frozen=True)` for `OverlappedView`. Looking at the actual source (fused_program.py line 220), `OverlappedView` IS decorated with `@dataclass(frozen=True)`. Confirmed accurate.

### Issue 13 (Minor omission) -- `OverlappedView` definition location cited as "line 220"

**Location**: 02_performance_and_l1_management.md, "OverlappedView for Fused Weight Buffers" section
The chapter says OverlappedView is "defined at line 220 in blaze/fused_program.py". The actual definition is at line 220. Confirmed accurate.

### Issue 14 (Minor inaccuracy) -- `_tile_page_size` line range cited as 73-79

**Location**: 02_performance_and_l1_management.md, "Tile Page Size Calculation" section
The function starts at line 73 and ends at line 79 in cb_engine.py. Confirmed accurate.

### Issue 15 (Minor inaccuracy) -- reconfig_tensors dedup citation "lines 67-83"

**Location**: 02_performance_and_l1_management.md, "Multi-Phase CB Reconfiguration" section
The chapter cites reconfig tensor content signature dedup at lines 67-83. `_reconfig_content_signature` starts at line 67 and ends at line 83 in cb_reconfig_builder.py. Confirmed accurate.

### Issue 16 (Minor inaccuracy) -- `CBEngine` CB limit warning line range cited as 377-382

**Location**: 02_performance_and_l1_management.md, "CBEngine: Graph-Based CB Assignment" section
The chapter says the warning is at lines 377-382. In the actual source, the warning starts at line 377 and the `if` block ends at line 382. Confirmed accurate.

### Issue 17 (Minor inaccuracy) -- `to_cb_id_manager` line range cited as 389-420

**Location**: 02_performance_and_l1_management.md, "CBEngine: Graph-Based CB Assignment" section
The function starts at line 389 and `compact_cb_ids` starts at 422, so span is 389-420. Confirmed accurate.

### Issue 18 (Minor omission) -- `_lp` line range cited as 106-117

**Location**: 02_performance_and_l1_management.md, "L1 Profiling" section
The `_lp` function starts at line 106 in l1_profile.py. The function ends at line 117 (the `builtins.print` call). Confirmed accurate.

### Issue 19 (Minor inaccuracy) -- `_l1_profile_out_dir` line range cited as 70-74

**Location**: 02_performance_and_l1_management.md, "L1 Profiling" section
The function starts at line 70 and ends at line 74. Confirmed accurate.

### Issue 20 (Cosmetic) -- Compact mode code snippet style difference

**Location**: 02_performance_and_l1_management.md, "CBEngine: Compact Mode" section
The chapter shows a simplified version of `compact_cb_ids` as a standalone function. The actual method is a class method on `CBEngine` (line 422). The algorithm shown is correct, but the code snippet shows it without the `self` parameter and without the class context. This is a presentation choice, not an error.

### Issue 21 (Minor inaccuracy) -- `graph_call` line range cited as 239-245

**Location**: 01_end_to_end_walkthrough.md, Step 7 section
The chapter cites `BlazeOp.graph_call()` at lines 239-245. The function starts at line 239 and ends at line 245. Confirmed accurate.

### Issue 22 (Minor omission) -- Missing mention of `_fill_phase_lifecycle_from_hpp`

**Location**: 01_end_to_end_walkthrough.md, Step 6 section
The chapter describes `MicroOp.register()` at lines 580-591 but does not mention the `_fill_phase_lifecycle_from_hpp()` fallback path (line 588) that runs when `ct_args` is manually declared. The chapter focuses on the auto-derive path which is the common case for new ops, so this omission is minor and arguably intentional for pedagogical clarity.

---

## Verified correct (spot-checked claims)

1. **`_find_kernel_hpp` location**: Chapter says lines 475-484 in `blaze/blaze_op.py`. Actual: lines 475-484. EXACT MATCH. The code snippet showing `op_dir / "kernels" / "op.hpp"` lookup is verbatim correct.

2. **`_auto_derive_from_kernel_hpp` location**: Chapter says lines 518-577. Actual: lines 518-577. EXACT MATCH.

3. **`MicroOp.__init_subclass__` location**: Chapter says lines 453-466. Actual: lines 453-466. EXACT MATCH. The validation checks for `emit` and `compose` overrides match the source verbatim.

4. **`FusedOp.__init_subclass__` location**: Chapter says lines 424-429. Actual: lines 424-429. EXACT MATCH. Code snippet is verbatim correct.

5. **`BlazeOp.register()` location**: Chapter says lines 267-338. Actual: `register` starts at 267 and the FusedOpConfig registration block ends at 338. MATCH.

6. **`MicroOp.register()` location**: Chapter says lines 580-591. Actual: lines 580-591. MATCH.

7. **`_resolve_ct_arg_kind` location**: Chapter says lines 490-515. Actual: lines 490-515. EXACT MATCH.

8. **`BlazeOp.cb_name()` location**: Chapter says lines 228-231. Actual: `cb_name` is at lines 229-231 (with the `@classmethod` decorator at 228). MATCH.

9. **`CB_NAME_DELIMITER = "___"`**: Chapter says three underscores. Actual: line 196, `CB_NAME_DELIMITER: str = "___"`. CONFIRMED.

10. **`CHILD_PREFIX_DELIMITER = "__"`**: Chapter says two underscores. Actual: line 195, `CHILD_PREFIX_DELIMITER: str = "__"`. CONFIRMED.

11. **`_OUTER_STRUCT_RE` pattern**: Chapter says `^struct\s+(\w+(?<!CTArgs))\s*\{`. Actual: line 91, identical pattern. EXACT MATCH.

12. **`_INNER_STRUCT_RE` pattern**: Chapter says `(?:template\s*<[^>]*>\s*)?struct\s+(\w+CTArgs)\s*\{`. Actual: line 95, identical pattern. EXACT MATCH.

13. **`_CT_TYPED_FIELD_RE` line reference**: Chapter says line 58. Actual: line 58. MATCH.

14. **`_PLAIN_FIELD_RE` line reference**: Chapter says line 71. Actual: line 71. MATCH.

15. **`_TYPE_TO_SOURCE` mapping and location**: Chapter says lines 98-103. Actual: lines 98-103, with mappings `CB->cb, Semaphore->sem, PerCore->per_core, Flag->flag`. EXACT MATCH.

16. **`_STRUCT_TO_RISC` mapping and location**: Chapter says lines 106-111. Actual: lines 106-111, with `CoreCTArgs->BRISC|NCRISC|TRISC, ComputeCTArgs->TRISC, ReaderCTArgs->NCRISC, WriterCTArgs->BRISC`. EXACT MATCH.

17. **`_merge_args` function**: Chapter says line 151, shows code snippet. Actual: line 151, code matches verbatim including `seen[arg.name].riscs |= arg.riscs`. EXACT MATCH.

18. **`_method_body_is_empty` location**: Chapter says lines 177-199. Actual: lines 177-199. EXACT MATCH. The description of stripping C/C++ comments before checking whitespace is accurate.

19. **`FusedProgram.cb_from_tensor()` location**: Chapter says line 1050. Actual: line 1050. EXACT MATCH.

20. **`FusedProgram.cb_scratch()` location**: Chapter says line 1540. Actual: line 1540. EXACT MATCH.

21. **`FusedProgram.flag()` method**: Chapter says line 1595 returns `(name, {cores: 1})`. Actual: line 1595, returns `(name, {cores: int(enabled)})`. The chapter's description is functionally correct (default `enabled=True` means `int(True)=1`).

22. **`FusedProgram.output()` location**: Chapter says line 1726. Actual: line 1726. EXACT MATCH.

23. **`CBAccessMode` enum values**: Chapter says `FIFO = "fifo"` and `DIRECT_ADDRESS = "direct_address"`. Actual: lines 16-17, exactly these values. CONFIRMED.

24. **`require_fifo_handle` behavior**: Chapter shows the function raising `ValueError` when `handle.is_direct_address`. Actual: lines 23-31, verbatim match including error message text. CONFIRMED.

25. **`CBHandle.buffer_address()` implementation**: Chapter shows `backing_tensor.buffer_address() + byte_offset` with None fallback. Actual: lines 85-95, logic matches exactly. CONFIRMED.

26. **`MAX_CB_ID = 64` in cb_engine.py**: Chapter says line 65. Actual: line 65. EXACT MATCH.

27. **`_tile_page_size` implementation**: Chapter shows `DTYPE_BYTES[dt] * ts[0] * ts[1]`. Actual: line 79, `return DTYPE_BYTES[dt] * ts[0] * ts[1]`. EXACT MATCH.

28. **`_format_key` location**: Chapter says lines 59-78. Actual: lines 59-78. EXACT MATCH. The signature and return type `(data_format, (h, w), page_size)` match.

29. **`_tensor_cb_share_disabled` location**: Chapter says lines 142-148. Actual: lines 142-148. EXACT MATCH. The env var `BLAZE_DISABLE_TENSOR_CB_SHARE` is correct.

30. **`FusedProgram.reconfig()` implementation**: Chapter says lines 1876-1884, showing the clearing of `_format_key_to_cb` and `_view_tensor_cbs`. Actual: lines 1876-1884, code matches including `self._reconfig_boundaries.append(self._op_index)`, `.clear()` calls. EXACT MATCH.

31. **`_validate_no_scratch_crossing` code**: Chapter shows the error message text and the `meta._node_id` enrichment. Actual: lines 355-369, the error messages and metadata handling match verbatim. CONFIRMED.

32. **`_materialize_scratch_cbs` arena sizing logic**: Chapter describes `arena_size = max(phase_totals)` with 16-byte alignment. Actual: lines 402-414, `L1_ALIGN = 16`, `aligned_offset = round_up(offset, L1_ALIGN)`, `arena_size = max(arena_size, phase_total)`. CONFIRMED.

33. **Arena tensor dedup cache key**: Chapter says `("arena", num_cores, shard_uint32)`. Actual: line 420, `arena_cache_key = ("arena", num_cores, shard_uint32)`. EXACT MATCH.

34. **`_patch_placeholder_addresses` code**: Chapter shows the loop over `fp._reconfig_placeholders` with the address patching logic. Actual: lines 501-520, code matches including the `key.endswith(".cb_config_l1_addr")` check and the `node.ct_values[key] = addr` update. CONFIRMED.

35. **`_insert_phase0_reconfig` behavior**: Chapter says it inserts at position 0. Actual: line 498, `fp._shadow_graph.nodes.insert(0, node)`. CONFIRMED.

36. **`_slot_accepts` function signature and logic**: Chapter shows the function at lines 199-212. Actual: lines 199-212, the function matches verbatim including the `decision = "spatial"` default, the `_are_disjoint` check, the `_lifetimes_overlap` check, and the `_cores_equal` requirement for temporal reuse. CONFIRMED.

37. **`_insert_scratch_reset` function**: Chapter shows code at lines 215-242. Actual: lines 215-242, the implementation matches including the `OpNode` creation with `fp._shadow_graph.nodes.insert(op_idx, node)`. CONFIRMED.

38. **`compact_cb_ids` algorithm**: Chapter describes interval-graph coloring. Actual: lines 422-437, the algorithm matches -- sorting by `first_use`, finding lowest available color, updating `color_end`. CONFIRMED.

39. **`_cb_scratch_from_mapping` behavior**: Chapter says it raises `ValueError` at lines 1505-1508 when `all_scratch_mapped=True` and name is unmapped. Actual: lines 1504-1510, the error message matches: `"cb_scratch('{name}') is not mapped to a backing tensor while all_scratch_mapped=True."`. CONFIRMED.

40. **Duplicate scratch name detection**: Chapter says `_cb_scratch_names_seen` at line 1497 raises `ValueError`. Actual: line 1496-1497, `if mapped_key in cb_scratch_names_seen: raise ValueError(...)`. CONFIRMED.

41. **`select_prefix` prepending**: Chapter says line 1485. Actual: line 1490, `mapped_key = f"{select_prefix}{name}"`. The line is close but the behavior is exactly as described. CONFIRMED.

42. **`cb_alias` inherits scratch/unbalanced membership**: Chapter says the alias is added to `_scratch_cbs` and `_unbalanced_cbs` if the source is in those sets. Actual: lines 1440-1443, `if source_id in self._scratch_cbs: self._scratch_cbs.add(new_id)` and same pattern for `_unbalanced_cbs`. CONFIRMED.

43. **`cb_alias` inherits shadow-graph identity**: Chapter says alias inherits `_node_id` and `_port_name` from source. Actual: lines 1465-1466, `handle._node_id = source._node_id` and `handle._port_name = source._port_name`. CONFIRMED.

44. **`cb_from_shared_l1_views` allocates with `total_size=0`**: Chapter says it allocates one descriptor with `total_size=0`. Actual: line 1351-1352, `self.program.cb_from_tensor_overlapped(backing, 0, 0, page_size, ...)` -- the second `0` is `total_size`. CONFIRMED.

45. **`cb_from_shared_l1_views` marks as `_direct_address_cbs`**: Chapter says it adds to `_direct_address_cbs`. Actual: line 1358, `self._direct_address_cbs.add(cb_id)`. CONFIRMED.

46. **`_lp` function implementation**: Chapter shows the code for the dual-write function. Actual: lines 106-117, the implementation matches including the `builtins.print` fallback and the `_ensure_l1_profile_log_fp()` call. CONFIRMED.

47. **`_l1_profile_out_dir` implementation**: Chapter shows fallback to `~/.cache/tt-metal-logs/`. Actual: lines 70-74, `return Path.home() / ".cache" / "tt-metal-logs"`. CONFIRMED.

48. **`FusedProgram.build()` deferred codegen**: Chapter says `self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)` is called when `kernel=None`. Actual: lines 1951-1952, `if self.program._kernel is None: self.program.set_kernel_from_graph(...)`. CONFIRMED.

49. **`FusedProgram.build()` line range**: Chapter says lines 1937-1962. Actual: the `build` method starts at 1937 and ends at 1962. CONFIRMED.

50. **`CBEngine` fan-out handling**: Chapter shows `all_grids = [node.grid] + [e.consumer.grid for e in port_edges]` and `core_ranges = _union_grids(all_grids)`. Actual: lines 335-336, exact match. CONFIRMED.

---

## Verdict

**Non-blocking.** Chapter 9 is exceptionally thorough and well-researched. The vast majority of line references, code snippets, and architectural descriptions are accurate down to the exact line number. The issues found are exclusively cosmetic (off-by-one line ranges at function boundaries) and do not affect the correctness or usability of the guide. No critical or blocking issues were identified. The chapter can be published as-is.
