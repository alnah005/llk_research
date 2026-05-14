# Agent B Review -- Chapter 8: Testing, Debugging, and Common Pitfalls

Reviewer: Agent B (independent critic)
Date: 2026-05-13
Files reviewed:
- ch08_testing_debugging/index.md
- ch08_testing_debugging/01_writing_tests.md
- ch08_testing_debugging/02_debugging_tools.md
- ch08_testing_debugging/03_common_hang_causes.md
- ch08_testing_debugging/04_common_gotchas.md

Source files checked:
- blaze/kernel_codegen.py (437 lines)
- blaze/cb_engine.py (438 lines)
- blaze/l1_profile.py (lines 1-200, 311-350, 482-520)
- blaze/cb_handle.py (96 lines)
- blaze/blaze_op.py (614 lines)
- blaze/fused_program.py (lines 1-200, 200-400, 400-700, 720-780, 1540-1625, 1964-2000)
- blaze/compiler.py (lines 1-400, 1002-1026)
- blaze/sem_engine.py (194 lines)
- blaze/ct_args.py (345 lines)
- blaze/graph.py (lines 86-140)
- blaze/visualizer.py (lines 40-70)
- blaze/cb_reconfig_builder.py (lines 105-125)
- tests/blaze/conftest.py (42 lines)

---

## Summary

Chapter 8 is thorough, well-organized, and technically accurate across the vast majority of its claims. Code snippets quoted from source files are overwhelmingly faithful to the actual implementations. One factual error was found regarding the default value of the `strict` parameter in `CTArgEngine.generate_tuples()`, and a handful of minor presentation issues were identified. The chapter is production-ready with one small correction.

---

## Issues

### Issue 1 (Minor inaccuracy) -- CTArgEngine `strict` default is False, not True

**Location**: 02_debugging_tools.md, lines 349-357

The chapter states: "When `strict=True` (the default), any CT arg declared in the schema that cannot be resolved triggers an error" and shows:

```python
ct_tuples = CTArgEngine().generate_tuples(
    graph, cb_for_ct, sem_for_ct,
    grid_context=grid_context,
    user_overrides=merged_overrides,
    strict=True,  # raises on missing args
)
```

In the actual source (`blaze/ct_args.py`, line 226), the default is `strict: bool = False`. The `generate()` method (line 131) also defaults to `strict: bool = False`. When `strict=False`, unresolved args produce a `warnings.warn()` rather than raising. The code example correctly passes `strict=True` explicitly, but the parenthetical "(the default)" is factually wrong.

**Fix**: Change "When `strict=True` (the default)" to "When `strict=True` (not the default -- must be passed explicitly)".

---

### Issue 2 (Minor omission) -- `generate_tuples` parameter names differ from chapter example

**Location**: 02_debugging_tools.md, lines 349-357

The chapter shows `generate_tuples(graph, cb_for_ct, sem_for_ct, grid_context=..., user_overrides=..., strict=True)`. The actual signature (`ct_args.py`, lines 219-227) uses `cb_assignments`, `sem_assignments`, `grid_context`, and `user_overrides` as parameter names. The chapter's `cb_for_ct` and `sem_for_ct` are not the actual parameter names. This is a minor cosmetic issue since the call site in `compiler.py` may use different variable names, but it could confuse a reader trying to call the function directly.

**Fix**: Use the actual parameter names from the function signature, or add a note that the variable names shown are from a specific call site.

---

### Issue 3 (Cosmetic) -- MicroOp compose TypeError message is truncated

**Location**: 04_common_gotchas.md, lines 488-494

The chapter shows the error message as:
```python
raise TypeError(
    f"{cls.__name__}: MicroOp must override compose()"
)
```

The actual error message (`blaze_op.py`, lines 463-466) is:
```python
raise TypeError(
    f"{cls.__name__}: MicroOp must override compose() to support the Graph API path"
)
```

This is cosmetic since the meaning is clear, but quoting the exact error message would help developers searching for it.

---

### Issue 4 (Minor omission) -- `_try_reuse_cb_id` uses `merge` not just disjoint check

**Location**: 02_debugging_tools.md, lines 268-276

The chapter correctly describes the `_try_reuse_cb_id` behavior and the `BLAZE_DISABLE_TENSOR_CB_SHARE` escape hatch, but does not mention that on a hit, the function calls `ex_cr.merge(core_ranges)` to extend the existing entry's grid. This is the mechanism that makes multi-caller sharing work progressively. A reader debugging CB sharing bugs would benefit from knowing that the grid grows with each reuse.

**Source**: `fused_program.py`, line 747: `entries[i] = (ex_cb_id, ex_cr.merge(core_ranges))`

---

### Issue 5 (Cosmetic) -- Missing `_is_abstract_op` guard in MicroOp `__init_subclass__`

**Location**: 04_common_gotchas.md, lines 483-494

The chapter's code snippet for MicroOp's `__init_subclass__` omits the `_is_abstract_op(cls)` early-return guard that skips validation for abstract intermediate classes (those without a `name` set). The actual code (`blaze_op.py`, lines 453-456) starts with:

```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if _is_abstract_op(cls):
        return
```

This guard is important because without it, defining an intermediate base class without `name` would raise a `TypeError`. The omission could confuse readers who create intermediate base classes.

---

### Issue 6 (Minor omission) -- Chapter does not mention `BLAZE_DISABLE_TEMPORAL_REUSE` source location

**Location**: 02_debugging_tools.md, lines 280-290

The chapter correctly describes `BLAZE_DISABLE_TEMPORAL_REUSE` and shows the source from `cb_reconfig_builder.py`, but the index page (`index.md`, line 27) lists the source as `blaze/compiler.py` for the `BLAZE_L1_PROFILE` gate while not listing `cb_reconfig_builder.py` in the Key Source File References table. Since `BLAZE_DISABLE_TEMPORAL_REUSE` is a significant debugging escape hatch, `cb_reconfig_builder.py` should appear in the reference table.

---

## Verified correct (spot-checked claims)

1. **`_RISC_GUARDS` dictionary**: Chapter quotes match exactly -- `blaze/kernel_codegen.py`, lines 99-106. All six RISC names and their preprocessor guards verified: `brisc`, `ncrisc`, `trisc`, `trisc0`, `trisc1`, `trisc2`.

2. **`_parse_debug_riscs()` function**: Full function body matches -- `kernel_codegen.py`, lines 109-135. Return type `tuple[bool, str | None]`, parsing logic for "0", "1"/"all", and comma-separated names all verified.

3. **`_to_ct_args_type()` function**: `prefix.replace(".", "__").replace("-", "_")` -- `kernel_codegen.py`, line 182.

4. **`_compute_prefix()` branch stripping logic**: Chapter matches `kernel_codegen.py`, lines 143-155. The `ct_prefix` kwarg fallback to `node.spec.name`, branch suffix stripping, all verified.

5. **`generate_kernel_file()` content hash**: `hashlib.sha256(source.encode()).hexdigest()[:12]` -- `kernel_codegen.py`, line 408.

6. **Debug kernel filename suffix**: `out_path = out_dir / f"{label}_{content_hash}_debug.cpp"` -- `kernel_codegen.py`, line 419.

7. **Node ordering**: `node_order = graph.nodes` comment "Use node insertion order" -- `kernel_codegen.py`, line 201.

8. **Mcast leader/follower codegen**: Leader gets `init()`/`teardown()` outside braces; followers use `run_as<>()` with the leader's variable -- `kernel_codegen.py`, lines 332-358.

9. **Non-mcast codegen pattern**: Variable declaration, init, operator call, teardown all inside `DeviceZoneScopedN` braces -- `kernel_codegen.py`, lines 370-378.

10. **`DPRINT` include guard**: Only included under the debug guard when a specific RISC is targeted -- `kernel_codegen.py`, lines 264-269.

11. **`deepseek_compute_kernel_init()` in TRISC guard**: Present at `kernel_codegen.py`, lines 281-283.

12. **`MAX_CB_ID = 64`**: `cb_engine.py`, line 65.

13. **`DEFAULT_NUM_PAGES = 1`**: `cb_engine.py`, line 70.

14. **CB engine warning threshold**: `if next_cb_id > self.max_cb_id - 8` -- `cb_engine.py`, line 377.

15. **`CBAssignment` dataclass fields**: `cb_id`, `total_size`, `page_size`, `num_pages`, `core_ranges`, `is_tensor_backed`, `data_format`, `tile_shape` -- `cb_engine.py`, lines 122-139.

16. **`CBEngine.assign()` num_pages resolution**: `num_pages = node.kwargs.get(f"num_tiles_{port_spec.name}", DEFAULT_NUM_PAGES)` -- `cb_engine.py`, line 224.

17. **`_SILICON_PATHS` tuple and `collect_ignore`**: Chapter content matches `tests/blaze/conftest.py`, lines 22-41 exactly. All paths verified: "backed", "fused_ops", "generality", "glm5_1", "micro-ops", "migration", plus 10 specific infra files.

18. **`CBAccessMode` enum**: `FIFO = "fifo"` and `DIRECT_ADDRESS = "direct_address"` -- `cb_handle.py`, lines 13-18.

19. **`require_fifo_handle()` function**: Raises `ValueError` with correct message -- `cb_handle.py`, lines 23-31.

20. **`CBHandle.__post_init__` validation**: Checks `is_direct_address and backing_tensor is None` -- `cb_handle.py`, lines 65-69.

21. **`Risc` Flag enum**: `class Risc(Flag): NCRISC = auto(); BRISC = auto(); TRISC = auto()` -- `blaze_op.py`, lines 47-50.

22. **`SemProtocol` enum values**: `SENDER = "sender_semaphore"`, `RECEIVER = "receiver_semaphore"`, `NOC0_RECEIVER = "noc0_receiver_semaphore"`, `NOC1_RECEIVER = "noc1_receiver_semaphore"` -- `blaze_op.py`, lines 96-103.

23. **`BlazeOp.CB_NAME_DELIMITER = "___"` and `CHILD_PREFIX_DELIMITER = "__"`**: `blaze_op.py`, lines 195-196.

24. **`BlazeOp.cb_name()` method**: `return f"{prefix}{cls.CB_NAME_DELIMITER}{name}"` -- `blaze_op.py`, lines 229-230.

25. **`BlazeOp.child_prefix()` method**: `return f"{prefix}{cls.CHILD_PREFIX_DELIMITER}{name}" if prefix else name` -- `blaze_op.py`, lines 224-226.

26. **`CompiledProgram.run()` BLAZE_L1_PROFILE gate**: `if os.environ.get("BLAZE_L1_PROFILE"): from blaze.l1_profile import print_cb_stats; print_cb_stats(self)` -- `compiler.py`, lines 119-121.

27. **`MeshCompiledProgram.run()` BLAZE_L1_PROFILE gate**: Same pattern -- `compiler.py`, lines 1022-1024.

28. **`_tensor_cb_share_disabled()` function**: Returns `bool(os.environ.get("BLAZE_DISABLE_TENSOR_CB_SHARE"))` -- `fused_program.py`, lines 142-148.

29. **`_try_reuse_cb_id()` first guard**: `if _tensor_cb_share_disabled() or core_ranges is None: return None` -- `fused_program.py`, lines 737-738.

30. **`cb_scratch()` signature with `balanced` parameter**: `balanced: bool = True` -- `fused_program.py`, line 1549.

31. **`cb_scratch()` unbalanced tracking**: `if not balanced: self._unbalanced_cbs.add(...)` -- `fused_program.py`, lines 1567-1568 and 1581-1582.

32. **`_unbalanced_cbs` docstring**: Comment matches chapter description about mcast dst pushed on every receiver but popped only by consumer subset -- `fused_program.py`, lines 513-517.

33. **`FusedProgram.flag()` method**: `return (name, {cores: int(enabled)})` -- `fused_program.py`, line 1602.

34. **`_per_core_ct_args_check()` duplicate detection**: Checks `name in self._ct_arg_names_seen` and raises `ValueError(f"Duplicate per-core CT arg: {name!r}")` -- `fused_program.py`, lines 1604-1608.

35. **`FusedProgram.run()` auto-output detection**: Uses `eligible[-1][1]` from `_io_tensors` excluding `_direct_address_cbs` -- `fused_program.py`, lines 1988-1999.

36. **`FusedProgram.run()` kernel auto-generation**: Calls `self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)` when `_kernel is None` -- `fused_program.py`, lines 1974-1975.

37. **`_alloc_mesh_semaphore()` dedup logic**: Checks `if name not in sem_dict` before allocating, uses `initial_value` only on first allocation -- `fused_program.py`, lines 151-161.

38. **`FusedProgram.semaphore()` duplicate detection**: `if any(s is sem for s in self.program._global_semaphores): raise RuntimeError(...)` -- `fused_program.py`, lines 604-608.

39. **`visualize()` function signature**: Matches chapter exactly -- `blaze/visualizer.py`, lines 45-54: `graph`, `engine_result`, `grid_config`, keyword-only `name`, `tensors`, `output_dir`, `open_browser`.

40. **`BLAZE_EXPORT` auto-export in compile()**: `if os.environ.get("BLAZE_EXPORT"): self._auto_export(...)` -- `compiler.py`, lines 338-339.

41. **`BLAZE_EXPORT_PATH` env var**: `export_dir = os.environ.get("BLAZE_EXPORT_PATH", self._EXPORT_DEFAULT_DIR)` where default is `"generated/viz"` -- `compiler.py`, lines 343, 359.

42. **`EngineResult` dataclass fields**: `cb_assignments`, `cb_by_node`, `sem_assignments`, `sem_by_node`, `ct_tuples` -- `graph.py`, lines 86-98.

43. **`TileInfo.from_tensor()` method**: Extracts tile, data_format, uses `tile.get_tile_size(data_format)` for page size -- `fused_program.py`, lines 41-48.

44. **`_view_total_size_from_shape()` alignment check**: `if shard_h % tile_h or shard_w % tile_w: raise ValueError(...)` -- `fused_program.py`, lines 97-103.

45. **`BLAZE_DISABLE_TEMPORAL_REUSE` env var**: `allow_temporal = not os.environ.get("BLAZE_DISABLE_TEMPORAL_REUSE")` -- `cb_reconfig_builder.py`, line 117.

46. **`extract_cb_names()` function exists and takes `descriptor_or_program`**: `l1_profile.py`, line 482.

47. **`print_cb_stats()` and `print_kernel_stats()` functions exist**: `l1_profile.py`, lines 328 and 311 respectively.

48. **L1 profile log path**: Falls back to `~/.cache/tt-metal-logs/` when `TT_METAL_CACHE` is unset -- `l1_profile.py`, lines 71-74.

49. **MicroOp `__init_subclass__` compose check**: Raises `TypeError` when `compose.__func__ is BlazeOp.compose.__func__` -- `blaze_op.py`, lines 463-466.

50. **`CTArgSchema.args_for_risc()` method**: Returns `[a for a in self.args if a.risc == risc]` -- `ct_args.py`, lines 69-71.

---

## Verdict

**Non-blocking.** The chapter is technically accurate, well-sourced, and comprehensive. The one substantive inaccuracy (Issue 1: `strict` default claimed as `True` when it is actually `False`) should be corrected before publication, as it could lead a developer to believe missing CT args will always raise rather than warn. The remaining issues are cosmetic or minor omissions that do not affect correctness of the guidance. The chapter provides excellent coverage of the debugging workflow, test infrastructure, and common pitfalls -- the catalog of hang causes and gotchas is particularly well-matched to the actual source code mechanisms.
