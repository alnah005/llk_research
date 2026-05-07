# Chapter 4 -- Correctness Review (Agent B)

Review date: 2026-05-07
Reviewer: Agent B (Correctness Verifier)

Sources checked:
- `blaze/cb_handle.py` (95 lines)
- `blaze/fused_program.py` (2360 lines)
- `blaze/cb_reconfig.py` (218 lines)
- `blaze/cb_reconfig_builder.py` (807 lines)
- `blaze/program.py` (BlazeProgram)

---

## File: 01_cbhandle_abstraction.md

### Verified Claims

- CBAccessMode enum inherits from (str, Enum), defines FIFO and DIRECT_ADDRESS values, custom `__str__` -- exact match with `cb_handle.py:13-20`.
- `require_fifo_handle()` signature, body, and error message -- exact match with `cb_handle.py:23-31`.
- CBHandle dataclass with `eq=False`, field names, types, defaults, and order -- exact match with `cb_handle.py:34-63`.
- `__post_init__` body (access_mode coercion, direct-address requires backing_tensor) -- exact match with `cb_handle.py:65-69`.
- `__int__` and `__index__` both return `self.cb_id` -- exact match with `cb_handle.py:71-75`.
- `is_fifo` and `is_direct_address` properties using `is` identity comparison -- exact match with `cb_handle.py:77-83`.
- `buffer_address()` method body (backing_tensor.buffer_address() + byte_offset, or None) -- exact match with `cb_handle.py:85-95`.
- `cb_from_tensor()` when passed a CBHandle calls `require_fifo_handle` -- exact match with `fused_program.py:1050-1052`.
- Shadow graph recording of `_node_id` and `_port_name` in `_record_op` -- matches `fused_program.py:2074-2082`.
- Edge wiring in `_wire_input_edges` using `source._node_id` and `source._port_name` -- matches `fused_program.py:2032-2053`.
- `_mark_cb_use` logic for lifetime tracking -- exact match with `fused_program.py:785-791`.
- `set_cb_core_ranges` mutates handle in place -- confirmed at `fused_program.py:1104-1112`.
- `cb_alias` inherits `_node_id` and `_port_name` from source -- confirmed at `fused_program.py:1465-1466`.
- `_tensor_handle_cache` vs `_cb_metadata` separation -- confirmed in `fused_program.py:530-533` and `cb_for_tensor_or_none` at `fused_program.py:872-891`.
- `cb_for_tensor_or_none` skips `_direct_address_cbs` -- confirmed at `fused_program.py:881,884`.

### Errors Found

**None.** All claims in this file are accurate.

---

## File: 02_fused_program.md

### Verified Claims

- FusedProgram constructor signature (all parameters, defaults) -- exact match with `fused_program.py:394-413`.
- Grid attributes (`sender`, `sender_grid`, `mcast_range`, `all_cores`, `num_mcast_cores`, `full_device_grid`) -- exact match with `fused_program.py:425-448`.
- Constructor also computes `matmul_cores`, `matmul_cores_cc`, `mcast_receiver_grid`, NOC coords -- confirmed at `fused_program.py:434-453`.
- `cb_from_tensor` dispatches: CBHandle -> require_fifo_handle, OverlappedView -> cb_from_view, else disjoint-share then fresh -- confirmed at `fused_program.py:1050-1065`.
- `cb_from_tensor_overlapped` flow -- confirmed at `fused_program.py:1067-1102`.
- `cb_output` / `cb_output_overlapped` mark tensor as output -- confirmed at `fused_program.py:1114-1162`.
- `cb_from_view` three paths (same-tensor disjoint, format-key, fresh) -- confirmed at `fused_program.py:1164-1255`.
- `cb_from_shared_l1_views` returns DIRECT_ADDRESS handles -- confirmed at `fused_program.py:1257-1368`.
- `cb_scratch` flow (mapping check then bare allocation) -- confirmed at `fused_program.py:1540-1593`.
- `cb_alias` behavior (extends lifetime, disjoint-cores sharing, inherits _node_id/_port_name) -- confirmed at `fused_program.py:1370-1468`.
- `set_cb_core_ranges` finds descriptor by `fmt.buffer_index` -- confirmed at `fused_program.py:1104-1112`.
- `wire_output` rejects direct-address, preserves tile/page_size from original -- confirmed at `fused_program.py:1626-1665`.
- `wire_output_from_graph` auto-detects from `_last_output_cb_ids` -- confirmed at `fused_program.py:1667-1695`.
- CT arg wrappers (unified, ncrisc, brisc, trisc) call `_track_cb_arg_positions` + `_capture_ct_values` + forward -- confirmed at `fused_program.py:793-811`.
- `_track_cb_arg_positions` body -- exact match with `fused_program.py:773-783`.
- Positional CT args (`ncrisc_positional_ct_args`, `brisc_positional_ct_args`) -- confirmed at `fused_program.py:813-827`.
- RT arg pass-throughs -- confirmed at `fused_program.py:829-863`.
- Per-core CT arg duplicate check via `_per_core_ct_args_check` -- confirmed at `fused_program.py:1604-1624`.
- `flag()` method returns `(name, {cores: int(enabled)})` -- exact match with `fused_program.py:1595-1602`.
- Semaphore method two allocation models (named mesh-global vs program-local) -- confirmed at `fused_program.py:560-610`.
- `named_tensor()` signature and dedup behavior -- confirmed at `fused_program.py:612-672`.
- `output()` signature and behavior (create OpNode, stamp handle, advance _op_index) -- confirmed at `fused_program.py:1726-1831`.
- `multi_output()` returns MultiOutput, outputs is `list[tuple[str, CBHandle]]` -- confirmed at `fused_program.py:1833-1872`.
- Internal tracking state table (all attributes listed) -- cross-checked against `fused_program.py:499-537`, all match.
- Context manager `__enter__`/`__exit__` with `_cleanup_tensors` -- exact match with `fused_program.py:551-558`.
- `_format_key()` function -- exact match with `fused_program.py:59-78`.
- `build()` method flow -- confirmed at `fused_program.py:1937-1962`.
- `run()` method -- confirmed at `fused_program.py:1964-2000`.
- `MeshFusedProgram` shares `_sem_dict`, `_tensor_dict`, `_internal_tensor_dict` -- confirmed at `fused_program.py:2264-2302`.
- Mesh attributes (`mesh_coord`, `mesh_shape`, `mesh_device`) -- confirmed at `fused_program.py:467-469`.
- `build_ab_grids` -- confirmed at `fused_program.py:539-545`.
- `set_core_ranges` -- confirmed at `fused_program.py:676-683`.
- `add_define` -- confirmed at `fused_program.py:684-686`.

### Errors Found

**Error 1**
- **Claim (Semaphores section):** "Allocates a `ttnn.GlobalSemaphore` on the full device grid, keyed by name in `_sem_dict`."
- **Source:** `_alloc_mesh_semaphore` at `fused_program.py:158-160` allocates on the compute-with-storage grid (`mesh_device.compute_with_storage_grid_size()`), not the full device grid. The grid is derived from `ttnn.num_cores_to_corerangeset(gs.x * gs.y, gs, row_wise=True)` where `gs = mesh_device.compute_with_storage_grid_size()`.
- **Severity:** MODERATE -- "full device grid" is misleading; the semaphore grid is the compute-with-storage grid, which excludes non-compute cores.
- **Suggested fix:** Replace "on the full device grid" with "on the compute-with-storage grid (derived from `device.compute_with_storage_grid_size()`)".

**Error 2**
- **Claim (Semaphores section):** "`core_ranges` is not valid here."
- **Source:** `fused_program.py:594-595` shows `core_ranges` is explicitly rejected with a `RuntimeError("core_ranges is only valid with program_semaphore=True")` when the named path is taken. The chapter statement is functionally correct but could be more precise: `core_ranges` IS accepted as a parameter in the named path, but raises RuntimeError if set.
- **Severity:** MINOR -- The claim is directionally correct (you cannot use `core_ranges` with named semaphores), but the phrasing "is not valid here" could be read as "is not accepted as a parameter" rather than "raises an error if provided". This is a very minor imprecision.
- **Suggested fix:** Rephrase to "`core_ranges` raises `RuntimeError` if provided with named mesh-global semaphores."

**Error 3**
- **Claim (RT args section):** Lists `trisc_rt_arg_arrays` but does NOT list it in the same block -- actually, looking again, the chapter does list `trisc_rt_arg_arrays` correctly in the RT arg arrays section. However, the per-core RT arg arrays section lists `trisc_per_core_rt_arg_arrays` which IS present in the source. But the chapter also lists `brisc_per_core_rt_arg_arrays(args)` which IS in the source at `fused_program.py:853-855`. All RT arg methods check out.
- Actually: No error here. All listed methods are confirmed in source.

---

## File: 03_overlapped_view.md

### Verified Claims

- OverlappedView dataclass is `frozen=True` with all listed fields -- exact match with `fused_program.py:219-261`.
- `page_size` property three-path logic -- exact match with `fused_program.py:264-283`.
- `memory_config()` returns `_OverlappedMemoryConfig(_OverlappedShardSpec(...))` -- exact match with `fused_program.py:285-290`.
- `_OverlappedShardSpec` and `_OverlappedMemoryConfig` helper dataclasses (frozen=True) -- exact match with `fused_program.py:206-216`.
- `from_overlapped_tensor()` classmethod body (field mapping: fused_tensor -> tensor, core_range_set -> core_ranges, etc.) -- exact match with `fused_program.py:292-314`.
- `cb_from_view()` three sharing paths (same-tensor disjoint, format-key cross-tensor, fresh) -- confirmed at `fused_program.py:1164-1255`.
- `cb_from_shared_l1_views()` validation checks (8 conditions listed) -- all confirmed at `fused_program.py:1269-1344`.
- `cb_from_shared_l1_views()` allocates with `address_offset=0, total_size=0` and `DIRECT_ADDRESS` -- confirmed at `fused_program.py:1351-1368`.
- `_format_key()` function -- exact match with `fused_program.py:59-78`.
- `_view_total_size_from_shape()` function -- exact match with `fused_program.py:94-103`.
- `_tensor_shard_bytes()` description -- matches `fused_program.py:106-139`.
- `_are_disjoint()` and `_core_ranges_subset()` functions -- exact match with `fused_program.py:52-91`.
- MultiOutput class with `__init__(self, outputs: dict[str, CBHandle])` and three access patterns -- exact match with `fused_program.py:317-358`.
- `__iter__` yields values (not keys), `__len__` returns count, `__contains__` checks key membership -- confirmed at `fused_program.py:347-354`.

### Errors Found

**None.** All claims in this file are accurate.

---

## File: 04_cb_reconfig.md

### Verified Claims

- `CircularBufferIdManager` class with `NUM_CIRCULAR_BUFFERS = 64`, `_id_to_format`, `_next_id = 0` -- exact match with `cb_reconfig.py:18-29`.
- `_normalize_tile()` static method -- exact match with `cb_reconfig.py:31-36`.
- `_allocate_id()` core logic (reuse path scanning `_id_to_format`, fresh allocation with overflow check) -- matches `cb_reconfig.py:53-75`. The code snippets shown accurately represent the logic.
- `Context` inner class with `_used_ids` set and `get_cb_id` -- exact match with `cb_reconfig.py:77-88`.
- `build_dummy_cb_descriptors()` with `page_size = 1`, `total_size = page_size` -- exact match with `cb_reconfig.py:92-108`.
- `CBContext` class (`__init__`, `get_cb_id`, `pin`, `build_reconfig_tensor`) -- exact match with `cb_reconfig.py:111-135`.
- `record_cb_metadata()` function -- exact match with `cb_reconfig.py:138-159`.
- `build_cb_reconfig_tensor()` function: 264 words per core, torch.uint32, core_to_idx, 4-word CB config, bitmask at 256-257 -- exact match with `cb_reconfig.py:162-218`.
- Synthetic keys: `_RECONFIG_TENSOR_KEY_BASE = -1000`, `_ARENA_TENSOR_KEY = -2000` -- exact match with `cb_reconfig_builder.py:27-28`.
- `prepare_for_build()` entry point with `_compaction_applied` guard and three paths -- exact match with `cb_reconfig_builder.py:31-48`.
- `_compact_scratch_cbs()` logic: spatial + temporal reuse, format key is 4-tuple `(data_format, tile_shape, page_size, num_pages)` -- confirmed at `cb_reconfig_builder.py:104-191`.
- `_Slot` NamedTuple definition -- exact match with `cb_reconfig_builder.py:98-101`.
- `_slot_accepts()` logic -- exact match with `cb_reconfig_builder.py:199-212`.
- `_cores_equal()` function -- exact match with `cb_reconfig_builder.py:194-196`.
- Eligibility filter via `desc_counts` -- confirmed at `cb_reconfig_builder.py:122-133`.
- Post-compaction cleanup order (remap, dedup, lifetime extension, insert in descending order) -- confirmed at `cb_reconfig_builder.py:174-191`.
- `_build_multi_phase()` 6-step pipeline -- confirmed at `cb_reconfig_builder.py:245-353`.
- `_validate_no_scratch_crossing()` logic -- confirmed at `cb_reconfig_builder.py:355-369`.
- `_materialize_scratch_cbs()`: `L1_ALIGN = 16`, per-phase offset assignment, arena_size = max(phase_totals) -- confirmed at `cb_reconfig_builder.py:372-476`.
- Arena dedup key `("arena", num_cores, shard_uint32)` -- confirmed at `cb_reconfig_builder.py:420`.
- Phase 3 (assign CB IDs): fresh Context per phase from shared manager -- confirmed at `cb_reconfig_builder.py:270-286`.
- Phase 4 (build reconfig tensors): `_partition_descriptors_by_phase`, content signature dedup, empty phase validation -- confirmed at `cb_reconfig_builder.py:288-329`.
- `_patch_placeholder_addresses()` patches both per-RISC arg lists and shadow graph node `ct_values` -- exact match with `cb_reconfig_builder.py:501-520`.
- `_insert_phase0_reconfig()` inserts at position 0 -- exact match with `cb_reconfig_builder.py:478-498`.
- `_dedup_cb_descriptors()` function (group, sort, claim, split partials) -- confirmed at `cb_reconfig_builder.py:760-783`.
- `_remap_cb_ids()` five-target rewrite (1a: _cb_arg_positions, 1b: spec fallback, 2: descriptors, 3: io_tensor keys, 4: metadata, 5: membership sets) -- confirmed at `cb_reconfig_builder.py:541-610`.
- `_partition_descriptors_by_phase()` dedup by `(phase, id(desc))` -- confirmed at `cb_reconfig_builder.py:786-807`.
- `_reconfig_content_signature()` uses sorted CB IDs and `_core_ranges_signature` -- confirmed at `cb_reconfig_builder.py:67-83`.
- Environment variable escape hatches (`BLAZE_DISABLE_TENSOR_CB_SHARE`, `BLAZE_DISABLE_TEMPORAL_REUSE`) -- confirmed at `fused_program.py:142-148` and `cb_reconfig_builder.py:117`.
- FusedProgram `reconfig()` clears `_format_key_to_cb` and `_view_tensor_cbs` -- confirmed at `fused_program.py:1876-1884`.
- FusedProgram `cb_context()` calls `reconfig()` internally -- confirmed at `fused_program.py:1903-1910`.
- FusedProgram `cb_manager` property (lazy init) -- confirmed at `fused_program.py:1912-1918`.
- 264-word layout table (64 CBs x 4 words, 2 bitmask words, 2 sync, 4 reserved) -- confirmed by `WORDS_PER_CORE = 264` and the construction loop in `cb_reconfig.py:175-198`.
- HEIGHT_SHARDED placement with `full_device_grid`, shard shape `(1, WORDS_PER_CORE)` -- confirmed at `cb_reconfig.py:204-209`.

### Errors Found

**Error 1**
- **Claim (seed_reserved_cb section):** The code snippet shown for `seed_reserved_cb` omits the type validation lines that the prose correctly describes:
  ```python
  def seed_reserved_cb(self, cb_id, data_format, tile) -> None:
      if cb_id not in self._id_to_format:
          ...
  ```
  The actual source (`cb_reconfig.py:38-51`) includes two `TypeError` checks before the `if cb_id not in` line:
  ```python
  if tile is not None and not isinstance(tile, ttnn.TileDescriptor):
      raise TypeError(...)
  if not isinstance(data_format, ttnn.DataType):
      raise TypeError(...)
  ```
- **Source:** `cb_reconfig.py:45-48`
- **Severity:** MINOR -- The prose correctly states "1. Validates that `tile` is a `TileDescriptor`..." but the code snippet is an incomplete reproduction. The behavior is described correctly in the numbered list; only the snippet is simplified.
- **Suggested fix:** Add the validation lines to the code snippet, or add a note "(validation elided for brevity)".

**Error 2**
- **Claim (_allocate_id section):** The code snippet for `_allocate_id` omits the type validation at the top of the method:
  ```python
  if tile is not None and not isinstance(tile, ttnn.TileDescriptor):
      raise TypeError(...)
  if not isinstance(data_format, ttnn.DataType):
      raise TypeError(...)
  ```
- **Source:** `cb_reconfig.py:60-63`
- **Severity:** MINOR -- Same pattern as Error 1. The validation exists in source but is omitted from the snippet. Since the chapter does not explicitly claim the method lacks validation, this is a minor omission.
- **Suggested fix:** Either add the validation or mark the snippet as simplified.

**Error 3**
- **Claim (_validate_no_scratch_crossing section):** The code snippet shows `name` as a simple variable in the f-string:
  ```python
  raise ValueError(
      f"CB '{name}' is scratch-allocated but referenced across "
  ```
  But the actual source (`cb_reconfig_builder.py:363-364`) computes `name` via:
  ```python
  meta = fp._cb_metadata.get(cb_id)
  name = f"{meta._node_id}.{meta._port_name}" if meta and meta._node_id else f"CB {cb_id}"
  ```
  The chapter shows `name` as though it comes from a local variable that is not defined in the snippet scope.
- **Source:** `cb_reconfig_builder.py:363-364`
- **Severity:** MINOR -- The error message text is correct; the snippet just omits the name-resolution logic. This is a simplification, not a factual error.
- **Suggested fix:** Add the `meta` lookup line or replace `name` with `f"CB {cb_id}"` in the snippet.

---

## Summary Verdict

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| MODERATE | 1 |
| MINOR | 3 |

### Detail

- **1 MODERATE** error in `02_fused_program.md`: Semaphore allocation grid described as "full device grid" when the source uses `compute_with_storage_grid_size()` (a meaningful distinction).
- **3 MINOR** errors: Code snippets that omit validation lines or name-resolution logic that the prose correctly describes or that don't materially affect understanding.

### Overall Assessment

Chapter 4 is **highly accurate**. Every dataclass definition, method signature, behavioral claim, constant value (64 CB slots, 264 words per core), and cross-reference has been verified against the source. The 264-word layout, the bitmask split at CB 32, the L1_ALIGN=16 value, the HEIGHT_SHARDED shard spec, the synthetic keys (-1000, -2000), and the six-step multi-phase pipeline are all correct. The one MODERATE error (semaphore grid) and three MINOR snippet omissions are the only discrepancies found across all four files.
