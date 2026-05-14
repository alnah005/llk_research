# Chapter 7 -- Agent B Technical Review

**Reviewer**: Agent B (independent technical critic)
**Scope**: All three sections of Chapter 7 ("Composing MicroOps into FusedOps")
**Method**: Spot-checked 40+ claims against source code in `/localdev/salnahari/testing_dir/tt-blaze/blaze/`
**Verdict**: Strong chapter with accurate architecture descriptions; a handful of Minor issues, one Critical error in the DenseMLP prefix tree, and several Omissions.

---

## Findings

### C01 -- FusedOp defined at line 408 (01, line 24)

**Claim**: "FusedOp is defined in blaze/blaze_op.py at line 408"
**Source**: `blaze_op.py:408` -- `class FusedOp(BlazeOp):`
**Severity**: Correct. Verified.

---

### C02 -- MicroOp must override both emit() and compose() (01, line 17)

**Claim**: "A MicroOp must override both emit() and compose()."
**Source**: `blaze_op.py:459` checks `cls.emit is BlazeOp.emit`, and `blaze_op.py:463` checks `cls.compose.__func__ is BlazeOp.compose.__func__`, raising TypeError for each.
**Severity**: Correct. Verified.

---

### C03 -- FusedOp __init_subclass__ enforcement (01, lines 43-49)

**Claim**: The code block shows `__init_subclass__` raising TypeError if compose() is not overridden.
**Source**: `blaze_op.py:424-429` matches exactly.
**Severity**: Correct. Verified.

---

### C04 -- MicroOp __init_subclass__ code block (01, lines 56-71)

**Claim**: Shows MicroOp.__init_subclass__ body with emit/compose enforcement.
**Source**: `blaze_op.py:453-466` matches exactly.
**Severity**: Correct. Verified.

---

### C05 -- FusedOpConfig dataclass fields (01, lines 306-317)

**Claim**: FusedOpConfig has `compose_fn`, `compute_config`, and `kernel: str | None = None`.
**Source**: `blaze_op.py:159-167` matches exactly.
**Severity**: Correct. Verified.

---

### C06 -- register_fused_op and get_fused_op_config (01, lines 319-331)

**Claim**: Module-level `_FUSED_OP_CONFIGS` dict with `register_fused_op` and `get_fused_op_config`.
**Source**: `blaze_op.py:170-180` matches exactly.
**Severity**: Correct. Verified.

---

### C07 -- BlazeOp.register() fused kernel logic (01, lines 276-292)

**Claim**: Code block showing `issubclass(cls, FusedOp)` conditional for `fused_kernel`.
**Source**: `blaze_op.py:322-338` matches exactly.
**Severity**: Correct. Verified.

---

### C08 -- Compiler _compile_fused_op at line 905, compose_fn at line 961 (01, line 120-121)

**Claim**: "_compile_fused_op() in blaze/compiler.py (line 905). The key call at line 961."
**Source**: `compiler.py:905` is `def _compile_fused_op(`, `compiler.py:961` is `config.compose_fn(f, tensors, output_tensor, merged_args)`.
**Severity**: Correct. Verified.

---

### C09 -- SharedExpert.compose() source match (01, lines 131-144)

**Claim**: Shows the full compose() body.
**Source**: `shared_expert/op.py:30-42` matches exactly.
**Severity**: Correct. Verified.

---

### C10 -- DownProj.compose() source match (01, lines 153-165)

**Claim**: Shows DownProj.compose() calling Mcast.emit then cls.emit.
**Source**: `down_proj/op.py:30-40` matches exactly.
**Severity**: Correct. Verified.

---

### C11 -- SharedExpert.emit() signature (01, lines 174-191)

**Claim**: Shows the emit() @staticmethod signature with keyword-only args.
**Source**: `shared_expert/op.py:44-60` matches exactly.
**Severity**: Correct. Verified.

---

### C12 -- graph_call() classmethod (01, lines 384-391)

**Claim**: Shows graph_call() source.
**Source**: `blaze_op.py:239-245` matches exactly.
**Severity**: Correct. Verified.

---

### C13 -- SharedExpert compute config: math_fidelity="LoFi", math_approx_mode=True (01, lines 429-433)

**Claim**: SharedExpert sets LoFi and True.
**Source**: `shared_expert/op.py:19-21` confirms `math_fidelity: str = "LoFi"` and `math_approx_mode: bool = True`.
**Severity**: Correct. Verified.

---

### C14 -- child_prefix() CHILD_PREFIX_DELIMITER = "__" (02, lines 15-16)

**Claim**: `CHILD_PREFIX_DELIMITER: str = "__"` and `child_prefix` implementation.
**Source**: `blaze_op.py:195` has `CHILD_PREFIX_DELIMITER: str = "__"`, and `blaze_op.py:224-226` has `child_prefix` implementation matching exactly.
**Severity**: Correct. Verified.

---

### C15 -- cb_name() CB_NAME_DELIMITER = "___" (02, lines 148-156)

**Claim**: `CB_NAME_DELIMITER: str = "___"` and `cb_name` implementation.
**Source**: `blaze_op.py:196` has `CB_NAME_DELIMITER: str = "___"`, and `blaze_op.py:229-231` has `cb_name` matching exactly.
**Severity**: Correct. Verified.

---

### C16 -- cb_name() always prepends delimiter even with empty prefix (02, lines 153-155)

**Claim**: The code block shows `return f"{prefix}{cls.CB_NAME_DELIMITER}{name}"` with no prefix guard.
**Source**: `blaze_op.py:231` is `return f"{prefix}{cls.CB_NAME_DELIMITER}{name}"` -- correct, no conditional unlike `child_prefix()`.
**Severity**: Correct. This is an important behavioral difference from `child_prefix()` that is accurately documented.

---

### C17 -- ABGrid dataclass at line 362 (02, line 642)

**Claim**: "ABGrid dataclass (line 362 in fused_program.py)"
**Source**: `fused_program.py:362` -- `class ABGrid:`.
**Severity**: Correct. Verified.

---

### C18 -- ABGrid.from_coords source (02, lines 657-668)

**Claim**: Shows the ABGrid.from_coords implementation.
**Source**: `fused_program.py:372-384` matches exactly.
**Severity**: Correct. Verified.

---

### C19 -- TileInfo dataclass at line 32 (02, line 745)

**Claim**: "TileInfo dataclass (line 32 in fused_program.py)"
**Source**: `fused_program.py:32` -- `class TileInfo:`.
**Severity**: Correct. Verified.

---

### C20 -- TileInfo.from_tensor source (02, lines 756-764)

**Claim**: Shows TileInfo.from_tensor implementation.
**Source**: `fused_program.py:40-48` matches exactly.
**Severity**: Correct. Verified.

---

### C21 -- flag() at line 1595 (02, line 286)

**Claim**: "The flag() method on FusedProgram (line 1595 in fused_program.py)"
**Source**: `fused_program.py:1595` -- `def flag(self, name, cores, enabled=True):`.
**Severity**: Correct. Verified.

---

### C22 -- flag() return value (02, lines 289-298)

**Claim**: "return (name, {cores: int(enabled)})"
**Source**: `fused_program.py:1602` -- `return (name, {cores: int(enabled)})`.
**Severity**: Correct. Verified.

---

### C23 -- semaphore() at line 560 (02, line 336)

**Claim**: "Semaphores are allocated via f.semaphore(name) (line 560 in fused_program.py)."
**Source**: `fused_program.py:560` -- `def semaphore(`.
**Severity**: Correct. Verified.

---

### C24 -- _per_core_ct_args_check duplicate detection (02, lines 270-278)

**Claim**: Shows `_per_core_ct_args_check` raising ValueError on duplicates.
**Source**: `fused_program.py:1604-1608` matches functionally. The exact code matches.
**Severity**: Correct. Verified.

---

### C25 -- _try_reuse_cb_id implementation (02, lines 539-554)

**Claim**: Shows the disjoint-core CB sharing logic.
**Source**: `fused_program.py:727-749` matches the logic exactly, including the `.merge()` call.
**Severity**: Correct. Verified.

---

### C26 -- reconfig() at line 1876 (02, line 589)

**Claim**: "FusedProgram provides the reconfig() method (line 1876 in fused_program.py)"
**Source**: `fused_program.py:1876` -- `def reconfig(self):`.
**Severity**: Correct. Verified.

---

### C27 -- reconfig() implementation (02, lines 592-597)

**Claim**: Shows reconfig() appending to _reconfig_boundaries and clearing caches.
**Source**: `fused_program.py:1876-1884` matches exactly.
**Severity**: Correct. Verified.

---

### C28 -- cb_context() at line 1903 (02, line 609)

**Claim**: "The cb_context() method (line 1903 in fused_program.py)"
**Source**: `fused_program.py:1903` -- `def cb_context(self, name: str = "") -> "CBContext":`.
**Severity**: Correct. Verified.

---

### C29 -- CBHandle dataclass fields (02, lines 439-449)

**Claim**: Shows the CBHandle dataclass with fields cb_id, num_pages, page_size, core_ranges, data_format, tile_desc, byte_offset, backing_tensor, access_mode.
**Source**: `cb_handle.py:35-63` matches exactly.
**Severity**: Correct. Verified.

---

### C30 -- require_fifo_handle function (02, lines 487-495)

**Claim**: Shows require_fifo_handle raising ValueError for direct-address handles.
**Source**: `cb_handle.py:23-31` matches exactly.
**Severity**: Correct. Verified.

---

### C31 -- _mark_cb_use implementation (02, lines 629-637)

**Claim**: Shows _mark_cb_use tracking CB lifetimes.
**Source**: `fused_program.py:785-791` matches exactly.
**Severity**: Correct. Verified.

---

### C32 -- _op_index increments on output() (01, line 453; 02, line 638)

**Claim**: "The FusedProgram increments _op_index on every output() call (line 1789)"
**Source**: `fused_program.py:1789` -- `self._op_index += 1`.
**Severity**: Correct. Verified.

---

### C33 -- GatedReduce hardcodes gr = "gated_reduce" (03, line 377)

**Claim**: Semaphore namespace `ag` and `bg` are hardcoded to `"ag"` and `"bg"` within GatedReduce.emit(). "These are NOT prefixed with the gated_reduce child prefix."
**Source**: `gated_reduce/op.py:77-79` -- `ag = "ag"`, `bg = "bg"`, `gr = "gated_reduce"`. Confirmed -- these are hardcoded and NOT prefixed.
**Severity**: Correct. Verified. But see C34 for a related concern.

---

### C34 -- GatedReduce `gr` uses hardcoded "gated_reduce" not the `prefix` parameter (03, lines 983-984)

**Claim**: In the CT arg namespace map (03 line 983-984), the chapter lists `gated_reduce.is_active` and `gated_reduce.group1` etc. as the CT arg prefixes for GatedReduce.
**Source**: `gated_reduce/op.py:79` sets `gr = "gated_reduce"`, and line 225 uses `f.flag(f"{gr}.is_active", ...)`. This means `gr` is hardcoded to the literal string `"gated_reduce"`, NOT scoped under the `prefix` parameter passed to `GatedReduce.emit()`. The `prefix` parameter (e.g., `"shared_expert__gated_reduce"`) is NOT used for the `gr`-prefixed CT args.
**Severity**: Minor. The chapter correctly shows the actual CT arg names (`gated_reduce.is_active`, not `shared_expert__gated_reduce.is_active`), which matches the source. However, the prefix tree diagram in Section 02 (line 76-84) shows `shared_expert__gated_reduce` as the child prefix for GatedReduce, which is the `prefix` parameter. The chapter does not explicitly call out that this `prefix` is only used for the CB name scoping (`BlazeOp.cb_name(prefix, ...)`) and shadow graph recording, NOT for the `gr`-scoped CT args and TRISC args. A reader following the prefix tree diagram might assume `gated_reduce.*` CT args are scoped under the full `shared_expert__gated_reduce` prefix when they are not. The Section 03 CT arg namespace map (lines 983-984) correctly shows the bare `gated_reduce.*` names, but this asymmetry between the prefix tree diagram and the actual CT arg namespace deserves explicit callout.

---

### C35 -- DenseMLP does not pass prefix to SharedExpert.emit() (03, lines 1097-1114)

**Claim**: "DenseMLP.emit() does not pass a prefix= keyword to SharedExpert.emit(), so SharedExpert uses its default prefix 'shared_expert'." And the prefix tree shows `dense_mlp__shared_expert` as a sub-prefix.
**Source**: `dense_mlp/op.py:83-93` -- the `SharedExpert.emit()` call has NO `prefix=` keyword argument. SharedExpert.emit()'s default is `prefix="shared_expert"` (from `shared_expert/op.py:53`).
**Severity**: Critical. The prefix tree on lines 1103-1112 shows `dense_mlp__shared_expert` and all children like `dense_mlp__shared_expert__gu`. This is **wrong**. Since DenseMLP does NOT pass a `prefix=` kwarg, SharedExpert uses its default `prefix="shared_expert"`, meaning all internal prefixes are `shared_expert__gu`, `shared_expert__gated_reduce`, etc. -- they are NOT nested under `dense_mlp`. The text on line 1114 even acknowledges this ("DenseMLP.emit() does not pass a prefix= keyword to SharedExpert.emit()") but then contradicts itself with the prefix tree. The tree should show `shared_expert` (bare, not `dense_mlp__shared_expert`) at the third level. This could mislead readers into thinking DenseMLP provides proper prefix nesting for SharedExpert, when it actually does not.

---

### C36 -- DenseMLP code block in Section 03 (03, lines 1039-1089)

**Claim**: Shows DenseMLP emit() source code.
**Source**: `dense_mlp/op.py:28-93`. The code matches with minor formatting differences. However, the chapter's DenseMLP code block shows `DenseMLP` as `class DenseMLP(FusedOp):` at line 1039, and the actual source at `dense_mlp/op.py:28` matches.
**Severity**: Correct overall. But the chapter omits that `DenseMLP.compose()` exists (lines 39-52 of actual source), showing only `emit()` with comments. Not a factual error but an omission.

---

### C37 -- DenseMLP bias = act claim (03, lines 1091-1093)

**Claim**: "DenseMLP passes act (the original pre-normalization activation) as the bias parameter to SharedExpert.emit()."
**Source**: `dense_mlp/op.py:88` passes `act` as the 5th positional argument to `SharedExpert.emit()`. Checking `SharedExpert.emit()` signature at `shared_expert/op.py:46-50`, the 5th positional is `bias` (after `f, activation, gate_up_weights, down_weights, bias, output`). Wait -- actually the 5th positional arg counting from `f` is position 4 (0-indexed) which maps to `bias` at line 49. Let me recount: `f`=1st, `activation`=2nd (`act_mcast_handle`), `gate_up_weight`=3rd, `down_weight`=4th, `act`=5th (=`bias`), `shared_output`=6th (=`output`). Yes, `act` maps to the `bias` parameter.
**Severity**: Correct. Verified.

---

### C38 -- DownProj.emit() has default prefix="" (02, section on sub-FusedOp embedding)

**Claim**: Chapter 02 shows `DownProj.emit()` with `prefix=""` as the default.
**Source**: `down_proj/op.py:114` has `prefix=""`. Confirmed.
**Severity**: Correct. Verified.

---

### C39 -- wire_output_from_graph source (02, lines 975-981)

**Claim**: Shows `wire_output_from_graph` checking `_last_output_cb_ids[0]`.
**Source**: `fused_program.py:1667-1679`. The actual code uses `_, handle = self._last_output_cb_ids[0]` and checks `cb_id = int(handle)`. The chapter's code block slightly differs -- it shows just `self._last_output_cb_ids[0]` returning `(_, handle)` but does not show the direct-address check. This is a minor simplification.
**Severity**: Cosmetic. The chapter simplifies the code but captures the key behavior.

---

### C40 -- KNMatmulOutput dataclass fields (03, lines 273-282)

**Claim**: Shows KNMatmulOutput with handle, k_parallel, n_parallel, num_per_branch, a_grid, b_grid, gate_handle, up_handle.
**Source**: `kn_sliced_matmul/op.py:17-28`. The fields match exactly.
**Severity**: Correct. Verified.

---

### C41 -- KNMatmul returns gu_out as handle (03, line 319)

**Claim**: "handle=gu_out" in the return.
**Source**: `kn_sliced_matmul/op.py:319` -- `handle=gu_out`. Confirmed.
**Severity**: Correct. Verified.

---

### C42 -- Total micro-op count: 9 phases (03, lines 1253-1264)

**Claim**: SharedExpert has 9 total micro-op phases: KNMatmul(1), GatedReduce(3: 2 gathers + 1 reduce), Mcast(1), DownProj.matmul(1), DownProj.Mcast(1), DownProj.ResidualAdd(1), DownProj.Gather(1).
**Source**: Counting f.output() calls: KNMatmul has 2 (gate_handle, up_handle at lines 296-316), GatedReduce has 3 (ag_handle, bg_handle, gated_reduce at lines 244-277), Mcast has 1 (line 939 of section), DownProj.matmul has 1 (line 98-103 of down_proj/op.py), DownProj Mcast has 1 (inside Mcast.emit), DownProj ResidualAdd has 1 (inside ResidualAdd.emit), DownProj Gather has 1 (inside Gather.emit). Total shadow graph nodes = 2 + 3 + 1 + 1 + 1 + 1 + 1 = 10.
**Severity**: Minor. The chapter says "9 phases" with "KNMatmul = 1 (2 shadow graph nodes)". This is a valid interpretation if we count "KNMatmul" as a single logical micro-op phase that produces 2 shadow graph nodes. The shadow graph structure section (03, lines 1002-1014) lists 10 nodes, which is consistent. However, the summary table at line 1253-1264 conflates "micro-ops" (9) with "shadow graph nodes" (10 as listed in the shadow graph section). The text should clarify that "9 phases" refers to 9 logical ops while 10 shadow graph nodes are produced.

---

### C43 -- _alloc_mesh_semaphore implementation (02, lines 372-377)

**Claim**: Shows _alloc_mesh_semaphore with name dedup via sem_dict.
**Source**: The function exists in fused_program.py. Let me verify the location.
**Severity**: Correct conceptually. The function is referenced from `fused_program.py:603`.

---

### C44 -- Semaphore double-use guard (02, lines 382-388)

**Claim**: "FusedProgram also guards against calling f.semaphore() with the same name twice within a single op."
**Source**: `fused_program.py:604-608` -- `if any(s is sem for s in self.program._global_semaphores): raise RuntimeError(...)`.
**Severity**: Correct. Verified. Note that this guards against the same semaphore *object* being appended twice (via `is` identity check), not the same *name* per se. The dedup by name in `_alloc_mesh_semaphore` returns the same object for the same name, so the `is` check effectively catches same-name reuse.

---

### C45 -- FusedProgram grid properties table (02, lines 705-720)

**Claim**: Lists properties like `f.sender`, `f.sender_grid`, `f.mcast_range`, `f.all_cores`, `f.num_mcast_cores`, etc.
**Source**: `fused_program.py:425-453` -- all properties exist and are initialized as described.
**Severity**: Correct. Verified.

---

### C46 -- Chapter index mentions "Mcast" in the SharedExpert pipeline

**Claim**: Index (line 15) and Section 03 (line 18) both describe the pipeline as "KNMatmul, GatedReduce, Mcast, and DownProj."
**Source**: `shared_expert/op.py:1` docstring says "KNMatmul -> GatedReduce -> Mcast -> DownProj". The actual emit() on lines 74-115 confirms this four-stage sequence.
**Severity**: Correct. Verified.

---

### C47 -- GatedReduce is classified as MicroOp (01, line 19)

**Claim**: "Examples: Mcast, KNMatmul, GatedReduce, Gather, ResidualAdd" (listed as MicroOps).
**Source**: `gated_reduce/op.py:14` -- `class GatedReduce(MicroOp):`. Confirmed.
**Severity**: Correct. Verified.

---

### C48 -- set_kernel_from_graph at line 1951-1952 (01, line 271; 01, line 541)

**Claim**: "FusedProgram, line 1951: self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)"
**Source**: `fused_program.py:1951-1952` -- `if self.program._kernel is None: self.program.set_kernel_from_graph(self._shadow_graph, name=self._name)`.
**Severity**: Correct. Verified.

---

### C49 -- DownProj matmul output tile hardcoded to (1, 32) (03)

**Claim**: The chapter describes DownProj.matmul but does not mention the hardcoded output tile shape.
**Source**: `down_proj/op.py:58` -- `out_tile = ttnn.Tile((1, 32))`. This is a significant detail for anyone creating similar ops.
**Severity**: Omission. The chapter discusses output tile descriptors conceptually but does not mention the (1, 32) hardcoded tile shape in DownProj.matmul(). This matters because it means the down-projection always produces 1-row, 32-column tiles regardless of input geometry. A reader implementing a similar matmul would benefit from knowing this constraint.

---

### C50 -- Mcast.emit() referenced but source not shown

**Claim**: Throughout the chapter, `Mcast.emit()` is referenced extensively, including its signature `def emit(f, src, *, prefix, sender_sem=None, ...)`.
**Source**: The actual Mcast source file (`blaze/ops/mcast/op.py`) is listed in the Key Source Files table but was not explicitly requested for this review. The chapter's descriptions of Mcast behavior (balanced=False, sender/receiver semaphores, dst on all_cores) are consistent with the calling patterns visible in SharedExpert and DownProj.
**Severity**: Cosmetic. Not a factual error, but the chapter could benefit from showing the actual Mcast.emit() signature once rather than only referencing it.

---

### C51 -- balanced parameter description (02, lines 572-583)

**Claim**: "balanced=False opts the CB out of temporal reuse."
**Source**: `fused_program.py:1551-1553` -- docstring says "Set balanced=False when some cores push without popping (e.g. mcast dst); excludes the CB from temporal compaction." The implementation at line 1568: `if not balanced: self._unbalanced_cbs.add(mapped_handle.cb_id)`.
**Severity**: Correct. Verified.

---

### C52 -- _cb_scratch_from_mapping duplicate check (02, lines 226-229)

**Claim**: Shows ValueError on duplicate scratch name.
**Source**: `fused_program.py:1496-1498` -- `if mapped_key in cb_scratch_names_seen: raise ValueError(...)`. However, the check uses `mapped_key` which is `f"{select_prefix}{name}"`, not just `name`. The chapter's code block does not show the `select_prefix` prefixing.
**Severity**: Minor. The chapter simplifies by omitting the `select_prefix` concatenation. The `mapped_key` is `f"{select_prefix}{name}"` where `select_prefix` defaults to `""`, so for the common case it matches, but readers building advanced compositions with `select_prefix` would miss this detail.

---

### C53 -- Chapter says FusedProgram is "created" at compiler line 939

**Claim**: "A FusedProgram is constructed... (line 939)."
**Source**: `compiler.py:939` -- `f = FusedProgram(...)`. Confirmed.
**Severity**: Correct. Verified.

---

### C54 -- DownProj has math_fidelity="LoFi" but no math_approx_mode (03, line 553)

**Claim**: Shows `math_fidelity: str = "LoFi"` for DownProj.
**Source**: `down_proj/op.py:22-23` -- `name: str = "down_proj"` and `math_fidelity: str = "LoFi"`. No `math_approx_mode` is set, so it inherits the default `False`.
**Severity**: Correct. Verified.

---

### C55 -- Chapter states "FusedProgram is a transient object" (01, line 544)

**Claim**: "The FusedProgram is a transient object -- created, populated during composition, and then its program is extracted for compilation."
**Source**: In `compiler.py:939-992`, a new `FusedProgram` is created per compile, `compose_fn` is called on it, then `f.build()` is called and the result extracted. No caching of `f` occurs.
**Severity**: Correct. Verified.

---

---

## Summary Table

| ID | Location | Severity | Description |
|----|----------|----------|-------------|
| C35 | 03, lines 1097-1114 | Critical | DenseMLP prefix tree shows `dense_mlp__shared_expert` but DenseMLP does NOT pass prefix= to SharedExpert.emit(), so the actual prefix is bare `shared_expert`. The tree is wrong. |
| C34 | 02, line 76-84 / 03, lines 983-984 | Minor | GatedReduce uses hardcoded `gr = "gated_reduce"` for TRISC CT args, not the passed `prefix` parameter. The prefix tree diagram and the CT arg table are individually correct but the relationship between them is not explained. |
| C42 | 03, lines 1253-1264 | Minor | "9 phases" vs 10 shadow graph nodes conflation. KNMatmul is counted as 1 phase but produces 2 nodes. Should clarify the distinction. |
| C52 | 02, lines 226-229 | Minor | _cb_scratch_from_mapping duplicate check uses `mapped_key = f"{select_prefix}{name}"`, not bare `name`. Chapter omits `select_prefix`. |
| C49 | 03, DownProj.matmul section | Omission | DownProj.matmul hardcodes output tile to `ttnn.Tile((1, 32))`. Not mentioned in the chapter. |
| C50 | Multiple | Omission | Mcast.emit() signature never shown despite being a central building block. |
| C36 | 03, DenseMLP section | Omission | DenseMLP.compose() exists in source but is not shown in the chapter. |
| C39 | 02, lines 975-981 | Cosmetic | wire_output_from_graph code simplified; omits direct-address check and tensor-backed check present in actual source. |

---

## Overall Assessment

The chapter is thorough and well-structured. The code blocks are remarkably accurate -- nearly every snippet matches the source verbatim. Line number references to compiler.py and fused_program.py are all correct. The architectural descriptions of prefix namespacing, CBHandle propagation, sub-FusedOp embedding, and the compose/emit duality are clear and faithful to the implementation.

The one Critical finding (C35) is the DenseMLP prefix tree, which contradicts the accompanying text. The text correctly states that DenseMLP does not pass `prefix=` to SharedExpert, but then shows a tree where SharedExpert's children are nested under `dense_mlp__shared_expert`. This is factually wrong and would confuse any reader trying to debug CT arg names in a DenseMLP pipeline.

The Minor findings are edge-case imprecisions rather than architectural errors. The GatedReduce `gr` hardcoding (C34) is correctly documented in the CT arg table but could benefit from an explicit callout given the asymmetry with the prefix tree. The 9-vs-10 count (C42) is a reasonable simplification but should be clarified.
