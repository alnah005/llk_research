# Agent B Review -- Chapter 5: Code Generation and Kernel Compilation

Reviewer: Agent B (independent critic)
Date: 2026-05-13
Files reviewed: index.md, 01_engine_pipeline.md, 02_auto_generated_kernels.md, 03_named_args_generated_header.md, 04_handwritten_kernels.md
Source files checked: blaze/kernel_codegen.py, blaze/compiler.py, blaze/cb_engine.py, blaze/sem_engine.py, blaze/ct_args.py, blaze/role_engine.py, blaze/unified_kernel_descriptor.py, blaze/fused_program.py, blaze/blaze_op.py, blaze/program.py

---

## Summary

Chapter 5 is a thorough and well-structured treatment of the TT-Blaze codegen pipeline. The vast majority of technical claims are accurate and directly traceable to source code. I found one critical issue (NCRISC/BRISC processor numbering is inverted), a few minor inaccuracies in pseudocode simplifications, and several minor omissions. The chapter is strong overall and will be highly useful to developers.

---

## Issues

### Issue 1 (Critical) -- NCRISC/BRISC DM0/DM1 processor numbering is inverted
**Location**: 03_named_args_generated_header.md, Section 5.3.4 table

The chapter states:
```
| 1 | COMPILE_FOR_NCRISC | DM0 (reader) |
| 2 | COMPILE_FOR_BRISC  | DM1 (writer) |
```

The actual source code in `unified_kernel_descriptor.py` (lines 328-332 and 348-352) shows:
- Reader (NCRISC) uses `DataMovementProcessor.RISCV_1` (not RISCV_0)
- Writer (BRISC) uses `DataMovementProcessor.RISCV_0` (not RISCV_1)

In TT-Metal convention, BRISC = RISCV_0 and NCRISC = RISCV_1. The "DM0/DM1" labels in the chapter are reversed. This could cause confusion when developers cross-reference TT-Metal documentation or debug at the hardware level.

**Fix**: Swap the DM labels: NCRISC is DM1 (reader), BRISC is DM0 (writer).

---

### Issue 2 (Minor inaccuracy) -- _create_global_semaphores pseudocode is oversimplified
**Location**: 01_engine_pipeline.md, Section 5.1.4

The chapter shows `_create_global_semaphores` as a simple loop:
```python
def _create_global_semaphores(self, sem_assignments):
    num_sems = max(sa.sem_id for sa in sem_assignments) + 1
    global_sems = [ttnn.create_global_semaphore(self._mesh_device, avail, 0)
                   for _ in range(num_sems)]
```

The actual implementation in `compiler.py` lines 640-664 uses `self._graph_sem_pool` with a `while` loop for deduplication and caching across repeated calls within a single `compile()`. The simplified version omits this caching behavior, which could mislead developers who try to understand how semaphore identity is maintained across per-device iterations.

---

### Issue 3 (Minor inaccuracy) -- CBAssignment.category docstring discrepancy
**Location**: 01_engine_pipeline.md, Section 5.1.3

The chapter lists four CB categories including `"internal"` and shows a `CBAssignment` dataclass with `category: str = ""  # "ext_in" | "intermed" | "ext_out" | "internal"`. The actual source at `cb_engine.py` line 136 only documents three: `"ext_in" | "intermed" | "ext_out"`. However, the code does use `category="internal"` at line 296. The chapter is actually more complete than the source docstring, but the presented dataclass code block does not exactly match the source.

---

### Issue 4 (Minor inaccuracy) -- _wire_input_edges code snippet differs from source
**Location**: 01_engine_pipeline.md, Section 5.1.2

The chapter shows `_wire_input_edges` accessing `self._shadow_graph.node_map[source._node_id]` as a dictionary lookup. The actual source at `fused_program.py` line 2039 uses `nmap = self._shadow_graph.node_map` and then `nmap[source._node_id]`, which is functionally identical. However, the chapter omits the ExternalTensor branch's handling of `tensor_to_ports` and `input_tensors` (lines 2051-2053). The chapter only shows `self._shadow_graph.external_input_ports.add(...)` for the ExternalTensor case, but the real code also populates `tensor_to_ports` and `input_tensors` dictionaries. These extra bookkeeping steps are relevant to downstream CBEngine operation (shared tensor deduplication depends on `tensor_to_ports`).

---

### Issue 5 (Minor inaccuracy) -- _create_op_node code snippet has slight structural difference
**Location**: 01_engine_pipeline.md, Section 5.1.2

The chapter's `_create_op_node` code omits the call to `self._validate_ct_args_against_schema()` that appears in the real code at `fused_program.py` lines 2027-2029. While not a factual error (the chapter did not claim the function is exhaustive), the omission of this validation step could surprise developers who read the chapter and then look at the source.

---

### Issue 6 (Omission) -- CBAssignment missing the "internal" category from source
**Location**: 01_engine_pipeline.md, Section 5.1.3

The chapter correctly identifies 4 categories (ext_in, internal, intermed, ext_out) in the table but the code snippet of the `CBAssignment` dataclass shows `category: str = ""` with a comment listing all 4. The actual source only lists 3 in the comment. While the chapter's table is correct, the presented code block subtly differs from the actual source on line 136.

---

### Issue 7 (Omission) -- No mention of PerCorePositionalCTADescriptor
**Location**: 03_named_args_generated_header.md, Section 5.3.3

The chapter covers `UnifiedCompileTimeCoreDescriptor` and `PerCoreCompileTimeDescriptor` but does not mention `PerCorePositionalCTADescriptor` (defined at `unified_kernel_descriptor.py` lines 81-106). This is a third per-core specialization mechanism for positional (non-named) compile-time args that differs per core, used for compressed tensor format arrays. For a comprehensive guide, mentioning its existence would help developers who encounter it.

---

### Issue 8 (Omission) -- Missing PerCoreRuntimeArgsDescriptor
**Location**: 03_named_args_generated_header.md, Section 5.3.5

The chapter lists several RT arg APIs (ncrisc_rt_args, brisc_rt_args, etc.) but does not mention `PerCoreRuntimeArgsDescriptor` (unified_kernel_descriptor.py lines 109-125), which provides per-core runtime args for each RISC. This is a distinct mechanism from the named per-core runtime arg arrays.

---

### Issue 9 (Minor inaccuracy) -- generate_kernel_file exception handling differs
**Location**: 02_auto_generated_kernels.md, Section 5.2.8

The chapter's pseudocode shows:
```python
except BaseException:
    os.close(fd)
    os.unlink(tmp)
    raise
```

The actual source at `kernel_codegen.py` lines 423-436 uses a `closed` flag to avoid double-closing the file descriptor, and uses `contextlib.suppress(OSError)` around the `os.unlink()` call. The chapter's simplified version would cause errors if `os.close(fd)` had already succeeded before the `os.rename()` failed.

---

### Issue 10 (Cosmetic) -- Copyright year discrepancy in generated kernel header
**Location**: 02_auto_generated_kernels.md, Section 5.2.4

The chapter shows the generated kernel header as:
```
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
```

The actual source at `kernel_codegen.py` line 255 uses an em dash:
```python
lines.append("// Auto-generated by blaze.kernel_codegen — do not edit manually.")
```
vs the chapter's double-dash `--`. Very minor but observable by developers comparing.

---

### Issue 11 (Cosmetic) -- Includes path is dynamic, not hardcoded
**Location**: 02_auto_generated_kernels.md, Section 5.2.4

The chapter shows hardcoded include paths:
```cpp
#include "blaze/kernels/ops.hpp"
#include "blaze/kernels/kernel_op_api.hpp"
```

The actual source at `kernel_codegen.py` lines 260-263 uses `KERNELS_DIR` variable:
```python
from . import KERNELS_DIR
lines.append(f'#include "{KERNELS_DIR}/ops.hpp"')
```

The actual path depends on the `KERNELS_DIR` constant. The chapter's version is a reasonable simplification but could cause confusion if the actual path differs.

---

### Issue 12 (Omission) -- No mention of `noc_mode` parameter
**Location**: 01_engine_pipeline.md

The `UnifiedKernelDescriptor` has a `noc_mode` field (default `DM_DEDICATED_NOC`) that affects the data movement configuration for both NCRISC and BRISC kernel descriptors. The chapter doesn't mention this parameter, which is relevant when working with dynamic NOC mode.

---

## Verified correct (spot-checked claims)

1. **PhaseInfo dataclass fields** (02_auto_generated_kernels.md, 5.2.1): All 6 fields (`cpp_type`, `alias_suffix`, `has_init_teardown`, `setup_method`, `init_is_empty`, `teardown_is_empty`) verified against `kernel_codegen.py` lines 36-54. Exact match.

2. **PhaseDecl dataclass fields** (02_auto_generated_kernels.md, 5.2.2): All 5 fields (`alias`, `op_type`, `prefix`, `template_args`, `phase_info`) verified against `kernel_codegen.py` lines 77-85. Exact match.

3. **McastGroup dataclass** (02_auto_generated_kernels.md, 5.2.3): Structure with `leader` and `followers` list verified against `kernel_codegen.py` lines 88-92. Exact match.

4. **_RISC_GUARDS dictionary** (02_auto_generated_kernels.md, 5.2.7): All 6 entries verified against `kernel_codegen.py` lines 99-106. Exact match.

5. **_parse_debug_riscs behavior** (02_auto_generated_kernels.md, 5.2.7): Table of values and behaviors verified against `kernel_codegen.py` lines 109-135. Correct.

6. **_to_alias function** (02_auto_generated_kernels.md, 5.2.2): Logic and all 4 examples verified against `kernel_codegen.py` lines 157-177. Correct.

7. **_to_ct_args_type function** (02_auto_generated_kernels.md, 5.2.2): Dot-to-double-underscore and dash-to-underscore replacement verified against `kernel_codegen.py` lines 180-182. Exact match.

8. **_compute_prefix for kernel_codegen** (02_auto_generated_kernels.md, 5.2.2): Branch suffix stripping logic verified against `kernel_codegen.py` lines 143-155. Exact match.

9. **generate_kernel_file content hash** (02_auto_generated_kernels.md, 5.2.8): SHA256 with 12-character truncation verified against `kernel_codegen.py` line 408. Correct.

10. **MAX_CB_ID = 64** (01_engine_pipeline.md, 5.1.3): Verified against `cb_engine.py` line 65. Exact match.

11. **Warning threshold at MAX_CB_ID - 8** (01_engine_pipeline.md, 5.1.3): Verified against `cb_engine.py` line 377 (`self.max_cb_id - 8`). Correct.

12. **compact_cb_ids algorithm** (01_engine_pipeline.md, 5.1.3): Interval coloring algorithm verified against `cb_engine.py` lines 422-437. Logic matches.

13. **CBAssignment dataclass** (01_engine_pipeline.md, 5.1.3): 12 fields verified against `cb_engine.py` lines 123-138. All present and correctly typed.

14. **SemAssignment dataclass** (01_engine_pipeline.md, 5.1.4): All 6 fields (`sem_id`, `global_sem`, `initial_value`, `core_ranges`, `protocol`, `node_id`) verified against `sem_engine.py` lines 41-49. Exact match.

15. **_McastSenderGroups class** (01_engine_pipeline.md, 5.1.4): Structure and `_sender_key` logic verified against `sem_engine.py` lines 52-85. Correct.

16. **CTArgSpec dataclass** (01_engine_pipeline.md, 5.1.6): All 6 fields verified against `ct_args.py` lines 34-56. Correct (note: source uses `source: str = "user"` not `"derived"` as default, but the chapter doesn't claim a specific default).

17. **CTArgEngine._resolve_value** (01_engine_pipeline.md, 5.1.6): Logic for "cb", "sem", "per_core", and "derived" sources verified against `ct_args.py` lines 276-313. Correct, including float-to-bits encoding.

18. **CTArgEngine._compute_prefix** (01_engine_pipeline.md, 5.1.6): Verified against `ct_args.py` lines 254-274. Behavior matches: dot suffix, empty prefix handling.

19. **_validate_no_collisions** (01_engine_pipeline.md, 5.1.6): Object identity deduplication via `seen_ids` verified against `ct_args.py` lines 315-344. Correct.

20. **GridConfig dataclass** (01_engine_pipeline.md, 5.1.5): Fields `grid_cols`, `grid_rows`, `dram_worker_positions` verified against `role_engine.py` lines 18-27. Exact match including `frozen=True`.

21. **GridConfig.sender_core** (01_engine_pipeline.md, 5.1.5): `(grid_cols - 1, grid_rows - 1)` verified against `role_engine.py` lines 29-30. Correct.

22. **GridConfig.from_device** (01_engine_pipeline.md, 5.1.5): Factory method verified against `role_engine.py` lines 108-119. Logic matches.

23. **flag() method** (01_engine_pipeline.md, 5.1.5): Returns `(name, {cores: int(enabled)})` verified against `fused_program.py` lines 1595-1602. Exact match.

24. **_EngineResults NamedTuple** (01_engine_pipeline.md, 5.1.7): Two fields (`cb_assignments`, `sem_assignments`) verified against `compiler.py` lines 81-85. Correct.

25. **FusedProgram.build() deferred codegen** (01_engine_pipeline.md, 5.1.2): The check `if self.program._kernel is None` and call to `set_kernel_from_graph` verified against `fused_program.py` lines 1950-1952. Correct.

26. **BlazeProgram.set_kernel_from_graph** (02_auto_generated_kernels.md, 5.2.8): Imports `generate_kernel_file` and sets `_generated = True` verified against `program.py` lines 172-177. Correct.

27. **BlazeProgram.generated_kernel property** (02_auto_generated_kernels.md, 5.2.8): Reads file and returns source string verified against `program.py` lines 179-185. Correct.

28. **UnifiedKernelDescriptor triple compilation** (03_named_args_generated_header.md, 5.3.4): Three kernel descriptors (NCRISC, BRISC, TRISC) verified against `unified_kernel_descriptor.py` lines 307-383. Correct structure.

29. **_get_split_kernel_descriptors core grouping** (03_named_args_generated_header.md, 5.3.4): Frozen-dict hashing for core grouping verified against `unified_kernel_descriptor.py` lines 475-490. Correct approach.

30. **BlazeOp.register() PhaseInfo auto-derivation** (02_auto_generated_kernels.md, 5.2.1): `cpp_type` derivation from `op_class` verified against `blaze_op.py` lines 306-320. Exact match.

31. **FusedOpConfig.kernel attribute** (04_handwritten_kernels.md, 5.4.2): `kernel: str | None = None` on `FusedOpConfig` verified against `blaze_op.py` line 167. Correct.

32. **register_fused_op with kernel** (04_handwritten_kernels.md, 5.4.2): MicroOps pass `kernel=None` while FusedOps pass `cls.kernel` verified against `blaze_op.py` lines 326-338. Correct.

33. **Mcast leader/follower codegen** (02_auto_generated_kernels.md, 5.2.5): Leader declared outside scope, followers use `setup_src<>()` and `run_as<>()`, deferred teardown verified against `kernel_codegen.py` lines 331-368. Correct.

34. **Noop kernel emission** (02_auto_generated_kernels.md, 5.2.4): Empty `DeviceZoneScopedN` block with `_NOOP` suffix verified against `kernel_codegen.py` lines 312-317. Correct.

35. **_tag_noop_nodes function** (01_engine_pipeline.md, 5.1.2): Verified against `compiler.py` lines 49-53. Correctly sets `noop_kernel=True` on matching nodes.

---

## Verdict

**Chapter requires one fix before approval.**

The NCRISC/BRISC DM0/DM1 processor numbering inversion (Issue 1) is a critical factual error that could lead to debugging confusion. All other issues are minor inaccuracies, cosmetic differences, or omissions that do not compromise the chapter's educational value.

**Required**: Fix Issue 1 (swap DM0/DM1 labels in Section 5.3.4 table).
**Recommended**: Address Issues 2, 4, 9 to align pseudocode more closely with actual implementations.
**Optional**: Add brief mentions of PerCorePositionalCTADescriptor and PerCoreRuntimeArgsDescriptor (Issues 7, 8) for completeness.
