# Agent B Review: Chapter 2 -- Anatomy of a Micro-Op

## Pass 1

### Issue 1: `register_all()` import path differs from actual code
**File:** `01_directory_structure.md`, Section 2.1.2
**Claim:** The simplified code shows `importlib.import_module(f".{modname}", __package__)` importing the sub-package directly.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/__init__.py`, line 33: `mod = importlib.import_module(f".{info.name}.op", __package__)` -- the actual code imports `.<name>.op` (the op module directly), not the sub-package.
**Impact:** A reader following the chapter's code to understand the import chain would look in `__init__.py` for the class, when the actual discovery goes straight to `op.py`. This also means the chapter's "step 3" claim that importing "Triggers `__init_subclass__` on any MicroOp subclass" is technically misleading because the import targets `op.py`, not the package `__init__.py`.

### Issue 2: `_OP_REGISTRY` key type is wrong
**File:** `01_directory_structure.md`, Section 2.1.3 (Five global registries table)
**Claim:** `_OP_REGISTRY` has key type `OpSpec` and value type `OpSpec` -- "Maps op identity (name, ports, is_inter_core) to itself for lookup."
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/registry.py`, line 12: `_OP_REGISTRY: dict[str, OpSpec] = {}` -- keyed by `str` (the op name), not `OpSpec`.
**What needs to change:** Table should show Key = `str` (name), Value = `OpSpec`.

### Issue 3: CT schema registry is named `_CT_ARG_SCHEMAS`, not `_CT_SCHEMAS`
**File:** `01_directory_structure.md`, Section 2.1.3 (Five global registries table)
**Claim:** Registry named `_CT_SCHEMAS` in module `blaze.ct_schema`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ct_args.py`, line 93: `_CT_ARG_SCHEMAS: dict[str, CTArgSchema] = {}` -- the variable is `_CT_ARG_SCHEMAS` in module `blaze.ct_args`, not `_CT_SCHEMAS` in `blaze.ct_schema`.
**What needs to change:** Fix both the registry name and module name.

### Issue 4: PhaseInfo registry is named `_PHASE_REGISTRY`, not "PhaseInfo registry"
**File:** `01_directory_structure.md`, Section 2.1.3 (Five global registries table)
**Claim:** Registry named "PhaseInfo registry" in `blaze.kernel_codegen`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/kernel_codegen.py`, line 58: `_PHASE_REGISTRY: dict[str, PhaseInfo] = {}`.
**What needs to change:** Use the actual variable name `_PHASE_REGISTRY` for consistency with how the other registries are presented.

### Issue 5: `compose()` signature is wrong
**File:** `02_python_class.md`, Section 2.2.4
**Claim:** The signature is `compose(cls, f: FusedProgram, tensors: dict[str, Tensor], **kwargs) -> Tensor | tuple[Tensor, ...]`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/blaze_op.py`, line 234: `def compose(cls, f, tensors, output, user_args)` -- the actual signature has `output` and `user_args` as separate positional parameters, not `**kwargs`. All real ops (RMSNorm, Copy, Mcast, Matmul) use this four-parameter signature.
**Impact:** This is a significant technical inaccuracy. A reader implementing a new op from the chapter's examples would produce non-functional code because the compose signature does not match what the framework dispatches.

### Issue 6: `emit()` is `@staticmethod`, not `@classmethod`
**File:** `02_python_class.md`, Sections 2.2.5, 2.2.6, 2.2.7
**Claim:** All `emit()` examples use `@classmethod` with `cls` as the first parameter.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/blaze_op.py`, line 218: `@staticmethod def emit(f, *args, **kwargs)`. All concrete ops (RMSNorm line 54, Copy line 59, Mcast line 26) use `@staticmethod`.
**Impact:** A reader would write `@classmethod` and reference `cls` in their emit, which would break when the framework calls `emit()` as a static method. The `__init_subclass__` check `cls.emit is BlazeOp.emit` works with static methods, and the walkthroughs show `cls.name`, `cls.output`, etc. which would not be available in a static method without explicit class reference.

### Issue 7: RMSNorm has no `Internal` ports (`im0`, `im1`)
**File:** `02_python_class.md`, Sections 2.2.2, 2.2.6; `01_directory_structure.md`, Section 2.1.1
**Claim:** RMSNorm has `im0: Internal = Internal(num_pages=1)` and `im1: Internal = Internal(num_pages=2)`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/op.py`, lines 27-29: Only `input: Input`, `gamma: Input`, and `output: Output` are declared. No Internal ports.
**Impact:** The entire walkthrough of RMSNorm's internal CB allocation is fabricated. The real RMSNorm does not allocate internal CBs via `f.internal_cb()`.

### Issue 8: Copy port names are `src`/`dst`, not `input`/`output`
**File:** `02_python_class.md`, Section 2.2.7; `01_directory_structure.md`, Section 2.1.1
**Claim:** Copy has `input: Input = Input()` and `output: Output = Output()`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/copy/op.py`, lines 38-39: `src: Input = Input()` and `dst: Output = Output()`.
**What needs to change:** All references to Copy's ports should use `src`/`dst`.

### Issue 9: RMSNorm `math_fidelity` is `"LoFi"`, not `"HiFi4"`
**File:** `02_python_class.md`, Sections 2.2.3, 2.2.6
**Claim:** `math_fidelity: str = "HiFi4"` and `math_approx_mode: bool = True` and `fp32_dest_acc_en: bool = True`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/op.py`, line 25: `math_fidelity: str = "LoFi"`. `math_approx_mode` is not overridden (defaults to `False` from BlazeOp). `fp32_dest_acc_en` on line 32 is `False`.
**What needs to change:** Fix the config values in the class example and contrast table.

---

## Pass 2

### Issue 10: RMSNorm has THREE CT arg structs, not two
**File:** `01_directory_structure.md`, Section 2.1.1; `03_cpp_kernel_header.md`, Section 2.3.8
**Claim:** "RMSNorm uses two CT arg structs (reader + compute)" and "CT arg struct count: 2 (Reader, Compute)".
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp`, lines 37-60: The header contains `CoreCTArgs`, `ReaderCTArgs`, and `ComputeCTArgs` -- three structs.
**Impact:** The parser sees CoreCTArgs and sets `has_brisc = True`, `has_ncrisc = True`, `has_trisc = True`. Combined with ReaderCTArgs and ComputeCTArgs, the final auto-derived `is_inter_core` would be `True` (has_brisc AND has_ncrisc). The chapter claims `is_inter_core = False` for RMSNorm. This is wrong per parser logic, though RMSNorm may override it on the class. Let me note: without CoreCTArgs the parser would NOT set `has_brisc = True`, so is_inter_core would be False. But with CoreCTArgs present, is_inter_core should be True.

### Issue 11: Mcast has THREE CT arg structs, not four
**File:** `01_directory_structure.md`, Section 2.1.1; `03_cpp_kernel_header.md`, Section 2.3.9
**Claim:** "Mcast uses all four [struct types]" and the walkthrough shows CoreCTArgs, ReaderCTArgs, WriterCTArgs, and ComputeCTArgs.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/mcast/kernels/op.hpp`: Only `CoreCTArgs`, `ReaderCTArgs`, and `WriterCTArgs` are present. There is no `ComputeCTArgs`. Verified by regex: `_INNER_STRUCT_RE.findall(text)` returns `['CoreCTArgs', 'ReaderCTArgs', 'WriterCTArgs']`.
**Impact:** The entire Mcast walkthrough in Section 2.3.9 is fabricated for ComputeCTArgs. The Mcast merged CT args, the comparison tables, and the is_inter_core reasoning all need correction.

### Issue 12: Mcast `CoreCTArgs` does not have `is_active`, `PerCore num_dests`, or `Semaphore` fields
**File:** `03_cpp_kernel_header.md`, Section 2.3.9; `04_auto_derivation.md`, Section 2.4.6
**Claim:** Mcast's CoreCTArgs contains `Flag is_active`, `PerCore num_dests`, `Semaphore sender_semaphore`, `Semaphore receiver_semaphore`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/mcast/kernels/op.hpp`, lines 174-179: CoreCTArgs has only `Flag is_sender`, `Flag is_receiver`, `Flag pop_src`, `Flag init_src`. No `is_active`, no `num_dests`, no semaphore fields.
**What needs to change:** The Mcast walkthrough needs to be rewritten to reflect the actual struct contents.

### Issue 13: Mcast `setup_src()` is NOT a static method and NOT detected by `_SETUP_RE`
**File:** `03_cpp_kernel_header.md`, Section 2.3.9; `04_auto_derivation.md`, Section 2.4.6
**Claim:** `setup_method="setup_src"` in parsed output. Section 2.3.9 says setup methods are detected by `_SETUP_RE`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/mcast/kernels/op.hpp`, line 303-304: `setup_src()` is declared as `template <typename A> void setup_src()` -- a non-static template method. The `_SETUP_RE` regex is `static\s+void\s+(setup_\w+)\s*\(\s*\)` which does NOT match this. Verified: `_SETUP_RE.search(mcast_text)` returns `None`.
**Impact:** The Mcast auto-derivation trace in Section 2.4.6 claims `setup_method="setup_src"` but the parser would actually produce `setup_method=None`.

### Issue 14: `_OUTER_STRUCT_RE` does NOT match indented struct declarations
**File:** `03_cpp_kernel_header.md`, Section 2.3.7 (regex table); implied in Section 2.3.8
**Claim:** `_OUTER_STRUCT_RE` matches `struct RMSNorm {`.
**Actual source:** The regex is `^struct\s+(\w+(?<!CTArgs))\s*\{` with `re.MULTILINE`. RMSNorm's struct at `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp` line 33 is indented: `    struct RMSNorm {`. The `^` anchor requires the line to start with `struct`, so this does NOT match. Verified: `_OUTER_STRUCT_RE.search(rmsnorm_text)` returns `None`. Other ops like Mcast, Copy, and Matmul have their outer struct at column 0 and DO match.
**Impact:** For RMSNorm, `parsed.struct_name` would be empty. The `if parsed.struct_name:` guard prevents `cls.op_class` from being overwritten. This means the RMSNorm auto-derivation trace in Section 2.1.3 and the claim that "auto-derivation unconditionally overwrites op_class" is wrong for RMSNorm specifically -- it only overwrites when the regex matches.

---

## Pass 3

### Issue 15: The `emit()` pattern uses `prefix` parameter, not `f.unique_op_prefix(cls.name)`
**File:** `02_python_class.md`, Section 2.2.5; `02_python_class.md`, Section 2.2.6
**Claim:** Every emit method starts with `prefix = f.unique_op_prefix(cls.name)`.
**Actual source:** All actual emit methods take `prefix` as a keyword argument (e.g., RMSNorm: `prefix: str = "rmsnorm"`, Copy: `prefix: str = "copy"`, Mcast: `prefix: str`). There is no `unique_op_prefix()` method on FusedProgram. The prefix is passed by the caller, not generated inside emit.
**Impact:** Misleading for developers implementing new ops. The real pattern is to accept `prefix` as a parameter.

### Issue 16: CT arg prefix uses `___` (CB_NAME_DELIMITER), not tied to `unique_op_prefix`
**File:** `02_python_class.md`, Section 2.2.5
**Claim:** `prefix = "rmsnorm___0"` (name + "___" + instance counter) created by `f.unique_op_prefix()`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/blaze_op.py`, line 196: `CB_NAME_DELIMITER = "___"` is used in `cb_name()` (line 231) for CB naming, not for CT arg prefixes. CT arg names in emit use the `prefix` parameter directly (e.g., `f"{prefix}.input"`). The triple-underscore is for CB names via `BlazeOp.cb_name(prefix, "out_cb")`, not for CT arg prefixes.
**What needs to change:** Clarify that `___` is used by `cb_name()` for CB naming, while CT arg names use the `prefix` parameter directly with a dot separator.

### Issue 17: Copy uses `f.unified_ct_args()`, not `f.all_core_ct_args()`
**File:** `02_python_class.md`, Sections 2.2.5 (table), 2.2.7, 2.2.8
**Claim:** Copy uses `f.all_core_ct_args()`. The method table lists `f.all_core_ct_args([(name, val), ...])` as the method for all-RISC args.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/copy/op.py`, line 129: `f.unified_ct_args([...])`. The FusedProgram class at `/localdev/salnahari/testing_dir/tt-blaze/blaze/fused_program.py` has `unified_ct_args` (line 793) but NO `all_core_ct_args` method. The method does not exist.
**Impact:** A developer writing `f.all_core_ct_args(...)` would get an `AttributeError` at runtime. The correct method name is `f.unified_ct_args()`.

### Issue 18: The CT arg registration methods table has a non-existent method name
**File:** `02_python_class.md`, Section 2.2.5 (CT arg registration methods table)
**Claim:** Lists `f.all_core_ct_args()` and `f.per_core_unified_ct_args()` as methods.
**Actual source:** The actual methods in FusedProgram are: `unified_ct_args` (line 793), `ncrisc_ct_args` (line 798), `brisc_ct_args` (line 803), `trisc_ct_args` (line 808), `per_core_unified_ct_args` (line 1610), `per_core_ncrisc_ct_args` (line 1614), `per_core_trisc_ct_args` (line 1618), `per_core_brisc_ct_args` (line 1622).
**What needs to change:** Replace `all_core_ct_args` with `unified_ct_args` in the table and all references.

### Issue 19: RMSNorm walkthrough init() code does not match actual source
**File:** `03_cpp_kernel_header.md`, Section 2.3.8
**Claim:** RMSNorm init() does `noc_async_read_one_packet_set_state(get_noc_id(), ...);` under COMPILE_FOR_NCRISC.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp`, lines 72-81: The actual init() under COMPILE_FOR_NCRISC does `cb_reserve_back`/`cb_push_back` for tensor-backed inputs and gamma. No `noc_async_read_one_packet_set_state`.
**What needs to change:** Update the init() walkthrough to reflect the actual code.

### Issue 20: RMSNorm `teardown()` IS empty -- chapter says it is not
**File:** `03_cpp_kernel_header.md`, Section 2.3.8; `04_auto_derivation.md`, Section 2.4.6 (PhaseInfo comparison table)
**Claim:** `teardown_is_empty = False` for RMSNorm. The walkthrough shows `teardown()` containing `noc_async_read_barrier();`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp`, line 161: `void teardown() {}` -- the teardown body is completely empty.
**What needs to change:** Fix `teardown_is_empty = True` for RMSNorm and remove the fabricated teardown code from the walkthrough.

### Issue 21: RMSNorm `operator()()` code does not match actual source
**File:** `03_cpp_kernel_header.md`, Section 2.3.8
**Claim:** The walkthrough shows `operator()()` with an NCRISC section doing `noc_async_read_one_packet_with_state(...)` and a TRISC section doing simplified `rmsnorm_compute(...)`.
**Actual source:** The actual NCRISC section has NO code in `operator()()` -- only TRISC code under `COMPILE_FOR_TRISC` (lines 84-159). The actual compute logic is much more complex (mul_reduce_scalar, add_rsqrt_tile, rmsnorm_mul_bcast_scalar_reuse_tiles, binary_dest_reuse_tiles, etc.). The chapter's simplified walkthrough invents an NCRISC section that does not exist and oversimplifies TRISC.
**What needs to change:** Clarify that the RMSNorm walkthrough is a simplified/pedagogical illustration or update it to reflect the actual structure (TRISC-only compute in operator()).

### Issue 22: RMSNorm emit walkthrough fabricates the structure
**File:** `02_python_class.md`, Section 2.2.6
**Claim:** The emit walkthrough shows `f.allocate_cb(cls.output, ...)`, `f.internal_cb(cls.im0, ...)`, `f.set_math_fidelity(cls.math_fidelity)`, and referencing `self.input`.
**Actual source:** `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/op.py`: emit is a `@staticmethod` (no `cls`), uses `f.cb_scratch(...)` for output (not `f.allocate_cb`), has no internal CB allocation, uses `f.per_core_unified_ct_args` with `f.flag()` for per-core flags, and does not call `set_math_fidelity/set_math_approx_mode/set_fp32_dest_acc_en`.
**What needs to change:** The emit walkthrough should be either corrected to match the actual code or clearly marked as pedagogical pseudocode.

---

## Pass 4

Summarizing the most critical findings that need correction before this chapter can serve as a reliable reference:

**Critical (would cause broken code if followed literally):**

1. **Issue 5:** `compose()` signature is `(cls, f, tensors, output, user_args)`, not `(cls, f, tensors, **kwargs)`.
2. **Issue 6:** `emit()` is `@staticmethod`, not `@classmethod`.
3. **Issue 17/18:** `f.all_core_ct_args()` does not exist. The correct method is `f.unified_ct_args()`.

**Significant factual errors (would mislead understanding):**

4. **Issue 10:** RMSNorm has 3 CT arg structs (Core + Reader + Compute), not 2.
5. **Issue 11:** Mcast has 3 CT arg structs (Core + Reader + Writer), not 4. No ComputeCTArgs.
6. **Issue 12:** Mcast CoreCTArgs contents are fabricated (actual: is_sender, is_receiver, pop_src, init_src).
7. **Issue 13:** Mcast setup_src is not static and NOT detected by _SETUP_RE.
8. **Issue 7:** RMSNorm has no Internal ports (im0, im1).
9. **Issue 8:** Copy ports are src/dst, not input/output.
10. **Issue 9:** RMSNorm math_fidelity is "LoFi", not "HiFi4"; other config values are also wrong.
11. **Issue 14:** _OUTER_STRUCT_RE fails on indented structs (like RMSNorm's).
12. **Issue 20:** RMSNorm teardown() is empty, not non-empty.

**Registry/naming errors:**

13. **Issue 2:** _OP_REGISTRY is keyed by str, not OpSpec.
14. **Issue 3:** Registry is _CT_ARG_SCHEMAS in blaze.ct_args, not _CT_SCHEMAS in blaze.ct_schema.

**Structural/pattern errors:**

15. **Issue 1:** register_all() imports `.{name}.op`, not `.{modname}`.
16. **Issue 15/16:** emit() takes prefix as a parameter; unique_op_prefix() does not exist.

No feedback beyond these findings -- chapter approved once these are addressed.

---

## Pass 3 -- Fixes Applied (Agent A)

All 22 confirmed issues have been fixed across the four content files. Below is a per-issue summary of changes.

### Critical fixes (would have produced broken code):

**Issue 5 (compose() signature):**
- `02_python_class.md`: Updated compose() signature from `(cls, f, tensors, **kwargs)` to `(cls, f, tensors, output, user_args)`. Updated parameter table to include `output` and `user_args`. Rewrote all three compose() walkthroughs (RMSNorm, Copy, Matmul) to use the correct signature and show `user_args.get(...)` pattern.

**Issue 6 (emit() is @staticmethod):**
- `02_python_class.md`: Changed all emit() declarations from `@classmethod` to `@staticmethod`. Removed `cls` parameter from emit signatures. Updated the general pattern section and both structural walkthroughs (RMSNorm, Copy). Updated key takeaways.

**Issue 17/18 (f.all_core_ct_args() does not exist):**
- `02_python_class.md`: Replaced `f.all_core_ct_args()` with `f.unified_ct_args()` in the CT arg registration methods table, the Copy walkthrough, the contrast table, and the key takeaways. Added explicit note that `f.all_core_ct_args()` does not exist.
- `03_cpp_kernel_header.md`: Updated the struct-to-Python-method mapping table from `f.all_core_ct_args()` to `f.unified_ct_args()` / `f.per_core_unified_ct_args()`.

### Significant factual error fixes:

**Issue 7 (RMSNorm has no Internal ports):**
- `01_directory_structure.md`: Updated RMSNorm directory listing to say "no Internal" ports. Updated OpSpec trace.
- `02_python_class.md`: Removed `im0` and `im1` from RMSNorm port declaration. Added note about scratch CBs via `f.cb_scratch()`. Updated error message example, OpSpec example, contrast table. Removed all `f.internal_cb()` references from RMSNorm walkthrough.
- `03_cpp_kernel_header.md`: Removed `im0`/`im1` from CT arg struct declarations. Updated merged CT args list.

**Issue 8 (Copy ports are src/dst):**
- `01_directory_structure.md`: Updated Copy directory listing to show `src (Input), dst (Output)`.
- `02_python_class.md`: Rewrote Copy walkthrough to use `src`/`dst` port names. Updated compose() walkthrough to use `tensors["src"]`.

**Issue 9 (RMSNorm math_fidelity is "LoFi"):**
- `02_python_class.md`: Changed RMSNorm config example from `math_fidelity = "HiFi4"` to `math_fidelity = "LoFi"`. Removed fabricated `math_approx_mode = True` and `fp32_dest_acc_en = True`. Added note clarifying that fp32_dest_acc_en is a runtime parameter. Updated contrast table.

**Issue 10 (RMSNorm has 3 CT arg structs):**
- `01_directory_structure.md`: Updated from "two CT arg structs (reader + compute)" to "three CT arg structs (core + reader + compute)". Updated RMSNorm registration trace.
- `02_python_class.md`: Updated contrast table from "ReaderCTArgs + ComputeCTArgs" to "CoreCTArgs + ReaderCTArgs + ComputeCTArgs".
- `03_cpp_kernel_header.md`: Added CoreCTArgs to the RMSNorm walkthrough. Updated parser output and merged CT args.

**Issue 11 (Mcast has 3 CT arg structs, not 4):**
- `01_directory_structure.md`: Updated from "four struct types" to "three (core + reader + writer)". Added explicit note about no ComputeCTArgs.
- `03_cpp_kernel_header.md`: Rewrote entire Mcast walkthrough to show only CoreCTArgs, ReaderCTArgs, WriterCTArgs. Removed fabricated ComputeCTArgs. Updated parser output and structural observations.
- `04_auto_derivation.md`: Updated Mcast trace table.

**Issue 12 (Mcast CoreCTArgs contents fabricated):**
- `03_cpp_kernel_header.md`: Replaced fabricated fields (is_active, PerCore num_dests, Semaphore sender_semaphore, Semaphore receiver_semaphore) with actual fields (Flag is_sender, Flag is_receiver, Flag pop_src, Flag init_src). Added note that semaphores are in ReaderCTArgs/WriterCTArgs, not CoreCTArgs.

**Issue 13 (Mcast setup_src() is NOT static):**
- `03_cpp_kernel_header.md`: Rewrote setup methods section to explain that `setup_src()` is a template member method, not static. Added explicit note about `_SETUP_RE` not matching. Changed from `static void setup_src()` to `template <typename A> void setup_src()`.
- `04_auto_derivation.md`: Updated Mcast trace: `setup_method=None` (not `"setup_src"`). Added explanation of why. Updated PhaseInfo comparison table.

**Issue 14 (_OUTER_STRUCT_RE fails on indented structs):**
- `01_directory_structure.md`: Updated RMSNorm registration trace to explain `struct_name = ""` and why op_class is NOT overwritten by auto-derivation for RMSNorm.
- `03_cpp_kernel_header.md`: Added note about indented struct in RMSNorm CT arg section. Updated parser output to show `struct_name = ""`.
- `04_auto_derivation.md`: Added new subsection "Parser limitation: _OUTER_STRUCT_RE and indented structs" with detailed explanation. Updated `struct_name` description in ParsedKernel table. Updated op_class assignment guard description. Updated key takeaway #5. Updated override scenario 1.

**Issue 20 (RMSNorm teardown() IS empty):**
- `03_cpp_kernel_header.md`: Changed RMSNorm teardown to `void teardown() {}`. Updated lifecycle detection to show `teardown_is_empty = True`.
- `04_auto_derivation.md`: Updated PhaseInfo comparison table to show `teardown_is_empty = True` for RMSNorm.

### Registry/naming error fixes:

**Issue 2 (_OP_REGISTRY keyed by str):**
- `01_directory_structure.md`: Updated five-registry table: Key = `str` (name), module = `blaze.registry`. Updated key takeaway.

**Issue 3 (CT schema registry):**
- `01_directory_structure.md`: Changed `_CT_SCHEMAS` to `_CT_ARG_SCHEMAS`, module from `blaze.ct_schema` to `blaze.ct_args`. Updated key takeaway.

**Issue 4 (PhaseInfo registry):**
- `01_directory_structure.md`: Changed "PhaseInfo registry" to `_PHASE_REGISTRY` in `blaze.kernel_codegen`. Updated key takeaway.

### Structural/pattern error fixes:

**Issue 1 (register_all() import path):**
- `01_directory_structure.md`: Rewrote register_all() code to match actual source: `f".{info.name}.op"` import path, `BlazeOp` base class check (not `MicroOp`), `obj.name` check, `info` variable name. Updated step table. Added note about import path targeting `op.py` directly.

**Issue 15 (prefix is a parameter):**
- `02_python_class.md`: Updated general emit() pattern to show `prefix: str = "opname"` as parameter. Removed `f.unique_op_prefix()`. Updated all walkthroughs. Updated key takeaways.

**Issue 16 (CB_NAME_DELIMITER vs CT arg naming):**
- `02_python_class.md`: Rewrote the "CT arg prefix and BlazeOp.cb_name()" section to clearly distinguish: CT arg names use dot separator (`prefix.field`), CB names use triple-underscore via `BlazeOp.cb_name()` (`prefix___cbname`).

### Walkthrough accuracy fixes:

**Issue 19 (RMSNorm init() code):**
- `03_cpp_kernel_header.md`: Replaced fabricated `noc_async_read_one_packet_set_state` with actual `cb_reserve_back`/`cb_push_back` for tensor-backed inputs. Updated the generic init/operator/teardown interaction example.

**Issue 21 (RMSNorm operator() code):**
- `03_cpp_kernel_header.md`: Removed fabricated NCRISC section from operator(). Corrected to show ONLY `COMPILE_FOR_TRISC` block. Added explicit observation about this difference.

**Issue 22 (RMSNorm emit walkthrough):**
- `02_python_class.md`: Rewrote the entire RMSNorm emit walkthrough to match actual source: `@staticmethod`, `prefix` parameter, `f.cb_from_tensor()`, `f.cb_scratch()`, `f.per_core_unified_ct_args([f.flag(...)])`, no `f.internal_cb()`, no `f.set_math_fidelity()`, `f.output()` return path.

### Additional corrections (found during fix pass):

- `02_python_class.md`: Updated `eps_bits` to `epsilon` in the float-to-bits conversion example to match actual RMSNorm field name.
- `02_python_class.md`: Updated CB allocation methods table to use actual methods (`f.cb_from_tensor()`, `f.cb_scratch()`, `f.cb_output()`) instead of fabricated ones.
- `03_cpp_kernel_header.md`: Updated generic preprocessor guard example to use patterns from actual Copy op code rather than fabricated function calls.
- `01_directory_structure.md`: Updated `_class_registry` module from `blaze.op_registry` to `blaze.blaze_op` (it lives on the `BlazeOp` class). Updated `_FUSED_OP_CONFIGS` module from `blaze.fused_op` to `blaze.blaze_op`.

---

## Pass 4 -- Verification of Pass 3 Fixes (Agent B)

### Verification method

Each of the 22 original issues was verified by:
1. Reading the fixed chapter text in the four content files.
2. Spot-checking critical claims against the actual TT-Blaze source code at `/localdev/salnahari/testing_dir/tt-blaze/`.

### Per-issue verification

**Issue 1 (register_all() import path):** FIXED CORRECTLY. `01_directory_structure.md` line 107 now shows `f".{info.name}.op"`. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/__init__.py` line 32: `mod = importlib.import_module(f".{info.name}.op", __package__)`. The base class check (`BlazeOp`) and variable name (`info`) also match. The explanatory note about targeting `op.py` directly is accurate.

**Issue 2 (_OP_REGISTRY key type):** FIXED CORRECTLY. `01_directory_structure.md` line 184 shows Key = `str` (name), Value = `OpSpec`. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/registry.py` line 12: `_OP_REGISTRY: dict[str, OpSpec] = {}`.

**Issue 3 (_CT_ARG_SCHEMAS registry name and module):** FIXED CORRECTLY. `01_directory_structure.md` line 186 shows `_CT_ARG_SCHEMAS` in `blaze.ct_args`. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ct_args.py` line 93: `_CT_ARG_SCHEMAS: dict[str, CTArgSchema] = {}`.

**Issue 4 (_PHASE_REGISTRY variable name):** FIXED CORRECTLY. `01_directory_structure.md` line 187 shows `_PHASE_REGISTRY` in `blaze.kernel_codegen`. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/kernel_codegen.py` line 58: `_PHASE_REGISTRY: dict[str, PhaseInfo] = {}`.

**Issue 5 (compose() signature):** FIXED CORRECTLY. All compose() signatures in `02_python_class.md` now use `(cls, f, tensors, output, user_args)`. Verified at lines 151, 165, 184, 205, 341, 432. The parameter table (lines 155-159) correctly lists `output` and `user_args` as separate parameters. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/blaze_op.py` line 234: `def compose(cls, f, tensors, output, user_args)`. RMSNorm compose (source line 37), Copy compose (source line 42), and Mcast compose (source line 21) all match.

**Issue 6 (emit() is @staticmethod):** FIXED CORRECTLY. All emit() declarations in `02_python_class.md` now use `@staticmethod` without `cls`. Verified at lines 227, 352, 445. The general pattern (line 223-253), RMSNorm walkthrough (line 352), and Copy walkthrough (line 445) all use `@staticmethod`. Verified against source: RMSNorm line 54, Copy line 59, Mcast line 25 all use `@staticmethod`.

**Issue 7 (RMSNorm has no Internal ports):** FIXED CORRECTLY. `02_python_class.md` lines 55-60 show RMSNorm with only `input`, `gamma`, and `output` ports. Line 63 explicitly states "RMSNorm has no `Internal` ports." The emit walkthrough (lines 352-401) uses `f.cb_scratch()` instead of `f.internal_cb()`. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/op.py` lines 27-29: only `input: Input`, `gamma: Input`, `output: Output`.

**Issue 8 (Copy ports are src/dst):** FIXED CORRECTLY. `02_python_class.md` lines 428-429 show `src: Input` and `dst: Output`. The compose walkthrough (line 437) uses `tensors["src"]`. `01_directory_structure.md` line 46 shows "src (Input), dst (Output) ports". Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/copy/op.py` lines 38-39: `src: Input = Input()`, `dst: Output = Output()`.

**Issue 9 (RMSNorm math_fidelity is "LoFi"):** FIXED CORRECTLY. `02_python_class.md` line 133 shows `math_fidelity: str = "LoFi"`. Lines 137-138 correctly note that `fp32_dest_acc_en` is `False` at the class level and passed as a runtime parameter. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/op.py` line 25: `math_fidelity: str = "LoFi"`, line 32: `fp32_dest_acc_en: bool = False`.

**Issue 10 (RMSNorm has 3 CT arg structs):** FIXED CORRECTLY. `01_directory_structure.md` line 31 says "core, reader, and compute CT args". `03_cpp_kernel_header.md` lines 466-495 show all three structs (CoreCTArgs, ReaderCTArgs, ComputeCTArgs). `02_python_class.md` line 511 shows "CoreCTArgs + ReaderCTArgs + ComputeCTArgs". Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp` lines 37-60: all three structs present.

**Issue 11 (Mcast has 3 CT arg structs, not 4):** FIXED CORRECTLY. `01_directory_structure.md` line 58 says "CoreCTArgs, ReaderCTArgs, WriterCTArgs". `03_cpp_kernel_header.md` line 584 says "three CT arg structs (core, reader, writer -- NO ComputeCTArgs)". Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/mcast/kernels/op.hpp`: only CoreCTArgs (lines 174-179), ReaderCTArgs (lines 182-189), WriterCTArgs (lines 192-209). No ComputeCTArgs.

**Issue 12 (Mcast CoreCTArgs contents):** FIXED CORRECTLY. `03_cpp_kernel_header.md` lines 594-599 show `is_sender`, `is_receiver`, `pop_src`, `init_src` -- all Flag types. Line 646 confirms "There is NO `is_active` flag in Mcast's `CoreCTArgs`" and semaphores live in Reader/WriterCTArgs. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/mcast/kernels/op.hpp` lines 175-178: exactly four Flag fields matching.

**Issue 13 (Mcast setup_src() is NOT static):** FIXED CORRECTLY. `03_cpp_kernel_header.md` lines 726-741 explain that `setup_src()` is a template member method, not static. Line 739 explicitly states "_SETUP_RE does NOT detect Mcast's template setup_src()". `04_auto_derivation.md` line 289 shows `setup_method=None` for Mcast. Verified against source: `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/mcast/kernels/op.hpp` lines 303-304: `template <typename A> void setup_src()`. The `_SETUP_RE` at `/localdev/salnahari/testing_dir/tt-blaze/blaze/cpp_parser.py` line 87 is `static\s+void\s+(setup_\w+)\s*\(\s*\)` -- does not match.

**Issue 14 (_OUTER_STRUCT_RE and indented structs):** FIXED CORRECTLY. `03_cpp_kernel_header.md` lines 497-500 explain the indentation issue. Line 500 shows `struct_name = ""`. `04_auto_derivation.md` lines 169-188 contain a dedicated subsection explaining this limitation. `01_directory_structure.md` lines 200-201 explain why `op_class` is NOT overwritten for RMSNorm. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp` line 33: `    struct RMSNorm {` (indented). Regex at `/localdev/salnahari/testing_dir/tt-blaze/blaze/cpp_parser.py` line 91 uses `^struct` with `re.MULTILINE`.

**Issue 15 (emit() prefix is a parameter):** FIXED CORRECTLY. `02_python_class.md` line 256 states "prefix is a parameter, not generated by a f.unique_op_prefix() method". All emit examples show `prefix: str = "opname"` as a keyword argument. No mention of `unique_op_prefix()` remains. Verified against source: RMSNorm `prefix: str = "rmsnorm"` (line 60), Copy `prefix: str = "copy"` (line 65).

**Issue 16 (CB_NAME_DELIMITER vs CT arg prefixes):** FIXED CORRECTLY. `02_python_class.md` lines 282-301 clearly separate the two naming patterns: CT arg names use dot separator (`prefix.field`), CB names use triple-underscore (`prefix___cbname`) via `BlazeOp.cb_name()`. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/blaze_op.py` line 196: `CB_NAME_DELIMITER: str = "___"`, line 231: `return f"{prefix}{cls.CB_NAME_DELIMITER}{name}"`.

**Issue 17 (f.all_core_ct_args() does not exist):** FIXED CORRECTLY. `02_python_class.md` line 278 explicitly states "There is no `f.all_core_ct_args()` method." Line 275 shows `f.unified_ct_args()` as the correct method. Copy walkthrough (line 467) uses `f.unified_ct_args()`. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/fused_program.py` line 793: `def unified_ct_args(self, args, **kwargs)`. No `all_core_ct_args` method exists in FusedProgram.

**Issue 18 (CT arg registration methods table):** FIXED CORRECTLY. The table at lines 270-277 shows `f.unified_ct_args()` and `f.per_core_unified_ct_args()`. No `f.all_core_ct_args()` remains. Verified against FusedProgram methods at lines 793-812 (unified, ncrisc, brisc, trisc) and 1610-1624 (per_core variants).

**Issue 19 (RMSNorm init() code):** FIXED CORRECTLY. `03_cpp_kernel_header.md` lines 534-543 show `cb_reserve_back`/`cb_push_back` for tensor-backed inputs, matching the actual source. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp` lines 72-81: same pattern with `input_is_tensor_backed` guard, `cb_reserve_back`, `cb_push_back`.

**Issue 20 (RMSNorm teardown() is empty):** FIXED CORRECTLY. `03_cpp_kernel_header.md` line 563 shows `void teardown() {}`. Line 572 states `teardown_is_empty = True`. `04_auto_derivation.md` line 308 shows `teardown_is_empty = True` for RMSNorm. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp` line 161: `void teardown() {}`.

**Issue 21 (RMSNorm operator()() is TRISC-only):** FIXED CORRECTLY. `03_cpp_kernel_header.md` lines 546-558 show `operator()()` with ONLY a `COMPILE_FOR_TRISC` block. Line 575 explicitly states "operator()() has ONLY a COMPILE_FOR_TRISC block. There is NO NCRISC section." Verified against source lines 83-158: entire `operator()()` is wrapped in `#if defined(COMPILE_FOR_TRISC)`.

**Issue 22 (RMSNorm emit walkthrough):** FIXED CORRECTLY. `02_python_class.md` lines 352-401 show a rewritten emit walkthrough using `@staticmethod`, `prefix` parameter, `f.cb_from_tensor()`, `f.cb_scratch()`, `f.per_core_unified_ct_args([f.flag(...)])`, `f.ncrisc_ct_args()`, `f.trisc_ct_args()`, and `f.output()`. No `f.internal_cb()`, no `f.set_math_fidelity()`, no `cls` references. Structural observations (lines 406-412) are accurate. Verified against `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/op.py` lines 54-180.

### Remaining issues

**Issue A (minor): Mcast port names listed as "Input, Output" instead of "src, dst" in directory listing.**
**File:** `01_directory_structure.md`, line 56.
**Text:** `op.py                # class Mcast(MicroOp): Input, Output ports`
**Problem:** The Copy entry (line 46) was correctly updated to say "src (Input), dst (Output) ports", but the Mcast entry still says "Input, Output ports" without specifying the actual attribute names. The actual Mcast class uses `src: Input` and `dst: Output` (verified at `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/mcast/op.py` lines 17-18). For consistency with the Copy entry and to prevent confusion, Mcast should say "src (Input), dst (Output) ports".
**Severity:** Low. The descriptor types are correct; only the attribute names are missing.

**Issue B (minor): Key takeaway #4 in `01_directory_structure.md` still says "unconditionally" for op_class.**
**File:** `01_directory_structure.md`, line 276.
**Text:** "Auto-derivation unconditionally sets `op_class` from the header struct name."
**Problem:** This contradicts the fix for Issue 14 in the same file (line 201) which correctly explains that `op_class` is NOT overwritten when `parsed.struct_name` is empty (as with RMSNorm's indented struct). The key takeaway should be qualified to say auto-derivation sets `op_class` when the parser detects a struct name at column 0. Section 2.1.4 (line 237) correctly says "If `_auto_derive_from_kernel_hpp()` runs and finds a `struct_name`...", and Section 2.4.2 (line 105) and key takeaway #5 in Section 2.4 (line 592) are also correct. Only this one key takeaway sentence in Section 2.1 remains unqualified.
**Severity:** Low. A careful reader who reads the full section will understand correctly, but the takeaway is technically inaccurate as a standalone statement.

**Issue C (minor): RMSNorm role flag table in Section 2.3.9 lists `pop_input` as a Flag in CoreCTArgs.**
**File:** `03_cpp_kernel_header.md`, line 757.
**Text:** "Role flags in CoreCTArgs: `is_active` (Flag), `pop_input` (Flag)"
**Problem:** RMSNorm's `CoreCTArgs` contains only `is_active` as a Flag. `pop_input` is in `ComputeCTArgs` as `bool`, not in `CoreCTArgs`. Verified at `/localdev/salnahari/testing_dir/tt-blaze/blaze/ops/rmsnorm/kernels/op.hpp`: CoreCTArgs (lines 37-39) has only `Flag is_active = A::is_active`. `pop_input` is at line 57 in `ComputeCTArgs` as `bool pop_input = A::pop_input`. The `pop_input` is registered as a per-core flag in `emit()` via `f.per_core_unified_ct_args([f.flag(...)])`, but it is NOT declared as a Flag field in CoreCTArgs.
**Severity:** Medium. This is a factual error about which struct contains `pop_input`. The comparison table should say "`is_active` (Flag in CoreCTArgs)" for RMSNorm, and note `pop_input` is a `bool` in `ComputeCTArgs` (registered as a per-core flag via emit, not via CoreCTArgs Flag type).

### Verdict

All 22 original issues have been correctly addressed. The chapter is substantially accurate. Three minor remaining issues were found (A, B, C), of which Issue C is the most notable because it misattributes `pop_input` to `CoreCTArgs` for RMSNorm. None of these remaining issues would cause broken code if a reader followed the chapter's instructions.

**Chapter approved** pending the three minor fixes above.
