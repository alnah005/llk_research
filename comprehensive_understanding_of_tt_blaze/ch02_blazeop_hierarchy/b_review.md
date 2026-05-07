# Agent B Review: Chapter 2 -- The BlazeOp Class Hierarchy and Op Authoring

## Pass 1

### Error 1

**File:** `03_cpp_parser.md`, lines 14-16. **Error:** The `ParsedKernel` dataclass is shown with `ct_args: list[ParsedCTArg] = []` and `pop_flags: list[str] = []`. In the actual source (`blaze/cpp_parser.py`, lines 44 and 51), both fields use `field(default_factory=list)`, not bare `= []`. This is not merely a stylistic simplification -- bare mutable defaults in a Python `@dataclass` raise `ValueError` at class definition time. The code as written in the chapter would not run. **Fix:** Change to `ct_args: list[ParsedCTArg] = field(default_factory=list)` and `pop_flags: list[str] = field(default_factory=list)`, or add a note that the defaults are simplified and the real code uses `field(default_factory=list)`.

### Error 2

**File:** `05_writing_a_micro_op.md`, lines 103-120. **Error:** The `WriterCTArgs` struct listing for Mcast omits the `src_num_pages` field. The actual C++ header (`blaze/ops/mcast/kernels/op.hpp`, line 201) declares `static constexpr uint32_t src_num_pages = A::src_num_pages;` between `src` and `dst` in `WriterCTArgs`. The Python `emit()` also registers this field in `brisc_ct_args` (source: `blaze/ops/mcast/op.py`, line 113). **Fix:** Add `static constexpr uint32_t src_num_pages = A::src_num_pages;` after the `src` line in the WriterCTArgs listing.

### Error 3

**File:** `05_writing_a_micro_op.md`, lines 138-139. **Error:** The chapter states "Mcast also defines `setup_src()`, a template method for re-initializing the source buffer with different parameters. The parser captures this as `setup_method = 'setup_src'`." In reality, `setup_src()` in the Mcast header (`blaze/ops/mcast/kernels/op.hpp`, line 304) is declared as `template <typename A> void setup_src()` -- a template instance method, NOT a `static` method. The parser regex `_SETUP_RE` (source: `blaze/cpp_parser.py`, line 87) requires `static\s+void\s+(setup_\w+)`, so it does NOT match `setup_src()`. Testing the regex against the Mcast header confirms no match. The parser will set `setup_method = None` for Mcast, not `"setup_src"`. **Fix:** Correct the statement to note that the parser's `_SETUP_RE` does not match Mcast's `setup_src()` because it is a template instance method rather than a static method. The `setup_method` field on `ParsedKernel` will be `None` for Mcast.

### Error 4

**File:** `03_cpp_parser.md`, lines 159-165. **Error:** The parser section describes "Step 7: Setup Methods" and says the pattern captures "static methods on the `Op` class". It then says "for example, `Mcast::Op::setup_src<ct_args::another_prefix>()` sets up the sharded buffer for a different mcast instance." However, `setup_src()` is a template instance method on `Op` (not static), and the regex `_SETUP_RE` would not match it. The description of the feature is self-consistent but the Mcast example is incorrect -- no op in the current codebase has a `static void setup_*()` method that the regex would match. A `grep` of all `blaze/ops/*/kernels/op.hpp` files confirms zero matches for `static.*void.*setup_`. **Fix:** Either update the example to use an op that actually has a matching static setup method, or clarify that the regex currently does not match Mcast's `setup_src()` and explain how setup method detection actually works in practice (e.g., whether the codegen uses a different mechanism to detect template setup methods).

### Error 5

**File:** `05_writing_a_micro_op.md`, lines 220-237. **Error:** The BRISC (writer) args code snippet omits `(f"{prefix}.src_num_pages", num_pages)`. The actual `Mcast.emit()` source (`blaze/ops/mcast/op.py`, line 113) includes `(f"{prefix}.src_num_pages", num_pages)` in the `brisc_ct_args` list. This corresponds to the `src_num_pages` field in `WriterCTArgs` (Error 2 above). **Fix:** Add `(f"{prefix}.src_num_pages", num_pages),` to the BRISC CT args snippet, after the `src` entry.

## Pass 2

All five Pass 1 fixes verified against source code:

1. `03_cpp_parser.md` lines 17, 24: `field(default_factory=list)` now matches `cpp_parser.py` lines 44, 51.
2. `05_writing_a_micro_op.md` line 114: `src_num_pages = A::src_num_pages` now matches `mcast/kernels/op.hpp` line 201.
3. `05_writing_a_micro_op.md` lines 140-141: Correctly states the parser's `_SETUP_RE` does not match Mcast's template `setup_src()`, and `setup_method` is `None`.
4. `03_cpp_parser.md` lines 159-165: No longer uses Mcast as an example of a matched static setup method. Correctly notes the `static` keyword requirement.
5. `05_writing_a_micro_op.md` line 234: `src_num_pages` now present in BRISC CT args snippet, matching `mcast/op.py` line 113.

No feedback — chapter approved.
