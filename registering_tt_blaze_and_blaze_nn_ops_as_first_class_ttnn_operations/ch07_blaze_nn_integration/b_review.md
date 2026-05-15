# Chapter 7 -- Agent B Review (Pass 1)

1. **File:** `02_rewired_dispatch_and_module_integration.md`, lines 61 and 141-153.  
   **Error:** `_FORCE_GENERIC` is evaluated once at module import time (`_FORCE_GENERIC = os.environ.get("BLAZE_NN_FORCE_GENERIC", "0") == "1"`), but `reset_dispatch_cache()` does not re-read it. Consequently, the test at lines 872-878 (`test_force_generic_env_var`) would not work: calling `monkeypatch.setenv` after the module is imported has no effect on the already-bound `_FORCE_GENERIC` variable, so `mapping.ttnn_callable` would still resolve to a callable (not `None`).  
   **Fix:** Either (a) make `_resolve_ttnn_callable()` read the env var live each time (e.g., `if os.environ.get("BLAZE_NN_FORCE_GENERIC", "0") == "1": return None`) instead of referencing the module-level constant, or (b) have `reset_dispatch_cache()` also re-read `_FORCE_GENERIC` from `os.environ`, or (c) update the test to also patch the module-level `_FORCE_GENERIC` variable directly (e.g., `monkeypatch.setattr(blaze_nn._registry, "_FORCE_GENERIC", True)`).

2. **File:** `02_rewired_dispatch_and_module_integration.md`, lines 158-183 and line 802.  
   **Error:** The "dual-resolution architecture" section claims "Both must agree for the named dispatch path to activate" (line 166), and the dispatch mechanics table says `_dispatch()` "Resolves and passes `ttnn_callable` to context" (line 802). However, neither the proposed `_dispatch()` code (not shown) nor the scenario walkthrough (lines 296-324) demonstrates Resolution Point 1's result gating or influencing Resolution Point 2 in any way. In the walkthrough, `mapping.ttnn_callable` is accessed (line 302) but its return value is never passed to `ctx.dispatch()` or used in the subsequent dispatch chain. The actual dispatch decision is made entirely by `_resolve_dispatch()` in `MeshCompiledProgram.__init__` (Resolution Point 2). A reader implementing this design would not know what `_dispatch()` is supposed to do with the `ttnn_callable` it resolves, since no modified `_dispatch()` code is shown and the walkthrough does not use the value.  
   **Fix:** Either (a) show the modified `_dispatch()` code that passes `ttnn_callable` to the context and explain how it reaches `MeshCompiledProgram`, or (b) remove the claim that Resolution Point 1 gates dispatch and clarify that it exists solely for the diagnostic API and early validation logging, while Resolution Point 2 is the sole dispatch decision maker.

3. **File:** `02_rewired_dispatch_and_module_integration.md`, line 839 vs. lines 296-324 and the end-to-end flow at lines 693-767.  
   **Error:** The per-file change summary says `blaze_nn/tracing.py` is modified: "`ComposeTracingContext.dispatch()` accepts `ttnn_callable` param (signature change)." But neither the scenario walkthrough (Call 1, lines 296-324) nor the end-to-end flow diagram (lines 693-767) shows `ttnn_callable` being passed to `ComposeTracingContext.dispatch()`. The walkthrough shows `ctx.dispatch("rmsnorm", [x, weight], {}, "1x32")` -- the same four-argument signature as the current system. A reader implementing this would see a claimed signature change with no corresponding code or usage example.  
   **Fix:** Either show the updated `ComposeTracingContext.dispatch()` signature and update the walkthrough to pass `ttnn_callable` as a fifth argument, or remove the claim that `tracing.py` changes if the `ttnn_callable` is not actually passed through the context.

4. **File:** `02_rewired_dispatch_and_module_integration.md`, line 166 vs. `ch04_adapter_pattern/03_python_dispatch_and_registration.md`, line 493.  
   **Error:** Section 4.3.7 states "blaze-nn benefits automatically without any code changes" and "No changes needed in blaze-nn!" Chapter 7 then proposes ~65 lines of changes to `blaze_nn/_registry.py` and `blaze_nn/tracing.py`, including modifications to `_dispatch()` and `ComposeTracingContext`. A reader who implements Section 4.3's design and reads its explicit "no changes needed" claim would skip Chapter 7's blaze-nn-layer changes entirely. The two chapters give contradictory implementation guidance on whether blaze-nn source files need modification.  
   **Fix:** In Section 4.3.7, qualify the "no changes needed" claim -- e.g., "blaze-nn benefits from named dispatch automatically without code changes; Chapter 7 describes optional enhancements (diagnostic API, early callable verification, env-var override) that add ~65 lines to blaze-nn for migration tooling and defense in depth." This makes clear that the core dispatch works without blaze-nn changes, while the Chapter 7 additions are supplementary.

## Pass 2

**No feedback — chapter approved.**

All four Pass 1 issues have been correctly resolved:

1. **`_FORCE_GENERIC` env var (lines 85-116, 128, 869-878):** `_resolve_ttnn_callable()` now reads the env var live via `os.environ.get("BLAZE_NN_FORCE_GENERIC", "0")` on each call rather than referencing a cached module-level constant. The design decisions text explicitly states "read live on each resolution (not cached at import time)." The test correctly uses `monkeypatch.setenv` followed by `reset_dispatch_cache()`, which works because the live read picks up the changed `os.environ` value on next resolution.

2. **Dual-resolution claim (lines 157-163, 799):** Resolution Point 1 is now described as existing "for the diagnostic/introspection API (`get_dispatch_info()`, `get_registration_report()`) and early validation logging. It does not gate dispatch." Resolution Point 2 is identified as "the sole dispatch decision maker." No "both must agree" language remains. The dispatch mechanics table describes `_dispatch()` as resolving `ttnn_callable` "for diagnostics."

3. **`tracing.py` signature (lines 836-837):** The per-file change summary now reads: "`ComposeTracingContext.dispatch()` is unchanged -- the `ttnn_callable` is resolved independently by `MeshCompiledProgram` via Resolution Point 2" with 0 lines modified. No false claim of a signature change.

4. **Ch4 Section 4.3.7 contradiction (ch04, lines 504-507):** The "no changes needed" claim is now qualified with an inline comment: "blaze-nn benefits from named dispatch automatically without code changes. Chapter 7 describes optional enhancements (diagnostic API, early callable verification, env-var override) that add ~60 lines to blaze-nn for migration tooling and defense in depth." This correctly distinguishes the automatic benefit from the optional Chapter 7 enhancements.

## Pass 3

**No feedback — chapter approved.**

Post-compression verification: all six compression targets (redundant walkthroughs condensed, duplicate before/after blocks removed, overlapping summary matrices merged, key takeaways deduplicated, scenario calls shortened, ASCII diagram condensed) have been applied without introducing correctness issues. Specifically:

- **No dangling references.** All internal section references (7.1.1, 7.1.6, 7.1.7, 7.1.8, 7.1.10, 7.2.7, 7.2.11) resolve to existing headings. Section numbering is contiguous in both files (7.1.1--7.1.10, 7.2.1--7.2.14) with no gaps.
- **No broken cross-references.** References to Chapters 2, 4, 5, and 6 point to files that exist. References to Chapter 8 point to a directory not yet created, but this is a pre-existing condition (Chapter 8 is unwritten), not a compression artifact.
- **No logical gaps.** The condensed "Calls 3--8" block (01, lines 358--360) correctly summarizes the remaining six calls by noting they "follow the same dispatch path demonstrated in Calls 1 and 2" and identifies the varying parameters (op name, grid). The before/after layer map in 02 (Section 7.2.11) is self-contained and references 7.1.10 for the baseline. The key takeaways in index.md cover all five design points without duplication.
- **Plan spec bullets fully covered.** All items from the plan's Chapter 7 file descriptions are present: `_REGISTRY`/`OpMapping`, `TensorProxy`, `Parameter`, `GraphTracingContext` vs `ComposeTracingContext`, `F.linear()` dispatch path, scenario walkthrough (file 01); new dispatch path, `ttnn_callable` field, fallback strategy, graph vs direct mode, module integration, state dict unchanged, composability with native TTNN ops, end-to-end flow (file 02).
