# Agent B Review: Chapter 3 -- Pass 1

## Issue 1: Cost formula in Section 3.1.7 contradicts Section 3.1.6's "compile-once, run-many" model
**File:** `01_blaze_vs_ttnn_lifecycles.md`, lines 330--338
**Claim:** "Blaze rebuilds descriptors on every invocation" and the cost formula shows C_engines + C_emit + C_build + C_descriptor as a fixed per-invocation cost even on cache hits.
**Problem:** Section 3.1.6 (same file, line 324) correctly states: "Compile-once, run-many. A MeshCompiledProgram can be compiled once and run many times with different tensor addresses (via runtime arg patching), amortizing the Python-side compilation cost." In the compile-once-run-many model, subsequent `run()` calls invoke `_run_program()` with the already-built `MeshProgramDescriptor` -- they do NOT re-run `emit()`, the engine pipeline, or descriptor assembly. The cost formula is therefore wrong for the common case of repeated execution: after the first compilation, per-invocation Blaze cost on a cache hit is C_hash + C_override (descriptor hashing plus runtime arg override), not the full pipeline. The formula as stated overstates Blaze's steady-state overhead and directly contradicts the chapter's own prior section. A downstream reader comparing TTNN vs Blaze per-invocation costs would reach incorrect performance conclusions.
**Severity:** High -- leads to wrong numerical reasoning about per-invocation costs and contradicts an earlier claim in the same file.
**Fix:** Revise Section 3.1.7 to distinguish first-invocation cost (full pipeline) from steady-state cost (reusing the compiled MeshProgramDescriptor). The formula should show two regimes: (1) first compilation pays C_engines + C_emit + C_build + C_descriptor + cache-dependent cost, and (2) subsequent runs with the same MeshCompiledProgram pay only C_hash + C_override on cache hit (or C_hash + C_create_at on miss). The remaining asymmetry -- that Blaze's C_hash is O(n) descriptor hashing vs TTNN's O(1) attribute hashing -- is real and should be the focus, not the full pipeline cost. Also fix the Key Takeaways bullet that repeats this claim ("Blaze pays the full Python compilation pipeline plus O(n) descriptor hashing even on hits").

---

## Issue 2: Index.md states "42,000--60,000 lines" but Section 3.2 computes different numbers
**File:** `index.md`, line 22, vs `02_translation_challenges.md`, lines 298--315
**Claim:** The index says "approximately 42,000--60,000 lines of C++ boilerplate." Section 3.2 provides per-component ranges (MicroOp: 250--600 lines, FusedOp: 400--850 lines) and computes the midpoint as 52,750.
**Problem:** The actual min-max from the stated per-component ranges is 80x250 + 30x400 = 32,000 (minimum) to 80x600 + 30x850 = 73,500 (maximum). The index's "42,000--60,000" range is neither the min-max (32K--73.5K), nor centered on the midpoint (52.75K), nor any obvious statistical interval. Section 3.2's summary table says "~53K lines." A reader who encounters the index first and then reads the detailed computation in Section 3.2 will find a numerical inconsistency.
**Severity:** Moderate -- the numbers are in the same order of magnitude and the argument still holds, but a reader doing their own arithmetic will notice the mismatch and question the rigor.
**Fix:** Either update the index to match the actual computed range ("approximately 32,000--73,500 lines") or state the midpoint ("approximately 53,000 lines") consistent with Section 3.2's summary table. If a narrowed range is preferred (e.g., interquartile estimate), state the methodology.

---

# Agent B Review: Chapter 3 -- Pass 2

**Pass 1 fix verification:**

1. **Issue 1 (cost formula contradiction) -- FIXED.** Section 3.1.7 now distinguishes first-invocation cost (full pipeline) from steady-state cost (hash + override only). The formulas correctly show two regimes. The explanatory paragraph explicitly states "The Python compilation pipeline is paid only on the first invocation -- subsequent `run()` calls on the same `MeshCompiledProgram` reuse the already-built descriptor." The Key Takeaways bullet has been corrected to focus on hash cost asymmetry rather than claiming Blaze pays the full pipeline on hits. No residual contradiction with Section 3.1.6.

2. **Issue 2 (index.md line count mismatch) -- FIXED.** The index now reads "approximately 53,000 lines of C++ boilerplate at midpoint estimates (range: 32,000--73,500)." Verified: min = 80*250 + 30*400 = 32,000; max = 80*600 + 30*850 = 73,500; midpoint = 52,750 rounds to ~53,000. All three numbers are consistent with Section 3.2.8's per-component table and aggregate computation.

**No feedback -- chapter approved.**

# Agent B Review: Chapter 3 — Pass 3

**Pass 2 fix verification:**

Both Pass 1 issues remain correctly resolved. The cost formulas in Section 3.1.7 properly distinguish first-invocation from steady-state regimes, and the index.md line counts (53,000 midpoint; 32,000--73,500 range) are arithmetically consistent with Section 3.2.8's per-component table.

**No feedback — chapter approved.**
