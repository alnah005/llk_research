# Agent B Review: Chapter 2 — Pass 1

## Issue 1: Incorrect P1 count in Key Takeaways (03_divergence_analysis.md)

**File:** `03_divergence_analysis.md`, Key Takeaways, item 1 (line ~280)

**Claim:** "6 are P0 (identity, attributes, output inference, validation, profiling, graph tracing) and **5 are P1** (output allocation, hash, tensor args, multi-output, validation on hit)."

**Problem:** The Master Divergence Table (Section 2.3.2) and the Summary Divergence Matrix (Section 2.3.7) both list **6** P1 items, not 5. The P1 items are: #4 (`create_output_tensors`), #6 (`validate_on_program_cache_hit`), #7 (`validate_inputs`), #8 (`compute_program_hash`), #10 (output tensor count), and #14 (`io_tensors` convention). The parenthetical in the takeaway lists only five labels and omits `validate_inputs` (#7). The total still adds to 15 (6 P0 + 6 P1 + 2 P2 + 1 N/A), confirming the count in the body tables is correct and the takeaway text is wrong.

**Severity:** Wrong numerical answer. The body of the chapter is correct; only the summary sentence is wrong.

**Fix:** Change "5 are P1" to "6 are P1" and add "input validation" to the parenthetical list.

---

## Issue 2: Misleading claim about factory selection on cache hit (03_divergence_analysis.md)

**File:** `03_divergence_analysis.md`, Section 2.3.4, Program-Cache Behavior Comparison table (line ~168)

**Claim:** "Factory selection | Can change factory on cache hit (checked per invocation) | Single factory; no selection"

**Problem:** In TTNN's dispatch path, the result of `select_program_factory` (i.e., the variant index) is incorporated into the program cache hash via `compute_mesh_workload_hash`. If a different factory is selected on a subsequent invocation, the hash changes and the lookup is a cache **miss**, not a hit. The statement "can change factory on cache hit" implies the cached program can be reused even when a different factory would be selected, which is incorrect. What actually happens is that the factory index is hashed, so same-factory invocations hit the cache and different-factory invocations miss.

**Severity:** Material conceptual misleading about cache mechanics. A reader relying on this row to understand the cache hit/miss semantics would draw the wrong conclusion.

**Fix:** Replace with something like: "Factory selection | Factory index is part of the hash; selecting a different factory causes a cache miss | Single factory; no selection variability."

---

**No feedback — chapter approved** on the remaining content. The generic_op internals (Section 2.1), compilation pipeline (Section 2.2), and the rest of the divergence analysis (Section 2.3) are factually coherent and consistent with the plan and with each other.

---

# Agent B Review: Chapter 2 — Pass 2

**Pass 1 fix verification:**

1. **P1 count fix (03_divergence_analysis.md, Key Takeaways, line ~280):** Now reads "**6 are P1** (output allocation, validation on hit, validate_inputs, hash, multi-output, tensor args)." Count is correct (matches 6 P1 items in the Master Divergence Table and Summary Divergence Matrix), and `validate_inputs` is included in the parenthetical. Fix applied correctly.

2. **Factory selection cache semantics (03_divergence_analysis.md, Section 2.3.4, line ~168):** Now reads "Factory variant is part of the program hash; a change in factory selection produces a cache miss, not a cache-hit factory swap | Single factory; no selection." This accurately describes TTNN's hash-based cache behavior where the factory variant index feeds into `compute_mesh_workload_hash`. Fix applied correctly.

**No feedback — chapter approved.**

---

# Agent B Review: Chapter 2 — Pass 3

**No feedback — chapter approved.**
