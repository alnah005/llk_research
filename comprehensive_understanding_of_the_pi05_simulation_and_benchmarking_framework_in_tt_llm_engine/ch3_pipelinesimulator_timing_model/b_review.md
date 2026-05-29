# Agent B Review: Chapter 3 — Pass 1

## Issue 1: Header comment quote contradicts the section's own analysis (01_simulator_design.md, Section 5)

Section 5 ("Why No Tick Thread") quotes the source header comment verbatim, including line 36's claim that `read_result()` does "sleep_until(exit_time)" and lines 45-47's claim that "timing precision is now bounded only by sleep_until resolution." The section then spends its remaining subsections explaining that the implementation actually uses a **busy-wait spin loop** (lines 127-131 of the source), not `sleep_until`, specifically because `sleep_until` has 5-50us jitter on non-RT kernels.

The quoted comment is the source code's own inaccuracy (the comment was likely written before the busy-wait was adopted and never updated), but the doc should explicitly note this discrepancy rather than presenting the contradictory quote without annotation. A reader encountering the `sleep_until` quote first will be confused when the same section immediately argues against using `sleep_until`. A one-sentence callout (e.g., "Note: the header comment references `sleep_until`, but the implementation switched to busy-wait for the jitter reasons described below") would resolve this.

**Severity:** Moderate -- creates internal contradiction within the same section that could mislead readers about the actual timing mechanism.

## Issue 2: Tick-thread timing precision claim is wrong in the comparative table (01_simulator_design.md, Section 5)

The comparative table in Section 5 gives the tick-thread approach a timing precision of "$\pm$ stagePeriod/2". The text immediately above the table says tick quantization pads "up to one stagePeriod" ($0 \le \epsilon < \text{stagePeriod}$). These are inconsistent: if the error range is $[0, \text{stagePeriod})$ then the expected error is $\text{stagePeriod}/2$, but the worst-case precision is $\pm \text{stagePeriod}$, not $\pm \text{stagePeriod}/2$. The "$\pm$" notation implies symmetric error around the true value, but tick quantization only adds positive padding (a token cannot exit *before* its true exit time). The table should say "$0$ to $+\text{stagePeriod}$" or simply "$+\text{stagePeriod}$ worst case" to match the text's own formula.

**Severity:** Minor -- the text gives the correct formula; only the table summary is imprecise.

---

No other factual errors, coherence problems, or structural gaps found. The line-number references, code quotes, invariant descriptions, batch-prefill semantics, token arithmetic, safe-vocab modulus analysis, and config-propagation traces all check out against the source.
