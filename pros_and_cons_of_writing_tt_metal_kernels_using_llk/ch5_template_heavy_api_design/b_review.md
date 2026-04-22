# Agent B Review: Chapter 5 — Template-Heavy API Design — Pass 1

1. **Reciprocal throughput numbers are wrong (flexibility_and_performance.md, lines 87/93/95).** The chapter states "~7-bit, 1 cycle/element", "~24-bit, 5 cycles/element", and "~8-bit, 3 cycles/element" for the three reciprocal tiers. The source comments in `ckernel_sfpu_recip.h` say "throughput of 1c/32", "throughput of 5c/32", and "throughput of 3c/32" respectively — meaning cycles per 32 elements, not per element. The chapter overstates the per-element cost by a factor of 32.

2. **Exponential fast path "8 elements per call" claim is wrong (flexibility_and_performance.md, line 72).** The chapter says the SFPLOADMACRO pipeline "processes 8 elements per call." The source comment at `ckernel_sfpu_exp.h:424` says the code is "hand-unrolled for 8 iterations," where each iteration processes 32 elements via SFPU vector operations. The function processes 8 x 32 = 256 elements total, not 8 elements.

# Agent B Review: Chapter 5 — Pass 2

No feedback — chapter approved.

Both Pass 1 fixes have been correctly applied:
- Reciprocal throughput now reads "1 cycle/32 elements", "5 cycles/32 elements", and "3 cycles/32 elements" (verified against `ckernel_sfpu_recip.h` comments: "throughput of 1c/32", "throughput of 3c/32", "throughput of 5c/32").
- Exponential element count now reads "256 elements per call (8 iterations x 32 elements)" (verified against `ckernel_sfpu_exp.h:424` comment: "hand-unrolled for 8 iterations", each processing 32-element SFPU vectors).
