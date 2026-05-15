# B Review -- Pass 3

## Issues Found

1. **Phase 4 LoC sub-items do not match their stated total (Section 8.2.6).** The quantitative assessment states "Modified Python lines | ~120 (_registry.py: ~80, tracing.py: ~20, tests: ~200)". The sub-items sum to 300, not 120. Either the total should be ~300 or the sub-item breakdown is wrong.

2. **Phase 1 per-phase LoC in the summary table does not match the detail section (Section 8.2.7 vs 8.2.3).** The cumulative effort summary table lists Phase 1 as ~810 LoC. The Phase 1 quantitative assessment lists ~725 new C++ + ~60 modified Python = ~785 LoC. Discrepancy of 25 LoC.

3. **Phase 2 per-phase LoC in the summary table far exceeds the detail section (Section 8.2.7 vs 8.2.4).** The summary table lists Phase 2 as ~410 LoC. The Phase 2 quantitative assessment only lists ~80 new C++ lines and 2 modified files, with no Python line count or test line count provided. The gap of ~330 LoC is unexplained.

4. **Phase 3 per-phase LoC in the summary table far exceeds the detail section (Section 8.2.7 vs 8.2.5).** The summary table lists Phase 3 as ~400 LoC. The Phase 3 quantitative assessment lists ~60 new C++ + ~40 modified Python = ~100 LoC. The gap of ~300 LoC is unexplained.

5. **Open Question 3 recommended phase contradicts between the priority matrix and the resolution timeline (Section 8.4.1 vs 8.4.4 vs 8.4.9).** The priority matrix (8.4.1) assigns Q3 urgency 3 ("must resolve before Phase 1 begins") and recommended phase "Phase 0--1." However, Q3's own resolution timeline (8.4.4) says "Before Phase 2 (batch registration with custom hashes). Phase 1 ops can use Tier 1 (descriptor walk) exclusively." The resolution roadmap (8.4.9) lists Q3 as blocking Phase 2, not Phase 1. These three statements are inconsistent.

6. **MicroOp + FusedOp counts do not cleanly sum to 112.** The text uses "~60 MicroOps" (Phase 2) and "~50 FusedOps" (Phase 3), which sum to ~110. However, "112 ops" is used as the total throughout (index, Section 8.1.5 build-time projections, Section 8.2.10 testing matrix). The approximate per-category counts should be reconciled with the total (e.g., ~62 MicroOps + ~50 FusedOps = ~112, or ~60 + ~52 = ~112).

## VERDICT
- Issues found: yes
