# Agent B Final Review: Cross-Chapter Coherence

1. **Top-level index says "no compute-kernel printf" but Chapter 6 documents DPRINT on compute kernels.** The `index.md` Chapter 6 row lists "no compute-kernel printf" as a Key Topic. However, `ch6_debugging_tools/available_tools.md` documents `DPRINT_MATH`, `DPRINT_UNPACK`, and `DPRINT_PACK` macros that provide printf capability on all three TRISC processors, and `gaps_and_limitations.md` Section 3 is titled "DPRINT limitations in compute kernels" (implying it works, with caveats). The plan.md also contains the outdated claim "No printf/logging from compute kernels: unlike data movement kernels which can use DPRINT, compute kernels running on TRISCs have extremely limited observability." The top-level index Key Topics and plan.md should be corrected to say something like "DPRINT available but limited on compute kernels" to match Ch6's actual findings.

2. **Ch2 abstraction_strengths.md references `acquire_dst` as the current API; Ch4 and Ch8 treat it as deprecated.** In `ch2_compute_kernel_api_abstraction/abstraction_strengths.md` (line 97), the Doxygen example says "The DST register buffer must be in acquired state via *acquire_dst* call" and lines 131-133 show the deprecated annotation. However, this is presented in a section praising the API's documentation quality, which could mislead a reader into thinking `acquire_dst` is the recommended call. Ch4 (`pattern_ergonomics.md`, `common_mistakes.md`) and Ch8 (`improvement_opportunities.md`) consistently present `tile_regs_acquire/commit/wait/release` as the current protocol and `acquire_dst/release_dst` as deprecated. Ch2 should add a brief note clarifying that the quoted docstring predates the deprecation, to avoid contradicting the rest of the guide.

3. **Ch7 index says "six distinct locations" and "six major pain points" in friction_points.md, but the top-level index says "six integration layers."** These counts are consistent with each other, so this is fine on the numbers. However, Ch7's friction_points.md lists six friction points (per-architecture duplication, sfpu_split_includes bottleneck, macro-based registration, header sprawl, lack of lightweight test harness, SFPU_OP_CHAIN_0 code injection), while the top-level index Key Topics column for Ch7 lists only four items: "Per-architecture duplication, sfpu_split_includes bottleneck, macro-based registration, testing overhead." The top-level summary omits "header sprawl" and "SFPU_OP_CHAIN_0 code injection" and renames "lack of lightweight test harness" to "testing overhead." This is a minor inconsistency but could cause a reader to miss two of the six friction points when scanning the top-level index.

4. **Ch8 improvement_opportunities.md Section 3 references `release_dst` as a current consumer of `DST_ACCUM_MODE`, contradicting Ch4/Ch8 Section 1 which treat it as deprecated.** In the YAML schema example at line 91, the `required_by` list for `DST_ACCUM_MODE` includes `release_dst` alongside `tile_regs_commit`. Since Ch4 and Ch8 Section 1 establish that `release_dst` is deprecated in favor of `tile_regs_commit/tile_regs_release`, the schema example should use only the non-deprecated names to maintain consistency within the same chapter and with Ch4.

5. No fifth issue found -- the remaining cross-chapter references (chapter numbering in Ch8's summary of prior chapters, cross-references to companion guide chapters, terminology for TRISC0/1/2, CB API call names, Quasar/Wormhole/Blackhole naming) are all consistent.

# Agent B Final Review — Pass 2

All four cross-chapter fixes have been verified:

1. **index.md DPRINT claim — FIXED.** The Ch6 Key Topics column now reads "DPRINT (available on all three TRISCs with timing-perturbation caveats)" instead of the old "no compute-kernel printf" claim. No residual incorrect language found in index.md.

2. **Ch2 abstraction_strengths.md acquire_dst deprecation note — FIXED.** Lines 98-99 now include an explicit clarification: "Note: `acquire_dst` is the deprecated name; the current API uses `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release`." Lines 132-136 also show the `[[deprecated]]` attribute with migration guidance. No longer misleading.

3. **index.md Ch7 Key Topics completeness — FIXED.** The Ch7 row now lists all six friction points including "header sprawl" and "SFPU_OP_CHAIN_0 code injection," which were previously omitted from the top-level summary.

4. **Ch8 YAML schema deprecated name — FIXED.** The `required_by` list for `DST_ACCUM_MODE` (line 91 of improvement_opportunities.md) now uses `tile_regs_release` and `tile_regs_commit` instead of the deprecated `release_dst`. Consistent with Ch4 and Ch8 Section 1.

No feedback — guide approved.
