# Compression Analysis -- Chapter 6: Template Design Trade-offs

**Agent:** C (Compressor, Pass 1)
**Date:** 2026-04-05
**Files reviewed:** `index.md` (51 lines), `compilation_impact.md` (139 lines), `binary_size_and_alternatives.md` (171 lines)
**Total lines reviewed:** 361

## VERDICT

Crucial updates: no

---

## Load-Bearing Evidence

- **`index.md`**: The five "Key Observations" (lines 24-50) restate facts that are explained in full in the two sub-pages. Observation 1 (6 template parameters, the wrapper forwarding 5) duplicates `compilation_impact.md` lines 10-68. Observation 2 (26 `if constexpr` branches) duplicates `binary_size_and_alternatives.md` lines 10-11. Observation 3 (JIT caching) duplicates `compilation_impact.md` lines 86-106. Observation 4 (`-O3 -flto=auto`) duplicates `compilation_impact.md` lines 110-135. Observation 5 (the trade-off is deliberate) duplicates `binary_size_and_alternatives.md` lines 132-166.

- **`compilation_impact.md`**: The "Header-Only Means Full Re-parsing" section (lines 72-83) restates the same point already made in `index.md` line 9 ("the entire LLK implementation is header-only, every kernel translation unit re-parses...") and again elaborated in `binary_size_and_alternatives.md` lines 157-161 (header change invalidates cached objects).

- **`binary_size_and_alternatives.md`**: The "Why the Current Approach Is Likely Correct" section (lines 132-151) repeats the JIT-caching justification from `compilation_impact.md` lines 86-106, repeats the SRAM constraint rationale from `index.md` line 49, and repeats the host-vs-device cost framing from `index.md` line 50.

---

## Cross-File Redundancy (major)

The following facts appear in two or all three files. Each repetition is a candidate for removal or reduction to a brief cross-reference.

| Fact | `index.md` | `compilation_impact.md` | `binary_size_and_alternatives.md` |
|------|-----------|------------------------|----------------------------------|
| 6 template parameters on `_llk_math_eltwise_binary_` | lines 24-28 | lines 10-23 (with code block) | -- |
| Metal wrapper forwards 5 params + `DST_SYNC_MODE` | lines 29-31 | lines 42-69 (with code block) | -- |
| 960 theoretical instantiations | -- | lines 25-38 (table + math) | -- |
| 26 `if constexpr` in one file | line 34 | -- | line 11 |
| Header-only = full re-parsing per TU | line 9 | lines 72-83 | lines 157-161 |
| JIT cache hashes params, skips on hit | lines 38-40 | lines 86-106 | lines 148-151 |
| `-O3 -flto=auto` flags | lines 43-45 | lines 110-135 (with code block) | line 145 |
| Trade-off is deliberate: host cost for device perf | lines 48-50 | -- | lines 132-166 |
| SRAM is limited / binaries must be small | line 49 | -- | lines 140-142 |

**Estimated saveable lines:** ~40-50 lines if `index.md` Key Observations are reduced to 1-2 sentence teasers with cross-references, and `binary_size_and_alternatives.md` "Why the Current Approach Is Likely Correct" back-references `compilation_impact.md` for JIT caching instead of restating it.

---

## MINOR Suggestions

1. **`index.md` Key Observations section (lines 24-50):** This 27-line section is essentially a prose abstract of the two sub-pages. Compress to ~8 lines: one sentence per observation with a link to the sub-page section that has the full explanation. Saves ~19 lines.

2. **`compilation_impact.md` "Header-Only Means Full Re-parsing" (lines 72-83):** The bullet at line 82-83 ("the compiler parses, type-checks, and potentially instantiates the entire LLK implementation for every kernel object file, even though each kernel typically uses only one combination") nearly duplicates the sentence at `index.md` line 9. Since this is the sub-page with full detail, keep it here and trim the `index.md` version.

3. **`binary_size_and_alternatives.md` lines 148-151 (JIT caching point):** This restates the caching mechanism already fully explained in `compilation_impact.md` lines 86-106. Replace with a single sentence and a cross-reference link. Saves ~3 lines.

4. **`binary_size_and_alternatives.md` "The Pain Point" section (lines 153-166):** The first bullet ("First-time builds are slow... at -O3 -flto=auto") restates `compilation_impact.md` lines 110-135. The second bullet ("Changing an LLK header invalidates many cached objects") restates `compilation_impact.md` lines 72-83. Consider moving this section to `compilation_impact.md` where all compilation-cost discussion lives, and replacing it here with a link. Saves ~10 lines.

5. **Hedging language in `binary_size_and_alternatives.md`:** "The template-heavy design is the right choice for the target hardware, for several reasons:" (line 134) followed by "This would... may push kernels past size limits" (line 123). The hedging "may" and "likely" in the heading "Why the Current Approach Is Likely Correct" could be tightened -- the reasoning presented is definitive, not speculative.

---

## Summary

- **Total lines reviewed:** 361
- **Estimated removable lines (redundancy):** ~40-50
- **Estimated compressible lines (verbose prose):** ~10-15
- **Net reduction estimate:** ~50-65 lines (~14-18%)
- **Risk:** Low. All flagged content is duplicated elsewhere in the chapter; removing it loses no information.
