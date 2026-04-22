# Compression Analysis: Chapter 4 -- Tile Processing Pattern

**Analyst:** Agent C (Compressor)
**Crucial updates:** No
**Scope:** `index.md`, `pattern_ergonomics.md`, `common_mistakes.md`

---

## Redundancies Found

### R1. `reg_api.h` semaphore definitions repeated three times (HIGH redundancy)

The same `tile_regs_acquire()` / `tile_regs_wait()` implementation from `reg_api.h` is quoted as a code block in three separate locations:

- **`index.md` lines 29-35** -- pseudocode form showing acquire, commit, wait, release with inline comments
- **`pattern_ergonomics.md` lines 28-36** -- exact C++ from `reg_api.h` showing `tile_regs_acquire` and `tile_regs_wait`
- **`common_mistakes.md` lines 16-23** -- same C++ block with minor comment variation ("blocks until PACK releases")
- **`common_mistakes.md` lines 160-163** -- the `tile_regs_acquire` definition repeated a fourth time in Section 5

**Suggestion:** Show the full `reg_api.h` definitions once in `index.md` (the canonical pattern section). In `pattern_ergonomics.md` and `common_mistakes.md`, reference them by name without re-quoting. Estimated savings: ~25 lines.

### R2. ACQ/REL helper pattern explained twice with overlapping snippets (MEDIUM redundancy)

The deprecated-to-new-API migration through `ACQ()`/`REL()` wrappers is covered in two places:

- **`pattern_ergonomics.md` Section 3** (lines 99-144): explains both the deprecated `acquire_dst()`-based form and the new four-call form, lists five kernel files using the pattern, and includes code snippets from `layernorm.cpp`, `rmsnorm_pre_allgather.cpp`, and `compute_depthwise_conv1d.cpp`.
- **`common_mistakes.md` Section 6** (lines 183-216): re-explains the same deprecated vs. new split, re-quotes the same `rmsnorm_pre_allgather.cpp` snippet (lines 199-206 vs. `pattern_ergonomics.md` lines 117-124, identical), and re-quotes the `layernorm.cpp` form.

The `rmsnorm_pre_allgather.cpp` ACQ/REL code block is character-for-character identical in both files. The `layernorm.cpp` ACQ/REL code block is also duplicated.

**Suggestion:** Consolidate the ACQ/REL discussion into `pattern_ergonomics.md` Section 3 (where it fits naturally as an ergonomic shorthand). In `common_mistakes.md` Section 6, reduce to a brief cross-reference: "The ACQ/REL aliases described in Pattern Ergonomics Section 3 create an additional hazard: developers reading different kernels see different patterns for the same operation." Estimated savings: ~20 lines.

### R3. Deprecated vs. new API table restates prose (MINOR redundancy)

`common_mistakes.md` lines 186-189 present a table mapping `acquire_dst()` to `tile_regs_acquire() + tile_regs_wait()` and `release_dst()` to `tile_regs_commit() + tile_regs_release()`. This exact mapping is already explained in prose form in `pattern_ergonomics.md` lines 103-127, where the code snippets show both forms side by side.

**Suggestion:** Keep the table (it is a useful quick reference) but remove the prose re-explanation that follows it in `common_mistakes.md` lines 191-216, replacing with a cross-reference. The table alone conveys the mapping; the surrounding narrative repeats `pattern_ergonomics.md`.

### R4. BMM kernel code shown in both files (MINOR redundancy)

The BMM kernel (`bmm_large_block_zm_fused_bias_activation.cpp`) appears as code blocks in:

- **`pattern_ergonomics.md` lines 82-94** (bias addition stage) and **lines 172-187** (accumulation with reload)
- **`common_mistakes.md` lines 32-51** (conditional branch paths) and **lines 169-178** (reload helper)

The reload helper context overlaps: `pattern_ergonomics.md` shows it as an example of flexibility (holding DST across reload), while `common_mistakes.md` shows the same function to illustrate implicit contracts. Both quote `reload_from_cb_to_dst` and reference it being called inside an acquire block. The two uses serve different analytical purposes, so the redundancy is tolerable, but the overlapping context could be trimmed.

**Suggestion:** In `common_mistakes.md` Section 5, trim the `reload_from_cb_to_dst` snippet to just the function signature and the key comment about its implicit contract, rather than re-quoting the full body. The reader has already seen the full code in `pattern_ergonomics.md`. Estimated savings: ~8 lines.

---

## Verbose Prose / Hedging

### V1. Restated conclusions in `common_mistakes.md`

The final paragraph of `common_mistakes.md` (lines 229) restates the summary table (lines 219-228) in prose form: "The fundamental issue across all these failure modes is the same: the protocol relies entirely on the developer to maintain invariants that the toolchain does not enforce. There are no scoped guards, no runtime assertions, and no deadlock detection." This repeats what the table's "Symptom" and "Diagnosis Difficulty" columns already convey and echoes the opening sentence of Section 1 (lines 7-8).

**Suggestion:** Cut the final paragraph. The table is the summary.

### V2. Hedging phrases throughout

Several phrases add no analytical content:

- `pattern_ergonomics.md` line 1: "provides several genuine advantages" -- "genuine" is hedging against an unstated counterargument
- `pattern_ergonomics.md` line 42: "This consistency means that once a developer learns the pattern from a simple kernel like eltwise binary, they can navigate layernorm, groupnorm, or matmul kernels without learning a fundamentally new structure." -- This restates the heading "Predictable Structure Across All Kernels" as a sentence
- `pattern_ergonomics.md` line 97: "The shape is always the same: acquire, compute in a loop, commit, wait, pack in a loop, release. The only variation is which math operation fills the compute loop and how many tiles are packed." -- The preceding three code examples already demonstrate this; the prose summary is redundant with the section heading and the code itself
- `common_mistakes.md` line 92: "If a developer changes `subblock_w` or `block_hw` without updating all the matching calls, the kernel deadlocks silently." -- restates the pattern already established in Section 1

**Suggestion:** These are minor. Trim the restated-heading sentences in V2 bullets 2 and 3 for tighter prose. The others can stay.

---

## Summary

| ID | Type | Severity | Est. lines saved |
|---|---|---|---|
| R1 | Duplicate code block | High | ~25 |
| R2 | Duplicate explanation + identical snippets | Medium | ~20 |
| R3 | Table restates prose from other file | Minor | ~10 |
| R4 | Overlapping BMM code context | Minor | ~8 |
| V1 | Restated summary | Minor | ~3 |
| V2 | Hedging / restated headings | Minor | ~5 |
| **Total** | | | **~71 lines** |

No factual errors were flagged. All findings are structural/editorial redundancy only.
