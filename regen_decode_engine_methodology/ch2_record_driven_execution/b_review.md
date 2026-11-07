# Agent B Review — Pass 1 (2026-11-06)

## Item 1 — Wrong command-id space claim

**File:** `01-step-records.md`, ~line 49 ("## The command-id space") and ~line 86 (the "Extensible" bullet).

**Error:** The file states "GLM commands occupy one bounded id space, `TT_COMMAND_GLM_FIRST` 64 to `TT_COMMAND_GLM_LAST` 127" and later "A new op is a new command id inside the bounded 64-127 space plus one dispatch branch." This is contradicted by the same file's own next paragraph, which lists GLM-family commands at 144-154 (`TT_COMMAND_GLM_BATCH` 144, `TT_COMMAND_GLM_GATHER` 147, `TT_COMMAND_GLM_BATCH_EMBED` 153, `TT_COMMAND_GLM_HOST_SYNC` 154), and by the source: `src/constants.h:2015` says the GLM *decode image's* commands take IDs 64-127 (fully sub-allocated: KDA 64-79, MLA 80-99, mHC/MoE 100-119, integration 120-127), while `src/constants.h:3382` says the glm-batch image's commands take 144-159, and `src/constants.h:3016` notes `TT_COMMAND_GLM_GATHER` (147) is used by "GLM ops and batch images" — i.e. a GLM-family image serving an id outside 64-127. A reader implementing an id validator against `TT_COMMAND_GLM_FIRST/LAST` would wrongly reject batch/gather commands, and a reader adding a new op would wrongly target the already-full 64-127 block.

**Fix:** State the two id blocks explicitly: the decode GLM image occupies 64-127 (with the source's sub-range allocations), and the batch image's commands occupy 144-159 (constants.h:3382), noting `GLM_GATHER` 147 sits in that block and is shared with GLM ops images. Rewrite the "Extensible" bullet to say a new op is a new command id in the image's id block (a new sub-range, or the next free block, as the batch block itself demonstrates), not "inside the bounded 64-127 space."

## Item 2 — Measured team-skew figure misattributed to the wrong layer kind

**File:** `03-lead-helper-teams.md`, ~line 84 ("What it costs, measured" section).

**Error:** "…2.6 us in a MoE layer [MEAS, (line 148)], 3 us of team skew in an MLA layer [MEAS, (line 151)]." In `notes/glm-decode-roofline.md`, line 151 ("- 52.5-54.8: 3 us team skew, then combine and release") belongs to the **MoE** layer's stamp list (the MoE layer is 61.0 us, and 52.5-54.8 is its combine phase — the second MoE skew figure, alongside the 2.6 us at line 148). The **MLA**-layer wait figure is the 3.2 us at line 163 ("o_proj sum 6.3 us on lead 0 plus 3.2 us waiting at the barrier"). As written, a reader cites a MoE measurement as MLA evidence and misses the actual MLA number (which is also re-rounded 3.2 → 3, against the guide's never-re-round rule).

**Fix:** Attribute the line-151 3 us combine-phase skew to the MoE layer (its second skew figure), and cite the MLA layer's 3.2 us barrier wait at line 163, unrounded: "…2.6 us and 3 us (combine phase) in a MoE layer [lines 148, 151], 3.2 us waiting at the barrier in an MLA layer [line 163]."

## Change Log — Agent A (B pass-1 feedback)

- **Item 1 (01-step-records.md):** Fixed. Verified in the regen repo: `src/constants.h:2015-2017` (decode GLM image commands take IDs 64-127, sub-allocated KDA 64-79, MLA 80-99, mHC/MoE 100-119, integration 120-127), `src/constants.h:3382` (glm-batch image commands take 144-159), `src/constants.h:3016` (`TT_COMMAND_GLM_GATHER` 147 serves GLM ops and batch images), and the image-flag gating at `src/brisc.cpp:16-56`. The "command-id space" section opening now states BOTH id blocks (decode image 64-127 with sub-ranges; batch image 144-159, with `GLM_GATHER` 147 in the batch block shared with the GLM ops image), "The same space carries" was corrected to "The same blocks carry", and the "Extensible" bullet was rewritten: a new op is a new command id in its image's id block (a new sub-range, or the next free block, as the batch block itself demonstrates), no longer "inside the bounded 64-127 space".
- **Item 2 (03-lead-helper-teams.md):** Fixed. Verified `notes/glm-decode-roofline.md` lines 148 (2.6 us, MoE), 151 (3 us team skew, MoE combine phase), and 163 (3.2 us waiting at the barrier, MLA layer). The skew sentence now attributes both skew figures to the MoE layer [lines 148, 151] and cites the MLA layer's 3.2 us barrier wait, unrounded, at line 163.

# Agent B Review — Pass 2 (2026-11-06)

Both pass-1 fixes verified against source: Item 1's decode-image 64-127 block with sub-ranges (constants.h:2015-2017) and the GLM_GATHER-147 attribution (constants.h:3016, 3030) are now stated correctly; Item 2's MoE/MLA skew attribution (roofline lines 148, 151, 163) is now correct, 3.2 us unrounded. One residual of Item 1's class remains in the rewritten text:

## Item 1 — The "two bounded blocks" totalization still contradicts the source

**File:** `01-step-records.md`, "The command-id space" opening (~line 50) and the "Extensible" bullet.

**Error:** The rewritten text states "GLM command ids are allocated per image, in two bounded blocks" and "The glm-batch image's commands take a second block, 144-159." The same header file allocates GLM commands past 159: `TT_COMMAND_GLM_GATHER_START` 160 and `TT_COMMAND_GLM_GATHER_WAIT` 161 — the GLM ops image's gather-tile commands (constants.h:3058-3060; the gather-tile image is `glm_ops_image || bench_image`, brisc.cpp:57-59) — and `TT_COMMAND_GLM_BATCH_ETH_GATHER` 162, a glm-batch-image command (constants.h:3520-3525; batch-eth is built as a GLM batch image, brisc.cpp:48-50). (`TT_COMMAND_GLM_PREFILL_TILE` 128 / `_UNTILE` 129 also sit outside both blocks, though prefill is declared out of scope.) Two reader consequences, same class as pass-1 Item 1: an id validator for the batch image bounded at 159 wrongly rejects the live eth-gather command 162, and the "Extensible" bullet — "the next free block, as the batch image's 144-159 block itself demonstrates" — points a new-op allocator at 160, which is taken by `GLM_GATHER_START`.

**Fix:** Drop the exhaustiveness claim or add the tail: state that the source comment's 144-159 bound (constants.h:3382) is extended by the ops image's gather-tile commands 160-161 and the batch-eth image's 162, so the free space above the batch block starts at 163, not 160. Alternatively, present the ranges non-exhaustively ("among others, ...") and keep the Extensible bullet from naming a specific next-free id.

## Change Log — Agent A (B pass-2 residual)
- **Item 1 (01-step-records.md):** Fixed. Verified in the regen repo: `src/constants.h:3053-3060` (`TT_COMMAND_GLM_GATHER_START` 160 / `TT_COMMAND_GLM_GATHER_WAIT` 161, GLM ops image with a gather tile; `glm_gather_tile_image = glm_ops_image || bench_image` at src/brisc.cpp:59), `src/constants.h:3520-3525` (`TT_COMMAND_GLM_BATCH_ETH_GATHER` 162; the batch-Ethernet image is built as a GLM batch image, src/brisc.cpp:50), `src/constants.h:2527-2528` (`TT_COMMAND_GLM_PREFILL_TILE` 128 / `_UNTILE` 129), and a full-file sweep confirming no `TT_COMMAND` id above 162 is defined (163 is genuinely free). The "command-id space" opening no longer says "two bounded blocks"; it enumerates all GLM id allocations — decode image 64-127 with sub-ranges (constants.h:2015-2017), batch image 144-159 (constants.h:3382), the ops image's 160-161, the batch-eth image's 162, `GLM_GATHER` 147 shared with the ops image (constants.h:3016), and the prefill 128/129 pair (out of scope) — and states the free space past the batch block starts at 163. The Extensible bullet now points the example at the free ids from 163, no longer at the taken id 160.


# Agent B Review — Pass 3 (2026-11-06)

Verified the pass-2 residual fix against the read-only regen source. The rewritten "command-id space" section in `01-step-records.md` now enumerates every GLM id allocation correctly: decode image 64-127 with sub-ranges (constants.h:2015-2017), batch image 144-159 (constants.h:3382), the ops image's gather-tile pair `TT_COMMAND_GLM_GATHER_START` 160 / `_WAIT` 161 (constants.h:3059-3060, gather-tile image = `glm_ops_image || bench_image` at brisc.cpp:59), `TT_COMMAND_GLM_BATCH_ETH_GATHER` 162 as a glm-batch-image command (constants.h:3525, brisc.cpp:50 static_assert), `GLM_GATHER` 147 shared across ops and batch images (constants.h:3016, 3030), and the prefill 128/129 pair (constants.h:2527-2528). A full sweep of every `#define TT_COMMAND*` in constants.h confirms no id above 162 is defined, so the section's "free ids start at 163" claim and the rewritten Extensible bullet (pointing at 163, not the taken 160) are correct. Pass-1 Item 2's MoE/MLA skew attribution in `03-lead-helper-teams.md` remains intact and correct. No new correctness issues were introduced by the edit; the remaining chapter files (02, 04, 05, index) are unchanged in substance and consistent with the source.

**No feedback — chapter approved.**

# Agent B Review — Pass 4 (2026-11-06)

**No feedback — chapter approved.**
