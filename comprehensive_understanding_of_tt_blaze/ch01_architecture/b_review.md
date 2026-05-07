# Agent B Review: Chapter 1 — Architecture and Position in the Stack

## Pass 1

1. **File:** `01_stack_position.md`, lines 64-69. **Error:** The chapter states "The submodule tracks the `blaze-metal` branch, which includes experimental features not yet merged to TT-Metal's main branch." In the actual repository, `.gitmodules` specifies no `branch` field at all, the submodule HEAD resolves to `origin/main`, and no `origin/blaze-metal` branch exists on the remote (`git branch -r | grep blaze` returns no match for `blaze-metal`). The README makes the same stale claim, but the chapter should reflect the actual repository state. **Fix:** State that the submodule is pinned to a specific commit on TT-Metal (currently on `main`), and note that the features listed (C++20 kernel compilation, named CT/RT arg infrastructure, `ProgramDescriptor` API) are available at the pinned commit. Remove or qualify the `blaze-metal` branch reference as a historical artifact from the README that no longer reflects the submodule configuration.

2. **File:** `01_stack_position.md`, lines 73-78. **Error:** The `SelectByRISCV` code snippet is shown at global scope, but in the actual source (`blaze/kernels/kernel_op_api.hpp` lines 32-37) it is defined inside `namespace unified_kernels { ... }`. A developer writing `SelectByRISCV<R, W, C>` based on this snippet would get a compilation error; the correct usage is `unified_kernels::SelectByRISCV<R, W, C>`. **Fix:** Either wrap the snippet in `namespace unified_kernels { ... }` to match the source, or add a note that the alias lives in the `unified_kernels` namespace.

## Pass 2

1. **File:** `02_repository_map.md`, line 17. **Error:** The directory-tree comment reads `tt-metal/ # Git submodule -- pinned tt-metal build (blaze-metal branch)`. This contradicts `01_stack_position.md` line 64, which the reader encounters first and which correctly states that the submodule is on `main` and the `blaze-metal` reference is a historical artifact. The Pass 1 fix to `01_stack_position.md` was applied, but this second occurrence of the stale `blaze-metal` claim in the adjacent file was not updated. **Fix:** Change the comment to `tt-metal/ # Git submodule -- pinned tt-metal build` (drop the branch parenthetical, or say `(main branch)`). This aligns with the corrected text already in `01_stack_position.md`.

## Pass 3

No feedback — chapter approved.
