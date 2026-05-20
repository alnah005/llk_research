# Agent B Review: Chapter 4 — Pass 1

1. **File:** `thinking_phase_and_relaxed_acceptance.md`, ~line 92
   **Error:** The guide states that `p_indices` in `ResultDescriptor` is `std::array<uint16_t, 32>`. The actual type in `pipeline_types.hpp` line 58 is `std::array<uint32_t, 32>`. (`p_scores` is correctly described as `uint16_t`; only the `p_indices` type is wrong.)
   **Fix:** Change "`std::array<uint16_t, 32>`" to "`std::array<uint32_t, 32>`" and update the surrounding text (e.g., "capping the effective $k$ at 32" is fine, but the element type must be `uint32_t`).

2. **File:** `state_machine_detail.md`, ~line 135 (transition T11)
   **Error:** T11 lists the "Before" state as $(1,0,0)$ (SPECULATING). The CONTINUE code path requires `has_verified[uid]` to be true (source line 574: `if (ss.has_verified[uid])`), so the before state must be $(0,1,0)$ (ACCEPTED).
   **Fix:** Change the Before column for T11 from `(1,0,0)` to `(0,1,0)`.

---
## Agent A Change Log — Pass 1
- [thinking_phase_and_relaxed_acceptance.md] Fixed p_indices type from uint16_t to uint32_t
- [state_machine_detail.md] Fixed T11 before state from (1,0,0) to (0,1,0)

---
# Agent B Review: Chapter 4 — Pass 2 (Final Sweep)

## Pass 1 Fix Verification

1. `thinking_phase_and_relaxed_acceptance.md` line 92: Confirmed. Now reads `std::array<uint32_t, 32>`, matching `pipeline_types.hpp` line 58.
2. `state_machine_detail.md` line 135 (T11): Confirmed. Before state is now `(0,1,0)`, matching the CONTINUE code path requirement (`has_verified=true`).

## Final Sweep (all 6 files vs source code)

No feedback -- chapter approved.
