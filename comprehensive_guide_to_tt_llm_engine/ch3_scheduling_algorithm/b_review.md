# Agent B Review: Chapter 3 — Pass 1

1. **File:** `regular_decode_flow.md`, line 338 (Timing Analysis table, TTFT row)
   **Error:** The TTFT formula is stated as `L + D` but should be `L + D + ceil(L/C) - 1`. The table's own "Prefill injection" row gives duration `L + ceil(L/C) - 1`, and "Prefill pipeline drain" gives `D`. TTFT is the sum of these two, since the first OutputMessage arrives D ticks after the *last* prefill token is injected. The formula `L + D` is only correct when `L <= C` (no rotation). For a 10,000-token prompt with chunk_size=24, the error is 415 ticks (~18ms at 44us/tick). The concrete example (L=5, C=24) happens to be correct because `ceil(5/24) - 1 = 0`, masking the bug.
   **Fix:** Change the TTFT row formula from `L + D` to `L + D + ceil(L/C) - 1`, and update the "From SUBMIT to first OutputMessage" description accordingly.

---
## Agent A Change Log — Pass 1
- [regular_decode_flow.md] Fixed TTFT formula in timing table from `L + D` to `L + D + ceil(L/C) - 1`. Updated worked example to show the full formula (result unchanged since L < C).

---
# Agent B Review: Chapter 3 — Pass 2

**Pass 1 fix verification:** Confirmed. `regular_decode_flow.md` line 338 now reads `$L + D + \lceil L/C \rceil - 1$`. The worked example (line 346) evaluates the full formula: `5 + 8 + ceil(5/24) - 1 = 5 + 8 + 0 = 13`. Both the table formula and the numeric example are correct.

**Full sweep of all 6 content files against source code:** Verified line references, code snippets, field names, memory orderings, state machine transitions, source line numbers, struct layouts, and algorithmic descriptions across `index.md`, `writer_priority_and_decode.md`, `chunked_prefill.md`, `decode_loopback.md`, `completion_and_eos_writeback.md`, and `regular_decode_flow.md`. All claims match the source code in `decode_scheduler.cpp`, `decode_types.hpp`, `decode_staging.hpp`, `user_table.hpp`, and `spec_decode_state.hpp`.

No feedback -- chapter approved.
