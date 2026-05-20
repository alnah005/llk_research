# Agent B Review: Chapter 2 — Pass 1

1. **File:** `threading_model.md`, ~line 145 (Thread Interaction Matrix, `decode_staging.fifo` row)
   **Error:** The matrix shows the API Thread column as `--` for `decode_staging.fifo`, implying the API thread never accesses the FIFO. In the source (`decode_scheduler.cpp:829`), `handle_disaggregated_continue()` calls `decode_staging.stage(uid, ...)`, which calls `fifo.try_push()`. The same omission applies to the `decode_staging.pending_count` row (line 146), where the API thread also calls `pending_count.fetch_add` via `stage()`. This contradicts the chapter's own `decode_staging.md` (line 224), which correctly documents that `stage()` is called by the "API thread (disaggregated continue)".
   **Fix:** In the Thread Interaction Matrix, change the API Thread cell for `decode_staging.fifo` from `--` to `W: stage/try_push (mutex) (disaggregated continue)`, and add a corresponding entry for `decode_staging.pending_count`.

No other correctness issues found. All data structure layouts, memory orderings, capacities, constants, code snippets, and algorithmic descriptions match the source code.

---

## Change Log (Applied by Agent A)

### Item 1 — FIXED
- **File:** `threading_model.md`, Thread Interaction Matrix
- **Change:** Updated `decode_staging.fifo` API Thread cell from `--` to `W: stage/try_push (mutex) (disaggregated continue)`. Updated `decode_staging.pending_count` API Thread cell from `--` to `W: fetch_add via stage (acq_rel) (disaggregated continue)`. Now consistent with decode_staging.md's "Who Calls What" table documenting that `stage()` is called by the API thread for disaggregated continue.

---
# Agent B Review: Chapter 2 — Pass 2

1. **File:** `threading_model.md`, ~line 151 (Thread Interaction Matrix, `response_queue` row) and ~line 156 (Key observations)
   **Error:** The matrix shows the Writer Thread column as `--` for `response_queue`. In the source (`decode_scheduler.cpp`), `writer_loop()` calls `maybe_finalize_cleanup(uid)` at lines 235 and 276 (both the decode-staging cancel path and the prefill cancel path). When the Writer wins the cleanup CAS, `maybe_finalize_cleanup` calls `push_cancel_ack()`, which pushes to `response_queue`. The same omission propagates to the key observations paragraph (line 156), which lists the Writer's mutex accesses as only `decode_staging.fifo.try_pop`, `prefill_queue`, and `output_queue.try_push` — omitting `response_queue.try_push` via the cleanup path.
   **Fix:** In the Thread Interaction Matrix, change the Writer Thread cell for `response_queue` from `--` to `W: try_push via push_cancel_ack (mutex) (cleanup winner, rare)`. In the key observations, add `response_queue.try_push` to the Writer's mutex access list (noting it is rare, cleanup-winner only).

2. **File:** `user_table.md`, ~line 149 (UserState FSM transition table, first row)
   **Error:** The table row `| INACTIVE | PREFILL | SUBMIT or CONTINUE request | API handler |` claims CONTINUE triggers the INACTIVE-to-PREFILL transition. In the source, CONTINUE is only processed for COMPLETE-state slots: `handle_local_continue()` (line 837) transitions COMPLETE->PREFILL, and `handle_disaggregated_continue()` (line 813) transitions COMPLETE->DECODE. Neither operates on INACTIVE slots. The FSM diagram immediately above the table correctly labels this transition as "ALLOCATE (api) + SUBMIT (api)" without mentioning CONTINUE, so the table contradicts the diagram.
   **Fix:** Change the trigger in the first data row from `SUBMIT or CONTINUE request` to `SUBMIT request` (or `ALLOCATE + SUBMIT` to match the diagram).

---
## Agent A Change Log — Pass 2
- [threading_model.md] Fixed Thread Interaction Matrix: Writer can push to response_queue via push_cancel_ack; updated Key observations
- [user_table.md] Fixed FSM transition table: INACTIVE→PREFILL triggered by ALLOCATE+SUBMIT, not SUBMIT or CONTINUE

---
# Agent B Review: Chapter 2 — Pass 3

Pass 2 fixes verified correct. Three remaining factual errors found in the Thread Interaction Matrix and BoundedQueue table:

1. **File:** `threading_model.md`, line 123 (Thread Interaction Matrix, `free_ids` row)
   **Error:** The matrix shows `free_ids` access as: API = `W: allocate (CAS, acq_rel)`, Writer = `--`, Reader = `W: free (fetch_or, release)`. In the source, `free_ids.free(uid)` is called from `maybe_finalize_cleanup()` (line 739 of `decode_scheduler.cpp`), which can be called by any of the three threads -- the Writer (lines 236, 276), the Reader (via ReaderClaim destructor, line 365), and the API thread (line 704). Additionally, the API thread calls `free_ids.free(uid)` directly in the synchronous INACTIVE-slot CANCEL path (line 700). The Writer column should not be `--`, and the API column should include `free`. The chapter's own `free_id_pool.md` (lines 93-101) correctly documents that all three threads can call `free()`, so the matrix contradicts the per-structure documentation.
   **Fix:** Change the `free_ids` row to: API = `W: allocate (CAS, acq_rel); W: free (fetch_or, release) (sync cancel + cleanup winner)`, Writer = `W: free (fetch_or, release) (cleanup winner, rare)`, Reader = `W: free (fetch_or, release) (cleanup winner)`.

2. **File:** `threading_model.md`, line 143 (Thread Interaction Matrix, `cancel_pending` row)
   **Error:** The Reader column shows `R: is_set (load, acquire); W: clear in cleanup (fetch_and, release)`. In the source, `cancel_pending.clear(uid)` is called only by the API thread: during ALLOCATE (line 633) and during synchronous CANCEL cleanup of an INACTIVE slot (line 699). It is NOT called in `maybe_finalize_cleanup()` -- the cancel bit remains set after asynchronous cleanup and is only cleared on the next ALLOCATE. The chapter's own `prompt_table_and_cancel_bitmap.md` (lines 126-128) explicitly states: "Important: `clear()` is NOT called during `maybe_finalize_cleanup()`." The matrix contradicts this.
   **Fix:** Change the Reader column for `cancel_pending` from `R: is_set (load, acquire); W: clear in cleanup (fetch_and, release)` to `R: is_set (load, acquire)`. Move the `W: clear` annotation to the API Thread column: `W: mark (fetch_or, release); W: clear on ALLOCATE/sync cancel (fetch_and, release)`.

3. **File:** `prefill_queue_and_bounded_queue.md`, line 157 (Queue Instances and Capacities table, `decode_staging.fifo` row)
   **Error:** The Producer column says `Reader thread (stage)`. In the source, `decode_staging.stage()` is also called by the API thread via `handle_disaggregated_continue()` (line 829). This was already fixed in the Thread Interaction Matrix (Pass 1), and `decode_staging.md` line 224 correctly lists the API thread as a caller of `stage()`, but the BoundedQueue capacity table was not updated to match.
   **Fix:** Change the Producer column for `decode_staging.fifo` from `Reader thread (stage)` to `Reader thread (stage), API thread (disaggregated continue)`.

---
## Agent A Change Log — Pass 3
- [threading_model.md] Fixed free_ids row: all three threads can call free() via maybe_finalize_cleanup; API also calls free() directly in sync CANCEL
- [threading_model.md] Fixed cancel_pending row: clear() is API-thread-only (ALLOCATE + sync CANCEL), not Reader; Reader only reads via is_set
- [prefill_queue_and_bounded_queue.md] Fixed decode_staging.fifo Producer: added API thread (disaggregated continue)

---
# Agent B Review: Chapter 2 — Pass 4

Pass 3 fixes verified correct. Two new errors found:

1. **File:** `free_id_pool.md`, line 79
   **Error:** Sentence omits Writer thread from callers of `free()`. Writer can win cleanup CAS via `maybe_finalize_cleanup()` (decode_scheduler.cpp lines 235, 276). Contradicts both source code and same file's own correct listing at lines 95-97.
   **Fix:** Changed to list all three threads as potential callers.

2. **File:** `index.md`, line 76
   **Error:** PrefillQueue Hot-Path Alloc says "Rare (SUBMIT only)" but `push()` is also called during local CONTINUE (decode_scheduler.cpp line 851).
   **Fix:** Changed to "Rare (SUBMIT/CONTINUE)".

---
## Agent A Change Log — Pass 4
- [free_id_pool.md] Fixed free() caller list to include Writer thread alongside Reader and API
- [index.md] Fixed PrefillQueue hot-path alloc to include CONTINUE alongside SUBMIT

---
# Agent B Review: Chapter 2 — Pass 5

Pass 4 fixes verified correct. Two new errors found:

1. **File:** `threading_model.md`, line 124 (Thread Interaction Matrix, `user_table.state` row, Writer Thread column)
   **Error:** Writer column says `R: load (acquire)` but Writer stores COMPLETE (line 304) and DECODE (line 337) with release, and performs CAS in cleanup. Contradicts same file's FSM table.
   **Fix:** Changed to `RW: store (release), CAS in cleanup (acq_rel)`.

2. **File:** `user_table.md`, line 44 (Full Field Catalog, `state` row, Writer column)
   **Error:** Writer column says `read (acquire)` — same error. Writer performs store and CAS.
   **Fix:** Changed to `write (store release), CAS in cleanup (acq_rel)`.

---
## Agent A Change Log — Pass 5
- [threading_model.md] Fixed user_table.state Writer column: RW with store(release) and CAS, not read-only
- [user_table.md] Fixed state field Writer column in Full Field Catalog to match actual write access

---
# Agent B Review: Chapter 2 — Pass 6

Pass 5 fixes verified correct. Five new errors found in Thread Interaction Matrix:

1. **File:** `threading_model.md` — `user_table.in_flight_count` API column: missing load(acquire) in cancel/cleanup paths
2. **File:** `threading_model.md` — `user_table.current_position` API column: missing release store on disaggregated continue and relaxed load on local continue
3. **File:** `threading_model.md` + `user_table.md` — `user_table.temperature` Reader column: marked `--` but Reader reads via device_sampling_params in check_acceptance
4. **File:** `threading_model.md` — `spec_state` Writer column: marked `--` but Writer can call reset via maybe_finalize_cleanup cleanup winner
5. **File:** `threading_model.md` — `prompt_table` Writer column: `R: get_token` but Writer also calls get_length (read) and clear (write)

---
## Agent A Change Log — Pass 6
- [threading_model.md] Fixed in_flight_count API column: added load(acquire) in cancel/cleanup
- [threading_model.md] Fixed current_position API column: added disaggregated continue store and local continue load
- [threading_model.md + user_table.md] Fixed temperature Reader column: Reader reads via device_sampling_params
- [threading_model.md] Fixed spec_state Writer column: cleanup winner can call reset
- [threading_model.md] Fixed prompt_table Writer column: Writer also calls get_length and clear

---
# Agent B Review: Chapter 2 — Pass 7

Pass 6 fixes verified correct. Four new errors found:

1. **File:** `threading_model.md` — `prefill_pos` Writer column: `R` but Writer writes this field. Changed to `RW`.
2. **File:** `threading_model.md` — `prefill_in_flight` and `post_complete_in_flight` API columns: `--` but API writes via reset(). Changed to `W: store 0 on init (relaxed)`.
3. **File:** `threading_model.md` — `in_thinking_phase` Reader column: missing load(acquire) in check_acceptance. Changed to `RW`.
4. **File:** `threading_model.md` — Reader loop step 4: "zero out in-flight counters" overstates; only prefill_in_flight and post_complete_in_flight are zeroed.

---
## Agent A Change Log — Pass 7
- [threading_model.md] Fixed prefill_pos Writer column: RW not R
- [threading_model.md] Fixed prefill_in_flight and post_complete_in_flight API columns: added store 0 on init
- [threading_model.md] Fixed in_thinking_phase Reader column: added load(acquire) from check_acceptance
- [threading_model.md] Fixed Reader loop cancel step: precise list of zeroed counters
