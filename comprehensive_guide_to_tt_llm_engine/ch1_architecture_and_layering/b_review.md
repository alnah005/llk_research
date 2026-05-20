# Agent B Review: Chapter 1 — Pass 1

### Item 1: [key_invariants.md] ~lines 73-79
**Error:** The five-step cleanup sequence listed for `maybe_finalize_cleanup()` does not match the actual implementation. The chapter claims the cleanup winner executes: (1) `pipeline_->reset_kv(uid)`, (2) `user_table.reset(uid)`, (3) `spec_state.reset(uid)`, (4) `cancel_pending.clear(uid)`, (5) `free_ids.free(uid)`. The actual code in `decode_scheduler.cpp` lines 737-740 performs only: (1) `pipeline_->reset_kv(uid)`, (2) `spec_state.reset(uid)`, (3) `free_ids.free(uid)`, (4) `push_cancel_ack(request_id, uid)`. Neither `user_table.reset(uid)` nor `cancel_pending.clear(uid)` is called during cleanup -- both are deferred to the next ALLOCATE for that slot (lines 632-634). A reader implementing a similar system based on this guide would place state clearing in the wrong code path, potentially introducing races where a freshly-allocated slot still carries stale cancel marks or stale user-table fields from a previous session before the allocating thread has a chance to clear them.
**Fix:** Replace the five-step list with the actual four-step sequence: (1) `pipeline_->reset_kv(uid)`, (2) `spec_state.reset(uid)`, (3) `free_ids.free(uid)`, (4) `push_cancel_ack(request_id, uid)`. Add a note that `user_table.reset(uid)` and `cancel_pending.clear(uid)` are performed lazily at the next ALLOCATE for the slot, not during cancel cleanup.

### Item 2: [ownership_contract.md] ~line 208
**Error:** The cancellation walkthrough step 3 states that `maybe_finalize_cleanup` will "reset user table and spec state, free the ID back to `FreeIdPool`". This repeats the same incorrect claim as Item 1 -- `user_table.reset()` is not called during cleanup. It is deferred to the next ALLOCATE. A downstream reader following this walkthrough to implement cancellation logic would incorrectly clear user-table state at cleanup time.
**Fix:** Change step 3 to read: "...CAS state to INACTIVE, call `pipeline_->reset_kv(3)`, reset spec state, free the ID back to `FreeIdPool`, and push a cancel ack response. (Note: `user_table.reset()` and `cancel_pending.clear()` are deferred to the next ALLOCATE for this slot.)"

**No further items -- remaining chapter content is factually accurate against the source code.**

---
## Agent A Change Log — Pass 1
- [key_invariants.md] Fixed five-step cleanup sequence to four-step actual sequence; added lazy-reset note
- [ownership_contract.md] Fixed cancellation walkthrough step 3; added deferred reset note

---

# Agent B Review: Chapter 1 — Pass 2

Pass 1 items verified fixed:
- Item 1 (key_invariants.md cleanup sequence): Now correctly shows 4 steps matching source lines 737-740. Deferred-reset note present.
- Item 2 (ownership_contract.md cancellation walkthrough): Now correctly omits user_table.reset from cleanup. Deferred-reset note present.

### Item 1: [ownership_contract.md] Three-Queue Boundary table, lines ~138-141

**Error:** The table labels `response_queue` as "SPSC" with "PM API handler thread" as the sole producer, and labels `output_queue` as "SPSC" with "PM Reader thread" as the sole producer. Both labels are wrong:

- **response_queue** is MPSC on the producer side. The API handler thread pushes allocation responses (`decode_scheduler.cpp` line 641), but `push_cancel_ack()` (`decode_scheduler.cpp` line 753) is called from `maybe_finalize_cleanup()`, which can be won by **any** of the three PM threads (Writer, Reader, or API handler) via the CAS on user state (lines 730-734). Therefore the response_queue has up to three concurrent producer threads.

- **output_queue** is also not strictly SPSC. The Reader thread pushes decode/spec output (`decode_scheduler.cpp` lines 489, 500), but the Writer thread also pushes to it in the prefill-side context-exhaustion path (`decode_scheduler.cpp` line 306). Therefore the output_queue has two producer threads.

A downstream reader relying on these SPSC labels to select a lock-free single-producer queue implementation (which is a natural engineering choice given the label) would introduce a data race.

**Fix:** Change the Producer column for `response_queue` from "PM API handler thread (SPSC)" to "PM threads (MPSC)" or similar, noting that cancel acks can originate from any thread that wins cleanup. Change the Producer column for `output_queue` from "PM Reader thread (SPSC)" to "PM Reader + Writer threads (MPSC)" or note that the Writer pushes in the prefill-exhaustion edge case. Alternatively, keep the predominant-producer description but change the queue type column from SPSC to MPSC for both.

No further items. All other factual claims, constants, code sequences, and structural references verified correct against the source repository.

---
## Agent A Change Log — Pass 2
- [ownership_contract.md] Fixed queue concurrency labels: response_queue and output_queue changed from SPSC to MPSC

---

# Agent B Review: Chapter 1 — Pass 3

Pass 2 item verified fixed:
- Item 1 (ownership_contract.md SPSC/MPSC queue labels): The Three-Queue Boundary table at lines 137-141 now correctly labels all three queues. `request_queue` shows "IS HTTP handlers (MPSC)" as producer. `response_queue` shows "PM threads (MPSC)" as producer. `output_queue` shows "PM Reader + Writer threads (MPSC)" as producer. All three labels verified against `decode_scheduler.cpp`: the response_queue is pushed by `handle_api_requests()` (api_thread) and `push_cancel_ack()` (any thread winning the CAS in `maybe_finalize_cleanup`); the output_queue is pushed by the Reader thread (lines 489, 500) and the Writer thread (line 306, prefill context-exhaustion path). Fix is correct.

### Item 1: [key_invariants.md] ~line 80

**Error:** The text states: "The Writer checks each staging entry's generation against the current generation, silently dropping stale entries from cancelled sessions. This eliminates timing dependencies between the cancel request and the Writer's drain of the FIFO." This is factually incorrect. The Writer does NOT check the generation counter. In `writer_loop()` (decode_scheduler.cpp lines 225-237), the Writer filters stale entries by checking `cancel_pending.is_set(uid)` -- it never reads the staging entry's `generation` field or compares it against `decode_staging.generation[uid]`. The code comment at lines 228-232 explains the actual mechanism: "pending_count keeps the slot pinned across recycle, so any stale entry still in the FIFO is guaranteed to have cancel_pending set." The generation counter IS advanced on CANCEL (line 686) and IS stamped onto OutputMessage (lines 302, 475) for IS-side stale-output filtering, but the Writer's FIFO drain uses `cancel_pending`, not generation comparison.

**Fix:** Replace the sentence with: "Additionally, a generation counter per slot (`DecodeStaging::generation`) is advanced on CANCEL. The Writer drops stale staging entries by checking `cancel_pending.is_set(uid)` -- the `pending_count` mechanism guarantees that any stale entry still in the FIFO belongs to a slot whose cancel_pending bit is still set. The generation counter is stamped onto each `OutputMessage` so the IS can detect and discard late-arriving output tokens from a previous session after a cancel-and-reallocate cycle."

No further items. All other factual claims, constants, code sequences, queue labels, cleanup sequences, and structural references verified correct against the source repository.

---
## Agent A Change Log — Pass 3
- [key_invariants.md] Fixed stale-entry filtering description: Writer uses cancel_pending.is_set(uid), not generation comparison. Generation counter is for IS-side OutputMessage filtering.

---

# Agent B Review: Chapter 1 — Pass 4

Pass 3 item verified fixed:
- Item 1 (key_invariants.md stale-entry filtering): The text at line 80 now correctly states that the Writer drops stale staging entries by checking `cancel_pending.is_set(uid)`, not by generation comparison. Verified against `decode_scheduler.cpp` lines 228-235: the Writer's `writer_loop()` checks `cancel_pending.is_set(uid)` after popping from the decode staging FIFO. The generation counter's role is correctly described as being advanced on CANCEL (`decode_staging.advance_generation(uid)` at line 686) and stamped onto each `OutputMessage` (lines 302, 475) for IS-side stale-output filtering. The `pending_count` invariant explanation ("any stale entry still in the FIFO belongs to a slot whose cancel_pending bit is still set") is consistent with the code comment at lines 228-232. Fix is correct.

Final sweep completed. All chapter files verified against source:
- **key_invariants.md**: Four-step cleanup sequence matches `decode_scheduler.cpp` lines 737-740. Deferred-reset note accurately reflects that `user_table.reset()` and `cancel_pending.clear()` run in the ALLOCATE handler (line 632-634), not in `maybe_finalize_cleanup()`. Stale-entry filtering mechanism correctly attributed to `cancel_pending.is_set()`. Six invariants, hardware deployment path sizing table, and structure-to-hardware mapping all verified correct.
- **ownership_contract.md**: Three-Queue Boundary table MPSC labels correct. Cancellation walkthrough matches actual code flow. Message type fields match `decode_types.hpp`. Generation-based IS-side output filtering correctly described.
- **component_map.md**: Library split, namespace mapping, directory layout, constants, and build targets all verified correct against source headers and `CMakeLists.txt`.
- **system_position.md**: Three-layer stack, timescale annotations, bridge metaphor, and semi-hardened deployment pattern all consistent with codebase structure and `docs/overview.md` references.
- **index.md**: Navigation table and reading paths consistent with chapter content.

No feedback -- chapter approved.
