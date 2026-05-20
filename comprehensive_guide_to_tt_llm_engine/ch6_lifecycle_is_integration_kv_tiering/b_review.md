# Agent B Review -- Chapter 6: Pass 1

## Errors Found

### Error 1: [is_integration_patterns.md] [lines 17-29] -- Wrong function signatures for public API

- **What the chapter says**: The chapter presents the public API as:
  ```cpp
  void push_request(ISRequest req);
  std::optional<SchedulerResponse> try_pop_response();
  std::optional<OutputMessage> try_pop_output();
  ```
  It claims `push_request` returns `void` and "spin-waits if request_queue is full," and that `try_pop_response`/`try_pop_output` return `std::optional`.

- **What the source code says**: In `decode_scheduler.hpp` lines 31-33 and `decode_scheduler.cpp` lines 868-872:
  ```cpp
  bool push_request(const ISRequest& request);
  bool try_pop_response(SchedulerResponse& response);
  bool try_pop_output(OutputMessage& output);
  ```
  `push_request` returns `bool` (from `try_push`, which returns false if the queue is full -- it does NOT spin-wait). `try_pop_response` and `try_pop_output` take an output reference parameter and return `bool`, not `std::optional`.

- **Fix**: Correct all three signatures. `push_request` returns `bool` and does not spin-wait (it is a single call to `request_queue.try_push()`). The try_pop functions take a reference parameter and return `bool`. Update the "Full Behavior" column for request_queue from "push_request spin-waits" to "push_request returns false."

---

### Error 2: [session_state_machine.md] [lines 240-242] -- Wrong function signatures for public API

- **What the chapter says**: The table describes:
  - `push_request(ISRequest)` as "Enqueue a request (any type); spin-waits if full"
  - `try_pop_response()` as "returns `std::optional`"
  - `try_pop_output()` as "returns `std::optional`"

- **What the source code says**: Same as Error 1. `push_request` returns `bool` (non-blocking). `try_pop_response` and `try_pop_output` take a reference and return `bool`.

- **Fix**: Correct signatures and remove the "spin-waits if full" claim. `push_request` returns false if the queue is full. `try_pop_response(SchedulerResponse& response)` returns `bool`. `try_pop_output(OutputMessage& output)` returns `bool`.

---

### Error 3: [session_state_machine.md] [line 246] -- "All methods are lock-free" is incorrect

- **What the chapter says**: "All methods are lock-free."

- **What the source code says**: `BoundedQueue` (bounded_queue.hpp lines 23, 29, 39) uses `std::mutex` with `std::lock_guard` in every operation (`try_push`, `try_pop`, `empty`, `size`). These queues are mutex-based, not lock-free.

- **Fix**: Change "All methods are lock-free" to something like "All methods are non-blocking (they return immediately rather than waiting for completion) but the underlying queues use a mutex internally."

---

### Error 4: [index.md] [line 22] -- "three lock-free queues" is incorrect

- **What the chapter says**: "The IS and PM communicate through exactly three lock-free queues."

- **What the source code says**: `BoundedQueue` is mutex-based (see `bounded_queue.hpp` line 23: `mutable std::mutex mtx`). The comment in `bounded_queue.hpp` line 15 explicitly states: "Not lock-free -- acceptable for the API path."

- **Fix**: Change "lock-free" to "bounded" or "mutex-based bounded" queues. Quote the source: "Not lock-free -- acceptable for the API path."

---

### Error 5: [is_integration_patterns.md] [lines 42, 60-63] -- Queues described as "lock-free"

- **What the chapter says**: Lines 42: "All communication between the IS and the decode scheduler flows through three lock-free queues." Line 60 table header "Full Behavior" states `push_request` spin-waits.

- **What the source code says**: The queues are mutex-based. `push_request` returns `bool` without spinning.

- **Fix**: Remove "lock-free" characterization. For request_queue full behavior: "returns false (caller must retry)."

---

### Error 6: [deferred_cancellation.md] [lines 69-71] -- Writer does NOT check generation counters

- **What the chapter says**: "**Generation counter filtering:** Even before checking `cancel_pending`, the Writer compares each staging entry's generation counter against the current generation for that slot. If they differ, the entry is stale (from a prior session) and is discarded unconditionally."

- **What the source code says**: In `decode_scheduler.cpp` lines 224-269 (the `writer_loop` decode path), there is NO generation counter comparison. The Writer pops an entry from `decode_staging.try_pop(entry)`, then checks only `cancel_pending.is_set(uid)`. There is no code that compares `entry.generation` against `decode_staging.generation[uid]`. The Writer either processes the entry or discards it based on the cancel bit.

- **Fix**: Remove the generation counter filtering claim for the Writer. The generation counter is used in `OutputMessage` for IS-side filtering, and `advance_generation` invalidates tokens for IS consumption, but the Writer itself does not filter by generation.

---

### Error 7: [deferred_cancellation.md] [lines 83-84] -- Reader cancel path memory ordering wrong for in_flight_count

- **What the chapter says**: The code snippet shows:
  ```cpp
  in_flight_count[uid].fetch_sub(1, std::memory_order_release);
  ```

- **What the source code says**: In `decode_scheduler.cpp` line 381:
  ```cpp
  user_table.in_flight_count[uid].fetch_sub(1, std::memory_order_relaxed);
  ```
  The Reader's `in_flight_count` decrement uses `memory_order_relaxed`, not `memory_order_release`.

- **Fix**: Change `memory_order_release` to `memory_order_relaxed` in the Reader's `in_flight_count` decrement.

---

### Error 8: [deferred_cancellation.md] [lines 148-149] -- CancelBitmap::clear uses release, not relaxed

- **What the chapter says**: "`clear()` uses `fetch_and` with relaxed ordering because it is only called after the cleanup CAS has been won, at which point no other thread can be racing on this slot."

- **What the source code says**: In `user_table.hpp` line 155:
  ```cpp
  bits[w].fetch_and(~(uint64_t(1) << bit), std::memory_order_release);
  ```
  `CancelBitmap::clear()` uses `memory_order_release`, not relaxed.

- **Fix**: Change "relaxed ordering" to "`memory_order_release`."

---

### Error 9: [deferred_cancellation.md] [lines 156-159] -- Writer in_flight_count increment uses release, not relaxed

- **What the chapter says**: The code snippet shows:
  ```cpp
  // Writer: before inject
  uint32_t delta = do_spec ? 2 : 1;
  in_flight_count[uid].fetch_add(delta, std::memory_order_relaxed);
  ```

- **What the source code says**: In `decode_scheduler.cpp` line 244:
  ```cpp
  user_table.in_flight_count[uid].fetch_add(do_spec ? 2 : 1, std::memory_order_release);
  ```
  The Writer uses `memory_order_release`, not `memory_order_relaxed`.

- **Fix**: Change `memory_order_relaxed` to `memory_order_release`.

---

### Error 10: [deferred_cancellation.md] [lines 163-165] -- Reader in_flight_count decrement uses relaxed, not release

- **What the chapter says**:
  ```cpp
  // Reader: on read_result
  in_flight_count[uid].fetch_sub(1, std::memory_order_release);
  ```

- **What the source code says**: In `decode_scheduler.cpp` line 381:
  ```cpp
  user_table.in_flight_count[uid].fetch_sub(1, std::memory_order_relaxed);
  ```
  The Reader uses `memory_order_relaxed`, not `memory_order_release`.

- **Fix**: Change `memory_order_release` to `memory_order_relaxed`.

---

### Error 11: [deferred_cancellation.md] [lines 173-175] -- pending_count attribution is reversed (Reader vs Writer)

- **What the chapter says**: "**FIFO entry tracking.** The Reader increments `pending_count` *before* pushing an entry into the DecodeStaging FIFO. The Writer decrements it *after* popping and processing (or discarding) the entry."

- **What the source code says**: In `decode_staging.hpp` lines 60-66, the `stage()` function (called by the Reader and api_thread) increments `pending_count` before pushing into the FIFO. In `decode_staging.hpp` line 84, the `release()` function (called by the Writer after popping AND by the Reader via ReaderClaim) decrements `pending_count`. But `stage()` is also called by `handle_disaggregated_continue` (api_thread) and by the Writer's prefill path indirectly through the non-spec decode path in the Reader. The description is directionally correct but could be clearer -- the key issue is the claim that "The Reader increments" when actually `stage()` is called by whatever thread is staging (usually the Reader thread for decode loopback). The framing is acceptable but mislabels which thread does what for the FIFO pop: both Writer AND Reader call `release()`.

- **Fix**: Clarify: "`pending_count` is incremented by `stage()` (called by the Reader for decode loopback, or by the api_thread for disaggregated continue) and decremented by `release()` (called by the Writer after popping from the FIFO, and by the ReaderClaim destructor)."

---

### Error 12: [session_state_machine.md] [lines 130-136] -- ISRequest field defaults omitted

- **What the chapter says**: The ISRequest struct is shown without default initializers:
  ```cpp
  struct ISRequest {
      RequestType type;           // ...
      uint32_t    request_id;     // ...
      uint32_t    slot_id;        // ...
      std::vector<uint32_t> tokens;
      GenerationParams gen;
  };
  ```

- **What the source code says**: In `decode_types.hpp` lines 78-84, all fields have defaults:
  ```cpp
  struct ISRequest {
      RequestType type = RequestType::ALLOCATE;
      uint32_t request_id = 0;
      uint32_t slot_id = INVALID_SLOT;
      std::vector<uint32_t> tokens;
      GenerationParams gen;
  };
  ```

- **Fix**: Add default initializers to match source: `type = RequestType::ALLOCATE`, `request_id = 0`, `slot_id = INVALID_SLOT`.

---

### Error 13: [session_state_machine.md] [lines 143-148] -- SchedulerResponse and OutputMessage field defaults omitted

- **What the chapter says**: Both structs are shown without default initializers.

- **What the source code says**: In `decode_types.hpp`:
  - `SchedulerResponse`: `request_id = 0`, `slot_id = INVALID_SLOT`, `error_code = 0`
  - `OutputMessage`: `slot_id = INVALID_SLOT`, `token_id = EMPTY_TOKEN`, `is_complete = false`, `ctx_exhausted = false`, `tokens_generated = 0`, `generation = 0`

- **Fix**: Add default initializers to match source. While omitting defaults may be a style choice, the source defines explicit defaults, and they convey important information (e.g., `slot_id` defaults to `INVALID_SLOT`, not 0).

---

### Error 14: [deferred_cancellation.md] [line 7] -- Three-thread pipeline incorrectly names "Scheduler" as a pipeline thread

- **What the chapter says**: "The system is a three-thread pipeline (Scheduler, Writer, Reader)"

- **What the source code says**: In `decode_scheduler.cpp` lines 73-75, the three threads are `writer_thread`, `reader_thread`, and `api_thread`. The code and comments consistently call the third thread `api_thread` / `api_loop`. "Scheduler" is ambiguous -- it could mean the api_thread or the overall DecodeScheduler class.

- **Fix**: Minor -- consider saying "(API handler, Writer, Reader)" or "(api_thread, writer_thread, reader_thread)" for precision. This is more of a naming-consistency issue than a factual error.

---

### Error 15: [is_integration_patterns.md] [line 97] -- "Scheduler processes one request per iteration" is incorrect

- **What the chapter says**: "The scheduler processes one request per iteration of its main loop, interleaved with pipeline management."

- **What the source code says**: In `decode_scheduler.cpp` lines 623-711, `handle_api_requests()` uses a `while` loop (`while (request_queue.try_pop(req))`) to drain all available requests in a single call, not just one per iteration.

- **Fix**: Change to "The scheduler drains all available requests each iteration of its main loop."

---

### Error 16: [context_exhaustion.md] [lines 162-170] -- Speculative decode context exhaustion formula is not from source

- **What the chapter says**: Claims specific formulae for speculative position checking and truncation:
  ```
  spec_pos + k >= max_seq_len
  k_effective = min(k, max_seq_len - spec_pos)
  ```

- **What the source code says**: In `decode_scheduler.cpp` lines 450-496 (`emit_token` lambda), the context exhaustion check is simply `ctx_pos >= params.max_seq_len` where `ctx_pos` is `result.predicted_token_pos` (line 516) for spec-decode users. There is no explicit "truncation of speculation length" formula -- the check is done at the token emission level, not at the batch planning level.

- **Fix**: Clarify that context exhaustion with speculative decode is checked via `ctx_pos >= max_seq_len` where `ctx_pos` is the `predicted_token_pos` from the result (the farthest-ahead position). The specific truncation formula is not present in the source. Remove the `k_effective` formula or note it as an IS-side consideration rather than a PM-side mechanism.

---

### Error 17: [multi_turn_continue.md] [line 30] -- GenerationParams.top_k default value is wrong

- **What the chapter says**: Shows `int32_t top_k = 1;` in the GenerationParams struct.

- **What the source code says**: In `decode_types.hpp` line 73: `int32_t top_k = 1;`

- **Fix**: Actually, the chapter is CORRECT here. `top_k = 1` is the default. (Self-correction: no error.)

---

### Error 18: [context_exhaustion.md] [lines 193-198] -- Edge case "Prompt exactly fills context" claim

- **What the chapter says**: "If the initial prompt is exactly `max_seq_len` tokens, prefill succeeds (all tokens fit), but the first decode step immediately triggers context exhaustion. The OutputMessage has `tokens_generated = 0` and `ctx_exhausted = true`."

- **What the source code says**: In `decode_scheduler.cpp` lines 291-309, the Writer checks `device_pos >= params.max_seq_len` DURING prefill. If the prompt is exactly `max_seq_len` tokens, then after processing the last prefill token, `device_pos` will equal `max_seq_len`. At that point `prompt_idx >= prompt_len` is also true (both conditions are checked at line 291). The Writer marks COMPLETE from PREFILL directly, before ever entering DECODE. The OutputMessage comes from the Writer's prefill-complete path (lines 296-309) with `tokens_generated = 0` and `ctx_exhausted = true` only if `device_pos >= max_seq_len`. But actually, re-reading the check: `prompt_idx >= prompt_len` would be true when all tokens are consumed, and `device_pos >= params.max_seq_len` would only be true if the position reached max_seq_len. For a prompt of exactly max_seq_len tokens starting from position 0, after the last token device_pos = max_seq_len. The check at line 291 is `prompt_idx >= prompt_len || device_pos >= params.max_seq_len`. Both conditions are true. The Writer pops from prefill_queue, clears prompt_table, and checks `ctx_exhausted = (device_pos >= params.max_seq_len)` which is true. So state goes PREFILL -> COMPLETE. It never enters DECODE. The chapter says "prefill succeeds... but the first decode step immediately triggers context exhaustion" which is incorrect -- the Writer catches it at the end of prefill, not during a decode step. But then the chapter under "Worked Example B" correctly describes the Writer-side detection. The edge case description is inconsistent with the earlier explanation.

- **Fix**: Clarify: "If the initial prompt is exactly `max_seq_len` tokens, the Writer detects context exhaustion at the end of prefill (when `device_pos >= max_seq_len`). The state transitions from PREFILL directly to COMPLETE (never entering DECODE). The OutputMessage has `tokens_generated = 0` and `ctx_exhausted = true`."

---

### Error 19: [is_integration_patterns.md] [line 19] -- push_request takes ISRequest by value, not const ref

- **What the chapter says**: `void push_request(ISRequest req);` (pass by value)

- **What the source code says**: In `decode_scheduler.hpp` line 31: `bool push_request(const ISRequest& request);` (const reference, returns bool)

- **Fix**: Change to `bool push_request(const ISRequest& request);`

---

### Error 20: [deferred_cancellation.md] [lines 61-67] -- Writer cancel check code snippet is misleading

- **What the chapter says**: Shows code as:
  ```cpp
  auto entry = staging.try_pop();
  if (cancel_pending.is_set(entry.slot_id)) {
      staging.release(entry.slot_id);
      maybe_finalize_cleanup(entry.slot_id);
      continue;
  }
  ```

- **What the source code says**: In `decode_scheduler.cpp` lines 224-237:
  ```cpp
  DecodeStagingEntry entry;
  if (decode_staging.try_pop(entry)) {
      uint32_t uid = entry.slot_id;
      if (cancel_pending.is_set(uid)) {
          decode_staging.release(uid);
          maybe_finalize_cleanup(uid);
          continue;
      }
  ```
  `try_pop` takes an output reference parameter and returns `bool` -- it doesn't return the entry directly. This is a meaningful API difference (the return indicates success/failure, not the entry itself).

- **Fix**: Show the correct two-line pattern: `DecodeStagingEntry entry; if (decode_staging.try_pop(entry)) { ... }`

---

### Error 21: [context_exhaustion.md] [line 195] -- "Prompt exceeds context" truncation behavior

- **What the chapter says**: "If the IS sends a prompt longer than `max_seq_len`, the Writer detects this during the first prefill chunk and truncates."

- **What the source code says**: In `user_table.hpp` (PromptTable) lines 113-119, `store()` clamps the length: `uint32_t clamped = std::min(len, max_seq_len_);`. Truncation happens at storage time (in the api_thread during SUBMIT/CONTINUE), not during the Writer's prefill processing. The Writer processes whatever was stored by the api_thread.

- **Fix**: Clarify that prompt truncation happens at `prompt_table.store()` time (during SUBMIT processing in the api_thread), not during the Writer's prefill execution.

---

### Error 22: [is_integration_patterns.md] [lines 33-36] -- Read-only accessor signatures missing const

- **What the chapter says**:
  ```cpp
  UserState get_user_state(uint32_t slot_id);
  uint32_t  get_current_position(uint32_t slot_id);
  ```

- **What the source code says**: In `decode_scheduler.hpp` lines 36, 42:
  ```cpp
  UserState get_user_state(uint32_t slot_id) const;
  uint32_t get_current_position(uint32_t slot_id) const;
  ```
  Both are `const` member functions.

- **Fix**: Add `const` qualifier to both signatures.

---

### Error 23: [deferred_cancellation.md] [lines 7, 93, etc.] -- "ReaderClaim" description references Ch2 but it is in decode_scheduler.cpp

- **What the chapter says**: Multiple references say "see Ch2 -- Lock-Free Design" or "see Ch2 for `CancelBitmap`" for `ReaderClaim`.

- **What the source code says**: `ReaderClaim` is a nested struct inside `DecodeScheduler::Impl` in `decode_scheduler.cpp` lines 357-369. It is not part of Ch2's data structures (UserTable, CancelBitmap, etc.). It is implementation-internal to the scheduler.

- **Fix**: This is a minor cross-reference issue. If Ch2 does not cover ReaderClaim, remove the "see Ch2" reference for it and note that ReaderClaim is defined in the scheduler implementation.

---

## No Issues Found In

- **kv_cache_tiering.md** -- Marked as PLANNED/NOT IMPLEMENTED. No source code claims to verify. The design discussion is speculative by nature.
- **multi_turn_continue.md** -- Code snippets and field assignments match the source correctly (with the exception of coverage by Error 17 which was self-corrected). The `handle_local_continue` and `handle_disaggregated_continue` logic matches.

## Summary

22 errors found across 5 files (index.md, session_state_machine.md, is_integration_patterns.md, deferred_cancellation.md, context_exhaustion.md). The most impactful errors are:

1. **Wrong function signatures** (Errors 1, 2, 19, 22): `push_request` returns `bool` and does not spin-wait; `try_pop_*` use reference parameters and return `bool`, not `std::optional`.
2. **False "lock-free" claims** (Errors 3, 4, 5): `BoundedQueue` is explicitly mutex-based per its own source comment.
3. **Wrong memory orderings** (Errors 7, 8, 9, 10): Three memory orderings are swapped (relaxed vs. release) across the deferred_cancellation chapter's code snippets.
4. **Fabricated Writer generation-counter filtering** (Error 6): The Writer does NOT filter staging entries by generation counter -- it only checks `cancel_pending`.
5. **Wrong API processing model** (Error 15): The api_loop drains ALL queued requests per iteration, not one.
