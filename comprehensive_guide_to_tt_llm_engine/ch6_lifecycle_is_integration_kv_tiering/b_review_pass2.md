# Agent B Review -- Chapter 6: Pass 2

## Verification of Pass 1 Fixes

### Error 1: [is_integration_patterns.md] -- Wrong function signatures for public API
**FIXED.** Lines 20-28 now show correct signatures:
- `bool push_request(const ISRequest& request);`
- `bool try_pop_response(SchedulerResponse& response);`
- `bool try_pop_output(OutputMessage& output);`
All three use bool return + reference parameter pattern.

### Error 2: [session_state_machine.md] -- Wrong function signatures in Public API table
**FIXED.** Lines 240-242 now show correct bool-returning signatures with reference parameters. The "spin-waits if full" claim for `push_request` is replaced with "returns `false` if queue is full."

### Error 3: [session_state_machine.md] -- "All methods are lock-free" claim
**FIXED.** Line 246 now reads: "All methods are non-blocking (they return immediately rather than waiting for completion). The underlying `BoundedQueue` uses `std::mutex` internally ('Not lock-free -- acceptable for the API path')."

### Error 4: [index.md] -- "three lock-free queues"
**FIXED.** Line 22 now reads: "three bounded mutex-based queues" with the source quote ("Not lock-free -- acceptable for the API path" -- `bounded_queue.hpp`).

### Error 5: [is_integration_patterns.md] -- Queues described as "lock-free"; request_queue spin-wait claim
**PARTIALLY FIXED.** Line 42 is now correctly "three bounded mutex-based queues" with the source quote. Line 61 correctly says `push_request` "returns `false` (caller must retry)." However, line 7 in the Overview paragraph still reads: "The decode scheduler exposes a narrow, **lock-free** interface that the IS integrates against." This is a residual occurrence of the incorrect "lock-free" characterization that was not caught during the fix.

### Error 6: [deferred_cancellation.md] -- Fabricated "Generation counter filtering" paragraph
**FIXED.** The fabricated paragraph about the Writer comparing generation counters has been removed. Line 72 now explicitly states: "The Writer checks only `cancel_pending.is_set(uid)` -- it does not filter staging entries by generation counter."

### Error 7: [deferred_cancellation.md] -- Reader in_flight_count decrement memory ordering
**FIXED.** Line 85 now shows `memory_order_relaxed` (was `memory_order_release`). Line 93 description also correctly says "relaxed ordering."

### Error 8: [deferred_cancellation.md] -- CancelBitmap::clear memory ordering
**FIXED.** Line 152 now correctly states `memory_order_release` (was incorrectly described as relaxed).

### Error 9: [deferred_cancellation.md] -- Writer in_flight_count increment memory ordering
**FIXED.** Line 159 now shows `memory_order_release` (was `memory_order_relaxed`).

### Error 10: [deferred_cancellation.md] -- Reader in_flight_count decrement (duplicate of Error 7)
**FIXED.** Line 166 now shows `memory_order_relaxed` (was `memory_order_release`).

### Error 11: [deferred_cancellation.md] -- pending_count attribution
**FIXED.** Line 175 now reads: "`pending_count` is incremented by `stage()` (called by the Reader for decode loopback, or by the api_thread for disaggregated continue) *before* pushing an entry into the DecodeStaging FIFO. It is decremented by `release()` (called by the Writer after popping from the FIFO, and by the `ReaderClaim` destructor)."

### Error 12: [session_state_machine.md] -- ISRequest field defaults omitted
**FIXED.** Lines 131-133 now show defaults: `type = RequestType::ALLOCATE`, `request_id = 0`, `slot_id = INVALID_SLOT`.

### Error 13: [session_state_machine.md] -- SchedulerResponse and OutputMessage field defaults omitted
**FIXED.** SchedulerResponse (lines 145-147) now shows `request_id = 0`, `slot_id = INVALID_SLOT`, `error_code = 0`. OutputMessage (lines 161-166) now shows `slot_id = INVALID_SLOT`, `token_id = EMPTY_TOKEN`, `is_complete = false`, `ctx_exhausted = false`, `tokens_generated = 0`, `generation = 0`.

### Error 14: [deferred_cancellation.md] -- "Scheduler, Writer, Reader" naming
**FIXED.** Line 7 now reads: "three-thread pipeline (`api_thread`, `writer_thread`, `reader_thread`)" using the correct source code thread names.

### Error 15: [is_integration_patterns.md] -- "processes one request per iteration"
**FIXED.** Line 97 now reads: "The scheduler drains all available requests each iteration of its main loop (via a `while (request_queue.try_pop(req))` loop), interleaved with pipeline management."

### Error 16: [context_exhaustion.md] -- Speculative decode context exhaustion formula
**FIXED.** Lines 162-166 no longer contain the fabricated `spec_pos + k >= max_seq_len` or `k_effective` formulas. The section now correctly explains that context exhaustion is checked via `ctx_pos >= max_seq_len` where `ctx_pos` is `result.predicted_token_pos`, at the token emission level (Reader thread).

### Error 18: [context_exhaustion.md] -- "Prompt exactly fills context" detection point
**FIXED.** Line 188 now reads: "the Writer detects context exhaustion at the end of prefill (when `device_pos >= max_seq_len`). The state transitions from PREFILL directly to COMPLETE -- it never enters DECODE." The incorrect claim about "first decode step triggers" has been removed.

### Error 19: [is_integration_patterns.md] -- push_request takes const ref, not value
**FIXED.** Line 20 now shows `bool push_request(const ISRequest& request);` (const reference, bool return).

### Error 20: [deferred_cancellation.md] -- Writer cancel check code snippet
**FIXED.** Lines 61-62 now show the correct two-line pattern:
```cpp
DecodeStagingEntry entry;
if (decode_staging.try_pop(entry)) {
```
The incorrect `auto entry = staging.try_pop();` pattern has been removed.

### Error 21: [context_exhaustion.md] -- Prompt truncation location
**FIXED.** Line 190 now reads: "truncation happens at `prompt_table.store()` time (during SUBMIT processing in the api_thread), which clamps the length: `uint32_t clamped = std::min(len, max_seq_len_)`. The Writer processes whatever was stored."

### Error 22: [is_integration_patterns.md] -- Read-only accessors missing const
**FIXED.** Lines 34-35 now show `const` qualifiers:
```cpp
UserState get_user_state(uint32_t slot_id) const;
uint32_t  get_current_position(uint32_t slot_id) const;
```

### Error 23: [deferred_cancellation.md] -- ReaderClaim "see Ch2" reference
**FIXED.** Line 95 now describes ReaderClaim as "defined in `decode_scheduler.cpp` as a nested struct inside `DecodeScheduler::Impl`" without a "see Ch2 -- Lock-Free Design" reference. The only remaining "see Ch2" reference in the file is on line 48 for `CancelBitmap`, which is appropriate since CancelBitmap is indeed defined in Ch2.

## Additional Verification: IS Polling Pseudocode

The polling pseudocode in is_integration_patterns.md (lines 101-126) correctly uses the bool+reference pattern:
```
OutputMessage msg;
while (scheduler.try_pop_output(msg)) { ... }
SchedulerResponse resp;
while (scheduler.try_pop_response(resp)) { ... }
```
This matches the actual API signatures. No std::optional usage.

## NEW Errors Found

### NEW Error 1: [is_integration_patterns.md] [line 7] -- Residual "lock-free" in Overview paragraph

The overview paragraph still reads: "The decode scheduler exposes a narrow, **lock-free** interface that the IS integrates against."

This is inconsistent with the fix applied on line 42 (which correctly says "bounded mutex-based queues") and with the fixes in index.md and session_state_machine.md. The queues are explicitly not lock-free per the source comment. This is a leftover from the original text that was missed when Error 5 was fixed.

**Recommended fix:** Change "lock-free interface" to "non-blocking interface" or "narrow interface" on line 7.

## Files With No Changes Expected

- **multi_turn_continue.md**: Confirmed clean. No errors found.
- **kv_cache_tiering.md**: Confirmed clean. PLANNED/NOT IMPLEMENTED section with no source code claims.

## Summary

| Error | Status |
|-------|--------|
| Error 1 | FIXED |
| Error 2 | FIXED |
| Error 3 | FIXED |
| Error 4 | FIXED |
| Error 5 | PARTIALLY FIXED (residual "lock-free" on line 7 of is_integration_patterns.md) |
| Error 6 | FIXED |
| Error 7 | FIXED |
| Error 8 | FIXED |
| Error 9 | FIXED |
| Error 10 | FIXED |
| Error 11 | FIXED |
| Error 12 | FIXED |
| Error 13 | FIXED |
| Error 14 | FIXED |
| Error 15 | FIXED |
| Error 16 | FIXED |
| Error 18 | FIXED |
| Error 19 | FIXED |
| Error 20 | FIXED |
| Error 21 | FIXED |
| Error 22 | FIXED |
| Error 23 | FIXED |
| NEW Error 1 | Residual "lock-free" on line 7 of is_integration_patterns.md |

21 of 22 original errors are fully fixed. 1 error (Error 5) is partially fixed -- the main instances were corrected but one residual "lock-free" reference remains in the overview paragraph. 1 new error found (the same residual reference).

## Verdict: CONDITIONAL PASS

All 22 substantive errors from Pass 1 have been addressed. The one remaining issue is a single residual "lock-free" in the overview paragraph of is_integration_patterns.md (line 7) that was missed during the Error 5 fix. This is a minor consistency issue -- the correct characterization already appears two paragraphs later on line 42. Once this single word is corrected, the chapter will be fully clean.
