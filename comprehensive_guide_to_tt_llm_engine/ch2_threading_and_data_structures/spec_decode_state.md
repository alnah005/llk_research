# SpecDecodeState

`SpecDecodeState` tracks the per-user speculative decode verification state. It is exclusively owned by the Reader thread -- no atomics are needed. The structure lives in `src/scheduler/decode/spec_decode_state.hpp`.

## Structure

```cpp
struct SpecDecodeState {
    std::vector<uint32_t> unverified_token;
    std::vector<bool>     has_unverified;
    std::vector<bool>     has_verified;
    std::vector<bool>     signal_to_exit;
    std::vector<OutputMessage> pending_complete;
    std::vector<bool>     has_pending_complete;
};
```

All vectors are sized to `max_users` at construction. Every field is a plain (non-atomic) type, accessed only by the Reader thread during normal operation. The API thread calls `reset()` during ALLOCATE, SUBMIT, CONTINUE, and CANCEL processing, but only when the slot is not actively being read (the slot is either not yet submitted, or has been cancelled with in-flight count at zero).

## Field Semantics

### unverified_token[uid]

The predicted token from the previous decode round that has not yet been verified against the next BASE result. In the speculative decode protocol (described in Chapter 1), the pipeline predicts a token alongside each sampled result. This prediction is stored here until the next BASE result arrives and can confirm or reject it.

Initialized to `EMPTY_TOKEN`. Set by the Reader when it receives a result containing a prediction.

### has_unverified[uid]

Boolean flag indicating whether `unverified_token[uid]` contains a valid prediction awaiting verification. The Reader checks this flag when a new BASE result arrives to determine whether the result is an INITIAL (no prior speculation), ACCEPT, or REJECT.

### has_verified[uid]

Boolean flag indicating that the most recent BASE result matched the speculation (ACCEPT). When `has_verified` is true, the Reader expects the next result for this slot to be a SPEC result (the CONTINUE case, carrying a bonus token).

### signal_to_exit[uid]

Boolean flag set when a completion (EOS, max_new_tokens, context exhaustion) is detected during an ACCEPT or REJECT, but a stale SPEC/STALE result is still expected from the pipeline. The flag tells the Reader to defer publishing the completion message until the stale result is consumed.

Purpose: without this flag, the completion message would be pushed to `output_queue` immediately. The IS might then send a CONTINUE request before the stale SPEC result arrives. The CONTINUE would set up new prefill state, but the stale SPEC result would decrement `prefill_in_flight` incorrectly or collide with the new turn's tracking.

### pending_complete[uid]

An `OutputMessage` stashed when completion is detected but must be deferred (see `signal_to_exit`). Held here until the paired SPEC/STALE result arrives, at which point it is flushed to `output_queue`.

### has_pending_complete[uid]

Boolean flag indicating that `pending_complete[uid]` contains a valid deferred completion message.

## How Fields Implement ACCEPT/CONTINUE/REJECT/STALE Discrimination

The speculative decode protocol produces BASE and SPEC results in interleaved order. The Reader uses the `has_unverified` and `has_verified` flags to classify each incoming result:

### INITIAL (first decode result after prefill)

```
Condition: !has_unverified && !has_verified
Action:    Emit the actual token. Store the predicted token as unverified.
           has_unverified = true
           Stage spec pair (base + predicted) for Writer.
```

The very first decode result has no prior speculation to verify. The Reader treats it as a single-token result, stores the prediction for next time, and stages a decode pair (base + spec) for the Writer.

### ACCEPT (BASE result, prediction matched)

```
Condition: result.actual_token_type == BASE && has_unverified
           && check_acceptance(unverified_token, result)
Action:    Increment spec_accepts counter.
           Emit the actual token. has_verified = true, has_unverified = false.
           Wait for SPEC result (do not stage a new pair yet).
```

The BASE result confirmed the prediction. The Reader does not stage a new decode pair -- it waits for the paired SPEC result, which will carry a bonus token and a new prediction.

If completion is detected during ACCEPT:
```
           signal_to_exit = true
           pending_complete = output message (stashed, not published)
```

### CONTINUE (SPEC result after ACCEPT)

```
Condition: result.actual_token_type == SPEC && has_verified
Action:    Emit the bonus token from the SPEC result.
           Store the new prediction as unverified.
           has_verified = false, has_unverified = true.
           Stage a new decode pair for the Writer.
```

If `signal_to_exit` is true (accept already completed):
```
           has_unverified = false, signal_to_exit = false
           tokens_generated = 0 (reset for next turn)
           Suppress bonus token (generation already complete).
           Flush pending_complete to output_queue.
```

This is the CONTINUE-suppress path: the ACCEPT already completed generation, so the SPEC bonus token is not emitted to the user. The deferred completion message is published now that the pipeline is clean.

### REJECT (BASE result, prediction did not match)

```
Condition: result.actual_token_type == BASE && has_unverified
           && !check_acceptance(unverified_token, result)
Action:    Increment spec_rejects counter.
           Emit the actual token. Store the new prediction as unverified.
           has_unverified = true (new prediction replaces old).
           Stage a new decode pair for the Writer (correction inject).
```

If completion is detected during REJECT:
```
           signal_to_exit = true
           pending_complete = output message (stashed, not published)
           Do not stage a new pair (generation is ending).
           Wait for STALE result to drain.
```

### STALE (SPEC result after REJECT)

```
Condition: result.actual_token_type == SPEC && !has_verified
Action:    Discard the result entirely. It was produced by the wrong KV state.
```

If `signal_to_exit` is true:
```
           has_unverified = false, signal_to_exit = false
           tokens_generated = 0
           Flush pending_complete to output_queue.
```

## State Transition Summary

```
                            INITIAL
                               |
                     emit token, store prediction
                      has_unverified = true
                               |
                    +----------+-----------+
                    |                       |
              (BASE matches)         (BASE mismatches)
                 ACCEPT                 REJECT
            has_verified=true        has_unverified=true
            has_unverified=false     (new prediction)
                    |                       |
            +-------+                +------+
            |                        |
       (SPEC arrives)           (SPEC arrives)
          CONTINUE                  STALE
       emit bonus token          discard
       has_verified=false
       has_unverified=true
       (new prediction)
            |
            +--------> back to ACCEPT/REJECT decision
```

Completion can happen at any ACCEPT, REJECT, or CONTINUE step. When it does during a paired phase (where a SPEC/STALE is still expected), the completion is deferred via `pending_complete` and `signal_to_exit`, then flushed when the paired result arrives.

## Deferred Completion: Why It Matters

The `defer_complete` mechanism prevents a race between the IS and the pipeline:

```
Without deferral (BROKEN):
  Reader: emit_token(COMPLETE) -> push to output_queue
  IS:     pop output -> sees completion -> sends CONTINUE
  API:    processes CONTINUE -> starts new prefill, resets prefill_in_flight
  Reader: receives stale SPEC result -> decrements prefill_in_flight
          -> underflow or miscount (BUG)

With deferral (CORRECT):
  Reader: emit_token(COMPLETE, defer=true) -> stash in pending_complete
  Reader: receives SPEC result (CONTINUE or STALE) -> drains it
  Reader: flush_pending_complete() -> push completion to output_queue
  IS:     pop output -> sees completion -> sends CONTINUE (safe now)
```

The stale SPEC/STALE result from the previous speculation is fully consumed before the IS learns about completion. This eliminates any possibility of the IS triggering a new turn before the pipeline has drained.

## reset() Semantics

```cpp
void reset(uint32_t uid) {
    unverified_token[uid] = EMPTY_TOKEN;
    has_unverified[uid]   = false;
    has_verified[uid]     = false;
    signal_to_exit[uid]   = false;
    has_pending_complete[uid] = false;
}
```

Called by:
- API thread during ALLOCATE (after slot allocation, before submission).
- API thread during SUBMIT and CONTINUE (before entering PREFILL state).
- API thread during CANCEL processing.
- Cleanup winner in `maybe_finalize_cleanup()` (after the cleanup CAS succeeds).

`reset()` clears all tracking flags and the stashed prediction. It does not clear `pending_complete[uid]` -- only the boolean `has_pending_complete` is cleared, which logically invalidates the stashed message. The actual `OutputMessage` data in `pending_complete[uid]` is left as stale bytes, harmless because `has_pending_complete` gates all reads.

### Thread Safety of reset()

`reset()` is called by the API thread, while the spec state fields are normally owned by the Reader thread. This is safe because:

- **On ALLOCATE**: the slot was just allocated from `FreeIdPool`; no Reader activity exists.
- **On SUBMIT/CONTINUE**: the user is transitioning from COMPLETE or INACTIVE state. No pipeline tokens are in-flight for this user (the Reader has finished processing all results from the previous turn). The API thread sets up the new state before pushing the user into the prefill queue, at which point the Reader will not see results until the Writer has injected tokens.
- **On CANCEL**: the Reader will observe `cancel_pending.is_set()` and skip all spec-decode processing. The API handler's `reset()` runs after `cancel_pending.mark()`.
- **During cleanup**: the CAS on `state` has already succeeded and no more results will arrive.

## Why No Atomics

Unlike `UserTable` fields that are touched by multiple threads, `SpecDecodeState` is exclusively used by the Reader thread during active decode processing. The API thread's `reset()` calls happen only during safe transition points where no concurrent Reader access to the same slot is possible. This single-writer design means plain vectors suffice -- no atomics, no fences, no cache line bouncing. In a hardware deployment, these would map to simple scratchpad registers in the Reader's local memory.

---

**Previous:** [PrefillQueue and BoundedQueue](./prefill_queue_and_bounded_queue.md) | **Next:** [Lock-Free Design](./lock_free_design.md)
