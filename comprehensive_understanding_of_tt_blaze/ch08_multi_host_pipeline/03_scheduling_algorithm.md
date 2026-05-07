# 08 -- Multi-Host Pipeline Infrastructure: Scheduling Algorithm

[<< Previous: Pipeline Manager Architecture](02_pipeline_manager_architecture.md) | [Next: KV Cache Migration >>](04_kv_cache_migration.md)

---

## 8.21 Overview

This section walks through the three thread loops that implement the pipeline manager's scheduling algorithm: the writer thread (token injection), the reader thread (result processing, decode loopback, speculative verification), and the API request handler (session lifecycle). The code analyzed here is the `PipelineManager::Impl` methods in `pipeline_manager/src/pipeline_manager.cpp`.

---

## 8.22 Writer Thread Loop

The writer thread runs a continuous loop with two priority levels: decode tokens first, prefill tokens second. Each iteration injects at most one logical unit of work (one token for regular decode, a BASE+SPEC pair for speculative decode) into the pipeline.

### Decode Path (Priority 1)

```cpp
void writer_loop() {
    while (running.load(std::memory_order_acquire)) {
        DecodeStagingEntry entry;
        if (decode_staging.try_pop(entry)) {
            uint32_t uid = entry.slot_id;

            if (cancel_pending.is_set(uid)) {
                decode_staging.release(uid);
                maybe_finalize_cleanup(uid);
                continue;
            }

            bool do_spec = user_table.spec_decode_enabled[uid] && !entry.skip_spec;
            user_table.in_flight_count[uid].fetch_add(
                do_spec ? 2 : 1, std::memory_order_release);
            decode_staging.release(uid);

            pipeline_->inject(InjectDescriptor{
                .slot_id = uid,
                .token_id = entry.token_id,
                .position = entry.position,
                .token_type = TokenType::BASE,
                .spec_flag = false,
                .temperature = user_table.temperature[uid],
                .top_p = user_table.top_p[uid],
                .top_k = user_table.top_k[uid],
            });

            if (do_spec) {
                pipeline_->inject(InjectDescriptor{
                    .slot_id = uid,
                    .token_id = entry.spec_token_id,
                    .position = entry.spec_position,
                    .token_type = TokenType::SPEC,
                    .spec_flag = true,
                    // ... same temperature/top_p/top_k
                });
            }
            continue;
        }
```

Key design decisions in the decode path:

1. **in_flight_count is bumped before release.** The `fetch_add` happens before `decode_staging.release(uid)`. This ensures the slot's busy count never momentarily drops to zero while transitioning from the staging claim to the in-flight count. Without this ordering, a concurrent CANCEL could observe `(pending=0, in_flight=0)` and prematurely finalize the slot.

2. **Cancel check before inject.** If the slot is pending cancellation, the writer releases the staging claim and calls `maybe_finalize_cleanup` without injecting. The entry is silently dropped.

3. **Speculative pair injection.** When `spec_decode_enabled` is true and `skip_spec` is false, the writer injects two tokens back-to-back: the BASE token at `entry.position` and the SPEC predicted token at `entry.spec_position`. The `in_flight_count` is incremented by 2 atomically before either inject to maintain the invariant.

4. **skip_spec for EOS writeback.** The `skip_spec` flag is set when the reader stages a post-completion EOS writeback. This inject writes the terminator token into KV so the next multi-turn continuation sees it as context, but it must not generate a speculative prediction.

### Prefill Path (Priority 2)

```cpp
        uint32_t pfuid;
        if (prefill_queue.try_front(pfuid)) {
            if (cancel_pending.is_set(pfuid)) {
                prefill_queue.pop_front();
                maybe_finalize_cleanup(pfuid);
                continue;
            }

            uint32_t prompt_len = prompt_table.get_length(pfuid);
            uint32_t device_pos = user_table.prefill_pos[pfuid];
            uint32_t prompt_idx = device_pos - user_table.prefill_start_pos[pfuid];
            uint32_t chunk_rem = user_table.prefill_chunk_remaining[pfuid];
```

The prefill path has three sub-cases:

**Chunk exhausted** (`chunk_rem == 0`): Reset the chunk counter and call `prefill_queue.rotate()`, which moves this user to the back of the deque. This implements chunked round-robin scheduling. With `chunk_size = 24`, the writer injects 24 prefill tokens for one user, then switches to the next user's prefill (or to decode tokens if any are staged).

**Prefill complete** (`prompt_idx >= prompt_len` or context exhausted): The user's prompt has been fully injected. The writer pops the user from the prefill queue, clears the prompt table, sets state to COMPLETE (if context exhausted), and pushes a completion `OutputMessage`. For context exhaustion, the `ctx_exhausted` flag is set on the output message.

**Normal prefill injection**: The writer injects one prompt token. For the last token (`is_last`), the inject uses `EMPTY_TOKEN` for `prefill_token_id`, which tells the pipeline to sample (this is the transition from prefill to decode). For non-last tokens, `prefill_token_id` is set to the *next* prompt token, signaling the pipeline that this is a non-sampled prefill position.

```cpp
            bool is_last = (prompt_idx == prompt_len - 1);
            user_table.prefill_pos[pfuid] = device_pos + 1;
            user_table.current_position[pfuid].store(device_pos + 1,
                std::memory_order_release);
            user_table.prefill_chunk_remaining[pfuid] = chunk_rem - 1;

            if (!is_last) {
                user_table.prefill_in_flight[pfuid].fetch_add(1,
                    std::memory_order_release);
            }
            user_table.in_flight_count[pfuid].fetch_add(1,
                std::memory_order_release);

            pipeline_->inject(InjectDescriptor{
                .slot_id = pfuid,
                .token_id = prompt_table.get_token(pfuid, prompt_idx),
                .prefill_token_id = is_last ? EMPTY_TOKEN
                    : prompt_table.get_token(pfuid, prompt_idx + 1),
                .position = device_pos,
                .token_type = TokenType::BASE,
            });

            if (is_last) {
                prefill_queue.pop_front();
                user_table.state[pfuid].store(UserState::DECODE,
                    std::memory_order_release);
                prompt_table.clear(pfuid);
            }
```

The `prefill_in_flight` counter is incremented for every non-last prefill token. The reader uses this counter to skip results from non-sampled prefill positions -- these results contain no meaningful output tokens and should be silently consumed.

When the writer processes the last prefill token, it transitions the user to `DECODE` state and clears the prompt table. The user is now in the decode loopback path: its first sampled result will be processed by the reader and staged back for decode injection.

---

## 8.23 Reader Thread Loop

The reader thread blocks on `pipeline_->read_result()` and processes each result through a series of checks: invalid slot (shutdown sentinel), cancellation, post-completion EOS draining, prefill skipping, and then the main decode/spec-decode logic.

### Result Processing Preamble

```cpp
void reader_loop() {
    while (running.load(std::memory_order_acquire)) {
        ResultDescriptor result = pipeline_->read_result();
        uint32_t uid = result.slot_id;
        if (uid == INVALID_SLOT) break;

        ReaderClaim scratch(*this, uid);
        user_table.in_flight_count[uid].fetch_sub(1, std::memory_order_relaxed);
```

`ReaderClaim` is an RAII guard that increments `decode_staging.pending_count[uid]` on construction and decrements it (plus calls `maybe_finalize_cleanup`) on destruction. This ensures the slot's busy count remains non-zero for the entire duration of the reader's processing of one result, preventing the cancel cleanup from racing with the reader's follow-on `stage()` calls.

### Cancellation Check

```cpp
        if (cancel_pending.is_set(uid)) {
            user_table.prefill_in_flight[uid].store(0, std::memory_order_relaxed);
            user_table.post_complete_in_flight[uid].store(0, std::memory_order_relaxed);
            maybe_finalize_cleanup(uid);
            continue;
        }
```

When a cancellation is pending, all results for this user are drained. The prefill and post-complete counters are zeroed (no further tracking needed for a cancelled user). `maybe_finalize_cleanup` checks if both `in_flight_count` and `pending_count` have reached zero; if so, it CAS-transitions the state to INACTIVE, calls `reset_kv`, and frees the slot.

### Post-Completion EOS Writeback Draining

```cpp
        if (result.actual_token_type == TokenType::BASE &&
            user_table.post_complete_in_flight[uid].load(
                std::memory_order_acquire) > 0) {
            user_table.post_complete_in_flight[uid].fetch_sub(1,
                std::memory_order_relaxed);
            continue;
        }
```

After a user completes generation, the reader stages an EOS writeback inject (to persist the terminator in KV for multi-turn). That inject's BASE result must be consumed but not processed as a new decode token. The `post_complete_in_flight` counter tracks these expected drain results. It is checked only for BASE-type results so it does not interfere with stale SPEC results that may also be in the pipeline.

### Prefill Result Skipping

```cpp
        if (user_table.prefill_in_flight[uid].load(
                std::memory_order_acquire) > 0) {
            user_table.prefill_in_flight[uid].fetch_sub(1,
                std::memory_order_relaxed);
            continue;
        }
```

Non-last prefill injects produce pipeline results that carry no sampled token. The reader skips them by decrementing the `prefill_in_flight` counter.

### Non-Speculative Decode

```cpp
        if (!user_table.spec_decode_enabled[uid]) {
            ctx_pos = result.actual_token_pos;
            if (!emit_token(result.actual_token, ctx_pos)) {
                decode_staging.stage(uid,
                    result.actual_token, result.actual_token_pos,
                    EMPTY_TOKEN, 0);
            }
            continue;
        }
```

For regular (non-speculative) decode, the reader calls `emit_token()` which increments `tokens_generated`, checks for EOS / max tokens / context exhaustion, and pushes an `OutputMessage` to the output queue. If the user is not complete, the token is staged in `decode_staging` for the writer to re-inject (the decode loopback).

---

## 8.24 Speculative Decode (MTP)

Speculative decode injects two tokens per user per round: a BASE token and a SPEC predicted token. The pipeline processes them in order. The reader verifies whether the pipeline's actual output matches the prediction.

### SpecDecodeState

Per-user tracking for the reader thread (single-threaded, no atomics needed):

```cpp
struct SpecDecodeState {
    std::vector<uint32_t> unverified_token;   // prediction awaiting verification
    std::vector<bool> has_unverified;          // true if a prediction is pending
    std::vector<bool> has_verified;            // true after ACCEPT, waiting for SPEC
    std::vector<bool> signal_to_exit;          // completion deferred through SPEC drain
    std::vector<OutputMessage> pending_complete;
    std::vector<bool> has_pending_complete;
};
```

### State Machine

The speculative decode reader implements a four-state machine per user. The states are tracked through combinations of `has_unverified` and `has_verified`:

```
INITIAL:  !has_unverified && !has_verified
          First sampled result after prefill. No prior speculation to verify.
          -> Store prediction, emit actual token, transition to has_unverified.

BASE result with has_unverified:
  if actual == unverified_token:   ACCEPT
    -> Emit actual, set has_verified, clear has_unverified.
    -> Wait for SPEC result (CONTINUE or CONTINUE-suppress).

  if actual != unverified_token:   REJECT
    -> Emit actual, store new prediction, keep has_unverified.
    -> Next SPEC result will be STALE (discard).

SPEC result with has_verified:     CONTINUE
    -> Emit bonus token (the SPEC's actual), store new prediction.
    -> Clear has_verified, set has_unverified.
    -> Stage pair for writer.

SPEC result without has_verified:  STALE
    -> Discard. This SPEC result was computed from stale KV after a reject.
```

### ACCEPT Path (Prediction Correct)

When the BASE result's `actual_token` matches the stored `unverified_token`:

```cpp
if (ss.has_unverified[uid] &&
    result.actual_token == ss.unverified_token[uid]) {
    user_table.spec_accepts[uid]++;
    ss.has_verified[uid] = true;
    ss.has_unverified[uid] = false;
    if (emit_token(result.actual_token, ctx_pos, /*defer_complete=*/true)) {
        ss.signal_to_exit[uid] = true;
    }
    continue;  // wait for SPEC result
}
```

The accept count is incremented for diagnostics. `defer_complete=true` means that if this token triggers completion (EOS/max tokens), the `OutputMessage` is stashed in `pending_complete` rather than immediately published. This prevents the IS from seeing completion (and potentially sending CONTINUE) before the stale SPEC result is consumed.

### CONTINUE Path (Bonus Token)

When the SPEC result arrives after an ACCEPT:

```cpp
if (result.actual_token_type == TokenType::SPEC) {
    if (ss.has_verified[uid]) {
        ss.has_verified[uid] = false;
        ss.has_unverified[uid] = true;
        ss.unverified_token[uid] = result.predicted_token;

        if (ss.signal_to_exit[uid]) {
            // Completion was deferred; flush it now
            ss.signal_to_exit[uid] = false;
            ss.has_unverified[uid] = false;
            user_table.tokens_generated[uid] = 0;
            flush_pending_complete();
            continue;
        }

        if (emit_token(result.actual_token, ctx_pos)) {
            ss.has_unverified[uid] = false;
        } else {
            stage_pair();
        }
        continue;
    }
```

In the match case, the SPEC result's `actual_token` is a bonus output token -- the reader emits it to the IS and stages a new BASE+SPEC pair for the writer. If `signal_to_exit` is set (the prior ACCEPT triggered completion), the bonus token is suppressed and the deferred completion message is flushed.

`stage_pair()` stages both the base token (for KV loopback) and the new prediction in a single `DecodeStagingEntry`:

```cpp
auto stage_pair = [&]() {
    decode_staging.stage(uid,
        result.actual_token, result.actual_token_pos,
        result.predicted_token, result.predicted_token_pos);
};
```

### REJECT Path (Prediction Wrong)

When the BASE result's `actual_token` does not match:

```cpp
} else {
    if (ss.has_unverified[uid]) {
        user_table.spec_rejects[uid]++;
    }
    ss.has_unverified[uid] = true;
    ss.unverified_token[uid] = result.predicted_token;
    if (emit_token(result.actual_token, ctx_pos, /*defer_complete=*/true)) {
        ss.signal_to_exit[uid] = true;
        continue;  // wait for STALE spec result
    }
    stage_pair();
    continue;
}
```

On reject, the reader emits only the actual token (the prediction was wrong, so no bonus). The writer will re-inject the actual token at the SPEC's position to overwrite the stale KV entry, and inject a new SPEC prediction at the next position. The stale SPEC result that follows will be discarded:

### STALE Path (Discard After Reject)

```cpp
} else {
    // STALE: spec result after reject
    if (ss.signal_to_exit[uid]) {
        ss.has_unverified[uid] = false;
        user_table.tokens_generated[uid] = 0;
        flush_pending_complete();
    }
    ss.signal_to_exit[uid] = false;
    continue;
}
```

The stale SPEC result is silently consumed. If completion was deferred (`signal_to_exit`), the pending completion message is flushed now that all in-flight results have been drained.

### Per-Round Throughput Summary

| Path | Output tokens | Pipeline slots used | Effective rate |
|---|---|---|---|
| ACCEPT + CONTINUE | 2 (base + bonus) | 2 (BASE + SPEC) | 2x |
| REJECT + STALE | 1 (base only) | 2 (BASE + SPEC) | 1x + 1 wasted slot |

---

## 8.25 API Request Handling

The API thread runs a polling loop that drains the request queue:

```cpp
void api_loop() {
    while (running.load(std::memory_order_acquire)) {
        if (!handle_api_requests()) {
            _mm_pause();
        }
    }
    handle_api_requests();  // drain remaining after stop
}
```

### ALLOCATE

```cpp
case RequestType::ALLOCATE: {
    uint32_t uid = free_ids.allocate();
    if (uid != INVALID_SLOT) {
        user_table.reset(uid);
        cancel_pending.clear(uid);
        spec_state.reset(uid);
    }
    PMResponse resp{.request_id = req.request_id, .slot_id = uid,
                    .error_code = (uid == INVALID_SLOT) ? 1 : 0};
    while (!response_queue.try_push(resp)) { _mm_pause(); }
    break;
}
```

Allocates a user slot from the bitmap pool. If the pool is exhausted, returns `error_code = 1` and `slot_id = INVALID_SLOT`. On success, resets all per-user state to defaults. The `cancel_pending.clear` is important: if this slot was previously cancelled and the cancel bitmap was not cleared (a defensive measure), clearing it here prevents stale cancel state from affecting the new allocation.

### SUBMIT

```cpp
case RequestType::SUBMIT: {
    uint32_t uid = req.slot_id;
    prompt_table.store(uid, req.tokens.data(), req.tokens.size());
    user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
    user_table.current_position[uid].store(0, std::memory_order_relaxed);
    user_table.prefill_pos[uid] = 0;
    user_table.prefill_start_pos[uid] = 0;
    user_table.max_new_tokens[uid] = req.gen.max_new_tokens;
    user_table.tokens_generated[uid] = 0;
    user_table.in_flight_count[uid].store(0, std::memory_order_relaxed);
    user_table.prefill_chunk_remaining[uid] = params.chunk_size;
    user_table.spec_decode_enabled[uid] = req.gen.spec_decode;
    user_table.ignore_eos[uid] = req.gen.ignore_eos;
    user_table.temperature[uid] = req.gen.temperature;
    user_table.top_p[uid] = req.gen.top_p;
    user_table.top_k[uid] = req.gen.top_k;
    spec_state.reset(uid);
    prefill_queue.push(uid);
    break;
}
```

Stores the prompt tokens, initializes all per-user metadata (starting from position 0), and pushes the user into the prefill queue. The writer will pick up this user and start injecting prompt tokens on its next iteration.

### CONTINUE

Two sub-paths: local continue (multi-turn on the same host) and disaggregated continue (after KV migration from a remote prefill host).

**Local continue** resumes from `current_position` (preserved after COMPLETE):

```cpp
void handle_local_continue(const ISRequest& req) {
    uint32_t uid = req.slot_id;
    uint32_t start_pos = user_table.current_position[uid].load(
        std::memory_order_relaxed);
    prompt_table.store(uid, req.tokens.data(), req.tokens.size());
    user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
    user_table.prefill_pos[uid] = start_pos;
    user_table.prefill_start_pos[uid] = start_pos;
    // ... set generation params, reset spec state
    prefill_queue.push(uid);
}
```

The `prefill_start_pos` is set to `current_position` so the writer offsets prompt indices from the stored KV position. The existing KV cache (from the previous turn) is reused -- no re-prefilling of conversation history.

**Disaggregated continue** is used after KV migration from a remote prefill endpoint:

```cpp
void handle_disaggregated_continue(const ISRequest& req) {
    uint32_t uid = req.slot_id;
    uint32_t n = static_cast<uint32_t>(req.tokens.size() - 1);
    uint32_t migrated_token = req.tokens.back();
    user_table.state[uid].store(UserState::DECODE, std::memory_order_release);
    user_table.current_position[uid].store(n + 1, std::memory_order_release);
    // ... set generation params
    decode_staging.stage(uid, migrated_token, n, EMPTY_TOKEN, 0);
}
```

In this case, the KV cache was already populated by a remote prefill host and migrated. The last token in `req.tokens` is the migrated decode token, and `n` is the number of prefill positions already present in KV. The user goes directly to DECODE state with the migrated token staged for immediate decode injection.

### CANCEL

```cpp
case RequestType::CANCEL: {
    uint32_t uid = req.slot_id;
    cancel_request_id[uid] = req.request_id;
    cancel_pending.mark(uid);
    decode_staging.advance_generation(uid);
    prefill_queue.remove(uid);

    UserState st = user_table.state[uid].load(std::memory_order_acquire);
    if (st == UserState::INACTIVE &&
        user_table.in_flight_count[uid].load(std::memory_order_acquire) == 0 &&
        decode_staging.pending_count[uid].load(std::memory_order_acquire) == 0) {
        // Synchronous cleanup for never-submitted users
        cancel_pending.clear(uid);
        pipeline_->reset_kv(uid);
        free_ids.free(uid);
        push_cancel_ack(req.request_id, uid);
    } else {
        maybe_finalize_cleanup(uid);
    }
    break;
}
```

Cancel proceeds in stages:

1. **Stash `cancel_request_id`** before marking the bitmap. The release store in `mark()` ensures any thread that observes `is_set(uid)` via an acquire load also sees the stashed request ID.

2. **Mark the cancel bitmap.** All three threads check this bitmap before processing work for this user.

3. **Advance decode staging generation.** Stale entries in the FIFO from before the cancel will have an old generation and be silently dropped.

4. **Remove from prefill queue.** Prevents the writer from injecting more prefill tokens.

5. **Attempt synchronous cleanup.** If the user was never submitted (INACTIVE with no in-flight work), cleanup completes immediately. Otherwise, cleanup is deferred to `maybe_finalize_cleanup`.

---

## 8.26 Cancellation Safety: maybe_finalize_cleanup

The `maybe_finalize_cleanup` function can be called from any of the three threads. It gates on three conditions:

```cpp
void maybe_finalize_cleanup(uint32_t uid) {
    if (!cancel_pending.is_set(uid)) return;
    if (user_table.in_flight_count[uid].load(std::memory_order_acquire) != 0) return;
    if (decode_staging.pending_count[uid].load(std::memory_order_acquire) != 0) return;

    uint32_t request_id = cancel_request_id[uid];

    // CAS state -> INACTIVE: exactly one thread wins
    UserState expected = user_table.state[uid].load(std::memory_order_acquire);
    if (expected == UserState::INACTIVE) return;
    if (!user_table.state[uid].compare_exchange_strong(
            expected, UserState::INACTIVE,
            std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
    }

    pipeline_->reset_kv(uid);
    spec_state.reset(uid);
    free_ids.free(uid);
    push_cancel_ack(request_id, uid);
}
```

The CAS on `state` ensures exactly one thread performs the cleanup, even if multiple threads call `maybe_finalize_cleanup` concurrently (the reader on each result drain, the writer on decode staging drain, the API thread on the initial cancel). The `cancel_request_id` is read before the CAS and before `free_ids.free(uid)` to prevent a race with a future ALLOCATE/CANCEL on the reused slot.

---

## 8.27 Session Lifecycle State Machine

```
                 INACTIVE
                    |
          ALLOCATE  |
                    v
                 INACTIVE  (slot allocated, state remains INACTIVE)
                    |
           SUBMIT   |
                    v
                 PREFILL -----> COMPLETE (context exhausted during prefill)
                    |
         all tokens |
          injected  |
                    v
                  DECODE
                    |
         EOS / max  |
         / ctx full |
                    v
                 COMPLETE  (KV retained for multi-turn)
                    |
         +----------+----------+
         |                     |
      CONTINUE              CANCEL
         |                     |
         v                     v
      PREFILL              INACTIVE
   (from stored pos)     (deferred until
                          in_flight == 0)
```

Key properties:

- **KV retention on COMPLETE.** `current_position` is preserved. A CONTINUE resumes prefill from this position, reusing all existing KV cache entries. Only new conversation tokens are injected.

- **Deferred cleanup on CANCEL.** The user slot is not freed until all in-flight pipeline tokens have drained. This prevents the pipeline from writing results for a freed/reused slot.

- **Generation counter.** Each CANCEL increments the decode staging generation counter. Stale staging entries from before the cancel carry an older generation and are dropped by the writer.

- **EOS writeback.** On completion, the reader stages a BASE-only inject of the terminator token at `actual_token_pos`. This writes the EOS into KV so the next CONTINUE turn sees it as context. The `post_complete_in_flight` counter ensures the reader drains this loopback result without processing it as a new decode token. After the EOS writeback result is consumed, `current_position` is set to `actual_token_pos + 1` -- the starting position for the next turn.

---

## 8.28 Context Exhaustion

The pipeline manager enforces `max_seq_len` (default 128K) in two locations:

### Decode-Side Detection

The reader's `emit_token` lambda checks `ctx_pos >= max_seq_len`:

```cpp
auto emit_token = [&](uint32_t tok, uint32_t ctx_pos,
                      bool defer_complete = false) -> bool {
    user_table.tokens_generated[uid]++;
    bool ctx_exhausted = (ctx_pos >= params.max_seq_len);
    bool complete = ctx_exhausted ||
                    (!user_table.ignore_eos[uid] && tok == params.eos_token) ||
                    (user_table.tokens_generated[uid] >= user_table.max_new_tokens[uid]);
    // ...
};
```

When `ctx_exhausted` is true, the `OutputMessage` carries the flag so the IS can take policy action (evict, context summarization, etc.).

### Prefill-Side Guard

The writer checks `device_pos >= params.max_seq_len` during prefill:

```cpp
if (prompt_idx >= prompt_len || device_pos >= params.max_seq_len) {
    prefill_queue.pop_front();
    prompt_table.clear(pfuid);
    // ... mark COMPLETE with ctx_exhausted
}
```

This is a defensive fallback. The IS is expected to reject prompts that would overflow before submitting them, using `get_current_position()` to check available context space.

---

[<< Previous: Pipeline Manager Architecture](02_pipeline_manager_architecture.md) | [Next: KV Cache Migration >>](04_kv_cache_migration.md)
