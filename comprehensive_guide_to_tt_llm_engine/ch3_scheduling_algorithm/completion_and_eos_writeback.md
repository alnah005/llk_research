# Completion and EOS Write-Back

Generation ends when the decode cycle detects a termination condition. The completion path must publish the final token, write the terminator into KV for multi-turn context, update `current_position`, and transition the user to COMPLETE state -- all in the correct order relative to concurrent threads and potential CONTINUE requests.

## Governing Invariants

**Invariant 1: on completion, the terminating token is written back to KV at its actual position before the user is visible as COMPLETE.**

This ensures that a CONTINUE request arriving after completion observes a KV cache that includes the terminator. If the write-back were omitted, the next turn's context would miss the EOS/final token.

**Invariant 2: `current_position` is set to one past the terminating token's position before the completion `OutputMessage` is published.**

A CONTINUE arriving in response to the completion message must see the correct starting position for the next turn's prefill.

**Invariant 3: `post_complete_in_flight` prevents the EOS write-back's pipeline result from re-entering the decode path.**

The write-back injects a token into the pipeline, which will produce a result. That result must be silently discarded, not treated as a new decode token.

## Three Completion Triggers

The `emit_token` lambda checks three conditions on every decoded token (lines 464-467):

```cpp
bool ctx_exhausted = (ctx_pos >= params.max_seq_len);
bool complete = ctx_exhausted ||
                (!user_table.ignore_eos[uid] && tok == params.eos_token) ||
                (user_table.tokens_generated[uid] >= user_table.max_new_tokens[uid]);
```

### Trigger 1: EOS Token Match

The sampled token matches `params.eos_token` (default: token ID 1, `DEFAULT_EOS_TOKEN`) and the user does not have `ignore_eos` set. The `ignore_eos` flag (per-user, set via `GenerationParams`) suppresses this trigger for benchmarking or streaming use cases. This is the normal completion path for well-behaved models.

### Trigger 2: `max_new_tokens` Reached

The user's `tokens_generated` counter has reached or exceeded `max_new_tokens` (set by the IS on SUBMIT or CONTINUE). The counter is incremented BEFORE the check (line 463: `user_table.tokens_generated[uid]++`), so the comparison uses the post-increment value. If `max_new_tokens = 100`, the 100th call to `emit_token` sets `tokens_generated` to 100 and triggers completion. This provides a hard upper bound on generation length, regardless of whether the model produces an EOS.

### Trigger 3: Context Exhaustion

The KV position (`ctx_pos`) has reached or exceeded `max_seq_len`. The model has filled the entire context window. `ctx_exhausted` is set to `true` in the `OutputMessage`, signaling to the IS that the completion was forced rather than natural.

For non-speculative decode, `ctx_pos` is `result.actual_token_pos` -- the position of the current token. For speculative decode, `ctx_pos` is `result.predicted_token_pos` -- the highest KV position that would be written if the speculation pair is staged (because the SPEC token writes at `actual_token_pos + 1`).

The evaluation order within the `complete` expression uses short-circuit OR. Context exhaustion is checked first (cheapest), then EOS (if `ignore_eos` is false), then token count.

## `emit_token` Lambda Internals

`emit_token` is defined inside `reader_loop()` (lines 450-496) and captures the Reader's local state. It is the central completion-detection and output-publication function. For regular decode, it is called once per decode loopback iteration with `defer_complete = false`.

### Step 1: Thinking-Phase Toggle

```cpp
if (params.think_open_token_id != EMPTY_TOKEN && tok == params.think_open_token_id) {
    user_table.in_thinking_phase[uid].store(true, std::memory_order_release);
} else if (params.think_close_token_id != EMPTY_TOKEN && tok == params.think_close_token_id) {
    user_table.in_thinking_phase[uid].store(false, std::memory_order_release);
}
```

**Invariant: the thinking-phase toggle is applied BEFORE any follow-on staging (loopback or EOS write-back).**

The toggle happens BEFORE any completion logic or staging. The release store is paired with the Writer's acquire load in `device_sampling_params()`, ensuring the Writer uses the correct sampling parameters for the next injection ($k = 1$ outside thinking, user's $k$ inside thinking). When thinking-phase tracking is disabled (both token IDs are `EMPTY_TOKEN`), neither branch is taken -- no token ever matches the sentinel.

### Step 2: Token Counter Increment

```cpp
user_table.tokens_generated[uid]++;
```

Plain increment -- no atomic needed because only the Reader writes this field during DECODE state. The counter is incremented before the completion check so that `max_new_tokens` triggers on the correct token count.

### Step 3: Completion Detection

The three-condition check described above. If none triggers, the token is not a completer.

### Step 4: Output Message Construction

```cpp
OutputMessage msg{
    .slot_id = uid,
    .token_id = tok,
    .is_complete = complete,
    .ctx_exhausted = ctx_exhausted,
    .tokens_generated = user_table.tokens_generated[uid],
    .generation = decode_staging.generation[uid].load(std::memory_order_relaxed),
};
```

Every output token -- including the completing one -- is published as an `OutputMessage`. The IS sees the `is_complete` flag and knows this is the final token. The `generation` field enables the IS to discard late-arriving tokens from a previous session (see the Generation Counter section below).

### Step 5: Completion Actions (if complete)

```cpp
if (complete) {
    user_table.state[uid].store(UserState::COMPLETE, std::memory_order_release);
    stage_eos_writeback();
    set_position_on_complete();
}
```

The ordering is critical:

1. **State transition** to COMPLETE (release store). Other threads can observe completion.
2. **EOS write-back** (stages a pipeline injection via `stage_eos_writeback()`).
3. **Position update** (`current_position` set to `actual_token_pos + 1` via `set_position_on_complete()`).

Steps 1-3 happen BEFORE the `OutputMessage` is published. This ordering ensures that if the IS receives the completion message and immediately issues a CONTINUE, the API handler sees `state == COMPLETE` and a valid `current_position`. The release stores in steps 1 and 3 guarantee this.

### Step 6: Publication

```cpp
if (complete && defer_complete) {
    ss.pending_complete[uid] = msg;
    ss.has_pending_complete[uid] = true;
    return true;
}
while (!output_queue.try_push(msg)) {
    _mm_pause();
}
if (complete) {
    user_table.tokens_generated[uid] = 0;
}
return complete;
```

For regular (non-speculative) decode, `defer_complete` is always `false` (the default). The message is published immediately via `output_queue.try_push()` with `_mm_pause()` spin on full. The `defer_complete` parameter exists for the speculative decode path, where the completion message must be stashed until the paired SPEC/STALE result is consumed (see [Chapter 4](../ch4_speculative_decode/index.md) for the full deferred-complete protocol).

After publishing a complete message, `tokens_generated[uid]` is reset to 0. This prepares the slot for a potential multi-turn CONTINUE. The reset happens after the `OutputMessage` has been pushed (the message already captured the final `tokens_generated` value).

`emit_token` returns `true` if complete, `false` otherwise. The caller uses this to decide whether to stage a loopback entry (false) or skip it (true, because the EOS write-back already handles the final injection).

## EOS Write-Back: `stage_eos_writeback`

```cpp
auto stage_eos_writeback = [&]() {
    user_table.post_complete_in_flight[uid].fetch_add(1, std::memory_order_release);
    decode_staging.stage(uid,
        result.actual_token, result.actual_token_pos,
        EMPTY_TOKEN, 0, /*skip_spec=*/true);
};
```

The write-back stages the completing token at its original KV position (`actual_token_pos`). This writes the terminator (typically the EOS token) into the KV cache so the next turn's context includes it. The entry is marked with `skip_spec = true` to prevent the Writer from pairing it with a SPEC injection, even if the user has speculative decode enabled.

### `post_complete_in_flight` Counter

**Invariant: the post-completion EOS write-back result is discarded, not looped back.**

The `post_complete_in_flight` counter is incremented BEFORE the staging call (release ordering). When the pipeline delivers the result for this write-back injection, the Reader checks:

```cpp
if (result.actual_token_type == TokenType::BASE &&
    user_table.post_complete_in_flight[uid].load(std::memory_order_acquire) > 0) {
    user_table.post_complete_in_flight[uid].fetch_sub(1, std::memory_order_relaxed);
    continue;
}
```

The result is silently discarded. The check is restricted to `TokenType::BASE` results. This is deliberate: in speculative decode mode, a SPEC result may still be pending from the pre-completion round. The `post_complete_in_flight` discard must not swallow that SPEC result, which needs to reach the spec-decode state machine for proper draining. For regular (non-spec) decode, all results are BASE type, so this distinction does not matter.

### EOS Write-Back Timeline

```
Reader (completion detected):
  |
  +--[A] post_complete_in_flight[uid]++       (release: visible to Reader on next result)
  |
  +--[B] decode_staging.stage(uid,
  |         actual_token, actual_token_pos,
  |         EMPTY_TOKEN, 0, skip_spec=true)   (pending_count++ inside stage())
  |
  : (entry sits in DecodeStaging FIFO)
  :
Writer (next available tick):
  |
  +--[C] decode_staging.try_pop(entry)         (gets the EOS writeback entry)
  |
  +--[D] cancel check                          (not cancelled -- this is a valid writeback)
  |
  +--[E] do_spec = spec_decode_enabled && !skip_spec
  |       = false (skip_spec is true)
  |
  +--[F] in_flight_count[uid] += 1             (release)
  |
  +--[G] decode_staging.release(uid)           (pending_count--)
  |
  +--[H] pipeline_->inject(BASE token at actual_token_pos)
  |
  : (token traverses D pipeline stages)
  :
Reader (D ticks later):
  |
  +--[I] read_result() for uid
  |
  +--[J] ReaderClaim(uid)                      (pending_count++)
  |
  +--[K] in_flight_count[uid]--                (relaxed)
  |
  +--[L] post_complete_in_flight > 0?          YES
  |       -> post_complete_in_flight--
  |       -> continue (discard this result)
  |
  +--[M] ~ReaderClaim()                        (pending_count--, maybe_finalize_cleanup)
```

## `set_position_on_complete`

```cpp
auto set_position_on_complete = [&]() {
    user_table.current_position[uid].store(
        result.actual_token_pos + 1, std::memory_order_release);
};
```

The stored position is `actual_token_pos + 1`. The "+1" accounts for the EOS write-back: `stage_eos_writeback` writes the completing token at `actual_token_pos`, so the next turn's prefill must begin at the position after it. This value becomes `prefill_start_pos` in `handle_local_continue()`:

```cpp
uint32_t start_pos = user_table.current_position[uid].load(std::memory_order_relaxed);
// ...
user_table.prefill_pos[uid] = start_pos;
user_table.prefill_start_pos[uid] = start_pos;
```

The relaxed load in `handle_local_continue` is safe because the CONTINUE request can only arrive after the IS has observed the completion message (which was published after `set_position_on_complete`), establishing a happens-before relationship through the output queue.

## State Transition: DECODE to COMPLETE

The transition from DECODE to COMPLETE preserves all slot state needed for multi-turn continuation:

| What is preserved | Why |
|---|---|
| KV cache (device-side) | Multi-turn continuation reuses existing KV |
| Slot allocation (not returned to `FreeIdPool`) | User may CONTINUE |
| `current_position` | Bookmark for next turn's `prefill_start_pos` |

| What is reset | When |
|---|---|
| `tokens_generated` | Set to 0 immediately after publishing the completion message |
| `state` | Set to COMPLETE (not INACTIVE) |

The slot remains allocated in COMPLETE state until the IS sends either a CONTINUE (re-enters PREFILL from `current_position`) or a CANCEL (triggers the deferred cleanup protocol, drains in-flight tokens, and frees the slot and KV cache). This is the foundation of [Chapter 1 -- Invariant 5: KV Cache Retained on Completion](../ch1_architecture_and_layering/key_invariants.md#invariant-5-kv-cache-retained-on-completion).

## State After Completion

After the full completion sequence (including EOS write-back drain), the slot reaches quiescence:

| Field | Value | Notes |
|-------|-------|-------|
| `state` | COMPLETE | KV cache retained, ready for CONTINUE |
| `current_position` | `actual_token_pos + 1` | Next turn starts here |
| `in_flight_count` | 0 | All pipeline tokens drained |
| `pending_count` | 0 | All staging entries drained |
| `tokens_generated` | 0 | Reset for next turn |
| `post_complete_in_flight` | 0 | Write-back result discarded |

The KV cache is NOT cleared. The device retains all key-value data up to `current_position - 1`. A subsequent CONTINUE will build on this existing context. Only CANCEL triggers `pipeline_->reset_kv(uid)` to release the KV cache.

## Context Exhaustion During Prefill

Context exhaustion can also occur during prefill (before any decode tokens are generated), handled directly in the Writer's prefill path rather than through `emit_token`. This path is covered in [`chunked_prefill.md`](./chunked_prefill.md#step-4-prompt-exhaustion-and-context-exhaustion). Notably, the prefill-phase context exhaustion does NOT call `stage_eos_writeback()` or `set_position_on_complete()` -- the user never entered decode, so there is no terminator to write back and no decode result to derive a position from. The `OutputMessage` carries `EMPTY_TOKEN` and `tokens_generated = 0`, clearly indicating that no output was produced.

## Generation Counter on `OutputMessage`

Every `OutputMessage` carries a `generation` field (loaded with relaxed ordering from `decode_staging.generation[uid]`) so the IS can detect and discard late-arriving tokens from a previous session after a slot has been reallocated. The full generation-counter protocol -- including the cancel-and-reallocate race it prevents -- is documented in [Chapter 2 -- Decode Staging, Generation Counter](../ch2_threading_and_data_structures/decode_staging.md#generation-counter).

## Completion Data Flow Summary

```
Reader receives ResultDescriptor
    |
    v
emit_token(tok = actual_token, ctx_pos = actual_token_pos)
    |
    | [1] Thinking-phase toggle
    |     READ:  params.think_open/close_token_id, tok
    |     WRITE: in_thinking_phase[uid] (release, if match)
    |
    | [2] tokens_generated[uid]++
    |
    | [3] Completion check
    |     READ:  ctx_pos, params.max_seq_len, ignore_eos[uid],
    |            params.eos_token, tokens_generated[uid],
    |            max_new_tokens[uid]
    |
    | [4] Build OutputMessage
    |     READ:  uid, tok, complete, ctx_exhausted,
    |            tokens_generated[uid],
    |            decode_staging.generation[uid] (relaxed)
    |
    |--- if complete: ----------------------------------|
    |                                                   |
    | [5a] state[uid] = COMPLETE (release)              |
    |                                                   |
    | [5b] stage_eos_writeback()                        |
    |      WRITE: post_complete_in_flight[uid]++ (rel)  |
    |      WRITE: staging.pending_count[uid]++          |
    |      WRITE: staging.fifo.push(entry)              |
    |          entry.token_id = actual_token            |
    |          entry.position = actual_token_pos        |
    |          entry.skip_spec = true                   |
    |                                                   |
    | [5c] set_position_on_complete()                   |
    |      WRITE: current_position[uid] =               |
    |             actual_token_pos + 1 (release)        |
    |                                                   |
    |<--------------------------------------------------|
    |
    | [6] output_queue.try_push(msg) (spin on full)
    |
    | [7] if complete: tokens_generated[uid] = 0
    |
    v
  return complete
```

---

**Next:** [`regular_decode_flow.md`](./regular_decode_flow.md)
