# Chunked Prefill

Prefill is the process of injecting a user's prompt tokens into the pipeline to populate the KV cache before decode can begin. Without chunking, a single long prompt would monopolize the Writer for thousands of ticks. Chunked prefill breaks prompts into fixed-size chunks and rotates between users to prevent starvation.

## Governing Invariants

**Invariant 1: no single prefill user consumes more than `chunk_size` consecutive Writer ticks before yielding to others.**

After injecting `chunk_size` tokens for one user, the Writer resets the chunk counter, rotates the `PrefillQueue` (moves front to back), and serves the next user. This bounds the maximum consecutive prefill ticks for any single user.

**Invariant 2: decode always preempts prefill.** Even within a chunk, if a decode token arrives in `DecodeStaging`, the Writer handles it immediately on the next iteration (see [`writer_priority_and_decode.md`](./writer_priority_and_decode.md)). The chunk counter is not decremented for ticks consumed by decode -- the prefill user simply resumes where it left off when the Writer returns to the prefill path.

## Chunk Size

The chunk size is configurable via `SchedulerParams::chunk_size`, with a default of `DEFAULT_CHUNK_SIZE = 24` (defined in `decode_types.hpp`). Each user's remaining chunk allowance is tracked in `user_table.prefill_chunk_remaining[uid]`, initialized to `chunk_size` on SUBMIT or CONTINUE.

The value 24 represents a balance: small enough that decode users see minimal added latency from prefill (at most 24 ticks, approximately 1ms at $\tau = 44\mu s$), large enough to amortize the overhead of queue rotation across multiple tokens.

## Prefill Path: Step by Step

The prefill path executes when `decode_staging.try_pop()` returns false (no decode work) and `prefill_queue.try_front(pfuid)` returns true (at least one user is waiting for prefill).

### Step 1: Cancel Check

```cpp
if (cancel_pending.is_set(pfuid)) {
    prefill_queue.pop_front();
    maybe_finalize_cleanup(pfuid);
    continue;
}
```

Cancelled users are removed from the prefill queue immediately. No tokens are injected.

### Step 2: Compute Position State

```cpp
uint32_t prompt_len = prompt_table.get_length(pfuid);
uint32_t device_pos = user_table.prefill_pos[pfuid];
uint32_t prompt_idx = device_pos - user_table.prefill_start_pos[pfuid];
uint32_t chunk_rem = user_table.prefill_chunk_remaining[pfuid];
```

Four values govern the injection decision:

- `prompt_len`: total number of tokens in the prompt (from `PromptTable`).
- `device_pos`: the next KV position to write on the device. Monotonically increasing across turns.
- `prompt_idx`: the index into the `PromptTable` for the current token. Computed as `device_pos - prefill_start_pos`. This decoupling allows the `PromptTable` to always store prompts starting at index 0, regardless of the device-side position (see [Chapter 2 -- UserTable](../ch2_threading_and_data_structures/user_table.md#prefill_start_pos-decoupling-prompt-index-from-device-position) for the multi-turn arithmetic).
- `chunk_rem`: tokens remaining in the current chunk before rotation.

### Step 3: Chunk Exhaustion and Rotation

```cpp
if (chunk_rem == 0) {
    user_table.prefill_chunk_remaining[pfuid] = params.chunk_size;
    prefill_queue.rotate();
    continue;
}
```

**Invariant enforcement:** when the chunk counter reaches zero, the Writer resets it to `chunk_size` and calls `prefill_queue.rotate()`, which moves the current user from the front to the back of the queue. The `continue` returns to the top of the loop, where the Writer will check decode staging first, then serve the next prefill user (now at the front of the queue).

The `rotate()` operation is a no-op if only one user is in the prefill queue (`queue.size() <= 1`). In that case, the single user's chunk counter is reset and injection continues on the next iteration.

This rotation has zero injection cost -- no token is injected on the rotation tick. The tick is "spent" on bookkeeping. In practice, this is negligible: one rotation tick per 24 injections (default chunk size).

### Step 4: Prompt Exhaustion and Context Exhaustion

```cpp
if (prompt_idx >= prompt_len || device_pos >= params.max_seq_len) {
    prefill_queue.pop_front();
    prompt_table.clear(pfuid);

    bool ctx_exhausted = (device_pos >= params.max_seq_len);
    OutputMessage msg{
        .slot_id = pfuid,
        .token_id = EMPTY_TOKEN,
        .is_complete = true,
        .ctx_exhausted = ctx_exhausted,
        .tokens_generated = 0,
        .generation = decode_staging.generation[pfuid].load(
            std::memory_order_relaxed),
    };
    user_table.state[pfuid].store(UserState::COMPLETE, std::memory_order_release);

    while (!output_queue.try_push(msg)) {
        _mm_pause();
    }
    continue;
}
```

Two conditions cause prefill to terminate without entering decode:

1. **Prompt index overrun** (`prompt_idx >= prompt_len`): This is a defensive check -- in normal operation the last-token path (Step 5) handles prompt completion. This catches edge cases such as zero-length prompts.

2. **Context exhaustion** (`device_pos >= max_seq_len`): The user has consumed the entire KV cache during prefill. This can happen with very long prompts or multi-turn conversations where prior turns consumed most of the context window. The user transitions directly to COMPLETE with `ctx_exhausted = true`. No output tokens are produced -- the `OutputMessage` carries `EMPTY_TOKEN` and `tokens_generated = 0`.

In both cases: the user is popped from the `PrefillQueue`, the prompt is cleared from `PromptTable`, and a completion `OutputMessage` is published. Note that this path does NOT call `stage_eos_writeback()` or `set_position_on_complete()` -- the user never entered decode, so there is no terminator to write back and no decode result to derive a position from.

### Step 5: Token Injection

```cpp
bool is_last = (prompt_idx == prompt_len - 1);

user_table.prefill_pos[pfuid] = device_pos + 1;
user_table.current_position[pfuid].store(device_pos + 1, std::memory_order_release);
user_table.prefill_chunk_remaining[pfuid] = chunk_rem - 1;
```

The Writer advances the device position, updates the atomic `current_position`, and decrements the chunk counter. These writes happen before the pipeline injection, establishing the position state for the next iteration.

### Step 5a: In-Flight Counters

```cpp
if (!is_last) {
    user_table.prefill_in_flight[pfuid].fetch_add(1, std::memory_order_release);
}
user_table.in_flight_count[pfuid].fetch_add(1, std::memory_order_release);
```

**Invariant: non-last prefill tokens increment `prefill_in_flight` so the Reader can silently discard their results.**

Two counters are incremented:

- **`in_flight_count`** is always incremented. It tracks every token inside the pipeline for the slot, regardless of type. The Reader decrements it when the result arrives.
- **`prefill_in_flight`** is incremented only for non-last prefill tokens. These tokens populate KV cache but do not produce sampled output. When the Reader sees a result with `prefill_in_flight > 0`, it decrements the counter and discards the result. The last token does NOT increment `prefill_in_flight` because its result carries the first sampled output token.

### Step 5b: Injection with Prefill Token Hint

```cpp
pipeline_->inject(InjectDescriptor{
    .slot_id = pfuid,
    .token_id = prompt_table.get_token(pfuid, prompt_idx),
    .prefill_token_id = is_last ? EMPTY_TOKEN
                                : prompt_table.get_token(pfuid, prompt_idx + 1),
    .position = device_pos,
    .token_type = TokenType::BASE,
    .spec_flag = false,
    .temperature = sp.temperature,
    .top_p = sp.top_p,
    .top_k = sp.top_k,
});
```

The `prefill_token_id` field carries a hint to the device: for non-last tokens, it is the next prompt token (at `prompt_idx + 1`), allowing the device to optimistically set up the next KV write. For the last token, it is `EMPTY_TOKEN`, signaling that no prefill hint is available -- the device should sample the next token instead.

### Step 5c: Last-Token Transition

```cpp
if (is_last) {
    prefill_queue.pop_front();
    user_table.state[pfuid].store(UserState::DECODE, std::memory_order_release);
    prompt_table.clear(pfuid);
}
```

**Invariant: on the last prefill token, the user transitions to DECODE state and is removed from the `PrefillQueue`.**

The last prompt token is injected with `prefill_token_id = EMPTY_TOKEN`, which tells the device to run the sampling kernel. The pipeline result for this token will carry the first sampled output token, which the Reader will process as a regular decode result. The `PromptTable` entry is cleared because the prompt data is no longer needed.

The state transition to DECODE uses release ordering, ensuring the Reader sees all preceding field writes (including the final `current_position` update) when it loads the state.

## Position Management

Prefill position tracking uses three fields to decouple the device-side position from the prompt-table index:

| Field | Set by | Updated by | Purpose |
|-------|--------|------------|---------|
| `prefill_start_pos` | API (SUBMIT/CONTINUE) | Never (immutable during prefill) | Device position at which this turn's prefill starts. For turn 1: 0. For CONTINUE after position $P$: $P$. |
| `prefill_pos` | API (init = `prefill_start_pos`) | Writer (increment on each inject) | Next device position to inject at. |
| `current_position` | API (init = `prefill_start_pos`) | Writer (release store on each inject), Reader (on completion) | Atomic position visible to all threads. |

The prompt-table index is always computed as:

$$\text{prompt\_idx} = \text{prefill\_pos} - \text{prefill\_start\_pos}$$

This ensures `PromptTable` row 0 always corresponds to the first token of the current prompt, regardless of the user's position in the context window. For a detailed treatment of this decoupling, see [Chapter 2 -- UserTable, prefill_start_pos](../ch2_threading_and_data_structures/user_table.md#prefill_start_pos-decoupling-prompt-index-from-device-position).

### Multi-Turn Position Example

```
Turn 1: prompt = "Hello world" (2 tokens)
  SUBMIT: prefill_start_pos = 0, prefill_pos = 0
  Inject token[0] at device_pos=0, prompt_idx=0
  Inject token[1] at device_pos=1, prompt_idx=1 (last -> DECODE)
  Decode generates 50 tokens at positions 2..51
  Completion: current_position = 52

Turn 2: CONTINUE with prompt = "How are you" (3 tokens)
  handle_local_continue: prefill_start_pos = 52, prefill_pos = 52
  Inject token[0] at device_pos=52, prompt_idx=0
  Inject token[1] at device_pos=53, prompt_idx=1
  Inject token[2] at device_pos=54, prompt_idx=2 (last -> DECODE)
  Decode resumes at position 55
```

## Interleaving Example

Consider three users: A in decode (actively generating), B in prefill (20-token prompt), C in prefill (8-token prompt). Chunk size is 4. The notation `D(A)` means a decode token for user A, `P(B,i)` means prefill token $i$ for user B.

```
PrefillQueue: [B, C]          DecodeStaging: [A]

Tick 1:  D(A)     -- decode has priority
Tick 2:  P(B,0)   -- no decode work, serve front of prefill queue
Tick 3:  P(B,1)
Tick 4:  P(B,2)
Tick 5:  P(B,3)   -- B's chunk exhausted (chunk_rem=0)
                   -- rotate: PrefillQueue becomes [C, B]
                   -- B's chunk_remaining reset to 4
Tick 6:  P(C,0)   -- now serving C
Tick 7:  D(A)     -- decode result arrives in staging, preempts prefill
Tick 8:  P(C,1)   -- back to prefill
Tick 9:  P(C,2)
Tick 10: P(C,3)   -- C's chunk exhausted, rotate: [B, C]
Tick 11: P(B,4)   -- back to B
Tick 12: P(B,5)
Tick 13: D(A)     -- another decode result, preempts
Tick 14: P(B,6)   -- resume B where we left off
Tick 15: P(B,7)   -- B's chunk exhausted again, rotate: [C, B]
Tick 16: P(C,4)
Tick 17: P(C,5)
Tick 18: P(C,6)
Tick 19: P(C,7)   -- C's last token (prompt_idx == 7 == prompt_len-1)
                   -- C transitions to DECODE, popped from queue
                   -- PrefillQueue becomes [B]
Tick 20: D(A)
Tick 21: P(B,8)   -- B continues, sole prefill user
...
```

Key observations:

1. Decode user A is never starved -- its tokens always preempt prefill.
2. Neither B nor C monopolizes the Writer for more than `chunk_size = 4` consecutive ticks.
3. C completes prefill while B still has tokens remaining; C enters decode and its tokens begin interleaving with A's in the pipeline.
4. Chunk state (specifically `prefill_chunk_remaining`) persists across decode preemptions. B's counter resumes where it was when the decode preemption occurred.

## `prefill_in_flight` and the Reader's Discard Logic

Non-last prefill tokens increment `prefill_in_flight`. When the Reader receives a pipeline result, it checks this counter:

```cpp
if (user_table.prefill_in_flight[uid].load(std::memory_order_acquire) > 0) {
    user_table.prefill_in_flight[uid].fetch_sub(1, std::memory_order_relaxed);
    continue;
}
```

If positive, the result is a KV-population result with no meaningful sampled output. The Reader decrements the counter and discards the result. The `ReaderClaim` destructor still runs (releasing `pending_count` and calling `maybe_finalize_cleanup`), so the busy-count protocol is maintained.

The last prefill token does not increment `prefill_in_flight`, so its pipeline result passes through the discard check and enters the decode result path -- producing the user's first sampled output token.

---

**Next:** [`decode_loopback.md`](./decode_loopback.md)
