# UserTable

`UserTable` is the per-user register file -- the central mutable state for every active slot in the scheduler. It is a collection of parallel vectors, all indexed by `slot_id` (also called `uid`) and sized to `max_users` at construction. The struct lives in `src/common/user_table.hpp`. The parallel-vector layout maps directly to the register file or SRAM bank layout expected by a hardware deployment target.

## Structure Overview

```cpp
struct UserTable {
    uint32_t max_users_;

    // Atomic fields (cross-thread communication)
    std::vector<std::atomic<UserState>> state;
    std::vector<std::atomic<uint32_t>>  current_position;
    std::vector<std::atomic<uint32_t>>  in_flight_count;
    std::vector<std::atomic<uint32_t>>  prefill_in_flight;
    std::vector<std::atomic<uint32_t>>  post_complete_in_flight;
    std::vector<std::atomic<bool>>      in_thinking_phase;

    // Plain fields (single-writer-per-phase discipline)
    std::vector<uint32_t> max_new_tokens;
    std::vector<uint32_t> tokens_generated;
    std::vector<uint32_t> prefill_pos;
    std::vector<uint32_t> prefill_start_pos;
    std::vector<uint32_t> prefill_chunk_remaining;
    std::vector<bool>     spec_decode_enabled;
    std::vector<bool>     ignore_eos;
    std::vector<float>    temperature;
    std::vector<float>    top_p;
    std::vector<int32_t>  top_k;
    std::vector<float>    relaxed_acceptance_threshold;
    std::vector<uint32_t> spec_accepts;
    std::vector<uint32_t> spec_rejects;
};
```

## Full Field Catalog

### Atomic Fields

These fields are accessed by multiple threads concurrently and use `std::atomic` for synchronization.

| Field | Type | Writer | Reader | API | Memory Ordering | Purpose |
|-------|------|--------|--------|-----|----------------|---------|
| `state` | `atomic<UserState>` | write (store release), CAS in cleanup (acq_rel) | read/CAS (cleanup, acq_rel) | write (store release) | release/acquire; acq_rel CAS in cleanup | FSM state: INACTIVE/PREFILL/DECODE/COMPLETE |
| `current_position` | `atomic<uint32_t>` | write during prefill (store release) | write on completion (store release) | init (store relaxed), store on disaggregated continue (release), read on local continue (relaxed) | release/acquire | KV cache write head, monotonically increasing across turns |
| `in_flight_count` | `atomic<uint32_t>` | increment (fetch_add release) | decrement (fetch_sub relaxed) | init (store relaxed), read in cleanup (load acquire) | release on increment, acquire on cleanup check | Tokens currently inside the pipeline for this user |
| `prefill_in_flight` | `atomic<uint32_t>` | increment (fetch_add release) | decrement (fetch_sub relaxed), zero on cancel | -- | release/acquire | Non-last prefill tokens in pipeline (reader discards these results) |
| `post_complete_in_flight` | `atomic<uint32_t>` | -- | increment (fetch_add release), decrement (fetch_sub relaxed), zero on cancel | -- | release/acquire | EOS write-back injects whose BASE results must be discarded |
| `in_thinking_phase` | `atomic<bool>` | read (load acquire) | write (store release) | init (store release) | release/acquire | Whether user is inside thinking-phase brackets; controls sampling params |

**`in_flight_count` ordering detail:** The Writer increments before injecting (not after) to prevent a race where the Reader processes the result before the Writer records the in-flight token. The Reader uses `relaxed` for the decrement because the `ReaderClaim` (see [Decode Staging](./decode_staging.md)) provides the necessary ordering boundary. A slot cannot be recycled until this reaches zero; `maybe_finalize_cleanup()` gates on `in_flight_count == 0`.

**`prefill_in_flight` detail:** Counts non-last prefill tokens in the pipeline whose results must be silently discarded. The Writer increments for each non-last prefill inject; the Reader decrements when discarding a prefill result.

**`post_complete_in_flight` detail:** Counts EOS write-back injects. After generation completes, the Reader stages a write-back of the completing token to KV so the next turn sees it as context. The resulting BASE pipeline output must not re-enter the decode path.

**`in_thinking_phase` detail:** Toggles between strict and relaxed speculative acceptance modes. The Reader stores `true` (release) on seeing `think_open_token_id`, `false` (release) on `think_close_token_id`. The Writer loads (acquire) in `device_sampling_params()` to decide whether to force `top_k=1` or pass through the user's sampling params.

### Plain Fields

These fields are non-atomic. Safety relies on **phase-based exclusion**: the API thread writes them during SUBMIT/CONTINUE (before the slot enters the prefill queue), and afterwards only one hot-path thread touches each field.

- The API thread writes them before the user enters PREFILL. The Writer does not read them until the user appears in the prefill queue. The Reader does not process results until tokens are in-flight.
- The Writer advances `prefill_pos` and `prefill_chunk_remaining` during PREFILL state. The Reader does not touch these fields.
- The Reader increments `tokens_generated` and `spec_accepts/rejects` during DECODE state. The API thread does not read these until after completion.

This phase-based ownership avoids the need for atomics on the hot path for these fields.

| Field | Type | Written By | Read By | Purpose |
|-------|------|-----------|---------|---------|
| `max_new_tokens` | `uint32_t` | API (SUBMIT, CONTINUE) | Reader (completion check) | Maximum tokens to generate this turn |
| `tokens_generated` | `uint32_t` | API (init to 0), Reader (increment on each output) | Reader (completion check) | Count of tokens emitted this turn |
| `prefill_pos` | `uint32_t` | API (init), Writer (advance each prefill inject) | Writer only | Next prompt index to inject (device-side position) |
| `prefill_start_pos` | `uint32_t` | API (init on SUBMIT/CONTINUE) | Writer (compute prompt_idx) | Device position at which this turn's prefill started |
| `prefill_chunk_remaining` | `uint32_t` | API (init), Writer (decrement, reload on rotation) | Writer only | Tokens remaining in current chunk before round-robin rotation |
| `spec_decode_enabled` | `bool` | API (set on SUBMIT/CONTINUE) | Writer (decide spec inject), Reader (choose decode path) | Whether spec decode is active for this user |
| `ignore_eos` | `bool` | API (set on SUBMIT/CONTINUE) | Reader (completion check) | If true, EOS token does not trigger completion |
| `temperature` | `float` | API (set on SUBMIT/CONTINUE) | Writer (device_sampling_params), Reader (device_sampling_params via check_acceptance) | Sampling temperature |
| `top_p` | `float` | API (set on SUBMIT/CONTINUE) | Writer (device_sampling_params), Reader (relaxed acceptance) | Top-P sampling parameter |
| `top_k` | `int32_t` | API (set on SUBMIT/CONTINUE) | Writer (device_sampling_params), Reader (relaxed acceptance scan bound) | Top-K sampling parameter |
| `relaxed_acceptance_threshold` | `float` | API (set on SUBMIT/CONTINUE) | Reader (check_acceptance during thinking phase) | Maximum probability delta for relaxed spec acceptance |
| `spec_accepts` | `uint32_t` | Reader (increment on ACCEPT) | Diagnostic only | Counter of successful speculations |
| `spec_rejects` | `uint32_t` | Reader (increment on REJECT) | Diagnostic only | Counter of failed speculations |

### prefill_start_pos: Decoupling Prompt Index from Device Position

Prompts are always stored in `PromptTable` starting at index 0, regardless of the user's current KV cache position. The `prefill_start_pos` field bridges the gap. The Writer computes the prompt table index as:

```
prompt_idx = prefill_pos - prefill_start_pos
```

This decoupling is essential for multi-turn continuation. On a CONTINUE request, the user's KV cache already has tokens at positions 0 through `current_position - 1`. The new prompt tokens are stored in the PromptTable starting at index 0, but `prefill_start_pos` is set to `current_position` so the Writer injects them at the correct device positions:

```
Turn 1: prompt = [A, B, C, D, E]
  prefill_start_pos = 0
  device positions: 0, 1, 2, 3, 4
  -> generates tokens, completes at current_position = 57

Turn 2: CONTINUE with prompt = [F, G, H]
  PromptTable row stores: [F, G, H] at indices 0, 1, 2
  prefill_start_pos = 57
  device positions: 57, 58, 59
  prompt_idx for device_pos 58 = 58 - 57 = 1 -> PromptTable[1] = G
```

Without this decoupling, the PromptTable would need to be offset-aware or the Writer would need to maintain a separate mapping. The current design keeps PromptTable simple and shifts the position arithmetic to two scalar fields in UserTable.

## UserState Finite State Machine

```
                +-----------+
                | INACTIVE  |  <-- initial state, slot free
                +-----+-----+
                      |
                      | ALLOCATE (api) + SUBMIT (api)
                      v
                +-----+-----+
         +----->|  PREFILL   |  <-- prompt tokens being injected
         |      +-----+-----+
         |            |
         |            | last prefill token processed (Writer)
         |            | OR prompt exhausted / ctx_exhausted
         |            v
         |      +-----+-----+
         |      |  DECODE    |  <-- generating output tokens
         |      +-----+-----+
         |            |
         |            | EOS / max_new_tokens / ctx_exhausted
         |            v
         |      +-----+-----+
         |      | COMPLETE   |  <-- generation done, KV cache RETAINED
         |      +-----+-----+
         |            |
         |      +-----+-----+
         |      |           |
         |  CONTINUE    CANCEL
         |  (api)       (api -> deferred cleanup)
         |      |           |
         +------+           v
                      +-----------+
                      | INACTIVE  |  <-- after in_flight==0, pending==0
                      +-----------+
```

State transitions and their writers:

| From | To | Trigger | Thread |
|---|---|---|---|
| INACTIVE | PREFILL | ALLOCATE + SUBMIT request | API handler |
| PREFILL | DECODE | Last prefill token injected | Writer |
| PREFILL | COMPLETE | Context exhausted during prefill (defensive) | Writer |
| DECODE | COMPLETE | EOS, max_new_tokens, or context exhausted | Reader (via emit_token) |
| COMPLETE | PREFILL | CONTINUE request (local multi-turn) | API handler |
| COMPLETE | DECODE | CONTINUE request (disaggregated) | API handler |
| Any non-INACTIVE | INACTIVE | CANCEL cleanup finalized (CAS in `maybe_finalize_cleanup`) | Any thread that wins CAS |
| INACTIVE | INACTIVE | CANCEL of never-submitted slot (API direct path) | API handler |

Note that CANCEL does not have its own state. Cancellation is tracked via the separate `CancelBitmap`. The state transitions to INACTIVE only after all in-flight tokens have drained and the cleanup CAS succeeds.

### Atomicity of State Transitions

Most transitions use `store` with `memory_order_release` to publish the new state. The critical exception is the cancel cleanup path, which uses `compare_exchange_strong` with `memory_order_acq_rel`:

```cpp
UserState expected = state[uid].load(memory_order_acquire);
if (expected == INACTIVE) return;  // already cleaned
if (!state[uid].compare_exchange_strong(
        expected, INACTIVE, memory_order_acq_rel, memory_order_relaxed)) {
    return;  // another thread won the cleanup race
}
```

This CAS ensures exactly one thread performs the cleanup sequence (reset_kv, free slot). See [Lock-Free Design](./lock_free_design.md) for the full cancel/recycle race analysis.

## reset() Semantics

```cpp
void reset(uint32_t uid) {
    state[uid].store(UserState::INACTIVE, std::memory_order_relaxed);
    current_position[uid].store(0, std::memory_order_relaxed);
    max_new_tokens[uid] = 0;
    tokens_generated[uid] = 0;
    in_flight_count[uid].store(0, std::memory_order_relaxed);
    prefill_in_flight[uid].store(0, std::memory_order_relaxed);
    post_complete_in_flight[uid].store(0, std::memory_order_relaxed);
    prefill_pos[uid] = 0;
    prefill_start_pos[uid] = 0;
    prefill_chunk_remaining[uid] = 0;
    spec_decode_enabled[uid] = false;
    ignore_eos[uid] = false;
    temperature[uid] = 1.0f;
    top_p[uid] = 1.0f;
    top_k[uid] = -1;
    relaxed_acceptance_threshold[uid] = 0.0f;
    in_thinking_phase[uid].store(false, std::memory_order_relaxed);
}
```

`reset()` is called by the API thread during ALLOCATE processing, immediately after `free_ids.allocate()` returns a slot. All stores use `relaxed` ordering because the slot is not yet visible to any other thread -- it has just been removed from the free pool and has not been submitted to any queue. The subsequent SUBMIT request will store `UserState::PREFILL` with release semantics, establishing the happens-before boundary that makes all these relaxed stores visible to the Writer and Reader threads.

Note that `reset()` does **not** clear `spec_accepts` or `spec_rejects`. These are cumulative diagnostic counters that persist across the slot's lifetime.

---

**Previous:** [FreeIdPool](./free_id_pool.md) | **Next:** [PromptTable and CancelBitmap](./prompt_table_and_cancel_bitmap.md)
