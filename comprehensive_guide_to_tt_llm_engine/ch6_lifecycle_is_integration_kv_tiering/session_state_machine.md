# Session State Machine

[< Chapter Overview](index.md) | [Multi-Turn Continue >](multi_turn_continue.md)

## The Four States

Every user slot in the decode scheduler is in exactly one of four states, represented by the `UserState` enum:

```cpp
enum class UserState : uint8_t {
    INACTIVE = 0,   // Slot is either unallocated or allocated-but-idle
    PREFILL  = 1,   // Prompt tokens are being chunked through the pipeline
    DECODE   = 2,   // Autoregressive token generation in progress
    COMPLETE = 3    // Generation finished; awaiting IS acknowledgment or CONTINUE
};
```

The `INACTIVE` state has two sub-meanings distinguished by whether the slot is in the free pool:
- **INACTIVE (unallocated):** The slot ID is in `FreeIdPool`. No user owns it.
- **INACTIVE (allocated):** A user owns the slot (it was removed from the free pool by `ALLOCATE`) but has not yet submitted a prompt.

This distinction matters because `CANCEL` on an `INACTIVE`-but-allocated slot triggers an immediate cleanup path rather than the deferred protocol.

## Complete State Transition Table

| Current State | Trigger | New State | Thread | Preconditions |
|---------------|---------|-----------|--------|---------------|
| INACTIVE (unallocated) | `ALLOCATE` request | INACTIVE (allocated) | Scheduler | Free slot available in `FreeIdPool` |
| INACTIVE (allocated) | `SUBMIT` request | PREFILL | Scheduler | Slot is allocated, prompt tokens provided |
| PREFILL | Writer completes final chunk | DECODE | Writer | All prompt chunks processed (see Ch3) |
| DECODE | EOS token emitted | COMPLETE | Reader | `token_id == eos_token_id` and `!ignore_eos` |
| DECODE | `max_new_tokens` reached | COMPLETE | Reader | `tokens_generated >= max_new_tokens` |
| DECODE | Context exhausted | COMPLETE | Reader | `ctx_pos >= max_seq_len` |
| PREFILL | Context exhausted (pre-check) | COMPLETE | Writer | `device_pos >= max_seq_len` (defensive) |
| COMPLETE | `CONTINUE` (local) | PREFILL | Scheduler | Slot still allocated, new tokens + params |
| COMPLETE | `CONTINUE` (disaggregated) | DECODE | Scheduler | Disaggregated mode, migrated token available |
| COMPLETE | No continue; IS frees | INACTIVE (unallocated) | Scheduler | `CANCEL` or explicit teardown |
| Any (active) | `CANCEL` request | *deferred* -> INACTIVE | Multi-phase | See [Deferred Cancellation](deferred_cancellation.md) |
| INACTIVE (allocated) | `CANCEL` request | INACTIVE (unallocated) | Scheduler | No in-flight, no pending -- immediate path |

## Request Types and Their Semantics

The IS communicates with the scheduler through `ISRequest` messages, each carrying one of four `RequestType` values:

```cpp
enum class RequestType : uint8_t {
    ALLOCATE = 1,   // Reserve a slot
    SUBMIT   = 2,   // Submit prompt tokens for a new generation
    CONTINUE = 3,   // Begin a new turn, reusing KV cache
    CANCEL   = 4    // Request slot teardown
};
```

### ALLOCATE

**Purpose:** Reserve a user slot from `FreeIdPool`.

```cpp
case RequestType::ALLOCATE: {
    uint32_t uid = free_ids.allocate();
    if (uid != INVALID_SLOT) {
        user_table.reset(uid);
        cancel_pending.clear(uid);
        spec_state.reset(uid);
    }
    SchedulerResponse resp{
        .request_id = req.request_id,
        .slot_id = uid,
        .error_code = (uid == INVALID_SLOT) ? 1 : 0
    };
    while (!response_queue.try_push(resp)) { _mm_pause(); }
    break;
}
```

Key observations:
1. `free_ids.allocate()` returns `INVALID_SLOT` if no slots are available -- the scheduler never blocks waiting for capacity.
2. On success, the slot's `UserTable` entry, `CancelBitmap` flag, and `SpecDecodeState` are all reset to a clean initial state.
3. The response is pushed synchronously (spin-waiting if the response queue is full), so the IS receives confirmation before any further operations on that slot.
4. `error_code = 1` specifically signals no free slots. The IS must handle this (e.g., by queuing the request or rejecting the client).

**Invariant:** After `ALLOCATE`, `in_flight_count[uid] == 0`, `pending_count[uid] == 0`, and `cancel_pending` is clear. This is a clean slate for the new session.

### SUBMIT

**Purpose:** Begin a new generation by submitting prompt tokens.

```cpp
case RequestType::SUBMIT: {
    uint32_t uid = req.slot_id;
    prompt_table.store(uid, req.tokens.data(), ...);
    user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
    user_table.current_position[uid].store(0, std::memory_order_relaxed);
    user_table.prefill_pos[uid] = 0;
    user_table.prefill_start_pos[uid] = 0;
    user_table.max_new_tokens[uid] = req.gen.max_new_tokens;
    user_table.tokens_generated[uid] = 0;
    user_table.in_flight_count[uid].store(0, std::memory_order_relaxed);
    user_table.prefill_chunk_remaining[uid] = params.chunk_size;
    user_table.spec_decode_enabled[uid] = req.gen.spec_decode;
    // ... all gen params stored ...
    spec_state.reset(uid);
    prefill_queue.push(uid);
    break;
}
```

Key observations:
1. The prompt is copied into `prompt_table` first, before any state transitions.
2. The state store uses `memory_order_release` -- this ensures all the field writes above are visible to the Writer thread before it observes the `PREFILL` state.
3. `current_position` starts at 0 (fresh generation, no KV reuse). Compare this to `handle_local_continue` where `current_position` preserves prior context.
4. `prefill_chunk_remaining` is initialized to `params.chunk_size`, establishing the chunked-prefill window (see Ch3 for the chunking protocol).
5. The user is pushed onto `prefill_queue`, where the Writer will pick it up in its next priority scan.

### CONTINUE

Handled by `handle_local_continue` or `handle_disaggregated_continue` depending on mode. See [Multi-Turn Continue](multi_turn_continue.md).

### CANCEL

Initiates the deferred cancellation protocol. See [Deferred Cancellation](deferred_cancellation.md).

## The Three Interface Structures

The IS and scheduler communicate through three data structures, each carried on its own queue.

### ISRequest (IS -> Scheduler)

```cpp
struct ISRequest {
    RequestType type = RequestType::ALLOCATE;
    uint32_t    request_id = 0;
    uint32_t    slot_id = INVALID_SLOT;
    std::vector<uint32_t> tokens;
    GenerationParams gen;
};
```

The `request_id` is opaque to the scheduler -- it is echoed back in `SchedulerResponse` so the IS can correlate responses to requests. The `slot_id` is meaningless for `ALLOCATE` (the scheduler assigns one) but required for all other request types.

### SchedulerResponse (Scheduler -> IS)

```cpp
struct SchedulerResponse {
    uint32_t request_id = 0;
    uint32_t slot_id = INVALID_SLOT;
    int32_t  error_code = 0;
};
```

Responses are generated for:
- `ALLOCATE`: confirms slot assignment or reports `error_code = 1` (no free slots).
- `CANCEL`: pushed as a cancel acknowledgment after cleanup completes (via `push_cancel_ack` to the `response_queue`). See [Deferred Cancellation](deferred_cancellation.md).

`SUBMIT` and `CONTINUE` do not produce a `SchedulerResponse` -- their effects are observable through `OutputMessage` tokens.

### OutputMessage (Scheduler -> IS)

```cpp
struct OutputMessage {
    uint32_t slot_id = INVALID_SLOT;
    uint32_t token_id = EMPTY_TOKEN;
    bool     is_complete = false;
    bool     ctx_exhausted = false;
    uint32_t tokens_generated = 0;
    uint32_t generation = 0;
};
```

The `generation` counter is critical for correctness. Each time the scheduler advances a slot's generation (via `decode_staging.advance_generation`), any in-flight tokens from the prior generation become stale. The IS uses this counter to discard late-arriving tokens from a cancelled or superseded generation. See [IS Integration Patterns](is_integration_patterns.md) for how this is consumed.

## The Generation Counter

The generation counter solves a fundamental problem in pipelined systems: when a cancellation or continuation occurs, there may be tokens from the previous generation still in flight through the pipeline. Without a generation counter, the IS could not distinguish "token 42 from the cancelled request" from "token 42 from the new request."

The lifecycle of the generation counter for a single slot:

1. **ALLOCATE**: Counter is implicitly 0 (via `user_table.reset`).
2. **SUBMIT**: Counter remains at 0 for the first generation.
3. **CANCEL**: `decode_staging.advance_generation(uid)` increments the counter.
4. **Reader emits token**: The `OutputMessage.generation` field reflects the current counter value at emission time.

The IS-side contract: if the IS receives an `OutputMessage` whose `generation` is less than the last `generation` it processed for that slot, the message is stale and must be discarded.

## Worked Example: Slot 7 Lifecycle

To make the above concrete, consider a single request flowing through the system:

| Parameter | Value |
|-----------|-------|
| Slot ID | 7 |
| Request IDs | 1001 (ALLOCATE), 1002 (SUBMIT) |
| Prompt tokens | 128 (`tok_0` through `tok_127`) |
| `max_new_tokens` | 64 |
| `temperature` | 0.8 |
| `max_seq_len` | 2048 |

### Complete Timeline

| Time | Thread | Action | State After |
|------|--------|--------|-------------|
| T+0.000ms | IS | `push_request(ALLOCATE, 1001)` | request_queue: 1 entry |
| T+0.002ms | Scheduler | Pop ALLOCATE, allocate slot 7, reset state | free_ids: slot 7 removed |
| T+0.002ms | Scheduler | `push_response({1001, 7, 0})` | response_queue: 1 entry |
| T+0.004ms | IS | `try_pop_response()` -> slot 7, error_code=0 | IS maps session to slot 7 |
| T+0.005ms | IS | `push_request(SUBMIT, 1002, slot=7, 128 toks)` | request_queue: 1 entry |
| T+0.008ms | Scheduler | Pop SUBMIT, store prompt, set PREFILL, push prefill_queue | state = PREFILL, pos = 0 |
| T+0.200ms | Writer | Process 128-token chunk, transition to DECODE | state = DECODE, pos = 128 |
| T+0.200ms | Reader | Decode step 1: token_id=4821 | pos = 129, tokens_generated = 1 |
| T+0.220ms | Reader | Decode step 2: token_id=1537 | pos = 130, tokens_generated = 2 |
| T+0.240ms | Reader | Decode step 3: token_id=892 | pos = 131, tokens_generated = 3 |
| ... | Reader | Steps 4--63 | pos incrementing |
| T+1.510ms | Reader | Decode step 64: EOS, set COMPLETE | state = COMPLETE, pos = 192 |
| T+1.512ms | IS | Drain output_queue, stream 64 tokens to client | output_queue: empty |

After this lifecycle, slot 7 is in `COMPLETE` state with its KV cache intact at position 192. The IS can send `CONTINUE` for multi-turn or `CANCEL` to free the slot.

## Queue Sizing and Public API

The three queues (request, response, output) and the full public API surface (`push_request`, `try_pop_response`, `try_pop_output`, read-only accessors) are covered in detail in [IS Integration Patterns](is_integration_patterns.md), including capacity rationale, backpressure behavior, and recommended polling patterns.

---

*[< Chapter Overview](index.md) | [Multi-Turn Continue >](multi_turn_continue.md)*
